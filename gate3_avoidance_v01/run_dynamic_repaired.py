from pathlib import Path
import ast
import csv
import json
import math
import subprocess
import sys
import xml.etree.ElementTree as ET

import mujoco

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SCENES = REPO / "gate3_obstacle_stop_v01"
OUT = ROOT / "step4_step5_parallel"
OUT.mkdir(exist_ok=True)

BASE_SCENE = SCENES / "step4_random_case_02.xml"
OLD_SCRIPT = ROOT / "test_step4_dynamic_parallel.py"
NEW_SCENE = SCENES / "step4_dynamic_repaired_scene.xml"
NEW_SCRIPT = ROOT / "test_step4_dynamic_repaired.py"

for p in (BASE_SCENE, OLD_SCRIPT):
    if not p.is_file():
        raise SystemExit(f"MISSING: {p}")

print("=== PHASE 1: REBUILD DORMANT OBSTACLE ===", flush=True)

base_model = mujoco.MjModel.from_xml_path(str(BASE_SCENE))
base_data = mujoco.MjData(base_model)
mujoco.mj_forward(base_model, base_data)

extra_id = mujoco.mj_name2id(
    base_model,
    mujoco.mjtObj.mjOBJ_GEOM,
    "extra_blocking_box"
)
if extra_id < 0:
    raise SystemExit("EXTRA_OBSTACLE_NOT_FOUND")

initial_pos = base_data.geom_xpos[extra_id].copy()
extra_size = base_model.geom_size[extra_id].copy()

tree = ET.parse(BASE_SCENE)
world = tree.getroot().find("worldbody")
if world is None:
    raise SystemExit("WORLDBODY_NOT_FOUND")

name = "new_dynamic_obstacle_gate3"

if any(e.get("name") == name for e in world.iter("geom")):
    raise SystemExit("DUPLICATE_DYNAMIC_OBSTACLE")

ET.SubElement(
    world,
    "geom",
    {
        "name": name,
        "type": "box",
        "pos": " ".join(f"{x:.9f}" for x in initial_pos),
        "size": "0.04 0.07 0.16",
        "rgba": "0 0 0 0",
        "contype": "0",
        "conaffinity": "0",
    }
)

tree.write(NEW_SCENE, encoding="unicode")

new_model = mujoco.MjModel.from_xml_path(str(NEW_SCENE))

print("Baseline model extent:", base_model.stat.extent)
print("Repaired model extent:", new_model.stat.extent)
print("Dormant position:", initial_pos)
print("Original extra size:", extra_size)

if (
    not math.isfinite(new_model.stat.extent)
    or new_model.stat.extent >
       base_model.stat.extent * 1.25
):
    raise SystemExit("SCENE_EXTENT_UNEXPECTED")

print("DORMANT_SCENE_BUILD: PASS")

print("\n=== PHASE 2: PREPARE CONTROLLER ===", flush=True)

code = OLD_SCRIPT.read_text()
code = code.replace(
    "step4_dynamic_parallel_scene.xml",
    NEW_SCENE.name
)

# All output names are changed to avoid reusing old results.
code = code.replace(
    "step4_dynamic_parallel_",
    "step4_dynamic_repaired_"
)

spawn_anchor = '''            mujoco.mj_forward(model, data)

            dynamic_spawned = True'''

if code.count(spawn_anchor) != 1:
    raise SystemExit("SPAWN_ANCHOR_INVALID")

code = code.replace(
    spawn_anchor,
    '''            # Make the parked obstacle visible and collidable
            # only when it appears during the walking stage.
            model.geom_rgba[dynamic_id] = [0.9, 0.2, 0.8, 1.0]
            model.geom_contype[dynamic_id] = 1
            model.geom_conaffinity[dynamic_id] = 1
            mujoco.mj_forward(model, data)

            dynamic_spawned = True''',
    1
)

# Add independently logged dynamic obstacle distance.
trace_anchor = '''        "red_clearance_m": red,'''
if code.count(trace_anchor) != 1:
    raise SystemExit("TRACE_ANCHOR_INVALID")

code = code.replace(
    trace_anchor,
    trace_anchor + '''
        "dynamic_clearance_m": float(
            xy_clearance(dynamic_id)
        ),
        "dynamic_visible": bool(dynamic_spawned),
        "obstacle_contact_count": sum(
            int(
                int(data.contact[j].geom1) in
                (extra_id, box_id, dynamic_id)
                or int(data.contact[j].geom2) in
                (extra_id, box_id, dynamic_id)
            )
            for j in range(data.ncon)
        ),''',
    1
)

if code.count("CLEARANCE = 0.521") != 1:
    raise SystemExit("RUNTIME_THRESHOLD_CHANGED")
if code.count("PLANNING_CLEARANCE = 0.551") != 1:
    raise SystemExit("PLANNING_THRESHOLD_CHANGED")

ast.parse(code)
compile(code, str(NEW_SCRIPT), "exec")
NEW_SCRIPT.write_text(code)

print("CONTROLLER_COMPILE: PASS")
print("Original controller: unchanged")
print("Thresholds: unchanged")

print("\n=== PHASE 3: RUN MUJOCO ===", flush=True)

prefix = "step4_dynamic_repaired_"
report_path = ROOT / (
    prefix + "live_autonomous_route_v3.json"
)
trace_path = ROOT / (
    prefix + "live_autonomous_route_v3.csv"
)
log_path = OUT / "dynamic_repaired.log"

for p in (report_path, trace_path):
    p.unlink(missing_ok=True)

try:
    with log_path.open("w") as log:
        proc = subprocess.run(
            [sys.executable, str(NEW_SCRIPT)],
            cwd=REPO / "dog_robot_v02",
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=420,
            check=False
        )
    rc = proc.returncode
except subprocess.TimeoutExpired:
    rc = 124

log = log_path.read_text(errors="replace")

for line in log.splitlines():
    if any(s in line for s in (
        "Stage 2 walk stop:",
        "NEW_BOX_DEPTH_STOP:",
        "GOAL_POLICY_",
        "DYNAMIC_OBSTACLE_APPEARED:",
        "Live turn 1:",
        "Live waypoint walk:",
        "GATE3_STEP3:",
        "Traceback",
        "Error:",
    )):
        print(line)

report = {}
if report_path.exists():
    report = json.loads(report_path.read_text())

rows = []
if trace_path.exists():
    with trace_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

def numeric_min(column, subset):
    values = []
    for row in subset:
        try:
            v = float(row[column])
            if math.isfinite(v):
                values.append(v)
        except (KeyError, ValueError, TypeError):
            pass
    return min(values) if values else None

def numeric_max(column, subset):
    values = []
    for row in subset:
        try:
            v = float(row[column])
            if math.isfinite(v):
                values.append(v)
        except (KeyError, ValueError, TypeError):
            pass
    return max(values) if values else None

after_spawn = [
    r for r in rows
    if r.get("dynamic_visible") == "True"
]

min_dynamic = numeric_min(
    "dynamic_clearance_m", after_spawn
)
min_est = numeric_min(
    "estimated_clearance_m", rows
)
min_height = numeric_min("height_m", rows)
max_tilt = numeric_max("tilt_deg", rows)

contacts = sum(
    int(float(r.get("obstacle_contact_count", 0)))
    for r in rows
)

guards = sum(
    bool(r.get("guard_reason", "").strip())
    for r in rows
)

appeared = "DYNAMIC_OBSTACLE_APPEARED:" in log
stop_reason = None
for line in log.splitlines():
    if "Live waypoint walk:" in line:
        stop_reason = line.split("Live waypoint walk:", 1)[1].strip()

distance_ok = (
    min_dynamic is not None
    and min_dynamic >= 0.521
    and min_est is not None
    and min_est >= 0.521
)

posture_ok = (
    min_height is not None
    and min_height >= 0.16
    and max_tilt is not None
    and max_tilt <= 12.0
)

verified_stop = (
    rc == 0
    and appeared
    and stop_reason in (
        "DEPTH_OBSTACLE_STOP",
        "CLEARANCE_STOP",
    )
    and distance_ok
    and posture_ok
    and guards == 0
    and contacts == 0
)

summary = {
    "exit_code": rc,
    "stage2_depth_stop": (
        "reason=DEPTH_OBSTACLE_STOP" in log
    ),
    "dynamic_appeared": appeared,
    "waypoint_walk_stop_reason": stop_reason,
    "trace_rows": len(rows),
    "post_spawn_rows": len(after_spawn),
    "minimum_dynamic_clearance_m": min_dynamic,
    "minimum_estimated_clearance_m": min_est,
    "minimum_height_m": min_height,
    "maximum_tilt_deg": max_tilt,
    "contact_samples": contacts,
    "guard_samples": guards,
    "dynamic_stop_pass": verified_stop,
    "reobservation_replan_resume_pass": False,
    "full_body_independent_audit_pass": False,
    "reported_stage3_status": report.get("status"),
}

summary_path = OUT / "dynamic_repaired_summary.json"
summary_path.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False)
)

print("\n=== DYNAMIC + SAFETY SUMMARY ===")
for key, value in summary.items():
    print(f"{key}: {value}")

print("Summary:", summary_path)
print("Full log:", log_path)
print(
    "STEP4_DYNAMIC_STOP:",
    "PASS" if verified_stop else "NOT_YET_PASS"
)
print("STEP4_REPLAN_RESUME: NOT_YET")
print("STEP5_FULL_BODY_AUDIT: NOT_YET")
print("GATE3_COMPLETE: NO")
