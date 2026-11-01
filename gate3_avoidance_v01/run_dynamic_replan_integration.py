from pathlib import Path
import ast
import csv
import json
import math
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
OUT = ROOT / "step4_step5_parallel"
OUT.mkdir(exist_ok=True)

SOURCE = ROOT / "test_step4_dynamic_repaired.py"
PLANNER = ROOT / "test_multi_obstacle_replanner.py"
TARGET = ROOT / "test_step4_dynamic_replan_integration.py"
PREFIX = "step4_dynamic_replan_integration_"

for p in (SOURCE, PLANNER):
    if not p.is_file():
        raise SystemExit(f"MISSING_SOURCE: {p}")

code = SOURCE.read_text()
ast.parse(code)

# Separate all result filenames from previous experiments.
old = "step4_dynamic_repaired_"
if old not in code:
    raise SystemExit("OUTPUT_PREFIX_NOT_FOUND")
code = code.replace(old, PREFIX)

# Move the runtime depth stop earlier.
# This does not change the 0.521m runtime clearance,
# the 0.551m planning clearance, or the turn reserve.
start = code.index("def walk_to(section, goal, renderer, timeout=28):")
end = code.index(
    'print("\\n=== GATE 3 STEP 3: LIVE AUTONOMOUS ROUTE ===")',
    start
)
walk = code[start:end]

stop_anchor = "if nearest < 0.55:"
if walk.count(stop_anchor) != 1:
    raise SystemExit("WALK_DEPTH_THRESHOLD_ANCHOR_INVALID")

walk = walk.replace(
    stop_anchor, "if nearest < 0.68:", 1
)
code = code[:start] + walk + code[end:]

# Insert reusable A* functions, but never execute the old
# standalone test's top-level simulation/test loop.
planner_tree = ast.parse(PLANNER.read_text())
names = (
    "to_xy", "to_node", "inside", "distance_to_box",
    "clearance", "point_safe", "edge_safe",
    "plan", "verify_path", "box"
)
found = {
    n.name: n
    for n in planner_tree.body
    if isinstance(n, ast.FunctionDef)
}
if not all(name in found for name in names):
    raise SystemExit("PLANNER_FUNCTION_MISSING")

planner_source = PLANNER.read_text()
planner_code = "\n\n".join(
    ast.get_source_segment(planner_source, found[name])
    for name in names
)

# Prefix the standalone planner constants so its spatial
# clearance does not overwrite the MuJoCo runtime constant.
a_star_setup = '''
import heapq

GRID = 0.05
SAMPLE = 0.01
XMIN, XMAX = -2.0, 0.85
YMIN, YMAX = -0.60, 2.60
NX = round((XMAX-XMIN)/GRID)+1
NY = round((YMAX-YMIN)/GRID)+1
DIRECTIONS = [
    (dx, dy)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if dx or dy
]
A_STAR_CLEARANCE = 0.551
'''

# The original A* references global CLEARANCE.
# Supply it temporarily during route planning only.
recovery_functions = '''
def replanning_obstacles():
    rectangles = [
        tuple(low.tolist() + high.tolist())
    ]
    for gid in (box_id, extra_id, dynamic_id):
        center = data.geom_xpos[gid, :2]
        half = model.geom_size[gid, :2]
        rectangles.append((
            float(center[0] - half[0]),
            float(center[1] - half[1]),
            float(center[0] + half[0]),
            float(center[1] + half[1]),
        ))
    return rectangles

def replan_safe_route(current, destination, rectangles):
    global CLEARANCE
    saved = CLEARANCE
    try:
        CLEARANCE = A_STAR_CLEARANCE
        return plan(
            tuple(float(x) for x in current),
            tuple(float(x) for x in destination),
            rectangles,
        )
    finally:
        CLEARANCE = saved

def route_segments_safe(path, rectangles):
    global CLEARANCE
    saved = CLEARANCE
    try:
        CLEARANCE = A_STAR_CLEARANCE
        return verify_path(path, rectangles)
    finally:
        CLEARANCE = saved

def compress_path(path, rectangles):
    if not path:
        return []
    result = [tuple(path[0])]
    i = 0
    while i < len(path) - 1:
        selected = None
        for j in range(len(path)-1, i, -1):
            if route_segments_safe(
                [tuple(path[i]), tuple(path[j])],
                rectangles
            ):
                selected = j
                break
        if selected is None:
            return []
        result.append(tuple(path[selected]))
        i = selected
    return result

def attempt_dynamic_recovery(renderer):
    global stage
    state = {
        "new_obstacle_spawned": bool(dynamic_spawned),
        "fresh_depth_m": None,
        "reobservation_valid": False,
        "map_type": "hybrid_depth_and_mujoco_geometry",
        "planner_result": "NOT_STARTED",
        "replan_nodes": 0,
        "waypoints_attempted": 0,
        "waypoints_completed": 0,
        "outcome": "NOT_STARTED",
    }
    if not dynamic_spawned or stage != "hold":
        state["outcome"] = "NOT_AT_DYNAMIC_STOP"
        return state

    if guard()[0]:
        state["outcome"] = "STOP_GUARD_ACTIVE"
        return state

    try:
        # A new camera measurement is acquired after stopping.
        fresh = float(detect_depth(renderer))
    except Exception as exc:
        state["outcome"] = "SENSOR_EXCEPTION"
        state["sensor_error"] = repr(exc)
        return state

    state["fresh_depth_m"] = (
        fresh if math.isfinite(fresh) else None
    )
    if not math.isfinite(fresh):
        state["outcome"] = "SENSOR_INVALID_STOP"
        return state

    state["reobservation_valid"] = True
    rectangles = replanning_obstacles()

    current = data.qpos[:2].copy()
    goal_now = np.array([-1.10, 1.50])

    path, decision = replan_safe_route(
        current, goal_now, rectangles
    )
    state["planner_result"] = decision

    if decision != "PLANNED":
        state["outcome"] = "SAFE_NO_ROUTE_STOP"
        return state

    reduced = compress_path(path, rectangles)
    state["replan_nodes"] = len(path)
    state["compressed_waypoints"] = [
        list(v) for v in reduced
    ]

    if len(reduced) < 2:
        state["outcome"] = "PATH_COMPRESSION_FAILED"
        return state

    # A long route cannot safely be approximated by arbitrary
    # untested movement; cap the experiment and stop.
    if len(reduced) - 1 > 8:
        state["outcome"] = "WAYPOINT_BUDGET_STOP"
        return state

    for index, point in enumerate(reduced[1:], 1):
        waypoint = np.array(point)
        state["waypoints_attempted"] += 1

        if not route_segments_safe(
            [tuple(data.qpos[:2]), tuple(waypoint)],
            rectangles
        ):
            state["outcome"] = "PATH_INVALIDATED_STOP"
            return state

        if (
            not math.isfinite(estimated_clearance())
            or min(
                estimated_clearance(),
                xy_clearance(dynamic_id),
                xy_clearance(extra_id),
                xy_clearance(box_id),
            ) < CLEARANCE + 0.090
        ):
            state["outcome"] = "TURN_RESERVE_STOP"
            return state

        result = turn_to(
            "DYNAMIC_RECOVERY_TURN", waypoint
        )
        if result not in (
            "TURN_REACHED", "TURN_ALREADY_ALIGNED"
        ):
            state["outcome"] = "TURN_" + result
            return state

        # Recompute safety from the actual post-turn pose.
        now = data.qpos[:2].copy()
        if not route_segments_safe(
            [tuple(now), tuple(waypoint)], rectangles
        ):
            state["outcome"] = "POST_TURN_ROUTE_UNSAFE"
            return state

        desired = math.degrees(math.atan2(
            -(waypoint[1] - now[1]),
            -(waypoint[0] - now[0])
        ))
        if abs(angle_error(desired, yaw_deg())) > 2.0:
            state["outcome"] = "TURN_ALIGNMENT_STOP"
            return state

        walk_result = walk_to(
            "DYNAMIC_RECOVERY_WALK",
            waypoint,
            renderer,
            timeout=32,
        )

        if walk_result != "GOAL_TRIGGERED":
            state["outcome"] = "WALK_" + walk_result
            return state

        state["waypoints_completed"] += 1

    error = float(np.linalg.norm(
        data.qpos[:2] - goal_now
    ))
    state["final_goal_error_m"] = error
    state["outcome"] = (
        "REPLAN_AND_ARRIVAL"
        if error <= 0.15
        else "FINAL_ERROR_STOP"
    )
    return state
'''

# Install A* and recovery helpers before the main Gate 3 block.
anchor = 'print("\\n=== GATE 3 STEP 3: LIVE AUTONOMOUS ROUTE ===")'
if code.count(anchor) != 1:
    raise SystemExit("STAGE3_ANCHOR_INVALID")

code = code.replace(
    anchor,
    a_star_setup + "\n" + planner_code
    + "\n" + recovery_functions + "\n" + anchor,
    1,
)

# The recovery runs in the same MuJoCo process immediately
# after the original waypoint walk stops.
anchor = 'print("Live waypoint walk:", walk1)'
if code.count(anchor) != 1:
    raise SystemExit("WALK_RESULT_ANCHOR_INVALID")

insertion = '''
                        if (
                            dynamic_spawned
                            and walk1 in (
                                "DEPTH_OBSTACLE_STOP",
                                "CLEARANCE_STOP",
                            )
                        ):
                            print(
                                "DYNAMIC_RECOVERY: START",
                                flush=True
                            )
                            dynamic_recovery_report = (
                                attempt_dynamic_recovery(renderer)
                            )
                            print(
                                "DYNAMIC_RECOVERY_RESULT:",
                                json.dumps(
                                    dynamic_recovery_report
                                ),
                                flush=True
                            )
'''

code = code.replace(
    anchor, anchor + "\n" + insertion, 1
)

# Validate the whole generated controller before writing.
if "CLEARANCE = 0.521" not in code:
    raise SystemExit("RUNTIME_CLEARANCE_NOT_PRESERVED")
if "PLANNING_CLEARANCE = 0.551" not in code:
    raise SystemExit("PLANNING_CLEARANCE_NOT_PRESERVED")

ast.parse(code, filename=str(TARGET))
compile(code, str(TARGET), "exec")
TARGET.write_text(code)

print("=== BUILD ===", flush=True)
print("Integrated controller:", TARGET)
print("Source preserved:", SOURCE)
print("Compile: PASS")
print("Runtime clearance: 0.521")
print("Planning clearance: 0.551")
print("Turn reserve: 0.090")
print("Map: hybrid, not camera-only")

log_path = OUT / "dynamic_replan_integration.log"
report_path = ROOT / (
    PREFIX + "live_autonomous_route_v3.json"
)
trace_path = ROOT / (
    PREFIX + "live_autonomous_route_v3.csv"
)

for p in (report_path, trace_path):
    p.unlink(missing_ok=True)

print("\n=== LIVE MUJOCO INTEGRATION ===", flush=True)

try:
    with log_path.open("w") as log:
        completed = subprocess.run(
            [sys.executable, str(TARGET)],
            cwd=REPO / "dog_robot_v02",
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=600,
            check=False
        )
    rc = completed.returncode
except subprocess.TimeoutExpired:
    rc = 124

log = log_path.read_text(errors="replace")

for line in log.splitlines():
    if any(key in line for key in (
        "Stage 2 walk stop:",
        "DYNAMIC_OBSTACLE_APPEARED:",
        "Live waypoint walk:",
        "DYNAMIC_RECOVERY:",
        "DYNAMIC_RECOVERY_RESULT:",
        "DYNAMIC_RECOVERY_TURN:",
        "DYNAMIC_RECOVERY_WALK:",
        "GATE3_STEP3:",
        "Traceback",
        "Error:",
    )):
        print(line)

recovery = {}
for line in log.splitlines():
    if line.startswith("DYNAMIC_RECOVERY_RESULT:"):
        try:
            recovery = json.loads(
                line.split(":", 1)[1].strip()
            )
        except ValueError:
            pass

rows = []
if trace_path.exists():
    with trace_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

def numeric(column, mode=min):
    values = []
    for row in rows:
        try:
            value = float(row[column])
            if math.isfinite(value):
                values.append(value)
        except (KeyError, TypeError, ValueError):
            pass
    return mode(values) if values else None

guard_events = sum(
    bool(row.get("guard_reason", "").strip())
    for row in rows
)
contacts = sum(
    int(float(row.get("obstacle_contact_count", 0)))
    for row in rows
)

clearances = {
    column: numeric(column)
    for column in (
        "estimated_clearance_m",
        "extra_clearance_m",
        "red_clearance_m",
        "dynamic_clearance_m",
    )
}

height = numeric("height_m")
tilt = numeric("tilt_deg", max)

trajectory_safe = (
    bool(rows)
    and guard_events == 0
    and contacts == 0
    and all(
        value is not None and value >= 0.521
        for value in clearances.values()
    )
    and height is not None
    and height >= 0.16
    and tilt is not None
    and tilt <= 12.0
)

spawned = "DYNAMIC_OBSTACLE_APPEARED:" in log
reobserved = (
    recovery.get("reobservation_valid") is True
)
planned = (
    recovery.get("planner_result") == "PLANNED"
)
arrived = (
    recovery.get("outcome") == "REPLAN_AND_ARRIVAL"
)

summary = {
    "exit_code": rc,
    "dynamic_appeared": spawned,
    "dynamic_stop_detected": (
        "Live waypoint walk: DEPTH_OBSTACLE_STOP" in log
    ),
    "reobservation_valid": reobserved,
    "hybrid_astar_planned": planned,
    "recovery": recovery,
    "trace_rows": len(rows),
    "clearances": clearances,
    "minimum_height_m": height,
    "maximum_tilt_deg": tilt,
    "guard_samples": guard_events,
    "obstacle_contact_samples": contacts,
    "trajectory_checks_pass": trajectory_safe,
    "integrated_replan_arrival_pass": (
        rc == 0 and spawned and reobserved
        and planned and arrived and trajectory_safe
    ),
    "full_body_leg_sweep_verified": False,
    "camera_only_multi_obstacle_mapping_verified": False,
}
out = OUT / "dynamic_replan_integration_summary.json"
out.write_text(json.dumps(summary, indent=2))

print("\n=== STEP 4 / STEP 5 PARALLEL SUMMARY ===")
for key, val in summary.items():
    if key != "recovery":
        print(f"{key}: {val}")
print("Recovery outcome:", recovery.get("outcome"))
print("Summary:", out)
print("Log:", log_path)
print(
    "DYNAMIC_REPLAN_AND_ARRIVAL:",
    "PASS" if summary["integrated_replan_arrival_pass"]
    else "NOT_YET_PASS"
)
print(
    "STEP5_TRAJECTORY_SAFETY:",
    "PASS" if trajectory_safe else "NOT_YET_PASS"
)
print("STEP5_FULL_BODY_AUDIT: NOT_YET")
print("GATE3_COMPLETE: NO")
