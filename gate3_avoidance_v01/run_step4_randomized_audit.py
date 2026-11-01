from pathlib import Path
import ast
import csv
import json
import math
import random
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

ROOT = Path(__file__).resolve().parent
REPO = ROOT.parent
SCENE = REPO / "gate3_obstacle_stop_v01/obstacle_scene_extra_box.xml"
SOURCE = ROOT / "test_step4_unified_x_minus_05.py"
OUT = ROOT / "step4_randomized_audit"
OUT.mkdir(exist_ok=True)

SEED = 20261102
COUNT = 12
RUNTIME = 0.521
PLANNING = 0.551
GOAL = (-1.10, 1.50)
GOAL_TOL = 0.15

rng = random.Random(SEED)

if not SOURCE.exists() or not SCENE.exists():
    raise SystemExit("Missing validated source or obstacle scene")

base = SOURCE.read_text()
old_prefix = "step4_unified_x_minus_05_"

# Isolate results without changing the validated controller.
if f'"{old_prefix}' not in base:
    raise SystemExit("Validated output prefix missing")

pattern = re.compile(
    r'model = mujoco\.MjModel\.from_xml_path\('
    r'str\(Path\(r"[^"]+"\)\)\)'
)
if len(pattern.findall(base)) != 1:
    raise SystemExit("Unexpected model loader structure")

for required in (
    "CLEARANCE = 0.521",
    "PLANNING_CLEARANCE = 0.551",
    "goal_policy_unsafe = goal_policy_clearance < PLANNING_CLEARANCE",
    "range(6)",
):
    if required not in base:
        raise SystemExit("Required validated behavior missing: " + required)

print("=== STEP4 RANDOMIZED EXPERIMENT ===")
print("Seed:", SEED)
print("Cases:", COUNT)
print("Runtime threshold:", RUNTIME)
print("Planning threshold:", PLANNING)
print("Original controller unchanged")
print()

# Deterministic coverage of both sides of the original box.
offsets = [
    (-0.08, -0.08),
    (-0.08,  0.08),
    ( 0.08, -0.08),
    ( 0.08,  0.08),
]
while len(offsets) < COUNT:
    offsets.append((
        round(rng.uniform(-0.10, 0.10), 4),
        round(rng.uniform(-0.10, 0.10), 4),
    ))

generated = []

# Phase 1: generate and COMPILE every case before simulation.
for i, (dx, dy) in enumerate(offsets):
    case = f"case_{i:02d}"

    tree = ET.parse(SCENE)
    extra = [
        geom for geom in tree.getroot().iter("geom")
        if geom.get("name") == "extra_blocking_box"
    ]
    if len(extra) != 1:
        raise SystemExit("Expected one extra_blocking_box")

    position = [float(v) for v in extra[0].get("pos").split()]
    if len(position) != 3:
        raise SystemExit("Invalid extra box position")

    position[0] += dx
    position[1] += dy
    extra[0].set(
        "pos", " ".join(f"{v:.8f}" for v in position)
    )

    # Keep beside source XML to preserve relative asset paths.
    case_scene = SCENE.parent / f"step4_random_{case}.xml"
    tree.write(case_scene, encoding="unicode")

    code = pattern.sub(
        lambda _:
        'model = mujoco.MjModel.from_xml_path('
        f'str(Path(r"{case_scene}")))',
        base,
        count=1
    )

    new_prefix = f"step4_random_{case}_"
    code = code.replace(
        f'"{old_prefix}', f'"{new_prefix}'
    )
    target = ROOT / f"test_step4_random_{case}.py"

    ast.parse(code, filename=str(target))
    compile(code, str(target), "exec")
    generated.append({
        "case": case,
        "dx": dx,
        "dy": dy,
        "script": target,
        "scene": case_scene,
        "prefix": new_prefix,
        "code": code
    })

for test in generated:
    test["script"].write_text(test.pop("code"))

print("ALL_12_CASES_COMPILE: PASS", flush=True)

def read_json(path):
    try:
        return json.loads(path.read_text())
    except (ValueError, OSError):
        return {}

def floats(rows, key):
    output = []
    for row in rows:
        try:
            value = float(row.get(key, "nan"))
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            output.append(value)
    return output

def audit_csv(path):
    if not path.exists():
        return {"rows": 0}

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    summary = {"rows": len(rows)}
    for label, key in (
        ("min_estimated", "estimated_clearance_m"),
        ("min_extra", "extra_clearance_m"),
        ("min_red", "red_clearance_m"),
        ("min_height", "height_m"),
        ("max_tilt", "tilt_deg"),
    ):
        values = floats(rows, key)
        if not values:
            summary[label] = None
        elif label == "max_tilt":
            summary[label] = max(abs(v) for v in values)
        else:
            summary[label] = min(values)

    summary["guard_events"] = sum(
        bool(row.get("guard_reason")) for row in rows
    )
    return summary

def clearance(goal, low, high):
    return math.hypot(
        max(low[0]-goal[0], goal[0]-high[0], 0),
        max(low[1]-goal[1], goal[1]-high[1], 0),
    )

results = []

for test in generated:
    case = test["case"]
    prefix = test["prefix"]
    report_file = ROOT / (prefix + "live_autonomous_route_v3.json")
    live_file = ROOT / (
        prefix + "early_live_replan_result_v3_early.json"
    )
    csv_file = ROOT / (prefix + "live_autonomous_route_v3.csv")
    log_file = OUT / f"{case}.log"

    for path in (report_file, live_file, csv_file):
        path.unlink(missing_ok=True)

    print(
        f"\n=== {case} "
        f"dx={test['dx']:+.4f} "
        f"dy={test['dy']:+.4f} ===",
        flush=True,
    )

    try:
        with log_file.open("w") as f:
            proc = subprocess.run(
                [sys.executable, str(test["script"])],
                cwd=REPO / "dog_robot_v02",
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=360,
                check=False
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = 124

    run = read_json(report_file)
    live = read_json(live_file)
    audit = audit_csv(csv_file)

    low = live.get("estimated_box_min")
    high = live.get("estimated_box_max")

    valid = False
    if (
        isinstance(low, list) and isinstance(high, list)
        and len(low) == len(high) == 2
    ):
        try:
            valid = (
                all(math.isfinite(float(v)) for v in low + high)
                and all(float(a) <= float(b)
                        for a, b in zip(low, high))
            )
        except (ValueError, TypeError):
            valid = False

    goal_clear = (
        clearance(GOAL, low, high) if valid else None
    )

    if not valid:
        expected = "GOAL_MAP_INVALID_REJECTED"
    elif goal_clear < PLANNING:
        expected = "GOAL_UNSAFE_REJECTED"
    else:
        expected = "PASS"

    status = run.get("status")
    final_xy = run.get("final_xy")

    try:
        final_error = math.dist(final_xy, GOAL)
    except (TypeError, ValueError):
        final_error = None

    observed = live.get("observed_points", 0)
    common = (
        rc == 0
        and isinstance(observed, (int, float))
        and observed >= 30
    )

    if expected == "PASS":
        metrics = (
            audit.get("min_estimated"),
            audit.get("min_extra"),
            audit.get("min_red"),
            audit.get("min_height"),
            audit.get("max_tilt"),
        )
        passed = (
            common
            and status == "PASS"
            and run.get("pass") is True
            and final_error is not None
            and final_error <= GOAL_TOL
            and audit["rows"] > 0
            and audit.get("guard_events") == 0
            and all(v is not None for v in metrics)
            and all(
                audit[key] >= RUNTIME
                for key in (
                    "min_estimated",
                    "min_extra",
                    "min_red",
                )
            )
            and audit["min_height"] >= 0.16
            and audit["max_tilt"] <= 12.0
        )
    else:
        passed = (
            common
            and status == expected
            and run.get("pass") is False
            and run.get("trace_samples") == 0
            and audit["rows"] == 0
        )

    item = {
        "case": case,
        "seed": SEED,
        "dx_m": test["dx"],
        "dy_m": test["dy"],
        "goal_clearance_m": goal_clear,
        "expected": expected,
        "actual": status,
        "pass": bool(passed),
        "return_code": rc,
        "observed_points": observed,
        "goal_error_m": final_error,
        "audit": audit,
        "log": str(log_file),
    }
    results.append(item)

    print("Expected:", expected)
    print("Actual:", status)
    print("Goal clearance:", goal_clear)
    print("Trace:", audit)
    print("CASE_RESULT:", "PASS" if passed else "FAIL", flush=True)

summary_path = OUT / "summary.json"
summary_path.write_text(
    json.dumps(results, indent=2, ensure_ascii=False)
)

passed = sum(x["pass"] for x in results)
print("\n=== STEP 4 RANDOMIZED SUMMARY ===")
for r in results:
    print(
        f'{r["case"]:8s}',
        f'dx={r["dx_m"]:+.4f}',
        f'dy={r["dy_m"]:+.4f}',
        f'{str(r["actual"]):28s}',
        "PASS" if r["pass"] else "FAIL"
    )

print(f"Passed: {passed}/{COUNT}")
print("SEED:", SEED)
print("SUMMARY:", summary_path)
print(
    "STEP4_RANDOMIZED:",
    "PASS" if passed == COUNT else "FAIL"
)
print("This verifies sampled single-extra-box layouts only.")
print("Multiple-obstacle maps: NOT_VERIFIED")
print("Dynamic obstacle appearance/re-observation: NOT_VERIFIED")
print("Full-body contact and leg sweep: NOT_VERIFIED")
