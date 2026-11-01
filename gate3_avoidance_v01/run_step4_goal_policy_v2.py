from pathlib import Path
import ast
import json
import math
import subprocess
import sys

import numpy as np

root = Path(__file__).resolve().parent
out = root / "step4_goal_policy_v2_results"
out.mkdir(exist_ok=True)

cases = ("baseline", "plus_10cm", "minus_10cm")

# Keep the original if/else structure intact.
anchor = '''if (
    live_report.get("status") != "PLANNED"
    or reason2 != "DEPTH_OBSTACLE_STOP"
    or stage != "hold"
):'''

modified_anchor = '''if (
    goal_policy_unsafe
    or live_report.get("status") != "PLANNED"
    or reason2 != "DEPTH_OBSTACLE_STOP"
    or stage != "hold"
):'''

policy_before = '''
# ===== Goal policy: evaluated before route-control execution =====
goal_policy_target = np.array([-1.10, 1.50])
goal_policy_low = np.asarray(
    live_report.get("estimated_box_min", []),
    dtype=float,
)
goal_policy_high = np.asarray(
    live_report.get("estimated_box_max", []),
    dtype=float,
)

goal_policy_valid = (
    goal_policy_low.shape == (2,)
    and goal_policy_high.shape == (2,)
    and np.isfinite(goal_policy_low).all()
    and np.isfinite(goal_policy_high).all()
    and np.all(goal_policy_low <= goal_policy_high)
)

goal_policy_clearance = float("nan")
goal_policy_unsafe = False

if goal_policy_valid:
    diff = np.maximum(
        np.maximum(
            goal_policy_low - goal_policy_target,
            goal_policy_target - goal_policy_high,
        ),
        0.0,
    )
    goal_policy_clearance = float(np.linalg.norm(diff))
    goal_policy_unsafe = goal_policy_clearance < 0.521

print("GOAL_POLICY_CLEARANCE:",
      f"{goal_policy_clearance:.4f}m")
print("GOAL_POLICY_UNSAFE:", goal_policy_unsafe)

'''

# This section runs after any route execution but BEFORE
# the report is assembled. It classifies a genuine refusal.
policy_after = '''
# ===== Finalize safe-goal-rejection classification =====
if goal_policy_unsafe:
    if (
        reason2 == "DEPTH_OBSTACLE_STOP"
        and stage == "hold"
        and len(trace) == 0
        and not stage3_ok
    ):
        stage3_reason = "GOAL_UNSAFE_REJECTED"
    else:
        stage3_reason = "GOAL_POLICY_STOP_VALIDATION_FAILED"

'''

report_anchor = 'report = {\n    "step": 3,'

outputs = [
    "waypoint1_early_live_v3.csv",
    "depth_early_live_v3.csv",
    "early_live_replan_result_v3_early.json",
    "live_obstacle_points_v3.csv",
    "live_autonomous_route_v3.csv",
    "live_autonomous_route_v3.json",
]

scripts = []

print("=== PHASE 1: BUILD AND VALIDATE ALL CASES ===")

for case in cases:
    original = root / f"test_step4_expanded_{case}.py"
    if not original.exists():
        raise SystemExit(f"MISSING_SOURCE: {original}")

    code = original.read_text()

    if code.count(anchor) != 1:
        raise SystemExit(
            f"UNEXPECTED_IF_STRUCTURE: {case} "
            f"count={code.count(anchor)}"
        )

    if code.count(report_anchor) != 1:
        raise SystemExit(
            f"UNEXPECTED_REPORT_STRUCTURE: {case}"
        )

    # Insert policy logic before the existing if/else.
    code = code.replace(
        anchor,
        policy_before + modified_anchor,
        1,
    )

    # Keep the original else exactly as it was.
    code = code.replace(
        report_anchor,
        policy_after + report_anchor,
        1,
    )

    # Rename ONLY output files.
    # Do not rename the scene XML path.
    for filename in outputs:
        old_name = f"step4_expanded_{case}_{filename}"
        new_name = f"step4_policy_v2_{case}_{filename}"

        if code.count(f'"{old_name}"') < 1:
            raise SystemExit(
                f"OUTPUT_REFERENCE_MISSING: {old_name}"
            )

        code = code.replace(
            f'"{old_name}"',
            f'"{new_name}"',
        )

    target = root / f"test_step4_goal_policy_v2_{case}.py"

    # Fail before writing or executing any broken code.
    ast.parse(code, filename=str(target))
    compile(code, str(target), "exec")
    target.write_text(code)

    scripts.append((case, target))

    print(f"BUILD_VALID: {case}")

print("ALL_CASES_COMPILE: PASS")

print("\n=== PHASE 2: SIMULATE ===")

results = []

for case, script in scripts:
    prefix = f"step4_policy_v2_{case}_"

    report_file = root / (
        prefix + "live_autonomous_route_v3.json"
    )
    live_file = root / (
        prefix + "early_live_replan_result_v3_early.json"
    )
    log_file = out / f"{case}.log"

    # Prevent old reports from being accepted.
    for path in (report_file, live_file):
        if path.exists():
            path.unlink()

    print(f"\n--- CASE: {case} ---", flush=True)

    try:
        with log_file.open("w") as log:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=root.parent / "dog_robot_v02",
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=360,
                check=False,
            )
        return_code = proc.returncode
    except subprocess.TimeoutExpired:
        return_code = 124

    def read_json(path):
        if not path.exists():
            return {}
        try:
            return json.loads(path.read_text())
        except (ValueError, OSError):
            return {}

    run = read_json(report_file)
    live = read_json(live_file)

    low = np.asarray(
        live.get("estimated_box_min", []), dtype=float
    )
    high = np.asarray(
        live.get("estimated_box_max", []), dtype=float
    )

    valid_map = (
        low.shape == (2,)
        and high.shape == (2,)
        and np.isfinite(low).all()
        and np.isfinite(high).all()
        and np.all(low <= high)
    )

    goal_clearance = float("nan")
    if valid_map:
        goal = np.array([-1.10, 1.50])
        diff = np.maximum(
            np.maximum(low-goal, goal-high), 0.0
        )
        goal_clearance = float(np.linalg.norm(diff))

    if not valid_map:
        expected = "INVALID_MAP_SAFE_STOP"
        passed = (
            return_code == 0
            and run.get("pass") is False
            and run.get("trace_samples") == 0
        )

    elif goal_clearance < 0.521:
        expected = "SAFE_GOAL_REJECTION"
        passed = (
            return_code == 0
            and run.get("status") == "GOAL_UNSAFE_REJECTED"
            and run.get("pass") is False
            and run.get("trace_samples") == 0
            and live.get("observed_points", 0) >= 30
        )

    else:
        expected = "SAFE_GOAL_ARRIVAL"
        passed = (
            return_code == 0
            and run.get("status") == "PASS"
            and run.get("pass") is True
            and run.get("trace_samples", 0) > 0
            and live.get("observed_points", 0) >= 30
            and np.linalg.norm(
                np.asarray(run.get("final_xy", [999, 999]))
                - np.array([-1.10, 1.50])
            ) <= 0.15
        )

    item = {
        "case": case,
        "expected": expected,
        "status": run.get("status", "NO_REPORT"),
        "goal_clearance_m": (
            goal_clearance if math.isfinite(goal_clearance)
            else None
        ),
        "observed_points": live.get("observed_points", 0),
        "trace_samples": run.get("trace_samples", 0),
        "return_code": return_code,
        "pass": bool(passed),
    }
    results.append(item)

    lines = log_file.read_text(
        errors="replace"
    ).splitlines()

    for line in lines:
        if any(key in line for key in (
            "GOAL_POLICY_CLEARANCE:",
            "GOAL_POLICY_UNSAFE:",
            "Live waypoint walk:",
            "Live final walk:",
            "GATE3_STEP3:",
            "Traceback",
            "SyntaxError:",
        )):
            print(line)

    print("Expected:", expected)
    print("Actual status:", item["status"])
    print("CASE_POLICY_V2:",
          "PASS" if passed else "FAIL")
    print("Log:", log_file)

summary_path = out / "summary.json"
summary_path.write_text(
    json.dumps(results, indent=2, ensure_ascii=False)
)

print("\n=== GATE 3 STEP 4 POLICY SUMMARY ===")
for item in results:
    print(
        item["case"],
        item["expected"],
        "PASS" if item["pass"] else "FAIL"
    )

n_pass = sum(item["pass"] for item in results)

print(f"Passed: {n_pass}/{len(results)}")
print(
    "STEP4_GOAL_POLICY_V2:",
    "PASS" if n_pass == len(results) else "FAIL"
)
print("Summary:", summary_path)
print("Remaining Step 4 tests: unknown layouts,")
print("multiple obstacles, sensor faults, safe stopping.")
