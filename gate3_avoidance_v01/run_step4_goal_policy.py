from pathlib import Path
import subprocess
import json
import sys

root = Path(__file__).resolve().parent
out = root / "step4_goal_policy_results"
out.mkdir(exist_ok=True)

cases = ("baseline", "plus_10cm", "minus_10cm")
summary = []

anchor = '''if (
    live_report.get("status") != "PLANNED"
    or reason2 != "DEPTH_OBSTACLE_STOP"
    or stage != "hold"
):'''

replacement = '''# Check goal clearance using THIS run's camera observation.
# Reject an unsafe goal before any new replan turn or walk.
bounds_low = np.asarray(
    live_report.get("estimated_box_min", []), dtype=float
)
bounds_high = np.asarray(
    live_report.get("estimated_box_max", []), dtype=float
)
goal_for_policy = np.array([-1.10, 1.50])

if bounds_low.shape == (2,) and bounds_high.shape == (2,):
    offset = np.maximum(
        np.maximum(bounds_low - goal_for_policy,
                   goal_for_policy - bounds_high), 0.0
    )
    goal_clearance_policy = float(np.linalg.norm(offset))
else:
    goal_clearance_policy = float("nan")

print("GOAL_POLICY_CLEARANCE:", goal_clearance_policy)

if reason2 != "DEPTH_OBSTACLE_STOP" or stage != "hold":
    stage3_reason = "OBSERVATION_NOT_READY"
elif not np.isfinite(goal_clearance_policy):
    stage3_reason = "GOAL_MAP_INVALID_REJECTED"
elif goal_clearance_policy < CLEARANCE:
    stage3_reason = "GOAL_UNSAFE_REJECTED"
elif (
    live_report.get("status") != "PLANNED"
):
    stage3_reason = "LIVE_OBSERVATION_NOT_READY"
else:'''

for case in cases:
    original = root / f"test_step4_expanded_{case}.py"
    if not original.exists():
        raise SystemExit(f"Missing prior test: {original}")

    code = original.read_text()

    if code.count(anchor) != 1:
        raise SystemExit(
            f"Unexpected policy insertion point: {case}"
        )
    code = code.replace(anchor, replacement, 1)

    # Ensure fresh output files, not results from older trials.
    old_prefix = f"step4_expanded_{case}_"
    new_prefix = f"step4_policy_{case}_"
    code = code.replace(old_prefix, new_prefix)

    target = root / f"test_step4_goal_policy_{case}.py"
    compile(code, str(target), "exec")
    target.write_text(code)

    report_path = root / (
        f"{new_prefix}live_autonomous_route_v3.json"
    )
    live_path = root / (
        f"{new_prefix}early_live_replan_result_v3_early.json"
    )

    # Prevent stale data from creating false PASS.
    for path in (report_path, live_path):
        if path.exists():
            path.unlink()

    log_path = out / f"{case}.log"
    print(f"\n=== GOAL POLICY: {case} ===", flush=True)

    try:
        with log_path.open("w") as log:
            process = subprocess.run(
                [sys.executable, str(target)],
                cwd=root.parent / "dog_robot_v02",
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=360
            )
        returncode = process.returncode
    except subprocess.TimeoutExpired:
        returncode = 124

    run = json.loads(report_path.read_text()) \
        if report_path.exists() else {}
    live = json.loads(live_path.read_text()) \
        if live_path.exists() else {}

    goal_unsafe = False
    low = live.get("estimated_box_min")
    high = live.get("estimated_box_max")
    if low and high and len(low) == len(high) == 2:
        import numpy as np
        goal = np.array([-1.10, 1.50])
        lower = np.array(low)
        upper = np.array(high)
        offset = np.maximum(
            np.maximum(lower-goal, goal-upper), 0
        )
        goal_unsafe = float(np.linalg.norm(offset)) < 0.521

    if goal_unsafe:
        passed = (
            returncode == 0
            and run.get("status") == "GOAL_UNSAFE_REJECTED"
            and run.get("trace_samples") == 0
            and run.get("pass") is False
        )
        expected = "SAFE_GOAL_REJECTION"
    else:
        passed = (
            returncode == 0
            and run.get("pass") is True
            and run.get("status") == "PASS"
        )
        expected = "SAFE_GOAL_ARRIVAL"

    item = {
        "case": case,
        "expected": expected,
        "status": run.get("status"),
        "pass": bool(passed),
        "returncode": returncode,
    }
    summary.append(item)

    print("Expected:", expected)
    print("Observed:", run.get("status", "NO_REPORT"))
    print("CASE_POLICY:",
          "PASS" if passed else "FAIL")
    print("Log:", log_path)

path = out / "summary.json"
path.write_text(json.dumps(summary, indent=2))

print("\n=== STEP 4 GOAL POLICY SUMMARY ===")
for item in summary:
    print(
        item["case"],
        item["expected"],
        "PASS" if item["pass"] else "FAIL"
    )

print("Passed:", sum(x["pass"] for x in summary),
      "/", len(summary))
print("STEP4_GOAL_POLICY:",
      "PASS" if all(x["pass"] for x in summary) else "FAIL")
print("More unknown-layout and sensor-fault tests still required.")
