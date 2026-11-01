from pathlib import Path
import ast
import csv
import json
import math
import subprocess
import sys

root = Path(__file__).resolve().parent
output = root / "step4_unified_regression"
output.mkdir(exist_ok=True)

cases = (
    "x_minus_05", "x_plus_05", "y_minus_05",
    "y_plus_05", "xy_shift", "nan_map_fault",
)
runtime_min = 0.521
planning_min = 0.551

built = []

print("=== PHASE 1: VERIFY AND BUILD ===", flush=True)

for case in cases:
    source = root / f"test_step4_robust_{case}.py"
    if not source.exists():
        raise SystemExit(f"Missing source: {source}")

    code = source.read_text()

    old_policy = (
        "goal_policy_unsafe = "
        "goal_policy_clearance < 0.521"
    )
    new_policy = (
        "goal_policy_unsafe = "
        "goal_policy_clearance < PLANNING_CLEARANCE"
    )

    if code.count(old_policy) != 1:
        raise SystemExit(
            f"Unexpected goal-policy structure: {case}"
        )
    code = code.replace(old_policy, new_policy, 1)

    tree = ast.parse(code)
    correction_loops = [
        node for node in ast.walk(tree)
        if isinstance(node, ast.For)
        and isinstance(node.target, ast.Name)
        and node.target.id == "correction_iteration"
    ]
    if len(correction_loops) != 1:
        raise SystemExit(
            f"Unexpected correction loop: {case}"
        )

    loop = correction_loops[0]
    lines = code.splitlines(keepends=True)
    line = lines[loop.lineno - 1]

    if line.count("range(3)") != 1:
        raise SystemExit(f"Unexpected loop limit: {case}")

    lines[loop.lineno - 1] = line.replace(
        "range(3)", "range(6)", 1
    )
    code = "".join(lines)

    for constant in (
        "CLEARANCE = 0.521",
        "PLANNING_CLEARANCE = 0.551",
    ):
        if constant not in code:
            raise SystemExit(
                f"Safety constant missing: {case}, {constant}"
            )

    # Keep each test's outputs separate.
    old_prefix = f"step4_robust_{case}_"
    new_prefix = f"step4_unified_{case}_"

    if f'"{old_prefix}' not in code:
        raise SystemExit(f"Output prefix missing: {case}")

    code = code.replace(
        f'"{old_prefix}', f'"{new_prefix}'
    )

    target = root / f"test_step4_unified_{case}.py"
    ast.parse(code, filename=str(target))
    compile(code, str(target), "exec")
    built.append((case, target, code, new_prefix))

# Only write after ALL six programs pass compilation.
for case, target, code, prefix in built:
    target.write_text(code)
    print(f"BUILD_PASS: {case}")

print("ALL_CASES_COMPILE: PASS", flush=True)

print("\n=== PHASE 2: SIX-CASE SIMULATION ===",
      flush=True)

results = []

def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def trace_audit(path):
    if not path.exists():
        return {"rows": 0, "guards": None, "min_est": None}

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    guards = sum(bool(r.get("guard_reason")) for r in rows)
    distances = []
    for r in rows:
        try:
            d = float(r.get("estimated_clearance_m", "nan"))
            if math.isfinite(d):
                distances.append(d)
        except ValueError:
            pass

    return {
        "rows": len(rows),
        "guards": guards,
        "min_est": min(distances) if distances else None,
    }

for case, script, _, prefix in built:
    report_file = root / (
        prefix + "live_autonomous_route_v3.json"
    )
    live_file = root / (
        prefix + "early_live_replan_result_v3_early.json"
    )
    trace_file = root / (
        prefix + "live_autonomous_route_v3.csv"
    )
    log_file = output / f"{case}.log"

    for file in (report_file, live_file, trace_file):
        file.unlink(missing_ok=True)

    print(f"\n--- {case} ---", flush=True)

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
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        returncode = 124

    report = read_json(report_file)
    live = read_json(live_file)
    audit = trace_audit(trace_file)

    status = report.get("status")
    points = live.get("observed_points", 0)

    # Expected classifications come from the independent
    # earlier audit, not the controller's new status.
    if case in ("y_plus_05", "xy_shift"):
        expected = "GOAL_UNSAFE_REJECTED"
    elif case == "nan_map_fault":
        expected = "GOAL_MAP_INVALID_REJECTED"
    else:
        expected = "PASS"

    if expected == "PASS":
        final = report.get("final_xy")
        try:
            goal_error = math.dist(
                final, [-1.10, 1.50]
            )
        except (TypeError, ValueError):
            goal_error = float("inf")

        passed = (
            returncode == 0
            and status == "PASS"
            and report.get("pass") is True
            and points >= 30
            and audit["rows"] > 0
            and audit["guards"] == 0
            and audit["min_est"] is not None
            and audit["min_est"] >= runtime_min
            and goal_error <= 0.15
        )
    else:
        passed = (
            returncode == 0
            and status == expected
            and report.get("pass") is False
            and points >= 30
            and report.get("trace_samples") == 0
            and audit["rows"] == 0
        )

    for line in log_file.read_text(
        errors="replace"
    ).splitlines():
        if any(word in line for word in (
            "GOAL_POLICY_CLEARANCE:",
            "GOAL_POLICY_UNSAFE:",
            "POST_TURN_CANDIDATES:",
            "POST_TURN_PREDICTED_CLEARANCE:",
            "Live waypoint walk:",
            "Live final walk:",
            "GATE3_STEP3:",
            "Traceback",
            "Error:",
        )):
            print(line)

    item = {
        "case": case,
        "expected": expected,
        "observed": status,
        "pass": bool(passed),
        "returncode": returncode,
        "observed_points": points,
        "trace": audit,
    }
    results.append(item)

    print("Expected:", expected)
    print("Observed:", status)
    print("CASE_RESULT:",
          "PASS" if passed else "FAIL")
    print("Log:", log_file)

summary = output / "summary.json"
summary.write_text(
    json.dumps(results, indent=2, ensure_ascii=False)
)

count = sum(r["pass"] for r in results)

print("\n=== STEP 4 UNIFIED REGRESSION ===")
for r in results:
    print(
        f'{r["case"]:17s}',
        f'{str(r["observed"]):28s}',
        "PASS" if r["pass"] else "FAIL",
    )

print(f"Passed: {count}/{len(results)}")
print(
    "STEP4_UNIFIED_REGRESSION:",
    "PASS" if count == len(results) else "FAIL"
)
print("Summary:", summary)
print("Original programs unchanged")
print("Full Step 4 acceptance: NOT YET")
