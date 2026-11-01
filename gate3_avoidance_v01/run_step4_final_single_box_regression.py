from pathlib import Path
import ast
import csv
import json
import math
import subprocess
import sys

ROOT = Path("../gate3_avoidance_v01")
OUT = ROOT / "step4_final_single_box_results"
OUT.mkdir(exist_ok=True)

RUNTIME = 0.521
PLANNING = 0.551
GOAL = [-1.10, 1.50]
TOL = 0.15

def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def trace_audit(path):
    if not path.exists():
        return {"rows": 0, "valid": False}

    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    result = {"rows": len(rows)}
    required = {
        "estimated_clearance_m": "min_est",
        "extra_clearance_m": "min_extra",
        "red_clearance_m": "min_red",
        "height_m": "min_height",
        "tilt_deg": "max_tilt",
    }

    for column, name in required.items():
        vals = []
        for row in rows:
            try:
                value = float(row[column])
            except (ValueError, TypeError, KeyError):
                result["valid"] = False
                result["error"] = f"Invalid {column}"
                return result

            if not math.isfinite(value):
                result["valid"] = False
                result["error"] = f"Nonfinite {column}"
                return result

            vals.append(value)

        result[name] = (
            max(abs(v) for v in vals)
            if name == "max_tilt" and vals
            else min(vals) if vals else None
        )

    result["guard_events"] = sum(
        bool(row.get("guard_reason", "").strip())
        for row in rows
    )

    result["valid"] = (
        bool(rows)
        and result["guard_events"] == 0
        and all(
            result[k] >= RUNTIME
            for k in ("min_est", "min_extra", "min_red")
        )
        and result["min_height"] >= 0.16
        and result["max_tilt"] <= 12.0
    )
    return result

def goal_error(report):
    try:
        point = [float(x) for x in report["final_xy"]]
        if len(point) != 2 or not all(map(math.isfinite, point)):
            return math.inf
        return math.dist(point, GOAL)
    except (KeyError, ValueError, TypeError):
        return math.inf

def observed_goal_clearance(live):
    try:
        low = [float(x) for x in live["estimated_box_min"]]
        high = [float(x) for x in live["estimated_box_max"]]
        if (
            len(low) != 2 or len(high) != 2
            or not all(map(math.isfinite, low + high))
            or any(a > b for a, b in zip(low, high))
        ):
            return None
        return math.hypot(*(
            max(a-g, g-b, 0.0)
            for a, b, g in zip(low, high, GOAL)
        ))
    except (KeyError, ValueError, TypeError):
        return None

print("=== PHASE 1: AUDIT FOUR RECENT ARRIVALS ===", flush=True)
four_results = []

for case in ("case_02", "case_07"):
    for cm in (52, 60):
        label = f"{case}_{cm}cm"
        prefix = f"step4_early_{label}_"
        report = read_json(
            ROOT / (prefix + "live_autonomous_route_v3.json")
        )
        trace = trace_audit(
            ROOT / (prefix + "live_autonomous_route_v3.csv")
        )
        err = goal_error(report)

        passed = (
            report.get("status") == "PASS"
            and report.get("pass") is True
            and trace["valid"]
            and err <= TOL
            and trace["rows"] == report.get("trace_samples")
        )
        four_results.append({"case": label, "pass": passed,
                             "error_m": err, "trace": trace})

        print(
            label,
            "PASS" if passed else "FAIL",
            f"goal_error={err:.4f}m",
            "minimum_estimated=",
            trace.get("min_est"),
            "guard_events=",
            trace.get("guard_events"),
            flush=True
        )

(OUT / "four_arrival_audit.json").write_text(
    json.dumps(four_results, indent=2)
)

if not all(x["pass"] for x in four_results):
    print("FOUR_ARRIVAL_AUDIT: FAIL")
    print("12-case regression not started.")
    raise SystemExit(1)

print("FOUR_ARRIVAL_AUDIT: PASS", flush=True)

print("\n=== PHASE 2: BUILD TWELVE TESTS ===", flush=True)
built = []

for i in range(12):
    case = f"case_{i:02d}"
    source = ROOT / f"test_step4_random_{case}.py"
    if not source.is_file():
        raise SystemExit(f"Missing source {source}")

    code = source.read_text()

    replacements = [
        ("WAYPOINT_TOLERANCE = 0.42",
         "WAYPOINT_TOLERANCE = 0.52"),
        ("0.37 <= error1 <= 0.44",
         "0.47 <= error1 <= 0.54"),
    ]

    for old, new in replacements:
        if code.count(old) != 1:
            raise SystemExit(
                f"{case}: unexpected replacement count for {old}"
            )
        code = code.replace(old, new, 1)

    # Experimental preturn gate, matching the prior verified test.
    turn_call = 'turn_to("LIVE_TURN1", via)'
    if code.count(turn_call) != 1:
        raise SystemExit(f"{case}: first-turn call not unique")

    code = code.replace(
        turn_call,
        '''(
            "PRE_TURN_RESERVE_STOP"
            if (
                not np.isfinite(estimated_clearance())
                or estimated_clearance() < CLEARANCE + 0.090
                or guard()[0]
            )
            else turn_to("LIVE_TURN1", via)
        )''',
        1
    )

    for required in (
        "CLEARANCE = 0.521",
        "PLANNING_CLEARANCE = 0.551",
        "goal_policy_unsafe = goal_policy_clearance < PLANNING_CLEARANCE",
        "range(6)",
    ):
        if required not in code:
            raise SystemExit(
                f"{case}: required safety condition absent: {required}"
            )

    old_prefix = f"step4_random_{case}_"
    new_prefix = f"step4_final_{case}_"
    if f'"{old_prefix}' not in code:
        raise SystemExit(f"{case}: output prefix missing")
    code = code.replace(
        f'"{old_prefix}', f'"{new_prefix}'
    )

    target = ROOT / f"test_step4_final_{case}.py"
    ast.parse(code, filename=str(target))
    compile(code, str(target), "exec")
    built.append((case, target, code, new_prefix))

for case, target, code, _ in built:
    target.write_text(code)
    print("BUILD_PASS:", case)

print("ALL_12_COMPILE: PASS", flush=True)

print("\n=== PHASE 3: SIMULATE AND AUDIT ===", flush=True)
results = []

for case, script, _, prefix in built:
    report_path = ROOT / (
        prefix + "live_autonomous_route_v3.json"
    )
    live_path = ROOT / (
        prefix + "early_live_replan_result_v3_early.json"
    )
    csv_path = ROOT / (
        prefix + "live_autonomous_route_v3.csv"
    )
    log_path = OUT / (case + ".log")

    for p in (report_path, live_path, csv_path):
        p.unlink(missing_ok=True)

    print(f"\n=== {case} ===", flush=True)

    try:
        with log_path.open("w") as f:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=ROOT.parent / "dog_robot_v02",
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=360,
                check=False
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = 124

    report = read_json(report_path)
    live = read_json(live_path)
    trace = trace_audit(csv_path)
    clear = observed_goal_clearance(live)
    err = goal_error(report)

    if clear is None:
        expected = "GOAL_MAP_INVALID_REJECTED"
    elif clear < PLANNING:
        expected = "GOAL_UNSAFE_REJECTED"
    else:
        expected = "ARRIVAL_OR_PREFLIGHT_REFUSAL"

    status = report.get("status", "NO_REPORT")

    arrival_pass = (
        rc == 0
        and status == "PASS"
        and report.get("pass") is True
        and trace["valid"]
        and err <= TOL
        and trace["rows"] == report.get("trace_samples")
    )

    rejection_pass = (
        rc == 0
        and expected in (
            "GOAL_UNSAFE_REJECTED",
            "GOAL_MAP_INVALID_REJECTED"
        )
        and status == expected
        and report.get("pass") is False
        and report.get("trace_samples") == 0
        and trace["rows"] == 0
    )

    # A preflight refusal can be a valid precaution but does not
    # prove that a physically feasible destination is unreachable.
    preflight_refusal = (
        rc == 0
        and "Live turn 1: PRE_TURN_RESERVE_STOP" in
        log_path.read_text(errors="replace")
        and report.get("pass") is False
        and report.get("trace_samples") == 0
    )

    passed = arrival_pass or rejection_pass
    item = {
        "case": case,
        "expected": expected,
        "observed": status,
        "goal_clearance_m": clear,
        "goal_error_m": err if math.isfinite(err) else None,
        "arrival_pass": arrival_pass,
        "rejection_pass": rejection_pass,
        "preflight_refusal": preflight_refusal,
        "pass": bool(passed),
        "return_code": rc,
        "trace": trace,
    }
    results.append(item)

    print("Expected:", expected)
    print("Observed:", status)
    print("Goal clearance:", clear)
    print("Arrival:", arrival_pass)
    print("Validated goal refusal:", rejection_pass)
    print("Preflight refusal:", preflight_refusal)
    print("Trace metrics:", trace)
    print("CASE_RESULT:", "PASS" if passed else "FAIL",
          flush=True)

summary_path = OUT / "summary.json"
summary_path.write_text(
    json.dumps(results, indent=2, ensure_ascii=False)
)

n = sum(x["pass"] for x in results)
print("\n=== STEP 4 SINGLE-BOX REGRESSION SUMMARY ===")
for x in results:
    print(x["case"], x["observed"],
          "PASS" if x["pass"] else "FAIL")

print(f"Passed: {n}/12")
print(
    "STEP4_SINGLE_BOX_REGRESSION:",
    "PASS" if n == 12 else "FAIL"
)
print("Summary:", summary_path)
print("Multiple obstacles: NOT YET TESTED")
print("Dynamic obstacle appearance: NOT YET TESTED")
print("Full-body independent safety audit: NOT DONE")
