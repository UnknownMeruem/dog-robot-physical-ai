from pathlib import Path
import ast
import json
import re
import subprocess
import sys

ROOT = Path("../gate3_avoidance_v01")
OUT = ROOT / "step4_early_staging_results"
OUT.mkdir(exist_ok=True)

CASES = ("case_02", "case_07")
STOP_TOLS = (0.52, 0.60)
BUILT = []

print("=== BUILD: EARLY STAGING EXPERIMENT ===", flush=True)

for case in CASES:
    source = ROOT / f"test_step4_preturn_{case}.py"
    if not source.is_file():
        raise SystemExit(f"Missing verified preturn source: {source}")

    original = source.read_text()

    expected = {
        "WAYPOINT_TOLERANCE = 0.42": 1,
        "0.37 <= error1 <= 0.44": 1,
        "CLEARANCE = 0.521": 1,
        "PLANNING_CLEARANCE = 0.551": 1,
        '"PRE_TURN_RESERVE_STOP"': 1,
    }
    for text, count in expected.items():
        if original.count(text) != count:
            raise SystemExit(
                f"{case}: unexpected source structure: "
                f"{text} count={original.count(text)}"
            )

    old_prefix = f"step4_preturn_{case}_"
    if f'"{old_prefix}' not in original:
        raise SystemExit(f"{case}: output prefix missing")

    for tolerance in STOP_TOLS:
        label = f"{int(round(tolerance*100)):02d}cm"
        name = f"{case}_{label}"
        code = original.replace(
            "WAYPOINT_TOLERANCE = 0.42",
            f"WAYPOINT_TOLERANCE = {tolerance:.2f}",
            1
        )
        code = code.replace(
            "0.37 <= error1 <= 0.44",
            f"{tolerance-0.05:.2f} <= error1 <= "
            f"{tolerance+0.02:.2f}",
            1
        )

        prefix = f"step4_early_{name}_"
        code = code.replace(
            f'"{old_prefix}', f'"{prefix}'
        )

        # Verify the old reference is gone and no safety
        # threshold or pre-turn gate was changed.
        if f'"{old_prefix}' in code:
            raise SystemExit(f"{name}: stale output path")
        for required in (
            "CLEARANCE = 0.521",
            "PLANNING_CLEARANCE = 0.551",
            "estimated_clearance() < CLEARANCE + 0.090",
        ):
            if required not in code:
                raise SystemExit(
                    f"{name}: safety condition missing: {required}"
                )

        target = ROOT / f"test_step4_early_{name}.py"
        ast.parse(code, filename=str(target))
        compile(code, str(target), "exec")
        BUILT.append((name, tolerance, prefix, target, code))

# Do not execute anything until all variants compile.
for name, tol, prefix, target, code in BUILT:
    target.write_text(code)
    print(f"BUILD_PASS: {name}")

print("ALL_4_CASES_COMPILE: PASS", flush=True)
print("Runtime clearance: 0.521m")
print("Planning clearance: 0.551m")
print("Pre-turn reserve: 0.090m")
print("Existing successful scripts: unchanged")

def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

RESULTS = []
print("\n=== RUN: EARLY STAGING EXPERIMENT ===", flush=True)

for name, tol, prefix, target, _ in BUILT:
    log_path = OUT / f"{name}.log"
    live_path = ROOT / (
        prefix + "early_live_replan_result_v3_early.json"
    )
    report_path = ROOT / (
        prefix + "live_autonomous_route_v3.json"
    )
    trace_path = ROOT / (
        prefix + "live_autonomous_route_v3.csv"
    )

    for path in (live_path, report_path, trace_path):
        path.unlink(missing_ok=True)

    print(f"\n=== {name} ===", flush=True)
    try:
        with log_path.open("w") as log:
            proc = subprocess.run(
                [sys.executable, str(target)],
                cwd=ROOT.parent / "dog_robot_v02",
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=360,
                check=False,
            )
        return_code = proc.returncode
    except subprocess.TimeoutExpired:
        return_code = 124

    log = log_path.read_text(errors="replace")
    live = read_json(live_path)
    report = read_json(report_path)

    markers = [
        "Stage 2 walk stop:",
        "Start clearance:",
        "GOAL_POLICY_CLEARANCE:",
        "Live turn 1:",
        "POST_TURN_CANDIDATES:",
        "Live waypoint walk:",
        "Live final walk:",
        "GATE3_STEP3:",
        "Traceback",
        "Error:",
    ]
    for line in log.splitlines():
        if any(marker in line for marker in markers):
            print(line)

    arrival = (
        return_code == 0
        and report.get("status") == "PASS"
        and report.get("pass") is True
    )

    refusal = (
        return_code == 0
        and "Live turn 1: PRE_TURN_RESERVE_STOP" in log
        and report.get("pass") is False
        and report.get("trace_samples") == 0
    )

    if arrival:
        outcome = "ARRIVAL_REPORTED_NEEDS_TRACE_AUDIT"
    elif refusal:
        outcome = "PREFLIGHT_REFUSAL"
    else:
        outcome = "NO_VERIFIED_ARRIVAL"

    item = {
        "case": name,
        "first_stop_tolerance_m": tol,
        "status": report.get("status"),
        "outcome": outcome,
        "exit_code": return_code,
        "observed_points": live.get("observed_points"),
        "trace_samples": report.get("trace_samples"),
        "log": str(log_path),
    }
    RESULTS.append(item)
    print("OUTCOME:", outcome, flush=True)

summary = OUT / "summary.json"
summary.write_text(
    json.dumps(RESULTS, indent=2, ensure_ascii=False)
)

print("\n=== EARLY STAGING SUMMARY ===")
for item in RESULTS:
    print(item["case"], item["outcome"],
          item["status"])

print("Reported arrivals:",
      sum(x["outcome"].startswith("ARRIVAL") for x in RESULTS))
print("Preflight refusals:",
      sum(x["outcome"] == "PREFLIGHT_REFUSAL" for x in RESULTS))
print("Summary:", summary)
print("STEP4_EARLY_STAGING_EXPERIMENT: COMPLETE")
print("Arrival is not safety-certified without trajectory audit.")
print("Step 4 complete: NO")
