from pathlib import Path
import ast
import csv
import json
import math
import subprocess
import sys

ROOT = Path("../gate3_avoidance_v01")
OUT = ROOT / "step4_pre_turn_gate_results"
OUT.mkdir(exist_ok=True)

CASES = ("case_00", "case_02", "case_06", "case_07")
RUNTIME = 0.521
PLANNING = 0.551
TURN_RESERVE = 0.090

built = []

print("=== PHASE 1: BUILD AND COMPILE ===", flush=True)

for case in CASES:
    source = ROOT / f"test_step4_random_{case}.py"
    if not source.is_file():
        raise SystemExit(f"Missing source: {source}")

    code = source.read_text()
    tree = ast.parse(code)

    calls = [
        n for n in ast.walk(tree)
        if isinstance(n, ast.Call)
        and isinstance(n.func, ast.Name)
        and n.func.id == "turn_to"
        and n.args
        and isinstance(n.args[0], ast.Constant)
        and n.args[0].value == "LIVE_TURN1"
    ]

    if len(calls) != 1:
        raise SystemExit(
            f"{case}: expected exactly one LIVE_TURN1 call"
        )

    call = calls[0]
    original = ast.get_source_segment(code, call)
    if original != 'turn_to("LIVE_TURN1", via)':
        raise SystemExit(
            f"{case}: unexpected first-turn expression: {original}"
        )

    replacement = '''(
        "PRE_TURN_RESERVE_STOP"
        if (
            not np.isfinite(estimated_clearance())
            or estimated_clearance() < CLEARANCE + 0.090
            or guard()[0]
        )
        else turn_to("LIVE_TURN1", via)
    )'''

    # Replace only the AST-verified call in its own line.
    lines = code.splitlines(keepends=True)
    index = call.lineno - 1
    line = lines[index]
    if line.count(original) != 1:
        raise SystemExit(f"{case}: ambiguous turn-call line")
    lines[index] = line.replace(original, replacement, 1)
    code = "".join(lines)

    old_prefix = f"step4_random_{case}_"
    new_prefix = f"step4_preturn_{case}_"

    if f'"{old_prefix}' not in code:
        raise SystemExit(f"{case}: output prefix missing")

    # Only quoted output filenames, never scene references.
    code = code.replace(
        f'"{old_prefix}', f'"{new_prefix}'
    )

    for required in (
        "CLEARANCE = 0.521",
        "PLANNING_CLEARANCE = 0.551",
    ):
        if required not in code:
            raise SystemExit(
                f"{case}: required safety setting missing"
            )

    target = ROOT / f"test_step4_preturn_{case}.py"
    ast.parse(code, filename=str(target))
    compile(code, str(target), "exec")
    built.append((case, target, code, new_prefix))

# Write only after every script has compiled.
for case, target, code, _ in built:
    target.write_text(code)
    print("BUILD_PASS:", case)

print("ALL_CASES_COMPILE: PASS", flush=True)
print("Runtime clearance:", RUNTIME)
print("Planning clearance:", PLANNING)
print("Experimental first-turn reserve:", TURN_RESERVE)
print("No original controllers modified", flush=True)

def read_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def audit_csv(path):
    if not path.exists():
        return {"rows": 0, "guards": 0, "minimum": None}
    with path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    distances = []
    for row in rows:
        try:
            value = float(
                row.get("estimated_clearance_m", "nan")
            )
            if math.isfinite(value):
                distances.append(value)
        except (TypeError, ValueError):
            pass

    return {
        "rows": len(rows),
        "guards": sum(
            bool(r.get("guard_reason")) for r in rows
        ),
        "minimum": min(distances) if distances else None,
    }

results = []

print("\n=== PHASE 2: REGRESSION ===", flush=True)

for case, target, _, prefix in built:
    report_file = ROOT / (
        prefix + "live_autonomous_route_v3.json"
    )
    trace_file = ROOT / (
        prefix + "live_autonomous_route_v3.csv"
    )
    log_file = OUT / f"{case}.log"

    report_file.unlink(missing_ok=True)
    trace_file.unlink(missing_ok=True)

    print(f"\n--- {case} ---", flush=True)

    try:
        with log_file.open("w") as log:
            proc = subprocess.run(
                [sys.executable, str(target)],
                cwd=ROOT.parent / "dog_robot_v02",
                stdout=log,
                stderr=subprocess.STDOUT,
                timeout=360,
                check=False,
            )
        rc = proc.returncode
    except subprocess.TimeoutExpired:
        rc = 124

    report = read_json(report_file)
    trace = audit_csv(trace_file)
    text = log_file.read_text(errors="replace")

    refused_before_turn = (
        "Live turn 1: PRE_TURN_RESERVE_STOP" in text
        and trace["rows"] == 0
    )

    reached_safely = (
        rc == 0
        and report.get("pass") is True
        and report.get("status") == "PASS"
        and trace["rows"] > 0
        and trace["guards"] == 0
        and trace["minimum"] is not None
        and trace["minimum"] >= RUNTIME
    )

    # A conservative refusal is NOT a traversal PASS.
    if reached_safely:
        outcome = "ARRIVAL_PASS"
    elif rc == 0 and refused_before_turn:
        outcome = "PREFLIGHT_REFUSAL"
    else:
        outcome = "UNRESOLVED_FAIL"

    result = {
        "case": case,
        "outcome": outcome,
        "status": report.get("status"),
        "return_code": rc,
        "trace": trace,
        "preflight_refusal": refused_before_turn,
        "arrival_pass": reached_safely,
    }
    results.append(result)

    for line in text.splitlines():
        if any(key in line for key in (
            "Start clearance:",
            "LIVE_TURN1:",
            "Live turn 1:",
            "POST_TURN_CANDIDATES:",
            "GATE3_STEP3:",
            "Traceback",
            "Error:",
        )):
            print(line)

    print("RESULT:", outcome)
    print("TRACE_AUDIT:", trace)
    print("LOG:", log_file)

path = OUT / "summary.json"
path.write_text(json.dumps(results, indent=2))

print("\n=== PRE-TURN SAFETY GATE SUMMARY ===")
for item in results:
    print(item["case"], item["outcome"])

print(
    "Arrival passes:",
    sum(x["arrival_pass"] for x in results),
    "/", len(results)
)
print(
    "Preflight refusals:",
    sum(x["preflight_refusal"] for x in results)
)
print("Summary:", path)
print("PRE_TURN_GATE_EXPERIMENT: COMPLETE")
print("Full Step 4 acceptance: NOT YET")
print("Full-body safety certification: NOT DONE")
