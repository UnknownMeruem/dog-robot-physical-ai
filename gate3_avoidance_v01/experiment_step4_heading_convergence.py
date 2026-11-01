from pathlib import Path
import ast
import json
import subprocess
import sys

root = Path(__file__).resolve().parent
source = root / "test_step4_robust_x_plus_05.py"
target = root / "test_step4_heading_convergence_x_plus_05.py"
out = root / "step4_heading_convergence"
out.mkdir(exist_ok=True)

if not source.is_file():
    raise SystemExit("Missing X+5cm regression source")

code = source.read_text()
tree = ast.parse(code)

# Confirm we are modifying precisely the post-turn correction loop.
loops = [
    n for n in ast.walk(tree)
    if isinstance(n, ast.For)
    and isinstance(n.target, ast.Name)
    and n.target.id == "correction_iteration"
]
if len(loops) != 1:
    raise SystemExit(
        f"Unexpected correction loops: {len(loops)}"
    )

loop = loops[0]
if not (
    isinstance(loop.iter, ast.Call)
    and isinstance(loop.iter.func, ast.Name)
    and loop.iter.func.id == "range"
    and len(loop.iter.args) == 1
    and isinstance(loop.iter.args[0], ast.Constant)
    and loop.iter.args[0].value == 3
):
    raise SystemExit("Correction loop does not match expected v3")

# Modify only the verified range(3) at this AST location.
lines = code.splitlines(keepends=True)
index = loop.lineno - 1
line = lines[index]
if line.count("range(3)") != 1:
    raise SystemExit("Cannot safely patch correction limit")

lines[index] = line.replace("range(3)", "range(6)", 1)
modified = "".join(lines)

# Leave all existing safety thresholds unchanged.
for constant in (
    "CLEARANCE = 0.521",
    "PLANNING_CLEARANCE = 0.551",
):
    if constant not in modified:
        raise SystemExit("Required safety constant missing: " + constant)

# Keep results isolated from all preceding successful tests.
old_prefix = "step4_robust_x_plus_05_"
new_prefix = "step4_heading_x_plus_05_"
if old_prefix not in modified:
    raise SystemExit("Expected result filenames missing")

# Replace only quoted result names, not the scene path.
modified = modified.replace(
    '"' + old_prefix,
    '"' + new_prefix
)

ast.parse(modified, filename=str(target))
compile(modified, str(target), "exec")
target.write_text(modified)

report_path = root / (
    new_prefix + "live_autonomous_route_v3.json"
)
report_path.unlink(missing_ok=True)

log_path = out / "x_plus_05.log"

print("=== STEP 4 HEADING CONVERGENCE EXPERIMENT ===", flush=True)
print("Correction iterations: 3 -> 6", flush=True)
print("Runtime clearance: 0.521m, unchanged", flush=True)
print("Planning clearance: 0.551m, unchanged", flush=True)
print("Original sources: unchanged", flush=True)

try:
    with log_path.open("w") as f:
        proc = subprocess.run(
            [sys.executable, str(target)],
            cwd=root.parent / "dog_robot_v02",
            stdout=f,
            stderr=subprocess.STDOUT,
            timeout=360,
            check=False,
        )
    returncode = proc.returncode
except subprocess.TimeoutExpired:
    returncode = 124

try:
    report = json.loads(report_path.read_text())
except (OSError, ValueError):
    report = {}

lines = log_path.read_text(
    errors="replace"
).splitlines()

print("\n=== CONTROLLER EVENTS ===")
for line in lines:
    if any(word in line for word in (
        "POST_TURN_CANDIDATES:",
        "POST_TURN_HEADING_CORRECTION:",
        "LIVE_CORRECTIVE_TURN:",
        "POST_TURN_PREDICTED_CLEARANCE:",
        "Live waypoint walk:",
        "Live final walk:",
        "GATE3_STEP3:",
        "First guard violation:",
        "Traceback",
        "Error:",
    )):
        print(line)

passed = (
    returncode == 0
    and report.get("status") == "PASS"
    and report.get("pass") is True
)

summary = {
    "case": "x_plus_05",
    "correction_limit": 6,
    "returncode": returncode,
    "status": report.get("status", "NO_REPORT"),
    "reported_pass": report.get("pass"),
    "experimental_pass": passed,
    "not_full_step4_acceptance": True,
}

summary_path = out / "summary.json"
summary_path.write_text(json.dumps(summary, indent=2))

print("\n=== EXPERIMENT RESULT ===")
print("Status:", summary["status"])
print("HEADING_CONVERGENCE_EXPERIMENT:",
      "PASS" if passed else "FAIL")
print("Log:", log_path)
print("Summary:", summary_path)
print("This experiment is not yet Step 4 acceptance.")
