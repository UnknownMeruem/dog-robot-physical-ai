from pathlib import Path
import ast
import json
import re
import subprocess
import sys
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent
scene = root.parent / "gate3_obstacle_stop_v01" / "obstacle_scene_extra_box.xml"
template = root / "test_step4_goal_policy_v2_baseline.py"
out = root / "step4_robustness_results"
out.mkdir(exist_ok=True)

if not template.is_file() or not scene.is_file():
    raise SystemExit("Required verified source or scene missing")

base = template.read_text()

# A missing or non-finite camera map must never authorize travel.
old = "goal_policy_unsafe = False"
new = "goal_policy_unsafe = not goal_policy_valid"
if base.count(old) != 1:
    raise SystemExit("Unexpected goal-policy structure")
base = base.replace(old, new, 1)

# Classify invalid maps distinctly before writing the report.
old = 'if goal_policy_unsafe:\n    if ('
new = '''if not goal_policy_valid:
    stage3_reason = "GOAL_MAP_INVALID_REJECTED"
elif goal_policy_unsafe:
    if ('''
if base.count(old) != 1:
    raise SystemExit("Unexpected report-policy structure")
base = base.replace(old, new, 1)

# Five new deterministic layouts, preserving the original box sizes.
cases = [
    ("x_minus_05", -0.05, 0.00),
    ("x_plus_05", 0.05, 0.00),
    ("y_minus_05", 0.00, -0.05),
    ("y_plus_05", 0.00, 0.05),
    ("xy_shift", 0.05, 0.05),
]

# Additional fault: deliberately corrupt the camera-derived map
# after observation but BEFORE goal authorization.
fault_marker = (
    "# ===== Goal policy: evaluated before "
    "route-control execution ====="
)
if base.count(fault_marker) != 1:
    raise SystemExit("Fault injection location not unique")

loader_pattern = re.compile(
    r'model = mujoco\.MjModel\.from_xml_path\('
    r'str\(Path\(r"[^"]+"\)\)\)'
)
if len(loader_pattern.findall(base)) != 1:
    raise SystemExit("Scene loader not unique")

prefix_old = "step4_policy_v2_baseline_"
if prefix_old not in base:
    raise SystemExit("Expected output prefix missing")

built = []

def build(case, dx, dy, inject_fault=False):
    tree = ET.parse(scene)
    geom = next(
        (g for g in tree.getroot().iter("geom")
         if g.get("name") == "extra_blocking_box"),
        None
    )
    if geom is None:
        raise RuntimeError("Extra box not found")

    pos = [float(v) for v in geom.get("pos").split()]
    if len(pos) != 3:
        raise RuntimeError("Invalid box position")
    pos[0] += dx
    pos[1] += dy
    geom.set("pos", " ".join(f"{v:.8f}" for v in pos))

    # Keep XML alongside the original to preserve relative assets.
    case_scene = scene.parent / f"step4_robust_{case}.xml"
    tree.write(case_scene, encoding="unicode")

    code = loader_pattern.sub(
        lambda m:
        f'model = mujoco.MjModel.from_xml_path('
        f'str(Path(r"{case_scene}")))',
        base,
        count=1
    )

    # Rename quoted output paths only, not any scene paths.
    prefix_new = f"step4_robust_{case}_"
    code = code.replace(
        f'"{prefix_old}',
        f'"{prefix_new}'
    )

    if inject_fault:
        code = code.replace(
            fault_marker,
            '''# Controlled sensor-map corruption test.
if reason2 == "DEPTH_OBSTACLE_STOP":
    live_report["estimated_box_min"] = [
        float("nan"), float("nan")
    ]
''' + fault_marker,
            1
        )

    target = root / f"test_step4_robust_{case}.py"
    ast.parse(code, filename=str(target))
    compile(code, str(target), "exec")
    target.write_text(code)
    built.append((case, target, inject_fault))

for case, dx, dy in cases:
    build(case, dx, dy)

build("nan_map_fault", 0.0, 0.0, inject_fault=True)

print("ALL_6_CASES_COMPILE: PASS", flush=True)

results = []

for case, script, fault in built:
    prefix = f"step4_robust_{case}_"
    run_file = root / (prefix+"live_autonomous_route_v3.json")
    live_file = root / (
        prefix+"early_live_replan_result_v3_early.json"
    )
    log_file = out / (case+".log")

    # Stale reports must never count as fresh results.
    for path in (run_file, live_file):
        path.unlink(missing_ok=True)

    print(f"\n=== {case} ===", flush=True)
    try:
        with log_file.open("w") as f:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=root.parent / "dog_robot_v02",
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=360,
                check=False,
            )
        returncode = proc.returncode
    except subprocess.TimeoutExpired:
        returncode = 124

    def read(path):
        try:
            return json.loads(path.read_text())
        except (OSError, ValueError):
            return {}

    report = read(run_file)
    live = read(live_file)

    status = report.get("status")
    observed = live.get("observed_points", 0)
    samples = report.get("trace_samples", 0)

    if fault:
        expected = "GOAL_MAP_INVALID_REJECTED"
        passed = (
            returncode == 0
            and status == expected
            and report.get("pass") is False
            and samples == 0
            and observed >= 30
        )
    else:
        # Either a safe arrival or a validated refusal is valid.
        expected = "SAFE_ARRIVAL_OR_VALIDATED_REFUSAL"
        allowed = {
            "PASS",
            "GOAL_UNSAFE_REJECTED",
        }
        passed = (
            returncode == 0
            and status in allowed
            and observed >= 30
            and (
                (
                    status == "PASS"
                    and report.get("pass") is True
                    and samples > 0
                )
                or (
                    status == "GOAL_UNSAFE_REJECTED"
                    and report.get("pass") is False
                    and samples == 0
                )
            )
        )

    # Show a compact subset; keep the complete log on disk.
    for line in log_file.read_text(
        errors="replace"
    ).splitlines():
        if any(k in line for k in (
            "GOAL_POLICY_CLEARANCE:",
            "GOAL_POLICY_UNSAFE:",
            "GATE3_STEP3:",
            "Traceback",
            "SyntaxError:",
        )):
            print(line)

    item = {
        "case": case,
        "expected": expected,
        "status": status,
        "pass": bool(passed),
        "returncode": returncode,
        "observed_points": observed,
        "trace_samples": samples,
    }
    results.append(item)
    print("CASE:", "PASS" if passed else "FAIL")
    print("Status:", status)
    print("Log:", log_file)

summary = out / "summary.json"
summary.write_text(json.dumps(results, indent=2))

print("\n=== STEP 4 ROBUSTNESS SUMMARY ===")
for r in results:
    print(r["case"], r["status"],
          "PASS" if r["pass"] else "FAIL")

print("Passed:", sum(r["pass"] for r in results),
      "/", len(results))
print(
    "STEP4_ROBUSTNESS_BATCH:",
    "PASS" if all(r["pass"] for r in results) else "FAIL"
)
print("Summary:", summary)
print("Still required: multiple obstacles, dynamic re-observation,")
print("full-body contact and posture auditing.")
