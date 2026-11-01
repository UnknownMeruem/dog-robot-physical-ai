from pathlib import Path
import json
import subprocess
import sys
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parent
scene = root.parent / "gate3_obstacle_stop_v01" / "obstacle_scene_extra_box.xml"
source = root / "test_live_autonomous_route_v3.py"
out = root / "step4_expanded_results"
out.mkdir(exist_ok=True)

cases = [
    ("baseline", 0.00),
    ("plus_10cm", 0.10),
    ("minus_10cm", -0.10),
]

print("=== PREVIOUS FAILURE ANALYSIS ===")
previous = root / "step4_batch_results/box_y_plus_10cm.log"
if previous.exists():
    for line in previous.read_text(errors="replace").splitlines():
        if any(key in line for key in [
            "Depth t=", "Obstacle points:",
            "Robot XY:", "Start clearance:",
            "Static route candidates:",
            "LIVE_REPLAN:", "GATE3_STEP3:"
        ]):
            print(line)

code = source.read_text()

# Widen the search only. Keep the original guard and
# safety thresholds unchanged.
changes = {
    "np.arange(1.25,1.86,0.05)":
        "np.arange(1.00,2.31,0.05)",
    "np.arange(-0.45,0.21,0.05)":
        "np.arange(-0.85,0.81,0.05)",
    "np.arange(1.35, 1.81, 0.05)":
        "np.arange(1.00, 2.31, 0.05)",
    "np.arange(-0.45, 0.31, 0.05)":
        "np.arange(-0.85, 0.81, 0.05)",
}

for old, new in changes.items():
    if code.count(old) != 1:
        raise RuntimeError(
            f"Planner replacement mismatch: {old} "
            f"count={code.count(old)}"
        )
    code = code.replace(old, new, 1)

if "PLANNING_CLEARANCE = 0.551" not in code:
    raise RuntimeError("Planning margin was not preserved")
if "CLEARANCE = 0.521" not in code:
    raise RuntimeError("Runtime clearance was not preserved")

loader = "model = mujoco.MjModel.from_xml_path(str(scene_path))"
if code.count(loader) != 1:
    raise RuntimeError("Unexpected scene loader")

outputs = [
    "waypoint1_early_live_v3.csv",
    "depth_early_live_v3.csv",
    "early_live_replan_result_v3_early.json",
    "live_obstacle_points_v3.csv",
    "live_autonomous_route_v3.csv",
    "live_autonomous_route_v3.json",
]

for filename in outputs:
    if f'"{filename}"' not in code:
        raise RuntimeError("Output name not found: " + filename)

results = []

for case, dy in cases:
    xml = ET.parse(scene)
    geom = next(
        (g for g in xml.getroot().iter("geom")
         if g.get("name") == "extra_blocking_box"),
        None
    )
    if geom is None:
        raise RuntimeError("Extra obstacle not found")

    pos = [float(v) for v in geom.get("pos").split()]
    pos[1] += dy
    geom.set("pos", " ".join(f"{v:.8f}" for v in pos))

    case_scene = scene.parent / f"step4_expanded_{case}.xml"
    xml.write(case_scene, encoding="unicode")

    variant = code.replace(
        loader,
        f'model = mujoco.MjModel.from_xml_path(str(Path(r"{case_scene}")))',
        1
    )

    for filename in outputs:
        new_name = f"step4_expanded_{case}_{filename}"
        variant = variant.replace(
            f'"{filename}"', f'"{new_name}"'
        )

    script = root / f"test_step4_expanded_{case}.py"
    compile(variant, str(script), "exec")
    script.write_text(variant)

    log = out / f"{case}.log"
    print(f"\n=== Expanded search: {case} ===", flush=True)

    with log.open("w") as f:
        try:
            proc = subprocess.run(
                [sys.executable, str(script)],
                cwd=root.parent / "dog_robot_v02",
                stdout=f,
                stderr=subprocess.STDOUT,
                timeout=360,
            )
            return_code = proc.returncode
        except subprocess.TimeoutExpired:
            return_code = 124

    report_file = root / (
        f"step4_expanded_{case}_live_autonomous_route_v3.json"
    )

    report = {}
    if report_file.exists():
        try:
            report = json.loads(report_file.read_text())
        except (OSError, ValueError):
            pass

    passed = (
        return_code == 0
        and report.get("pass") is True
    )

    lines = log.read_text(errors="replace").splitlines()
    important = [
        s for s in lines
        if any(k in s for k in (
            "Obstacle points:",
            "Static route candidates:",
            "LIVE_REPLAN:",
            "Candidates with 3cm",
            "POST_TURN_CANDIDATES:",
            "POST_TURN_PREDICTED_CLEARANCE:",
            "Live waypoint walk:",
            "Live final walk:",
            "GATE3_STEP3:",
            "Traceback",
            "Error:",
        ))
    ]
    for item in important[-18:]:
        print(item)

    print("CASE:", case, "PASS" if passed else "FAIL",
          flush=True)
    results.append({
        "case": case,
        "shift_y_m": dy,
        "pass": passed,
        "return_code": return_code,
        "report": report,
    })

summary = out / "summary.json"
summary.write_text(json.dumps(results, indent=2))

print("\n=== STEP 4 EXPANDED SUMMARY ===")
for item in results:
    print(item["case"], "PASS" if item["pass"] else "FAIL")
print("Passed:", sum(x["pass"] for x in results),
      "/", len(results))
print("Summary:", summary)
print("STEP4_EXPANDED_BATCH:",
      "PASS" if all(x["pass"] for x in results) else "FAIL")
print("This is not yet full Step 4 acceptance.")
