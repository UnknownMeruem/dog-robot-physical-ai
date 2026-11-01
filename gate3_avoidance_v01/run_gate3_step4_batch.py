from pathlib import Path
import xml.etree.ElementTree as ET
import subprocess
import json
import sys
import re

root = Path(__file__).resolve().parent
scene = root.parent / "gate3_obstacle_stop_v01" / "obstacle_scene_extra_box.xml"
source = root / "test_live_autonomous_route_v3.py"
result_dir = root / "step4_batch_results"
result_dir.mkdir(exist_ok=True)

if not scene.is_file() or not source.is_file():
    raise SystemExit("Missing scene or successful v3 source")

source_code = source.read_text()

# Only the obstacle layout changes.
# Values are offsets relative to the existing extra-box position.
cases = [
    ("baseline", 0.00, 0.00),
    ("box_y_plus_10cm", 0.00, 0.10),
    ("box_y_minus_10cm", 0.00, -0.10),
]

# Check the source replacement points before generating anything.
model_line = "model = mujoco.MjModel.from_xml_path(str(scene_path))"
if source_code.count(model_line) != 1:
    raise SystemExit("Unexpected model-loading structure")

expected_outputs = [
    "waypoint1_early_live_v3.csv",
    "depth_early_live_v3.csv",
    "early_live_replan_result_v3_early.json",
    "live_obstacle_points_v3.csv",
    "live_autonomous_route_v3.csv",
    "live_autonomous_route_v3.json",
]

for filename in expected_outputs:
    if filename not in source_code:
        raise SystemExit("Missing output reference: " + filename)

summary = []

for case, dx, dy in cases:
    xml_tree = ET.parse(scene)
    xml_root = xml_tree.getroot()

    geom = next(
        (g for g in xml_root.iter("geom")
         if g.get("name") == "extra_blocking_box"),
        None
    )
    if geom is None:
        raise SystemExit("extra_blocking_box not found")

    original_pos = [
        float(x) for x in geom.get("pos", "0 0 0").split()
    ]
    if len(original_pos) != 3:
        raise SystemExit("Invalid obstacle position")

    modified_pos = original_pos.copy()
    modified_pos[0] += dx
    modified_pos[1] += dy
    geom.set("pos", " ".join(f"{v:.8f}" for v in modified_pos))

    # Place the generated XML beside the original so relative
    # asset/include paths preserve their behavior.
    case_scene = scene.parent / f"step4_{case}.xml"
    xml_tree.write(case_scene, encoding="unicode")

    test_path = root / f"test_step4_{case}.py"
    code = source_code.replace(
        model_line,
        'model = mujoco.MjModel.from_xml_path('
        f'str(Path(r"{case_scene}")))',
        1
    )

    for filename in expected_outputs:
        stem, suffix = filename.rsplit(".", 1)
        code = code.replace(
            f'"{filename}"',
            f'"step4_{case}_{stem}.{suffix}"'
        )

    # Prevent overwriting previous reports.
    code = code.replace(
        '"GATE3_STEP3_V3:"',
        f'"GATE3_STEP4_{case}:"'
    )

    compile(code, str(test_path), "exec")
    test_path.write_text(code)

    log_path = result_dir / f"{case}.log"
    print(f"\n=== STEP 4 CASE: {case} ===", flush=True)
    print(f"Obstacle shift: dx={dx:+.2f}, dy={dy:+.2f} m",
          flush=True)

    with log_path.open("w") as log:
        process = subprocess.run(
            [sys.executable, str(test_path)],
            cwd=root.parent / "dog_robot_v02",
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=300,
            check=False,
        )

    report = root / f"step4_{case}_live_autonomous_route_v3.json"
    outcome = {
        "case": case,
        "dx_m": dx,
        "dy_m": dy,
        "exit_code": process.returncode,
        "reported_pass": False,
        "final_xy": None,
    }

    if report.exists():
        try:
            data = json.loads(report.read_text())
            outcome["reported_pass"] = bool(data.get("pass", False))
            outcome["final_xy"] = data.get("final_xy")
            outcome["trace_samples"] = data.get("trace_samples")
        except (ValueError, OSError) as exc:
            outcome["report_error"] = str(exc)

    outcome["pass"] = (
        process.returncode == 0
        and outcome["reported_pass"]
    )
    summary.append(outcome)

    lines = log_path.read_text(errors="replace").splitlines()
    interesting = [
        line for line in lines
        if any(word in line for word in (
            "LIVE_REPLAN:",
            "POST_TURN_CANDIDATES:",
            "POST_TURN_PREDICTED_CLEARANCE:",
            "Live waypoint walk:",
            "Live final walk:",
            "GATE3_STEP3:",
            "Traceback",
            "Error:",
        ))
    ]
    for line in interesting[-12:]:
        print(line, flush=True)

    print("CASE_RESULT:", "PASS" if outcome["pass"] else "FAIL",
          flush=True)
    print("Log:", log_path, flush=True)

output = result_dir / "step4_summary.json"
output.write_text(json.dumps(summary, indent=2))

print("\n=== GATE 3 STEP 4 BATCH SUMMARY ===")
for item in summary:
    print(item["case"],
          "PASS" if item["pass"] else "FAIL")

print("Passed:", sum(x["pass"] for x in summary), "/", len(summary))
print("Summary:", output)
print("STEP4_INITIAL_BATCH:",
      "PASS" if all(x["pass"] for x in summary) else "FAIL")
print("Initial batch only; not full Step 4 acceptance.")
