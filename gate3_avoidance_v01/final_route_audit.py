from pathlib import Path
import csv
import numpy as np
import mujoco

root = Path("../gate3_avoidance_v01")
scene = Path("../gate3_obstacle_stop_v01/obstacle_scene.xml")

model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

def geometry(name):
    gid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, name
    )
    if gid < 0:
        raise RuntimeError(f"Missing geom: {name}")
    return (
        data.geom_xpos[gid, :2].copy(),
        model.geom_size[gid, :2].copy(),
    )

def read_csv(filename):
    with (root / filename).open(newline="") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise RuntimeError(f"Empty CSV: {filename}")
    return rows

segments = {
    "waypoint1": read_csv("waypoint1_braking.csv"),
    "waypoint2": read_csv("stage2_waypoint.csv"),
}

all_rows = segments["waypoint1"] + segments["waypoint2"]

xy = np.array([
    [float(r["x_m"]), float(r["y_m"])]
    for r in all_rows
])

def min_clearance(name):
    center, half = geometry(name)
    delta = np.maximum(np.abs(xy - center) - half, 0.0)
    return float(np.linalg.norm(delta, axis=1).min())

final = segments["waypoint2"][-1]
final_xy = np.array([
    float(final["x_m"]),
    float(final["y_m"]),
])

goal = np.array([-1.10, 0.85])
goal_error = float(np.linalg.norm(final_xy - goal))

min_height = min(float(r["height_m"]) for r in all_rows)
max_tilt = max(float(r["tilt_deg"]) for r in all_rows)

box_contacts = max(
    int(float(r["box_contacts_total"]))
    for r in all_rows
)

clearance_box = min_clearance("blocking_box")
clearance_poster = min_clearance("person_poster_board")
required = 0.441

checks = {
    "FINAL_GOAL": goal_error <= 0.08,
    "FINAL_HOLD": final["stage"] == "hold",
    "POSTURE_HEIGHT": min_height >= 0.16,
    "POSTURE_TILT": max_tilt <= 12.0,
    "NO_BOX_CONTACT": box_contacts == 0,
    "BOX_CLEARANCE": clearance_box >= required,
    "POSTER_CLEARANCE": clearance_poster >= required,
}

print("=== Gate 3 Final Route Audit ===")
print("Final XY:", np.round(final_xy, 4))
print("Goal XY:", goal)
print("Final goal error:", f"{goal_error:.4f}m")
print("Minimum torso height:", f"{min_height:.4f}m")
print("Maximum tilt:", f"{max_tilt:.2f}deg")
print("Box contact samples:", box_contacts)
print("Minimum box XY clearance:", f"{clearance_box:.3f}m")
print("Minimum poster XY clearance:", f"{clearance_poster:.3f}m")
print("Recorded samples:", len(all_rows))

print("\n=== Acceptance Checks ===")
for name, passed in checks.items():
    print(f"{name}: {'PASS' if passed else 'FAIL'}")

print("\n=== Result ===")
if all(checks.values()):
    print("PLANNED_ROUTE_SIMULATION_PASS")
else:
    print("ROUTE_REVIEW_REQUIRED")

print("\nSimulation logs and approximate XY geometry only.")
print("Not proof of autonomous replanning or hardware safety.")
