from pathlib import Path
import numpy as np
import mujoco

scene = Path("../gate3_obstacle_stop_v01/obstacle_scene.xml")
model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

gid = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "blocking_box"
)
if gid < 0:
    raise RuntimeError("blocking_box not found")

box_xy = data.geom_xpos[gid, :2].copy()
box_half = model.geom_size[gid, :2].copy()

# 前回測定した概算外形半径と追加余裕
ROBOT_RADIUS = 0.291
BASE_MARGIN = 0.050
EXTRA_MARGIN = 0.100

REQUIRED_CLEARANCE = (
    ROBOT_RADIUS + BASE_MARGIN + EXTRA_MARGIN
)

start = data.qpos[:2].copy()

# 人物看板の手前で終わる、暫定的な目標点
goal = np.array([-1.25, 0.48])

def sample_path(waypoints):
    samples = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        length = np.linalg.norm(b - a)
        count = max(2, int(np.ceil(length / 0.005)) + 1)
        for t in np.linspace(0, 1, count):
            samples.append(a * (1 - t) + b * t)
    return np.asarray(samples)

def box_clearance(points):
    offset = np.maximum(
        np.abs(points - box_xy) - box_half,
        0.0,
    )
    return np.linalg.norm(offset, axis=1)

print("=== Route optimization ===")
print("Required XY clearance:", f"{REQUIRED_CLEARANCE:.3f}m")
print("Goal:", goal)

valid_routes = []

for lane_y in np.arange(0.55, 1.01, 0.05):
    waypoints = [
        start,
        np.array([-0.40, lane_y]),
        np.array([-1.30, lane_y]),
        goal,
    ]

    samples = sample_path(waypoints)
    min_distance = box_clearance(samples).min()
    remaining = min_distance - REQUIRED_CLEARANCE

    path_length = sum(
        np.linalg.norm(b - a)
        for a, b in zip(waypoints[:-1], waypoints[1:])
    )

    passed = remaining >= 0
    print(
        f"lane_y={lane_y:.2f}m "
        f"length={path_length:.3f}m "
        f"clearance={min_distance:.3f}m "
        f"reserve={remaining:+.3f}m "
        f"{'CANDIDATE' if passed else 'REJECT'}"
    )

    if passed:
        valid_routes.append((
            path_length, lane_y, remaining, waypoints
        ))

if not valid_routes:
    print("\nNo route satisfies the requested clearance.")
else:
    valid_routes.sort(key=lambda x: x[0])
    length, lane_y, reserve, waypoints = valid_routes[0]

    print("\n=== Preferred geometric candidate ===")
    print("Lane Y:", f"{lane_y:.2f}m")
    print("Path length:", f"{length:.3f}m")
    print("Reserve beyond required clearance:", f"{reserve:.3f}m")
    print("Waypoints:")
    for point in waypoints:
        print(" ", np.round(point, 3))

print("\nStatic XY analysis only.")
print("Not permission to execute the route.")
