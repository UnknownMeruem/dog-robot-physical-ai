from pathlib import Path
import mujoco
import numpy as np

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

# 仮のロボット外形余裕。実寸・脚軌跡の検証前。
CLEARANCE = 0.20

lower = box_xy - box_half - CLEARANCE
upper = box_xy + box_half + CLEARANCE

# 姿勢は元のモデルの初期位置を使う。
start = data.qpos[:2].copy()

# 左右の候補経路。世界座標系で指定。
routes = {
    "WORLD_Y_POSITIVE": [
        start,
        np.array([-0.40,  0.55]),
        np.array([-1.30,  0.55]),
        np.array([-1.35,  0.48]),
    ],
    "WORLD_Y_NEGATIVE": [
        start,
        np.array([-0.40, -0.55]),
        np.array([-1.30, -0.55]),
        np.array([-1.35,  0.48]),
    ],
}

def sample_route(waypoints, spacing=0.01):
    samples = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        length = np.linalg.norm(b - a)
        n = max(2, int(np.ceil(length / spacing)) + 1)
        for t in np.linspace(0, 1, n):
            samples.append(a * (1 - t) + b * t)
    return np.asarray(samples)

def clearance_to_box(points):
    # 元の箱（余裕を含まない）とのXY最短距離
    delta = np.maximum(
        np.abs(points - box_xy) - box_half,
        0.0,
    )
    return np.linalg.norm(delta, axis=1)

print("=== Obstacle route geometry ===")
print("Robot start XY:", np.round(start, 3))
print("Box center XY:", np.round(box_xy, 3))
print("Box half size XY:", np.round(box_half, 3))
print("Assumed clearance:", CLEARANCE, "m")

for name, waypoints in routes.items():
    samples = sample_route(waypoints)

    inside = np.all(
        (samples >= lower) & (samples <= upper),
        axis=1,
    )

    distances = clearance_to_box(samples)
    total_length = sum(
        np.linalg.norm(b - a)
        for a, b in zip(waypoints[:-1], waypoints[1:])
    )

    print(f"\n=== {name} ===")
    print("Path length:", f"{total_length:.3f}m")
    print("Min box XY clearance:", f"{distances.min():.3f}m")
    print("Expanded box intersection:", bool(inside.any()))
    print("Waypoints:")
    for p in waypoints:
        print(" ", np.round(p, 3))

    if inside.any():
        print("Result: REJECT")
    else:
        print("Result: GEOMETRY_CANDIDATE_ONLY")

print("\nNo walking performed.")
print("Not a verified collision-free robot trajectory.")
