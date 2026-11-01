from pathlib import Path
import numpy as np
import mujoco

scene = Path("../gate3_obstacle_stop_v01/obstacle_scene.xml")
model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

torso_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "torso"
)
box_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "blocking_box"
)

if torso_id < 0 or box_id < 0:
    raise RuntimeError("torso または blocking_box が見つかりません")

def belongs_to_torso(body_id):
    while body_id > 0:
        if body_id == torso_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return False

torso_xy = data.xpos[torso_id, :2].copy()
radius = 0.0
largest = []

for gid in range(model.ngeom):
    body_id = int(model.geom_bodyid[gid])
    if not belongs_to_torso(body_id):
        continue

    name = mujoco.mj_id2name(
        model, mujoco.mjtObj.mjOBJ_GEOM, gid
    ) or f"geom_{gid}"

    # 形状の外接球を利用する保守的な概算
    center_dist = np.linalg.norm(
        data.geom_xpos[gid, :2] - torso_xy
    )
    extent = float(model.geom_rbound[gid])
    estimate = center_dist + extent
    radius = max(radius, estimate)
    largest.append((estimate, name))

# 推定誤差・動作による余裕。実機の保証値ではない
MARGIN = 0.05
effective_radius = radius + MARGIN

box_center = data.geom_xpos[box_id, :2].copy()
box_half = model.geom_size[box_id, :2].copy()

start = data.qpos[:2].copy()
routes = {
    "WORLD_Y_POSITIVE": [
        start,
        np.array([-0.40, 0.55]),
        np.array([-1.30, 0.55]),
        np.array([-1.35, 0.48]),
    ],
    "WORLD_Y_NEGATIVE": [
        start,
        np.array([-0.40, -0.55]),
        np.array([-1.30, -0.55]),
        np.array([-1.35, 0.48]),
    ],
}

def sample_path(waypoints, spacing=0.005):
    result = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        distance = np.linalg.norm(b - a)
        count = max(2, int(np.ceil(distance / spacing)) + 1)
        for t in np.linspace(0.0, 1.0, count):
            result.append(a * (1 - t) + b * t)
    return np.asarray(result)

def distance_to_box(points):
    delta = np.maximum(
        np.abs(points - box_center) - box_half,
        0.0,
    )
    return np.linalg.norm(delta, axis=1)

print("=== Robot geometric envelope ===")
print("Torso XY:", np.round(torso_xy, 3))
print("Estimated radius:", f"{radius:.3f}m")
print("Additional margin:", f"{MARGIN:.3f}m")
print("Required clearance:", f"{effective_radius:.3f}m")

print("\nLargest contributing geoms:")
for extent, name in sorted(largest, reverse=True)[:5]:
    print(f"  {name}: {extent:.3f}m")

print("\n=== Route comparison ===")
for name, waypoints in routes.items():
    samples = sample_path(waypoints)
    distances = distance_to_box(samples)

    min_clearance = float(distances.min())
    remaining = min_clearance - effective_radius

    print(f"\n{name}")
    print("Min box distance:", f"{min_clearance:.3f}m")
    print("Remaining clearance:", f"{remaining:.3f}m")

    if remaining <= 0:
        print("Result: REJECT")
    else:
        print("Result: STATIC_ENVELOPE_CANDIDATE")

print("\nNo movement performed.")
print("This is not dynamic collision verification.")
