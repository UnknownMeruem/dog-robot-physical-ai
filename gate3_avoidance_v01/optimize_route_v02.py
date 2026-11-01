from pathlib import Path
import itertools
import numpy as np
import mujoco

scene = Path("../gate3_obstacle_stop_v01/obstacle_scene.xml")
model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

def box_geometry(name):
    gid = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, name
    )
    if gid < 0:
        raise RuntimeError(f"{name} not found")

    return (
        data.geom_xpos[gid, :2].copy(),
        model.geom_size[gid, :2].copy()
    )

box_center, box_half = box_geometry("blocking_box")
poster_center, poster_half = box_geometry(
    "person_poster_board"
)

REQUIRED = 0.441
start = data.qpos[:2].copy()

def sample_path(waypoints):
    result = []
    for a, b in zip(waypoints[:-1], waypoints[1:]):
        distance = np.linalg.norm(b - a)
        n = max(2, int(np.ceil(distance / 0.005)) + 1)
        t = np.linspace(0.0, 1.0, n)[:, None]
        result.append(a + t * (b - a))
    return np.vstack(result)

def distance_from_box(points, center, half):
    delta = np.maximum(
        np.abs(points - center) - half, 0.0
    )
    return np.linalg.norm(delta, axis=1)

candidates = []

# 最初に横方向へ離れ、その後に前進する経路
for sign, x1, lane, x2, gy in itertools.product(
    (-1, 1),
    (-0.15, -0.25, -0.35),
    (0.70, 0.80, 0.90, 1.00),
    (-1.10, -1.20, -1.30),
    (0.75, 0.85, 0.95, 1.05),
):
    lane_y = sign * lane
    goal = np.array([x2, sign * gy])

    waypoints = [
        start,
        np.array([x1, lane_y]),
        np.array([x2, lane_y]),
        goal,
    ]

    target_distance = np.linalg.norm(
        goal - poster_center
    )
    if not 0.60 <= target_distance <= 1.00:
        continue

    samples = sample_path(waypoints)

    d_box = float(
        distance_from_box(
            samples, box_center, box_half
        ).min()
    )

    d_poster = float(
        distance_from_box(
            samples, poster_center, poster_half
        ).min()
    )

    if min(d_box, d_poster) < REQUIRED:
        continue

    path_length = sum(
        np.linalg.norm(b - a)
        for a, b in zip(
            waypoints[:-1], waypoints[1:]
        )
    )

    candidates.append((
        path_length, d_box, d_poster,
        target_distance, waypoints
    ))

candidates.sort(key=lambda item: item[0])

print("=== Route optimization v02 ===")
print("Required clearance:", REQUIRED, "m")
print("Valid geometric candidates:", len(candidates))

for index, result in enumerate(candidates[:5], 1):
    length, d_box, d_poster, target_d, route = result

    print(f"\nCandidate {index}")
    print("Path length:", f"{length:.3f}m")
    print("Box clearance:", f"{d_box:.3f}m")
    print("Poster clearance:", f"{d_poster:.3f}m")
    print("Distance to target:", f"{target_d:.3f}m")
    print("Waypoints:")
    for point in route:
        print(" ", np.round(point, 3))

if not candidates:
    print("No suitable candidate in the searched area.")

print("\nStatic geometric planning only.")
print("No walking or collision simulation performed.")
