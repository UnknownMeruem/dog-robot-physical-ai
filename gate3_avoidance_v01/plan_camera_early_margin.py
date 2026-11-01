from pathlib import Path
import csv
import numpy as np

root = Path("../gate3_avoidance_v01")
points = np.loadtxt(
    root / "detected_obstacle_world_points.csv",
    delimiter=",", skiprows=1
)
points = np.atleast_2d(points)

with (root / "waypoint1_camera_early_margin.csv").open() as f:
    rows = list(csv.DictReader(f))

last = rows[-1]
start = np.array([
    float(last["x_m"]),
    float(last["y_m"])
])
goal = np.array([-1.10, 1.50])

if (
    points.shape[1] != 3
    or len(points) < 30
    or not np.isfinite(points).all()
):
    raise RuntimeError("Invalid camera point cloud")

observed_min = points[:, :2].min(axis=0)
observed_max = points[:, :2].max(axis=0)

# Trial-only uncertainty allowance for unseen obstacle surfaces.
box_min = observed_min - np.array([0.15, 0.05])
box_max = observed_max + np.array([0.01, 0.05])

required = 0.441
reserve = 0.080
target = required + reserve

def clearance(xy):
    xy = np.asarray(xy)
    delta = np.maximum(
        np.maximum(box_min - xy, xy - box_max),
        0.0
    )
    return np.linalg.norm(delta, axis=-1)

def sample(a, b):
    n = max(2, int(np.ceil(np.linalg.norm(b-a)/0.005))+1)
    t = np.linspace(0, 1, n)[:, None]
    return a + t*(b-a)

candidates = []
for y in np.arange(1.20, 1.86, 0.05):
    for x in np.arange(-0.45, 0.21, 0.05):
        via = np.array([x, y])
        path = np.vstack([
            sample(start, via),
            sample(via, goal)
        ])
        c = float(clearance(path).min())
        length = (
            np.linalg.norm(via-start)
            + np.linalg.norm(goal-via)
        )
        if c >= target:
            candidates.append((length, c, via))

start_clearance = float(clearance(start))

print("=== Camera Planner v02 ===")
print("Early stop XY:", np.round(start, 4))
print("Camera-based box min:", np.round(box_min, 4))
print("Camera-based box max:", np.round(box_max, 4))
print("Required including reserve:", f"{target:.3f}m")
print("Start clearance:", f"{start_clearance:.3f}m")
print("Accepted route candidates:", len(candidates))

if start_clearance < target:
    print("TURN_START: REJECT")
else:
    print("TURN_START: STATIC_CANDIDATE")

if not candidates:
    print("CAMERA_PLANNER_V02: REJECT")
else:
    length, margin, via = min(candidates, key=lambda r:r[0])
    print("Generated waypoint:", np.round(via, 4))
    print("Route length:", f"{length:.3f}m")
    print("Minimum estimated clearance:", f"{margin:.3f}m")
    if start_clearance >= target:
        print("CAMERA_PLANNER_V02: STATIC_PASS")
    else:
        print("CAMERA_PLANNER_V02: START_REJECT")

print("No robot movement or dynamic safety validation performed.")
