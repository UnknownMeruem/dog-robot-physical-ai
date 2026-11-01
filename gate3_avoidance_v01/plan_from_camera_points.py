from pathlib import Path
import numpy as np

ROOT = Path("../gate3_avoidance_v01")
points = np.loadtxt(
    ROOT / "detected_obstacle_world_points.csv",
    delimiter=",", skiprows=1,
)
points = np.atleast_2d(points)

if points.shape[1] != 3 or not np.isfinite(points).all():
    raise RuntimeError("Invalid camera point cloud")
if len(points) < 30:
    raise RuntimeError("Too few obstacle points")

# Observed surface bounds: do not use the ground-truth box center.
observed_min = points[:, :2].min(axis=0)
observed_max = points[:, :2].max(axis=0)

# Conservative unobserved-side padding for this exploratory scene.
# The farther (-X) side is less visible from the stopped robot.
estimate_min = observed_min - np.array([0.15, 0.05])
estimate_max = observed_max + np.array([0.01, 0.05])

# Actual position where the camera detected the obstacle.
start = np.array([-0.3750, 0.8053])
goal = np.array([-1.10, 1.50])

REQUIRED = 0.441
TURN_RESERVE = 0.080

def clearance(xy):
    xy = np.asarray(xy)
    delta = np.maximum(
        np.maximum(estimate_min - xy, xy - estimate_max),
        0.0,
    )
    return np.linalg.norm(delta, axis=-1)

def sample_segment(a, b):
    length = np.linalg.norm(b - a)
    steps = max(2, int(np.ceil(length / 0.005)) + 1)
    t = np.linspace(0.0, 1.0, steps)[:, None]
    return a + t * (b - a)

candidates = []
for y in np.arange(1.25, 1.81, 0.05):
    for x in np.arange(-0.45, 0.16, 0.05):
        via = np.array([x, y])
        samples = np.vstack([
            sample_segment(start, via),
            sample_segment(via, goal),
        ])
        minimum = float(clearance(samples).min())
        length = (
            np.linalg.norm(via - start)
            + np.linalg.norm(goal - via)
        )
        if minimum >= REQUIRED:
            candidates.append((length, minimum, via))

print("=== Camera-Only New Obstacle Planning ===")
print("Camera points:", len(points))
print("Observed min XY:", np.round(observed_min, 4))
print("Observed max XY:", np.round(observed_max, 4))
print("Conservative min XY:", np.round(estimate_min, 4))
print("Conservative max XY:", np.round(estimate_max, 4))
print("Planning start XY:", start)
print("Goal XY:", goal)

print("\n=== Route Search ===")
print("Collision-free static candidates:", len(candidates))

if candidates:
    # Prefer a candidate with useful reserve, not simply
    # the shortest mathematically valid line.
    candidates.sort(
        key=lambda c: (c[1] < REQUIRED + 0.05, c[0])
    )
    length, minimum, via = candidates[0]
    print("Chosen intermediate XY:", np.round(via, 4))
    print("Planned length:", f"{length:.3f}m")
    print("Minimum estimated clearance:", f"{minimum:.3f}m")
    print("STATIC_CAMERA_ROUTE: PASS")
else:
    print("STATIC_CAMERA_ROUTE: REJECT")

print("\n=== Turning Start Check ===")
start_clearance = float(clearance(start))
start_reserve = start_clearance - REQUIRED

print("Estimated starting clearance:",
      f"{start_clearance:.3f}m")
print("Reserve:", f"{start_reserve:+.3f}m")
print("Required reserve for turning:",
      f"{TURN_RESERVE:.3f}m")

if start_reserve >= TURN_RESERVE:
    print("TURN_START_MARGIN: CANDIDATE")
else:
    print("TURN_START_MARGIN: INSUFFICIENT")
    print("Action: stop earlier or re-observe and replan")

print("\nNo robot motion performed.")
print("Estimated obstacle extent is not ground truth.")
print("No automatic driving authorization.")
