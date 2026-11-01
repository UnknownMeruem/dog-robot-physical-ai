from pathlib import Path
import csv
import math
import numpy as np

root = Path("../gate3_avoidance_v01")

def load_csv(name):
    with (root / name).open(newline="") as f:
        return list(csv.DictReader(f))

turn = load_csv("camera_full_turn.csv")
walk = load_csv("camera_full_walk.csv")

last = turn[-1]
start = np.array([
    float(last["x_m"]),
    float(last["y_m"])
])
yaw = float(last["yaw_deg"])

points = np.atleast_2d(np.loadtxt(
    root / "detected_obstacle_world_points.csv",
    delimiter=",", skiprows=1
))

if points.shape[1] != 3 or not np.isfinite(points).all():
    raise RuntimeError("Invalid camera points")

# Estimate obstacle bounds from camera measurements only.
low = points[:, :2].min(axis=0) - [0.15, 0.05]
high = points[:, :2].max(axis=0) + [0.01, 0.05]

goal = np.array([-1.10, 1.50])
required = 0.521
extra_buffer = 0.030
planning_threshold = required + extra_buffer

def clearance(positions):
    positions = np.asarray(positions)
    offset = np.maximum(
        np.maximum(low - positions, positions - high),
        0.0
    )
    return np.linalg.norm(offset, axis=-1)

def segment(a, b):
    n = max(
        2, int(np.ceil(np.linalg.norm(b-a) / 0.005))+1
    )
    t = np.linspace(0, 1, n)[:, None]
    return a + t*(b-a)

heading = np.array([
    -math.cos(math.radians(yaw)),
    -math.sin(math.radians(yaw))
])

candidates = []

for y in np.arange(1.45, 1.76, 0.05):
    for x in np.arange(-0.25, 0.31, 0.05):
        via = np.array([x, y])

        first = segment(start, via)
        second = segment(via, goal)

        first_min = float(clearance(first).min())
        second_min = float(clearance(second).min())
        minimum = min(first_min, second_min)

        if minimum < planning_threshold:
            continue

        vector = via - start
        direction = vector / np.linalg.norm(vector)

        angle = math.degrees(math.acos(
            float(np.clip(np.dot(heading, direction), -1, 1))
        ))

        total_length = (
            np.linalg.norm(via-start)
            + np.linalg.norm(goal-via)
        )

        # Prefer strong clearance, small initial reheading
        # and then reasonable travel distance.
        score = (
            3.0 * angle / 90.0
            + total_length
            + 5.0 / max(minimum, 0.01)
        )

        candidates.append(
            (score, minimum, angle, total_length, via)
        )

print("=== Post-Turn Camera Replanning ===")
print("Actual turn-end XY:", np.round(start, 4))
print("Actual turn-end yaw:", f"{yaw:+.2f}deg")
print("Camera-estimated obstacle min:", np.round(low, 4))
print("Camera-estimated obstacle max:", np.round(high, 4))
print("Minimum planning clearance:",
      f"{planning_threshold:.3f}m")
print("Start clearance:",
      f"{float(clearance(start)):.3f}m")
print("Accepted candidate count:", len(candidates))

if not candidates:
    print("POST_TURN_REPLAN: REJECT")
else:
    score, minimum, angle, length, via = min(
        candidates, key=lambda c: c[0]
    )

    print("Generated corrected waypoint:",
          np.round(via, 4))
    print("Minimum predicted clearance:",
          f"{minimum:.3f}m")
    print("Required heading correction:",
          f"{angle:.2f}deg")
    print("Route length:", f"{length:.3f}m")

    output = root / "post_turn_camera_route.csv"
    path = np.vstack([
        segment(start, via),
        segment(via, goal)
    ])

    np.savetxt(
        output,
        path,
        delimiter=",",
        header="x_m,y_m",
        comments=""
    )

    print("Saved candidate route:", output)
    print("POST_TURN_REPLAN: STATIC_PASS")

# Report what happened in the previous actual walk.
observed_xy = np.array([
    [float(r["x_m"]), float(r["y_m"])]
    for r in walk
])

observed_clearance = clearance(observed_xy)

print("\n=== Previous Walk Comparison ===")
print("Recorded walk samples:", len(walk))
print("Previous estimated minimum clearance:",
      f"{float(observed_clearance.min()):.3f}m")
print("Previous final XY:",
      np.round(observed_xy[-1], 4))
print("Previous path reserve:",
      f"{float(observed_clearance.min()-required):+.3f}m")
print("No robot motion performed")
print("Camera model and uncertainty bounds remain provisional")
