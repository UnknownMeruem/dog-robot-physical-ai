from pathlib import Path
import heapq
import json
import math
import random

ROOT = Path("../gate3_avoidance_v01")
OUT = ROOT / "step4_multi_obstacle_results"
OUT.mkdir(exist_ok=True)

SEED = 20261102
GRID = 0.05
CLEARANCE = 0.551
SAMPLE = 0.01

START = (-0.22, 0.43)
GOAL = (-1.10, 1.50)

XMIN, XMAX = -2.0, 0.85
YMIN, YMAX = -0.60, 2.60

NX = round((XMAX-XMIN)/GRID)+1
NY = round((YMAX-YMIN)/GRID)+1

rng = random.Random(SEED)

def to_xy(n):
    return (
        XMIN+n[0]*GRID,
        YMIN+n[1]*GRID,
    )

def to_node(p):
    return (
        round((p[0]-XMIN)/GRID),
        round((p[1]-YMIN)/GRID),
    )

def inside(n):
    return 0 <= n[0] < NX and 0 <= n[1] < NY

def distance_to_box(p, box):
    x0, y0, x1, y1 = box
    dx = max(x0-p[0], p[0]-x1, 0.0)
    dy = max(y0-p[1], p[1]-y1, 0.0)
    return math.hypot(dx, dy)

def clearance(p, obstacles):
    if not obstacles:
        return math.inf
    return min(distance_to_box(p, b) for b in obstacles)

def point_safe(p, obstacles):
    return clearance(p, obstacles) >= CLEARANCE - 1e-10

def edge_safe(a, b, obstacles):
    length = math.dist(a, b)
    count = max(1, math.ceil(length/SAMPLE))
    for i in range(count+1):
        t = i/count
        p = (
            a[0]+t*(b[0]-a[0]),
            a[1]+t*(b[1]-a[1]),
        )
        if not point_safe(p, obstacles):
            return False
    return True

DIRECTIONS = [
    (dx, dy)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if dx or dy
]

def plan(start, goal, obstacles):
    if not point_safe(start, obstacles):
        return None, "START_UNSAFE"
    if not point_safe(goal, obstacles):
        return None, "GOAL_UNSAFE"

    s = to_node(start)
    g = to_node(goal)

    if not inside(s) or not inside(g):
        return None, "OUTSIDE_PLANNING_AREA"

    if not edge_safe(start, to_xy(s), obstacles):
        return None, "START_GRID_CONNECTION_UNSAFE"
    if not edge_safe(to_xy(g), goal, obstacles):
        return None, "GOAL_GRID_CONNECTION_UNSAFE"

    queue = [(0.0, 0.0, s)]
    best = {s: 0.0}
    parent = {s: None}
    closed = set()

    # Cache occupancy; verify every diagonal and cardinal edge.
    safe_cache = {}
    def safe_node(n):
        if n not in safe_cache:
            safe_cache[n] = point_safe(to_xy(n), obstacles)
        return safe_cache[n]

    found = False

    while queue:
        _, cost, current = heapq.heappop(queue)

        if current in closed:
            continue
        closed.add(current)

        if current == g:
            found = True
            break

        for dx, dy in DIRECTIONS:
            nxt = (current[0]+dx, current[1]+dy)

            if (
                not inside(nxt)
                or nxt in closed
                or not safe_node(nxt)
            ):
                continue

            a = to_xy(current)
            b = to_xy(nxt)

            if not edge_safe(a, b, obstacles):
                continue

            step = math.hypot(dx, dy)*GRID
            new_cost = cost+step

            if new_cost >= best.get(nxt, math.inf):
                continue

            best[nxt] = new_cost
            parent[nxt] = current

            estimate = (
                new_cost + math.dist(b, to_xy(g))
            )
            heapq.heappush(
                queue, (estimate, new_cost, nxt)
            )

    if not found:
        return None, "NO_SAFE_ROUTE"

    nodes = []
    current = g
    while current is not None:
        nodes.append(current)
        current = parent[current]
    nodes.reverse()

    path = [start]
    path.extend(to_xy(n) for n in nodes)
    path.append(goal)

    if not verify_path(path, obstacles):
        raise AssertionError(
            "A* returned an invalid path"
        )

    return path, "PLANNED"

def verify_path(path, obstacles):
    if not path:
        return False
    if not all(point_safe(p, obstacles) for p in path):
        return False
    return all(
        edge_safe(a, b, obstacles)
        for a, b in zip(path, path[1:])
    )

def box(cx, cy, hx, hy):
    return (
        cx-hx, cy-hy,
        cx+hx, cy+hy,
    )

# Representative camera-derived obstacle geometry.
# This is NOT a live-camera measurement for each test.
BASE = box(-1.015, 0.80, 0.155, 0.13)

def make_obstacles(case):
    count = 2+(case % 2)
    obstacles = [BASE]

    for _ in range(count-1):
        cx = rng.uniform(-1.55, 0.20)
        cy = rng.uniform(0.55, 1.75)
        hx = rng.uniform(0.035, 0.08)
        hy = rng.uniform(0.035, 0.08)
        obstacles.append(box(cx, cy, hx, hy))

    return obstacles

results = []

print("=== STEP 4 MULTI-OBSTACLE A* TEST ===")
print("Seed:", SEED)
print("Grid:", GRID)
print("Clearance:", CLEARANCE)

for case in range(24):
    obstacles = make_obstacles(case)
    path, decision = plan(
        START, GOAL, obstacles
    )

    if decision == "PLANNED":
        valid = verify_path(path, obstacles)
        result = "SAFE_ROUTE_FOUND" if valid else "INVALID_ROUTE"
    elif decision in (
        "GOAL_UNSAFE",
        "START_UNSAFE",
        "NO_SAFE_ROUTE",
    ):
        valid = path is None
        result = (
            "SAFE_REJECTION" if valid
            else "INVALID_REJECTION"
        )
    else:
        valid = False
        result = decision

    item = {
        "case": case,
        "obstacles": obstacles,
        "decision": decision,
        "path": path,
        "valid": valid,
    }
    results.append(item)

    print(
        f"case_{case:02d}",
        f"obstacles={len(obstacles)}",
        result,
        f"nodes={len(path) if path else 0}",
    )

print("\n=== DYNAMIC OBSTACLE INSERTION ===")

dynamic_results = []

# Start from maps where an initial route exists,
# then add a blocker to the remaining route.
for item in results:
    if item["decision"] != "PLANNED":
        continue

    path = item["path"]
    if len(path) < 7:
        continue

    index = max(2, len(path)//3)
    current = tuple(path[index])
    future = tuple(path[min(index+5, len(path)-2)])

    added = box(
        future[0], future[1],
        0.055, 0.055
    )

    updated = item["obstacles"]+[added]

    # Newly observed obstacle invalidates the old remaining path.
    remaining = path[index:]
    old_path_still_safe = verify_path(
        remaining, updated
    )

    if old_path_still_safe:
        continue

    # Stop at the last known safe point, update the
    # occupancy map, and replan from that pose.
    new_path, decision = plan(
        current, GOAL, updated
    )

    if decision == "PLANNED":
        safe = verify_path(new_path, updated)
        outcome = (
            "SAFE_DYNAMIC_REPLAN"
            if safe else "UNSAFE_REPLAN"
        )
    else:
        safe = (
            new_path is None
            and decision in (
                "START_UNSAFE",
                "GOAL_UNSAFE",
                "NO_SAFE_ROUTE",
            )
        )
        outcome = (
            "SAFE_DYNAMIC_STOP"
            if safe else "UNSAFE_DYNAMIC_DECISION"
        )

    dynamic_results.append({
        "source_case": item["case"],
        "added_obstacle": added,
        "decision": decision,
        "outcome": outcome,
        "valid": safe,
        "replanned_path": new_path,
    })

    print(
        "source_case", item["case"],
        "decision", decision,
        "result", outcome,
    )

    if len(dynamic_results) == 8:
        break

passed_static = sum(x["valid"] for x in results)
passed_dynamic = sum(x["valid"] for x in dynamic_results)

# Require actual route results and actual dynamic invalidations.
success = (
    passed_static == len(results)
    and len(dynamic_results) >= 3
    and passed_dynamic == len(dynamic_results)
    and any(
        x["outcome"] == "SAFE_DYNAMIC_REPLAN"
        for x in dynamic_results
    )
)

report = {
    "seed": SEED,
    "clearance_m": CLEARANCE,
    "static_passed": passed_static,
    "static_total": len(results),
    "dynamic_passed": passed_dynamic,
    "dynamic_total": len(dynamic_results),
    "static_tests": results,
    "dynamic_tests": dynamic_results,
    "passed": success,
    "limitations": [
        "2D planning only",
        "Representative rectangles, not live MuJoCo depth",
        "No full-body or leg-sweep physics audit",
        "Dynamic map insertion is simulated",
        "No robot locomotion is executed here",
    ],
}

output = OUT / "summary.json"
output.write_text(
    json.dumps(report, indent=2)
)

print("\n=== STEP 4 MULTI-OBSTACLE SUMMARY ===")
print(f"Static decisions: {passed_static}/{len(results)}")
print(
    f"Dynamic decisions: "
    f"{passed_dynamic}/{len(dynamic_results)}"
)
print(
    "STEP4_MULTI_OBSTACLE_PLANNER:",
    "PASS" if success else "FAIL"
)
print("Summary:", output)
print("Live MuJoCo integration: NOT YET")
print("Full Step 4 acceptance: NOT YET")

if not success:
    raise SystemExit(1)
