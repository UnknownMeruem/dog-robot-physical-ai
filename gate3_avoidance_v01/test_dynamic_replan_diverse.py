from pathlib import Path
import ast
import heapq
import json
import math

ROOT = Path("../gate3_avoidance_v01")
SOURCE = ROOT / "test_multi_obstacle_replanner.py"
OUT = ROOT / "step4_multi_obstacle_results"
OUT.mkdir(exist_ok=True)

# Reuse the existing A* unchanged.
source_text = SOURCE.read_text()
tree = ast.parse(source_text)

names = (
    "to_xy", "to_node", "inside", "distance_to_box",
    "clearance", "point_safe", "edge_safe",
    "plan", "verify_path", "box",
)

found = {
    n.name: n for n in tree.body
    if isinstance(n, ast.FunctionDef)
}
missing = [name for name in names if name not in found]
if missing:
    raise SystemExit(f"Missing planner functions: {missing}")

module = ast.Module(
    body=[found[name] for name in names],
    type_ignores=[]
)
ast.fix_missing_locations(module)

GRID = 0.05
CLEARANCE = 0.551
SAMPLE = 0.01
XMIN, XMAX = -2.0, 0.85
YMIN, YMAX = -0.60, 2.60
NX = round((XMAX-XMIN)/GRID)+1
NY = round((YMAX-YMIN)/GRID)+1
DIRECTIONS = [
    (dx, dy)
    for dx in (-1, 0, 1)
    for dy in (-1, 0, 1)
    if dx or dy
]

START = (-0.22, 0.43)
GOAL = (-1.10, 1.50)
BASE = box if False else None

exec(compile(module, str(SOURCE), "exec"), globals())

BASE = box(-1.015, 0.80, 0.155, 0.13)

def signature(path):
    # Quantize for a conservative check of route diversity.
    return tuple(
        (round(p[0], 1), round(p[1], 1))
        for p in path[::max(1, len(path)//12)]
    )

def arclength_locations(path):
    cumulative = 0.0
    result = []
    for a, b in zip(path, path[1:]):
        length = math.dist(a, b)
        cumulative += length
        result.append((cumulative, tuple(b)))
    return result

# Deterministic initial two-obstacle configurations.
# Each must have a safe original route.
initial_layouts = []
for x in (-1.85, -1.50, -1.15, -0.80, -0.45):
    for y in (-0.20, 2.25):
        initial_layouts.append([
            BASE,
            box(x, y, 0.04, 0.04),
        ])

candidates = []
rejected = []
attempts = 0

print("=== DIVERSE DYNAMIC REPLANNING ===")
print("Planner source:", SOURCE)
print("Planning clearance:", CLEARANCE)
print("Initial layouts:", len(initial_layouts))

for layout_index, obstacles in enumerate(initial_layouts):
    initial_path, initial_decision = plan(
        START, GOAL, obstacles
    )

    if initial_decision != "PLANNED":
        rejected.append({
            "layout": layout_index,
            "reason": "INITIAL_" + initial_decision,
        })
        continue

    # Place obstacles at different forward distances and sizes.
    locations = arclength_locations(initial_path)

    for target_distance in (0.85, 1.05, 1.25, 1.45):
        sites = [
            (abs(d-target_distance), p)
            for d, p in locations
        ]
        if not sites:
            continue

        _, position = min(sites, key=lambda v: v[0])

        for half in (0.035, 0.055, 0.075):
            attempts += 1
            new_obstacle = box(
                position[0], position[1],
                half, half
            )
            updated = obstacles + [new_obstacle]

            # Establish the preconditions independently.
            start_safe = point_safe(START, updated)
            goal_safe = point_safe(GOAL, updated)
            old_blocked = not verify_path(
                initial_path, updated
            )

            if not (start_safe and goal_safe and old_blocked):
                rejected.append({
                    "layout": layout_index,
                    "target_distance": target_distance,
                    "half_size": half,
                    "reason": "DYNAMIC_PRECONDITION_FAILED",
                })
                continue

            new_path, decision = plan(
                START, GOAL, updated
            )

            valid = (
                decision == "PLANNED"
                and new_path is not None
                and verify_path(new_path, updated)
            )

            candidates.append({
                "layout": layout_index,
                "target_distance_m": target_distance,
                "half_size_m": half,
                "initial_obstacles": obstacles,
                "added_obstacle": new_obstacle,
                "initial_path": initial_path,
                "replanned_path": new_path,
                "decision": decision,
                "valid": bool(valid),
                "route_signature": (
                    signature(new_path) if new_path else None
                ),
            })

# Select diverse outcomes, rather than the first eight successes.
selected = []
seen = set()

for item in candidates:
    if not item["valid"]:
        continue
    sig = item["route_signature"]
    if sig in seen:
        continue
    selected.append(item)
    seen.add(sig)
    if len(selected) == 8:
        break

# The test never fabricates additional cases to reach eight.
valid_count = sum(item["valid"] for item in selected)
unique_count = len(seen)

print("Candidate attempts:", attempts)
print("Valid dynamic replans found:",
      sum(item["valid"] for item in candidates))
print("Distinct replans selected:", unique_count)

for index, item in enumerate(selected):
    print(
        f"case_{index:02d}",
        "layout=", item["layout"],
        "distance=", item["target_distance_m"],
        "half=", item["half_size_m"],
        "decision=", item["decision"],
        "PASS" if item["valid"] else "FAIL",
    )

# Separate negative test: the original goal becomes unsafe.
goal_blocker = box(
    GOAL[0], GOAL[1], 0.045, 0.045
)
negative_path, negative_reason = plan(
    START, GOAL, [BASE, goal_blocker]
)
negative_pass = (
    negative_path is None
    and negative_reason == "GOAL_UNSAFE"
)

print("Unsafe goal rejection:",
      "PASS" if negative_pass else "FAIL")

# Require eight distinct valid routes, rather than counting
# repeated routes as separate evidence.
passed = (
    len(selected) == 8
    and valid_count == 8
    and unique_count == 8
    and negative_pass
)

report = {
    "planner_unchanged": True,
    "planning_clearance_m": CLEARANCE,
    "candidate_attempts": attempts,
    "valid_candidate_count": sum(
        item["valid"] for item in candidates
    ),
    "distinct_valid_replans": unique_count,
    "selected_cases": selected,
    "negative_goal_test": {
        "decision": negative_reason,
        "pass": negative_pass,
    },
    "skipped_candidates": rejected,
    "passed": bool(passed),
    "limitations": [
        "Synthetic 2D obstacle maps",
        "Replanning starts at a fixed safe pose",
        "Not integrated with live MuJoCo depth",
        "No locomotion or full-body contact audit",
    ],
}

path = OUT / "diverse_dynamic_summary.json"
path.write_text(
    json.dumps(report, indent=2, ensure_ascii=False)
)

print("\n=== FINAL SUMMARY ===")
print("Distinct replans:", unique_count, "/ 8 required")
print("Unsafe goal rejection:",
      "PASS" if negative_pass else "FAIL")
print(
    "STEP4_DIVERSE_DYNAMIC_PLANNER:",
    "PASS" if passed else "FAIL"
)
print("Summary:", path)
print("MuJoCo runtime integration: NOT YET")

if not passed:
    raise SystemExit(1)
