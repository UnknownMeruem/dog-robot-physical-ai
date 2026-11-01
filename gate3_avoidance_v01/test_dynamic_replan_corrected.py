from pathlib import Path
import ast
import heapq
import json
import math

ROOT = Path("../gate3_avoidance_v01")
SOURCE = ROOT / "test_multi_obstacle_replanner.py"
OUT = ROOT / "step4_multi_obstacle_results"
OUT.mkdir(exist_ok=True)

# Extract the original A* implementation without executing
# its earlier benchmark or modifying its algorithm.
tree = ast.parse(SOURCE.read_text())
required = (
    "to_xy", "to_node", "inside", "distance_to_box",
    "clearance", "point_safe", "edge_safe",
    "plan", "verify_path", "box",
)
functions = {
    node.name: node for node in tree.body
    if isinstance(node, ast.FunctionDef)
}
missing = set(required) - set(functions)
if missing:
    raise SystemExit(f"Missing planner functions: {missing}")

module = ast.Module(
    body=[functions[name] for name in required],
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

exec(compile(module, str(SOURCE), "exec"), globals())

def path_length(path):
    return sum(
        math.dist(a, b)
        for a, b in zip(path, path[1:])
    )

def make_dynamic_case(case):
    # Two separate initial obstacles.
    # Each case places the second obstacle at a different
    # position, producing different initial routes.
    first = box(-1.015, 0.80, 0.155, 0.13)
    second = box(
        -1.65 + (case % 4)*0.25,
        0.35 + (case // 4)*1.65,
        0.035, 0.035,
    )
    return [first, second]

def construct_test(case):
    initial = make_dynamic_case(case)
    path, decision = plan(START, GOAL, initial)

    if decision != "PLANNED":
        return {
            "case": case,
            "valid": False,
            "reason": f"INITIAL_{decision}",
        }

    # The robot remains at a currently verified safe stop.
    # Place a NEW obstacle sufficiently far ahead along
    # the planned path. This emulates look-ahead detection,
    # not an obstacle materializing around the robot.
    candidate = None
    for point in path[1:-1]:
        distance = math.dist(START, point)
        if distance >= 0.80 and distance <= 1.15:
            candidate = tuple(point)
            break

    if candidate is None:
        return {
            "case": case,
            "valid": False,
            "reason": "NO_FORWARD_INSERTION_SITE",
        }

    added = box(candidate[0], candidate[1], 0.055, 0.055)
    updated = initial + [added]

    start_clear = clearance(START, updated)
    old_route_safe = verify_path(path, updated)

    # Stop -> update map -> replan from the actual safe pose.
    new_path, decision = plan(START, GOAL, updated)

    if decision == "PLANNED":
        safe = verify_path(new_path, updated)
        outcome = "SAFE_DYNAMIC_REPLAN" if safe else "UNSAFE_PATH"
    else:
        safe = (
            new_path is None
            and decision in (
                "START_UNSAFE",
                "GOAL_UNSAFE",
                "NO_SAFE_ROUTE",
            )
        )
        outcome = "SAFE_DYNAMIC_STOP" if safe else "INVALID_STOP"

    # Reject a test that did not actually introduce a new
    # obstruction or that made the initial pose unsafe.
    valid = (
        safe
        and start_clear >= CLEARANCE
        and not old_route_safe
    )

    return {
        "case": case,
        "valid": bool(valid),
        "outcome": outcome,
        "decision": decision,
        "initial_obstacles": initial,
        "added_obstacle": added,
        "start_clearance_m": start_clear,
        "old_route_invalidated": not old_route_safe,
        "initial_path_length_m": path_length(path),
        "new_path_length_m": (
            path_length(new_path)
            if new_path is not None else None
        ),
        "initial_path": path,
        "new_path": new_path,
    }

print("=== CORRECTED DYNAMIC REPLANNING TEST ===")
print("Planner:", SOURCE)
print("Runtime safety threshold remains: 0.521m")
print("Planning threshold remains: 0.551m")
print("Number of initial obstacles: 2")
print("Dynamic cases: 8")
print("Existing source unchanged", flush=True)

results = [construct_test(i) for i in range(8)]

for item in results:
    print(f"\ncase_{item['case']:02d}")
    print("Outcome:", item.get("outcome", item.get("reason")))
    print("Decision:", item.get("decision"))
    print("Start clearance:", item.get("start_clearance_m"))
    print("Old route invalidated:",
          item.get("old_route_invalidated"))
    print("New route length:",
          item.get("new_path_length_m"))
    print("CASE_RESULT:",
          "PASS" if item["valid"] else "FAIL")

replans = sum(
    r.get("outcome") == "SAFE_DYNAMIC_REPLAN"
    and r["valid"]
    for r in results
)
safe_stops = sum(
    r.get("outcome") == "SAFE_DYNAMIC_STOP"
    and r["valid"]
    for r in results
)
valid_count = sum(r["valid"] for r in results)

success = (
    valid_count == len(results)
    and replans >= 3
)

report = {
    "static_planner": str(SOURCE),
    "cases": results,
    "valid_cases": valid_count,
    "dynamic_replans": replans,
    "safe_dynamic_stops": safe_stops,
    "pass": success,
    "limitations": [
        "Standalone 2D planner validation only",
        "Obstacles are synthetic rectangles",
        "Robot remains at a fixed pre-motion stopping pose",
        "No live MuJoCo camera reacquisition",
        "No robot motion or full-body physics tested",
    ],
}

output = OUT / "corrected_dynamic_summary.json"
output.write_text(json.dumps(report, indent=2))

print("\n=== CORRECTED DYNAMIC SUMMARY ===")
print(f"Valid decisions: {valid_count}/{len(results)}")
print("Actual dynamic replans:", replans)
print("Safe dynamic stops:", safe_stops)
print(
    "STEP4_DYNAMIC_PLANNER:",
    "PASS" if success else "FAIL"
)
print("Report:", output)
print("Live MuJoCo dynamic integration: NOT YET")

if not success:
    raise SystemExit(1)
