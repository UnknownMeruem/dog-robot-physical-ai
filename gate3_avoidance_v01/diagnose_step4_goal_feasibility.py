from pathlib import Path
import json
import numpy as np

root = Path("../gate3_avoidance_v01")
out = root / "step4_expanded_results"
goal = np.array([-1.10, 1.50])
REQUIRED = 0.521
PLANNING = 0.551

cases = ["baseline", "plus_10cm", "minus_10cm"]
results = []

def clearance(xy, low, high):
    delta = np.maximum(
        np.maximum(low - xy, xy - high), 0.0
    )
    return float(np.linalg.norm(delta))

print("=== STEP 4 GOAL FEASIBILITY AUDIT ===")
print("Goal:", goal)
print("Runtime minimum:", REQUIRED)
print("Planning minimum:", PLANNING)

for case in cases:
    prefix = f"step4_expanded_{case}_"

    live_file = root / (
        prefix + "early_live_replan_result_v3_early.json"
    )
    run_file = root / (
        prefix + "live_autonomous_route_v3.json"
    )

    print(f"\n--- {case} ---")

    if not live_file.exists():
        print("MISSING_LIVE_REPORT:", live_file)
        results.append({
            "case": case, "classification": "MISSING_DATA"
        })
        continue

    live = json.loads(live_file.read_text())
    low = np.asarray(
        live.get("estimated_box_min", []), dtype=float
    )
    high = np.asarray(
        live.get("estimated_box_max", []), dtype=float
    )

    if (
        low.shape != (2,) or high.shape != (2,)
        or not np.isfinite(low).all()
        or not np.isfinite(high).all()
        or not np.all(low <= high)
    ):
        print("INVALID_OBSTACLE_BOUNDS")
        results.append({
            "case": case, "classification": "INVALID_MAP"
        })
        continue

    start = np.asarray(live["start_xy"], dtype=float)
    start_clear = clearance(start, low, high)
    goal_clear = clearance(goal, low, high)

    if goal_clear < REQUIRED:
        classification = "GOAL_VIOLATES_RUNTIME_CLEARANCE"
    elif goal_clear < PLANNING:
        classification = "GOAL_VIOLATES_PLANNING_RESERVE"
    else:
        classification = "GOAL_FEASIBLE_CHECK_ROUTE"

    print("Observed obstacle bounds:", low, high)
    print("Start XY:", start)
    print("Start clearance:", f"{start_clear:.4f}m")
    print("Goal clearance:", f"{goal_clear:.4f}m")
    print("Runtime reserve:", f"{goal_clear-REQUIRED:+.4f}m")
    print("Planning reserve:", f"{goal_clear-PLANNING:+.4f}m")
    print("Live candidates:", live.get("candidate_count"))
    print("CLASSIFICATION:", classification)

    if run_file.exists():
        run = json.loads(run_file.read_text())
        print("Previous execution:", run.get("status"))

    # Candidate alternative goal locations are diagnostics only.
    # They do not count as reaching the original target.
    alternatives = []
    for x in np.arange(-1.60, -0.69, 0.05):
        for y in np.arange(1.30, 2.01, 0.05):
            point = np.array([x, y])
            margin = clearance(point, low, high)
            if margin >= PLANNING:
                alternatives.append((
                    float(np.linalg.norm(point-goal)),
                    float(margin), point
                ))

    if alternatives:
        distance, margin, point = min(
            alternatives, key=lambda item: item[0]
        )
        print("Nearest margin-compliant alternative:",
              np.round(point, 3))
        print("Alternative goal offset:",
              f"{distance:.3f}m")
        print("Alternative goal clearance:",
              f"{margin:.3f}m")
    else:
        point = None
        print("No alternative goal in diagnostic grid")

    results.append({
        "case": case,
        "classification": classification,
        "start_clearance_m": start_clear,
        "original_goal_clearance_m": goal_clear,
        "runtime_margin_m": goal_clear-REQUIRED,
        "planning_margin_m": goal_clear-PLANNING,
        "candidate_count": live.get("candidate_count"),
        "alternative_goal": (
            point.tolist() if point is not None else None
        ),
        "alternative_not_executed": True
    })

report = out / "goal_feasibility_audit.json"
report.write_text(json.dumps(results, indent=2))

print("\n=== OVERALL DIAGNOSIS ===")
for item in results:
    print(item["case"], ":", item["classification"])
print("Report:", report)
print("No robot motion executed")
print("No safety thresholds changed")
