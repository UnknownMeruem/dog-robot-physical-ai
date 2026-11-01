from pathlib import Path
import csv
import json
import math
import numpy as np

root = Path(__file__).resolve().parent
folder = root / "step4_robustness_results"

RUNTIME = 0.521
PLANNING = 0.551
GOAL = np.array([-1.10, 1.50])

cases = [
    "x_minus_05",
    "x_plus_05",
    "y_minus_05",
    "y_plus_05",
    "xy_shift",
    "nan_map_fault",
]

def load_json(path):
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def clearance(point, low, high):
    offset = np.maximum(
        np.maximum(low-point, point-high), 0
    )
    return float(np.linalg.norm(offset))

def read_trace(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def number(row, key):
    try:
        return float(row.get(key, "nan"))
    except (TypeError, ValueError):
        return float("nan")

def minimum(rows, key):
    values = [
        (number(row, key), row)
        for row in rows
    ]
    values = [
        item for item in values
        if math.isfinite(item[0])
    ]
    return min(values, key=lambda x: x[0]) if values else None

summary = []

print("=== STEP 4 ROOT-CAUSE AUDIT ===")
print("Runtime:", RUNTIME)
print("Planning:", PLANNING)

for case in cases:
    prefix = f"step4_robust_{case}_"

    live = load_json(
        root / (
            prefix +
            "early_live_replan_result_v3_early.json"
        )
    )
    result = load_json(
        root / (
            prefix + "live_autonomous_route_v3.json"
        )
    )
    trace = read_trace(
        root / (
            prefix + "live_autonomous_route_v3.csv"
        )
    )
    log_file = folder / (case + ".log")
    lines = (
        log_file.read_text(errors="replace").splitlines()
        if log_file.exists() else []
    )

    low = np.asarray(
        live.get("estimated_box_min", []), dtype=float
    )
    high = np.asarray(
        live.get("estimated_box_max", []), dtype=float
    )
    valid_map = (
        low.shape == (2,)
        and high.shape == (2,)
        and np.isfinite(low).all()
        and np.isfinite(high).all()
        and np.all(low <= high)
    )

    goal_distance = (
        clearance(GOAL, low, high)
        if valid_map else None
    )
    status = result.get("status", "NO_REPORT")

    events = [
        line.strip() for line in lines
        if any(term in line for term in (
            "LIVE_REPLAN:",
            "POST_TURN_CANDIDATES:",
            "POST_TURN_PREDICTED_CLEARANCE:",
            "LIVE_TURN1:",
            "LIVE_CORRECTIVE_TURN:",
            "Live waypoint walk:",
            "LIVE_FINAL_TURN:",
            "Live final walk:",
            "GATE3_STEP3:",
            "Traceback",
            "Error:",
        ))
    ]

    # Classification describes observed evidence,
    # not a claim of certified physical safety.
    if status == "PASS":
        category = "REACHED_GOAL"
    elif status == "GOAL_MAP_INVALID_REJECTED":
        category = "INVALID_MAP_REJECTED"
    elif not valid_map:
        category = "INVALID_MAP_UNRESOLVED"
    elif goal_distance < RUNTIME:
        category = "GOAL_RUNTIME_CLEARANCE_VIOLATION"
    elif goal_distance < PLANNING:
        category = "GOAL_PLANNING_MARGIN_INSUFFICIENT"
    elif status in (
        "NO_ROBUST_ROUTE",
        "LIVE_OBSERVATION_NOT_READY",
    ):
        category = "ROUTE_OR_OBSERVATION_LIMITATION"
    else:
        category = "DYNAMIC_CONTROL_OR_GUARD_FAILURE"

    minima = {}
    for label, field in (
        ("estimated", "estimated_clearance_m"),
        ("extra", "extra_clearance_m"),
        ("red", "red_clearance_m"),
        ("height", "height_m"),
        ("tilt", "tilt_deg"),
    ):
        item = minimum(trace, field)
        minima[label] = item[0] if item else None

    guard_events = [
        {
            "time": r.get("time_s"),
            "reason": r.get("guard_reason"),
            "xy": [
                number(r, "x_m"),
                number(r, "y_m"),
            ],
        }
        for r in trace if r.get("guard_reason")
    ]

    row = {
        "case": case,
        "status": status,
        "classification": category,
        "goal_clearance_m": goal_distance,
        "runtime_goal_reserve_m": (
            goal_distance-RUNTIME
            if goal_distance is not None else None
        ),
        "planning_goal_reserve_m": (
            goal_distance-PLANNING
            if goal_distance is not None else None
        ),
        "live_candidate_count": live.get("candidate_count"),
        "observed_points": live.get("observed_points"),
        "trace_samples": len(trace),
        "minima": minima,
        "first_guard_event": (
            guard_events[0] if guard_events else None
        ),
        "events": events[-20:],
    }
    summary.append(row)

    print(f"\n--- {case} ---")
    print("Status:", status)
    print("Classification:", category)
    print("Live candidates:",
          live.get("candidate_count"))
    print("Goal clearance:",
          f"{goal_distance:.4f}m"
          if goal_distance is not None else "INVALID")
    print("Trace samples:", len(trace))
    print("Minimum metrics:", minima)
    print("First guard event:",
          row["first_guard_event"] or "NONE_RECORDED")
    for event in events[-12:]:
        print(" ", event)

report = folder / "root_cause_audit.json"
report.write_text(
    json.dumps(summary, indent=2, ensure_ascii=False)
)

print("\n=== NEXT DEVELOPMENT TARGETS ===")
for row in summary:
    if row["status"] != "PASS":
        print(row["case"], "=>", row["classification"])

print("Audit report:", report)
print("Independent contact and leg-sweep audit: NOT_DONE")
print("No safety thresholds modified")
print("No simulations rerun")
