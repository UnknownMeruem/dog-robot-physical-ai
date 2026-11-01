from pathlib import Path
import ast
import csv
import json
import math
import re

root = Path("../gate3_avoidance_v01")
source = root / "test_step4_unified_x_minus_05.py"
random_dir = root / "step4_randomized_audit"
output = random_dir / "turn_sweep_design_audit.json"

RUNTIME = 0.521
PLANNING = 0.551

if not source.is_file():
    raise SystemExit("SOURCE_MISSING")
if not random_dir.is_dir():
    raise SystemExit("RANDOM_RESULTS_MISSING")

def load_json(path):
    try:
        return json.loads(path.read_text())
    except (OSError, ValueError):
        return {}

def load_csv(path):
    if not path.exists():
        return []
    with path.open(newline="") as f:
        return list(csv.DictReader(f))

def value(row, key):
    try:
        return float(row.get(key, "nan"))
    except (ValueError, TypeError):
        return float("nan")

def finite_min(rows, key):
    vals = [value(r, key) for r in rows]
    vals = [x for x in vals if math.isfinite(x)]
    return min(vals) if vals else None

def distance(a, b):
    return math.hypot(a[0]-b[0], a[1]-b[1])

cases = []
print("=== ALL-CASE TURN SWEEP AUDIT ===")

for i in range(12):
    case = f"case_{i:02d}"
    prefix = f"step4_random_{case}_"

    rows = load_csv(root / (
        prefix + "live_autonomous_route_v3.csv"
    ))
    report = load_json(root / (
        prefix + "live_autonomous_route_v3.json"
    ))
    log_path = random_dir / f"{case}.log"
    log = (
        log_path.read_text(errors="replace")
        if log_path.exists() else ""
    )

    sections = {}
    for r in rows:
        name = r.get("section", "")
        if name:
            sections.setdefault(name, []).append(r)

    turns = []
    for name, segment in sections.items():
        if "TURN" not in name.upper():
            continue

        valid = [
            r for r in segment
            if all(math.isfinite(value(r, k))
                   for k in ("x_m", "y_m"))
        ]
        if not valid:
            continue

        start = (
            value(valid[0], "x_m"),
            value(valid[0], "y_m"),
        )

        displacement = [
            distance(
                (
                    value(r, "x_m"),
                    value(r, "y_m"),
                ),
                start
            )
            for r in valid
        ]

        minimum = finite_min(
            valid, "estimated_clearance_m"
        )

        turns.append({
            "section": name,
            "samples": len(valid),
            "maximum_xy_drift_m": max(displacement),
            "final_xy_drift_m": displacement[-1],
            "minimum_estimated_clearance_m": minimum,
            "runtime_violation": (
                minimum < RUNTIME
                if minimum is not None else None
            ),
        })

    guards = [
        {
            "section": r.get("section"),
            "reason": r.get("guard_reason"),
            "time_s": r.get("time_s"),
        }
        for r in rows
        if r.get("guard_reason", "").strip()
    ]

    item = {
        "case": case,
        "status": report.get("status", "NO_REPORT"),
        "turns": turns,
        "guard_samples": len(guards),
        "first_guard": guards[0] if guards else None,
        "trace_samples": len(rows),
        "log_has_post_turn_zero_candidates": bool(
            re.search(
                r"POST_TURN_CANDIDATES:\s*0\b", log
            )
        ),
    }
    cases.append(item)

    max_drift = max(
        (t["maximum_xy_drift_m"] for t in turns),
        default=None,
    )
    print(
        f"{case} | "
        f"status={item['status']} | "
        f"max_turn_drift="
        f"{max_drift:.4f}m"
        if max_drift is not None
        else f"{case} | no turn trace"
    )
    print(
        "   guards:", len(guards),
        "post-turn zero candidates:",
        item["log_has_post_turn_zero_candidates"]
    )

# Extract the actual planning/controller code without
# assuming fixed line numbers.
code = source.read_text()
tree = ast.parse(code)

controllers = {}
for node in tree.body:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
        if node.name in (
            "route_candidates",
            "route_clearance",
            "segment",
            "turn_to",
            "walk_to",
            "guard",
        ):
            controllers[node.name] = {
                "first_line": node.lineno,
                "last_line": node.end_lineno,
                "source": ast.get_source_segment(code, node),
            }

# Check which control operations precede or follow
# the actual LIVE_TURN1 call.
calls = []
for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
        continue
    if not isinstance(node.func, ast.Name):
        continue
    if node.func.id not in (
        "turn_to", "walk_to", "route_candidates"
    ):
        continue

    snippet = ast.get_source_segment(code, node) or ""
    if (
        "LIVE_TURN1" in snippet
        or "LIVE_CORRECTIVE_TURN" in snippet
        or node.func.id == "route_candidates"
    ):
        calls.append({
            "line": node.lineno,
            "call": snippet[:500],
        })

observed_drifts = [
    t["maximum_xy_drift_m"]
    for item in cases
    for t in item["turns"]
]
max_observed = (
    max(observed_drifts)
    if observed_drifts else None
)

report = {
    "runtime_clearance_m": RUNTIME,
    "planning_clearance_m": PLANNING,
    "observed_maximum_turn_drift_m": max_observed,
    "note": (
        "Observed maximum is retrospective only. "
        "It is NOT a certified future drift bound. "
        "No new controller motion is authorized."
    ),
    "cases": cases,
    "controller_definitions": controllers,
    "controller_calls": sorted(
        calls, key=lambda x: x["line"]
    ),
    "next_design": {
        "required": [
            "Predict turn sweep before first turn",
            "Reject turn if sweep margin is inadequate",
            "Check map and clearance during turning",
            "Reobserve from the stopped pose",
            "Replan without assuming the original waypoint",
            "Treat absent safe route as a safe stop",
            "Verify multiple obstacle occupancy separately",
        ],
        "current_step4_complete": False,
    },
}

output.write_text(
    json.dumps(report, indent=2, ensure_ascii=False)
)

print("\n=== DESIGN AUDIT SUMMARY ===")
print("Cases:", len(cases))
print("Observed max turn drift:", max_observed)
print(
    "Cases with zero post-turn routes:",
    [
        x["case"] for x in cases
        if x["log_has_post_turn_zero_candidates"]
    ]
)
print("Controller functions found:",
      sorted(controllers))
print("Report:", output)
print("Simulation reruns: 0")
print("Safety criteria changed: 0")
print("STEP4_TURN_SWEEP_DESIGN_AUDIT: COMPLETE")
