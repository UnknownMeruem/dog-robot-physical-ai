from pathlib import Path
import ast
import json
import re

ROOT = Path("../gate3_avoidance_v01")
SOURCE = ROOT / "test_step4_unified_x_minus_05.py"
OUTPUT = ROOT / "step4_randomized_audit/controller_architecture.json"
EXCERPT = ROOT / "step4_randomized_audit/controller_architecture_excerpt.txt"

code = SOURCE.read_text()
tree = ast.parse(code)
lines = code.splitlines()

keywords = (
    "LIVE_TURN1",
    "LIVE_CORRECTIVE_TURN",
    "POST_TURN_CANDIDATES",
    "POST_TURN_PREDICTED_CLEARANCE",
    "LIVE_WAYPOINT_WALK",
    "LIVE_FINAL_TURN",
    "FINAL_WALK",
    "route_candidates",
    "route_clearance",
    "estimated_clearance",
    "def guard",
    "def turn_to",
    "def walk_to",
    "PLANNING_CLEARANCE",
    "CLEARANCE =",
    "GOAL_UNSAFE_REJECTED",
)

hits = []
for i, line in enumerate(lines, 1):
    if any(k in line for k in keywords):
        hits.append(i)

# Merge overlapping source windows to make a concise,
# reproducible excerpt of the actual controller.
windows = []
for n in hits:
    start = max(1, n - 7)
    end = min(len(lines), n + 15)
    if windows and start <= windows[-1][1] + 1:
        windows[-1] = (windows[-1][0],
                       max(end, windows[-1][1]))
    else:
        windows.append((start, end))

sections = []
for start, end in windows:
    content = "\n".join(
        f"{i:04d} {lines[i-1]}"
        for i in range(start, end+1)
    )
    sections.append(content)

EXCERPT.write_text(
    "\n\n==========\n\n".join(sections)
)

functions = []
for node in ast.walk(tree):
    if isinstance(node, ast.FunctionDef):
        functions.append({
            "name": node.name,
            "line": node.lineno,
            "end_line": node.end_lineno,
        })

turn_calls = []
for node in ast.walk(tree):
    if not isinstance(node, ast.Call):
        continue
    text = ast.get_source_segment(code, node) or ""
    if any(k in text for k in (
        '"LIVE_TURN1"',
        '"LIVE_CORRECTIVE_TURN"',
        '"LIVE_FINAL_TURN"',
    )):
        turn_calls.append({
            "line": node.lineno,
            "call": text[:300],
        })

assertions = {
    "has_first_turn": '"LIVE_TURN1"' in code,
    "has_corrective_turn": '"LIVE_CORRECTIVE_TURN"' in code,
    "has_post_turn_candidates": "POST_TURN_CANDIDATES" in code,
    "has_guard": any(f["name"] == "guard" for f in functions),
    "has_turn_controller": any(
        f["name"] == "turn_to" for f in functions
    ),
    "has_walk_controller": any(
        f["name"] == "walk_to" for f in functions
    ),
    "runtime_clearance_unchanged": (
        "CLEARANCE = 0.521" in code
    ),
    "planning_clearance_unchanged": (
        "PLANNING_CLEARANCE = 0.551" in code
    ),
}

report = {
    "source": str(SOURCE),
    "source_lines": len(lines),
    "assertions": assertions,
    "functions": functions,
    "turn_calls": turn_calls,
    "excerpt_file": str(EXCERPT),
    "recommended_architecture": {
        "pre_turn": [
            "Validate camera observation and map",
            "Evaluate target and complete route",
            "Check prospective turning envelope",
            "Do not start if estimated margin is insufficient",
        ],
        "during_turn": [
            "Monitor posture and all known obstacle distances",
            "Stop immediately on guard failure",
            "Record actual swept geometry",
        ],
        "after_turn": [
            "Hold and measure actual pose",
            "Reobserve camera",
            "Update obstacle map",
            "Replan from actual pose",
            "Retry only within explicit safety and iteration limits",
        ],
        "during_walk": [
            "Continuously check new depth observations",
            "Stop when a newly observed obstacle blocks the route",
            "Repeat observation and replanning before resuming",
        ],
    },
}

OUTPUT.write_text(json.dumps(report, indent=2))

print("=== CONTROLLER ARCHITECTURE AUDIT ===")
for name, value in assertions.items():
    print(
        f"{name}:",
        "PASS" if value else "MISSING"
    )

print("\nTurn call locations:")
for item in turn_calls:
    print(item["line"], item["call"])

print("\nFunction locations:")
for item in functions:
    if item["name"] in (
        "guard", "turn_to", "walk_to",
        "route_candidates", "route_clearance",
    ):
        print(item)

print("\n=== CRITICAL SOURCE EXCERPT ===")
print(EXCERPT.read_text()[-25000:])

print("\n=== AUDIT OUTPUTS ===")
print("Architecture:", OUTPUT)
print("Source excerpt:", EXCERPT)
print("Simulation runs: 0")
print("Controller changes: 0")

print(
    "CONTROLLER_ARCHITECTURE_AUDIT:",
    "PASS" if all(assertions.values()) else "REVIEW_REQUIRED"
)
