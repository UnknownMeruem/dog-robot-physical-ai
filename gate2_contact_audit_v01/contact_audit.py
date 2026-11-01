import sys
import csv
import math
import bisect
import importlib.util
from pathlib import Path
from collections import defaultdict

import mujoco
import numpy as np

base_file = Path("../gate2_integrated_soft_stop_v01/integrated_trial.py").resolve()
spec = importlib.util.spec_from_file_location("gate2_base", base_file)
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

legs = (
    "front_left",
    "front_right",
    "hind_left",
    "hind_right",
)

floor_id = None
foot_ids = None
records = []

original_step = mujoco.mj_step

def traced_step(model, data, *args, **kwargs):
    global floor_id, foot_ids

    original_step(model, data, *args, **kwargs)

    if floor_id is None:
        floor_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_GEOM, "floor"
        )
        foot_ids = {
            mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_GEOM, leg + "_foot"
            ): leg
            for leg in legs
        }
        if floor_id < 0 or any(i < 0 for i in foot_ids):
            raise RuntimeError("Missing floor or foot geom")

    for i in range(data.ncon):
        c = data.contact[i]
        if c.efc_address < 0:
            continue

        pair = (int(c.geom1), int(c.geom2))

        if floor_id not in pair:
            continue

        foot_id = pair[0] if pair[1] == floor_id else pair[1]
        if foot_id not in foot_ids:
            continue

        force = np.zeros(6)
        mujoco.mj_contactForce(model, data, i, force)

        body_id = int(model.geom_bodyid[foot_id])

        jac = np.zeros((3, model.nv))
        mujoco.mj_jac(
            model, data, jac, None,
            np.array(c.pos), body_id
        )
        velocity = jac @ data.qvel

        normal = np.array(c.frame).reshape(3, 3)[0]
        tangent = velocity - np.dot(velocity, normal) * normal

        records.append({
            "time_s": float(data.time),
            "foot": foot_ids[foot_id],
            "normal_force_N": float(max(force[0], 0)),
            "tangent_speed_mps": float(np.linalg.norm(tangent)),
        })

mujoco.mj_step = traced_step

output = Path("gate2_results/contact_audit")
output.mkdir(parents=True, exist_ok=True)

original_args = sys.argv
sys.argv = [
    str(base_file),
    "--model", "dog_quadruped.xml",
    "--stop-joint-rate", "2.0",
    "--out", str(output / "integrated.csv"),
]

try:
    base.main()
finally:
    sys.argv = original_args
    mujoco.mj_step = original_step

with (output / "integrated.csv").open(newline="") as f:
    stages = list(csv.DictReader(f))

times = [float(r["time_s"]) for r in stages]

def stage_at(t):
    idx = bisect.bisect_left(times, t)
    if idx >= len(stages):
        idx = len(stages) - 1
    return stages[idx]["stage"]

for r in records:
    r["stage"] = stage_at(r["time_s"])

with (output / "contacts.csv").open("w", newline="") as f:
    columns = [
        "time_s", "stage", "foot",
        "normal_force_N", "tangent_speed_mps"
    ]
    writer = csv.DictWriter(f, fieldnames=columns)
    writer.writeheader()
    writer.writerows(records)

print("\n=== Contact Audit ===")
print("stage             contacts   peak tangent speed   peak normal force")

groups = defaultdict(list)
for r in records:
    groups[r["stage"]].append(r)

for stage, values in groups.items():
    peak_speed = max(r["tangent_speed_mps"] for r in values)
    peak_force = max(r["normal_force_N"] for r in values)
    print(
        f"{stage:<19} {len(values):>6} "
        f"{peak_speed:>16.4f} m/s "
        f"{peak_force:>12.3f} N"
    )

print("\nContact CSV:", output / "contacts.csv")
print("NOTE: contact tangent speed is a slip proxy, not measured slip distance.")
