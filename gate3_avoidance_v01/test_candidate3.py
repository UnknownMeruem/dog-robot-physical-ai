from pathlib import Path
import math
import numpy as np
import mujoco

scene = Path("../gate3_obstacle_stop_v01/obstacle_scene.xml")
model = mujoco.MjModel.from_xml_path(str(scene.resolve()))
data = mujoco.MjData(model)

box_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "blocking_box"
)
torso_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "torso"
)

if box_id < 0 or torso_id < 0:
    raise RuntimeError("box or torso not found")

# このモデルは胴体のfreejointが先頭にある前提。
if (
    model.njnt == 0
    or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE
    or model.jnt_qposadr[0] != 0
    or model.jnt_bodyid[0] != torso_id
):
    raise RuntimeError("freejoint configuration is unexpected")

waypoints = np.array([
    [0.00, 0.00],
    [-0.35, 0.80],
    [-1.10, 0.80],
    [-1.10, 0.85],
])

# 前回のモデルの立位高さを使用した近似検査
STANDING_Z = 0.2111

samples = []
for a, b in zip(waypoints[:-1], waypoints[1:]):
    direction = b - a
    distance = float(np.linalg.norm(direction))
    steps = max(2, math.ceil(distance / 0.01) + 1)

    # この犬の前方はローカル-X
    yaw = math.atan2(-direction[1], -direction[0])

    for t in np.linspace(0.0, 1.0, steps):
        xy = a + t * direction
        samples.append((xy[0], xy[1], yaw))

hits = []
minimum_geom_distance = float("inf")
checked_contacts = 0

for i, (x, y, yaw) in enumerate(samples):
    data.qpos[:] = model.qpos0
    data.qvel[:] = 0

    data.qpos[0] = x
    data.qpos[1] = y
    data.qpos[2] = STANDING_Z

    data.qpos[3:7] = [
        math.cos(yaw / 2),
        0.0,
        0.0,
        math.sin(yaw / 2),
    ]

    mujoco.mj_forward(model, data)

    # 静的姿勢で衝突判定
    mujoco.mj_collision(model, data)

    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]

        if (
            int(contact.geom1) == box_id
            or int(contact.geom2) == box_id
        ):
            checked_contacts += 1
            minimum_geom_distance = min(
                minimum_geom_distance,
                float(contact.dist)
            )
            if contact.dist <= 0:
                hits.append((i, x, y, float(contact.dist)))

print("=== Candidate 3 static collision test ===")
print("Sampled poses:", len(samples))
print("Standing torso Z:", STANDING_Z)
print("Contacts involving blocking_box:", checked_contacts)
print("Penetrating/touching samples:", len(hits))

if math.isfinite(minimum_geom_distance):
    print(
        "Minimum reported contact distance:",
        f"{minimum_geom_distance:.4f}m"
    )
else:
    print("No box contact pairs reported")

if hits:
    print("\nFirst collision samples:")
    for item in hits[:10]:
        i, x, y, depth = item
        print(
            f"sample={i} XY=({x:.3f}, {y:.3f}) "
            f"contact_distance={depth:.4f}m"
        )
    print("\nResult: REJECT_COLLISION")
else:
    print("\nResult: NO_BOX_CONTACT_IN_SAMPLED_POSES")

print("\nIMPORTANT:")
print("This is a static pose test, not gait simulation.")
print("Turning at waypoints and moving legs were not tested.")
print("Not a hardware safety certification.")
