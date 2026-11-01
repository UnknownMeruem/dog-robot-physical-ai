from pathlib import Path
import importlib.util
import csv
import math

import mujoco
import numpy as np
from ultralytics import YOLO

# Gate 2の歩行関数を再利用
source = Path(
    "../gate2_integrated_soft_stop_v01/integrated_trial.py"
).resolve()
spec = importlib.util.spec_from_file_location("gate2_gait", source)
gait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gait)

scene = Path("../gate3_ai_depth_v01/far_scene.xml").resolve()
output = Path("gate3_results/ai_walk")
output.mkdir(parents=True, exist_ok=True)

model = mujoco.MjModel.from_xml_path(str(scene))
data = mujoco.MjData(model)
ai = YOLO("yolo26n.pt")

LEGS = (
    ("front", "left"),
    ("front", "right"),
    ("hind", "left"),
    ("hind", "right"),
)

offsets = {
    f"{s}_{side}": (
        0.0 if (s, side) in (
            ("front", "left"), ("hind", "right")
        ) else 0.5
    )
    for s, side in LEGS
}

ids = {}
for s, side in LEGS:
    tag = f"{s}_{side}"
    for joint in ("hip", "knee"):
        name = f"{tag}_{joint}_drive"
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        if aid < 0:
            raise RuntimeError(f"Actuator not found: {name}")
        ids[name] = aid

# 他の関節には初期姿勢保持指令
for i in range(model.nu):
    jid = int(model.actuator_trnid[i, 0])
    if 0 <= jid < model.njnt:
        qadr = int(model.jnt_qposadr[jid])
        lo, hi = model.actuator_ctrlrange[i]
        data.ctrl[i] = float(np.clip(data.qpos[qadr], lo, hi))

REST_Z = -0.163
STRIDE = 0.045
LIFT = 0.012
RATE = 0.8
STOP_DEPTH = 0.70
MAX_WALK_TIME = 12.0
SCAN_INTERVAL = 0.5
STOP_JOINT_RATE = 2.0

stage = "stand"
stage_start = 0.0
walk_start = None
next_scan = 2.0
decision = "UNKNOWN_STOP"
last_depth = float("nan")
rows = []

def set_legs(moving, elapsed):
    stride = (
        STRIDE * gait.smoothstep(elapsed / 0.5)
        if moving else 0.0
    )
    lift = (
        LIFT * gait.smoothstep(elapsed / 1.0)
        if moving else 0.0
    )

    for s, side in LEGS:
        tag = f"{s}_{side}"

        if moving:
            x, z = gait.trajectory(
                elapsed, offsets[tag], stride, lift, RATE
            )
        else:
            x, z = 0.0, REST_Z

        hip, knee = gait.ik(x, z)

        for joint, target in (
            ("hip", hip), ("knee", knee)
        ):
            aid = ids[f"{tag}_{joint}_drive"]
            lo, hi = model.actuator_ctrlrange[aid]
            target = float(np.clip(target, lo, hi))

            if stage in ("stopping", "hold"):
                max_delta = STOP_JOINT_RATE * model.opt.timestep
                old = float(data.ctrl[aid])
                target = float(np.clip(
                    target, old - max_delta, old + max_delta
                ))

            data.ctrl[aid] = target

def posture_ok():
    # 胴体の姿勢を回転行列から確認
    rot = np.zeros(9)
    mujoco.mju_quat2Mat(rot, data.qpos[3:7])
    R = rot.reshape(3, 3)

    tilt = math.degrees(
        math.acos(float(np.clip(R[2, 2], -1.0, 1.0)))
    )

    return float(data.qpos[2]) >= 0.16 and tilt <= 10.0

def perceive(renderer):
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera="dog_eye")
    image = renderer.render().copy()

    # RGB画像のAI検出
    result = ai.predict(
        source=image,
        device="cpu",
        conf=0.25,
        verbose=False,
    )[0]

    people = [
        box for box in result.boxes
        if result.names[int(box.cls.item())] == "person"
    ]

    if not people:
        return "UNKNOWN_STOP", float("nan"), 0

    best = max(
        people, key=lambda box: float(box.conf.item())
    )

    # 同じカメラ姿勢から深度取得
    renderer.enable_depth_rendering()
    renderer.update_scene(data, camera="dog_eye")
    depth_map = renderer.render().copy()

    x1, y1, x2, y2 = best.xyxy[0].tolist()
    h, w = depth_map.shape
    cx = int(np.clip((x1 + x2) / 2, 0, w - 1))
    cy = int(np.clip((y1 + y2) / 2, 0, h - 1))

    depth = float(depth_map[cy, cx])

    if not math.isfinite(depth) or depth <= 0:
        return "UNKNOWN_STOP", depth, len(people)

    if depth <= STOP_DEPTH:
        return "STOP_NEAR_PERSON", depth, len(people)

    return "FORWARD_ALLOWED", depth, len(people)

def transition(new_stage):
    global stage, stage_start
    print(f"t={data.time:.3f}s: {stage} -> {new_stage}")
    stage = new_stage
    stage_start = float(data.time)

mujoco.mj_forward(model, data)

with mujoco.Renderer(
    model, width=640, height=360
) as renderer:

    while data.time < 30.0:
        t = float(data.time)

        # 姿勢監視はAIとは独立
        if not posture_ok():
            decision = "EMERGENCY_STOP"
            if stage not in ("stopping", "hold"):
                transition("stopping")

        if stage == "stand" and t >= 2.0:
            decision, last_depth, count = perceive(renderer)
            print(
                f"Initial AI: {decision}, "
                f"depth={last_depth:.3f}m, people={count}"
            )
            if decision == "FORWARD_ALLOWED":
                transition("walking")
                walk_start = t
                next_scan = t + SCAN_INTERVAL
            else:
                transition("stopping")

        elif stage == "walking":
            if t >= next_scan:
                decision, last_depth, count = perceive(renderer)
                print(
                    f"t={t:.2f}s AI={decision}, "
                    f"depth={last_depth:.3f}m, people={count}"
                )
                next_scan = t + SCAN_INTERVAL

                if decision != "FORWARD_ALLOWED":
                    transition("stopping")

            if (
                stage == "walking"
                and t - walk_start >= MAX_WALK_TIME
            ):
                decision = "TIMEOUT_STOP"
                transition("stopping")

        elif stage == "stopping" and t - stage_start >= 1.0:
            transition("hold")

        elif stage == "hold" and t - stage_start >= 2.0:
            break

        moving = stage == "walking"
        elapsed = t - stage_start
        set_legs(moving, elapsed)

        mujoco.mj_step(model, data)

        rows.append({
            "time_s": float(data.time),
            "stage": stage,
            "x_m": float(data.qpos[0]),
            "height_m": float(data.qpos[2]),
            "depth_m": last_depth,
            "ai_decision": decision,
        })

with (output / "ai_walk.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n=== Gate 3 AI walking result ===")
print("Final stage:", stage)
print("AI decision:", decision)
print("Final depth:", last_depth)
print("Final X:", float(data.qpos[0]))
print("Final height:", float(data.qpos[2]))
print("CSV:", output / "ai_walk.csv")
print("Exploratory simulation; not hardware-safe.")
