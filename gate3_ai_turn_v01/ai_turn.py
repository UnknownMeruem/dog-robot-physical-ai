from pathlib import Path
import importlib.util
import csv
import math

import mujoco
import numpy as np
from PIL import Image
from ultralytics import YOLO

folder = Path("../gate3_ai_turn_v01").resolve()
scene = Path(
    "../gate3_ai_head_tracking_offset_v01/offset_scene.xml"
).resolve()

# Gate 2の歩行関数を読み込む
source = Path(
    "../gate2_integrated_soft_stop_v01/integrated_trial.py"
).resolve()
spec = importlib.util.spec_from_file_location("gate2_gait", source)
gait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gait)

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

act = {}
for s, side in LEGS:
    tag = f"{s}_{side}"
    for joint in ("hip", "knee"):
        name = f"{tag}_{joint}_drive"
        idx = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        if idx < 0:
            raise RuntimeError(f"Missing actuator: {name}")
        act[name] = idx

neck_joint = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "neck_yaw"
)
neck_drive = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "neck_yaw_drive"
)
neck_qpos = int(model.jnt_qposadr[neck_joint])

if neck_joint < 0 or neck_drive < 0:
    raise RuntimeError("Neck joint/actuator not found")

# 初期姿勢を各位置制御アクチュエータで保持
for i in range(model.nu):
    jid = int(model.actuator_trnid[i, 0])
    if 0 <= jid < model.njnt:
        qadr = int(model.jnt_qposadr[jid])
        lo, hi = model.actuator_ctrlrange[i]
        data.ctrl[i] = float(
            np.clip(data.qpos[qadr], lo, hi)
        )

def detect_person(renderer, filename):
    renderer.disable_depth_rendering()
    renderer.update_scene(data, camera="dog_eye")
    rgb = renderer.render().copy()

    Image.fromarray(rgb).save(folder / filename)

    result = ai.predict(
        source=rgb,
        device="cpu",
        conf=0.25,
        verbose=False,
    )[0]

    people = []
    for box in result.boxes:
        if result.names[int(box.cls.item())] != "person":
            continue

        x1, y1, x2, y2 = box.xyxy[0].tolist()

        people.append({
            "confidence": float(box.conf.item()),
            "center_error_px": abs(
                (x1 + x2) / 2 - rgb.shape[1] / 2
            ),
        })

    if not people:
        return None, 0

    best = max(
        people, key=lambda p: p["confidence"]
    )
    return best, len(people)

def set_legs(stage, elapsed):
    turning = stage == "turn"

    stride = (
        0.045 * gait.smoothstep(elapsed / 0.5)
        if turning else 0.0
    )
    lift = (
        0.012 * gait.smoothstep(elapsed / 1.0)
        if turning else 0.0
    )

    for s, side in LEGS:
        tag = f"{s}_{side}"

        if turning:
            # 左旋回＝Yaw負、右旋回＝Yaw正
            direction = (
                -1.0 if side == "left" else 1.0
            )
            if selected_angle > 0:
                direction *= -1.0

            x, z = gait.trajectory(
                elapsed,
                offsets[tag],
                stride * direction,
                lift,
                0.8,
            )
        else:
            x, z = 0.0, -0.163

        hip, knee = gait.ik(x, z)

        for joint, val in (
            ("hip", hip), ("knee", knee)
        ):
            idx = act[f"{tag}_{joint}_drive"]
            lo, hi = model.actuator_ctrlrange[idx]
            target = float(np.clip(val, lo, hi))

            # 停止中はGate 2と同じ指令変化率制限
            if stage in ("stopping", "hold"):
                delta = 2.0 * model.opt.timestep
                previous = float(data.ctrl[idx])
                target = float(np.clip(
                    target,
                    previous - delta,
                    previous + delta,
                ))

            data.ctrl[idx] = target

def yaw_deg():
    yaw, _, _ = gait.orientation(data)
    return math.degrees(yaw)

def posture_ok():
    rotation = np.zeros(9)
    mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
    R = rotation.reshape(3, 3)

    tilt = math.degrees(
        math.acos(float(np.clip(R[2, 2], -1, 1)))
    )
    return (
        float(data.qpos[2]) >= 0.16
        and tilt <= 12.0
    )

# -------- AIによる方向選択 --------
scan_results = []

with mujoco.Renderer(
    model, width=640, height=360
) as renderer:

    print("=== AI direction scan ===")

    # 今回の探索は頭部姿勢を直接切り替える簡易方式
    for angle in (-20, 0, 20):
        data.qpos[neck_qpos] = math.radians(angle)
        mujoco.mj_forward(model, data)

        person, count = detect_person(
            renderer, f"scan_{angle:+d}.png"
        )

        if person:
            error = person["center_error_px"]
            confidence = person["confidence"]
        else:
            error = float("inf")
            confidence = 0.0

        scan_results.append({
            "angle": angle,
            "count": count,
            "confidence": confidence,
            "center_error_px": error,
        })

        print(
            f"neck={angle:+d}deg "
            f"people={count} "
            f"error={error:.1f}px "
            f"conf={confidence:.3f}"
        )

    candidates = [
        r for r in scan_results if r["count"] > 0
    ]

    # 人物なしなら旋回せず終了
    if not candidates:
        print("AI decision: UNKNOWN_STOP")
        raise SystemExit(0)

    best = min(
        candidates,
        key=lambda r: r["center_error_px"]
    )
    selected_angle = best["angle"]

    print("\nAI selected angle:", selected_angle)

    # 探索による姿勢の直接変更をリセット
    mujoco.mj_resetData(model, data)

    for i in range(model.nu):
        jid = int(model.actuator_trnid[i, 0])
        if 0 <= jid < model.njnt:
            qadr = int(model.jnt_qposadr[jid])
            lo, hi = model.actuator_ctrlrange[i]
            data.ctrl[i] = float(
                np.clip(data.qpos[qadr], lo, hi)
            )

    stage = "stand"
    stage_start = 0.0
    start_yaw = None
    stop_reason = ""
    rows = []

    while data.time < 25.0:
        t = float(data.time)
        elapsed = t - stage_start

        if not posture_ok() and stage not in (
            "stopping", "hold"
        ):
            stop_reason = "POSTURE_STOP"
            stage = "stopping"
            stage_start = t
            elapsed = 0.0

        if stage == "stand" and elapsed >= 2.0:
            if selected_angle == 0:
                stage = "hold"
                stop_reason = "ALREADY_ALIGNED"
            else:
                stage = "turn"
                stop_reason = "TURNING"
                start_yaw = yaw_deg()

            stage_start = t
            elapsed = 0.0
            print(f"t={t:.3f}s -> {stage}")

        elif stage == "turn":
            yaw_change = (
                (yaw_deg() - start_yaw + 180) % 360
            ) - 180

            target_magnitude = max(
                0.0, abs(selected_angle) - 1.5
            )

            reached = (
                yaw_change <= -target_magnitude
                if selected_angle < 0
                else yaw_change >= target_magnitude
            )

            if reached or elapsed >= 15.0:
                stop_reason = (
                    "ANGLE_REACHED"
                    if reached else "TURN_TIMEOUT"
                )
                stage = "stopping"
                stage_start = t
                elapsed = 0.0
                print(
                    f"t={t:.3f}s -> stopping "
                    f"yaw_change={yaw_change:+.2f}deg "
                    f"reason={stop_reason}"
                )

        elif stage == "stopping" and elapsed >= 1.0:
            stage = "hold"
            stage_start = t
            elapsed = 0.0

        elif stage == "hold" and elapsed >= 2.0:
            break

        # 正面を見るように首を指令
        data.ctrl[neck_drive] = 0.0
        set_legs(stage, elapsed)

        mujoco.mj_step(model, data)

        rows.append({
            "time_s": float(data.time),
            "stage": stage,
            "yaw_deg": yaw_deg(),
            "x_m": float(data.qpos[0]),
            "y_m": float(data.qpos[1]),
            "height_m": float(data.qpos[2]),
        })

    # 旋回後に正面を再撮影
    final_person, final_count = detect_person(
        renderer, "after_turn.png"
    )

with (folder / "ai_turn.csv").open(
    "w", newline=""
) as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)

print("\n=== Gate 3 AI Turn Result ===")
print("Selected angle:", selected_angle)
print("Stop reason:", stop_reason)
print("Final stage:", stage)

if start_yaw is not None:
    final_turn = (
        (yaw_deg() - start_yaw + 180) % 360
    ) - 180
    print("Actual body turn:", f"{final_turn:+.2f} deg")

print("Persons after turn:", final_count)
if final_person:
    print(
        "Center error after turn:",
        f"{final_person['center_error_px']:.1f} px"
    )
else:
    print("Center error after turn: unavailable")

print("Final height:", float(data.qpos[2]))
print("CSV:", folder / "ai_turn.csv")
print("Exploratory simulation only.")
