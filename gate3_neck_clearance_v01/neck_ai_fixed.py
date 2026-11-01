from pathlib import Path
import math
import csv

import mujoco
import numpy as np
from PIL import Image
from ultralytics import YOLO

folder = Path("../gate3_neck_clearance_v01").resolve()
model_path = Path("../gate3_neck_clearance_v01/neck_contact_fixed.xml").resolve()

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

yaw_joint = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "neck_yaw"
)
yaw_motor = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "neck_yaw_drive"
)

if yaw_joint < 0 or yaw_motor < 0:
    raise RuntimeError("首の関節またはアクチュエータが見つかりません")

yaw_qpos = int(model.jnt_qposadr[yaw_joint])

# 他の位置制御アクチュエータは初期関節角を保持
for i in range(model.nu):
    joint_id = int(model.actuator_trnid[i, 0])
    if 0 <= joint_id < model.njnt:
        qadr = int(model.jnt_qposadr[joint_id])
        lo, hi = model.actuator_ctrlrange[i]
        data.ctrl[i] = np.clip(data.qpos[qadr], lo, hi)

ai = YOLO("yolo26n.pt")

# 今回は首の駆動に集中するため、胴体の自由関節を固定。
# これは仮想的な試験治具で、歩行中の動作ではありません。
root_qpos = data.qpos[:7].copy()

targets = [-20, 0, 20]
hold_seconds = 1.5
rows = []

with mujoco.Renderer(model, width=640, height=360) as renderer:
    for target in targets:
        data.ctrl[yaw_motor] = math.radians(target)
        steps = round(hold_seconds / model.opt.timestep)

        for _ in range(steps):
            mujoco.mj_step(model, data)

            # 胴体の位置と向きを固定する実験用の処理
            data.qpos[:7] = root_qpos
            data.qvel[:6] = 0
            mujoco.mj_forward(model, data)

        actual_deg = math.degrees(float(data.qpos[yaw_qpos]))

        renderer.update_scene(data, camera="dog_eye")
        rgb = renderer.render().copy()

        image_path = folder / f"neck_{target:+d}.png"
        Image.fromarray(rgb).save(image_path)

        result = ai.predict(
            source=str(image_path),
            device="cpu",
            conf=0.25,
            verbose=False,
        )[0]

        person_scores = [
            float(box.conf.item())
            for box in result.boxes
            if result.names[int(box.cls.item())] == "person"
        ]

        command = "STOP" if person_scores else "UNKNOWN_STOP"

        rows.append({
            "target_deg": target,
            "actual_deg": actual_deg,
            "person_count": len(person_scores),
            "highest_confidence": max(person_scores, default=0),
            "command": command,
        })

        print(f"\n=== 首の目標角 {target:+d}° ===")
        print(f"実際の角度: {actual_deg:.2f}°")
        print(f"人物検出数: {len(person_scores)}")
        print(f"最高信頼度: {max(person_scores, default=0):.3f}")
        print(f"行動判断: {command}")
        print(f"保存画像: {image_path}")

with (folder / "neck_ai_results.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)

print("\n首のアクチュエータ駆動とAI推論が完了しました")
print("※ 胴体固定の視覚実験です。脚の歩行は行っていません。")
