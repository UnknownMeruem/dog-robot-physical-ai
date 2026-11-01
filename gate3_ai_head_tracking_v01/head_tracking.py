from pathlib import Path
import csv
import math

import mujoco
import numpy as np
from PIL import Image
from ultralytics import YOLO

folder = Path("../gate3_ai_head_tracking_v01").resolve()
model_path = Path(
    "../gate3_neck_clearance_v01/neck_contact_fixed.xml"
).resolve()

model = mujoco.MjModel.from_xml_path(str(model_path))
data = mujoco.MjData(model)

jid = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "neck_yaw"
)
aid = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_ACTUATOR, "neck_yaw_drive"
)
if jid < 0 or aid < 0:
    raise RuntimeError("首関節または駆動アクチュエータがありません")

qadr = int(model.jnt_qposadr[jid])
root_pose = data.qpos[:7].copy()

# その他の位置制御アクチュエータは初期姿勢を保持
for i in range(model.nu):
    joint = int(model.actuator_trnid[i, 0])
    if 0 <= joint < model.njnt:
        qa = int(model.jnt_qposadr[joint])
        lo, hi = model.actuator_ctrlrange[i]
        data.ctrl[i] = float(np.clip(data.qpos[qa], lo, hi))

ai = YOLO("yolo26n.pt")

def drive_neck(degrees, seconds=1.5):
    lo, hi = model.actuator_ctrlrange[aid]
    data.ctrl[aid] = float(np.clip(math.radians(degrees), lo, hi))

    for _ in range(round(seconds / model.opt.timestep)):
        mujoco.mj_step(model, data)
        # 視覚実験用の胴体固定。歩行中の動作ではない。
        data.qpos[:7] = root_pose
        data.qvel[:6] = 0.0
        mujoco.mj_forward(model, data)

    return math.degrees(float(data.qpos[qadr]))

results = []

with mujoco.Renderer(model, width=640, height=360) as renderer:
    for target in (-20, 0, 20):
        actual = drive_neck(target)

        renderer.update_scene(data, camera="dog_eye")
        image = renderer.render().copy()
        image_path = folder / f"scan_{target:+d}.png"
        Image.fromarray(image).save(image_path)

        detection = ai.predict(
            source=str(image_path),
            conf=0.25,
            device="cpu",
            verbose=False,
        )[0]

        people = []
        for box in detection.boxes:
            if detection.names[int(box.cls.item())] != "person":
                continue

            x1, y1, x2, y2 = box.xyxy[0].tolist()
            center_x = (x1 + x2) / 2
            error_px = abs(center_x - image.shape[1] / 2)
            people.append({
                "confidence": float(box.conf.item()),
                "center_error_px": error_px,
            })

        # 複数検出時は最も信頼度の高い人物を使用する
        person = max(
            people, key=lambda p: p["confidence"],
            default=None,
        )

        row = {
            "target_deg": target,
            "actual_deg": round(actual, 3),
            "person_count": len(people),
            "confidence": person["confidence"] if person else 0,
            "center_error_px": (
                person["center_error_px"] if person else float("inf")
            ),
        }
        results.append(row)

        print(f"\n首の角度: {actual:+.2f}°")
        print(f"人物検出数: {len(people)}")
        if person:
            print(f"選択した人物の信頼度: {person['confidence']:.3f}")
            print(f"画面中央からのずれ: {person['center_error_px']:.1f}px")
        else:
            print("人物なし")

    candidates = [
        r for r in results if r["person_count"] > 0
    ]

    if candidates:
        best = min(candidates, key=lambda r: r["center_error_px"])
        actual = drive_neck(best["target_deg"])

        renderer.update_scene(data, camera="dog_eye")
        final_image = renderer.render().copy()
        Image.fromarray(final_image).save(folder / "selected_view.png")

        print("\n=== AI認識に基づく首の向き ===")
        print(f"選択角度: {best['target_deg']:+d}°")
        print(f"最終実角度: {actual:+.2f}°")
        print(f"画面中央からのずれ: {best['center_error_px']:.1f}px")
        print("行動命令: LOOK_AT_PERSON")
    else:
        actual = drive_neck(0)
        print("\n=== AI認識に基づく首の向き ===")
        print("人物が検出できなかったため正面に戻ります")
        print("行動命令: UNKNOWN_STOP")

with (folder / "tracking_results.csv").open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(results[0]))
    writer.writeheader()
    writer.writerows(results)

print("\n結果保存先:", folder)
print("※ 胴体固定。脚は動かしていません。")
