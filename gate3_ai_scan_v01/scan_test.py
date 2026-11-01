from pathlib import Path
import math
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
from ultralytics import YOLO

folder = Path("../gate3_ai_scan_v01").resolve()
source_xml = Path("../gate3_person_vision_v01/dog_person_scene.xml").resolve()
output_xml = folder / "scan_scene.xml"

# 既存のシーンをコピーし、青い箱だけを除去
tree = ET.parse(source_xml)
root = tree.getroot()
world = root.find("worldbody")

for geom in list(world):
    if geom.tag == "geom" and geom.get("name") == "vision_test_box":
        world.remove(geom)

tree.write(output_xml, encoding="unicode")

model = mujoco.MjModel.from_xml_path(str(output_xml))
data = mujoco.MjData(model)

# 首の左右回転関節を特定
joint_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_JOINT, "neck_yaw"
)
if joint_id < 0:
    raise RuntimeError("neck_yaw joint not found")

qpos_id = int(model.jnt_qposadr[joint_id])

# 学習済みAI
ai = YOLO("yolo26n.pt")

# 首を3方向に向けて撮影
angles_deg = [-20, 0, 20]

with mujoco.Renderer(model, width=640, height=360) as renderer:
    for angle in angles_deg:
        # 今回は撮影実験のため首の姿勢を直接指定する。
        # 実機の首の動作速度・トルク制御とは異なる。
        data.qpos[qpos_id] = math.radians(angle)
        mujoco.mj_forward(model, data)

        renderer.update_scene(data, camera="dog_eye")
        rgb = renderer.render().copy()

        image_path = folder / f"view_{angle:+d}.png"
        Image.fromarray(rgb).save(image_path)

        result = ai.predict(
            source=str(image_path),
            device="cpu",
            conf=0.25,
            verbose=False,
        )[0]

        people = [
            box for box in result.boxes
            if result.names[int(box.cls.item())] == "person"
        ]

        print(f"\n=== 首の角度 {angle:+d}° ===")
        print("人物検出数:", len(people))

        for box in people:
            print(
                "person confidence:",
                f"{float(box.conf.item()):.3f}"
            )

        if people:
            print("行動判断: STOP")
        else:
            print("行動判断: UNKNOWN_STOP")

        print("画像:", image_path)

print("\nAIスキャン実験完了")
print("※ 今回も脚は動かしていません。")
