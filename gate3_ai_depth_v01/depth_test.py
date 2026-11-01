from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco
import numpy as np
from PIL import Image
from ultralytics import YOLO

folder = Path("../gate3_ai_depth_v01").resolve()
source = Path(
    "../gate3_neck_clearance_v01/neck_contact_fixed.xml"
).resolve()

ai = YOLO("yolo26n.pt")

for distance_label, board_x in (
    ("near", -0.85),
    ("far", -1.35),
):
    tree = ET.parse(source)
    root = tree.getroot()

    board = root.find(
        ".//geom[@name='person_poster_board']"
    )
    if board is None:
        raise RuntimeError("人物看板が見つかりません")

    board.set("pos", f"{board_x} 0 0.32")

    xml_path = folder / f"{distance_label}_scene.xml"
    tree.write(xml_path, encoding="unicode")

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    with mujoco.Renderer(
        model, width=640, height=360
    ) as renderer:
        renderer.update_scene(data, camera="dog_eye")
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="dog_eye")
        depth = renderer.render().copy()

    image_path = folder / f"{distance_label}.png"
    Image.fromarray(rgb).save(image_path)

    result = ai.predict(
        source=str(image_path),
        device="cpu",
        conf=0.25,
        verbose=False,
    )[0]

    detections = [
        box for box in result.boxes
        if result.names[int(box.cls.item())] == "person"
    ]

    print(f"\n=== {distance_label} ===")
    print("看板のX座標:", board_x)
    print("人物検出数:", len(detections))

    if not detections:
        print("人物が見つからないため距離判断を保留")
        print("行動判断: UNKNOWN_STOP")
        continue

    best = max(
        detections,
        key=lambda b: float(b.conf.item())
    )

    x1, y1, x2, y2 = best.xyxy[0].tolist()

    # 検出枠の中央点の深度を取得
    cx = int(np.clip((x1 + x2) / 2, 0, 639))
    cy = int(np.clip((y1 + y2) / 2, 0, 359))

    distance = float(depth[cy, cx])

    print("検出信頼度:", round(float(best.conf.item()), 3))
    print("検出中心:", (cx, cy))
    print("中心点の深度:", f"{distance:.3f} m")

    if not np.isfinite(distance) or distance <= 0:
        print("行動判断: UNKNOWN_STOP")
    elif distance < 0.60:
        print("行動判断: STOP_NEAR_PERSON")
    else:
        print("行動判断: DISTANCE_CHECKED")

    print("※ 移動指令は出していません")

print("\n距離認識実験が終了しました")
