from pathlib import Path
import xml.etree.ElementTree as ET

from PIL import Image, ImageOps
from ultralytics import YOLO
import mujoco
import numpy as np

folder = Path("../gate3_person_vision_v01").resolve()
source_xml = Path("../gate3_ai_vision_v01/dog_with_camera.xml").resolve()
photo = Path("bus.jpg").resolve()

if not photo.exists():
    raise FileNotFoundError(
        "bus.jpg が見つかりません。先ほどのYOLOサンプルを再実行してください。"
    )

# 1. サンプル写真から人物を探す
ai = YOLO("yolo26n.pt")
result = ai.predict(str(photo), device="cpu", verbose=False)[0]

people = []
for box in result.boxes:
    name = result.names[int(box.cls.item())]
    if name == "person":
        xyxy = box.xyxy[0].cpu().numpy().tolist()
        x1, y1, x2, y2 = xyxy
        people.append(((x2-x1)*(y2-y1), xyxy))

if not people:
    raise RuntimeError("サンプル画像に人物が見つかりません")

_, (x1, y1, x2, y2) = max(people)

image = Image.open(photo).convert("RGB")
w, h = image.size

# 人物の周囲に少し余白を追加
margin_x = (x2-x1)*0.12
margin_y = (y2-y1)*0.08
crop = image.crop((
    max(0, int(x1-margin_x)),
    max(0, int(y1-margin_y)),
    min(w, int(x2+margin_x)),
    min(h, int(y2+margin_y)),
))

poster = ImageOps.fit(crop, (384, 512))
poster_path = folder / "person_poster.png"
poster.save(poster_path)

# 2. 既存のカメラ付きモデルをコピーして編集
tree = ET.parse(source_xml)
root = tree.getroot()

asset = root.find("asset")
if asset is None:
    asset = ET.Element("asset")
    root.insert(0, asset)

ET.SubElement(asset, "texture", {
    "name": "person_photo",
    "type": "2d",
    "file": str(poster_path),
})

ET.SubElement(asset, "material", {
    "name": "person_poster_material",
    "texture": "person_photo",
    "rgba": "1 1 1 1",
    "specular": "0",
})

world = root.find("worldbody")
if world is None:
    raise RuntimeError("worldbody not found")

# カメラの正面に縦長の写真看板を置く
ET.SubElement(world, "geom", {
    "name": "person_poster_board",
    "type": "box",
    "pos": "-0.85 0 0.32",
    "size": "0.005 0.23 0.30",
    "material": "person_poster_material",
    "contype": "0",
    "conaffinity": "0",
})

output_xml = folder / "dog_person_scene.xml"
ET.indent(tree, space="  ")
tree.write(output_xml, encoding="unicode")

# 3. 犬のカメラで撮影
model = mujoco.MjModel.from_xml_path(str(output_xml))
data = mujoco.MjData(model)
mujoco.mj_forward(model, data)

with mujoco.Renderer(model, width=640, height=360) as renderer:
    renderer.update_scene(data, camera="dog_eye")
    rgb = renderer.render().copy()

camera_path = folder / "dog_sees_person.png"
Image.fromarray(rgb).save(camera_path)

# 4. 撮影画像をAIに入力
print("\n=== 犬のカメラによるAI認識 ===")
detections = ai.predict(
    str(camera_path),
    device="cpu",
    conf=0.15,
    verbose=False
)[0]

person_count = 0

for box in detections.boxes:
    name = detections.names[int(box.cls.item())]
    conf = float(box.conf.item())
    print(f"{name}: {conf:.3f}")
    if name == "person":
        person_count += 1

if person_count:
    print("行動命令: STOP")
    print("理由: カメラ画像に人物を検出")
else:
    print("行動命令: UNKNOWN_STOP")
    print("理由: 人物を検出できない")

print("\n撮影画像:", camera_path)
print("写真看板:", poster_path)
print("モデル:", output_xml)
print("※ 脚は動かしていません")
