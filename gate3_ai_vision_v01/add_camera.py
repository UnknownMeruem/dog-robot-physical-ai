from pathlib import Path
import xml.etree.ElementTree as ET

source = Path("dog_quadruped.xml")
output = Path("../gate3_ai_vision_v01/dog_with_camera.xml")

tree = ET.parse(source)
root = tree.getroot()

head = root.find(".//body[@name='head']")
if head is None:
    raise RuntimeError("head body not found")

camera = ET.SubElement(
    head,
    "camera",
    {
        "name": "dog_eye",
        "pos": "-0.095 0 0.015",
        "xyaxes": "0 1 0 0 0 1",
        "fovy": "75",
    },
)

# 仮想カメラの視界確認用の物体。
# 衝突しないため、歩行実験の接触条件には影響しません。
worldbody = root.find("worldbody")
ET.SubElement(
    worldbody,
    "geom",
    {
        "name": "vision_test_box",
        "type": "box",
        "pos": "-0.9 0 0.12",
        "size": "0.10 0.12 0.12",
        "rgba": "0.15 0.55 0.95 1",
        "contype": "0",
        "conaffinity": "0",
    },
)

ET.indent(tree, space="  ")
tree.write(output, encoding="unicode")

print("Created:", output)
