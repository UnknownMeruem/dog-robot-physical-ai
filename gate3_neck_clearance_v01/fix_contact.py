from pathlib import Path
import xml.etree.ElementTree as ET

source = Path("../gate3_ai_scan_v01/scan_scene.xml")
output = Path("../gate3_neck_clearance_v01/neck_contact_fixed.xml")

tree = ET.parse(source)
root = tree.getroot()

contact = root.find("contact")
if contact is None:
    contact = ET.Element("contact")
    worldbody_index = next(
        i for i, child in enumerate(root)
        if child.tag == "worldbody"
    )
    root.insert(worldbody_index, contact)

# 頭部と胴体の内部接触だけを除外する。
# 足と床、頭と外部障害物の接触はそのまま。
ET.SubElement(
    contact,
    "exclude",
    {
        "body1": "torso",
        "body2": "head",
    },
)

ET.indent(tree, space="  ")
tree.write(output, encoding="unicode")

print("Created:", output)
