from pathlib import Path

import mujoco
from PIL import Image

model_path = Path("../gate3_ai_vision_v01/dog_with_camera.xml")
output_path = Path("../gate3_ai_vision_v01/dog_eye.png")

model = mujoco.MjModel.from_xml_path(str(model_path.resolve()))
data = mujoco.MjData(model)

camera_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_CAMERA, "dog_eye"
)
if camera_id < 0:
    raise RuntimeError("dog_eye camera not found")

# 立位の初期状態を作るため、必要に応じてモデルの
# デフォルト関節姿勢をそのまま使用します。
mujoco.mj_forward(model, data)

with mujoco.Renderer(model, width=640, height=360) as renderer:
    renderer.update_scene(data, camera="dog_eye")
    rgb = renderer.render().copy()

Image.fromarray(rgb).save(output_path)

print("Camera:", "dog_eye")
print("Image resolution:", rgb.shape[1], "x", rgb.shape[0])
print("Saved:", output_path.resolve())
