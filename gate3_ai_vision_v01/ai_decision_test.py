from pathlib import Path
from ultralytics import YOLO

folder = Path("../gate3_ai_vision_v01")
model = YOLO("yolo26n.pt")

images = {
    "dog_eye": folder / "dog_eye.png",
    "sample": Path("bus.jpg"),
}

def decide(image):
    result = model.predict(
        source=str(image),
        device="cpu",
        conf=0.25,
        verbose=False,
    )[0]

    persons = []

    for box in result.boxes:
        class_id = int(box.cls.item())
        name = result.names[class_id]

        if name == "person":
            persons.append(float(box.conf.item()))

    if persons:
        return "STOP", f"人物を{len(persons)}人検出"
    else:
        return "UNKNOWN_STOP", "人物を確認できない"

for name, path in images.items():
    if not path.exists():
        print(f"{name}: 画像がありません: {path}")
        continue

    command, reason = decide(path)

    print(f"\n=== {name} ===")
    print("AI判断:", reason)
    print("行動命令:", command)

print("\n※ 脚は動かしていません。判断のみのテストです。")
