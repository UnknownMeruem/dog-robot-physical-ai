from pathlib import Path
from ultralytics import YOLO

folder = Path("../gate3_ai_vision_v01")
model = YOLO("yolo26n.pt")

sources = {
    "dog_eye": str(folder / "dog_eye.png"),
    "sample": "https://ultralytics.com/images/bus.jpg",
}

for name, source in sources.items():
    print(f"\n=== {name} ===")

    results = model.predict(
        source=source,
        device="cpu",
        conf=0.25,
        save=True,
        project="gate3_results/ai_vision",
        name=name,
        exist_ok=True,
        verbose=False,
    )

    for result in results:
        boxes = result.boxes

        if len(boxes) == 0:
            print("検出された物体: なし")
            continue

        for box in boxes:
            class_id = int(box.cls.item())
            confidence = float(box.conf.item())
            print(
                f"物体: {result.names[class_id]}, "
                f"信頼度: {confidence:.3f}"
            )

print("\nAI推論が完了しました")
