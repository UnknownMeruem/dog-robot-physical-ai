from pathlib import Path
import csv
import numpy as np

source = Path("../gate3_depth_pointcloud_v01")

scenes = {
    "with_box": source / "with_box_points.csv",
    "without_box": source / "without_box_points.csv",
}

# ロボット胴体基準：前方が +forward
# 横方向は -lateral / +lateral
FORWARD_MAX = 1.10
HEIGHT_MIN = 0.06
HEIGHT_MAX = 0.45

# 候補経路の中心位置。まだ走行許可は出さない。
LANES = {
    "right_side": (-0.65, -0.25),
    "center": (-0.25, 0.25),
    "left_side": (0.25, 0.65),
}

for name, filename in scenes.items():
    if not filename.exists():
        raise FileNotFoundError(filename)

    with filename.open() as f:
        rows = list(csv.DictReader(f))

    points = np.array([
        [
            float(r["forward_m"]),
            float(r["lateral_m"]),
            float(r["height_m"]),
        ]
        for r in rows
    ], dtype=float)

    elevated = (
        (points[:, 2] >= HEIGHT_MIN)
        & (points[:, 2] <= HEIGHT_MAX)
        & (points[:, 0] > 0)
        & (points[:, 0] < FORWARD_MAX)
    )

    print(f"\n===== {name} =====")
    print("有効な3D点:", len(points))

    for lane, (left, right) in LANES.items():
        region = (
            elevated
            & (points[:, 1] >= left)
            & (points[:, 1] < right)
        )
        selected = points[region]

        if len(selected):
            nearest = selected[:, 0].min()
            print(
                f"{lane}: {len(selected)} points, "
                f"nearest={nearest:.3f}m"
            )
        else:
            print(f"{lane}: 0 points")

    print("\n前方距離別の3D障害物分布")
    print("記号: #=候補3点以上  +=1〜2点  .=0点")
    print("横方向: right(-) → center → left(+)")

    lateral_bins = np.linspace(-0.75, 0.75, 11)
    forward_bins = np.linspace(0.0, 1.10, 12)

    for i in reversed(range(len(forward_bins) - 1)):
        marks = []

        for j in range(len(lateral_bins) - 1):
            mask = (
                elevated
                & (points[:, 0] >= forward_bins[i])
                & (points[:, 0] < forward_bins[i + 1])
                & (points[:, 1] >= lateral_bins[j])
                & (points[:, 1] < lateral_bins[j + 1])
            )

            count = int(mask.sum())
            if count >= 3:
                marks.append("#")
            elif count:
                marks.append("+")
            else:
                marks.append(".")

        print(
            f"{forward_bins[i]:.1f}-"
            f"{forward_bins[i + 1]:.1f}m "
            + "".join(marks)
        )

print("\n注意: 点がない領域も通行安全とは限りません。")
print("未観測領域やロボット幅は別途評価が必要です。")
