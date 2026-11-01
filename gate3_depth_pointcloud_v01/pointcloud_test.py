from pathlib import Path
import math
import csv

import mujoco
import numpy as np
from PIL import Image

out = Path("../gate3_depth_pointcloud_v01").resolve()

scenes = {
    "with_box": Path(
        "../gate3_obstacle_stop_v01/obstacle_scene.xml"
    ),
    "without_box": Path(
        "../gate3_obstacle_control_v01/no_box_scene.xml"
    ),
}

WIDTH, HEIGHT = 640, 360
SAMPLE_STEP = 4

# 今回の診断用の仮設定。実機の安全基準ではない。
MIN_HEIGHT = 0.06
MAX_HEIGHT = 0.45
MAX_FORWARD = 1.10
HALF_WIDTH = 0.25

summary = []

for label, xml_path in scenes.items():
    model = mujoco.MjModel.from_xml_path(
        str(xml_path.resolve())
    )
    data = mujoco.MjData(model)

    # 前回の旋回後に近い姿勢で両シーンを比較
    yaw = math.radians(-18.58)
    data.qpos[3:7] = [
        math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)
    ]
    mujoco.mj_forward(model, data)

    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "dog_eye"
    )
    if camera_id < 0:
        raise RuntimeError("dog_eye camera not found")

    with mujoco.Renderer(
        model, width=WIDTH, height=HEIGHT
    ) as renderer:
        renderer.update_scene(data, camera="dog_eye")
        rgb = renderer.render().copy()

        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="dog_eye")
        depth = renderer.render().copy()

    Image.fromarray(rgb).save(out / f"{label}_rgb.png")

    camera_pos = data.cam_xpos[camera_id].copy()
    camera_rot = data.cam_xmat[camera_id].reshape(3, 3).copy()

    # 垂直FOVから焦点距離を計算
    fovy = math.radians(float(model.cam_fovy[camera_id]))
    fy = HEIGHT / (2.0 * math.tan(fovy / 2.0))
    fx = fy

    # MuJoCo camera: local -Z forward, +Y up
    # RGB・depthは画像左上が原点
    pixels = []
    for v in range(0, HEIGHT, SAMPLE_STEP):
        for u in range(0, WIDTH, SAMPLE_STEP):
            z = float(depth[v, u])

            # 遠方背景や無効値を除外
            if not math.isfinite(z) or not 0.01 < z < 10.0:
                continue

            x_cam = (u + 0.5 - WIDTH / 2) * z / fx
            y_cam = -(v + 0.5 - HEIGHT / 2) * z / fy

            point_cam = np.array([x_cam, y_cam, -z])
            point_world = camera_pos + camera_rot @ point_cam

            # ロボットの胴体座標へ変換
            body_rot = data.xmat[1].reshape(3, 3)
            body_pos = data.xpos[1]
            point_body = body_rot.T @ (point_world - body_pos)

            # このモデルの前方は胴体 -X
            forward = -float(point_body[0])
            lateral = float(point_body[1])
            height = float(point_world[2])

            pixels.append((forward, lateral, height, z))

    points = np.array(pixels, dtype=float)
    if len(points) == 0:
        raise RuntimeError(f"{label}: 有効な3D点がありません")

    # 床面から浮いた点のみを候補にする
    elevated = (
        (points[:, 2] >= MIN_HEIGHT)
        & (points[:, 2] <= MAX_HEIGHT)
    )

    corridor = (
        elevated
        & (points[:, 0] > 0.0)
        & (points[:, 0] < MAX_FORWARD)
        & (np.abs(points[:, 1]) < HALF_WIDTH)
    )

    candidates = points[corridor]

    print(f"\n=== {label} ===")
    print("有効な3D点数:", len(points))
    print("床より高い点数:", int(elevated.sum()))
    print("前方通路内の候補点数:", len(candidates))

    if len(candidates):
        print(
            "候補の最短前方距離:",
            f"{candidates[:, 0].min():.3f}m"
        )
        print(
            "候補の高さ範囲:",
            f"{candidates[:, 2].min():.3f}〜"
            f"{candidates[:, 2].max():.3f}m"
        )
    else:
        print("前方通路内の候補点: なし")

    with (out / f"{label}_points.csv").open(
        "w", newline=""
    ) as f:
        writer = csv.writer(f)
        writer.writerow([
            "forward_m", "lateral_m",
            "height_m", "camera_depth_m"
        ])
        writer.writerows(points.tolist())

    summary.append({
        "scene": label,
        "candidate_points": len(candidates),
        "nearest_forward_m": (
            float(candidates[:, 0].min())
            if len(candidates) else ""
        ),
    })

with (out / "summary.csv").open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(summary[0])
    )
    writer.writeheader()
    writer.writerows(summary)

print("\n結果保存先:", out)
print("3D幾何の診断のみ。歩行制御は実行していません。")
