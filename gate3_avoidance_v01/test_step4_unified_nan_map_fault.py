from pathlib import Path
import importlib.util
import math
import csv

import mujoco
import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

gait_path = (
    ROOT / "gate2_integrated_soft_stop_v01"
    / "integrated_trial.py"
)
scene_path = (
    ROOT / "gate3_obstacle_stop_v01"
    / "obstacle_scene_extra_box.xml"
)

spec = importlib.util.spec_from_file_location(
    "gate2_gait", gait_path
)
gait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gait)

model = mujoco.MjModel.from_xml_path(str(Path(r"/home/kato/physical_ai_todai/dog_robot/gate3_obstacle_stop_v01/step4_robust_nan_map_fault.xml")))
data = mujoco.MjData(model)

LEGS = (
    ("front", "left"),
    ("front", "right"),
    ("hind", "left"),
    ("hind", "right"),
)

offsets = {
    f"{s}_{side}": (
        0.0 if (s, side) in (
            ("front", "left"), ("hind", "right")
        ) else 0.5
    )
    for s, side in LEGS
}

act = {}
for s, side in LEGS:
    for joint in ("hip", "knee"):
        name = f"{s}_{side}_{joint}_drive"
        aid = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_ACTUATOR, name
        )
        if aid < 0:
            raise RuntimeError(f"Missing actuator: {name}")
        act[name] = aid

torso_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_BODY, "torso"
)
box_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "blocking_box"
)

if torso_id < 0 or box_id < 0:
    raise RuntimeError("torso or blocking_box missing")

# 初期姿勢を保持
for i in range(model.nu):
    jid = int(model.actuator_trnid[i, 0])
    if 0 <= jid < model.njnt:
        qadr = int(model.jnt_qposadr[jid])
        lo, hi = model.actuator_ctrlrange[i]
        data.ctrl[i] = float(
            np.clip(data.qpos[qadr], lo, hi)
        )

# Candidate 3の第一地点
target_xy = np.array([-0.35, 0.80])
start_xy = data.qpos[:2].copy()
direction = target_xy - start_xy

# このモデルでは前進方向がローカル-X
target_angle = math.degrees(
    math.atan2(-direction[1], -direction[0])
)

print("=== Candidate 3: first turning test ===")
print("Initial XY:", np.round(start_xy, 3))
print("First waypoint:", target_xy)
print("Requested relative turn:", f"{target_angle:+.2f}deg")

stage = "stand"
stage_start = 0.0
walk_start_xy = None
walk_direction = None
walk_start_time = None
walk_distance = 0.0
cross_track = 0.0
WALK_TARGET = 0.95
WAYPOINT_TOLERANCE = 0.42
WALK_TIMEOUT = 22.0
CROSS_TRACK_LIMIT = 0.08
start_yaw = None
stop_reason = ""
rows = []
box_contacts = 0
box_penetrations = 0
max_tilt = 0.0

def yaw_deg():
    yaw, _, _ = gait.orientation(data)
    return math.degrees(yaw)

def posture():
    R = data.xmat[torso_id].reshape(3, 3)
    tilt = math.degrees(
        math.acos(
            float(np.clip(R[2, 2], -1.0, 1.0))
        )
    )
    height = float(data.qpos[2])
    return height, tilt

def set_legs(stage, elapsed):
    turning = stage == "turn"
    walking = stage == "walk"

    stride = (
        0.045 * gait.smoothstep(elapsed / 0.5)
        if turning else 0.0
    )
    lift = (
        0.012 * gait.smoothstep(elapsed / 1.0)
        if turning else 0.0
    )

    for s, side in LEGS:
        tag = f"{s}_{side}"

        if walking:
            x, z = gait.trajectory(
                elapsed,
                offsets[tag],
                0.045 * gait.smoothstep(elapsed / 0.5),
                0.012 * gait.smoothstep(elapsed / 1.0),
                0.8,
            )
        elif turning:
            # 負のYaw方向への旋回
            direction_sign = (
                -1.0 if side == "left" else 1.0
            )
            if target_angle > 0:
                direction_sign *= -1.0

            x, z = gait.trajectory(
                elapsed,
                offsets[tag],
                stride * direction_sign,
                lift,
                0.8,
            )
        else:
            x, z = 0.0, -0.163

        hip, knee = gait.ik(x, z)

        for joint, val in (
            ("hip", hip), ("knee", knee)
        ):
            aid = act[f"{tag}_{joint}_drive"]
            lo, hi = model.actuator_ctrlrange[aid]
            target = float(np.clip(val, lo, hi))

            if stage in ("stopping", "hold"):
                max_delta = 2.0 * model.opt.timestep
                previous = float(data.ctrl[aid])
                target = float(np.clip(
                    target,
                    previous - max_delta,
                    previous + max_delta,
                ))

            data.ctrl[aid] = target

mujoco.mj_forward(model, data)

last_log_second = -1

while data.time < 45.0:
    t = float(data.time)
    elapsed = t - stage_start
    height, tilt = posture()
    max_tilt = max(max_tilt, tilt)

    if (
        (height < 0.16 or tilt > 12.0)
        and stage not in ("stopping", "hold")
    ):
        stop_reason = "POSTURE_STOP"
        stage = "stopping"
        stage_start = t
        print(f"t={t:.3f}s: POSTURE_STOP")

    if stage == "stand" and elapsed >= 2.0:
        stage = "turn"
        stage_start = t
        start_yaw = yaw_deg()
        print(f"t={t:.3f}s: stand -> turn")

    elif stage == "turn":
        yaw_change = (
            (yaw_deg() - start_yaw + 180) % 360
        ) - 180

        # 停止時の旋回量を考慮した既存の早期停止
        threshold = max(
            0.0, abs(target_angle) - 1.5
        )

        reached = (
            yaw_change <= -threshold
            if target_angle < 0
            else yaw_change >= threshold
        )

        if reached or elapsed >= 18.0:
            stop_reason = (
                "ANGLE_REACHED"
                if reached else "TURN_TIMEOUT"
            )
            stage = "stopping"
            stage_start = t
            print(
                f"t={t:.3f}s: turn -> stopping "
                f"yaw_change={yaw_change:+.2f}deg "
                f"reason={stop_reason}"
            )

    elif stage == "stopping" and elapsed >= 1.0:
        stage = "hold"
        stage_start = t
        print(f"t={t:.3f}s: stopping -> hold")

    elif stage == "hold" and elapsed >= 2.0:
        if walk_start_xy is None and stop_reason == "ANGLE_REACHED":
            # 旋回後の実際のYaw・XYから歩行方向を計算
            walk_start_xy = data.qpos[:2].copy()
            heading = math.radians(yaw_deg())
            walk_direction = np.array([
                -math.cos(heading),
                -math.sin(heading),
            ])
            walk_start_time = t
            stage = "walk"
            stage_start = t
            print(
                f"t={t:.3f}s: hold -> walk "
                f"start_xy={walk_start_xy.round(4)}"
            )
        else:
            break

    if stage == "walk":
        delta_xy = data.qpos[:2] - walk_start_xy
        walk_distance = float(
            np.dot(delta_xy, walk_direction)
        )
        cross_track = abs(float(
            walk_direction[0] * delta_xy[1]
            - walk_direction[1] * delta_xy[0]
        ))

        if cross_track > CROSS_TRACK_LIMIT:
            stop_reason = "PATH_DEVIATION_STOP"
        elif float(np.linalg.norm(data.qpos[:2] - target_xy)) <= WAYPOINT_TOLERANCE:
            stop_reason = "WAYPOINT1_REACHED"
        elif walk_distance >= WALK_TARGET:
            stop_reason = "DISTANCE_LIMIT_STOP"
        elif t - walk_start_time >= WALK_TIMEOUT:
            stop_reason = "WALK_TIMEOUT"
        elif abs(
            ((yaw_deg() - target_angle + 180) % 360) - 180
        ) > 10.0:
            stop_reason = "HEADING_DEVIATION_STOP"

        if stop_reason != "ANGLE_REACHED":
            stage = "stopping"
            stage_start = t
            print(
                f"t={t:.3f}s: walk -> stopping "
                f"distance={walk_distance:.3f}m "
                f"cross_track={cross_track:.3f}m "
                f"reason={stop_reason}"
            )

    set_legs(stage, t - stage_start)
    mujoco.mj_step(model, data)

    for i in range(data.ncon):
        contact = data.contact[i]
        if (
            int(contact.geom1) == box_id
            or int(contact.geom2) == box_id
        ):
            box_contacts += 1
            if float(contact.dist) <= 0.0:
                box_penetrations += 1

            if stage not in ("stopping", "hold"):
                stop_reason = "BOX_CONTACT_STOP"
                stage = "stopping"
                stage_start = float(data.time)
                print(
                    f"t={data.time:.3f}s: BOX_CONTACT_STOP"
                )

    height, tilt = posture()

    rows.append({
        "time_s": float(data.time),
        "stage": stage,
        "x_m": float(data.qpos[0]),
        "y_m": float(data.qpos[1]),
        "yaw_deg": yaw_deg(),
        "height_m": height,
        "tilt_deg": tilt,
        "box_contacts_total": box_contacts,
        "box_penetrations_total": box_penetrations,
    })

    second = int(data.time)
    if second > last_log_second:
        last_log_second = second
        print(
            f"t={data.time:.2f}s "
            f"stage={stage} "
            f"yaw={yaw_deg():+.2f}deg "
            f"height={height:.3f}m "
            f"tilt={tilt:.2f}deg"
        )

if not stop_reason:
    stop_reason = "SIMULATION_TIMEOUT"

output = HERE / "step4_unified_nan_map_fault_waypoint1_early_live_v3.csv"
with output.open("w", newline="") as f:
    writer = csv.DictWriter(
        f, fieldnames=list(rows[0])
    )
    writer.writeheader()
    writer.writerows(rows)

final_xy = data.qpos[:2].copy()
final_yaw = yaw_deg()

yaw_error = (
    (final_yaw - target_angle + 180) % 360
) - 180

print("\n=== Waypoint 1 Braking Result ===")
print("Walk forward distance:", f"{walk_distance:.3f}m")
print("Walk cross-track error:", f"{cross_track:.3f}m")
print("Target yaw:", f"{target_angle:+.2f}deg")
print("Final yaw:", f"{final_yaw:+.2f}deg")
print("Yaw error:", f"{yaw_error:+.2f}deg")
print("Final stage:", stage)
print("Stop reason:", stop_reason)
print("Final XY:", np.round(final_xy, 4))
print(
    "Horizontal drift:",
    f"{np.linalg.norm(final_xy-start_xy)*1000:.2f}mm"
)
print("Maximum tilt:", f"{max_tilt:.2f}deg")
print("Box contact samples:", box_contacts)
print("Box penetration samples:", box_penetrations)
waypoint_distance = float(
    np.linalg.norm(final_xy - target_xy)
)
print(
    "Distance to first waypoint:",
    f"{waypoint_distance:.3f}m"
)
if waypoint_distance <= 0.08:
    print("Final waypoint accuracy: PASS")
else:
    print("Final waypoint accuracy: FAIL")
print("CSV:", output)
print("Exploratory simulation only; not hardware-safe.")


# ===== Stage 2: turn toward waypoint 2, then walk 20cm =====

# === Depth guard setup: load only a function, not the old trial ===
import ast
from pathlib import Path as _DepthPath

_depth_source = (
    _DepthPath(__file__).resolve().parent
    / "../gate3_obstacle_stop_v01/obstacle_stop.py"
).resolve()

_depth_tree = ast.parse(_depth_source.read_text())
_depth_functions = [
    n for n in _depth_tree.body
    if isinstance(n, ast.FunctionDef)
    and n.name == "obstacle_distance"
]
if len(_depth_functions) != 1:
    raise RuntimeError("obstacle_distance function not found")

_depth_module = ast.Module(
    body=_depth_functions,
    type_ignores=[]
)
_depth_namespace = {
    "mujoco": mujoco,
    "np": np,
    "math": math,
    "model": model,
    "data": data,
}
exec(
    compile(_depth_module, str(_depth_source), "exec"),
    _depth_namespace
)
_depth_distance = _depth_namespace["obstacle_distance"]

_depth_renderer = mujoco.Renderer(
    model, width=640, height=360
)
_depth_last_time = -1e9
_depth_latest = float("nan")
_depth_samples = []
_depth_limit = 0.85

print("Depth camera guard enabled")
print("Depth stop threshold: 0.85m")
print("Depth sample interval: 0.40s")

reason2 = "NOT_STARTED"
print("\n=== Stage 2: dynamic heading and short walk ===")
waypoint2 = np.array([-1.10, 0.80])
position1 = data.qpos[:2].copy()
error1 = float(np.linalg.norm(position1 - target_xy))

if (
    stop_reason != "WAYPOINT1_REACHED"
    or not (0.37 <= error1 <= 0.44)
    or stage != "hold"
    or box_contacts != 0
    or posture()[0] < 0.16
    or posture()[1] > 12.0
):
    print("STAGE2_NOT_STARTED: waypoint 1 acceptance failed")
else:
    delta = waypoint2 - position1
    desired_yaw = math.degrees(
        math.atan2(-delta[1], -delta[0])
    )
    current_yaw = yaw_deg()
    relative_turn = (
        (desired_yaw - current_yaw + 180) % 360
    ) - 180

    # set_legs() uses target_angle to choose turn direction.
    saved_target_angle = target_angle
    target_angle = relative_turn

    print("Start XY:", np.round(position1, 4))
    print("Waypoint 2:", waypoint2)
    print("Current yaw:", f"{current_yaw:+.2f}deg")
    print("Desired yaw:", f"{desired_yaw:+.2f}deg")
    print("Turn command:", f"{relative_turn:+.2f}deg")

    stage = "turn"
    stage_start = float(data.time)
    turn_start_yaw = current_yaw
    reason2 = ""
    max_tilt2 = 0.0
    contact2 = 0
    stage2_rows = []

    def inspect_step():
        global contact2, max_tilt2, reason2, stage, stage_start
        height, tilt = posture()
        max_tilt2 = max(max_tilt2, tilt)

        if height < 0.16 or tilt > 12.0:
            if stage not in ("stopping", "hold"):
                reason2 = "POSTURE_STOP"
                stage = "stopping"
                stage_start = float(data.time)

        for ci in range(data.ncon):
            c = data.contact[ci]
            if box_id in (int(c.geom1), int(c.geom2)):
                contact2 += 1
                if stage not in ("stopping", "hold"):
                    reason2 = "BOX_CONTACT_STOP"
                    stage = "stopping"
                    stage_start = float(data.time)

        stage2_rows.append({
            "time_s": float(data.time),
            "stage": stage,
            "x_m": float(data.qpos[0]),
            "y_m": float(data.qpos[1]),
            "yaw_deg": yaw_deg(),
            "height_m": height,
            "tilt_deg": tilt,
            "box_contacts_total": contact2,
        })

    # Turning, with feedback and soft stopping
    while data.time - stage_start < 20.0:
        t = float(data.time)
        elapsed = t - stage_start

        if stage == "turn":
            actual = (
                (yaw_deg() - turn_start_yaw + 180) % 360
            ) - 180
            threshold = max(0.0, abs(relative_turn) - 1.5)
            reached = (
                actual >= threshold
                if relative_turn >= 0
                else actual <= -threshold
            )
            if reached or elapsed >= 16.0:
                reason2 = (
                    "TURN2_REACHED" if reached else "TURN2_TIMEOUT"
                )
                stage = "stopping"
                stage_start = t
                print(
                    f"Stage 2 turn stop: yaw_change={actual:+.2f}deg "
                    f"reason={reason2}"
                )

        elif stage == "stopping" and elapsed >= 1.0:
            stage = "hold"
            stage_start = t

        elif stage == "hold" and elapsed >= 1.0:
            break

        set_legs(stage, t - stage_start)
        mujoco.mj_step(model, data)
        inspect_step()

    print("Yaw after turn:", f"{yaw_deg():+.2f}deg")

    if reason2 == "TURN2_REACHED" and contact2 == 0:
        stage = "walk"
        stage_start = float(data.time)
        walk2_start_time = float(data.time)
        walk2_start_xy = data.qpos[:2].copy()
        heading = math.radians(yaw_deg())
        walk2_dir = np.array([
            -math.cos(heading), -math.sin(heading)
        ])

        while data.time - walk2_start_time < 19.0:
            t = float(data.time)
            displacement = data.qpos[:2] - walk2_start_xy
            forward2 = float(np.dot(displacement, walk2_dir))
            lateral2 = abs(float(np.linalg.det(
                np.column_stack((walk2_dir, displacement))
            )))

            if stage == "walk":
                depth_stop2 = False
                if t - _depth_last_time >= 0.40:
                    _depth_last_time = t
                    try:
                        _depth_latest = float(
                            _depth_distance(_depth_renderer)
                        )
                    except Exception as exc:
                        print(
                            "Depth sensor error:",
                            type(exc).__name__, str(exc)
                        )
                        _depth_latest = float("nan")

                    _depth_samples.append(
                        (t, _depth_latest)
                    )
                    print(
                        f"Depth t={t:.2f}s "
                        f"nearest={_depth_latest:.3f}m"
                    )

                if not math.isfinite(_depth_latest):
                    depth_stop2 = True
                    depth_reason2 = "DEPTH_SENSOR_STOP"
                elif _depth_latest < _depth_limit:
                    depth_stop2 = True
                    depth_reason2 = "DEPTH_OBSTACLE_STOP"

                goal_error2 = float(
                    np.linalg.norm(data.qpos[:2] - waypoint2)
                )
                box_xy = data.geom_xpos[box_id, :2].copy()
                box_half = model.geom_size[box_id, :2]
                delta_box = np.maximum(
                    np.abs(data.qpos[:2] - box_xy) - box_half,
                    0.0
                )
                box_clearance = float(np.linalg.norm(delta_box))

                if depth_stop2:
                    reason2 = depth_reason2
                elif box_clearance < 0.441:
                    reason2 = "CLEARANCE_STOP"
                elif goal_error2 <= 0.075:
                    reason2 = "WAYPOINT2_REACHED"
                elif forward2 >= 0.78:
                    reason2 = "DISTANCE_LIMIT_STOP"
                elif lateral2 > 0.08:
                    reason2 = "PATH_DEVIATION_STOP"
                elif abs((
                    (yaw_deg() - desired_yaw + 180) % 360
                ) - 180) > 10.0:
                    reason2 = "HEADING_DEVIATION_STOP"
                elif t - walk2_start_time >= 16.0:
                    reason2 = "WALK2_TIMEOUT"

                if reason2 != "TURN2_REACHED":
                    stage = "stopping"
                    stage_start = t
                    print(
                        f"Stage 2 walk stop: "
                        f"distance={forward2:.3f}m "
                        f"reason={reason2}"
                    )

            elif stage == "stopping" and t - stage_start >= 1.0:
                stage = "hold"
                stage_start = t

            elif stage == "hold" and t - stage_start >= 1.0:
                break

            set_legs(stage, t - stage_start)
            mujoco.mj_step(model, data)
            inspect_step()

    result2 = HERE / "step4_unified_nan_map_fault_depth_early_live_v3.csv"
    with result2.open("w", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=list(stage2_rows[0])
        )
        writer.writeheader()
        writer.writerows(stage2_rows)

    print("\n=== Stage 2 Result ===")
    print("Final XY:", np.round(data.qpos[:2], 4))
    print("Final yaw:", f"{yaw_deg():+.2f}deg")
    print("Distance to waypoint 2:",
          f"{np.linalg.norm(data.qpos[:2]-waypoint2):.3f}m")
    print("Final stage:", stage)
    print("Stop reason:", reason2)
    print("Maximum tilt:", f"{max_tilt2:.2f}deg")
    print("Box contact samples:", contact2)
    print("CSV:", result2)

    final_error2 = float(
        np.linalg.norm(data.qpos[:2] - waypoint2)
    )
    passed2 = (
        reason2 == "WAYPOINT2_REACHED"
        and stage == "hold"
        and final_error2 <= 0.08
        and contact2 == 0
        and max_tilt2 <= 12.0
    )
    print("Final waypoint 2 error:", f"{final_error2:.3f}m")
    print("Waypoint 2 final accuracy:",
          "PASS" if passed2 else "FAIL")

    target_angle = saved_target_angle

print("Stage 2 is exploratory simulation, not hardware-safe.")


print("\n=== Depth Guard Summary ===")
print("Depth sample count:", len(_depth_samples))
if _depth_samples:
    valid_depths = [
        d for _, d in _depth_samples
        if math.isfinite(d)
    ]
    if valid_depths:
        print(
            "Nearest detected obstacle:",
            f"{min(valid_depths):.3f}m"
        )
    else:
        print("No valid depth readings")

if "reason2" in globals():
    print("Depth guard final stop reason:", reason2)

_depth_renderer.close()
print("Depth guard simulation finished.")

print("\n=== New Box Test Acceptance ===")
new_box_pass = (
    reason2 == "DEPTH_OBSTACLE_STOP"
    and stage == "hold"
    and contact2 == 0
    and max_tilt2 <= 12.0
)
print(
    "NEW_BOX_DEPTH_STOP:",
    "PASS" if new_box_pass else "FAIL"
)
print("Additional box collision must be audited separately.")

import json
from pathlib import Path
print("\n=== LIVE DEPTH REPLANNING TEST ===")

import json
from pathlib import Path

out_dir = Path("../gate3_avoidance_v01")
live_report = {"test": "LIVE_DEPTH_REPLANNING"}

if not (
    globals().get("reason2") == "DEPTH_OBSTACLE_STOP"
    and stage == "hold"
):
    print("LIVE_REPLAN: NOT_STARTED")
    live_report["status"] = "NO_DEPTH_STOP"
else:
    camera_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_CAMERA, "dog_eye"
    )
    torso_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_BODY, "torso"
    )
    if camera_id < 0 or torso_id < 0:
        raise RuntimeError("Camera or torso not found")

    with mujoco.Renderer(
        model, width=640, height=360
    ) as renderer:
        renderer.enable_depth_rendering()
        renderer.update_scene(data, camera="dog_eye")
        depth = renderer.render().copy()
        renderer.disable_depth_rendering()

    h, w = depth.shape
    vv, uu = np.mgrid[0:h:4, 0:w:4]
    dd = depth[::4, ::4]
    valid = np.isfinite(dd) & (dd > 0.01) & (dd < 10)

    if not np.any(valid):
        print("LIVE_REPLAN: SENSOR_INVALID")
        live_report["status"] = "SENSOR_INVALID"
    else:
        focal = h / (
            2 * math.tan(
                math.radians(
                    float(model.cam_fovy[camera_id])
                ) / 2
            )
        )
        d = dd[valid]
        u, v = uu[valid], vv[valid]

        camera_pts = np.column_stack((
            (u + 0.5 - w/2)*d/focal,
            -(v + 0.5 - h/2)*d/focal,
            -d
        ))

        cam_rot = data.cam_xmat[camera_id].reshape(3,3)
        world = (
            camera_pts @ cam_rot.T
            + data.cam_xpos[camera_id]
        )
        torso_rot = data.xmat[torso_id].reshape(3,3)
        body = (
            world - data.xpos[torso_id]
        ) @ torso_rot

        forward = -body[:,0]
        lateral = body[:,1]
        height = world[:,2]

        candidate = (
            (height >= 0.06)
            & (height <= 0.45)
            & (forward > 0)
            & (forward < 1.10)
            & (np.abs(lateral) < 0.25)
        )

        print("Valid depth pixels:", int(valid.sum()))
        print("Obstacle points:", int(candidate.sum()))

        if candidate.sum() < 30:
            print("LIVE_REPLAN: INSUFFICIENT_POINTS")
            live_report["status"] = "INSUFFICIENT_POINTS"
        else:
            near = float(forward[candidate].min())
            selected = candidate & (forward <= near + 0.15)
            points = world[selected]

            np.savetxt(
                out_dir / "step4_unified_nan_map_fault_live_obstacle_points_v3.csv",
                points,
                delimiter=",",
                header="world_x,world_y,world_z",
                comments=""
            )

            # Trial uncertainty bounds. Not a certified
            # model of unseen obstacle surfaces.
            low = points[:,:2].min(axis=0) - [0.15,0.05]
            high = points[:,:2].max(axis=0) + [0.01,0.05]

            start = data.qpos[:2].copy()
            goal = np.array([-1.10,1.50])
            clearance_required = 0.521

            def clearance(xy):
                delta = np.maximum(
                    np.maximum(low-xy,xy-high),0
                )
                return np.linalg.norm(delta,axis=-1)

            def segment(a,b):
                n = max(
                    2,
                    int(np.ceil(np.linalg.norm(b-a)/0.005))+1
                )
                t = np.linspace(0,1,n)[:,None]
                return a + t*(b-a)

            candidates = []
            for y in np.arange(1.00,2.31,0.05):
                for x in np.arange(-0.85,0.81,0.05):
                    via = np.array([x,y])
                    path = np.vstack((
                        segment(start,via),
                        segment(via,goal)
                    ))
                    margin = float(clearance(path).min())
                    if margin >= clearance_required:
                        length = (
                            np.linalg.norm(start-via)
                            + np.linalg.norm(via-goal)
                        )
                        candidates.append(
                            (length,margin,via)
                        )

            start_clear = float(clearance(start))
            live_report.update({
                "status": "PLANNED" if candidates
                          else "NO_STATIC_ROUTE",
                "start_xy": start.tolist(),
                "depth_nearest_m": near,
                "observed_points": int(len(points)),
                "estimated_box_min": low.tolist(),
                "estimated_box_max": high.tolist(),
                "start_clearance_m": start_clear,
                "required_clearance_m": clearance_required,
                "candidate_count": len(candidates),
            })

            print("Robot XY:", np.round(start,4))
            print("Nearest depth:", f"{near:.3f}m")
            print("Estimated box min:", np.round(low,4))
            print("Estimated box max:", np.round(high,4))
            print("Start clearance:", f"{start_clear:.3f}m")
            print("Static route candidates:",len(candidates))

            if candidates:
                length,minimum,via = min(
                    candidates,key=lambda x:x[0]
                )
                live_report.update({
                    "waypoint": via.tolist(),
                    "route_clearance_m": minimum,
                    "route_length_m": length,
                })
                print("Generated waypoint:",np.round(via,4))
                print("Route clearance:",f"{minimum:.3f}m")

            if start_clear < clearance_required:
                print("LIVE_REPLAN: START_TOO_CLOSE")
                live_report["status"] = "START_TOO_CLOSE"
            elif not candidates:
                print("LIVE_REPLAN: NO_STATIC_ROUTE")
            else:
                print("LIVE_REPLAN: STATIC_CANDIDATE")

report_path = out_dir / "step4_unified_nan_map_fault_early_live_replan_result_v3_early.json"
report_path.write_text(
    json.dumps(live_report,indent=2)
)
print("Report:",report_path)
print("No new walking commands were issued.")
print("Simulation only; not hardware-safe.")

print("\n=== Gate 3 Step 2 Acceptance ===")

status = live_report.get("status")
has_observation = (
    live_report.get("observed_points", 0) >= 30
)
has_plan = (
    status in ("PLANNED", "START_TOO_CLOSE")
    and live_report.get("candidate_count", 0) > 0
)
decision_is_safe = status in (
    "START_TOO_CLOSE",
    "PLANNED",
    "NO_STATIC_ROUTE",
    "SENSOR_INVALID",
    "INSUFFICIENT_POINTS",
)
step2_pass = (
    globals().get("reason2") == "DEPTH_OBSTACLE_STOP"
    and stage == "hold"
    and has_observation
    and has_plan
    and decision_is_safe
)
print("Depth stop:", globals().get("reason2"))
print("Observation points:",
      live_report.get("observed_points", 0))
print("Route candidates:",
      live_report.get("candidate_count", 0))
print("Decision:", status)
print("GATE3_STEP2:",
      "PASS" if step2_pass else "FAIL")

print("\n=== EARLY LIVE REPLAN ACCEPTANCE ===")
status = live_report.get("status")
n = live_report.get("candidate_count", 0)
start_margin = live_report.get(
    "start_clearance_m", float("nan")
)
required = live_report.get(
    "required_clearance_m", 0.521
)

accepted = (
    globals().get("reason2") == "DEPTH_OBSTACLE_STOP"
    and stage == "hold"
    and status == "PLANNED"
    and n > 0
    and start_margin >= required
)

print("Live status:", status)
print("Candidates:", n)
print("Start clearance:", start_margin)
print("Required:", required)
print(
    "GATE3_STEP2_EARLY:",
    "PASS" if accepted else "FAIL"
)
print("No new walking commands issued.")

print("\n=== GATE 3 STEP 2 FINAL AUDIT ===")

decision = live_report.get("status", "NOT_STARTED")
observed = int(live_report.get("observed_points", 0))
candidate_count = int(live_report.get("candidate_count", 0))
start_clearance = float(
    live_report.get("start_clearance_m", float("nan"))
)
required = float(
    live_report.get("required_clearance_m", 0.521)
)

observed_ok = (
    reason2 == "DEPTH_OBSTACLE_STOP"
    and stage == "hold"
    and observed >= 30
)

planning_ok = (
    decision == "PLANNED"
    and candidate_count > 0
    and start_clearance >= required
)

step2_pass = observed_ok and planning_ok

print("Depth stop:", reason2)
print("Observation:", "PASS" if observed_ok else "FAIL")
print("Planner decision:", decision)
print("Candidates:", candidate_count)
print("Start clearance:", start_clearance)
print("Required:", required)
print("Planning:", "PASS" if planning_ok else "FAIL")
print("GATE3_STEP2_FINAL:",
      "PASS" if step2_pass else "FAIL")


# ===== LIVE CAMERA -> ROUTE -> CONTROL =====
import ast
import csv
import json
import math
import numpy as np

def xy_clearance(geom_id):
    center = data.geom_xpos[geom_id, :2]
    half = model.geom_size[geom_id, :2]
    offset = np.maximum(
        np.abs(data.qpos[:2] - center) - half, 0.0
    )
    return float(np.linalg.norm(offset))

def angle_error(target, current):
    return (target - current + 180) % 360 - 180

def guard():
    est = estimated_clearance()
    extra = xy_clearance(extra_id)
    red = xy_clearance(box_id)
    height, tilt = posture()

    collision = False
    for j in range(data.ncon):
        c = data.contact[j]
        if extra_id in (int(c.geom1), int(c.geom2)):
            collision = True
        if box_id in (int(c.geom1), int(c.geom2)):
            collision = True

    if collision:
        reason = "CONTACT_STOP"
    elif not np.all(np.isfinite([
        est, extra, red, height, tilt
    ])):
        reason = "STATE_INVALID_STOP"
    elif min(est, extra, red) < CLEARANCE:
        reason = "CLEARANCE_STOP"
    elif height < MIN_HEIGHT or tilt > MAX_TILT:
        reason = "POSTURE_STOP"
    else:
        reason = None

    return reason, est, extra, red, height, tilt

def record(section, reason, depth):
    state, est, extra, red, height, tilt = guard()
    trace.append({
        "section": section,
        "time_s": float(data.time),
        "stage": stage,
        "x_m": float(data.qpos[0]),
        "y_m": float(data.qpos[1]),
        "yaw_deg": float(yaw_deg()),
        "estimated_clearance_m": est,
        "extra_clearance_m": extra,
        "red_clearance_m": red,
        "height_m": height,
        "tilt_deg": tilt,
        "depth_m": depth,
        "guard_reason": state or "",
        "stop_reason": reason,
    })
    return state

def soft_hold(section, reason):
    global stage, stage_start
    stage = "stopping"
    stage_start = float(data.time)
    start = float(data.time)

    while float(data.time)-start < 2.2:
        elapsed = float(data.time)-stage_start
        if stage == "stopping" and elapsed >= 1.0:
            stage = "hold"
            stage_start = float(data.time)
        elif stage == "hold" and elapsed >= 1.0:
            break

        set_legs(stage, float(data.time)-stage_start)
        mujoco.mj_step(model, data)
        record(section, reason, float("nan"))

    return stage == "hold"

def turn_to(section, goal):
    global stage, stage_start, target_angle

    delta = goal-data.qpos[:2]
    desired = math.degrees(math.atan2(-delta[1],-delta[0]))
    initial = yaw_deg()
    command = angle_error(desired, initial)

    print(f"{section}: turn command={command:+.2f}deg")

    if guard()[0]:
        return "START_GUARD_STOP"

    if abs(command) <= 2.0:
        return "TURN_ALREADY_ALIGNED"

    target_angle = command
    stage = "turn"
    stage_start = float(data.time)
    beginning = float(data.time)
    reason = "TURN_TIMEOUT"

    while float(data.time)-beginning < 30:
        actual = angle_error(yaw_deg(), initial)

        if abs(command)-abs(actual) <= 1.5:
            reason = "TURN_REACHED"
            break

        set_legs("turn", float(data.time)-stage_start)
        mujoco.mj_step(model,data)
        failure = record(section, "", float("nan"))
        if failure:
            reason = failure
            break

    soft_hold(section, reason)
    final_error = abs(angle_error(desired,yaw_deg()))

    if reason == "TURN_REACHED" and final_error > 5.0:
        reason = "TURN_ACCURACY_FAIL"

    if guard()[0]:
        reason = guard()[0]

    print(
        f"{section}: {reason}, "
        f"yaw_error={final_error:.2f}deg"
    )
    return reason

def walk_to(section, goal, renderer, timeout=28):
    global stage, stage_start

    if guard()[0]:
        return "START_GUARD_STOP"

    start_xy = data.qpos[:2].copy()
    start_yaw = yaw_deg()
    heading = math.radians(start_yaw)
    direction = np.array([
        -math.cos(heading),-math.sin(heading)
    ])
    initial_goal_distance = float(np.linalg.norm(goal-start_xy))

    stage = "walk"
    stage_start = float(data.time)
    beginning = float(data.time)
    last_depth = -1e9
    nearest = float("nan")
    reason = "WALK_TIMEOUT"

    while float(data.time)-beginning < timeout:
        t = float(data.time)
        xy = data.qpos[:2].copy()
        displacement = xy-start_xy
        forward = float(np.dot(displacement,direction))
        lateral = abs(float(
            direction[0]*displacement[1]
            -direction[1]*displacement[0]
        ))
        error = float(np.linalg.norm(goal-xy))

        if t-last_depth >= 0.40:
            last_depth = t
            try:
                nearest = float(detect_depth(renderer))
            except Exception as exc:
                print("Depth exception:",repr(exc))
                nearest = float("nan")

        failure = guard()[0]
        if failure:
            reason = failure
            break
        if not math.isfinite(nearest):
            reason = "DEPTH_SENSOR_STOP"
            break
        if nearest < 0.55:
            reason = "DEPTH_OBSTACLE_STOP"
            break
        if lateral > 0.08:
            reason = "PATH_DEVIATION_STOP"
            break
        if abs(angle_error(start_yaw,yaw_deg())) > 10:
            reason = "HEADING_STOP"
            break
        if error <= (
            FINAL_TOL if section == "FINAL_WALK"
            else WAYPOINT_TOL
        ):
            reason = "GOAL_TRIGGERED"
            break
        if forward > initial_goal_distance+0.10:
            reason = "DISTANCE_LIMIT_STOP"
            break

        set_legs("walk",t-stage_start)
        mujoco.mj_step(model,data)
        failure = record(section,"",nearest)
        if failure:
            reason = failure
            break

    soft_hold(section,reason)
    final_error = float(np.linalg.norm(goal-data.qpos[:2]))

    if reason == "GOAL_TRIGGERED" and final_error > 0.15:
        reason = "FINAL_ACCURACY_FAIL"
    if guard()[0]:
        reason = guard()[0]

    print(
        f"{section}: {reason}, "
        f"goal_error={final_error:.3f}m, "
        f"final_xy={np.round(data.qpos[:2],4)}"
    )
    return reason

print("\n=== GATE 3 STEP 3: LIVE AUTONOMOUS ROUTE ===")

CLEARANCE = 0.521
PLANNING_CLEARANCE = 0.551
MAX_TILT = 12.0
MIN_HEIGHT = 0.16
WAYPOINT_TOL = 0.11
FINAL_TOL = 0.11

trace = []
results = []
stage3_reason = "NOT_STARTED"
stage3_ok = False

extra_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM,
    "extra_blocking_box"
)
if extra_id < 0 or box_id < 0:
    raise RuntimeError("Obstacle geometry IDs unavailable")


# Controlled sensor-map corruption test.
if reason2 == "DEPTH_OBSTACLE_STOP":
    live_report["estimated_box_min"] = [
        float("nan"), float("nan")
    ]
# ===== Goal policy: evaluated before route-control execution =====
goal_policy_target = np.array([-1.10, 1.50])
goal_policy_low = np.asarray(
    live_report.get("estimated_box_min", []),
    dtype=float,
)
goal_policy_high = np.asarray(
    live_report.get("estimated_box_max", []),
    dtype=float,
)

goal_policy_valid = (
    goal_policy_low.shape == (2,)
    and goal_policy_high.shape == (2,)
    and np.isfinite(goal_policy_low).all()
    and np.isfinite(goal_policy_high).all()
    and np.all(goal_policy_low <= goal_policy_high)
)

goal_policy_clearance = float("nan")
goal_policy_unsafe = not goal_policy_valid

if goal_policy_valid:
    diff = np.maximum(
        np.maximum(
            goal_policy_low - goal_policy_target,
            goal_policy_target - goal_policy_high,
        ),
        0.0,
    )
    goal_policy_clearance = float(np.linalg.norm(diff))
    goal_policy_unsafe = goal_policy_clearance < PLANNING_CLEARANCE

print("GOAL_POLICY_CLEARANCE:",
      f"{goal_policy_clearance:.4f}m")
print("GOAL_POLICY_UNSAFE:", goal_policy_unsafe)

if (
    goal_policy_unsafe
    or live_report.get("status") != "PLANNED"
    or reason2 != "DEPTH_OBSTACLE_STOP"
    or stage != "hold"
):
    stage3_reason = "LIVE_OBSERVATION_NOT_READY"
else:
    low = np.array(live_report["estimated_box_min"])
    high = np.array(live_report["estimated_box_max"])
    goal = np.array([-1.10, 1.50])

    def estimated_clearance():
        xy = data.qpos[:2]
        delta = np.maximum(
            np.maximum(low - xy, xy - high), 0.0
        )
        return float(np.linalg.norm(delta))

    def segment(a, b):
        steps = max(
            2,
            int(np.ceil(np.linalg.norm(b-a)/0.005)) + 1
        )
        t = np.linspace(0, 1, steps)[:, None]
        return a + t*(b-a)

    def route_clearance(path):
        delta = np.maximum(
            np.maximum(low-path, path-high), 0.0
        )
        return float(np.linalg.norm(delta, axis=1).min())

    def route_candidates():
        start = data.qpos[:2].copy()
        current_yaw = yaw_deg()
        choices = []
        for y in np.arange(1.00, 2.31, 0.05):
            for x in np.arange(-0.85, 0.81, 0.05):
                via = np.array([x, y])
                path = np.vstack((
                    segment(start, via),
                    segment(via, goal)
                ))
                minimum = route_clearance(path)
                if minimum < PLANNING_CLEARANCE:
                    continue

                displacement = via-start
                desired = math.degrees(math.atan2(
                    -displacement[1], -displacement[0]
                ))
                correction = abs(
                    (desired-current_yaw+180)%360-180
                )
                length = (
                    np.linalg.norm(via-start)
                    + np.linalg.norm(goal-via)
                )
                score = length + 0.005*correction \
                        + 4.0/max(minimum, 0.001)
                choices.append(
                    (score, minimum, length, via)
                )
        return sorted(choices, key=lambda item: item[0])

    choices = route_candidates()
    print("Live observed points:",
          live_report.get("observed_points"))
    print("Current XY:", np.round(data.qpos[:2], 4))
    print("Start clearance:",
          f"{estimated_clearance():.3f}m")
    print("Candidates with 3cm extra reserve:",
          len(choices))

    if estimated_clearance() < CLEARANCE:
        stage3_reason = "START_CLEARANCE_INSUFFICIENT"
    elif not choices:
        stage3_reason = "NO_ROBUST_ROUTE"
    else:
        _, predicted, length, via = choices[0]
        print("Selected live waypoint:",
              np.round(via, 4))
        print("Predicted minimum clearance:",
              f"{predicted:.3f}m")
        print("Planned path length:",
              f"{length:.3f}m")

        depth_path = (
            HERE / "../gate3_obstacle_stop_v01/"
            "obstacle_stop_background_test.py"
        ).resolve()
        depth_tree = ast.parse(depth_path.read_text())
        depth_functions = [
            n for n in depth_tree.body
            if isinstance(n, ast.FunctionDef)
            and n.name == "obstacle_distance"
        ]
        if len(depth_functions) != 1:
            raise RuntimeError("Depth function not found")

        namespace = {
            "np": np, "math": math, "mujoco": mujoco,
            "model": model, "data": data,
        }
        exec(
            compile(
                ast.Module(
                    body=depth_functions,
                    type_ignores=[]
                ),
                str(depth_path),
                "exec"
            ),
            namespace
        )
        detect_depth = namespace["obstacle_distance"]

        stage3_reason = "RUNNING"

        with mujoco.Renderer(
            model, width=640, height=360
        ) as renderer:
            # First heading adjustment uses actual stop pose.
            turn1 = turn_to("LIVE_TURN1", via)
            print("Live turn 1:", turn1)

            # Re-evaluate using the actual pose AFTER turning.
            # A turn can translate the robot by several centimeters.
            walk1 = "NOT_STARTED"
            if turn1 in ("TURN_REACHED", "TURN_ALREADY_ALIGNED"):
                for correction_iteration in range(6):
                    candidates_now = route_candidates()
                    print(
                        "POST_TURN_CANDIDATES:",
                        len(candidates_now),
                        "iteration:", correction_iteration
                    )

                    if not candidates_now:
                        walk1 = "NO_SAFE_POST_TURN_ROUTE"
                        break

                    # Favor a wider margin over the shortest route.
                    candidates_now.sort(
                        key=lambda c: (
                            -c[1],
                            c[2],
                            c[0]
                        )
                    )
                    _, margin_now, _, new_via = candidates_now[0]
                    via = new_via.copy()

                    print(
                        "POST_TURN_WAYPOINT:",
                        np.round(via, 4),
                        "predicted_margin:",
                        f"{margin_now:.3f}m"
                    )

                    requested = math.degrees(math.atan2(
                        -(via[1] - data.qpos[1]),
                        -(via[0] - data.qpos[0])
                    ))
                    yaw_error = abs(angle_error(
                        requested, yaw_deg()
                    ))

                    if yaw_error <= 2.0:
                        print("POST_TURN_HEADING: ALIGNED")
                        break

                    print(
                        "POST_TURN_HEADING_CORRECTION:",
                        f"{yaw_error:.2f}deg"
                    )
                    correction_result = turn_to(
                        "LIVE_CORRECTIVE_TURN",
                        via
                    )
                    if correction_result not in (
                        "TURN_REACHED",
                        "TURN_ALREADY_ALIGNED"
                    ):
                        walk1 = correction_result
                        break
                else:
                    walk1 = "CORRECTION_ITERATION_LIMIT"

                if walk1 == "NOT_STARTED":
                    start_now = data.qpos[:2].copy()
                    delta_now = via - start_now
                    length_now = float(np.linalg.norm(delta_now))
                    heading_now = math.radians(yaw_deg())
                    direction_now = np.array([
                        -math.cos(heading_now),
                        -math.sin(heading_now)
                    ])

                    # Check both waypoint line and commanded
                    # forward-heading line, including residual
                    # yaw error and a small travel overshoot.
                    commanded_end = (
                        start_now + direction_now *
                        (length_now + 0.04)
                    )
                    planned_line = segment(start_now, via)
                    commanded_line = segment(
                        start_now, commanded_end
                    )
                    predicted_line = min(
                        route_clearance(planned_line),
                        route_clearance(commanded_line)
                    )

                    print(
                        "POST_TURN_ACTUAL_XY:",
                        np.round(start_now, 4)
                    )
                    print(
                        "POST_TURN_PREDICTED_CLEARANCE:",
                        f"{predicted_line:.3f}m"
                    )

                    if guard()[0]:
                        walk1 = "POST_TURN_GUARD_STOP"
                    elif predicted_line < PLANNING_CLEARANCE:
                        walk1 = "POST_TURN_ROUTE_REJECTED"
                    else:
                        walk1 = walk_to(
                            "LIVE_WAYPOINT_WALK",
                            via, renderer, timeout=32
                        )
                        print("Live waypoint walk:", walk1)
            else:
                walk1 = "FIRST_TURN_FAILED"

            if walk1 == "GOAL_TRIGGERED":
                turn2 = turn_to("LIVE_FINAL_TURN", goal)
                print("Live final turn:", turn2)
            else:
                turn2 = "NOT_STARTED"

            if turn2 in (
                "TURN_REACHED", "TURN_ALREADY_ALIGNED"
            ):
                walk2 = walk_to(
                    "FINAL_WALK",
                    goal, renderer, timeout=32
                )
                print("Live final walk:", walk2)
            else:
                walk2 = "NOT_STARTED"

        final_error = float(np.linalg.norm(
            data.qpos[:2]-goal
        ))

        # The trace also includes stopping and holding.
        clear_ok = (
            bool(trace)
            and all(
                np.isfinite([
                    r["estimated_clearance_m"],
                    r["extra_clearance_m"],
                    r["red_clearance_m"],
                    r["height_m"],
                    r["tilt_deg"]
                ]).all()
                and min(
                    r["estimated_clearance_m"],
                    r["extra_clearance_m"],
                    r["red_clearance_m"]
                ) >= CLEARANCE
                and r["height_m"] >= MIN_HEIGHT
                and r["tilt_deg"] <= MAX_TILT
                and r["guard_reason"] == ""
                for r in trace
            )
        )

        stage3_ok = (
            turn1 in ("TURN_REACHED","TURN_ALREADY_ALIGNED")
            and walk1 == "GOAL_TRIGGERED"
            and turn2 in ("TURN_REACHED","TURN_ALREADY_ALIGNED")
            and walk2 == "GOAL_TRIGGERED"
            and final_error <= 0.15
            and stage == "hold"
            and clear_ok
        )
        stage3_reason = (
            "PASS" if stage3_ok
            else "TRAJECTORY_OR_GUARD_FAIL"
        )

        if trace:
            csv_file = HERE/"step4_unified_nan_map_fault_live_autonomous_route_v3.csv"
            with csv_file.open("w",newline="") as f:
                writer = csv.DictWriter(
                    f, fieldnames=list(trace[0])
                )
                writer.writeheader()
                writer.writerows(trace)
            print("Trajectory CSV:", csv_file)


# ===== Finalize safe-goal-rejection classification =====
if not goal_policy_valid:
    stage3_reason = "GOAL_MAP_INVALID_REJECTED"
elif goal_policy_unsafe:
    if (
        reason2 == "DEPTH_OBSTACLE_STOP"
        and stage == "hold"
        and len(trace) == 0
        and not stage3_ok
    ):
        stage3_reason = "GOAL_UNSAFE_REJECTED"
    else:
        stage3_reason = "GOAL_POLICY_STOP_VALIDATION_FAILED"

report = {
    "step": 3,
    "status": stage3_reason,
    "pass": bool(stage3_ok),
    "depth_source": "current simulation observation",
    "goal": [-1.10,1.50],
    "final_xy": [float(v) for v in data.qpos[:2]],
    "trace_samples": len(trace),
}
report_path = HERE/"step4_unified_nan_map_fault_live_autonomous_route_v3.json"
report_path.write_text(json.dumps(report,indent=2))

print("\n=== GATE 3 STEP 3 SUMMARY ===")
print("Status:", stage3_reason)
print("Final XY:", np.round(data.qpos[:2],4))
print("Trace samples:",len(trace))
print("GATE3_STEP3:",
      "PASS" if stage3_ok else "FAIL")
print("Report:",report_path)
print("Simulation only. Full-body safety not certified.")


print("\n=== GATE 3 STEP 3 DIAGNOSTIC ===")
diagnostic_csv = HERE / "step4_unified_nan_map_fault_live_autonomous_route_v3.csv"

if diagnostic_csv.exists():
    with diagnostic_csv.open(newline="") as f:
        rows = list(csv.DictReader(f))

    if rows:
        columns = {
            "camera_estimate": "estimated_clearance_m",
            "extra_box": "extra_clearance_m",
            "red_box": "red_clearance_m",
        }

        for label, key in columns.items():
            pairs = [
                (float(r[key]), r)
                for r in rows
                if math.isfinite(float(r[key]))
            ]
            if pairs:
                value, row = min(pairs, key=lambda p: p[0])
                print(
                    f"{label}: minimum={value:.4f}m "
                    f"at t={row['time_s']} "
                    f"XY=({row['x_m']},{row['y_m']})"
                )

        violations = [
            r for r in rows
            if r["guard_reason"]
        ]
        if violations:
            first = violations[0]
            print("First guard violation:",
                  first["guard_reason"])
            print("First violation position:",
                  first["x_m"], first["y_m"])
        else:
            print("Recorded guard violations: 0")
else:
    print("No trajectory CSV created")

print("GATE3_STEP3_V3:",
      "PASS" if stage3_ok else "FAIL")
