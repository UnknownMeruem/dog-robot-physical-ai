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
    / "obstacle_scene.xml"
)

spec = importlib.util.spec_from_file_location(
    "gate2_gait", gait_path
)
gait = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gait)

model = mujoco.MjModel.from_xml_path(str(scene_path))
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
WAYPOINT_TOLERANCE = 0.065
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

output = HERE / "waypoint1_braking.csv"
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

print("\n=== Stage 2: dynamic heading and short walk ===")
waypoint2 = np.array([-1.10, 0.80])
position1 = data.qpos[:2].copy()
error1 = float(np.linalg.norm(position1 - target_xy))

if (
    stop_reason != "WAYPOINT1_REACHED"
    or error1 > 0.08
    or box_contacts != 0
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

    result2 = HERE / "depth_emergency_stop.csv"
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

print("\n=== Emergency Stop Acceptance ===")
emergency_pass = (
    reason2 == "DEPTH_OBSTACLE_STOP"
    and stage == "hold"
    and contact2 == 0
    and max_tilt2 <= 12.0
)
print("Final stage:", stage)
print("Stop reason:", reason2)
print("Box contact samples:", contact2)
print(
    "DEPTH_EMERGENCY_STOP:",
    "PASS" if emergency_pass else "FAIL"
)
