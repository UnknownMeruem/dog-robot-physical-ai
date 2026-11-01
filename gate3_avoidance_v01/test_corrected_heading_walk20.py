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
WAYPOINT_TOLERANCE = 0.30
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

output = HERE / "waypoint1_camera_full.csv"
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


# ===== Replanned route: first turn only =====
print("\n=== Replanned Turn Test ===")


# Generate the waypoint from saved depth-camera points.
import runpy
planner_result = runpy.run_path(
    str(HERE / "plan_camera_waypoint_full.py"),
    run_name="__camera_planner__",
)

if not planner_result.get("candidates"):
    raise RuntimeError("Camera planner found no valid route")

generated_goal = np.asarray(
    planner_result["via"], dtype=float
)
planned_start = np.asarray(
    planner_result["start"], dtype=float
)
required_start = float(planner_result["target"])
observed_start_clearance = float(
    planner_result["start_clearance"]
)

if observed_start_clearance < required_start:
    raise RuntimeError("Camera-based start margin insufficient")

if np.linalg.norm(data.qpos[:2] - planned_start) > 0.02:
    raise RuntimeError(
        "Current robot pose differs from planner start"
    )

replan_goal = generated_goal
print("CAMERA_WAYPOINT_USED:", np.round(replan_goal, 4))
print("CAMERA_TURN_START_MARGIN:",
      f"{observed_start_clearance-required_start:+.3f}m")

current_xy = data.qpos[:2].copy()
delta = replan_goal - current_xy

desired_yaw = math.degrees(math.atan2(-delta[1], -delta[0]))
starting_yaw = yaw_deg()
turn_command = (desired_yaw - starting_yaw + 180) % 360 - 180

extra_id = mujoco.mj_name2id(
    model, mujoco.mjtObj.mjOBJ_GEOM, "extra_blocking_box"
)
if extra_id < 0:
    raise RuntimeError("Extra obstacle geom missing")

def xy_clearance(geom_id):
    center = data.geom_xpos[geom_id, :2]
    half = model.geom_size[geom_id, :2]
    offset = np.maximum(
        np.abs(data.qpos[:2] - center) - half, 0.0
    )
    return float(np.linalg.norm(offset))

required = 0.441

print("Starting XY:", np.round(current_xy, 4))
print("Replan waypoint:", replan_goal)
print("Starting yaw:", f"{starting_yaw:+.2f}deg")
print("Desired yaw:", f"{desired_yaw:+.2f}deg")
print("Requested turn:", f"{turn_command:+.2f}deg")

can_turn = (
    stop_reason == "WAYPOINT1_REACHED"
    and stage == "hold"
    and box_contacts == 0
    and xy_clearance(extra_id) >= required + 0.08
    and xy_clearance(box_id) >= required
)

if not can_turn:
    print("REPLAN_TURN_NOT_STARTED: prerequisites failed")
else:
    target_angle = turn_command
    stage = "turn"
    stage_start = float(data.time)
    begin_time = float(data.time)

    turn_reason = ""
    turn_contacts = 0
    minimum_extra_clearance = float("inf")
    minimum_red_clearance = float("inf")
    maximum_turn_tilt = 0.0
    minimum_turn_height = float("inf")
    turn_rows = []

    while float(data.time) - begin_time < 32.0:
        t = float(data.time)
        elapsed = t - stage_start

        actual = (yaw_deg() - starting_yaw + 180) % 360 - 180
        remaining = abs(turn_command) - abs(actual)

        if stage == "turn":
            if remaining <= 1.5:
                turn_reason = "REPLAN_ANGLE_REACHED"
                stage = "stopping"
                stage_start = t
                print(
                    "Turn target reached:",
                    f"{actual:+.2f}deg"
                )
            elif t - begin_time >= 27.0:
                turn_reason = "REPLAN_TURN_TIMEOUT"
                stage = "stopping"
                stage_start = t

        elif stage == "stopping" and elapsed >= 1.0:
            stage = "hold"
            stage_start = t

        elif stage == "hold" and elapsed >= 1.0:
            break

        set_legs(stage, t - stage_start)
        mujoco.mj_step(model, data)

        height, tilt = posture()
        maximum_turn_tilt = max(maximum_turn_tilt, tilt)
        minimum_turn_height = min(minimum_turn_height, height)

        extra_clearance = xy_clearance(extra_id)
        red_clearance = xy_clearance(box_id)

        minimum_extra_clearance = min(
            minimum_extra_clearance, extra_clearance
        )
        minimum_red_clearance = min(
            minimum_red_clearance, red_clearance
        )

        for ci in range(data.ncon):
            c = data.contact[ci]
            if (
                extra_id in (int(c.geom1), int(c.geom2))
                or box_id in (int(c.geom1), int(c.geom2))
            ):
                turn_contacts += 1

        if stage == "turn":
            if height < 0.16 or tilt > 12.0:
                turn_reason = "REPLAN_POSTURE_STOP"
            elif turn_contacts:
                turn_reason = "REPLAN_CONTACT_STOP"
            elif min(extra_clearance, red_clearance) < required:
                turn_reason = "REPLAN_CLEARANCE_STOP"

            if turn_reason:
                stage = "stopping"
                stage_start = float(data.time)
                print("Turn interrupted:", turn_reason)

        turn_rows.append({
            "time_s": float(data.time),
            "stage": stage,
            "x_m": float(data.qpos[0]),
            "y_m": float(data.qpos[1]),
            "yaw_deg": yaw_deg(),
            "tilt_deg": tilt,
            "height_m": height,
            "extra_clearance_m": extra_clearance,
            "red_clearance_m": red_clearance,
            "contact_samples": turn_contacts,
        })

    if not turn_reason:
        turn_reason = "REPLAN_TEST_TIMEOUT"

    turn_csv = HERE / "camera_full_turn.csv"
    if turn_rows:
        with turn_csv.open("w", newline="") as f:
            writer = csv.DictWriter(
                f, fieldnames=list(turn_rows[0])
            )
            writer.writeheader()
            writer.writerows(turn_rows)

    final_yaw_error = (
        yaw_deg() - desired_yaw + 180
    ) % 360 - 180

    passed = (
        turn_reason == "REPLAN_ANGLE_REACHED"
        and stage == "hold"
        and abs(final_yaw_error) <= 5.0
        and turn_contacts == 0
        and minimum_turn_height >= 0.16
        and maximum_turn_tilt <= 12.0
        and minimum_extra_clearance >= required
        and minimum_red_clearance >= required
    )

    print("\n=== Replanned Turn Result ===")
    print("Final XY:", np.round(data.qpos[:2], 4))
    print("Final yaw:", f"{yaw_deg():+.2f}deg")
    print("Final yaw error:", f"{final_yaw_error:+.2f}deg")
    print("Final stage:", stage)
    print("Stop reason:", turn_reason)
    print("Contact samples:", turn_contacts)
    print(
        "Minimum extra-box clearance:",
        f"{minimum_extra_clearance:.3f}m"
    )
    print(
        "Minimum red-box clearance:",
        f"{minimum_red_clearance:.3f}m"
    )
    print("Maximum tilt:", f"{maximum_turn_tilt:.2f}deg")
    print("CSV:", turn_csv)
    print(
        "REPLANNED_TURN:",
        "PASS" if passed else "FAIL"
    )

print("Simulation only; not hardware-safe.")




print("\n=== Dynamic Heading Correction ===")
from pathlib import Path

route = np.atleast_2d(np.loadtxt(
    HERE / "post_turn_camera_route.csv",
    delimiter=",", skiprows=1,
))

start_xy = data.qpos[:2].copy()
initial_yaw = yaw_deg()

if not (
    turn_reason == "REPLAN_ANGLE_REACHED"
    and stage == "hold"
    and turn_contacts == 0
):
    raise RuntimeError("First replanned turn has not passed")

if np.linalg.norm(start_xy - route[0]) > 0.03:
    raise RuntimeError(
        "Saved route does not match current turn-end pose"
    )

# Look 15cm ahead along the camera-derived route.
distances = np.linalg.norm(route - route[0], axis=1)
index = min(
    len(route) - 1,
    max(1, int(np.searchsorted(distances, 0.15)))
)
lookahead = route[index]
delta = lookahead - start_xy

desired_yaw = math.degrees(
    math.atan2(-delta[1], -delta[0])
)
command = (
    desired_yaw - initial_yaw + 180
) % 360 - 180

box_low = planner_result["box_min"]
box_high = planner_result["box_max"]

def estimated_clearance():
    xy = data.qpos[:2]
    offset = np.maximum(
        np.maximum(box_low - xy, xy - box_high),
        0.0,
    )
    return float(np.linalg.norm(offset))

required = 0.521

print("Start XY:", np.round(start_xy, 4))
print("Current yaw:", f"{initial_yaw:+.2f}deg")
print("Lookahead XY:", np.round(lookahead, 4))
print("Desired yaw:", f"{desired_yaw:+.2f}deg")
print("Correction command:", f"{command:+.2f}deg")

if (
    estimated_clearance() < required
    or xy_clearance(extra_id) < required
    or xy_clearance(box_id) < required
):
    raise RuntimeError("Insufficient start clearance")

target_angle = command
stage = "turn"
stage_start = float(data.time)
beginning = float(data.time)
correction_reason = "NOT_REACHED"
correction_contacts = 0
min_estimated = float("inf")
min_extra = float("inf")
min_red = float("inf")
max_tilt = 0.0
min_height = float("inf")

while float(data.time) - beginning < 27.0:
    t = float(data.time)
    elapsed = t - stage_start
    actual = (
        yaw_deg() - initial_yaw + 180
    ) % 360 - 180

    if stage == "turn":
        remaining = abs(command) - abs(actual)
        if remaining <= 1.5:
            correction_reason = "ANGLE_REACHED"
            stage = "stopping"
            stage_start = t
        elif t - beginning > 23.0:
            correction_reason = "TURN_TIMEOUT"
            stage = "stopping"
            stage_start = t
    elif stage == "stopping" and elapsed >= 1.0:
        stage = "hold"
        stage_start = t
    elif stage == "hold" and elapsed >= 1.0:
        break

    set_legs(stage, t - stage_start)
    mujoco.mj_step(model, data)

    est = estimated_clearance()
    extra = xy_clearance(extra_id)
    red = xy_clearance(box_id)
    height, tilt = posture()

    min_estimated = min(min_estimated, est)
    min_extra = min(min_extra, extra)
    min_red = min(min_red, red)
    min_height = min(min_height, height)
    max_tilt = max(max_tilt, tilt)

    for i in range(data.ncon):
        c = data.contact[i]
        if (
            extra_id in (int(c.geom1), int(c.geom2))
            or box_id in (int(c.geom1), int(c.geom2))
        ):
            correction_contacts += 1

    failure = None
    if correction_contacts:
        failure = "CONTACT_STOP"
    elif min(est, extra, red) < required:
        failure = "CLEARANCE_STOP"
    elif height < 0.16 or tilt > 12:
        failure = "POSTURE_STOP"

    if failure is not None and correction_reason not in (
        "CONTACT_STOP", "CLEARANCE_STOP", "POSTURE_STOP"
    ):
        correction_reason = failure
        stage = "stopping"
        stage_start = float(data.time)

final_error = abs(
    (yaw_deg() - desired_yaw + 180) % 360 - 180
)

passed = (
    correction_reason == "ANGLE_REACHED"
    and stage == "hold"
    and final_error <= 5.0
    and correction_contacts == 0
    and min(min_estimated, min_extra, min_red) >= required
    and min_height >= 0.16
    and max_tilt <= 12.0
)

print("\n=== Dynamic Correction Result ===")
print("Final XY:", np.round(data.qpos[:2], 4))
print("Final yaw:", f"{yaw_deg():+.2f}deg")
print("Yaw error:", f"{final_error:.2f}deg")
print("Stop reason:", correction_reason)
print("Final stage:", stage)
print("Estimated min clearance:", f"{min_estimated:.3f}m")
print("Extra box min clearance:", f"{min_extra:.3f}m")
print("Red box min clearance:", f"{min_red:.3f}m")
print("Contacts:", correction_contacts)
print("Maximum tilt:", f"{max_tilt:.2f}deg")
print(
    "DYNAMIC_HEADING_CORRECTION:",
    "PASS" if passed else "FAIL"
)
print("Simulation only; not hardware-safe.")

# ===== Guarded walk after dynamic heading correction =====
import ast
import csv

print("\n=== Corrected Heading 20cm Walk ===")

if not passed:
    print("CORRECTED_WALK20: NOT_STARTED")
else:
    depth_path = (
        HERE / "../gate3_obstacle_stop_v01/"
        "obstacle_stop_background_test.py"
    ).resolve()

    tree = ast.parse(depth_path.read_text())
    funcs = [
        node for node in tree.body
        if isinstance(node, ast.FunctionDef)
        and node.name == "obstacle_distance"
    ]
    if len(funcs) != 1:
        raise RuntimeError("Depth detector not found")

    ns = {
        "np": np,
        "math": math,
        "mujoco": mujoco,
        "model": model,
        "data": data,
    }
    exec(
        compile(
            ast.Module(body=funcs, type_ignores=[]),
            str(depth_path),
            "exec",
        ),
        ns,
    )
    detect_depth = ns["obstacle_distance"]

    walk_start = data.qpos[:2].copy()
    walk_yaw = yaw_deg()
    angle = math.radians(walk_yaw)
    direction = np.array([-math.cos(angle), -math.sin(angle)])

    stage = "walk"
    stage_start = float(data.time)
    begin = float(data.time)
    last_depth = -1e9
    nearest = float("nan")
    walk_reason = "RUNNING"
    contacts = 0
    min_est = float("inf")
    min_extra = float("inf")
    min_red = float("inf")
    min_height = float("inf")
    max_tilt = 0.0
    records = []
    depth_count = 0

    print("Start XY:", np.round(walk_start, 4))
    print("Start yaw:", f"{walk_yaw:+.2f}deg")
    print("Distance target: 0.20m")
    print("Clearance threshold: 0.521m")

    with mujoco.Renderer(model, width=640, height=360) as renderer:
        while float(data.time) - begin < 14.0:
            t = float(data.time)
            elapsed = t - stage_start
            xy = data.qpos[:2].copy()
            displacement = xy - walk_start
            forward = float(np.dot(displacement, direction))
            lateral = abs(float(
                direction[0] * displacement[1]
                - direction[1] * displacement[0]
            ))

            est = estimated_clearance()
            extra = xy_clearance(extra_id)
            red = xy_clearance(box_id)
            height, tilt = posture()

            min_est = min(min_est, est)
            min_extra = min(min_extra, extra)
            min_red = min(min_red, red)
            min_height = min(min_height, height)
            max_tilt = max(max_tilt, tilt)

            for j in range(data.ncon):
                c = data.contact[j]
                pair = (int(c.geom1), int(c.geom2))
                if extra_id in pair or box_id in pair:
                    contacts += 1

            if stage == "walk":
                if t - last_depth >= 0.40:
                    last_depth = t
                    depth_count += 1
                    try:
                        nearest = float(detect_depth(renderer))
                    except Exception as exc:
                        print("Depth exception:", repr(exc))
                        nearest = float("nan")
                    print(
                        f"Depth t={t:.2f}s "
                        f"nearest={nearest:.3f}m"
                    )

                yaw_delta = (
                    yaw_deg() - walk_yaw + 180
                ) % 360 - 180

                if not math.isfinite(nearest):
                    walk_reason = "DEPTH_SENSOR_STOP"
                elif nearest < 0.55:
                    walk_reason = "DEPTH_OBSTACLE_STOP"
                elif contacts:
                    walk_reason = "CONTACT_STOP"
                elif min(est, extra, red) < 0.521:
                    walk_reason = "CLEARANCE_STOP"
                elif height < 0.16 or tilt > 12:
                    walk_reason = "POSTURE_STOP"
                elif lateral > 0.08:
                    walk_reason = "PATH_DEVIATION_STOP"
                elif abs(yaw_delta) > 10:
                    walk_reason = "HEADING_STOP"
                elif forward >= 0.20:
                    walk_reason = "WALK20_REACHED"
                elif t - begin >= 10.0:
                    walk_reason = "WALK_TIMEOUT"

                if walk_reason != "RUNNING":
                    stage = "stopping"
                    stage_start = t
                    print("Walk stop:", walk_reason)

            elif stage == "stopping" and elapsed >= 1.0:
                stage = "hold"
                stage_start = t
            elif stage == "hold" and elapsed >= 1.0:
                break

            set_legs(stage, t - stage_start)
            mujoco.mj_step(model, data)

            records.append({
                "time_s": float(data.time),
                "stage": stage,
                "x_m": float(data.qpos[0]),
                "y_m": float(data.qpos[1]),
                "estimated_clearance_m": estimated_clearance(),
                "extra_clearance_m": xy_clearance(extra_id),
                "red_clearance_m": xy_clearance(box_id),
                "contacts": contacts,
                "depth_m": nearest,
            })

    # Check final held state as well.
    final_xy = data.qpos[:2].copy()
    final_forward = float(np.dot(final_xy - walk_start, direction))
    final_est = estimated_clearance()
    final_extra = xy_clearance(extra_id)
    final_red = xy_clearance(box_id)
    min_est = min(min_est, final_est)
    min_extra = min(min_extra, final_extra)
    min_red = min(min_red, final_red)

    out = HERE / "corrected_heading_walk20.csv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0]))
        writer.writeheader()
        writer.writerows(records)

    ok = (
        walk_reason == "WALK20_REACHED"
        and stage == "hold"
        and contacts == 0
        and min(min_est, min_extra, min_red) >= 0.521
        and min_height >= 0.16
        and max_tilt <= 12.0
    )

    print("\n=== Corrected Walk Result ===")
    print("Final XY:", np.round(final_xy, 4))
    print("Final forward:", f"{final_forward:.3f}m")
    print("Stop reason:", walk_reason)
    print("Final stage:", stage)
    print("Depth samples:", depth_count)
    print("Contacts:", contacts)
    print("Min estimated clearance:", f"{min_est:.3f}m")
    print("Min extra clearance:", f"{min_extra:.3f}m")
    print("Min red clearance:", f"{min_red:.3f}m")
    print("Max tilt:", f"{max_tilt:.2f}deg")
    print("CSV:", out)
    print("CORRECTED_WALK20:", "PASS" if ok else "FAIL")
    print("Simulation only; not hardware-safe.")
