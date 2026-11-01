"""Gate 2 integrated open-loop gait / yaw-feedback turning demonstration.

Sequence: stand -> forward 1.0m -> settle -> left 90deg -> settle -> forward 0.5m -> hold.
This is an exploratory MuJoCo simulation, NOT a hardware safety controller.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import mujoco
import numpy as np

LEGS = (("front", "left"), ("front", "right"), ("hind", "left"), ("hind", "right"))
L1 = L2 = .087
REST_Z = -.163


def smoothstep(v):
    v = max(0., min(1., v))
    return v * v * (3. - 2. * v)


def ik(x, z):
    c = max(-1., min(1., (x*x + z*z - L1*L1 - L2*L2) / (2*L1*L2)))
    knee = -math.acos(c)
    hip = math.atan2(-x, -z) - math.atan2(L2*math.sin(knee), L1+L2*math.cos(knee))
    return hip, knee


def trajectory(t, offset, stride, lift, rate):
    p = (rate*t + offset) % 1.
    if p < .65:
        u = p/.65
        return -stride/2 + stride*u, REST_Z
    u = (p-.65)/.35
    return stride/2 - stride*smoothstep(u), REST_Z + lift*math.sin(math.pi*u)


def orientation(data):
    mat = np.zeros(9)
    mujoco.mju_quat2Mat(mat, data.qpos[3:7])
    m = mat.reshape((3, 3))
    yaw = math.atan2(m[1, 0], m[0, 0])
    pitch = math.degrees(math.atan2(-m[2, 0], math.hypot(m[2, 1], m[2, 2])))
    roll = math.degrees(math.atan2(m[2, 1], m[2, 2]))
    return yaw, pitch, roll


def run(a):
    model = mujoco.MjModel.from_xml_path(str(a.model.resolve()))
    data = mujoco.MjData(model)
    if model.nq < 7 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError("Expected floating-base root freejoint")
    ids = {}
    for s, side in LEGS:
        tag = f"{s}_{side}"
        for joint in ("hip", "knee"):
            name = f"{tag}_{joint}_drive"
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            if idx < 0:
                raise RuntimeError("Missing actuator: " + name)
            ids[name] = idx
    for name in ("neck_yaw", "neck_pitch", "tail_yaw", "chest_expand"):
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name+"_drive")
        if idx >= 0:
            ids[name] = idx

    offsets = {f"{s}_{side}": 0. if (s, side) in (("front", "left"), ("hind", "right")) else .5 for s, side in LEGS}
    stages = ["stand", "forward_1m", "pause_1", "turn_left_90", "pause_2", "forward_0p5m", "hold"]
    limits = {"forward_1m": a.max_forward_time, "forward_0p5m": a.max_forward_time, "turn_left_90": a.max_turn_time}
    stage_idx, stage_start = 0, 0.
    base_x, base_y, base_yaw = 0., 0., 0.
    triggered = {}
    timed_out = []
    rows = []
    last_log = -1.
    raw_yaw_prev = None
    yaw_unwrapped = 0.
    viewer_ctx = None
    ended = False
    if a.viewer:
        from mujoco import viewer
        viewer_ctx = viewer.launch_passive(model, data)
        wall_start = time.perf_counter()
    try:
        while True:
            t = float(data.time)
            stage = stages[stage_idx]
            elapsed = t-stage_start
            raw_yaw, pitch, roll = orientation(data)
            if raw_yaw_prev is None:
                raw_yaw_prev = raw_yaw
            else:
                yaw_unwrapped += (raw_yaw-raw_yaw_prev+math.pi)%(2*math.pi)-math.pi
                raw_yaw_prev = raw_yaw

            # Forward distance projected along the heading recorded at entry to each forward stage.
            dx = float(data.qpos[0])-base_x
            dy = float(data.qpos[1])-base_y
            progress = -(dx*math.cos(base_yaw)+dy*math.sin(base_yaw))
            yaw_progress = math.degrees(yaw_unwrapped-base_yaw)
            target = 1.0 if stage == "forward_1m" else .5
            duration = a.settle if stage == "stand" else (a.pause if stage.startswith("pause") else a.hold)
            transition = False
            if stage in ("forward_1m", "forward_0p5m"):
                if progress >= target:
                    transition = True
                    triggered[stage] = dict(time=t, progress=progress)
                elif elapsed >= limits[stage]:
                    transition = True
                    timed_out.append(stage)
                    triggered[stage] = dict(time=t, progress=progress)
            elif stage == "turn_left_90":
                if yaw_progress <= -(a.target_angle-a.lead_angle):
                    transition = True
                    triggered[stage] = dict(time=t, yaw=yaw_progress)
                elif elapsed >= a.max_turn_time:
                    transition = True
                    timed_out.append(stage)
                    triggered[stage] = dict(time=t, yaw=yaw_progress)
            elif elapsed >= duration:
                transition = True
            if transition:
                if stage_idx == len(stages)-1:
                    ended = True
                    break
                stage_idx += 1
                stage = stages[stage_idx]
                stage_start = t
                elapsed = 0.
                base_x, base_y, base_yaw = float(data.qpos[0]), float(data.qpos[1]), yaw_unwrapped
                progress, yaw_progress = 0., 0.

            for s, side in LEGS:
                tag = f"{s}_{side}"
                if stage in ("forward_1m", "forward_0p5m", "turn_left_90"):
                    stride = a.stride*smoothstep(elapsed/a.stride_ramp)
                    lift = a.lift*smoothstep(elapsed/a.lift_ramp)
                    if stage == "turn_left_90":
                        # Consistent with measured -yaw response of the previous turn controller.
                        stride *= -1. if side == "left" else 1.
                    x, z = trajectory(elapsed, offsets[tag], stride, lift, a.rate)
                else:
                    x, z = 0., REST_Z
                hip, knee = ik(x, z)
                for joint, val in (("hip", hip), ("knee", knee)):
                    idx = ids[f"{tag}_{joint}_drive"]
                    lo, hi = model.actuator_ctrlrange[idx]
                    target_ctrl = float(np.clip(val, lo, hi))

                    # Rate-limit joint targets only while stopping/holding.
                    if stage in ("pause_1", "pause_2", "hold") and a.stop_joint_rate > 0:
                        max_change = a.stop_joint_rate * model.opt.timestep
                        previous_ctrl = float(data.ctrl[idx])
                        target_ctrl = float(np.clip(
                            target_ctrl,
                            previous_ctrl - max_change,
                            previous_ctrl + max_change,
                        ))

                    data.ctrl[idx] = target_ctrl
            for name, val in (("neck_yaw", 0.), ("neck_pitch", 0.), ("tail_yaw", 0.),
                              ("chest_expand", .0035*(1.+math.sin(2.*math.pi*t/3.5)))):
                if name in ids:
                    data.ctrl[ids[name]] = val

            mujoco.mj_step(model, data)
            if float(data.time)-last_log >= a.log_dt-1e-9:
                last_log = float(data.time)
                rows.append(dict(time_s=float(data.time), stage=stage, x_m=float(data.qpos[0]),
                                 y_m=float(data.qpos[1]), height_m=float(data.qpos[2]),
                                 yaw_change_deg=math.degrees(yaw_unwrapped), roll_deg=roll,
                                 pitch_deg=pitch, stage_forward_m=progress,
                                 stage_yaw_deg=yaw_progress,
                                 horizontal_speed_mps=math.hypot(*data.qvel[:2]),
                                 max_actuator_force=float(np.max(np.abs(data.actuator_force)))))
            if viewer_ctx is not None:
                if not viewer_ctx.is_running():
                    break
                delay = float(data.time)-(time.perf_counter()-wall_start)
                if delay > 0:
                    time.sleep(delay)
                viewer_ctx.sync()
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()

    if not rows:
        raise RuntimeError("No data recorded")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    with a.out.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    print("Integrated sequence: " + " -> ".join(stages))
    for stage in ("forward_1m", "turn_left_90", "forward_0p5m"):
        event = triggered.get(stage)
        if event is None:
            print(f"{stage}: NOT REACHED")
        else:
            measure = f"{event['progress']:.3f} m" if "progress" in event else f"{event['yaw']:+.2f} deg"
            print(f"{stage}: trigger at {event['time']:.3f}s, {measure}, " + ("TIMEOUT" if stage in timed_out else "threshold reached"))
    for stage in stages:
        r = [row for row in rows if row['stage'] == stage]
        if not r:
            continue
        print(f"{stage}: {r[0]['time_s']:.2f}-{r[-1]['time_s']:.2f}s; max |roll|={max(abs(v['roll_deg']) for v in r):.2f}deg; max |pitch|={max(abs(v['pitch_deg']) for v in r):.2f}deg; min height={min(v['height_m'] for v in r):.3f}m")
    turn_rows = [r for r in rows if r['stage'] == 'turn_left_90']
    hold_rows = [r for r in rows if r['stage'] == 'pause_2']
    if turn_rows and hold_rows:
        final_turn = hold_rows[-1]['yaw_change_deg']-turn_rows[0]['yaw_change_deg']
        print(f"Yaw after turning and pause: {final_turn:+.2f} deg (target -{a.target_angle:.1f}deg)")
    final_hold = [r for r in rows if r['stage'] == 'hold']
    if final_hold:
        print(f"Final hold max horizontal speed: {max(r['horizontal_speed_mps'] for r in final_hold):.4f}m/s")
    print(f"Sequence completed: {ended}; timed-out stages: {timed_out}")
    print(f"CSV: {a.out.resolve()}")
    print("Exploratory simulation only; not certified for foot slip, impacts, motor limits or hardware safety.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model', type=Path, default=Path('dog_quadruped.xml'))
    p.add_argument('--out', type=Path, default=Path('gate2_results/integrated/integrated.csv'))
    p.add_argument('--viewer', action='store_true')
    p.add_argument('--settle', type=float, default=2.)
    p.add_argument('--pause', type=float, default=1.)
    p.add_argument('--hold', type=float, default=3.)
    p.add_argument('--stride', type=float, default=.045)
    p.add_argument('--lift', type=float, default=.012)
    p.add_argument('--rate', type=float, default=.8)
    p.add_argument('--stride-ramp', type=float, default=.5)
    p.add_argument('--lift-ramp', type=float, default=1.)
    p.add_argument('--target-angle', type=float, default=90.)
    p.add_argument('--lead-angle', type=float, default=1.5)
    p.add_argument('--max-forward-time', type=float, default=35.)
    p.add_argument('--max-turn-time', type=float, default=25.)
    p.add_argument('--log-dt', type=float, default=.005)
    p.add_argument(
        '--stop-joint-rate',
        type=float,
        default=2.0,
        help='Max joint target change in rad/s during stop; 0 disables limiting',
    )
    a = p.parse_args()
    if min(a.settle, a.pause, a.hold, a.stride, a.lift, a.rate, a.stride_ramp,
           a.lift_ramp, a.target_angle, a.max_forward_time, a.max_turn_time, a.log_dt) <= 0:
        p.error('All durations and gait parameters must be positive')
    if not 0 <= a.lead_angle < a.target_angle:
        p.error('lead-angle must be >= 0 and less than target-angle')
    run(a)


if __name__ == '__main__':
    main()
