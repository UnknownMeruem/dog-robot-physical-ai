"""Gate 2 reverse walking and stop experiments. Run from dog_robot_v02 with the V0.2 MJCF.

Uses the same hybrid trajectory and independent lift/stride startup ramps
as gate2_lift_ramp_v01. Compares abrupt stand command vs smooth foot-target
interpolation. This is exploratory simulation, not a safety certification.
"""
import argparse
import csv
import math
from pathlib import Path
import sys

import mujoco
import numpy as np

LEGS = (("front", "left"), ("front", "right"), ("hind", "left"), ("hind", "right"))
L1 = L2 = 0.087
REST_Z = -0.163


def smoothstep(u):
    u = max(0.0, min(1.0, u))
    return u * u * (3.0 - 2.0 * u)


def ik(x, z):
    r2 = x*x + z*z
    c = max(-1.0, min(1.0, (r2-L1*L1-L2*L2)/(2*L1*L2)))
    knee = -math.acos(c)
    hip = math.atan2(-x, -z) - math.atan2(L2*math.sin(knee), L1+L2*math.cos(knee))
    return hip, knee


def trajectory(t, offset, stride, lift, rate):
    p = (rate * t + offset) % 1.0
    if p < 0.65:
        u = p / 0.65
        return -stride/2 + stride*u, REST_Z
    u = (p - 0.65) / 0.35
    s = smoothstep(u)
    return stride/2 - stride*s, REST_Z + lift*math.sin(math.pi*u)


def walk_target(elapsed, offset, args):
    stride_scale = smoothstep(elapsed / args.start_stride_ramp)
    lift_scale = smoothstep(elapsed / args.start_lift_ramp)
    sign = -1.0 if args.direction == "backward" else 1.0
    return trajectory(elapsed, offset, sign*args.stride*stride_scale,
                      args.lift*lift_scale, args.rate)


def load_actuators(model):
    ids = {}
    for section, side in LEGS:
        for joint in ("hip", "knee"):
            key = f"{section}_{side}_{joint}_drive"
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, key)
            if idx < 0:
                raise ValueError(f"Missing actuator: {key}")
            ids[key] = idx
    for n in ("neck_yaw", "neck_pitch", "tail_yaw", "chest_expand"):
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n + "_drive")
        if idx >= 0:
            ids[n] = idx
    return ids


def run(args):
    model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    data = mujoco.MjData(model)
    ids = load_actuators(model)
    start = args.settle
    stop = start + args.walk
    finish = stop + args.stop_ramp
    total = finish + args.hold
    steps = math.ceil(total / model.opt.timestep)
    offsets = {f"{s}_{side}": (0.0 if (s, side) in (("front", "left"), ("hind", "right")) else 0.5)
               for s, side in LEGS}
    # Freeze the foot targets at the stop command, then blend toward neutral stance.
    frozen = {tag: walk_target(args.walk, phase, args) for tag, phase in offsets.items()}
    rows = []
    previous_log = -1.0
    viewer_ctx = None
    if args.viewer:
        import time
        from mujoco import viewer
        viewer_ctx = viewer.launch_passive(model, data)
        wall_start = time.perf_counter()
    try:
        for _ in range(steps):
            t = float(data.time)
            if t < start:
                state = "stand"
            elif t < stop:
                state = "walk"
            elif t < finish:
                state = "brake"
            else:
                state = "hold"
            for s, side in LEGS:
                tag = f"{s}_{side}"
                if state == "walk":
                    x, z = walk_target(t-start, offsets[tag], args)
                elif state == "brake" and args.mode == "smooth":
                    alpha = smoothstep((t-stop)/args.stop_ramp)
                    fx, fz = frozen[tag]
                    x, z = (1-alpha)*fx, (1-alpha)*fz + alpha*REST_Z
                else:
                    x, z = 0.0, REST_Z
                hip, knee = ik(x, z)
                for joint, val in (("hip", hip), ("knee", knee)):
                    idx = ids[f"{tag}_{joint}_drive"]
                    lo, hi = model.actuator_ctrlrange[idx]
                    data.ctrl[idx] = np.clip(val, lo, hi)
            for extra, value in (("neck_yaw", 0.0), ("neck_pitch", 0.0),
                                 ("tail_yaw", 0.0),
                                 ("chest_expand", 0.0035*(1+math.sin(2*math.pi*t/3.5)))):
                if extra in ids:
                    data.ctrl[ids[extra]] = value
            mujoco.mj_step(model, data)
            if data.time - previous_log >= args.log_dt - 1e-9:
                previous_log = float(data.time)
                mat = np.zeros(9)
                mujoco.mju_quat2Mat(mat, data.qpos[3:7])
                mat = mat.reshape((3, 3))
                roll = math.degrees(math.atan2(mat[2,1], mat[2,2]))
                pitch = math.degrees(math.atan2(-mat[2,0], math.hypot(mat[2,1], mat[2,2])))
                speed = math.hypot(float(data.qvel[0]), float(data.qvel[1]))
                rows.append(dict(time_s=float(data.time), state=state,
                                 torso_x_m=float(data.qpos[0]), torso_y_m=float(data.qpos[1]),
                                 torso_z_m=float(data.qpos[2]), horizontal_speed_mps=speed,
                                 roll_deg=roll, pitch_deg=pitch,
                                 max_actuator_force=float(np.max(np.abs(data.actuator_force)))))
            if viewer_ctx is not None:
                if not viewer_ctx.is_running():
                    break
                elapsed_wall = time.perf_counter() - wall_start
                delay = float(data.time) - elapsed_wall
                if delay > 0:
                    time.sleep(delay)
                viewer_ctx.sync()
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    post = [r for r in rows if r["time_s"] >= stop]
    hold = [r for r in rows if r["state"] == "hold"]
    walk = [r for r in rows if r["state"] == "walk"]
    print(f"Direction: {args.direction}; stop mode: {args.mode}; command at t={stop:.3f}s")
    if walk:
        dx = walk[-1]["torso_x_m"] - walk[0]["torso_x_m"]
        print(f"Walk displacement X: {dx:+.3f} m (+X is backward)")
        ok = dx >= 0.5 if args.direction == "backward" else dx <= -1.0
        print(f"Distance target: {'reached' if ok else 'not reached'} (sim only)")
    if post:
        print(f"Max |roll| after stop: {max(abs(r['roll_deg']) for r in post):.2f} deg")
        print(f"Max |pitch| after stop: {max(abs(r['pitch_deg']) for r in post):.2f} deg")
        print(f"Min torso height after stop: {min(r['torso_z_m'] for r in post):.3f} m")
    if post:
        x0 = min(rows, key=lambda r: abs(r["time_s"]-stop))["torso_x_m"]
        peak = max(abs(r["torso_x_m"]-x0) for r in post)
        print(f"Max |X displacement| after stop command: {peak*1000:.1f} mm")
    if hold:
        print(f"Mean horizontal speed during hold: {sum(r['horizontal_speed_mps'] for r in hold)/len(hold):.4f} m/s")
        print(f"Max horizontal speed during hold: {max(r['horizontal_speed_mps'] for r in hold):.4f} m/s")
    # Require a whole contiguous 0.2s of low velocity, at or after command.
    window = 0.2
    settle_time = None
    for i, row in enumerate(post):
        if row["time_s"] > stop + 1.0:
            break
        future = [r for r in post[i:] if r["time_s"] < row["time_s"] + window]
        if future and future[-1]["time_s"] - row["time_s"] >= window-args.log_dt*1.1:
            if all(r["horizontal_speed_mps"] <= args.speed_threshold for r in future):
                settle_time = row["time_s"] - stop
                break
    print("Time to sustained speed <= {:.3f} m/s (0.2s window): {}".format(
        args.speed_threshold, "not confirmed within 1s" if settle_time is None else f"{settle_time:.3f}s"))
    print(f"CSV: {args.out.resolve()}")
    print("Exploratory open-loop simulation only; joint loads, foot slip, collision and hardware safety are not certified.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("dog_quadruped.xml"))
    p.add_argument("--out", type=Path, default=Path("gate2_results/reverse/reverse_trial.csv"))
    p.add_argument("--direction", choices=("backward", "forward"), default="backward")
    p.add_argument("--mode", choices=("abrupt", "smooth"), default="smooth")
    p.add_argument("--settle", type=float, default=2.0)
    p.add_argument("--walk", type=float, default=12.0, help="Seconds walking before stop command")
    p.add_argument("--stop-ramp", type=float, default=0.7)
    p.add_argument("--hold", type=float, default=3.0)
    p.add_argument("--stride", type=float, default=0.045)
    p.add_argument("--lift", type=float, default=0.012)
    p.add_argument("--rate", type=float, default=0.8)
    p.add_argument("--start-stride-ramp", type=float, default=0.5)
    p.add_argument("--start-lift-ramp", type=float, default=1.0)
    p.add_argument("--speed-threshold", type=float, default=0.02)
    p.add_argument("--log-dt", type=float, default=0.005)
    p.add_argument("--viewer", action="store_true")
    args = p.parse_args()
    if min(args.stop_ramp, args.walk, args.hold, args.start_stride_ramp,
           args.start_lift_ramp, args.log_dt) <= 0:
        p.error("durations and log-dt must be positive")
    run(args)


if __name__ == "__main__":
    main()
