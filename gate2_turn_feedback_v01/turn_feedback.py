"""Experimental differential quadruped turn. Intended for dog_quadruped.xml.

Left command: left legs backward (+X) and right legs forward (-X),
Measured model response: left command produces negative world yaw, right produces positive world yaw.
Feedback threshold stops slightly before the requested angle. Simulation only; not safety-certified.
"""
import argparse
import csv
import math
import time
from pathlib import Path

import mujoco
import numpy as np

LEGS = (("front", "left"), ("front", "right"),
        ("hind", "left"), ("hind", "right"))
L1 = L2 = 0.087
REST_Z = -0.163


def smoothstep(u):
    u = min(1.0, max(0.0, u))
    return u*u*(3.0-2.0*u)


def ik(x, z):
    c = max(-1.0, min(1.0, (x*x+z*z-L1*L1-L2*L2)/(2*L1*L2)))
    knee = -math.acos(c)
    hip = math.atan2(-x, -z) - math.atan2(L2*math.sin(knee), L1+L2*math.cos(knee))
    return hip, knee


def trajectory(t, offset, stride, lift, rate):
    phase = (rate*t+offset) % 1.0
    if phase < .65:
        u = phase/.65
        return -stride/2 + stride*u, REST_Z
    u = (phase-.65)/.35
    return stride/2-stride*smoothstep(u), REST_Z+lift*math.sin(math.pi*u)


def yaw_pitch_roll(data):
    # The model's root is a floating base (free joint) at qpos[0:7].
    matrix = np.zeros(9)
    mujoco.mju_quat2Mat(matrix, data.qpos[3:7])
    m = matrix.reshape(3, 3)
    yaw = math.atan2(m[1,0], m[0,0])
    pitch = math.atan2(-m[2,0], math.hypot(m[2,1], m[2,2]))
    roll = math.atan2(m[2,1], m[2,2])
    return yaw, math.degrees(pitch), math.degrees(roll)


def run(a):
    model = mujoco.MjModel.from_xml_path(str(a.model.resolve()))
    data = mujoco.MjData(model)
    if model.nq < 7 or model.nv < 6 or model.jnt_type[0] != mujoco.mjtJoint.mjJNT_FREE:
        raise RuntimeError("Expected a floating-base freejoint first in model")
    ids = {}
    for section, side in LEGS:
        tag = f"{section}_{side}"
        for joint in ("hip", "knee"):
            key = f"{tag}_{joint}_drive"
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, key)
            if idx < 0:
                raise RuntimeError("Missing actuator: " + key)
            ids[key] = idx
    for key in ("neck_yaw", "neck_pitch", "tail_yaw", "chest_expand"):
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, key+"_drive")
        if idx >= 0:
            ids[key] = idx

    deadline = a.settle+a.max_turn_time
    offsets = {f"{s}_{side}": (0.0 if (s, side) in (("front", "left"), ("hind", "right")) else 0.5)
               for s, side in LEGS}
    direction_sign = 1.0 if a.direction == "left" else -1.0
    measured_yaw_sign = -1.0 if a.direction == "left" else 1.0
    rows = []
    last_log = -1.0
    viewer_ctx = None
    if a.viewer:
        from mujoco import viewer
        viewer_ctx = viewer.launch_passive(model, data)
        wall_start = time.perf_counter()
    prev_yaw = None
    unwrapped_yaw = 0.0
    initial_yaw = None
    baseline_yaw = None
    stop_time = None
    trigger_yaw_deg = None
    timed_out = False
    try:
        while True:
            t = float(data.time)
            if stop_time is not None and t >= stop_time + a.hold:
                break
            state = "stand" if t < a.settle else ("turn" if stop_time is None else "hold")
            if state == "turn" and baseline_yaw is None:
                baseline_yaw = unwrapped_yaw
            if state == "turn" and t >= deadline:
                timed_out = True
                stop_time = t
                trigger_yaw_deg = math.degrees(unwrapped_yaw - baseline_yaw)
                state = "hold"
            for section, side in LEGS:
                tag = f"{section}_{side}"
                if state == "turn":
                    elapsed = t-a.settle
                    stride_scale = smoothstep(elapsed/a.stride_ramp)
                    lift_scale = smoothstep(elapsed/a.lift_ramp)
                    # Left-turn hypothesis: left feet travel backward, right feet forward.
                    sign = (-direction_sign if side == "left" else direction_sign)
                    x, z = trajectory(elapsed, offsets[tag], sign*a.stride*stride_scale,
                                      a.lift*lift_scale, a.rate)
                else:
                    x, z = 0.0, REST_Z
                hip, knee = ik(x, z)
                for joint, val in (("hip", hip), ("knee", knee)):
                    idx = ids[f"{tag}_{joint}_drive"]
                    lo, hi = model.actuator_ctrlrange[idx]
                    data.ctrl[idx] = np.clip(val, lo, hi)
            for key, val in (("neck_yaw", 0.0), ("neck_pitch", 0.0), ("tail_yaw", 0.0),
                             ("chest_expand", .0035*(1+math.sin(2*math.pi*t/3.5)))):
                if key in ids:
                    data.ctrl[ids[key]] = val
            mujoco.mj_step(model, data)
            yaw, pitch, roll = yaw_pitch_roll(data)
            if prev_yaw is None:
                prev_yaw = yaw
                initial_yaw = yaw
            else:
                delta = (yaw-prev_yaw+math.pi) % (2*math.pi)-math.pi
                unwrapped_yaw += delta
                prev_yaw = yaw
            if state == "turn" and baseline_yaw is not None:
                current_deg = math.degrees(unwrapped_yaw - baseline_yaw)
                if measured_yaw_sign * current_deg >= max(0.0, a.target_angle - a.lead_angle):
                    stop_time = float(data.time)
                    trigger_yaw_deg = current_deg
            if float(data.time)-last_log >= a.log_dt-1e-9:
                last_log = float(data.time)
                rows.append(dict(time_s=float(data.time), state=state,
                                 x_m=float(data.qpos[0]), y_m=float(data.qpos[1]),
                                 height_m=float(data.qpos[2]),
                                 yaw_change_deg=math.degrees(unwrapped_yaw),
                                 roll_deg=roll, pitch_deg=pitch,
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
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    turn = [r for r in rows if r["state"] == "turn"]
    hold = [r for r in rows if r["state"] == "hold"]
    if not turn:
        raise RuntimeError("Turn not recorded; viewer may have closed")
    yaw = turn[-1]["yaw_change_deg"] - turn[0]["yaw_change_deg"]
    dx = turn[-1]["x_m"] - turn[0]["x_m"]
    dy = turn[-1]["y_m"] - turn[0]["y_m"]
    if stop_time is None:
        print("WARNING: simulation ended before a stop decision")
    print(f"Direction requested: {a.direction}; target magnitude: {a.target_angle:.2f} deg")
    print(f"Yaw threshold: {max(0.0,a.target_angle-a.lead_angle):.2f} deg; maximum turn time: {a.max_turn_time:.2f}s")
    print(f"Stop trigger: {stop_time if stop_time is not None else float('nan'):.3f}s; measured yaw at trigger: {trigger_yaw_deg if trigger_yaw_deg is not None else float('nan'):+.2f} deg")
    print(f"Timed out: {timed_out}")
    print(f"Actual turn interval: {turn[0]['time_s']:.3f}-{turn[-1]['time_s']:.3f}s")
    print(f"Yaw change during turn: {yaw:+.2f} deg (positive=CCW around world Z)")
    if baseline_yaw is not None:
        final_deg = math.degrees(unwrapped_yaw - baseline_yaw)
        signed_target = measured_yaw_sign * a.target_angle
        print(f"Final yaw change relative to turn start: {final_deg:+.2f} deg")
        print(f"Final signed yaw error: {final_deg-signed_target:+.2f} deg; absolute error: {abs(final_deg-signed_target):.2f} deg")
        print(f"Target accuracy within ±{a.tolerance:.1f} deg: {('YES' if abs(final_deg-signed_target)<=a.tolerance and not timed_out and stop_time is not None and hold else 'NO / incomplete')}")
    print(f"X/Y displacement during turn: {dx:+.3f} / {dy:+.3f} m")
    print(f"Max |roll| during turn: {max(abs(r['roll_deg']) for r in turn):.2f} deg")
    print(f"Max |pitch| during turn: {max(abs(r['pitch_deg']) for r in turn):.2f} deg")
    print(f"Minimum height during turn: {min(r['height_m'] for r in turn):.3f} m")
    print(f"Max actuator force during turn: {max(r['max_actuator_force'] for r in turn):.3f} (actuator force units)")
    if hold:
        print(f"Yaw change after stop: {hold[-1]['yaw_change_deg']-turn[-1]['yaw_change_deg']:+.2f} deg")
    else:
        print("WARNING: hold not recorded (viewer may have closed early)")
    print(f"CSV: {a.out.resolve()}")
    print("Experimental simulation. No 90-degree PASS or hardware/foot-slip safety certification.")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=Path, default=Path("dog_quadruped.xml"))
    p.add_argument("--out", type=Path, default=Path("gate2_results/turn/turn_trial.csv"))
    p.add_argument("--direction", choices=("left", "right"), default="left")
    p.add_argument("--settle", type=float, default=2.0)
    p.add_argument("--target-angle", type=float, default=90.0, help="Desired absolute yaw angle in degrees")
    p.add_argument("--lead-angle", type=float, default=1.5, help="Stop command this many degrees early")
    p.add_argument("--tolerance", type=float, default=5.0)
    p.add_argument("--max-turn-time", type=float, default=25.0, help="Safety timeout in simulated seconds")
    p.add_argument("--hold", type=float, default=3.0)
    p.add_argument("--stride", type=float, default=.045)
    p.add_argument("--lift", type=float, default=.012)
    p.add_argument("--rate", type=float, default=.8)
    p.add_argument("--stride-ramp", type=float, default=.5)
    p.add_argument("--lift-ramp", type=float, default=1.0)
    p.add_argument("--log-dt", type=float, default=.005)
    p.add_argument("--viewer", action="store_true")
    a = p.parse_args()
    if min(a.settle, a.max_turn_time, a.target_angle, a.hold, a.stride, a.lift, a.rate,
           a.stride_ramp, a.lift_ramp, a.log_dt) <= 0:
        p.error("All durations, gait parameters and log-dt must be positive")
    if a.lead_angle < 0 or a.lead_angle >= a.target_angle or a.tolerance <= 0:
        p.error("Lead angle must be >=0 and < target; tolerance must be positive")
    run(a)

if __name__ == "__main__":
    main()
