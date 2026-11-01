"""Gate 2 contact / slip diagnostic: forward-walk test for dog_robot_v02/dog_quadruped.xml.

Uses a sagittal 2-link foot trajectory + inverse kinematics with an
open-loop diagonal trot. NOT a balancing controller or a PASS certification.
"""
import argparse
import csv
import math
from pathlib import Path
import time

import mujoco
import numpy as np

LEGS = (("front", "left"), ("front", "right"),
        ("hind", "left"), ("hind", "right"))
L1 = L2 = 0.087  # metres, lengths in original generated model


def ik(x, z):
    """Planar foot-to-hip IK; X forward is negative in this model."""
    r2 = x*x + z*z
    c = max(-1.0, min(1.0, (r2-L1*L1-L2*L2)/(2*L1*L2)))
    knee = -math.acos(c)
    hip = math.atan2(-x, -z) - math.atan2(L2*math.sin(knee), L1+L2*math.cos(knee))
    return hip, knee


def trajectory(t, offset, stride, lift, rate):
    p = (rate*t + offset) % 1.0
    # 65% stance, 35% swing. Foot moves +X during stance because head faces -X.
    if p < .65:
        u = p / .65
        x = -stride/2 + stride*u
        z = -0.163
    else:
        u = (p-.65)/.35
        x = stride/2 - stride*u
        z = -0.163 + lift*math.sin(math.pi*u)
    return x, z


def run(args):
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)
    names = {}
    for section, side in LEGS:
        for joint in ("hip", "knee"):
            key = f'{section}_{side}_{joint}_drive'
            idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, key)
            if idx < 0:
                raise ValueError(f'Missing actuator {key}. Use the V0.2 dog_quadruped.xml model.')
            names[key] = idx
    for n in ('neck_yaw', 'neck_pitch', 'tail_yaw', 'chest_expand'):
        idx = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n+'_drive')
        if idx >= 0:
            names[n] = idx
    # Active foot contacts and foot-center horizontal speed proxy.
    floor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    foot_ids = {f'{a}_{b}': mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, f'{a}_{b}_foot') for a,b in LEGS}
    if floor_id < 0 or any(v < 0 for v in foot_ids.values()):
        raise ValueError('Required floor / *_foot geoms absent; use V0.2 XML.')
    counts = {tag: 0 for tag in foot_ids}
    slip_dist = {tag: 0.0 for tag in foot_ids}
    samples_walk = 0
    nonfoot_contact_samples = 0
    max_foot_speed = {tag: 0.0 for tag in foot_ids}
    foot_last_xy = None
    detailed = []
    # run standing warm-up before walking, then return to standing
    start = args.settle
    stop = args.settle + args.walk
    total = stop + args.cooldown
    steps = math.ceil(total / model.opt.timestep)
    samples = []
    last_log = -1.0
    viewer_ctx = None
    if args.viewer:
        from mujoco import viewer as mj_viewer
        viewer_ctx = mj_viewer.launch_passive(model, data)

    try:
        start_x = None
        for _ in range(steps):
            tick = time.monotonic()
            t = float(data.time)
            moving = start <= t < stop
            if start_x is None and t >= start:
                start_x = float(data.qpos[0])
            for section, side in LEGS:
                tag = f'{section}_{side}'
                if moving:
                    offset = 0.0 if (section,side) in (("front","left"),("hind","right")) else 0.5
                    x, z = trajectory(t-start, offset, args.stride, args.lift, args.rate)
                else:
                    x,z = 0.0, -0.163
                hip, knee = ik(x,z)
                for joint,val in (("hip",hip),("knee",knee)):
                    idx = names[f'{tag}_{joint}_drive']
                    low,high = model.actuator_ctrlrange[idx]
                    data.ctrl[idx] = np.clip(val, low, high)
            for extra, val in (("neck_yaw",0.0),("neck_pitch",0.0),
                               ("tail_yaw",0.0),("chest_expand",0.0035*(1+math.sin(2*math.pi*t/3.5)))):
                if extra in names:
                    data.ctrl[names[extra]] = val
            mujoco.mj_step(model,data)
            # Measure every physics step, but accumulate metrics only during walking.
            active = set()
            nonfoot_floor = False
            for ci in range(data.ncon):
                c = data.contact[ci]
                if c.efc_address < 0:
                    continue  # inactive contact does not support load
                g1, g2 = int(c.geom1), int(c.geom2)
                if floor_id not in (g1,g2):
                    continue
                other = g2 if g1 == floor_id else g1
                matches = [name for name, gid in foot_ids.items() if gid == other]
                if matches:
                    active.update(matches)
                else:
                    nonfoot_floor = True
            xy = {tag: np.array(data.geom_xpos[gid][:2], dtype=float).copy() for tag,gid in foot_ids.items()}
            hspeed = {tag: (float(np.linalg.norm(xy[tag] - foot_last_xy[tag])) / model.opt.timestep
                            if foot_last_xy is not None else 0.0) for tag in foot_ids}
            foot_last_xy = xy
            if moving:
                samples_walk += 1
                if nonfoot_floor:
                    nonfoot_contact_samples += 1
                for tag in foot_ids:
                    if tag in active:
                        counts[tag] += 1
                        slip_dist[tag] += hspeed[tag] * model.opt.timestep
                        max_foot_speed[tag] = max(max_foot_speed[tag], hspeed[tag])
                detailed.append([float(data.time), float(data.qpos[0]), float(data.qpos[2]),
                                 int(nonfoot_floor), *(int(tag in active) for tag in foot_ids),
                                 *(round(hspeed[tag], 6) if tag in active else 0.0 for tag in foot_ids)])
            if t-last_log >= 0.02:
                last_log = t
                rotation=np.zeros(9)
                mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
                mat=rotation.reshape((3,3))
                roll=math.degrees(math.atan2(mat[2,1],mat[2,2]))
                pitch=math.degrees(math.atan2(-mat[2,0], math.hypot(mat[2,1],mat[2,2])))
                samples.append((float(data.time), 'walk' if moving else 'stand',
                                float(data.qpos[0]),float(data.qpos[1]),float(data.qpos[2]),
                                roll,pitch,float(np.max(np.abs(data.actuator_force)))))
            if viewer_ctx is not None:
                if not viewer_ctx.is_running():
                    break
                viewer_ctx.sync()
                time.sleep(max(0,model.opt.timestep-(time.monotonic()-tick)))
    finally:
        if viewer_ctx is not None:
            viewer_ctx.close()
    out=Path(args.out)
    out.parent.mkdir(parents=True,exist_ok=True)
    with out.open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['time_s','state','torso_x_m','torso_y_m','torso_z_m','roll_deg','pitch_deg','max_actuator_force'])
        w.writerows(samples)
    walking=[row for row in samples if row[1]=='walk']
    if len(walking)>=2:
        displacement=walking[-1][2]-walking[0][2]
        print(f'Walk-window displacement X: {displacement:+.3f} m (forward is -X)')
        print(f'Min height during walk: {min(row[4] for row in walking):.3f} m')
        print(f'Max |roll| during walk: {max(abs(row[5]) for row in walking):.2f} deg')
        print(f'Max |pitch| during walk: {max(abs(row[6]) for row in walking):.2f} deg')
    metric_path = out.with_name('contact_slip_timeseries.csv')
    with metric_path.open('w', newline='') as f:
        w=csv.writer(f)
        w.writerow(['time_s','torso_x_m','torso_z_m','nonfoot_floor_contact',
                    *(f'{tag}_contact' for tag in foot_ids),
                    *(f'{tag}_foot_center_speed_mps' for tag in foot_ids)])
        w.writerows(detailed)
    print('--- Contact / foot-center motion diagnostics (walk window only) ---')
    for tag in foot_ids:
        pct = 100 * counts[tag] / samples_walk if samples_walk else 0
        print(f'{tag:12s} contact={pct:5.1f}%  contact_center_travel={1000*slip_dist[tag]:7.1f}mm  peak_center_speed={max_foot_speed[tag]:.3f}m/s')
    print(f'Non-foot floor-contact fraction: {100*nonfoot_contact_samples/samples_walk if samples_walk else 0:.1f}%')
    print('WARNING: foot-center motion while contacting is only a SLIP PROXY, not true contact-point slip.')
    print('WARNING: active contact may indicate support; contact rate alone does not prove good gait.')
    print('CSV:',out.resolve())
    print('Contact CSV:',metric_path.resolve())
    print('This is an open-loop gait experiment; no claim of stable walking or 1 m travel.')


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',type=Path,default=Path('dog_quadruped.xml'))
    ap.add_argument('--out',default='gate2_results/forward_contact_trial.csv')
    ap.add_argument('--viewer',action='store_true')
    ap.add_argument('--settle',type=float,default=2)
    ap.add_argument('--walk',type=float,default=8)
    ap.add_argument('--cooldown',type=float,default=2)
    ap.add_argument('--stride',type=float,default=0.035)
    ap.add_argument('--lift',type=float,default=0.018)
    ap.add_argument('--rate',type=float,default=0.8)
    run(ap.parse_args())
