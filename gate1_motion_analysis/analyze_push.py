"""Gate 1: quantify lateral displacement, velocity and recovery after a lateral push.

Separate, reproducible baseline and 3 N runs. Uses the same standing controls
as test_gate1_extended.py. Requires a free-root torso and eight named leg motors.
"""
import argparse
import csv
import math
from pathlib import Path

import mujoco
import numpy as np

LEG_NAMES = [f'{where}_{side}_{joint}_drive'
             for where in ('front', 'hind')
             for side in ('left', 'right')
             for joint in ('hip', 'knee')]


def run(model_path, force_n, duration, out_path):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    if torso_id < 0 or model.nq < 7 or model.nv < 6:
        raise ValueError('Expected a free-root torso named torso')
    ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n)
           for n in LEG_NAMES]
    if min(ids) < 0:
        raise ValueError('One or more leg drives missing; confirm XML model')
    data.ctrl[:] = 0.0
    for name, idx in zip(LEG_NAMES, ids):
        if '_knee_' in name:
            data.ctrl[idx] = math.radians(-12)
    mujoco.mj_forward(model, data)
    dt = float(model.opt.timestep)
    steps = math.ceil(duration / dt)
    rows = []
    for _ in range(steps):
        t = float(data.time)
        data.xfrc_applied[:] = 0.0
        if 2.0 <= t < 2.2:
            data.xfrc_applied[torso_id, 1] = force_n
        mujoco.mj_step(model, data)
        if not np.isfinite(data.qpos).all() or not np.isfinite(data.qvel).all():
            raise RuntimeError(f'Non-finite state at {data.time:.3f}s')
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
        r = rotation.reshape(3, 3)
        roll = math.degrees(math.atan2(r[2, 1], r[2, 2]))
        pitch = math.degrees(math.atan2(-r[2, 0], math.hypot(r[2, 1], r[2, 2])))
        rows.append(dict(time_s=float(data.time), torso_y_m=float(data.qpos[1]),
                         torso_vy_m_s=float(data.qvel[1]), torso_height_m=float(data.xpos[torso_id, 2]),
                         roll_deg=roll, pitch_deg=pitch,
                         push_n=float(force_n if 2.0 <= t < 2.2 else 0.0)))
    with out_path.open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def sustained_recovery_time(rows, peak_start=2.2, tol_roll_deg=.05, tol_vy=.01, hold=.2):
    """After push-off, first time roll and lateral velocity remain small for hold seconds.

    This measures settling near zero tilt/velocity, NOT return to original Y position.
    """
    times = np.array([r['time_s'] for r in rows])
    dt = float(np.median(np.diff(times)))
    hold_steps = max(1, math.ceil(hold / dt))
    for i, r in enumerate(rows):
        if r['time_s'] < peak_start:
            continue
        seg = rows[i:i+hold_steps]
        if len(seg) < hold_steps:
            break
        if all(abs(s['roll_deg']) <= tol_roll_deg and abs(s['torso_vy_m_s']) <= tol_vy for s in seg):
            return max(0.0, r['time_s'] - peak_start)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--model', default='dog_quadruped.xml')
    ap.add_argument('--duration', type=float, default=8.0)
    ap.add_argument('--force', type=float, default=3.0)
    ap.add_argument('--outdir', default='gate1_motion_results')
    args = ap.parse_args()
    if args.duration < 3.0:
        ap.error('--duration must be >= 3 seconds')
    dest = Path(args.outdir)
    dest.mkdir(parents=True, exist_ok=True)
    baseline = run(Path(args.model), 0.0, args.duration, dest / 'baseline.csv')
    pushed = run(Path(args.model), args.force, args.duration, dest / 'push.csv')
    # Comparisons at identical simulation timesteps, not rounded print values.
    delta = []
    for b, p in zip(baseline, pushed):
        delta.append(dict(time_s=p['time_s'], delta_y_m=p['torso_y_m']-b['torso_y_m'],
                          delta_vy_m_s=p['torso_vy_m_s']-b['torso_vy_m_s'],
                          delta_roll_deg=p['roll_deg']-b['roll_deg']))
    with (dest / 'difference.csv').open('w', newline='') as fp:
        writer = csv.DictWriter(fp, fieldnames=list(delta[0]))
        writer.writeheader()
        writer.writerows(delta)
    after_push = [r for r in delta if r['time_s'] >= 2.0]
    maximum_lateral_displacement = max(abs(r['delta_y_m']) for r in after_push)
    maximum_lateral_speed_change = max(abs(r['delta_vy_m_s']) for r in after_push)
    maximum_roll_change = max(abs(r['delta_roll_deg']) for r in after_push)
    # Recovery uses differences from baseline to distinguish push effects from natural settling.
    recovery_rows = [dict(time_s=r['time_s'], roll_deg=r['delta_roll_deg'], torso_vy_m_s=r['delta_vy_m_s']) for r in delta]
    recovery = sustained_recovery_time(recovery_rows)
    end_y = delta[-1]['delta_y_m']
    print(f'Applied: {args.force:.2f} N along world +Y from 2.0s to 2.2s')
    print(f'Peak |Y position difference|: {maximum_lateral_displacement * 1000:.2f} mm')
    print(f'Peak |Y speed difference|: {maximum_lateral_speed_change:.4f} m/s')
    print(f'Peak |roll difference|: {maximum_roll_change:.4f} deg')
    print(f'Final Y position difference: {end_y * 1000:.2f} mm')
    print('Settling time after push ends (|delta roll|<=0.05 deg and '
          '|delta vy|<=0.01 m/s for 0.2s):',
          f'{recovery:.3f}s' if recovery is not None else 'not achieved within duration')
    print('Note: settling does not imply return to original location, and no formal PASS is assigned.')
    print('Saved:', dest.resolve())

if __name__ == '__main__':
    main()
