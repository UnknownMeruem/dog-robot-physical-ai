import argparse
import csv
import math
from pathlib import Path

import mujoco
import numpy as np

LEG_NAMES = [f'{where}_{side}_{joint}_drive' for where in ('front','hind') for side in ('left','right') for joint in ('hip','knee')]


def run_case(xml_path, name, duration, output_dir):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    torso_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, 'torso')
    if torso_id < 0:
        raise ValueError('body torso not found')
    leg_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, n) for n in LEG_NAMES]
    if any(i < 0 for i in leg_ids):
        raise ValueError('One or more leg actuators missing')

    # Change torso mass as a proxy for additional equipment; inertia scales with mass.
    if name in ('mass_plus_300g', 'mass_plus_600g'):
        added = 0.3 if name == 'mass_plus_300g' else 0.6
        original_mass = float(model.body_mass[torso_id])
        model.body_mass[torso_id] += added
        model.body_inertia[torso_id] *= (original_mass + added) / original_mass
        mujoco.mj_setConst(model, data)

    # Switch off leg actuators genuinely, not just setting targets to zero.
    if name == 'motors_off':
        for idx in leg_ids:
            model.actuator_gainprm[idx, :] = 0.0
            model.actuator_biasprm[idx, :] = 0.0

    # Constant standing joint targets, matching initial standing experiment.
    data.ctrl[:] = 0.0
    for idx in leg_ids:
        if '_knee_' in LEG_NAMES[leg_ids.index(idx)]:
            data.ctrl[idx] = math.radians(-12)
    mujoco.mj_forward(model, data)

    rows = []
    timestep = float(model.opt.timestep)
    nsteps = int(duration / timestep)
    max_torque = 0.0
    force_peak = 0.0
    for k in range(nsteps):
        t = data.time
        data.xfrc_applied[:] = 0
        # Horizontal push at torso COM for 0.2 seconds, after a 2s settling phase.
        if name == 'push_1N' and 2.0 <= t < 2.2:
            data.xfrc_applied[torso_id, 1] = 1.0
        if name == 'push_3N' and 2.0 <= t < 2.2:
            data.xfrc_applied[torso_id, 1] = 3.0
        mujoco.mj_step(model, data)
        if k % max(1, round(0.05 / timestep)) != 0:
            continue
        # quat is free root at qpos[3:7], torso may rotate about itself.
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, data.qpos[3:7])
        # pitch and roll from rotation matrix (standard ZYX representation)
        R = rotation.reshape(3, 3)
        roll = math.degrees(math.atan2(R[2, 1], R[2, 2]))
        pitch = math.degrees(math.atan2(-R[2, 0], math.hypot(R[2, 1], R[2, 2])))
        torso_height = float(data.xpos[torso_id, 2])
        torque = float(max(abs(data.actuator_force[i]) for i in leg_ids))
        max_torque = max(max_torque, torque)
        rows.append({'case':name,'time_s':round(data.time,4),'torso_height_m':round(torso_height,4),
                     'roll_deg':round(roll,3),'pitch_deg':round(pitch,3),
                     'max_leg_actuator_force':round(torque,3)})
        if not np.isfinite(data.qpos).all():
            raise RuntimeError(f'non-finite state at t={data.time}')

    out = output_dir / f'{name}.csv'
    with out.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    max_tilt = max(max(abs(r['roll_deg']),abs(r['pitch_deg'])) for r in rows)
    min_height = min(r['torso_height_m'] for r in rows)
    # Research screening only; does not prove stable walking or physical safety.
    candidate_pass = min_height >= 0.17 and max_tilt <= 5.0 and max_torque <= 3.05
    print(f'{name:16s} height_min={min_height:.3f}m max_tilt={max_tilt:.1f}deg max_act_force={max_torque:.2f}  candidate={"PASS" if candidate_pass else "FAIL"}')
    return {'case':name,'min_height_m':min_height,'max_tilt_deg':max_tilt,'max_actuator_force':max_torque,'candidate_pass':candidate_pass}


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--model',default='dog_quadruped.xml')
    ap.add_argument('--duration',type=float,default=8)
    ap.add_argument('--outdir',default='gate1_extended_results')
    args=ap.parse_args()
    output_dir=Path(args.outdir);output_dir.mkdir(parents=True,exist_ok=True)
    names=['baseline','push_1N','push_3N','mass_plus_300g','mass_plus_600g','motors_off']
    summaries=[run_case(Path(args.model),n,args.duration,output_dir) for n in names]
    with (output_dir/'summary.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(summaries[0]));w.writeheader();w.writerows(summaries)
    print('Saved to:',output_dir.resolve())
    print('NOTE: Candidate checks only. Compare push cases after the force ends, and baseline against motors_off.')

if __name__ == '__main__':
    main()
