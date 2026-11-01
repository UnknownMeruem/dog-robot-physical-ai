"""Gate 2 diagnostics: stance timeline and contact-point tangential speed.

Reuses gate2_contact_v01 gait definitions; no changes to the dog model.
The metric is point-relative tangential velocity at an active floor/foot contact.
This is a more direct contact slip proxy than foot-center speed, but does not
measure real-world sliding or validate a controller. Requires MuJoCo + NumPy.
"""
import argparse
import csv
import math
import sys
from pathlib import Path

import mujoco
import numpy as np

# Load original gait implementation for identical IK and trajectory.
ORIGINAL = Path(__file__).resolve().parent.parent / 'gate2_contact_v01'
if not (ORIGINAL / 'gate2_contact.py').is_file():
    raise SystemExit(f'Original gait script missing: {ORIGINAL / "gate2_contact.py"}. Put both folders beside dog_robot_v02.')
sys.path.insert(0, str(ORIGINAL))
from gate2_contact import LEGS, ik, trajectory  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, default=Path('dog_quadruped.xml'))
    p.add_argument('--settle', type=float, default=2.0)
    p.add_argument('--walk', type=float, default=8.0)
    p.add_argument('--cooldown', type=float, default=2.0)
    p.add_argument('--ramp-time', type=float, default=1.0)
    p.add_argument('--stride', type=float, default=0.035)
    p.add_argument('--lift', type=float, default=0.018)
    p.add_argument('--rate', type=float, default=0.8)
    p.add_argument('--outdir', type=Path, default=Path('gate2_results/gait_analysis_v02'))
    args = p.parse_args()
    if args.walk <= 0 or args.rate <= 0 or args.stride <= 0 or args.lift < 0:
        p.error('walk/rate/stride must be positive, lift nonnegative')
    m = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
    d = mujoco.MjData(m)
    foot_names = [f'{a}_{b}' for a,b in LEGS]
    foot_geom = {}
    actuators = {}
    for tag in foot_names:
        foot_geom[tag] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, tag+'_foot')
        for joint in ('hip', 'knee'):
            actuator = tag+'_'+joint+'_drive'
            actuators[actuator] = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator)
    for extra in ('neck_yaw', 'neck_pitch', 'tail_yaw', 'chest_expand'):
        actuators[extra] = mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_ACTUATOR,extra+'_drive')
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, 'floor')
    if floor < 0 or any(v<0 for v in foot_geom.values()) or any(v<0 for v in actuators.values()):
        raise SystemExit('Model must include floor, four named feet and V0.2 actuators')
    gid_to_tag = {g:tag for tag,g in foot_geom.items()}
    dt = m.opt.timestep
    t_start = args.settle
    t_end = args.settle + args.walk
    total = t_end + args.cooldown
    rows = []
    jacp=np.zeros((3,m.nv))
    jacr=np.zeros((3,m.nv))
    while d.time < total - dt/2:
        t=float(d.time)
        moving=t_start <= t < t_end
        for section,side in LEGS:
            tag=f'{section}_{side}'
            if moving:
                offset=0 if (section,side) in (('front','left'),('hind','right')) else .5
                ramp = min(1.0, max(0.0, (t - t_start) / args.ramp_time))
                smooth = ramp * ramp * (3.0 - 2.0 * ramp)
                effective_stride = args.stride * smooth
                effective_lift = args.lift * smooth
                x,z=trajectory(t-t_start,offset,effective_stride,effective_lift,args.rate)
            else:
                x,z=0.0,-0.163
            hip,knee=ik(x,z)
            for joint,target in (('hip',hip),('knee',knee)):
                aid=actuators[tag+'_'+joint+'_drive']
                low,high=m.actuator_ctrlrange[aid]
                d.ctrl[aid]=np.clip(target,low,high)
        for n,target in (('neck_yaw',0.0),('neck_pitch',0.0),('tail_yaw',0.0),('chest_expand',0.0035*(1+math.sin(2*math.pi*t/3.5)))):
            d.ctrl[actuators[n]]=target
        mujoco.mj_step(m,d)
        if not moving:
            continue
        # Contact frame: first row is contact normal; remove normal component.
        contacts={tag:[] for tag in foot_names}
        for i in range(d.ncon):
            c=d.contact[i]
            if c.efc_address < 0:
                continue
            g1,g2=int(c.geom1),int(c.geom2)
            if floor not in (g1,g2):
                continue
            other=g2 if g1==floor else g1
            if other not in gid_to_tag:
                continue
            tag=gid_to_tag[other]
            body=int(m.geom_bodyid[other])
            mujoco.mj_jac(m,d,jacp,jacr,np.asarray(c.pos),body)
            v=jacp @ d.qvel  # world-frame contact-point velocity of foot body
            normal=np.asarray(c.frame).reshape(3,3)[0]
            tangent=v-normal*np.dot(v,normal)
            speed=float(np.linalg.norm(tangent))
            force=np.zeros(6)
            mujoco.mj_contactForce(m,d,i,force)
            contacts[tag].append((speed,abs(float(force[0]))))
        values=[float(d.time), float(d.qpos[0]), float(d.qpos[2])]
        for tag in foot_names:
            arr=contacts[tag]
            values += [int(bool(arr)), max((v for v,f in arr),default=0.0), sum(f for v,f in arr)]
        rows.append(values)
    args.outdir.mkdir(parents=True,exist_ok=True)
    csvpath=args.outdir/'contact_points.csv'
    fields=['time_s','torso_x_m','torso_z_m']
    for tag in foot_names:
        fields += [tag+'_contact',tag+'_tangent_speed_mps',tag+'_normal_force_N']
    with csvpath.open('w',newline='') as f:
        w=csv.writer(f);w.writerow(fields);w.writerows(rows)
    dx=rows[-1][1]-rows[0][1] if len(rows)>1 else 0
    print(f'Walk displacement: {dx:.3f} m (forward -X)')
    print('Foot            Contact rate   Mean tangential speed    Peak tangential speed')
    for j,tag in enumerate(foot_names):
        ci=3+3*j
        active=[r[ci+1] for r in rows if r[ci]]
        rate=100*len(active)/len(rows) if rows else 0
        mean=np.mean(active) if active else 0
        peak=max(active) if active else 0
        print(f'{tag:14s} {rate:8.1f}%        {mean:8.4f} m/s          {peak:8.4f} m/s')
    print('CSV:',csvpath.resolve())
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        tt=np.array([r[0] for r in rows])
        fig,axs=plt.subplots(2,1,figsize=(10,5),sharex=True,layout='constrained')
        for j,tag in enumerate(foot_names):
            ci=3+3*j
            yy=np.array([r[ci] for r in rows])
            axs[0].plot(tt, yy+3-j, drawstyle='steps-post', label=tag, linewidth=1)
            speed=np.array([r[ci+1] if r[ci] else np.nan for r in rows])
            axs[1].plot(tt,speed,label=tag,linewidth=1)
        axs[0].set_yticks([1,2,3,4],foot_names[::-1]);axs[0].set_ylabel('Contact (raised=1)')
        axs[0].set_title('Foot contact timeline')
        axs[1].set_ylabel('Contact point tangent speed (m/s)')
        axs[1].set_xlabel('Simulation time (s)')
        axs[1].legend(ncol=2,fontsize=8)
        path=args.outdir/'gait_timeline.png';fig.savefig(path,dpi=140);plt.close(fig)
        print('Plot:',path.resolve())
    except ImportError:
        print('Plot skipped (optional): python -m pip install matplotlib')
    print('NOTE: point tangent speed is a contact slip proxy; no formal gait PASS.')

if __name__ == '__main__':
    main()
