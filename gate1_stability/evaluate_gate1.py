"""Gate 1 standing metrics for the V0.2 dog_quadruped.xml model.
Does not implement a stabilizing controller; measures the existing candidate.
"""
import argparse, csv, math
from pathlib import Path
import numpy as np
import mujoco

p=argparse.ArgumentParser()
p.add_argument('--model', default='dog_quadruped.xml')
p.add_argument('--duration',type=float,default=30.0)
p.add_argument('--out',default='gate1_results.csv')
a=p.parse_args()
model=mujoco.MjModel.from_xml_path(str(Path(a.model).resolve()))
data=mujoco.MjData(model)
if model.nq<7: raise RuntimeError('Model must have a free-joint torso with quaternion qpos[3:7]')
torso_id=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_BODY,'torso')
if torso_id<0: raise RuntimeError('Missing body: torso')
feet=[]
for front in ('front','hind'):
 for side in ('left','right'):
  name=f'{front}_{side}_foot'
  gid=mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_GEOM,name)
  if gid<0: raise RuntimeError(f'Missing foot geometry {name}')
  feet.append(gid)
# Assumes first 8 actuators are hip/knee position servos as V0.2 generator defines.
if model.nu<8: raise RuntimeError('Expected at least 8 leg actuators')
# A neutral position target, not a verified balancing policy.
for i in range(8):
 data.ctrl[i]=0.0 if i%2==0 else math.radians(-12)
rows=[]
while data.time<a.duration:
 mujoco.mj_step(model,data)
 if len(rows)==0 or data.time-rows[-1]['t']>=0.02:
  qw,qx,qy,qz=data.qpos[3:7]
  roll=math.degrees(math.atan2(2*(qw*qx+qy*qz),1-2*(qx*qx+qy*qy)))
  pitch=math.degrees(math.asin(max(-1,min(1,2*(qw*qy-qz*qx)))))
  touched=set()
  for i in range(data.ncon):
   contact=data.contact[i]
   if contact.geom1 in feet and contact.geom2==0: touched.add(contact.geom1)
   if contact.geom2 in feet and contact.geom1==0: touched.add(contact.geom2)
  rows.append(dict(t=round(data.time,4),height=float(data.xpos[torso_id,2]),roll=roll,pitch=pitch,feet=len(touched),max_leg_actuator_force=float(np.max(np.abs(data.actuator_force[:8])))))
  if not np.isfinite(data.qpos).all(): raise RuntimeError('NaN / Inf detected')
with open(a.out,'w',newline='') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
steady=[r for r in rows if r['t']>=2.0]  # ignore start transient; also report total
if not steady: steady=rows
mean_height=sum(r['height'] for r in steady)/len(steady)
checks={
 'height_above_0.17m':all(r['height']>0.17 for r in steady),
 'roll_pitch_within_5deg':all(abs(r['roll'])<=5 and abs(r['pitch'])<=5 for r in steady),
 'height_within_20mm_of_steady_mean':all(abs(r['height']-mean_height)<=0.020 for r in steady),
 'four_feet_contact_at_least_90pct':sum(r['feet']==4 for r in steady)/len(steady)>=0.90,
 'max_leg_actuator_force_within_3Nm':all(r['max_leg_actuator_force']<=3.01 for r in steady),
}
print('Gate1 metrics:', f'height_mean={mean_height:.3f}m',f'min_height={min(r["height"] for r in steady):.3f}m', f'max_abs_roll={max(abs(r["roll"]) for r in steady):.1f}deg',f'max_abs_pitch={max(abs(r["pitch"]) for r in steady):.1f}deg')
for k,v in checks.items():print(('PASS' if v else 'FAIL'),k)
print('OVERALL:', 'PASS (candidate checks only)' if all(checks.values()) else 'FAIL / NEEDS DESIGN ITERATION')
print('Results:',a.out)
