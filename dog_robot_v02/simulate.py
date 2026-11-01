"""Educational gait and behavior prototype. Not a validated stable controller."""
import argparse
import math
import time
from pathlib import Path
import mujoco
from mujoco import viewer as mj_viewer

MODEL = Path(__file__).with_name('dog_quadruped.xml')
PHASES = ('rest', 'stand', 'walk', 'approach', 'rest')

def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--headless', action='store_true')
    parser.add_argument('--duration', type=float, default=40)
    args=parser.parse_args()
    model=mujoco.MjModel.from_xml_path(str(MODEL))
    data=mujoco.MjData(model)
    actuator={name:mujoco.mj_name2id(model,mujoco.mjtObj.mjOBJ_ACTUATOR,name+'_drive') for name in [*(f'{l}_{s}_{j}' for l in ('front','hind') for s in ('left','right') for j in ('hip','knee')), 'neck_yaw','neck_pitch','tail_yaw','chest_expand']}
    def update():
        t=data.time
        # Scripted planner stands in for perception and autonomous behavior.
        stage=PHASES[min(int(t // 8),len(PHASES)-1)]
        walking=stage in ('walk','approach')
        for l in ('front','hind'):
            for s in ('left','right'):
                # Diagonal trot phase offsets; open-loop gait (may fall or slip).
                phase=0 if (l,s) in [('front','left'),('hind','right')] else math.pi
                swing=math.sin(2*math.pi*0.8*t+phase)
                hip=math.radians(13)*swing if walking else 0
                knee=math.radians(-14-22*max(0,swing)) if walking else math.radians(-12)
                data.ctrl[actuator[f'{l}_{s}_hip']]=hip
                data.ctrl[actuator[f'{l}_{s}_knee']]=knee
        data.ctrl[actuator['neck_yaw']]=0.2*math.sin(t*0.6) if stage=='approach' else 0
        data.ctrl[actuator['neck_pitch']]=0.08 if stage=='approach' else 0
        data.ctrl[actuator['tail_yaw']]=(0.35 if walking else 0.1)*math.sin(2*math.pi*1.5*t)
        data.ctrl[actuator['chest_expand']]=0.0035*(1+math.sin(2*math.pi*t/3.5))
        return stage
    def loop(viewer=None):
        last_stage=None
        while data.time < args.duration and (viewer is None or viewer.is_running()):
            step_start=time.monotonic()
            stage=update()
            if stage!=last_stage:
                print(f't={data.time:.1f}s state={stage}',flush=True)
                last_stage=stage
            mujoco.mj_step(model,data)
            if viewer:
                viewer.sync()
                time.sleep(max(0,model.opt.timestep-(time.monotonic()-step_start)))
        print(f'complete: simulated={data.time:.2f}s, torso height={data.qpos[2]:.3f}m')
        print('NOTE: scripted motion is not proof of successful walking or stability.')
    if args.headless:
        loop()
    else:
        with mj_viewer.launch_passive(model,data) as viewer:
            loop(viewer)
if __name__=='__main__':
    main()
