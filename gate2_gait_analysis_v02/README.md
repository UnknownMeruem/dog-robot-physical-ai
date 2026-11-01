# Gate 2: contact timing and contact-point slip diagnostic

## Prerequisite folder structure

```
dog_robot/
  dog_robot_v02/dog_quadruped.xml
  gate2_contact_v01/gate2_contact.py
  gate2_gait_analysis_v02/analyze_gait.py
```

## Run from the model folder

```bash
cd ~/physical_ai_todai/dog_robot/dog_robot_v02
# activate your existing mujoco virtual environment if needed
python -m pip install matplotlib  # optional, for a PNG graph
python ../gate2_gait_analysis_v02/analyze_gait.py --model dog_quadruped.xml
```

Output is in `dog_robot_v02/gate2_results/gait_analysis_v02/`:
- `contact_points.csv` (time, contact yes/no, tangential contact point speed, normal contact force)
- `gait_timeline.png` (if matplotlib installed)

It reuses exactly the IK and foot trajectory definitions from `gate2_contact_v01`.
Only the selected active foot-floor contact points are used. Ground is stationary. The velocity is the foot body's contact point velocity projected onto the tangential plane. This is a much better slip proxy than foot-center travel, but is not a measurement of real-world slip and cannot by itself demonstrate stable gait. In a multi-contact situation the script retains the maximum speed per foot and sums the normal forces.

The supplied program was syntax-checked; MuJoCo runtime execution has not been verified in this environment.
