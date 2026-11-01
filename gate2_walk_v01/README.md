# Gate 2-1 — Forward locomotion experiment

This runs against your existing `dog_robot_v02/dog_quadruped.xml` without modifying the model.

## Run from dog_robot_v02 on Linux

```bash
python ../gate2_walk_v01/gate2_forward.py --model dog_quadruped.xml
python ../gate2_walk_v01/gate2_forward.py --model dog_quadruped.xml --viewer
```

Output: `gate2_results/forward_trial.csv`. The nose faces **-X**, so forward displacement is negative X.

## Important limitations

- Open-loop two-link planar inverse kinematics; not a proven forward gait.
- Four legs have only two joints each, no active lateral foot placement.
- We have not validated the XML+controller together in MuJoCo in this environment.
- View gait for tumbling, slipping, and whether feet actually alternate contact; report displacement AND height/tilt.
- If knees rapidly strike joint limits or the body falls, stop; shorten stride or reduce gait frequency, but investigate model geometry before tuning.

## Next step

Add ground-contact force logging and closed-loop stance recovery before declaring Gate 2 PASS.
