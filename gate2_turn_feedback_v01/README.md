# Gate 2 turn feedback V0.1

Based on existing turn_trial.py. Monitors unwrapped world yaw at each MuJoCo step and triggers a standing command at target-angle minus lead-angle. It records a 3-second post-stop hold and the signed final angle error. It does not certify stability or safe loads. In the tested robot, left command yielded negative world yaw and right positive world yaw. This mapping is model-dependent.

From dog_robot_v02:

```bash
python ../gate2_turn_feedback_v01/turn_feedback.py --model dog_quadruped.xml --direction left --target-angle 90 --out gate2_results/turn_feedback/left.csv
python ../gate2_turn_feedback_v01/turn_feedback.py --model dog_quadruped.xml --direction right --target-angle 90 --out gate2_results/turn_feedback/right.csv
```

Use --viewer separately for visual inspection; closing viewer ends simulation early.
