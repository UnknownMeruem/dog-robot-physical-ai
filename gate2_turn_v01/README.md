# Gate 2: experimental differential turning

Run from `dog_robot_v02` with `mujoco` and `numpy` installed:

```bash
python ../gate2_turn_v01/turn_trial.py --model dog_quadruped.xml --direction left --walk 8 --out gate2_results/turn/left.csv
python ../gate2_turn_v01/turn_trial.py --model dog_quadruped.xml --direction right --walk 8 --out gate2_results/turn/right.csv
```

Add `--viewer` for visual inspection; **do not close the viewer until the simulation finishes** if you want a complete run. Numerical testing is best without `--viewer`.

**Caution:** All leg hip/knee axes are lateral (`0 1 0`) so turning needs lateral contact slip; even if positive yaw occurs this is not evidence of hardware-safe turning. This prototype uses an open-loop sequence and abrupt stand target at stop. It only measures yaw and posture, and does not yet implement a 90-degree target or a slip-force check. Forward direction is -X, so turn direction names must be checked against measured yaw / visual motion.
