# Gate 2 — Stop control V0.1

Exploratory MuJoCo simulation of stopping from quadruped gait. Based on the
hybrid trajectory (linear stance and smoothstep swing) and independent
startup stride/lift ramps used in Gate 2.

## Installation

Unzip `gate2_stop_v01.zip` under `~/physical_ai_todai/dog_robot/` (next to
`dog_robot_v02`). Activate your existing Python environment with `mujoco` and
`numpy` installed.

## Run

```bash
cd ~/physical_ai_todai/dog_robot/dog_robot_v02
python ../gate2_stop_v01/stop_trial.py --model dog_quadruped.xml --mode abrupt --out gate2_results/stop/abrupt.csv
python ../gate2_stop_v01/stop_trial.py --model dog_quadruped.xml --mode smooth --out gate2_results/stop/smooth.csv
```

`--walk 8` issues the stop at simulation time 10 s (after 2 s settling).
`--stop-ramp 0.7` blends **foot targets** from their commanded walking
positions at stop time to neutral stance. It is **not** a guaranteed 0.7-s
velocity stop and may cause undesirable contacts while swing feet are lowered.

Inspect the CSV `horizontal_speed_mps`, `torso_y_m`, `roll_deg`, `pitch_deg`.
The printed sustained-low-speed time requires 0.2 continuous seconds with
horizontal speed <= 0.02 m/s, starting within 1 s after stop command.

## Limitations

- MuJoCo was not available in the authoring environment; syntax checked, not
  run against your actual model. Please report errors and output.
- `abrupt` resets foot *targets* to standing immediately; `smooth` interpolates
  target foot positions but does not explicitly impose zero body velocity.
- Contact, friction, and collision safety require further analysis.
- Do not treat these results as a real-hardware safety certification.
