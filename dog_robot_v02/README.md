# Dog Robot V0.2 — quadruped research prototype

Educational articulated MJCF model, not a physical manufacturing design. The robot has a free-floating torso, four 2-DOF legs, a 2-DOF neck, tail yaw, and a simplified chest slider (12 actuators, 18 generalized degrees of freedom including the floating base).

## Setup (Ubuntu / Debian)

```bash
sudo apt update && sudo apt install -y python3-venv python3-pip libgl1 libglfw3 libosmesa6 mesa-utils
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip mujoco numpy
python generate_model.py
python simulate.py --headless --duration 5
python simulate.py
```

## Scope and limitations

- Geometry and masses are preliminary estimates only. The model covers roughly 0.4m including head and tail; refine after CAD.
- Head points toward negative X. Leg joints move only in a sagittal plane; 8-DOF walking may not turn robustly.
- The open-loop diagonal gait can slip or fall. No state estimation, inverse kinematics, foot trajectory optimization, balancing, navigation, obstacle perception, or fall recovery yet.
- `approach` is a scripted state and does NOT locate or reach a person.
- The chest slider is a rigid visual proxy, not a deformable soft body, and does not establish safe tactile design.
- No batteries, collision-safe actuation, fault detection, or clinical/wellness features.

## Next development gates

1. Confirm model loads and inspect contacts / base height.
2. Improve standing controller using foot kinematics and measured joint torques.
3. Test slow walking for stability, slip, and falls.
4. Add perception, lift detection, and safe stopping.
5. CAD structural design and tactile material testing before any user interaction.
