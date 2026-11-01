# Dog Robot — Gate 3 Autonomous Navigation

## Current status

Gate 3 provisional progress: 98%.

### Demonstrated in MuJoCo

- Quadruped walking, turning, braking and holding
- Forward depth-camera obstacle detection
- 3D point cloud to world-coordinate estimation
- Camera-derived waypoint generation
- Post-turn heading correction
- Integrated waypoint and final-goal traversal
- Contact, posture and XY-clearance monitoring

### Key successful experiment

`test_integrated_camera_route.py`

Reported outcome:

- `INTEGRATED_CAMERA_ROUTE: PASS`
- Final goal error: 0.110 m
- Minimum estimated XY clearance: 0.552 m
- Minimum extra-box XY clearance: 0.564 m
- Maximum tilt: 3.62 degrees
- 9,785 recorded samples

The integrated route uses previously recorded camera data.
It is not yet autonomous online replanning.

### Current development

`test_early_live_replanning_v3.py`

Objective: detect an obstacle in the active simulation,
stop, capture depth, estimate occupancy and generate a
new route without using the obstacle's known center.

### Remaining Gate 3 acceptance stages

1. Camera-derived complete traversal — passed
2. Live depth observation and route generation — testing
3. Integrated online stop, observe, replan and traverse
4. Unknown obstacle configurations and repeatability
5. Integrated safety and acceptance audit

## Other simulators

- MuJoCo: active
- Genesis 1.4.5: CPU initialization passed
- ROS 2 Jazzy: installation blocked by apt dependencies
- Gazebo Harmonic: not yet installed

## Safety limitations

Simulation research only.

XY clearance approximations do not certify full-body
or leg clearance. The camera can have blind spots.
No hardware safety certification or authorization
for human-facing operation is implied.
