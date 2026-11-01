# Gate 2 integrated walking and turning experiment

Sequence: stand 2 s → forward 1.0 m → pause 1 s → left 90° → pause 1 s → forward 0.5 m → hold 3 s.

Uses the same diagonal-phase Hybrid swing trajectory, 0.5 s stride ramp, 1.0 s lift ramp and differential-turn sign as the prior experiments. Distances use the heading **at each forward-stage entry**, not world X alone. Stopping is currently an abrupt position-target transition and is not certified safe for hardware.

From `dog_robot_v02` run:

```bash
python ../gate2_integrated_v01/integrated_trial.py --model dog_quadruped.xml --out gate2_results/integrated/integrated.csv
```

For visual inspection, add `--viewer`. Do not close the Viewer before completion if you need a full CSV. Numeric tests should be run without `--viewer` first.

**Notes:** This script has only been syntax-checked, not simulated with your MJCF. Feedback angle is measured as a signed unwrapped yaw relative to turn start. A yaw-target threshold of 88.5° is the previous experimental heuristic. The turn command's signs follow your previously measured behavior, not an abstract vehicle model. Timeouts are reported, not passed as successful achievements. Foot contact forces/slip and actuator rate limits are not certified. CSV orientation readings are measured immediately before each simulation step, so they correspond approximately to the beginning of each logged step.
