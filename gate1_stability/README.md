# Gate 1 — standing evaluation (V0.1)

Use with `dog_quadruped.xml` from the Dog Robot V0.2 project. It reports a standing-candidate evaluation, not a certified stable controller.

```
python -m pip install mujoco numpy
python evaluate_gate1.py --model /path/to/dog_quadruped.xml --duration 30 --out gate1_results.csv
```

Metrics: torso height, roll, pitch, contact count and peak actuator force. Evaluation after the first 2 seconds avoids penalizing initial transient, but initial fall still needs separate review. Contact detection assumes floor geom ID 0 in V0.2. Adapt contact logic if model geom order changes. `max_leg_actuator_force` reports MuJoCo actuator force; verify joint torque mapping if gears are not unity. First 8 actuator channels must be hip/knee controls. Early-stage thresholds are provisional.

Important: V0.2 open-loop controller can collapse. This is expected; a failed Gate 1 is diagnostic evidence, not a broken evaluator. Foot-contact percentage should not be construed as full static stability, and safe physical performance requires extensive further validation.
