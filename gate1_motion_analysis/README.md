# Gate 1 motion analysis: push versus baseline

No changes to the quadruped XML or the previous evaluation scripts are necessary.

## Running in Linux

Activate your existing Python virtual environment containing MuJoCo, and enter the folder containing `dog_quadruped.xml`.

```bash
python ../gate1_motion_analysis/analyze_push.py --model dog_quadruped.xml --duration 8 --force 3
```

If the extracted folder is elsewhere, use its absolute path instead.

Generated CSVs in `gate1_motion_results/`:
- `baseline.csv`: no external force
- `push.csv`: lateral 3 N for 2.0-2.2 seconds
- `difference.csv`: push - baseline at each simulation time step

Y displacement is in meters, lateral speed is in m/s, and roll angle in degrees.
The settling criterion measures recovery of *roll and lateral speed differences* relative to the unpushed trajectory (within 0.05 degrees and 0.01 m/s for a continuous 0.2s), not return to original X/Y position. The tolerances are preliminary, not safety standards.

The model can stay upright yet slide laterally. No claim of a formal Gate 1 pass or safe hardware behavior is made.
