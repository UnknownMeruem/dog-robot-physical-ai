# Gate 2-1 contact & slip diagnostic (V0.1)

This script replicates the existing `gate2_walk_v01/gate2_forward.py` IK and trajectory, while adding foot-floor active contact detection and a **foot-center velocity proxy** for slip. It does **not** prove true contact-patch slip, load transfer, or a stable gait.

Run from the directory containing `dog_quadruped.xml` after activating the MuJoCo Python environment:

```bash
python ../gate2_contact_v01/gate2_contact.py --model dog_quadruped.xml
```

Outputs under `gate2_results/`:
- `forward_contact_trial.csv`: body pose time series.
- `contact_slip_timeseries.csv`: per-step foot active-contact indicators and planar foot-center speed during contact.

The script checks for `floor` and four named `*_foot` geoms and filters out inactive contacts with `efc_address < 0`. Slip proxy is horizontal travel of the foot **geometry center** during active foot-floor contact, not exact contact-point tangential sliding. Real slip requires contact-point velocity and friction-force evaluation. Nonfoot-floor contact fraction alerts when knees/torso/etc. collide with the floor.

Do not interpret the reported contact rate as a PASS criterion. Check body stability and net translation alongside contact and motion metrics. Geometry was not modified.
