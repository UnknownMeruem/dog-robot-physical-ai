# Gate 1 Extended tests

Place `test_gate1_extended.py` beside `dog_quadruped.xml` or pass `--model`.

```bash
python test_gate1_extended.py --model dog_quadruped.xml --duration 8
```

Results in `gate1_extended_results/`.

* Baseline: stand without perturbation.
* push_1N / push_3N: apply horizontal force to torso for 0.2s starting 2s after reset.
* mass_plus_300g / mass_plus_600g: approximate added equipment uniformly at the torso body, scaling inertia; not a faithful representation of equipment placed on top or off-center.
* motors_off: leg position actuators disabled by zeroing gains and biases. Neck/tail/chest actuators remain on.

Caution: This is exploratory. `candidate_pass` is a coarse screening rule, not a full Gate 1 pass/fail. Measure recovery after 2.2s and assess persistent tilt and height; a separate evaluation of fall, contact, and mass distribution is needed. The original model is preliminary and may have geometry or parameter limitations.
