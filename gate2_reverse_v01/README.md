# Gate 2 reverse walking V0.1

Single-file MuJoCo experiment copied from the validated stop-trial control architecture, with negative stride for reverse walking. This is an **open-loop** exploratory controller, not safe for real hardware.

From `~/physical_ai_todai/dog_robot/dog_robot_v02`:

```bash
python ../gate2_reverse_v01/reverse_trial.py --model dog_quadruped.xml --direction backward --walk 12 --mode abrupt --viewer --out gate2_results/reverse/backward.csv
```

Press close in MuJoCo to exit early; closing early may mean the distance trial is incomplete. Simulator's +X is the robot's backward direction. A target of +0.5m is the Gate 2 reverse distance criterion. Viewer's playback is paced to approximately real time.

For headless run omit `--viewer`. For baseline try `--direction forward --walk 12` (not a full independent validation of existing forward implementation).

Check torso pitch/roll, sideways drift, actual joint force and contact velocity before judging successful gait. Default stop is abrupt and causes some visible motion; not hardware certified.
