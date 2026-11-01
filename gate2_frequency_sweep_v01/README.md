# Gate 2 frequency sweep V0.1

Compares 0.64, 0.80, 0.96 Hz (0.8x, 1.0x, 1.2x baseline). **Only the gait rate changes.** Stride=0.035m, lift=0.018m, settle=2s, walk=8s, cooldown=2s are kept at the existing programs' defaults. Each run starts from its own reset MuJoCo model. Eight seconds contains different numbers of cycles at different rates; compare speed and contact proxies as well as total travel.

## Install

Extract `gate2_frequency_sweep_v01` alongside `gate2_walk_v01`, `gate2_contact_v01`, `gate2_gait_analysis_v02` and `dog_robot_v02` under the same `dog_robot` directory. Make sure to activate the Python environment where MuJoCo is installed.

```bash
cd ~/physical_ai_todai/dog_robot/dog_robot_v02
python ../gate2_frequency_sweep_v01/frequency_sweep.py --model dog_quadruped.xml
```

Outputs: `gate2_results/frequency_sweep_v01/summary.csv`, per-rate gait contact CSV and timeline PNG, per-rate `forward_trial.csv`, and optional `comparison.png`.

## Interpreting

This is not a formal gait PASS. `forward_m` positive means forward (-X) motion; `mean_contact_tangent_speed_mps` and `peak_contact_tangent_speed_mps` are slip proxies, not verified slip distance. Comparison of fixed wall-clock walk duration is useful for speed and stability; it is not a per-cycle controlled comparison. Check floor contact and joint limits before hardware decisions. This script builds on earlier scripts and assumes their code has not been substantially changed.
