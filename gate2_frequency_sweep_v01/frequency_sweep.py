"""Compare 0.8x, 1.0x, 1.2x of the original 0.8 Hz trot frequency.
Reuses the unchanged V0.1 forward-walk and V0.2 contact diagnostics.
This is an exploratory comparison, not a walking certification.
"""
import argparse
import csv
import math
import subprocess
import sys
from pathlib import Path


def read_csv(path):
    with open(path, newline='') as f:
        return list(csv.DictReader(f))


def maximum(rows, key, absolute=False):
    vals=[float(r[key]) for r in rows]
    return max(map(abs,vals)) if absolute else max(vals)


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--model', type=Path, default=Path('dog_quadruped.xml'))
    parser.add_argument('--base-rate', type=float, default=0.8)
    parser.add_argument('--walk', type=float, default=8.0)
    parser.add_argument('--outdir', type=Path, default=Path('gate2_results/frequency_sweep_v01'))
    args=parser.parse_args()
    if args.base_rate<=0 or args.walk<=0:
        parser.error('--base-rate and --walk must be positive')
    model=args.model.resolve()
    if not model.exists():
        parser.error(f'Model not found: {model}')
    root=Path(__file__).resolve().parent.parent
    gait=root/'gate2_gait_analysis_v02'/'analyze_gait.py'
    walk=root/'gate2_walk_v01'/'gate2_forward.py'
    for file in (gait,walk,root/'gate2_contact_v01'/'gate2_contact.py'):
        if not file.is_file():
            parser.error(f'Missing existing script: {file}. Place this folder alongside the earlier Gate 2 folders.')
    args.outdir.mkdir(parents=True,exist_ok=True)
    results=[]
    feet=['front_left','front_right','hind_left','hind_right']
    for factor in (0.8,1.0,1.2):
        rate=args.base_rate*factor
        label=f'{factor:.1f}x'
        run_dir=args.outdir/label.replace('.','p')
        run_dir.mkdir(exist_ok=True)
        print(f'\n=== {label}: {rate:.3f} Hz ===',flush=True)
        command=[sys.executable,str(gait),'--model',str(model),'--rate',str(rate),'--walk',str(args.walk),'--outdir',str(run_dir/'contact')]
        subprocess.run(command,check=True)
        command=[sys.executable,str(walk),'--model',str(model),'--rate',str(rate),'--walk',str(args.walk),'--out',str(run_dir/'forward_trial.csv')]
        subprocess.run(command,check=True)
        contacts=read_csv(run_dir/'contact'/'contact_points.csv')
        forward=read_csv(run_dir/'forward_trial.csv')
        walking=[r for r in forward if r['state']=='walk']
        if len(contacts)<2 or len(walking)<2:
            raise RuntimeError('Insufficient walk samples')
        dx=float(contacts[-1]['torso_x_m'])-float(contacts[0]['torso_x_m'])
        # Walk sample outputs are sampled differently in these two scripts.
        min_height=min(float(r['torso_z_m']) for r in walking)
        max_roll=maximum(walking,'roll_deg',absolute=True)
        max_pitch=maximum(walking,'pitch_deg',absolute=True)
        per_foot={}
        all_speeds=[]
        for foot in feet:
            selected=[float(r[f'{foot}_tangent_speed_mps']) for r in contacts if int(r[f'{foot}_contact'])]
            per_foot[f'{foot}_contact_pct']=100*len(selected)/len(contacts)
            per_foot[f'{foot}_mean_tangent_mps']=sum(selected)/len(selected) if selected else float('nan')
            per_foot[f'{foot}_peak_tangent_mps']=max(selected) if selected else float('nan')
            all_speeds.extend(selected)
        rec={'factor':factor,'rate_hz':rate,'walk_duration_s':args.walk,
             'forward_m':-dx,'forward_mps':-dx/args.walk,
             'min_height_m':min_height,'max_abs_roll_deg':max_roll,
             'max_abs_pitch_deg':max_pitch,
             'mean_contact_tangent_speed_mps':sum(all_speeds)/len(all_speeds) if all_speeds else float('nan'),
             'peak_contact_tangent_speed_mps':max(all_speeds) if all_speeds else float('nan'),
             **per_foot}
        results.append(rec)
        print(f'Comparison: forward={-dx:.3f} m, height_min={min_height:.3f} m, '
              f'max_roll={max_roll:.2f}°, peak_contact_proxy={rec["peak_contact_tangent_speed_mps"]:.4f} m/s')
    csv_path=args.outdir/'summary.csv'
    with csv_path.open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(results[0].keys()))
        writer.writeheader()
        writer.writerows(results)
    print('\n=== FREQUENCY SWEEP SUMMARY ===')
    print(f'{"Rate":>8} {"Forward":>10} {"Min Z":>10} {"Max roll":>10} {"Mean proxy":>12} {"Peak proxy":>12}')
    for r in results:
        print(f'{r["rate_hz"]:>7.2f}Hz {r["forward_m"]:>9.3f}m {r["min_height_m"]:>9.3f}m '
              f'{r["max_abs_roll_deg"]:>9.2f}° {r["mean_contact_tangent_speed_mps"]:>11.4f} '
              f'{r["peak_contact_tangent_speed_mps"]:>11.4f}')
    print('Saved:',csv_path.resolve())
    try:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        names=[f'{r["rate_hz"]:.2f} Hz' for r in results]
        fig,axs=plt.subplots(3,1,figsize=(8,8),layout='constrained')
        axs[0].bar(names,[r['forward_m'] for r in results]);axs[0].set_ylabel('Forward displacement (m)')
        axs[1].bar(names,[r['max_abs_roll_deg'] for r in results]);axs[1].set_ylabel('Max abs roll (deg)')
        axs[2].bar(names,[r['peak_contact_tangent_speed_mps'] for r in results]);axs[2].set_ylabel('Peak tangent speed proxy (m/s)')
        fig.suptitle('Gate 2 — frequency comparison (same 8 s walk window)')
        plot=args.outdir/'comparison.png'
        fig.savefig(plot,dpi=150)
        plt.close(fig)
        print('Plot:',plot.resolve())
    except ImportError:
        print('Optional graph skipped: pip install matplotlib')
    print('Exploratory metrics only; no formal walking PASS. Contact-point speed is a slip proxy.')

if __name__=='__main__':
    main()
