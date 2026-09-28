#!/usr/bin/env python3
"""Jupiter expanded sweep - signal-flip+TP, fine TP grid, prime hours.
Runs locally on Jupiter. Output: cnn_wf_expanded_sweep_results/
"""
import subprocess, glob, os, time, sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BINARY = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_norm_sweep_predictions"
OUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_expanded_sweep_results"
os.makedirs(OUT_DIR, exist_ok=True)

TARGET_NORMS = ['ema_zscore_span5000', 'smooth10_expanding']
TARGET_VOLS = [50, 70]

# Build date -> mbo mapping
mbo_map = {}
for f in glob.glob(os.path.join(MBO_DIR, "*.mbo.dbn.zst")):
    bn = os.path.basename(f)
    parts = bn.replace('.mbo.dbn.zst', '').split('-')
    if len(parts) >= 3:
        d = parts[-1]
        if len(d) == 8:
            date = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
            mbo_map[date] = f

# Build pred file index: (date, norm, vol) -> path
pred_index = {}
for f in glob.glob(os.path.join(PRED_DIR, "*.npz")):
    bn = os.path.basename(f).replace('.npz', '')
    parts = bn.split('_', 1)
    if len(parts) < 2:
        continue
    date = parts[0]
    rest = parts[1]
    vol_idx = rest.rfind('_vol')
    if vol_idx < 0:
        continue
    norm = rest[:vol_idx]
    vol_str = rest[vol_idx+4:]
    try:
        vol = int(vol_str)
    except:
        continue
    pred_index[(date, norm, vol)] = f

print(f"MBO dates: {len(mbo_map)}, Pred entries: {len(pred_index)}")

avail_dates = sorted(set(d for (d, n, v) in pred_index.keys()
                         if d in mbo_map and n in TARGET_NORMS and v in TARGET_VOLS))
print(f"Available dates with target preds + MBO: {len(avail_dates)}")

jobs = []

for date in avail_dates:
    mbo = mbo_map[date]
    for norm in TARGET_NORMS:
        for vol in TARGET_VOLS:
            key = (date, norm, vol)
            if key not in pred_index:
                continue
            pred = pred_index[key]
            tag = f"{norm}_vol{vol}"

            # ===== SWEEP 1: Signal-flip-exit + TP combos =====
            for conv in [2.0, 2.5]:
                for tp in [5, 10]:
                    label = f"{tag}_flipTP{tp}_conv{int(conv*10)}_60m_{date}"
                    out = os.path.join(OUT_DIR, f"{label}.json")
                    if os.path.exists(out):
                        continue
                    jobs.append({
                        'mbo': mbo, 'pred': pred, 'out': out,
                        'conv': conv, 'hold_ms': 3600000,
                        'tp': tp, 'trail': None,
                        'flip': True, 'prime': False,
                        'chase_t': 0, 'chase_r': 0,
                    })

            # ===== SWEEP 2: Fine TP grid (3,4,5,6,7,8) =====
            for conv in [2.0, 2.5]:
                for tp in [3, 4, 5, 6, 7, 8]:
                    # Passive + TP + 30min
                    label = f"{tag}_tp{tp}_conv{int(conv*10)}_30m_passive_{date}"
                    out = os.path.join(OUT_DIR, f"{label}.json")
                    if not os.path.exists(out):
                        jobs.append({
                            'mbo': mbo, 'pred': pred, 'out': out,
                            'conv': conv, 'hold_ms': 1800000,
                            'tp': tp, 'trail': None,
                            'flip': False, 'prime': False,
                            'chase_t': 0, 'chase_r': 0,
                        })

                    # Chase 1t/3r + TP + 30min
                    label2 = f"{tag}_tp{tp}_conv{int(conv*10)}_30m_chase1t3r_{date}"
                    out2 = os.path.join(OUT_DIR, f"{label2}.json")
                    if not os.path.exists(out2):
                        jobs.append({
                            'mbo': mbo, 'pred': pred, 'out': out2,
                            'conv': conv, 'hold_ms': 1800000,
                            'tp': tp, 'trail': None,
                            'flip': False, 'prime': False,
                            'chase_t': 1, 'chase_r': 3,
                        })

            # ===== SWEEP 3: Prime hours flag =====
            for conv in [2.0, 2.5]:
                # Prime + 30min passive
                label = f"{tag}_prime_conv{int(conv*10)}_30m_passive_{date}"
                out = os.path.join(OUT_DIR, f"{label}.json")
                if not os.path.exists(out):
                    jobs.append({
                        'mbo': mbo, 'pred': pred, 'out': out,
                        'conv': conv, 'hold_ms': 1800000,
                        'tp': None, 'trail': None,
                        'flip': False, 'prime': True,
                        'chase_t': 0, 'chase_r': 0,
                    })

                # Prime + 30min chase 1t/3r
                label2 = f"{tag}_prime_conv{int(conv*10)}_30m_chase1t3r_{date}"
                out2 = os.path.join(OUT_DIR, f"{label2}.json")
                if not os.path.exists(out2):
                    jobs.append({
                        'mbo': mbo, 'pred': pred, 'out': out2,
                        'conv': conv, 'hold_ms': 1800000,
                        'tp': None, 'trail': None,
                        'flip': False, 'prime': True,
                        'chase_t': 1, 'chase_r': 3,
                    })

                # Prime + TP5 + 30min passive
                label3 = f"{tag}_prime_tp5_conv{int(conv*10)}_30m_passive_{date}"
                out3 = os.path.join(OUT_DIR, f"{label3}.json")
                if not os.path.exists(out3):
                    jobs.append({
                        'mbo': mbo, 'pred': pred, 'out': out3,
                        'conv': conv, 'hold_ms': 1800000,
                        'tp': 5, 'trail': None,
                        'flip': False, 'prime': True,
                        'chase_t': 0, 'chase_r': 0,
                    })

print(f"\nTotal jobs: {len(jobs)}")
if len(jobs) == 0:
    print("All done!")
    sys.exit(0)

flip_jobs = sum(1 for j in jobs if j['flip'])
tp_grid = sum(1 for j in jobs if not j['flip'] and not j['prime'] and j.get('tp') is not None)
prime_jobs = sum(1 for j in jobs if j['prime'])
print(f"  Signal-flip+TP: {flip_jobs}")
print(f"  Fine TP grid:   {tp_grid}")
print(f"  Prime hours:    {prime_jobs}")
est = len(jobs) * 2.5 / 14 / 3600
print(f"  Est time: {est:.1f}h at 14 workers")

def run_job(job):
    cmd = [BINARY,
           "--mbo-file", job['mbo'],
           "--predictions", job['pred'],
           "--output", job['out'],
           "--hold-ms", str(job['hold_ms']),
           "--signal-threshold", str(job['conv']),
           "--latency-ms", "0",
           "--quiet"]
    if job['flip']:
        cmd.append("--signal-flip-exit")
    if job['prime']:
        cmd.append("--prime-hours")
    if job['tp']:
        cmd += ["--take-profit-ticks", str(job['tp'])]
    if job['trail']:
        cmd += ["--trailing-ticks", str(job['trail'])]
    if job['chase_t'] > 0:
        cmd += ["--chase-entry",
                "--chase-max-ticks", str(job['chase_t']),
                "--chase-max-reprices", str(job['chase_r'])]
    try:
        subprocess.run(cmd, capture_output=True, timeout=180)
    except:
        pass

done = [0]
start = time.time()

def wrapped(job):
    run_job(job)
    done[0] += 1
    if done[0] % 200 == 0:
        elapsed = time.time() - start
        rate = done[0] / elapsed * 60 if elapsed > 0 else 0
        remain = (len(jobs) - done[0]) / (done[0] / elapsed) if done[0] > 0 else 0
        print(f"  Progress: {done[0]}/{len(jobs)} ({done[0]/len(jobs)*100:.1f}%) | {rate:.0f}/min | ETA: {remain/3600:.1f}h", flush=True)

with ThreadPoolExecutor(max_workers=14) as ex:
    list(ex.map(wrapped, jobs))

elapsed = time.time() - start
print(f"\nDone: {done[0]} jobs in {elapsed/3600:.1f}h")
print(f"Results in: {OUT_DIR}")
print(f"Total files: {len(glob.glob(os.path.join(OUT_DIR, '*.json')))}")
