#!/usr/bin/env python3
"""Adaptive sweep: TP/SL combos, short holds, fine-grained conviction grid.
Deployed 2026-03-14. Uses ThreadPoolExecutor(14 workers).
"""
import subprocess, os, json, glob, time, datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_norm_sweep_predictions"
OUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_adaptive_sweep_results"
MAX_WORKERS = 14

os.makedirs(OUT_DIR, exist_ok=True)

# Build date mapping: YYYY-MM-DD -> MBO file path
mbo_files = sorted(glob.glob(os.path.join(MBO_DIR, "glbx-mdp3-*.mbo.dbn.zst")))
date_to_mbo = {}
for f in mbo_files:
    base = os.path.basename(f)
    d = base.replace("glbx-mdp3-", "").replace(".mbo.dbn.zst", "")
    iso = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
    date_to_mbo[iso] = f

# Find prediction dates for each norm
def get_pred_dates(norm_pattern):
    preds = sorted(glob.glob(os.path.join(PRED_DIR, f"*_{norm_pattern}.npz")))
    result = []
    for p in preds:
        base = os.path.basename(p)
        date = base[:10]  # YYYY-MM-DD
        if date in date_to_mbo:
            result.append((date, p, date_to_mbo[date]))
    return result

# ── Config Generation ──────────────────────────────────────────

jobs = []

# === SWEEP 1: TP + SL combos on top normalizations ===
# TP 5,8,10 ticks + SL 15,20,25 ticks at conv 2.0,2.5,3.0
# Hold max 60min (3600000 ms), let TP/SL decide exit
for norm in ["ema_zscore_span5000_vol70", "ema_zscore_span5000_vol50",
             "smooth10_expanding_vol70", "smooth10_expanding_vol50"]:
    for tp in [5, 8, 10]:
        for sl in [15, 20, 25]:
            for conv in [2.0, 2.5, 3.0]:
                tag = f"tpsl_{norm}_tp{tp}_sl{sl}_c{conv}"
                for date, pred, mbo in get_pred_dates(norm):
                    out_file = os.path.join(OUT_DIR, f"{tag}_{date}.json")
                    if os.path.exists(out_file):
                        continue
                    cmd = [
                        FILL_SIM,
                        "--mbo-file", mbo,
                        "--predictions", pred,
                        "--output", out_file,
                        "--signal-threshold", str(conv),
                        "--hold-ms", "3600000",
                        "--take-profit-ticks", str(tp),
                        "--trailing-ticks", str(sl),
                        "--chase-entry",
                        "--chase-max-ticks", "1",
                        "--chase-max-reprices", "3",
                        "--quiet",
                    ]
                    jobs.append((tag, date, cmd, out_file))

# === SWEEP 2: Very short holds, high conviction, pure prediction ===
# Hold 3min (180000ms), 5min (300000ms) at conv 3.0, 3.5
# No TP, no SL — pure time-based exit
for norm in ["ema_zscore_span5000_vol70", "ema_zscore_span5000_vol50",
             "smooth10_expanding_vol70", "smooth10_expanding_vol50"]:
    for hold_min, hold_ms in [(3, 180000), (5, 300000)]:
        for conv in [3.0, 3.5]:
            tag = f"shorthold_{norm}_h{hold_min}m_c{conv}"
            for date, pred, mbo in get_pred_dates(norm):
                out_file = os.path.join(OUT_DIR, f"{tag}_{date}.json")
                if os.path.exists(out_file):
                    continue
                cmd = [
                    FILL_SIM,
                    "--mbo-file", mbo,
                    "--predictions", pred,
                    "--output", out_file,
                    "--signal-threshold", str(conv),
                    "--hold-ms", str(hold_ms),
                    "--chase-entry",
                    "--chase-max-ticks", "1",
                    "--chase-max-reprices", "3",
                    "--quiet",
                ]
                jobs.append((tag, date, cmd, out_file))

# === SWEEP 3: Fine-grained conviction grid on ema_zscore_span5000_vol70 ===
# Conv 1.0 through 3.0 in 0.25 steps, hold 30min (1800000ms)
norm = "ema_zscore_span5000_vol70"
for conv_x100 in range(100, 325, 25):  # 1.00, 1.25, ..., 3.00
    conv = conv_x100 / 100.0
    tag = f"finegrid_{norm}_c{conv}_h30m"
    for date, pred, mbo in get_pred_dates(norm):
        out_file = os.path.join(OUT_DIR, f"{tag}_{date}.json")
        if os.path.exists(out_file):
            continue
        cmd = [
            FILL_SIM,
            "--mbo-file", mbo,
            "--predictions", pred,
            "--output", out_file,
            "--signal-threshold", str(conv),
            "--hold-ms", "1800000",
            "--chase-entry",
            "--chase-max-ticks", "1",
            "--chase-max-reprices", "3",
            "--quiet",
        ]
        jobs.append((tag, date, cmd, out_file))

print(f"Total jobs to run: {len(jobs)}")
print(f"Sweep 1 (TP/SL): 4 norms x 3 TP x 3 SL x 3 conv x days")
print(f"Sweep 2 (short hold): 4 norms x 2 holds x 2 convs x days")
print(f"Sweep 3 (fine grid): 9 convs x days on ema_zscore_span5000_vol70")
print(f"MBO dates available: {len(date_to_mbo)}")

if len(jobs) == 0:
    print("No jobs to run (all results already exist). Exiting.")
    exit(0)

# ── Execution ──────────────────────────────────────────────────

completed = 0
failed = 0
start_time = time.time()

def run_job(args):
    tag, date, cmd, out_file = args
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=600)
        if result.returncode == 0 and os.path.exists(out_file):
            return ("ok", tag, date)
        else:
            err = result.stderr.decode(errors="replace")[:200]
            return ("fail", tag, date, err)
    except Exception as e:
        return ("fail", tag, date, str(e)[:200])

print(f"\n[{datetime.datetime.now().strftime('%H:%M:%S')}] Starting sweep with {MAX_WORKERS} workers...")

with ThreadPoolExecutor(max_workers=MAX_WORKERS) as pool:
    futures = {pool.submit(run_job, j): j for j in jobs}
    for fut in as_completed(futures):
        result = fut.result()
        if result[0] == "ok":
            completed += 1
        else:
            failed += 1
            if failed <= 10:
                print(f"  FAIL: {result[1]} {result[2]}: {result[3] if len(result) > 3 else 'unknown'}")
        total = completed + failed
        if total % 50 == 0 or total == len(jobs):
            elapsed = time.time() - start_time
            rate = total / elapsed if elapsed > 0 else 0
            eta = (len(jobs) - total) / rate if rate > 0 else 0
            print(f"[{datetime.datetime.now().strftime('%H:%M:%S')}] {total}/{len(jobs)} done "
                  f"({completed} ok, {failed} fail) | {rate:.1f} jobs/s | ETA: {eta/60:.0f}min")

elapsed = time.time() - start_time
print(f"\n=== SWEEP COMPLETE ===")
print(f"Total: {completed + failed}/{len(jobs)} | OK: {completed} | Failed: {failed}")
print(f"Time: {elapsed/3600:.1f}h | Results in: {OUT_DIR}")
