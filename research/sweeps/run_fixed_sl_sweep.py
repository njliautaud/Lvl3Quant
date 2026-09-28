#!/usr/bin/env python3
"""Fixed Stop-Loss Sweep on Jupiter — 14 workers"""
import sys, json, time, subprocess, os
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
OUT_DIR = LVL3_ROOT / "data" / "processed" / "fixed_sl_sweep_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Sweep parameters
STRATEGIES = [
    "book_predstdExit_conv1.5_vol50",
    "book_predstdExit_conv2.5_vol70",
]
FIXED_SL = [5, 8, 10, 12, 15]
TP_VALUES = [5, 10, 15, None]
WB_VALUES = [20, 30]       # max_wait_bars
LAT_VALUES = [0, 50]       # latency_ms
WORKERS = 14

def run_sim(mbo, pred, out, sl, tp, wb, lat):
    """Run one fill_sim job with fixed stop loss."""
    cmd = [
        str(BINARY),
        "--mbo-file", str(mbo),
        "--predictions", str(pred),
        "--output", str(out),
        "--hold-ms", "3600000",          # 60 min safety net
        "--signal-threshold", "0.1",
        "--max-wait-bars", str(wb),
        "--latency-ms", str(lat),
        "--chase-entry",
        "--chase-max-ticks", "1",
        "--chase-max-reprices", "3",
        "--signal-flip-exit",
        "--stop-loss-ticks", str(sl),    # FIXED stop loss (new flag!)
        "--quiet",
    ]
    if tp is not None:
        cmd += ["--take-profit-ticks", str(tp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if r.returncode == 0 and Path(out).exists():
            with open(out) as f:
                data = json.load(f)
                return data
    except Exception as e:
        pass
    return None

# Build job list
jobs = []
for strat in STRATEGIES:
    pred_files = sorted(PRED_DIR.glob(f"*_{strat}.npz"))
    for pf in pred_files:
        date = pf.stem[:10]
        nodash = date.replace("-", "")
        mbo = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn.zst"
        if not mbo.exists():
            mbo = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn"
        if not mbo.exists():
            continue
        for sl in FIXED_SL:
            for tp in TP_VALUES:
                for wb in WB_VALUES:
                    for lat in LAT_VALUES:
                        tp_str = f"tp{tp}" if tp is not None else "tpN"
                        label = f"{strat}_fsl{sl}_{tp_str}_wb{wb}_lat{lat}_{date}"
                        out_file = OUT_DIR / f"{label}.json"
                        if out_file.exists():
                            continue
                        jobs.append((str(mbo), str(pf), str(out_file), sl, tp, wb, lat))

print(f"Total jobs: {len(jobs)}")
print(f"Workers: {WORKERS}")
print(f"Strategies: {STRATEGIES}")
print(f"Fixed SL: {FIXED_SL}")
print(f"TP: {TP_VALUES}")
print(f"WB: {WB_VALUES}")
print(f"Latency: {LAT_VALUES}")
sys.stdout.flush()

if not jobs:
    print("No jobs to run! All results already exist.")
    sys.exit(0)

done = 0
failed = 0
t0 = time.time()
results_summary = []

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {}
    for mbo, pred, out, sl, tp, wb, lat in jobs:
        f = executor.submit(run_sim, mbo, pred, out, sl, tp, wb, lat)
        futures[f] = (out, sl, tp, wb, lat)

    for future in as_completed(futures):
        out_path, sl, tp, wb, lat = futures[future]
        result = future.result()
        done += 1
        if result is None:
            failed += 1
        else:
            results_summary.append({
                "file": os.path.basename(out_path),
                "pnl": result.get("total_pnl_dollars", 0),
                "trades": result.get("total_trades", 0),
                "sharpe": result.get("sharpe_per_trade", 0),
                "wr": result.get("win_rate", 0),
                "fr": result.get("fill_rate", 0),
            })

        if done % 100 == 0 or done == len(jobs):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(jobs) - done) / rate / 60 if rate > 0 else 0
            print(f"[{done}/{len(jobs)}] {rate:.1f}/s, ETA {eta:.0f}min, failed={failed}")
            sys.stdout.flush()

elapsed = time.time() - t0
print(f"\n=== SWEEP COMPLETE ===")
print(f"Total: {done} jobs in {elapsed:.0f}s ({elapsed/60:.1f}min)")
print(f"Failed: {failed}")
print(f"Rate: {done/elapsed:.1f} jobs/s")

# Save summary
summary_file = OUT_DIR / "sweep_summary.json"
with open(summary_file, "w") as f:
    json.dump({
        "total_jobs": done,
        "failed": failed,
        "elapsed_s": elapsed,
        "results": results_summary,
    }, f, indent=2)
print(f"Summary saved to: {summary_file}")
