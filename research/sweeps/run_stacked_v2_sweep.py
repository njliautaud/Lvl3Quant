#!/usr/bin/env python3
"""Stacked Exit Sweep V2 — Remote worker (NO --signal-flip-exit)."""
import sys, json, time, subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

WORKERS = 14
LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
OUT_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_v2_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TP_VALUES = [None, 5, 8, 10, 15, 20]
SL_VALUES = [None, 10, 15, 20, 25]

def run_sim(mbo, pred, out, tp, sl):
    cmd = [str(BINARY), "--mbo-file", str(mbo), "--predictions", str(pred),
           "--output", str(out), "--hold-ms", "3600000", "--signal-threshold", "0.1",
           "--latency-ms", "0", "--quiet", "--chase-entry",
           "--chase-max-ticks", "1", "--chase-max-reprices", "3"]
    # NO --signal-flip-exit — MA exit is encoded in prediction signal
    if tp is not None:
        cmd += ["--take-profit-ticks", str(tp)]
    if sl is not None:
        cmd += ["--trailing-ticks", str(sl)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and Path(out).exists():
            return True
    except:
        pass
    return False

pred_files = sorted(PRED_DIR.glob("*.npz"))
print(f"Found {len(pred_files)} prediction files", flush=True)

if not BINARY.exists():
    print(f"ERROR: Binary not found: {BINARY}", flush=True)
    sys.exit(1)

jobs = []
skipped = 0
for pf in pred_files:
    stem = pf.stem
    date = stem[:10]
    combo_label = stem[11:]
    nodash = date.replace("-", "")
    mbo = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn.zst"
    if not mbo.exists():
        mbo = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn"
    if not mbo.exists():
        continue
    for tp in TP_VALUES:
        for sl in SL_VALUES:
            tp_str = f"tp{tp}" if tp is not None else "tpN"
            sl_str = f"sl{sl}" if sl is not None else "slN"
            full_label = f"{combo_label}_{tp_str}_{sl_str}"
            out_file = OUT_DIR / f"{full_label}_{date}.json"
            if out_file.exists():
                skipped += 1
                continue
            jobs.append((str(mbo), str(pf), str(out_file), tp, sl))

print(f"Jobs to run: {len(jobs)} (skipped {skipped} existing)", flush=True)
if not jobs:
    print("No jobs to run — all done!", flush=True)
    sys.exit(0)

done = 0
failed = 0
t0 = time.time()

with ThreadPoolExecutor(max_workers=WORKERS) as executor:
    futures = {}
    for mbo, pred, out, tp, sl in jobs:
        f = executor.submit(run_sim, mbo, pred, out, tp, sl)
        futures[f] = out

    for future in as_completed(futures):
        done += 1
        try:
            if not future.result():
                failed += 1
        except:
            failed += 1
        if done % 500 == 0 or done == len(jobs):
            elapsed = time.time() - t0
            rate = done / elapsed if elapsed > 0 else 0
            eta = (len(jobs) - done) / rate / 60 if rate > 0 else 0
            print(f"  [{done}/{len(jobs)}] {rate:.1f}/s, ETA {eta:.1f}min, {failed} failed", flush=True)

elapsed = time.time() - t0
print(f"DONE: {done} jobs in {elapsed:.0f}s ({failed} failed)", flush=True)
results_count = len(list(OUT_DIR.glob("*.json")))
print(f"Total result files: {results_count}", flush=True)
