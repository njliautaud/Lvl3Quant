#!/usr/bin/env python3
import subprocess, json, glob, os, numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

BINARY = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_norm_sweep_predictions"
OUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/cnn_wf_norm_sweep_results"
os.makedirs(OUT_DIR, exist_ok=True)

SIM_CFGS = []
for conv in [1.5, 2.0, 2.5, 3.0]:
    for hold in [600000, 900000, 1200000, 1800000]:
        hm = hold // 60000
        SIM_CFGS.append((conv, hold, None, None, f"conv{int(conv*10)}_hold{hm}m"))
for conv in [2.0, 2.5]:
    for tp in [5, 8, 10]:
        SIM_CFGS.append((conv, 1800000, tp, None, f"conv{int(conv*10)}_hold30m_tp{tp}"))
# TP/SL combos
for conv in [2.0, 2.5, 3.0]:
    for tp in [5, 8, 10, 15]:
        for sl in [15, 20, 25]:
            SIM_CFGS.append((conv, 3600000, tp, sl, f"conv{int(conv*10)}_tpsl_tp{tp}_sl{sl}"))

pred_files = sorted(glob.glob(os.path.join(PRED_DIR, "*.npz")))
print(f"Preds: {len(pred_files)}, Configs: {len(SIM_CFGS)}")

jobs = []
for pf in pred_files:
    base = Path(pf).stem
    parts = base.split("_")
    date = parts[0]
    mbo_list = glob.glob(os.path.join(MBO_DIR, f"*{date.replace(chr(45),'')}*.dbn.zst"))
    if not mbo_list: continue
    norm_vol = "_".join(parts[1:])
    for conv, hold, tp, trail, label in SIM_CFGS:
        out = os.path.join(OUT_DIR, f"{norm_vol}_{label}_{date}.json")
        if os.path.exists(out): continue
        jobs.append((mbo_list[0], pf, out, conv, hold, tp, trail))

print(f"Jobs: {len(jobs)}")

def run(job):
    mbo, pred, out, conv, hold, tp, trail = job
    cmd = [BINARY, "--mbo-file", mbo, "--predictions", pred, "--output", out,
           "--hold-ms", str(hold), "--signal-threshold", str(conv),
           "--latency-ms", "0", "--chase-entry", "--chase-max-ticks", "1",
           "--chase-max-reprices", "3", "--quiet"]
    if tp: cmd += ["--take-profit-ticks", str(tp)]
    if trail: cmd += ["--trailing-ticks", str(trail)]
    try:
        subprocess.run(cmd, capture_output=True, timeout=120)
    except: pass

done = 0
with ThreadPoolExecutor(max_workers=14) as ex:
    for _ in ex.map(run, jobs):
        done += 1
        if done % 500 == 0: print(f"  {done}/{len(jobs)}")

print(f"Done: {done}")

