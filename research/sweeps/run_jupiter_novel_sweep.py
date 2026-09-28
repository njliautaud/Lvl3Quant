#!/usr/bin/env python3
import sys, json, time, subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_novel_predictions"
OUT_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_novel_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

SIM_CONFIGS = [
    (1800000, 1, 3, None, "hold30m_chase"),
    (1800000, 1, 3, 5, "hold30m_tp5_chase"),
    (1800000, 1, 3, 8, "hold30m_tp8_chase"),
    (1800000, 1, 3, 10, "hold30m_tp10_chase"),
]

def run_sim(mbo, pred, out, hold_ms, chase_t, chase_r, tp):
    cmd = [str(BINARY), "--mbo-file", str(mbo), "--predictions", str(pred),
           "--output", str(out), "--hold-ms", str(hold_ms), "--signal-threshold", "0",
           "--latency-ms", "0", "--quiet", "--chase-entry",
           "--chase-max-ticks", str(chase_t), "--chase-max-reprices", str(chase_r)]
    if tp is not None:
        cmd += ["--take-profit-ticks", str(tp)]
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if r.returncode == 0 and Path(out).exists():
            with open(out) as f:
                return json.load(f)
    except:
        pass
    return None

pred_files = sorted(PRED_DIR.glob("*.npz"))
print(f"Found {len(pred_files)} prediction files")

jobs = []
for pf in pred_files:
    stem = pf.stem
    date = stem[:10]
    nodash = date.replace("-", "")
    mbo = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn.zst"
    if not mbo.exists():
        mbo = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn"
    if not mbo.exists():
        continue
    for hold_ms, chase_t, chase_r, tp, sim_label in SIM_CONFIGS:
        out_label = f"{stem[11:]}_{sim_label}"
        out_file = OUT_DIR / f"{out_label}_{date}.json"
        if out_file.exists():
            continue
        jobs.append((str(mbo), str(pf), str(out_file), hold_ms, chase_t, chase_r, tp))

print(f"Jobs to run: {len(jobs)}")
completed = 0
t0 = time.time()

with ThreadPoolExecutor(max_workers=14) as executor:
    futures = {executor.submit(run_sim, *j): j for j in jobs}
    for f in as_completed(futures):
        completed += 1
        if completed % 100 == 0 or completed == len(jobs):
            el = time.time() - t0
            rate = completed / el if el > 0 else 0
            eta = (len(jobs) - completed) / rate / 60 if rate > 0 else 0
            print(f"  [{completed}/{len(jobs)}] {rate:.1f}/s, ETA {eta:.1f}min")

print(f"Done: {completed} jobs in {time.time()-t0:.0f}s")
