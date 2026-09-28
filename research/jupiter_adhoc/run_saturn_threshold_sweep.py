#!/usr/bin/env python3
"""Remote threshold optimization sweep on Saturn — 40 workers."""
import sys, json, time, subprocess
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed

LVL3_ROOT = Path("/home/saturn/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_threshold_opt_predictions"
OUT_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_threshold_opt_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TP_VARIANTS = [
    (None,  "hold"),
    (3,     "tp3"),
    (5,     "tp5"),
    (8,     "tp8"),
    (10,    "tp10"),
    (15,    "tp15"),
    (20,    "tp20"),
]

def run_sim(mbo, pred, out, tp):
    cmd = [str(BINARY), "--mbo-file", str(mbo), "--predictions", str(pred),
           "--output", str(out), "--hold-ms", "3600000", "--signal-threshold", "0.1",
           "--latency-ms", "0", "--quiet", "--chase-entry",
           "--chase-max-ticks", "1", "--chase-max-reprices", "3"]
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
    combo_label = stem[11:]
    nodash = date.replace("-", "")
    mbo = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn.zst"
    if not mbo.exists():
        mbo = MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn"
    if not mbo.exists():
        continue
    for tp, sim_suffix in TP_VARIANTS:
        full_label = f"{combo_label}_{sim_suffix}"
        out_file = OUT_DIR / f"{full_label}_{date}.json"
        if out_file.exists():
            continue
        jobs.append((str(mbo), str(pf), str(out_file), tp))

print(f"Jobs to run: {len(jobs)}")
if not jobs:
    print("Nothing to do!")
    sys.exit(0)

completed = 0
t0 = time.time()

with ThreadPoolExecutor(max_workers=40) as executor:
    futures = {executor.submit(run_sim, *j): j for j in jobs}
    for f in as_completed(futures):
        completed += 1
        if completed % 100 == 0 or completed == len(jobs):
            el = time.time() - t0
            rate = completed / el if el > 0 else 0
            eta = (len(jobs) - completed) / rate / 60 if rate > 0 else 0
            print(f"  [{completed}/{len(jobs)}] {rate:.1f}/s, ETA {eta:.1f}min")

print(f"Done: {completed} jobs in {time.time()-t0:.0f}s")
