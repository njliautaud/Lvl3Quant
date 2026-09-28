#!/usr/bin/env python3
"""
Standalone launcher for train_fifo_rl_sac.py that forces all output
to a log file before any other imports, bypassing systemd/nohup stdout issues.
"""
import os, sys

LOG = "/home/nick/Lvl3Quant/output/fifo_sac_rl_neptune_v2.log"

# Open log file and redirect stdout/stderr to it FIRST (line-buffered)
logf = open(LOG, "a", buffering=1)
sys.stdout = logf
sys.stderr = logf

import os
print(f"[launcher] PID={os.getpid()}", flush=True)

import runpy, sys
sys.argv = [
    "train_fifo_rl_sac.py",
    "--epochs", "80",
    "--train-days", "40",
    "--eval-days", "5",
    "--lr", "1e-4",
    "--hidden-dim", "256",
    "--buffer-size", "1000000",
    "--batch-size", "512",
    "--warmup-steps", "2000",
    "--updates-per-step", "2",
    "--max-steps-per-episode", "100000",
    "--data-dir", "/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3",
    "--pred-dir", "/home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar",
    "--patchtst-dir", "/home/nick/Lvl3Quant/output/patchtst_smart_v3_mar",
    "--output-dir", "/home/nick/Lvl3Quant/output/fifo_sac_rl_neptune_v2",
]
os.environ["PYTHONUNBUFFERED"] = "1"
os.environ["MLFLOW_TRACKING_URI"] = "http://localhost:5000"

sys.path.insert(0, "/home/nick/Lvl3Quant/alpha_discovery/execution")
runpy.run_path(
    "/home/nick/Lvl3Quant/alpha_discovery/execution/train_fifo_rl_sac.py",
    run_name="__main__",
)
