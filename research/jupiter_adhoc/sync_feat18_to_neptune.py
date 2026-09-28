#!/usr/bin/env python3
"""Push feat18 NPZ files from Jupiter to Neptune via scp (key auth)."""
import subprocess, os, glob, sys, time

FEAT18_DIR = "/home/jupiter/Lvl3Quant/data/processed/mbo_events_feat18"
NEPTUNE_IP = "neptune-win"
NEPTUNE_USER = "Footb"
NEPTUNE_DEST = "C:/Users/Footb/Documents/Github/Lvl3Quant/data/processed/mbo_events_feat18/"

files = sorted(glob.glob(f"{FEAT18_DIR}/*.npz"))
print(f"Found {len(files)} feat18 files to sync", flush=True)

ok = 0
fail = 0
for i, f in enumerate(files):
    fname = os.path.basename(f)
    dest = f"{NEPTUNE_USER}@{NEPTUNE_IP}:{NEPTUNE_DEST}{fname}"
    cmd = ["scp", "-o", "StrictHostKeyChecking=no", "-o", "ConnectTimeout=30", f, dest]
    result = subprocess.run(cmd, capture_output=True, timeout=120)
    if result.returncode == 0:
        ok += 1
    else:
        fail += 1
        print(f"  FAIL [{i+1}/{len(files)}] {fname}: {result.stderr.decode()[:100]}", flush=True)
    if (i+1) % 20 == 0:
        print(f"  Progress: {i+1}/{len(files)} done ({ok} ok, {fail} fail)", flush=True)

print(f"\nDone: {ok} synced, {fail} failed", flush=True)
