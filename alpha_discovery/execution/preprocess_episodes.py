#!/usr/bin/env python3
"""
Preprocess MBO .npz files into fast-load .npy format for RL training.
HC #153: Save processed files so we don't regenerate each run.

Converts compressed .npz (slow random access) to directory of .npy files
(memory-mappable, instant load). Each date gets a directory with:
  - events.npy (N, 25) float32
  - event_type_raw.npy (N,) int8  
  - timestamps.npy (N,) int64

Usage: python preprocess_episodes.py [--src-dir DIR] [--dst-dir DIR]
"""
import argparse
import numpy as np
from pathlib import Path
import time
import sys

def preprocess_all(src_dir: Path, dst_dir: Path, force: bool = False):
    src_dir = Path(src_dir)
    dst_dir = Path(dst_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    
    files = sorted(src_dir.glob("20??????_mbo_events.npz"))
    print(f"Found {len(files)} .npz files in {src_dir}")
    
    skipped = 0
    converted = 0
    
    for i, f in enumerate(files):
        date_str = f.stem.replace("_mbo_events", "")
        out_dir = dst_dir / date_str
        
        # Skip if already preprocessed
        if not force and (out_dir / "events.npy").exists():
            skipped += 1
            continue
        
        t0 = time.time()
        try:
            data = np.load(f)
            out_dir.mkdir(exist_ok=True)
            np.save(out_dir / "events.npy", data["events"])
            np.save(out_dir / "event_type_raw.npy", data["event_type_raw"])
            np.save(out_dir / "timestamps.npy", data["timestamps"])
            elapsed = time.time() - t0
            converted += 1
            print(f"  [{i+1}/{len(files)}] {date_str}: {data['events'].shape[0]:,} events -> .npy ({elapsed:.1f}s)")
        except Exception as e:
            print(f"  [{i+1}/{len(files)}] {date_str}: ERROR - {e}")
    
    print(f"\nDone: {converted} converted, {skipped} skipped (already exist)")
    print(f"Fast-load dir: {dst_dir}")

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--src-dir", default="/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3")
    p.add_argument("--dst-dir", default="/home/nick/Lvl3Quant/data/processed/mbo_events_fast")
    p.add_argument("--force", action="store_true")
    args = p.parse_args()
    preprocess_all(args.src_dir, args.dst_dir, args.force)
