#!/usr/bin/env python3
"""
extract_queue_fill_windows.py — Pre-extract training windows on Jupiter (64GB RAM)
for queue-position predictor training on Razer (16GB RAM).

Loads each day's smart_v3 npz, extracts fixed-size windows with fill labels,
saves as compact npz files that Razer can load instantly.

Output per date:
  /output/queue_fill_windows/<date>_windows.npz
    windows: (N, 25, 256) float32  — ~3000 windows per day, ~77 MB
    bid_labels: (N, 3) float32     — fill labels for 1s/5s/10s
    ask_labels: (N, 3) float32

Author: Claude (Lvl3 Quant)
"""
import gc
import sys
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Config
MBO_DIR = Path('/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3')
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/queue_fill_windows')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_SIZE = 256
STRIDE = 64
SUBSAMPLE = 4
MAX_WINDOWS_PER_DAY = 5000
FILL_THRESHOLD_TICKS = 0.25
HORIZONS = ['1s', '5s', '10s']
N_FEATURES = 25


def process_one_day(date_str: str) -> dict:
    """Extract windows from one day. Returns summary dict."""
    fpath = MBO_DIR / f'{date_str}_mbo_events.npz'
    out_path = OUTPUT_DIR / f'{date_str}_windows.npz'

    if out_path.exists():
        # Already processed
        return {'date': date_str, 'status': 'exists', 'n_windows': -1}

    try:
        d = np.load(fpath, allow_pickle=True)
        events = d['events']  # (N, 25) — keep native dtype

        if events.ndim != 2 or events.shape[1] != N_FEATURES:
            return {'date': date_str, 'status': f'bad_shape_{events.shape}', 'n_windows': 0}

        # Load labels
        labels = {}
        for hz in HORIZONS:
            key = f'labels_{hz}'
            if key not in d:
                del d, events
                return {'date': date_str, 'status': f'missing_{key}', 'n_windows': 0}
            labels[hz] = d[key]

        n_events = len(events)

        # Find valid positions
        valid = np.ones(n_events, dtype=bool)
        for hz in HORIZONS:
            valid &= ~np.isnan(labels[hz])
        valid[:WINDOW_SIZE] = False

        valid_indices = np.where(valid)[0]
        valid_indices = valid_indices[::STRIDE * SUBSAMPLE]

        if len(valid_indices) > MAX_WINDOWS_PER_DAY:
            rng = np.random.RandomState(hash(date_str) % 2**31)
            valid_indices = rng.choice(valid_indices, MAX_WINDOWS_PER_DAY, replace=False)
            valid_indices.sort()

        if len(valid_indices) == 0:
            del d, events, labels
            return {'date': date_str, 'status': 'no_valid', 'n_windows': 0}

        # Extract windows — each is (256, 25) → transpose to (25, 256)
        windows_list = []
        good_indices = []

        for idx in valid_indices:
            w = events[idx - WINDOW_SIZE:idx].astype(np.float32)
            if np.any(np.isnan(w)):
                continue
            windows_list.append(w.T)  # (25, 256)
            good_indices.append(idx)

        del d, events
        gc.collect()

        if len(windows_list) == 0:
            del labels
            return {'date': date_str, 'status': 'all_nan', 'n_windows': 0}

        windows = np.stack(windows_list, dtype=np.float32)
        del windows_list
        good_indices = np.array(good_indices)

        # Extract labels
        bid_lbl = np.zeros((len(good_indices), len(HORIZONS)), dtype=np.float32)
        ask_lbl = np.zeros((len(good_indices), len(HORIZONS)), dtype=np.float32)

        for hi, hz in enumerate(HORIZONS):
            pc = labels[hz][good_indices].astype(np.float32)
            bid_lbl[:, hi] = (pc <= -FILL_THRESHOLD_TICKS).astype(np.float32)
            ask_lbl[:, hi] = (pc >= FILL_THRESHOLD_TICKS).astype(np.float32)

        del labels
        gc.collect()

        # Also save a normalization sample (first 20K events)
        # Re-load just for the sample — this is on Jupiter with 64GB so it's fine
        d2 = np.load(fpath, allow_pickle=True)
        norm_sample = d2['events'][:20000].astype(np.float32)
        del d2

        # Save
        np.savez_compressed(out_path,
                           windows=windows,
                           bid_labels=bid_lbl,
                           ask_labels=ask_lbl,
                           norm_sample=norm_sample,
                           date=date_str)

        return {'date': date_str, 'status': 'ok', 'n_windows': len(windows)}

    except Exception as e:
        return {'date': date_str, 'status': f'error: {e}', 'n_windows': 0}


def main():
    print("=" * 70)
    print("QUEUE FILL WINDOW EXTRACTION (Jupiter → Razer)")
    print("=" * 70)

    # Get all dates
    dates = sorted([f.stem.split('_')[0] for f in MBO_DIR.glob('*_mbo_events.npz')])
    print(f"Found {len(dates)} dates")

    # Process sequentially (each file is 1-2 GB, parallel would OOM even on 64GB)
    t0 = time.time()
    ok_count = 0
    skip_count = 0

    for i, date_str in enumerate(dates):
        t1 = time.time()
        result = process_one_day(date_str)
        dt = time.time() - t1

        if result['status'] == 'ok':
            ok_count += 1
            print(f"  [{i+1}/{len(dates)}] {date_str}: {result['n_windows']} windows ({dt:.1f}s)")
        elif result['status'] == 'exists':
            skip_count += 1
            if i % 50 == 0:
                print(f"  [{i+1}/{len(dates)}] {date_str}: already exists (skipping)")
        else:
            print(f"  [{i+1}/{len(dates)}] {date_str}: {result['status']} ({dt:.1f}s)")

        gc.collect()

    total_time = time.time() - t0
    print(f"\nDone: {ok_count} processed, {skip_count} skipped, {total_time:.0f}s total")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
