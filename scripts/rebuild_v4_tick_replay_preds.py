#!/usr/bin/env python3
"""
Rebuild V4 Multihead Predictions for Tick Replay
==================================================
Properly assigns predictions to calendar dates using timestamps.

The v4 multihead training uses processed MBO events files that span
multiple calendar days (Globex sessions). The predictions at stride=500,
seq_len=100 need to be:
1. Timestamped using the processed events file
2. Grouped by calendar date (ET timezone for ES futures)
3. Saved as per-date NPZ files for tick replay alignment

Author: Claude
"""

import numpy as np
import os
import glob
from datetime import datetime, timezone, timedelta
from collections import defaultdict

# =============================================================================
# Config
# =============================================================================

FOLD_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_pressure_v1'
PROC_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3'
MBO_RAW_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds_v2'

# Training parameters (from training.log)
STRIDE = 500
SEQ_LEN = 100

# ET timezone offset for date assignment
# During EST: UTC-5, during EDT: UTC-4
# ES RTH: 9:30-16:00 ET, Globex: 18:00 prev day - 17:00 current day
# For date assignment, use the calendar date in ET of when the prediction occurs


def get_et_date(ns_timestamp):
    """Convert nanosecond timestamp to ET calendar date string."""
    dt = datetime.fromtimestamp(ns_timestamp / 1e9, tz=timezone.utc)
    # Approximate ET offset (EDT: Apr-Nov, EST: Nov-Mar)
    month = dt.month
    if month >= 3 and month <= 10:
        et_offset = timedelta(hours=-4)  # EDT
    else:
        et_offset = timedelta(hours=-5)  # EST
    et_dt = dt + et_offset

    # For ES futures, the "trading day" starts at 18:00 ET the previous calendar day
    # Events between 18:00-23:59 belong to the NEXT trading day
    if et_dt.hour >= 18:
        et_dt += timedelta(days=1)

    return et_dt.strftime('%Y%m%d')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Find available raw MBO dates
    raw_mbo_dates = set()
    for f in glob.glob(os.path.join(MBO_RAW_DIR, 'glbx-mdp3-*.mbo.dbn.zst')):
        d = os.path.basename(f).split('-')[2].split('.')[0]
        raw_mbo_dates.add(d)
    print(f"Found {len(raw_mbo_dates)} raw MBO dates")

    # Process each fold
    date_predictions = defaultdict(lambda: {
        'dir_1s': [], 'composite': [], 'eofi_1s': [], 'pdi_1s': [],
        'timestamps': [], 'proc_indices': []
    })

    for fold in range(126, 152):
        fold_path = os.path.join(FOLD_DIR, f'fold_{fold}_oot_predictions.npz')
        if not os.path.exists(fold_path):
            continue

        d = np.load(fold_path, allow_pickle=True)
        n_preds = d['preds_dir'].shape[0]
        oot_file = d['oot_files'][0].split('/')[-1]
        proc_date = oot_file.replace('_mbo_events.npz', '')

        # Load processed events timestamps
        proc_path = os.path.join(PROC_DIR, oot_file)
        if not os.path.exists(proc_path):
            # Try local path variant
            proc_path = os.path.join(PROC_DIR, f'{proc_date}_mbo_events.npz')

        if not os.path.exists(proc_path):
            print(f"  fold {fold}: SKIP - processed file not found for {proc_date}")
            continue

        with np.load(proc_path, allow_pickle=False) as proc:
            proc_ts = proc['timestamps']
            n_events = proc['events'].shape[0]

        # Compute prediction indices in processed events
        pred_indices = np.arange(SEQ_LEN, SEQ_LEN + n_preds * STRIDE, STRIDE)
        # The prediction is made at the END of the sequence window
        # So pred[i] corresponds to processed event index seq_len + i*stride

        # Clamp to valid range
        valid = pred_indices < n_events
        pred_indices = pred_indices[valid]
        n_valid = len(pred_indices)

        if n_valid != n_preds:
            print(f"  fold {fold}: WARNING - only {n_valid}/{n_preds} predictions have valid indices")

        # Get timestamps for each prediction
        pred_timestamps = proc_ts[pred_indices]

        # Extract predictions
        dir_1s = d['preds_dir'][:n_valid, 0]  # 1s direction prediction

        # Compute composite signal (dir + eofi)
        eofi_1s = d['preds_eofi'][:n_valid, 0] if 'preds_eofi' in d else np.zeros(n_valid)
        pdi_1s = d['preds_pdi'][:n_valid, 0] if 'preds_pdi' in d else np.zeros(n_valid)

        # Composite: dir * (1 + eofi_weight * eofi_signal)
        eofi_signal = eofi_1s - np.median(eofi_1s)  # center
        composite = dir_1s * (1.0 + 0.3 * np.sign(dir_1s) * eofi_signal)

        # Group by calendar date
        for i in range(n_valid):
            cal_date = get_et_date(pred_timestamps[i])
            if cal_date in raw_mbo_dates:
                date_predictions[cal_date]['dir_1s'].append(dir_1s[i])
                date_predictions[cal_date]['composite'].append(composite[i])
                date_predictions[cal_date]['eofi_1s'].append(eofi_1s[i])
                date_predictions[cal_date]['pdi_1s'].append(pdi_1s[i])
                date_predictions[cal_date]['timestamps'].append(pred_timestamps[i])
                date_predictions[cal_date]['proc_indices'].append(pred_indices[i])

        n_dates = len(set(get_et_date(t) for t in pred_timestamps))
        print(f"  fold {fold}: {n_valid} preds from {proc_date}, spans {n_dates} calendar dates")

    # Save per-date files
    print(f"\n{'='*60}")
    print(f"SAVING {len(date_predictions)} date files")
    print(f"{'='*60}")

    total_preds = 0
    for date in sorted(date_predictions.keys()):
        dp = date_predictions[date]
        n = len(dp['dir_1s'])
        if n == 0:
            continue

        # Sort by timestamp
        order = np.argsort(dp['timestamps'])

        np.savez(
            os.path.join(OUTPUT_DIR, f'oot_{date}.npz'),
            pred_log_ret_1s=np.array(dp['dir_1s'], dtype=np.float32)[order],
            composite_signal=np.array(dp['composite'], dtype=np.float32)[order],
            eofi_1s=np.array(dp['eofi_1s'], dtype=np.float32)[order],
            pdi_1s=np.array(dp['pdi_1s'], dtype=np.float32)[order],
            timestamps_ns=np.array(dp['timestamps'], dtype=np.int64)[order],
            n_preds=n,
        )
        total_preds += n
        print(f"  {date}: {n} predictions")

    print(f"\nTotal: {total_preds} predictions across {len(date_predictions)} dates")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
