#!/usr/bin/env python3
"""
Rebuild V4 Multihead Predictions WITH Timestamps
=================================================
Fixes critical temporal mismatch bug: the old prediction builder saved predictions
WITHOUT timestamps, causing tick replay scripts to map predictions by positional
index (PRED_STRIDE=250 on RAW events) instead of by timestamp (STRIDE=500 on
PROCESSED events). This caused up to 5 hours of drift by end of day.

This script adds pred_timestamps_ns (int64 nanosecond timestamps) to each
per-date prediction file so tick replay can align predictions by actual time.

Output: output/v4_tick_replay_preds_ts/oot_{date}.npz
"""

import numpy as np
import os
from datetime import datetime, timezone, timedelta
from collections import defaultdict

FOLD_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_pressure_v1'
PROC_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/v4_tick_replay_preds_ts'

STRIDE = 500
SEQ_LEN = 100
FOLD_START = 126
FOLD_END = 175
MIN_PREDS = 50


def get_et_date(ns_timestamp):
    """Convert nanosecond timestamp to ET calendar date string."""
    dt = datetime.fromtimestamp(ns_timestamp / 1e9, tz=timezone.utc)
    month = dt.month
    if month >= 3 and month <= 10:
        et_offset = timedelta(hours=-4)  # EDT
    else:
        et_offset = timedelta(hours=-5)  # EST
    et_dt = dt + et_offset
    if et_dt.hour >= 18:
        et_dt += timedelta(days=1)
    return et_dt.strftime('%Y%m%d')


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Collect all predictions keyed by ET date
    date_predictions = defaultdict(lambda: {
        'dir_1s': [], 'composite': [], 'eofi_1s': [], 'pdi_1s': [],
        'timestamps': []
    })

    folds_processed = 0
    folds_skipped = 0

    for fold in range(FOLD_START, FOLD_END + 1):
        fold_path = os.path.join(FOLD_DIR, f'fold_{fold}_oot_predictions.npz')
        if not os.path.exists(fold_path):
            continue

        d = np.load(fold_path, allow_pickle=True)
        n_preds = d['preds_dir'].shape[0]
        oot_file = str(d['oot_files'][0]).split('/')[-1]
        proc_date = oot_file.replace('_mbo_events.npz', '')

        # Load processed events to get timestamps
        proc_path = os.path.join(PROC_DIR, oot_file)
        if not os.path.exists(proc_path):
            proc_path = os.path.join(PROC_DIR, f'{proc_date}_mbo_events.npz')

        if not os.path.exists(proc_path):
            print(f"  fold {fold}: SKIP - no processed file for {proc_date}")
            folds_skipped += 1
            continue

        with np.load(proc_path, allow_pickle=False) as proc:
            proc_ts = proc['timestamps']
            n_events = proc['events'].shape[0]

        # Map prediction indices to processed event indices
        pred_indices = np.arange(SEQ_LEN, SEQ_LEN + n_preds * STRIDE, STRIDE)
        valid = pred_indices < n_events
        pred_indices = pred_indices[valid]
        n_valid = len(pred_indices)

        if n_valid == 0:
            print(f"  fold {fold}: SKIP - no valid predictions for {proc_date}")
            folds_skipped += 1
            continue

        # Get the CORRECT timestamps from processed events
        pred_timestamps = proc_ts[pred_indices]

        # Extract prediction arrays
        dir_1s = d['preds_dir'][:n_valid, 0]
        eofi_1s = d['preds_eofi'][:n_valid, 0] if 'preds_eofi' in d else np.zeros(n_valid)
        pdi_1s = d['preds_pdi'][:n_valid, 0] if 'preds_pdi' in d else np.zeros(n_valid)

        # Composite signal: dir * (1 + 0.3 * sign(dir) * (eofi - median(eofi)))
        eofi_signal = eofi_1s - np.median(eofi_1s)
        composite = dir_1s * (1.0 + 0.3 * np.sign(dir_1s) * eofi_signal)

        # Group by ET date
        for i in range(n_valid):
            cal_date = get_et_date(pred_timestamps[i])
            date_predictions[cal_date]['dir_1s'].append(dir_1s[i])
            date_predictions[cal_date]['composite'].append(composite[i])
            date_predictions[cal_date]['eofi_1s'].append(eofi_1s[i])
            date_predictions[cal_date]['pdi_1s'].append(pdi_1s[i])
            date_predictions[cal_date]['timestamps'].append(pred_timestamps[i])

        folds_processed += 1
        n_dates = len(set(get_et_date(t) for t in pred_timestamps))
        print(f"  fold {fold}: {n_valid} preds from {proc_date}, spans {n_dates} ET dates")

    # Save per-date files with timestamps
    print(f"\n{'='*60}")
    print(f"Processed {folds_processed} folds, skipped {folds_skipped}")
    print(f"Found {len(date_predictions)} unique ET dates")
    print(f"{'='*60}\n")

    total_preds = 0
    dates_saved = 0
    dates_skipped = 0

    for date in sorted(date_predictions.keys()):
        dp = date_predictions[date]
        n = len(dp['dir_1s'])
        if n < MIN_PREDS:
            print(f"  {date}: SKIP ({n} preds < {MIN_PREDS} minimum)")
            dates_skipped += 1
            continue

        # Sort all arrays by timestamp
        timestamps = np.array(dp['timestamps'], dtype=np.int64)
        order = np.argsort(timestamps)

        np.savez(
            os.path.join(OUTPUT_DIR, f'oot_{date}.npz'),
            pred_log_ret_1s=np.array(dp['dir_1s'], dtype=np.float32)[order],
            composite_signal=np.array(dp['composite'], dtype=np.float32)[order],
            eofi_1s=np.array(dp['eofi_1s'], dtype=np.float32)[order],
            pdi_1s=np.array(dp['pdi_1s'], dtype=np.float32)[order],
            pred_timestamps_ns=timestamps[order],
            n_preds=n,
        )
        total_preds += n
        dates_saved += 1
        print(f"  {date}: {n} predictions saved")

    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Folds processed:  {folds_processed}")
    print(f"Dates saved:      {dates_saved}")
    print(f"Dates skipped:    {dates_skipped} (< {MIN_PREDS} preds)")
    print(f"Total predictions: {total_preds}")
    print(f"Output directory:  {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
