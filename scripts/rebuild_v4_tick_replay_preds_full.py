#!/usr/bin/env python3
"""
Rebuild V4 Multihead Predictions for Tick Replay — FULL RANGE
==============================================================
Processes ALL folds (126-175) to maximize OOT date coverage.
Outputs to v4_multihead_tick_replay_preds_v3/ for use with tick replay engine.
"""

import numpy as np
import os
import glob
from datetime import datetime, timezone, timedelta
from collections import defaultdict

FOLD_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_pressure_v1'
PROC_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3'
MBO_RAW_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
MBO_PREPROC_DIR = '/home/jupiter/Lvl3Quant/data/preprocessed_mbo'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds_v3'

STRIDE = 500
SEQ_LEN = 100


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

    # Find MBO dates we have preprocessed data for
    mbo_dates = set()
    for f in os.listdir(MBO_PREPROC_DIR):
        if f.endswith('.npz'):
            d = f.replace('mbo_', '').replace('.npz', '')
            mbo_dates.add(d)
    print(f"Preprocessed MBO dates available: {len(mbo_dates)}")

    # Also check raw MBO
    raw_mbo_dates = set()
    for f in glob.glob(os.path.join(MBO_RAW_DIR, 'glbx-mdp3-*.mbo.dbn.zst')):
        d = os.path.basename(f).split('-')[2].split('.')[0]
        raw_mbo_dates.add(d)
    print(f"Raw MBO dates: {len(raw_mbo_dates)}")

    all_valid_dates = mbo_dates | raw_mbo_dates
    print(f"Total valid dates: {len(all_valid_dates)}")

    # Process ALL folds
    date_predictions = defaultdict(lambda: {
        'dir_1s': [], 'composite': [], 'eofi_1s': [], 'pdi_1s': [],
        'timestamps': [], 'proc_indices': []
    })

    for fold in range(126, 200):  # wide range to catch everything
        fold_path = os.path.join(FOLD_DIR, f'fold_{fold}_oot_predictions.npz')
        if not os.path.exists(fold_path):
            continue

        d = np.load(fold_path, allow_pickle=True)
        n_preds = d['preds_dir'].shape[0]
        oot_file = str(d['oot_files'][0]).split('/')[-1]
        proc_date = oot_file.replace('_mbo_events.npz', '')

        # Load processed events timestamps
        proc_path = os.path.join(PROC_DIR, oot_file)
        if not os.path.exists(proc_path):
            proc_path = os.path.join(PROC_DIR, f'{proc_date}_mbo_events.npz')

        if not os.path.exists(proc_path):
            print(f"  fold {fold}: SKIP - no processed file for {proc_date}")
            continue

        with np.load(proc_path, allow_pickle=False) as proc:
            proc_ts = proc['timestamps']
            n_events = proc['events'].shape[0]

        pred_indices = np.arange(SEQ_LEN, SEQ_LEN + n_preds * STRIDE, STRIDE)
        valid = pred_indices < n_events
        pred_indices = pred_indices[valid]
        n_valid = len(pred_indices)

        pred_timestamps = proc_ts[pred_indices]

        dir_1s = d['preds_dir'][:n_valid, 0]
        eofi_1s = d['preds_eofi'][:n_valid, 0] if 'preds_eofi' in d else np.zeros(n_valid)
        pdi_1s = d['preds_pdi'][:n_valid, 0] if 'preds_pdi' in d else np.zeros(n_valid)

        eofi_signal = eofi_1s - np.median(eofi_1s)
        composite = dir_1s * (1.0 + 0.3 * np.sign(dir_1s) * eofi_signal)

        for i in range(n_valid):
            cal_date = get_et_date(pred_timestamps[i])
            if cal_date in all_valid_dates:
                date_predictions[cal_date]['dir_1s'].append(dir_1s[i])
                date_predictions[cal_date]['composite'].append(composite[i])
                date_predictions[cal_date]['eofi_1s'].append(eofi_1s[i])
                date_predictions[cal_date]['pdi_1s'].append(pdi_1s[i])
                date_predictions[cal_date]['timestamps'].append(pred_timestamps[i])
                date_predictions[cal_date]['proc_indices'].append(pred_indices[i])

        n_dates = len(set(get_et_date(t) for t in pred_timestamps))
        print(f"  fold {fold}: {n_valid} preds from {proc_date}, spans {n_dates} dates")

    # Save per-date files
    print(f"\n{'='*60}")
    print(f"SAVING {len(date_predictions)} date files")
    print(f"{'='*60}")

    total_preds = 0
    dates_with_mbo = []
    for date in sorted(date_predictions.keys()):
        dp = date_predictions[date]
        n = len(dp['dir_1s'])
        if n < 50:  # skip dates with tiny number of predictions
            print(f"  {date}: SKIP ({n} preds < 50 minimum)")
            continue

        order = np.argsort(dp['timestamps'])

        np.savez(
            os.path.join(OUTPUT_DIR, f'oot_{date}.npz'),
            pred_log_ret_1s=np.array(dp['dir_1s'], dtype=np.float32)[order],
            composite_signal=np.array(dp['composite'], dtype=np.float32)[order],
            eofi_1s=np.array(dp['eofi_1s'], dtype=np.float32)[order],
            pdi_1s=np.array(dp['pdi_1s'], dtype=np.float32)[order],
            n_preds=n,
        )
        total_preds += n
        has_mbo = date in mbo_dates
        dates_with_mbo.append(date) if has_mbo else None
        mbo_flag = " ✅ HAS_MBO" if has_mbo else " ❌ no_mbo"
        print(f"  {date}: {n} predictions{mbo_flag}")

    print(f"\nTotal: {total_preds} predictions across {len(date_predictions)} dates")
    print(f"Dates with BOTH predictions AND MBO: {len(dates_with_mbo)}")
    print(f"Output: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
