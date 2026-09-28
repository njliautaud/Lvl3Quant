#!/usr/bin/env python3
"""
Time-of-Day Analysis — Production Meta-Model v1 (SHORT predictions)
====================================================================
Reconstructs timestamps for each meta-model prediction by replaying
the exact pipeline: CNN-Mamba bulk OOT -> top-3% short filter -> MBO timestamps.

Buckets predictions into 30-minute windows and reports edge metrics per bucket.
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pytz

BASE = Path('/home/jupiter/Lvl3Quant')
META_DIR = BASE / 'output' / 'meta_production_v1'
PRED_DIR = BASE / 'output' / 'cnn_mamba_v2_bulk_oot_v2'
MBO_DIR = BASE / 'data' / 'processed' / 'mbo_events_smart_v3'
OUT_DIR = BASE / 'output' / 'tod_analysis_v1'
OUT_DIR.mkdir(parents=True, exist_ok=True)

NY = pytz.timezone('America/New_York')

# Same constants as train_meta_production_v1.py
SHORT_PERCENTILE = 3
COMMISSION_RT = 0.376
STOP_TICKS = 2.0
STOP_SLIPPAGE = 1.0

# 30-minute buckets from 9:30 to 16:00
BUCKET_EDGES = []
for h in range(9, 16):
    for m in (0, 30):
        if h == 9 and m == 0:
            continue  # skip 9:00
        BUCKET_EDGES.append((h, m))
BUCKET_EDGES.append((16, 0))

BUCKET_LABELS = []
for i in range(len(BUCKET_EDGES) - 1):
    h1, m1 = BUCKET_EDGES[i]
    h2, m2 = BUCKET_EDGES[i + 1]
    BUCKET_LABELS.append(f"{h1:02d}:{m1:02d}-{h2:02d}:{m2:02d}")


def minutes_since_midnight(h, m):
    return h * 60 + m


def reconstruct_timestamps_for_fold(date_str):
    """
    Replay the exact meta-model pipeline for one date to get timestamps
    for the top-3% short signals.

    Returns: timestamps_ns array aligned with meta-model predictions for this fold.
    """
    pred_file = PRED_DIR / f'{date_str}_predictions.npz'
    mbo_file = MBO_DIR / f'{date_str}_mbo_events.npz'

    if not pred_file.exists() or not mbo_file.exists():
        print(f"  SKIP {date_str}: missing pred or mbo file")
        return None

    pred_data = np.load(pred_file, allow_pickle=True)
    preds = pred_data['predictions']
    n_windows = int(pred_data['n_windows'])
    window_size = int(pred_data['window_size'])
    stride = int(pred_data['stride'])

    mbo = np.load(mbo_file, allow_pickle=True)
    events = mbo['events']
    l1s = mbo['labels_1s']
    l5s = mbo['labels_5s'] if 'labels_5s' in mbo else None
    timestamps = mbo['timestamps']

    # Compute window indices (same as train_meta_production_v1.py)
    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = min(len(events), len(l1s)) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    preds = preds[:len(indices)]

    feat = events[indices]
    label_1s = l1s[indices]
    label_5s = l5s[indices] if l5s is not None else None
    ts = timestamps[indices]

    # Valid mask (same as training script)
    valid_mask = ~(np.isnan(label_1s) | np.any(np.isnan(feat), axis=1))
    if label_5s is not None:
        valid_mask &= ~np.isnan(label_5s)

    preds = preds[valid_mask]
    ts = ts[valid_mask]

    if len(preds) == 0:
        return None

    # Top 3% short filter (same as training script)
    pred_1s = preds[:, 0]
    threshold = np.percentile(pred_1s, SHORT_PERCENTILE)
    short_mask = pred_1s <= threshold

    return ts[short_mask]


def compute_bucket_stats(pnl_arr):
    """Compute stats for a bucket of P&L values."""
    n = len(pnl_arr)
    if n == 0:
        return {'trades': 0, 'mean_pnl': 0, 'wr': 0, 'pf': 0, 'sharpe': 0, 'total_pnl': 0}

    mean_pnl = float(np.mean(pnl_arr))
    wr = float(np.mean(pnl_arr > 0)) * 100
    wins = pnl_arr[pnl_arr > 0].sum()
    losses = abs(pnl_arr[pnl_arr < 0].sum())
    pf = float(wins / losses) if losses > 0 else float('inf')
    std = float(np.std(pnl_arr))
    sharpe = float(mean_pnl / std) if std > 0 else 0
    total_pnl = float(np.sum(pnl_arr))

    return {
        'trades': n,
        'mean_pnl': mean_pnl,
        'wr': wr,
        'pf': pf,
        'sharpe': sharpe,
        'total_pnl': total_pnl,
    }


def assign_bucket(ts_ns):
    """Convert nanosecond timestamp to bucket index."""
    dt = datetime.fromtimestamp(ts_ns / 1e9, tz=NY)
    mins = dt.hour * 60 + dt.minute
    for i in range(len(BUCKET_EDGES) - 1):
        edge_start = minutes_since_midnight(*BUCKET_EDGES[i])
        edge_end = minutes_since_midnight(*BUCKET_EDGES[i + 1])
        if edge_start <= mins < edge_end:
            return i
    return -1  # outside trading hours


def main():
    print("=" * 80)
    print("TIME-OF-DAY ANALYSIS — Production Meta-Model v1 (SHORTS)")
    print("=" * 80)

    # Load meta-model results
    with open(META_DIR / 'results.json') as f:
        results = json.load(f)

    meta = np.load(META_DIR / 'concat_predictions.npz')
    meta_preds = meta['predictions']
    meta_actuals = meta['actuals']
    per_fold = results['per_fold']

    print(f"\nMeta-model: {len(meta_preds):,} predictions across {len(per_fold)} folds")
    print(f"Actuals already net of {COMMISSION_RT} tick commission\n")

    # Reconstruct timestamps for each fold
    all_timestamps = []
    all_meta_preds = []
    all_meta_actuals = []
    offset = 0

    for fold in per_fold:
        date_str = fold['date']
        n_test = fold['n_test']
        fold_preds = meta_preds[offset:offset + n_test]
        fold_actuals = meta_actuals[offset:offset + n_test]
        offset += n_test

        ts = reconstruct_timestamps_for_fold(date_str)
        if ts is None:
            print(f"  WARNING: Could not reconstruct timestamps for {date_str}")
            continue

        if len(ts) != n_test:
            print(f"  WARNING: {date_str} timestamp count {len(ts)} != n_test {n_test}, "
                  f"using min({len(ts)}, {n_test})")
            n_use = min(len(ts), n_test)
            ts = ts[:n_use]
            fold_preds = fold_preds[:n_use]
            fold_actuals = fold_actuals[:n_use]

        all_timestamps.append(ts)
        all_meta_preds.append(fold_preds)
        all_meta_actuals.append(fold_actuals)
        print(f"  {date_str}: {len(ts):,} signals recovered")

    timestamps = np.concatenate(all_timestamps)
    preds = np.concatenate(all_meta_preds)
    actuals = np.concatenate(all_meta_actuals)
    print(f"\nTotal matched: {len(timestamps):,} / {len(meta_preds):,}")

    # Assign buckets
    buckets = np.array([assign_bucket(t) for t in timestamps])

    # ===== TABLE 1: ALL META SIGNALS =====
    print("\n" + "=" * 80)
    print("TABLE 1: ALL META-MODEL SHORT SIGNALS BY TIME-OF-DAY")
    print("=" * 80)
    header = f"{'Bucket':<14} {'Trades':>7} {'Mean PnL':>9} {'WR%':>6} {'PF':>6} {'Sharpe':>7} {'Total PnL':>10}"
    print(header)
    print("-" * len(header))

    table1_data = []
    for i, label in enumerate(BUCKET_LABELS):
        mask = buckets == i
        stats = compute_bucket_stats(actuals[mask])
        table1_data.append({'bucket': label, **stats})
        if stats['trades'] > 0:
            pf_str = f"{stats['pf']:.2f}" if stats['pf'] < 100 else "inf"
            print(f"{label:<14} {stats['trades']:>7,} {stats['mean_pnl']:>+9.3f} "
                  f"{stats['wr']:>5.1f}% {pf_str:>6} {stats['sharpe']:>+7.3f} {stats['total_pnl']:>+10.2f}")

    # Totals
    total_stats = compute_bucket_stats(actuals)
    pf_str = f"{total_stats['pf']:.2f}" if total_stats['pf'] < 100 else "inf"
    print("-" * len(header))
    print(f"{'TOTAL':<14} {total_stats['trades']:>7,} {total_stats['mean_pnl']:>+9.3f} "
          f"{total_stats['wr']:>5.1f}% {pf_str:>6} {total_stats['sharpe']:>+7.3f} {total_stats['total_pnl']:>+10.2f}")

    # ===== TABLE 2: TOP 50% META SIGNALS (meta_pred > median) =====
    print("\n" + "=" * 80)
    print("TABLE 2: TOP 50% META-FILTERED SHORTS BY TIME-OF-DAY")
    print("  (meta prediction > median — model says these are the best shorts)")
    print("=" * 80)

    median_pred = np.median(preds)
    top50_mask = preds >= median_pred
    print(f"  Meta prediction median: {median_pred:.4f}")
    print(f"  Top 50% count: {top50_mask.sum():,} / {len(preds):,}\n")

    print(header)
    print("-" * len(header))

    table2_data = []
    for i, label in enumerate(BUCKET_LABELS):
        mask = (buckets == i) & top50_mask
        stats = compute_bucket_stats(actuals[mask])
        table2_data.append({'bucket': label, **stats})
        if stats['trades'] > 0:
            pf_str = f"{stats['pf']:.2f}" if stats['pf'] < 100 else "inf"
            print(f"{label:<14} {stats['trades']:>7,} {stats['mean_pnl']:>+9.3f} "
                  f"{stats['wr']:>5.1f}% {pf_str:>6} {stats['sharpe']:>+7.3f} {stats['total_pnl']:>+10.2f}")

    top50_stats = compute_bucket_stats(actuals[top50_mask])
    pf_str = f"{top50_stats['pf']:.2f}" if top50_stats['pf'] < 100 else "inf"
    print("-" * len(header))
    print(f"{'TOTAL':<14} {top50_stats['trades']:>7,} {top50_stats['mean_pnl']:>+9.3f} "
          f"{top50_stats['wr']:>5.1f}% {pf_str:>6} {top50_stats['sharpe']:>+7.3f} {top50_stats['total_pnl']:>+10.2f}")

    # ===== TABLE 3: TOP 25% META SIGNALS =====
    print("\n" + "=" * 80)
    print("TABLE 3: TOP 25% META-FILTERED SHORTS BY TIME-OF-DAY")
    print("  (meta prediction > 75th percentile)")
    print("=" * 80)

    p75_pred = np.percentile(preds, 75)
    top25_mask = preds >= p75_pred
    print(f"  Meta prediction 75th pct: {p75_pred:.4f}")
    print(f"  Top 25% count: {top25_mask.sum():,} / {len(preds):,}\n")

    print(header)
    print("-" * len(header))

    table3_data = []
    for i, label in enumerate(BUCKET_LABELS):
        mask = (buckets == i) & top25_mask
        stats = compute_bucket_stats(actuals[mask])
        table3_data.append({'bucket': label, **stats})
        if stats['trades'] > 0:
            pf_str = f"{stats['pf']:.2f}" if stats['pf'] < 100 else "inf"
            print(f"{label:<14} {stats['trades']:>7,} {stats['mean_pnl']:>+9.3f} "
                  f"{stats['wr']:>5.1f}% {pf_str:>6} {stats['sharpe']:>+7.3f} {stats['total_pnl']:>+10.2f}")

    top25_stats = compute_bucket_stats(actuals[top25_mask])
    pf_str = f"{top25_stats['pf']:.2f}" if top25_stats['pf'] < 100 else "inf"
    print("-" * len(header))
    print(f"{'TOTAL':<14} {top25_stats['trades']:>7,} {top25_stats['mean_pnl']:>+9.3f} "
          f"{top25_stats['wr']:>5.1f}% {pf_str:>6} {top25_stats['sharpe']:>+7.3f} {top25_stats['total_pnl']:>+10.2f}")

    # ===== Save results =====
    results_out = {
        'analysis': 'tod_production_meta_v1_shorts',
        'total_signals': int(len(timestamps)),
        'total_matched': int(len(timestamps)),
        'folds': len(per_fold),
        'commission_already_deducted': COMMISSION_RT,
        'bucket_labels': BUCKET_LABELS,
        'all_signals': table1_data,
        'top50_filtered': table2_data,
        'top25_filtered': table3_data,
        'all_total': total_stats,
        'top50_total': top50_stats,
        'top25_total': top25_stats,
    }

    out_file = OUT_DIR / 'tod_results.json'
    with open(out_file, 'w') as f:
        json.dump(results_out, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")

    # Also save the raw per-signal data for further analysis
    np.savez_compressed(
        OUT_DIR / 'tod_signals.npz',
        timestamps=timestamps,
        meta_predictions=preds,
        meta_actuals=actuals,
        buckets=buckets,
    )
    print(f"Signal data saved to {OUT_DIR / 'tod_signals.npz'}")


if __name__ == '__main__':
    main()
