#!/usr/bin/env python3
"""
Market-Order Entry Analysis for Tick-Level Replay
==================================================

Hypothesis: Passive FIFO entries suffer adverse selection (only fill when
market moves against you). Market-order entries avoid this but cost 1 extra
tick per entry.

Method: For each prediction above threshold, use the preprocessed tick data
to compute realized MFE/MAE within hold_seconds. Market order entry means
immediate fill at current ask (long) or bid (short).

This is analysis-only (no full sim), measuring raw edge BEFORE exit strategy.
If raw edge < 1.376 ticks (market entry cost), no exit strategy can save it.

Uses preprocessed .npz files from tick_replay_fast.py.
"""

import numpy as np
import glob
import os
import sys
import json
from collections import defaultdict

# Canonical costs
COMMISSION_RT_TICKS = 0.376
MARKET_ENTRY_COST = COMMISSION_RT_TICKS + 1.0  # 1.376 ticks
PASSIVE_EXIT_COST = COMMISSION_RT_TICKS        # 0.376 ticks
TICK_SIZE = 0.25  # ES tick = 0.25 points (prices stored as float points)

DEFAULT_PRED_DIR = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
DEFAULT_PREPROC_DIR = "/home/jupiter/Lvl3Quant/data/preprocessed_mbo"
PRED_WINDOW = 40000
PRED_STRIDE = 6250  # 250ms stride


def load_predictions(pred_dir, head='pred_log_ret_1s'):
    pred_map = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, 'oot_*.npz'))):
        date = os.path.basename(f).replace('oot_', '').replace('.npz', '')
        data = np.load(f)
        if head in data:
            pred_map[date] = data[head]
    return pred_map


def find_preprocessed(preproc_dir):
    pmap = {}
    for f in sorted(glob.glob(os.path.join(preproc_dir, 'mbo_*.npz'))):
        date = os.path.basename(f).replace('mbo_', '').replace('.npz', '')
        pmap[date] = f
    return pmap


def match_dates(preproc_map, pred_map):
    matched = []
    for date in sorted(set(preproc_map.keys()) & set(pred_map.keys())):
        matched.append((date, preproc_map[date], pred_map[date]))
    return matched


def analyze_market_entry_day(preproc_path, predictions, threshold=0.3,
                              hold_seconds=30.0, horizons=[1, 2, 5, 10, 30]):
    """For each prediction above threshold, compute raw MFE/MAE at market entry."""
    data = np.load(preproc_path)
    ts_ns = data['ts_ns']
    price_arr = data['price']
    action_arr = data['action']
    side_arr = data['side']

    n_events = len(ts_ns)
    n_preds = len(predictions)
    pred_indices = np.arange(PRED_WINDOW, PRED_WINDOW + n_preds * PRED_STRIDE, PRED_STRIDE, dtype=np.int64)
    valid_mask = pred_indices < n_events
    pred_indices = pred_indices[valid_mask]
    predictions = predictions[:len(pred_indices)]

    hold_ns = int(hold_seconds * 1e9)

    # For each valid prediction, find entry price and forward path
    results = []

    for i, (idx, pred) in enumerate(zip(pred_indices, predictions)):
        if abs(pred) < threshold:
            continue

        direction = 1 if pred > 0 else -1  # +1=long, -1=short

        # Find last trade price at entry point (market order fills at this)
        # Look backwards from idx for most recent trade
        entry_price = None
        for j in range(idx, max(idx - 5000, -1), -1):
            if action_arr[j] == 4:  # TRADE action
                entry_price = price_arr[j]
                break

        if entry_price is None:
            continue

        entry_ts = ts_ns[idx]
        end_ts = entry_ts + hold_ns

        # Walk forward through all events within hold window
        mfe_ticks = 0.0
        mae_ticks = 0.0

        # Compute returns at each horizon
        horizon_returns = {}
        horizon_ns = {h: int(h * 1e9) for h in horizons}
        horizon_found = {h: False for h in horizons}

        for j in range(idx + 1, min(idx + 500000, n_events)):
            if ts_ns[j] > end_ts:
                break

            if action_arr[j] == 4:  # Trade event
                move_ticks = (price_arr[j] - entry_price) / TICK_SIZE * direction
                mfe_ticks = max(mfe_ticks, move_ticks)
                mae_ticks = min(mae_ticks, move_ticks)

                # Check horizon targets
                elapsed = ts_ns[j] - entry_ts
                for h in horizons:
                    if not horizon_found[h] and elapsed >= horizon_ns[h]:
                        horizon_returns[h] = move_ticks
                        horizon_found[h] = True

        results.append({
            'direction': direction,
            'confidence': abs(pred),
            'mfe': mfe_ticks,
            'mae': mae_ticks,
            'horizons': horizon_returns,
        })

    return results


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--threshold', type=float, default=0.25)
    parser.add_argument('--hold', type=float, default=30.0)
    parser.add_argument('--max-days', type=int, default=None)
    parser.add_argument('--head', default='pred_log_ret_1s')
    args = parser.parse_args()

    print("=" * 70)
    print("MARKET-ORDER ENTRY ANALYSIS")
    print("=" * 70)
    print(f"Threshold: {args.threshold}")
    print(f"Hold window: {args.hold}s")
    print(f"Prediction head: {args.head}")
    print(f"Market entry cost: {MARKET_ENTRY_COST} ticks")
    print()

    pred_map = load_predictions(DEFAULT_PRED_DIR, head=args.head)
    preproc_map = find_preprocessed(DEFAULT_PREPROC_DIR)
    matched = match_dates(preproc_map, pred_map)
    if args.max_days:
        matched = matched[:args.max_days]

    print(f"Matched dates: {len(matched)}")

    all_results = []
    horizons = [1, 2, 5, 10, 30]

    for date, preproc_path, preds in matched:
        day_results = analyze_market_entry_day(
            preproc_path, preds,
            threshold=args.threshold,
            hold_seconds=args.hold,
            horizons=horizons,
        )
        all_results.extend(day_results)
        n_long = sum(1 for r in day_results if r['direction'] == 1)
        n_short = sum(1 for r in day_results if r['direction'] == -1)
        if day_results:
            avg_mfe = np.mean([r['mfe'] for r in day_results])
            print(f"  {date}: {len(day_results)} signals (L={n_long} S={n_short}), "
                  f"avg MFE={avg_mfe:.2f}t")

    if not all_results:
        print("No signals found!")
        return

    # Aggregate analysis
    print(f"\n{'='*70}")
    print(f"AGGREGATE: {len(all_results)} signals across {len(matched)} days")
    print(f"{'='*70}")

    mfes = np.array([r['mfe'] for r in all_results])
    maes = np.array([r['mae'] for r in all_results])
    directions = np.array([r['direction'] for r in all_results])
    confidences = np.array([r['confidence'] for r in all_results])

    print(f"\nMFE (Maximum Favorable Excursion within {args.hold}s):")
    print(f"  Mean:   {np.mean(mfes):.3f} ticks")
    print(f"  Median: {np.median(mfes):.3f} ticks")
    print(f"  p25:    {np.percentile(mfes, 25):.3f} ticks")
    print(f"  p75:    {np.percentile(mfes, 75):.3f} ticks")
    print(f"  p90:    {np.percentile(mfes, 90):.3f} ticks")

    print(f"\nMAE (Maximum Adverse Excursion within {args.hold}s):")
    print(f"  Mean:   {np.mean(maes):.3f} ticks")
    print(f"  Median: {np.median(maes):.3f} ticks")
    print(f"  p10:    {np.percentile(maes, 10):.3f} ticks")

    # By direction
    for dir_val, dir_name in [(1, 'LONG'), (-1, 'SHORT')]:
        mask = directions == dir_val
        if np.sum(mask) == 0:
            continue
        print(f"\n  {dir_name} ({np.sum(mask)} signals):")
        print(f"    MFE mean: {np.mean(mfes[mask]):.3f}t, MAE mean: {np.mean(maes[mask]):.3f}t")

    # By confidence bucket
    print("\nBy confidence bucket:")
    for lo, hi, label in [(args.threshold, 0.25, f'{args.threshold:.2f}-0.25'),
                          (0.25, 0.30, '0.25-0.30'),
                          (0.30, 0.50, '0.30-0.50'),
                          (0.50, 1.00, '0.50+')]:
        mask = (confidences >= lo) & (confidences < hi)
        if np.sum(mask) < 5:
            continue
        print(f"  [{label}] n={np.sum(mask):>6}, MFE={np.mean(mfes[mask]):>+.3f}t, "
              f"MAE={np.mean(maes[mask]):>+.3f}t, "
              f"MFE-|MAE|={np.mean(mfes[mask])-abs(np.mean(maes[mask])):>+.3f}t")

    # Horizon returns (critical for TP setting)
    print(f"\nRealized returns at each horizon (BEFORE costs):")
    for h in horizons:
        h_rets = [r['horizons'].get(h, np.nan) for r in all_results]
        h_rets = [x for x in h_rets if not np.isnan(x)]
        if len(h_rets) > 10:
            h_rets = np.array(h_rets)
            wr = np.mean(h_rets > 0)
            print(f"  {h:>2}s: mean={np.mean(h_rets):>+.3f}t, median={np.median(h_rets):>+.3f}t, "
                  f"WR={wr:.1%}, n={len(h_rets)}")

    # TP feasibility analysis
    print(f"\nTP FEASIBILITY (can raw MFE cover costs?):")
    print(f"  Market entry + passive TP exit cost: {MARKET_ENTRY_COST + PASSIVE_EXIT_COST:.3f} ticks")
    for tp in [2, 3, 4, 6, 8]:
        hit_rate = np.mean(mfes >= tp)
        print(f"  TP={tp}: {hit_rate:.1%} of signals reach MFE>={tp}t within {args.hold}s")

    # Save results
    output = {
        'threshold': args.threshold,
        'hold_seconds': args.hold,
        'n_signals': len(all_results),
        'n_days': len(matched),
        'mfe_mean': float(np.mean(mfes)),
        'mfe_median': float(np.median(mfes)),
        'mae_mean': float(np.mean(maes)),
        'market_entry_cost': MARKET_ENTRY_COST,
        'verdict': 'EDGE' if np.mean(mfes) > MARKET_ENTRY_COST + PASSIVE_EXIT_COST else 'NO_EDGE',
    }
    out_path = f'/home/jupiter/Lvl3Quant/output/market_order_analysis_t{int(args.threshold*100):02d}.json'
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2)
    print(f"\nSaved to {out_path}")


if __name__ == '__main__':
    main()
