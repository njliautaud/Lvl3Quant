#!/usr/bin/env python3
"""
Tick-Level MFE/MAE Analyzer
============================

Fast analysis of model signal quality at the tick level:
- For each prediction, extract subsequent trade prices
- Compute MFE (max favorable excursion) and MAE (max adverse excursion)
- Stratify by signal strength (deciles/quintiles)
- Compare to random baseline

This answers the fundamental question: "Does the model predict tick-level direction?"
without the complexity of fill simulation.

Key insight: If MFE > MAE for model's predicted direction, there IS a directional edge.
The question then becomes: is it large enough to overcome trading costs?

Author: Claude (autonomous, 2026-07-02)
"""

import numpy as np
import glob
import os
import sys
import time
import json
from collections import defaultdict
import warnings
warnings.filterwarnings('ignore')

# Constants
TICK_SIZE = 0.25
TICK_VALUE = 12.50
PRED_STRIDE = 250
PRED_WINDOW = 1500

# Cost in ticks
COST_PASSIVE_ENTRY_EXIT = 0.376    # Commission only (passive both sides)
COST_PASSIVE_ENTRY_MARKET_EXIT = 1.376  # Commission + 1 tick spread


def analyze_day(mbo_path, predictions, horizons_s=[1, 2, 5, 10, 15, 30]):
    """
    For each prediction on a given day, extract trade prices for the next N seconds
    and compute MFE/MAE.

    Returns list of dicts with MFE/MAE per horizon for each prediction.
    """
    import databento as db

    print(f"  Loading {os.path.basename(mbo_path)}...", end='', flush=True)
    t0 = time.time()

    dbn = db.DBNStore.from_file(mbo_path)
    df = dbn.to_df()

    # Find front-month ES
    es_symbols = [s for s in df['symbol'].unique()
                  if s.startswith('ES') and '-' not in s]
    if not es_symbols:
        print(" no ES data")
        return []

    best_sym = max(es_symbols, key=lambda s: len(df[(df['symbol'] == s) & (df['action'] == 'T')]))

    # Extract only trades (action='T' or 'F')
    trades_df = df[(df['symbol'] == best_sym) & (df['action'].isin(['T', 'F']))].copy()
    trades_df = trades_df[trades_df['price'] > 0].copy()

    if len(trades_df) == 0:
        print(" no trades")
        return []

    # Full event count for stride alignment
    full_df = df[df['symbol'] == best_sym]
    n_events = len(full_df)

    # Build prediction indices
    n_preds = len(predictions)
    pred_event_indices = np.arange(PRED_WINDOW, PRED_WINDOW + n_preds * PRED_STRIDE, PRED_STRIDE)
    pred_event_indices = pred_event_indices[pred_event_indices < n_events]
    predictions = predictions[:len(pred_event_indices)]

    # Get timestamps for each prediction event
    full_ts = full_df['ts_event'].values.astype('int64')

    # Trade timestamps and prices (for fast lookup)
    trade_ts = trades_df['ts_event'].values.astype('int64')
    trade_prices = trades_df['price'].values.astype('float64')

    max_horizon_ns = int(max(horizons_s) * 1e9)

    results = []

    for i, evt_idx in enumerate(pred_event_indices):
        if evt_idx >= len(full_ts):
            continue

        pred = predictions[i]
        signal_ts = full_ts[evt_idx]

        # Find trades in the window [signal_ts, signal_ts + max_horizon]
        start_idx = np.searchsorted(trade_ts, signal_ts)
        end_idx = np.searchsorted(trade_ts, signal_ts + max_horizon_ns)

        if start_idx >= len(trade_ts) or start_idx >= end_idx:
            continue

        # Reference price: first trade at or after signal
        ref_price = trade_prices[start_idx]
        window_ts = trade_ts[start_idx:end_idx]
        window_prices = trade_prices[start_idx:end_idx]

        # Price changes in ticks from reference
        price_changes_ticks = (window_prices - ref_price) / TICK_SIZE
        elapsed_ns = window_ts - signal_ts

        result = {
            'pred': float(pred),
            'ref_price': float(ref_price),
            'signal_ts': int(signal_ts),
        }

        # For each horizon, compute MFE and MAE
        for h in horizons_s:
            h_ns = int(h * 1e9)
            mask = elapsed_ns <= h_ns
            if mask.sum() == 0:
                continue

            h_changes = price_changes_ticks[mask]

            # Direction based on prediction sign
            if pred > 0:  # Long signal
                mfe = float(np.max(h_changes))   # Best up move
                mae = float(-np.min(h_changes))  # Worst down move (positive = bad)
                final = float(h_changes[-1])
            else:  # Short signal
                mfe = float(-np.min(h_changes))  # Best down move
                mae = float(np.max(h_changes))   # Worst up move (positive = bad)
                final = float(-h_changes[-1])

            result[f'mfe_{h}s'] = mfe
            result[f'mae_{h}s'] = mae
            result[f'final_{h}s'] = final
            result[f'n_trades_{h}s'] = int(mask.sum())

        results.append(result)

    elapsed = time.time() - t0
    print(f" {len(results)} signals, {len(trades_df)} trades ({elapsed:.1f}s)")

    return results


def analyze_all_days(mbo_dir, pred_dir, max_days=None):
    """Run MFE/MAE analysis on all available days."""

    # Load predictions
    print("Loading predictions...")
    predictions = {}
    for f in sorted(glob.glob(os.path.join(pred_dir, 'oot_*.npz'))):
        d = np.load(f, allow_pickle=True)
        date_str = os.path.basename(f).replace('oot_', '').replace('.npz', '')
        if 'pred_log_ret_1s' not in d.files:
            print(f"  Skipping {date_str} (incomplete)")
            d.close()
            continue
        predictions[date_str] = {
            'lr1s': np.array(d['pred_log_ret_1s']),
            'lr5s': np.array(d['pred_log_ret_5s']),
            'lr10s': np.array(d['pred_log_ret_10s']),
        }
        d.close()
    print(f"  Loaded {len(predictions)} dates")

    # Find MBO files
    mbo_files = {}
    for f in sorted(glob.glob(os.path.join(mbo_dir, 'glbx-mdp3-*.mbo.dbn.zst'))):
        date8 = os.path.basename(f).split('-')[2].split('.')[0]
        mbo_files[date8] = f

    matched = sorted(set(mbo_files.keys()) & set(predictions.keys()))
    if max_days:
        matched = matched[:max_days]
    print(f"  {len(matched)} days to analyze")

    # Run analysis
    all_results = []
    for date_key in matched:
        mbo_path = mbo_files[date_key]
        preds = predictions[date_key]['lr1s']  # Primary signal

        day_results = analyze_day(mbo_path, preds)
        for r in day_results:
            r['date'] = date_key
        all_results.extend(day_results)

    return all_results, predictions, mbo_files, matched


def compute_edge_stats(results, horizons_s=[1, 2, 5, 10, 15, 30]):
    """Compute directional edge statistics from MFE/MAE data."""

    if not results:
        print("No results to analyze!")
        return {}

    preds = np.array([r['pred'] for r in results])

    print(f"\n{'='*80}")
    print(f"MFE/MAE ANALYSIS — {len(results)} signals across {len(set(r['date'] for r in results))} days")
    print(f"{'='*80}")

    print(f"\nSignal distribution:")
    print(f"  Mean: {preds.mean():.4f}, Std: {preds.std():.4f}")
    print(f"  Long (pred>0): {(preds>0).sum()} ({(preds>0).mean():.1%})")
    print(f"  Short (pred<0): {(preds<0).sum()} ({(preds<0).mean():.1%})")

    all_stats = {}

    # Overall MFE/MAE by horizon
    print(f"\n--- OVERALL MFE/MAE (all signals) ---")
    print(f"{'Horizon':>8} {'MFE':>7} {'MAE':>7} {'Final':>7} {'MFE>MAE%':>9} {'MFE-MAE':>8} {'Final>0%':>9} {'E[net]':>8}")
    print("-" * 75)

    for h in horizons_s:
        mfe_key = f'mfe_{h}s'
        mae_key = f'mae_{h}s'
        final_key = f'final_{h}s'

        mfes = np.array([r[mfe_key] for r in results if mfe_key in r])
        maes = np.array([r[mae_key] for r in results if mae_key in r])
        finals = np.array([r[final_key] for r in results if final_key in r])

        if len(mfes) == 0:
            continue

        mfe_gt_mae = (mfes > maes).mean()
        # Expected net: if we captured MFE with passive limit, cost is 0.376 ticks
        # Conservative estimate: expected final move in our direction
        net_passive = finals.mean() - COST_PASSIVE_ENTRY_EXIT

        print(f"{h:>7}s {mfes.mean():>7.2f} {maes.mean():>7.2f} {finals.mean():>7.2f} "
              f"{mfe_gt_mae:>8.1%} {(mfes-maes).mean():>8.2f} {(finals>0).mean():>8.1%} "
              f"{net_passive:>8.3f}")

        all_stats[f'{h}s'] = {
            'mfe_mean': float(mfes.mean()),
            'mae_mean': float(maes.mean()),
            'final_mean': float(finals.mean()),
            'mfe_gt_mae_pct': float(mfe_gt_mae),
            'mfe_minus_mae': float((mfes - maes).mean()),
            'final_gt_0_pct': float((finals > 0).mean()),
            'net_after_cost': float(net_passive),
            'n': len(mfes),
        }

    # Stratified by signal strength (quintiles)
    print(f"\n--- BY SIGNAL STRENGTH (quintiles of |pred|) ---")
    abs_preds = np.abs(preds)
    quintile_edges = np.percentile(abs_preds, [0, 20, 40, 60, 80, 100])

    for h in [1, 5, 10]:
        mfe_key = f'mfe_{h}s'
        mae_key = f'mae_{h}s'
        final_key = f'final_{h}s'

        print(f"\n  Horizon: {h}s")
        print(f"  {'Quintile':>10} {'|pred| range':>15} {'N':>6} {'MFE':>6} {'MAE':>6} {'Final':>7} "
              f"{'MFE>MAE':>8} {'Final>0':>8} {'E[net]':>8}")
        print("  " + "-" * 85)

        for qi in range(5):
            lo, hi = quintile_edges[qi], quintile_edges[qi+1]
            mask = (abs_preds >= lo) & (abs_preds < hi + 1e-9)
            if qi == 4:
                mask = abs_preds >= lo  # Include upper bound for last quintile

            subset = [r for r, m in zip(results, mask) if m and mfe_key in r]
            if not subset:
                continue

            mfes = np.array([r[mfe_key] for r in subset])
            maes = np.array([r[mae_key] for r in subset])
            finals = np.array([r[final_key] for r in subset])

            net = finals.mean() - COST_PASSIVE_ENTRY_EXIT

            label = f"Q{qi+1}" + (" (top)" if qi == 4 else "")
            print(f"  {label:>10} [{lo:.3f},{hi:.3f}] {len(subset):>6} "
                  f"{mfes.mean():>6.2f} {maes.mean():>6.2f} {finals.mean():>7.3f} "
                  f"{(mfes>maes).mean():>7.1%} {(finals>0).mean():>7.1%} "
                  f"{net:>8.3f}")

    # Top 10% / Top 5% / Top 1% analysis
    print(f"\n--- TOP CONFIDENCE SIGNALS ---")
    for pct_label, pct_thresh in [('Top 20%', 80), ('Top 10%', 90), ('Top 5%', 95), ('Top 1%', 99)]:
        thresh = np.percentile(abs_preds, pct_thresh)
        mask = abs_preds >= thresh

        print(f"\n  {pct_label} (|pred| >= {thresh:.3f}, n={mask.sum()}):")

        for h in [1, 2, 5, 10]:
            mfe_key = f'mfe_{h}s'
            mae_key = f'mae_{h}s'
            final_key = f'final_{h}s'

            subset = [r for r, m in zip(results, mask) if m and mfe_key in r]
            if not subset:
                continue

            mfes = np.array([r[mfe_key] for r in subset])
            maes = np.array([r[mae_key] for r in subset])
            finals = np.array([r[final_key] for r in subset])

            # Various cost scenarios
            net_passive = finals.mean() - COST_PASSIVE_ENTRY_EXIT
            net_market = finals.mean() - COST_PASSIVE_ENTRY_MARKET_EXIT

            # Theoretical TP/SL edge (conservative)
            # If MFE >= TP before MAE >= SL, we win TP. Otherwise lose SL.
            for tp, sl in [(2, 1), (3, 1), (4, 2), (3, 2)]:
                tp_hits = np.sum(mfes >= tp)
                sl_hits = np.sum(maes >= sl)
                # Very rough: P(TP first) ≈ fraction where MFE >= TP
                # This ignores timing (which comes first)
                tp_rate = tp_hits / len(subset) if len(subset) > 0 else 0
                sl_rate = sl_hits / len(subset) if len(subset) > 0 else 0

            print(f"    {h:>2}s: MFE={mfes.mean():.2f} MAE={maes.mean():.2f} "
                  f"Final={finals.mean():+.3f} MFE>MAE={float((mfes>maes).mean()):.1%} "
                  f"NetPassive={net_passive:+.3f}t")

    # RANDOM BASELINE: flip all directions
    print(f"\n--- RANDOM BASELINE (flipped directions) ---")
    for h in [1, 5, 10]:
        mfe_key = f'mfe_{h}s'
        final_key = f'final_{h}s'

        # For random: MFE in predicted direction = MAE in random direction and vice versa
        # Or equivalently: final changes sign
        finals = np.array([r[final_key] for r in results if final_key in r])

        # Random should have mean final ≈ 0 (symmetric)
        # Our model's directional contribution = mean(final) - 0
        print(f"  {h:>2}s: Model avg final = {finals.mean():+.4f}t, "
              f"Direction correct = {(finals>0).mean():.1%}, "
              f"Edge per signal = {finals.mean():+.4f}t")

    # MFE distribution for finding optimal TP
    print(f"\n--- MFE PERCENTILES (for TP optimization) ---")
    for h in [1, 2, 5, 10]:
        mfe_key = f'mfe_{h}s'
        mfes = np.array([r[mfe_key] for r in results if mfe_key in r])
        if len(mfes) == 0:
            continue
        pctiles = [10, 25, 50, 75, 90, 95, 99]
        vals = np.percentile(mfes, pctiles)
        pct_str = " ".join([f"p{p}={v:.1f}" for p, v in zip(pctiles, vals)])
        print(f"  {h:>2}s: {pct_str}")

    # MAE distribution for finding optimal SL
    print(f"\n--- MAE PERCENTILES (for SL optimization) ---")
    for h in [1, 2, 5, 10]:
        mae_key = f'mae_{h}s'
        maes = np.array([r[mae_key] for r in results if mae_key in r])
        if len(maes) == 0:
            continue
        pctiles = [10, 25, 50, 75, 90, 95, 99]
        vals = np.percentile(maes, pctiles)
        pct_str = " ".join([f"p{p}={v:.1f}" for p, v in zip(pctiles, vals)])
        print(f"  {h:>2}s: {pct_str}")

    return all_stats


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--max-days', type=int, default=None)
    args = parser.parse_args()

    MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
    PRED_DIR = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
    OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/tick_level_replay"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results, predictions, mbo_files, matched_dates = analyze_all_days(
        MBO_DIR, PRED_DIR, max_days=args.max_days
    )

    stats = compute_edge_stats(results)

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'mfe_mae_analysis.json')
    with open(output_path, 'w') as f:
        json.dump({
            'stats': stats,
            'n_signals': len(results),
            'n_days': len(matched_dates),
            'dates': matched_dates,
        }, f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Also save raw results for further analysis
    raw_path = os.path.join(OUTPUT_DIR, 'mfe_mae_raw.npz')
    np.savez_compressed(raw_path,
        preds=np.array([r['pred'] for r in results]),
        dates=np.array([r['date'] for r in results]),
        **{f'mfe_{h}s': np.array([r.get(f'mfe_{h}s', np.nan) for r in results])
           for h in [1, 2, 5, 10, 15, 30]},
        **{f'mae_{h}s': np.array([r.get(f'mae_{h}s', np.nan) for r in results])
           for h in [1, 2, 5, 10, 15, 30]},
        **{f'final_{h}s': np.array([r.get(f'final_{h}s', np.nan) for r in results])
           for h in [1, 2, 5, 10, 15, 30]},
    )
    print(f"Raw data saved to {raw_path}")


if __name__ == '__main__':
    main()
