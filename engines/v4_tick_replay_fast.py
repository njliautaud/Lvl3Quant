#!/usr/bin/env python3
"""
V4 Multihead Tick Replay — FAST version
=========================================
Optimized: loads each MBO file ONCE, runs all configs on cached arrays.
Uses PYTHONUNBUFFERED-style flushing for progress monitoring.

Author: Claude
"""

import numpy as np
import os
import sys
import time
import json
import glob
from collections import defaultdict

# Force unbuffered output
sys.stdout = os.fdopen(sys.stdout.fileno(), 'w', buffering=1)

sys.path.insert(0, os.path.dirname(__file__))
from tick_replay_engine import TickReplayEngine, compute_metrics, TICK_VALUE

# =============================================================================
# Config
# =============================================================================

PRED_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds_v2'
MBO_RAW_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/v4_tick_replay_results'

MIN_PREDS_PER_DAY = 1000
HOLD_SECONDS = 30.0
CANCEL_SECONDS = 12.0


def find_matching_dates():
    """Find dates with predictions + raw MBO + sufficient count."""
    matched = []
    for f in sorted(glob.glob(os.path.join(PRED_DIR, 'oot_*.npz'))):
        date = os.path.basename(f).replace('oot_', '').replace('.npz', '')
        mbo_path = os.path.join(MBO_RAW_DIR, f'glbx-mdp3-{date}.mbo.dbn.zst')
        if not os.path.exists(mbo_path):
            continue
        d = np.load(f)
        n = len(d['pred_log_ret_1s'])
        if n < MIN_PREDS_PER_DAY:
            continue
        matched.append((date, mbo_path, f, n))
    return matched


def load_mbo_arrays(mbo_path):
    """Load MBO file into numpy arrays (the expensive step — do once)."""
    import databento as db
    dbn = db.DBNStore.from_file(mbo_path)
    df = dbn.to_df()

    # Find front-month ES
    es_symbols = [s for s in df['symbol'].unique()
                  if s.startswith('ES') and '-' not in s]
    if not es_symbols:
        return None

    best_sym = max(es_symbols,
                   key=lambda s: len(df[(df['symbol'] == s) & (df['action'] == 'T')]))
    df = df[df['symbol'] == best_sym].copy()

    return {
        'ts_event': df['ts_event'].values.astype('int64'),
        'actions': df['action'].values,
        'sides': df['side'].values,
        'prices': df['price'].values.astype('float64'),
        'sizes': df['size'].values.astype('int64'),
        'order_ids': df['order_id'].values.astype('int64'),
        'n_events': len(df),
    }


def align_to_raw(pred_ts_ns, raw_ts_ns):
    """Map prediction timestamps to raw MBO event indices."""
    valid = (pred_ts_ns >= raw_ts_ns[0]) & (pred_ts_ns <= raw_ts_ns[-1])
    if not np.any(valid):
        return np.array([], dtype=np.int64), valid
    indices = np.searchsorted(raw_ts_ns, pred_ts_ns[valid], side='left')
    indices = np.clip(indices, 0, len(raw_ts_ns) - 1)
    return indices, valid


def apply_quantile_filter(signal, quantile):
    """Zero out signals below the top quantile threshold."""
    if quantile is None or quantile <= 0:
        return signal
    abs_sig = np.abs(signal)
    nonzero = abs_sig[abs_sig > 0]
    if len(nonzero) == 0:
        return signal
    threshold = np.quantile(nonzero, 1.0 - quantile)
    filtered = np.zeros_like(signal)
    mask = abs_sig >= threshold
    filtered[mask] = signal[mask]
    return filtered


def run_config_on_arrays(mbo_arrays, pred_indices, signal,
                         tp_ticks, sl_ticks):
    """Run one config on pre-loaded MBO arrays."""
    engine = TickReplayEngine(
        tp_ticks=tp_ticks, sl_ticks=sl_ticks,
        hold_seconds=HOLD_SECONDS,
        signal_threshold=0.0,
        cancel_seconds=CANCEL_SECONDS,
    )
    trades = engine.run_day_from_arrays(
        mbo_arrays['ts_event'], mbo_arrays['actions'],
        mbo_arrays['sides'], mbo_arrays['prices'],
        mbo_arrays['sizes'], mbo_arrays['order_ids'],
        signal, pred_indices=pred_indices,
    )
    return trades


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("V4 MULTIHEAD TICK REPLAY (FAST — load-once)")
    print("=" * 70)

    matched = find_matching_dates()
    print(f"\n{len(matched)} dates with >= {MIN_PREDS_PER_DAY} predictions")

    if not matched:
        print("ERROR: No dates")
        return

    for d, _, _, n in matched:
        print(f"  {d}: {n} preds")

    # Configs to test
    configs = [
        # (tp, sl, quantile, use_composite, label)
        (3, 6, None, False, "all_dir"),
        (3, 6, 0.10, False, "dir_top10%"),
        (3, 6, 0.05, False, "dir_top5%"),
        (3, 6, 0.03, False, "dir_top3%"),
        (3, 6, 0.01, False, "dir_top1%"),
        (3, 6, 0.03, True, "comp_top3%"),
        (2, 4, 0.05, False, "dir_top5%_tight"),
        (4, 8, 0.03, False, "dir_top3%_wide"),
    ]

    # =========================================================================
    # Phase 1: Screening on 3 representative days
    # =========================================================================
    if len(matched) >= 6:
        screen_idx = [0, len(matched)//2, len(matched)-1]
    else:
        screen_idx = list(range(min(3, len(matched))))
    screening = [matched[i] for i in screen_idx]

    print(f"\n{'='*70}")
    print(f"PHASE 1: SCREENING ({len(screening)} days)")
    print(f"{'='*70}")

    # Accumulate results per config across screening days
    config_trades = {label: [] for _, _, _, _, label in configs}

    for date, mbo_path, pred_path, n_preds in screening:
        print(f"\n--- Loading MBO for {date} ---")
        t_load = time.time()
        mbo_arrays = load_mbo_arrays(mbo_path)
        if mbo_arrays is None:
            print(f"  SKIP: no ES contract")
            continue
        print(f"  Loaded {mbo_arrays['n_events']:,} events ({time.time()-t_load:.1f}s)")

        # Load predictions
        pred_data = np.load(pred_path)
        dir_signal = pred_data['pred_log_ret_1s']
        composite = pred_data['composite_signal']
        pred_ts = pred_data['timestamps_ns']

        # Align to raw MBO
        pred_indices, valid_mask = align_to_raw(pred_ts, mbo_arrays['ts_event'])
        n_aligned = len(pred_indices)
        print(f"  Aligned {n_aligned}/{len(pred_ts)} predictions to raw events")

        if n_aligned == 0:
            print(f"  SKIP: no aligned predictions")
            continue

        # Pre-filter signals
        aligned_dir = dir_signal[valid_mask]
        aligned_comp = composite[valid_mask]

        # Run all configs on this day's cached data
        for tp, sl, qt, use_comp, label in configs:
            signal = aligned_comp.copy() if use_comp else aligned_dir.copy()
            signal = apply_quantile_filter(signal, qt)

            t0 = time.time()
            trades = run_config_on_arrays(mbo_arrays, pred_indices, signal, tp, sl)
            elapsed = time.time() - t0

            pnl = sum(t.pnl_ticks for t in trades)
            print(f"  {label:<20s}: {len(trades):>4d} trades, "
                  f"PnL={pnl:>+7.1f}t ({elapsed:.0f}s)")
            config_trades[label].extend(trades)

        # Free MBO memory
        del mbo_arrays

    # Compute Phase 1 metrics
    print(f"\n{'='*70}")
    print(f"PHASE 1 RESULTS")
    print(f"{'='*70}")
    print(f"{'Config':<20s} {'Trades':>6s} {'AvgPnL':>8s} {'WR':>6s} {'MFE':>6s} "
          f"{'MAE':>6s} {'Net':>8s} {'FillLat':>8s}")
    print("-" * 75)

    phase1_results = []
    for tp, sl, qt, use_comp, label in configs:
        trades = config_trades[label]
        if not trades:
            print(f"{label:<20s} {'0':>6s}")
            continue

        metrics = compute_metrics(trades, label)
        metrics['config'] = label
        metrics['tp'] = tp
        metrics['sl'] = sl
        metrics['quantile'] = qt
        metrics['use_composite'] = use_comp
        phase1_results.append(metrics)

        # Avg fill latency
        fill_lats = [t.fill_latency_ns / 1e9 for t in trades if t.fill_latency_ns > 0]
        avg_lat = np.mean(fill_lats) if fill_lats else 0

        print(f"{label:<20s} {metrics['n_trades']:>6d} "
              f"{metrics['avg_pnl_ticks']:>+8.3f} "
              f"{metrics['win_rate']:>5.1%} "
              f"{metrics['avg_mfe']:>+6.2f} "
              f"{metrics['avg_mae']:>+6.2f} "
              f"{metrics['net_pnl_ticks']:>+8.1f} "
              f"{avg_lat:>7.1f}s")

    # Save
    p1_path = os.path.join(OUTPUT_DIR, 'phase1_fast.json')
    with open(p1_path, 'w') as f:
        json.dump(phase1_results, f, indent=2, default=str)

    # =========================================================================
    # Phase 2: Full sweep on all dates if promising
    # =========================================================================
    promising = [r for r in phase1_results
                 if r.get('avg_pnl_ticks', -999) > -0.5 and r.get('n_trades', 0) > 5]

    if promising:
        print(f"\n{'='*70}")
        print(f"PHASE 2: FULL {len(matched)}-DAY SWEEP ({len(promising)} promising configs)")
        print(f"{'='*70}")

        full_trades = {r['config']: [] for r in promising}
        full_daily = {r['config']: [] for r in promising}

        for date, mbo_path, pred_path, n_preds in matched:
            print(f"\n--- {date} ---")
            t_load = time.time()
            mbo_arrays = load_mbo_arrays(mbo_path)
            if mbo_arrays is None:
                continue
            print(f"  {mbo_arrays['n_events']:,} events ({time.time()-t_load:.1f}s)")

            pred_data = np.load(pred_path)
            dir_signal = pred_data['pred_log_ret_1s']
            composite = pred_data['composite_signal']
            pred_ts = pred_data['timestamps_ns']

            pred_indices, valid_mask = align_to_raw(pred_ts, mbo_arrays['ts_event'])
            if len(pred_indices) == 0:
                continue

            aligned_dir = dir_signal[valid_mask]
            aligned_comp = composite[valid_mask]

            for r in promising:
                tp = r['tp']
                sl = r['sl']
                qt = r['quantile']
                use_comp = r['use_composite']
                label = r['config']

                signal = aligned_comp.copy() if use_comp else aligned_dir.copy()
                signal = apply_quantile_filter(signal, qt)

                trades = run_config_on_arrays(mbo_arrays, pred_indices, signal, tp, sl)
                day_pnl = sum(t.pnl_ticks for t in trades)
                full_trades[label].extend(trades)
                full_daily[label].append(day_pnl)

                print(f"  {label:<20s}: {len(trades):>4d} trades, PnL={day_pnl:>+7.1f}t")

            del mbo_arrays

        # Phase 2 results
        print(f"\n{'='*70}")
        print(f"PHASE 2 RESULTS")
        print(f"{'='*70}")

        phase2_results = []
        for r in promising:
            label = r['config']
            trades = full_trades[label]
            daily = np.array(full_daily[label])

            if not trades:
                continue

            metrics = compute_metrics(trades, f"FULL_{label}")
            metrics['config'] = label
            metrics['tp'] = r['tp']
            metrics['sl'] = r['sl']
            metrics['quantile'] = r['quantile']
            metrics['use_composite'] = r['use_composite']
            metrics['n_full_days'] = len(matched)

            if len(daily) > 1 and np.std(daily) > 0:
                metrics['daily_sharpe'] = float(np.mean(daily) / np.std(daily) * np.sqrt(252))
            else:
                metrics['daily_sharpe'] = 0.0
            metrics['green_days'] = int(np.sum(daily > 0))
            metrics['red_days'] = int(np.sum(daily < 0))

            phase2_results.append(metrics)

            print(f"\n  {label}:")
            print(f"    {metrics['n_trades']} trades over {len(daily)} days")
            print(f"    Avg PnL: {metrics['avg_pnl_ticks']:+.3f}t")
            print(f"    WR: {metrics['win_rate']:.1%}, PF: {metrics['profit_factor']:.2f}")
            print(f"    Daily Sharpe: {metrics['daily_sharpe']:.2f}")
            print(f"    Green/Red: {metrics['green_days']}/{metrics['red_days']}")
            print(f"    Net: {metrics['net_pnl_ticks']:+.1f}t (${metrics['net_pnl_dollars']:+.0f})")

        p2_path = os.path.join(OUTPUT_DIR, 'phase2_fast.json')
        with open(p2_path, 'w') as f:
            json.dump(phase2_results, f, indent=2, default=str)
    else:
        print(f"\n{'='*70}")
        print("PHASE 2 SKIPPED — no configs with avg > -0.5t")
        if phase1_results:
            best = max(phase1_results, key=lambda r: r.get('avg_pnl_ticks', -999))
            print(f"Best: {best['config']} avg={best.get('avg_pnl_ticks',-999):+.3f}t")
        print(f"{'='*70}")

    print(f"\nDone. Results in {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
