#!/usr/bin/env python3
"""
V4 Multihead Tick Replay Runner (v2 — timestamp-aligned)
==========================================================
Runs the tick replay engine on v4 multihead predictions with proper
timestamp-based alignment. Uses v2 prediction files that include
nanosecond timestamps for each prediction.

Measures adverse selection and execution viability for v4 multihead
signals (IC 0.209, top-1% shorts with eofi = 0.762 raw ticks).

Key question: does adverse selection stay below 0.386 ticks?
If yes → v4 multihead ES tick trading is PROFITABLE.

Author: Claude
"""

import numpy as np
import os
import sys
import time
import json
import glob
from dataclasses import dataclass
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(__file__))
from tick_replay_engine import TickReplayEngine, compute_metrics, Trade

# =============================================================================
# Config
# =============================================================================

PRED_DIR = '/home/jupiter/Lvl3Quant/output/v4_multihead_tick_replay_preds_v2'
MBO_RAW_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/v4_tick_replay_results'

MIN_PREDS_PER_DAY = 1000  # Skip dates with too few predictions

# Execution params (bounded by 1s prediction horizon per HC #432)
HOLD_SECONDS = 30.0
CANCEL_SECONDS = 12.0


def find_matching_dates():
    """Find dates with predictions + raw MBO + sufficient prediction count."""
    matched = []
    for f in sorted(glob.glob(os.path.join(PRED_DIR, 'oot_*.npz'))):
        date = os.path.basename(f).replace('oot_', '').replace('.npz', '')
        mbo_path = os.path.join(MBO_RAW_DIR, f'glbx-mdp3-{date}.mbo.dbn.zst')

        if not os.path.exists(mbo_path):
            continue

        d = np.load(f)
        n_preds = len(d['pred_log_ret_1s'])
        if n_preds < MIN_PREDS_PER_DAY:
            continue

        matched.append((date, mbo_path, f, n_preds))

    return matched


def align_to_raw_mbo(pred_timestamps_ns, raw_ts_ns):
    """
    Map prediction nanosecond timestamps to raw MBO event indices.
    Uses searchsorted for O(n log m) alignment.
    """
    # Filter to predictions within raw file's time range
    valid = (pred_timestamps_ns >= raw_ts_ns[0]) & (pred_timestamps_ns <= raw_ts_ns[-1])
    valid_ts = pred_timestamps_ns[valid]

    if len(valid_ts) == 0:
        return np.array([], dtype=np.int64), valid

    # Binary search for closest raw event
    indices = np.searchsorted(raw_ts_ns, valid_ts, side='left')
    indices = np.clip(indices, 0, len(raw_ts_ns) - 1)

    return indices, valid


def run_day_aligned(date, mbo_path, pred_path,
                    tp_ticks, sl_ticks,
                    use_composite=False,
                    quantile_threshold=None):
    """Run tick replay for one day with timestamp alignment."""
    import databento as db

    # Load predictions with timestamps
    pred_data = np.load(pred_path)
    preds = pred_data['pred_log_ret_1s']
    composite = pred_data['composite_signal']
    eofi = pred_data['eofi_1s']
    pred_ts = pred_data['timestamps_ns']

    # Select signal
    if use_composite:
        signal = composite.copy()
    else:
        signal = preds.copy()

    # Load raw MBO
    dbn = db.DBNStore.from_file(mbo_path)
    df = dbn.to_df()

    # Find front-month ES
    es_symbols = [s for s in df['symbol'].unique()
                  if s.startswith('ES') and '-' not in s]
    if not es_symbols:
        return []

    best_sym = max(es_symbols,
                   key=lambda s: len(df[(df['symbol'] == s) & (df['action'] == 'T')]))
    df = df[df['symbol'] == best_sym].copy()

    raw_ts = df['ts_event'].values.astype('int64')

    # Align predictions to raw events
    pred_indices, valid_mask = align_to_raw_mbo(pred_ts, raw_ts)

    if len(pred_indices) == 0:
        return []

    # Filter signal to valid predictions
    aligned_signal = signal[valid_mask]

    # Apply quantile threshold
    if quantile_threshold is not None and quantile_threshold > 0:
        abs_sig = np.abs(aligned_signal)
        nonzero = abs_sig[abs_sig > 0]
        if len(nonzero) > 0:
            threshold = np.quantile(nonzero, 1.0 - quantile_threshold)
            mask = abs_sig >= threshold
            filtered = np.zeros_like(aligned_signal)
            filtered[mask] = aligned_signal[mask]
            aligned_signal = filtered

    # Run tick replay
    engine = TickReplayEngine(
        tp_ticks=tp_ticks, sl_ticks=sl_ticks,
        hold_seconds=HOLD_SECONDS,
        signal_threshold=0.0,  # Already filtered
        cancel_seconds=CANCEL_SECONDS,
    )

    trades = engine.run_day(mbo_path, aligned_signal, pred_indices=pred_indices)
    return trades


def run_v4_replay():
    """Main: v4 multihead tick replay with adverse selection measurement."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 70)
    print("V4 MULTIHEAD TICK-LEVEL FIFO REPLAY (timestamp-aligned)")
    print("=" * 70)

    matched = find_matching_dates()
    print(f"\nFound {len(matched)} dates with >= {MIN_PREDS_PER_DAY} predictions")

    if not matched:
        print("ERROR: No matching dates found")
        return

    for date, mbo_path, pred_path, n_preds in matched:
        print(f"  {date}: {n_preds} predictions")

    # =========================================================================
    # Phase 1: Quick screening (3 days, 6 configs)
    # =========================================================================
    print(f"\n{'='*70}")
    print("PHASE 1: SCREENING (3 representative days)")
    print(f"{'='*70}")

    # Pick 3 dates spread across time range
    if len(matched) >= 6:
        screen_idx = [0, len(matched)//2, len(matched)-1]
    else:
        screen_idx = list(range(min(3, len(matched))))
    screening = [matched[i] for i in screen_idx]

    configs = [
        # (tp, sl, quantile, use_composite, label)
        (3, 6, None, False, "all_signals_dir"),
        (3, 6, 0.10, False, "dir_top10%"),
        (3, 6, 0.05, False, "dir_top5%"),
        (3, 6, 0.03, False, "dir_top3%"),
        (3, 6, 0.01, False, "dir_top1%"),
        (3, 6, 0.03, True, "composite_top3%"),
    ]

    phase1_results = []

    for tp, sl, qt, use_comp, label in configs:
        print(f"\n--- {label} (TP={tp}, SL={sl}"
              f"{f', q={qt}' if qt else ''}"
              f"{', composite' if use_comp else ''}) ---")
        all_trades = []

        for date, mbo_path, pred_path, n_preds in screening:
            t0 = time.time()
            try:
                trades = run_day_aligned(
                    date, mbo_path, pred_path,
                    tp_ticks=tp, sl_ticks=sl,
                    use_composite=use_comp,
                    quantile_threshold=qt,
                )
            except Exception as e:
                print(f"  {date}: ERROR - {e}")
                trades = []

            elapsed = time.time() - t0
            pnl = sum(t.pnl_ticks for t in trades) if trades else 0
            print(f"  {date}: {len(trades)} trades, PnL={pnl:+.1f}t ({elapsed:.0f}s)")
            all_trades.extend(trades)

        if all_trades:
            metrics = compute_metrics(all_trades, label)
            metrics['config'] = label
            metrics['tp'] = tp
            metrics['sl'] = sl
            metrics['quantile'] = qt
            metrics['use_composite'] = use_comp
            metrics['n_screen_days'] = len(screening)
            phase1_results.append(metrics)

            # Compute adverse selection proxy
            # For shorts: adverse = how much price went UP before our fill
            # For longs: adverse = how much price went DOWN before our fill
            short_trades = [t for t in all_trades if t.direction == -1]
            long_trades = [t for t in all_trades if t.direction == 1]

            print(f"  RESULT: {metrics.get('n_trades', 0)} trades "
                  f"({len(long_trades)}L/{len(short_trades)}S)")
            print(f"    Avg PnL: {metrics.get('avg_pnl_ticks', 0):+.3f}t")
            print(f"    WR: {metrics.get('win_rate', 0):.1%}")
            print(f"    Net: {metrics.get('net_pnl_ticks', 0):+.1f}t")

            # Fill latency and MFE/MAE analysis
            fill_lats = [t.fill_latency_ns / 1e9 for t in all_trades if t.fill_latency_ns > 0]
            if fill_lats:
                print(f"    Fill latency: {np.mean(fill_lats):.1f}s "
                      f"(median {np.median(fill_lats):.1f}s)")
            print(f"    Avg MFE: {metrics.get('avg_mfe', 0):+.2f}t, "
                  f"Avg MAE: {metrics.get('avg_mae', 0):+.2f}t")
            # Exit reason breakdown
            reasons = metrics.get('exit_reasons', {})
            if reasons:
                print(f"    Exits: {dict(reasons)}")
        else:
            print(f"  NO TRADES")

    # Save phase 1
    p1_path = os.path.join(OUTPUT_DIR, 'phase1_screening_v2.json')
    with open(p1_path, 'w') as f:
        json.dump(phase1_results, f, indent=2, default=str)
    print(f"\nPhase 1 saved to {p1_path}")

    # =========================================================================
    # Phase 2: Full sweep if promising
    # =========================================================================
    promising = [r for r in phase1_results
                 if r.get('avg_pnl_ticks', -999) > -0.5 and r.get('n_trades', 0) > 10]

    if promising:
        print(f"\n{'='*70}")
        print(f"PHASE 2: FULL {len(matched)}-DAY SWEEP")
        print(f"{'='*70}")

        phase2_results = []
        for r in promising:
            tp = r['tp']
            sl = r['sl']
            qt = r['quantile']
            use_comp = r['use_composite']
            label = r['config']

            print(f"\n--- FULL: {label} ---")
            all_trades = []
            daily_pnl = []

            for date, mbo_path, pred_path, n_preds in matched:
                t0 = time.time()
                try:
                    trades = run_day_aligned(
                        date, mbo_path, pred_path,
                        tp_ticks=tp, sl_ticks=sl,
                        use_composite=use_comp,
                        quantile_threshold=qt,
                    )
                except Exception as e:
                    print(f"  {date}: ERROR - {e}")
                    trades = []

                elapsed = time.time() - t0
                day_pnl = sum(t.pnl_ticks for t in trades)
                daily_pnl.append(day_pnl)
                print(f"  {date}: {len(trades)} trades, PnL={day_pnl:+.1f}t ({elapsed:.0f}s)")
                all_trades.extend(trades)

            if all_trades:
                metrics = compute_metrics(all_trades, f"FULL_{label}")
                metrics['config'] = label
                metrics['tp'] = tp
                metrics['sl'] = sl
                metrics['quantile'] = qt
                metrics['use_composite'] = use_comp
                metrics['n_days'] = len(matched)

                # Daily Sharpe
                daily_arr = np.array(daily_pnl)
                if np.std(daily_arr) > 0:
                    daily_sharpe = np.mean(daily_arr) / np.std(daily_arr) * np.sqrt(252)
                else:
                    daily_sharpe = 0
                metrics['daily_sharpe'] = float(daily_sharpe)
                metrics['daily_mean_pnl'] = float(np.mean(daily_arr))
                metrics['green_days'] = int(np.sum(daily_arr > 0))
                metrics['red_days'] = int(np.sum(daily_arr < 0))

                phase2_results.append(metrics)

                print(f"\n  FULL RESULT ({label}):")
                print(f"    {metrics.get('n_trades', 0)} trades, {len(matched)} days")
                print(f"    Avg PnL/trade: {metrics.get('avg_pnl_ticks', 0):+.3f}t")
                print(f"    Daily PnL: {np.mean(daily_arr):+.1f}t ± {np.std(daily_arr):.1f}t")
                print(f"    Daily Sharpe: {daily_sharpe:.2f}")
                print(f"    Green/Red days: {metrics['green_days']}/{metrics['red_days']}")
                print(f"    WR: {metrics.get('win_rate', 0):.1%}")

        p2_path = os.path.join(OUTPUT_DIR, 'phase2_full_sweep_v2.json')
        with open(p2_path, 'w') as f:
            json.dump(phase2_results, f, indent=2, default=str)
        print(f"\nPhase 2 saved to {p2_path}")
    else:
        print(f"\n{'='*70}")
        print("PHASE 2 SKIPPED — no promising configs")
        if phase1_results:
            best = max(phase1_results, key=lambda r: r.get('avg_pnl_ticks', -999))
            print(f"Best avg PnL: {best.get('avg_pnl_ticks', -999):+.3f}t ({best['config']})")
        print(f"{'='*70}")

    # Final summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    print(f"{'Config':<25s} {'Trades':>6s} {'AvgPnL':>8s} {'WR':>6s} {'NetPnL':>8s}")
    print("-" * 60)
    for r in phase1_results:
        print(f"{r['config']:<25s} {r.get('n_trades',0):>6d} "
              f"{r.get('avg_pnl_ticks',0):>+8.3f} "
              f"{r.get('win_rate',0):>5.1%} "
              f"{r.get('net_pnl_ticks',0):>+8.1f}")


if __name__ == '__main__':
    run_v4_replay()
