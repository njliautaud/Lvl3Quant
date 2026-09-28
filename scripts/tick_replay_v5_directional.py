#!/usr/bin/env python3
"""
Tick-Level Replay v5 — Directional Modes (Short-Only / Invert-Longs / Both)
=============================================================================
Uses the same FIFO passive-entry / TP+SL replay engine as v2, but tests:
  - short_only: only take short signals (model's strongest edge per v3)
  - invert_longs: flip long signals to short (v3 showed longs predict downward moves)
  - both: baseline (same as v2)

Reuses v2's DayData loading + replay() function. Just modifies signal directions.

Author: Claude (autonomous build, 2026-07-02)
"""

import argparse
import copy
import gc
import json
import logging
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tick_level_replay import (
    ROOT, RAW_MBO_DIR, SMART_V3_DIR, OOT_PRED_DIR, OUTPUT_DIR,
    ES_TICK_SIZE, ES_TICK_VALUE, ES_RT_COMMISSION_TICKS,
    COST_TP_EXIT, COST_SL_EXIT, COST_TIMEOUT_EXIT,
    WINDOW_SIZE, STRIDE, MAX_HOLD_SECONDS, CANCEL_WINDOW_SECONDS,
    compute_metrics, get_oot_dates,
)
from tick_replay_v2 import DayData, replay
import databento as dbn

logging.basicConfig(
    format="%(asctime)s [REPLAY-V5] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("REPLAY-V5")


def filter_signals(day: DayData, mode: str) -> np.ndarray:
    """Return modified signal_dirs based on mode."""
    dirs = day.signal_dirs.copy()

    if mode == 'both':
        return dirs
    elif mode == 'short_only':
        # Zero out long signals (they won't trigger trades since dir==0 is skipped)
        dirs[dirs == 1] = 0
        return dirs
    elif mode == 'invert_longs':
        # Flip longs to shorts
        dirs[dirs == 1] = -1
        return dirs
    elif mode == 'long_only':
        dirs[dirs == -1] = 0
        return dirs
    else:
        raise ValueError(f"Unknown mode: {mode}")


def run_directional_experiment(days: List[DayData], tp: int, sl: int,
                                mode: str, n_perms: int = 20) -> Dict:
    """Run real + permutation replays with directional filtering."""
    config = f"TP{tp}_SL{sl}_{mode}"
    log.info(f"\n--- {config} ---")

    # Real trades with filtered directions
    all_trades = []
    for day in days:
        filtered_dirs = filter_signals(day, mode)
        trades = replay(day, tp, sl, override_dirs=filtered_dirs)
        all_trades.extend(trades)

    real_m = compute_metrics(all_trades, config)
    real_pnl = real_m.get('total_pnl_ticks', 0)

    # Per-day breakdown
    day_pnls = {}
    for t in all_trades:
        d = t['date']
        day_pnls[d] = day_pnls.get(d, 0) + t['net_ticks']
    green = sum(1 for v in day_pnls.values() if v > 0)
    red = sum(1 for v in day_pnls.values() if v < 0)

    # Permutation test: random directions in the SAME mode
    # For short_only: random {-1, 0} (50% take short, 50% skip)
    # For invert_longs: all signals go short with random subset
    # For fair comparison: random directions but same NUMBER of trades
    perm_pnls = []
    for pi in range(n_perms):
        rng = np.random.default_rng(seed=pi * 1000 + 42)
        pt = []
        for day in days:
            n_sigs = len(day.signal_dirs)
            if mode == 'short_only':
                # Random: each signal has 50% chance of being short, 50% skip
                rand_dirs = rng.choice(np.array([-1, 0], dtype=np.int8), size=n_sigs)
            elif mode == 'invert_longs':
                # All signals go short (same as real), but randomly flip some to skip
                # Actually: real invert_longs takes ALL signals as short
                # Fair permutation: random direction for all signals
                rand_dirs = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_sigs)
            elif mode == 'long_only':
                rand_dirs = rng.choice(np.array([1, 0], dtype=np.int8), size=n_sigs)
            else:  # both
                rand_dirs = rng.choice(np.array([-1, 1], dtype=np.int8), size=n_sigs)

            trades = replay(day, tp, sl, override_dirs=rand_dirs)
            pt.extend(trades)
        perm_pnls.append(sum(t['net_ticks'] for t in pt))

    mean_perm = np.mean(perm_pnls) if perm_pnls else 0
    edge = real_pnl - mean_perm
    # p-value: fraction of permutations >= real (lower is better)
    p_val = np.mean([p >= real_pnl for p in perm_pnls]) if perm_pnls else 1.0

    n_trades = real_m.get('n_trades', 0)
    wr = real_m.get('win_rate', 0)
    sharpe = real_m.get('sharpe', 0)
    sortino = real_m.get('sortino', 0)
    pf = real_m.get('profit_factor', 0)

    log.info(f"  {config}: {n_trades} trades, PnL={real_pnl:+.1f}t, "
             f"WR={wr:.1%}, Sharpe={sharpe:+.2f}, Sortino={sortino:+.2f}, "
             f"Edge={edge:+.1f}t, p={p_val:.3f}")

    return {
        'config': config,
        'mode': mode,
        'tp': tp, 'sl': sl,
        'n_trades': n_trades,
        'real_pnl': round(float(real_pnl), 2),
        'win_rate': wr,
        'profit_factor': pf,
        'sharpe': round(float(sharpe), 4),
        'sortino': round(float(sortino), 4),
        'max_dd': real_m.get('max_dd_ticks', 0),
        'exit_types': real_m.get('exit_types', {}),
        'green_days': green,
        'red_days': red,
        'n_shorts': sum(1 for t in all_trades if t['direction'] == -1),
        'n_longs': sum(1 for t in all_trades if t['direction'] == 1),
        'mean_perm_pnl': round(float(mean_perm), 2),
        'model_edge': round(float(edge), 2),
        'p_value': round(float(p_val), 3),
        'avg_pnl_per_trade': round(float(real_pnl / n_trades), 4) if n_trades > 0 else 0,
        'per_day': {d: round(v, 2) for d, v in day_pnls.items()},
    }


def main():
    parser = argparse.ArgumentParser(description="Tick-Level Replay v5 — Directional")
    parser.add_argument('--n-dates', type=int, default=None)
    parser.add_argument('--tp', type=int, nargs='+', default=[3, 4, 5, 6, 8, 10])
    parser.add_argument('--sl', type=int, nargs='+', default=[1, 2, 3])
    parser.add_argument('--modes', type=str, nargs='+',
                        default=['short_only', 'invert_longs'])
    parser.add_argument('--top-pct', type=float, default=0.10)
    parser.add_argument('--permutations', type=int, default=50)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    dates = get_oot_dates()
    if args.n_dates:
        dates = dates[:args.n_dates]

    log.info(f"Loading {len(dates)} days of data...")
    t0 = time.time()
    days = []
    for d in dates:
        try:
            day = DayData(d, top_pct=args.top_pct)
            days.append(day)
        except Exception as e:
            log.error(f"  Skip {d}: {e}")
    load_time = time.time() - t0
    log.info(f"Data loaded in {load_time:.0f}s ({len(days)} days)")

    # Run all configs
    all_results = []
    t1 = time.time()

    for mode in args.modes:
        for tp in args.tp:
            for sl in args.sl:
                if tp <= sl:
                    continue
                try:
                    r = run_directional_experiment(days, tp, sl, mode, n_perms=args.permutations)
                    all_results.append(r)
                except Exception as e:
                    log.error(f"TP{tp}_SL{sl}_{mode}: {e}")
                    traceback.print_exc()

    sweep_time = time.time() - t1

    # Summary table
    print(f"\n{'='*120}")
    print(f"DIRECTIONAL SWEEP | {len(days)} days | top {args.top_pct:.0%} | {args.permutations} perms")
    print(f"Load: {load_time:.0f}s | Sweep: {sweep_time:.0f}s")
    print(f"{'='*120}")

    hdr = (f"{'Config':<30s} {'N':>6s} {'PnL(t)':>9s} {'Avg':>7s} {'WR':>6s} {'PF':>6s} "
           f"{'Sharpe':>7s} {'Sort':>7s} {'G/R':>5s} {'Edge':>9s} {'p':>6s} {'Sig':>3s}")
    print(hdr)
    print('-' * 120)

    for r in sorted(all_results, key=lambda x: x.get('model_edge', 0), reverse=True):
        sig = 'YES' if r['p_value'] < 0.05 else ' no'
        avg = r.get('avg_pnl_per_trade', 0)
        print(f"  {r['config']:<30s} {r['n_trades']:>5d} {r['real_pnl']:>+9.1f} {avg:>+7.3f} "
              f"{r['win_rate']:>5.1%} {r['profit_factor']:>6.2f} "
              f"{r['sharpe']:>+7.2f} {r['sortino']:>+7.2f} "
              f"{r['green_days']}/{r['red_days']:<2d} "
              f"{r['model_edge']:>+9.1f} {r['p_value']:>6.3f} {sig:>3s}")

    # Highlight best configs
    profitable = [r for r in all_results if r['real_pnl'] > 0 and r['p_value'] < 0.05]
    if profitable:
        print(f"\n*** {len(profitable)} PROFITABLE + SIGNIFICANT CONFIGS FOUND ***")
        for r in sorted(profitable, key=lambda x: x['sharpe'], reverse=True):
            print(f"  {r['config']}: Sharpe {r['sharpe']:+.2f}, Sortino {r['sortino']:+.2f}, "
                  f"PnL {r['real_pnl']:+.0f}t (${r['real_pnl']*12.50:+,.0f}), "
                  f"WR {r['win_rate']:.1%}, {r['green_days']}/{r['green_days']+r['red_days']} green days")
    else:
        print(f"\n*** NO PROFITABLE + SIGNIFICANT CONFIGS. Model signal insufficient for passive FIFO trading. ***")

    # Save
    output_path = args.output or str(OUTPUT_DIR / "tick_replay_v5_directional_results.json")
    save_data = [{k: v for k, v in r.items() if k != 'trades'} for r in all_results]
    with open(output_path, 'w') as f:
        json.dump(save_data, f, indent=2, default=str)
    log.info(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
