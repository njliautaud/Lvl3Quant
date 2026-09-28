#!/usr/bin/env python3
"""
Tick-Level Replay v5 — Vol-Gated Conviction Strategy
======================================================
Combines two findings:
  1. Model edge scales with volatility (top 5% vol → 2x better)
  2. Consecutive same-direction predictions = higher conviction

Strategy:
  - Compute trailing realized vol from recent 1s labels
  - Only consider entries when vol > threshold
  - Require K consecutive same-direction predictions (conviction filter)
  - Enter at market, hold until signal reverses or max time
  - Test on ALL available OOT dates for honest assessment

Author: Claude (autonomous build, 2026-07-02)
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tick_level_replay import (
    ROOT, RAW_MBO_DIR, SMART_V3_DIR, OOT_PRED_DIR, OUTPUT_DIR,
    ES_TICK_SIZE, ES_TICK_VALUE, ES_RT_COMMISSION_TICKS,
    WINDOW_SIZE, STRIDE,
    compute_metrics, get_oot_dates,
)

logging.basicConfig(
    format="%(asctime)s [V5] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("V5")

# Cost models
COST_MARKET_RT = ES_RT_COMMISSION_TICKS + 1.0  # 1.376 ticks (market in + market out)
COST_PASSIVE_RT = ES_RT_COMMISSION_TICKS  # 0.376 ticks (passive in + passive out, best case)


class DayData:
    """Pre-loaded data for one day. Lightweight — only loads what we need."""

    def __init__(self, date_str: str):
        self.date_str = date_str
        self.valid = False

        mbo_path = SMART_V3_DIR / f"{date_str}_mbo_events.npz"
        pred_path = OOT_PRED_DIR / f"oot_{date_str}.npz"

        if not mbo_path.exists() or not pred_path.exists():
            log.warning(f"  {date_str}: missing data")
            return

        try:
            mbo = np.load(str(mbo_path), allow_pickle=True)
            pred = np.load(str(pred_path), allow_pickle=True)

            # Check for required keys
            if 'pred_log_ret_1s' not in pred:
                log.warning(f"  {date_str}: missing pred_log_ret_1s")
                return

            self.pred_1s = pred['pred_log_ret_1s'].astype(np.float64)
            n_pred = len(self.pred_1s)
            n_events = len(mbo['timestamps'])

            pred_indices = np.arange(WINDOW_SIZE - 1, n_events, STRIDE)[:n_pred]
            self.pred_ts = mbo['timestamps'][pred_indices]

            # Labels for ground truth analysis
            self.labels_1s = mbo['labels_1s'][pred_indices].astype(np.float64)
            self.labels_10s = mbo['labels_10s'][pred_indices].astype(np.float64)
            self.labels_30s = mbo['labels_30s'][pred_indices].astype(np.float64)

            # Compute trailing realized vol (rolling std of labels_1s over last 240 predictions)
            vol_window = 240  # ~60 seconds
            l1s_all = mbo['labels_1s'].astype(np.float64)
            self.trailing_vol = np.full(n_pred, np.nan)
            for i in range(vol_window, n_pred):
                start_idx = pred_indices[i - vol_window]
                end_idx = pred_indices[i]
                if end_idx < len(l1s_all):
                    segment = l1s_all[start_idx:end_idx]
                    valid_seg = segment[~np.isnan(segment)]
                    if len(valid_seg) > 10:
                        self.trailing_vol[i] = np.std(valid_seg)

            self.valid = True
            n_valid_vol = np.sum(~np.isnan(self.trailing_vol))
            log.info(f"  {date_str}: {n_pred} predictions, {n_valid_vol} with vol")

        except Exception as e:
            log.error(f"  {date_str}: error loading - {e}")


def run_volgated_conviction(day: DayData, vol_threshold: float,
                            conviction_k: int, hold_horizon: str,
                            mode: str = 'both',
                            shuffle: bool = False,
                            rng: np.random.Generator = None) -> List[Dict]:
    """
    Vol-gated conviction strategy.

    Args:
        vol_threshold: minimum trailing vol to consider entry (in ticks)
        conviction_k: require K consecutive same-direction predictions
        hold_horizon: '10s' or '30s' — which label to use for PnL
        mode: 'both', 'short_only', 'long_only'
        shuffle: permutation test flag
    """
    if not day.valid:
        return []

    preds = day.pred_1s.copy()
    if shuffle:
        if rng is None:
            rng = np.random.default_rng(42)
        rng.shuffle(preds)

    directions = np.sign(preds)
    vol = day.trailing_vol

    if hold_horizon == '10s':
        labels = day.labels_10s
    elif hold_horizon == '30s':
        labels = day.labels_30s
    else:
        labels = day.labels_10s

    n = len(preds)
    results = []

    # Track consecutive same-direction predictions
    streak_count = 0
    streak_dir = 0

    last_trade_idx = -conviction_k * 2  # ensure first trade can happen

    for i in range(n):
        if np.isnan(vol[i]) or np.isnan(labels[i]):
            streak_count = 0
            streak_dir = 0
            continue

        d = directions[i]
        if d == 0:
            streak_count = 0
            streak_dir = 0
            continue

        # Update streak
        if d == streak_dir:
            streak_count += 1
        else:
            streak_dir = d
            streak_count = 1

        # Check entry conditions
        if streak_count < conviction_k:
            continue

        if vol[i] < vol_threshold:
            continue

        # Mode filter
        if mode == 'short_only' and streak_dir != -1:
            continue
        if mode == 'long_only' and streak_dir != 1:
            continue

        # Cooldown: don't trade within conviction_k predictions of last trade
        if i - last_trade_idx < conviction_k:
            continue

        # TRADE: enter in streak direction, hold for horizon
        trade_dir = int(streak_dir)
        label_val = labels[i]

        if trade_dir == 1:
            pnl_ticks = label_val  # long profits from positive return
        else:
            pnl_ticks = -label_val  # short profits from negative return

        results.append({
            'date': day.date_str,
            'direction': trade_dir,
            'pnl_ticks': round(float(pnl_ticks), 4),
            'vol': round(float(vol[i]), 4),
            'streak': streak_count,
            'exit_type': 'TIMEOUT',
            'fill_time_ns': 0,
            'hold_time_ns': 0,
        })

        last_trade_idx = i
        streak_count = 0  # reset after trade

    return results


def main():
    parser = argparse.ArgumentParser(description="V5 — Vol-Gated Conviction")
    parser.add_argument('--n-dates', type=int, default=None)
    parser.add_argument('--vol-thresholds', type=float, nargs='+',
                        default=[0, 1.5, 2.0, 2.5, 3.0, 4.0])
    parser.add_argument('--conviction-k', type=int, nargs='+', default=[1, 3, 5, 10, 20])
    parser.add_argument('--horizons', nargs='+', default=['10s', '30s'])
    parser.add_argument('--modes', nargs='+', default=['both', 'short_only'])
    parser.add_argument('--permutations', type=int, default=20)
    parser.add_argument('--output', type=str, default=None)
    args = parser.parse_args()

    dates = get_oot_dates()
    if args.n_dates:
        dates = dates[:args.n_dates]

    log.info(f"Loading {len(dates)} days (lightweight — no raw MBO needed)...")
    t0 = time.time()
    days = []
    for d in dates:
        day = DayData(d)
        if day.valid:
            days.append(day)
    load_time = time.time() - t0
    log.info(f"Loaded {len(days)} valid days in {load_time:.0f}s")

    all_results = []
    t1 = time.time()

    for mode in args.modes:
        for horizon in args.horizons:
            for vol_t in args.vol_thresholds:
                for conv_k in args.conviction_k:
                    label = f"{mode}_{horizon}_vol{vol_t}_k{conv_k}"

                    # Real trades
                    real_trades = []
                    for day in days:
                        trades = run_volgated_conviction(
                            day, vol_threshold=vol_t, conviction_k=conv_k,
                            hold_horizon=horizon, mode=mode
                        )
                        real_trades.extend(trades)

                    if len(real_trades) < 10:
                        continue

                    gross_pnl = sum(t['pnl_ticks'] for t in real_trades)
                    n_trades = len(real_trades)
                    avg_gross = gross_pnl / n_trades

                    # Day breakdown
                    day_pnls = {}
                    for t in real_trades:
                        d = t['date']
                        day_pnls[d] = day_pnls.get(d, 0) + t['pnl_ticks']
                    green = sum(1 for v in day_pnls.values() if v > 0)
                    red = sum(1 for v in day_pnls.values() if v < 0)

                    wr = sum(1 for t in real_trades if t['pnl_ticks'] > 0) / n_trades

                    # Permutation test
                    perm_pnls = []
                    for pi in range(args.permutations):
                        rng = np.random.default_rng(seed=pi * 1000 + 42)
                        pt = []
                        for day in days:
                            trades = run_volgated_conviction(
                                day, vol_threshold=vol_t, conviction_k=conv_k,
                                hold_horizon=horizon, mode=mode,
                                shuffle=True, rng=rng
                            )
                            pt.extend(trades)
                        perm_pnls.append(sum(t['pnl_ticks'] for t in pt))

                    mean_perm = np.mean(perm_pnls) if perm_pnls else 0
                    edge = gross_pnl - mean_perm
                    p_val = np.mean([p >= gross_pnl for p in perm_pnls]) if perm_pnls else 1.0

                    net_market = gross_pnl - n_trades * COST_MARKET_RT
                    net_passive = gross_pnl - n_trades * COST_PASSIVE_RT
                    avg_net_market = net_market / n_trades
                    avg_net_passive = net_passive / n_trades

                    all_results.append({
                        'label': label,
                        'mode': mode,
                        'horizon': horizon,
                        'vol_threshold': vol_t,
                        'conviction_k': conv_k,
                        'n_trades': n_trades,
                        'gross_pnl': round(float(gross_pnl), 1),
                        'avg_gross': round(float(avg_gross), 4),
                        'net_market': round(float(net_market), 1),
                        'net_passive': round(float(net_passive), 1),
                        'avg_net_passive': round(float(avg_net_passive), 4),
                        'win_rate': round(float(wr), 4),
                        'green_days': green,
                        'red_days': red,
                        'n_days_traded': green + red,
                        'mean_perm_pnl': round(float(mean_perm), 1),
                        'model_edge': round(float(edge), 1),
                        'p_value': round(float(p_val), 3),
                        'trades_per_day': round(n_trades / len(days), 1),
                    })

                    sig = '✅' if p_val < 0.05 else ''
                    prof = '💰' if net_passive > 0 else ''
                    log.info(f"  {label}: n={n_trades}, gross={avg_gross:+.3f}t/tr, "
                             f"net_passive={avg_net_passive:+.3f}t/tr, "
                             f"edge={edge:+.1f}t, p={p_val:.3f} {sig} {prof}")

    sweep_time = time.time() - t1

    # Summary
    print(f"\n{'='*130}")
    print(f"VOL-GATED CONVICTION SWEEP | {len(days)} days | {args.permutations} perms")
    print(f"Load: {load_time:.0f}s | Sweep: {sweep_time:.0f}s")
    print(f"Market RT cost: {COST_MARKET_RT:.3f}t | Passive RT cost: {COST_PASSIVE_RT:.3f}t")
    print(f"{'='*130}")
    print(f"{'Label':<35} {'N':>5} {'Gr/tr':>7} {'N_mkt':>7} {'N_pas':>7} {'WR':>6} "
          f"{'G/R':>5} {'Edge':>8} {'p':>6} {'$':>3}")
    print("-" * 130)

    for r in sorted(all_results, key=lambda x: x.get('avg_net_passive', -999), reverse=True)[:40]:
        sig = '✅' if r['p_value'] < 0.05 else '❌'
        prof = '💰' if r['net_passive'] > 0 else '  '
        print(f"  {r['label']:<35} {r['n_trades']:>4} {r['avg_gross']:>+7.3f} "
              f"{r['net_market']:>+7.0f} {r['net_passive']:>+7.0f} {r['win_rate']:>5.1%} "
              f"{r['green_days']}/{r['red_days']:<2} "
              f"{r['model_edge']:>+8.1f} {r['p_value']:>6.3f} {prof}{sig}")

    # Save
    output_path = args.output or str(OUTPUT_DIR / "tick_replay_v5_volgated_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    log.info(f"Saved to {output_path}")

    # Highlight profitable configs
    profitable_passive = [r for r in all_results if r['net_passive'] > 0 and r['p_value'] < 0.10]
    if profitable_passive:
        print(f"\n{'='*80}")
        print(f"POTENTIALLY PROFITABLE CONFIGS (net passive > 0 AND p < 0.10)")
        print(f"{'='*80}")
        for r in sorted(profitable_passive, key=lambda x: x['net_passive'], reverse=True):
            daily_pnl = r['net_passive'] / r['n_days_traded'] if r['n_days_traded'] > 0 else 0
            print(f"  {r['label']}:")
            print(f"    {r['n_trades']} trades across {r['n_days_traded']} days ({r['trades_per_day']:.0f}/day)")
            print(f"    Gross: {r['avg_gross']:+.3f}t/trade, Net passive: {r['avg_net_passive']:+.3f}t/trade")
            print(f"    Total net: {r['net_passive']:+.0f}t (${r['net_passive']*ES_TICK_VALUE:+,.0f})")
            print(f"    WR: {r['win_rate']:.1%}, Green/Red: {r['green_days']}/{r['red_days']}")
            print(f"    Edge vs random: {r['model_edge']:+.1f}t, p={r['p_value']:.3f}")
    else:
        print(f"\n  No configs profitable with passive execution at p < 0.10")


if __name__ == '__main__':
    main()
