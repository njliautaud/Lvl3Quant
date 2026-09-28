#!/usr/bin/env python3
"""
V5b — Deep analysis of the k=20 conviction finding.
=====================================================
The v5 sweep found that requiring 20 consecutive same-direction
predictions makes the signal profitable with passive execution.

This script does deep validation:
  1. Per-day PnL breakdown (green/red days, consistency)
  2. Direction split (long vs short separately)
  3. Autocorrelation test: are 20-streaks just prediction autocorrelation?
  4. Time-of-day distribution (morning vs afternoon edge)
  5. Larger permutation test (100 shuffles, not 20)
  6. Multiple k values around 20 for sensitivity (15, 18, 20, 25, 30)
  7. Is k=20 over-fit? Test on 10s and 30s horizons both

Author: Claude (autonomous research, 2026-07-02)
"""

import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tick_level_replay import (
    ROOT, SMART_V3_DIR, OOT_PRED_DIR, OUTPUT_DIR,
    ES_TICK_SIZE, ES_TICK_VALUE, ES_RT_COMMISSION_TICKS,
    WINDOW_SIZE, STRIDE,
    get_oot_dates,
)

logging.basicConfig(
    format="%(asctime)s [V5b] %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("V5b")

COST_PASSIVE_RT = ES_RT_COMMISSION_TICKS  # 0.376 ticks


class DayData:
    """Same lightweight loader as v5."""
    def __init__(self, date_str: str):
        self.date_str = date_str
        self.valid = False

        mbo_path = SMART_V3_DIR / f"{date_str}_mbo_events.npz"
        pred_path = OOT_PRED_DIR / f"oot_{date_str}.npz"

        if not mbo_path.exists() or not pred_path.exists():
            return

        try:
            mbo = np.load(str(mbo_path), allow_pickle=True)
            pred = np.load(str(pred_path), allow_pickle=True)

            if 'pred_log_ret_1s' not in pred:
                return

            self.pred_1s = pred['pred_log_ret_1s'].astype(np.float64)
            n_pred = len(self.pred_1s)
            n_events = len(mbo['timestamps'])

            pred_indices = np.arange(WINDOW_SIZE - 1, n_events, STRIDE)[:n_pred]
            self.pred_ts = mbo['timestamps'][pred_indices]

            self.labels_1s = mbo['labels_1s'][pred_indices].astype(np.float64)
            self.labels_10s = mbo['labels_10s'][pred_indices].astype(np.float64)
            self.labels_30s = mbo['labels_30s'][pred_indices].astype(np.float64)

            # Trailing vol
            vol_window = 240
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
        except Exception as e:
            log.error(f"  {date_str}: error - {e}")


def run_conviction(day: DayData, conviction_k: int, hold_horizon: str,
                   vol_threshold: float = 0.0, mode: str = 'both',
                   shuffle: bool = False, rng=None) -> List[Dict]:
    """Same logic as v5 but returns more detailed trade info."""
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
    else:
        labels = day.labels_30s

    n = len(preds)
    results = []
    streak_count = 0
    streak_dir = 0
    last_trade_idx = -conviction_k * 2

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

        if d == streak_dir:
            streak_count += 1
        else:
            streak_dir = d
            streak_count = 1

        if streak_count < conviction_k:
            continue
        if vol_threshold > 0 and vol[i] < vol_threshold:
            continue
        if mode == 'short_only' and streak_dir != -1:
            continue
        if mode == 'long_only' and streak_dir != 1:
            continue
        if i - last_trade_idx < conviction_k:
            continue

        trade_dir = int(streak_dir)
        label_val = labels[i]
        pnl_ticks = label_val if trade_dir == 1 else -label_val

        # Extract prediction index for time-of-day analysis
        ts = day.pred_ts[i] if i < len(day.pred_ts) else 0

        results.append({
            'date': day.date_str,
            'direction': trade_dir,
            'pnl_ticks': round(float(pnl_ticks), 4),
            'vol': round(float(vol[i]), 4),
            'streak': streak_count,
            'pred_idx': i,
            'timestamp': int(ts),
            'pred_val': float(preds[i]),
        })

        last_trade_idx = i
        streak_count = 0

    return results


def main():
    dates = get_oot_dates()
    log.info(f"Loading {len(dates)} days...")
    t0 = time.time()
    days = []
    for d in dates:
        day = DayData(d)
        if day.valid:
            days.append(day)
    log.info(f"Loaded {len(days)} valid days in {time.time()-t0:.0f}s")

    # ================================================================
    # TEST 1: k sensitivity (15-40) at vol=0 and vol=1.5 — NO PERMS (fast)
    # ================================================================
    print(f"\n{'='*100}")
    print("TEST 1: CONVICTION-K SENSITIVITY (10s horizon, both directions, no perms)")
    print(f"{'='*100}")
    print(f"{'K':>4} {'Vol':>5} {'N':>6} {'Gross/tr':>9} {'Net/tr':>9} {'WR':>6} {'G/R':>6}")
    print("-" * 60)

    for vol_t in [0.0, 1.5]:
        for k in [15, 18, 20, 22, 25, 30, 40]:
            trades = []
            for day in days:
                trades.extend(run_conviction(day, k, '10s', vol_threshold=vol_t))
            if len(trades) < 5:
                print(f"  {k:>3} {vol_t:>5.1f}  <5 trades, skipped")
                continue
            gross = sum(t['pnl_ticks'] for t in trades)
            n = len(trades)
            avg_gross = gross / n
            avg_net = avg_gross - COST_PASSIVE_RT
            wr = sum(1 for t in trades if t['pnl_ticks'] > 0) / n

            # Per-day breakdown
            day_pnls = {}
            for t in trades:
                day_pnls[t['date']] = day_pnls.get(t['date'], 0) + t['pnl_ticks']
            green = sum(1 for v in day_pnls.values() if v > 0)
            red = sum(1 for v in day_pnls.values() if v < 0)

            prof = '💰' if avg_net > 0 else '  '
            print(f"  {k:>3} {vol_t:>5.1f} {n:>5} {avg_gross:>+8.3f}t {avg_net:>+8.3f}t "
                  f"{wr:>5.1%} {green}/{red:<2} {prof}")

    # ================================================================
    # TEST 2: Direction split for k=20 (best config)
    # ================================================================
    print(f"\n{'='*100}")
    print("TEST 2: DIRECTION SPLIT (k=20, vol=1.5, 10s horizon)")
    print(f"{'='*100}")

    for mode in ['both', 'long_only', 'short_only']:
        trades = []
        for day in days:
            trades.extend(run_conviction(day, 20, '10s', vol_threshold=1.5, mode=mode))
        if len(trades) < 3:
            print(f"  {mode}: only {len(trades)} trades, skipping")
            continue
        gross = sum(t['pnl_ticks'] for t in trades)
        n = len(trades)
        avg_gross = gross / n
        avg_net = avg_gross - COST_PASSIVE_RT
        wr = sum(1 for t in trades if t['pnl_ticks'] > 0) / n

        day_pnls = {}
        for t in trades:
            day_pnls[t['date']] = day_pnls.get(t['date'], 0) + t['pnl_ticks']
        green = sum(1 for v in day_pnls.values() if v > 0)
        red = sum(1 for v in day_pnls.values() if v < 0)

        # Direction breakdown within 'both'
        if mode == 'both':
            longs = [t for t in trades if t['direction'] == 1]
            shorts = [t for t in trades if t['direction'] == -1]
            if longs:
                l_avg = sum(t['pnl_ticks'] for t in longs) / len(longs)
                print(f"  LONGS:  n={len(longs)}, gross={l_avg:+.3f}t/tr, net={l_avg-COST_PASSIVE_RT:+.3f}t/tr")
            if shorts:
                s_avg = sum(t['pnl_ticks'] for t in shorts) / len(shorts)
                print(f"  SHORTS: n={len(shorts)}, gross={s_avg:+.3f}t/tr, net={s_avg-COST_PASSIVE_RT:+.3f}t/tr")

        print(f"  {mode:>12}: n={n}, gross={avg_gross:+.3f}t/tr, net={avg_net:+.3f}t/tr, "
              f"WR={wr:.1%}, G/R={green}/{red}")

    # ================================================================
    # TEST 3: Per-day PnL table for k=20, vol=1.5
    # ================================================================
    print(f"\n{'='*100}")
    print("TEST 3: PER-DAY BREAKDOWN (k=20, vol=1.5, 10s horizon)")
    print(f"{'='*100}")
    print(f"{'Date':>12} {'N':>4} {'Gross':>8} {'Net':>8} {'$/day':>8} {'WR':>6}")
    print("-" * 55)

    trades = []
    for day in days:
        trades.extend(run_conviction(day, 20, '10s', vol_threshold=1.5))

    day_trades = {}
    for t in trades:
        day_trades.setdefault(t['date'], []).append(t)

    total_net = 0
    for date in sorted(day_trades.keys()):
        dt = day_trades[date]
        n = len(dt)
        gross = sum(t['pnl_ticks'] for t in dt)
        net = gross - n * COST_PASSIVE_RT
        dollar = net * ES_TICK_VALUE
        wr = sum(1 for t in dt if t['pnl_ticks'] > 0) / n if n > 0 else 0
        total_net += net
        emoji = '🟢' if net > 0 else '🔴'
        print(f"  {date:>10} {n:>4} {gross:>+7.1f}t {net:>+7.1f}t ${dollar:>+7.0f} {wr:>5.1%} {emoji}")

    print(f"  {'TOTAL':>10} {len(trades):>4} {sum(t['pnl_ticks'] for t in trades):>+7.1f}t "
          f"{total_net:>+7.1f}t ${total_net * ES_TICK_VALUE:>+7.0f}")

    # ================================================================
    # TEST 4: Autocorrelation analysis — is k=20 just serial correlation?
    # ================================================================
    print(f"\n{'='*100}")
    print("TEST 4: PREDICTION AUTOCORRELATION CHECK")
    print(f"{'='*100}")
    print("If predictions are highly autocorrelated, 20-streaks are common")
    print("and the finding may be an artifact of prediction smoothness.\n")

    all_preds = np.concatenate([d.pred_1s for d in days if d.valid])
    all_dirs = np.sign(all_preds)

    # Count streaks in real predictions
    streak_lengths = []
    cur_len = 1
    for i in range(1, len(all_dirs)):
        if all_dirs[i] == all_dirs[i-1] and all_dirs[i] != 0:
            cur_len += 1
        else:
            streak_lengths.append(cur_len)
            cur_len = 1
    streak_lengths.append(cur_len)

    streak_arr = np.array(streak_lengths)
    print(f"  Real predictions: {len(all_dirs):,} total")
    print(f"  Streaks ≥20: {np.sum(streak_arr >= 20):,} ({100*np.mean(streak_arr >= 20):.2f}%)")
    print(f"  Streaks ≥10: {np.sum(streak_arr >= 10):,} ({100*np.mean(streak_arr >= 10):.2f}%)")
    print(f"  Mean streak length: {np.mean(streak_arr):.1f}")
    print(f"  Max streak length: {np.max(streak_arr)}")

    # Autocorrelation of directions
    from numpy import corrcoef
    lags = [1, 5, 10, 20, 50]
    print(f"\n  Direction autocorrelation:")
    for lag in lags:
        if lag < len(all_dirs):
            corr = np.corrcoef(all_dirs[:-lag], all_dirs[lag:])[0, 1]
            print(f"    lag-{lag}: {corr:.4f}")

    # Compare to random baseline
    rng = np.random.default_rng(42)
    random_dirs = rng.choice([-1, 1], size=len(all_dirs))
    rand_streaks = []
    cur_len = 1
    for i in range(1, len(random_dirs)):
        if random_dirs[i] == random_dirs[i-1]:
            cur_len += 1
        else:
            rand_streaks.append(cur_len)
            cur_len = 1
    rand_streaks.append(cur_len)
    rand_arr = np.array(rand_streaks)
    print(f"\n  Random baseline:")
    print(f"  Streaks ≥20: {np.sum(rand_arr >= 20):,} (expected ~{len(all_dirs)/2**20:.1f})")
    print(f"  Streaks ≥10: {np.sum(rand_arr >= 10):,}")
    print(f"  Mean streak length: {np.mean(rand_arr):.1f}")

    enrichment = np.sum(streak_arr >= 20) / max(np.sum(rand_arr >= 20), 1)
    print(f"\n  20-streak enrichment vs random: {enrichment:.0f}x")
    print(f"  ⚠️  HIGH autocorrelation = predictions are smooth, NOT surprising")
    print(f"  The question is: does the MODEL add value beyond this autocorrelation?")
    print(f"  → That's what the permutation test answers (shuffling preserves marginal dist)")

    # ================================================================
    # TEST 5: Large permutation test (100 shuffles)
    # ================================================================
    print(f"\n{'='*100}")
    print("TEST 5: LARGE PERMUTATION TEST (100 shuffles, k=20, vol=1.5, 10s)")
    print(f"{'='*100}")

    real_trades = trades  # from test 3
    real_gross = sum(t['pnl_ticks'] for t in real_trades)
    real_n = len(real_trades)

    perm_results = []
    for pi in range(100):
        rng = np.random.default_rng(pi * 1337 + 7)
        pt = []
        for day in days:
            pt.extend(run_conviction(day, 20, '10s', vol_threshold=1.5,
                                     shuffle=True, rng=rng))
        perm_gross = sum(t['pnl_ticks'] for t in pt)
        perm_n = len(pt)
        perm_results.append({'gross': perm_gross, 'n': perm_n})

    perm_grosses = [p['gross'] for p in perm_results]
    perm_ns = [p['n'] for p in perm_results]
    p_val = np.mean([p >= real_gross for p in perm_grosses])

    print(f"  Real: gross={real_gross:+.1f}t, n={real_n}")
    print(f"  Perm: gross={np.mean(perm_grosses):+.1f}t ± {np.std(perm_grosses):.1f}t, "
          f"n={np.mean(perm_ns):.0f} ± {np.std(perm_ns):.0f}")
    print(f"  Edge: {real_gross - np.mean(perm_grosses):+.1f}t")
    print(f"  p-value: {p_val:.3f} (100 permutations)")

    # CRITICAL: also check per-trade edge
    real_avg = real_gross / real_n if real_n > 0 else 0
    perm_avgs = [p['gross']/p['n'] if p['n'] > 0 else 0 for p in perm_results]
    p_avg = np.mean([pa >= real_avg for pa in perm_avgs])
    print(f"\n  Per-trade: real={real_avg:+.4f}t, perm={np.mean(perm_avgs):+.4f}t ± {np.std(perm_avgs):.4f}t")
    print(f"  Per-trade p-value: {p_avg:.3f}")

    # ================================================================
    # TEST 6: 30s horizon comparison
    # ================================================================
    print(f"\n{'='*100}")
    print("TEST 6: 30s HORIZON (k=20, vol=1.5)")
    print(f"{'='*100}")

    trades_30s = []
    for day in days:
        trades_30s.extend(run_conviction(day, 20, '30s', vol_threshold=1.5))

    if trades_30s:
        gross_30 = sum(t['pnl_ticks'] for t in trades_30s)
        n_30 = len(trades_30s)
        avg_30 = gross_30 / n_30
        net_30 = avg_30 - COST_PASSIVE_RT
        wr_30 = sum(1 for t in trades_30s if t['pnl_ticks'] > 0) / n_30

        print(f"  10s horizon: n={real_n}, gross={real_gross/real_n:+.3f}t/tr, "
              f"net={real_gross/real_n - COST_PASSIVE_RT:+.3f}t/tr")
        print(f"  30s horizon: n={n_30}, gross={avg_30:+.3f}t/tr, "
              f"net={net_30:+.3f}t/tr, WR={wr_30:.1%}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*100}")
    print("SUMMARY: K=20 CONVICTION STRATEGY DEEP VALIDATION")
    print(f"{'='*100}")

    # Best config recap
    best_gross = real_gross / real_n if real_n > 0 else 0
    best_net = best_gross - COST_PASSIVE_RT
    print(f"\n  Best config: k=20, vol≥1.5, 10s horizon, both directions")
    print(f"  Trades: {real_n} across {len(day_trades)} days ({real_n/len(day_trades):.1f}/day)")
    print(f"  Gross: {best_gross:+.3f} ticks/trade")
    print(f"  Net (passive): {best_net:+.3f} ticks/trade")
    print(f"  p-value (100 perms): {p_val:.3f}")
    print(f"  Total net: {total_net:+.1f} ticks = ${total_net * ES_TICK_VALUE:+,.0f} over {len(day_trades)} days")
    daily_avg = total_net / len(day_trades) if day_trades else 0
    print(f"  Average: {daily_avg:+.1f} ticks/day = ${daily_avg * ES_TICK_VALUE:+,.0f}/day")

    # Save detailed results
    output = {
        'config': {'conviction_k': 20, 'vol_threshold': 1.5, 'horizon': '10s'},
        'trades': trades,
        'per_day': {d: {'n': len(ts), 'gross': sum(t['pnl_ticks'] for t in ts),
                        'net': sum(t['pnl_ticks'] for t in ts) - len(ts) * COST_PASSIVE_RT}
                    for d, ts in day_trades.items()},
        'perm_test': {'real_gross': real_gross, 'real_n': real_n,
                      'perm_mean': float(np.mean(perm_grosses)),
                      'perm_std': float(np.std(perm_grosses)),
                      'p_value': float(p_val)},
    }
    out_path = OUTPUT_DIR / "v5b_k20_deep_analysis.json"
    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Saved to {out_path}")


if __name__ == '__main__':
    main()
