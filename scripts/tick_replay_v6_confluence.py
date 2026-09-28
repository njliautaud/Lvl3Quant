#!/usr/bin/env python3
"""
Tick-Level Replay v6 — Multi-Horizon Confluence Strategy
=========================================================
Instead of using k-consecutive predictions (which was just autocorrelation
bias), this tests whether MULTI-HORIZON agreement produces a stronger signal.

The model predicts at 1s, 5s, 10s, 30s horizons simultaneously.
If all horizons agree on direction, that's a fundamentally different
kind of conviction than temporal autocorrelation.

Also tests: per-direction percentile selection to avoid long bias.

Key insight from v5 failure: 61.6% of 1s predictions are positive (bimodal).
So we MUST normalize per-direction or we'll just always go long.

Strategy variants tested:
  A. Raw confluence: all horizons agree on direction
  B. Percentile confluence: each horizon in top/bottom N% AND all agree
  C. Vol-gated confluence: A or B but only in high-vol periods

Author: Claude (autonomous research, 2026-07-02)
"""

import json
import logging
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
    format="%(asctime)s [V6] %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("V6")

COST_PASSIVE_RT = ES_RT_COMMISSION_TICKS  # 0.376 ticks
COST_MARKET_RT = ES_RT_COMMISSION_TICKS + 1.0  # 1.376 ticks


class DayData:
    """Load multi-horizon predictions for one day."""

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

            # Need all 4 horizons
            needed = ['pred_log_ret_1s', 'pred_log_ret_5s', 'pred_log_ret_10s', 'pred_log_ret_30s']
            for k in needed:
                if k not in pred:
                    return

            n_events = len(mbo['timestamps'])
            self.preds = {}
            for h in ['1s', '5s', '10s', '30s']:
                p = pred[f'pred_log_ret_{h}'].astype(np.float64)
                self.preds[h] = p

            n_pred = len(self.preds['1s'])
            pred_indices = np.arange(WINDOW_SIZE - 1, n_events, STRIDE)[:n_pred]

            # Align all predictions to same length
            min_len = min(len(self.preds[h]) for h in self.preds)
            min_len = min(min_len, len(pred_indices))
            for h in self.preds:
                self.preds[h] = self.preds[h][:min_len]
            pred_indices = pred_indices[:min_len]

            self.pred_ts = mbo['timestamps'][pred_indices]
            self.labels_10s = mbo['labels_10s'][pred_indices].astype(np.float64)
            self.labels_30s = mbo['labels_30s'][pred_indices].astype(np.float64)
            self.n_pred = min_len

            # Trailing vol
            vol_window = 240
            l1s_all = mbo['labels_1s'].astype(np.float64)
            self.trailing_vol = np.full(min_len, np.nan)
            for i in range(vol_window, min_len):
                start_idx = pred_indices[i - vol_window]
                end_idx = pred_indices[i]
                if end_idx < len(l1s_all):
                    segment = l1s_all[start_idx:end_idx]
                    valid_seg = segment[~np.isnan(segment)]
                    if len(valid_seg) > 10:
                        self.trailing_vol[i] = np.std(valid_seg)

            self.valid = True
            log.info(f"  {date_str}: {min_len} predictions (all 4 horizons)")

        except Exception as e:
            log.error(f"  {date_str}: error - {e}")


def compute_percentile_ranks(days: list, horizons=['1s', '5s', '10s', '30s']):
    """Compute per-direction percentile ranks across all days.

    For each horizon, separately rank positive and negative predictions.
    Returns percentile rank (0-1) where 1.0 = strongest signal in that direction.
    """
    # First pass: collect all predictions per horizon to compute percentiles
    all_pos = {h: [] for h in horizons}
    all_neg = {h: [] for h in horizons}

    for day in days:
        for h in horizons:
            p = day.preds[h]
            pos_mask = p > 0
            neg_mask = p < 0
            all_pos[h].extend(p[pos_mask].tolist())
            all_neg[h].extend(p[neg_mask].tolist())

    # Compute percentile thresholds
    thresholds = {}
    for h in horizons:
        pos_arr = np.array(all_pos[h])
        neg_arr = np.array(all_neg[h])
        thresholds[h] = {
            'pos_p50': np.percentile(pos_arr, 50) if len(pos_arr) > 0 else 0,
            'pos_p75': np.percentile(pos_arr, 75) if len(pos_arr) > 0 else 0,
            'pos_p90': np.percentile(pos_arr, 90) if len(pos_arr) > 0 else 0,
            'neg_p50': np.percentile(neg_arr, 50) if len(neg_arr) > 0 else 0,  # more negative
            'neg_p25': np.percentile(neg_arr, 25) if len(neg_arr) > 0 else 0,
            'neg_p10': np.percentile(neg_arr, 10) if len(neg_arr) > 0 else 0,
        }
        log.info(f"  {h} pos: p50={thresholds[h]['pos_p50']:.4f} p90={thresholds[h]['pos_p90']:.4f}")
        log.info(f"  {h} neg: p50={thresholds[h]['neg_p50']:.4f} p10={thresholds[h]['neg_p10']:.4f}")

    return thresholds


def run_confluence(day: DayData, horizons_required: list,
                   hold_horizon: str = '10s',
                   percentile_filter: str = None,
                   thresholds: dict = None,
                   vol_threshold: float = 0.0,
                   cooldown: int = 4,
                   shuffle: bool = False, rng=None) -> List[Dict]:
    """
    Multi-horizon confluence strategy.

    Args:
        horizons_required: list of horizons that must agree (e.g. ['1s', '5s', '10s'])
        hold_horizon: which label to use for PnL
        percentile_filter: None, 'p50', 'p75', 'p90' — require each horizon above this %
        thresholds: from compute_percentile_ranks
        vol_threshold: minimum trailing vol
        cooldown: minimum predictions between trades
    """
    if not day.valid:
        return []

    preds = {}
    for h in horizons_required:
        p = day.preds[h].copy()
        if shuffle:
            if rng is None:
                rng = np.random.default_rng(42)
            rng.shuffle(p)
        preds[h] = p

    if hold_horizon == '10s':
        labels = day.labels_10s
    else:
        labels = day.labels_30s

    vol = day.trailing_vol
    n = day.n_pred
    results = []
    last_trade_idx = -cooldown * 2

    for i in range(n):
        if np.isnan(vol[i]) or np.isnan(labels[i]):
            continue

        if vol_threshold > 0 and vol[i] < vol_threshold:
            continue

        if i - last_trade_idx < cooldown:
            continue

        # Check all horizons agree on direction
        dirs = [np.sign(preds[h][i]) for h in horizons_required]
        if any(d == 0 for d in dirs):
            continue
        if len(set(dirs)) != 1:
            continue  # not all same direction

        trade_dir = int(dirs[0])

        # Percentile filter: require each prediction in top N% for its direction
        if percentile_filter and thresholds:
            passed = True
            for h in horizons_required:
                p = preds[h][i]
                if trade_dir == 1:
                    # Long: require prediction above threshold
                    thresh_key = f'pos_{percentile_filter}'
                    if p < thresholds[h].get(thresh_key, 0):
                        passed = False
                        break
                else:
                    # Short: require prediction below threshold (more negative)
                    thresh_key = f'neg_{percentile_filter.replace("p90","p10").replace("p75","p25")}'
                    if p > thresholds[h].get(thresh_key, 0):
                        passed = False
                        break
            if not passed:
                continue

        # Execute trade
        label_val = labels[i]
        pnl_ticks = label_val if trade_dir == 1 else -label_val

        results.append({
            'date': day.date_str,
            'direction': trade_dir,
            'pnl_ticks': round(float(pnl_ticks), 4),
            'vol': round(float(vol[i]), 4),
        })

        last_trade_idx = i

    return results


def analyze_config(label, trades, days, n_perms=30, perm_fn=None):
    """Analyze a set of trades with permutation test."""
    if len(trades) < 10:
        return None

    n = len(trades)
    gross = sum(t['pnl_ticks'] for t in trades)
    avg_gross = gross / n
    avg_net = avg_gross - COST_PASSIVE_RT

    # Direction breakdown
    longs = [t for t in trades if t['direction'] == 1]
    shorts = [t for t in trades if t['direction'] == -1]
    long_frac = len(longs) / n

    # Per-day
    day_pnls = {}
    for t in trades:
        day_pnls[t['date']] = day_pnls.get(t['date'], 0) + t['pnl_ticks']
    green = sum(1 for v in day_pnls.values() if v > 0)
    red = sum(1 for v in day_pnls.values() if v < 0)

    wr = sum(1 for t in trades if t['pnl_ticks'] > 0) / n

    # Permutation test
    p_val = 1.0
    edge = 0
    if perm_fn:
        perm_grosses = []
        for pi in range(n_perms):
            rng = np.random.default_rng(pi * 1337 + 42)
            pt = perm_fn(rng)
            perm_grosses.append(sum(t['pnl_ticks'] for t in pt))
        p_val = np.mean([p >= gross for p in perm_grosses])
        edge = gross - np.mean(perm_grosses)

    return {
        'label': label,
        'n_trades': n,
        'avg_gross': round(avg_gross, 4),
        'avg_net': round(avg_net, 4),
        'win_rate': round(wr, 4),
        'green_days': green,
        'red_days': red,
        'long_frac': round(long_frac, 3),
        'p_value': round(p_val, 3),
        'edge': round(edge, 1),
        'gross_total': round(gross, 1),
        'net_total': round(gross - n * COST_PASSIVE_RT, 1),
    }


def main():
    dates = get_oot_dates()
    log.info(f"Loading {len(dates)} days with multi-horizon predictions...")
    t0 = time.time()
    days = []
    for d in dates:
        day = DayData(d)
        if day.valid:
            days.append(day)
    log.info(f"Loaded {len(days)} valid days in {time.time()-t0:.0f}s")

    if not days:
        log.error("No valid days with multi-horizon predictions!")
        return

    # Compute percentile thresholds
    log.info("Computing per-direction percentile thresholds...")
    thresholds = compute_percentile_ranks(days)

    results = []
    t1 = time.time()

    # ================================================================
    # CONFIG SWEEP
    # ================================================================
    horizon_combos = [
        ['1s', '5s'],
        ['1s', '10s'],
        ['1s', '5s', '10s'],
        ['1s', '5s', '10s', '30s'],
        ['5s', '10s'],
        ['5s', '10s', '30s'],
        ['10s', '30s'],
    ]

    pct_filters = [None, 'p50', 'p75', 'p90']
    vol_thresholds = [0.0, 1.5, 2.5]

    for horizons in horizon_combos:
        h_label = '+'.join(horizons)
        for pct in pct_filters:
            for vol_t in vol_thresholds:
                pct_label = pct or 'raw'
                label = f"{h_label}_{pct_label}_vol{vol_t}"

                # Real trades
                trades = []
                for day in days:
                    trades.extend(run_confluence(
                        day, horizons, hold_horizon='10s',
                        percentile_filter=pct, thresholds=thresholds,
                        vol_threshold=vol_t, cooldown=4
                    ))

                if len(trades) < 10:
                    continue

                # Permutation function
                def make_perm_fn(horizons, pct, vol_t):
                    def perm_fn(rng):
                        pt = []
                        for day in days:
                            pt.extend(run_confluence(
                                day, horizons, hold_horizon='10s',
                                percentile_filter=pct, thresholds=thresholds,
                                vol_threshold=vol_t, cooldown=4,
                                shuffle=True, rng=rng
                            ))
                        return pt
                    return perm_fn

                result = analyze_config(
                    label, trades, days, n_perms=30,
                    perm_fn=make_perm_fn(horizons, pct, vol_t)
                )

                if result:
                    results.append(result)
                    sig = '✅' if result['p_value'] < 0.05 else ''
                    prof = '💰' if result['avg_net'] > 0 else ''
                    log.info(f"  {label}: n={result['n_trades']}, "
                             f"gross={result['avg_gross']:+.3f}, net={result['avg_net']:+.3f}, "
                             f"L%={result['long_frac']:.0%}, "
                             f"p={result['p_value']:.3f} {sig} {prof}")

    sweep_time = time.time() - t1

    # ================================================================
    # RESULTS TABLE
    # ================================================================
    print(f"\n{'='*120}")
    print(f"MULTI-HORIZON CONFLUENCE SWEEP | {len(days)} days | 30 perms")
    print(f"Sweep: {sweep_time:.0f}s")
    print(f"{'='*120}")
    print(f"{'Label':<45} {'N':>5} {'Gr/tr':>7} {'N/tr':>7} {'WR':>6} "
          f"{'L%':>5} {'G/R':>5} {'Edge':>8} {'p':>6}")
    print("-" * 120)

    for r in sorted(results, key=lambda x: x.get('avg_net', -999), reverse=True)[:30]:
        sig = '✅' if r['p_value'] < 0.05 else '❌'
        prof = '💰' if r['avg_net'] > 0 else '  '
        print(f"  {r['label']:<45} {r['n_trades']:>4} {r['avg_gross']:>+7.3f} "
              f"{r['avg_net']:>+7.3f} {r['win_rate']:>5.1%} "
              f"{r['long_frac']:>4.0%} {r['green_days']}/{r['red_days']:<2} "
              f"{r['edge']:>+8.1f} {r['p_value']:>6.3f} {prof}{sig}")

    # ================================================================
    # DIRECTION BALANCE CHECK
    # ================================================================
    print(f"\n{'='*80}")
    print("DIRECTION BALANCE CHECK — configs with >30% shorts")
    print(f"{'='*80}")
    balanced = [r for r in results if r['long_frac'] < 0.70]
    if balanced:
        for r in sorted(balanced, key=lambda x: x['avg_net'], reverse=True)[:10]:
            sig = '✅' if r['p_value'] < 0.05 else '❌'
            print(f"  {r['label']}: n={r['n_trades']}, net={r['avg_net']:+.3f}, "
                  f"L%={r['long_frac']:.0%}, p={r['p_value']:.3f} {sig}")
    else:
        print("  ⚠️  ALL configs are >70% long — same bias problem as v5")

    # Save
    output_path = OUTPUT_DIR / "tick_replay_v6_confluence_results.json"
    with open(str(output_path), 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
