#!/usr/bin/env python3
"""
Tick-Level Replay v6 — Deep Dive on Breakthrough Config
========================================================
Focus: both_30s_vol2.5_k20 showed +1.2 ticks/trade net (p=0.000 from 20 perms).
Uses same data loading as v5_volgated (labels-based PnL).

Validates with:
  1. 200 permutation test (rigorous)
  2. Per-day PnL + drawdown analysis
  3. Direction breakdown (long vs short contribution)
  4. Parameter sensitivity (nearby configs)
  5. Different hold horizons
  6. Short-only and invert-longs variants

Author: Claude (autonomous, 2026-07-02)
"""

import gc
import json
import logging
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tick_level_replay import (
    ROOT, RAW_MBO_DIR, SMART_V3_DIR, OOT_PRED_DIR, OUTPUT_DIR,
    ES_TICK_SIZE, ES_TICK_VALUE, ES_RT_COMMISSION_TICKS,
    WINDOW_SIZE, STRIDE,
    get_oot_dates,
)

logging.basicConfig(
    format="%(asctime)s [V6-DEEP] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[logging.StreamHandler()],
)
log = logging.getLogger("V6")

COST_PASSIVE_RT = ES_RT_COMMISSION_TICKS  # 0.376 ticks


class DayData:
    """Same loader as v5_volgated."""
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

            # Trailing vol (same as v5_volgated)
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


def run_strategy(day: DayData, vol_threshold: float, conviction_k: int,
                 hold_horizon: str, mode: str = 'both',
                 shuffle: bool = False, rng=None) -> List[Dict]:
    """Same logic as v5_volgated."""
    if not day.valid:
        return []

    preds = day.pred_1s.copy()
    if shuffle:
        if rng is None:
            rng = np.random.default_rng(42)
        rng.shuffle(preds)

    directions = np.sign(preds)
    vol = day.trailing_vol

    if hold_horizon == '30s':
        labels = day.labels_30s
    elif hold_horizon == '10s':
        labels = day.labels_10s
    else:
        labels = day.labels_1s

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
        if vol[i] < vol_threshold:
            continue

        if mode == 'short_only' and streak_dir != -1:
            continue
        if mode == 'long_only' and streak_dir != 1:
            continue

        if i - last_trade_idx < conviction_k:
            continue

        trade_dir = int(streak_dir)
        if mode == 'invert_longs' and trade_dir == 1:
            trade_dir = -1

        label_val = labels[i]
        if trade_dir == 1:
            pnl_ticks = label_val
        else:
            pnl_ticks = -label_val

        results.append({
            'date': day.date_str,
            'direction': trade_dir,
            'pnl_ticks': round(float(pnl_ticks), 4),
            'vol': round(float(vol[i]), 4),
            'streak': streak_count,
            'pred_idx': i,
        })

        last_trade_idx = i
        streak_count = 0

    return results


def compute_stats(trades: list) -> dict:
    if not trades:
        return {"n_trades": 0, "net_pnl": 0, "avg_net": 0, "sharpe_annual": 0}

    pnls = np.array([t["pnl_ticks"] for t in trades])
    net_pnls = pnls - COST_PASSIVE_RT

    gross = float(np.sum(pnls))
    net = float(np.sum(net_pnls))
    n = len(pnls)

    wins = net_pnls[net_pnls > 0]
    losses = net_pnls[net_pnls < 0]
    wr = len(wins) / n
    pf = float(np.sum(wins) / abs(np.sum(losses))) if len(losses) > 0 else float("inf")

    # Daily
    daily = defaultdict(float)
    daily_n = defaultdict(int)
    for t in trades:
        daily[t["date"]] += t["pnl_ticks"] - COST_PASSIVE_RT
        daily_n[t["date"]] += 1

    daily_pnls = np.array([daily[d] for d in sorted(daily.keys())])
    green = int(np.sum(daily_pnls > 0))
    red = int(np.sum(daily_pnls < 0))
    total_days = len(daily_pnls)

    if len(daily_pnls) > 1 and np.std(daily_pnls) > 0:
        sharpe = float(np.mean(daily_pnls) / np.std(daily_pnls) * np.sqrt(252))
        down = daily_pnls[daily_pnls < 0]
        sortino = float(np.mean(daily_pnls) / np.std(down) * np.sqrt(252)) if len(down) > 0 else 0
    else:
        sharpe = sortino = 0

    cum = np.cumsum(net_pnls)
    peak = np.maximum.accumulate(cum)
    max_dd = float(np.max(peak - cum))

    # Direction split
    longs = [t for t in trades if t["direction"] == 1]
    shorts = [t for t in trades if t["direction"] == -1]
    l_gross = sum(t["pnl_ticks"] for t in longs)
    s_gross = sum(t["pnl_ticks"] for t in shorts)
    l_net = l_gross - len(longs) * COST_PASSIVE_RT
    s_net = s_gross - len(shorts) * COST_PASSIVE_RT

    return {
        "n_trades": n,
        "gross_pnl": round(gross, 2),
        "net_pnl": round(net, 2),
        "avg_gross": round(gross / n, 4),
        "avg_net": round(net / n, 4),
        "win_rate": round(wr, 4),
        "profit_factor": round(pf, 3),
        "sharpe_annual": round(sharpe, 2),
        "sortino_annual": round(sortino, 2),
        "max_drawdown": round(max_dd, 2),
        "green_days": green,
        "red_days": red,
        "total_days": total_days,
        "trades_per_day": round(n / total_days, 1) if total_days else 0,
        "n_longs": len(longs),
        "n_shorts": len(shorts),
        "long_gross": round(l_gross, 2),
        "short_gross": round(s_gross, 2),
        "long_net": round(l_net, 2),
        "short_net": round(s_net, 2),
        "per_day": {d: {"n": daily_n[d], "net": round(daily[d], 2)} for d in sorted(daily.keys())},
    }


def run_perm_test(days: list, vol_t: float, k: int, horizon: str,
                  mode: str, n_perms: int) -> dict:
    """Permutation test: shuffle predictions, measure PnL distribution."""
    # Real PnL
    real_trades = []
    for day in days:
        real_trades.extend(run_strategy(day, vol_t, k, horizon, mode))
    real_pnl = sum(t["pnl_ticks"] for t in real_trades)
    real_n = len(real_trades)

    if real_n == 0:
        return {"real_pnl": 0, "n_trades": 0, "p_value": 1.0, "n_perms": n_perms}

    # Permutations
    perm_pnls = []
    rng = np.random.default_rng(42)
    for p in range(n_perms):
        perm_trades = []
        for day in days:
            perm_trades.extend(run_strategy(day, vol_t, k, horizon, mode,
                                           shuffle=True, rng=rng))
        perm_pnls.append(sum(t["pnl_ticks"] for t in perm_trades))

    perm_pnls = np.array(perm_pnls)
    p_value = float(np.mean(perm_pnls >= real_pnl))

    return {
        "real_pnl": round(float(real_pnl), 2),
        "n_trades": real_n,
        "perm_mean": round(float(np.mean(perm_pnls)), 2),
        "perm_std": round(float(np.std(perm_pnls)), 2),
        "perm_p5": round(float(np.percentile(perm_pnls, 5)), 2),
        "perm_p95": round(float(np.percentile(perm_pnls, 95)), 2),
        "p_value": round(p_value, 4),
        "n_perms": n_perms,
    }


def main():
    log.info("=== V6 DEEP DIVE: Validating Breakthrough Config ===")
    log.info("Target: both_30s_vol2.5_k20 (+1.2 ticks/trade net, p=0.000)")

    dates = get_oot_dates()
    log.info(f"Loading {len(dates)} OOT dates...")

    t0 = time.time()
    days = []
    for d in dates:
        day = DayData(d)
        if day.valid:
            days.append(day)
    log.info(f"Loaded {len(days)} valid days in {time.time()-t0:.0f}s")

    results = {}

    configs = [
        # Primary breakthrough
        ("BOTH_30s_vol2.5_k20", 2.5, 20, "30s", "both", 200),
        # Short-only and inverted
        ("SHORT_30s_vol2.5_k20", 2.5, 20, "30s", "short_only", 100),
        ("INVL_30s_vol2.5_k20", 2.5, 20, "30s", "invert_longs", 100),
        # Vol sensitivity
        ("BOTH_30s_vol2.0_k20", 2.0, 20, "30s", "both", 50),
        ("BOTH_30s_vol3.0_k20", 3.0, 20, "30s", "both", 50),
        ("BOTH_30s_vol3.5_k20", 3.5, 20, "30s", "both", 50),
        # Conviction sensitivity
        ("BOTH_30s_vol2.5_k15", 2.5, 15, "30s", "both", 50),
        ("BOTH_30s_vol2.5_k25", 2.5, 25, "30s", "both", 50),
        ("BOTH_30s_vol2.5_k30", 2.5, 30, "30s", "both", 50),
        ("BOTH_30s_vol2.5_k10", 2.5, 10, "30s", "both", 50),
        # Horizon sensitivity
        ("BOTH_10s_vol2.5_k20", 2.5, 20, "10s", "both", 50),
        ("BOTH_1s_vol2.5_k20", 2.5, 20, "1s", "both", 50),
        # Sweet spot combos
        ("BOTH_30s_vol3.0_k15", 3.0, 15, "30s", "both", 50),
        ("SHORT_30s_vol2.0_k20", 2.0, 20, "30s", "short_only", 50),
        ("SHORT_30s_vol3.0_k20", 3.0, 20, "30s", "short_only", 50),
    ]

    for name, vol_t, k, horizon, mode, n_perms in configs:
        log.info(f"\n{'='*60}")
        log.info(f"{name}: vol>{vol_t}, k={k}, {horizon}, mode={mode}")

        # Get trades
        all_trades = []
        for day in days:
            all_trades.extend(run_strategy(day, vol_t, k, horizon, mode))

        stats = compute_stats(all_trades)

        if stats["n_trades"] > 0:
            log.info(f"  {stats['n_trades']} trades ({stats['trades_per_day']}/day), "
                     f"Net: {stats['net_pnl']:+.1f}t, Avg: {stats['avg_net']:+.4f}t/tr, "
                     f"WR: {stats['win_rate']:.1%}, PF: {stats['profit_factor']:.2f}, "
                     f"Sharpe: {stats['sharpe_annual']:.2f}, Sortino: {stats['sortino_annual']:.2f}")
            log.info(f"  Green: {stats['green_days']}/{stats['total_days']}, MaxDD: {stats['max_drawdown']:.1f}t")
            log.info(f"  Longs: {stats['n_longs']} (net={stats['long_net']:+.1f}t), "
                     f"Shorts: {stats['n_shorts']} (net={stats['short_net']:+.1f}t)")

            # Permutation test
            t1 = time.time()
            perm = run_perm_test(days, vol_t, k, horizon, mode, n_perms)
            log.info(f"  Perm ({n_perms}x, {time.time()-t1:.0f}s): "
                     f"real={perm['real_pnl']:+.1f}t, rand_mean={perm['perm_mean']:+.1f}t, "
                     f"p={perm['p_value']:.4f}")

            tag = ""
            if stats['net_pnl'] > 0 and perm['p_value'] < 0.05:
                tag = " ✅ NET PROFITABLE + SIGNIFICANT"
            elif stats['net_pnl'] > 0:
                tag = " ⚠️ NET PROFITABLE but p>0.05"
            elif perm['p_value'] < 0.05:
                tag = " 📊 SIGNIFICANT EDGE but net negative"
            log.info(f"  VERDICT:{tag}")
        else:
            perm = {"real_pnl": 0, "n_trades": 0, "p_value": 1.0, "n_perms": 0}
            log.info(f"  NO TRADES")

        results[name] = {"stats": stats, "perm": perm}

        # INCREMENTAL SAVE after each config (crash protection)
        out_path = OUTPUT_DIR / "v6_deep_dive_results.json"
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=str)
        log.info(f"  [saved {len(results)} configs to disk]")

        gc.collect()

    # Final save
    out_path = OUTPUT_DIR / "v6_deep_dive_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nSaved to {out_path}")

    # Final summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY")
    log.info("=" * 70)
    profitable = []
    for name, r in results.items():
        s = r["stats"]
        p = r["perm"]
        if s["n_trades"] == 0:
            continue
        flag = ""
        if s["net_pnl"] > 0 and p["p_value"] < 0.05:
            flag = "✅"
            profitable.append(name)
        elif s["net_pnl"] > 0:
            flag = "⚠️"
        elif p["p_value"] < 0.05:
            flag = "📊"
        else:
            flag = "❌"
        log.info(f"  {flag} {name}: n={s['n_trades']}, net={s['net_pnl']:+.1f}t, "
                 f"avg={s['avg_net']:+.4f}t, Sharpe={s['sharpe_annual']:.2f}, p={p['p_value']:.4f}")

    if profitable:
        log.info(f"\n🎯 {len(profitable)} configs passed both net-positive AND permutation gates!")
    else:
        log.info(f"\n❌ No config passed both gates. Signal exists but cannot cover costs.")


if __name__ == "__main__":
    main()
