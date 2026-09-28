#!/usr/bin/env python3
"""
Time-of-Day & Volatility Regime Analysis for Tick-Level FIFO Replay
====================================================================

Runs the tick replay engine across all 34 OOT days with the best config
(TP2_SL10, threshold 0.30) and analyzes per-trade timestamps to find
profitable sub-windows by:

1. Time-of-day (30-min buckets across RTH 9:30-16:00 ET)
2. First-30-min vs rest-of-day
3. Intraday volatility regime (high vs low vol periods)
4. Combined time + vol filters

Uses the existing TickReplayEngine from engines/tick_replay_engine.py.
"""

import numpy as np
import os
import sys
import json
import time
import glob
from collections import defaultdict
from datetime import datetime, timezone, timedelta

# Add project root to path
sys.path.insert(0, '/home/jupiter/Lvl3Quant')
from engines.tick_replay_engine import (
    TickReplayEngine, Trade, compute_metrics, load_predictions,
    find_mbo_files, TICK_VALUE, PRED_STRIDE, PRED_WINDOW
)

# =============================================================================
# Constants
# =============================================================================

MBO_DIR = "/home/jupiter/Lvl3Quant/data/raw/mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/tick_level_replay"

# ET timezone offset (EST = UTC-5, EDT = UTC-4)
# Feb-Apr 2026 dates: Feb is EST, Mar 8+ is EDT
# DST 2026 starts Mar 8
EDT_OFFSET_NS = -4 * 3600 * int(1e9)
EST_OFFSET_NS = -5 * 3600 * int(1e9)
DST_START_2026 = 20260308  # March 8 2026

# RTH in ET
RTH_OPEN_HOUR = 9
RTH_OPEN_MIN = 30
RTH_CLOSE_HOUR = 16

# Configs to test
# Prediction distribution: std=0.195, p90=0.225, p95=0.27, p99=0.28
# threshold 0.20 = top 36%, 0.25 = top 8%, 0.30 = top 0.5%
CONFIGS = [
    # (tp, sl, threshold, hold_s, cancel_s, label)
    (2, 10, 0.20, 30.0, 15.0, "TP2_SL10_t20"),   # Lenient threshold, wide SL
    (2, 10, 0.25, 30.0, 15.0, "TP2_SL10_t25"),   # Moderate threshold
    (2, 10, 0.30, 30.0, 15.0, "TP2_SL10_t30"),   # Strict threshold
    (4, 10, 0.20, 30.0, 15.0, "TP4_SL10_t20"),   # Wider TP
    (4, 10, 0.25, 30.0, 15.0, "TP4_SL10_t25"),   # Wider TP, moderate threshold
    (2, 6, 0.25, 30.0, 15.0, "TP2_SL6_t25"),     # Tighter SL
    (3, 8, 0.25, 30.0, 15.0, "TP3_SL8_t25"),     # Balanced
]


def ns_to_et_hour_min(ts_ns: int, date_int: int) -> tuple:
    """Convert nanosecond UTC timestamp to (hour, minute) in ET."""
    if date_int >= DST_START_2026:
        offset = EDT_OFFSET_NS
    else:
        offset = EST_OFFSET_NS

    et_ns = ts_ns + offset
    # Extract time of day
    secs_from_midnight = (et_ns % (24 * 3600 * int(1e9))) / 1e9
    hour = int(secs_from_midnight // 3600)
    minute = int((secs_from_midnight % 3600) // 60)
    return hour, minute


def get_30min_bucket(hour: int, minute: int) -> str:
    """Return 30-min bucket label like '09:30', '10:00', etc."""
    bucket_min = 0 if minute < 30 else 30
    return f"{hour:02d}:{bucket_min:02d}"


def compute_intraday_vol(trades_for_day: list, date_int: int) -> dict:
    """
    Compute 30-min realized volatility from trade entry prices.
    Returns dict of bucket -> vol estimate (std of price changes in ticks).
    """
    # Group entry prices by 30-min bucket
    bucket_prices = defaultdict(list)
    for t in sorted(trades_for_day, key=lambda x: x.entry_time_ns):
        h, m = ns_to_et_hour_min(t.entry_time_ns, date_int)
        bucket = get_30min_bucket(h, m)
        bucket_prices[bucket].append(t.entry_price)

    vol_by_bucket = {}
    for bucket, prices in bucket_prices.items():
        if len(prices) > 2:
            diffs = np.diff(prices)
            vol_by_bucket[bucket] = float(np.std(diffs) / 0.25)  # in ticks
        else:
            vol_by_bucket[bucket] = 0.0

    return vol_by_bucket


def compute_bucket_metrics(trades: list, label: str = "") -> dict:
    """Compute metrics for a subset of trades."""
    if not trades:
        return {
            'label': label, 'n_trades': 0, 'net_pnl_ticks': 0.0,
            'win_rate': 0.0, 'profit_factor': 0.0, 'avg_pnl': 0.0,
            'avg_mfe': 0.0, 'avg_mae': 0.0, 'sharpe': 0.0,
            'pnl_per_trade': 0.0, 'n_days': 0,
        }

    pnls = np.array([t.pnl_ticks for t in trades])
    n = len(pnls)
    wins = np.sum(pnls > 0)
    losses = np.sum(pnls <= 0)
    wr = wins / n

    gross_profit = np.sum(pnls[pnls > 0]) if wins > 0 else 0
    gross_loss = abs(np.sum(pnls[pnls <= 0])) if losses > 0 else 0.001
    pf = gross_profit / gross_loss

    # Daily PnL for Sharpe
    daily_pnl = defaultdict(float)
    for t in trades:
        day_key = t.entry_time_ns // (24 * 3600 * int(1e9))
        daily_pnl[day_key] += t.pnl_ticks

    daily_returns = np.array(list(daily_pnl.values()))
    if len(daily_returns) > 1 and np.std(daily_returns) > 0:
        sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252)
    else:
        sharpe = 0.0

    return {
        'label': label,
        'n_trades': n,
        'n_days': len(daily_pnl),
        'trades_per_day': n / max(len(daily_pnl), 1),
        'net_pnl_ticks': float(np.sum(pnls)),
        'pnl_per_trade': float(np.mean(pnls)),
        'win_rate': float(wr),
        'profit_factor': float(pf),
        'avg_mfe': float(np.mean([t.mfe_ticks for t in trades])),
        'avg_mae': float(np.mean([t.mae_ticks for t in trades])),
        'sharpe': float(sharpe),
        'long_trades': sum(1 for t in trades if t.side == 'long'),
        'short_trades': sum(1 for t in trades if t.side == 'short'),
        'exit_reasons': dict(defaultdict(int, {t.exit_reason: 0 for t in trades}) |
                            {r: c for r, c in zip(*np.unique([t.exit_reason for t in trades], return_counts=True))}),
    }


def run_all_days(tp, sl, threshold, hold_s, cancel_s, predictions, mbo_files, matched_dates):
    """Run tick replay across all matched dates, return list of (trade, date_int) tuples."""
    all_trades_with_date = []

    for mbo_path, date_key in matched_dates:
        date_int = int(date_key)
        engine = TickReplayEngine(
            tp_ticks=tp, sl_ticks=sl,
            hold_seconds=hold_s,
            signal_threshold=threshold,
            cancel_seconds=cancel_s,
        )
        preds = predictions[date_key]
        t0 = time.time()
        trades = engine.run_day(mbo_path, preds)
        elapsed = time.time() - t0

        day_pnl = sum(t.pnl_ticks for t in trades)
        print(f"  {date_key}: {len(trades):3d} trades, PnL={day_pnl:+6.1f}t ({elapsed:.1f}s)")

        for t in trades:
            all_trades_with_date.append((t, date_int))

    return all_trades_with_date


def analyze_time_of_day(trades_with_date: list, config_label: str):
    """
    Analyze PnL by time-of-day in 30-min buckets.
    Also compute first-30min vs rest, and by side (long/short).
    """
    print(f"\n{'='*70}")
    print(f"TIME-OF-DAY ANALYSIS: {config_label}")
    print(f"{'='*70}")

    # Group trades by 30-min bucket
    bucket_trades = defaultdict(list)
    first_30_trades = []
    rest_trades = []

    for trade, date_int in trades_with_date:
        h, m = ns_to_et_hour_min(trade.entry_time_ns, date_int)
        bucket = get_30min_bucket(h, m)
        bucket_trades[bucket].append(trade)

        # First 30 min: 9:30-10:00
        if h == 9 and m >= 30:
            first_30_trades.append(trade)
        elif h == 10 and m < 0:  # Won't happen but be safe
            first_30_trades.append(trade)
        else:
            rest_trades.append(trade)

    # Print bucket analysis
    print(f"\n{'Bucket':<8} {'Trades':>7} {'PnL(t)':>8} {'PnL/Tr':>7} {'WR':>6} "
          f"{'PF':>6} {'MFE':>5} {'MAE':>5} {'Sharpe':>7} {'L/S':>7}")
    print("-" * 85)

    results = {}
    for bucket in sorted(bucket_trades.keys()):
        trades = bucket_trades[bucket]
        m = compute_bucket_metrics(trades, bucket)
        results[bucket] = m

        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else ">99"
        print(f"{bucket:<8} {m['n_trades']:>7} {m['net_pnl_ticks']:>8.1f} "
              f"{m['pnl_per_trade']:>7.3f} {m['win_rate']:>5.1%} "
              f"{pf_str:>6} {m['avg_mfe']:>5.1f} {m['avg_mae']:>5.1f} "
              f"{m['sharpe']:>7.2f} {m['long_trades']}/{m['short_trades']}")

    # First 30 vs rest
    print(f"\n--- FIRST 30 MIN (9:30-10:00) vs REST ---")
    m_first = compute_bucket_metrics(first_30_trades, "First 30min")
    m_rest = compute_bucket_metrics(rest_trades, "Rest of day")
    for label, m in [("First 30min", m_first), ("Rest of day", m_rest)]:
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else ">99"
        print(f"  {label:<15} trades={m['n_trades']:>5}, PnL={m['net_pnl_ticks']:>7.1f}t, "
              f"PnL/tr={m['pnl_per_trade']:>+.3f}t, WR={m['win_rate']:.1%}, PF={pf_str}, "
              f"MFE={m['avg_mfe']:.1f}, Sharpe={m['sharpe']:.2f}")

    # By side
    print(f"\n--- BY SIDE ---")
    long_trades = [t for t, _ in trades_with_date if t.side == 'long']
    short_trades = [t for t, _ in trades_with_date if t.side == 'short']
    for label, trades in [("Long", long_trades), ("Short", short_trades)]:
        m = compute_bucket_metrics(trades, label)
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else ">99"
        print(f"  {label:<15} trades={m['n_trades']:>5}, PnL={m['net_pnl_ticks']:>7.1f}t, "
              f"PnL/tr={m['pnl_per_trade']:>+.3f}t, WR={m['win_rate']:.1%}, PF={pf_str}, "
              f"MFE={m['avg_mfe']:.1f}, Sharpe={m['sharpe']:.2f}")

    # Short-only by time bucket
    print(f"\n--- SHORT-ONLY BY TIME BUCKET ---")
    short_bucket_trades = defaultdict(list)
    for trade, date_int in trades_with_date:
        if trade.side == 'short':
            h, m = ns_to_et_hour_min(trade.entry_time_ns, date_int)
            bucket = get_30min_bucket(h, m)
            short_bucket_trades[bucket].append(trade)

    print(f"{'Bucket':<8} {'Trades':>7} {'PnL(t)':>8} {'PnL/Tr':>7} {'WR':>6} {'PF':>6} {'Sharpe':>7}")
    print("-" * 55)
    for bucket in sorted(short_bucket_trades.keys()):
        trades = short_bucket_trades[bucket]
        m = compute_bucket_metrics(trades, bucket)
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else ">99"
        print(f"{bucket:<8} {m['n_trades']:>7} {m['net_pnl_ticks']:>8.1f} "
              f"{m['pnl_per_trade']:>7.3f} {m['win_rate']:>5.1%} {pf_str:>6} {m['sharpe']:>7.2f}")

    return results


def analyze_volatility(trades_with_date: list, config_label: str):
    """
    Analyze by intraday volatility regime.
    Uses the model's predicted realized vol (from the npz) and also
    a simple proxy: MFE+MAE of surrounding trades.
    """
    print(f"\n{'='*70}")
    print(f"VOLATILITY REGIME ANALYSIS: {config_label}")
    print(f"{'='*70}")

    # Group trades by date first
    by_date = defaultdict(list)
    for trade, date_int in trades_with_date:
        by_date[date_int].append(trade)

    # Compute daily vol as avg(MFE+MAE) per day
    day_vol = {}
    for date_int, trades in by_date.items():
        avg_excursion = np.mean([t.mfe_ticks + t.mae_ticks for t in trades])
        day_vol[date_int] = avg_excursion

    # Split into high/low vol days (median split)
    vol_values = list(day_vol.values())
    vol_median = np.median(vol_values)
    vol_p25 = np.percentile(vol_values, 25)
    vol_p75 = np.percentile(vol_values, 75)

    print(f"\nDaily vol (avg MFE+MAE): median={vol_median:.2f}t, "
          f"p25={vol_p25:.2f}t, p75={vol_p75:.2f}t")

    high_vol_trades = []
    low_vol_trades = []
    very_high_vol_trades = []
    very_low_vol_trades = []

    for trade, date_int in trades_with_date:
        v = day_vol[date_int]
        if v >= vol_median:
            high_vol_trades.append(trade)
        else:
            low_vol_trades.append(trade)
        if v >= vol_p75:
            very_high_vol_trades.append(trade)
        if v <= vol_p25:
            very_low_vol_trades.append(trade)

    print(f"\n--- BY DAILY VOL REGIME ---")
    for label, trades in [
        ("Low Vol (< median)", low_vol_trades),
        ("High Vol (>= median)", high_vol_trades),
        ("Very Low Vol (< p25)", very_low_vol_trades),
        ("Very High Vol (> p75)", very_high_vol_trades),
    ]:
        m = compute_bucket_metrics(trades, label)
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else ">99"
        print(f"  {label:<25} trades={m['n_trades']:>5}, PnL={m['net_pnl_ticks']:>7.1f}t, "
              f"PnL/tr={m['pnl_per_trade']:>+.3f}t, WR={m['win_rate']:.1%}, PF={pf_str}, "
              f"MFE={m['avg_mfe']:.1f}, MAE={m['avg_mae']:.1f}, Sharpe={m['sharpe']:.2f}")

    # Per-trade vol proxy: use individual trade's MFE+MAE
    all_excursions = [t.mfe_ticks + t.mae_ticks for t, _ in trades_with_date]
    exc_median = np.median(all_excursions)
    exc_p75 = np.percentile(all_excursions, 75)

    print(f"\n--- BY PER-TRADE EXCURSION (MFE+MAE) ---")
    print(f"  Trade excursion median={exc_median:.2f}t, p75={exc_p75:.2f}t")

    high_exc = [t for t, _ in trades_with_date if t.mfe_ticks + t.mae_ticks >= exc_p75]
    low_exc = [t for t, _ in trades_with_date if t.mfe_ticks + t.mae_ticks < exc_median]
    mid_exc = [t for t, _ in trades_with_date
               if exc_median <= t.mfe_ticks + t.mae_ticks < exc_p75]

    for label, trades in [
        ("Low excursion (<median)", low_exc),
        ("Mid excursion", mid_exc),
        ("High excursion (>p75)", high_exc),
    ]:
        m = compute_bucket_metrics(trades, label)
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else ">99"
        print(f"  {label:<25} trades={m['n_trades']:>5}, PnL={m['net_pnl_ticks']:>7.1f}t, "
              f"PnL/tr={m['pnl_per_trade']:>+.3f}t, WR={m['win_rate']:.1%}, PF={pf_str}, "
              f"MFE={m['avg_mfe']:.1f}, MAE={m['avg_mae']:.1f}")

    return day_vol


def analyze_signal_strength_buckets(trades_with_date: list, config_label: str):
    """Analyze by signal strength quantile."""
    print(f"\n{'='*70}")
    print(f"SIGNAL STRENGTH ANALYSIS: {config_label}")
    print(f"{'='*70}")

    strengths = [abs(t.signal_strength) for t, _ in trades_with_date]
    p50 = np.percentile(strengths, 50)
    p75 = np.percentile(strengths, 75)
    p90 = np.percentile(strengths, 90)

    print(f"Signal |strength| percentiles: p50={p50:.4f}, p75={p75:.4f}, p90={p90:.4f}")

    buckets = [
        ("Bottom 50%", [t for t, _ in trades_with_date if abs(t.signal_strength) < p50]),
        ("50-75%", [t for t, _ in trades_with_date if p50 <= abs(t.signal_strength) < p75]),
        ("75-90%", [t for t, _ in trades_with_date if p75 <= abs(t.signal_strength) < p90]),
        ("Top 10%", [t for t, _ in trades_with_date if abs(t.signal_strength) >= p90]),
    ]

    print(f"\n{'Quantile':<15} {'Trades':>7} {'PnL(t)':>8} {'PnL/Tr':>7} {'WR':>6} "
          f"{'PF':>6} {'MFE':>5} {'MAE':>5} {'Sharpe':>7}")
    print("-" * 75)

    for label, trades in buckets:
        m = compute_bucket_metrics(trades, label)
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else ">99"
        print(f"{label:<15} {m['n_trades']:>7} {m['net_pnl_ticks']:>8.1f} "
              f"{m['pnl_per_trade']:>7.3f} {m['win_rate']:>5.1%} {pf_str:>6} "
              f"{m['avg_mfe']:>5.1f} {m['avg_mae']:>5.1f} {m['sharpe']:>7.2f}")

    # Top 10% signal by side
    print(f"\n--- TOP 10% SIGNAL BY SIDE ---")
    top10_long = [t for t, _ in trades_with_date
                  if abs(t.signal_strength) >= p90 and t.side == 'long']
    top10_short = [t for t, _ in trades_with_date
                   if abs(t.signal_strength) >= p90 and t.side == 'short']
    for label, trades in [("Top10% Long", top10_long), ("Top10% Short", top10_short)]:
        m = compute_bucket_metrics(trades, label)
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] < 100 else ">99"
        print(f"  {label:<20} trades={m['n_trades']:>5}, PnL={m['net_pnl_ticks']:>7.1f}t, "
              f"PnL/tr={m['pnl_per_trade']:>+.3f}t, WR={m['win_rate']:.1%}, PF={pf_str}")


def analyze_combined_filters(trades_with_date: list, day_vol: dict, config_label: str):
    """
    Find the best combined time + vol + side filter.
    This is the money shot — can we find a sub-window where PF > 1.0?
    """
    print(f"\n{'='*70}")
    print(f"COMBINED FILTER SEARCH: {config_label}")
    print(f"{'='*70}")

    vol_median = np.median(list(day_vol.values()))

    # Signal strength percentiles
    strengths = [abs(t.signal_strength) for t, _ in trades_with_date]
    sig_p75 = np.percentile(strengths, 75)
    sig_p90 = np.percentile(strengths, 90)

    # Define filter combinations
    filters = []

    # Time windows
    time_windows = {
        'open_30m': lambda h, m: (h == 9 and m >= 30) or (h == 10 and m < 0),
        'open_60m': lambda h, m: (h == 9 and m >= 30) or (h == 10),
        'morning': lambda h, m: h < 12,
        'midday': lambda h, m: 11 <= h <= 13,
        'afternoon': lambda h, m: h >= 13,
        'last_hour': lambda h, m: h >= 15,
        'core_10_14': lambda h, m: 10 <= h < 14,
        'all_day': lambda h, m: True,
    }

    # Vol filters
    vol_filters = {
        'all_vol': lambda d: True,
        'high_vol': lambda d: day_vol.get(d, 0) >= vol_median,
        'low_vol': lambda d: day_vol.get(d, 0) < vol_median,
    }

    # Side filters
    side_filters = {
        'both': lambda t: True,
        'short_only': lambda t: t.side == 'short',
        'long_only': lambda t: t.side == 'long',
    }

    # Signal filters
    sig_filters = {
        'all_sig': lambda t: True,
        'top25_sig': lambda t: abs(t.signal_strength) >= sig_p75,
        'top10_sig': lambda t: abs(t.signal_strength) >= sig_p90,
    }

    results = []

    for tw_name, tw_fn in time_windows.items():
        for vf_name, vf_fn in vol_filters.items():
            for sf_name, sf_fn in side_filters.items():
                for sgf_name, sgf_fn in sig_filters.items():
                    filtered = []
                    for trade, date_int in trades_with_date:
                        h, m = ns_to_et_hour_min(trade.entry_time_ns, date_int)
                        if tw_fn(h, m) and vf_fn(date_int) and sf_fn(trade) and sgf_fn(trade):
                            filtered.append(trade)

                    if len(filtered) >= 20:  # Need meaningful sample
                        m = compute_bucket_metrics(filtered,
                            f"{tw_name}|{vf_name}|{sf_name}|{sgf_name}")
                        m['filter'] = f"{tw_name}|{vf_name}|{sf_name}|{sgf_name}"
                        results.append(m)

    # Sort by profit factor, show top 20
    results.sort(key=lambda x: x['profit_factor'], reverse=True)

    print(f"\n--- TOP 20 FILTER COMBOS (PF desc, min 20 trades) ---")
    print(f"{'Filter':<50} {'Trades':>6} {'PnL(t)':>8} {'PnL/Tr':>7} {'WR':>6} "
          f"{'PF':>6} {'Sharpe':>7} {'Days':>5}")
    print("-" * 105)

    for r in results[:20]:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else ">99"
        print(f"{r['filter']:<50} {r['n_trades']:>6} {r['net_pnl_ticks']:>8.1f} "
              f"{r['pnl_per_trade']:>7.3f} {r['win_rate']:>5.1%} {pf_str:>6} "
              f"{r['sharpe']:>7.2f} {r['n_days']:>5}")

    # Also show results sorted by Sharpe
    results.sort(key=lambda x: x['sharpe'], reverse=True)
    print(f"\n--- TOP 20 FILTER COMBOS (Sharpe desc, min 20 trades) ---")
    print(f"{'Filter':<50} {'Trades':>6} {'PnL(t)':>8} {'PnL/Tr':>7} {'WR':>6} "
          f"{'PF':>6} {'Sharpe':>7} {'Days':>5}")
    print("-" * 105)

    for r in results[:20]:
        pf_str = f"{r['profit_factor']:.2f}" if r['profit_factor'] < 100 else ">99"
        print(f"{r['filter']:<50} {r['n_trades']:>6} {r['net_pnl_ticks']:>8.1f} "
              f"{r['pnl_per_trade']:>7.3f} {r['win_rate']:>5.1%} {pf_str:>6} "
              f"{r['sharpe']:>7.2f} {r['n_days']:>5}")

    # Find profitable combos (PF > 1.0 with 50+ trades)
    profitable = [r for r in results if r['profit_factor'] > 1.0 and r['n_trades'] >= 50]
    if profitable:
        profitable.sort(key=lambda x: x['n_trades'], reverse=True)
        print(f"\n*** PROFITABLE FILTERS (PF > 1.0, >= 50 trades) ***")
        for r in profitable[:10]:
            print(f"  {r['filter']}")
            print(f"    trades={r['n_trades']}, PnL={r['net_pnl_ticks']:.1f}t, "
                  f"PnL/tr={r['pnl_per_trade']:+.3f}t, WR={r['win_rate']:.1%}, "
                  f"PF={r['profit_factor']:.2f}, Sharpe={r['sharpe']:.2f}, "
                  f"days={r['n_days']}")
    else:
        # Relax to 30 trades
        profitable_30 = [r for r in results if r['profit_factor'] > 1.0 and r['n_trades'] >= 30]
        if profitable_30:
            profitable_30.sort(key=lambda x: x['n_trades'], reverse=True)
            print(f"\n*** PROFITABLE FILTERS (PF > 1.0, >= 30 trades) ***")
            for r in profitable_30[:10]:
                print(f"  {r['filter']}")
                print(f"    trades={r['n_trades']}, PnL={r['net_pnl_ticks']:.1f}t, "
                      f"PnL/tr={r['pnl_per_trade']:+.3f}t, WR={r['win_rate']:.1%}, "
                      f"PF={r['profit_factor']:.2f}, Sharpe={r['sharpe']:.2f}, "
                      f"days={r['n_days']}")
        else:
            print(f"\n*** NO FILTER COMBO ACHIEVES PF > 1.0 WITH >= 30 TRADES ***")
            # Show nearest misses
            near_misses = [r for r in results if r['n_trades'] >= 30]
            near_misses.sort(key=lambda x: x['profit_factor'], reverse=True)
            print(f"Nearest misses (top 5):")
            for r in near_misses[:5]:
                print(f"  {r['filter']}: PF={r['profit_factor']:.3f}, "
                      f"trades={r['n_trades']}, PnL/tr={r['pnl_per_trade']:+.4f}t")

    return results


def main():
    print("=" * 70)
    print("TICK REPLAY TIME-OF-DAY & VOLATILITY ANALYSIS")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Load data
    print(f"\nLoading predictions from {PRED_DIR}...")
    predictions = load_predictions(PRED_DIR)
    print(f"  Loaded {len(predictions)} dates")

    mbo_files = find_mbo_files(MBO_DIR)
    print(f"  Found {len(mbo_files)} MBO files")

    # Match dates
    matched_dates = []
    for mbo_path in mbo_files:
        date8 = os.path.basename(mbo_path).split('-')[2].split('.')[0]
        if date8 in predictions:
            matched_dates.append((mbo_path, date8))
    matched_dates.sort(key=lambda x: x[1])

    print(f"  Matched {len(matched_dates)} dates")
    if not matched_dates:
        print("ERROR: No matching dates!")
        return

    all_results = {}

    # Run each config
    for tp, sl, threshold, hold_s, cancel_s, label in CONFIGS:
        print(f"\n{'#'*70}")
        print(f"RUNNING CONFIG: {label} (TP={tp}, SL={sl}, threshold={threshold})")
        print(f"{'#'*70}")

        t0 = time.time()
        trades_with_date = run_all_days(tp, sl, threshold, hold_s, cancel_s,
                                         predictions, mbo_files, matched_dates)
        elapsed = time.time() - t0

        total_trades = len(trades_with_date)
        total_pnl = sum(t.pnl_ticks for t, _ in trades_with_date)
        print(f"\n  TOTAL: {total_trades} trades, PnL={total_pnl:.1f}t, time={elapsed:.0f}s")

        if total_trades < 10:
            print(f"  SKIPPING analysis — too few trades")
            continue

        # Run analyses
        tod_results = analyze_time_of_day(trades_with_date, label)
        day_vol = analyze_volatility(trades_with_date, label)
        analyze_signal_strength_buckets(trades_with_date, label)
        combined_results = analyze_combined_filters(trades_with_date, day_vol, label)

        all_results[label] = {
            'total_trades': total_trades,
            'total_pnl_ticks': total_pnl,
            'time_of_day': {k: v for k, v in tod_results.items()},
            'top_filters': [r for r in combined_results[:20]],
        }

    # Save results
    output_path = os.path.join(OUTPUT_DIR, 'time_vol_analysis_results.json')

    # Convert for JSON serialization
    def make_serializable(obj):
        if isinstance(obj, (np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, (np.float64, np.float32)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        return obj

    with open(output_path, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2)
    print(f"\nResults saved to {output_path}")

    # Final summary
    print(f"\n{'='*70}")
    print("EXECUTIVE SUMMARY")
    print(f"{'='*70}")

    for label, res in all_results.items():
        print(f"\n{label}:")
        print(f"  Total: {res['total_trades']} trades, PnL={res['total_pnl_ticks']:.1f}t")

        # Find best time bucket
        if res['time_of_day']:
            best_bucket = max(res['time_of_day'].items(),
                            key=lambda x: x[1].get('profit_factor', 0))
            print(f"  Best time bucket: {best_bucket[0]} "
                  f"(PF={best_bucket[1]['profit_factor']:.2f}, "
                  f"trades={best_bucket[1]['n_trades']}, "
                  f"PnL/tr={best_bucket[1]['pnl_per_trade']:+.3f}t)")

        # Best combined filter
        if res['top_filters']:
            best = res['top_filters'][0]
            print(f"  Best filter: {best['filter']}")
            print(f"    PF={best['profit_factor']:.2f}, trades={best['n_trades']}, "
                  f"Sharpe={best['sharpe']:.2f}")


if __name__ == '__main__':
    main()
