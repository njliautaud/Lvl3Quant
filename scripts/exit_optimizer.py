#!/usr/bin/env python3
"""
Exit Strategy Optimizer for Agentic Options Trading
====================================================

Uses historical options chain data (Dolt cache, ~1200 snapshots per ETF, 2019-2026)
to simulate and compare exit strategies for single-leg options positions.

Answers:
  1. Is +30% TP optimal, or would +20% / +40% capture more total profit?
  2. How often do options that hit +30% continue to +50%+? (left on table)
  3. How often do options that are up +20% reverse to negative? (trailing calibration)
  4. What trailing stop parameters maximize risk-adjusted returns?
  5. Does optimal exit differ by signal type (momentum vs mean-reversion)?

Data source: /home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/options_real/chains/
Output:      /home/jupiter/Lvl3Quant/state/exit_optimization_results.json

Usage:
  python3 exit_optimizer.py                 # Run full analysis on all ETFs
  python3 exit_optimizer.py --ticker XLU    # Single ticker
  python3 exit_optimizer.py --quick         # Quick mode (subset of dates)
"""

import argparse
import json
import sys
import warnings
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

_print = print
def print(*args, **kwargs):
    _print(*args, **kwargs, flush=True)


# ── CONFIG ──────────────────────────────────────────────────────────────────

BASE = Path("/home/jupiter/Lvl3Quant")
CHAINS_DIR = BASE / "wheel_strategy_v1" / "data" / "cache" / "options_real" / "chains"
OUTPUT_FILE = BASE / "state" / "exit_optimization_results.json"

SECTOR_ETFS = ["XLF", "XLE", "XLU", "XLK", "XLV", "XLY", "XLP", "XLI", "XLB", "XLRE", "XLC"]

# Trade entry criteria: OTM puts/calls with 30-45 DTE, delta 0.20-0.40
MIN_DTE_ENTRY = 25
MAX_DTE_ENTRY = 50
MIN_ABS_DELTA = 0.15
MAX_ABS_DELTA = 0.45
MIN_OPTION_PRICE = 0.10  # minimum mid price to avoid illiquid junk
MAX_HOLD_DAYS = 15  # max calendar days to track a position

# Commission per contract (buy + sell)
COMMISSION_RT = 1.30  # $0.65 per leg on Robinhood options (actually free, but model small cost)

# Exit strategies to test
FIXED_TP_LEVELS = [0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.50, 0.60, 0.75, 1.00]
FIXED_SL_LEVELS = [-0.15, -0.20, -0.25, -0.30, -0.40, -0.50]
TRAILING_CONFIGS = [
    # (activation_pct, giveback_pct) -- activate trailing when up X%, close when drops Y% from peak
    (0.10, 0.30),
    (0.10, 0.50),
    (0.15, 0.30),
    (0.15, 0.50),
    (0.20, 0.30),
    (0.20, 0.50),
    (0.25, 0.40),
    (0.25, 0.50),
    (0.30, 0.40),
    (0.30, 0.50),
]
TIME_STOP_DAYS = [3, 5, 7, 10]


# ── DATA LOADING ────────────────────────────────────────────────────────────

def load_chain_data(ticker: str) -> pd.DataFrame:
    """Load and preprocess chain data for a ticker."""
    path = CHAINS_DIR / f"{ticker}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"No chain data for {ticker} at {path}")

    df = pd.read_parquet(path)
    df["date"] = pd.to_datetime(df["date"])
    df["expiration"] = pd.to_datetime(df["expiration"])

    # Filter to reasonable options
    df = df[df["mid"] > MIN_OPTION_PRICE].copy()
    df = df[df["bid"] > 0].copy()  # must have a bid (can actually sell it)

    return df


def estimate_underlying_price(df: pd.DataFrame, date, opt_type: str = "p") -> float:
    """Estimate underlying price from ATM options (strike nearest to where delta ~ 0.5)."""
    day_data = df[(df["date"] == date) & (df["type"] == opt_type)]
    if len(day_data) == 0:
        return np.nan

    # ATM option has |delta| closest to 0.5
    day_data = day_data.copy()
    day_data["abs_delta_diff"] = abs(abs(day_data["delta"]) - 0.5)
    atm_row = day_data.loc[day_data["abs_delta_diff"].idxmin()]
    return float(atm_row["strike"])


# ── TRADE SIMULATION ────────────────────────────────────────────────────────

def find_entry_candidates(df: pd.DataFrame, snapshot_dates: list) -> list:
    """
    Find all valid entry points: OTM options with 30-45 DTE and delta in range.
    Returns list of dicts with entry info.
    """
    entries = []

    for date in snapshot_dates:
        day_data = df[df["date"] == date]
        if len(day_data) == 0:
            continue

        for opt_type in ["p", "c"]:
            type_data = day_data[
                (day_data["type"] == opt_type) &
                (day_data["dte"].between(MIN_DTE_ENTRY, MAX_DTE_ENTRY)) &
                (abs(day_data["delta"]).between(MIN_ABS_DELTA, MAX_ABS_DELTA))
            ]

            for _, row in type_data.iterrows():
                entries.append({
                    "entry_date": date,
                    "expiration": row["expiration"],
                    "strike": row["strike"],
                    "opt_type": opt_type,
                    "entry_mid": row["mid"],
                    "entry_bid": row["bid"],
                    "entry_ask": row["ask"],
                    "entry_delta": row["delta"],
                    "entry_dte": row["dte"],
                    "entry_iv": row["vol"],
                })

    return entries


def track_position(df: pd.DataFrame, entry: dict, max_days: int = MAX_HOLD_DAYS) -> list:
    """
    Track option price over subsequent days after entry.
    Returns list of daily snapshots [{day: N, mid: X, bid: X, ask: X, pnl_pct: X}, ...]
    """
    entry_date = entry["entry_date"]
    expiration = entry["expiration"]
    strike = entry["strike"]
    opt_type = entry["opt_type"]
    entry_mid = entry["entry_mid"]

    # Find all subsequent dates for this specific option contract
    future_data = df[
        (df["date"] > entry_date) &
        (df["date"] <= entry_date + pd.Timedelta(days=max_days + 5)) &  # buffer for weekends
        (df["expiration"] == expiration) &
        (df["strike"] == strike) &
        (df["type"] == opt_type)
    ].sort_values("date")

    path = []
    for _, row in future_data.iterrows():
        days_held = (row["date"] - entry_date).days
        if days_held > max_days:
            break
        trading_days = len(df[(df["date"] > entry_date) & (df["date"] <= row["date"])]["date"].unique())
        # Use midpoint for fair value tracking
        mid = row["mid"]
        pnl_pct = (mid - entry_mid) / entry_mid

        path.append({
            "day": days_held,
            "trading_day": min(trading_days, days_held),  # approximate
            "date": row["date"],
            "mid": mid,
            "bid": row["bid"],
            "ask": row["ask"],
            "pnl_pct": pnl_pct,
            "iv": row["vol"],
            "delta": row["delta"],
            "dte": row["dte"],
        })

    return path


def simulate_exit(path: list, entry_mid: float, tp: float = None, sl: float = None,
                  trailing_activate: float = None, trailing_giveback: float = None,
                  time_stop_days: int = None) -> dict:
    """
    Simulate exit strategy on a price path.

    Args:
        path: list of daily snapshots from track_position
        entry_mid: entry price
        tp: take-profit threshold as fraction (e.g., 0.30 = +30%)
        sl: stop-loss threshold as fraction (e.g., -0.25 = -25%)
        trailing_activate: activate trailing stop when gain >= this (e.g., 0.15)
        trailing_giveback: close when price drops this fraction from peak (e.g., 0.50)
        time_stop_days: close after N calendar days regardless

    Returns:
        dict with exit info
    """
    if not path:
        return {"exit_reason": "no_data", "exit_pnl_pct": 0, "exit_day": 0, "peak_pnl_pct": 0}

    peak_mid = entry_mid
    peak_pnl = 0.0
    trailing_active = False

    for snap in path:
        mid = snap["mid"]
        pnl_pct = snap["pnl_pct"]

        # Track peak
        if mid > peak_mid:
            peak_mid = mid
            peak_pnl = pnl_pct

        # Check exits in priority order

        # 1. Take Profit
        if tp is not None and pnl_pct >= tp:
            return {
                "exit_reason": "take_profit",
                "exit_pnl_pct": pnl_pct,
                "exit_day": snap["day"],
                "peak_pnl_pct": peak_pnl,
                "exit_price": mid,
            }

        # 2. Stop Loss
        if sl is not None and pnl_pct <= sl:
            return {
                "exit_reason": "stop_loss",
                "exit_pnl_pct": pnl_pct,
                "exit_day": snap["day"],
                "peak_pnl_pct": peak_pnl,
                "exit_price": mid,
            }

        # 3. Trailing Stop
        if trailing_activate is not None and trailing_giveback is not None:
            if not trailing_active and pnl_pct >= trailing_activate:
                trailing_active = True

            if trailing_active and peak_pnl > 0:
                giveback_from_peak = 1.0 - (mid / peak_mid)
                if giveback_from_peak >= trailing_giveback:
                    return {
                        "exit_reason": "trailing_stop",
                        "exit_pnl_pct": pnl_pct,
                        "exit_day": snap["day"],
                        "peak_pnl_pct": peak_pnl,
                        "exit_price": mid,
                    }

        # 4. Time Stop
        if time_stop_days is not None and snap["day"] >= time_stop_days:
            return {
                "exit_reason": "time_stop",
                "exit_pnl_pct": pnl_pct,
                "exit_day": snap["day"],
                "peak_pnl_pct": peak_pnl,
                "exit_price": mid,
            }

    # Held to end of tracking window
    last = path[-1]
    return {
        "exit_reason": "hold_to_end",
        "exit_pnl_pct": last["pnl_pct"],
        "exit_day": last["day"],
        "peak_pnl_pct": peak_pnl,
        "exit_price": last["mid"],
    }


# ── ANALYSIS FUNCTIONS ──────────────────────────────────────────────────────

def compute_strategy_metrics(results: list) -> dict:
    """Compute risk-adjusted metrics from a list of trade results."""
    if not results:
        return {"n": 0}

    pnls = [r["exit_pnl_pct"] for r in results]
    pnls = np.array(pnls)
    n = len(pnls)

    wins = pnls > 0
    win_rate = wins.mean()
    avg_win = pnls[wins].mean() if wins.any() else 0
    avg_loss = pnls[~wins].mean() if (~wins).any() else 0
    avg_pnl = pnls.mean()
    std_pnl = pnls.std() if n > 1 else 0

    # Sharpe (annualized assuming ~50 trades/year)
    sharpe = (avg_pnl / std_pnl) * np.sqrt(50) if std_pnl > 0 else 0

    # Sortino
    downside = pnls[pnls < 0]
    downside_std = downside.std() if len(downside) > 1 else std_pnl
    sortino = (avg_pnl / downside_std) * np.sqrt(50) if downside_std > 0 else 0

    # Profit Factor
    gross_profit = pnls[wins].sum() if wins.any() else 0
    gross_loss = abs(pnls[~wins].sum()) if (~wins).any() else 0.001
    profit_factor = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Average hold time
    avg_hold = np.mean([r["exit_day"] for r in results])

    # Peak P&L left on table
    peak_pnls = [r["peak_pnl_pct"] for r in results]
    avg_peak = np.mean(peak_pnls)
    left_on_table = np.mean([r["peak_pnl_pct"] - r["exit_pnl_pct"] for r in results])

    # Exit reason distribution
    reasons = {}
    for r in results:
        reason = r["exit_reason"]
        reasons[reason] = reasons.get(reason, 0) + 1

    return {
        "n": n,
        "win_rate": round(win_rate, 4),
        "avg_pnl_pct": round(avg_pnl, 4),
        "avg_win_pct": round(avg_win, 4),
        "avg_loss_pct": round(avg_loss, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(min(profit_factor, 99.9), 3),
        "avg_hold_days": round(avg_hold, 1),
        "avg_peak_pnl_pct": round(avg_peak, 4),
        "avg_left_on_table_pct": round(left_on_table, 4),
        "exit_reasons": reasons,
    }


def run_continuation_analysis(all_paths: list, all_entries: list) -> dict:
    """
    Key question: How often do options that hit +X% continue to +Y%?
    And how often do they reverse to negative?
    """
    thresholds = [0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
    results = {}

    for thresh in thresholds:
        # Track what happens AFTER hitting this threshold
        hit_count = 0
        continued_higher = {t: 0 for t in thresholds if t > thresh}
        reversed_negative = 0
        reversed_half = 0  # gave back 50%+ of gains
        max_after_hit = []

        for path, entry in zip(all_paths, all_entries):
            entry_mid = entry["entry_mid"]
            hit_idx = None

            for i, snap in enumerate(path):
                if snap["pnl_pct"] >= thresh:
                    hit_idx = i
                    break

            if hit_idx is None:
                continue

            hit_count += 1

            # Track what happens after hitting threshold
            post_hit = path[hit_idx:]
            post_pnls = [s["pnl_pct"] for s in post_hit]
            max_post = max(post_pnls) if post_pnls else thresh
            min_post = min(post_pnls) if post_pnls else thresh
            final_pnl = post_pnls[-1] if post_pnls else thresh

            max_after_hit.append(max_post)

            for higher_thresh in continued_higher:
                if max_post >= higher_thresh:
                    continued_higher[higher_thresh] += 1

            if final_pnl < 0:
                reversed_negative += 1
            if final_pnl < thresh * 0.5:
                reversed_half += 1

        if hit_count > 0:
            results[f"+{int(thresh*100)}%"] = {
                "n_hit": hit_count,
                "pct_of_all_trades": round(hit_count / len(all_paths), 4),
                "continued_higher": {
                    f"+{int(t*100)}%": round(v / hit_count, 4) for t, v in continued_higher.items()
                },
                "reversed_negative": round(reversed_negative / hit_count, 4),
                "reversed_half_gains": round(reversed_half / hit_count, 4),
                "avg_max_after_hit": round(np.mean(max_after_hit), 4) if max_after_hit else 0,
            }

    return results


def run_reversal_analysis(all_paths: list) -> dict:
    """
    How often do profitable positions reverse?
    For positions that are up +X%, what's the probability of ending negative?
    """
    gain_levels = [0.05, 0.10, 0.15, 0.20, 0.25, 0.30]
    results = {}

    for gain in gain_levels:
        hit_count = 0
        ended_negative = 0
        ended_below_half = 0
        final_pnls = []

        for path in all_paths:
            hit = False
            for snap in path:
                if snap["pnl_pct"] >= gain:
                    hit = True
                    break

            if not hit:
                continue

            hit_count += 1
            final_pnl = path[-1]["pnl_pct"]
            final_pnls.append(final_pnl)

            if final_pnl < 0:
                ended_negative += 1
            if final_pnl < gain * 0.5:
                ended_below_half += 1

        if hit_count > 0:
            results[f"+{int(gain*100)}%"] = {
                "n_hit": hit_count,
                "pct_ended_negative": round(ended_negative / hit_count, 4),
                "pct_ended_below_half": round(ended_below_half / hit_count, 4),
                "avg_final_pnl": round(np.mean(final_pnls), 4),
                "median_final_pnl": round(np.median(final_pnls), 4),
            }

    return results


def classify_signal_type(entry: dict) -> str:
    """
    Classify entry as momentum or mean-reversion based on delta/IV.
    High-delta entries (close to ATM) with low IV = momentum-like.
    Low-delta entries (OTM) with high IV = mean-reversion-like.
    """
    delta = abs(entry.get("entry_delta", 0.3))
    iv = entry.get("entry_iv", 0.3)

    if delta >= 0.30 and iv < 0.35:
        return "momentum"
    elif delta < 0.25 and iv >= 0.35:
        return "mean_reversion"
    else:
        return "mixed"


# ── MAIN OPTIMIZER ──────────────────────────────────────────────────────────

def optimize_for_ticker(ticker: str, quick: bool = False) -> dict:
    """Run full exit optimization for a single ticker."""
    print(f"\n{'='*70}")
    print(f"OPTIMIZING EXIT STRATEGY: {ticker}")
    print(f"{'='*70}")

    df = load_chain_data(ticker)
    snapshot_dates = sorted(df["date"].unique())
    print(f"  Loaded {len(df)} rows, {len(snapshot_dates)} snapshot dates")
    print(f"  Date range: {snapshot_dates[0].date()} to {snapshot_dates[-1].date()}")

    if quick:
        # Use every 5th date for speed
        snapshot_dates = snapshot_dates[::5]
        print(f"  Quick mode: using {len(snapshot_dates)} dates")

    # Find entry candidates
    print("  Finding entry candidates...")
    entries = find_entry_candidates(df, snapshot_dates)
    print(f"  Found {len(entries)} valid entry candidates")

    if len(entries) < 20:
        print(f"  WARNING: Too few entries for {ticker}. Skipping.")
        return {"ticker": ticker, "error": "too_few_entries", "n_entries": len(entries)}

    # Track price paths for all entries
    print("  Tracking position price paths...")
    paths = []
    valid_entries = []
    for i, entry in enumerate(entries):
        if i % 500 == 0 and i > 0:
            print(f"    ...tracked {i}/{len(entries)} positions")
        path = track_position(df, entry)
        if len(path) >= 2:  # need at least 2 days of data
            paths.append(path)
            valid_entries.append(entry)

    print(f"  {len(paths)} positions with trackable paths (out of {len(entries)})")

    if len(paths) < 20:
        print(f"  WARNING: Too few trackable paths for {ticker}. Skipping.")
        return {"ticker": ticker, "error": "too_few_paths", "n_paths": len(paths)}

    # ── 1. BASELINE: No exit (hold to end of window) ──
    print("\n  Running baseline (hold to end)...")
    baseline_results = [simulate_exit(p, e["entry_mid"]) for p, e in zip(paths, valid_entries)]
    baseline_metrics = compute_strategy_metrics(baseline_results)
    print(f"    Baseline: WR={baseline_metrics['win_rate']:.1%}, Avg={baseline_metrics['avg_pnl_pct']:.2%}, Sharpe={baseline_metrics['sharpe']:.2f}")

    # ── 2. FIXED TAKE PROFIT SWEEP ──
    print("\n  Running fixed TP sweep...")
    tp_results = {}
    for tp in FIXED_TP_LEVELS:
        results = [simulate_exit(p, e["entry_mid"], tp=tp) for p, e in zip(paths, valid_entries)]
        metrics = compute_strategy_metrics(results)
        tp_results[f"+{int(tp*100)}%"] = metrics
        print(f"    TP +{int(tp*100)}%: WR={metrics['win_rate']:.1%}, Avg={metrics['avg_pnl_pct']:.2%}, Sharpe={metrics['sharpe']:.2f}, PF={metrics['profit_factor']:.2f}")

    # ── 3. FIXED STOP LOSS SWEEP ──
    print("\n  Running fixed SL sweep...")
    sl_results = {}
    for sl in FIXED_SL_LEVELS:
        results = [simulate_exit(p, e["entry_mid"], sl=sl) for p, e in zip(paths, valid_entries)]
        metrics = compute_strategy_metrics(results)
        sl_results[f"{int(sl*100)}%"] = metrics
        print(f"    SL {int(sl*100)}%: WR={metrics['win_rate']:.1%}, Avg={metrics['avg_pnl_pct']:.2%}, Sharpe={metrics['sharpe']:.2f}")

    # ── 4. COMBINED TP + SL SWEEP (key combos only) ──
    print("\n  Running TP+SL combined sweep...")
    combined_results = {}
    for tp in [0.20, 0.25, 0.30, 0.40, 0.50]:
        for sl in [-0.20, -0.25, -0.30]:
            key = f"TP+{int(tp*100)}/SL{int(sl*100)}"
            results = [simulate_exit(p, e["entry_mid"], tp=tp, sl=sl) for p, e in zip(paths, valid_entries)]
            metrics = compute_strategy_metrics(results)
            combined_results[key] = metrics

    # Find best combined strategy
    best_combined = max(combined_results.items(), key=lambda x: x[1].get("sharpe", 0))
    print(f"    Best combined: {best_combined[0]} → Sharpe={best_combined[1]['sharpe']:.2f}, WR={best_combined[1]['win_rate']:.1%}")

    # ── 5. TRAILING STOP SWEEP ──
    print("\n  Running trailing stop sweep...")
    trailing_results = {}
    for act, gb in TRAILING_CONFIGS:
        key = f"trail_act{int(act*100)}_gb{int(gb*100)}"
        results = [
            simulate_exit(p, e["entry_mid"], trailing_activate=act, trailing_giveback=gb)
            for p, e in zip(paths, valid_entries)
        ]
        metrics = compute_strategy_metrics(results)
        trailing_results[key] = metrics

    best_trailing = max(trailing_results.items(), key=lambda x: x[1].get("sharpe", 0))
    print(f"    Best trailing: {best_trailing[0]} → Sharpe={best_trailing[1]['sharpe']:.2f}")

    # ── 6. FULL COMBO: TP + SL + TRAILING + TIME STOP ──
    print("\n  Running full combo optimization...")
    full_combo_results = {}

    # Test best TP/SL combos WITH trailing and time stops
    for tp in [0.25, 0.30, 0.40]:
        for sl in [-0.25, -0.30]:
            for act, gb in [(0.15, 0.50), (0.20, 0.40), (0.20, 0.50), (0.25, 0.50)]:
                for ts in [5, 7, 10]:
                    key = f"TP{int(tp*100)}/SL{int(sl*100)}/trail{int(act*100)}-{int(gb*100)}/time{ts}"
                    results = [
                        simulate_exit(p, e["entry_mid"], tp=tp, sl=sl,
                                     trailing_activate=act, trailing_giveback=gb,
                                     time_stop_days=ts)
                        for p, e in zip(paths, valid_entries)
                    ]
                    metrics = compute_strategy_metrics(results)
                    full_combo_results[key] = metrics

    best_full = max(full_combo_results.items(), key=lambda x: x[1].get("sharpe", 0))
    print(f"    Best full combo: {best_full[0]}")
    print(f"      Sharpe={best_full[1]['sharpe']:.2f}, Sortino={best_full[1]['sortino']:.2f}, WR={best_full[1]['win_rate']:.1%}, PF={best_full[1]['profit_factor']:.2f}")
    print(f"      Avg P&L={best_full[1]['avg_pnl_pct']:.2%}, Avg hold={best_full[1]['avg_hold_days']:.1f}d")

    # ── 7. CONTINUATION & REVERSAL ANALYSIS ──
    print("\n  Running continuation analysis...")
    continuation = run_continuation_analysis(paths, valid_entries)
    for level, data in continuation.items():
        print(f"    Hit {level}: {data['n_hit']} trades ({data['pct_of_all_trades']:.1%})")
        if data.get("continued_higher"):
            for higher, pct in data["continued_higher"].items():
                print(f"      → continued to {higher}: {pct:.1%}")
        print(f"      → reversed negative: {data['reversed_negative']:.1%}")

    print("\n  Running reversal analysis...")
    reversal = run_reversal_analysis(paths)
    for level, data in reversal.items():
        print(f"    Was up {level}: {data['n_hit']} cases → {data['pct_ended_negative']:.1%} ended negative, final avg {data['avg_final_pnl']:.2%}")

    # ── 8. SIGNAL TYPE BREAKDOWN ──
    print("\n  Running signal-type breakdown...")
    signal_breakdown = {}
    for sig_type in ["momentum", "mean_reversion", "mixed"]:
        type_mask = [classify_signal_type(e) == sig_type for e in valid_entries]
        type_paths = [p for p, m in zip(paths, type_mask) if m]
        type_entries = [e for e, m in zip(valid_entries, type_mask) if m]

        if len(type_paths) < 10:
            signal_breakdown[sig_type] = {"n": len(type_paths), "note": "too_few"}
            continue

        # Run the top strategies on this subset
        sub_results = {}
        for tp in [0.20, 0.25, 0.30, 0.40]:
            for sl in [-0.25, -0.30]:
                key = f"TP{int(tp*100)}/SL{int(sl*100)}"
                results = [simulate_exit(p, e["entry_mid"], tp=tp, sl=sl) for p, e in zip(type_paths, type_entries)]
                metrics = compute_strategy_metrics(results)
                sub_results[key] = metrics

        best = max(sub_results.items(), key=lambda x: x[1].get("sharpe", 0))
        signal_breakdown[sig_type] = {
            "n": len(type_paths),
            "best_strategy": best[0],
            "best_sharpe": best[1]["sharpe"],
            "best_win_rate": best[1]["win_rate"],
            "best_avg_pnl": best[1]["avg_pnl_pct"],
            "all_results": sub_results,
        }
        print(f"    {sig_type} ({len(type_paths)} trades): best={best[0]}, Sharpe={best[1]['sharpe']:.2f}")

    # ── COMPILE RESULTS ──
    # Rank all strategies by Sharpe
    all_strategies = {}
    all_strategies.update({f"tp_only/{k}": v for k, v in tp_results.items()})
    all_strategies.update({f"sl_only/{k}": v for k, v in sl_results.items()})
    all_strategies.update({f"tp_sl/{k}": v for k, v in combined_results.items()})
    all_strategies.update({f"trailing/{k}": v for k, v in trailing_results.items()})
    all_strategies.update({f"full/{k}": v for k, v in full_combo_results.items()})
    all_strategies["baseline/hold_to_end"] = baseline_metrics

    top_10 = sorted(all_strategies.items(), key=lambda x: x[1].get("sharpe", 0), reverse=True)[:10]

    print(f"\n  TOP 10 STRATEGIES BY SHARPE:")
    for rank, (name, m) in enumerate(top_10, 1):
        print(f"    {rank}. {name}: Sharpe={m['sharpe']:.2f}, WR={m['win_rate']:.1%}, PF={m['profit_factor']:.2f}, Avg={m['avg_pnl_pct']:.2%}")

    return {
        "ticker": ticker,
        "n_entries": len(valid_entries),
        "n_snapshot_dates": len(snapshot_dates),
        "date_range": f"{snapshot_dates[0].date()} to {snapshot_dates[-1].date()}",
        "baseline": baseline_metrics,
        "top_10_strategies": [{"name": n, **m} for n, m in top_10],
        "best_overall": {"name": best_full[0], **best_full[1]},
        "continuation_analysis": continuation,
        "reversal_analysis": reversal,
        "signal_type_breakdown": signal_breakdown,
        "fixed_tp_sweep": tp_results,
        "fixed_sl_sweep": sl_results,
        "combined_tp_sl": combined_results,
        "trailing_sweep": trailing_results,
    }


def generate_recommendations(all_results: dict) -> dict:
    """Generate per-signal-type exit recommendations from cross-ticker analysis."""
    recs = {}

    # Aggregate best strategies across tickers
    tp_sharpes = {}
    for ticker, result in all_results.items():
        if "error" in result:
            continue
        tp_sweep = result.get("fixed_tp_sweep", {})
        for level, metrics in tp_sweep.items():
            if level not in tp_sharpes:
                tp_sharpes[level] = []
            tp_sharpes[level].append(metrics.get("sharpe", 0))

    # Average Sharpe by TP level
    avg_tp_sharpe = {level: np.mean(sharpes) for level, sharpes in tp_sharpes.items()}
    best_tp = max(avg_tp_sharpe.items(), key=lambda x: x[1])

    # Continuation summary
    continuation_summary = {}
    for ticker, result in all_results.items():
        if "error" in result:
            continue
        cont = result.get("continuation_analysis", {})
        for level, data in cont.items():
            if level not in continuation_summary:
                continuation_summary[level] = {"reversed_neg": [], "continued": {}}
            continuation_summary[level]["reversed_neg"].append(data.get("reversed_negative", 0))
            for higher, pct in data.get("continued_higher", {}).items():
                if higher not in continuation_summary[level]["continued"]:
                    continuation_summary[level]["continued"][higher] = []
                continuation_summary[level]["continued"][higher].append(pct)

    # Signal type recommendations
    for sig_type in ["momentum", "mean_reversion", "mixed"]:
        type_sharpes = {}
        for ticker, result in all_results.items():
            if "error" in result:
                continue
            breakdown = result.get("signal_type_breakdown", {}).get(sig_type, {})
            if breakdown.get("n", 0) >= 10:
                best = breakdown.get("best_strategy", "")
                sharpe = breakdown.get("best_sharpe", 0)
                if best not in type_sharpes:
                    type_sharpes[best] = []
                type_sharpes[best].append(sharpe)

        if type_sharpes:
            # Find strategy that appears most AND has good avg sharpe
            best_strat = max(type_sharpes.items(), key=lambda x: np.mean(x[1]) * len(x[1]))
            recs[sig_type] = {
                "recommended_strategy": best_strat[0],
                "avg_sharpe": round(np.mean(best_strat[1]), 3),
                "n_tickers_tested": len(best_strat[1]),
            }

    # Overall recommendation
    overall_best = {}
    for ticker, result in all_results.items():
        if "error" in result:
            continue
        best = result.get("best_overall", {})
        name = best.get("name", "")
        if name:
            if name not in overall_best:
                overall_best[name] = {"sharpes": [], "win_rates": [], "profit_factors": []}
            overall_best[name]["sharpes"].append(best.get("sharpe", 0))
            overall_best[name]["win_rates"].append(best.get("win_rate", 0))
            overall_best[name]["profit_factors"].append(best.get("profit_factor", 0))

    # Best overall = highest avg sharpe across tickers
    if overall_best:
        best_overall = max(overall_best.items(), key=lambda x: np.mean(x[1]["sharpes"]))
        recs["overall"] = {
            "recommended_strategy": best_overall[0],
            "avg_sharpe": round(np.mean(best_overall[1]["sharpes"]), 3),
            "avg_win_rate": round(np.mean(best_overall[1]["win_rates"]), 4),
            "avg_profit_factor": round(np.mean(best_overall[1]["profit_factors"]), 3),
        }

    return {
        "recommendations": recs,
        "avg_tp_sharpe_by_level": {k: round(v, 3) for k, v in avg_tp_sharpe.items()},
        "best_tp_level": best_tp[0],
        "continuation_summary": {
            level: {
                "avg_pct_reversed_negative": round(np.mean(data["reversed_neg"]), 4),
                "avg_pct_continued": {
                    h: round(np.mean(pcts), 4) for h, pcts in data["continued"].items()
                },
            }
            for level, data in continuation_summary.items()
        },
    }


# ── MAIN ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Exit Strategy Optimizer")
    parser.add_argument("--ticker", type=str, help="Single ticker to analyze")
    parser.add_argument("--quick", action="store_true", help="Quick mode (every 5th date)")
    parser.add_argument("--tickers", type=str, nargs="+", help="List of tickers")
    args = parser.parse_args()

    start_time = datetime.now()
    print(f"Exit Strategy Optimizer — started at {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Data source: {CHAINS_DIR}")

    if args.ticker:
        tickers = [args.ticker]
    elif args.tickers:
        tickers = args.tickers
    else:
        tickers = SECTOR_ETFS

    all_results = {}
    for ticker in tickers:
        try:
            result = optimize_for_ticker(ticker, quick=args.quick)
            all_results[ticker] = result
        except Exception as e:
            print(f"\n  ERROR processing {ticker}: {e}")
            import traceback
            traceback.print_exc()
            all_results[ticker] = {"ticker": ticker, "error": str(e)}

    # Generate cross-ticker recommendations
    print(f"\n{'='*70}")
    print("GENERATING CROSS-TICKER RECOMMENDATIONS")
    print(f"{'='*70}")

    recommendations = generate_recommendations(all_results)

    print(f"\n  Best TP level across all ETFs: {recommendations['best_tp_level']}")
    print(f"  TP Sharpe by level: {recommendations['avg_tp_sharpe_by_level']}")

    if "overall" in recommendations.get("recommendations", {}):
        overall = recommendations["recommendations"]["overall"]
        print(f"\n  OVERALL BEST EXIT STRATEGY: {overall['recommended_strategy']}")
        print(f"    Avg Sharpe: {overall['avg_sharpe']:.2f}")
        print(f"    Avg WR: {overall['avg_win_rate']:.1%}")
        print(f"    Avg PF: {overall['avg_profit_factor']:.2f}")

    for sig_type in ["momentum", "mean_reversion", "mixed"]:
        if sig_type in recommendations.get("recommendations", {}):
            rec = recommendations["recommendations"][sig_type]
            print(f"\n  {sig_type.upper()} recommendation: {rec['recommended_strategy']}")
            print(f"    Avg Sharpe: {rec['avg_sharpe']:.2f}, tested on {rec['n_tickers_tested']} tickers")

    # Continuation analysis summary
    print(f"\n  CONTINUATION ANALYSIS (avg across tickers):")
    for level, data in recommendations.get("continuation_summary", {}).items():
        rev = data.get("avg_pct_reversed_negative", 0)
        cont = data.get("avg_pct_continued", {})
        cont_str = ", ".join(f"→{h}: {p:.0%}" for h, p in cont.items())
        print(f"    Hit {level}: {rev:.0%} reversed negative | {cont_str}")

    # Save results
    output = {
        "generated_at": datetime.now().isoformat(),
        "tickers_analyzed": tickers,
        "per_ticker_results": all_results,
        "cross_ticker_recommendations": recommendations,
        "current_rules_kb281": {
            "tp": 0.30,
            "sl": -0.25,
            "trailing_activate": 0.15,
            "trailing_giveback": 0.50,
            "time_stop_days": 5,
            "theta_cutoff_dte": 7,
        },
    }

    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_FILE, "w") as f:
        json.dump(output, f, indent=2, default=str)

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\n{'='*70}")
    print(f"COMPLETE — {elapsed:.0f}s elapsed")
    print(f"Results saved to: {OUTPUT_FILE}")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
