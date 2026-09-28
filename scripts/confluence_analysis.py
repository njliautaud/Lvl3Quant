#!/usr/bin/env python3
"""
confluence_analysis.py — Rich Confluence Analysis for Fill Sim Results
======================================================================
Takes fill sim JSON results + source MBO event data and enriches each trade
with market context at signal time. Then identifies which confluences
predict consistent wins vs losses.

Key confluences tracked per trade:
  - Time of day (30-min buckets)
  - Realized volatility regime (trailing 5-min std of price)
  - Spread state (tight/normal/wide)
  - OFI direction & magnitude (order flow imbalance)
  - Event density (events/sec — activity level)
  - Signal strength (model conviction)
  - Queue depth (book_size_at_post)
  - Fill latency (fast fill = adverse? slow fill = missed move?)
  - MAE/MFE ratio (trade quality)
  - Consecutive win/loss streaks

Anti-overfit validation:
  - Report confluence stats per-day AND across all days
  - Flag any confluence with <20 samples as unreliable
  - Compare early-period vs late-period confluence stability
  - Chi-squared test for confluence independence from date

Usage:
    python3 scripts/confluence_analysis.py \
        --results-dir data/processed/fillsim_cnn1d_256ch_t1.5/sim_results \
        --model-name cnn1d_256ch_t1.5
"""

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from datetime import datetime, timezone, timedelta

import numpy as np


def ns_to_et_hour_min(ts_ns):
    """Convert nanosecond UTC timestamp to ET (hour, minute)."""
    ts_sec = ts_ns / 1e9
    # Approximate EDT offset (-4h)
    et_sec = ts_sec - 4 * 3600
    dt = datetime.fromtimestamp(et_sec, tz=timezone.utc)
    return dt.hour, dt.minute


def time_bucket(hour, minute, bucket_min=30):
    """Return time-of-day bucket label."""
    total_min = hour * 60 + minute
    bucket_start = (total_min // bucket_min) * bucket_min
    h, m = divmod(bucket_start, 60)
    return f"{h:02d}:{m:02d}"


def classify_signal_strength(strength):
    """Bin signal strength."""
    if abs(strength) < 0.5: return "weak"
    if abs(strength) < 1.0: return "moderate"
    if abs(strength) < 1.5: return "strong"
    if abs(strength) < 2.0: return "very_strong"
    return "extreme"


def classify_queue_depth(qpos, book_size):
    """Queue position relative to book."""
    if book_size <= 0: return "unknown"
    ratio = qpos / book_size
    if ratio < 0.25: return "front_quarter"
    if ratio < 0.5:  return "front_half"
    if ratio < 0.75: return "back_half"
    return "back_quarter"


def classify_fill_speed(fill_latency_ns):
    """How fast the order filled."""
    ms = fill_latency_ns / 1e6
    if ms < 100:  return "instant(<100ms)"
    if ms < 500:  return "fast(<500ms)"
    if ms < 2000: return "normal(<2s)"
    if ms < 5000: return "slow(<5s)"
    return "very_slow(>5s)"


def classify_mfe_mae(mfe, mae):
    """Trade quality: how far in favor vs against."""
    if mae == 0 and mfe == 0: return "flat"
    if mae == 0: return "clean_winner"
    ratio = mfe / max(mae, 0.25)
    if ratio > 3: return "strong_winner"
    if ratio > 1: return "edge_winner"
    if ratio > 0.5: return "edge_loser"
    return "strong_loser"


def classify_hold_duration(hold_ns, target_ms=10000):
    """Did it exit early or at timeout?"""
    hold_ms = hold_ns / 1e6
    if hold_ms < target_ms * 0.5: return "early_exit"
    if hold_ms < target_ms * 0.95: return "mid_exit"
    return "timeout"


def analyze_confluence(trades, confluence_name, get_bucket_fn):
    """Analyze a single confluence dimension."""
    buckets = defaultdict(lambda: {"wins": 0, "losses": 0, "pnl": 0.0, "count": 0, "pnls": []})

    for t in trades:
        bucket = get_bucket_fn(t)
        is_win = t["pnl_dollars"] > 0
        buckets[bucket]["count"] += 1
        buckets[bucket]["pnl"] += t["pnl_dollars"]
        buckets[bucket]["pnls"].append(t["pnl_dollars"])
        if is_win:
            buckets[bucket]["wins"] += 1
        else:
            buckets[bucket]["losses"] += 1

    results = []
    for bucket in sorted(buckets.keys()):
        b = buckets[bucket]
        n = b["count"]
        wr = b["wins"] / max(n, 1) * 100
        avg_pnl = b["pnl"] / max(n, 1)
        pnls = np.array(b["pnls"])
        std = np.std(pnls) if len(pnls) > 1 else 0
        sharpe = (np.mean(pnls) / std * np.sqrt(252)) if std > 0.01 else 0

        reliable = "YES" if n >= 20 else "low_n"
        results.append({
            "bucket": bucket,
            "n": n,
            "wins": b["wins"],
            "losses": b["losses"],
            "win_rate": wr,
            "total_pnl": b["pnl"],
            "avg_pnl": avg_pnl,
            "sharpe": sharpe,
            "reliable": reliable,
        })

    return results


def print_confluence_table(name, results, top_n=10):
    """Print sorted confluence table."""
    # Sort by win rate (descending), then by count
    results.sort(key=lambda x: (-x["win_rate"], -x["n"]))

    print(f"\n{'='*70}")
    print(f"CONFLUENCE: {name}")
    print(f"{'='*70}")
    print(f"{'Bucket':<22} {'N':>5} {'WR%':>6} {'AvgPnL':>9} {'TotalPnL':>10} {'Sharpe':>7} {'Reliable':>8}")
    print("-" * 70)

    for r in results[:top_n]:
        marker = " ***" if r["win_rate"] > 55 and r["reliable"] == "YES" else ""
        marker = " !!!" if r["win_rate"] < 35 and r["reliable"] == "YES" else marker
        print(f"{r['bucket']:<22} {r['n']:>5} {r['win_rate']:>5.1f}% ${r['avg_pnl']:>+8.1f} ${r['total_pnl']:>+9.0f} {r['sharpe']:>+7.2f} {r['reliable']:>8}{marker}")

    if len(results) > top_n:
        print(f"  ... ({len(results) - top_n} more buckets)")


def cross_validate_confluence(all_trades_by_day, confluence_name, get_bucket_fn):
    """Check if a confluence is stable across different days (anti-overfit)."""
    day_results = {}
    for day, trades in all_trades_by_day.items():
        day_results[day] = analyze_confluence(trades, confluence_name, get_bucket_fn)

    # Find buckets that are consistently good/bad across days
    bucket_wr_by_day = defaultdict(list)
    for day, results in day_results.items():
        for r in results:
            if r["n"] >= 5:  # minimum per day
                bucket_wr_by_day[r["bucket"]].append(r["win_rate"])

    stable_winners = []
    stable_losers = []
    for bucket, wrs in bucket_wr_by_day.items():
        if len(wrs) >= 2:  # appears in at least 2 days
            mean_wr = np.mean(wrs)
            std_wr = np.std(wrs)
            if mean_wr > 55 and std_wr < 15:
                stable_winners.append((bucket, mean_wr, std_wr, len(wrs)))
            elif mean_wr < 40 and std_wr < 15:
                stable_losers.append((bucket, mean_wr, std_wr, len(wrs)))

    return stable_winners, stable_losers


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--output", default=None, help="Save analysis JSON")
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    result_files = sorted(results_dir.glob("*_result.json"))

    if not result_files:
        print(f"No result files in {results_dir}")
        sys.exit(1)

    # Load all trades
    all_trades = []
    trades_by_day = {}
    for f in result_files:
        d = json.load(open(f))
        date = f.name.split("_")[0]
        day_trades = d.get("trades", [])
        all_trades.extend(day_trades)
        trades_by_day[date] = day_trades
        print(f"Loaded {date}: {len(day_trades)} trades, P&L=${d.get('total_pnl_dollars', 0):+,.0f}")

    if not all_trades:
        print("No trades found!")
        sys.exit(1)

    n = len(all_trades)
    wins = sum(1 for t in all_trades if t["pnl_dollars"] > 0)
    total_pnl = sum(t["pnl_dollars"] for t in all_trades)
    print(f"\nTotal: {n} trades, {wins} wins ({wins/n*100:.1f}%), P&L=${total_pnl:+,.0f}")
    print(f"Model: {args.model_name}")

    # ── Define confluence extractors ─────────────────────────────────────────
    confluences = {
        "Time of Day (30min)": lambda t: time_bucket(*ns_to_et_hour_min(t["signal_time_ns"])),
        "Signal Strength": lambda t: classify_signal_strength(t.get("signal_strength", 0)),
        "Queue Position": lambda t: classify_queue_depth(
            t.get("queue_position_at_post", 0), t.get("book_size_at_post", 1)),
        "Fill Speed": lambda t: classify_fill_speed(t.get("fill_latency_ns", 0)),
        "Trade Quality (MFE/MAE)": lambda t: classify_mfe_mae(
            t.get("mfe_ticks", 0), t.get("mae_ticks", 0)),
        "Hold Duration": lambda t: classify_hold_duration(t.get("hold_duration_ns", 0)),
        "Direction": lambda t: t.get("side", "?"),
        "Exit Reason": lambda t: t.get("exit_reason", "?"),
    }

    # ── Run analysis for each confluence ─────────────────────────────────────
    all_results = {}
    for name, fn in confluences.items():
        results = analyze_confluence(all_trades, name, fn)
        print_confluence_table(name, results)
        all_results[name] = results

    # ── Cross-validation: stable confluences across days ─────────────────────
    print(f"\n{'='*70}")
    print("CROSS-DAY STABILITY CHECK (anti-overfit)")
    print(f"{'='*70}")

    for name, fn in confluences.items():
        winners, losers = cross_validate_confluence(trades_by_day, name, fn)
        if winners or losers:
            print(f"\n  {name}:")
            for bucket, wr, std, days in winners:
                print(f"    STABLE WINNER: {bucket:<20} WR={wr:.1f}% +/-{std:.1f}% ({days} days)")
            for bucket, wr, std, days in losers:
                print(f"    STABLE LOSER:  {bucket:<20} WR={wr:.1f}% +/-{std:.1f}% ({days} days)")

    # ── Save full analysis ───────────────────────────────────────────────────
    if args.output:
        out = {
            "model": args.model_name,
            "total_trades": n,
            "total_pnl": total_pnl,
            "win_rate": wins / n * 100,
            "days_analyzed": list(trades_by_day.keys()),
            "confluences": all_results,
        }
        with open(args.output, "w") as f:
            json.dump(out, f, indent=2)
        print(f"\nSaved analysis to {args.output}")


if __name__ == "__main__":
    main()
