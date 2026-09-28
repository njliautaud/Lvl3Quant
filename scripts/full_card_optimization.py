#!/usr/bin/env python3
"""
full_card_optimization.py — Comprehensive deep analysis of ALL trading cards.

For each card (1, 3, 4, 5, 6, 7), runs fill_sim with the card's CURRENT best config
across all OOT dates (2025-12-01 to 2026-03-08), collects per-trade data, then
performs deep analysis:

  A. MAE/MFE Analysis (optimal cutoffs, MFE utilization)
  B. Edge Decay / Hold Time Analysis (optimal hold per card)
  C. Time-of-Day Analysis (best/worst hours, lunch lull)
  D. Signal Strength Analysis (z-score buckets, optimal threshold)
  E. Side Analysis (LONG vs SHORT split on all metrics)
  F. Correlation Analysis (signal vs PnL, MAE recovery, etc.)

Run on Jupiter with 14 workers:
  nohup python3 full_card_optimization.py 2>&1 | tee card_deep_analysis.log &

Output: /home/jupiter/Lvl3Quant/data/processed/card_deep_analysis/
"""

import os
import sys
import json
import time
import subprocess
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone, timedelta
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ─────────────────────────────────────────────────────────────────────
FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/card_deep_analysis")

OOT_START = "2025-12-01"
OOT_END = "2026-03-08"
WORKERS = 14

# ── Card Definitions (CURRENT BEST CONFIGS) ──────────────────────────────────
CARDS = {
    "c1": {
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "signal_threshold": 0.1,
        "tp_ticks": 8,
        "sl_ticks": None,
        "hold_ms": 7200000,
        "conviction_bars": None,
        "conviction_mag": None,
        "desc": "Card1: book predstdExit conv1.5 vol50, TP8, 2hr hold",
    },
    "c3": {
        "pred_suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "signal_threshold": 0.3,
        "tp_ticks": 10,
        "sl_ticks": None,
        "hold_ms": 3600000,
        "conviction_bars": 100,
        "conviction_mag": 0.8,
        "desc": "Card3: raw rawExit conv0.15 vol70, TP10, 1hr, convExit(100/0.8)",
    },
    "c4": {
        "pred_suffix": "book_predstdExit_conv2.0_vol70",
        "signal_threshold": 0.1,
        "tp_ticks": 20,
        "sl_ticks": None,
        "hold_ms": 7200000,
        "conviction_bars": None,
        "conviction_mag": None,
        "desc": "Card4: book predstdExit conv2.0 vol70, TP20, 2hr hold",
    },
    "c5": {
        "pred_suffix": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "signal_threshold": 0.1,
        "tp_ticks": None,
        "sl_ticks": None,
        "hold_ms": 3600000,
        "conviction_bars": None,
        "conviction_mag": None,
        "desc": "Card5: raw rawExit conv0.05 ethr0.5 vol0, no TP/SL, 1hr",
    },
    "c6": {
        "pred_suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "signal_threshold": 0.1,
        "tp_ticks": 20,
        "sl_ticks": 25,
        "hold_ms": 3600000,
        "conviction_bars": 60,
        "conviction_mag": 1.5,
        "desc": "Card6: raw rawExit conv0.15 vol70, TP20/SL25, 1hr, convExit(60/1.5)",
    },
    "c7": {
        "pred_suffix": "smooth_smoothExit_conv1.5_ethr0.0_vol70",
        "signal_threshold": 0.1,
        "tp_ticks": None,
        "sl_ticks": 20,
        "hold_ms": 3600000,
        "conviction_bars": None,
        "conviction_mag": None,
        "desc": "Card7: smooth smoothExit conv1.5 vol70, SL20, 1hr",
    },
}

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("card_deep")


# ── Helper Functions ──────────────────────────────────────────────────────────

def discover_dates():
    """Find all OOT dates with MBO data."""
    dates = []
    for f in sorted(MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst")):
        name = f.stem.split(".")[0]
        date_raw = name.split("-")[-1]
        if len(date_raw) == 8:
            date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
            if OOT_START <= date_str <= OOT_END:
                dates.append(date_str)
    return dates


def build_task(card_id, card_def, date_str):
    """Build a single fill_sim_cli command for a card+date."""
    date_compact = date_str.replace("-", "")
    mbo_file = MBO_DIR / f"glbx-mdp3-{date_compact}.mbo.dbn.zst"
    pred_file = PRED_DIR / f"{date_str}_{card_def['pred_suffix']}.npz"

    if not mbo_file.exists() or not pred_file.exists():
        return None

    out_dir = OUTPUT_DIR / card_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{date_str}.json"

    if out_file.exists():
        return None  # Skip completed

    cmd = [
        FILL_SIM,
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--signal-threshold", str(card_def["signal_threshold"]),
        "--hold-ms", str(card_def["hold_ms"]),
        "--quiet",
    ]

    if card_def["tp_ticks"] is not None:
        cmd += ["--take-profit-ticks", str(card_def["tp_ticks"])]
    if card_def["sl_ticks"] is not None:
        cmd += ["--stop-loss-ticks", str(card_def["sl_ticks"])]
    if card_def["conviction_bars"] is not None:
        cmd += ["--conviction-exit-bars", str(card_def["conviction_bars"])]
    if card_def["conviction_mag"] is not None:
        cmd += ["--conviction-exit-mag", str(card_def["conviction_mag"])]

    return {
        "cmd": cmd,
        "card": card_id,
        "date": date_str,
        "out_file": str(out_file),
    }


def run_task(task):
    """Execute a single fill_sim_cli task."""
    try:
        result = subprocess.run(task["cmd"], capture_output=True, timeout=600)
        if result.returncode != 0:
            return {"status": "FAIL", **task, "error": result.stderr.decode(errors='replace')[:200]}
        return {"status": "OK", **task}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", **task}
    except Exception as e:
        return {"status": "ERROR", **task, "error": str(e)[:200]}


def ns_to_datetime(ns):
    """Convert nanosecond timestamp to datetime."""
    return datetime.fromtimestamp(ns / 1e9, tz=timezone.utc)


def ns_to_hour_bucket(ns):
    """Convert ns timestamp to ET hour string like '09:30-10:30'."""
    dt = ns_to_datetime(ns)
    et = dt - timedelta(hours=5)  # UTC -> ET (EST, approximate)
    h = et.hour
    if h < 10:
        return "09:30-10:30"
    elif h < 11:
        return "10:30-11:30"
    elif h < 12:
        return "11:30-12:30"
    elif h < 13:
        return "12:30-13:30"
    elif h < 14:
        return "13:30-14:30"
    elif h < 15:
        return "14:30-15:30"
    else:
        return "15:30-16:00"


def hold_time_bucket(hold_ns):
    """Classify hold time into buckets."""
    hold_sec = hold_ns / 1e9
    if hold_sec <= 60:
        return "0-1min"
    elif hold_sec <= 300:
        return "1-5min"
    elif hold_sec <= 900:
        return "5-15min"
    elif hold_sec <= 1800:
        return "15-30min"
    elif hold_sec <= 3600:
        return "30-60min"
    else:
        return "60-120min"


def signal_bucket(z):
    """Classify absolute z-score into buckets."""
    z = abs(z)
    if z < 0.5:
        return "0.0-0.5"
    elif z < 1.0:
        return "0.5-1.0"
    elif z < 1.5:
        return "1.0-1.5"
    elif z < 2.0:
        return "1.5-2.0"
    else:
        return "2.0+"


def safe_sharpe(pnls, annualize=252):
    """Compute annualized Sharpe from a list of PnLs."""
    if len(pnls) < 2:
        return 0.0
    arr = np.array(pnls)
    std = np.std(arr)
    if std == 0:
        return 0.0
    return float(np.mean(arr) / std * np.sqrt(annualize))


def percentiles(arr, pcts=[25, 50, 75, 90, 95, 99]):
    """Compute percentiles of an array."""
    if len(arr) == 0:
        return {f"p{p}": 0 for p in pcts}
    arr = np.array(arr)
    return {f"p{p}": round(float(np.percentile(arr, p)), 4) for p in pcts}


def group_stats(trades_in_group):
    """Compute comprehensive stats for a group of trades."""
    if not trades_in_group:
        return {"n": 0}

    pnls = [t["pnl_dollars"] for t in trades_in_group]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]

    mae_vals = [t.get("mae_ticks", 0) for t in trades_in_group]
    mfe_vals = [t.get("mfe_ticks", 0) for t in trades_in_group]
    hold_vals = [t.get("hold_duration_ns", 0) / 1e9 for t in trades_in_group]

    return {
        "n": len(trades_in_group),
        "total_pnl": round(sum(pnls), 2),
        "avg_pnl": round(np.mean(pnls), 2),
        "median_pnl": round(float(np.median(pnls)), 2),
        "win_rate": round(len(wins) / len(pnls) * 100, 1),
        "avg_win": round(np.mean(wins), 2) if wins else 0,
        "avg_loss": round(np.mean(losses), 2) if losses else 0,
        "avg_mae": round(np.mean(mae_vals), 2),
        "avg_mfe": round(np.mean(mfe_vals), 2),
        "avg_hold_sec": round(np.mean(hold_vals), 1),
        "sharpe_contribution": round(np.mean(pnls) / (np.std(pnls) + 1e-9), 4),
    }


# ── Analysis Functions ────────────────────────────────────────────────────────

def analyze_mae_mfe(all_trades, card_def):
    """A. MAE/MFE Analysis."""
    winners = [t for t in all_trades if t.get("pnl_dollars", 0) > 0]
    losers = [t for t in all_trades if t.get("pnl_dollars", 0) <= 0]

    w_mae = [t.get("mae_ticks", 0) for t in winners]
    l_mae = [t.get("mae_ticks", 0) for t in losers]
    w_mfe = [t.get("mfe_ticks", 0) for t in winners]
    l_mfe = [t.get("mfe_ticks", 0) for t in losers]

    # Optimal MAE cutoff: find where we cut more losers than winners (% basis)
    mae_cutoffs = []
    all_mae = [t.get("mae_ticks", 0) for t in all_trades]
    if all_mae:
        for cutoff in range(1, int(np.percentile(all_mae, 99)) + 1):
            w_cut = sum(1 for m in w_mae if m >= cutoff) / max(len(w_mae), 1)
            l_cut = sum(1 for m in l_mae if m >= cutoff) / max(len(l_mae), 1)
            surviving_w = len(winners) - sum(1 for m in w_mae if m >= cutoff)
            surviving_l = len(losers) - sum(1 for m in l_mae if m >= cutoff)
            surviving_trades = [t for t in all_trades if t.get("mae_ticks", 0) < cutoff]
            surviving_pnl = sum(t.get("pnl_dollars", 0) for t in surviving_trades)
            mae_cutoffs.append({
                "cutoff_ticks": cutoff,
                "pct_winners_cut": round(w_cut * 100, 1),
                "pct_losers_cut": round(l_cut * 100, 1),
                "net_selectivity": round((l_cut - w_cut) * 100, 1),
                "surviving_trades": len(surviving_trades),
                "surviving_pnl": round(surviving_pnl, 2),
            })

    # Find optimal MAE cutoff (max net selectivity with positive PnL)
    optimal_mae = None
    if mae_cutoffs:
        profitable = [c for c in mae_cutoffs if c["surviving_pnl"] > 0 and c["surviving_trades"] >= 20]
        if profitable:
            optimal_mae = max(profitable, key=lambda x: x["surviving_pnl"])

    # MFE utilization: how much of MFE does TP capture?
    tp_ticks = card_def.get("tp_ticks")
    mfe_utilization = None
    if tp_ticks and w_mfe:
        # For winners, MFE >= TP (they hit TP). What fraction of total MFE was TP?
        mfe_utilization = {
            "tp_ticks": tp_ticks,
            "avg_winner_mfe": round(np.mean(w_mfe), 2),
            "pct_captured": round(tp_ticks / max(np.mean(w_mfe), 0.01) * 100, 1),
            "mfe_above_tp": round(np.mean([m - tp_ticks for m in w_mfe if m >= tp_ticks]), 2) if any(m >= tp_ticks for m in w_mfe) else 0,
        }

    return {
        "winners": {
            "n": len(winners),
            "mae_percentiles": percentiles(w_mae),
            "mfe_percentiles": percentiles(w_mfe),
            "avg_mae": round(np.mean(w_mae), 2) if w_mae else 0,
            "avg_mfe": round(np.mean(w_mfe), 2) if w_mfe else 0,
        },
        "losers": {
            "n": len(losers),
            "mae_percentiles": percentiles(l_mae),
            "mfe_percentiles": percentiles(l_mfe),
            "avg_mae": round(np.mean(l_mae), 2) if l_mae else 0,
            "avg_mfe": round(np.mean(l_mfe), 2) if l_mfe else 0,
        },
        "all_trades": {
            "mae_percentiles": percentiles(all_mae),
            "mfe_percentiles": percentiles([t.get("mfe_ticks", 0) for t in all_trades]),
        },
        "mae_cutoff_analysis": mae_cutoffs[:30],  # Top 30 cutoffs
        "optimal_mae_cutoff": optimal_mae,
        "mfe_utilization": mfe_utilization,
    }


def analyze_edge_decay(all_trades):
    """B. Edge Decay / Hold Time Analysis."""
    buckets_order = ["0-1min", "1-5min", "5-15min", "15-30min", "30-60min", "60-120min"]
    bucket_trades = defaultdict(list)

    for t in all_trades:
        hold_ns = t.get("hold_duration_ns", 0)
        if hold_ns <= 0:
            # Estimate from fill/exit times
            hold_ns = t.get("exit_time_ns", 0) - t.get("fill_time_ns", 0)
        bucket = hold_time_bucket(hold_ns)
        bucket_trades[bucket].append(t)

    results = {}
    for b in buckets_order:
        trades = bucket_trades.get(b, [])
        results[b] = group_stats(trades)

    # Identify optimal hold time bucket
    best_bucket = max(
        [(b, results[b]) for b in buckets_order if results[b]["n"] >= 10],
        key=lambda x: x[1].get("avg_pnl", 0),
        default=(None, None)
    )

    # Check for edge decay cliff
    pnl_by_bucket = []
    for b in buckets_order:
        if results[b]["n"] >= 5:
            pnl_by_bucket.append((b, results[b]["avg_pnl"]))

    cliff_detected = None
    for i in range(1, len(pnl_by_bucket)):
        prev_pnl = pnl_by_bucket[i-1][1]
        curr_pnl = pnl_by_bucket[i][1]
        if prev_pnl > 0 and curr_pnl < 0:
            cliff_detected = f"Edge cliff between {pnl_by_bucket[i-1][0]} and {pnl_by_bucket[i][0]}"
            break
        if prev_pnl > 0 and curr_pnl < prev_pnl * 0.3:
            cliff_detected = f"Edge decay cliff at {pnl_by_bucket[i][0]} (dropped to {curr_pnl:.2f} from {prev_pnl:.2f})"
            break

    return {
        "buckets": results,
        "optimal_bucket": best_bucket[0] if best_bucket[0] else "N/A",
        "cliff_detected": cliff_detected,
        "edge_decay_pattern": pnl_by_bucket,
    }


def analyze_time_of_day(all_trades):
    """C. Time-of-Day Analysis."""
    hours_order = ["09:30-10:30", "10:30-11:30", "11:30-12:30", "12:30-13:30",
                   "13:30-14:30", "14:30-15:30", "15:30-16:00"]
    hour_trades = defaultdict(list)

    for t in all_trades:
        fill_ns = t.get("fill_time_ns", 0)
        if fill_ns > 0:
            hour = ns_to_hour_bucket(fill_ns)
            hour_trades[hour].append(t)

    results = {}
    for h in hours_order:
        trades = hour_trades.get(h, [])
        results[h] = group_stats(trades)

    # Find best and worst hours
    active_hours = [(h, results[h]) for h in hours_order if results[h]["n"] >= 5]
    best_hour = max(active_hours, key=lambda x: x[1].get("avg_pnl", 0), default=(None, None))
    worst_hour = min(active_hours, key=lambda x: x[1].get("avg_pnl", 0), default=(None, None))

    # Lunch lull check (12:30-13:30)
    lunch = results.get("12:30-13:30", {"n": 0})
    pre_lunch = results.get("11:30-12:30", {"n": 0})
    post_lunch = results.get("13:30-14:30", {"n": 0})

    lunch_lull = None
    if lunch["n"] >= 5 and pre_lunch["n"] >= 5 and post_lunch["n"] >= 5:
        if lunch.get("avg_pnl", 0) < min(pre_lunch.get("avg_pnl", 0), post_lunch.get("avg_pnl", 0)):
            lunch_lull = f"YES: lunch avg_pnl={lunch['avg_pnl']:.2f} vs pre={pre_lunch['avg_pnl']:.2f}, post={post_lunch['avg_pnl']:.2f}"
        else:
            lunch_lull = "NO: lunch hour not worse than surrounding hours"

    return {
        "hours": results,
        "best_hour": best_hour[0] if best_hour[0] else "N/A",
        "worst_hour": worst_hour[0] if worst_hour[0] else "N/A",
        "lunch_lull": lunch_lull,
    }


def analyze_signal_strength(all_trades):
    """D. Signal Strength Analysis."""
    buckets_order = ["0.0-0.5", "0.5-1.0", "1.0-1.5", "1.5-2.0", "2.0+"]
    bucket_trades = defaultdict(list)

    for t in all_trades:
        sig = t.get("signal_strength", 0)
        bucket = signal_bucket(sig)
        bucket_trades[bucket].append(t)

    results = {}
    for b in buckets_order:
        trades = bucket_trades.get(b, [])
        results[b] = group_stats(trades)

    # Does stronger signal = better trades?
    signal_trend = []
    for b in buckets_order:
        if results[b]["n"] >= 5:
            signal_trend.append((b, results[b]["avg_pnl"], results[b]["win_rate"]))

    monotonic_pnl = all(
        signal_trend[i][1] <= signal_trend[i+1][1]
        for i in range(len(signal_trend) - 1)
    ) if len(signal_trend) >= 2 else False

    # Optimal signal threshold
    best_threshold = None
    if signal_trend:
        cumulative_from_right = []
        for i in range(len(signal_trend)):
            trades_above = []
            for b in buckets_order[i:]:
                trades_above.extend(bucket_trades.get(b, []))
            if len(trades_above) >= 10:
                pnl = sum(t.get("pnl_dollars", 0) for t in trades_above)
                cumulative_from_right.append({
                    "min_bucket": buckets_order[i],
                    "n_trades": len(trades_above),
                    "total_pnl": round(pnl, 2),
                    "avg_pnl": round(pnl / len(trades_above), 2),
                })
        if cumulative_from_right:
            best_threshold = max(cumulative_from_right, key=lambda x: x["avg_pnl"])

    return {
        "buckets": results,
        "signal_trend": signal_trend,
        "stronger_signal_better": monotonic_pnl,
        "optimal_threshold": best_threshold,
    }


def analyze_sides(all_trades):
    """E. Side Analysis (LONG vs SHORT)."""
    longs = [t for t in all_trades if t.get("side") == "BUY"]
    shorts = [t for t in all_trades if t.get("side") == "SELL"]

    def side_deep_stats(trades, label):
        if not trades:
            return {"n": 0, "label": label}

        pnls = [t["pnl_dollars"] for t in trades]
        wins = [p for p in pnls if p > 0]
        losses = [p for p in pnls if p <= 0]
        mae_vals = [t.get("mae_ticks", 0) for t in trades]
        mfe_vals = [t.get("mfe_ticks", 0) for t in trades]
        hold_vals = [t.get("hold_duration_ns", 0) / 1e9 for t in trades]

        # Daily PnLs for Sharpe
        daily_pnl = defaultdict(float)
        for t in trades:
            fill_ns = t.get("fill_time_ns", 0)
            if fill_ns > 0:
                dt = ns_to_datetime(fill_ns)
                day = (dt - timedelta(hours=5)).strftime("%Y-%m-%d")
                daily_pnl[day] += t["pnl_dollars"]
        daily_vals = list(daily_pnl.values())

        return {
            "label": label,
            "n": len(trades),
            "total_pnl": round(sum(pnls), 2),
            "avg_pnl": round(np.mean(pnls), 2),
            "win_rate": round(len(wins) / len(pnls) * 100, 1),
            "avg_win": round(np.mean(wins), 2) if wins else 0,
            "avg_loss": round(np.mean(losses), 2) if losses else 0,
            "daily_sharpe": round(safe_sharpe(daily_vals), 2) if len(daily_vals) >= 2 else 0,
            "avg_mae": round(np.mean(mae_vals), 2),
            "avg_mfe": round(np.mean(mfe_vals), 2),
            "avg_hold_sec": round(np.mean(hold_vals), 1),
            "mae_percentiles": percentiles(mae_vals),
            "mfe_percentiles": percentiles(mfe_vals),
        }

    long_stats = side_deep_stats(longs, "LONG")
    short_stats = side_deep_stats(shorts, "SHORT")

    # Recommendation
    rec = None
    if long_stats["n"] >= 20 and short_stats["n"] >= 20:
        l_sharpe = long_stats.get("daily_sharpe", 0)
        s_sharpe = short_stats.get("daily_sharpe", 0)
        if l_sharpe > s_sharpe * 2 and l_sharpe > 1.0:
            rec = f"LONG-ONLY recommended: long Sharpe={l_sharpe:.2f} >> short Sharpe={s_sharpe:.2f}"
        elif s_sharpe > l_sharpe * 2 and s_sharpe > 1.0:
            rec = f"SHORT-ONLY recommended: short Sharpe={s_sharpe:.2f} >> long Sharpe={l_sharpe:.2f}"
        elif l_sharpe > 1.0 and s_sharpe > 1.0:
            rec = f"Both sides profitable: long Sharpe={l_sharpe:.2f}, short Sharpe={s_sharpe:.2f}"
        else:
            rec = f"Neither side strong alone: long Sharpe={l_sharpe:.2f}, short Sharpe={s_sharpe:.2f}"

    return {
        "long": long_stats,
        "short": short_stats,
        "recommendation": rec,
    }


def analyze_correlations(all_trades):
    """F. Correlation Analysis."""
    if len(all_trades) < 10:
        return {"error": "Too few trades for correlation analysis"}

    pnls = np.array([t.get("pnl_dollars", 0) for t in all_trades])
    signals = np.array([abs(t.get("signal_strength", 0)) for t in all_trades])
    maes = np.array([t.get("mae_ticks", 0) for t in all_trades])
    mfes = np.array([t.get("mfe_ticks", 0) for t in all_trades])
    holds = np.array([t.get("hold_duration_ns", 0) / 1e9 for t in all_trades])

    def safe_corr(a, b):
        if len(a) < 3 or np.std(a) == 0 or np.std(b) == 0:
            return 0.0
        return round(float(np.corrcoef(a, b)[0, 1]), 4)

    # High-MAE trade recovery: do trades with MAE > median still end up profitable?
    mae_median = float(np.median(maes)) if len(maes) > 0 else 0
    high_mae = [t for t in all_trades if t.get("mae_ticks", 0) > mae_median]
    high_mae_pnl = sum(t.get("pnl_dollars", 0) for t in high_mae)
    high_mae_wr = sum(1 for t in high_mae if t.get("pnl_dollars", 0) > 0) / max(len(high_mae), 1) * 100

    # Entry hour vs PnL
    hour_pnls = defaultdict(list)
    for t in all_trades:
        fill_ns = t.get("fill_time_ns", 0)
        if fill_ns > 0:
            hour = ns_to_hour_bucket(fill_ns)
            hour_pnls[hour].append(t.get("pnl_dollars", 0))

    return {
        "signal_vs_pnl": safe_corr(signals, pnls),
        "mae_vs_pnl": safe_corr(maes, pnls),
        "mfe_vs_hold": safe_corr(mfes, holds),
        "signal_vs_mae": safe_corr(signals, maes),
        "signal_vs_mfe": safe_corr(signals, mfes),
        "high_mae_recovery": {
            "mae_median": mae_median,
            "n_high_mae_trades": len(high_mae),
            "high_mae_total_pnl": round(high_mae_pnl, 2),
            "high_mae_win_rate": round(high_mae_wr, 1),
        },
        "interpretation": {
            "signal_pnl": "Stronger signals lead to better trades" if safe_corr(signals, pnls) > 0.05 else "Signal strength weakly correlated with PnL",
            "mae_recovery": f"High-MAE trades {'DO recover' if high_mae_wr > 50 else 'do NOT recover'} (WR={high_mae_wr:.1f}%)",
        }
    }


# ── Main Report ───────────────────────────────────────────────────────────────

def load_all_trades(card_id):
    """Load all trades for a card from per-date JSON files."""
    result_dir = OUTPUT_DIR / card_id
    if not result_dir.exists():
        return [], []

    all_trades = []
    daily_pnls = []

    for f in sorted(result_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            pnl = data.get("total_pnl_dollars", 0)
            daily_pnls.append(pnl)
            trades = data.get("trades", [])
            all_trades.extend(trades)
        except Exception:
            continue

    return all_trades, daily_pnls


def print_card_report(card_id, card_def, analysis):
    """Print comprehensive report for a single card."""
    print(f"\n{'='*120}")
    print(f"  {card_def['desc']}")
    print(f"{'='*120}")

    summary = analysis.get("summary", {})
    print(f"\n  SUMMARY: {summary.get('n_trades', 0)} trades over {summary.get('n_days', 0)} days")
    print(f"  Total PnL: ${summary.get('total_pnl', 0):,.2f} | Sharpe: {summary.get('sharpe', 0):.2f} | "
          f"WR: {summary.get('win_rate', 0):.1f}% | PF: {summary.get('profit_factor', 0):.2f}")
    print(f"  Max DD: ${summary.get('max_dd', 0):,.2f} | Avg Win: ${summary.get('avg_win', 0):.2f} | "
          f"Avg Loss: ${summary.get('avg_loss', 0):.2f}")

    # A. MAE/MFE
    mae = analysis.get("mae_mfe", {})
    print(f"\n  --- A. MAE/MFE ANALYSIS ---")
    w = mae.get("winners", {})
    l = mae.get("losers", {})
    print(f"  Winners (n={w.get('n',0)}): avg_MAE={w.get('avg_mae',0):.1f}t, avg_MFE={w.get('avg_mfe',0):.1f}t")
    print(f"  Losers  (n={l.get('n',0)}): avg_MAE={l.get('avg_mae',0):.1f}t, avg_MFE={l.get('avg_mfe',0):.1f}t")
    print(f"  MAE P50/P90/P99: {mae.get('all_trades',{}).get('mae_percentiles',{}).get('p50',0):.1f} / "
          f"{mae.get('all_trades',{}).get('mae_percentiles',{}).get('p90',0):.1f} / "
          f"{mae.get('all_trades',{}).get('mae_percentiles',{}).get('p99',0):.1f}")
    opt = mae.get("optimal_mae_cutoff")
    if opt:
        print(f"  ** OPTIMAL MAE CUTOFF: {opt['cutoff_ticks']}t — cuts {opt['pct_losers_cut']:.1f}% losers vs "
              f"{opt['pct_winners_cut']:.1f}% winners, surviving PnL=${opt['surviving_pnl']:,.2f}")
    mfe_util = mae.get("mfe_utilization")
    if mfe_util:
        print(f"  MFE Utilization: TP captures {mfe_util['pct_captured']:.1f}% of avg winner MFE ({mfe_util['avg_winner_mfe']:.1f}t)")
        print(f"  Avg MFE above TP: {mfe_util['mfe_above_tp']:.1f}t left on table")

    # B. Edge Decay
    decay = analysis.get("edge_decay", {})
    print(f"\n  --- B. EDGE DECAY / HOLD TIME ---")
    for bucket, stats in decay.get("buckets", {}).items():
        if stats["n"] > 0:
            print(f"  {bucket:>10}: n={stats['n']:>4}, WR={stats['win_rate']:>5.1f}%, "
                  f"avgPnL=${stats['avg_pnl']:>7.2f}, avgMAE={stats['avg_mae']:.1f}t, "
                  f"avgMFE={stats['avg_mfe']:.1f}t")
    print(f"  Optimal bucket: {decay.get('optimal_bucket', 'N/A')}")
    if decay.get("cliff_detected"):
        print(f"  ** CLIFF: {decay['cliff_detected']}")

    # C. Time of Day
    tod = analysis.get("time_of_day", {})
    print(f"\n  --- C. TIME-OF-DAY ---")
    for hour, stats in tod.get("hours", {}).items():
        if stats["n"] > 0:
            print(f"  {hour}: n={stats['n']:>4}, WR={stats['win_rate']:>5.1f}%, "
                  f"avgPnL=${stats['avg_pnl']:>7.2f}, totalPnL=${stats['total_pnl']:>9.2f}")
    print(f"  Best hour: {tod.get('best_hour', 'N/A')} | Worst hour: {tod.get('worst_hour', 'N/A')}")
    if tod.get("lunch_lull"):
        print(f"  Lunch lull: {tod['lunch_lull']}")

    # D. Signal Strength
    sig = analysis.get("signal_strength", {})
    print(f"\n  --- D. SIGNAL STRENGTH ---")
    for bucket, stats in sig.get("buckets", {}).items():
        if stats["n"] > 0:
            print(f"  |z|={bucket}: n={stats['n']:>4}, WR={stats['win_rate']:>5.1f}%, "
                  f"avgPnL=${stats['avg_pnl']:>7.2f}, avgMAE={stats['avg_mae']:.1f}t, "
                  f"avgMFE={stats['avg_mfe']:.1f}t, avgHold={stats['avg_hold_sec']:.0f}s")
    print(f"  Stronger signal = better: {sig.get('stronger_signal_better', 'N/A')}")
    opt_thr = sig.get("optimal_threshold")
    if opt_thr:
        print(f"  ** Optimal threshold: >= {opt_thr['min_bucket']} ({opt_thr['n_trades']} trades, "
              f"avgPnL=${opt_thr['avg_pnl']:.2f})")

    # E. Side Analysis
    sides = analysis.get("sides", {})
    print(f"\n  --- E. LONG vs SHORT ---")
    for side_key in ["long", "short"]:
        s = sides.get(side_key, {})
        if s.get("n", 0) > 0:
            print(f"  {s['label']:>5}: n={s['n']:>4}, PnL=${s['total_pnl']:>9.2f}, "
                  f"Sharpe={s['daily_sharpe']:>5.2f}, WR={s['win_rate']:>5.1f}%, "
                  f"avgMAE={s['avg_mae']:.1f}t, avgMFE={s['avg_mfe']:.1f}t, "
                  f"avgHold={s['avg_hold_sec']:.0f}s")
    if sides.get("recommendation"):
        print(f"  ** {sides['recommendation']}")

    # F. Correlations
    corr = analysis.get("correlations", {})
    print(f"\n  --- F. CORRELATIONS ---")
    print(f"  signal_strength vs PnL: r={corr.get('signal_vs_pnl', 0):.4f}")
    print(f"  MAE vs final PnL:       r={corr.get('mae_vs_pnl', 0):.4f}")
    print(f"  MFE vs hold_time:       r={corr.get('mfe_vs_hold', 0):.4f}")
    hr = corr.get("high_mae_recovery", {})
    print(f"  High-MAE recovery: WR={hr.get('high_mae_win_rate', 0):.1f}%, "
          f"PnL=${hr.get('high_mae_total_pnl', 0):,.2f}")
    interp = corr.get("interpretation", {})
    for k, v in interp.items():
        print(f"  -> {v}")


def print_optimization_suggestions(all_analyses):
    """Print cross-card optimization suggestions."""
    print(f"\n\n{'#'*120}")
    print(f"  OPTIMIZATION SUGGESTIONS (DATA-DRIVEN)")
    print(f"{'#'*120}")

    for card_id, analysis in all_analyses.items():
        print(f"\n  {card_id.upper()}:")
        suggestions = []

        # Check MAE cutoff
        mae = analysis.get("mae_mfe", {})
        opt = mae.get("optimal_mae_cutoff")
        if opt and opt["net_selectivity"] > 5:
            suggestions.append(f"Add MAE exit at {opt['cutoff_ticks']}t (cuts {opt['pct_losers_cut']:.1f}% losers vs {opt['pct_winners_cut']:.1f}% winners)")

        # Check MFE utilization
        mfe_util = mae.get("mfe_utilization")
        if mfe_util and mfe_util["pct_captured"] < 50:
            suggestions.append(f"TP too tight — only captures {mfe_util['pct_captured']:.1f}% of MFE. Consider raising TP.")
        elif mfe_util and mfe_util["mfe_above_tp"] > 5:
            suggestions.append(f"Consider trailing stop after TP hit — {mfe_util['mfe_above_tp']:.1f}t left on table")

        # Check side asymmetry
        sides = analysis.get("sides", {})
        l_sharpe = sides.get("long", {}).get("daily_sharpe", 0)
        s_sharpe = sides.get("short", {}).get("daily_sharpe", 0)
        if l_sharpe > 1.0 and s_sharpe < 0.5:
            suggestions.append(f"Test LONG-ONLY (long Sharpe={l_sharpe:.2f}, short={s_sharpe:.2f})")
        elif s_sharpe > 1.0 and l_sharpe < 0.5:
            suggestions.append(f"Test SHORT-ONLY (short Sharpe={s_sharpe:.2f}, long={l_sharpe:.2f})")

        # Check time of day
        tod = analysis.get("time_of_day", {})
        worst = tod.get("worst_hour")
        if worst:
            worst_stats = tod.get("hours", {}).get(worst, {})
            if worst_stats.get("avg_pnl", 0) < -5 and worst_stats.get("n", 0) >= 10:
                suggestions.append(f"Test excluding {worst} (avgPnL=${worst_stats['avg_pnl']:.2f}, n={worst_stats['n']})")

        # Check lunch lull
        if tod.get("lunch_lull", "").startswith("YES"):
            suggestions.append("Test excluding 12:30-13:30 (lunch lull detected)")

        # Check signal threshold
        sig = analysis.get("signal_strength", {})
        opt_thr = sig.get("optimal_threshold")
        if opt_thr and opt_thr["min_bucket"] != "0.0-0.5":
            suggestions.append(f"Test higher signal gate >= {opt_thr['min_bucket']} (avg_pnl ${opt_thr['avg_pnl']:.2f})")

        # Check edge decay
        decay = analysis.get("edge_decay", {})
        if decay.get("cliff_detected"):
            suggestions.append(f"Shorten hold — {decay['cliff_detected']}")

        if not suggestions:
            suggestions.append("No clear optimizations suggested by data")

        for i, s in enumerate(suggestions, 1):
            print(f"    {i}. {s}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dates = discover_dates()
    log.info(f"Found {len(dates)} OOT dates: {dates[0]} to {dates[-1]}")

    # Build all tasks
    tasks = []
    skipped = 0
    for card_id, card_def in CARDS.items():
        for date_str in dates:
            task = build_task(card_id, card_def, date_str)
            if task is not None:
                tasks.append(task)
            else:
                skipped += 1

    total_possible = len(CARDS) * len(dates)
    log.info(f"Tasks: {len(tasks)} new + {skipped} skipped = {total_possible} total")
    log.info(f"Cards: {list(CARDS.keys())}")
    log.info(f"Workers: {WORKERS}")

    # Execute fill_sim for all cards
    if tasks:
        t0 = time.time()
        ok = fail = 0

        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(run_task, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures)):
                result = future.result()
                if result["status"] == "OK":
                    ok += 1
                else:
                    fail += 1
                    log.warning(f"  FAIL: {result['card']} {result['date']}: {result.get('error', '')[:100]}")

                if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed * 60
                    eta_min = (len(tasks) - i - 1) / rate if rate > 0 else 0
                    log.info(
                        f"Progress: {i+1}/{len(tasks)} ({ok} OK, {fail} FAIL) | "
                        f"{rate:.0f}/min | ETA: {eta_min:.1f}min"
                    )

        log.info(f"\nExecution complete: {ok} OK, {fail} FAIL in {(time.time()-t0)/60:.1f}min")
    else:
        log.info("All fill_sim tasks already completed!")

    # ── Deep Analysis Phase ───────────────────────────────────────────────────
    log.info("\n" + "="*80)
    log.info("DEEP ANALYSIS PHASE — Processing all per-trade data")
    log.info("="*80)

    all_analyses = {}

    for card_id, card_def in CARDS.items():
        log.info(f"\nAnalyzing {card_id}...")
        all_trades, daily_pnls = load_all_trades(card_id)

        if not all_trades:
            log.warning(f"  No trades for {card_id}, skipping")
            continue

        # Summary stats
        pnl_arr = np.array(daily_pnls)
        wins = [t for t in all_trades if t.get("pnl_dollars", 0) > 0]
        losses = [t for t in all_trades if t.get("pnl_dollars", 0) <= 0]
        gross_wins = sum(t["pnl_dollars"] for t in wins)
        gross_losses = abs(sum(t["pnl_dollars"] for t in losses))

        cumulative = np.cumsum(pnl_arr)
        running_max = np.maximum.accumulate(cumulative)
        max_dd = float(np.max(running_max - cumulative)) if len(cumulative) > 0 else 0

        summary = {
            "n_days": len(daily_pnls),
            "n_trades": len(all_trades),
            "total_pnl": round(sum(daily_pnls), 2),
            "sharpe": round(safe_sharpe(daily_pnls), 2),
            "win_rate": round(len(wins) / len(all_trades) * 100, 1),
            "avg_win": round(gross_wins / len(wins), 2) if wins else 0,
            "avg_loss": round(-gross_losses / len(losses), 2) if losses else 0,
            "profit_factor": round(gross_wins / max(gross_losses, 0.01), 2),
            "max_dd": round(max_dd, 2),
            "pos_days": sum(1 for p in daily_pnls if p > 0),
            "neg_days": sum(1 for p in daily_pnls if p <= 0),
        }

        analysis = {
            "summary": summary,
            "mae_mfe": analyze_mae_mfe(all_trades, card_def),
            "edge_decay": analyze_edge_decay(all_trades),
            "time_of_day": analyze_time_of_day(all_trades),
            "signal_strength": analyze_signal_strength(all_trades),
            "sides": analyze_sides(all_trades),
            "correlations": analyze_correlations(all_trades),
        }

        all_analyses[card_id] = analysis

        # Print report
        print_card_report(card_id, card_def, analysis)

    # Print cross-card optimization suggestions
    print_optimization_suggestions(all_analyses)

    # Save full JSON
    json_out = OUTPUT_DIR / "full_analysis.json"
    with open(json_out, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "oot_range": f"{OOT_START} to {OOT_END}",
            "cards": {k: v["desc"] for k, v in CARDS.items()},
            "analyses": all_analyses,
        }, f, indent=2, default=str)
    log.info(f"\nSaved full analysis JSON: {json_out}")

    # Save per-card trade-level data for the optimization script
    for card_id in CARDS:
        trades, _ = load_all_trades(card_id)
        if trades:
            trades_file = OUTPUT_DIR / f"{card_id}_all_trades.json"
            with open(trades_file, "w") as f:
                json.dump(trades, f)
            log.info(f"Saved {len(trades)} trades: {trades_file}")

    print(f"\n{'='*120}")
    print(f"  ANALYSIS COMPLETE — Results in {OUTPUT_DIR}")
    print(f"{'='*120}")


if __name__ == "__main__":
    main()
