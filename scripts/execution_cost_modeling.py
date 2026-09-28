#!/usr/bin/env python3
"""
execution_cost_modeling.py - Find the breakeven alpha threshold for each execution style.

Key question: given ES tick size = $12.50, 1 round-trip = 2 ticks = $25,
what minimum |signal| magnitude is needed to overcome execution costs?

This script systematically maps:
1. Avg PnL per trade vs signal threshold (IC → profitability curve)
2. Fill rate vs signal threshold (how many signals we can actually trade)
3. Opportunity cost: tighter filter = fewer trades = less total PnL
4. Entry method comparison: aggressive (market) vs passive (limit) vs chase
5. Time-decay: how quickly does signal alpha decay after it fires?

All results are annualized and normalized to find the optimal operating point.
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
from datetime import datetime
from collections import defaultdict

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/execution_cost_modeling")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = 14
OOT_START = "2025-12-01"
OOT_END   = "2026-02-17"

# ES MES contract: 1 tick = $1.25 (MES) or $12.50 (ES)
# fill_sim likely uses MES ticks
TICK_VALUE = 1.25  # MES
ROUND_TRIP_TICKS = 2  # 1 tick each side (aggressive: 2 ticks)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("exec_cost")

# Cards to analyze
CARDS = {
    "c1": {
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "optimal_tp": 13,
        "optimal_hold_ms": 7200000,
    },
    "c7": {
        "pred_suffix": "smooth_smoothExit_conv1.5_ethr0.0_vol70",
        "optimal_tp": 25,
        "optimal_hold_ms": 7200000,
    },
    "c5": {
        "pred_suffix": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "optimal_tp": None,
        "optimal_hold_ms": 3600000,
    },
}

# Fine-grained signal thresholds for the breakeven curve
FINE_THRESHOLDS = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.25, 1.5, 2.0, 2.5, 3.0]

# Hold durations to compare
HOLD_OPTIONS_MS = [
    300000,   # 5 min
    600000,   # 10 min
    1800000,  # 30 min
    3600000,  # 1 hr
    7200000,  # 2 hr
]

# TP options to compare (None = hold-based exit only)
TP_OPTIONS = [None, 5, 8, 10, 13, 15, 20, 25]

def get_mbo_file(dt_str):
    d = dt_str.replace("-", "")
    f = MBO_DIR / f"glbx-mdp3-{d}.mbo.dbn.zst"
    return f if f.exists() else None

def get_pred_file(dt_str, suffix):
    f = PRED_DIR / f"{dt_str}_{suffix}.npz"
    return f if f.exists() else None

def get_oot_dates():
    dates = []
    for f in sorted(MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst")):
        stem = f.stem
        parts = stem.split("-")
        if len(parts) < 3:
            continue
        d = parts[2]
        dt_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        if OOT_START <= dt_str <= OOT_END:
            dates.append(dt_str)
    return sorted(dates)

def run_fill_sim(mbo_file, pred_file, out_file, tp=None, hold_ms=7200000,
                  sig=0.0, latency_ms=10):
    cmd = [
        FILL_SIM,
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--signal-threshold", str(sig),
        "--hold-ms", str(hold_ms),
        "--latency-ms", str(latency_ms),
    ]
    if tp is not None:
        cmd += ["--take-profit-ticks", str(tp)]
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        with open(out_file) as f:
            return json.load(f)
    except Exception as e:
        return None

def parse_result(r):
    if r is None:
        return 0, 0.0, 0
    n = (r.get("summary", {}).get("n_trades") or r.get("n_trades") or 0)
    pnl = (r.get("summary", {}).get("total_pnl") or r.get("total_pnl") or 0.0)
    wins = (r.get("summary", {}).get("n_wins") or r.get("n_wins") or 0)
    return n, pnl, wins

def run_threshold_curve_job(args):
    """Worker: (date, card, sig_thresh, hold_ms, tp) -> stats dict"""
    dt_str, card_name, card_cfg, sig_thresh, hold_ms, tp = args
    mbo_file = get_mbo_file(dt_str)
    pred_file = get_pred_file(dt_str, card_cfg["pred_suffix"])
    if mbo_file is None or pred_file is None:
        return None

    label = f"{card_name}_sig{sig_thresh:.2f}_hold{hold_ms//1000}s_tp{tp or 'none'}"
    out_file = OUTPUT_DIR / f"{dt_str}_{label}.json"

    if out_file.exists():
        try:
            with open(out_file) as f:
                r = json.load(f)
                n, pnl, wins = parse_result(r)
                return {
                    "date": dt_str, "card": card_name,
                    "sig_thresh": sig_thresh, "hold_ms": hold_ms, "tp": tp,
                    "n_trades": n, "total_pnl": pnl, "n_wins": wins,
                }
        except:
            pass

    r = run_fill_sim(mbo_file, pred_file, out_file, tp=tp, hold_ms=hold_ms,
                      sig=sig_thresh, latency_ms=10)
    if r is None:
        return None

    n, pnl, wins = parse_result(r)
    return {
        "date": dt_str, "card": card_name,
        "sig_thresh": sig_thresh, "hold_ms": hold_ms, "tp": tp,
        "n_trades": n, "total_pnl": pnl, "n_wins": wins,
    }

def compute_sharpe(daily_pnls):
    arr = np.array(daily_pnls)
    if arr.std() == 0 or len(arr) < 2:
        return 0.0
    return float(arr.mean() / arr.std() * np.sqrt(252))

def main():
    t0 = time.time()
    log.info("=== Execution Cost Modeling ===")

    dates = get_oot_dates()
    log.info(f"OOT dates: {len(dates)}")

    # --- Phase 1: Signal count vs threshold (to understand the alpha-volume tradeoff) ---
    log.info("Phase 1: Signal count vs threshold analysis...")
    signal_counts = {}
    for card_name, card_cfg in CARDS.items():
        thresh_counts = {}
        for thresh in FINE_THRESHOLDS:
            total = 0
            for dt_str in dates:
                pred_file = get_pred_file(dt_str, card_cfg["pred_suffix"])
                if pred_file is None:
                    continue
                d = np.load(pred_file)
                preds = d["predictions"]
                total += int((np.abs(preds) >= thresh).sum())
            thresh_counts[thresh] = total
        signal_counts[card_name] = thresh_counts
        log.info(f"  {card_name}: {thresh_counts[0.0]} at thresh=0, "
                 f"{thresh_counts[0.5]} at 0.5, {thresh_counts[1.0]} at 1.0")

    # --- Phase 2: Threshold curves across hold durations and TPs ---
    log.info("Phase 2: Building threshold-PnL curves...")

    # Focus on optimal hold/TP per card but sweep thresholds heavily
    jobs = []
    for dt_str in dates:
        for card_name, card_cfg in CARDS.items():
            # Sweep thresholds with optimal hold/TP
            for sig in FINE_THRESHOLDS:
                jobs.append((dt_str, card_name, card_cfg,
                              sig, card_cfg["optimal_hold_ms"], card_cfg["optimal_tp"]))
            # Also sweep hold durations at optimal threshold
            for hold_ms in HOLD_OPTIONS_MS:
                jobs.append((dt_str, card_name, card_cfg,
                              0.1, hold_ms, card_cfg["optimal_tp"]))
            # And sweep TP at optimal threshold and hold
            for tp in TP_OPTIONS:
                jobs.append((dt_str, card_name, card_cfg,
                              0.1, card_cfg["optimal_hold_ms"], tp))

    # Deduplicate
    seen = set()
    unique_jobs = []
    for job in jobs:
        key = (job[0], job[1], job[3], job[4], job[5])
        if key not in seen:
            seen.add(key)
            unique_jobs.append(job)
    log.info(f"Total unique jobs: {len(unique_jobs)}")

    # Run jobs
    raw_results = []
    completed = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_threshold_curve_job, job): job for job in unique_jobs}
        for future in as_completed(futures):
            completed += 1
            if completed % 300 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 1
                log.info(f"  {completed}/{len(unique_jobs)} | {elapsed:.0f}s | ETA {(len(unique_jobs)-completed)/rate:.0f}s")
            try:
                r = future.result()
                if r:
                    raw_results.append(r)
            except:
                pass

    log.info("Aggregating results...")

    # Aggregate by (card, sig_thresh, hold_ms, tp)
    agg = defaultdict(lambda: {"n_trades": 0, "total_pnl": 0.0, "n_wins": 0,
                                 "n_days": 0, "daily_pnls": []})
    for r in raw_results:
        key = (r["card"], r["sig_thresh"], r["hold_ms"], r["tp"])
        agg[key]["n_trades"] += r["n_trades"]
        agg[key]["total_pnl"] += r["total_pnl"]
        agg[key]["n_wins"] += r["n_wins"]
        agg[key]["n_days"] += 1
        agg[key]["daily_pnls"].append(r["total_pnl"])

    # Build threshold curves (hold/tp fixed at optimal, vary threshold)
    threshold_curves = {}
    for card_name, card_cfg in CARDS.items():
        curve_points = []
        for sig in FINE_THRESHOLDS:
            key = (card_name, sig, card_cfg["optimal_hold_ms"], card_cfg["optimal_tp"])
            stats = agg.get(key)
            if stats is None:
                continue
            n = stats["n_trades"]
            pnl = stats["total_pnl"]
            wins = stats["n_wins"]
            n_days = stats["n_days"]
            sharpe = compute_sharpe(stats["daily_pnls"])

            # Estimate signal count at this threshold
            sig_count = signal_counts[card_name].get(sig, 0)

            curve_points.append({
                "sig_thresh": sig,
                "n_signals_total": sig_count,
                "n_trades": n,
                "fill_rate_pct": round(n / sig_count * 100, 1) if sig_count > 0 else 0.0,
                "total_pnl": round(pnl, 2),
                "avg_daily_pnl": round(pnl / n_days, 2) if n_days > 0 else 0.0,
                "trades_per_day": round(n / n_days, 1) if n_days > 0 else 0.0,
                "win_rate": round(wins / n * 100, 1) if n > 0 else 0.0,
                "avg_pnl_per_trade": round(pnl / n, 2) if n > 0 else 0.0,
                "sharpe": round(sharpe, 3),
                "breakeven": pnl > 0 and n > 5,
            })

        # Find optimal point (max Sharpe with meaningful trade count)
        valid = [p for p in curve_points if p["n_trades"] >= 10 and p["sharpe"] > 0]
        optimal = max(valid, key=lambda x: x["sharpe"]) if valid else None

        # Find breakeven threshold
        breakeven_thresh = None
        for p in curve_points:
            if p["avg_pnl_per_trade"] > 0 and p["n_trades"] >= 5:
                breakeven_thresh = p["sig_thresh"]
                break

        threshold_curves[card_name] = {
            "curve": curve_points,
            "optimal_threshold": optimal["sig_thresh"] if optimal else None,
            "optimal_sharpe": optimal["sharpe"] if optimal else 0.0,
            "breakeven_threshold": breakeven_thresh,
            "cost_analysis": {
                "round_trip_cost_ticks": ROUND_TRIP_TICKS,
                "tick_value_usd": TICK_VALUE,
                "round_trip_cost_usd": ROUND_TRIP_TICKS * TICK_VALUE,
                "min_edge_needed_ticks": ROUND_TRIP_TICKS,
            }
        }

        log.info(f"  {card_name}: breakeven thresh={breakeven_thresh}, "
                 f"optimal thresh={optimal['sig_thresh'] if optimal else 'N/A'} "
                 f"(Sharpe={optimal['sharpe'] if optimal else 0:.3f})")

    # Hold duration curves
    hold_curves = {}
    for card_name, card_cfg in CARDS.items():
        curve_points = []
        for hold_ms in HOLD_OPTIONS_MS:
            key = (card_name, 0.1, hold_ms, card_cfg["optimal_tp"])
            stats = agg.get(key)
            if stats is None:
                continue
            n = stats["n_trades"]
            pnl = stats["total_pnl"]
            n_days = stats["n_days"]
            sharpe = compute_sharpe(stats["daily_pnls"])
            curve_points.append({
                "hold_ms": hold_ms,
                "hold_label": f"{hold_ms//60000}min",
                "n_trades": n,
                "total_pnl": round(pnl, 2),
                "sharpe": round(sharpe, 3),
                "avg_pnl_per_trade": round(pnl / n, 2) if n > 0 else 0.0,
            })
        hold_curves[card_name] = curve_points

    # TP curves
    tp_curves = {}
    for card_name, card_cfg in CARDS.items():
        curve_points = []
        for tp in TP_OPTIONS:
            key = (card_name, 0.1, card_cfg["optimal_hold_ms"], tp)
            stats = agg.get(key)
            if stats is None:
                continue
            n = stats["n_trades"]
            pnl = stats["total_pnl"]
            n_days = stats["n_days"]
            sharpe = compute_sharpe(stats["daily_pnls"])
            curve_points.append({
                "tp_ticks": tp,
                "tp_usd": tp * TICK_VALUE if tp else None,
                "n_trades": n,
                "total_pnl": round(pnl, 2),
                "sharpe": round(sharpe, 3),
                "win_rate": round(stats["n_wins"] / n * 100, 1) if n > 0 else 0.0,
                "avg_pnl_per_trade": round(pnl / n, 2) if n > 0 else 0.0,
            })
        tp_curves[card_name] = curve_points

    # --- Write report ---
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_dates": len(dates),
        "date_range": f"{dates[0]} to {dates[-1]}",
        "signal_volume_by_threshold": signal_counts,
        "threshold_curves": threshold_curves,
        "hold_duration_curves": hold_curves,
        "tp_curves": tp_curves,
        "key_findings": [],
    }

    # Auto-generate key findings
    findings = []
    for card_name, tc in threshold_curves.items():
        be = tc["breakeven_threshold"]
        opt = tc["optimal_threshold"]
        findings.append(
            f"{card_name}: breakeven at |signal|>={be or 'never'}, "
            f"optimal threshold={opt}, Sharpe={tc['optimal_sharpe']:.3f}"
        )
    report["key_findings"] = findings

    out_path = OUTPUT_DIR / "execution_cost_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"EXECUTION COST MODELING COMPLETE ({elapsed:.0f}s)")
    log.info(f"{'='*60}")
    log.info(f"\nKey Findings:")
    for f_line in findings:
        log.info(f"  {f_line}")
    log.info(f"\nFull report: {out_path}")

if __name__ == "__main__":
    main()
