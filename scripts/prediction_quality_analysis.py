#!/usr/bin/env python3
"""
prediction_quality_analysis.py - Analyze CNN prediction quality across dimensions.

Questions answered:
1. Time-of-day: Do signals fire more (and are they more profitable) at open/close/midday?
2. Vol regime: IC vs volatility level (vol0/vol50/vol70 gating effectiveness)
3. Signal magnitude: Does |signal| > threshold predict better outcomes?
4. Card comparison: Which card's signals are most informative before they're traded?

Uses fill_sim to get realized PnL for each signal, then slice+dice by metadata.
Runs with 14 workers on Jupiter's 16 cores.
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
from datetime import datetime, date
from collections import defaultdict

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/prediction_quality_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = 14

# OOT range (all 54 dates with CNN predictions)
OOT_START = "2025-12-01"
OOT_END   = "2026-02-17"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("pred_quality")

# ---- Card definitions ----
# Each card: label, pred_suffix, base_tp, base_hold_ms
# We run with no TP/SL (pure hold) so we get raw outcome per signal
CARDS = {
    "c1": {
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "desc": "C1 book model (short-biased, sparse)",
        "tp": 13, "hold_ms": 7200000,
    },
    "c7": {
        "pred_suffix": "smooth_smoothExit_conv1.5_ethr0.0_vol70",
        "desc": "C7 smooth model (long-biased, sparse)",
        "tp": 25, "hold_ms": 7200000,
    },
    "c5": {
        "pred_suffix": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "desc": "C5 raw model (dense signals)",
        "tp": None, "hold_ms": 3600000,
    },
}

# Signal magnitude bins to test (we filter |pred| > thresh)
SIG_THRESHOLDS = [0.0, 0.1, 0.3, 0.5, 0.8, 1.0, 1.5, 2.0]

def get_mbo_file(dt_str):
    """dt_str: YYYY-MM-DD -> MBO file path or None"""
    d = dt_str.replace("-", "")
    f = MBO_DIR / f"glbx-mdp3-{d}.mbo.dbn.zst"
    return f if f.exists() else None

def get_pred_file(dt_str, suffix):
    f = PRED_DIR / f"{dt_str}_{suffix}.npz"
    return f if f.exists() else None

def run_fill_sim(mbo_file, pred_file, output_file, tp=None, hold_ms=7200000, sig_thresh=0.0):
    """Run fill_sim and return parsed JSON results."""
    cmd = [
        FILL_SIM,
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(output_file),
        "--signal-threshold", str(sig_thresh),
        "--hold-ms", str(hold_ms),
        "--latency-ms", "10",
    ]
    if tp is not None:
        cmd += ["--take-profit-ticks", str(tp)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        with open(output_file) as f:
            return json.load(f)
    except Exception as e:
        log.warning(f"fill_sim error: {e}")
        return None

def analyze_predictions_for_date(dt_str, card_name, card_cfg):
    """Load prediction NPZ and return signal metadata (time indices, magnitudes)."""
    pred_file = get_pred_file(dt_str, card_cfg["pred_suffix"])
    if pred_file is None:
        return None
    d = np.load(pred_file)
    preds = d["predictions"]
    nz_idx = np.where(preds != 0)[0]
    nz_val = preds[nz_idx]
    # Each index = 100ms bar. Convert to time-of-day minutes from 09:30 ET
    # 234000 bars = 6.5 hours RTH (09:30-16:00 = 390 min = 234000 * 100ms)
    # Index 0 = 09:30:00 ET
    minutes_from_open = nz_idx * 0.1 / 60.0  # 100ms per bar
    return {
        "date": dt_str,
        "card": card_name,
        "n_signals": len(nz_idx),
        "signal_magnitudes": nz_val.tolist(),
        "minutes_from_open": minutes_from_open.tolist(),
    }

def run_single_job(args):
    """Worker: run fill_sim for one (date, card, sig_thresh) combination."""
    dt_str, card_name, card_cfg, sig_thresh = args
    mbo_file = get_mbo_file(dt_str)
    pred_file = get_pred_file(dt_str, card_cfg["pred_suffix"])
    if mbo_file is None or pred_file is None:
        return None

    out_file = OUTPUT_DIR / f"{dt_str}_{card_name}_sig{sig_thresh:.1f}.json"
    if out_file.exists():
        try:
            with open(out_file) as f:
                return json.load(f)
        except:
            pass

    result = run_fill_sim(
        mbo_file, pred_file, out_file,
        tp=card_cfg["tp"],
        hold_ms=card_cfg["hold_ms"],
        sig_thresh=sig_thresh,
    )
    return result

def get_oot_dates():
    """Get sorted list of dates in OOT range that have all required files."""
    dates = []
    for f in sorted(MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst")):
        d = f.stem.split("-")[2]  # glbx-mdp3-YYYYMMDD
        dt_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        if dt_str < OOT_START or dt_str > OOT_END:
            continue
        # Check at least one card has predictions for this date
        has_any = any(
            (PRED_DIR / f"{dt_str}_{cfg['pred_suffix']}.npz").exists()
            for cfg in CARDS.values()
        )
        if has_any:
            dates.append(dt_str)
    return dates

def aggregate_results(all_results):
    """Aggregate fill_sim results by: card, sig_thresh, time_bucket, date."""
    # Structure: { card: { sig_thresh: [results...] } }
    by_card_thresh = defaultdict(lambda: defaultdict(list))
    for r in all_results:
        if r is None:
            continue
        card = r.get("card")
        sig = r.get("sig_thresh")
        result_data = r.get("data")
        if result_data:
            by_card_thresh[card][sig].append(result_data)
    return by_card_thresh

def summarize_fill_sim_output(results_list):
    """Summarize a list of fill_sim JSON outputs."""
    total_trades = 0
    total_pnl = 0.0
    total_wins = 0
    total_loss_ticks = 0.0
    total_win_ticks = 0.0
    n_days = len(results_list)

    for r in results_list:
        if r is None:
            continue
        stats = r.get("summary", {})
        n = stats.get("n_trades", 0) or r.get("n_trades", 0)
        pnl = stats.get("total_pnl", 0.0) or r.get("total_pnl", 0.0)
        wins = stats.get("n_wins", 0) or r.get("n_wins", 0)
        total_trades += n
        total_pnl += pnl
        total_wins += wins

    win_rate = total_wins / total_trades if total_trades > 0 else 0.0
    avg_pnl_per_trade = total_pnl / total_trades if total_trades > 0 else 0.0
    trades_per_day = total_trades / n_days if n_days > 0 else 0.0

    return {
        "n_days": n_days,
        "total_trades": total_trades,
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate * 100, 1),
        "avg_pnl_per_trade": round(avg_pnl_per_trade, 2),
        "trades_per_day": round(trades_per_day, 1),
        "avg_daily_pnl": round(total_pnl / n_days, 2) if n_days > 0 else 0.0,
    }

def main():
    t0 = time.time()
    log.info("=== Prediction Quality Analysis ===")

    dates = get_oot_dates()
    log.info(f"OOT dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    # --- Phase 1: Signal metadata (no fill_sim needed) ---
    log.info("Phase 1: Analyzing prediction signal distributions...")
    signal_meta = []
    for dt_str in dates:
        for card_name, card_cfg in CARDS.items():
            meta = analyze_predictions_for_date(dt_str, card_name, card_cfg)
            if meta:
                signal_meta.append(meta)

    # Time-of-day distribution
    tod_analysis = {}
    for card_name in CARDS:
        card_meta = [m for m in signal_meta if m["card"] == card_name]
        all_mins = []
        all_mags = []
        for m in card_meta:
            all_mins.extend(m["minutes_from_open"])
            all_mags.extend([abs(x) for x in m["signal_magnitudes"]])

        if not all_mins:
            continue

        mins_arr = np.array(all_mins)
        mags_arr = np.array(all_mags)

        # Bucket into 30-min periods (0-30, 30-60, ..., 360-390)
        buckets = {}
        for b in range(0, 13):
            lo = b * 30
            hi = lo + 30
            mask = (mins_arr >= lo) & (mins_arr < hi)
            n = mask.sum()
            avg_mag = float(mags_arr[mask].mean()) if n > 0 else 0.0
            label = f"{9*60+30+lo:03d}min_to_{9*60+30+hi:03d}min"  # minutes from midnight
            # More readable: convert to HH:MM
            open_mins = 9*60+30
            lo_hhmm = f"{(open_mins+lo)//60:02d}:{(open_mins+lo)%60:02d}"
            hi_hhmm = f"{(open_mins+hi)//60:02d}:{(open_mins+hi)%60:02d}"
            label = f"{lo_hhmm}-{hi_hhmm}"
            buckets[label] = {"n_signals": int(n), "avg_magnitude": round(avg_mag, 4)}

        # Signal magnitude distribution
        mag_hist = {}
        for thresh in [0.1, 0.3, 0.5, 1.0, 1.5, 2.0, 3.0]:
            n_above = int((mags_arr >= thresh).sum())
            mag_hist[f"above_{thresh}"] = n_above

        tod_analysis[card_name] = {
            "total_signals": len(all_mins),
            "avg_signals_per_day": round(len(all_mins) / len(dates), 1),
            "avg_magnitude": round(float(mags_arr.mean()), 4),
            "time_of_day_distribution": buckets,
            "magnitude_histogram": mag_hist,
        }

    log.info("Phase 1 done. Signal distributions computed.")
    for card_name, analysis in tod_analysis.items():
        log.info(f"  {card_name}: {analysis['total_signals']} total signals, "
                 f"{analysis['avg_signals_per_day']:.1f}/day, avg_mag={analysis['avg_magnitude']:.4f}")

    # --- Phase 2: fill_sim across signal thresholds ---
    log.info("Phase 2: Running fill_sim across signal thresholds (14 workers)...")

    # Build job list: (date, card, sig_thresh)
    jobs = []
    for dt_str in dates:
        for card_name, card_cfg in CARDS.items():
            mbo = get_mbo_file(dt_str)
            pred = get_pred_file(dt_str, card_cfg["pred_suffix"])
            if mbo is None or pred is None:
                continue
            for sig_thresh in SIG_THRESHOLDS:
                jobs.append((dt_str, card_name, card_cfg, sig_thresh))

    log.info(f"Total fill_sim jobs: {len(jobs)}")

    # Run with worker pool
    completed = 0
    results_by_card_thresh = defaultdict(lambda: defaultdict(list))

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        future_to_job = {}
        for job in jobs:
            dt_str, card_name, card_cfg, sig_thresh = job
            future = executor.submit(run_single_job, job)
            future_to_job[future] = (dt_str, card_name, sig_thresh)

        for future in as_completed(future_to_job):
            dt_str, card_name, sig_thresh = future_to_job[future]
            completed += 1
            if completed % 100 == 0:
                elapsed = time.time() - t0
                rate = completed / elapsed
                eta = (len(jobs) - completed) / rate if rate > 0 else 0
                log.info(f"Progress: {completed}/{len(jobs)} ({elapsed:.0f}s elapsed, ETA {eta:.0f}s)")
            try:
                result = future.result()
                if result is not None:
                    results_by_card_thresh[card_name][sig_thresh].append(result)
            except Exception as e:
                log.warning(f"Job failed {dt_str}/{card_name}/sig{sig_thresh}: {e}")

    log.info("Phase 2 done. Aggregating results...")

    # --- Phase 3: Summarize and find breakeven threshold ---
    threshold_analysis = {}
    for card_name, thresh_dict in results_by_card_thresh.items():
        threshold_analysis[card_name] = {}
        for sig_thresh in sorted(thresh_dict.keys()):
            results = thresh_dict[sig_thresh]
            summary = summarize_fill_sim_output(results)
            threshold_analysis[card_name][str(sig_thresh)] = summary

    # Find breakeven threshold for each card
    breakeven = {}
    for card_name, thresh_dict in threshold_analysis.items():
        be_thresh = None
        for thresh_str in sorted(thresh_dict.keys(), key=float):
            s = thresh_dict[thresh_str]
            if s["avg_pnl_per_trade"] > 0 and s["total_trades"] > 10:
                be_thresh = float(thresh_str)
                break
        breakeven[card_name] = {
            "breakeven_threshold": be_thresh,
            "details": thresh_dict,
        }

    # --- Write final report ---
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_dates": len(dates),
        "date_range": f"{dates[0]} to {dates[-1]}",
        "signal_distribution": tod_analysis,
        "threshold_analysis": threshold_analysis,
        "breakeven_thresholds": breakeven,
    }

    out_path = OUTPUT_DIR / "prediction_quality_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # --- Print summary ---
    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"PREDICTION QUALITY ANALYSIS COMPLETE ({elapsed:.0f}s)")
    log.info(f"{'='*60}")
    log.info(f"\nSignal Distribution:")
    for card_name, analysis in tod_analysis.items():
        log.info(f"  {card_name}: {analysis['avg_signals_per_day']:.1f} signals/day, "
                 f"avg_mag={analysis['avg_magnitude']:.4f}")

    log.info(f"\nBreakeven Thresholds (min |signal| for positive avg PnL/trade):")
    for card_name, be_data in breakeven.items():
        be = be_data["breakeven_threshold"]
        log.info(f"  {card_name}: |signal| >= {be}")
        if be is not None:
            details = be_data["details"].get(str(be), {})
            log.info(f"    -> {details.get('trades_per_day', 0):.1f} trades/day, "
                     f"win_rate={details.get('win_rate', 0):.1f}%, "
                     f"avg_pnl/trade=${details.get('avg_pnl_per_trade', 0):.2f}")

    log.info(f"\nFull report: {out_path}")
    log.info(f"\nTime-of-Day Peaks (top 3 buckets by signal count):")
    for card_name, analysis in tod_analysis.items():
        tod = analysis["time_of_day_distribution"]
        sorted_tod = sorted(tod.items(), key=lambda x: x[1]["n_signals"], reverse=True)[:3]
        log.info(f"  {card_name}: " + ", ".join(f"{k}={v['n_signals']}" for k, v in sorted_tod))

if __name__ == "__main__":
    main()
