#!/usr/bin/env python3
"""
expanding_window_card_sweep.py - Re-optimize all cards using OOT prediction data.

The 371-sweep was done on sliding-window predictions.
We now have 54 OOT dates (Dec 2025 - Feb 2026) with full predictions.
This sweep re-runs the full parameter grid on the OOT data to find:
- Best TP/SL for each card type
- Best hold duration
- Best signal threshold
- Best MAE exit params
- Long vs short side breakdown

Covers Cards 1, 5, 7 (the 3 with OOT predictions).
Also sweeps over ALL available pred_suffixes to find best-performing model variant.

Runs with 14 workers, ~300-500 configs × 54 dates = ~20K fill_sim calls.
ETA: ~30-45 min on Jupiter.
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
import itertools

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/expanding_window_sweep")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = 14
OOT_START = "2025-12-01"
OOT_END   = "2026-02-17"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("ew_sweep")

# ---- Sweep space ----
# Focused on what we know works + exploration around it

# Card 1: book model, short-biased. Best found: TP13/2h
C1_VARIANTS = []
for tp in [8, 10, 12, 13, 14, 15, 18, 20, 25]:
    for hold_h in [1, 2]:
        for sig in [0.0, 0.1, 0.3, 0.5]:
            for mae_t, mae_s in [(0, 0), (20, 300), (25, 300), (30, 600)]:
                label = f"c1_tp{tp}_h{hold_h}h_sig{sig:.1f}_mae{mae_t}t{mae_s}s"
                C1_VARIANTS.append({
                    "card": "c1",
                    "pred_suffix": "book_predstdExit_conv1.5_vol50",
                    "tp": tp, "sl": None,
                    "hold_ms": hold_h * 3600000,
                    "sig": sig,
                    "mae_ticks": mae_t, "mae_secs": mae_s,
                    "label": label,
                })

# Card 7: smooth model, long-biased. Best found: TP25/2h
C7_VARIANTS = []
for tp in [15, 20, 22, 25, 28, 30, 35]:
    for hold_h in [1, 2]:
        for sig in [0.0, 0.1, 0.3, 0.5]:
            for mae_t, mae_s in [(0, 0), (20, 300), (25, 300)]:
                label = f"c7_tp{tp}_h{hold_h}h_sig{sig:.1f}_mae{mae_t}t{mae_s}s"
                C7_VARIANTS.append({
                    "card": "c7",
                    "pred_suffix": "smooth_smoothExit_conv1.5_ethr0.0_vol70",
                    "tp": tp, "sl": None,
                    "hold_ms": hold_h * 3600000,
                    "sig": sig,
                    "mae_ticks": mae_t, "mae_secs": mae_s,
                    "label": label,
                })

# Card 5: raw model, dense signals. Best found: MAE50t/300s/1h
C5_VARIANTS = []
for tp in [None, 15, 20, 25, 30]:
    for hold_h in [1, 2]:
        for sig in [0.0, 0.1, 0.3]:
            for mae_t, mae_s in [(0, 0), (30, 300), (40, 300), (50, 300), (50, 600)]:
                label = f"c5_tp{tp or 'none'}_h{hold_h}h_sig{sig:.1f}_mae{mae_t}t{mae_s}s"
                C5_VARIANTS.append({
                    "card": "c5",
                    "pred_suffix": "raw_rawExit_conv0.05_ethr0.5_vol0",
                    "tp": tp, "sl": None,
                    "hold_ms": hold_h * 3600000,
                    "sig": sig,
                    "mae_ticks": mae_t, "mae_secs": mae_s,
                    "label": label,
                })

# Also explore other pred_suffixes for C1 (to find if conv2.0/2.5 beats conv1.5)
C1_VARIANT_SUFFIXES = []
for suf in ["book_predstdExit_conv2.0_vol50", "book_predstdExit_conv2.5_vol50",
             "book_predstdExit_conv1.5_vol0", "book_predstdExit_conv1.5_vol70"]:
    label = f"c1_suffix_{suf}_tp13_h2h"
    C1_VARIANT_SUFFIXES.append({
        "card": "c1_varsuf",
        "pred_suffix": suf,
        "tp": 13, "sl": None,
        "hold_ms": 7200000,
        "sig": 0.1,
        "mae_ticks": 0, "mae_secs": 0,
        "label": label,
    })

ALL_VARIANTS = C1_VARIANTS + C7_VARIANTS + C5_VARIANTS + C1_VARIANT_SUFFIXES
log.info(f"Total variants: {len(ALL_VARIANTS)}")


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
        stem = f.stem  # glbx-mdp3-YYYYMMDD.mbo.dbn
        parts = stem.split("-")
        if len(parts) < 3:
            continue
        d = parts[2]
        dt_str = f"{d[:4]}-{d[4:6]}-{d[6:8]}"
        if dt_str < OOT_START or dt_str > OOT_END:
            continue
        dates.append(dt_str)
    return sorted(dates)

def run_single_variant_date(args):
    """Worker: (variant_cfg, dt_str) -> summary dict or None"""
    variant, dt_str = args
    mbo_file = get_mbo_file(dt_str)
    pred_file = get_pred_file(dt_str, variant["pred_suffix"])
    if mbo_file is None or pred_file is None:
        return None

    out_file = OUTPUT_DIR / f"{dt_str}_{variant['label']}.json"
    if out_file.exists():
        try:
            with open(out_file) as f:
                data = json.load(f)
                data["_date"] = dt_str
                data["_label"] = variant["label"]
                return data
        except:
            pass

    cmd = [
        FILL_SIM,
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--signal-threshold", str(variant["sig"]),
        "--hold-ms", str(variant["hold_ms"]),
        "--latency-ms", "10",
    ]
    if variant["tp"] is not None:
        cmd += ["--take-profit-ticks", str(variant["tp"])]
    if variant.get("mae_ticks", 0) > 0:
        cmd += [
            "--vol-exit-ticks", str(variant["mae_ticks"]),
            "--vol-exit-bars", str(variant["mae_secs"] * 10),  # 100ms bars
        ]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        with open(out_file) as f:
            data = json.load(f)
            data["_date"] = dt_str
            data["_label"] = variant["label"]
            return data
    except Exception as e:
        return None

def extract_stats(results_list):
    """Aggregate fill_sim results for one variant across all dates."""
    total_trades = 0
    total_pnl = 0.0
    total_wins = 0
    daily_pnls = []
    n_dates = len(results_list)

    for r in results_list:
        if r is None:
            continue
        # fill_sim JSON structure: varies by version, be flexible
        n = (r.get("summary", {}).get("n_trades") or
             r.get("n_trades") or
             r.get("total_trades") or 0)
        pnl = (r.get("summary", {}).get("total_pnl") or
               r.get("total_pnl") or
               r.get("pnl") or 0.0)
        wins = (r.get("summary", {}).get("n_wins") or
                r.get("n_wins") or
                r.get("wins") or 0)
        total_trades += n
        total_pnl += pnl
        total_wins += wins
        daily_pnls.append(pnl)

    if total_trades == 0:
        return None

    daily_pnls = np.array(daily_pnls)
    sharpe = 0.0
    if len(daily_pnls) > 1 and daily_pnls.std() > 0:
        sharpe = float(daily_pnls.mean() / daily_pnls.std() * np.sqrt(252))

    return {
        "n_days": n_dates,
        "total_trades": total_trades,
        "total_pnl": round(total_pnl, 2),
        "avg_daily_pnl": round(total_pnl / n_dates, 2) if n_dates > 0 else 0.0,
        "trades_per_day": round(total_trades / n_dates, 1) if n_dates > 0 else 0.0,
        "win_rate": round(total_wins / total_trades * 100, 1) if total_trades > 0 else 0.0,
        "sharpe": round(sharpe, 3),
        "max_dd": round(float(-min(np.minimum.accumulate(np.cumsum(daily_pnls)))), 2),
    }

def main():
    t0 = time.time()
    log.info("=== Expanding Window Card Sweep ===")

    dates = get_oot_dates()
    log.info(f"OOT dates: {len(dates)} ({dates[0] if dates else 'none'} to {dates[-1] if dates else 'none'})")

    # Build job list
    jobs = []
    for variant in ALL_VARIANTS:
        for dt_str in dates:
            mbo = get_mbo_file(dt_str)
            pred = get_pred_file(dt_str, variant["pred_suffix"])
            if mbo and pred:
                jobs.append((variant, dt_str))

    log.info(f"Total jobs: {len(jobs)} ({len(ALL_VARIANTS)} variants × ~{len(dates)} dates)")

    # Run
    results_by_label = defaultdict(list)
    completed = 0

    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        future_to_job = {executor.submit(run_single_variant_date, job): job for job in jobs}

        for future in as_completed(future_to_job):
            variant, dt_str = future_to_job[future]
            completed += 1
            if completed % 200 == 0 or completed == len(jobs):
                elapsed = time.time() - t0
                rate = completed / elapsed if elapsed > 0 else 1
                eta = (len(jobs) - completed) / rate
                log.info(f"Progress: {completed}/{len(jobs)} | elapsed={elapsed:.0f}s | ETA={eta:.0f}s")
            try:
                result = future.result()
                results_by_label[variant["label"]].append(result)
            except Exception as e:
                results_by_label[variant["label"]].append(None)

    log.info("All jobs done. Computing summaries...")

    # Aggregate per variant
    summaries = []
    for variant in ALL_VARIANTS:
        label = variant["label"]
        results = results_by_label[label]
        stats = extract_stats(results)
        if stats is None:
            continue
        summaries.append({
            "label": label,
            "card": variant["card"],
            "pred_suffix": variant["pred_suffix"],
            "tp": variant["tp"],
            "hold_ms": variant["hold_ms"],
            "sig": variant["sig"],
            "mae_ticks": variant.get("mae_ticks", 0),
            "mae_secs": variant.get("mae_secs", 0),
            **stats
        })

    # Sort by Sharpe
    summaries.sort(key=lambda x: x["sharpe"], reverse=True)

    # Top per card
    top_per_card = {}
    for card_name in ["c1", "c7", "c5", "c1_varsuf"]:
        card_results = [s for s in summaries if s["card"] == card_name and s["total_trades"] > 50]
        if card_results:
            top_per_card[card_name] = card_results[:5]

    # Write results
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_dates": len(dates),
        "date_range": f"{dates[0] if dates else ''} to {dates[-1] if dates else ''}",
        "n_variants": len(ALL_VARIANTS),
        "top_20_overall": summaries[:20],
        "top_per_card": top_per_card,
        "all_results": summaries,
    }

    out_path = OUTPUT_DIR / "expanding_window_sweep_summary.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    # Print top results
    elapsed = time.time() - t0
    log.info(f"\n{'='*70}")
    log.info(f"EXPANDING WINDOW SWEEP COMPLETE ({elapsed:.0f}s)")
    log.info(f"{'='*70}")
    log.info(f"\nTop 10 Overall (by Sharpe):")
    for s in summaries[:10]:
        log.info(f"  {s['label'][:60]:<60} | "
                 f"Sharpe={s['sharpe']:.3f} | PnL=${s['total_pnl']:,.0f} | "
                 f"WR={s['win_rate']:.0f}% | {s['trades_per_day']:.1f}t/day")

    log.info(f"\nTop Per Card:")
    for card_name, tops in top_per_card.items():
        if tops:
            best = tops[0]
            log.info(f"  {card_name}: {best['label'][:50]} | "
                     f"Sharpe={best['sharpe']:.3f} | PnL=${best['total_pnl']:,.0f}")

    log.info(f"\nFull report: {out_path}")

if __name__ == "__main__":
    main()
