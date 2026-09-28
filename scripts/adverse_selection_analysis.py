#!/usr/bin/env python3
"""
adverse_selection_analysis.py - Analyze adverse selection in paper trades.

Problem: IC = 0.07-0.17 but paper PnL = -$131 across 157 trades.
Some cards profitable (C4: +$1128, C9S: +$1131), others lose (C1: -$789, C10L: -$1507).

This script uses fill_sim to measure:
1. Are we entering at the wrong time within the signal sequence?
   (e.g., signal fires at 0.5, we enter, then signal goes to 2.0 before our TP)
2. Does the fill_sim (ideal) outperform paper by how much?
   The gap = execution friction + adverse selection
3. Long vs short asymmetry: does the model have directional bias?
4. Entry timing: how long after signal fires does the trade trigger?
5. Signal persistence: do high-mag signals sustain longer?

Runs fill_sim in "diagnostic mode" (no TP/SL, full hold) then with optimal params,
comparing ideal vs constrained outcomes.
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
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/adverse_selection_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WORKERS = 12
OOT_START = "2025-12-01"
OOT_END   = "2026-02-17"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("adverse_sel")

# Cards to analyze in depth
CARDS = {
    "c1": {
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "desc": "C1 book model (was -$789 in paper)",
        "optimal_tp": 13,
        "optimal_hold_ms": 7200000,
        "paper_result": -789,
    },
    "c7": {
        "pred_suffix": "smooth_smoothExit_conv1.5_ethr0.0_vol70",
        "desc": "C7 smooth model",
        "optimal_tp": 25,
        "optimal_hold_ms": 7200000,
        "paper_result": None,
    },
    "c5": {
        "pred_suffix": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "desc": "C5 raw model",
        "optimal_tp": None,
        "optimal_hold_ms": 3600000,
        "paper_result": None,
    },
}

# Compare: passive limit entry vs chase entry (1 tick, 3 reprices)
ENTRY_MODES = [
    {"label": "passive", "chase_ticks": 0, "chase_reprices": 0},
    # Note: fill_sim doesn't have explicit chase param in current version
    # We simulate by reducing latency (0ms = best possible entry timing)
    {"label": "latency0", "latency_ms": 0},
    {"label": "latency10", "latency_ms": 10},
    {"label": "latency50", "latency_ms": 50},
    {"label": "latency100", "latency_ms": 100},
]

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
                  sig=0.1, latency_ms=10, max_wait_bars=None):
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
    if max_wait_bars is not None:
        cmd += ["--max-wait-bars", str(max_wait_bars)]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        with open(out_file) as f:
            return json.load(f)
    except Exception as e:
        return None

def analyze_signal_autocorrelation(dt_str, card_cfg):
    """Analyze how signals evolve over time - do they strengthen after entry?"""
    pred_file = get_pred_file(dt_str, card_cfg["pred_suffix"])
    if pred_file is None:
        return None

    d = np.load(pred_file)
    preds = d["predictions"]

    # Find signal onset points (transitions from 0 to non-zero, or crossings of threshold)
    nz_mask = preds != 0
    onsets = []
    i = 0
    while i < len(preds):
        if nz_mask[i]:
            # Find the end of this signal run
            j = i
            while j < len(preds) and nz_mask[j]:
                j += 1
            segment = preds[i:j]
            if len(segment) >= 2:
                # Analyze: does magnitude increase or decrease over the signal run?
                start_mag = abs(segment[0])
                peak_mag = np.max(np.abs(segment))
                end_mag = abs(segment[-1])
                onsets.append({
                    "start_idx": i,
                    "length": len(segment),
                    "start_mag": float(start_mag),
                    "peak_mag": float(peak_mag),
                    "end_mag": float(end_mag),
                    "peak_vs_start": float(peak_mag / start_mag) if start_mag > 0 else 0.0,
                    "direction": 1 if segment[0] > 0 else -1,
                })
            i = j
        else:
            i += 1

    if not onsets:
        return None

    # Aggregate
    peak_vs_start = np.array([o["peak_vs_start"] for o in onsets])
    signal_lengths = np.array([o["length"] for o in onsets])
    start_mags = np.array([o["start_mag"] for o in onsets])
    long_frac = np.mean([1 if o["direction"] > 0 else 0 for o in onsets])

    return {
        "date": dt_str,
        "n_signals": len(onsets),
        "avg_signal_length_bars": float(signal_lengths.mean()),
        "avg_signal_length_ms": float(signal_lengths.mean() * 100),
        "avg_peak_vs_start_ratio": float(peak_vs_start.mean()),
        "pct_signals_that_strengthen": float((peak_vs_start > 1.5).mean() * 100),
        "long_fraction": float(long_frac),
        "short_fraction": float(1 - long_frac),
        "avg_start_magnitude": float(start_mags.mean()),
    }

def run_latency_comparison(args):
    """Worker: compare different latency settings for one (date, card)."""
    dt_str, card_name, card_cfg = args
    mbo_file = get_mbo_file(dt_str)
    pred_file = get_pred_file(dt_str, card_cfg["pred_suffix"])
    if mbo_file is None or pred_file is None:
        return None

    results = {}
    for latency_ms in [0, 10, 50, 100, 200, 500]:
        out_file = OUTPUT_DIR / f"{dt_str}_{card_name}_lat{latency_ms}.json"
        r = run_fill_sim(
            mbo_file, pred_file, out_file,
            tp=card_cfg["optimal_tp"],
            hold_ms=card_cfg["optimal_hold_ms"],
            sig=0.1,
            latency_ms=latency_ms,
        )
        if r:
            n = (r.get("summary", {}).get("total_trades") or r.get("total_trades") or 0)
            pnl = (r.get("summary", {}).get("total_pnl_dollars") or r.get("total_pnl_dollars") or 0.0)
            wins = round((r.get("win_rate") or r.get("summary", {}).get("win_rate") or 0.0) * n)
            results[f"lat_{latency_ms}ms"] = {
                "n_trades": n,
                "total_pnl": round(pnl, 2),
                "win_rate": round(wins / n * 100, 1) if n > 0 else 0.0,
                "avg_pnl_per_trade": round(pnl / n, 2) if n > 0 else 0.0,
            }

    return {"date": dt_str, "card": card_name, "results": results}

def run_max_wait_analysis(args):
    """Worker: test effect of max_wait_bars on fill rate and PnL."""
    dt_str, card_name, card_cfg = args
    mbo_file = get_mbo_file(dt_str)
    pred_file = get_pred_file(dt_str, card_cfg["pred_suffix"])
    if mbo_file is None or pred_file is None:
        return None

    results = {}
    for max_wait in [None, 5, 10, 20, 50, 100]:  # None = wait forever
        label = f"wait_{max_wait or 'inf'}"
        out_file = OUTPUT_DIR / f"{dt_str}_{card_name}_{label}.json"
        r = run_fill_sim(
            mbo_file, pred_file, out_file,
            tp=card_cfg["optimal_tp"],
            hold_ms=card_cfg["optimal_hold_ms"],
            sig=0.1,
            latency_ms=10,
            max_wait_bars=max_wait,
        )
        if r:
            n = (r.get("summary", {}).get("total_trades") or r.get("total_trades") or 0)
            pnl = (r.get("summary", {}).get("total_pnl_dollars") or r.get("total_pnl_dollars") or 0.0)
            results[label] = {
                "n_trades": n,
                "total_pnl": round(pnl, 2),
                "avg_pnl_per_trade": round(pnl / n, 2) if n > 0 else 0.0,
            }

    return {"date": dt_str, "card": card_name, "results": results}

def main():
    t0 = time.time()
    log.info("=== Adverse Selection Analysis ===")

    dates = get_oot_dates()
    log.info(f"OOT dates: {len(dates)} ({dates[0]} to {dates[-1]})")

    # --- Phase 1: Signal autocorrelation ---
    log.info("Phase 1: Signal autocorrelation (do signals strengthen after entry?)...")
    autocorr_results = defaultdict(list)
    for dt_str in dates:
        for card_name, card_cfg in CARDS.items():
            result = analyze_signal_autocorrelation(dt_str, card_cfg)
            if result:
                autocorr_results[card_name].append(result)

    autocorr_summary = {}
    for card_name, results in autocorr_results.items():
        avg_strengthen = np.mean([r["pct_signals_that_strengthen"] for r in results])
        avg_long_frac = np.mean([r["long_fraction"] for r in results])
        avg_peak_ratio = np.mean([r["avg_peak_vs_start_ratio"] for r in results])
        avg_length_ms = np.mean([r["avg_signal_length_ms"] for r in results])
        autocorr_summary[card_name] = {
            "avg_pct_signals_strengthen": round(float(avg_strengthen), 1),
            "avg_long_fraction": round(float(avg_long_frac), 3),
            "avg_short_fraction": round(float(1 - avg_long_frac), 3),
            "avg_peak_vs_start_ratio": round(float(avg_peak_ratio), 3),
            "avg_signal_duration_ms": round(float(avg_length_ms), 0),
            "interpretation": (
                "Signals tend to STRENGTHEN after entry (adverse selection risk low)"
                if avg_peak_ratio > 2.0 else
                "Signals tend to WEAKEN after entry (adverse selection risk HIGH)"
            ),
        }
        log.info(f"  {card_name}: {avg_strengthen:.1f}% strengthen, "
                 f"long_frac={avg_long_frac:.2f}, peak/start={avg_peak_ratio:.2f}x, "
                 f"duration={avg_length_ms:.0f}ms")

    # --- Phase 2: Latency impact ---
    log.info("Phase 2: Latency impact analysis...")
    latency_jobs = []
    for dt_str in dates:
        for card_name, card_cfg in CARDS.items():
            latency_jobs.append((dt_str, card_name, card_cfg))

    latency_results = defaultdict(list)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_latency_comparison, job): job for job in latency_jobs}
        done = 0
        for future in as_completed(futures):
            done += 1
            if done % 50 == 0:
                log.info(f"  Latency jobs: {done}/{len(latency_jobs)}")
            try:
                r = future.result()
                if r:
                    latency_results[r["card"]].append(r)
            except Exception as e:
                pass

    # Aggregate latency results per card
    latency_summary = {}
    for card_name, results in latency_results.items():
        by_latency = defaultdict(lambda: {"total_pnl": 0.0, "n_trades": 0, "n_days": 0})
        for r in results:
            for lat_label, stats in r["results"].items():
                by_latency[lat_label]["total_pnl"] += stats["total_pnl"]
                by_latency[lat_label]["n_trades"] += stats["n_trades"]
                by_latency[lat_label]["n_days"] += 1

        latency_summary[card_name] = {}
        baseline_pnl = None
        for lat_label in sorted(by_latency.keys()):
            s = by_latency[lat_label]
            n = s["n_trades"]
            pnl = s["total_pnl"]
            avg_per_trade = pnl / n if n > 0 else 0.0
            if baseline_pnl is None:
                baseline_pnl = pnl
            latency_summary[card_name][lat_label] = {
                "total_pnl": round(pnl, 2),
                "n_trades": n,
                "avg_pnl_per_trade": round(avg_per_trade, 2),
                "pnl_vs_0ms": round(pnl - (baseline_pnl or 0), 2),
            }

        log.info(f"  {card_name} latency impact:")
        for lat_label, stats in latency_summary[card_name].items():
            log.info(f"    {lat_label}: ${stats['total_pnl']:,.0f} total, "
                     f"${stats['avg_pnl_per_trade']:.2f}/trade, "
                     f"Δ${stats['pnl_vs_0ms']:,.0f} vs 0ms")

    # --- Phase 3: Max wait bars (passive fill rate analysis) ---
    log.info("Phase 3: Max wait analysis (passive fill rate)...")
    wait_jobs = [(dt_str, card_name, card_cfg)
                 for dt_str in dates[:20]  # Use 20 dates for this analysis
                 for card_name, card_cfg in CARDS.items()]

    wait_results = defaultdict(list)
    with ProcessPoolExecutor(max_workers=WORKERS) as executor:
        futures = {executor.submit(run_max_wait_analysis, job): job for job in wait_jobs}
        for future in as_completed(futures):
            try:
                r = future.result()
                if r:
                    wait_results[r["card"]].append(r)
            except:
                pass

    wait_summary = {}
    for card_name, results in wait_results.items():
        by_wait = defaultdict(lambda: {"total_pnl": 0.0, "n_trades": 0})
        for r in results:
            for wait_label, stats in r["results"].items():
                by_wait[wait_label]["total_pnl"] += stats["total_pnl"]
                by_wait[wait_label]["n_trades"] += stats["n_trades"]
        wait_summary[card_name] = {
            k: {"total_pnl": round(v["total_pnl"], 2), "n_trades": v["n_trades"]}
            for k, v in by_wait.items()
        }
        log.info(f"  {card_name} wait analysis:")
        for wait_label in sorted(by_wait.keys()):
            s = by_wait[wait_label]
            log.info(f"    {wait_label}: {s['n_trades']} trades, ${s['total_pnl']:,.0f}")

    # --- Write report ---
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_dates": len(dates),
        "date_range": f"{dates[0]} to {dates[-1]}",
        "signal_autocorrelation": autocorr_summary,
        "latency_impact": latency_summary,
        "max_wait_analysis": wait_summary,
        "conclusions": [],
    }

    # Auto-generate conclusions
    conclusions = []
    for card_name, ac in autocorr_summary.items():
        if ac["avg_long_fraction"] > 0.7:
            conclusions.append(f"{card_name}: STRONGLY long-biased ({ac['avg_long_fraction']*100:.0f}% long signals)")
        elif ac["avg_long_fraction"] < 0.3:
            conclusions.append(f"{card_name}: STRONGLY short-biased ({(1-ac['avg_long_fraction'])*100:.0f}% short signals)")
        if ac["avg_peak_vs_start_ratio"] < 1.2:
            conclusions.append(f"{card_name}: Signals DON'T strengthen after entry — possible adverse selection")
        else:
            conclusions.append(f"{card_name}: Signals strengthen {ac['avg_peak_vs_start_ratio']:.1f}x after entry — favorable")

    report["conclusions"] = conclusions

    out_path = OUTPUT_DIR / "adverse_selection_report.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2)

    elapsed = time.time() - t0
    log.info(f"\n{'='*60}")
    log.info(f"ADVERSE SELECTION ANALYSIS COMPLETE ({elapsed:.0f}s)")
    log.info(f"{'='*60}")
    log.info(f"\nKey Conclusions:")
    for c in conclusions:
        log.info(f"  - {c}")
    log.info(f"\nFull report: {out_path}")

if __name__ == "__main__":
    main()
