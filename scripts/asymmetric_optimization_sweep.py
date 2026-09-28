#!/usr/bin/env python3
"""
asymmetric_optimization_sweep.py — Optimize long-only and short-only card variants.

Based on the side asymmetry discovery:
- C3/C4/C6/C7 are long-biased → create optimized long-only variants
- C1/C5 are short-biased → create optimized short-only variants

Tests variations of TP, hold time, signal threshold, and conviction exit
specifically tuned for each direction.

Run on Jupiter after 371-variant sweep completes.
"""

import os
import sys
import json
import glob
import time
import subprocess
import logging
import numpy as np
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/asymmetric_opt_sweep")

OOT_START = "2025-12-01"
OOT_END = "2026-03-08"
WORKERS = 14

# Long-biased cards: optimize for LONG trades
LONG_CARDS = {
    "L_c3": {
        "pred_suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "base_sig": 0.3,
        "desc": "C3 Long-Only optimized",
    },
    "L_c4": {
        "pred_suffix": "book_predstdExit_conv2.0_vol70",
        "base_sig": 0.1,
        "desc": "C4 Long-Only optimized",
    },
    "L_c6": {
        "pred_suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "base_sig": 0.1,
        "desc": "C6 Long-Only optimized",
    },
    "L_c2": {
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "base_sig": 0.5,
        "desc": "C2 Long-Only optimized",
    },
}

# Short-biased cards: optimize for SHORT trades
SHORT_CARDS = {
    "S_c1": {
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "base_sig": 0.1,
        "desc": "C1 Short-Only optimized",
    },
    "S_c5": {
        "pred_suffix": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "base_sig": 0.1,
        "desc": "C5 Short-Only optimized",
    },
}

# Long optimization grid: wider TPs, longer holds (longs trend)
LONG_VARIANTS = [
    # (tp, sl, hold_ms, sig, conv_bars, conv_mag, mae_ticks, mae_sec, prime_hours, label)
    (8, None, 3600000, 0.3, 100, 0.8, 0, 0, False, "tp8_1h"),
    (10, None, 3600000, 0.3, 100, 0.8, 0, 0, False, "tp10_1h"),
    (12, None, 3600000, 0.3, 100, 0.8, 0, 0, False, "tp12_1h"),
    (15, None, 3600000, 0.3, 100, 0.8, 0, 0, False, "tp15_1h"),
    (20, None, 3600000, 0.3, 100, 0.8, 0, 0, False, "tp20_1h"),
    (10, None, 7200000, 0.3, 100, 0.8, 0, 0, False, "tp10_2h"),
    (15, None, 7200000, 0.3, 100, 0.8, 0, 0, False, "tp15_2h"),
    (20, None, 7200000, 0.3, 100, 0.8, 0, 0, False, "tp20_2h"),
    # With MAE exit
    (10, None, 3600000, 0.3, 100, 0.8, 25, 60, False, "tp10_1h_mae25"),
    (15, None, 3600000, 0.3, 100, 0.8, 25, 60, False, "tp15_1h_mae25"),
    (10, None, 3600000, 0.3, 100, 0.8, 30, 120, False, "tp10_1h_mae30"),
    # Higher signal threshold
    (10, None, 3600000, 0.5, 100, 0.8, 0, 0, False, "tp10_1h_sig05"),
    (10, None, 3600000, 1.0, 100, 0.8, 0, 0, False, "tp10_1h_sig10"),
    (10, None, 3600000, 1.5, 100, 0.8, 0, 0, False, "tp10_1h_sig15"),
    # Prime hours only (10:30-14:30)
    (10, None, 3600000, 0.3, 100, 0.8, 0, 0, True, "tp10_1h_prime"),
    (15, None, 3600000, 0.3, 100, 0.8, 0, 0, True, "tp15_1h_prime"),
    # No conviction exit (baseline comparison)
    (10, None, 3600000, 0.3, 0, 0, 0, 0, False, "tp10_1h_noconv"),
    # Conviction variants
    (10, None, 3600000, 0.3, 60, 1.5, 0, 0, False, "tp10_1h_conv6s"),
    (10, None, 3600000, 0.3, 150, 0.5, 0, 0, False, "tp10_1h_conv15s"),
]

# Short optimization grid: tighter TPs, shorter holds (shorts spike fast)
SHORT_VARIANTS = [
    (4, None, 1800000, 0.1, 0, 0, 0, 0, False, "tp4_30m"),
    (6, None, 1800000, 0.1, 0, 0, 0, 0, False, "tp6_30m"),
    (8, None, 1800000, 0.1, 0, 0, 0, 0, False, "tp8_30m"),
    (10, None, 1800000, 0.1, 0, 0, 0, 0, False, "tp10_30m"),
    (4, None, 3600000, 0.1, 0, 0, 0, 0, False, "tp4_1h"),
    (6, None, 3600000, 0.1, 0, 0, 0, 0, False, "tp6_1h"),
    (8, None, 3600000, 0.1, 0, 0, 0, 0, False, "tp8_1h"),
    (10, None, 3600000, 0.1, 0, 0, 0, 0, False, "tp10_1h"),
    # With stop loss
    (8, 15, 1800000, 0.1, 0, 0, 0, 0, False, "tp8_30m_sl15"),
    (8, 20, 1800000, 0.1, 0, 0, 0, 0, False, "tp8_30m_sl20"),
    (10, 20, 3600000, 0.1, 0, 0, 0, 0, False, "tp10_1h_sl20"),
    # Higher signal threshold
    (8, None, 1800000, 0.5, 0, 0, 0, 0, False, "tp8_30m_sig05"),
    (8, None, 1800000, 1.0, 0, 0, 0, 0, False, "tp8_30m_sig10"),
    (8, None, 1800000, 1.5, 0, 0, 0, 0, False, "tp8_30m_sig15"),
    # With MAE exit
    (8, None, 1800000, 0.1, 0, 0, 20, 30, False, "tp8_30m_mae20"),
    (8, None, 1800000, 0.1, 0, 0, 15, 30, False, "tp8_30m_mae15"),
    # Prime hours
    (8, None, 1800000, 0.1, 0, 0, 0, 0, True, "tp8_30m_prime"),
    # Conviction exit
    (8, None, 1800000, 0.1, 60, 1.5, 0, 0, False, "tp8_30m_conv6s"),
    (8, None, 1800000, 0.1, 100, 1.0, 0, 0, False, "tp8_30m_conv10s"),
]

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("asym_opt")


def discover_dates():
    dates = []
    for f in sorted(MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst")):
        name = f.stem.split(".")[0]
        date_raw = name.split("-")[-1]
        if len(date_raw) == 8:
            ds = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
            if OOT_START <= ds <= OOT_END:
                dates.append(ds)
    return dates


def build_task(card_id, card_def, date_str, variant, direction):
    tp, sl, hold, sig, conv_bars, conv_mag, mae_t, mae_s, prime, label = variant
    date_compact = date_str.replace("-", "")
    mbo = MBO_DIR / f"glbx-mdp3-{date_compact}.mbo.dbn.zst"
    pred = PRED_DIR / f"{date_str}_{card_def['pred_suffix']}.npz"
    if not mbo.exists() or not pred.exists():
        return None

    out_dir = OUTPUT_DIR / f"{card_id}_{label}"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{date_str}.json"
    if out_file.exists():
        return None

    cmd = [FILL_SIM, "--mbo-file", str(mbo), "--predictions", str(pred),
           "--output", str(out_file), "--signal-threshold", str(sig),
           "--hold-ms", str(hold), "--quiet"]
    if tp: cmd += ["--take-profit-ticks", str(tp)]
    if sl: cmd += ["--stop-loss-ticks", str(sl)]
    if conv_bars > 0: cmd += ["--conviction-exit-bars", str(conv_bars), "--conviction-exit-mag", str(conv_mag)]
    if mae_t > 0: cmd += ["--mae-exit-ticks", str(mae_t), "--mae-exit-hold-sec", str(mae_s)]
    if prime: cmd += ["--prime-hours"]

    return {"cmd": cmd, "card": card_id, "date": date_str, "variant": label,
            "direction": direction, "out_file": str(out_file)}


def run_task(task):
    try:
        result = subprocess.run(task["cmd"], capture_output=True, timeout=600)
        if result.returncode != 0:
            return {"status": "FAIL", **task}
        return {"status": "OK", **task}
    except:
        return {"status": "ERROR", **task}


def aggregate_by_side(card_id, label, direction, dates):
    """Aggregate results filtering by trade side."""
    result_dir = OUTPUT_DIR / f"{card_id}_{label}"
    daily_pnls = []
    all_trades = []

    for f in sorted(result_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            trades = data.get("trades", [])
            # Filter by direction
            if direction == "LONG":
                trades = [t for t in trades if t.get("side") == "BUY" or t.get("side") == "Buy"]
            elif direction == "SHORT":
                trades = [t for t in trades if t.get("side") == "SELL" or t.get("side") == "Sell"]

            day_pnl = sum(t.get("pnl_dollars", 0) for t in trades)
            daily_pnls.append(day_pnl)
            all_trades.extend(trades)
        except:
            continue

    if not all_trades:
        return None

    pnl_arr = np.array(daily_pnls)
    sharpe = float(np.mean(pnl_arr) / np.std(pnl_arr) * np.sqrt(252)) if np.std(pnl_arr) > 0 else 0
    wins = [t for t in all_trades if t.get("pnl_dollars", 0) > 0]
    losses = [t for t in all_trades if t.get("pnl_dollars", 0) <= 0]

    return {
        "card": card_id, "variant": label, "direction": direction,
        "sharpe": round(sharpe, 2), "total_pnl": round(sum(daily_pnls), 2),
        "trades": len(all_trades), "trades_per_day": round(len(all_trades)/max(len(daily_pnls),1), 1),
        "win_rate": round(len(wins)/max(len(all_trades),1)*100, 1),
        "avg_win": round(np.mean([t["pnl_dollars"] for t in wins]), 2) if wins else 0,
        "avg_loss": round(np.mean([t["pnl_dollars"] for t in losses]), 2) if losses else 0,
        "pos_days": sum(1 for p in daily_pnls if p > 0),
        "neg_days": sum(1 for p in daily_pnls if p <= 0),
        "max_dd": round(min(np.minimum.accumulate(np.cumsum(pnl_arr)) - np.cumsum(pnl_arr)), 2) if len(pnl_arr) > 0 else 0,
        "n_days": len(daily_pnls),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = discover_dates()
    log.info(f"Found {len(dates)} OOT dates")

    tasks = []
    # Long cards
    for card_id, card_def in LONG_CARDS.items():
        for variant in LONG_VARIANTS:
            for ds in dates:
                t = build_task(card_id, card_def, ds, variant, "LONG")
                if t: tasks.append(t)
    # Short cards
    for card_id, card_def in SHORT_CARDS.items():
        for variant in SHORT_VARIANTS:
            for ds in dates:
                t = build_task(card_id, card_def, ds, variant, "SHORT")
                if t: tasks.append(t)

    log.info(f"Total tasks: {len(tasks)}")

    if tasks:
        t0 = time.time()
        ok = fail = 0
        with ProcessPoolExecutor(max_workers=WORKERS) as pool:
            futures = {pool.submit(run_task, t): t for t in tasks}
            for i, future in enumerate(as_completed(futures)):
                r = future.result()
                if r["status"] == "OK": ok += 1
                else: fail += 1
                if (i+1) % 100 == 0 or (i+1) == len(tasks):
                    elapsed = time.time() - t0
                    rate = (i+1) / elapsed * 60
                    eta = (len(tasks) - i - 1) / rate if rate > 0 else 0
                    log.info(f"Progress: {i+1}/{len(tasks)} ({ok} OK, {fail} FAIL) | {rate:.0f}/min | ETA: {eta:.1f}min")

    # Aggregate results
    log.info("\n" + "="*120)
    log.info("ASYMMETRIC CARD OPTIMIZATION RESULTS")
    log.info("="*120)

    all_results = []

    # Long results
    for card_id in LONG_CARDS:
        for variant in LONG_VARIANTS:
            label = variant[-1]
            agg = aggregate_by_side(card_id, label, "LONG", dates)
            if agg: all_results.append(agg)

    # Short results
    for card_id in SHORT_CARDS:
        for variant in SHORT_VARIANTS:
            label = variant[-1]
            agg = aggregate_by_side(card_id, label, "SHORT", dates)
            if agg: all_results.append(agg)

    # Sort by Sharpe
    all_results.sort(key=lambda x: -x["sharpe"])

    # Print
    print(f"\n{'Card':>8} {'Variant':>20} {'Dir':>5} | {'Sharpe':>7} {'PnL':>9} {'Trades':>6} {'WR%':>5} {'AvgW':>8} {'AvgL':>8} {'MaxDD':>8} | {'Pos':>3}/{' Neg':>3}")
    print("-"*120)
    for r in all_results[:50]:  # Top 50
        print(f"{r['card']:>8} {r['variant']:>20} {r['direction']:>5} | {r['sharpe']:>7.2f} {r['total_pnl']:>9.0f} {r['trades']:>6d} {r['win_rate']:>5.1f} {r['avg_win']:>8.2f} {r['avg_loss']:>8.2f} {r['max_dd']:>8.0f} | {r['pos_days']:>3}/{r['neg_days']:>3}")

    # Save
    summary_file = OUTPUT_DIR / "asymmetric_opt_summary.json"
    with open(summary_file, "w") as f:
        json.dump({"timestamp": datetime.now().isoformat(), "results": all_results}, f, indent=2)
    log.info(f"\nSaved: {summary_file}")


if __name__ == "__main__":
    main()
