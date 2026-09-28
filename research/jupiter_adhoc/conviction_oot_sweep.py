#!/usr/bin/env python3
"""
conviction_oot_sweep.py — Run full OOT sweep with conviction exit on C3 and C6.

Uses the new --conviction-exit-bars and --conviction-exit-mag flags in fill_sim_cli.
Runs baseline (no conviction) + conviction variants for direct comparison.

Cards:
  C3: raw_rawExit_conv0.15_ethr0.0_vol70, best conviction = 10s/mag0.0
  C6: raw_rawExit_conv0.15_ethr0.0_vol70, best conviction = 10s/mag0.0
  C1: book_predstdExit_conv1.5_vol50, best conviction = 5s/mag0.0 (marginal)

For each card, we run:
  1. Baseline (no conviction exit)
  2. conviction_exit_bars=50 (5s)
  3. conviction_exit_bars=100 (10s)  ← best for C3/C6
  4. conviction_exit_bars=150 (15s)

All other params match the card's validated config.
"""

import os
import sys
import json
import glob
import time
import subprocess
import logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime

# Config
FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/conviction_exit_sweep")

# OOT date range
OOT_START = "2025-12-01"
OOT_END = "2026-03-08"

WORKERS = 14  # Jupiter has 14 threads

# Card definitions
CARDS = {
    "c3": {
        "pred_suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "signal_threshold": 0.3,
        "tp_ticks": 10,
        "sl_ticks": None,
        "hold_ms": 3600000,  # 1 hour
        "desc": "Card3: raw rawExit conv0.15 vol70",
    },
    "c6": {
        "pred_suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "signal_threshold": 0.1,
        "tp_ticks": 20,
        "sl_ticks": 25,
        "hold_ms": 3600000,
        "desc": "Card6: raw rawExit conv0.15 vol70",
    },
    "c1": {
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "signal_threshold": 0.1,
        "tp_ticks": 8,
        "sl_ticks": None,
        "hold_ms": 7200000,  # 2 hours
        "desc": "Card1: book predstdExit conv1.5 vol50",
    },
}

# Conviction exit variants to test
CONVICTION_VARIANTS = [
    {"bars": 0, "mag": 0.0, "label": "baseline"},
    {"bars": 50, "mag": 0.0, "label": "conv5s_m0"},
    {"bars": 100, "mag": 0.0, "label": "conv10s_m0"},
    {"bars": 150, "mag": 0.0, "label": "conv15s_m0"},
    {"bars": 100, "mag": 0.5, "label": "conv10s_m05"},
    {"bars": 100, "mag": 1.0, "label": "conv10s_m10"},
]

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("conviction_sweep")


def discover_dates():
    """Find all OOT dates with MBO data."""
    dates = []
    for f in sorted(MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst")):
        # Extract date: glbx-mdp3-20251201.mbo.dbn.zst -> 2025-12-01
        name = f.stem.split(".")[0]  # glbx-mdp3-20251201
        date_raw = name.split("-")[-1]  # 20251201
        if len(date_raw) == 8:
            date_str = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
            if OOT_START <= date_str <= OOT_END:
                dates.append(date_str)
    return dates


def build_task(card_id, card_def, date_str, variant):
    """Build a single fill_sim_cli command."""
    date_compact = date_str.replace("-", "")
    mbo_file = MBO_DIR / f"glbx-mdp3-{date_compact}.mbo.dbn.zst"
    pred_file = PRED_DIR / f"{date_str}_{card_def['pred_suffix']}.npz"

    if not mbo_file.exists() or not pred_file.exists():
        return None

    out_dir = OUTPUT_DIR / f"{card_id}_{variant['label']}"
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

    # Conviction exit params
    if variant["bars"] > 0:
        cmd += ["--conviction-exit-bars", str(variant["bars"])]
        if variant["mag"] > 0:
            cmd += ["--conviction-exit-mag", str(variant["mag"])]

    return {
        "cmd": cmd,
        "card": card_id,
        "date": date_str,
        "variant": variant["label"],
        "out_file": str(out_file),
    }


def run_task(task):
    """Execute a single fill_sim_cli task."""
    try:
        result = subprocess.run(
            task["cmd"],
            capture_output=True,
            timeout=600,
        )
        if result.returncode != 0:
            return {"status": "FAIL", **task, "error": result.stderr.decode()[:200]}
        return {"status": "OK", **task}
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", **task}
    except Exception as e:
        return {"status": "ERROR", **task, "error": str(e)[:200]}


def aggregate_results(card_id, variant_label):
    """Aggregate per-date results for a card+variant into summary stats."""
    result_dir = OUTPUT_DIR / f"{card_id}_{variant_label}"
    if not result_dir.exists():
        return None

    all_trades = []
    daily_pnls = []
    n_days = 0

    for f in sorted(result_dir.glob("*.json")):
        try:
            data = json.loads(f.read_text())
            n_days += 1
            pnl = data.get("total_pnl_dollars", 0)
            daily_pnls.append(pnl)
            trades = data.get("trades", [])
            all_trades.extend(trades)
        except Exception:
            continue

    if n_days == 0:
        return None

    total_pnl = sum(daily_pnls)
    n_trades = len(all_trades)
    wins = [t for t in all_trades if t.get("pnl_dollars", 0) > 0]
    losses = [t for t in all_trades if t.get("pnl_dollars", 0) <= 0]

    import numpy as np
    daily_arr = np.array(daily_pnls)
    sharpe = float(np.mean(daily_arr) / np.std(daily_arr) * np.sqrt(252)) if np.std(daily_arr) > 0 else 0

    # Count exit reasons
    exit_reasons = {}
    for t in all_trades:
        reason = t.get("exit_reason", "Unknown")
        exit_reasons[reason] = exit_reasons.get(reason, 0) + 1

    return {
        "card": card_id,
        "variant": variant_label,
        "n_days": n_days,
        "n_trades": n_trades,
        "total_pnl": round(total_pnl, 2),
        "daily_sharpe": round(sharpe, 2),
        "win_rate": round(len(wins) / n_trades * 100, 1) if n_trades > 0 else 0,
        "avg_win": round(sum(t["pnl_dollars"] for t in wins) / len(wins), 2) if wins else 0,
        "avg_loss": round(sum(t["pnl_dollars"] for t in losses) / len(losses), 2) if losses else 0,
        "pos_days": sum(1 for p in daily_pnls if p > 0),
        "neg_days": sum(1 for p in daily_pnls if p <= 0),
        "exit_reasons": exit_reasons,
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dates = discover_dates()
    log.info(f"Found {len(dates)} OOT dates: {dates[0]} to {dates[-1]}")

    # Build all tasks
    tasks = []
    for card_id, card_def in CARDS.items():
        for variant in CONVICTION_VARIANTS:
            for date_str in dates:
                task = build_task(card_id, card_def, date_str, variant)
                if task is not None:
                    tasks.append(task)

    log.info(f"Total tasks: {len(tasks)} ({len(tasks)//len(dates)} configs x {len(dates)} dates)")

    if not tasks:
        log.info("All tasks already completed!")
    else:
        # Execute
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
                    log.warning(f"  FAIL: {result['card']} {result['date']} {result['variant']}: {result.get('error', '')[:100]}")

                if (i + 1) % 50 == 0 or (i + 1) == len(tasks):
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed * 60
                    eta_min = (len(tasks) - i - 1) / rate if rate > 0 else 0
                    log.info(f"Progress: {i+1}/{len(tasks)} ({ok} OK, {fail} FAIL) | {rate:.0f}/min | ETA: {eta_min:.1f}min")

    # Aggregate and print results
    log.info("\n" + "="*100)
    log.info("CONVICTION EXIT OOT SWEEP RESULTS")
    log.info("="*100)

    summary = []
    for card_id in CARDS:
        for variant in CONVICTION_VARIANTS:
            agg = aggregate_results(card_id, variant["label"])
            if agg:
                summary.append(agg)

    # Print comparison table
    header = f"{'Card':>5} {'Variant':>15} | {'Days':>4} {'Trades':>6} {'PnL':>10} {'Sharpe':>7} {'WR%':>5} {'AvgW':>8} {'AvgL':>8} | {'Pos':>3} {'Neg':>3} | Exit Reasons"
    print()
    print(header)
    print("-" * 130)

    for s in summary:
        exits = ", ".join(f"{k}:{v}" for k, v in sorted(s["exit_reasons"].items()))
        print(f"{s['card']:>5} {s['variant']:>15} | {s['n_days']:>4} {s['n_trades']:>6} {s['total_pnl']:>10.2f} {s['daily_sharpe']:>7.2f} {s['win_rate']:>5.1f} {s['avg_win']:>8.2f} {s['avg_loss']:>8.2f} | {s['pos_days']:>3} {s['neg_days']:>3} | {exits}")

    # Save summary JSON
    summary_file = OUTPUT_DIR / "conviction_sweep_summary.json"
    with open(summary_file, "w") as f:
        json.dump({
            "timestamp": datetime.now().isoformat(),
            "cards": list(CARDS.keys()),
            "variants": [v["label"] for v in CONVICTION_VARIANTS],
            "results": summary,
        }, f, indent=2)
    log.info(f"\nSaved: {summary_file}")


if __name__ == "__main__":
    main()
