#!/usr/bin/env python3
"""
conviction_refined_sweep.py — Refined conviction exit parameter sweep for C6 and C3.

Tests a finer grid of conviction_exit_bars x conviction_exit_mag combos:
  - bars: [60, 80, 100, 120, 150, 200]  (6s, 8s, 10s, 12s, 15s, 20s)
  - mag:  [0.7, 0.8, 1.0, 1.2, 1.5, 2.0]
  = 36 combos x 2 cards x ~84 OOT dates = ~6,048 tasks

Cards:
  C6: signal_threshold=0.1, tp=20, sl=25, hold=3600000
  C3: signal_threshold=0.3, tp=10, hold=3600000

Output: /home/jupiter/Lvl3Quant/data/processed/conviction_refined_sweep/
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

# Paths
FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/conviction_refined_sweep")

# OOT date range
OOT_START = "2025-12-01"
OOT_END = "2026-03-08"

WORKERS = 14

# Card definitions
CARDS = {
    "c6": {
        "pred_suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "signal_threshold": 0.1,
        "tp_ticks": 20,
        "sl_ticks": 25,
        "hold_ms": 3600000,
        "desc": "Card6: raw rawExit conv0.15 vol70 (TP20/SL25)",
    },
    "c3": {
        "pred_suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "signal_threshold": 0.3,
        "tp_ticks": 10,
        "sl_ticks": None,
        "hold_ms": 3600000,
        "desc": "Card3: raw rawExit conv0.15 vol70 (TP10)",
    },
}

# Refined grid
CONVICTION_BARS = [60, 80, 100, 120, 150, 200]
CONVICTION_MAGS = [0.7, 0.8, 1.0, 1.2, 1.5, 2.0]

# Build variant list: baseline + all combos
VARIANTS = [{"bars": 0, "mag": 0.0, "label": "baseline"}]
for bars in CONVICTION_BARS:
    for mag in CONVICTION_MAGS:
        secs = bars // 10
        mag_str = str(mag).replace(".", "")
        VARIANTS.append({
            "bars": bars,
            "mag": mag,
            "label": f"b{bars}_m{mag_str}",
        })

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("conviction_refined")


def discover_dates():
    """Find all OOT dates with MBO data in range."""
    dates = []
    for f in sorted(MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst")):
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

    daily_arr = np.array(daily_pnls)
    sharpe = float(np.mean(daily_arr) / np.std(daily_arr) * np.sqrt(252)) if np.std(daily_arr) > 0 else 0

    # Max drawdown
    cumulative = np.cumsum(daily_arr)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0

    # Profit factor
    gross_wins = sum(t.get("pnl_dollars", 0) for t in wins)
    gross_losses = abs(sum(t.get("pnl_dollars", 0) for t in losses))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Exit reasons
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
        "avg_win": round(gross_wins / len(wins), 2) if wins else 0,
        "avg_loss": round(gross_losses / len(losses) * -1, 2) if losses else 0,
        "profit_factor": round(profit_factor, 2),
        "max_drawdown": round(max_dd, 2),
        "pos_days": sum(1 for p in daily_pnls if p > 0),
        "neg_days": sum(1 for p in daily_pnls if p <= 0),
        "exit_reasons": exit_reasons,
    }


def print_comparison_table(summary):
    """Print sorted comparison table by card then Sharpe."""
    print("\n" + "=" * 150)
    print("CONVICTION REFINED SWEEP RESULTS — SORTED BY SHARPE (per card)")
    print("=" * 150)

    for card_id in CARDS:
        card_results = [s for s in summary if s["card"] == card_id]
        card_results.sort(key=lambda x: x["daily_sharpe"], reverse=True)

        print(f"\n{'─' * 150}")
        print(f"  {CARDS[card_id]['desc']}")
        print(f"{'─' * 150}")

        header = (
            f"{'Rank':>4} {'Variant':>12} | {'Days':>4} {'Trades':>6} "
            f"{'PnL':>10} {'Sharpe':>7} {'PF':>6} {'WR%':>5} "
            f"{'AvgW':>8} {'AvgL':>8} {'MaxDD':>9} | {'Pos':>3}/{' Neg':>3} | Exit Reasons"
        )
        print(header)
        print("-" * 150)

        for rank, s in enumerate(card_results, 1):
            exits = ", ".join(f"{k}:{v}" for k, v in sorted(s["exit_reasons"].items()))
            pf_str = f"{s['profit_factor']:>6.2f}" if s['profit_factor'] < 999 else "   inf"
            marker = " ***" if rank <= 3 else ""
            print(
                f"{rank:>4} {s['variant']:>12} | {s['n_days']:>4} {s['n_trades']:>6} "
                f"{s['total_pnl']:>10.2f} {s['daily_sharpe']:>7.2f} {pf_str} {s['win_rate']:>5.1f} "
                f"{s['avg_win']:>8.2f} {s['avg_loss']:>8.2f} {s['max_drawdown']:>9.2f} | "
                f"{s['pos_days']:>3}/{s['neg_days']:>3} | {exits}{marker}"
            )

    # Cross-card best configs
    print(f"\n{'=' * 150}")
    print("TOP 5 OVERALL (all cards, sorted by Sharpe)")
    print("=" * 150)
    all_sorted = sorted(summary, key=lambda x: x["daily_sharpe"], reverse=True)[:5]
    for rank, s in enumerate(all_sorted, 1):
        print(
            f"  #{rank}: {s['card'].upper()} {s['variant']} — "
            f"Sharpe={s['daily_sharpe']:.2f}, PnL=${s['total_pnl']:.2f}, "
            f"Trades={s['n_trades']}, WR={s['win_rate']:.1f}%, PF={s['profit_factor']:.2f}"
        )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    dates = discover_dates()
    log.info(f"Found {len(dates)} OOT dates: {dates[0]} to {dates[-1]}")

    # Build all tasks
    tasks = []
    skipped = 0
    for card_id, card_def in CARDS.items():
        for variant in VARIANTS:
            for date_str in dates:
                task = build_task(card_id, card_def, date_str, variant)
                if task is not None:
                    tasks.append(task)
                else:
                    skipped += 1

    total_possible = len(CARDS) * len(VARIANTS) * len(dates)
    log.info(f"Total tasks: {len(tasks)} new + {skipped} skipped/missing = {total_possible} possible")
    log.info(f"Grid: {len(VARIANTS)} variants x {len(CARDS)} cards x {len(dates)} dates")
    log.info(f"Workers: {WORKERS}")

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
                    log.warning(f"  FAIL: {result['card']} {result['date']} {result['variant']}: {result.get('error', '')[:100]}")

                if (i + 1) % 100 == 0 or (i + 1) == len(tasks):
                    elapsed = time.time() - t0
                    rate = (i + 1) / elapsed * 60
                    eta_min = (len(tasks) - i - 1) / rate if rate > 0 else 0
                    log.info(
                        f"Progress: {i+1}/{len(tasks)} ({ok} OK, {fail} FAIL) | "
                        f"{rate:.0f}/min | ETA: {eta_min:.1f}min | "
                        f"Elapsed: {elapsed/60:.1f}min"
                    )

        log.info(f"\nExecution complete: {ok} OK, {fail} FAIL in {(time.time()-t0)/60:.1f}min")
    else:
        log.info("All tasks already completed!")

    # Aggregate and display
    summary = []
    for card_id in CARDS:
        for variant in VARIANTS:
            agg = aggregate_results(card_id, variant["label"])
            if agg:
                summary.append(agg)

    if summary:
        print_comparison_table(summary)

        # Save summary JSON
        summary_file = OUTPUT_DIR / "conviction_refined_summary.json"
        with open(summary_file, "w") as f:
            json.dump({
                "timestamp": datetime.now().isoformat(),
                "cards": {k: v["desc"] for k, v in CARDS.items()},
                "grid": {
                    "conviction_exit_bars": CONVICTION_BARS,
                    "conviction_exit_mag": CONVICTION_MAGS,
                    "total_variants": len(VARIANTS),
                },
                "results": summary,
            }, f, indent=2)
        log.info(f"\nSaved summary: {summary_file}")
    else:
        log.warning("No results to aggregate!")


if __name__ == "__main__":
    main()
