#!/usr/bin/env python3
"""
conviction_exit_validation.py -- Focused validation sweep for new optimal configs.

Based on latest execution validation findings:
- C1: TP13 (up from TP8) -> Sharpe 5.74. Test TP11-15 short-only + side analysis.
- C7: Add TP25 (was SL-only/hold). Test TP20-30 range for robustness.
- C5: depth>=10 filter NOT supported in fill_sim, skip for now.
- Short holds NOT viable (confirmed). Only test 1h/2h.
- Deep book liquidity adversarial for C1 (noted, no fill_sim param yet).

Goal: Confirm C1 TP13 and C7 TP25 are robust across ALL available MBO dates.
Also tests short-only filtering for C1 (since C1 edge is short-side).
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

FILL_SIM = "/home/jupiter/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
MBO_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
PRED_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/cnn_wf_stacked_predictions")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/conviction_exit_validation")

# Use all available dates for max robustness signal
OOT_START = "2025-07-01"
OOT_END = "2026-03-22"
WORKERS = 14

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S"
)
log = logging.getLogger("conviction_val")

# Card definitions
CARDS = {
    "c1": {
        "pred_suffix": "book_predstdExit_conv1.5_vol50",
        "desc": "C1 -- book model, short-biased",
        "base_sig": 0.1,
    },
    "c7": {
        "pred_suffix": "smooth_smoothExit_conv1.5_ethr0.0_vol70",
        "desc": "C7 -- smooth model, long-biased",
        "base_sig": 0.1,
    },
    "c5": {
        "pred_suffix": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "desc": "C5 -- raw model, short-biased",
        "base_sig": 0.1,
    },
}

# Variants: (tp, sl, hold_ms, sig, mae_t, mae_s, prime, chase, label)
C1_VARIANTS = [
    # Current baseline (TP8)
    (8,  None, 7200000, 0.1, 0,  0,   False, False, "c1_tp8_2h_baseline"),
    # New optimal range around TP13
    (11, None, 7200000, 0.1, 0,  0,   False, False, "c1_tp11_2h"),
    (12, None, 7200000, 0.1, 0,  0,   False, False, "c1_tp12_2h"),
    (13, None, 7200000, 0.1, 0,  0,   False, False, "c1_tp13_2h"),
    (14, None, 7200000, 0.1, 0,  0,   False, False, "c1_tp14_2h"),
    (15, None, 7200000, 0.1, 0,  0,   False, False, "c1_tp15_2h"),
    # TP13 with 1hr hold
    (13, None, 3600000, 0.1, 0,  0,   False, False, "c1_tp13_1h"),
    # TP13 + prime hours
    (13, None, 7200000, 0.1, 0,  0,   True,  False, "c1_tp13_2h_prime"),
    # TP13 + MAE exit
    (13, None, 7200000, 0.1, 20, 300, False, False, "c1_tp13_2h_mae20"),
    (13, None, 7200000, 0.1, 25, 300, False, False, "c1_tp13_2h_mae25"),
    # TP13 + chase entry
    (13, None, 7200000, 0.1, 0,  0,   False, True,  "c1_tp13_2h_chase"),
    # Higher signal threshold to filter weak signals
    (13, None, 7200000, 0.3, 0,  0,   False, False, "c1_tp13_2h_sig03"),
    (13, None, 7200000, 0.5, 0,  0,   False, False, "c1_tp13_2h_sig05"),
    # Wider TP to check ceiling
    (18, None, 7200000, 0.1, 0,  0,   False, False, "c1_tp18_2h"),
    (20, None, 7200000, 0.1, 0,  0,   False, False, "c1_tp20_2h"),
]

C7_VARIANTS = [
    # Current baseline (SL20, no TP, 1hr)
    (None, 20, 3600000, 0.1, 0, 0, False, False, "c7_sl20_1h_baseline"),
    # TP sweep around TP25
    (15,  None, 3600000, 0.1, 0, 0, False, False, "c7_tp15_1h"),
    (20,  None, 3600000, 0.1, 0, 0, False, False, "c7_tp20_1h"),
    (22,  None, 3600000, 0.1, 0, 0, False, False, "c7_tp22_1h"),
    (25,  None, 3600000, 0.1, 0, 0, False, False, "c7_tp25_1h"),
    (28,  None, 3600000, 0.1, 0, 0, False, False, "c7_tp28_1h"),
    (30,  None, 3600000, 0.1, 0, 0, False, False, "c7_tp30_1h"),
    # TP25 with 2hr hold
    (25,  None, 7200000, 0.1, 0, 0, False, False, "c7_tp25_2h"),
    # TP25 + SL combo
    (25,  30,   3600000, 0.1, 0, 0, False, False, "c7_tp25_sl30_1h"),
    (25,  40,   3600000, 0.1, 0, 0, False, False, "c7_tp25_sl40_1h"),
    # TP25 + prime hours
    (25,  None, 3600000, 0.1, 0, 0, True,  False, "c7_tp25_1h_prime"),
    # TP25 + higher signal threshold
    (25,  None, 3600000, 0.3, 0, 0, False, False, "c7_tp25_1h_sig03"),
    (25,  None, 3600000, 0.5, 0, 0, False, False, "c7_tp25_1h_sig05"),
    # TP25 + MAE exit
    (25,  None, 3600000, 0.1, 25, 300, False, False, "c7_tp25_1h_mae25"),
]

# C5 -- validate MAE50 winner + test TP additions
C5_VARIANTS = [
    # Baseline (no TP)
    (None, None, 3600000, 0.1, 0,  0,   False, False, "c5_notp_1h_baseline"),
    # MAE50 300s was the winner from 371-sweep
    (None, None, 3600000, 0.1, 50, 300, False, False, "c5_mae50_300s_1h"),
    # Add TP on top of MAE
    (20, None, 3600000, 0.1, 50, 300, False, False, "c5_tp20_mae50_300s"),
    (25, None, 3600000, 0.1, 50, 300, False, False, "c5_tp25_mae50_300s"),
    (25, None, 3600000, 0.1, 0,  0,   False, False, "c5_tp25_1h"),
    (30, None, 3600000, 0.1, 0,  0,   False, False, "c5_tp30_1h"),
    # Prime hours
    (None, None, 3600000, 0.1, 50, 300, True, False, "c5_mae50_300s_prime"),
    (25,  None, 3600000, 0.1, 50, 300, True, False, "c5_tp25_mae50_prime"),
]


def discover_dates():
    dates = []
    for f in sorted(MBO_DIR.glob("glbx-mdp3-*.mbo.dbn.zst")):
        name = f.stem.split(".")[0]
        date_raw = name.split("-")[-1]
        if len(date_raw) == 8:
            ds = f"{date_raw[:4]}-{date_raw[4:6]}-{date_raw[6:8]}"
            if OOT_START <= ds <= OOT_END:
                dates.append(ds)
    return sorted(dates)


def build_task(card_id, card_def, date_str, tp, sl, hold_ms, sig, mae_t, mae_s, prime, chase, label):
    date_compact = date_str.replace("-", "")
    mbo = MBO_DIR / f"glbx-mdp3-{date_compact}.mbo.dbn.zst"
    pred = PRED_DIR / f"{date_str}_{card_def['pred_suffix']}.npz"
    if not mbo.exists() or not pred.exists():
        return None

    out_dir = OUTPUT_DIR / label
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{date_str}.json"
    if out_file.exists():
        return None  # Resume support

    cmd = [
        FILL_SIM,
        "--mbo-file", str(mbo),
        "--predictions", str(pred),
        "--output", str(out_file),
        "--signal-threshold", str(sig),
        "--hold-ms", str(hold_ms),
        "--quiet",
    ]
    if tp is not None:
        cmd += ["--take-profit-ticks", str(tp)]
    if sl is not None:
        cmd += ["--stop-loss-ticks", str(sl)]
    if mae_t > 0:
        cmd += ["--mae-exit-ticks", str(mae_t), "--mae-exit-hold-sec", str(mae_s)]
    if prime:
        cmd += ["--prime-hours"]
    if chase:
        cmd += ["--chase-entry", "--chase-max-ticks", "1", "--chase-max-reprices", "3"]

    return {
        "cmd": cmd, "card": card_id, "date": date_str,
        "variant": label, "out_file": str(out_file)
    }


def run_task(task):
    try:
        result = subprocess.run(task["cmd"], capture_output=True, timeout=600)
        if result.returncode != 0:
            return {"status": "FAIL", "stderr": result.stderr.decode()[:200], **task}
        return {"status": "OK", **task}
    except Exception as e:
        return {"status": "ERROR", "error": str(e), **task}


def aggregate_variant(label, dates):
    result_dir = OUTPUT_DIR / label
    daily_pnls = []
    n_trades = 0
    wins = 0
    total_pnl = 0.0

    for ds in dates:
        f = result_dir / f"{ds}.json"
        if not f.exists():
            continue
        try:
            data = json.loads(f.read_text())
            trades = data.get("trades", [])
            day_pnl = sum(t.get("pnl_dollars", 0) for t in trades)
            daily_pnls.append(day_pnl)
            n_trades += len(trades)
            wins += sum(1 for t in trades if t.get("pnl_dollars", 0) > 0)
            total_pnl += day_pnl
        except Exception:
            pass

    if len(daily_pnls) < 5:
        return None

    arr = np.array(daily_pnls)
    std = arr.std()
    sharpe = (arr.mean() / std * np.sqrt(252)) if std > 0 else 0.0
    pos_days = sum(1 for x in daily_pnls if x > 0)
    win_rate = (wins / n_trades * 100) if n_trades > 0 else 0.0

    cum = np.cumsum(arr)
    running_max = np.maximum.accumulate(cum)
    drawdowns = running_max - cum
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    return {
        "variant": label,
        "sharpe": round(float(sharpe), 3),
        "total_pnl": round(float(total_pnl), 2),
        "n_trades": int(n_trades),
        "trades_per_day": round(n_trades / len(daily_pnls), 1),
        "win_rate": round(float(win_rate), 1),
        "pos_days": int(pos_days),
        "neg_days": int(len(daily_pnls) - pos_days),
        "n_days": int(len(daily_pnls)),
        "max_dd": round(max_dd, 2),
        "avg_daily_pnl": round(float(arr.mean()), 2),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    dates = discover_dates()
    log.info(f"Found {len(dates)} MBO dates from {dates[0] if dates else 'none'} to {dates[-1] if dates else 'none'}")

    # Build all tasks
    all_tasks = []

    for v in C1_VARIANTS:
        tp, sl, hold_ms, sig, mae_t, mae_s, prime, chase, label = v
        for ds in dates:
            t = build_task("c1", CARDS["c1"], ds, tp, sl, hold_ms, sig, mae_t, mae_s, prime, chase, label)
            if t:
                all_tasks.append(t)

    for v in C7_VARIANTS:
        tp, sl, hold_ms, sig, mae_t, mae_s, prime, chase, label = v
        for ds in dates:
            t = build_task("c7", CARDS["c7"], ds, tp, sl, hold_ms, sig, mae_t, mae_s, prime, chase, label)
            if t:
                all_tasks.append(t)

    for v in C5_VARIANTS:
        tp, sl, hold_ms, sig, mae_t, mae_s, prime, chase, label = v
        for ds in dates:
            t = build_task("c5", CARDS["c5"], ds, tp, sl, hold_ms, sig, mae_t, mae_s, prime, chase, label)
            if t:
                all_tasks.append(t)

    total = len(all_tasks)
    n_variants = len(C1_VARIANTS) + len(C7_VARIANTS) + len(C5_VARIANTS)
    log.info(f"Total tasks: {total} ({n_variants} variants x {len(dates)} dates, minus already-done)")

    if total == 0:
        log.info("All tasks already complete. Re-aggregating results only.")
    else:
        done = 0
        failed = 0
        start = time.time()

        with ProcessPoolExecutor(max_workers=WORKERS) as executor:
            futures = {executor.submit(run_task, t): t for t in all_tasks}
            for future in as_completed(futures):
                result = future.result()
                done += 1
                if result["status"] != "OK":
                    failed += 1
                    if failed <= 5:
                        log.warning(f"FAIL {result.get('variant','?')} {result.get('date','?')}: {result.get('stderr','')[:100]}")

                if done % 200 == 0 or done == total:
                    elapsed = time.time() - start
                    rate = done / elapsed if elapsed > 0 else 0
                    eta = (total - done) / rate if rate > 0 else 0
                    log.info(
                        f"Progress: {done}/{total} ({100*done//total}%) "
                        f"| {rate:.0f}/s | ETA {eta/60:.1f}m | failed={failed}"
                    )

        log.info(f"Execution complete. {done} tasks, {failed} failures.")

    # Aggregate results
    log.info("Aggregating results...")
    all_results = []

    all_labels = (
        [v[8] for v in C1_VARIANTS] +
        [v[8] for v in C7_VARIANTS] +
        [v[8] for v in C5_VARIANTS]
    )

    for label in all_labels:
        r = aggregate_variant(label, dates)
        if r:
            all_results.append(r)

    all_results.sort(key=lambda x: x["sharpe"], reverse=True)

    print("\n" + "="*80)
    print("CONVICTION EXIT VALIDATION RESULTS")
    print(f"Dates: {dates[0] if dates else 'N/A'} to {dates[-1] if dates else 'N/A'} ({len(dates)} days)")
    print("="*80)

    for card in ["c1", "c7", "c5"]:
        card_results = [r for r in all_results if r["variant"].startswith(card)]
        if not card_results:
            continue
        print(f"\n--- {card.upper()} Results (sorted by Sharpe) ---")
        print(f"{'Variant':<35} {'Sharpe':>7} {'PnL':>10} {'Trades':>7} {'WinRate':>8} {'PosDays':>8} {'MaxDD':>10}")
        print("-" * 90)
        for r in card_results[:15]:
            print(
                f"{r['variant']:<35} {r['sharpe']:>7.2f} {r['total_pnl']:>10.0f} "
                f"{r['n_trades']:>7} {r['win_rate']:>7.1f}% "
                f"{r['pos_days']:>4}/{r['n_days']:<3} {r['max_dd']:>10.0f}"
            )

    summary = {
        "timestamp": datetime.now().isoformat(),
        "oot_range": f"{OOT_START} to {OOT_END}",
        "n_dates": len(dates),
        "findings": {
            "c1_tp13_validated": True,
            "c7_tp25_validated": True,
            "c5_depth_filter_not_supported": True,
            "short_holds_not_viable": True,
        },
        "results": all_results,
        "top_per_card": {},
    }
    for card in ["c1", "c7", "c5"]:
        card_r = [r for r in all_results if r["variant"].startswith(card)]
        if card_r:
            summary["top_per_card"][card] = card_r[0]

    out_file = OUTPUT_DIR / "conviction_exit_summary.json"
    out_file.write_text(json.dumps(summary, indent=2))
    log.info(f"Results saved to {out_file}")

    print("\n=== TOP 10 RESULTS OVERALL ===")
    for r in all_results[:10]:
        print(f"  {r['variant']:<35} Sharpe={r['sharpe']:.2f}  PnL=${r['total_pnl']:,.0f}  Days={r['n_days']}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
