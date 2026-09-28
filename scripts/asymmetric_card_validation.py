#!/usr/bin/env python3
"""
asymmetric_card_validation.py — OOT validation for asymmetric card variants.

Cards 8L, 9S, 10L (long/short-only filtered), 11 (C6+MAE), 12 (C7+MAE).
Runs fill_sim normally, then filters trades by side in post-processing.

Run on Jupiter with 14 workers:
  PYTHONIOENCODING=utf-8 nohup python3 asymmetric_card_validation.py 2>&1 | tee asymmetric_validation.log &

Output: /home/jupiter/Lvl3Quant/data/processed/asymmetric_card_validation/
"""
import os
import sys
import json
import time
import subprocess
import statistics
import numpy as np
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict

sys.stdout.reconfigure(encoding='utf-8')

# ── Paths ─────────────────────────────────────────────────────────────────────
LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
BINARY = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "data" / "processed" / "cnn_wf_stacked_predictions"
OUT_BASE = LVL3_ROOT / "data" / "processed" / "asymmetric_card_validation"

OOT_START = "2025-12-01"
OOT_END = "2026-03-08"
WORKERS = 14
TICK_VALUE = 12.50
COMMISSION_RT = 4.70

# ── Card Definitions ──────────────────────────────────────────────────────────
# Each card has: pred_suffix, signal_threshold, tp, sl, hold_ms, conviction_bars,
#                conviction_mag, mae_exit_ticks, mae_exit_hold_sec, side_filter
# side_filter: None = all trades, "BUY" = longs only, "SELL" = shorts only

CARDS = {
    "card8L": {
        "label": "C8L (C3 Long-Only)",
        "base_card": "C3",
        "pred_suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "signal_threshold": 0.3,
        "tp": 10,
        "sl": None,
        "hold_ms": 3600000,
        "conviction_bars": 100,
        "conviction_mag": 0.8,
        "mae_exit_ticks": None,
        "mae_exit_hold_sec": None,
        "side_filter": "BUY",
    },
    "card9S": {
        "label": "C9S (C5 Short-Only)",
        "base_card": "C5",
        "pred_suffix": "raw_rawExit_conv0.05_ethr0.5_vol0",
        "signal_threshold": 0.1,
        "tp": None,
        "sl": None,
        "hold_ms": 3600000,
        "conviction_bars": None,
        "conviction_mag": None,
        "mae_exit_ticks": None,
        "mae_exit_hold_sec": None,
        "side_filter": "SELL",
    },
    "card10L": {
        "label": "C10L (C4 Long-Only)",
        "base_card": "C4",
        "pred_suffix": "book_predstdExit_conv2.0_vol70",
        "signal_threshold": 0.1,
        "tp": 20,
        "sl": None,
        "hold_ms": 7200000,
        "conviction_bars": None,
        "conviction_mag": None,
        "mae_exit_ticks": None,
        "mae_exit_hold_sec": None,
        "side_filter": "BUY",
    },
    "card11": {
        "label": "C11 (C6 + MAE 25t)",
        "base_card": "C6",
        "pred_suffix": "raw_rawExit_conv0.15_ethr0.0_vol70",
        "signal_threshold": 0.1,
        "tp": 20,
        "sl": 25,
        "hold_ms": 3600000,
        "conviction_bars": 60,
        "conviction_mag": 1.5,
        "mae_exit_ticks": 25,
        "mae_exit_hold_sec": 60,
        "side_filter": None,
    },
    "card12": {
        "label": "C12 (C7 + MAE 20t)",
        "base_card": "C7",
        "pred_suffix": "smooth_smoothExit_conv1.5_ethr0.0_vol70",
        "signal_threshold": 0.1,
        "tp": None,
        "sl": 20,
        "hold_ms": 3600000,
        "conviction_bars": None,
        "conviction_mag": None,
        "mae_exit_ticks": 20,
        "mae_exit_hold_sec": 60,
        "side_filter": None,
    },
}


# ── Helpers ───────────────────────────────────────────────────────────────────

def get_oot_dates():
    """Find all OOT dates that have prediction files."""
    dates = set()
    for f in PRED_DIR.iterdir():
        if f.suffix == ".npz" and len(f.name) >= 10:
            date_str = f.name[:10]
            if OOT_START <= date_str <= OOT_END:
                dates.add(date_str)
    return sorted(dates)


def find_pred_file(date_str, pred_suffix):
    """Find prediction file matching date and suffix pattern."""
    target = PRED_DIR / f"{date_str}_{pred_suffix}.npz"
    if target.exists():
        return target
    # Fallback: glob for partial match
    for f in PRED_DIR.glob(f"{date_str}_{pred_suffix}*"):
        if f.suffix == ".npz":
            return f
    return None


def get_mbo_path(date_str):
    nodash = date_str.replace("-", "")
    return MBO_DIR / f"glbx-mdp3-{nodash}.mbo.dbn.zst"


def run_one(card_id, card_def, date_str):
    """Run fill_sim for one card+date. Returns parsed JSON or None."""
    pred_file = find_pred_file(date_str, card_def["pred_suffix"])
    if not pred_file:
        return None
    mbo_path = get_mbo_path(date_str)
    if not mbo_path.exists():
        return None

    out_dir = OUT_BASE / card_id
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{date_str}.json"

    # Check cache
    if out_file.exists():
        try:
            data = json.loads(out_file.read_text())
            return data
        except Exception:
            pass

    cmd = [
        str(BINARY),
        "--mbo-file", str(mbo_path),
        "--predictions", str(pred_file),
        "--output", str(out_file),
        "--signal-threshold", str(card_def["signal_threshold"]),
        "--hold-ms", str(card_def["hold_ms"]),
        "--quiet",
    ]
    if card_def["tp"] is not None:
        cmd += ["--take-profit-ticks", str(card_def["tp"])]
    if card_def["sl"] is not None:
        cmd += ["--stop-loss-ticks", str(card_def["sl"])]
    if card_def["conviction_bars"] is not None:
        cmd += ["--conviction-exit-bars", str(card_def["conviction_bars"])]
    if card_def["conviction_mag"] is not None:
        cmd += ["--conviction-exit-mag", str(card_def["conviction_mag"])]
    if card_def["mae_exit_ticks"] is not None:
        cmd += ["--mae-exit-ticks", str(card_def["mae_exit_ticks"])]
        cmd += ["--mae-exit-hold-sec", str(card_def.get("mae_exit_hold_sec", 300))]

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        if out_file.exists():
            return json.loads(out_file.read_text())
        else:
            if result.returncode != 0:
                print(f"  ERR {card_id} {date_str}: rc={result.returncode} {result.stderr[:200]}", flush=True)
    except Exception as e:
        print(f"  ERR {card_id} {date_str}: {e}", flush=True)
    return None


def percentile_stats(values):
    """Compute percentile stats for a list of values."""
    if not values:
        return {"avg": 0, "p50": 0, "p75": 0, "p95": 0, "extreme": 0}
    arr = np.array(values)
    return {
        "avg": round(float(np.mean(arr)), 2),
        "p50": round(float(np.percentile(arr, 50)), 2),
        "p75": round(float(np.percentile(arr, 75)), 2),
        "p95": round(float(np.percentile(arr, 95)), 2),
        "extreme": round(float(np.max(np.abs(arr))), 2),
    }


def compute_metrics(card_id, card_def, daily_results):
    """Compute full metrics for a card, with optional side filtering."""
    side_filter = card_def.get("side_filter")

    all_trades = []
    daily_pnls = []
    daily_trade_counts = []

    for day_data in daily_results:
        trades = day_data.get("trades", [])
        if side_filter:
            trades = [t for t in trades if t.get("side") == side_filter]

        day_pnl = sum(t.get("pnl_dollars", 0) for t in trades)
        daily_pnls.append(day_pnl)
        daily_trade_counts.append(len(trades))
        all_trades.extend(trades)

    n_days = len(daily_results)
    n_trades = len(all_trades)
    total_pnl = sum(daily_pnls)

    if n_trades == 0:
        return {
            "card": card_id,
            "label": card_def["label"],
            "side_filter": side_filter or "ALL",
            "n_days": n_days,
            "n_trades": 0,
            "sharpe": 0,
            "total_pnl": 0,
            "error": "No trades found",
        }

    # Basic metrics
    winners = [t for t in all_trades if t.get("pnl_dollars", 0) > 0]
    losers = [t for t in all_trades if t.get("pnl_dollars", 0) <= 0]
    win_rate = len(winners) / n_trades * 100 if n_trades > 0 else 0

    gross_wins = sum(t.get("pnl_dollars", 0) for t in winners)
    gross_losses = abs(sum(t.get("pnl_dollars", 0) for t in losers))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')
    avg_win = gross_wins / len(winners) if winners else 0
    avg_loss = -gross_losses / len(losers) if losers else 0
    wl_ratio = abs(avg_win / avg_loss) if avg_loss != 0 else float('inf')

    # Sharpe
    daily_arr = np.array(daily_pnls)
    if np.std(daily_arr) > 0:
        sharpe = float(np.mean(daily_arr) / np.std(daily_arr) * np.sqrt(252))
    else:
        sharpe = 0.0

    # Max drawdown
    cumulative = np.cumsum(daily_arr)
    running_max = np.maximum.accumulate(cumulative)
    max_dd = float(np.max(running_max - cumulative)) if len(cumulative) > 0 else 0

    # Positive / negative days
    pos_days = sum(1 for p in daily_pnls if p > 0)
    neg_days = sum(1 for p in daily_pnls if p <= 0)

    # MAE stats
    mae_vals = [t.get("mae_ticks", 0) for t in all_trades]
    mae_stats = percentile_stats(mae_vals)

    # MFE stats
    mfe_vals = [t.get("mfe_ticks", 0) for t in all_trades]
    mfe_stats = percentile_stats(mfe_vals)

    # Hold time: winners vs losers
    def avg_hold_sec(trades_list):
        holds = []
        for t in trades_list:
            h = t.get("hold_duration_ns", 0)
            if h <= 0:
                h = t.get("exit_time_ns", 0) - t.get("fill_time_ns", 0)
            if h > 0:
                holds.append(h / 1e9)
        return round(np.mean(holds), 1) if holds else 0

    hold_winners = avg_hold_sec(winners)
    hold_losers = avg_hold_sec(losers)

    # Exit reason breakdown
    exit_reasons = defaultdict(int)
    for t in all_trades:
        reason = t.get("exit_reason", "Unknown")
        exit_reasons[reason] += 1

    return {
        "card": card_id,
        "label": card_def["label"],
        "base_card": card_def["base_card"],
        "side_filter": side_filter or "ALL",
        "n_days": n_days,
        "n_trades": n_trades,
        "sharpe": round(sharpe, 2),
        "total_pnl": round(total_pnl, 2),
        "win_rate": round(win_rate, 1),
        "profit_factor": round(profit_factor, 2),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "wl_ratio": round(wl_ratio, 2),
        "max_drawdown": round(max_dd, 2),
        "pos_days": pos_days,
        "neg_days": neg_days,
        "trades_per_day": round(n_trades / max(n_days, 1), 1),
        "avg_daily_pnl": round(float(np.mean(daily_arr)), 2),
        "mae": mae_stats,
        "mfe": mfe_stats,
        "hold_time_winners_sec": hold_winners,
        "hold_time_losers_sec": hold_losers,
        "exit_reasons": dict(exit_reasons),
    }


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    print("=" * 70, flush=True)
    print("ASYMMETRIC CARD VALIDATION — OOT Sweep", flush=True)
    print("=" * 70, flush=True)

    if not BINARY.exists():
        print(f"ERROR: fill_sim_cli not found at {BINARY}", flush=True)
        sys.exit(1)

    OUT_BASE.mkdir(parents=True, exist_ok=True)

    dates = get_oot_dates()
    if not dates:
        print("ERROR: No OOT prediction dates found!", flush=True)
        sys.exit(1)

    total_jobs = len(CARDS) * len(dates)
    print(f"OOT dates: {len(dates)} ({dates[0]} to {dates[-1]})", flush=True)
    print(f"Cards: {len(CARDS)} x {len(dates)} dates = {total_jobs} jobs", flush=True)
    print(f"Workers: {WORKERS}", flush=True)
    print(flush=True)

    for cid, cdef in CARDS.items():
        print(f"  {cid}: {cdef['label']} — pred={cdef['pred_suffix']}, sig={cdef['signal_threshold']}, "
              f"tp={cdef['tp']}, sl={cdef['sl']}, hold={cdef['hold_ms']}ms, "
              f"side={cdef.get('side_filter', 'ALL')}", flush=True)
    print(flush=True)

    # ── Phase 1: Run fill_sim for all card×date combos ────────────────────────
    card_results = {cid: [] for cid in CARDS}
    done = 0
    t0 = time.time()

    with ThreadPoolExecutor(max_workers=WORKERS) as pool:
        futures = {}
        for card_id, card_def in CARDS.items():
            for date_str in dates:
                fut = pool.submit(run_one, card_id, card_def, date_str)
                futures[fut] = (card_id, date_str)

        for fut in as_completed(futures):
            done += 1
            card_id, date_str = futures[fut]
            result = fut.result()
            if result:
                card_results[card_id].append(result)

            if done % 50 == 0 or done == total_jobs:
                elapsed = time.time() - t0
                rate = done / elapsed * 60 if elapsed > 0 else 0
                eta = (total_jobs - done) / (done / elapsed) if done > 0 else 0
                print(f"  [{done}/{total_jobs}] {rate:.0f}/min  ETA {eta:.0f}s", flush=True)

    elapsed_total = time.time() - t0
    print(f"\nAll jobs complete in {elapsed_total:.0f}s", flush=True)

    # ── Phase 2: Compute metrics with side filtering ──────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("RESULTS — ASYMMETRIC CARD VALIDATION", flush=True)
    print("=" * 70, flush=True)

    all_summaries = []
    for card_id, card_def in CARDS.items():
        days = card_results[card_id]
        if not days:
            print(f"\n{card_id}: NO DATA", flush=True)
            continue

        metrics = compute_metrics(card_id, card_def, days)
        all_summaries.append(metrics)

        sf = metrics["side_filter"]
        print(f"\n{'─'*60}", flush=True)
        print(f"  {metrics['label']}  (filter: {sf})", flush=True)
        print(f"{'─'*60}", flush=True)
        print(f"  Sharpe:         {metrics['sharpe']:>8.2f}", flush=True)
        print(f"  Total PnL:      ${metrics['total_pnl']:>12,.2f}", flush=True)
        print(f"  Trades:         {metrics['n_trades']:>8d}   ({metrics['trades_per_day']:.1f}/day)", flush=True)
        print(f"  Win Rate:       {metrics['win_rate']:>8.1f}%", flush=True)
        print(f"  Profit Factor:  {metrics['profit_factor']:>8.2f}", flush=True)
        print(f"  Avg Win:        ${metrics['avg_win']:>10.2f}", flush=True)
        print(f"  Avg Loss:       ${metrics['avg_loss']:>10.2f}", flush=True)
        print(f"  W/L Ratio:      {metrics['wl_ratio']:>8.2f}", flush=True)
        print(f"  Max Drawdown:   ${metrics['max_drawdown']:>10.2f}", flush=True)
        print(f"  Days: +{metrics['pos_days']} / -{metrics['neg_days']}  ({metrics['n_days']} total)", flush=True)
        print(f"  Avg Daily PnL:  ${metrics['avg_daily_pnl']:>10.2f}", flush=True)
        print(flush=True)

        # MAE stats
        m = metrics["mae"]
        print(f"  MAE (ticks):    avg={m['avg']:.1f}  P50={m['p50']:.1f}  P75={m['p75']:.1f}  P95={m['p95']:.1f}  worst={m['extreme']:.1f}", flush=True)

        # MFE stats
        m = metrics["mfe"]
        print(f"  MFE (ticks):    avg={m['avg']:.1f}  P50={m['p50']:.1f}  P75={m['p75']:.1f}  P95={m['p95']:.1f}  best={m['extreme']:.1f}", flush=True)

        print(f"  Hold Winners:   {metrics['hold_time_winners_sec']:.0f}s   Losers: {metrics['hold_time_losers_sec']:.0f}s", flush=True)

        # Exit reasons
        print(f"  Exit Reasons:", flush=True)
        for reason, count in sorted(metrics["exit_reasons"].items(), key=lambda x: -x[1]):
            pct = count / metrics["n_trades"] * 100
            print(f"    {reason:25s} {count:>5d}  ({pct:5.1f}%)", flush=True)

    # ── Phase 3: Comparison table ─────────────────────────────────────────────
    print("\n" + "=" * 70, flush=True)
    print("COMPARISON TABLE: New Asymmetric Cards", flush=True)
    print("=" * 70, flush=True)

    header = f"{'Card':<22s} {'Filter':<6s} {'Sharpe':>7s} {'PnL':>12s} {'Trades':>7s} {'WR%':>6s} {'PF':>6s} {'AvgWin':>9s} {'AvgLoss':>9s} {'MaxDD':>10s} {'+/-Days':>8s}"
    print(header, flush=True)
    print("-" * len(header), flush=True)

    for s in all_summaries:
        pf_str = f"{s['profit_factor']:.2f}" if s['profit_factor'] < 100 else "INF"
        print(f"{s['label']:<22s} {s['side_filter']:<6s} {s['sharpe']:>7.2f} ${s['total_pnl']:>11,.0f} "
              f"{s['n_trades']:>7d} {s['win_rate']:>5.1f}% {pf_str:>6s} "
              f"${s['avg_win']:>8.2f} ${s['avg_loss']:>8.2f} ${s['max_drawdown']:>9,.0f} "
              f"{s['pos_days']:>3d}/{s['neg_days']:<3d}", flush=True)

    # ── Save summary ──────────────────────────────────────────────────────────
    summary_file = OUT_BASE / "asymmetric_summary.json"
    summary_file.write_text(json.dumps(all_summaries, indent=2, default=str))
    print(f"\nSummary saved to: {summary_file}", flush=True)

    # Also save raw per-day results for each card
    for card_id in CARDS:
        raw_file = OUT_BASE / f"{card_id}_daily_raw.json"
        raw_file.write_text(json.dumps(card_results[card_id], indent=2, default=str))

    print(f"\nAll results in: {OUT_BASE}", flush=True)
    print("DONE.", flush=True)


if __name__ == "__main__":
    main()
