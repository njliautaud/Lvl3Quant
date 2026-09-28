#!/usr/bin/env python3
"""
Side + Exit Strategy Sweep — Fill Sim Analysis
Tests buy-only vs sell-only vs both, combined with exit strategies
(trailing stop, signal flip, ratchet, afternoon window).

Based on trade-level decomposition findings:
- BUY side PF 1.065, SELL side PF 0.88
- BUY + afternoon (14-16 ET) had PF 1.44, 76% WR
- 38% of losers touched +4 ticks MFE — trailing stop opportunity
"""

import subprocess
import json
import os
import sys
import numpy as np
from pathlib import Path
from datetime import datetime

BASE = Path("/home/nick/Lvl3Quant")
FILL_SIM = BASE / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = BASE / "data" / "raw" / "mbo"
PRED_DIR = BASE / "output" / "optimizer_gate_validation" / "pred_npzs"
BUY_PRED_DIR = BASE / "output" / "optimizer_gate_validation" / "pred_npzs_buy_only"
SELL_PRED_DIR = BASE / "output" / "optimizer_gate_validation" / "pred_npzs_sell_only"
OUTPUT_DIR = BASE / "output" / "side_exit_sweep"

OOT_DATES = [
    "20260413", "20260414", "20260415", "20260416", "20260417",
    "20260419", "20260420", "20260421", "20260422", "20260423",
    "20260424", "20260426", "20260427", "20260428", "20260429",
]

# ── Step 1: Create buy-only and sell-only prediction NPZs ──────────────

def create_side_npzs():
    """Zero out opposite-side signals to create side-specific predictions."""
    print("=" * 60)
    print("STEP 1: Creating side-specific prediction NPZs")
    print("=" * 60)

    for date in OOT_DATES:
        src = PRED_DIR / f"{date}_unfiltered.npz"
        if not src.exists():
            print(f"  SKIP {date}: source not found")
            continue

        preds = np.load(str(src))["predictions"]

        # Buy-only: zero out negative (sell) signals
        buy_preds = preds.copy()
        buy_preds[buy_preds < 0] = 0.0
        buy_path = BUY_PRED_DIR / f"{date}_unfiltered.npz"
        np.savez(str(buy_path), predictions=buy_preds.astype(np.float32))

        # Sell-only: zero out positive (buy) signals
        sell_preds = preds.copy()
        sell_preds[sell_preds > 0] = 0.0
        sell_path = SELL_PRED_DIR / f"{date}_unfiltered.npz"
        np.savez(str(sell_path), predictions=sell_preds.astype(np.float32))

        n_buy = (buy_preds > 0.3).sum()
        n_sell = (sell_preds < -0.3).sum()
        print(f"  {date}: buy signals={n_buy}, sell signals={n_sell}")

    print("Done creating side NPZs.\n")


# ── Step 2: Define sweep configs ───────────────────────────────────────

# Base params shared by all configs
BASE_CFG = {
    "signal_threshold": 0.3,
    "take_profit_ticks": 8,
    "stop_loss_ticks": 16,
    "hold_ms": 1800000,   # 30 minutes
    "latency_ms": 10,
}

# (config_name, pred_dir_key, extra_cli_flags)
# pred_dir_key: "both", "buy", "sell"
SWEEP_CONFIGS = [
    # ── Both sides (original predictions) ──
    ("both_baseline",           "both", []),
    ("both_trailing4",          "both", ["--trailing-ticks", "4"]),
    ("both_signal_flip",        "both", ["--signal-flip-exit"]),
    ("both_ratchet",            "both", ["--ratchet-stop"]),
    ("both_afternoon",          "both", ["--time-window-start", "14:00", "--time-window-end", "16:00"]),

    # ── Buy-only ──
    ("buy_allday",              "buy",  []),
    ("buy_afternoon",           "buy",  ["--time-window-start", "14:00", "--time-window-end", "16:00"]),
    ("buy_trailing4",           "buy",  ["--trailing-ticks", "4"]),
    ("buy_signal_flip",         "buy",  ["--signal-flip-exit"]),
    ("buy_ratchet",             "buy",  ["--ratchet-stop"]),
    ("buy_afternoon_trailing4", "buy",  ["--time-window-start", "14:00", "--time-window-end", "16:00", "--trailing-ticks", "4"]),
    ("buy_afternoon_ratchet",   "buy",  ["--time-window-start", "14:00", "--time-window-end", "16:00", "--ratchet-stop"]),

    # ── Sell-only (comparison) ──
    ("sell_allday",             "sell", []),
]

PRED_DIRS = {
    "both": PRED_DIR,
    "buy":  BUY_PRED_DIR,
    "sell": SELL_PRED_DIR,
}


# ── Step 3: Run fill sim ──────────────────────────────────────────────

def run_fillsim(mbo_path, pred_npz_path, output_path, extra_flags):
    """Run the Rust fill_sim_cli and return parsed JSON results."""
    cmd = [
        str(FILL_SIM),
        "--mbo-file", str(mbo_path),
        "--predictions", str(pred_npz_path),
        "--output", str(output_path),
        "--signal-threshold", str(BASE_CFG["signal_threshold"]),
        "--take-profit-ticks", str(BASE_CFG["take_profit_ticks"]),
        "--stop-loss-ticks", str(BASE_CFG["stop_loss_ticks"]),
        "--hold-ms", str(BASE_CFG["hold_ms"]),
        "--latency-ms", str(BASE_CFG["latency_ms"]),
        "--quiet",
    ] + extra_flags

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        if result.returncode != 0:
            print(f"    ERROR: {result.stderr[:200]}")
            return None
        if os.path.exists(output_path):
            with open(output_path) as f:
                return json.load(f)
    except Exception as e:
        print(f"    EXCEPTION: {e}")
    return None


def compute_aggregate_metrics(results_list):
    """Compute aggregate metrics across all dates for a config."""
    if not results_list:
        return None

    all_trade_pnls = []
    daily_pnls = []
    total_pnl = 0
    total_trades = 0
    total_gross_win = 0
    total_gross_loss = 0

    for r in results_list:
        pnl = r.get("total_pnl_dollars", 0)
        total_pnl += pnl
        total_trades += r.get("total_trades", 0)
        daily_pnls.append(pnl)
        for t in r.get("trades", []):
            tp = t.get("pnl_dollars", 0)
            all_trade_pnls.append(tp)
            if tp > 0:
                total_gross_win += tp
            else:
                total_gross_loss += abs(tp)

    if not all_trade_pnls:
        return None

    arr = np.array(all_trade_pnls)
    wins = arr[arr > 0]
    losses = arr[arr <= 0]

    wr = len(wins) / len(arr) if len(arr) > 0 else 0
    pf = total_gross_win / total_gross_loss if total_gross_loss > 0 else float('inf')

    mean_pnl = arr.mean()
    std_pnl = arr.std() if len(arr) > 1 else 1.0
    sharpe = mean_pnl / std_pnl if std_pnl > 0 else 0

    avg_pnl_ticks = (mean_pnl / 12.50) if mean_pnl != 0 else 0

    # Daily Sharpe (annualized)
    if len(daily_pnls) > 1:
        d_arr = np.array(daily_pnls)
        daily_sharpe = (d_arr.mean() / d_arr.std()) * np.sqrt(252) if d_arr.std() > 0 else 0
    else:
        daily_sharpe = 0

    return {
        "trades": total_trades,
        "wr": wr,
        "pf": pf,
        "avg_pnl_ticks": avg_pnl_ticks,
        "total_pnl": total_pnl,
        "sharpe_per_trade": sharpe,
        "daily_sharpe_ann": daily_sharpe,
        "avg_win": wins.mean() if len(wins) > 0 else 0,
        "avg_loss": losses.mean() if len(losses) > 0 else 0,
        "n_days": len(results_list),
    }


def run_sweep():
    """Run the full sweep."""
    print("=" * 60)
    print("STEP 2: Running Fill Sim Sweep")
    print(f"  {len(SWEEP_CONFIGS)} configs x {len(OOT_DATES)} dates")
    print("=" * 60)

    all_metrics = {}
    start_time = datetime.now()

    for cfg_name, side_key, extra_flags in SWEEP_CONFIGS:
        pred_dir = PRED_DIRS[side_key]
        print(f"\n--- {cfg_name} ({side_key} preds) ---")

        day_results = []
        for date in OOT_DATES:
            pred_path = pred_dir / f"{date}_unfiltered.npz"
            mbo_path = MBO_DIR / f"glbx-mdp3-{date}.mbo.dbn.zst"

            if not pred_path.exists():
                print(f"  SKIP {date}: pred not found")
                continue
            if not mbo_path.exists():
                print(f"  SKIP {date}: MBO not found")
                continue

            out_path = OUTPUT_DIR / f"{cfg_name}_{date}.json"
            r = run_fillsim(mbo_path, pred_path, out_path, extra_flags)
            if r is not None:
                day_results.append(r)
                pnl = r.get("total_pnl_dollars", 0)
                trades = r.get("total_trades", 0)
                wr = r.get("win_rate", 0)
                print(f"  {date}: ${pnl:+.0f}  trades={trades}  WR={wr*100:.1f}%")

        metrics = compute_aggregate_metrics(day_results)
        if metrics:
            all_metrics[cfg_name] = metrics

    elapsed = (datetime.now() - start_time).total_seconds()
    print(f"\nSweep completed in {elapsed:.0f}s")
    return all_metrics


# ── Step 4: Summary table ──────────────────────────────────────────────

def print_summary(all_metrics):
    print("\n" + "=" * 110)
    print("SIDE + EXIT STRATEGY SWEEP — SUMMARY")
    print("=" * 110)
    header = f"{'Config':<28} {'Trades':>7} {'WR':>7} {'PF':>7} {'AvgTicks':>9} {'TotalPnL':>10} {'Sharpe/T':>9} {'DailySh':>8}"
    print(header)
    print("-" * 110)

    # Group by section
    sections = {
        "BOTH SIDES": [k for k in all_metrics if k.startswith("both_")],
        "BUY ONLY":   [k for k in all_metrics if k.startswith("buy_")],
        "SELL ONLY":  [k for k in all_metrics if k.startswith("sell_")],
    }

    for section_name, keys in sections.items():
        if not keys:
            continue
        print(f"\n  -- {section_name} --")
        for cfg_name in keys:
            m = all_metrics[cfg_name]
            print(f"  {cfg_name:<26} {m['trades']:>7} {m['wr']*100:>6.1f}% {m['pf']:>7.3f} {m['avg_pnl_ticks']:>+8.3f} {m['total_pnl']:>+10.0f} {m['sharpe_per_trade']:>9.4f} {m['daily_sharpe_ann']:>8.2f}")

    # Save summary JSON
    summary_path = OUTPUT_DIR / "sweep_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2, default=str)
    print(f"\nSummary saved to {summary_path}")


# ── Main ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    print(f"Side + Exit Strategy Sweep — {datetime.now().isoformat()}")
    print(f"Fill Sim: {FILL_SIM}")
    print(f"OOT dates: {len(OOT_DATES)}")
    print()

    # Step 1: Create side-specific NPZs
    create_side_npzs()

    # Step 2-3: Run sweep
    all_metrics = run_sweep()

    # Step 4: Print summary
    if all_metrics:
        print_summary(all_metrics)
    else:
        print("ERROR: No results produced!")
        sys.exit(1)

    print("\nDone.")
