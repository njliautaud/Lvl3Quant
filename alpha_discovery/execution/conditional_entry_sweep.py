#!/usr/bin/env python3
"""
Conditional Entry Sweep — Smart Execution Configurations
=========================================================

Instead of brute-force configs (all negative), test SMARTER execution:
1. Queue position filtering — only trade when near top of book (less adverse selection)
2. Fast cancel — short max-wait, cancel if not filled quickly
3. Ratchet trailing stop — lock in profit as MFE grows
4. Time windows — trade only during periods with best signal-to-noise
5. Conviction exits — wait for sustained reversal, not instant flip
6. Latency modeling — realistic 10-50ms order latency
7. Combined smart configs — stack multiple favorable conditions

The hypothesis: FIFO adverse selection is ~5-10 ticks because we're deep in queue.
If we only trade when near TOP of queue (queue_pos < 20-50), adverse selection drops.

Author: Claude
Date: 2026-05-08
"""

import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

LOG = logging.getLogger("COND_ENTRY")
LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))

FILL_SIM_BIN = LVL3_ROOT / "rust_cache_builder" / "target" / "release" / "fill_sim_cli"
MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
PRED_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_all_oot"

COMMISSION_TICKS = 0.376


# ============================================================
# CONFIGURATIONS — Focus on reducing adverse selection
# ============================================================

CONFIGS = {
    # --- Track 1: Queue Position Filtering ---
    # Hypothesis: signals where we're near TOP of queue have less adverse selection
    "qpos_top10_tp4_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 4, "stop_loss_ticks": 4, "hold_ms": 15000,
        "max_queue_pos": 10,  # Only trade when in first 10 contracts
    },
    "qpos_top20_tp4_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 4, "stop_loss_ticks": 4, "hold_ms": 15000,
        "max_queue_pos": 20,
    },
    "qpos_top50_tp4_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 4, "stop_loss_ticks": 4, "hold_ms": 15000,
        "max_queue_pos": 50,
    },
    "qpos_top10_tp3_sl3": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 3, "stop_loss_ticks": 3, "hold_ms": 10000,
        "max_queue_pos": 10,
    },

    # --- Track 2: Fast Cancel (reduce adverse selection exposure) ---
    # Hypothesis: if signal is right, fill should come fast. Slow fills = adverse.
    "fastcancel_500ms_tp4_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 4, "stop_loss_ticks": 4, "hold_ms": 15000,
        "max_wait_bars": 5,  # 500ms max wait for fill
    },
    "fastcancel_1s_tp4_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 4, "stop_loss_ticks": 4, "hold_ms": 15000,
        "max_wait_bars": 10,  # 1s max wait
    },
    "fastcancel_2s_tp4_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 4, "stop_loss_ticks": 4, "hold_ms": 15000,
        "max_wait_bars": 20,  # 2s max wait
    },

    # --- Track 3: Ratchet Trailing Stop ---
    # Hypothesis: fixed TP/SL leaves money on table. Ratchet locks in profit dynamically.
    "ratchet_basic_sl4": {
        "signal_threshold": 0.5,
        "stop_loss_ticks": 4, "hold_ms": 30000,
        "ratchet_stop": True,  # Enable ratcheting stop
    },
    "ratchet_trail2_sl4": {
        "signal_threshold": 0.5,
        "trailing_ticks": 2, "stop_loss_ticks": 4, "hold_ms": 30000,
        "ratchet_stop": True,
    },

    # --- Track 4: Conviction Exits (don't exit on noise) ---
    # Hypothesis: instant signal-flip-exit exits on noise. Wait for sustained reversal.
    "conviction_10bars_tp6_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 6, "stop_loss_ticks": 4, "hold_ms": 30000,
        "conviction_exit_bars": 10, "conviction_exit_mag": 0.5,
    },
    "conviction_20bars_tp6_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 6, "stop_loss_ticks": 4, "hold_ms": 30000,
        "conviction_exit_bars": 20, "conviction_exit_mag": 0.5,
    },

    # --- Track 5: Time Windows ---
    "opening_range_tp4_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 4, "stop_loss_ticks": 4, "hold_ms": 15000,
        "time_window_start": "09:30", "time_window_end": "10:30",
    },
    "midday_tp4_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 4, "stop_loss_ticks": 4, "hold_ms": 15000,
        "time_window_start": "11:00", "time_window_end": "14:00",
    },

    # --- Track 6: Latency Modeling (realistic) ---
    "lat10ms_tp4_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 4, "stop_loss_ticks": 4, "hold_ms": 15000,
        "latency_ms": 10,  # Co-located latency
    },
    "lat50ms_tp4_sl4": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 4, "stop_loss_ticks": 4, "hold_ms": 15000,
        "latency_ms": 50,  # Retail latency
    },

    # --- Track 7: COMBINED SMART CONFIGS ---
    # Stack the best conditions: top of queue + fast cancel + prime hours + high conf
    "smart_qpos20_fast1s_prime_tp4_sl4": {
        "signal_threshold": 1.0,  # High confidence only
        "take_profit_ticks": 4, "stop_loss_ticks": 4, "hold_ms": 15000,
        "max_queue_pos": 20,
        "max_wait_bars": 10,
        "prime_hours": True,
    },
    "smart_qpos10_fast500ms_ratchet": {
        "signal_threshold": 1.0,
        "stop_loss_ticks": 4, "hold_ms": 30000,
        "max_queue_pos": 10,
        "max_wait_bars": 5,
        "ratchet_stop": True,
    },
    "smart_qpos20_conviction_prime": {
        "signal_threshold": 0.75,
        "take_profit_ticks": 6, "stop_loss_ticks": 4, "hold_ms": 30000,
        "max_queue_pos": 20,
        "conviction_exit_bars": 10, "conviction_exit_mag": 0.5,
        "prime_hours": True,
    },
    "smart_full_stack": {
        "signal_threshold": 1.0,
        "stop_loss_ticks": 3, "hold_ms": 20000,
        "max_queue_pos": 15,
        "max_wait_bars": 10,
        "ratchet_stop": True,
        "prime_hours": True,
        "latency_ms": 10,
    },
    # MAE-based exit: cut losers fast
    "mae_exit_5t_3s_tp6_sl6": {
        "signal_threshold": 0.5,
        "take_profit_ticks": 6, "stop_loss_ticks": 6, "hold_ms": 30000,
        "mae_exit_ticks": 5, "mae_exit_hold_sec": 3,
    },
}


def find_mbo_file(date_str):
    for pattern in [
        f"glbx-mdp3-{date_str}.mbo.dbn.zst",
        f"*{date_str}*.dbn.zst",
        f"*{date_str}*.dbn",
    ]:
        matches = list(MBO_DIR.glob(pattern))
        if matches:
            return matches[0]
    return None


def run_fill_sim(config_name, config, mbo_file, pred_file, output_file):
    """Run Rust fill sim with given config."""
    cmd = [
        str(FILL_SIM_BIN),
        "--mbo-file", str(mbo_file),
        "--predictions", str(pred_file),
        "--output", str(output_file),
        "--quiet",
    ]

    if "signal_threshold" in config:
        cmd.extend(["--signal-threshold", str(config["signal_threshold"])])
    if "take_profit_ticks" in config:
        cmd.extend(["--take-profit-ticks", str(config["take_profit_ticks"])])
    if "stop_loss_ticks" in config:
        cmd.extend(["--stop-loss-ticks", str(config["stop_loss_ticks"])])
    if "hold_ms" in config:
        cmd.extend(["--hold-ms", str(config["hold_ms"])])
    if "trailing_ticks" in config:
        cmd.extend(["--trailing-ticks", str(config["trailing_ticks"])])
    if "max_wait_bars" in config:
        cmd.extend(["--max-wait-bars", str(config["max_wait_bars"])])
    if "max_queue_pos" in config:
        cmd.extend(["--max-queue-pos", str(config["max_queue_pos"])])
    if "min_queue_pos" in config:
        cmd.extend(["--min-queue-pos", str(config["min_queue_pos"])])
    if config.get("ratchet_stop"):
        cmd.append("--ratchet-stop")
    if config.get("prime_hours"):
        cmd.append("--prime-hours")
    if config.get("signal_flip_exit"):
        cmd.append("--signal-flip-exit")
    if "conviction_exit_bars" in config:
        cmd.extend(["--conviction-exit-bars", str(config["conviction_exit_bars"])])
    if "conviction_exit_mag" in config:
        cmd.extend(["--conviction-exit-mag", str(config["conviction_exit_mag"])])
    if "latency_ms" in config:
        cmd.extend(["--latency-ms", str(config["latency_ms"])])
    if "time_window_start" in config:
        cmd.extend(["--time-window-start", config["time_window_start"]])
    if "time_window_end" in config:
        cmd.extend(["--time-window-end", config["time_window_end"]])
    if "mae_exit_ticks" in config:
        cmd.extend(["--mae-exit-ticks", str(config["mae_exit_ticks"])])
    if "mae_exit_hold_sec" in config:
        cmd.extend(["--mae-exit-hold-sec", str(config["mae_exit_hold_sec"])])

    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
        if result.returncode != 0:
            return None
        with open(str(output_file)) as f:
            return json.load(f)
    except Exception as e:
        LOG.warning(f"  Fill sim error: {e}")
        return None


def analyze_results(all_trades, config_name, num_dates):
    """Aggregate trade-level results."""
    if not all_trades:
        LOG.info(f"  No trades for {config_name}")
        return

    n_trades = len(all_trades)
    pnl_list = [t["pnl_ticks"] - COMMISSION_TICKS for t in all_trades]
    wins = sum(1 for p in pnl_list if p > 0)
    wr = wins / n_trades if n_trades > 0 else 0

    total_pnl = sum(pnl_list)
    avg_pnl = total_pnl / n_trades if n_trades else 0

    gross_profit = sum(p for p in pnl_list if p > 0)
    gross_loss = abs(sum(p for p in pnl_list if p < 0))
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    # Daily P&L for Sortino
    daily_pnl = defaultdict(float)
    for t in all_trades:
        date = t.get("date", "unknown")
        daily_pnl[date] += t["pnl_ticks"] - COMMISSION_TICKS

    daily_returns = list(daily_pnl.values())
    if len(daily_returns) > 1:
        neg_returns = [r for r in daily_returns if r < 0]
        downside_std = np.std(neg_returns) if neg_returns else 1e-6
        mean_daily = np.mean(daily_returns)
        sortino = mean_daily / max(downside_std, 1e-6)
    else:
        sortino = 0

    green_days = sum(1 for r in daily_returns if r > 0)

    avg_win = np.mean([p for p in pnl_list if p > 0]) if wins > 0 else 0
    avg_loss = np.mean([p for p in pnl_list if p <= 0]) if (n_trades - wins) > 0 else 0

    # Fill rate (for configs with max_wait_bars)
    n_cancelled = sum(1 for t in all_trades if t.get("cancelled", False))

    LOG.info(f"\n  *** {config_name} ***")
    LOG.info(f"  Trades: {n_trades} ({n_trades/max(num_dates,1):.1f}/day)")
    LOG.info(f"  WR: {wr*100:.1f}%")
    LOG.info(f"  Avg P&L: {avg_pnl:.3f} ticks")
    LOG.info(f"  Total P&L: {total_pnl:.1f} ticks")
    LOG.info(f"  PF: {pf:.3f}")
    LOG.info(f"  Sortino: {sortino:.3f}")
    LOG.info(f"  Green: {green_days}/{len(daily_returns)} ({green_days/max(len(daily_returns),1)*100:.0f}%)")
    LOG.info(f"  Avg Win: +{avg_win:.2f}  Avg Loss: {avg_loss:.2f}")
    if n_cancelled > 0:
        LOG.info(f"  Cancelled (max-wait): {n_cancelled}")

    return {
        "config": config_name,
        "trades": n_trades,
        "wr": wr,
        "avg_pnl": avg_pnl,
        "total_pnl": total_pnl,
        "pf": pf,
        "sortino": sortino,
        "green_pct": green_days / max(len(daily_returns), 1),
        "green_days": green_days,
        "total_days": len(daily_returns),
        "avg_win": avg_win,
        "avg_loss": avg_loss,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str,
                       default=str(LVL3_ROOT / "output" / "conditional_entry_v1"))
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(str(output_dir / "run.log"), mode="w"),
        ],
    )

    # Find all dates with both MBO + predictions
    pred_files = sorted(PRED_DIR.glob("*_predictions.npz"))
    dates = []
    for pf in pred_files:
        date_str = pf.stem.split("_")[0]
        mbo = find_mbo_file(date_str)
        if mbo:
            dates.append((date_str, mbo, pf))
    LOG.info(f"Found {len(dates)} dates with both MBO + predictions")

    all_results = []

    for ci, (config_name, config) in enumerate(CONFIGS.items()):
        LOG.info(f"\n{'='*60}")
        LOG.info(f"Config {ci+1}/{len(CONFIGS)}: {config_name}")
        LOG.info(f"{'='*60}")

        all_trades = []
        for di, (date_str, mbo_file, pred_file) in enumerate(dates):
            out_file = output_dir / f"{config_name}_{date_str}.json"
            result = run_fill_sim(config_name, config, mbo_file, pred_file, out_file)
            if result:
                trades = result.get("trades", [])
                # Add date to each trade for daily aggregation
                for t in trades:
                    t["date"] = date_str
                all_trades.extend(trades)
                n_trades = len(trades)
                LOG.info(f"  {date_str}: {n_trades} trades")
            else:
                LOG.info(f"  {date_str}: no trades or error")

        result_summary = analyze_results(all_trades, config_name, len(dates))
        if result_summary:
            all_results.append(result_summary)

    # Final summary table
    LOG.info(f"\n{'='*80}")
    LOG.info("FINAL SUMMARY — Conditional Entry Sweep v1")
    LOG.info(f"{'='*80}")
    LOG.info(f"{'Config':<45} {'Trades':>7} {'WR':>6} {'AvgPnL':>8} {'PF':>6} {'Sort':>7} {'Green%':>7}")
    LOG.info("-" * 88)

    # Sort by Sortino descending
    all_results.sort(key=lambda x: x["sortino"], reverse=True)
    for r in all_results:
        LOG.info(
            f"  {r['config']:<43} {r['trades']:>7} {r['wr']*100:>5.1f}% "
            f"{r['avg_pnl']:>+7.3f} {r['pf']:>5.3f} {r['sortino']:>+6.3f} "
            f"{r['green_days']}/{r['total_days']}"
        )

    # Save results
    with open(str(output_dir / "results_summary.json"), "w") as f:
        json.dump(all_results, f, indent=2)

    LOG.info(f"\nResults saved to {output_dir}")


if __name__ == "__main__":
    main()
