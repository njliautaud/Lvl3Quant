#!/usr/bin/env python3
"""
Z-score normalization test for CNN-Mamba v2 predictions.
Hypothesis: model calibration drifted between March and April.
Running z-score normalization should make signal comparable across months.

Uses EMA-based z-score with 1-hour (36000 bars) half-life.
"""

import os
import sys
import json
import subprocess
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict

# Paths
PRED_DIR = Path("/home/nick/Lvl3Quant/output/extended_oot_validation/pred_npzs")
ZSCORE_DIR = Path("/home/nick/Lvl3Quant/output/extended_oot_validation/pred_npzs_zscore")
MBO_DIR = Path("/home/nick/Lvl3Quant/data/raw/mbo")
FILL_SIM = "/home/nick/Lvl3Quant/rust_cache_builder/target/release/fill_sim_cli"
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/zscore_pred_test")

ZSCORE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# EMA half-life in bars (1 hour = 36000 bars at 100ms)
HALFLIFE = 36000
ALPHA = 1 - np.exp(-np.log(2) / HALFLIFE)  # decay factor

# All dates from pred dir
DATES = sorted([f.stem.replace("_unfiltered", "") for f in PRED_DIR.glob("*_unfiltered.npz")])
print(f"Found {len(DATES)} dates: {DATES[0]} to {DATES[-1]}")

MARCH_DATES = [d for d in DATES if d.startswith("202603")]
APRIL_DATES = [d for d in DATES if d.startswith("202604")]
print(f"March: {len(MARCH_DATES)} dates, April: {len(APRIL_DATES)} dates")


def ema_zscore(preds: np.ndarray) -> np.ndarray:
    """
    Compute running z-score using exponential moving average and std.
    For warm-up (first hour), uses expanding window.
    Non-zero check: only updates stats on non-zero predictions (in case of forward-fill zeros).
    """
    n = len(preds)
    z = np.zeros(n, dtype=np.float32)

    # EMA state
    ema_mean = 0.0
    ema_var = 0.0
    count = 0

    for i in range(n):
        p = float(preds[i])

        if count == 0:
            # First non-zero value: initialize
            if p != 0.0:
                ema_mean = p
                ema_var = 0.0
                count = 1
                z[i] = 0.0  # can't z-score with 1 sample
            else:
                z[i] = 0.0
            continue

        if p == 0.0:
            z[i] = 0.0
            continue

        count += 1

        # During warm-up (< halflife samples), blend expanding and EMA
        if count < HALFLIFE:
            # Use a faster alpha during warm-up for quicker adaptation
            warmup_alpha = max(ALPHA, 2.0 / (count + 1))
        else:
            warmup_alpha = ALPHA

        # Update EMA mean and variance
        delta = p - ema_mean
        ema_mean = ema_mean + warmup_alpha * delta
        ema_var = (1 - warmup_alpha) * (ema_var + warmup_alpha * delta * delta)

        ema_std = np.sqrt(ema_var) if ema_var > 0 else 1.0

        # Z-score: how many stds is current pred from running mean
        if ema_std > 1e-8:
            z[i] = (p - ema_mean) / ema_std
        else:
            z[i] = 0.0

    return z


def process_date(date: str):
    """Load predictions, compute z-score, save variants."""
    pred_file = PRED_DIR / f"{date}_unfiltered.npz"
    preds = np.load(pred_file)["predictions"]

    # Compute z-score
    z = ema_zscore(preds)

    # Save full z-scored predictions
    zscore_file = ZSCORE_DIR / f"{date}_zscore.npz"
    np.savez_compressed(zscore_file, predictions=z)

    # Save buy-only z-scored (zero out negative z-scores)
    z_buy = z.copy()
    z_buy[z_buy < 0] = 0.0
    zscore_buy_file = ZSCORE_DIR / f"{date}_zscore_buy.npz"
    np.savez_compressed(zscore_buy_file, predictions=z_buy)

    stats = {
        "raw_mean": float(preds.mean()),
        "raw_std": float(preds.std()),
        "z_mean": float(z.mean()),
        "z_std": float(z.std()),
        "z_buy_nonzero": int((z_buy > 0).sum()),
        "z_nonzero": int((z != 0).sum()),
    }
    return stats


# ============================================================
# Step 1: Generate z-scored NPZs for all dates
# ============================================================
print("\n=== Step 1: Generating z-scored predictions ===")
all_stats = {}
for date in DATES:
    stats = process_date(date)
    all_stats[date] = stats
    print(f"  {date}: raw(μ={stats['raw_mean']:+.3f}, σ={stats['raw_std']:.3f}) → z(μ={stats['z_mean']:+.3f}, σ={stats['z_std']:.3f}) buy_signals={stats['z_buy_nonzero']}")

# Save stats
with open(OUTPUT_DIR / "zscore_stats.json", "w") as f:
    json.dump(all_stats, f, indent=2)

# ============================================================
# Step 2: Run fill_sim_cli for all configs
# ============================================================
print("\n=== Step 2: Running fill simulations ===")

# Config definitions
# Each config: (name, pred_suffix, extra_args)
# pred_suffix: "_zscore" or "_zscore_buy"
THRESHOLDS = [0.5, 1.0, 1.5]

BASE_CONFIGS = [
    {
        "name": "zscore_buy_afternoon",
        "pred_suffix": "_zscore_buy",
        "extra_args": ["--time-window-start", "14:00", "--time-window-end", "16:00"],
    },
    {
        "name": "zscore_buy_allday",
        "pred_suffix": "_zscore_buy",
        "extra_args": [],
    },
    {
        "name": "zscore_both_baseline",
        "pred_suffix": "_zscore",
        "extra_args": [],
    },
]

COMMON_ARGS = [
    "--take-profit-ticks", "8",
    "--stop-loss-ticks", "16",
    "--hold-ms", "1800000",  # 30 minutes
    "--latency-ms", "10",
    "--quiet",
]

results = defaultdict(list)  # config_name -> list of per-date results

total_runs = len(BASE_CONFIGS) * len(THRESHOLDS) * len(DATES)
run_count = 0

for cfg in BASE_CONFIGS:
    for thresh in THRESHOLDS:
        config_name = f"{cfg['name']}_t{thresh}"
        print(f"\n--- Config: {config_name} ---")

        for date in DATES:
            run_count += 1
            if run_count % 50 == 0:
                print(f"  Progress: {run_count}/{total_runs}")

            pred_file = ZSCORE_DIR / f"{date}{cfg['pred_suffix']}.npz"
            mbo_file = MBO_DIR / f"glbx-mdp3-{date}.mbo.dbn.zst"
            out_file = OUTPUT_DIR / f"{config_name}_{date}.json"

            if not mbo_file.exists():
                print(f"  WARNING: MBO file missing for {date}, skipping")
                continue

            cmd = [
                FILL_SIM,
                "--mbo-file", str(mbo_file),
                "--predictions", str(pred_file),
                "--output", str(out_file),
                "--signal-threshold", str(thresh),
            ] + COMMON_ARGS + cfg["extra_args"]

            try:
                proc = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
                if proc.returncode != 0:
                    print(f"  ERROR {date}: {proc.stderr[:200]}")
                    continue

                if out_file.exists():
                    with open(out_file) as f:
                        res = json.load(f)
                    res["date"] = date
                    results[config_name].append(res)
            except subprocess.TimeoutExpired:
                print(f"  TIMEOUT {date}")
            except Exception as e:
                print(f"  EXCEPTION {date}: {e}")


# ============================================================
# Step 3: Summarize results
# ============================================================
print("\n\n" + "=" * 80)
print("RESULTS SUMMARY: Z-Score Normalization Test")
print("=" * 80)

def summarize(date_results, label="ALL"):
    if not date_results:
        return None
    trades = sum(r.get("total_trades", 0) for r in date_results)
    wins = sum(r.get("winning_trades", 0) for r in date_results)
    pnl_ticks = sum(r.get("net_pnl_ticks", r.get("pnl_ticks", 0)) for r in date_results)
    gross_ticks = sum(r.get("gross_pnl_ticks", 0) for r in date_results)
    days = len(date_results)

    wr = wins / trades * 100 if trades > 0 else 0

    # Per-day P&L for Sharpe
    daily_pnl = [r.get("net_pnl_ticks", r.get("pnl_ticks", 0)) for r in date_results]
    daily_arr = np.array(daily_pnl)
    sharpe = (daily_arr.mean() / daily_arr.std() * np.sqrt(252)) if daily_arr.std() > 0 else 0

    # Sortino (downside deviation)
    neg = daily_arr[daily_arr < 0]
    downside_std = np.sqrt((neg ** 2).mean()) if len(neg) > 0 else 1e-8
    sortino = daily_arr.mean() / downside_std * np.sqrt(252) if downside_std > 1e-8 else 0

    # Profit factor
    gross_wins = sum(max(0, p) for p in daily_pnl)
    gross_losses = sum(abs(min(0, p)) for p in daily_pnl)
    pf = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    green_days = sum(1 for p in daily_pnl if p > 0)

    return {
        "label": label,
        "days": days,
        "trades": trades,
        "trades_per_day": trades / days if days > 0 else 0,
        "wr": wr,
        "pnl_ticks": pnl_ticks,
        "pnl_per_day": pnl_ticks / days if days > 0 else 0,
        "sharpe": sharpe,
        "sortino": sortino,
        "pf": pf,
        "green_days": green_days,
        "green_pct": green_days / days * 100 if days > 0 else 0,
    }


all_summaries = {}

for config_name in sorted(results.keys()):
    date_results = results[config_name]

    march_results = [r for r in date_results if r["date"].startswith("202603")]
    april_results = [r for r in date_results if r["date"].startswith("202604")]

    s_all = summarize(date_results, "ALL")
    s_mar = summarize(march_results, "MARCH")
    s_apr = summarize(april_results, "APRIL")

    all_summaries[config_name] = {"all": s_all, "march": s_mar, "april": s_apr}

    print(f"\n{'─' * 70}")
    print(f"CONFIG: {config_name}")
    print(f"{'─' * 70}")

    for s in [s_all, s_mar, s_apr]:
        if s is None:
            continue
        print(f"  {s['label']:>6s} | {s['days']}d | {s['trades']} trades ({s['trades_per_day']:.1f}/d) | "
              f"WR {s['wr']:.1f}% | PnL {s['pnl_ticks']:+.1f}t ({s['pnl_per_day']:+.1f}t/d) | "
              f"Sharpe {s['sharpe']:.2f} | Sortino {s['sortino']:.2f} | PF {s['pf']:.2f} | "
              f"Green {s['green_days']}/{s['days']} ({s['green_pct']:.0f}%)")

    # March vs April delta
    if s_mar and s_apr and s_mar['pnl_per_day'] != 0:
        delta = s_apr['pnl_per_day'] - s_mar['pnl_per_day']
        print(f"  DELTA  | April - March per-day PnL: {delta:+.1f} ticks")

# Save full results
with open(OUTPUT_DIR / "summary.json", "w") as f:
    json.dump(all_summaries, f, indent=2, default=str)

print(f"\n\nResults saved to {OUTPUT_DIR}")
print("Done!")
