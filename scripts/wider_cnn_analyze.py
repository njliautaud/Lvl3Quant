#!/usr/bin/env python3
"""Quick analysis of wider CNN fill_sim results."""
import json
import glob
import numpy as np
from collections import defaultdict

OUTPUT_DIR = "/home/jupiter/Lvl3Quant/data/processed/wider_cnn_fill_sim"
files = sorted(glob.glob(f"{OUTPUT_DIR}/*.json"))

if not files:
    print("No result files found!")
    exit(1)

print(f"Total result files: {len(files)}")
print()

# Parse all results
by_config = defaultdict(list)
for f in files:
    fname = f.split("/")[-1]
    if fname == "wider_cnn_summary.json":
        continue
    try:
        d = json.load(open(f))
        date = fname[:10]
        config = fname[11:].replace(".json", "")
        d["_date"] = date
        d["_config"] = config
        by_config[config].append(d)
    except Exception as e:
        print(f"ERROR reading {fname}: {e}")

# Summary by config
print(f"{'Config':<30s} {'Days':>4s} {'Trades':>7s} {'Fills':>7s} {'FillR%':>6s} {'TotalPnL':>10s} {'AvgPnL':>8s} {'Sharpe':>7s}")
print("-" * 90)

configs_summary = []
for config in sorted(by_config.keys()):
    runs = by_config[config]
    pnls = [r.get("total_pnl_dollars", 0) for r in runs]
    trades = sum(r.get("total_trades", 0) for r in runs)
    fills = sum(r.get("total_fills", 0) for r in runs)
    signals = sum(r.get("total_signals", 0) for r in runs)
    fill_rate = fills / signals * 100 if signals > 0 else 0

    pnls_arr = np.array(pnls)
    total_pnl = pnls_arr.sum()
    avg_pnl = pnls_arr.mean()
    std_pnl = pnls_arr.std()
    sharpe = (avg_pnl / std_pnl * (252 ** 0.5)) if std_pnl > 0 else 0

    configs_summary.append((config, len(runs), trades, fills, fill_rate, total_pnl, avg_pnl, sharpe))
    print(f"{config:<30s} {len(runs):>4d} {trades:>7d} {fills:>7d} {fill_rate:>5.1f}% ${total_pnl:>9,.0f} ${avg_pnl:>7,.0f} {sharpe:>7.2f}")

# Sort by Sharpe
print()
print("=" * 90)
print("TOP CONFIGS BY SHARPE:")
print("=" * 90)
for config, days, trades, fills, fill_rate, total_pnl, avg_pnl, sharpe in sorted(configs_summary, key=lambda x: x[7], reverse=True):
    print(f"  {config:<30s} Sharpe={sharpe:>6.2f}  PnL=${total_pnl:>9,.0f}  Trades={trades:>5d}  FillRate={fill_rate:.1f}%")

# Per-date breakdown for top config
print()
best_config = sorted(configs_summary, key=lambda x: x[7], reverse=True)[0][0]
print(f"DAILY BREAKDOWN: {best_config}")
print("-" * 70)
for r in sorted(by_config[best_config], key=lambda x: x["_date"]):
    pnl = r.get("total_pnl_dollars", 0)
    trades = r.get("total_trades", 0)
    fills = r.get("total_fills", 0)
    signals = r.get("total_signals", 0)
    print(f"  {r['_date']}: PnL=${pnl:>8,.1f}  Trades={trades:>4d}  Fills={fills:>4d}/{signals:>5d}")

# Also show HFT chase stats if available
print()
print("HFT CHASE DAILY BREAKDOWN:")
print("-" * 70)
if "wider_hft_tp4_chase" in by_config:
    for r in sorted(by_config["wider_hft_tp4_chase"], key=lambda x: x["_date"]):
        pnl = r.get("total_pnl_dollars", 0)
        trades = r.get("total_trades", 0)
        fills = r.get("total_fills", 0)
        signals = r.get("total_signals", 0)
        print(f"  {r['_date']}: PnL=${pnl:>8,.1f}  Trades={trades:>4d}  Fills={fills:>4d}/{signals:>5d}")
