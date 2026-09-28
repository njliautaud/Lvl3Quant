#!/usr/bin/env python3
"""
Quick threshold analysis for meta v7 production predictions.
Checks if optimal filter thresholds changed with v7's stronger signal.
"""
import numpy as np
from collections import defaultdict

COMMISSION_TICKS = 0.376  # passive-passive canonical

# Load data
d = np.load("/home/jupiter/Lvl3Quant/output/meta_v7_prod/concat_oot_predictions.npz", allow_pickle=True)
preds = d["predictions"]
labels = d["labels"]
dates = d["dates"]
unique_dates = np.unique(dates)
n_days = len(unique_dates)

print(f"Meta v7 concat OOT: {len(preds):,} predictions, {n_days} days")
print(f"Date range: {unique_dates[0]} - {unique_dates[-1]}")
print()

# We analyze SHORT signals only (pred < 0), since that's where the edge is
# For shorts: profit = -label (we sell, price drops = profit)
# Signal strength = |pred| for shorts (more negative = stronger short signal)

short_mask = preds < 0
short_preds = preds[short_mask]
short_labels = labels[short_mask]
short_dates = dates[short_mask]

print(f"Short signals: {short_mask.sum():,} ({short_mask.mean()*100:.1f}%)")
print()

# Meta filter percentiles: filter on prediction STRENGTH (magnitude)
# Higher magnitude = stronger signal
short_strength = np.abs(short_preds)

# Signal threshold percentiles (top N% strongest shorts)
signal_thresholds = [0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50, 1.00]

# For each threshold, compute metrics
print(f"{'Thresh':>8} {'Trades':>8} {'Tr/Day':>8} {'AvgNet':>8} {'WR':>6} {'PF':>7} {'Sharpe':>8} {'Sortino':>8}")
print("-" * 80)

for thresh in signal_thresholds:
    # Top thresh% of short signals by magnitude
    cutoff = np.percentile(short_strength, (1.0 - thresh) * 100)
    mask = short_strength >= cutoff

    sel_labels = short_labels[mask]
    sel_dates = short_dates[mask]
    n_trades = len(sel_labels)

    if n_trades < 10:
        continue

    # For shorts: PnL = -label - cost
    pnl_ticks = -sel_labels - COMMISSION_TICKS

    avg_net = pnl_ticks.mean()
    wr = (pnl_ticks > 0).mean() * 100

    gross_wins = pnl_ticks[pnl_ticks > 0].sum()
    gross_losses = -pnl_ticks[pnl_ticks < 0].sum()
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    # Daily Sharpe
    daily_pnl = []
    for day in unique_dates:
        day_mask = sel_dates == day
        if day_mask.sum() > 0:
            daily_pnl.append(pnl_ticks[day_mask].sum())
        else:
            daily_pnl.append(0.0)
    daily_pnl = np.array(daily_pnl)

    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

    # Sortino
    downside = daily_pnl[daily_pnl < 0]
    downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-9
    sortino = daily_pnl.mean() / downside_std * np.sqrt(252)

    trades_per_day = n_trades / n_days

    print(f"{thresh*100:>7.0f}% {n_trades:>8,} {trades_per_day:>8.1f} {avg_net:>+8.3f} {wr:>5.1f}% {pf:>7.2f} {sharpe:>+8.2f} {sortino:>+8.2f}")

print()
print("=" * 80)
print("LONG side check (top N% positive predictions)")
print(f"{'Thresh':>8} {'Trades':>8} {'Tr/Day':>8} {'AvgNet':>8} {'WR':>6} {'PF':>7} {'Sharpe':>8}")
print("-" * 80)

long_mask = preds > 0
long_preds = preds[long_mask]
long_labels = labels[long_mask]
long_dates = dates[long_mask]
long_strength = np.abs(long_preds)

for thresh in [0.02, 0.05, 0.10, 0.20]:
    cutoff = np.percentile(long_strength, (1.0 - thresh) * 100)
    mask = long_strength >= cutoff

    sel_labels = long_labels[mask]
    sel_dates = long_dates[mask]
    n_trades = len(sel_labels)

    if n_trades < 10:
        continue

    # For longs: PnL = +label - cost
    pnl_ticks = sel_labels - COMMISSION_TICKS

    avg_net = pnl_ticks.mean()
    wr = (pnl_ticks > 0).mean() * 100

    gross_wins = pnl_ticks[pnl_ticks > 0].sum()
    gross_losses = -pnl_ticks[pnl_ticks < 0].sum()
    pf = gross_wins / gross_losses if gross_losses > 0 else float("inf")

    daily_pnl = []
    for day in unique_dates:
        day_mask = sel_dates == day
        if day_mask.sum() > 0:
            daily_pnl.append(pnl_ticks[day_mask].sum())
        else:
            daily_pnl.append(0.0)
    daily_pnl = np.array(daily_pnl)

    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0
    trades_per_day = n_trades / n_days

    print(f"{thresh*100:>7.0f}% {n_trades:>8,} {trades_per_day:>8.1f} {avg_net:>+8.3f} {wr:>5.1f}% {pf:>7.2f} {sharpe:>+8.2f}")

# Green/Red day breakdown for best config
print()
print("=" * 80)
print("REGIME CHECK: Top 5% shorts — daily PnL by day")
print("-" * 80)

cutoff_5 = np.percentile(short_strength, 95)
mask_5 = short_strength >= cutoff_5
sel_labels_5 = short_labels[mask_5]
sel_dates_5 = short_dates[mask_5]
pnl_5 = -sel_labels_5 - COMMISSION_TICKS

green_days = 0
red_days = 0
flat_days = 0
green_pnl = []
red_pnl = []

for day in unique_dates:
    day_mask = sel_dates_5 == day
    day_total = pnl_5[day_mask].sum() if day_mask.sum() > 0 else 0.0
    n_day = day_mask.sum()

    # Classify day regime by label mean (crude proxy)
    all_day_mask = dates == day
    day_label_mean = labels[all_day_mask].mean()

    if day_label_mean > 0.05:
        regime = "GREEN"
        green_days += 1
        green_pnl.append(day_total)
    elif day_label_mean < -0.05:
        regime = "RED"
        red_days += 1
        red_pnl.append(day_total)
    else:
        regime = "FLAT"
        flat_days += 1

    print(f"  {day} | {regime:5s} | trades={n_day:4d} | pnl={day_total:+8.1f} ticks")

green_pnl = np.array(green_pnl) if green_pnl else np.array([0.0])
red_pnl = np.array(red_pnl) if red_pnl else np.array([0.0])

green_sharpe = green_pnl.mean() / green_pnl.std() * np.sqrt(252) if green_pnl.std() > 0 else 0
red_sharpe = red_pnl.mean() / red_pnl.std() * np.sqrt(252) if red_pnl.std() > 0 else 0

print()
print(f"Green days: {green_days}, avg pnl={green_pnl.mean():+.1f}, Sharpe={green_sharpe:+.2f}")
print(f"Red days: {red_days}, avg pnl={red_pnl.mean():+.1f}, Sharpe={red_sharpe:+.2f}")

if max(abs(green_sharpe), abs(red_sharpe)) > 0:
    skew = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe))
    print(f"Regime skew: {skew:.2f} (reject if > 0.50)")
