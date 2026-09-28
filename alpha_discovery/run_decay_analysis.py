#!/usr/bin/env python3
"""Analyze if Strategy B edge decays or grows over time."""
import sys, json, time, numpy as np
from pathlib import Path
from collections import defaultdict
sys.path.insert(0, str(Path(__file__).parent))
from high_conviction_strategy import (
    load_aligned_data, precompute_signals, simulate_trades,
    get_precomputed, HOLD_PERIODS, COST_STRUCTURES
)

print(f"{time.strftime('%H:%M:%S')} Loading data...")
days = load_aligned_data()
print(f"{time.strftime('%H:%M:%S')} Pre-computing signals...")
precompute_signals(days)

cost = COST_STRUCTURES['ES_futures']
hold_bars = HOLD_PERIODS['30min']

# Run best config
daily_pnl = {}
for day in days[20:]:
    signal, c, n_agree, vol_pred = get_precomputed(day)
    trades = simulate_trades(
        day, signal, c, n_agree, vol_pred,
        hold_bars=hold_bars,
        cost_spread_ticks=cost['spread_ticks'],
        cost_comm_ticks=cost['comm_ticks'],
        conviction_threshold=0.75,
        min_agreement=1,
        vol_percentile_min=60,
        time_filter='morning_afternoon',
    )
    day_pnl = sum(t['net_pnl_ticks'] for t in trades)
    day_trades = len(trades)
    daily_pnl[day['date']] = {'pnl': day_pnl, 'trades': day_trades}

# Sort by date
dates = sorted(daily_pnl.keys())
pnls = [daily_pnl[d]['pnl'] for d in dates]
trades_per_day = [daily_pnl[d]['trades'] for d in dates]

# Compute rolling metrics
window = 15  # 15-day rolling window
print(f"\n{'='*70}")
print(f"EDGE DECAY ANALYSIS -- 15-day rolling windows")
print(f"{'='*70}")
print(f"{'Window':>12} {'Dates':>25} {'PnL':>8} {'Trades':>7} {'$/Tr':>8} {'WR':>6}")
print(f"{'-'*70}")

for i in range(0, len(dates) - window + 1, window):
    chunk_dates = dates[i:i+window]
    chunk_pnl = sum(pnls[i:i+window])
    chunk_trades = sum(trades_per_day[i:i+window])
    avg_per_trade = chunk_pnl / max(chunk_trades, 1)
    
    # Win rate for this chunk
    wins = sum(1 for j in range(i, min(i+window, len(dates))) if pnls[j] > 0)
    wr = 100 * wins / len(chunk_dates)
    
    print(f"  {i+1:>3}-{min(i+window, len(dates)):>3}   {chunk_dates[0]:>10} to {chunk_dates[-1]:>10}  {chunk_pnl:>+7.1f}  {chunk_trades:>5}   {avg_per_trade:>+7.2f}  {wr:>5.1f}%")

# Cumulative P&L
cumsum = np.cumsum(pnls)
print(f"\nCumulative P&L trajectory:")
for i in [0, len(dates)//4, len(dates)//2, 3*len(dates)//4, len(dates)-1]:
    print(f"  Day {i+1:>3} ({dates[i]}): {cumsum[i]:+.1f} ticks (${cumsum[i]*12.50:+,.0f})")

# Linear regression of daily P&L to check trend
x = np.arange(len(pnls), dtype=float)
slope, intercept = np.polyfit(x, pnls, 1)
print(f"\nEdge trend: {slope:+.3f} ticks/day")
print(f"  Interpretation: {'GROWING' if slope > 0.1 else 'STABLE' if slope > -0.1 else 'DECAYING'}")
print(f"  Day 1 predicted: {intercept:+.1f}t, Day {len(pnls)} predicted: {intercept + slope*len(pnls):+.1f}t")

print(f"\n{time.strftime('%H:%M:%S')} Done.")
