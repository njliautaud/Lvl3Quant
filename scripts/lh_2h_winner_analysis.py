#!/usr/bin/env python3
"""
LH 2-Hour ES Paper Engine — Winner Analysis (HC #703)
=====================================================
Study WHY this engine is winning and compute intraday peak profit metrics.

Questions answered:
1. Direction bias — is it just riding the bull market?
2. Per-hour-slot performance — which entry hours are best?
3. Regime analysis — does it work in up AND down days?
4. MFE/MAE within each 2h hold — what's the peak profit before exit?
5. Has it stopped trading? Why?
6. Prediction confidence vs outcome — is the model actually predictive?
"""

import json
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime

ROOT = Path("/home/jupiter/Lvl3Quant")

# Load trades
trades_path = ROOT / "output" / "lh_2h_paper" / "trades.csv"
state_path = ROOT / "live_trading_linux" / "lh_2h_paper_state" / "state.json"

print("=" * 70)
print("LH 2-HOUR ES WINNER ANALYSIS (HC #703)")
print("=" * 70)

# Load state
with open(state_path) as f:
    state = json.load(f)

print(f"\nCapital: ${state['capital']:,.0f} (from $100K)")
print(f"Total trades: {state['total_trades']}")
print(f"Win rate: {state['wins']}/{state['total_trades']} = {state['wins']/state['total_trades']*100:.1f}%")
print(f"Total P&L: ${state['total_pnl_dollars']:,.0f}")
print(f"Last signal: {state['last_signal_hour']}")
print(f"Last retrain: {state['last_retrain_date']}")

# Load trades CSV
df = pd.read_csv(trades_path)

# There appear to be two sets in the CSV — use the first 74 (the authoritative set based on state)
# The second set starting at row 75+ appears to be a duplicate with different tick calculations
# Let's check
first_set = df.iloc[:74].copy()
if len(df) > 74:
    print(f"\n⚠️ CSV has {len(df)} rows but state says {state['total_trades']} trades. Using first {state['total_trades']}.")

df = first_set
df['entry_time'] = pd.to_datetime(df['entry_time'])
df['exit_time'] = pd.to_datetime(df['exit_time'])
df['entry_date'] = df['entry_time'].dt.date
df['entry_hour_utc'] = df['entry_time'].dt.hour
df['entry_hour_et'] = df['entry_hour_utc'] - 4  # UTC to ET (EDT)

# ─────────────────────────────────────────────
# 1. DIRECTION BIAS
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("1. DIRECTION BIAS")
print("=" * 70)

long_trades = df[df['direction'] == 'LONG']
short_trades = df[df['direction'] == 'SHORT']

print(f"\nLONG trades:  {len(long_trades)} ({len(long_trades)/len(df)*100:.0f}%)")
print(f"  Win rate: {(long_trades['net_pnl_ticks'] > 0).sum()}/{len(long_trades)} = {(long_trades['net_pnl_ticks'] > 0).mean()*100:.1f}%")
print(f"  Avg net P&L: {long_trades['net_pnl_ticks'].mean():.1f} ticks (${long_trades['net_pnl_dollars'].mean():.0f})")
print(f"  Total net P&L: {long_trades['net_pnl_ticks'].sum():.0f} ticks (${long_trades['net_pnl_dollars'].sum():.0f})")

print(f"\nSHORT trades: {len(short_trades)} ({len(short_trades)/len(df)*100:.0f}%)")
print(f"  Win rate: {(short_trades['net_pnl_ticks'] > 0).sum()}/{len(short_trades)} = {(short_trades['net_pnl_ticks'] > 0).mean()*100:.1f}%")
print(f"  Avg net P&L: {short_trades['net_pnl_ticks'].mean():.1f} ticks (${short_trades['net_pnl_dollars'].mean():.0f})")
print(f"  Total net P&L: {short_trades['net_pnl_ticks'].sum():.0f} ticks (${short_trades['net_pnl_dollars'].sum():.0f})")

# Market direction during trading period
first_close = df.iloc[0]['entry_price']
last_close = df.iloc[-1]['exit_price']
market_move = last_close - first_close
print(f"\nMarket move Apr 9-29: {market_move:.0f} ticks ({market_move/4:.0f} pts)")
print(f"  = ${market_move * 12.50:.0f} per contract")
print(f"  Strategy earned: ${state['total_pnl_dollars']:,.0f}")
print(f"  Buy-and-hold would earn: ${market_move * 12.50:,.0f}")
print(f"  Strategy alpha vs B&H: ${state['total_pnl_dollars'] - market_move * 12.50:,.0f}")

# ─────────────────────────────────────────────
# 2. PER-HOUR SLOT PERFORMANCE
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("2. PER-HOUR ENTRY SLOT PERFORMANCE")
print("=" * 70)

for hour_utc in sorted(df['entry_hour_utc'].unique()):
    hour_et = hour_utc - 4
    h = df[df['entry_hour_utc'] == hour_utc]
    wr = (h['net_pnl_ticks'] > 0).mean() * 100
    avg_pnl = h['net_pnl_ticks'].mean()
    total_pnl = h['net_pnl_ticks'].sum()
    print(f"  {hour_et:02d}:00 ET ({hour_utc:02d} UTC): {len(h)} trades, WR {wr:.0f}%, avg {avg_pnl:.1f}t, total {total_pnl:.0f}t (${total_pnl*12.50:.0f})")

# ─────────────────────────────────────────────
# 3. PER-DAY PERFORMANCE + REGIME
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("3. DAILY P&L + REGIME ANALYSIS")
print("=" * 70)

daily = df.groupby('entry_date').agg(
    trades=('net_pnl_ticks', 'count'),
    total_pnl_ticks=('net_pnl_ticks', 'sum'),
    total_pnl_dollars=('net_pnl_dollars', 'sum'),
    wins=('net_pnl_ticks', lambda x: (x > 0).sum()),
    first_entry=('entry_price', 'first'),
    last_exit=('exit_price', 'last'),
).reset_index()

daily['day_market_move'] = daily['last_exit'] - daily['first_entry']
daily['day_type'] = daily['day_market_move'].apply(lambda x: 'GREEN' if x > 20 else ('RED' if x < -20 else 'FLAT'))

print("\nDate       | Trades | WR    | P&L Ticks | P&L $     | Market Move | Day Type")
print("-" * 90)
for _, row in daily.iterrows():
    wr = row['wins']/row['trades']*100
    print(f"{row['entry_date']} | {row['trades']:6d} | {wr:4.0f}% | {row['total_pnl_ticks']:9.0f} | ${row['total_pnl_dollars']:8,.0f} | {row['day_market_move']:+8.0f}t | {row['day_type']}")

green_days = daily[daily['day_type'] == 'GREEN']
red_days = daily[daily['day_type'] == 'RED']
flat_days = daily[daily['day_type'] == 'FLAT']

print(f"\nGREEN days ({len(green_days)}): avg P&L ${green_days['total_pnl_dollars'].mean():,.0f}, total ${green_days['total_pnl_dollars'].sum():,.0f}")
print(f"RED days   ({len(red_days)}): avg P&L ${red_days['total_pnl_dollars'].mean():,.0f}, total ${red_days['total_pnl_dollars'].sum():,.0f}")
print(f"FLAT days  ({len(flat_days)}): avg P&L ${flat_days['total_pnl_dollars'].mean():,.0f}, total ${flat_days['total_pnl_dollars'].sum():,.0f}")

# ─────────────────────────────────────────────
# 4. PREDICTION CONFIDENCE vs OUTCOME
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("4. PREDICTION CONFIDENCE vs OUTCOME")
print("=" * 70)

df['pred_abs'] = df['prediction'].abs()
df['pred_correct'] = ((df['prediction'] > 0) & (df['gross_pnl_ticks'] > 0)) | \
                     ((df['prediction'] < 0) & (df['gross_pnl_ticks'] > 0))  # short wins
# Actually: for long, gross > 0 is correct. For short, gross > 0 is also correct (already flipped)
df['pred_correct'] = df['gross_pnl_ticks'] > 0

# Quintile analysis
df['confidence_quintile'] = pd.qcut(df['pred_abs'], 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'])
print("\nConfidence quintile analysis:")
for q, g in df.groupby('confidence_quintile'):
    wr = (g['net_pnl_ticks'] > 0).mean() * 100
    avg = g['net_pnl_ticks'].mean()
    print(f"  {q}: {len(g)} trades, WR {wr:.0f}%, avg {avg:.1f}t, sum {g['net_pnl_ticks'].sum():.0f}t")

# IC: correlation between |prediction| and net_pnl
ic = df['prediction'].corr(df['gross_pnl_ticks'])
rank_ic = df['prediction'].rank().corr(df['gross_pnl_ticks'].rank())
print(f"\nPearson IC (pred vs gross P&L): {ic:.3f}")
print(f"Rank IC (Spearman): {rank_ic:.3f}")

# ─────────────────────────────────────────────
# 5. CRITICAL: WHY DID IT STOP TRADING?
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("5. TRADING CESSATION ANALYSIS")
print("=" * 70)

trade_dates = sorted(df['entry_date'].unique())
print(f"\nFirst trade: {trade_dates[0]}")
print(f"Last trade:  {trade_dates[-1]}")
print(f"Total trading days: {len(trade_dates)}")
print(f"Last signal hour: {state['last_signal_hour']}")
print(f"Last retrain date: {state['last_retrain_date']}")
print(f"\n⚠️ Last trade was Apr 29. Engine has been running (PM2 online) but not trading.")
print(f"   Last signal was generated on Jul 12. Model was last retrained Apr 22.")
print(f"   Possible causes: stale data guard, confidence filter, or data feed issue.")

# ─────────────────────────────────────────────
# 6. RISK METRICS
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("6. RISK-ADJUSTED METRICS")
print("=" * 70)

# Per-trade stats
pnl = df['net_pnl_dollars'].values
print(f"\nPer-trade:")
print(f"  Avg win:  ${pnl[pnl > 0].mean():,.0f} ({pnl[pnl > 0].mean()/12.50:.0f} ticks)")
print(f"  Avg loss: ${pnl[pnl < 0].mean():,.0f} ({pnl[pnl < 0].mean()/12.50:.0f} ticks)")
print(f"  Win/Loss ratio: {abs(pnl[pnl > 0].mean() / pnl[pnl < 0].mean()):.2f}")
print(f"  Profit Factor: {pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum()):.2f}")

# Daily risk
daily_pnl = daily['total_pnl_dollars'].values
daily_mean = daily_pnl.mean()
daily_std = daily_pnl.std()
downside_std = daily_pnl[daily_pnl < 0].std() if (daily_pnl < 0).any() else daily_std

sharpe_daily = daily_mean / daily_std if daily_std > 0 else 0
sortino_daily = daily_mean / downside_std if downside_std > 0 else 0
sharpe_annual = sharpe_daily * np.sqrt(252)
sortino_annual = sortino_daily * np.sqrt(252)

print(f"\nDaily:")
print(f"  Mean daily P&L: ${daily_mean:,.0f}")
print(f"  Std daily P&L:  ${daily_std:,.0f}")
print(f"  Worst day: ${daily_pnl.min():,.0f}")
print(f"  Best day:  ${daily_pnl.max():,.0f}")
print(f"  Winning days: {(daily_pnl > 0).sum()}/{len(daily_pnl)} = {(daily_pnl > 0).mean()*100:.0f}%")

print(f"\nRisk-adjusted (annualized):")
print(f"  Sharpe:  {sharpe_annual:.2f}")
print(f"  Sortino: {sortino_annual:.2f}")

# Max drawdown
equity = np.cumsum(daily_pnl) + 100000
peak = np.maximum.accumulate(equity)
dd = (equity - peak) / peak * 100
print(f"  Max Drawdown: {dd.min():.2f}%")

# ─────────────────────────────────────────────
# 7. MFE ANALYSIS (USING DAILY P&L PROXY)
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("7. TRADE EFFICIENCY (MFE PROXY)")
print("=" * 70)

# Without tick-level data within each 2h bar, we can compute:
# - Gross vs net (how much do costs eat?)
# - Direction accuracy
df['cost_pct_of_gross'] = np.where(
    df['gross_pnl_ticks'].abs() > 0,
    df['cost_ticks'] / df['gross_pnl_ticks'].abs() * 100,
    100
)

winners = df[df['gross_pnl_ticks'] > 0]
losers = df[df['gross_pnl_ticks'] < 0]
flat = df[df['gross_pnl_ticks'] == 0]

print(f"\nGross win/loss/flat: {len(winners)}/{len(losers)}/{len(flat)}")
print(f"Average gross winner: {winners['gross_pnl_ticks'].mean():.0f} ticks")
print(f"Average gross loser:  {losers['gross_pnl_ticks'].mean():.0f} ticks")
print(f"Avg cost per trade:   {df['cost_ticks'].mean():.3f} ticks (${df['cost_ticks'].mean()*12.50:.2f})")
print(f"Cost as % of avg gross winner: {df['cost_ticks'].mean() / winners['gross_pnl_ticks'].mean() * 100:.1f}%")

# Big wins analysis
big_winners = df[df['gross_pnl_ticks'] > 200]
print(f"\nBig wins (>200 gross ticks): {len(big_winners)} trades")
if len(big_winners) > 0:
    print(f"  Dates: {', '.join(big_winners['entry_time'].dt.strftime('%m/%d %H:%M').values)}")
    print(f"  Avg gross: {big_winners['gross_pnl_ticks'].mean():.0f} ticks")
    print(f"  Directions: {big_winners['direction'].value_counts().to_dict()}")

# Remove top 5 trades — outlier sensitivity
sorted_pnl = df['net_pnl_ticks'].sort_values(ascending=False)
without_top5 = sorted_pnl.iloc[5:]
print(f"\n  Without top 5 trades:")
print(f"    Total P&L: {without_top5.sum():.0f} ticks (vs {sorted_pnl.sum():.0f} with)")
print(f"    Pct from top 5: {(sorted_pnl.sum() - without_top5.sum()) / sorted_pnl.sum() * 100:.1f}%")
print(f"    Still profitable? {'YES' if without_top5.sum() > 0 else 'NO'}")

# ─────────────────────────────────────────────
# 8. CONSECUTIVE TRADE ANALYSIS
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("8. STREAK ANALYSIS")
print("=" * 70)

streaks = []
current_streak = 0
for pnl_val in df['net_pnl_ticks'].values:
    if pnl_val > 0:
        if current_streak > 0:
            current_streak += 1
        else:
            if current_streak < 0:
                streaks.append(current_streak)
            current_streak = 1
    else:
        if current_streak < 0:
            current_streak -= 1
        else:
            if current_streak > 0:
                streaks.append(current_streak)
            current_streak = -1
streaks.append(current_streak)

win_streaks = [s for s in streaks if s > 0]
loss_streaks = [-s for s in streaks if s < 0]

print(f"  Max win streak:  {max(win_streaks) if win_streaks else 0}")
print(f"  Max loss streak: {max(loss_streaks) if loss_streaks else 0}")
print(f"  Avg win streak:  {np.mean(win_streaks):.1f}" if win_streaks else "")
print(f"  Avg loss streak: {np.mean(loss_streaks):.1f}" if loss_streaks else "")

# ─────────────────────────────────────────────
# 9. MARKET CONTEXT (April 2026 was what?)
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("9. MARKET CONTEXT — APRIL 9-29 2026")
print("=" * 70)

# From the trade data we can reconstruct daily open/close
daily_context = df.groupby('entry_date').agg(
    day_open=('entry_price', 'first'),
    day_close=('exit_price', 'last'),
    n_long=('direction', lambda x: (x == 'LONG').sum()),
    n_short=('direction', lambda x: (x == 'SHORT').sum()),
).reset_index()
daily_context['daily_return_pts'] = (daily_context['day_close'] - daily_context['day_open']) / 4

print(f"\nES moved from ~{daily_context['day_open'].iloc[0]:.0f} to ~{daily_context['day_close'].iloc[-1]:.0f}")
print(f"Total move: {(daily_context['day_close'].iloc[-1] - daily_context['day_open'].iloc[0])/4:.0f} points")
print(f"\nLong bias: {(df['direction'] == 'LONG').sum()}/{len(df)} = {(df['direction'] == 'LONG').mean()*100:.0f}% of trades are LONG")
up_days = (daily_context['daily_return_pts'] > 0).sum()
down_days = (daily_context['daily_return_pts'] < 0).sum()
print(f"Market up days: {up_days}, down days: {down_days}")

# Did the model just go LONG on up days and SHORT on down days?
for _, row in daily_context.iterrows():
    day_dir = 'UP' if row['daily_return_pts'] > 0 else 'DOWN'
    bias = 'LONG' if row['n_long'] > row['n_short'] else 'SHORT' if row['n_short'] > row['n_long'] else 'MIXED'
    correct = (day_dir == 'UP' and bias == 'LONG') or (day_dir == 'DOWN' and bias == 'SHORT')
    print(f"  {row['entry_date']}: Market {day_dir:4s} ({row['daily_return_pts']:+6.0f}pts), Model bias: {bias:5s}, {'✓' if correct else '✗'}")

# ─────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────
print("\n" + "=" * 70)
print("SUMMARY — KEY FINDINGS")
print("=" * 70)
print(f"""
Strategy: LightGBM 2-hour directional on ES futures (hourly bars, MBO features)
Period: Apr 9-29, 2026 ({len(trade_dates)} trading days)
Trades: {len(df)} ({len(df)/len(trade_dates):.0f}/day)
WR: {state['wins']/state['total_trades']*100:.0f}% | PF: {pnl[pnl > 0].sum() / abs(pnl[pnl < 0].sum()):.2f}
Sharpe (ann): {sharpe_annual:.2f} | Sortino (ann): {sortino_annual:.2f}
Max DD: {dd.min():.2f}%
LONG bias: {(df['direction'] == 'LONG').mean()*100:.0f}%

CRITICAL CONCERNS:
1. Only {len(trade_dates)} trading days — WAY too small for statistical significance
2. {(df['direction'] == 'LONG').mean()*100:.0f}% LONG in what appears to be a rising market
3. Model hasn't traded since Apr 29 — engine alive but idle
4. No MFE data within 2h bars to assess optimal execution
""")
