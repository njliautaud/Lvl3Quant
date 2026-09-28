#!/usr/bin/env python3
"""Slippage sensitivity analysis for integrated pipeline paper engine."""

import pandas as pd
import numpy as np

trades = pd.read_csv('/home/jupiter/Lvl3Quant/paper_engines/logs/integrated_pipeline_paper/trades.csv')
print(f'Loaded {len(trades)} trades')

results = []
for extra_slip in np.arange(0, 3.1, 0.25):
    adjusted_pnl = trades['net_pnl_ticks'] - extra_slip

    daily = trades.copy()
    daily['adj_pnl'] = adjusted_pnl
    daily_agg = daily.groupby('date')['adj_pnl'].sum()

    total = adjusted_pnl.sum()
    n_trades = len(adjusted_pnl)
    wins = (adjusted_pnl > 0).sum()
    wr = wins / n_trades * 100

    daily_returns = daily_agg.values
    n_days = len(daily_returns)
    sharpe = np.mean(daily_returns) / np.std(daily_returns) * np.sqrt(252) if np.std(daily_returns) > 0 else 0

    gross_win = adjusted_pnl[adjusted_pnl > 0].sum()
    gross_loss = abs(adjusted_pnl[adjusted_pnl < 0].sum())
    pf = gross_win / gross_loss if gross_loss > 0 else np.inf

    cumsum = daily_returns.cumsum()
    running_max = np.maximum.accumulate(cumsum)
    dd = running_max - cumsum
    max_dd = dd.max()

    neg_rets = daily_returns[daily_returns < 0]
    downside_std = np.std(neg_rets) if len(neg_rets) > 1 else 0.01
    sortino = np.mean(daily_returns) / downside_std * np.sqrt(252) if downside_std > 0 else 0

    results.append({
        'extra_slip': extra_slip,
        'total_ticks': total,
        'total_dollars': total * 12.50,
        'wr': wr,
        'sharpe': sharpe,
        'sortino': sortino,
        'pf': pf,
        'max_dd_ticks': max_dd,
        'avg_per_trade': total / n_trades,
        'per_day_avg': total / n_days
    })

print()
print(f'SLIPPAGE SENSITIVITY ANALYSIS -- {n_trades} trades, {n_days} active days')
print(f'Base already includes: 0.376t RT passive, 1.376t RT market')
print()
header = f'{"Extra":>8} {"Net":>10} {"Net":>10} {"WR":>6} {"Sharpe":>8} {"Sortino":>9} {"PF":>6} {"MaxDD":>8} {"Avg":>8}'
print(header)
subheader = f'{"Slip":>8} {"Ticks":>10} {"Dollars":>10} {"%":>6} {"":>8} {"":>9} {"":>6} {"Ticks":>8} {"$/tr":>8}'
print(subheader)
print('-' * 82)

for r in results:
    flag = ''
    if r['sharpe'] < 3: flag = ' << WEAK'
    if r['sharpe'] < 1: flag = ' << DEAD'
    if r['pf'] < 1: flag = ' << UNPROFITABLE'
    dollar_per_trade = r['total_dollars'] / n_trades
    print(f'{r["extra_slip"]:>7.2f}t {r["total_ticks"]:>10.1f} {r["total_dollars"]:>10.0f} {r["wr"]:>5.1f}% {r["sharpe"]:>8.2f} {r["sortino"]:>9.2f} {r["pf"]:>6.2f} {r["max_dd_ticks"]:>7.1f}t {dollar_per_trade:>7.1f}{flag}')

print()
# Find key thresholds
for r in results:
    if r['total_ticks'] <= 0:
        print(f'BREAKEVEN at ~{r["extra_slip"]:.2f} ticks extra slippage')
        break

for r in results:
    if r['sharpe'] < 3:
        print(f'Sharpe < 3.0 at {r["extra_slip"]:.2f}t extra slippage')
        break

for r in results:
    if r['sharpe'] < 1:
        print(f'Sharpe < 1.0 at {r["extra_slip"]:.2f}t extra slippage')
        break

# Also analyze by exit type
print('\n--- SLIPPAGE IMPACT BY EXIT TYPE ---')
for etype in ['TP', 'SL', 'TIME']:
    subset = trades[trades['exit_type'] == etype]
    if len(subset) > 0:
        avg_raw = subset['raw_pnl_ticks'].mean()
        avg_net = subset['net_pnl_ticks'].mean()
        avg_cost = subset['cost_ticks'].mean()
        print(f'{etype:>4}: {len(subset):>4} trades, avg raw {avg_raw:>+7.2f}t, cost {avg_cost:.3f}t, net {avg_net:>+7.2f}t')

# Direction analysis under stress
print('\n--- DIRECTION ROBUSTNESS UNDER 1t EXTRA SLIPPAGE ---')
for direction in ['LONG', 'SHORT']:
    sub = trades[trades['direction'] == direction]
    adj = sub['net_pnl_ticks'] - 1.0
    wins = (adj > 0).sum()
    total = adj.sum()
    print(f'{direction:>5}: {len(sub)} trades, adj WR {wins/len(sub)*100:.1f}%, adj total {total:+.1f}t ({total*12.50:+.0f})')

# Confidence tier analysis under stress
print('\n--- CONFIDENCE TIERS UNDER 1t EXTRA SLIPPAGE ---')
for label, lo, hi in [('Very High', 0.65, 1.0), ('High', 0.60, 0.65), ('Medium', 0.55, 0.60), ('Low', 0.52, 0.55)]:
    sub = trades[(trades['confidence'] >= lo) & (trades['confidence'] < hi)]
    if len(sub) == 0:
        continue
    adj = sub['net_pnl_ticks'] - 1.0
    wins = (adj > 0).sum()
    total = adj.sum()
    wr = wins / len(sub) * 100
    print(f'{label:>9} ({lo:.2f}-{hi:.2f}): {len(sub):>4} trades, adj WR {wr:.1f}%, adj total {total:+.1f}t ({total*12.50:+.0f})')
