#!/usr/bin/env python3
"""
Visualize Queue Entry Selector v2 results.
Generates: threshold_performance.png, equity_curve.png, fold_heatmap.png
"""
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

OUT_DIR = Path(__file__).parent

# ── Load data ──
with open(OUT_DIR / "results.json") as f:
    results = json.load(f)

trades = pd.read_parquet(OUT_DIR / "all_oot_trades.parquet")

# ════════════════════════════════════════════════════════════════
# Chart 1: Threshold vs Performance (multi-panel)
# ════════════════════════════════════════════════════════════════

perf = results['filtered_performance']
thresholds = [0.50, 0.52, 0.55, 0.58, 0.60]
thresh_data = []
for t in thresholds:
    key = str(t)
    if key not in perf:
        continue
    d = perf[key]
    # Compute daily sharpe from trades
    sel = trades[trades['pred_prob'] >= t]
    daily_pnl = sel.groupby('date')['net_ticks'].sum()
    sharpe = float(daily_pnl.mean() / daily_pnl.std()) if len(daily_pnl) > 1 and daily_pnl.std() > 0 else 0
    thresh_data.append({
        'threshold': t,
        'wr': d['wr'],
        'pf': d['pf'],
        'sharpe': sharpe,
        'n_trades': d['n_trades'],
    })

td = pd.DataFrame(thresh_data)

fig, axes = plt.subplots(2, 2, figsize=(12, 9), facecolor='white')
fig.suptitle('Queue Entry Selector v2 — Threshold Analysis', fontsize=14, fontweight='bold')

# WR
ax = axes[0, 0]
ax.bar(td['threshold'].astype(str), td['wr'] * 100, color='#2196F3', edgecolor='#1565C0')
ax.axhline(50, color='gray', ls='--', lw=0.8, label='50%')
ax.set_ylabel('Win Rate (%)')
ax.set_xlabel('Probability Threshold')
ax.set_title('Win Rate by Threshold')
for i, row in td.iterrows():
    ax.text(i, row['wr'] * 100 + 0.3, f"{row['wr']*100:.1f}%", ha='center', fontsize=9)

# PF
ax = axes[0, 1]
colors = ['#4CAF50' if v > 1.5 else '#FF9800' if v > 1.0 else '#F44336' for v in td['pf']]
ax.bar(td['threshold'].astype(str), td['pf'], color=colors, edgecolor='#333')
ax.axhline(1.0, color='gray', ls='--', lw=0.8, label='Breakeven')
ax.set_ylabel('Profit Factor')
ax.set_xlabel('Probability Threshold')
ax.set_title('Profit Factor by Threshold')
for i, row in td.iterrows():
    ax.text(i, row['pf'] + 0.03, f"{row['pf']:.2f}", ha='center', fontsize=9)

# Sharpe
ax = axes[1, 0]
ax.bar(td['threshold'].astype(str), td['sharpe'], color='#9C27B0', edgecolor='#6A1B9A')
ax.axhline(0, color='gray', ls='--', lw=0.8)
ax.set_ylabel('Daily Sharpe')
ax.set_xlabel('Probability Threshold')
ax.set_title('Sharpe Ratio by Threshold')
for i, row in td.iterrows():
    ax.text(i, row['sharpe'] + 0.01, f"{row['sharpe']:.2f}", ha='center', fontsize=9)

# Trade count
ax = axes[1, 1]
ax.bar(td['threshold'].astype(str), td['n_trades'], color='#607D8B', edgecolor='#37474F')
ax.set_ylabel('Number of Trades')
ax.set_xlabel('Probability Threshold')
ax.set_title('Trade Count by Threshold')
for i, row in td.iterrows():
    ax.text(i, row['n_trades'] + 500, f"{int(row['n_trades']):,}", ha='center', fontsize=9)

plt.tight_layout()
plt.savefig(OUT_DIR / "threshold_performance.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved threshold_performance.png")


# ════════════════════════════════════════════════════════════════
# Chart 2: Cumulative PnL Equity Curves (threshold 0.58 and 0.60)
# ════════════════════════════════════════════════════════════════

fig, ax = plt.subplots(figsize=(14, 6), facecolor='white')
fig.suptitle('Queue Entry Selector v2 — Cumulative PnL (OOT)', fontsize=14, fontweight='bold')

for thresh, color, ls in [(0.58, '#2196F3', '-'), (0.60, '#F44336', '--')]:
    sel = trades[trades['pred_prob'] >= thresh].copy()
    daily = sel.groupby('date')['net_ticks'].sum().sort_index()
    cum = daily.cumsum()
    # Convert date strings to datetime for plotting
    dates = pd.to_datetime(cum.index, format='%Y%m%d')
    ax.plot(dates, cum.values, label=f'Threshold {thresh:.2f} (n={len(sel):,})',
            color=color, ls=ls, lw=2)
    # Add fill
    ax.fill_between(dates, 0, cum.values, alpha=0.1, color=color)

ax.axhline(0, color='gray', ls='-', lw=0.5)
ax.set_ylabel('Cumulative PnL (ticks)')
ax.set_xlabel('Date')
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Add annotations for final values
for thresh, color in [(0.58, '#2196F3'), (0.60, '#F44336')]:
    sel = trades[trades['pred_prob'] >= thresh]
    daily = sel.groupby('date')['net_ticks'].sum().sort_index()
    cum = daily.cumsum()
    final_val = cum.iloc[-1]
    final_date = pd.to_datetime(cum.index[-1], format='%Y%m%d')
    ax.annotate(f'{final_val:+.0f}t', xy=(final_date, final_val),
                fontsize=10, fontweight='bold', color=color,
                xytext=(10, 0), textcoords='offset points')

plt.tight_layout()
plt.savefig(OUT_DIR / "equity_curve.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved equity_curve.png")


# ════════════════════════════════════════════════════════════════
# Chart 3: Per-Fold Performance Heatmap (threshold 0.58)
# ════════════════════════════════════════════════════════════════

THRESH = 0.58
sel = trades[trades['pred_prob'] >= THRESH].copy()

fold_metrics = []
for fold_num in sorted(sel['fold'].unique()):
    fold_df = sel[sel['fold'] == fold_num]
    if len(fold_df) < 5:
        continue
    wr = fold_df['target'].mean()
    winners = fold_df[fold_df['net_ticks'] > 0]['net_ticks'].sum()
    losers = abs(fold_df[fold_df['net_ticks'] < 0]['net_ticks'].sum())
    pf = winners / max(losers, 1e-6)
    pnl = fold_df['net_ticks'].sum()
    n = len(fold_df)
    dates = sorted(fold_df['date'].unique())
    date_label = f"{dates[0][-4:]}-{dates[-1][-4:]}"
    fold_metrics.append({
        'fold': fold_num,
        'dates': date_label,
        'wr': wr,
        'pf': min(pf, 5.0),  # cap for display
        'pnl': pnl,
        'n_trades': n,
    })

fm = pd.DataFrame(fold_metrics)

fig, axes = plt.subplots(1, 3, figsize=(16, 7), facecolor='white')
fig.suptitle(f'Queue Entry Selector v2 — Per-Fold Stability (threshold={THRESH})',
             fontsize=14, fontweight='bold')

labels = [f"F{int(r['fold'])}\n{r['dates']}" for _, r in fm.iterrows()]

# WR heatmap as horizontal bar
ax = axes[0]
colors_wr = ['#4CAF50' if v > 0.55 else '#FF9800' if v > 0.50 else '#F44336' for v in fm['wr']]
bars = ax.barh(range(len(fm)), fm['wr'] * 100, color=colors_wr, edgecolor='#555', height=0.7)
ax.axvline(50, color='gray', ls='--', lw=0.8)
ax.set_yticks(range(len(fm)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('Win Rate (%)')
ax.set_title('Win Rate')
ax.invert_yaxis()
for i, v in enumerate(fm['wr']):
    ax.text(v * 100 + 0.3, i, f'{v*100:.1f}%', va='center', fontsize=7)

# PF heatmap as horizontal bar
ax = axes[1]
colors_pf = ['#4CAF50' if v > 1.5 else '#FF9800' if v > 1.0 else '#F44336' for v in fm['pf']]
bars = ax.barh(range(len(fm)), fm['pf'], color=colors_pf, edgecolor='#555', height=0.7)
ax.axvline(1.0, color='gray', ls='--', lw=0.8)
ax.set_yticks(range(len(fm)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('Profit Factor')
ax.set_title('Profit Factor')
ax.invert_yaxis()
for i, v in enumerate(fm['pf']):
    ax.text(v + 0.05, i, f'{v:.2f}', va='center', fontsize=7)

# PnL as horizontal bar
ax = axes[2]
colors_pnl = ['#4CAF50' if v > 0 else '#F44336' for v in fm['pnl']]
bars = ax.barh(range(len(fm)), fm['pnl'], color=colors_pnl, edgecolor='#555', height=0.7)
ax.axvline(0, color='gray', ls='-', lw=0.8)
ax.set_yticks(range(len(fm)))
ax.set_yticklabels(labels, fontsize=7)
ax.set_xlabel('PnL (ticks)')
ax.set_title('Total PnL')
ax.invert_yaxis()
for i, v in enumerate(fm['pnl']):
    offset = 2 if v >= 0 else -2
    ha = 'left' if v >= 0 else 'right'
    ax.text(v + offset, i, f'{v:+.0f}', va='center', ha=ha, fontsize=7)

plt.tight_layout()
plt.savefig(OUT_DIR / "fold_heatmap.png", dpi=150, bbox_inches='tight')
plt.close()
print("Saved fold_heatmap.png")

print("\nAll charts generated successfully.")
