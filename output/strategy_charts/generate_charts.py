#!/usr/bin/env python3
"""Generate strategy performance charts."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
import numpy as np
import pandas as pd
import json
import os

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/strategy_charts'
os.makedirs(OUTPUT_DIR, exist_ok=True)

plt.style.use('dark_background')
COLORS = {
    'green': '#00E676',
    'red': '#FF5252',
    'amber': '#FFD740',
    'blue': '#40C4FF',
    'purple': '#E040FB',
    'grey': '#9E9E9E',
    'bg': '#1a1a2e',
    'card': '#16213e',
    'card_good': '#0a3d0a',
    'card_bad': '#3d0a0a',
    'card_maybe': '#2e2e0a',
}

def styled_fig(nrows=1, ncols=1, figsize=(16, 9)):
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, facecolor=COLORS['bg'])
    if isinstance(axes, np.ndarray):
        for ax in axes.flat:
            ax.set_facecolor(COLORS['bg'])
    else:
        axes.set_facecolor(COLORS['bg'])
    return fig, axes


# =============================================================================
# CHART 1: Strategy Overview — 3-panel summary
# =============================================================================
def chart1_overview():
    fig = plt.figure(figsize=(18, 10), facecolor=COLORS['bg'])
    fig.suptitle('STRATEGY OVERVIEW — June 2026', fontsize=22, fontweight='bold',
                 color='white', y=0.97)

    gs = gridspec.GridSpec(1, 3, figure=fig, wspace=0.3, left=0.05, right=0.95,
                           top=0.88, bottom=0.08)

    # --- Panel 1: Wheel Strategy (PROFITABLE) ---
    ax1 = fig.add_subplot(gs[0])
    ax1.set_facecolor(COLORS['card_good'])
    ax1.set_xlim(0, 10); ax1.set_ylim(0, 10)
    ax1.axis('off')

    # Border
    for spine in ax1.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['green'])
        spine.set_linewidth(2)

    ax1.text(5, 9.3, 'WHEEL STRATEGY', ha='center', fontsize=16, fontweight='bold',
             color=COLORS['green'])
    ax1.text(5, 8.5, 'SPY Options · Paper Trading', ha='center', fontsize=11, color=COLORS['grey'])
    ax1.text(5, 7.5, '✓ PROFITABLE', ha='center', fontsize=14, fontweight='bold',
             color=COLORS['green'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#0a3d0a', edgecolor=COLORS['green']))

    metrics = [
        ('Sharpe Ratio', '1.49', COLORS['green']),
        ('Annual Return', '13.8%', COLORS['green']),
        ('Win Rate', '90%', COLORS['green']),
        ('Max Drawdown', '-8.4%', COLORS['amber']),
        ('Realized P&L', '+$338', COLORS['green']),
        ('Unrealized', '+$136', COLORS['blue']),
    ]
    for i, (label, val, color) in enumerate(metrics):
        y = 6.2 - i * 0.95
        ax1.text(1.5, y, label, fontsize=12, color=COLORS['grey'], va='center')
        ax1.text(8.5, y, val, fontsize=13, fontweight='bold', color=color, ha='right', va='center')

    ax1.text(5, 0.6, 'Status: ACTIVE — Continue paper trading', ha='center',
             fontsize=9, color=COLORS['green'], style='italic')

    # --- Panel 2: 30-Min FIFO Champion (DEAD) ---
    ax2 = fig.add_subplot(gs[1])
    ax2.set_facecolor(COLORS['card_bad'])
    ax2.set_xlim(0, 10); ax2.set_ylim(0, 10)
    ax2.axis('off')
    for spine in ax2.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['red'])
        spine.set_linewidth(2)

    ax2.text(5, 9.3, '30-MIN FIFO CHAMPION', ha='center', fontsize=16, fontweight='bold',
             color=COLORS['red'])
    ax2.text(5, 8.5, 'ES Futures · Backtested', ha='center', fontsize=11, color=COLORS['grey'])
    ax2.text(5, 7.5, '✗ INVALIDATED', ha='center', fontsize=14, fontweight='bold',
             color=COLORS['red'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#3d0a0a', edgecolor=COLORS['red']))

    metrics2 = [
        ('Before Fix Sharpe', '33.8', COLORS['grey']),
        ('After Fix Sharpe', '-28.9', COLORS['red']),
        ('Before Fix PF', '3.89', COLORS['grey']),
        ('After Fix PF', '0.30', COLORS['red']),
        ('Before Fix WR', '46.6%', COLORS['grey']),
        ('After Fix WR', '6.4%', COLORS['red']),
    ]
    for i, (label, val, color) in enumerate(metrics2):
        y = 6.2 - i * 0.95
        ax2.text(1.5, y, label, fontsize=12, color=COLORS['grey'], va='center')
        ax2.text(8.5, y, val, fontsize=13, fontweight='bold', color=color, ha='right', va='center')

    ax2.text(5, 0.6, 'Status: DEAD — Bar-timing bug, 0/100 configs profitable',
             ha='center', fontsize=9, color=COLORS['red'], style='italic')

    # --- Panel 3: Queue Entry Selector v2 (PROMISING) ---
    ax3 = fig.add_subplot(gs[2])
    ax3.set_facecolor(COLORS['card_maybe'])
    ax3.set_xlim(0, 10); ax3.set_ylim(0, 10)
    ax3.axis('off')
    for spine in ax3.spines.values():
        spine.set_visible(True)
        spine.set_color(COLORS['amber'])
        spine.set_linewidth(2)

    ax3.text(5, 9.3, 'QUEUE ENTRY SELECTOR v2', ha='center', fontsize=16, fontweight='bold',
             color=COLORS['amber'])
    ax3.text(5, 8.5, 'ES Futures · Tick-Level · WF OOT', ha='center', fontsize=11, color=COLORS['grey'])
    ax3.text(5, 7.5, '◐ PROMISING', ha='center', fontsize=14, fontweight='bold',
             color=COLORS['amber'],
             bbox=dict(boxstyle='round,pad=0.3', facecolor='#2e2e0a', edgecolor=COLORS['amber']))

    metrics3 = [
        ('Best Sharpe (t≥0.58)', '0.36', COLORS['green']),
        ('Best PF (t≥0.60)', '3.10', COLORS['green']),
        ('Best WR (t≥0.60)', '74.3%', COLORS['green']),
        ('Trade Count (t≥0.58)', '338', COLORS['amber']),
        ('Regime Gate', 'PASS at t=0.60', COLORS['green']),
        ('Caveats', 'Low N, long-only', COLORS['red']),
    ]
    for i, (label, val, color) in enumerate(metrics3):
        y = 6.2 - i * 0.95
        ax3.text(1.5, y, label, fontsize=12, color=COLORS['grey'], va='center')
        ax3.text(8.5, y, val, fontsize=13, fontweight='bold', color=color, ha='right', va='center')

    ax3.text(5, 0.6, 'Status: RESEARCH — Needs more OOT days + short-side',
             ha='center', fontsize=9, color=COLORS['amber'], style='italic')

    plt.savefig(os.path.join(OUTPUT_DIR, 'strategy_overview.png'), dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print("Saved strategy_overview.png")


# =============================================================================
# CHART 2: Champion Invalidated — Before/After
# =============================================================================
def chart2_champion_invalidated():
    fig, axes = styled_fig(1, 3, figsize=(18, 8))
    fig.suptitle('30-MIN FIFO CHAMPION — INVALIDATED BY BAR-TIMING BUG',
                 fontsize=20, fontweight='bold', color=COLORS['red'], y=0.97)

    # --- Bar chart: Sharpe before/after ---
    ax = axes[0]
    labels = ['Before Fix', 'After Fix']
    sharpes = [33.8, -28.9]
    colors = [COLORS['grey'], COLORS['red']]
    bars = ax.bar(labels, sharpes, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    ax.axhline(0, color='white', linewidth=0.5, alpha=0.5)
    ax.set_title('Sharpe Ratio', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('Sharpe', fontsize=12)
    for bar, val in zip(bars, sharpes):
        ypos = val + (1.5 if val > 0 else -3)
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:+.1f}',
                ha='center', fontsize=14, fontweight='bold',
                color=COLORS['green'] if val > 0 else COLORS['red'])
    # Add big arrow
    ax.annotate('', xy=(1, -28.9), xytext=(0, 33.8),
                arrowprops=dict(arrowstyle='->', color=COLORS['red'], lw=3))
    ax.text(0.5, 5, '-186%', ha='center', fontsize=16, fontweight='bold',
            color=COLORS['red'], rotation=-70)

    # --- Bar chart: Profit Factor ---
    ax = axes[1]
    pfs = [3.89, 0.30]
    bars = ax.bar(labels, pfs, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    ax.axhline(1.0, color=COLORS['amber'], linewidth=1, linestyle='--', alpha=0.7, label='Breakeven (PF=1)')
    ax.set_title('Profit Factor', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('PF', fontsize=12)
    ax.legend(fontsize=9)
    for bar, val in zip(bars, pfs):
        ypos = val + 0.15
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.2f}',
                ha='center', fontsize=14, fontweight='bold',
                color=COLORS['green'] if val > 1 else COLORS['red'])

    # --- Bar chart: Win Rate ---
    ax = axes[2]
    wrs = [46.6, 6.4]
    bars = ax.bar(labels, wrs, color=colors, width=0.5, edgecolor='white', linewidth=0.5)
    ax.axhline(50, color=COLORS['amber'], linewidth=1, linestyle='--', alpha=0.7, label='50% WR')
    ax.set_title('Win Rate %', fontsize=14, fontweight='bold', color='white')
    ax.set_ylabel('WR %', fontsize=12)
    ax.set_ylim(0, 60)
    ax.legend(fontsize=9)
    for bar, val in zip(bars, wrs):
        ypos = val + 1.5
        ax.text(bar.get_x() + bar.get_width()/2, ypos, f'{val:.1f}%',
                ha='center', fontsize=14, fontweight='bold',
                color=COLORS['green'] if val > 50 else COLORS['red'])

    # Bottom annotation
    fig.text(0.5, 0.02,
             'Root cause: bar-timing alignment bug inflated all pre-fix results. '
             '0 of 100 configs profitable after correction. Best post-fix PF = 0.85. Strategy is DEAD.',
             ha='center', fontsize=11, color=COLORS['red'], style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['card_bad'], edgecolor=COLORS['red']))

    plt.tight_layout(rect=[0, 0.06, 1, 0.92])
    plt.savefig(os.path.join(OUTPUT_DIR, 'champion_invalidated.png'), dpi=150, bbox_inches='tight',
                facecolor=COLORS['bg'])
    plt.close()
    print("Saved champion_invalidated.png")


# =============================================================================
# CHART 3: Queue Entry Selector — Threshold Analysis + Cumulative PnL
# =============================================================================
def chart3_queue_selector():
    # Load actual data
    parquet_path = '/home/jupiter/Lvl3Quant/output/queue_entry_selector_v2/all_oot_trades.parquet'
    json_path = '/home/jupiter/Lvl3Quant/output/queue_entry_selector_v2/results.json'

    with open(json_path) as f:
        results = json.load(f)

    fp = results['filtered_performance']
    thresholds = sorted([float(k) for k in fp.keys()])
    n_trades = [fp[str(t)]['n_trades'] for t in thresholds]
    wrs = [fp[str(t)]['wr'] * 100 for t in thresholds]
    pfs = [fp[str(t)]['pf'] for t in thresholds]
    sharpes = [fp[str(t)]['sharpe'] for t in thresholds]
    pnls = [fp[str(t)]['pnl'] for t in thresholds]

    fig = plt.figure(figsize=(20, 14), facecolor=COLORS['bg'])
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3,
                           left=0.07, right=0.95, top=0.92, bottom=0.06)
    fig.suptitle('QUEUE ENTRY SELECTOR v2 — Threshold Analysis',
                 fontsize=20, fontweight='bold', color=COLORS['amber'], y=0.97)

    # --- Top-left: WR + PF by threshold ---
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(COLORS['bg'])
    ax1b = ax1.twinx()

    thresh_labels = [f'{t:.2f}' for t in thresholds]
    x = np.arange(len(thresholds))

    wr_colors = [COLORS['green'] if w > 50 else COLORS['red'] for w in wrs]
    bars1 = ax1.bar(x - 0.18, wrs, 0.35, color=wr_colors, alpha=0.8, label='Win Rate %')
    pf_colors = [COLORS['blue'] if p > 1 else COLORS['purple'] for p in pfs]
    bars2 = ax1b.bar(x + 0.18, pfs, 0.35, color=pf_colors, alpha=0.8, label='Profit Factor')

    ax1.axhline(50, color=COLORS['amber'], linewidth=1, linestyle='--', alpha=0.5)
    ax1b.axhline(1, color=COLORS['blue'], linewidth=1, linestyle='--', alpha=0.5)

    ax1.set_xticks(x)
    ax1.set_xticklabels(thresh_labels, fontsize=10)
    ax1.set_xlabel('Confidence Threshold', fontsize=12)
    ax1.set_ylabel('Win Rate %', fontsize=12, color=COLORS['green'])
    ax1b.set_ylabel('Profit Factor', fontsize=12, color=COLORS['blue'])
    ax1.set_title('Win Rate & Profit Factor by Threshold', fontsize=13, fontweight='bold')

    # Annotate bars
    for bar, val in zip(bars1, wrs):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                 f'{val:.0f}%', ha='center', fontsize=8, color='white', fontweight='bold')
    for bar, val in zip(bars2, pfs):
        ax1b.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                  f'{val:.2f}', ha='center', fontsize=8, color='white', fontweight='bold')

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left', fontsize=9)

    # --- Top-right: Sharpe + Trade count ---
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(COLORS['bg'])
    ax2b = ax2.twinx()

    sharpe_colors = [COLORS['green'] if s > 0 else COLORS['red'] for s in sharpes]
    ax2.bar(x - 0.18, sharpes, 0.35, color=sharpe_colors, alpha=0.8, label='Sharpe')
    ax2.axhline(0, color='white', linewidth=0.5, alpha=0.5)

    ax2b.plot(x, n_trades, 'o-', color=COLORS['amber'], linewidth=2, markersize=8, label='Trade Count')
    ax2b.set_yscale('log')

    ax2.set_xticks(x)
    ax2.set_xticklabels(thresh_labels, fontsize=10)
    ax2.set_xlabel('Confidence Threshold', fontsize=12)
    ax2.set_ylabel('Sharpe Ratio', fontsize=12, color=COLORS['green'])
    ax2b.set_ylabel('Trade Count (log)', fontsize=12, color=COLORS['amber'])
    ax2.set_title('Sharpe Ratio & Trade Count by Threshold', fontsize=13, fontweight='bold')

    # Annotate
    for i, (s, n) in enumerate(zip(sharpes, n_trades)):
        ax2.text(i - 0.18, s + (0.05 if s >= 0 else -0.15), f'{s:.2f}',
                 ha='center', fontsize=8, color='white', fontweight='bold')
        ax2b.text(i + 0.18, n * 1.3, f'{n:,}', ha='center', fontsize=8,
                  color=COLORS['amber'], fontweight='bold')

    lines1, labels1 = ax2.get_legend_handles_labels()
    lines2, labels2 = ax2b.get_legend_handles_labels()
    ax2.legend(lines1 + lines2, labels1 + labels2, loc='upper right', fontsize=9)

    # --- Bottom: Cumulative PnL curves ---
    df = pd.read_parquet(parquet_path)

    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_facecolor(COLORS['bg'])

    for thresh, color, lw in [(0.55, COLORS['blue'], 1.5),
                               (0.58, COLORS['green'], 2.5),
                               (0.60, COLORS['purple'], 2.0)]:
        mask = df['pred_prob'] >= thresh
        subset = df[mask].sort_values('ts_ns').copy()
        subset['cum_pnl'] = subset['net_ticks'].cumsum()
        label = f't≥{thresh:.2f} ({len(subset)} trades, PF={fp[str(thresh)]["pf"]:.2f})'
        ax3.plot(range(len(subset)), subset['cum_pnl'].values, color=color,
                 linewidth=lw, label=label, alpha=0.9)

    # Also plot 0.52 as a "barely breakeven" reference
    mask_52 = df['pred_prob'] >= 0.52
    sub_52 = df[mask_52].sort_values('ts_ns').copy()
    sub_52['cum_pnl'] = sub_52['net_ticks'].cumsum()
    ax3.plot(range(len(sub_52)), sub_52['cum_pnl'].values, color=COLORS['grey'],
             linewidth=1, label=f't≥0.52 ({len(sub_52)} trades, PF={fp["0.52"]["pf"]:.2f})',
             alpha=0.6, linestyle='--')

    ax3.axhline(0, color='white', linewidth=0.5, alpha=0.3)
    ax3.set_xlabel('Trade Number', fontsize=12)
    ax3.set_ylabel('Cumulative Net Ticks', fontsize=12)
    ax3.set_title('Cumulative PnL by Confidence Threshold (OOT Walk-Forward)',
                  fontsize=14, fontweight='bold')
    ax3.legend(fontsize=11, loc='upper left')

    # Add caveat box
    caveat_text = ('Caveats: Regime gate fails at high thresholds (sparse red-day data). '
                   'Long-only bias at t≥0.58. Per-fold instability observed. '
                   'Needs more OOT days before live deployment.')
    ax3.text(0.5, -0.12, caveat_text, transform=ax3.transAxes, ha='center',
             fontsize=10, color=COLORS['amber'], style='italic',
             bbox=dict(boxstyle='round,pad=0.4', facecolor=COLORS['card_maybe'],
                       edgecolor=COLORS['amber'], alpha=0.8))

    plt.savefig(os.path.join(OUTPUT_DIR, 'queue_selector_performance.png'), dpi=150,
                bbox_inches='tight', facecolor=COLORS['bg'])
    plt.close()
    print("Saved queue_selector_performance.png")


if __name__ == '__main__':
    chart1_overview()
    chart2_champion_invalidated()
    chart3_queue_selector()
    print(f"\nAll charts saved to {OUTPUT_DIR}/")
