"""
Phase 1 Walk-Forward LightGBM Results Chart
Generates a professional dark-themed multi-panel chart.
"""

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os

# ── Data ──────────────────────────────────────────────────────────────────────
horizons   = ['3s', '5s', '10s', '30s', '1m']
x_pos      = np.arange(len(horizons))

ic         = [0.1638, 0.1354, 0.1036, 0.0621, 0.0392]
icir       = [1.43,   1.49,   1.30,   0.89,   0.74  ]
t_stat     = [15.58,  16.23,  14.22,  9.76,   7.96  ]
hit_rate   = [14.1,   15.9,   18.0,   20.5,   22.0  ]   # percent
consistency= [90,     91,     91,     91,     85    ]   # percent
avg_move   = [0.32,   0.43,   0.63,   1.13,   1.62  ]   # ticks
cost_ratio = [0.26,   0.35,   0.51,   0.91,   1.31  ]

# ── Style constants ────────────────────────────────────────────────────────────
BG_DARK    = '#0d1117'   # outer background
BG_PANEL   = '#161b22'   # individual panel background
BG_AXES    = '#1c2128'   # axes face
GRID_CLR   = '#30363d'   # grid lines
SPINE_CLR  = '#30363d'   # axis spines

GREEN      = '#3fb950'   # primary positive
GREEN_LITE = '#56d364'
TEAL       = '#39d0c8'
BLUE       = '#58a6ff'
ORANGE     = '#f0883e'
RED        = '#f85149'
YELLOW     = '#e3b341'
PURPLE     = '#bc8cff'
TEXT_MAIN  = '#e6edf3'
TEXT_DIM   = '#8b949e'
TEXT_MUTED = '#484f58'

BAR_W = 0.55

# ── Figure layout ─────────────────────────────────────────────────────────────
fig = plt.figure(figsize=(16, 11), facecolor=BG_DARK)

# Title area
fig.text(
    0.5, 0.970,
    'Phase 1: Walk-Forward LightGBM — 124 Days, 24.5M Bars, 129 Features',
    ha='center', va='top',
    fontsize=17, fontweight='bold', color=TEXT_MAIN,
    fontfamily='DejaVu Sans'
)
fig.text(
    0.5, 0.947,
    'All predictions are 100% out-of-sample (expanding window walk-forward)',
    ha='center', va='top',
    fontsize=11, color=TEXT_DIM, style='italic',
    fontfamily='DejaVu Sans'
)

# 2×2 grid with tight margins
gs = fig.add_gridspec(
    2, 2,
    left=0.07, right=0.97,
    top=0.90,  bottom=0.09,
    wspace=0.38, hspace=0.52
)

def style_ax(ax, title, ylabel_left=None, ylabel_right=None):
    """Apply consistent dark-theme styling to an axis."""
    ax.set_facecolor(BG_AXES)
    for spine in ax.spines.values():
        spine.set_edgecolor(SPINE_CLR)
        spine.set_linewidth(0.8)
    ax.tick_params(colors=TEXT_DIM, labelsize=9, length=3, width=0.7)
    ax.yaxis.label.set_color(TEXT_DIM)
    ax.xaxis.label.set_color(TEXT_DIM)
    ax.set_title(title, color=TEXT_MAIN, fontsize=11, fontweight='bold', pad=8)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(['3s', '5s', '10s', '30s', '1m'],
                        color=TEXT_DIM, fontsize=9.5)
    ax.set_xlabel('Prediction Horizon', color=TEXT_DIM, fontsize=9)
    ax.grid(axis='y', color=GRID_CLR, linewidth=0.6, alpha=0.7, linestyle='--')
    ax.set_axisbelow(True)
    if ylabel_left:
        ax.set_ylabel(ylabel_left, color=TEXT_DIM, fontsize=9)

# ── Panel 1 – IC and ICIR ─────────────────────────────────────────────────────
ax1 = fig.add_subplot(gs[0, 0])
style_ax(ax1, 'IC & ICIR Across Horizons', ylabel_left='Information Coefficient (IC)')

bar_ic = ax1.bar(x_pos - BAR_W/4, ic, width=BAR_W/2 + 0.05,
                  color=GREEN, alpha=0.88, label='IC', zorder=3,
                  edgecolor='none')

ax1_r = ax1.twinx()
ax1_r.set_facecolor('none')
ax1_r.spines['right'].set_edgecolor(TEAL)
ax1_r.spines['right'].set_linewidth(0.8)
for s in ['top', 'left', 'bottom']:
    ax1_r.spines[s].set_visible(False)
ax1_r.tick_params(colors=TEAL, labelsize=9, length=3, width=0.7)

bar_icir = ax1_r.bar(x_pos + BAR_W/4, icir, width=BAR_W/2 + 0.05,
                      color=TEAL, alpha=0.78, label='ICIR', zorder=3,
                      edgecolor='none')
ax1_r.set_ylabel('IC Information Ratio (ICIR)', color=TEAL, fontsize=9)
ax1_r.tick_params(axis='y', colors=TEAL)

# Value labels
for rect, val in zip(bar_ic, ic):
    ax1.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.003,
             f'{val:.4f}', ha='center', va='bottom', fontsize=7.5,
             color=GREEN_LITE, fontweight='bold')
for rect, val in zip(bar_icir, icir):
    ax1_r.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.02,
               f'{val:.2f}', ha='center', va='bottom', fontsize=7.5,
               color=TEAL, fontweight='bold')

ax1.set_ylim(0, max(ic) * 1.40)
ax1_r.set_ylim(0, max(icir) * 1.40)

leg_patches = [
    mpatches.Patch(color=GREEN, alpha=0.88, label='IC'),
    mpatches.Patch(color=TEAL,  alpha=0.78, label='ICIR'),
]
ax1.legend(handles=leg_patches, loc='upper right', fontsize=8,
           facecolor=BG_PANEL, edgecolor=SPINE_CLR, labelcolor=TEXT_DIM)

# ── Panel 2 – Cost Ratio ──────────────────────────────────────────────────────
ax2 = fig.add_subplot(gs[0, 1])
style_ax(ax2, 'Cost Ratio vs Viability Thresholds', ylabel_left='Cost Ratio (signal / costs)')

colors_cr = [GREEN if v < 0.5 else (YELLOW if v < 1.0 else RED) for v in cost_ratio]
bars_cr = ax2.bar(x_pos, cost_ratio, width=BAR_W,
                   color=colors_cr, alpha=0.88, zorder=3, edgecolor='none')

ax2.axhline(y=1.0, color=RED,    linestyle='--', linewidth=1.6, zorder=4,
            label='Breakeven (1.0)')
ax2.axhline(y=0.5, color=YELLOW, linestyle='--', linewidth=1.6, zorder=4,
            label='Viability (0.5)')

# Value labels
for rect, val in zip(bars_cr, cost_ratio):
    c = GREEN_LITE if val < 0.5 else (YELLOW if val < 1.0 else RED)
    ax2.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.015,
             f'{val:.2f}', ha='center', va='bottom', fontsize=8.5,
             color=c, fontweight='bold')

ax2.set_ylim(0, max(cost_ratio) * 1.35)

# Threshold labels on the right
ax2.text(len(horizons) - 0.44, 1.0 + 0.03, 'Breakeven',
         color=RED, fontsize=7.5, va='bottom', ha='right', style='italic')
ax2.text(len(horizons) - 0.44, 0.5 + 0.03, 'Viable',
         color=YELLOW, fontsize=7.5, va='bottom', ha='right', style='italic')

ax2.legend(loc='upper left', fontsize=8, facecolor=BG_PANEL,
           edgecolor=SPINE_CLR, labelcolor=TEXT_DIM)

# ── Panel 3 – t-statistic ─────────────────────────────────────────────────────
ax3 = fig.add_subplot(gs[1, 0])
style_ax(ax3, 't-Statistic Across Horizons', ylabel_left='t-Statistic')

colors_t = [GREEN if v >= 10 else (YELLOW if v >= 2 else RED) for v in t_stat]
bars_t = ax3.bar(x_pos, t_stat, width=BAR_W,
                  color=colors_t, alpha=0.88, zorder=3, edgecolor='none')

ax3.axhline(y=2.0, color=RED, linestyle='--', linewidth=1.6, zorder=4,
            label='Significance (t=2.0)')

# Significance label
ax3.text(len(horizons) - 0.44, 2.0 + 0.25, 'p<0.05',
         color=RED, fontsize=7.5, va='bottom', ha='right', style='italic')

for rect, val in zip(bars_t, t_stat):
    ax3.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.2,
             f'{val:.2f}', ha='center', va='bottom', fontsize=8.5,
             color=GREEN_LITE, fontweight='bold')

ax3.set_ylim(0, max(t_stat) * 1.28)
ax3.legend(loc='upper right', fontsize=8, facecolor=BG_PANEL,
           edgecolor=SPINE_CLR, labelcolor=TEXT_DIM)

# ── Panel 4 – Avg Move & Hit Rate ─────────────────────────────────────────────
ax4 = fig.add_subplot(gs[1, 1])
style_ax(ax4, 'Avg Move (ticks) & Hit Rate', ylabel_left='Average Move (ticks)')

bar_move = ax4.bar(x_pos - BAR_W/4, avg_move, width=BAR_W/2 + 0.05,
                    color=BLUE, alpha=0.88, label='Avg Move', zorder=3,
                    edgecolor='none')

ax4_r = ax4.twinx()
ax4_r.set_facecolor('none')
ax4_r.spines['right'].set_edgecolor(ORANGE)
ax4_r.spines['right'].set_linewidth(0.8)
for s in ['top', 'left', 'bottom']:
    ax4_r.spines[s].set_visible(False)
ax4_r.tick_params(colors=ORANGE, labelsize=9, length=3, width=0.7)

bar_hr = ax4_r.bar(x_pos + BAR_W/4, hit_rate, width=BAR_W/2 + 0.05,
                    color=ORANGE, alpha=0.78, label='Hit Rate %', zorder=3,
                    edgecolor='none')
ax4_r.set_ylabel('Hit Rate (%)', color=ORANGE, fontsize=9)
ax4_r.tick_params(axis='y', colors=ORANGE)

for rect, val in zip(bar_move, avg_move):
    ax4.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.02,
             f'{val:.2f}', ha='center', va='bottom', fontsize=7.5,
             color=BLUE, fontweight='bold')
for rect, val in zip(bar_hr, hit_rate):
    ax4_r.text(rect.get_x() + rect.get_width()/2, rect.get_height() + 0.3,
               f'{val:.1f}%', ha='center', va='bottom', fontsize=7.5,
               color=ORANGE, fontweight='bold')

ax4.set_ylim(0, max(avg_move) * 1.40)
ax4_r.set_ylim(0, max(hit_rate) * 1.40)

leg_patches4 = [
    mpatches.Patch(color=BLUE,   alpha=0.88, label='Avg Move (ticks)'),
    mpatches.Patch(color=ORANGE, alpha=0.78, label='Hit Rate (%)'),
]
ax4.legend(handles=leg_patches4, loc='upper left', fontsize=8,
           facecolor=BG_PANEL, edgecolor=SPINE_CLR, labelcolor=TEXT_DIM)

# ── Watermark / footer ────────────────────────────────────────────────────────
fig.text(
    0.5, 0.013,
    'Lvl3Quant | ES (MES) Futures | CME Globex | Feb 2026',
    ha='center', va='bottom',
    fontsize=8, color=TEXT_MUTED, style='italic'
)

# ── Save ──────────────────────────────────────────────────────────────────────
OUT_PATH = r'C:\Users\Footb\Documents\Github\Lvl3Quant\alpha_discovery\results\phase1_results_chart.png'
os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)

fig.savefig(
    OUT_PATH,
    dpi=180,
    facecolor=BG_DARK,
    bbox_inches='tight',
    pad_inches=0.18,
)
plt.close(fig)

print(f'Chart saved -> {OUT_PATH}')
