"""
Drawdown & Risk Analysis — Meta Production v1 (Shorts, Top 50% Filter)

Reads concat_predictions.npz and results.json to reconstruct per-day P&L
with the meta top-50% filter applied, then computes:
  1. Max drawdown (ticks)
  2. Max consecutive red days
  3. Calmar ratio
  4. Recovery time from max drawdown
  5. Worst single day P&L
  6. Daily P&L distribution (skew, kurtosis)
  7. VaR/CVaR (95%, 99%)
  8. Equity curve statistics

NOTE: actuals in the NPZ are ALREADY net of 0.376-tick passive commission.
      No additional cost subtraction.
"""

import json
import numpy as np
import os
from scipy import stats
from datetime import datetime

# ── paths ──
BASE = "/home/jupiter/Lvl3Quant/output/meta_production_v1"
OUT  = "/home/jupiter/Lvl3Quant/output/drawdown_analysis_v1"

npz  = np.load(os.path.join(BASE, "concat_predictions.npz"))
with open(os.path.join(BASE, "results.json")) as f:
    results = json.load(f)

predictions = npz["predictions"]
actuals     = npz["actuals"]
per_fold    = results["per_fold"]

# ── reconstruct per-day P&L with top-50% filter ──
daily_pnl = []
daily_dates = []
daily_n_trades = []

offset = 0
for fold in per_fold:
    n = fold["n_test"]
    date_str = fold["date"]

    fold_preds   = predictions[offset : offset + n]
    fold_actuals = actuals[offset : offset + n]
    offset += n

    # Top 50% filter: select trades where prediction score is in the top 50%
    # For shorts model, more negative prediction = stronger short signal
    # But looking at data, predictions are mixed sign — use absolute magnitude
    # Actually, for a meta-model, higher prediction = higher expected P&L
    # Select top 50% by prediction score (highest predicted P&L)
    if n < 2:
        # Skip folds with too few trades
        continue

    threshold = np.median(fold_preds)
    mask = fold_preds >= threshold

    filtered_actuals = fold_actuals[mask]
    n_filtered = len(filtered_actuals)

    if n_filtered == 0:
        continue

    day_total_pnl = float(filtered_actuals.sum())
    day_mean_pnl  = float(filtered_actuals.mean())

    daily_pnl.append(day_total_pnl)
    daily_dates.append(date_str)
    daily_n_trades.append(n_filtered)

daily_pnl = np.array(daily_pnl)
daily_n_trades = np.array(daily_n_trades)
n_days = len(daily_pnl)

# ── 1. Max Drawdown ──
cumulative = np.cumsum(daily_pnl)
running_max = np.maximum.accumulate(cumulative)
drawdowns = cumulative - running_max  # negative values = drawdown
max_dd = float(drawdowns.min())
max_dd_idx = int(np.argmin(drawdowns))
# Find peak before max DD
peak_idx = int(np.argmax(cumulative[:max_dd_idx + 1]))

# ── 2. Max Consecutive Red Days ──
red_days = (daily_pnl < 0).astype(int)
max_consec_red = 0
current_streak = 0
for r in red_days:
    if r:
        current_streak += 1
        max_consec_red = max(max_consec_red, current_streak)
    else:
        current_streak = 0

# ── 3. Calmar Ratio ──
total_return = float(cumulative[-1])
annualized_return = total_return / n_days * 252  # annualize
calmar = annualized_return / abs(max_dd) if max_dd != 0 else float('inf')

# ── 4. Recovery Time from Max Drawdown ──
# How many days after the max DD trough to recover to the previous peak
peak_level = running_max[max_dd_idx]
recovered = False
recovery_days = None
for i in range(max_dd_idx + 1, n_days):
    if cumulative[i] >= peak_level:
        recovery_days = i - max_dd_idx
        recovered = True
        break
if not recovered:
    recovery_days = n_days - max_dd_idx  # still in drawdown

# ── 5. Worst Single Day P&L ──
worst_day_idx = int(np.argmin(daily_pnl))
worst_day_pnl = float(daily_pnl[worst_day_idx])
worst_day_date = daily_dates[worst_day_idx]
best_day_idx = int(np.argmax(daily_pnl))
best_day_pnl = float(daily_pnl[best_day_idx])
best_day_date = daily_dates[best_day_idx]

# ── 6. Daily P&L Distribution ──
pnl_mean = float(np.mean(daily_pnl))
pnl_std  = float(np.std(daily_pnl, ddof=1))
pnl_median = float(np.median(daily_pnl))
pnl_skew = float(stats.skew(daily_pnl))
pnl_kurt = float(stats.kurtosis(daily_pnl))  # excess kurtosis

# ── 7. VaR and CVaR ──
var_95 = float(np.percentile(daily_pnl, 5))   # 5th percentile = 95% VaR
var_99 = float(np.percentile(daily_pnl, 1))   # 1st percentile = 99% VaR
cvar_95 = float(daily_pnl[daily_pnl <= var_95].mean()) if np.any(daily_pnl <= var_95) else var_95
cvar_99 = float(daily_pnl[daily_pnl <= var_99].mean()) if np.any(daily_pnl <= var_99) else var_99

# ── 8. Equity Curve Statistics ──
green_days = int(np.sum(daily_pnl > 0))
red_days_count = int(np.sum(daily_pnl < 0))
flat_days = int(np.sum(daily_pnl == 0))
win_rate = green_days / n_days * 100

avg_win = float(daily_pnl[daily_pnl > 0].mean()) if green_days > 0 else 0
avg_loss = float(daily_pnl[daily_pnl < 0].mean()) if red_days_count > 0 else 0
profit_factor = abs(daily_pnl[daily_pnl > 0].sum() / daily_pnl[daily_pnl < 0].sum()) if red_days_count > 0 else float('inf')

# Daily Sharpe (annualized)
daily_sharpe = (pnl_mean / pnl_std) * np.sqrt(252) if pnl_std > 0 else 0
# Sortino
downside = daily_pnl[daily_pnl < 0]
downside_std = float(np.std(downside, ddof=1)) if len(downside) > 1 else pnl_std
daily_sortino = (pnl_mean / downside_std) * np.sqrt(252) if downside_std > 0 else 0

# Mean per-trade P&L (weighted by trades per day)
total_trades = int(daily_n_trades.sum())
total_ticks = float(daily_pnl.sum())
mean_per_trade = total_ticks / total_trades if total_trades > 0 else 0

# ── Print Report ──
report = []
def p(s=""):
    report.append(s)
    print(s)

p("=" * 70)
p("  DRAWDOWN & RISK ANALYSIS — Meta Production v1 (Top 50% Filter)")
p("=" * 70)
p()
p(f"  OOT Period: {daily_dates[0]} to {daily_dates[-1]}  ({n_days} trading days)")
p(f"  Total trades (filtered): {total_trades:,}")
p(f"  Avg trades/day: {total_trades/n_days:.0f}")
p()

p("─── EQUITY CURVE ───")
p(f"  Total P&L:            {total_ticks:+.1f} ticks  (${total_ticks * 12.50:+,.0f})")
p(f"  Mean daily P&L:       {pnl_mean:+.1f} ticks  (${pnl_mean * 12.50:+,.0f})")
p(f"  Median daily P&L:     {pnl_median:+.1f} ticks")
p(f"  Mean per-trade P&L:   {mean_per_trade:+.4f} ticks  (${mean_per_trade * 12.50:+.2f})")
p(f"  Std daily P&L:        {pnl_std:.1f} ticks")
p()

p("─── RISK METRICS ───")
p(f"  Max Drawdown:         {max_dd:.1f} ticks  (${max_dd * 12.50:,.0f})")
p(f"    Peak date:          {daily_dates[peak_idx]}")
p(f"    Trough date:        {daily_dates[max_dd_idx]}")
p(f"  Recovery:             {'Recovered in ' + str(recovery_days) + ' days' if recovered else 'NOT YET RECOVERED (' + str(recovery_days) + ' days so far)'}")
p(f"  Max Consec Red Days:  {max_consec_red}")
p()
p(f"  Worst Day:            {worst_day_pnl:+.1f} ticks on {worst_day_date}  ({daily_n_trades[worst_day_idx]} trades)")
p(f"  Best Day:             {best_day_pnl:+.1f} ticks on {best_day_date}  ({daily_n_trades[best_day_idx]} trades)")
p()

p("─── RATIOS ───")
p(f"  Calmar Ratio:         {calmar:.2f}  (annualized return / max DD)")
p(f"  Sharpe (ann):         {daily_sharpe:.2f}")
p(f"  Sortino (ann):        {daily_sortino:.2f}")
p(f"  Profit Factor:        {profit_factor:.2f}")
p(f"  Win Rate (days):      {win_rate:.1f}%  ({green_days}G / {red_days_count}R / {flat_days}F)")
p(f"  Avg Win Day:          {avg_win:+.1f} ticks")
p(f"  Avg Loss Day:         {avg_loss:+.1f} ticks")
p(f"  Win/Loss Ratio:       {abs(avg_win/avg_loss):.2f}" if avg_loss != 0 else "  Win/Loss Ratio:       N/A")
p()

p("─── DISTRIBUTION ───")
p(f"  Skewness:             {pnl_skew:+.3f}  ({'right-skewed' if pnl_skew > 0 else 'left-skewed'})")
p(f"  Excess Kurtosis:      {pnl_kurt:+.3f}  ({'fat tails' if pnl_kurt > 0 else 'thin tails'})")
p()

p("─── VALUE AT RISK (daily) ───")
p(f"  VaR 95%:              {var_95:+.1f} ticks  (${var_95 * 12.50:+,.0f})")
p(f"  VaR 99%:              {var_99:+.1f} ticks  (${var_99 * 12.50:+,.0f})")
p(f"  CVaR 95%:             {cvar_95:+.1f} ticks  (${cvar_95 * 12.50:+,.0f})")
p(f"  CVaR 99%:             {cvar_99:+.1f} ticks  (${cvar_99 * 12.50:+,.0f})")
p()

p("─── PER-DAY BREAKDOWN ───")
p(f"  {'Date':<12} {'Trades':>6} {'P&L':>10} {'Cum P&L':>10} {'DD':>10}")
p(f"  {'─'*12} {'─'*6} {'─'*10} {'─'*10} {'─'*10}")
for i in range(n_days):
    marker = " ***" if i == max_dd_idx else ""
    p(f"  {daily_dates[i]:<12} {daily_n_trades[i]:>6} {daily_pnl[i]:>+10.1f} {cumulative[i]:>+10.1f} {drawdowns[i]:>+10.1f}{marker}")

p()
p("=" * 70)

# ── Save outputs ──
# Save report
with open(os.path.join(OUT, "report.txt"), "w") as f:
    f.write("\n".join(report))

# Save JSON summary
summary = {
    "period": {"start": daily_dates[0], "end": daily_dates[-1], "n_days": n_days},
    "trades": {"total": total_trades, "avg_per_day": round(total_trades / n_days, 1)},
    "equity": {
        "total_pnl_ticks": round(total_ticks, 2),
        "total_pnl_usd": round(total_ticks * 12.50, 2),
        "mean_daily_pnl_ticks": round(pnl_mean, 2),
        "median_daily_pnl_ticks": round(pnl_median, 2),
        "std_daily_pnl_ticks": round(pnl_std, 2),
        "mean_per_trade_ticks": round(mean_per_trade, 4),
    },
    "risk": {
        "max_drawdown_ticks": round(max_dd, 2),
        "max_drawdown_usd": round(max_dd * 12.50, 2),
        "peak_date": daily_dates[peak_idx],
        "trough_date": daily_dates[max_dd_idx],
        "recovered": recovered,
        "recovery_days": recovery_days,
        "max_consecutive_red_days": max_consec_red,
        "worst_day_ticks": round(worst_day_pnl, 2),
        "worst_day_date": worst_day_date,
        "best_day_ticks": round(best_day_pnl, 2),
        "best_day_date": best_day_date,
    },
    "ratios": {
        "calmar": round(calmar, 2),
        "sharpe_annualized": round(daily_sharpe, 2),
        "sortino_annualized": round(daily_sortino, 2),
        "profit_factor": round(profit_factor, 2),
        "win_rate_pct": round(win_rate, 1),
        "avg_win_ticks": round(avg_win, 2),
        "avg_loss_ticks": round(avg_loss, 2),
    },
    "distribution": {
        "skewness": round(pnl_skew, 3),
        "excess_kurtosis": round(pnl_kurt, 3),
    },
    "var": {
        "var_95_ticks": round(var_95, 2),
        "var_99_ticks": round(var_99, 2),
        "cvar_95_ticks": round(cvar_95, 2),
        "cvar_99_ticks": round(cvar_99, 2),
    },
    "daily": [
        {"date": daily_dates[i], "n_trades": int(daily_n_trades[i]),
         "pnl_ticks": round(float(daily_pnl[i]), 2),
         "cum_pnl_ticks": round(float(cumulative[i]), 2),
         "drawdown_ticks": round(float(drawdowns[i]), 2)}
        for i in range(n_days)
    ],
    "timestamp": datetime.now().isoformat(),
}

with open(os.path.join(OUT, "summary.json"), "w") as f:
    json.dump(summary, f, indent=2)

# Save equity curve as NPZ
np.savez(os.path.join(OUT, "equity_curve.npz"),
         dates=np.array(daily_dates),
         daily_pnl=daily_pnl,
         cumulative_pnl=cumulative,
         drawdowns=drawdowns,
         n_trades=daily_n_trades)

print(f"\nOutputs saved to {OUT}/")
