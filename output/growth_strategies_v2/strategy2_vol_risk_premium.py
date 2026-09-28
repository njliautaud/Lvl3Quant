"""
Strategy 2: Volatility Risk Premium Harvesting
================================================
Systematically harvest the vol risk premium (IV > RV spread) on SPY.

Approach (ETF-based proxy since we don't have options data):
- Use VIX as implied vol, realized vol of SPY as RV
- When VIX/RV ratio > threshold => short vol (proxy: short VXX/VIXY + long SPY)
- When VIX/RV ratio < threshold => flat or long vol
- Size position by magnitude of vol premium
- T-1 signals only

Proxy instruments:
- Short vol position: Short VIXY (or inverse: long SVXY) + partial SPY
- We use VIXY for the vol instrument (available 2011+)
- Before VIXY: use VXX

Transaction costs: 10 bps per trade on ETFs
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_strategies_v2'

# ── Data Download ──
tickers = ['SPY', '^VIX']
print("Downloading data...")
data = yf.download(tickers, start='2009-01-01', end='2026-07-18', auto_adjust=True)
prices = data['Close'].copy()
prices.columns = ['SPY', 'VIX']
prices = prices.ffill().dropna()

# Also download SVXY (short vol ETF) as a check/alternative
try:
    svxy_data = yf.download('SVXY', start='2011-10-01', end='2026-07-18', auto_adjust=True)
    svxy_prices = svxy_data['Close']
    has_svxy = True
    print("SVXY data available for validation")
except:
    has_svxy = False
    print("SVXY data not available, using synthetic approach")

# ── Calculate Realized Vol ──
spy_daily_ret = prices['SPY'].pct_change()
# 21-day realized vol (annualized)
rv_21d = spy_daily_ret.rolling(21).std() * np.sqrt(252) * 100  # in % terms like VIX
# 5-day realized vol for short-term comparison
rv_5d = spy_daily_ret.rolling(5).std() * np.sqrt(252) * 100

# VIX/RV ratio
vix_rv_ratio = prices['VIX'] / rv_21d

# ── Strategy Logic ──
# Weekly rebalance for more granularity
# Resample to weekly
weekly_prices = prices.resample('W-FRI').last()
weekly_spy_ret = weekly_prices['SPY'].pct_change()
weekly_vix = weekly_prices['VIX']
weekly_rv = rv_21d.resample('W-FRI').last()
weekly_rv5 = rv_5d.resample('W-FRI').last()
weekly_ratio = weekly_vix / weekly_rv

COST_BPS = 10

def run_vol_premium_strategy(weekly_spy_ret, weekly_ratio, weekly_vix, lag=1):
    """
    Vol premium harvesting:
    - When VIX/RV > 1.2: premium is rich => harvest (long SPY, overweight)
    - When VIX/RV > 1.5: very rich => max harvest
    - When VIX/RV < 1.0: vol is cheap => reduce exposure (risk-off)
    - Position sizing: proportional to vol premium magnitude
    """
    start_idx = max(30, lag)  # need enough data for RV calc
    valid_idx = weekly_ratio.dropna().index
    dates = valid_idx[start_idx:]

    returns_list = []
    weights_list = []
    prev_weight = 0

    for date in dates:
        idx = valid_idx.get_loc(date)
        signal_idx = idx - lag

        if signal_idx < 0:
            continue

        ratio = weekly_ratio.iloc[signal_idx]
        vix_val = weekly_vix.iloc[signal_idx]

        if pd.isna(ratio) or pd.isna(vix_val):
            continue

        # Position sizing based on vol premium
        if ratio > 1.5 and vix_val < 35:
            # Very rich premium, moderate VIX => max harvest
            weight = 1.5  # levered long SPY (harvesting rich premium)
        elif ratio > 1.2 and vix_val < 30:
            # Rich premium => harvest
            weight = 1.2
        elif ratio > 1.0 and vix_val < 25:
            # Mild premium, low VIX => normal
            weight = 1.0
        elif ratio < 0.9 or vix_val > 35:
            # Vol is cheap or very high VIX => defensive
            weight = 0.3
        elif vix_val > 30:
            # High VIX but ratio ok => cautious
            weight = 0.5
        else:
            weight = 0.8

        # Get return for this period
        ret_loc = weekly_spy_ret.index.get_loc(date) if date in weekly_spy_ret.index else None
        if ret_loc is None:
            continue

        period_ret = weekly_spy_ret.iloc[ret_loc]
        if pd.isna(period_ret):
            continue

        # Strategy return = weight * SPY return
        strat_ret = weight * period_ret

        # Transaction cost from weight change
        turnover = abs(weight - prev_weight)
        cost = turnover * COST_BPS / 10000
        strat_ret -= cost

        returns_list.append(strat_ret)
        weights_list.append({'date': str(date), 'weight': weight, 'ratio': ratio, 'vix': vix_val})
        prev_weight = weight

    return pd.Series(returns_list, index=[pd.Timestamp(w['date']) for w in weights_list]), weights_list


# ── Run Strategy ──
print("Running vol premium strategy with T-1 lag...")
strat_returns, weights = run_vol_premium_strategy(weekly_spy_ret, weekly_ratio, weekly_vix, lag=1)

print("Running vol premium strategy with T-0 lag (lookahead test)...")
strat_returns_t0, _ = run_vol_premium_strategy(weekly_spy_ret, weekly_ratio, weekly_vix, lag=0)

# Benchmark: SPY buy and hold (weekly)
spy_aligned = weekly_spy_ret.reindex(strat_returns.index).fillna(0)

# ── Performance Metrics ──
def calc_metrics(returns, name="Strategy", periods_per_year=52):
    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    n_years = len(returns) / periods_per_year

    ann_ret = returns.mean() * periods_per_year
    ann_vol = returns.std() * np.sqrt(periods_per_year)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(periods_per_year)
    sortino = ann_ret / downside if downside > 0 else 0

    cagr = (1 + total_ret) ** (1/n_years) - 1 if n_years > 0 else 0

    cum_max = cum.cummax()
    drawdown = (cum - cum_max) / cum_max
    max_dd = drawdown.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    wr = len(wins) / len(returns) if len(returns) > 0 else 0
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    return {
        'name': name,
        'CAGR': f"{cagr:.4f}",
        'Sharpe': f"{sharpe:.3f}",
        'Sortino': f"{sortino:.3f}",
        'MaxDD': f"{max_dd:.4f}",
        'Calmar': f"{calmar:.3f}",
        'AnnVol': f"{ann_vol:.4f}",
        'WinRate': f"{wr:.3f}",
        'ProfitFactor': f"{pf:.3f}",
        'TotalReturn': f"{total_ret:.4f}",
        'N_periods': len(returns),
        'N_years': f"{n_years:.1f}",
    }


metrics_t1 = calc_metrics(strat_returns, "VolPremium (T-1)")
metrics_t0 = calc_metrics(strat_returns_t0, "VolPremium (T-0)")
metrics_spy = calc_metrics(spy_aligned, "SPY B&H")

print("\n=== PERFORMANCE COMPARISON ===")
for m in [metrics_t1, metrics_t0, metrics_spy]:
    print(f"\n{m['name']}:")
    for k, v in m.items():
        if k != 'name':
            print(f"  {k}: {v}")

# Lag sensitivity
t0_sharpe = float(metrics_t0['Sharpe'])
t1_sharpe = float(metrics_t1['Sharpe'])
if t0_sharpe > 0:
    degradation = (t0_sharpe - t1_sharpe) / t0_sharpe * 100
else:
    degradation = 0

print(f"\n=== LAG SENSITIVITY TEST ===")
print(f"T-0 Sharpe: {t0_sharpe:.3f}")
print(f"T-1 Sharpe: {t1_sharpe:.3f}")
print(f"Degradation: {degradation:.1f}%")
print(f"{'PASS' if abs(degradation) < 50 else 'FAIL'}: Lag sensitivity")

# SPY correlation
corr = strat_returns.corr(spy_aligned)
print(f"\nSPY correlation: {corr:.3f}")

# ── Regime Test ──
spy_weekly_direction = spy_aligned > 0
strat_green = strat_returns[spy_weekly_direction.reindex(strat_returns.index, fill_value=False)]
strat_red = strat_returns[~spy_weekly_direction.reindex(strat_returns.index, fill_value=True)]

regime_gap = None
if len(strat_green) > 20 and len(strat_red) > 20:
    sharpe_green = strat_green.mean() / strat_green.std() * np.sqrt(52) if strat_green.std() > 0 else 0
    sharpe_red = strat_red.mean() / strat_red.std() * np.sqrt(52) if strat_red.std() > 0 else 0
    regime_gap = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red)) if max(abs(sharpe_green), abs(sharpe_red)) > 0 else 0

    print(f"\n=== REGIME TEST ===")
    print(f"Sharpe (green weeks): {sharpe_green:.3f}")
    print(f"Sharpe (red weeks): {sharpe_red:.3f}")
    print(f"Regime gap ratio: {regime_gap:.3f}")
    print(f"{'PASS' if regime_gap < 0.50 else 'FAIL'}: Regime gap {'<' if regime_gap < 0.50 else '>='} 0.50")

# ── Sub-period Stability ──
n = len(strat_returns)
quarters = [strat_returns.iloc[i*n//4:(i+1)*n//4] for i in range(4)]
quarter_sharpes = []
for i, q in enumerate(quarters):
    if len(q) > 10 and q.std() > 0:
        qs = q.mean() / q.std() * np.sqrt(52)
        quarter_sharpes.append(qs)
        print(f"Sub-period {i+1} Sharpe: {qs:.3f}")

cv = None
if len(quarter_sharpes) >= 2 and np.mean(quarter_sharpes) != 0:
    cv = np.std(quarter_sharpes) / abs(np.mean(quarter_sharpes))
    print(f"\n=== SUB-PERIOD STABILITY ===")
    print(f"CV of Sharpe: {cv:.3f}")
    print(f"{'PASS' if cv < 0.70 else 'FAIL'}: CV {'<' if cv < 0.70 else '>='} 0.70")

# ── Permutation Test ──
print("\n=== PERMUTATION TEST (200 permutations) ===")
actual_sharpe = float(metrics_t1['Sharpe'])
n_perms = 200
perm_sharpes = []

np.random.seed(42)
returns_array = strat_returns.values.copy()
for _ in range(n_perms):
    shuffled = np.random.permutation(returns_array)
    s = pd.Series(shuffled)
    if s.std() > 0:
        perm_sharpes.append(s.mean() / s.std() * np.sqrt(52))
    else:
        perm_sharpes.append(0)

p_value = np.mean([ps >= actual_sharpe for ps in perm_sharpes])
print(f"Actual Sharpe: {actual_sharpe:.3f}")
print(f"Permutation mean Sharpe: {np.mean(perm_sharpes):.3f}")
print(f"P-value: {p_value:.4f}")
print(f"{'PASS' if p_value < 0.05 else 'FAIL'}: p-value {'<' if p_value < 0.05 else '>='} 0.05")

# ── Weight Distribution Analysis ──
weight_vals = [w['weight'] for w in weights]
print(f"\n=== WEIGHT DISTRIBUTION ===")
print(f"Mean weight: {np.mean(weight_vals):.2f}")
print(f"Median weight: {np.median(weight_vals):.2f}")
print(f"% time at max (1.5): {np.mean([w == 1.5 for w in weight_vals])*100:.1f}%")
print(f"% time defensive (<=0.5): {np.mean([w <= 0.5 for w in weight_vals])*100:.1f}%")

# ── Save Results ──
results = {
    'strategy': 'Volatility Risk Premium Harvesting',
    'metrics_t1': metrics_t1,
    'metrics_t0': metrics_t0,
    'metrics_spy': metrics_spy,
    'spy_correlation': float(corr),
    'lag_degradation_pct': degradation,
    'permutation_p_value': float(p_value),
    'regime_gap': float(regime_gap) if regime_gap is not None else None,
    'subperiod_cv': float(cv) if cv is not None else None,
}

with open(os.path.join(OUT_DIR, 'strategy2_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

strat_returns.to_csv(os.path.join(OUT_DIR, 'strategy2_returns.csv'))
print(f"\nResults saved to {OUT_DIR}/strategy2_*")
print("\n=== STRATEGY 2 COMPLETE ===")
