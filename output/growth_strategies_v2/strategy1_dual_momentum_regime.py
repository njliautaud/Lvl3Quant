"""
Strategy 1: Dual Momentum + Regime Filter
==========================================
Combine absolute momentum (is asset trending up?) with relative momentum
(is it outperforming alternatives?) with a VIX regime filter.

Universe: SPY/QQQ/IWM/EFA/EEM (equities), TLT/IEF/SHY (safety), GLD (alt)
Rules:
- Monthly rebalance
- Absolute momentum: 12M return > 0 (trending up)
- Relative momentum: rank by 12M return, pick top 2 from equity universe
- Regime: VIX < 25 = risk-on (equities), VIX >= 25 = risk-off (TLT/IEF/SHY)
- If risk-on but no equity passes absolute momentum => safety assets
- Equal weight selected assets
- T-1 signals only (use prior month-end data)
- 10 bps transaction costs for ETFs
"""

import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_strategies_v2'

# ── Data Download ──
EQUITY_UNIVERSE = ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM']
SAFETY_UNIVERSE = ['TLT', 'IEF', 'SHY']
ALT_UNIVERSE = ['GLD']
ALL_TICKERS = EQUITY_UNIVERSE + SAFETY_UNIVERSE + ALT_UNIVERSE + ['^VIX']

print("Downloading data...")
data = yf.download(ALL_TICKERS, start='2009-01-01', end='2026-07-18', auto_adjust=True)
prices = data['Close'].copy()
prices.columns = [c if c != '^VIX' else 'VIX' for c in prices.columns]
prices = prices.ffill().dropna()

# Monthly resample - use last trading day of each month
monthly = prices.resample('ME').last()
monthly_ret = monthly.pct_change()

# ── Strategy Logic ──
LOOKBACK = 12  # months for momentum
VIX_THRESHOLD = 25
TOP_N = 2
COST_BPS = 10  # basis points per trade

def run_strategy(monthly_prices, monthly_returns, vix_series, lag=1):
    """Run dual momentum + regime strategy.
    lag=1 means we use T-1 signals (no lookahead).
    lag=0 means contemporaneous (for lookahead test).
    """
    equity_tickers = EQUITY_UNIVERSE
    safety_tickers = SAFETY_UNIVERSE

    start_idx = LOOKBACK + lag
    dates = monthly_prices.index[start_idx:]

    weights_history = []
    returns_list = []
    turnover_list = []
    prev_weights = {}

    for i, date in enumerate(dates):
        signal_idx = monthly_prices.index.get_loc(date) - lag

        # 12-month momentum (absolute returns)
        mom_end_idx = signal_idx
        mom_start_idx = signal_idx - LOOKBACK
        if mom_start_idx < 0:
            continue

        mom_end_prices = monthly_prices.iloc[mom_end_idx]
        mom_start_prices = monthly_prices.iloc[mom_start_idx]
        mom_returns = (mom_end_prices / mom_start_prices) - 1

        # VIX regime
        vix_val = vix_series.iloc[signal_idx]
        risk_on = vix_val < VIX_THRESHOLD

        # Select assets
        if risk_on:
            # Filter equities with positive absolute momentum
            eq_mom = {t: mom_returns[t] for t in equity_tickers if t in mom_returns.index and mom_returns[t] > 0}

            if len(eq_mom) >= 1:
                # Rank by momentum, pick top N
                sorted_eq = sorted(eq_mom.items(), key=lambda x: x[1], reverse=True)
                selected = [t for t, _ in sorted_eq[:TOP_N]]
            else:
                # No equity has positive momentum => safety
                selected = safety_tickers[:2]
        else:
            # Risk-off: safety assets
            # Pick best momentum among safety + GLD
            safe_candidates = safety_tickers + ALT_UNIVERSE
            safe_mom = {t: mom_returns[t] for t in safe_candidates if t in mom_returns.index}
            sorted_safe = sorted(safe_mom.items(), key=lambda x: x[1], reverse=True)
            selected = [t for t, _ in sorted_safe[:TOP_N]]

        # Equal weight
        w = 1.0 / len(selected)
        new_weights = {t: w for t in selected}

        # Calculate turnover
        all_tickers_set = set(list(prev_weights.keys()) + list(new_weights.keys()))
        turnover = sum(abs(new_weights.get(t, 0) - prev_weights.get(t, 0)) for t in all_tickers_set)
        turnover_list.append(turnover)

        # Calculate return for this period (next month's return)
        ret_idx = monthly_prices.index.get_loc(date)
        if ret_idx >= len(monthly_returns):
            break

        period_ret = sum(new_weights.get(t, 0) * monthly_returns.iloc[ret_idx].get(t, 0)
                        for t in new_weights)

        # Subtract transaction costs
        cost = turnover * COST_BPS / 10000
        period_ret -= cost

        returns_list.append(period_ret)
        weights_history.append({'date': date, 'weights': new_weights, 'vix': vix_val, 'regime': 'risk_on' if risk_on else 'risk_off'})
        prev_weights = new_weights

    return pd.Series(returns_list, index=dates[:len(returns_list)]), weights_history, turnover_list


# ── Run Strategy ──
print("Running strategy with T-1 lag...")
strat_returns, weights_hist, turnover = run_strategy(monthly, monthly_ret, monthly['VIX'], lag=1)

print("Running strategy with T-0 lag (lookahead test)...")
strat_returns_t0, _, _ = run_strategy(monthly, monthly_ret, monthly['VIX'], lag=0)

# Benchmark: SPY buy and hold
spy_monthly_ret = monthly_ret['SPY']
spy_aligned = spy_monthly_ret.reindex(strat_returns.index).fillna(0)

# ── Performance Metrics ──
def calc_metrics(returns, name="Strategy"):
    """Calculate comprehensive performance metrics."""
    cum = (1 + returns).cumprod()
    total_ret = cum.iloc[-1] - 1
    n_years = len(returns) / 12
    cagr = (1 + total_ret) ** (1/n_years) - 1

    ann_vol = returns.std() * np.sqrt(12)
    sharpe = (returns.mean() * 12) / (returns.std() * np.sqrt(12)) if returns.std() > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(12)
    sortino = (returns.mean() * 12) / downside if downside > 0 else 0

    # Max drawdown
    cum_max = cum.cummax()
    drawdown = (cum - cum_max) / cum_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # Win rate and profit factor
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


metrics_t1 = calc_metrics(strat_returns, "DualMom+Regime (T-1)")
metrics_t0 = calc_metrics(strat_returns_t0, "DualMom+Regime (T-0)")
metrics_spy = calc_metrics(spy_aligned, "SPY B&H")

print("\n=== PERFORMANCE COMPARISON ===")
for m in [metrics_t1, metrics_t0, metrics_spy]:
    print(f"\n{m['name']}:")
    for k, v in m.items():
        if k != 'name':
            print(f"  {k}: {v}")

# Lag sensitivity
print(f"\n=== LAG SENSITIVITY TEST ===")
print(f"T-0 Sharpe: {metrics_t0['Sharpe']}")
print(f"T-1 Sharpe: {metrics_t1['Sharpe']}")
t0_sharpe = float(metrics_t0['Sharpe'])
t1_sharpe = float(metrics_t1['Sharpe'])
if t1_sharpe > 0:
    degradation = (t0_sharpe - t1_sharpe) / t0_sharpe * 100 if t0_sharpe > 0 else -999
    print(f"Degradation T-0 to T-1: {degradation:.1f}%")
    if degradation > 50:
        print("WARNING: >50% degradation suggests lookahead bias")
    else:
        print("PASS: Lag sensitivity acceptable")

# ── Correlation with SPY ──
corr = strat_returns.corr(spy_aligned)
print(f"\nSPY correlation: {corr:.3f}")

# ── Regime Analysis ──
spy_daily = prices['SPY'].resample('ME').last().pct_change()
green_months = spy_daily > 0
red_months = spy_daily <= 0

strat_green = strat_returns[strat_returns.index.isin(spy_daily[green_months].index)]
strat_red = strat_returns[strat_returns.index.isin(spy_daily[red_months].index)]

if len(strat_green) > 6 and len(strat_red) > 6:
    sharpe_green = strat_green.mean() / strat_green.std() * np.sqrt(12) if strat_green.std() > 0 else 0
    sharpe_red = strat_red.mean() / strat_red.std() * np.sqrt(12) if strat_red.std() > 0 else 0
    regime_gap = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red)) if max(abs(sharpe_green), abs(sharpe_red)) > 0 else 0

    print(f"\n=== REGIME TEST ===")
    print(f"Sharpe (green months): {sharpe_green:.3f}")
    print(f"Sharpe (red months): {sharpe_red:.3f}")
    print(f"Regime gap ratio: {regime_gap:.3f}")
    print(f"{'PASS' if regime_gap < 0.50 else 'FAIL'}: Regime gap {'<' if regime_gap < 0.50 else '>='} 0.50")

# ── Sub-period Stability ──
n = len(strat_returns)
quarters = [strat_returns.iloc[i*n//4:(i+1)*n//4] for i in range(4)]
quarter_sharpes = []
for i, q in enumerate(quarters):
    if len(q) > 3 and q.std() > 0:
        qs = q.mean() / q.std() * np.sqrt(12)
        quarter_sharpes.append(qs)
        print(f"Sub-period {i+1} Sharpe: {qs:.3f}")

if len(quarter_sharpes) >= 2:
    cv = np.std(quarter_sharpes) / abs(np.mean(quarter_sharpes)) if np.mean(quarter_sharpes) != 0 else 999
    print(f"\n=== SUB-PERIOD STABILITY ===")
    print(f"CV of Sharpe across sub-periods: {cv:.3f}")
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
        perm_sharpes.append(s.mean() / s.std() * np.sqrt(12))
    else:
        perm_sharpes.append(0)

p_value = np.mean([ps >= actual_sharpe for ps in perm_sharpes])
print(f"Actual Sharpe: {actual_sharpe:.3f}")
print(f"Permutation mean Sharpe: {np.mean(perm_sharpes):.3f}")
print(f"P-value: {p_value:.4f}")
print(f"{'PASS' if p_value < 0.05 else 'FAIL'}: p-value {'<' if p_value < 0.05 else '>='} 0.05")

# ── Save Results ──
results = {
    'strategy': 'Dual Momentum + Regime',
    'metrics_t1': metrics_t1,
    'metrics_t0': metrics_t0,
    'metrics_spy': metrics_spy,
    'spy_correlation': corr,
    'lag_degradation_pct': degradation if t0_sharpe > 0 else None,
    'permutation_p_value': p_value,
    'regime_gap': regime_gap if len(strat_green) > 6 and len(strat_red) > 6 else None,
    'subperiod_cv': cv if len(quarter_sharpes) >= 2 else None,
}

with open(os.path.join(OUT_DIR, 'strategy1_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

strat_returns.to_csv(os.path.join(OUT_DIR, 'strategy1_returns.csv'))
print(f"\nResults saved to {OUT_DIR}/strategy1_*")
print("\n=== STRATEGY 1 COMPLETE ===")
