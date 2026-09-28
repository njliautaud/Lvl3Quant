"""
Lane 5: Cross-Asset Momentum
==============================
Strategy:
- Assets: SPY, TLT, GLD, USO, VNQ (equities, bonds, gold, oil, REITs)
- NO CRYPTO (HC #697)
- Time-series momentum (each asset's own trend) + cross-sectional (relative ranking)
- Risk parity weighting
- Walk-forward: train lookback parameters on 252d, test 21d OOT

This is a well-documented academic strategy (Moskowitz, Ooi, Pedersen 2012).
The question is: does it work walk-forward with realistic execution?

Survivorship bias: LOW — using broad ETFs with long histories.
Commission: $0 on Robinhood/IBKR (HC #694).
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r3'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Asset universe
ASSETS = {
    'SPY': 'US Equities',
    'TLT': 'US Long Bonds',
    'GLD': 'Gold',
    'USO': 'Oil',
    'VNQ': 'REITs',
}

print("Downloading data for Cross-Asset Momentum strategy...")
data = {}
for ticker, desc in ASSETS.items():
    try:
        df = yf.download(ticker, start='2010-01-01', end='2026-07-11', progress=False)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        data[ticker] = df
        print(f"  {ticker} ({desc}): {len(df)} days")
    except Exception as e:
        print(f"  {ticker} FAILED: {e}")

# Build aligned price matrix
closes = pd.DataFrame({t: data[t]['Close'] for t in data}).dropna()
returns = closes.pct_change().dropna()
print(f"  Aligned: {len(closes)} days, {closes.shape[1]} assets")

# --- Strategy Components ---

def time_series_momentum(returns, lookback):
    """Time-series momentum: go long if past return > 0, else cash."""
    cum_ret = returns.rolling(lookback).sum()
    signal = (cum_ret > 0).astype(float)
    return signal

def cross_sectional_momentum(returns, lookback):
    """Cross-sectional: rank assets by past return, overweight top, underweight bottom."""
    cum_ret = returns.rolling(lookback).sum()
    # Rank each day: 1 = best momentum, N = worst
    ranks = cum_ret.rank(axis=1, ascending=False)
    n_assets = ranks.shape[1]
    # Convert to weights: top assets get more, bottom get less
    # Weight = (N+1-rank) / sum(N+1-rank)
    inv_rank = n_assets + 1 - ranks
    weights = inv_rank.div(inv_rank.sum(axis=1), axis=0)
    return weights

def risk_parity_weights(returns, lookback=60):
    """Inverse-volatility weighting."""
    vol = returns.rolling(lookback).std()
    inv_vol = 1.0 / vol.clip(lower=0.001)
    weights = inv_vol.div(inv_vol.sum(axis=1), axis=0)
    return weights

def combined_strategy(returns, ts_lookback, xs_lookback, rp_lookback=60,
                      ts_weight=0.4, xs_weight=0.3, rp_weight=0.3):
    """
    Combined strategy:
    - Time-series momentum signals
    - Cross-sectional momentum weights
    - Risk parity base weights
    """
    ts_signal = time_series_momentum(returns, ts_lookback)
    xs_weights = cross_sectional_momentum(returns, xs_lookback)
    rp_weights = risk_parity_weights(returns, rp_lookback)

    # Combine: RP base * TS filter * XS tilt
    # TS signal is binary (0/1), so assets with negative momentum get 0
    # XS provides relative overweight
    combined = rp_weights * rp_weight + xs_weights * xs_weight
    # Apply TS filter
    combined = combined * (ts_signal * ts_weight + (1 - ts_weight))

    # Renormalize
    row_sums = combined.sum(axis=1)
    combined = combined.div(row_sums.replace(0, 1), axis=0)

    return combined

# --- Walk-Forward ---
TRAIN_DAYS = 252
TEST_DAYS = 21

# Parameter candidates
PARAM_SETS = [
    {'ts_lookback': 126, 'xs_lookback': 126, 'ts_weight': 0.4, 'xs_weight': 0.3},
    {'ts_lookback': 63, 'xs_lookback': 63, 'ts_weight': 0.4, 'xs_weight': 0.3},
    {'ts_lookback': 252, 'xs_lookback': 126, 'ts_weight': 0.5, 'xs_weight': 0.25},
    {'ts_lookback': 126, 'xs_lookback': 63, 'ts_weight': 0.3, 'xs_weight': 0.4},
    {'ts_lookback': 63, 'xs_lookback': 126, 'ts_weight': 0.5, 'xs_weight': 0.3},
    {'ts_lookback': 200, 'xs_lookback': 200, 'ts_weight': 0.4, 'xs_weight': 0.3},
]

print("\nRunning walk-forward cross-asset momentum...")

oot_returns_strat = []
oot_returns_eq_weight = []  # Equal-weight buy-and-hold benchmark
oot_returns_spy = []
oot_dates = []
window_results = []

start = max(TRAIN_DAYS + 252, 300)  # Need lookback
while start + TEST_DAYS <= len(returns):
    train_slice = slice(start - TRAIN_DAYS, start)

    # Train: find best params
    best_params = PARAM_SETS[0]
    best_sharpe = -999

    for ps in PARAM_SETS:
        weights = combined_strategy(
            returns.iloc[train_slice],
            ts_lookback=ps['ts_lookback'],
            xs_lookback=ps['xs_lookback'],
            ts_weight=ps['ts_weight'],
            xs_weight=ps['xs_weight'],
        )
        # Portfolio return
        port_ret = (weights.shift(1) * returns.iloc[train_slice]).sum(axis=1).dropna()
        if len(port_ret) > 20 and port_ret.std() > 0:
            s = port_ret.mean() / port_ret.std() * np.sqrt(252)
            if s > best_sharpe:
                best_sharpe = s
                best_params = ps

    # Test OOT
    test_end = min(start + TEST_DAYS, len(returns))
    # Compute weights using data up to start of test
    full_weights = combined_strategy(
        returns.iloc[:test_end],
        ts_lookback=best_params['ts_lookback'],
        xs_lookback=best_params['xs_lookback'],
        ts_weight=best_params['ts_weight'],
        xs_weight=best_params['xs_weight'],
    )

    for t in range(start, test_end):
        # Strategy return
        w = full_weights.iloc[t-1] if t-1 < len(full_weights) else pd.Series(1/len(ASSETS), index=returns.columns)
        strat_ret = (w * returns.iloc[t]).sum()
        oot_returns_strat.append(strat_ret)

        # Equal-weight benchmark
        eq_ret = returns.iloc[t].mean()
        oot_returns_eq_weight.append(eq_ret)

        # SPY
        spy_ret = returns['SPY'].iloc[t]
        oot_returns_spy.append(spy_ret)

        oot_dates.append(returns.index[t])

    window_results.append({
        'test_start': str(returns.index[start].date()),
        'best_ts_lookback': best_params['ts_lookback'],
        'best_xs_lookback': best_params['xs_lookback'],
        'train_sharpe': round(best_sharpe, 3),
    })

    start += TEST_DAYS

print(f"  Completed {len(window_results)} windows")

# --- Analysis ---
strat = pd.Series(oot_returns_strat, index=pd.DatetimeIndex(oot_dates))
eq_w = pd.Series(oot_returns_eq_weight, index=pd.DatetimeIndex(oot_dates))
spy_s = pd.Series(oot_returns_spy, index=pd.DatetimeIndex(oot_dates))

strat = strat.groupby(strat.index).mean().clip(-0.5, 0.5)
eq_w = eq_w.groupby(eq_w.index).mean().clip(-0.5, 0.5)
spy_s = spy_s.groupby(spy_s.index).mean().clip(-0.5, 0.5)

n_days = len(strat)

# Regime analysis
spy_full = data['SPY'].reindex(strat.index) if 'SPY' in data else pd.DataFrame()
if len(spy_full) > 0 and 'Open' in spy_full.columns:
    green = spy_full['Close'] > spy_full['Open']
    red = ~green
    g_rets = strat[green.reindex(strat.index, fill_value=False)]
    r_rets = strat[red.reindex(strat.index, fill_value=False)]
    sg = float(g_rets.mean() / g_rets.std() * np.sqrt(252)) if len(g_rets) > 10 and g_rets.std() > 0 else 0
    sr = float(r_rets.mean() / r_rets.std() * np.sqrt(252)) if len(r_rets) > 10 and r_rets.std() > 0 else 0
    regime_gap = abs(sg - sr) / max(abs(sg), abs(sr), 0.001)
else:
    sg = sr = regime_gap = 0

def calc_full_metrics(series, label):
    if len(series) == 0 or series.std() == 0:
        return {}
    ar = float(series.mean() * 252)
    av = float(series.std() * np.sqrt(252))
    sharpe = ar / av
    ds = series[series < 0]
    dv = float(ds.std() * np.sqrt(252)) if len(ds) > 0 else av
    sortino = ar / dv if dv > 0 else 0
    cum = (1 + series).cumprod()
    tr = float(cum.iloc[-1])
    yrs = len(series) / 252
    cagr = (tr ** (1/yrs) - 1) if yrs > 0 and tr > 0 else 0
    mdd = max(float(((cum - cum.cummax()) / cum.cummax()).min()), -1.0)

    # Calmar ratio
    calmar = cagr / abs(mdd) if abs(mdd) > 0.001 else 0

    # Win rate
    wr = float((series > 0).mean() * 100)

    # Profit factor
    gross_profit = series[series > 0].sum()
    gross_loss = abs(series[series < 0].sum())
    pf = float(gross_profit / gross_loss) if gross_loss > 0 else 0

    return {
        f'{label}_ann_ret_pct': round(ar * 100, 2),
        f'{label}_ann_vol_pct': round(av * 100, 2),
        f'{label}_sharpe': round(sharpe, 3),
        f'{label}_sortino': round(sortino, 3),
        f'{label}_cagr_pct': round(cagr * 100, 2),
        f'{label}_max_dd_pct': round(mdd * 100, 2),
        f'{label}_calmar': round(calmar, 3),
        f'{label}_win_rate_pct': round(wr, 1),
        f'{label}_profit_factor': round(pf, 3),
    }

strat_m = calc_full_metrics(strat, 'strategy')
eq_m = calc_full_metrics(eq_w, 'equal_weight')
spy_m = calc_full_metrics(spy_s, 'spy')

# Per-asset contribution analysis
print("\nPer-asset momentum analysis:")
for asset in ASSETS:
    asset_ret = returns[asset].loc[strat.index]
    if len(asset_ret) > 0:
        ts_mom = time_series_momentum(returns[asset], 126)
        ts_aligned = ts_mom.reindex(strat.index).fillna(0)
        pct_positive_mom = float(ts_aligned.mean() * 100)
        ann_ret = float(asset_ret.mean() * 252 * 100)
        print(f"  {asset}: ann_ret={ann_ret:.1f}%, pct_time_positive_mom={pct_positive_mom:.0f}%")

summary = {
    'strategy': 'Lane 5: Cross-Asset Momentum',
    'assets': list(ASSETS.keys()),
    'asset_descriptions': ASSETS,
    'oot_days': n_days,
    'oot_years': round(n_days/252, 1),
    **strat_m,
    **eq_m,
    **spy_m,
    'regime_sharpe_green': round(sg, 3),
    'regime_sharpe_red': round(sr, 3),
    'regime_gap': round(regime_gap, 3),
    'regime_test_pass': regime_gap < 0.50,
    'value_added_vs_equal_weight': round(
        strat_m.get('strategy_sharpe', 0) - eq_m.get('equal_weight_sharpe', 0), 3
    ),
    'value_added_vs_spy': round(
        strat_m.get('strategy_sharpe', 0) - spy_m.get('spy_sharpe', 0), 3
    ),
    'walk_forward_windows': len(window_results),
    'survivorship_bias': 'LOW - broad ETFs with long histories',
    'commission': '$0 (HC #694)',
    'methodology': 'Time-series + cross-sectional momentum with risk-parity base weights',
}

print("\n" + "="*60)
print("LANE 5: CROSS-ASSET MOMENTUM - RESULTS")
print("="*60)
for k, v in summary.items():
    print(f"  {k}: {v}")

with open(os.path.join(OUTPUT_DIR, 'lane5_cross_asset_momentum_results.json'), 'w') as f:
    json.dump(summary, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/lane5_cross_asset_momentum_results.json")
