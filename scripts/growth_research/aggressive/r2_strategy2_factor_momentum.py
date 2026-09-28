#!/usr/bin/env python3
"""
Round 2 Strategy 2: Factor Momentum (Rotate Between Factors)
Track which factor ETFs performed best trailing 3-6 months.
Monthly rotation into winning factor.

Key improvement over Round 1: add crash filter and test regime-adaptive variants.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import json
import warnings
import os
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/aggressive/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("ROUND 2 — STRATEGY 2: FACTOR MOMENTUM")
print("=" * 70)

# Factor ETFs
factor_etfs = {
    'MTUM': 'Momentum',
    'QUAL': 'Quality',
    'VLUE': 'Value',
    'USMV': 'Low Volatility',
    'SIZE': 'Small Cap',
}

# Add sector ETFs for broader universe
sector_etfs = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLI': 'Industrials',
    'XLY': 'Consumer Disc',
    'XLP': 'Consumer Staples',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLC': 'Communication',
}

# Defensive assets
defensive = {
    'SHY': 'Short Treasury',
    'TLT': 'Long Treasury',
    'GLD': 'Gold',
}

all_tickers = list(factor_etfs.keys()) + list(sector_etfs.keys()) + list(defensive.keys()) + ['SPY', 'QQQ']

print("\n[1/5] Downloading ETF data...")
data = yf.download(all_tickers, start='2013-01-01', end='2026-07-01', progress=False)
if isinstance(data.columns, pd.MultiIndex):
    prices = data['Close'] if 'Close' in data.columns.get_level_values(0) else data.xs('Close', level=0, axis=1)
else:
    prices = data[['Close']]

# Forward fill and drop completely empty columns
prices = prices.ffill().dropna(axis=1, how='all')
print(f"  Got data for {len(prices.columns)} ETFs")
print(f"  Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")

# Monthly prices
monthly = prices.resample('ME').last()
monthly_ret = monthly.pct_change()

# SPY for regime classification
spy_monthly_ret = monthly_ret['SPY'] if 'SPY' in monthly_ret.columns else None

# Regime map: green/red months based on SPY
regime_map = {}
if spy_monthly_ret is not None:
    for dt, ret in spy_monthly_ret.items():
        if pd.notna(ret):
            regime_map[dt.strftime('%Y-%m')] = 'green' if ret >= 0 else 'red'

print(f"  Regime: {sum(1 for v in regime_map.values() if v=='green')} green, {sum(1 for v in regime_map.values() if v=='red')} red months")


def factor_momentum_backtest(etf_list, lookback_months=6, top_n=1,
                              crash_filter=False, adaptive=False,
                              train_window=24, initial_capital=441):
    """
    Factor momentum strategy with walk-forward.
    - Rank ETFs by trailing lookback_months return
    - Buy top_n ETFs
    - Monthly rebalance
    - crash_filter: if SPY < 200d MA, go to SHY
    - adaptive: in red regime, pick from defensive assets instead
    """
    available = [t for t in etf_list if t in monthly_ret.columns]
    if len(available) < 3:
        return None, None

    # Walk-forward: start after lookback + train_window months
    start_idx = max(lookback_months, train_window) + 1
    months = monthly_ret.index[start_idx:]

    portfolio_returns = []

    for i, month in enumerate(months):
        month_idx = monthly_ret.index.get_loc(month)

        # Compute trailing returns for ranking
        lookback_rets = {}
        for etf in available:
            trail = monthly_ret[etf].iloc[month_idx - lookback_months:month_idx]
            if trail.notna().sum() >= lookback_months * 0.8:
                cum_ret = (1 + trail).prod() - 1
                lookback_rets[etf] = cum_ret

        if len(lookback_rets) < top_n:
            portfolio_returns.append({'date': month, 'return': 0, 'holdings': 'CASH'})
            continue

        # Crash filter: SPY below 200d MA
        if crash_filter and 'SPY' in prices.columns:
            spy_daily = prices['SPY']
            month_date = month
            recent_spy = spy_daily[spy_daily.index <= month_date]
            if len(recent_spy) >= 200:
                ma200 = recent_spy.iloc[-200:].mean()
                if recent_spy.iloc[-1] < ma200:
                    # Risk-off: go to SHY or cash
                    if 'SHY' in monthly_ret.columns:
                        ret = monthly_ret.loc[month, 'SHY'] if pd.notna(monthly_ret.loc[month, 'SHY']) else 0
                    else:
                        ret = 0
                    portfolio_returns.append({'date': month, 'return': ret, 'holdings': 'SHY(crash_filter)'})
                    continue

        # Adaptive: in red regime, switch to defensive universe
        if adaptive:
            ym = month.strftime('%Y-%m')
            # Look at trailing 2 month momentum of SPY
            spy_trail = monthly_ret['SPY'].iloc[month_idx-2:month_idx]
            if spy_trail.sum() < -0.03:  # SPY down >3% in last 2 months
                # Use defensive ETFs
                defensive_avail = [t for t in defensive.keys() if t in monthly_ret.columns]
                if defensive_avail:
                    def_rets = {}
                    for etf in defensive_avail:
                        trail = monthly_ret[etf].iloc[month_idx - lookback_months:month_idx]
                        if trail.notna().sum() >= lookback_months * 0.8:
                            def_rets[etf] = (1 + trail).prod() - 1
                    if def_rets:
                        ranked = sorted(def_rets.items(), key=lambda x: x[1], reverse=True)
                        picks = [r[0] for r in ranked[:top_n]]
                        ret = np.mean([monthly_ret.loc[month, p] for p in picks if pd.notna(monthly_ret.loc[month, p])])
                        portfolio_returns.append({'date': month, 'return': ret, 'holdings': '+'.join(picks)})
                        continue

        # Normal: pick top_n by trailing return
        ranked = sorted(lookback_rets.items(), key=lambda x: x[1], reverse=True)
        picks = [r[0] for r in ranked[:top_n]]

        # This month's return
        pick_rets = [monthly_ret.loc[month, p] for p in picks if pd.notna(monthly_ret.loc[month, p])]
        if pick_rets:
            ret = np.mean(pick_rets)
        else:
            ret = 0

        portfolio_returns.append({'date': month, 'return': ret, 'holdings': '+'.join(picks)})

    port_df = pd.DataFrame(portfolio_returns)
    port_df['date'] = pd.to_datetime(port_df['date'])
    port_df = port_df.set_index('date')

    return port_df, available


def compute_metrics_monthly(port_df, initial_capital=441):
    """Compute metrics from monthly returns."""
    rets = port_df['return'].values
    n_months = len(rets)
    n_years = n_months / 12

    equity = initial_capital * np.cumprod(1 + rets)

    # CAGR
    cagr = (equity[-1] / initial_capital) ** (1/n_years) - 1 if n_years > 0 else 0

    # Sharpe (annualized from monthly)
    sharpe = rets.mean() / rets.std() * np.sqrt(12) if rets.std() > 0 else 0

    # Sortino
    neg = rets[rets < 0]
    downside = neg.std() if len(neg) > 0 else rets.std()
    sortino = rets.mean() / downside * np.sqrt(12) if downside > 0 else 0

    # Win rate
    wr = np.sum(rets > 0) / len(rets) * 100

    # Profit factor
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets <= 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Max drawdown
    cum = np.cumprod(1 + rets)
    peak = np.maximum.accumulate(cum)
    dd = (cum - peak) / peak
    max_dd = dd.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'CAGR': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(wr, 1),
        'profit_factor': round(pf, 2),
        'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'n_months': n_months,
        'n_years': round(n_years, 1),
        'final_equity': round(equity[-1], 2),
        'cum_return': round((equity[-1] / initial_capital - 1) * 100, 2),
    }


def regime_test_monthly(port_df):
    """R1 regime test on monthly returns."""
    rets = port_df['return']
    green_rets = []
    red_rets = []

    for dt, ret in rets.items():
        ym = dt.strftime('%Y-%m')
        if ym in regime_map:
            if regime_map[ym] == 'green':
                green_rets.append(ret)
            else:
                red_rets.append(ret)

    green_rets = np.array(green_rets)
    red_rets = np.array(red_rets)

    if len(green_rets) < 5 or len(red_rets) < 5:
        return None, None, None, False

    sg = green_rets.mean() / green_rets.std() * np.sqrt(12) if green_rets.std() > 0 else 0
    sr = red_rets.mean() / red_rets.std() * np.sqrt(12) if red_rets.std() > 0 else 0

    max_abs = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / max_abs if max_abs > 0 else 0

    return sg, sr, gap, gap <= 0.50


def permutation_test_monthly(port_df, n_perms=100):
    """Permutation test on monthly returns."""
    rets = port_df['return'].values
    real_sharpe = rets.mean() / rets.std() * np.sqrt(12) if rets.std() > 0 else 0

    count = 0
    for _ in range(n_perms):
        shuf = np.random.permutation(rets)
        s = shuf.mean() / shuf.std() * np.sqrt(12) if shuf.std() > 0 else 0
        if s >= real_sharpe:
            count += 1

    return count / n_perms


# Test configurations
print("\n[2/5] Running factor momentum variants...")

configs = [
    # Pure factor momentum
    {'etfs': list(factor_etfs.keys()), 'lookback': 6, 'top_n': 1, 'crash': False, 'adaptive': False, 'name': 'Factor_6m_Top1'},
    {'etfs': list(factor_etfs.keys()), 'lookback': 3, 'top_n': 1, 'crash': False, 'adaptive': False, 'name': 'Factor_3m_Top1'},
    {'etfs': list(factor_etfs.keys()), 'lookback': 6, 'top_n': 2, 'crash': False, 'adaptive': False, 'name': 'Factor_6m_Top2'},

    # With crash filter
    {'etfs': list(factor_etfs.keys()), 'lookback': 6, 'top_n': 1, 'crash': True, 'adaptive': False, 'name': 'Factor_6m_Top1_CrashFilter'},
    {'etfs': list(factor_etfs.keys()), 'lookback': 3, 'top_n': 1, 'crash': True, 'adaptive': False, 'name': 'Factor_3m_Top1_CrashFilter'},

    # Adaptive (switch to defensive in downtrend)
    {'etfs': list(factor_etfs.keys()), 'lookback': 6, 'top_n': 1, 'crash': False, 'adaptive': True, 'name': 'Factor_6m_Top1_Adaptive'},
    {'etfs': list(factor_etfs.keys()), 'lookback': 6, 'top_n': 1, 'crash': True, 'adaptive': True, 'name': 'Factor_6m_Top1_CrashAdaptive'},

    # Sector rotation
    {'etfs': list(sector_etfs.keys()), 'lookback': 6, 'top_n': 1, 'crash': False, 'adaptive': False, 'name': 'Sector_6m_Top1'},
    {'etfs': list(sector_etfs.keys()), 'lookback': 6, 'top_n': 1, 'crash': True, 'adaptive': False, 'name': 'Sector_6m_Top1_CrashFilter'},
    {'etfs': list(sector_etfs.keys()), 'lookback': 6, 'top_n': 2, 'crash': True, 'adaptive': False, 'name': 'Sector_6m_Top2_CrashFilter'},

    # Combined (factors + sectors)
    {'etfs': list(factor_etfs.keys()) + list(sector_etfs.keys()), 'lookback': 6, 'top_n': 1, 'crash': True, 'adaptive': True, 'name': 'Combined_6m_Top1_Full'},
]

results = []
for cfg in configs:
    print(f"\n  Testing {cfg['name']}...")
    port_df, available = factor_momentum_backtest(
        cfg['etfs'], lookback_months=cfg['lookback'], top_n=cfg['top_n'],
        crash_filter=cfg['crash'], adaptive=cfg['adaptive']
    )

    if port_df is None or len(port_df) < 24:
        print(f"    Insufficient data")
        continue

    metrics = compute_metrics_monthly(port_df)
    sg, sr, gap, r1_pass = regime_test_monthly(port_df)
    perm_p = permutation_test_monthly(port_df)

    result = {
        'strategy': cfg['name'],
        **metrics,
        'sharpe_green': round(sg, 3) if sg is not None else None,
        'sharpe_red': round(sr, 3) if sr is not None else None,
        'R1_regime_gap': round(gap, 3) if gap is not None else None,
        'R1_pass': str(r1_pass),
        'perm_p_value': round(perm_p, 3),
        'perm_pass': str(perm_p < 0.05),
    }
    results.append(result)

    print(f"    Sharpe={metrics['sharpe']:.3f}, CAGR={metrics['CAGR']:.1f}%, WR={metrics['win_rate']:.1f}%")
    print(f"    MaxDD={metrics['max_drawdown']:.1f}%, Calmar={metrics['calmar']:.3f}")
    if gap is not None:
        print(f"    R1: gap={gap:.3f} ({'PASS' if r1_pass else 'FAIL'}), Green={sg:.3f}, Red={sr:.3f}")
    print(f"    Perm p={perm_p:.3f} ({'PASS' if perm_p < 0.05 else 'FAIL'})")

# Save results
print("\n[5/5] Saving results...")
with open(os.path.join(OUTPUT_DIR, 'r2_strategy2_factor_momentum.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {OUTPUT_DIR}r2_strategy2_factor_momentum.json")
print("\n" + "=" * 70)
print("FACTOR MOMENTUM SUMMARY")
print("=" * 70)
for r in results:
    flags = []
    if r['R1_pass'] == 'True': flags.append('R1-PASS')
    else: flags.append('R1-FAIL')
    if r['perm_pass'] == 'True': flags.append('PERM-PASS')
    else: flags.append('PERM-FAIL')
    print(f"  {r['strategy']}: Sharpe={r['sharpe']:.3f}, CAGR={r['CAGR']:.1f}%, "
          f"WR={r['win_rate']:.0f}%, MaxDD={r['max_drawdown']:.1f}%, "
          f"[{', '.join(flags)}]")
