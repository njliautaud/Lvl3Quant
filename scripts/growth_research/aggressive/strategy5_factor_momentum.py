#!/usr/bin/env python3
"""
Strategy 5: Factor Momentum (Momentum OF Factors)
- Track which factor ETF is performing best over trailing 3-6 months
- Rotate into the winning factor
- Industry-agnostic by construction
- Walk-forward backtest with 10+ years of data
"""
import json
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime

OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_research/aggressive"

# Factor ETFs
FACTOR_ETFS = {
    'MTUM': 'Momentum',
    'QUAL': 'Quality',
    'VLUE': 'Value',
    'SIZE': 'Size (Small)',
    'USMV': 'Low Vol',
    'RPV': 'Deep Value',
    'SPHB': 'High Beta',
    'SPLV': 'Low Vol (S&P)',
}

# Additional broader ETFs for longer history
BROAD_ETFS = {
    'IWM': 'Small Cap',
    'IWF': 'Growth',
    'IWD': 'Value',
    'MDY': 'Mid Cap',
    'QQQ': 'Tech/Growth',
    'XLF': 'Financials',
    'XLE': 'Energy',
    'XLV': 'Healthcare',
    'XLK': 'Technology',
    'XLI': 'Industrials',
    'XLP': 'Consumer Staples',
    'XLY': 'Consumer Disc',
    'XLU': 'Utilities',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
}

def download_data():
    print("Downloading factor ETF data...")
    end = datetime(2026, 7, 1)
    start = datetime(2014, 1, 1)  # 12+ years

    all_tickers = list(FACTOR_ETFS.keys()) + list(BROAD_ETFS.keys()) + ['SPY']
    data = yf.download(all_tickers, start=start, end=end, auto_adjust=True, progress=False)
    close = data['Close'].dropna(how='all')

    print(f"Downloaded {len(close.columns)} ETFs, {len(close)} days")
    return close

def backtest_factor_rotation(close, etf_universe, lookback_months=6, hold_months=1,
                               top_n=1, use_leverage=False, name='default'):
    """
    Factor rotation strategy:
    - Rank ETFs by trailing lookback_months return
    - Hold top_n ETFs for hold_months
    - Monthly rebalance
    """
    # Filter to available ETFs
    available = [t for t in etf_universe if t in close.columns]
    if len(available) < 3:
        print(f"  Only {len(available)} ETFs available, skipping")
        return None

    prices = close[available]
    monthly = prices.resample('ME').last()
    monthly_ret = monthly.pct_change()

    portfolio_returns = []
    dates = []
    selected_etfs = []

    for i in range(lookback_months, len(monthly) - hold_months):
        # Rank by lookback return
        start_prices = monthly.iloc[i - lookback_months]
        end_prices = monthly.iloc[i]
        returns = (end_prices / start_prices - 1).dropna()

        if len(returns) < 3:
            continue

        # Pick top_n
        top = returns.nlargest(top_n).index.tolist()

        # Forward return
        fwd_start = monthly.iloc[i]
        fwd_end = monthly.iloc[i + hold_months]

        fwd_rets = []
        for etf in top:
            if pd.notna(fwd_start[etf]) and pd.notna(fwd_end[etf]) and fwd_start[etf] > 0:
                fwd_rets.append(fwd_end[etf] / fwd_start[etf] - 1)

        if fwd_rets:
            port_ret = np.mean(fwd_rets)
            if use_leverage:
                port_ret *= 2  # Simulated 2x leverage
            portfolio_returns.append(port_ret)
            dates.append(monthly.index[i + hold_months])
            selected_etfs.append(top)

    return pd.Series(portfolio_returns, index=dates), selected_etfs

def analyze(returns, name, spy_close):
    if returns is None or len(returns) < 12:
        return None

    n_months = len(returns)
    n_years = n_months / 12

    cum_ret = (1 + returns).prod() - 1
    cagr = (1 + cum_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_ret = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(12)
    sortino = ann_ret / downside if downside > 0 else 0

    wr = (returns > 0).mean()
    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
    pf = avg_win / avg_loss if avg_loss > 0 else float('inf')

    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # R1
    spy_monthly = spy_close.resample('ME').last().pct_change()
    common_idx = returns.index.intersection(spy_monthly.index)
    r1_pass = False
    sharpe_green = sharpe_red = regime_gap = 0
    n_green = n_red = 0

    if len(common_idx) > 10:
        sm = returns.loc[common_idx]
        spy_m = spy_monthly.loc[common_idx]
        green = sm[spy_m > 0.01]
        red = sm[spy_m < -0.01]
        n_green = len(green)
        n_red = len(red)
        sharpe_green = green.mean() / green.std() * np.sqrt(12) if len(green) > 2 and green.std() > 0 else 0
        sharpe_red = red.mean() / red.std() * np.sqrt(12) if len(red) > 2 and red.std() > 0 else 0
        max_s = max(abs(sharpe_green), abs(sharpe_red))
        regime_gap = abs(sharpe_green - sharpe_red) / max_s if max_s > 0 else 0
        r1_pass = regime_gap <= 0.50

    # Permutation test
    perm_sharpes = []
    for _ in range(100):
        perm = returns.sample(frac=1, replace=False).values
        pm = perm.mean() * 12
        ps = perm.std() * np.sqrt(12)
        perm_sharpes.append(pm / ps if ps > 0 else 0)
    perm_p = np.mean([s >= sharpe for s in perm_sharpes])

    final_441 = 441 * (1 + cum_ret)

    return {
        'strategy': name,
        'n_months': n_months,
        'n_years': round(n_years, 1),
        'CAGR': round(cagr * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(wr * 100, 1),
        'profit_factor': round(pf, 3),
        'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'cum_return': round(cum_ret * 100, 2),
        'final_441': round(final_441, 2),
        'R1_regime_gap': round(regime_gap, 3),
        'R1_pass': r1_pass,
        'sharpe_green': round(sharpe_green, 3),
        'sharpe_red': round(sharpe_red, 3),
        'n_green': n_green,
        'n_red': n_red,
        'perm_p_value': round(perm_p, 3),
        'feasible_441': True,
    }

def main():
    close = download_data()
    spy = close['SPY']

    configs = [
        # Factor ETFs only
        {'etf_universe': list(FACTOR_ETFS.keys()), 'lookback_months': 6, 'top_n': 1,
         'name': 'FactorMom_6m_Top1'},
        {'etf_universe': list(FACTOR_ETFS.keys()), 'lookback_months': 3, 'top_n': 1,
         'name': 'FactorMom_3m_Top1'},
        {'etf_universe': list(FACTOR_ETFS.keys()), 'lookback_months': 6, 'top_n': 2,
         'name': 'FactorMom_6m_Top2'},
        {'etf_universe': list(FACTOR_ETFS.keys()), 'lookback_months': 12, 'top_n': 1,
         'name': 'FactorMom_12m_Top1'},

        # With leverage
        {'etf_universe': list(FACTOR_ETFS.keys()), 'lookback_months': 6, 'top_n': 1,
         'use_leverage': True, 'name': 'FactorMom_6m_Top1_2xLev'},

        # Sector rotation (broader ETFs)
        {'etf_universe': list(BROAD_ETFS.keys()), 'lookback_months': 6, 'top_n': 1,
         'name': 'SectorRot_6m_Top1'},
        {'etf_universe': list(BROAD_ETFS.keys()), 'lookback_months': 3, 'top_n': 2,
         'name': 'SectorRot_3m_Top2'},
        {'etf_universe': list(BROAD_ETFS.keys()), 'lookback_months': 6, 'top_n': 1,
         'use_leverage': True, 'name': 'SectorRot_6m_Top1_2xLev'},

        # Combined factor + sector
        {'etf_universe': list(FACTOR_ETFS.keys()) + list(BROAD_ETFS.keys()),
         'lookback_months': 6, 'top_n': 1, 'name': 'Combined_6m_Top1'},
        {'etf_universe': list(FACTOR_ETFS.keys()) + list(BROAD_ETFS.keys()),
         'lookback_months': 3, 'top_n': 2, 'name': 'Combined_3m_Top2'},
    ]

    results = []
    for cfg in configs:
        name = cfg.pop('name')
        print(f"\nTesting {name}...")
        ret, selections = backtest_factor_rotation(close, **cfg, name=name)
        if ret is not None and len(ret) > 0:
            r = analyze(ret, name, spy)
            if r:
                results.append(r)
                print(f"  CAGR={r['CAGR']}%, Sharpe={r['sharpe']}, Sortino={r['sortino']}, "
                      f"WR={r['win_rate']}%, MaxDD={r['max_drawdown']}%, R1={'PASS' if r['R1_pass'] else 'FAIL'}")

    with open(f"{OUTPUT_DIR}/strategy5_factor_momentum_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print("\n" + "="*80)
    print("STRATEGY 5: FACTOR MOMENTUM — RESULTS")
    print("="*80)
    if results:
        df = pd.DataFrame(results)
        print(df[['strategy', 'CAGR', 'sharpe', 'sortino', 'win_rate',
                  'max_drawdown', 'R1_pass', 'final_441', 'perm_p_value']].to_string(index=False))

    return results

if __name__ == '__main__':
    results = main()
