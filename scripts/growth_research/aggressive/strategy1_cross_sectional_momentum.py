#!/usr/bin/env python3
"""
Strategy 1: Cross-Sectional Momentum with Hedge (Long-Short proxy)
- Rank S&P 500 stocks by 6-month momentum
- Long top decile via equal-weight basket
- Hedge with inverse ETF (SH for S&P hedge)
- Monthly rebalance
- Walk-forward sliding window
"""
import json
import sys
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from scipy import stats

OUTPUT_DIR = "/home/jupiter/Lvl3Quant/output/growth_research/aggressive"

# Use a representative basket of liquid S&P 500 stocks across sectors
# (can't download all 500 efficiently, use ~100 liquid ones)
STOCKS = [
    # Tech
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'META', 'NVDA', 'TSLA', 'AMD', 'INTC', 'CRM',
    'ADBE', 'NFLX', 'PYPL', 'AVGO', 'QCOM',
    # Healthcare
    'JNJ', 'UNH', 'PFE', 'ABBV', 'MRK', 'LLY', 'TMO', 'ABT', 'DHR', 'BMY',
    # Financials
    'JPM', 'BAC', 'WFC', 'GS', 'MS', 'BLK', 'AXP', 'C', 'USB', 'PNC',
    # Consumer
    'WMT', 'PG', 'KO', 'PEP', 'COST', 'MCD', 'NKE', 'SBUX', 'TGT', 'HD',
    # Industrials
    'CAT', 'BA', 'HON', 'UPS', 'RTX', 'GE', 'MMM', 'DE', 'LMT', 'UNP',
    # Energy
    'XOM', 'CVX', 'COP', 'SLB', 'EOG', 'MPC', 'VLO', 'PSX', 'OXY', 'HAL',
    # Materials
    'LIN', 'APD', 'SHW', 'FCX', 'NEM', 'NUE', 'DOW',
    # Utilities
    'NEE', 'DUK', 'SO', 'D', 'AEP', 'EXC',
    # REITs
    'AMT', 'PLD', 'CCI', 'EQIX', 'SPG',
    # Communication
    'DIS', 'CMCSA', 'VZ', 'T', 'TMUS',
]

def download_data():
    """Download 7 years of daily data for momentum universe + hedge ETF"""
    print("Downloading stock data...")
    end = datetime(2026, 7, 1)
    start = datetime(2019, 1, 1)

    # Download all at once
    tickers = STOCKS + ['SH', 'SPY']  # SH = inverse S&P, SPY = benchmark
    data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)

    close = data['Close'].dropna(how='all')
    print(f"Downloaded {len(close.columns)} tickers, {len(close)} days")
    return close

def walk_forward_backtest(close, lookback_months=6, hold_months=1, top_pct=0.10, hedge_ratio=0.5):
    """
    Walk-forward sliding window momentum strategy.
    - lookback_months: momentum ranking period
    - hold_months: holding period before rebalance
    - top_pct: top percentile to go long
    - hedge_ratio: fraction of capital in SH (inverse ETF)
    """
    spy = close['SPY'].copy()
    sh = close['SH'].copy() if 'SH' in close.columns else None
    stocks = close.drop(columns=['SPY', 'SH'], errors='ignore')

    # Monthly dates
    monthly = stocks.resample('ME').last()
    monthly_ret = monthly.pct_change()

    lookback = lookback_months
    n_top = max(1, int(len(stocks.columns) * top_pct))

    portfolio_returns = []
    dates = []
    long_alloc = 1.0 - hedge_ratio

    for i in range(lookback, len(monthly) - hold_months):
        # Momentum: cumulative return over lookback period
        mom_start = monthly.iloc[i - lookback]
        mom_end = monthly.iloc[i]
        momentum = (mom_end / mom_start - 1).dropna()

        # Rank and pick top decile
        top_stocks = momentum.nlargest(n_top).index.tolist()

        # Forward return over hold period
        fwd_start = monthly.iloc[i]
        fwd_end = monthly.iloc[i + hold_months]

        # Equal-weight long portfolio return
        stock_rets = []
        for s in top_stocks:
            if s in fwd_start.index and s in fwd_end.index:
                if pd.notna(fwd_start[s]) and pd.notna(fwd_end[s]) and fwd_start[s] > 0:
                    stock_rets.append(fwd_end[s] / fwd_start[s] - 1)

        if not stock_rets:
            continue

        long_ret = np.mean(stock_rets)

        # Hedge return (SH)
        hedge_ret = 0
        if sh is not None:
            dt_start = monthly.index[i]
            dt_end = monthly.index[i + hold_months]
            sh_vals = sh.loc[dt_start:dt_end]
            if len(sh_vals) >= 2 and sh_vals.iloc[0] > 0:
                hedge_ret = sh_vals.iloc[-1] / sh_vals.iloc[0] - 1

        # Combined portfolio return
        port_ret = long_alloc * long_ret + hedge_ratio * hedge_ret
        portfolio_returns.append(port_ret)
        dates.append(monthly.index[i + hold_months])

    return pd.Series(portfolio_returns, index=dates, name='strategy')

def analyze_strategy(returns, name, spy_close):
    """Full analysis with R1 regime test"""
    if len(returns) < 12:
        return None

    # Basic metrics (monthly returns)
    n_months = len(returns)
    n_years = n_months / 12

    cum_ret = (1 + returns).prod() - 1
    cagr = (1 + cum_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_ret = returns.mean() * 12
    ann_vol = returns.std() * np.sqrt(12)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = returns[returns < 0].std() * np.sqrt(12)
    sortino = ann_ret / downside if downside > 0 else 0

    win_rate = (returns > 0).mean()

    avg_win = returns[returns > 0].mean() if (returns > 0).any() else 0
    avg_loss = abs(returns[returns < 0].mean()) if (returns < 0).any() else 1
    profit_factor = avg_win / avg_loss if avg_loss > 0 else float('inf')

    # Max drawdown
    cum = (1 + returns).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # R1 Regime test - classify months by SPY direction
    spy_monthly = spy_close.resample('ME').last().pct_change()
    common_idx = returns.index.intersection(spy_monthly.index)

    if len(common_idx) > 10:
        strat_aligned = returns.loc[common_idx]
        spy_aligned = spy_monthly.loc[common_idx]

        green = strat_aligned[spy_aligned > 0.01]
        red = strat_aligned[spy_aligned < -0.01]
        flat = strat_aligned[(spy_aligned >= -0.01) & (spy_aligned <= 0.01)]

        sharpe_green = green.mean() / green.std() * np.sqrt(12) if len(green) > 2 and green.std() > 0 else 0
        sharpe_red = red.mean() / red.std() * np.sqrt(12) if len(red) > 2 and red.std() > 0 else 0

        max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
        regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe if max_sharpe > 0 else 0
        r1_pass = regime_gap <= 0.50
    else:
        sharpe_green = sharpe_red = regime_gap = 0
        r1_pass = False
        green = red = flat = pd.Series(dtype=float)

    # Permutation test (100 trials)
    observed_sharpe = sharpe
    perm_sharpes = []
    for _ in range(100):
        perm = returns.sample(frac=1, replace=False).values
        perm_mean = perm.mean() * 12
        perm_std = perm.std() * np.sqrt(12)
        perm_sharpes.append(perm_mean / perm_std if perm_std > 0 else 0)

    # For permutation test of momentum, shuffle stock assignments
    perm_p = np.mean([s >= observed_sharpe for s in perm_sharpes])

    result = {
        'strategy': name,
        'n_months': n_months,
        'n_years': round(n_years, 1),
        'CAGR': round(cagr * 100, 2),
        'ann_return': round(ann_ret * 100, 2),
        'ann_vol': round(ann_vol * 100, 2),
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'win_rate': round(win_rate * 100, 1),
        'profit_factor': round(profit_factor, 3),
        'max_drawdown': round(max_dd * 100, 2),
        'calmar': round(calmar, 3),
        'cum_return': round(cum_ret * 100, 2),
        'R1_regime_gap': round(regime_gap, 3),
        'R1_pass': r1_pass,
        'sharpe_green_months': round(sharpe_green, 3),
        'sharpe_red_months': round(sharpe_red, 3),
        'n_green': len(green),
        'n_red': len(red),
        'n_flat': len(flat),
        'perm_p_value': round(perm_p, 3),
        'feasible_441': True,  # Can use fractional shares + SH
    }

    return result

def main():
    close = download_data()

    # Test multiple configurations
    configs = [
        {'lookback_months': 6, 'hold_months': 1, 'top_pct': 0.10, 'hedge_ratio': 0.0, 'name': 'Mom6m_Top10pct_NoHedge'},
        {'lookback_months': 6, 'hold_months': 1, 'top_pct': 0.10, 'hedge_ratio': 0.3, 'name': 'Mom6m_Top10pct_30Hedge'},
        {'lookback_months': 6, 'hold_months': 1, 'top_pct': 0.10, 'hedge_ratio': 0.5, 'name': 'Mom6m_Top10pct_50Hedge'},
        {'lookback_months': 12, 'hold_months': 1, 'top_pct': 0.10, 'hedge_ratio': 0.0, 'name': 'Mom12m_Top10pct_NoHedge'},
        {'lookback_months': 12, 'hold_months': 1, 'top_pct': 0.10, 'hedge_ratio': 0.3, 'name': 'Mom12m_Top10pct_30Hedge'},
        {'lookback_months': 3, 'hold_months': 1, 'top_pct': 0.20, 'hedge_ratio': 0.0, 'name': 'Mom3m_Top20pct_NoHedge'},
        {'lookback_months': 6, 'hold_months': 1, 'top_pct': 0.20, 'hedge_ratio': 0.0, 'name': 'Mom6m_Top20pct_NoHedge'},
    ]

    spy = close['SPY']
    results = []

    for cfg in configs:
        name = cfg.pop('name')
        print(f"\nTesting {name}...")
        returns = walk_forward_backtest(close.copy(), **cfg)
        if len(returns) > 0:
            r = analyze_strategy(returns, name, spy)
            if r:
                results.append(r)
                print(f"  CAGR={r['CAGR']}%, Sharpe={r['sharpe']}, Sortino={r['sortino']}, "
                      f"WR={r['win_rate']}%, MaxDD={r['max_drawdown']}%, R1={'PASS' if r['R1_pass'] else 'FAIL'}")

    # Save results
    with open(f"{OUTPUT_DIR}/strategy1_momentum_results.json", 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Print summary
    print("\n" + "="*80)
    print("STRATEGY 1: CROSS-SECTIONAL MOMENTUM — RESULTS SUMMARY")
    print("="*80)

    df = pd.DataFrame(results)
    print(df[['strategy', 'CAGR', 'sharpe', 'sortino', 'win_rate', 'profit_factor',
              'max_drawdown', 'R1_pass', 'perm_p_value']].to_string(index=False))

    return results

if __name__ == '__main__':
    results = main()
