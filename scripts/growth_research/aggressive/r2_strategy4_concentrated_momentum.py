#!/usr/bin/env python3
"""
Round 2 Strategy 4: Concentrated Momentum — Top 3 Stocks, Monthly Rebalance
Rank S&P 500 by 6-month momentum (skip most recent month to avoid reversal).
Buy top 3 stocks equally weighted. Monthly rebalance.
$441 can buy fractional shares on Robinhood.

Walk-forward sliding window backtest.
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
print("ROUND 2 — STRATEGY 4: CONCENTRATED MOMENTUM (TOP 3 STOCKS)")
print("=" * 70)

# Get S&P 500 tickers
print("\n[1/6] Getting S&P 500 constituents...")
try:
    sp500_table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')[0]
    tickers = sp500_table['Symbol'].str.replace('.', '-', regex=False).tolist()
except:
    tickers = ['AAPL','MSFT','AMZN','GOOGL','META','NVDA','TSLA','JPM','JNJ','V',
               'PG','UNH','HD','MA','DIS','BAC','ADBE','CRM','NFLX',
               'CMCSA','XOM','COST','TMO','ABT','PEP','AVGO','ACN','NKE','MRK',
               'LLY','WMT','DHR','TXN','PM','QCOM','LOW','UNP','LIN','NEE',
               'BMY','RTX','MDT','AMGN','HON','SBUX','IBM','CAT','GE','DE',
               'CVX','COP','MCD','INTC','AMD','ISRG','NOW','BKNG','SYK','GILD',
               'BLK','MDLZ','ADP','TJX','VRTX','REGN','ZTS','CI','MMC','PLD',
               'SCHW','CB','BDX','SO','DUK','ICE','NSC','AON','BSX','FIS',
               'CL','SHW','MCO','HUM','PNC','USB','TFC','MET','AIG','ALL',
               'ORCL','WBA','BIIB','KLAC','MCHP','SNPS','CDNS','FTNT','PANW','CRWD']

print(f"  Got {len(tickers)} tickers")

# Download all price data
print("\n[2/6] Downloading price data (5+ years)...")
all_prices = {}
batch_size = 50
for i in range(0, len(tickers), batch_size):
    batch = tickers[i:i+batch_size]
    try:
        data = yf.download(batch, start='2016-01-01', end='2026-07-01', progress=False)
        if isinstance(data.columns, pd.MultiIndex):
            close = data['Close'] if 'Close' in data.columns.get_level_values(0) else data.xs('Close', level=0, axis=1)
        else:
            close = data[['Close']]
        for t in batch:
            if t in close.columns:
                s = close[t].dropna()
                if len(s) > 252:
                    all_prices[t] = s
            elif len(batch) == 1:
                s = close.iloc[:, 0].dropna()
                if len(s) > 252:
                    all_prices[t] = s
    except:
        pass
    if (i // batch_size) % 2 == 0:
        print(f"  Downloaded {min(i+batch_size, len(tickers))}/{len(tickers)} tickers...")

print(f"  Got price data for {len(all_prices)} stocks")

# Build monthly returns matrix
print("\n[3/6] Building monthly returns matrix...")
price_df = pd.DataFrame(all_prices)
monthly_close = price_df.resample('ME').last()
monthly_ret = monthly_close.pct_change()

# SPY for regime and crash filter
spy = yf.download('SPY', start='2016-01-01', end='2026-07-01', progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy.columns = spy.columns.get_level_values(0)
spy_close = spy['Close']
spy_monthly = spy_close.resample('ME').last()
spy_monthly_ret = spy_monthly.pct_change()

regime_map = {}
for dt, ret in spy_monthly_ret.items():
    if pd.notna(ret):
        regime_map[dt.strftime('%Y-%m')] = 'green' if ret >= 0 else 'red'

print(f"  Monthly returns matrix: {monthly_ret.shape}")
print(f"  Date range: {monthly_ret.index[0].strftime('%Y-%m-%d')} to {monthly_ret.index[-1].strftime('%Y-%m-%d')}")


def concentrated_momentum_backtest(monthly_ret, spy_close,
                                    lookback_months=6, skip_recent=1,
                                    top_n=3, crash_filter=False,
                                    min_price=5.0, initial_capital=441):
    """
    Concentrated momentum strategy.
    - Rank stocks by trailing lookback_months return (skipping most recent skip_recent months)
    - Buy top_n stocks equally weighted
    - Monthly rebalance
    - crash_filter: if SPY < 200d MA, go to cash
    """
    months = monthly_ret.index
    start_idx = lookback_months + skip_recent + 1

    portfolio_returns = []

    for i in range(start_idx, len(months)):
        month = months[i]

        # Crash filter
        if crash_filter:
            recent_spy = spy_close[spy_close.index <= month]
            if len(recent_spy) >= 200:
                ma200 = recent_spy.iloc[-200:].mean()
                if recent_spy.iloc[-1] < ma200:
                    portfolio_returns.append({'date': month, 'return': 0, 'holdings': 'CASH', 'n_candidates': 0})
                    continue

        # Compute trailing momentum (skip most recent month)
        mom_start = i - lookback_months - skip_recent
        mom_end = i - skip_recent

        if mom_start < 0:
            continue

        trailing_rets = monthly_ret.iloc[mom_start:mom_end]

        # Cumulative return for each stock
        cum_rets = {}
        for col in trailing_rets.columns:
            vals = trailing_rets[col].dropna()
            if len(vals) >= lookback_months * 0.8:
                cum = (1 + vals).prod() - 1
                cum_rets[col] = cum

        if len(cum_rets) < top_n:
            portfolio_returns.append({'date': month, 'return': 0, 'holdings': 'CASH', 'n_candidates': len(cum_rets)})
            continue

        # Rank and pick top N
        ranked = sorted(cum_rets.items(), key=lambda x: x[1], reverse=True)
        picks = [r[0] for r in ranked[:top_n]]

        # This month's return (equal weight)
        pick_rets = []
        for p in picks:
            if p in monthly_ret.columns and pd.notna(monthly_ret.loc[month, p]):
                pick_rets.append(monthly_ret.loc[month, p])

        if pick_rets:
            ret = np.mean(pick_rets)
        else:
            ret = 0

        portfolio_returns.append({
            'date': month,
            'return': ret,
            'holdings': '+'.join(picks),
            'n_candidates': len(cum_rets)
        })

    return pd.DataFrame(portfolio_returns).set_index('date')


def compute_metrics_monthly(port_df, initial_capital=441):
    rets = port_df['return'].values
    n_months = len(rets)
    n_years = n_months / 12

    if n_years == 0 or np.std(rets) == 0:
        return None

    equity = initial_capital * np.cumprod(1 + rets)
    cagr = (equity[-1] / initial_capital) ** (1/n_years) - 1

    sharpe = rets.mean() / rets.std() * np.sqrt(12)
    neg = rets[rets < 0]
    downside = neg.std() if len(neg) > 0 else rets.std()
    sortino = rets.mean() / downside * np.sqrt(12) if downside > 0 else 0

    wr = np.sum(rets > 0) / len(rets) * 100
    gains = rets[rets > 0].sum()
    losses = abs(rets[rets <= 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

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
print("\n[4/6] Running concentrated momentum variants...")

configs = [
    # Classic momentum
    {'lookback': 6, 'skip': 1, 'top_n': 3, 'crash': False, 'name': 'Mom6m_Skip1_Top3'},
    {'lookback': 6, 'skip': 1, 'top_n': 5, 'crash': False, 'name': 'Mom6m_Skip1_Top5'},
    {'lookback': 12, 'skip': 1, 'top_n': 3, 'crash': False, 'name': 'Mom12m_Skip1_Top3'},
    {'lookback': 3, 'skip': 1, 'top_n': 3, 'crash': False, 'name': 'Mom3m_Skip1_Top3'},

    # With crash filter
    {'lookback': 6, 'skip': 1, 'top_n': 3, 'crash': True, 'name': 'Mom6m_Skip1_Top3_CrashFilter'},
    {'lookback': 12, 'skip': 1, 'top_n': 3, 'crash': True, 'name': 'Mom12m_Skip1_Top3_CrashFilter'},
    {'lookback': 6, 'skip': 1, 'top_n': 5, 'crash': True, 'name': 'Mom6m_Skip1_Top5_CrashFilter'},

    # Without skip (risk of reversal)
    {'lookback': 6, 'skip': 0, 'top_n': 3, 'crash': True, 'name': 'Mom6m_NoSkip_Top3_CrashFilter'},

    # Top 1 (ultra concentrated)
    {'lookback': 6, 'skip': 1, 'top_n': 1, 'crash': True, 'name': 'Mom6m_Skip1_Top1_CrashFilter'},

    # Top 10 (more diversified)
    {'lookback': 6, 'skip': 1, 'top_n': 10, 'crash': True, 'name': 'Mom6m_Skip1_Top10_CrashFilter'},
]

results = []
for cfg in configs:
    print(f"\n  Testing {cfg['name']}...")
    port_df = concentrated_momentum_backtest(
        monthly_ret, spy_close,
        lookback_months=cfg['lookback'],
        skip_recent=cfg['skip'],
        top_n=cfg['top_n'],
        crash_filter=cfg['crash']
    )

    if port_df is None or len(port_df) < 24:
        print(f"    Insufficient data")
        continue

    metrics = compute_metrics_monthly(port_df)
    if metrics is None:
        continue

    sg, sr, gap, r1_pass = regime_test_monthly(port_df)
    perm_p = permutation_test_monthly(port_df)

    r = {
        'strategy': cfg['name'],
        **metrics,
        'sharpe_green': round(sg, 3) if sg is not None else None,
        'sharpe_red': round(sr, 3) if sr is not None else None,
        'R1_regime_gap': round(gap, 3) if gap is not None else None,
        'R1_pass': str(r1_pass),
        'perm_p_value': round(perm_p, 3),
        'perm_pass': str(perm_p < 0.05),
    }
    results.append(r)

    print(f"    Sharpe={metrics['sharpe']:.3f}, CAGR={metrics['CAGR']:.1f}%, WR={metrics['win_rate']:.1f}%")
    print(f"    MaxDD={metrics['max_drawdown']:.1f}%, Calmar={metrics['calmar']:.3f}")
    if gap is not None:
        print(f"    R1: gap={gap:.3f} ({'PASS' if r1_pass else 'FAIL'}), Green={sg:.3f}, Red={sr:.3f}")
    print(f"    Perm p={perm_p:.3f} ({'PASS' if perm_p < 0.05 else 'FAIL'})")

# Save
print("\n[6/6] Saving results...")
with open(os.path.join(OUTPUT_DIR, 'r2_strategy4_concentrated_momentum.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved.")
print("\n" + "=" * 70)
print("CONCENTRATED MOMENTUM SUMMARY")
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
