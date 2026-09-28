#!/usr/bin/env python3
"""
Monthly Seasonality Analysis for Vol-Adjusted System
=====================================================
Prior research showed Jul/Nov/May are best months, Sep/Feb worst.
Can we exploit this by adjusting vol thresholds or leverage seasonally?

Tests:
1. Baseline vol-adjusted system
2. "Sell in May" overlay (cash May-Oct)
3. Aggressive in best months (wider vol threshold)
4. Conservative in worst months (tighter vol threshold)
5. Seasonal vol threshold schedule (month-specific thresholds)
6. "Santa rally" (always UPRO Nov-Jan)
7. September hedge (go to SPY in September)
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/seasonality'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100


def download_data():
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT']
    data = yf.download(tickers, start='2012-01-01', period='max',
                       auto_adjust=True, threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data
    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except:
            pass
    return closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])


def simulate(closes, vol_threshold_func, force_regime_func=None, name=""):
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    spy_sma50 = spy.rolling(50).mean()
    returns = closes.pct_change().fillna(0)

    warmup = 63
    holdings = {}
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    daily_values = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]
        month = date.month

        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        for ticker in list(holdings.keys()):
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    holdings[ticker] *= (1 + r)

        vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15
        protection = spy.iloc[i] > spy_sma50.iloc[i] if not np.isnan(spy_sma50.iloc[i]) else True

        # Get month-specific thresholds
        low_thresh, high_thresh = vol_threshold_func(month)

        # Check for forced regime override
        forced = force_regime_func(month, vol, protection) if force_regime_func else None

        if forced:
            regime = forced[0]
            target = forced[1]
        elif not protection:
            regime = 'CASH'
            target = {'SPY': 1.0}
        elif vol < low_thresh:
            regime = 'UPRO'
            target = {'UPRO': 1.0}
        elif vol < high_thresh:
            regime = 'SPY'
            target = {'SPY': 1.0}
        else:
            regime = 'SAFE'
            target = {'GLD': 0.5, 'TLT': 0.5}

        if regime != last_regime:
            total_val = cash + sum(holdings.values())
            holdings = {t: total_val * w for t, w in target.items()
                       if w > 0 and t in closes.columns}
            cash = 0
            last_regime = regime
        elif cash > 50 and holdings:
            total_h = sum(holdings.values())
            if total_h > 0:
                for t in holdings:
                    holdings[t] += cash * (holdings[t] / total_h)
                cash = 0

        portfolio_val = cash + sum(holdings.values())
        daily_values.append(portfolio_val)

    return pd.Series(daily_values, index=closes.index[warmup:]), total_contributed


def compute_metrics(portfolio, total_contributed):
    r = portfolio.pct_change().dropna()
    if len(r) < 63:
        return None
    years = len(r) / 252
    final = portfolio.iloc[-1]
    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    neg = r[r < 0]
    downside_vol = neg.std() * np.sqrt(252) if len(neg) > 0 else ann_vol
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0
    peak = portfolio.expanding().max()
    max_dd = ((portfolio - peak) / peak).min()
    cagr = (final / portfolio.iloc[0]) ** (1/years) - 1
    return {
        'final_value': float(final),
        'cagr': float(cagr * 100),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd * 100),
    }


def main():
    print("="*70)
    print("MONTHLY SEASONALITY ANALYSIS")
    print("="*70)

    closes = download_data()
    print(f"  Data: {len(closes)} days")

    # --- Monthly returns for UPRO and SPY ---
    print("\n  MONTHLY RETURNS (UPRO, 2012-2026):")
    upro_monthly = closes['UPRO'].resample('ME').last().pct_change().dropna()
    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

    for m in range(1, 13):
        month_ret = upro_monthly[upro_monthly.index.month == m]
        if len(month_ret) > 0:
            avg = month_ret.mean() * 100
            med = month_ret.median() * 100
            wr = (month_ret > 0).mean() * 100
            worst = month_ret.min() * 100
            best = month_ret.max() * 100
            print(f"    {month_names[m-1]}: avg {avg:>6.1f}%, med {med:>6.1f}%, "
                  f"WR {wr:>4.0f}%, range [{worst:>6.1f}%, {best:>5.1f}%]")

    # --- Strategy variants ---
    strategies = {}

    # 1. Baseline (flat 20%/30% thresholds)
    strategies['1. Baseline (20/30)'] = (
        lambda m: (0.20, 0.30),
        None
    )

    # 2. Sell in May (SPY May-Oct)
    strategies['2. Sell in May'] = (
        lambda m: (0.20, 0.30),
        lambda m, v, p: ('SPY', {'SPY': 1.0}) if 5 <= m <= 10 else None
    )

    # 3. Aggressive best months (Jul, Nov, May, Apr — wider threshold)
    best_months = {4, 5, 7, 11}  # Apr, May, Jul, Nov
    strategies['3. Aggressive best months'] = (
        lambda m: (0.25, 0.35) if m in best_months else (0.20, 0.30),
        None
    )

    # 4. Conservative worst months (Sep, Feb, Oct — tighter threshold)
    worst_months = {2, 9, 10}  # Feb, Sep, Oct
    strategies['4. Conservative Sep/Feb/Oct'] = (
        lambda m: (0.15, 0.25) if m in worst_months else (0.20, 0.30),
        None
    )

    # 5. Combined: aggressive in best, conservative in worst
    strategies['5. Combined seasonal'] = (
        lambda m: (0.25, 0.35) if m in best_months else (0.15, 0.25) if m in worst_months else (0.20, 0.30),
        None
    )

    # 6. Santa rally (always UPRO Nov-Jan regardless of vol)
    strategies['6. Santa rally (Nov-Jan)'] = (
        lambda m: (0.20, 0.30),
        lambda m, v, p: ('UPRO', {'UPRO': 1.0}) if m in {11, 12, 1} and p else None
    )

    # 7. September hedge (go to SPY in Sep)
    strategies['7. September SPY hedge'] = (
        lambda m: (0.20, 0.30),
        lambda m, v, p: ('SPY', {'SPY': 1.0}) if m == 9 else None
    )

    # 8. Best-half year (UPRO Nov-Apr, SPY May-Oct)
    strategies['8. Best half (Nov-Apr UPRO)'] = (
        lambda m: (0.20, 0.30),
        lambda m, v, p: ('SPY', {'SPY': 1.0}) if 5 <= m <= 10 and v < 0.30 and p else None
    )

    # 9. Q4 aggressive (Oct-Dec wider thresholds)
    strategies['9. Q4 aggressive'] = (
        lambda m: (0.25, 0.40) if m in {10, 11, 12} else (0.20, 0.30),
        None
    )

    # 10. Monthly-optimized thresholds (from data)
    # Use empirical monthly vol stats to set thresholds
    monthly_vol_means = {}
    spy_ret = closes['SPY'].pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    for m in range(1, 13):
        mask = closes.index.month == m
        monthly_vol_means[m] = vol_21d[mask].dropna().mean()

    strategies['10. Vol-adaptive seasonal'] = (
        lambda m: (monthly_vol_means.get(m, 0.20) * 1.2,
                   monthly_vol_means.get(m, 0.30) * 1.8),
        None
    )

    print(f"\nTesting {len(strategies)} strategies...")

    results = {}
    for name, (thresh_func, force_func) in strategies.items():
        print(f"  {name}...", end=" ", flush=True)
        portfolio, total = simulate(closes, thresh_func, force_func, name)
        m = compute_metrics(portfolio, total)
        if m:
            results[name] = m
            print(f"${m['final_value']:,.0f} | Sharpe {m['sharpe']:.3f} | MaxDD {m['max_dd']:.1f}%")

    # --- Results ---
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)

    print("\n" + "="*70)
    print("RESULTS — RANKED BY SHARPE")
    print("="*70)

    print(f"\n  {'Strategy':<32s} {'Final $':>10s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s} {'CAGR':>7s}")
    print("  " + "-"*72)
    for name, m in sorted_results:
        print(f"  {name:<32s} ${m['final_value']:>9,.0f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['max_dd']:>6.1f}% {m['cagr']:>6.1f}%")

    baseline = results.get('1. Baseline (20/30)', {})
    if baseline:
        print(f"\n  vs Baseline:")
        for name, m in sorted_results:
            if '1. Baseline' in name:
                continue
            val_diff = m['final_value'] - baseline['final_value']
            sharpe_diff = m['sharpe'] - baseline['sharpe']
            print(f"    {name:<30s}: value {val_diff:>+10,.0f}, Sharpe {sharpe_diff:+.3f}")

    # Save
    output = {'run_date': pd.Timestamp.now().isoformat(), 'results': results}
    with open(os.path.join(OUTPUT_DIR, 'seasonality_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
