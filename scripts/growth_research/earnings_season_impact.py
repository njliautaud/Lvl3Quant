#!/usr/bin/env python3
"""
Earnings Season Impact on Vol-Adjusted Strategy
================================================
Tests whether adjusting allocation during earnings season improves returns.

Earnings seasons typically increase volatility, which could trigger
unnecessary vol-regime switches in our strategy.

Questions:
1. Does vol systematically rise during earnings season?
2. Does our strategy underperform during earnings months?
3. Should we adjust vol thresholds during earnings season?
4. Is there an "earnings drift" we can capture?
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/earnings_impact'
os.makedirs(OUTPUT_DIR, exist_ok=True)

def download_data():
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT', 'VIXY']
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
    closes = closes.dropna(how='all').dropna(subset=['UPRO', 'SPY'])
    return closes

def is_earnings_season(date):
    """Earnings season: Jan 15-Feb 15, Apr 15-May 15, Jul 15-Aug 15, Oct 15-Nov 15."""
    month, day = date.month, date.day
    if month == 1 and day >= 15: return True
    if month == 2 and day <= 15: return True
    if month == 4 and day >= 15: return True
    if month == 5 and day <= 15: return True
    if month == 7 and day >= 15: return True
    if month == 8 and day <= 15: return True
    if month == 10 and day >= 15: return True
    if month == 11 and day <= 15: return True
    return False

def simulate_vol_adjusted(closes, vol_low=0.20, vol_high=0.30, earnings_adjust=False, earnings_vol_low=None, earnings_vol_high=None):
    """Simulate vol-adjusted strategy, optionally adjusting thresholds during earnings."""
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    spy_sma50 = spy.rolling(50).mean()
    returns = closes.pct_change().fillna(0)

    warmup = 63
    portfolio_val = 500.0
    holdings = {}
    cash = 500.0
    total_contributed = 500.0
    last_week = None
    last_regime = None

    daily_values = []
    daily_dates = []
    regime_log = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]

        # DCA
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += 100
            total_contributed += 100
            last_week = week_key

        # Update
        for ticker in list(holdings.keys()):
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    holdings[ticker] *= (1 + r)

        vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15
        protection = spy.iloc[i] > spy_sma50.iloc[i] if not np.isnan(spy_sma50.iloc[i]) else True

        # Get thresholds (adjust for earnings if enabled)
        if earnings_adjust and is_earnings_season(date):
            vl = earnings_vol_low if earnings_vol_low is not None else vol_low
            vh = earnings_vol_high if earnings_vol_high is not None else vol_high
        else:
            vl = vol_low
            vh = vol_high

        if not protection:
            regime = 'CASH'
            target = {'SPY': 1.0}
        elif vol < vl:
            regime = 'UPRO'
            target = {'UPRO': 1.0}
        elif vol < vh:
            regime = 'SPY'
            target = {'SPY': 1.0}
        else:
            regime = 'SAFE'
            target = {'GLD': 0.5, 'TLT': 0.5}

        if regime != last_regime:
            total_val = cash + sum(holdings.values())
            holdings = {t: total_val * w for t, w in target.items() if w > 0}
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
        daily_dates.append(date)
        regime_log.append((date, regime, is_earnings_season(date)))

    return pd.Series(daily_values, index=daily_dates), total_contributed, regime_log

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
    dd = (portfolio - peak) / peak
    max_dd = dd.min()
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
    print("EARNINGS SEASON IMPACT ON VOL-ADJUSTED STRATEGY")
    print("="*70)

    closes = download_data()
    spy_ret = closes['SPY'].pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)

    # --- 1. Vol analysis by earnings season ---
    print("\n1. VOLATILITY DURING EARNINGS SEASON")
    print("-"*50)

    vol_valid = vol_21d.dropna()
    earnings_mask = pd.Series([is_earnings_season(d) for d in vol_valid.index], index=vol_valid.index)

    vol_earnings = vol_valid[earnings_mask]
    vol_normal = vol_valid[~earnings_mask]

    print(f"  Earnings season vol: mean {vol_earnings.mean()*100:.1f}%, median {vol_earnings.median()*100:.1f}%")
    print(f"  Normal period vol:   mean {vol_normal.mean()*100:.1f}%, median {vol_normal.median()*100:.1f}%")
    print(f"  Difference: {(vol_earnings.mean() - vol_normal.mean())*100:+.1f}pp")

    # How often does vol breach 20% during earnings vs normal
    breach_earnings = (vol_earnings > 0.20).mean()
    breach_normal = (vol_normal > 0.20).mean()
    print(f"  Vol > 20% frequency: earnings {breach_earnings*100:.1f}%, normal {breach_normal*100:.1f}%")

    # --- 2. Strategy performance in earnings vs normal ---
    print("\n2. STRATEGY PERFORMANCE BY PERIOD")
    print("-"*50)

    portfolio, total_cont, regime_log = simulate_vol_adjusted(closes)

    port_ret = portfolio.pct_change().dropna()
    earnings_days = pd.Series([is_earnings_season(d) for d in port_ret.index], index=port_ret.index)

    earn_ret = port_ret[earnings_days]
    norm_ret = port_ret[~earnings_days]

    earn_sharpe = earn_ret.mean() / earn_ret.std() * np.sqrt(252)
    norm_sharpe = norm_ret.mean() / norm_ret.std() * np.sqrt(252)

    print(f"  Earnings season: Sharpe {earn_sharpe:.3f}, avg daily ret {earn_ret.mean()*100:.3f}%")
    print(f"  Normal period:   Sharpe {norm_sharpe:.3f}, avg daily ret {norm_ret.mean()*100:.3f}%")

    # --- 3. Test adjusted thresholds during earnings ---
    print("\n3. THRESHOLD ADJUSTMENTS DURING EARNINGS")
    print("-"*50)

    configs = [
        ("Baseline (20/30)", 0.20, 0.30, False, None, None),
        ("Wider earnings (25/35)", 0.20, 0.30, True, 0.25, 0.35),
        ("Much wider (30/40)", 0.20, 0.30, True, 0.30, 0.40),
        ("Tighter earnings (15/25)", 0.20, 0.30, True, 0.15, 0.25),
        ("Always UPRO in earnings", 0.20, 0.30, True, 0.50, 0.60),
        ("Always SPY in earnings", 0.20, 0.30, True, 0.01, 0.30),
    ]

    results = {}
    for name, vl, vh, adjust, evl, evh in configs:
        portfolio, total_cont, _ = simulate_vol_adjusted(closes, vl, vh, adjust, evl, evh)
        m = compute_metrics(portfolio, total_cont)
        if m:
            results[name] = m

    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_value'], reverse=True)

    print(f"\n  {'Config':<30s} {'Final $':>10s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s} {'CAGR':>7s}")
    print("  " + "-"*72)
    for name, m in sorted_results:
        print(f"  {name:<30s} ${m['final_value']:>9,.0f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['max_dd']:>6.1f}% {m['cagr']:>6.1f}%")

    # --- 4. Monthly return analysis ---
    print("\n4. MONTHLY RETURN BY MONTH")
    print("-"*50)

    monthly_ret = portfolio.resample('ME').last().pct_change().dropna()
    month_stats = {}
    for month in range(1, 13):
        m_ret = monthly_ret[monthly_ret.index.month == month]
        month_stats[month] = {
            'mean': float(m_ret.mean() * 100),
            'std': float(m_ret.std() * 100),
            'sharpe': float(m_ret.mean() / m_ret.std() * np.sqrt(12)) if m_ret.std() > 0 else 0,
            'win_rate': float((m_ret > 0).mean() * 100),
        }

    month_names = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                   'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    print(f"  {'Month':<6s} {'Avg Ret':>8s} {'Sharpe':>7s} {'WR':>6s} {'Earnings':>9s}")
    print("  " + "-"*40)
    for m in range(1, 13):
        s = month_stats[m]
        is_earn = m in [1, 2, 4, 5, 7, 8, 10, 11]
        print(f"  {month_names[m-1]:<6s} {s['mean']:>+7.1f}% {s['sharpe']:>7.2f} {s['win_rate']:>5.0f}% {'  YES' if is_earn else ''}")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'vol_analysis': {
            'earnings_mean_vol': float(vol_earnings.mean() * 100),
            'normal_mean_vol': float(vol_normal.mean() * 100),
            'vol_breach_20_earnings': float(breach_earnings * 100),
            'vol_breach_20_normal': float(breach_normal * 100),
        },
        'strategy_results': results,
        'month_stats': month_stats,
    }

    with open(os.path.join(OUTPUT_DIR, 'earnings_impact_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved.")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    main()
