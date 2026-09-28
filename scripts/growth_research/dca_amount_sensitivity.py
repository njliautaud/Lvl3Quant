#!/usr/bin/env python3
"""
DCA Amount Sensitivity Analysis
=================================
How does the system scale with different DCA amounts?
At what point does DCA amount matter less than compound growth?

Also: lump sum vs DCA comparison for initial capital.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/dca_sensitivity'
os.makedirs(OUTPUT_DIR, exist_ok=True)


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


def simulate_v2(closes, initial, weekly_dca):
    """Simulate GAMEPLAN v2 with given initial and DCA amounts."""
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    sma20 = spy.rolling(20).mean()
    sma200 = spy.rolling(200).mean()
    returns = closes.pct_change().fillna(0)

    warmup = 260
    holdings = {}
    cash = float(initial)
    total_contributed = float(initial)
    last_week = None
    last_regime = None
    daily_values = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]
        month = date.month

        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += weekly_dca
            total_contributed += weekly_dca
            last_week = week_key

        for ticker in list(holdings.keys()):
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    holdings[ticker] *= (1 + r)

        vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15

        # 20/200 crossover protection
        protection = True
        if not np.isnan(sma20.iloc[i]) and not np.isnan(sma200.iloc[i]):
            protection = sma20.iloc[i] > sma200.iloc[i]

        # September hedge
        if month == 9 and protection:
            regime = 'SPY'
            target = {'SPY': 1.0}
        elif not protection:
            regime = 'CASH'
            target = {'SPY': 1.0}
        elif vol < 0.20:
            regime = 'UPRO'
            target = {'UPRO': 1.0}
        elif vol < 0.30:
            regime = 'SPY'
            target = {'SPY': 1.0}
        else:
            regime = 'SAFE'
            target = {'GLD': 1.0}

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

    portfolio = pd.Series(daily_values, index=closes.index[warmup:])
    return portfolio, total_contributed


def compute_metrics(portfolio, total_contributed):
    r = portfolio.pct_change().dropna()
    years = len(r) / 252
    final = portfolio.iloc[-1]
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ((1 + r).prod() ** (252 / len(r)) - 1) / ann_vol if ann_vol > 0 else 0
    peak = portfolio.expanding().max()
    max_dd = ((portfolio - peak) / peak).min()
    profit = final - total_contributed
    roi = profit / total_contributed * 100
    return {
        'final': float(final),
        'contributed': float(total_contributed),
        'profit': float(profit),
        'roi': float(roi),
        'sharpe': float(sharpe),
        'max_dd': float(max_dd * 100),
    }


def main():
    print("="*70)
    print("DCA AMOUNT SENSITIVITY ANALYSIS")
    print("="*70)

    closes = download_data()
    print(f"  Data: {len(closes)} days")

    # --- DCA amount sweep ---
    print("\n  DCA AMOUNT SWEEP ($500 initial):")
    print(f"  {'DCA/wk':>8s} {'Final':>12s} {'Contributed':>12s} {'Profit':>12s} {'ROI':>8s} {'Sharpe':>7s} {'MaxDD':>7s}")
    print("  " + "-"*72)

    dca_amounts = [0, 25, 50, 75, 100, 150, 200, 300, 500, 1000]
    dca_results = {}

    for dca in dca_amounts:
        portfolio, total = simulate_v2(closes, 500, dca)
        m = compute_metrics(portfolio, total)
        dca_results[dca] = m
        print(f"  ${dca:>6d}/wk ${m['final']:>11,.0f} ${m['contributed']:>11,.0f} "
              f"${m['profit']:>11,.0f} {m['roi']:>7.0f}% {m['sharpe']:>7.3f} {m['max_dd']:>6.1f}%")

    # --- Initial capital sweep ---
    print("\n  INITIAL CAPITAL SWEEP ($100/wk DCA):")
    print(f"  {'Initial':>8s} {'Final':>12s} {'Contributed':>12s} {'Profit':>12s} {'ROI':>8s}")
    print("  " + "-"*54)

    initials = [0, 100, 500, 1000, 2500, 5000, 10000, 25000, 50000]
    init_results = {}

    for init in initials:
        portfolio, total = simulate_v2(closes, init, 100)
        m = compute_metrics(portfolio, total)
        init_results[init] = m
        print(f"  ${init:>6d}    ${m['final']:>11,.0f} ${m['contributed']:>11,.0f} "
              f"${m['profit']:>11,.0f} {m['roi']:>7.0f}%")

    # --- Lump sum vs DCA ---
    print("\n  LUMP SUM vs DCA ($10,000 total):")

    # Lump sum: $10K upfront, no DCA
    portfolio_ls, total_ls = simulate_v2(closes, 10000, 0)
    m_ls = compute_metrics(portfolio_ls, total_ls)

    # DCA: $0 upfront, $192/wk for 52 weeks ≈ $10K
    portfolio_dca, total_dca = simulate_v2(closes, 0, 192)
    m_dca = compute_metrics(portfolio_dca, total_dca)

    # Hybrid: $5K upfront + $96/wk
    portfolio_hyb, total_hyb = simulate_v2(closes, 5000, 96)
    m_hyb = compute_metrics(portfolio_hyb, total_hyb)

    print(f"    Lump sum ($10K): ${m_ls['final']:,.0f} (ROI {m_ls['roi']:.0f}%, MaxDD {m_ls['max_dd']:.1f}%)")
    print(f"    DCA ($192/wk):   ${m_dca['final']:,.0f} (ROI {m_dca['roi']:.0f}%, MaxDD {m_dca['max_dd']:.1f}%)")
    print(f"    Hybrid (50/50):  ${m_hyb['final']:,.0f} (ROI {m_hyb['roi']:.0f}%, MaxDD {m_hyb['max_dd']:.1f}%)")

    # --- Key insights ---
    print(f"\n{'='*70}")
    print("KEY INSIGHTS")
    print(f"{'='*70}")

    # DCA multiplier effect
    base = dca_results[100]
    doubled = dca_results[200]
    print(f"\n  Doubling DCA ($100→$200/wk):")
    print(f"    Final: ${base['final']:,.0f} → ${doubled['final']:,.0f} ({doubled['final']/base['final']:.2f}x)")
    print(f"    Contributed: ${base['contributed']:,.0f} → ${doubled['contributed']:,.0f} (2.0x)")
    print(f"    Profit: ${base['profit']:,.0f} → ${doubled['profit']:,.0f} ({doubled['profit']/base['profit']:.2f}x)")

    # When does compound growth overtake DCA?
    print(f"\n  DCA vs Compound growth dominance:")
    for dca in [50, 100, 200, 500]:
        m = dca_results[dca]
        pct_from_dca = m['contributed'] / m['final'] * 100
        pct_from_growth = (1 - m['contributed'] / m['final']) * 100
        print(f"    ${dca}/wk: {pct_from_dca:.0f}% from contributions, {pct_from_growth:.0f}% from growth")

    # Marginal value of extra $1/week
    print(f"\n  Marginal value of extra $1/week DCA:")
    for i in range(1, len(dca_amounts)):
        prev_dca = dca_amounts[i-1]
        curr_dca = dca_amounts[i]
        diff_dca = curr_dca - prev_dca
        diff_final = dca_results[curr_dca]['final'] - dca_results[prev_dca]['final']
        per_dollar = diff_final / diff_dca if diff_dca > 0 else 0
        print(f"    ${prev_dca}→${curr_dca}/wk: +${diff_final:,.0f} total, "
              f"${per_dollar:,.0f} per extra $/wk")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'dca_results': dca_results,
        'initial_results': init_results,
    }
    with open(os.path.join(OUTPUT_DIR, 'dca_sensitivity.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
