#!/usr/bin/env python3
"""
Withdrawal Strategy Optimization
==================================
For Phase 4+ ($50K+), test how to sustainably withdraw income while
maintaining portfolio growth.

Tests:
1. Fixed percentage withdrawal (2%, 3%, 4%, 5% annually)
2. Variable withdrawal (higher in good years, lower in bad)
3. Guardrails method (Guyton-Klinger: floor/ceiling on withdrawals)
4. Bucket strategy (3 years cash, rest invested)
5. Dividend-only (live on dividends, never touch principal)

All on top of our vol-adjusted leverage strategy.
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/withdrawal'
os.makedirs(OUTPUT_DIR, exist_ok=True)

STARTING_CAPITAL = 50000
ANNUAL_CONTRIBUTION = 0  # No more contributions in withdrawal phase

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
    print(f"  Data: {len(closes)} days, {closes.index[0].strftime('%Y-%m-%d')} to {closes.index[-1].strftime('%Y-%m-%d')}")
    return closes

def get_vol_regime(spy, spy_ret, vol_21d, spy_sma50, i):
    """Get vol-adjusted allocation."""
    vol = vol_21d.iloc[i] if not np.isnan(vol_21d.iloc[i]) else 0.15
    protection = spy.iloc[i] > spy_sma50.iloc[i] if not np.isnan(spy_sma50.iloc[i]) else True

    if not protection:
        return {'SPY': 1.0}  # Cash proxy
    elif vol < 0.20:
        return {'UPRO': 1.0}
    elif vol < 0.30:
        return {'SPY': 1.0}
    else:
        return {'GLD': 0.5, 'TLT': 0.5}

def simulate_withdrawal(closes, withdrawal_func, name=""):
    """
    Simulate portfolio with withdrawals.
    withdrawal_func(year, portfolio_value, prev_withdrawal, annual_return) -> withdrawal_amount
    """
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    spy_sma50 = spy.rolling(50).mean()
    returns = closes.pct_change().fillna(0)

    warmup = 63
    portfolio_val = STARTING_CAPITAL
    holdings = {}
    cash = float(STARTING_CAPITAL)
    last_alloc = None
    last_year = None
    prev_withdrawal = 0
    total_withdrawn = 0
    annual_withdrawals = []
    year_start_val = STARTING_CAPITAL

    daily_values = []
    daily_dates = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]

        # Update holdings
        for ticker in list(holdings.keys()):
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    holdings[ticker] *= (1 + r)

        portfolio_val = cash + sum(holdings.values())

        # Annual withdrawal (on first trading day of year)
        current_year = date.year
        if current_year != last_year and last_year is not None:
            # Calculate last year's return
            annual_return = (portfolio_val - year_start_val) / year_start_val if year_start_val > 0 else 0

            # Get withdrawal amount
            withdrawal = withdrawal_func(current_year, portfolio_val, prev_withdrawal, annual_return)

            if withdrawal > 0 and portfolio_val > withdrawal:
                # Withdraw proportionally from holdings
                withdrawal_pct = withdrawal / portfolio_val
                for t in holdings:
                    holdings[t] *= (1 - withdrawal_pct)
                portfolio_val -= withdrawal
                total_withdrawn += withdrawal
                prev_withdrawal = withdrawal
                annual_withdrawals.append((current_year, withdrawal, portfolio_val))

            year_start_val = portfolio_val
        last_year = current_year

        # Rebalance based on vol regime
        target = get_vol_regime(spy, spy_ret, vol_21d, spy_sma50, i)

        should_switch = False
        if last_alloc is None:
            should_switch = True
        else:
            for t in set(list(target.keys()) + list(last_alloc.keys())):
                if abs(target.get(t, 0) - last_alloc.get(t, 0)) > 0.10:
                    should_switch = True
                    break

        if should_switch:
            total_val = cash + sum(holdings.values())
            holdings = {t: total_val * w for t, w in target.items() if w > 0}
            cash = 0
            last_alloc = dict(target)

        portfolio_val = cash + sum(holdings.values())
        daily_values.append(portfolio_val)
        daily_dates.append(date)

    portfolio = pd.Series(daily_values, index=daily_dates)
    return portfolio, total_withdrawn, annual_withdrawals

def compute_metrics(portfolio, total_withdrawn, annual_withdrawals):
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

    # Ruin check
    ruin = portfolio.min() < STARTING_CAPITAL * 0.1  # Below 10% of starting

    # Average annual withdrawal
    if annual_withdrawals:
        avg_withdrawal = np.mean([w for _, w, _ in annual_withdrawals])
        min_withdrawal = min(w for _, w, _ in annual_withdrawals)
        max_withdrawal = max(w for _, w, _ in annual_withdrawals)
    else:
        avg_withdrawal = min_withdrawal = max_withdrawal = 0

    return {
        'final_value': float(final),
        'total_withdrawn': float(total_withdrawn),
        'total_value_created': float(final + total_withdrawn - STARTING_CAPITAL),
        'avg_annual_withdrawal': float(avg_withdrawal),
        'min_annual_withdrawal': float(min_withdrawal),
        'max_annual_withdrawal': float(max_withdrawal),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd * 100),
        'ruin': bool(ruin),
        'years': float(years),
    }

def main():
    print("="*70)
    print("WITHDRAWAL STRATEGY OPTIMIZATION")
    print("="*70)
    print(f"  Starting capital: ${STARTING_CAPITAL:,}")
    print(f"  Using vol-adjusted leverage strategy")

    closes = download_data()

    # Define withdrawal strategies
    strategies = {}

    # 1. No withdrawal (growth benchmark)
    strategies['0. No withdrawal'] = lambda y, pv, pw, ar: 0

    # 2-5. Fixed percentage
    for pct in [0.02, 0.03, 0.04, 0.05, 0.08, 0.10]:
        pct_val = pct
        strategies[f'{int(pct*100)}% fixed annual'] = lambda y, pv, pw, ar, p=pct_val: pv * p

    # 6. Variable (4% in good years, 2% in bad)
    strategies['Variable (4%/2%)'] = lambda y, pv, pw, ar: pv * (0.04 if ar > 0 else 0.02)

    # 7. Guardrails (Guyton-Klinger)
    def guardrails(y, pv, pw, ar):
        if pw == 0:
            return pv * 0.04  # Initial withdrawal = 4%
        # Adjust for inflation (assume 3%)
        target = pw * 1.03
        # Floor: never withdraw less than 3% of portfolio
        floor = pv * 0.03
        # Ceiling: never more than 6%
        ceiling = pv * 0.06
        return min(max(target, floor), ceiling)
    strategies['Guardrails (3-6%)'] = guardrails

    # 8. Percentage of gains only
    strategies['Gains only (50% of gains)'] = lambda y, pv, pw, ar: max(0, pv * ar * 0.5) if ar > 0 else 0

    # 9. Constant dollar (inflation-adjusted)
    def constant_dollar(y, pv, pw, ar):
        if pw == 0:
            return 2000  # $2K/year to start
        return pw * 1.03  # Inflate 3%/year
    strategies['Constant $2K/yr (+3% inflation)'] = constant_dollar

    print(f"\nTesting {len(strategies)} withdrawal strategies...\n")

    results = {}
    for name, func in strategies.items():
        portfolio, total_w, annual_w = simulate_withdrawal(closes, func, name)
        m = compute_metrics(portfolio, total_w, annual_w)
        if m:
            results[name] = m

    # Sort by total value created (final + withdrawn - initial)
    sorted_total = sorted(results.items(), key=lambda x: x[1]['total_value_created'], reverse=True)

    print("="*70)
    print("RESULTS — RANKED BY TOTAL VALUE CREATED (final + withdrawn)")
    print("="*70)

    print(f"\n  {'Strategy':<30s} {'Final $':>10s} {'Withdrawn':>10s} {'Total Created':>13s} {'Avg/yr':>8s} {'MaxDD':>7s} {'Ruin':>5s}")
    print("  " + "-"*88)
    for name, m in sorted_total:
        print(f"  {name:<30s} ${m['final_value']:>9,.0f} ${m['total_withdrawn']:>9,.0f} "
              f"${m['total_value_created']:>12,.0f} ${m['avg_annual_withdrawal']:>7,.0f} "
              f"{m['max_dd']:>6.1f}% {'YES' if m['ruin'] else 'no':>4s}")

    # Sustainability analysis
    print("\n" + "="*70)
    print("SUSTAINABILITY ANALYSIS")
    print("="*70)

    for name, m in sorted_total:
        if m['total_withdrawn'] > 0:
            sustainability = m['final_value'] / STARTING_CAPITAL
            print(f"  {name:<30s}: portfolio is {sustainability:.1f}x starting capital after {m['years']:.0f}yr + ${m['total_withdrawn']:,.0f} withdrawn")

    # Practical recommendation
    print("\n" + "="*70)
    print("RECOMMENDATION")
    print("="*70)

    # Find best sustainable withdrawal (portfolio still growing)
    sustainable = [(n, m) for n, m in sorted_total
                   if m['final_value'] > STARTING_CAPITAL and m['total_withdrawn'] > 0 and not m['ruin']]

    if sustainable:
        best_name, best_m = max(sustainable, key=lambda x: x[1]['avg_annual_withdrawal'])
        print(f"\n  BEST SUSTAINABLE WITHDRAWAL: {best_name}")
        print(f"    Avg annual income: ${best_m['avg_annual_withdrawal']:,.0f}")
        print(f"    Final portfolio: ${best_m['final_value']:,.0f} ({best_m['final_value']/STARTING_CAPITAL:.1f}x starting)")
        print(f"    Total withdrawn: ${best_m['total_withdrawn']:,.0f}")
        print(f"    Total value created: ${best_m['total_value_created']:,.0f}")
        print(f"    MaxDD: {best_m['max_dd']:.1f}%")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'starting_capital': STARTING_CAPITAL,
        'results': results,
        'ranking': [n for n, _ in sorted_total],
    }

    output_path = os.path.join(OUTPUT_DIR, 'withdrawal_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved.")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    main()
