#!/usr/bin/env python3
"""
International Diversification Analysis
=======================================
Tests whether adding international exposure improves our vol-adjusted system.

Questions:
1. Does international diversification reduce risk for a US-focused portfolio?
2. Is there a momentum-based international rotation that adds alpha?
3. How does currency risk affect returns?
4. Should we add EEM/IEFA at any phase?

Tests:
1. US-only (baseline UPRO)
2. US + Europe (VGK)
3. US + Emerging (EEM)
4. US + Japan (EWJ)
5. Global momentum (rotate between US/Europe/EM/Japan)
6. Risk parity with international
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/international'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100

def download_data():
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT', 'VIXY',
               'VGK', 'EEM', 'EWJ', 'IEFA', 'VWO', 'FXI', 'EWZ',
               'UUP']

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
    closes = closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])
    print(f"  Data: {len(closes)} days, {closes.shape[1]} tickers")
    return closes

def simulate_strategy(closes, get_intl_alloc, name=""):
    """Simulate vol-adjusted strategy with international allocation."""
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    spy_sma50 = spy.rolling(50).mean()
    returns = closes.pct_change().fillna(0)

    warmup = 63
    portfolio_val = float(INITIAL)
    holdings = {}
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    last_month = None

    daily_values = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]

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

        if not protection:
            regime = 'CASH'
            target = {'SPY': 1.0}
        elif vol < 0.20:
            regime = 'UPRO'
            # Get international allocation (monthly rebalance only)
            if date.month != last_month:
                target = get_intl_alloc(date, closes, i)
                last_month = date.month
            else:
                target = get_intl_alloc(date, closes, i) if last_regime != regime else None
        elif vol < 0.30:
            regime = 'SPY'
            target = {'SPY': 1.0}
        else:
            regime = 'SAFE'
            target = {'GLD': 0.5, 'TLT': 0.5}

        if regime != last_regime and target:
            total_val = cash + sum(holdings.values())
            holdings = {t: total_val * w for t, w in target.items() if w > 0 and t in closes.columns}
            if not holdings:
                holdings = {'SPY': total_val}
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
    dd = (portfolio - peak) / peak
    max_dd = dd.min()
    cagr = (final / portfolio.iloc[0]) ** (1/years) - 1

    # SPY correlation
    spy_ret = portfolio.pct_change().dropna()

    return {
        'final_value': float(final),
        'profit': float(final - total_contributed),
        'cagr': float(cagr * 100),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd * 100),
    }

def main():
    print("="*70)
    print("INTERNATIONAL DIVERSIFICATION ANALYSIS")
    print("="*70)

    closes = download_data()

    # Check what international ETFs are available
    intl_tickers = ['VGK', 'EEM', 'EWJ', 'IEFA', 'VWO', 'FXI', 'EWZ']
    avail_intl = [t for t in intl_tickers if t in closes.columns]
    print(f"  Available international ETFs: {', '.join(avail_intl)}")

    # --- Correlation analysis ---
    print("\n  CORRELATION MATRIX (21d returns):")
    corr_tickers = ['SPY', 'UPRO'] + avail_intl
    ret_21d = closes[corr_tickers].pct_change(21).dropna()
    corr = ret_21d.corr()

    for t in corr_tickers:
        if t != 'SPY' and t in corr.columns:
            print(f"    SPY ↔ {t}: {corr.loc['SPY', t]:.3f}")

    # --- Standalone performance ---
    print("\n  STANDALONE PERFORMANCE (2012-2026):")
    for t in corr_tickers:
        if t in closes.columns:
            p = closes[t].dropna()
            if len(p) > 252:
                total_ret = p.iloc[-1] / p.iloc[0] - 1
                years = len(p) / 252
                cagr = (1 + total_ret) ** (1/years) - 1
                vol = p.pct_change().std() * np.sqrt(252)
                sharpe = cagr / vol if vol > 0 else 0
                print(f"    {t:<6s}: CAGR {cagr*100:>6.1f}%, Vol {vol*100:>5.1f}%, Sharpe {sharpe:>5.2f}")

    # --- Strategy variants ---
    strategies = {}

    # 1. US-only (baseline)
    strategies['US only (UPRO)'] = lambda d, c, i: {'UPRO': 1.0}

    # 2. US + Europe
    if 'VGK' in avail_intl:
        strategies['80% UPRO + 20% VGK'] = lambda d, c, i: {'UPRO': 0.80, 'VGK': 0.20}
        strategies['70% UPRO + 30% VGK'] = lambda d, c, i: {'UPRO': 0.70, 'VGK': 0.30}

    # 3. US + Emerging
    if 'EEM' in avail_intl:
        strategies['80% UPRO + 20% EEM'] = lambda d, c, i: {'UPRO': 0.80, 'EEM': 0.20}

    # 4. US + Japan
    if 'EWJ' in avail_intl:
        strategies['80% UPRO + 20% EWJ'] = lambda d, c, i: {'UPRO': 0.80, 'EWJ': 0.20}

    # 5. Global equal weight
    if len(avail_intl) >= 3:
        def global_equal(d, c, i):
            alloc = {'UPRO': 0.50}
            intl_weight = 0.50 / min(3, len(avail_intl))
            for t in avail_intl[:3]:
                alloc[t] = intl_weight
            return alloc
        strategies['50% UPRO + 50% Global'] = global_equal

    # 6. International momentum
    if len(avail_intl) >= 3:
        def intl_momentum(d, c, i):
            rets = {}
            for t in avail_intl:
                if t in c.columns and i >= 63:
                    r = c[t].iloc[i] / c[t].iloc[i-63] - 1
                    if not np.isnan(r):
                        rets[t] = r
            if len(rets) >= 2:
                best = sorted(rets, key=rets.get, reverse=True)[:2]
                alloc = {'UPRO': 0.60}
                for t in best:
                    alloc[t] = 0.20
                return alloc
            return {'UPRO': 1.0}
        strategies['60% UPRO + Top 2 Intl Mom'] = intl_momentum

    # 7. Inverse correlation (go international when US weakening)
    if 'EEM' in avail_intl and 'VGK' in avail_intl:
        def us_weakness_intl(d, c, i):
            if i < 21:
                return {'UPRO': 1.0}
            spy_mom = c['SPY'].iloc[i] / c['SPY'].iloc[i-21] - 1
            if spy_mom < -0.02:
                return {'UPRO': 0.50, 'EEM': 0.25, 'VGK': 0.25}
            return {'UPRO': 1.0}
        strategies['Intl when US weak'] = us_weakness_intl

    print(f"\nTesting {len(strategies)} strategies...")

    results = {}
    for name, alloc in strategies.items():
        print(f"  Running: {name}...", end=" ", flush=True)
        portfolio, total_cont = simulate_strategy(closes, alloc, name)
        m = compute_metrics(portfolio, total_cont)
        if m:
            results[name] = m
            print(f"${m['final_value']:,.0f} | Sharpe {m['sharpe']:.3f} | MaxDD {m['max_dd']:.1f}%")

    sorted_results = sorted(results.items(), key=lambda x: x[1]['final_value'], reverse=True)

    print("\n" + "="*70)
    print("RESULTS")
    print("="*70)

    print(f"\n  {'Strategy':<35s} {'Final $':>10s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s} {'CAGR':>7s}")
    print("  " + "-"*76)
    for name, m in sorted_results:
        print(f"  {name:<35s} ${m['final_value']:>9,.0f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['max_dd']:>6.1f}% {m['cagr']:>6.1f}%")

    baseline = results.get('US only (UPRO)', {})
    if baseline:
        print(f"\n  vs US-only baseline:")
        for name, m in sorted_results:
            if name == 'US only (UPRO)':
                continue
            val_diff = m['final_value'] - baseline['final_value']
            sharpe_diff = m['sharpe'] - baseline['sharpe']
            dd_diff = m['max_dd'] - baseline['max_dd']
            print(f"    {name:<33s}: value {'+' if val_diff > 0 else ''}{val_diff:,.0f}, "
                  f"Sharpe {sharpe_diff:+.3f}, MaxDD {dd_diff:+.1f}pp")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'results': results,
        'ranking': [n for n, _ in sorted_results],
    }
    with open(os.path.join(OUTPUT_DIR, 'international_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved.")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    main()
