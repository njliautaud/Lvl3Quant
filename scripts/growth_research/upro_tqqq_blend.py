#!/usr/bin/env python3
"""
UPRO/TQQQ Blend Optimization
==============================
Tests optimal mix of UPRO (3x S&P) and TQQQ (3x Nasdaq) within vol-adjusted system.

The GAMEPLAN suggests 50/50 UPRO+TQQQ for Phase 3 ($10K+). Let's test:
1. UPRO-only baseline
2. TQQQ-only
3. Various UPRO/TQQQ splits (90/10 through 10/90)
4. Momentum-switched (hold whichever is trending better)
5. Vol-weighted (inverse vol allocation)
6. QQQ vs SPY relative strength switching
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/upro_tqqq_blend'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100


def download_data():
    tickers = ['SPY', 'QQQ', 'UPRO', 'TQQQ', 'GLD', 'TLT']
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
    closes = closes.dropna(how='all').dropna(subset=['SPY', 'UPRO', 'TQQQ'])
    print(f"  Data: {len(closes)} days")
    return closes


def simulate(closes, get_leverage_alloc, name=""):
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
            target = get_leverage_alloc(date, closes, i)
        elif vol < 0.30:
            regime = 'SPY'
            target = {'SPY': 1.0}
        else:
            regime = 'SAFE'
            target = {'GLD': 0.5, 'TLT': 0.5}

        if regime != last_regime:
            total_val = cash + sum(holdings.values())
            valid = {t: w for t, w in target.items() if t in closes.columns}
            total_w = sum(valid.values())
            if total_w > 0:
                valid = {t: w/total_w for t, w in valid.items()}
            else:
                valid = {'SPY': 1.0}
            holdings = {t: total_val * w for t, w in valid.items() if w > 0}
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
        'ann_vol': float(ann_vol * 100),
    }


def main():
    print("="*70)
    print("UPRO/TQQQ BLEND OPTIMIZATION")
    print("="*70)

    closes = download_data()

    # --- Correlation and performance ---
    print("\n  UPRO vs TQQQ COMPARISON:")
    for t in ['SPY', 'QQQ', 'UPRO', 'TQQQ']:
        if t in closes.columns:
            p = closes[t].dropna()
            if len(p) > 252:
                total_ret = p.iloc[-1] / p.iloc[0] - 1
                years = len(p) / 252
                cagr = (1 + total_ret) ** (1/years) - 1
                vol = p.pct_change().std() * np.sqrt(252)
                mdd = ((p - p.expanding().max()) / p.expanding().max()).min()
                print(f"    {t:<5s}: CAGR {cagr*100:>7.1f}%, Vol {vol*100:>5.1f}%, MaxDD {mdd*100:>6.1f}%")

    # Correlation
    corr = closes[['UPRO', 'TQQQ']].pct_change().dropna().corr()
    print(f"\n    UPRO↔TQQQ daily correlation: {corr.loc['UPRO', 'TQQQ']:.4f}")

    # Rolling correlation
    rolling_corr = closes['UPRO'].pct_change().rolling(252).corr(closes['TQQQ'].pct_change())
    print(f"    Rolling 1yr mean: {rolling_corr.dropna().mean():.4f}")
    print(f"    Rolling 1yr range: [{rolling_corr.dropna().min():.4f}, {rolling_corr.dropna().max():.4f}]")

    # --- Fixed blends ---
    strategies = {}

    for tqqq_pct in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100]:
        upro_w = (100 - tqqq_pct) / 100
        tqqq_w = tqqq_pct / 100
        name = f'{100-tqqq_pct}/{tqqq_pct} UPRO/TQQQ'
        if tqqq_pct == 0:
            name = 'UPRO only'
        elif tqqq_pct == 100:
            name = 'TQQQ only'

        alloc = {'UPRO': upro_w, 'TQQQ': tqqq_w}
        strategies[name] = lambda d, c, i, a=alloc: a

    # Momentum-switched
    def momentum_switch(d, c, i):
        if i >= 63:
            upro_mom = c['UPRO'].iloc[i] / c['UPRO'].iloc[i-63] - 1
            tqqq_mom = c['TQQQ'].iloc[i] / c['TQQQ'].iloc[i-63] - 1
            if not np.isnan(upro_mom) and not np.isnan(tqqq_mom):
                if tqqq_mom > upro_mom:
                    return {'TQQQ': 1.0}
                return {'UPRO': 1.0}
        return {'UPRO': 0.5, 'TQQQ': 0.5}
    strategies['Momentum (3mo)'] = momentum_switch

    # Short-term momentum (21d)
    def momentum_21d(d, c, i):
        if i >= 21:
            upro_mom = c['UPRO'].iloc[i] / c['UPRO'].iloc[i-21] - 1
            tqqq_mom = c['TQQQ'].iloc[i] / c['TQQQ'].iloc[i-21] - 1
            if not np.isnan(upro_mom) and not np.isnan(tqqq_mom):
                if tqqq_mom > upro_mom:
                    return {'TQQQ': 1.0}
                return {'UPRO': 1.0}
        return {'UPRO': 0.5, 'TQQQ': 0.5}
    strategies['Momentum (1mo)'] = momentum_21d

    # Inverse-vol weighted
    def inv_vol(d, c, i):
        if i >= 63:
            upro_vol = c['UPRO'].pct_change().iloc[i-63:i].std()
            tqqq_vol = c['TQQQ'].pct_change().iloc[i-63:i].std()
            if upro_vol > 0 and tqqq_vol > 0:
                inv_u = 1/upro_vol
                inv_t = 1/tqqq_vol
                total = inv_u + inv_t
                return {'UPRO': inv_u/total, 'TQQQ': inv_t/total}
        return {'UPRO': 0.5, 'TQQQ': 0.5}
    strategies['Inverse vol'] = inv_vol

    # QQQ > SPY relative strength
    def rel_strength(d, c, i):
        if i >= 63:
            spy_mom = c['SPY'].iloc[i] / c['SPY'].iloc[i-63] - 1
            qqq_mom = c['QQQ'].iloc[i] / c['QQQ'].iloc[i-63] - 1
            if not np.isnan(spy_mom) and not np.isnan(qqq_mom):
                if qqq_mom > spy_mom:
                    return {'TQQQ': 0.7, 'UPRO': 0.3}
                return {'UPRO': 0.7, 'TQQQ': 0.3}
        return {'UPRO': 0.5, 'TQQQ': 0.5}
    strategies['QQQ/SPY rel strength'] = rel_strength

    print(f"\nTesting {len(strategies)} strategies...")

    results = {}
    for name, alloc in strategies.items():
        portfolio, total = simulate(closes, alloc, name)
        m = compute_metrics(portfolio, total)
        if m:
            results[name] = m

    # --- Results ---
    sorted_by_value = sorted(results.items(), key=lambda x: x[1]['final_value'], reverse=True)
    sorted_by_sharpe = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)

    print("\n" + "="*70)
    print("RESULTS — RANKED BY FINAL VALUE")
    print("="*70)
    print(f"\n  {'Strategy':<28s} {'Final $':>10s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s} {'CAGR':>7s} {'Vol':>6s}")
    print("  " + "-"*76)
    for name, m in sorted_by_value:
        print(f"  {name:<28s} ${m['final_value']:>9,.0f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['max_dd']:>6.1f}% {m['cagr']:>6.1f}% {m['ann_vol']:>5.1f}%")

    print(f"\n  RANKED BY SHARPE:")
    for rank, (name, m) in enumerate(sorted_by_sharpe[:10], 1):
        print(f"    {rank}. {name:<26s}: Sharpe {m['sharpe']:.3f}, ${m['final_value']:,.0f}")

    baseline = results.get('UPRO only', {})
    if baseline:
        print(f"\n  vs UPRO-only:")
        for name, m in sorted_by_value:
            if name == 'UPRO only':
                continue
            val_diff = m['final_value'] - baseline['final_value']
            sharpe_diff = m['sharpe'] - baseline['sharpe']
            dd_diff = m['max_dd'] - baseline['max_dd']
            print(f"    {name:<26s}: value {val_diff:>+10,.0f} ({val_diff/baseline['final_value']*100:>+5.1f}%), "
                  f"Sharpe {sharpe_diff:+.3f}, MaxDD {dd_diff:+.1f}pp")

    # --- Efficient frontier ---
    print(f"\n  EFFICIENT FRONTIER (blends only):")
    blends = {k: v for k, v in results.items() if 'UPRO/TQQQ' in k or k in ['UPRO only', 'TQQQ only']}
    for name, m in sorted(blends.items(), key=lambda x: x[1]['sharpe'], reverse=True):
        print(f"    {name:<28s}: Sharpe {m['sharpe']:.3f} | ${m['final_value']:,.0f} | MaxDD {m['max_dd']:.1f}%")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'results': results,
    }
    with open(os.path.join(OUTPUT_DIR, 'blend_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
