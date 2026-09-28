#!/usr/bin/env python3
"""
Safe Haven Optimization for Vol-Adjusted System
=================================================
The current system uses GLD+TLT (50/50) as the safe-haven allocation
when vol > 30%. Is this optimal?

Tests:
1. GLD + TLT 50/50 (current baseline)
2. GLD + SHY (short-term treasuries)
3. GLD + BIL (T-bills)
4. GLD + TIP (TIPS)
5. GLD only
6. TLT only
7. Cash only (money market)
8. GLD + TLT + SHY (1/3 each)
9. Dynamic: TLT when rates falling, SHY when rates rising
10. Inverse-vol weighted GLD+TLT

Also tests the MEDIUM vol allocation (currently SPY only when vol 20-30%):
11. SPY + GLD (90/10) in medium vol
12. SPY + TLT (90/10) in medium vol
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/safe_haven'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100


def download_data():
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT', 'SHY', 'BIL', 'TIP', 'IEF', 'AGG']
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


def simulate_strategy(closes, safe_haven_alloc, med_vol_alloc=None, name=""):
    """Simulate vol-adjusted strategy with custom safe haven allocation."""
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
    regime_days = {'UPRO': 0, 'SPY': 0, 'SAFE': 0, 'CASH': 0}

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
            target = {'UPRO': 1.0}
        elif vol < 0.30:
            regime = 'SPY'
            if med_vol_alloc:
                target = med_vol_alloc(date, closes, i)
            else:
                target = {'SPY': 1.0}
        else:
            regime = 'SAFE'
            target = safe_haven_alloc(date, closes, i)

        regime_days[regime] += 1

        if regime != last_regime and target:
            total_val = cash + sum(holdings.values())
            valid = {t: w for t, w in target.items()
                     if t in closes.columns and not np.isnan(closes[t].iloc[i])}
            if not valid:
                valid = {'SPY': 1.0}
            total_w = sum(valid.values())
            valid = {t: w/total_w for t, w in valid.items()}

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

    return pd.Series(daily_values, index=closes.index[warmup:]), total_contributed, regime_days


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
    print("SAFE HAVEN OPTIMIZATION — VOL-ADJUSTED SYSTEM")
    print("="*70)

    closes = download_data()

    # --- Safe haven performance during high-vol periods ---
    print("\n  SAFE HAVEN BEHAVIOR DURING HIGH-VOL (SPY vol > 30%):")
    spy_ret = closes['SPY'].pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    high_vol_mask = vol_21d > 0.30

    haven_tickers = ['GLD', 'TLT', 'SHY', 'BIL', 'TIP', 'IEF', 'AGG']
    for t in haven_tickers:
        if t in closes.columns:
            t_ret = closes[t].pct_change()
            hv_ret = t_ret[high_vol_mask & t_ret.notna()]
            if len(hv_ret) > 10:
                ann_r = hv_ret.mean() * 252
                ann_v = hv_ret.std() * np.sqrt(252)
                sr = ann_r / ann_v if ann_v > 0 else 0
                spy_corr = t_ret.rolling(63).corr(spy_ret)
                hv_corr = spy_corr[high_vol_mask].dropna().mean()
                print(f"    {t:<5s}: Ann ret {ann_r*100:>7.1f}%, Vol {ann_v*100:>5.1f}%, "
                      f"Sharpe {sr:>5.2f}, SPY corr {hv_corr:>6.3f}")

    # --- Strategies ---
    strategies = {}

    # 1. Current baseline
    strategies['1. GLD+TLT 50/50 (current)'] = (
        lambda d, c, i: {'GLD': 0.5, 'TLT': 0.5},
        None
    )

    # 2. GLD + SHY
    strategies['2. GLD+SHY 50/50'] = (
        lambda d, c, i: {'GLD': 0.5, 'SHY': 0.5},
        None
    )

    # 3. GLD + BIL (T-bills)
    strategies['3. GLD+BIL 50/50'] = (
        lambda d, c, i: {'GLD': 0.5, 'BIL': 0.5},
        None
    )

    # 4. GLD + TIP
    strategies['4. GLD+TIP 50/50'] = (
        lambda d, c, i: {'GLD': 0.5, 'TIP': 0.5},
        None
    )

    # 5. GLD only
    strategies['5. GLD only'] = (
        lambda d, c, i: {'GLD': 1.0},
        None
    )

    # 6. TLT only
    strategies['6. TLT only'] = (
        lambda d, c, i: {'TLT': 1.0},
        None
    )

    # 7. Cash only (SHY as proxy)
    strategies['7. Cash (SHY)'] = (
        lambda d, c, i: {'SHY': 1.0},
        None
    )

    # 8. Triple split
    strategies['8. GLD+TLT+SHY 1/3'] = (
        lambda d, c, i: {'GLD': 0.333, 'TLT': 0.333, 'SHY': 0.334},
        None
    )

    # 9. Dynamic: TLT when rates falling (TLT trending up), SHY otherwise
    def dynamic_bond(d, c, i):
        if 'TLT' in c.columns and i >= 50:
            tlt_sma = c['TLT'].iloc[i-50:i].mean()
            tlt_price = c['TLT'].iloc[i]
            if not np.isnan(tlt_price) and not np.isnan(tlt_sma):
                if tlt_price > tlt_sma:
                    return {'GLD': 0.5, 'TLT': 0.5}
                else:
                    return {'GLD': 0.5, 'SHY': 0.5}
        return {'GLD': 0.5, 'TLT': 0.5}
    strategies['9. Dynamic GLD+TLT/SHY'] = (dynamic_bond, None)

    # 10. Inverse-vol weighted
    def inv_vol_haven(d, c, i):
        vols = {}
        for t in ['GLD', 'TLT']:
            if t in c.columns and i >= 63:
                t_vol = c[t].pct_change().iloc[i-63:i].std() * np.sqrt(252)
                if not np.isnan(t_vol) and t_vol > 0:
                    vols[t] = t_vol
        if len(vols) >= 2:
            inv = {t: 1/v for t, v in vols.items()}
            total = sum(inv.values())
            return {t: w/total for t, w in inv.items()}
        return {'GLD': 0.5, 'TLT': 0.5}
    strategies['10. Inv-vol GLD+TLT'] = (inv_vol_haven, None)

    # 11. SPY + GLD in medium vol
    strategies['11. SPY+GLD 90/10 (med vol)'] = (
        lambda d, c, i: {'GLD': 0.5, 'TLT': 0.5},  # high vol same
        lambda d, c, i: {'SPY': 0.90, 'GLD': 0.10}   # med vol different
    )

    # 12. SPY + TLT in medium vol
    strategies['12. SPY+TLT 90/10 (med vol)'] = (
        lambda d, c, i: {'GLD': 0.5, 'TLT': 0.5},
        lambda d, c, i: {'SPY': 0.90, 'TLT': 0.10}
    )

    # 13. AGG (total bond market)
    strategies['13. GLD+AGG 50/50'] = (
        lambda d, c, i: {'GLD': 0.5, 'AGG': 0.5},
        None
    )

    print(f"\nTesting {len(strategies)} strategies...")

    results = {}
    for name, (safe_alloc, med_alloc) in strategies.items():
        print(f"  Running: {name}...", end=" ", flush=True)
        portfolio, total_cont, regime_days = simulate_strategy(
            closes, safe_alloc, med_alloc, name)
        m = compute_metrics(portfolio, total_cont)
        if m:
            m['safe_days'] = regime_days['SAFE']
            m['total_days'] = sum(regime_days.values())
            m['safe_pct'] = regime_days['SAFE'] / sum(regime_days.values()) * 100
            results[name] = m
            print(f"${m['final_value']:,.0f} | Sharpe {m['sharpe']:.3f} | MaxDD {m['max_dd']:.1f}%")

    # --- Results ---
    sorted_results = sorted(results.items(), key=lambda x: x[1]['sharpe'], reverse=True)

    print("\n" + "="*70)
    print("RESULTS — RANKED BY SHARPE")
    print("="*70)

    print(f"\n  {'Strategy':<33s} {'Final $':>10s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s} {'CAGR':>7s}")
    print("  " + "-"*74)
    for name, m in sorted_results:
        print(f"  {name:<33s} ${m['final_value']:>9,.0f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['max_dd']:>6.1f}% {m['cagr']:>6.1f}%")

    baseline = results.get('1. GLD+TLT 50/50 (current)', {})
    if baseline:
        print(f"\n  vs current (GLD+TLT 50/50):")
        for name, m in sorted_results:
            if '1. GLD+TLT' in name:
                continue
            val_diff = m['final_value'] - baseline['final_value']
            sharpe_diff = m['sharpe'] - baseline['sharpe']
            dd_diff = m['max_dd'] - baseline['max_dd']
            print(f"    {name:<31s}: value {val_diff:>+10,.0f}, "
                  f"Sharpe {sharpe_diff:+.3f}, MaxDD {dd_diff:+.1f}pp")

    # How much time in each regime
    if results:
        sample = list(results.values())[0]
        print(f"\n  Time in SAFE haven regime: {sample['safe_pct']:.1f}% of days ({sample['safe_days']} days)")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'results': results,
    }
    with open(os.path.join(OUTPUT_DIR, 'safe_haven_results.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
