#!/usr/bin/env python3
"""
SMA Protection Lookback Optimization
======================================
The current system uses SPY > 50-day SMA as the protection overlay.
Is 50 optimal, or would a different lookback work better?

Tests SMA periods: 20, 30, 40, 50, 60, 80, 100, 150, 200, 250
Also tests:
- EMA vs SMA
- Dual SMA (fast/slow crossover)
- SMA + buffer zone (require SPY > SMA*(1+buffer))
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/sma_lookback'
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
    closes = closes.dropna(how='all').dropna(subset=['SPY', 'UPRO'])
    print(f"  Data: {len(closes)} days")
    return closes


def simulate(closes, protection_func, name=""):
    spy = closes['SPY']
    spy_ret = spy.pct_change()
    vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
    returns = closes.pct_change().fillna(0)

    warmup = 260  # Max lookback
    holdings = {}
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    switches = 0

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
        protection = protection_func(spy, i)

        if not protection:
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
            target = {'GLD': 0.5, 'TLT': 0.5}

        if regime != last_regime:
            total_val = cash + sum(holdings.values())
            holdings = {t: total_val * w for t, w in target.items()
                       if w > 0 and t in closes.columns}
            cash = 0
            last_regime = regime
            switches += 1
        elif cash > 50 and holdings:
            total_h = sum(holdings.values())
            if total_h > 0:
                for t in holdings:
                    holdings[t] += cash * (holdings[t] / total_h)
                cash = 0

        portfolio_val = cash + sum(holdings.values())
        daily_values.append(portfolio_val)

    portfolio = pd.Series(daily_values, index=closes.index[warmup:])
    return portfolio, total_contributed, switches


def compute_metrics(portfolio, total_contributed, switches):
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
        'switches': switches,
        'switches_per_year': switches / years,
    }


def main():
    print("="*70)
    print("SMA PROTECTION LOOKBACK OPTIMIZATION")
    print("="*70)

    closes = download_data()
    spy = closes['SPY']

    # === TEST 1: SMA Period ===
    print("\n  TEST 1: SMA LOOKBACK PERIOD")
    print("  " + "-"*60)
    sma_periods = [20, 30, 40, 50, 60, 80, 100, 150, 200, 250]
    sma_results = {}

    for period in sma_periods:
        sma = spy.rolling(period).mean()
        prot_func = lambda s, i, sma=sma: s.iloc[i] > sma.iloc[i] if not np.isnan(sma.iloc[i]) else True
        portfolio, total, switches = simulate(closes, prot_func, f"SMA{period}")
        m = compute_metrics(portfolio, total, switches)
        if m:
            sma_results[f'SMA{period}'] = m
            print(f"    SMA{period:<4d}: ${m['final_value']:>9,.0f} | Sharpe {m['sharpe']:.3f} | "
                  f"MaxDD {m['max_dd']:.1f}% | Switches {m['switches']}")

    # No protection baseline
    no_prot = lambda s, i: True
    portfolio, total, switches = simulate(closes, no_prot, "No Protection")
    m = compute_metrics(portfolio, total, switches)
    if m:
        sma_results['No Protection'] = m
        print(f"    No Prot: ${m['final_value']:>9,.0f} | Sharpe {m['sharpe']:.3f} | "
              f"MaxDD {m['max_dd']:.1f}% | Switches {m['switches']}")

    # === TEST 2: EMA vs SMA ===
    print("\n  TEST 2: EMA vs SMA (at key periods)")
    print("  " + "-"*60)
    ema_results = {}

    for period in [50, 100, 200]:
        # EMA
        ema = spy.ewm(span=period, adjust=False).mean()
        prot_func = lambda s, i, ema=ema: s.iloc[i] > ema.iloc[i] if not np.isnan(ema.iloc[i]) else True
        portfolio, total, switches = simulate(closes, prot_func, f"EMA{period}")
        m = compute_metrics(portfolio, total, switches)
        if m:
            ema_results[f'EMA{period}'] = m
            # Get SMA equivalent
            sma_m = sma_results.get(f'SMA{period}', {})
            diff = m['final_value'] - sma_m.get('final_value', 0)
            print(f"    EMA{period:<4d}: ${m['final_value']:>9,.0f} | Sharpe {m['sharpe']:.3f} | "
                  f"MaxDD {m['max_dd']:.1f}% | vs SMA: {'+' if diff > 0 else ''}{diff:,.0f}")

    # === TEST 3: Dual SMA crossover ===
    print("\n  TEST 3: DUAL SMA CROSSOVER (SPY fast > slow)")
    print("  " + "-"*60)
    cross_results = {}

    pairs = [(20, 50), (20, 100), (50, 100), (50, 200), (20, 200)]
    for fast, slow in pairs:
        sma_fast = spy.rolling(fast).mean()
        sma_slow = spy.rolling(slow).mean()
        prot_func = lambda s, i, f=sma_fast, sl=sma_slow: (
            f.iloc[i] > sl.iloc[i] if not np.isnan(f.iloc[i]) and not np.isnan(sl.iloc[i]) else True
        )
        portfolio, total, switches = simulate(closes, prot_func, f"{fast}/{slow}")
        m = compute_metrics(portfolio, total, switches)
        if m:
            cross_results[f'{fast}/{slow}'] = m
            print(f"    {fast:>3d}/{slow:<3d}: ${m['final_value']:>9,.0f} | Sharpe {m['sharpe']:.3f} | "
                  f"MaxDD {m['max_dd']:.1f}% | Switches {m['switches']}")

    # === TEST 4: SMA + Buffer ===
    print("\n  TEST 4: SMA50 WITH BUFFER (SPY > SMA50 * (1+buffer))")
    print("  " + "-"*60)
    buffer_results = {}
    sma50 = spy.rolling(50).mean()

    for buffer in [0, 0.01, 0.02, 0.03, 0.05]:
        prot_func = lambda s, i, b=buffer: (
            s.iloc[i] > sma50.iloc[i] * (1 + b) if not np.isnan(sma50.iloc[i]) else True
        )
        portfolio, total, switches = simulate(closes, prot_func, f"Buffer {buffer*100:.0f}%")
        m = compute_metrics(portfolio, total, switches)
        if m:
            buffer_results[f'Buffer {buffer*100:.0f}%'] = m
            print(f"    Buffer {buffer*100:>2.0f}%: ${m['final_value']:>9,.0f} | Sharpe {m['sharpe']:.3f} | "
                  f"MaxDD {m['max_dd']:.1f}% | Switches {m['switches']}")

    # === TEST 5: Combined SMA + vol-only (no SMA) ===
    print("\n  TEST 5: PROTECTION METHOD COMPARISON")
    print("  " + "-"*60)

    # Vol-only (no SMA protection)
    portfolio, total, switches = simulate(closes, lambda s, i: True, "Vol-only")
    m_vol_only = compute_metrics(portfolio, total, switches)
    if m_vol_only:
        print(f"    Vol-only (no SMA): ${m_vol_only['final_value']:>9,.0f} | Sharpe {m_vol_only['sharpe']:.3f} | "
              f"MaxDD {m_vol_only['max_dd']:.1f}%")

    # SMA200 + SMA50 (must be above both)
    sma200 = spy.rolling(200).mean()
    dual_func = lambda s, i: (
        s.iloc[i] > sma50.iloc[i] and s.iloc[i] > sma200.iloc[i]
        if not np.isnan(sma50.iloc[i]) and not np.isnan(sma200.iloc[i]) else True
    )
    portfolio, total, switches = simulate(closes, dual_func, "SMA50 AND SMA200")
    m_dual = compute_metrics(portfolio, total, switches)
    if m_dual:
        print(f"    SMA50 AND SMA200:  ${m_dual['final_value']:>9,.0f} | Sharpe {m_dual['sharpe']:.3f} | "
              f"MaxDD {m_dual['max_dd']:.1f}%")

    # SMA50 OR SMA200 (above either)
    either_func = lambda s, i: (
        s.iloc[i] > sma50.iloc[i] or s.iloc[i] > sma200.iloc[i]
        if not np.isnan(sma50.iloc[i]) and not np.isnan(sma200.iloc[i]) else True
    )
    portfolio, total, switches = simulate(closes, either_func, "SMA50 OR SMA200")
    m_either = compute_metrics(portfolio, total, switches)
    if m_either:
        print(f"    SMA50 OR SMA200:   ${m_either['final_value']:>9,.0f} | Sharpe {m_either['sharpe']:.3f} | "
              f"MaxDD {m_either['max_dd']:.1f}%")

    # === SUMMARY ===
    print("\n" + "="*70)
    print("OVERALL RANKING (by Sharpe)")
    print("="*70)

    all_results = {}
    all_results.update(sma_results)
    all_results.update({f'EMA: {k}': v for k, v in ema_results.items()})
    all_results.update({f'Cross: {k}': v for k, v in cross_results.items()})
    all_results.update({f'SMA50+{k}': v for k, v in buffer_results.items()})

    sorted_all = sorted(all_results.items(), key=lambda x: x[1]['sharpe'], reverse=True)

    print(f"\n  {'Config':<25s} {'Final $':>10s} {'Sharpe':>7s} {'Sortino':>8s} {'MaxDD':>7s} {'SW/yr':>6s}")
    print("  " + "-"*66)
    for name, m in sorted_all[:15]:
        print(f"  {name:<25s} ${m['final_value']:>9,.0f} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['max_dd']:>6.1f}% {m['switches_per_year']:>5.1f}")

    # vs SMA50 baseline
    baseline = sma_results.get('SMA50', {})
    if baseline:
        print(f"\n  vs SMA50 baseline (${baseline['final_value']:,.0f}, Sharpe {baseline['sharpe']:.3f}):")
        for name, m in sorted_all[:10]:
            if name == 'SMA50':
                continue
            val_diff = m['final_value'] - baseline['final_value']
            sharpe_diff = m['sharpe'] - baseline['sharpe']
            dd_diff = m['max_dd'] - baseline['max_dd']
            print(f"    {name:<23s}: value {val_diff:>+10,.0f}, Sharpe {sharpe_diff:+.3f}, MaxDD {dd_diff:+.1f}pp")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'sma_results': sma_results,
        'ema_results': ema_results,
        'crossover_results': cross_results,
        'buffer_results': buffer_results,
    }
    with open(os.path.join(OUTPUT_DIR, 'sma_optimization.json'), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print("\n" + "="*70)
    print("DONE")
    print("="*70)


if __name__ == '__main__':
    main()
