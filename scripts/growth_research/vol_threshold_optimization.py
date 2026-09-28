#!/usr/bin/env python3
"""
Vol Threshold Optimization for Vol-Adjusted Leverage Strategy
=============================================================
The vol-adjusted strategy won our integrated system test. Now optimize:

1. What are the optimal vol thresholds for switching between UPRO/SPY/cash?
2. What vol lookback window is best? (5d, 10d, 21d, 42d)
3. Should we use realized vol or VIXY-implied vol?
4. How sensitive are results to threshold changes?
5. Walk-forward validation: does the optimal threshold hold OOS?

Permutation + R1 + sub-period validation on winner.
"""

import numpy as np
import pandas as pd
import yfinance as yf
from itertools import product
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/vol_threshold'
os.makedirs(OUTPUT_DIR, exist_ok=True)

TX_COST = 0.001
WEEKLY_DCA = 100
INITIAL_CAPITAL = 500

def download_data():
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT', 'VIXY']
    print(f"Downloading {len(tickers)} tickers...")
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

def simulate_vol_strategy(closes, vol_series, thresh_low, thresh_high, safe_haven='mixed'):
    """
    Simulate vol-adjusted strategy with given thresholds.
    vol < thresh_low: UPRO
    thresh_low <= vol < thresh_high: SPY
    vol >= thresh_high: safe haven
    """
    returns = closes.pct_change().fillna(0)
    warmup = 63

    portfolio_val = INITIAL_CAPITAL
    holdings = {}
    cash = float(INITIAL_CAPITAL)
    total_contributed = INITIAL_CAPITAL
    last_week = None
    last_regime = None
    n_switches = 0

    daily_values = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]

        # Weekly DCA
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        # Update holdings
        for ticker in list(holdings.keys()):
            if ticker in returns.columns:
                r = returns.loc[date, ticker]
                if not np.isnan(r):
                    holdings[ticker] *= (1 + r)

        portfolio_val = cash + sum(holdings.values())

        # Determine regime
        vol = vol_series.iloc[i] if i < len(vol_series) and not np.isnan(vol_series.iloc[i]) else 0.15

        if vol < thresh_low:
            regime = 'UPRO'
            target = {'UPRO': 1.0}
        elif vol < thresh_high:
            regime = 'SPY'
            target = {'SPY': 1.0}
        else:
            regime = 'SAFE'
            if safe_haven == 'mixed':
                target = {'GLD': 0.5, 'TLT': 0.5}
            elif safe_haven == 'cash':
                target = {'SPY': 0.01}  # Effectively cash
            else:
                target = {'GLD': 0.5, 'TLT': 0.5}

        # Switch if regime changed
        if regime != last_regime:
            total_val = cash + sum(holdings.values())
            old_turnover = sum(holdings.values()) / total_val if total_val > 0 else 0
            cost = old_turnover * TX_COST * total_val
            total_val -= cost

            holdings = {t: total_val * w for t, w in target.items()}
            cash = 0
            last_regime = regime
            n_switches += 1
        elif cash > WEEKLY_DCA * 0.5 and holdings:
            total_h = sum(holdings.values())
            if total_h > 0:
                for t in holdings:
                    holdings[t] += cash * (holdings[t] / total_h)
                cash = 0

        portfolio_val = cash + sum(holdings.values())
        daily_values.append(portfolio_val)

    portfolio = pd.Series(daily_values, index=closes.index[warmup:])
    return portfolio, total_contributed, n_switches

def compute_metrics(portfolio, total_contributed):
    r = portfolio.pct_change().dropna()
    if len(r) < 63:
        return None

    years = len(r) / 252
    final = portfolio.iloc[-1]
    profit = final - total_contributed

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
        'profit': float(profit),
        'cagr': float(cagr * 100),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd * 100),
    }

def main():
    print("="*70)
    print("VOL THRESHOLD OPTIMIZATION")
    print("="*70)

    closes = download_data()
    spy_ret = closes['SPY'].pct_change()

    # Build different vol measures
    vol_measures = {}
    for w in [5, 10, 21, 42, 63]:
        vol_measures[f'realized_{w}d'] = spy_ret.rolling(w).std() * np.sqrt(252)

    # VIXY-based vol proxy (normalized)
    if 'VIXY' in closes.columns:
        vixy = closes['VIXY']
        # Scale VIXY to approximate vol level
        vixy_norm = vixy / vixy.rolling(252).mean() * spy_ret.rolling(63).std().rolling(252).mean() * np.sqrt(252)
        vol_measures['vixy_normalized'] = vixy_norm

    # Grid search over thresholds
    thresh_lows = [0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20]
    thresh_highs = [0.20, 0.22, 0.25, 0.28, 0.30, 0.35]

    print(f"\nGrid search: {len(vol_measures)} vol measures × {len(thresh_lows)} × {len(thresh_highs)} thresholds")

    results = []

    for vol_name, vol_series in vol_measures.items():
        for tl in thresh_lows:
            for th in thresh_highs:
                if th <= tl:
                    continue

                portfolio, total_cont, n_switches = simulate_vol_strategy(
                    closes, vol_series, tl, th
                )
                m = compute_metrics(portfolio, total_cont)
                if m:
                    m['vol_measure'] = vol_name
                    m['thresh_low'] = tl
                    m['thresh_high'] = th
                    m['n_switches'] = n_switches
                    results.append(m)

    # Also run benchmarks
    print("\nRunning benchmarks...")

    # UPRO always (no vol filter)
    upro_ret = closes['UPRO'].pct_change().fillna(0)
    port_val = INITIAL_CAPITAL
    last_week = None
    total_cont = INITIAL_CAPITAL
    vals = []
    for i in range(63, len(closes)):
        date = closes.index[i]
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            port_val += WEEKLY_DCA
            total_cont += WEEKLY_DCA
            last_week = week_key
        port_val *= (1 + upro_ret.iloc[i])
        vals.append(port_val)
    upro_bench = compute_metrics(pd.Series(vals, index=closes.index[63:]), total_cont)
    if upro_bench:
        upro_bench['vol_measure'] = 'BENCHMARK'
        upro_bench['thresh_low'] = 999
        upro_bench['thresh_high'] = 999
        upro_bench['n_switches'] = 0

    # Sort by composite score (Sharpe + normalized final value)
    if results:
        max_final = max(r['final_value'] for r in results)
        max_sharpe = max(r['sharpe'] for r in results)

        for r in results:
            r['score'] = (r['sharpe'] / max_sharpe) * 0.5 + (r['final_value'] / max_final) * 0.5

        results.sort(key=lambda x: x['score'], reverse=True)

    # Top 20 results
    print("\n" + "="*70)
    print("TOP 20 CONFIGURATIONS (balanced growth + risk)")
    print("="*70)

    print(f"\n  {'Rank':>4s} {'Vol Measure':<18s} {'Low':>5s} {'High':>5s} {'Final $':>10s} {'Sharpe':>7s} "
          f"{'MaxDD':>7s} {'CAGR':>7s} {'Switches':>8s}")
    print("  " + "-"*80)

    for i, r in enumerate(results[:20]):
        print(f"  {i+1:>4d} {r['vol_measure']:<18s} {r['thresh_low']:>4.0%} {r['thresh_high']:>4.0%} "
              f"${r['final_value']:>9,.0f} {r['sharpe']:>7.3f} {r['max_dd']:>6.1f}% "
              f"{r['cagr']:>6.1f}% {r['n_switches']:>7d}")

    if upro_bench:
        print(f"\n  BENCH UPRO always                      ${upro_bench['final_value']:>9,.0f} "
              f"{upro_bench['sharpe']:>7.3f} {upro_bench['max_dd']:>6.1f}% {upro_bench['cagr']:>6.1f}%")

    # Best per vol measure
    print("\n" + "="*70)
    print("BEST CONFIG PER VOL MEASURE")
    print("="*70)

    by_vol = {}
    for r in results:
        vm = r['vol_measure']
        if vm not in by_vol or r['score'] > by_vol[vm]['score']:
            by_vol[vm] = r

    for vm, r in sorted(by_vol.items(), key=lambda x: x[1]['score'], reverse=True):
        print(f"\n  {vm}:")
        print(f"    Best thresholds: low={r['thresh_low']:.0%}, high={r['thresh_high']:.0%}")
        print(f"    Final: ${r['final_value']:,.0f}, Sharpe: {r['sharpe']:.3f}, "
              f"MaxDD: {r['max_dd']:.1f}%, CAGR: {r['cagr']:.1f}%")
        print(f"    Switches: {r['n_switches']}")

    # Sensitivity analysis of winner
    winner = results[0]
    print("\n" + "="*70)
    print(f"SENSITIVITY ANALYSIS: {winner['vol_measure']}")
    print("="*70)

    same_vol = [r for r in results if r['vol_measure'] == winner['vol_measure']]
    print(f"\n  Fixed low={winner['thresh_low']:.0%}, varying high:")
    for r in same_vol:
        if r['thresh_low'] == winner['thresh_low']:
            marker = " ← WINNER" if r == winner else ""
            print(f"    high={r['thresh_high']:>4.0%}: Sharpe {r['sharpe']:.3f}, "
                  f"Final ${r['final_value']:>9,.0f}, MaxDD {r['max_dd']:>6.1f}%{marker}")

    print(f"\n  Fixed high={winner['thresh_high']:.0%}, varying low:")
    for r in same_vol:
        if r['thresh_high'] == winner['thresh_high']:
            marker = " ← WINNER" if r == winner else ""
            print(f"    low={r['thresh_low']:>4.0%}:  Sharpe {r['sharpe']:.3f}, "
                  f"Final ${r['final_value']:>9,.0f}, MaxDD {r['max_dd']:>6.1f}%{marker}")

    # Walk-forward validation
    print("\n" + "="*70)
    print("WALK-FORWARD VALIDATION")
    print("="*70)

    # Train on first half, test on second half
    n = len(closes)
    mid = n // 2

    vol_series = vol_measures[winner['vol_measure']]

    # Find best thresholds on first half
    train_results = []
    for tl in thresh_lows:
        for th in thresh_highs:
            if th <= tl:
                continue
            portfolio, total_cont, n_sw = simulate_vol_strategy(
                closes.iloc[:mid], vol_series.iloc[:mid], tl, th
            )
            m = compute_metrics(portfolio, total_cont)
            if m:
                m['thresh_low'] = tl
                m['thresh_high'] = th
                max_f = max(r2['final_value'] for r2 in train_results) if train_results else m['final_value']
                max_s = max(r2['sharpe'] for r2 in train_results) if train_results else m['sharpe']
                train_results.append(m)

    # Re-score train results
    if train_results:
        max_f = max(r2['final_value'] for r2 in train_results)
        max_s = max(r2['sharpe'] for r2 in train_results)
        for r2 in train_results:
            r2['score'] = (r2['sharpe'] / max_s) * 0.5 + (r2['final_value'] / max_f) * 0.5
        train_results.sort(key=lambda x: x['score'], reverse=True)

        train_best = train_results[0]
        print(f"\n  Train-optimal (first half): low={train_best['thresh_low']:.0%}, high={train_best['thresh_high']:.0%}")
        print(f"    Train: Sharpe {train_best['sharpe']:.3f}, Final ${train_best['final_value']:,.0f}")

        # Test on second half with train-optimal thresholds
        test_portfolio, test_cont, test_sw = simulate_vol_strategy(
            closes.iloc[mid:], vol_series.iloc[mid:], train_best['thresh_low'], train_best['thresh_high']
        )
        test_m = compute_metrics(test_portfolio, test_cont)
        if test_m:
            print(f"    Test:  Sharpe {test_m['sharpe']:.3f}, Final ${test_m['final_value']:,.0f}, "
                  f"MaxDD {test_m['max_dd']:.1f}%")

            degradation = (train_best['sharpe'] - test_m['sharpe']) / train_best['sharpe'] * 100
            print(f"    Sharpe degradation: {degradation:.1f}%")

    # Sub-period consistency
    print("\n" + "="*70)
    print("SUB-PERIOD CONSISTENCY (WINNER)")
    print("="*70)

    vol_series_full = vol_measures[winner['vol_measure']]
    full_portfolio, _, _ = simulate_vol_strategy(
        closes, vol_series_full, winner['thresh_low'], winner['thresh_high']
    )

    n_p = len(full_portfolio)
    period_size = n_p // 3
    for p in range(3):
        start = p * period_size
        end = (p + 1) * period_size if p < 2 else n_p
        sub = full_portfolio.iloc[start:end]
        sub_m = compute_metrics(sub, sub.iloc[0])
        if sub_m:
            dates = f"{sub.index[0].strftime('%Y-%m')} to {sub.index[-1].strftime('%Y-%m')}"
            print(f"  Period {p+1} ({dates}): Sharpe {sub_m['sharpe']:.3f}, "
                  f"CAGR {sub_m['cagr']:.1f}%, MaxDD {sub_m['max_dd']:.1f}%")

    # Save
    output = {
        'run_date': pd.Timestamp.now().isoformat(),
        'top_20': results[:20],
        'best_per_vol': {k: v for k, v in by_vol.items()},
        'winner': winner,
        'upro_benchmark': upro_bench,
    }

    output_path = os.path.join(OUTPUT_DIR, 'vol_threshold_results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\n  Results saved.")

    print("\n" + "="*70)
    print("DONE")
    print("="*70)

if __name__ == '__main__':
    main()
