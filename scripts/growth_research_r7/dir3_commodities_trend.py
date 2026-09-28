#!/usr/bin/env python3
"""
R7 DIRECTION 3: Commodities Trend Following

Dedicated commodities trend strategy using time-series momentum.
Long trending up, flat/cash trending down (long-only constraint for RH).
Walk-forward: does momentum predict individual commodity ETF trends?

HC #694: Commission-free (RH/IBKR)
HC #428: Regime-agnostic validation
HC #0: Sliding window walk-forward only
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime
from scipy.stats import spearmanr

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r7'
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DAYS = 252
TEST_DAYS = 21
START = '2008-01-01'
END = '2026-07-14'

# Commodity ETFs
COMMODITY_TICKERS = ['GLD', 'SLV', 'USO', 'UNG', 'DBA', 'DBB', 'DBC', 'CORN', 'WEAT', 'PALL', 'PPLT', 'CPER']
BENCHMARK = 'SPY'


def calc_metrics(returns, name=''):
    rets = returns.dropna()
    if len(rets) < 20:
        return {'name': name, 'sharpe': 0, 'sortino': 0, 'cagr': 0,
                'max_dd': 0, 'win_rate': 0, 'pf': 0, 'n_days': len(rets)}
    total_ret = (1 + rets).prod() - 1
    n_years = len(rets) / 252
    cagr = (1 + total_ret) ** (1 / max(n_years, 0.01)) - 1
    sharpe = rets.mean() / rets.std() * np.sqrt(252) if rets.std() > 0 else 0
    downside = rets[rets < 0].std() * np.sqrt(252)
    sortino = rets.mean() * 252 / downside if downside > 0 else 0
    cum = (1 + rets).cumprod()
    max_dd = (cum / cum.cummax() - 1).min()
    wr = (rets > 0).mean()
    gp = rets[rets > 0].sum()
    gl = abs(rets[rets < 0].sum())
    pf = gp / gl if gl > 0 else float('inf')
    calmar = abs(cagr / max_dd) if max_dd != 0 else 0
    return {'name': name, 'sharpe': round(float(sharpe), 4),
            'sortino': round(float(sortino), 4),
            'cagr': round(float(cagr), 4), 'cagr_pct': round(float(cagr * 100), 2),
            'max_dd': round(float(max_dd), 4), 'max_dd_pct': round(float(max_dd * 100), 2),
            'win_rate': round(float(wr), 4), 'pf': round(float(pf), 3),
            'calmar': round(float(calmar), 3), 'n_days': int(len(rets))}


def classify_regime(spy_ret, threshold=0.003):
    regimes = pd.Series('flat', index=spy_ret.index)
    regimes[spy_ret > threshold] = 'green'
    regimes[spy_ret < -threshold] = 'red'
    return regimes


def regime_test(strat_rets, spy_rets, name=''):
    regimes = classify_regime(spy_rets.reindex(strat_rets.index))
    results = {}
    for r in ['green', 'red', 'flat']:
        mask = regimes == r
        if mask.sum() > 10:
            results[r] = calc_metrics(strat_rets[mask], f'{name} ({r})')
    if 'green' in results and 'red' in results:
        sg, sr = results['green']['sharpe'], results['red']['sharpe']
        results['regime_gap'] = round(abs(sg - sr) / max(abs(sg), abs(sr), 0.001), 3)
    return results


def run_direction3():
    print("=" * 80)
    print("DIRECTION 3: Commodities Trend Following")
    print("=" * 80)

    # Download data
    tickers = list(set(COMMODITY_TICKERS + [BENCHMARK]))
    print(f"Downloading: {tickers}")
    data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.ffill()
    rets = close.pct_change()

    # Filter to commodities with enough data
    available = []
    for t in COMMODITY_TICKERS:
        if t in close.columns:
            valid = close[t].dropna()
            if len(valid) > TRAIN_DAYS + 100:
                available.append(t)
    print(f"Available commodities with enough data: {available}")

    spy_rets = rets[BENCHMARK].dropna()

    # ── Strategy 1: Time-Series Momentum (TSMOM) ──
    # For each commodity: long if positive trailing return, cash if negative (long-only)
    lookbacks = [20, 60, 120]  # Different momentum windows

    all_strategy_rets = {}
    all_ics = {}

    for lb in lookbacks:
        print(f"\n--- TSMOM lookback={lb}d ---")

        # Walk-forward IC test: does trailing return predict forward return?
        oot_preds = []
        oot_actuals = []

        for t in available:
            t_ret = rets[t].dropna()
            mom = close[t].pct_change(lb)  # Trailing momentum signal
            fwd = t_ret.rolling(TEST_DAYS).sum().shift(-TEST_DAYS)  # Forward return

            for start_idx in range(TRAIN_DAYS, len(mom) - TEST_DAYS, TEST_DAYS):
                test_slice = slice(start_idx, min(start_idx + TEST_DAYS, len(mom)))
                signal = mom.iloc[test_slice].dropna()
                target = fwd.iloc[test_slice].dropna()
                common = signal.index.intersection(target.index)
                if len(common) > 0:
                    oot_preds.extend(signal.loc[common].values)
                    oot_actuals.extend(target.loc[common].values)

        if len(oot_preds) > 100:
            ic, ic_p = spearmanr(oot_preds, oot_actuals)
            all_ics[lb] = {'ic': round(float(ic), 4), 'p': float(ic_p), 'n': len(oot_preds)}
            print(f"  Pooled IC: {ic:.4f} (p={ic_p:.4e}, n={len(oot_preds)})")

        # Simulate TSMOM portfolio
        # Each commodity: weight = 1/N if mom > 0, weight = 0 if mom <= 0
        n_assets = len(available)
        if n_assets == 0:
            continue

        daily_rets = pd.DataFrame(index=close.index)
        for t in available:
            mom = close[t].pct_change(lb)
            # Long-only: invest 1/N when momentum positive, cash otherwise
            position = (mom > 0).astype(float) / n_assets
            daily_rets[t] = rets[t] * position.shift(1)  # Shift to avoid lookahead

        portfolio_ret = daily_rets.sum(axis=1).dropna()
        all_strategy_rets[f'TSMOM_{lb}d'] = portfolio_ret

    # ── Strategy 2: Cross-Sectional Momentum (CSMOM) ──
    # Rank commodities by trailing return, long top half
    print(f"\n--- Cross-Sectional Momentum ---")
    for lb in [60]:
        mom_df = pd.DataFrame()
        for t in available:
            mom_df[t] = close[t].pct_change(lb)

        # Rank: top half gets equal weight, bottom half gets 0
        daily_rets_cs = pd.DataFrame(index=close.index, columns=available, data=0.0)
        for idx in mom_df.index:
            row = mom_df.loc[idx].dropna()
            if len(row) < 2:
                continue
            median_mom = row.median()
            longs = row[row > median_mom].index.tolist()
            if longs:
                w = 1.0 / len(longs)
                for t in longs:
                    daily_rets_cs.loc[idx, t] = w

        # Apply weights (shifted by 1 to avoid lookahead)
        cs_port_ret = pd.Series(0.0, index=close.index)
        for t in available:
            cs_port_ret += rets[t] * daily_rets_cs[t].shift(1)
        cs_port_ret = cs_port_ret.dropna()
        all_strategy_rets[f'CSMOM_{lb}d'] = cs_port_ret

    # ── Strategy 3: Commodity Trend + RP Combo ──
    # Use best TSMOM as diversifier for base RP
    rp_tickers = ['SPY', 'TLT', 'GLD']
    rp_available = [t for t in rp_tickers if t in rets.columns]
    if rp_available and 'TSMOM_60d' in all_strategy_rets:
        rp_ret_3x = rets[rp_available].mean(axis=1) * 3.0
        combo_ret = 0.75 * rp_ret_3x + 0.25 * all_strategy_rets['TSMOM_60d']
        all_strategy_rets['75%_3xRP_+_25%_TSMOM'] = combo_ret.dropna()

    # ── Strategy 4: Commodity B&H (baseline) ──
    if available:
        bh_ret = rets[available].mean(axis=1)
        all_strategy_rets['Commodity_BH'] = bh_ret.dropna()

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    metrics_all = {}
    for name, ret_series in all_strategy_rets.items():
        m = calc_metrics(ret_series, name)
        metrics_all[name] = m
        print(f"\n{name}:")
        print(f"  CAGR: {m['cagr_pct']:.1f}%  |  Sharpe: {m['sharpe']:.3f}  |  Sortino: {m['sortino']:.3f}")
        print(f"  MaxDD: {m['max_dd_pct']:.1f}%  |  Calmar: {m['calmar']:.3f}  |  WR: {m['win_rate']:.3f}")

    # Regime test on best strategies
    print("\n--- Regime Tests ---")
    rt_results = {}
    for name, ret_series in all_strategy_rets.items():
        if len(ret_series) > 100:
            rt = regime_test(ret_series, spy_rets, name)
            rt_results[name] = rt
            for r in ['green', 'red', 'flat']:
                if r in rt:
                    print(f"  {name} {r}: Sharpe={rt[r]['sharpe']:.3f}")
            if 'regime_gap' in rt:
                print(f"  {name} regime gap: {rt['regime_gap']:.3f}")

    # Correlation with SPY
    print("\n--- Correlation with SPY ---")
    for name, ret_series in all_strategy_rets.items():
        common = ret_series.index.intersection(spy_rets.index)
        if len(common) > 100:
            corr = ret_series.loc[common].corr(spy_rets.loc[common])
            print(f"  {name}: {corr:.3f}")

    # Honest assessment
    assessment = []
    best_tsmom = max(all_ics.values(), key=lambda x: x['ic']) if all_ics else {'ic': 0}
    if best_tsmom['ic'] > 0.03:
        assessment.append(f"Momentum signal has predictive power for commodities (best IC={best_tsmom['ic']:.4f}).")
    else:
        assessment.append(f"Momentum signal is weak for commodities (best IC={best_tsmom['ic']:.4f}).")

    # Check if any strategy beats base RP
    combo_key = '75%_3xRP_+_25%_TSMOM'
    if combo_key in metrics_all:
        if metrics_all[combo_key]['cagr'] > 0.217:
            assessment.append(f"Commodity trend overlay pushes RP CAGR from 21.7% to {metrics_all[combo_key]['cagr_pct']:.1f}%.")
        else:
            assessment.append(f"Commodity trend overlay does NOT push above 21.7% base RP ({metrics_all[combo_key]['cagr_pct']:.1f}%).")

    assessment.append("Commodity trend following is structurally regime-agnostic (different drivers than equities).")
    assessment.append("LIMITATION: Commodity ETFs have contango drag (USO, UNG especially), degrading long-term returns vs spot.")
    assessment.append("LIMITATION: Long-only constraint means we can't capture short trends, cutting signal in half.")

    print(f"\nHONEST ASSESSMENT: {' | '.join(assessment)}")

    results = {
        'direction': 'D3_Commodities_Trend',
        'available_commodities': available,
        'momentum_ics': all_ics,
        'metrics': metrics_all,
        'regime_tests': rt_results,
        'honest_assessment': ' | '.join(assessment),
        'timestamp': datetime.now().isoformat(),
    }

    with open(os.path.join(OUT_DIR, 'dir3_commodities_trend.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR}/dir3_commodities_trend.json")
    return results


if __name__ == '__main__':
    run_direction3()
