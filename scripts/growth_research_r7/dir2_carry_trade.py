#!/usr/bin/env python3
"""
R7 DIRECTION 2: Carry Trade (Interest Rate Differentials)

Long high-yield bonds/currencies, short low-yield (long-only for RH: overweight high-yield, underweight low-yield).
ETFs: EMB (EM bonds), HYG (high yield corporate), TLT (US treasuries as funding proxy).
Walk-forward IC test on carry signal predicting forward returns.

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

# Carry assets
CARRY_LONG = ['HYG', 'EMB']  # High yield
CARRY_SHORT_PROXY = ['SHY']  # Low yield (funding leg)
ALL_TICKERS = CARRY_LONG + CARRY_SHORT_PROXY + ['TLT', 'IEF', 'SPY', 'LQD', 'BNDX']


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


def run_direction2():
    print("=" * 80)
    print("DIRECTION 2: Carry Trade (Interest Rate Differentials)")
    print("=" * 80)

    # Download data
    tickers = list(set(ALL_TICKERS))
    print(f"Downloading: {tickers}")
    data = yf.download(tickers, start=START, end=END, auto_adjust=True, progress=False)
    close = data['Close'] if isinstance(data.columns, pd.MultiIndex) else data
    close = close.ffill().dropna(how='all')
    rets = close.pct_change()

    spy_rets = rets['SPY'] if 'SPY' in rets.columns else rets.iloc[:, 0]
    available_carry_long = [t for t in CARRY_LONG if t in close.columns]
    available_carry_short = [t for t in CARRY_SHORT_PROXY if t in close.columns]
    print(f"Available carry long: {available_carry_long}, short proxy: {available_carry_short}")
    print(f"Data range: {close.index[0].date()} to {close.index[-1].date()}")

    # ── Strategy 1: Static Carry (equal weight long HYG+EMB) ──
    if available_carry_long:
        carry_static_ret = rets[available_carry_long].mean(axis=1)
        m_static = calc_metrics(carry_static_ret.dropna(), 'Static Carry (HYG+EMB EW)')
    else:
        m_static = calc_metrics(pd.Series(dtype=float), 'Static Carry')

    # ── Strategy 2: Carry Signal — spread-momentum timing ──
    # Signal: when the HYG-TLT spread is widening (carry assets outperforming), go heavier on carry
    # When spread narrowing (risk-off), reduce carry
    carry_signal_results = {'oot_ic': 0, 'oot_days': 0}
    dynamic_carry_rets = pd.Series(dtype=float)

    if 'HYG' in close.columns and 'TLT' in close.columns and available_carry_long:
        # Carry signal: rolling spread return HYG vs TLT
        spread = np.log(close['HYG']) - np.log(close['TLT'])
        spread_mom = spread.diff(20)  # 20-day spread momentum

        # Walk-forward: use spread momentum to predict forward carry returns
        carry_ret = rets[available_carry_long].mean(axis=1)
        fwd_carry = carry_ret.rolling(TEST_DAYS).sum().shift(-TEST_DAYS)

        # IC test
        oot_preds_all = []
        oot_actuals_all = []

        for start_idx in range(TRAIN_DAYS, len(spread_mom) - TEST_DAYS, TEST_DAYS):
            train_slice = slice(start_idx - TRAIN_DAYS, start_idx)
            test_slice = slice(start_idx, min(start_idx + TEST_DAYS, len(spread_mom)))

            signal_train = spread_mom.iloc[train_slice].dropna()
            target_train = fwd_carry.iloc[train_slice].dropna()
            common = signal_train.index.intersection(target_train.index)
            if len(common) < 50:
                continue

            # Simple: use sign of spread momentum as signal
            signal_test = spread_mom.iloc[test_slice].dropna()
            target_test = fwd_carry.iloc[test_slice].dropna()
            common_test = signal_test.index.intersection(target_test.index)
            if len(common_test) == 0:
                continue

            oot_preds_all.extend(signal_test.loc[common_test].values)
            oot_actuals_all.extend(target_test.loc[common_test].values)

        if len(oot_preds_all) > 50:
            ic, ic_p = spearmanr(oot_preds_all, oot_actuals_all)
            carry_signal_results['oot_ic'] = round(float(ic), 4)
            carry_signal_results['oot_ic_pval'] = float(ic_p)
            carry_signal_results['oot_days'] = len(oot_preds_all)
            print(f"\nCarry signal OOT IC: {ic:.4f} (p={ic_p:.4e}), n={len(oot_preds_all)}")

        # Dynamic carry: overweight carry when signal positive, underweight when negative
        # Use rolling z-score of spread momentum
        spread_z = (spread_mom - spread_mom.rolling(TRAIN_DAYS).mean()) / spread_mom.rolling(TRAIN_DAYS).std().clip(lower=1e-6)
        # Map z-score to weight: z>0 → heavier carry (up to 1.5x), z<0 → lighter (down to 0.5x)
        carry_weight = (0.5 + spread_z.clip(-2, 2) / 4.0).clip(0.3, 1.5)
        dynamic_carry_rets = carry_ret * carry_weight
        dynamic_carry_rets = dynamic_carry_rets.dropna()

    m_dynamic = calc_metrics(dynamic_carry_rets, 'Dynamic Carry (Spread-Timed)')

    # ── Strategy 3: Carry + Duration Hedge ──
    # Long carry assets, short-duration hedge with inverse TLT weight
    if available_carry_long and 'TLT' in rets.columns:
        # Long 70% carry, short 30% duration (approximated as underweight TLT)
        hedge_ret = 0.7 * rets[available_carry_long].mean(axis=1) - 0.3 * rets['TLT']
        m_hedged = calc_metrics(hedge_ret.dropna(), 'Carry + Duration Hedge')
    else:
        m_hedged = calc_metrics(pd.Series(dtype=float), 'Carry + Duration Hedge')

    # ── Strategy 4: Carry combined with base 3x RP ──
    # 80% base RP, 20% carry overlay
    rp_tickers = ['SPY', 'TLT', 'GLD']
    rp_available = [t for t in rp_tickers if t in rets.columns]
    if rp_available and available_carry_long:
        rp_ew_ret = rets[rp_available].mean(axis=1) * 3.0  # 3x leveraged RP proxy
        combo_ret = 0.80 * rp_ew_ret + 0.20 * carry_static_ret
        m_combo = calc_metrics(combo_ret.dropna(), '80% 3x-RP + 20% Carry')
    else:
        m_combo = calc_metrics(pd.Series(dtype=float), 'Combo')

    # Print results
    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)
    for m in [m_static, m_dynamic, m_hedged, m_combo]:
        print(f"\n{m['name']}:")
        print(f"  CAGR: {m['cagr_pct']:.1f}%  |  Sharpe: {m['sharpe']:.3f}  |  Sortino: {m['sortino']:.3f}")
        print(f"  MaxDD: {m['max_dd_pct']:.1f}%  |  Calmar: {m['calmar']:.3f}  |  WR: {m['win_rate']:.3f}")

    # Regime tests
    print("\n--- Regime Tests ---")
    rt_results = {}
    for name, ret_series in [('Static Carry', carry_static_ret if available_carry_long else pd.Series()),
                              ('Dynamic Carry', dynamic_carry_rets)]:
        if len(ret_series) > 50:
            rt = regime_test(ret_series, spy_rets, name)
            rt_results[name] = rt
            for r in ['green', 'red', 'flat']:
                if r in rt:
                    print(f"  {name} {r}: Sharpe={rt[r]['sharpe']:.3f}")
            if 'regime_gap' in rt:
                print(f"  {name} regime gap: {rt['regime_gap']:.3f}")

    # Honest assessment
    assessment = []
    if carry_signal_results.get('oot_ic', 0) > 0.03:
        assessment.append(f"Spread momentum has weak but positive IC ({carry_signal_results['oot_ic']:.4f}) for timing carry.")
    else:
        assessment.append(f"Spread momentum signal lacks predictive edge (IC={carry_signal_results.get('oot_ic', 0):.4f}).")

    if m_static['cagr'] > 0:
        assessment.append(f"Static carry delivers {m_static['cagr_pct']:.1f}% CAGR — a real risk premium.")
    else:
        assessment.append(f"Static carry returns are negative/flat — carry premium not visible in ETF data.")

    # Key limitation
    assessment.append("LIMITATION: Long-only carry via ETFs captures credit spread premium but misses FX carry (no FX ETFs in universe).")
    assessment.append("LIMITATION: Carry drawdowns are correlated with equity drawdowns (risk-on/risk-off), so regime gap will be large.")

    print(f"\nHONEST ASSESSMENT: {' | '.join(assessment)}")

    results = {
        'direction': 'D2_Carry_Trade',
        'carry_signal': carry_signal_results,
        'metrics': {
            'static_carry': m_static,
            'dynamic_carry': m_dynamic,
            'hedged_carry': m_hedged,
            'combo_rp_carry': m_combo,
        },
        'regime_tests': {k: v for k, v in rt_results.items()},
        'honest_assessment': ' | '.join(assessment),
        'timestamp': datetime.now().isoformat(),
    }

    with open(os.path.join(OUT_DIR, 'dir2_carry_trade.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\nResults saved to {OUT_DIR}/dir2_carry_trade.json")
    return results


if __name__ == '__main__':
    run_direction2()
