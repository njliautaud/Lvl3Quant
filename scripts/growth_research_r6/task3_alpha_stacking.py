#!/usr/bin/env python3
"""
R6 TASK 3: ALPHA STACKING — MULTIPLE UNCORRELATED SIGNALS

Combine: Momentum + Carry + Value + Vol Timing on multi-asset universe.
Each signal alone is weak but combined they might be stronger.
Walk-forward the combined signal.

HC #694: Commission-free (Robinhood)
HC #428: Regime test (report honestly)
HC #0: Sliding window walk-forward
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
from datetime import datetime

OUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research_r6'
os.makedirs(OUT_DIR, exist_ok=True)

TRAIN_DAYS = 252
TEST_DAYS = 21
START = '2007-01-01'
END = '2026-07-14'

# Multi-asset universe: 20 liquid ETFs across asset classes
UNIVERSE = {
    # US Equities
    'SPY': 'US Large Cap',
    'QQQ': 'US Tech',
    'IWM': 'US Small Cap',
    # International
    'EFA': 'Dev Intl',
    'EEM': 'Emerging Markets',
    # Bonds
    'TLT': 'US Long Bonds',
    'IEF': 'US Int Bonds',
    'TIP': 'US TIPS',
    'HYG': 'US High Yield',
    # Commodities
    'GLD': 'Gold',
    'DBC': 'Broad Commodities',
    'USO': 'Oil',
    # Real Estate
    'VNQ': 'US REITs',
    # Sectors
    'XLK': 'Tech Sector',
    'XLE': 'Energy Sector',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLU': 'Utilities',
    'XLP': 'Consumer Staples',
    'XLI': 'Industrials',
}


# ─── Shared Utilities ───────────────────────────────────────────────────────

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
            'cagr': round(float(cagr), 4), 'cagr_pct': round(float(cagr*100), 2),
            'max_dd': round(float(max_dd), 4), 'max_dd_pct': round(float(max_dd*100), 2),
            'win_rate': round(float(wr), 4), 'pf': round(float(pf), 3),
            'calmar': round(float(calmar), 3),
            'n_days': int(len(rets))}


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
        m = calc_metrics(strat_rets[mask], f'{name} ({r})')
        results[r] = m
    sg = results['green']['sharpe']
    sr = results['red']['sharpe']
    max_s = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / max_s if max_s > 0 else float('inf')
    return {
        'per_regime': results,
        'sharpe_green': round(float(sg), 4),
        'sharpe_red': round(float(sr), 4),
        'regime_gap': round(float(gap), 4),
        'regime_pass': gap < 0.50,
    }


def permutation_test(strat_rets, n_perms=200):
    """Test if strategy returns are significantly different from random timing."""
    real_sharpe = calc_metrics(strat_rets)['sharpe']
    count_beat = 0
    for _ in range(n_perms):
        shuffled = strat_rets.sample(frac=1.0).values
        shuffled_s = pd.Series(shuffled, index=strat_rets.index)
        perm_sharpe = calc_metrics(shuffled_s)['sharpe']
        if perm_sharpe >= real_sharpe:
            count_beat += 1
    p_value = count_beat / n_perms
    return p_value


# ─── SIGNAL GENERATORS ────────────────────────────────────────────────────

def compute_momentum_signal(prices, lookback_slow=252, lookback_fast=21):
    """12-1 month cross-sectional momentum signal."""
    # Return from 12 months ago to 1 month ago (skip recent month)
    mom = prices.shift(lookback_fast) / prices.shift(lookback_slow) - 1
    # Cross-sectional z-score
    z_scores = mom.sub(mom.mean(axis=1), axis=0).div(mom.std(axis=1), axis=0)
    return z_scores.clip(-3, 3)


def compute_carry_signal(prices, lookback=63):
    """Carry signal: 3-month return as proxy for yield/carry."""
    carry = prices / prices.shift(lookback) - 1
    z_scores = carry.sub(carry.mean(axis=1), axis=0).div(carry.std(axis=1), axis=0)
    return z_scores.clip(-3, 3)


def compute_value_signal(prices, lookback=252*5):
    """Value signal: mean reversion over 5 years (buy cheap, sell expensive)."""
    if len(prices) < lookback:
        lookback = len(prices) // 2

    # 5-year return: negative = cheap = buy
    long_ret = prices / prices.shift(lookback) - 1
    # Invert: low returns = high value signal
    value = -long_ret
    z_scores = value.sub(value.mean(axis=1), axis=0).div(value.std(axis=1), axis=0)
    return z_scores.clip(-3, 3)


def compute_vol_timing_signal(returns, lookback=63):
    """Vol timing: inverse vol signal (buy low vol assets, sell high vol)."""
    vol = returns.rolling(lookback).std() * np.sqrt(252)
    # Cross-sectional: buy low vol, sell high vol
    inv_vol = 1.0 / vol.replace(0, np.nan)
    z_scores = inv_vol.sub(inv_vol.mean(axis=1), axis=0).div(inv_vol.std(axis=1), axis=0)
    return z_scores.clip(-3, 3)


def compute_trend_signal(prices, short_window=50, long_window=200):
    """Time-series trend: price vs moving average."""
    sma_short = prices.rolling(short_window).mean()
    sma_long = prices.rolling(long_window).mean()

    # Signal: short MA above long MA = uptrend
    trend = (sma_short / sma_long - 1)
    # This is time-series, not cross-sectional, so normalize per asset
    z_scores = (trend - trend.rolling(252).mean()) / trend.rolling(252).std()
    return z_scores.clip(-3, 3)


# ─── WALK-FORWARD ALPHA STACKING ──────────────────────────────────────────

def run_alpha_stacking(prices, returns, spy_ret, signal_names, signal_weights,
                       top_n=5, leverage=1.0, long_short=False, name=''):
    """
    Walk-forward alpha stacking strategy.

    Each month:
    1. Compute all signals
    2. Combine with given weights
    3. Go long top N, optionally short bottom N
    4. Hold for TEST_DAYS

    signal_weights: dict of signal_name -> weight
    """
    assets = list(prices.columns)
    n_assets = len(assets)

    # Pre-compute all signals
    all_signals = {}
    if 'momentum' in signal_names:
        all_signals['momentum'] = compute_momentum_signal(prices)
    if 'carry' in signal_names:
        all_signals['carry'] = compute_carry_signal(prices)
    if 'value' in signal_names:
        all_signals['value'] = compute_value_signal(prices)
    if 'vol_timing' in signal_names:
        all_signals['vol_timing'] = compute_vol_timing_signal(returns)
    if 'trend' in signal_names:
        all_signals['trend'] = compute_trend_signal(prices)

    common_idx = prices.index
    strat_rets_list = []

    rebal_dates = list(range(max(TRAIN_DAYS, 252*5+10), len(common_idx) - TEST_DAYS, TEST_DAYS))

    for i in rebal_dates:
        dt = common_idx[i]

        # Combine signals
        composite = pd.Series(0.0, index=assets)
        total_weight = 0

        for sig_name, weight in signal_weights.items():
            if sig_name not in all_signals:
                continue
            sig = all_signals[sig_name]
            sig_row = sig.iloc[i]
            valid = sig_row.dropna()
            if len(valid) > 0:
                composite[valid.index] += weight * valid
                total_weight += weight

        if total_weight == 0:
            continue

        composite = composite / total_weight

        # Rank and select
        ranked = composite.dropna().sort_values(ascending=False)
        if len(ranked) < top_n:
            continue

        # Long top N
        longs = ranked.head(top_n).index.tolist()
        # Short bottom N (if long_short)
        shorts = ranked.tail(top_n).index.tolist() if long_short else []

        # Equal weight within long/short legs
        long_weight = leverage / top_n
        short_weight = -leverage / top_n if long_short else 0

        for j in range(i, min(i + TEST_DAYS, len(common_idx) - 1)):
            daily_ret = 0
            for asset in longs:
                r = float(returns[asset].iloc[j + 1]) if j + 1 < len(returns) and asset in returns else 0
                if not np.isnan(r):
                    daily_ret += long_weight * r
            for asset in shorts:
                r = float(returns[asset].iloc[j + 1]) if j + 1 < len(returns) and asset in returns else 0
                if not np.isnan(r):
                    daily_ret += short_weight * r

            strat_rets_list.append({'date': common_idx[j], 'ret': daily_ret})

    s = pd.DataFrame(strat_rets_list).set_index('date')['ret']
    s = s[~s.index.duplicated(keep='first')]
    return s


# ─── MAIN ──────────────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("R6 TASK 3: ALPHA STACKING — MULTIPLE UNCORRELATED SIGNALS")
    print(f"Period: {START} to {END}")
    print("=" * 70)

    # Download data
    tickers = list(UNIVERSE.keys())
    print(f"\nDownloading {len(tickers)} ETFs...")
    all_data = {}
    for t in tickers:
        try:
            df = yf.download(t, start=START, end=END, progress=False)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 252:
                all_data[t] = df
                print(f"  {t}: {len(df)} days")
        except Exception as e:
            print(f"  {t}: FAILED - {e}")

    # Build price and return matrices
    valid_tickers = [t for t in tickers if t in all_data]
    first_idx = all_data[valid_tickers[0]].index
    for t in valid_tickers[1:]:
        first_idx = first_idx.intersection(all_data[t].index)

    prices = pd.DataFrame({t: all_data[t]['Close'].reindex(first_idx) for t in valid_tickers})
    returns = prices.pct_change()

    spy_ret = returns['SPY']

    print(f"\n  Common period: {first_idx[0].strftime('%Y-%m-%d')} to {first_idx[-1].strftime('%Y-%m-%d')}")
    print(f"  {len(valid_tickers)} ETFs, {len(first_idx)} trading days")

    all_results = {}

    # ─── Part A: Individual Signals ───────────────────────────────────
    print("\n" + "=" * 60)
    print("PART A: INDIVIDUAL SIGNAL PERFORMANCE (LONG-ONLY)")
    print("=" * 60)

    signal_names = ['momentum', 'carry', 'value', 'vol_timing', 'trend']

    for sig in signal_names:
        print(f"\n  Testing {sig} signal (long top 5)...")
        s = run_alpha_stacking(
            prices, returns, spy_ret,
            signal_names=[sig],
            signal_weights={sig: 1.0},
            top_n=5, leverage=1.0, long_short=False,
            name=sig,
        )
        m = calc_metrics(s, sig)
        rt = regime_test(s, spy_ret.reindex(s.index).dropna(), sig)

        print(f"    Sharpe={m['sharpe']}, CAGR={m['cagr_pct']}%, "
              f"MaxDD={m['max_dd_pct']}%, Calmar={m['calmar']}")
        print(f"    Regime gap={rt['regime_gap']}, "
              f"Green={rt['sharpe_green']}, Red={rt['sharpe_red']}")

        all_results[f'individual_{sig}'] = {'metrics': m, 'regime': rt}

    # ─── Part B: Individual Signals (LONG-SHORT) ──────────────────────
    print("\n" + "=" * 60)
    print("PART B: INDIVIDUAL SIGNAL PERFORMANCE (LONG-SHORT)")
    print("=" * 60)

    for sig in signal_names:
        print(f"\n  Testing {sig} signal (long top 5, short bottom 5)...")
        s = run_alpha_stacking(
            prices, returns, spy_ret,
            signal_names=[sig],
            signal_weights={sig: 1.0},
            top_n=5, leverage=1.0, long_short=True,
            name=f'{sig}_LS',
        )
        m = calc_metrics(s, f'{sig}_LS')
        rt = regime_test(s, spy_ret.reindex(s.index).dropna(), f'{sig}_LS')

        print(f"    Sharpe={m['sharpe']}, CAGR={m['cagr_pct']}%, "
              f"MaxDD={m['max_dd_pct']}%, Calmar={m['calmar']}")
        print(f"    Regime gap={rt['regime_gap']}")

        all_results[f'individual_{sig}_LS'] = {'metrics': m, 'regime': rt}

    # ─── Part C: Signal Correlations ──────────────────────────────────
    print("\n" + "=" * 60)
    print("PART C: SIGNAL RETURN CORRELATIONS")
    print("=" * 60)

    signal_rets = {}
    for sig in signal_names:
        s = run_alpha_stacking(
            prices, returns, spy_ret,
            signal_names=[sig], signal_weights={sig: 1.0},
            top_n=5, leverage=1.0, long_short=False,
        )
        signal_rets[sig] = s

    common = signal_rets[signal_names[0]].index
    for sig in signal_names[1:]:
        common = common.intersection(signal_rets[sig].index)

    sig_df = pd.DataFrame({sig: signal_rets[sig].reindex(common) for sig in signal_names})
    corr = sig_df.corr()

    print(f"\n  {'':>12}", end='')
    for sig in signal_names:
        print(f"  {sig[:10]:>10}", end='')
    print()
    for s1 in signal_names:
        print(f"  {s1[:12]:>12}", end='')
        for s2 in signal_names:
            c = float(corr.loc[s1, s2])
            print(f"  {c:>10.3f}", end='')
        print()

    all_results['signal_correlations'] = corr.to_dict()

    # ─── Part D: Combined Signal Stacks ───────────────────────────────
    print("\n" + "=" * 60)
    print("PART D: COMBINED SIGNAL STACKS (LONG-ONLY)")
    print("=" * 60)

    # Test various combinations
    combos = {
        'mom+carry': {'momentum': 0.5, 'carry': 0.5},
        'mom+value': {'momentum': 0.5, 'value': 0.5},
        'mom+vol': {'momentum': 0.5, 'vol_timing': 0.5},
        'mom+trend': {'momentum': 0.5, 'trend': 0.5},
        'mom+carry+value': {'momentum': 0.33, 'carry': 0.33, 'value': 0.34},
        'mom+carry+vol': {'momentum': 0.33, 'carry': 0.33, 'vol_timing': 0.34},
        'all_equal': {sig: 0.2 for sig in signal_names},
        'mom_heavy': {'momentum': 0.4, 'carry': 0.2, 'value': 0.15,
                      'vol_timing': 0.15, 'trend': 0.10},
        'diversified': {'momentum': 0.25, 'carry': 0.25, 'value': 0.25,
                        'vol_timing': 0.25},
    }

    for combo_name, weights in combos.items():
        print(f"\n  Testing {combo_name}...")
        s = run_alpha_stacking(
            prices, returns, spy_ret,
            signal_names=list(weights.keys()),
            signal_weights=weights,
            top_n=5, leverage=1.0, long_short=False,
            name=combo_name,
        )
        m = calc_metrics(s, combo_name)
        rt = regime_test(s, spy_ret.reindex(s.index).dropna(), combo_name)

        print(f"    Sharpe={m['sharpe']}, CAGR={m['cagr_pct']}%, "
              f"MaxDD={m['max_dd_pct']}%, Calmar={m['calmar']}")
        print(f"    Regime gap={rt['regime_gap']}")

        all_results[f'combo_{combo_name}'] = {
            'metrics': m, 'regime': rt, 'weights': weights,
        }

    # ─── Part E: Leveraged Combined Stack ─────────────────────────────
    print("\n" + "=" * 60)
    print("PART E: LEVERAGED COMBINED STACK (2x, 3x)")
    print("=" * 60)

    best_combo_name = max(
        [k for k in all_results if k.startswith('combo_')],
        key=lambda k: all_results[k]['metrics']['sharpe']
    )
    best_weights = all_results[best_combo_name].get('weights', {sig: 0.2 for sig in signal_names})

    for lev in [1.5, 2.0, 3.0]:
        label = f'best_stack_{lev}x'
        print(f"\n  Testing {label} ({best_combo_name} weights)...")
        s = run_alpha_stacking(
            prices, returns, spy_ret,
            signal_names=list(best_weights.keys()),
            signal_weights=best_weights,
            top_n=5, leverage=lev, long_short=False,
            name=label,
        )
        m = calc_metrics(s, label)
        rt = regime_test(s, spy_ret.reindex(s.index).dropna(), label)

        print(f"    Sharpe={m['sharpe']}, CAGR={m['cagr_pct']}%, "
              f"MaxDD={m['max_dd_pct']}%, Calmar={m['calmar']}")
        print(f"    Regime gap={rt['regime_gap']}")

        all_results[f'leveraged_{label}'] = {'metrics': m, 'regime': rt,
                                              'leverage': lev}

    # ─── Part F: Permutation Test on Best ─────────────────────────────
    print("\n" + "=" * 60)
    print("PART F: PERMUTATION TEST ON BEST COMBO")
    print("=" * 60)

    # Re-run best combo
    s_best = run_alpha_stacking(
        prices, returns, spy_ret,
        signal_names=list(best_weights.keys()),
        signal_weights=best_weights,
        top_n=5, leverage=1.0, long_short=False,
    )

    print(f"  Running 200-trial permutation test on {best_combo_name}...")
    p_val = permutation_test(s_best, n_perms=200)
    print(f"  Permutation p-value: {p_val:.3f} ({'SIGNIFICANT' if p_val < 0.05 else 'NOT SIGNIFICANT'})")

    all_results['permutation_test'] = {
        'combo': best_combo_name,
        'p_value': round(p_val, 4),
        'significant': p_val < 0.05,
        'n_permutations': 200,
    }

    # ─── FINAL SUMMARY ────────────────────────────────────────────────
    print("\n" + "=" * 70)
    print("FINAL SUMMARY")
    print("=" * 70)

    # Individual signals
    print("\n  INDIVIDUAL SIGNALS (LONG-ONLY):")
    print(f"  {'Signal':<15} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Gap':>8}")
    print(f"  {'-'*15} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    for sig in signal_names:
        k = f'individual_{sig}'
        if k in all_results:
            m = all_results[k]['metrics']
            rt = all_results[k]['regime']
            print(f"  {sig:<15} {m['cagr_pct']:>7.1f}% {m['sharpe']:>8.3f} "
                  f"{m['max_dd_pct']:>7.1f}% {rt['regime_gap']:>8.3f}")

    # Combined stacks
    print("\n  COMBINED SIGNAL STACKS:")
    print(f"  {'Combo':<20} {'CAGR':>8} {'Sharpe':>8} {'MaxDD':>8} {'Gap':>8}")
    print(f"  {'-'*20} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    combo_keys = sorted([k for k in all_results if k.startswith('combo_')],
                        key=lambda k: all_results[k]['metrics']['sharpe'], reverse=True)
    for k in combo_keys:
        m = all_results[k]['metrics']
        rt = all_results[k]['regime']
        name = k.replace('combo_', '')
        print(f"  {name:<20} {m['cagr_pct']:>7.1f}% {m['sharpe']:>8.3f} "
              f"{m['max_dd_pct']:>7.1f}% {rt['regime_gap']:>8.3f}")

    # Leveraged
    print("\n  LEVERAGED BEST STACK:")
    for k in sorted([k for k in all_results if k.startswith('leveraged_')]):
        m = all_results[k]['metrics']
        rt = all_results[k]['regime']
        lev = all_results[k].get('leverage', '?')
        print(f"  {lev}x: CAGR={m['cagr_pct']}%, Sharpe={m['sharpe']}, "
              f"MaxDD={m['max_dd_pct']}%, Gap={rt['regime_gap']}")

    # Permutation
    perm = all_results.get('permutation_test', {})
    if perm:
        print(f"\n  PERMUTATION TEST: p={perm['p_value']} "
              f"({'PASS' if perm['significant'] else 'FAIL'})")

    # Save results
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, pd.Timestamp):
            return str(obj)
        else:
            return obj

    out_file = os.path.join(OUT_DIR, 'task3_alpha_stacking_results.json')
    with open(out_file, 'w') as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)
    print(f"\n  Results saved to {out_file}")


if __name__ == '__main__':
    main()
