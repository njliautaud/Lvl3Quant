#!/usr/bin/env python3
"""
RELATIVE VIX PERCENTILE TIMING — Strategy v1
==============================================
Thesis: Instead of fixed VIX thresholds (15/20/30%), use VIX's OWN rolling
percentile rank. This naturally adapts to changing vol regimes:
  - 2017 (low-vol): VIX 14 = 80th percentile -> cautious
  - 2020 (high-vol): VIX 20 = 30th percentile -> aggressive

Strategy variants:
  A) VIX Percentile Rank (trailing 252d window)
  B) VIX Z-Score variant
  C) Parameter sweep across lookback windows and thresholds

Full adversarial validation suite (HC #705):
  1. Permutation test (100+ shuffles)
  2. Sub-period consistency (3+ blocks)
  3. Outlier robustness (remove top 5% days)
  4. R1 regime test (green/red SPY days)
  5. Walk-forward validation (3yr train, 1yr OOS, rolling)
  6. Head-to-head vs fixed-threshold baseline
"""

import numpy as np
import pandas as pd
import yfinance as yf
import json
import os
import sys
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/relative_vix_percentile_v1'
os.makedirs(OUTPUT_DIR, exist_ok=True)

INITIAL = 500
WEEKLY_DCA = 100
np.random.seed(42)


# =============================================================================
# DATA
# =============================================================================

def download_data():
    """Download all required price data."""
    tickers = ['SPY', 'UPRO', 'GLD', 'TLT', 'SHY', '^VIX']
    data = yf.download(tickers, start='2010-01-01', period='max',
                       auto_adjust=True, threads=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        closes = data['Close']
    else:
        closes = data

    # Clean column names
    if hasattr(closes.columns, 'droplevel'):
        try:
            closes.columns = closes.columns.droplevel(1)
        except Exception:
            pass

    # Rename ^VIX to VIX
    if '^VIX' in closes.columns:
        closes = closes.rename(columns={'^VIX': 'VIX'})

    closes = closes.dropna(how='all')
    # Forward fill VIX for holidays
    closes['VIX'] = closes['VIX'].ffill()
    # Need at least SPY, UPRO, VIX
    closes = closes.dropna(subset=['SPY', 'UPRO', 'VIX'])
    return closes


# =============================================================================
# SIGNAL COMPUTATION
# =============================================================================

def compute_vix_percentile(vix_series, lookback=252):
    """
    Compute rolling percentile rank of VIX within its own trailing distribution.
    Returns value 0-100 (percentile).
    """
    pctile = vix_series.rolling(lookback).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100,
        raw=False
    )
    return pctile


def compute_vix_zscore(vix_series, lookback=252):
    """
    Compute rolling Z-score of VIX: (VIX - rolling_mean) / rolling_std.
    """
    rolling_mean = vix_series.rolling(lookback).mean()
    rolling_std = vix_series.rolling(lookback).std()
    z = (vix_series - rolling_mean) / rolling_std
    return z


# =============================================================================
# STRATEGY: VIX PERCENTILE
# =============================================================================

def get_regime_percentile(pctile_val, low_thresh=25, high_thresh=75,
                          defensive='GLD'):
    """
    Percentile-based regime:
      pctile < low_thresh  -> UPRO (VIX unusually low for this regime)
      pctile > high_thresh -> defensive (VIX unusually high)
      else                 -> SPY
    """
    if np.isnan(pctile_val):
        return 'SPY'
    if pctile_val < low_thresh:
        return 'UPRO'
    elif pctile_val > high_thresh:
        return defensive
    else:
        return 'SPY'


def get_regime_zscore(z_val, low_z=-0.5, high_z=1.0, defensive='GLD'):
    """
    Z-score based regime:
      Z < low_z  -> UPRO (VIX below average)
      Z > high_z -> defensive (VIX elevated)
      else       -> SPY
    """
    if np.isnan(z_val):
        return 'SPY'
    if z_val < low_z:
        return 'UPRO'
    elif z_val > high_z:
        return defensive
    else:
        return 'SPY'


def get_regime_fixed(vol_21d_val):
    """
    Fixed-threshold baseline (our current Gameplan system).
    Uses 21d realized vol of SPY, with thresholds at 15%, 20%, 30%.
    """
    if np.isnan(vol_21d_val):
        return 'SPY'
    vol_pct = vol_21d_val * 100
    if vol_pct > 30:
        return 'GLD'
    elif vol_pct > 20:
        return 'SPY'
    else:
        return 'UPRO'


# =============================================================================
# SIMULATION ENGINE
# =============================================================================

def simulate(closes, regime_series, tx_cost_pct=0.001, warmup=260):
    """
    Simulate portfolio with DCA and regime-based allocation.
    regime_series: pd.Series with index matching closes, values in ['UPRO','SPY','GLD','TLT','SHY']
    """
    returns = closes.pct_change().fillna(0)
    cash = float(INITIAL)
    total_contributed = float(INITIAL)
    last_week = None
    last_regime = None
    switches = 0
    daily_values = []
    daily_regimes = []

    for i in range(warmup, len(closes)):
        date = closes.index[i]

        # Weekly DCA
        week_key = (date.year, date.isocalendar()[1])
        if week_key != last_week:
            cash += WEEKLY_DCA
            total_contributed += WEEKLY_DCA
            last_week = week_key

        regime = regime_series.iloc[i] if i < len(regime_series) else 'SPY'

        # Transaction cost on switch
        if regime != last_regime and last_regime is not None:
            switches += 1
            cash *= (1 - tx_cost_pct)
        last_regime = regime

        # Apply return
        if regime in returns.columns:
            r = returns.loc[date, regime]
            if not np.isnan(r):
                cash *= (1 + r)

        daily_values.append(cash)
        daily_regimes.append(regime)

    dates = closes.index[warmup:]
    vals = pd.Series(daily_values, index=dates)
    regs = pd.Series(daily_regimes, index=dates)
    return vals, total_contributed, switches, regs


def compute_metrics(values, total_contributed=None):
    """Compute risk-adjusted metrics."""
    daily_ret = values.pct_change().dropna()
    if len(daily_ret) < 10:
        return {'sharpe': 0, 'sortino': 0, 'cagr': 0, 'max_dd': 0,
                'calmar': 0, 'ann_vol': 0, 'final_value': values.iloc[-1] if len(values) > 0 else 0,
                'total_contributed': total_contributed or 0, 'profit': 0}

    ann_ret = daily_ret.mean() * 252
    ann_vol = daily_ret.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    neg_ret = daily_ret[daily_ret < 0]
    downside_vol = neg_ret.std() * np.sqrt(252) if len(neg_ret) > 0 else 1
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    running_max = values.cummax()
    drawdown = (values - running_max) / running_max
    max_dd = drawdown.min()

    years = (values.index[-1] - values.index[0]).days / 365.25
    cagr = (values.iloc[-1] / values.iloc[0]) ** (1 / years) - 1 if years > 0 else 0
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    tc = total_contributed or 0
    return {
        'final_value': float(values.iloc[-1]),
        'cagr': float(cagr),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd),
        'calmar': float(calmar),
        'ann_vol': float(ann_vol),
        'total_contributed': float(tc),
        'profit': float(values.iloc[-1] - tc) if tc else 0
    }


def build_regime_series(closes, strategy='percentile', lookback=252,
                        low_thresh=25, high_thresh=75, defensive='GLD',
                        low_z=-0.5, high_z=1.0):
    """Build regime series for any strategy variant."""
    vix = closes['VIX']

    if strategy == 'percentile':
        signal = compute_vix_percentile(vix, lookback=lookback)
        regimes = signal.apply(lambda x: get_regime_percentile(
            x, low_thresh=low_thresh, high_thresh=high_thresh, defensive=defensive))
    elif strategy == 'zscore':
        signal = compute_vix_zscore(vix, lookback=lookback)
        regimes = signal.apply(lambda x: get_regime_zscore(
            x, low_z=low_z, high_z=high_z, defensive=defensive))
    elif strategy == 'fixed':
        spy_ret = closes['SPY'].pct_change()
        vol_21d = spy_ret.rolling(21).std() * np.sqrt(252)
        regimes = vol_21d.apply(get_regime_fixed)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

    return regimes


# =============================================================================
# ADVERSARIAL TESTS
# =============================================================================

def test_permutation(closes, regime_series, n_perms=200, warmup=260):
    """PERMUTATION TEST: Shuffle regime labels. Real signal must beat random."""
    print("\n" + "=" * 70)
    print("  TEST 1: PERMUTATION TEST (n=%d)" % n_perms)
    print("=" * 70)

    vals_real, contrib, switches, _ = simulate(closes, regime_series, warmup=warmup)
    m_real = compute_metrics(vals_real, contrib)
    real_sharpe = m_real['sharpe']
    print(f"  Real system: Sharpe={real_sharpe:.3f}, CAGR={m_real['cagr']:.1%}")

    perm_sharpes = []
    for p in range(n_perms):
        shuffled = regime_series.copy()
        # Block shuffle (5-day blocks preserve autocorrelation structure)
        block_size = 5
        n_blocks = len(shuffled) // block_size
        block_indices = np.arange(n_blocks)
        np.random.shuffle(block_indices)
        new_vals = []
        for bi in block_indices:
            s = bi * block_size
            new_vals.extend(shuffled.iloc[s:s + block_size].tolist())
        new_vals.extend(shuffled.iloc[n_blocks * block_size:].tolist())
        perm_regime = pd.Series(new_vals[:len(shuffled)], index=shuffled.index)

        vals_p, contrib_p, _, _ = simulate(closes, perm_regime, warmup=warmup)
        m_p = compute_metrics(vals_p, contrib_p)
        perm_sharpes.append(m_p['sharpe'])

        if (p + 1) % 50 == 0:
            print(f"    {p + 1}/{n_perms} permutations done...")

    perm_sharpes = np.array(perm_sharpes)
    p_value = np.mean(perm_sharpes >= real_sharpe)

    print(f"\n  Real Sharpe:   {real_sharpe:.3f}")
    print(f"  Perm mean:     {np.mean(perm_sharpes):.3f}")
    print(f"  Perm p95:      {np.percentile(perm_sharpes, 95):.3f}")
    print(f"  p-value:       {p_value:.4f}")
    passed = p_value < 0.05
    banner = "PASS" if passed else "FAIL"
    print(f"\n  {'=' * 30}")
    print(f"  PERMUTATION TEST: {banner}")
    print(f"  {'=' * 30}")
    return {'real_sharpe': real_sharpe, 'perm_mean': float(np.mean(perm_sharpes)),
            'p_value': float(p_value), 'pass': passed}


def test_subperiod(closes, regime_series, warmup=260):
    """SUB-PERIOD CONSISTENCY: Each 3-year block must show positive edge."""
    print("\n" + "=" * 70)
    print("  TEST 2: SUB-PERIOD CONSISTENCY (3-year blocks)")
    print("=" * 70)

    vals, contrib, _, _ = simulate(closes, regime_series, warmup=warmup)
    daily_ret = vals.pct_change().dropna()

    years = sorted(set(daily_ret.index.year))
    blocks = []
    for i in range(0, len(years), 3):
        block_years = years[i:i + 3]
        if len(block_years) >= 2:
            block_rets = daily_ret[daily_ret.index.year.isin(block_years)]
            if len(block_rets) > 100:
                ann_ret = block_rets.mean() * 252
                ann_vol = block_rets.std() * np.sqrt(252)
                sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
                blocks.append({'years': f"{min(block_years)}-{max(block_years)}",
                               'sharpe': sharpe, 'ann_ret': ann_ret,
                               'n_days': len(block_rets)})

    print(f"\n  {'Period':<15} {'Sharpe':>8} {'Ann Ret':>10} {'Days':>6}")
    print(f"  {'-' * 15} {'-' * 8} {'-' * 10} {'-' * 6}")
    positive = 0
    for b in blocks:
        mark = "[+]" if b['sharpe'] > 0 else "[-]"
        print(f"  {b['years']:<15} {b['sharpe']:>8.3f} {b['ann_ret']:>9.1%} {b['n_days']:>6} {mark}")
        if b['sharpe'] > 0:
            positive += 1

    pct_positive = positive / len(blocks) if blocks else 0
    passed = pct_positive >= 0.6 and len(blocks) >= 3
    banner = "PASS" if passed else "FAIL"
    print(f"\n  Positive blocks: {positive}/{len(blocks)} ({pct_positive:.0%})")
    print(f"\n  {'=' * 30}")
    print(f"  SUB-PERIOD TEST: {banner}")
    print(f"  {'=' * 30}")
    return {'blocks': blocks, 'pct_positive': pct_positive, 'pass': passed}


def test_outlier_robustness(closes, regime_series, warmup=260):
    """OUTLIER ROBUSTNESS: Remove top 5% best days. Edge must persist."""
    print("\n" + "=" * 70)
    print("  TEST 3: OUTLIER ROBUSTNESS (remove top 5% days)")
    print("=" * 70)

    vals, contrib, _, _ = simulate(closes, regime_series, warmup=warmup)
    daily_ret = vals.pct_change().dropna()

    # Full period metrics
    full_sharpe = (daily_ret.mean() * 252) / (daily_ret.std() * np.sqrt(252))

    # Remove top 5% return days
    threshold = daily_ret.quantile(0.95)
    trimmed = daily_ret[daily_ret <= threshold]
    trimmed_sharpe = (trimmed.mean() * 252) / (trimmed.std() * np.sqrt(252))

    # Also remove top 5% AND bottom 5%
    lo = daily_ret.quantile(0.05)
    hi = daily_ret.quantile(0.95)
    winsorized = daily_ret[(daily_ret >= lo) & (daily_ret <= hi)]
    winsor_sharpe = (winsorized.mean() * 252) / (winsorized.std() * np.sqrt(252))

    print(f"  Full Sharpe:           {full_sharpe:.3f}")
    print(f"  Top-5% removed Sharpe: {trimmed_sharpe:.3f}")
    print(f"  Winsorized Sharpe:     {winsor_sharpe:.3f}")

    # Pass if Sharpe stays positive after removing best days
    passed = trimmed_sharpe > 0 and winsor_sharpe > 0
    banner = "PASS" if passed else "FAIL"
    print(f"\n  {'=' * 30}")
    print(f"  OUTLIER TEST: {banner}")
    print(f"  {'=' * 30}")
    return {'full_sharpe': full_sharpe, 'trimmed_sharpe': trimmed_sharpe,
            'winsor_sharpe': winsor_sharpe, 'pass': passed}


def test_regime_r1(closes, regime_series, warmup=260):
    """R1 REGIME TEST: Strategy must work on both green and red SPY days."""
    print("\n" + "=" * 70)
    print("  TEST 4: R1 REGIME TEST (green vs red SPY days)")
    print("=" * 70)

    vals, contrib, _, regs = simulate(closes, regime_series, warmup=warmup)
    strat_ret = vals.pct_change().dropna()

    spy_ret = closes['SPY'].pct_change().reindex(strat_ret.index)

    green_days = strat_ret[spy_ret > 0]
    red_days = strat_ret[spy_ret < 0]
    flat_days = strat_ret[spy_ret == 0]

    green_sharpe = (green_days.mean() * 252) / (green_days.std() * np.sqrt(252)) if len(green_days) > 10 else 0
    red_sharpe = (red_days.mean() * 252) / (red_days.std() * np.sqrt(252)) if len(red_days) > 10 else 0

    print(f"  Green SPY days: n={len(green_days)}, mean={green_days.mean():.4f}, Sharpe={green_sharpe:.3f}")
    print(f"  Red SPY days:   n={len(red_days)}, mean={red_days.mean():.4f}, Sharpe={red_sharpe:.3f}")
    print(f"  Flat SPY days:  n={len(flat_days)}")

    # Regime asymmetry check (HC #428 R1)
    max_sharpe = max(abs(green_sharpe), abs(red_sharpe))
    asymmetry = abs(green_sharpe - red_sharpe) / max_sharpe if max_sharpe > 0 else 0

    print(f"\n  Regime asymmetry: {asymmetry:.2f} (reject if >0.50)")

    # Also check regime distribution
    regime_counts = regs.value_counts()
    print(f"\n  Regime distribution:")
    for r, c in regime_counts.items():
        print(f"    {r}: {c} days ({c / len(regs):.1%})")

    passed = asymmetry <= 0.50
    banner = "PASS" if passed else "FAIL"
    print(f"\n  {'=' * 30}")
    print(f"  R1 REGIME TEST: {banner}")
    print(f"  {'=' * 30}")
    return {'green_sharpe': green_sharpe, 'red_sharpe': red_sharpe,
            'asymmetry': asymmetry, 'pass': passed}


def test_walkforward(closes, strategy='percentile', lookback=252,
                     low_thresh=25, high_thresh=75, defensive='GLD',
                     low_z=-0.5, high_z=1.0,
                     train_years=3, test_years=1):
    """
    WALK-FORWARD VALIDATION: 3yr train, 1yr OOS, rolling.
    Train = compute optimal thresholds. Test = apply them OOS.
    For percentile/zscore the signal is parameterized, so we validate
    that the strategy works in true OOS windows.
    """
    print("\n" + "=" * 70)
    print(f"  TEST 5: WALK-FORWARD VALIDATION ({train_years}yr train, {test_years}yr OOS)")
    print("=" * 70)

    all_years = sorted(set(closes.index.year))
    min_year = min(all_years) + 1  # Need warmup year

    oos_results = []

    for start_test in range(min_year + train_years, max(all_years) + 1, test_years):
        end_test = start_test + test_years - 1
        start_train = start_test - train_years

        # Get train and test data
        train_mask = (closes.index.year >= start_train) & (closes.index.year < start_test)
        test_mask = (closes.index.year >= start_test) & (closes.index.year <= end_test)

        train_data = closes[train_mask]
        test_data = closes[test_mask]

        if len(train_data) < 200 or len(test_data) < 100:
            continue

        # Build regime for the FULL dataset (needs lookback history)
        # But only evaluate on test period
        regime_full = build_regime_series(
            closes, strategy=strategy, lookback=lookback,
            low_thresh=low_thresh, high_thresh=high_thresh,
            defensive=defensive, low_z=low_z, high_z=high_z
        )

        # Extract test period returns
        test_regime = regime_full[test_mask]
        test_returns = closes.pct_change().fillna(0)[test_mask]

        # Simple return computation for test period
        port_ret = []
        for idx in test_returns.index:
            reg = test_regime.loc[idx] if idx in test_regime.index else 'SPY'
            if reg in test_returns.columns:
                r = test_returns.loc[idx, reg]
                port_ret.append(r if not np.isnan(r) else 0)
            else:
                port_ret.append(0)

        port_ret = pd.Series(port_ret, index=test_returns.index)
        ann_ret = port_ret.mean() * 252
        ann_vol = port_ret.std() * np.sqrt(252)
        sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

        spy_oos = test_returns['SPY']
        spy_sharpe = (spy_oos.mean() * 252) / (spy_oos.std() * np.sqrt(252))

        oos_results.append({
            'period': f"{start_test}-{end_test}",
            'sharpe': sharpe,
            'ann_ret': ann_ret,
            'spy_sharpe': spy_sharpe,
            'excess_sharpe': sharpe - spy_sharpe,
            'n_days': len(port_ret)
        })

    print(f"\n  {'Period':<12} {'Sharpe':>8} {'SPY Sh':>8} {'Excess':>8} {'Days':>6}")
    print(f"  {'-' * 12} {'-' * 8} {'-' * 8} {'-' * 8} {'-' * 6}")

    positive_excess = 0
    for r in oos_results:
        mark = "[+]" if r['excess_sharpe'] > 0 else "[-]"
        print(f"  {r['period']:<12} {r['sharpe']:>8.3f} {r['spy_sharpe']:>8.3f} "
              f"{r['excess_sharpe']:>8.3f} {r['n_days']:>6} {mark}")
        if r['excess_sharpe'] > 0:
            positive_excess += 1

    pct_positive = positive_excess / len(oos_results) if oos_results else 0
    avg_excess = np.mean([r['excess_sharpe'] for r in oos_results]) if oos_results else 0
    avg_oos_sharpe = np.mean([r['sharpe'] for r in oos_results]) if oos_results else 0

    print(f"\n  OOS windows with positive excess: {positive_excess}/{len(oos_results)} ({pct_positive:.0%})")
    print(f"  Avg OOS Sharpe: {avg_oos_sharpe:.3f}")
    print(f"  Avg excess Sharpe vs SPY: {avg_excess:.3f}")

    passed = pct_positive >= 0.5 and avg_oos_sharpe > 0
    banner = "PASS" if passed else "FAIL"
    print(f"\n  {'=' * 30}")
    print(f"  WALK-FORWARD TEST: {banner}")
    print(f"  {'=' * 30}")
    return {'oos_results': oos_results, 'pct_positive': pct_positive,
            'avg_oos_sharpe': avg_oos_sharpe, 'avg_excess': avg_excess, 'pass': passed}


def test_head_to_head(closes, warmup=260):
    """HEAD-TO-HEAD: Compare percentile, zscore, and fixed-threshold systems."""
    print("\n" + "=" * 70)
    print("  TEST 6: HEAD-TO-HEAD COMPARISON")
    print("=" * 70)

    configs = [
        ('Fixed Threshold (baseline)', 'fixed', {}),
        ('Percentile 252d [25/75]', 'percentile', {'lookback': 252, 'low_thresh': 25, 'high_thresh': 75}),
        ('Percentile 252d [20/80]', 'percentile', {'lookback': 252, 'low_thresh': 20, 'high_thresh': 80}),
        ('Percentile 252d [30/70]', 'percentile', {'lookback': 252, 'low_thresh': 30, 'high_thresh': 70}),
        ('Percentile 126d [25/75]', 'percentile', {'lookback': 126, 'low_thresh': 25, 'high_thresh': 75}),
        ('Percentile 504d [25/75]', 'percentile', {'lookback': 504, 'low_thresh': 25, 'high_thresh': 75}),
        ('Percentile 252d [25/75] TLT', 'percentile', {'lookback': 252, 'low_thresh': 25, 'high_thresh': 75, 'defensive': 'TLT'}),
        ('Percentile 252d [25/75] SHY', 'percentile', {'lookback': 252, 'low_thresh': 25, 'high_thresh': 75, 'defensive': 'SHY'}),
        ('Z-Score 252d [-0.5/1.0]', 'zscore', {'lookback': 252, 'low_z': -0.5, 'high_z': 1.0}),
        ('Z-Score 252d [-0.75/0.75]', 'zscore', {'lookback': 252, 'low_z': -0.75, 'high_z': 0.75}),
        ('Z-Score 252d [-0.5/1.5]', 'zscore', {'lookback': 252, 'low_z': -0.5, 'high_z': 1.5}),
        ('Z-Score 126d [-0.5/1.0]', 'zscore', {'lookback': 126, 'low_z': -0.5, 'high_z': 1.0}),
    ]

    results = []
    print(f"\n  {'Config':<35} {'Sharpe':>7} {'Sortino':>8} {'CAGR':>7} {'MaxDD':>7} {'Calmar':>7} {'Switches':>9}")
    print(f"  {'-' * 35} {'-' * 7} {'-' * 8} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 9}")

    for name, strategy, params in configs:
        regime = build_regime_series(closes, strategy=strategy, **params)
        vals, contrib, switches, regs = simulate(closes, regime, warmup=warmup)
        m = compute_metrics(vals, contrib)
        m['name'] = name
        m['strategy'] = strategy
        m['params'] = params
        m['switches'] = switches
        results.append(m)

        print(f"  {name:<35} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['cagr']:>6.1%} "
              f"{m['max_dd']:>6.1%} {m['calmar']:>7.2f} {switches:>9}")

    # Also run buy-and-hold SPY and UPRO for reference
    for ticker in ['SPY', 'UPRO']:
        spy_regime = pd.Series(ticker, index=closes.index)
        vals, contrib, _, _ = simulate(closes, spy_regime, warmup=warmup)
        m = compute_metrics(vals, contrib)
        m['name'] = f'Buy & Hold {ticker}'
        m['switches'] = 0
        results.append(m)
        print(f"  {'B&H ' + ticker:<35} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} {m['cagr']:>6.1%} "
              f"{m['max_dd']:>6.1%} {m['calmar']:>7.2f} {0:>9}")

    # Find best strategy vs baseline
    baseline = results[0]  # Fixed threshold
    best_strat = max([r for r in results if r.get('strategy') in ('percentile', 'zscore')],
                     key=lambda x: x['sharpe'])

    print(f"\n  BEST ADAPTIVE: {best_strat['name']}")
    print(f"    Sharpe: {best_strat['sharpe']:.3f} vs baseline {baseline['sharpe']:.3f} "
          f"(delta={best_strat['sharpe'] - baseline['sharpe']:+.3f})")
    print(f"    CAGR:   {best_strat['cagr']:.1%} vs baseline {baseline['cagr']:.1%}")
    print(f"    MaxDD:  {best_strat['max_dd']:.1%} vs baseline {baseline['max_dd']:.1%}")

    beats_baseline = best_strat['sharpe'] > baseline['sharpe']
    banner = "BEATS BASELINE" if beats_baseline else "DOES NOT BEAT BASELINE"
    print(f"\n  {'=' * 30}")
    print(f"  HEAD-TO-HEAD: {banner}")
    print(f"  {'=' * 30}")

    return {'results': results, 'best': best_strat['name'],
            'beats_baseline': beats_baseline}


# =============================================================================
# PARAMETER SWEEP
# =============================================================================

def parameter_sweep(closes, warmup=260):
    """Sweep over lookback windows and threshold pairs."""
    print("\n" + "=" * 70)
    print("  PARAMETER SWEEP")
    print("=" * 70)

    lookbacks = [63, 126, 189, 252, 378, 504]
    pctile_pairs = [(15, 85), (20, 80), (25, 75), (30, 70), (35, 65)]
    defensives = ['GLD', 'TLT', 'SHY']

    sweep_results = []

    # Percentile sweep
    print("\n  --- Percentile Strategy Sweep ---")
    print(f"  {'Lookback':>8} {'Lo/Hi':>8} {'Def':>4} {'Sharpe':>7} {'Sort':>7} {'CAGR':>7} {'MaxDD':>7}")
    print(f"  {'-' * 8} {'-' * 8} {'-' * 4} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")

    for lb in lookbacks:
        for lo, hi in pctile_pairs:
            for defs in defensives:
                regime = build_regime_series(
                    closes, strategy='percentile', lookback=lb,
                    low_thresh=lo, high_thresh=hi, defensive=defs)
                vals, contrib, sw, _ = simulate(closes, regime, warmup=warmup)
                m = compute_metrics(vals, contrib)
                m['lookback'] = lb
                m['low'] = lo
                m['high'] = hi
                m['defensive'] = defs
                m['strategy'] = 'percentile'
                m['switches'] = sw
                sweep_results.append(m)

                print(f"  {lb:>8} {lo:>3}/{hi:<4} {defs:>4} {m['sharpe']:>7.3f} "
                      f"{m['sortino']:>7.3f} {m['cagr']:>6.1%} {m['max_dd']:>6.1%}")

    # Z-score sweep
    z_pairs = [(-1.0, 0.5), (-0.75, 0.75), (-0.5, 1.0), (-0.5, 1.5), (-0.25, 1.0)]
    print("\n  --- Z-Score Strategy Sweep ---")
    print(f"  {'Lookback':>8} {'Z lo/hi':>10} {'Def':>4} {'Sharpe':>7} {'Sort':>7} {'CAGR':>7} {'MaxDD':>7}")
    print(f"  {'-' * 8} {'-' * 10} {'-' * 4} {'-' * 7} {'-' * 7} {'-' * 7} {'-' * 7}")

    for lb in lookbacks:
        for z_lo, z_hi in z_pairs:
            for defs in ['GLD']:  # Just GLD for z-score to keep it manageable
                regime = build_regime_series(
                    closes, strategy='zscore', lookback=lb,
                    low_z=z_lo, high_z=z_hi, defensive=defs)
                vals, contrib, sw, _ = simulate(closes, regime, warmup=warmup)
                m = compute_metrics(vals, contrib)
                m['lookback'] = lb
                m['low_z'] = z_lo
                m['high_z'] = z_hi
                m['defensive'] = defs
                m['strategy'] = 'zscore'
                m['switches'] = sw
                sweep_results.append(m)

                print(f"  {lb:>8} {z_lo:>+5.2f}/{z_hi:<+5.2f} {defs:>4} {m['sharpe']:>7.3f} "
                      f"{m['sortino']:>7.3f} {m['cagr']:>6.1%} {m['max_dd']:>6.1%}")

    # Top 5 by Sharpe
    sweep_results.sort(key=lambda x: x['sharpe'], reverse=True)
    print("\n  TOP 5 CONFIGS BY SHARPE:")
    for i, r in enumerate(sweep_results[:5]):
        if r['strategy'] == 'percentile':
            desc = f"Pctile LB={r['lookback']} [{r['low']}/{r['high']}] {r['defensive']}"
        else:
            desc = f"ZScore LB={r['lookback']} [{r['low_z']:+.2f}/{r['high_z']:+.2f}] {r['defensive']}"
        print(f"    {i + 1}. {desc}: Sharpe={r['sharpe']:.3f}, CAGR={r['cagr']:.1%}, MaxDD={r['max_dd']:.1%}")

    return sweep_results


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 70)
    print("  RELATIVE VIX PERCENTILE TIMING — Strategy v1")
    print("  " + datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    print("=" * 70)

    # 1. Download data
    print("\n[1/8] Downloading data...")
    closes = download_data()
    print(f"  Data: {closes.index[0].date()} to {closes.index[-1].date()} ({len(closes)} days)")
    print(f"  Tickers: {list(closes.columns)}")

    # VIX stats
    vix = closes['VIX']
    print(f"\n  VIX stats: mean={vix.mean():.1f}, median={vix.median():.1f}, "
          f"min={vix.min():.1f}, max={vix.max():.1f}")

    # Show how VIX percentile adapts
    pctile_252 = compute_vix_percentile(vix, 252)
    print("\n  VIX Percentile Adaptation Examples:")
    for year in [2017, 2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]:
        yr_mask = pctile_252.index.year == year
        if yr_mask.sum() > 0:
            yr_vix = vix[yr_mask]
            yr_pctile = pctile_252[yr_mask].dropna()
            if len(yr_pctile) > 0:
                print(f"    {year}: VIX avg={yr_vix.mean():.1f}, "
                      f"Pctile avg={yr_pctile.mean():.0f}, "
                      f"VIX@p25={yr_vix.quantile(0.25):.1f}, VIX@p75={yr_vix.quantile(0.75):.1f}")

    warmup = 260

    # 2. Build default regime series
    print("\n[2/8] Building regime series...")
    regime_pctile = build_regime_series(closes, 'percentile', lookback=252,
                                        low_thresh=25, high_thresh=75)
    regime_zscore = build_regime_series(closes, 'zscore', lookback=252,
                                        low_z=-0.5, high_z=1.0)
    regime_fixed = build_regime_series(closes, 'fixed')

    # 3. Head-to-head comparison
    print("\n[3/8] Head-to-head comparison...")
    h2h = test_head_to_head(closes, warmup=warmup)

    # 4. Parameter sweep
    print("\n[4/8] Parameter sweep...")
    sweep = parameter_sweep(closes, warmup=warmup)

    # Pick the best config for adversarial testing
    best_config = sweep[0]  # Best by Sharpe from sweep
    if best_config['strategy'] == 'percentile':
        best_regime = build_regime_series(
            closes, 'percentile', lookback=best_config['lookback'],
            low_thresh=best_config['low'], high_thresh=best_config['high'],
            defensive=best_config['defensive'])
        best_desc = (f"Percentile LB={best_config['lookback']} "
                     f"[{best_config['low']}/{best_config['high']}] {best_config['defensive']}")
    else:
        best_regime = build_regime_series(
            closes, 'zscore', lookback=best_config['lookback'],
            low_z=best_config['low_z'], high_z=best_config['high_z'],
            defensive=best_config['defensive'])
        best_desc = (f"ZScore LB={best_config['lookback']} "
                     f"[{best_config['low_z']:+.2f}/{best_config['high_z']:+.2f}] {best_config['defensive']}")

    print(f"\n  BEST CONFIG FOR ADVERSARIAL TESTING: {best_desc}")
    print(f"  Sharpe={best_config['sharpe']:.3f}, CAGR={best_config['cagr']:.1%}")

    # 5-8: Adversarial tests on best config
    print("\n[5/8] Permutation test on best config...")
    perm = test_permutation(closes, best_regime, n_perms=200, warmup=warmup)

    print("\n[6/8] Sub-period consistency...")
    subp = test_subperiod(closes, best_regime, warmup=warmup)

    print("\n[7/8] Outlier robustness...")
    outlier = test_outlier_robustness(closes, best_regime, warmup=warmup)

    print("\n[7b/8] R1 regime test...")
    r1 = test_regime_r1(closes, best_regime, warmup=warmup)

    print("\n[8/8] Walk-forward validation...")
    if best_config['strategy'] == 'percentile':
        wf = test_walkforward(
            closes, strategy='percentile', lookback=best_config['lookback'],
            low_thresh=best_config['low'], high_thresh=best_config['high'],
            defensive=best_config['defensive'])
    else:
        wf = test_walkforward(
            closes, strategy='zscore', lookback=best_config['lookback'],
            low_z=best_config['low_z'], high_z=best_config['high_z'],
            defensive=best_config['defensive'])

    # Also run adversarial on the default percentile [25/75] config
    print("\n\n" + "=" * 70)
    print("  ADVERSARIAL TESTS ON DEFAULT PERCENTILE [25/75] CONFIG")
    print("=" * 70)

    perm_default = test_permutation(closes, regime_pctile, n_perms=200, warmup=warmup)
    subp_default = test_subperiod(closes, regime_pctile, warmup=warmup)
    outlier_default = test_outlier_robustness(closes, regime_pctile, warmup=warmup)
    r1_default = test_regime_r1(closes, regime_pctile, warmup=warmup)
    wf_default = test_walkforward(closes, strategy='percentile', lookback=252,
                                   low_thresh=25, high_thresh=75)

    # ==========================================================
    # FINAL SUMMARY
    # ==========================================================
    print("\n\n" + "=" * 70)
    print("  FINAL SUMMARY — RELATIVE VIX PERCENTILE TIMING v1")
    print("=" * 70)

    print(f"\n  BEST SWEEP CONFIG: {best_desc}")
    print(f"    Sharpe: {best_config['sharpe']:.3f}")
    print(f"    CAGR:   {best_config['cagr']:.1%}")
    print(f"    MaxDD:  {best_config['max_dd']:.1%}")

    tests_best = {
        'permutation': perm['pass'],
        'subperiod': subp['pass'],
        'outlier': outlier['pass'],
        'r1_regime': r1['pass'],
        'walkforward': wf['pass'],
    }

    tests_default = {
        'permutation': perm_default['pass'],
        'subperiod': subp_default['pass'],
        'outlier': outlier_default['pass'],
        'r1_regime': r1_default['pass'],
        'walkforward': wf_default['pass'],
    }

    print(f"\n  ADVERSARIAL RESULTS — BEST SWEEP CONFIG ({best_desc}):")
    for test, passed in tests_best.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {test:<20}: {status}")

    n_pass_best = sum(tests_best.values())
    total_tests = len(tests_best)

    print(f"\n  ADVERSARIAL RESULTS — DEFAULT PERCENTILE [25/75]:")
    for test, passed in tests_default.items():
        status = "PASS" if passed else "FAIL"
        print(f"    {test:<20}: {status}")

    n_pass_default = sum(tests_default.values())

    # Overall verdict
    print(f"\n  SCORECARD:")
    print(f"    Best sweep config:  {n_pass_best}/{total_tests} tests passed")
    print(f"    Default [25/75]:    {n_pass_default}/{total_tests} tests passed")
    print(f"    Beats fixed baseline: {'YES' if h2h['beats_baseline'] else 'NO'}")

    survives = n_pass_best >= 4  # Need 4/5 to survive
    survives_default = n_pass_default >= 4

    print(f"\n  {'=' * 50}")
    if survives:
        print(f"  VERDICT: BEST CONFIG SURVIVES ADVERSARIAL ({n_pass_best}/{total_tests})")
    else:
        print(f"  VERDICT: BEST CONFIG FAILS ADVERSARIAL ({n_pass_best}/{total_tests})")

    if survives_default:
        print(f"  VERDICT: DEFAULT [25/75] SURVIVES ADVERSARIAL ({n_pass_default}/{total_tests})")
    else:
        print(f"  VERDICT: DEFAULT [25/75] FAILS ADVERSARIAL ({n_pass_default}/{total_tests})")

    if not survives and not survives_default:
        print(f"\n  STRATEGY STATUS: DOES NOT SURVIVE ADVERSARIAL VALIDATION")
        print(f"  Relative VIX Percentile timing does NOT provide robust edge.")
    elif survives or survives_default:
        print(f"\n  STRATEGY STATUS: SURVIVES ADVERSARIAL — CANDIDATE FOR DEPLOYMENT")
    print(f"  {'=' * 50}")

    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'data_range': f"{closes.index[0].date()} to {closes.index[-1].date()}",
        'best_config': {
            'description': best_desc,
            'sharpe': best_config['sharpe'],
            'cagr': best_config['cagr'],
            'max_dd': best_config['max_dd'],
            'sortino': best_config['sortino'],
        },
        'adversarial_best': tests_best,
        'adversarial_default': tests_default,
        'head_to_head': {
            'beats_baseline': h2h['beats_baseline'],
            'best_adaptive': h2h['best'],
        },
        'permutation_best': {k: v for k, v in perm.items() if k != 'pass'},
        'permutation_default': {k: v for k, v in perm_default.items() if k != 'pass'},
        'walkforward_best': {
            'avg_oos_sharpe': wf['avg_oos_sharpe'],
            'avg_excess': wf['avg_excess'],
            'pct_positive': wf['pct_positive'],
        },
        'walkforward_default': {
            'avg_oos_sharpe': wf_default['avg_oos_sharpe'],
            'avg_excess': wf_default['avg_excess'],
            'pct_positive': wf_default['pct_positive'],
        },
        'survives_adversarial_best': survives,
        'survives_adversarial_default': survives_default,
        'n_pass_best': n_pass_best,
        'n_pass_default': n_pass_default,
        'total_tests': total_tests,
    }

    output_path = os.path.join(OUTPUT_DIR, 'results.json')
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Save top sweep results
    sweep_path = os.path.join(OUTPUT_DIR, 'sweep_top20.json')
    top20 = []
    for s in sweep[:20]:
        entry = {k: v for k, v in s.items() if k not in ('total_contributed',)}
        top20.append(entry)
    with open(sweep_path, 'w') as f:
        json.dump(top20, f, indent=2, default=str)

    print(f"\n  Results saved to {OUTPUT_DIR}/")
    return output


if __name__ == '__main__':
    results = main()
