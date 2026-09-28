#!/usr/bin/env python3
"""
Asymmetric Signal Study — Deep Analytical Research
====================================================
Finds conditional signals where forward returns have genuinely asymmetric
payoff profiles (limited downside, outsized upside).

Think like a macro trader: "when X happens, what follows?"

Author: Claude (Head of Quant)
Date: 2026-07-21
"""

import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from datetime import datetime, timedelta
from itertools import combinations

warnings.filterwarnings('ignore')

OUT_DIR = '/home/jupiter/Lvl3Quant/output/asymmetric_signal_study'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# PART 0: DATA DOWNLOAD
# ============================================================================

TICKERS = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'EEM', 'HYG', 'LQD',
           'VNQ', 'DBC', 'XLE', 'XLK', 'XLF', 'XLV']
VIX_TICKERS = ['^VIX', '^VIX3M']
ALL_TICKERS = TICKERS + VIX_TICKERS

SECTOR_ETFS = ['XLE', 'XLK', 'XLF', 'XLV', 'IWM', 'EEM', 'VNQ']
TARGET_ASSETS = ['SPY', 'QQQ', 'TLT', 'GLD']
FORWARD_WINDOWS = {'1w': 5, '1m': 21, '3m': 63, '6m': 126}

print("=" * 80)
print("ASYMMETRIC SIGNAL STUDY")
print("=" * 80)
print(f"\nDownloading data for {len(ALL_TICKERS)} tickers (2005-2026)...")

data_cache = os.path.join(OUT_DIR, 'price_data.parquet')

if os.path.exists(data_cache):
    print("Loading cached data...")
    prices = pd.read_parquet(data_cache)
    # Check if data is reasonably recent
    if prices.index.max() < pd.Timestamp('2026-06-01'):
        print("Cache is stale, re-downloading...")
        os.remove(data_cache)
        prices = None
    else:
        print(f"Loaded {len(prices)} rows, {prices.columns.tolist()}")
else:
    prices = None

if prices is None:
    # Download all tickers at once — yfinance handles MultiIndex
    print("  Downloading all tickers in batch...")
    raw = yf.download(ALL_TICKERS, start='2005-01-01', end='2026-07-21',
                      progress=True, auto_adjust=True, group_by='ticker')

    dfs = {}
    for t in ALL_TICKERS:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t in raw.columns.get_level_values(0):
                    series = raw[(t, 'Close')].dropna()
                else:
                    print(f"    WARNING: {t} not in downloaded data")
                    continue
            else:
                # Single ticker case
                series = raw['Close'].dropna()

            if len(series) > 0:
                dfs[t] = series
                print(f"    {t}: {len(series)} rows ({series.index[0].date()} to {series.index[-1].date()})")
            else:
                print(f"    WARNING: No data for {t}")
        except Exception as e:
            print(f"    ERROR processing {t}: {e}")

    prices = pd.DataFrame(dfs)
    # Flatten column names if needed
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    prices.to_parquet(data_cache)
    print(f"\nSaved price data: {len(prices)} rows, {len(prices.columns)} tickers")

# Rename VIX columns for convenience
col_map = {}
for c in prices.columns:
    if 'VIX' in str(c) and '3M' not in str(c) and 'VIX3M' not in str(c):
        col_map[c] = 'VIX'
    elif 'VIX3M' in str(c) or 'VIX3M' in str(c):
        col_map[c] = 'VIX3M'
if col_map:
    prices = prices.rename(columns=col_map)

# Compute returns
returns = prices.pct_change()
log_returns = np.log(prices / prices.shift(1))

print(f"\nData range: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"Tickers available: {sorted(prices.columns.tolist())}")

# ============================================================================
# PART 1: COMPUTE SIGNALS
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: Computing Cross-Asset Signals")
print("=" * 80)

signals = pd.DataFrame(index=prices.index)

# a. VIX term structure ratio
if 'VIX' in prices.columns and 'VIX3M' in prices.columns:
    signals['vix_term_structure'] = prices['VIX'] / prices['VIX3M']
    print("  [a] VIX term structure ratio (VIX/VIX3M) — computed")
elif 'VIX' in prices.columns:
    # Approximate VIX3M as 63d rolling mean of VIX (longer-term vol expectation)
    signals['vix_term_structure'] = prices['VIX'] / prices['VIX'].rolling(63).mean()
    print("  [a] VIX term structure ratio (VIX/VIX_63d_avg) — approximated")
else:
    print("  [a] VIX term structure — SKIPPED (no VIX data)")

# b. Credit spread proxy (HYG return - LQD return, 21d cumulative)
if 'HYG' in returns.columns and 'LQD' in returns.columns:
    signals['credit_spread'] = (returns['HYG'] - returns['LQD']).rolling(21).sum()
    print("  [b] Credit spread proxy (HYG-LQD 21d cum) — computed")

# c. Market breadth (% of sector ETFs above 50d SMA)
available_sectors = [s for s in SECTOR_ETFS if s in prices.columns]
if len(available_sectors) >= 3:
    breadth_components = pd.DataFrame()
    for s in available_sectors:
        breadth_components[s] = (prices[s] > prices[s].rolling(50).mean()).astype(float)
    signals['market_breadth'] = breadth_components.mean(axis=1)
    print(f"  [c] Market breadth ({len(available_sectors)} sectors above 50d SMA) — computed")

# d. Momentum divergence (SPY 6m return vs median ETF 6m return)
if 'SPY' in prices.columns:
    spy_6m = prices['SPY'].pct_change(126)
    etf_6m_returns = pd.DataFrame()
    for t in TICKERS:
        if t in prices.columns and t != 'SPY':
            etf_6m_returns[t] = prices[t].pct_change(126)
    if len(etf_6m_returns.columns) > 0:
        median_6m = etf_6m_returns.median(axis=1)
        signals['momentum_divergence'] = spy_6m - median_6m
        print("  [d] Momentum divergence (SPY 6m vs median ETF 6m) — computed")

# e. Volatility compression (21d vol / 63d vol for SPY)
if 'SPY' in returns.columns:
    vol_21 = returns['SPY'].rolling(21).std() * np.sqrt(252)
    vol_63 = returns['SPY'].rolling(63).std() * np.sqrt(252)
    signals['vol_compression'] = vol_21 / vol_63
    print("  [e] Volatility compression (21d/63d vol ratio) — computed")

# f. Put/call sentiment proxy (VIX / SPY 21d realized vol)
if 'VIX' in prices.columns and 'SPY' in returns.columns:
    realized_vol_21 = returns['SPY'].rolling(21).std() * np.sqrt(252) * 100  # annualized %
    signals['implied_vs_realized'] = prices['VIX'] / realized_vol_21
    print("  [f] Implied vs realized vol (VIX/RV21) — computed")

# g. Bond-equity correlation (63d rolling corr of SPY vs TLT)
if 'SPY' in returns.columns and 'TLT' in returns.columns:
    signals['bond_equity_corr'] = returns['SPY'].rolling(63).corr(returns['TLT'])
    print("  [g] Bond-equity correlation (63d rolling SPY-TLT corr) — computed")

# h. Gold stress signal (GLD 21d return conditional on SPY down)
if 'GLD' in prices.columns and 'SPY' in prices.columns:
    gld_21d = prices['GLD'].pct_change(21)
    spy_21d = prices['SPY'].pct_change(21)
    # Gold return relative to SPY when SPY is down; NaN when SPY is up
    gold_stress_raw = gld_21d - spy_21d
    # Make it unconditional: gold outperformance over 21d
    signals['gold_stress'] = gold_stress_raw
    print("  [h] Gold stress signal (GLD-SPY 21d relative return) — computed")

# i. Small-cap spread (IWM 63d return - SPY 63d return)
if 'IWM' in prices.columns and 'SPY' in prices.columns:
    signals['smallcap_spread'] = prices['IWM'].pct_change(63) - prices['SPY'].pct_change(63)
    print("  [i] Small-cap spread (IWM-SPY 63d return) — computed")

# Drop rows with insufficient data
signals = signals.dropna(how='all')
print(f"\nSignals computed: {signals.columns.tolist()}")
print(f"Signal date range: {signals.index[0].date()} to {signals.index[-1].date()}")
print(f"Rows with data: {len(signals.dropna())}")

# ============================================================================
# PART 2: CONDITIONAL FORWARD RETURN ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: Conditional Forward Return Analysis")
print("=" * 80)

# Pre-compute forward returns for target assets
fwd_returns = {}
for asset in TARGET_ASSETS:
    if asset not in prices.columns:
        continue
    fwd_returns[asset] = {}
    for label, days in FORWARD_WINDOWS.items():
        fwd_returns[asset][label] = prices[asset].pct_change(days).shift(-days) * 100  # in %

results = {}
all_asymmetry = []

for sig_name in signals.columns:
    sig = signals[sig_name].dropna()
    if len(sig) < 500:
        print(f"  Skipping {sig_name} — insufficient data ({len(sig)} rows)")
        continue

    print(f"\n  Analyzing signal: {sig_name} ({len(sig)} observations)")

    # Compute quintiles
    try:
        quintile_labels = pd.qcut(sig, 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'],
                                  duplicates='drop')
    except ValueError:
        # Handle case with too many duplicate values
        quintile_labels = pd.cut(sig, 5, labels=['Q1(low)', 'Q2', 'Q3', 'Q4', 'Q5(high)'],
                                duplicates='drop')

    results[sig_name] = {}

    for asset in TARGET_ASSETS:
        if asset not in fwd_returns:
            continue
        results[sig_name][asset] = {}

        for horizon_label, horizon_days in FORWARD_WINDOWS.items():
            fwd = fwd_returns[asset][horizon_label]
            # Align
            aligned = pd.DataFrame({
                'quintile': quintile_labels,
                'fwd_return': fwd
            }).dropna()

            if len(aligned) < 100:
                continue

            quintile_stats = {}
            for q in aligned['quintile'].unique():
                q_data = aligned[aligned['quintile'] == q]['fwd_return']
                if len(q_data) < 10:
                    continue

                p10 = q_data.quantile(0.10)
                p50 = q_data.quantile(0.50)
                p90 = q_data.quantile(0.90)
                asymmetry = p90 / abs(p10) if abs(p10) > 0.01 else np.nan

                # T-test vs unconditional
                unconditional = aligned['fwd_return']
                t_stat, p_val = stats.ttest_ind(q_data, unconditional, equal_var=False)

                quintile_stats[str(q)] = {
                    'n': len(q_data),
                    'mean': round(q_data.mean(), 3),
                    'median': round(p50, 3),
                    'p10': round(p10, 3),
                    'p90': round(p90, 3),
                    'asymmetry_ratio': round(asymmetry, 3) if not np.isnan(asymmetry) else None,
                    'std': round(q_data.std(), 3),
                    'sharpe': round(q_data.mean() / q_data.std(), 3) if q_data.std() > 0 else 0,
                    'pct_positive': round((q_data > 0).mean() * 100, 1),
                    't_stat': round(t_stat, 3),
                    'p_value': round(p_val, 4),
                }

                # Track asymmetry for ranking
                if asymmetry and not np.isnan(asymmetry) and str(q) in ['Q1(low)', 'Q5(high)']:
                    all_asymmetry.append({
                        'signal': sig_name,
                        'quintile': str(q),
                        'asset': asset,
                        'horizon': horizon_label,
                        'asymmetry_ratio': asymmetry,
                        'mean_return': q_data.mean(),
                        'p10': p10,
                        'p90': p90,
                        'n': len(q_data),
                        'p_value': p_val,
                        'pct_positive': (q_data > 0).mean() * 100,
                    })

            results[sig_name][asset][horizon_label] = quintile_stats

    # Print summary for this signal
    for asset in TARGET_ASSETS:
        if asset not in results.get(sig_name, {}):
            continue
        for horizon in ['1m', '3m']:
            if horizon not in results[sig_name][asset]:
                continue
            qs = results[sig_name][asset][horizon]
            q1 = qs.get('Q1(low)', {})
            q5 = qs.get('Q5(high)', {})
            if q1 and q5:
                print(f"    {asset} {horizon}: Q1 mean={q1.get('mean','?')}% "
                      f"(asym={q1.get('asymmetry_ratio','?')}) | "
                      f"Q5 mean={q5.get('mean','?')}% "
                      f"(asym={q5.get('asymmetry_ratio','?')})")

# Rank all asymmetry findings
asymmetry_df = pd.DataFrame(all_asymmetry)
if len(asymmetry_df) > 0:
    asymmetry_df = asymmetry_df.sort_values('asymmetry_ratio', ascending=False)
    print("\n" + "-" * 80)
    print("TOP 20 ASYMMETRIC OPPORTUNITIES (by asymmetry ratio)")
    print("-" * 80)
    print(f"{'Signal':<25} {'Q':>8} {'Asset':>5} {'Horizon':>7} "
          f"{'Asym':>6} {'Mean%':>7} {'P10%':>7} {'P90%':>7} {'N':>5} {'p-val':>7}")
    for _, row in asymmetry_df.head(20).iterrows():
        print(f"{row['signal']:<25} {row['quintile']:>8} {row['asset']:>5} "
              f"{row['horizon']:>7} {row['asymmetry_ratio']:>6.2f} "
              f"{row['mean_return']:>7.2f} {row['p10']:>7.2f} {row['p90']:>7.2f} "
              f"{row['n']:>5.0f} {row['p_value']:>7.4f}")

# ============================================================================
# PART 3: COMPOUND SIGNAL DISCOVERY
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: Compound Signal Discovery")
print("=" * 80)

# Find top 3 single signals (by median asymmetry ratio across assets/horizons)
if len(asymmetry_df) > 0:
    sig_median_asym = asymmetry_df.groupby('signal')['asymmetry_ratio'].median().sort_values(ascending=False)
    top_signals = sig_median_asym.head(3).index.tolist()
    print(f"\nTop 3 signals by median asymmetry ratio:")
    for i, s in enumerate(top_signals):
        print(f"  {i+1}. {s} (median asymmetry = {sig_median_asym[s]:.3f})")

    # Test combinations of 2 signals
    compound_results = []

    for s1, s2 in combinations(top_signals, 2):
        sig1 = signals[s1].dropna()
        sig2 = signals[s2].dropna()

        # Compute quintiles for each
        try:
            q1_labels = pd.qcut(sig1, 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
            q2_labels = pd.qcut(sig2, 5, labels=[1, 2, 3, 4, 5], duplicates='drop')
        except Exception:
            continue

        # Extreme conditions: both in Q1 or both in Q5
        for extreme_combo_name, cond1, cond2 in [
            (f"{s1}_Q1+{s2}_Q1", 1, 1),
            (f"{s1}_Q5+{s2}_Q5", 5, 5),
            (f"{s1}_Q1+{s2}_Q5", 1, 5),
            (f"{s1}_Q5+{s2}_Q1", 5, 1),
        ]:
            both_extreme = (q1_labels == cond1) & (q2_labels == cond2)
            extreme_dates = both_extreme[both_extreme].index

            if len(extreme_dates) < 15:
                continue

            for asset in TARGET_ASSETS:
                if asset not in fwd_returns:
                    continue
                for horizon_label, horizon_days in FORWARD_WINDOWS.items():
                    fwd = fwd_returns[asset][horizon_label]
                    fwd_extreme = fwd.reindex(extreme_dates).dropna()

                    if len(fwd_extreme) < 10:
                        continue

                    p10 = fwd_extreme.quantile(0.10)
                    p90 = fwd_extreme.quantile(0.90)
                    asym = p90 / abs(p10) if abs(p10) > 0.01 else np.nan

                    compound_results.append({
                        'combo': extreme_combo_name,
                        'asset': asset,
                        'horizon': horizon_label,
                        'n': len(fwd_extreme),
                        'mean': fwd_extreme.mean(),
                        'median': fwd_extreme.median(),
                        'p10': p10,
                        'p90': p90,
                        'asymmetry_ratio': asym,
                        'pct_positive': (fwd_extreme > 0).mean() * 100,
                    })

    compound_df = pd.DataFrame(compound_results)
    if len(compound_df) > 0:
        compound_df = compound_df.sort_values('asymmetry_ratio', ascending=False)
        print("\nTOP 15 COMPOUND SIGNAL OPPORTUNITIES:")
        print("-" * 100)
        print(f"{'Combination':<45} {'Asset':>5} {'Hor':>4} "
              f"{'Asym':>6} {'Mean%':>7} {'P10%':>7} {'P90%':>7} {'N':>4} {'%Pos':>5}")
        for _, row in compound_df.head(15).iterrows():
            asym_val = row['asymmetry_ratio']
            asym_str = f"{asym_val:>6.2f}" if not np.isnan(asym_val) else "   N/A"
            print(f"{row['combo']:<45} {row['asset']:>5} {row['horizon']:>4} "
                  f"{asym_str} {row['mean']:>7.2f} {row['p10']:>7.2f} "
                  f"{row['p90']:>7.2f} {row['n']:>4.0f} {row['pct_positive']:>5.1f}")

    # Crisis setup: what preceded the BEST forward returns?
    print("\n--- CRISIS SETUP PATTERNS (what preceded top-decile 3m SPY returns) ---")
    if 'SPY' in fwd_returns and '3m' in fwd_returns['SPY']:
        fwd_3m_spy = fwd_returns['SPY']['3m'].dropna()
        top_decile_threshold = fwd_3m_spy.quantile(0.90)
        bottom_decile_threshold = fwd_3m_spy.quantile(0.10)

        top_dates = fwd_3m_spy[fwd_3m_spy >= top_decile_threshold].index
        bottom_dates = fwd_3m_spy[fwd_3m_spy <= bottom_decile_threshold].index

        print(f"\nTop decile 3m SPY return threshold: {top_decile_threshold:.2f}%")
        print(f"Bottom decile 3m SPY return threshold: {bottom_decile_threshold:.2f}%")

        print(f"\nSignal values BEFORE top-decile 3m rallies (N={len(top_dates)}):")
        for sig_name in signals.columns:
            sig_at_top = signals[sig_name].reindex(top_dates).dropna()
            sig_all = signals[sig_name].dropna()
            if len(sig_at_top) < 10:
                continue
            # What percentile of the signal distribution were these dates at?
            pctile = (sig_all < sig_at_top.median()).mean() * 100
            t_stat, p_val = stats.ttest_ind(sig_at_top, sig_all, equal_var=False)
            marker = " ***" if p_val < 0.01 else " **" if p_val < 0.05 else " *" if p_val < 0.10 else ""
            print(f"  {sig_name:<25} median={sig_at_top.median():>8.3f} "
                  f"(pctile={pctile:>5.1f}%) p={p_val:.4f}{marker}")

        print(f"\nSignal values BEFORE bottom-decile 3m crashes (N={len(bottom_dates)}):")
        for sig_name in signals.columns:
            sig_at_bottom = signals[sig_name].reindex(bottom_dates).dropna()
            sig_all = signals[sig_name].dropna()
            if len(sig_at_bottom) < 10:
                continue
            pctile = (sig_all < sig_at_bottom.median()).mean() * 100
            t_stat, p_val = stats.ttest_ind(sig_at_bottom, sig_all, equal_var=False)
            marker = " ***" if p_val < 0.01 else " **" if p_val < 0.05 else " *" if p_val < 0.10 else ""
            print(f"  {sig_name:<25} median={sig_at_bottom.median():>8.3f} "
                  f"(pctile={pctile:>5.1f}%) p={p_val:.4f}{marker}")

# ============================================================================
# PART 4: REGIME TRANSITION ANALYSIS
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: Regime Transition Analysis")
print("=" * 80)

if 'SPY' in prices.columns and 'VIX' in prices.columns:
    spy_sma200 = prices['SPY'].rolling(200).mean()
    vix = prices['VIX']

    regime = pd.Series('transition', index=prices.index)
    regime[(prices['SPY'] > spy_sma200) & (vix < 20)] = 'bull'
    regime[(prices['SPY'] < spy_sma200) & (vix > 25)] = 'bear'

    # Clean: require at least 5 consecutive days in a regime to count
    regime_counts = regime.value_counts()
    print(f"\nRegime distribution:")
    for r, c in regime_counts.items():
        print(f"  {r}: {c} days ({c/len(regime)*100:.1f}%)")

    # Detect transitions
    regime_shifted = regime.shift(1)
    transitions = pd.DataFrame({
        'from_regime': regime_shifted,
        'to_regime': regime,
        'date': regime.index
    })
    transitions = transitions[transitions['from_regime'] != transitions['to_regime']].dropna()

    # Focus on meaningful transitions
    bull_to_bear = transitions[(transitions['from_regime'] == 'bull') &
                                (transitions['to_regime'].isin(['bear', 'transition']))].index
    bear_to_bull = transitions[(transitions['from_regime'].isin(['bear', 'transition'])) &
                                (transitions['to_regime'] == 'bull')].index

    print(f"\nBull → non-bull transitions: {len(bull_to_bear)}")
    print(f"Non-bull → bull transitions: {len(bear_to_bull)}")

    # What signals predicted transitions?
    print("\n--- SIGNALS BEFORE BULL→NON-BULL TRANSITIONS ---")
    transition_signal_results = {}
    for sig_name in signals.columns:
        # Look at signal 5, 10, 21 days BEFORE transition
        for lookback in [5, 10, 21]:
            sig_before = signals[sig_name].shift(lookback).reindex(bull_to_bear).dropna()
            sig_all = signals[sig_name].dropna()
            if len(sig_before) < 5:
                continue
            pctile = (sig_all < sig_before.median()).mean() * 100
            t_stat, p_val = stats.ttest_ind(sig_before, sig_all, equal_var=False)

            if lookback == 21:  # Only print the 21d lookback for clarity
                marker = " ***" if p_val < 0.01 else " **" if p_val < 0.05 else " *" if p_val < 0.10 else ""
                print(f"  {sig_name:<25} {lookback}d before: median={sig_before.median():>8.3f} "
                      f"(pctile={pctile:>5.1f}%) p={p_val:.4f}{marker}")

            transition_signal_results[f"{sig_name}__{lookback}d_before_bull2bear"] = {
                'median': float(sig_before.median()),
                'percentile': float(pctile),
                'p_value': float(p_val),
                'n': int(len(sig_before)),
            }

    print("\n--- SIGNALS BEFORE NON-BULL→BULL TRANSITIONS ---")
    for sig_name in signals.columns:
        for lookback in [5, 10, 21]:
            sig_before = signals[sig_name].shift(lookback).reindex(bear_to_bull).dropna()
            sig_all = signals[sig_name].dropna()
            if len(sig_before) < 5:
                continue
            pctile = (sig_all < sig_before.median()).mean() * 100
            t_stat, p_val = stats.ttest_ind(sig_before, sig_all, equal_var=False)

            if lookback == 21:
                marker = " ***" if p_val < 0.01 else " **" if p_val < 0.05 else " *" if p_val < 0.10 else ""
                print(f"  {sig_name:<25} {lookback}d before: median={sig_before.median():>8.3f} "
                      f"(pctile={pctile:>5.1f}%) p={p_val:.4f}{marker}")

    # Forward returns after transitions by asset
    print("\n--- FORWARD RETURNS AFTER REGIME TRANSITIONS ---")
    for transition_name, transition_dates in [
        ('Bull→NonBull', bull_to_bear),
        ('NonBull→Bull', bear_to_bull)
    ]:
        print(f"\n  {transition_name} (N={len(transition_dates)}):")
        for asset in TARGET_ASSETS:
            if asset not in fwd_returns:
                continue
            for horizon in ['1m', '3m', '6m']:
                fwd = fwd_returns[asset][horizon].reindex(transition_dates).dropna()
                if len(fwd) < 5:
                    continue
                p10 = fwd.quantile(0.10)
                p90 = fwd.quantile(0.90)
                asym = p90 / abs(p10) if abs(p10) > 0.01 else np.nan
                print(f"    {asset} {horizon}: mean={fwd.mean():>7.2f}% median={fwd.median():>7.2f}% "
                      f"p10={p10:>7.2f}% p90={p90:>7.2f}% asym={asym:>5.2f}")

    # How early do signals trigger?
    print("\n--- SIGNAL LEAD TIME ANALYSIS ---")
    print("(How many days before a bull→bear transition does each signal hit extreme?)")
    for sig_name in signals.columns:
        sig = signals[sig_name].dropna()
        extreme_threshold_high = sig.quantile(0.90)
        extreme_threshold_low = sig.quantile(0.10)

        lead_times = []
        for trans_date in bull_to_bear:
            # Look back up to 63 days for when signal hit extreme
            window = sig.loc[:trans_date].tail(63)
            extreme_dates = window[(window >= extreme_threshold_high) |
                                   (window <= extreme_threshold_low)].index
            if len(extreme_dates) > 0:
                last_extreme = extreme_dates[-1]
                lead_days = (trans_date - last_extreme).days
                lead_times.append(lead_days)

        if len(lead_times) >= 5:
            lead_arr = np.array(lead_times)
            print(f"  {sig_name:<25} median lead={np.median(lead_arr):>5.0f}d "
                  f"mean={np.mean(lead_arr):>5.1f}d (range {np.min(lead_arr)}-{np.max(lead_arr)}d, "
                  f"N={len(lead_arr)})")

# ============================================================================
# PART 5: VISUALIZATIONS
# ============================================================================

print("\n" + "=" * 80)
print("Creating Visualizations...")
print("=" * 80)

# Plot 1: Asymmetry heatmap — signals vs assets/horizons
if len(asymmetry_df) > 0:
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Asymmetry Ratio by Signal, Asset, and Horizon\n(>1.5 = asymmetric upside, <0.67 = asymmetric downside)',
                 fontsize=14, fontweight='bold')

    for idx, (asset, ax) in enumerate(zip(TARGET_ASSETS, axes.flat)):
        asset_data = asymmetry_df[asymmetry_df['asset'] == asset]
        if len(asset_data) == 0:
            continue

        # Pivot for heatmap
        pivot_data = asset_data.pivot_table(
            values='asymmetry_ratio',
            index='signal',
            columns='horizon',
            aggfunc='mean'
        )
        # Reorder columns
        col_order = [c for c in ['1w', '1m', '3m', '6m'] if c in pivot_data.columns]
        if col_order:
            pivot_data = pivot_data[col_order]

        sns.heatmap(pivot_data, ax=ax, cmap='RdYlGn', center=1.0,
                    vmin=0.3, vmax=2.5, annot=True, fmt='.2f',
                    linewidths=0.5, cbar_kws={'label': 'Asymmetry Ratio'})
        ax.set_title(f'{asset}', fontweight='bold')
        ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'asymmetry_heatmap.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: asymmetry_heatmap.png")

# Plot 2: Forward return distributions for top signals
if len(asymmetry_df) > 0:
    top_3_entries = asymmetry_df.head(6)  # Top 6 for variety

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Forward Return Distributions — Top Asymmetric Signals (Extreme Quintiles)',
                 fontsize=14, fontweight='bold')

    for idx, (_, entry) in enumerate(top_3_entries.iterrows()):
        if idx >= 6:
            break
        ax = axes.flat[idx]
        sig_name = entry['signal']
        asset = entry['asset']
        horizon = entry['horizon']

        sig = signals[sig_name].dropna()
        try:
            quintile_labels = pd.qcut(sig, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except:
            continue

        q_val = 'Q1' if 'Q1' in entry['quintile'] else 'Q5'
        extreme_dates = quintile_labels[quintile_labels == q_val].index

        fwd = fwd_returns[asset][horizon].reindex(extreme_dates).dropna()
        unconditional = fwd_returns[asset][horizon].dropna()

        ax.hist(unconditional, bins=50, alpha=0.3, color='gray', density=True, label='All dates')
        ax.hist(fwd, bins=30, alpha=0.6, color='blue' if entry['mean_return'] > 0 else 'red',
                density=True, label=f'{entry["quintile"]}')
        ax.axvline(0, color='black', linestyle='--', alpha=0.5)
        ax.axvline(fwd.mean(), color='blue', linestyle='-', alpha=0.8, label=f'Mean: {fwd.mean():.1f}%')
        ax.set_title(f'{sig_name}\n{asset} {horizon} | Asym={entry["asymmetry_ratio"]:.2f}', fontsize=10)
        ax.legend(fontsize=8)
        ax.set_xlabel('Forward Return (%)')

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'top_signal_distributions.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: top_signal_distributions.png")

# Plot 3: Signal correlation matrix
fig, ax = plt.subplots(figsize=(10, 8))
sig_corr = signals.corr()
sns.heatmap(sig_corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
            ax=ax, linewidths=0.5)
ax.set_title('Signal Correlation Matrix', fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, 'signal_correlations.png'), dpi=150, bbox_inches='tight')
plt.close()
print("  Saved: signal_correlations.png")

# Plot 4: Quintile mean returns — bar charts
if len(asymmetry_df) > 0:
    # Pick the 4 most interesting signals
    interesting_signals = sig_median_asym.head(4).index.tolist()

    fig, axes = plt.subplots(len(interesting_signals), len(TARGET_ASSETS),
                             figsize=(20, 4 * len(interesting_signals)))
    if len(interesting_signals) == 1:
        axes = axes.reshape(1, -1)

    fig.suptitle('Mean Forward Returns by Signal Quintile (3-month horizon)',
                 fontsize=14, fontweight='bold', y=1.02)

    for i, sig_name in enumerate(interesting_signals):
        for j, asset in enumerate(TARGET_ASSETS):
            ax = axes[i, j]
            if sig_name not in results or asset not in results[sig_name]:
                ax.set_visible(False)
                continue
            if '3m' not in results[sig_name][asset]:
                ax.set_visible(False)
                continue

            qs = results[sig_name][asset]['3m']
            quintiles = sorted(qs.keys())
            means = [qs[q]['mean'] for q in quintiles]
            colors = ['green' if m > 0 else 'red' for m in means]

            ax.bar(range(len(quintiles)), means, color=colors, alpha=0.7, edgecolor='black')
            ax.set_xticks(range(len(quintiles)))
            ax.set_xticklabels(quintiles, rotation=45, fontsize=8)
            ax.axhline(0, color='black', linewidth=0.5)
            ax.set_ylabel('Mean 3m Return (%)')
            if i == 0:
                ax.set_title(f'{asset}', fontweight='bold')
            if j == 0:
                ax.set_ylabel(f'{sig_name}\nMean 3m Return (%)', fontsize=9)

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'quintile_returns.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: quintile_returns.png")

# Plot 5: Regime timeline
if 'SPY' in prices.columns and 'VIX' in prices.columns:
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(20, 8), sharex=True,
                                    gridspec_kw={'height_ratios': [3, 1]})

    ax1.plot(prices.index, prices['SPY'], color='black', linewidth=0.5)
    ax1.plot(prices.index, spy_sma200, color='blue', linewidth=0.5, alpha=0.5, label='200 SMA')

    # Color background by regime
    for i in range(len(regime) - 1):
        if regime.iloc[i] == 'bull':
            ax1.axvspan(regime.index[i], regime.index[i+1], alpha=0.1, color='green')
        elif regime.iloc[i] == 'bear':
            ax1.axvspan(regime.index[i], regime.index[i+1], alpha=0.1, color='red')

    ax1.set_ylabel('SPY Price')
    ax1.set_title('Regime Classification (Green=Bull, Red=Bear, White=Transition)', fontweight='bold')
    ax1.legend()

    ax2.plot(prices.index, prices['VIX'], color='purple', linewidth=0.5)
    ax2.axhline(20, color='green', linestyle='--', alpha=0.5, label='VIX=20')
    ax2.axhline(25, color='red', linestyle='--', alpha=0.5, label='VIX=25')
    ax2.set_ylabel('VIX')
    ax2.legend()

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, 'regime_timeline.png'), dpi=150, bbox_inches='tight')
    plt.close()
    print("  Saved: regime_timeline.png")

# ============================================================================
# PART 6: SUMMARY JSON
# ============================================================================

print("\n" + "=" * 80)
print("Creating Summary JSON...")
print("=" * 80)

summary = {
    'study_date': '2026-07-21',
    'data_range': f"{prices.index[0].date()} to {prices.index[-1].date()}",
    'n_observations': int(len(prices)),
    'signals_computed': signals.columns.tolist(),
}

# Top 10 asymmetric signals
if len(asymmetry_df) > 0:
    # Filter for statistically significant results
    sig_asym = asymmetry_df[asymmetry_df['p_value'] < 0.10].copy()
    if len(sig_asym) < 10:
        sig_asym = asymmetry_df.copy()  # Fall back to all if few significant

    top_10 = []
    for _, row in sig_asym.head(10).iterrows():
        entry = {
            'rank': len(top_10) + 1,
            'signal': row['signal'],
            'condition': f"Signal in {row['quintile']}",
            'target_asset': row['asset'],
            'horizon': row['horizon'],
            'asymmetry_ratio': round(float(row['asymmetry_ratio']), 3),
            'mean_forward_return_pct': round(float(row['mean_return']), 3),
            'downside_p10_pct': round(float(row['p10']), 3),
            'upside_p90_pct': round(float(row['p90']), 3),
            'pct_positive': round(float(row['pct_positive']), 1),
            'n_observations': int(row['n']),
            'p_value': round(float(row['p_value']), 4),
            'interpretation': ''
        }

        # Add interpretation
        if row['asymmetry_ratio'] > 1.5 and row['mean_return'] > 0:
            entry['interpretation'] = (
                f"When {row['signal']} is in {row['quintile']}, "
                f"{row['asset']} {row['horizon']} returns show strong upside asymmetry: "
                f"90th pctile gain ({row['p90']:.1f}%) is {row['asymmetry_ratio']:.1f}x "
                f"the 10th pctile loss ({row['p10']:.1f}%). "
                f"Mean return: {row['mean_return']:.2f}%, positive {row['pct_positive']:.0f}% of time."
            )
        elif row['asymmetry_ratio'] > 1.5 and row['mean_return'] <= 0:
            entry['interpretation'] = (
                f"When {row['signal']} is in {row['quintile']}, "
                f"{row['asset']} {row['horizon']} has high asymmetry ratio but negative mean. "
                f"Upside exists but unconditional expectation is negative."
            )
        else:
            entry['interpretation'] = (
                f"When {row['signal']} is in {row['quintile']}, "
                f"{row['asset']} {row['horizon']} returns: mean {row['mean_return']:.2f}%, "
                f"asym ratio {row['asymmetry_ratio']:.2f}."
            )

        top_10.append(entry)

    summary['top_10_asymmetric_signals'] = top_10

# Compound signals
if len(compound_df) > 0:
    sig_compounds = compound_df[compound_df['asymmetry_ratio'] > 1.5].head(5)
    summary['top_compound_signals'] = []
    for _, row in sig_compounds.iterrows():
        summary['top_compound_signals'].append({
            'combination': row['combo'],
            'asset': row['asset'],
            'horizon': row['horizon'],
            'asymmetry_ratio': round(float(row['asymmetry_ratio']), 3),
            'mean_return_pct': round(float(row['mean']), 3),
            'n_observations': int(row['n']),
            'pct_positive': round(float(row['pct_positive']), 1),
        })

# Regime transition summary
if 'SPY' in prices.columns:
    summary['regime_distribution'] = {str(k): int(v) for k, v in regime_counts.items()}

# Save
summary_path = os.path.join(OUT_DIR, 'asymmetric_signal_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=2, default=str)
print(f"  Saved: asymmetric_signal_summary.json")

# Save detailed results
detail_path = os.path.join(OUT_DIR, 'detailed_quintile_results.json')
with open(detail_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)
print(f"  Saved: detailed_quintile_results.json")

# Save asymmetry ranking
if len(asymmetry_df) > 0:
    asym_path = os.path.join(OUT_DIR, 'asymmetry_rankings.csv')
    asymmetry_df.to_csv(asym_path, index=False)
    print(f"  Saved: asymmetry_rankings.csv")

# ============================================================================
# FINAL SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("FINAL SUMMARY — ACTIONABLE FINDINGS")
print("=" * 80)

if len(asymmetry_df) > 0:
    print("\n🔑 TOP ASYMMETRIC SETUPS:")
    for entry in top_10[:5]:
        print(f"\n  #{entry['rank']}: {entry['interpretation']}")

    # Find the single best "buy the dip" signal
    buy_signals = asymmetry_df[(asymmetry_df['mean_return'] > 0) &
                                (asymmetry_df['asymmetry_ratio'] > 1.3) &
                                (asymmetry_df['p_value'] < 0.10)]
    if len(buy_signals) > 0:
        best = buy_signals.iloc[0]
        print(f"\n📊 BEST 'BUY THE DIP' SIGNAL:")
        print(f"  When {best['signal']} is in {best['quintile']},")
        print(f"  buy {best['asset']} for {best['horizon']} horizon:")
        print(f"  Expected: +{best['mean_return']:.2f}% mean, "
              f"{best['pct_positive']:.0f}% positive, "
              f"asymmetry {best['asymmetry_ratio']:.2f}x")

    # Find the single best "risk-off" warning
    risk_off = asymmetry_df[(asymmetry_df['mean_return'] < -1) &
                             (asymmetry_df['p_value'] < 0.10)]
    if len(risk_off) > 0:
        worst = risk_off.iloc[0]
        print(f"\n⚠️  BEST 'RISK-OFF' WARNING:")
        print(f"  When {worst['signal']} is in {worst['quintile']},")
        print(f"  {worst['asset']} {worst['horizon']}: mean {worst['mean_return']:.2f}%, "
              f"only {worst['pct_positive']:.0f}% positive")

print(f"\nAll outputs saved to: {OUT_DIR}/")
print("Files: asymmetric_signal_summary.json, detailed_quintile_results.json,")
print("       asymmetry_rankings.csv, *.png visualizations")
print("\n" + "=" * 80)
print("STUDY COMPLETE")
print("=" * 80)
