#!/usr/bin/env python3
"""
Correlation Timing Strategy v1
==============================
HC #735: Observation-first approach.
HC #734: Rare high-confidence events don't need confluence.
HC #428 R1: Must test across ALL regimes, report per-regime metrics.

OBSERVATION: Sector average correlation has IC=0.36 for predicting SPY 3-month returns.
When correlations spike (stress), mean 6-month return is +13.4% with 92% WR, 29x asymmetry.

PHASES:
  1. Deep observation of sector correlation index
  2. Three strategy variants (threshold, proportional, sector rotation)
  3. Validation (permutation test, regime test, sub-period stability, lag sensitivity)
"""

import json
import warnings
import os
import sys
from pathlib import Path
from datetime import datetime
from itertools import combinations

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import seaborn as sns
from scipy import stats

warnings.filterwarnings('ignore')

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/correlation_timing_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

SECTOR_ETFS = ['XLB', 'XLC', 'XLE', 'XLF', 'XLI', 'XLK', 'XLP', 'XLRE', 'XLU', 'XLV', 'XLY']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTOR_ETFS + [BENCHMARK]

CORR_WINDOW = 63  # ~3 months rolling correlation
RF_ANNUAL = 0.04  # risk-free rate for Sharpe calc
TRADING_DAYS_YEAR = 252

START_DATE = '2008-01-01'
END_DATE = '2026-07-18'

# Strategy parameters
THRESHOLD_PCT = 80  # Variant A: percentile threshold for "stress" regime
PROPORTIONAL_FLOOR = 0.60  # Variant B: min equity weight
PROPORTIONAL_CEIL = 1.50   # Variant B: max equity weight (leveraged)
PROPORTIONAL_ZSCORE_RANGE = (-1.0, 2.0)  # z-score range to map allocation

# Bond proxy for 60/40: use AGG (or BND). We'll use total return proxy.
BOND_ETF = 'AGG'

N_PERMUTATIONS = 200

print(f"Correlation Timing Strategy v1")
print(f"{'='*60}")
print(f"Start: {START_DATE}, End: {END_DATE}")
print(f"Correlation window: {CORR_WINDOW}d")
print()

# ---------------------------------------------------------------------------
# PHASE 0: DATA DOWNLOAD
# ---------------------------------------------------------------------------
print("PHASE 0: Downloading data...")

def download_data():
    """Download daily adjusted close prices for all tickers."""
    tickers = ALL_TICKERS + [BOND_ETF]
    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data

    # Drop any ticker that has no data
    prices = prices.dropna(how='all', axis=1)

    print(f"  Downloaded {len(prices)} days of data for {len(prices.columns)} tickers")
    print(f"  Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")

    # XLC started in 2018, XLRE in 2015 — note missing early data
    for t in tickers:
        if t in prices.columns:
            first_valid = prices[t].first_valid_index()
            if first_valid is not None and first_valid > pd.Timestamp(START_DATE) + pd.Timedelta(days=30):
                print(f"  NOTE: {t} data starts {first_valid.strftime('%Y-%m-%d')}")
        else:
            print(f"  WARNING: {t} not found in download")

    return prices

prices = download_data()

# Compute daily returns
returns = prices.pct_change().dropna()

# ---------------------------------------------------------------------------
# PHASE 1: DEEP OBSERVATION
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("PHASE 1: DEEP OBSERVATION — Sector Correlation Index")
print(f"{'='*60}\n")

def compute_sector_correlation_index(returns_df, window=CORR_WINDOW):
    """
    Compute rolling average pairwise correlation among sector ETFs.
    Returns a Series of the average off-diagonal correlation.
    """
    available_sectors = [s for s in SECTOR_ETFS if s in returns_df.columns]
    sector_returns = returns_df[available_sectors].copy()

    # Need at least 5 sectors for meaningful correlation
    # For each date, compute rolling correlation matrix and take mean off-diagonal
    n = len(available_sectors)
    n_pairs = n * (n - 1) // 2

    # More efficient: compute rolling pairwise correlations
    corr_index = pd.Series(index=returns_df.index, dtype=float)

    for i in range(window, len(sector_returns)):
        window_data = sector_returns.iloc[i-window:i]
        # Drop sectors with all NaN in this window
        valid_cols = window_data.dropna(axis=1, how='all').columns
        if len(valid_cols) < 5:
            continue
        corr_matrix = window_data[valid_cols].corr()
        # Average off-diagonal
        mask = np.ones(corr_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr = corr_matrix.values[mask].mean()
        corr_index.iloc[i] = avg_corr

    return corr_index.dropna()

print("Computing rolling sector correlation index...")
corr_index = compute_sector_correlation_index(returns)
print(f"  Correlation index computed: {len(corr_index)} observations")
print(f"  Date range: {corr_index.index[0].strftime('%Y-%m-%d')} to {corr_index.index[-1].strftime('%Y-%m-%d')}")

# Basic statistics
print(f"\n--- Correlation Index Distribution ---")
print(f"  Mean:   {corr_index.mean():.4f}")
print(f"  Median: {corr_index.median():.4f}")
print(f"  Std:    {corr_index.std():.4f}")
print(f"  Min:    {corr_index.min():.4f} ({corr_index.idxmin().strftime('%Y-%m-%d')})")
print(f"  Max:    {corr_index.max():.4f} ({corr_index.idxmax().strftime('%Y-%m-%d')})")
print(f"  Skew:   {corr_index.skew():.4f}")
print(f"  Kurt:   {corr_index.kurtosis():.4f}")

# Percentiles
pcts = [5, 10, 20, 25, 50, 75, 80, 90, 95]
print(f"\n--- Percentiles ---")
for p in pcts:
    print(f"  P{p:02d}: {corr_index.quantile(p/100):.4f}")

p80 = corr_index.quantile(0.80)
p90 = corr_index.quantile(0.90)
p95 = corr_index.quantile(0.95)

# How often does correlation reach extreme levels?
print(f"\n--- Extreme Level Frequency ---")
for threshold_pct in [80, 90, 95]:
    threshold_val = corr_index.quantile(threshold_pct / 100)
    n_above = (corr_index > threshold_val).sum()
    pct_time = n_above / len(corr_index) * 100
    print(f"  > P{threshold_pct} ({threshold_val:.4f}): {n_above} days ({pct_time:.1f}% of time)")

# Quintile analysis
print(f"\n--- Forward Return Analysis by Correlation Quintile ---")
spy_returns = returns[BENCHMARK].reindex(corr_index.index)

# Align SPY prices for forward return calculation
spy_prices = prices[BENCHMARK].reindex(corr_index.index)

def compute_forward_returns(price_series, periods):
    """Compute forward returns for multiple horizons."""
    fwd = {}
    for p in periods:
        fwd[f'fwd_{p}d'] = price_series.pct_change(p).shift(-p)
    return pd.DataFrame(fwd, index=price_series.index)

fwd_periods = [21, 63, 126]  # 1m, 3m, 6m
fwd_returns = compute_forward_returns(spy_prices, fwd_periods)

# Merge correlation index with forward returns
analysis_df = pd.DataFrame({
    'corr_index': corr_index,
}).join(fwd_returns).dropna()

# Quintile labels
analysis_df['quintile'] = pd.qcut(analysis_df['corr_index'], 5, labels=['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)'])

print(f"\n  {'Quintile':<12} {'Corr Range':<20} {'Fwd 1M':>8} {'Fwd 3M':>8} {'Fwd 6M':>8} {'WR 3M':>8} {'N':>6}")
print(f"  {'-'*70}")

quintile_stats = {}
for q in ['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']:
    subset = analysis_df[analysis_df['quintile'] == q]
    corr_lo = subset['corr_index'].min()
    corr_hi = subset['corr_index'].max()
    fwd_1m = subset['fwd_21d'].mean() * 100
    fwd_3m = subset['fwd_63d'].mean() * 100
    fwd_6m = subset['fwd_126d'].mean() * 100
    wr_3m = (subset['fwd_63d'] > 0).mean() * 100
    n = len(subset)
    print(f"  {q:<12} [{corr_lo:.3f}, {corr_hi:.3f}]  {fwd_1m:>7.2f}% {fwd_3m:>7.2f}% {fwd_6m:>7.2f}% {wr_3m:>7.1f}% {n:>6}")
    quintile_stats[q] = {
        'corr_range': [float(corr_lo), float(corr_hi)],
        'fwd_1m_mean': float(fwd_1m),
        'fwd_3m_mean': float(fwd_3m),
        'fwd_6m_mean': float(fwd_6m),
        'wr_3m': float(wr_3m),
        'n': int(n),
    }

# Monotonicity check
q5_3m = quintile_stats['Q5 (High)']['fwd_3m_mean']
q1_3m = quintile_stats['Q1 (Low)']['fwd_3m_mean']
print(f"\n  Spread Q5-Q1 (3M): {q5_3m - q1_3m:.2f}%")
print(f"  Monotonic: {'YES' if q5_3m > q1_3m else 'NO'} (high correlation -> higher forward returns)")

# Time between extreme readings
print(f"\n--- Time Between Extreme Readings (> P90) ---")
extreme_dates = corr_index[corr_index > p90].index
if len(extreme_dates) > 1:
    # Find "episodes" — cluster consecutive extreme days
    gaps = pd.Series(extreme_dates).diff().dt.days
    episode_starts = extreme_dates[gaps.fillna(999).values > 20]  # 20-day gap = new episode
    if len(episode_starts) > 1:
        inter_episode_gaps = pd.Series(episode_starts).diff().dt.days.dropna()
        print(f"  Number of P90+ episodes: {len(episode_starts)}")
        print(f"  Mean gap between episodes: {inter_episode_gaps.mean():.0f} days")
        print(f"  Median gap: {inter_episode_gaps.median():.0f} days")
        print(f"  Min gap: {inter_episode_gaps.min():.0f} days")
        print(f"  Max gap: {inter_episode_gaps.max():.0f} days")

# Plot 1: Correlation Index time series
fig, axes = plt.subplots(3, 1, figsize=(14, 12))

ax = axes[0]
ax.plot(corr_index.index, corr_index.values, linewidth=0.8, color='steelblue')
ax.axhline(p80, color='orange', linestyle='--', alpha=0.7, label=f'P80 = {p80:.3f}')
ax.axhline(p90, color='red', linestyle='--', alpha=0.7, label=f'P90 = {p90:.3f}')
ax.axhline(corr_index.median(), color='gray', linestyle=':', alpha=0.5, label=f'Median = {corr_index.median():.3f}')
ax.set_title('Sector Average Correlation Index (63d Rolling)', fontsize=14)
ax.set_ylabel('Average Pairwise Correlation')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)

# Plot 2: Distribution
ax = axes[1]
ax.hist(corr_index.values, bins=60, density=True, alpha=0.7, color='steelblue', edgecolor='white')
ax.axvline(p80, color='orange', linestyle='--', label=f'P80')
ax.axvline(p90, color='red', linestyle='--', label=f'P90')
ax.set_title('Distribution of Sector Correlation Index', fontsize=14)
ax.set_xlabel('Average Pairwise Correlation')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)

# Plot 3: Forward returns by quintile
ax = axes[2]
quintile_labels = ['Q1 (Low)', 'Q2', 'Q3', 'Q4', 'Q5 (High)']
fwd_1m_vals = [quintile_stats[q]['fwd_1m_mean'] for q in quintile_labels]
fwd_3m_vals = [quintile_stats[q]['fwd_3m_mean'] for q in quintile_labels]
fwd_6m_vals = [quintile_stats[q]['fwd_6m_mean'] for q in quintile_labels]

x = np.arange(len(quintile_labels))
width = 0.25
ax.bar(x - width, fwd_1m_vals, width, label='Fwd 1M', color='skyblue')
ax.bar(x, fwd_3m_vals, width, label='Fwd 3M', color='steelblue')
ax.bar(x + width, fwd_6m_vals, width, label='Fwd 6M', color='navy')
ax.set_title('Mean Forward SPY Returns by Correlation Quintile', fontsize=14)
ax.set_ylabel('Return (%)')
ax.set_xticks(x)
ax.set_xticklabels(quintile_labels)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')
ax.axhline(0, color='black', linewidth=0.5)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase1_observation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved phase1_observation.png")

# ---------------------------------------------------------------------------
# PHASE 2: STRATEGY BACKTESTS
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("PHASE 2: STRATEGY BACKTESTS")
print(f"{'='*60}\n")

# Prepare monthly signals and returns
# Signal: month-end correlation index
# Execution: next trading day open (approximate with close for daily data)

# Resample to month-end
monthly_corr = corr_index.resample('ME').last()
monthly_spy = prices[BENCHMARK].resample('ME').last()
monthly_bond = prices[BOND_ETF].resample('ME').last() if BOND_ETF in prices.columns else None

# Monthly returns (for next month — shift signal by 1 month for T+1 execution)
spy_monthly_ret = monthly_spy.pct_change()
bond_monthly_ret = monthly_bond.pct_change() if monthly_bond is not None else pd.Series(0.003/12, index=monthly_spy.index)

# Compute z-score of correlation index
corr_mean = monthly_corr.expanding(min_periods=12).mean()
corr_std = monthly_corr.expanding(min_periods=12).std()
corr_zscore = (monthly_corr - corr_mean) / corr_std

# Compute expanding percentile rank
corr_pctrank = monthly_corr.expanding(min_periods=12).rank(pct=True) * 100

# Align: signal at month t, trade in month t+1
# Use .shift(1) on the signal so we're using PRIOR month's signal for THIS month's return
signal_pctrank = corr_pctrank.shift(1)
signal_zscore = corr_zscore.shift(1)
signal_corr = monthly_corr.shift(1)

# Create aligned dataframe
bt_df = pd.DataFrame({
    'spy_ret': spy_monthly_ret,
    'bond_ret': bond_monthly_ret,
    'signal_pctrank': signal_pctrank,
    'signal_zscore': signal_zscore,
    'signal_corr': signal_corr,
}).dropna()

print(f"Backtest period: {bt_df.index[0].strftime('%Y-%m-%d')} to {bt_df.index[-1].strftime('%Y-%m-%d')}")
print(f"Number of monthly observations: {len(bt_df)}")

# Sector monthly returns for Variant C
sector_monthly_rets = {}
for s in SECTOR_ETFS:
    if s in prices.columns:
        sector_monthly_rets[s] = prices[s].resample('ME').last().pct_change()

sector_monthly_df = pd.DataFrame(sector_monthly_rets)

# ---------------------------------------------------------------------------
# Strategy implementations
# ---------------------------------------------------------------------------

def compute_metrics(returns_series, name="Strategy"):
    """Compute standard performance metrics for a monthly return series."""
    r = returns_series.dropna()
    if len(r) < 12:
        return {}

    total_ret = (1 + r).prod() - 1
    n_years = len(r) / 12
    cagr = (1 + total_ret) ** (1 / n_years) - 1 if n_years > 0 else 0

    ann_ret = r.mean() * 12
    ann_vol = r.std() * np.sqrt(12)
    sharpe = (ann_ret - RF_ANNUAL) / ann_vol if ann_vol > 0 else 0

    downside = r[r < 0].std() * np.sqrt(12) if (r < 0).any() else 1e-6
    sortino = (ann_ret - RF_ANNUAL) / downside

    # Max drawdown from monthly equity curve
    equity = (1 + r).cumprod()
    rolling_max = equity.cummax()
    drawdown = (equity - rolling_max) / rolling_max
    max_dd = drawdown.min()

    # Win rate (monthly)
    wr = (r > 0).mean()

    # Profit factor
    gains = r[r > 0].sum()
    losses = abs(r[r < 0].sum())
    pf = gains / losses if losses > 0 else float('inf')

    # Calmar
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    return {
        'name': name,
        'total_return': float(total_ret),
        'cagr': float(cagr),
        'ann_vol': float(ann_vol),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'max_dd': float(max_dd),
        'calmar': float(calmar),
        'win_rate': float(wr),
        'profit_factor': float(pf),
        'n_months': int(len(r)),
    }


def variant_a_threshold(bt_df, threshold_pct=THRESHOLD_PCT):
    """
    Variant A: Simple threshold.
    When correlation > expanding Pth percentile → 100% SPY (stress = buy signal).
    Otherwise → 60% SPY / 40% Bond.
    """
    stress_signal = bt_df['signal_pctrank'] > threshold_pct

    # Allocation
    spy_weight = np.where(stress_signal, 1.0, 0.60)
    bond_weight = np.where(stress_signal, 0.0, 0.40)

    strategy_ret = spy_weight * bt_df['spy_ret'] + bond_weight * bt_df['bond_ret']
    strategy_ret.name = 'Variant A: Threshold'

    n_stress = stress_signal.sum()
    print(f"\n  Variant A: Threshold (P{threshold_pct})")
    print(f"    Stress months: {n_stress} / {len(bt_df)} ({n_stress/len(bt_df)*100:.1f}%)")

    return strategy_ret, stress_signal


def variant_b_proportional(bt_df):
    """
    Variant B: Proportional allocation.
    Scale equity weight from 60% to 150% based on correlation z-score.
    Higher z-score (more stress) → more equity exposure.
    """
    z = bt_df['signal_zscore'].clip(PROPORTIONAL_ZSCORE_RANGE[0], PROPORTIONAL_ZSCORE_RANGE[1])

    # Linear mapping: z=-1 → 60%, z=2 → 150%
    z_range = PROPORTIONAL_ZSCORE_RANGE[1] - PROPORTIONAL_ZSCORE_RANGE[0]
    alloc_range = PROPORTIONAL_CEIL - PROPORTIONAL_FLOOR
    spy_weight = PROPORTIONAL_FLOOR + (z - PROPORTIONAL_ZSCORE_RANGE[0]) / z_range * alloc_range

    # Remaining goes to bonds (can be negative if leveraged — that's fine, it means borrowing at RF)
    bond_weight = 1.0 - spy_weight
    # For leverage > 100%, we borrow at RF (simplification)
    # Strategy return: spy_weight * spy_ret + bond_weight * bond_ret
    # If spy_weight > 1, bond_weight < 0 → we're borrowing bonds / shorting bonds

    strategy_ret = spy_weight * bt_df['spy_ret'] + bond_weight * bt_df['bond_ret']
    strategy_ret.name = 'Variant B: Proportional'

    print(f"\n  Variant B: Proportional (z-score mapped to {PROPORTIONAL_FLOOR:.0%}-{PROPORTIONAL_CEIL:.0%})")
    print(f"    Mean equity weight: {spy_weight.mean():.1%}")
    print(f"    Min/Max equity weight: {spy_weight.min():.1%} / {spy_weight.max():.1%}")

    return strategy_ret, spy_weight


def variant_c_sector_rotation(bt_df, sector_monthly_df):
    """
    Variant C: Sector rotation on correlation spikes.
    When correlation > P80 → buy the 3 most beaten-down sectors (worst trailing 3M return).
    Otherwise → equal-weight SPY.
    """
    stress_signal = bt_df['signal_pctrank'] > THRESHOLD_PCT

    # Trailing 3-month sector returns (at signal time = shifted by 1 month)
    trailing_3m = sector_monthly_df.rolling(3).apply(lambda x: (1+x).prod()-1, raw=True)
    trailing_3m_shifted = trailing_3m.shift(1)  # available at decision time

    strategy_ret = pd.Series(index=bt_df.index, dtype=float)

    for date in bt_df.index:
        if date not in sector_monthly_df.index:
            strategy_ret[date] = bt_df.loc[date, 'spy_ret']
            continue

        if stress_signal.get(date, False):
            # Pick bottom 3 sectors by trailing 3M return
            if date in trailing_3m_shifted.index:
                trailing = trailing_3m_shifted.loc[date].dropna()
                if len(trailing) >= 3:
                    bottom_3 = trailing.nsmallest(3).index.tolist()
                    # Equal weight the bottom 3 sectors
                    month_rets = sector_monthly_df.loc[date, bottom_3] if date in sector_monthly_df.index else pd.Series()
                    if len(month_rets) > 0:
                        strategy_ret[date] = month_rets.mean()
                    else:
                        strategy_ret[date] = bt_df.loc[date, 'spy_ret']
                else:
                    strategy_ret[date] = bt_df.loc[date, 'spy_ret']
            else:
                strategy_ret[date] = bt_df.loc[date, 'spy_ret']
        else:
            # 60/40
            strategy_ret[date] = 0.60 * bt_df.loc[date, 'spy_ret'] + 0.40 * bt_df.loc[date, 'bond_ret']

    strategy_ret.name = 'Variant C: Sector Rotation'
    n_stress = stress_signal.sum()
    print(f"\n  Variant C: Sector Rotation (buy beaten-down sectors on stress)")
    print(f"    Stress months: {n_stress} / {len(bt_df)} ({n_stress/len(bt_df)*100:.1f}%)")

    return strategy_ret, stress_signal


# Run all variants
print("Running strategy variants...")

# Benchmark: Buy & Hold SPY
bnh_ret = bt_df['spy_ret'].copy()
bnh_ret.name = 'Buy & Hold SPY'
bnh_metrics = compute_metrics(bnh_ret, 'Buy & Hold SPY')

# Benchmark: 60/40
sixtyforty_ret = 0.60 * bt_df['spy_ret'] + 0.40 * bt_df['bond_ret']
sixtyforty_ret.name = '60/40'
sixtyforty_metrics = compute_metrics(sixtyforty_ret, '60/40')

# Variants
va_ret, va_signal = variant_a_threshold(bt_df)
va_metrics = compute_metrics(va_ret, 'Variant A: Threshold')

vb_ret, vb_weights = variant_b_proportional(bt_df)
vb_metrics = compute_metrics(vb_ret, 'Variant B: Proportional')

vc_ret, vc_signal = variant_c_sector_rotation(bt_df, sector_monthly_df)
vc_metrics = compute_metrics(vc_ret, 'Variant C: Sector Rotation')

# Print comparison
print(f"\n{'='*60}")
print("STRATEGY COMPARISON")
print(f"{'='*60}")
all_metrics = [bnh_metrics, sixtyforty_metrics, va_metrics, vb_metrics, vc_metrics]
header = f"  {'Strategy':<28} {'CAGR':>7} {'Sharpe':>7} {'Sortino':>8} {'MaxDD':>8} {'WR':>6} {'PF':>6}"
print(header)
print(f"  {'-'*75}")
for m in all_metrics:
    print(f"  {m['name']:<28} {m['cagr']:>6.1%} {m['sharpe']:>7.2f} {m['sortino']:>8.2f} {m['max_dd']:>7.1%} {m['win_rate']:>5.1%} {m['profit_factor']:>6.2f}")

# Equity curve plot
fig, ax = plt.subplots(figsize=(14, 7))
for ret, label, color in [
    (bnh_ret, 'Buy & Hold SPY', 'gray'),
    (sixtyforty_ret, '60/40', 'lightblue'),
    (va_ret, 'Variant A: Threshold', 'green'),
    (vb_ret, 'Variant B: Proportional', 'orange'),
    (vc_ret, 'Variant C: Sector Rotation', 'purple'),
]:
    equity = (1 + ret).cumprod()
    ax.plot(equity.index, equity.values, label=f"{label}", linewidth=1.5, color=color)

ax.set_title('Correlation Timing Strategy — Equity Curves', fontsize=14)
ax.set_ylabel('Growth of $1')
ax.legend(loc='upper left')
ax.grid(True, alpha=0.3)
ax.set_yscale('log')
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase2_equity_curves.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\n  Saved phase2_equity_curves.png")

# Drawdown comparison plot
fig, axes = plt.subplots(2, 1, figsize=(14, 8))
for ret, label, color in [
    (bnh_ret, 'SPY', 'gray'),
    (va_ret, 'Variant A', 'green'),
    (vb_ret, 'Variant B', 'orange'),
    (vc_ret, 'Variant C', 'purple'),
]:
    equity = (1 + ret).cumprod()
    dd = (equity - equity.cummax()) / equity.cummax()
    axes[0].plot(dd.index, dd.values, label=label, linewidth=1, color=color, alpha=0.8)

axes[0].set_title('Drawdown Comparison', fontsize=14)
axes[0].set_ylabel('Drawdown')
axes[0].legend(loc='lower left')
axes[0].grid(True, alpha=0.3)

# Correlation index with stress periods highlighted
axes[1].plot(corr_index.index, corr_index.values, linewidth=0.8, color='steelblue')
axes[1].axhline(p80, color='orange', linestyle='--', alpha=0.7, label=f'P80')
axes[1].fill_between(corr_index.index, p80, corr_index.values,
                      where=corr_index.values > p80, alpha=0.3, color='red', label='Stress Zone')
axes[1].set_title('Sector Correlation Index with Stress Zones', fontsize=14)
axes[1].set_ylabel('Avg Correlation')
axes[1].legend(loc='upper left')
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase2_drawdown_and_signal.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved phase2_drawdown_and_signal.png")

# ---------------------------------------------------------------------------
# PHASE 3: VALIDATION
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("PHASE 3: VALIDATION")
print(f"{'='*60}\n")

# 3A. Permutation Test (200 shuffles)
print("3A. Permutation Test (200 shuffles)...")

def permutation_test(signal_series, returns_df, n_perms=N_PERMUTATIONS, strategy_func=None):
    """
    Shuffle signal dates, re-run strategy, build null Sharpe distribution.
    """
    # Actual Sharpe
    actual_ret = strategy_func(returns_df, signal_series)
    actual_sharpe = compute_metrics(actual_ret, 'actual')['sharpe']

    null_sharpes = []
    rng = np.random.RandomState(42)

    for i in range(n_perms):
        # Shuffle signal values
        shuffled_signal = signal_series.copy()
        shuffled_signal.values[:] = rng.permutation(signal_series.values)
        perm_ret = strategy_func(returns_df, shuffled_signal)
        perm_sharpe = compute_metrics(perm_ret, f'perm_{i}')
        if perm_sharpe:
            null_sharpes.append(perm_sharpe['sharpe'])

    null_sharpes = np.array(null_sharpes)
    p_value = (null_sharpes >= actual_sharpe).mean()

    return actual_sharpe, null_sharpes, p_value


def variant_a_from_signal(returns_df, signal_pctrank):
    """Reconstruct Variant A returns from signal."""
    stress = signal_pctrank > THRESHOLD_PCT
    spy_w = np.where(stress, 1.0, 0.60)
    bond_w = np.where(stress, 0.0, 0.40)
    return spy_w * returns_df['spy_ret'] + bond_w * returns_df['bond_ret']

actual_sharpe_a, null_sharpes_a, p_value_a = permutation_test(
    bt_df['signal_pctrank'], bt_df, N_PERMUTATIONS, variant_a_from_signal
)
print(f"  Variant A: Actual Sharpe = {actual_sharpe_a:.3f}, p-value = {p_value_a:.3f}")
print(f"    Null Sharpe: mean={null_sharpes_a.mean():.3f}, std={null_sharpes_a.std():.3f}")
print(f"    Percentile of actual in null: {(null_sharpes_a < actual_sharpe_a).mean()*100:.1f}th")

# Permutation plot
fig, ax = plt.subplots(figsize=(10, 5))
ax.hist(null_sharpes_a, bins=30, density=True, alpha=0.7, color='lightgray', edgecolor='white', label='Null Distribution')
ax.axvline(actual_sharpe_a, color='red', linewidth=2, label=f'Actual Sharpe = {actual_sharpe_a:.3f}')
ax.set_title(f'Variant A Permutation Test (N={N_PERMUTATIONS}, p={p_value_a:.3f})', fontsize=14)
ax.set_xlabel('Sharpe Ratio')
ax.set_ylabel('Density')
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(OUTPUT_DIR / 'phase3_permutation_test.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"  Saved phase3_permutation_test.png")

# 3B. Regime Test
print(f"\n3B. Regime Test...")

# Define regimes: VIX > 25 = "Stress", VIX <= 25 = "Calm"
# Also use simple: SPY trailing 12M return > 0 = Bull, < 0 = Bear
vix = yf.download('^VIX', start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix_close = vix['Close'].iloc[:, 0] if vix['Close'].ndim > 1 else vix['Close']
else:
    vix_close = vix['Close']

vix_monthly = vix_close.resample('ME').last()

# SPY trailing 12M return as bull/bear proxy
spy_12m = monthly_spy.pct_change(12)

regime_df = pd.DataFrame({
    'vix': vix_monthly,
    'spy_12m': spy_12m,
}).reindex(bt_df.index)

# VIX regime
regime_df['vix_regime'] = np.where(regime_df['vix'] > 25, 'High VIX (>25)', 'Low VIX (<=25)')
# Trend regime
regime_df['trend_regime'] = np.where(regime_df['spy_12m'] > 0, 'Bull (12M>0)', 'Bear (12M<0)')

print(f"\n  --- Regime Analysis: Variant A ---")
for regime_col, regime_name in [('vix_regime', 'VIX Regime'), ('trend_regime', 'Trend Regime')]:
    print(f"\n  {regime_name}:")
    regime_sharpes = {}
    for regime in regime_df[regime_col].dropna().unique():
        mask = regime_df[regime_col] == regime
        r = va_ret[mask].dropna()
        if len(r) >= 6:
            m = compute_metrics(r, regime)
            regime_sharpes[regime] = m['sharpe']
            print(f"    {regime:<20}: Sharpe={m['sharpe']:.2f}, CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}, N={m['n_months']}")
        else:
            print(f"    {regime:<20}: Insufficient data (N={len(r)})")

    # Regime gap check (HC #428 R1)
    if len(regime_sharpes) == 2:
        vals = list(regime_sharpes.values())
        gap = abs(vals[0] - vals[1]) / max(abs(vals[0]), abs(vals[1])) if max(abs(vals[0]), abs(vals[1])) > 0 else 0
        verdict = "PASS" if gap < 0.50 else "FAIL"
        print(f"    Regime gap: {gap:.2f} ({verdict} — threshold 0.50)")

# Same for Variant B
print(f"\n  --- Regime Analysis: Variant B ---")
for regime_col, regime_name in [('vix_regime', 'VIX Regime'), ('trend_regime', 'Trend Regime')]:
    print(f"\n  {regime_name}:")
    regime_sharpes = {}
    for regime in regime_df[regime_col].dropna().unique():
        mask = regime_df[regime_col] == regime
        r = vb_ret[mask].dropna()
        if len(r) >= 6:
            m = compute_metrics(r, regime)
            regime_sharpes[regime] = m['sharpe']
            print(f"    {regime:<20}: Sharpe={m['sharpe']:.2f}, CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}, N={m['n_months']}")

    if len(regime_sharpes) == 2:
        vals = list(regime_sharpes.values())
        gap = abs(vals[0] - vals[1]) / max(abs(vals[0]), abs(vals[1])) if max(abs(vals[0]), abs(vals[1])) > 0 else 0
        verdict = "PASS" if gap < 0.50 else "FAIL"
        print(f"    Regime gap: {gap:.2f} ({verdict} — threshold 0.50)")

# Variant C regime analysis
print(f"\n  --- Regime Analysis: Variant C ---")
for regime_col, regime_name in [('vix_regime', 'VIX Regime'), ('trend_regime', 'Trend Regime')]:
    print(f"\n  {regime_name}:")
    regime_sharpes = {}
    for regime in regime_df[regime_col].dropna().unique():
        mask = regime_df[regime_col] == regime
        r = vc_ret[mask].dropna()
        if len(r) >= 6:
            m = compute_metrics(r, regime)
            regime_sharpes[regime] = m['sharpe']
            print(f"    {regime:<20}: Sharpe={m['sharpe']:.2f}, CAGR={m['cagr']:.1%}, MaxDD={m['max_dd']:.1%}, N={m['n_months']}")

    if len(regime_sharpes) == 2:
        vals = list(regime_sharpes.values())
        gap = abs(vals[0] - vals[1]) / max(abs(vals[0]), abs(vals[1])) if max(abs(vals[0]), abs(vals[1])) > 0 else 0
        verdict = "PASS" if gap < 0.50 else "FAIL"
        print(f"    Regime gap: {gap:.2f} ({verdict} — threshold 0.50)")

# 3C. Sub-period stability
print(f"\n3C. Sub-period Stability...")
n_periods = 3
period_len = len(bt_df) // n_periods

print(f"\n  --- Sub-period Analysis ---")
for variant_name, variant_ret in [('Variant A', va_ret), ('Variant B', vb_ret), ('Variant C', vc_ret), ('SPY B&H', bnh_ret)]:
    print(f"\n  {variant_name}:")
    period_sharpes = []
    for p in range(n_periods):
        start_idx = p * period_len
        end_idx = (p + 1) * period_len if p < n_periods - 1 else len(bt_df)
        period_ret = variant_ret.iloc[start_idx:end_idx]
        m = compute_metrics(period_ret, f'Period {p+1}')
        if m:
            period_sharpes.append(m['sharpe'])
            start_date = period_ret.index[0].strftime('%Y-%m')
            end_date = period_ret.index[-1].strftime('%Y-%m')
            print(f"    Period {p+1} ({start_date} to {end_date}): Sharpe={m['sharpe']:.2f}, CAGR={m['cagr']:.1%}")

    if len(period_sharpes) >= 2:
        all_positive = all(s > 0 for s in period_sharpes)
        print(f"    All periods positive Sharpe: {'YES' if all_positive else 'NO'}")

# 3D. Lag sensitivity
print(f"\n3D. Lag Sensitivity...")
print(f"\n  Testing T+0 (same month, look-ahead) vs T+1 (actual tradeable) vs T+2:")

for lag, lag_name in [(0, 'T+0 (look-ahead)'), (1, 'T+1 (tradeable)'), (2, 'T+2 (delayed)')]:
    lagged_signal = corr_pctrank.shift(lag)
    temp_df = pd.DataFrame({
        'spy_ret': spy_monthly_ret,
        'bond_ret': bond_monthly_ret,
        'signal_pctrank': lagged_signal,
    }).dropna()

    stress = temp_df['signal_pctrank'] > THRESHOLD_PCT
    spy_w = np.where(stress, 1.0, 0.60)
    bond_w = np.where(stress, 0.0, 0.40)
    lag_ret = spy_w * temp_df['spy_ret'] + bond_w * temp_df['bond_ret']

    m = compute_metrics(lag_ret, lag_name)
    if m:
        print(f"  {lag_name:<22}: Sharpe={m['sharpe']:.2f}, CAGR={m['cagr']:.1%}")

# ---------------------------------------------------------------------------
# PHASE 4: FINAL VERDICT & OUTPUT
# ---------------------------------------------------------------------------
print(f"\n{'='*60}")
print("PHASE 4: FINAL VERDICT")
print(f"{'='*60}\n")

verdicts = {}

for variant_name, variant_metrics, variant_ret in [
    ('Variant A: Threshold', va_metrics, va_ret),
    ('Variant B: Proportional', vb_metrics, vb_ret),
    ('Variant C: Sector Rotation', vc_metrics, vc_ret),
]:
    passes = []
    fails = []

    # Check 1: Does it beat buy-and-hold Sharpe?
    if variant_metrics['sharpe'] > bnh_metrics['sharpe']:
        passes.append(f"Sharpe > SPY B&H ({variant_metrics['sharpe']:.2f} > {bnh_metrics['sharpe']:.2f})")
    else:
        fails.append(f"Sharpe <= SPY B&H ({variant_metrics['sharpe']:.2f} <= {bnh_metrics['sharpe']:.2f})")

    # Check 2: Lower max drawdown?
    if variant_metrics['max_dd'] > bnh_metrics['max_dd']:  # less negative = better
        passes.append(f"Better MaxDD ({variant_metrics['max_dd']:.1%} vs {bnh_metrics['max_dd']:.1%})")
    else:
        fails.append(f"Worse MaxDD ({variant_metrics['max_dd']:.1%} vs {bnh_metrics['max_dd']:.1%})")

    # Check 3: Positive Sharpe in all sub-periods?
    period_sharpes = []
    for p in range(n_periods):
        start_idx = p * period_len
        end_idx = (p + 1) * period_len if p < n_periods - 1 else len(bt_df)
        m = compute_metrics(variant_ret.iloc[start_idx:end_idx])
        if m:
            period_sharpes.append(m['sharpe'])

    if all(s > 0 for s in period_sharpes):
        passes.append("Positive Sharpe all sub-periods")
    else:
        fails.append("Negative Sharpe in some sub-periods")

    verdict = "PASS" if len(fails) == 0 else ("CONDITIONAL PASS" if len(fails) <= 1 else "FAIL")
    verdicts[variant_name] = {
        'verdict': verdict,
        'passes': passes,
        'fails': fails,
        'metrics': variant_metrics,
    }

    print(f"  {variant_name}: {verdict}")
    for p in passes:
        print(f"    [+] {p}")
    for f in fails:
        print(f"    [-] {f}")
    print()

# Permutation p-value for best variant
print(f"  Permutation test p-value (Variant A): {p_value_a:.3f}")
if p_value_a < 0.05:
    print(f"    Signal is statistically significant (p < 0.05)")
else:
    print(f"    Signal is NOT statistically significant (p >= 0.05)")

# ---------------------------------------------------------------------------
# Save summary
# ---------------------------------------------------------------------------
summary = {
    'generated': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
    'observation': {
        'signal': 'sector_average_correlation',
        'correlation_window': CORR_WINDOW,
        'data_range': f"{corr_index.index[0].strftime('%Y-%m-%d')} to {corr_index.index[-1].strftime('%Y-%m-%d')}",
        'n_observations': int(len(corr_index)),
        'distribution': {
            'mean': float(corr_index.mean()),
            'median': float(corr_index.median()),
            'std': float(corr_index.std()),
            'p80': float(p80),
            'p90': float(p90),
            'p95': float(p95),
        },
        'quintile_analysis': quintile_stats,
    },
    'backtest': {
        'period': f"{bt_df.index[0].strftime('%Y-%m-%d')} to {bt_df.index[-1].strftime('%Y-%m-%d')}",
        'n_months': int(len(bt_df)),
        'benchmarks': {
            'spy_bnh': bnh_metrics,
            'sixty_forty': sixtyforty_metrics,
        },
        'variants': {
            'A_threshold': va_metrics,
            'B_proportional': vb_metrics,
            'C_sector_rotation': vc_metrics,
        },
    },
    'validation': {
        'permutation_test': {
            'variant_a_sharpe': float(actual_sharpe_a),
            'null_mean': float(null_sharpes_a.mean()),
            'null_std': float(null_sharpes_a.std()),
            'p_value': float(p_value_a),
            'significant': bool(p_value_a < 0.05),
        },
    },
    'verdicts': {k: {'verdict': v['verdict'], 'passes': v['passes'], 'fails': v['fails']} for k, v in verdicts.items()},
}

with open(OUTPUT_DIR / 'summary.json', 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nSummary saved to {OUTPUT_DIR / 'summary.json'}")
print(f"\nDone. All output in {OUTPUT_DIR}/")
