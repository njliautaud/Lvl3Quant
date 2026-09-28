#!/usr/bin/env python3
"""
Cross-Asset Leading Indicators for Sector ETF Dip-Buying
=========================================================
Question: Do moves in TLT, GLD, UUP, HYG LEAD sector ETF recoveries
when RSI < 35 dip-buy entries fire?

Hypotheses:
  - TLT falling (flight-to-quality ending) -> better dip-buy returns
  - GLD falling (risk-on resuming) -> better dip-buy returns
  - UUP falling (dollar weakening = risk-on) -> better dip-buy returns
  - HYG rising (credit improving) -> better dip-buy returns

Composite "risk-on resuming" score: 0-4
  sum of (TLT_5d < 0) + (GLD_5d < 0) + (UUP_5d < 0) + (HYG_5d > 0)
"""

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# ============================================================
# PARAMETERS
# ============================================================
SECTOR_ETFS = ['XLK', 'XLP', 'XLC', 'XLY', 'XLF', 'XLI', 'XLV', 'XLE', 'XLU', 'XLB', 'XLRE']
CROSS_ASSETS = ['TLT', 'GLD', 'UUP', 'HYG']
BENCHMARK = 'SPY'
ALL_TICKERS = SECTOR_ETFS + CROSS_ASSETS + [BENCHMARK]

RSI_PERIOD = 14
RSI_THRESHOLD = 35
LOOKBACK_DAYS = 5       # cross-asset momentum lookback
FORWARD_DAYS = 5        # forward return horizon
MIN_GAP_BETWEEN_ENTRIES = 5  # avoid clustering
N_PERMUTATIONS = 1000
REGIME_GAP_THRESHOLD = 0.50  # HC #428

START_DATE = '2019-01-01'
END_DATE = '2026-08-20'

np.random.seed(42)

# ============================================================
# HELPERS
# ============================================================
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def sharpe(returns):
    if len(returns) < 2 or returns.std() == 0:
        return 0.0
    return returns.mean() / returns.std() * np.sqrt(252)

def sortino(returns):
    if len(returns) < 2:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return np.inf if returns.mean() > 0 else 0.0
    return returns.mean() / downside.std() * np.sqrt(252)

def profit_factor(returns):
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss == 0:
        return np.inf if gross_profit > 0 else 0.0
    return gross_profit / gross_loss

def win_rate(returns):
    if len(returns) == 0:
        return 0.0
    return (returns > 0).mean() * 100

def stats_summary(returns, label=""):
    n = len(returns)
    if n == 0:
        return {'label': label, 'N': 0, 'mean_ret': 0, 'sharpe': 0, 'sortino': 0, 'pf': 0, 'wr': 0}
    return {
        'label': label,
        'N': n,
        'mean_ret_bps': returns.mean() * 10000,
        'median_ret_bps': np.median(returns) * 10000,
        'sharpe': sharpe(returns),
        'sortino': sortino(returns),
        'pf': profit_factor(returns),
        'wr': win_rate(returns),
    }

# ============================================================
# DATA DOWNLOAD
# ============================================================
print("=" * 70)
print("CROSS-ASSET LEADING INDICATORS FOR SECTOR ETF DIP-BUYING")
print("=" * 70)
print(f"\nDownloading {len(ALL_TICKERS)} tickers from {START_DATE} to {END_DATE}...")

data = yf.download(ALL_TICKERS, start=START_DATE, end=END_DATE, progress=False)
close = data['Close'] if 'Close' in data.columns.get_level_values(0) else data['Adj Close']

# Handle MultiIndex columns
if isinstance(close.columns, pd.MultiIndex):
    close.columns = close.columns.get_level_values(-1)

close = close.dropna(how='all')
print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")

# Check we have all tickers
missing = [t for t in ALL_TICKERS if t not in close.columns]
if missing:
    print(f"WARNING: Missing tickers: {missing}")

# ============================================================
# COMPUTE RSI AND CROSS-ASSET FEATURES
# ============================================================
rsi_df = pd.DataFrame(index=close.index)
for etf in SECTOR_ETFS:
    if etf in close.columns:
        rsi_df[etf] = compute_rsi(close[etf], RSI_PERIOD)

# Cross-asset 5-day returns
cross_ret = pd.DataFrame(index=close.index)
for ca in CROSS_ASSETS:
    if ca in close.columns:
        cross_ret[f'{ca}_5d'] = close[ca].pct_change(LOOKBACK_DAYS)

# SPY 5-day return for regime classification
spy_5d = close[BENCHMARK].pct_change(LOOKBACK_DAYS)
# Regime: use trailing 20-day SPY return
spy_20d = close[BENCHMARK].pct_change(20)

# Forward 5-day returns for sector ETFs
fwd_ret = pd.DataFrame(index=close.index)
for etf in SECTOR_ETFS:
    if etf in close.columns:
        fwd_ret[etf] = close[etf].pct_change(FORWARD_DAYS).shift(-FORWARD_DAYS)

# ============================================================
# COLLECT ALL RSI<35 ENTRIES
# ============================================================
entries = []
for etf in SECTOR_ETFS:
    if etf not in rsi_df.columns or etf not in fwd_ret.columns:
        continue
    rsi_col = rsi_df[etf]
    fwd_col = fwd_ret[etf]

    last_entry_idx = -999
    for i in range(RSI_PERIOD + LOOKBACK_DAYS, len(close) - FORWARD_DAYS):
        if i - last_entry_idx < MIN_GAP_BETWEEN_ENTRIES:
            continue
        if pd.isna(rsi_col.iloc[i]) or pd.isna(fwd_col.iloc[i]):
            continue
        if rsi_col.iloc[i] < RSI_THRESHOLD:
            last_entry_idx = i
            row = {
                'date': close.index[i],
                'etf': etf,
                'rsi': rsi_col.iloc[i],
                'fwd_5d_ret': fwd_col.iloc[i],
            }
            # Cross-asset features
            for ca in CROSS_ASSETS:
                col = f'{ca}_5d'
                if col in cross_ret.columns:
                    row[col] = cross_ret[col].iloc[i]
            # Regime
            row['spy_20d'] = spy_20d.iloc[i] if not pd.isna(spy_20d.iloc[i]) else 0.0
            entries.append(row)

df = pd.DataFrame(entries)
print(f"\nTotal RSI<{RSI_THRESHOLD} entries found: {len(df)}")
print(f"Unique ETFs with entries: {df['etf'].nunique()}")
print(f"Date range of entries: {df['date'].min().date()} to {df['date'].max().date()}")
print(f"\nEntries per sector:")
print(df['etf'].value_counts().sort_index().to_string())

# ============================================================
# BASELINE: ALL RSI<35 ENTRIES
# ============================================================
baseline_rets = df['fwd_5d_ret'].values
baseline_stats = stats_summary(baseline_rets, "ALL RSI<35 entries")

print(f"\n{'='*70}")
print("BASELINE: ALL RSI<35 ENTRIES (no cross-asset filter)")
print(f"{'='*70}")
print(f"  N = {baseline_stats['N']}")
print(f"  Mean 5d return = {baseline_stats['mean_ret_bps']:.1f} bps")
print(f"  Median 5d return = {baseline_stats['median_ret_bps']:.1f} bps")
print(f"  Sharpe = {baseline_stats['sharpe']:.2f}")
print(f"  Sortino = {baseline_stats['sortino']:.2f}")
print(f"  PF = {baseline_stats['pf']:.2f}")
print(f"  WR = {baseline_stats['wr']:.1f}%")

# ============================================================
# INDIVIDUAL CROSS-ASSET SIGNAL ANALYSIS
# ============================================================
print(f"\n{'='*70}")
print("INDIVIDUAL CROSS-ASSET SIGNALS AT RSI<35 ENTRY")
print(f"{'='*70}")

signal_results = {}
for ca in CROSS_ASSETS:
    col = f'{ca}_5d'
    if col not in df.columns:
        continue
    valid = df.dropna(subset=[col])

    # Define "favorable" direction
    if ca in ['TLT', 'GLD', 'UUP']:
        # Falling = risk-on resuming
        favorable = valid[valid[col] < 0]['fwd_5d_ret'].values
        unfavorable = valid[valid[col] >= 0]['fwd_5d_ret'].values
        direction = "FALLING (< 0)"
    else:  # HYG
        # Rising = credit improving
        favorable = valid[valid[col] > 0]['fwd_5d_ret'].values
        unfavorable = valid[valid[col] <= 0]['fwd_5d_ret'].values
        direction = "RISING (> 0)"

    fav_stats = stats_summary(favorable, f"{ca} {direction}")
    unfav_stats = stats_summary(unfavorable, f"{ca} opposite")

    signal_results[ca] = {
        'favorable': fav_stats,
        'unfavorable': unfav_stats,
        'favorable_rets': favorable,
        'unfavorable_rets': unfavorable,
    }

    print(f"\n--- {ca} ---")
    print(f"  Hypothesis: {ca} {direction} at entry -> better dip-buy returns")
    print(f"  FAVORABLE ({direction}):  N={fav_stats['N']:4d}  Mean={fav_stats['mean_ret_bps']:+7.1f}bps  "
          f"Sharpe={fav_stats['sharpe']:+.2f}  PF={fav_stats['pf']:.2f}  WR={fav_stats['wr']:.1f}%")
    print(f"  UNFAVORABLE (opposite):   N={unfav_stats['N']:4d}  Mean={unfav_stats['mean_ret_bps']:+7.1f}bps  "
          f"Sharpe={unfav_stats['sharpe']:+.2f}  PF={unfav_stats['pf']:.2f}  WR={unfav_stats['wr']:.1f}%")

    # Edge improvement
    if unfav_stats['N'] > 0 and fav_stats['N'] > 0:
        edge_diff = fav_stats['mean_ret_bps'] - unfav_stats['mean_ret_bps']
        print(f"  Edge difference: {edge_diff:+.1f} bps (favorable - unfavorable)")

# ============================================================
# COMPOSITE SCORE
# ============================================================
print(f"\n{'='*70}")
print("COMPOSITE 'RISK-ON RESUMING' SCORE (0-4)")
print("  = (TLT_5d<0) + (GLD_5d<0) + (UUP_5d<0) + (HYG_5d>0)")
print(f"{'='*70}")

df['score'] = 0
if 'TLT_5d' in df.columns:
    df['score'] += (df['TLT_5d'] < 0).astype(int)
if 'GLD_5d' in df.columns:
    df['score'] += (df['GLD_5d'] < 0).astype(int)
if 'UUP_5d' in df.columns:
    df['score'] += (df['UUP_5d'] < 0).astype(int)
if 'HYG_5d' in df.columns:
    df['score'] += (df['HYG_5d'] > 0).astype(int)

# Remove rows where any cross-asset data is missing
valid_mask = True
for ca in CROSS_ASSETS:
    col = f'{ca}_5d'
    if col in df.columns:
        valid_mask &= df[col].notna()
df_valid = df[valid_mask].copy()

# Add regime classification early so filtered inherits it
def classify_regime(spy_20d_val):
    if spy_20d_val > 0.02:
        return 'GREEN'
    elif spy_20d_val < -0.02:
        return 'RED'
    else:
        return 'FLAT'

df_valid['regime'] = df_valid['spy_20d'].apply(classify_regime)

print(f"\nEntries with complete cross-asset data: {len(df_valid)}")
print(f"\nScore distribution:")

for score in range(5):
    subset = df_valid[df_valid['score'] == score]
    rets = subset['fwd_5d_ret'].values
    s = stats_summary(rets, f"Score={score}")
    print(f"  Score {score}: N={s['N']:4d}  Mean={s['mean_ret_bps']:+7.1f}bps  "
          f"Sharpe={s['sharpe']:+.2f}  Sortino={s['sortino']:+.2f}  PF={s['pf']:.2f}  WR={s['wr']:.1f}%")

# ============================================================
# FILTERED: SCORE >= 3 (STRONG RISK-ON)
# ============================================================
print(f"\n{'='*70}")
print("FILTERED ENTRIES: COMPOSITE SCORE >= 3 (STRONG RISK-ON)")
print(f"{'='*70}")

filtered = df_valid[df_valid['score'] >= 3]
filtered_rets = filtered['fwd_5d_ret'].values
filtered_stats = stats_summary(filtered_rets, "Score >= 3")

unfiltered_rets = df_valid['fwd_5d_ret'].values
unfiltered_stats = stats_summary(unfiltered_rets, "All (with data)")

print(f"\n  ALL entries (baseline):   N={unfiltered_stats['N']:4d}  Mean={unfiltered_stats['mean_ret_bps']:+7.1f}bps  "
      f"Sharpe={unfiltered_stats['sharpe']:+.2f}  Sortino={unfiltered_stats['sortino']:+.2f}  "
      f"PF={unfiltered_stats['pf']:.2f}  WR={unfiltered_stats['wr']:.1f}%")
print(f"  FILTERED (score>=3):      N={filtered_stats['N']:4d}  Mean={filtered_stats['mean_ret_bps']:+7.1f}bps  "
      f"Sharpe={filtered_stats['sharpe']:+.2f}  Sortino={filtered_stats['sortino']:+.2f}  "
      f"PF={filtered_stats['pf']:.2f}  WR={filtered_stats['wr']:.1f}%")

# Score == 4 (perfect risk-on)
perfect = df_valid[df_valid['score'] == 4]
perfect_rets = perfect['fwd_5d_ret'].values
perfect_stats = stats_summary(perfect_rets, "Score == 4")
print(f"  PERFECT (score==4):       N={perfect_stats['N']:4d}  Mean={perfect_stats['mean_ret_bps']:+7.1f}bps  "
      f"Sharpe={perfect_stats['sharpe']:+.2f}  Sortino={perfect_stats['sortino']:+.2f}  "
      f"PF={perfect_stats['pf']:.2f}  WR={perfect_stats['wr']:.1f}%")

# ============================================================
# SCORE >= 3: SECTOR BREAKDOWN
# ============================================================
print(f"\n  Sector breakdown (Score >= 3):")
for etf in sorted(SECTOR_ETFS):
    sub = filtered[filtered['etf'] == etf]
    if len(sub) == 0:
        continue
    s = stats_summary(sub['fwd_5d_ret'].values, etf)
    print(f"    {etf}: N={s['N']:3d}  Mean={s['mean_ret_bps']:+7.1f}bps  Sharpe={s['sharpe']:+.2f}  WR={s['wr']:.1f}%")

# ============================================================
# PERMUTATION TEST (Score >= 3 mean return)
# ============================================================
print(f"\n{'='*70}")
print(f"PERMUTATION TEST ({N_PERMUTATIONS} shuffles)")
print(f"{'='*70}")

observed_mean = filtered_rets.mean() if len(filtered_rets) > 0 else 0.0
all_rets = df_valid['fwd_5d_ret'].values
n_filtered = len(filtered_rets)

perm_means = np.zeros(N_PERMUTATIONS)
for i in range(N_PERMUTATIONS):
    idx = np.random.choice(len(all_rets), size=n_filtered, replace=False)
    perm_means[i] = all_rets[idx].mean()

p_value = (perm_means >= observed_mean).mean()
print(f"\n  Observed mean (score>=3): {observed_mean*10000:+.1f} bps")
print(f"  Permutation distribution: mean={perm_means.mean()*10000:+.1f} bps, std={perm_means.std()*10000:.1f} bps")
print(f"  P-value (one-sided, >= observed): {p_value:.4f}")
print(f"  Significant at 5%? {'YES' if p_value < 0.05 else 'NO'}")
print(f"  Significant at 10%? {'YES' if p_value < 0.10 else 'NO'}")

# ============================================================
# REGIME STRATIFICATION (HC #428)
# ============================================================
print(f"\n{'='*70}")
print("REGIME STRATIFICATION (HC #428)")
print(f"{'='*70}")

# Regime already classified on df_valid above
# Re-derive filtered to ensure regime column is present
filtered = df_valid[df_valid['score'] >= 3].copy()
filtered_rets = filtered['fwd_5d_ret'].values

print(f"\n  Regime distribution of RSI<35 entries:")
print(f"  {df_valid['regime'].value_counts().to_dict()}")

# Baseline by regime
print(f"\n  BASELINE (all entries) by regime:")
for regime in ['GREEN', 'FLAT', 'RED']:
    sub = df_valid[df_valid['regime'] == regime]
    if len(sub) == 0:
        continue
    s = stats_summary(sub['fwd_5d_ret'].values, regime)
    print(f"    {regime:5s}: N={s['N']:4d}  Mean={s['mean_ret_bps']:+7.1f}bps  Sharpe={s['sharpe']:+.2f}  WR={s['wr']:.1f}%")

# Score >= 3 by regime
print(f"\n  FILTERED (score>=3) by regime:")
regime_sharpes = {}
for regime in ['GREEN', 'FLAT', 'RED']:
    sub = filtered[filtered['regime'] == regime]
    if len(sub) == 0:
        regime_sharpes[regime] = 0.0
        print(f"    {regime:5s}: N=   0  (no entries)")
        continue
    s = stats_summary(sub['fwd_5d_ret'].values, regime)
    regime_sharpes[regime] = s['sharpe']
    print(f"    {regime:5s}: N={s['N']:4d}  Mean={s['mean_ret_bps']:+7.1f}bps  Sharpe={s['sharpe']:+.2f}  "
          f"Sortino={s['sortino']:+.2f}  PF={s['pf']:.2f}  WR={s['wr']:.1f}%")

# Regime gap check (HC #428)
sharpe_green = regime_sharpes.get('GREEN', 0.0)
sharpe_red = regime_sharpes.get('RED', 0.0)
max_sharpe = max(abs(sharpe_green), abs(sharpe_red))
if max_sharpe > 0:
    regime_gap = abs(sharpe_green - sharpe_red) / max_sharpe
else:
    regime_gap = 0.0

print(f"\n  REGIME GAP CHECK (HC #428):")
print(f"    Sharpe_GREEN = {sharpe_green:+.2f}")
print(f"    Sharpe_RED   = {sharpe_red:+.2f}")
print(f"    |gap| / max  = {regime_gap:.2f}")
print(f"    Threshold    = {REGIME_GAP_THRESHOLD:.2f}")
print(f"    PASS? {'YES (regime-agnostic)' if regime_gap <= REGIME_GAP_THRESHOLD else 'NO (regime-dependent -- REJECT per HC #428)'}")

# ============================================================
# MONOTONICITY CHECK: Does higher score = better returns?
# ============================================================
print(f"\n{'='*70}")
print("MONOTONICITY CHECK: Score vs Forward Return")
print(f"{'='*70}")

score_means = []
for score in range(5):
    sub = df_valid[df_valid['score'] == score]
    if len(sub) > 0:
        score_means.append((score, sub['fwd_5d_ret'].mean() * 10000))
    else:
        score_means.append((score, np.nan))

print("\n  Score | Mean 5d Return (bps)")
print("  ------|---------------------")
for score, mean_bps in score_means:
    if np.isnan(mean_bps):
        print(f"    {score}   |  N/A (no entries)")
    else:
        bar = "+" * max(0, int(mean_bps / 5)) if mean_bps > 0 else "-" * max(0, int(-mean_bps / 5))
        print(f"    {score}   | {mean_bps:+7.1f}  {bar}")

# Check monotonicity
valid_means = [m for _, m in score_means if not np.isnan(m)]
if len(valid_means) >= 3:
    diffs = [valid_means[i+1] - valid_means[i] for i in range(len(valid_means)-1)]
    monotonic_up = all(d > 0 for d in diffs)
    mostly_up = sum(1 for d in diffs if d > 0) >= len(diffs) * 0.6
    print(f"\n  Strictly monotonic increasing? {'YES' if monotonic_up else 'NO'}")
    print(f"  Mostly increasing (60%+ steps)? {'YES' if mostly_up else 'NO'}")

# ============================================================
# YEARLY STABILITY CHECK
# ============================================================
print(f"\n{'='*70}")
print("YEARLY STABILITY (Score >= 3)")
print(f"{'='*70}")

if len(filtered) > 0:
    filtered_copy = filtered.copy()
    filtered_copy['year'] = filtered_copy['date'].dt.year
    print(f"\n  Year | N  | Mean(bps) | Sharpe |   WR  | PF")
    print(f"  -----|----|-----------| -------|-------|------")
    for year in sorted(filtered_copy['year'].unique()):
        sub = filtered_copy[filtered_copy['year'] == year]
        s = stats_summary(sub['fwd_5d_ret'].values, str(year))
        print(f"  {year} | {s['N']:2d} | {s['mean_ret_bps']:+7.1f}   | {s['sharpe']:+5.2f} | {s['wr']:5.1f}% | {s['pf']:.2f}")

# ============================================================
# FINAL VERDICT
# ============================================================
print(f"\n{'='*70}")
print("FINAL VERDICT")
print(f"{'='*70}")

# Decision criteria
has_edge = filtered_stats['mean_ret_bps'] > unfiltered_stats['mean_ret_bps']
significant = p_value < 0.10
regime_ok = regime_gap <= REGIME_GAP_THRESHOLD
enough_trades = filtered_stats['N'] >= 30
sharpe_positive = filtered_stats['sharpe'] > 0.5

criteria = {
    'Edge over baseline': has_edge,
    'Permutation p < 0.10': significant,
    'Regime-agnostic (HC #428)': regime_ok,
    'Sufficient trades (N>=30)': enough_trades,
    'Sharpe > 0.5': sharpe_positive,
}

print()
for criterion, passed in criteria.items():
    print(f"  {'PASS' if passed else 'FAIL'} | {criterion}")

n_pass = sum(criteria.values())
n_total = len(criteria)

if n_pass == n_total:
    verdict = "ALIVE"
    verdict_text = "All criteria passed. Cross-asset composite score is a valid filter for RSI<35 dip-buying."
elif n_pass >= 3:
    verdict = "PARTIAL"
    verdict_text = "Some criteria passed. Signal has merit but needs refinement or combination with other filters."
else:
    verdict = "DEAD"
    verdict_text = "Insufficient evidence. Cross-asset leading indicators do not reliably predict sector ETF dip-buy outcomes."

print(f"\n  VERDICT: {verdict}")
print(f"  {verdict_text}")

# Key takeaway
print(f"\n  KEY NUMBERS (score >= 3 filter):")
print(f"    N = {filtered_stats['N']}")
print(f"    Mean 5d return = {filtered_stats['mean_ret_bps']:+.1f} bps")
print(f"    Sharpe = {filtered_stats['sharpe']:+.2f}")
print(f"    Sortino = {filtered_stats['sortino']:+.2f}")
print(f"    PF = {filtered_stats['pf']:.2f}")
print(f"    WR = {filtered_stats['wr']:.1f}%")
print(f"    Permutation p-value = {p_value:.4f}")
print(f"    Regime gap = {regime_gap:.2f} (threshold {REGIME_GAP_THRESHOLD})")
print(f"\n{'='*70}")
