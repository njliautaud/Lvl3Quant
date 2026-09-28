"""
Research Area 4: Cross-Sector Correlation Dynamics
- When correlations spike (risk-off), does dip strategy perform differently?
- Sector dispersion as a timing signal
- Correlation regime detection
"""
import pandas as pd
import numpy as np
from scipy import stats
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUT_DIR = '/home/jupiter/Lvl3Quant/research/sector_rotation'
SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLB', 'XLU', 'XLRE']

daily_closes = pd.read_parquet(os.path.join(OUT_DIR, 'sector_daily_closes.parquet'))
daily_returns = daily_closes[SECTOR_ETFS].pct_change().dropna()
daily_returns = daily_returns.dropna()

# Also load VIX
vix = daily_closes['VIX'].reindex(daily_returns.index)

print(f"Date range: {daily_returns.index.min().date()} to {daily_returns.index.max().date()}")
print(f"Trading days: {len(daily_returns)}")

results = {}

# ============================================================
# 4A. ROLLING AVERAGE PAIRWISE CORRELATION
# ============================================================
print("\n" + "="*80)
print("4A. ROLLING SECTOR CORRELATION")
print("="*80)

for window in [21, 63, 126]:
    # Compute rolling pairwise correlation (average of all pairs)
    avg_corrs = []
    dates = []
    for i in range(window, len(daily_returns)):
        chunk = daily_returns.iloc[i-window:i]
        corr_matrix = chunk.corr()
        # Average off-diagonal
        mask = np.ones(corr_matrix.shape, dtype=bool)
        np.fill_diagonal(mask, False)
        avg_corr = corr_matrix.values[mask].mean()
        avg_corrs.append(avg_corr)
        dates.append(daily_returns.index[i])

    corr_series = pd.Series(avg_corrs, index=dates, name=f'avg_corr_{window}d')

    print(f"\n{window}d rolling avg pairwise correlation:")
    print(f"  Mean: {corr_series.mean():.4f}")
    print(f"  Std:  {corr_series.std():.4f}")
    print(f"  Min:  {corr_series.min():.4f} on {corr_series.idxmin().date()}")
    print(f"  Max:  {corr_series.max():.4f} on {corr_series.idxmax().date()}")
    print(f"  Current: {corr_series.iloc[-1]:.4f}")

    # Quintile analysis: when correlation is high vs low, what happens to dip recoveries?
    corr_quintiles = pd.qcut(corr_series, 5, labels=['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High'])

    # Forward returns by quintile
    spy_ret = daily_closes['SPY'].pct_change().reindex(corr_series.index)

    for fwd_days in [5, 21]:
        fwd = spy_ret.rolling(fwd_days).sum().shift(-fwd_days).reindex(corr_series.index)

        print(f"\n  SPY {fwd_days}d forward return by correlation quintile ({window}d window):")
        for q in ['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High']:
            mask = corr_quintiles == q
            q_fwd = fwd[mask].dropna()
            if len(q_fwd) > 20:
                t, p = stats.ttest_1samp(q_fwd, 0)
                print(f"    {q}: mean={q_fwd.mean()*100:.2f}%, WR={( q_fwd>0).mean()*100:.0f}%, n={len(q_fwd)}, p={p:.3f}")

    if window == 63:
        # Save this for dispersion analysis
        corr_63 = corr_series.copy()

    results[f'correlation_{window}d'] = {
        'mean': round(corr_series.mean(), 4),
        'std': round(corr_series.std(), 4),
        'current': round(corr_series.iloc[-1], 4),
    }

# ============================================================
# 4B. SECTOR DISPERSION AS TIMING SIGNAL
# ============================================================
print("\n" + "="*80)
print("4B. SECTOR DISPERSION AS TIMING SIGNAL")
print("="*80)

# Cross-sectional dispersion = std of sector returns each day
daily_dispersion = daily_returns.std(axis=1)
rolling_disp_21 = daily_dispersion.rolling(21).mean()
rolling_disp_63 = daily_dispersion.rolling(63).mean()

print(f"Daily cross-sectional dispersion:")
print(f"  Mean: {daily_dispersion.mean()*100:.3f}%")
print(f"  Std:  {daily_dispersion.std()*100:.3f}%")

# When dispersion is high, sectors are moving independently
# When dispersion is low, they're all moving together (likely correlation driven)

# Test: does high dispersion predict dip recovery?
disp_quintiles = pd.qcut(rolling_disp_21.dropna(), 5, labels=['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High'])

# For each sector, find dip days (>2% drawdown from 20d high)
# Then measure 5d/10d/21d forward return, stratified by dispersion quintile
print("\nDip recovery stratified by dispersion regime:")

all_dip_fwd = {q: {h: [] for h in [5, 10, 21]} for q in ['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High']}

for sector in SECTOR_ETFS:
    prices = daily_closes[sector].reindex(daily_returns.index).dropna()
    high_20d = prices.rolling(20).max()
    drawdown = (prices - high_20d) / high_20d

    dip_days = drawdown[drawdown < -0.03].index  # 3% drawdown from 20d high

    for dip_date in dip_days:
        if dip_date not in disp_quintiles.index:
            continue
        q = disp_quintiles.loc[dip_date]

        for fwd_h in [5, 10, 21]:
            fwd_idx = prices.index.get_indexer([dip_date])[0] + fwd_h
            if fwd_idx < len(prices):
                fwd_ret = prices.iloc[fwd_idx] / prices.loc[dip_date] - 1
                all_dip_fwd[q][fwd_h].append(fwd_ret)

print(f"\n{'Dispersion Q':<15}", end="")
for h in [5, 10, 21]:
    print(f"  {h}d Mean%   {h}d WR%  {h}d N", end="")
print()
print("-" * 100)

for q in ['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High']:
    print(f"{q:<15}", end="")
    for h in [5, 10, 21]:
        rets = np.array(all_dip_fwd[q][h])
        if len(rets) > 10:
            print(f"  {np.mean(rets)*100:>7.2f}  {(rets>0).mean()*100:>6.0f}% {len(rets):>5}", end="")
        else:
            print(f"  {'N/A':>7}  {'N/A':>7} {'N/A':>5}", end="")
    print()

results['dispersion_dip_recovery'] = {
    q: {
        h: {
            'mean_pct': round(np.mean(all_dip_fwd[q][h]) * 100, 3) if len(all_dip_fwd[q][h]) > 10 else None,
            'win_rate': round((np.array(all_dip_fwd[q][h]) > 0).mean() * 100, 1) if len(all_dip_fwd[q][h]) > 10 else None,
            'n': len(all_dip_fwd[q][h])
        } for h in [5, 10, 21]
    } for q in ['Q1_Low', 'Q2', 'Q3', 'Q4', 'Q5_High']
}

# ============================================================
# 4C. VIX vs DISPERSION — which better predicts dip recovery?
# ============================================================
print("\n" + "="*80)
print("4C. VIX vs DISPERSION — PREDICTIVE COMPARISON")
print("="*80)

# Combine VIX and dispersion for multivariate analysis
combined = pd.DataFrame({
    'vix': vix,
    'dispersion': rolling_disp_21,
    'correlation': corr_63 if 'corr_63' in dir() else np.nan
}).dropna()

# For each sector, compute dip recovery returns
all_recoveries = []
for sector in SECTOR_ETFS:
    prices = daily_closes[sector].reindex(combined.index).dropna()
    high_20d = prices.rolling(20).max()
    drawdown = (prices - high_20d) / high_20d

    dip_days = drawdown[drawdown < -0.03].index

    for dip_date in dip_days:
        if dip_date not in combined.index:
            continue
        fwd_idx = prices.index.get_indexer([dip_date])[0] + 10
        if fwd_idx < len(prices):
            fwd_ret = prices.iloc[fwd_idx] / prices.loc[dip_date] - 1
            all_recoveries.append({
                'date': dip_date,
                'sector': sector,
                'fwd_10d_ret': fwd_ret,
                'vix': combined.loc[dip_date, 'vix'],
                'dispersion': combined.loc[dip_date, 'dispersion'],
                'drawdown': drawdown.loc[dip_date]
            })

rec_df = pd.DataFrame(all_recoveries)
print(f"Total dip observations: {len(rec_df)}")

# Correlation of predictors with recovery
print(f"\nCorrelation with 10d forward return after dip:")
for col in ['vix', 'dispersion', 'drawdown']:
    corr = rec_df['fwd_10d_ret'].corr(rec_df[col])
    # Spearman rank correlation
    sp_corr, sp_p = stats.spearmanr(rec_df['fwd_10d_ret'], rec_df[col])
    print(f"  {col:>12}: Pearson={corr:.4f}, Spearman={sp_corr:.4f} (p={sp_p:.4f})")

results['predictor_correlation'] = {
    col: {
        'pearson': round(rec_df['fwd_10d_ret'].corr(rec_df[col]), 4),
        'spearman': round(stats.spearmanr(rec_df['fwd_10d_ret'], rec_df[col])[0], 4),
        'spearman_p': round(stats.spearmanr(rec_df['fwd_10d_ret'], rec_df[col])[1], 4),
    }
    for col in ['vix', 'dispersion', 'drawdown']
}

# VIX tercile interaction with drawdown depth
print(f"\nVIX tercile x Drawdown depth interaction:")
rec_df['vix_tercile'] = pd.qcut(rec_df['vix'], 3, labels=['Low_VIX', 'Med_VIX', 'High_VIX'])
rec_df['dd_tercile'] = pd.qcut(rec_df['drawdown'], 3, labels=['Mild_DD', 'Med_DD', 'Deep_DD'])

print(f"{'VIX':>10} {'DD':>10} {'10d Ret%':>10} {'WR%':>8} {'N':>5}")
for vt in ['Low_VIX', 'Med_VIX', 'High_VIX']:
    for dt in ['Mild_DD', 'Med_DD', 'Deep_DD']:
        mask = (rec_df['vix_tercile'] == vt) & (rec_df['dd_tercile'] == dt)
        subset = rec_df[mask]
        if len(subset) > 5:
            print(f"{vt:>10} {dt:>10} {subset['fwd_10d_ret'].mean()*100:>9.2f}% {(subset['fwd_10d_ret']>0).mean()*100:>7.0f}% {len(subset):>5}")

# ============================================================
# 4D. CORRELATION REGIME SHIFTS — when correlations spike
# ============================================================
print("\n" + "="*80)
print("4D. CORRELATION SPIKES AND DIP STRATEGY PERFORMANCE")
print("="*80)

if 'corr_63' in dir():
    # Define correlation spike as >1.5 std above mean
    corr_mean = corr_63.mean()
    corr_std = corr_63.std()
    spike_threshold = corr_mean + 1.5 * corr_std

    spike_days = corr_63[corr_63 > spike_threshold]
    normal_days = corr_63[corr_63 <= spike_threshold]

    print(f"Correlation spike threshold (1.5 std): {spike_threshold:.4f}")
    print(f"Spike days: {len(spike_days)} ({len(spike_days)/len(corr_63)*100:.1f}%)")

    # Dip recovery during spike vs normal
    spike_recs = rec_df[rec_df['date'].isin(spike_days.index)]
    normal_recs = rec_df[~rec_df['date'].isin(spike_days.index)]

    if len(spike_recs) > 10 and len(normal_recs) > 10:
        print(f"\nDip recovery during correlation spikes:")
        print(f"  Spike:  mean={spike_recs['fwd_10d_ret'].mean()*100:.2f}%, WR={( spike_recs['fwd_10d_ret']>0).mean()*100:.0f}%, n={len(spike_recs)}")
        print(f"  Normal: mean={normal_recs['fwd_10d_ret'].mean()*100:.2f}%, WR={(normal_recs['fwd_10d_ret']>0).mean()*100:.0f}%, n={len(normal_recs)}")

        t, p = stats.ttest_ind(spike_recs['fwd_10d_ret'], normal_recs['fwd_10d_ret'])
        print(f"  Difference t-stat: {t:.3f}, p-value: {p:.4f}")

        results['correlation_spike_effect'] = {
            'spike_mean_pct': round(spike_recs['fwd_10d_ret'].mean() * 100, 3),
            'spike_wr': round((spike_recs['fwd_10d_ret'] > 0).mean() * 100, 1),
            'spike_n': len(spike_recs),
            'normal_mean_pct': round(normal_recs['fwd_10d_ret'].mean() * 100, 3),
            'normal_wr': round((normal_recs['fwd_10d_ret'] > 0).mean() * 100, 1),
            'normal_n': len(normal_recs),
            't_stat': round(t, 3),
            'p_value': round(p, 4)
        }

# Save
with open(os.path.join(OUT_DIR, 'correlation_dispersion_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n=== Results saved ===")
