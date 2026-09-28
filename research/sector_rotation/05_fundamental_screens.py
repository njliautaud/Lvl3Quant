"""
Research Area 3: Fundamental Factor Screens
- Do cheap sectors (by P/E, relative valuation) recover faster from dips?
- Relative valuation as sector selection criterion
- Historical PE-based dip selection
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
daily_returns = daily_closes[SECTOR_ETFS].pct_change().dropna().dropna()

# Load current fundamentals
with open(os.path.join(OUT_DIR, 'sector_fundamentals.json')) as f:
    fund_data = json.load(f)

results = {}

# ============================================================
# 3A. CURRENT FUNDAMENTAL SNAPSHOT
# ============================================================
print("="*80)
print("3A. CURRENT SECTOR FUNDAMENTAL SNAPSHOT")
print("="*80)

print(f"\n{'Sector':<8} {'Name':<40} {'Trailing PE':>12} {'Forward PE':>12} {'P/B':>8} {'DivYld%':>8}")
print("-" * 92)

pe_data = {}
for sector in SECTOR_ETFS:
    d = fund_data.get(sector, {})
    name = d.get('longName', sector)[:40]
    tpe = d.get('trailingPE')
    fpe = d.get('forwardPE')
    pb = d.get('priceToBook')
    dy = d.get('dividendYield')

    print(f"{sector:<8} {name:<40} {tpe if tpe else 'N/A':>12} {fpe if fpe else 'N/A':>12} {pb if pb else 'N/A':>8} {dy*100 if dy else 'N/A':>8}")

    if tpe:
        pe_data[sector] = tpe

results['current_fundamentals'] = fund_data

# ============================================================
# 3B. HISTORICAL RELATIVE VALUATION PROXY
# Since we don't have historical P/E data, we'll use a proxy:
# Relative performance vs SPY over various lookbacks as a "cheapness" indicator
# (sectors that have underperformed are relatively cheaper)
# ============================================================
print("\n" + "="*80)
print("3B. RELATIVE CHEAPNESS (UNDERPERFORMANCE) AS DIP RECOVERY PREDICTOR")
print("="*80)

spy_returns = daily_closes['SPY'].pct_change().reindex(daily_returns.index)

# Compute relative performance (sector return - SPY return) over lookback
for lookback in [21, 63, 126]:
    rel_perf = pd.DataFrame()
    for sector in SECTOR_ETFS:
        sector_cum = daily_returns[sector].rolling(lookback).sum()
        spy_cum = spy_returns.rolling(lookback).sum()
        rel_perf[sector] = sector_cum - spy_cum

    rel_perf = rel_perf.dropna()

    # When a sector has underperformed, find its forward return from dips
    print(f"\n--- Lookback {lookback}d ---")

    # Rank sectors by relative performance each day
    rel_ranks = rel_perf.rank(axis=1, ascending=True)  # 1 = most underperforming (cheapest)

    # When a sector is in bottom 3 (cheapest) AND has a dip, compare recovery to top 3
    cheap_fwd = []
    expensive_fwd = []

    for sector in SECTOR_ETFS:
        prices = daily_closes[sector].reindex(rel_perf.index).dropna()
        high_20d = prices.rolling(20).max()
        drawdown = (prices - high_20d) / high_20d

        dip_days = drawdown[drawdown < -0.03].index

        for dip_date in dip_days:
            if dip_date not in rel_ranks.index:
                continue

            rank = rel_ranks.loc[dip_date, sector]

            fwd_idx = prices.index.get_indexer([dip_date])[0] + 10
            if fwd_idx >= len(prices):
                continue
            fwd_ret = prices.iloc[fwd_idx] / prices.loc[dip_date] - 1

            if rank <= 3:  # Bottom 3 (cheapest)
                cheap_fwd.append(fwd_ret)
            elif rank >= len(SECTOR_ETFS) - 2:  # Top 3 (most expensive)
                expensive_fwd.append(fwd_ret)

    cheap_arr = np.array(cheap_fwd)
    exp_arr = np.array(expensive_fwd)

    print(f"  Cheap sectors (bottom 3 relative perf) dip recovery 10d:")
    print(f"    Mean: {np.mean(cheap_arr)*100:.2f}%, WR: {(cheap_arr>0).mean()*100:.0f}%, N={len(cheap_arr)}")
    print(f"  Expensive sectors (top 3 relative perf) dip recovery 10d:")
    print(f"    Mean: {np.mean(exp_arr)*100:.2f}%, WR: {(exp_arr>0).mean()*100:.0f}%, N={len(exp_arr)}")

    t, p = stats.ttest_ind(cheap_arr, exp_arr)
    print(f"  Difference t-stat: {t:.3f}, p-value: {p:.4f}")
    print(f"  Cheap - Expensive spread: {(np.mean(cheap_arr) - np.mean(exp_arr))*100:.2f}%")

    results[f'cheap_vs_expensive_{lookback}d'] = {
        'cheap_mean_pct': round(np.mean(cheap_arr) * 100, 3),
        'cheap_wr': round((cheap_arr > 0).mean() * 100, 1),
        'cheap_n': len(cheap_arr),
        'expensive_mean_pct': round(np.mean(exp_arr) * 100, 3),
        'expensive_wr': round((exp_arr > 0).mean() * 100, 1),
        'expensive_n': len(exp_arr),
        't_stat': round(t, 3),
        'p_value': round(p, 4),
        'spread_pct': round((np.mean(cheap_arr) - np.mean(exp_arr)) * 100, 3)
    }

# ============================================================
# 3C. EARNINGS MOMENTUM PROXY (price momentum as proxy)
# Sectors with improving relative strength = earnings upgrades proxy
# ============================================================
print("\n" + "="*80)
print("3C. EARNINGS MOMENTUM PROXY — RELATIVE STRENGTH ACCELERATION")
print("="*80)

# RS acceleration = 1-month relative perf minus 3-month relative perf (normalized)
for sector in SECTOR_ETFS:
    rs_1m = (daily_returns[sector].rolling(21).sum() - spy_returns.rolling(21).sum())
    rs_3m = (daily_returns[sector].rolling(63).sum() - spy_returns.rolling(63).sum()) / 3

# Aggregate: which sectors have improving vs declining RS?
rs_accel = pd.DataFrame()
for sector in SECTOR_ETFS:
    rs_1m = daily_returns[sector].rolling(21).sum() - spy_returns.rolling(21).sum()
    rs_3m = (daily_returns[sector].rolling(63).sum() - spy_returns.rolling(63).sum()) / 3
    rs_accel[sector] = rs_1m - rs_3m  # positive = accelerating outperformance

rs_accel = rs_accel.dropna()

# Test: does RS acceleration predict forward returns?
print(f"\nRS Acceleration vs 21d forward return (all sectors pooled):")

all_accel_rets = []
for sector in SECTOR_ETFS:
    accel = rs_accel[sector]
    fwd = daily_returns[sector].rolling(21).sum().shift(-21).reindex(accel.index)
    valid = accel.dropna().index.intersection(fwd.dropna().index)
    for date in valid[::21]:  # Non-overlapping
        all_accel_rets.append({'accel': accel.loc[date], 'fwd_21d': fwd.loc[date]})

accel_df = pd.DataFrame(all_accel_rets)
corr_p, p_p = stats.pearsonr(accel_df['accel'], accel_df['fwd_21d'])
corr_s, p_s = stats.spearmanr(accel_df['accel'], accel_df['fwd_21d'])

print(f"  Pearson: {corr_p:.4f} (p={p_p:.4f})")
print(f"  Spearman: {corr_s:.4f} (p={p_s:.4f})")
print(f"  N observations: {len(accel_df)}")

# Quintile analysis
accel_df['q'] = pd.qcut(accel_df['accel'], 5, labels=['Q1_Decel', 'Q2', 'Q3', 'Q4', 'Q5_Accel'])
print(f"\n{'Quintile':<12} {'Mean 21d%':>10} {'WR%':>8} {'N':>6}")
print("-" * 40)
for q in ['Q1_Decel', 'Q2', 'Q3', 'Q4', 'Q5_Accel']:
    subset = accel_df[accel_df['q'] == q]
    print(f"{q:<12} {subset['fwd_21d'].mean()*100:>9.2f}% {(subset['fwd_21d']>0).mean()*100:>7.0f}% {len(subset):>6}")

results['rs_acceleration'] = {
    'pearson_corr': round(corr_p, 4),
    'pearson_p': round(p_p, 4),
    'spearman_corr': round(corr_s, 4),
    'spearman_p': round(p_s, 4),
    'n': len(accel_df)
}

# ============================================================
# 3D. CURRENT PE RANK — does current PE predict dip recovery cross-sectionally?
# ============================================================
print("\n" + "="*80)
print("3D. CURRENT SECTOR PE vs HISTORICAL DIP RECOVERY")
print("="*80)

# Use the pe_data we have (current snapshot) as a proxy
# This is a cross-sectional test: do low-PE sectors recover better?
print(f"\nCurrent PE rankings:")
pe_sorted = sorted(pe_data.items(), key=lambda x: x[1])
for rank, (sector, pe) in enumerate(pe_sorted, 1):
    print(f"  {rank}. {sector}: PE={pe:.1f}")

# Historical recovery by sector
print(f"\nHistorical 10d dip recovery by sector (3% dip threshold):")
recovery_by_sector = {}
for sector in SECTOR_ETFS:
    prices = daily_closes[sector].dropna()
    high_20d = prices.rolling(20).max()
    drawdown = (prices - high_20d) / high_20d
    dip_days = drawdown[drawdown < -0.03].index

    fwd_rets = []
    for dip_date in dip_days:
        fwd_idx = prices.index.get_indexer([dip_date])[0] + 10
        if fwd_idx < len(prices):
            fwd_rets.append(prices.iloc[fwd_idx] / prices.loc[dip_date] - 1)

    if fwd_rets:
        arr = np.array(fwd_rets)
        recovery_by_sector[sector] = {
            'mean_pct': np.mean(arr) * 100,
            'median_pct': np.median(arr) * 100,
            'win_rate': (arr > 0).mean() * 100,
            'n_dips': len(arr),
            'pe': pe_data.get(sector)
        }

print(f"\n{'Sector':<8} {'PE':>8} {'10d Recov%':>12} {'WR%':>8} {'N dips':>8}")
print("-" * 50)
for sector in [s for s, _ in pe_sorted]:
    if sector in recovery_by_sector:
        d = recovery_by_sector[sector]
        pe = d.get('pe', 'N/A')
        pe_str = f"{pe:.1f}" if pe else "N/A"
        print(f"{sector:<8} {pe_str:>8} {d['mean_pct']:>11.2f}% {d['win_rate']:>7.0f}% {d['n_dips']:>8}")

# Correlation between PE and recovery rate
pe_list = []
rec_list = []
for sector in recovery_by_sector:
    if recovery_by_sector[sector].get('pe'):
        pe_list.append(recovery_by_sector[sector]['pe'])
        rec_list.append(recovery_by_sector[sector]['mean_pct'])

if len(pe_list) > 4:
    corr, p = stats.pearsonr(pe_list, rec_list)
    print(f"\nCorrelation(PE, recovery): {corr:.3f} (p={p:.3f}), n={len(pe_list)}")
    results['pe_recovery_correlation'] = {'pearson': round(corr, 3), 'p_value': round(p, 3)}

results['recovery_by_sector'] = {
    k: {kk: round(vv, 3) if isinstance(vv, float) else vv for kk, vv in v.items()}
    for k, v in recovery_by_sector.items()
}

# Save
with open(os.path.join(OUT_DIR, 'fundamental_screen_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n=== Results saved ===")
