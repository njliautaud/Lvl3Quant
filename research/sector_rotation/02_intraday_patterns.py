"""
Research Area 1: Time-of-Day Return Patterns for Sector ETFs
- Persistent intraday hourly patterns
- Optimal hour to enter dip-buying positions
- First hour vs last hour effects
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

# Load hourly data
hourly_closes = pd.read_parquet(os.path.join(OUT_DIR, 'sector_hourly_closes.parquet'))
hourly_volumes = pd.read_parquet(os.path.join(OUT_DIR, 'sector_hourly_volumes.parquet'))

# Load daily data for regime classification
daily_closes = pd.read_parquet(os.path.join(OUT_DIR, 'sector_daily_closes.parquet'))

print(f"Hourly data: {hourly_closes.index.min()} to {hourly_closes.index.max()}, {len(hourly_closes)} bars")
print(f"Sectors: {SECTOR_ETFS}")

# ============================================================
# Convert to ET and extract hour
# ============================================================
hourly_closes.index = hourly_closes.index.tz_convert('US/Eastern') if hourly_closes.index.tz else hourly_closes.index.tz_localize('UTC').tz_convert('US/Eastern')
hourly_volumes.index = hourly_volumes.index.tz_convert('US/Eastern') if hourly_volumes.index.tz else hourly_volumes.index.tz_localize('UTC').tz_convert('US/Eastern')

# Compute hourly returns
hourly_returns = hourly_closes.pct_change()
hourly_returns['hour'] = hourly_returns.index.hour
hourly_returns['date'] = hourly_returns.index.date

# Filter to market hours only (9:30-16:00 ET = hours 9,10,11,12,13,14,15)
# yfinance hourly bars: 9:30 bar covers 9:30-10:30, etc.
market_hours = hourly_returns[hourly_returns['hour'].isin([9, 10, 11, 12, 13, 14, 15])]

print(f"\nMarket-hour returns: {len(market_hours)} bars")
print(f"Unique trading days: {market_hours['date'].nunique()}")

# ============================================================
# 1A. Average return by hour — all sectors
# ============================================================
results = {}

print("\n" + "="*80)
print("1A. AVERAGE HOURLY RETURNS BY SECTOR (annualized bps)")
print("="*80)

hour_labels = {9: '9:30-10:30', 10: '10:30-11:30', 11: '11:30-12:30',
               12: '12:30-13:30', 13: '13:30-14:30', 14: '14:30-15:30', 15: '15:30-16:00'}

hourly_stats = {}
for sector in SECTOR_ETFS:
    sector_data = market_hours[[sector, 'hour']].dropna()
    stats_by_hour = {}
    for hour in sorted(sector_data['hour'].unique()):
        rets = sector_data[sector_data['hour'] == hour][sector]
        mean_bps = rets.mean() * 10000
        std_bps = rets.std() * 10000
        sharpe = (rets.mean() / rets.std()) * np.sqrt(252) if rets.std() > 0 else 0
        t_stat, p_val = stats.ttest_1samp(rets, 0)
        n = len(rets)
        stats_by_hour[int(hour)] = {
            'mean_bps': round(mean_bps, 2),
            'std_bps': round(std_bps, 2),
            'sharpe_ann': round(sharpe, 3),
            't_stat': round(t_stat, 3),
            'p_value': round(p_val, 4),
            'n': n,
            'significant': p_val < 0.05
        }
    hourly_stats[sector] = stats_by_hour

# Print summary table
print(f"\n{'Sector':<8}", end="")
for h in [9,10,11,12,13,14,15]:
    print(f"  {hour_labels[h]:>12}", end="")
print()
print("-" * 100)

for sector in SECTOR_ETFS:
    print(f"{sector:<8}", end="")
    for h in [9,10,11,12,13,14,15]:
        if h in hourly_stats[sector]:
            val = hourly_stats[sector][h]['mean_bps']
            sig = '*' if hourly_stats[sector][h]['significant'] else ' '
            print(f"  {val:>10.2f}{sig}", end="")
        else:
            print(f"  {'N/A':>11}", end="")
    print()

results['hourly_returns_by_sector'] = hourly_stats

# ============================================================
# 1B. First hour vs last hour effect — aggregate
# ============================================================
print("\n" + "="*80)
print("1B. FIRST HOUR vs LAST HOUR EFFECT")
print("="*80)

first_last = {}
for sector in SECTOR_ETFS:
    sector_data = market_hours[[sector, 'hour']].dropna()
    first_hour = sector_data[sector_data['hour'] == 9][sector]
    last_hour = sector_data[sector_data['hour'] == 15][sector]

    # T-test: are they different?
    t_stat, p_val = stats.ttest_ind(first_hour, last_hour)

    first_last[sector] = {
        'first_hour_mean_bps': round(first_hour.mean() * 10000, 2),
        'first_hour_std_bps': round(first_hour.std() * 10000, 2),
        'first_hour_sharpe': round((first_hour.mean() / first_hour.std()) * np.sqrt(252), 3) if first_hour.std() > 0 else 0,
        'last_hour_mean_bps': round(last_hour.mean() * 10000, 2),
        'last_hour_std_bps': round(last_hour.std() * 10000, 2),
        'last_hour_sharpe': round((last_hour.mean() / last_hour.std()) * np.sqrt(252), 3) if last_hour.std() > 0 else 0,
        'diff_t_stat': round(t_stat, 3),
        'diff_p_value': round(p_val, 4),
        'first_n': len(first_hour),
        'last_n': len(last_hour)
    }

print(f"\n{'Sector':<8} {'1st Hr Mean':>12} {'1st Hr Sharpe':>14} {'Last Hr Mean':>12} {'Last Hr Sharpe':>14} {'t-stat':>8} {'p-val':>8}")
print("-" * 80)
for sector in SECTOR_ETFS:
    d = first_last[sector]
    print(f"{sector:<8} {d['first_hour_mean_bps']:>10.2f}bp {d['first_hour_sharpe']:>13.3f} {d['last_hour_mean_bps']:>10.2f}bp {d['last_hour_sharpe']:>13.3f} {d['diff_t_stat']:>8.3f} {d['diff_p_value']:>8.4f}")

results['first_vs_last_hour'] = first_last

# ============================================================
# 1C. Optimal entry hour for DIP BUYING
# After sector drops >1% intraday, which hour gives best forward return?
# ============================================================
print("\n" + "="*80)
print("1C. OPTIMAL DIP ENTRY HOUR (after intraday drawdown)")
print("="*80)

# For each day, compute cumulative intraday return at each hour
# If sector is down >1% at any point, look at returns from each subsequent hour to close

dip_entry_results = {}
for sector in SECTOR_ETFS:
    sector_hourly = market_hours[[sector, 'hour', 'date']].dropna()

    # Get daily opens (first bar close approximates previous close)
    daily_groups = sector_hourly.groupby('date')

    entry_returns = {h: [] for h in [9,10,11,12,13,14,15]}

    for date, group in daily_groups:
        group = group.sort_index()
        if len(group) < 3:
            continue

        # Cumulative return from first bar
        cum_ret = (1 + group[sector]).cumprod() - 1

        # Check if day had a dip > 1%
        min_cum_ret = cum_ret.cummin()
        if min_cum_ret.min() < -0.01:  # Had >1% drawdown at some point
            close_ret = cum_ret.iloc[-1]  # return at close

            for i, (idx, row) in enumerate(group.iterrows()):
                hour = int(row['hour'])
                # Return from this hour to close
                ret_to_close = (1 + close_ret) / (1 + cum_ret.iloc[i]) - 1
                entry_returns[hour].append(ret_to_close)

    stats_by_entry = {}
    for hour in [9,10,11,12,13,14,15]:
        rets = np.array(entry_returns[hour])
        if len(rets) > 10:
            stats_by_entry[hour] = {
                'mean_bps': round(np.mean(rets) * 10000, 2),
                'median_bps': round(np.median(rets) * 10000, 2),
                'win_rate': round((rets > 0).mean() * 100, 1),
                'sharpe': round(np.mean(rets) / np.std(rets) * np.sqrt(252), 3) if np.std(rets) > 0 else 0,
                'n_dip_days': len(rets)
            }

    dip_entry_results[sector] = stats_by_entry

# Aggregate across all sectors
print(f"\nAggregate dip-entry returns (after >1% intraday drawdown):")
print(f"{'Hour':<15} {'Mean bps':>10} {'Med bps':>10} {'WR%':>8} {'Sharpe':>8} {'N days':>8}")
print("-" * 65)

agg_by_hour = {}
for h in [9,10,11,12,13,14,15]:
    all_rets = []
    for sector in SECTOR_ETFS:
        if h in dip_entry_results[sector]:
            # reconstruct from mean*n
            pass
    # Direct aggregation
    all_means = []
    all_wrs = []
    all_n = 0
    for sector in SECTOR_ETFS:
        if h in dip_entry_results[sector]:
            all_means.append(dip_entry_results[sector][h]['mean_bps'])
            all_wrs.append(dip_entry_results[sector][h]['win_rate'])
            all_n += dip_entry_results[sector][h]['n_dip_days']
    if all_means:
        avg_mean = np.mean(all_means)
        avg_wr = np.mean(all_wrs)
        print(f"{hour_labels.get(h, str(h)):<15} {avg_mean:>10.1f} {'':>10} {avg_wr:>7.1f}% {'':>8} {all_n:>8}")
        agg_by_hour[h] = {'avg_mean_bps': round(avg_mean, 2), 'avg_win_rate': round(avg_wr, 1), 'total_obs': all_n}

results['dip_entry_by_hour'] = dip_entry_results
results['dip_entry_aggregate'] = agg_by_hour

# Per-sector best entry hour
print(f"\nBest dip-entry hour by sector:")
for sector in SECTOR_ETFS:
    if dip_entry_results[sector]:
        best_h = max(dip_entry_results[sector].keys(),
                     key=lambda h: dip_entry_results[sector][h].get('mean_bps', -999))
        d = dip_entry_results[sector][best_h]
        print(f"  {sector}: {hour_labels.get(best_h, str(best_h))} — {d['mean_bps']:.1f}bps mean, {d['win_rate']:.0f}% WR, n={d['n_dip_days']}")

# ============================================================
# 1D. Regime-stratified intraday patterns (VIX high/low)
# ============================================================
print("\n" + "="*80)
print("1D. REGIME-STRATIFIED INTRADAY PATTERNS (VIX HIGH vs LOW)")
print("="*80)

# Get daily VIX
vix_daily = daily_closes['VIX'].dropna()
vix_median = vix_daily.median()
print(f"VIX median: {vix_median:.1f}")

# Map each hourly bar to VIX regime
hourly_dates = pd.Series(market_hours.index.date, index=market_hours.index)
vix_regime = {}
for d in hourly_dates.unique():
    ts = pd.Timestamp(d)
    if ts in vix_daily.index:
        vix_regime[d] = 'HIGH_VIX' if vix_daily.loc[ts] > vix_median else 'LOW_VIX'

market_hours_regime = market_hours.copy()
market_hours_regime['regime'] = [vix_regime.get(d, 'UNKNOWN') for d in market_hours_regime['date']]
market_hours_regime = market_hours_regime[market_hours_regime['regime'] != 'UNKNOWN']

regime_hourly = {}
for regime in ['HIGH_VIX', 'LOW_VIX']:
    regime_data = market_hours_regime[market_hours_regime['regime'] == regime]
    regime_stats = {}
    for sector in SECTOR_ETFS:
        sector_regime = regime_data[[sector, 'hour']].dropna()
        by_hour = {}
        for hour in [9,10,11,12,13,14,15]:
            rets = sector_regime[sector_regime['hour'] == hour][sector]
            if len(rets) > 10:
                by_hour[int(hour)] = {
                    'mean_bps': round(rets.mean() * 10000, 2),
                    'sharpe': round((rets.mean() / rets.std()) * np.sqrt(252), 3) if rets.std() > 0 else 0,
                    'n': len(rets)
                }
        regime_stats[sector] = by_hour
    regime_hourly[regime] = regime_stats

# Print aggregate first-hour vs last-hour by regime
for regime in ['HIGH_VIX', 'LOW_VIX']:
    print(f"\n{regime} (VIX {'>' if regime == 'HIGH_VIX' else '<='} {vix_median:.1f}):")
    first_means = []
    last_means = []
    for sector in SECTOR_ETFS:
        if 9 in regime_hourly[regime][sector]:
            first_means.append(regime_hourly[regime][sector][9]['mean_bps'])
        if 15 in regime_hourly[regime][sector]:
            last_means.append(regime_hourly[regime][sector][15]['mean_bps'])
    if first_means and last_means:
        print(f"  First hour avg: {np.mean(first_means):.2f} bps")
        print(f"  Last hour avg:  {np.mean(last_means):.2f} bps")

results['regime_hourly_patterns'] = regime_hourly

# ============================================================
# 1E. Day-of-week effects
# ============================================================
print("\n" + "="*80)
print("1E. DAY-OF-WEEK RETURN PATTERNS (daily)")
print("="*80)

daily_returns = daily_closes[SECTOR_ETFS].pct_change().dropna()
daily_returns['dow'] = daily_returns.index.dayofweek  # 0=Mon, 4=Fri

dow_labels = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
dow_stats = {}
for sector in SECTOR_ETFS:
    by_dow = {}
    for dow in range(5):
        rets = daily_returns[daily_returns['dow'] == dow][sector].dropna()
        t, p = stats.ttest_1samp(rets, 0)
        by_dow[dow_labels[dow]] = {
            'mean_bps': round(rets.mean() * 10000, 2),
            'sharpe_ann': round((rets.mean() / rets.std()) * np.sqrt(252), 3) if rets.std() > 0 else 0,
            'p_value': round(p, 4),
            'n': len(rets)
        }
    dow_stats[sector] = by_dow

# Print aggregate
print(f"\n{'Day':<12}", end="")
for sector in SECTOR_ETFS:
    print(f" {sector:>6}", end="")
print()
print("-" * (12 + 7 * len(SECTOR_ETFS)))
for dow in ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']:
    print(f"{dow:<12}", end="")
    for sector in SECTOR_ETFS:
        val = dow_stats[sector][dow]['mean_bps']
        print(f" {val:>6.1f}", end="")
    print()

results['day_of_week'] = dow_stats

# Save results
with open(os.path.join(OUT_DIR, 'intraday_patterns_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n=== Results saved to {OUT_DIR}/intraday_patterns_results.json ===")
