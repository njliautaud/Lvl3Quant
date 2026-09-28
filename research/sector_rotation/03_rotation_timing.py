"""
Research Area 2: Sector Rotation Timing
- How does sector leadership rotate? Duration of dominance?
- Can lagging sectors predict next leaders? (mean-reversion vs momentum)
- Optimal lookback/holding periods
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

# Load daily data
daily_closes = pd.read_parquet(os.path.join(OUT_DIR, 'sector_daily_closes.parquet'))
daily_returns = daily_closes[SECTOR_ETFS].pct_change().dropna()

# Use common date range where all sectors exist
daily_returns = daily_returns.dropna()
print(f"Common date range: {daily_returns.index.min().date()} to {daily_returns.index.max().date()}")
print(f"Trading days: {len(daily_returns)}")

results = {}

# ============================================================
# 2A. SECTOR LEADERSHIP PERSISTENCE
# Who's #1 and how long do they stay #1?
# ============================================================
print("\n" + "="*80)
print("2A. SECTOR LEADERSHIP PERSISTENCE")
print("="*80)

for lookback in [21, 63, 126, 252]:  # 1mo, 3mo, 6mo, 1yr
    rolling_ret = daily_returns.rolling(lookback).sum()
    rolling_ret = rolling_ret.dropna()

    # Rank sectors each day (1 = best)
    ranks = rolling_ret.rank(axis=1, ascending=False)

    # Who is #1?
    leaders = ranks.idxmin(axis=1)  # Actually we want rank==1
    leaders = rolling_ret.idxmax(axis=1)

    # Count consecutive days of leadership
    streaks = []
    current_leader = None
    current_streak = 0
    for date, leader in leaders.items():
        if leader == current_leader:
            current_streak += 1
        else:
            if current_leader is not None:
                streaks.append({'sector': current_leader, 'streak_days': current_streak})
            current_leader = leader
            current_streak = 1
    if current_leader:
        streaks.append({'sector': current_leader, 'streak_days': current_streak})

    streak_df = pd.DataFrame(streaks)
    avg_streak = streak_df['streak_days'].mean()
    median_streak = streak_df['streak_days'].median()
    max_streak = streak_df['streak_days'].max()

    print(f"\nLookback={lookback}d ({lookback//21:.0f}mo):")
    print(f"  Leadership changes: {len(streaks)}")
    print(f"  Avg streak: {avg_streak:.1f} days")
    print(f"  Median streak: {median_streak:.0f} days")
    print(f"  Max streak: {max_streak} days")
    print(f"  Leader frequency:")
    leader_counts = streak_df['sector'].value_counts()
    for s, c in leader_counts.head(5).items():
        avg_s = streak_df[streak_df['sector'] == s]['streak_days'].mean()
        print(f"    {s}: {c} times, avg streak {avg_s:.0f} days")

    results[f'leadership_{lookback}d'] = {
        'n_changes': len(streaks),
        'avg_streak_days': round(avg_streak, 1),
        'median_streak_days': int(median_streak),
        'max_streak_days': int(max_streak),
        'leader_counts': leader_counts.to_dict(),
    }

# ============================================================
# 2B. MEAN REVERSION vs MOMENTUM — SYSTEMATIC TEST
# Buy past losers vs buy past winners at different horizons
# ============================================================
print("\n" + "="*80)
print("2B. MEAN REVERSION vs MOMENTUM AT SECTOR LEVEL")
print("="*80)

lookbacks = [5, 10, 21, 42, 63, 126, 252]
holding_periods = [5, 10, 21, 42, 63]

momentum_results = {}
reversion_results = {}

for lb in lookbacks:
    for hp in holding_periods:
        # Past return
        past_ret = daily_returns.rolling(lb).sum()

        # Forward return
        fwd_ret = daily_returns.rolling(hp).sum().shift(-hp)

        # Drop NaN
        valid = past_ret.dropna().index.intersection(fwd_ret.dropna().index)
        past_ret = past_ret.loc[valid]
        fwd_ret = fwd_ret.loc[valid]

        # Strategy: each period, rank sectors by past return
        # MOMENTUM: buy top 3, short bottom 3
        # REVERSION: buy bottom 3, short top 3

        mom_rets = []
        rev_rets = []

        # Sample every hp days to avoid overlapping
        sample_dates = valid[::hp]

        for date in sample_dates:
            past = past_ret.loc[date]
            fwd = fwd_ret.loc[date]
            if past.isna().any() or fwd.isna().any():
                continue

            ranked = past.sort_values()
            bottom3 = ranked.index[:3]
            top3 = ranked.index[-3:]

            # Momentum: long top3, short bottom3
            mom_ret = fwd[top3].mean() - fwd[bottom3].mean()
            mom_rets.append(mom_ret)

            # Reversion: long bottom3, short top3
            rev_ret = fwd[bottom3].mean() - fwd[top3].mean()
            rev_rets.append(rev_ret)

        if len(mom_rets) > 10:
            mom_arr = np.array(mom_rets)
            rev_arr = np.array(rev_rets)

            mom_t, mom_p = stats.ttest_1samp(mom_arr, 0)
            rev_t, rev_p = stats.ttest_1samp(rev_arr, 0)

            # Annualize
            periods_per_year = 252 / hp
            mom_sharpe = (np.mean(mom_arr) / np.std(mom_arr)) * np.sqrt(periods_per_year) if np.std(mom_arr) > 0 else 0
            rev_sharpe = (np.mean(rev_arr) / np.std(rev_arr)) * np.sqrt(periods_per_year) if np.std(rev_arr) > 0 else 0

            key = f'lb{lb}_hp{hp}'
            momentum_results[key] = {
                'lookback': lb, 'holding': hp,
                'mean_ret': round(np.mean(mom_arr) * 100, 3),
                'sharpe': round(mom_sharpe, 3),
                'win_rate': round((mom_arr > 0).mean() * 100, 1),
                't_stat': round(mom_t, 3),
                'p_value': round(mom_p, 4),
                'n': len(mom_arr)
            }
            reversion_results[key] = {
                'lookback': lb, 'holding': hp,
                'mean_ret': round(np.mean(rev_arr) * 100, 3),
                'sharpe': round(rev_sharpe, 3),
                'win_rate': round((rev_arr > 0).mean() * 100, 1),
                't_stat': round(rev_t, 3),
                'p_value': round(rev_p, 4),
                'n': len(rev_arr)
            }

# Print as heatmap
print("\nMOMENTUM SHARPE (buy winners, short losers):")
hdr = 'LB\\HP'
print(f"{hdr:<8}", end="")
for hp in holding_periods:
    print(f"  {hp:>5}d", end="")
print()
print("-" * 45)
for lb in lookbacks:
    print(f"{lb:>5}d  ", end="")
    for hp in holding_periods:
        key = f'lb{lb}_hp{hp}'
        if key in momentum_results:
            s = momentum_results[key]['sharpe']
            marker = '**' if momentum_results[key]['p_value'] < 0.05 else '  '
            print(f"  {s:>5.2f}{marker[0]}", end="")
        else:
            print(f"  {'N/A':>6}", end="")
    print()

print("\nMEAN REVERSION SHARPE (buy losers, short winners):")
print(f"{hdr:<8}", end="")
for hp in holding_periods:
    print(f"  {hp:>5}d", end="")
print()
print("-" * 45)
for lb in lookbacks:
    print(f"{lb:>5}d  ", end="")
    for hp in holding_periods:
        key = f'lb{lb}_hp{hp}'
        if key in reversion_results:
            s = reversion_results[key]['sharpe']
            marker = '**' if reversion_results[key]['p_value'] < 0.05 else '  '
            print(f"  {s:>5.2f}{marker[0]}", end="")
        else:
            print(f"  {'N/A':>6}", end="")
    print()

results['momentum'] = momentum_results
results['mean_reversion'] = reversion_results

# ============================================================
# 2C. LAGGING-TO-LEADING TRANSITION PROBABILITY
# If sector is in bottom quintile, what's P(top quintile next period)?
# ============================================================
print("\n" + "="*80)
print("2C. LAGGING-TO-LEADING TRANSITION MATRIX")
print("="*80)

for horizon in [21, 63]:
    rolling_ret = daily_returns.rolling(horizon).sum().dropna()

    # Rank into terciles: Top, Middle, Bottom
    def tercile(row):
        n = len(row)
        ranked = row.rank()
        return pd.Series(
            ['Top' if r > n * 2/3 else ('Bottom' if r <= n * 1/3 else 'Middle') for r in ranked],
            index=row.index
        )

    terciles = rolling_ret.apply(tercile, axis=1)

    # Transition matrix: what tercile next period given current tercile?
    transition = {}
    for sector in SECTOR_ETFS:
        current = terciles[sector]
        future = terciles[sector].shift(-horizon)
        valid = current.dropna().index.intersection(future.dropna().index)

        for from_state in ['Top', 'Middle', 'Bottom']:
            for to_state in ['Top', 'Middle', 'Bottom']:
                mask = (current.loc[valid] == from_state) & (future.loc[valid] == to_state)
                from_mask = current.loc[valid] == from_state
                if from_mask.sum() > 0:
                    key = f'{from_state}->{to_state}'
                    if key not in transition:
                        transition[key] = []
                    transition[key].append(mask.sum() / from_mask.sum())

    print(f"\nTransition probabilities ({horizon}d horizon, averaged across sectors):")
    ft_hdr = 'From\\To'
    print(f"{ft_hdr:<15} {'Top':>8} {'Middle':>8} {'Bottom':>8}")
    print("-" * 42)
    for from_state in ['Top', 'Middle', 'Bottom']:
        print(f"{from_state:<15}", end="")
        for to_state in ['Top', 'Middle', 'Bottom']:
            key = f'{from_state}->{to_state}'
            if key in transition:
                prob = np.mean(transition[key])
                print(f" {prob:>7.1%}", end="")
            else:
                print(f" {'N/A':>7}", end="")
        print()

    results[f'transition_matrix_{horizon}d'] = {
        k: {'mean_prob': round(np.mean(v), 4), 'std_prob': round(np.std(v), 4)}
        for k, v in transition.items()
    }

# ============================================================
# 2D. SECTOR ROTATION SPEED — Auto-correlation of ranks
# ============================================================
print("\n" + "="*80)
print("2D. RANK AUTO-CORRELATION (leadership persistence)")
print("="*80)

for horizon in [21, 63, 126]:
    rolling_ret = daily_returns.rolling(horizon).sum().dropna()
    ranks = rolling_ret.rank(axis=1)

    # Rank auto-correlation at various lags
    print(f"\n{horizon}d rolling return rank autocorrelation:")
    print(f"{'Lag':<8}", end="")
    for sector in SECTOR_ETFS[:6]:  # Show first 6 for space
        print(f" {sector:>6}", end="")
    print(f" {'AVG':>6}")

    for lag in [5, 10, 21, 42, 63, 126]:
        print(f"{lag:>5}d  ", end="")
        autocorrs = []
        for sector in SECTOR_ETFS:
            corr = ranks[sector].corr(ranks[sector].shift(lag))
            autocorrs.append(corr)
            if sector in SECTOR_ETFS[:6]:
                print(f" {corr:>6.3f}", end="")
        print(f" {np.mean(autocorrs):>6.3f}")

    results[f'rank_autocorr_{horizon}d'] = {
        lag: round(np.mean([ranks[s].corr(ranks[s].shift(lag)) for s in SECTOR_ETFS]), 4)
        for lag in [5, 10, 21, 42, 63, 126]
    }

# Save
with open(os.path.join(OUT_DIR, 'rotation_timing_results.json'), 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n=== Results saved ===")
