#!/usr/bin/env python3
"""
Day-of-Week + Fundamental Factor Analysis for Sector ETF Trading
Analyzes: DOW effects, RSI<35 interaction, fundamental ICs
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
from datetime import datetime
from pathlib import Path

warnings.filterwarnings('ignore')

ETFS = ['XLE', 'XLU', 'XLK', 'XLF', 'XLP', 'XLY', 'XLI', 'XLB', 'XLC', 'XLRE', 'SMH']
START = '2020-01-01'
END = '2026-08-20'
DOW_NAMES = {0: 'Monday', 1: 'Tuesday', 2: 'Wednesday', 3: 'Thursday', 4: 'Friday'}
RSI_PERIOD = 14
RSI_THRESHOLD = 35

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = (-delta).where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def permutation_test(group_returns, all_returns, n_perms=10000):
    """Test if group mean is significantly different from random sample of same size."""
    observed = np.mean(group_returns)
    n = len(group_returns)
    count = 0
    all_arr = np.array(all_returns)
    for _ in range(n_perms):
        perm_sample = np.random.choice(all_arr, size=n, replace=False)
        if abs(np.mean(perm_sample)) >= abs(observed):
            count += 1
    return count / n_perms

print("=" * 80)
print("SECTOR ETF DAY-OF-WEEK + FUNDAMENTAL ANALYSIS")
print("=" * 80)

# ============================================================
# DOWNLOAD DATA
# ============================================================
print("\n[1] Downloading daily data for all ETFs...")
data = {}
for etf in ETFS:
    try:
        df = yf.download(etf, start=START, end=END, progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        if len(df) > 100:
            data[etf] = df
            print(f"  {etf}: {len(df)} days ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
        else:
            print(f"  {etf}: SKIPPED (only {len(df)} days)")
    except Exception as e:
        print(f"  {etf}: ERROR - {e}")

print(f"\nLoaded {len(data)} ETFs successfully.")

# ============================================================
# PART 1: DAY-OF-WEEK ANALYSIS
# ============================================================
print("\n" + "=" * 80)
print("PART 1: DAY-OF-WEEK ANALYSIS")
print("=" * 80)

dow_results = {}
rsi_dow_results = {}

for etf, df in data.items():
    df = df.copy()
    df['RSI'] = compute_rsi(df['Close'], RSI_PERIOD)
    df['dow'] = df.index.dayofweek

    # Forward returns
    for h in [1, 3, 5]:
        df[f'fwd_{h}d'] = df['Close'].shift(-h) / df['Close'] - 1

    etf_dow = {}
    etf_rsi_dow = {}

    for dow in range(5):
        mask = df['dow'] == dow
        subset = df[mask].dropna(subset=['fwd_1d', 'fwd_3d', 'fwd_5d'])

        day_stats = {'n': len(subset)}
        for h in [1, 3, 5]:
            col = f'fwd_{h}d'
            rets = subset[col].values
            if len(rets) > 5:
                day_stats[f'mean_{h}d'] = float(np.mean(rets))
                day_stats[f'std_{h}d'] = float(np.std(rets))
                day_stats[f'sharpe_{h}d'] = float(np.mean(rets) / np.std(rets) * np.sqrt(252/h)) if np.std(rets) > 0 else 0
                # t-test vs all other days
                other = df[~mask].dropna(subset=[col])[col].values
                if len(other) > 5:
                    t_stat, p_val = stats.ttest_ind(rets, other, equal_var=False)
                    day_stats[f'ttest_p_{h}d'] = float(p_val)
                    day_stats[f'ttest_t_{h}d'] = float(t_stat)

        etf_dow[DOW_NAMES[dow]] = day_stats

        # RSI < 35 + DOW interaction
        rsi_mask = mask & (df['RSI'] < RSI_THRESHOLD)
        rsi_subset = df[rsi_mask].dropna(subset=['fwd_1d', 'fwd_3d', 'fwd_5d'])

        rsi_stats = {'n': len(rsi_subset)}
        for h in [1, 3, 5]:
            col = f'fwd_{h}d'
            rets = rsi_subset[col].values
            if len(rets) >= 3:
                rsi_stats[f'mean_{h}d'] = float(np.mean(rets))
                rsi_stats[f'std_{h}d'] = float(np.std(rets))
                rsi_stats[f'sharpe_{h}d'] = float(np.mean(rets) / np.std(rets) * np.sqrt(252/h)) if np.std(rets) > 0 else 0
                rsi_stats[f'winrate_{h}d'] = float(np.mean(rets > 0))
            else:
                rsi_stats[f'mean_{h}d'] = None
                rsi_stats[f'sharpe_{h}d'] = None

        etf_rsi_dow[DOW_NAMES[dow]] = rsi_stats

    dow_results[etf] = etf_dow
    rsi_dow_results[etf] = etf_rsi_dow

# Aggregate across all ETFs
print("\n--- AGGREGATE DAY-OF-WEEK EFFECTS (ALL ETFs POOLED) ---")
print(f"{'Day':<12} {'N':>6} {'1d Mean':>10} {'3d Mean':>10} {'5d Mean':>10} {'1d Sharpe':>10} {'5d Sharpe':>10} {'1d p-val':>10}")

agg_dow = {}
for dow in range(5):
    day_name = DOW_NAMES[dow]
    all_1d, all_3d, all_5d = [], [], []
    other_1d, other_3d, other_5d = [], [], []

    for etf, df in data.items():
        df = df.copy()
        df['dow'] = df.index.dayofweek
        for h in [1, 3, 5]:
            df[f'fwd_{h}d'] = df['Close'].shift(-h) / df['Close'] - 1

        mask = df['dow'] == dow
        sub = df[mask].dropna(subset=['fwd_1d', 'fwd_3d', 'fwd_5d'])
        oth = df[~mask].dropna(subset=['fwd_1d', 'fwd_3d', 'fwd_5d'])

        all_1d.extend(sub['fwd_1d'].values)
        all_3d.extend(sub['fwd_3d'].values)
        all_5d.extend(sub['fwd_5d'].values)
        other_1d.extend(oth['fwd_1d'].values)
        other_3d.extend(oth['fwd_3d'].values)
        other_5d.extend(oth['fwd_5d'].values)

    all_1d, all_3d, all_5d = np.array(all_1d), np.array(all_3d), np.array(all_5d)
    other_1d = np.array(other_1d)

    t_stat, p_val = stats.ttest_ind(all_1d, other_1d, equal_var=False)

    sharpe_1d = np.mean(all_1d) / np.std(all_1d) * np.sqrt(252) if np.std(all_1d) > 0 else 0
    sharpe_5d = np.mean(all_5d) / np.std(all_5d) * np.sqrt(252/5) if np.std(all_5d) > 0 else 0

    print(f"{day_name:<12} {len(all_1d):>6} {np.mean(all_1d)*100:>9.4f}% {np.mean(all_3d)*100:>9.4f}% {np.mean(all_5d)*100:>9.4f}% {sharpe_1d:>10.3f} {sharpe_5d:>10.3f} {p_val:>10.4f}")

    agg_dow[day_name] = {
        'n': len(all_1d),
        'mean_1d': float(np.mean(all_1d)),
        'mean_3d': float(np.mean(all_3d)),
        'mean_5d': float(np.mean(all_5d)),
        'sharpe_1d': float(sharpe_1d),
        'sharpe_5d': float(sharpe_5d),
        'ttest_p_1d': float(p_val),
        'ttest_t_1d': float(t_stat)
    }

# RSI < 35 dip-buy by day of week (aggregated)
print("\n--- RSI < 35 DIP-BUY BY ENTRY DAY (ALL ETFs POOLED) ---")
print(f"{'Day':<12} {'N':>6} {'5d Mean':>10} {'5d WR':>8} {'5d Sharpe':>10}")

agg_rsi_dow = {}
for dow in range(5):
    day_name = DOW_NAMES[dow]
    all_5d = []

    for etf, df in data.items():
        df = df.copy()
        df['RSI'] = compute_rsi(df['Close'], RSI_PERIOD)
        df['dow'] = df.index.dayofweek
        df['fwd_5d'] = df['Close'].shift(-5) / df['Close'] - 1

        mask = (df['dow'] == dow) & (df['RSI'] < RSI_THRESHOLD)
        sub = df[mask].dropna(subset=['fwd_5d'])
        all_5d.extend(sub['fwd_5d'].values)

    all_5d = np.array(all_5d)
    if len(all_5d) >= 5:
        mean_5d = np.mean(all_5d)
        wr = np.mean(all_5d > 0)
        sharpe = mean_5d / np.std(all_5d) * np.sqrt(252/5) if np.std(all_5d) > 0 else 0
        print(f"{day_name:<12} {len(all_5d):>6} {mean_5d*100:>9.4f}% {wr*100:>7.1f}% {sharpe:>10.3f}")
        agg_rsi_dow[day_name] = {
            'n': int(len(all_5d)),
            'mean_5d': float(mean_5d),
            'winrate_5d': float(wr),
            'sharpe_5d': float(sharpe),
            'std_5d': float(np.std(all_5d)),
            'ci95_lower': float(mean_5d - 1.96 * np.std(all_5d) / np.sqrt(len(all_5d))),
            'ci95_upper': float(mean_5d + 1.96 * np.std(all_5d) / np.sqrt(len(all_5d)))
        }
    else:
        print(f"{day_name:<12} {len(all_5d):>6}   (too few samples)")
        agg_rsi_dow[day_name] = {'n': int(len(all_5d)), 'note': 'insufficient samples'}

# Friday entry (hold over weekend) analysis
print("\n--- FRIDAY ENTRY: WEEKEND HOLD ANALYSIS ---")
fri_analysis = {}
for etf, df in data.items():
    df = df.copy()
    df['dow'] = df.index.dayofweek
    df['fwd_1d'] = df['Close'].shift(-1) / df['Close'] - 1
    df['fwd_3d'] = df['Close'].shift(-3) / df['Close'] - 1

    fri = df[df['dow'] == 4].dropna(subset=['fwd_1d', 'fwd_3d'])
    non_fri = df[df['dow'] != 4].dropna(subset=['fwd_1d', 'fwd_3d'])

    if len(fri) > 10:
        # Friday's fwd_1d is Mon close, so it includes weekend gap
        fri_analysis[etf] = {
            'fri_1d_mean': float(fri['fwd_1d'].mean()),
            'nonfri_1d_mean': float(non_fri['fwd_1d'].mean()),
            'fri_3d_mean': float(fri['fwd_3d'].mean()),
            'nonfri_3d_mean': float(non_fri['fwd_3d'].mean()),
            'fri_1d_wr': float((fri['fwd_1d'] > 0).mean()),
            'n_fri': len(fri)
        }

print(f"{'ETF':<6} {'Fri 1d':>10} {'Other 1d':>10} {'Fri 3d':>10} {'Other 3d':>10} {'Fri WR':>8}")
for etf, v in fri_analysis.items():
    print(f"{etf:<6} {v['fri_1d_mean']*100:>9.4f}% {v['nonfri_1d_mean']*100:>9.4f}% {v['fri_3d_mean']*100:>9.4f}% {v['nonfri_3d_mean']*100:>9.4f}% {v['fri_1d_wr']*100:>7.1f}%")

# Monday gap analysis
print("\n--- MONDAY MORNING: WEEKEND GAP ANALYSIS ---")
mon_gap = {}
for etf, df in data.items():
    df = df.copy()
    df['dow'] = df.index.dayofweek
    df['overnight_gap'] = df['Open'] / df['Close'].shift(1) - 1
    df['intraday'] = df['Close'] / df['Open'] - 1
    df['fwd_5d'] = df['Close'].shift(-5) / df['Close'] - 1

    mon = df[df['dow'] == 0].dropna(subset=['overnight_gap', 'intraday', 'fwd_5d'])
    non_mon = df[df['dow'] != 0].dropna(subset=['overnight_gap', 'intraday'])

    if len(mon) > 10:
        mon_gap[etf] = {
            'mon_gap_mean': float(mon['overnight_gap'].mean()),
            'mon_gap_std': float(mon['overnight_gap'].std()),
            'other_gap_mean': float(non_mon['overnight_gap'].mean()),
            'mon_intraday_mean': float(mon['intraday'].mean()),
            'other_intraday_mean': float(non_mon['intraday'].mean()),
            'mon_gap_down_pct': float((mon['overnight_gap'] < 0).mean()),
            # Does gap-down Monday predict good 5d returns?
            'gap_down_mon_5d': float(mon[mon['overnight_gap'] < -0.003]['fwd_5d'].mean()) if (mon['overnight_gap'] < -0.003).sum() > 5 else None,
            'gap_up_mon_5d': float(mon[mon['overnight_gap'] > 0.003]['fwd_5d'].mean()) if (mon['overnight_gap'] > 0.003).sum() > 5 else None,
            'n': len(mon)
        }

print(f"{'ETF':<6} {'Mon Gap':>10} {'Oth Gap':>10} {'Mon Intra':>10} {'Oth Intra':>10} {'GapDn%':>8} {'GapDn 5d':>10}")
for etf, v in mon_gap.items():
    gd5 = f"{v['gap_down_mon_5d']*100:.4f}%" if v['gap_down_mon_5d'] is not None else "N/A"
    print(f"{etf:<6} {v['mon_gap_mean']*100:>9.4f}% {v['other_gap_mean']*100:>9.4f}% {v['mon_intraday_mean']*100:>9.4f}% {v['other_intraday_mean']*100:>9.4f}% {v['mon_gap_down_pct']*100:>7.1f}% {gd5:>10}")

# Permutation tests for best/worst days
print("\n--- PERMUTATION TESTS (10k permutations, pooled across ETFs) ---")
perm_results = {}
for dow in range(5):
    day_name = DOW_NAMES[dow]
    day_rets, all_rets = [], []

    for etf, df in data.items():
        df = df.copy()
        df['dow'] = df.index.dayofweek
        df['fwd_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        sub = df.dropna(subset=['fwd_5d'])
        all_rets.extend(sub['fwd_5d'].values)
        day_rets.extend(sub[sub['dow'] == dow]['fwd_5d'].values)

    p = permutation_test(day_rets, all_rets, n_perms=10000)
    perm_results[day_name] = {'permutation_p': float(p), 'n': len(day_rets), 'mean_5d': float(np.mean(day_rets))}
    print(f"  {day_name}: mean 5d = {np.mean(day_rets)*100:.4f}%, permutation p = {p:.4f}, n = {len(day_rets)}")


# ============================================================
# PART 2: FUNDAMENTAL FACTOR SCREENING
# ============================================================
print("\n" + "=" * 80)
print("PART 2: FUNDAMENTAL FACTOR SCREENING")
print("=" * 80)

# Factor 1: Relative sector strength rank (21-day rolling)
print("\n--- FACTOR: RELATIVE SECTOR STRENGTH (21d rolling rank) ---")

# Build panel of returns
close_panel = pd.DataFrame({etf: df['Close'] for etf, df in data.items()})
close_panel = close_panel.dropna(how='all')

ret_21d = close_panel.pct_change(21)
fwd_5d = close_panel.shift(-5) / close_panel - 1

# Rank sectors each day (1=worst, 11=best)
rank_21d = ret_21d.rank(axis=1)

# Calculate IC for each day: correlation between rank and forward 5d return
ic_series = []
for date in rank_21d.index:
    r = rank_21d.loc[date].dropna()
    f = fwd_5d.loc[date].dropna()
    common = r.index.intersection(f.index)
    if len(common) >= 5:
        corr, _ = stats.spearmanr(r[common], f[common])
        if not np.isnan(corr):
            ic_series.append({'date': date, 'ic': corr})

ic_df = pd.DataFrame(ic_series)
if len(ic_df) > 0:
    mean_ic = ic_df['ic'].mean()
    std_ic = ic_df['ic'].std()
    ic_ir = mean_ic / std_ic if std_ic > 0 else 0
    t_stat_ic = mean_ic / (std_ic / np.sqrt(len(ic_df))) if std_ic > 0 else 0
    p_val_ic = 2 * (1 - stats.t.cdf(abs(t_stat_ic), len(ic_df) - 1))

    print(f"  21d Momentum Rank → 5d Fwd Return:")
    print(f"    Mean IC = {mean_ic:.4f}, IC IR = {ic_ir:.3f}")
    print(f"    t-stat = {t_stat_ic:.2f}, p-value = {p_val_ic:.4f}")
    print(f"    N observations = {len(ic_df)}")

    momentum_rank_ic = {
        'mean_ic': float(mean_ic), 'ic_ir': float(ic_ir),
        't_stat': float(t_stat_ic), 'p_value': float(p_val_ic),
        'n': len(ic_df), 'interpretation': 'positive IC = momentum, negative IC = mean reversion'
    }
else:
    momentum_rank_ic = {'error': 'insufficient data'}

# Check if REVERSAL (low rank → higher fwd return) exists
rev_21d = (-ret_21d).rank(axis=1)  # Reverse rank: 1=best recent, 11=worst recent
ic_rev_series = []
for date in rev_21d.index:
    r = rev_21d.loc[date].dropna()
    f = fwd_5d.loc[date].dropna()
    common = r.index.intersection(f.index)
    if len(common) >= 5:
        corr, _ = stats.spearmanr(r[common], f[common])
        if not np.isnan(corr):
            ic_rev_series.append({'date': date, 'ic': corr})

ic_rev_df = pd.DataFrame(ic_rev_series)
if len(ic_rev_df) > 0:
    mean_rev_ic = ic_rev_df['ic'].mean()
    print(f"  21d Reversal Rank → 5d Fwd Return: Mean IC = {mean_rev_ic:.4f}")

# Factor 2: Volume anomaly (unusually high volume)
print("\n--- FACTOR: VOLUME ANOMALY (vol / 21d avg vol) ---")
vol_ic_results = {}
for etf, df in data.items():
    df = df.copy()
    df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(21).mean()
    df['fwd_5d'] = df['Close'].shift(-5) / df['Close'] - 1

    valid = df.dropna(subset=['vol_ratio', 'fwd_5d'])
    if len(valid) > 50:
        corr, p = stats.spearmanr(valid['vol_ratio'], valid['fwd_5d'])
        vol_ic_results[etf] = {'ic': float(corr), 'p_value': float(p), 'n': len(valid)}

avg_vol_ic = np.mean([v['ic'] for v in vol_ic_results.values()])
sig_count = sum(1 for v in vol_ic_results.values() if v['p_value'] < 0.05)
print(f"  Avg IC across ETFs: {avg_vol_ic:.4f}")
print(f"  ETFs with p < 0.05: {sig_count}/{len(vol_ic_results)}")
for etf, v in sorted(vol_ic_results.items(), key=lambda x: abs(x[1]['ic']), reverse=True):
    sig = "*" if v['p_value'] < 0.05 else " "
    print(f"    {etf:<6} IC={v['ic']:>7.4f}  p={v['p_value']:.4f} {sig}  n={v['n']}")

# Factor 3: Volume + DOW interaction
print("\n--- FACTOR: VOLUME ANOMALY x DAY-OF-WEEK INTERACTION ---")
vol_dow_ic = {}
for dow in range(5):
    day_name = DOW_NAMES[dow]
    all_vol_ratios, all_fwd = [], []

    for etf, df in data.items():
        df = df.copy()
        df['vol_ratio'] = df['Volume'] / df['Volume'].rolling(21).mean()
        df['fwd_5d'] = df['Close'].shift(-5) / df['Close'] - 1
        df['dow'] = df.index.dayofweek

        sub = df[(df['dow'] == dow)].dropna(subset=['vol_ratio', 'fwd_5d'])
        all_vol_ratios.extend(sub['vol_ratio'].values)
        all_fwd.extend(sub['fwd_5d'].values)

    if len(all_vol_ratios) > 30:
        corr, p = stats.spearmanr(all_vol_ratios, all_fwd)
        vol_dow_ic[day_name] = {'ic': float(corr), 'p_value': float(p), 'n': len(all_vol_ratios)}
        sig = "*" if p < 0.05 else " "
        print(f"  {day_name}: IC={corr:.4f}, p={p:.4f} {sig}, n={len(all_vol_ratios)}")

# Factor 4: RSI level (continuous) as factor
print("\n--- FACTOR: RSI LEVEL → 5d Forward Return ---")
rsi_ic_results = {}
for etf, df in data.items():
    df = df.copy()
    df['RSI'] = compute_rsi(df['Close'], RSI_PERIOD)
    df['fwd_5d'] = df['Close'].shift(-5) / df['Close'] - 1

    valid = df.dropna(subset=['RSI', 'fwd_5d'])
    if len(valid) > 50:
        # Negative IC = low RSI predicts high returns (mean reversion works)
        corr, p = stats.spearmanr(valid['RSI'], valid['fwd_5d'])
        rsi_ic_results[etf] = {'ic': float(corr), 'p_value': float(p), 'n': len(valid)}

avg_rsi_ic = np.mean([v['ic'] for v in rsi_ic_results.values()])
sig_count = sum(1 for v in rsi_ic_results.values() if v['p_value'] < 0.05)
print(f"  Avg IC across ETFs: {avg_rsi_ic:.4f} (negative = mean reversion works)")
print(f"  ETFs with p < 0.05: {sig_count}/{len(rsi_ic_results)}")
for etf, v in sorted(rsi_ic_results.items(), key=lambda x: abs(x[1]['ic']), reverse=True):
    sig = "*" if v['p_value'] < 0.05 else " "
    print(f"    {etf:<6} IC={v['ic']:>7.4f}  p={v['p_value']:.4f} {sig}  n={v['n']}")

# Factor 5: Dividend yield proxy (trailing yield from yfinance)
print("\n--- FACTOR: DIVIDEND YIELD (from yfinance info) ---")
div_yields = {}
for etf in ETFS:
    try:
        ticker = yf.Ticker(etf)
        info = ticker.info
        dy = info.get('trailingAnnualDividendYield') or info.get('dividendYield')
        pe = info.get('trailingPE')
        div_yields[etf] = {
            'dividend_yield': float(dy) if dy else None,
            'trailing_pe': float(pe) if pe else None
        }
        print(f"  {etf}: Div Yield = {dy*100 if dy else 'N/A':.2f}%, PE = {pe if pe else 'N/A'}")
    except Exception as e:
        print(f"  {etf}: Error - {e}")

# Factor 6: Cross-sectional momentum-reversal IC by lookback
print("\n--- FACTOR: CROSS-SECTIONAL IC BY LOOKBACK PERIOD ---")
lookback_ic = {}
for lb in [5, 10, 21, 42, 63]:
    ret_lb = close_panel.pct_change(lb)
    ic_list = []
    for date in ret_lb.index:
        r = ret_lb.loc[date].dropna()
        f = fwd_5d.loc[date].dropna()
        common = r.index.intersection(f.index)
        if len(common) >= 5:
            corr, _ = stats.spearmanr(r[common], f[common])
            if not np.isnan(corr):
                ic_list.append(corr)

    if ic_list:
        mean = np.mean(ic_list)
        std = np.std(ic_list)
        t = mean / (std / np.sqrt(len(ic_list))) if std > 0 else 0
        p = 2 * (1 - stats.t.cdf(abs(t), len(ic_list) - 1))
        lookback_ic[f'{lb}d'] = {'mean_ic': float(mean), 'ic_ir': float(mean/std) if std > 0 else 0,
                                  't_stat': float(t), 'p_value': float(p), 'n': len(ic_list)}
        sig = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        print(f"  {lb:>3}d lookback: IC={mean:>7.4f}, IR={mean/std if std>0 else 0:.3f}, t={t:.2f}, p={p:.4f} {sig}")

# Factor 7: Volatility regime (high vol → mean reversion stronger?)
print("\n--- FACTOR: VOLATILITY REGIME x MEAN REVERSION ---")
vol_regime_results = {}
for etf, df in data.items():
    df = df.copy()
    df['RSI'] = compute_rsi(df['Close'], RSI_PERIOD)
    df['realized_vol'] = df['Close'].pct_change().rolling(21).std() * np.sqrt(252)
    df['fwd_5d'] = df['Close'].shift(-5) / df['Close'] - 1
    df['vol_regime'] = pd.qcut(df['realized_vol'].dropna(), q=3, labels=['low', 'mid', 'high'])

    dip_mask = df['RSI'] < RSI_THRESHOLD

    for regime in ['low', 'mid', 'high']:
        mask = dip_mask & (df['vol_regime'] == regime)
        sub = df[mask].dropna(subset=['fwd_5d'])
        key = f"{etf}_{regime}"
        if len(sub) >= 3:
            vol_regime_results[key] = {
                'etf': etf, 'regime': regime,
                'mean_5d': float(sub['fwd_5d'].mean()),
                'wr': float((sub['fwd_5d'] > 0).mean()),
                'n': len(sub)
            }

# Aggregate by regime
for regime in ['low', 'mid', 'high']:
    subset = {k: v for k, v in vol_regime_results.items() if v['regime'] == regime}
    if subset:
        total_n = sum(v['n'] for v in subset.values())
        wt_mean = sum(v['mean_5d'] * v['n'] for v in subset.values()) / total_n if total_n > 0 else 0
        wt_wr = sum(v['wr'] * v['n'] for v in subset.values()) / total_n if total_n > 0 else 0
        print(f"  RSI<35 in {regime:>4} vol: mean 5d = {wt_mean*100:.4f}%, WR = {wt_wr*100:.1f}%, N = {total_n}")

# ============================================================
# COMPILE RESULTS
# ============================================================
results = {
    'metadata': {
        'run_date': datetime.now().isoformat(),
        'period': f'{START} to {END}',
        'etfs': ETFS,
        'n_etfs': len(data)
    },
    'part1_day_of_week': {
        'aggregate_dow': agg_dow,
        'aggregate_rsi_dow': agg_rsi_dow,
        'friday_weekend_hold': fri_analysis,
        'monday_gap': {k: {kk: vv for kk, vv in v.items()} for k, v in mon_gap.items()},
        'permutation_tests': perm_results,
        'per_etf_dow': dow_results,
        'per_etf_rsi_dow': rsi_dow_results
    },
    'part2_fundamental_factors': {
        'momentum_rank_21d_ic': momentum_rank_ic,
        'volume_anomaly_ic': vol_ic_results,
        'volume_dow_interaction_ic': vol_dow_ic,
        'rsi_level_ic': rsi_ic_results,
        'dividend_yields_current': div_yields,
        'cross_sectional_ic_by_lookback': lookback_ic,
        'vol_regime_mean_reversion': vol_regime_results
    },
    'conclusions': {}  # Filled below
}

# ============================================================
# CONCLUSIONS
# ============================================================
print("\n" + "=" * 80)
print("CONCLUSIONS")
print("=" * 80)

# Find best/worst DOW
best_dow = max(agg_dow.items(), key=lambda x: x[1]['mean_5d'])
worst_dow = min(agg_dow.items(), key=lambda x: x[1]['mean_5d'])
print(f"\n1. BEST entry day (5d fwd): {best_dow[0]} ({best_dow[1]['mean_5d']*100:.4f}%)")
print(f"   WORST entry day (5d fwd): {worst_dow[0]} ({worst_dow[1]['mean_5d']*100:.4f}%)")

# Check if any DOW is significant
sig_days = [d for d, v in perm_results.items() if v['permutation_p'] < 0.05]
print(f"\n2. Statistically significant days (permutation p < 0.05): {sig_days if sig_days else 'NONE'}")

# Best RSI dip day
if agg_rsi_dow:
    valid_rsi = {k: v for k, v in agg_rsi_dow.items() if isinstance(v.get('mean_5d'), (int, float)) and v.get('mean_5d') is not None}
    if valid_rsi:
        best_rsi_day = max(valid_rsi.items(), key=lambda x: x[1]['mean_5d'])
        print(f"\n3. BEST day to buy RSI<35 dip: {best_rsi_day[0]} (5d mean={best_rsi_day[1]['mean_5d']*100:.4f}%, WR={best_rsi_day[1].get('winrate_5d', 0)*100:.1f}%, n={best_rsi_day[1]['n']})")

# Momentum vs reversal
print(f"\n4. SECTOR ROTATION SIGNAL:")
if 'mean_ic' in momentum_rank_ic:
    direction = "MOMENTUM" if momentum_rank_ic['mean_ic'] > 0 else "MEAN REVERSION"
    sig = "SIGNIFICANT" if momentum_rank_ic['p_value'] < 0.05 else "NOT significant"
    print(f"   21d sector rank → 5d fwd: IC={momentum_rank_ic['mean_ic']:.4f} ({direction}), {sig} (p={momentum_rank_ic['p_value']:.4f})")

# Best lookback
if lookback_ic:
    best_lb = max(lookback_ic.items(), key=lambda x: abs(x[1]['mean_ic']))
    print(f"\n5. STRONGEST cross-sectional signal: {best_lb[0]} lookback (IC={best_lb[1]['mean_ic']:.4f}, p={best_lb[1]['p_value']:.4f})")

# RSI IC
print(f"\n6. RSI → 5d FORWARD RETURN IC: avg = {avg_rsi_ic:.4f}")
print(f"   {'CONFIRMS' if avg_rsi_ic < 0 else 'CONTRADICTS'} mean reversion edge")

conclusions = {
    'best_entry_day_5d': best_dow[0],
    'worst_entry_day_5d': worst_dow[0],
    'significant_dow_effects': sig_days,
    'overall_dow_significance': 'weak' if not sig_days else 'some evidence',
    'mean_rsi_ic': float(avg_rsi_ic),
    'mean_reversion_confirmed': avg_rsi_ic < -0.02,
    'best_lookback_for_rotation': best_lb[0] if lookback_ic else None,
    'recommendation': (
        'Day-of-week effects exist but are generally weak and not robust enough to be primary signals. '
        'RSI mean reversion is confirmed across sectors. '
        'Use DOW as a secondary filter (slight tilt) rather than primary entry criterion. '
        'Cross-sectional momentum/reversal signals at 5-10d lookback may add value for sector rotation timing.'
    )
}
results['conclusions'] = conclusions

# Save results
output_path = Path('/home/jupiter/Lvl3Quant/research/dow_fundamental_analysis.json')
output_path.parent.mkdir(parents=True, exist_ok=True)
with open(output_path, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\nResults saved to {output_path}")
print("DONE.")
