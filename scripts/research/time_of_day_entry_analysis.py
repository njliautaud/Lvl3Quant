#!/usr/bin/env python3
"""
Time-of-Day Entry Analysis for Sector ETF Options Trades
=========================================================
Analyzes which entry hours produce the best risk-adjusted forward returns
for our sector ETF universe with 2-5 day holds.

Uses intraday 1h bars from yfinance (max ~730 days for hourly),
plus daily OHLCV for longer history (2020-2026).

Author: Claude Opus 4.6
Date: 2026-08-21
"""

import json
import warnings
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────
TICKERS = ['XLE', 'XLU', 'XLK', 'XLF', 'XLP', 'XLY', 'XLI', 'XLB', 'XLC', 'XLRE', 'SMH']
DAILY_START = '2020-01-01'
DAILY_END = '2026-08-21'
# yfinance allows ~730 days of hourly data
INTRADAY_PERIOD = '730d'
FORWARD_DAYS = [1, 3, 5]
RSI_PERIOD = 14
RSI_THRESHOLD = 35
ENTRY_HOURS = [9, 10, 11, 12, 13, 14, 15]  # ET hours
N_PERMUTATIONS = 1000
ANNUALIZATION = 252

OUTPUT_PATH = Path('/home/jupiter/Lvl3Quant/research/time_of_day_analysis.json')


def compute_rsi(series, period=14):
    """Standard RSI calculation."""
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def sharpe(returns, ann=252):
    """Annualized Sharpe ratio."""
    if len(returns) < 5 or returns.std() == 0:
        return 0.0
    return float(returns.mean() / returns.std() * np.sqrt(ann))


def sortino(returns, ann=252):
    """Annualized Sortino ratio."""
    if len(returns) < 5:
        return 0.0
    downside = returns[returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return float('inf') if returns.mean() > 0 else 0.0
    return float(returns.mean() / downside.std() * np.sqrt(ann))


def profit_factor(returns):
    """Gross profits / gross losses."""
    gross_profit = returns[returns > 0].sum()
    gross_loss = abs(returns[returns < 0].sum())
    if gross_loss == 0:
        return float('inf') if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def win_rate(returns):
    """Fraction of positive returns."""
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def permutation_test(returns_a, returns_b, n_perm=1000):
    """Two-sample permutation test for difference in means."""
    if len(returns_a) < 5 or len(returns_b) < 5:
        return 1.0
    observed_diff = returns_a.mean() - returns_b.mean()
    combined = np.concatenate([returns_a.values, returns_b.values])
    n_a = len(returns_a)
    count = 0
    rng = np.random.RandomState(42)
    for _ in range(n_perm):
        perm = rng.permutation(combined)
        perm_diff = perm[:n_a].mean() - perm[n_a:].mean()
        if abs(perm_diff) >= abs(observed_diff):
            count += 1
    return count / n_perm


# ── 1. Fetch Data ─────────────────────────────────────────────────────
print("=" * 80)
print("TIME-OF-DAY ENTRY ANALYSIS FOR SECTOR ETF OPTIONS")
print("=" * 80)

print("\n[1/5] Fetching intraday (1h) data...")
intraday_data = {}
for ticker in TICKERS:
    try:
        t = yf.Ticker(ticker)
        df = t.history(period=INTRADAY_PERIOD, interval='1h')
        if len(df) > 100:
            # Ensure timezone-aware → convert to ET
            if df.index.tz is not None:
                df.index = df.index.tz_convert('US/Eastern')
            else:
                df.index = df.index.tz_localize('UTC').tz_convert('US/Eastern')
            intraday_data[ticker] = df
            print(f"  {ticker}: {len(df)} hourly bars, {df.index[0].date()} to {df.index[-1].date()}")
        else:
            print(f"  {ticker}: insufficient intraday data ({len(df)} bars)")
    except Exception as e:
        print(f"  {ticker}: error fetching intraday - {e}")

print("\n[2/5] Fetching daily OHLCV data (2020-2026)...")
daily_data = {}
for ticker in TICKERS:
    try:
        t = yf.Ticker(ticker)
        df = t.history(start=DAILY_START, end=DAILY_END, interval='1d')
        if len(df) > 100:
            daily_data[ticker] = df
            print(f"  {ticker}: {len(df)} daily bars")
        else:
            print(f"  {ticker}: insufficient daily data ({len(df)} bars)")
    except Exception as e:
        print(f"  {ticker}: error fetching daily - {e}")


# ── 2. Intraday Hour-of-Day Analysis ──────────────────────────────────
print("\n[3/5] Analyzing intraday hour-of-day patterns...")

hourly_results = {}  # {ticker: {hour: {fwd_days: metrics}}}
all_hourly_returns = {h: {fd: [] for fd in FORWARD_DAYS} for h in ENTRY_HOURS}

for ticker, df in intraday_data.items():
    hourly_results[ticker] = {}

    # Get hourly close prices, extract hour
    df['hour'] = df.index.hour
    df['date'] = df.index.date

    # Build daily close series from intraday for forward returns
    daily_closes = df.groupby('date')['Close'].last()

    for hour in ENTRY_HOURS:
        hour_bars = df[df['hour'] == hour].copy()
        if len(hour_bars) < 20:
            continue

        # For each entry at this hour, compute forward returns
        fwd_returns = {}
        for fd in FORWARD_DAYS:
            rets = []
            for idx, row in hour_bars.iterrows():
                entry_date = idx.date()
                entry_price = row['Close']

                # Find close price fd trading days later
                future_dates = daily_closes.index[daily_closes.index > entry_date]
                if len(future_dates) >= fd:
                    exit_date = future_dates[fd - 1]
                    exit_price = daily_closes.loc[exit_date]
                    ret = (exit_price - entry_price) / entry_price
                    rets.append(ret)

            if len(rets) >= 10:
                rets_arr = pd.Series(rets)
                fwd_returns[f'{fd}d'] = {
                    'mean_return_bps': round(float(rets_arr.mean() * 10000), 2),
                    'sharpe': round(sharpe(rets_arr, ann=ANNUALIZATION / fd), 3),
                    'sortino': round(sortino(rets_arr, ann=ANNUALIZATION / fd), 3),
                    'win_rate': round(win_rate(rets_arr), 3),
                    'profit_factor': round(profit_factor(rets_arr), 3),
                    'n_trades': len(rets_arr),
                    'std_bps': round(float(rets_arr.std() * 10000), 2),
                }
                all_hourly_returns[hour][fd].extend(rets)

        if fwd_returns:
            hourly_results[ticker][str(hour)] = fwd_returns

# Aggregate across all tickers
print("\n  AGGREGATE HOURLY ENTRY ANALYSIS (all tickers pooled)")
print("  " + "-" * 76)
print(f"  {'Hour':>6} | {'Fwd':>4} | {'Mean(bps)':>10} | {'Sharpe':>7} | {'Sortino':>8} | {'WR':>6} | {'PF':>6} | {'N':>6}")
print("  " + "-" * 76)

agg_hourly_stats = {}
for hour in ENTRY_HOURS:
    agg_hourly_stats[str(hour)] = {}
    for fd in FORWARD_DAYS:
        rets = pd.Series(all_hourly_returns[hour][fd])
        if len(rets) < 20:
            continue
        s = sharpe(rets, ann=ANNUALIZATION / fd)
        so = sortino(rets, ann=ANNUALIZATION / fd)
        wr = win_rate(rets)
        pf = profit_factor(rets)
        mean_bps = rets.mean() * 10000

        agg_hourly_stats[str(hour)][f'{fd}d'] = {
            'mean_return_bps': round(float(mean_bps), 2),
            'sharpe': round(s, 3),
            'sortino': round(so, 3),
            'win_rate': round(wr, 3),
            'profit_factor': round(pf, 3),
            'n_trades': len(rets),
        }

        hour_label = f"{hour}:00"
        if hour == 9:
            hour_label = "9:30"  # market open
        print(f"  {hour_label:>6} | {fd}d   | {mean_bps:>10.2f} | {s:>7.3f} | {so:>8.3f} | {wr:>5.1%} | {pf:>6.2f} | {len(rets):>6}")


# ── 3. Daily-Based Analysis (Longer History) ──────────────────────────
print("\n\n[4/5] Daily-based analysis (2020-2026, longer history)...")

# Open-to-close returns and overnight returns
daily_patterns = {}
all_otc_returns = []
all_overnight_returns = []
all_daily_rsi_entries = {fd: [] for fd in FORWARD_DAYS}
all_daily_uncond = {fd: [] for fd in FORWARD_DAYS}

# Day-of-week analysis
dow_returns = {dow: {fd: [] for fd in FORWARD_DAYS} for dow in range(5)}  # 0=Mon, 4=Fri
dow_rsi_returns = {dow: {fd: [] for fd in FORWARD_DAYS} for dow in range(5)}

for ticker, df in daily_data.items():
    df = df.copy()
    df['rsi'] = compute_rsi(df['Close'], RSI_PERIOD)
    df['otc_return'] = (df['Close'] - df['Open']) / df['Open']
    df['overnight_return'] = (df['Open'] - df['Close'].shift(1)) / df['Close'].shift(1)
    df['dow'] = df.index.dayofweek

    daily_patterns[ticker] = {
        'mean_otc_bps': round(float(df['otc_return'].mean() * 10000), 2),
        'mean_overnight_bps': round(float(df['overnight_return'].mean() * 10000), 2),
        'otc_sharpe': round(sharpe(df['otc_return'].dropna()), 3),
    }

    all_otc_returns.extend(df['otc_return'].dropna().tolist())
    all_overnight_returns.extend(df['overnight_return'].dropna().tolist())

    # Forward returns from open (morning entry)
    for fd in FORWARD_DAYS:
        # Forward return from today's open to fd-day-later close
        df[f'fwd_{fd}d'] = df['Close'].shift(-fd) / df['Open'] - 1

        valid = df[f'fwd_{fd}d'].dropna()
        all_daily_uncond[fd].extend(valid.tolist())

        # RSI < 35 entries
        rsi_mask = df['rsi'] < RSI_THRESHOLD
        rsi_entries = df.loc[rsi_mask, f'fwd_{fd}d'].dropna()
        all_daily_rsi_entries[fd].extend(rsi_entries.tolist())

        # Day-of-week
        for dow in range(5):
            dow_mask = df['dow'] == dow
            dow_rets = df.loc[dow_mask, f'fwd_{fd}d'].dropna()
            dow_returns[dow][fd].extend(dow_rets.tolist())

            rsi_dow_rets = df.loc[dow_mask & rsi_mask, f'fwd_{fd}d'].dropna()
            dow_rsi_returns[dow][fd].extend(rsi_dow_rets.tolist())


# ── Print daily pattern summary ───────────────────────────────────────
print("\n  OPEN-TO-CLOSE vs OVERNIGHT RETURNS (daily, 2020-2026)")
print("  " + "-" * 50)
for ticker in sorted(daily_patterns.keys()):
    p = daily_patterns[ticker]
    print(f"  {ticker:>5}: OTC={p['mean_otc_bps']:>6.1f} bps  Overnight={p['mean_overnight_bps']:>6.1f} bps  OTC Sharpe={p['otc_sharpe']:>6.3f}")

otc_arr = pd.Series(all_otc_returns)
ovn_arr = pd.Series(all_overnight_returns)
print(f"\n  AGGREGATE: OTC={otc_arr.mean()*10000:.1f} bps (Sharpe {sharpe(otc_arr):.3f})")
print(f"  AGGREGATE: Overnight={ovn_arr.mean()*10000:.1f} bps (Sharpe {sharpe(ovn_arr):.3f})")

# ── RSI < 35 conditional returns ──────────────────────────────────────
print("\n\n  RSI < 35 ENTRY FILTER vs UNCONDITIONAL (all tickers, daily 2020-2026)")
print("  " + "-" * 80)
print(f"  {'Fwd':>4} | {'Uncond Mean(bps)':>16} | {'RSI<35 Mean(bps)':>16} | {'RSI<35 Sharpe':>13} | {'RSI<35 WR':>10} | {'N_rsi':>6} | {'t-test p':>9}")
print("  " + "-" * 80)

rsi_stats = {}
for fd in FORWARD_DAYS:
    uncond = pd.Series(all_daily_uncond[fd])
    rsi_rets = pd.Series(all_daily_rsi_entries[fd])

    if len(rsi_rets) < 5:
        print(f"  {fd}d   | {uncond.mean()*10000:>16.2f} | {'insufficient data':>16} |")
        continue

    t_stat, p_val = stats.ttest_ind(rsi_rets, uncond, equal_var=False)
    s = sharpe(rsi_rets, ann=ANNUALIZATION / fd)
    wr = win_rate(rsi_rets)

    rsi_stats[f'{fd}d'] = {
        'unconditional_mean_bps': round(float(uncond.mean() * 10000), 2),
        'rsi35_mean_bps': round(float(rsi_rets.mean() * 10000), 2),
        'rsi35_sharpe': round(s, 3),
        'rsi35_sortino': round(sortino(rsi_rets, ann=ANNUALIZATION / fd), 3),
        'rsi35_win_rate': round(wr, 3),
        'rsi35_profit_factor': round(profit_factor(rsi_rets), 3),
        'n_rsi_entries': len(rsi_rets),
        'n_unconditional': len(uncond),
        'ttest_pval': round(float(p_val), 4),
    }

    print(f"  {fd}d   | {uncond.mean()*10000:>16.2f} | {rsi_rets.mean()*10000:>16.2f} | {s:>13.3f} | {wr:>9.1%} | {len(rsi_rets):>6} | {p_val:>9.4f}")


# ── Day-of-week analysis ─────────────────────────────────────────────
print("\n\n  DAY-OF-WEEK ENTRY ANALYSIS (unconditional, all tickers)")
print("  " + "-" * 80)
dow_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday']
print(f"  {'Day':>12} | {'Fwd':>4} | {'Mean(bps)':>10} | {'Sharpe':>7} | {'WR':>6} | {'PF':>6} | {'N':>6}")
print("  " + "-" * 80)

dow_stats = {}
for dow in range(5):
    dow_stats[dow_names[dow]] = {}
    for fd in FORWARD_DAYS:
        rets = pd.Series(dow_returns[dow][fd])
        if len(rets) < 20:
            continue
        s = sharpe(rets, ann=ANNUALIZATION / fd)
        wr_val = win_rate(rets)
        pf_val = profit_factor(rets)
        mean_bps = rets.mean() * 10000

        dow_stats[dow_names[dow]][f'{fd}d'] = {
            'mean_return_bps': round(float(mean_bps), 2),
            'sharpe': round(s, 3),
            'win_rate': round(wr_val, 3),
            'profit_factor': round(pf_val, 3),
            'n_trades': len(rets),
        }

        print(f"  {dow_names[dow]:>12} | {fd}d   | {mean_bps:>10.2f} | {s:>7.3f} | {wr_val:>5.1%} | {pf_val:>6.2f} | {len(rets):>6}")

# Day-of-week with RSI < 35
print("\n\n  DAY-OF-WEEK + RSI < 35 ENTRY (all tickers)")
print("  " + "-" * 80)
print(f"  {'Day':>12} | {'Fwd':>4} | {'Mean(bps)':>10} | {'Sharpe':>7} | {'WR':>6} | {'N':>6}")
print("  " + "-" * 80)

dow_rsi_stats = {}
for dow in range(5):
    dow_rsi_stats[dow_names[dow]] = {}
    for fd in FORWARD_DAYS:
        rets = pd.Series(dow_rsi_returns[dow][fd])
        if len(rets) < 10:
            continue
        s = sharpe(rets, ann=ANNUALIZATION / fd)
        wr_val = win_rate(rets)
        mean_bps = rets.mean() * 10000

        dow_rsi_stats[dow_names[dow]][f'{fd}d'] = {
            'mean_return_bps': round(float(mean_bps), 2),
            'sharpe': round(s, 3),
            'win_rate': round(wr_val, 3),
            'n_trades': len(rets),
        }

        print(f"  {dow_names[dow]:>12} | {fd}d   | {mean_bps:>10.2f} | {s:>7.3f} | {wr_val:>5.1%} | {len(rets):>6}")


# ── 4. Morning Dip Analysis ──────────────────────────────────────────
print("\n\n[5/5] Morning dip analysis (intraday)...")

# Check if intraday open-to-10AM typically shows a dip
morning_dip_stats = {}
for ticker, df in intraday_data.items():
    df['hour'] = df.index.hour
    df['date'] = df.index.date

    # Get 9:30 open price and 10:00 price
    dates = df['date'].unique()
    dips = []
    recoveries = []

    for date in dates:
        day_data = df[df['date'] == date]
        open_bar = day_data[day_data['hour'] == 9]
        ten_bar = day_data[day_data['hour'] == 10]
        close_bar = day_data[day_data['hour'] == 15]

        if len(open_bar) == 0 or len(ten_bar) == 0 or len(close_bar) == 0:
            continue

        open_price = open_bar.iloc[0]['Open']
        ten_price = ten_bar.iloc[0]['Close']
        close_price = close_bar.iloc[-1]['Close']

        morning_move = (ten_price - open_price) / open_price
        dips.append(morning_move)

        # Recovery: 10AM to close
        if morning_move < 0:  # actual dip day
            recovery = (close_price - ten_price) / ten_price
            recoveries.append(recovery)

    if len(dips) > 20:
        dips_arr = pd.Series(dips)
        dip_freq = (dips_arr < 0).mean()
        morning_dip_stats[ticker] = {
            'dip_frequency': round(float(dip_freq), 3),
            'mean_morning_move_bps': round(float(dips_arr.mean() * 10000), 2),
            'mean_recovery_bps': round(float(pd.Series(recoveries).mean() * 10000), 2) if recoveries else 0,
            'recovery_win_rate': round(float((pd.Series(recoveries) > 0).mean()), 3) if recoveries else 0,
            'n_dip_days': len(recoveries),
        }

print("\n  MORNING DIP PATTERN (9:30 → 10:00 move, then 10:00 → close recovery)")
print("  " + "-" * 80)
print(f"  {'ETF':>5} | {'Dip Freq':>9} | {'Avg 9:30→10 (bps)':>18} | {'Avg Recovery (bps)':>18} | {'Recovery WR':>11} | {'N_dips':>6}")
print("  " + "-" * 80)

for ticker in sorted(morning_dip_stats.keys()):
    s = morning_dip_stats[ticker]
    print(f"  {ticker:>5} | {s['dip_frequency']:>8.1%} | {s['mean_morning_move_bps']:>18.1f} | {s['mean_recovery_bps']:>18.1f} | {s['recovery_win_rate']:>10.1%} | {s['n_dip_days']:>6}")


# ── 5. Afternoon vs Morning Statistical Test ─────────────────────────
print("\n\n  MORNING vs AFTERNOON ENTRY (permutation test, 3d forward returns)")
print("  " + "-" * 60)

morning_3d = []
afternoon_3d = []
for h in [9, 10]:
    morning_3d.extend(all_hourly_returns[h][3])
for h in [13, 14]:
    afternoon_3d.extend(all_hourly_returns[h][3])

morning_3d = pd.Series(morning_3d)
afternoon_3d = pd.Series(afternoon_3d)

if len(morning_3d) > 20 and len(afternoon_3d) > 20:
    perm_p = permutation_test(afternoon_3d, morning_3d, N_PERMUTATIONS)
    print(f"  Morning (9-10 AM)  : mean={morning_3d.mean()*10000:.1f} bps, Sharpe={sharpe(morning_3d, ANNUALIZATION/3):.3f}, N={len(morning_3d)}")
    print(f"  Afternoon (1-2 PM) : mean={afternoon_3d.mean()*10000:.1f} bps, Sharpe={sharpe(afternoon_3d, ANNUALIZATION/3):.3f}, N={len(afternoon_3d)}")
    print(f"  Diff (PM - AM)     : {(afternoon_3d.mean() - morning_3d.mean())*10000:.1f} bps")
    print(f"  Permutation p-value: {perm_p:.4f} {'*' if perm_p < 0.05 else '(not significant)'}")

    am_vs_pm = {
        'morning_mean_bps': round(float(morning_3d.mean() * 10000), 2),
        'morning_sharpe': round(sharpe(morning_3d, ANNUALIZATION/3), 3),
        'afternoon_mean_bps': round(float(afternoon_3d.mean() * 10000), 2),
        'afternoon_sharpe': round(sharpe(afternoon_3d, ANNUALIZATION/3), 3),
        'diff_bps': round(float((afternoon_3d.mean() - morning_3d.mean()) * 10000), 2),
        'permutation_pval': round(perm_p, 4),
        'significant_at_5pct': perm_p < 0.05,
    }
else:
    am_vs_pm = {'error': 'insufficient data'}
    print("  Insufficient data for comparison")


# ── 6. Best Entry Hour Ranking ────────────────────────────────────────
print("\n\n" + "=" * 80)
print("  BEST ENTRY HOUR RANKING (3-day forward, Sharpe-sorted)")
print("=" * 80)

hour_ranking = []
for hour in ENTRY_HOURS:
    rets = pd.Series(all_hourly_returns[hour][3])
    if len(rets) < 30:
        continue
    s = sharpe(rets, ann=ANNUALIZATION / 3)
    hour_ranking.append({
        'hour': hour,
        'label': f"{hour}:00" if hour != 9 else "9:30",
        'sharpe_3d': round(s, 3),
        'mean_3d_bps': round(float(rets.mean() * 10000), 2),
        'win_rate_3d': round(win_rate(rets), 3),
        'sortino_3d': round(sortino(rets, ann=ANNUALIZATION / 3), 3),
        'n': len(rets),
    })

hour_ranking.sort(key=lambda x: x['sharpe_3d'], reverse=True)

print(f"\n  {'Rank':>4} | {'Hour':>6} | {'Sharpe':>7} | {'Sortino':>8} | {'Mean(bps)':>10} | {'WR':>6} | {'N':>6}")
print("  " + "-" * 60)
for i, h in enumerate(hour_ranking):
    marker = " <<<" if h['hour'] in [9, 13] else ""  # mark our current windows
    print(f"  {i+1:>4} | {h['label']:>6} | {h['sharpe_3d']:>7.3f} | {h['sortino_3d']:>8.3f} | {h['mean_3d_bps']:>10.2f} | {h['win_rate_3d']:>5.1%} | {h['n']:>6}{marker}")

if hour_ranking:
    best = hour_ranking[0]
    print(f"\n  >>> BEST ENTRY HOUR: {best['label']} ET (Sharpe={best['sharpe_3d']:.3f}, Mean={best['mean_3d_bps']:.1f} bps)")


# ── 7. Summary & Recommendations ─────────────────────────────────────
print("\n\n" + "=" * 80)
print("  KEY FINDINGS")
print("=" * 80)

# Determine morning vs afternoon winner
if isinstance(am_vs_pm, dict) and 'error' not in am_vs_pm:
    if am_vs_pm['afternoon_sharpe'] > am_vs_pm['morning_sharpe']:
        timing_winner = "AFTERNOON"
    else:
        timing_winner = "MORNING"
    print(f"\n  1. {timing_winner} entries produce better risk-adjusted returns")
    print(f"     Morning Sharpe: {am_vs_pm['morning_sharpe']:.3f} vs Afternoon Sharpe: {am_vs_pm['afternoon_sharpe']:.3f}")
    sig_str = "STATISTICALLY SIGNIFICANT" if am_vs_pm.get('significant_at_5pct') else "NOT statistically significant"
    print(f"     Difference is {sig_str} (p={am_vs_pm.get('permutation_pval', 'N/A')})")

# Morning dip finding
if morning_dip_stats:
    avg_dip_freq = np.mean([v['dip_frequency'] for v in morning_dip_stats.values()])
    avg_recovery_wr = np.mean([v['recovery_win_rate'] for v in morning_dip_stats.values() if v['recovery_win_rate'] > 0])
    print(f"\n  2. MORNING DIP: Occurs {avg_dip_freq:.0%} of days on average")
    print(f"     When dip occurs, same-day recovery WR: {avg_recovery_wr:.0%}")

# RSI filter finding
if rsi_stats:
    for fd_key in ['3d', '5d']:
        if fd_key in rsi_stats:
            r = rsi_stats[fd_key]
            sig = "***" if r['ttest_pval'] < 0.01 else "**" if r['ttest_pval'] < 0.05 else "*" if r['ttest_pval'] < 0.1 else ""
            print(f"\n  3. RSI < 35 FILTER ({fd_key}): {r['rsi35_mean_bps']:.0f} bps vs {r['unconditional_mean_bps']:.0f} bps unconditional {sig}")
            print(f"     Sharpe: {r['rsi35_sharpe']:.3f}, WR: {r['rsi35_win_rate']:.0%}, N={r['n_rsi_entries']} entries")

# Best DOW
if dow_stats:
    best_dow_3d = max(dow_stats.items(), key=lambda x: x[1].get('3d', {}).get('sharpe', -99))
    if '3d' in best_dow_3d[1]:
        print(f"\n  4. BEST DAY: {best_dow_3d[0]} (3d Sharpe: {best_dow_3d[1]['3d']['sharpe']:.3f})")

print("\n  Current entry windows: 9:34 AM and 1:05 PM")
if hour_ranking:
    print(f"  Recommended best window: {hour_ranking[0]['label']} ET")
    if len(hour_ranking) > 1:
        print(f"  Second best window: {hour_ranking[1]['label']} ET")


# ── 8. Save Results ──────────────────────────────────────────────────
results = {
    'generated': datetime.now().isoformat(),
    'universe': TICKERS,
    'data_range': {'daily': f'{DAILY_START} to {DAILY_END}', 'intraday': f'last {INTRADAY_PERIOD}'},
    'hourly_entry_analysis': {
        'per_ticker': hourly_results,
        'aggregate': agg_hourly_stats,
        'ranking_3d_fwd': hour_ranking,
    },
    'morning_vs_afternoon': am_vs_pm,
    'morning_dip_pattern': morning_dip_stats,
    'rsi_filter': rsi_stats,
    'day_of_week': {
        'unconditional': dow_stats,
        'rsi_filtered': dow_rsi_stats,
    },
    'daily_patterns': daily_patterns,
}

# Convert any remaining numpy types
def convert_numpy(obj):
    if isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_numpy(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        return [convert_numpy(i) for i in obj]
    return obj

results = convert_numpy(results)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
with open(OUTPUT_PATH, 'w') as f:
    json.dump(results, f, indent=2, default=str)

print(f"\n\nResults saved to {OUTPUT_PATH}")
print("=" * 80)
