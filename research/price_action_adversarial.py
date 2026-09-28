#!/usr/bin/env python3
"""
Adversarial Validation of Price Action Structure Signal
========================================================
Stress-tests the "triple filter" RSI<35 dip-buy signal:
  - near_50d_low (within 2%) + swing_low within 5d + below 200 SMA
  - Claimed: Sharpe 10.17, WR 96.6%, 149 trades over 6 years on 11 sector ETFs

Six adversarial tests:
  1. Inverse signal test
  2. Random timing test (1000 iterations)
  3. Cost sensitivity
  4. Sub-period stability
  5. Parameter robustness
  6. Re-implementation check
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from scipy.signal import argrelextrema
from datetime import datetime
import sys

# ============================================================
# CONFIG
# ============================================================
SECTOR_ETFS = ['XLK', 'XLP', 'XLC', 'XLY', 'XLF', 'XLI', 'XLV', 'XLE', 'XLU', 'XLB', 'XLRE']
RSI_THRESHOLD = 35
RSI_PERIOD = 14
FWD_RETURN_DAYS = 5
LOOKBACK_YEARS = 6
SWING_ORDER = 5
SEED = 42
RANDOM_ITERS = 1000

# Original study results (for comparison)
ORIG_SHARPE = 10.17
ORIG_TRADE_COUNT = 149
ORIG_WR = 96.6

np.random.seed(SEED)


# ============================================================
# HELPER FUNCTIONS
# ============================================================
def calc_rsi(close, period=14):
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = (-delta.clip(upper=0))
    avg_gain = gain.ewm(alpha=1/period, min_periods=period).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def find_swing_lows(low_series, order=5):
    vals = low_series.values
    idx = argrelextrema(vals, np.less_equal, order=order)[0]
    return idx


def sharpe_ratio(returns, fwd_days=5):
    """Annualized Sharpe from trade returns with given holding period."""
    if len(returns) < 2:
        return 0.0
    periods_per_year = 252 / fwd_days
    mu = returns.mean()
    sigma = returns.std()
    if sigma == 0:
        return 0.0
    return mu / sigma * np.sqrt(periods_per_year)


def win_rate(returns):
    if len(returns) == 0:
        return 0.0
    return (returns > 0).mean() * 100


def profit_factor(returns):
    gains = returns[returns > 0].sum()
    losses = abs(returns[returns < 0].sum())
    if losses == 0:
        return np.inf if gains > 0 else 0.0
    return gains / losses


# ============================================================
# DATA DOWNLOAD + FEATURE COMPUTATION
# ============================================================
def download_data():
    """Download fresh data from yfinance."""
    print(f"Downloading {LOOKBACK_YEARS}yr daily data for {len(SECTOR_ETFS)} sector ETFs...")
    end_date = datetime.now()
    start_date = end_date - pd.DateOffset(years=LOOKBACK_YEARS)

    raw = yf.download(
        SECTOR_ETFS, start=start_date.strftime('%Y-%m-%d'),
        end=end_date.strftime('%Y-%m-%d'), auto_adjust=True, progress=False
    )
    return raw


def compute_all_entries(raw):
    """Compute features and return all RSI<35 entries with filter flags."""
    all_entries = []

    for ticker in SECTOR_ETFS:
        try:
            df = pd.DataFrame({
                'Open': raw['Open'][ticker],
                'High': raw['High'][ticker],
                'Low': raw['Low'][ticker],
                'Close': raw['Close'][ticker],
                'Volume': raw['Volume'][ticker],
            }).dropna()

            c = df['Close']
            h = df['High']
            l = df['Low']

            # RSI
            df['rsi'] = calc_rsi(c, RSI_PERIOD)

            # 50d low proximity
            roll_low_50 = l.rolling(50).min()
            df['dist_from_50d_low_pct'] = (c - roll_low_50) / roll_low_50 * 100
            df['near_50d_low'] = (df['dist_from_50d_low_pct'] < 2.0).astype(int)

            # 200 SMA
            df['sma_200'] = c.rolling(200).mean()
            df['below_sma_200'] = (c < df['sma_200']).astype(int)

            # 50 SMA
            df['sma_50'] = c.rolling(50).mean()
            df['below_sma_50'] = (c < df['sma_50']).astype(int)

            # Swing lows
            swing_low_idx = find_swing_lows(l, order=SWING_ORDER)
            days_since_swing_low = np.full(len(df), np.nan)
            for i in range(len(df)):
                past_lows = swing_low_idx[swing_low_idx <= i]
                if len(past_lows) > 0:
                    days_since_swing_low[i] = i - past_lows[-1]
            df['days_since_swing_low'] = days_since_swing_low

            # Forward returns
            df['fwd_5d_ret'] = c.shift(-FWD_RETURN_DAYS) / c - 1

            # Filter to RSI < 35
            rsi_entries = df[df['rsi'] < RSI_THRESHOLD].copy()
            rsi_entries['ticker'] = ticker
            all_entries.append(rsi_entries)
        except Exception as e:
            print(f"  WARNING: {ticker} failed: {e}")

    entries = pd.concat(all_entries, ignore_index=False)
    entries = entries.dropna(subset=['fwd_5d_ret'])
    return entries


def apply_triple_filter(entries):
    """Apply the triple filter: near_50d_low + swing_low_within_5d + below_200sma."""
    mask = (
        (entries['near_50d_low'] == 1) &
        (entries['days_since_swing_low'] <= 5) &
        (entries['below_sma_200'] == 1)
    )
    return mask


# ============================================================
# TEST 1: INVERSE SIGNAL
# ============================================================
def test_inverse_signal(entries):
    """If the filter works, the OPPOSITE entries should underperform."""
    print("\n" + "=" * 70)
    print("TEST 1: INVERSE SIGNAL")
    print("=" * 70)

    triple_mask = apply_triple_filter(entries)
    # Inverse: RSI<35 entries that FAIL at least one of the three conditions
    inverse_mask = ~triple_mask

    filtered_ret = entries.loc[triple_mask, 'fwd_5d_ret']
    inverse_ret = entries.loc[inverse_mask, 'fwd_5d_ret']

    filt_sharpe = sharpe_ratio(filtered_ret)
    inv_sharpe = sharpe_ratio(inverse_ret)
    filt_wr = win_rate(filtered_ret)
    inv_wr = win_rate(inverse_ret)

    print(f"  Triple filter:  n={len(filtered_ret)}, Sharpe={filt_sharpe:.2f}, WR={filt_wr:.1f}%")
    print(f"  Inverse filter: n={len(inverse_ret)}, Sharpe={inv_sharpe:.2f}, WR={inv_wr:.1f}%")
    print(f"  Sharpe delta (filter - inverse): {filt_sharpe - inv_sharpe:.2f}")

    # PASS if filter outperforms inverse
    passed = filt_sharpe > inv_sharpe
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} — filter {'outperforms' if passed else 'underperforms'} inverse")
    return passed, filt_sharpe, inv_sharpe


# ============================================================
# TEST 2: RANDOM TIMING
# ============================================================
def test_random_timing(entries, real_sharpe):
    """Randomly sample same-count subsets of RSI<35, compute Sharpe distribution."""
    print("\n" + "=" * 70)
    print("TEST 2: RANDOM TIMING (1000 iterations)")
    print("=" * 70)

    triple_mask = apply_triple_filter(entries)
    n_trades = triple_mask.sum()
    all_rsi_returns = entries['fwd_5d_ret'].values

    if n_trades == 0:
        print("  No trades from filter. FAIL.")
        return False, 0, 0

    random_sharpes = []
    for _ in range(RANDOM_ITERS):
        idx = np.random.choice(len(all_rsi_returns), size=n_trades, replace=False)
        sample_ret = pd.Series(all_rsi_returns[idx])
        random_sharpes.append(sharpe_ratio(sample_ret))

    random_sharpes = np.array(random_sharpes)
    percentile = (random_sharpes < real_sharpe).mean() * 100
    p_value = (random_sharpes >= real_sharpe).mean()

    print(f"  Real filter Sharpe: {real_sharpe:.2f}")
    print(f"  Random Sharpe distribution: mean={random_sharpes.mean():.2f}, "
          f"std={random_sharpes.std():.2f}, max={random_sharpes.max():.2f}")
    print(f"  Percentile of real Sharpe: {percentile:.1f}th")
    print(f"  P-value (fraction random >= real): {p_value:.4f}")

    # PASS if p-value < 0.05 (real filter is in top 5%)
    passed = p_value < 0.05
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} — p={p_value:.4f} {'<' if passed else '>='} 0.05")
    return passed, percentile, p_value


# ============================================================
# TEST 3: COST SENSITIVITY
# ============================================================
def test_cost_sensitivity(entries):
    """Test at escalating transaction costs."""
    print("\n" + "=" * 70)
    print("TEST 3: COST SENSITIVITY")
    print("=" * 70)

    triple_mask = apply_triple_filter(entries)
    filtered_ret = entries.loc[triple_mask, 'fwd_5d_ret'].copy()

    costs_bps = [0, 25, 50, 75, 100, 150, 200]
    breakeven_cost = None

    print(f"  {'Cost (bps RT)':>14} {'Sharpe':>8} {'Avg Ret%':>9} {'WR%':>6} {'PF':>6}")
    print("  " + "-" * 45)

    for cost in costs_bps:
        cost_frac = cost / 10000  # convert bps to fraction
        net_ret = filtered_ret - cost_frac
        s = sharpe_ratio(net_ret)
        wr = win_rate(net_ret)
        pf = profit_factor(net_ret)
        pf_str = f"{pf:.2f}" if pf < 100 else "inf"
        print(f"  {cost:>14} {s:>8.2f} {net_ret.mean()*100:>9.3f} {wr:>6.1f} {pf_str:>6}")

        if s <= 0 and breakeven_cost is None:
            breakeven_cost = cost

    # Find more precise breakeven via binary search
    if breakeven_cost is None:
        breakeven_cost = costs_bps[-1]  # survives all costs
        print(f"\n  Survives up to {costs_bps[-1]}bps cost!")
    else:
        # Binary search between previous cost and this one
        lo = costs_bps[costs_bps.index(breakeven_cost) - 1] if costs_bps.index(breakeven_cost) > 0 else 0
        hi = breakeven_cost
        for _ in range(20):
            mid = (lo + hi) / 2
            net = filtered_ret - mid / 10000
            if sharpe_ratio(net) > 0:
                lo = mid
            else:
                hi = mid
        breakeven_cost = int(round(hi))
        print(f"\n  Breakeven cost: ~{breakeven_cost}bps round-trip")

    # PASS if survives >= 50bps (realistic ETF trading cost)
    net_50 = filtered_ret - 0.0050
    sharpe_50 = sharpe_ratio(net_50)
    passed = sharpe_50 > 0
    print(f"  Sharpe at 50bps: {sharpe_50:.2f}")
    print(f"  RESULT: {'PASS' if passed else 'FAIL'} — {'survives' if passed else 'dies at'} 50bps cost")
    return passed, breakeven_cost


# ============================================================
# TEST 4: SUB-PERIOD STABILITY
# ============================================================
def test_subperiod_stability(entries):
    """Split into 4 equal sub-periods, all must have positive Sharpe."""
    print("\n" + "=" * 70)
    print("TEST 4: SUB-PERIOD STABILITY")
    print("=" * 70)

    triple_mask = apply_triple_filter(entries)
    filtered = entries.loc[triple_mask].copy()
    filtered = filtered.sort_index()

    dates = filtered.index
    min_date = dates.min()
    max_date = dates.max()
    total_days = (max_date - min_date).days
    period_days = total_days / 4

    print(f"  Date range: {min_date.strftime('%Y-%m-%d')} to {max_date.strftime('%Y-%m-%d')}")
    print(f"  Total span: {total_days} days, ~{total_days/365:.1f} years")
    print(f"  Sub-period length: ~{period_days:.0f} days (~{period_days/365:.1f} years)")

    boundaries = [min_date + pd.Timedelta(days=period_days * i) for i in range(5)]
    sub_sharpes = []
    all_positive = True

    print(f"\n  {'Period':>10} {'Dates':>25} {'N':>5} {'Sharpe':>8} {'WR%':>6} {'AvgRet%':>8}")
    print("  " + "-" * 65)

    for i in range(4):
        start = boundaries[i]
        end = boundaries[i + 1]
        sub = filtered[(filtered.index >= start) & (filtered.index < end)]
        ret = sub['fwd_5d_ret']
        s = sharpe_ratio(ret) if len(ret) >= 2 else 0.0
        wr = win_rate(ret) if len(ret) > 0 else 0.0
        avg = ret.mean() * 100 if len(ret) > 0 else 0.0
        sub_sharpes.append(s)

        date_str = f"{start.strftime('%Y-%m')}-{end.strftime('%Y-%m')}"
        print(f"  P{i+1:>8} {date_str:>25} {len(ret):>5} {s:>8.2f} {wr:>6.1f} {avg:>8.3f}")

        if s <= 0:
            all_positive = False

    print(f"\n  All sub-periods positive Sharpe: {'YES' if all_positive else 'NO'}")
    print(f"  Min sub-period Sharpe: {min(sub_sharpes):.2f}")
    print(f"  RESULT: {'PASS' if all_positive else 'FAIL'}")
    return all_positive, sub_sharpes


# ============================================================
# TEST 5: PARAMETER ROBUSTNESS
# ============================================================
def test_parameter_robustness(entries):
    """Test 100+ parameter combinations to check for overfitting."""
    print("\n" + "=" * 70)
    print("TEST 5: PARAMETER ROBUSTNESS")
    print("=" * 70)

    # Parameter grid
    low_thresholds = [1.0, 2.0, 3.0, 5.0]  # % near 50d low
    swing_lookbacks = [3, 5, 7, 10]  # days since swing low
    use_200sma = [True, False]
    use_50sma = [True, False]

    results = []
    total_combos = len(low_thresholds) * len(swing_lookbacks) * len(use_200sma) * len(use_50sma)

    for lt in low_thresholds:
        for sl in swing_lookbacks:
            for sma200 in use_200sma:
                for sma50 in use_50sma:
                    mask = (
                        (entries['dist_from_50d_low_pct'] < lt) &
                        (entries['days_since_swing_low'] <= sl)
                    )
                    if sma200:
                        mask = mask & (entries['below_sma_200'] == 1)
                    if sma50:
                        mask = mask & (entries['below_sma_50'] == 1)

                    filtered_ret = entries.loc[mask, 'fwd_5d_ret']
                    n = len(filtered_ret)
                    if n >= 10:
                        s = sharpe_ratio(filtered_ret)
                        wr = win_rate(filtered_ret)
                    else:
                        s = np.nan
                        wr = np.nan

                    results.append({
                        'low_thresh': lt,
                        'swing_lb': sl,
                        'sma200': sma200,
                        'sma50': sma50,
                        'n_trades': n,
                        'sharpe': s,
                        'wr': wr,
                    })

    rdf = pd.DataFrame(results)
    valid = rdf.dropna(subset=['sharpe'])

    n_total = len(valid)
    n_sharpe_pos = (valid['sharpe'] > 0).sum()
    n_sharpe_1 = (valid['sharpe'] > 1.0).sum()
    n_sharpe_3 = (valid['sharpe'] > 3.0).sum()

    frac_pos = n_sharpe_pos / n_total if n_total > 0 else 0
    frac_1 = n_sharpe_1 / n_total if n_total > 0 else 0
    frac_3 = n_sharpe_3 / n_total if n_total > 0 else 0

    print(f"  Total parameter combinations: {total_combos}")
    print(f"  Valid (>=10 trades): {n_total}")
    print(f"  Sharpe > 0: {n_sharpe_pos}/{n_total} ({frac_pos*100:.1f}%)")
    print(f"  Sharpe > 1: {n_sharpe_1}/{n_total} ({frac_1*100:.1f}%)")
    print(f"  Sharpe > 3: {n_sharpe_3}/{n_total} ({frac_3*100:.1f}%)")

    # Show best and worst
    if n_total > 0:
        best = valid.loc[valid['sharpe'].idxmax()]
        worst = valid.loc[valid['sharpe'].idxmin()]
        median_sharpe = valid['sharpe'].median()
        print(f"\n  Median Sharpe across all combos: {median_sharpe:.2f}")
        print(f"  Best:  low={best['low_thresh']}%, swing={best['swing_lb']}d, "
              f"sma200={best['sma200']}, sma50={best['sma50']} -> "
              f"n={best['n_trades']:.0f}, Sharpe={best['sharpe']:.2f}")
        print(f"  Worst: low={worst['low_thresh']}%, swing={worst['swing_lb']}d, "
              f"sma200={worst['sma200']}, sma50={worst['sma50']} -> "
              f"n={worst['n_trades']:.0f}, Sharpe={worst['sharpe']:.2f}")

    # PASS if >50% of combos have Sharpe > 1.0 (robust, not cherry-picked)
    passed = frac_1 > 0.50
    print(f"\n  RESULT: {'PASS' if passed else 'FAIL'} — {frac_1*100:.1f}% of combos have Sharpe > 1.0 "
          f"(threshold: >50%)")
    return passed, frac_1, rdf


# ============================================================
# TEST 6: RE-IMPLEMENTATION CHECK
# ============================================================
def test_reimplementation(entries):
    """Re-implement from scratch, compare trade count and Sharpe."""
    print("\n" + "=" * 70)
    print("TEST 6: RE-IMPLEMENTATION CHECK")
    print("=" * 70)

    print("  Re-downloading data and computing everything from scratch...")

    # Fresh download
    end_dt = datetime.now()
    start_dt = end_dt - pd.DateOffset(years=LOOKBACK_YEARS)
    raw2 = yf.download(
        SECTOR_ETFS, start=start_dt.strftime('%Y-%m-%d'),
        end=end_dt.strftime('%Y-%m-%d'), auto_adjust=True, progress=False
    )

    all_trades2 = []

    for ticker in SECTOR_ETFS:
        try:
            close = raw2['Close'][ticker].dropna()
            high = raw2['High'][ticker].dropna()
            low = raw2['Low'][ticker].dropna()

            # Align
            common_idx = close.index.intersection(high.index).intersection(low.index)
            close = close.loc[common_idx]
            high = high.loc[common_idx]
            low = low.loc[common_idx]

            # RSI — independent implementation (Wilder smoothing via EWM)
            delta = close.diff()
            up = delta.where(delta > 0, 0.0)
            dn = -delta.where(delta < 0, 0.0)
            avg_up = up.ewm(alpha=1.0/RSI_PERIOD, min_periods=RSI_PERIOD).mean()
            avg_dn = dn.ewm(alpha=1.0/RSI_PERIOD, min_periods=RSI_PERIOD).mean()
            rs = avg_up / avg_dn
            rsi = 100.0 - 100.0 / (1.0 + rs)

            # 50-day rolling low
            low_50d = low.rolling(50).min()
            dist_50d = (close - low_50d) / low_50d * 100.0

            # 200 SMA
            sma200 = close.rolling(200).mean()

            # Swing lows (order=5)
            low_arr = low.values
            swing_idx = argrelextrema(low_arr, np.less_equal, order=SWING_ORDER)[0]
            d_since_swing = pd.Series(np.nan, index=close.index)
            for i in range(len(close)):
                past = swing_idx[swing_idx <= i]
                if len(past) > 0:
                    d_since_swing.iloc[i] = i - past[-1]

            # Forward 5d return
            fwd_ret = close.shift(-FWD_RETURN_DAYS) / close - 1.0

            # Build mask
            rsi_mask = rsi < RSI_THRESHOLD
            near_50d = dist_50d < 2.0
            below_200 = close < sma200
            swing_5d = d_since_swing <= 5.0

            triple = rsi_mask & near_50d & swing_5d & below_200
            trade_ret = fwd_ret[triple].dropna()
            all_trades2.append(trade_ret)

        except Exception as e:
            print(f"    WARNING: {ticker}: {e}")

    reimpl_returns = pd.concat(all_trades2)
    reimpl_n = len(reimpl_returns)
    reimpl_sharpe = sharpe_ratio(reimpl_returns)
    reimpl_wr = win_rate(reimpl_returns)

    # Original from our first run
    triple_mask = apply_triple_filter(entries)
    orig_returns = entries.loc[triple_mask, 'fwd_5d_ret']
    orig_n = len(orig_returns)
    orig_sharpe = sharpe_ratio(orig_returns)
    orig_wr = win_rate(orig_returns)

    print(f"\n  Original implementation:      n={orig_n}, Sharpe={orig_sharpe:.2f}, WR={orig_wr:.1f}%")
    print(f"  Re-implementation:            n={reimpl_n}, Sharpe={reimpl_sharpe:.2f}, WR={reimpl_wr:.1f}%")

    # Check within 10%
    count_match = abs(reimpl_n - orig_n) / max(orig_n, 1) < 0.10
    sharpe_match = abs(reimpl_sharpe - orig_sharpe) / max(abs(orig_sharpe), 0.01) < 0.10

    print(f"\n  Trade count match (<10% diff): {'YES' if count_match else 'NO'} "
          f"(diff={abs(reimpl_n - orig_n)}, {abs(reimpl_n - orig_n)/max(orig_n,1)*100:.1f}%)")
    print(f"  Sharpe match (<10% diff):      {'YES' if sharpe_match else 'NO'} "
          f"(diff={abs(reimpl_sharpe - orig_sharpe):.2f}, "
          f"{abs(reimpl_sharpe - orig_sharpe)/max(abs(orig_sharpe),0.01)*100:.1f}%)")

    passed = count_match and sharpe_match
    print(f"  RESULT: {'PASS' if passed else 'FAIL'}")
    return passed, reimpl_n, reimpl_sharpe


# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 70)
    print("ADVERSARIAL VALIDATION: PRICE ACTION STRUCTURE SIGNAL")
    print(f"Run date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 70)

    # Download and compute
    raw = download_data()
    entries = compute_all_entries(raw)
    print(f"Total RSI<{RSI_THRESHOLD} entries: {len(entries)}")

    # Get real filter stats
    triple_mask = apply_triple_filter(entries)
    filtered_ret = entries.loc[triple_mask, 'fwd_5d_ret']
    real_sharpe = sharpe_ratio(filtered_ret)
    real_n = len(filtered_ret)
    real_wr = win_rate(filtered_ret)

    print(f"\nTriple filter results: n={real_n}, Sharpe={real_sharpe:.2f}, "
          f"WR={real_wr:.1f}%, PF={profit_factor(filtered_ret):.2f}")

    # Run all 6 tests
    results = {}

    # Test 1
    p1, _, _ = test_inverse_signal(entries)
    results['1. Inverse Signal'] = p1

    # Test 2
    p2, percentile, pval = test_random_timing(entries, real_sharpe)
    results['2. Random Timing'] = p2

    # Test 3
    p3, breakeven = test_cost_sensitivity(entries)
    results['3. Cost Sensitivity'] = p3

    # Test 4
    p4, sub_sharpes = test_subperiod_stability(entries)
    results['4. Sub-Period Stability'] = p4

    # Test 5
    p5, frac_robust, param_df = test_parameter_robustness(entries)
    results['5. Parameter Robustness'] = p5

    # Test 6
    p6, reimpl_n, reimpl_sharpe = test_reimplementation(entries)
    results['6. Re-Implementation'] = p6

    # ============================================================
    # FINAL SCORECARD
    # ============================================================
    print("\n" + "=" * 70)
    print("ADVERSARIAL SCORECARD")
    print("=" * 70)

    passed_count = 0
    for test_name, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {test_name:<30} {status}")
        if passed:
            passed_count += 1

    print(f"\n  SCORE: {passed_count}/6")

    if passed_count >= 5:
        verdict = "ALIVE"
    elif passed_count >= 3:
        verdict = "PARTIAL"
    else:
        verdict = "DEAD"

    print(f"  VERDICT: {verdict}")

    # Concerns
    print("\n" + "=" * 70)
    print("ADDITIONAL CONCERNS")
    print("=" * 70)

    concerns = []

    # Survivorship bias
    concerns.append(
        "SURVIVORSHIP BIAS: Using only current sector ETFs (XLRE started 2015). "
        "No delisted or restructured ETFs included. Mild concern — sector ETFs rarely delist, "
        "but the universe is static."
    )

    # Look-ahead in swing detection
    concerns.append(
        "LOOK-AHEAD (SWING LOWS): argrelextrema with order=5 uses both past AND future data "
        "to identify swing lows. A swing low at day T is only confirmed at day T+5. "
        "The 'days_since_swing_low' feature therefore has 5 days of look-ahead bias. "
        "This is a SIGNIFICANT concern — in live trading you wouldn't know a swing low "
        "occurred until 5 bars later."
    )

    # Overlapping returns
    concerns.append(
        "OVERLAPPING RETURNS: 5-day forward returns with entries that may be consecutive days "
        "creates overlapping holding periods. This inflates effective sample size and "
        "understates standard errors. The high Sharpe may partly reflect autocorrelation."
    )

    # Annualization
    concerns.append(
        f"ANNUALIZATION: Sharpe of {real_sharpe:.2f} is annualized from {real_n} trade returns "
        f"over 6 years. With ~{real_n/6:.0f} trades/year and 5-day holding, this uses "
        f"sqrt(252/5)=sqrt(50.4)=7.1x annualization factor. A high per-trade mean/std "
        f"gets amplified significantly."
    )

    for i, c in enumerate(concerns, 1):
        print(f"\n  {i}. {c}")

    print("\n" + "=" * 70)
    print(f"FINAL VERDICT: {verdict} ({passed_count}/6 tests passed)")
    print("=" * 70)


if __name__ == '__main__':
    main()
