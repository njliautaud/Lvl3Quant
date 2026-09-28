#!/usr/bin/env python3
"""
VIX Regime Filter for Sector ETF Dip-Buying
============================================
Research question: Does VIX level at RSI<35 entry predict which dips recover faster?

Tests:
1. VIX bucket analysis (Low/Med/High/Extreme)
2. VIX 5-day change analysis (rising vs falling VIX)
3. Permutation test on best filter
4. Regime stratification (SPY green vs red days) per HC #428
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

# ── Config ──────────────────────────────────────────────────────────────
SECTORS = ['XLK', 'XLP', 'XLC', 'XLY', 'XLF', 'XLI', 'XLV', 'XLE', 'XLU', 'XLB', 'XLRE']
RSI_THRESHOLD = 35
RSI_PERIOD = 14
HOLD_DAYS = 5
VIX_CHANGE_LOOKBACK = 5
N_PERMUTATIONS = 1000
YEARS_BACK = 6  # 6 years to ensure 5+ years of usable data after warmup

# VIX buckets
VIX_BUCKETS = {
    'Low (<15)':      (0, 15),
    'Medium (15-25)': (15, 25),
    'High (25-35)':   (25, 35),
    'Extreme (>35)':  (35, 999),
}

# VIX change buckets
VIX_CHANGE_BUCKETS = {
    'Falling (< -3)':  (-999, -3),
    'Slight Fall (-3,0)': (-3, 0),
    'Slight Rise (0,3)':  (0, 3),
    'Rising (> 3)':       (3, 999),
}


def compute_rsi(series, period=14):
    """Wilder's RSI."""
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = (-delta).clip(lower=0)
    avg_gain = gain.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))


def calc_metrics(returns):
    """Calculate Sharpe (ann from 5d), win rate, profit factor, count."""
    n = len(returns)
    if n < 5:
        return {'count': n, 'avg_ret': np.nan, 'sharpe': np.nan, 'win_rate': np.nan, 'profit_factor': np.nan}

    avg = returns.mean()
    std = returns.std()
    # Annualize from 5-day holding periods (~50 periods/year)
    sharpe = (avg / std) * np.sqrt(252 / HOLD_DAYS) if std > 0 else 0.0

    wins = returns[returns > 0]
    losses = returns[returns < 0]
    wr = len(wins) / n if n > 0 else 0
    pf = wins.sum() / abs(losses.sum()) if len(losses) > 0 and losses.sum() != 0 else np.inf

    return {
        'count': n,
        'avg_ret': avg * 100,  # percent
        'sharpe': sharpe,
        'win_rate': wr * 100,
        'profit_factor': pf,
    }


def permutation_test(returns_filtered, returns_all, n_perms=1000):
    """Test if filtered Sharpe is significantly better than random subset of same size."""
    n_filt = len(returns_filtered)
    if n_filt < 10:
        return np.nan

    obs_sharpe = calc_metrics(returns_filtered)['sharpe']
    all_arr = returns_all.values

    count_ge = 0
    for _ in range(n_perms):
        idx = np.random.choice(len(all_arr), size=n_filt, replace=False)
        sample = pd.Series(all_arr[idx])
        s = calc_metrics(sample)['sharpe']
        if s >= obs_sharpe:
            count_ge += 1

    return count_ge / n_perms


def main():
    print("=" * 80)
    print("VIX REGIME FILTER FOR SECTOR ETF DIP-BUYING")
    print("=" * 80)

    # ── 1. Download data ────────────────────────────────────────────────
    end_date = datetime.now()
    start_date = end_date - timedelta(days=365 * YEARS_BACK)

    tickers = SECTORS + ['^VIX', 'SPY']
    print(f"\nDownloading {len(tickers)} tickers, {start_date.date()} to {end_date.date()}...")

    data = yf.download(tickers, start=start_date, end=end_date, progress=False)

    # Handle multi-level columns from yfinance
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close'] if 'Close' in data.columns.get_level_values(0) else data['Adj Close']
    else:
        close = data

    # Rename ^VIX to VIX
    if '^VIX' in close.columns:
        close = close.rename(columns={'^VIX': 'VIX'})

    print(f"Data shape: {close.shape}, date range: {close.index[0].date()} to {close.index[-1].date()}")

    # ── 2. Compute RSI for all sectors ──────────────────────────────────
    rsi_df = pd.DataFrame(index=close.index)
    for sector in SECTORS:
        if sector in close.columns:
            rsi_df[sector] = compute_rsi(close[sector], RSI_PERIOD)

    # SPY daily returns for regime classification
    spy_ret = close['SPY'].pct_change()

    # VIX 5-day change
    vix = close['VIX']
    vix_change_5d = vix.diff(VIX_CHANGE_LOOKBACK)

    # ── 3. Find all RSI<35 entry points ─────────────────────────────────
    entries = []
    for sector in SECTORS:
        if sector not in close.columns or sector not in rsi_df.columns:
            continue

        sector_close = close[sector].dropna()
        sector_rsi = rsi_df[sector].dropna()

        for i in range(RSI_PERIOD + 1, len(sector_close) - HOLD_DAYS):
            date = sector_close.index[i]

            if date not in sector_rsi.index:
                continue

            rsi_val = sector_rsi.loc[date]
            if rsi_val >= RSI_THRESHOLD:
                continue

            # Check previous day wasn't also RSI<35 (avoid double-counting sustained oversold)
            prev_date_idx = sector_close.index.get_loc(date) - 1
            if prev_date_idx >= 0:
                prev_date = sector_close.index[prev_date_idx]
                if prev_date in sector_rsi.index and sector_rsi.loc[prev_date] < RSI_THRESHOLD:
                    continue  # Only take first day of oversold

            # 5-day forward return
            entry_price = sector_close.iloc[i]
            exit_idx = i + HOLD_DAYS
            if exit_idx >= len(sector_close):
                continue
            exit_price = sector_close.iloc[exit_idx]
            ret_5d = (exit_price / entry_price) - 1

            # VIX at entry
            if date not in vix.index or pd.isna(vix.loc[date]):
                continue
            vix_at_entry = vix.loc[date]

            # VIX 5d change
            vix_chg = vix_change_5d.loc[date] if date in vix_change_5d.index else np.nan

            # SPY regime (green/red on entry day)
            spy_regime = 'green' if (date in spy_ret.index and spy_ret.loc[date] > 0) else 'red'

            entries.append({
                'date': date,
                'sector': sector,
                'rsi': rsi_val,
                'vix': vix_at_entry,
                'vix_change_5d': vix_chg,
                'ret_5d': ret_5d,
                'spy_regime': spy_regime,
            })

    df = pd.DataFrame(entries)
    print(f"\nTotal RSI<35 entry signals: {len(df)}")
    print(f"Sectors represented: {df['sector'].nunique()}")
    print(f"Date range: {df['date'].min().date()} to {df['date'].max().date()}")

    # ── 4. VIX Bucket Analysis ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VIX BUCKET ANALYSIS (RSI<35 dip entries)")
    print("=" * 80)

    df['vix_bucket'] = pd.cut(
        df['vix'],
        bins=[0, 15, 25, 35, 999],
        labels=['Low (<15)', 'Medium (15-25)', 'High (25-35)', 'Extreme (>35)'],
        right=False
    )

    bucket_results = {}
    print(f"\n{'Bucket':<20} {'Count':>6} {'Avg 5d%':>8} {'Sharpe':>8} {'WR%':>7} {'PF':>7}")
    print("-" * 60)

    for bucket_name in ['Low (<15)', 'Medium (15-25)', 'High (25-35)', 'Extreme (>35)']:
        mask = df['vix_bucket'] == bucket_name
        rets = df.loc[mask, 'ret_5d']
        m = calc_metrics(rets)
        bucket_results[bucket_name] = m
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] != np.inf else "inf"
        print(f"{bucket_name:<20} {m['count']:>6} {m['avg_ret']:>8.2f} {m['sharpe']:>8.2f} {m['win_rate']:>6.1f}% {pf_str:>7}")

    # Baseline (all entries)
    all_metrics = calc_metrics(df['ret_5d'])
    pf_str = f"{all_metrics['profit_factor']:.2f}" if all_metrics['profit_factor'] != np.inf else "inf"
    print(f"\n{'ALL (baseline)':<20} {all_metrics['count']:>6} {all_metrics['avg_ret']:>8.2f} {all_metrics['sharpe']:>8.2f} {all_metrics['win_rate']:>6.1f}% {pf_str:>7}")

    # ── 5. VIX Change Analysis ──────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VIX 5-DAY CHANGE ANALYSIS")
    print("=" * 80)

    df['vix_change_bucket'] = pd.cut(
        df['vix_change_5d'],
        bins=[-999, -3, 0, 3, 999],
        labels=['Falling (< -3)', 'Slight Fall (-3,0)', 'Slight Rise (0,3)', 'Rising (> 3)'],
        right=False
    )

    print(f"\n{'VIX Change':<25} {'Count':>6} {'Avg 5d%':>8} {'Sharpe':>8} {'WR%':>7} {'PF':>7}")
    print("-" * 65)

    vix_change_results = {}
    for label in ['Falling (< -3)', 'Slight Fall (-3,0)', 'Slight Rise (0,3)', 'Rising (> 3)']:
        mask = df['vix_change_bucket'] == label
        rets = df.loc[mask, 'ret_5d']
        m = calc_metrics(rets)
        vix_change_results[label] = m
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] != np.inf else "inf"
        print(f"{label:<25} {m['count']:>6} {m['avg_ret']:>8.2f} {m['sharpe']:>8.2f} {m['win_rate']:>6.1f}% {pf_str:>7}")

    # ── 6. Find Best Filter ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("BEST FILTER SEARCH")
    print("=" * 80)

    # Test all VIX bucket combinations
    best_sharpe = -999
    best_filter = None
    best_rets = None

    bucket_names = ['Low (<15)', 'Medium (15-25)', 'High (25-35)', 'Extreme (>35)']

    # Test single buckets and pairs
    from itertools import combinations
    for r in range(1, len(bucket_names) + 1):
        for combo in combinations(bucket_names, r):
            mask = df['vix_bucket'].isin(combo)
            rets = df.loc[mask, 'ret_5d']
            if len(rets) < 20:
                continue
            m = calc_metrics(rets)
            if m['sharpe'] > best_sharpe:
                best_sharpe = m['sharpe']
                best_filter = combo
                best_rets = rets

    # Also test VIX change filters
    change_labels = ['Falling (< -3)', 'Slight Fall (-3,0)', 'Slight Rise (0,3)', 'Rising (> 3)']
    for r in range(1, len(change_labels) + 1):
        for combo in combinations(change_labels, r):
            mask = df['vix_change_bucket'].isin(combo)
            rets = df.loc[mask, 'ret_5d']
            if len(rets) < 20:
                continue
            m = calc_metrics(rets)
            label = 'VIX_CHG:' + '+'.join([c.split('(')[0].strip() for c in combo])
            if m['sharpe'] > best_sharpe:
                best_sharpe = m['sharpe']
                best_filter = combo
                best_rets = rets

    # Also test combined VIX level + change
    for vix_b in bucket_names:
        for chg_b in change_labels:
            mask = (df['vix_bucket'] == vix_b) & (df['vix_change_bucket'] == chg_b)
            rets = df.loc[mask, 'ret_5d']
            if len(rets) < 20:
                continue
            m = calc_metrics(rets)
            if m['sharpe'] > best_sharpe:
                best_sharpe = m['sharpe']
                best_filter = (vix_b, 'AND', chg_b)
                best_rets = rets

    best_m = calc_metrics(best_rets)
    pf_str = f"{best_m['profit_factor']:.2f}" if best_m['profit_factor'] != np.inf else "inf"
    print(f"\nBest filter: {best_filter}")
    print(f"  Count:         {best_m['count']}")
    print(f"  Avg 5d return: {best_m['avg_ret']:.2f}%")
    print(f"  Sharpe (ann):  {best_m['sharpe']:.2f}")
    print(f"  Win Rate:      {best_m['win_rate']:.1f}%")
    print(f"  Profit Factor: {pf_str}")

    # ── 7. Permutation Test ─────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PERMUTATION TEST (1000 shuffles)")
    print("=" * 80)

    np.random.seed(42)
    p_value = permutation_test(best_rets, df['ret_5d'], N_PERMUTATIONS)
    print(f"\nObserved Sharpe: {best_m['sharpe']:.2f}")
    print(f"p-value: {p_value:.3f}")
    print(f"Significant at 5%: {'YES' if p_value < 0.05 else 'NO'}")
    print(f"Significant at 10%: {'YES' if p_value < 0.10 else 'NO'}")

    # ── 8. Regime Stratification (HC #428) ──────────────────────────────
    print("\n" + "=" * 80)
    print("REGIME STRATIFICATION (SPY green vs red day at entry) — HC #428")
    print("=" * 80)

    # Apply best filter first
    if len(best_filter) == 3 and best_filter[1] == 'AND':
        best_mask = (df['vix_bucket'] == best_filter[0]) & (df['vix_change_bucket'] == best_filter[2])
    else:
        # It's a tuple of bucket names
        best_mask = df['vix_bucket'].isin(best_filter) | df['vix_change_bucket'].isin(best_filter)

    # Re-check: use the actual best filter properly
    # Determine if best_filter is VIX buckets or VIX change buckets
    is_vix_bucket = any(b in bucket_names for b in best_filter if b != 'AND')
    is_vix_change = any(b in change_labels for b in best_filter if b != 'AND')

    if len(best_filter) == 3 and best_filter[1] == 'AND':
        filtered_df = df[(df['vix_bucket'] == best_filter[0]) & (df['vix_change_bucket'] == best_filter[2])]
    elif is_vix_bucket and not is_vix_change:
        filtered_df = df[df['vix_bucket'].isin(best_filter)]
    elif is_vix_change and not is_vix_bucket:
        filtered_df = df[df['vix_change_bucket'].isin(best_filter)]
    else:
        filtered_df = df[df['vix_bucket'].isin(best_filter) | df['vix_change_bucket'].isin(best_filter)]

    # Stratify by regime
    green_rets = filtered_df.loc[filtered_df['spy_regime'] == 'green', 'ret_5d']
    red_rets = filtered_df.loc[filtered_df['spy_regime'] == 'red', 'ret_5d']

    green_m = calc_metrics(green_rets)
    red_m = calc_metrics(red_rets)

    print(f"\n{'Regime':<15} {'Count':>6} {'Avg 5d%':>8} {'Sharpe':>8} {'WR%':>7} {'PF':>7}")
    print("-" * 55)

    for label, m in [('SPY Green', green_m), ('SPY Red', red_m)]:
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] != np.inf else "inf"
        if np.isnan(m['sharpe']):
            print(f"{label:<15} {m['count']:>6} {'N/A':>8} {'N/A':>8} {'N/A':>7} {'N/A':>7}")
        else:
            print(f"{label:<15} {m['count']:>6} {m['avg_ret']:>8.2f} {m['sharpe']:>8.2f} {m['win_rate']:>6.1f}% {pf_str:>7}")

    # Regime gap check (HC #428 R1)
    if not np.isnan(green_m['sharpe']) and not np.isnan(red_m['sharpe']):
        max_sharpe = max(abs(green_m['sharpe']), abs(red_m['sharpe']))
        if max_sharpe > 0:
            regime_gap = abs(green_m['sharpe'] - red_m['sharpe']) / max_sharpe
        else:
            regime_gap = 0
        print(f"\nRegime gap: |Sharpe_green - Sharpe_red| / max = {regime_gap:.2f}")
        print(f"HC #428 threshold: 0.50")
        regime_pass = regime_gap <= 0.50
        print(f"Regime-agnostic: {'PASS' if regime_pass else 'FAIL (regime-tailored, not real edge)'}")
    else:
        regime_gap = np.nan
        regime_pass = False
        print("\nInsufficient data for regime gap calculation.")

    # ── Also do regime stratification on UNFILTERED baseline ────────────
    print("\n--- Baseline (all RSI<35 entries) regime stratification ---")
    green_all = df.loc[df['spy_regime'] == 'green', 'ret_5d']
    red_all = df.loc[df['spy_regime'] == 'red', 'ret_5d']
    green_all_m = calc_metrics(green_all)
    red_all_m = calc_metrics(red_all)

    for label, m in [('SPY Green', green_all_m), ('SPY Red', red_all_m)]:
        pf_str = f"{m['profit_factor']:.2f}" if m['profit_factor'] != np.inf else "inf"
        print(f"{label:<15} {m['count']:>6} {m['avg_ret']:>8.2f} {m['sharpe']:>8.2f} {m['win_rate']:>6.1f}% {pf_str:>7}")

    max_s_base = max(abs(green_all_m['sharpe']), abs(red_all_m['sharpe']))
    if max_s_base > 0:
        base_gap = abs(green_all_m['sharpe'] - red_all_m['sharpe']) / max_s_base
    else:
        base_gap = 0
    print(f"Baseline regime gap: {base_gap:.2f}")

    # ── 9. Per-Sector Breakdown ─────────────────────────────────────────
    print("\n" + "=" * 80)
    print("PER-SECTOR PERFORMANCE (all RSI<35 entries)")
    print("=" * 80)

    print(f"\n{'Sector':<8} {'Count':>6} {'Avg 5d%':>8} {'Sharpe':>8} {'WR%':>7} {'Avg VIX':>8}")
    print("-" * 50)

    for sector in sorted(SECTORS):
        mask = df['sector'] == sector
        rets = df.loc[mask, 'ret_5d']
        m = calc_metrics(rets)
        avg_vix = df.loc[mask, 'vix'].mean()
        if m['count'] > 0:
            print(f"{sector:<8} {m['count']:>6} {m['avg_ret']:>8.2f} {m['sharpe']:>8.2f} {m['win_rate']:>6.1f}% {avg_vix:>8.1f}")

    # ── 10. Final Verdict ───────────────────────────────────────────────
    print("\n" + "=" * 80)
    print("VERDICT")
    print("=" * 80)

    # Criteria for ALIVE:
    # - Best filter Sharpe > baseline Sharpe by meaningful amount
    # - Permutation p < 0.10
    # - Regime gap < 0.50
    # - Count >= 30

    sharpe_improvement = best_m['sharpe'] - all_metrics['sharpe']

    print(f"\nBaseline Sharpe:       {all_metrics['sharpe']:.2f}")
    print(f"Best filter Sharpe:    {best_m['sharpe']:.2f}")
    print(f"Sharpe improvement:    {sharpe_improvement:+.2f}")
    print(f"Permutation p-value:   {p_value:.3f}")
    print(f"Regime gap:            {regime_gap:.2f}" if not np.isnan(regime_gap) else "Regime gap:            N/A")
    print(f"Filter count:          {best_m['count']}")

    if best_m['sharpe'] > 1.0 and p_value < 0.05 and regime_pass and best_m['count'] >= 30:
        verdict = "ALIVE"
        reason = "Significant edge, regime-agnostic, sufficient count"
    elif best_m['sharpe'] > 0.5 and p_value < 0.10 and best_m['count'] >= 20:
        verdict = "PARTIAL"
        if not regime_pass:
            reason = "Some edge but fails regime gap check -- may be regime-tailored"
        elif p_value >= 0.05:
            reason = "Borderline significance (p < 0.10 but >= 0.05)"
        else:
            reason = "Moderate edge, needs more data or better filter"
    else:
        verdict = "DEAD"
        if p_value >= 0.10:
            reason = "No statistically significant improvement from VIX filter"
        elif best_m['count'] < 20:
            reason = "Insufficient trade count for reliability"
        else:
            reason = "Edge too weak or regime-dependent"

    print(f"\n>>> VERDICT: {verdict}")
    print(f">>> Reason:  {reason}")

    print("\n" + "=" * 80)
    print("STUDY COMPLETE")
    print("=" * 80)


if __name__ == '__main__':
    main()
