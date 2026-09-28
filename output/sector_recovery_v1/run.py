#!/usr/bin/env python3
"""
Sector Recovery Analysis During Asymmetric Fear Conditions
Extends HC #725/#726 findings on stock-level asymmetric setups.

Questions answered:
1. Which sectors LEAD recoveries from fear conditions?
2. Which sectors UNDERPERFORM during recovery?
3. Is there a rotation pattern (defensive -> cyclical)?
4. Does the worst-performing sector INTO fear have the best recovery? (mean reversion)
5. Optimal sector basket to buy in fear conditions

Author: Claude Opus 4.6
Date: 2026-07-21
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
import os
import sys

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/sector_recovery_v1'

# --- Sector ETF tickers ---
SECTOR_ETFS = {
    'XLK': 'Technology',
    'XLF': 'Financials',
    'XLV': 'Healthcare',
    'XLE': 'Energy',
    'XLY': 'Cons. Discr.',
    'XLP': 'Cons. Staples',
    'XLI': 'Industrials',
    'XLB': 'Materials',
    'XLRE': 'Real Estate',
    'XLC': 'Comms',
    'XLU': 'Utilities',
}
SECTOR_TICKERS = list(SECTOR_ETFS.keys())
ALL_TICKERS = SECTOR_TICKERS + ['SPY']
VIX_TICKERS = ['^VIX', '^VIX3M']

FORWARD_WINDOWS = {'1w': 5, '1m': 21, '3m': 63}
N_BOOTSTRAP = 5000
CONFIDENCE_LEVEL = 0.95


def download_data():
    """Download all price data 2010-2026."""
    print("Downloading sector ETF + SPY data...")
    prices = yf.download(ALL_TICKERS, start='2010-01-01', end='2026-07-21',
                         auto_adjust=False, progress=False)
    close = prices['Adj Close'] if 'Adj Close' in prices.columns.get_level_values(0) else prices['Close']

    # Handle multi-level columns
    if isinstance(close.columns, pd.MultiIndex):
        close = close.droplevel(0, axis=1) if close.columns.nlevels > 1 else close

    print(f"  Price data: {close.shape[0]} days, {close.shape[1]} tickers")

    print("Downloading VIX + VIX3M...")
    vix_data = yf.download(VIX_TICKERS, start='2010-01-01', end='2026-07-21',
                           auto_adjust=False, progress=False)
    vix_close = vix_data['Adj Close'] if 'Adj Close' in vix_data.columns.get_level_values(0) else vix_data['Close']
    if isinstance(vix_close.columns, pd.MultiIndex):
        vix_close = vix_close.droplevel(0, axis=1) if vix_close.columns.nlevels > 1 else vix_close

    # Rename VIX columns
    vix_close = vix_close.rename(columns={'^VIX': 'VIX', '^VIX3M': 'VIX3M'})
    print(f"  VIX data: {vix_close.shape[0]} days")

    # Align on common dates
    combined = close.join(vix_close, how='inner').dropna()
    print(f"  Combined: {combined.shape[0]} days after alignment")
    return combined


def compute_fear_conditions(df):
    """
    Define fear conditions (T-1 signals):
    1. VIX > 25
    2. VIX backwardation (VIX/VIX3M > 1.0)
    3. SPY below 50d SMA
    4. At least 6 of 11 sectors below their 50d SMA (breadth collapse)
    Any 2 of 4 = fear regime active.
    """
    print("\nComputing fear conditions...")

    # Condition 1: VIX > 25
    c1 = df['VIX'] > 25

    # Condition 2: VIX backwardation
    c2 = (df['VIX'] / df['VIX3M']) > 1.0

    # Condition 3: SPY below 50d SMA
    spy_sma50 = df['SPY'].rolling(50).mean()
    c3 = df['SPY'] < spy_sma50

    # Condition 4: Breadth collapse - 6+ sectors below 50d SMA
    sectors_below_sma = pd.DataFrame()
    for ticker in SECTOR_TICKERS:
        if ticker in df.columns:
            sma50 = df[ticker].rolling(50).mean()
            sectors_below_sma[ticker] = df[ticker] < sma50
    n_below = sectors_below_sma.sum(axis=1)
    c4 = n_below >= 6

    # Any 2 of 4 = fear regime
    condition_sum = c1.astype(int) + c2.astype(int) + c3.astype(int) + c4.astype(int)
    fear_active = condition_sum >= 2

    # Shift to T-1 (signals known at close of prior day)
    fear_active = fear_active.shift(1).fillna(False)

    print(f"  Condition 1 (VIX>25): {c1.sum()} days")
    print(f"  Condition 2 (VIX backwardation): {c2.sum()} days")
    print(f"  Condition 3 (SPY < 50d SMA): {c3.sum()} days")
    print(f"  Condition 4 (breadth collapse): {c4.sum()} days")
    print(f"  Fear regime active (any 2 of 4): {fear_active.sum()} days")

    return fear_active, {'c1': c1, 'c2': c2, 'c3': c3, 'c4': c4}


def find_fear_entries(fear_active, min_gap_days=10):
    """
    Find ENTRY points into fear regime (first day conditions activate).
    Require min_gap_days between entries to avoid counting the same episode twice.
    """
    entries = []
    fear_dates = fear_active[fear_active].index

    if len(fear_dates) == 0:
        return entries

    entries.append(fear_dates[0])
    for d in fear_dates[1:]:
        if (d - entries[-1]).days >= min_gap_days:
            entries.append(d)

    print(f"  Fear regime entries (min gap {min_gap_days}d): {len(entries)} episodes")
    return entries


def compute_forward_returns(df, entries, tickers):
    """Compute forward returns for each entry date and ticker."""
    results = []

    for entry_date in entries:
        entry_idx = df.index.get_loc(entry_date)

        # Prior 1m return (going INTO fear)
        prior_1m_idx = max(0, entry_idx - 21)

        for ticker in tickers:
            if ticker not in df.columns:
                continue

            row = {
                'entry_date': entry_date,
                'ticker': ticker,
                'entry_price': df[ticker].iloc[entry_idx],
            }

            # Prior 1m return
            if entry_idx >= 21:
                row['prior_1m_ret'] = df[ticker].iloc[entry_idx] / df[ticker].iloc[prior_1m_idx] - 1
            else:
                row['prior_1m_ret'] = np.nan

            # Forward returns
            for label, days in FORWARD_WINDOWS.items():
                fwd_idx = entry_idx + days
                if fwd_idx < len(df):
                    row[f'fwd_{label}'] = df[ticker].iloc[fwd_idx] / df[ticker].iloc[entry_idx] - 1
                else:
                    row[f'fwd_{label}'] = np.nan

            results.append(row)

    return pd.DataFrame(results)


def bootstrap_ci(data, n_boot=N_BOOTSTRAP, ci=CONFIDENCE_LEVEL):
    """Bootstrap confidence interval for the mean."""
    data = data.dropna()
    if len(data) < 3:
        return np.nan, np.nan, np.nan

    boot_means = np.array([
        np.mean(np.random.choice(data.values, size=len(data), replace=True))
        for _ in range(n_boot)
    ])

    alpha = (1 - ci) / 2
    lo = np.percentile(boot_means, alpha * 100)
    hi = np.percentile(boot_means, (1 - alpha) * 100)
    return np.mean(data), lo, hi


def analyze_sector_recovery(fwd_df):
    """Main analysis: sector-level forward returns from fear entries."""
    print("\n" + "="*80)
    print("SECTOR RECOVERY ANALYSIS DURING FEAR CONDITIONS")
    print("="*80)

    report_lines = []
    report_lines.append("="*80)
    report_lines.append("SECTOR ROTATION DURING ASYMMETRIC FEAR CONDITIONS")
    report_lines.append(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    report_lines.append("Extends HC #725/#726 findings")
    report_lines.append("="*80)

    n_episodes = fwd_df['entry_date'].nunique()
    date_range = f"{fwd_df['entry_date'].min().strftime('%Y-%m-%d')} to {fwd_df['entry_date'].max().strftime('%Y-%m-%d')}"
    report_lines.append(f"\nTotal fear regime entries: {n_episodes}")
    report_lines.append(f"Date range: {date_range}")
    report_lines.append(f"Entry dates: {sorted(fwd_df['entry_date'].unique())[:5]}... (showing first 5)")

    # ---- Q1: Which sectors LEAD the recovery? ----
    report_lines.append("\n" + "="*80)
    report_lines.append("Q1: SECTOR FORWARD RETURNS FROM FEAR ENTRY (ranked by 1m return)")
    report_lines.append("="*80)

    all_sector_stats = []

    for horizon in ['1w', '1m', '3m']:
        report_lines.append(f"\n--- Forward {horizon} Returns ---")
        report_lines.append(f"{'Sector':<18} {'Mean':>8} {'95% CI':>20} {'HR':>8} {'Up/Down':>8} {'N':>5}")
        report_lines.append("-" * 70)

        sector_means = {}

        for ticker in SECTOR_TICKERS + ['SPY']:
            mask = fwd_df['ticker'] == ticker
            data = fwd_df.loc[mask, f'fwd_{horizon}'].dropna()

            if len(data) < 5:
                continue

            mean_ret, ci_lo, ci_hi = bootstrap_ci(data)
            hit_rate = (data > 0).mean()
            up_mean = data[data > 0].mean() if (data > 0).any() else 0
            down_mean = abs(data[data <= 0].mean()) if (data <= 0).any() else 1e-9
            up_down = up_mean / down_mean if down_mean > 0 else np.inf

            label = SECTOR_ETFS.get(ticker, ticker)
            sector_means[ticker] = mean_ret

            all_sector_stats.append({
                'ticker': ticker,
                'sector': label,
                'horizon': horizon,
                'mean_ret': mean_ret,
                'ci_lo': ci_lo,
                'ci_hi': ci_hi,
                'hit_rate': hit_rate,
                'up_down': up_down,
                'n': len(data)
            })

            report_lines.append(
                f"{label:<18} {mean_ret:>+7.2%} [{ci_lo:>+7.2%}, {ci_hi:>+7.2%}] {hit_rate:>7.1%} {up_down:>7.2f}x  {len(data):>4}"
            )

        # Rank by mean return
        ranked = sorted(sector_means.items(), key=lambda x: x[1], reverse=True)
        report_lines.append(f"\n  Ranking ({horizon}): " + " > ".join(
            [f"{SECTOR_ETFS.get(t, t)} ({r:+.1%})" for t, r in ranked[:5]]
        ))

    stats_df = pd.DataFrame(all_sector_stats)

    # ---- Q2: Sector alpha vs SPY ----
    report_lines.append("\n" + "="*80)
    report_lines.append("Q2: SECTOR ALPHA vs SPY (sector return - SPY return)")
    report_lines.append("="*80)

    for horizon in ['1w', '1m', '3m']:
        report_lines.append(f"\n--- Alpha at {horizon} ---")

        spy_returns = fwd_df[fwd_df['ticker'] == 'SPY'].set_index('entry_date')[f'fwd_{horizon}']

        alpha_results = {}
        for ticker in SECTOR_TICKERS:
            sector_returns = fwd_df[fwd_df['ticker'] == ticker].set_index('entry_date')[f'fwd_{horizon}']
            # Align
            common = spy_returns.index.intersection(sector_returns.index)
            if len(common) < 5:
                continue
            alpha = sector_returns[common] - spy_returns[common]
            mean_alpha, ci_lo, ci_hi = bootstrap_ci(alpha)
            alpha_results[ticker] = {
                'mean': mean_alpha, 'ci_lo': ci_lo, 'ci_hi': ci_hi,
                'pct_positive': (alpha > 0).mean()
            }

        # Sort by alpha
        sorted_alpha = sorted(alpha_results.items(), key=lambda x: x[1]['mean'], reverse=True)
        report_lines.append(f"{'Sector':<18} {'Alpha':>8} {'95% CI':>20} {'% Positive':>12}")
        report_lines.append("-" * 60)
        for ticker, a in sorted_alpha:
            label = SECTOR_ETFS.get(ticker, ticker)
            report_lines.append(
                f"{label:<18} {a['mean']:>+7.2%} [{a['ci_lo']:>+7.2%}, {a['ci_hi']:>+7.2%}] {a['pct_positive']:>11.1%}"
            )

    # ---- Q3: Rotation pattern (defensive -> cyclical?) ----
    report_lines.append("\n" + "="*80)
    report_lines.append("Q3: ROTATION PATTERN - Do defensives lead early, cyclicals later?")
    report_lines.append("="*80)

    DEFENSIVE = ['XLP', 'XLU', 'XLV']  # Staples, Utilities, Healthcare
    CYCLICAL = ['XLY', 'XLK', 'XLI', 'XLF', 'XLE']  # Discr, Tech, Industrials, Financials, Energy

    for horizon in ['1w', '1m', '3m']:
        def_rets = fwd_df[(fwd_df['ticker'].isin(DEFENSIVE))][f'fwd_{horizon}'].dropna()
        cyc_rets = fwd_df[(fwd_df['ticker'].isin(CYCLICAL))][f'fwd_{horizon}'].dropna()

        def_mean = def_rets.mean()
        cyc_mean = cyc_rets.mean()

        report_lines.append(f"  {horizon}: Defensive avg={def_mean:+.2%}  Cyclical avg={cyc_mean:+.2%}  "
                          f"Diff (Cyc-Def)={cyc_mean - def_mean:+.2%}")

    # Per-episode rotation tracking
    report_lines.append("\n  Per-episode: Does defensive leadership at 1w predict cyclical leadership at 3m?")

    rotation_count = 0
    total_episodes = 0

    for entry_date in fwd_df['entry_date'].unique():
        ep = fwd_df[fwd_df['entry_date'] == entry_date]

        def_1w = ep[ep['ticker'].isin(DEFENSIVE)]['fwd_1w'].mean()
        cyc_1w = ep[ep['ticker'].isin(CYCLICAL)]['fwd_1w'].mean()
        def_3m = ep[ep['ticker'].isin(DEFENSIVE)]['fwd_3m'].mean()
        cyc_3m = ep[ep['ticker'].isin(CYCLICAL)]['fwd_3m'].mean()

        if pd.notna(def_1w) and pd.notna(cyc_3m):
            total_episodes += 1
            # Rotation = defensives lead at 1w but cyclicals lead at 3m
            if def_1w > cyc_1w and cyc_3m > def_3m:
                rotation_count += 1

    if total_episodes > 0:
        report_lines.append(f"  Rotation pattern (def leads 1w, cyc leads 3m): {rotation_count}/{total_episodes} episodes ({rotation_count/total_episodes:.0%})")

    # ---- Q4: Mean reversion by sector ----
    report_lines.append("\n" + "="*80)
    report_lines.append("Q4: MEAN REVERSION - Does worst sector INTO fear have best recovery?")
    report_lines.append("="*80)

    # For each episode, rank sectors by prior_1m_ret and forward_1m/3m
    reversal_corrs = {'1m': [], '3m': []}

    for entry_date in fwd_df['entry_date'].unique():
        ep = fwd_df[(fwd_df['entry_date'] == entry_date) & (fwd_df['ticker'].isin(SECTOR_TICKERS))]

        if len(ep) < 8:
            continue

        for horizon in ['1m', '3m']:
            valid = ep[['ticker', 'prior_1m_ret', f'fwd_{horizon}']].dropna()
            if len(valid) >= 6:
                corr = valid['prior_1m_ret'].corr(valid[f'fwd_{horizon}'])
                reversal_corrs[horizon].append(corr)

    for horizon in ['1m', '3m']:
        corrs = np.array(reversal_corrs[horizon])
        if len(corrs) > 0:
            report_lines.append(
                f"  Correlation(prior_1m_ret, fwd_{horizon}_ret) across episodes:")
            report_lines.append(
                f"    Mean corr: {corrs.mean():+.3f}  Median: {np.median(corrs):+.3f}  "
                f"% negative: {(corrs < 0).mean():.0%}  N={len(corrs)}")
            report_lines.append(
                f"    (Negative = mean reversion: worst prior performers recover most)")

    # Quintile analysis: bottom 3 vs top 3 sectors by prior 1m return
    report_lines.append("\n  Quintile analysis: Buy WORST 3 vs BEST 3 sectors entering fear")

    for horizon in ['1m', '3m']:
        worst3_rets = []
        best3_rets = []

        for entry_date in fwd_df['entry_date'].unique():
            ep = fwd_df[(fwd_df['entry_date'] == entry_date) & (fwd_df['ticker'].isin(SECTOR_TICKERS))]
            valid = ep[['ticker', 'prior_1m_ret', f'fwd_{horizon}']].dropna()

            if len(valid) >= 8:
                sorted_ep = valid.sort_values('prior_1m_ret')
                worst3_rets.append(sorted_ep.head(3)[f'fwd_{horizon}'].mean())
                best3_rets.append(sorted_ep.tail(3)[f'fwd_{horizon}'].mean())

        if worst3_rets:
            w3 = np.array(worst3_rets)
            b3 = np.array(best3_rets)
            report_lines.append(
                f"\n  {horizon}: Worst 3 sectors avg fwd return: {w3.mean():+.2%} (HR: {(w3>0).mean():.0%})")
            report_lines.append(
                f"  {horizon}: Best 3 sectors avg fwd return:  {b3.mean():+.2%} (HR: {(b3>0).mean():.0%})")
            report_lines.append(
                f"  {horizon}: Worst-Best spread: {(w3-b3).mean():+.2%}")

    # ---- Q5: Optimal basket ----
    report_lines.append("\n" + "="*80)
    report_lines.append("Q5: OPTIMAL SECTOR BASKET IN FEAR CONDITIONS")
    report_lines.append("="*80)

    # Compare strategies
    strategies = {}

    for horizon in ['1m', '3m']:
        report_lines.append(f"\n--- {horizon} horizon ---")

        # Strategy 1: Buy SPY
        spy_rets = fwd_df[fwd_df['ticker'] == 'SPY'][f'fwd_{horizon}'].dropna()
        strategies[f'SPY_{horizon}'] = spy_rets

        # Strategy 2: Equal-weight all sectors
        all_sector_rets = []
        for entry_date in fwd_df['entry_date'].unique():
            ep = fwd_df[(fwd_df['entry_date'] == entry_date) & (fwd_df['ticker'].isin(SECTOR_TICKERS))]
            r = ep[f'fwd_{horizon}'].dropna().mean()
            if pd.notna(r):
                all_sector_rets.append(r)
        all_sector_rets = pd.Series(all_sector_rets)

        # Strategy 3: Best 3 sectors historically (determined from 1m stats)
        # Use in-sample top 3 for now, then we'll do proper OOS
        sector_1m_means = stats_df[(stats_df['horizon'] == '1m') &
                                    (stats_df['ticker'].isin(SECTOR_TICKERS))].sort_values('mean_ret', ascending=False)
        top3_tickers = sector_1m_means.head(3)['ticker'].tolist()

        top3_rets = []
        for entry_date in fwd_df['entry_date'].unique():
            ep = fwd_df[(fwd_df['entry_date'] == entry_date) & (fwd_df['ticker'].isin(top3_tickers))]
            r = ep[f'fwd_{horizon}'].dropna().mean()
            if pd.notna(r):
                top3_rets.append(r)
        top3_rets = pd.Series(top3_rets)

        # Strategy 4: Worst 3 prior-month sectors (mean reversion)
        worst3_rets_strat = []
        for entry_date in fwd_df['entry_date'].unique():
            ep = fwd_df[(fwd_df['entry_date'] == entry_date) & (fwd_df['ticker'].isin(SECTOR_TICKERS))]
            valid = ep[['ticker', 'prior_1m_ret', f'fwd_{horizon}']].dropna()
            if len(valid) >= 8:
                sorted_ep = valid.sort_values('prior_1m_ret')
                worst3_rets_strat.append(sorted_ep.head(3)[f'fwd_{horizon}'].mean())
        worst3_rets_strat = pd.Series(worst3_rets_strat)

        # Strategy 5: Defensive basket only
        def_rets_strat = []
        for entry_date in fwd_df['entry_date'].unique():
            ep = fwd_df[(fwd_df['entry_date'] == entry_date) & (fwd_df['ticker'].isin(DEFENSIVE))]
            r = ep[f'fwd_{horizon}'].dropna().mean()
            if pd.notna(r):
                def_rets_strat.append(r)
        def_rets_strat = pd.Series(def_rets_strat)

        report_lines.append(f"{'Strategy':<30} {'Mean':>8} {'HR':>8} {'Up/Dn':>8} {'Sharpe*':>8} {'N':>5}")
        report_lines.append("-" * 70)

        for name, rets in [
            ('Buy SPY', spy_rets),
            ('EW All Sectors', all_sector_rets),
            (f'Top 3 Sectors ({",".join(top3_tickers)})', top3_rets),
            ('Worst 3 Prior-Mo Sectors', worst3_rets_strat),
            ('Defensive Only (XLP,XLU,XLV)', def_rets_strat),
        ]:
            if len(rets) < 3:
                continue
            mean_r = rets.mean()
            hr = (rets > 0).mean()
            up = rets[rets > 0].mean() if (rets > 0).any() else 0
            dn = abs(rets[rets <= 0].mean()) if (rets <= 0).any() else 1e-9
            ud = up / dn
            sharpe = mean_r / rets.std() if rets.std() > 0 else 0

            report_lines.append(
                f"{name:<30} {mean_r:>+7.2%} {hr:>7.0%} {ud:>7.2f}x {sharpe:>7.2f}  {len(rets):>4}"
            )

        report_lines.append(f"\n  Top 3 sectors (by avg 1m fwd return): {', '.join([SECTOR_ETFS.get(t,t) for t in top3_tickers])}")

    # ---- Q6: Cross-reference with ML ranker findings ----
    report_lines.append("\n" + "="*80)
    report_lines.append("Q6: CROSS-REFERENCE WITH ML RANKER (sector_ret_1m = #3 feature)")
    report_lines.append("="*80)

    report_lines.append("\nWhich sectors' prior 1m returns are most predictive of forward returns?")
    report_lines.append("(High |correlation| = sector's prior momentum is informative for recovery)")

    for ticker in SECTOR_TICKERS:
        mask = fwd_df['ticker'] == ticker
        valid = fwd_df.loc[mask, ['prior_1m_ret', 'fwd_1m']].dropna()
        if len(valid) >= 10:
            corr = valid['prior_1m_ret'].corr(valid['fwd_1m'])
            label = SECTOR_ETFS.get(ticker, ticker)
            report_lines.append(f"  {label:<18} corr(prior_1m, fwd_1m) = {corr:+.3f}  (N={len(valid)})")

    # ---- Episode timeline ----
    report_lines.append("\n" + "="*80)
    report_lines.append("APPENDIX: FEAR REGIME ENTRY DATES")
    report_lines.append("="*80)

    for entry_date in sorted(fwd_df['entry_date'].unique()):
        ep = fwd_df[(fwd_df['entry_date'] == entry_date) & (fwd_df['ticker'] == 'SPY')]
        if len(ep) > 0:
            spy_1m = ep['fwd_1m'].values[0]
            spy_3m = ep['fwd_3m'].values[0]
            spy_str = f"SPY fwd_1m={spy_1m:+.1%}" if pd.notna(spy_1m) else "SPY fwd_1m=N/A"
            spy_str += f"  fwd_3m={spy_3m:+.1%}" if pd.notna(spy_3m) else "  fwd_3m=N/A"
            report_lines.append(f"  {entry_date.strftime('%Y-%m-%d')}  {spy_str}")

    return report_lines, stats_df, top3_tickers


def save_heatmap_csv(fwd_df, stats_df):
    """Save sector recovery heatmap data to CSV."""
    # Pivot: sectors x horizons
    heatmap_data = []

    for ticker in SECTOR_TICKERS + ['SPY']:
        row = {'ticker': ticker, 'sector': SECTOR_ETFS.get(ticker, ticker)}
        for horizon in ['1w', '1m', '3m']:
            mask = (stats_df['ticker'] == ticker) & (stats_df['horizon'] == horizon)
            if mask.any():
                row[f'mean_ret_{horizon}'] = stats_df.loc[mask, 'mean_ret'].values[0]
                row[f'hit_rate_{horizon}'] = stats_df.loc[mask, 'hit_rate'].values[0]
                row[f'up_down_{horizon}'] = stats_df.loc[mask, 'up_down'].values[0]
                row[f'ci_lo_{horizon}'] = stats_df.loc[mask, 'ci_lo'].values[0]
                row[f'ci_hi_{horizon}'] = stats_df.loc[mask, 'ci_hi'].values[0]
                row[f'n_{horizon}'] = stats_df.loc[mask, 'n'].values[0]
        heatmap_data.append(row)

    heatmap_df = pd.DataFrame(heatmap_data)
    csv_path = os.path.join(OUTPUT_DIR, 'sector_recovery_heatmap.csv')
    heatmap_df.to_csv(csv_path, index=False)
    print(f"\nSaved heatmap CSV to {csv_path}")

    # Also save raw forward returns
    raw_path = os.path.join(OUTPUT_DIR, 'sector_forward_returns_raw.csv')
    fwd_df.to_csv(raw_path, index=False)
    print(f"Saved raw forward returns to {raw_path}")

    return heatmap_df


def main():
    np.random.seed(42)

    # Step 1: Download data
    df = download_data()

    # Step 2: Compute fear conditions
    fear_active, conditions = compute_fear_conditions(df)

    # Step 3: Find fear regime entries
    entries = find_fear_entries(fear_active, min_gap_days=10)

    if len(entries) < 5:
        print("ERROR: Too few fear regime entries. Check data.")
        return

    # Step 4: Compute forward returns
    print(f"\nComputing forward returns for {len(entries)} episodes x {len(ALL_TICKERS)} tickers...")
    fwd_df = compute_forward_returns(df, entries, ALL_TICKERS)
    print(f"  Forward returns table: {fwd_df.shape}")

    # Step 5: Full analysis
    report_lines, stats_df, top3 = analyze_sector_recovery(fwd_df)

    # Step 6: Save outputs
    report_text = '\n'.join(report_lines)

    report_path = os.path.join(OUTPUT_DIR, 'summary_report.txt')
    with open(report_path, 'w') as f:
        f.write(report_text)
    print(f"\nSaved report to {report_path}")

    heatmap_df = save_heatmap_csv(fwd_df, stats_df)

    # Print report to stdout
    print("\n" + report_text)

    print("\n\nDONE. Analysis complete.")


if __name__ == '__main__':
    main()
