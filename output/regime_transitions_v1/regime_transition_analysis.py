#!/usr/bin/env python3
"""
Regime Transition Analysis — Comprehensive study of market regime shifts
and leading indicators for asymmetric opportunities.

Author: Claude (Head of Quant)
Date: 2026-07-21
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
from pathlib import Path
from scipy import stats

OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/regime_transitions_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. DATA DOWNLOAD
# ============================================================
def download_data():
    """Download all required tickers from yfinance."""

    tickers_main = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'HYG', 'LQD', 'EEM', 'EFA', 'VNQ']
    tickers_vix = ['^VIX', '^VIX3M']
    sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']

    all_tickers = tickers_main + tickers_vix + sectors

    print(f"Downloading {len(all_tickers)} tickers from 2005-01-01 to 2026-07-21...")

    data = {}
    for ticker in all_tickers:
        try:
            df = yf.download(ticker, start='2005-01-01', end='2026-07-21', progress=False, auto_adjust=True)
            # Flatten MultiIndex columns if present (yfinance 1.x returns ('Close', 'SPY'))
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            if len(df) > 100:
                data[ticker] = df
                print(f"  {ticker}: {len(df)} rows ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")
            else:
                print(f"  {ticker}: SKIPPED (only {len(df)} rows)")
        except Exception as e:
            print(f"  {ticker}: ERROR - {e}")

    return data

# ============================================================
# 2. REGIME CLASSIFICATION (T-1 data only)
# ============================================================
def classify_regimes(data):
    """
    Define 4 market regimes using T-1 data only:
    - BULL: SPY above 200d SMA, VIX < 20
    - CAUTION: Mixed signals
    - BEAR: SPY below 200d SMA, VIX >= 20
    - CRISIS: SPY below 200d SMA, VIX >= 30
    """
    spy = data['SPY']['Close'].copy()
    vix = data['^VIX']['Close'].copy()

    # Align dates
    common_idx = spy.index.intersection(vix.index)
    spy = spy.loc[common_idx]
    vix = vix.loc[common_idx]

    # 200d SMA (using T-1 data: shift by 1)
    spy_sma200 = spy.rolling(200).mean().shift(1)
    vix_prev = vix.shift(1)
    spy_prev = spy.shift(1)

    # Regime classification
    regime = pd.Series('UNKNOWN', index=common_idx)

    above_sma = spy_prev > spy_sma200
    below_sma = spy_prev <= spy_sma200

    # CRISIS: below SMA, VIX >= 30
    regime[below_sma & (vix_prev >= 30)] = 'CRISIS'
    # BEAR: below SMA, VIX >= 20 (but < 30)
    regime[below_sma & (vix_prev >= 20) & (vix_prev < 30)] = 'BEAR'
    # CAUTION: mixed signals
    regime[(above_sma & (vix_prev >= 20)) | (below_sma & (vix_prev < 20))] = 'CAUTION'
    # BULL: above SMA, VIX < 20
    regime[above_sma & (vix_prev < 20)] = 'BULL'

    # Drop initial NaN period
    regime = regime[regime != 'UNKNOWN']

    print(f"\nRegime distribution ({regime.index[0].strftime('%Y-%m-%d')} to {regime.index[-1].strftime('%Y-%m-%d')}):")
    counts = regime.value_counts()
    for r in ['BULL', 'CAUTION', 'BEAR', 'CRISIS']:
        if r in counts:
            pct = counts[r] / len(regime) * 100
            print(f"  {r}: {counts[r]} days ({pct:.1f}%)")

    return regime, spy, vix, spy_sma200

# ============================================================
# 3. MAP REGIME TRANSITIONS
# ============================================================
def map_transitions(regime):
    """Identify all regime transition dates and types."""

    transitions = []
    prev_regime = regime.iloc[0]

    for i in range(1, len(regime)):
        curr_regime = regime.iloc[i]
        if curr_regime != prev_regime:
            transitions.append({
                'date': regime.index[i],
                'from_regime': prev_regime,
                'to_regime': curr_regime,
                'transition': f"{prev_regime} -> {curr_regime}",
                'days_in_prev': None  # filled below
            })
        prev_regime = curr_regime

    # Calculate days spent in previous regime
    for i, t in enumerate(transitions):
        if i == 0:
            t['days_in_prev'] = (t['date'] - regime.index[0]).days
        else:
            t['days_in_prev'] = (t['date'] - transitions[i-1]['date']).days

    df = pd.DataFrame(transitions)

    print(f"\nTotal transitions: {len(df)}")
    print("\nTransition type counts:")
    print(df['transition'].value_counts().to_string())

    return df

# ============================================================
# 4. COMPUTE LEADING INDICATORS
# ============================================================
def compute_leading_indicators(data, transitions_df):
    """
    For each transition, compute leading indicators at -20, -10, -5 days.
    Returns a dataframe with signal values before each transition.
    """
    spy = data['SPY']['Close']
    vix = data['^VIX']['Close']

    # Compute indicator time series
    indicators = {}

    # 1. VIX Term Structure (VIX/VIX3M — >1 = backwardation = fear)
    if '^VIX3M' in data:
        vix3m = data['^VIX3M']['Close']
        common = vix.index.intersection(vix3m.index)
        vix_ts = (vix.loc[common] / vix3m.loc[common])
        indicators['vix_term_structure'] = vix_ts

    # 2. Credit spread proxy (HYG/LQD ratio — declining = widening spreads)
    if 'HYG' in data and 'LQD' in data:
        hyg = data['HYG']['Close']
        lqd = data['LQD']['Close']
        common = hyg.index.intersection(lqd.index)
        credit = (hyg.loc[common] / lqd.loc[common])
        # Z-score over 60d rolling
        credit_z = (credit - credit.rolling(60).mean()) / credit.rolling(60).std()
        indicators['credit_spread_z'] = credit_z
        indicators['hyg_lqd_ratio'] = credit

    # 3. Breadth: % of sector ETFs above 50d SMA
    sectors = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLY', 'XLP', 'XLU', 'XLB']
    available_sectors = [s for s in sectors if s in data]
    if len(available_sectors) >= 5:
        # Build breadth on common dates
        sector_above = []
        for s in available_sectors:
            close = data[s]['Close']
            sma50 = close.rolling(50).mean()
            above = (close > sma50).astype(float)
            sector_above.append(above)

        breadth_df = pd.concat(sector_above, axis=1)
        breadth = breadth_df.mean(axis=1)  # % above 50d SMA
        indicators['breadth_pct'] = breadth

    # 4. Small-cap divergence (IWM vs SPY relative performance, 20d)
    if 'IWM' in data:
        iwm = data['IWM']['Close']
        common = iwm.index.intersection(spy.index)
        iwm_rel = (iwm.loc[common].pct_change(20) - spy.loc[common].pct_change(20))
        indicators['smallcap_divergence_20d'] = iwm_rel

    # 5. Gold relative strength (GLD vs SPY, 20d)
    if 'GLD' in data:
        gld = data['GLD']['Close']
        common = gld.index.intersection(spy.index)
        gold_rel = (gld.loc[common].pct_change(20) - spy.loc[common].pct_change(20))
        indicators['gold_rel_strength_20d'] = gold_rel

    # 6. VIX vs realized vol (VIX level minus 20d realized vol annualized)
    spy_ret = spy.pct_change()
    realized_vol = spy_ret.rolling(20).std() * np.sqrt(252) * 100
    common = vix.index.intersection(realized_vol.index)
    vix_rv_spread = vix.loc[common] - realized_vol.loc[common]
    indicators['vix_minus_realized_vol'] = vix_rv_spread

    # 7. SPY momentum (20d return)
    spy_mom_20 = spy.pct_change(20)
    indicators['spy_momentum_20d'] = spy_mom_20

    # 8. TLT relative performance (flight to quality)
    if 'TLT' in data:
        tlt = data['TLT']['Close']
        common = tlt.index.intersection(spy.index)
        tlt_rel = (tlt.loc[common].pct_change(20) - spy.loc[common].pct_change(20))
        indicators['tlt_rel_strength_20d'] = tlt_rel

    # Now for each transition, look back at -20, -10, -5 days
    lookbacks = [5, 10, 20]
    results = []

    for _, trans in transitions_df.iterrows():
        t_date = trans['date']
        row = {
            'date': t_date,
            'transition': trans['transition'],
            'from_regime': trans['from_regime'],
            'to_regime': trans['to_regime'],
            'days_in_prev': trans['days_in_prev'],
        }

        for ind_name, ind_series in indicators.items():
            # Value at transition date
            if t_date in ind_series.index:
                row[f'{ind_name}_at_transition'] = ind_series.loc[t_date]

            for lb in lookbacks:
                # Find the date ~lb trading days before transition
                idx_pos = ind_series.index.get_indexer([t_date], method='pad')[0]
                if idx_pos >= lb and idx_pos >= 0:
                    lb_date = ind_series.index[idx_pos - lb]
                    val_before = ind_series.iloc[idx_pos - lb]
                    val_at = ind_series.iloc[idx_pos] if idx_pos < len(ind_series) else np.nan

                    row[f'{ind_name}_t-{lb}'] = val_before
                    row[f'{ind_name}_chg_{lb}d'] = val_at - val_before if not np.isnan(val_at) else np.nan

        results.append(row)

    return pd.DataFrame(results), indicators

# ============================================================
# 5. FORWARD RETURNS AFTER TRANSITIONS
# ============================================================
def compute_forward_returns(data, transitions_df):
    """Compute forward returns after each regime transition."""

    spy = data['SPY']['Close']
    forward_periods = [5, 10, 21, 63, 126, 252]  # ~1w, 2w, 1m, 3m, 6m, 12m
    labels = ['1w', '2w', '1m', '3m', '6m', '12m']

    results = []

    for _, trans in transitions_df.iterrows():
        t_date = trans['date']
        row = {
            'date': t_date,
            'transition': trans['transition'],
            'from_regime': trans['from_regime'],
            'to_regime': trans['to_regime'],
        }

        idx_pos = spy.index.get_indexer([t_date], method='pad')[0]
        if idx_pos < 0:
            continue

        entry_price = spy.iloc[idx_pos]

        for period, label in zip(forward_periods, labels):
            exit_idx = idx_pos + period
            if exit_idx < len(spy):
                exit_price = spy.iloc[exit_idx]
                fwd_ret = (exit_price / entry_price - 1) * 100
                row[f'fwd_{label}_pct'] = fwd_ret
            else:
                row[f'fwd_{label}_pct'] = np.nan

        # Also compute max drawdown in next 63 days (3m)
        if idx_pos + 63 < len(spy):
            future_prices = spy.iloc[idx_pos:idx_pos+63]
            running_max = future_prices.cummax()
            drawdowns = (future_prices / running_max - 1) * 100
            row['max_dd_3m_pct'] = drawdowns.min()

            # Max runup
            running_min = future_prices.cummin()
            runups = (future_prices / running_min - 1) * 100
            row['max_runup_3m_pct'] = runups.max()

        results.append(row)

    return pd.DataFrame(results)

# ============================================================
# 6. SIGNAL LEAD TIME ANALYSIS
# ============================================================
def analyze_signal_lead_times(indicators, transitions_df, regime_series):
    """
    For each transition type, determine how many days before the transition
    each indicator started showing deterioration/improvement.
    """

    # Focus on the key transitions
    key_transitions = [
        'BULL -> CAUTION',
        'CAUTION -> BEAR',
        'CAUTION -> CRISIS',
        'BEAR -> CAUTION',
        'CRISIS -> BEAR',
        'CRISIS -> CAUTION',
        'CAUTION -> BULL',
        'BEAR -> BULL',
    ]

    results = []

    for trans_type in key_transitions:
        trans_dates = transitions_df[transitions_df['transition'] == trans_type]['date'].values

        if len(trans_dates) == 0:
            continue

        for ind_name, ind_series in indicators.items():
            lead_times = []

            for t_date in trans_dates:
                t_date_ts = pd.Timestamp(t_date)
                idx_pos = ind_series.index.get_indexer([t_date_ts], method='pad')[0]

                if idx_pos < 40:
                    continue

                # Get the value at transition and look back 40 days
                val_at_trans = ind_series.iloc[idx_pos]

                # Compute z-score of indicator over past 60 days before the lookback window
                baseline_start = max(0, idx_pos - 100)
                baseline_end = idx_pos - 40
                if baseline_end <= baseline_start:
                    continue

                baseline = ind_series.iloc[baseline_start:baseline_end]
                baseline_mean = baseline.mean()
                baseline_std = baseline.std()

                if baseline_std == 0 or np.isnan(baseline_std):
                    continue

                # Find the first day (looking back from transition) where the z-score
                # crossed a threshold — this is the "warning" day
                for lookback in range(1, 41):
                    check_idx = idx_pos - lookback
                    if check_idx < 0:
                        break
                    val = ind_series.iloc[check_idx]
                    z = (val - baseline_mean) / baseline_std

                    # For deterioration signals (toward risk-off):
                    # VIX term structure rising, credit spread falling, breadth falling
                    if 'bull' in trans_type.lower() or 'recovery' in trans_type.lower():
                        # Recovery transition — look for improvement
                        if abs(z) > 1.5:
                            lead_times.append(lookback)
                            break
                    else:
                        if abs(z) > 1.5:
                            lead_times.append(lookback)
                            break

            if lead_times:
                results.append({
                    'transition_type': trans_type,
                    'indicator': ind_name,
                    'n_transitions': len(trans_dates),
                    'n_with_signal': len(lead_times),
                    'signal_rate': len(lead_times) / len(trans_dates),
                    'median_lead_days': np.median(lead_times),
                    'mean_lead_days': np.mean(lead_times),
                    'min_lead_days': np.min(lead_times),
                    'max_lead_days': np.max(lead_times),
                })

    return pd.DataFrame(results)

# ============================================================
# 7. DETAILED TRANSITION EPISODES
# ============================================================
def analyze_major_episodes(transitions_df, forward_returns_df, regime_series):
    """
    Identify and analyze the major market episodes:
    GFC, European Crisis, COVID, 2022 Bear, etc.
    """

    episodes = {
        'GFC': ('2007-10-01', '2009-06-30'),
        'European_Debt_Crisis': ('2011-05-01', '2012-01-31'),
        'China_Devaluation_2015': ('2015-07-01', '2016-03-31'),
        'COVID': ('2020-02-01', '2020-06-30'),
        'Bear_2022': ('2022-01-01', '2022-12-31'),
    }

    episode_analysis = []

    for ep_name, (start, end) in episodes.items():
        start_dt = pd.Timestamp(start)
        end_dt = pd.Timestamp(end)

        ep_trans = transitions_df[
            (transitions_df['date'] >= start_dt) &
            (transitions_df['date'] <= end_dt)
        ]

        ep_regimes = regime_series[
            (regime_series.index >= start_dt) &
            (regime_series.index <= end_dt)
        ]

        episode_analysis.append({
            'episode': ep_name,
            'period': f"{start} to {end}",
            'n_transitions': len(ep_trans),
            'transitions': ep_trans['transition'].tolist() if len(ep_trans) > 0 else [],
            'transition_dates': ep_trans['date'].tolist() if len(ep_trans) > 0 else [],
            'regime_days': ep_regimes.value_counts().to_dict() if len(ep_regimes) > 0 else {},
        })

    return episode_analysis

# ============================================================
# 8. GENERATE SUMMARY REPORT
# ============================================================
def generate_summary_report(regime, transitions_df, forward_returns_df,
                           lead_times_df, episode_analysis, indicators_df):
    """Generate plain English summary report."""

    lines = []
    lines.append("=" * 80)
    lines.append("REGIME TRANSITION ANALYSIS — COMPREHENSIVE REPORT")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"Period: {regime.index[0].strftime('%Y-%m-%d')} to {regime.index[-1].strftime('%Y-%m-%d')}")
    lines.append("=" * 80)

    # Section 1: Regime Distribution
    lines.append("\n" + "=" * 60)
    lines.append("1. REGIME DISTRIBUTION")
    lines.append("=" * 60)

    counts = regime.value_counts()
    total = len(regime)
    for r in ['BULL', 'CAUTION', 'BEAR', 'CRISIS']:
        if r in counts:
            pct = counts[r] / total * 100
            lines.append(f"  {r:10s}: {counts[r]:5d} days ({pct:5.1f}%)")
    lines.append(f"  {'TOTAL':10s}: {total:5d} days")

    # Section 2: Transition Counts
    lines.append("\n" + "=" * 60)
    lines.append("2. REGIME TRANSITIONS — FREQUENCY")
    lines.append("=" * 60)

    trans_counts = transitions_df['transition'].value_counts()
    for t, c in trans_counts.items():
        lines.append(f"  {t:25s}: {c:3d} times")
    lines.append(f"\n  Total transitions: {len(transitions_df)}")
    lines.append(f"  Average transitions per year: {len(transitions_df) / ((regime.index[-1] - regime.index[0]).days / 365.25):.1f}")

    # Section 3: Forward Returns by Transition Type
    lines.append("\n" + "=" * 60)
    lines.append("3. FORWARD RETURNS AFTER EACH TRANSITION TYPE")
    lines.append("   (SPY total return, percent)")
    lines.append("=" * 60)

    key_transitions = [
        'CRISIS -> BEAR', 'CRISIS -> CAUTION', 'BEAR -> CAUTION',
        'BEAR -> BULL', 'CAUTION -> BULL',
        'BULL -> CAUTION', 'CAUTION -> BEAR', 'CAUTION -> CRISIS',
    ]

    for trans_type in key_transitions:
        subset = forward_returns_df[forward_returns_df['transition'] == trans_type]
        if len(subset) == 0:
            continue

        lines.append(f"\n  {trans_type} (n={len(subset)}):")
        lines.append(f"  {'Period':>8s}  {'Mean':>8s}  {'Median':>8s}  {'Min':>8s}  {'Max':>8s}  {'%Pos':>6s}")
        lines.append(f"  {'─' * 50}")

        for col, label in [('fwd_1w_pct', '1 week'), ('fwd_2w_pct', '2 week'),
                          ('fwd_1m_pct', '1 month'), ('fwd_3m_pct', '3 month'),
                          ('fwd_6m_pct', '6 month'), ('fwd_12m_pct', '12 month')]:
            vals = subset[col].dropna()
            if len(vals) > 0:
                pct_pos = (vals > 0).mean() * 100
                lines.append(f"  {label:>8s}  {vals.mean():>+7.2f}%  {vals.median():>+7.2f}%  "
                           f"{vals.min():>+7.2f}%  {vals.max():>+7.2f}%  {pct_pos:>5.1f}%")

        # Asymmetry analysis
        if 'max_dd_3m_pct' in subset.columns and 'max_runup_3m_pct' in subset.columns:
            dd = subset['max_dd_3m_pct'].dropna()
            ru = subset['max_runup_3m_pct'].dropna()
            if len(dd) > 0 and len(ru) > 0:
                lines.append(f"\n  3-month asymmetry:")
                lines.append(f"    Avg max drawdown:  {dd.mean():+.2f}%")
                lines.append(f"    Avg max runup:     {ru.mean():+.2f}%")
                lines.append(f"    Upside/Downside:   {abs(ru.mean()/dd.mean()):.2f}x")

    # Section 4: Leading Indicators
    lines.append("\n" + "=" * 60)
    lines.append("4. LEADING INDICATORS — WHICH SIGNALS WARN EARLIEST?")
    lines.append("=" * 60)

    if len(lead_times_df) > 0:
        for trans_type in key_transitions:
            subset = lead_times_df[lead_times_df['transition_type'] == trans_type]
            if len(subset) == 0:
                continue

            subset_sorted = subset.sort_values('median_lead_days', ascending=False)

            lines.append(f"\n  {trans_type}:")
            lines.append(f"  {'Indicator':>30s}  {'Signal Rate':>11s}  {'Median Lead':>11s}  {'Mean Lead':>9s}")
            lines.append(f"  {'─' * 65}")

            for _, row in subset_sorted.iterrows():
                lines.append(f"  {row['indicator']:>30s}  "
                           f"{row['signal_rate']:>10.0%}  "
                           f"{row['median_lead_days']:>9.1f}d  "
                           f"{row['mean_lead_days']:>7.1f}d")

    # Section 5: Major Episodes
    lines.append("\n" + "=" * 60)
    lines.append("5. MAJOR MARKET EPISODES — TRANSITION SEQUENCES")
    lines.append("=" * 60)

    for ep in episode_analysis:
        lines.append(f"\n  {ep['episode']} ({ep['period']}):")
        lines.append(f"    Transitions: {ep['n_transitions']}")
        if ep['transitions']:
            for t, d in zip(ep['transitions'], ep['transition_dates']):
                d_str = pd.Timestamp(d).strftime('%Y-%m-%d')
                lines.append(f"      {d_str}: {t}")
        if ep['regime_days']:
            lines.append(f"    Regime days: {ep['regime_days']}")

    # Section 6: The Ultimate Question
    lines.append("\n" + "=" * 60)
    lines.append("6. THE ULTIMATE QUESTION: CRISIS -> RECOVERY ENTRY")
    lines.append("=" * 60)

    # Find CRISIS exits
    crisis_exits = forward_returns_df[
        forward_returns_df['from_regime'] == 'CRISIS'
    ]

    if len(crisis_exits) > 0:
        lines.append(f"\n  Total CRISIS exit transitions: {len(crisis_exits)}")
        lines.append(f"\n  Forward returns after exiting CRISIS regime:")

        for col, label in [('fwd_1m_pct', '1 month'), ('fwd_3m_pct', '3 month'),
                          ('fwd_6m_pct', '6 month'), ('fwd_12m_pct', '12 month')]:
            vals = crisis_exits[col].dropna()
            if len(vals) > 0:
                pct_pos = (vals > 0).mean() * 100
                lines.append(f"    {label:>8s}: mean={vals.mean():+.2f}%, "
                           f"median={vals.median():+.2f}%, "
                           f"worst={vals.min():+.2f}%, "
                           f"best={vals.max():+.2f}%, "
                           f"win_rate={pct_pos:.0f}%")

        if 'max_dd_3m_pct' in crisis_exits.columns:
            dd = crisis_exits['max_dd_3m_pct'].dropna()
            ru = crisis_exits['max_runup_3m_pct'].dropna()
            if len(dd) > 0:
                lines.append(f"\n    3-month risk/reward after CRISIS exit:")
                lines.append(f"      Max drawdown:  avg={dd.mean():+.2f}%, worst={dd.min():+.2f}%")
                lines.append(f"      Max runup:     avg={ru.mean():+.2f}%, best={ru.max():+.2f}%")
                lines.append(f"      Reward/Risk:   {abs(ru.mean()/dd.mean()):.2f}x")

        lines.append(f"\n  Specific CRISIS exit dates and what happened:")
        for _, row in crisis_exits.iterrows():
            d = row['date'].strftime('%Y-%m-%d')
            to = row['to_regime']
            fwd_3m = row.get('fwd_3m_pct', np.nan)
            fwd_12m = row.get('fwd_12m_pct', np.nan)
            lines.append(f"    {d}: CRISIS -> {to}  | 3m: {fwd_3m:+.1f}%, 12m: {fwd_12m:+.1f}%"
                        if not np.isnan(fwd_3m) and not np.isnan(fwd_12m)
                        else f"    {d}: CRISIS -> {to}")

    # Section 7: Indicator composite for CRISIS exit
    lines.append("\n" + "=" * 60)
    lines.append("7. BEST COMPOSITE SIGNAL FOR CRISIS EXIT DETECTION")
    lines.append("=" * 60)

    if len(lead_times_df) > 0:
        crisis_recovery = lead_times_df[
            lead_times_df['transition_type'].str.contains('CRISIS')
        ]

        if len(crisis_recovery) > 0:
            crisis_sorted = crisis_recovery.sort_values(
                ['signal_rate', 'median_lead_days'],
                ascending=[False, False]
            )

            lines.append("\n  Ranked by reliability * lead time:")
            lines.append(f"  {'Indicator':>30s}  {'Signal Rate':>11s}  {'Median Lead':>11s}  {'Score':>7s}")
            lines.append(f"  {'─' * 65}")

            for _, row in crisis_sorted.iterrows():
                score = row['signal_rate'] * row['median_lead_days']
                lines.append(f"  {row['indicator']:>30s}  "
                           f"{row['signal_rate']:>10.0%}  "
                           f"{row['median_lead_days']:>9.1f}d  "
                           f"{score:>6.1f}")

    # Section 8: Actionable Trading Rules
    lines.append("\n" + "=" * 60)
    lines.append("8. ACTIONABLE TRADING RULES (DERIVED FROM DATA)")
    lines.append("=" * 60)

    # Rule 1: After CRISIS exit, average 3m/6m/12m returns
    crisis_fwd = forward_returns_df[forward_returns_df['from_regime'] == 'CRISIS']
    if len(crisis_fwd) > 0:
        avg_3m = crisis_fwd['fwd_3m_pct'].dropna().mean()
        avg_6m = crisis_fwd['fwd_6m_pct'].dropna().mean()
        avg_12m = crisis_fwd['fwd_12m_pct'].dropna().mean()
        wr_3m = (crisis_fwd['fwd_3m_pct'].dropna() > 0).mean() * 100
        wr_12m = (crisis_fwd['fwd_12m_pct'].dropna() > 0).mean() * 100

        lines.append(f"\n  RULE 1 — BUY CRISIS EXIT:")
        lines.append(f"    When market exits CRISIS regime (VIX drops below 30 OR SPY crosses above 200d SMA)")
        lines.append(f"    Historical: avg 3m return = {avg_3m:+.1f}%, 12m = {avg_12m:+.1f}%")
        lines.append(f"    Win rate: {wr_3m:.0f}% at 3m, {wr_12m:.0f}% at 12m")

    # Rule 2: BULL -> CAUTION is a warning but not a sell signal
    bull_caution = forward_returns_df[forward_returns_df['transition'] == 'BULL -> CAUTION']
    if len(bull_caution) > 0:
        avg_3m = bull_caution['fwd_3m_pct'].dropna().mean()
        wr_3m = (bull_caution['fwd_3m_pct'].dropna() > 0).mean() * 100

        lines.append(f"\n  RULE 2 — BULL -> CAUTION (WARNING, NOT PANIC):")
        lines.append(f"    Avg 3m forward return: {avg_3m:+.1f}%, win rate: {wr_3m:.0f}%")
        lines.append(f"    This transition happens frequently and often reverts. Reduce, don't sell.")

    # Rule 3: CAUTION -> CRISIS is the danger zone
    caution_crisis = forward_returns_df[forward_returns_df['transition'] == 'CAUTION -> CRISIS']
    if len(caution_crisis) > 0:
        avg_3m = caution_crisis['fwd_3m_pct'].dropna().mean()
        avg_dd = caution_crisis['max_dd_3m_pct'].dropna().mean()

        lines.append(f"\n  RULE 3 — CAUTION -> CRISIS (DANGER):")
        lines.append(f"    Avg 3m forward return: {avg_3m:+.1f}%, avg max drawdown: {avg_dd:+.1f}%")
        lines.append(f"    This is the acceleration into a bear. Hedge or exit risk.")

    lines.append("\n" + "=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return '\n'.join(lines)

# ============================================================
# 9. TRANSITION MAP ENRICHMENT
# ============================================================
def build_enriched_transition_map(transitions_df, indicators_df, forward_returns_df):
    """Build the comprehensive transition map CSV with all signals and forward returns."""

    # Merge transitions with indicators
    merged = transitions_df.merge(
        indicators_df[['date', 'transition'] + [c for c in indicators_df.columns
                       if c not in ['date', 'transition', 'from_regime', 'to_regime', 'days_in_prev']]],
        on=['date', 'transition'],
        how='left'
    )

    # Merge with forward returns
    fwd_cols = [c for c in forward_returns_df.columns
                if c not in ['date', 'transition', 'from_regime', 'to_regime']]
    merged = merged.merge(
        forward_returns_df[['date', 'transition'] + fwd_cols],
        on=['date', 'transition'],
        how='left'
    )

    return merged

# ============================================================
# MAIN
# ============================================================
def main():
    print("=" * 60)
    print("REGIME TRANSITION ANALYSIS v1")
    print("=" * 60)

    # Step 1: Download data
    data = download_data()

    if 'SPY' not in data or '^VIX' not in data:
        print("ERROR: Missing critical data (SPY or VIX)")
        sys.exit(1)

    # Step 2: Classify regimes
    regime, spy, vix, spy_sma200 = classify_regimes(data)

    # Step 3: Map transitions
    transitions_df = map_transitions(regime)

    # Step 4: Compute leading indicators
    indicators_df, indicator_series = compute_leading_indicators(data, transitions_df)

    # Step 5: Forward returns
    forward_returns_df = compute_forward_returns(data, transitions_df)

    # Step 6: Signal lead times
    lead_times_df = analyze_signal_lead_times(indicator_series, transitions_df, regime)

    # Step 7: Episode analysis
    episode_analysis = analyze_major_episodes(transitions_df, forward_returns_df, regime)

    # Step 8: Build enriched transition map
    transition_map = build_enriched_transition_map(transitions_df, indicators_df, forward_returns_df)

    # Step 9: Generate report
    report = generate_summary_report(regime, transitions_df, forward_returns_df,
                                      lead_times_df, episode_analysis, indicators_df)

    # ======== SAVE OUTPUTS ========
    print("\n" + "=" * 60)
    print("SAVING OUTPUTS")
    print("=" * 60)

    # transition_map.csv
    transition_map.to_csv(OUTPUT_DIR / 'transition_map.csv', index=False)
    print(f"Saved: transition_map.csv ({len(transition_map)} rows, {len(transition_map.columns)} cols)")

    # leading_indicators.csv
    lead_times_df.to_csv(OUTPUT_DIR / 'leading_indicators.csv', index=False)
    print(f"Saved: leading_indicators.csv ({len(lead_times_df)} rows)")

    # forward_returns.csv
    forward_returns_df.to_csv(OUTPUT_DIR / 'forward_returns.csv', index=False)
    print(f"Saved: forward_returns.csv ({len(forward_returns_df)} rows)")

    # summary_report.txt
    with open(OUTPUT_DIR / 'summary_report.txt', 'w') as f:
        f.write(report)
    print(f"Saved: summary_report.txt")

    # Also save regime series for future use
    regime_df = pd.DataFrame({'date': regime.index, 'regime': regime.values})
    regime_df.to_csv(OUTPUT_DIR / 'regime_series.csv', index=False)
    print(f"Saved: regime_series.csv ({len(regime_df)} rows)")

    # Print the report
    print("\n")
    print(report)

    return regime, transitions_df, forward_returns_df, lead_times_df

if __name__ == '__main__':
    main()
