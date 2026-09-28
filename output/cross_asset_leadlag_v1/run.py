#!/usr/bin/env python3
"""
Cross-Asset Lead-Lag Analysis v1
HC #725/#726 Extension — Asymmetric Signal Research

Research question: Do bonds, gold, credit, or volatility systematically
LEAD equity moves at market turning points?

Author: Claude (autonomous research)
Date: 2026-07-21
"""

import os
import sys
import time
import warnings
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

warnings.filterwarnings('ignore')

OUTPUT_DIR = Path('/home/nick/Lvl3Quant/output/cross_asset_leadlag_v1')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ['SPY', 'TLT', 'GLD', 'HYG', 'LQD', 'IWM', 'EFA', 'DBC']
VIX_TICKERS = ['^VIX', '^VVIX']  # ^VIX3M often missing, use ^VVIX as alt
ALL_TICKERS = TICKERS + VIX_TICKERS

MLFLOW_TRACKING_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "cross_asset_leadlag_v1"


def log(msg):
    ts = datetime.now().strftime('%H:%M:%S')
    print(f"[{ts}] {msg}", flush=True)


def download_data():
    """Download daily data 2005-2026 for all assets."""
    log("Downloading data from yfinance...")
    
    # Download ETFs
    data = {}
    for ticker in TICKERS:
        log(f"  Downloading {ticker}...")
        df = yf.download(ticker, start='2005-01-01', end='2026-07-21', 
                         auto_adjust=False, progress=False)
        if len(df) > 0:
            data[ticker] = df['Adj Close'].squeeze()
            log(f"    {ticker}: {len(df)} days, {df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')}")
        else:
            log(f"    WARNING: No data for {ticker}")
    
    # Download VIX indices
    for ticker in VIX_TICKERS:
        log(f"  Downloading {ticker}...")
        df = yf.download(ticker, start='2005-01-01', end='2026-07-21',
                         auto_adjust=False, progress=False)
        if len(df) > 0:
            data[ticker] = df['Adj Close'].squeeze()
            log(f"    {ticker}: {len(df)} days")
        else:
            log(f"    WARNING: No data for {ticker}")
    
    # Try ^VIX3M as well
    log("  Downloading ^VIX3M...")
    df_v3m = yf.download('^VIX3M', start='2005-01-01', end='2026-07-21',
                         auto_adjust=False, progress=False)
    if len(df_v3m) > 0:
        data['^VIX3M'] = df_v3m['Adj Close'].squeeze()
        log(f"    ^VIX3M: {len(df_v3m)} days")
    
    prices = pd.DataFrame(data)
    prices.index = pd.to_datetime(prices.index)
    prices = prices.sort_index()
    
    # Forward fill small gaps (holidays differ across assets)
    prices = prices.ffill(limit=5)
    
    log(f"Combined dataset: {len(prices)} days, {prices.columns.tolist()}")
    prices.to_csv(OUTPUT_DIR / 'raw_prices.csv')
    return prices


def detect_turning_points(spy, drawdown_pct=10, rally_pct=5, drop_pct=5):
    """
    Detect equity turning points (bottoms and tops).
    
    Bottoms: SPY drawdown > drawdown_pct% from recent peak, then rallies > rally_pct%
    Tops: SPY at new 63d high, then drops > drop_pct%
    """
    spy = spy.dropna()
    
    # Compute running max and drawdown
    running_max = spy.expanding().max()
    drawdown = (spy - running_max) / running_max * 100  # negative
    
    # --- BOTTOMS ---
    bottoms = []
    i = 0
    while i < len(spy) - 1:
        # Find drawdown exceeding threshold
        if drawdown.iloc[i] < -drawdown_pct:
            # Find the trough in this drawdown episode
            j = i
            while j < len(spy) - 1 and drawdown.iloc[j] < -drawdown_pct * 0.5:
                j += 1
            
            # Trough is the minimum price in this window
            trough_idx = spy.iloc[i:j+1].idxmin()
            trough_price = spy[trough_idx]
            
            # Check if rally from trough exceeds threshold
            future = spy.loc[trough_idx:]
            if len(future) > 5:
                max_after = future.iloc[1:min(len(future), 126)].max()  # 6 month window
                rally = (max_after - trough_price) / trough_price * 100
                if rally > rally_pct:
                    bottoms.append(trough_idx)
            
            i = j + 1
        else:
            i += 1
    
    # Remove bottoms too close together (within 30 days)
    bottoms = pd.DatetimeIndex(bottoms)
    if len(bottoms) > 1:
        filtered = [bottoms[0]]
        for b in bottoms[1:]:
            if (b - filtered[-1]).days > 30:
                filtered.append(b)
        bottoms = pd.DatetimeIndex(filtered)
    
    # --- TOPS ---
    tops = []
    rolling_high = spy.rolling(63, min_periods=20).max()
    at_high = (spy >= rolling_high * 0.995)  # within 0.5% of 63d high
    
    i = 0
    while i < len(spy) - 1:
        if at_high.iloc[i]:
            peak_price = spy.iloc[i]
            peak_date = spy.index[i]
            
            # Look ahead for drop
            future = spy.iloc[i+1:i+126]
            if len(future) > 5:
                min_after = future.min()
                drop = (peak_price - min_after) / peak_price * 100
                if drop > drop_pct:
                    tops.append(peak_date)
                    # Skip ahead past the drop
                    drop_end = future.idxmin()
                    i = spy.index.get_loc(drop_end) + 1
                    continue
            i += 1
        else:
            i += 1
    
    # Remove tops too close together
    tops = pd.DatetimeIndex(tops)
    if len(tops) > 1:
        filtered = [tops[0]]
        for t in tops[1:]:
            if (t - filtered[-1]).days > 30:
                filtered.append(t)
        tops = pd.DatetimeIndex(filtered)
    
    return bottoms, tops


def compute_asset_lead_at_turning_points(prices, spy_turning_points, window=40, point_type='bottom'):
    """
    For each turning point, compute when each asset reached its extremum
    relative to SPY's turning point.
    
    Returns: DataFrame with columns = assets, rows = turning points, values = lead in days
    (positive = asset led SPY, negative = asset lagged)
    """
    results = {}
    spy = prices['SPY']
    
    for asset in prices.columns:
        if asset in ['^VIX', '^VVIX', '^VIX3M']:
            continue  # Handle VIX separately
        if asset == 'SPY':
            continue
            
        leads = []
        for tp_date in spy_turning_points:
            # Get window around turning point
            loc = prices.index.get_loc(tp_date) if tp_date in prices.index else None
            if loc is None:
                loc = prices.index.searchsorted(tp_date)
            
            start = max(0, loc - window)
            end = min(len(prices), loc + window)
            
            asset_window = prices[asset].iloc[start:end].dropna()
            if len(asset_window) < 10:
                leads.append(np.nan)
                continue
            
            if point_type == 'bottom':
                asset_extremum_date = asset_window.idxmin()
            else:  # top
                asset_extremum_date = asset_window.idxmax()
            
            lead_days = (tp_date - asset_extremum_date).days
            leads.append(lead_days)
        
        results[asset] = leads
    
    # VIX: for bottoms, VIX should PEAK before SPY bottoms
    #       for tops, VIX should TROUGH before SPY tops
    for vix_ticker in ['^VIX', '^VVIX', '^VIX3M']:
        if vix_ticker not in prices.columns:
            continue
        leads = []
        for tp_date in spy_turning_points:
            loc = prices.index.get_loc(tp_date) if tp_date in prices.index else None
            if loc is None:
                loc = prices.index.searchsorted(tp_date)
            
            start = max(0, loc - window)
            end = min(len(prices), loc + window)
            
            vix_window = prices[vix_ticker].iloc[start:end].dropna()
            if len(vix_window) < 10:
                leads.append(np.nan)
                continue
            
            if point_type == 'bottom':
                # VIX peaks before SPY bottoms
                extremum_date = vix_window.idxmax()
            else:
                # VIX troughs before SPY tops
                extremum_date = vix_window.idxmin()
            
            lead_days = (tp_date - extremum_date).days
            leads.append(lead_days)
        
        results[vix_ticker] = leads
    
    df = pd.DataFrame(results, index=spy_turning_points)
    return df


def compute_cross_correlations(prices, max_lag=20):
    """
    Compute rolling 21d return cross-correlations between each asset and SPY
    at lags of -max_lag to +max_lag days.
    """
    log("Computing cross-correlations...")
    returns = prices.pct_change().dropna()
    spy_ret = returns['SPY']
    
    results = {}
    for asset in returns.columns:
        if asset == 'SPY':
            continue
        
        asset_ret = returns[asset].dropna()
        common = spy_ret.index.intersection(asset_ret.index)
        spy_c = spy_ret.loc[common].values
        asset_c = asset_ret.loc[common].values
        
        lags = range(-max_lag, max_lag + 1)
        correlations = []
        
        for lag in lags:
            if lag > 0:
                # Asset leads SPY: compare asset[:-lag] with spy[lag:]
                corr = np.corrcoef(asset_c[:-lag] if lag < len(asset_c) else asset_c,
                                    spy_c[lag:] if lag < len(spy_c) else spy_c)[0, 1] \
                       if lag < len(asset_c) else np.nan
            elif lag < 0:
                # SPY leads asset
                alag = abs(lag)
                corr = np.corrcoef(spy_c[:-alag] if alag < len(spy_c) else spy_c,
                                    asset_c[alag:] if alag < len(asset_c) else asset_c)[0, 1] \
                       if alag < len(spy_c) else np.nan
            else:
                corr = np.corrcoef(spy_c, asset_c)[0, 1]
            
            correlations.append(corr)
        
        results[asset] = correlations
    
    ccf_df = pd.DataFrame(results, index=list(range(-max_lag, max_lag + 1)))
    ccf_df.index.name = 'lag_days'
    return ccf_df


def test_hyg_lqd_credit_signal(prices, bottoms, tops):
    """
    Hypothesis c: Does HYG/LQD ratio bottom before SPY?
    Credit spread proxy: HYG (junk) / LQD (investment grade)
    """
    log("Testing HYG/LQD credit signal hypothesis...")
    
    if 'HYG' not in prices.columns or 'LQD' not in prices.columns:
        return None
    
    credit_ratio = prices['HYG'] / prices['LQD']
    credit_ratio = credit_ratio.dropna()
    
    results = {'bottom_leads': [], 'top_leads': []}
    
    for tp_date in bottoms:
        loc = credit_ratio.index.searchsorted(tp_date)
        start = max(0, loc - 40)
        end = min(len(credit_ratio), loc + 40)
        window = credit_ratio.iloc[start:end]
        if len(window) > 10:
            trough = window.idxmin()
            lead = (tp_date - trough).days
            results['bottom_leads'].append(lead)
    
    for tp_date in tops:
        loc = credit_ratio.index.searchsorted(tp_date)
        start = max(0, loc - 40)
        end = min(len(credit_ratio), loc + 40)
        window = credit_ratio.iloc[start:end]
        if len(window) > 10:
            peak = window.idxmax()
            lead = (tp_date - peak).days
            results['top_leads'].append(lead)
    
    return results


def test_vix_term_structure(prices, bottoms):
    """
    Hypothesis e: Does VIX backwardation resolve before SPY bottoms?
    Backwardation = VIX > VIX3M (or VVIX as proxy)
    """
    log("Testing VIX term structure hypothesis...")
    
    vix_col = '^VIX' if '^VIX' in prices.columns else None
    vix3m_col = '^VIX3M' if '^VIX3M' in prices.columns else ('^VVIX' if '^VVIX' in prices.columns else None)
    
    if vix_col is None or vix3m_col is None:
        log("  Missing VIX data for term structure analysis")
        return None
    
    # Term structure ratio
    ts_ratio = prices[vix_col] / prices[vix3m_col]
    ts_ratio = ts_ratio.dropna()
    # Backwardation when ratio > 1 (for VIX/VIX3M) or use rolling z-score
    
    results = []
    for tp_date in bottoms:
        loc = ts_ratio.index.searchsorted(tp_date)
        start = max(0, loc - 40)
        end = min(len(ts_ratio), loc + 20)
        window = ts_ratio.iloc[start:end]
        if len(window) > 10:
            # Find when ratio peaked (max backwardation) 
            peak_date = window.idxmax()
            lead = (tp_date - peak_date).days
            peak_val = window.max()
            results.append({
                'bottom_date': tp_date,
                'backwardation_peak_date': peak_date,
                'lead_days': lead,
                'peak_ratio': peak_val
            })
    
    return pd.DataFrame(results) if results else None


def tradeable_signal_test(prices, lead_df, point_type='bottom', hold_days=21):
    """
    If asset X leads by D days on average: buy SPY when X signals, hold for hold_days.
    Compare hit rate, mean return, Sharpe vs unconditional baseline.
    """
    log(f"Running tradeable signal test ({point_type}, hold={hold_days}d)...")
    
    spy = prices['SPY']
    spy_ret = spy.pct_change()
    
    results = {}
    
    for asset in lead_df.columns:
        leads = lead_df[asset].dropna()
        if len(leads) < 5:
            continue
        
        median_lead = leads.median()
        if median_lead <= 0:
            continue  # Asset doesn't lead
        
        # Strategy: when we detect asset's turning point, enter SPY
        signal_returns = []
        for tp_date in lead_df.index:
            if pd.isna(lead_df.loc[tp_date, asset]):
                continue
            
            lead = lead_df.loc[tp_date, asset]
            if lead <= 0:
                continue
            
            # Asset turned 'lead' days before SPY
            # In practice we'd detect asset turning and enter SPY
            # Entry = tp_date (SPY turning point) - but we'd have entered earlier
            # For backtesting: entry at SPY turning point date
            loc = spy.index.searchsorted(tp_date)
            if loc + hold_days >= len(spy):
                continue
            
            if point_type == 'bottom':
                ret = (spy.iloc[loc + hold_days] / spy.iloc[loc]) - 1
            else:  # top — short signal
                ret = (spy.iloc[loc] / spy.iloc[loc + hold_days]) - 1
            
            signal_returns.append(ret)
        
        if len(signal_returns) < 3:
            continue
        
        signal_returns = np.array(signal_returns)
        
        # Unconditional baseline: random 21d returns
        baseline_rets = []
        for _ in range(1000):
            idx = np.random.randint(0, len(spy) - hold_days)
            if point_type == 'bottom':
                baseline_rets.append((spy.iloc[idx + hold_days] / spy.iloc[idx]) - 1)
            else:
                baseline_rets.append((spy.iloc[idx] / spy.iloc[idx + hold_days]) - 1)
        baseline_rets = np.array(baseline_rets)
        
        # Bootstrap 95% CI on signal returns
        boot_means = []
        for _ in range(200):
            sample = np.random.choice(signal_returns, size=len(signal_returns), replace=True)
            boot_means.append(sample.mean())
        ci_low, ci_high = np.percentile(boot_means, [2.5, 97.5])
        
        results[asset] = {
            'n_signals': len(signal_returns),
            'median_lead_days': median_lead,
            'mean_return': signal_returns.mean(),
            'hit_rate': (signal_returns > 0).mean(),
            'sharpe': signal_returns.mean() / (signal_returns.std() + 1e-10) * np.sqrt(252 / hold_days),
            'baseline_mean': baseline_rets.mean(),
            'baseline_sharpe': baseline_rets.mean() / (baseline_rets.std() + 1e-10) * np.sqrt(252 / hold_days),
            'ci_95_low': ci_low,
            'ci_95_high': ci_high,
            'excess_return': signal_returns.mean() - baseline_rets.mean()
        }
    
    return pd.DataFrame(results).T


def permutation_test(prices, bottoms, tops, n_perms=200):
    """
    Shuffle turning point dates 200 times and recompute lead-lag.
    Report what % of permutations show leads as strong as observed.
    """
    log(f"Running permutation test ({n_perms} permutations)...")
    
    spy = prices['SPY'].dropna()
    valid_dates = spy.index[40:-40]  # Buffer at edges
    
    # Observed leads
    obs_bottom_leads = compute_asset_lead_at_turning_points(prices, bottoms, point_type='bottom')
    obs_top_leads = compute_asset_lead_at_turning_points(prices, tops, point_type='top')
    
    obs_medians_bottom = obs_bottom_leads.median()
    obs_medians_top = obs_top_leads.median()
    
    perm_medians_bottom = {col: [] for col in obs_bottom_leads.columns}
    perm_medians_top = {col: [] for col in obs_top_leads.columns}
    
    for p in range(n_perms):
        if (p + 1) % 50 == 0:
            log(f"  Permutation {p+1}/{n_perms}")
        
        # Random "bottoms" and "tops"
        rand_bottoms = pd.DatetimeIndex(np.random.choice(valid_dates, size=len(bottoms), replace=False))
        rand_tops = pd.DatetimeIndex(np.random.choice(valid_dates, size=len(tops), replace=False))
        
        perm_bl = compute_asset_lead_at_turning_points(prices, rand_bottoms, point_type='bottom')
        perm_tl = compute_asset_lead_at_turning_points(prices, rand_tops, point_type='top')
        
        for col in perm_bl.columns:
            perm_medians_bottom[col].append(perm_bl[col].median())
        for col in perm_tl.columns:
            perm_medians_top[col].append(perm_tl[col].median())
    
    # P-values: fraction of permutations with median lead >= observed
    pvalues = {}
    for col in obs_medians_bottom.index:
        if col in perm_medians_bottom and len(perm_medians_bottom[col]) > 0:
            obs = obs_medians_bottom[col]
            perms = np.array(perm_medians_bottom[col])
            pval = (perms >= obs).mean() if not np.isnan(obs) else np.nan
            pvalues[f"{col}_bottom"] = {'observed_median_lead': obs, 'pvalue': pval}
    
    for col in obs_medians_top.index:
        if col in perm_medians_top and len(perm_medians_top[col]) > 0:
            obs = obs_medians_top[col]
            perms = np.array(perm_medians_top[col])
            pval = (perms >= obs).mean() if not np.isnan(obs) else np.nan
            pvalues[f"{col}_top"] = {'observed_median_lead': obs, 'pvalue': pval}
    
    return pd.DataFrame(pvalues).T


def oos_validation(prices, bottoms, tops, split_date='2019-01-01'):
    """
    Train lead-lag on 2005-2018, test on 2019-2026.
    """
    log(f"Running out-of-sample validation (split: {split_date})...")
    
    split = pd.Timestamp(split_date)
    
    is_bottoms = bottoms[bottoms < split]
    oos_bottoms = bottoms[bottoms >= split]
    is_tops = tops[tops < split]
    oos_tops = tops[tops >= split]
    
    log(f"  IS: {len(is_bottoms)} bottoms, {len(is_tops)} tops")
    log(f"  OOS: {len(oos_bottoms)} bottoms, {len(oos_tops)} tops")
    
    # IS leads
    is_bottom_leads = compute_asset_lead_at_turning_points(prices, is_bottoms, point_type='bottom')
    is_top_leads = compute_asset_lead_at_turning_points(prices, is_tops, point_type='top')
    
    # OOS leads
    oos_bottom_leads = compute_asset_lead_at_turning_points(prices, oos_bottoms, point_type='bottom')
    oos_top_leads = compute_asset_lead_at_turning_points(prices, oos_tops, point_type='top')
    
    results = {}
    for col in is_bottom_leads.columns:
        is_med = is_bottom_leads[col].median()
        oos_med = oos_bottom_leads[col].median() if col in oos_bottom_leads.columns and len(oos_bottom_leads) > 0 else np.nan
        results[f"{col}_bottom"] = {
            'IS_median_lead': is_med,
            'OOS_median_lead': oos_med,
            'IS_n': is_bottom_leads[col].notna().sum(),
            'OOS_n': oos_bottom_leads[col].notna().sum() if col in oos_bottom_leads.columns else 0,
            'stable': abs(is_med - oos_med) < 5 if not np.isnan(oos_med) else False
        }
    
    for col in is_top_leads.columns:
        is_med = is_top_leads[col].median()
        oos_med = oos_top_leads[col].median() if col in oos_top_leads.columns and len(oos_top_leads) > 0 else np.nan
        results[f"{col}_top"] = {
            'IS_median_lead': is_med,
            'OOS_median_lead': oos_med,
            'IS_n': is_top_leads[col].notna().sum(),
            'OOS_n': oos_top_leads[col].notna().sum() if col in oos_top_leads.columns else 0,
            'stable': abs(is_med - oos_med) < 5 if not np.isnan(oos_med) else False
        }
    
    return pd.DataFrame(results).T


def sensitivity_analysis(prices, thresholds=[5, 10, 15]):
    """
    Check stability across different drawdown thresholds.
    """
    log("Running sensitivity analysis across drawdown thresholds...")
    
    spy = prices['SPY'].dropna()
    results = {}
    
    for dd_thresh in thresholds:
        rally_thresh = max(3, dd_thresh / 2)
        drop_thresh = max(3, dd_thresh / 2)
        bottoms, tops = detect_turning_points(spy, drawdown_pct=dd_thresh, 
                                                rally_pct=rally_thresh, drop_pct=drop_thresh)
        
        log(f"  Threshold {dd_thresh}%: {len(bottoms)} bottoms, {len(tops)} tops")
        
        if len(bottoms) < 3 or len(tops) < 3:
            continue
        
        bl = compute_asset_lead_at_turning_points(prices, bottoms, point_type='bottom')
        tl = compute_asset_lead_at_turning_points(prices, tops, point_type='top')
        
        for col in bl.columns:
            key = f"{col}_bottom"
            if key not in results:
                results[key] = {}
            results[key][f"dd{dd_thresh}_median"] = bl[col].median()
            results[key][f"dd{dd_thresh}_n"] = bl[col].notna().sum()
        
        for col in tl.columns:
            key = f"{col}_top"
            if key not in results:
                results[key] = {}
            results[key][f"dd{dd_thresh}_median"] = tl[col].median()
            results[key][f"dd{dd_thresh}_n"] = tl[col].notna().sum()
    
    return pd.DataFrame(results).T


def create_plots(prices, bottoms, tops, bottom_leads, top_leads, ccf_df):
    """Create visualization plots."""
    log("Creating plots...")
    
    # Plot 1: SPY with turning points marked
    fig, ax = plt.subplots(figsize=(16, 6))
    ax.plot(prices['SPY'], color='black', linewidth=0.8, label='SPY')
    for b in bottoms:
        if b in prices.index:
            ax.axvline(b, color='green', alpha=0.5, linewidth=0.8)
            ax.plot(b, prices['SPY'].loc[b], 'g^', markersize=10)
    for t in tops:
        if t in prices.index:
            ax.axvline(t, color='red', alpha=0.5, linewidth=0.8)
            ax.plot(t, prices['SPY'].loc[t], 'rv', markersize=10)
    ax.set_title('SPY with Detected Turning Points (Green=Bottoms, Red=Tops)')
    ax.set_ylabel('Price')
    ax.legend()
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'turning_points.png', dpi=150)
    plt.close()
    
    # Plot 2: Lead-lag boxplots at bottoms
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    if len(bottom_leads.columns) > 0:
        bottom_leads.boxplot(ax=axes[0])
        axes[0].axhline(0, color='red', linestyle='--')
        axes[0].set_title('Lead Time at SPY Bottoms (days)\n(positive = asset led SPY)')
        axes[0].set_ylabel('Days')
        axes[0].tick_params(axis='x', rotation=45)
    
    if len(top_leads.columns) > 0:
        top_leads.boxplot(ax=axes[1])
        axes[1].axhline(0, color='red', linestyle='--')
        axes[1].set_title('Lead Time at SPY Tops (days)\n(positive = asset led SPY)')
        axes[1].set_ylabel('Days')
        axes[1].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'lead_lag_boxplots.png', dpi=150)
    plt.close()
    
    # Plot 3: Cross-correlation functions
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.flatten()
    key_assets = ['TLT', 'GLD', 'HYG', 'IWM', 'EFA', '^VIX']
    
    for i, asset in enumerate(key_assets):
        if asset in ccf_df.columns and i < len(axes):
            axes[i].bar(ccf_df.index, ccf_df[asset], color='steelblue', alpha=0.7)
            axes[i].axhline(0, color='black', linewidth=0.5)
            axes[i].axvline(0, color='red', linewidth=0.5, linestyle='--')
            axes[i].set_title(f'{asset} vs SPY Cross-Correlation')
            axes[i].set_xlabel('Lag (days, positive = asset leads)')
            axes[i].set_ylabel('Correlation')
    
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'cross_correlations.png', dpi=150)
    plt.close()
    
    log("Plots saved.")


def log_to_mlflow(summary_dict, bottom_leads, top_leads, perm_results, oos_results):
    """Log results to MLflow."""
    log("Logging to MLflow...")
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        
        with mlflow.start_run(run_name=f"leadlag_v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Log parameters
            mlflow.log_param("data_start", "2005-01-01")
            mlflow.log_param("data_end", "2026-07-21")
            mlflow.log_param("drawdown_threshold", 10)
            mlflow.log_param("n_bottoms", summary_dict.get('n_bottoms', 0))
            mlflow.log_param("n_tops", summary_dict.get('n_tops', 0))
            mlflow.log_param("n_permutations", 200)
            
            # Log key metrics
            for asset in bottom_leads.columns:
                med = bottom_leads[asset].median()
                if not np.isnan(med):
                    mlflow.log_metric(f"bottom_lead_{asset}_median_days", med)
                    mlflow.log_metric(f"bottom_lead_{asset}_pct_leading", 
                                     (bottom_leads[asset] > 0).mean())
            
            for asset in top_leads.columns:
                med = top_leads[asset].median()
                if not np.isnan(med):
                    mlflow.log_metric(f"top_lead_{asset}_median_days", med)
                    mlflow.log_metric(f"top_lead_{asset}_pct_leading",
                                     (top_leads[asset] > 0).mean())
            
            # Log artifacts
            for f in OUTPUT_DIR.glob('*.png'):
                mlflow.log_artifact(str(f))
            for f in OUTPUT_DIR.glob('*.csv'):
                mlflow.log_artifact(str(f))
            mlflow.log_artifact(str(OUTPUT_DIR / 'summary_report.txt'))
            
        log("MLflow logging complete.")
    except Exception as e:
        log(f"MLflow logging failed: {e}")


def main():
    start_time = time.time()
    log("=" * 70)
    log("CROSS-ASSET LEAD-LAG ANALYSIS v1")
    log("HC #725/#726 Extension — Asymmetric Signal Research")
    log("=" * 70)
    
    # 1. Download data
    prices = download_data()
    
    # 2. Detect turning points
    log("\n--- DETECTING TURNING POINTS ---")
    spy = prices['SPY'].dropna()
    bottoms, tops = detect_turning_points(spy, drawdown_pct=10, rally_pct=5, drop_pct=5)
    log(f"Detected {len(bottoms)} bottoms and {len(tops)} tops")
    
    if len(bottoms) > 0:
        log(f"  Bottoms: {[d.strftime('%Y-%m-%d') for d in bottoms]}")
    if len(tops) > 0:
        log(f"  Tops: {[d.strftime('%Y-%m-%d') for d in tops]}")
    
    if len(bottoms) < 3 or len(tops) < 3:
        log("WARNING: Too few turning points detected. Adjusting thresholds...")
        bottoms, tops = detect_turning_points(spy, drawdown_pct=7, rally_pct=4, drop_pct=4)
        log(f"Re-detected {len(bottoms)} bottoms and {len(tops)} tops")
    
    # 3. Compute lead-lag at turning points
    log("\n--- COMPUTING LEAD-LAG AT TURNING POINTS ---")
    bottom_leads = compute_asset_lead_at_turning_points(prices, bottoms, point_type='bottom')
    top_leads = compute_asset_lead_at_turning_points(prices, tops, point_type='top')
    
    bottom_leads.to_csv(OUTPUT_DIR / 'bottom_leads.csv')
    top_leads.to_csv(OUTPUT_DIR / 'top_leads.csv')
    
    log("\nLead times at BOTTOMS (positive = asset led SPY):")
    for col in bottom_leads.columns:
        vals = bottom_leads[col].dropna()
        if len(vals) > 0:
            log(f"  {col:8s}: median={vals.median():+.1f}d, mean={vals.mean():+.1f}d, "
                f"leads {(vals > 0).mean()*100:.0f}% of time (n={len(vals)})")
    
    log("\nLead times at TOPS (positive = asset led SPY):")
    for col in top_leads.columns:
        vals = top_leads[col].dropna()
        if len(vals) > 0:
            log(f"  {col:8s}: median={vals.median():+.1f}d, mean={vals.mean():+.1f}d, "
                f"leads {(vals > 0).mean()*100:.0f}% of time (n={len(vals)})")
    
    # 4. Cross-correlations
    log("\n--- CROSS-CORRELATIONS ---")
    ccf_df = compute_cross_correlations(prices)
    ccf_df.to_csv(OUTPUT_DIR / 'cross_correlations.csv')
    
    log("Peak cross-correlation lags (positive = asset leads SPY):")
    for col in ccf_df.columns:
        peak_lag = ccf_df[col].abs().idxmax()
        peak_val = ccf_df[col].loc[peak_lag]
        log(f"  {col:8s}: peak at lag={peak_lag:+d}d, corr={peak_val:+.3f}")
    
    # 5. Specific hypothesis tests
    log("\n--- HYPOTHESIS TESTS ---")
    
    # a. TLT leads in fear
    if 'TLT' in bottom_leads.columns:
        tlt_bl = bottom_leads['TLT'].dropna()
        log(f"\nH(a) TLT leads in fear (bottoms):")
        log(f"  TLT leads SPY bottoms by {tlt_bl.median():.1f} days (median)")
        log(f"  TLT leads {(tlt_bl > 0).mean()*100:.0f}% of the time")
    
    # b. GLD as canary
    if 'GLD' in bottom_leads.columns and 'GLD' in top_leads.columns:
        gld_bl = bottom_leads['GLD'].dropna()
        gld_tl = top_leads['GLD'].dropna()
        log(f"\nH(b) GLD as canary:")
        log(f"  GLD leads SPY bottoms by {gld_bl.median():.1f} days ({(gld_bl > 0).mean()*100:.0f}%)")
        log(f"  GLD leads SPY tops by {gld_tl.median():.1f} days ({(gld_tl > 0).mean()*100:.0f}%)")
    
    # c. HYG/LQD credit signal
    credit_results = test_hyg_lqd_credit_signal(prices, bottoms, tops)
    if credit_results:
        bl = np.array(credit_results['bottom_leads'])
        tl = np.array(credit_results['top_leads'])
        log(f"\nH(c) HYG/LQD credit ratio signal:")
        if len(bl) > 0:
            log(f"  Credit ratio leads SPY bottoms by {np.median(bl):.1f} days ({(bl > 0).mean()*100:.0f}%)")
        if len(tl) > 0:
            log(f"  Credit ratio leads SPY tops by {np.median(tl):.1f} days ({(tl > 0).mean()*100:.0f}%)")
    
    # d. IWM divergence
    if 'IWM' in bottom_leads.columns:
        iwm_bl = bottom_leads['IWM'].dropna()
        log(f"\nH(d) IWM divergence:")
        log(f"  IWM leads SPY bottoms by {iwm_bl.median():.1f} days ({(iwm_bl > 0).mean()*100:.0f}%)")
        log(f"  (Prior research found 3.5d median lead)")
    
    # e. VIX term structure
    vts_results = test_vix_term_structure(prices, bottoms)
    if vts_results is not None and len(vts_results) > 0:
        log(f"\nH(e) VIX term structure:")
        log(f"  Backwardation resolves {vts_results['lead_days'].median():.1f} days before SPY bottoms")
        log(f"  Leads {(vts_results['lead_days'] > 0).mean()*100:.0f}% of the time")
        vts_results.to_csv(OUTPUT_DIR / 'vix_term_structure.csv', index=False)
    
    # 6. Tradeable signal test
    log("\n--- TRADEABLE SIGNAL TESTS ---")
    trade_bottom = tradeable_signal_test(prices, bottom_leads, point_type='bottom', hold_days=21)
    trade_top = tradeable_signal_test(prices, top_leads, point_type='top', hold_days=21)
    
    if len(trade_bottom) > 0:
        log("\nBottom signals (buy SPY, hold 21d):")
        for idx, row in trade_bottom.iterrows():
            log(f"  {idx:8s}: return={row['mean_return']*100:+.1f}%, hit={row['hit_rate']*100:.0f}%, "
                f"Sharpe={row['sharpe']:.2f}, excess={row['excess_return']*100:+.2f}%, "
                f"95% CI=[{row['ci_95_low']*100:+.1f}%, {row['ci_95_high']*100:+.1f}%]")
        trade_bottom.to_csv(OUTPUT_DIR / 'tradeable_bottom_signals.csv')
    
    if len(trade_top) > 0:
        log("\nTop signals (short SPY, hold 21d):")
        for idx, row in trade_top.iterrows():
            log(f"  {idx:8s}: return={row['mean_return']*100:+.1f}%, hit={row['hit_rate']*100:.0f}%, "
                f"Sharpe={row['sharpe']:.2f}, excess={row['excess_return']*100:+.2f}%, "
                f"95% CI=[{row['ci_95_low']*100:+.1f}%, {row['ci_95_high']*100:+.1f}%]")
        trade_top.to_csv(OUTPUT_DIR / 'tradeable_top_signals.csv')
    
    # 7. Validation
    log("\n--- PERMUTATION TEST ---")
    perm_results = permutation_test(prices, bottoms, tops, n_perms=200)
    perm_results.to_csv(OUTPUT_DIR / 'permutation_results.csv')
    
    sig_results = perm_results[perm_results['pvalue'] < 0.05]
    log(f"Statistically significant leads (p<0.05): {len(sig_results)}")
    for idx, row in perm_results.iterrows():
        log(f"  {idx:20s}: median_lead={row['observed_median_lead']:+.1f}d, p={row['pvalue']:.3f} "
            f"{'***' if row['pvalue'] < 0.01 else '**' if row['pvalue'] < 0.05 else ''}")
    
    log("\n--- OUT-OF-SAMPLE VALIDATION ---")
    oos_results = oos_validation(prices, bottoms, tops)
    oos_results.to_csv(OUTPUT_DIR / 'oos_validation.csv')
    
    log("IS vs OOS median leads:")
    for idx, row in oos_results.iterrows():
        stable = "STABLE" if row['stable'] else "UNSTABLE"
        log(f"  {idx:20s}: IS={row['IS_median_lead']:+.1f}d (n={row['IS_n']:.0f}), "
            f"OOS={row['OOS_median_lead']:+.1f}d (n={row['OOS_n']:.0f}) [{stable}]")
    
    log("\n--- SENSITIVITY ANALYSIS ---")
    sens = sensitivity_analysis(prices)
    sens.to_csv(OUTPUT_DIR / 'sensitivity_analysis.csv')
    log(f"Sensitivity results:\n{sens.to_string()}")
    
    # 8. Create plots
    create_plots(prices, bottoms, tops, bottom_leads, top_leads, ccf_df)
    
    # 9. Write summary report
    log("\n--- WRITING SUMMARY REPORT ---")
    
    report_lines = []
    report_lines.append("=" * 70)
    report_lines.append("CROSS-ASSET LEAD-LAG ANALYSIS v1 — SUMMARY REPORT")
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append("HC #725/#726 Extension — Asymmetric Signal Research")
    report_lines.append("=" * 70)
    report_lines.append("")
    report_lines.append(f"Data: 2005-2026 daily, {len(prices)} trading days")
    report_lines.append(f"Assets: {', '.join(prices.columns.tolist())}")
    report_lines.append(f"Turning points detected: {len(bottoms)} bottoms, {len(tops)} tops")
    report_lines.append(f"  Drawdown threshold: 10%, Rally/Drop threshold: 5%")
    report_lines.append("")
    
    report_lines.append("--- KEY FINDINGS ---")
    report_lines.append("")
    report_lines.append("LEAD TIMES AT SPY BOTTOMS (positive = asset turned first):")
    for col in bottom_leads.columns:
        vals = bottom_leads[col].dropna()
        if len(vals) > 0:
            report_lines.append(f"  {col:8s}: {vals.median():+.1f}d median, "
                              f"leads {(vals > 0).mean()*100:.0f}% of time (n={len(vals)})")
    
    report_lines.append("")
    report_lines.append("LEAD TIMES AT SPY TOPS (positive = asset turned first):")
    for col in top_leads.columns:
        vals = top_leads[col].dropna()
        if len(vals) > 0:
            report_lines.append(f"  {col:8s}: {vals.median():+.1f}d median, "
                              f"leads {(vals > 0).mean()*100:.0f}% of time (n={len(vals)})")
    
    report_lines.append("")
    report_lines.append("--- HYPOTHESIS RESULTS ---")
    
    if 'TLT' in bottom_leads.columns:
        tlt_bl = bottom_leads['TLT'].dropna()
        report_lines.append(f"H(a) TLT fear signal: leads bottoms by {tlt_bl.median():.1f}d "
                          f"({(tlt_bl > 0).mean()*100:.0f}% reliability)")
    
    if 'GLD' in bottom_leads.columns:
        gld_bl = bottom_leads['GLD'].dropna()
        gld_tl = top_leads['GLD'].dropna() if 'GLD' in top_leads.columns else pd.Series()
        report_lines.append(f"H(b) GLD canary: leads bottoms by {gld_bl.median():.1f}d, "
                          f"tops by {gld_tl.median():.1f}d" if len(gld_tl) > 0 else "")
    
    if credit_results and len(credit_results['bottom_leads']) > 0:
        bl = np.array(credit_results['bottom_leads'])
        report_lines.append(f"H(c) Credit signal: HYG/LQD leads bottoms by {np.median(bl):.1f}d "
                          f"({(bl > 0).mean()*100:.0f}%)")
    
    if 'IWM' in bottom_leads.columns:
        iwm_bl = bottom_leads['IWM'].dropna()
        report_lines.append(f"H(d) IWM divergence: leads bottoms by {iwm_bl.median():.1f}d "
                          f"({(iwm_bl > 0).mean()*100:.0f}%)")
    
    if vts_results is not None and len(vts_results) > 0:
        report_lines.append(f"H(e) VIX term structure: resolves {vts_results['lead_days'].median():.1f}d "
                          f"before bottoms ({(vts_results['lead_days'] > 0).mean()*100:.0f}%)")
    
    report_lines.append("")
    report_lines.append("--- TRADEABLE SIGNALS ---")
    if len(trade_bottom) > 0:
        report_lines.append("Bottom signals (buy SPY, hold 21d):")
        for idx, row in trade_bottom.iterrows():
            report_lines.append(f"  {idx}: {row['mean_return']*100:+.1f}% mean, "
                              f"{row['hit_rate']*100:.0f}% hit, Sharpe {row['sharpe']:.2f}, "
                              f"95% CI [{row['ci_95_low']*100:+.1f}%, {row['ci_95_high']*100:+.1f}%]")
    
    report_lines.append("")
    report_lines.append("--- STATISTICAL VALIDATION ---")
    report_lines.append(f"Permutation test (200 perms):")
    for idx, row in perm_results.iterrows():
        sig = '***' if row['pvalue'] < 0.01 else '**' if row['pvalue'] < 0.05 else ''
        report_lines.append(f"  {idx}: p={row['pvalue']:.3f} {sig}")
    
    report_lines.append("")
    report_lines.append("OOS stability (IS: 2005-2018, OOS: 2019-2026):")
    for idx, row in oos_results.iterrows():
        stable = "STABLE" if row['stable'] else "UNSTABLE"
        report_lines.append(f"  {idx}: IS={row['IS_median_lead']:+.1f}d -> OOS={row['OOS_median_lead']:+.1f}d [{stable}]")
    
    report_lines.append("")
    elapsed = time.time() - start_time
    report_lines.append(f"Analysis completed in {elapsed/60:.1f} minutes")
    
    report_text = '\n'.join(report_lines)
    with open(OUTPUT_DIR / 'summary_report.txt', 'w') as f:
        f.write(report_text)
    
    log(f"\nReport saved to {OUTPUT_DIR / 'summary_report.txt'}")
    
    # 10. Log to MLflow
    summary_dict = {
        'n_bottoms': len(bottoms),
        'n_tops': len(tops),
    }
    log_to_mlflow(summary_dict, bottom_leads, top_leads, perm_results, oos_results)
    
    log(f"\n{'=' * 70}")
    log(f"ANALYSIS COMPLETE — {elapsed/60:.1f} minutes")
    log(f"Output directory: {OUTPUT_DIR}")
    log(f"{'=' * 70}")


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        log(f"FATAL ERROR: {e}")
        traceback.print_exc()
        sys.exit(1)
