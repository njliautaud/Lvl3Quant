#!/usr/bin/env python3
"""
Lead/Lag Analysis + Expanded Asymmetric Signal System
=====================================================
HC #726 Expansion — Comprehensive cross-asset lead/lag relationships,
stock-level distress signals, and market-wide predictive indicators.

Answers:
  1. What LEADS vs LAGS? (Granger-style predictive relationships)
  2. What predicts asymmetric upside? (conditional forward returns)
  3. Stock-level signals (individual name distress/recovery)
  4. Market-wide regime signals (breadth, rotation, flows)

Author: Claude (Head of Quant)
Date: 2026-07-22
"""
import os
import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy import stats
from datetime import datetime, timedelta
from collections import defaultdict

warnings.filterwarnings('ignore')

OUT_DIR = '/home/jupiter/Lvl3Quant/output/lead_lag_analysis'
os.makedirs(OUT_DIR, exist_ok=True)

# ============================================================================
# CONFIGURATION
# ============================================================================

# Broad asset universe
CORE_ASSETS = ['SPY', 'QQQ', 'IWM', 'DIA', 'TLT', 'IEF', 'SHY', 'GLD', 'SLV',
               'USO', 'UNG', 'HYG', 'LQD', 'JNK', 'EEM', 'EFA', 'VNQ', 'DBC']

SECTOR_ETFS = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP',
               'XLU', 'XLB', 'XLRE']

# Thematic/factor ETFs for rotation analysis
FACTOR_ETFS = ['MTUM', 'VLUE', 'QUAL', 'SIZE', 'USMV']  # momentum, value, quality, size, min-vol

VIX_FAMILY = ['^VIX', '^VIX3M']

# Large-cap individual stocks for stock-level analysis
STOCK_UNIVERSE = [
    'AAPL', 'MSFT', 'AMZN', 'GOOGL', 'META', 'NVDA', 'TSLA', 'BRK-B',
    'UNH', 'JNJ', 'V', 'XOM', 'JPM', 'PG', 'MA', 'HD', 'CVX', 'MRK',
    'ABBV', 'LLY', 'PEP', 'KO', 'COST', 'AVGO', 'TMO', 'MCD', 'WMT',
    'ACN', 'CSCO', 'ABT', 'CRM', 'NKE', 'TXN', 'NEE', 'AMD', 'QCOM',
    'HON', 'LOW', 'AMGN', 'INTC', 'BA', 'GS', 'CAT', 'BLK', 'ISRG',
    'SYK', 'ADP', 'NFLX', 'ADBE', 'ORCL'
]

ALL_TICKERS = list(set(CORE_ASSETS + SECTOR_ETFS + FACTOR_ETFS + VIX_FAMILY + STOCK_UNIVERSE))

FORWARD_WINDOWS = {'1w': 5, '2w': 10, '1m': 21, '3m': 63, '6m': 126}

print("=" * 80)
print("LEAD/LAG ANALYSIS + EXPANDED ASYMMETRIC SIGNALS")
print("=" * 80)

# ============================================================================
# PART 0: DATA
# ============================================================================

cache_file = os.path.join(OUT_DIR, 'price_data_expanded.parquet')

if os.path.exists(cache_file):
    prices = pd.read_parquet(cache_file)
    if prices.index.max() < pd.Timestamp('2026-07-15'):
        print("Cache stale, re-downloading...")
        os.remove(cache_file)
        prices = None
    else:
        print(f"Loaded cached data: {len(prices)} rows, {len(prices.columns)} cols")
else:
    prices = None

if prices is None:
    print(f"Downloading {len(ALL_TICKERS)} tickers (2005-2026)...")
    raw = yf.download(ALL_TICKERS, start='2005-01-01', progress=True, auto_adjust=True, group_by='ticker')

    dfs = {}
    for t in ALL_TICKERS:
        try:
            if isinstance(raw.columns, pd.MultiIndex):
                if t in raw.columns.get_level_values(0):
                    series = raw[(t, 'Close')].dropna()
                else:
                    continue
            else:
                series = raw['Close'].dropna()
            if len(series) > 0:
                dfs[t] = series
        except Exception:
            pass

    prices = pd.DataFrame(dfs)
    if isinstance(prices.columns, pd.MultiIndex):
        prices.columns = prices.columns.get_level_values(0)

    # Rename VIX columns
    for c in list(prices.columns):
        if c == '^VIX':
            prices = prices.rename(columns={c: 'VIX'})
        elif c == '^VIX3M':
            prices = prices.rename(columns={c: 'VIX3M'})

    prices.to_parquet(cache_file)
    print(f"Saved: {len(prices)} rows, {len(prices.columns)} tickers")

returns = prices.pct_change()
log_ret = np.log(prices / prices.shift(1))

print(f"\nData: {prices.index[0].date()} to {prices.index[-1].date()}")
print(f"Tickers: {len(prices.columns)}")

# ============================================================================
# PART 1: LEAD/LAG ANALYSIS — What predicts what?
# ============================================================================

print("\n" + "=" * 80)
print("PART 1: CROSS-ASSET LEAD/LAG RELATIONSHIPS")
print("=" * 80)

def compute_lead_lag(ret_a, ret_b, max_lag=10, min_obs=500):
    """
    Compute cross-correlation at different lags.
    Positive lag means A leads B (A at t predicts B at t+lag).
    Returns dict of lag -> (correlation, p_value, n_obs).
    """
    results = {}
    for lag in range(-max_lag, max_lag + 1):
        if lag > 0:
            a = ret_a.iloc[:-lag]
            b = ret_b.iloc[lag:]
        elif lag < 0:
            a = ret_a.iloc[-lag:]
            b = ret_b.iloc[:lag]
        else:
            a = ret_a
            b = ret_b

        # Align indices
        common = a.index.intersection(b.index)
        if len(common) < min_obs:
            continue
        a_aligned = a.loc[common].dropna()
        b_aligned = b.loc[common].dropna()
        common2 = a_aligned.index.intersection(b_aligned.index)
        if len(common2) < min_obs:
            continue

        corr, pval = stats.pearsonr(a_aligned.loc[common2], b_aligned.loc[common2])
        results[lag] = {'corr': round(corr, 5), 'pval': round(pval, 6), 'n': len(common2)}

    return results

# Key pairs for lead/lag analysis
LEAD_LAG_PAIRS = [
    # Credit leads equity?
    ('HYG', 'SPY', 'Credit → Equity'),
    ('JNK', 'SPY', 'Junk Bonds → Equity'),

    # Bonds lead equity?
    ('TLT', 'SPY', 'Long Bonds → Equity'),
    ('IEF', 'SPY', 'Mid Bonds → Equity'),

    # VIX leads equity?
    ('VIX', 'SPY', 'VIX → Equity'),

    # Gold leads equity?
    ('GLD', 'SPY', 'Gold → Equity'),
    ('GLD', 'TLT', 'Gold → Bonds'),

    # Sector rotation signals
    ('XLY', 'XLP', 'Consumer Disc → Staples (risk appetite)'),
    ('XLK', 'SPY', 'Tech → Broad Market'),
    ('XLF', 'SPY', 'Financials → Broad Market'),
    ('XLE', 'SPY', 'Energy → Broad Market'),

    # Small vs large
    ('IWM', 'SPY', 'Small Cap → Large Cap'),
    ('IWM', 'QQQ', 'Small Cap → Tech'),

    # EM leads DM?
    ('EEM', 'SPY', 'Emerging → US Equity'),

    # Commodities lead equity?
    ('DBC', 'SPY', 'Commodities → Equity'),
    ('USO', 'XLE', 'Oil → Energy Sector'),

    # Factor rotation
    ('MTUM', 'VLUE', 'Momentum → Value'),
    ('USMV', 'SPY', 'Min-Vol → Broad (defensive rotation)'),
]

print("\nComputing cross-asset lead/lag correlations...")
lead_lag_results = {}

for ticker_a, ticker_b, label in LEAD_LAG_PAIRS:
    if ticker_a not in returns.columns or ticker_b not in returns.columns:
        print(f"  SKIP {label} — missing data")
        continue

    ret_a = returns[ticker_a].dropna()
    ret_b = returns[ticker_b].dropna()

    ll = compute_lead_lag(ret_a, ret_b, max_lag=10)
    if not ll:
        continue

    # Find strongest leading signal
    best_lag = max(ll.keys(), key=lambda k: abs(ll[k]['corr']))
    best = ll[best_lag]

    # Also compute with weekly returns for cleaner signal
    wk_a = ret_a.resample('W').sum().dropna()
    wk_b = ret_b.resample('W').sum().dropna()
    ll_weekly = compute_lead_lag(wk_a, wk_b, max_lag=4, min_obs=100)

    lead_lag_results[label] = {
        'pair': f"{ticker_a} → {ticker_b}",
        'daily_best_lag': best_lag,
        'daily_best_corr': best['corr'],
        'daily_best_pval': best['pval'],
        'daily_lag0_corr': ll.get(0, {}).get('corr', None),
        'daily_all_lags': {str(k): v for k, v in ll.items()},
    }

    if ll_weekly:
        wk_best_lag = max(ll_weekly.keys(), key=lambda k: abs(ll_weekly[k]['corr']))
        lead_lag_results[label]['weekly_best_lag'] = wk_best_lag
        lead_lag_results[label]['weekly_best_corr'] = ll_weekly[wk_best_lag]['corr']

    direction = "LEADS" if best_lag > 0 else ("LAGS" if best_lag < 0 else "CONCURRENT")
    sig = "***" if best['pval'] < 0.001 else "**" if best['pval'] < 0.01 else "*" if best['pval'] < 0.05 else ""

    print(f"  {label}: {direction} by {abs(best_lag)}d, corr={best['corr']:.4f} {sig}")

# ============================================================================
# PART 2: EXPANDED SIGNAL SET — New predictive indicators
# ============================================================================

print("\n" + "=" * 80)
print("PART 2: EXPANDED SIGNAL COMPUTATION")
print("=" * 80)

signals = pd.DataFrame(index=prices.index)

# --- A. ORIGINAL 9 SIGNALS (from study) ---
if 'VIX' in prices.columns and 'VIX3M' in prices.columns:
    signals['vix_term_structure'] = prices['VIX'] / prices['VIX3M']
elif 'VIX' in prices.columns:
    signals['vix_term_structure'] = prices['VIX'] / prices['VIX'].rolling(63).mean()

if 'HYG' in returns.columns and 'LQD' in returns.columns:
    signals['credit_spread'] = (returns['HYG'] - returns['LQD']).rolling(21).sum()

available_sectors = [s for s in SECTOR_ETFS if s in prices.columns]
if len(available_sectors) >= 3:
    breadth = pd.DataFrame()
    for s in available_sectors:
        breadth[s] = (prices[s] > prices[s].rolling(50).mean()).astype(float)
    signals['market_breadth'] = breadth.mean(axis=1)

if 'SPY' in prices.columns:
    spy_6m = prices['SPY'].pct_change(126)
    etf_rets = pd.DataFrame({t: prices[t].pct_change(126) for t in CORE_ASSETS
                             if t in prices.columns and t != 'SPY'})
    if len(etf_rets.columns) > 0:
        signals['momentum_divergence'] = spy_6m - etf_rets.median(axis=1)

if 'SPY' in returns.columns:
    v21 = returns['SPY'].rolling(21).std() * np.sqrt(252)
    v63 = returns['SPY'].rolling(63).std() * np.sqrt(252)
    signals['vol_compression'] = v21 / v63

if 'VIX' in prices.columns and 'SPY' in returns.columns:
    rv21 = returns['SPY'].rolling(21).std() * np.sqrt(252) * 100
    signals['implied_vs_realized'] = prices['VIX'] / rv21

if 'SPY' in returns.columns and 'TLT' in returns.columns:
    signals['bond_equity_corr'] = returns['SPY'].rolling(63).corr(returns['TLT'])

if 'GLD' in prices.columns and 'SPY' in prices.columns:
    signals['gold_stress'] = prices['GLD'].pct_change(21) - prices['SPY'].pct_change(21)

if 'IWM' in prices.columns and 'SPY' in prices.columns:
    signals['smallcap_spread'] = prices['IWM'].pct_change(63) - prices['SPY'].pct_change(63)

print(f"  Original 9 signals: {len([c for c in signals.columns])} computed")

# --- B. NEW SIGNALS ---

# B1. SECTOR ROTATION MOMENTUM — which sectors are gaining/losing RS
print("  Computing sector rotation signals...")
if len(available_sectors) >= 5:
    for sec in available_sectors:
        if sec in prices.columns and 'SPY' in prices.columns:
            # Relative strength: sector 21d return - SPY 21d return
            rs_21 = prices[sec].pct_change(21) - prices['SPY'].pct_change(21)
            # RS acceleration: change in relative strength
            signals[f'rs_accel_{sec}'] = rs_21 - rs_21.shift(21)

    # Rotation breadth: how many sectors have IMPROVING relative strength
    rs_accel_cols = [c for c in signals.columns if c.startswith('rs_accel_')]
    if rs_accel_cols:
        signals['rotation_breadth'] = (signals[rs_accel_cols] > 0).sum(axis=1) / len(rs_accel_cols)
        print(f"    Rotation breadth: {len(rs_accel_cols)} sectors tracked")

# B2. RISK APPETITE INDEX — XLY/XLP ratio (consumer discretionary vs staples)
if 'XLY' in prices.columns and 'XLP' in prices.columns:
    risk_appetite = np.log(prices['XLY'] / prices['XLP'])
    signals['risk_appetite'] = risk_appetite - risk_appetite.rolling(63).mean()
    signals['risk_appetite_trend'] = risk_appetite.pct_change(21)
    print("    Risk appetite (XLY/XLP) — computed")

# B3. CREDIT QUALITY ROTATION — HYG vs LQD relative performance
if 'HYG' in prices.columns and 'LQD' in prices.columns:
    credit_quality = np.log(prices['HYG'] / prices['LQD'])
    signals['credit_quality_z'] = (credit_quality - credit_quality.rolling(63).mean()) / credit_quality.rolling(63).std()
    signals['credit_quality_accel'] = credit_quality.pct_change(10) - credit_quality.pct_change(10).shift(10)
    print("    Credit quality rotation — computed")

# B4. VOLATILITY REGIME SIGNALS
if 'VIX' in prices.columns:
    signals['vix_z_score'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()
    signals['vix_5d_change'] = prices['VIX'].pct_change(5)
    signals['vix_mean_reversion'] = prices['VIX'].rolling(5).mean() / prices['VIX'].rolling(63).mean()
    # VIX crush signal — VIX falling rapidly from elevated levels
    signals['vix_crush'] = (prices['VIX'].shift(5) - prices['VIX']) / prices['VIX'].shift(5)
    print("    VIX regime signals — computed")

# B5. YIELD CURVE PROXY — TLT/SHY or TLT/IEF
if 'TLT' in prices.columns and 'SHY' in prices.columns:
    yield_curve = np.log(prices['TLT'] / prices['SHY'])
    signals['yield_curve_slope'] = yield_curve - yield_curve.rolling(63).mean()
    signals['yield_curve_momentum'] = yield_curve.pct_change(21)
    print("    Yield curve proxy (TLT/SHY) — computed")
elif 'TLT' in prices.columns and 'IEF' in prices.columns:
    yield_curve = np.log(prices['TLT'] / prices['IEF'])
    signals['yield_curve_slope'] = yield_curve - yield_curve.rolling(63).mean()
    signals['yield_curve_momentum'] = yield_curve.pct_change(21)
    print("    Yield curve proxy (TLT/IEF) — computed")

# B6. GOLD-TO-EQUITY RATIO — safe haven demand
if 'GLD' in prices.columns and 'SPY' in prices.columns:
    g2e = np.log(prices['GLD'] / prices['SPY'])
    signals['gold_equity_ratio_z'] = (g2e - g2e.rolling(126).mean()) / g2e.rolling(126).std()
    signals['gold_equity_momentum'] = g2e.pct_change(21)
    print("    Gold/equity ratio — computed")

# B7. INTERNATIONAL DIVERGENCE — EEM and EFA relative to SPY
for intl in ['EEM', 'EFA']:
    if intl in prices.columns and 'SPY' in prices.columns:
        rel = np.log(prices[intl] / prices['SPY'])
        signals[f'{intl.lower()}_relative_z'] = (rel - rel.rolling(63).mean()) / rel.rolling(63).std()
        signals[f'{intl.lower()}_relative_mom'] = rel.pct_change(21)
        print(f"    {intl} relative strength — computed")

# B8. FACTOR MOMENTUM — which factors are winning
available_factors = [f for f in FACTOR_ETFS if f in prices.columns]
if len(available_factors) >= 2 and 'SPY' in prices.columns:
    for fac in available_factors:
        signals[f'factor_{fac}_rs'] = prices[fac].pct_change(21) - prices['SPY'].pct_change(21)
    print(f"    Factor momentum: {len(available_factors)} factors tracked")

# B9. VOLUME REGIME — SPY volume vs 20d average
if 'SPY' in prices.columns:
    # Use a proxy: price range as volatility measure
    spy_ret_abs = returns['SPY'].abs()
    signals['activity_regime'] = spy_ret_abs.rolling(5).mean() / spy_ret_abs.rolling(63).mean()
    print("    Activity regime — computed")

# B10. MULTI-ASSET MOMENTUM BREADTH — % of assets with positive 1m return
momentum_assets = [a for a in CORE_ASSETS + SECTOR_ETFS if a in prices.columns]
if len(momentum_assets) >= 5:
    mom_breadth = pd.DataFrame()
    for a in momentum_assets:
        mom_breadth[a] = (prices[a].pct_change(21) > 0).astype(float)
    signals['multi_asset_momentum_breadth'] = mom_breadth.mean(axis=1)
    # Also: % with positive 3m momentum
    mom_3m_breadth = pd.DataFrame()
    for a in momentum_assets:
        mom_3m_breadth[a] = (prices[a].pct_change(63) > 0).astype(float)
    signals['multi_asset_3m_breadth'] = mom_3m_breadth.mean(axis=1)
    print(f"    Multi-asset momentum breadth ({len(momentum_assets)} assets) — computed")

# B11. CORRELATION REGIME — average pairwise correlation of sectors
if len(available_sectors) >= 5:
    sector_returns = returns[available_sectors].dropna()
    # 63-day rolling average pairwise correlation
    def rolling_avg_corr(df, window=63):
        result = pd.Series(index=df.index, dtype=float)
        for i in range(window, len(df)):
            chunk = df.iloc[i-window:i]
            corr_matrix = chunk.corr()
            # Average of upper triangle
            mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
            avg_corr = corr_matrix.values[mask].mean()
            result.iloc[i] = avg_corr
        return result

    print("    Computing rolling sector correlation (this takes a moment)...")
    signals['sector_avg_corr'] = rolling_avg_corr(sector_returns, 63)
    print("    Sector average correlation — computed")

# B12. DRAWDOWN SIGNALS
if 'SPY' in prices.columns:
    spy_cummax = prices['SPY'].cummax()
    signals['spy_drawdown'] = (prices['SPY'] / spy_cummax) - 1
    signals['spy_drawdown_speed'] = signals['spy_drawdown'] - signals['spy_drawdown'].shift(5)
    print("    SPY drawdown signals — computed")

# B13. EARNINGS MOMENTUM PROXY — sector-level price acceleration around earnings seasons
# (Jan, Apr, Jul, Oct = heavy earnings months)
if 'SPY' in prices.columns:
    # Create earnings season indicator
    signals['earnings_season'] = prices.index.month.isin([1, 4, 7, 10]).astype(float)
    # Pre-earnings drift: 10d momentum going into earnings season
    signals['pre_earnings_drift'] = prices['SPY'].pct_change(10) * signals['earnings_season'].shift(-10).fillna(0)
    print("    Earnings season proxy — computed")

# Drop pure sector RS columns to keep signal set manageable
rs_cols = [c for c in signals.columns if c.startswith('rs_accel_')]
signals_clean = signals.drop(columns=rs_cols, errors='ignore')

print(f"\nTotal signals computed: {len(signals_clean.columns)}")
print(f"Signal list: {sorted(signals_clean.columns.tolist())}")

# ============================================================================
# PART 3: PREDICTIVE POWER RANKING — Which signals best predict forward returns?
# ============================================================================

print("\n" + "=" * 80)
print("PART 3: SIGNAL PREDICTIVE POWER RANKING")
print("=" * 80)

# Pre-compute forward returns
fwd = {}
for asset in ['SPY', 'QQQ', 'TLT', 'GLD', 'IWM']:
    if asset not in prices.columns:
        continue
    fwd[asset] = {}
    for label, days in FORWARD_WINDOWS.items():
        fwd[asset][label] = prices[asset].pct_change(days).shift(-days) * 100

# Rank signals by IC (information coefficient = Spearman rank correlation with fwd returns)
ic_results = []

for sig_name in signals_clean.columns:
    sig = signals_clean[sig_name].dropna()
    if len(sig) < 500:
        continue

    for asset in ['SPY', 'QQQ']:
        if asset not in fwd:
            continue
        for horizon_label in ['1m', '3m']:
            fwd_ret = fwd[asset][horizon_label]
            aligned = pd.DataFrame({'signal': sig, 'fwd': fwd_ret}).dropna()
            if len(aligned) < 300:
                continue

            ic, pval = stats.spearmanr(aligned['signal'], aligned['fwd'])
            ic_results.append({
                'signal': sig_name,
                'asset': asset,
                'horizon': horizon_label,
                'IC': round(ic, 4),
                'abs_IC': abs(round(ic, 4)),
                'p_value': round(pval, 6),
                'n_obs': len(aligned),
                'significant': pval < 0.05
            })

ic_df = pd.DataFrame(ic_results)
if len(ic_df) > 0:
    ic_df = ic_df.sort_values('abs_IC', ascending=False)

    print("\nTop 20 Signal-Asset-Horizon combinations by |IC|:")
    print("-" * 90)
    for _, row in ic_df.head(20).iterrows():
        sig_flag = "***" if row['p_value'] < 0.001 else "**" if row['p_value'] < 0.01 else "*" if row['p_value'] < 0.05 else ""
        direction = "+" if row['IC'] > 0 else "-"
        print(f"  {row['signal']:35s} → {row['asset']} {row['horizon']}: IC={row['IC']:+.4f} {sig_flag} (n={row['n_obs']})")

# ============================================================================
# PART 4: CONDITIONAL FORWARD RETURNS — Quintile analysis for top signals
# ============================================================================

print("\n" + "=" * 80)
print("PART 4: CONDITIONAL FORWARD RETURN ANALYSIS (Top Signals)")
print("=" * 80)

# Take top 15 signals by IC
if len(ic_df) > 0:
    top_signals = ic_df.groupby('signal')['abs_IC'].max().nlargest(15).index.tolist()
else:
    top_signals = list(signals_clean.columns[:15])

conditional_results = []

for sig_name in top_signals:
    sig = signals_clean[sig_name].dropna()
    if len(sig) < 500:
        continue

    try:
        quintiles = pd.qcut(sig, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
    except ValueError:
        try:
            quintiles = pd.cut(sig, 5, labels=['Q1', 'Q2', 'Q3', 'Q4', 'Q5'], duplicates='drop')
        except Exception:
            continue

    for asset in ['SPY', 'QQQ', 'GLD', 'TLT']:
        if asset not in fwd:
            continue
        for horizon_label in ['1m', '3m', '6m']:
            if horizon_label not in fwd[asset]:
                continue
            fwd_ret = fwd[asset][horizon_label]
            aligned = pd.DataFrame({'q': quintiles, 'fwd': fwd_ret}).dropna()
            if len(aligned) < 200:
                continue

            for q_label in ['Q1', 'Q5']:
                q_data = aligned[aligned['q'] == q_label]['fwd']
                if len(q_data) < 30:
                    continue
                unconditional = aligned['fwd']
                t_stat, p_val = stats.ttest_ind(q_data, unconditional, equal_var=False)

                p10 = q_data.quantile(0.10)
                p90 = q_data.quantile(0.90)
                asym = p90 / abs(p10) if abs(p10) > 0.01 else np.nan

                conditional_results.append({
                    'signal': sig_name,
                    'condition': f'{q_label} ({"low" if q_label == "Q1" else "high"})',
                    'asset': asset,
                    'horizon': horizon_label,
                    'mean_fwd_pct': round(q_data.mean(), 3),
                    'median_fwd_pct': round(q_data.median(), 3),
                    'p10': round(p10, 3),
                    'p90': round(p90, 3),
                    'asymmetry_ratio': round(asym, 2) if not np.isnan(asym) else None,
                    'pct_positive': round((q_data > 0).mean() * 100, 1),
                    'n': len(q_data),
                    'p_value': round(p_val, 5),
                })

cond_df = pd.DataFrame(conditional_results)
if len(cond_df) > 0:
    # Filter to significant + high asymmetry
    sig_asym = cond_df[(cond_df['p_value'] < 0.05) & (cond_df['asymmetry_ratio'].notna()) & (cond_df['asymmetry_ratio'] > 2)].copy()
    sig_asym = sig_asym.sort_values('asymmetry_ratio', ascending=False)

    print(f"\nSignificant asymmetric setups (p<0.05, asymmetry>2x):")
    print("-" * 100)
    for _, row in sig_asym.head(25).iterrows():
        print(f"  {row['signal']:30s} {row['condition']:12s} → {row['asset']} {row['horizon']}: "
              f"mean={row['mean_fwd_pct']:+.2f}%, asym={row['asymmetry_ratio']:.1f}x, "
              f"win={row['pct_positive']:.0f}%, p10={row['p10']:+.1f}%, p90={row['p90']:+.1f}% (n={row['n']})")

# ============================================================================
# PART 5: STOCK-LEVEL DISTRESS/RECOVERY SCREEN
# ============================================================================

print("\n" + "=" * 80)
print("PART 5: STOCK-LEVEL DISTRESS/RECOVERY SIGNALS")
print("=" * 80)

stock_signals = []
available_stocks = [s for s in STOCK_UNIVERSE if s in prices.columns]

for stock in available_stocks:
    p = prices[stock].dropna()
    if len(p) < 252:
        continue

    current = p.iloc[-1]
    high_52w = p.iloc[-252:].max()
    low_52w = p.iloc[-252:].min()
    sma_50 = p.iloc[-50:].mean()
    sma_200 = p.iloc[-200:].mean() if len(p) >= 200 else None
    mom_1m = (current / p.iloc[-21] - 1) * 100 if len(p) >= 21 else None
    mom_3m = (current / p.iloc[-63] - 1) * 100 if len(p) >= 63 else None
    mom_6m = (current / p.iloc[-126] - 1) * 100 if len(p) >= 126 else None
    dist_high = (current / high_52w - 1) * 100
    r = p.pct_change().dropna()
    vol_20d = r.iloc[-20:].std() * np.sqrt(252) * 100 if len(r) >= 20 else None

    # Mean reversion score: how far below SMA200 + negative momentum
    mr_score = 0
    if sma_200 and current < sma_200:
        mr_score += abs(current / sma_200 - 1) * 100
    if mom_3m and mom_3m < -10:
        mr_score += abs(mom_3m)
    if dist_high < -20:
        mr_score += abs(dist_high) * 0.5

    # Recovery signal: was deeply distressed, now showing momentum reversal
    recovery = False
    if mom_3m and mom_1m and mom_3m < -15 and mom_1m > 0:
        recovery = True

    stock_signals.append({
        'ticker': stock,
        'price': round(current, 2),
        'dist_52w_high_pct': round(dist_high, 1),
        'mom_1m_pct': round(mom_1m, 1) if mom_1m else None,
        'mom_3m_pct': round(mom_3m, 1) if mom_3m else None,
        'mom_6m_pct': round(mom_6m, 1) if mom_6m else None,
        'vol_20d_ann_pct': round(vol_20d, 1) if vol_20d else None,
        'above_50sma': current > sma_50,
        'above_200sma': current > sma_200 if sma_200 else None,
        'distress_score': round(mr_score, 1),
        'recovery_signal': recovery,
    })

stock_df = pd.DataFrame(stock_signals)
if len(stock_df) > 0:
    # Top distressed stocks (potential asymmetric long)
    distressed = stock_df[stock_df['distress_score'] > 15].sort_values('distress_score', ascending=False)
    print(f"\nDistressed stocks (potential asymmetric upside on recovery):")
    for _, row in distressed.head(10).iterrows():
        recovery_flag = " 🔄 RECOVERY" if row['recovery_signal'] else ""
        print(f"  {row['ticker']:6s}: {row['dist_52w_high_pct']:+.1f}% from high, "
              f"1m={row['mom_1m_pct']:+.1f}%, 3m={row['mom_3m_pct']:+.1f}%, "
              f"vol={row['vol_20d_ann_pct']:.0f}%, distress={row['distress_score']:.0f}{recovery_flag}")

    # Momentum leaders (breakout candidates)
    leaders = stock_df[(stock_df['mom_1m_pct'] > 5) & (stock_df['above_50sma'] == True)].sort_values('mom_1m_pct', ascending=False)
    print(f"\nMomentum leaders (strong uptrend):")
    for _, row in leaders.head(10).iterrows():
        print(f"  {row['ticker']:6s}: 1m={row['mom_1m_pct']:+.1f}%, 3m={row['mom_3m_pct']:+.1f}%, "
              f"{row['dist_52w_high_pct']:+.1f}% from high")

# ============================================================================
# PART 6: CURRENT SIGNAL READINGS — Where are we NOW?
# ============================================================================

print("\n" + "=" * 80)
print("PART 6: CURRENT SIGNAL READINGS")
print("=" * 80)

current_readings = {}
for sig_name in signals_clean.columns:
    sig = signals_clean[sig_name].dropna()
    if len(sig) < 100:
        continue
    current_val = sig.iloc[-1]
    percentile = stats.percentileofscore(sig.iloc[-252*5:] if len(sig) > 252*5 else sig, current_val)  # 5yr percentile
    current_readings[sig_name] = {
        'value': round(float(current_val), 4),
        'percentile_5y': round(percentile, 1),
        'z_score': round((current_val - sig.mean()) / sig.std(), 2) if sig.std() > 0 else 0,
    }

# Sort by extremeness
sorted_readings = sorted(current_readings.items(), key=lambda x: abs(x[1]['z_score']), reverse=True)

print("\nCurrent signal readings (sorted by extremeness):")
print("-" * 70)
for name, vals in sorted_readings[:20]:
    zone = "EXTREME" if abs(vals['z_score']) > 2 else "ELEVATED" if abs(vals['z_score']) > 1 else "NORMAL"
    print(f"  {name:35s}: {vals['value']:+8.4f} (pctile={vals['percentile_5y']:5.1f}%, z={vals['z_score']:+.2f}) [{zone}]")

# ============================================================================
# PART 7: SAVE RESULTS
# ============================================================================

print("\n" + "=" * 80)
print("SAVING RESULTS")
print("=" * 80)

# Save lead/lag
with open(os.path.join(OUT_DIR, 'lead_lag_results.json'), 'w') as f:
    json.dump(lead_lag_results, f, indent=2, default=str)

# Save IC rankings
if len(ic_df) > 0:
    ic_df.to_csv(os.path.join(OUT_DIR, 'signal_ic_rankings.csv'), index=False)

# Save conditional results
if len(cond_df) > 0:
    cond_df.to_csv(os.path.join(OUT_DIR, 'conditional_forward_returns.csv'), index=False)

# Save stock signals
if len(stock_df) > 0:
    stock_df.to_csv(os.path.join(OUT_DIR, 'stock_signals.csv'), index=False)

# Save current readings
with open(os.path.join(OUT_DIR, 'current_readings.json'), 'w') as f:
    json.dump(current_readings, f, indent=2, default=str)

# Save comprehensive summary
summary = {
    'run_date': datetime.now().isoformat(),
    'data_range': f"{prices.index[0].date()} to {prices.index[-1].date()}",
    'n_signals': len(signals_clean.columns),
    'n_assets': len(prices.columns),
    'lead_lag_summary': {},
    'top_predictive_signals': [],
    'current_regime': {},
    'stock_distress_count': len(distressed) if len(stock_df) > 0 and len(distressed) > 0 else 0,
    'stock_recovery_count': len(stock_df[stock_df['recovery_signal'] == True]) if len(stock_df) > 0 else 0,
}

# Summarize lead/lag findings
for label, data in lead_lag_results.items():
    lag = data['daily_best_lag']
    corr = data['daily_best_corr']
    summary['lead_lag_summary'][label] = {
        'relationship': 'LEADS' if lag > 0 else 'LAGS' if lag < 0 else 'CONCURRENT',
        'lag_days': abs(lag),
        'correlation': corr,
    }

# Top predictive signals
if len(ic_df) > 0:
    for _, row in ic_df.head(10).iterrows():
        summary['top_predictive_signals'].append({
            'signal': row['signal'],
            'asset': row['asset'],
            'horizon': row['horizon'],
            'IC': row['IC'],
        })

# Current regime
extreme_signals = [name for name, vals in current_readings.items() if abs(vals['z_score']) > 1.5]
summary['current_regime'] = {
    'extreme_signals': extreme_signals,
    'n_extreme': len(extreme_signals),
    'overall_assessment': 'STRESSED' if len(extreme_signals) > 5 else 'ELEVATED' if len(extreme_signals) > 2 else 'CALM'
}

with open(os.path.join(OUT_DIR, 'comprehensive_summary.json'), 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\nAll results saved to {OUT_DIR}/")
print("Files: lead_lag_results.json, signal_ic_rankings.csv, conditional_forward_returns.csv,")
print("       stock_signals.csv, current_readings.json, comprehensive_summary.json")

# ============================================================================
# FINAL: PLAIN-ENGLISH SUMMARY
# ============================================================================

print("\n" + "=" * 80)
print("PLAIN-ENGLISH SUMMARY")
print("=" * 80)

print("""
KEY FINDINGS — LEAD/LAG RELATIONSHIPS:
""")

# Print the most important lead/lag findings
for label, data in sorted(lead_lag_results.items(), key=lambda x: abs(x[1]['daily_best_corr']), reverse=True)[:10]:
    lag = data['daily_best_lag']
    corr = data['daily_best_corr']
    pair = data['pair']
    if lag > 0:
        print(f"  {pair} — {pair.split(' → ')[0]} LEADS by {lag} days (corr {corr:+.4f})")
    elif lag < 0:
        print(f"  {pair} — {pair.split(' → ')[0]} LAGS by {abs(lag)} days (corr {corr:+.4f})")
    else:
        print(f"  {pair} — CONCURRENT (corr {corr:+.4f})")

print(f"""
CURRENT MARKET REGIME:
  Extreme signals firing: {len(extreme_signals)}
  Assessment: {summary['current_regime']['overall_assessment']}
  Signals at extremes: {', '.join(extreme_signals[:5]) if extreme_signals else 'None'}

STOCK-LEVEL:
  Distressed names: {summary.get('stock_distress_count', 0)}
  Recovery signals: {summary.get('stock_recovery_count', 0)}
""")

print("DONE.")
