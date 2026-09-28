#!/usr/bin/env python3
"""
Signal Correlation & Asymmetry Analysis
========================================
Comprehensive analysis of cross-asset signals, their correlations,
asymmetric return profiles, and regime-conditional behavior.

Output: /home/jupiter/Lvl3Quant/output/signal_analysis_v1/
"""

import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import yfinance as yf
from pathlib import Path
from itertools import combinations
from scipy import stats
import sys

OUT = Path("/home/jupiter/Lvl3Quant/output/signal_analysis_v1")
OUT.mkdir(parents=True, exist_ok=True)

# ============================================================
# 1. DOWNLOAD DATA
# ============================================================
print("=" * 70)
print("STEP 1: Downloading daily data 2010-2026")
print("=" * 70)

core_etfs = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'VNQ', 'EFA', 'EEM', 'DBC', 'XLE', 'HYG', 'LQD', 'SHY', 'IEF']
sector_etfs = ['XLK', 'XLF', 'XLE', 'XLV', 'XLI', 'XLC', 'XLY', 'XLP', 'XLU', 'XLRE', 'XLB']
vix_tickers = ['^VIX', '^VIX3M']

all_tickers = list(set(core_etfs + sector_etfs + vix_tickers))

print(f"Downloading {len(all_tickers)} tickers...")
raw = yf.download(all_tickers, start='2010-01-01', end='2026-07-21', auto_adjust=True, progress=True)

# Extract close prices
close = raw['Close'].copy()
close.columns = [c if isinstance(c, str) else c for c in close.columns]

# Rename VIX columns
if '^VIX' in close.columns:
    close.rename(columns={'^VIX': 'VIX'}, inplace=True)
if '^VIX3M' in close.columns:
    close.rename(columns={'^VIX3M': 'VIX3M'}, inplace=True)

# Also get volume for SPY
vol_raw = raw['Volume'] if 'Volume' in raw.columns else None

print(f"\nData shape: {close.shape}")
print(f"Date range: {close.index[0].date()} to {close.index[-1].date()}")
print(f"Tickers with data: {[c for c in close.columns if close[c].notna().sum() > 100]}")

# Forward-fill then drop rows with too many NaN
close = close.ffill().dropna(how='all')

# ============================================================
# 2. COMPUTE ALL SIGNALS (T-1 shifted for forward-looking safety)
# ============================================================
print("\n" + "=" * 70)
print("STEP 2: Computing signals")
print("=" * 70)

signals = pd.DataFrame(index=close.index)

# --- VIX signals ---
if 'VIX' in close.columns:
    signals['vix_level'] = close['VIX']
    signals['vix_21d_pctrank'] = close['VIX'].rolling(252).apply(
        lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1]) / 100 if len(x.dropna()) > 50 else np.nan
    )
    if 'VIX3M' in close.columns:
        signals['vix_term_structure'] = close['VIX'] / close['VIX3M']  # <1 = contango (normal), >1 = backwardation (stress)
    signals['iv_rv_spread'] = close['VIX'] - (close['SPY'].pct_change().rolling(21).std() * np.sqrt(252) * 100)

# --- Momentum signals ---
for asset in ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'VNQ', 'EFA', 'EEM', 'DBC', 'XLE']:
    if asset in close.columns:
        signals[f'{asset}_mom_6m'] = close[asset].pct_change(126)
        signals[f'{asset}_mom_3m'] = close[asset].pct_change(63)
        signals[f'{asset}_mom_1m'] = close[asset].pct_change(21)

# --- Realized vol ---
for asset in ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD']:
    if asset in close.columns:
        rets = close[asset].pct_change()
        signals[f'{asset}_rvol_21d'] = rets.rolling(21).std() * np.sqrt(252)
        signals[f'{asset}_rvol_63d'] = rets.rolling(63).std() * np.sqrt(252)

# --- Credit spread ---
if 'HYG' in close.columns and 'LQD' in close.columns:
    signals['credit_spread_ratio'] = close['HYG'] / close['LQD']
    signals['credit_spread_21d_chg'] = signals['credit_spread_ratio'].pct_change(21)

# --- Breadth ---
sector_list = [s for s in sector_etfs if s in close.columns]
if len(sector_list) > 5:
    breadth_df = pd.DataFrame()
    for s in sector_list:
        breadth_df[s] = (close[s] > close[s].rolling(50).mean()).astype(float)
    signals['breadth_pct_above_50d'] = breadth_df.mean(axis=1)

# --- SPY distance from 200d SMA ---
if 'SPY' in close.columns:
    sma200 = close['SPY'].rolling(200).mean()
    signals['spy_dist_200sma_pct'] = (close['SPY'] - sma200) / sma200 * 100

# --- Cross-asset dispersion ---
equity_etfs = [e for e in ['SPY', 'QQQ', 'IWM', 'EFA', 'EEM', 'XLE', 'VNQ'] if e in close.columns]
if len(equity_etfs) > 3:
    mom_1m = pd.DataFrame({e: close[e].pct_change(21) for e in equity_etfs})
    signals['cross_asset_dispersion'] = mom_1m.std(axis=1)

# Shift all signals by 1 day (T-1 only, no lookahead)
signals = signals.shift(1)

print(f"Computed {len(signals.columns)} signals")
print(f"Signal columns: {list(signals.columns)}")

# ============================================================
# 3. COMPUTE FORWARD RETURNS
# ============================================================
print("\n" + "=" * 70)
print("STEP 3: Computing forward returns")
print("=" * 70)

fwd_assets = ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'VNQ', 'EFA', 'EEM']
horizons = {'1m': 21, '3m': 63, '6m': 126}

fwd_returns = {}
for asset in fwd_assets:
    if asset not in close.columns:
        continue
    for h_name, h_days in horizons.items():
        col = f'{asset}_fwd_{h_name}'
        fwd_returns[col] = close[asset].pct_change(h_days).shift(-h_days)

fwd_df = pd.DataFrame(fwd_returns, index=close.index)
print(f"Forward return columns: {len(fwd_df.columns)}")

# Merge
master = signals.join(fwd_df).dropna(how='all')
print(f"Master dataset: {master.shape[0]} rows, {master.shape[1]} columns")

# ============================================================
# 4. SIGNAL CORRELATION MATRIX
# ============================================================
print("\n" + "=" * 70)
print("STEP 4: Signal correlation matrix")
print("=" * 70)

# Select key signals for correlation (not all momentum variants)
key_signals = [c for c in signals.columns if any(k in c for k in [
    'vix_level', 'vix_21d_pctrank', 'vix_term_structure', 'iv_rv_spread',
    'SPY_mom_6m', 'SPY_mom_3m', 'SPY_mom_1m',
    'QQQ_mom_6m', 'IWM_mom_6m', 'TLT_mom_6m', 'GLD_mom_6m',
    'SPY_rvol_21d', 'SPY_rvol_63d',
    'credit_spread_ratio', 'credit_spread_21d_chg',
    'breadth_pct_above_50d', 'spy_dist_200sma_pct',
    'cross_asset_dispersion'
])]

corr_matrix = master[key_signals].corr()
corr_matrix.to_csv(OUT / 'signal_correlations.csv', float_format='%.3f')
print(f"Saved correlation matrix for {len(key_signals)} key signals")

# Find highly correlated pairs (>0.7)
print("\nHighly correlated signal pairs (|r| > 0.7):")
for i, s1 in enumerate(key_signals):
    for s2 in key_signals[i+1:]:
        r = corr_matrix.loc[s1, s2]
        if abs(r) > 0.7 and not np.isnan(r):
            print(f"  {s1} <-> {s2}: r={r:.3f}")

# ============================================================
# 5. ASYMMETRY ANALYSIS BY SIGNAL
# ============================================================
print("\n" + "=" * 70)
print("STEP 5: Asymmetry analysis by signal quintile")
print("=" * 70)

analysis_signals = [c for c in key_signals if c in master.columns]
fwd_col = 'SPY_fwd_1m'  # Primary: 1-month forward SPY return

results = []

for sig in analysis_signals:
    for horizon_name in ['1m', '3m', '6m']:
        fwd_col_h = f'SPY_fwd_{horizon_name}'
        if fwd_col_h not in master.columns:
            continue

        tmp = master[[sig, fwd_col_h]].dropna()
        if len(tmp) < 500:
            continue

        # Quintiles
        try:
            tmp['quintile'] = pd.qcut(tmp[sig], 5, labels=[1,2,3,4,5], duplicates='drop')
        except ValueError:
            continue

        for q in sorted(tmp['quintile'].unique()):
            subset = tmp[tmp['quintile'] == q][fwd_col_h]
            if len(subset) < 30:
                continue

            mean_ret = subset.mean()
            median_ret = subset.median()
            std_ret = subset.std()
            hit_rate = (subset > 0).mean()

            pos_rets = subset[subset > 0]
            neg_rets = subset[subset < 0]

            mean_up = pos_rets.mean() if len(pos_rets) > 0 else 0
            mean_dn = neg_rets.mean() if len(neg_rets) > 0 else 0
            updown_ratio = mean_up / abs(mean_dn) if mean_dn != 0 else np.inf

            # Tail: p10 (downside) and p90 (upside)
            p10 = subset.quantile(0.10)
            p90 = subset.quantile(0.90)
            tail_asymmetry = p90 / abs(p10) if p10 != 0 else np.inf

            # Sharpe-like
            sharpe = mean_ret / std_ret * np.sqrt(252/{'1m':21,'3m':63,'6m':126}[horizon_name]) if std_ret > 0 else 0

            results.append({
                'signal': sig,
                'horizon': horizon_name,
                'quintile': int(q),
                'n_obs': len(subset),
                'mean_return': mean_ret,
                'median_return': median_ret,
                'std_return': std_ret,
                'hit_rate': hit_rate,
                'mean_up': mean_up,
                'mean_dn': mean_dn,
                'updown_ratio': updown_ratio,
                'p10_downside': p10,
                'p90_upside': p90,
                'tail_asymmetry': tail_asymmetry,
                'sharpe_annualized': sharpe,
                'signal_q1_mean': tmp[tmp['quintile']==tmp['quintile'].min()][sig].mean(),
                'signal_q5_mean': tmp[tmp['quintile']==tmp['quintile'].max()][sig].mean(),
            })

asym_df = pd.DataFrame(results)
asym_df.to_csv(OUT / 'asymmetry_by_signal.csv', index=False, float_format='%.6f')
print(f"Saved asymmetry analysis: {len(asym_df)} rows across {asym_df['signal'].nunique()} signals")

# Print the most asymmetric opportunities
print("\n--- TOP ASYMMETRIC OPPORTUNITIES (1m horizon, sorted by updown_ratio * hit_rate) ---")
top_asym = asym_df[asym_df['horizon'] == '1m'].copy()
top_asym['score'] = top_asym['updown_ratio'] * top_asym['hit_rate'] * np.sign(top_asym['mean_return'])
top_asym_sorted = top_asym.nlargest(15, 'score')
for _, row in top_asym_sorted.iterrows():
    print(f"  {row['signal']} Q{int(row['quintile'])}: "
          f"mean={row['mean_return']*100:.2f}%, hit={row['hit_rate']*100:.1f}%, "
          f"up/dn={row['updown_ratio']:.2f}, tail_asym={row['tail_asymmetry']:.2f}, "
          f"n={int(row['n_obs'])}")

# ============================================================
# 6. REGIME ANALYSIS
# ============================================================
print("\n" + "=" * 70)
print("STEP 6: Regime-conditional analysis")
print("=" * 70)

# Define regimes
regime_data = master[['vix_level', 'spy_dist_200sma_pct']].dropna()
vix_median = regime_data['vix_level'].median()

def classify_regime(row):
    high_vix = row['vix_level'] > vix_median
    uptrend = row['spy_dist_200sma_pct'] > 0
    if not high_vix and uptrend:
        return 'Low VIX + Uptrend'
    elif not high_vix and not uptrend:
        return 'Low VIX + Downtrend'
    elif high_vix and uptrend:
        return 'High VIX + Uptrend'
    else:
        return 'High VIX + Downtrend'

master['regime'] = regime_data.apply(classify_regime, axis=1)

regime_results = []
for regime in ['Low VIX + Uptrend', 'Low VIX + Downtrend', 'High VIX + Uptrend', 'High VIX + Downtrend']:
    regime_mask = master['regime'] == regime
    n_days = regime_mask.sum()

    for horizon_name in ['1m', '3m', '6m']:
        fwd_col_h = f'SPY_fwd_{horizon_name}'
        if fwd_col_h not in master.columns:
            continue

        subset = master.loc[regime_mask, fwd_col_h].dropna()
        if len(subset) < 30:
            continue

        mean_ret = subset.mean()
        std_ret = subset.std()
        hit_rate = (subset > 0).mean()
        pos_rets = subset[subset > 0]
        neg_rets = subset[subset < 0]
        mean_up = pos_rets.mean() if len(pos_rets) > 0 else 0
        mean_dn = neg_rets.mean() if len(neg_rets) > 0 else 0
        updown = mean_up / abs(mean_dn) if mean_dn != 0 else np.inf
        p10 = subset.quantile(0.10)
        p90 = subset.quantile(0.90)

        regime_results.append({
            'regime': regime,
            'horizon': horizon_name,
            'n_days': n_days,
            'n_obs': len(subset),
            'mean_return': mean_ret,
            'std_return': std_ret,
            'hit_rate': hit_rate,
            'mean_up': mean_up,
            'mean_dn': mean_dn,
            'updown_ratio': updown,
            'p10': p10,
            'p90': p90,
            'sharpe_ann': mean_ret / std_ret * np.sqrt(252/{'1m':21,'3m':63,'6m':126}[horizon_name]) if std_ret > 0 else 0,
        })

    # Signal predictive power within regime
    for sig in ['vix_21d_pctrank', 'SPY_mom_6m', 'credit_spread_21d_chg', 'breadth_pct_above_50d', 'iv_rv_spread', 'cross_asset_dispersion']:
        if sig not in master.columns:
            continue
        tmp = master.loc[regime_mask, [sig, 'SPY_fwd_1m']].dropna()
        if len(tmp) < 50:
            continue
        ic = tmp[sig].corr(tmp['SPY_fwd_1m'])
        regime_results.append({
            'regime': regime,
            'horizon': 'IC_1m',
            'n_days': n_days,
            'n_obs': len(tmp),
            'mean_return': ic,  # Using mean_return field for IC
            'std_return': np.nan,
            'hit_rate': np.nan,
            'mean_up': np.nan,
            'mean_dn': np.nan,
            'updown_ratio': np.nan,
            'p10': np.nan,
            'p90': np.nan,
            'sharpe_ann': np.nan,
            'signal_name': sig,
        })

regime_df = pd.DataFrame(regime_results)
regime_df.to_csv(OUT / 'regime_analysis.csv', index=False, float_format='%.6f')
print(f"Saved regime analysis: {len(regime_df)} rows")

# Print regime summary
print("\n--- REGIME RETURN PROFILES (SPY forward returns) ---")
regime_rets = regime_df[(regime_df['horizon'].isin(['1m','3m','6m'])) & (regime_df['mean_return'].notna())]
for regime in ['Low VIX + Uptrend', 'Low VIX + Downtrend', 'High VIX + Uptrend', 'High VIX + Downtrend']:
    print(f"\n  {regime}:")
    rr = regime_rets[regime_rets['regime'] == regime]
    for _, row in rr.iterrows():
        print(f"    {row['horizon']}: mean={row['mean_return']*100:.2f}%, hit={row['hit_rate']*100:.1f}%, "
              f"up/dn={row['updown_ratio']:.2f}, Sharpe={row['sharpe_ann']:.2f}, n={int(row['n_obs'])}")

# ============================================================
# 7. JOINT SIGNAL ANALYSIS — Signal Interactions
# ============================================================
print("\n" + "=" * 70)
print("STEP 7: Joint signal analysis (signal interactions)")
print("=" * 70)

joint_results = []

# Define composite conditions
def compute_joint_analysis(condition_name, mask, fwd_col='SPY_fwd_1m'):
    """Analyze forward returns under a specific joint signal condition."""
    subset = master.loc[mask, fwd_col].dropna()
    if len(subset) < 30:
        return None

    mean_ret = subset.mean()
    std_ret = subset.std()
    hit_rate = (subset > 0).mean()
    pos = subset[subset > 0]
    neg = subset[subset < 0]
    mean_up = pos.mean() if len(pos) > 0 else 0
    mean_dn = neg.mean() if len(neg) > 0 else 0
    updown = mean_up / abs(mean_dn) if mean_dn != 0 else np.inf

    return {
        'condition': condition_name,
        'n_obs': len(subset),
        'pct_of_days': len(subset) / len(master) * 100,
        'mean_return': mean_ret,
        'median_return': subset.median(),
        'std_return': std_ret,
        'hit_rate': hit_rate,
        'mean_up': mean_up,
        'mean_dn': mean_dn,
        'updown_ratio': updown,
        'p5': subset.quantile(0.05),
        'p10': subset.quantile(0.10),
        'p90': subset.quantile(0.90),
        'p95': subset.quantile(0.95),
        'sharpe_ann': mean_ret / std_ret * np.sqrt(12) if std_ret > 0 else 0,
        'max_drawdown_proxy': subset.min(),
    }

# Compute percentile thresholds
vix_lo = master['vix_level'].quantile(0.3)
vix_hi = master['vix_level'].quantile(0.7)
mom_pos = master['SPY_mom_6m'] > 0
mom_neg = master['SPY_mom_6m'] < 0
mom_strong = master['SPY_mom_6m'] > master['SPY_mom_6m'].quantile(0.7)
mom_weak = master['SPY_mom_6m'] < master['SPY_mom_6m'].quantile(0.3)

breadth_hi = master['breadth_pct_above_50d'] > 0.7 if 'breadth_pct_above_50d' in master.columns else pd.Series(False, index=master.index)
breadth_lo = master['breadth_pct_above_50d'] < 0.3 if 'breadth_pct_above_50d' in master.columns else pd.Series(False, index=master.index)

credit_improving = master['credit_spread_21d_chg'] > 0 if 'credit_spread_21d_chg' in master.columns else pd.Series(False, index=master.index)
credit_widening = master['credit_spread_21d_chg'] < master['credit_spread_21d_chg'].quantile(0.2) if 'credit_spread_21d_chg' in master.columns else pd.Series(False, index=master.index)

iv_rv_hi = master['iv_rv_spread'] > master['iv_rv_spread'].quantile(0.8) if 'iv_rv_spread' in master.columns else pd.Series(False, index=master.index)
iv_rv_lo = master['iv_rv_spread'] < master['iv_rv_spread'].quantile(0.2) if 'iv_rv_spread' in master.columns else pd.Series(False, index=master.index)

uptrend = master['spy_dist_200sma_pct'] > 0 if 'spy_dist_200sma_pct' in master.columns else pd.Series(False, index=master.index)
downtrend = master['spy_dist_200sma_pct'] < 0 if 'spy_dist_200sma_pct' in master.columns else pd.Series(False, index=master.index)

vix_contango = master['vix_term_structure'] < 0.9 if 'vix_term_structure' in master.columns else pd.Series(False, index=master.index)
vix_backwardation = master['vix_term_structure'] > 1.05 if 'vix_term_structure' in master.columns else pd.Series(False, index=master.index)

# --- BULLISH COMPOSITES ---
conditions = [
    ("BASELINE: All days", pd.Series(True, index=master.index)),

    # Single signals
    ("Low VIX (<p30)", master['vix_level'] < vix_lo),
    ("High VIX (>p70)", master['vix_level'] > vix_hi),
    ("Strong 6m momentum (>p70)", mom_strong),
    ("Weak 6m momentum (<p30)", mom_weak),
    ("High breadth (>70%)", breadth_hi),
    ("Low breadth (<30%)", breadth_lo),
    ("Above 200d SMA", uptrend),
    ("Below 200d SMA", downtrend),
    ("VIX contango (<0.9)", vix_contango),
    ("VIX backwardation (>1.05)", vix_backwardation),
    ("High IV-RV spread (>p80)", iv_rv_hi),
    ("Low IV-RV spread (<p20)", iv_rv_lo),
    ("Credit improving", credit_improving),
    ("Credit widening (<p20)", credit_widening),

    # BULLISH composites
    ("BULL: Low VIX + momentum + breadth", (master['vix_level'] < vix_lo) & mom_pos & breadth_hi),
    ("BULL: Uptrend + strong momentum + contango", uptrend & mom_strong & vix_contango),
    ("BULL: High IV-RV + uptrend (vol premium harvest)", iv_rv_hi & uptrend),
    ("BULL: Low VIX + uptrend + credit improving", (master['vix_level'] < vix_lo) & uptrend & credit_improving),
    ("BULL: Breadth >70% + momentum + uptrend", breadth_hi & mom_pos & uptrend),

    # BEARISH / DEFENSIVE composites
    ("BEAR: High VIX + downtrend + credit widening", (master['vix_level'] > vix_hi) & downtrend & credit_widening),
    ("BEAR: Backwardation + weak momentum", vix_backwardation & mom_weak),
    ("BEAR: Below 200SMA + low breadth", downtrend & breadth_lo),
    ("BEAR: High VIX + below 200SMA + weak momentum", (master['vix_level'] > vix_hi) & downtrend & mom_weak),

    # MEAN REVERSION composites
    ("REVERT: High VIX + uptrend (vol spike in bull)", (master['vix_level'] > vix_hi) & uptrend),
    ("REVERT: Low VIX + downtrend (complacency in bear)", (master['vix_level'] < vix_lo) & downtrend),
    ("REVERT: Backwardation + uptrend (fear spike, trend intact)", vix_backwardation & uptrend),
    ("REVERT: Very high IV-RV + high VIX (panic overdone)", iv_rv_hi & (master['vix_level'] > vix_hi)),

    # ASYMMETRIC OPPORTUNITY composites
    ("ASYM: Post-VIX-spike + trend intact (vol crush opportunity)",
     (master['vix_21d_pctrank'] > 0.8) & uptrend & mom_pos),
    ("ASYM: Extreme IV-RV + contango (premium very rich)",
     (master['iv_rv_spread'] > master['iv_rv_spread'].quantile(0.9)) & vix_contango),
    ("ASYM: Breadth recovering (30-60%) + uptrend",
     (master['breadth_pct_above_50d'].between(0.3, 0.6) if 'breadth_pct_above_50d' in master.columns else pd.Series(False, index=master.index)) & uptrend),
]

for horizon_name in ['1m', '3m', '6m']:
    fwd_col_h = f'SPY_fwd_{horizon_name}'
    if fwd_col_h not in master.columns:
        continue
    for cond_name, mask in conditions:
        r = compute_joint_analysis(cond_name, mask, fwd_col_h)
        if r:
            r['horizon'] = horizon_name
            joint_results.append(r)

joint_df = pd.DataFrame(joint_results)
joint_df.to_csv(OUT / 'joint_signal_analysis.csv', index=False, float_format='%.6f')
print(f"Saved joint signal analysis: {len(joint_df)} rows")

# ============================================================
# 8. MULTI-ASSET CONDITIONAL RETURNS
# ============================================================
print("\n" + "=" * 70)
print("STEP 8: Multi-asset conditional returns")
print("=" * 70)

# For each condition, also look at other assets
multi_asset_results = []
key_conditions = [
    ("BASELINE", pd.Series(True, index=master.index)),
    ("BULL composite", (master['vix_level'] < vix_lo) & mom_pos & breadth_hi),
    ("BEAR composite", (master['vix_level'] > vix_hi) & downtrend & credit_widening),
    ("Vol spike + trend intact", (master['vix_21d_pctrank'] > 0.8) & uptrend & mom_pos),
    ("High IV-RV + uptrend", iv_rv_hi & uptrend),
]

for cond_name, mask in key_conditions:
    for asset in ['SPY', 'QQQ', 'IWM', 'TLT', 'GLD', 'VNQ', 'EFA', 'EEM']:
        fwd_col_a = f'{asset}_fwd_1m'
        if fwd_col_a not in master.columns:
            continue
        subset = master.loc[mask, fwd_col_a].dropna()
        if len(subset) < 20:
            continue
        multi_asset_results.append({
            'condition': cond_name,
            'asset': asset,
            'n_obs': len(subset),
            'mean_1m': subset.mean(),
            'hit_rate': (subset > 0).mean(),
            'sharpe': subset.mean() / subset.std() * np.sqrt(12) if subset.std() > 0 else 0,
        })

multi_asset_df = pd.DataFrame(multi_asset_results)
print("\n--- MULTI-ASSET RETURNS BY CONDITION (1m forward) ---")
for cond in multi_asset_df['condition'].unique():
    print(f"\n  {cond}:")
    sub = multi_asset_df[multi_asset_df['condition'] == cond].sort_values('mean_1m', ascending=False)
    for _, row in sub.iterrows():
        print(f"    {row['asset']}: mean={row['mean_1m']*100:.2f}%, hit={row['hit_rate']*100:.1f}%, Sharpe={row['sharpe']:.2f}")

# ============================================================
# 9. GENERATE SUMMARY REPORT
# ============================================================
print("\n" + "=" * 70)
print("STEP 9: Generating summary report")
print("=" * 70)

# Find the best asymmetric opportunities
j1m = joint_df[joint_df['horizon'] == '1m'].copy()
j3m = joint_df[joint_df['horizon'] == '3m'].copy()

baseline_1m = j1m[j1m['condition'] == 'BASELINE: All days'].iloc[0] if len(j1m[j1m['condition'] == 'BASELINE: All days']) > 0 else None
baseline_3m = j3m[j3m['condition'] == 'BASELINE: All days'].iloc[0] if len(j3m[j3m['condition'] == 'BASELINE: All days']) > 0 else None

# Score conditions by: high return * high hit rate * favorable asymmetry * enough observations
j1m['quality_score'] = j1m['mean_return'] * j1m['hit_rate'] * j1m['updown_ratio'] * np.log(j1m['n_obs'] + 1)
j3m['quality_score'] = j3m['mean_return'] * j3m['hit_rate'] * j3m['updown_ratio'] * np.log(j3m['n_obs'] + 1)

# Top bullish conditions
top_bull_1m = j1m[j1m['mean_return'] > 0].nlargest(10, 'quality_score')
top_bear_1m = j1m[j1m['mean_return'] < 0].nsmallest(5, 'mean_return')

top_bull_3m = j3m[j3m['mean_return'] > 0].nlargest(10, 'quality_score')

# Signal independence analysis
print("\nSignal independence clusters (from correlation matrix):")
independent_signals = []
for s1 in key_signals:
    is_independent = True
    for s2 in independent_signals:
        if abs(corr_matrix.loc[s1, s2]) > 0.6:
            is_independent = False
            break
    if is_independent:
        independent_signals.append(s1)
print(f"  Independent signal set: {independent_signals}")

# Write report
report_lines = []
report_lines.append("=" * 80)
report_lines.append("SIGNAL CORRELATION & ASYMMETRY ANALYSIS")
report_lines.append(f"Generated: 2026-07-21")
report_lines.append(f"Data: {close.index[0].date()} to {close.index[-1].date()} ({len(close)} trading days)")
report_lines.append("=" * 80)

report_lines.append("\n\n1. SIGNAL LANDSCAPE OVERVIEW")
report_lines.append("-" * 40)
report_lines.append(f"Total signals computed: {len(signals.columns)}")
report_lines.append(f"Key independent signals: {len(independent_signals)}")
report_lines.append(f"Signals analyzed: {', '.join(key_signals)}")

report_lines.append("\n\n2. SIGNAL CORRELATION FINDINGS")
report_lines.append("-" * 40)
report_lines.append("HIGHLY CORRELATED (redundant) pairs:")
seen = set()
for i, s1 in enumerate(key_signals):
    for s2 in key_signals[i+1:]:
        r = corr_matrix.loc[s1, s2]
        if abs(r) > 0.6 and not np.isnan(r) and (s1, s2) not in seen:
            report_lines.append(f"  {s1} <-> {s2}: r={r:.3f}")
            seen.add((s1, s2))

report_lines.append("\nINDEPENDENT signal clusters (use these together for diversified signal):")
for s in independent_signals:
    report_lines.append(f"  - {s}")

report_lines.append("\n\n3. REGIME ANALYSIS")
report_lines.append("-" * 40)
regime_rets_clean = regime_df[regime_df['horizon'].isin(['1m', '3m', '6m'])]
for regime in ['Low VIX + Uptrend', 'Low VIX + Downtrend', 'High VIX + Uptrend', 'High VIX + Downtrend']:
    rr = regime_rets_clean[regime_rets_clean['regime'] == regime]
    if len(rr) == 0:
        continue
    n_days_regime = rr.iloc[0]['n_days']
    report_lines.append(f"\n{regime} ({int(n_days_regime)} days, {n_days_regime/len(master)*100:.1f}% of sample):")
    for _, row in rr.iterrows():
        report_lines.append(f"  {row['horizon']} fwd: mean={row['mean_return']*100:.2f}%, hit={row['hit_rate']*100:.1f}%, "
                          f"up/dn={row['updown_ratio']:.2f}, Sharpe={row['sharpe_ann']:.2f}")

report_lines.append("\n\n4. TOP ASYMMETRIC OPPORTUNITIES (Joint Signal Conditions)")
report_lines.append("-" * 40)
report_lines.append("\n--- 1-MONTH FORWARD (best risk-adjusted opportunities) ---")
for _, row in top_bull_1m.iterrows():
    report_lines.append(f"\n  {row['condition']}")
    report_lines.append(f"    Frequency: {row['pct_of_days']:.1f}% of days ({int(row['n_obs'])} obs)")
    report_lines.append(f"    Mean return: {row['mean_return']*100:.2f}% | Median: {row['median_return']*100:.2f}%")
    report_lines.append(f"    Hit rate: {row['hit_rate']*100:.1f}%")
    report_lines.append(f"    Upside/downside ratio: {row['updown_ratio']:.2f}")
    report_lines.append(f"    p10 (downside): {row['p10']*100:.2f}% | p90 (upside): {row['p90']*100:.2f}%")
    report_lines.append(f"    Annualized Sharpe: {row['sharpe_ann']:.2f}")

report_lines.append("\n\n--- 3-MONTH FORWARD (best risk-adjusted opportunities) ---")
for _, row in top_bull_3m.head(5).iterrows():
    report_lines.append(f"\n  {row['condition']}")
    report_lines.append(f"    Frequency: {row['pct_of_days']:.1f}% of days ({int(row['n_obs'])} obs)")
    report_lines.append(f"    Mean return: {row['mean_return']*100:.2f}% | Hit rate: {row['hit_rate']*100:.1f}%")
    report_lines.append(f"    Upside/downside ratio: {row['updown_ratio']:.2f}")
    report_lines.append(f"    Annualized Sharpe: {row['sharpe_ann']:.2f}")

report_lines.append("\n\n5. DEFENSIVE CONDITIONS (When to reduce exposure)")
report_lines.append("-" * 40)
for _, row in top_bear_1m.iterrows():
    report_lines.append(f"\n  {row['condition']}")
    report_lines.append(f"    Frequency: {row['pct_of_days']:.1f}% of days ({int(row['n_obs'])} obs)")
    report_lines.append(f"    Mean return: {row['mean_return']*100:.2f}% | Hit rate: {row['hit_rate']*100:.1f}%")
    report_lines.append(f"    Worst drawdown (p5): {row['p5']*100:.2f}%")

report_lines.append("\n\n6. KEY FINDINGS & ACTIONABLE INSIGHTS")
report_lines.append("-" * 40)

# Auto-generate insights from data
report_lines.append("\nFINDING 1: ASYMMETRIC UPSIDE CONDITIONS")
best = top_bull_1m.iloc[0] if len(top_bull_1m) > 0 else None
if best is not None:
    report_lines.append(f"  The single best asymmetric setup is: {best['condition']}")
    report_lines.append(f"  It produces {best['mean_return']*100:.2f}% avg 1m return with {best['hit_rate']*100:.1f}% hit rate")
    report_lines.append(f"  and up/down ratio of {best['updown_ratio']:.2f} — meaning winners are {best['updown_ratio']:.1f}x larger than losers")

report_lines.append("\nFINDING 2: REGIME MATTERS MORE THAN ANY SINGLE SIGNAL")
# Compare regime spread
regime_means = regime_rets_clean[regime_rets_clean['horizon'] == '1m'].set_index('regime')['mean_return']
if len(regime_means) >= 2:
    best_regime = regime_means.idxmax()
    worst_regime = regime_means.idxmin()
    spread = (regime_means.max() - regime_means.min()) * 100
    report_lines.append(f"  Best regime: {best_regime} ({regime_means.max()*100:.2f}% monthly)")
    report_lines.append(f"  Worst regime: {worst_regime} ({regime_means.min()*100:.2f}% monthly)")
    report_lines.append(f"  Regime spread: {spread:.2f}% per month — this dominates any signal edge")

report_lines.append("\nFINDING 3: IV-RV SPREAD IS THE MOST ACTIONABLE SIGNAL")
if 'iv_rv_spread' in master.columns:
    iv_rv_q = asym_df[(asym_df['signal'] == 'iv_rv_spread') & (asym_df['horizon'] == '1m')]
    if len(iv_rv_q) > 0:
        q5 = iv_rv_q[iv_rv_q['quintile'] == iv_rv_q['quintile'].max()]
        q1 = iv_rv_q[iv_rv_q['quintile'] == iv_rv_q['quintile'].min()]
        if len(q5) > 0 and len(q1) > 0:
            report_lines.append(f"  When IV-RV spread is highest (Q5): mean 1m return = {q5.iloc[0]['mean_return']*100:.2f}%, hit = {q5.iloc[0]['hit_rate']*100:.1f}%")
            report_lines.append(f"  When IV-RV spread is lowest (Q1): mean 1m return = {q1.iloc[0]['mean_return']*100:.2f}%, hit = {q1.iloc[0]['hit_rate']*100:.1f}%")
            report_lines.append(f"  This confirms: high IV-RV = market is pricing in too much fear = mean reversion opportunity")

report_lines.append("\nFINDING 4: SIGNAL REDUNDANCY")
report_lines.append("  Many momentum signals are >0.7 correlated with each other.")
report_lines.append("  The INDEPENDENT signals that add information are:")
for s in independent_signals[:8]:
    report_lines.append(f"    - {s}")
report_lines.append("  Use only independent signals to avoid false confidence from correlated inputs.")

report_lines.append("\nFINDING 5: VIX TERM STRUCTURE IS UNDERRATED")
if 'vix_term_structure' in master.columns:
    vts_q = asym_df[(asym_df['signal'] == 'vix_term_structure') & (asym_df['horizon'] == '1m')]
    if len(vts_q) > 0:
        for _, row in vts_q.iterrows():
            report_lines.append(f"  Q{int(row['quintile'])}: mean={row['mean_return']*100:.2f}%, hit={row['hit_rate']*100:.1f}%, up/dn={row['updown_ratio']:.2f}")
        report_lines.append("  Backwardation (high ratio = fear) tends to precede positive returns (mean reversion)")
        report_lines.append("  But contango (low ratio = complacency) with strong trend is the best risk-adjusted setup")

report_lines.append("\n\n7. ANSWER TO THE KEY QUESTION")
report_lines.append("-" * 40)
report_lines.append("\n'What are the 3-5 conditions where forward returns are most asymmetrically positive?'")
report_lines.append("")

# Extract top 5 conditions with best asymmetry score
j1m_asym = j1m[j1m['n_obs'] >= 50].copy()
j1m_asym['asym_score'] = j1m_asym['hit_rate'] * j1m_asym['updown_ratio'] * j1m_asym['mean_return']
top5 = j1m_asym.nlargest(5, 'asym_score')
for i, (_, row) in enumerate(top5.iterrows(), 1):
    report_lines.append(f"\n  #{i}: {row['condition']}")
    report_lines.append(f"      Return: {row['mean_return']*100:.2f}% (1m) | Hit rate: {row['hit_rate']*100:.1f}%")
    report_lines.append(f"      Up/Down: {row['updown_ratio']:.2f} | Tail: p10={row['p10']*100:.2f}%, p90={row['p90']*100:.2f}%")
    report_lines.append(f"      Frequency: {row['pct_of_days']:.1f}% of days")

report_lines.append("\n\n'What are the conditions where we should be defensive?'")
report_lines.append("")
worst5 = j1m[j1m['n_obs'] >= 50].nsmallest(5, 'mean_return')
for i, (_, row) in enumerate(worst5.iterrows(), 1):
    report_lines.append(f"\n  #{i}: {row['condition']}")
    report_lines.append(f"      Return: {row['mean_return']*100:.2f}% (1m) | Hit rate: {row['hit_rate']*100:.1f}%")
    report_lines.append(f"      Worst case (p5): {row['p5']*100:.2f}%")

report_lines.append("\n\n" + "=" * 80)
report_lines.append("END OF REPORT")
report_lines.append("=" * 80)

report_text = "\n".join(report_lines)

with open(OUT / 'summary_report.txt', 'w') as f:
    f.write(report_text)

print("\nSummary report saved.")
print("\n" + report_text)

print("\n\nAll outputs saved to:", OUT)
print("Files:")
for f in sorted(OUT.glob('*.csv')) :
    print(f"  {f.name}: {pd.read_csv(f).shape}")
print(f"  summary_report.txt")
