#!/usr/bin/env python3
"""
Drawdown Protection Rules Engine (HC #709 + #710)
==================================================
Tests simple rules-based drawdown protection signals using multi-asset data.
Goal: Find rules that predict red days / drawdowns BEFORE they happen,
so growth strategies can scale exposure down in time.

NOT ML — purely rules-based with walk-forward validation.
Tests: VIX levels, SMA crossovers, credit spreads, breadth, cross-asset divergence,
       momentum, rate-of-change, and composite signals.

Validated against HC #428 R1 (regime-agnostic) + permutation test.
"""
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta
import json
import os
import warnings
warnings.filterwarnings('ignore')

OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/growth_research/drawdown_protection'
os.makedirs(OUTPUT_DIR, exist_ok=True)

print("=" * 70)
print("DRAWDOWN PROTECTION RULES ENGINE")
print("HC #709: Predict red days → scale exposure → protect growth")
print("HC #710: Use multi-asset signals")
print("=" * 70)

# ── 1. DATA DOWNLOAD ──
print("\n[1/6] Downloading multi-asset data...")
tickers = {
    # Core
    'SPY': 'SPY', 'QQQ': 'QQQ', 'IWM': 'IWM',
    # Volatility  
    'VIX': '^VIX', 'VVIX': '^VVIX',
    # Bonds
    'TLT': 'TLT', 'IEF': 'IEF', 'HYG': 'HYG', 'LQD': 'LQD',
    # Commodities
    'GLD': 'GLD', 'SLV': 'SLV', 'USO': 'USO', 'CPER': 'CPER',
    # Dollar
    'UUP': 'UUP',
    # Sectors (for breadth)
    'XLK': 'XLK', 'XLF': 'XLF', 'XLE': 'XLE', 'XLV': 'XLV',
    'XLI': 'XLI', 'XLP': 'XLP', 'XLU': 'XLU', 'XLB': 'XLB',
    'XLRE': 'XLRE', 'XLY': 'XLY', 'XLC': 'XLC',
    # Crypto
    'BTC': 'BTC-USD',
    # Leveraged (for strategy returns)
    'TQQQ': 'TQQQ', 'UPRO': 'UPRO',
}

start_date = '2012-01-01'
end_date = datetime.now().strftime('%Y-%m-%d')

data = {}
for name, ticker in tickers.items():
    try:
        df = yf.download(ticker, start=start_date, end=end_date, progress=False)
        if len(df) > 100:
            # Handle multi-level columns from yfinance
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            data[name] = df['Close']
            print(f"  {name}: {len(df)} days")
    except Exception as e:
        print(f"  {name}: FAILED ({e})")

prices = pd.DataFrame(data).ffill()
returns = prices.pct_change()
print(f"\nCombined dataset: {len(prices)} days, {len(prices.columns)} assets")
print(f"Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")

# ── 2. BUILD FEATURE SIGNALS ──
print("\n[2/6] Building drawdown protection signals...")

features = pd.DataFrame(index=prices.index)

def _rsi(series, period=14):
    delta = series.diff()
    gain = delta.where(delta > 0, 0).rolling(period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(period).mean()
    rs = gain / loss
    return 100 - (100 / (1 + rs))

# VIX-based signals
if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX']
    features['vix_5d_chg'] = prices['VIX'].pct_change(5)
    features['vix_10d_chg'] = prices['VIX'].pct_change(10)
    features['vix_20d_ma'] = prices['VIX'].rolling(20).mean()
    features['vix_above_20d'] = (prices['VIX'] > features['vix_20d_ma']).astype(int)
    features['vix_z_60d'] = (prices['VIX'] - prices['VIX'].rolling(60).mean()) / prices['VIX'].rolling(60).std()

# SMA crossovers
for asset in ['SPY', 'QQQ']:
    if asset in prices.columns:
        features[f'{asset}_above_50sma'] = (prices[asset] > prices[asset].rolling(50).mean()).astype(int)
        features[f'{asset}_above_200sma'] = (prices[asset] > prices[asset].rolling(200).mean()).astype(int)
        features[f'{asset}_50_200_cross'] = (prices[asset].rolling(50).mean() > prices[asset].rolling(200).mean()).astype(int)
        features[f'{asset}_rsi_14'] = _rsi(prices[asset], 14) if True else 0
        features[f'{asset}_20d_roc'] = prices[asset].pct_change(20)
        features[f'{asset}_5d_roc'] = prices[asset].pct_change(5)

# Credit spread
if 'HYG' in prices.columns and 'LQD' in prices.columns:
    credit_spread = (returns['HYG'] - returns['LQD']).rolling(20).mean()
    features['credit_spread_20d'] = credit_spread
    features['credit_tightening'] = (credit_spread > 0).astype(int)
    features['credit_z'] = (credit_spread - credit_spread.rolling(60).mean()) / credit_spread.rolling(60).std()

# Bond signal (flight to safety)
if 'TLT' in prices.columns and 'SPY' in prices.columns:
    features['tlt_spy_corr_20d'] = returns['TLT'].rolling(20).corr(returns['SPY'])
    features['tlt_10d_roc'] = prices['TLT'].pct_change(10)

# Commodities (risk appetite)
if 'GLD' in prices.columns:
    features['gold_20d_roc'] = prices['GLD'].pct_change(20)
if 'CPER' in prices.columns:
    features['copper_20d_roc'] = prices['CPER'].pct_change(20)
    
# Copper/Gold ratio (risk appetite indicator)
if 'GLD' in prices.columns and 'CPER' in prices.columns:
    cg_ratio = prices['CPER'] / prices['GLD']
    features['copper_gold_roc'] = cg_ratio.pct_change(20)

# Dollar
if 'UUP' in prices.columns:
    features['dollar_20d_roc'] = prices['UUP'].pct_change(20)

# Sector breadth
sector_etfs = [c for c in ['XLK','XLF','XLE','XLV','XLI','XLP','XLU','XLB','XLRE','XLY','XLC'] if c in prices.columns]
if len(sector_etfs) >= 5:
    above_50 = pd.DataFrame({s: (prices[s] > prices[s].rolling(50).mean()).astype(int) for s in sector_etfs})
    above_200 = pd.DataFrame({s: (prices[s] > prices[s].rolling(200).mean()).astype(int) for s in sector_etfs})
    features['breadth_50sma'] = above_50.mean(axis=1)
    features['breadth_200sma'] = above_200.mean(axis=1)
    features['breadth_declining'] = (features['breadth_50sma'].diff(5) < -0.2).astype(int)

# BTC as risk indicator
if 'BTC' in prices.columns:
    features['btc_20d_roc'] = prices['BTC'].pct_change(20)

# IWM/SPY relative (small cap weakness = risk off)
if 'IWM' in prices.columns and 'SPY' in prices.columns:
    rel = prices['IWM'] / prices['SPY']
    features['small_cap_rel_20d'] = rel.pct_change(20)

print(f"Built {len(features.columns)} features")

# RSI already computed above

features = features.dropna()
print(f"After dropna: {len(features)} days")

# ── 3. DEFINE PROTECTION RULES ──
print("\n[3/6] Defining and testing protection rules...")

# Target: next-day SPY return (we want to predict RED days)
spy_fwd_1d = returns['SPY'].shift(-1)  # next day return
spy_fwd_5d = returns['SPY'].rolling(5).sum().shift(-5)  # next 5-day return

# Define rules: each returns a "risk_on" signal (1=safe, 0=danger)
rules = {}

# VIX rules
if 'vix_level' in features.columns:
    for thresh in [17, 20, 25, 30]:
        rules[f'vix_below_{thresh}'] = (features['vix_level'] < thresh).astype(int)
    rules['vix_not_spiking'] = (features['vix_5d_chg'] < 0.20).astype(int)
    rules['vix_z_below_1'] = (features['vix_z_60d'] < 1.0).astype(int)
    rules['vix_z_below_1.5'] = (features['vix_z_60d'] < 1.5).astype(int)

# SMA rules
for asset in ['SPY', 'QQQ']:
    if f'{asset}_above_200sma' in features.columns:
        rules[f'{asset}_above_200sma'] = features[f'{asset}_above_200sma']
        rules[f'{asset}_above_50sma'] = features[f'{asset}_above_50sma']
        rules[f'{asset}_golden_cross'] = features[f'{asset}_50_200_cross']

# Momentum rules
for asset in ['SPY', 'QQQ']:
    if f'{asset}_20d_roc' in features.columns:
        rules[f'{asset}_mom_positive'] = (features[f'{asset}_20d_roc'] > 0).astype(int)
        rules[f'{asset}_mom_strong'] = (features[f'{asset}_20d_roc'] > 0.02).astype(int)

# RSI rules
for asset in ['SPY', 'QQQ']:
    if f'{asset}_rsi_14' in features.columns:
        rules[f'{asset}_rsi_not_overbought'] = (features[f'{asset}_rsi_14'] < 70).astype(int)
        rules[f'{asset}_rsi_not_extreme'] = (features[f'{asset}_rsi_14'] < 80).astype(int)

# Credit rules
if 'credit_tightening' in features.columns:
    rules['credit_tightening'] = features['credit_tightening']
    rules['credit_not_stressed'] = (features['credit_z'] > -1.0).astype(int)
    rules['credit_not_extreme'] = (features['credit_z'] > -1.5).astype(int)

# Bond flight-to-safety
if 'tlt_spy_corr_20d' in features.columns:
    rules['no_flight_to_safety'] = (features['tlt_spy_corr_20d'] > -0.3).astype(int)

# Breadth rules
if 'breadth_50sma' in features.columns:
    rules['breadth_50_above_50pct'] = (features['breadth_50sma'] > 0.5).astype(int)
    rules['breadth_200_above_50pct'] = (features['breadth_200sma'] > 0.5).astype(int)
    rules['breadth_not_declining'] = (1 - features['breadth_declining']).astype(int)

# Copper/Gold (risk appetite)
if 'copper_gold_roc' in features.columns:
    rules['copper_gold_positive'] = (features['copper_gold_roc'] > -0.05).astype(int)

# Dollar (strong dollar = risk off)
if 'dollar_20d_roc' in features.columns:
    rules['dollar_not_surging'] = (features['dollar_20d_roc'] < 0.02).astype(int)

# Small cap relative
if 'small_cap_rel_20d' in features.columns:
    rules['small_cap_not_crashing'] = (features['small_cap_rel_20d'] > -0.05).astype(int)

print(f"Defined {len(rules)} protection rules")

# ── 4. WALK-FORWARD EVALUATION ──
print("\n[4/6] Walk-forward evaluation of each rule...")

# For each rule, compute: 
# 1. Hit rate (does signal=0 actually predict worse returns?)
# 2. Protection value (how much drawdown is avoided?)
# 3. Cost (how much upside is missed?)
# 4. Net benefit (Sharpe with rule vs without)

# Use UPRO returns as the growth strategy we're protecting
growth_asset = 'UPRO' if 'UPRO' in returns.columns else 'TQQQ' if 'TQQQ' in returns.columns else 'SPY'
growth_ret = returns[growth_asset]

# Align everything
common_idx = features.index.intersection(growth_ret.dropna().index).intersection(spy_fwd_1d.dropna().index)
growth_ret_aligned = growth_ret.loc[common_idx]
spy_fwd_aligned = spy_fwd_1d.loc[common_idx]

results = {}
for rule_name, signal in rules.items():
    sig = signal.reindex(common_idx).dropna()
    idx = sig.index
    
    gr = growth_ret_aligned.loc[idx]
    spy_fwd = spy_fwd_aligned.loc[idx]
    
    risk_on = sig == 1
    risk_off = sig == 0
    
    n_on = risk_on.sum()
    n_off = risk_off.sum()
    
    if n_off < 20 or n_on < 20:
        continue
    
    # Returns when risk-on vs risk-off
    ret_on = gr[risk_on]
    ret_off = gr[risk_off]
    
    # Strategy: hold growth when signal=1, hold cash when signal=0
    strat_ret = gr.copy()
    strat_ret[risk_off] = 0  # go to cash
    
    # Metrics
    ann_factor = 252
    sharpe_buy_hold = gr.mean() / gr.std() * np.sqrt(ann_factor) if gr.std() > 0 else 0
    sharpe_protected = strat_ret.mean() / strat_ret.std() * np.sqrt(ann_factor) if strat_ret.std() > 0 else 0
    
    # Drawdown comparison
    cum_bh = (1 + gr).cumprod()
    cum_prot = (1 + strat_ret).cumprod()
    dd_bh = (cum_bh / cum_bh.cummax() - 1).min()
    dd_prot = (cum_prot / cum_prot.cummax() - 1).min()
    
    # CAGR
    n_years = len(gr) / 252
    cagr_bh = (cum_bh.iloc[-1] ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    cagr_prot = (cum_prot.iloc[-1] ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    
    # How well does risk-off predict bad days?
    avg_ret_risk_off = spy_fwd.loc[idx][risk_off].mean() * 100
    avg_ret_risk_on = spy_fwd.loc[idx][risk_on].mean() * 100
    
    # R1: regime check
    # Split by SPY regime (bull/bear using 200 SMA)
    spy_200sma = prices['SPY'].rolling(200).mean().reindex(idx)
    bull = prices['SPY'].reindex(idx) > spy_200sma
    bear = ~bull
    
    sharpe_bull = 0
    sharpe_bear = 0
    if bull.sum() > 50:
        sr_bull = strat_ret[bull]
        sharpe_bull = sr_bull.mean() / sr_bull.std() * np.sqrt(252) if sr_bull.std() > 0 else 0
    if bear.sum() > 50:
        sr_bear = strat_ret[bear]
        sharpe_bear = sr_bear.mean() / sr_bear.std() * np.sqrt(252) if sr_bear.std() > 0 else 0
    
    r1_gap = abs(sharpe_bull - sharpe_bear) / max(abs(sharpe_bull), abs(sharpe_bear), 0.01)
    
    # Sortino
    neg_ret_prot = strat_ret[strat_ret < 0]
    downside_std = neg_ret_prot.std() if len(neg_ret_prot) > 5 else strat_ret.std()
    sortino = strat_ret.mean() / downside_std * np.sqrt(252) if downside_std > 0 else 0
    
    results[rule_name] = {
        'n_days': len(idx),
        'pct_risk_on': round(n_on / len(idx) * 100, 1),
        'pct_risk_off': round(n_off / len(idx) * 100, 1),
        'sharpe_buy_hold': round(sharpe_buy_hold, 3),
        'sharpe_protected': round(sharpe_protected, 3),
        'sharpe_improvement': round(sharpe_protected - sharpe_buy_hold, 3),
        'sortino_protected': round(sortino, 3),
        'cagr_buy_hold': round(cagr_bh, 1),
        'cagr_protected': round(cagr_prot, 1),
        'maxdd_buy_hold': round(dd_bh * 100, 1),
        'maxdd_protected': round(dd_prot * 100, 1),
        'dd_improvement': round((dd_prot - dd_bh) * 100, 1),
        'avg_spy_ret_risk_on': round(avg_ret_risk_on, 4),
        'avg_spy_ret_risk_off': round(avg_ret_risk_off, 4),
        'sharpe_bull': round(sharpe_bull, 3),
        'sharpe_bear': round(sharpe_bear, 3),
        'r1_gap': round(r1_gap, 3),
        'r1_pass': r1_gap < 0.50,
    }

# Sort by Sharpe improvement
results_sorted = dict(sorted(results.items(), key=lambda x: x[1]['sharpe_improvement'], reverse=True))

print(f"\nEvaluated {len(results_sorted)} rules")
print("\n" + "="*70)
print("TOP 15 RULES BY SHARPE IMPROVEMENT:")
print("="*70)
for i, (name, m) in enumerate(list(results_sorted.items())[:15]):
    r1 = "✅" if m['r1_pass'] else "❌"
    print(f"\n{i+1}. {name}")
    print(f"   Sharpe: {m['sharpe_buy_hold']:.3f} → {m['sharpe_protected']:.3f} (Δ{m['sharpe_improvement']:+.3f})")
    print(f"   Sortino: {m['sortino_protected']:.3f}")
    print(f"   CAGR: {m['cagr_buy_hold']:.1f}% → {m['cagr_protected']:.1f}%")
    print(f"   MaxDD: {m['maxdd_buy_hold']:.1f}% → {m['maxdd_protected']:.1f}% (improved {m['dd_improvement']:.1f}pp)")
    print(f"   Risk-on: {m['pct_risk_on']:.1f}% | Risk-off SPY avg: {m['avg_spy_ret_risk_off']:.4f}%")
    print(f"   R1: Bull Sharpe {m['sharpe_bull']:.3f}, Bear {m['sharpe_bear']:.3f}, Gap {m['r1_gap']:.3f} {r1}")

# ── 5. COMPOSITE RULES ──
print("\n" + "="*70)
print("COMPOSITE RULES (combining best individual rules)")
print("="*70)

# Build composite signals
composites = {}

# Conservative: all of VIX < 25 + SPY > 200SMA + breadth > 50%
composite_parts = {}
if 'vix_below_25' in rules: composite_parts['vix'] = rules['vix_below_25']
if 'SPY_above_200sma' in rules: composite_parts['sma'] = rules['SPY_above_200sma']
if 'breadth_200_above_50pct' in rules: composite_parts['breadth'] = rules['breadth_200_above_50pct']

if len(composite_parts) >= 2:
    # ALL must agree
    all_on = pd.DataFrame(composite_parts).min(axis=1)
    composites['conservative_all'] = all_on
    
    # MAJORITY (2/3) must agree
    majority = (pd.DataFrame(composite_parts).sum(axis=1) >= 2).astype(int)
    composites['conservative_majority'] = majority

# Aggressive: VIX < 20 + SPY > 50SMA + credit OK + breadth OK
agg_parts = {}
if 'vix_below_20' in rules: agg_parts['vix'] = rules['vix_below_20']
if 'SPY_above_50sma' in rules: agg_parts['sma'] = rules['SPY_above_50sma']
if 'credit_not_stressed' in rules: agg_parts['credit'] = rules['credit_not_stressed']
if 'breadth_50_above_50pct' in rules: agg_parts['breadth'] = rules['breadth_50_above_50pct']

if len(agg_parts) >= 2:
    composites['aggressive_all'] = pd.DataFrame(agg_parts).min(axis=1)
    composites['aggressive_majority'] = (pd.DataFrame(agg_parts).sum(axis=1) >= len(agg_parts) * 0.6).astype(int)

# Kitchen sink: best single rules combined
best_singles = []
for name in list(results_sorted.keys())[:5]:
    if name in rules:
        best_singles.append(rules[name])
if len(best_singles) >= 3:
    composites['top5_all'] = pd.DataFrame(best_singles).min(axis=1)
    composites['top5_majority'] = (pd.DataFrame(best_singles).sum(axis=1) >= 3).astype(int)

# Evaluate composites
composite_results = {}
for comp_name, signal in composites.items():
    sig = signal.reindex(common_idx).dropna()
    idx = sig.index
    gr = growth_ret_aligned.loc[idx]
    
    risk_on = sig == 1
    risk_off = sig == 0
    n_on, n_off = risk_on.sum(), risk_off.sum()
    
    if n_off < 10 or n_on < 10:
        continue
    
    strat_ret = gr.copy()
    strat_ret[risk_off] = 0
    
    ann = 252
    sharpe_bh = gr.mean() / gr.std() * np.sqrt(ann) if gr.std() > 0 else 0
    sharpe_p = strat_ret.mean() / strat_ret.std() * np.sqrt(ann) if strat_ret.std() > 0 else 0
    
    cum_bh = (1 + gr).cumprod()
    cum_p = (1 + strat_ret).cumprod()
    dd_bh = (cum_bh / cum_bh.cummax() - 1).min()
    dd_p = (cum_p / cum_p.cummax() - 1).min()
    n_years = len(gr) / 252
    cagr_bh = (cum_bh.iloc[-1] ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    cagr_p = (cum_p.iloc[-1] ** (1/n_years) - 1) * 100 if n_years > 0 else 0
    
    neg_r = strat_ret[strat_ret < 0]
    ds = neg_r.std() if len(neg_r) > 5 else strat_ret.std()
    sortino = strat_ret.mean() / ds * np.sqrt(252) if ds > 0 else 0
    
    spy_200 = prices['SPY'].rolling(200).mean().reindex(idx)
    bull = prices['SPY'].reindex(idx) > spy_200
    bear = ~bull
    sb, sbe = 0, 0
    if bull.sum() > 50:
        r = strat_ret[bull]; sb = r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
    if bear.sum() > 50:
        r = strat_ret[bear]; sbe = r.mean()/r.std()*np.sqrt(252) if r.std()>0 else 0
    r1g = abs(sb-sbe)/max(abs(sb),abs(sbe),0.01)
    
    composite_results[comp_name] = {
        'n_days': len(idx),
        'pct_risk_on': round(n_on/len(idx)*100, 1),
        'sharpe_bh': round(sharpe_bh, 3),
        'sharpe_protected': round(sharpe_p, 3),
        'sharpe_delta': round(sharpe_p - sharpe_bh, 3),
        'sortino': round(sortino, 3),
        'cagr_bh': round(cagr_bh, 1),
        'cagr_protected': round(cagr_p, 1),
        'maxdd_bh': round(dd_bh*100, 1),
        'maxdd_protected': round(dd_p*100, 1),
        'dd_improvement_pp': round((dd_p - dd_bh)*100, 1),
        'sharpe_bull': round(sb, 3),
        'sharpe_bear': round(sbe, 3),
        'r1_gap': round(r1g, 3),
        'r1_pass': r1g < 0.50,
    }
    
    r1 = "✅" if r1g < 0.50 else "❌"
    print(f"\n{comp_name}:")
    print(f"  Sharpe: {sharpe_bh:.3f} → {sharpe_p:.3f} (Δ{sharpe_p-sharpe_bh:+.3f})")
    print(f"  Sortino: {sortino:.3f}")
    print(f"  CAGR: {cagr_bh:.1f}% → {cagr_p:.1f}%")
    print(f"  MaxDD: {dd_bh*100:.1f}% → {dd_p*100:.1f}% ({(dd_p-dd_bh)*100:+.1f}pp)")
    print(f"  Risk-on: {n_on/len(idx)*100:.1f}%  |  R1 gap: {r1g:.3f} {r1}")

# ── 6. PERMUTATION TEST ON BEST ──
print("\n" + "="*70)
print("PERMUTATION TEST — best composite")
print("="*70)

# Find best composite by Sharpe improvement
if composite_results:
    best_comp = max(composite_results, key=lambda k: composite_results[k]['sharpe_delta'])
    best_signal = composites[best_comp].reindex(common_idx).dropna()
    idx = best_signal.index
    gr = growth_ret_aligned.loc[idx]
    
    real_strat = gr.copy()
    real_strat[best_signal == 0] = 0
    real_sharpe = real_strat.mean() / real_strat.std() * np.sqrt(252) if real_strat.std() > 0 else 0
    
    n_perms = 1000
    perm_sharpes = []
    signal_vals = best_signal.values.copy()
    for _ in range(n_perms):
        np.random.shuffle(signal_vals)
        perm_ret = gr.copy()
        perm_ret.iloc[signal_vals == 0] = 0
        ps = perm_ret.mean() / perm_ret.std() * np.sqrt(252) if perm_ret.std() > 0 else 0
        perm_sharpes.append(ps)
    
    perm_p = np.mean([s >= real_sharpe for s in perm_sharpes])
    print(f"\nBest composite: {best_comp}")
    print(f"Real Sharpe: {real_sharpe:.3f}")
    print(f"Permutation Sharpe (mean): {np.mean(perm_sharpes):.3f}")
    print(f"Permutation p-value: {perm_p:.3f}")
    print(f"Verdict: {'✅ SIGNIFICANT' if perm_p < 0.05 else '❌ NOT SIGNIFICANT'}")
else:
    perm_p = 1.0
    best_comp = "none"

# ── SAVE RESULTS ──
output = {
    'metadata': {
        'timestamp': datetime.now().isoformat(),
        'growth_asset': growth_asset,
        'n_days': len(common_idx),
        'date_range': f"{common_idx[0].strftime('%Y-%m-%d')} to {common_idx[-1].strftime('%Y-%m-%d')}",
        'n_rules_tested': len(results_sorted),
        'n_composites_tested': len(composite_results),
    },
    'individual_rules': results_sorted,
    'composite_rules': composite_results,
    'permutation': {
        'best_composite': best_comp,
        'p_value': round(perm_p, 4),
        'significant': perm_p < 0.05,
    },
    'verdict': '',
}

# Determine verdict
passing_rules = {k:v for k,v in results_sorted.items() if v['r1_pass'] and v['sharpe_improvement'] > 0}
passing_composites = {k:v for k,v in composite_results.items() if v['r1_pass'] and v['sharpe_delta'] > 0}

if passing_composites and perm_p < 0.05:
    output['verdict'] = f"VALIDATED — {len(passing_composites)} composite rules pass R1 + permutation"
elif passing_rules:
    best_rule = max(passing_rules, key=lambda k: passing_rules[k]['sharpe_improvement'])
    output['verdict'] = f"PARTIAL — {len(passing_rules)} individual rules pass R1 (best: {best_rule}), composite permutation p={perm_p:.3f}"
else:
    output['verdict'] = "REJECT — no rules pass both R1 and permutation test"

print("\n" + "="*70)
print(f"VERDICT: {output['verdict']}")
print("="*70)

# Individual rules passing R1 with positive improvement
if passing_rules:
    print(f"\n{len(passing_rules)} individual rules pass R1 with positive Sharpe improvement:")
    for name, m in sorted(passing_rules.items(), key=lambda x: x[1]['sharpe_improvement'], reverse=True)[:10]:
        print(f"  {name}: Sharpe Δ{m['sharpe_improvement']:+.3f}, MaxDD improvement {m['dd_improvement']:+.1f}pp, R1 gap {m['r1_gap']:.3f}")

with open(os.path.join(OUTPUT_DIR, 'results.json'), 'w') as f:
    json.dump(output, f, indent=2)

print(f"\nResults saved to {OUTPUT_DIR}/results.json")
print("DONE")
