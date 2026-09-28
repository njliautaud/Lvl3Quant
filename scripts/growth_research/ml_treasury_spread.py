#!/usr/bin/env python3
"""
ML Treasury Maturity Spread
==============================
Strategy: ML predicts whether short-term (SHY 1-3yr) or long-term (TLT 20yr+)
treasuries will outperform over next 21 days, and rotates between them.

This is a WITHIN fixed-income relative value play. Both instruments have near-zero
equity correlation, so the strategy should be regime-agnostic.

Features: yield curve shape, VIX, credit, inflation expectations, dollar, momentum.
Walk-forward: 252d train, 21d advance, sliding window.
Adversarial: 4-gate validation.
"""

import numpy as np
import pandas as pd
import warnings
import json
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')
import yfinance as yf
import lightgbm as lgb

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_treasury_spread")
OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"{'='*60}")
print(f"ML TREASURY SPREAD — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}")

SHORT_BOND = 'SHY'
LONG_BOND = 'TLT'
MID_BOND = 'IEF'
MACRO = ['^VIX', 'UUP', 'SPY', 'GLD', 'HYG', 'LQD', '^TNX', 'TIP', 'DBC', 'EEM']

TRAIN_WINDOW = 252
ADVANCE_DAYS = 21
CAPITAL = 100000

print("Downloading data...")
start_date = '2008-01-01'
all_tickers = list(set([SHORT_BOND, LONG_BOND, MID_BOND] + MACRO))
data = {}
for t in all_tickers:
    try:
        df = yf.download(t, start=start_date, progress=False, auto_adjust=True)
        if len(df) > 50:
            if isinstance(df.columns, pd.MultiIndex):
                close = df[('Close', t)].copy()
            else:
                close = df['Close'].copy()
            clean = t.replace('^', '')
            close.name = clean
            data[clean] = close
    except:
        pass

prices = pd.DataFrame(data).dropna(how='all').ffill().dropna()
returns = prices.pct_change()
print(f"Data: {len(prices)} days, {prices.index[0].date()} to {prices.index[-1].date()}")

features = pd.DataFrame(index=prices.index)

# Duration spread
prices['dur_ratio'] = prices[LONG_BOND] / prices[SHORT_BOND]
features['dur_ratio'] = prices['dur_ratio']
features['dur_zscore_21d'] = (prices['dur_ratio'] - prices['dur_ratio'].rolling(21).mean()) / prices['dur_ratio'].rolling(21).std()
features['dur_zscore_63d'] = (prices['dur_ratio'] - prices['dur_ratio'].rolling(63).mean()) / prices['dur_ratio'].rolling(63).std()
features['dur_zscore_126d'] = (prices['dur_ratio'] - prices['dur_ratio'].rolling(126).mean()) / prices['dur_ratio'].rolling(126).std()
features['dur_mom_5d'] = prices['dur_ratio'].pct_change(5)
features['dur_mom_21d'] = prices['dur_ratio'].pct_change(21)
features['dur_mom_63d'] = prices['dur_ratio'].pct_change(63)

# Individual bond momentum
for b in [SHORT_BOND, LONG_BOND, MID_BOND]:
    if b in prices.columns:
        features[f'{b}_mom_5d'] = prices[b].pct_change(5)
        features[f'{b}_mom_21d'] = prices[b].pct_change(21)
        features[f'{b}_mom_63d'] = prices[b].pct_change(63)
        features[f'{b}_vol_21d'] = returns[b].rolling(21).std()

# Curve shape
if MID_BOND in prices.columns:
    features['curve_belly'] = prices[MID_BOND] / prices[SHORT_BOND]
    features['curve_long'] = prices[LONG_BOND] / prices[MID_BOND]
    features['butterfly'] = 2 * prices[MID_BOND] - prices[SHORT_BOND] - prices[LONG_BOND]

# Yields
if 'TNX' in prices.columns:
    features['tnx_level'] = prices['TNX']
    features['tnx_mom_21d'] = prices['TNX'].pct_change(21)
    features['tnx_mom_63d'] = prices['TNX'].pct_change(63)

# VIX
if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX']
    features['vix_zscore'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()

# Credit
if 'HYG' in prices.columns and 'LQD' in prices.columns:
    features['credit_spread'] = (returns['HYG'] - returns['LQD']).rolling(21).mean()
    features['credit_mom_21d'] = (prices['HYG'] / prices['LQD']).pct_change(21)

# TIPS (inflation)
if 'TIP' in prices.columns:
    features['tip_mom_21d'] = prices['TIP'].pct_change(21)
    features['bei_proxy'] = prices['TIP'] / prices[LONG_BOND]
    features['bei_mom_21d'] = features['bei_proxy'].pct_change(21)

# USD
if 'UUP' in prices.columns:
    features['usd_mom_21d'] = prices['UUP'].pct_change(21)

# SPY (risk appetite)
if 'SPY' in prices.columns:
    features['spy_mom_21d'] = prices['SPY'].pct_change(21)
    features['spy_vol_21d'] = returns['SPY'].rolling(21).std()

# Gold
if 'GLD' in prices.columns:
    features['gld_mom_21d'] = prices['GLD'].pct_change(21)

# EM
if 'EEM' in prices.columns:
    features['eem_mom_21d'] = prices['EEM'].pct_change(21)

# Commodities
if 'DBC' in prices.columns:
    features['dbc_mom_21d'] = prices['DBC'].pct_change(21)

features = features.dropna()
print(f"Features: {features.shape[1]} cols, {len(features)} rows")

# Walk-forward
print("\nRunning walk-forward backtest...")
returns['tlt_vs_shy'] = returns[LONG_BOND] - returns[SHORT_BOND]
fwd = returns['tlt_vs_shy'].rolling(ADVANCE_DAYS).sum().shift(-ADVANCE_DAYS)

all_dates, all_actuals, all_positions = [], [], []

for i in range(TRAIN_WINDOW + 252, len(features) - ADVANCE_DAYS, ADVANCE_DAYS):
    train_start = max(0, i - TRAIN_WINDOW)
    train_idx = features.index[train_start:i]
    test_idx = features.index[i:min(i + ADVANCE_DAYS, len(features))]

    train_labels = fwd.reindex(train_idx).dropna()
    common = train_idx.intersection(train_labels.index)
    if len(common) < 50: continue

    X_tr = np.nan_to_num(features.loc[common].values, nan=0, posinf=0, neginf=0)
    y_tr = (train_labels.loc[common] > 0).astype(int).values
    if len(np.unique(y_tr)) < 2: continue

    model = lgb.LGBMClassifier(n_estimators=100, max_depth=4, learning_rate=0.05,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8, verbose=-1, n_jobs=1)
    model.fit(X_tr, y_tr)

    X_test = np.nan_to_num(features.loc[test_idx].values, nan=0, posinf=0, neginf=0)
    probs = model.predict_proba(X_test)[:, 1]

    for dt, prob in zip(test_idx, probs):
        if dt not in returns.index: continue
        if prob > 0.6:
            daily_ret = returns.loc[dt, LONG_BOND]; pos = 'LONG_DURATION'
        elif prob < 0.4:
            daily_ret = returns.loc[dt, SHORT_BOND]; pos = 'SHORT_DURATION'
        else:
            daily_ret = 0.5 * returns.loc[dt, LONG_BOND] + 0.5 * returns.loc[dt, SHORT_BOND]; pos = 'NEUTRAL'
        all_dates.append(dt); all_actuals.append(daily_ret); all_positions.append(pos)

print(f"Walk-forward: {len(all_dates)} trading days")

equity = pd.Series(index=all_dates, data=np.cumprod(1 + np.array(all_actuals)) * CAPITAL)
daily_rets = pd.Series(index=all_dates, data=all_actuals)
positions = pd.Series(index=all_dates, data=all_positions)

ann_ret = daily_rets.mean() * 252
ann_vol = daily_rets.std() * np.sqrt(252)
sharpe = float(ann_ret / ann_vol) if ann_vol > 0 else 0
neg_vol = daily_rets[daily_rets < 0].std() * np.sqrt(252) if len(daily_rets[daily_rets < 0]) > 0 else 1
sortino = float(ann_ret / neg_vol)
max_dd = float((equity / equity.cummax() - 1).min())
calmar = float(ann_ret / abs(max_dd)) if max_dd != 0 else 0
years = len(daily_rets) / 252
cagr = float((equity.iloc[-1] / CAPITAL) ** (1/years) - 1) if years > 0 else 0
win_rate = float((daily_rets > 0).mean())
pf = float(abs(daily_rets[daily_rets > 0].sum() / daily_rets[daily_rets < 0].sum())) if daily_rets[daily_rets < 0].sum() != 0 else 0

spy_rets = returns['SPY'].reindex(daily_rets.index).fillna(0) if 'SPY' in returns.columns else daily_rets * 0
spy_corr = float(daily_rets.corr(spy_rets))

print(f"\nSTRATEGY METRICS")
print(f"CAGR: {cagr:.1%}, Sharpe: {sharpe:.3f}, Sortino: {sortino:.3f}")
print(f"MaxDD: {max_dd:.1%}, Calmar: {calmar:.2f}, PF: {pf:.2f}, WR: {win_rate:.1%}")
print(f"SPY Corr: {spy_corr:.3f}")
print(f"Positions: {positions.value_counts().to_dict()}")

# Adversarial
gates_passed = 0

perm_sharpes = [float(np.mean(daily_rets.sample(frac=1, replace=False).values) / daily_rets.std() * np.sqrt(252)) for _ in range(100)]
p_value = float(np.mean([ps >= sharpe for ps in perm_sharpes]))
perm_pass = p_value < 0.05
if perm_pass: gates_passed += 1
print(f"G1 Perm: p={p_value:.3f} {'PASS' if perm_pass else 'FAIL'}")

block_size = len(daily_rets) // 4
block_sharpes = [float(daily_rets.iloc[b*block_size:(b+1)*block_size].mean() / daily_rets.iloc[b*block_size:(b+1)*block_size].std() * np.sqrt(252)) if daily_rets.iloc[b*block_size:(b+1)*block_size].std() > 0 else 0 for b in range(4)]
all_pos = all(s > 0 for s in block_sharpes)
cv = float(np.std(block_sharpes) / np.mean(block_sharpes)) if np.mean(block_sharpes) != 0 else 999
sub_pass = all_pos and cv < 1.0
if sub_pass: gates_passed += 1
print(f"G2 Sub: blocks={[round(s,2) for s in block_sharpes]}, CV={cv:.3f} {'PASS' if sub_pass else 'FAIL'}")

p5, p95 = daily_rets.quantile(0.05), daily_rets.quantile(0.95)
trimmed = daily_rets[(daily_rets >= p5) & (daily_rets <= p95)]
t_sharpe = float(trimmed.mean() / trimmed.std() * np.sqrt(252)) if trimmed.std() > 0 else 0
degrad = float(1 - t_sharpe / sharpe) if sharpe != 0 else 0
o_pass = abs(degrad) < 0.30
if o_pass: gates_passed += 1
print(f"G3 Outlier: degrad={degrad:.1%} {'PASS' if o_pass else 'FAIL'}")

green_rets = daily_rets[spy_rets > 0]
red_rets = daily_rets[spy_rets < 0]
gs = float(green_rets.mean() / green_rets.std() * np.sqrt(252)) if green_rets.std() > 0 else 0
rs = float(red_rets.mean() / red_rets.std() * np.sqrt(252)) if red_rets.std() > 0 else 0
gap = float(abs(gs - rs) / max(abs(gs), abs(rs), 0.01))
r1_pass = gap < 0.50
if r1_pass: gates_passed += 1
print(f"G4 R1: green={gs:.3f}, red={rs:.3f}, gap={gap:.3f} {'PASS' if r1_pass else 'FAIL'}")

verdict = "PASS" if gates_passed >= 4 else "FAIL"
print(f"\nVERDICT: {gates_passed}/4 — {verdict}")

results = {
    'strategy': 'ML Treasury Maturity Spread',
    'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
    'cagr': round(cagr * 100, 1), 'max_dd': round(max_dd * 100, 1),
    'calmar': round(calmar, 2), 'profit_factor': round(pf, 2),
    'win_rate': round(win_rate * 100, 1), 'spy_corr': round(spy_corr, 3),
    'gates_passed': gates_passed, 'verdict': verdict,
    'adversarial': {
        'permutation': {'p_value': round(p_value, 3), 'pass': bool(perm_pass)},
        'sub_period': {'cv': round(cv, 3), 'all_positive': bool(all_pos), 'pass': bool(sub_pass)},
        'outlier': {'degradation': round(degrad, 3), 'pass': bool(o_pass)},
        'r1_regime': {'green': round(gs, 3), 'red': round(rs, 3), 'gap': round(gap, 3), 'pass': bool(r1_pass)},
    },
}
with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)
equity.to_csv(OUTPUT / 'equity_curve.csv')
print(f"\nSaved. Done.")
