#!/usr/bin/env python3
"""
ML Growth vs Value Relative Value
====================================
Strategy: ML predicts whether growth (IWF) or value (IWD) will outperform
over next 21 days. This is a within-equity relative value trade.

Rationale: Growth/value rotation is one of the most persistent factor rotations.
It's driven by interest rates, credit conditions, earnings expectations, and
momentum. Unlike cross-asset rotation, this is a relative bet within equities,
giving lower correlation to SPY direction.

Walk-forward: 252d train, 21d advance, sliding window (HC #0)
Adversarial: 4-gate validation
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

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_growth_value_spread")
OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"{'='*60}")
print(f"ML GROWTH vs VALUE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}")

# ─── Config ───
GROWTH = 'IWF'   # iShares Russell 1000 Growth
VALUE = 'IWD'    # iShares Russell 1000 Value
MACRO_TICKERS = ['^VIX', 'UUP', 'SPY', 'TLT', 'GLD', 'HYG', 'LQD', '^TNX', 'IEF', 'QQQ', 'DBC', 'TIP']

TRAIN_WINDOW = 252
ADVANCE_DAYS = 21
CAPITAL = 100000

# ─── Download data ───
print("Downloading data...")
start_date = '2008-01-01'
all_tickers = list(set([GROWTH, VALUE] + MACRO_TICKERS))
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

# ─── Build features ───
features = pd.DataFrame(index=prices.index)

# Growth/Value ratio
prices['gv_ratio'] = prices[GROWTH] / prices[VALUE]
features['gv_ratio'] = prices['gv_ratio']
features['gv_zscore_21d'] = (prices['gv_ratio'] - prices['gv_ratio'].rolling(21).mean()) / prices['gv_ratio'].rolling(21).std()
features['gv_zscore_63d'] = (prices['gv_ratio'] - prices['gv_ratio'].rolling(63).mean()) / prices['gv_ratio'].rolling(63).std()
features['gv_zscore_126d'] = (prices['gv_ratio'] - prices['gv_ratio'].rolling(126).mean()) / prices['gv_ratio'].rolling(126).std()
features['gv_mom_5d'] = prices['gv_ratio'].pct_change(5)
features['gv_mom_21d'] = prices['gv_ratio'].pct_change(21)
features['gv_mom_63d'] = prices['gv_ratio'].pct_change(63)
features['gv_vol_21d'] = prices['gv_ratio'].pct_change().rolling(21).std()

# Individual factor momentum
for f_etf in [GROWTH, VALUE]:
    if f_etf in prices.columns:
        features[f'{f_etf}_mom_5d'] = prices[f_etf].pct_change(5)
        features[f'{f_etf}_mom_21d'] = prices[f_etf].pct_change(21)
        features[f'{f_etf}_mom_63d'] = prices[f_etf].pct_change(63)
        features[f'{f_etf}_vol_21d'] = returns[f_etf].rolling(21).std()

# Relative vol
features['vol_ratio'] = returns[GROWTH].rolling(21).std() / returns[VALUE].rolling(21).std().clip(lower=1e-6)

# Interest rates (key driver of growth vs value)
if 'TNX' in prices.columns:
    features['tnx_level'] = prices['TNX']
    features['tnx_mom_5d'] = prices['TNX'].pct_change(5)
    features['tnx_mom_21d'] = prices['TNX'].pct_change(21)
    features['tnx_mom_63d'] = prices['TNX'].pct_change(63)

# Yield curve (steepening favors value)
if 'IEF' in prices.columns and 'TLT' in prices.columns:
    features['curve'] = prices['TLT'] / prices['IEF']
    features['curve_mom_21d'] = features['curve'].pct_change(21)

# Bonds (duration matters for growth)
if 'TLT' in prices.columns:
    features['tlt_mom_21d'] = prices['TLT'].pct_change(21)
    features['tlt_mom_63d'] = prices['TLT'].pct_change(63)

# VIX (vol regime)
if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX']
    features['vix_zscore'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()
    features['vix_mom_21d'] = prices['VIX'].pct_change(21)

# Credit (value more credit-sensitive)
if 'HYG' in prices.columns and 'LQD' in prices.columns:
    features['credit_spread'] = (returns['HYG'] - returns['LQD']).rolling(21).mean()
    features['credit_mom_21d'] = (prices['HYG'] / prices['LQD']).pct_change(21)

# USD
if 'UUP' in prices.columns:
    features['usd_mom_21d'] = prices['UUP'].pct_change(21)

# TIPS (inflation expectations favor value)
if 'TIP' in prices.columns:
    features['tip_mom_21d'] = prices['TIP'].pct_change(21)

# Commodities (favor value/cyclicals)
if 'DBC' in prices.columns:
    features['dbc_mom_21d'] = prices['DBC'].pct_change(21)

# Gold
if 'GLD' in prices.columns:
    features['gld_mom_21d'] = prices['GLD'].pct_change(21)

# QQQ (tech/growth proxy)
if 'QQQ' in prices.columns:
    features['qqq_mom_21d'] = prices['QQQ'].pct_change(21)
    features['qqq_vs_spy'] = prices['QQQ'].pct_change(21) - prices['SPY'].pct_change(21) if 'SPY' in prices.columns else 0

features = features.dropna()
print(f"Features: {features.shape[1]} cols, {len(features)} rows")

# ─── Walk-forward ───
print("\nRunning walk-forward backtest...")

returns['gv_spread'] = returns[GROWTH] - returns[VALUE]
fwd = returns['gv_spread'].rolling(ADVANCE_DAYS).sum().shift(-ADVANCE_DAYS)

all_dates = []
all_actuals = []
all_positions = []

for i in range(TRAIN_WINDOW + 252, len(features) - ADVANCE_DAYS, ADVANCE_DAYS):
    train_start = max(0, i - TRAIN_WINDOW)
    train_idx = features.index[train_start:i]
    test_idx = features.index[i:min(i + ADVANCE_DAYS, len(features))]

    train_labels = fwd.reindex(train_idx).dropna()
    common = train_idx.intersection(train_labels.index)

    if len(common) < 50:
        continue

    X_tr = np.nan_to_num(features.loc[common].values, nan=0, posinf=0, neginf=0)
    y_tr = (train_labels.loc[common] > 0).astype(int).values

    if len(np.unique(y_tr)) < 2:
        continue

    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, n_jobs=1
    )
    model.fit(X_tr, y_tr)

    X_test = np.nan_to_num(features.loc[test_idx].values, nan=0, posinf=0, neginf=0)
    probs = model.predict_proba(X_test)[:, 1]

    for dt, prob in zip(test_idx, probs):
        if dt not in returns.index:
            continue

        if prob > 0.6:
            daily_ret = returns.loc[dt, GROWTH]
            pos = 'GROWTH'
        elif prob < 0.4:
            daily_ret = returns.loc[dt, VALUE]
            pos = 'VALUE'
        else:
            daily_ret = 0.5 * returns.loc[dt, GROWTH] + 0.5 * returns.loc[dt, VALUE]
            pos = 'BLEND'

        all_dates.append(dt)
        all_actuals.append(daily_ret)
        all_positions.append(pos)

print(f"Walk-forward: {len(all_dates)} trading days")

# ─── Equity curve & metrics ───
equity = pd.Series(index=all_dates, data=np.cumprod(1 + np.array(all_actuals)) * CAPITAL)
daily_rets = pd.Series(index=all_dates, data=all_actuals)
positions = pd.Series(index=all_dates, data=all_positions)

ann_ret = daily_rets.mean() * 252
ann_vol = daily_rets.std() * np.sqrt(252)
sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
neg_vol = daily_rets[daily_rets < 0].std() * np.sqrt(252) if len(daily_rets[daily_rets < 0]) > 0 else 1
sortino = ann_ret / neg_vol
max_dd = (equity / equity.cummax() - 1).min()
calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
years = len(daily_rets) / 252
cagr = (equity.iloc[-1] / CAPITAL) ** (1/years) - 1 if years > 0 else 0
win_rate = (daily_rets > 0).mean()
pf = abs(daily_rets[daily_rets > 0].sum() / daily_rets[daily_rets < 0].sum()) if daily_rets[daily_rets < 0].sum() != 0 else 0

spy_rets = returns['SPY'].reindex(daily_rets.index).fillna(0) if 'SPY' in returns.columns else daily_rets * 0
spy_corr = daily_rets.corr(spy_rets)

print(f"\n{'='*60}")
print(f"STRATEGY METRICS")
print(f"{'='*60}")
print(f"CAGR:           {cagr:.1%}")
print(f"Sharpe:         {sharpe:.3f}")
print(f"Sortino:        {sortino:.3f}")
print(f"Max Drawdown:   {max_dd:.1%}")
print(f"Calmar:         {calmar:.2f}")
print(f"PF:             {pf:.2f}")
print(f"Win Rate:       {win_rate:.1%}")
print(f"SPY Corr:       {spy_corr:.3f}")
print(f"Positions:      {positions.value_counts().to_dict()}")

# ─── Adversarial ───
print(f"\n{'='*60}")
print(f"ADVERSARIAL VALIDATION")
print(f"{'='*60}")

gates_passed = 0

# Gate 1
perm_sharpes = [np.mean(daily_rets.sample(frac=1, replace=False).values) / daily_rets.std() * np.sqrt(252) for _ in range(100)]
p_value = np.mean([ps >= sharpe for ps in perm_sharpes])
perm_pass = p_value < 0.05
if perm_pass: gates_passed += 1
print(f"G1 Perm: p={p_value:.3f} {'PASS' if perm_pass else 'FAIL'}")

# Gate 2
block_size = len(daily_rets) // 4
block_sharpes = []
for b in range(4):
    block = daily_rets.iloc[b*block_size:(b+1)*block_size]
    bs = block.mean() / block.std() * np.sqrt(252) if block.std() > 0 else 0
    block_sharpes.append(bs)
all_pos = all(s > 0 for s in block_sharpes)
cv = np.std(block_sharpes) / np.mean(block_sharpes) if np.mean(block_sharpes) != 0 else 999
sub_pass = all_pos and cv < 1.0
if sub_pass: gates_passed += 1
print(f"G2 Sub-period: blocks={[round(s,2) for s in block_sharpes]}, CV={cv:.3f} {'PASS' if sub_pass else 'FAIL'}")

# Gate 3
p5, p95 = daily_rets.quantile(0.05), daily_rets.quantile(0.95)
trimmed = daily_rets[(daily_rets >= p5) & (daily_rets <= p95)]
t_sharpe = trimmed.mean() / trimmed.std() * np.sqrt(252) if trimmed.std() > 0 else 0
degrad = 1 - t_sharpe / sharpe if sharpe != 0 else 0
o_pass = abs(degrad) < 0.30
if o_pass: gates_passed += 1
print(f"G3 Outlier: degrad={degrad:.1%} {'PASS' if o_pass else 'FAIL'}")

# Gate 4
green_rets = daily_rets[spy_rets > 0]
red_rets = daily_rets[spy_rets < 0]
gs = green_rets.mean() / green_rets.std() * np.sqrt(252) if green_rets.std() > 0 else 0
rs = red_rets.mean() / red_rets.std() * np.sqrt(252) if red_rets.std() > 0 else 0
gap = abs(gs - rs) / max(abs(gs), abs(rs), 0.01)
r1_pass = gap < 0.50
if r1_pass: gates_passed += 1
print(f"G4 R1: green={gs:.3f}, red={rs:.3f}, gap={gap:.3f} {'PASS' if r1_pass else 'FAIL'}")

verdict = "PASS" if gates_passed >= 4 else "FAIL"
print(f"\nVERDICT: {gates_passed}/4 — {verdict}")

# ─── Save ───
results = {
    'strategy': 'ML Growth vs Value Spread',
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
