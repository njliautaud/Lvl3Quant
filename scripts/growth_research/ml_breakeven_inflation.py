#!/usr/bin/env python3
"""
ML Breakeven Inflation Trade
================================
Strategy: ML predicts whether breakeven inflation will rise (long TIPS, short
nominal Treasuries) or fall (long nominal Treasuries, short TIPS).

Rationale: Breakeven inflation = nominal yield - TIPS yield. It represents
the market's inflation expectation. When breakevens rise, TIPS outperform
nominal bonds. When they fall, nominal bonds outperform.

This is a pure relative value trade within fixed income with ZERO equity
exposure — very low SPY correlation expected. Similar structure to the
yield curve trade that passed 4/4.

Instruments:
- TIP (TIPS ETF) — inflation-protected
- IEF (7-10yr nominal Treasury) — similar duration to TIP, clean spread
- Alternative: TLT vs TIP but duration mismatch makes it messy

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

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_breakeven_inflation")
OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"{'='*60}")
print(f"ML BREAKEVEN INFLATION TRADE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}")

# ─── Config ───
TIPS_ETF = 'TIP'
NOMINAL_ETF = 'IEF'  # Duration-matched with TIP (~7yr)
MACRO_TICKERS = ['^VIX', 'UUP', 'SPY', 'TLT', 'GLD', 'HYG', 'LQD', '^TNX',
                 'DBC', 'SHY', 'EEM', 'XLE', 'SCHP']

TRAIN_WINDOW = 252
ADVANCE_DAYS = 21
CAPITAL = 100000

# ─── Download ───
print("Downloading data...")
start_date = '2008-01-01'
all_tickers = list(set([TIPS_ETF, NOMINAL_ETF] + MACRO_TICKERS))
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

# ─── Features ───
features = pd.DataFrame(index=prices.index)

# Breakeven spread: TIP/IEF ratio (rises when inflation expectations increase)
prices['bei_ratio'] = prices[TIPS_ETF] / prices[NOMINAL_ETF]
features['bei_ratio'] = prices['bei_ratio']
features['bei_zscore_21d'] = (prices['bei_ratio'] - prices['bei_ratio'].rolling(21).mean()) / prices['bei_ratio'].rolling(21).std()
features['bei_zscore_63d'] = (prices['bei_ratio'] - prices['bei_ratio'].rolling(63).mean()) / prices['bei_ratio'].rolling(63).std()
features['bei_zscore_126d'] = (prices['bei_ratio'] - prices['bei_ratio'].rolling(126).mean()) / prices['bei_ratio'].rolling(126).std()
features['bei_zscore_252d'] = (prices['bei_ratio'] - prices['bei_ratio'].rolling(252).mean()) / prices['bei_ratio'].rolling(252).std()
features['bei_mom_5d'] = prices['bei_ratio'].pct_change(5)
features['bei_mom_21d'] = prices['bei_ratio'].pct_change(21)
features['bei_mom_63d'] = prices['bei_ratio'].pct_change(63)
features['bei_vol_21d'] = prices['bei_ratio'].pct_change().rolling(21).std()
features['bei_vol_63d'] = prices['bei_ratio'].pct_change().rolling(63).std()

# TIPS momentum
features['tip_mom_5d'] = prices[TIPS_ETF].pct_change(5)
features['tip_mom_21d'] = prices[TIPS_ETF].pct_change(21)
features['tip_mom_63d'] = prices[TIPS_ETF].pct_change(63)
features['tip_vol_21d'] = returns[TIPS_ETF].rolling(21).std()

# SCHP (another TIPS proxy for confirmation)
if 'SCHP' in prices.columns:
    features['schp_mom_21d'] = prices['SCHP'].pct_change(21)

# Nominal bond momentum
features['ief_mom_5d'] = prices[NOMINAL_ETF].pct_change(5)
features['ief_mom_21d'] = prices[NOMINAL_ETF].pct_change(21)
features['ief_mom_63d'] = prices[NOMINAL_ETF].pct_change(63)
features['ief_vol_21d'] = returns[NOMINAL_ETF].rolling(21).std()

# Long bonds (duration signal)
if 'TLT' in prices.columns:
    features['tlt_mom_21d'] = prices['TLT'].pct_change(21)
    features['tlt_mom_63d'] = prices['TLT'].pct_change(63)

# Short-term rates
if 'SHY' in prices.columns:
    features['shy_mom_21d'] = prices['SHY'].pct_change(21)

# Commodity momentum (key inflation driver)
if 'DBC' in prices.columns:
    features['dbc_mom_5d'] = prices['DBC'].pct_change(5)
    features['dbc_mom_21d'] = prices['DBC'].pct_change(21)
    features['dbc_mom_63d'] = prices['DBC'].pct_change(63)
    features['dbc_vol_21d'] = returns['DBC'].rolling(21).std()

# Energy (inflation driver)
if 'XLE' in prices.columns:
    features['xle_mom_21d'] = prices['XLE'].pct_change(21)

# Gold (inflation hedge)
if 'GLD' in prices.columns:
    features['gld_mom_21d'] = prices['GLD'].pct_change(21)
    features['gld_mom_63d'] = prices['GLD'].pct_change(63)

# USD (strong dollar = deflationary)
if 'UUP' in prices.columns:
    features['usd_mom_21d'] = prices['UUP'].pct_change(21)
    features['usd_mom_63d'] = prices['UUP'].pct_change(63)

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
    features['credit'] = (returns['HYG'] - returns['LQD']).rolling(21).mean()

# Equity (risk appetite)
if 'SPY' in prices.columns:
    features['spy_mom_21d'] = prices['SPY'].pct_change(21)
    features['spy_vol_21d'] = returns['SPY'].rolling(21).std()

# EM (inflation-sensitive)
if 'EEM' in prices.columns:
    features['eem_mom_21d'] = prices['EEM'].pct_change(21)

features = features.dropna()
print(f"Features: {features.shape[1]} cols, {len(features)} rows")

# ─── Walk-forward ───
print("\nRunning walk-forward backtest...")

returns['bei_trade'] = returns[TIPS_ETF] - returns[NOMINAL_ETF]
fwd = returns['bei_trade'].rolling(ADVANCE_DAYS).sum().shift(-ADVANCE_DAYS)

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
    y_tr = (train_labels.loc[common] > 0).astype(int).values  # 1 = TIPS outperforms (inflation rising)

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
            # Inflation rising: long TIPS
            daily_ret = returns.loc[dt, TIPS_ETF]
            pos = 'INFLATION_UP'
        elif prob < 0.4:
            # Inflation falling: long nominal bonds
            daily_ret = returns.loc[dt, NOMINAL_ETF]
            pos = 'INFLATION_DOWN'
        else:
            # Neutral: equal weight
            daily_ret = 0.5 * returns.loc[dt, TIPS_ETF] + 0.5 * returns.loc[dt, NOMINAL_ETF]
            pos = 'NEUTRAL'

        all_dates.append(dt)
        all_actuals.append(daily_ret)
        all_positions.append(pos)

print(f"Walk-forward: {len(all_dates)} trading days")

# ─── Metrics ───
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

print(f"\n{'='*60}")
print(f"STRATEGY METRICS")
print(f"{'='*60}")
print(f"CAGR: {cagr:.1%}, Sharpe: {sharpe:.3f}, Sortino: {sortino:.3f}")
print(f"MaxDD: {max_dd:.1%}, Calmar: {calmar:.2f}, PF: {pf:.2f}, WR: {win_rate:.1%}")
print(f"SPY Corr: {spy_corr:.3f}")
print(f"Positions: {positions.value_counts().to_dict()}")

# ─── Adversarial ───
print(f"\n{'='*60}")
print(f"ADVERSARIAL VALIDATION")
print(f"{'='*60}")

gates_passed = 0

# Gate 1
perm_sharpes = [float(np.mean(daily_rets.sample(frac=1, replace=False).values) / daily_rets.std() * np.sqrt(252)) for _ in range(100)]
p_value = float(np.mean([ps >= sharpe for ps in perm_sharpes]))
perm_pass = p_value < 0.05
if perm_pass: gates_passed += 1
print(f"G1 Perm: p={p_value:.3f} {'PASS' if perm_pass else 'FAIL'}")

# Gate 2
block_size = len(daily_rets) // 4
block_sharpes = []
for b in range(4):
    block = daily_rets.iloc[b*block_size:(b+1)*block_size]
    bs = float(block.mean() / block.std() * np.sqrt(252)) if block.std() > 0 else 0
    block_sharpes.append(bs)
all_pos = all(s > 0 for s in block_sharpes)
cv = float(np.std(block_sharpes) / np.mean(block_sharpes)) if np.mean(block_sharpes) != 0 else 999
sub_pass = all_pos and cv < 1.0
if sub_pass: gates_passed += 1
print(f"G2 Sub: blocks={[round(s,2) for s in block_sharpes]}, CV={cv:.3f} {'PASS' if sub_pass else 'FAIL'}")

# Gate 3
p5, p95 = daily_rets.quantile(0.05), daily_rets.quantile(0.95)
trimmed = daily_rets[(daily_rets >= p5) & (daily_rets <= p95)]
t_sharpe = float(trimmed.mean() / trimmed.std() * np.sqrt(252)) if trimmed.std() > 0 else 0
degrad = float(1 - t_sharpe / sharpe) if sharpe != 0 else 0
o_pass = abs(degrad) < 0.30
if o_pass: gates_passed += 1
print(f"G3 Outlier: degrad={degrad:.1%} {'PASS' if o_pass else 'FAIL'}")

# Gate 4
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

# ─── Per-year returns ───
daily_rets_df = daily_rets.to_frame('ret')
daily_rets_df['year'] = daily_rets_df.index.year
yearly = daily_rets_df.groupby('year')['ret'].apply(lambda x: (1+x).prod()-1) * 100
print(f"\nPer-year returns:")
for yr, r in yearly.items():
    print(f"  {yr}: {r:+.1f}%")
neg_years = (yearly < 0).sum()
print(f"Negative years: {neg_years}")

# ─── Save ───
results = {
    'strategy': 'ML Breakeven Inflation Trade',
    'sharpe': round(sharpe, 3), 'sortino': round(sortino, 3),
    'cagr': round(cagr * 100, 1), 'max_dd': round(max_dd * 100, 1),
    'calmar': round(calmar, 2), 'profit_factor': round(pf, 2),
    'win_rate': round(win_rate * 100, 1), 'spy_corr': round(spy_corr, 3),
    'negative_years': int(neg_years),
    'gates_passed': gates_passed, 'verdict': verdict,
    'adversarial': {
        'permutation': {'p_value': round(p_value, 3), 'pass': bool(perm_pass)},
        'sub_period': {'cv': round(cv, 3), 'all_positive': bool(all_pos), 'pass': bool(sub_pass)},
        'outlier': {'degradation': round(degrad, 3), 'pass': bool(o_pass)},
        'r1_regime': {'green': round(gs, 3), 'red': round(rs, 3), 'gap': round(gap, 3), 'pass': bool(r1_pass)},
    },
    'yearly_returns': {str(yr): round(r, 1) for yr, r in yearly.items()},
}

with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

equity.to_csv(OUTPUT / 'equity_curve.csv')
print(f"\nSaved. Done.")
