#!/usr/bin/env python3
"""
ML Inflation Trade Timing
===========================
Strategy: ML predicts inflation regime shifts and rotates between
inflation-benefiting assets and deflation-benefiting assets.

Rationale: Inflation regimes are one of the most important macro drivers.
Different assets perform very differently in rising vs falling inflation:
- Rising inflation: commodities (DBC), TIPS (TIP), gold (GLD), energy (XLE)
- Falling inflation: long bonds (TLT), growth stocks (QQQ), cash (SHY)

ML uses breakeven inflation (TIP vs TLT), commodity momentum, dollar strength,
credit conditions, and VIX to predict which regime is coming.

Walk-forward: 252d train, 21d advance, sliding window (HC #0)
Adversarial: 4-gate validation
"""

import numpy as np
import pandas as pd
import warnings
import json
from pathlib import Path
from datetime import datetime, timedelta

warnings.filterwarnings('ignore')
import yfinance as yf
import lightgbm as lgb

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_inflation_trade")
OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"{'='*60}")
print(f"ML INFLATION TRADE — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}")

# ─── Config ───
# Inflation beneficiaries
TIPS = 'TIP'
COMMODITIES = 'DBC'
GOLD = 'GLD'
ENERGY = 'XLE'

# Deflation beneficiaries
LONG_BONDS = 'TLT'
GROWTH = 'QQQ'
CASH = 'SHY'

MACRO_TICKERS = ['^VIX', 'UUP', 'SPY', 'HYG', 'LQD', 'EEM', '^TNX', 'IEF']

TRAIN_WINDOW = 252
ADVANCE_DAYS = 21
CAPITAL = 100000

# ─── Download data ───
print("Downloading data...")
start_date = '2008-01-01'
all_tickers = list(set([TIPS, COMMODITIES, GOLD, ENERGY, LONG_BONDS, GROWTH, CASH, 'SPY'] + MACRO_TICKERS))
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

# Breakeven inflation proxy: TIP vs TLT spread
if TIPS in prices.columns and LONG_BONDS in prices.columns:
    prices['bei_proxy'] = prices[TIPS] / prices[LONG_BONDS]
    features['bei_level'] = prices['bei_proxy']
    features['bei_zscore_63d'] = (prices['bei_proxy'] - prices['bei_proxy'].rolling(63).mean()) / prices['bei_proxy'].rolling(63).std()
    features['bei_zscore_126d'] = (prices['bei_proxy'] - prices['bei_proxy'].rolling(126).mean()) / prices['bei_proxy'].rolling(126).std()
    features['bei_mom_5d'] = prices['bei_proxy'].pct_change(5)
    features['bei_mom_21d'] = prices['bei_proxy'].pct_change(21)
    features['bei_mom_63d'] = prices['bei_proxy'].pct_change(63)

# Commodity momentum (inflation signal)
if COMMODITIES in prices.columns:
    features['dbc_mom_5d'] = prices[COMMODITIES].pct_change(5)
    features['dbc_mom_21d'] = prices[COMMODITIES].pct_change(21)
    features['dbc_mom_63d'] = prices[COMMODITIES].pct_change(63)
    features['dbc_vol_21d'] = returns[COMMODITIES].rolling(21).std()

# Gold momentum
if GOLD in prices.columns:
    features['gld_mom_21d'] = prices[GOLD].pct_change(21)
    features['gld_mom_63d'] = prices[GOLD].pct_change(63)

# Energy
if ENERGY in prices.columns:
    features['xle_mom_21d'] = prices[ENERGY].pct_change(21)
    features['xle_mom_63d'] = prices[ENERGY].pct_change(63)

# Bond momentum
if LONG_BONDS in prices.columns:
    features['tlt_mom_21d'] = prices[LONG_BONDS].pct_change(21)
    features['tlt_mom_63d'] = prices[LONG_BONDS].pct_change(63)
    features['tlt_vol_21d'] = returns[LONG_BONDS].rolling(21).std()

# Growth vs value (inflation signal)
if GROWTH in prices.columns and 'SPY' in prices.columns:
    features['growth_vs_broad'] = prices[GROWTH].pct_change(21) - prices['SPY'].pct_change(21)

# TIPS momentum
if TIPS in prices.columns:
    features['tip_mom_21d'] = prices[TIPS].pct_change(21)
    features['tip_mom_63d'] = prices[TIPS].pct_change(63)

# Yield level and momentum
if 'TNX' in prices.columns:
    features['tnx_level'] = prices['TNX']
    features['tnx_mom_21d'] = prices['TNX'].pct_change(21)
    features['tnx_mom_63d'] = prices['TNX'].pct_change(63)

# VIX
if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX']
    features['vix_zscore'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()

# USD (strong dollar = deflationary)
if 'UUP' in prices.columns:
    features['usd_mom_21d'] = prices['UUP'].pct_change(21)
    features['usd_mom_63d'] = prices['UUP'].pct_change(63)

# Equity
if 'SPY' in prices.columns:
    features['spy_mom_21d'] = prices['SPY'].pct_change(21)
    features['spy_vol_21d'] = returns['SPY'].rolling(21).std()

# Credit
if 'HYG' in prices.columns and 'LQD' in prices.columns:
    features['credit_spread'] = (returns['HYG'] - returns['LQD']).rolling(21).mean()

# EM (inflation-sensitive)
if 'EEM' in prices.columns:
    features['eem_mom_21d'] = prices['EEM'].pct_change(21)

# Curve shape
if 'IEF' in prices.columns and LONG_BONDS in prices.columns:
    features['curve_slope'] = prices[LONG_BONDS] / prices['IEF']
    features['curve_mom_21d'] = features['curve_slope'].pct_change(21)

features = features.dropna()
print(f"Features: {features.shape[1]} cols, {len(features)} rows")

# ─── Walk-forward backtest ───
print("\nRunning walk-forward backtest...")

# Target: inflation basket outperforms deflation basket over next 21 days
# Inflation basket: equal weight TIP + DBC + GLD
# Deflation basket: equal weight TLT + QQQ
inf_tickers = [t for t in [TIPS, COMMODITIES, GOLD] if t in returns.columns]
def_tickers = [t for t in [LONG_BONDS, GROWTH] if t in returns.columns]

if not inf_tickers or not def_tickers:
    print("ERROR: Missing key tickers")
    exit(1)

returns['inflation_basket'] = returns[inf_tickers].mean(axis=1)
returns['deflation_basket'] = returns[def_tickers].mean(axis=1)
returns['inf_vs_def'] = returns['inflation_basket'] - returns['deflation_basket']

fwd = returns['inf_vs_def'].rolling(ADVANCE_DAYS).sum().shift(-ADVANCE_DAYS)

all_dates = []
all_actuals = []
all_positions = []

for i in range(TRAIN_WINDOW + 252, len(features) - ADVANCE_DAYS, ADVANCE_DAYS):
    train_end = i
    train_start = max(0, i - TRAIN_WINDOW)

    train_idx = features.index[train_start:train_end]
    test_start = i
    test_end = min(i + ADVANCE_DAYS, len(features))
    test_idx = features.index[test_start:test_end]

    train_labels = fwd.reindex(train_idx).dropna()
    common_train = train_idx.intersection(train_labels.index)

    if len(common_train) < 50:
        continue

    X_tr = np.nan_to_num(features.loc[common_train].values, nan=0, posinf=0, neginf=0)
    y_tr = (train_labels.loc[common_train] > 0).astype(int).values  # 1 = inflation wins

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
            # Inflation regime: overweight inflation beneficiaries
            daily_ret = returns.loc[dt, inf_tickers].mean()
            pos = 'INFLATION'
        elif prob < 0.4:
            # Deflation regime: overweight deflation beneficiaries
            daily_ret = returns.loc[dt, def_tickers].mean()
            pos = 'DEFLATION'
        else:
            # Balanced
            daily_ret = 0.5 * returns.loc[dt, inf_tickers].mean() + 0.5 * returns.loc[dt, def_tickers].mean()
            pos = 'BALANCED'

        all_dates.append(dt)
        all_actuals.append(daily_ret)
        all_positions.append(pos)

print(f"Walk-forward: {len(all_dates)} trading days")

# ─── Equity curve ───
equity = pd.Series(index=all_dates, data=np.cumprod(1 + np.array(all_actuals)) * CAPITAL)
daily_rets = pd.Series(index=all_dates, data=all_actuals)
positions = pd.Series(index=all_dates, data=all_positions)

# ─── Metrics ───
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
print(f"Annual Vol:     {ann_vol:.1%}")
print(f"Sharpe:         {sharpe:.3f}")
print(f"Sortino:        {sortino:.3f}")
print(f"Max Drawdown:   {max_dd:.1%}")
print(f"Calmar:         {calmar:.2f}")
print(f"Profit Factor:  {pf:.2f}")
print(f"Win Rate:       {win_rate:.1%}")
print(f"SPY Correlation:{spy_corr:.3f}")
print(f"Position mix:   {positions.value_counts().to_dict()}")

# ─── Adversarial Validation ───
print(f"\n{'='*60}")
print(f"ADVERSARIAL VALIDATION")
print(f"{'='*60}")

gates_passed = 0

# Gate 1: Permutation
print("\nGate 1: Permutation test...")
perm_sharpes = []
for _ in range(100):
    shuffled = daily_rets.sample(frac=1, replace=False).values
    ps = shuffled.mean() / shuffled.std() * np.sqrt(252) if shuffled.std() > 0 else 0
    perm_sharpes.append(ps)
p_value = np.mean([ps >= sharpe for ps in perm_sharpes])
perm_pass = p_value < 0.05
if perm_pass: gates_passed += 1
print(f"  Real: {sharpe:.3f}, Perm mean: {np.mean(perm_sharpes):.3f}, p={p_value:.3f} {'PASS' if perm_pass else 'FAIL'}")

# Gate 2: Sub-period
print("\nGate 2: Sub-period consistency...")
n_blocks = 4
block_size = len(daily_rets) // n_blocks
block_sharpes = []
for b in range(n_blocks):
    block = daily_rets.iloc[b*block_size:(b+1)*block_size]
    bs = block.mean() / block.std() * np.sqrt(252) if block.std() > 0 else 0
    block_sharpes.append(bs)
    print(f"  Block {b+1}: Sharpe {bs:.3f}")
all_positive = all(s > 0 for s in block_sharpes)
cv = np.std(block_sharpes) / np.mean(block_sharpes) if np.mean(block_sharpes) != 0 else 999
sub_pass = all_positive and cv < 1.0
if sub_pass: gates_passed += 1
print(f"  CV: {cv:.3f}, All positive: {all_positive} {'PASS' if sub_pass else 'FAIL'}")

# Gate 3: Outlier removal
print("\nGate 3: Outlier removal...")
p5, p95 = daily_rets.quantile(0.05), daily_rets.quantile(0.95)
trimmed = daily_rets[(daily_rets >= p5) & (daily_rets <= p95)]
trimmed_sharpe = trimmed.mean() / trimmed.std() * np.sqrt(252) if trimmed.std() > 0 else 0
degradation = 1 - trimmed_sharpe / sharpe if sharpe != 0 else 0
outlier_pass = abs(degradation) < 0.30
if outlier_pass: gates_passed += 1
print(f"  Original: {sharpe:.3f}, Trimmed: {trimmed_sharpe:.3f}, Degrad: {degradation:.1%} {'PASS' if outlier_pass else 'FAIL'}")

# Gate 4: R1 Regime
print("\nGate 4: R1 Regime test...")
green_mask = spy_rets > 0
red_mask = spy_rets < 0
green_rets = daily_rets[green_mask]
red_rets = daily_rets[red_mask]
green_sharpe = green_rets.mean() / green_rets.std() * np.sqrt(252) if green_rets.std() > 0 else 0
red_sharpe = red_rets.mean() / red_rets.std() * np.sqrt(252) if red_rets.std() > 0 else 0
gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)
r1_pass = gap < 0.50
if r1_pass: gates_passed += 1
print(f"  Green: {green_sharpe:.3f}, Red: {red_sharpe:.3f}, Gap: {gap:.3f} {'PASS' if r1_pass else 'FAIL'}")

verdict = "PASS" if gates_passed >= 4 else "FAIL"
print(f"\n{'='*60}")
print(f"VERDICT: {gates_passed}/4 gates — {verdict}")
print(f"{'='*60}")

# ─── Save ───
results = {
    'strategy': 'ML Inflation Trade Timing',
    'sharpe': round(sharpe, 3),
    'sortino': round(sortino, 3),
    'cagr': round(cagr * 100, 1),
    'max_dd': round(max_dd * 100, 1),
    'calmar': round(calmar, 2),
    'profit_factor': round(pf, 2),
    'win_rate': round(win_rate * 100, 1),
    'spy_corr': round(spy_corr, 3),
    'gates_passed': gates_passed,
    'adversarial': {
        'permutation': {'p_value': round(p_value, 3), 'pass': perm_pass},
        'sub_period': {'cv': round(cv, 3), 'all_positive': all_positive, 'pass': sub_pass},
        'outlier': {'degradation': round(degradation, 3), 'pass': outlier_pass},
        'r1_regime': {'green': round(green_sharpe, 3), 'red': round(red_sharpe, 3), 'gap': round(gap, 3), 'pass': r1_pass},
    },
    'verdict': verdict,
}

with open(OUTPUT / 'results.json', 'w') as f:
    json.dump(results, f, indent=2)

equity.to_csv(OUTPUT / 'equity_curve.csv')
print(f"\nResults saved. Done.")
