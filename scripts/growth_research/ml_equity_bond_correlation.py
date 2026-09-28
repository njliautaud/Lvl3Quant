#!/usr/bin/env python3
"""
ML Equity-Bond Correlation Regime
===================================
Strategy: ML predicts shifts in the equity-bond correlation regime.

Rationale: The stock-bond correlation is not constant. In some regimes bonds
hedge equity risk (negative correlation), in others they don't (positive
correlation, e.g., 2022 when both stocks and bonds fell). This strategy:

- When correlation is expected to be NEGATIVE (bonds hedge): overweight 60/40
- When correlation is expected to be POSITIVE (bonds don't hedge): switch to
  equity + gold or equity + cash for hedging instead

Instruments:
- SPY (equity), TLT (long bonds), GLD (gold), SHY (cash)
- 60/40 SPY/TLT when bonds hedge; SPY+GLD or SPY+SHY when they don't

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

OUTPUT = Path("/home/jupiter/Lvl3Quant/output/ml_equity_bond_correlation")
OUTPUT.mkdir(parents=True, exist_ok=True)

print(f"{'='*60}")
print(f"ML EQUITY-BOND CORRELATION — {datetime.now().strftime('%Y-%m-%d %H:%M')}")
print(f"{'='*60}")

# ─── Config ───
EQUITY = 'SPY'
BONDS = 'TLT'
GOLD = 'GLD'
CASH = 'SHY'
MACRO_TICKERS = ['^VIX', 'UUP', 'HYG', 'LQD', 'DBC', 'EEM', '^TNX', 'IEF', 'TIP']

TRAIN_WINDOW = 252
ADVANCE_DAYS = 21
CAPITAL = 100000
CORR_WINDOW = 63  # Rolling window for equity-bond correlation

# ─── Download data ───
print("Downloading data...")
start_date = '2008-01-01'
all_tickers = list(set([EQUITY, BONDS, GOLD, CASH] + MACRO_TICKERS))
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

# Core: rolling equity-bond correlation
eq_bond_corr = returns[EQUITY].rolling(CORR_WINDOW).corr(returns[BONDS])
features['eq_bond_corr_63d'] = eq_bond_corr
features['eq_bond_corr_21d'] = returns[EQUITY].rolling(21).corr(returns[BONDS])
features['eq_bond_corr_126d'] = returns[EQUITY].rolling(126).corr(returns[BONDS])

# Correlation momentum (is correlation trending more positive?)
features['corr_mom_21d'] = eq_bond_corr - eq_bond_corr.shift(21)
features['corr_mom_63d'] = eq_bond_corr - eq_bond_corr.shift(63)
features['corr_zscore'] = (eq_bond_corr - eq_bond_corr.rolling(252).mean()) / eq_bond_corr.rolling(252).std()

# Equity features
features['spy_mom_5d'] = prices[EQUITY].pct_change(5)
features['spy_mom_21d'] = prices[EQUITY].pct_change(21)
features['spy_mom_63d'] = prices[EQUITY].pct_change(63)
features['spy_vol_21d'] = returns[EQUITY].rolling(21).std()
features['spy_vol_63d'] = returns[EQUITY].rolling(63).std()
features['spy_drawdown'] = prices[EQUITY] / prices[EQUITY].rolling(252).max() - 1

# Bond features
features['tlt_mom_5d'] = prices[BONDS].pct_change(5)
features['tlt_mom_21d'] = prices[BONDS].pct_change(21)
features['tlt_mom_63d'] = prices[BONDS].pct_change(63)
features['tlt_vol_21d'] = returns[BONDS].rolling(21).std()

# Gold features
if GOLD in prices.columns:
    features['gld_mom_21d'] = prices[GOLD].pct_change(21)
    features['gld_mom_63d'] = prices[GOLD].pct_change(63)
    features['eq_gold_corr_63d'] = returns[EQUITY].rolling(CORR_WINDOW).corr(returns[GOLD])

# Yield features
if 'TNX' in prices.columns:
    features['tnx_level'] = prices['TNX']
    features['tnx_mom_21d'] = prices['TNX'].pct_change(21)
    features['tnx_mom_63d'] = prices['TNX'].pct_change(63)

# VIX
if 'VIX' in prices.columns:
    features['vix_level'] = prices['VIX']
    features['vix_zscore'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()
    features['vix_mom_21d'] = prices['VIX'].pct_change(21)

# Credit spread
if 'HYG' in prices.columns and 'LQD' in prices.columns:
    features['credit_spread'] = (returns['HYG'] - returns['LQD']).rolling(21).mean()
    features['credit_mom_21d'] = (prices['HYG'] / prices['LQD']).pct_change(21)

# TIPS (inflation expectations)
if 'TIP' in prices.columns:
    features['tip_mom_21d'] = prices['TIP'].pct_change(21)
    features['tip_vs_tlt'] = prices['TIP'].pct_change(21) - prices[BONDS].pct_change(21)

# IEF (intermediate bonds, curve shape)
if 'IEF' in prices.columns:
    features['ief_mom_21d'] = prices['IEF'].pct_change(21)
    features['curve_slope'] = prices[BONDS] / prices['IEF']

# USD
if 'UUP' in prices.columns:
    features['usd_mom_21d'] = prices['UUP'].pct_change(21)

# EM (global risk appetite)
if 'EEM' in prices.columns:
    features['eem_mom_21d'] = prices['EEM'].pct_change(21)

# Commodities
if 'DBC' in prices.columns:
    features['dbc_mom_21d'] = prices['DBC'].pct_change(21)

features = features.dropna()
print(f"Features: {features.shape[1]} cols, {len(features)} rows")

# ─── Walk-forward backtest ───
print("\nRunning walk-forward backtest...")

# Target: Will equity-bond correlation be NEGATIVE over next 21 days?
# When negative, 60/40 is good. When positive, need alternative hedge.
fwd_corr = returns[EQUITY].rolling(21).corr(returns[BONDS]).shift(-21)

# Portfolio construction:
# Regime NEGATIVE_CORR: 60% SPY + 40% TLT (classic 60/40, bonds hedge)
# Regime POSITIVE_CORR: 50% SPY + 30% GLD + 20% SHY (gold/cash hedge instead)
# Regime NEUTRAL: 40% SPY + 20% TLT + 20% GLD + 20% SHY (diversified)

all_dates = []
all_actuals = []
all_positions = []
all_preds = []

for i in range(TRAIN_WINDOW + 252, len(features) - ADVANCE_DAYS, ADVANCE_DAYS):
    train_end = i
    train_start = max(0, i - TRAIN_WINDOW)

    train_idx = features.index[train_start:train_end]
    test_start = i
    test_end = min(i + ADVANCE_DAYS, len(features))
    test_idx = features.index[test_start:test_end]

    # Training
    train_labels = fwd_corr.reindex(train_idx).dropna()
    common_train = train_idx.intersection(train_labels.index)

    if len(common_train) < 50:
        continue

    X_tr = np.nan_to_num(features.loc[common_train].values, nan=0, posinf=0, neginf=0)
    y_tr = (train_labels.loc[common_train] < 0).astype(int).values  # 1 = negative corr (bonds hedge)

    if len(np.unique(y_tr)) < 2:
        continue

    model = lgb.LGBMClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.05,
        min_child_samples=5, subsample=0.8, colsample_bytree=0.8,
        verbose=-1, n_jobs=1
    )
    model.fit(X_tr, y_tr)

    # Predict
    X_test = np.nan_to_num(features.loc[test_idx].values, nan=0, posinf=0, neginf=0)
    probs = model.predict_proba(X_test)[:, 1]  # P(bonds hedge)

    for dt, prob in zip(test_idx, probs):
        if dt not in returns.index:
            continue

        spy_ret = returns.loc[dt, EQUITY]
        tlt_ret = returns.loc[dt, BONDS]
        gld_ret = returns.loc[dt, GOLD] if GOLD in returns.columns else 0
        shy_ret = returns.loc[dt, CASH]

        if prob > 0.6:
            # Bonds expected to hedge → classic 60/40
            daily_ret = 0.60 * spy_ret + 0.40 * tlt_ret
            pos = 'BONDS_HEDGE'
        elif prob < 0.4:
            # Bonds NOT hedging → use gold/cash instead
            daily_ret = 0.50 * spy_ret + 0.30 * gld_ret + 0.20 * shy_ret
            pos = 'ALT_HEDGE'
        else:
            # Uncertain → diversified
            daily_ret = 0.40 * spy_ret + 0.20 * tlt_ret + 0.20 * gld_ret + 0.20 * shy_ret
            pos = 'DIVERSIFIED'

        all_dates.append(dt)
        all_actuals.append(daily_ret)
        all_positions.append(pos)
        all_preds.append(prob)

print(f"Walk-forward: {len(all_dates)} trading days")

# ─── Build equity curve ───
equity = pd.Series(index=all_dates, data=np.cumprod(1 + np.array(all_actuals)) * CAPITAL)
daily_rets = pd.Series(index=all_dates, data=all_actuals)
positions = pd.Series(index=all_dates, data=all_positions)

# Benchmark: static 60/40
bench_rets = 0.60 * returns[EQUITY] + 0.40 * returns[BONDS]
bench_rets = bench_rets.reindex(daily_rets.index).fillna(0)

# ─── Metrics ───
def calc_metrics(rets, name="Strategy"):
    ann_ret = rets.mean() * 252
    ann_vol = rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0
    neg_vol = rets[rets < 0].std() * np.sqrt(252) if len(rets[rets < 0]) > 0 else 1
    sortino = ann_ret / neg_vol
    eq = (1 + rets).cumprod()
    max_dd = (eq / eq.cummax() - 1).min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0
    years = len(rets) / 252
    cagr = eq.iloc[-1] ** (1/years) - 1 if years > 0 else 0
    win_rate = (rets > 0).mean()
    pf = abs(rets[rets > 0].sum() / rets[rets < 0].sum()) if rets[rets < 0].sum() != 0 else 0
    return {
        'name': name, 'sharpe': sharpe, 'sortino': sortino, 'cagr': cagr,
        'max_dd': max_dd, 'calmar': calmar, 'pf': pf, 'wr': win_rate,
        'ann_vol': ann_vol, 'years': years
    }

strat = calc_metrics(daily_rets, "ML Eq-Bond Corr")
bench = calc_metrics(bench_rets, "Static 60/40")

# SPY correlation
spy_rets = returns[EQUITY].reindex(daily_rets.index).fillna(0)
spy_corr = daily_rets.corr(spy_rets)

print(f"\n{'='*60}")
print(f"STRATEGY METRICS")
print(f"{'='*60}")
for m in [strat, bench]:
    print(f"\n{m['name']}:")
    print(f"  CAGR: {m['cagr']:.1%}, Vol: {m['ann_vol']:.1%}, Sharpe: {m['sharpe']:.3f}")
    print(f"  Sortino: {m['sortino']:.3f}, MaxDD: {m['max_dd']:.1%}, Calmar: {m['calmar']:.2f}")
    print(f"  PF: {m['pf']:.2f}, WR: {m['wr']:.1%}")
print(f"\nSPY correlation: {spy_corr:.3f}")
print(f"Position mix: {positions.value_counts().to_dict()}")

# ─── Adversarial Validation ───
print(f"\n{'='*60}")
print(f"ADVERSARIAL VALIDATION")
print(f"{'='*60}")

gates_passed = 0
sharpe = strat['sharpe']

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
print(f"  Sharpe: {sharpe:.3f}, Perm mean: {np.mean(perm_sharpes):.3f}, p={p_value:.3f} {'PASS' if perm_pass else 'FAIL'}")

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
    'strategy': 'ML Equity-Bond Correlation Regime',
    'sharpe': round(sharpe, 3),
    'sortino': round(strat['sortino'], 3),
    'cagr': round(strat['cagr'] * 100, 1),
    'max_dd': round(strat['max_dd'] * 100, 1),
    'calmar': round(strat['calmar'], 2),
    'profit_factor': round(strat['pf'], 2),
    'win_rate': round(strat['wr'] * 100, 1),
    'spy_corr': round(spy_corr, 3),
    'benchmark_sharpe': round(bench['sharpe'], 3),
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
