#!/usr/bin/env python3
"""
ML Volatility Targeting Strategy
=================================
Concept: Use ML to forecast next-period realized volatility, then size UPRO
inversely to expected vol. When vol is low → full UPRO. When vol rises →
reduce to SPY or SHY. Target constant portfolio volatility.

This is different from regime prediction (binary) — it's a continuous
risk estimate that scales position size smoothly.

Features: cross-asset vol signals, term structure, momentum, credit spreads.
Walk-forward: 252d sliding window (HC #0).
Fixed capital: $100K (HC #713).
Adversarial validation: permutation, sub-period, outlier, R1 (HC #705).

Author: Claude (HC #714 ML exploration)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

from pathlib import Path
import json
from datetime import datetime
import sys

# Paths
BASE = Path("/home/jupiter/Lvl3Quant")
OUTPUT = BASE / "output" / "ml_vol_targeting"
OUTPUT.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print("ML VOLATILITY TARGETING STRATEGY")
print("=" * 70)

# ─────────────────────────────────────────────
# Step 1: Load data
# ─────────────────────────────────────────────
print("\n[1/7] Loading data...")

import yfinance as yf

tickers = {
    'SPY': 'SPY', 'UPRO': 'UPRO', 'QQQ': 'QQQ',
    'VIX': '^VIX', 'GLD': 'GLD', 'TLT': 'TLT',
    'HYG': 'HYG', 'IEF': 'IEF', 'SHY': 'SHY',
    'UUP': 'UUP', 'XLF': 'XLF', 'XLU': 'XLU',
    'IWM': 'IWM', 'EEM': 'EEM',
}

cache_file = OUTPUT / "raw_data.parquet"
if cache_file.exists():
    print("  Loading from cache...")
    prices = pd.read_parquet(cache_file)
else:
    print("  Downloading from yfinance...")
    data = yf.download(list(tickers.values()), start='2010-01-01', end='2026-07-18',
                       auto_adjust=True, progress=False)
    prices = data['Close'].copy()
    # Rename columns back to clean names
    inv_map = {v: k for k, v in tickers.items()}
    prices.columns = [inv_map.get(c, c) for c in prices.columns]
    prices.to_parquet(cache_file)

prices = prices.dropna(subset=['SPY', 'VIX'])
print(f"  Data: {len(prices)} days ({prices.index[0].date()} to {prices.index[-1].date()})")

# ─────────────────────────────────────────────
# Step 2: Build features
# ─────────────────────────────────────────────
print("\n[2/7] Building features...")

df = prices.copy()
spy_ret = df['SPY'].pct_change()

features = pd.DataFrame(index=df.index)

# Realized volatility features (target we're predicting)
for w in [5, 10, 20, 60]:
    features[f'spy_rvol_{w}d'] = spy_ret.rolling(w).std() * np.sqrt(252)

# VIX features
features['vix_level'] = df['VIX']
features['vix_5d_chg'] = df['VIX'].pct_change(5)
features['vix_20d_chg'] = df['VIX'].pct_change(20)
features['vix_z_20d'] = (df['VIX'] - df['VIX'].rolling(20).mean()) / df['VIX'].rolling(20).std()

# Term structure proxy (VIX vs realized vol ratio)
features['vix_rv_ratio'] = df['VIX'] / (spy_ret.rolling(20).std() * np.sqrt(252) * 100)

# Credit spread proxy (HYG vs IEF)
if 'HYG' in df.columns and 'IEF' in df.columns:
    hyg_ief = (df['HYG'] / df['IEF'])
    features['credit_spread_z'] = (hyg_ief - hyg_ief.rolling(60).mean()) / hyg_ief.rolling(60).std()
    features['credit_5d'] = hyg_ief.pct_change(5)

# SPY momentum features
for w in [5, 10, 20, 60]:
    features[f'spy_mom_{w}d'] = df['SPY'].pct_change(w)

# Drawdown from high
spy_cummax = df['SPY'].cummax()
features['spy_drawdown'] = (df['SPY'] - spy_cummax) / spy_cummax

# Cross-asset momentum
for asset in ['GLD', 'TLT', 'UUP', 'QQQ', 'IWM', 'EEM', 'XLF', 'XLU']:
    if asset in df.columns:
        features[f'{asset.lower()}_mom_20d'] = df[asset].pct_change(20)
        features[f'{asset.lower()}_vol_20d'] = df[asset].pct_change().rolling(20).std() * np.sqrt(252)

# Intraday range proxy (high-low would be better but we only have close)
features['spy_abs_ret_5d_avg'] = spy_ret.abs().rolling(5).mean()
features['spy_abs_ret_20d_avg'] = spy_ret.abs().rolling(20).mean()

# Skewness and kurtosis
features['spy_skew_20d'] = spy_ret.rolling(20).skew()
features['spy_kurt_20d'] = spy_ret.rolling(20).kurt()

# Vol of vol
rv20 = spy_ret.rolling(20).std() * np.sqrt(252)
features['vol_of_vol_20d'] = rv20.rolling(20).std()

# Correlation features (flight to quality)
features['spy_tlt_corr_20d'] = spy_ret.rolling(20).corr(df['TLT'].pct_change()) if 'TLT' in df.columns else np.nan
features['spy_gld_corr_20d'] = spy_ret.rolling(20).corr(df['GLD'].pct_change()) if 'GLD' in df.columns else np.nan

features = features.dropna()
print(f"  Features: {features.shape[1]} columns, {len(features)} valid rows")

# ─────────────────────────────────────────────
# Step 3: Define target — next 5-day realized vol
# ─────────────────────────────────────────────
print("\n[3/7] Building target (next-5d realized vol)...")

fwd_rvol = spy_ret.rolling(5).std().shift(-5) * np.sqrt(252)
fwd_rvol.name = 'fwd_rvol_5d'

# Also compute UPRO and SPY forward returns for strategy backtest
spy_fwd_1d = spy_ret.shift(-1)
upro_ret = df['UPRO'].pct_change() if 'UPRO' in df.columns else spy_ret * 3  # approximate
upro_fwd_1d = upro_ret.shift(-1)
shy_ret = df['SHY'].pct_change() if 'SHY' in df.columns else pd.Series(0.0002/252, index=df.index)  # ~risk-free
shy_fwd_1d = shy_ret.shift(-1)

# Align everything
common_idx = features.index.intersection(fwd_rvol.dropna().index)
common_idx = common_idx.intersection(spy_fwd_1d.dropna().index)
features = features.loc[common_idx]
target = fwd_rvol.loc[common_idx]

print(f"  Target: next-5d annualized vol. Median={target.median():.1%}, Mean={target.mean():.1%}")
print(f"  Valid samples: {len(common_idx)}")

# ─────────────────────────────────────────────
# Step 4: Walk-forward ML training
# ─────────────────────────────────────────────
print("\n[4/7] Walk-forward training (SLIDING 252d window)...")

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score

TRAIN_WINDOW = 252
predictions = pd.Series(dtype=float, name='pred_vol')
actuals = pd.Series(dtype=float, name='actual_vol')

n_folds = 0
for i in range(TRAIN_WINDOW, len(features) - 1):
    train_start = i - TRAIN_WINDOW

    X_train = features.iloc[train_start:i]
    y_train = target.iloc[train_start:i]
    X_test = features.iloc[i:i+1]
    y_test = target.iloc[i:i+1]

    # Drop any NaN in training
    valid = y_train.notna() & X_train.notna().all(axis=1)
    if valid.sum() < 100:
        continue

    model = GradientBoostingRegressor(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        random_state=42
    )
    model.fit(X_train[valid], y_train[valid])

    pred = model.predict(X_test)[0]
    predictions.loc[X_test.index[0]] = pred
    actuals.loc[X_test.index[0]] = y_test.values[0]
    n_folds += 1

    if n_folds % 500 == 0:
        print(f"  Fold {n_folds}: R² so far = {r2_score(actuals.dropna(), predictions.loc[actuals.dropna().index]):.4f}")

print(f"\n  Total folds: {n_folds}")
r2 = r2_score(actuals.dropna(), predictions.loc[actuals.dropna().index])
rmse = np.sqrt(mean_squared_error(actuals.dropna(), predictions.loc[actuals.dropna().index]))
corr = actuals.dropna().corr(predictions.loc[actuals.dropna().index])
print(f"  R²: {r2:.4f}")
print(f"  RMSE: {rmse:.4f}")
print(f"  Corr(pred, actual): {corr:.4f}")

# ─────────────────────────────────────────────
# Step 5: Build vol-targeting strategy
# ─────────────────────────────────────────────
print("\n[5/7] Building vol-targeting strategy...")

# Target 15% annual portfolio volatility
TARGET_VOL = 0.15
INITIAL_CAPITAL = 100_000

# Strategy: scale UPRO allocation inversely to predicted vol
# weight = min(1.0, TARGET_VOL / pred_vol)
# When pred_vol is low → full UPRO. When high → scale down, rest in SHY.

pred_dates = predictions.dropna().index
strategy_returns = pd.DataFrame(index=pred_dates)

# Align returns
spy_aligned = spy_fwd_1d.reindex(pred_dates)
upro_aligned = upro_fwd_1d.reindex(pred_dates)
shy_aligned = shy_fwd_1d.reindex(pred_dates)

# ML vol-targeting
ml_weights = (TARGET_VOL / predictions.loc[pred_dates]).clip(0.0, 1.0)
# UPRO's vol is ~3x SPY, so we need to account for that
# If pred_vol is SPY vol, then UPRO vol ≈ 3 * pred_vol
upro_vol_est = predictions.loc[pred_dates] * 3
ml_weights_upro = (TARGET_VOL / upro_vol_est).clip(0.0, 1.0)

strategy_returns['ml_vol_target'] = ml_weights_upro * upro_aligned + (1 - ml_weights_upro) * shy_aligned
strategy_returns['buy_hold_spy'] = spy_aligned
strategy_returns['buy_hold_upro'] = upro_aligned

# Also: simple vol targeting using REALIZED vol (no ML)
rv20_aligned = features['spy_rvol_20d'].reindex(pred_dates)
simple_weights_upro = (TARGET_VOL / (rv20_aligned * 3)).clip(0.0, 1.0)
strategy_returns['simple_vol_target'] = simple_weights_upro * upro_aligned + (1 - simple_weights_upro) * shy_aligned

# Also: VIX-based vol targeting (simplest)
vix_aligned = features['vix_level'].reindex(pred_dates)
vix_weights = pd.Series(1.0, index=pred_dates)
vix_weights[vix_aligned > 20] = 0.7
vix_weights[vix_aligned > 25] = 0.4
vix_weights[vix_aligned > 30] = 0.1
vix_weights[vix_aligned > 40] = 0.0
strategy_returns['vix_stepped'] = vix_weights * upro_aligned + (1 - vix_weights) * shy_aligned

strategy_returns = strategy_returns.dropna()
print(f"  Strategy period: {strategy_returns.index[0].date()} to {strategy_returns.index[-1].date()}")
print(f"  Trading days: {len(strategy_returns)}")

# Allocation stats for ML vol target
print(f"\n  ML Vol Target allocation stats:")
print(f"    Mean UPRO weight: {ml_weights_upro.mean():.1%}")
print(f"    Median UPRO weight: {ml_weights_upro.median():.1%}")
print(f"    Full UPRO (>90%): {(ml_weights_upro > 0.9).mean():.1%}")
print(f"    Defensive (<30%): {(ml_weights_upro < 0.3).mean():.1%}")

# ─────────────────────────────────────────────
# Step 6: Performance metrics
# ─────────────────────────────────────────────
print("\n[6/7] Computing performance metrics...")

def compute_metrics(returns, name):
    """Compute risk-adjusted metrics for a return series."""
    r = returns.dropna()
    if len(r) < 252:
        return None

    ann_ret = (1 + r).prod() ** (252 / len(r)) - 1
    ann_vol = r.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside = r[r < 0].std() * np.sqrt(252) if (r < 0).sum() > 10 else ann_vol
    sortino = ann_ret / downside if downside > 0 else 0

    cum = (1 + r).cumprod()
    peak = cum.cummax()
    dd = (cum - peak) / peak
    max_dd = dd.min()
    calmar = ann_ret / abs(max_dd) if max_dd != 0 else 0

    total_ret = cum.iloc[-1] - 1

    # Profit factor
    gross_profit = r[r > 0].sum()
    gross_loss = abs(r[r < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')

    wr = (r > 0).mean()

    final_equity = INITIAL_CAPITAL * (1 + total_ret)

    return {
        'name': name,
        'CAGR': ann_ret,
        'Sharpe': sharpe,
        'Sortino': sortino,
        'MaxDD': max_dd,
        'Calmar': calmar,
        'WinRate': wr,
        'ProfitFactor': pf,
        'AnnVol': ann_vol,
        'TotalReturn': total_ret,
        'FinalEquity': final_equity,
    }

strategies = ['ml_vol_target', 'simple_vol_target', 'vix_stepped', 'buy_hold_spy', 'buy_hold_upro']
results = {}
for s in strategies:
    m = compute_metrics(strategy_returns[s], s)
    if m:
        results[s] = m

print(f"\n{'Strategy':<22} {'CAGR':>8} {'Sharpe':>8} {'Sortino':>8} {'MaxDD':>8} {'Calmar':>8} {'WR':>6} {'PF':>6} {'Vol':>8}")
print("-" * 100)
for s in strategies:
    if s in results:
        r = results[s]
        print(f"{r['name']:<22} {r['CAGR']:>7.1%} {r['Sharpe']:>8.3f} {r['Sortino']:>8.3f} {r['MaxDD']:>7.1%} {r['Calmar']:>8.3f} {r['WinRate']:>5.1%} {r['ProfitFactor']:>5.2f} {r['AnnVol']:>7.1%}")

# ─────────────────────────────────────────────
# Step 7: Adversarial Validation (HC #705)
# ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("ADVERSARIAL VALIDATION — HC #705")
print("=" * 60)

best_strat = 'ml_vol_target'
best_ret = strategy_returns[best_strat].dropna()

# Test 1: Permutation test (shuffle predictions → random allocation)
print("\n--- Test 1: Permutation Test (100 shuffles) ---")
actual_sharpe = results[best_strat]['Sharpe']
perm_sharpes = []
for p in range(100):
    np.random.seed(p)
    shuffled_weights = ml_weights_upro.sample(frac=1.0).values
    perm_ret = shuffled_weights * upro_aligned.values + (1 - shuffled_weights) * shy_aligned.values
    perm_ret = pd.Series(perm_ret, index=pred_dates).dropna()
    perm_ann_ret = (1 + perm_ret).prod() ** (252/len(perm_ret)) - 1
    perm_ann_vol = perm_ret.std() * np.sqrt(252)
    perm_sharpes.append(perm_ann_ret / perm_ann_vol if perm_ann_vol > 0 else 0)

perm_p = (np.array(perm_sharpes) >= actual_sharpe).mean()
print(f"  Actual Sharpe: {actual_sharpe:.4f}")
print(f"  Permuted Sharpe (mean): {np.mean(perm_sharpes):.4f}")
print(f"  p-value: {perm_p:.4f}")
print(f"  {'PASS' if perm_p < 0.05 else 'FAIL'}: Strategy {'IS' if perm_p < 0.05 else 'NOT'} significantly better than random")

# Test 2: Sub-period consistency
print("\n--- Test 2: Sub-Period Consistency ---")
n = len(best_ret)
quarter = n // 4
period_sharpes = []
for i in range(4):
    start = i * quarter
    end = (i + 1) * quarter if i < 3 else n
    sub = best_ret.iloc[start:end]
    sub_ann = (1 + sub).prod() ** (252/len(sub)) - 1
    sub_vol = sub.std() * np.sqrt(252)
    s = sub_ann / sub_vol if sub_vol > 0 else 0
    sub_cum = (1 + sub).cumprod()
    sub_dd = ((sub_cum - sub_cum.cummax()) / sub_cum.cummax()).min()
    period_sharpes.append(s)
    dates = f"{sub.index[0].strftime('%Y-%m')} to {sub.index[-1].strftime('%Y-%m')}"
    print(f"  Period {i+1} ({dates}): Sharpe={s:.3f}, MaxDD={sub_dd:.1%}")

pos_periods = sum(1 for s in period_sharpes if s > 0)
print(f"  Positive Sharpe periods: {pos_periods}/4")
sub_pass = pos_periods >= 3
print(f"  {'PASS' if sub_pass else 'FAIL'}: Sharpe positive in {'>=' if sub_pass else '<'} 3/4 sub-periods")

# Test 3: Outlier removal
print("\n--- Test 3: Outlier Removal (drop top/bottom 1% of days) ---")
trimmed = best_ret[(best_ret > best_ret.quantile(0.01)) & (best_ret < best_ret.quantile(0.99))]
trim_ann = (1 + trimmed).prod() ** (252/len(trimmed)) - 1
trim_vol = trimmed.std() * np.sqrt(252)
trim_sharpe = trim_ann / trim_vol if trim_vol > 0 else 0
outlier_ratio = trim_sharpe / actual_sharpe if actual_sharpe != 0 else 0
print(f"  Full Sharpe: {actual_sharpe:.3f}")
print(f"  Trimmed Sharpe (1%-99%): {trim_sharpe:.3f}")
print(f"  {'PASS' if outlier_ratio > 0.5 else 'FAIL'}: Trimmed/Full ratio = {outlier_ratio:.2f}")

# Test 4: R1 regime check (HC #709 nuance: growth strategies get relaxed 0.80 threshold)
print("\n--- Test 4: R1 Regime Test ---")
spy_daily = spy_fwd_1d.reindex(best_ret.index)
bull_mask = spy_daily > 0
bear_mask = spy_daily <= 0

bull_ret = best_ret[bull_mask]
bear_ret = best_ret[bear_mask]

bull_sharpe = ((1+bull_ret).prod()**(252/len(bull_ret))-1) / (bull_ret.std()*np.sqrt(252)) if len(bull_ret) > 50 else 0
bear_sharpe = ((1+bear_ret).prod()**(252/len(bear_ret))-1) / (bear_ret.std()*np.sqrt(252)) if len(bear_ret) > 50 else 0

regime_gap = abs(bull_sharpe - bear_sharpe) / max(abs(bull_sharpe), abs(bear_sharpe), 0.01)
print(f"  Bull days: {bull_mask.sum()}, Sharpe: {bull_sharpe:.3f}")
print(f"  Bear days: {bear_mask.sum()}, Sharpe: {bear_sharpe:.3f}")
print(f"  Regime gap: {regime_gap:.3f} (threshold: 0.80 for growth)")
r1_pass = regime_gap < 0.80  # HC #709 relaxed threshold for growth
print(f"  {'PASS' if r1_pass else 'FAIL'}: Regime gap {'<' if r1_pass else '>'} 0.80")

# Test 5: Does ML beat simple?
print("\n--- Test 5: ML vs Simple Baselines ---")
ml_sharpe = results['ml_vol_target']['Sharpe']
simple_sharpe = results['simple_vol_target']['Sharpe']
vix_sharpe = results['vix_stepped']['Sharpe']
print(f"  ML Vol Target: Sharpe {ml_sharpe:.3f}")
print(f"  Simple RV20 Target: Sharpe {simple_sharpe:.3f}")
print(f"  VIX Stepped: Sharpe {vix_sharpe:.3f}")
ml_beats_simple = ml_sharpe > simple_sharpe
print(f"  {'PASS' if ml_beats_simple else 'FAIL'}: ML {'beats' if ml_beats_simple else 'loses to'} simple realized-vol targeting")

# Summary
gates_passed = sum([perm_p < 0.05, sub_pass, outlier_ratio > 0.5, r1_pass])
total_gates = 4
print(f"\n{'='*60}")
print(f"ADVERSARIAL SUMMARY: {gates_passed}/{total_gates} checks passed")
if ml_beats_simple:
    print(f"ML adds value over simple baselines: YES")
else:
    print(f"ML adds value over simple baselines: NO — simple rules may suffice")
print(f"{'='*60}")

# ─────────────────────────────────────────────
# Save results
# ─────────────────────────────────────────────
strategy_returns.to_csv(OUTPUT / "strategy_returns.csv")

summary = {
    'timestamp': datetime.now().isoformat(),
    'vol_prediction': {
        'r2': float(r2),
        'rmse': float(rmse),
        'correlation': float(corr),
        'n_folds': n_folds,
    },
    'strategies': {k: {kk: float(vv) if isinstance(vv, (np.floating, float)) else vv
                       for kk, vv in v.items()}
                  for k, v in results.items()},
    'adversarial': {
        'permutation_p': float(perm_p),
        'sub_period_positive': int(pos_periods),
        'outlier_ratio': float(outlier_ratio),
        'regime_gap': float(regime_gap),
        'gates_passed': gates_passed,
        'total_gates': total_gates,
        'ml_beats_simple': bool(ml_beats_simple),
    },
    'feature_importance': {},
}

# Get feature importance from last model
if hasattr(model, 'feature_importances_'):
    fi = pd.Series(model.feature_importances_, index=features.columns).sort_values(ascending=False)
    summary['feature_importance'] = {k: float(v) for k, v in fi.head(15).items()}
    print("\nTop 15 features (last fold):")
    for feat, imp in fi.head(15).items():
        print(f"  {feat}: {imp:.4f}")

with open(OUTPUT / "results.json", 'w') as f:
    json.dump(summary, f, indent=2, default=str)

print(f"\n{'='*70}")
print(f"COMPLETED in {(datetime.now() - datetime.now()).seconds / 60:.1f} minutes")
print(f"Results saved to: {OUTPUT}/")
print(f"{'='*70}")
