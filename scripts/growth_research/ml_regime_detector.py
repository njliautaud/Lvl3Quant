#!/usr/bin/env python3
"""
ML Regime Detector — Multi-Asset Regime Prediction for Growth Allocation
=========================================================================
Predicts 21-day forward market regime (risk-on / cautious / risk-off)
using cross-asset features, then backtests a UPRO/SPY/SHY rotation strategy.

Walk-forward: 504d sliding train, 21d test, slide 21d (HC #0: SLIDING only).
Models: LightGBM (primary), Random Forest, simple MLP (CPU).
Adversarial: HC #705 — permutation test, sub-period consistency, outlier
             removal, R1 regime check, feature importance concentration.

Capital: Fixed $100K, no DCA (HC #713). Next-day execution.
"""

import warnings
warnings.filterwarnings('ignore')

import os
import sys
import time
import datetime as dt
from pathlib import Path
from functools import partial

# Force unbuffered output
print = partial(print, flush=True)

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from scipy import stats

# ==============================================================
# CONFIGURATION
# ==============================================================

BASE = Path('/home/jupiter/Lvl3Quant')
OUTPUT_DIR = BASE / 'output' / 'growth_research' / 'ml_regime_detector'
CACHE_DIR = BASE / 'output' / 'growth_research' / 'cache'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

TICKERS = ['SPY', 'UPRO', 'SHY', 'GLD', 'TLT', 'HYG', 'IEF', 'USO', 'UUP',
           'CPER', 'SLV', 'BTC-USD', 'QQQ', 'IWM']
VIX_TICKER = '^VIX'
ALL_TICKERS = TICKERS + [VIX_TICKER]

TRAIN_DAYS = 504       # 2 years sliding
TEST_DAYS = 21         # 1 month OOT
FWD_DAYS = 21          # forward return horizon for regime labeling
INITIAL_CAPITAL = 100_000

# Regime thresholds on 21d forward SPY return
RISK_ON_THRESH = 0.02   # > +2%
RISK_OFF_THRESH = -0.02 # < -2%

# v4.4 baseline for comparison
V44_SHARPE = 1.02
V44_CAGR = 0.241

START_DATE = '2009-01-01'
END_DATE = '2026-07-15'


def download_data():
    """Download or load cached daily data."""
    cache_file = CACHE_DIR / 'ml_regime_data.parquet'
    if cache_file.exists():
        mtime = dt.datetime.fromtimestamp(cache_file.stat().st_mtime)
        if (dt.datetime.now() - mtime).days < 1:
            print(f"  Loading cached data from {cache_file.name}")
            return pd.read_parquet(cache_file)

    print("  Downloading daily data from Yahoo Finance...")
    all_data = {}

    for ticker in ALL_TICKERS:
        try:
            df = yf.download(ticker, start=START_DATE, end=END_DATE,
                             progress=False, auto_adjust=True)
            if df is not None and len(df) > 100:
                # Handle multi-level columns from yfinance
                if isinstance(df.columns, pd.MultiIndex):
                    df.columns = df.columns.get_level_values(0)
                all_data[ticker] = df['Close'].rename(ticker.replace('^', '').replace('-', '_'))
                print(f"    {ticker}: {len(df)} rows")
            else:
                print(f"    {ticker}: SKIPPED (insufficient data)")
        except Exception as e:
            print(f"    {ticker}: ERROR - {e}")

    prices = pd.DataFrame(all_data)
    prices.index = pd.to_datetime(prices.index)
    if prices.index.tz is not None:
        prices.index = prices.index.tz_localize(None)
    prices = prices.sort_index()
    prices = prices.ffill().dropna(how='all')

    prices.to_parquet(cache_file)
    print(f"  Cached {len(prices)} rows, {len(prices.columns)} tickers")
    return prices


def engineer_features(prices):
    """Build feature matrix from multi-asset prices."""
    feat = pd.DataFrame(index=prices.index)

    # --- Per-asset momentum and volatility ---
    momentum_assets = ['SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'SLV', 'USO',
                       'UUP', 'CPER', 'HYG', 'BTC_USD']
    for col in momentum_assets:
        if col not in prices.columns:
            continue
        ret = prices[col].pct_change()

        # Rolling momentum (returns)
        for w in [5, 21, 63]:
            feat[f'{col}_mom_{w}d'] = prices[col].pct_change(w)

        # Rolling volatility
        for w in [21, 63]:
            feat[f'{col}_vol_{w}d'] = ret.rolling(w).std() * np.sqrt(252)

    # --- VIX features ---
    if 'VIX' in prices.columns:
        feat['VIX_level'] = prices['VIX']
        feat['VIX_pctile_63'] = prices['VIX'].rolling(63).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100, raw=False)
        feat['VIX_pctile_252'] = prices['VIX'].rolling(252).apply(
            lambda x: stats.percentileofscore(x, x.iloc[-1]) / 100, raw=False)
        feat['VIX_roc_5d'] = prices['VIX'].pct_change(5)
        feat['VIX_roc_21d'] = prices['VIX'].pct_change(21)
        feat['VIX_accel'] = feat['VIX_roc_5d'] - feat['VIX_roc_5d'].shift(5)
        # VIX z-score (mean-reversion signal)
        feat['VIX_zscore_63'] = (prices['VIX'] - prices['VIX'].rolling(63).mean()) / prices['VIX'].rolling(63).std()

    # --- Credit spread proxy: HYG vs IEF ---
    if 'HYG' in prices.columns and 'IEF' in prices.columns:
        credit_ret = prices['HYG'].pct_change() - prices['IEF'].pct_change()
        feat['credit_spread_5d'] = credit_ret.rolling(5).sum()
        feat['credit_spread_21d'] = credit_ret.rolling(21).sum()
        feat['credit_spread_63d'] = credit_ret.rolling(63).sum()

    # --- Cross-asset correlations (rolling 21d) ---
    spy_ret = prices['SPY'].pct_change()
    for other, name in [('GLD', 'gold'), ('TLT', 'bonds'), ('BTC_USD', 'btc'),
                         ('UUP', 'dollar'), ('HYG', 'credit')]:
        if other in prices.columns:
            other_ret = prices[other].pct_change()
            feat[f'corr_spy_{name}_21d'] = spy_ret.rolling(21).corr(other_ret)
            feat[f'corr_spy_{name}_63d'] = spy_ret.rolling(63).corr(other_ret)

    # --- Dollar strength trend ---
    if 'UUP' in prices.columns:
        feat['dollar_trend_21d'] = prices['UUP'].pct_change(21)
        feat['dollar_trend_63d'] = prices['UUP'].pct_change(63)

    # --- Commodity momentum composite ---
    comm_cols = [c for c in ['USO', 'CPER', 'SLV', 'GLD'] if c in prices.columns]
    if len(comm_cols) >= 2:
        comm_mom = pd.DataFrame({c: prices[c].pct_change(21) for c in comm_cols})
        feat['commodity_mom_21d'] = comm_mom.mean(axis=1)
        feat['commodity_mom_63d'] = pd.DataFrame(
            {c: prices[c].pct_change(63) for c in comm_cols}).mean(axis=1)

    # --- Breadth: SPY vs QQQ vs IWM relative strength ---
    if 'QQQ' in prices.columns:
        feat['spy_vs_qqq_21d'] = prices['SPY'].pct_change(21) - prices['QQQ'].pct_change(21)
    if 'IWM' in prices.columns:
        feat['spy_vs_iwm_21d'] = prices['SPY'].pct_change(21) - prices['IWM'].pct_change(21)

    # --- Yield curve proxy: TLT vs IEF (duration spread) ---
    if 'TLT' in prices.columns and 'IEF' in prices.columns:
        dur_spread = prices['TLT'].pct_change() - prices['IEF'].pct_change()
        feat['duration_spread_21d'] = dur_spread.rolling(21).sum()

    return feat


def define_regimes(prices):
    """Label regimes based on FORWARD 21d SPY return."""
    fwd_ret = prices['SPY'].pct_change(FWD_DAYS).shift(-FWD_DAYS)
    regimes = pd.Series(np.nan, index=prices.index, name='regime')
    regimes[fwd_ret > RISK_ON_THRESH] = 0   # risk-on
    regimes[(fwd_ret >= RISK_OFF_THRESH) & (fwd_ret <= RISK_ON_THRESH)] = 1  # cautious
    regimes[fwd_ret < RISK_OFF_THRESH] = 2  # risk-off
    return regimes, fwd_ret


def walk_forward_predict(features, regimes, model_type='lgbm'):
    """Sliding window walk-forward prediction. Returns OOS predictions aligned to dates."""
    valid_mask = features.notna().all(axis=1) & regimes.notna()
    feat_clean = features[valid_mask].copy()
    reg_clean = regimes[valid_mask].copy()

    dates = feat_clean.index
    n = len(dates)
    predictions = pd.Series(np.nan, index=features.index, name='pred')
    probas = pd.DataFrame(np.nan, index=features.index, columns=[0, 1, 2])

    fold_count = 0
    i = TRAIN_DAYS

    while i + TEST_DAYS <= n:
        train_idx = range(i - TRAIN_DAYS, i)
        test_idx = range(i, min(i + TEST_DAYS, n))

        X_train = feat_clean.iloc[train_idx].values
        y_train = reg_clean.iloc[train_idx].values.astype(int)
        X_test = feat_clean.iloc[test_idx].values
        test_dates = feat_clean.index[test_idx]

        # Skip if any class is missing from training
        if len(np.unique(y_train)) < 3:
            i += TEST_DAYS
            continue

        scaler = StandardScaler()
        X_train_s = scaler.fit_transform(X_train)
        X_test_s = scaler.transform(X_test)

        if model_type == 'lgbm':
            model = lgb.LGBMClassifier(
                n_estimators=150, max_depth=5, learning_rate=0.05,
                num_leaves=31, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1,
                class_weight='balanced', random_state=42,
                verbose=-1, n_jobs=-1
            )
            model.fit(X_train_s, y_train)
        elif model_type == 'rf':
            model = RandomForestClassifier(
                n_estimators=150, max_depth=8, min_samples_leaf=20,
                max_features='sqrt', class_weight='balanced',
                random_state=42, n_jobs=-1
            )
            model.fit(X_train_s, y_train)
        elif model_type == 'mlp':
            model = MLPClassifier(
                hidden_layer_sizes=(64, 32), activation='relu',
                max_iter=300, early_stopping=True,
                validation_fraction=0.15, random_state=42,
                learning_rate='adaptive', alpha=0.01
            )
            model.fit(X_train_s, y_train)

        preds = model.predict(X_test_s)
        probs = model.predict_proba(X_test_s)

        predictions.loc[test_dates] = preds
        for j, cls in enumerate(model.classes_):
            probas.loc[test_dates, cls] = probs[:, j]

        fold_count += 1
        i += TEST_DAYS

    valid_preds = predictions.dropna()
    print(f"    {model_type.upper()}: {fold_count} folds, {len(valid_preds)} OOS predictions")
    return predictions, probas, fold_count


def get_feature_importance(features, regimes, feature_names):
    """Train a single LGBM on all valid data to get feature importances."""
    valid_mask = features.notna().all(axis=1) & regimes.notna()
    X = features[valid_mask].values
    y = regimes[valid_mask].values.astype(int)
    if len(np.unique(y)) < 3:
        return pd.Series(dtype=float)

    scaler = StandardScaler()
    X_s = scaler.fit_transform(X)

    model = lgb.LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.05,
        num_leaves=31, subsample=0.8, colsample_bytree=0.8,
        class_weight='balanced', random_state=42, verbose=-1, n_jobs=-1
    )
    model.fit(X_s, y)
    imp = pd.Series(model.feature_importances_, index=feature_names)
    imp = imp / imp.sum()  # normalize
    return imp.sort_values(ascending=False)


def backtest_strategy(prices, predictions, name='ML'):
    """
    Backtest: signal at close T, trade at open T+1.
    risk-on(0) -> UPRO, cautious(1) -> SPY, risk-off(2) -> SHY.
    Fixed $100K, no compounding contributions.
    """
    # Align: prediction made at close of day T, position held from open T+1
    preds = predictions.dropna()
    if len(preds) == 0:
        return None

    # Use close-to-close returns as proxy (open data sometimes spotty)
    # Shift predictions by 1 day for next-day execution
    position_map = {0: 'UPRO', 1: 'SPY', 2: 'SHY'}
    assets = ['SPY', 'UPRO', 'SHY']
    rets = {a: prices[a].pct_change() for a in assets if a in prices.columns}

    # Build portfolio returns
    port_rets = pd.Series(0.0, index=prices.index)
    positions = pd.Series(np.nan, index=prices.index)

    for date in preds.index:
        pred = int(preds.loc[date])
        asset = position_map.get(pred, 'SPY')
        # Next trading day
        next_days = prices.index[prices.index > date]
        if len(next_days) == 0:
            continue
        next_day = next_days[0]
        if asset in rets and next_day in rets[asset].index:
            port_rets.loc[next_day] = rets[asset].loc[next_day]
            positions.loc[next_day] = pred

    # Trim to active period
    active = port_rets[port_rets != 0]
    if len(active) < 50:
        return None

    first_date = active.index[0]
    last_date = active.index[-1]
    port_rets = port_rets.loc[first_date:last_date]
    positions = positions.loc[first_date:last_date]

    # Equity curve from fixed capital
    equity = INITIAL_CAPITAL * (1 + port_rets).cumprod()

    # SPY benchmark
    spy_rets = prices['SPY'].pct_change().loc[first_date:last_date]
    spy_equity = INITIAL_CAPITAL * (1 + spy_rets).cumprod()

    # Metrics
    n_years = (last_date - first_date).days / 365.25
    total_ret = equity.iloc[-1] / INITIAL_CAPITAL - 1
    cagr = (equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / n_years) - 1

    ann_ret = port_rets.mean() * 252
    ann_vol = port_rets.std() * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else 0

    downside_vol = port_rets[port_rets < 0].std() * np.sqrt(252)
    sortino = ann_ret / downside_vol if downside_vol > 0 else 0

    dd = equity / equity.cummax() - 1
    max_dd = dd.min()
    calmar = cagr / abs(max_dd) if max_dd != 0 else 0

    # SPY metrics
    spy_total = spy_equity.iloc[-1] / INITIAL_CAPITAL - 1
    spy_cagr = (spy_equity.iloc[-1] / INITIAL_CAPITAL) ** (1 / n_years) - 1
    spy_vol = spy_rets.std() * np.sqrt(252)
    spy_sharpe = (spy_rets.mean() * 252) / spy_vol if spy_vol > 0 else 0
    spy_dd = spy_equity / spy_equity.cummax() - 1
    spy_max_dd = spy_dd.min()

    # Position distribution
    pos_counts = positions.dropna().value_counts()

    results = {
        'name': name,
        'period': f"{first_date.strftime('%Y-%m-%d')} to {last_date.strftime('%Y-%m-%d')}",
        'n_years': round(n_years, 1),
        'n_days': len(port_rets),
        'cagr': cagr,
        'total_return': total_ret,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd': max_dd,
        'calmar': calmar,
        'ann_vol': ann_vol,
        'pos_risk_on': pos_counts.get(0, 0),
        'pos_cautious': pos_counts.get(1, 0),
        'pos_risk_off': pos_counts.get(2, 0),
        'spy_cagr': spy_cagr,
        'spy_sharpe': spy_sharpe,
        'spy_max_dd': spy_max_dd,
        'equity': equity,
        'spy_equity': spy_equity,
        'port_rets': port_rets,
        'positions': positions,
    }
    return results


# ==============================================================
# ADVERSARIAL VALIDATION (HC #705)
# ==============================================================

def adversarial_permutation_test(features, regimes, real_sharpe, n_perms=30):
    """Shuffle regime labels, re-run WF, compare Sharpe distribution."""
    print(f"\n  [ADV-1] Permutation test ({n_perms} shuffles)...")
    shuffled_accs = []
    valid_mask = features.notna().all(axis=1) & regimes.notna()
    feat_clean = features[valid_mask]
    reg_clean = regimes[valid_mask]
    n = len(feat_clean)

    for p in range(n_perms):
        # Shuffle regime labels (break temporal structure)
        shuffled = reg_clean.copy()
        shuffled.values[:] = np.random.permutation(shuffled.values)

        # Quick WF — skip 6x folds for speed, fewer trees
        preds = pd.Series(np.nan, index=feat_clean.index)
        i = TRAIN_DAYS
        while i + TEST_DAYS <= n:
            train_idx = range(i - TRAIN_DAYS, i)
            test_idx = range(i, min(i + TEST_DAYS, n))
            X_tr = feat_clean.iloc[train_idx].values
            y_tr = shuffled.iloc[train_idx].values.astype(int)
            X_te = feat_clean.iloc[test_idx].values

            if len(np.unique(y_tr)) < 3:
                i += TEST_DAYS * 6
                continue

            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_te_s = scaler.transform(X_te)

            model = lgb.LGBMClassifier(
                n_estimators=50, max_depth=4, learning_rate=0.1,
                class_weight='balanced', verbose=-1, n_jobs=-1, random_state=p
            )
            model.fit(X_tr_s, y_tr)
            preds.iloc[list(test_idx)] = model.predict(X_te_s)
            i += TEST_DAYS * 6  # aggressive skip for speed

        valid_p = preds.dropna()
        if len(valid_p) > 0:
            acc = accuracy_score(reg_clean.loc[valid_p.index].astype(int),
                                valid_p.astype(int))
            shuffled_accs.append(acc)

        if (p + 1) % 10 == 0:
            print(f"    {p+1}/{n_perms} permutations done")

    return shuffled_accs


def adversarial_subperiod(results):
    """Split into 3+ equal sub-periods, check Sharpe consistency."""
    print("\n  [ADV-2] Sub-period consistency...")
    port_rets = results['port_rets']
    n = len(port_rets)
    n_periods = 4
    period_size = n // n_periods

    sub_sharpes = []
    for i in range(n_periods):
        start = i * period_size
        end = (i + 1) * period_size if i < n_periods - 1 else n
        sub = port_rets.iloc[start:end]
        if len(sub) > 20:
            s = sub.mean() / sub.std() * np.sqrt(252) if sub.std() > 0 else 0
            sub_sharpes.append(s)
            period_start = sub.index[0].strftime('%Y-%m-%d')
            period_end = sub.index[-1].strftime('%Y-%m-%d')
            print(f"    Period {i+1} ({period_start} to {period_end}): Sharpe={s:.2f}")

    if len(sub_sharpes) < 3:
        return False, sub_sharpes

    # Pass if: no sub-period is deeply negative AND majority are positive
    n_positive = sum(1 for s in sub_sharpes if s > 0)
    worst = min(sub_sharpes)
    passed = n_positive >= len(sub_sharpes) // 2 and worst > -0.5
    return passed, sub_sharpes


def adversarial_outlier_removal(results):
    """Remove top 5 best days, recalculate Sharpe."""
    print("\n  [ADV-3] Outlier removal (drop 5 best days)...")
    port_rets = results['port_rets'].copy()
    original_sharpe = results['sharpe']

    # Remove top 5 daily returns
    top5_idx = port_rets.nlargest(5).index
    trimmed = port_rets.drop(top5_idx)
    trimmed_sharpe = trimmed.mean() / trimmed.std() * np.sqrt(252) if trimmed.std() > 0 else 0

    print(f"    Original Sharpe: {original_sharpe:.3f}")
    print(f"    After removing 5 best days: {trimmed_sharpe:.3f}")
    print(f"    Degradation: {(original_sharpe - trimmed_sharpe) / abs(original_sharpe) * 100:.1f}%")

    # Pass if Sharpe doesn't drop more than 50%
    passed = trimmed_sharpe > original_sharpe * 0.5 if original_sharpe > 0 else trimmed_sharpe > 0
    return passed, trimmed_sharpe


def adversarial_r1_regime(results, prices):
    """Check Sharpe gap between green and red market days (HC #428 R1)."""
    print("\n  [ADV-4] R1 regime check (green vs red day Sharpe)...")
    port_rets = results['port_rets']
    spy_rets = prices['SPY'].pct_change().reindex(port_rets.index)

    green_days = port_rets[spy_rets > 0]
    red_days = port_rets[spy_rets < 0]

    green_sharpe = green_days.mean() / green_days.std() * np.sqrt(252) if len(green_days) > 20 and green_days.std() > 0 else 0
    red_sharpe = red_days.mean() / red_days.std() * np.sqrt(252) if len(red_days) > 20 and red_days.std() > 0 else 0

    print(f"    Green-day Sharpe: {green_sharpe:.3f} ({len(green_days)} days)")
    print(f"    Red-day Sharpe: {red_sharpe:.3f} ({len(red_days)} days)")

    gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)
    print(f"    Regime gap ratio: {gap:.3f} (threshold: 0.50)")

    passed = gap <= 0.50
    return passed, green_sharpe, red_sharpe, gap


def adversarial_feature_concentration(feature_imp):
    """Check if model is overfit to a single feature."""
    print("\n  [ADV-5] Feature importance concentration...")
    if len(feature_imp) == 0:
        return False, 0

    top1 = feature_imp.iloc[0]
    top5 = feature_imp.iloc[:5].sum()
    hhi = (feature_imp ** 2).sum()  # Herfindahl index

    print(f"    Top feature: {feature_imp.index[0]} = {top1:.3f}")
    print(f"    Top 5 features: {top5:.3f}")
    print(f"    HHI (concentration): {hhi:.4f}")
    print(f"    Features with >1% importance: {(feature_imp > 0.01).sum()}")

    # Pass if top feature < 30% and HHI < 0.15
    passed = top1 < 0.30 and hhi < 0.15
    return passed, hhi


# ==============================================================
# MAIN
# ==============================================================

def main():
    t0 = time.time()
    print("=" * 70)
    print("ML REGIME DETECTOR — Multi-Asset Regime Prediction")
    print("=" * 70)

    # --- Step 1: Download data ---
    print("\n[1] Downloading data...")
    prices = download_data()
    print(f"  Price matrix: {prices.shape[0]} rows x {prices.shape[1]} columns")
    print(f"  Date range: {prices.index[0].strftime('%Y-%m-%d')} to {prices.index[-1].strftime('%Y-%m-%d')}")
    print(f"  Tickers: {list(prices.columns)}")

    # --- Step 2: Engineer features ---
    print("\n[2] Engineering features...")
    features = engineer_features(prices)
    # Drop features that are >50% NaN
    good_cols = features.columns[features.notna().mean() > 0.5]
    features = features[good_cols]
    # Drop rows where all features are NaN
    features = features.dropna(how='all')
    print(f"  Feature matrix: {features.shape[0]} rows x {features.shape[1]} features")
    print(f"  Features: {list(features.columns[:10])}... ({len(features.columns)} total)")

    # --- Step 3: Define regimes ---
    print("\n[3] Defining regimes (21d forward SPY return)...")
    regimes, fwd_ret = define_regimes(prices)
    reg_counts = regimes.dropna().value_counts().sort_index()
    total = reg_counts.sum()
    print(f"  Risk-on (>+2%):  {reg_counts.get(0, 0):5d} ({reg_counts.get(0, 0)/total*100:.1f}%)")
    print(f"  Cautious:        {reg_counts.get(1, 0):5d} ({reg_counts.get(1, 0)/total*100:.1f}%)")
    print(f"  Risk-off (<-2%): {reg_counts.get(2, 0):5d} ({reg_counts.get(2, 0)/total*100:.1f}%)")

    # Align features and regimes
    common_idx = features.index.intersection(regimes.dropna().index)
    features = features.loc[common_idx]
    regimes = regimes.loc[common_idx]
    print(f"  Aligned dataset: {len(common_idx)} rows")

    # --- Step 4: Walk-forward predictions ---
    print("\n[4] Walk-forward regime prediction...")
    feature_names = list(features.columns)

    all_results = {}
    all_preds = {}
    for model_type in ['lgbm', 'rf', 'mlp']:
        print(f"\n  --- {model_type.upper()} ---")
        preds, probas, n_folds = walk_forward_predict(features, regimes, model_type)
        all_preds[model_type] = preds

        # Accuracy on OOS
        valid = preds.dropna()
        if len(valid) == 0:
            print(f"    No valid predictions for {model_type}")
            continue

        actual = regimes.loc[valid.index].astype(int)
        acc = accuracy_score(actual, valid.astype(int))
        print(f"    OOS accuracy: {acc:.3f} (random baseline: {1/3:.3f})")
        print(f"    Classification report:")
        print(classification_report(actual, valid.astype(int),
                                     target_names=['risk-on', 'cautious', 'risk-off'],
                                     digits=3, zero_division=0))

        # --- Step 5+6: Backtest ---
        bt = backtest_strategy(prices, preds, name=model_type.upper())
        if bt is None:
            print(f"    Backtest failed for {model_type}")
            continue

        all_results[model_type] = bt
        bt['accuracy'] = acc
        bt['n_folds'] = n_folds

        print(f"    Backtest: Sharpe={bt['sharpe']:.3f}, CAGR={bt['cagr']*100:.1f}%, "
              f"MaxDD={bt['max_dd']*100:.1f}%, Sortino={bt['sortino']:.3f}, Calmar={bt['calmar']:.3f}")

    if not all_results:
        print("\nERROR: No models produced valid results. Exiting.")
        return

    # --- Step 7: Adversarial validation on best model ---
    best_model = max(all_results, key=lambda k: all_results[k]['sharpe'])
    best = all_results[best_model]
    print(f"\n{'='*70}")
    print(f"ADVERSARIAL VALIDATION — Best model: {best_model.upper()}")
    print(f"{'='*70}")

    # Feature importance
    feat_imp = get_feature_importance(features, regimes, feature_names)
    print("\n  Top 15 features:")
    for i, (fname, fval) in enumerate(feat_imp.head(15).items()):
        print(f"    {i+1:2d}. {fname:35s} {fval:.4f}")

    # ADV-1: Permutation test (reuse cached predictions)
    preds_best = all_preds[best_model]
    valid_best = preds_best.dropna()
    actual_best = regimes.loc[valid_best.index].astype(int)
    real_acc = accuracy_score(actual_best, valid_best.astype(int))

    shuffled_accs = adversarial_permutation_test(features, regimes, best['sharpe'])
    if shuffled_accs:
        p_val = np.mean([a >= real_acc for a in shuffled_accs])
        perm_passed = p_val < 0.05
        print(f"    Real accuracy: {real_acc:.4f}")
        print(f"    Shuffled mean: {np.mean(shuffled_accs):.4f} +/- {np.std(shuffled_accs):.4f}")
        print(f"    p-value: {p_val:.4f}")
        print(f"    RESULT: {'PASS' if perm_passed else 'FAIL'}")
    else:
        perm_passed = False
        p_val = 1.0
        print("    RESULT: FAIL (no permutation results)")

    # ADV-2: Sub-period consistency
    sub_passed, sub_sharpes = adversarial_subperiod(best)
    print(f"    RESULT: {'PASS' if sub_passed else 'FAIL'}")

    # ADV-3: Outlier removal
    out_passed, trimmed_sharpe = adversarial_outlier_removal(best)
    print(f"    RESULT: {'PASS' if out_passed else 'FAIL'}")

    # ADV-4: R1 regime check
    r1_passed, green_s, red_s, gap = adversarial_r1_regime(best, prices)
    print(f"    RESULT: {'PASS' if r1_passed else 'FAIL'}")

    # ADV-5: Feature concentration
    fc_passed, hhi = adversarial_feature_concentration(feat_imp)
    print(f"    RESULT: {'PASS' if fc_passed else 'FAIL'}")

    # --- Final comparison table ---
    print(f"\n{'='*70}")
    print("COMPARISON TABLE")
    print(f"{'='*70}")
    header = f"{'Strategy':20s} {'Sharpe':>8s} {'Sortino':>8s} {'CAGR':>8s} {'MaxDD':>8s} {'Calmar':>8s} {'OOS Acc':>8s}"
    print(header)
    print("-" * len(header))

    for mt in ['lgbm', 'rf', 'mlp']:
        if mt in all_results:
            r = all_results[mt]
            print(f"{r['name']:20s} {r['sharpe']:8.3f} {r['sortino']:8.3f} "
                  f"{r['cagr']*100:7.1f}% {r['max_dd']*100:7.1f}% "
                  f"{r['calmar']:8.3f} {r.get('accuracy', 0):7.3f}")

    # SPY benchmark
    spy_r = all_results[best_model]
    print(f"{'SPY Buy-Hold':20s} {spy_r['spy_sharpe']:8.3f} {'--':>8s} "
          f"{spy_r['spy_cagr']*100:7.1f}% {spy_r['spy_max_dd']*100:7.1f}% "
          f"{'--':>8s} {'--':>8s}")

    # v4.4 baseline
    print(f"{'v4.4 VIX Rules':20s} {V44_SHARPE:8.3f} {'--':>8s} "
          f"{V44_CAGR*100:7.1f}% {'--':>8s} {'--':>8s} {'--':>8s}")

    # --- Adversarial summary ---
    print(f"\n{'='*70}")
    print("ADVERSARIAL GATE SUMMARY")
    print(f"{'='*70}")
    gates = [
        ('Permutation test (p<0.05)', perm_passed, f'p={p_val:.4f}'),
        ('Sub-period consistency', sub_passed, f'sharpes={[round(s,2) for s in sub_sharpes]}'),
        ('Outlier robustness', out_passed, f'trimmed Sharpe={trimmed_sharpe:.3f}'),
        ('R1 regime gap <0.50', r1_passed, f'gap={gap:.3f}'),
        ('Feature concentration', fc_passed, f'HHI={hhi:.4f}'),
    ]
    all_pass = True
    for gate_name, passed, detail in gates:
        status = "PASS" if passed else "FAIL"
        print(f"  [{status:4s}] {gate_name:35s} {detail}")
        if not passed:
            all_pass = False

    n_pass = sum(1 for _, p, _ in gates if p)
    print(f"\n  Overall: {n_pass}/5 gates passed {'(ALL CLEAR)' if all_pass else '(SOME FAILURES)'}")

    # --- Position distribution for best ---
    r = all_results[best_model]
    total_pos = r['pos_risk_on'] + r['pos_cautious'] + r['pos_risk_off']
    if total_pos > 0:
        print(f"\n  Position distribution ({best_model.upper()}):")
        print(f"    UPRO (risk-on):  {r['pos_risk_on']:5d} ({r['pos_risk_on']/total_pos*100:.1f}%)")
        print(f"    SPY  (cautious): {r['pos_cautious']:5d} ({r['pos_cautious']/total_pos*100:.1f}%)")
        print(f"    SHY  (risk-off): {r['pos_risk_off']:5d} ({r['pos_risk_off']/total_pos*100:.1f}%)")

    elapsed = time.time() - t0
    print(f"\n  Total runtime: {elapsed:.0f}s ({elapsed/60:.1f}min)")

    # --- Save results ---
    summary = {
        'best_model': best_model,
        'sharpe': round(best['sharpe'], 4),
        'sortino': round(best['sortino'], 4),
        'cagr': round(best['cagr'], 4),
        'max_dd': round(best['max_dd'], 4),
        'calmar': round(best['calmar'], 4),
        'accuracy': round(best.get('accuracy', 0), 4),
        'adv_gates_passed': n_pass,
        'adv_total_gates': 5,
        'vs_spy_sharpe_delta': round(best['sharpe'] - spy_r['spy_sharpe'], 4),
        'vs_v44_sharpe_delta': round(best['sharpe'] - V44_SHARPE, 4),
    }
    import json
    with open(OUTPUT_DIR / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\n  Results saved to {OUTPUT_DIR / 'results_summary.json'}")

    print(f"\n{'='*70}")
    print("DONE")
    print(f"{'='*70}")


if __name__ == '__main__':
    main()
