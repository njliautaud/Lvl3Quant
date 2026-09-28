#!/usr/bin/env python3
"""
Sub-Sector Rotation ML — 6-Test Adversarial Validation
=======================================================
Tests the top 5 pairs from the initial 5-gate screen against 6 adversarial tests.
A pair PASSES overall only if it passes at least 5/6 tests.

Tests:
1. Re-implementation test (rebuild from scratch, compare Sharpe)
2. Inverse signal test (buy leader instead of laggard)
3. Random timing test (1000 permutations of entry dates)
4. Sub-period stability (4 equal periods, all must have Sharpe > 0)
5. Top-3 removal (remove 3 best trades, check Sharpe drop)
6. Parameter sensitivity (5+ param variations, 80% must have Sharpe > 0.30)
"""

import json
import warnings
import numpy as np
import pandas as pd
import yfinance as yf
from datetime import datetime
from pathlib import Path
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

# ============================================================================
# CONFIG
# ============================================================================
START_DATE = '2020-01-01'
END_DATE = '2026-07-31'
CONFIDENCE_THRESHOLD = 0.6

# Top 5 pairs to test (from initial results) with their best horizons
PAIRS_TO_TEST = [
    {'name': 'REITs_vs_BroadRE', 'etf_a': 'VNQ', 'etf_b': 'XLRE', 'horizon': 5,
     'original_sharpe': 3.627, 'original_wr': 0.7321},
    {'name': 'GoldMiners_vs_MetalsMining', 'etf_a': 'GDX', 'etf_b': 'XME', 'horizon': 5,
     'original_sharpe': 2.607, 'original_wr': 0.6075},
    {'name': 'RegBanks_vs_BroadFin', 'etf_a': 'KRE', 'etf_b': 'XLF', 'horizon': 10,
     'original_sharpe': 2.133, 'original_wr': 0.6881},
    {'name': 'ConsDisc_vs_ConsStaples', 'etf_a': 'XLY', 'etf_b': 'XLP', 'horizon': 5,
     'original_sharpe': 2.091, 'original_wr': 0.6125},
    {'name': 'Banks_vs_Insurance', 'etf_a': 'KBE', 'etf_b': 'KIE', 'horizon': 5,
     'original_sharpe': 2.012, 'original_wr': 0.6812},
]

# ============================================================================
# DATA
# ============================================================================
def download_data():
    tickers = set()
    for p in PAIRS_TO_TEST:
        tickers.add(p['etf_a'])
        tickers.add(p['etf_b'])
    tickers.add('SPY')
    tickers = sorted(tickers)
    print(f"Downloading {len(tickers)} tickers: {', '.join(tickers)}")
    data = yf.download(tickers, start=START_DATE, end=END_DATE, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        close = data['Close']
    else:
        close = data
    close = close.dropna(axis=1, thresh=300)
    print(f"Data: {close.shape[0]} days, {close.index[0].date()} to {close.index[-1].date()}")
    return close


# ============================================================================
# ORIGINAL IMPLEMENTATION (from v1 script — exact replica)
# ============================================================================
def compute_features_v1(close_a, close_b, spy_close):
    """Original feature engineering from v1."""
    ret_a = close_a.pct_change()
    ret_b = close_b.pct_change()
    ratio = close_a / close_b
    ratio_ret = ratio.pct_change()

    features = pd.DataFrame(index=close_a.index)

    for w in [5, 10, 21, 63]:
        features[f'rel_ret_{w}d'] = (close_a / close_a.shift(w)) / (close_b / close_b.shift(w)) - 1

    delta = ratio.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    features['ratio_rsi_14'] = 100 - (100 / (1 + rs))

    for w in [21, 63]:
        roll_mean = ratio_ret.rolling(w).mean()
        roll_std = ratio_ret.rolling(w).std()
        features[f'rel_zscore_{w}d'] = (ratio_ret - roll_mean) / roll_std.replace(0, np.nan)

    for w in [21, 63]:
        features[f'corr_{w}d'] = ret_a.rolling(w).corr(ret_b)

    features['corr_change_21d'] = features['corr_21d'] - features['corr_21d'].shift(21)

    vol_a = ret_a.rolling(21).std()
    vol_b = ret_b.rolling(21).std()
    features['vol_ratio_21d'] = vol_a / vol_b.replace(0, np.nan)

    mom_a = close_a / close_a.shift(21) - 1
    mom_b = close_b / close_b.shift(21) - 1
    features['mom_divergence_21d'] = mom_a - mom_b

    ratio_sma = ratio.rolling(63).mean()
    features['ratio_dist_sma63'] = (ratio / ratio_sma) - 1

    spy_sma200 = spy_close.rolling(200).mean()
    features['spy_regime'] = (spy_close > spy_sma200).astype(int)
    features['spy_ret_21d'] = spy_close.pct_change(21)

    return features


def compute_targets_v1(close_a, close_b, horizon):
    """Original target from v1."""
    fwd_ret_a = close_a.pct_change(horizon).shift(-horizon)
    fwd_ret_b = close_b.pct_change(horizon).shift(-horizon)
    past_rel = (close_a / close_a.shift(21)) / (close_b / close_b.shift(21)) - 1

    reversal = np.where(
        past_rel < 0,
        fwd_ret_a > fwd_ret_b,
        fwd_ret_b > fwd_ret_a
    ).astype(float)

    reversal_pnl = np.where(
        past_rel < 0,
        fwd_ret_a - fwd_ret_b,
        fwd_ret_b - fwd_ret_a
    )

    return (pd.Series(reversal, index=close_a.index),
            pd.Series(reversal_pnl, index=close_a.index))


def walk_forward_lgbm_v1(features, target, pnl, horizon, train_days=252, test_days=21,
                          lgbm_params=None, threshold=0.6):
    """Original walk-forward from v1."""
    combined = features.copy()
    combined['target'] = target
    combined['pnl'] = pnl
    combined = combined.dropna()

    if len(combined) < train_days + test_days + 50:
        return None

    feature_cols = [c for c in features.columns if c in combined.columns]

    if lgbm_params is None:
        lgbm_params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'num_leaves': 31, 'learning_rate': 0.05, 'feature_fraction': 0.8,
            'bagging_fraction': 0.8, 'bagging_freq': 5, 'min_child_samples': 20,
            'n_estimators': 200, 'random_state': 42,
        }

    results = []
    idx = 0
    while idx + train_days + test_days <= len(combined):
        train_end = idx + train_days
        test_end = min(train_end + test_days, len(combined))

        X_train = combined.iloc[idx:train_end][feature_cols].values
        y_train = combined.iloc[idx:train_end]['target'].values
        X_test = combined.iloc[train_end:test_end][feature_cols].values
        y_test = combined.iloc[train_end:test_end]['target'].values
        pnl_test = combined.iloc[train_end:test_end]['pnl'].values
        test_dates = combined.iloc[train_end:test_end].index

        if len(np.unique(y_train)) < 2 or len(y_test) == 0:
            idx += test_days
            continue

        model = lgb.LGBMClassifier(**lgbm_params)
        model.fit(X_train, y_train, eval_set=[(X_test, y_test)],
                  callbacks=[lgb.early_stopping(30, verbose=False), lgb.log_evaluation(0)])

        proba = model.predict_proba(X_test)[:, 1]

        for i in range(len(y_test)):
            results.append({
                'date': test_dates[i],
                'y_true': y_test[i],
                'proba': proba[i],
                'pnl': pnl_test[i],
            })

        idx += test_days

    if not results:
        return None

    return pd.DataFrame(results)


def calc_sharpe(returns, horizon):
    """Annualized Sharpe from trade returns."""
    if len(returns) < 2:
        return 0.0
    m = returns.mean()
    s = returns.std()
    if s == 0 or np.isnan(s):
        return 0.0
    trades_per_year = 252 / horizon
    return float((m / s) * np.sqrt(trades_per_year))


def calc_win_rate(returns):
    if len(returns) == 0:
        return 0.0
    return float((returns > 0).mean())


def get_strategy_returns(results_df, threshold=0.6):
    """Filter to high-confidence trades and return their PnL."""
    high_conf = results_df[results_df['proba'] >= threshold]
    if len(high_conf) == 0:
        return np.array([])
    return high_conf['pnl'].values


# ============================================================================
# RE-IMPLEMENTATION (Test 1): Build from scratch with different code
# ============================================================================
def compute_features_v2(close_a, close_b, spy_close):
    """Independent re-implementation of features. Same logic, different code path."""
    df = pd.DataFrame(index=close_a.index)

    # Relative performance over windows
    for lookback in [5, 10, 21, 63]:
        perf_a = close_a / close_a.shift(lookback) - 1
        perf_b = close_b / close_b.shift(lookback) - 1
        df[f'rel_ret_{lookback}d'] = (1 + perf_a) / (1 + perf_b) - 1

    # RSI of the price ratio
    price_ratio = close_a / close_b
    ratio_changes = price_ratio.diff()
    ups = ratio_changes.clip(lower=0).rolling(window=14).mean()
    downs = (-ratio_changes.clip(upper=0)).rolling(window=14).mean()
    relative_strength = ups / downs.replace(0, np.nan)
    df['ratio_rsi_14'] = 100.0 - 100.0 / (1.0 + relative_strength)

    # Z-scores of ratio returns
    ratio_daily_ret = price_ratio.pct_change()
    for window in [21, 63]:
        mu = ratio_daily_ret.rolling(window).mean()
        sigma = ratio_daily_ret.rolling(window).std().replace(0, np.nan)
        df[f'rel_zscore_{window}d'] = (ratio_daily_ret - mu) / sigma

    # Rolling correlations
    daily_ret_a = close_a.pct_change()
    daily_ret_b = close_b.pct_change()
    for window in [21, 63]:
        df[f'corr_{window}d'] = daily_ret_a.rolling(window).corr(daily_ret_b)

    # Correlation change
    df['corr_change_21d'] = df['corr_21d'].diff(21)

    # Volatility ratio
    sig_a = daily_ret_a.rolling(21).std()
    sig_b = daily_ret_b.rolling(21).std().replace(0, np.nan)
    df['vol_ratio_21d'] = sig_a / sig_b

    # Momentum divergence
    df['mom_divergence_21d'] = (close_a / close_a.shift(21) - 1) - (close_b / close_b.shift(21) - 1)

    # Ratio distance from 63d moving average
    sma63 = price_ratio.rolling(63).mean()
    df['ratio_dist_sma63'] = price_ratio / sma63 - 1

    # SPY regime
    spy_ma200 = spy_close.rolling(200).mean()
    df['spy_regime'] = (spy_close > spy_ma200).astype(int)
    df['spy_ret_21d'] = spy_close.pct_change(21)

    return df


def compute_targets_v2(close_a, close_b, horizon):
    """Independent re-implementation of targets."""
    # Forward returns
    fwd_a = close_a.shift(-horizon) / close_a - 1
    fwd_b = close_b.shift(-horizon) / close_b - 1

    # Who lagged over past 21d?
    past_a = close_a / close_a.shift(21) - 1
    past_b = close_b / close_b.shift(21) - 1
    a_lagged = past_a < past_b  # A underperformed B

    # Mean reversion: laggard outperforms leader going forward
    target = np.where(a_lagged, fwd_a > fwd_b, fwd_b > fwd_a).astype(float)
    pnl = np.where(a_lagged, fwd_a - fwd_b, fwd_b - fwd_a)

    return (pd.Series(target, index=close_a.index),
            pd.Series(pnl, index=close_a.index))


# ============================================================================
# INVERSE SIGNAL (Test 2): Buy leader instead of laggard
# ============================================================================
def compute_targets_inverse(close_a, close_b, horizon):
    """Buy the LEADING sub-sector (anti-mean-reversion / momentum)."""
    fwd_ret_a = close_a.pct_change(horizon).shift(-horizon)
    fwd_ret_b = close_b.pct_change(horizon).shift(-horizon)
    past_rel = (close_a / close_a.shift(21)) / (close_b / close_b.shift(21)) - 1

    # INVERSE: bet on the leader continuing to lead (momentum)
    reversal_pnl = np.where(
        past_rel < 0,
        fwd_ret_b - fwd_ret_a,  # A lagged -> buy B (leader) instead
        fwd_ret_a - fwd_ret_b   # B lagged -> buy A (leader) instead
    )
    target = (reversal_pnl > 0).astype(float)

    return (pd.Series(target, index=close_a.index),
            pd.Series(reversal_pnl, index=close_a.index))


# ============================================================================
# TEST FUNCTIONS
# ============================================================================

def test_1_reimplementation(close_a, close_b, spy, horizon, original_sharpe):
    """Re-implementation test: rebuild strategy from scratch."""
    print("    Test 1: Re-implementation...")

    # Run original
    features_orig = compute_features_v1(close_a, close_b, spy)
    target_orig, pnl_orig = compute_targets_v1(close_a, close_b, horizon)
    results_orig = walk_forward_lgbm_v1(features_orig, target_orig, pnl_orig, horizon)

    if results_orig is None:
        return {'pass': False, 'reason': 'Original failed to produce results'}

    returns_orig = get_strategy_returns(results_orig)
    sharpe_orig = calc_sharpe(returns_orig, horizon)

    # Run re-implementation
    features_v2 = compute_features_v2(close_a, close_b, spy)
    target_v2, pnl_v2 = compute_targets_v2(close_a, close_b, horizon)
    results_v2 = walk_forward_lgbm_v1(features_v2, target_v2, pnl_v2, horizon)

    if results_v2 is None:
        return {'pass': False, 'reason': 'V2 failed to produce results'}

    returns_v2 = get_strategy_returns(results_v2)
    sharpe_v2 = calc_sharpe(returns_v2, horizon)

    # Check divergence
    if sharpe_orig == 0:
        divergence = abs(sharpe_v2)
    else:
        divergence = abs(sharpe_v2 - sharpe_orig) / abs(sharpe_orig)

    passed = divergence <= 0.30

    return {
        'pass': passed,
        'sharpe_original': round(sharpe_orig, 3),
        'sharpe_reimplemented': round(sharpe_v2, 3),
        'divergence_pct': round(divergence * 100, 1),
        'threshold': '30%',
    }


def test_2_inverse_signal(close_a, close_b, spy, horizon, original_sharpe):
    """Inverse signal test: buy leader instead of laggard."""
    print("    Test 2: Inverse signal...")

    features = compute_features_v1(close_a, close_b, spy)

    # Forward (original) signal
    target_fwd, pnl_fwd = compute_targets_v1(close_a, close_b, horizon)
    results_fwd = walk_forward_lgbm_v1(features, target_fwd, pnl_fwd, horizon)

    if results_fwd is None:
        return {'pass': False, 'reason': 'Forward signal failed'}

    returns_fwd = get_strategy_returns(results_fwd)
    sharpe_fwd = calc_sharpe(returns_fwd, horizon)

    # Inverse signal
    target_inv, pnl_inv = compute_targets_inverse(close_a, close_b, horizon)
    results_inv = walk_forward_lgbm_v1(features, target_inv, pnl_inv, horizon)

    if results_inv is None:
        return {'pass': True, 'reason': 'Inverse failed to produce results (good)',
                'sharpe_forward': round(sharpe_fwd, 3), 'sharpe_inverse': 0.0, 'ratio': 0.0}

    returns_inv = get_strategy_returns(results_inv)
    sharpe_inv = calc_sharpe(returns_inv, horizon)

    # FAIL if inverse is too good relative to forward
    if abs(sharpe_fwd) < 0.01:
        ratio = abs(sharpe_inv)
    else:
        ratio = abs(sharpe_inv) / abs(sharpe_fwd)

    passed = ratio <= 0.50

    return {
        'pass': passed,
        'sharpe_forward': round(sharpe_fwd, 3),
        'sharpe_inverse': round(sharpe_inv, 3),
        'ratio': round(ratio, 3),
        'threshold': '0.50',
    }


def test_3_random_timing(close_a, close_b, spy, horizon, original_sharpe, n_perms=1000):
    """Random timing test: randomize entry dates, keep same trade count."""
    print("    Test 3: Random timing (1000 permutations)...")

    features = compute_features_v1(close_a, close_b, spy)
    target, pnl = compute_targets_v1(close_a, close_b, horizon)
    results = walk_forward_lgbm_v1(features, target, pnl, horizon)

    if results is None:
        return {'pass': False, 'reason': 'Strategy failed to produce results'}

    returns = get_strategy_returns(results)
    if len(returns) == 0:
        return {'pass': False, 'reason': 'No high-confidence trades'}

    actual_sharpe = calc_sharpe(returns, horizon)
    n_trades = len(returns)

    # All available PnL values (not just high-confidence)
    all_pnl = results['pnl'].values

    # Permutation test: randomly sample n_trades from all available PnLs
    perm_sharpes = []
    for _ in range(n_perms):
        random_idx = np.random.choice(len(all_pnl), size=n_trades, replace=False) \
            if n_trades <= len(all_pnl) else np.random.choice(len(all_pnl), size=n_trades, replace=True)
        random_returns = all_pnl[random_idx]
        perm_sharpes.append(calc_sharpe(random_returns, horizon))

    perm_sharpes = np.array(perm_sharpes)
    p_value = float((np.sum(perm_sharpes >= actual_sharpe) + 1) / (n_perms + 1))

    passed = p_value < 0.05

    return {
        'pass': passed,
        'actual_sharpe': round(actual_sharpe, 3),
        'random_mean_sharpe': round(float(np.mean(perm_sharpes)), 3),
        'random_p95_sharpe': round(float(np.percentile(perm_sharpes, 95)), 3),
        'p_value': round(p_value, 4),
        'n_permutations': n_perms,
        'n_trades': n_trades,
    }


def test_4_subperiod_stability(close_a, close_b, spy, horizon, original_sharpe):
    """Sub-period stability: split into 4 equal periods, all must have Sharpe > 0."""
    print("    Test 4: Sub-period stability...")

    features = compute_features_v1(close_a, close_b, spy)
    target, pnl = compute_targets_v1(close_a, close_b, horizon)
    results = walk_forward_lgbm_v1(features, target, pnl, horizon)

    if results is None:
        return {'pass': False, 'reason': 'Strategy failed to produce results'}

    high_conf = results[results['proba'] >= CONFIDENCE_THRESHOLD].copy()
    if len(high_conf) < 20:
        return {'pass': False, 'reason': f'Only {len(high_conf)} high-conf trades, need >= 20'}

    # Split into 4 equal periods
    n = len(high_conf)
    quarter = n // 4
    periods = []
    for i in range(4):
        start = i * quarter
        end = (i + 1) * quarter if i < 3 else n
        period_returns = high_conf.iloc[start:end]['pnl'].values
        period_sharpe = calc_sharpe(period_returns, horizon)
        period_wr = calc_win_rate(period_returns)
        periods.append({
            'period': i + 1,
            'n_trades': len(period_returns),
            'sharpe': round(period_sharpe, 3),
            'win_rate': round(period_wr, 3),
            'mean_return_pct': round(float(period_returns.mean() * 100), 3),
            'start_date': str(high_conf.iloc[start]['date'].date()) if hasattr(high_conf.iloc[start]['date'], 'date') else str(high_conf.iloc[start]['date']),
            'end_date': str(high_conf.iloc[end - 1]['date'].date()) if hasattr(high_conf.iloc[end - 1]['date'], 'date') else str(high_conf.iloc[end - 1]['date']),
        })

    any_negative = any(p['sharpe'] < 0 for p in periods)
    passed = not any_negative

    return {
        'pass': passed,
        'periods': periods,
        'any_negative_sharpe': any_negative,
    }


def test_5_top3_removal(close_a, close_b, spy, horizon, original_sharpe):
    """Top-3 removal: remove 3 best trades, check if Sharpe drops > 50%."""
    print("    Test 5: Top-3 trade removal...")

    features = compute_features_v1(close_a, close_b, spy)
    target, pnl = compute_targets_v1(close_a, close_b, horizon)
    results = walk_forward_lgbm_v1(features, target, pnl, horizon)

    if results is None:
        return {'pass': False, 'reason': 'Strategy failed to produce results'}

    returns = get_strategy_returns(results)
    if len(returns) < 10:
        return {'pass': False, 'reason': f'Only {len(returns)} trades, need >= 10'}

    full_sharpe = calc_sharpe(returns, horizon)

    # Remove top 3 trades by PnL
    sorted_idx = np.argsort(returns)[::-1]  # descending
    top3_values = returns[sorted_idx[:3]]
    remaining = np.delete(returns, sorted_idx[:3])
    reduced_sharpe = calc_sharpe(remaining, horizon)

    if abs(full_sharpe) < 0.01:
        drop_pct = 100.0  # if original is ~0, any change is 100%
    else:
        drop_pct = (full_sharpe - reduced_sharpe) / abs(full_sharpe) * 100

    passed = drop_pct <= 50.0

    return {
        'pass': passed,
        'full_sharpe': round(full_sharpe, 3),
        'reduced_sharpe': round(reduced_sharpe, 3),
        'sharpe_drop_pct': round(drop_pct, 1),
        'n_trades_full': len(returns),
        'n_trades_reduced': len(remaining),
        'top3_returns_pct': [round(float(v * 100), 3) for v in top3_values],
        'threshold': '50%',
    }


def test_6_parameter_sensitivity(close_a, close_b, spy, horizon, original_sharpe):
    """Parameter sensitivity: test 5+ parameter variations."""
    print("    Test 6: Parameter sensitivity...")

    # Vary: train_days, num_leaves, learning_rate, threshold, lookback for target
    param_variations = [
        {'label': 'baseline', 'train_days': 252, 'num_leaves': 31, 'lr': 0.05, 'threshold': 0.6, 'target_lookback': 21},
        {'label': 'short_train', 'train_days': 189, 'num_leaves': 31, 'lr': 0.05, 'threshold': 0.6, 'target_lookback': 21},
        {'label': 'long_train', 'train_days': 315, 'num_leaves': 31, 'lr': 0.05, 'threshold': 0.6, 'target_lookback': 21},
        {'label': 'fewer_leaves', 'train_days': 252, 'num_leaves': 15, 'lr': 0.05, 'threshold': 0.6, 'target_lookback': 21},
        {'label': 'more_leaves', 'train_days': 252, 'num_leaves': 63, 'lr': 0.05, 'threshold': 0.6, 'target_lookback': 21},
        {'label': 'slow_lr', 'train_days': 252, 'num_leaves': 31, 'lr': 0.02, 'threshold': 0.6, 'target_lookback': 21},
        {'label': 'fast_lr', 'train_days': 252, 'num_leaves': 31, 'lr': 0.1, 'threshold': 0.6, 'target_lookback': 21},
        {'label': 'lower_thresh', 'train_days': 252, 'num_leaves': 31, 'lr': 0.05, 'threshold': 0.55, 'target_lookback': 21},
        {'label': 'higher_thresh', 'train_days': 252, 'num_leaves': 31, 'lr': 0.05, 'threshold': 0.65, 'target_lookback': 21},
        {'label': 'short_lookback', 'train_days': 252, 'num_leaves': 31, 'lr': 0.05, 'threshold': 0.6, 'target_lookback': 10},
        {'label': 'long_lookback', 'train_days': 252, 'num_leaves': 31, 'lr': 0.05, 'threshold': 0.6, 'target_lookback': 42},
    ]

    variation_results = []

    for var in param_variations:
        features = compute_features_v1(close_a, close_b, spy)

        # Compute target with variable lookback
        lookback = var['target_lookback']
        fwd_ret_a = close_a.pct_change(horizon).shift(-horizon)
        fwd_ret_b = close_b.pct_change(horizon).shift(-horizon)
        past_rel = (close_a / close_a.shift(lookback)) / (close_b / close_b.shift(lookback)) - 1
        reversal = np.where(past_rel < 0, fwd_ret_a > fwd_ret_b, fwd_ret_b > fwd_ret_a).astype(float)
        reversal_pnl = np.where(past_rel < 0, fwd_ret_a - fwd_ret_b, fwd_ret_b - fwd_ret_a)
        target = pd.Series(reversal, index=close_a.index)
        pnl = pd.Series(reversal_pnl, index=close_a.index)

        lgbm_params = {
            'objective': 'binary', 'metric': 'auc', 'verbosity': -1,
            'num_leaves': var['num_leaves'], 'learning_rate': var['lr'],
            'feature_fraction': 0.8, 'bagging_fraction': 0.8, 'bagging_freq': 5,
            'min_child_samples': 20, 'n_estimators': 200, 'random_state': 42,
        }

        results = walk_forward_lgbm_v1(features, target, pnl, horizon,
                                        train_days=var['train_days'],
                                        lgbm_params=lgbm_params,
                                        threshold=var['threshold'])

        if results is None:
            variation_results.append({'label': var['label'], 'sharpe': 0.0, 'n_trades': 0})
            continue

        returns = get_strategy_returns(results, threshold=var['threshold'])
        s = calc_sharpe(returns, horizon)
        variation_results.append({
            'label': var['label'],
            'sharpe': round(s, 3),
            'n_trades': len(returns),
            'win_rate': round(calc_win_rate(returns), 3),
        })

    above_threshold = sum(1 for v in variation_results if v['sharpe'] > 0.30)
    total = len(variation_results)
    pct_above = above_threshold / total if total > 0 else 0

    passed = pct_above >= 0.80

    return {
        'pass': passed,
        'variations': variation_results,
        'pct_above_030': round(pct_above * 100, 1),
        'count_above_030': above_threshold,
        'total_variations': total,
        'threshold': '80%',
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    print("=" * 80)
    print("SUB-SECTOR ROTATION — 6-TEST ADVERSARIAL VALIDATION")
    print("=" * 80)
    print(f"Date range: {START_DATE} to {END_DATE}")
    print(f"Testing {len(PAIRS_TO_TEST)} pairs")
    print()

    close = download_data()
    spy = close['SPY']

    all_results = {}

    for pair_info in PAIRS_TO_TEST:
        name = pair_info['name']
        etf_a = pair_info['etf_a']
        etf_b = pair_info['etf_b']
        horizon = pair_info['horizon']
        orig_sharpe = pair_info['original_sharpe']

        print(f"\n{'='*70}")
        print(f"PAIR: {name} ({etf_a} vs {etf_b}) — {horizon}d horizon")
        print(f"Original Sharpe: {orig_sharpe:.3f}, WR: {pair_info['original_wr']:.1%}")
        print(f"{'='*70}")

        if etf_a not in close.columns or etf_b not in close.columns:
            print(f"  SKIPPING: Missing data for {etf_a} or {etf_b}")
            continue

        # Align data
        common = close[[etf_a, etf_b]].dropna().index
        common = common.intersection(spy.dropna().index)
        ca = close[etf_a].loc[common]
        cb = close[etf_b].loc[common]
        sp = spy.loc[common]

        test_results = {}

        # Run all 6 tests
        test_results['test_1_reimplementation'] = test_1_reimplementation(ca, cb, sp, horizon, orig_sharpe)
        test_results['test_2_inverse_signal'] = test_2_inverse_signal(ca, cb, sp, horizon, orig_sharpe)
        test_results['test_3_random_timing'] = test_3_random_timing(ca, cb, sp, horizon, orig_sharpe)
        test_results['test_4_subperiod_stability'] = test_4_subperiod_stability(ca, cb, sp, horizon, orig_sharpe)
        test_results['test_5_top3_removal'] = test_5_top3_removal(ca, cb, sp, horizon, orig_sharpe)
        test_results['test_6_parameter_sensitivity'] = test_6_parameter_sensitivity(ca, cb, sp, horizon, orig_sharpe)

        # Scorecard
        tests_passed = sum(1 for t in test_results.values() if t.get('pass', False))
        overall_pass = tests_passed >= 5

        print(f"\n  SCORECARD: {tests_passed}/6 tests passed — {'VALIDATED' if overall_pass else 'FAILED'}")
        for tname, tres in test_results.items():
            status = 'PASS' if tres.get('pass', False) else 'FAIL'
            print(f"    {tname}: {status}")

        all_results[name] = {
            'pair': name,
            'etf_a': etf_a,
            'etf_b': etf_b,
            'horizon': horizon,
            'original_sharpe': orig_sharpe,
            'original_wr': pair_info['original_wr'],
            'tests_passed': tests_passed,
            'overall_verdict': 'VALIDATED' if overall_pass else 'FAILED',
            'test_details': test_results,
        }

    # ========================================================================
    # SUMMARY
    # ========================================================================
    print("\n" + "=" * 80)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 80)

    n_validated = 0
    for name, r in all_results.items():
        verdict = r['overall_verdict']
        tp = r['tests_passed']
        if verdict == 'VALIDATED':
            n_validated += 1
        print(f"  {name}: {tp}/6 — {verdict}")
        for tname, tres in r['test_details'].items():
            status = 'PASS' if tres.get('pass', False) else 'FAIL'
            # Print key metric
            detail = ''
            if 'divergence_pct' in tres:
                detail = f"divergence={tres['divergence_pct']}%"
            elif 'ratio' in tres:
                detail = f"inv/fwd ratio={tres['ratio']}"
            elif 'p_value' in tres:
                detail = f"p={tres['p_value']}"
            elif 'any_negative_sharpe' in tres:
                detail = f"neg_period={'YES' if tres['any_negative_sharpe'] else 'no'}"
            elif 'sharpe_drop_pct' in tres:
                detail = f"drop={tres['sharpe_drop_pct']}%"
            elif 'pct_above_030' in tres:
                detail = f"{tres['pct_above_030']}% above 0.30"
            print(f"      {tname}: {status} ({detail})")

    print(f"\n  VALIDATED: {n_validated}/{len(all_results)} pairs")
    if n_validated == 0:
        print("  CONCLUSION: No pairs survived adversarial validation. Strategy is not robust.")
    elif n_validated <= 2:
        print("  CONCLUSION: Limited robustness. Only a few pairs validated — proceed with caution.")
    else:
        print(f"  CONCLUSION: {n_validated} pairs validated. Strategy shows genuine edge in these pairs.")

    # Save results
    output = {
        'metadata': {
            'script': 'subsector_rotation_adversarial.py',
            'run_date': datetime.now().isoformat(),
            'date_range': f'{START_DATE} to {END_DATE}',
            'n_pairs_tested': len(PAIRS_TO_TEST),
            'tests': [
                '1. Re-implementation (Sharpe divergence < 30%)',
                '2. Inverse signal (inv/fwd ratio < 0.50)',
                '3. Random timing (p-value < 0.05, 1000 perms)',
                '4. Sub-period stability (no negative Sharpe in 4 quarters)',
                '5. Top-3 removal (Sharpe drop < 50%)',
                '6. Parameter sensitivity (80%+ variations Sharpe > 0.30)',
            ],
            'pass_threshold': '5/6 tests',
        },
        'summary': {
            'pairs_validated': n_validated,
            'pairs_failed': len(all_results) - n_validated,
            'validated_pairs': [name for name, r in all_results.items() if r['overall_verdict'] == 'VALIDATED'],
            'failed_pairs': [name for name, r in all_results.items() if r['overall_verdict'] == 'FAILED'],
        },
        'pair_results': {},
    }

    # Serialize results (handle numpy types)
    def clean_val(v):
        if isinstance(v, (np.integer,)):
            return int(v)
        if isinstance(v, (np.floating,)):
            return float(v)
        if isinstance(v, (np.bool_,)):
            return bool(v)
        if isinstance(v, (np.ndarray,)):
            return v.tolist()
        if isinstance(v, pd.Timestamp):
            return str(v)
        if isinstance(v, dict):
            return {k: clean_val(vv) for k, vv in v.items()}
        if isinstance(v, list):
            return [clean_val(vv) for vv in v]
        return v

    for name, r in all_results.items():
        output['pair_results'][name] = clean_val(r)

    results_path = Path('/home/jupiter/Lvl3Quant/research_results/subsector_rotation_adversarial.json')
    results_path.parent.mkdir(parents=True, exist_ok=True)
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    print(f"\nResults saved to {results_path}")
    return output


if __name__ == '__main__':
    main()
