#!/usr/bin/env python3
"""
ML Tail Risk Predictor — Defensive Overlay for v4.4
=====================================================
Predicts extreme down days (SPY drops > 2%) one day ahead using cross-asset
features and LightGBM. When P(crash) > threshold, overrides v4.4 allocation
to SHY as a defensive hedge.

Walk-forward: 504d sliding train, 21d test, step 21d.
Adversarial validation per HC #705:
  1. Permutation test (200 shuffles)
  2. Sub-period consistency (4 blocks, CV < 0.50)
  3. Outlier robustness (trim 5%, Sharpe drop < 50%)
  4. R1 regime check (green/red asymmetry < 0.50)

$100K initial capital, NO DCA (HC #713).

Output: /home/jupiter/Lvl3Quant/output/ml_tail_risk/
"""

import os
import json
import time
import warnings
import datetime as dt
from pathlib import Path

import numpy as np
import pandas as pd
import yfinance as yf
import lightgbm as lgb
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, precision_score, recall_score, f1_score,
    precision_recall_curve, classification_report
)
from scipy import stats

warnings.filterwarnings('ignore')
np.random.seed(42)

# =============================================================================
# CONSTANTS
# =============================================================================
OUTPUT_DIR = Path('/home/jupiter/Lvl3Quant/output/ml_tail_risk')
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_CAPITAL = 100_000
REBAL_COST_BPS = 5
CRASH_THRESHOLD = -0.02  # SPY daily return < -2%

WF_TRAIN_DAYS = 504      # ~2 years
WF_TEST_DAYS = 21         # 1 month
WF_STEP_DAYS = 21

N_PERMUTATIONS = 200
N_SUBPERIOD_BLOCKS = 4

TICKERS = [
    'SPY', 'QQQ', 'IWM', 'GLD', 'TLT', 'SHY', 'IEF', 'HYG', 'LQD',
    'UUP', '^VIX', 'USO', 'DIA', 'XLF', 'XLU', 'BTC-USD', 'UPRO'
]

OVERLAY_THRESHOLDS = [0.3, 0.5, 0.7]


# =============================================================================
# DATA DOWNLOAD
# =============================================================================
def download_data():
    """Download daily data for all tickers."""
    print("=" * 80)
    print("ML TAIL RISK PREDICTOR — v4.4 Defensive Overlay")
    print("=" * 80)
    print(f"\nRun started: {dt.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Initial capital: ${INITIAL_CAPITAL:,}, NO DCA")

    cache_file = OUTPUT_DIR / 'price_cache.parquet'
    if cache_file.exists():
        age_hours = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_hours < 12:
            prices = pd.read_parquet(cache_file)
            print(f"  Loaded from cache: {prices.shape} ({age_hours:.1f}h old)")
            return prices

    print("\n[1] Downloading data (2010-2026)...")
    data = yf.download(TICKERS, start='2010-01-01', auto_adjust=True,
                       threads=True, progress=False)

    if isinstance(data.columns, pd.MultiIndex):
        prices = data['Close']
    else:
        prices = data

    # Flatten multi-index if needed
    if hasattr(prices.columns, 'droplevel'):
        try:
            prices.columns = prices.columns.droplevel(1)
        except Exception:
            pass

    prices = prices.rename(columns={'^VIX': 'VIX', 'BTC-USD': 'BTC'})
    prices = prices.ffill().dropna(subset=['SPY'])

    print(f"  Downloaded: {prices.shape[0]} days, {prices.shape[1]} tickers")
    print(f"  Date range: {prices.index[0].date()} to {prices.index[-1].date()}")

    prices.to_parquet(cache_file)
    return prices


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================
def build_features(prices):
    """Build cross-asset features. All features lagged by 1 day (no look-ahead)."""
    print("\n[2] Building features...")

    df = pd.DataFrame(index=prices.index)

    spy = prices['SPY']
    spy_ret = spy.pct_change()

    # --- SPY/QQQ/IWM returns ---
    for ticker in ['SPY', 'QQQ', 'IWM', 'DIA']:
        if ticker not in prices.columns:
            continue
        ret = prices[ticker].pct_change()
        for w in [1, 5, 10, 20]:
            if w == 1:
                df[f'{ticker}_ret_{w}d'] = ret
            else:
                df[f'{ticker}_ret_{w}d'] = prices[ticker].pct_change(w)

    # --- Realized vol ---
    for w in [5, 10, 20, 60]:
        df[f'SPY_rvol_{w}d'] = spy_ret.rolling(w).std() * np.sqrt(252)

    # --- VIX features ---
    if 'VIX' in prices.columns:
        vix = prices['VIX']
        df['VIX_level'] = vix
        df['VIX_chg_1d'] = vix.pct_change()
        df['VIX_chg_5d'] = vix.pct_change(5)
        df['VIX_pctile_63d'] = vix.rolling(63).apply(
            lambda x: stats.percentileofscore(x[:-1], x.iloc[-1]) if len(x) > 1 else 50,
            raw=False
        )
        # VIX / realized vol ratio (term structure proxy)
        rvol_20 = spy_ret.rolling(20).std() * np.sqrt(252) * 100
        df['VIX_rvol_ratio'] = vix / rvol_20.replace(0, np.nan)

    # --- Credit spread proxy: HYG - IEF ---
    if 'HYG' in prices.columns and 'IEF' in prices.columns:
        credit_spread = prices['HYG'].pct_change() - prices['IEF'].pct_change()
        for w in [1, 5, 20]:
            if w == 1:
                df[f'credit_spread_{w}d'] = credit_spread
            else:
                df[f'credit_spread_{w}d'] = (
                    prices['HYG'].pct_change(w) - prices['IEF'].pct_change(w)
                )

    # --- Flight to quality: TLT - SPY ---
    if 'TLT' in prices.columns:
        for w in [1, 5]:
            df[f'flight_to_quality_{w}d'] = (
                prices['TLT'].pct_change(w) - spy.pct_change(w)
            )

    # --- Dollar momentum ---
    if 'UUP' in prices.columns:
        for w in [5, 20]:
            df[f'UUP_mom_{w}d'] = prices['UUP'].pct_change(w)

    # --- Gold momentum ---
    if 'GLD' in prices.columns:
        for w in [5, 20]:
            df[f'GLD_mom_{w}d'] = prices['GLD'].pct_change(w)

    # --- Oil ---
    if 'USO' in prices.columns:
        for w in [5, 20]:
            df[f'USO_mom_{w}d'] = prices['USO'].pct_change(w)

    # --- Risk appetite: XLF - XLU ---
    if 'XLF' in prices.columns and 'XLU' in prices.columns:
        for w in [5, 20]:
            df[f'risk_appetite_{w}d'] = (
                prices['XLF'].pct_change(w) - prices['XLU'].pct_change(w)
            )

    # --- SPY distance from SMAs ---
    for w in [20, 50, 200]:
        sma = spy.rolling(w).mean()
        df[f'SPY_dist_sma{w}'] = (spy - sma) / sma

    # --- SPY drawdown from recent high ---
    for w in [20, 60]:
        rolling_max = spy.rolling(w).max()
        df[f'SPY_dd_from_{w}d_high'] = (spy - rolling_max) / rolling_max

    # --- Volume z-score ---
    if 'SPY' in prices.columns:
        try:
            vol_data = yf.download('SPY', start='2010-01-01', auto_adjust=True,
                                   progress=False)
            if 'Volume' in vol_data.columns:
                vol = vol_data['Volume'].reindex(prices.index).ffill()
                vol_mean = vol.rolling(20).mean()
                vol_std = vol.rolling(20).std()
                df['SPY_vol_zscore'] = (vol - vol_mean) / vol_std.replace(0, np.nan)
        except Exception:
            pass

    # --- BTC momentum ---
    if 'BTC' in prices.columns:
        for w in [5, 20]:
            df[f'BTC_mom_{w}d'] = prices['BTC'].pct_change(w)

    # --- Calendar features ---
    df['day_of_week'] = prices.index.dayofweek
    df['month'] = prices.index.month

    # --- Target: SPY drops > 2% TOMORROW ---
    target = (spy_ret.shift(-1) < CRASH_THRESHOLD).astype(int)

    # LAG ALL FEATURES by 1 day (avoid look-ahead)
    # Features at time t predict target at time t (which is tomorrow's return)
    # Since target = spy_ret.shift(-1), features are already "today's" values
    # predicting tomorrow. But to be safe, we lag features by 1 more day
    # to ensure we only use data available BEFORE today's close.
    # Actually: features use today's close prices (available EOD), target is
    # tomorrow's return. This is standard — no look-ahead as long as features
    # don't use tomorrow's data. Our features are all based on today's or
    # earlier prices. The shift(-1) on target means: "given features at t,
    # will SPY drop >2% on day t+1?"
    # No additional lag needed.

    print(f"  Features: {df.shape[1]}")
    print(f"  Crash days (SPY < -2%): {target.sum()} / {len(target)} "
          f"({100*target.mean():.1f}%)")

    return df, target


# =============================================================================
# v4.4 REGIME LOGIC (standalone, no imports)
# =============================================================================
def compute_v44_signals(prices):
    """Compute signals needed for v4.4 regime."""
    spy = prices['SPY']
    vix = prices['VIX'] if 'VIX' in prices.columns else pd.Series(15.0, index=prices.index)
    spy_ret = spy.pct_change()

    sig = {}
    sig['mom_5d'] = spy.pct_change(5)

    # RSI 10
    delta = spy_ret.copy()
    gain = delta.where(delta > 0, 0).rolling(10).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(10).mean()
    rs = gain / loss.replace(0, np.nan)
    sig['rsi_10'] = 100 - (100 / (1 + rs))

    sig['sma_20'] = spy.rolling(20).mean()
    sig['sma_50'] = spy.rolling(50).mean()
    sig['sma_200'] = spy.rolling(200).mean()
    sig['sma_200_slope'] = sig['sma_200'].pct_change(20)
    sig['vol_21d'] = spy_ret.rolling(21).std() * np.sqrt(252) * 100
    sig['vol_63d'] = spy_ret.rolling(63).std() * np.sqrt(252) * 100
    sig['vol_63d_trend'] = sig['vol_63d'] - sig['vol_63d'].rolling(21).mean()

    sig['vix_pctile_63'] = vix.rolling(63).apply(
        lambda x: (x.iloc[-1] > x.iloc[:-1]).sum() / (len(x) - 1) * 100
        if len(x) > 1 else 50,
        raw=False
    )

    return sig


def confluence_score(sig, i):
    """6-factor confluence score (0-3)."""
    s = 0.0
    m = sig['mom_5d'].iloc[i]
    r = sig['rsi_10'].iloc[i]
    s20 = sig['sma_20'].iloc[i]
    s50 = sig['sma_50'].iloc[i]
    v21 = sig['vol_21d'].iloc[i]
    slope = sig['sma_200_slope'].iloc[i]
    vt = sig['vol_63d_trend'].iloc[i]

    if not np.isnan(m) and m > 0: s += 0.5
    if not np.isnan(r) and r > 50: s += 0.5
    if not np.isnan(s20) and not np.isnan(s50) and s20 > s50: s += 0.5
    if not np.isnan(v21) and v21 < 15: s += 0.5
    if not np.isnan(slope) and slope > 0: s += 0.5
    if not np.isnan(vt) and vt < 0: s += 0.5
    return s


def v44_regime(sig, i, date, in_upro):
    """v4.4 Full Adaptive: VIX percentile + adaptive confluence."""
    s20 = sig['sma_20'].iloc[i]
    s200 = sig['sma_200'].iloc[i]

    if date.month == 9:
        return 'SPY', False
    if not np.isnan(s20) and not np.isnan(s200) and s20 < s200:
        return 'SPY', False

    pctile = sig['vix_pctile_63'].iloc[i]
    if np.isnan(pctile):
        return 'SPY', False

    if pctile > 80:
        return 'GLD', False

    # Adaptive thresholds
    if pctile > 60:
        entry, exit_t = 3.0, 2.5
    elif pctile < 30:
        entry, exit_t = 2.0, 1.5
    else:
        entry, exit_t = 2.5, 2.0

    if pctile > 20:
        entry = max(entry, 2.5)

    score = confluence_score(sig, i)
    if in_upro:
        if score < exit_t:
            return 'SPY', False
        return 'UPRO', True
    else:
        if score >= entry:
            return 'UPRO', True
        return 'SPY', False


# =============================================================================
# WALK-FORWARD ML
# =============================================================================
def run_walk_forward(features, target, prices):
    """Walk-forward: 504d train, 21d test, step 21d."""
    print("\n[3] Running walk-forward ML...")

    valid_mask = features.notna().all(axis=1) & target.notna()
    valid_idx = features.index[valid_mask]

    if len(valid_idx) < WF_TRAIN_DAYS + WF_TEST_DAYS:
        print("  ERROR: Not enough valid data for walk-forward")
        return None, None

    feature_cols = [c for c in features.columns if c not in ['day_of_week', 'month']]
    cat_features = ['day_of_week', 'month']
    all_cols = feature_cols + cat_features

    # Clean features
    X = features[all_cols].copy()
    X = X.replace([np.inf, -np.inf], np.nan)
    # Fill NaN with column median
    col_medians = X.median()
    X = X.fillna(col_medians)

    y = target.copy()

    # Build folds
    folds = []
    start = 0
    dates = features.index

    while start + WF_TRAIN_DAYS + WF_TEST_DAYS <= len(dates):
        train_end = start + WF_TRAIN_DAYS
        test_end = min(train_end + WF_TEST_DAYS, len(dates))

        train_idx = list(range(start, train_end))
        test_idx = list(range(train_end, test_end))

        # Only include if we have valid data
        train_valid = [i for i in train_idx if valid_mask.iloc[i]]
        test_valid = [i for i in test_idx if valid_mask.iloc[i]]

        if len(train_valid) > 100 and len(test_valid) > 0:
            folds.append((train_valid, test_valid))

        start += WF_STEP_DAYS

    print(f"  Walk-forward folds: {len(folds)}")
    print(f"  Train window: {WF_TRAIN_DAYS}d, Test window: {WF_TEST_DAYS}d, Step: {WF_STEP_DAYS}d")

    # LightGBM and RF predictions
    lgb_predictions = pd.Series(dtype=float, index=dates)
    rf_predictions = pd.Series(dtype=float, index=dates)
    fold_metrics = []

    for fold_i, (train_idx, test_idx) in enumerate(folds):
        X_train = X.iloc[train_idx].values
        y_train = y.iloc[train_idx].values
        X_test = X.iloc[test_idx].values
        y_test = y.iloc[test_idx].values
        test_dates = dates[test_idx]

        # Compute class weight
        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        if n_pos == 0:
            scale_pos = 1.0
        else:
            scale_pos = n_neg / n_pos

        # LightGBM
        lgb_params = {
            'objective': 'binary',
            'metric': 'auc',
            'verbosity': -1,
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.05,
            'num_leaves': 15,
            'min_child_samples': 20,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos,
            'random_state': 42,
            'n_jobs': -1,
        }

        lgb_model = lgb.LGBMClassifier(**lgb_params)
        lgb_model.fit(X_train, y_train,
                      eval_set=[(X_test, y_test)],
                      callbacks=[lgb.log_evaluation(period=0)])

        lgb_proba = lgb_model.predict_proba(X_test)[:, 1]
        for d, p in zip(test_dates, lgb_proba):
            lgb_predictions.loc[d] = p

        # Random Forest
        rf_model = RandomForestClassifier(
            n_estimators=200, max_depth=6, min_samples_leaf=20,
            class_weight='balanced', random_state=42, n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
        rf_proba = rf_model.predict_proba(X_test)[:, 1]
        for d, p in zip(test_dates, rf_proba):
            rf_predictions.loc[d] = p

        # Fold metrics
        if len(np.unique(y_test)) > 1:
            auc = roc_auc_score(y_test, lgb_proba)
        else:
            auc = np.nan

        fold_metrics.append({
            'fold': fold_i,
            'test_start': str(test_dates[0].date()),
            'test_end': str(test_dates[-1].date()),
            'n_crashes': int(y_test.sum()),
            'auc': auc,
        })

        if (fold_i + 1) % 10 == 0:
            recent_aucs = [m['auc'] for m in fold_metrics[-10:] if not np.isnan(m['auc'])]
            avg_auc = np.mean(recent_aucs) if recent_aucs else np.nan
            print(f"  Fold {fold_i+1}/{len(folds)} | "
                  f"Recent AUC: {avg_auc:.3f} | "
                  f"Test: {test_dates[0].date()} to {test_dates[-1].date()}")

    # Drop NaN predictions
    lgb_predictions = lgb_predictions.dropna()
    rf_predictions = rf_predictions.dropna()

    # Feature importance (last model)
    if len(folds) > 0:
        importance = pd.Series(
            lgb_model.feature_importances_,
            index=all_cols
        ).sort_values(ascending=False)
        print(f"\n  Top 15 features (last fold):")
        for feat, imp in importance.head(15).items():
            print(f"    {feat}: {imp}")

    print(f"\n  LGB predictions: {len(lgb_predictions)}, RF predictions: {len(rf_predictions)}")

    return lgb_predictions, rf_predictions, fold_metrics


# =============================================================================
# EVALUATION METRICS
# =============================================================================
def evaluate_predictions(predictions, target, model_name='LGB'):
    """Evaluate prediction quality."""
    print(f"\n[4] Evaluating {model_name} predictions...")

    # Align
    common_idx = predictions.index.intersection(target.index)
    pred = predictions.loc[common_idx]
    actual = target.loc[common_idx]

    valid = pred.notna() & actual.notna()
    pred = pred[valid]
    actual = actual[valid]

    if len(pred) == 0 or actual.sum() == 0:
        print("  No valid predictions or no crash days in test period")
        return {}

    # Overall AUC
    auc = roc_auc_score(actual, pred)
    print(f"  Overall AUC: {auc:.4f}")

    # Precision/Recall at various thresholds
    results = {'auc': auc, 'n_predictions': len(pred),
               'n_crashes': int(actual.sum()), 'thresholds': {}}

    for thresh in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
        pred_pos = (pred >= thresh).astype(int)
        if pred_pos.sum() == 0:
            continue

        prec = precision_score(actual, pred_pos, zero_division=0)
        rec = recall_score(actual, pred_pos, zero_division=0)
        f1 = f1_score(actual, pred_pos, zero_division=0)
        n_alerts = int(pred_pos.sum())
        true_pos = int((pred_pos & actual).sum())

        results['thresholds'][str(thresh)] = {
            'precision': round(prec, 4),
            'recall': round(rec, 4),
            'f1': round(f1, 4),
            'n_alerts': n_alerts,
            'true_positives': true_pos,
        }
        print(f"  Threshold {thresh:.1f}: Prec={prec:.3f} Rec={rec:.3f} "
              f"F1={f1:.3f} Alerts={n_alerts} TP={true_pos}")

    return results


# =============================================================================
# v4.4 BACKTEST (with and without overlay)
# =============================================================================
def backtest_v44(prices, overlay_predictions=None, overlay_threshold=0.5, label='v4.4'):
    """Run v4.4 strategy backtest, optionally with tail risk overlay."""
    sig = compute_v44_signals(prices)
    spy_ret = prices['SPY'].pct_change()

    # Asset returns
    asset_returns = {}
    for asset in ['SPY', 'UPRO', 'GLD', 'SHY']:
        if asset in prices.columns:
            asset_returns[asset] = prices[asset].pct_change()
        else:
            asset_returns[asset] = pd.Series(0.0, index=prices.index)

    dates = prices.index
    n = len(dates)

    capital = float(INITIAL_CAPITAL)
    equity_curve = np.zeros(n)
    equity_curve[0] = capital
    daily_returns = np.zeros(n)
    allocations = []
    in_upro = False
    prev_asset = 'SPY'
    override_count = 0

    # Start after warmup
    warmup = 252

    for i in range(1, n):
        date = dates[i]

        if i < warmup:
            equity_curve[i] = capital
            continue

        # v4.4 regime decision
        asset, in_upro = v44_regime(sig, i - 1, date, in_upro)

        # Tail risk overlay: override to SHY if P(crash) > threshold
        if overlay_predictions is not None and date in overlay_predictions.index:
            p_crash = overlay_predictions.loc[date]
            if not np.isnan(p_crash) and p_crash > overlay_threshold:
                asset = 'SHY'
                in_upro = False
                override_count += 1

        # Switching cost
        if asset != prev_asset:
            cost = capital * REBAL_COST_BPS / 10000
            capital -= cost

        # Apply return
        ret = asset_returns.get(asset, pd.Series(0.0, index=dates))
        if hasattr(ret, 'iloc') and i < len(ret):
            day_ret = ret.iloc[i]
        else:
            day_ret = 0.0

        if np.isnan(day_ret) or np.isinf(day_ret):
            day_ret = 0.0

        capital *= (1 + day_ret)
        equity_curve[i] = capital
        daily_returns[i] = day_ret
        prev_asset = asset

    # Compute metrics
    valid_rets = daily_returns[warmup:]
    valid_rets = valid_rets[valid_rets != 0]

    if len(valid_rets) < 50:
        return None

    # Trim to valid equity curve
    ec = pd.Series(equity_curve[warmup:], index=dates[warmup:])
    dr = pd.Series(daily_returns[warmup:], index=dates[warmup:])

    metrics = compute_metrics(ec, dr, label)
    metrics['override_count'] = override_count

    return {
        'equity_curve': ec,
        'daily_returns': dr,
        'metrics': metrics,
    }


def compute_metrics(equity_curve, daily_returns, label=''):
    """Compute Sharpe, Sortino, CAGR, MaxDD."""
    dr = daily_returns[daily_returns != 0]
    if len(dr) < 50:
        return {}

    # Annualized return
    total_return = equity_curve.iloc[-1] / equity_curve.iloc[0]
    n_years = len(dr) / 252
    cagr = total_return ** (1 / n_years) - 1 if n_years > 0 else 0

    # Sharpe
    sharpe = np.sqrt(252) * dr.mean() / dr.std() if dr.std() > 0 else 0

    # Sortino
    downside = dr[dr < 0]
    downside_std = downside.std() if len(downside) > 10 else dr.std()
    sortino = np.sqrt(252) * dr.mean() / downside_std if downside_std > 0 else 0

    # Max drawdown
    cummax = equity_curve.cummax()
    drawdown = (equity_curve - cummax) / cummax
    maxdd = drawdown.min()

    # Win rate (on active days)
    wr = (dr > 0).mean()

    # Profit factor
    gross_profit = dr[dr > 0].sum()
    gross_loss = abs(dr[dr < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else np.inf

    # Calmar
    calmar = cagr / abs(maxdd) if maxdd != 0 else 0

    return {
        'label': label,
        'sharpe': round(sharpe, 3),
        'sortino': round(sortino, 3),
        'cagr': round(cagr * 100, 2),
        'maxdd': round(maxdd * 100, 2),
        'calmar': round(calmar, 3),
        'profit_factor': round(pf, 3),
        'win_rate': round(wr * 100, 1),
        'final_capital': round(equity_curve.iloc[-1], 2),
        'n_days': len(dr),
    }


# =============================================================================
# ADVERSARIAL VALIDATION (HC #705)
# =============================================================================
def adversarial_validation(predictions, target, prices, overlay_results):
    """Full adversarial validation suite."""
    print("\n[6] Adversarial Validation (HC #705)...")
    results = {}

    # Find the best overlay threshold
    best_thresh = None
    best_sharpe = -999
    for thresh_key, res in overlay_results.items():
        if res and 'metrics' in res and res['metrics'].get('sharpe', -999) > best_sharpe:
            best_sharpe = res['metrics']['sharpe']
            best_thresh = thresh_key

    if best_thresh is None:
        print("  No valid overlay results for adversarial validation")
        return results

    best_overlay = overlay_results[best_thresh]
    print(f"  Testing best overlay: threshold={best_thresh}, Sharpe={best_sharpe:.3f}")

    # ── 1. Permutation Test (200 shuffles) ──
    print(f"\n  [6a] Permutation test ({N_PERMUTATIONS} shuffles)...")
    real_sharpe = best_overlay['metrics']['sharpe']
    perm_sharpes = []

    common_idx = predictions.index.intersection(target.index)
    pred_values = predictions.loc[common_idx].values.copy()

    for perm_i in range(N_PERMUTATIONS):
        # Shuffle predictions
        shuffled = pred_values.copy()
        np.random.shuffle(shuffled)
        shuffled_pred = pd.Series(shuffled, index=common_idx)

        perm_result = backtest_v44(
            prices, overlay_predictions=shuffled_pred,
            overlay_threshold=float(best_thresh),
            label=f'perm_{perm_i}'
        )
        if perm_result and 'metrics' in perm_result:
            perm_sharpes.append(perm_result['metrics'].get('sharpe', 0))

        if (perm_i + 1) % 50 == 0:
            print(f"    Permutation {perm_i+1}/{N_PERMUTATIONS}")

    if perm_sharpes:
        p_value = np.mean([s >= real_sharpe for s in perm_sharpes])
        results['permutation_test'] = {
            'real_sharpe': real_sharpe,
            'perm_mean': round(np.mean(perm_sharpes), 3),
            'perm_std': round(np.std(perm_sharpes), 3),
            'p_value': round(p_value, 4),
            'pass': p_value < 0.05,
        }
        print(f"    Real Sharpe: {real_sharpe:.3f} | "
              f"Perm mean: {np.mean(perm_sharpes):.3f} +/- {np.std(perm_sharpes):.3f} | "
              f"p-value: {p_value:.4f} | "
              f"{'PASS' if p_value < 0.05 else 'FAIL'}")

    # ── 2. Sub-period Consistency (4 blocks, CV < 0.50) ──
    print(f"\n  [6b] Sub-period consistency ({N_SUBPERIOD_BLOCKS} blocks)...")
    dr = best_overlay['daily_returns']
    block_size = len(dr) // N_SUBPERIOD_BLOCKS
    block_sharpes = []

    for b in range(N_SUBPERIOD_BLOCKS):
        start = b * block_size
        end = (b + 1) * block_size if b < N_SUBPERIOD_BLOCKS - 1 else len(dr)
        block_dr = dr.iloc[start:end]
        block_dr_active = block_dr[block_dr != 0]

        if len(block_dr_active) > 20 and block_dr_active.std() > 0:
            block_sharpe = np.sqrt(252) * block_dr_active.mean() / block_dr_active.std()
        else:
            block_sharpe = 0.0

        block_sharpes.append(block_sharpe)
        print(f"    Block {b+1}: {dr.index[start].date()} to {dr.index[min(end-1, len(dr)-1)].date()} | "
              f"Sharpe: {block_sharpe:.3f}")

    if len(block_sharpes) > 1 and np.mean(block_sharpes) != 0:
        cv = np.std(block_sharpes) / abs(np.mean(block_sharpes))
    else:
        cv = 999

    results['subperiod_consistency'] = {
        'block_sharpes': [round(s, 3) for s in block_sharpes],
        'cv': round(cv, 3),
        'pass': cv < 0.50,
    }
    print(f"    CV: {cv:.3f} | {'PASS' if cv < 0.50 else 'FAIL'} (threshold: 0.50)")

    # ── 3. Outlier Robustness (trim 5%, Sharpe drop < 50%) ──
    print("\n  [6c] Outlier robustness (trim top/bottom 5%)...")
    dr_active = dr[dr != 0]
    lower = dr_active.quantile(0.025)
    upper = dr_active.quantile(0.975)
    trimmed = dr_active[(dr_active >= lower) & (dr_active <= upper)]

    if len(trimmed) > 20 and trimmed.std() > 0:
        trimmed_sharpe = np.sqrt(252) * trimmed.mean() / trimmed.std()
    else:
        trimmed_sharpe = 0

    if real_sharpe != 0:
        sharpe_drop = 1 - (trimmed_sharpe / real_sharpe)
    else:
        sharpe_drop = 0

    results['outlier_robustness'] = {
        'full_sharpe': real_sharpe,
        'trimmed_sharpe': round(trimmed_sharpe, 3),
        'sharpe_drop_pct': round(sharpe_drop * 100, 1),
        'pass': abs(sharpe_drop) < 0.50,
    }
    print(f"    Full Sharpe: {real_sharpe:.3f} | "
          f"Trimmed: {trimmed_sharpe:.3f} | "
          f"Drop: {sharpe_drop*100:.1f}% | "
          f"{'PASS' if abs(sharpe_drop) < 0.50 else 'FAIL'}")

    # ── 4. R1 Regime Check (green/red asymmetry < 0.50) ──
    print("\n  [6d] R1 Regime check (green vs red day Sharpe asymmetry)...")
    spy_ret = prices['SPY'].pct_change()
    common_dates = dr.index.intersection(spy_ret.index)
    spy_ret_aligned = spy_ret.loc[common_dates]
    dr_aligned = dr.loc[common_dates]

    green_mask = spy_ret_aligned > 0.001
    red_mask = spy_ret_aligned < -0.001

    green_dr = dr_aligned[green_mask]
    red_dr = dr_aligned[red_mask]

    green_active = green_dr[green_dr != 0]
    red_active = red_dr[red_dr != 0]

    if len(green_active) > 20 and green_active.std() > 0:
        green_sharpe = np.sqrt(252) * green_active.mean() / green_active.std()
    else:
        green_sharpe = 0

    if len(red_active) > 20 and red_active.std() > 0:
        red_sharpe = np.sqrt(252) * red_active.mean() / red_active.std()
    else:
        red_sharpe = 0

    max_abs = max(abs(green_sharpe), abs(red_sharpe))
    if max_abs > 0:
        asymmetry = abs(green_sharpe - red_sharpe) / max_abs
    else:
        asymmetry = 0

    results['r1_regime'] = {
        'green_sharpe': round(green_sharpe, 3),
        'red_sharpe': round(red_sharpe, 3),
        'asymmetry': round(asymmetry, 3),
        'pass': asymmetry < 0.50,
    }
    print(f"    Green day Sharpe: {green_sharpe:.3f} | "
          f"Red day Sharpe: {red_sharpe:.3f} | "
          f"Asymmetry: {asymmetry:.3f} | "
          f"{'PASS' if asymmetry < 0.50 else 'FAIL'}")

    # ── Summary ──
    n_pass = sum(1 for v in results.values() if isinstance(v, dict) and v.get('pass', False))
    n_total = len(results)
    print(f"\n  Adversarial Summary: {n_pass}/{n_total} tests passed")

    return results


# =============================================================================
# MAIN
# =============================================================================
def main():
    t0 = time.time()

    # 1. Download data
    prices = download_data()

    # 2. Build features and target
    features, target = build_features(prices)

    # 3. Walk-forward ML
    lgb_preds, rf_preds, fold_metrics = run_walk_forward(features, target, prices)

    if lgb_preds is None:
        print("ERROR: Walk-forward failed")
        return

    # 4. Evaluate predictions
    lgb_eval = evaluate_predictions(lgb_preds, target, 'LightGBM')
    rf_eval = evaluate_predictions(rf_preds, target, 'RandomForest')

    # 5. v4.4 Backtest with overlay comparison
    print("\n[5] v4.4 Strategy Comparison...")
    print("=" * 60)

    # Baseline: v4.4 alone
    baseline = backtest_v44(prices, label='v4.4 Baseline')
    if baseline:
        m = baseline['metrics']
        print(f"\n  v4.4 Baseline:")
        print(f"    Sharpe: {m['sharpe']:.3f} | Sortino: {m['sortino']:.3f} | "
              f"CAGR: {m['cagr']:.1f}% | MaxDD: {m['maxdd']:.1f}% | "
              f"Final: ${m['final_capital']:,.0f}")

    # SPY Buy & Hold benchmark
    spy_bh = backtest_spy_bh(prices)
    if spy_bh:
        m = spy_bh['metrics']
        print(f"\n  SPY Buy & Hold:")
        print(f"    Sharpe: {m['sharpe']:.3f} | Sortino: {m['sortino']:.3f} | "
              f"CAGR: {m['cagr']:.1f}% | MaxDD: {m['maxdd']:.1f}% | "
              f"Final: ${m['final_capital']:,.0f}")

    # Overlay at different thresholds (LGB)
    overlay_results = {}
    print(f"\n  Testing LGB overlay at thresholds: {OVERLAY_THRESHOLDS}")
    for thresh in OVERLAY_THRESHOLDS:
        label = f'v4.4 + LGB Overlay (P>{thresh})'
        result = backtest_v44(prices, overlay_predictions=lgb_preds,
                              overlay_threshold=thresh, label=label)
        if result:
            m = result['metrics']
            print(f"\n  {label}:")
            print(f"    Sharpe: {m['sharpe']:.3f} | Sortino: {m['sortino']:.3f} | "
                  f"CAGR: {m['cagr']:.1f}% | MaxDD: {m['maxdd']:.1f}% | "
                  f"Overrides: {m['override_count']} | Final: ${m['final_capital']:,.0f}")
            overlay_results[str(thresh)] = result

    # RF overlay comparison
    print(f"\n  Testing RF overlay at threshold 0.5...")
    rf_overlay = backtest_v44(prices, overlay_predictions=rf_preds,
                              overlay_threshold=0.5, label='v4.4 + RF Overlay (P>0.5)')
    if rf_overlay:
        m = rf_overlay['metrics']
        print(f"\n  v4.4 + RF Overlay (P>0.5):")
        print(f"    Sharpe: {m['sharpe']:.3f} | Sortino: {m['sortino']:.3f} | "
              f"CAGR: {m['cagr']:.1f}% | MaxDD: {m['maxdd']:.1f}% | "
              f"Overrides: {m['override_count']} | Final: ${m['final_capital']:,.0f}")

    # 6. Adversarial validation
    adv_results = adversarial_validation(lgb_preds, target, prices, overlay_results)

    # 7. Save results
    print("\n[7] Saving results...")
    save_results(prices, lgb_preds, rf_preds, lgb_eval, rf_eval,
                 baseline, overlay_results, rf_overlay, adv_results,
                 fold_metrics, t0)


def backtest_spy_bh(prices):
    """SPY buy-and-hold benchmark."""
    spy_ret = prices['SPY'].pct_change()
    dates = prices.index
    warmup = 252

    capital = float(INITIAL_CAPITAL)
    equity_curve = np.zeros(len(dates))
    equity_curve[0] = capital
    daily_returns = np.zeros(len(dates))

    for i in range(1, len(dates)):
        if i < warmup:
            equity_curve[i] = capital
            continue
        ret = spy_ret.iloc[i]
        if np.isnan(ret) or np.isinf(ret):
            ret = 0.0
        capital *= (1 + ret)
        equity_curve[i] = capital
        daily_returns[i] = ret

    ec = pd.Series(equity_curve[warmup:], index=dates[warmup:])
    dr = pd.Series(daily_returns[warmup:], index=dates[warmup:])
    metrics = compute_metrics(ec, dr, 'SPY B&H')

    return {'equity_curve': ec, 'daily_returns': dr, 'metrics': metrics}


def save_results(prices, lgb_preds, rf_preds, lgb_eval, rf_eval,
                 baseline, overlay_results, rf_overlay, adv_results,
                 fold_metrics, t0):
    """Save all results to output directory."""
    elapsed = time.time() - t0

    # Predictions
    lgb_preds.to_csv(OUTPUT_DIR / 'lgb_predictions.csv', header=True)
    rf_preds.to_csv(OUTPUT_DIR / 'rf_predictions.csv', header=True)

    # Summary JSON
    summary = {
        'run_timestamp': dt.datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'initial_capital': INITIAL_CAPITAL,
        'crash_threshold': CRASH_THRESHOLD,
        'wf_train_days': WF_TRAIN_DAYS,
        'wf_test_days': WF_TEST_DAYS,
        'n_folds': len(fold_metrics),
        'lgb_evaluation': lgb_eval,
        'rf_evaluation': rf_eval,
        'baseline_metrics': baseline['metrics'] if baseline else {},
        'overlay_metrics': {
            k: v['metrics'] for k, v in overlay_results.items() if v
        },
        'rf_overlay_metrics': rf_overlay['metrics'] if rf_overlay else {},
        'adversarial_validation': adv_results,
    }

    with open(OUTPUT_DIR / 'results_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # Fold metrics
    pd.DataFrame(fold_metrics).to_csv(OUTPUT_DIR / 'fold_metrics.csv', index=False)

    # Equity curves
    eq_data = {}
    if baseline:
        eq_data['v44_baseline'] = baseline['equity_curve']
    for k, v in overlay_results.items():
        if v:
            eq_data[f'overlay_{k}'] = v['equity_curve']
    if rf_overlay:
        eq_data['rf_overlay_0.5'] = rf_overlay['equity_curve']

    if eq_data:
        eq_df = pd.DataFrame(eq_data)
        eq_df.to_csv(OUTPUT_DIR / 'equity_curves.csv')

    # Print comparison table
    print("\n" + "=" * 90)
    print("FINAL COMPARISON")
    print("=" * 90)
    print(f"{'Strategy':<35} {'Sharpe':>7} {'Sortino':>8} {'CAGR%':>7} "
          f"{'MaxDD%':>7} {'PF':>6} {'WR%':>5} {'Final$':>12}")
    print("-" * 90)

    all_results = []
    if baseline:
        all_results.append(baseline['metrics'])
    for k, v in sorted(overlay_results.items()):
        if v:
            all_results.append(v['metrics'])
    if rf_overlay:
        all_results.append(rf_overlay['metrics'])

    for m in all_results:
        print(f"{m['label']:<35} {m['sharpe']:>7.3f} {m['sortino']:>8.3f} "
              f"{m['cagr']:>7.1f} {m['maxdd']:>7.1f} {m['profit_factor']:>6.2f} "
              f"{m['win_rate']:>5.1f} {m['final_capital']:>12,.0f}")

    # Adversarial summary
    print("\n" + "=" * 60)
    print("ADVERSARIAL VALIDATION SUMMARY")
    print("=" * 60)
    for test_name, test_result in adv_results.items():
        if isinstance(test_result, dict) and 'pass' in test_result:
            status = "PASS" if test_result['pass'] else "FAIL"
            print(f"  {test_name}: {status}")
            for k, v in test_result.items():
                if k != 'pass':
                    print(f"    {k}: {v}")

    print(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f}m)")
    print(f"Results saved to: {OUTPUT_DIR}")


if __name__ == '__main__':
    main()
