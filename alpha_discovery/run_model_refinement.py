"""
Model Refinement — Feature Selection, Linear Models, PnL Simulation

Tests whether we can improve on LightGBM IC=0.079 (ret_3s) and IC=0.075 (ret_1s)
through:
1. Feature selection (top-N features, less overfitting)
2. Linear model sanity check (Ridge regression)
3. Session-conditioned models (morning vs afternoon)
4. Realistic PnL simulation with ES futures costs
5. Ensemble of horizons

Usage:
    python alpha_discovery/run_model_refinement.py
    python alpha_discovery/run_model_refinement.py --targets ret_1s ret_3s
    python alpha_discovery/run_model_refinement.py --pnl-only
"""

import sys
import gc
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, ttest_1samp
from typing import Dict, List, Optional, Tuple

# Setup path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.mbo_features import get_feature_names, TOTAL_FEATURES
from alpha_discovery.run_return_multihorizon import (
    load_feature_cache, compute_return_targets,
    EXCLUDE_FEATURES_DIRECTION,
)

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / 'model_refinement.log', mode='a'),
    ]
)
logger = logging.getLogger("model_refinement")

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE = 0.25       # ES futures tick
TICK_VALUE = 12.50     # $12.50 per tick per contract
ROUND_TRIP_TICKS = 1.0 # ~1 tick round-trip cost (conservative)
ROUND_TRIP_COST = ROUND_TRIP_TICKS * TICK_VALUE  # $12.50

# Top features from the full return scan (aggregated importance across all targets)
# These are from the completed scan — sorted by total importance
TOP_FEATURES_RANKED = [
    'depth_ratio_l1',        # 5164
    'ask_L1_orders',         # 4729
    'bid_L1_conc',           # 4343
    'ask_L1_conc',           # 3579
    'bid_L1_orders',         # 3104
    'total_bid_vol',         # 2630
    'bid_L5_conc',           # 2356
    'ret_100',               # 2266
    'ofi_5',                 # 1945
    'ask_L4_orders',         # 1942
    'depth_ratio_l3',        # 1837
    'ofi_50',                # 1816
    'depth_ratio_l5',        # 1716
    'bid_pressure',          # 1635
    'ask_L5_orders',         # 1606
    'vol_regime',            # 1519
    'depth_concentration',   # 1347
    'bid_L5_orders',         # 1306
    'ofi_20',                # 1276
    'total_ask_vol',         # 1220
    'ask_L5_conc',           # ~1100 (estimated from prior scans)
    'ask_L2_orders',         # ~1050
    'bid_L4_conc',           # ~1000
    'bid_L2_conc',           # ~980
    'ask_L3_orders',         # ~950
    'bid_slope',             # ~920
    'ask_slope',             # ~900
    'bid_L3_conc',           # ~880
    'bid_L4_orders',         # ~860
    'bid_L2_orders',         # ~840
]


# ============================================================================
# WALK-FORWARD ENGINE (parameterized)
# ============================================================================

def walk_forward_evaluate(
    features: np.ndarray,
    target: np.ndarray,
    day_boundaries: list,
    feature_names: List[str],
    model_type: str = 'lgbm',  # 'lgbm', 'ridge', 'lasso'
    min_train_days: int = 3,
    hour_of_day: np.ndarray = None,
    time_since_rth: np.ndarray = None,
    session_filter: str = None,  # None, 'morning', 'afternoon'
    lgbm_params: dict = None,
    ridge_alpha: float = 1.0,
) -> dict:
    """
    Unified walk-forward evaluation for multiple model types.

    Returns dict with IC, fold_ics, predictions, actuals, etc.
    """
    import lightgbm as lgb
    from sklearn.linear_model import Ridge, Lasso
    from sklearn.preprocessing import StandardScaler

    n_days = len(day_boundaries) - 1
    if n_days < min_train_days + 1:
        return {'error': f'Need {min_train_days + 1} days, have {n_days}'}

    if lgbm_params is None:
        lgbm_params = {
            'n_estimators': 500,
            'max_depth': 6,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 100,
            'verbose': -1,
            'n_jobs': -1,
        }

    all_preds = []
    all_actuals = []
    all_indices = []  # bar indices for each prediction
    all_hours_list = []
    fold_ics = []
    fold_details = []
    feature_importance = np.zeros(features.shape[1])

    for test_day in range(min_train_days, n_days):
        train_start = day_boundaries[0]
        train_end = day_boundaries[test_day]  # purge gap = 1 day (skip test_day-1 to test_day)
        test_start = day_boundaries[test_day]
        test_end = day_boundaries[test_day + 1]

        X_train = features[train_start:train_end]
        y_train = target[train_start:train_end]
        X_test = features[test_start:test_end]
        y_test = target[test_start:test_end]

        # Session filter (if requested)
        if session_filter and hour_of_day is not None:
            h_train = hour_of_day[train_start:train_end]
            h_test = hour_of_day[test_start:test_end]

            if session_filter == 'morning':
                train_mask = h_train < 12.0
                test_mask = h_test < 12.0
            elif session_filter == 'afternoon':
                train_mask = h_train >= 12.0
                test_mask = h_test >= 12.0
            else:
                train_mask = np.ones(len(X_train), dtype=bool)
                test_mask = np.ones(len(X_test), dtype=bool)

            X_train = X_train[train_mask]
            y_train = y_train[train_mask]
            X_test = X_test[test_mask]
            y_test = y_test[test_mask]

        # Remove NaN targets
        train_valid = np.isfinite(y_train)
        test_valid = np.isfinite(y_test)

        if train_valid.sum() < 500 or test_valid.sum() < 50:
            continue

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]
        X_te = X_test[test_valid]
        y_te = y_test[test_valid]

        try:
            if model_type == 'lgbm':
                split = int(len(X_tr) * 0.8)
                model = lgb.LGBMRegressor(**lgbm_params)
                model.fit(
                    X_tr[:split], y_tr[:split],
                    eval_set=[(X_tr[split:], y_tr[split:])],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                preds = model.predict(X_te)
                if hasattr(model, 'feature_importances_'):
                    feature_importance += model.feature_importances_

            elif model_type == 'ridge':
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                model = Ridge(alpha=ridge_alpha)
                model.fit(X_tr_s, y_tr)
                preds = model.predict(X_te_s)
                feature_importance += np.abs(model.coef_)

            elif model_type == 'lasso':
                scaler = StandardScaler()
                X_tr_s = scaler.fit_transform(X_tr)
                X_te_s = scaler.transform(X_te)
                model = Lasso(alpha=ridge_alpha / 10000, max_iter=5000)
                model.fit(X_tr_s, y_tr)
                preds = model.predict(X_te_s)
                feature_importance += np.abs(model.coef_)
            else:
                raise ValueError(f"Unknown model_type: {model_type}")

        except Exception as e:
            logger.warning(f"  Training failed day {test_day}: {e}")
            continue

        all_preds.append(preds)
        all_actuals.append(y_te)

        # Track bar indices for each prediction
        bar_indices = np.arange(test_start, test_end)
        if session_filter and hour_of_day is not None:
            bar_indices = bar_indices[test_mask]
        bar_indices = bar_indices[test_valid]
        all_indices.append(bar_indices)

        if hour_of_day is not None:
            h_te = hour_of_day[test_start:test_end]
            if session_filter:
                h_te = h_te[test_mask if session_filter else np.ones(len(h_te), dtype=bool)]
            h_te = h_te[test_valid]
            all_hours_list.append(h_te)

        # Per-fold IC
        if len(preds) > 10:
            try:
                ic_fold = spearmanr(preds, y_te)[0]
                if np.isfinite(ic_fold):
                    fold_ics.append(float(ic_fold))
                    fold_details.append({
                        'day': test_day,
                        'ic': float(ic_fold),
                        'n_samples': int(len(preds)),
                        'train_size': int(train_valid.sum()),
                    })
            except Exception:
                pass

        del model
        gc.collect()

    if not all_preds:
        return {'error': 'No valid predictions'}

    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)
    pred_indices = np.concatenate(all_indices) if all_indices else np.array([], dtype=np.int64)

    valid = np.isfinite(predictions) & np.isfinite(actuals)
    p, a = predictions[valid], actuals[valid]

    if len(p) < 50:
        return {'error': f'Too few predictions: {len(p)}'}

    # Metrics
    ic = float(spearmanr(p, a)[0])

    if len(fold_ics) > 2:
        ic_mean = float(np.mean(fold_ics))
        ic_std = float(np.std(fold_ics))
        icir = ic_mean / ic_std if ic_std > 0 else 0.0
        tstat = ic_mean / ic_std * np.sqrt(len(fold_ics)) if ic_std > 0 else 0.0
        try:
            _, pvalue = ttest_1samp(fold_ics, 0)
            pvalue = float(pvalue)
        except Exception:
            pvalue = 1.0
        fold_con = float(np.mean([1 for x in fold_ics if x > 0]))
    else:
        ic_mean = ic
        ic_std, icir, tstat, pvalue = 0.0, 0.0, 0.0, 1.0
        fold_con = float(np.mean([1 for x in fold_ics if x > 0])) if fold_ics else 0.0

    # Feature importance ranking
    top_k = min(10, len(feature_names))
    top_idx = np.argsort(feature_importance)[-top_k:][::-1]
    top_features = [(feature_names[i], float(feature_importance[i])) for i in top_idx]

    return {
        'ic': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'tstat': tstat,
        'pvalue': pvalue,
        'fold_ics': fold_ics,
        'fold_con': fold_con,
        'n_folds': len(fold_ics),
        'n_preds': len(p),
        'top_features': top_features,
        'predictions': predictions,
        'actuals': actuals,
        'pred_indices': pred_indices,
        'fold_details': fold_details,
    }


# ============================================================================
# PNL SIMULATION
# ============================================================================

def simulate_pnl(
    predictions: np.ndarray,
    actuals: np.ndarray,
    target_horizon_sec: float = 3.0,
    sample_interval_ms: int = 100,
    threshold_quantile: float = 0.7,  # only trade when prediction is strong
    tick_size: float = 0.25,
    tick_value: float = 12.50,
    round_trip_ticks: float = 1.0,
    contracts: int = 1,
) -> dict:
    """
    Simulate PnL for a directional strategy.

    Strategy: At each bar, if |prediction| > threshold, take position.
    Position: sign(prediction) * contracts
    Hold: until next prediction (bar-by-bar rebalancing)
    Cost: round_trip_ticks per trade entry/exit

    More realistic: only trade on signal strength, not every bar.
    """
    valid = np.isfinite(predictions) & np.isfinite(actuals)
    preds = predictions[valid]
    acts = actuals[valid]

    n = len(preds)
    if n < 100:
        return {'error': 'Too few samples'}

    # Determine threshold: only trade when |pred| > quantile of |pred|
    pred_abs = np.abs(preds - np.median(preds))
    threshold = np.quantile(pred_abs, threshold_quantile)

    # Position: sign(prediction) when strong enough, 0 otherwise
    position = np.where(pred_abs > threshold, np.sign(preds - np.median(preds)), 0)

    # PnL per bar (in log returns) * position
    bar_pnl_raw = position * acts  # log return units

    # Convert to dollar PnL per contract
    # For ES futures: 1 point = $50, so log_return * price * $50
    # Approx: log_return * mean_price * $50
    # But for simplicity, convert log return to ticks:
    # log_return / (tick_size / mean_price) ≈ log_return * mean_price / tick_size
    # Then multiply by tick_value
    # Simpler: assume mean_price ≈ 5500, tick_size = 0.25
    # 1 log_return unit = price * log_return points = 5500 * log_return
    # In ticks: 5500 * log_return / 0.25 = 22000 * log_return
    # In dollars: 22000 * log_return * 12.50 = 275000 * log_return

    mean_price = 5500.0  # approximate ES price
    dollars_per_logret = mean_price / tick_size * tick_value * contracts

    bar_pnl_dollars = bar_pnl_raw * dollars_per_logret

    # Cost: every position change costs round-trip
    position_changes = np.abs(np.diff(position, prepend=0))
    # Each change incurs half round-trip (entering), each exit incurs half
    # Simplify: every change from 0 to +/-1 = 0.5 RT, from +1 to -1 = 1 RT
    cost_per_change = round_trip_ticks * tick_value * contracts
    total_cost_per_bar = position_changes * cost_per_change * 0.5  # half for entry

    # Net PnL
    net_pnl = bar_pnl_dollars - total_cost_per_bar

    # Statistics
    cum_pnl = np.cumsum(net_pnl)
    cum_gross = np.cumsum(bar_pnl_dollars)

    # Number of trades (position changes)
    n_trades = int(np.sum(position_changes > 0))
    n_bars_traded = int(np.sum(position != 0))
    pct_traded = n_bars_traded / n

    # Bars per second at 100ms interval
    bars_per_sec = 1000.0 / sample_interval_ms
    bars_per_day = bars_per_sec * 6.5 * 3600  # 6.5h trading day

    # Daily stats (approximate)
    n_days_sim = n / bars_per_day
    total_gross = float(cum_gross[-1])
    total_cost = float(np.sum(total_cost_per_bar))
    total_net = float(cum_pnl[-1])

    daily_pnl = total_net / max(n_days_sim, 1)
    gross_daily = total_gross / max(n_days_sim, 1)
    cost_daily = total_cost / max(n_days_sim, 1)

    # Sharpe-like: daily_pnl / std(daily_pnl)
    # Chunk into pseudo-days
    chunk_size = int(bars_per_day)
    if chunk_size > 0 and n > chunk_size * 2:
        daily_chunks = [
            float(np.sum(net_pnl[i:i+chunk_size]))
            for i in range(0, n - chunk_size, chunk_size)
        ]
        if len(daily_chunks) > 2:
            daily_mean = float(np.mean(daily_chunks))
            daily_std = float(np.std(daily_chunks))
            daily_sharpe = daily_mean / daily_std * np.sqrt(252) if daily_std > 0 else 0.0
            win_days = sum(1 for x in daily_chunks if x > 0)
            pct_win_days = win_days / len(daily_chunks)
        else:
            daily_sharpe = 0.0
            pct_win_days = 0.0
            daily_chunks = []
    else:
        daily_sharpe = 0.0
        pct_win_days = 0.0
        daily_chunks = []

    # Max drawdown
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_dd = float(np.min(drawdown))

    return {
        'total_gross_pnl': total_gross,
        'total_cost': total_cost,
        'total_net_pnl': total_net,
        'n_days_sim': float(n_days_sim),
        'daily_gross_pnl': gross_daily,
        'daily_cost': cost_daily,
        'daily_net_pnl': daily_pnl,
        'daily_sharpe': daily_sharpe,
        'pct_win_days': pct_win_days,
        'max_drawdown': max_dd,
        'n_trades': n_trades,
        'pct_bars_traded': pct_traded,
        'threshold_quantile': threshold_quantile,
        'daily_chunks': daily_chunks,
    }


# ============================================================================
# MAIN EXPERIMENTS
# ============================================================================

def run_feature_selection_experiment(
    scanner, targets, target_names, exclude_features,
    feature_counts=[10, 20, 30, 50, 129],
):
    """Test different numbers of features to find optimal sparsity."""
    results = {}

    # Build base feature mask
    all_features = scanner.feature_names
    keep_mask = np.array([fn not in exclude_features for fn in all_features])
    features_clean = scanner.features[:, keep_mask]
    names_clean = [fn for fn in all_features if fn not in exclude_features]

    for target_name in target_names:
        target = targets[target_name]
        results[target_name] = {}

        for n_feat in feature_counts:
            logger.info(f"\n{'='*60}")
            logger.info(f"Feature selection: {target_name} with top-{n_feat} features")
            logger.info(f"{'='*60}")

            if n_feat >= len(names_clean):
                # Use all features
                feat_subset = features_clean
                names_subset = names_clean
                label = f"all_{len(names_clean)}"
            else:
                # Select top-N features from ranked list
                selected = []
                for fn in TOP_FEATURES_RANKED:
                    if fn in names_clean and fn not in selected:
                        selected.append(fn)
                    if len(selected) >= n_feat:
                        break

                # If we don't have enough from the ranked list, add remaining
                if len(selected) < n_feat:
                    for fn in names_clean:
                        if fn not in selected:
                            selected.append(fn)
                        if len(selected) >= n_feat:
                            break

                feat_idx = [names_clean.index(fn) for fn in selected]
                feat_subset = features_clean[:, feat_idx]
                names_subset = selected
                label = f"top_{n_feat}"

            res = walk_forward_evaluate(
                features=feat_subset,
                target=target,
                day_boundaries=scanner.day_boundaries,
                feature_names=names_subset,
                model_type='lgbm',
                hour_of_day=scanner.hour_of_day,
            )

            if 'error' not in res:
                logger.info(
                    f"  {label}: IC={res['ic']:.4f} t={res['tstat']:.2f} "
                    f"ICIR={res['icir']:.2f} FoldC={res['fold_con']:.0%} "
                    f"[{', '.join(f'{x:+.3f}' for x in res['fold_ics'])}]"
                )
                # Store without numpy arrays
                results[target_name][label] = {
                    k: v for k, v in res.items()
                    if k not in ('predictions', 'actuals')
                }
            else:
                logger.warning(f"  {label}: {res['error']}")
                results[target_name][label] = res

    return results


def run_linear_model_experiment(
    scanner, targets, target_names, exclude_features,
):
    """Test Ridge/Lasso vs LightGBM to check if signal is linear."""
    results = {}

    keep_mask = np.array([fn not in exclude_features for fn in scanner.feature_names])
    features_clean = scanner.features[:, keep_mask]
    names_clean = [fn for fn in scanner.feature_names if fn not in exclude_features]

    # Use top-20 features for cleaner comparison
    selected = []
    for fn in TOP_FEATURES_RANKED[:20]:
        if fn in names_clean:
            selected.append(fn)
    feat_idx = [names_clean.index(fn) for fn in selected]
    feat_top20 = features_clean[:, feat_idx]

    for target_name in target_names:
        target = targets[target_name]
        results[target_name] = {}

        for model_type, alpha in [('ridge', 1.0), ('ridge', 10.0), ('ridge', 100.0), ('lgbm', None)]:
            label = f"{model_type}" + (f"_a{alpha:.0f}" if alpha else "")
            logger.info(f"\n{'='*60}")
            logger.info(f"Linear model: {target_name} / {label} (top-20 features)")
            logger.info(f"{'='*60}")

            kwargs = {
                'features': feat_top20,
                'target': target,
                'day_boundaries': scanner.day_boundaries,
                'feature_names': selected,
                'model_type': model_type if alpha else 'lgbm',
                'hour_of_day': scanner.hour_of_day,
            }
            if alpha:
                kwargs['ridge_alpha'] = alpha

            res = walk_forward_evaluate(**kwargs)

            if 'error' not in res:
                logger.info(
                    f"  {label}: IC={res['ic']:.4f} t={res['tstat']:.2f} "
                    f"ICIR={res['icir']:.2f} FoldC={res['fold_con']:.0%}"
                )
                results[target_name][label] = {
                    k: v for k, v in res.items()
                    if k not in ('predictions', 'actuals')
                }
            else:
                logger.warning(f"  {label}: {res['error']}")
                results[target_name][label] = res

    return results


def run_session_experiment(
    scanner, targets, target_names, exclude_features,
):
    """Test separate models for morning vs afternoon."""
    results = {}

    keep_mask = np.array([fn not in exclude_features for fn in scanner.feature_names])
    features_clean = scanner.features[:, keep_mask]
    names_clean = [fn for fn in scanner.feature_names if fn not in exclude_features]

    for target_name in target_names:
        target = targets[target_name]
        results[target_name] = {}

        for session in [None, 'morning', 'afternoon']:
            label = session or 'all_day'
            logger.info(f"\n{'='*60}")
            logger.info(f"Session: {target_name} / {label}")
            logger.info(f"{'='*60}")

            res = walk_forward_evaluate(
                features=features_clean,
                target=target,
                day_boundaries=scanner.day_boundaries,
                feature_names=names_clean,
                model_type='lgbm',
                hour_of_day=scanner.hour_of_day,
                session_filter=session,
            )

            if 'error' not in res:
                logger.info(
                    f"  {label}: IC={res['ic']:.4f} t={res['tstat']:.2f} "
                    f"ICIR={res['icir']:.2f} FoldC={res['fold_con']:.0%}"
                )
                results[target_name][label] = {
                    k: v for k, v in res.items()
                    if k not in ('predictions', 'actuals')
                }
            else:
                logger.warning(f"  {label}: {res['error']}")
                results[target_name][label] = res

    return results


def run_pnl_experiment(
    scanner, targets, target_names, exclude_features,
    thresholds=[0.0, 0.5, 0.6, 0.7, 0.8, 0.9],
):
    """Run PnL simulation at different trading thresholds."""
    results = {}

    keep_mask = np.array([fn not in exclude_features for fn in scanner.feature_names])
    features_clean = scanner.features[:, keep_mask]
    names_clean = [fn for fn in scanner.feature_names if fn not in exclude_features]

    for target_name in target_names:
        target = targets[target_name]
        results[target_name] = {}

        # Get horizon in seconds from target name
        if target_name == 'ret_1s':
            hz_sec = 1.0
        elif target_name == 'ret_3s':
            hz_sec = 3.0
        elif target_name == 'ret_5s':
            hz_sec = 5.0
        else:
            hz_sec = 3.0

        # First, get predictions from walk-forward
        logger.info(f"\n{'='*60}")
        logger.info(f"PnL simulation: {target_name}")
        logger.info(f"{'='*60}")

        res = walk_forward_evaluate(
            features=features_clean,
            target=target,
            day_boundaries=scanner.day_boundaries,
            feature_names=names_clean,
            model_type='lgbm',
            hour_of_day=scanner.hour_of_day,
        )

        if 'error' in res:
            logger.warning(f"  {target_name}: {res['error']}")
            results[target_name] = {'error': res['error']}
            continue

        logger.info(f"  Base IC={res['ic']:.4f}, got {res['n_preds']:,} predictions")

        # Test different thresholds
        for thr in thresholds:
            pnl = simulate_pnl(
                predictions=res['predictions'],
                actuals=res['actuals'],
                target_horizon_sec=hz_sec,
                threshold_quantile=thr,
            )

            if 'error' not in pnl:
                logger.info(
                    f"  thr={thr:.1f}: gross=${pnl['daily_gross_pnl']:+.2f}/day "
                    f"cost=${pnl['daily_cost']:.2f}/day "
                    f"net=${pnl['daily_net_pnl']:+.2f}/day "
                    f"Sharpe={pnl['daily_sharpe']:.2f} "
                    f"WinDays={pnl['pct_win_days']:.0%} "
                    f"trades={pnl['n_trades']:,} "
                    f"DD=${pnl['max_drawdown']:,.0f}"
                )
                # Remove daily_chunks from saved results (too large)
                pnl_save = {k: v for k, v in pnl.items() if k != 'daily_chunks'}
                results[target_name][f"thr_{thr:.1f}"] = pnl_save
            else:
                results[target_name][f"thr_{thr:.1f}"] = pnl

    return results


def run_regularization_experiment(
    scanner, targets, target_names, exclude_features,
):
    """Test different LightGBM regularization levels."""
    results = {}

    keep_mask = np.array([fn not in exclude_features for fn in scanner.feature_names])
    features_clean = scanner.features[:, keep_mask]
    names_clean = [fn for fn in scanner.feature_names if fn not in exclude_features]

    configs = [
        ('default', {'n_estimators': 500, 'max_depth': 6, 'learning_rate': 0.03,
                      'subsample': 0.8, 'colsample_bytree': 0.7,
                      'reg_alpha': 0.1, 'reg_lambda': 1.0,
                      'min_child_samples': 100, 'verbose': -1, 'n_jobs': -1}),
        ('shallow', {'n_estimators': 300, 'max_depth': 4, 'learning_rate': 0.02,
                      'subsample': 0.7, 'colsample_bytree': 0.5,
                      'reg_alpha': 1.0, 'reg_lambda': 5.0,
                      'min_child_samples': 200, 'verbose': -1, 'n_jobs': -1}),
        ('very_shallow', {'n_estimators': 200, 'max_depth': 3, 'learning_rate': 0.01,
                           'subsample': 0.6, 'colsample_bytree': 0.4,
                           'reg_alpha': 5.0, 'reg_lambda': 10.0,
                           'min_child_samples': 500, 'verbose': -1, 'n_jobs': -1}),
        ('deep', {'n_estimators': 800, 'max_depth': 8, 'learning_rate': 0.05,
                   'subsample': 0.9, 'colsample_bytree': 0.8,
                   'reg_alpha': 0.01, 'reg_lambda': 0.1,
                   'min_child_samples': 50, 'verbose': -1, 'n_jobs': -1}),
    ]

    for target_name in target_names:
        target = targets[target_name]
        results[target_name] = {}

        for config_name, params in configs:
            logger.info(f"\n{'='*60}")
            logger.info(f"Regularization: {target_name} / {config_name}")
            logger.info(f"{'='*60}")

            res = walk_forward_evaluate(
                features=features_clean,
                target=target,
                day_boundaries=scanner.day_boundaries,
                feature_names=names_clean,
                model_type='lgbm',
                lgbm_params=params,
                hour_of_day=scanner.hour_of_day,
            )

            if 'error' not in res:
                logger.info(
                    f"  {config_name}: IC={res['ic']:.4f} t={res['tstat']:.2f} "
                    f"ICIR={res['icir']:.2f} FoldC={res['fold_con']:.0%}"
                )
                results[target_name][config_name] = {
                    k: v for k, v in res.items()
                    if k not in ('predictions', 'actuals')
                }
            else:
                logger.warning(f"  {config_name}: {res['error']}")
                results[target_name][config_name] = res

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--targets', nargs='+', default=['ret_1s', 'ret_3s'],
                        help='Which targets to test')
    parser.add_argument('--pnl-only', action='store_true',
                        help='Only run PnL simulation')
    parser.add_argument('--skip-pnl', action='store_true',
                        help='Skip PnL simulation')
    parser.add_argument('--experiments', nargs='+',
                        default=['feature_selection', 'linear', 'session', 'regularization', 'pnl'],
                        help='Which experiments to run')
    args = parser.parse_args()

    t0_total = time.time()

    logger.info("=" * 75)
    logger.info("MODEL REFINEMENT — Feature Selection, Linear Models, PnL Simulation")
    logger.info(f"  Targets: {args.targets}")
    logger.info(f"  Experiments: {args.experiments}")
    logger.info("=" * 75)

    # Load data
    scanner = MBOAlphaScanner()
    stats = load_feature_cache(scanner)
    if stats is None:
        logger.error("No feature cache found. Run return_multihorizon first.")
        sys.exit(1)

    logger.info(f"Data: {scanner.features.shape[0]:,} snapshots, "
                f"{len(scanner.day_boundaries)-1} days, "
                f"{scanner.features.shape[1]} features")

    # Compute targets
    horizons = {}
    for t in args.targets:
        if t.startswith('ret_'):
            sec = t.replace('ret_', '').replace('s', '')
            horizons[t.replace('ret_', '')] = int(sec)

    targets = compute_return_targets(
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        horizons_sec=horizons if horizons else {'3s': 3},
        include_flow_target=False,
    )

    # Filter to requested targets
    target_names = [t for t in args.targets if t in targets]
    if not target_names:
        logger.error(f"No valid targets found. Available: {list(targets.keys())}")
        sys.exit(1)

    logger.info(f"Targets ready: {target_names}")

    all_results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
        'targets': target_names,
        'experiments': {},
    }

    # ============================================================
    # Experiment 1: Feature Selection
    # ============================================================
    if 'feature_selection' in args.experiments:
        logger.info("\n" + "=" * 75)
        logger.info("EXPERIMENT 1: Feature Selection")
        logger.info("=" * 75)

        fs_results = run_feature_selection_experiment(
            scanner, targets, target_names,
            EXCLUDE_FEATURES_DIRECTION,
            feature_counts=[10, 15, 20, 30, 50, 129],
        )
        all_results['experiments']['feature_selection'] = fs_results

    # ============================================================
    # Experiment 2: Linear Models
    # ============================================================
    if 'linear' in args.experiments:
        logger.info("\n" + "=" * 75)
        logger.info("EXPERIMENT 2: Linear Models (Ridge vs LightGBM)")
        logger.info("=" * 75)

        linear_results = run_linear_model_experiment(
            scanner, targets, target_names,
            EXCLUDE_FEATURES_DIRECTION,
        )
        all_results['experiments']['linear_models'] = linear_results

    # ============================================================
    # Experiment 3: Session-Conditioned Models
    # ============================================================
    if 'session' in args.experiments:
        logger.info("\n" + "=" * 75)
        logger.info("EXPERIMENT 3: Session-Conditioned Models")
        logger.info("=" * 75)

        session_results = run_session_experiment(
            scanner, targets, target_names,
            EXCLUDE_FEATURES_DIRECTION,
        )
        all_results['experiments']['session_models'] = session_results

    # ============================================================
    # Experiment 4: Regularization Sweep
    # ============================================================
    if 'regularization' in args.experiments:
        logger.info("\n" + "=" * 75)
        logger.info("EXPERIMENT 4: Regularization Sweep")
        logger.info("=" * 75)

        reg_results = run_regularization_experiment(
            scanner, targets, target_names,
            EXCLUDE_FEATURES_DIRECTION,
        )
        all_results['experiments']['regularization'] = reg_results

    # ============================================================
    # Experiment 5: PnL Simulation
    # ============================================================
    if 'pnl' in args.experiments:
        logger.info("\n" + "=" * 75)
        logger.info("EXPERIMENT 5: PnL Simulation with Costs")
        logger.info("=" * 75)

        pnl_results = run_pnl_experiment(
            scanner, targets, target_names,
            EXCLUDE_FEATURES_DIRECTION,
        )
        all_results['experiments']['pnl_simulation'] = pnl_results

    # ============================================================
    # Summary
    # ============================================================
    elapsed = time.time() - t0_total
    all_results['elapsed_sec'] = elapsed

    logger.info("\n" + "=" * 75)
    logger.info("COMPREHENSIVE RESULTS SUMMARY")
    logger.info("=" * 75)

    # Feature selection summary
    if 'feature_selection' in all_results['experiments']:
        logger.info("\n--- Feature Selection ---")
        for tgt, configs in all_results['experiments']['feature_selection'].items():
            logger.info(f"\n  {tgt}:")
            for label, res in sorted(configs.items()):
                if 'error' not in res:
                    logger.info(
                        f"    {label:>12s}: IC={res['ic']:.4f} t={res['tstat']:.2f} "
                        f"ICIR={res['icir']:.2f} FoldC={res['fold_con']:.0%} "
                        f"folds={res['n_folds']}"
                    )

    # Linear model summary
    if 'linear_models' in all_results['experiments']:
        logger.info("\n--- Linear vs Nonlinear (top-20 features) ---")
        for tgt, configs in all_results['experiments']['linear_models'].items():
            logger.info(f"\n  {tgt}:")
            for label, res in configs.items():
                if 'error' not in res:
                    logger.info(
                        f"    {label:>12s}: IC={res['ic']:.4f} t={res['tstat']:.2f}"
                    )

    # Session summary
    if 'session_models' in all_results['experiments']:
        logger.info("\n--- Session Models ---")
        for tgt, configs in all_results['experiments']['session_models'].items():
            logger.info(f"\n  {tgt}:")
            for label, res in configs.items():
                if 'error' not in res:
                    logger.info(
                        f"    {label:>12s}: IC={res['ic']:.4f} t={res['tstat']:.2f} "
                        f"FoldC={res['fold_con']:.0%}"
                    )

    # Regularization summary
    if 'regularization' in all_results['experiments']:
        logger.info("\n--- Regularization ---")
        for tgt, configs in all_results['experiments']['regularization'].items():
            logger.info(f"\n  {tgt}:")
            for label, res in configs.items():
                if 'error' not in res:
                    logger.info(
                        f"    {label:>12s}: IC={res['ic']:.4f} t={res['tstat']:.2f} "
                        f"ICIR={res['icir']:.2f}"
                    )

    # PnL summary
    if 'pnl_simulation' in all_results['experiments']:
        logger.info("\n--- PnL Simulation (1 contract ES) ---")
        for tgt, configs in all_results['experiments']['pnl_simulation'].items():
            if isinstance(configs, dict) and 'error' not in configs:
                logger.info(f"\n  {tgt}:")
                for label, res in configs.items():
                    if isinstance(res, dict) and 'error' not in res:
                        logger.info(
                            f"    {label}: net=${res['daily_net_pnl']:+.2f}/day "
                            f"gross=${res['daily_gross_pnl']:+.2f} "
                            f"cost=${res['daily_cost']:.2f} "
                            f"Sharpe={res['daily_sharpe']:.2f} "
                            f"WinDays={res['pct_win_days']:.0%}"
                        )

    # Save results
    out_path = RESULTS_DIR / f"model_refinement_{all_results['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {out_path}")
    logger.info(f"Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
