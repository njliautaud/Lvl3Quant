"""
Spread Prediction — Predict future bid-ask spread changes.

This is a completely different target from direction prediction.
Monetizable via:
  - If spread about to widen  -> don't place limit orders (avoid getting filled at stale price)
  - If spread about to narrow -> aggressively place limit orders (capture spread compression)

Targets tested:
  1. spread_change_{horizon}: future_spread - current_spread (regression)
  2. spread_ratio_{horizon}: future_spread / current_spread (regression)
  3. spread_widen_binary: will spread widen by >0.25 (1 tick) in next 5s? (binary)

Uses ALL features (including vol proxies — these are legitimately correlated with spread).

Walk-forward LightGBM evaluation with:
  - Spread persistence as naive baseline (autocorrelation at various lags)
  - Incremental IC over naive "future_spread = current_spread"
  - Daily PnL simulation of a limit-order strategy conditioned on spread forecast

Usage:
    python alpha_discovery/run_spread_pred.py
    python alpha_discovery/run_spread_pred.py --horizons 1s 5s 10s 30s
    python alpha_discovery/run_spread_pred.py --fast
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
from alpha_discovery.run_return_multihorizon import load_feature_cache

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / 'spread_pred.log', mode='a', encoding='utf-8'),
    ]
)
logger = logging.getLogger("spread_pred")

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE = 0.25
TICK_VALUE = 12.50
ES_POINT_VALUE = 50.0
BARS_PER_SEC = 10       # 100ms bars
SAMPLE_INTERVAL_MS = 100


# ============================================================================
# SPREAD TARGET COMPUTATION
# ============================================================================

def compute_spread_targets(
    features: np.ndarray,
    feature_names: List[str],
    mid_prices: np.ndarray,
    day_boundaries: list,
    horizons_sec: Dict[str, int],
) -> Dict[str, np.ndarray]:
    """
    Compute spread prediction targets.

    Spread is extracted from the 'spread' column in the feature matrix.
    If 'spread' is not available, uses best_ask - best_bid.

    Returns dict of target arrays, all strictly causal (no look-ahead).
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    # Extract current spread from features
    feat_name_to_idx = {n: i for i, n in enumerate(feature_names)}

    if 'spread' in feat_name_to_idx:
        current_spread = features[:, feat_name_to_idx['spread']].astype(np.float64)
        logger.info(f"Using 'spread' feature column (idx={feat_name_to_idx['spread']})")
    elif 'best_ask' in feat_name_to_idx and 'best_bid' in feat_name_to_idx:
        best_ask = features[:, feat_name_to_idx['best_ask']].astype(np.float64)
        best_bid = features[:, feat_name_to_idx['best_bid']].astype(np.float64)
        current_spread = best_ask - best_bid
        logger.info("Using best_ask - best_bid for spread")
    else:
        logger.warning("Cannot find spread features — using 0.25 constant (1 tick)")
        current_spread = np.full(N, TICK_SIZE, dtype=np.float64)

    # Replace zeros and negatives with NaN
    current_spread = np.where(current_spread > 0, current_spread, np.nan)

    spread_stats = {
        'mean': float(np.nanmean(current_spread)),
        'std': float(np.nanstd(current_spread)),
        'median': float(np.nanmedian(current_spread)),
        'min': float(np.nanmin(current_spread)),
        'max': float(np.nanmax(current_spread)),
    }
    logger.info(f"Spread stats: mean={spread_stats['mean']:.4f} std={spread_stats['std']:.4f} "
                f"median={spread_stats['median']:.4f}")

    # Helper: NaN-fill bars whose forward window crosses a day boundary
    def nan_fill_crossings(arr: np.ndarray, steps: int) -> np.ndarray:
        if n_days > 1:
            for d in range(n_days - 1):
                day_end = day_boundaries[d + 1]
                nan_start = max(day_boundaries[d], day_end - steps)
                arr[nan_start:day_end] = np.nan
        return arr

    targets = {}

    # ---- 1. Spread change and ratio at multiple horizons ----
    for hz_name, hz_sec in horizons_sec.items():
        steps = int(hz_sec * BARS_PER_SEC)
        if steps >= N:
            logger.warning(f"  {hz_name} ({steps} steps) >= N={N}, skipping")
            continue

        # Future spread: strictly causal (future is looked-up, but we're computing targets)
        future_spread = np.empty(N, dtype=np.float64)
        future_spread[:N - steps] = current_spread[steps:]
        future_spread[N - steps:] = np.nan

        # NaN-fill day boundary crossings
        future_spread = nan_fill_crossings(future_spread, steps)

        # Spread change: future - current
        spread_change = future_spread - current_spread
        spread_change = spread_change.astype(np.float32)
        targets[f'spread_change_{hz_name}'] = spread_change

        # Spread ratio: future / current (only where current > 0)
        spread_ratio = np.where(
            current_spread > TICK_SIZE * 0.1,
            future_spread / current_spread,
            np.nan
        ).astype(np.float32)
        targets[f'spread_ratio_{hz_name}'] = spread_ratio

        valid_c = np.isfinite(spread_change).sum()
        valid_r = np.isfinite(spread_ratio).sum()
        mean_c = float(np.nanmean(np.abs(spread_change)))
        logger.info(
            f"  spread_change_{hz_name}: {valid_c:,} valid, mean |change|={mean_c:.5f}  |  "
            f"spread_ratio_{hz_name}: {valid_r:,} valid"
        )

    # ---- 2. Binary: will spread widen by >= 1 tick in next 5s? ----
    hz_5s_steps = int(5 * BARS_PER_SEC)
    if hz_5s_steps < N:
        future_spread_5s = np.empty(N, dtype=np.float64)
        future_spread_5s[:N - hz_5s_steps] = current_spread[hz_5s_steps:]
        future_spread_5s[N - hz_5s_steps:] = np.nan
        future_spread_5s = nan_fill_crossings(future_spread_5s, hz_5s_steps)

        widen_by_one_tick = np.where(
            np.isfinite(future_spread_5s) & np.isfinite(current_spread),
            (future_spread_5s - current_spread >= TICK_SIZE * 0.9).astype(np.float32),
            np.nan,
        ).astype(np.float32)
        targets['spread_widen_binary_5s'] = widen_by_one_tick
        pct_widen = float(np.nanmean(widen_by_one_tick))
        logger.info(f"  spread_widen_binary_5s: {np.isfinite(widen_by_one_tick).sum():,} valid, "
                    f"{pct_widen:.1%} widen events")

    return targets, current_spread, spread_stats


# ============================================================================
# SPREAD PERSISTENCE (NAIVE BASELINE)
# ============================================================================

def compute_spread_persistence(
    current_spread: np.ndarray,
    day_boundaries: list,
    max_lag_bars: int = 300,
) -> Dict[str, float]:
    """
    Compute autocorrelation of spread at various lags.

    This is the naive baseline: "future_spread = current_spread".
    High persistence = model must do better than this.

    Returns: {lag_bars: autocorrelation} for lags 1..max_lag_bars
    """
    N = len(current_spread)
    n_days = len(day_boundaries) - 1

    # Build mask of within-day valid pairs at each lag
    persistence = {}
    check_lags = [1, 5, 10, 20, 50, 100, 150, 200, 300]

    for lag in check_lags:
        if lag >= N:
            continue

        future = current_spread[lag:]
        current = current_spread[:N - lag]

        valid = np.isfinite(future) & np.isfinite(current)

        # Mask out pairs that cross day boundaries
        if n_days > 1:
            day_boundary_set = set()
            for d in range(n_days):
                day_boundary_set.update(range(
                    max(0, day_boundaries[d + 1] - lag),
                    day_boundaries[d + 1]
                ))
            # Create cross-day mask
            cross_day = np.zeros(N - lag, dtype=bool)
            for d in range(n_days):
                d_end = day_boundaries[d + 1]
                nan_start = max(0, d_end - lag)
                cross_day[nan_start:d_end] = True

            valid = valid & ~cross_day[:N - lag]

        if valid.sum() < 50:
            continue

        try:
            corr = float(spearmanr(current[valid], future[valid])[0])
        except Exception:
            corr = float('nan')

        persistence[f'lag_{lag}bars'] = {
            'lag_bars': lag,
            'lag_sec': lag / BARS_PER_SEC,
            'autocorr': corr,
            'n_pairs': int(valid.sum()),
        }

    return persistence


# ============================================================================
# INCREMENTAL IC VS NAIVE BASELINE
# ============================================================================

def compute_incremental_ic(
    preds: np.ndarray,
    actuals: np.ndarray,
    current_spread_at_pred_bars: np.ndarray,
) -> dict:
    """
    Compute incremental IC of model vs naive persistence baseline.

    Naive: predict spread_change = 0 (i.e., future_spread = current_spread)
    Model: LightGBM predictions

    For spread CHANGE targets, naive prediction is 0.
    For spread RATIO targets, naive prediction is 1.0.
    """
    valid = np.isfinite(preds) & np.isfinite(actuals)
    p, a = preds[valid], actuals[valid]

    if len(p) < 50:
        return {'error': 'Too few valid predictions'}

    model_ic = float(spearmanr(p, a)[0])

    # Naive baseline: predict 0 change
    naive_preds = np.zeros_like(p)
    try:
        naive_ic = float(spearmanr(naive_preds, a)[0])
    except Exception:
        naive_ic = 0.0

    # Mean-absolute-error comparison
    model_mae = float(np.mean(np.abs(p - a)))
    naive_mae = float(np.mean(np.abs(naive_preds - a)))
    mae_improvement_pct = (naive_mae - model_mae) / naive_mae * 100 if naive_mae > 0 else 0.0

    return {
        'model_ic': model_ic,
        'naive_ic': naive_ic,
        'incremental_ic': model_ic - naive_ic,
        'model_mae': model_mae,
        'naive_mae': naive_mae,
        'mae_improvement_pct': mae_improvement_pct,
        'n_valid': int(valid.sum()),
    }


# ============================================================================
# WALK-FORWARD EVALUATION FOR SPREAD TARGETS
# ============================================================================

def walk_forward_spread(
    features: np.ndarray,
    target: np.ndarray,
    day_boundaries: list,
    feature_names: List[str],
    is_binary: bool = False,
    min_train_days: int = 3,
) -> dict:
    """
    Walk-forward LightGBM evaluation for spread prediction targets.

    Uses ALL features (vol proxies allowed — they're genuinely correlated with spreads).
    Returns: IC, fold ICs, top features, and full predictions array.
    """
    import lightgbm as lgb

    n_days = len(day_boundaries) - 1
    if n_days < min_train_days + 1:
        return {'error': f'Need {min_train_days + 1} days, have {n_days}'}

    n_features = features.shape[1]

    if is_binary:
        params = {
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'min_child_samples': 50,
            'verbose': -1,
            'n_jobs': -1,
            'objective': 'binary',
            'metric': 'auc',
        }
    else:
        params = {
            'n_estimators': 500,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.05,
            'reg_lambda': 0.5,
            'min_child_samples': 50,
            'verbose': -1,
            'n_jobs': -1,
            'objective': 'regression',
            'metric': 'rmse',
        }

    all_preds = []
    all_actuals = []
    fold_ics = []
    fold_metrics = []
    feature_importance = np.zeros(n_features)

    last_log = time.time()

    for test_day in range(min_train_days, n_days):
        train_start = day_boundaries[0]
        train_end = day_boundaries[test_day]       # purge gap: skip test_day-1 -> test_day
        test_start = day_boundaries[test_day]
        test_end = day_boundaries[test_day + 1]

        X_train = features[train_start:train_end]
        y_train = target[train_start:train_end]
        X_test = features[test_start:test_end]
        y_test = target[test_start:test_end]

        train_valid = np.isfinite(y_train)
        test_valid = np.isfinite(y_test)

        if train_valid.sum() < 500 or test_valid.sum() < 50:
            continue

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]
        X_te = X_test[test_valid]
        y_te = y_test[test_valid]

        if is_binary:
            y_tr_fit = (y_tr > 0.5).astype(int)
        else:
            y_tr_fit = y_tr

        split = int(len(X_tr) * 0.8)
        try:
            if is_binary:
                model = lgb.LGBMClassifier(**params)
            else:
                model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr[:split], y_tr_fit[:split],
                eval_set=[(X_tr[split:], y_tr_fit[split:])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        except Exception as e:
            logger.warning(f"  Day {test_day} training failed: {e}")
            continue

        if is_binary:
            preds = model.predict_proba(X_te)[:, 1] if hasattr(model, 'predict_proba') else model.predict(X_te)
        else:
            preds = model.predict(X_te)

        all_preds.append(preds)
        all_actuals.append(y_te)

        if len(preds) > 10:
            try:
                ic_fold = float(spearmanr(preds, y_te)[0])
                if np.isfinite(ic_fold):
                    fold_ics.append(ic_fold)
                    fold_metrics.append({
                        'day': test_day,
                        'ic': ic_fold,
                        'n_samples': len(preds),
                        'train_size': int(train_valid.sum()),
                    })
            except Exception:
                pass

        if hasattr(model, 'feature_importances_'):
            feature_importance += model.feature_importances_

        del model

        # Periodic progress log
        now = time.time()
        if now - last_log > 30:
            logger.info(f"    Day {test_day}/{n_days - 1} done, "
                        f"{len(fold_ics)} folds so far, "
                        f"mean IC={np.mean(fold_ics):.5f}" if fold_ics else "")
            last_log = now

        gc.collect()

    if not all_preds:
        return {'error': 'No valid predictions'}

    predictions = np.concatenate(all_preds)
    actuals = np.concatenate(all_actuals)

    valid = np.isfinite(predictions) & np.isfinite(actuals)
    p, a = predictions[valid], actuals[valid]

    if len(p) < 50:
        return {'error': f'Too few predictions: {len(p)}'}

    ic = float(spearmanr(p, a)[0])

    # ICIR and t-stat
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
    else:
        ic_mean, ic_std = ic, 0.0
        icir, tstat, pvalue = 0.0, 0.0, 1.0

    n_positive_folds = sum(1 for x in fold_ics if x > 0)
    fold_consistency = n_positive_folds / len(fold_ics) if fold_ics else 0.0

    # Top features
    top_feat_idx = np.argsort(feature_importance)[::-1][:20]
    top_features = [
        (feature_names[i], float(feature_importance[i]))
        for i in top_feat_idx if feature_importance[i] > 0
    ]

    passed = abs(ic) > 0.02 and abs(tstat) > 2.0 and fold_consistency > 0.60

    return {
        # Core metrics
        'ic': ic,
        'ic_mean': ic_mean,
        'ic_std': ic_std,
        'icir': icir,
        'tstat': tstat,
        'pvalue': pvalue,
        'fold_consistency': fold_consistency,
        'n_positive_folds': n_positive_folds,
        'n_folds': len(fold_ics),
        'fold_ics': [float(x) for x in fold_ics],
        'fold_metrics': fold_metrics,
        'top_features': top_features,
        'n_predictions': int(valid.sum()),
        'passed': passed,
        # Return full predictions for downstream analysis
        '_predictions': predictions,
        '_actuals': actuals,
    }


# ============================================================================
# LIMIT ORDER PNL SIMULATION
# ============================================================================

def simulate_limit_order_strategy(
    predictions: np.ndarray,
    actuals: np.ndarray,
    current_spread: np.ndarray,
    mid_prices: np.ndarray,
    day_boundaries: list,
    pred_bar_indices: Optional[np.ndarray] = None,
    narrow_threshold: float = -0.0625,  # spread narrows by >0.25/4 = half tick
    n_days_for_rate: int = None,
) -> dict:
    """
    Simulate a limit order strategy conditioned on spread forecast.

    Logic:
      - When model predicts spread will NARROW by > narrow_threshold:
        place a limit order at mid; if spread does narrow you capture improvement
      - Simplified: if predicted_change < narrow_threshold AND actual_change < 0:
        you successfully placed a limit order at a better effective price
      - PnL per favorable fill = |actual_spread_improvement| * ES_POINT_VALUE / 2

    This is a simplified simulation since we don't have actual order fill data.

    Returns per-day and aggregate stats.
    """
    N_pred = len(predictions)
    if N_pred < 100:
        return {'error': 'Too few predictions'}

    valid = np.isfinite(predictions) & np.isfinite(actuals)
    p, a = predictions[valid], actuals[valid]

    if len(p) < 50:
        return {'error': 'Too few valid pairs'}

    # Count favorable opportunities: model correctly predicts spread narrowing
    pred_narrow = p < narrow_threshold          # model predicts narrowing
    actual_narrow = a < -TICK_SIZE * 0.1        # spread did narrow by >= 0.5 tick

    n_pred_narrow = int(pred_narrow.sum())
    n_correct_narrow = int((pred_narrow & actual_narrow).sum())
    n_missed_narrow = int((~pred_narrow & actual_narrow).sum())

    precision = n_correct_narrow / n_pred_narrow if n_pred_narrow > 0 else 0.0
    recall = n_correct_narrow / (n_correct_narrow + n_missed_narrow) if (n_correct_narrow + n_missed_narrow) > 0 else 0.0

    # Estimated PnL per favorable fill (half-spread capture)
    # When spread narrows by X, a limit order at mid captures X/2 in edge
    favorable_changes = a[pred_narrow & actual_narrow]
    mean_spread_improvement = float(np.mean(np.abs(favorable_changes))) if len(favorable_changes) > 0 else 0.0
    pnl_per_fill = mean_spread_improvement / 2.0 * ES_POINT_VALUE  # half spread * $50/pt

    # Daily rate
    if n_days_for_rate is None:
        n_days_for_rate = len(day_boundaries) - 1
    daily_correct_narrow = n_correct_narrow / n_days_for_rate if n_days_for_rate > 0 else 0.0
    daily_pnl_estimate = daily_correct_narrow * pnl_per_fill

    return {
        'narrow_threshold': narrow_threshold,
        'n_pred_narrow': n_pred_narrow,
        'n_correct_narrow': n_correct_narrow,
        'n_missed_narrow': n_missed_narrow,
        'precision': round(precision, 4),
        'recall': round(recall, 4),
        'f1': round(2 * precision * recall / (precision + recall), 4) if (precision + recall) > 0 else 0.0,
        'mean_spread_improvement': round(mean_spread_improvement, 5),
        'pnl_per_fill_estimate': round(pnl_per_fill, 2),
        'daily_correct_fills': round(daily_correct_narrow, 1),
        'daily_pnl_estimate': round(daily_pnl_estimate, 2),
        'n_valid': int(valid.sum()),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Spread prediction experiment')
    parser.add_argument('--horizons', nargs='+',
                        default=['1s', '5s', '10s', '30s'],
                        help='Spread prediction horizons (default: 1s 5s 10s 30s)')
    parser.add_argument('--fast', action='store_true',
                        help='Faster run: skip binary target and ratio targets')
    parser.add_argument('--min-train-days', type=int, default=3)
    args = parser.parse_args()

    logger.info("=" * 75)
    logger.info("SPREAD PREDICTION EXPERIMENT")
    logger.info(f"  Horizons: {args.horizons}")
    logger.info(f"  Fast mode: {args.fast}")
    logger.info("=" * 75)

    t_start = time.time()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    stats = load_feature_cache(scanner)
    if stats is None:
        logger.info("No feature cache, computing from scratch (slow)...")
        stats = scanner.load_from_cache()

    logger.info(f"Data: {len(scanner.mid_prices):,} snapshots, "
                f"{len(scanner.day_boundaries) - 1} days, "
                f"{len(scanner.feature_names)} features")

    # -----------------------------------------------------------------------
    # Compute spread targets (uses ALL features)
    # -----------------------------------------------------------------------
    logger.info("\nComputing spread targets...")
    horizons_sec = {}
    for hz in args.horizons:
        if hz.endswith('s'):
            horizons_sec[hz] = int(hz[:-1])
        elif hz.endswith('m'):
            horizons_sec[hz] = int(hz[:-1]) * 60
        else:
            horizons_sec[hz] = int(hz)

    targets, current_spread, spread_stats = compute_spread_targets(
        features=scanner.features,
        feature_names=scanner.feature_names,
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        horizons_sec=horizons_sec,
    )
    logger.info(f"Computed {len(targets)} spread targets: {list(targets.keys())}")

    # -----------------------------------------------------------------------
    # Spread persistence analysis (naive baseline)
    # -----------------------------------------------------------------------
    logger.info("\nComputing spread autocorrelation (naive persistence baseline)...")
    persistence = compute_spread_persistence(
        current_spread=current_spread,
        day_boundaries=scanner.day_boundaries,
        max_lag_bars=300,
    )
    logger.info("Spread autocorrelation by lag:")
    for lag_key, lag_data in persistence.items():
        logger.info(
            f"  lag={lag_data['lag_bars']:4d} bars ({lag_data['lag_sec']:5.1f}s): "
            f"autocorr={lag_data['autocorr']:+.5f} n={lag_data['n_pairs']:,}"
        )

    # -----------------------------------------------------------------------
    # Walk-forward evaluation for each target
    # -----------------------------------------------------------------------
    results = {}
    last_log_time = time.time()

    # Determine which targets to run
    targets_to_run = {}
    for tgt_name, tgt_arr in targets.items():
        if args.fast:
            # Only run spread_change in fast mode, skip ratio and binary
            if 'change' not in tgt_name:
                continue
        targets_to_run[tgt_name] = tgt_arr

    total = len(targets_to_run)
    logger.info(f"\nRunning {total} targets with ALL {len(scanner.feature_names)} features")

    for idx, (tgt_name, tgt_arr) in enumerate(targets_to_run.items()):
        logger.info(f"\n{'='*65}")
        logger.info(f"[{idx+1}/{total}] Evaluating: {tgt_name}")
        logger.info(f"{'='*65}")

        is_binary = 'binary' in tgt_name

        t0 = time.time()
        result = walk_forward_spread(
            features=scanner.features,
            target=tgt_arr,
            day_boundaries=scanner.day_boundaries,
            feature_names=scanner.feature_names,
            is_binary=is_binary,
            min_train_days=args.min_train_days,
        )
        elapsed = time.time() - t0
        result['elapsed_sec'] = elapsed

        if 'error' in result:
            logger.info(f"  ERROR: {result['error']}")
            results[tgt_name] = result
            continue

        status = "ALPHA" if result['passed'] else "---"
        logger.info(
            f"  [{status}] IC={result['ic']:.5f} ICIR={result['icir']:.2f} "
            f"t={result['tstat']:.2f} FoldC={result['fold_consistency']:.0%} "
            f"({elapsed:.0f}s)"
        )
        if result.get('top_features'):
            top3 = ", ".join(f"{n}={v:.0f}" for n, v in result['top_features'][:3])
            logger.info(f"  Top features: {top3}")

        # Incremental IC vs naive
        preds = result.pop('_predictions', None)
        acts = result.pop('_actuals', None)
        if preds is not None and acts is not None:
            incr = compute_incremental_ic(preds, acts, current_spread[:len(preds)])
            result['incremental_ic_vs_naive'] = incr
            logger.info(
                f"  vs naive: model_IC={incr.get('model_ic', 0):.5f} "
                f"naive_IC={incr.get('naive_ic', 0):.5f} "
                f"incremental={incr.get('incremental_ic', 0):.5f} "
                f"MAE_improvement={incr.get('mae_improvement_pct', 0):.2f}%"
            )

            # Limit order simulation for spread_change targets
            if 'change' in tgt_name and not is_binary:
                lo_sim = simulate_limit_order_strategy(
                    predictions=preds,
                    actuals=acts,
                    current_spread=current_spread[:len(preds)],
                    mid_prices=scanner.mid_prices,
                    day_boundaries=scanner.day_boundaries,
                    n_days_for_rate=len(scanner.day_boundaries) - 1,
                )
                result['limit_order_simulation'] = lo_sim
                logger.info(
                    f"  LimitOrder sim: precision={lo_sim.get('precision', 0):.1%} "
                    f"recall={lo_sim.get('recall', 0):.1%} "
                    f"daily_pnl_est=${lo_sim.get('daily_pnl_estimate', 0):.2f} "
                    f"per_fill=${lo_sim.get('pnl_per_fill_estimate', 0):.2f}"
                )

        results[tgt_name] = result

        now = time.time()
        if now - last_log_time > 30:
            logger.info(f"[Progress] {idx+1}/{total} targets done, "
                        f"{(now - t_start)/60:.1f} min elapsed")
            last_log_time = now

        gc.collect()

    # -----------------------------------------------------------------------
    # Scoreboard
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 95)
    logger.info("SPREAD PREDICTION SCOREBOARD")
    logger.info("=" * 95)
    logger.info(
        f"{'Target':<30s} {'IC':>8s} {'ICIR':>6s} {'t':>6s} "
        f"{'FoldC':>6s} {'Passed':>7s} {'IncIC':>8s}"
    )
    logger.info("-" * 95)

    sorted_results = sorted(
        [(k, v) for k, v in results.items() if 'error' not in v],
        key=lambda x: abs(x[1].get('ic', 0)),
        reverse=True,
    )
    for tgt_name, r in sorted_results:
        incr_ic = r.get('incremental_ic_vs_naive', {}).get('incremental_ic', float('nan'))
        incr_str = f"{incr_ic:+.5f}" if np.isfinite(incr_ic) else "  N/A  "
        passed = "YES" if r.get('passed') else "no"
        logger.info(
            f"  {tgt_name:<28s} {r['ic']:>8.5f} {r['icir']:>6.2f} {r['tstat']:>6.2f} "
            f"{r['fold_consistency']:>6.0%} {passed:>7s} {incr_str:>8s}"
        )

    logger.info("=" * 95)

    # -----------------------------------------------------------------------
    # Per-fold IC for top targets
    # -----------------------------------------------------------------------
    logger.info("\nPER-FOLD IC EVOLUTION (top 5):")
    for tgt_name, r in sorted_results[:5]:
        if r.get('fold_ics'):
            ics_str = " ".join(f"{x:+.4f}" for x in r['fold_ics'])
            logger.info(f"  {tgt_name}: [{ics_str}]")

    # -----------------------------------------------------------------------
    # Top features across all spread targets
    # -----------------------------------------------------------------------
    logger.info("\nTOP FEATURES FOR SPREAD PREDICTION:")
    feat_agg = {}
    for tgt_name, r in results.items():
        if 'error' in r:
            continue
        for fname, fimp in r.get('top_features', []):
            feat_agg[fname] = feat_agg.get(fname, 0) + fimp
    for fname, fimp in sorted(feat_agg.items(), key=lambda x: x[1], reverse=True)[:20]:
        logger.info(f"  {fname:<35s}: {fimp:>10.0f}")

    # -----------------------------------------------------------------------
    # Spread persistence summary
    # -----------------------------------------------------------------------
    logger.info("\nSPREAD PERSISTENCE (autocorrelation):")
    for lag_key, lag_data in persistence.items():
        logger.info(
            f"  {lag_data['lag_sec']:6.1f}s: autocorr={lag_data['autocorr']:+.5f}"
        )

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = RESULTS_DIR / f"spread_pred_{timestamp}.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, float) and not np.isfinite(obj):
            return str(obj)
        return obj

    # Remove large numpy arrays before saving
    clean_results = {}
    for k, v in results.items():
        clean_v = {kk: vv for kk, vv in v.items() if kk not in ('_predictions', '_actuals')}
        clean_results[k] = clean_v

    output = {
        'timestamp': timestamp,
        'horizons_tested': args.horizons,
        'n_days': len(scanner.day_boundaries) - 1,
        'n_snapshots': len(scanner.mid_prices),
        'n_features': len(scanner.feature_names),
        'spread_stats': spread_stats,
        'spread_persistence': make_serializable(persistence),
        'results': make_serializable(clean_results),
        'elapsed_sec': time.time() - t_start,
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {result_file}")
    logger.info(f"Total elapsed: {(time.time() - t_start)/60:.1f} min")

    # Final verdict
    winners = [(k, v) for k, v in results.items() if v.get('passed', False)]
    logger.info(f"\nSPREAD PREDICTION VERDICT:")
    if winners:
        logger.info(f"  ALPHA FOUND in {len(winners)} spread targets:")
        for tgt_name, r in winners:
            lo = r.get('limit_order_simulation', {})
            logger.info(
                f"    {tgt_name}: IC={r['ic']:.5f} t={r['tstat']:.2f} "
                f"daily_pnl_est=${lo.get('daily_pnl_estimate', 0):.2f}"
            )
    else:
        best = sorted_results[0] if sorted_results else None
        if best:
            logger.info(
                f"  No targets pass all criteria. Best: {best[0]} IC={best[1]['ic']:.5f} "
                f"t={best[1]['tstat']:.2f}"
            )
        else:
            logger.info("  No valid results.")


if __name__ == '__main__':
    main()
