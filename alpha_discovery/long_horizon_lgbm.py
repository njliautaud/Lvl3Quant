"""
Long-Horizon LightGBM Walk-Forward Training
=============================================
Train LightGBM on 340 MBO features targeting LONGER forward returns (5min, 15min, 1hr).

Key hypothesis: Raw feature IC=0.10 at 1hr. Trained model might boost to IC=0.15-0.20.
At 1hr hold, cost ratio is only 3% of average move → could be profitable if IC high enough.

Walk-forward: expanding window, train on all prior days, predict next day.
Save predictions as NPZ files for MBO sim testing.

Usage:
    python alpha_discovery/long_horizon_lgbm.py --horizon 5m --workers 12
    python alpha_discovery/long_horizon_lgbm.py --horizon 1h --workers 12
    python alpha_discovery/long_horizon_lgbm.py --horizon all --workers 12
"""

import sys
import time
import json
import logging
import argparse
import os
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

# Setup paths
SCRIPT_DIR = Path(__file__).parent
LVL3_ROOT = SCRIPT_DIR.parent
FEATURE_CACHE = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
OUTPUT_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR = SCRIPT_DIR / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Bars per second (10 bars/sec for MBO data)
BARS_PER_SEC = 10

# Horizon configs
HORIZON_CONFIGS = {
    '5m':  {'bars': 3000,  'label': '5min',  'subsample_step': 30},   # every 3s
    '15m': {'bars': 9000,  'label': '15min', 'subsample_step': 100},  # every 10s
    '30m': {'bars': 18000, 'label': '30min', 'subsample_step': 200},  # every 20s
    '1h':  {'bars': 36000, 'label': '1hr',   'subsample_step': 300},  # every 30s
}

# Minimum training days before we start predicting
MIN_TRAIN_DAYS = 15

# Features to EXCLUDE: absolute price levels that leak trend info at longer horizons
# 0=mid, 3=microprice, 8=best_bid, 9=best_ask
EXCLUDE_FEATURES = [0, 3, 8, 9]

# Setup logging
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = RESULTS_DIR / f"long_horizon_lgbm_{_ts}.log"
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers[:]:
    _root.removeHandler(_h)
_fmt = logging.Formatter('%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
_fh = logging.FileHandler(str(_log_file), mode='w')
_fh.setFormatter(_fmt)
_root.addHandler(_fh)
_ch = logging.StreamHandler()
_ch.setFormatter(_fmt)
_root.addHandler(_ch)

logger = logging.getLogger('long_lgbm')


def discover_days() -> List[Tuple[str, Path]]:
    """Find available days with feature caches."""
    days = []
    for f in sorted(FEATURE_CACHE.glob("*_mbo_features.npz")):
        date_str = f.stem.replace("_mbo_features", "")
        days.append((date_str, f))
    return days


def load_day(fpath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load one day of feature data + mid prices, excluding price-level features."""
    data = np.load(str(fpath))
    raw_features = data['mbo_features']  # (N, 340)
    mid_prices = raw_features[:, 0].copy()  # col 0 = mid

    # Exclude price-level features that leak trend info at longer horizons
    keep_cols = [i for i in range(raw_features.shape[1]) if i not in EXCLUDE_FEATURES]
    features = raw_features[:, keep_cols]  # (N, 336)
    del raw_features
    return features, mid_prices


def compute_forward_return(mid_prices: np.ndarray, horizon_bars: int) -> np.ndarray:
    """Compute forward return in ticks at given horizon."""
    n = len(mid_prices)
    fwd_ret = np.full(n, np.nan)
    tick_size = 0.25  # ES tick
    for i in range(n - horizon_bars):
        if mid_prices[i] > 0 and mid_prices[i + horizon_bars] > 0:
            fwd_ret[i] = (mid_prices[i + horizon_bars] - mid_prices[i]) / tick_size
    return fwd_ret


def prepare_training_data(features: np.ndarray, fwd_ret: np.ndarray,
                          subsample_step: int) -> Tuple[np.ndarray, np.ndarray]:
    """Prepare X, y for training with subsampling to reduce correlation."""
    # Mask: valid forward return, no NaN features
    valid = np.isfinite(fwd_ret)
    # Skip first 5000 bars (warm-up for rolling features)
    valid[:5000] = False
    # Skip last horizon_bars (no forward return)

    # Subsample to reduce autocorrelation
    indices = np.arange(len(features))
    subsample_mask = (indices % subsample_step) == 0
    mask = valid & subsample_mask

    X = features[mask]
    y = fwd_ret[mask]

    # Remove any remaining NaN/inf
    finite_mask = np.all(np.isfinite(X), axis=1) & np.isfinite(y)
    X = X[finite_mask]
    y = y[finite_mask]

    return X, y


def train_and_predict_fold(train_days_data, test_day_data, horizon_cfg, lgbm_params):
    """Train LightGBM on train_days, predict on test_day."""
    import lightgbm as lgb

    subsample_step = horizon_cfg['subsample_step']

    # Combine training data
    X_trains = []
    y_trains = []
    for features, fwd_ret in train_days_data:
        X, y = prepare_training_data(features, fwd_ret, subsample_step)
        if len(X) > 0:
            X_trains.append(X)
            y_trains.append(y)

    if not X_trains:
        return None, None

    X_train = np.vstack(X_trains)
    y_train = np.concatenate(y_trains)

    # Cap training size to avoid memory issues (max 2M samples)
    if len(X_train) > 2_000_000:
        idx = np.random.choice(len(X_train), 2_000_000, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    # Normalize target to zero mean, unit std
    y_mean = np.mean(y_train)
    y_std = np.std(y_train)
    if y_std < 1e-10:
        return None, None
    y_train_norm = (y_train - y_mean) / y_std

    # Train
    train_data = lgb.Dataset(X_train, label=y_train_norm, free_raw_data=True)

    model = lgb.train(
        lgbm_params,
        train_data,
        num_boost_round=200,
        valid_sets=[train_data],
        callbacks=[lgb.log_evaluation(0)],  # suppress output
    )

    # Predict on test day (ALL bars, not subsampled)
    test_features, test_fwd_ret = test_day_data
    valid_mask = np.all(np.isfinite(test_features), axis=1)

    predictions = np.zeros(len(test_features))
    if np.any(valid_mask):
        preds_raw = model.predict(test_features[valid_mask])
        # Convert back to tick space
        predictions[valid_mask] = preds_raw * y_std + y_mean

    # Compute IC on subsampled (for fair comparison)
    X_test, y_test = prepare_training_data(test_features, test_fwd_ret, subsample_step)
    if len(X_test) > 100:
        p_test = model.predict(X_test) * y_std + y_mean
        from scipy.stats import pearsonr
        ic, _ = pearsonr(p_test, y_test)
    else:
        ic = 0.0

    # Feature importance
    importance = model.feature_importance(importance_type='gain')
    top_features = np.argsort(importance)[::-1][:10]

    return predictions, ic, top_features, importance


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=str, default='5m',
                       choices=list(HORIZON_CONFIGS.keys()) + ['all'])
    parser.add_argument('--workers', type=int, default=10)
    parser.add_argument('--min-train-days', type=int, default=MIN_TRAIN_DAYS)
    args = parser.parse_args()

    horizons = list(HORIZON_CONFIGS.keys()) if args.horizon == 'all' else [args.horizon]

    days = discover_days()
    # Load feature names for logging
    all_feature_names = None
    feature_names = None
    try:
        sys.path.insert(0, str(LVL3_ROOT))
        from alpha_discovery.mbo_features import get_feature_names
        all_feature_names = get_feature_names()
        keep_cols = [i for i in range(len(all_feature_names)) if i not in EXCLUDE_FEATURES]
        feature_names = [all_feature_names[i] for i in keep_cols]
    except Exception:
        keep_cols = [i for i in range(340) if i not in EXCLUDE_FEATURES]

    logger.info(f"Found {len(days)} days of MBO features")
    logger.info(f"Horizons to test: {horizons}")
    logger.info(f"Workers: {args.workers}")
    if all_feature_names:
        logger.info(f"Excluded features: {[all_feature_names[i] for i in EXCLUDE_FEATURES]}")
    else:
        logger.info(f"Excluded feature indices: {EXCLUDE_FEATURES}")
    logger.info(f"Using {len(keep_cols)} features (was 340)")

    # LightGBM params — conservative to avoid overfitting
    lgbm_params = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 6,
        'min_child_samples': 200,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': args.workers,
        'seed': 42,
    }

    for horizon_key in horizons:
        cfg = HORIZON_CONFIGS[horizon_key]
        horizon_bars = cfg['bars']
        subsample_step = cfg['subsample_step']

        logger.info(f"\n{'='*70}")
        logger.info(f"HORIZON: {cfg['label']} ({horizon_bars} bars = {horizon_bars/BARS_PER_SEC:.0f}s)")
        logger.info(f"{'='*70}")

        # Memory-efficient: pre-compute subsampled X,y per day (much smaller than raw)
        logger.info("Pre-computing subsampled training data per day...")
        day_train_data = []  # (date, X_sub, y_sub) — subsampled for training
        day_paths = []  # keep paths for loading full test-day features
        t0 = time.time()

        for date_str, fpath in days:
            try:
                features, mid_prices = load_day(fpath)
                fwd_ret = compute_forward_return(mid_prices, horizon_bars)
                X_sub, y_sub = prepare_training_data(features, fwd_ret, subsample_step)
                day_train_data.append((date_str, X_sub, y_sub))
                day_paths.append((date_str, fpath))
                del features, mid_prices, fwd_ret
            except Exception as e:
                logger.error(f"  {date_str}: Failed to load: {e}")

        total_samples = sum(len(d[1]) for d in day_train_data)
        elapsed = time.time() - t0
        logger.info(f"Pre-computed {len(day_train_data)} days, {total_samples:,} total samples ({elapsed:.1f}s)")

        # Walk-forward training
        ics = []
        sig_name = f"lgbm_{horizon_key}"

        for test_idx in range(args.min_train_days, len(day_train_data)):
            test_date = day_train_data[test_idx][0]

            # Combine training data from all prior days
            X_trains = [day_train_data[i][1] for i in range(test_idx) if len(day_train_data[i][1]) > 0]
            y_trains = [day_train_data[i][2] for i in range(test_idx) if len(day_train_data[i][2]) > 0]

            if not X_trains:
                logger.warning(f"  [{test_idx+1}/{len(day_train_data)}] {test_date}: SKIPPED")
                continue

            X_train = np.vstack(X_trains)
            y_train = np.concatenate(y_trains)

            # Cap training size
            if len(X_train) > 2_000_000:
                idx = np.random.RandomState(42).choice(len(X_train), 2_000_000, replace=False)
                X_train = X_train[idx]
                y_train = y_train[idx]

            try:
                import lightgbm as lgb
                from scipy.stats import pearsonr

                # Normalize target
                y_mean = np.mean(y_train)
                y_std = np.std(y_train)
                if y_std < 1e-10:
                    continue
                y_train_norm = (y_train - y_mean) / y_std

                # Train
                train_ds = lgb.Dataset(X_train, label=y_train_norm, free_raw_data=True)
                model = lgb.train(
                    lgbm_params, train_ds, num_boost_round=200,
                    valid_sets=[train_ds], callbacks=[lgb.log_evaluation(0)],
                )

                # Load full test day for prediction
                test_features, test_mid = load_day(day_paths[test_idx][1])
                test_fwd_ret = compute_forward_return(test_mid, horizon_bars)
                del test_mid

                valid_mask = np.all(np.isfinite(test_features), axis=1)
                predictions = np.zeros(len(test_features))
                if np.any(valid_mask):
                    preds_raw = model.predict(test_features[valid_mask])
                    predictions[valid_mask] = preds_raw * y_std + y_mean

                # IC on subsampled test data
                X_test = day_train_data[test_idx][1]
                y_test = day_train_data[test_idx][2]
                if len(X_test) > 100:
                    p_test = model.predict(X_test) * y_std + y_mean
                    ic, _ = pearsonr(p_test, y_test)
                else:
                    ic = 0.0
                ics.append(ic)

                # Save predictions
                out_path = OUTPUT_DIR / f"{sig_name}_{test_date}.npz"
                np.savez_compressed(str(out_path), predictions=predictions)

                # Feature importance
                importance = model.feature_importance(importance_type='gain')
                top_feats = np.argsort(importance)[::-1][:5]

                if (test_idx - args.min_train_days) % 5 == 0 or test_idx == args.min_train_days:
                    top_names = [feature_names[i] if feature_names else str(i) for i in top_feats[:3]]
                    logger.info(f"  [{test_idx+1}/{len(day_train_data)}] {test_date}  "
                              f"IC={ic:+.4f}  train={len(X_train):,}  "
                              f"top3={top_names}")

                del test_features, test_fwd_ret, predictions, model, train_ds

            except Exception as e:
                logger.error(f"  [{test_idx+1}/{len(day_train_data)}] {test_date}: ERROR: {e}")
                import traceback
                logger.error(traceback.format_exc())
                ics.append(0.0)

            del X_train, y_train

        # Summary
        ics_arr = np.array(ics)
        valid_ics = ics_arr[ics_arr != 0.0]
        n_valid = len(valid_ics)

        if n_valid > 0:
            mean_ic = np.mean(valid_ics)
            std_ic = np.std(valid_ics)
            t_stat = mean_ic / (std_ic / np.sqrt(n_valid)) if std_ic > 0 else 0.0
            pct_pos = 100 * np.mean(valid_ics > 0)

            logger.info(f"\n{'='*70}")
            logger.info(f"SUMMARY: {sig_name}")
            logger.info(f"{'='*70}")
            logger.info(f"  Days predicted: {n_valid}")
            logger.info(f"  Mean IC: {mean_ic:+.4f}")
            logger.info(f"  Std IC:  {std_ic:.4f}")
            logger.info(f"  t-stat:  {t_stat:.2f}")
            logger.info(f"  Pct positive: {pct_pos:.1f}%")
            logger.info(f"  Min IC: {np.min(valid_ics):+.4f}")
            logger.info(f"  Max IC: {np.max(valid_ics):+.4f}")

            logger.info(f"\n  Baseline raw feature IC @ {cfg['label']}: ~0.10")
            if mean_ic > 0.10:
                logger.info(f"  >>> MODEL ADDS VALUE: IC improved from ~0.10 to {mean_ic:+.4f}")
            else:
                logger.info(f"  >>> MODEL DOES NOT ADD VALUE over raw features")
        else:
            logger.info(f"  No valid predictions generated for {sig_name}")

        del day_train_data

    logger.info(f"\nDone. Log: {_log_file}")


if __name__ == '__main__':
    main()
