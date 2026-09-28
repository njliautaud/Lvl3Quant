#!/usr/bin/env python3
"""
Volatility Prediction — Predict MAGNITUDE of moves, not direction.

Key insight: If we can predict when volatility will spike, we can:
  1. Trade straddle-like strategies (profit from magnitude, not direction)
  2. Time market order entries to coincide with big moves (higher signal/cost ratio)
  3. Size positions dynamically (bigger when calm → big move expected)

Targets:
  - abs_return_Ns: absolute value of N-second forward return
  - realized_vol_Ns: rolling realized volatility over next N seconds
  - large_move_Ns: binary flag for |return| > threshold in next N seconds

Uses existing 340 MBO features (excluding price levels).

Usage:
  python volatility_prediction.py --horizon 10s --n-days 100 --workers 8
  python volatility_prediction.py --horizon 1m --n-days 50
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import numpy as np
from scipy.stats import spearmanr

ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR))

logging.basicConfig(
    format='%(asctime)s [vol_pred] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('vol_pred')

# Feature indices to exclude (price levels)
EXCLUDE_FEATURES = [0, 3, 8, 9]  # mid, microprice, best_bid, best_ask

# Horizon configs
HORIZON_CONFIGS = {
    '1s':   {'bars': 100,   'subsample': 10,   'label': '1s'},
    '10s':  {'bars': 1000,  'subsample': 100,  'label': '10s'},
    '30s':  {'bars': 3000,  'subsample': 300,  'label': '30s'},
    '1m':   {'bars': 6000,  'subsample': 600,  'label': '1min'},
    '5m':   {'bars': 30000, 'subsample': 3000, 'label': '5min'},
}


def load_day(fpath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load raw 340-feature MBO snapshot. Returns (features_336, mid_prices)."""
    data = np.load(str(fpath))
    raw = data['mbo_features']
    mid = raw[:, 0].copy()

    # Forward-fill NaN mid prices
    mask = np.isnan(mid)
    if mask.any():
        first_valid = np.argmax(~mask)
        mid[:first_valid] = mid[first_valid]

    keep = [i for i in range(raw.shape[1]) if i not in EXCLUDE_FEATURES]
    features = raw[:, keep].astype(np.float32)
    np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

    del raw
    return features, mid


def compute_vol_targets(mid: np.ndarray, horizon_bars: int) -> dict:
    """Compute multiple volatility-related targets (vectorized)."""
    N = len(mid)
    returns = np.diff(mid, prepend=mid[0]) / np.where(mid > 0, mid, 1.0)

    # 1. Absolute forward return (already vectorized)
    fwd_return = np.full(N, np.nan)
    fwd_return[:N-horizon_bars] = (mid[horizon_bars:] - mid[:N-horizon_bars]) / mid[:N-horizon_bars]
    abs_fwd_return = np.abs(fwd_return)

    # 2. Forward realized volatility — vectorized via cumsum
    fwd_rvol = np.full(N, np.nan)
    r = returns[1:]  # skip prepended zero
    cs = np.cumsum(r)
    cs2 = np.cumsum(r ** 2)
    h = horizon_bars
    if len(r) >= h:
        # sum and sum_sq over windows [i+1, i+1+h)
        sum_w = np.empty(N - h)
        sum_sq_w = np.empty(N - h)
        sum_w[:h] = cs[:h]  # not used directly, we need offset windows
        # Window starting at returns index i (0-based): sum of r[i:i+h]
        sum_all = np.concatenate(([0.0], cs))
        sum_sq_all = np.concatenate(([0.0], cs2))
        for_start = np.arange(0, N - h)
        for_end = for_start + h
        # Clip to valid range
        valid_end = np.minimum(for_end, len(r))
        valid_mask = valid_end > for_start
        s = sum_all[valid_end] - sum_all[for_start]
        s2 = sum_sq_all[valid_end] - sum_sq_all[for_start]
        mean_r = s / h
        mean_r2 = s2 / h
        var = mean_r2 - mean_r ** 2
        var = np.clip(var, 0, None)
        fwd_rvol[:N-h] = np.sqrt(var)

    # 3. Max favorable excursion — vectorized via stride_tricks sliding window
    fwd_mfe = np.full(N, np.nan)
    if N > h:
        from numpy.lib.stride_tricks import sliding_window_view
        price_windows = sliding_window_view(mid[1:], h)  # shape: (N-1-h+1, h)
        n_windows = min(N - h, price_windows.shape[0])
        win_max = np.max(price_windows[:n_windows], axis=1)
        win_min = np.min(price_windows[:n_windows], axis=1)
        ref_prices = mid[:n_windows]
        max_up = win_max - ref_prices
        max_down = ref_prices - win_min
        fwd_mfe[:n_windows] = np.maximum(max_up, max_down) / ref_prices

    # 4. Large move indicator (|return| > 2 ticks in next horizon)
    tick_size = 0.25  # ES mini tick
    large_threshold = 2 * tick_size / np.nanmean(mid)  # 2 ticks as fraction
    large_move = (abs_fwd_return > large_threshold).astype(np.float64)

    return {
        'abs_return': abs_fwd_return,
        'rvol': fwd_rvol,
        'mfe': fwd_mfe,
        'large_move': large_move,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=str, default='10s', choices=list(HORIZON_CONFIGS.keys()))
    parser.add_argument('--n-days', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    parser.add_argument('--data-dir', type=str,
                        default=str(ROOT_DIR / 'data' / 'processed' / 'mbo_features_cache'))
    parser.add_argument('--output-dir', type=str,
                        default=str(ROOT_DIR / 'alpha_discovery' / 'results'))
    parser.add_argument('--target', type=str, default='abs_return',
                        choices=['abs_return', 'rvol', 'mfe', 'large_move'])
    args = parser.parse_args()

    config = HORIZON_CONFIGS[args.horizon]
    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = output_dir / f'vol_pred_{args.target}_{args.horizon}_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s [vol_pred] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    files = sorted(data_dir.glob('*_mbo_features.npz'))
    if not files:
        files = sorted(data_dir.glob('mbo_features_*.npz'))
    if args.n_days > 0:
        files = files[:args.n_days]

    logger.info(f"Volatility Prediction: {args.target} @ {args.horizon}")
    logger.info(f"Found {len(files)} days")
    logger.info(f"Horizon: {config['bars']} bars = {config['bars']/100:.0f}s")
    logger.info(f"Subsample: {config['subsample']} bars")
    logger.info(f"Workers: {args.workers}")

    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed")
        sys.exit(1)

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

    MIN_TRAIN_DAYS = 15

    # Pre-compute features and targets per day
    logger.info("Pre-computing features and volatility targets...")
    t0 = time.time()
    day_data = []

    for i, fpath in enumerate(files):
        stem = fpath.stem
        date_str = stem.replace('_mbo_features', '').replace('mbo_features_', '')

        try:
            features, mid = load_day(fpath)
            targets = compute_vol_targets(mid, config['bars'])
            target_arr = targets[args.target]

            # Subsample
            indices = np.arange(config['subsample'] - 1, len(features), config['subsample'])
            X = features[indices]
            y = target_arr[indices]

            # Remove NaN
            valid = ~np.isnan(y) & np.all(np.isfinite(X), axis=1) & ~np.isnan(y)
            X = X[valid].astype(np.float32)
            y = y[valid].astype(np.float32)

            day_data.append({'date': date_str, 'X': X, 'y': y, 'n': len(y)})

            if (i + 1) % 10 == 0 or i == 0:
                logger.info(f"  [{i+1}/{len(files)}] {date_str}: {len(y)} samples")

            del features, mid, targets
            gc.collect()

        except Exception as e:
            logger.warning(f"  Error {date_str}: {e}")
            continue

    elapsed = time.time() - t0
    total = sum(d['n'] for d in day_data)
    logger.info(f"Pre-computed {len(day_data)} days, {total:,} total samples ({elapsed:.1f}s)")

    # Get feature names
    feature_names = None
    try:
        from alpha_discovery.mbo_features import get_feature_names
        all_names = get_feature_names()
        keep = [i for i in range(len(all_names)) if i not in EXCLUDE_FEATURES]
        feature_names = [all_names[i] for i in keep]
    except Exception:
        feature_names = [f'f{i}' for i in range(day_data[0]['X'].shape[1])]

    # Walk-forward
    logger.info(f"\n{'='*70}")
    logger.info(f"WALK-FORWARD: Volatility LightGBM ({args.target} @ {args.horizon})")
    logger.info(f"{'='*70}")

    all_ics = []
    feature_imp_total = None

    for test_idx in range(MIN_TRAIN_DAYS, len(day_data)):
        train_days = day_data[:test_idx]
        test_day = day_data[test_idx]

        X_train = np.vstack([d['X'] for d in train_days])
        y_train = np.concatenate([d['y'] for d in train_days])
        X_test = test_day['X']
        y_test = test_day['y']

        if len(y_test) < 10:
            continue

        X_train = X_train.astype(np.float32)
        np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = X_test.astype(np.float32)
        np.nan_to_num(X_test, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_names)
        model = lgb.train(lgbm_params, dtrain, num_boost_round=200)

        preds = model.predict(X_test)
        ic, _ = spearmanr(preds, y_test)
        all_ics.append(ic)

        imp = model.feature_importance(importance_type='gain')
        if feature_imp_total is None:
            feature_imp_total = imp.astype(np.float64)
        else:
            feature_imp_total += imp

        if (test_idx - MIN_TRAIN_DAYS) % 5 == 0 or test_idx == len(day_data) - 1:
            mean_ic = np.mean(all_ics)
            top3_idx = np.argsort(imp)[-3:][::-1]
            top3 = [feature_names[j] for j in top3_idx]
            logger.info(f"  [{test_idx+1}/{len(day_data)}] {test_day['date']}  "
                       f"IC={ic:+.4f}  mean_IC={mean_ic:+.4f}  "
                       f"train={len(y_train):,}  top3={top3}")

        del X_train, y_train, dtrain, model
        gc.collect()

    # Summary
    if all_ics:
        ics = np.array(all_ics)
        mean_ic = np.mean(ics)
        std_ic = np.std(ics)
        t_stat = mean_ic / (std_ic / np.sqrt(len(ics))) if std_ic > 0 else 0
        pct_pos = np.mean(ics > 0) * 100

        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY: {args.target} @ {args.horizon}")
        logger.info(f"{'='*70}")
        logger.info(f"  Days predicted: {len(ics)}")
        logger.info(f"  Mean IC: {mean_ic:+.4f}")
        logger.info(f"  Std IC:  {std_ic:.4f}")
        logger.info(f"  t-stat:  {t_stat:.2f}")
        logger.info(f"  Pct positive: {pct_pos:.1f}%")
        logger.info(f"  Min IC: {np.min(ics):+.4f}")
        logger.info(f"  Max IC: {np.max(ics):+.4f}")

        # Top 10 features
        if feature_imp_total is not None:
            top10_idx = np.argsort(feature_imp_total)[-10:][::-1]
            logger.info(f"\n  Top 10 features:")
            for rank, j in enumerate(top10_idx, 1):
                logger.info(f"    {rank:2d}. {feature_names[j]:40s}  gain={feature_imp_total[j]:.0f}")

        # Assess value
        if mean_ic > 0.15:
            logger.info(f"\n  >>> STRONG VOLATILITY PREDICTION (IC={mean_ic:+.4f})")
            logger.info(f"  >>> This could enable magnitude-aware trading!")
        elif mean_ic > 0.08:
            logger.info(f"\n  >>> Moderate volatility prediction. Worth investigating further.")
        else:
            logger.info(f"\n  >>> Weak volatility prediction. May not be actionable.")

        # Save
        results = {
            'target': args.target,
            'horizon': args.horizon,
            'n_days': len(day_data),
            'n_folds': len(ics),
            'mean_ic': float(mean_ic),
            'std_ic': float(std_ic),
            't_stat': float(t_stat),
            'pct_positive': float(pct_pos),
            'fold_ics': ics.tolist(),
            'feature_names': feature_names[:10] if feature_names else [],
        }
        results_path = output_dir / f'vol_pred_{args.target}_{args.horizon}_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n  Results: {results_path}")

    logger.info("\nDone.")


if __name__ == '__main__':
    main()
