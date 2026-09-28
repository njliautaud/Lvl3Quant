#!/usr/bin/env python3
"""
Walk-Forward Test: Slow-Decay Features at Long Horizons (120s-300s)

KEY TEST: Do order book structural features (total_depth, pressure, L2+ orders)
predict 2-5 minute returns in a walk-forward framework?

Uses ONLY the top 10-15 slow-decay features (not all 336).
Targets: 120s return (most samples) and 300s return.
"""

import argparse
import gc
import json
import logging
import numpy as np
import sys
import time
from datetime import datetime
from pathlib import Path
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

logging.basicConfig(format='%(asctime)s [slow_wf] %(message)s', datefmt='%H:%M:%S', level=logging.INFO)
logger = logging.getLogger('slow_wf')

EXCLUDE_FEATURES = [0, 3, 8, 9]
COST_TICKS = 1.24

# Top slow-decay features from full_ic_decay analysis
SLOW_DECAY_FEATURES = [
    'total_depth_log', 'total_ask_vol', 'mean_ask_size', 'ask_pressure',
    'bid_pressure', 'ask_L2_orders', 'ask_L3_orders', 'ask_L4_orders',
    'ask_L5_orders', 'total_bid_vol', 'mean_bid_size',
    'rvol_10', 'rvol_20', 'rvol_50',
    'event_int_50', 'event_int_20',
    'depletion_dir_20', 'book_refresh',
    'ret_5', 'ret_vel_5',
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--horizon', type=str, default='120s', choices=['60s', '120s', '300s'])
    parser.add_argument('--n-days', type=int, default=100)
    parser.add_argument('--workers', type=int, default=8)
    args = parser.parse_args()

    horizon_bars = {'60s': 6000, '120s': 12000, '300s': 30000}[args.horizon]
    horizon_sec = horizon_bars / 100

    data_dir = ROOT / 'data' / 'processed' / 'mbo_features_cache'
    output_dir = ROOT / 'alpha_discovery' / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = output_dir / f'slow_decay_wf_{args.horizon}_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s [slow_wf] %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    files = sorted(data_dir.glob('*_mbo_features.npz'))[:args.n_days]
    logger.info(f"Slow-Decay Walk-Forward: {args.horizon} ({horizon_sec:.0f}s)")
    logger.info(f"Days: {len(files)}, Workers: {args.workers}")

    # Get feature names and indices
    feature_names = None
    try:
        from alpha_discovery.mbo_features import get_feature_names
        all_names = get_feature_names()
        keep = [i for i in range(len(all_names)) if i not in EXCLUDE_FEATURES]
        feature_names = [all_names[i] for i in keep]
    except Exception:
        feature_names = [f'f{i}' for i in range(336)]

    name_to_idx = {n: i for i, n in enumerate(feature_names)}
    slow_indices = [name_to_idx[f] for f in SLOW_DECAY_FEATURES if f in name_to_idx]
    slow_names = [f for f in SLOW_DECAY_FEATURES if f in name_to_idx]
    logger.info(f"Using {len(slow_indices)} slow-decay features: {slow_names[:5]}...")

    try:
        import lightgbm as lgb
    except ImportError:
        logger.error("lightgbm not installed")
        sys.exit(1)

    # Pre-load data
    logger.info("Loading data...")
    t0 = time.time()
    day_data = []

    for i, fpath in enumerate(files):
        date = fpath.stem.replace('_mbo_features', '')
        data = np.load(str(fpath))
        raw = data['mbo_features']
        mid = raw[:, 0].copy()
        mask = np.isnan(mid)
        if mask.any():
            fv = np.argmax(~mask)
            mid[:fv] = mid[fv]

        keep_cols = [j for j in range(raw.shape[1]) if j not in EXCLUDE_FEATURES]
        features = raw[:, keep_cols][:, slow_indices]  # Only slow-decay features
        features = np.nan_to_num(features, nan=0.0, posinf=0.0, neginf=0.0)

        N = len(mid)
        if N <= horizon_bars + 100:
            continue

        # Forward return at target horizon
        fwd_ret = np.full(N, np.nan)
        fwd_ret[:N - horizon_bars] = (mid[horizon_bars:] - mid[:N - horizon_bars]) / np.where(
            mid[:N - horizon_bars] > 0, mid[:N - horizon_bars], 1.0)

        # Subsample: space by horizon_bars/2 for more samples but some overlap
        spacing = max(horizon_bars // 2, 1000)
        indices = np.arange(0, N - horizon_bars, spacing)
        X = features[indices]
        y = fwd_ret[indices]
        valid = np.isfinite(y) & np.all(np.isfinite(X), axis=1)
        X = X[valid].astype(np.float32)
        y = y[valid].astype(np.float32)

        if len(y) < 5:
            continue

        day_data.append({'date': date, 'X': X, 'y': y, 'n': len(y)})

        if (i+1) % 10 == 0 or i == 0:
            logger.info(f"  [{i+1}/{len(files)}] {date}: {len(y)} samples")

        del raw, features, mid
        gc.collect()

    elapsed = time.time() - t0
    total = sum(d['n'] for d in day_data)
    logger.info(f"Loaded {len(day_data)} days, {total:,} samples ({elapsed:.1f}s)")

    # LightGBM params — regularized for small sample regime
    lgbm_params = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.03,
        'num_leaves': 15,  # Small — prevent overfitting
        'max_depth': 4,
        'min_child_samples': 50,
        'subsample': 0.6,
        'colsample_bytree': 0.6,
        'reg_alpha': 1.0,
        'reg_lambda': 5.0,
        'verbose': -1,
        'n_jobs': args.workers,
        'seed': 42,
    }

    MIN_TRAIN_DAYS = 10

    # Walk-forward
    logger.info(f"\n{'='*70}")
    logger.info(f"WALK-FORWARD: Slow-Decay LightGBM ({args.horizon})")
    logger.info(f"{'='*70}")

    all_ics = []
    all_raw_ics = {}  # feature -> list of daily ICs
    for fn in slow_names:
        all_raw_ics[fn] = []
    feature_imp_total = None

    for test_idx in range(MIN_TRAIN_DAYS, len(day_data)):
        train_days = day_data[:test_idx]
        test_day = day_data[test_idx]

        X_train = np.vstack([d['X'] for d in train_days])
        y_train = np.concatenate([d['y'] for d in train_days])
        X_test = test_day['X']
        y_test = test_day['y']

        if len(y_test) < 5:
            continue

        X_train = np.nan_to_num(X_train, nan=0.0, posinf=0.0, neginf=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0, posinf=0.0, neginf=0.0)

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=slow_names)
        model = lgb.train(lgbm_params, dtrain, num_boost_round=100)

        preds = model.predict(X_test)
        ic, _ = spearmanr(preds, y_test)
        all_ics.append(ic)

        imp = model.feature_importance(importance_type='gain')
        if feature_imp_total is None:
            feature_imp_total = imp.astype(np.float64)
        else:
            feature_imp_total += imp

        # Also compute raw feature ICs for comparison
        for fi, fn in enumerate(slow_names):
            raw_ic, _ = spearmanr(X_test[:, fi], y_test)
            if np.isfinite(raw_ic):
                all_raw_ics[fn].append(raw_ic)

        if (test_idx - MIN_TRAIN_DAYS) % 5 == 0 or test_idx == len(day_data) - 1:
            mean_ic = np.mean(all_ics)
            top3_idx = np.argsort(imp)[-3:][::-1]
            top3 = [slow_names[j] for j in top3_idx]
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

        # Move sizes
        avg_price = 5800
        tick_frac = 0.25 / avg_price
        # Use the actual average move for this horizon
        all_y = np.concatenate([d['y'] for d in day_data[MIN_TRAIN_DAYS:]])
        avg_move_ticks = np.mean(np.abs(all_y)) / tick_frac
        expected_pnl = abs(mean_ic) * avg_move_ticks - COST_TICKS

        logger.info(f"\n{'='*70}")
        logger.info(f"SUMMARY: Slow-Decay LightGBM @ {args.horizon}")
        logger.info(f"{'='*70}")
        logger.info(f"  Folds: {len(ics)}")
        logger.info(f"  Mean IC: {mean_ic:+.4f}")
        logger.info(f"  Std IC:  {std_ic:.4f}")
        logger.info(f"  t-stat:  {t_stat:.2f}")
        logger.info(f"  Pct positive: {pct_pos:.1f}%")
        logger.info(f"  Min IC: {np.min(ics):+.4f}")
        logger.info(f"  Max IC: {np.max(ics):+.4f}")
        logger.info(f"  Avg move: {avg_move_ticks:.1f} ticks")
        logger.info(f"  Expected PnL: {expected_pnl:+.2f} ticks/trade")
        logger.info(f"  Breakeven IC: {COST_TICKS/avg_move_ticks:.4f}")

        if expected_pnl > 0:
            logger.info(f"\n  >>> PROFITABLE! E[PnL] = +{expected_pnl:.2f} ticks/trade")
        else:
            logger.info(f"\n  >>> NOT PROFITABLE. Gap = {COST_TICKS/(abs(mean_ic)*avg_move_ticks):.2f}x")

        # Compare with raw features
        logger.info(f"\n  Raw Feature IC comparison:")
        for fn in slow_names:
            raw_ics_arr = all_raw_ics[fn]
            if raw_ics_arr:
                raw_mean = np.mean(raw_ics_arr)
                logger.info(f"    {fn:30s} raw_IC={raw_mean:+.4f}")

        # Top features
        if feature_imp_total is not None:
            top10_idx = np.argsort(feature_imp_total)[-10:][::-1]
            logger.info(f"\n  Top 10 features by gain:")
            for rank, j in enumerate(top10_idx, 1):
                logger.info(f"    {rank:2d}. {slow_names[j]:30s} gain={feature_imp_total[j]:.0f}")

        # Save
        results = {
            'horizon': args.horizon,
            'n_days': len(day_data),
            'n_folds': len(ics),
            'mean_ic': float(mean_ic),
            'std_ic': float(std_ic),
            't_stat': float(t_stat),
            'pct_positive': float(pct_pos),
            'avg_move_ticks': float(avg_move_ticks),
            'expected_pnl_ticks': float(expected_pnl),
            'breakeven_ic': float(COST_TICKS / avg_move_ticks),
            'fold_ics': ics.tolist(),
            'feature_names': slow_names,
        }
        results_path = output_dir / f'slow_decay_wf_{args.horizon}_{timestamp}.json'
        with open(results_path, 'w') as f:
            json.dump(results, f, indent=2)
        logger.info(f"\n  Results: {results_path}")

    logger.info("\nDone.")


if __name__ == '__main__':
    main()
