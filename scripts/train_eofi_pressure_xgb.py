#!/usr/bin/env python3
"""
Train XGBoost GPU model to predict EOFI smooth pressure targets.
EOFI (Exponential Order Flow Imbalance) has IC=0.190, strongest of all pressure targets.

Per HC #504 R2a: XGBoost GPU is whitelisted for Razer.
Per HC #509: Smooth pressure targets are the new prediction target.
Per HC #511: Commission = 0.376 ticks RT.

Walk-forward sliding window: 20-day train, 1-day OOT, oldest-day-drop.
"""

import os
import sys
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr

# Try MLflow
try:
    import mlflow
    HAS_MLFLOW = True
except ImportError:
    HAS_MLFLOW = False

# XGBoost
import xgboost as xgb

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s: %(message)s')
log = logging.getLogger(__name__)


def load_day(event_dir: Path, pressure_dir: Path, date_str: str, target: str = 'eofi', horizon: str = '10s'):
    """Load features and pressure labels for a single day."""
    event_file = event_dir / f"{date_str}_mbo_events.npz"
    pressure_file = pressure_dir / f"{date_str}_pressure.npz"

    if not event_file.exists() or not pressure_file.exists():
        return None

    try:
        ev = np.load(event_file)
        pr = np.load(pressure_file)
    except Exception as e:
        log.warning(f"Failed to load {date_str}: {e}")
        return None

    features = ev['events']  # (N, 25)

    label_key = f"{target}_label_{horizon}"
    if label_key not in pr:
        log.warning(f"No {label_key} in pressure file for {date_str}")
        ev.close()
        pr.close()
        return None

    labels = pr[label_key]

    # Verify alignment by shape
    if features.shape[0] != labels.shape[0]:
        log.warning(f"Shape mismatch {date_str}: events={features.shape[0]}, labels={labels.shape[0]}")
        return None

    # Simple filter: keep rows where both features and labels are valid
    valid = np.isfinite(labels) & np.all(np.isfinite(features), axis=1)

    if valid.sum() < 1000:
        log.warning(f"Too few valid samples for {date_str}: {valid.sum()}")
        return None

    # Subsample to limit memory (500K per day max)
    max_per_day = 500_000
    valid_idx = np.where(valid)[0]
    if len(valid_idx) > max_per_day:
        rng = np.random.RandomState(42)
        valid_idx = rng.choice(valid_idx, max_per_day, replace=False)
        valid_idx.sort()

    return {
        'date': date_str,
        'features': features[valid_idx].copy(),
        'labels': labels[valid_idx].copy(),
        'n_samples': len(valid_idx),
    }


def train_fold(train_days, test_day, params, target, horizon):
    """Train one WF fold: train on train_days, test on test_day."""
    # Concatenate training data
    X_train = np.concatenate([d['features'] for d in train_days])
    y_train = np.concatenate([d['labels'] for d in train_days])

    X_test = test_day['features']
    y_test = test_day['labels']

    # Subsample if training set too large (>5M rows)
    if X_train.shape[0] > 5_000_000:
        idx = np.random.choice(X_train.shape[0], 5_000_000, replace=False)
        X_train = X_train[idx]
        y_train = y_train[idx]

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    model = xgb.train(
        params,
        dtrain,
        num_boost_round=200,
        evals=[(dtest, 'test')],
        early_stopping_rounds=20,
        verbose_eval=False,
    )

    preds = model.predict(dtest)

    # Compute metrics
    ic, _ = spearmanr(preds, y_test)
    mse = float(np.mean((preds - y_test) ** 2))
    mae = float(np.mean(np.abs(preds - y_test)))

    # Directional accuracy (sign agreement)
    dir_acc = float(np.mean(np.sign(preds) == np.sign(y_test)))

    return {
        'fold_date': test_day['date'],
        'n_train': X_train.shape[0],
        'n_test': X_test.shape[0],
        'ic': float(ic) if not np.isnan(ic) else 0.0,
        'mse': mse,
        'mae': mae,
        'dir_acc': dir_acc,
        'best_iteration': model.best_iteration if hasattr(model, 'best_iteration') else 200,
        'predictions': preds,
        'actuals': y_test,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data-dir', type=str, required=True, help='Path to smart_v3 events')
    parser.add_argument('--pressure-dir', type=str, required=True, help='Path to smooth pressure targets')
    parser.add_argument('--output-dir', type=str, required=True, help='Output directory')
    parser.add_argument('--target', type=str, default='eofi', choices=['eofi', 'pdi', 'ntps', 'tia'])
    parser.add_argument('--horizon', type=str, default='10s', choices=['1s', '5s', '10s', '30s'])
    parser.add_argument('--train-window', type=int, default=20, help='Training window in days')
    parser.add_argument('--year-min', type=int, default=2026, help='Minimum year for data')
    parser.add_argument('--mlflow-uri', type=str, default=None)
    parser.add_argument('--experiment-name', type=str, default='eofi_pressure_xgb')
    args = parser.parse_args()

    event_dir = Path(args.data_dir)
    pressure_dir = Path(args.pressure_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find available dates (intersection of events and pressure)
    event_dates = {f.stem.split('_')[0] for f in event_dir.glob('*_mbo_events.npz')}
    pressure_dates = {f.stem.split('_')[0] for f in pressure_dir.glob('*_pressure.npz')}
    common_dates = sorted(event_dates & pressure_dates)

    if args.year_min:
        common_dates = [d for d in common_dates if int(d[:4]) >= args.year_min]

    log.info(f"Found {len(common_dates)} dates with both events and pressure targets")
    log.info(f"Target: {args.target}, Horizon: {args.horizon}")
    log.info(f"Date range: {common_dates[0]} to {common_dates[-1]}")

    # Load all days
    log.info("Loading data...")
    all_days = []
    for date_str in common_dates:
        day = load_day(event_dir, pressure_dir, date_str, args.target, args.horizon)
        if day is not None:
            all_days.append(day)
            if len(all_days) % 10 == 0:
                log.info(f"  Loaded {len(all_days)} days...")

    log.info(f"Loaded {len(all_days)} valid days, {sum(d['n_samples'] for d in all_days):,} total samples")

    if len(all_days) < args.train_window + 1:
        log.error(f"Not enough days ({len(all_days)}) for {args.train_window}-day window")
        return

    # XGBoost params
    params = {
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'tree_method': 'hist',
        'device': 'cuda',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'min_child_weight': 100,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'seed': 42,
    }

    # Try GPU
    try:
        test_dm = xgb.DMatrix(np.random.randn(100, 5))
        xgb.train({'tree_method': 'hist', 'device': 'cuda', 'objective': 'reg:squarederror'},
                  test_dm, num_boost_round=1, verbose_eval=False)
        log.info("Using GPU (CUDA)")
    except Exception:
        log.info("GPU not available, using CPU")
        params['device'] = 'cpu'

    # MLflow setup
    if HAS_MLFLOW and args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.experiment_name)
        run = mlflow.start_run(run_name=f"{args.target}_{args.horizon}_xgb")
        mlflow.log_params({
            'target': args.target,
            'horizon': args.horizon,
            'train_window': args.train_window,
            'n_dates': len(all_days),
            **{k: v for k, v in params.items() if isinstance(v, (int, float, str))},
        })

    # Walk-forward training
    fold_results = []
    all_preds = []
    all_actuals = []

    tw = args.train_window
    n_folds = len(all_days) - tw
    log.info(f"Running {n_folds} walk-forward folds (window={tw})")

    for i in range(n_folds):
        train_days = all_days[i:i + tw]
        test_day = all_days[i + tw]

        t0 = time.time()
        result = train_fold(train_days, test_day, params, args.target, args.horizon)
        elapsed = time.time() - t0

        fold_results.append({k: v for k, v in result.items() if k not in ('predictions', 'actuals')})
        all_preds.append(result['predictions'])
        all_actuals.append(result['actuals'])

        log.info(f"Fold {i+1}/{n_folds} [{result['fold_date']}]: "
                 f"IC={result['ic']:.4f}, DirAcc={result['dir_acc']:.3f}, "
                 f"n_test={result['n_test']:,}, {elapsed:.1f}s")

        if HAS_MLFLOW and args.mlflow_uri:
            mlflow.log_metric('fold_ic', result['ic'], step=i)
            mlflow.log_metric('fold_dir_acc', result['dir_acc'], step=i)

    # Compute concat metrics
    concat_preds = np.concatenate(all_preds)
    concat_actuals = np.concatenate(all_actuals)
    concat_ic, _ = spearmanr(concat_preds, concat_actuals)
    concat_dir_acc = float(np.mean(np.sign(concat_preds) == np.sign(concat_actuals)))

    per_fold_ics = [r['ic'] for r in fold_results]

    summary = {
        'target': args.target,
        'horizon': args.horizon,
        'n_folds': n_folds,
        'n_oos_samples': len(concat_preds),
        'concat_spearman_ic': float(concat_ic),
        'concat_dir_acc': concat_dir_acc,
        'per_fold_mean_ic': float(np.mean(per_fold_ics)),
        'per_fold_median_ic': float(np.median(per_fold_ics)),
        'per_fold_min_ic': float(np.min(per_fold_ics)),
        'per_fold_max_ic': float(np.max(per_fold_ics)),
        'per_fold_std_ic': float(np.std(per_fold_ics)),
        'fold_details': fold_results,
    }

    log.info(f"\n{'='*60}")
    log.info(f"RESULTS: {args.target} @ {args.horizon}")
    log.info(f"  Concat Spearman IC: {concat_ic:.4f}")
    log.info(f"  Concat Dir Acc: {concat_dir_acc:.3f}")
    log.info(f"  Per-fold mean IC: {np.mean(per_fold_ics):.4f} (±{np.std(per_fold_ics):.4f})")
    log.info(f"  Per-fold range: [{np.min(per_fold_ics):.4f}, {np.max(per_fold_ics):.4f}]")
    log.info(f"  OOS samples: {len(concat_preds):,}")
    log.info(f"{'='*60}")

    # Save results
    summary_path = output_dir / f"summary_{args.target}_{args.horizon}.json"
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Saved summary to {summary_path}")

    # Save predictions
    preds_path = output_dir / f"predictions_{args.target}_{args.horizon}.npz"
    np.savez_compressed(preds_path, predictions=concat_preds, actuals=concat_actuals)
    log.info(f"Saved predictions to {preds_path}")

    if HAS_MLFLOW and args.mlflow_uri:
        mlflow.log_metrics({
            'concat_ic': float(concat_ic),
            'concat_dir_acc': concat_dir_acc,
            'mean_fold_ic': float(np.mean(per_fold_ics)),
            'n_folds': n_folds,
        })
        mlflow.log_artifact(str(summary_path))
        mlflow.end_run()

    log.info("Done.")


if __name__ == "__main__":
    main()
