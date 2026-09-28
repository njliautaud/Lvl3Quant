#!/usr/bin/env python3
"""
LGBM Training - Full 30-Feature Order Book Data with MLflow
Designed for CPU nodes (Saturn/Jupiter)

Features:
- 30-feature order book microstructure data
- Expanding window walk-forward (NOT sliding)
- 5-day exponential decay weighting
- 5-day test window
- Confidence-stratified IC reporting
- **MANDATORY MLflow tracking**
"""
import logging
import json
import numpy as np
from pathlib import Path
import lightgbm as lgb
from scipy.stats import spearmanr
from datetime import datetime, timedelta
import mlflow
import socket

logging.basicConfig(format='%(asctime)s %(message)s', level=logging.INFO)
log = logging.getLogger()

# Config
ROOT = Path('/home/jupiter/Lvl3Quant')
DATA_DIR = ROOT / 'data' / 'processed' / 'mbo_book_features'
OUT_DIR = ROOT / 'alpha_discovery' / 'results' / 'lgbm_book_features'
OUT_DIR.mkdir(parents=True, exist_ok=True)

TEST_DAYS = 5
DECAY_DAYS = 5  # Weight decay parameter
HORIZON = 'labels_10s'
MAX_TRAIN_SAMPLES = 20_000_000  # Cap to prevent OOM (64GB machine, ~44GB was OOM)

# MLflow setup
mlflow.set_tracking_uri('http://localhost:5000')
mlflow.set_experiment('LGBM_BookFeatures')

FEATURE_NAMES = [
    'bid_price_1','bid_price_2','bid_price_3','bid_price_4','bid_price_5',
    'ask_price_1','ask_price_2','ask_price_3','ask_price_4','ask_price_5',
    'bid_size_1', 'bid_size_2', 'bid_size_3', 'bid_size_4', 'bid_size_5',
    'ask_size_1', 'ask_size_2', 'ask_size_3', 'ask_size_4', 'ask_size_5',
    'cum_delta','rolling_imbalance_100','trade_intensity_100',
    'depth_imbalance_5','spread_ticks',
    'bid_size_change','ask_size_change','mid_price_change_ticks',
    'spread_change_ticks','net_order_flow',
]

def apply_decay_weights(dates, decay_days=5):
    """
    Apply exponential decay weights to training samples.
    More recent data gets higher weight.
    """
    unique_dates = np.array(sorted(set(dates)))
    if len(unique_dates) == 0:
        return np.ones(len(dates))

    days_old = np.array([(unique_dates[-1] - d).days for d in dates])
    date_weights = np.exp(-days_old / decay_days)
    date_weights = date_weights / date_weights.sum() * len(unique_dates)

    # Map to sample weights
    date_to_weight = {d: w for d, w in zip(unique_dates, date_weights)}
    weights = np.array([date_to_weight[d] for d in dates])
    return weights

def load_features(fp):
    """Load 30-feature book data."""
    d = np.load(fp, allow_pickle=True)
    X = d['features'].astype('f4')  # (N, 30)
    y = d[HORIZON].astype('f4')  # (N,)
    return X, y

def conf_ic(preds, labels):
    """Confidence-stratified IC."""
    abs_p = np.abs(preds)
    res = {}
    for pct, lbl in [(100, 'all'), (50, 'top50'), (25, 'top25'), (10, 'top10')]:
        mask = abs_p >= np.percentile(abs_p, 100-pct) if pct < 100 else np.ones(len(preds), bool)
        if mask.sum() < 50:
            continue
        ic = float(spearmanr(preds[mask], labels[mask]).correlation)
        da = float(np.mean(np.sign(preds[mask]) == np.sign(labels[mask])))
        n = int(mask.sum())
        res[lbl] = {'ic': ic, 'dir_acc': da, 'n': n}
        log.info(f'    [{lbl:6s}] n={n:6d} IC={ic:+.4f} DirAcc={da:.4f}')
    return res

def main():
    files = sorted([f for f in DATA_DIR.glob('*_book_features.npz')])
    if len(files) == 0:
        log.error(f'No files found in {DATA_DIR}')
        return

    log.info(f'Files: {len(files)} ({files[0].stem[:8]}..{files[-1].stem[:8]})')

    start = datetime.strptime(files[0].stem[:8], '%Y%m%d')
    end = datetime.strptime(files[-1].stem[:8], '%Y%m%d')

    # Start MLflow run
    run_name = f'LGBM_Book30_{datetime.now().strftime("%Y%m%d_%H%M%S")}'

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        mlflow.log_param('model', 'LightGBM')
        mlflow.log_param('features', '30_book_features')
        mlflow.log_param('n_estimators', 300)
        mlflow.log_param('learning_rate', 0.05)
        mlflow.log_param('num_leaves', 63)
        mlflow.log_param('min_child_samples', 50)
        mlflow.log_param('decay_days', DECAY_DAYS)
        mlflow.log_param('test_days', TEST_DAYS)
        mlflow.log_param('horizon', HORIZON)
        mlflow.log_param('n_files', len(files))
        mlflow.log_param('node', socket.gethostname())
        mlflow.log_param('window_mode', 'expanding')
        mlflow.log_param('data_dir', str(DATA_DIR))
        mlflow.log_param('output_dir', str(OUT_DIR))

        results = {'folds': [], 'model': 'lgbm_book_30features', 'horizon': HORIZON}
        fold = 0

        # Expanding window walk-forward
        train_files = []

        for i in range(0, len(files), TEST_DAYS):
            if i + TEST_DAYS > len(files):
                break

            # Expanding window: use ALL data up to test start
            test_files = files[i:i+TEST_DAYS]
            train_files = files[:i]  # All files before test

            if len(train_files) < 10:  # Need minimum training data
                continue

            trs = train_files[0].stem[:8]
            tre = train_files[-1].stem[:8]
            xes = test_files[0].stem[:8]
            xee = test_files[-1].stem[:8]

            log.info(f'\nFOLD {fold:02d} train:{trs}..{tre}({len(train_files)}f) test:{xes}..{xee}({len(test_files)}f)')

            # Load training data with per-file subsampling to prevent OOM during vstack
            Xtr = []
            ytr = []
            train_dates = []
            total_rows_loaded = 0

            # Budget per file: distribute MAX_TRAIN_SAMPLES across files
            # Recent files get full allocation, old files get subsampled
            per_file_budget = max(MAX_TRAIN_SAMPLES // max(len(train_files), 1), 100_000)

            for fi, f in enumerate(train_files):
                try:
                    X, y = load_features(f)
                    if X is not None and len(X) > 0:
                        file_date = datetime.strptime(f.stem[:8], '%Y%m%d')
                        # Subsample old files if too many rows (recent 30% of files kept full)
                        recent_file_cutoff = int(len(train_files) * 0.7)
                        if fi < recent_file_cutoff and len(X) > per_file_budget:
                            rng_f = np.random.default_rng(fold * 1000 + fi)
                            idx = rng_f.choice(len(X), size=per_file_budget, replace=False)
                            idx.sort()
                            X = X[idx]
                            y = y[idx]
                        Xtr.append(X)
                        ytr.append(y)
                        train_dates.extend([file_date] * len(X))
                        total_rows_loaded += len(X)
                except Exception as e:
                    log.warning(f' skip {f.name}: {e}')

            if not Xtr:
                fold += 1
                continue

            log.info(f'  Loaded {total_rows_loaded:,d} rows from {len(Xtr)} files (budget/file={per_file_budget:,d})')
            Xtr = np.vstack(Xtr)
            ytr = np.concatenate(ytr)

            # Subsample if too large (prevents OOM on 46GB machine)
            if len(Xtr) > MAX_TRAIN_SAMPLES:
                rng = np.random.default_rng(fold)
                n = len(Xtr)
                # Allocate 60% recent, 40% old — but enforce total cap
                recent_budget = int(MAX_TRAIN_SAMPLES * 0.6)
                old_budget = MAX_TRAIN_SAMPLES - recent_budget
                recent_cutoff = int(n * 0.7)
                # Recent samples (subsample if exceeds budget)
                recent_pool = np.arange(recent_cutoff, n)
                if len(recent_pool) > recent_budget:
                    recent_idx = rng.choice(recent_pool, size=recent_budget, replace=False)
                else:
                    recent_idx = recent_pool
                    old_budget = MAX_TRAIN_SAMPLES - len(recent_idx)  # give surplus to old
                # Old samples
                old_idx = rng.choice(recent_cutoff, size=min(old_budget, recent_cutoff), replace=False)
                keep_idx = np.sort(np.concatenate([old_idx, recent_idx]))
                log.info(f'  Subsampled {n:,d} -> {len(keep_idx):,d} (recent bias, OOM prevention)')
                train_dates_arr = np.array(train_dates)
                Xtr = Xtr[keep_idx]
                ytr = ytr[keep_idx]
                train_dates = train_dates_arr[keep_idx].tolist()

            # Apply decay weights
            train_dates = np.array(train_dates)
            weights = apply_decay_weights(train_dates, DECAY_DAYS)

            log.info(f'  Train samples: {Xtr.shape} (weights: {weights.min():.3f}-{weights.max():.3f})')

            # Load test data
            Xte = []
            yte = []

            for f in test_files:
                try:
                    X, y = load_features(f)
                    if X is not None and len(X) > 0:
                        Xte.append(X)
                        yte.append(y)
                except Exception as e:
                    log.warning(f' skip {f.name}: {e}')

            if not Xte:
                fold += 1
                continue

            Xte = np.vstack(Xte)
            yte = np.concatenate(yte)
            log.info(f'  Test samples: {Xte.shape}')

            # Filter out NaN values
            valid_mask = ~(np.isnan(ytr) | np.isnan(Xtr).any(axis=1) | np.isnan(weights))
            if valid_mask.sum() < len(ytr):
                log.warning(f'  Filtered {len(ytr) - valid_mask.sum()} training rows with NaN')
            Xtr = Xtr[valid_mask]
            ytr = ytr[valid_mask]
            weights = weights[valid_mask]

            # Filter test NaNs
            valid_test = ~(np.isnan(yte) | np.isnan(Xte).any(axis=1))
            if valid_test.sum() < len(yte):
                log.warning(f'  Filtered {len(yte) - valid_test.sum()} test rows with NaN')
            Xte = Xte[valid_test]
            yte = yte[valid_test]

            # Train LGBM with sample weights
            m = lgb.LGBMRegressor(
                n_estimators=300,
                learning_rate=0.05,
                num_leaves=63,
                min_child_samples=50,
                n_jobs=-1,
                verbose=-1
            )
            m.fit(Xtr, ytr, sample_weight=weights)

            # Predict
            preds = m.predict(Xte)

            # Save predictions + leaf indices for meta-model/ensemble
            pred_path = OUT_DIR / f'fold_{fold:02d}_predictions.npz'
            save_dict = dict(predictions=preds, labels=yte)
            # Leaf indices can be huge for large test sets — cap at 2M samples
            if len(Xte) <= 2_000_000:
                leaf_indices = m.predict(Xte, pred_leaf=True).astype(np.int16)
                save_dict['leaf_indices'] = leaf_indices
                log.info(f'  Leaf indices: {leaf_indices.shape}')
            else:
                # For large folds, save leaf indices on a subsample
                rng = np.random.default_rng(fold)
                sub_idx = rng.choice(len(Xte), size=2_000_000, replace=False)
                sub_idx.sort()
                leaf_indices = m.predict(Xte[sub_idx], pred_leaf=True).astype(np.int16)
                save_dict['leaf_indices'] = leaf_indices
                save_dict['leaf_sample_idx'] = sub_idx
                log.info(f'  Leaf indices: {leaf_indices.shape} (subsampled from {len(Xte)})')
            np.savez_compressed(pred_path, **save_dict)
            log.info(f'  Saved predictions → {pred_path}')

            # Save model for later inference
            model_path = OUT_DIR / f'fold_{fold:02d}_model.txt'
            m.booster_.save_model(str(model_path))

            log.info(f'  {HORIZON} confidence-stratified IC:')

            fr = {
                'fold': fold,
                'train': f'{trs}..{tre}',
                'test': f'{xes}..{xee}',
                'n_train': int(len(Xtr)),
                'n_test': int(len(Xte)),
                'decay_days': DECAY_DAYS
            }
            fr['conf_ic'] = conf_ic(preds, yte)
            results['folds'].append(fr)

            # Log to MLflow
            for bucket, metrics in fr['conf_ic'].items():
                mlflow.log_metric(f'fold{fold:02d}_ic_{bucket}', metrics['ic'], step=fold)
                mlflow.log_metric(f'fold{fold:02d}_da_{bucket}', metrics['dir_acc'], step=fold)

            mlflow.log_metric(f'fold{fold:02d}_n_train', fr['n_train'], step=fold)
            mlflow.log_metric(f'fold{fold:02d}_n_test', fr['n_test'], step=fold)

            fold += 1

        # Summary
        log.info('\n=== LGBM 30-FEATURE BOOK DATA SUMMARY ===')
        for bkt in ['all', 'top50', 'top25', 'top10']:
            ics = [f['conf_ic'][bkt]['ic'] for f in results['folds'] if bkt in f.get('conf_ic', {})]
            das = [f['conf_ic'][bkt]['dir_acc'] for f in results['folds'] if bkt in f.get('conf_ic', {})]
            if ics:
                log.info(f'  {HORIZON} [{bkt:6s}] avg_IC={np.mean(ics):+.4f} '
                        f'avg_DirAcc={np.mean(das):.4f} folds={len(ics)}')

                # Log summary metrics
                mlflow.log_metric(f'avg_ic_{bkt}', np.mean(ics))
                mlflow.log_metric(f'avg_da_{bkt}', np.mean(das))

        # Save concat predictions + leaf indices across all folds for meta-model
        all_preds, all_labels, all_leaves = [], [], []
        for fi in range(fold):
            pp = OUT_DIR / f'fold_{fi:02d}_predictions.npz'
            if pp.exists():
                d = np.load(pp)
                all_preds.append(d['predictions'])
                all_labels.append(d['labels'])
                all_leaves.append(d['leaf_indices'])
        if all_preds:
            concat_path = OUT_DIR / 'concat_oot_predictions.npz'
            np.savez_compressed(concat_path,
                predictions=np.concatenate(all_preds),
                labels=np.concatenate(all_labels),
                leaf_indices=np.concatenate(all_leaves),
            )
            log.info(f'Saved concat predictions + leaf indices → {concat_path}')

        out = OUT_DIR / f'lgbm_book30_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
        json.dump(results, open(out, 'w'), indent=2)
        mlflow.log_artifact(str(out))
        log.info(f'Saved: {out}')
        log.info(f'MLflow run: {mlflow.active_run().info.run_id}')

if __name__ == '__main__':
    main()
