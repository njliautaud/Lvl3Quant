#!/usr/bin/env python3
"""
XGBoost Execution Gate with CNN-Mamba Embeddings v1

Combines:
- CNN-Mamba predictions + embeddings (rich vol/regime info)
- Execution features (microstructure context)
- XGBoost for small-dataset robustness

Walk-forward: leave-one-fold-out across 11 CNN-Mamba OOT folds.
Target: Binary — is 10s return > 0.376 ticks (profitable limit order)?
"""

import os
import sys
import re
import glob
import time
import json
import warnings
import traceback
import numpy as np
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.metrics import roc_auc_score, precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler
from datetime import datetime

warnings.filterwarnings('ignore')

# ── Config ──────────────────────────────────────────────────────────────────
FOLD_DIR = "/home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar"
EXEC_DIR = "/home/nick/Lvl3Quant/output/exec_features_v1"
OUTPUT_DIR = "/home/nick/Lvl3Quant/output/xgb_exec_embed_v1"
MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "xgb_exec_embed"

COST_TICKS = 0.376  # RT commission for passive limit order
TICK_VALUE = 12.50
PCA_DIMS = 32
PRED_STRIDE_RATIO = 10  # fold predictions have 10x more samples than exec features

# Top 15 exec features to use
TOP_EXEC_FEATURES = [
    'bid_depth_l1', 'ask_depth_l1',
    'queue_consumption_velocity_bid', 'queue_consumption_velocity_ask',
    'book_turnover', 'time_in_spread', 'depth_imbalance_l1',
    'trade_rate', 'spread_mean_10k',
    'cancel_velocity_bid', 'cancel_velocity_ask',
    'tod_sin', 'tod_cos',
    'price_volatility_window', 'queue_replenish_ratio'
]

# XGBoost params
XGB_PARAMS = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'tree_method': 'hist',
    'device': 'cuda',
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.7,
    'min_child_weight': 10,
    'gamma': 0.1,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'scale_pos_weight': 1.0,  # Will be adjusted based on class balance
    'seed': 42,
    'verbosity': 0,
}

THRESHOLDS = [0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8]


def load_fold_data(fold_path):
    """Load CNN-Mamba fold predictions + embeddings."""
    f = np.load(fold_path, allow_pickle=True)
    return {
        'predictions': f['predictions'],    # (N, 3) for 1s, 5s, 10s
        'labels': f['labels'],              # (N, 3)
        'embeddings': f['embeddings'],      # (N, 96)
        'oot_files': f['oot_files'],        # array of file paths
        'horizons': f['horizons'],          # ['1s', '5s', '10s']
    }


def load_exec_features(date_str):
    """Load execution features for a given date."""
    path = os.path.join(EXEC_DIR, f"{date_str}_exec_features.npz")
    if not os.path.exists(path):
        return None
    f = np.load(path, allow_pickle=True)
    return {
        'features': f['features'],          # (M, 44)
        'feature_names': list(f['feature_names']),
    }


def align_data(fold_data, exec_data):
    """
    Align fold predictions (stride ~500) with exec features (stride 5000).
    Downsample fold data by averaging groups of ~10 samples to match exec windows.
    """
    n_fold = fold_data['predictions'].shape[0]
    n_exec = exec_data['features'].shape[0]

    if n_exec == 0 or n_fold == 0:
        return None

    # Calculate actual ratio
    ratio = n_fold / n_exec

    # Downsample fold data to match exec feature count
    # Use block averaging for predictions/embeddings, take the label at the center
    aligned_preds = np.zeros((n_exec, 3), dtype=np.float32)
    aligned_labels = np.zeros((n_exec, 3), dtype=np.float32)
    aligned_embeds = np.zeros((n_exec, fold_data['embeddings'].shape[1]), dtype=np.float32)

    for i in range(n_exec):
        start = int(i * ratio)
        end = int((i + 1) * ratio)
        end = min(end, n_fold)
        if start >= n_fold:
            start = n_fold - 1
            end = n_fold

        # Average predictions and embeddings over the window
        aligned_preds[i] = fold_data['predictions'][start:end].mean(axis=0)
        aligned_labels[i] = fold_data['labels'][start:end].mean(axis=0)
        aligned_embeds[i] = fold_data['embeddings'][start:end].mean(axis=0)

    return aligned_preds, aligned_labels, aligned_embeds


def build_features(aligned_preds, aligned_embeds, exec_features, exec_names,
                   pca_model=None, scaler=None, fit=False):
    """
    Build feature matrix:
    - 9 prediction features (3 raw, 3 abs, 3 z-score)
    - 32 PCA embedding features
    - 15 exec features
    - 4 interaction features
    Total: ~60 features
    """
    n = aligned_preds.shape[0]
    feature_list = []
    feature_names = []

    # ── 1. Prediction features (9) ──
    # Raw predictions
    feature_list.append(aligned_preds)
    feature_names.extend(['pred_1s', 'pred_5s', 'pred_10s'])

    # Absolute predictions
    feature_list.append(np.abs(aligned_preds))
    feature_names.extend(['abs_pred_1s', 'abs_pred_5s', 'abs_pred_10s'])

    # Z-score predictions (standardize within this batch)
    pred_mean = aligned_preds.mean(axis=0, keepdims=True)
    pred_std = aligned_preds.std(axis=0, keepdims=True) + 1e-8
    z_preds = (aligned_preds - pred_mean) / pred_std
    feature_list.append(z_preds)
    feature_names.extend(['zscore_pred_1s', 'zscore_pred_5s', 'zscore_pred_10s'])

    # ── 2. PCA embeddings (32) ──
    if fit:
        if pca_model is None:
            pca_model = PCA(n_components=PCA_DIMS, random_state=42)
        pca_embeds = pca_model.fit_transform(aligned_embeds)
    else:
        pca_embeds = pca_model.transform(aligned_embeds)
    feature_list.append(pca_embeds.astype(np.float32))
    feature_names.extend([f'emb_pca_{i}' for i in range(PCA_DIMS)])

    # ── 3. Top exec features (15) ──
    exec_idx = []
    for name in TOP_EXEC_FEATURES:
        if name in exec_names:
            exec_idx.append(exec_names.index(name))
    selected_exec = exec_features[:, exec_idx]

    # Standardize exec features
    if fit:
        if scaler is None:
            scaler = StandardScaler()
        selected_exec_scaled = scaler.fit_transform(selected_exec)
    else:
        selected_exec_scaled = scaler.transform(selected_exec)
    feature_list.append(selected_exec_scaled.astype(np.float32))
    feature_names.extend([f'exec_{name}' for name in TOP_EXEC_FEATURES if name in exec_names])

    # ── 4. Interaction features (4) ──
    pred_10s = aligned_preds[:, 2]  # 10s prediction
    vol_idx = exec_names.index('price_volatility_window') if 'price_volatility_window' in exec_names else None
    spread_idx = exec_names.index('spread_mean_10k') if 'spread_mean_10k' in exec_names else None
    tod_cos_idx = exec_names.index('tod_cos') if 'tod_cos' in exec_names else None

    interactions = []
    int_names = []

    if vol_idx is not None:
        interactions.append((pred_10s * exec_features[:, vol_idx]).reshape(-1, 1))
        int_names.append('pred10s_x_vol')

    if spread_idx is not None:
        interactions.append((pred_10s * exec_features[:, spread_idx]).reshape(-1, 1))
        int_names.append('pred10s_x_spread')

    # fill_prob interaction
    fill_idx = exec_names.index('fill_prob_10s') if 'fill_prob_10s' in exec_names else None
    if fill_idx is not None:
        interactions.append((pred_10s * exec_features[:, fill_idx]).reshape(-1, 1))
        int_names.append('pred10s_x_fillprob')

    if tod_cos_idx is not None:
        interactions.append((np.abs(pred_10s) * exec_features[:, tod_cos_idx]).reshape(-1, 1))
        int_names.append('abspred10s_x_todcos')

    if interactions:
        feature_list.append(np.hstack(interactions).astype(np.float32))
        feature_names.extend(int_names)

    X = np.hstack(feature_list)

    # Replace NaN/Inf
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X, feature_names, pca_model, scaler


def make_target(aligned_labels, horizon_idx=2):
    """
    Binary target: is the 10s return > COST_TICKS (profitable limit order)?
    Labels are in tick units (return predictions).
    """
    returns_10s = aligned_labels[:, horizon_idx]
    # Profitable if return exceeds cost
    target = (returns_10s > COST_TICKS).astype(np.int32)
    return target, returns_10s


def compute_metrics(y_true, y_prob, returns, threshold=0.5):
    """Compute trading metrics for a given threshold."""
    y_pred = (y_prob >= threshold).astype(int)
    n_trades = y_pred.sum()

    if n_trades == 0:
        return {
            'threshold': threshold,
            'n_trades': 0, 'wr': 0, 'avg_pnl': 0,
            'pf': 0, 'sharpe': 0, 'sortino': 0,
        }

    # Filter to traded samples
    traded_returns = returns[y_pred == 1]
    traded_pnl = (traded_returns - COST_TICKS) * TICK_VALUE  # PnL in dollars

    wins = (traded_pnl > 0).sum()
    losses = (traded_pnl <= 0).sum()
    wr = wins / n_trades if n_trades > 0 else 0

    gross_profit = traded_pnl[traded_pnl > 0].sum() if wins > 0 else 0
    gross_loss = abs(traded_pnl[traded_pnl <= 0].sum()) if losses > 0 else 1e-8
    pf = gross_profit / gross_loss if gross_loss > 0 else 0

    avg_pnl = traded_pnl.mean()
    std_pnl = traded_pnl.std() + 1e-8
    sharpe = (avg_pnl / std_pnl) * np.sqrt(252) if std_pnl > 1e-6 else 0

    downside = traded_pnl[traded_pnl < 0]
    downside_std = downside.std() + 1e-8 if len(downside) > 0 else 1e-8
    sortino = (avg_pnl / downside_std) * np.sqrt(252) if downside_std > 1e-6 else 0

    return {
        'threshold': threshold,
        'n_trades': int(n_trades),
        'wr': float(wr),
        'avg_pnl': float(avg_pnl),
        'pf': float(pf),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
    }


def run():
    print("=" * 70)
    print("XGBoost Execution Gate with CNN-Mamba Embeddings v1")
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Setup MLflow ──
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run = mlflow.start_run(run_name=f"xgb_exec_embed_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    mlflow.log_params({
        'model': 'xgboost',
        'device': 'cuda',
        'pca_dims': PCA_DIMS,
        'cost_ticks': COST_TICKS,
        'max_depth': XGB_PARAMS['max_depth'],
        'n_estimators': 500,
        'learning_rate': XGB_PARAMS['learning_rate'],
        'subsample': XGB_PARAMS['subsample'],
        'colsample_bytree': XGB_PARAMS['colsample_bytree'],
        'pred_stride_ratio': PRED_STRIDE_RATIO,
        'n_exec_features': len(TOP_EXEC_FEATURES),
        'n_interaction_features': 4,
    })

    # ── Load all fold data ──
    fold_paths = sorted(glob.glob(os.path.join(FOLD_DIR, "fold_*_oot_predictions.npz")))
    print(f"\nFound {len(fold_paths)} folds")

    folds = []
    for fp in fold_paths:
        fold_data = load_fold_data(fp)
        # Extract date from oot_files
        date_match = re.search(r'(\d{8})', str(fold_data['oot_files'][0]))
        if not date_match:
            print(f"  WARNING: Cannot extract date from {fp}, skipping")
            continue
        date_str = date_match.group(1)

        # Load exec features for this date
        exec_data = load_exec_features(date_str)
        if exec_data is None:
            print(f"  WARNING: No exec features for {date_str}, skipping")
            continue

        # Align data
        result = align_data(fold_data, exec_data)
        if result is None:
            print(f"  WARNING: Alignment failed for {date_str}, skipping")
            continue

        aligned_preds, aligned_labels, aligned_embeds = result

        folds.append({
            'date': date_str,
            'fold_path': fp,
            'aligned_preds': aligned_preds,
            'aligned_labels': aligned_labels,
            'aligned_embeds': aligned_embeds,
            'exec_features': exec_data['features'],
            'exec_names': exec_data['feature_names'],
            'n_samples': aligned_preds.shape[0],
        })
        print(f"  Fold {len(folds)-1}: date={date_str}, samples={aligned_preds.shape[0]}")

    print(f"\nLoaded {len(folds)} valid folds, total samples: {sum(f['n_samples'] for f in folds)}")

    if len(folds) < 3:
        print("ERROR: Not enough folds for walk-forward evaluation")
        mlflow.end_run(status='FAILED')
        return

    # ── Walk-forward: Leave-one-fold-out ──
    print("\n" + "=" * 70)
    print("Walk-Forward Evaluation (Leave-One-Fold-Out)")
    print("=" * 70)

    all_oof_probs = []
    all_oof_targets = []
    all_oof_returns = []
    all_oof_dates = []
    fold_results = []
    all_importances = []

    for test_idx in range(len(folds)):
        test_fold = folds[test_idx]
        train_folds = [f for i, f in enumerate(folds) if i != test_idx]

        print(f"\n--- Test fold {test_idx}: {test_fold['date']} ({test_fold['n_samples']} samples) ---")
        print(f"    Training on {len(train_folds)} folds")

        # Build training data
        train_preds_list = []
        train_labels_list = []
        train_embeds_list = []
        train_exec_list = []

        for tf in train_folds:
            train_preds_list.append(tf['aligned_preds'])
            train_labels_list.append(tf['aligned_labels'])
            train_embeds_list.append(tf['aligned_embeds'])
            train_exec_list.append(tf['exec_features'])

        train_preds = np.vstack(train_preds_list)
        train_labels = np.vstack(train_labels_list)
        train_embeds = np.vstack(train_embeds_list)
        train_exec = np.vstack(train_exec_list)
        exec_names = train_folds[0]['exec_names']

        # Build features (fit PCA + scaler on training data)
        X_train, feat_names, pca_model, scaler = build_features(
            train_preds, train_embeds, train_exec, exec_names,
            fit=True
        )
        y_train, train_returns = make_target(train_labels)

        # Build test features (transform only)
        X_test, _, _, _ = build_features(
            test_fold['aligned_preds'], test_fold['aligned_embeds'],
            test_fold['exec_features'], test_fold['exec_names'],
            pca_model=pca_model, scaler=scaler, fit=False
        )
        y_test, test_returns = make_target(test_fold['aligned_labels'])

        # Class balance
        pos_rate = y_train.mean()
        neg_rate = 1 - pos_rate
        spw = neg_rate / (pos_rate + 1e-8)
        print(f"    Train: {len(y_train)} samples, pos_rate={pos_rate:.3f}, scale_pos_weight={spw:.2f}")
        print(f"    Test:  {len(y_test)} samples, pos_rate={y_test.mean():.3f}")

        # Update XGBoost params with class balance
        params = XGB_PARAMS.copy()
        params['scale_pos_weight'] = spw

        # Train XGBoost
        dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feat_names)
        dtest = xgb.DMatrix(X_test, label=y_test, feature_names=feat_names)

        # Split training data for early stopping (last fold as validation)
        val_size = min(len(train_folds[-1]['aligned_preds']), len(X_train) // 5)
        X_val = X_train[-val_size:]
        y_val = y_train[-val_size:]
        X_tr = X_train[:-val_size]
        y_tr = y_train[:-val_size]

        dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=feat_names)
        dval = xgb.DMatrix(X_val, label=y_val, feature_names=feat_names)

        t0 = time.time()
        model = xgb.train(
            params,
            dtr,
            num_boost_round=500,
            evals=[(dtr, 'train'), (dval, 'val')],
            early_stopping_rounds=50,
            verbose_eval=False,
        )
        train_time = time.time() - t0

        # Predict on test
        y_prob = model.predict(dtest)

        # AUC
        try:
            auc = roc_auc_score(y_test, y_prob)
        except ValueError:
            auc = 0.5

        # Feature importance
        importance = model.get_score(importance_type='gain')
        all_importances.append(importance)

        # Precision at top percentiles
        sorted_idx = np.argsort(-y_prob)
        n_test = len(y_test)
        p5_n = max(1, int(0.05 * n_test))
        p10_n = max(1, int(0.10 * n_test))
        prec_5 = y_test[sorted_idx[:p5_n]].mean()
        prec_10 = y_test[sorted_idx[:p10_n]].mean()

        print(f"    AUC={auc:.4f}, P@5%={prec_5:.3f}, P@10%={prec_10:.3f}, "
              f"best_iter={model.best_iteration}, time={train_time:.1f}s")

        fold_results.append({
            'fold': test_idx,
            'date': test_fold['date'],
            'auc': auc,
            'prec_5': prec_5,
            'prec_10': prec_10,
            'n_test': n_test,
            'pos_rate': float(y_test.mean()),
            'best_iteration': model.best_iteration,
            'train_time': train_time,
        })

        all_oof_probs.append(y_prob)
        all_oof_targets.append(y_test)
        all_oof_returns.append(test_returns)
        all_oof_dates.append(np.full(len(y_test), test_fold['date']))

        # Log per-fold metrics
        mlflow.log_metrics({
            f'fold_{test_idx}_auc': auc,
            f'fold_{test_idx}_prec5': prec_5,
            f'fold_{test_idx}_prec10': prec_10,
        }, step=test_idx)

    # ── Aggregate results ──
    print("\n" + "=" * 70)
    print("AGGREGATE RESULTS")
    print("=" * 70)

    all_probs = np.concatenate(all_oof_probs)
    all_targets = np.concatenate(all_oof_targets)
    all_returns = np.concatenate(all_oof_returns)

    concat_auc = roc_auc_score(all_targets, all_probs)
    mean_auc = np.mean([r['auc'] for r in fold_results])
    std_auc = np.std([r['auc'] for r in fold_results])

    print(f"\nConcat AUC: {concat_auc:.4f}")
    print(f"Mean AUC:   {mean_auc:.4f} +/- {std_auc:.4f}")

    mlflow.log_metrics({
        'concat_auc': concat_auc,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
    })

    # ── Baseline: unfiltered (all z>=2) ──
    print("\n--- Baseline (Unfiltered) ---")
    baseline_pnl = (all_returns - COST_TICKS) * TICK_VALUE
    baseline_wr = (baseline_pnl > 0).mean()
    baseline_avg = baseline_pnl.mean()
    baseline_pf = (baseline_pnl[baseline_pnl > 0].sum() /
                   (abs(baseline_pnl[baseline_pnl <= 0].sum()) + 1e-8))
    print(f"  N={len(all_returns)}, WR={baseline_wr:.3f}, AvgPnL=${baseline_avg:.2f}, PF={baseline_pf:.2f}")

    mlflow.log_metrics({
        'baseline_n_trades': len(all_returns),
        'baseline_wr': float(baseline_wr),
        'baseline_avg_pnl': float(baseline_avg),
        'baseline_pf': float(baseline_pf),
    })

    # ── Trade simulation at various thresholds ──
    print("\n--- Trade Simulation (Filtered by XGBoost) ---")
    print(f"{'Thresh':>8} {'N_Trades':>10} {'WR':>8} {'AvgPnL':>10} {'PF':>8} {'Sharpe':>8} {'Sortino':>8}")
    print("-" * 70)

    sim_results = []
    for thresh in THRESHOLDS:
        metrics = compute_metrics(all_targets, all_probs, all_returns, threshold=thresh)
        sim_results.append(metrics)
        print(f"{thresh:>8.2f} {metrics['n_trades']:>10d} {metrics['wr']:>8.3f} "
              f"${metrics['avg_pnl']:>9.2f} {metrics['pf']:>8.2f} "
              f"{metrics['sharpe']:>8.2f} {metrics['sortino']:>8.2f}")

        mlflow.log_metrics({
            f't{int(thresh*100)}_n_trades': metrics['n_trades'],
            f't{int(thresh*100)}_wr': metrics['wr'],
            f't{int(thresh*100)}_avg_pnl': metrics['avg_pnl'],
            f't{int(thresh*100)}_pf': metrics['pf'],
            f't{int(thresh*100)}_sharpe': metrics['sharpe'],
            f't{int(thresh*100)}_sortino': metrics['sortino'],
        })

    # ── WR/PF improvement ──
    print("\n--- Improvement vs Baseline ---")
    for sr in sim_results:
        if sr['n_trades'] > 0:
            wr_imp = sr['wr'] - baseline_wr
            pf_imp = sr['pf'] - baseline_pf
            print(f"  Threshold {sr['threshold']:.2f}: WR {wr_imp:+.3f}, PF {pf_imp:+.2f}, "
                  f"Trades filtered: {len(all_returns) - sr['n_trades']}/{len(all_returns)}")

    # ── Feature importance ──
    print("\n" + "=" * 70)
    print("FEATURE IMPORTANCE (Average Gain)")
    print("=" * 70)

    # Aggregate importance across folds
    agg_importance = {}
    for imp in all_importances:
        for feat, gain in imp.items():
            if feat not in agg_importance:
                agg_importance[feat] = []
            agg_importance[feat].append(gain)

    # Average and sort
    avg_importance = {k: np.mean(v) for k, v in agg_importance.items()}
    sorted_importance = sorted(avg_importance.items(), key=lambda x: -x[1])

    # Categorize features
    embed_gain = 0
    exec_gain = 0
    pred_gain = 0
    inter_gain = 0

    print(f"\n{'Rank':>4} {'Feature':>30} {'Avg Gain':>12} {'Category':>12}")
    print("-" * 62)
    for i, (feat, gain) in enumerate(sorted_importance[:30]):
        if feat.startswith('emb_pca'):
            cat = 'embedding'
            embed_gain += gain
        elif feat.startswith('exec_'):
            cat = 'exec'
            exec_gain += gain
        elif feat.startswith('pred') or feat.startswith('abs_pred') or feat.startswith('zscore'):
            cat = 'prediction'
            pred_gain += gain
        else:
            cat = 'interaction'
            inter_gain += gain
        print(f"{i+1:>4} {feat:>30} {gain:>12.1f} {cat:>12}")

    total_gain = embed_gain + exec_gain + pred_gain + inter_gain + 1e-8
    print(f"\n{'Category':>20} {'Total Gain':>12} {'Share':>8}")
    print("-" * 42)
    print(f"{'Embeddings':>20} {embed_gain:>12.1f} {embed_gain/total_gain*100:>7.1f}%")
    print(f"{'Exec Features':>20} {exec_gain:>12.1f} {exec_gain/total_gain*100:>7.1f}%")
    print(f"{'Predictions':>20} {pred_gain:>12.1f} {pred_gain/total_gain*100:>7.1f}%")
    print(f"{'Interactions':>20} {inter_gain:>12.1f} {inter_gain/total_gain*100:>7.1f}%")

    mlflow.log_metrics({
        'importance_embed_pct': embed_gain / total_gain * 100,
        'importance_exec_pct': exec_gain / total_gain * 100,
        'importance_pred_pct': pred_gain / total_gain * 100,
        'importance_inter_pct': inter_gain / total_gain * 100,
    })

    # ── Save results ──
    results = {
        'concat_auc': concat_auc,
        'mean_auc': mean_auc,
        'std_auc': std_auc,
        'fold_results': fold_results,
        'simulation': sim_results,
        'baseline': {
            'n_trades': len(all_returns),
            'wr': float(baseline_wr),
            'avg_pnl': float(baseline_avg),
            'pf': float(baseline_pf),
        },
        'feature_importance': dict(sorted_importance[:30]),
        'category_importance': {
            'embeddings_pct': embed_gain / total_gain * 100,
            'exec_features_pct': exec_gain / total_gain * 100,
            'predictions_pct': pred_gain / total_gain * 100,
            'interactions_pct': inter_gain / total_gain * 100,
        },
    }

    results_path = os.path.join(OUTPUT_DIR, "results.json")
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {results_path}")

    # Save OOF predictions
    np.savez_compressed(
        os.path.join(OUTPUT_DIR, "oof_predictions.npz"),
        probabilities=all_probs,
        targets=all_targets,
        returns=all_returns,
    )

    mlflow.log_artifact(results_path)
    mlflow.end_run()

    print(f"\nMLflow run logged to {MLFLOW_URI}, experiment: {EXPERIMENT_NAME}")
    print(f"\nCompleted: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 70)


if __name__ == '__main__':
    try:
        run()
    except Exception as e:
        print(f"\nFATAL ERROR: {e}")
        traceback.print_exc()
        # Try to end MLflow run on error
        try:
            import mlflow
            mlflow.end_run(status='FAILED')
        except:
            pass
        sys.exit(1)
