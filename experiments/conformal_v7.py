"""
Conformal Prediction Wrapper for Meta v7 (HC #488 — MODEL axis)

Converts v7's uncalibrated point predictions into calibrated prediction intervals
with guaranteed coverage using split conformal prediction (Vovk et al.).

Two approaches:
  A) Split Conformal: nonconformity scores on calibration half -> quantiles -> intervals
  B) Auxiliary Residual MLP: learn to predict |y_true - v7_pred| for heteroscedastic uncertainty

Key outputs:
  - Coverage calibration (actual vs nominal at 80/90/95%)
  - Interval width distribution
  - Conditional coverage on strong signals
  - P&L conditioned on interval width (narrow = high confidence)
  - Adaptive Kelly sizing simulation
  - Sharpness score filtering: does high-sharpness lift Sharpe?

Data: /home/nick/Lvl3Quant/output/meta_v7_prod/concat_oot_predictions.npz
      + per-fold npz for fold boundaries
      + raw features reconstructed from MBO + CNN-Mamba for auxiliary model
"""

import os
import sys
import json
import time
import logging
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from datetime import datetime
from collections import OrderedDict

# MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'output_dir': '/home/nick/Lvl3Quant/output/conformal_v7',
    'v7_dir': '/home/nick/Lvl3Quant/output/meta_v7_prod',
    'cm_dir': '/home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_oot',
    'mbo_dir': '/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3',
    # Conformal settings
    'cal_fraction': 0.5,        # fraction of OOT used for calibration
    'coverage_levels': [0.80, 0.90, 0.95],
    # Auxiliary model
    'aux_hidden': [64, 32],
    'aux_dropout': 0.15,
    'aux_epochs': 30,
    'aux_lr': 5e-4,
    'aux_weight_decay': 1e-4,
    'aux_patience': 7,
    'aux_batch_size': 2048,
    # v7 fold structure
    'train_days': 10,
    'eval_days': 3,
    # Cost
    'commission_ticks': 0.376,
    'tick_value': 12.50,
}

# ============================================================
# Logging
# ============================================================
os.makedirs(CONFIG['output_dir'], exist_ok=True)
log_path = os.path.join(CONFIG['output_dir'], 'conformal.log')

class Formatter(logging.Formatter):
    def format(self, record):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"{ts} [CONFORMAL_V7] {record.levelname}: {record.msg}"

logger = logging.getLogger('conformal_v7')
logger.setLevel(logging.INFO)
logger.propagate = False

fh = logging.FileHandler(log_path)
fh.setFormatter(Formatter())
logger.addHandler(fh)

class FlushHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

sh = FlushHandler(sys.stdout)
sh.setFormatter(Formatter())
logger.addHandler(sh)


# ============================================================
# Auxiliary Residual MLP
# ============================================================
class ResidualMLP(nn.Module):
    """Small MLP that predicts |y_true - v7_pred| (absolute residual)."""
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.15):
        super().__init__()
        layers = []
        prev = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev = h
        layers.append(nn.Linear(prev, 1))
        layers.append(nn.Softplus())  # residual must be non-negative
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# Feature loading (mirrors meta_v7_prod.py)
# ============================================================
def load_date_features(date_str):
    """Load features for a date, same as meta_v7_prod.py."""
    import zipfile
    cm_path = os.path.join(CONFIG['cm_dir'], f'{date_str}_predictions.npz')
    if not os.path.exists(cm_path):
        return None, None, None

    cm_data = np.load(cm_path, allow_pickle=True)
    cm_preds = cm_data['predictions']
    cm_n = len(cm_preds)
    cm_window = int(cm_data['window_size'])
    cm_stride = int(cm_data['stride'])
    cm_event_idx = np.array([cm_window + i * cm_stride for i in range(cm_n)])

    mbo_path = os.path.join(CONFIG['mbo_dir'], f'{date_str}_mbo_events.npz')
    try:
        mbo_data = np.load(mbo_path)
    except Exception:
        return None, None, None

    mbo_events = mbo_data['events']
    mbo_labels_1s = mbo_data['labels_1s']
    n_mbo = len(mbo_events)

    valid_cm = cm_event_idx < n_mbo
    cm_event_idx = cm_event_idx[valid_cm]
    cm_preds = cm_preds[valid_cm]

    if len(cm_preds) < 10:
        return None, None, None

    mbo_at_cm = mbo_events[cm_event_idx]
    target = mbo_labels_1s[cm_event_idx]
    cm_confidence = np.abs(cm_preds)

    features = np.column_stack([
        cm_preds,
        cm_confidence,
        mbo_at_cm,
    ]).astype(np.float32)

    return features, target.astype(np.float32), date_str


# ============================================================
# Part A: Split Conformal Prediction
# ============================================================
def run_split_conformal(predictions, labels, dates):
    """
    Split conformal prediction on the concat OOT data.
    For each walk-forward fold (grouped by date), split into cal/test.
    """
    logger.info("=" * 60)
    logger.info("PART A: Split Conformal Prediction")
    logger.info("=" * 60)

    unique_dates = np.unique(dates)
    n_dates = len(unique_dates)
    logger.info(f"Total OOT dates: {n_dates}, total samples: {len(predictions)}")

    # Build per-date groups
    date_groups = OrderedDict()
    for dt in unique_dates:
        mask = dates == dt
        date_groups[dt] = {
            'preds': predictions[mask],
            'labels': labels[mask],
            'n': mask.sum(),
        }

    # We'll do conformal on the entire concat, splitting each date 50/50
    # This preserves temporal structure within each fold
    all_results = {cov: {'actual_coverage': [], 'interval_widths': [], 'test_preds': [],
                          'test_labels': [], 'test_intervals': []}
                   for cov in CONFIG['coverage_levels']}

    # Global approach: pool calibration scores across all dates, test on remaining
    # More principled: per-date calibration (respects non-stationarity)
    # We'll do BOTH and compare

    # --- Method 1: Per-date conformal ---
    logger.info("\n--- Method 1: Per-Date Conformal ---")
    per_date_results = {cov: {'coverages': [], 'widths': [], 'test_preds': [],
                               'test_labels': []}
                        for cov in CONFIG['coverage_levels']}

    for dt, grp in date_groups.items():
        n = grp['n']
        if n < 100:
            logger.info(f"  {dt}: only {n} samples, skip")
            continue

        # Random split for cal/test
        rng = np.random.RandomState(int(dt))
        idx = rng.permutation(n)
        n_cal = int(n * CONFIG['cal_fraction'])
        cal_idx, test_idx = idx[:n_cal], idx[n_cal:]

        cal_preds = grp['preds'][cal_idx]
        cal_labels = grp['labels'][cal_idx]
        test_preds = grp['preds'][test_idx]
        test_labels = grp['labels'][test_idx]

        # Nonconformity scores
        cal_scores = np.abs(cal_labels - cal_preds)

        for cov in CONFIG['coverage_levels']:
            # Quantile with finite-sample correction
            q = np.ceil((1 + len(cal_scores)) * cov) / len(cal_scores)
            q = min(q, 1.0)
            quantile = np.quantile(cal_scores, q)

            # Prediction intervals on test set
            lower = test_preds - quantile
            upper = test_preds + quantile
            width = upper - lower  # = 2 * quantile (constant for this date)

            # Actual coverage
            covered = (test_labels >= lower) & (test_labels <= upper)
            actual_cov = covered.mean()

            per_date_results[cov]['coverages'].append(actual_cov)
            per_date_results[cov]['widths'].append(quantile * 2)
            per_date_results[cov]['test_preds'].extend(test_preds.tolist())
            per_date_results[cov]['test_labels'].extend(test_labels.tolist())

        logger.info(f"  {dt}: n_cal={n_cal}, n_test={n-n_cal}, "
                     f"90% coverage={per_date_results[0.90]['coverages'][-1]:.3f}, "
                     f"width={per_date_results[0.90]['widths'][-1]:.3f}")

    logger.info("\nPer-Date Conformal Summary:")
    conformal_summary = {}
    for cov in CONFIG['coverage_levels']:
        coverages = per_date_results[cov]['coverages']
        widths = per_date_results[cov]['widths']
        if len(coverages) == 0:
            continue
        avg_cov = np.mean(coverages)
        std_cov = np.std(coverages)
        avg_width = np.mean(widths)
        logger.info(f"  {cov*100:.0f}% nominal: actual={avg_cov:.4f} +/- {std_cov:.4f}, "
                     f"avg_width={avg_width:.3f} ticks")
        cov_label = f'{int(cov*100)}pct'
        conformal_summary[f'perdate_{cov_label}_actual_coverage'] = float(avg_cov)
        conformal_summary[f'perdate_{cov_label}_coverage_std'] = float(std_cov)
        conformal_summary[f'perdate_{cov_label}_avg_width'] = float(avg_width)

    # --- Method 2: Pooled conformal with rolling calibration ---
    logger.info("\n--- Method 2: Rolling Pooled Conformal ---")
    # Use first half of dates as calibration, second half as test
    sorted_dates = sorted(unique_dates)
    n_cal_dates = len(sorted_dates) // 2
    cal_dates = set(sorted_dates[:n_cal_dates])
    test_dates = set(sorted_dates[n_cal_dates:])

    cal_mask = np.isin(dates, list(cal_dates))
    test_mask = np.isin(dates, list(test_dates))

    cal_preds = predictions[cal_mask]
    cal_labels = labels[cal_mask]
    test_preds = predictions[test_mask]
    test_labels = labels[test_mask]

    cal_scores = np.abs(cal_labels - cal_preds)

    logger.info(f"  Cal dates: {n_cal_dates} ({cal_mask.sum()} samples)")
    logger.info(f"  Test dates: {len(sorted_dates) - n_cal_dates} ({test_mask.sum()} samples)")

    pooled_summary = {}
    for cov in CONFIG['coverage_levels']:
        q = np.ceil((1 + len(cal_scores)) * cov) / len(cal_scores)
        q = min(q, 1.0)
        quantile = np.quantile(cal_scores, q)

        lower = test_preds - quantile
        upper = test_preds + quantile
        covered = (test_labels >= lower) & (test_labels <= upper)
        actual_cov = covered.mean()
        width = quantile * 2

        # Per-date breakdown on test set
        per_date_cov = []
        for dt in sorted(test_dates):
            dt_mask = dates[test_mask] == dt
            if dt_mask.sum() < 10:
                continue
            dt_cov = covered[dt_mask].mean()
            per_date_cov.append(dt_cov)

        logger.info(f"  {cov*100:.0f}% nominal: actual={actual_cov:.4f}, width={width:.3f}, "
                     f"per-date range=[{min(per_date_cov):.3f}, {max(per_date_cov):.3f}]")

        cov_label = f'{int(cov*100)}pct'
        pooled_summary[f'pooled_{cov_label}_actual_coverage'] = float(actual_cov)
        pooled_summary[f'pooled_{cov_label}_width'] = float(width)
        pooled_summary[f'pooled_{cov_label}_min_date_coverage'] = float(min(per_date_cov))
        pooled_summary[f'pooled_{cov_label}_max_date_coverage'] = float(max(per_date_cov))

    # --- Conditional coverage on strong signals ---
    logger.info("\n--- Conditional Coverage: Strong Signals ---")
    # Use pooled calibration quantile at 90%
    q90 = np.quantile(cal_scores, min(np.ceil((1 + len(cal_scores)) * 0.90) / len(cal_scores), 1.0))

    abs_test_preds = np.abs(test_preds)
    p95_threshold = np.percentile(abs_test_preds, 95)
    p90_threshold = np.percentile(abs_test_preds, 90)

    strong_mask_5 = abs_test_preds >= p95_threshold
    strong_mask_10 = abs_test_preds >= p90_threshold

    lower_all = test_preds - q90
    upper_all = test_preds + q90
    covered_all = (test_labels >= lower_all) & (test_labels <= upper_all)

    cov_top5 = covered_all[strong_mask_5].mean() if strong_mask_5.sum() > 0 else 0
    cov_top10 = covered_all[strong_mask_10].mean() if strong_mask_10.sum() > 0 else 0
    cov_rest = covered_all[~strong_mask_10].mean()

    logger.info(f"  90% interval — Top 5% signals: coverage={cov_top5:.4f} (n={strong_mask_5.sum()})")
    logger.info(f"  90% interval — Top 10% signals: coverage={cov_top10:.4f} (n={strong_mask_10.sum()})")
    logger.info(f"  90% interval — Bottom 90% signals: coverage={cov_rest:.4f}")

    conditional_summary = {
        'cond_coverage_top5': float(cov_top5),
        'cond_coverage_top10': float(cov_top10),
        'cond_coverage_bottom90': float(cov_rest),
    }

    # --- P&L conditioned on interval width (using per-date varying widths) ---
    logger.info("\n--- P&L vs Interval Width (Per-Date Varying Widths) ---")
    # Re-run per-date conformal on ALL data (no cal/test split) using leave-one-date-out
    # For each test date, calibrate on all other dates
    all_sharpness = np.zeros(len(predictions))
    all_interval_widths = np.zeros(len(predictions))

    for i, dt in enumerate(sorted_dates):
        dt_mask = dates == dt
        other_mask = ~dt_mask

        other_scores = np.abs(labels[other_mask] - predictions[other_mask])
        q90_loo = np.quantile(other_scores, 0.90)

        # For this date, the conformal interval is prediction +/- q90_loo
        # But we want ADAPTIVE widths. Standard conformal gives constant width.
        # Sharpness here = 1 / q90_loo (same for all predictions on this date)
        all_interval_widths[dt_mask] = q90_loo * 2
        all_sharpness[dt_mask] = 1.0 / max(q90_loo * 2, 1e-6)

    # Split into narrow (high sharpness) vs wide (low sharpness) dates
    median_width = np.median(all_interval_widths)
    narrow_mask = all_interval_widths <= median_width
    wide_mask = all_interval_widths > median_width

    comm = CONFIG['commission_ticks']

    def compute_pnl_metrics(preds, lbls, name):
        if len(preds) < 10:
            return {}
        # Top 10% by confidence
        threshold = np.percentile(np.abs(preds), 90)
        strong = np.abs(preds) >= threshold
        trade_pnl = np.sign(preds[strong]) * lbls[strong] - comm
        if len(trade_pnl) == 0:
            return {}
        net = trade_pnl.sum()
        avg = trade_pnl.mean()
        wr = (trade_pnl > 0).mean()
        sharpe = trade_pnl.mean() / max(trade_pnl.std(), 1e-8) * np.sqrt(252)
        pf = trade_pnl[trade_pnl > 0].sum() / max(-trade_pnl[trade_pnl < 0].sum(), 1e-8)
        logger.info(f"  {name}: n_trades={len(trade_pnl)}, net={net:.1f} ticks, "
                     f"avg={avg:.3f}, WR={wr:.3f}, Sharpe~{sharpe:.2f}, PF={pf:.2f}")
        return {
            f'{name}_net_ticks': float(net),
            f'{name}_avg_ticks': float(avg),
            f'{name}_wr': float(wr),
            f'{name}_sharpe_approx': float(sharpe),
            f'{name}_pf': float(pf),
            f'{name}_n_trades': int(len(trade_pnl)),
        }

    narrow_metrics = compute_pnl_metrics(predictions[narrow_mask], labels[narrow_mask], 'narrow_interval')
    wide_metrics = compute_pnl_metrics(predictions[wide_mask], labels[wide_mask], 'wide_interval')
    all_metrics = compute_pnl_metrics(predictions, labels, 'all')

    # --- Sharpness-based filtering ---
    logger.info("\n--- Sharpness Filtering (standard conformal) ---")
    logger.info("Note: Standard conformal gives CONSTANT width per calibration set.")
    logger.info("The real value comes from the Auxiliary Model (Part B) which gives per-prediction varying widths.")
    logger.info("Here we test date-level sharpness as a crude filter.")

    results_a = {
        **conformal_summary, **pooled_summary, **conditional_summary,
        **narrow_metrics, **wide_metrics, **all_metrics,
    }

    return results_a, all_sharpness


# ============================================================
# Part B: Auxiliary Residual MLP
# ============================================================
def run_auxiliary_model(predictions, labels, dates, device):
    """
    Train a small MLP to predict |y_true - v7_pred| using the same features as v7.
    Walk-forward: same fold structure as v7 (10d train, 3d eval).
    Features: 31 original v7 features + v7_pred + |v7_pred| = 33 features.
    """
    logger.info("\n" + "=" * 60)
    logger.info("PART B: Auxiliary Residual MLP")
    logger.info("=" * 60)

    # Get all available dates from CM predictions
    cm_dates = sorted([f.replace('_predictions.npz', '')
                        for f in os.listdir(CONFIG['cm_dir'])
                        if f.endswith('_predictions.npz')])
    logger.info(f"Available CM dates: {len(cm_dates)}")

    # Load features for all OOT dates
    # OOT dates from v7 concat
    oot_dates = sorted(np.unique(dates))
    logger.info(f"v7 OOT dates: {len(oot_dates)}")

    # For each OOT date, load raw features and match with v7 predictions
    # The meta_v7 predictions correspond to the same feature rows
    # We reconstruct features and pair them with v7 predictions + residuals

    all_features = []
    all_residuals = []
    all_dates_feat = []
    all_v7_preds = []
    all_labels_feat = []

    for dt in oot_dates:
        feats, target, _ = load_date_features(dt)
        if feats is None:
            logger.warning(f"  {dt}: cannot load features, skip")
            continue

        # Get v7 predictions for this date
        dt_mask = dates == dt
        v7_preds_dt = predictions[dt_mask]
        v7_labels_dt = labels[dt_mask]

        # Align — feature count may differ by a few rows due to boundary effects
        n_feat = len(feats)
        n_v7 = len(v7_preds_dt)
        if abs(n_feat - n_v7) > max(20, n_feat * 0.001):
            logger.warning(f"  {dt}: feature rows ({n_feat}) too different from v7 predictions ({n_v7}), skip")
            continue
        # Truncate to the shorter length
        n_use = min(n_feat, n_v7)
        feats = feats[:n_use]
        v7_preds_dt = v7_preds_dt[:n_use]
        v7_labels_dt = v7_labels_dt[:n_use]

        # Augmented features: original 31 + v7_pred + |v7_pred|
        v7_pred_col = v7_preds_dt.reshape(-1, 1)
        v7_abs_col = np.abs(v7_preds_dt).reshape(-1, 1)
        aug_feats = np.hstack([feats, v7_pred_col, v7_abs_col]).astype(np.float32)

        # Target: absolute residual
        residual = np.abs(v7_labels_dt - v7_preds_dt).astype(np.float32)

        all_features.append(aug_feats)
        all_residuals.append(residual)
        all_dates_feat.extend([dt] * len(feats))
        all_v7_preds.extend(v7_preds_dt.tolist())
        all_labels_feat.extend(v7_labels_dt.tolist())

    if len(all_features) == 0:
        logger.error("No features loaded — cannot train auxiliary model")
        return {}

    all_features = np.concatenate(all_features)
    all_residuals = np.concatenate(all_residuals)
    all_dates_feat = np.array(all_dates_feat)
    all_v7_preds = np.array(all_v7_preds)
    all_labels_feat = np.array(all_labels_feat)

    n_features = all_features.shape[1]
    logger.info(f"Loaded {len(all_features)} samples, {n_features} features (31 + v7_pred + |v7_pred|)")
    logger.info(f"Residual stats: mean={all_residuals.mean():.4f}, median={np.median(all_residuals):.4f}, "
                 f"std={all_residuals.std():.4f}, max={all_residuals.max():.4f}")

    # Walk-forward — use shorter windows since we only have v7 OOT dates (not full history)
    sorted_dates_all = sorted(np.unique(all_dates_feat))
    n_avail = len(sorted_dates_all)
    # Adaptive: if we have plenty of dates use v7 config, otherwise shrink
    if n_avail >= 20:
        train_days = CONFIG['train_days']
        eval_days = CONFIG['eval_days']
    else:
        train_days = max(3, n_avail // 4)
        eval_days = max(1, n_avail // 8)
    logger.info(f"Available dates for WF: {n_avail} (using {train_days}d train / {eval_days}d eval)")

    if n_avail < train_days + eval_days:
        logger.error(f"Not enough dates ({n_avail}) for WF with {train_days}+{eval_days}")
        return {}

    concat_oot_preds = []
    concat_oot_labels = []
    concat_oot_residuals = []
    concat_oot_pred_residuals = []
    concat_oot_v7preds = []
    concat_oot_dates = []
    fold_spearman_list = []

    n_folds = 0
    start_idx = 0

    while start_idx + train_days + eval_days <= n_avail:
        train_date_set = set(sorted_dates_all[start_idx:start_idx + train_days])
        eval_date_set = set(sorted_dates_all[start_idx + train_days:start_idx + train_days + eval_days])

        train_mask = np.isin(all_dates_feat, list(train_date_set))
        eval_mask = np.isin(all_dates_feat, list(eval_date_set))

        train_X = all_features[train_mask]
        train_y = all_residuals[train_mask]
        eval_X = all_features[eval_mask]
        eval_y = all_residuals[eval_mask]
        eval_v7 = all_v7_preds[eval_mask]
        eval_labels = all_labels_feat[eval_mask]
        eval_dt = all_dates_feat[eval_mask]

        if len(train_X) < 100 or len(eval_X) < 50:
            start_idx += eval_days
            continue

        # Normalize features
        feat_mean = train_X.mean(axis=0)
        feat_std = train_X.std(axis=0) + 1e-8
        train_X_norm = (train_X - feat_mean) / feat_std
        eval_X_norm = (eval_X - feat_mean) / feat_std

        # Train auxiliary model
        model = ResidualMLP(n_features, CONFIG['aux_hidden'], CONFIG['aux_dropout']).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['aux_lr'],
                                       weight_decay=CONFIG['aux_weight_decay'])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['aux_epochs'])

        train_ds = TensorDataset(
            torch.tensor(train_X_norm, dtype=torch.float32),
            torch.tensor(train_y, dtype=torch.float32)
        )
        train_loader = DataLoader(train_ds, batch_size=CONFIG['aux_batch_size'],
                                   shuffle=True, num_workers=4, pin_memory=True)

        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(CONFIG['aux_epochs']):
            model.train()
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                optimizer.zero_grad()
                pred = model(X_batch)
                loss = nn.HuberLoss(delta=1.0)(pred, y_batch)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            scheduler.step()

            # Validation
            model.eval()
            with torch.no_grad():
                eval_tensor = torch.tensor(eval_X_norm, dtype=torch.float32).to(device)
                eval_pred = model(eval_tensor).cpu().numpy()
                val_loss = np.mean(np.abs(eval_pred - eval_y))

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= CONFIG['aux_patience']:
                    break

        # Evaluate best model
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        with torch.no_grad():
            eval_tensor = torch.tensor(eval_X_norm, dtype=torch.float32).to(device)
            pred_residuals = model(eval_tensor).cpu().numpy()

        # Spearman correlation between predicted and actual residuals
        sp, sp_p = stats.spearmanr(pred_residuals, eval_y)
        fold_spearman_list.append(sp)

        logger.info(f"  Fold {n_folds}: train={len(train_X)}, eval={len(eval_X)}, "
                     f"dates={list(eval_date_set)}, "
                     f"residual_spearman={sp:.4f} (p={sp_p:.2e}), val_MAE={best_val_loss:.4f}")

        concat_oot_preds.append(eval_v7)
        concat_oot_labels.append(eval_labels)
        concat_oot_residuals.append(eval_y)
        concat_oot_pred_residuals.append(pred_residuals)
        concat_oot_v7preds.append(eval_v7)
        concat_oot_dates.append(eval_dt)

        n_folds += 1
        start_idx += eval_days  # slide by eval_days

    if n_folds == 0:
        logger.error("No folds completed for auxiliary model")
        return {}

    # Concatenate all OOT results
    concat_v7 = np.concatenate(concat_oot_preds)
    concat_labels = np.concatenate(concat_oot_labels)
    concat_true_res = np.concatenate(concat_oot_residuals)
    concat_pred_res = np.concatenate(concat_oot_pred_residuals)
    concat_dt = np.concatenate(concat_oot_dates)

    # Overall metrics
    overall_sp, _ = stats.spearmanr(concat_pred_res, concat_true_res)
    logger.info(f"\nAuxiliary Model Summary:")
    logger.info(f"  Folds: {n_folds}")
    logger.info(f"  Per-fold Spearman: {[f'{s:.4f}' for s in fold_spearman_list]}")
    logger.info(f"  Mean fold Spearman: {np.mean(fold_spearman_list):.4f}")
    logger.info(f"  Concat OOT Spearman: {overall_sp:.4f}")
    logger.info(f"  Concat samples: {len(concat_v7)}")

    # --- Adaptive conformal using predicted residuals ---
    logger.info("\n--- Adaptive Conformal (Auxiliary-based) ---")
    # Use predicted residual as the interval half-width (scaled)
    # Calibrate the scaling factor using a portion of the data

    n_total = len(concat_v7)
    cal_n = n_total // 2
    cal_idx = np.arange(cal_n)  # first half (temporal order)
    test_idx = np.arange(cal_n, n_total)

    for cov in CONFIG['coverage_levels']:
        # Conformity score: |y - pred| / predicted_residual
        cal_conf_scores = np.abs(concat_labels[cal_idx] - concat_v7[cal_idx]) / np.maximum(concat_pred_res[cal_idx], 1e-6)
        q = np.quantile(cal_conf_scores, cov)

        # Test intervals: v7_pred +/- q * predicted_residual
        test_half_width = q * concat_pred_res[test_idx]
        test_lower = concat_v7[test_idx] - test_half_width
        test_upper = concat_v7[test_idx] + test_half_width
        covered = (concat_labels[test_idx] >= test_lower) & (concat_labels[test_idx] <= test_upper)
        actual_cov = covered.mean()
        avg_width = (test_half_width * 2).mean()
        width_std = (test_half_width * 2).std()

        logger.info(f"  {cov*100:.0f}% adaptive: actual={actual_cov:.4f}, "
                     f"avg_width={avg_width:.3f} +/- {width_std:.3f}")

    # --- KEY ANALYSIS: P&L conditioned on predicted uncertainty ---
    logger.info("\n--- P&L vs Predicted Uncertainty (Auxiliary Model) ---")
    # Use test set only
    test_v7 = concat_v7[test_idx]
    test_labels_arr = concat_labels[test_idx]
    test_pred_res = concat_pred_res[test_idx]
    test_dt = concat_dt[test_idx]

    # Sharpness = 1 / predicted_residual (higher = more confident)
    sharpness = 1.0 / np.maximum(test_pred_res, 1e-6)

    # Split into quartiles of sharpness
    quartile_thresholds = np.percentile(sharpness, [25, 50, 75])
    quartile_masks = [
        sharpness <= quartile_thresholds[0],
        (sharpness > quartile_thresholds[0]) & (sharpness <= quartile_thresholds[1]),
        (sharpness > quartile_thresholds[1]) & (sharpness <= quartile_thresholds[2]),
        sharpness > quartile_thresholds[2],
    ]
    quartile_names = ['Q1 (wide/uncertain)', 'Q2', 'Q3', 'Q4 (narrow/confident)']

    comm = CONFIG['commission_ticks']
    pnl_by_quartile = {}

    for q_name, q_mask in zip(quartile_names, quartile_masks):
        q_preds = test_v7[q_mask]
        q_labels = test_labels_arr[q_mask]
        n_q = len(q_preds)

        if n_q < 50:
            continue

        # Trade ALL predictions in this quartile (directional trade)
        trade_pnl = np.sign(q_preds) * q_labels - comm
        net = trade_pnl.sum()
        avg = trade_pnl.mean()
        wr = (trade_pnl > 0).mean()
        sharpe_approx = trade_pnl.mean() / max(trade_pnl.std(), 1e-8) * np.sqrt(252)

        # Also look at top-10% within this quartile
        if n_q >= 100:
            top10_thr = np.percentile(np.abs(q_preds), 90)
            top10_mask = np.abs(q_preds) >= top10_thr
            top10_pnl = np.sign(q_preds[top10_mask]) * q_labels[top10_mask] - comm
            top10_avg = top10_pnl.mean()
            top10_wr = (top10_pnl > 0).mean()
        else:
            top10_avg = 0
            top10_wr = 0

        logger.info(f"  {q_name}: n={n_q}, all_avg={avg:.4f}, all_WR={wr:.3f}, "
                     f"all_Sharpe~{sharpe_approx:.2f}, top10_avg={top10_avg:.4f}, top10_WR={top10_wr:.3f}")

        pnl_by_quartile[q_name] = {
            'n': n_q, 'avg_ticks': float(avg), 'wr': float(wr),
            'sharpe_approx': float(sharpe_approx), 'net_ticks': float(net),
            'top10_avg': float(top10_avg), 'top10_wr': float(top10_wr),
        }

    # --- Sharpness-filtered Sharpe comparison ---
    logger.info("\n--- Sharpness Filtering: Does High-Confidence Lift Sharpe? ---")

    # Top 50% sharpness (most confident half)
    sharp_median = np.median(sharpness)
    confident_mask = sharpness >= sharp_median

    # Compare: all trades vs confident-only trades (using top-10% signal strength)
    for name, mask in [('all_predictions', np.ones(len(test_v7), dtype=bool)),
                        ('high_sharpness_only', confident_mask)]:
        sub_preds = test_v7[mask]
        sub_labels = test_labels_arr[mask]
        if len(sub_preds) < 100:
            continue

        top10_thr = np.percentile(np.abs(sub_preds), 90)
        top10 = np.abs(sub_preds) >= top10_thr
        pnl = np.sign(sub_preds[top10]) * sub_labels[top10] - comm

        net = pnl.sum()
        avg = pnl.mean()
        wr = (pnl > 0).mean()
        sharpe = pnl.mean() / max(pnl.std(), 1e-8) * np.sqrt(252)
        sortino_denom = pnl[pnl < 0].std() if (pnl < 0).sum() > 1 else 1e-8
        sortino = pnl.mean() / sortino_denom * np.sqrt(252)
        pf = pnl[pnl > 0].sum() / max(-pnl[pnl < 0].sum(), 1e-8)

        logger.info(f"  {name} top-10%: n={len(pnl)}, avg={avg:.4f}, WR={wr:.3f}, "
                     f"Sharpe~{sharpe:.2f}, Sortino~{sortino:.2f}, PF={pf:.2f}")

    # --- Adaptive Kelly sizing simulation ---
    logger.info("\n--- Adaptive Kelly Sizing Simulation ---")
    # Kelly fraction proportional to sharpness (confidence)
    # f* = edge / odds, approximated as: f* proportional to 1/predicted_uncertainty

    # Normalize sharpness to [0, 1] range for sizing
    sharp_min, sharp_max = sharpness.min(), sharpness.max()
    kelly_frac = (sharpness - sharp_min) / (sharp_max - sharp_min + 1e-8)
    # Clip to reasonable range [0.1, 1.0] — never zero, never overleveraged
    kelly_frac = np.clip(kelly_frac * 0.9 + 0.1, 0.1, 1.0)

    # Only trade top-10% signals
    top10_thr = np.percentile(np.abs(test_v7), 90)
    top10_mask = np.abs(test_v7) >= top10_thr

    # Fixed sizing P&L
    fixed_pnl = np.sign(test_v7[top10_mask]) * test_labels_arr[top10_mask] - comm
    # Kelly sizing P&L (scale by kelly_frac, but commission is fixed per contract)
    kelly_pnl = kelly_frac[top10_mask] * (np.sign(test_v7[top10_mask]) * test_labels_arr[top10_mask]) - comm

    fixed_sharpe = fixed_pnl.mean() / max(fixed_pnl.std(), 1e-8) * np.sqrt(252)
    kelly_sharpe = kelly_pnl.mean() / max(kelly_pnl.std(), 1e-8) * np.sqrt(252)

    fixed_net = fixed_pnl.sum()
    kelly_net = kelly_pnl.sum()

    logger.info(f"  Fixed sizing: net={fixed_net:.1f} ticks, Sharpe~{fixed_sharpe:.2f}")
    logger.info(f"  Kelly sizing: net={kelly_net:.1f} ticks, Sharpe~{kelly_sharpe:.2f}")
    logger.info(f"  Sharpe improvement: {(kelly_sharpe - fixed_sharpe) / max(abs(fixed_sharpe), 1e-8) * 100:.1f}%")

    aux_results = {
        'aux_n_folds': n_folds,
        'aux_mean_fold_spearman': float(np.mean(fold_spearman_list)),
        'aux_concat_spearman': float(overall_sp),
        'aux_concat_samples': int(len(concat_v7)),
        'aux_pnl_by_quartile': pnl_by_quartile,
        'kelly_fixed_sharpe': float(fixed_sharpe),
        'kelly_adaptive_sharpe': float(kelly_sharpe),
        'kelly_sharpe_improvement_pct': float((kelly_sharpe - fixed_sharpe) / max(abs(fixed_sharpe), 1e-8) * 100),
    }

    # Save per-prediction uncertainty estimates
    np.savez_compressed(
        os.path.join(CONFIG['output_dir'], 'uncertainty_estimates.npz'),
        v7_predictions=concat_v7,
        labels=concat_labels,
        predicted_residuals=concat_pred_res,
        sharpness=np.concatenate([np.zeros(cal_n), sharpness]),  # cal part = 0 (not estimated)
        dates=concat_dt,
    )
    logger.info(f"\nSaved uncertainty estimates to {CONFIG['output_dir']}/uncertainty_estimates.npz")

    return aux_results


# ============================================================
# Main
# ============================================================
def main():
    t0 = time.time()
    logger.info("=" * 70)
    logger.info("Conformal Prediction for Meta v7 — START")
    logger.info(f"Config: {json.dumps(CONFIG, indent=2, default=str)}")
    logger.info("=" * 70)
    sys.stdout.flush()

    # MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("conformal_v7")
        mlflow.start_run(run_name=f"conformal_v7_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            'cal_fraction': CONFIG['cal_fraction'],
            'coverage_levels': str(CONFIG['coverage_levels']),
            'aux_hidden': str(CONFIG['aux_hidden']),
            'aux_epochs': CONFIG['aux_epochs'],
            'aux_lr': CONFIG['aux_lr'],
            'commission_ticks': CONFIG['commission_ticks'],
        })

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # Load v7 concat predictions
    logger.info("\nLoading v7 concat predictions...")
    data = np.load(os.path.join(CONFIG['v7_dir'], 'concat_oot_predictions.npz'))
    predictions = data['predictions']
    labels = data['labels']
    dates = data['dates']
    logger.info(f"Loaded {len(predictions)} predictions, {len(np.unique(dates))} dates")

    # Part A: Split Conformal
    results_a, sharpness_a = run_split_conformal(predictions, labels, dates)

    # Part B: Auxiliary Model
    results_b = run_auxiliary_model(predictions, labels, dates, device)

    # Combined results
    all_results = {**results_a, **results_b}

    # Save full results
    results_path = os.path.join(CONFIG['output_dir'], 'conformal_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to {results_path}")

    # MLflow logging
    if MLFLOW_AVAILABLE:
        # Log scalar metrics (flatten dicts, skip nested)
        for k, v in all_results.items():
            if isinstance(v, (int, float)):
                # Sanitize metric name: replace % and other invalid chars
                safe_k = k.replace('%', 'pct').replace(' ', '_')
                try:
                    mlflow.log_metric(safe_k, v)
                except Exception as e:
                    logger.warning(f"Failed to log metric {safe_k}: {e}")
        mlflow.log_artifact(results_path)
        mlflow.log_artifact(log_path)
        unc_path = os.path.join(CONFIG['output_dir'], 'uncertainty_estimates.npz')
        if os.path.exists(unc_path):
            mlflow.log_artifact(unc_path)
        mlflow.end_run()
        logger.info("MLflow run logged and ended")

    elapsed = time.time() - t0
    logger.info(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    logger.info("=" * 70)
    logger.info("Conformal Prediction for Meta v7 — COMPLETE")
    logger.info("=" * 70)


if __name__ == '__main__':
    main()
