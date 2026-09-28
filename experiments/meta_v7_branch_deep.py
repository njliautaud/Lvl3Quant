"""
Meta-Model v7 PRODUCTION (1s Horizon): CNN-Mamba + Microstructure

Production version with smaller sliding windows for maximum OOT coverage.
10d train / 3d eval -> ~10 folds for regime-agnostic validation (HC #428 R1).

Based on meta_v7_branch_deep.py — same architecture, features, lr, etc.
Only change: fold structure for more OOT dates.
"""

import os
import sys
import json
import time
import logging
import zipfile
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats
from datetime import datetime

# MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'output_dir': '/home/nick/Lvl3Quant/output/meta_v7_branch_deep',
    'cm_dir': '/home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_oot',
    'mbo_dir': '/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3',
    # Walk-forward — CHANGED for production: smaller windows, more folds
    'train_days': 10,
    'eval_days': 3,
    # Model (UNCHANGED from v7)
    'hidden_dims': [512, 256, 128, 64],
    'dropout': 0.3,
    'batch_size': 2048,
    'epochs': 20,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 5,
    # Cost constants
    'tick_value': 12.50,
    'commission_ticks': 0.376,
    # v6 baseline for comparison (5s target)
    'v6_spearman': 0.167,
    # KEY CHANGE: 1s target instead of 5s
    'target_horizon': '1s',
}

# ============================================================
# Logging
# ============================================================
os.makedirs(CONFIG['output_dir'], exist_ok=True)
log_path = os.path.join(CONFIG['output_dir'], 'training.log')

class MetaV7Formatter(logging.Formatter):
    def format(self, record):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"{ts} [META_V7_PROD] {record.levelname}: {record.msg}"

logger = logging.getLogger('meta_v7_prod')
logger.setLevel(logging.INFO)
logger.propagate = False

fh = logging.FileHandler(log_path)
fh.setFormatter(MetaV7Formatter())
logger.addHandler(fh)

class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

sh = FlushStreamHandler(sys.stdout)
sh.setFormatter(MetaV7Formatter())
logger.addHandler(sh)

# ============================================================
# Model Definition (UNCHANGED)
# ============================================================
class ConfluenceMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], dropout=0.2):
        super().__init__()
        layers = []
        prev_dim = input_dim
        for h in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h),
                nn.BatchNorm1d(h),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h
        layers.append(nn.Linear(prev_dim, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ============================================================
# Data Loading & Alignment (UNCHANGED from v7)
# ============================================================
def get_overlapping_dates():
    cm_dates = set(f[:8] for f in os.listdir(CONFIG['cm_dir'])
                   if f.endswith('_predictions.npz') and f[0] == '2')
    mbo_dates = set(f[:8] for f in os.listdir(CONFIG['mbo_dir'])
                    if f.endswith('_mbo_events.npz'))
    overlap = sorted(cm_dates & mbo_dates)
    logger.info(f"CNN-Mamba dates: {len(cm_dates)}, MBO: {len(mbo_dates)}, "
                f"overlap: {len(overlap)}")
    return overlap


def load_date_features(date_str):
    cm_path = os.path.join(CONFIG['cm_dir'], f'{date_str}_predictions.npz')
    cm_data = np.load(cm_path, allow_pickle=True)
    cm_preds = cm_data['predictions']
    cm_n = len(cm_preds)
    cm_window = int(cm_data['window_size'])
    cm_stride = int(cm_data['stride'])

    cm_event_idx = np.array([cm_window + i * cm_stride for i in range(cm_n)])

    mbo_path = os.path.join(CONFIG['mbo_dir'], f'{date_str}_mbo_events.npz')
    try:
        mbo_data = np.load(mbo_path)
    except (zipfile.BadZipFile, Exception) as e:
        logger.warning(f"  {date_str}: Bad MBO file ({e}), skip")
        return None, None
    mbo_events = mbo_data['events']
    mbo_labels_1s = mbo_data['labels_1s']
    n_mbo = len(mbo_events)

    valid_cm = cm_event_idx < n_mbo
    cm_event_idx = cm_event_idx[valid_cm]
    cm_preds = cm_preds[valid_cm]
    cm_n = len(cm_preds)

    if cm_n < 10:
        logger.warning(f"  {date_str}: Only {cm_n} valid CNN-Mamba rows, skipping")
        return None, None

    mbo_at_cm = mbo_events[cm_event_idx]
    target = mbo_labels_1s[cm_event_idx]
    cm_confidence = np.abs(cm_preds)

    features = np.column_stack([
        cm_preds,
        cm_confidence,
        mbo_at_cm,
    ]).astype(np.float32)

    return features, target.astype(np.float32)


# ============================================================
# Training (UNCHANGED)
# ============================================================
def train_fold(model, train_X, train_y, val_X, val_y, device, fold_idx):
    train_ds = TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(val_X, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                               shuffle=True, num_workers=8, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'] * 2,
                             shuffle=False, num_workers=8, pin_memory=True)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                                   weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss_sum = 0.0
        train_n = 0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            pred = model(X_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss_sum += loss.item() * len(y_batch)
            train_n += len(y_batch)

        scheduler.step()

        model.eval()
        val_loss_sum = 0.0
        val_n = 0
        val_preds_list = []
        val_labels_list = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                pred = model(X_batch)
                loss = criterion(pred, y_batch)
                val_loss_sum += loss.item() * len(y_batch)
                val_n += len(y_batch)
                val_preds_list.append(pred.cpu().numpy())
                val_labels_list.append(y_batch.cpu().numpy())

        train_loss = train_loss_sum / max(train_n, 1)
        val_loss = val_loss_sum / max(val_n, 1)

        val_preds_arr = np.concatenate(val_preds_list)
        val_labels_arr = np.concatenate(val_labels_list)
        spearman_r, _ = stats.spearmanr(val_preds_arr, val_labels_arr)

        if epoch % 5 == 0 or epoch == CONFIG['epochs'] - 1:
            logger.info(f"  Fold {fold_idx} Epoch {epoch}: "
                        f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                        f"spearman={spearman_r:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
            sys.stdout.flush()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                logger.info(f"  Fold {fold_idx}: Early stopping at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss


def evaluate_fold(model, eval_X, eval_y, device):
    model.eval()
    ds = TensorDataset(
        torch.tensor(eval_X, dtype=torch.float32),
        torch.tensor(eval_y, dtype=torch.float32)
    )
    loader = DataLoader(ds, batch_size=CONFIG['batch_size'] * 2,
                         shuffle=False, num_workers=0)
    all_preds = []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            pred = model(X_batch)
            all_preds.append(pred.cpu().numpy())
    return np.concatenate(all_preds)


def compute_metrics(predictions, labels, prefix=""):
    spearman_r, _ = stats.spearmanr(predictions, labels)
    pearson_r, _ = stats.pearsonr(predictions, labels)

    comm = CONFIG['commission_ticks']
    results = {
        f'{prefix}spearman': spearman_r,
        f'{prefix}pearson': pearson_r,
        f'{prefix}n_samples': len(predictions),
    }

    for pct_name, pct in [('top5', 95), ('top10', 90), ('top20', 80), ('top50', 50)]:
        threshold = np.percentile(np.abs(predictions), pct)
        mask = np.abs(predictions) >= threshold
        n_trades = mask.sum()
        if n_trades < 5:
            results[f'{prefix}{pct_name}_net_ticks'] = 0.0
            results[f'{prefix}{pct_name}_n_trades'] = 0
            results[f'{prefix}{pct_name}_wr'] = 0.0
            continue

        trade_pnl = np.sign(predictions[mask]) * labels[mask] - comm
        net_ticks = trade_pnl.sum()
        avg_ticks = trade_pnl.mean()
        win_rate = (trade_pnl > 0).mean()
        direction_correct = (np.sign(predictions[mask]) == np.sign(labels[mask]))
        precision = direction_correct.mean()

        results[f'{prefix}{pct_name}_net_ticks'] = float(net_ticks)
        results[f'{prefix}{pct_name}_avg_ticks'] = float(avg_ticks)
        results[f'{prefix}{pct_name}_n_trades'] = int(n_trades)
        results[f'{prefix}{pct_name}_wr'] = float(win_rate)
        results[f'{prefix}{pct_name}_precision'] = float(precision)

    return results


# ============================================================
# Main Walk-Forward Loop
# ============================================================
def main():
    logger.info("=" * 60)
    logger.info("Meta-Model v7 PRODUCTION (1s Horizon) — Training Start")
    logger.info("=" * 60)
    logger.info("PRODUCTION RUN: 10d train / 3d eval for max OOT coverage")
    logger.info("Same architecture/features/lr as v7 initial run")
    logger.info(f"Config: {json.dumps(CONFIG, indent=2, default=str)}")
    sys.stdout.flush()

    # MLflow setup
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("meta_v7_branch_deep")
        mlflow.start_run(run_name=f"meta_v7_deep_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            'target_horizon': '1s',
            'train_days': CONFIG['train_days'],
            'eval_days': CONFIG['eval_days'],
            'hidden_dims': str(CONFIG['hidden_dims']),
            'dropout': CONFIG['dropout'],
            'batch_size': CONFIG['batch_size'],
            'epochs': CONFIG['epochs'],
            'lr': CONFIG['lr'],
            'weight_decay': CONFIG['weight_decay'],
            'patience': CONFIG['patience'],
            'commission_ticks': CONFIG['commission_ticks'],
            'v6_baseline_spearman': CONFIG['v6_spearman'],
            'run_type': 'production',
        })
        logger.info("MLflow tracking enabled — experiment: meta_v7_prod")
    else:
        logger.warning("MLflow not available — training without tracking")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No CUDA device found — training will be slow on CPU")
    sys.stdout.flush()

    # Get overlapping dates
    dates = get_overlapping_dates()
    if len(dates) < CONFIG['train_days'] + CONFIG['eval_days']:
        logger.error(f"Not enough dates ({len(dates)}) for walk-forward")
        return

    # Pre-load all date data
    logger.info("Loading all date data...")
    sys.stdout.flush()
    date_data = {}
    valid_dates = []
    for d in dates:
        features, target = load_date_features(d)
        if features is None:
            continue
        if np.nanstd(target) < 1e-6:
            logger.warning(f"  {d}: ZERO-VARIANCE target, REJECTED")
            continue
        valid_mask = ~np.isnan(target)
        if valid_mask.mean() < 0.5:
            logger.warning(f"  {d}: Too many NaN targets, REJECTED")
            continue
        features = features[valid_mask]
        target = target[valid_mask]
        features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)
        date_data[d] = (features, target)
        valid_dates.append(d)
        logger.info(f"  Loaded {d}: {len(features)} samples, target std={np.std(target):.4f}")
        sys.stdout.flush()

    logger.info(f"Valid dates after filtering: {len(valid_dates)}")

    if len(valid_dates) < CONFIG['train_days'] + CONFIG['eval_days']:
        logger.error(f"Not enough valid dates ({len(valid_dates)})")
        return

    input_dim = date_data[valid_dates[0]][0].shape[1]
    logger.info(f"Input dimension: {input_dim} (expected 31)")
    if input_dim != 31:
        logger.warning(f"Expected 31 features but got {input_dim}")
    sys.stdout.flush()

    # Walk-forward: SLIDING window
    train_days = CONFIG['train_days']
    eval_days = CONFIG['eval_days']
    n_folds = (len(valid_dates) - train_days) // eval_days

    logger.info(f"Walk-forward: {n_folds} folds, {train_days}d train / {eval_days}d eval")
    logger.info(f"Expected OOT dates: {n_folds * eval_days}")
    sys.stdout.flush()

    if MLFLOW_AVAILABLE:
        mlflow.log_metrics({
            'n_valid_dates': len(valid_dates),
            'n_folds': n_folds,
            'input_dim': input_dim,
        })

    all_oot_preds = []
    all_oot_labels = []
    all_oot_dates = []
    fold_results = []

    for fold_idx in range(n_folds):
        fold_start = fold_idx * eval_days
        train_dates = valid_dates[fold_start:fold_start + train_days]
        eval_start = fold_start + train_days
        eval_end = min(eval_start + eval_days, len(valid_dates))
        eval_dates = valid_dates[eval_start:eval_end]

        if len(eval_dates) == 0:
            break

        logger.info(f"\n{'='*40}")
        logger.info(f"Fold {fold_idx}: Train {train_dates[0]}..{train_dates[-1]} "
                     f"({len(train_dates)}d) -> Eval {eval_dates[0]}..{eval_dates[-1]} "
                     f"({len(eval_dates)}d)")
        sys.stdout.flush()

        # Assemble train data
        train_X = np.concatenate([date_data[d][0] for d in train_dates])
        train_y = np.concatenate([date_data[d][1] for d in train_dates])

        # Normalize features using train statistics
        feat_mean = train_X.mean(axis=0)
        feat_std = train_X.std(axis=0) + 1e-8
        train_X_norm = (train_X - feat_mean) / feat_std

        logger.info(f"  Train: {len(train_X)} samples")

        # Build and train model
        model = ConfluenceMLP(
            input_dim=input_dim,
            hidden_dims=CONFIG['hidden_dims'],
            dropout=CONFIG['dropout']
        ).to(device)

        best_val_loss = train_fold(
            model, train_X_norm, train_y,
            train_X_norm[-len(train_X)//5:], train_y[-len(train_y)//5:],
            device, fold_idx
        )

        # Save model weights
        weight_path = os.path.join(CONFIG['output_dir'], f'fold_{fold_idx:02d}_model.pt')
        torch.save({
            'model_state_dict': model.state_dict(),
            'feat_mean': feat_mean,
            'feat_std': feat_std,
            'input_dim': input_dim,
            'config': CONFIG,
            'train_dates': train_dates,
            'eval_dates': eval_dates,
        }, weight_path)

        # Save normalization stats
        norm_path = os.path.join(CONFIG['output_dir'], f'fold_{fold_idx:02d}_norm_stats.npz')
        np.savez_compressed(norm_path, feat_mean=feat_mean, feat_std=feat_std)

        # Evaluate on OOT dates
        fold_preds = []
        fold_labels = []
        fold_date_labels = []
        for eval_date in eval_dates:
            eval_X, eval_y = date_data[eval_date]
            eval_X_norm = (eval_X - feat_mean) / feat_std
            preds = evaluate_fold(model, eval_X_norm, eval_y, device)
            fold_preds.append(preds)
            fold_labels.append(eval_y)
            fold_date_labels.extend([eval_date] * len(preds))

        fold_preds = np.concatenate(fold_preds)
        fold_labels = np.concatenate(fold_labels)

        # Save fold OOT predictions
        pred_path = os.path.join(CONFIG['output_dir'], f'fold_{fold_idx:02d}_oot_predictions.npz')
        np.savez_compressed(pred_path,
                            predictions=fold_preds,
                            labels=fold_labels,
                            dates=np.array(fold_date_labels),
                            train_dates=np.array(train_dates),
                            eval_dates=np.array(eval_dates),
                            feat_mean=feat_mean,
                            feat_std=feat_std)

        # Compute metrics
        metrics = compute_metrics(fold_preds, fold_labels, prefix=f'fold{fold_idx}_')
        fold_results.append(metrics)

        fold_sp = metrics[f'fold{fold_idx}_spearman']
        logger.info(f"  Fold {fold_idx} OOT: Spearman={fold_sp:.4f}, "
                     f"Pearson={metrics[f'fold{fold_idx}_pearson']:.4f}, "
                     f"n={metrics[f'fold{fold_idx}_n_samples']}")
        for pct in ['top5', 'top10', 'top20', 'top50']:
            key = f'fold{fold_idx}_{pct}_net_ticks'
            if key in metrics:
                logger.info(f"    {pct}: net={metrics[key]:.2f}t, "
                            f"WR={metrics.get(f'fold{fold_idx}_{pct}_wr', 0):.3f}, "
                            f"n={metrics.get(f'fold{fold_idx}_{pct}_n_trades', 0)}")
        sys.stdout.flush()

        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                f'fold_{fold_idx}_spearman': fold_sp,
                f'fold_{fold_idx}_val_loss': best_val_loss,
            }, step=fold_idx)

        all_oot_preds.append(fold_preds)
        all_oot_labels.append(fold_labels)
        all_oot_dates.extend(fold_date_labels)

    # ============================================================
    # Concat OOT Analysis
    # ============================================================
    if all_oot_preds:
        concat_preds = np.concatenate(all_oot_preds)
        concat_labels = np.concatenate(all_oot_labels)
        concat_dates = np.array(all_oot_dates)

        logger.info(f"\n{'='*60}")
        logger.info("CONCAT OOT RESULTS (all folds)")
        logger.info(f"{'='*60}")

        concat_metrics = compute_metrics(concat_preds, concat_labels, prefix='concat_')

        logger.info(f"Total samples: {concat_metrics['concat_n_samples']}")
        logger.info(f"Spearman: {concat_metrics['concat_spearman']:.4f}")
        logger.info(f"Pearson:  {concat_metrics['concat_pearson']:.4f}")

        for pct in ['top5', 'top10', 'top20', 'top50']:
            key = f'concat_{pct}_net_ticks'
            if key in concat_metrics:
                logger.info(f"{pct}: net={concat_metrics[key]:.2f}t, "
                            f"avg={concat_metrics.get(f'concat_{pct}_avg_ticks', 0):.4f}, "
                            f"WR={concat_metrics.get(f'concat_{pct}_wr', 0):.3f}, "
                            f"precision={concat_metrics.get(f'concat_{pct}_precision', 0):.3f}, "
                            f"n={concat_metrics.get(f'concat_{pct}_n_trades', 0)}")

        # v6 vs v7 comparison
        v6_spearman = CONFIG['v6_spearman']
        v7_spearman = concat_metrics['concat_spearman']
        delta = v7_spearman - v6_spearman
        pct_change = (delta / abs(v6_spearman)) * 100 if v6_spearman != 0 else 0.0

        logger.info(f"\n{'='*60}")
        logger.info("v6 (5s target) vs v7 (1s target) COMPARISON")
        logger.info(f"{'='*60}")
        logger.info(f"v6 Spearman (5s target): {v6_spearman:.4f}")
        logger.info(f"v7 Spearman (1s target): {v7_spearman:.4f}")
        logger.info(f"Delta: {delta:+.4f} ({pct_change:+.1f}%)")
        if delta > 0:
            logger.info("RESULT: v7 (1s) BEATS v6 (5s) -- 1s horizon hypothesis CONFIRMED")
        elif delta == 0:
            logger.info("RESULT: v7 MATCHES v6 -- same performance, faster horizon")
        else:
            logger.info(f"RESULT: v7 underperforms v6 by {abs(delta):.4f}")

        # Per-date breakdown
        logger.info(f"\nPer-date OOT breakdown:")
        unique_dates = sorted(set(all_oot_dates))
        date_sharpes = []
        positive_dates = 0
        for d in unique_dates:
            mask = concat_dates == d
            d_preds = concat_preds[mask]
            d_labels = concat_labels[mask]
            d_metrics = compute_metrics(d_preds, d_labels)
            d_spearman = d_metrics.get('spearman', 0)

            threshold = np.percentile(np.abs(d_preds), 80)
            d_mask = np.abs(d_preds) >= threshold
            if d_mask.sum() > 5:
                d_pnl = np.sign(d_preds[d_mask]) * d_labels[d_mask] - CONFIG['commission_ticks']
                d_sharpe = d_pnl.mean() / (d_pnl.std() + 1e-8) * np.sqrt(252)
                date_sharpes.append(d_sharpe)
                if d_spearman > 0:
                    positive_dates += 1
            else:
                d_sharpe = 0.0

            logger.info(f"  {d}: n={mask.sum()}, spearman={d_spearman:.4f}, "
                        f"daily_sharpe={d_sharpe:.2f}")

        if date_sharpes:
            avg_sharpe = np.mean(date_sharpes)
            logger.info(f"\nAvg daily Sharpe (top20%): {avg_sharpe:.2f}")
            logger.info(f"Positive Spearman days: {positive_dates}/{len(unique_dates)}")
            pos_sharpe_days = sum(1 for s in date_sharpes if s > 0)
            logger.info(f"Positive Sharpe days: {pos_sharpe_days}/{len(date_sharpes)}")

        # REGIME-AGNOSTIC ANALYSIS (HC #428 R1)
        logger.info(f"\n{'='*60}")
        logger.info("REGIME-AGNOSTIC ANALYSIS (HC #428 R1)")
        logger.info(f"{'='*60}")
        logger.info(f"Total OOT dates: {len(unique_dates)}")
        if date_sharpes:
            # Split into positive/negative sharpe halves as proxy for green/red days
            sorted_sharpes = sorted(zip(unique_dates, date_sharpes), key=lambda x: x[1])
            mid = len(sorted_sharpes) // 2
            red_half = sorted_sharpes[:mid]
            green_half = sorted_sharpes[mid:]
            
            red_avg = np.mean([s for _, s in red_half])
            green_avg = np.mean([s for _, s in green_half])
            
            if max(abs(green_avg), abs(red_avg)) > 0:
                regime_skew = abs(green_avg - red_avg) / max(abs(green_avg), abs(red_avg))
            else:
                regime_skew = 0.0
            
            logger.info(f"Bottom-half avg Sharpe (red proxy): {red_avg:.2f}")
            logger.info(f"Top-half avg Sharpe (green proxy): {green_avg:.2f}")
            logger.info(f"Regime skew ratio: {regime_skew:.3f} (reject if > 0.50)")
            if regime_skew > 0.50:
                logger.warning("REGIME SKEW ABOVE 0.50 — may be regime-tailored, not true edge")
            else:
                logger.info("REGIME SKEW OK — edge appears regime-agnostic")

        sys.stdout.flush()

        if MLFLOW_AVAILABLE:
            mlflow.log_metrics({
                'concat_spearman': float(v7_spearman),
                'concat_pearson': float(concat_metrics['concat_pearson']),
                'v6_v7_delta': float(delta),
                'n_oot_dates': len(unique_dates),
                'positive_spearman_dates': positive_dates,
                'avg_daily_sharpe_top20': float(avg_sharpe) if date_sharpes else 0.0,
            })

        # Save concat predictions
        concat_path = os.path.join(CONFIG['output_dir'], 'concat_oot_predictions.npz')
        np.savez_compressed(concat_path,
                            predictions=concat_preds,
                            labels=concat_labels,
                            dates=concat_dates)

        # Save summary
        summary = {
            'version': 'v7_prod_1s_horizon',
            'description': 'Meta-model v7 PRODUCTION: 10d train / 3d eval for max OOT coverage',
            'target_horizon': '1s',
            'input_dim': input_dim,
            'config': {k: str(v) if not isinstance(v, (int, float, str, list)) else v
                       for k, v in CONFIG.items()},
            'n_folds': len(fold_results),
            'n_valid_dates': len(valid_dates),
            'n_oot_dates': len(unique_dates),
            'v6_baseline_spearman': v6_spearman,
            'v7_spearman': float(v7_spearman),
            'v7_vs_v6_delta': float(delta),
            'concat_metrics': {k: float(v) if isinstance(v, (float, np.floating)) else v
                               for k, v in concat_metrics.items()},
            'fold_results': [{k: float(v) if isinstance(v, (float, np.floating)) else v
                              for k, v in fr.items()} for fr in fold_results],
            'date_sharpes': {d: float(s) for d, s in zip(unique_dates, date_sharpes)} if date_sharpes else {},
            'timestamp': datetime.now().isoformat(),
        }
        summary_path = os.path.join(CONFIG['output_dir'], 'training_summary.json')
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)

        logger.info(f"\nSaved summary and concat predictions")

        if MLFLOW_AVAILABLE:
            mlflow.log_artifact(summary_path)
            mlflow.log_artifact(log_path)

    if MLFLOW_AVAILABLE:
        mlflow.end_run()

    logger.info("\n" + "=" * 60)
    logger.info("Meta-Model v7 PRODUCTION (1s Horizon) — Training Complete")
    logger.info("=" * 60)
    sys.stdout.flush()


if __name__ == '__main__':
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    sys.stdout.flush()
