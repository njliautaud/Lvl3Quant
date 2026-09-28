"""
Confluence Meta-Model v6 (Streamlined): CNN-Mamba + Microstructure ONLY

Per-head ablation showed PatchTST predictions, cross-model agreement features,
and time-of-day encoding are dead weight (removing them IMPROVES performance).
This v6 keeps ONLY CNN-Mamba predictions + microstructure features.

Features (31 total):
  - CNN-Mamba predictions (3): 1s, 5s, 10s horizons
  - CNN-Mamba confidence (3): abs(prediction) per horizon
  - Microstructure from MBO events (25): spread, depth imbalance, OFI, etc.

Dropped from v5 (14 features):
  - PatchTST predictions (3), PatchTST confidence (3)
  - Sign agreement (3), magnitude ratio (3)
  - Time-of-day sin/cos (2)

Walk-forward: SLIDING 20-day train / 5-day eval
Architecture: MLP 256->128->64, BatchNorm, GELU, Dropout 0.2
Target: 5s mid-price movement (regression, Huber loss)
v5 baseline Spearman: 0.192
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

# ============================================================
# Configuration
# ============================================================
CONFIG = {
    'output_dir': r'C:\Users\claude\Lvl3Quant\output\confluence_meta_v6',
    'cm_dir': r'C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_bulk_oot_v2',
    'mbo_dir': r'C:\Users\claude\Lvl3Quant\data\processed\mbo_events_smart_v3',
    # Walk-forward
    'train_days': 20,
    'eval_days': 5,
    # Model
    'hidden_dims': [256, 128, 64],
    'dropout': 0.2,
    'batch_size': 2048,
    'epochs': 20,
    'lr': 1e-3,
    'weight_decay': 1e-4,
    'patience': 5,
    # Cost constants
    'tick_value': 12.50,
    'commission_ticks': 0.376,
    # v5 baseline for comparison
    'v5_spearman': 0.192,
}

# ============================================================
# Logging — format: timestamp [META_V6] level: message
# ============================================================
os.makedirs(CONFIG['output_dir'], exist_ok=True)
log_path = os.path.join(CONFIG['output_dir'], 'training.log')

class MetaV6Formatter(logging.Formatter):
    def format(self, record):
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        return f"{ts} [META_V6] {record.levelname}: {record.msg}"

logger = logging.getLogger('meta_v6')
logger.setLevel(logging.INFO)
logger.propagate = False

# File handler
fh = logging.FileHandler(log_path)
fh.setFormatter(MetaV6Formatter())
logger.addHandler(fh)

# Stream handler (unbuffered via flush)
class FlushStreamHandler(logging.StreamHandler):
    def emit(self, record):
        super().emit(record)
        self.flush()

sh = FlushStreamHandler(sys.stdout)
sh.setFormatter(MetaV6Formatter())
logger.addHandler(sh)

# ============================================================
# Model Definition
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
# Data Loading & Alignment (v6: CNN-Mamba + MBO only)
# ============================================================
def get_overlapping_dates():
    """Find dates where CNN-Mamba AND MBO both have data.
    v6: No PatchTST dependency — only need CNN-Mamba + MBO overlap.
    """
    cm_dates = set(f[:8] for f in os.listdir(CONFIG['cm_dir'])
                   if f.endswith('_predictions.npz') and f[0] == '2')
    mbo_dates = set(f[:8] for f in os.listdir(CONFIG['mbo_dir'])
                    if f.endswith('_mbo_events.npz'))
    overlap = sorted(cm_dates & mbo_dates)
    logger.info(f"CNN-Mamba dates: {len(cm_dates)}, MBO: {len(mbo_dates)}, "
                f"overlap: {len(overlap)}")
    return overlap


def load_date_features(date_str):
    """
    Load and align CNN-Mamba predictions + MBO microstructure for a single date.
    Returns (features, target) arrays aligned at CNN-Mamba resolution.

    v6 STREAMLINED: No PatchTST, no agreement features, no time-of-day.
    Features = CNN-Mamba preds (3) + CNN-Mamba confidence (3) + MBO micro (25) = 31

    CNN-Mamba: stride=250 events, window=1000 -> event_idx = 1000 + i*250
    """
    # Load CNN-Mamba predictions
    cm_path = os.path.join(CONFIG['cm_dir'], f'{date_str}_predictions.npz')
    cm_data = np.load(cm_path, allow_pickle=True)
    cm_preds = cm_data['predictions']  # (N_cm, 3) for 1s/5s/10s
    cm_n = len(cm_preds)
    cm_window = int(cm_data['window_size'])
    cm_stride = int(cm_data['stride'])

    # CNN-Mamba event indices (center of last window position)
    cm_event_idx = np.array([cm_window + i * cm_stride for i in range(cm_n)])

    # Load MBO events for microstructure features
    mbo_path = os.path.join(CONFIG['mbo_dir'], f'{date_str}_mbo_events.npz')
    try:
        mbo_data = np.load(mbo_path)
    except (zipfile.BadZipFile, Exception) as e:
        logger.warning(f"  {date_str}: Bad MBO file ({e}), skip")
        return None, None
    mbo_events = mbo_data['events']        # (N_mbo, 25)
    mbo_labels_5s = mbo_data['labels_5s']  # target: 5s mid-price move
    n_mbo = len(mbo_events)

    # Clamp CNN-Mamba indices to valid MBO range
    valid_cm = cm_event_idx < n_mbo
    cm_event_idx = cm_event_idx[valid_cm]
    cm_preds = cm_preds[valid_cm]
    cm_n = len(cm_preds)

    if cm_n < 10:
        logger.warning(f"  {date_str}: Only {cm_n} valid CNN-Mamba rows, skipping")
        return None, None

    # Extract microstructure features at CNN-Mamba event positions
    mbo_at_cm = mbo_events[cm_event_idx]  # (cm_n, 25)

    # Target: 5s label at CNN-Mamba event positions
    target = mbo_labels_5s[cm_event_idx]  # (cm_n,)

    # CNN-Mamba confidence = absolute prediction magnitude
    cm_confidence = np.abs(cm_preds)  # (cm_n, 3)

    # ---- Build feature matrix (31 features) ----
    features = np.column_stack([
        cm_preds,       # 3: CNN-Mamba 1s/5s/10s predictions
        cm_confidence,  # 3: CNN-Mamba confidence (abs pred)
        mbo_at_cm,      # 25: microstructure features
    ]).astype(np.float32)  # Total: 3+3+25 = 31

    return features, target.astype(np.float32)


def load_all_dates(dates):
    """Load features and targets for multiple dates, filtering bad dates."""
    all_features = []
    all_targets = []
    all_date_labels = []
    valid_dates = []

    for date_str in dates:
        features, target = load_date_features(date_str)
        if features is None:
            continue

        # Filter: skip dates with zero-variance targets
        target_std = np.nanstd(target)
        if target_std < 1e-6:
            logger.warning(f"  {date_str}: ZERO-VARIANCE target (std={target_std:.8f}), SKIPPING")
            continue

        # Filter NaN targets
        valid_mask = ~np.isnan(target)
        nan_pct = 1.0 - valid_mask.mean()
        if nan_pct > 0.5:
            logger.warning(f"  {date_str}: {nan_pct*100:.1f}% NaN targets, SKIPPING")
            continue

        features = features[valid_mask]
        target = target[valid_mask]

        # Replace any NaN/inf in features with 0
        features = np.nan_to_num(features, nan=0.0, posinf=5.0, neginf=-5.0)

        all_features.append(features)
        all_targets.append(target)
        all_date_labels.extend([date_str] * len(features))
        valid_dates.append(date_str)
        logger.info(f"  {date_str}: {len(features)} samples, target std={target_std:.4f}, "
                     f"NaN%={nan_pct*100:.1f}%")

    if not all_features:
        return None, None, [], []

    return (np.concatenate(all_features),
            np.concatenate(all_targets),
            all_date_labels,
            valid_dates)


# ============================================================
# Training
# ============================================================
def train_fold(model, train_X, train_y, val_X, val_y, device, fold_idx):
    """Train one fold with early stopping."""
    train_ds = TensorDataset(
        torch.tensor(train_X, dtype=torch.float32),
        torch.tensor(train_y, dtype=torch.float32)
    )
    val_ds = TensorDataset(
        torch.tensor(val_X, dtype=torch.float32),
        torch.tensor(val_y, dtype=torch.float32)
    )
    train_loader = DataLoader(train_ds, batch_size=CONFIG['batch_size'],
                               shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=CONFIG['batch_size'] * 2,
                             shuffle=False, num_workers=0, pin_memory=True)

    criterion = nn.HuberLoss(delta=1.0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=CONFIG['lr'],
                                   weight_decay=CONFIG['weight_decay'])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=CONFIG['epochs'])

    best_val_loss = float('inf')
    best_state = None
    patience_counter = 0

    for epoch in range(CONFIG['epochs']):
        # Train
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

        # Validate
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

        # Compute Spearman correlation on validation
        val_preds_arr = np.concatenate(val_preds_list)
        val_labels_arr = np.concatenate(val_labels_list)
        spearman_r, _ = stats.spearmanr(val_preds_arr, val_labels_arr)

        if epoch % 5 == 0 or epoch == CONFIG['epochs'] - 1:
            logger.info(f"  Fold {fold_idx} Epoch {epoch}: "
                        f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, "
                        f"spearman={spearman_r:.4f}, lr={scheduler.get_last_lr()[0]:.6f}")
            sys.stdout.flush()

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= CONFIG['patience']:
                logger.info(f"  Fold {fold_idx}: Early stopping at epoch {epoch}")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return best_val_loss


def evaluate_fold(model, eval_X, eval_y, device):
    """Evaluate model on OOT data. Returns predictions."""
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
    """Compute evaluation metrics for a fold."""
    spearman_r, spearman_p = stats.spearmanr(predictions, labels)
    pearson_r, pearson_p = stats.pearsonr(predictions, labels)

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
    logger.info("Confluence Meta-Model v6 STREAMLINED — Training Start")
    logger.info("=" * 60)
    logger.info("v6 drops PatchTST, agreement, time-of-day (dead weight per ablation)")
    logger.info("Features: CNN-Mamba preds (3) + confidence (3) + MBO micro (25) = 31")
    logger.info(f"v5 baseline Spearman: {CONFIG['v5_spearman']}")
    logger.info(f"Config: {json.dumps(CONFIG, indent=2, default=str)}")
    sys.stdout.flush()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Device: {device}")
    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
    else:
        logger.warning("No CUDA device found — training will be slow on CPU")
    sys.stdout.flush()

    # Get overlapping dates (v6: only need CNN-Mamba + MBO)
    dates = get_overlapping_dates()
    if len(dates) < CONFIG['train_days'] + CONFIG['eval_days']:
        logger.error(f"Not enough dates ({len(dates)}) for walk-forward "
                     f"({CONFIG['train_days']} train + {CONFIG['eval_days']} eval)")
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
        # Zero-variance filter
        if np.nanstd(target) < 1e-6:
            logger.warning(f"  {d}: ZERO-VARIANCE target, REJECTED")
            continue
        # NaN filter
        valid_mask = ~np.isnan(target)
        if valid_mask.mean() < 0.5:
            logger.warning(f"  {d}: Too many NaN targets ({(1-valid_mask.mean())*100:.1f}%), REJECTED")
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
        logger.warning(f"Expected 31 features but got {input_dim} — check feature construction")
    sys.stdout.flush()

    # Walk-forward: SLIDING window
    train_days = CONFIG['train_days']
    eval_days = CONFIG['eval_days']
    n_folds = (len(valid_dates) - train_days) // eval_days

    logger.info(f"Walk-forward: {n_folds} folds, {train_days}d train / {eval_days}d eval")
    sys.stdout.flush()

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
            train_X_norm[-len(train_X)//5:], train_y[-len(train_y)//5:],  # last 20% as val
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

        # Save normalization stats separately
        norm_path = os.path.join(CONFIG['output_dir'], f'fold_{fold_idx:02d}_norm_stats.npz')
        np.savez_compressed(norm_path,
                            feat_mean=feat_mean,
                            feat_std=feat_std)

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

        logger.info(f"  Fold {fold_idx} OOT results:")
        logger.info(f"    Spearman: {metrics[f'fold{fold_idx}_spearman']:.4f}")
        logger.info(f"    Pearson:  {metrics[f'fold{fold_idx}_pearson']:.4f}")
        logger.info(f"    Samples:  {metrics[f'fold{fold_idx}_n_samples']}")
        for pct in ['top5', 'top10', 'top20', 'top50']:
            key = f'fold{fold_idx}_{pct}_net_ticks'
            if key in metrics:
                wr_key = f'fold{fold_idx}_{pct}_wr'
                n_key = f'fold{fold_idx}_{pct}_n_trades'
                logger.info(f"    {pct}: net={metrics[key]:.2f} ticks, "
                            f"WR={metrics.get(wr_key, 0):.3f}, "
                            f"n={metrics.get(n_key, 0)}")
        sys.stdout.flush()

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
                logger.info(f"{pct}: net={concat_metrics[key]:.2f} ticks, "
                            f"avg={concat_metrics.get(f'concat_{pct}_avg_ticks', 0):.4f}, "
                            f"WR={concat_metrics.get(f'concat_{pct}_wr', 0):.3f}, "
                            f"precision={concat_metrics.get(f'concat_{pct}_precision', 0):.3f}, "
                            f"n={concat_metrics.get(f'concat_{pct}_n_trades', 0)}")

        # ---- v5 vs v6 comparison ----
        v5_spearman = CONFIG['v5_spearman']
        v6_spearman = concat_metrics['concat_spearman']
        delta = v6_spearman - v5_spearman
        pct_change = (delta / abs(v5_spearman)) * 100 if v5_spearman != 0 else 0.0

        logger.info(f"\n{'='*60}")
        logger.info("v5 vs v6 COMPARISON")
        logger.info(f"{'='*60}")
        logger.info(f"v5 Spearman (baseline): {v5_spearman:.4f} (45 features)")
        logger.info(f"v6 Spearman (streamlined): {v6_spearman:.4f} (31 features)")
        logger.info(f"Delta: {delta:+.4f} ({pct_change:+.1f}%)")
        if delta >= 0:
            logger.info("RESULT: v6 MATCHES OR BEATS v5 with 31% fewer features")
        else:
            logger.info(f"RESULT: v6 underperforms v5 by {abs(delta):.4f}")
        logger.info(f"Features removed: PatchTST(6) + agreement(6) + time-of-day(2) = 14 dead weight")

        # Per-date breakdown
        logger.info(f"\nPer-date OOT breakdown:")
        unique_dates = sorted(set(all_oot_dates))
        date_sharpes = []
        for d in unique_dates:
            mask = concat_dates == d
            d_preds = concat_preds[mask]
            d_labels = concat_labels[mask]
            d_metrics = compute_metrics(d_preds, d_labels)
            d_spearman = d_metrics.get('spearman', 0)

            # Compute daily Sharpe on top-20% signals
            threshold = np.percentile(np.abs(d_preds), 80)
            d_mask = np.abs(d_preds) >= threshold
            if d_mask.sum() > 5:
                d_pnl = np.sign(d_preds[d_mask]) * d_labels[d_mask] - CONFIG['commission_ticks']
                d_sharpe = d_pnl.mean() / (d_pnl.std() + 1e-8) * np.sqrt(252)
                date_sharpes.append(d_sharpe)
            else:
                d_sharpe = 0.0

            logger.info(f"  {d}: n={mask.sum()}, spearman={d_spearman:.4f}, "
                        f"daily_sharpe={d_sharpe:.2f}")

        if date_sharpes:
            avg_sharpe = np.mean(date_sharpes)
            logger.info(f"\nAvg daily Sharpe (top20%): {avg_sharpe:.2f}")
            logger.info(f"Sharpe std: {np.std(date_sharpes):.2f}")
            logger.info(f"Positive Sharpe days: {sum(1 for s in date_sharpes if s > 0)}/{len(date_sharpes)}")

        sys.stdout.flush()

        # Save concat predictions
        concat_path = os.path.join(CONFIG['output_dir'], 'concat_oot_predictions.npz')
        np.savez_compressed(concat_path,
                            predictions=concat_preds,
                            labels=concat_labels,
                            dates=concat_dates)

        # Save summary
        summary = {
            'version': 'v6_streamlined',
            'description': 'CNN-Mamba + microstructure only (dropped PatchTST, agreement, time-of-day)',
            'input_dim': input_dim,
            'features_kept': ['cm_pred_1s', 'cm_pred_5s', 'cm_pred_10s',
                              'cm_conf_1s', 'cm_conf_5s', 'cm_conf_10s',
                              'mbo_micro_0..24'],
            'features_dropped': ['pt_pred_1s', 'pt_pred_5s', 'pt_pred_10s',
                                 'pt_conf_1s', 'pt_conf_5s', 'pt_conf_10s',
                                 'sign_match_1s', 'sign_match_5s', 'sign_match_10s',
                                 'mag_ratio_1s', 'mag_ratio_5s', 'mag_ratio_10s',
                                 'tod_sin', 'tod_cos'],
            'config': {k: str(v) if not isinstance(v, (int, float, str, list)) else v
                       for k, v in CONFIG.items()},
            'n_folds': len(fold_results),
            'n_valid_dates': len(valid_dates),
            'n_oot_dates': len(unique_dates),
            'v5_baseline_spearman': v5_spearman,
            'v6_spearman': float(v6_spearman),
            'v6_vs_v5_delta': float(delta),
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

        logger.info(f"\nSaved summary to {summary_path}")
        logger.info(f"Saved concat predictions to {concat_path}")

    logger.info("\n" + "=" * 60)
    logger.info("Confluence Meta-Model v6 STREAMLINED — Training Complete")
    logger.info("=" * 60)
    sys.stdout.flush()


if __name__ == '__main__':
    t0 = time.time()
    main()
    elapsed = time.time() - t0
    logger.info(f"Total runtime: {elapsed:.1f}s ({elapsed/60:.1f}m)")
    sys.stdout.flush()
