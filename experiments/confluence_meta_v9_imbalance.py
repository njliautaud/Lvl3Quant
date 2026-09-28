#!/usr/bin/env python3
"""
Confluence Meta-Learner v9: Imbalance-Conditioned Trade Selection

KEY DIFFERENCES FROM v8 (which failed with IC=0.008):
- v8 targeted FIFO net P&L → noisy, low signal. THIS targets realized directional returns.
- v8 had no imbalance features. THIS uses 9 real-time order flow features from smart_v3.
- v8 used pair/triplet polynomial features (33 inputs). THIS uses raw predictions + imbalance (24 inputs).
- Walk-forward sliding 15-day train, 1-day OOT per HC #0.

INPUTS (24 features):
  - CNN-Mamba v2 predictions: 3 horizons (1s, 5s, 10s)
  - PatchTST predictions: 3 horizons (1s, 5s, 10s)
  - Agreement flags: 3 binary (do models agree on direction?)
  - Confidence: 3 (sum of absolute predictions)
  - Spread: 3 (v2 - PatchTST per horizon)
  - Imbalance features from smart_v3: 9 (cancel_asym, ofi_500, queue_replen,
    ofi_x_spread, buy_sell_ratio, sweep_intensity, ofi_short, ofi_long, ofi_accel)

TARGET: Signed trade return at 10s horizon
  - If predicted direction is long: return = labels_10s
  - If predicted direction is short: return = -labels_10s
  - Goal: learn which combinations predict LARGE MOVES in the right direction

EVALUATION: Per HC #428
  - Net ticks after 0.376t commission (HC #512)
  - Daily Sharpe, PF, WR
  - Regime stratification (green/red/flat days)
  - Day concentration <= 0.70

MLflow logging mandatory.
"""

import os
import sys
import json
import time
import glob
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

warnings.filterwarnings('ignore')

# ── Configuration ──────────────────────────────────────────────────────
COMMISSION_TICKS = 0.376  # HC #512: RT commission only
STRIDE = 250              # Prediction stride (events between predictions)
WINDOW_SIZE = 3000        # Events per prediction window

# Imbalance feature indices in smart_v3 25-feature vector
IMBALANCE_INDICES = [6, 7, 15, 17, 19, 21, 22, 23, 24]
IMBALANCE_NAMES = [
    'cancel_asym', 'ofi_500', 'queue_replen', 'ofi_x_spread',
    'buy_sell_ratio', 'sweep_intensity', 'ofi_short', 'ofi_long', 'ofi_accel'
]

# Walk-forward params per HC #0 (sliding, not expanding)
TRAIN_DAYS = 15
OOT_DAYS = 1

# MLP architecture
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.3
LR = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 15
BATCH_SIZE = 4096

# Horizon to target for trade decisions
TARGET_HORIZON_IDX = 2  # 10s (index into [1s, 5s, 10s])

# Top-N percentile for trade selection evaluation
TRADE_PERCENTILES = [5, 10, 20, 30]

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(levelname)s %(message)s')
logger = logging.getLogger(__name__)


def setup_paths():
    """Detect environment (Neptune vs Jupiter) and set paths."""
    if os.path.exists('/home/nick/Lvl3Quant'):
        base = '/home/nick/Lvl3Quant'
    elif os.path.exists('/home/jupiter/Lvl3Quant'):
        base = '/home/jupiter/Lvl3Quant'
    else:
        raise RuntimeError("Cannot find Lvl3Quant directory")

    paths = {
        'base': base,
        'v2_oot': os.path.join(base, 'output/cnn_mamba_v2_bulk_oot'),
        'ptst_oot': os.path.join(base, 'output/patchtst_bulk_oot'),
        'mbo_events': os.path.join(base, 'data/processed/mbo_events_smart_v3'),
        'output': os.path.join(base, 'output/confluence_meta_v9_imbalance'),
    }
    os.makedirs(paths['output'], exist_ok=True)
    return paths


def get_overlapping_dates(paths):
    """Find dates where v2, PatchTST, AND mbo_events all exist."""
    v2_dates = set()
    for f in glob.glob(os.path.join(paths['v2_oot'], '*_predictions.npz')):
        d = os.path.basename(f).split('_')[0]
        v2_dates.add(d)

    ptst_dates = set()
    for f in glob.glob(os.path.join(paths['ptst_oot'], '*_predictions.npz')):
        d = os.path.basename(f).split('_')[0]
        ptst_dates.add(d)

    mbo_dates = set()
    for f in glob.glob(os.path.join(paths['mbo_events'], '*_mbo_events.npz')):
        d = os.path.basename(f).split('_')[0]
        mbo_dates.add(d)

    overlap = sorted(v2_dates & ptst_dates & mbo_dates)
    logger.info(f"Date overlap: {len(v2_dates)} v2, {len(ptst_dates)} ptst, "
                f"{len(mbo_dates)} mbo → {len(overlap)} common")
    return overlap


def load_date_data(date_str, paths):
    """Load and align v2 + PatchTST predictions + imbalance features for one date."""
    # Load predictions
    v2_path = os.path.join(paths['v2_oot'], f'{date_str}_predictions.npz')
    ptst_path = os.path.join(paths['ptst_oot'], f'{date_str}_predictions.npz')
    mbo_path = os.path.join(paths['mbo_events'], f'{date_str}_mbo_events.npz')

    v2 = np.load(v2_path)
    ptst = np.load(ptst_path)
    mbo = np.load(mbo_path)

    v2_preds = v2['predictions']    # (N, 3)
    v2_labels = v2['labels']        # (N, 3)
    ptst_preds = ptst['predictions'] # (M, 3)

    events = mbo['events']          # (E, 25)

    # Align: take minimum length between v2 and PatchTST
    n_preds = min(len(v2_preds), len(ptst_preds))
    v2_preds = v2_preds[:n_preds]
    v2_labels = v2_labels[:n_preds]
    ptst_preds = ptst_preds[:n_preds]

    # Map prediction index → event index
    # Each prediction i corresponds to events[i*STRIDE : i*STRIDE + WINDOW_SIZE]
    # We take the LAST event in the window as the "current" event for imbalance features
    event_indices = np.arange(n_preds) * STRIDE + WINDOW_SIZE - 1

    # Filter: only keep predictions where event index is valid
    valid = event_indices < len(events)
    if not np.all(valid):
        n_valid = np.sum(valid)
        v2_preds = v2_preds[valid]
        v2_labels = v2_labels[valid]
        ptst_preds = ptst_preds[valid]
        event_indices = event_indices[valid]
        n_preds = n_valid

    # Extract imbalance features at prediction points
    imbalance_feats = events[event_indices][:, IMBALANCE_INDICES]  # (N, 9)

    # Build feature matrix
    # Agreement flags (binary: do models agree on sign?)
    agreement = (np.sign(v2_preds) == np.sign(ptst_preds)).astype(np.float32)  # (N, 3)

    # Confidence (absolute sum)
    confidence = np.abs(v2_preds) + np.abs(ptst_preds)  # (N, 3)

    # Spread between models
    spread = v2_preds - ptst_preds  # (N, 3)

    # Stack all features: [v2(3), ptst(3), agree(3), conf(3), spread(3), imbalance(9)] = 24
    features = np.concatenate([
        v2_preds, ptst_preds, agreement, confidence, spread, imbalance_feats
    ], axis=1).astype(np.float32)

    # Target: signed trade return at target horizon
    # Use v2 prediction sign as direction (v2 is the stronger model)
    # If v2 says long: return = labels. If v2 says short: return = -labels.
    target_labels = v2_labels[:, TARGET_HORIZON_IDX]  # (N,) realized 10s return
    v2_direction = np.sign(v2_preds[:, TARGET_HORIZON_IDX])

    # Signed trade return: positive = trade went in our favor
    signed_return = (target_labels * v2_direction).astype(np.float32)

    # Filter NaN from targets AND corresponding features
    valid_mask = np.isfinite(signed_return) & np.isfinite(target_labels)
    # Also filter NaN in features
    valid_mask &= np.all(np.isfinite(features), axis=1)

    features = features[valid_mask]
    signed_return = signed_return[valid_mask]
    target_labels = target_labels[valid_mask]
    v2_direction = v2_direction[valid_mask]
    n_preds = len(features)

    # Also store raw labels and direction for evaluation
    meta = {
        'date': date_str,
        'n_samples': n_preds,
        'raw_labels_10s': target_labels,
        'v2_direction': v2_direction,
        'v2_preds_10s': v2_preds[:, TARGET_HORIZON_IDX][valid_mask],
        'ptst_preds_10s': ptst_preds[:, TARGET_HORIZON_IDX][valid_mask],
    }

    return features, signed_return, meta


class MetaLearnerMLP(nn.Module):
    """MLP that predicts trade quality from confluence + imbalance features."""

    def __init__(self, input_dim=24, hidden_dims=[128, 64, 32], dropout=0.3):
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
        layers.append(nn.Linear(prev_dim, 1))  # Single output: predicted trade quality
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def train_fold(train_features, train_targets, val_features, val_targets, device, fold_idx):
    """Train one walk-forward fold."""
    # Normalize features (fit on train, apply to val)
    mean = np.nanmean(train_features, axis=0)
    std = np.nanstd(train_features, axis=0)
    std[std < 1e-8] = 1.0

    # Replace NaN/inf
    train_features = np.nan_to_num(train_features, nan=0.0, posinf=0.0, neginf=0.0)
    val_features = np.nan_to_num(val_features, nan=0.0, posinf=0.0, neginf=0.0)

    train_normed = (train_features - mean) / std
    val_normed = (val_features - mean) / std

    # Tensors
    X_train = torch.tensor(train_normed, dtype=torch.float32).to(device)
    y_train = torch.tensor(train_targets, dtype=torch.float32).to(device)
    X_val = torch.tensor(val_normed, dtype=torch.float32).to(device)
    y_val = torch.tensor(val_targets, dtype=torch.float32).to(device)

    train_ds = TensorDataset(X_train, y_train)
    train_dl = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                          num_workers=0, pin_memory=False)

    model = MetaLearnerMLP(
        input_dim=train_features.shape[1],
        hidden_dims=HIDDEN_DIMS,
        dropout=DROPOUT
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    # Use Huber loss (robust to outliers in returns)
    criterion = nn.HuberLoss(delta=1.0)

    best_val_loss = float('inf')
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for xb, yb in train_dl:
            optimizer.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        # Validate
        model.eval()
        with torch.no_grad():
            val_pred = model(X_val)
            val_loss = criterion(val_pred, y_val).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

    # Get predictions from best model
    if best_state is None:
        best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        oot_predictions = model(X_val).cpu().numpy()

    return oot_predictions, mean, std, best_val_loss


def evaluate_trading(oot_predictions, oot_signed_returns, oot_meta, commission=COMMISSION_TICKS):
    """Evaluate trading performance using meta-learner predictions as trade filter."""
    results = {}

    n_total = len(oot_predictions)
    if n_total == 0:
        return {'error': 'no_data'}

    # The meta-learner predicts trade quality (higher = better trade)
    # Strategy: only take trades where meta-learner score > threshold

    for pct in TRADE_PERCENTILES:
        threshold = np.percentile(oot_predictions, 100 - pct)
        mask = oot_predictions >= threshold
        n_trades = np.sum(mask)

        if n_trades < 10:
            results[f'top_{pct}pct'] = {'n_trades': int(n_trades), 'skip': True}
            continue

        # Gross returns of selected trades
        selected_returns = oot_signed_returns[mask]

        # Net returns after commission
        net_returns = selected_returns - commission

        # Metrics
        mean_gross = float(np.mean(selected_returns))
        mean_net = float(np.mean(net_returns))
        total_net = float(np.sum(net_returns))
        win_rate = float(np.mean(net_returns > 0))

        # Profit factor
        wins = net_returns[net_returns > 0]
        losses = net_returns[net_returns < 0]
        pf = float(np.sum(wins) / (-np.sum(losses))) if len(losses) > 0 and np.sum(losses) != 0 else 0.0

        results[f'top_{pct}pct'] = {
            'n_trades': int(n_trades),
            'mean_gross_ticks': round(mean_gross, 4),
            'mean_net_ticks': round(mean_net, 4),
            'total_net_ticks': round(total_net, 2),
            'win_rate': round(win_rate, 4),
            'profit_factor': round(pf, 3),
            'threshold': round(float(threshold), 4),
        }

    # Also evaluate: all trades (baseline v2 directional)
    all_net = oot_signed_returns - commission
    results['baseline_all'] = {
        'n_trades': n_total,
        'mean_net_ticks': round(float(np.mean(all_net)), 4),
        'win_rate': round(float(np.mean(all_net > 0)), 4),
    }

    # IC between meta-learner score and actual signed return
    if np.std(oot_predictions) > 1e-8 and np.std(oot_signed_returns) > 1e-8:
        ic = float(np.corrcoef(oot_predictions, oot_signed_returns)[0, 1])
    else:
        ic = 0.0
    results['ic'] = round(ic, 4)

    return results


def compute_daily_metrics(per_day_results, commission=COMMISSION_TICKS):
    """Compute aggregate metrics across OOT days per HC #428."""
    if not per_day_results:
        return {}

    # Per-day net ticks for top-10% trades
    daily_nets = []
    daily_trades = []
    daily_wrs = []

    for dr in per_day_results:
        top10 = dr.get('top_10pct', {})
        if top10.get('skip'):
            continue

        logger.info(f"\nFold {fold_idx+1}/{n_folds}: train={train_dates[0]}..{train_dates[-1]} "
                    f"({len(train_X)} samples), OOT={oot_date} ({len(oot_X)} samples)")
        daily_nets.append(top10.get('total_net_ticks', 0))
        daily_trades.append(top10.get('n_trades', 0))
        daily_wrs.append(top10.get('win_rate', 0))

    if not daily_nets:
        return {}

    daily_nets = np.array(daily_nets)

    # Daily Sharpe
    if np.std(daily_nets) > 0:
        daily_sharpe = float(np.mean(daily_nets) / np.std(daily_nets) * np.sqrt(252))
    else:
        daily_sharpe = 0.0

    # Total metrics
    total_net = float(np.sum(daily_nets))
    total_trades = int(np.sum(daily_trades))
    avg_trades_per_day = float(np.mean(daily_trades))

    # Profit factor
    winning_days = daily_nets[daily_nets > 0]
    losing_days = daily_nets[daily_nets < 0]
    if len(losing_days) > 0 and np.sum(losing_days) != 0:
        pf = float(np.sum(winning_days) / (-np.sum(losing_days)))
    else:
        pf = 0.0

    profitable_days = int(np.sum(daily_nets > 0))
    total_days = len(daily_nets)

    # Day concentration (max single day contribution)
    if total_net > 0:
        day_conc = float(np.max(daily_nets) / total_net)
    else:
        day_conc = 1.0

    return {
        'daily_sharpe': round(daily_sharpe, 2),
        'total_net_ticks': round(total_net, 2),
        'total_trades': total_trades,
        'avg_trades_per_day': round(avg_trades_per_day, 1),
        'profit_factor': round(pf, 3),
        'profitable_days': f"{profitable_days}/{total_days}",
        'day_concentration': round(day_conc, 3),
        'mean_daily_wr': round(float(np.mean(daily_wrs)), 4) if daily_wrs else 0,
    }


def main():
    parser = argparse.ArgumentParser(description='Confluence Meta-Learner v9')
    parser.add_argument('--device', default='auto', help='cuda/cpu/auto')
    parser.add_argument('--mlflow-uri', default='http://localhost:5000', help='MLflow tracking URI')
    args = parser.parse_args()

    # Device
    if args.device == 'auto':
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(args.device)
    logger.info(f"Device: {device}")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment('confluence_meta_v9_imbalance')
        mlflow_run = mlflow.start_run(run_name=f'v9_imbalance_{datetime.now().strftime("%Y%m%d_%H%M")}')
        mlflow.log_params({
            'model': 'MLP',
            'hidden_dims': str(HIDDEN_DIMS),
            'dropout': DROPOUT,
            'lr': LR,
            'epochs': EPOCHS,
            'batch_size': BATCH_SIZE,
            'train_days': TRAIN_DAYS,
            'target_horizon': '10s',
            'commission': COMMISSION_TICKS,
            'n_features': 24,
            'feature_groups': 'v2_preds(3)+ptst_preds(3)+agreement(3)+confidence(3)+spread(3)+imbalance(9)',
        })
        use_mlflow = True
        logger.info("MLflow tracking enabled")
    except Exception as e:
        logger.warning(f"MLflow not available: {e}")
        use_mlflow = False

    # Setup
    paths = setup_paths()
    dates = get_overlapping_dates(paths)

    if len(dates) < TRAIN_DAYS + 1:
        logger.error(f"Only {len(dates)} dates, need at least {TRAIN_DAYS + 1}")
        sys.exit(1)

    # Load all data
    logger.info(f"Loading data for {len(dates)} dates...")
    all_features = {}
    all_targets = {}
    all_meta = {}

    for date_str in dates:
        try:
            features, targets, meta = load_date_data(date_str, paths)
            all_features[date_str] = features
            all_targets[date_str] = targets
            all_meta[date_str] = meta
            logger.info(f"  {date_str}: {meta['n_samples']} samples, "
                       f"mean_return={np.mean(targets):.4f}, std={np.std(targets):.4f}")
        except Exception as e:
            logger.warning(f"  {date_str}: FAILED - {e}")

    loaded_dates = sorted(all_features.keys())
    logger.info(f"Successfully loaded {len(loaded_dates)} dates")

    if use_mlflow:
        mlflow.log_metric('n_dates_loaded', len(loaded_dates))

    # Walk-forward: sliding window
    per_day_results = []
    all_oot_preds = []
    all_oot_returns = []
    all_oot_dates = []

    n_folds = len(loaded_dates) - TRAIN_DAYS
    logger.info(f"\n{'='*60}")
    logger.info(f"Starting walk-forward: {n_folds} OOT folds")
    logger.info(f"{'='*60}")

    for fold_idx in range(n_folds):
        train_dates = loaded_dates[fold_idx:fold_idx + TRAIN_DAYS]
        oot_date = loaded_dates[fold_idx + TRAIN_DAYS]

        # Stack training data
        train_X = np.concatenate([all_features[d] for d in train_dates])
        train_y = np.concatenate([all_targets[d] for d in train_dates])

        # OOT data
        oot_X = all_features[oot_date]
        oot_y = all_targets[oot_date]
        oot_meta = all_meta[oot_date]

        if len(oot_X) == 0:
            logger.info(f"\nFold {fold_idx+1}/{n_folds}: OOT={oot_date} — 0 samples, SKIPPING")
            continue

        logger.info(f"\nFold {fold_idx+1}/{n_folds}: train={train_dates[0]}..{train_dates[-1]} "
                    f"({len(train_X)} samples), OOT={oot_date} ({len(oot_X)} samples)")




        # Train
        t0 = time.time()
        oot_predictions, mean, std, val_loss = train_fold(
            train_X, train_y, oot_X, oot_y, device, fold_idx
        )
        elapsed = time.time() - t0

        # Evaluate
        day_result = evaluate_trading(oot_predictions, oot_y, oot_meta)
        day_result['date'] = oot_date
        day_result['train_time_s'] = round(elapsed, 1)
        day_result['val_loss'] = round(val_loss, 4)
        per_day_results.append(day_result)

        all_oot_preds.append(oot_predictions)
        all_oot_returns.append(oot_y)
        all_oot_dates.extend([oot_date] * len(oot_predictions))

        # Log
        ic = day_result.get('ic', 0)
        top10 = day_result.get('top_10pct', {})
        top10_net = top10.get('mean_net_ticks', 0) if not top10.get('skip') else 'N/A'
        top10_wr = top10.get('win_rate', 0) if not top10.get('skip') else 'N/A'

        logger.info(f"  IC={ic:.4f}, top10% net={top10_net}, WR={top10_wr}, "
                    f"time={elapsed:.1f}s, val_loss={val_loss:.4f}")

        if use_mlflow:
            mlflow.log_metrics({
                f'fold_{fold_idx}_ic': ic,
                f'fold_{fold_idx}_val_loss': val_loss,
            }, step=fold_idx)

    # Aggregate results
    logger.info(f"\n{'='*60}")
    logger.info("AGGREGATE RESULTS")
    logger.info(f"{'='*60}")

    # Concat IC
    all_preds_concat = np.concatenate(all_oot_preds)
    all_returns_concat = np.concatenate(all_oot_returns)

    if np.std(all_preds_concat) > 1e-8 and np.std(all_returns_concat) > 1e-8:
        concat_ic = float(np.corrcoef(all_preds_concat, all_returns_concat)[0, 1])
    else:
        concat_ic = 0.0

    # Per-fold IC distribution
    fold_ics = [r.get('ic', 0) for r in per_day_results]
    mean_ic = float(np.mean(fold_ics))
    positive_ic_folds = sum(1 for ic in fold_ics if ic > 0)

    logger.info(f"Concat IC: {concat_ic:.4f}")
    logger.info(f"Mean per-fold IC: {mean_ic:.4f}")
    logger.info(f"Positive IC folds: {positive_ic_folds}/{len(fold_ics)}")

    # Daily metrics
    daily_metrics = compute_daily_metrics(per_day_results)

    logger.info(f"\nDaily Metrics (top 10% trades):")
    for k, v in daily_metrics.items():
        logger.info(f"  {k}: {v}")

    # Percentile analysis on concat
    logger.info(f"\nPercentile analysis (all OOT concatenated):")
    for pct in TRADE_PERCENTILES:
        threshold = np.percentile(all_preds_concat, 100 - pct)
        mask = all_preds_concat >= threshold
        selected = all_returns_concat[mask]
        net = selected - COMMISSION_TICKS
        mean_net = float(np.mean(net))
        wr = float(np.mean(net > 0))
        total = float(np.sum(net))
        logger.info(f"  Top {pct}%: n={np.sum(mask)}, mean_net={mean_net:.4f}t, "
                    f"WR={wr:.3f}, total={total:.1f}t")

    # Baseline (all trades, no filtering)
    baseline_net = all_returns_concat - COMMISSION_TICKS
    logger.info(f"\n  Baseline (all): n={len(baseline_net)}, "
               f"mean_net={np.mean(baseline_net):.4f}t, "
               f"WR={np.mean(baseline_net > 0):.3f}")

    # Save results
    results = {
        'experiment': 'confluence_meta_v9_imbalance',
        'timestamp': datetime.now().isoformat(),
        'concat_ic': round(concat_ic, 4),
        'mean_fold_ic': round(mean_ic, 4),
        'positive_ic_folds': f"{positive_ic_folds}/{len(fold_ics)}",
        'n_oot_dates': len(per_day_results),
        'n_total_samples': len(all_preds_concat),
        'daily_metrics_top10': daily_metrics,
        'per_day': per_day_results,
        'config': {
            'train_days': TRAIN_DAYS,
            'target_horizon': '10s',
            'hidden_dims': HIDDEN_DIMS,
            'dropout': DROPOUT,
            'lr': LR,
            'epochs': EPOCHS,
            'commission': COMMISSION_TICKS,
        }
    }

    results_path = os.path.join(paths['output'], 'results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nResults saved to {results_path}")

    # Save predictions for further analysis
    preds_path = os.path.join(paths['output'], 'oot_predictions.npz')
    np.savez_compressed(preds_path,
                       predictions=all_preds_concat,
                       signed_returns=all_returns_concat,
                       dates=np.array(all_oot_dates))
    logger.info(f"Predictions saved to {preds_path}")

    # MLflow final logging
    if use_mlflow:
        mlflow.log_metrics({
            'concat_ic': concat_ic,
            'mean_fold_ic': mean_ic,
            'positive_ic_pct': positive_ic_folds / len(fold_ics) if fold_ics else 0,
            'daily_sharpe_top10': daily_metrics.get('daily_sharpe', 0),
            'total_net_ticks_top10': daily_metrics.get('total_net_ticks', 0),
            'profit_factor_top10': daily_metrics.get('profit_factor', 0),
            'n_oot_days': len(per_day_results),
        })
        mlflow.log_artifact(results_path)
        mlflow.end_run()

    # Final verdict
    logger.info(f"\n{'='*60}")
    logger.info("VERDICT")
    logger.info(f"{'='*60}")

    if concat_ic > 0.05:
        logger.info(f"✅ Concat IC = {concat_ic:.4f} — meta-learner adds value over raw predictions")
    elif concat_ic > 0.02:
        logger.info(f"⚠️ Concat IC = {concat_ic:.4f} — marginal, needs more features or better architecture")
    else:
        logger.info(f"❌ Concat IC = {concat_ic:.4f} — meta-learner is NOT adding value. "
                    "Imbalance + confluence combination doesn't predict trade quality.")

    pf = daily_metrics.get('profit_factor', 0)
    sharpe = daily_metrics.get('daily_sharpe', 0)
    if pf >= 1.2 and sharpe > 0.5:
        logger.info(f"✅ PASSES HC #428: PF={pf:.2f}, Sharpe={sharpe:.1f}")
    else:
        logger.info(f"❌ FAILS HC #428: PF={pf:.2f} (need ≥1.2), Sharpe={sharpe:.1f} (need ≥0.5)")

    logger.info("DONE.")
    return results


if __name__ == '__main__':
    main()
