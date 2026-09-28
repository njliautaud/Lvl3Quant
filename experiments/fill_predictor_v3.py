#!/usr/bin/env python3
"""
Fill Predictor v3 — Passive Limit Order Fill Probability
=========================================================
Binary classifier: "If I place a passive limit at this moment, will it fill within 5 seconds?"

Key improvements over v2 (AUC 0.558):
  - Uses meta v7 predictions (Spearman 0.308) instead of v6
  - Additional engineered features: book pressure, trade tape velocity, queue proxy
  - MLP architecture with BatchNorm+GELU+Dropout (128->64->32)
  - Walk-forward: 10d train / 3d eval matching v7 production schedule

Fill label construction (from MBO price dynamics):
  - For SHORT entries: fill = best ask was hit within 5s (price rose to our limit)
    Proxy: label_5s >= +1.0 tick (price moved UP, someone lifted our offer)
  - For LONG entries: fill = best bid was hit within 5s (price fell to our limit)
    Proxy: label_5s <= -1.0 tick (price moved DOWN, someone hit our bid)
  - We also use label_1s for "fast fill" detection

  Note: labels are forward price changes in ticks. For a passive SHORT entry at the ask:
    - We need price to come UP to our ask level -> positive label = fill likely
  For a passive LONG entry at the bid:
    - We need price to come DOWN to our bid level -> negative label = fill likely

MLflow experiment: fill_predictor_v3
"""

import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import roc_auc_score, precision_recall_curve, average_precision_score

# === CONFIGURATION ===
EXPERIMENT_NAME = "fill_predictor_v3"
V7_PRED_DIR = Path("/home/nick/Lvl3Quant/output/meta_v7_prod")
CM_DIR = Path("/home/nick/Lvl3Quant/output/cnn_mamba_v2_bulk_oot")
MBO_DIR = Path("/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3")
OUT_DIR = Path("/home/nick/Lvl3Quant/output/fill_predictor_v3")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Walk-forward config (matches v7 prod)
TRAIN_DAYS = 10
EVAL_DAYS = 3

# Model config
HIDDEN_DIMS = [128, 64, 32]
DROPOUT = 0.2
BATCH_SIZE = 4096
EPOCHS = 30
LR = 1e-3
WEIGHT_DECAY = 1e-4
PATIENCE = 7

# Fill label config
FILL_THRESHOLD_TICKS = 1.0  # 1 tick = minimum for passive fill
FAST_FILL_THRESHOLD = 0.5   # Half-tick for "fast fill" at 1s
WINDOW_SIZE = 3000
STRIDE = 250

# Cost constants
TICK_VALUE = 12.50
COMMISSION_TICKS = 0.376

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# === LOGGING ===
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(OUT_DIR / 'training.log'),
        logging.StreamHandler()
    ]
)
log = logging.getLogger(__name__)

# === MLFLOW ===
try:
    import mlflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment(EXPERIMENT_NAME)
    USE_MLFLOW = True
    log.info("MLflow connected: http://localhost:5000")
except Exception as e:
    log.warning(f"MLflow unavailable: {e}")
    USE_MLFLOW = False


class FillPredictorMLP(nn.Module):
    """MLP with BatchNorm + GELU + Dropout for fill probability prediction."""
    def __init__(self, input_dim, hidden_dims=[128, 64, 32], dropout=0.2):
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
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(-1)


def engineer_extra_features(events, timestamps):
    """
    Engineer additional features from MBO microstructure data.

    The 25 base MBO columns contain normalized microstructure features.
    We engineer additional features to capture fill-relevant dynamics:

    1. Book pressure asymmetry (bid vs ask depth proxy)
    2. Trade tape velocity (recent trade intensity)
    3. Spread dynamics proxy
    4. Rolling fill rate proxy (how often price retraces)
    5. Volatility burst indicator
    """
    n = len(events)
    extra = np.zeros((n, 8), dtype=np.float32)

    # Col interpretations from stats analysis:
    # col2: direction indicator (-1/+1 range) - likely trade side
    # col3: magnitude indicator (-2/+2 range) - likely tick move
    # col4: time delta proxy (small positive values)
    # col5: book level indicator (0-4 range)
    # col6: price change (small, symmetric around 0)
    # col7: normalized feature (~N(0,1))
    # col9, col10: signed features (book imbalance proxies)
    # col11: signed feature (accumulated imbalance)
    # col19: another signed feature (-1 to +0.7)

    # Feature 1: Book pressure (bid-ask imbalance proxy)
    # Use cols 9 and 10 which look like bid/ask depth indicators
    extra[:, 0] = events[:, 9] - events[:, 10]  # book imbalance

    # Feature 2: Absolute imbalance (regardless of direction)
    extra[:, 1] = np.abs(events[:, 9] - events[:, 10])

    # Feature 3: Trade direction pressure (rolling)
    # col2 is direction indicator, col3 is magnitude
    extra[:, 2] = events[:, 2] * np.abs(events[:, 3])  # signed momentum

    # Feature 4: Volatility burst (large moves in col7)
    extra[:, 3] = events[:, 7] ** 2  # squared returns proxy

    # Feature 5: Queue depth proxy (col5 is book level, col12 looks like a rate)
    extra[:, 4] = events[:, 5] * events[:, 12]  # depth * rate interaction

    # Feature 6: Spread state (col6 price change * col4 time)
    extra[:, 5] = np.abs(events[:, 6]) * (1.0 + events[:, 4])

    # Feature 7: Accumulated pressure (col11 * col19 - two signed features)
    extra[:, 6] = events[:, 11] * events[:, 19]

    # Feature 8: Recent activity intensity (col8 looks like event count proxy)
    extra[:, 7] = events[:, 8] * (1.0 + np.abs(events[:, 7]))

    return extra


def load_day_data(date_str):
    """
    Load and align v7 predictions with MBO data for one date.

    Returns features and fill labels for all evaluation points.
    """
    # Load CNN-Mamba base predictions (has stride/window info)
    cm_file = CM_DIR / f"{date_str}_predictions.npz"
    if not cm_file.exists():
        return None

    cm = np.load(cm_file, allow_pickle=True)
    n_windows = int(cm['n_windows'])
    window_size = int(cm['window_size'])
    stride = int(cm['stride'])
    cm_preds = cm['predictions']  # (n_windows, 3) for 1s/5s/10s

    # Load MBO events
    mbo_file = MBO_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_file.exists():
        return None

    mbo = np.load(mbo_file, allow_pickle=True)
    events = mbo['events']       # (N, 25)
    l1s = mbo['labels_1s']       # (N,)
    l5s = mbo['labels_5s']       # (N,)
    l10s = mbo['labels_10s']     # (N,)
    timestamps = mbo['timestamps']  # (N,)

    # Compute MBO indices that correspond to predictions
    indices = np.arange(n_windows) * stride + (window_size - 1)
    max_idx = min(len(events), len(l1s), len(l5s)) - 1
    valid = indices <= max_idx
    indices = indices[valid]
    cm_preds = cm_preds[:len(indices)]

    # Extract features at evaluation points
    feat_mbo = events[indices]          # (n, 25)
    label_1s = l1s[indices]
    label_5s = l5s[indices]
    label_10s = l10s[indices]
    ts = timestamps[indices]

    # Validity mask
    valid_mask = ~(np.isnan(label_1s) | np.isnan(label_5s) |
                   np.isnan(label_10s) | np.any(np.isnan(feat_mbo), axis=1))

    feat_mbo = feat_mbo[valid_mask]
    label_1s = label_1s[valid_mask]
    label_5s = label_5s[valid_mask]
    label_10s = label_10s[valid_mask]
    cm_preds = cm_preds[valid_mask]
    ts = ts[valid_mask]

    if len(feat_mbo) == 0:
        return None

    # Now load v7 meta predictions for this date
    v7 = np.load(V7_PRED_DIR / "concat_oot_predictions.npz", allow_pickle=True)
    v7_dates = v7['dates']
    v7_preds = v7['predictions']
    v7_labels = v7['labels']

    date_mask = v7_dates == date_str
    v7_pred_day = v7_preds[date_mask]
    v7_label_day = v7_labels[date_mask]

    if len(v7_pred_day) == 0:
        # This date doesn't have v7 predictions (not in OOT set)
        # Use CNN-Mamba predictions directly (v7 is built on these)
        v7_pred_day = None

    # Align: v7 predictions should be same length as our valid events
    # v7 was trained on these exact evaluation points
    n_valid = len(feat_mbo)

    if v7_pred_day is not None and len(v7_pred_day) == n_valid:
        meta_score = v7_pred_day
    elif v7_pred_day is not None:
        # Length mismatch - v7 may have slightly different filtering
        # Use the shorter length
        min_len = min(len(v7_pred_day), n_valid)
        meta_score = v7_pred_day[:min_len]
        feat_mbo = feat_mbo[:min_len]
        label_1s = label_1s[:min_len]
        label_5s = label_5s[:min_len]
        label_10s = label_10s[:min_len]
        cm_preds = cm_preds[:min_len]
        ts = ts[:min_len]
        n_valid = min_len
    else:
        # No v7 preds for this date - use CM 1s pred as proxy
        meta_score = cm_preds[:, 0]

    # Engineer extra features
    extra_feats = engineer_extra_features(feat_mbo, ts)  # (n, 8)

    # Compute signal ranks (percentile of meta score)
    ranks = np.argsort(np.argsort(meta_score)).astype(np.float32) / len(meta_score)

    # Compute absolute meta score (confidence magnitude)
    abs_meta = np.abs(meta_score)

    # === CONSTRUCT FILL LABELS ===
    # For SHORT entries (negative meta_score = bearish prediction):
    #   Passive entry at ASK. Fill = someone lifts our offer = price rises to our level
    #   Proxy: label_5s > 0 means price went UP -> favorable for short fill
    #   More precisely: max(label_1s, label_5s) >= THRESHOLD -> price touched our level
    #
    # For LONG entries (positive meta_score = bullish prediction):
    #   Passive entry at BID. Fill = someone hits our bid = price drops to our level
    #   Proxy: label_5s < 0 means price went DOWN -> favorable for long fill
    #   More precisely: min(label_1s, label_5s) <= -THRESHOLD
    #
    # We create a UNIFIED fill label regardless of direction:
    #   fill = |max_favorable_move_within_5s| >= THRESHOLD

    # For each event, compute maximum favorable move for passive fill
    # Short side: favorable = price going UP (label positive)
    # Long side: favorable = price going DOWN (label negative)
    # Since we don't know direction at label time, create both:

    max_up_move = np.maximum(label_1s, label_5s)    # Best upward move within 5s
    max_down_move = np.minimum(label_1s, label_5s)  # Best downward move within 5s

    # For the actual signal direction:
    is_short_signal = meta_score < 0  # Negative prediction = short

    # Favorable move magnitude for each signal direction
    favorable_move = np.where(
        is_short_signal,
        max_up_move,     # Short: want price UP to fill our ask
        -max_down_move   # Long: want price DOWN to fill our bid (negate to get magnitude)
    )

    fill_label = (favorable_move >= FILL_THRESHOLD_TICKS).astype(np.float32)
    fast_fill = (np.where(
        is_short_signal,
        label_1s,        # Short: 1s upward move
        -label_1s        # Long: 1s downward move
    ) >= FAST_FILL_THRESHOLD).astype(np.float32)

    # === BUILD FEATURE MATRIX ===
    # 25 MBO + 3 CM predictions + 1 meta score + 1 abs meta + 1 rank + 8 engineered = 39 features
    features = np.column_stack([
        feat_mbo,                    # 25 MBO microstructure
        cm_preds,                    # 3 CNN-Mamba predictions (1s, 5s, 10s)
        meta_score.reshape(-1, 1),   # 1 v7 meta score
        abs_meta.reshape(-1, 1),     # 1 confidence magnitude
        ranks.reshape(-1, 1),        # 1 signal percentile rank
        extra_feats,                 # 8 engineered features
    ]).astype(np.float32)

    return {
        'date': date_str,
        'features': features,
        'fill_label': fill_label,
        'fast_fill': fast_fill,
        'meta_score': meta_score,
        'label_5s': label_5s,
        'label_1s': label_1s,
        'is_short': is_short_signal,
        'favorable_move': favorable_move,
        'n_events': n_valid,
    }


def get_all_dates():
    """Get sorted list of all dates with both CM predictions and MBO data."""
    cm_dates = set()
    for f in CM_DIR.glob("*_predictions.npz"):
        cm_dates.add(f.stem.split('_')[0])

    mbo_dates = set()
    for f in MBO_DIR.glob("*_mbo_events.npz"):
        mbo_dates.add(f.stem.split('_')[0])

    # Get v7 OOT dates
    v7 = np.load(V7_PRED_DIR / "concat_oot_predictions.npz", allow_pickle=True)
    v7_dates = set(np.unique(v7['dates']))

    # We need dates with CM + MBO data. v7 OOT dates are bonus (have meta score).
    common = sorted(cm_dates & mbo_dates)
    log.info(f"Total dates with CM+MBO: {len(common)}")
    log.info(f"V7 OOT dates: {len(v7_dates)}")
    log.info(f"Dates with all three: {len(set(common) & v7_dates)}")

    return common


def train_fold(train_data, eval_data, fold_idx):
    """Train one walk-forward fold."""
    # Concatenate training data
    train_X = np.concatenate([d['features'] for d in train_data])
    train_y = np.concatenate([d['fill_label'] for d in train_data])

    eval_X = np.concatenate([d['features'] for d in eval_data])
    eval_y = np.concatenate([d['fill_label'] for d in eval_data])
    eval_meta = np.concatenate([d['meta_score'] for d in eval_data])
    eval_l5s = np.concatenate([d['label_5s'] for d in eval_data])
    eval_short = np.concatenate([d['is_short'] for d in eval_data])
    eval_favorable = np.concatenate([d['favorable_move'] for d in eval_data])

    if len(train_X) < 500 or len(eval_X) < 100:
        log.warning(f"Fold {fold_idx}: insufficient data (train={len(train_X)}, eval={len(eval_X)})")
        return None

    # Normalize features
    mean = train_X.mean(axis=0)
    std = train_X.std(axis=0) + 1e-8
    train_X_n = (train_X - mean) / std
    eval_X_n = (eval_X - mean) / std

    # Class balance
    pos_rate = train_y.mean()
    if pos_rate <= 0.01 or pos_rate >= 0.99:
        log.warning(f"Fold {fold_idx}: extreme class imbalance (pos_rate={pos_rate:.4f})")
        return None
    pos_weight = torch.tensor([(1 - pos_rate) / pos_rate]).to(device)

    # DataLoader
    train_ds = TensorDataset(
        torch.from_numpy(train_X_n).float(),
        torch.from_numpy(train_y).float()
    )
    train_loader = DataLoader(
        train_ds, batch_size=BATCH_SIZE, shuffle=True,
        num_workers=4, pin_memory=True, drop_last=False
    )

    # Model
    input_dim = train_X_n.shape[1]
    model = FillPredictorMLP(input_dim, HIDDEN_DIMS, DROPOUT).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    # Training loop with early stopping
    best_loss = float('inf')
    patience_counter = 0
    best_state = None

    for epoch in range(EPOCHS):
        model.train()
        epoch_loss = 0
        n_batches = 0
        for X_b, y_b in train_loader:
            X_b, y_b = X_b.to(device), y_b.to(device)
            optimizer.zero_grad()
            logits = model(X_b)
            loss = criterion(logits, y_b)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()

        avg_loss = epoch_loss / max(n_batches, 1)

        # Validation loss
        model.eval()
        with torch.no_grad():
            eval_logits = model(torch.from_numpy(eval_X_n).float().to(device)).cpu().numpy()
            eval_loss = nn.functional.binary_cross_entropy_with_logits(
                torch.from_numpy(eval_logits), torch.from_numpy(eval_y),
                pos_weight=pos_weight.cpu()
            ).item()

        if eval_loss < best_loss:
            best_loss = eval_loss
            patience_counter = 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                log.info(f"  Early stop at epoch {epoch+1}")
                break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final evaluation
    model.eval()
    with torch.no_grad():
        eval_logits = model(torch.from_numpy(eval_X_n).float().to(device)).cpu().numpy()
        eval_probs = 1.0 / (1.0 + np.exp(-eval_logits))

    # === METRICS ===
    try:
        auc = roc_auc_score(eval_y, eval_probs)
    except ValueError:
        auc = 0.5

    try:
        ap = average_precision_score(eval_y, eval_probs)
    except ValueError:
        ap = 0.0

    # Calibration analysis at various thresholds
    thresholds = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
    calibration = {}
    for t in thresholds:
        mask = eval_probs >= t
        if mask.sum() > 0:
            actual_fill_rate = eval_y[mask].mean()
            n_selected = mask.sum()
            calibration[f"t{int(t*100)}"] = {
                'predicted_threshold': t,
                'actual_fill_rate': float(actual_fill_rate),
                'n_selected': int(n_selected),
                'pct_selected': float(mask.mean()),
            }

    # === P&L CONDITIONED ON FILL PROBABILITY ===
    # Key question: do high fill-prob events also have better P&L?
    # For shorts: P&L = -label_5s (price went down = profit)
    # For longs: P&L = +label_5s (price went up = profit)
    raw_pnl = np.where(eval_short, -eval_l5s, eval_l5s)

    # Compare P&L in high vs low fill-prob buckets
    median_prob = np.median(eval_probs)
    high_fill = eval_probs >= np.percentile(eval_probs, 75)
    low_fill = eval_probs < np.percentile(eval_probs, 25)

    pnl_high = raw_pnl[high_fill].mean() if high_fill.sum() > 0 else 0
    pnl_low = raw_pnl[low_fill].mean() if low_fill.sum() > 0 else 0

    # Also check: among TOP meta_score events, does fill_prob help?
    top_meta = np.abs(eval_meta) >= np.percentile(np.abs(eval_meta), 90)
    if top_meta.sum() > 10:
        top_high_fill = top_meta & (eval_probs >= np.median(eval_probs[top_meta]))
        top_low_fill = top_meta & (eval_probs < np.median(eval_probs[top_meta]))
        pnl_top_high = raw_pnl[top_high_fill].mean() if top_high_fill.sum() > 0 else 0
        pnl_top_low = raw_pnl[top_low_fill].mean() if top_low_fill.sum() > 0 else 0
    else:
        pnl_top_high = 0
        pnl_top_low = 0

    eval_dates = [d['date'] for d in eval_data]

    result = {
        'fold': fold_idx,
        'eval_dates': eval_dates,
        'n_train': len(train_X),
        'n_eval': len(eval_X),
        'pos_rate_train': float(pos_rate),
        'pos_rate_eval': float(eval_y.mean()),
        'auc': float(auc),
        'avg_precision': float(ap),
        'best_val_loss': float(best_loss),
        'calibration': calibration,
        'pnl_high_fill': float(pnl_high),
        'pnl_low_fill': float(pnl_low),
        'pnl_delta': float(pnl_high - pnl_low),
        'pnl_top_meta_high_fill': float(pnl_top_high),
        'pnl_top_meta_low_fill': float(pnl_top_low),
    }

    # Save fold predictions
    np.savez_compressed(
        OUT_DIR / f"fold_{fold_idx:02d}_predictions.npz",
        predictions=eval_probs,
        fill_labels=eval_y,
        meta_scores=eval_meta,
        label_5s=eval_l5s,
        is_short=eval_short,
        favorable_move=eval_favorable,
        dates=np.concatenate([np.full(d['n_events'], d['date']) for d in eval_data]),
        feat_mean=mean,
        feat_std=std,
    )

    # Save model
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_dim': input_dim,
        'hidden_dims': HIDDEN_DIMS,
        'dropout': DROPOUT,
        'feat_mean': mean,
        'feat_std': std,
    }, OUT_DIR / f"fold_{fold_idx:02d}_model.pt")

    return result


def main():
    log.info("=" * 60)
    log.info("Fill Predictor v3 — Starting Training")
    log.info(f"Device: {device}")
    log.info(f"Output: {OUT_DIR}")
    log.info("=" * 60)

    start_time = time.time()

    # Start MLflow run
    mlflow_run = None
    if USE_MLFLOW:
        mlflow_run = mlflow.start_run(run_name=f"fill_pred_v3_{datetime.now().strftime('%Y%m%d_%H%M')}")
        mlflow.log_params({
            'train_days': TRAIN_DAYS,
            'eval_days': EVAL_DAYS,
            'hidden_dims': str(HIDDEN_DIMS),
            'dropout': DROPOUT,
            'batch_size': BATCH_SIZE,
            'epochs': EPOCHS,
            'lr': LR,
            'weight_decay': WEIGHT_DECAY,
            'patience': PATIENCE,
            'fill_threshold_ticks': FILL_THRESHOLD_TICKS,
            'model_type': 'MLP',
            'device': str(device),
        })

    # Get all available dates
    all_dates = get_all_dates()

    if len(all_dates) < TRAIN_DAYS + EVAL_DAYS:
        log.error(f"Not enough dates: {len(all_dates)} < {TRAIN_DAYS + EVAL_DAYS}")
        return

    # Load all data (memory efficient: load per date)
    log.info("Loading data for all dates...")
    date_data = {}
    for date_str in all_dates:
        data = load_day_data(date_str)
        if data is not None:
            date_data[date_str] = data
            log.info(f"  {date_str}: {data['n_events']} events, fill_rate={data['fill_label'].mean():.3f}")

    valid_dates = sorted(date_data.keys())
    log.info(f"Loaded {len(valid_dates)} dates with valid data")

    if len(valid_dates) < TRAIN_DAYS + EVAL_DAYS:
        log.error(f"Not enough valid dates: {len(valid_dates)}")
        return

    # Walk-forward training
    fold_results = []
    all_probs = []
    all_labels = []
    all_dates_list = []

    fold_idx = 0
    i = TRAIN_DAYS
    while i + EVAL_DAYS <= len(valid_dates):
        train_dates = valid_dates[i - TRAIN_DAYS:i]
        eval_dates = valid_dates[i:i + EVAL_DAYS]

        train_data = [date_data[d] for d in train_dates]
        eval_data = [date_data[d] for d in eval_dates]

        log.info(f"\n{'='*40}")
        log.info(f"Fold {fold_idx}: train={train_dates[0]}..{train_dates[-1]}, eval={eval_dates[0]}..{eval_dates[-1]}")

        result = train_fold(train_data, eval_data, fold_idx)

        if result is not None:
            fold_results.append(result)
            log.info(f"  AUC={result['auc']:.4f}, AP={result['avg_precision']:.4f}")
            log.info(f"  Fill rate: train={result['pos_rate_train']:.3f}, eval={result['pos_rate_eval']:.3f}")
            log.info(f"  P&L high-fill={result['pnl_high_fill']:.3f}, low-fill={result['pnl_low_fill']:.3f}, delta={result['pnl_delta']:.3f}")

            if USE_MLFLOW:
                mlflow.log_metrics({
                    f'fold{fold_idx}_auc': result['auc'],
                    f'fold{fold_idx}_ap': result['avg_precision'],
                    f'fold{fold_idx}_pnl_delta': result['pnl_delta'],
                }, step=fold_idx)

            # Collect for concat metrics
            fold_pred = np.load(OUT_DIR / f"fold_{fold_idx:02d}_predictions.npz")
            all_probs.append(fold_pred['predictions'])
            all_labels.append(fold_pred['fill_labels'])
            all_dates_list.append(fold_pred['dates'])

        fold_idx += 1
        i += EVAL_DAYS  # Slide by eval window

    # === CONCAT METRICS ===
    if len(all_probs) > 0:
        concat_probs = np.concatenate(all_probs)
        concat_labels = np.concatenate(all_labels)
        concat_dates = np.concatenate(all_dates_list)

        try:
            concat_auc = roc_auc_score(concat_labels, concat_probs)
        except ValueError:
            concat_auc = 0.5

        try:
            concat_ap = average_precision_score(concat_labels, concat_probs)
        except ValueError:
            concat_ap = 0.0

        concat_fill_rate = concat_labels.mean()

        log.info(f"\n{'='*60}")
        log.info(f"CONCAT RESULTS ({len(concat_probs)} events, {len(fold_results)} folds)")
        log.info(f"  AUC: {concat_auc:.4f}")
        log.info(f"  Avg Precision: {concat_ap:.4f}")
        log.info(f"  Overall fill rate: {concat_fill_rate:.3f}")

        # Per-threshold analysis
        log.info(f"\n  Threshold | Selected% | Actual Fill Rate | Lift vs Base")
        for t in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            mask = concat_probs >= t
            if mask.sum() > 0:
                actual = concat_labels[mask].mean()
                lift = actual / concat_fill_rate if concat_fill_rate > 0 else 0
                log.info(f"  {t:.1f}       | {mask.mean()*100:5.1f}%    | {actual:.4f}           | {lift:.2f}x")

        # Save concat predictions
        np.savez_compressed(
            OUT_DIR / "concat_oot_predictions.npz",
            predictions=concat_probs,
            fill_labels=concat_labels,
            dates=concat_dates,
        )

        # Mean fold metrics
        mean_auc = np.mean([r['auc'] for r in fold_results])
        mean_ap = np.mean([r['avg_precision'] for r in fold_results])
        mean_pnl_delta = np.mean([r['pnl_delta'] for r in fold_results])

        log.info(f"\n  Mean fold AUC: {mean_auc:.4f}")
        log.info(f"  Mean fold AP: {mean_ap:.4f}")
        log.info(f"  Mean P&L delta (high-fill minus low-fill): {mean_pnl_delta:.4f} ticks")

        if USE_MLFLOW:
            mlflow.log_metrics({
                'concat_auc': concat_auc,
                'concat_ap': concat_ap,
                'concat_fill_rate': concat_fill_rate,
                'mean_fold_auc': mean_auc,
                'mean_fold_ap': mean_ap,
                'mean_pnl_delta': mean_pnl_delta,
                'n_folds': len(fold_results),
                'n_total_events': len(concat_probs),
            })

    # Save training summary
    elapsed = time.time() - start_time
    summary = {
        'experiment': EXPERIMENT_NAME,
        'version': 'v3',
        'description': 'Fill probability predictor using meta v7 predictions + engineered MBO features',
        'model_type': 'MLP',
        'hidden_dims': HIDDEN_DIMS,
        'n_folds': len(fold_results),
        'n_total_events': int(len(concat_probs)) if len(all_probs) > 0 else 0,
        'concat_auc': float(concat_auc) if len(all_probs) > 0 else None,
        'concat_ap': float(concat_ap) if len(all_probs) > 0 else None,
        'mean_fold_auc': float(mean_auc) if fold_results else None,
        'mean_pnl_delta': float(mean_pnl_delta) if fold_results else None,
        'fold_results': fold_results,
        'elapsed_seconds': elapsed,
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'v7_baseline': 'meta_v7_prod (Spearman 0.308)',
    }

    with open(OUT_DIR / 'training_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    if USE_MLFLOW:
        mlflow.log_artifact(str(OUT_DIR / 'training_summary.json'))
        mlflow.end_run()

    log.info(f"\nDone in {elapsed/60:.1f} minutes. Results saved to {OUT_DIR}")


if __name__ == '__main__':
    main()
