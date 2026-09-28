#!/usr/bin/env python3
"""
train_queue_position_predictor_v1.py — Queue Fill Probability Predictor
========================================================================
Predicts P(passive limit order fills within N seconds) using MBO event features.

Model: 1D CNN over a window of recent MBO events (25 smart_v3 features each).
Target: Binary — did price touch/cross our passive level within {1s, 5s, 10s}?
  - For BID fill: price must drop to bid level → labels_Ns <= -spread/2
  - For ASK fill: price must rise to ask level → labels_Ns >= spread/2
  Combined target: max(P(bid_fill), P(ask_fill)) or separate heads.

Walk-forward: 60-day train, 1-day OOT, sliding window (HC #0 — NEVER expanding).
Saves .pt weights + .npz predictions per fold, logs to MLflow.

Data: C:\\Users\\claude\\Lvl3Quant\\data\\processed\\mbo_events_smart_v3\\

smart_v3 25 features per event:
  RAW (0-5):   time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks
  DERIVED (6-14): cancel_side_asym, rolling_ofi, event_density, price_mom, qty_price_mom,
                   price_sign_mom, event_type_entropy, fill_add_restoration, spread_velocity
  V2 (15-21): queue_replenishment, mom_divergence, ofi_x_spread, vol_weighted_pmom,
              buy_sell_intensity_ratio, realized_volatility, sweep_intensity
  V3 (22-24): ofi_short, ofi_long, ofi_acceleration

labels_{1s,5s,10s,30s}: mid-price change in ticks at each horizon.

Author: Claude (Lvl3 Quant)
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ============================================================================
# Paths — cross-platform
# ============================================================================
if sys.platform == 'win32':
    _BASE = Path(r'C:\Users\claude\Lvl3Quant')
else:
    _BASE = Path('/home/jupiter/Lvl3Quant')

MBO_DIR = _BASE / 'data' / 'processed' / 'mbo_events_smart_v3'
OUT_DIR = _BASE / 'output' / 'queue_position_predictor_v1'
WEIGHTS_DIR = OUT_DIR / 'weights'
PRED_DIR = OUT_DIR / 'predictions'

OUT_DIR.mkdir(parents=True, exist_ok=True)
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
PRED_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# Config
# ============================================================================
N_FEATURES = 25                  # smart_v3 feature count
WINDOW_SIZE = 256                # events lookback window for 1D CNN
STRIDE = 64                     # stride for extracting windows (memory efficient)
TRAIN_WINDOW_DAYS = 15           # Reduced from 60: only 33 dates + 16GB RAM on Razer. Still sliding (HC #0).
BATCH_SIZE = 512                 # fits 8GB with window=256
LR = 5e-4
EPOCHS = 15
WEIGHT_DECAY = 1e-4
NUM_WORKERS = 4                  # Razer limited RAM
PIN_MEMORY = True
SUBSAMPLE_RATE = 8               # subsample events to reduce dataset size (8x for 16GB RAM)

# Fill label thresholds: price must move >= this many ticks toward our side
# ES book is 1 tick wide during RTH; spread_ticks col=5 gives actual spread
# If spread=1, price at mid needs to move 0.5 ticks to cross bid/ask.
# We use 0.25 ticks as threshold (half a tick) — conservative, means price
# reached the passive limit level.
FILL_THRESHOLD_TICKS = 0.25

# Horizons for multi-task prediction
HORIZONS = ['1s', '5s', '10s']

# MLflow config
MLFLOW_URI = 'http://neptune-win:5000'
EXPERIMENT_NAME = 'queue_position_predictor_v1'

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# ============================================================================
# Feature names (for reference / logging)
# ============================================================================
FEATURE_NAMES = [
    'time_delta_log', 'event_type_id', 'side_id',
    'price_rel_ticks', 'qty_log', 'spread_ticks',
    'cancel_side_asym_50', 'rolling_ofi_500', 'event_density_20',
    'price_mom_10', 'qty_price_mom_50', 'price_sign_momentum_200',
    'event_type_entropy_200', 'fill_add_restoration_100', 'spread_velocity_50',
    'queue_replenishment', 'mom_divergence', 'ofi_x_spread',
    'vol_weighted_pmom', 'buy_sell_intensity_ratio', 'realized_volatility',
    'sweep_intensity',
    'ofi_short_100', 'ofi_long_2000', 'ofi_acceleration',
]


# ============================================================================
# 1D CNN Model
# ============================================================================
class QueueFillCNN(nn.Module):
    """
    1D CNN for queue fill prediction.

    Input:  (batch, N_FEATURES, WINDOW_SIZE) — sequence of MBO events
    Output: (batch, 3) — P(fill) at 1s, 5s, 10s horizons (bid side)
            (batch, 3) — P(fill) at 1s, 5s, 10s horizons (ask side)

    Architecture: 3 conv layers with batch norm, global avg pool, separate
    FC heads for bid/ask fill probability.
    """

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 64):
        super().__init__()

        # Conv backbone
        self.conv1 = nn.Conv1d(n_features, hidden, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden)

        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden)

        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden)

        # Residual projection (input channels -> hidden)
        self.res_proj = nn.Conv1d(n_features, hidden, kernel_size=1)

        # Global average pool -> FC heads
        # Bid fill head
        self.bid_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, len(HORIZONS)),
        )

        # Ask fill head
        self.ask_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, len(HORIZONS)),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        x: (B, n_features, seq_len)
        Returns: bid_logits (B, 3), ask_logits (B, 3)
        """
        # Conv block 1
        h = F.relu(self.bn1(self.conv1(x)))

        # Conv block 2 with residual from block 1
        h2 = F.relu(self.bn2(self.conv2(h)))
        h = h + h2  # residual

        # Conv block 3 with residual
        h3 = F.relu(self.bn3(self.conv3(h)))
        # Add skip from input
        h = h3 + self.res_proj(x)

        # Global average pool: (B, hidden, seq) -> (B, hidden)
        h = h.mean(dim=2)

        bid_logits = self.bid_head(h)
        ask_logits = self.ask_head(h)

        return bid_logits, ask_logits


# ============================================================================
# Dataset
# ============================================================================
class QueueFillDataset(Dataset):
    """
    Builds windows from MBO event data and creates fill labels.

    Fill label logic:
      - Bid fill: price drops by >= FILL_THRESHOLD_TICKS within horizon
        → labels_Ns <= -FILL_THRESHOLD_TICKS → label=1
      - Ask fill: price rises by >= FILL_THRESHOLD_TICKS within horizon
        → labels_Ns >= FILL_THRESHOLD_TICKS → label=1

    Each sample: (features_window, bid_fill_labels, ask_fill_labels)
      features_window: (N_FEATURES, WINDOW_SIZE) float32
      bid_fill_labels: (3,) float32 — for 1s, 5s, 10s
      ask_fill_labels: (3,) float32 — for 1s, 5s, 10s
    """

    def __init__(self, events_list: List[np.ndarray],
                 labels_dict: Dict[str, List[np.ndarray]],
                 window_size: int = WINDOW_SIZE,
                 stride: int = STRIDE,
                 subsample: int = SUBSAMPLE_RATE):
        """
        events_list: list of (N_events, 25) arrays, one per day
        labels_dict: {'1s': [arr_day0, ...], '5s': [...], '10s': [...]}
        """
        self.window_size = window_size

        # Build index: (day_idx, event_idx) for each valid window
        self.index = []
        self.events = events_list
        self.labels = labels_dict

        for day_idx, events in enumerate(events_list):
            n_events = len(events)
            if n_events < window_size:
                continue

            # Get valid positions (all horizons must have non-NaN labels)
            valid = np.ones(n_events, dtype=bool)
            for hz in HORIZONS:
                lbl = labels_dict[hz][day_idx]
                valid &= ~np.isnan(lbl)

            # Also check features for NaN
            valid &= ~np.any(np.isnan(events), axis=1)

            # Only use positions where we have a full window behind us
            valid[:window_size] = False

            # Get valid indices with stride and subsample
            valid_indices = np.where(valid)[0]
            valid_indices = valid_indices[::stride * subsample]

            for idx in valid_indices:
                self.index.append((day_idx, idx))

    def __len__(self):
        return len(self.index)

    def __getitem__(self, i):
        day_idx, event_idx = self.index[i]

        # Extract window: events[event_idx - window_size : event_idx]
        window = self.events[day_idx][event_idx - self.window_size:event_idx]
        # Transpose to (features, time) for Conv1d
        x = window.T.astype(np.float32)  # (25, WINDOW_SIZE)

        # Build fill labels for the event at event_idx
        bid_labels = np.zeros(len(HORIZONS), dtype=np.float32)
        ask_labels = np.zeros(len(HORIZONS), dtype=np.float32)

        for hi, hz in enumerate(HORIZONS):
            price_change = self.labels[hz][day_idx][event_idx]
            # Bid fill: price dropped (went toward bid)
            bid_labels[hi] = 1.0 if price_change <= -FILL_THRESHOLD_TICKS else 0.0
            # Ask fill: price rose (went toward ask)
            ask_labels[hi] = 1.0 if price_change >= FILL_THRESHOLD_TICKS else 0.0

        return torch.from_numpy(x), torch.from_numpy(bid_labels), torch.from_numpy(ask_labels)


# ============================================================================
# Data loading
# ============================================================================
def get_available_dates() -> List[str]:
    """Get sorted list of available dates from MBO directory."""
    dates = []
    for f in sorted(MBO_DIR.glob('*_mbo_events.npz')):
        date_str = f.stem.split('_')[0]
        dates.append(date_str)
    return dates


def extract_windows_from_day(date_str: str, window_size: int = WINDOW_SIZE,
                              stride: int = STRIDE, subsample: int = SUBSAMPLE_RATE,
                              normalizer: 'FeatureNormalizer' = None,
                              max_windows: int = 5000) -> Optional[Dict]:
    """Load one day, extract windows immediately, free raw data.

    Returns dict with pre-extracted window tensors instead of raw events.
    This is memory-efficient: only ~max_windows × 256 × 25 × 4 bytes kept.
    """
    fpath = MBO_DIR / f'{date_str}_mbo_events.npz'
    if not fpath.exists():
        return None

    try:
        # MEMORY-CRITICAL: These npz files are 1-2 GB compressed.
        # events array is ~10M rows × 25 cols × 8 bytes = ~2GB float64.
        # NEVER copy full array (no .astype on full array, no arithmetic on full array).
        # Process ONLY indexed rows to stay under 16GB RAM.

        d = np.load(fpath, allow_pickle=True)
        events = d['events']  # Keep as-is (float64), DO NOT copy/convert

        if events.ndim != 2 or events.shape[1] != N_FEATURES:
            print(f"  WARN: {date_str} shape mismatch {events.shape}", flush=True)
            del d, events
            gc.collect()
            return None

        n_events = len(events)

        # For normalization-only mode, sample tiny subset
        if max_windows == 0:
            sample_n = min(20000, n_events)
            idx = np.linspace(0, n_events - 1, sample_n, dtype=int)
            norm_sample = events[idx].astype(np.float32)  # Only 20K rows → ~2 MB
            del d, events
            gc.collect()
            return {
                'date': date_str,
                'windows': np.empty((0, N_FEATURES, window_size), dtype=np.float32),
                'bid_labels': np.empty((0, len(HORIZONS)), dtype=np.float32),
                'ask_labels': np.empty((0, len(HORIZONS)), dtype=np.float32),
                'norm_sample': norm_sample,
                'n_windows': 0,
            }

        # Load labels (much smaller than events: ~80 MB each)
        labels = {}
        for hz in HORIZONS:
            key = f'labels_{hz}'
            if key in d:
                labels[hz] = d[key]  # Keep native dtype
            else:
                del d, events
                gc.collect()
                return None

        # Quick validity check using labels only (don't touch events yet)
        valid_1s = labels['1s']
        valid_count = np.sum(~np.isnan(valid_1s))
        if valid_count < window_size * 2:
            print(f"  SKIP {date_str}: only {valid_count} valid labels", flush=True)
            del d, events, labels
            gc.collect()
            return None

        # Find valid positions using labels + sparse event NaN check
        # Check NaN only at subsampled positions to avoid full-array scan
        valid = np.ones(n_events, dtype=bool)
        for hz in HORIZONS:
            valid &= ~np.isnan(labels[hz])
        valid[:window_size] = False

        valid_indices = np.where(valid)[0]
        valid_indices = valid_indices[::stride * subsample]

        # Limit windows
        if len(valid_indices) > max_windows:
            rng = np.random.RandomState(42)
            valid_indices = rng.choice(valid_indices, max_windows, replace=False)
            valid_indices.sort()

        if len(valid_indices) == 0:
            del d, events, labels
            gc.collect()
            return None

        # Extract ONLY the needed rows from events — each window is 256 rows
        # Instead of loading full array, extract windows one by one
        # Peak memory: max_windows × 256 × 25 × 4 bytes ≈ 77 MB for 3000 windows
        windows_list = []
        good_indices = []
        norm_mean = normalizer.mean if (normalizer is not None and normalizer.mean is not None) else None
        norm_std = normalizer.std if (normalizer is not None and normalizer.std is not None) else None

        for idx in valid_indices:
            w = events[idx - window_size:idx].astype(np.float32)  # 256×25×4 = 25 KB
            if np.any(np.isnan(w)):
                continue
            if norm_mean is not None:
                w = (w - norm_mean) / norm_std
            windows_list.append(w.T)  # (25, 256)
            good_indices.append(idx)

        del d, events  # FREE the big array ASAP
        gc.collect()

        if len(windows_list) == 0:
            del labels
            gc.collect()
            return None

        windows = np.stack(windows_list, dtype=np.float32)  # (N, 25, 256)
        del windows_list
        gc.collect()

        good_indices = np.array(good_indices)

        # Extract labels at good positions
        bid_lbl = np.zeros((len(good_indices), len(HORIZONS)), dtype=np.float32)
        ask_lbl = np.zeros((len(good_indices), len(HORIZONS)), dtype=np.float32)

        for hi, hz in enumerate(HORIZONS):
            price_changes = labels[hz][good_indices].astype(np.float32)
            bid_lbl[:, hi] = (price_changes <= -FILL_THRESHOLD_TICKS).astype(np.float32)
            ask_lbl[:, hi] = (price_changes >= FILL_THRESHOLD_TICKS).astype(np.float32)

        del labels
        gc.collect()

        return {
            'date': date_str,
            'windows': windows,
            'bid_labels': bid_lbl,
            'ask_labels': ask_lbl,
            'norm_sample': None,
            'n_windows': len(good_indices),
        }

    except Exception as e:
        print(f"  ERROR loading {date_str}: {e}", flush=True)
        import traceback
        traceback.print_exc()
        return None


class PreExtractedDataset(Dataset):
    """Dataset from pre-extracted windows (memory-efficient)."""

    def __init__(self, windows: np.ndarray, bid_labels: np.ndarray, ask_labels: np.ndarray):
        self.windows = torch.from_numpy(windows)
        self.bid_labels = torch.from_numpy(bid_labels)
        self.ask_labels = torch.from_numpy(ask_labels)

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        return self.windows[i], self.bid_labels[i], self.ask_labels[i]


def load_day_data(date_str: str) -> Optional[Dict]:
    """Legacy loader — kept for compatibility. Prefer extract_windows_from_day."""
    fpath = MBO_DIR / f'{date_str}_mbo_events.npz'
    if not fpath.exists():
        return None

    try:
        d = np.load(fpath, allow_pickle=True)
        events = d['events'].astype(np.float32)

        if events.shape[1] != N_FEATURES:
            return None

        result = {'date': date_str, 'events': events, 'labels': {}}
        for hz in HORIZONS:
            key = f'labels_{hz}'
            if key in d:
                result['labels'][hz] = d[key].astype(np.float32)
            else:
                return None

        valid_count = np.sum(~np.isnan(result['labels']['1s']))
        if valid_count < WINDOW_SIZE * 2:
            return None

        return result

    except Exception as e:
        print(f"  ERROR loading {date_str}: {e}")
        return None


# ============================================================================
# Normalization
# ============================================================================
class FeatureNormalizer:
    """Per-feature z-score normalization computed on training data."""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, events_list: List[np.ndarray]):
        """Compute mean/std from list of event arrays."""
        # Sample to keep memory bounded
        samples = []
        for events in events_list:
            n = len(events)
            if n > 50000:
                idx = np.random.choice(n, 50000, replace=False)
                samples.append(events[idx])
            else:
                samples.append(events)

        all_data = np.concatenate(samples, axis=0)
        self.mean = np.nanmean(all_data, axis=0).astype(np.float32)
        self.std = np.nanstd(all_data, axis=0).astype(np.float32)
        # Prevent division by zero
        self.std[self.std < 1e-8] = 1.0

    def transform(self, events: np.ndarray) -> np.ndarray:
        """Apply z-score normalization."""
        return ((events - self.mean) / self.std).astype(np.float32)

    def transform_list(self, events_list: List[np.ndarray]) -> List[np.ndarray]:
        """Apply to list of arrays."""
        return [self.transform(e) for e in events_list]


# ============================================================================
# Training
# ============================================================================
def train_one_epoch(model, loader, optimizer, scheduler=None):
    """Train for one epoch, returns average loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for x, bid_y, ask_y in loader:
        x = x.to(device)
        bid_y = bid_y.to(device)
        ask_y = ask_y.to(device)

        bid_logits, ask_logits = model(x)

        # Binary cross-entropy for each horizon
        bid_loss = F.binary_cross_entropy_with_logits(bid_logits, bid_y)
        ask_loss = F.binary_cross_entropy_with_logits(ask_logits, ask_y)
        loss = bid_loss + ask_loss

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if scheduler is not None:
        scheduler.step()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader):
    """Evaluate model, returns metrics dict."""
    model.eval()
    total_loss = 0.0
    n_batches = 0

    all_bid_probs = []
    all_ask_probs = []
    all_bid_labels = []
    all_ask_labels = []

    for x, bid_y, ask_y in loader:
        x = x.to(device)
        bid_y = bid_y.to(device)
        ask_y = ask_y.to(device)

        bid_logits, ask_logits = model(x)

        bid_loss = F.binary_cross_entropy_with_logits(bid_logits, bid_y)
        ask_loss = F.binary_cross_entropy_with_logits(ask_logits, ask_y)
        loss = bid_loss + ask_loss
        total_loss += loss.item()
        n_batches += 1

        all_bid_probs.append(torch.sigmoid(bid_logits).cpu().numpy())
        all_ask_probs.append(torch.sigmoid(ask_logits).cpu().numpy())
        all_bid_labels.append(bid_y.cpu().numpy())
        all_ask_labels.append(ask_y.cpu().numpy())

    avg_loss = total_loss / max(n_batches, 1)

    bid_probs = np.concatenate(all_bid_probs, axis=0)
    ask_probs = np.concatenate(all_ask_probs, axis=0)
    bid_labels = np.concatenate(all_bid_labels, axis=0)
    ask_labels = np.concatenate(all_ask_labels, axis=0)

    metrics = {'loss': avg_loss}

    # Per-horizon accuracy and AUC
    for hi, hz in enumerate(HORIZONS):
        for side, probs, labels in [('bid', bid_probs[:, hi], bid_labels[:, hi]),
                                     ('ask', ask_probs[:, hi], ask_labels[:, hi])]:
            preds_binary = (probs > 0.5).astype(float)
            acc = np.mean(preds_binary == labels)
            fill_rate = np.mean(labels)

            # AUC (manual — avoid sklearn dependency)
            auc = compute_auc(labels, probs)

            # Brier score
            brier = np.mean((probs - labels) ** 2)

            # Calibration: mean predicted vs actual fill rate
            cal_pred = np.mean(probs)
            cal_actual = fill_rate

            metrics[f'{side}_{hz}_acc'] = acc
            metrics[f'{side}_{hz}_auc'] = auc
            metrics[f'{side}_{hz}_brier'] = brier
            metrics[f'{side}_{hz}_fill_rate'] = fill_rate
            metrics[f'{side}_{hz}_cal_pred'] = cal_pred

    return metrics, bid_probs, ask_probs, bid_labels, ask_labels


def compute_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Compute AUC-ROC without sklearn. Returns 0.5 if degenerate."""
    pos = labels == 1
    neg = labels == 0
    n_pos = pos.sum()
    n_neg = neg.sum()

    if n_pos == 0 or n_neg == 0:
        return 0.5

    # Wilcoxon-Mann-Whitney statistic
    pos_scores = scores[pos]
    neg_scores = scores[neg]

    # Sample if too large (for speed)
    if n_pos > 10000:
        idx = np.random.choice(n_pos, 10000, replace=False)
        pos_scores = pos_scores[idx]
        n_pos = 10000
    if n_neg > 10000:
        idx = np.random.choice(n_neg, 10000, replace=False)
        neg_scores = neg_scores[idx]
        n_neg = 10000

    # Count concordant pairs
    concordant = 0
    for ps in pos_scores:
        concordant += np.sum(ps > neg_scores) + 0.5 * np.sum(ps == neg_scores)

    return concordant / (n_pos * n_neg)


# ============================================================================
# MLflow logging
# ============================================================================
def init_mlflow():
    """Initialize MLflow tracking. Returns (mlflow, run) or (None, None)."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        run = mlflow.start_run(
            run_name=f'queue_fill_cnn_v1_{datetime.now().strftime("%Y%m%d_%H%M%S")}'
        )
        mlflow.log_params({
            'model': 'QueueFillCNN',
            'n_features': N_FEATURES,
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'subsample_rate': SUBSAMPLE_RATE,
            'train_window_days': TRAIN_WINDOW_DAYS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'epochs': EPOCHS,
            'weight_decay': WEIGHT_DECAY,
            'fill_threshold_ticks': FILL_THRESHOLD_TICKS,
            'horizons': ','.join(HORIZONS),
            'device': str(device),
        })
        print(f"MLflow run started: {run.info.run_id}")
        return mlflow, run
    except Exception as e:
        print(f"MLflow init failed: {e} — continuing without tracking")
        return None, None


def log_fold_metrics(mlflow_mod, fold_idx: int, metrics: dict):
    """Log fold metrics to MLflow."""
    if mlflow_mod is None:
        return
    try:
        for k, v in metrics.items():
            if isinstance(v, (int, float)) and not np.isnan(v):
                mlflow_mod.log_metric(f'fold_{fold_idx:03d}/{k}', v, step=fold_idx)
    except Exception as e:
        print(f"  MLflow log error: {e}")


# ============================================================================
# Main walk-forward training
# ============================================================================
def main():
    print(f"{'='*70}")
    print(f"QUEUE FILL PROBABILITY PREDICTOR v1")
    print(f"{'='*70}")
    print(f"Device:          {device}")
    print(f"MBO dir:         {MBO_DIR}")
    print(f"Output dir:      {OUT_DIR}")
    print(f"Window size:     {WINDOW_SIZE} events")
    print(f"Stride:          {STRIDE}")
    print(f"Train window:    {TRAIN_WINDOW_DAYS} days (sliding)")
    print(f"Fill threshold:  {FILL_THRESHOLD_TICKS} ticks")
    print(f"Horizons:        {HORIZONS}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Epochs:          {EPOCHS}")
    print(f"LR:              {LR}")
    print(f"{'='*70}", flush=True)

    # Get available dates
    all_dates = get_available_dates()
    print(f"\nFound {len(all_dates)} dates in MBO directory")

    if len(all_dates) < TRAIN_WINDOW_DAYS + 1:
        print(f"ERROR: Need at least {TRAIN_WINDOW_DAYS + 1} dates, have {len(all_dates)}")
        return

    # Pre-load all days (lazy — just filenames, load on demand per fold)
    # Actually, load days in advance for the current fold window
    print(f"\nStarting walk-forward training...")
    print(f"Total folds: {len(all_dates) - TRAIN_WINDOW_DAYS}")

    # Initialize MLflow
    mlflow_mod, mlflow_run = init_mlflow()

    # Collect OOT predictions for concat evaluation
    concat_bid_probs = []
    concat_ask_probs = []
    concat_bid_labels = []
    concat_ask_labels = []
    concat_dates = []
    fold_metrics_all = []

    normalizer = FeatureNormalizer()

    n_folds = len(all_dates) - TRAIN_WINDOW_DAYS
    t_start = time.time()

    for fold_idx in range(n_folds):
        fold_t0 = time.time()
        train_dates = all_dates[fold_idx:fold_idx + TRAIN_WINDOW_DAYS]
        oot_date = all_dates[fold_idx + TRAIN_WINDOW_DAYS]

        print(f"\n--- Fold {fold_idx:03d}/{n_folds-1:03d}: "
              f"train={train_dates[0]}..{train_dates[-1]}, OOT={oot_date} ---",
              flush=True)

        # PHASE 1: Fit normalizer on sampled training data (one day at a time)
        print(f"  Fitting normalizer on training data...", flush=True)
        norm_samples = []
        for d in train_dates:
            result = extract_windows_from_day(d, max_windows=0)  # Just get norm_sample
            if result is not None and result.get('norm_sample') is not None:
                norm_samples.append(result['norm_sample'])
            del result
            gc.collect()

        if len(norm_samples) < 5:
            # Try simpler: load first 20K rows of a few files for normalization
            for d in train_dates[:5]:
                fpath = MBO_DIR / f'{d}_mbo_events.npz'
                if fpath.exists():
                    dd = np.load(fpath, allow_pickle=True)
                    ev = dd['events'][:20000].astype(np.float32)
                    norm_samples.append(ev)
                    del dd
                    gc.collect()

        if norm_samples:
            all_norm = np.concatenate(norm_samples, axis=0)
            normalizer.mean = np.nanmean(all_norm, axis=0).astype(np.float32)
            normalizer.std = np.nanstd(all_norm, axis=0).astype(np.float32)
            normalizer.std[normalizer.std < 1e-8] = 1.0
            del all_norm, norm_samples
            gc.collect()
        else:
            print(f"  SKIP fold: no valid training data for normalizer")
            continue

        # PHASE 2: Extract windows from each training day (one at a time, memory-efficient)
        print(f"  Extracting training windows day-by-day...", flush=True)
        all_train_windows = []
        all_train_bid = []
        all_train_ask = []
        loaded_train = 0

        for d in train_dates:
            result = extract_windows_from_day(d, normalizer=normalizer,
                                               max_windows=3000)
            if result is None:
                continue
            all_train_windows.append(result['windows'])
            all_train_bid.append(result['bid_labels'])
            all_train_ask.append(result['ask_labels'])
            loaded_train += 1
            print(f"    {d}: {result['n_windows']} windows", flush=True)
            del result
            gc.collect()

        if loaded_train < 5:
            print(f"  SKIP fold: only {loaded_train} valid train days")
            del all_train_windows, all_train_bid, all_train_ask
            gc.collect()
            continue

        # PHASE 3: Extract OOT windows
        oot_result = extract_windows_from_day(oot_date, normalizer=normalizer,
                                               max_windows=50000, subsample=1)
        if oot_result is None:
            print(f"  SKIP fold: OOT day {oot_date} not loadable")
            del all_train_windows, all_train_bid, all_train_ask
            gc.collect()
            continue

        # Combine training windows
        train_windows = np.concatenate(all_train_windows, axis=0)
        train_bid = np.concatenate(all_train_bid, axis=0)
        train_ask = np.concatenate(all_train_ask, axis=0)
        del all_train_windows, all_train_bid, all_train_ask
        gc.collect()

        # Build datasets using PreExtractedDataset (no raw events in memory)
        train_ds = PreExtractedDataset(train_windows, train_bid, train_ask)
        oot_ds = PreExtractedDataset(oot_result['windows'], oot_result['bid_labels'],
                                      oot_result['ask_labels'])
        del train_windows, train_bid, train_ask
        gc.collect()

        if len(train_ds) < 100:
            print(f"  SKIP fold: only {len(train_ds)} train samples")
            del train_ds, oot_ds, oot_result
            gc.collect()
            continue

        print(f"  Train samples: {len(train_ds):,}, OOT samples: {len(oot_ds):,}", flush=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=0, pin_memory=False,
                                   drop_last=True)
        oot_loader = DataLoader(oot_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=False)

        # Initialize model fresh each fold
        model = QueueFillCNN(n_features=N_FEATURES, hidden=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # Train
        best_val_loss = float('inf')
        best_state = None

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)

            if (epoch + 1) % 5 == 0 or epoch == EPOCHS - 1:
                oot_metrics, _, _, _, _ = evaluate(model, oot_loader)
                print(f"    Epoch {epoch+1:2d}/{EPOCHS}: "
                      f"train_loss={train_loss:.4f}, oot_loss={oot_metrics['loss']:.4f}, "
                      f"bid_5s_auc={oot_metrics.get('bid_5s_auc', 0):.3f}, "
                      f"ask_5s_auc={oot_metrics.get('ask_5s_auc', 0):.3f}",
                      flush=True)

                if oot_metrics['loss'] < best_val_loss:
                    best_val_loss = oot_metrics['loss']
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Restore best model
        if best_state is not None:
            model.load_state_dict(best_state)

        # Final OOT evaluation
        oot_metrics, bid_probs, ask_probs, bid_labels, ask_labels = evaluate(model, oot_loader)

        fold_time = time.time() - fold_t0
        print(f"  Fold {fold_idx:03d} complete ({fold_time:.0f}s):")
        for hz in HORIZONS:
            bid_auc = oot_metrics.get(f'bid_{hz}_auc', 0)
            ask_auc = oot_metrics.get(f'ask_{hz}_auc', 0)
            bid_fr = oot_metrics.get(f'bid_{hz}_fill_rate', 0)
            ask_fr = oot_metrics.get(f'ask_{hz}_fill_rate', 0)
            print(f"    {hz}: bid_AUC={bid_auc:.3f} (FR={bid_fr:.3f}), "
                  f"ask_AUC={ask_auc:.3f} (FR={ask_fr:.3f})")

        # Save fold weights
        weight_path = WEIGHTS_DIR / f'fold_{fold_idx:03d}_{oot_date}.pt'
        torch.save({
            'model_state_dict': model.state_dict(),
            'normalizer_mean': normalizer.mean,
            'normalizer_std': normalizer.std,
            'fold_idx': fold_idx,
            'oot_date': oot_date,
            'train_dates': train_dates,
            'metrics': oot_metrics,
        }, weight_path)

        # Save fold predictions
        pred_path = PRED_DIR / f'fold_{fold_idx:03d}_{oot_date}.npz'
        np.savez_compressed(pred_path,
                            bid_probs=bid_probs,
                            ask_probs=ask_probs,
                            bid_labels=bid_labels,
                            ask_labels=ask_labels,
                            date=oot_date,
                            fold_idx=fold_idx)

        # Accumulate for concat metrics
        concat_bid_probs.append(bid_probs)
        concat_ask_probs.append(ask_probs)
        concat_bid_labels.append(bid_labels)
        concat_ask_labels.append(ask_labels)
        concat_dates.append(oot_date)
        fold_metrics_all.append({**oot_metrics, 'fold_idx': fold_idx, 'oot_date': oot_date})

        # Log to MLflow
        log_fold_metrics(mlflow_mod, fold_idx, oot_metrics)

        # Cleanup
        del model, optimizer, scheduler, train_ds, oot_ds
        del train_loader, oot_loader, oot_result, best_state
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ========================================================================
    # Concat evaluation
    # ========================================================================
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD COMPLETE — {len(fold_metrics_all)} folds in {total_time:.0f}s")
    print(f"{'='*70}")

    if len(concat_bid_probs) == 0:
        print("No valid folds — exiting")
        if mlflow_mod:
            mlflow_mod.end_run()
        return

    all_bid_probs = np.concatenate(concat_bid_probs, axis=0)
    all_ask_probs = np.concatenate(concat_ask_probs, axis=0)
    all_bid_labels = np.concatenate(concat_bid_labels, axis=0)
    all_ask_labels = np.concatenate(concat_ask_labels, axis=0)

    print(f"\nCONCAT OOT RESULTS ({len(all_bid_probs):,} total samples, "
          f"{len(concat_dates)} OOT dates):")
    print(f"  Dates: {concat_dates[0]} to {concat_dates[-1]}")

    concat_metrics = {}
    for hi, hz in enumerate(HORIZONS):
        for side, probs, labels in [('bid', all_bid_probs[:, hi], all_bid_labels[:, hi]),
                                     ('ask', all_ask_probs[:, hi], all_ask_labels[:, hi])]:
            auc = compute_auc(labels, probs)
            brier = np.mean((probs - labels) ** 2)
            fill_rate = np.mean(labels)
            cal_pred = np.mean(probs)
            acc = np.mean((probs > 0.5).astype(float) == labels)

            concat_metrics[f'concat_{side}_{hz}_auc'] = auc
            concat_metrics[f'concat_{side}_{hz}_brier'] = brier
            concat_metrics[f'concat_{side}_{hz}_fill_rate'] = fill_rate
            concat_metrics[f'concat_{side}_{hz}_acc'] = acc

            print(f"  {side}_{hz}: AUC={auc:.4f}, Brier={brier:.4f}, "
                  f"Acc={acc:.4f}, FillRate={fill_rate:.3f}, PredMean={cal_pred:.3f}")

    # Calibration analysis: decile reliability
    print(f"\n  CALIBRATION (bid_5s fill probability):")
    probs_5s = all_bid_probs[:, 1]
    labels_5s = all_bid_labels[:, 1]
    deciles = np.percentile(probs_5s, np.arange(10, 101, 10))
    bins = np.digitize(probs_5s, deciles)
    for b in range(len(deciles)):
        mask = bins == b
        if mask.sum() > 0:
            pred_mean = probs_5s[mask].mean()
            actual_mean = labels_5s[mask].mean()
            print(f"    Decile {b}: pred={pred_mean:.3f}, actual={actual_mean:.3f}, "
                  f"n={mask.sum()}")

    # Save concat predictions
    concat_path = OUT_DIR / 'concat_oot_predictions.npz'
    np.savez_compressed(concat_path,
                        bid_probs=all_bid_probs,
                        ask_probs=all_ask_probs,
                        bid_labels=all_bid_labels,
                        ask_labels=all_ask_labels,
                        dates=np.array(concat_dates),
                        horizons=np.array(HORIZONS))
    print(f"\nConcat predictions saved to {concat_path}")

    # Save fold summary
    summary = {
        'n_folds': len(fold_metrics_all),
        'total_time_s': total_time,
        'concat_metrics': {k: float(v) for k, v in concat_metrics.items()},
        'per_fold': fold_metrics_all,
        'config': {
            'window_size': WINDOW_SIZE,
            'stride': STRIDE,
            'train_window_days': TRAIN_WINDOW_DAYS,
            'fill_threshold_ticks': FILL_THRESHOLD_TICKS,
            'horizons': HORIZONS,
            'batch_size': BATCH_SIZE,
            'lr': LR,
            'epochs': EPOCHS,
        }
    }
    summary_path = OUT_DIR / 'training_summary.json'
    with open(summary_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved to {summary_path}")

    # Log concat metrics to MLflow
    if mlflow_mod:
        try:
            for k, v in concat_metrics.items():
                mlflow_mod.log_metric(k, v)
            mlflow_mod.log_metric('n_folds', len(fold_metrics_all))
            mlflow_mod.log_metric('total_time_s', total_time)
            mlflow_mod.log_artifact(str(summary_path))
            mlflow_mod.end_run()
            print("MLflow run completed successfully")
        except Exception as e:
            print(f"MLflow finalization error: {e}")
            try:
                mlflow_mod.end_run()
            except:
                pass

    # Print per-fold AUC summary
    print(f"\n--- Per-fold AUC (bid_5s) ---")
    for fm in fold_metrics_all:
        auc = fm.get('bid_5s_auc', 0)
        date = fm.get('oot_date', '?')
        print(f"  {date}: {auc:.4f}")

    mean_bid_5s = np.mean([fm.get('bid_5s_auc', 0.5) for fm in fold_metrics_all])
    mean_ask_5s = np.mean([fm.get('ask_5s_auc', 0.5) for fm in fold_metrics_all])
    print(f"\n  Mean bid_5s AUC: {mean_bid_5s:.4f}")
    print(f"  Mean ask_5s AUC: {mean_ask_5s:.4f}")
    print(f"\nDONE.")


if __name__ == '__main__':
    main()
