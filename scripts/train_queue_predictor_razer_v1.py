#!/usr/bin/env python3
"""
train_queue_predictor_razer_v1.py — Queue Fill Probability Predictor (Razer)
=============================================================================
Trains a 1D CNN to predict P(passive limit order fills within 1s/5s/10s).

Uses PRE-EXTRACTED windows from Jupiter (extract_queue_fill_windows.py).
Each date file is ~62 MB (vs 1-2 GB raw), fitting easily in 16GB RAM.

Walk-forward: 15-day train, 1-day OOT, sliding window (HC #0).
Saves .pt weights + .npz predictions per fold, logs to MLflow.

Data: output/queue_fill_windows/<date>_windows.npz
  windows:    (N, 25, 256) float32
  bid_labels: (N, 3) float32
  ask_labels: (N, 3) float32
  norm_sample: (20000, 25) float32

Author: Claude (Lvl3 Quant)
"""
from __future__ import annotations

import gc
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, ConcatDataset

# ============================================================================
# Config
# ============================================================================
WINDOWS_DIR = Path(r'C:\Users\claude\Lvl3Quant\output\queue_fill_windows')
OUTPUT_DIR = Path(r'C:\Users\claude\Lvl3Quant\output\queue_position_predictor_v1')
WEIGHTS_DIR = OUTPUT_DIR / 'weights'
PRED_DIR = OUTPUT_DIR / 'predictions'

for d in [OUTPUT_DIR, WEIGHTS_DIR, PRED_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Training
TRAIN_WINDOW_DAYS = 15
BATCH_SIZE = 512
EPOCHS = 12
LR = 0.0005
WEIGHT_DECAY = 1e-4
N_FEATURES = 25
WINDOW_SIZE = 256
HORIZONS = ['1s', '5s', '10s']

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# MLflow
MLFLOW_URI = 'http://neptune-win:5000'
EXPERIMENT_NAME = 'queue_fill_predictor_v1'


# ============================================================================
# Model: 1D CNN for fill probability
# ============================================================================
class QueueFillCNN(nn.Module):
    """1D CNN predicting fill probability at multiple horizons."""

    def __init__(self, n_features: int = 25, hidden: int = 64):
        super().__init__()

        # Conv blocks with residual connections
        self.conv1 = nn.Conv1d(n_features, hidden, kernel_size=7, padding=3)
        self.bn1 = nn.BatchNorm1d(hidden)

        self.conv2 = nn.Conv1d(hidden, hidden, kernel_size=5, padding=2)
        self.bn2 = nn.BatchNorm1d(hidden)

        self.conv3 = nn.Conv1d(hidden, hidden, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(hidden)

        # Project input to hidden dim for residual
        self.proj = nn.Conv1d(n_features, hidden, kernel_size=1)

        # Global average pooling → FC heads
        self.bid_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, len(HORIZONS)),
        )
        self.ask_head = nn.Sequential(
            nn.Linear(hidden, 32),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(32, len(HORIZONS)),
        )

    def forward(self, x):
        # x: (B, 25, 256)
        residual = self.proj(x)

        h = F.gelu(self.bn1(self.conv1(x)))
        h = F.gelu(self.bn2(self.conv2(h)))
        h = F.gelu(self.bn3(self.conv3(h))) + residual

        # Global avg pool
        h = h.mean(dim=2)  # (B, hidden)

        bid_logits = self.bid_head(h)
        ask_logits = self.ask_head(h)
        return bid_logits, ask_logits


# ============================================================================
# Dataset from pre-extracted windows
# ============================================================================
class WindowDataset(Dataset):
    """Memory-efficient: keeps numpy arrays, normalizes on-the-fly in __getitem__."""
    def __init__(self, windows: np.ndarray, bid_labels: np.ndarray, ask_labels: np.ndarray,
                 feat_mean: np.ndarray = None, feat_std: np.ndarray = None):
        self.windows = windows  # Keep as numpy to avoid doubling memory
        self.bid_labels = bid_labels
        self.ask_labels = ask_labels
        self.feat_mean = feat_mean  # (25,) or None
        self.feat_std = feat_std    # (25,) or None

    def __len__(self):
        return len(self.windows)

    def __getitem__(self, i):
        w = self.windows[i].astype(np.float32)  # (25, 256)
        if self.feat_mean is not None:
            w = (w - self.feat_mean[:, None]) / self.feat_std[:, None]
        return (torch.from_numpy(w),
                torch.from_numpy(self.bid_labels[i]),
                torch.from_numpy(self.ask_labels[i]))


def load_window_file(date_str: str) -> Optional[Dict]:
    """Load pre-extracted windows for one date."""
    fpath = WINDOWS_DIR / f'{date_str}_windows.npz'
    if not fpath.exists():
        return None
    try:
        d = np.load(fpath)
        result = {
            'windows': d['windows'],      # (N, 25, 256) float32
            'bid_labels': d['bid_labels'], # (N, 3) float32
            'ask_labels': d['ask_labels'], # (N, 3) float32
        }
        if 'norm_sample' in d:
            result['norm_sample'] = d['norm_sample']
        return result
    except Exception as e:
        print(f"  ERROR loading {date_str}: {e}")
        return None


# ============================================================================
# Training functions
# ============================================================================
def train_one_epoch(model, loader, optimizer, scheduler=None):
    model.train()
    total_loss = 0
    n_batches = 0

    for windows, bid_lbl, ask_lbl in loader:
        windows = windows.to(device)
        bid_lbl = bid_lbl.to(device)
        ask_lbl = ask_lbl.to(device)

        bid_logits, ask_logits = model(windows)
        loss = F.binary_cross_entropy_with_logits(bid_logits, bid_lbl) + \
               F.binary_cross_entropy_with_logits(ask_logits, ask_lbl)

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    if scheduler:
        scheduler.step()

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    all_bid_probs, all_ask_probs = [], []
    all_bid_labels, all_ask_labels = [], []
    total_loss = 0
    n_batches = 0

    for windows, bid_lbl, ask_lbl in loader:
        windows = windows.to(device)
        bid_lbl = bid_lbl.to(device)
        ask_lbl = ask_lbl.to(device)

        bid_logits, ask_logits = model(windows)
        loss = F.binary_cross_entropy_with_logits(bid_logits, bid_lbl) + \
               F.binary_cross_entropy_with_logits(ask_logits, ask_lbl)

        total_loss += loss.item()
        n_batches += 1

        all_bid_probs.append(torch.sigmoid(bid_logits).cpu().numpy())
        all_ask_probs.append(torch.sigmoid(ask_logits).cpu().numpy())
        all_bid_labels.append(bid_lbl.cpu().numpy())
        all_ask_labels.append(ask_lbl.cpu().numpy())

    bid_probs = np.concatenate(all_bid_probs)
    ask_probs = np.concatenate(all_ask_probs)
    bid_labels = np.concatenate(all_bid_labels)
    ask_labels = np.concatenate(all_ask_labels)

    # Compute AUC per horizon
    from sklearn.metrics import roc_auc_score
    metrics = {'loss': total_loss / max(n_batches, 1)}

    for hi, hz in enumerate(HORIZONS):
        for side, probs, labels in [('bid', bid_probs[:, hi], bid_labels[:, hi]),
                                     ('ask', ask_probs[:, hi], ask_labels[:, hi])]:
            try:
                auc = roc_auc_score(labels, probs)
            except ValueError:
                auc = 0.5
            fill_rate = labels.mean()
            metrics[f'{side}_{hz}_auc'] = auc
            metrics[f'{side}_{hz}_fill_rate'] = fill_rate

    return metrics, bid_probs, ask_probs, bid_labels, ask_labels


# ============================================================================
# MLflow
# ============================================================================
def init_mlflow():
    try:
        import mlflow
        import threading
        result = [None, None]
        exc = [None]

        def _connect():
            try:
                mlflow.set_tracking_uri(MLFLOW_URI)
                mlflow.set_experiment(EXPERIMENT_NAME)
                run = mlflow.start_run(run_name=f'queue_fill_cnn_v1_{time.strftime("%Y%m%d_%H%M")}')
                mlflow.log_params({
                    'model': 'QueueFillCNN_1D',
                    'train_window_days': TRAIN_WINDOW_DAYS,
                    'batch_size': BATCH_SIZE,
                    'epochs': EPOCHS,
                    'lr': LR,
                    'window_size': WINDOW_SIZE,
                    'horizons': str(HORIZONS),
                })
                result[0] = mlflow
                result[1] = run
            except Exception as e:
                exc[0] = e

        t = threading.Thread(target=_connect, daemon=True)
        t.start()
        t.join(timeout=15)  # 15 second timeout
        if t.is_alive():
            print("  MLflow init timed out (15s) — continuing without tracking", flush=True)
            return None, None
        if exc[0]:
            print(f"  MLflow init failed: {exc[0]}", flush=True)
            return None, None
        print("  MLflow connected successfully", flush=True)
        return result[0], result[1]
    except Exception as e:
        print(f"  MLflow init failed: {e}", flush=True)
        return None, None


def log_fold_metrics(mlflow_mod, fold_idx, metrics):
    if mlflow_mod is None:
        return
    try:
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                mlflow_mod.log_metric(f'fold_{k}', v, step=fold_idx)
    except Exception:
        pass


# ============================================================================
# Main
# ============================================================================
def main():
    print("=" * 70)
    print("QUEUE FILL PROBABILITY PREDICTOR v1 (Pre-extracted windows)")
    print("=" * 70)
    print(f"Device:          {device}")
    print(f"Windows dir:     {WINDOWS_DIR}")
    print(f"Output dir:      {OUTPUT_DIR}")
    print(f"Train window:    {TRAIN_WINDOW_DAYS} days (sliding)")
    print(f"Horizons:        {HORIZONS}")
    print(f"Batch size:      {BATCH_SIZE}")
    print(f"Epochs:          {EPOCHS}")
    print(f"LR:              {LR}")
    print("=" * 70, flush=True)

    # Get available dates
    all_dates = sorted([f.stem.split('_')[0]
                        for f in WINDOWS_DIR.glob('*_windows.npz')])
    print(f"\nFound {len(all_dates)} dates with pre-extracted windows")

    if len(all_dates) < TRAIN_WINDOW_DAYS + 1:
        print(f"ERROR: Need at least {TRAIN_WINDOW_DAYS + 1} dates, have {len(all_dates)}")
        return

    n_folds = len(all_dates) - TRAIN_WINDOW_DAYS
    print(f"Total folds: {n_folds}")

    # MLflow
    mlflow_mod, mlflow_run = init_mlflow()

    # Concat tracking
    concat_bid_probs, concat_ask_probs = [], []
    concat_bid_labels, concat_ask_labels = [], []
    concat_dates = []
    fold_metrics_all = []

    t_start = time.time()

    for fold_idx in range(n_folds):
        fold_t0 = time.time()
        train_dates = all_dates[fold_idx:fold_idx + TRAIN_WINDOW_DAYS]
        oot_date = all_dates[fold_idx + TRAIN_WINDOW_DAYS]

        print(f"\n--- Fold {fold_idx:03d}/{n_folds-1:03d}: "
              f"train={train_dates[0]}..{train_dates[-1]}, OOT={oot_date} ---",
              flush=True)

        # PHASE 1: Compute normalization stats from training data (small samples only)
        norm_samples = []
        for d in train_dates[:8]:  # Sample from first 8 days
            data = load_window_file(d)
            if data is not None and 'norm_sample' in data and data['norm_sample'] is not None:
                norm_samples.append(data['norm_sample'])
            del data
            gc.collect()

        if norm_samples:
            all_norm = np.concatenate(norm_samples)
            feat_mean = np.nanmean(all_norm, axis=0).astype(np.float32)
            feat_std = np.nanstd(all_norm, axis=0).astype(np.float32)
            feat_std[feat_std < 1e-8] = 1.0
            del all_norm, norm_samples
            gc.collect()
        else:
            feat_mean = np.zeros(N_FEATURES, dtype=np.float32)
            feat_std = np.ones(N_FEATURES, dtype=np.float32)

        # PHASE 2: Create per-day datasets (NO concatenation — uses ConcatDataset)
        # Each day's data stays as its own numpy array. Normalization applied on-the-fly.
        day_datasets = []
        loaded = 0
        total_samples = 0

        for d in train_dates:
            data = load_window_file(d)
            if data is None:
                continue
            # Subsample to 2000 windows max per day to cap memory
            n = len(data['windows'])
            if n > 2000:
                idx = np.random.choice(n, 2000, replace=False)
                ds = WindowDataset(data['windows'][idx], data['bid_labels'][idx],
                                    data['ask_labels'][idx], feat_mean, feat_std)
            else:
                ds = WindowDataset(data['windows'], data['bid_labels'],
                                    data['ask_labels'], feat_mean, feat_std)
            day_datasets.append(ds)
            total_samples += len(ds)
            loaded += 1
            del data
            gc.collect()

        if loaded < 5:
            print(f"  SKIP fold: only {loaded} valid train days")
            del day_datasets
            gc.collect()
            continue

        # Use ConcatDataset — no numpy concatenation, minimal extra memory
        train_ds = ConcatDataset(day_datasets)

        # Load OOT
        oot_data = load_window_file(oot_date)
        if oot_data is None:
            print(f"  SKIP fold: OOT {oot_date} not loadable")
            del train_ds, day_datasets
            gc.collect()
            continue

        oot_ds = WindowDataset(oot_data['windows'], oot_data['bid_labels'],
                                oot_data['ask_labels'], feat_mean, feat_std)
        del oot_data
        gc.collect()

        if total_samples < 100:
            print(f"  SKIP fold: only {total_samples} train samples")
            del train_ds, oot_ds, day_datasets
            gc.collect()
            continue

        print(f"  Train: {total_samples:,} samples ({loaded} days), OOT: {len(oot_ds):,} samples", flush=True)

        train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,
                                   num_workers=0, pin_memory=True, drop_last=True)
        oot_loader = DataLoader(oot_ds, batch_size=BATCH_SIZE, shuffle=False,
                                 num_workers=0, pin_memory=True)

        # Fresh model each fold
        model = QueueFillCNN(n_features=N_FEATURES, hidden=64).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        best_val_loss = float('inf')
        best_state = None

        for epoch in range(EPOCHS):
            train_loss = train_one_epoch(model, train_loader, optimizer, scheduler)

            if (epoch + 1) % 4 == 0 or epoch == EPOCHS - 1:
                oot_metrics, _, _, _, _ = evaluate(model, oot_loader)
                print(f"    Epoch {epoch+1:2d}/{EPOCHS}: "
                      f"loss={train_loss:.4f}, oot_loss={oot_metrics['loss']:.4f}, "
                      f"bid_5s_auc={oot_metrics.get('bid_5s_auc', 0):.3f}, "
                      f"ask_5s_auc={oot_metrics.get('ask_5s_auc', 0):.3f}",
                      flush=True)

                if oot_metrics['loss'] < best_val_loss:
                    best_val_loss = oot_metrics['loss']
                    best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        if best_state is not None:
            model.load_state_dict(best_state)

        # Final evaluation
        oot_metrics, bid_probs, ask_probs, bid_labels, ask_labels = evaluate(model, oot_loader)

        fold_time = time.time() - fold_t0
        print(f"  Fold {fold_idx:03d} complete ({fold_time:.0f}s):")
        for hz in HORIZONS:
            ba = oot_metrics.get(f'bid_{hz}_auc', 0)
            aa = oot_metrics.get(f'ask_{hz}_auc', 0)
            bf = oot_metrics.get(f'bid_{hz}_fill_rate', 0)
            af = oot_metrics.get(f'ask_{hz}_fill_rate', 0)
            print(f"    {hz}: bid_AUC={ba:.3f} (FR={bf:.3f}), ask_AUC={aa:.3f} (FR={af:.3f})")

        # Save weights
        torch.save({
            'model_state_dict': model.state_dict(),
            'feat_mean': feat_mean,
            'feat_std': feat_std,
            'fold_idx': fold_idx,
            'oot_date': oot_date,
            'metrics': oot_metrics,
        }, WEIGHTS_DIR / f'fold_{fold_idx:03d}_{oot_date}.pt')

        # Save predictions
        np.savez_compressed(PRED_DIR / f'fold_{fold_idx:03d}_{oot_date}.npz',
                           bid_probs=bid_probs, ask_probs=ask_probs,
                           bid_labels=bid_labels, ask_labels=ask_labels,
                           date=oot_date, fold_idx=fold_idx)

        # Accumulate
        concat_bid_probs.append(bid_probs)
        concat_ask_probs.append(ask_probs)
        concat_bid_labels.append(bid_labels)
        concat_ask_labels.append(ask_labels)
        concat_dates.append(oot_date)
        fold_metrics_all.append({**oot_metrics, 'fold_idx': fold_idx, 'oot_date': oot_date})

        log_fold_metrics(mlflow_mod, fold_idx, oot_metrics)

        # Cleanup
        del model, optimizer, scheduler, train_ds, oot_ds, day_datasets
        del train_loader, oot_loader, best_state
        gc.collect()
        torch.cuda.empty_cache()

    # ========================================================================
    # Concat evaluation
    # ========================================================================
    total_time = time.time() - t_start
    print(f"\n{'='*70}")
    print(f"WALK-FORWARD COMPLETE — {len(fold_metrics_all)} folds in {total_time:.0f}s")
    print(f"{'='*70}")

    if not concat_bid_probs:
        print("No valid folds")
        if mlflow_mod:
            mlflow_mod.end_run()
        return

    all_bid_p = np.concatenate(concat_bid_probs)
    all_ask_p = np.concatenate(concat_ask_probs)
    all_bid_l = np.concatenate(concat_bid_labels)
    all_ask_l = np.concatenate(concat_ask_labels)

    print(f"\nCONCAT OOT RESULTS ({len(all_bid_p):,} samples, {len(concat_dates)} dates):")

    from sklearn.metrics import roc_auc_score, brier_score_loss

    for hi, hz in enumerate(HORIZONS):
        for side, probs, labels in [('bid', all_bid_p[:, hi], all_bid_l[:, hi]),
                                     ('ask', all_ask_p[:, hi], all_ask_l[:, hi])]:
            try:
                auc = roc_auc_score(labels, probs)
                brier = brier_score_loss(labels, probs)
            except ValueError:
                auc, brier = 0.5, 0.25

            fr = labels.mean()
            print(f"  {side}_{hz}: AUC={auc:.4f}, Brier={brier:.4f}, FR={fr:.3f}")

            if mlflow_mod:
                try:
                    mlflow_mod.log_metric(f'concat_{side}_{hz}_auc', auc)
                    mlflow_mod.log_metric(f'concat_{side}_{hz}_brier', brier)
                except Exception:
                    pass

    # Per-fold AUC summary
    print(f"\nPer-fold AUC (bid_5s):")
    bid_5s_aucs = [m.get('bid_5s_auc', 0) for m in fold_metrics_all]
    for i, (m, d) in enumerate(zip(fold_metrics_all, concat_dates)):
        print(f"  Fold {i}: {d} — bid_5s={m.get('bid_5s_auc',0):.3f}, "
              f"ask_5s={m.get('ask_5s_auc',0):.3f}")
    print(f"  Mean bid_5s AUC: {np.mean(bid_5s_aucs):.4f}")
    print(f"  Folds > 0.55: {sum(1 for a in bid_5s_aucs if a > 0.55)}/{len(bid_5s_aucs)}")

    if mlflow_mod:
        try:
            mlflow_mod.log_metric('mean_bid_5s_auc', np.mean(bid_5s_aucs))
            mlflow_mod.log_metric('n_folds', len(fold_metrics_all))
            mlflow_mod.log_metric('n_oot_dates', len(concat_dates))
            mlflow_mod.end_run()
        except Exception:
            pass

    print(f"\nWeights: {WEIGHTS_DIR}")
    print(f"Predictions: {PRED_DIR}")
    print("DONE.")


if __name__ == '__main__':
    main()
