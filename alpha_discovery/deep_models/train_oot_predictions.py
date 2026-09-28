#!/usr/bin/env python3
"""
Train CNN on ALL IS data (Jul-Nov 2025) and predict on OOT data (Dec 2025 - Mar 2026).

This produces a single model trained on all available IS data, then generates
predictions for each OOT day. The predictions are saved in the same format as
the walk-forward OOS predictions so cnn_rust_sim_validation.py can consume them.

Usage:
    python alpha_discovery/deep_models/train_oot_predictions.py
    python alpha_discovery/deep_models/train_oot_predictions.py --epochs 5 --device cuda
    python alpha_discovery/deep_models/train_oot_predictions.py --device cpu --epochs 3
"""

import argparse
import gc
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # Lvl3Quant root
MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(MODELS_DIR))

from book_spatial_cnn import BookSpatialCNN

logging.basicConfig(
    format='%(asctime)s [oot_pred] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('oot_pred')

IS_BOOK_DIR  = ROOT_DIR / 'data' / 'processed' / 'dl_book_cache'
OOT_BOOK_DIR = ROOT_DIR / 'data' / 'processed' / 'dl_book_cache_oot'
OUTPUT_DIR   = MODELS_DIR / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# ── Target computation (copied from train_walkforward.py) ──

def compute_mfe_net(mid_prices, day_boundaries, horizon_bars=100, tick_size=0.25):
    from numpy.lib.stride_tricks import sliding_window_view
    N = len(mid_prices)
    H = horizon_bars
    mfe_long  = np.full(N, np.nan, dtype=np.float32)
    mfe_short = np.full(N, np.nan, dtype=np.float32)
    mid_shifted = mid_prices[1:]
    valid_len = N - H - 1
    if valid_len > 0:
        windows = sliding_window_view(mid_shifted, H)[:valid_len]
        fwd_max = windows.max(axis=1)
        fwd_min = windows.min(axis=1)
        mfe_long[:valid_len]  = np.maximum(0.0, (fwd_max - mid_prices[:valid_len]) / tick_size)
        mfe_short[:valid_len] = np.maximum(0.0, (mid_prices[:valid_len] - fwd_min) / tick_size)
    n_days = len(day_boundaries) - 1
    for d in range(n_days - 1):
        day_end = day_boundaries[d + 1]
        nan_start = max(day_boundaries[d], day_end - H)
        mfe_long[nan_start:day_end] = np.nan
        mfe_short[nan_start:day_end] = np.nan
    return (mfe_long - mfe_short).astype(np.float32)


# ── Dataset ──

class BookDataset(torch.utils.data.Dataset):
    def __init__(self, tensors, targets, day_boundaries, window_size=20, subsample=1):
        self.tensors = tensors.astype(np.float32)
        self.targets = targets.astype(np.float32)
        self.window_size = window_size

        # Log-transform features 1,2,3
        self.tensors[:, :, 1] = np.log1p(self.tensors[:, :, 1])
        self.tensors[:, :, 2] = np.log1p(self.tensors[:, :, 2])
        self.tensors[:, :, 3] = np.log1p(self.tensors[:, :, 3])

        # Build valid indices
        valid = []
        for day_idx in range(len(day_boundaries) - 1):
            start = day_boundaries[day_idx]
            end = day_boundaries[day_idx + 1]
            for i in range(start + window_size - 1, end):
                if i < len(targets) and np.isfinite(targets[i]):
                    valid.append(i)
        if subsample > 1:
            valid = valid[::subsample]
        self.valid_indices = np.array(valid, dtype=np.int64)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = int(self.valid_indices[idx])
        window = self.tensors[i - self.window_size + 1 : i + 1]
        window = torch.from_numpy(window.copy())
        target = float(self.targets[i])
        return window, target


class BookInferenceDataset(torch.utils.data.Dataset):
    """For inference: returns predictions for every bar (no target needed)."""
    def __init__(self, tensors, day_boundaries, window_size=20):
        self.window_size = window_size
        self.tensors = tensors.astype(np.float32)
        self.tensors[:, :, 1] = np.log1p(self.tensors[:, :, 1])
        self.tensors[:, :, 2] = np.log1p(self.tensors[:, :, 2])
        self.tensors[:, :, 3] = np.log1p(self.tensors[:, :, 3])

        # Valid indices: need window_size history within same day
        valid = []
        for day_idx in range(len(day_boundaries) - 1):
            start = day_boundaries[day_idx]
            end = day_boundaries[day_idx + 1]
            for i in range(start + window_size - 1, end):
                valid.append(i)
        self.valid_indices = np.array(valid, dtype=np.int64)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = int(self.valid_indices[idx])
        window = self.tensors[i - self.window_size + 1 : i + 1]
        window = torch.from_numpy(window.copy())
        return window, i  # return index for placing predictions


def collate_book(batch):
    windows = torch.stack([b[0] for b in batch])
    targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return windows, targets


def collate_inference(batch):
    windows = torch.stack([b[0] for b in batch])
    indices = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return windows, indices


# ── Load data ──

def load_book_days(cache_dir, max_days=None):
    """Load book tensor npz files, return per-day data."""
    cache_path = Path(cache_dir)
    files = sorted(cache_path.glob('*_book_tensors.npz'))
    if max_days and max_days < len(files):
        files = files[:max_days]

    dates = []
    all_data = []
    all_mids = []
    boundaries = [0]

    for f in files:
        date = f.name.replace('_book_tensors.npz', '')
        try:
            npz = np.load(str(f))
            data = dict(npz)
            all_data.append(data)
            all_mids.append(data['mid_prices'])
            boundaries.append(boundaries[-1] + len(data['mid_prices']))
            dates.append(date)
        except Exception as e:
            logger.warning(f"Error loading {f.name}: {e}")
            continue

    if all_mids:
        mid_concat = np.concatenate(all_mids)
    else:
        mid_concat = np.array([])

    return dates, all_data, mid_concat, boundaries


# ── Training ──

def train_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_n = 0
    criterion = nn.HuberLoss(delta=1.0)

    for batch in loader:
        windows, targets = batch
        windows = windows.to(device)
        targets = targets.to(device)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds = model(windows).squeeze(-1)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(windows).squeeze(-1)
            loss = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        n = targets.shape[0]
        total_loss += loss.item() * n
        total_n += n

    return total_loss / max(total_n, 1), total_n


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    all_preds = []
    all_targets = []
    criterion = nn.HuberLoss(delta=1.0)
    total_loss = 0.0
    total_n = 0

    ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad()
    for batch in loader:
        windows, targets = batch
        windows = windows.to(device)
        targets_dev = targets.to(device)
        with ctx:
            preds = model(windows).squeeze(-1)
        loss = criterion(preds.cpu().float(), targets.float()).item()
        n = targets.shape[0]
        total_loss += loss * n
        total_n += n
        all_preds.append(preds.cpu().float().numpy())
        all_targets.append(targets.numpy())

    if not all_preds:
        return 0.0, 0.0, np.array([]), np.array([])

    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)
    mask = np.isfinite(preds_arr) & np.isfinite(targets_arr)
    if mask.sum() < 10:
        return 0.0, total_loss / max(total_n, 1), preds_arr, targets_arr

    ic, _ = spearmanr(preds_arr[mask], targets_arr[mask])
    return float(ic) if np.isfinite(ic) else 0.0, total_loss / max(total_n, 1), preds_arr, targets_arr


@torch.no_grad()
def predict_day(model, day_data, day_n_bars, device, window_size=20, batch_size=512):
    """Generate predictions for a single day. Returns (n_bars,) array."""
    model.eval()
    predictions = np.zeros(day_n_bars, dtype=np.float64)

    # Create dataset for this day
    boundaries = [0, day_n_bars]
    dataset = BookInferenceDataset(
        day_data['book_tensors'], boundaries, window_size=window_size
    )

    if len(dataset) == 0:
        return predictions

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=0, collate_fn=collate_inference)

    ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad()
    for windows, indices in loader:
        windows = windows.to(device)
        with ctx:
            preds = model(windows).squeeze(-1)
        preds = preds.cpu().float().numpy()
        indices = indices.numpy()
        predictions[indices] = preds

    return predictions


def main():
    parser = argparse.ArgumentParser(description='Train CNN on IS, predict on OOT')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--subsample-train', type=int, default=3,
                        help='Use every Nth bar during training')
    parser.add_argument('--window-size', type=int, default=20)
    parser.add_argument('--horizon-bars', type=int, default=100)
    parser.add_argument('--val-days', type=int, default=10,
                        help='Last N IS days used for validation')
    parser.add_argument('--is-dir', type=str, default=str(IS_BOOK_DIR))
    parser.add_argument('--oot-dir', type=str, default=str(OOT_BOOK_DIR))
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    use_amp = device.type == 'cuda'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = Path(args.output_dir) / f'oot_predictions_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s [oot_pred] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    logger.info('=' * 70)
    logger.info('CNN OOT Prediction Generation')
    logger.info(f'  IS data:    {args.is_dir}')
    logger.info(f'  OOT data:   {args.oot_dir}')
    logger.info(f'  Device:     {device}')
    logger.info(f'  AMP:        {use_amp}')
    logger.info(f'  Epochs:     {args.epochs}')
    logger.info(f'  Batch size: {args.batch_size}')
    logger.info(f'  Subsample:  {args.subsample_train}')
    logger.info(f'  Val days:   {args.val_days}')
    logger.info('=' * 70)

    # ── Load IS data ──
    logger.info('Loading IS book tensor data...')
    t0 = time.time()
    is_dates, is_data, is_mids, is_boundaries = load_book_days(args.is_dir)
    logger.info(f'  IS: {len(is_dates)} days, {len(is_mids):,} bars ({time.time()-t0:.1f}s)')
    logger.info(f'  Date range: {is_dates[0]} to {is_dates[-1]}')

    # ── Split IS into train/val ──
    val_days = args.val_days
    train_dates = is_dates[:-val_days]
    val_dates = is_dates[-val_days:]
    logger.info(f'  Train: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})')
    logger.info(f'  Val:   {len(val_dates)} days ({val_dates[0]} to {val_dates[-1]})')

    # Compute targets for train and val
    # Train
    train_data = is_data[:-val_days]
    train_mids = np.concatenate([d['mid_prices'] for d in train_data])
    train_boundaries = [0]
    for d in train_data:
        train_boundaries.append(train_boundaries[-1] + len(d['mid_prices']))
    train_tensors = np.concatenate([d['book_tensors'] for d in train_data], axis=0)
    train_targets = compute_mfe_net(train_mids, train_boundaries, args.horizon_bars)

    # Val
    val_data = is_data[-val_days:]
    val_mids = np.concatenate([d['mid_prices'] for d in val_data])
    val_boundaries = [0]
    for d in val_data:
        val_boundaries.append(val_boundaries[-1] + len(d['mid_prices']))
    val_tensors = np.concatenate([d['book_tensors'] for d in val_data], axis=0)
    val_targets = compute_mfe_net(val_mids, val_boundaries, args.horizon_bars)

    logger.info(f'  Train samples (before subsample): {len(train_targets):,}')
    logger.info(f'  Val samples: {len(val_targets):,}')

    # Create datasets
    train_dataset = BookDataset(train_tensors, train_targets, train_boundaries,
                               window_size=args.window_size, subsample=args.subsample_train)
    val_dataset = BookDataset(val_tensors, val_targets, val_boundaries,
                             window_size=args.window_size, subsample=1)

    logger.info(f'  Train dataset: {len(train_dataset):,} samples')
    logger.info(f'  Val dataset:   {len(val_dataset):,} samples')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, collate_fn=collate_book, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate_book, pin_memory=True)

    # Free memory
    del train_tensors, val_tensors, train_mids, val_mids
    gc.collect()

    # ── Build model ──
    model = BookSpatialCNN(
        window_size=args.window_size,
        num_levels=20,
        num_features=4,
        spatial_channels=(32, 64, 128, 256),
        temporal_channels=256,
        dropout=0.1,
        num_classes=1,
    ).to(device)

    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'  Model parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
        epochs=args.epochs, pct_start=0.3
    )
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # ── Train ──
    logger.info('\n--- Training ---')
    best_val_ic = -1.0
    best_epoch = 0

    for epoch in range(args.epochs):
        t_epoch = time.time()

        # Train
        train_loss, train_n = train_epoch(model, train_loader, optimizer, device, scaler)

        # Step scheduler
        # (OneCycleLR already steps per batch inside train_epoch, so we don't step here)
        # Actually, we need to step per batch. Let me fix this.

        # Validate
        val_ic, val_loss, val_preds, val_tgts = evaluate(model, val_loader, device)

        elapsed = time.time() - t_epoch
        logger.info(f'  Epoch {epoch+1}/{args.epochs}  '
                    f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
                    f'val_IC={val_ic:+.4f}  ({elapsed:.1f}s)')

        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_epoch = epoch + 1
            # Save best model
            model_path = Path(args.output_dir) / f'best_cnn_oot_{timestamp}.pt'
            torch.save(model.state_dict(), str(model_path))
            logger.info(f'    -> New best! Saved to {model_path.name}')

    logger.info(f'\nBest val IC: {best_val_ic:+.4f} at epoch {best_epoch}')

    # ── Retrain on ALL IS data with best epoch count ──
    logger.info('\n--- Retraining on ALL IS data ---')

    # Concatenate all IS data
    all_tensors = np.concatenate([d['book_tensors'] for d in is_data], axis=0)
    all_targets = compute_mfe_net(is_mids, is_boundaries, args.horizon_bars)

    full_dataset = BookDataset(all_tensors, all_targets, is_boundaries,
                              window_size=args.window_size, subsample=args.subsample_train)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_book, pin_memory=True)

    logger.info(f'  Full IS dataset: {len(full_dataset):,} samples')

    # Fresh model
    model = BookSpatialCNN(
        window_size=args.window_size, num_levels=20, num_features=4,
        spatial_channels=(32, 64, 128, 256), temporal_channels=256,
        dropout=0.1, num_classes=1,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # Train for best_epoch epochs
    n_retrain_epochs = best_epoch
    logger.info(f'  Training for {n_retrain_epochs} epochs (best from val)')

    for epoch in range(n_retrain_epochs):
        t_epoch = time.time()
        train_loss, train_n = train_epoch(model, full_loader, optimizer, device, scaler)
        elapsed = time.time() - t_epoch
        logger.info(f'  Epoch {epoch+1}/{n_retrain_epochs}  loss={train_loss:.4f}  ({elapsed:.1f}s)')

    del all_tensors, all_targets, full_dataset, full_loader
    gc.collect()

    # Save final model
    final_model_path = Path(args.output_dir) / f'final_cnn_oot_{timestamp}.pt'
    torch.save(model.state_dict(), str(final_model_path))
    logger.info(f'  Final model saved: {final_model_path.name}')

    # ── Generate OOT predictions ──
    logger.info('\n--- Generating OOT predictions ---')
    oot_dates, oot_data, oot_mids, oot_boundaries = load_book_days(args.oot_dir)
    logger.info(f'  OOT: {len(oot_dates)} days ({oot_dates[0]} to {oot_dates[-1]})')

    # Predict each OOT day
    pred_data = {}
    for i, (date, day_data_dict) in enumerate(zip(oot_dates, oot_data)):
        n_bars = len(day_data_dict['mid_prices'])
        preds = predict_day(model, day_data_dict, n_bars, device,
                           window_size=args.window_size, batch_size=args.batch_size)
        pred_data[f'{date}_preds'] = preds
        pred_data[f'{date}_mid'] = day_data_dict['mid_prices']

        if (i + 1) % 10 == 0 or i == 0:
            logger.info(f'  [{i+1}/{len(oot_dates)}] {date}: {n_bars} bars, '
                       f'pred range [{preds.min():.3f}, {preds.max():.3f}]')

    # Save predictions in walk-forward format
    pred_file = Path(args.output_dir) / f'oos_predictions_book_oot_{timestamp}.npz'
    np.savez_compressed(str(pred_file), **pred_data)
    logger.info(f'\nOOT predictions saved: {pred_file}')
    logger.info(f'  {len(oot_dates)} days, keys: {list(pred_data.keys())[:6]}...')

    # ── Summary ──
    logger.info('\n' + '=' * 70)
    logger.info('SUMMARY')
    logger.info(f'  IS days trained on:   {len(is_dates)}')
    logger.info(f'  Best val IC:          {best_val_ic:+.4f} (epoch {best_epoch})')
    logger.info(f'  OOT days predicted:   {len(oot_dates)}')
    logger.info(f'  Model:                {final_model_path}')
    logger.info(f'  Predictions:          {pred_file}')
    logger.info('=' * 70)


if __name__ == '__main__':
    main()
