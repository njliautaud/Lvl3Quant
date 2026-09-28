#!/usr/bin/env python3
"""
Train GNN on ALL IS data (Jul-Nov 2025) and predict on OOT data (Dec 2025 - Mar 2026).

Mirrors train_oot_predictions.py but for BookGNN instead of BookSpatialCNN.
Outputs predictions in the same format so generate_oot_sim_predictions.py can
convert them to per-day vol-gated files for the Rust fill simulator.

Usage:
    python alpha_discovery/deep_models/train_gnn_oot_predictions.py
    python alpha_discovery/deep_models/train_gnn_oot_predictions.py --epochs 5 --device cuda
    python alpha_discovery/deep_models/train_gnn_oot_predictions.py --device cpu --epochs 3
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
from torch.utils.data import DataLoader, Dataset

ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # Lvl3Quant root
MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(MODELS_DIR))

from book_gnn import BookGNN, count_parameters

logging.basicConfig(
    format='%(asctime)s [gnn_oot] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('gnn_oot')

IS_BOOK_DIR  = ROOT_DIR / 'data' / 'processed' / 'dl_book_cache'
OOT_BOOK_DIR = ROOT_DIR / 'data' / 'processed' / 'dl_book_cache_oot'
OUTPUT_DIR   = MODELS_DIR / 'results'
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


# -- Target computation (same as train_gnn.py / train_oot_predictions.py) --

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


# -- Dataset (same as train_gnn.py GNNBarDataset) --

class GNNBarDataset(Dataset):
    def __init__(self, day_data_list, target, day_boundaries, window_size=20, subsample=1):
        self.window_size = window_size
        tensors = np.concatenate([d['book_tensors'] for d in day_data_list], axis=0)
        tensors[:, :, 1] = np.log1p(tensors[:, :, 1])
        tensors[:, :, 2] = np.log1p(tensors[:, :, 2])
        tensors[:, :, 3] = np.log1p(tensors[:, :, 3])
        self.tensors = tensors.astype(np.float32)
        self.target = target.astype(np.float32)

        valid = []
        for day_idx in range(len(day_boundaries) - 1):
            start = day_boundaries[day_idx]
            end = day_boundaries[day_idx + 1]
            for i in range(start + window_size - 1, end):
                if i < len(target) and np.isfinite(target[i]):
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
        target = float(self.target[i])
        return window, target


class GNNInferenceDataset(Dataset):
    """For inference: returns predictions for every bar (no target needed)."""
    def __init__(self, tensors, day_boundaries, window_size=20):
        self.window_size = window_size
        self.tensors = tensors.astype(np.float32)
        self.tensors[:, :, 1] = np.log1p(self.tensors[:, :, 1])
        self.tensors[:, :, 2] = np.log1p(self.tensors[:, :, 2])
        self.tensors[:, :, 3] = np.log1p(self.tensors[:, :, 3])

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
        return window, i


def collate_gnn(batch):
    windows = torch.stack([b[0] for b in batch])
    targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return windows, targets


def collate_inference(batch):
    windows = torch.stack([b[0] for b in batch])
    indices = torch.tensor([b[1] for b in batch], dtype=torch.long)
    return windows, indices


# -- Load data --

def load_book_days(cache_dir, max_days=None):
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

    mid_concat = np.concatenate(all_mids) if all_mids else np.array([])
    return dates, all_data, mid_concat, boundaries


# -- Training --

def train_epoch(model, loader, optimizer, device, scaler=None):
    model.train()
    total_loss = 0.0
    total_n = 0
    criterion = nn.HuberLoss(delta=1.0)

    for windows, targets in loader:
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

    for windows, targets in loader:
        windows = windows.to(device)
        targets_dev = targets.to(device)
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                preds = model(windows).squeeze(-1)
        else:
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

    boundaries = [0, day_n_bars]
    dataset = GNNInferenceDataset(
        day_data['book_tensors'], boundaries, window_size=window_size
    )

    if len(dataset) == 0:
        return predictions

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                       num_workers=0, collate_fn=collate_inference)

    for windows, indices in loader:
        windows = windows.to(device)
        if device.type == 'cuda':
            with torch.amp.autocast('cuda'):
                preds = model(windows).squeeze(-1)
        else:
            preds = model(windows).squeeze(-1)
        preds = preds.cpu().float().numpy()
        indices = indices.numpy()
        predictions[indices] = preds

    return predictions


def main():
    parser = argparse.ArgumentParser(description='Train GNN on IS, predict on OOT')
    parser.add_argument('--epochs', type=int, default=5)
    parser.add_argument('--batch-size', type=int, default=512)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--subsample-train', type=int, default=3)
    parser.add_argument('--window-size', type=int, default=20)
    parser.add_argument('--horizon-bars', type=int, default=100)
    parser.add_argument('--val-days', type=int, default=10)
    parser.add_argument('--hidden-dim', type=int, default=64)
    parser.add_argument('--num-gcn-layers', type=int, default=3)
    parser.add_argument('--temporal-dim', type=int, default=128)
    parser.add_argument('--dropout', type=float, default=0.2)
    parser.add_argument('--is-dir', type=str, default=str(IS_BOOK_DIR))
    parser.add_argument('--oot-dir', type=str, default=str(OOT_BOOK_DIR))
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR))
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
    use_amp = device.type == 'cuda'

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path = Path(args.output_dir) / f'gnn_oot_predictions_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s [gnn_oot] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    logger.info('=' * 70)
    logger.info('GNN OOT Prediction Generation')
    logger.info(f'  IS data:      {args.is_dir}')
    logger.info(f'  OOT data:     {args.oot_dir}')
    logger.info(f'  Device:       {device}')
    logger.info(f'  AMP:          {use_amp}')
    logger.info(f'  Epochs:       {args.epochs}')
    logger.info(f'  Batch size:   {args.batch_size}')
    logger.info(f'  Subsample:    {args.subsample_train}')
    logger.info(f'  Val days:     {args.val_days}')
    logger.info(f'  Hidden dim:   {args.hidden_dim}')
    logger.info(f'  GCN layers:   {args.num_gcn_layers}')
    logger.info(f'  Temporal dim: {args.temporal_dim}')
    logger.info('=' * 70)

    # -- Load IS data --
    logger.info('Loading IS book tensor data...')
    t0 = time.time()
    is_dates, is_data, is_mids, is_boundaries = load_book_days(args.is_dir)
    logger.info(f'  IS: {len(is_dates)} days, {len(is_mids):,} bars ({time.time()-t0:.1f}s)')
    logger.info(f'  Date range: {is_dates[0]} to {is_dates[-1]}')

    # -- Split IS into train/val --
    val_days = args.val_days
    train_dates = is_dates[:-val_days]
    val_dates = is_dates[-val_days:]
    logger.info(f'  Train: {len(train_dates)} days ({train_dates[0]} to {train_dates[-1]})')
    logger.info(f'  Val:   {len(val_dates)} days ({val_dates[0]} to {val_dates[-1]})')

    # Train data
    train_data = is_data[:-val_days]
    train_mids = np.concatenate([d['mid_prices'] for d in train_data])
    train_boundaries = [0]
    for d in train_data:
        train_boundaries.append(train_boundaries[-1] + len(d['mid_prices']))
    train_targets = compute_mfe_net(train_mids, train_boundaries, args.horizon_bars)

    # Val data
    val_data = is_data[-val_days:]
    val_mids = np.concatenate([d['mid_prices'] for d in val_data])
    val_boundaries = [0]
    for d in val_data:
        val_boundaries.append(val_boundaries[-1] + len(d['mid_prices']))
    val_targets = compute_mfe_net(val_mids, val_boundaries, args.horizon_bars)

    logger.info(f'  Train samples: {len(train_targets):,}')
    logger.info(f'  Val samples: {len(val_targets):,}')

    # Create datasets
    train_dataset = GNNBarDataset(train_data, train_targets, train_boundaries,
                                  window_size=args.window_size, subsample=args.subsample_train)
    val_dataset = GNNBarDataset(val_data, val_targets, val_boundaries,
                                window_size=args.window_size, subsample=1)

    logger.info(f'  Train dataset: {len(train_dataset):,} samples')
    logger.info(f'  Val dataset:   {len(val_dataset):,} samples')

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             num_workers=0, collate_fn=collate_gnn, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                           num_workers=0, collate_fn=collate_gnn, pin_memory=True)

    del train_mids, val_mids
    gc.collect()

    # -- Build model --
    model = BookGNN(
        window_size=args.window_size,
        num_nodes=20,
        in_features=4,
        node_features=5,
        hidden_dim=args.hidden_dim,
        num_gcn_layers=args.num_gcn_layers,
        temporal_dim=args.temporal_dim,
        dropout=args.dropout,
        num_classes=1,
        use_gat=False,
    ).to(device)

    n_params = count_parameters(model)
    logger.info(f'  Model parameters: {n_params:,}')

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    # -- Train (find best epoch) --
    logger.info('\n--- Training (finding best epoch) ---')
    best_val_ic = -1.0
    best_epoch = 0

    for epoch in range(args.epochs):
        t_epoch = time.time()
        train_loss, train_n = train_epoch(model, train_loader, optimizer, device, scaler)
        val_ic, val_loss, _, _ = evaluate(model, val_loader, device)
        elapsed = time.time() - t_epoch
        logger.info(f'  Epoch {epoch+1}/{args.epochs}  '
                    f'train_loss={train_loss:.4f}  val_loss={val_loss:.4f}  '
                    f'val_IC={val_ic:+.4f}  ({elapsed:.1f}s)')

        if val_ic > best_val_ic:
            best_val_ic = val_ic
            best_epoch = epoch + 1
            model_path = Path(args.output_dir) / f'best_gnn_oot_{timestamp}.pt'
            torch.save(model.state_dict(), str(model_path))
            logger.info(f'    -> New best! Saved to {model_path.name}')

    logger.info(f'\nBest val IC: {best_val_ic:+.4f} at epoch {best_epoch}')

    # -- Retrain on ALL IS data with best epoch count --
    logger.info('\n--- Retraining on ALL IS data ---')

    all_targets = compute_mfe_net(is_mids, is_boundaries, args.horizon_bars)
    full_dataset = GNNBarDataset(is_data, all_targets, is_boundaries,
                                 window_size=args.window_size, subsample=args.subsample_train)
    full_loader = DataLoader(full_dataset, batch_size=args.batch_size, shuffle=True,
                            num_workers=0, collate_fn=collate_gnn, pin_memory=True)

    logger.info(f'  Full IS dataset: {len(full_dataset):,} samples')

    # Fresh model
    model = BookGNN(
        window_size=args.window_size, num_nodes=20, in_features=4, node_features=5,
        hidden_dim=args.hidden_dim, num_gcn_layers=args.num_gcn_layers,
        temporal_dim=args.temporal_dim, dropout=args.dropout, num_classes=1, use_gat=False,
    ).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler('cuda') if use_amp else None

    n_retrain_epochs = best_epoch
    logger.info(f'  Training for {n_retrain_epochs} epochs (best from val)')

    for epoch in range(n_retrain_epochs):
        t_epoch = time.time()
        train_loss, train_n = train_epoch(model, full_loader, optimizer, device, scaler)
        elapsed = time.time() - t_epoch
        logger.info(f'  Epoch {epoch+1}/{n_retrain_epochs}  loss={train_loss:.4f}  ({elapsed:.1f}s)')

    del all_targets, full_dataset, full_loader
    gc.collect()

    # Save final model
    final_model_path = Path(args.output_dir) / f'final_gnn_oot_{timestamp}.pt'
    torch.save(model.state_dict(), str(final_model_path))
    logger.info(f'  Final model saved: {final_model_path.name}')

    # -- Generate OOT predictions --
    logger.info('\n--- Generating OOT predictions ---')
    oot_dates, oot_data, oot_mids, oot_boundaries = load_book_days(args.oot_dir)
    logger.info(f'  OOT: {len(oot_dates)} days ({oot_dates[0]} to {oot_dates[-1]})')

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

    # Save predictions in walk-forward format (same as CNN train_oot_predictions.py)
    pred_file = Path(args.output_dir) / f'oos_predictions_gnn_oot_{timestamp}.npz'
    np.savez_compressed(str(pred_file), **pred_data)
    logger.info(f'\nGNN OOT predictions saved: {pred_file}')
    logger.info(f'  {len(oot_dates)} days, keys: {list(pred_data.keys())[:6]}...')

    # -- Summary --
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
