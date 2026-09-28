"""
Walk-Forward Training Script for BookGNN.

Uses the SAME protocol as train_walkforward.py so results are directly comparable
to BookSpatialCNN:
  - Expanding-window walk-forward cross-validation
  - 1-day purge gap between train and test
  - Target: mfe_net_10s (z-score normalized using train statistics)
  - Metric: Spearman IC per fold
  - Same NPZ data from dl_book_cache

Usage:
  python train_gnn.py --folds 5 --epochs 3 --hidden 64 --layers 2
  python train_gnn.py --folds 20 --epochs 5 --hidden 64 --layers 3 --use-gat
  python train_gnn.py --folds 1 --epochs 1  # quick sanity check

Results saved to: alpha_discovery/deep_models/results/
Checkpoints saved to: alpha_discovery/deep_models/checkpoints/
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # Lvl3Quant root
MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(MODELS_DIR))

from book_gnn import BookGNN, count_parameters  # noqa: E402

# Try to import notifier (optional)
try:
    from alpha_discovery.compute_notifier import notify_complete  # noqa: E402
except ImportError:
    def notify_complete(*args, **kwargs):
        pass

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('train_gnn')

DEFAULT_BOOK_DIR = str(ROOT_DIR / 'data' / 'processed' / 'dl_book_cache')
DEFAULT_OUTPUT_DIR = str(MODELS_DIR / 'results')
DEFAULT_CKPT_DIR = str(MODELS_DIR / 'checkpoints')


# ---------------------------------------------------------------------------
# MFE Target (copied from train_walkforward.py for standalone usage)
# ---------------------------------------------------------------------------

def compute_mfe_net(mid_prices: np.ndarray, day_boundaries: List[int],
                    horizon_bars: int = 100, tick_size: float = 0.25) -> np.ndarray:
    """
    Compute mfe_net_10s = mfe_long - mfe_short over a forward window.

    mfe_long[t]  = max(0, max(mid[t+1:t+H+1]) - mid[t]) / tick_size
    mfe_short[t] = max(0, mid[t] - min(mid[t+1:t+H+1])) / tick_size
    mfe_net[t]   = mfe_long[t] - mfe_short[t]

    NaN at bars where the forward window crosses a day boundary.
    """
    from numpy.lib.stride_tricks import sliding_window_view

    N = len(mid_prices)
    H = horizon_bars

    mfe_long = np.full(N, np.nan, dtype=np.float32)
    mfe_short = np.full(N, np.nan, dtype=np.float32)

    mid_shifted = mid_prices[1:]
    valid_len = N - H - 1

    if valid_len > 0:
        windows = sliding_window_view(mid_shifted, H)[:valid_len]
        fwd_max = windows.max(axis=1)
        fwd_min = windows.min(axis=1)

        mfe_long[:valid_len] = np.maximum(0.0, (fwd_max - mid_prices[:valid_len]) / tick_size)
        mfe_short[:valid_len] = np.maximum(0.0, (mid_prices[:valid_len] - fwd_min) / tick_size)

    n_days = len(day_boundaries) - 1
    for d in range(n_days - 1):
        day_end = day_boundaries[d + 1]
        nan_start = max(day_boundaries[d], day_end - H)
        mfe_long[nan_start:day_end] = np.nan
        mfe_short[nan_start:day_end] = np.nan

    return (mfe_long - mfe_short).astype(np.float32)


# ---------------------------------------------------------------------------
# Data Loading
# ---------------------------------------------------------------------------

def get_available_dates(cache_dir: str) -> List[str]:
    """Return sorted list of YYYY-MM-DD date strings in cache dir."""
    files = sorted(Path(cache_dir).glob('*_book_tensors.npz'))
    return [f.name.replace('_book_tensors.npz', '') for f in files]


def load_day_files(cache_dir: str, dates: List[str]):
    """
    Load book tensor NPZ files for the given dates.

    Returns:
        data_list: list of dicts with 'book_tensors' and 'mid_prices'
        mid_concat: concatenated mid_prices
        boundaries: day boundary indices
    """
    cache_path = Path(cache_dir)
    all_data = []
    all_mids = []
    boundaries = [0]

    for date in dates:
        fname = cache_path / f'{date}_book_tensors.npz'
        if not fname.exists():
            logger.warning(f'  Missing file: {fname.name}, skipping')
            continue
        npz = np.load(fname)
        all_data.append(dict(npz))
        all_mids.append(npz['mid_prices'])
        boundaries.append(boundaries[-1] + len(npz['mid_prices']))

    mid_concat = np.concatenate(all_mids) if all_mids else np.array([])
    return all_data, mid_concat, boundaries


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class GNNBarDataset(Dataset):
    """
    Dataset for BookGNN training.
    Same interface as BarDataset in train_walkforward.py for 'book' model type.

    Loads book tensors, log-transforms features 1-3, builds valid indices
    respecting day boundaries and window requirements.
    """

    def __init__(
        self,
        day_data_list: List[Dict],
        target: np.ndarray,
        day_boundaries: List[int],
        window_size: int = 20,
        subsample: int = 1,
    ):
        self.window_size = window_size
        self.subsample = subsample

        # Concatenate book tensors
        tensors = np.concatenate([d['book_tensors'] for d in day_data_list], axis=0)
        # Log-transform features 1,2,3 (depth, orders, age) — same as CNN pipeline
        tensors[:, :, 1] = np.log1p(tensors[:, :, 1])
        tensors[:, :, 2] = np.log1p(tensors[:, :, 2])
        tensors[:, :, 3] = np.log1p(tensors[:, :, 3])
        self.tensors = tensors.astype(np.float32)

        self.target = target.astype(np.float32)
        self.day_boundaries = day_boundaries
        N = len(target)

        # Build valid indices: need window_size bars of history within same day
        valid_set = set()
        for day_idx in range(len(day_boundaries) - 1):
            start = day_boundaries[day_idx]
            end = day_boundaries[day_idx + 1]
            for i in range(start + window_size - 1, end):
                if i < N and np.isfinite(target[i]):
                    valid_set.add(i)

        all_valid = sorted(valid_set)
        if subsample > 1:
            all_valid = all_valid[::subsample]

        self.valid_indices = np.array(all_valid, dtype=np.int64)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = int(self.valid_indices[idx])
        window = self.tensors[i - self.window_size + 1: i + 1]  # (W, 20, 4)
        window = torch.from_numpy(window)
        target = float(self.target[i])
        return window, target


def collate_gnn(batch):
    """Collate function for GNN DataLoader."""
    windows = torch.stack([b[0] for b in batch])  # (B, W, 20, 4)
    targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)  # (B,)
    return windows, targets


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> Tuple[float, int]:
    """Train for one epoch. Returns (avg_loss, n_samples)."""
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
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Evaluate model on loader.
    Returns (IC, val_loss, preds_array, targets_array).
    """
    model.eval()
    all_preds = []
    all_targets = []
    criterion = nn.HuberLoss(delta=1.0)
    total_loss = 0.0
    total_n = 0

    ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad()

    for windows, targets in loader:
        windows = windows.to(device)
        targets_dev = targets.to(device)

        with ctx:
            preds = model(windows).squeeze(-1)

        val_l = criterion(preds.cpu().float(), targets.float()).item()
        n = targets.shape[0]
        total_loss += val_l * n
        total_n += n
        all_preds.append(preds.cpu().float().numpy())
        all_targets.append(targets.numpy())

    if not all_preds:
        return 0.0, 0.0, np.array([]), np.array([])

    val_loss = total_loss / max(total_n, 1)
    preds_arr = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)

    mask = np.isfinite(preds_arr) & np.isfinite(targets_arr)
    if mask.sum() < 10:
        return 0.0, val_loss, preds_arr, targets_arr

    ic, _ = spearmanr(preds_arr[mask], targets_arr[mask])
    return float(ic) if np.isfinite(ic) else 0.0, val_loss, preds_arr, targets_arr


# ---------------------------------------------------------------------------
# Walk-Forward Main Loop
# ---------------------------------------------------------------------------

def train_walkforward(args):
    """Walk-forward training for BookGNN."""
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device(
        'cuda' if torch.cuda.is_available() and args.device == 'cuda' else 'cpu'
    )
    use_amp = (device.type == 'cuda')

    os.makedirs(args.output_dir, exist_ok=True)
    os.makedirs(args.ckpt_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # File logging
    log_path = Path(args.output_dir) / f'walkforward_gnn_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter('%(asctime)s [%(name)s] %(levelname)s: %(message)s',
                                       datefmt='%H:%M:%S'))
    logger.addHandler(fh)

    logger.info('=' * 70)
    logger.info('Walk-Forward Training: BookGNN')
    logger.info(f'  Log file:        {log_path}')
    logger.info(f'  Cache dir:       {args.book_dir}')
    logger.info(f'  Hidden dim:      {args.hidden}')
    logger.info(f'  GCN layers:      {args.layers}')
    logger.info(f'  Temporal dim:    {args.temporal_dim}')
    logger.info(f'  Use GAT:         {args.use_gat}')
    logger.info(f'  Window size:     {args.window_size}')
    logger.info(f'  Epochs/fold:     {args.epochs}')
    logger.info(f'  Batch size:      {args.batch_size}')
    logger.info(f'  LR:              {args.lr}')
    logger.info(f'  Subsample train: {args.subsample_train}')
    logger.info(f'  Device:          {device}')
    logger.info(f'  AMP:             {use_amp}')
    logger.info('=' * 70)

    # Get available dates
    dates = get_available_dates(args.book_dir)
    n_total = len(dates)
    logger.info(f'  Available: {n_total} days: {dates[0]} .. {dates[-1]}')

    # Apply folds limit
    min_train_days = args.min_train_days
    purge_days = args.purge_days
    max_folds = n_total - min_train_days - purge_days

    if args.folds and args.folds < max_folds:
        # Only use enough days for the requested number of folds
        n_use = min_train_days + purge_days + args.folds
        dates = dates[:n_use]
        n_total = len(dates)
        logger.info(f'  Using {n_total} days for {args.folds} folds')

    n_folds = n_total - min_train_days - purge_days
    if n_folds <= 0:
        logger.error(f'Not enough days for walk-forward: {n_total} days, '
                     f'need at least {min_train_days + purge_days + 1}')
        return

    logger.info(f'  Total folds: {n_folds}')

    # Build model once to show param count
    model_tmp = BookGNN(
        window_size=args.window_size, hidden_dim=args.hidden,
        num_gcn_layers=args.layers, temporal_dim=args.temporal_dim,
        dropout=args.dropout, num_classes=1,
        use_gat=args.use_gat, gat_heads=args.gat_heads,
    )
    logger.info(f'  Model parameters: {count_parameters(model_tmp):,}')
    del model_tmp

    fold_ics = []
    fold_details = []
    all_oos_preds = []
    all_oos_tgts = []

    for fold_idx, test_day_idx in enumerate(range(min_train_days + purge_days, n_total)):

        train_end_idx = test_day_idx - purge_days
        train_dates = dates[:train_end_idx]  # expanding window
        test_dates = [dates[test_day_idx]]

        t_fold_start = time.time()
        logger.info(f'\n--- Fold {fold_idx+1}/{n_folds} | '
                    f'Train: {train_dates[0]}..{train_dates[-1]} ({len(train_dates)}d) | '
                    f'Test: {test_dates[0]} ---')

        # Load train data
        train_data, train_mids, train_bounds = load_day_files(args.book_dir, train_dates)
        if not train_data:
            logger.warning(f'  Fold {fold_idx+1}: no train data, skipping')
            continue

        # Compute MFE target
        train_target = compute_mfe_net(train_mids, train_bounds, args.horizon_bars)
        n_train_valid = int(np.isfinite(train_target).sum())
        if n_train_valid < 1000:
            logger.warning(f'  Fold {fold_idx+1}: too few train samples ({n_train_valid})')
            continue

        # Load test data
        test_data, test_mids, test_bounds = load_day_files(args.book_dir, test_dates)
        if not test_data:
            logger.warning(f'  Fold {fold_idx+1}: no test data, skipping')
            continue

        test_target = compute_mfe_net(test_mids, test_bounds, args.horizon_bars)
        n_test_valid = int(np.isfinite(test_target).sum())
        if n_test_valid < 100:
            logger.warning(f'  Fold {fold_idx+1}: too few test samples ({n_test_valid})')
            continue

        # Z-score normalize targets (using TRAIN statistics only)
        finite_mask = np.isfinite(train_target)
        tgt_mean = float(train_target[finite_mask].mean())
        tgt_std = float(train_target[finite_mask].std())
        if tgt_std < 1e-8:
            tgt_std = 1.0
        train_target = (train_target - tgt_mean) / tgt_std
        test_target = (test_target - tgt_mean) / tgt_std
        logger.info(f'  Target norm: mean={tgt_mean:.3f} std={tgt_std:.3f} ticks')
        logger.info(f'  Train: {n_train_valid:,} valid | Test: {n_test_valid:,} valid')

        # Build datasets
        train_dataset = GNNBarDataset(
            train_data, train_target, train_bounds,
            window_size=args.window_size, subsample=args.subsample_train,
        )
        test_dataset = GNNBarDataset(
            test_data, test_target, test_bounds,
            window_size=args.window_size, subsample=1,
        )

        logger.info(f'  Dataset: {len(train_dataset):,} train '
                    f'(sub={args.subsample_train}x) | {len(test_dataset):,} test')

        if len(train_dataset) < 100:
            logger.warning(f'  Too few train samples, skipping')
            continue

        # Free raw arrays
        del train_data, test_data, train_mids, test_mids
        gc.collect()

        # DataLoaders
        train_loader = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=True,
            num_workers=0, pin_memory=True, collate_fn=collate_gnn,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=args.batch_size * 2, shuffle=False,
            num_workers=0, pin_memory=True, collate_fn=collate_gnn,
        )

        # Build fresh model each fold (no information leakage)
        model = BookGNN(
            window_size=args.window_size, hidden_dim=args.hidden,
            num_gcn_layers=args.layers, temporal_dim=args.temporal_dim,
            dropout=args.dropout, num_classes=1,
            use_gat=args.use_gat, gat_heads=args.gat_heads,
        ).to(device)

        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.lr, weight_decay=1e-4,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
            epochs=args.epochs, pct_start=0.3,
        )
        scaler = torch.amp.GradScaler('cuda') if use_amp else None

        # Train
        best_ic = -1.0
        best_epoch = -1
        for epoch in range(args.epochs):
            t_ep = time.time()

            # Override scheduler step to be called per batch
            # (OneCycleLR needs per-batch stepping)
            model.train()
            total_loss = 0.0
            total_n = 0
            criterion = nn.HuberLoss(delta=1.0)

            for windows, targets in train_loader:
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

                scheduler.step()
                n = targets.shape[0]
                total_loss += loss.item() * n
                total_n += n

            train_loss = total_loss / max(total_n, 1)

            # Evaluate
            ic, val_loss, preds_arr, tgts_arr = evaluate(model, test_loader, device)
            ep_time = time.time() - t_ep

            logger.info(f'  Epoch {epoch+1}/{args.epochs}: '
                       f'train_loss={train_loss:.4f} val_loss={val_loss:.4f} '
                       f'IC={ic:+.4f} ({ep_time:.1f}s)')

            # Save checkpoint
            if ic > best_ic:
                best_ic = ic
                best_epoch = epoch + 1
                ckpt_path = Path(args.ckpt_dir) / f'gnn_fold{fold_idx+1}_epoch{epoch+1}.pt'
                torch.save({
                    'model_state_dict': model.state_dict(),
                    'epoch': epoch + 1,
                    'fold': fold_idx + 1,
                    'ic': ic,
                    'train_loss': train_loss,
                    'val_loss': val_loss,
                    'args': vars(args),
                }, str(ckpt_path))

        # Record fold results
        fold_time = time.time() - t_fold_start
        logger.info(f'  Fold {fold_idx+1} complete: best IC={best_ic:+.4f} '
                    f'(epoch {best_epoch}) [{fold_time:.1f}s]')

        fold_ics.append(best_ic)
        fold_details.append({
            'fold': fold_idx + 1,
            'test_date': test_dates[0],
            'train_days': len(train_dates),
            'best_ic': best_ic,
            'best_epoch': best_epoch,
            'n_train': len(train_dataset),
            'n_test': len(test_dataset),
            'fold_time_s': fold_time,
        })

        all_oos_preds.append(preds_arr)
        all_oos_tgts.append(tgts_arr)

        # Clean up
        del model, optimizer, scheduler, scaler
        del train_dataset, test_dataset, train_loader, test_loader
        del train_target, test_target
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---------------------------------------------------------------------------
    # Summary
    # ---------------------------------------------------------------------------
    if not fold_ics:
        logger.error('No folds completed!')
        return

    fold_ics_arr = np.array(fold_ics)
    mean_ic = float(fold_ics_arr.mean())
    std_ic = float(fold_ics_arr.std())
    icir = mean_ic / std_ic if std_ic > 1e-8 else 0.0
    pct_positive = float((fold_ics_arr > 0).mean() * 100)

    logger.info('\n' + '=' * 70)
    logger.info('WALK-FORWARD RESULTS: BookGNN')
    logger.info('=' * 70)
    logger.info(f'  Folds completed:   {len(fold_ics)}')
    logger.info(f'  Mean IC:           {mean_ic:+.4f}')
    logger.info(f'  Std IC:            {std_ic:.4f}')
    logger.info(f'  ICIR:              {icir:.2f}')
    logger.info(f'  Pct positive:      {pct_positive:.1f}%')
    logger.info(f'  Per-fold ICs:      {[f"{x:+.4f}" for x in fold_ics]}')
    logger.info('=' * 70)

    # Save results
    results = {
        'model': 'BookGNN',
        'variant': 'GAT' if args.use_gat else 'GCN',
        'timestamp': timestamp,
        'mean_ic': mean_ic,
        'std_ic': std_ic,
        'icir': icir,
        'pct_positive': pct_positive,
        'fold_ics': fold_ics,
        'fold_details': fold_details,
        'config': {
            'hidden_dim': args.hidden,
            'num_gcn_layers': args.layers,
            'temporal_dim': args.temporal_dim,
            'window_size': args.window_size,
            'epochs': args.epochs,
            'batch_size': args.batch_size,
            'lr': args.lr,
            'dropout': args.dropout,
            'subsample_train': args.subsample_train,
            'horizon_bars': args.horizon_bars,
            'use_gat': args.use_gat,
            'gat_heads': args.gat_heads,
        },
    }

    results_path = Path(args.output_dir) / f'walkforward_gnn_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f'  Results saved to: {results_path}')

    # Save OOS predictions
    if all_oos_preds:
        preds_dict = {}
        for i, det in enumerate(fold_details):
            date = det['test_date']
            preds_dict[f'{date}_preds'] = all_oos_preds[i]
            preds_dict[f'{date}_targets'] = all_oos_tgts[i]
        preds_path = Path(args.output_dir) / f'gnn_oos_preds_{timestamp}.npz'
        np.savez_compressed(str(preds_path), **preds_dict)
        logger.info(f'  OOS predictions saved to: {preds_path}')

    # Try to send notification
    try:
        notify_complete(
            f"GNN Walk-Forward Complete: IC={mean_ic:+.4f}, ICIR={icir:.2f}, "
            f"{len(fold_ics)} folds, {pct_positive:.0f}% positive"
        )
    except Exception:
        pass

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description='BookGNN Walk-Forward Training')

    # Data
    parser.add_argument('--book-dir', type=str, default=DEFAULT_BOOK_DIR,
                        help='Directory with *_book_tensors.npz files')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory for results and logs')
    parser.add_argument('--ckpt-dir', type=str, default=DEFAULT_CKPT_DIR,
                        help='Directory for model checkpoints')

    # Walk-forward
    parser.add_argument('--folds', type=int, default=5,
                        help='Number of walk-forward folds')
    parser.add_argument('--min-train-days', type=int, default=5,
                        help='Minimum training days before first fold')
    parser.add_argument('--purge-days', type=int, default=1,
                        help='Gap days between train and test')
    parser.add_argument('--horizon-bars', type=int, default=100,
                        help='MFE horizon in bars (100 = 10s at 100ms)')

    # Model
    parser.add_argument('--hidden', type=int, default=64,
                        help='GCN hidden dimension')
    parser.add_argument('--layers', type=int, default=2,
                        help='Number of GCN layers')
    parser.add_argument('--temporal-dim', type=int, default=128,
                        help='Temporal conv channel dimension')
    parser.add_argument('--dropout', type=float, default=0.2,
                        help='Dropout rate')
    parser.add_argument('--use-gat', action='store_true',
                        help='Use GAT instead of GCN')
    parser.add_argument('--gat-heads', type=int, default=4,
                        help='Number of GAT attention heads')

    # Training
    parser.add_argument('--epochs', type=int, default=3,
                        help='Epochs per fold')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--window-size', type=int, default=20,
                        help='Number of consecutive bars per sample')
    parser.add_argument('--subsample-train', type=int, default=3,
                        help='Use every Nth training bar')

    # Misc
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device: cuda or cpu')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    train_walkforward(args)
