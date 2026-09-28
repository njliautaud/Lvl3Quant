"""
Multi-Horizon CNN Walk-Forward Launcher.

Monkey-patches train_walkforward.py to train a MultiHorizonCNN instead of the
standard single-head BookSpatialCNN. Each fold computes 4 MFE targets at
different horizons (10s, 30s, 1min, 5min) and trains all heads simultaneously
using a weighted loss.

Output directory: results/multi_horizon_cnn/

Key differences from standard walk-forward:
  1. build_model returns MultiHorizonCNN (shared backbone + 4 heads)
  2. _forward_batch returns multi-horizon predictions + targets
  3. Training loop uses MultiHorizonLoss (weighted sum of per-horizon Huber)
  4. Evaluation reports IC per horizon, not just a single IC
  5. Uses research harness static holdout for fast testing

Usage:
    python run_multi_horizon_cnn.py                      # defaults (GPU)
    python run_multi_horizon_cnn.py --device cpu          # CPU mode
    python run_multi_horizon_cnn.py --epochs 5            # more epochs
    python run_multi_horizon_cnn.py --wider               # wider backbone
"""

import sys
import os
import gc
import json
import time
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# Set thread limits BEFORE importing torch
os.environ['OMP_NUM_THREADS'] = '8'
os.environ['MKL_NUM_THREADS'] = '8'
os.environ['OPENBLAS_NUM_THREADS'] = '8'

import torch
import torch.nn as nn
from scipy.stats import spearmanr
from torch.utils.data import DataLoader

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent  # Lvl3Quant root
MODELS_DIR   = Path(__file__).resolve().parent
sys.path.insert(0, str(MODELS_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

import train_walkforward as twf
from multi_horizon_cnn import MultiHorizonCNN, MultiHorizonLoss

logger = logging.getLogger('multi_horizon')

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
HORIZONS     = [100, 300, 600, 3000]   # 10s, 30s, 1min, 5min
LOSS_WEIGHTS = {100: 1.0, 300: 0.8, 600: 0.6, 3000: 0.4}

RESULTS_DIR = MODELS_DIR / 'results' / 'multi_horizon_cnn'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

DEFAULT_BOOK_DIR = str(PROJECT_ROOT / 'data' / 'processed' / 'dl_book_cache')


# =============================================================================
# Monkey-patched forward batch: returns multi-horizon predictions + targets
# =============================================================================

def _forward_batch_multi(model, batch, model_type, device):
    """
    Forward batch for MultiHorizonCNN.

    batch is a tuple of (windows, target_dict) where target_dict maps
    horizon -> target tensor. This is assembled by our custom collate_fn.

    Returns:
        predictions: Dict[int, Tensor] from model
        targets: Dict[int, Tensor] moved to device
    """
    windows, target_dict = batch
    windows = windows.to(device)

    predictions = model(windows)

    targets_on_device = {}
    for h, tgt in target_dict.items():
        targets_on_device[h] = tgt.to(device)

    return predictions, targets_on_device


# =============================================================================
# Multi-horizon dataset: computes targets at all 4 horizons
# =============================================================================

class MultiHorizonBarDataset(torch.utils.data.Dataset):
    """
    Wraps book tensor data with multi-horizon MFE targets.

    For each bar, computes mfe_net at horizons [100, 300, 600, 3000] bars.
    Samples are windows of consecutive book snapshots, same as the standard
    BarDataset used by train_walkforward.py.
    """

    def __init__(
        self,
        day_data_list: List[Dict],
        targets_by_horizon: Dict[int, np.ndarray],
        day_boundaries: List[int],
        horizons: List[int],
        window_size: int = 20,
        subsample: int = 1,
    ):
        self.window_size = window_size
        self.horizons = horizons

        # Concatenate book tensors across days
        all_tensors = []
        for d in day_data_list:
            all_tensors.append(d['book_tensors'])
        self.tensors = np.concatenate(all_tensors, axis=0)  # (N, 20, 4)

        # Store targets per horizon
        self.targets = {}
        for h in horizons:
            self.targets[h] = targets_by_horizon[h]

        # Build valid indices: must have window_size history AND valid target
        # at the longest horizon. Use the longest horizon to determine validity.
        max_horizon = max(horizons)
        n_total = len(self.tensors)

        self.valid_indices = []
        n_days = len(day_boundaries) - 1

        for d in range(n_days):
            day_start = day_boundaries[d]
            day_end   = day_boundaries[d + 1]

            # Need window_size bars before, and max_horizon bars after
            for i in range(day_start + window_size - 1, day_end - max_horizon):
                # Check that at least the shortest horizon target is finite
                if np.isfinite(self.targets[horizons[0]][i]):
                    self.valid_indices.append(i)

        # Subsample
        if subsample > 1:
            self.valid_indices = self.valid_indices[::subsample]

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]

        # Window of book snapshots: (window_size, 20, 4)
        window = self.tensors[i - self.window_size + 1 : i + 1].copy()
        window = torch.from_numpy(window.astype(np.float32))

        # Targets per horizon
        target_dict = {}
        for h in self.horizons:
            target_dict[h] = torch.tensor(self.targets[h][i], dtype=torch.float32)

        return window, target_dict


def multi_horizon_collate_fn(batch):
    """
    Custom collate that handles the (window, target_dict) format.

    Returns:
        windows: (B, window_size, 20, 4) tensor
        targets: Dict[int, (B,) tensor] per horizon
    """
    windows = torch.stack([b[0] for b in batch])

    horizons = list(batch[0][1].keys())
    targets = {}
    for h in horizons:
        targets[h] = torch.stack([b[1][h] for b in batch])

    return windows, targets


# =============================================================================
# Training loop for multi-horizon model
# =============================================================================

def train_epoch_multi(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: MultiHorizonLoss,
    device: torch.device,
    scaler=None,
    use_amp: bool = False,
) -> Tuple[float, Dict[int, float]]:
    """
    Train one epoch with multi-horizon loss.
    Uses bfloat16 autocast when use_amp=True (no GradScaler needed for bfloat16).

    Returns:
        (avg_total_loss, avg_per_horizon_losses)
    """
    model.train()
    total_loss = 0.0
    total_n = 0
    accumulated_per_h = {h: 0.0 for h in criterion.horizons}

    for batch in loader:
        optimizer.zero_grad()

        if use_amp:
            with torch.amp.autocast('cuda', dtype=torch.bfloat16):
                preds, targets = _forward_batch_multi(model, batch, 'book', device)
                loss, per_h = criterion(preds, targets)
        else:
            preds, targets = _forward_batch_multi(model, batch, 'book', device)
            loss, per_h = criterion(preds, targets)

        if not torch.isfinite(loss):
            optimizer.zero_grad()
            continue

        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        n = next(iter(targets.values())).shape[0]
        total_loss += loss.item() * n
        total_n += n
        for h in criterion.horizons:
            accumulated_per_h[h] += per_h.get(h, 0.0) * n

    avg_loss = total_loss / max(total_n, 1)
    avg_per_h = {h: v / max(total_n, 1) for h, v in accumulated_per_h.items()}
    return avg_loss, avg_per_h


@torch.no_grad()
def evaluate_multi(
    model: nn.Module,
    loader: DataLoader,
    criterion: MultiHorizonLoss,
    device: torch.device,
    horizons: List[int],
) -> Tuple[Dict[int, float], float, Dict[int, float]]:
    """
    Evaluate multi-horizon model. Computes IC per horizon.

    Returns:
        ic_per_horizon: {horizon: Spearman IC}
        avg_total_loss: float
        loss_per_horizon: {horizon: avg loss}
    """
    model.eval()

    all_preds = {h: [] for h in horizons}
    all_targets = {h: [] for h in horizons}
    total_loss = 0.0
    total_n = 0

    ctx = torch.amp.autocast('cuda', dtype=torch.bfloat16) if device.type == 'cuda' else torch.no_grad()

    for batch in loader:
        with ctx:
            preds, targets = _forward_batch_multi(model, batch, 'book', device)
            loss, per_h = criterion(preds, targets)

        n = next(iter(targets.values())).shape[0]
        total_loss += loss.item() * n
        total_n += n

        for h in horizons:
            all_preds[h].append(preds[h].cpu().float().numpy())
            all_targets[h].append(targets[h].cpu().float().numpy())

    avg_loss = total_loss / max(total_n, 1)

    # Compute IC per horizon
    ic_per_horizon = {}
    loss_per_horizon = {}

    for h in horizons:
        if not all_preds[h]:
            ic_per_horizon[h] = 0.0
            loss_per_horizon[h] = 0.0
            continue

        p = np.concatenate(all_preds[h])
        t = np.concatenate(all_targets[h])
        mask = np.isfinite(p) & np.isfinite(t)

        if mask.sum() < 10:
            ic_per_horizon[h] = 0.0
        else:
            ic, _ = spearmanr(p[mask], t[mask])
            ic_per_horizon[h] = float(ic) if np.isfinite(ic) else 0.0

        loss_per_horizon[h] = per_h.get(h, 0.0)

    return ic_per_horizon, avg_loss, loss_per_horizon


# =============================================================================
# Holdout Mode — Fast Validation (train on N days, test on M days, ONE run)
# =============================================================================

def run_multi_horizon_holdout(
    book_dir: str = DEFAULT_BOOK_DIR,
    train_days: int = 60,
    oot_days: int = 20,
    window_size: int = 20,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 3e-4,
    subsample_train: int = 5,
    device_str: str = 'cuda',
    seed: int = 42,
    wider: bool = False,
    use_attention_heads: bool = False,
):
    """
    Static holdout: train on first N days, test on last M days.
    Much faster than WF — one training run gives definitive IC per horizon.
    """
    import train_walkforward as twf

    torch.manual_seed(seed)
    np.random.seed(seed)
    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    use_amp = device.type == 'cuda'

    spatial = (64, 128, 256, 512) if wider else (32, 64, 128, 256)
    temporal = 512 if wider else 256

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = RESULTS_DIR / f'holdout_{timestamp}.log'
    logger = logging.getLogger('multi_horizon_holdout')
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    ch = logging.StreamHandler()
    ch.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(ch)

    # Load all dates
    cache_dir = Path(book_dir)
    day_files = sorted(cache_dir.glob('*_book_tensors.npz'))
    dates = [f.name.split('_book_tensors')[0] for f in day_files]

    total = len(dates)
    if total < train_days + oot_days:
        logger.error(f'Not enough days: {total} < {train_days} + {oot_days}')
        return

    train_dates = dates[:train_days]
    oot_dates = dates[train_days:train_days + oot_days]

    logger.info('=' * 70)
    logger.info('MULTI-HORIZON CNN — HOLDOUT VALIDATION')
    logger.info(f'  Train: {train_dates[0]} to {train_dates[-1]} ({len(train_dates)} days)')
    logger.info(f'  OOT:   {oot_dates[0]} to {oot_dates[-1]} ({len(oot_dates)} days)')
    logger.info(f'  Horizons: {HORIZONS}')
    logger.info(f'  Architecture: spatial={spatial} temporal={temporal}')
    logger.info(f'  Settings: epochs={epochs}, batch={batch_size}, lr={lr}, subsample={subsample_train}')
    logger.info('=' * 70)

    # Load data
    train_data_list, train_mids, train_bounds = twf.load_day_files(cache_dir, 'book', train_dates)
    oot_data_list, oot_mids, oot_bounds = twf.load_day_files(cache_dir, 'book', oot_dates)

    if not train_data_list or not oot_data_list:
        logger.error('Failed to load data')
        return

    # Compute targets at all horizons
    train_targets = {}
    oot_targets = {}
    for h in HORIZONS:
        train_tgt = twf.compute_mfe_net(train_mids, train_bounds, horizon_bars=h)
        oot_tgt = twf.compute_mfe_net(oot_mids, oot_bounds, horizon_bars=h)
        # Z-score normalize using train stats
        mask = np.isfinite(train_tgt)
        tgt_mean = float(train_tgt[mask].mean()) if mask.sum() > 0 else 0.0
        tgt_std = float(train_tgt[mask].std()) if mask.sum() > 0 else 1.0
        if tgt_std < 1e-8: tgt_std = 1.0
        train_targets[h] = ((train_tgt - tgt_mean) / tgt_std).astype(np.float32)
        oot_targets[h] = ((oot_tgt - tgt_mean) / tgt_std).astype(np.float32)
        label = MultiHorizonCNN.HORIZON_LABELS.get(h, str(h))
        logger.info(f'  {label}: mean={tgt_mean:.3f} std={tgt_std:.3f} ticks')

    # Build datasets
    train_dataset = MultiHorizonBarDataset(train_data_list, train_targets, train_bounds,
                                           horizons=HORIZONS, window_size=window_size, subsample=subsample_train)
    oot_dataset = MultiHorizonBarDataset(oot_data_list, oot_targets, oot_bounds,
                                         horizons=HORIZONS, window_size=window_size, subsample=1)

    logger.info(f'  Dataset: {len(train_dataset):,} train | {len(oot_dataset):,} OOT')

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=0, pin_memory=False, collate_fn=multi_horizon_collate_fn)
    oot_loader = DataLoader(oot_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=0, pin_memory=False, collate_fn=multi_horizon_collate_fn)

    # Build model
    model = MultiHorizonCNN(
        horizons=HORIZONS, spatial_channels=spatial, temporal_channels=temporal,
        dropout=0.15 if wider else 0.1, window_size=window_size,
        use_attention_heads=use_attention_heads,
    ).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f'  Model: {n_params:,} params')

    criterion = MultiHorizonLoss(horizons=HORIZONS)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)

    # Train
    best_oot_loss = float('inf')
    best_ics = {}
    for epoch in range(epochs):
        t_ep = time.time()
        train_loss, _ = train_epoch_multi(model, train_loader, optimizer, criterion, device,
                                          scaler=None, use_amp=use_amp)
        ic_per_h, oot_loss, _ = evaluate_multi(model, oot_loader, criterion, device, HORIZONS)

        ic_str = ' | '.join(f'{MultiHorizonCNN.HORIZON_LABELS.get(h, str(h))}={ic_per_h.get(h, 0):.4f}' for h in HORIZONS)
        elapsed = time.time() - t_ep
        logger.info(f'  Epoch {epoch+1}/{epochs} ({elapsed:.0f}s): train_loss={train_loss:.4f} oot_loss={oot_loss:.4f} | IC: {ic_str}')

        if oot_loss < best_oot_loss:
            best_oot_loss = oot_loss
            best_ics = dict(ic_per_h)

    # Save model + results
    torch.save(model.state_dict(), RESULTS_DIR / f'holdout_model_{timestamp}.pt')

    result = {
        'mode': 'holdout',
        'train_dates': f'{train_dates[0]}..{train_dates[-1]}',
        'oot_dates': f'{oot_dates[0]}..{oot_dates[-1]}',
        'train_days': len(train_dates),
        'oot_days': len(oot_dates),
        'architecture': f'spatial={spatial} temporal={temporal}',
        'params': n_params,
        'epochs': epochs,
        'best_ics': {str(h): float(v) for h, v in best_ics.items()},
        'best_oot_loss': float(best_oot_loss),
        'timestamp': timestamp,
    }
    with open(RESULTS_DIR / f'holdout_result_{timestamp}.json', 'w') as f:
        json.dump(result, f, indent=2)

    logger.info('')
    logger.info('=' * 70)
    logger.info('HOLDOUT RESULT:')
    for h in HORIZONS:
        label = MultiHorizonCNN.HORIZON_LABELS.get(h, str(h))
        ic = best_ics.get(h, 0)
        logger.info(f'  {label}: IC={ic:+.4f}')
    logger.info(f'  Params: {n_params:,} | OOT loss: {best_oot_loss:.4f}')
    logger.info('=' * 70)

    return result


# =============================================================================
# Walk-Forward with Multi-Horizon Targets
# =============================================================================

def run_multi_horizon_walkforward(
    book_dir: str = DEFAULT_BOOK_DIR,
    n_days: Optional[int] = None,
    min_train_days: int = 5,
    max_train_days: Optional[int] = 15,
    purge_days: int = 1,
    window_size: int = 20,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 3e-4,
    subsample_train: int = 5,
    device_str: str = 'cuda',
    seed: int = 42,
    wider: bool = False,
    warm_start: bool = True,
    start_fold: int = 0,
    use_attention_heads: bool = False,
):
    """
    Walk-forward training for the multi-horizon CNN.

    Uses the same expanding/sliding window protocol as train_walkforward.py
    but with multi-horizon targets and evaluation.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device_str if torch.cuda.is_available() else 'cpu')
    # Re-enabled AMP with float32 loss computation fix in MultiHorizonLoss
    use_amp = device.type == 'cuda'

    # Determine spatial channels
    if wider:
        spatial_channels = (64, 128, 256, 512)
        temporal_channels = 512
        dropout = 0.15
    else:
        spatial_channels = (32, 64, 128, 256)
        temporal_channels = 256
        dropout = 0.1

    # Discover available dates from book cache
    book_path = Path(book_dir)
    npz_files = sorted(book_path.glob('*_book_tensors.npz'))
    if not npz_files:
        raise FileNotFoundError(f'No book tensor files in {book_dir}')

    dates = [f.name.split('_book_tensors')[0] for f in npz_files]
    if n_days is not None:
        dates = dates[-n_days:]

    n_total = len(dates)
    n_folds = n_total - min_train_days - purge_days
    if n_folds <= 0:
        raise ValueError(f'Not enough dates ({n_total}) for walk-forward '
                         f'with min_train_days={min_train_days}')

    timestamp = time.strftime('%Y%m%d_%H%M%S')
    log_path = RESULTS_DIR / f'multi_horizon_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s: %(message)s', '%H:%M:%S'))
    logger.addHandler(fh)
    logger.setLevel(logging.INFO)
    # Also log to stdout
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        logger.addHandler(logging.StreamHandler())

    logger.info('=' * 70)
    logger.info('MULTI-HORIZON CNN WALK-FORWARD')
    logger.info(f'  Horizons: {HORIZONS} bars = [10s, 30s, 1min, 5min]')
    logger.info(f'  Loss weights: {LOSS_WEIGHTS}')
    logger.info(f'  Architecture: spatial={spatial_channels} temporal={temporal_channels}')
    logger.info(f'  Dates: {dates[0]} to {dates[-1]} ({n_total} days, {n_folds} folds)')
    logger.info(f'  Settings: epochs={epochs}, batch={batch_size}, lr={lr}, '
                f'subsample={subsample_train}, device={device}')
    logger.info('=' * 70)

    criterion = MultiHorizonLoss(horizons=HORIZONS, weights=LOSS_WEIGHTS)

    fold_results = []
    prev_model_state = None

    for fold_idx, test_day_idx in enumerate(range(min_train_days + purge_days, n_total)):
        if fold_idx < start_fold:
            continue

        train_end_idx = test_day_idx - purge_days
        if max_train_days and train_end_idx > max_train_days:
            train_start_idx = train_end_idx - max_train_days
        else:
            train_start_idx = 0
        train_dates = dates[train_start_idx:train_end_idx]
        test_dates  = [dates[test_day_idx]]

        t_fold = time.time()
        logger.info(f'\n--- Fold {fold_idx+1}/{n_folds} | '
                     f'Train: {train_dates[0]}..{train_dates[-1]} ({len(train_dates)}d) | '
                     f'Test: {test_dates[0]} ---')

        # ---- Load train data ----
        train_data_list, train_mids, train_bounds = twf.load_day_files(
            book_dir, 'book', train_dates
        )
        if not train_data_list:
            logger.warning(f'  Fold {fold_idx+1}: no train data, skipping')
            continue

        # ---- Load test data ----
        test_data_list, test_mids, test_bounds = twf.load_day_files(
            book_dir, 'book', test_dates
        )
        if not test_data_list:
            logger.warning(f'  Fold {fold_idx+1}: no test data, skipping')
            continue

        # ---- Compute MFE targets at ALL horizons ----
        train_targets = {}
        test_targets  = {}
        tgt_stats     = {}  # for z-score normalisation

        for h in HORIZONS:
            train_tgt = twf.compute_mfe_net(train_mids, train_bounds, horizon_bars=h)
            test_tgt  = twf.compute_mfe_net(test_mids, test_bounds, horizon_bars=h)

            # Z-score normalise using train statistics
            finite_mask = np.isfinite(train_tgt)
            tgt_mean = float(train_tgt[finite_mask].mean()) if finite_mask.sum() > 0 else 0.0
            tgt_std  = float(train_tgt[finite_mask].std())  if finite_mask.sum() > 0 else 1.0
            if tgt_std < 1e-8:
                tgt_std = 1.0

            train_targets[h] = ((train_tgt - tgt_mean) / tgt_std).astype(np.float32)
            test_targets[h]  = ((test_tgt  - tgt_mean) / tgt_std).astype(np.float32)
            tgt_stats[h] = {'mean': tgt_mean, 'std': tgt_std}

        n_valid = int(np.isfinite(train_targets[HORIZONS[0]]).sum())
        logger.info(f'  Targets computed: {n_valid:,} valid bars (shortest horizon)')
        for h in HORIZONS:
            s = tgt_stats[h]
            label = MultiHorizonCNN.HORIZON_LABELS.get(h, str(h))
            logger.info(f'    {label:>5s}: mean={s["mean"]:.3f} std={s["std"]:.3f} ticks')

        if n_valid < 1000:
            logger.warning(f'  Too few valid train bars ({n_valid}), skipping')
            continue

        # ---- Build datasets ----
        train_dataset = MultiHorizonBarDataset(
            train_data_list, train_targets, train_bounds,
            horizons=HORIZONS, window_size=window_size,
            subsample=subsample_train,
        )
        test_dataset = MultiHorizonBarDataset(
            test_data_list, test_targets, test_bounds,
            horizons=HORIZONS, window_size=window_size,
            subsample=1,
        )

        logger.info(f'  Dataset: {len(train_dataset):,} train | {len(test_dataset):,} test')

        if len(train_dataset) < 100:
            logger.warning(f'  Too few train samples, skipping')
            continue

        # Free raw data
        del train_data_list, test_data_list
        del train_mids, test_mids, train_targets, test_targets
        gc.collect()

        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=multi_horizon_collate_fn, num_workers=0,
            pin_memory=False, drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size * 2, shuffle=False,
            collate_fn=multi_horizon_collate_fn, num_workers=0,
            pin_memory=False,
        )

        # ---- Build model ----
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()

        model = MultiHorizonCNN(
            horizons=HORIZONS,
            head_hidden_dim=64,
            window_size=window_size,
            spatial_channels=spatial_channels,
            temporal_channels=temporal_channels,
            dropout=dropout,
        ).to(device)

        n_params = sum(p.numel() for p in model.parameters())

        if warm_start and prev_model_state is not None:
            try:
                model.load_state_dict(prev_model_state)
                logger.info(f'  Model: {n_params:,} params (warm start)')
            except Exception as e:
                logger.warning(f'  Warm start failed ({e}), fresh weights')
                logger.info(f'  Model: {n_params:,} params')
        else:
            logger.info(f'  Model: {n_params:,} params')

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3,
        )
        # bfloat16 autocast: no GradScaler needed (no underflow unlike float16)
        # use_amp flag passed to train_epoch_multi for bfloat16 autocast

        # ---- Train ----
        best_val_loss = float('inf')
        for epoch in range(epochs):
            t_ep = time.time()
            train_loss, train_per_h = train_epoch_multi(
                model, train_loader, optimizer, criterion, device,
                scaler=None, use_amp=use_amp
            )
            # Step scheduler after each batch is handled by OneCycleLR
            # (it auto-steps via the training loop)

            # Quick validation
            ic_per_h, val_loss, val_per_h = evaluate_multi(
                model, test_loader, criterion, device, HORIZONS
            )

            elapsed = time.time() - t_ep
            ic_str = ' | '.join(
                f'{MultiHorizonCNN.HORIZON_LABELS[h]}={ic_per_h[h]:.4f}'
                for h in HORIZONS
            )
            logger.info(
                f'  Epoch {epoch+1}/{epochs} ({elapsed:.0f}s): '
                f'train_loss={train_loss:.4f} val_loss={val_loss:.4f} | IC: {ic_str}'
            )

            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # ---- Final evaluation ----
        ic_per_h, val_loss, _ = evaluate_multi(
            model, test_loader, criterion, device, HORIZONS
        )

        fold_result = {
            'fold': fold_idx + 1,
            'test_date': test_dates[0],
            'val_loss': val_loss,
            'ic_per_horizon': {str(h): ic_per_h[h] for h in HORIZONS},
        }
        fold_results.append(fold_result)

        ic_summary = ' | '.join(
            f'{MultiHorizonCNN.HORIZON_LABELS[h]}={ic_per_h[h]:.4f}'
            for h in HORIZONS
        )
        logger.info(f'  RESULT: {ic_summary}  (val_loss={val_loss:.4f})')

        # Save model state for warm start
        prev_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}

        # Save checkpoint
        ckpt_path = RESULTS_DIR / f'checkpoint_multi_horizon_{timestamp}.json'
        ckpt_data = {
            'completed_folds': fold_idx + 1,
            'fold_results': fold_results,
            'horizons': HORIZONS,
            'spatial_channels': list(spatial_channels),
            'temporal_channels': temporal_channels,
            'timestamp': timestamp,
        }
        with open(ckpt_path, 'w') as f:
            json.dump(ckpt_data, f, indent=2)

        # Save model weights
        torch.save(
            model.state_dict(),
            RESULTS_DIR / f'latest_multi_horizon.pt',
        )

        # Cleanup — robust handling to prevent CUDA driver crashes between folds
        try:
            if device.type == 'cuda':
                model.cpu()  # Move model to CPU before deleting
            del model, optimizer, scheduler
            if scaler is not None:
                del scaler
            del train_loader, test_loader, train_dataset, test_dataset
        except Exception as e:
            logger.warning(f'  Cleanup warning: {e}')
        gc.collect()
        if device.type == 'cuda':
            try:
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            except Exception as e:
                logger.warning(f'  CUDA cleanup warning: {e}')
        # Brief sleep to let GPU driver stabilize between folds
        time.sleep(2)

        fold_time = time.time() - t_fold
        logger.info(f'  Fold time: {fold_time:.0f}s')

    # ========================================================================
    # Summary
    # ========================================================================
    logger.info('\n' + '=' * 70)
    logger.info('MULTI-HORIZON CNN — WALK-FORWARD COMPLETE')
    logger.info('=' * 70)

    if fold_results:
        for h in HORIZONS:
            label = MultiHorizonCNN.HORIZON_LABELS[h]
            ics = [r['ic_per_horizon'][str(h)] for r in fold_results]
            mean_ic = np.mean(ics)
            std_ic  = np.std(ics)
            logger.info(f'  {label:>5s} IC: mean={mean_ic:.4f} std={std_ic:.4f} '
                         f'(min={min(ics):.4f} max={max(ics):.4f}) over {len(ics)} folds')

        # Save final summary
        summary = {
            'experiment': 'multi_horizon_cnn',
            'horizons': HORIZONS,
            'loss_weights': LOSS_WEIGHTS,
            'spatial_channels': list(spatial_channels),
            'temporal_channels': temporal_channels,
            'n_folds': len(fold_results),
            'fold_results': fold_results,
            'mean_ic_per_horizon': {},
        }
        for h in HORIZONS:
            label = MultiHorizonCNN.HORIZON_LABELS[h]
            ics = [r['ic_per_horizon'][str(h)] for r in fold_results]
            summary['mean_ic_per_horizon'][label] = {
                'mean': float(np.mean(ics)),
                'std': float(np.std(ics)),
                'min': float(min(ics)),
                'max': float(max(ics)),
            }

        summary_path = RESULTS_DIR / f'summary_multi_horizon_{timestamp}.json'
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=2)
        logger.info(f'\nSummary saved to: {summary_path}')
    else:
        logger.warning('No folds completed.')

    logger.info('Done.')
    return fold_results


# =============================================================================
# CLI
# =============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Multi-Horizon CNN Walk-Forward Training',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--book-dir', type=str, default=DEFAULT_BOOK_DIR,
                        help='Directory with _book_tensors.npz files')
    parser.add_argument('--days', type=int, default=None,
                        help='Number of trading days (None=all)')
    parser.add_argument('--min-train-days', type=int, default=5)
    parser.add_argument('--max-train-days', type=int, default=15,
                        help='Sliding window cap (None=expanding)')
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--subsample-train', type=int, default=5)
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'])
    parser.add_argument('--wider', action='store_true', default=False,
                        help='Use wider backbone (64,128,256,512)')
    parser.add_argument('--attention-heads', action='store_true', default=False,
                        help='Use attention-based horizon heads (each head learns which features matter)')
    parser.add_argument('--warm-start', action='store_true', default=True)
    parser.add_argument('--no-warm-start', action='store_true', default=False)
    parser.add_argument('--start-fold', type=int, default=0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--mode', type=str, default='walkforward',
                        choices=['walkforward', 'holdout'],
                        help='holdout: train on first N days, test on last M days (fast validation)')
    parser.add_argument('--train-days', type=int, default=60,
                        help='Days for training in holdout mode')
    parser.add_argument('--oot-days', type=int, default=20,
                        help='Days for OOT testing in holdout mode')

    args = parser.parse_args()

    warm_start = args.warm_start and not args.no_warm_start

    if args.mode == 'holdout':
        run_multi_horizon_holdout(
            book_dir=args.book_dir,
            train_days=args.train_days,
            oot_days=args.oot_days,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            subsample_train=args.subsample_train,
            device_str=args.device,
            seed=args.seed,
            wider=args.wider,
            use_attention_heads=args.attention_heads,
        )
    else:
        run_multi_horizon_walkforward(
            book_dir=args.book_dir,
            n_days=args.days,
            min_train_days=args.min_train_days,
            max_train_days=args.max_train_days,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            subsample_train=args.subsample_train,
            device_str=args.device,
            seed=args.seed,
            wider=args.wider,
            warm_start=warm_start,
            start_fold=args.start_fold,
            use_attention_heads=args.attention_heads,
        )


if __name__ == '__main__':
    main()
