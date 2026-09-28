"""
MFE/MAE Prediction CNN
======================
Same BookSpatialCNN backbone (wider variant, ~12.6M params), two REGRESSION heads:
  1. mfe_head: predicts Max Favorable Excursion in next N bars (always >= 0, in ticks)
  2. mae_head: predicts Max Adverse Excursion in next N bars (always >= 0, in ticks)

Entry signal:
  IF mfe_pred > threshold * mae_pred => ENTER (direction determined by sign of mid_return)
  Threshold default: 2.0 (MFE must be at least 2x MAE to enter)

Loss:
  Pure regression: MSE on mfe_head + MSE on mae_head
  Combined: 0.5 * MSE(mfe_pred, mfe_actual) + 0.5 * MSE(mae_pred, mae_actual)
  No classification head => no NaN collapse risk

Metrics (per fold):
  - MFE IC: Spearman(mfe_pred, mfe_actual)
  - MAE IC: Spearman(mae_pred, mae_actual)
  - Ratio IC: Spearman(mfe_pred/mae_pred, mfe_actual/mae_actual)  [trade quality signal]
  - Entry accuracy: when mfe_pred > 2*mae_pred, what % of days had mfe_actual > mae_actual?
  - Entry PnL: simulated trade PnL when entering on mfe_pred > ratio_threshold * mae_pred

Walk-forward: expanding window (same as wider CNN, proven approach)
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.cuda.amp import GradScaler, autocast

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
sys.path.insert(0, str(Path(__file__).parent))
from book_spatial_cnn import SpatialResBlock, TemporalResBlock

# Thread limits
os.environ.setdefault('OMP_NUM_THREADS', '16')
os.environ.setdefault('MKL_NUM_THREADS', '16')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '16')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [mfe_mae] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('mfe_mae')


# =============================================================================
# Dataset
# =============================================================================

class MFEMAEDataset(torch.utils.data.Dataset):
    """
    Loads book tensor NPZ files and creates windowed samples.

    Each sample returns:
      book_window: (window_size, 20, 4) float32
      mfe: float32 — max favorable excursion in next `horizon` bars (ticks, >= 0)
      mae: float32 — max adverse excursion in next `horizon` bars (ticks, >= 0)
      direction: int  — 1=long, -1=short, 0=flat (sign of mid_return at horizon)

    MFE/MAE computation:
      - For each bar i, look at mid_prices[i+1 : i+horizon+1]
      - MFE = max(mid_prices[i+1:i+H+1] - mid_prices[i]) / 0.25 in ticks  [favorable if going long]
      - MAE = max(mid_prices[i] - mid_prices[i+1:i+H+1]) / 0.25 in ticks  [adverse if going long]
      Both are always >= 0. The model learns from the LONG perspective.
      Trade entry is symmetric: enter if mfe >> mae (direction = sign of net_return).
    """

    def __init__(
        self,
        npz_files: List[Path],
        window_size: int = 20,
        horizon: int = 20,
        augment: bool = False,
    ):
        self.window_size = window_size
        self.horizon = horizon
        self.augment = augment

        all_tensors = []
        all_mids = []
        day_boundaries = [0]

        for f in sorted(npz_files):
            data = np.load(f)
            tensors = data['book_tensors'].astype(np.float32, copy=False)
            mids = data['mid_prices'].astype(np.float64)
            all_tensors.append(tensors)
            all_mids.append(mids)
            day_boundaries.append(day_boundaries[-1] + len(mids))

        self.tensors = np.concatenate(all_tensors, axis=0)
        self.mids = np.concatenate(all_mids, axis=0)

        # Log-transform book features (same as all other models)
        self.tensors[:, :, 1] = np.log1p(self.tensors[:, :, 1])
        self.tensors[:, :, 2] = np.log1p(self.tensors[:, :, 2])
        self.tensors[:, :, 3] = np.log1p(self.tensors[:, :, 3])

        N = len(self.mids)
        tick = 0.25  # ES tick size

        # Pre-compute MFE and MAE for all bars — vectorized per day
        self._mfe_raw = np.full(N, np.nan, dtype=np.float32)
        self._mae_raw = np.full(N, np.nan, dtype=np.float32)
        self._net_return = np.full(N, np.nan, dtype=np.float32)

        for d in range(len(day_boundaries) - 1):
            start = day_boundaries[d]
            end = day_boundaries[d + 1]
            n_valid = end - start - horizon
            if n_valid <= 0:
                continue

            mids_day = self.mids[start:end]  # (day_len,)
            # Build a (n_valid, horizon) matrix of future mid prices
            # future_matrix[i, j] = mids_day[i + j + 1] for j in [0, horizon)
            idx = np.arange(n_valid)[:, None] + np.arange(1, horizon + 1)[None, :]  # (n_valid, H)
            future_mids_matrix = mids_day[idx]  # (n_valid, H)
            mid_now = mids_day[:n_valid, None]   # (n_valid, 1)

            delta = future_mids_matrix - mid_now  # (n_valid, H) — positive = up
            mfe = np.max(delta, axis=1) / tick    # (n_valid,)
            mae = np.max(-delta, axis=1) / tick   # (n_valid,)
            net = delta[:, -1]                    # (n_valid,) — net return at horizon

            self._mfe_raw[start:start + n_valid] = np.maximum(mfe, 0.0).astype(np.float32)
            self._mae_raw[start:start + n_valid] = np.maximum(mae, 0.0).astype(np.float32)
            self._net_return[start:start + n_valid] = (net / tick).astype(np.float32)

        # Valid sample indices (no cross-day windows, must have horizon)
        self.valid_indices = []
        for d in range(len(day_boundaries) - 1):
            start = day_boundaries[d]
            end = day_boundaries[d + 1]
            for i in range(start + window_size - 1, end - horizon):
                if np.isfinite(self._mfe_raw[i]) and np.isfinite(self._mae_raw[i]):
                    self.valid_indices.append(i)

        # Targets (will be set after normalization)
        self.mfe = self._mfe_raw.copy()
        self.mae = self._mae_raw.copy()

    def compute_normalization(self) -> Tuple[float, float, float, float]:
        """Return (mfe_mean, mfe_std, mae_mean, mae_std) of valid samples."""
        valid_indices = np.array(self.valid_indices)
        mfe_vals = self._mfe_raw[valid_indices]
        mae_vals = self._mae_raw[valid_indices]
        mfe_vals = mfe_vals[np.isfinite(mfe_vals)]
        mae_vals = mae_vals[np.isfinite(mae_vals)]
        return (
            float(mfe_vals.mean()), float(mfe_vals.std() + 1e-8),
            float(mae_vals.mean()), float(mae_vals.std() + 1e-8),
        )

    def set_normalization(self, mfe_mean, mfe_std, mae_mean, mae_std):
        """Z-score normalize MFE and MAE targets. Call after compute_normalization()."""
        self.mfe = (self._mfe_raw - mfe_mean) / mfe_std
        self.mae = (self._mae_raw - mae_mean) / mae_std

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        window = self.tensors[i - self.window_size + 1: i + 1]  # (W, 20, 4)

        if self.augment:
            window = window + np.random.normal(0, 0.01, window.shape).astype(np.float32)

        mfe_val = float(self.mfe[i]) if np.isfinite(self.mfe[i]) else 0.0
        mae_val = float(self.mae[i]) if np.isfinite(self.mae[i]) else 0.0
        net = float(self._net_return[i]) if np.isfinite(self._net_return[i]) else 0.0
        direction = int(np.sign(net))  # -1, 0, 1

        return (
            torch.from_numpy(window.copy()),           # (W, 20, 4)
            torch.tensor(mfe_val, dtype=torch.float32),
            torch.tensor(mae_val, dtype=torch.float32),
            torch.tensor(direction, dtype=torch.int8),
        )


# =============================================================================
# Model — Wider BookSpatialCNN backbone + two regression heads
# =============================================================================

class MFEMAEModel(nn.Module):
    """
    Wider BookSpatialCNN backbone (identical to the proven wider CNN WF run)
    with two pure-regression heads for MFE and MAE prediction.

    ~12.6M params. No classification head => no collapse risk.
    """

    def __init__(
        self,
        window_size: int = 20,
        num_levels: int = 20,
        num_features: int = 4,
        spatial_channels: Tuple[int, ...] = (64, 128, 256, 512),   # wider variant
        temporal_channels: int = 512,
        dropout: float = 0.15,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_levels = num_levels
        self.num_features = num_features

        # ---- Shared backbone (wider CNN, identical to Neptune WF run) ----
        self.spatial_stem = nn.Sequential(
            nn.Conv2d(1, spatial_channels[0], kernel_size=(3, 3), padding=(1, 1), bias=False),
            nn.BatchNorm2d(spatial_channels[0]),
            nn.GELU(),
        )
        spatial_res_layers = []
        for i in range(len(spatial_channels) - 1):
            spatial_res_layers.append(
                SpatialResBlock(spatial_channels[i], spatial_channels[i + 1], dropout=dropout * 0.5)
            )
        self.spatial_res_blocks = nn.Sequential(*spatial_res_layers)
        self.spatial_pool = nn.AdaptiveAvgPool2d((num_levels, 1))

        spatial_out_dim = spatial_channels[-1] * num_levels  # 512 * 20 = 10240
        self.spatial_compress = nn.Sequential(
            nn.Linear(spatial_out_dim, 512),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.bid_conv = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.ask_conv = nn.Sequential(
            nn.Conv1d(num_features, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        temporal_in = 512 + 128  # spatial_compress + bid + ask
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(temporal_in, temporal_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(temporal_channels),
            nn.GELU(),
        )
        self.temporal_res1 = TemporalResBlock(temporal_channels, kernel_size=5, dropout=dropout)
        self.temporal_res2 = TemporalResBlock(temporal_channels, kernel_size=3, dropout=dropout)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # ---- Two pure regression heads ----
        self.mfe_head = nn.Sequential(
            nn.Linear(temporal_channels, temporal_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_channels // 2, 1),
        )
        self.mae_head = nn.Sequential(
            nn.Linear(temporal_channels, temporal_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_channels // 2, 1),
        )

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        B, T, L, F = x.shape
        x_spatial = x.reshape(B * T, 1, L, F)
        x_spatial = self.spatial_stem(x_spatial)
        x_spatial = self.spatial_res_blocks(x_spatial)
        x_spatial = self.spatial_pool(x_spatial)
        x_spatial = x_spatial.reshape(B * T, -1)
        x_spatial = self.spatial_compress(x_spatial)
        x_spatial = x_spatial.reshape(B, T, -1)

        bid_in = x[:, :, :10, :].reshape(B * T, 10, F).permute(0, 2, 1)
        ask_in = x[:, :, 10:, :].reshape(B * T, 10, F).permute(0, 2, 1)
        bid_feats = self.bid_conv(bid_in).squeeze(-1).reshape(B, T, -1)
        ask_feats = self.ask_conv(ask_in).squeeze(-1).reshape(B, T, -1)

        combined = torch.cat([x_spatial, bid_feats, ask_feats], dim=-1)
        combined = combined.permute(0, 2, 1)

        out = self.temporal_stem(combined)
        out = self.temporal_res1(out)
        out = self.temporal_res2(out)
        out = self.temporal_pool(out).squeeze(-1)  # (B, temporal_channels)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns: mfe_pred (B,), mae_pred (B,)"""
        emb = self._backbone(x)
        mfe_pred = self.mfe_head(emb).squeeze(-1)
        mae_pred = self.mae_head(emb).squeeze(-1)
        return mfe_pred, mae_pred


# =============================================================================
# Training loop
# =============================================================================

def train_epoch(model, loader, optimizer, scaler, device):
    model.train()
    total_loss = total_n = 0
    for book_windows, mfe_targets, mae_targets, _ in loader:
        book_windows = book_windows.to(device)
        mfe_targets = mfe_targets.to(device)
        mae_targets = mae_targets.to(device)

        optimizer.zero_grad()
        with autocast():
            mfe_pred, mae_pred = model(book_windows)
            mfe_loss = F.mse_loss(mfe_pred, mfe_targets)
            mae_loss = F.mse_loss(mae_pred, mae_targets)
            loss = 0.5 * mfe_loss + 0.5 * mae_loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        n = book_windows.size(0)
        total_loss += loss.item() * n
        total_n += n

    return total_loss / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, device, ratio_threshold: float = 2.0):
    model.eval()
    all_mfe_pred = []
    all_mae_pred = []
    all_mfe_true = []
    all_mae_true = []
    all_direction = []
    total_loss = total_n = 0

    for book_windows, mfe_targets, mae_targets, directions in loader:
        book_windows = book_windows.to(device)
        mfe_targets_gpu = mfe_targets.to(device)
        mae_targets_gpu = mae_targets.to(device)

        with autocast():
            mfe_pred, mae_pred = model(book_windows)
            mfe_loss = F.mse_loss(mfe_pred, mfe_targets_gpu)
            mae_loss = F.mse_loss(mae_pred, mae_targets_gpu)
            loss = 0.5 * mfe_loss + 0.5 * mae_loss

        n = book_windows.size(0)
        total_loss += loss.item() * n
        total_n += n

        all_mfe_pred.append(mfe_pred.cpu().float().numpy())
        all_mae_pred.append(mae_pred.cpu().float().numpy())
        all_mfe_true.append(mfe_targets.numpy())
        all_mae_true.append(mae_targets.numpy())
        all_direction.append(directions.numpy())

    mfe_pred = np.concatenate(all_mfe_pred)
    mae_pred = np.concatenate(all_mae_pred)
    mfe_true = np.concatenate(all_mfe_true)
    mae_true = np.concatenate(all_mae_true)
    directions = np.concatenate(all_direction)

    val_loss = total_loss / max(total_n, 1)

    # --- Metrics ---
    # IC on MFE and MAE predictions individually
    mfe_ic, _ = spearmanr(mfe_pred, mfe_true)
    mae_ic, _ = spearmanr(mae_pred, mae_true)
    if np.isnan(mfe_ic): mfe_ic = 0.0
    if np.isnan(mae_ic): mae_ic = 0.0

    # Ratio IC: does predicted ratio correlate with actual ratio?
    pred_ratio = mfe_pred / (np.abs(mae_pred) + 1e-6)
    true_ratio = mfe_true / (np.abs(mae_true) + 1e-6)
    ratio_ic, _ = spearmanr(pred_ratio, true_ratio)
    if np.isnan(ratio_ic): ratio_ic = 0.0

    # Entry quality: when model says enter (mfe_pred > threshold * mae_pred)
    # what % of time was mfe_true actually > mae_true? (good entries)
    entry_mask = mfe_pred > ratio_threshold * np.abs(mae_pred)
    entry_n = int(entry_mask.sum())
    if entry_n > 0:
        entry_quality = float(np.mean(mfe_true[entry_mask] > mae_true[entry_mask]))
        # Simulated entry PnL: if mfe_true > mae_true => +mfe_true, else -mae_true
        pnl_per_trade = np.where(
            mfe_true[entry_mask] > mae_true[entry_mask],
            mfe_true[entry_mask],   # favorable outcome (z-scored ticks)
            -mae_true[entry_mask],  # unfavorable outcome
        )
        entry_mean_pnl = float(np.mean(pnl_per_trade))
        entry_sharpe = float(np.mean(pnl_per_trade) / (np.std(pnl_per_trade) + 1e-8))
    else:
        entry_quality = float('nan')
        entry_mean_pnl = float('nan')
        entry_sharpe = float('nan')

    entry_rate = float(entry_n / max(len(mfe_pred), 1))

    return {
        'val_loss': val_loss,
        'mfe_ic': float(mfe_ic),
        'mae_ic': float(mae_ic),
        'ratio_ic': float(ratio_ic),
        'entry_quality': entry_quality,
        'entry_rate': entry_rate,
        'entry_n': entry_n,
        'entry_mean_pnl': entry_mean_pnl,
        'entry_sharpe': entry_sharpe,
        'n_samples': len(mfe_pred),
    }


# =============================================================================
# Walk-forward runner
# =============================================================================

def find_npz_files(cache_dir: Path) -> List[Tuple]:
    """Return list of (date, path) sorted by date."""
    from datetime import datetime
    results = []
    for f in sorted(cache_dir.glob('*_book_tensors.npz')):
        try:
            date_str = f.name[:10]
            results.append((datetime.strptime(date_str, '%Y-%m-%d'), f))
        except ValueError:
            log.warning(f'Cannot parse date from {f.name}, skipping')
    results.sort(key=lambda x: x[0])
    return results


def run_walkforward(
    cache_dir: str,
    output_dir: str,
    min_train_days: int = 5,
    max_train_days: Optional[int] = None,  # None = expanding window
    epochs_per_fold: int = 2,
    batch_size: int = 256,
    lr: float = 5e-4,
    ratio_threshold: float = 2.0,
    window_size: int = 20,
    horizon: int = 20,
    subsample: int = 10,
    num_workers: int = 8,
    device: str = 'cuda',
    checkpoint_interval: int = 5,
):
    cache_dir = Path(cache_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_dates = find_npz_files(cache_dir)
    if not file_dates:
        log.error(f'No NPZ files found in {cache_dir}')
        return

    dates = [d for d, _ in file_dates]
    files = [f for _, f in file_dates]
    n_days = len(dates)
    total_folds = n_days - min_train_days

    log.info('=' * 70)
    log.info('MFE/MAE PREDICTION CNN — Expanding Window Walk-Forward')
    log.info(f'  Architecture: BookSpatialCNN wider (64,128,256,512), temporal=512, ~12.6M params')
    log.info(f'  Objective: MSE(MFE_pred, MFE_true) + MSE(MAE_pred, MAE_true)')
    log.info(f'  Entry signal: mfe_pred > {ratio_threshold} * mae_pred')
    log.info(f'  {n_days} days: {dates[0].date()} .. {dates[-1].date()}')
    log.info(f'  Total folds: {total_folds} | epochs/fold: {epochs_per_fold}')
    log.info(f'  Window mode: {"expanding" if max_train_days is None else f"sliding max {max_train_days}d"}')
    log.info('=' * 70)

    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device_obj}')

    # Checkpoint resume
    checkpoint_path = output_dir / 'mfe_mae_checkpoint.json'
    weights_path = output_dir / 'mfe_mae_latest.pt'
    start_fold = 0
    fold_results = []
    mfe_ic_sum = mae_ic_sum = ratio_ic_sum = 0.0
    ic_count = 0
    model = None

    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_fold = ckpt.get('completed_folds', 0)
        mfe_ic_sum = ckpt.get('mfe_ic_sum', 0.0)
        mae_ic_sum = ckpt.get('mae_ic_sum', 0.0)
        ratio_ic_sum = ckpt.get('ratio_ic_sum', 0.0)
        ic_count = ckpt.get('ic_count', 0)
        fold_results = ckpt.get('fold_results', [])
        log.info(f'CHECKPOINT RESUME: Skipping {start_fold} completed folds '
                 f'(MFE IC={mfe_ic_sum/max(ic_count,1):.4f}, '
                 f'Ratio IC={ratio_ic_sum/max(ic_count,1):.4f})')
        if weights_path.exists():
            model_state = torch.load(weights_path, map_location='cpu', weights_only=True)
            model = MFEMAEModel(window_size=window_size).to(device_obj)
            model.load_state_dict(model_state)
            log.info(f'Loaded weights from {weights_path}')

    for fold_idx in range(total_folds):
        if fold_idx < start_fold:
            continue

        test_day_idx = min_train_days + fold_idx
        test_date = dates[test_day_idx]
        test_file = [files[test_day_idx]]

        train_end = test_day_idx
        if max_train_days is not None:
            train_start = max(0, train_end - max_train_days)
        else:
            train_start = 0
        train_files = files[train_start:train_end]

        fold_num = fold_idx + 1
        log.info(
            f'\n--- Fold {fold_num}/{total_folds} | '
            f'Train: {dates[train_start].date()}..{dates[train_end-1].date()} '
            f'({train_end - train_start}d) | Test: {test_date.date()} ---'
        )

        # Build datasets
        train_ds = MFEMAEDataset(train_files, window_size=window_size, horizon=horizon, augment=False)
        norm = train_ds.compute_normalization()
        mfe_mean, mfe_std, mae_mean, mae_std = norm
        train_ds.set_normalization(*norm)
        log.info(f'  MFE norm: mean={mfe_mean:.3f} std={mfe_std:.3f} | MAE norm: mean={mae_mean:.3f} std={mae_std:.3f}')

        test_ds = MFEMAEDataset(test_file, window_size=window_size, horizon=horizon, augment=False)
        test_ds.set_normalization(*norm)

        # Subsample
        if subsample > 1 and len(train_ds) > subsample:
            indices = list(range(0, len(train_ds), subsample))
            train_ds_sub = torch.utils.data.Subset(train_ds, indices)
        else:
            train_ds_sub = train_ds

        log.info(f'  Train: {len(train_ds_sub)} samples (subsample={subsample}x) | Test: {len(test_ds)} samples')

        train_loader = torch.utils.data.DataLoader(
            train_ds_sub, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size * 2, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        # Build/reuse model
        if model is None:
            model = MFEMAEModel(window_size=window_size).to(device_obj)
            log.info(f'  Model params: {sum(p.numel() for p in model.parameters()):,}')
        else:
            log.info(f'  Model params: {sum(p.numel() for p in model.parameters()):,} (warm start)')

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = GradScaler(init_scale=1024)

        best_ratio_ic = -np.inf
        best_state = None
        best_metrics = None

        for ep in range(epochs_per_fold):
            t0 = time.time()
            train_loss = train_epoch(model, train_loader, optimizer, scaler, device_obj)
            metrics = evaluate(model, test_loader, device_obj, ratio_threshold=ratio_threshold)
            elapsed = time.time() - t0

            tag = ''
            if metrics['ratio_ic'] > best_ratio_ic:
                best_ratio_ic = metrics['ratio_ic']
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                best_metrics = metrics
                tag = ' *BEST*'

            log.info(
                f'  Epoch {ep+1}/{epochs_per_fold}: '
                f'train_loss={train_loss:.5f} '
                f'val_loss={metrics["val_loss"]:.5f} '
                f'MFE_IC={metrics["mfe_ic"]:+.4f} '
                f'MAE_IC={metrics["mae_ic"]:+.4f} '
                f'Ratio_IC={metrics["ratio_ic"]:+.4f} '
                f'EntryQ={metrics["entry_quality"]:.3f} ({metrics["entry_rate"]*100:.1f}% entries) '
                f'({elapsed:.1f}s){tag}'
            )

        # Restore best
        if best_state is not None:
            model.load_state_dict(best_state)
        if best_metrics is None:
            best_metrics = metrics

        fold_result = {
            'fold': fold_num,
            'test_date': str(test_date.date()),
            'mfe_ic': best_metrics['mfe_ic'],
            'mae_ic': best_metrics['mae_ic'],
            'ratio_ic': best_metrics['ratio_ic'],
            'entry_quality': best_metrics['entry_quality'],
            'entry_rate': best_metrics['entry_rate'],
            'entry_mean_pnl': best_metrics['entry_mean_pnl'],
            'entry_sharpe': best_metrics['entry_sharpe'],
            'train_days': train_end - train_start,
        }
        fold_results.append(fold_result)
        mfe_ic_sum += best_metrics['mfe_ic']
        mae_ic_sum += best_metrics['mae_ic']
        ratio_ic_sum += best_metrics['ratio_ic']
        ic_count += 1

        log.info(
            f'  Fold {fold_num} | MFE_IC={best_metrics["mfe_ic"]:+.4f} '
            f'MAE_IC={best_metrics["mae_ic"]:+.4f} '
            f'Ratio_IC={best_metrics["ratio_ic"]:+.4f} | '
            f'Mean Ratio_IC={ratio_ic_sum/ic_count:+.4f} ({ic_count} folds)'
        )

        # Checkpoint
        if fold_num % checkpoint_interval == 0 or fold_num == total_folds:
            torch.save(model.state_dict(), weights_path)
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'completed_folds': fold_num,
                    'mfe_ic_sum': mfe_ic_sum,
                    'mae_ic_sum': mae_ic_sum,
                    'ratio_ic_sum': ratio_ic_sum,
                    'ic_count': ic_count,
                    'fold_results': fold_results,
                }, f, indent=2)
            log.info(f'  Checkpoint saved at fold {fold_num}')

    # Final summary
    if ic_count > 0:
        mean_mfe_ic = mfe_ic_sum / ic_count
        mean_mae_ic = mae_ic_sum / ic_count
        mean_ratio_ic = ratio_ic_sum / ic_count
        pos_ratio_ic = sum(1 for r in fold_results if r['ratio_ic'] > 0)
        valid_eq = [r['entry_quality'] for r in fold_results if r['entry_quality'] is not None and not (isinstance(r['entry_quality'], float) and np.isnan(r['entry_quality']))]
        mean_eq = float(np.mean(valid_eq)) if valid_eq else float('nan')

        log.info('\n' + '=' * 70)
        log.info('MFE/MAE CNN WALK-FORWARD COMPLETE')
        log.info(f'  Folds:              {ic_count}')
        log.info(f'  Mean MFE IC:        {mean_mfe_ic:+.4f}')
        log.info(f'  Mean MAE IC:        {mean_mae_ic:+.4f}')
        log.info(f'  Mean Ratio IC:      {mean_ratio_ic:+.4f}')
        log.info(f'  Pos Ratio IC folds: {pos_ratio_ic}/{ic_count} ({100*pos_ratio_ic/ic_count:.0f}%)')
        log.info(f'  Mean Entry Quality: {mean_eq:.3f}')
        log.info('=' * 70)

        final = {
            'summary': {
                'mean_mfe_ic': mean_mfe_ic,
                'mean_mae_ic': mean_mae_ic,
                'mean_ratio_ic': mean_ratio_ic,
                'pos_ratio_ic_folds': f'{pos_ratio_ic}/{ic_count}',
                'mean_entry_quality': mean_eq,
                'total_folds': ic_count,
            },
            'folds': fold_results,
        }
        with open(output_dir / 'mfe_mae_wf_results.json', 'w') as f:
            json.dump(final, f, indent=2)
        log.info(f'Results saved to {output_dir / "mfe_mae_wf_results.json"}')


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MFE/MAE Prediction CNN — Walk-Forward')
    parser.add_argument('--data_dir', required=True, help='Path to book tensor NPZ cache')
    parser.add_argument('--output_dir', default='results/mfe_mae_wf', help='Output directory')
    parser.add_argument('--min_train_days', type=int, default=5)
    parser.add_argument('--max_train_days', type=int, default=None, help='None=expanding window')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--ratio_threshold', type=float, default=2.0, help='MFE/MAE ratio for entry signal')
    parser.add_argument('--horizon', type=int, default=20, help='Bars forward for MFE/MAE computation')
    parser.add_argument('--subsample', type=int, default=10)
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--checkpoint_interval', type=int, default=5)
    args = parser.parse_args()

    run_walkforward(
        cache_dir=args.data_dir,
        output_dir=args.output_dir,
        min_train_days=args.min_train_days,
        max_train_days=args.max_train_days,
        epochs_per_fold=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        ratio_threshold=args.ratio_threshold,
        horizon=args.horizon,
        subsample=args.subsample,
        num_workers=args.num_workers,
        device=args.device,
        checkpoint_interval=args.checkpoint_interval,
    )
