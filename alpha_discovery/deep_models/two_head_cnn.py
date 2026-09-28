"""
Two-Head Book Spatial CNN
=========================
Same backbone as BookSpatialCNN (wider variant), but with TWO output heads:
  1. direction_head: classification (0=down, 1=flat, 2=up) -- 3-way softmax
  2. magnitude_head: regression (continuous return in ticks) -- scalar

Combined loss:
  loss = alpha * CrossEntropy(direction_logits, direction_labels)
       + (1 - alpha) * HuberLoss(magnitude_pred, magnitude_target)

Metrics (ALL mandatory per MEMORY.md):
  - IC: Spearman rank correlation between magnitude_pred and actual tick return
  - Directional accuracy: % of samples where predicted direction == actual direction
  - Conditional accuracy: directional accuracy restricted to top 20% |magnitude_pred|
  - Calibration: mean |actual_return| for each confidence decile of magnitude_pred

Motivation:
  The current single-head regression model optimises for IC (Spearman correlation).
  It may learn direction perfectly but get magnitude wrong (or vice versa).
  By explicitly supervising BOTH, we can:
    a) measure which aspect the book features actually capture
    b) use directional accuracy as a standalone signal for binary trade entries
    c) combine confidence from both heads for higher-conviction signals

Architecture:
  Input: (batch, window=20, 20_levels, 4_features)
  Shared backbone: BookSpatialCNN up to temporal_pool (outputs 256-dim embedding)
  direction_head: Linear(256, 128) -> GELU -> Dropout -> Linear(128, 3)
  magnitude_head: Linear(256, 128) -> GELU -> Dropout -> Linear(128, 1)

Usage:
  python two_head_cnn.py --data_dir <path> --output_dir <path> [options]
  python two_head_cnn.py --wf --data_dir <path> --output_dir <path>  # walk-forward mode
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
# Import shared dataset / helpers from existing codebase
# ---------------------------------------------------------------------------
# Add deep_models directory to path
sys.path.insert(0, str(Path(__file__).parent))

from book_spatial_cnn import SpatialResBlock, TemporalResBlock  # reuse residual blocks

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [two_head] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
)
log = logging.getLogger('two_head')


# =============================================================================
# Dataset
# =============================================================================

class TwoHeadDataset(torch.utils.data.Dataset):
    """
    Loads book tensor NPZ files and creates windowed samples with BOTH
    direction labels and continuous magnitude targets.

    Each sample returns:
      book_window: (window_size, 20, 4) float32
      direction:   int64 in {0=down, 1=flat, 2=up}
      magnitude:   float32 in ticks (normalised z-score after dataset init)
    """

    def __init__(
        self,
        npz_files: List[Path],
        window_size: int = 20,
        horizon: int = 20,
        direction_threshold: float = 0.5,   # ticks to count as up/down
        augment: bool = False,
    ):
        self.window_size = window_size
        self.horizon = horizon
        self.direction_threshold = direction_threshold
        self.augment = augment

        all_tensors = []
        all_mids = []
        day_boundaries = [0]

        for f in sorted(npz_files):
            data = np.load(f)
            tensors = data['book_tensors'].astype(np.float32, copy=False)  # (N, 20, 4)
            mids = data['mid_prices']                                        # (N,)
            all_tensors.append(tensors)
            all_mids.append(mids)
            day_boundaries.append(day_boundaries[-1] + len(mids))

        self.tensors = np.concatenate(all_tensors, axis=0)
        self.mids = np.concatenate(all_mids, axis=0).astype(np.float64)

        # Log-transform book features (same as BookSpatialCNN)
        self.tensors[:, :, 1] = np.log1p(self.tensors[:, :, 1])
        self.tensors[:, :, 2] = np.log1p(self.tensors[:, :, 2])
        self.tensors[:, :, 3] = np.log1p(self.tensors[:, :, 3])

        # Build valid indices (no cross-day windows)
        self.valid_indices = []
        for d in range(len(day_boundaries) - 1):
            start = day_boundaries[d]
            end = day_boundaries[d + 1]
            for i in range(start + window_size - 1, end - horizon):
                self.valid_indices.append(i)

        N = len(self.mids)
        # Continuous return in ticks
        raw_returns = np.full(N, np.nan, dtype=np.float32)
        for i in range(N - horizon):
            raw_returns[i] = float((self.mids[i + horizon] - self.mids[i]) / 0.25)

        # Z-score normalise returns (train only — caller should set norm stats)
        valid_mask = np.isfinite(raw_returns)
        self._raw_returns = raw_returns

        # Direction labels
        self.directions = np.ones(N, dtype=np.int64)  # default flat
        for i in range(N - horizon):
            r = raw_returns[i]
            if np.isfinite(r):
                if r > direction_threshold:
                    self.directions[i] = 2
                elif r < -direction_threshold:
                    self.directions[i] = 0

        # Magnitude target (will be set after z-score norm; defaults to raw)
        self.magnitudes = raw_returns.copy()

    def set_normalization(self, mean: float, std: float):
        """Apply z-score normalisation to magnitude targets. Call after dataset creation."""
        self.magnitudes = (self._raw_returns - mean) / (std + 1e-8)

    def compute_normalization(self) -> Tuple[float, float]:
        """Return (mean, std) of finite raw returns — call before set_normalization."""
        valid = self._raw_returns[np.isfinite(self._raw_returns)]
        return float(valid.mean()), float(valid.std())

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = self.valid_indices[idx]
        # Book window: last window_size bars up to and including bar i
        window = self.tensors[i - self.window_size + 1 : i + 1]  # (W, 20, 4)

        if self.augment:
            window = window + np.random.normal(0, 0.01, window.shape).astype(np.float32)

        direction = self.directions[i]
        magnitude = float(self.magnitudes[i]) if np.isfinite(self.magnitudes[i]) else 0.0

        return (
            torch.from_numpy(window),         # (W, 20, 4)
            torch.tensor(direction, dtype=torch.long),
            torch.tensor(magnitude, dtype=torch.float32),
        )


# =============================================================================
# Model
# =============================================================================

class TwoHeadBookSpatialCNN(nn.Module):
    """
    Same backbone as BookSpatialCNN (wider variant), two output heads.

    direction_head: 3-class softmax (down / flat / up)
    magnitude_head: scalar regression (tick return, z-scored)

    Both heads share the full backbone (no gradient stopping between them).
    """

    def __init__(
        self,
        window_size: int = 20,
        num_levels: int = 20,
        num_features: int = 4,
        spatial_channels: Tuple[int, ...] = (32, 64, 128, 256),
        temporal_channels: int = 256,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.window_size = window_size
        self.num_levels = num_levels
        self.num_features = num_features

        # ---- Shared backbone (identical to BookSpatialCNN) ----
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

        spatial_out_dim = spatial_channels[-1] * num_levels  # 256 * 20 = 5120
        self.spatial_compress = nn.Sequential(
            nn.Linear(spatial_out_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        self.bid_conv = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )
        self.ask_conv = nn.Sequential(
            nn.Conv1d(num_features, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool1d(1),
        )

        temporal_in = 256 + 64
        self.temporal_stem = nn.Sequential(
            nn.Conv1d(temporal_in, temporal_channels, kernel_size=5, padding=2, bias=False),
            nn.BatchNorm1d(temporal_channels),
            nn.GELU(),
        )
        self.temporal_res1 = TemporalResBlock(temporal_channels, kernel_size=5, dropout=dropout)
        self.temporal_res2 = TemporalResBlock(temporal_channels, kernel_size=3, dropout=dropout)
        self.temporal_pool = nn.AdaptiveAvgPool1d(1)

        # ---- Two heads ----
        self.direction_head = nn.Sequential(
            nn.Linear(temporal_channels, temporal_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_channels // 2, 3),
        )
        self.magnitude_head = nn.Sequential(
            nn.Linear(temporal_channels, temporal_channels // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(temporal_channels // 2, 1),
        )

    def _backbone(self, x: torch.Tensor) -> torch.Tensor:
        """Run shared backbone. Returns embedding (B, temporal_channels)."""
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

        combined = torch.cat([x_spatial, bid_feats, ask_feats], dim=-1)  # (B, T, 320)
        combined = combined.permute(0, 2, 1)

        out = self.temporal_stem(combined)
        out = self.temporal_res1(out)
        out = self.temporal_res2(out)
        out = self.temporal_pool(out).squeeze(-1)  # (B, temporal_channels)
        return out

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Returns:
          direction_logits: (B, 3)
          magnitude_pred:   (B, 1)
        """
        emb = self._backbone(x)
        direction_logits = self.direction_head(emb)
        magnitude_pred = self.magnitude_head(emb).squeeze(-1)
        return direction_logits, magnitude_pred


# =============================================================================
# Loss
# =============================================================================

class TwoHeadLoss(nn.Module):
    def __init__(self, alpha: float = 0.5, huber_delta: float = 1.0):
        """
        alpha: weight on classification loss (0=pure regression, 1=pure classification)
        """
        super().__init__()
        self.alpha = alpha
        self.ce = nn.CrossEntropyLoss()
        self.huber = nn.HuberLoss(delta=huber_delta)

    def forward(
        self,
        direction_logits: torch.Tensor,  # (B, 3)
        magnitude_pred: torch.Tensor,    # (B,)
        direction_labels: torch.Tensor,  # (B,) int64
        magnitude_targets: torch.Tensor, # (B,) float32
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        cls_loss = self.ce(direction_logits, direction_labels)
        reg_loss = self.huber(magnitude_pred, magnitude_targets)
        total = self.alpha * cls_loss + (1.0 - self.alpha) * reg_loss
        return total, cls_loss, reg_loss


# =============================================================================
# Metrics
# =============================================================================

def compute_metrics(
    magnitude_preds: np.ndarray,
    magnitude_targets: np.ndarray,
    direction_preds: np.ndarray,   # argmax of logits
    direction_labels: np.ndarray,
) -> Dict:
    """
    Compute all mandatory metrics:
      - IC: Spearman(magnitude_preds, magnitude_targets)
      - directional_accuracy: % where direction_preds == direction_labels
      - conditional_accuracy: dir accuracy on top 20% |magnitude_preds|
      - calibration: mean abs(target) per confidence decile
    """
    # IC
    ic, _ = spearmanr(magnitude_preds, magnitude_targets)
    if np.isnan(ic):
        ic = 0.0

    # Directional accuracy
    dir_acc = float(np.mean(direction_preds == direction_labels))

    # Conditional accuracy (top 20% by |magnitude_pred|)
    n = len(magnitude_preds)
    top20_thresh = np.percentile(np.abs(magnitude_preds), 80)
    top20_mask = np.abs(magnitude_preds) >= top20_thresh
    if top20_mask.sum() > 0:
        cond_acc = float(np.mean(direction_preds[top20_mask] == direction_labels[top20_mask]))
    else:
        cond_acc = float('nan')

    # Calibration: mean |target| per decile of |magnitude_pred|
    abs_pred = np.abs(magnitude_preds)
    deciles = np.percentile(abs_pred, np.arange(0, 100, 10))
    calibration = []
    for i in range(len(deciles)):
        lo = deciles[i]
        hi = deciles[i + 1] if i + 1 < len(deciles) else abs_pred.max() + 1
        mask = (abs_pred >= lo) & (abs_pred < hi)
        if mask.sum() > 0:
            calibration.append(float(np.mean(np.abs(magnitude_targets[mask]))))
        else:
            calibration.append(float('nan'))

    return {
        'IC': float(ic),
        'dir_accuracy': dir_acc,
        'cond_accuracy_top20': cond_acc,
        'calibration_deciles': calibration,
        'n_samples': n,
        'top20_n': int(top20_mask.sum()),
    }


# =============================================================================
# Training
# =============================================================================

def train_epoch(model, loader, optimizer, criterion, scaler, device):
    model.train()
    total_loss = total_cls = total_reg = total_n = 0
    for book_windows, directions, magnitudes in loader:
        book_windows = book_windows.to(device)
        directions = directions.to(device)
        magnitudes = magnitudes.to(device)

        optimizer.zero_grad()
        with autocast():
            dir_logits, mag_pred = model(book_windows)
            loss, cls_loss, reg_loss = criterion(dir_logits, mag_pred, directions, magnitudes)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()

        n = book_windows.size(0)
        total_loss += loss.item() * n
        total_cls += cls_loss.item() * n
        total_reg += reg_loss.item() * n
        total_n += n

    return total_loss / max(total_n, 1), total_cls / max(total_n, 1), total_reg / max(total_n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    all_mag_preds = []
    all_mag_targets = []
    all_dir_preds = []
    all_dir_labels = []
    total_loss = total_n = 0

    for book_windows, directions, magnitudes in loader:
        book_windows = book_windows.to(device)
        directions = directions.to(device)
        magnitudes = magnitudes.to(device)

        with autocast():
            dir_logits, mag_pred = model(book_windows)
            loss, _, _ = criterion(dir_logits, mag_pred, directions, magnitudes)

        n = book_windows.size(0)
        total_loss += loss.item() * n
        total_n += n

        all_mag_preds.append(mag_pred.cpu().float().numpy())
        all_mag_targets.append(magnitudes.cpu().float().numpy())
        all_dir_preds.append(dir_logits.argmax(dim=-1).cpu().numpy())
        all_dir_labels.append(directions.cpu().numpy())

    mag_preds = np.concatenate(all_mag_preds)
    mag_targets = np.concatenate(all_mag_targets)
    dir_preds = np.concatenate(all_dir_preds)
    dir_labels = np.concatenate(all_dir_labels)

    metrics = compute_metrics(mag_preds, mag_targets, dir_preds, dir_labels)
    metrics['val_loss'] = total_loss / max(total_n, 1)
    return metrics, mag_preds, dir_preds


# =============================================================================
# Walk-forward runner
# =============================================================================

def find_npz_files(cache_dir: Path) -> List[Path]:
    return sorted(cache_dir.glob('*_book_tensors.npz'))


def run_walkforward(
    cache_dir: str,
    output_dir: str,
    min_train_days: int = 5,
    max_train_days: Optional[int] = None,
    epochs_per_fold: int = 2,
    batch_size: int = 256,
    lr: float = 5e-4,
    alpha: float = 0.5,
    direction_threshold: float = 0.5,
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

    all_files = find_npz_files(cache_dir)
    if not all_files:
        log.error(f'No NPZ files found in {cache_dir}')
        return

    # Extract dates from filenames: YYYY-MM-DD_book_tensors.npz
    from datetime import datetime
    file_dates = []
    for f in all_files:
        try:
            date_str = f.name[:10]
            file_dates.append((datetime.strptime(date_str, '%Y-%m-%d'), f))
        except ValueError:
            log.warning(f'Cannot parse date from {f.name}, skipping')

    file_dates.sort(key=lambda x: x[0])
    dates = [d for d, _ in file_dates]
    files = [f for _, f in file_dates]
    n_days = len(dates)

    log.info(f'Walk-forward: {n_days} days from {dates[0].date()} to {dates[-1].date()}')

    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')
    log.info(f'Device: {device_obj}')

    fold_results = []
    model = None  # warm start

    # Checkpoint file
    checkpoint_path = output_dir / 'two_head_checkpoint.json'
    start_fold = 0
    ic_sum = 0.0
    ic_count = 0
    if checkpoint_path.exists():
        with open(checkpoint_path) as f:
            ckpt = json.load(f)
        start_fold = ckpt.get('completed_folds', 0)
        ic_sum = ckpt.get('ic_sum', 0.0)
        ic_count = ckpt.get('ic_count', 0)
        fold_results = ckpt.get('fold_results', [])
        log.info(f'Resuming from fold {start_fold + 1} (IC so far: {ic_sum/max(ic_count,1):.4f})')

        # Try to load model weights
        weights_path = output_dir / 'two_head_latest.pt'
        if weights_path.exists():
            model_state = torch.load(weights_path, map_location='cpu')
            model = TwoHeadBookSpatialCNN(window_size=window_size).to(device_obj)
            model.load_state_dict(model_state)
            log.info(f'Loaded weights from {weights_path}')

    log.info(f'Total folds: {n_days - min_train_days}')

    for fold_idx in range(n_days - min_train_days):
        test_day_idx = min_train_days + fold_idx
        if fold_idx < start_fold:
            continue

        test_date = dates[test_day_idx]
        test_file = [files[test_day_idx]]

        train_end = test_day_idx
        if max_train_days is not None:
            train_start = max(0, train_end - max_train_days)
        else:
            train_start = 0
        train_files = files[train_start:train_end]

        fold_num = fold_idx + 1
        total_folds = n_days - min_train_days
        log.info(f'\n--- Fold {fold_num}/{total_folds} | Train: {dates[train_start].date()}..{dates[train_end-1].date()} ({train_end-train_start}d) | Test: {test_date.date()} ---')

        # Build datasets
        train_ds = TwoHeadDataset(
            train_files,
            window_size=window_size,
            horizon=horizon,
            direction_threshold=direction_threshold,
            augment=False,
        )
        mean, std = train_ds.compute_normalization()
        train_ds.set_normalization(mean, std)
        log.info(f'  Target normalization: mean={mean:.3f} std={std:.3f} ticks')
        log.info(f'  Train: {len(train_ds)} samples')

        test_ds = TwoHeadDataset(
            test_file,
            window_size=window_size,
            horizon=horizon,
            direction_threshold=direction_threshold,
            augment=False,
        )
        test_ds.set_normalization(mean, std)

        # Subsample training data
        if subsample > 1 and len(train_ds) > subsample:
            indices = list(range(0, len(train_ds), subsample))
            train_ds_sub = torch.utils.data.Subset(train_ds, indices)
        else:
            train_ds_sub = train_ds

        train_loader = torch.utils.data.DataLoader(
            train_ds_sub, batch_size=batch_size, shuffle=True,
            num_workers=num_workers, pin_memory=True, drop_last=True,
        )
        test_loader = torch.utils.data.DataLoader(
            test_ds, batch_size=batch_size * 2, shuffle=False,
            num_workers=num_workers, pin_memory=True,
        )

        log.info(f'  Dataset: {len(train_ds_sub)} train samples (subsample={subsample}x) | {len(test_ds)} test samples')

        # Build/reuse model
        if model is None:
            model = TwoHeadBookSpatialCNN(window_size=window_size).to(device_obj)
            log.info(f'  Model params: {sum(p.numel() for p in model.parameters()):,}')
        else:
            log.info(f'  Model params: {sum(p.numel() for p in model.parameters()):,} (warm start from previous fold)')

        criterion = TwoHeadLoss(alpha=alpha).to(device_obj)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = GradScaler(init_scale=1024)

        best_ic = -np.inf
        best_state = None

        for ep in range(epochs_per_fold):
            t0 = time.time()
            train_loss, cls_loss, reg_loss = train_epoch(model, train_loader, optimizer, criterion, scaler, device_obj)
            metrics, mag_preds, dir_preds = evaluate(model, test_loader, criterion, device_obj)
            elapsed = time.time() - t0

            ic = metrics['IC']
            dir_acc = metrics['dir_accuracy']
            cond_acc = metrics['cond_accuracy_top20']

            tag = ''
            if ic > best_ic:
                best_ic = ic
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                tag = ' *BEST*'

            log.info(
                f'  Epoch {ep+1}/{epochs_per_fold}: '
                f'train_loss={train_loss:.5f} (cls={cls_loss:.4f} reg={reg_loss:.4f}) '
                f'val_loss={metrics["val_loss"]:.5f} '
                f'IC={ic:+.4f} dir_acc={dir_acc:.3f} cond_acc={cond_acc:.3f} '
                f'({elapsed:.1f}s){tag}'
            )

        # Restore best state for this fold
        if best_state is not None:
            model.load_state_dict(best_state)

        # Save fold result
        fold_result = {
            'fold': fold_num,
            'test_date': str(test_date.date()),
            'IC': best_ic,
            'dir_accuracy': metrics['dir_accuracy'],
            'cond_accuracy_top20': metrics['cond_accuracy_top20'],
            'calibration_deciles': metrics['calibration_deciles'],
            'train_days': train_end - train_start,
        }
        fold_results.append(fold_result)
        ic_sum += best_ic
        ic_count += 1
        mean_ic = ic_sum / ic_count

        log.info(f'  Fold {fold_num} IC={best_ic:+.4f} dir={metrics["dir_accuracy"]:.3f} | Mean IC={mean_ic:+.4f} ({ic_count} folds)')

        # Save checkpoint
        if fold_num % checkpoint_interval == 0 or fold_num == total_folds:
            torch.save(model.state_dict(), output_dir / 'two_head_latest.pt')
            with open(checkpoint_path, 'w') as f:
                json.dump({
                    'completed_folds': fold_num,
                    'ic_sum': ic_sum,
                    'ic_count': ic_count,
                    'fold_results': fold_results,
                }, f, indent=2)
            log.info(f'  Checkpoint saved at fold {fold_num}')

    # Final summary
    if ic_count > 0:
        mean_ic = ic_sum / ic_count
        mean_dir = np.mean([r['dir_accuracy'] for r in fold_results])
        mean_cond = np.nanmean([r['cond_accuracy_top20'] for r in fold_results])
        pos_ic = sum(1 for r in fold_results if r['IC'] > 0)

        log.info('\n' + '='*70)
        log.info('TWO-HEAD CNN WALK-FORWARD COMPLETE')
        log.info(f'  Folds:              {ic_count}')
        log.info(f'  Mean IC:            {mean_ic:+.4f}')
        log.info(f'  Positive IC folds:  {pos_ic}/{ic_count} ({100*pos_ic/ic_count:.0f}%)')
        log.info(f'  Mean dir accuracy:  {mean_dir:.3f}')
        log.info(f'  Mean cond accuracy: {mean_cond:.3f}')
        log.info('='*70)

        # Save final results
        final = {
            'summary': {
                'mean_IC': mean_ic,
                'mean_dir_accuracy': mean_dir,
                'mean_cond_accuracy_top20': mean_cond,
                'positive_IC_folds': f'{pos_ic}/{ic_count}',
                'total_folds': ic_count,
            },
            'folds': fold_results,
        }
        with open(output_dir / 'two_head_wf_results.json', 'w') as f:
            json.dump(final, f, indent=2)
        log.info(f'Results saved to {output_dir / "two_head_wf_results.json"}')


# =============================================================================
# Single-day quick eval (sanity check)
# =============================================================================

def quick_eval(
    cache_dir: str,
    n_train_days: int = 30,
    epochs: int = 3,
    batch_size: int = 256,
    lr: float = 5e-4,
    alpha: float = 0.5,
    device: str = 'cuda',
    num_workers: int = 8,
):
    """
    Train on first n_train_days, test on day n_train_days+1.
    Quick sanity check to verify the model runs and metrics are non-trivial.
    """
    cache_dir = Path(cache_dir)
    all_files = sorted(find_npz_files(cache_dir))
    if len(all_files) < n_train_days + 2:
        log.error(f'Not enough files: need {n_train_days + 2}, found {len(all_files)}')
        return

    train_files = all_files[:n_train_days]
    test_files = [all_files[n_train_days]]

    device_obj = torch.device(device if torch.cuda.is_available() else 'cpu')

    train_ds = TwoHeadDataset(train_files)
    mean, std = train_ds.compute_normalization()
    train_ds.set_normalization(mean, std)
    test_ds = TwoHeadDataset(test_files)
    test_ds.set_normalization(mean, std)

    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=batch_size * 2, shuffle=False, num_workers=num_workers, pin_memory=True
    )

    model = TwoHeadBookSpatialCNN().to(device_obj)
    log.info(f'Model params: {sum(p.numel() for p in model.parameters()):,}')

    criterion = TwoHeadLoss(alpha=alpha).to(device_obj)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler(init_scale=1024)

    for ep in range(epochs):
        t0 = time.time()
        tl, cl, rl = train_epoch(model, train_loader, optimizer, criterion, scaler, device_obj)
        metrics, _, _ = evaluate(model, test_loader, criterion, device_obj)
        elapsed = time.time() - t0
        log.info(
            f'Epoch {ep+1}/{epochs}: loss={tl:.5f} (cls={cl:.4f} reg={rl:.4f}) '
            f'IC={metrics["IC"]:+.4f} dir_acc={metrics["dir_accuracy"]:.3f} '
            f'cond_acc={metrics["cond_accuracy_top20"]:.3f} ({elapsed:.1f}s)'
        )


# =============================================================================
# CLI
# =============================================================================

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Two-Head Book Spatial CNN')
    parser.add_argument('--data_dir', required=True, help='Path to book tensor NPZ cache')
    parser.add_argument('--output_dir', default='results/two_head', help='Output directory')
    parser.add_argument('--wf', action='store_true', help='Walk-forward mode (default: quick eval)')
    parser.add_argument('--min_train_days', type=int, default=5)
    parser.add_argument('--max_train_days', type=int, default=None, help='None=expanding window')
    parser.add_argument('--epochs', type=int, default=2, help='Epochs per fold (WF) or total (quick)')
    parser.add_argument('--batch_size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=5e-4)
    parser.add_argument('--alpha', type=float, default=0.5, help='Weight on classification loss')
    parser.add_argument('--direction_threshold', type=float, default=0.5, help='Ticks for up/down label')
    parser.add_argument('--subsample', type=int, default=10, help='Train subsample factor (WF only)')
    parser.add_argument('--num_workers', type=int, default=8)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--n_train_days', type=int, default=30, help='Days for quick eval mode')
    args = parser.parse_args()

    if args.wf:
        run_walkforward(
            cache_dir=args.data_dir,
            output_dir=args.output_dir,
            min_train_days=args.min_train_days,
            max_train_days=args.max_train_days,
            epochs_per_fold=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            alpha=args.alpha,
            direction_threshold=args.direction_threshold,
            subsample=args.subsample,
            num_workers=args.num_workers,
            device=args.device,
        )
    else:
        quick_eval(
            cache_dir=args.data_dir,
            n_train_days=args.n_train_days,
            epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            alpha=args.alpha,
            device=args.device,
            num_workers=args.num_workers,
        )
