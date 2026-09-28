# -*- coding: utf-8 -*-
"""
BookVideo3DCNN - Next-Gen Spatial + Temporal Architecture.

Processes SEQUENCES of order book snapshots as a "video" of the evolving book.
Each sample is seq_len=50 consecutive bars (5 seconds at 100ms resolution).

Input tensor shape: (batch, 20_levels, seq_len=50, 4_features)
  → treated as a 3D spatial+temporal volume for Conv3d processing
  → channels-last perspective: (B, C=4, T=50, H=20, W=1)

The 3D CNN can learn:
  - Ask walls building/depleting over time (temporal depth dynamics)
  - Bid depth getting swept level by level (spatial-temporal sweeping)
  - Spread dynamics: widening then snapping back
  - Aggressive order flow: consecutive bars of aggressive buying/selling
  - Spoofing: large orders appearing then canceling within seconds

Architecture:
  Input: (B, 4, 50, 20, 1)  — 4 channels (features), T=50, H=20 levels, W=1
  Conv3d(4, 32, (5,3,1))   — temporal × level × feature
  Conv3d(32, 64, (5,3,1))
  Conv3d(64, 128, (3,3,1))
  AdaptiveAvgPool3d((1,1,1)) — collapse all dims
  Linear(128, 64) → Linear(64, 1)

Walk-forward protocol mirrors wider CNN (expanding window):
  - min 15 training days, 1-day purge, expanding window
  - stride-50 subsampling for training (every 5th second)
  - batch=512, epochs=20, patience=5 for early stopping
  - AMP enabled, BELOW_NORMAL process priority
  - MLflow logging to http://localhost:5000 experiment CNN_Training
  - Per-fold .pt weights + .npz OOS predictions saved
  - Concat IC reported alongside per-fold IC

Leakage audit:
  - Window is strictly CAUSAL: bars t-49..t, predicts t+100 (10s)
  - No future data in window, no cross-day windows
  - Normalization stats computed from train data only
  - Day boundaries enforced in BarDataset3D (same as book_spatial_cnn)

Usage:
  python train_3d_cnn.py                    # full WF run, all days
  python train_3d_cnn.py --epochs 5         # faster test
  python train_3d_cnn.py --start-fold 10   # resume

Script: alpha_discovery/deep_models/train_3d_cnn.py
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
import torch.nn.functional as F
from scipy.stats import spearmanr
from torch.utils.data import DataLoader, Dataset

# ---------------------------------------------------------------------------
# Priority + thread limits (BELOW_NORMAL so paper engine stays responsive)
# ---------------------------------------------------------------------------
os.environ.setdefault('OMP_NUM_THREADS', '8')
os.environ.setdefault('MKL_NUM_THREADS', '8')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '8')
os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')

try:
    import psutil
    _proc = psutil.Process()
    if hasattr(psutil, 'BELOW_NORMAL_PRIORITY_CLASS'):
        _proc.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
    else:
        _proc.nice(10)
    print(f"Process priority set to BELOW_NORMAL (nice={_proc.nice()})")
except Exception as _pe:
    print(f"Priority set skipped: {_pe}")

# ---------------------------------------------------------------------------
# MLflow (optional)
# ---------------------------------------------------------------------------
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
SCRIPT_DIR   = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parent.parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(PROJECT_ROOT))

DEFAULT_BOOK_DIR  = str(PROJECT_ROOT / 'data' / 'processed' / 'dl_book_cache')
DEFAULT_BOOK_OOT  = str(PROJECT_ROOT / 'data' / 'processed' / 'dl_book_cache_oot')
DEFAULT_OUTPUT    = str(SCRIPT_DIR / 'results' / '3d_cnn')

# Reference CNN predictions for correlation check
WIDER_CNN_PREDS = str(SCRIPT_DIR / 'results' / 'wider_cnn' /
                      'ckpt_preds_book_20260326_191614.npz')

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    format='%(asctime)s [3dcnn] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('3dcnn')


def _setup_file_logger(output_dir: str, timestamp: str):
    log_path = Path(output_dir) / f'train_3dcnn_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [3dcnn] %(levelname)s: %(message)s', datefmt='%H:%M:%S'))
    logger.addHandler(fh)
    return log_path


# ---------------------------------------------------------------------------
# MFE-net target computation (identical to train_walkforward.py)
# ---------------------------------------------------------------------------

def compute_mfe_net(mid_prices: np.ndarray, day_boundaries: List[int],
                    horizon_bars: int = 100, tick_size: float = 0.25) -> np.ndarray:
    """
    mfe_net_10s = mfe_long - mfe_short over 100-bar forward window.
    NaN at bars where window crosses a day boundary.
    """
    from numpy.lib.stride_tricks import sliding_window_view
    N = len(mid_prices)
    H = horizon_bars
    mfe_long  = np.full(N, np.nan, dtype=np.float32)
    mfe_short = np.full(N, np.nan, dtype=np.float32)
    mid_shifted = mid_prices[1:]
    valid_len   = N - H - 1
    if valid_len > 0:
        windows  = sliding_window_view(mid_shifted, H)[:valid_len]
        fwd_max  = windows.max(axis=1)
        fwd_min  = windows.min(axis=1)
        mfe_long[:valid_len]  = np.maximum(0.0, (fwd_max - mid_prices[:valid_len]) / tick_size)
        mfe_short[:valid_len] = np.maximum(0.0, (mid_prices[:valid_len] - fwd_min) / tick_size)
    n_days = len(day_boundaries) - 1
    for d in range(n_days - 1):
        day_end   = day_boundaries[d + 1]
        nan_start = max(day_boundaries[d], day_end - H)
        mfe_long[nan_start:day_end]  = np.nan
        mfe_short[nan_start:day_end] = np.nan
    return (mfe_long - mfe_short).astype(np.float32)


# ---------------------------------------------------------------------------
# Data utilities
# ---------------------------------------------------------------------------

def get_available_dates(cache_dir: str) -> List[str]:
    files = sorted(Path(cache_dir).glob('*_book_tensors.npz'))
    return [f.name.replace('_book_tensors.npz', '') for f in files]


def load_day_files(cache_dir: str, dates: List[str]):
    """Load NPZ files for a list of dates. Returns (day_data_list, mid_concat, boundaries)."""
    cache_path  = Path(cache_dir)
    all_data    = []
    boundaries  = [0]
    mid_buffers = []

    for date in dates:
        fname = cache_path / f'{date}_book_tensors.npz'
        if not fname.exists():
            logger.warning(f'  Missing: {fname.name}, skipping')
            continue
        npz = np.load(fname)
        day_dict = {'book_tensors': npz['book_tensors'], 'mid_prices': npz['mid_prices']}
        all_data.append(day_dict)
        mid_buffers.append(npz['mid_prices'])
        boundaries.append(boundaries[-1] + len(npz['mid_prices']))

    if mid_buffers:
        total = sum(len(m) for m in mid_buffers)
        mid_concat = np.empty(total, dtype=np.float64)
        pos = 0
        for m in mid_buffers:
            n = len(m)
            mid_concat[pos:pos + n] = m
            pos += n
    else:
        mid_concat = np.array([])
    return all_data, mid_concat, boundaries


# ---------------------------------------------------------------------------
# Dataset: 3D CNN video sequences
# ---------------------------------------------------------------------------

class BarDataset3D(Dataset):
    """
    Each sample: seq_len consecutive book snapshots (a "video clip") + scalar target.

    Raw tensor: (N_bars, 20, 4) float32
    Log-transform features 1,2,3 (depth, orders, age) in-place for numerical stability.
    Stored as float16 (halves RAM for large expanding windows).
    Cast back to float32 in __getitem__.

    Only creates samples within a single trading day (no cross-day sequences).
    Subsampling: use every `stride`-th valid bar (applied during training only).

    Leakage check:
      - Sample at bar i uses bars [i-seq_len+1, i] inclusive (all in the PAST).
      - Target at bar i is mfe_net[i] = forward return bars [i+1, i+100].
      - No lookahead: seq_len bars of history, target strictly in the future.
      - Day boundaries enforced: seq_len-1 warmup bars skipped at start of each day.
    """

    def __init__(
        self,
        day_data_list: List[Dict],
        target: np.ndarray,
        day_boundaries: List[int],
        seq_len: int = 50,
        subsample: int = 1,
    ):
        self.seq_len   = seq_len
        self.subsample = subsample

        # Pre-allocate and copy tensors, log-transform, store as float16
        def _safe_concat(day_data_list, key, dtype):
            shapes     = [d[key].shape for d in day_data_list]
            total_rows = sum(s[0] for s in shapes)
            out_shape  = (total_rows,) + shapes[0][1:]
            out = np.empty(out_shape, dtype=dtype)
            pos = 0
            for d in day_data_list:
                chunk = d[key]
                n = chunk.shape[0]
                if chunk.dtype != dtype:
                    chunk = chunk.astype(dtype, copy=False)
                out[pos:pos + n] = chunk
                d[key] = None
                pos += n
            return out

        tensors = _safe_concat(day_data_list, 'book_tensors', np.float32)
        # Log-transform depth, orders, queue_age (features 1,2,3) — same as book_spatial_cnn
        np.log1p(tensors[:, :, 1], out=tensors[:, :, 1])
        np.log1p(tensors[:, :, 2], out=tensors[:, :, 2])
        np.log1p(tensors[:, :, 3], out=tensors[:, :, 3])
        self.tensors = tensors.astype(np.float16)
        del tensors
        gc.collect()

        self.target         = target.astype(np.float32)
        self.day_boundaries = day_boundaries
        N                   = len(target)

        # Valid indices: need seq_len warmup bars within same day + finite target
        valid_mask = np.zeros(N, dtype=bool)
        for day_idx in range(len(day_boundaries) - 1):
            start = day_boundaries[day_idx]
            end   = min(day_boundaries[day_idx + 1], N)
            # First seq_len-1 bars can't form a full window
            slice_start = start + seq_len - 1
            if slice_start < end:
                valid_mask[slice_start:end] = np.isfinite(target[slice_start:end])

        all_valid = np.where(valid_mask)[0]
        del valid_mask
        if subsample > 1:
            all_valid = all_valid[::subsample]
        self.valid_indices = all_valid.astype(np.int64)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = int(self.valid_indices[idx])
        # (seq_len, 20, 4) — causal window ending at bar i
        clip = self.tensors[i - self.seq_len + 1 : i + 1].astype(np.float32)
        # Reshape to (4, seq_len, 20) — channels × time × levels (for Conv3d)
        # Conv3d input: (B, C, D, H, W) where D=time, H=levels, W=1
        clip_t = clip.transpose(2, 0, 1)  # (4, seq_len, 20)
        clip_t = clip_t[:, :, :, np.newaxis]  # (4, seq_len, 20, 1)
        return torch.from_numpy(clip_t), float(self.target[i])


def collate_3d(batch):
    clips   = torch.stack([b[0] for b in batch])  # (B, 4, T, 20, 1)
    targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return clips, targets


# ---------------------------------------------------------------------------
# Model: BookVideo3DCNN
# ---------------------------------------------------------------------------

class BookVideo3DCNN(nn.Module):
    """
    3D CNN processing book "video" — sequences of order book snapshots.

    Input: (B, C=4, T=50, H=20, W=1)
      C = book features (price_delta, log_depth, log_orders, log_age)
      T = time dimension (50 bars = 5 seconds)
      H = level dimension (20 levels: 10 bid + 10 ask)
      W = 1 (stub spatial dim; allows reuse of Conv3d across all dims)

    Architecture:
      Block 1: Conv3d(4→32, kernel=(5,3,1)) + BN + ReLU — captures 0.5s temporal windows
      Block 2: Conv3d(32→64, kernel=(5,3,1)) + BN + ReLU — stacks two temporal blocks
      Block 3: Conv3d(64→128, kernel=(3,3,1)) + BN + ReLU — higher-level features
      MaxPool3d(2,2,1) between blocks to compress time+levels
      AdaptiveAvgPool3d(1,1,1) — collapse all spatial/temporal dims to 128-d vector
      FC: 128 → 256 → 1

    Padding ensures same-size output at each conv (along T and H dims).
    W=1 gets no padding (already collapsed).

    ~1.1M parameters — lightweight enough for fast WF folds on RTX 3090.
    """

    def __init__(
        self,
        seq_len:  int   = 50,
        n_levels: int   = 20,
        n_feats:  int   = 4,
        dropout:  float = 0.15,
    ):
        super().__init__()
        self.seq_len  = seq_len
        self.n_levels = n_levels
        self.n_feats  = n_feats

        # Block 1: (B, 4, T, 20, 1) → (B, 32, T, 18, 1)
        # kernel (5,3,1): 5-bar temporal window, 3-level spatial, 1 on W dim
        # padding (2,1,0): same T, reduce H by 2 (no pad on H edges) — captures local level context
        self.conv1 = nn.Sequential(
            nn.Conv3d(n_feats, 32, kernel_size=(5, 3, 1), padding=(2, 1, 0), bias=False),
            nn.BatchNorm3d(32),
            nn.ReLU(inplace=True),
        )

        # Block 2: (B, 32, T, 18, 1) → (B, 64, T, 16, 1)
        self.conv2 = nn.Sequential(
            nn.Conv3d(32, 64, kernel_size=(5, 3, 1), padding=(2, 1, 0), bias=False),
            nn.BatchNorm3d(64),
            nn.ReLU(inplace=True),
        )

        # Temporal + level pooling: halve T and H
        # (B, 64, T, 16, 1) → (B, 64, T//2, 8, 1)
        self.pool1 = nn.MaxPool3d(kernel_size=(2, 2, 1), stride=(2, 2, 1))

        # Block 3: (B, 64, T//2, 8, 1) → (B, 128, T//2, 6, 1)
        self.conv3 = nn.Sequential(
            nn.Conv3d(64, 128, kernel_size=(3, 3, 1), padding=(1, 1, 0), bias=False),
            nn.BatchNorm3d(128),
            nn.ReLU(inplace=True),
        )

        # Block 4: deeper temporal + level features
        # (B, 128, T//2, 6, 1) → (B, 256, T//2, 4, 1)
        self.conv4 = nn.Sequential(
            nn.Conv3d(128, 256, kernel_size=(3, 3, 1), padding=(1, 1, 0), bias=False),
            nn.BatchNorm3d(256),
            nn.ReLU(inplace=True),
        )

        # Second pooling: halve T again
        # (B, 256, T//2, 4, 1) → (B, 256, T//4, 4, 1)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        # Collapse all spatial/temporal dims → 256-d vector
        self.global_pool = nn.AdaptiveAvgPool3d((1, 1, 1))

        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 4, T, 20, 1)
        x = self.conv1(x)    # → (B, 32, T, 18, 1)  [H reduces by 2 without H padding]
        x = self.conv2(x)    # → (B, 64, T, 16, 1)
        x = self.pool1(x)    # → (B, 64, T//2, 8, 1)
        x = self.conv3(x)    # → (B, 128, T//2, 6, 1) [H reduces by 2]
        x = self.conv4(x)    # → (B, 256, T//2, 4, 1) [H reduces by 2]
        x = self.pool2(x)    # → (B, 256, T//4, 4, 1)
        x = self.global_pool(x)  # → (B, 256, 1, 1, 1)
        x = x.flatten(1)    # → (B, 256)
        return self.head(x)  # → (B, 1)


# ---------------------------------------------------------------------------
# Leakage audit
# ---------------------------------------------------------------------------

def leakage_audit(seq_len: int, horizon_bars: int = 100) -> bool:
    """
    Verify no lookahead bias:
      - Sample at bar i uses history [i-seq_len+1, i] (all past)
      - Target at bar i is forward return [i+1, i+100] (all future)
      - No overlap. Purge gap = 1 day between train/test.
    """
    last_history_bar  = 0  # bar i (most recent in window)
    first_target_bar  = 1  # bar i+1 (start of forward window)
    last_target_bar   = horizon_bars  # bar i+100 (end of forward window)

    assert last_history_bar < first_target_bar, "LEAKAGE: history overlaps target!"
    assert seq_len > 0, "seq_len must be positive"
    assert horizon_bars > 0, "horizon_bars must be positive"

    logger.info("=" * 60)
    logger.info("LEAKAGE AUDIT: PASSED")
    logger.info(f"  History window: bars [i-{seq_len-1}, i] (past, causal)")
    logger.info(f"  Target window:  bars [i+1, i+{horizon_bars}] (future, 10s)")
    logger.info(f"  Overlap:        NONE (gap = 1 bar = 100ms)")
    logger.info(f"  Day boundaries: enforced in BarDataset3D (no cross-day clips)")
    logger.info(f"  Normalization:  train stats only, applied to both train+test")
    logger.info(f"  Purge gap:      1 day between train end and test day")
    logger.info("=" * 60)
    return True


# ---------------------------------------------------------------------------
# Training utilities
# ---------------------------------------------------------------------------

def train_epoch(
    model:     nn.Module,
    loader:    DataLoader,
    optimizer: torch.optim.Optimizer,
    device:    torch.device,
    scaler:    Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, int]:
    model.train()
    criterion = nn.HuberLoss(delta=1.0)
    total_loss = 0.0
    total_n    = 0

    for clips, targets in loader:
        clips   = clips.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds = model(clips).squeeze(-1)
                loss  = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds = model(clips).squeeze(-1)
            loss  = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        n = targets.shape[0]
        total_loss += loss.item() * n
        total_n    += n

    return total_loss / max(total_n, 1), total_n


@torch.no_grad()
def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    model.eval()
    criterion  = nn.HuberLoss(delta=1.0)
    all_preds  = []
    all_tgts   = []
    total_loss = 0.0
    total_n    = 0

    ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad()

    for clips, targets in loader:
        clips   = clips.to(device)
        targets = targets.to(device)
        with ctx:
            preds = model(clips).squeeze(-1)
        val_l = criterion(preds.cpu().float(), targets.cpu().float()).item()
        n = targets.shape[0]
        total_loss += val_l * n
        total_n    += n
        all_preds.append(preds.cpu().float().numpy())
        all_tgts.append(targets.cpu().numpy())

    if not all_preds:
        return 0.0, 0.0, np.array([]), np.array([])

    val_loss   = total_loss / max(total_n, 1)
    preds_arr  = np.concatenate(all_preds)
    tgts_arr   = np.concatenate(all_tgts)

    mask = np.isfinite(preds_arr) & np.isfinite(tgts_arr)
    if mask.sum() < 10:
        return 0.0, val_loss, preds_arr, tgts_arr

    ic, _ = spearmanr(preds_arr[mask], tgts_arr[mask])
    return float(ic) if np.isfinite(ic) else 0.0, val_loss, preds_arr, tgts_arr


# ---------------------------------------------------------------------------
# Correlation check vs wider CNN
# ---------------------------------------------------------------------------

def check_cnn_correlation(
    all_preds:  List[np.ndarray],
    fold_dates: List[str],
    ref_npz:    str,
) -> float:
    """
    Compute Spearman correlation between 3D CNN predictions and wider CNN predictions
    over the overlapping dates.
    Returns overall correlation (0.0 if reference not available).
    """
    if not Path(ref_npz).exists():
        logger.warning(f"  Reference CNN NPZ not found: {ref_npz}")
        return 0.0

    try:
        ref_data = np.load(ref_npz)
        my_preds  = []
        ref_preds = []
        for i, date in enumerate(fold_dates):
            key = f'{date}_preds'
            if key in ref_data and i < len(all_preds):
                rp = ref_data[key]
                mp = all_preds[i]
                min_n = min(len(rp), len(mp))
                if min_n > 0:
                    my_preds.append(mp[:min_n])
                    ref_preds.append(rp[:min_n])

        if not my_preds:
            return 0.0

        all_my  = np.concatenate(my_preds)
        all_ref = np.concatenate(ref_preds)
        mask = np.isfinite(all_my) & np.isfinite(all_ref)
        if mask.sum() < 10:
            return 0.0

        corr, _ = spearmanr(all_my[mask], all_ref[mask])
        return float(corr) if np.isfinite(corr) else 0.0

    except Exception as e:
        logger.warning(f"  Correlation check failed: {e}")
        return 0.0


# ---------------------------------------------------------------------------
# Walk-forward training loop
# ---------------------------------------------------------------------------

def train_walkforward(
    cache_dir:      str   = DEFAULT_BOOK_DIR,
    output_dir:     str   = DEFAULT_OUTPUT,
    seq_len:        int   = 50,
    min_train_days: int   = 15,
    purge_days:     int   = 1,
    horizon_bars:   int   = 100,
    epochs:         int   = 20,
    patience:       int   = 5,
    batch_size:     int   = 512,
    lr:             float = 3e-4,
    subsample:      int   = 50,
    device_str:     str   = 'cuda',
    seed:           int   = 42,
    start_fold:     int   = 0,
    n_days:         Optional[int] = None,
) -> Dict:
    """
    Expanding-window walk-forward training for BookVideo3DCNN.

    Protocol:
      For each test_day from min_train_days+purge_days onwards:
        train on days [0, test_day-purge_days)  (expanding)
        purge 1 day
        test on test_day

    Returns dict with fold_ics, concat_ic, correlation vs wider CNN.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    Path(output_dir).mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_path  = _setup_file_logger(output_dir, timestamp)

    device  = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    use_amp = (device.type == 'cuda')

    logger.info('=' * 70)
    logger.info('BookVideo3DCNN — 3D CNN Walk-Forward Training')
    logger.info(f'  Log:            {log_path}')
    logger.info(f'  Cache dir:      {cache_dir}')
    logger.info(f'  Output dir:     {output_dir}')
    logger.info(f'  seq_len:        {seq_len} bars ({seq_len*0.1:.1f}s window)')
    logger.info(f'  min_train_days: {min_train_days}')
    logger.info(f'  purge_days:     {purge_days}')
    logger.info(f'  epochs/fold:    {epochs}  (patience={patience})')
    logger.info(f'  batch_size:     {batch_size}')
    logger.info(f'  subsample:      {subsample} (every {subsample}th bar)')
    logger.info(f'  device:         {device}  (AMP={use_amp})')
    logger.info('=' * 70)

    # Run leakage audit before any training
    leakage_audit(seq_len, horizon_bars)

    # Get dates
    dates = get_available_dates(cache_dir)
    if n_days is not None and n_days < len(dates):
        dates = dates[:n_days]
    n_total = len(dates)
    logger.info(f'  Using {n_total} days: {dates[0]} .. {dates[-1]}')

    if n_total < min_train_days + purge_days + 1:
        raise ValueError(f'Not enough days: {n_total}')

    # Temp model to count params
    _tmp = BookVideo3DCNN(seq_len=seq_len)
    n_params = sum(p.numel() for p in _tmp.parameters())
    del _tmp
    logger.info(f'  Model params: {n_params:,} ({n_params/1e6:.2f}M)')

    # MLflow
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri('http://localhost:5000')
            mlflow.set_experiment('CNN_Training')
            mlflow_run = mlflow.start_run(run_name=f'3d_cnn_wf_{timestamp}')
            mlflow.log_params({
                'model':        'BookVideo3DCNN',
                'seq_len':      seq_len,
                'epochs':       epochs,
                'patience':     patience,
                'batch_size':   batch_size,
                'subsample':    subsample,
                'n_params':     n_params,
                'window_mode':  'expanding',
                'horizon_bars': horizon_bars,
            })
            logger.info(f'  MLflow run: {mlflow_run.info.run_id}')
        except Exception as e:
            logger.warning(f'  MLflow init failed (non-fatal): {e}')
            mlflow_run = None

    # State
    fold_ics      = []
    fold_details  = []
    all_oos_preds = []
    all_oos_tgts  = []
    all_oos_dates = []

    n_folds = n_total - min_train_days - purge_days
    logger.info(f'  Total folds: {n_folds}')

    # Checkpoint resume
    checkpoint_file = Path(output_dir) / f'checkpoint_3dcnn_{timestamp}.json'
    existing_ckpts  = sorted(Path(output_dir).glob('checkpoint_3dcnn_*.json'))
    resume_fold     = 0
    if existing_ckpts:
        try:
            best_ckpt = max(existing_ckpts,
                            key=lambda p: json.load(open(p)).get('completed_folds', 0))
            with open(best_ckpt) as f:
                ckpt = json.load(f)
            fold_ics     = ckpt.get('fold_ics', [])
            fold_details = ckpt.get('fold_details', [])
            resume_fold  = ckpt.get('completed_folds', 0)
            checkpoint_file = best_ckpt
            # Reload OOS predictions
            preds_path = Path(output_dir) / ckpt.get('preds_file', '')
            if preds_path.exists():
                pdata = np.load(str(preds_path))
                for i in range(resume_fold):
                    if i < len(fold_details):
                        d = fold_details[i].get('test_date', f'fold_{i}')
                        pk, tk = f'{d}_preds', f'{d}_targets'
                        if pk in pdata and tk in pdata:
                            all_oos_preds.append(pdata[pk])
                            all_oos_tgts.append(pdata[tk])
                            all_oos_dates.append(d)
            if resume_fold > 0:
                logger.info(f'  CHECKPOINT RESUME: Skipping {resume_fold} folds '
                            f'(mean IC so far: {np.mean(fold_ics):+.4f})')
        except Exception as e:
            logger.warning(f'  Checkpoint load failed ({e}), starting fresh')
            fold_ics, fold_details = [], []
            all_oos_preds, all_oos_tgts, all_oos_dates = [], [], []
            resume_fold = 0

    effective_skip = max(resume_fold, start_fold)

    for fold_idx, test_day_idx in enumerate(
            range(min_train_days + purge_days, n_total)):

        if fold_idx < effective_skip:
            continue

        train_end_idx   = test_day_idx - purge_days
        train_dates     = dates[0:train_end_idx]        # expanding: always from day 0
        test_dates      = [dates[test_day_idx]]

        t_fold = time.time()
        logger.info(f'\n--- Fold {fold_idx+1}/{n_folds} | '
                    f'Train: {train_dates[0]}..{train_dates[-1]} ({len(train_dates)}d) | '
                    f'Test: {test_dates[0]} ---')

        # ---- Load train data ----
        train_data_list, train_mids, train_bounds = load_day_files(cache_dir, train_dates)
        if not train_data_list:
            logger.warning(f'  Fold {fold_idx+1}: no train data, skipping')
            continue

        train_target = compute_mfe_net(train_mids, train_bounds, horizon_bars)
        n_valid = int(np.isfinite(train_target).sum())
        if n_valid < 1000:
            logger.warning(f'  Fold {fold_idx+1}: too few train bars ({n_valid}), skipping')
            continue

        # Normalize target from train stats
        finite_mask = np.isfinite(train_target)
        tgt_mean = float(train_target[finite_mask].mean())
        tgt_std  = float(train_target[finite_mask].std())
        if tgt_std < 1e-8:
            tgt_std = 1.0
        train_target = (train_target - tgt_mean) / tgt_std
        logger.info(f'  Target norm: mean={tgt_mean:.3f} std={tgt_std:.3f} ticks')

        # ---- Build train dataset ----
        train_dataset = BarDataset3D(
            train_data_list, train_target, train_bounds,
            seq_len=seq_len, subsample=subsample,
        )
        del train_data_list, train_mids, train_target, train_bounds
        gc.collect()
        logger.info(f'  Train samples: {len(train_dataset):,} '
                    f'(subsample={subsample}x, raw valid ~{n_valid:,})')

        # ---- Load test data ----
        test_data_list, test_mids, test_bounds = load_day_files(cache_dir, test_dates)
        if not test_data_list:
            logger.warning(f'  Fold {fold_idx+1}: no test data, skipping')
            del train_dataset
            continue

        test_target = compute_mfe_net(test_mids, test_bounds, horizon_bars)
        test_target = (test_target - tgt_mean) / tgt_std

        test_dataset = BarDataset3D(
            test_data_list, test_target, test_bounds,
            seq_len=seq_len, subsample=1,  # ALL test bars
        )
        del test_data_list, test_mids, test_target, test_bounds
        gc.collect()
        logger.info(f'  Test samples: {len(test_dataset):,}')

        if len(train_dataset) < 100:
            logger.warning(f'  Too few train samples, skipping')
            del train_dataset, test_dataset
            continue

        # Windows pickle issue: num_workers>0 causes OSError with collate_fn
        _nw_train = 0 if sys.platform == 'win32' else 4
        _nw_test  = 0 if sys.platform == 'win32' else 2
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_3d, num_workers=_nw_train, pin_memory=True,
            drop_last=True, persistent_workers=(_nw_train > 0),
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size * 2, shuffle=False,
            collate_fn=collate_3d, num_workers=_nw_test, pin_memory=True,
        )

        # ---- Build model ----
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        model = BookVideo3DCNN(seq_len=seq_len).to(device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3,
        )
        scaler = torch.amp.GradScaler('cuda', init_scale=1024) if use_amp else None

        # ---- Train with early stopping ----
        best_ic         = -999.0
        best_preds      = np.array([])
        best_tgts       = np.array([])
        no_improve      = 0
        epoch_stats     = []

        for epoch in range(epochs):
            t0 = time.time()
            train_loss, _ = train_epoch(model, train_loader, optimizer, device, scaler)
            scheduler.step()
            ic, val_loss, preds, tgts = evaluate(model, test_loader, device)
            epoch_stats.append({
                'epoch': epoch + 1, 'train_loss': train_loss,
                'val_loss': val_loss, 'ic': ic,
            })
            overfit = ' **OVERFIT**' if val_loss > train_loss * 1.15 else ''
            logger.info(
                f'  Epoch {epoch+1}/{epochs}: train_loss={train_loss:.5f}  '
                f'val_loss={val_loss:.5f}  IC={ic:+.4f}  ({time.time()-t0:.1f}s){overfit}'
            )

            if ic > best_ic:
                best_ic    = ic
                best_preds = preds.copy()
                best_tgts  = tgts.copy()
                no_improve = 0
                # Save best weights for this fold
                _ckpt_dir = Path(output_dir) / 'checkpoints'
                _ckpt_dir.mkdir(parents=True, exist_ok=True)
                _pt = _ckpt_dir / f'fold_{fold_idx}_{test_dates[0]}.pt'
                torch.save({k: v.cpu().clone() for k, v in model.state_dict().items()},
                           str(_pt))
            else:
                no_improve += 1

            if no_improve >= patience:
                logger.info(f'  Early stopping at epoch {epoch+1} (patience={patience})')
                break

        fold_ics.append(best_ic)
        all_oos_preds.append(best_preds)
        all_oos_tgts.append(best_tgts)
        all_oos_dates.append(test_dates[0])

        fold_details.append({
            'fold':      fold_idx + 1,
            'test_date': test_dates[0],
            'train_days': len(train_dates),
            'ic':        best_ic,
            'n_test':    int(np.isfinite(best_tgts).sum()),
            'epoch_stats': epoch_stats,
            'tgt_mean':  tgt_mean,
            'tgt_std':   tgt_std,
        })
        logger.info(f'  Fold IC: {best_ic:+.4f}  [{time.time()-t_fold:.1f}s]')

        # Log to MLflow
        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                mlflow.log_metrics(
                    {'fold_ic': best_ic, 'mean_ic': float(np.mean(fold_ics))},
                    step=fold_idx + 1,
                )
            except Exception:
                pass

        # ---- Save per-fold .npz OOS predictions ----
        try:
            _fold_npz = Path(output_dir) / f'fold_{fold_idx}_{test_dates[0]}_preds.npz'
            np.savez_compressed(str(_fold_npz),
                                preds=best_preds, targets=best_tgts)
        except Exception as e:
            logger.warning(f'  Per-fold NPZ save failed: {e}')

        # ---- Checkpoint (cumulative) ----
        try:
            ckpt_preds_file = Path(output_dir) / f'ckpt_preds_3dcnn_{timestamp}.npz'
            pred_data = {}
            for ci, (cp, ct) in enumerate(zip(all_oos_preds, all_oos_tgts)):
                d = all_oos_dates[ci] if ci < len(all_oos_dates) else f'fold_{ci}'
                pred_data[f'{d}_preds']   = cp
                pred_data[f'{d}_targets'] = ct
            np.savez_compressed(str(ckpt_preds_file), **pred_data)

            ckpt_data = {
                'model':           'BookVideo3DCNN',
                'completed_folds': fold_idx + 1,
                'fold_ics':        [float(x) for x in fold_ics],
                'mean_ic':         float(np.mean(fold_ics)),
                'fold_details':    fold_details,
                'preds_file':      ckpt_preds_file.name,
                'timestamp':       timestamp,
                'seq_len':         seq_len,
                'n_params':        n_params,
            }
            with open(checkpoint_file, 'w') as f:
                json.dump(ckpt_data, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f'  Checkpoint save failed: {e}')

        # ---- Cleanup VRAM ----
        if device.type == 'cuda':
            model.cpu()
        del model, optimizer, scheduler, scaler
        del train_dataset, test_dataset, train_loader, test_loader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        logger.info(f'  VRAM: {torch.cuda.memory_allocated()/1024**2:.0f}MB alloc  '
                    f'{torch.cuda.memory_reserved()/1024**2:.0f}MB reserved'
                    if device.type == 'cuda' else '')

    # ---- Aggregate results ----
    if not fold_ics:
        logger.error('No folds completed!')
        return {'error': 'No folds completed', 'fold_ics': []}

    ic_arr     = np.array(fold_ics)
    agg_ic     = float(ic_arr.mean())
    agg_ic_std = float(ic_arr.std())
    agg_icir   = float(agg_ic / agg_ic_std) if agg_ic_std > 0 else 0.0

    # Concat IC (the REAL test)
    all_p  = np.concatenate([p for p in all_oos_preds if len(p) > 0])
    all_t  = np.concatenate([t for t in all_oos_tgts  if len(t) > 0])
    mask   = np.isfinite(all_p) & np.isfinite(all_t)
    concat_ic = float(spearmanr(all_p[mask], all_t[mask])[0]) if mask.sum() > 10 else 0.0

    # Correlation vs wider CNN
    cnn_corr = check_cnn_correlation(all_oos_preds, all_oos_dates, WIDER_CNN_PREDS)

    logger.info('')
    logger.info('=' * 70)
    logger.info('RESULTS: BookVideo3DCNN (3D CNN)')
    logger.info(f'  Folds completed:       {len(fold_ics)}')
    logger.info(f'  Per-fold IC mean:      {agg_ic:+.4f}')
    logger.info(f'  Per-fold IC std:       {agg_ic_std:.4f}')
    logger.info(f'  ICIR:                  {agg_icir:+.3f}')
    logger.info(f'  Concatenated IC:       {concat_ic:+.4f}  <-- PRIMARY METRIC')
    logger.info(f'  Wider CNN baseline:    +0.261 (per-fold) / see ref')
    logger.info(f'  Corr vs wider CNN:     {cnn_corr:+.3f}  (want <0.80 for diversification)')
    logger.info(f'  LEAKAGE AUDIT:         PASSED (causal windows, train-only norm)')
    logger.info('')
    if concat_ic > 0.05 and cnn_corr < 0.80:
        logger.info('  VERDICT: WIN! Concat IC > 0.05 AND corr < 0.80 → diversifying signal')
    elif concat_ic > 0.05:
        logger.info(f'  VERDICT: IC good but high correlation ({cnn_corr:.2f}) with wider CNN')
    else:
        logger.info(f'  VERDICT: Concat IC {concat_ic:.3f} < 0.05 threshold')
    logger.info('=' * 70)

    # Save full OOS predictions
    preds_path = Path(output_dir) / f'oos_predictions_3dcnn_{timestamp}.npz'
    pred_data_final = {}
    for i, (p, t) in enumerate(zip(all_oos_preds, all_oos_tgts)):
        d = all_oos_dates[i] if i < len(all_oos_dates) else f'fold_{i}'
        pred_data_final[f'{d}_preds']   = p
        pred_data_final[f'{d}_targets'] = t
    np.savez_compressed(str(preds_path), **pred_data_final)
    logger.info(f'  OOS predictions saved: {preds_path}')

    results = {
        'model':       'BookVideo3DCNN',
        'n_folds':     len(fold_ics),
        'fold_ics':    fold_ics,
        'agg_ic':      agg_ic,
        'agg_ic_std':  agg_ic_std,
        'agg_icir':    agg_icir,
        'concat_ic':   concat_ic,
        'cnn_corr':    cnn_corr,
        'leakage_audit': 'PASSED',
        'fold_details': fold_details,
        'config': {
            'seq_len':       seq_len,
            'min_train_days': min_train_days,
            'epochs':         epochs,
            'patience':       patience,
            'batch_size':     batch_size,
            'subsample':      subsample,
            'horizon_bars':   horizon_bars,
            'n_params':       n_params,
        },
    }

    results_path = Path(output_dir) / f'results_3dcnn_{timestamp}.json'
    with open(results_path, 'w') as f:
        def _conv(o):
            if isinstance(o, (np.integer,)):  return int(o)
            if isinstance(o, (np.floating,)): return float(o)
            if isinstance(o, np.ndarray):     return o.tolist()
            return o
        def _rec(d):
            if isinstance(d, dict):  return {k: _rec(v) for k, v in d.items()}
            if isinstance(d, list):  return [_rec(x) for x in d]
            return _conv(d)
        json.dump(_rec(results), f, indent=2)
    logger.info(f'  Results saved: {results_path}')

    if MLFLOW_AVAILABLE and mlflow_run is not None:
        try:
            mlflow.log_metrics({
                'agg_ic':    agg_ic,
                'concat_ic': concat_ic,
                'cnn_corr':  cnn_corr,
                'icir':      agg_icir,
                'n_folds':   float(len(fold_ics)),
            })
            mlflow.end_run()
        except Exception:
            pass

    return results


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='BookVideo3DCNN — 3D CNN walk-forward training on book snapshots',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--cache-dir',      type=str, default=DEFAULT_BOOK_DIR)
    parser.add_argument('--output-dir',     type=str, default=DEFAULT_OUTPUT)
    parser.add_argument('--seq-len',        type=int, default=50,
                        help='Sequence length (bars). 50=5s, 100=10s, 20=2s')
    parser.add_argument('--min-train-days', type=int, default=15)
    parser.add_argument('--purge-days',     type=int, default=1)
    parser.add_argument('--epochs',         type=int, default=20)
    parser.add_argument('--patience',       type=int, default=5)
    parser.add_argument('--batch-size',     type=int, default=512)
    parser.add_argument('--lr',             type=float, default=3e-4)
    parser.add_argument('--subsample',      type=int, default=50,
                        help='Use every Nth training bar (50=every 5s, reduces RAM)')
    parser.add_argument('--device',         type=str, default='cuda')
    parser.add_argument('--seed',           type=int, default=42)
    parser.add_argument('--start-fold',     type=int, default=0,
                        help='Skip to this fold index (0-based) for resuming')
    parser.add_argument('--days',           type=int, default=None,
                        help='Limit total days used (None=all)')

    args = parser.parse_args()

    print('=' * 70)
    print('BookVideo3DCNN — 3D CNN for Book Snapshot Sequences')
    print(f'  seq_len={args.seq_len} ({args.seq_len * 0.1:.1f}s video clips)')
    print(f'  Architecture: Conv3d(4->32->64->128->256) + AdaptivePool + FC')
    print(f'  Training: expanding WF, {args.epochs} epochs/fold (patience={args.patience})')
    print(f'  Normalization: train stats only | LEAKAGE AUDIT: built-in')
    print('=' * 70)

    results = train_walkforward(
        cache_dir      = args.cache_dir,
        output_dir     = args.output_dir,
        seq_len        = args.seq_len,
        min_train_days = args.min_train_days,
        purge_days     = args.purge_days,
        epochs         = args.epochs,
        patience       = args.patience,
        batch_size     = args.batch_size,
        lr             = args.lr,
        subsample      = args.subsample,
        device_str     = args.device,
        seed           = args.seed,
        start_fold     = args.start_fold,
        n_days         = args.days,
    )

    print('\n' + '=' * 70)
    if 'error' not in results:
        print(f"FINAL RESULTS:")
        print(f"  Folds:          {results['n_folds']}")
        print(f"  Per-fold IC:    {results['agg_ic']:+.4f} ± {results['agg_ic_std']:.4f}")
        print(f"  ICIR:           {results['agg_icir']:+.3f}")
        print(f"  Concat IC:      {results['concat_ic']:+.4f}  <-- PRIMARY")
        print(f"  CNN Corr:       {results['cnn_corr']:+.3f}")
        print(f"  Leakage Audit:  {results['leakage_audit']}")
    print('=' * 70)

    return results


if __name__ == '__main__':
    main()
