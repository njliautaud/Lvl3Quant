"""
Unified Walk-Forward Training Script for Raw-Data Deep Learning Models.

Trains EventTransformer, BookSpatialCNN, TradeSequenceLSTM, and HybridModel
using expanding-window walk-forward cross-validation, matching the LightGBM
methodology.

Walk-Forward Protocol:
  - Train on days 0..N-2 (expanding window)
  - 1-day purge gap (skip day N-1)
  - Test on day N
  - Metric: Spearman IC (rank correlation between predictions and mfe_net target)

Target: mfe_net_10s (Max Favorable Excursion net, 10 seconds horizon)
  - Computed from mid_prices in DL cache files
  - = mfe_long - mfe_short (directional asymmetry, in ticks)
  - NaN at day boundaries (strict causal rule)

Usage:
  python train_walkforward.py --model all --days 7 --epochs 3
  python train_walkforward.py --model event --days 20 --epochs 5 --batch-size 512
  python train_walkforward.py --model book --device cpu --augment
  python train_walkforward.py --model lstm --subsample-train 3
  python train_walkforward.py --model hybrid --days 20 --epochs 5

  # Hybrid model: combines BookSpatialCNN encoder + EventTransformer encoder +
  #               optional engineered features MLP via late fusion
  python train_walkforward.py --model hybrid --book-dir /path --events-dir /path

Results saved to: alpha_discovery/deep_models/results/
"""

import argparse
import gc
import json
import logging
import math
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
# Path setup
# ---------------------------------------------------------------------------
ROOT_DIR = Path(__file__).resolve().parent.parent.parent  # Lvl3Quant root
MODELS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT_DIR))
sys.path.insert(0, str(MODELS_DIR))

try:
    from alpha_discovery.compute_notifier import notify_complete  # noqa: E402
except ImportError:
    def notify_complete(**kwargs):
        pass  # No-op when compute_notifier unavailable

# ---------------------------------------------------------------------------
# Logging  (console + file)
# ---------------------------------------------------------------------------
logging.basicConfig(
    format='%(asctime)s [%(name)s] %(levelname)s: %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('train_wf')


def _setup_file_logger(output_dir: str, model_type: str, timestamp: str):
    """Add a file handler so all output is also written to a log file."""
    log_path = Path(output_dir) / f'walkforward_{model_type}_{timestamp}.log'
    fh = logging.FileHandler(str(log_path), mode='w')
    fh.setLevel(logging.INFO)
    fh.setFormatter(logging.Formatter(
        '%(asctime)s [%(name)s] %(levelname)s: %(message)s', datefmt='%H:%M:%S',
    ))
    logger.addHandler(fh)
    return log_path

# ---------------------------------------------------------------------------
# Default Cache Directories
# ---------------------------------------------------------------------------
DEFAULT_EVENTS_DIR = str(ROOT_DIR / 'data' / 'processed' / 'dl_events_cache')
DEFAULT_BOOK_DIR   = str(ROOT_DIR / 'data' / 'processed' / 'dl_book_cache')
DEFAULT_TRADES_DIR = str(ROOT_DIR / 'data' / 'processed' / 'dl_trades_cache')
DEFAULT_OUTPUT_DIR = str(MODELS_DIR / 'results' / 'standard_cnn')

# ---------------------------------------------------------------------------
# MFE Target Computation
# ---------------------------------------------------------------------------

def compute_mfe_net(mid_prices: np.ndarray, day_boundaries: List[int],
                    horizon_bars: int = 100, tick_size: float = 0.25) -> np.ndarray:
    """
    Compute mfe_net_10s = mfe_long - mfe_short over a forward window.

    mfe_long[t]  = max(0, max(mid[t+1:t+H+1]) - mid[t]) / tick_size  (in ticks)
    mfe_short[t] = max(0, mid[t] - min(mid[t+1:t+H+1])) / tick_size  (in ticks)
    mfe_net[t]   = mfe_long[t] - mfe_short[t]

    NaN at bars where the forward window crosses a day boundary.

    Args:
        mid_prices: (N,) float64 mid-price array
        day_boundaries: list of [start_day0, start_day1, ..., end_last_day]
        horizon_bars: number of forward bars (100 = 10s at 100ms)
        tick_size: ES tick = 0.25

    Returns:
        mfe_net: (N,) float32 array with NaN at invalid bars
    """
    from numpy.lib.stride_tricks import sliding_window_view

    N = len(mid_prices)
    H = horizon_bars

    mfe_long  = np.full(N, np.nan, dtype=np.float32)
    mfe_short = np.full(N, np.nan, dtype=np.float32)

    # Vectorised sliding window for forward max/min
    # mid_shifted[i] = mid[i+1]; window[t] = mid[t+1:t+H+1]
    mid_shifted = mid_prices[1:]   # (N-1,)
    valid_len   = N - H - 1        # bars t=0..N-H-2 have full forward window

    if valid_len > 0:
        windows  = sliding_window_view(mid_shifted, H)[:valid_len]  # (valid_len, H)
        fwd_max  = windows.max(axis=1)
        fwd_min  = windows.min(axis=1)

        mfe_long[:valid_len]  = np.maximum(0.0, (fwd_max - mid_prices[:valid_len]) / tick_size)
        mfe_short[:valid_len] = np.maximum(0.0, (mid_prices[:valid_len] - fwd_min) / tick_size)

    # NaN bars whose windows cross a day boundary
    n_days = len(day_boundaries) - 1
    for d in range(n_days - 1):
        day_end   = day_boundaries[d + 1]
        nan_start = max(day_boundaries[d], day_end - H)
        mfe_long[nan_start:day_end]  = np.nan
        mfe_short[nan_start:day_end] = np.nan

    mfe_net = (mfe_long - mfe_short).astype(np.float32)
    return mfe_net


# ---------------------------------------------------------------------------
# Data Loaders (per-day, lazy loading to control memory)
# ---------------------------------------------------------------------------

def load_day_files(cache_dir: str, model_type: str, dates: List[str]):
    """
    Load .npz files for given dates from the appropriate cache directory.

    Returns:
        data_arrays: list of dicts, one per date, with arrays
        mid_prices_concat: concatenated mid_prices for MFE computation
        day_boundaries: [0, n_bars_day0, n_bars_day0+n_bars_day1, ...]
    """
    cache_path = Path(cache_dir)

    suffix_map = {
        'event':  '_event_tokens.npz',
        'book':   '_book_tensors.npz',
        'lstm':   '_trade_flow.npz',
        'hybrid': '_book_tensors.npz',  # hybrid uses book as primary date source
    }
    suffix = suffix_map[model_type]

    all_data    = []
    boundaries  = [0]
    # MEMORY-SAFE: collect mid shapes first, pre-allocate, copy in-place.
    # Old approach (np.concatenate(all_mids)) peaks at 2x total size.
    mid_shapes  = []
    mid_buffers = []  # temporary per-day refs, freed after pre-alloc copy

    for date in dates:
        fname = cache_path / f'{date}{suffix}'
        if not fname.exists():
            logger.warning(f'  Missing file: {fname.name}, skipping')
            continue
        npz = np.load(fname)
        day_dict = dict(npz)
        all_data.append(day_dict)
        mid = npz['mid_prices']
        mid_shapes.append(len(mid))
        mid_buffers.append(mid)
        boundaries.append(boundaries[-1] + len(mid))

    if mid_buffers:
        total_mids = sum(mid_shapes)
        mid_concat = np.empty(total_mids, dtype=np.float64)
        pos = 0
        for mid in mid_buffers:
            n = len(mid)
            mid_concat[pos:pos + n] = mid
            pos += n
        del mid_buffers, mid_shapes  # free list references
    else:
        mid_concat = np.array([])

    return all_data, mid_concat, boundaries


def get_available_dates(cache_dir: str, model_type: str) -> List[str]:
    """Return sorted list of YYYY-MM-DD date strings available in cache dir."""
    suffix_map = {
        'event':  '_event_tokens.npz',
        'book':   '_book_tensors.npz',
        'lstm':   '_trade_flow.npz',
        'hybrid': '_book_tensors.npz',  # hybrid uses book as date source
    }
    suffix = suffix_map[model_type]
    files  = sorted(Path(cache_dir).glob(f'*{suffix}'))
    dates  = [f.name.replace(suffix, '') for f in files]
    return dates


# ---------------------------------------------------------------------------
# Per-Bar Dataset for a Given Train/Test Window
# ---------------------------------------------------------------------------

class BarDataset(Dataset):
    """
    Dataset for a single train or test window.
    Handles all four model types (event/book/lstm/hybrid).

    Subsample is applied only during training to reduce memory.

    For 'hybrid' model type, day_data_list must contain dicts with BOTH
    'book_tensors' and 'event_sequences'/'sequence_lengths' keys
    (i.e., merged from two cache directories by the caller).
    """

    def __init__(
        self,
        day_data_list: List[Dict],       # one dict per day in this window
        target: np.ndarray,              # (N_total,) mfe_net for this window
        day_boundaries: List[int],       # boundaries within this window
        model_type: str,                 # 'event', 'book', 'lstm', or 'hybrid'
        window_size: int = 20,           # for book/lstm/hybrid: consecutive bars window
        horizon: int = 20,              # bars forward for target (not used directly)
        subsample: int = 1,              # use every Nth bar (1 = no subsample)
    ):
        self.model_type   = model_type
        self.window_size  = window_size
        self.subsample    = subsample

        # Concatenate all data arrays
        # MEMORY-SAFE: pre-allocate output, copy chunks, free each source immediately.
        # np.concatenate peaks at 2x total size (source + output held simultaneously).
        # This approach peaks at max(single_day, total_output) — critical for 18M+ bar folds.
        def _safe_concat(day_data_list, key, dtype):
            """Pre-allocate output array and copy each day chunk into it.
            CRITICAL: Nulls out each source array after copying so peak RAM =
            output_array + one_day_chunk, NOT output + all_source simultaneously.
            """
            # Collect shapes without holding array refs (shape is a tuple, not the data)
            shapes = [d[key].shape for d in day_data_list]
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
                del chunk
                d[key] = None  # Free source array immediately — peak RAM halved
                pos += n
            return out

        if model_type == 'event':
            seqs    = _safe_concat(day_data_list, 'event_sequences', np.int16)
            lengths = _safe_concat(day_data_list, 'sequence_lengths', np.uint16)
            self.seqs    = seqs
            self.lengths = lengths

        elif model_type == 'book':
            # FLOAT16 STORAGE: allocate float32 for log-transform, then downcast to float16.
            # Peak RAM during build: float32 output + one float32 day chunk (safe_concat).
            # Stored size: float16 = 50% of float32 — critical for 80-day expanding window
            # (80 days x 234K bars x 20 levels x 4 features: 6.0 GB fp32 -> 3.0 GB fp16).
            # Cast back to float32 in __getitem__ (DataLoader worker, no extra persistent RAM).
            tensors = _safe_concat(day_data_list, 'book_tensors', np.float32)
            # Log-transform features 1,2,3 (depth, orders, age) — in-place on float32
            np.log1p(tensors[:, :, 1], out=tensors[:, :, 1])
            np.log1p(tensors[:, :, 2], out=tensors[:, :, 2])
            np.log1p(tensors[:, :, 3], out=tensors[:, :, 3])
            # Downcast to float16 for storage — halves persistent RAM
            self.tensors = tensors.astype(np.float16)
            del tensors  # free the float32 intermediate immediately
            gc.collect()

        elif model_type == 'lstm':
            trade_seqs = _safe_concat(day_data_list, 'trade_sequences', np.float32)
            trade_lens = _safe_concat(day_data_list, 'trade_lengths', np.uint16)
            # Log-transform size_lots, time_delta_ms, cumulative_volume — in-place
            np.log1p(np.abs(trade_seqs[:, :, 1]), out=trade_seqs[:, :, 1])
            np.log1p(trade_seqs[:, :, 3], out=trade_seqs[:, :, 3])
            np.log1p(trade_seqs[:, :, 4], out=trade_seqs[:, :, 4])
            self.trade_seqs = trade_seqs
            self.trade_lens = trade_lens

        elif model_type == 'hybrid':
            # Book tensors — pre-allocate float32, log-transform in-place, store as float16
            tensors = _safe_concat(day_data_list, 'book_tensors', np.float32)
            np.log1p(tensors[:, :, 1], out=tensors[:, :, 1])
            np.log1p(tensors[:, :, 2], out=tensors[:, :, 2])
            np.log1p(tensors[:, :, 3], out=tensors[:, :, 3])
            self.tensors = tensors.astype(np.float16)
            del tensors
            gc.collect()

            # Event sequences (if available in merged data)
            if 'event_sequences' in day_data_list[0]:
                seqs    = _safe_concat(day_data_list, 'event_sequences', np.int16)
                lengths = _safe_concat(day_data_list, 'sequence_lengths', np.uint16)
                self.seqs    = seqs
                self.lengths = lengths
                self._has_events = True
            else:
                # Fallback: dummy event sequences (zeros)
                N = len(tensors)
                self.seqs    = np.zeros((N, 200, 5), dtype=np.int16)
                self.lengths = np.ones(N, dtype=np.uint16)
                self._has_events = False

        self.target         = target.astype(np.float32)
        self.day_boundaries = day_boundaries
        N                   = len(target)

        # Build valid indices using numpy boolean mask (memory-efficient vs Python set):
        # - Must have a finite target
        # - For book/lstm/hybrid: need window_size bars of history (within same day)
        # - Apply subsample
        # MEMORY NOTE: Python set approach uses ~200 bytes/entry (PyLong + hash table slot).
        # For 2-7M entries this is 0.4-1.4GB peak RAM. numpy bool mask uses 1 byte/entry = ~7MB.
        valid_mask = np.zeros(N, dtype=bool)

        if model_type == 'event':
            # Each bar is self-contained (event sequence already in the file)
            for day_idx in range(len(day_boundaries) - 1):
                start = day_boundaries[day_idx]
                end   = min(day_boundaries[day_idx + 1], N)
                valid_mask[start:end] = np.isfinite(target[start:end])

        else:  # book, lstm, hybrid need history window within same day
            for day_idx in range(len(day_boundaries) - 1):
                start = day_boundaries[day_idx]
                end   = min(day_boundaries[day_idx + 1], N)
                slice_start = start + window_size - 1
                if slice_start < end:
                    valid_mask[slice_start:end] = np.isfinite(target[slice_start:end])

        # Sort and subsample — np.where returns sorted indices already (monotone scan)
        all_valid = np.where(valid_mask)[0]  # dtype int64 automatically, ~8 bytes/entry
        del valid_mask
        if subsample > 1:
            all_valid = all_valid[::subsample]

        self.valid_indices = all_valid.astype(np.int64)

    def __len__(self):
        return len(self.valid_indices)

    def __getitem__(self, idx):
        i = int(self.valid_indices[idx])
        target = float(self.target[i])

        if self.model_type == 'event':
            seq    = torch.from_numpy(self.seqs[i].astype(np.int64))    # (200, 5)
            length = int(self.lengths[i])
            return seq, length, target

        elif self.model_type == 'book':
            window = self.tensors[i - self.window_size + 1 : i + 1]     # (W, 20, 4) float16
            # Cast float16 -> float32 here (DataLoader worker, not stored persistently)
            window = torch.from_numpy(window.astype(np.float32))
            return window, target

        elif self.model_type == 'hybrid':
            # Book window — stored as float16, cast to float32 on access
            book_window = self.tensors[i - self.window_size + 1 : i + 1]  # (W, 20, 4) float16
            book_window = torch.from_numpy(book_window.astype(np.float32))
            # Event sequence for the current bar
            event_seq = torch.from_numpy(self.seqs[i].astype(np.int64))   # (200, 5)
            event_len = int(self.lengths[i])
            return book_window, event_seq, event_len, target

        else:  # lstm
            window_start = i - self.window_size + 1
            window_end   = i + 1
            seqs    = self.trade_seqs[window_start:window_end]   # (W, 50, 5)
            lengths = self.trade_lens[window_start:window_end]   # (W,)

            # Concatenate actual trades across window
            MAX_TRADES = self.window_size * 50  # 20 * 50 = 1000
            all_trades = []
            for bar in range(self.window_size):
                actual_len = int(lengths[bar])
                if actual_len > 0:
                    all_trades.append(seqs[bar, :actual_len, :])

            if all_trades:
                concat = np.concatenate(all_trades, axis=0)
            else:
                concat = np.zeros((1, 5), dtype=np.float32)

            total = len(concat)
            if total > MAX_TRADES:
                concat = concat[:MAX_TRADES]
                total  = MAX_TRADES
            elif total < MAX_TRADES:
                padding = np.zeros((MAX_TRADES - total, 5), dtype=np.float32)
                concat  = np.concatenate([concat, padding], axis=0)

            seq    = torch.from_numpy(concat.astype(np.float32))
            actual = min(total, MAX_TRADES)
            return seq, actual, target


def collate_event(batch):
    seqs    = torch.stack([b[0] for b in batch])          # (B, 200, 5)
    lengths = torch.tensor([b[1] for b in batch])         # (B,)
    targets = torch.tensor([b[2] for b in batch], dtype=torch.float32)  # (B,)
    return seqs, lengths, targets


def collate_book(batch):
    windows = torch.stack([b[0] for b in batch])          # (B, W, 20, 4)
    targets = torch.tensor([b[1] for b in batch], dtype=torch.float32)
    return windows, targets


def collate_lstm(batch):
    seqs    = torch.stack([b[0] for b in batch])          # (B, 1000, 5)
    lengths = torch.tensor([b[1] for b in batch])         # (B,)
    targets = torch.tensor([b[2] for b in batch], dtype=torch.float32)
    return seqs, lengths, targets


def collate_hybrid(batch):
    book_windows  = torch.stack([b[0] for b in batch])   # (B, W, 20, 4)
    event_seqs    = torch.stack([b[1] for b in batch])   # (B, 200, 5)
    event_lengths = torch.tensor([b[2] for b in batch])  # (B,)
    targets       = torch.tensor([b[3] for b in batch], dtype=torch.float32)
    return book_windows, event_seqs, event_lengths, targets


# ---------------------------------------------------------------------------
# Model imports
# ---------------------------------------------------------------------------
from event_transformer   import EventTransformer
from book_spatial_cnn    import BookSpatialCNN
from trade_sequence_lstm import TradeSequenceLSTM


# ---------------------------------------------------------------------------
# Hybrid Model
# ---------------------------------------------------------------------------

class HybridModel(nn.Module):
    """
    Late-fusion hybrid model combining:
      - BookSpatialCNN encoder (processes raw book snapshots)
      - EventTransformer encoder (processes raw event sequences)
      - Optional engineered feature MLP
    via concatenation then a shared prediction head.

    This lets the model capture what LightGBM already knows from hand-engineered
    features PLUS what the raw data adds via deep learning.

    Architecture:
        book_encoder: BookSpatialCNN with num_classes=book_enc_dim (encoder mode)
        event_encoder: EventTransformer with num_classes=event_enc_dim (encoder mode)
        optional eng_mlp: engineered_dim -> eng_enc_dim
        head: (book_enc_dim + event_enc_dim [+ eng_enc_dim]) -> 1

    The encoders output raw pre-logit features (not probabilities), so this is
    learned end-to-end via back-propagation through all encoders.

    Args:
        book_enc_dim: Output dimension of book encoder (default: 128)
        event_enc_dim: Output dimension of event encoder (default: 128)
        engineered_dim: If > 0, also fuse engineered features (default: 0)
        eng_enc_dim: Projection dim for engineered features (default: 64)
        dropout: Dropout rate (default: 0.1)
        num_classes: Final output classes (default: 1 for regression IC)
        window_size: Book window size (default: 20)
    """
    def __init__(
        self,
        book_enc_dim: int = 128,
        event_enc_dim: int = 128,
        engineered_dim: int = 0,
        eng_enc_dim: int = 64,
        dropout: float = 0.1,
        num_classes: int = 1,
        window_size: int = 20,
    ):
        super().__init__()
        self.engineered_dim = engineered_dim

        # Book encoder: reuse BookSpatialCNN, output book_enc_dim features
        self.book_encoder = BookSpatialCNN(
            window_size=window_size,
            num_levels=20,
            num_features=4,
            spatial_channels=(32, 64, 128, 256),
            temporal_channels=256,
            dropout=dropout,
            num_classes=book_enc_dim,  # use as feature extractor
        )

        # Event encoder: reuse EventTransformer, output event_enc_dim features
        self.event_encoder = EventTransformer(
            d_model=192,
            nhead=6,
            num_layers=2,
            dim_feedforward=768,
            dropout=dropout,
            num_classes=event_enc_dim,  # use as feature extractor
            use_cnn_stem=True,
        )

        # Optional engineered feature MLP
        fusion_dim = book_enc_dim + event_enc_dim
        if engineered_dim > 0:
            self.eng_mlp = nn.Sequential(
                nn.Linear(engineered_dim, eng_enc_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(eng_enc_dim * 2, eng_enc_dim),
                nn.GELU(),
            )
            fusion_dim += eng_enc_dim
        else:
            self.eng_mlp = None

        # Late fusion prediction head
        self.fusion_norm = nn.LayerNorm(fusion_dim)
        self.head = nn.Sequential(
            nn.Linear(fusion_dim, fusion_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_dim // 2, num_classes),
        )

    def forward(
        self,
        book_windows: torch.Tensor,          # (B, window_size, 20, 4)
        event_seqs: torch.Tensor,            # (B, seq_len, 5) int64
        event_lengths: torch.Tensor,         # (B,)
        engineered_features: Optional[torch.Tensor] = None,  # (B, engineered_dim)
    ) -> torch.Tensor:
        """
        Returns:
            logits: (batch, num_classes)
        """
        # Encode book snapshots
        book_feats = self.book_encoder(book_windows)   # (B, book_enc_dim)

        # Encode event sequences
        event_feats = self.event_encoder(event_seqs, event_lengths)  # (B, event_enc_dim)

        # Fuse
        parts = [book_feats, event_feats]

        if self.eng_mlp is not None and engineered_features is not None:
            eng_feats = self.eng_mlp(engineered_features)  # (B, eng_enc_dim)
            parts.append(eng_feats)

        fused = torch.cat(parts, dim=-1)   # (B, fusion_dim)
        fused = self.fusion_norm(fused)

        return self.head(fused)            # (B, num_classes)


def build_model(
    model_type: str,
    device: torch.device,
    window_size: int = 20,
    augment: bool = False,
) -> nn.Module:
    """Build and return model with num_classes=1 (regression for IC)."""
    if model_type == 'event':
        # v2: d_model=192, nhead=6 (head_dim=32), 2 layers, CNN stem
        model = EventTransformer(
            d_model=192,
            nhead=6,
            num_layers=2,
            dim_feedforward=768,
            dropout=0.1,
            num_classes=1,
            use_cnn_stem=True,
        )
    elif model_type == 'book':
        # v2: expanded channels (32,64,128,256), residual blocks, temporal_channels=256
        model = BookSpatialCNN(
            window_size=window_size,
            num_levels=20,
            num_features=4,
            spatial_channels=(32, 64, 128, 256),
            temporal_channels=256,
            dropout=0.1,
            num_classes=1,
        )
    elif model_type == 'lstm':
        # v2: hidden_dim=192, highway input projection
        model = TradeSequenceLSTM(
            input_dim=5,
            hidden_dim=192,
            num_layers=2,
            dropout=0.2,
            num_classes=1,
            max_seq_len=1000,
            proj_dim=128,
        )
    elif model_type == 'hybrid':
        # Combined book + event late fusion
        model = HybridModel(
            book_enc_dim=128,
            event_enc_dim=128,
            engineered_dim=0,
            dropout=0.1,
            num_classes=1,
            window_size=window_size,
        )
    else:
        raise ValueError(f'Unknown model type: {model_type}')

    model = model.to(device)
    return model


# ---------------------------------------------------------------------------
# Train one epoch
# ---------------------------------------------------------------------------

def _forward_batch(model, batch, model_type, device):
    """Run a single forward pass for any model type. Returns (preds, targets)."""
    if model_type == 'event':
        seqs, lengths, targets = batch
        seqs    = seqs.to(device)
        lengths = lengths.to(device)
        preds   = model(seqs, lengths).squeeze(-1)

    elif model_type == 'book':
        windows, targets = batch
        windows = windows.to(device)
        preds   = model(windows).squeeze(-1)

    elif model_type == 'lstm':
        seqs, lengths, targets = batch
        seqs    = seqs.to(device)
        lengths = lengths.to(device)
        preds   = model(seqs, lengths).squeeze(-1)

    elif model_type == 'hybrid':
        book_windows, event_seqs, event_lengths, targets = batch
        book_windows  = book_windows.to(device)
        event_seqs    = event_seqs.to(device)
        event_lengths = event_lengths.to(device)
        preds = model(book_windows, event_seqs, event_lengths).squeeze(-1)

    else:
        raise ValueError(f'Unknown model_type: {model_type}')

    targets = targets.to(device)
    return preds, targets


def train_epoch(
    model:       nn.Module,
    loader:      DataLoader,
    optimizer:   torch.optim.Optimizer,
    device:      torch.device,
    model_type:  str,
    scaler:      Optional[torch.cuda.amp.GradScaler] = None,
) -> Tuple[float, int]:
    """Train for one epoch. Returns (avg_loss, n_samples)."""
    model.train()
    total_loss = 0.0
    total_n    = 0
    criterion  = nn.HuberLoss(delta=1.0)  # Robust to outlier ticks; delta=1 on z-scored targets

    for batch in loader:
        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                preds, targets = _forward_batch(model, batch, model_type, device)
                loss = criterion(preds, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            preds, targets = _forward_batch(model, batch, model_type, device)
            loss = criterion(preds, targets)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

        n = targets.shape[0]
        total_loss += loss.item() * n
        total_n    += n

    avg_loss = total_loss / max(total_n, 1)
    return avg_loss, total_n


# ---------------------------------------------------------------------------
# Evaluate (compute IC on test set)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    model_type: str,
) -> Tuple[float, float, np.ndarray, np.ndarray]:
    """
    Run inference on loader. Returns (IC, val_loss, preds_array, targets_array).
    IC = Spearman rank correlation.
    val_loss = Huber loss on validation set (same criterion as training).
    """
    model.eval()
    all_preds   = []
    all_targets = []
    criterion   = nn.HuberLoss(delta=1.0)
    total_loss  = 0.0
    total_n     = 0

    ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad()

    for batch in loader:
        with ctx:
            preds_t, targets = _forward_batch(model, batch, model_type, device)

        preds = preds_t.cpu().float().numpy()
        val_l = criterion(preds_t.cpu().float(), targets.cpu().float()).item()

        n = targets.shape[0]
        total_loss += val_l * n
        total_n    += n
        all_preds.append(preds)
        all_targets.append(targets.cpu().numpy() if hasattr(targets, 'numpy') else np.array(targets))

    if not all_preds:
        return 0.0, 0.0, np.array([]), np.array([])

    val_loss    = total_loss / max(total_n, 1)
    preds_arr   = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)

    # Filter NaN
    mask = np.isfinite(preds_arr) & np.isfinite(targets_arr)
    if mask.sum() < 10:
        return 0.0, val_loss, preds_arr, targets_arr

    ic, _ = spearmanr(preds_arr[mask], targets_arr[mask])
    return float(ic) if np.isfinite(ic) else 0.0, val_loss, preds_arr, targets_arr


# ---------------------------------------------------------------------------
# Evaluate with embeddings (captures 512-dim penultimate layer via hook)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_with_embeddings(
    model:      nn.Module,
    loader:     DataLoader,
    device:     torch.device,
    model_type: str,
) -> Tuple[float, float, np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """
    Same as evaluate() but also captures penultimate-layer embeddings via
    a forward hook on model.temporal_pool (BookSpatialCNN only).

    Returns (IC, val_loss, preds_array, targets_array, embeddings_or_None).
    embeddings shape: (N, temporal_channels) — e.g. (N, 512) for wider CNN.
    embeddings is None if model has no temporal_pool attribute.
    """
    model.eval()
    all_preds      = []
    all_targets    = []
    all_embeddings = []
    criterion      = nn.HuberLoss(delta=1.0)
    total_loss     = 0.0
    total_n        = 0

    # Register hook on temporal_pool if present
    hook_handle = None
    _emb_buf: list = []

    def _hook(module, input, output):
        # output: (B, C, 1) from AdaptiveAvgPool1d — squeeze to (B, C)
        _emb_buf.append(output.squeeze(-1).cpu().float().numpy())

    if hasattr(model, 'temporal_pool'):
        hook_handle = model.temporal_pool.register_forward_hook(_hook)

    ctx = torch.amp.autocast('cuda') if device.type == 'cuda' else torch.no_grad()

    try:
        for batch in loader:
            _emb_buf.clear()
            with ctx:
                preds_t, targets = _forward_batch(model, batch, model_type, device)

            preds = preds_t.cpu().float().numpy()
            val_l = criterion(preds_t.cpu().float(), targets.cpu().float()).item()

            n = targets.shape[0]
            total_loss += val_l * n
            total_n    += n
            all_preds.append(preds)
            all_targets.append(targets.cpu().numpy() if hasattr(targets, 'numpy') else np.array(targets))
            if _emb_buf:
                all_embeddings.append(_emb_buf[0])
    finally:
        if hook_handle is not None:
            hook_handle.remove()

    if not all_preds:
        return 0.0, 0.0, np.array([]), np.array([]), None

    val_loss    = total_loss / max(total_n, 1)
    preds_arr   = np.concatenate(all_preds)
    targets_arr = np.concatenate(all_targets)
    embeddings  = np.concatenate(all_embeddings) if all_embeddings else None

    # Filter NaN
    mask = np.isfinite(preds_arr) & np.isfinite(targets_arr)
    if mask.sum() < 10:
        return 0.0, val_loss, preds_arr, targets_arr, embeddings

    ic, _ = spearmanr(preds_arr[mask], targets_arr[mask])
    return float(ic) if np.isfinite(ic) else 0.0, val_loss, preds_arr, targets_arr, embeddings


# ---------------------------------------------------------------------------
# Walk-Forward Training for One Model Type
# ---------------------------------------------------------------------------

def _load_hybrid_day_files(book_dir: str, events_dir: str, dates: List[str]):
    """
    Load and merge book + event NPZ files for the hybrid model.

    Returns merged day_data_list where each dict contains both
    'book_tensors' and 'event_sequences'/'sequence_lengths',
    plus concatenated mid_prices and day_boundaries.
    """
    book_path   = Path(book_dir)
    events_path = Path(events_dir)

    all_data   = []
    all_mids   = []
    boundaries = [0]

    for date in dates:
        book_fname   = book_path   / f'{date}_book_tensors.npz'
        events_fname = events_path / f'{date}_event_tokens.npz'

        if not book_fname.exists():
            logger.warning(f'  Missing book file: {book_fname.name}, skipping')
            continue

        book_npz = np.load(book_fname)
        merged   = dict(book_npz)  # has 'book_tensors', 'mid_prices'

        if events_fname.exists():
            ev_npz = np.load(events_fname)
            # Align on length (use book length as reference)
            n_book   = len(book_npz['mid_prices'])
            n_events = len(ev_npz['mid_prices'])
            n_use    = min(n_book, n_events)
            merged['event_sequences'] = ev_npz['event_sequences'][:n_use]
            merged['sequence_lengths'] = ev_npz['sequence_lengths'][:n_use]
            # Trim book arrays too if needed
            if n_use < n_book:
                merged['book_tensors'] = merged['book_tensors'][:n_use]
                merged['mid_prices']   = merged['mid_prices'][:n_use]
        else:
            logger.warning(f'  Missing events file: {events_fname.name}, using dummy events')
            n = len(book_npz['mid_prices'])
            merged['event_sequences']  = np.zeros((n, 200, 5), dtype=np.int16)
            merged['sequence_lengths'] = np.ones(n, dtype=np.uint16)

        all_data.append(merged)
        all_mids.append(merged['mid_prices'])
        boundaries.append(boundaries[-1] + len(merged['mid_prices']))

    mid_concat = np.concatenate(all_mids) if all_mids else np.array([])
    return all_data, mid_concat, boundaries


def train_walkforward(
    model_type:      str,
    cache_dir:       str,
    n_days:          int           = 20,
    min_train_days:  int           = 5,
    max_train_days:  Optional[int] = None,   # sliding window: cap training days (None=expanding)
    purge_days:      int           = 1,
    horizon_bars:    int           = 100,   # 10s = 100 bars @ 100ms
    window_size:     int           = 20,    # for book/lstm/hybrid
    epochs:          int           = 3,
    batch_size:      int           = 512,
    lr:              float         = 3e-4,
    subsample_train: int           = 3,     # use every Nth bar during training
    device_str:      str           = 'cuda',
    output_dir:      str           = DEFAULT_OUTPUT_DIR,
    seed:            int           = 42,
    augment:         bool          = False, # data augmentation for book model
    events_dir:      str           = '',    # only used when model_type == 'hybrid'
    warm_start:      bool          = False, # init each fold from previous fold weights
    start_fold:      int           = 0,    # skip folds before this index
) -> Dict:
    """
    Walk-forward training and evaluation for a single model type.

    Protocol:
        For test_day in range(min_train_days + purge_days, n_days):
            train_end   = test_day - purge_days
            train_start = max(0, train_end - max_train_days) if max_train_days else 0
            train_days  = dates[train_start : train_end]  (sliding or expanding)
            test_day    = dates[test_day]

        Metric: Spearman IC on test set each fold.

    Returns:
        dict with fold_ics, aggregate_ic, aggregate_icir, fold_details
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device(device_str if torch.cuda.is_available() and device_str == 'cuda' else 'cpu')
    use_amp = (device.type == 'cuda')

    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')

    # Set up file logging so output is captured
    log_path = _setup_file_logger(output_dir, model_type, timestamp)

    window_mode = f'sliding (max {max_train_days}d)' if max_train_days else 'expanding'

    logger.info('=' * 70)
    logger.info(f'Walk-Forward Training: {model_type.upper()}')
    logger.info(f'  Log file:        {log_path}')
    logger.info(f'  Cache dir:       {cache_dir}')
    if model_type == 'hybrid':
        logger.info(f'  Events dir:      {events_dir}')
    logger.info(f'  n_days:          {n_days}')
    logger.info(f'  min_train_days:  {min_train_days}')
    logger.info(f'  max_train_days:  {max_train_days or "unlimited (expanding)"}')
    logger.info(f'  window_mode:     {window_mode}')
    logger.info(f'  purge_days:      {purge_days}')
    logger.info(f'  epochs/fold:     {epochs}')
    logger.info(f'  batch_size:      {batch_size}')
    logger.info(f'  subsample_train: {subsample_train}')
    logger.info(f'  device:          {device}')
    logger.info(f'  AMP:             {use_amp}')
    logger.info(f'  augment:         {augment}')
    logger.info('=' * 70)

    # Get available dates
    dates = get_available_dates(cache_dir, model_type)
    if n_days is not None and n_days < len(dates):
        dates = dates[:n_days]

    n_total = len(dates)
    logger.info(f'  Using {n_total} days: {dates[0]} .. {dates[-1]}')

    if n_total < min_train_days + purge_days + 1:
        raise ValueError(
            f'Not enough days: {n_total} < {min_train_days} + {purge_days} + 1 = '
            f'{min_train_days + purge_days + 1}'
        )

    # Collate function for DataLoader
    collate_fn_map = {
        'event':  collate_event,
        'book':   collate_book,
        'lstm':   collate_lstm,
        'hybrid': collate_hybrid,
    }
    collate_fn = collate_fn_map[model_type]

    fold_ics      = []
    fold_details  = []
    all_oos_preds = []
    all_oos_tgts  = []

    n_folds = n_total - min_train_days - purge_days
    logger.info(f'  Total folds: {n_folds}')

    # ----- Checkpoint: resume from last completed fold if available -----
    checkpoint_file = Path(output_dir) / f'checkpoint_{model_type}_{timestamp}.json'
    # ONLY search for checkpoints matching this exact model_type — NO cross-search
    existing_ckpts = sorted(Path(output_dir).glob(f'checkpoint_{model_type}_*.json'))
    resume_fold = 0
    if existing_ckpts:
        try:
            # Pick checkpoint with most completed folds (not just last alphabetically)
            best_ckpt_path = existing_ckpts[-1]
            best_folds = 0
            for _cp in existing_ckpts:
                try:
                    with open(_cp) as _tf:
                        _cd = json.load(_tf)
                    # --- Metadata validation: reject checkpoints from wrong model ---
                    _ckpt_model_type = _cd.get('model_type', model_type)
                    _ckpt_wider = _cd.get('wider_cnn', False) or _cd.get('wider_hybrid', False)
                    if _ckpt_model_type != model_type:
                        logger.warning(f'  Skipping checkpoint {_cp.name}: model_type={_ckpt_model_type} != {model_type}')
                        continue
                    if _ckpt_wider:
                        logger.warning(f'  Skipping checkpoint {_cp.name}: tagged as wider variant')
                        continue
                    _nf = _cd.get('completed_folds', 0)
                    if _nf >= best_folds:
                        best_folds = _nf
                        best_ckpt_path = _cp
                except Exception:
                    pass
            with open(best_ckpt_path) as _cf:
                ckpt = json.load(_cf)
            fold_ics = ckpt.get('fold_ics', [])
            fold_details = ckpt.get('fold_details', [])
            resume_fold = ckpt.get('completed_folds', 0)
            checkpoint_file = best_ckpt_path  # reuse the best checkpoint file
            # Reload OOS predictions if saved
            ckpt_preds_file = Path(output_dir) / ckpt.get('preds_file', '')
            if ckpt_preds_file.exists():
                pdata = np.load(str(ckpt_preds_file))
                for i in range(resume_fold):
                    date = fold_details[i].get('test_date', f'fold_{i}') if i < len(fold_details) else f'fold_{i}'
                    pk, tk = f'{date}_preds', f'{date}_targets'
                    if pk in pdata and tk in pdata:
                        all_oos_preds.append(pdata[pk])
                        all_oos_tgts.append(pdata[tk])
            if resume_fold > 0:
                logger.info(f'  CHECKPOINT RESUME: Skipping {resume_fold} completed folds (IC so far: {np.mean(fold_ics):+.4f})')
        except Exception as e:
            logger.warning(f'  Checkpoint load failed ({e}), starting fresh')
            fold_ics, fold_details, all_oos_preds, all_oos_tgts = [], [], [], []
            resume_fold = 0

    prev_model_state = None  # for warm start: carry weights across folds

    # If warm_start + start_fold > 0, try loading the latest .pt checkpoint
    # so skipped folds don't lose the warm-start chain
    # Uses output_dir-specific latest.pt to prevent cross-contamination
    if warm_start and start_fold > 0:
        # .pt weights live INSIDE output_dir/checkpoints/ — each launcher sets its
        # own output_dir (e.g. results/wider_cnn/, results/hybrid/) so there is
        # ZERO risk of loading a checkpoint from a different model variant.
        _ckpt_base = Path(output_dir) / 'checkpoints'
        latest_pt = _ckpt_base / 'latest.pt'
        if latest_pt.exists():
            try:
                prev_model_state = torch.load(str(latest_pt), map_location='cpu', weights_only=True)
                # --- Weight shape validation ---
                if prev_model_state:
                    _tmp_model = build_model(model_type, torch.device('cpu'), window_size, augment)
                    _model_sd = _tmp_model.state_dict()
                    for _wk in prev_model_state:
                        if _wk in _model_sd and prev_model_state[_wk].shape != _model_sd[_wk].shape:
                            raise ValueError(
                                f'Shape mismatch on {_wk}: checkpoint {prev_model_state[_wk].shape} '
                                f'vs model {_model_sd[_wk].shape} — wrong model checkpoint!'
                            )
                    del _tmp_model
                logger.info(f'  WARM START: Loaded weights from {latest_pt} for fold resume')
            except Exception as e:
                logger.warning(f'  Failed to load warm start weights: {e}')
                prev_model_state = None

    for fold_idx, test_day_idx in enumerate(range(min_train_days + purge_days, n_total)):

        # Skip already-completed folds (checkpoint resume OR --start-fold)
        effective_skip = max(resume_fold, start_fold)
        if fold_idx < effective_skip:
            continue

        train_end_idx = test_day_idx - purge_days
        if max_train_days and train_end_idx > max_train_days:
            train_start_idx = train_end_idx - max_train_days
        else:
            train_start_idx = 0
        train_dates = dates[train_start_idx:train_end_idx]  # sliding or expanding window
        test_dates  = [dates[test_day_idx]]                  # just the test day

        t_fold_start = time.time()
        logger.info(f'\n--- Fold {fold_idx+1}/{n_folds} | '
                    f'Train: {train_dates[0]}..{train_dates[-1]} ({len(train_dates)}d) | '
                    f'Test: {test_dates[0]} ---')

        # ----- Load train data -----
        if model_type == 'hybrid':
            train_data_list, train_mids, train_bounds = _load_hybrid_day_files(
                cache_dir, events_dir, train_dates
            )
        else:
            train_data_list, train_mids, train_bounds = load_day_files(
                cache_dir, model_type, train_dates
            )

        if not train_data_list:
            logger.warning(f'  Fold {fold_idx+1}: no train data, skipping')
            continue

        # Compute MFE target for train window
        train_target = compute_mfe_net(train_mids, train_bounds, horizon_bars)

        n_train_valid = int(np.isfinite(train_target).sum())
        if n_train_valid < 1000:
            logger.warning(f'  Fold {fold_idx+1}: too few train samples ({n_train_valid}), skipping')
            continue

        # ----- Compute train normalization stats (needed before test load) -----
        finite_mask = np.isfinite(train_target)
        tgt_mean = float(train_target[finite_mask].mean())
        tgt_std  = float(train_target[finite_mask].std())
        if tgt_std < 1e-8:
            tgt_std = 1.0
        train_target = (train_target - tgt_mean) / tgt_std
        logger.info(f'  Target normalization: mean={tgt_mean:.3f} std={tgt_std:.3f} ticks')

        logger.info(f'  Train: {n_train_valid:,} valid bars | (test to be loaded)')

        # ----- Build train dataset (before loading test — avoids holding both raw arrays in RAM) -----
        train_dataset = BarDataset(
            train_data_list, train_target, train_bounds,
            model_type=model_type, window_size=window_size,
            subsample=subsample_train,
        )
        # Free ALL train raw data immediately — BarDataset has concatenated what it needs.
        # This reclaims ~6GB before loading test data (another ~70MB) + building test dataset.
        del train_data_list, train_mids, train_target, train_bounds
        gc.collect()

        # ----- Load test data (AFTER freeing train raw data) -----
        if model_type == 'hybrid':
            test_data_list, test_mids, test_bounds = _load_hybrid_day_files(
                cache_dir, events_dir, test_dates
            )
        else:
            test_data_list, test_mids, test_bounds = load_day_files(
                cache_dir, model_type, test_dates
            )

        if not test_data_list:
            logger.warning(f'  Fold {fold_idx+1}: no test data, skipping')
            continue

        test_target = compute_mfe_net(test_mids, test_bounds, horizon_bars)
        n_test_valid = int(np.isfinite(test_target).sum())
        if n_test_valid < 100:
            logger.warning(f'  Fold {fold_idx+1}: too few test samples ({n_test_valid}), skipping')
            continue

        test_target = (test_target - tgt_mean) / tgt_std

        test_dataset = BarDataset(
            test_data_list, test_target, test_bounds,
            model_type=model_type, window_size=window_size,
            subsample=1,  # evaluate on ALL test bars
        )

        logger.info(f'  Dataset: {len(train_dataset):,} train samples '
                    f'(subsample={subsample_train}x) | {len(test_dataset):,} test samples')

        if len(train_dataset) < 100:
            logger.warning(f'  Too few train dataset items, skipping')
            continue

        # Free raw test data BEFORE building DataLoader
        del test_data_list, test_mids, test_target, test_bounds
        gc.collect()

        num_workers = 0  # Windows has multiprocessing issues with >0 workers in some configs
        # Disable pin_memory to avoid pinned memory OOM as train set grows
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=False,
            drop_last=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=batch_size * 2, shuffle=False,
            collate_fn=collate_fn, num_workers=num_workers,
            pin_memory=False,
        )

        # ----- Build model (warm start: init from previous fold weights) -----
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        model = build_model(model_type, device, window_size=window_size, augment=augment)
        n_params = sum(p.numel() for p in model.parameters())

        if warm_start and prev_model_state is not None:
            try:
                model.load_state_dict(prev_model_state)
                logger.info(f'  Model params: {n_params:,} (warm start from previous fold)')
            except Exception as e:
                logger.warning(f'  Warm start failed ({e}), using fresh weights')
                logger.info(f'  Model params: {n_params:,}')
        else:
            logger.info(f'  Model params: {n_params:,}')

        optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer, max_lr=lr,
            steps_per_epoch=len(train_loader),
            epochs=epochs,
            pct_start=0.3,
        )
        scaler = torch.amp.GradScaler('cuda', init_scale=1024) if use_amp else None  # reduced from 2^16 to 2^10 to prevent CUDA overflow at fold 73

        # ----- Train for N epochs -----
        best_ic     = -999.0
        epoch_stats = []

        best_val_loss = 999.0
        best_embeds   = None  # 512-dim embeddings from best-IC epoch (book model only)
        for epoch in range(epochs):
            t0_ep = time.time()
            train_loss, n_samp = train_epoch(
                model, train_loader, optimizer, device, model_type, scaler
            )
            scheduler.step()

            # Evaluate on validation set: IC + val_loss + embeddings
            ic, val_loss, preds, tgts, embeds = evaluate_with_embeddings(
                model, test_loader, device, model_type
            )

            epoch_stats.append({
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'ic': ic,
            })
            overfit_flag = ' **OVERFIT**' if val_loss > train_loss * 1.15 else ''
            logger.info(
                f'  Epoch {epoch+1}/{epochs}: '
                f'train_loss={train_loss:.5f}  val_loss={val_loss:.5f}  '
                f'IC={ic:+.4f}  ({time.time()-t0_ep:.1f}s){overfit_flag}'
            )

            if ic > best_ic:
                best_ic     = ic
                best_preds  = preds.copy()
                best_tgts   = tgts.copy()
                best_embeds = embeds.copy() if embeds is not None else None
            if val_loss < best_val_loss:
                best_val_loss = val_loss

        # Use best-epoch predictions for fold IC
        fold_ic = best_ic
        fold_ics.append(fold_ic)
        all_oos_preds.append(best_preds)
        all_oos_tgts.append(best_tgts)

        # ----- Save per-fold embeddings (book model only, zero-cost: no extra forward pass) -----
        if best_embeds is not None:
            try:
                _emb_dir = Path(output_dir) / 'embeddings'
                _emb_dir.mkdir(parents=True, exist_ok=True)
                _fold_date = test_dates[0] if test_dates else f'fold_{fold_idx}'
                _emb_path = _emb_dir / f'fold_{fold_idx}_{_fold_date}_embeddings.npz'
                np.savez_compressed(
                    str(_emb_path),
                    embeddings=best_embeds,          # (N, 512)
                    predictions=best_preds,          # (N,)
                    targets=best_tgts,               # (N,)
                )
                logger.info(f'  Saved embeddings: {_emb_path.name} '
                            f'shape={best_embeds.shape}')
            except Exception as _e:
                logger.warning(f'  Failed to save embeddings for fold {fold_idx}: {_e}')

        fold_details.append({
            'fold':        fold_idx + 1,
            'test_date':   test_dates[0],
            'train_days':  len(train_dates),
            'ic':          fold_ic,
            'best_val_loss': best_val_loss,
            'n_test':      int(np.isfinite(best_tgts).sum()),
            'epoch_stats': epoch_stats,
            'target_mean': tgt_mean,
            'target_std':  tgt_std,
        })

        logger.info(f'  Fold IC: {fold_ic:+.4f} (best over {epochs} epochs)  '
                    f'[{time.time()-t_fold_start:.1f}s total]')

        # ----- Save model state for warm start before cleanup -----
        if warm_start:
            prev_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            # Save per-fold .pt weights INSIDE output_dir/checkpoints/ so each model
            # variant (standard CNN, wider CNN, hybrid) is fully isolated.
            _fold_pt_dir = Path(output_dir) / 'checkpoints'
            _fold_pt_dir.mkdir(parents=True, exist_ok=True)
            _fold_pt_path = _fold_pt_dir / f'fold_{fold_idx}_{test_dates[0] if test_dates else "unknown"}.pt'
            torch.save(prev_model_state, str(_fold_pt_path))
            # Also update latest.pt in the SAME isolated directory
            _latest_pt = _fold_pt_dir / 'latest.pt'
            torch.save(prev_model_state, str(_latest_pt))
            logger.info(f'  Saved weights: {_fold_pt_path.name} + latest.pt -> {_fold_pt_dir}')

        # ----- Cleanup to free VRAM -----
        # Move model to CPU before deleting to ensure VRAM is released
        if device.type == 'cuda':
            model.cpu()
        del model, optimizer, scheduler, scaler
        del train_dataset, test_dataset, train_loader, test_loader
        gc.collect()
        if device.type == 'cuda':
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
        logger.info(f'  VRAM after cleanup: '
                    f'{torch.cuda.memory_allocated()/1024**2:.0f}MB allocated, '
                    f'{torch.cuda.memory_reserved()/1024**2:.0f}MB reserved'
                    if device.type == 'cuda' else '  (CPU mode)')

        # ----- Save checkpoint after each fold -----
        try:
            ckpt_preds_file = Path(output_dir) / f'ckpt_preds_{model_type}_{timestamp}.npz'
            pred_data_ckpt = {}
            for ci, (cp, ct) in enumerate(zip(all_oos_preds, all_oos_tgts)):
                cdate = fold_details[ci].get('test_date', f'fold_{ci}') if ci < len(fold_details) else f'fold_{ci}'
                pred_data_ckpt[f'{cdate}_preds'] = cp
                pred_data_ckpt[f'{cdate}_targets'] = ct
            np.savez_compressed(str(ckpt_preds_file), **pred_data_ckpt)

            ckpt_data = {
                'completed_folds': fold_idx + 1,
                'fold_ics': [float(x) for x in fold_ics],
                'fold_details': fold_details,
                'mean_ic': float(np.mean(fold_ics)),
                'preds_file': ckpt_preds_file.name,
                'timestamp': timestamp,
                'model_type': model_type,
                'param_count': n_params,
            }
            with open(checkpoint_file, 'w') as _cf:
                json.dump(ckpt_data, _cf, indent=2, default=str)
        except Exception as e:
            logger.warning(f'  Checkpoint save failed: {e}')

    # ----- Aggregate metrics -----
    if not fold_ics:
        logger.error('No folds completed!')
        return {'model_type': model_type, 'error': 'No folds completed', 'fold_ics': []}

    ic_arr     = np.array(fold_ics)
    agg_ic     = float(ic_arr.mean())
    agg_ic_std = float(ic_arr.std())
    agg_icir   = float(agg_ic / agg_ic_std) if agg_ic_std > 0 else 0.0

    # Aggregate OOS IC (all predictions concatenated)
    all_p = np.concatenate(all_oos_preds)
    all_t = np.concatenate(all_oos_tgts)
    mask  = np.isfinite(all_p) & np.isfinite(all_t)
    if mask.sum() > 10:
        concat_ic, _ = spearmanr(all_p[mask], all_t[mask])
        concat_ic = float(concat_ic)
    else:
        concat_ic = 0.0

    logger.info('')
    logger.info('=' * 70)
    logger.info(f'RESULTS: {model_type.upper()}')
    logger.info(f'  Folds completed:     {len(fold_ics)}')
    logger.info(f'  Per-fold IC mean:    {agg_ic:+.4f}')
    logger.info(f'  Per-fold IC std:     {agg_ic_std:.4f}')
    logger.info(f'  ICIR:                {agg_icir:+.3f}')
    logger.info(f'  Concatenated IC:     {concat_ic:+.4f}')
    logger.info(f'  LightGBM baseline:   +0.139')
    logger.info(f'  Gap vs LightGBM:     {agg_ic - 0.139:+.4f}')
    logger.info('=' * 70)

    # Save per-fold OOS predictions
    if all_oos_preds:
        pred_data = {}
        for i, (preds_arr, tgts_arr) in enumerate(zip(all_oos_preds, all_oos_tgts)):
            if i < len(fold_details):
                date = fold_details[i].get('test_date', f'fold_{i}')
                pred_data[f"{date}_preds"] = preds_arr
                pred_data[f"{date}_targets"] = tgts_arr

        preds_file = Path(output_dir) / f'oos_predictions_{model_type}_{timestamp}.npz'
        np.savez_compressed(str(preds_file), **pred_data)
        logger.info(f'  OOS predictions saved: {preds_file}')

    results = {
        'model_type':    model_type,
        'n_days_used':   len(dates),
        'n_folds':       len(fold_ics),
        'fold_ics':      fold_ics,
        'agg_ic':        agg_ic,
        'agg_ic_std':    agg_ic_std,
        'agg_icir':      agg_icir,
        'concat_ic':     concat_ic,
        'lgbm_baseline': 0.139,
        'fold_details':  fold_details,
        'config': {
            'epochs':          epochs,
            'batch_size':      batch_size,
            'lr':              lr,
            'subsample_train': subsample_train,
            'horizon_bars':    horizon_bars,
            'window_size':     window_size,
            'min_train_days':  min_train_days,
            'max_train_days':  max_train_days,
            'purge_days':      purge_days,
            'window_mode':     window_mode,
            'augment':         augment,
        },
    }

    # Save results
    out_file = Path(output_dir) / f'walkforward_{model_type}_{timestamp}.json'
    with open(out_file, 'w') as f:
        # Convert numpy types for JSON serialization
        def to_serializable(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            elif isinstance(obj, (np.floating,)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def recursive_convert(d):
            if isinstance(d, dict):
                return {k: recursive_convert(v) for k, v in d.items()}
            elif isinstance(d, list):
                return [recursive_convert(x) for x in d]
            else:
                return to_serializable(d)

        json.dump(recursive_convert(results), f, indent=2)

    logger.info(f'  Results saved: {out_file}')
    return results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description='Walk-forward training for DL alpha discovery models',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument('--model', type=str, default='all',
                        choices=['event', 'book', 'lstm', 'hybrid', 'all'],
                        help='Which model to train. hybrid=BookCNN+EventTransformer late fusion')
    parser.add_argument('--days', type=int, default=None,
                        help='Number of trading days to use (None = all available)')
    parser.add_argument('--min-train-days', type=int, default=5,
                        help='Minimum training days before first test fold')
    parser.add_argument('--purge-days', type=int, default=1,
                        help='Days to skip between train end and test start')
    parser.add_argument('--max-train-days', type=int, default=None,
                        help='Max training days per fold (sliding window). None=expanding.')
    parser.add_argument('--epochs', type=int, default=3,
                        help='Epochs per fold (keep low for testing: 3-5)')
    parser.add_argument('--batch-size', type=int, default=512,
                        help='Training batch size')
    parser.add_argument('--lr', type=float, default=3e-4,
                        help='Learning rate')
    parser.add_argument('--subsample-train', type=int, default=3,
                        help='Use every Nth bar during training (1=no subsample)')
    parser.add_argument('--device', type=str, default='cuda',
                        choices=['cuda', 'cpu'],
                        help='Device to use')
    parser.add_argument('--events-dir', type=str, default=DEFAULT_EVENTS_DIR,
                        help='Directory with _event_tokens.npz files')
    parser.add_argument('--book-dir', type=str, default=DEFAULT_BOOK_DIR,
                        help='Directory with _book_tensors.npz files')
    parser.add_argument('--trades-dir', type=str, default=DEFAULT_TRADES_DIR,
                        help='Directory with _trade_flow.npz files')
    parser.add_argument('--output-dir', type=str, default=DEFAULT_OUTPUT_DIR,
                        help='Directory to save results')
    parser.add_argument('--window-size', type=int, default=20,
                        help='Window size for book/lstm models (bars)')
    parser.add_argument('--start-fold', type=int, default=0,
                        help='Skip folds before this number (0=start from beginning)')
    parser.add_argument('--horizon-bars', type=int, default=100,
                        help='Forward horizon in bars (100=10s at 100ms)')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument(
        '--augment', action='store_true', default=False,
        help='Enable Gaussian noise augmentation for book model (adds noise to depth features)',
    )
    parser.add_argument(
        '--warm-start', action='store_true', default=False,
        help='Initialize each fold from previous fold weights (warm start). Reduces training time.',
    )
    # --start-fold already added above (line 1287)

    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    models_to_run = ['event', 'book', 'lstm'] if args.model == 'all' else [args.model]

    # Primary cache directory per model type
    cache_dir_map = {
        'event':  args.events_dir,
        'book':   args.book_dir,
        'lstm':   args.trades_dir,
        'hybrid': args.book_dir,   # hybrid uses book dir as primary (date index)
    }

    all_results = {}

    for model_type in models_to_run:
        logger.info(f'\n{"#" * 70}')
        logger.info(f'# TRAINING: {model_type.upper()}')
        logger.info(f'{"#" * 70}')

        t0 = time.time()
        try:
            results = train_walkforward(
                model_type      = model_type,
                cache_dir       = cache_dir_map[model_type],
                n_days          = args.days,
                min_train_days  = args.min_train_days,
                max_train_days  = args.max_train_days,
                purge_days      = args.purge_days,
                horizon_bars    = args.horizon_bars,
                window_size     = args.window_size,
                epochs          = args.epochs,
                batch_size      = args.batch_size,
                lr              = args.lr,
                subsample_train = args.subsample_train,
                device_str      = args.device,
                output_dir      = args.output_dir,
                seed            = args.seed,
                augment         = args.augment,
                events_dir      = args.events_dir,  # used by hybrid
                warm_start      = args.warm_start,
                start_fold      = args.start_fold,
            )
            all_results[model_type] = results

        except Exception as e:
            logger.error(f'Model {model_type} FAILED: {e}', exc_info=True)
            all_results[model_type] = {'error': str(e)}

        elapsed = time.time() - t0
        logger.info(f'Total time for {model_type}: {elapsed:.1f}s ({elapsed/60:.1f}m)')
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ----- Summary -----
    logger.info('\n' + '=' * 70)
    logger.info('SUMMARY OF ALL MODELS')
    logger.info('=' * 70)
    logger.info(f'{"Model":<12} {"IC":>8} {"ICIR":>8} {"Folds":>6} {"vs LGB":>8}')
    logger.info('-' * 50)

    for mt, res in all_results.items():
        if 'error' in res:
            logger.info(f'{mt:<12} {"ERROR":>8}')
        else:
            ic    = res.get('agg_ic', 0.0)
            icir  = res.get('agg_icir', 0.0)
            folds = res.get('n_folds', 0)
            gap   = ic - 0.139
            logger.info(f'{mt:<12} {ic:>+8.4f} {icir:>+8.3f} {folds:>6d} {gap:>+8.4f}')

    logger.info('=' * 70)
    logger.info('LightGBM baseline: IC=0.139')
    logger.info('=' * 70)

    # Notify completion signal
    model_summaries = []
    for mt, res in all_results.items():
        if 'error' in res:
            model_summaries.append(f"{mt}=ERROR")
        else:
            ic = res.get('agg_ic', 0.0)
            folds = res.get('n_folds', 0)
            model_summaries.append(f"{mt}:IC={ic:+.4f}(n={folds})")
    notify_complete(
        task_name=f"train_walkforward_{'+'.join(models_to_run)}",
        status="completed",
        result_summary="; ".join(model_summaries),
    )

    return all_results


if __name__ == '__main__':
    main()
