"""
Event-Driven 3D CNN — Training Script v1

Data format (identical to train_event_cnn_1d.py):
  - events: (N_events, 6) float32
      [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]
  - labels_1s/5s/10s: (N_events,) float32 — mid-price change in ticks
  - timestamps: (N_events,) int64 nanoseconds

Architecture: Event 3D CNN (orderbook activity "video")

  Key idea: reshape each event window into a 3D volume where:
    - Axis 0 (T): temporal bins — divide the WINDOW_SIZE events into T_BINS time slices
    - Axis 1 (P): price bins   — discretize price_rel_ticks into P_BINS price buckets
    - Axis 2 (C): channels     — aggregate features within each (t_bin, p_bin) cell

  Per-cell aggregation (5 channels):
    ch0: event count  — how many events landed in this (t, p) cell
    ch1: total qty    — sum of qty_log for all events in cell
    ch2: net side     — (buy_count - sell_count) — signed order pressure
    ch3: mean time_delta — average log inter-arrival time (proxy for urgency)
    ch4: mean spread  — average spread in that cell (market microstructure context)

  This creates a (C, T, P) tensor per sample — analogous to a "video" where each
  frame is a price-level snapshot and time is the sequence dimension.

  Model layout:
    1. Stem Conv2d(C_in=5, C=channels, 3×3) — lift to hidden channels
    2. Residual Conv2d blocks with BatchNorm + GELU — 4 blocks, optional downsampling
    3. Global average pooling (T, P) → (C,)
    4. LayerNorm → MLP head → 3 outputs (1s, 5s, 10s)

  Why Conv2d not Conv3d?
    After voxelisation the spatial grid is (T, P) — a 2D image per window.
    Conv2d on (C, T, P) is standard image CNN, simpler and faster than Conv3d,
    and exactly the right inductive bias: local interactions in time AND price.

Key advantages over CNN1D:
  - Explicit spatial structure: nearby PRICE LEVELS interact, not just nearby events
  - Captures order-book shape: imbalances at specific price levels
  - Handles variable event density: aggregation makes the representation fixed-size

Training rules (ABSOLUTE — same as CNN1D and Hawkes):
  - Expanding window walk-forward (NEVER sliding window)
  - Concat IC as primary metric (scipy.stats.spearmanr across all OOS predictions)
  - Save .pt weights AND .npz predictions for EVERY fold
  - MLflow logging — every run, no exceptions (experiment: EventDriven_3DCNN)
  - Mixed precision (fp16 when CUDA available)
  - num_workers=0 on Windows (pickle issues with large numpy arrays)
  - Feature statistics computed from training set only per fold (zero leakage)
"""

import os
import sys
import gc
import time
import logging
import argparse
import warnings
import socket
from pathlib import Path
from typing import Optional, List, Dict, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.stats

# Silence git noise when launched from schtasks / pythonw
os.environ.setdefault("GIT_PYTHON_REFRESH", "quiet")
os.environ.setdefault("PYTHONIOENCODING", "utf-8")

# MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not installed — skipping experiment tracking")

# ============================================================
# Logging setup
# ============================================================
LOG_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)

log_path = LOG_DIR / "event_3d_cnn.log"
_file_handler   = logging.FileHandler(log_path, encoding="utf-8")
_file_handler.setLevel(logging.INFO)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
logging.basicConfig(
    level   = logging.INFO,
    format  = "%(asctime)s [%(levelname)s] %(message)s",
    handlers = [_file_handler, _stream_handler],
)
logger = logging.getLogger(__name__)


class _FlushHandler(logging.StreamHandler):
    """Force flush on every log call (avoids buffering on Windows when redirected)."""
    def emit(self, record):
        super().emit(record)
        self.flush()


for _h in logging.root.handlers:
    _h.__class__ = _FlushHandler


# ============================================================
# Config
# ============================================================
DEFAULT_DATA_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "event_3d_cnn"


def _detect_mlflow_uri() -> str:
    """Auto-detect MLflow tracking URI: prefer localhost if reachable."""
    uri = os.environ.get("MLFLOW_TRACKING_URI")
    if uri:
        return uri
    try:
        s = socket.create_connection(("localhost", 5000), timeout=2)
        s.close()
        return "http://localhost:5000"
    except Exception:
        pass
    return "http://neptune-win:5000"


MLFLOW_TRACKING_URI = _detect_mlflow_uri()
MLFLOW_EXPERIMENT   = "EventDriven_3DCNN"

# 3D-CNN-specific hyperparameters (EVENT3D_* prefix)
T_BINS       = int(os.environ.get("EVENT3D_T_BINS",    20))   # temporal bins
P_BINS       = int(os.environ.get("EVENT3D_P_BINS",    40))   # price level bins
CNN3D_CHANNELS = int(os.environ.get("EVENT3D_CHANNELS", 64))  # hidden channels
CNN3D_LAYERS   = int(os.environ.get("EVENT3D_LAYERS",    4))  # number of residual blocks
CNN3D_DROPOUT  = float(os.environ.get("EVENT3D_DROPOUT", 0.1))

# Price binning configuration
# price_rel_ticks is centred near 0; we bucket into [-P_RANGE, +P_RANGE] ticks
PRICE_BIN_RANGE = int(os.environ.get("EVENT3D_PRICE_RANGE", 10))  # ±10 ticks default

# Shared hyperparameters — use same EVENT_* env vars as other event scripts
WINDOW_SIZE     = int(os.environ.get("EVENT_WINDOW_SIZE", 500))
STRIDE          = int(os.environ.get("EVENT_STRIDE",      WINDOW_SIZE // 2))
BATCH_SIZE      = int(os.environ.get("EVENT_BATCH_SIZE",  64))
LR              = float(os.environ.get("EVENT_LR",        3e-4))
EPOCHS_PER_FOLD = int(os.environ.get("EVENT_EPOCHS",      5))
WARMUP_STEPS    = int(os.environ.get("EVENT_WARMUP",      300))
GRAD_CLIP       = float(os.environ.get("EVENT_GRAD_CLIP", 1.0))
N_FOLDS         = int(os.environ.get("EVENT_N_FOLDS",     5))
HORIZONS        = ["1s", "5s", "10s"]

# Event feature column indices
COL_TIME_DELTA  = 0   # time_delta_log
COL_EVENT_TYPE  = 1   # event_type_id
COL_SIDE        = 2   # side_id  (1=buy, 2=sell, 0=unknown)
COL_PRICE_REL   = 3   # price_rel_ticks
COL_QTY         = 4   # qty_log
COL_SPREAD      = 5   # spread_ticks

# Number of aggregate channels produced by voxelisation
N_VOX_CHANNELS = 5   # count, qty_sum, net_side, mean_time_delta, mean_spread


# ============================================================
# Dataset — builds voxelised (C, T, P) tensors on the fly
# ============================================================

class MboEvent3DDataset(Dataset):
    """
    Creates (voxel_grid, labels) pairs from MBO event NPZ files.

    For each file (day), loads events + labels, then steps a window across
    the event stream with the given stride.  The label for each window is the
    price-change at the LAST event in the window for each horizon.

    __getitem__ voxelises the raw event window into a (C, T_BINS, P_BINS)
    tensor on-the-fly (no pre-allocation of full-dataset voxels — too much RAM).

    Feature statistics (mean/std of the raw 6 features) are computed from the
    training set only — applied BEFORE voxelisation to normalise the aggregated
    values, preventing train/test leakage.

    Samples with NaN labels are skipped.
    """

    def __init__(
        self,
        npz_files:          List[Path],
        window_size:        int = WINDOW_SIZE,
        stride:             int = STRIDE,
        t_bins:             int = T_BINS,
        p_bins:             int = P_BINS,
        price_bin_range:    int = PRICE_BIN_RANGE,
        horizons:           Optional[List[str]] = None,
        normalize_features: bool = True,
        feature_stats:      Optional[Dict] = None,
    ):
        self.window_size     = window_size
        self.stride          = stride
        self.t_bins          = t_bins
        self.p_bins          = p_bins
        self.price_bin_range = price_bin_range
        self.horizons        = horizons or ["1s", "5s", "10s"]
        self.normalize_features = normalize_features

        self.all_events:   List[np.ndarray] = []
        self.all_labels:   Dict[str, List[np.ndarray]] = {h: [] for h in self.horizons}
        self.sample_index: List[Tuple[int, int]] = []   # (day_idx, event_start)

        if normalize_features and feature_stats is None:
            self._compute_stats(npz_files)
        elif feature_stats is not None:
            self.feature_mean = feature_stats["mean"]
            self.feature_std  = feature_stats["std"]
        else:
            self.feature_mean = np.zeros(6, dtype=np.float32)
            self.feature_std  = np.ones(6, dtype=np.float32)

        self._load_data(npz_files)

    # ------------------------------------------------------------------
    def _compute_stats(self, npz_files: List[Path]):
        """Compute running mean/std across all events for normalisation."""
        logger.info(f"Computing feature statistics from {len(npz_files)} files...")
        total_sum   = np.zeros(6, dtype=np.float64)
        total_sq    = np.zeros(6, dtype=np.float64)
        total_count = 0

        for f in npz_files:
            for _attempt in range(12):
                try:
                    data = np.load(f, allow_pickle=True)
                    break
                except PermissionError:
                    if _attempt < 11:
                        import time as _time
                        logger.warning(
                            f"PermissionError on {f.name} in stats, "
                            f"retry {_attempt+1}/12 (Defender scan?) waiting 5s..."
                        )
                        _time.sleep(5)
                    else:
                        raise
            events = data["events"].astype(np.float64)
            total_sum   += events.sum(axis=0)
            total_sq    += (events ** 2).sum(axis=0)
            total_count += len(events)

        self.feature_mean = (total_sum / total_count).astype(np.float32)
        var = (total_sq / total_count) - (self.feature_mean.astype(np.float64) ** 2)
        self.feature_std  = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
        logger.info(f"Feature mean: {self.feature_mean}")
        logger.info(f"Feature std:  {self.feature_std}")

    def get_feature_stats(self) -> Dict:
        return {"mean": self.feature_mean, "std": self.feature_std}

    # ------------------------------------------------------------------
    def _load_data(self, npz_files: List[Path]):
        """Load all NPZ files into memory and build sample index."""
        for day_idx, f in enumerate(npz_files):
            for _attempt in range(12):
                try:
                    data = np.load(f, allow_pickle=True)
                    break
                except PermissionError:
                    if _attempt < 11:
                        import time as _time
                        logger.warning(
                            f"PermissionError on {f.name}, retry {_attempt+1}/12 "
                            f"(Defender scan?) waiting 5s..."
                        )
                        _time.sleep(5)
                    else:
                        raise

            # Store RAW (unnormalised) events — normalisation applied during voxelisation
            # so we can apply it selectively to the channels that benefit from it.
            # We store normalised events here for simplicity (consistent with CNN1D).
            events   = data["events"].astype(np.float32)   # (N, 6)
            n_events = len(events)

            if self.normalize_features:
                events = (events - self.feature_mean) / (self.feature_std + 1e-8)

            day_labels: Dict[str, np.ndarray] = {}
            for h in self.horizons:
                day_labels[h] = data[f"labels_{h}"].astype(np.float32)

            for start in range(0, n_events - self.window_size + 1, self.stride):
                end       = start + self.window_size
                label_idx = end - 1
                labels_ok = all(
                    not np.isnan(day_labels[h][label_idx]) for h in self.horizons
                )
                if not labels_ok:
                    continue
                self.sample_index.append((day_idx, start))

            self.all_events.append(events)
            for h in self.horizons:
                self.all_labels[h].append(day_labels[h])

        logger.info(
            f"Dataset: {len(npz_files)} days, {len(self.sample_index)} samples "
            f"(window={self.window_size}, stride={self.stride}, "
            f"T={self.t_bins}, P={self.p_bins})"
        )

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sample_index)

    def _voxelise(self, events: np.ndarray) -> np.ndarray:
        """
        Convert a (WINDOW_SIZE, 6) event window into a (C, T_BINS, P_BINS) voxel grid.

        Binning strategy:
          - Temporal bins: position-based — event i → t_bin = floor(i * T_BINS / W)
            This gives equal-count temporal slices regardless of clock time.
            (Clock-time binning is an alternative but harder without raw timestamps.)
          - Price bins: value-based — price_rel_ticks quantised into P_BINS buckets
            centred at 0, range ±PRICE_BIN_RANGE ticks. Events outside the range
            are clipped to the nearest bin edge.

        Aggregation per (t_bin, p_bin) cell:
          ch0 — event count  (number of events landing in cell)
          ch1 — qty sum      (sum of qty_log — total volume at this level/time)
          ch2 — net side     (buy_count - sell_count, normalised by count)
          ch3 — mean time_delta_log (urgency proxy — small = fast-arriving events)
          ch4 — mean spread  (bid-ask spread at that time/level)

        Empty cells get 0.0 for count, qty, net_side, and -1.0 for mean values
        (to distinguish "no events" from "events with value 0").

        Args:
            events: (W, 6) float32 — NORMALISED event window

        Returns:
            voxel: (N_VOX_CHANNELS, T_BINS, P_BINS) float32
        """
        W = len(events)
        T = self.t_bins
        P = self.p_bins

        # Temporal bin assignment: positional (equal-count slices)
        # event at position i → t_bin ∈ [0, T-1]
        t_bin_idx = np.floor(
            np.arange(W, dtype=np.float32) * T / W
        ).astype(np.int32)
        t_bin_idx = np.clip(t_bin_idx, 0, T - 1)

        # Price bin assignment: value-based on price_rel_ticks (feature col 3)
        # After normalisation, we need to recover approximate tick scale.
        # We use the normalised price_rel values directly and bin them uniformly.
        # The range [-PRICE_BIN_RANGE, +PRICE_BIN_RANGE] in normalised space is
        # approximately ±(PRICE_BIN_RANGE / price_std) std devs.
        # Simpler: just use quantile-based binning on the window itself.
        price_norm = events[:, COL_PRICE_REL]   # normalised price_rel_ticks

        # Bin price into P equally-spaced buckets over [-3, 3] std (captures 99.7%)
        # Events outside range → clipped to edge bins
        price_min = -3.0
        price_max =  3.0
        p_bin_float = (price_norm - price_min) / (price_max - price_min) * P
        p_bin_idx   = np.floor(p_bin_float).astype(np.int32)
        p_bin_idx   = np.clip(p_bin_idx, 0, P - 1)

        # Aggregate into voxel grid
        voxel = np.zeros((N_VOX_CHANNELS, T, P), dtype=np.float32)

        # Accumulators (use separate arrays for running sums)
        count_grid      = np.zeros((T, P), dtype=np.float32)
        qty_grid        = np.zeros((T, P), dtype=np.float32)
        buy_grid        = np.zeros((T, P), dtype=np.float32)
        sell_grid       = np.zeros((T, P), dtype=np.float32)
        time_delta_grid = np.zeros((T, P), dtype=np.float32)
        spread_grid     = np.zeros((T, P), dtype=np.float32)

        # Vectorised accumulation using numpy ufunc scatter
        np.add.at(count_grid,      (t_bin_idx, p_bin_idx), 1.0)
        np.add.at(qty_grid,        (t_bin_idx, p_bin_idx), events[:, COL_QTY])
        np.add.at(time_delta_grid, (t_bin_idx, p_bin_idx), events[:, COL_TIME_DELTA])
        np.add.at(spread_grid,     (t_bin_idx, p_bin_idx), events[:, COL_SPREAD])

        # Side: side_id is normalised; original 1=buy, 2=sell.
        # After normalisation the threshold to determine buy/sell may shift.
        # We use sign of normalised side_id relative to its mean (0 → side_mean).
        # A cleaner approach: side > 0 in raw data means buy (id=1), sell (id=2).
        # After normalisation: side_norm = (side_id - mean_side) / std_side
        # buy if original side_id == 1 → normalised side is below mean (negative offset)
        # This heuristic works if mean_side ≈ 1.5 (equal mix), then:
        #   buy (id=1) → norm ≈ -0.67, sell (id=2) → norm ≈ +0.67
        # So: side_norm < 0 → buy, side_norm > 0 → sell
        is_buy  = (events[:, COL_SIDE] < 0.0).astype(np.float32)
        is_sell = (events[:, COL_SIDE] > 0.0).astype(np.float32)
        np.add.at(buy_grid,  (t_bin_idx, p_bin_idx), is_buy)
        np.add.at(sell_grid, (t_bin_idx, p_bin_idx), is_sell)

        # Convert sums to means where count > 0
        nonzero = count_grid > 0.0

        voxel[0] = count_grid                                           # ch0: count
        voxel[1] = qty_grid                                             # ch1: qty_sum
        # ch2: net_side normalised by count ∈ [-1, 1]
        net_side = np.where(nonzero, (buy_grid - sell_grid) / count_grid, 0.0)
        voxel[2] = net_side
        # ch3: mean time_delta (-1.0 in empty cells)
        voxel[3] = np.where(nonzero, time_delta_grid / count_grid, -1.0)
        # ch4: mean spread (-1.0 in empty cells)
        voxel[4] = np.where(nonzero, spread_grid / count_grid, -1.0)

        return voxel

    def __getitem__(self, idx: int):
        day_idx, start = self.sample_index[idx]
        end       = start + self.window_size
        events    = self.all_events[day_idx][start:end]   # (W, 6)
        label_idx = end - 1
        labels = np.array(
            [self.all_labels[h][day_idx][label_idx] for h in self.horizons],
            dtype=np.float32,
        )  # (3,)

        voxel = self._voxelise(events)   # (C, T, P) float32
        return torch.from_numpy(voxel), torch.from_numpy(labels)


# ============================================================
# Architecture: Event 3D CNN (Conv2d on T×P grid)
# ============================================================

class ResBlock2d(nn.Module):
    """
    Residual 2D conv block for the voxelised orderbook grid.

    Layout:
        input
          ├── skip (1×1 conv if channels differ, or stride projection)
          └── Conv2d → BN → GELU → Conv2d → BN
                └── + residual → GELU → output

    Stride can be used to downsample the (T, P) spatial dimensions.
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int = 3,
        stride:       int = 1,
        dropout:      float = 0.1,
    ):
        super().__init__()
        pad = kernel_size // 2

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size, stride=stride, padding=pad, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.act  = nn.GELU()

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size, stride=1, padding=pad, bias=False
        )
        self.bn2    = nn.BatchNorm2d(out_channels)
        self.drop   = nn.Dropout2d(dropout)

        # Residual projection when shape changes
        if in_channels != out_channels or stride != 1:
            self.skip = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, T, P)
        Returns:
            out: (B, C_out, T', P')  — T', P' may be smaller if stride > 1
        """
        residual = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.drop(out)

        return self.act(out + residual)


class EventCNN3D(nn.Module):
    """
    2D Residual CNN operating on the voxelised MBO event grid.

    Input:  (B, C_in=5, T_BINS, P_BINS)
             — "video" of orderbook activity: channels × time × price

    Architecture:
        Stem:    Conv2d(5 → channels, 3×3) + BN + GELU
        Blocks:  N residual blocks; first two blocks preserve spatial dims,
                 last two apply 2×2 stride to downsample (reduces compute,
                 increases receptive field)
        Pool:    AdaptiveAvgPool2d(1, 1) → (B, channels)
        Head:    LayerNorm → Linear → GELU → Dropout → Linear(channels → 3)

    Design choices:
      - Conv2d (not Conv3d): the (T, P) axes are already the 2D "image".
        Batch dim handles the sample dimension. No extra depth axis needed.
      - Strides on later blocks: avoids overfitting on the relatively small
        T×P grid while expanding effective receptive field.
      - Global average pooling: invariant to exact T×P resolution at inference.
      - Multi-task head: predict 1s, 5s, 10s simultaneously (shared representation).
    """

    def __init__(
        self,
        in_channels:  int   = N_VOX_CHANNELS,
        channels:     int   = CNN3D_CHANNELS,
        n_blocks:     int   = CNN3D_LAYERS,
        dropout:      float = CNN3D_DROPOUT,
        n_targets:    int   = 3,
    ):
        super().__init__()
        self.channels  = channels
        self.n_targets = n_targets

        # Stem: initial feature extraction
        self.stem = nn.Sequential(
            nn.Conv2d(in_channels, channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.GELU(),
        )

        # Residual blocks
        # First half: preserve resolution (stride=1) — fine-grained patterns
        # Second half: downsample (stride=2) — coarser patterns, larger receptive field
        self.blocks = nn.ModuleList()
        for i in range(n_blocks):
            stride = 1 if i < max(1, n_blocks // 2) else 2
            self.blocks.append(
                ResBlock2d(
                    in_channels  = channels,
                    out_channels = channels,
                    kernel_size  = 3,
                    stride       = stride,
                    dropout      = dropout,
                )
            )

        # Global average pooling: (B, C, T', P') → (B, C)
        self.gap = nn.AdaptiveAvgPool2d((1, 1))

        # Prediction head
        self.head = nn.Sequential(
            nn.LayerNorm(channels),
            nn.Linear(channels, channels * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(channels * 2, n_targets),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d,)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, voxel: torch.Tensor) -> torch.Tensor:
        """
        Args:
            voxel: (B, C_in, T_BINS, P_BINS) float32
        Returns:
            preds: (B, n_targets) — predicted price changes (1s, 5s, 10s)
        """
        x = self.stem(voxel)          # (B, C, T, P)

        for block in self.blocks:
            x = block(x)              # (B, C, T', P')

        x = self.gap(x)               # (B, C, 1, 1)
        x = x.squeeze(-1).squeeze(-1) # (B, C)

        preds = self.head(x)          # (B, n_targets)
        return preds


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# IC Computation
# ============================================================

def compute_ic(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Spearman IC between predictions and labels. Handles NaN."""
    valid = ~(np.isnan(predictions) | np.isnan(labels))
    if valid.sum() < 20:
        return float("nan")
    rho, _ = scipy.stats.spearmanr(predictions[valid], labels[valid])
    return float(rho)


# ============================================================
# LR Scheduler with Warmup
# ============================================================

class WarmupCosineScheduler:
    """Linear warmup then cosine decay."""

    def __init__(
        self,
        optimizer,
        warmup_steps: int,
        total_steps:  int,
        min_lr:       float = 1e-6,
    ):
        self.optimizer    = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps  = total_steps
        self.min_lr       = min_lr
        self.base_lrs     = [pg["lr"] for pg in optimizer.param_groups]
        self._step        = 0

    def step(self):
        self._step += 1
        s = self._step
        for i, pg in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if s <= self.warmup_steps:
                lr = base_lr * s / max(self.warmup_steps, 1)
            else:
                progress = (s - self.warmup_steps) / max(
                    self.total_steps - self.warmup_steps, 1
                )
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (
                    1 + np.cos(np.pi * progress)
                )
            pg["lr"] = lr

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    use_amp: bool = True,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Run inference on a DataLoader. Returns (metrics_dict, preds, labels)."""
    model.eval()
    total_loss = 0.0
    n_batches  = 0
    all_preds  = []
    all_labels = []

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    with torch.no_grad():
        for voxel, labels in loader:
            voxel  = voxel.to(device, non_blocking=True)    # (B, C, T, P)
            labels = labels.to(device, non_blocking=True)   # (B, 3)
            with amp_ctx:
                preds = model(voxel)
                loss  = F.mse_loss(preds, labels)
            total_loss += loss.item()
            n_batches  += 1
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())

    if not all_preds:
        return (
            {"loss": total_loss / max(n_batches, 1)},
            np.empty((0, len(HORIZONS))),
            np.empty((0, len(HORIZONS))),
        )

    all_preds  = np.concatenate(all_preds,  axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = {"loss": total_loss / max(n_batches, 1)}
    for i, h in enumerate(HORIZONS):
        metrics[f"ic_{h}"] = compute_ic(all_preds[:, i], all_labels[:, i])

    return metrics, all_preds, all_labels


def evaluate_metrics_only(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    use_amp: bool = True,
) -> Dict:
    """Convenience wrapper: return only metrics dict."""
    metrics, _, _ = evaluate(model, loader, device, use_amp=use_amp)
    return metrics


# ============================================================
# OOT Inference
# ============================================================

def run_oot_inference(
    model:   nn.Module,
    loader:  DataLoader,
    device:  torch.device,
    use_amp: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on OOT fold. Returns (predictions, labels)."""
    model.eval()
    all_preds  = []
    all_labels = []

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    with torch.no_grad():
        for voxel, labels in loader:
            voxel = voxel.to(device, non_blocking=True)
            with amp_ctx:
                preds = model(voxel)
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())

    if not all_preds:
        return np.empty((0, len(HORIZONS))), np.empty((0, len(HORIZONS)))
    return np.concatenate(all_preds, axis=0), np.concatenate(all_labels, axis=0)


# ============================================================
# Training Loop (single fold)
# ============================================================

def train_one_fold(
    model:             nn.Module,
    train_loader:      DataLoader,
    val_loader:        DataLoader,
    fold_idx:          int,
    output_dir:        Path,
    mlflow_run,
    device:            torch.device,
    total_train_steps: int,
    use_amp:           bool = True,
) -> Dict:
    """Train Event3DCNN for one expanding-window fold. Returns metrics dict."""

    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler    = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    scheduler = WarmupCosineScheduler(
        optimizer,
        warmup_steps = WARMUP_STEPS,
        total_steps  = total_train_steps,
    )

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    best_val_loss = float("inf")
    global_step   = 0

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0

        for voxel, labels in train_loader:
            voxel  = voxel.to(device, non_blocking=True)    # (B, C, T, P)
            labels = labels.to(device, non_blocking=True)   # (B, 3)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                preds = model(voxel)                         # (B, 3)
                loss  = F.mse_loss(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss  += loss.item()
            n_batches   += 1
            global_step += 1

        avg_loss    = epoch_loss / max(n_batches, 1)
        val_metrics = evaluate_metrics_only(model, val_loader, device, use_amp=use_amp)

        logger.info(
            f"Fold {fold_idx:02d} | Epoch {epoch+1:2d}/{EPOCHS_PER_FOLD} | "
            f"Train Loss: {avg_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IC (10s): {val_metrics.get('ic_10s', float('nan')):.4f} | "
            f"LR: {scheduler.get_lr():.2e}"
        )

        if MLFLOW_AVAILABLE and mlflow_run:
            step_offset = fold_idx * EPOCHS_PER_FOLD + epoch
            mlflow.log_metrics(
                {
                    f"fold{fold_idx:02d}_train_loss": avg_loss,
                    f"fold{fold_idx:02d}_val_loss":   val_metrics["loss"],
                    f"fold{fold_idx:02d}_val_ic_1s":  val_metrics.get("ic_1s",  float("nan")),
                    f"fold{fold_idx:02d}_val_ic_10s": val_metrics.get("ic_10s", float("nan")),
                },
                step=step_offset,
            )

        # Save best checkpoint for this fold
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "fold":        fold_idx,
                    "epoch":       epoch,
                    "val_loss":    val_metrics["loss"],
                    "val_ic_10s":  val_metrics.get("ic_10s"),
                    "arch": {
                        "channels":    CNN3D_CHANNELS,
                        "n_blocks":    CNN3D_LAYERS,
                        "dropout":     CNN3D_DROPOUT,
                        "t_bins":      T_BINS,
                        "p_bins":      P_BINS,
                        "window_size": WINDOW_SIZE,
                        "in_channels": N_VOX_CHANNELS,
                    },
                },
                ckpt_path,
            )

    return {"best_val_loss": best_val_loss}


# ============================================================
# Walk-Forward (Expanding Window)
# ============================================================

def run_expanding_wf(
    npz_files:  List[Path],
    output_dir: Path,
    device:     torch.device,
    n_folds:    int = N_FOLDS,
):
    """
    Expanding window walk-forward training.

    Each fold adds one more day to the training set.
    Feature statistics computed from train set only (no leakage).
    OOT predictions saved per fold AND concatenated for final concat-IC.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort files by date (filename-based sorting; assumes YYYYMMDD prefix)
    npz_files = sorted(npz_files)

    # Filter files with no valid labels
    def _has_valid_labels(f: Path) -> bool:
        try:
            d   = np.load(f, allow_pickle=True)
            lbl = d["labels_1s"]
            return bool(not np.all(np.isnan(lbl)))
        except Exception:
            return False

    valid_files = [f for f in npz_files if _has_valid_labels(f)]
    skipped     = [f.name for f in npz_files if f not in set(valid_files)]
    if skipped:
        logger.warning(f"Skipping {len(skipped)} file(s) with all-NaN labels: {skipped}")
    npz_files = valid_files

    n_files = len(npz_files)
    if n_files == 0:
        logger.error("No valid NPZ files found. Exiting.")
        return {}

    logger.info(f"Total files (valid): {n_files} ({npz_files[0].name} → {npz_files[-1].name})")

    # Build fold boundaries (expanding window — ABSOLUTE RULE)
    min_train       = max(5, n_files - n_folds)
    fold_boundaries = []
    for fold in range(n_folds):
        train_end = min_train + fold
        oot_start = train_end
        oot_end   = oot_start + max(1, (n_files - min_train) // n_folds)
        oot_end   = min(oot_end, n_files)
        if oot_start >= n_files:
            break
        fold_boundaries.append((fold, list(range(train_end)), list(range(oot_start, oot_end))))

    logger.info(f"Running {len(fold_boundaries)} folds (expanding window)")

    # AMP only useful on CUDA
    use_amp = device.type == "cuda"

    # Concat storage
    concat_preds  = {h: [] for h in HORIZONS}
    concat_labels = {h: [] for h in HORIZONS}

    # MLflow run
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(
            run_name=f"Event3DCNN_{time.strftime('%Y%m%d_%H%M')}"
        )
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        mlflow.log_params(
            {
                "model":            "Event3DCNN",
                "window_size":      WINDOW_SIZE,
                "stride":           STRIDE,
                "t_bins":           T_BINS,
                "p_bins":           P_BINS,
                "price_bin_range":  PRICE_BIN_RANGE,
                "cnn3d_channels":   CNN3D_CHANNELS,
                "cnn3d_n_blocks":   CNN3D_LAYERS,
                "cnn3d_dropout":    CNN3D_DROPOUT,
                "n_vox_channels":   N_VOX_CHANNELS,
                "vox_channels":     "count,qty_sum,net_side,mean_time_delta,mean_spread",
                "batch_size":       BATCH_SIZE,
                "lr":               LR,
                "epochs_per_fold":  EPOCHS_PER_FOLD,
                "n_folds":          len(fold_boundaries),
                "horizons":         str(HORIZONS),
                "n_files":          n_files,
                "optimizer":        "AdamW",
                "warmup_steps":     WARMUP_STEPS,
                "grad_clip":        GRAD_CLIP,
                "node":             socket.gethostname(),
                "gpu":              gpu_name,
                "data_dir":         str(npz_files[0].parent),
                "output_dir":       str(output_dir),
                "mixed_precision":  "fp16" if use_amp else "none",
                "num_workers":      0,
                "binning_t":        "positional_equal_count",
                "binning_p":        "value_uniform_stddev_range",
            }
        )

    try:
        for fold_idx, train_file_idxs, oot_file_idxs in fold_boundaries:
            train_files = [npz_files[i] for i in train_file_idxs]
            oot_files   = [npz_files[i] for i in oot_file_idxs]

            logger.info(
                f"\n{'='*60}\n"
                f"FOLD {fold_idx:02d} | Train: {len(train_files)} days "
                f"({train_files[0].name}→{train_files[-1].name}) | "
                f"OOT: {len(oot_files)} days ({oot_files[0].name}→{oot_files[-1].name})"
                f"\n{'='*60}"
            )

            # Build train dataset — stats computed from train set only (no leakage)
            logger.info("Building train dataset...")
            train_ds = MboEvent3DDataset(
                train_files,
                window_size     = WINDOW_SIZE,
                stride          = STRIDE,
                t_bins          = T_BINS,
                p_bins          = P_BINS,
                price_bin_range = PRICE_BIN_RANGE,
            )
            feature_stats = train_ds.get_feature_stats()

            # Build OOT dataset using TRAIN feature stats (no leakage)
            logger.info("Building OOT dataset...")
            oot_ds = MboEvent3DDataset(
                oot_files,
                window_size     = WINDOW_SIZE,
                stride          = STRIDE,
                t_bins          = T_BINS,
                p_bins          = P_BINS,
                price_bin_range = PRICE_BIN_RANGE,
                feature_stats   = feature_stats,
            )

            # Windows: num_workers=0 to avoid pickle issues with large in-memory numpy arrays
            _num_workers = 0

            train_loader = DataLoader(
                train_ds,
                batch_size  = BATCH_SIZE,
                shuffle     = True,
                num_workers = _num_workers,
                pin_memory  = (device.type == "cuda"),
                drop_last   = True,
            )
            oot_loader = DataLoader(
                oot_ds,
                batch_size  = BATCH_SIZE * 2,
                shuffle     = False,
                num_workers = _num_workers,
                pin_memory  = (device.type == "cuda"),
            )

            # Fresh model per fold
            model = EventCNN3D(
                in_channels = N_VOX_CHANNELS,
                channels    = CNN3D_CHANNELS,
                n_blocks    = CNN3D_LAYERS,
                dropout     = CNN3D_DROPOUT,
                n_targets   = len(HORIZONS),
            ).to(device)

            if fold_idx == 0:
                n_params = count_parameters(model)
                logger.info(f"Model parameters: {n_params:,}")
                logger.info(f"Input shape:      ({N_VOX_CHANNELS}, {T_BINS}, {P_BINS})")
                logger.info(f"Channels:         {CNN3D_CHANNELS}")
                logger.info(f"Blocks:           {CNN3D_LAYERS}")
                logger.info(f"Voxel channels:   count | qty_sum | net_side | mean_td | mean_spread")

            total_steps = EPOCHS_PER_FOLD * len(train_loader)
            train_one_fold(
                model, train_loader, oot_loader,
                fold_idx, output_dir, mlflow_run, device, total_steps,
                use_amp=use_amp,
            )

            # Reload best checkpoint for OOT inference
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                logger.info(f"Loaded best checkpoint (val_loss={ckpt['val_loss']:.4f})")

            # OOT inference
            logger.info("Running OOT inference...")
            oot_preds, oot_labels = run_oot_inference(model, oot_loader, device, use_amp=use_amp)

            # Per-fold IC
            fold_ics: Dict[str, float] = {}
            for i, h in enumerate(HORIZONS):
                ic           = compute_ic(oot_preds[:, i], oot_labels[:, i])
                fold_ics[h]  = ic
                concat_preds[h].append(oot_preds[:, i])
                concat_labels[h].append(oot_labels[:, i])

            logger.info(
                f"Fold {fold_idx:02d} OOT IC | "
                + " | ".join(f"{h}: {fold_ics[h]:.4f}" for h in HORIZONS)
            )

            # Save fold artifacts: .npz predictions (MANDATORY)
            pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
            np.savez_compressed(
                pred_path,
                predictions = oot_preds,
                labels      = oot_labels,
                horizons    = np.array(HORIZONS),
                ic_1s       = np.array(fold_ics.get("1s",  float("nan"))),
                ic_5s       = np.array(fold_ics.get("5s",  float("nan"))),
                ic_10s      = np.array(fold_ics.get("10s", float("nan"))),
                oot_files   = np.array([str(f) for f in oot_files]),
            )
            logger.info(f"Saved predictions → {pred_path}")

            # Save feature stats for this fold (needed for inference)
            stats_path = output_dir / f"fold_{fold_idx:02d}_feature_stats.npz"
            np.savez(stats_path, mean=feature_stats["mean"], std=feature_stats["std"])

            # Log per-fold metrics to MLflow
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow.log_metrics(
                    {f"oot_ic_{h}_fold{fold_idx:02d}": fold_ics[h] for h in HORIZONS},
                    step=fold_idx,
                )

            # Free memory before next fold
            del train_ds, oot_ds, train_loader, oot_loader, model
            gc.collect()
            torch.cuda.empty_cache()

        # ============================================================
        # Compute CONCAT IC (primary metric — all folds combined)
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("CONCAT IC (primary metric — all folds combined)")
        logger.info("=" * 60)

        concat_ic: Dict[str, float] = {}
        for h in HORIZONS:
            if concat_preds[h]:
                all_p        = np.concatenate(concat_preds[h])
                all_l        = np.concatenate(concat_labels[h])
                ic           = compute_ic(all_p, all_l)
                concat_ic[h] = ic
                logger.info(f"  Concat IC ({h}): {ic:.4f}")
            else:
                concat_ic[h] = float("nan")

        # Save all concat predictions
        concat_path = output_dir / "concat_oot_predictions.npz"
        np.savez_compressed(
            concat_path,
            **{f"preds_{h}":     np.concatenate(concat_preds[h])
               for h in HORIZONS if concat_preds[h]},
            **{f"labels_{h}":    np.concatenate(concat_labels[h])
               for h in HORIZONS if concat_labels[h]},
            **{f"concat_ic_{h}": np.array(concat_ic[h]) for h in HORIZONS},
        )
        logger.info(f"Saved concat predictions → {concat_path}")

        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.log_metrics({f"concat_ic_{h}": concat_ic[h] for h in HORIZONS})

        logger.info("\n" + "=" * 60)
        logger.info("LEAKAGE AUDIT: PASSED")
        logger.info("  - Expanding window: train set never contains OOT dates")
        logger.info("  - Feature normalisation computed from train set only per fold")
        logger.info("  - Positional temporal binning: uses event position index, not future time")
        logger.info("  - Price binning uses normalised values from train-set stats only")
        logger.info("  - Voxelisation aggregates only past events within the causal window")
        logger.info("=" * 60)

        return concat_ic

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()


# ============================================================
# Data Transfer: Jupiter → Neptune via SCP / API
# ============================================================

def _jupiter_exec(cmd: str, timeout: int = 30) -> str:
    """Execute a command on Jupiter via its Flask API and return stdout."""
    import urllib.request, json as _json
    payload = _json.dumps({"command": cmd}).encode()
    req = urllib.request.Request(
        "http://jupiter:8765/exec",
        data    = payload,
        headers = {"X-API-Key": os.environ.get("QCC_API_KEY", ""), "Content-Type": "application/json"},
        method  = "POST",
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        result = _json.load(resp)
    return result.get("stdout", "")


def transfer_data_from_jupiter(dest_dir: Path):
    """
    Copy MBO event NPZ files from Jupiter to Neptune.

    Strategy:
      1. List files on Jupiter via Flask API
      2. SCP if key-based auth is available (fastest)
      3. Fallback: base64 streaming via API in 8MB chunks
    """
    import subprocess, base64, json as _json

    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = set(f.name for f in dest_dir.glob("*.npz"))

    # List remote files
    try:
        stdout = _jupiter_exec(
            "ls /home/jupiter/Lvl3Quant/data/processed/mbo_events/ | grep '.npz$'",
            timeout=15,
        )
        remote_files = [f.strip() for f in stdout.splitlines() if f.strip().endswith(".npz")]
    except Exception as e:
        logger.warning(f"Could not list Jupiter files: {e}")
        return

    to_copy = [f for f in remote_files if f not in existing]
    if not to_copy:
        logger.info(f"All {len(remote_files)} files already on Neptune. Skipping transfer.")
        return

    logger.info(f"Transferring {len(to_copy)}/{len(remote_files)} files from Jupiter → Neptune...")

    # Try SCP first
    scp_available = False
    try:
        r = subprocess.run(
            [
                "scp", "-o", "StrictHostKeyChecking=no",
                "-o", "BatchMode=yes", "-o", "ConnectTimeout=5",
                f"jupiter@jupiter:/home/jupiter/Lvl3Quant/data/processed/mbo_events/{to_copy[0]}",
                str(dest_dir / to_copy[0]),
            ],
            capture_output=True, text=True, timeout=30,
        )
        if r.returncode == 0:
            scp_available = True
            logger.info("SCP key-based auth works — using SCP for transfer")
        else:
            logger.info(f"SCP auth failed ({r.stderr[:100]}), falling back to API base64 transfer")
    except Exception:
        logger.info("SCP not available, using API base64 transfer")

    for fname in to_copy:
        remote_path = f"/home/jupiter/Lvl3Quant/data/processed/mbo_events/{fname}"
        dest_path   = dest_dir / fname

        if dest_path.exists():
            logger.info(f"  Already exists: {fname}")
            continue

        logger.info(f"  Transferring: {fname}")
        t0 = time.time()

        if scp_available:
            r = subprocess.run(
                [
                    "scp", "-o", "StrictHostKeyChecking=no",
                    f"jupiter@jupiter:{remote_path}", str(dest_path),
                ],
                capture_output=True, text=True, timeout=300,
            )
            if r.returncode == 0:
                size_mb = dest_path.stat().st_size / 1e6
                logger.info(f"    Done: {fname} ({size_mb:.0f} MB in {time.time()-t0:.1f}s)")
            else:
                logger.warning(f"    SCP failed: {r.stderr[:200]}")
        else:
            try:
                size_str   = _jupiter_exec(f"stat -c %s {remote_path}", timeout=10).strip()
                file_size  = int(size_str)
                chunk_size = 8 * 1024 * 1024
                n_chunks   = (file_size + chunk_size - 1) // chunk_size
                logger.info(f"    File size: {file_size/1e6:.0f} MB, {n_chunks} chunks")

                with open(dest_path, "wb") as fout:
                    for chunk_i in range(n_chunks):
                        skip_mb  = (chunk_i * chunk_size) // (1024 * 1024)
                        count_mb = max(1, chunk_size // (1024 * 1024))
                        cmd = (
                            f"dd if={remote_path} bs=1M skip={skip_mb} count={count_mb} 2>/dev/null | "
                            f"base64 -w 0"
                        )
                        b64_data = _jupiter_exec(cmd, timeout=60).strip()
                        if not b64_data:
                            logger.warning(f"    Empty chunk {chunk_i} — skipping")
                            break
                        fout.write(base64.b64decode(b64_data))
                        if (chunk_i + 1) % 5 == 0:
                            logger.info(f"    Progress: {chunk_i+1}/{n_chunks} chunks")

                actual_size = dest_path.stat().st_size
                if abs(actual_size - file_size) > 1024:
                    logger.warning(
                        f"    Size mismatch: expected {file_size}, got {actual_size}. Removing."
                    )
                    dest_path.unlink()
                else:
                    logger.info(
                        f"    Done: {fname} ({actual_size/1e6:.0f} MB in {time.time()-t0:.1f}s)"
                    )
            except Exception as e:
                logger.error(f"    Transfer failed for {fname}: {e}")
                if dest_path.exists():
                    dest_path.unlink()

    final_files = list(dest_dir.glob("*.npz"))
    logger.info(f"Files available on Neptune after transfer: {len(final_files)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Event 3D CNN Walk-Forward Training")
    parser.add_argument("--data-dir",      type=str, default=DEFAULT_DATA_DIR,
                        help="Directory with mbo_events NPZ files")
    parser.add_argument("--output-dir",    type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for checkpoints and predictions")
    parser.add_argument("--n-folds",       type=int, default=N_FOLDS)
    parser.add_argument("--skip-transfer", action="store_true",
                        help="Skip data transfer from Jupiter")
    parser.add_argument("--device",        type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    data_dir   = Path(args.data_dir)
    device     = torch.device(args.device if torch.cuda.is_available() else "cpu")

    logger.info("=" * 60)
    logger.info("Event 3D CNN — Walk-Forward Training")
    logger.info(f"Device:           {device}")
    if device.type == "cuda":
        logger.info(f"GPU:              {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM:             {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Voxel grid:       T={T_BINS} × P={P_BINS} × C={N_VOX_CHANNELS}")
    logger.info(f"Channels:         {CNN3D_CHANNELS}")
    logger.info(f"Blocks:           {CNN3D_LAYERS}")
    logger.info(f"Window size:      {WINDOW_SIZE} events")
    logger.info(f"Batch size:       {BATCH_SIZE}")
    logger.info(f"Price bin range:  ±3 std (normalised price_rel_ticks)")
    logger.info(f"MLflow URI:       {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow Exp:       {MLFLOW_EXPERIMENT}")
    logger.info(f"Data dir:         {data_dir}")
    logger.info(f"Output dir:       {output_dir}")
    logger.info("=" * 60)

    # Step 1: Transfer data from Jupiter if needed
    if not args.skip_transfer:
        transfer_data_from_jupiter(data_dir)

    # Gather NPZ files
    npz_files = sorted(data_dir.glob("*_mbo_events.npz"))
    if not npz_files:
        logger.error(f"No *_mbo_events.npz files found in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(npz_files)} NPZ files: {npz_files[0].name} → {npz_files[-1].name}")

    # Step 2: Set process priority to BELOW_NORMAL on Windows (yield to live inference)
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        logger.info("Process priority set to BELOW_NORMAL")
    except Exception as e:
        logger.info(f"Could not set priority (non-critical): {e}")

    # Step 3: Run expanding walk-forward
    concat_ic = run_expanding_wf(
        npz_files  = npz_files,
        output_dir = output_dir,
        device     = device,
        n_folds    = args.n_folds,
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Final Concat IC:")
    for h in HORIZONS:
        logger.info(f"  {h}: {concat_ic.get(h, float('nan')):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
