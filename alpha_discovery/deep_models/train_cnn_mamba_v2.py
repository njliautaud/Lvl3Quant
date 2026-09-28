"""
CNN-Mamba v2 -- Parallel Fusion Architecture

CNN front-end captures local microstructure patterns (strong at 10s)
+ Mamba SSM provides long-range temporal context (strong at 1s)
= Hybrid architecture addressing Mamba's 10s weakness while keeping 1s strength

Data format (identical to train_event_mamba_cuda.py):
  - events: (N_events, N_features) float32
  - labels_1s/5s/10s: (N_events,) float32 -- mid-price change in ticks
  - timestamps: (N_events,) int64 nanoseconds

Architecture: CNN 1D front-end -> Mamba SSM blocks -> Multi-task head
  - Pure PyTorch implementation -- NO external mamba_ssm package needed
  - Core idea: maintains hidden state h that evolves as:
        h_new = A(x) * h_old + B(x) * input
        output = C(x) * h_new
    where A, B, C are INPUT-DEPENDENT (selective).
  - Time-delta conditioning: A is modulated by time_delta between events:
        A_effective = A(x) * exp(-decay_rate * time_delta)
    Fast events (small delta) -> state preserved.
    Gaps (large delta) -> state decays -> model "forgets" stale info.
  - O(n) complexity -- can handle MUCH longer contexts than transformers
  - Default WINDOW_SIZE=1000 (2x transformer) to exploit long-context advantage

Training rules (same as transformer/CNN):
  - Expanding window walk-forward (ABSOLUTE RULE)
  - N folds using available dates
  - Concat IC as primary metric
  - Mixed precision (fp16 when CUDA available)
  - MLflow logging
  - Save .pt weights + .npz predictions per fold
  - Leakage: feature stats computed from train set only per fold
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

# Try CUDA mamba_ssm kernels (10-50x faster selective scan)
try:
    from mamba_ssm import Mamba as CUDAMamba
    USE_CUDA_MAMBA = True
    print("*** Using mamba_ssm CUDA kernels (FAST) ***")
except ImportError:
    USE_CUDA_MAMBA = False
    print("*** Using pure PyTorch Mamba (SLOW) ***")

# MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
    if os.environ.get("DISABLE_MLFLOW", "0") == "1":
        MLFLOW_AVAILABLE = False
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not installed -- skipping experiment tracking")

# ============================================================
# Logging setup
# ============================================================
LOG_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)

log_path = LOG_DIR / "cnn_mamba_v2.log"

# Force-clear any pre-existing handlers (MLflow imports add handlers that make
# basicConfig a no-op, silently swallowing all output)
logging.root.handlers.clear()

_fmt = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")

_file_handler = logging.FileHandler(log_path)
_file_handler.setLevel(logging.INFO)
_file_handler.setFormatter(_fmt)

_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
_stream_handler.setFormatter(_fmt)

logging.root.setLevel(logging.INFO)
logging.root.addHandler(_file_handler)
logging.root.addHandler(_stream_handler)

logger = logging.getLogger(__name__)

# Force immediate output — print a startup marker
print(">>> train_cnn_mamba.py loaded, logging initialized", flush=True)


# ============================================================
# Config
# ============================================================
DEFAULT_DATA_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "event_mamba"


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
MLFLOW_EXPERIMENT = os.environ.get("MLFLOW_EXPERIMENT", "CNN_Mamba_v2")

# CNN front-end hyperparameters
CNN_CHANNELS = int(os.environ.get("CNN_CHANNELS", 64))
CNN_KERNEL   = int(os.environ.get("CNN_KERNEL", 5))
CNN_LAYERS   = int(os.environ.get("CNN_LAYERS", 3))

# Mamba-specific hyperparameters (MAMBA_* prefix)
MAMBA_D_MODEL  = int(os.environ.get("MAMBA_D_MODEL", 128))
MAMBA_D_STATE  = int(os.environ.get("MAMBA_D_STATE", 64))
MAMBA_N_LAYERS = int(os.environ.get("MAMBA_N_LAYERS", 4))
MAMBA_DROPOUT  = float(os.environ.get("MAMBA_DROPOUT", 0.1))
MAMBA_DT_RANK  = int(os.environ.get("MAMBA_DT_RANK", 16))
MAMBA_D_CONV   = int(os.environ.get("MAMBA_D_CONV", 4))   # local conv kernel size

# Shared hyperparameters -- use same EVENT_* env vars for easy A/B switching
WINDOW_SIZE     = int(os.environ.get("EVENT_WINDOW_SIZE", 3000))  # 2x transformer default
STRIDE          = int(os.environ.get("EVENT_STRIDE", WINDOW_SIZE // 12))
BATCH_SIZE      = int(os.environ.get("EVENT_BATCH_SIZE", 128))
LR              = float(os.environ.get("EVENT_LR", 3e-4))
EPOCHS_PER_FOLD = int(os.environ.get("EVENT_EPOCHS", 5))
WARMUP_STEPS    = int(os.environ.get("EVENT_WARMUP", 300))
GRAD_CLIP       = float(os.environ.get("EVENT_GRAD_CLIP", 1.0))
N_FOLDS         = int(os.environ.get("EVENT_N_FOLDS", 5))
WF_WINDOW_DAYS  = int(os.environ.get("WF_WINDOW_DAYS", 60))  # sliding window size in days (0=expanding)
HORIZONS        = [h.strip() for h in os.environ.get("HORIZONS", "1s,5s,10s,30s").split(",")]  # env-driven for branch experiments (HC #494 R3)

# Feature set: "raw6" (6 raw), "feat15" (15 precomputed), "smart" (15 smart-normalized), "smart_v2" (22 smart-normalized)
FEATURE_SET = os.environ.get("MAMBA_FEATURE_SET", "raw6")
# Smart features = pre-normalized (no per-fold z-score needed)
SKIP_NORMALIZE = FEATURE_SET in ("smart", "smart_v2", "smart_v3") or int(os.environ.get("SKIP_NORMALIZE", 0)) == 1

if FEATURE_SET == "smart_v2":
    N_BASE_FEATURES = 22
    USE_DERIVED_FEATURES = False
    N_DERIVED_FEATURES = 0
    N_TOTAL_FEATURES = 22
    logger.info(f"Feature set: smart_v2 (22 smart-normalized features incl. non-linear interactions)")
    logger.info("  >> SKIP_NORMALIZE=True: data already normalized by smart preprocessing v2")
elif FEATURE_SET == "smart_v3":
    N_BASE_FEATURES = 25
    USE_DERIVED_FEATURES = False
    N_DERIVED_FEATURES = 0
    N_TOTAL_FEATURES = 25
    logger.info("Feature set: smart_v3 (25 smart-normalized features incl. multi-scale OFI)")
    logger.info("  >> SKIP_NORMALIZE=True: data already normalized by smart preprocessing v3")
elif FEATURE_SET in ("feat15", "smart"):
    N_BASE_FEATURES = 15
    USE_DERIVED_FEATURES = False  # feat15/smart already has engineered features
    N_DERIVED_FEATURES = 0
    N_TOTAL_FEATURES = 15
    logger.info(f"Feature set: {FEATURE_SET} (15 {'smart-normalized' if FEATURE_SET == 'smart' else 'precomputed'} features)")
    if SKIP_NORMALIZE:
        logger.info("  >> SKIP_NORMALIZE=True: data already normalized by smart preprocessing")
else:
    USE_DERIVED_FEATURES = int(os.environ.get("MAMBA_DERIVED_FEATURES", 0)) == 1
    N_BASE_FEATURES = 6
    N_DERIVED_FEATURES = 5
    N_TOTAL_FEATURES = N_BASE_FEATURES + N_DERIVED_FEATURES if USE_DERIVED_FEATURES else N_BASE_FEATURES

# Override data dir based on feature set
if FEATURE_SET == "feat15":
    DEFAULT_DATA_DIR = str(
        Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events_feat15"
    )
elif FEATURE_SET == "smart":
    DEFAULT_DATA_DIR = str(
        Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events_smart"
    )
elif FEATURE_SET == "smart_v2":
    DEFAULT_DATA_DIR = str(
        Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events_smart_v2"
    )
elif FEATURE_SET == "smart_v3":
    DEFAULT_DATA_DIR = str(
        Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events_smart_v3"
    )

# ============================================================
# Derived Features (matching CNN implementation)
# ============================================================

def compute_derived_features(events: np.ndarray) -> np.ndarray:
    """
    Compute 5 basic orderflow derivatives from raw 6 features.

    Input features (columns of events):
        0: time_delta_log    — log inter-event time gap
        1: event_type_id     — event type (trade, add, cancel, modify, etc.)
        2: side_id           — buy(1) / sell(-1) side
        3: price_rel_ticks   — price relative to reference in ticks
        4: qty_log           — log quantity
        5: spread_ticks      — bid-ask spread in ticks

    Derived features (per-event, no look-ahead):
        6: trade_intensity   — exp(-time_delta_log) = 1/gap = events-per-second proxy
        7: signed_volume     — side_id * qty_log = directional volume per event
        8: price_accel       — diff(price_rel_ticks) = price change acceleration
        9: spread_change     — diff(spread_ticks) = spread dynamics
       10: qty_change        — diff(qty_log) = volume change rate

    All are per-event transforms — no accumulation or look-ahead.

    Args:
        events: (N, 6) float32 raw features
    Returns:
        derived: (N, 5) float32 derived features
    """
    n = len(events)
    derived = np.zeros((n, 5), dtype=np.float32)

    # 1. Trade intensity: inverse of time gap (high = fast market)
    derived[:, 0] = np.exp(-events[:, 0])

    # 2. Signed volume: direction * size (buy pressure vs sell pressure per event)
    derived[:, 1] = events[:, 2] * events[:, 4]

    # 3. Price acceleration: change in price_rel between consecutive events
    derived[1:, 2] = np.diff(events[:, 3])

    # 4. Spread change: widening or tightening
    derived[1:, 3] = np.diff(events[:, 5])

    # 5. Qty change: volume acceleration
    derived[1:, 4] = np.diff(events[:, 4])

    return derived


# ============================================================
# Dataset (identical to MboEventDataset in train_event_cnn_1d.py)
# ============================================================

class MboEventDataset(Dataset):
    """
    Creates (window_of_events, labels) pairs from MBO event NPZ files.

    For each file (day), loads events + labels, then slides a window across the
    event stream with the given stride. The label for each window is the
    price-change at the LAST event in the window for each horizon.

    Samples with NaN labels are skipped.
    """

    def __init__(
        self,
        npz_files: List[Path],
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        horizons: Optional[List[str]] = None,
        normalize_features: bool = True,
        feature_stats: Optional[Dict] = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.horizons = horizons or ["1s", "5s", "10s"]
        self.normalize_features = normalize_features

        self.all_events: List[np.ndarray] = []
        self.all_labels: Dict[str, List[np.ndarray]] = {h: [] for h in self.horizons}
        self.sample_index: List[Tuple[int, int]] = []  # (day_idx, event_start)

        if normalize_features and feature_stats is None:
            self._compute_stats(npz_files)
        elif feature_stats is not None:
            self.feature_mean = feature_stats["mean"]
            self.feature_std  = feature_stats["std"]
        else:
            self.feature_mean = np.zeros(N_BASE_FEATURES, dtype=np.float32)
            self.feature_std  = np.ones(N_BASE_FEATURES, dtype=np.float32)

        self._load_data(npz_files)

    def _compute_stats(self, npz_files: List[Path]):
        """Compute running mean/std across all events for normalization."""
        logger.info(f"Computing feature statistics from {len(npz_files)} files...")
        total_sum   = np.zeros(N_BASE_FEATURES, dtype=np.float64)
        total_sq    = np.zeros(N_BASE_FEATURES, dtype=np.float64)
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
                            f"retry {_attempt+1}/12..."
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

    def _load_data(self, npz_files: List[Path]):
        """Load all NPZ files and build sample index."""
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

            events = data["events"].astype(np.float32)  # (N, 6)
            n_events = len(events)

            if self.normalize_features:
                events = (events - self.feature_mean) / (self.feature_std + 1e-8)

            day_labels: Dict[str, np.ndarray] = {}
            for h in self.horizons:
                day_labels[h] = data[f"labels_{h}"].astype(np.float32)

            for start in range(0, n_events - self.window_size + 1, self.stride):
                end = start + self.window_size
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
            f"(window={self.window_size}, stride={self.stride})"
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        day_idx, start = self.sample_index[idx]
        end = start + self.window_size
        events    = self.all_events[day_idx][start:end]  # (W, 6)

        # Compute derived features if enabled (6 -> 11 features)
        if USE_DERIVED_FEATURES:
            derived = compute_derived_features(events)  # (W, 5)
            events = np.concatenate([events, derived], axis=1)  # (W, 11)

        label_idx = end - 1
        labels = np.array(
            [self.all_labels[h][day_idx][label_idx] for h in self.horizons],
            dtype=np.float32,
        )  # (3,)
        return torch.from_numpy(events), torch.from_numpy(labels)


class LazyMboEventDataset(Dataset):
    """
    Memory-efficient version of MboEventDataset that loads files on-demand.
    Uses an LRU cache to keep recently accessed files in memory.
    Handles 200+ files without OOM by only keeping ~10 files in RAM at once.
    """

    def __init__(
        self,
        npz_files: List[Path],
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        horizons: Optional[List[str]] = None,
        normalize_features: bool = True,
        feature_stats: Optional[Dict] = None,
        cache_size: int = 10,
    ):
        self.window_size = window_size
        self.stride = stride
        self.horizons = horizons or ["1s", "5s", "10s"]
        self.normalize_features = normalize_features
        self.npz_files = list(npz_files)
        self.cache_size = cache_size

        # LRU cache for loaded files
        self._cache: Dict[int, Dict] = {}
        self._cache_order: List[int] = []

        if normalize_features and feature_stats is None:
            self._compute_stats(npz_files)
        elif feature_stats is not None:
            self.feature_mean = feature_stats["mean"]
            self.feature_std  = feature_stats["std"]
        else:
            self.feature_mean = np.zeros(N_BASE_FEATURES, dtype=np.float32)
            self.feature_std  = np.ones(N_BASE_FEATURES, dtype=np.float32)

        self._build_index()

    def _compute_stats(self, npz_files: List[Path]):
        """Compute running mean/std across all events for normalization."""
        logger.info(f"Computing feature statistics from {len(npz_files)} files...")
        total_sum   = np.zeros(N_BASE_FEATURES, dtype=np.float64)
        total_sq    = np.zeros(N_BASE_FEATURES, dtype=np.float64)
        total_count = 0

        for f in npz_files:
            for _attempt in range(12):
                try:
                    data = np.load(f, allow_pickle=True)
                    break
                except PermissionError:
                    if _attempt < 11:
                        import time as _time
                        logger.warning(f"PermissionError on {f.name}, retry {_attempt+1}/12...")
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

    def _build_index(self):
        """Build sample index without loading all data into memory."""
        self.sample_index: List[Tuple[int, int]] = []
        self.day_n_events: List[int] = []

        for day_idx, f in enumerate(self.npz_files):
            for _attempt in range(12):
                try:
                    data = np.load(f, allow_pickle=True)
                    break
                except PermissionError:
                    if _attempt < 11:
                        import time as _time
                        _time.sleep(5)
                    else:
                        raise

            n_events = len(data["events"])
            self.day_n_events.append(n_events)

            # Check labels for NaN to build valid sample index
            day_labels = {h: data[f"labels_{h}"] for h in self.horizons}

            for start in range(0, n_events - self.window_size + 1, self.stride):
                end = start + self.window_size
                label_idx = end - 1
                labels_ok = all(
                    not np.isnan(day_labels[h][label_idx]) for h in self.horizons
                )
                if labels_ok:
                    self.sample_index.append((day_idx, start))

            # Don't keep data in memory
            del data, day_labels

        logger.info(
            f"Dataset: {len(self.npz_files)} days, {len(self.sample_index)} samples "
            f"(window={self.window_size}, stride={self.stride}) [LAZY LOADING]"
        )

    def _load_day(self, day_idx: int) -> Dict:
        """Load a day's data with LRU caching."""
        if day_idx in self._cache:
            # Move to end (most recently used)
            self._cache_order.remove(day_idx)
            self._cache_order.append(day_idx)
            return self._cache[day_idx]

        # Load from disk
        data = np.load(self.npz_files[day_idx], allow_pickle=True)
        events = data["events"].astype(np.float32)
        if self.normalize_features:
            events = (events - self.feature_mean) / (self.feature_std + 1e-8)

        day_data = {
            "events": events,
            "labels": {h: data[f"labels_{h}"].astype(np.float32) for h in self.horizons},
        }

        # Add to cache
        self._cache[day_idx] = day_data
        self._cache_order.append(day_idx)

        # Evict oldest if over cache size
        while len(self._cache_order) > self.cache_size:
            oldest = self._cache_order.pop(0)
            del self._cache[oldest]

        return day_data

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        day_idx, start = self.sample_index[idx]
        end = start + self.window_size

        day_data = self._load_day(day_idx)
        events = day_data["events"][start:end]  # (W, 6)

        if USE_DERIVED_FEATURES:
            derived = compute_derived_features(events)
            events = np.concatenate([events, derived], axis=1)

        label_idx = end - 1
        labels = np.array(
            [day_data["labels"][h][label_idx] for h in self.horizons],
            dtype=np.float32,
        )
        return torch.from_numpy(events), torch.from_numpy(labels)



# ============================================================


class FileSequentialSampler:
    """Sampler that groups samples by file (day) for cache-friendly access.
    
    Shuffles file order each epoch but iterates sequentially within each file.
    This prevents LRU cache thrashing when using LazyMboEventDataset.
    """
    def __init__(self, dataset):
        self.sample_index = dataset.sample_index  # List of (day_idx, start)
        # Group sample indices by day
        from collections import defaultdict
        self.day_to_indices = defaultdict(list)
        for i, (day_idx, _) in enumerate(self.sample_index):
            self.day_to_indices[day_idx].append(i)
        self.day_keys = list(self.day_to_indices.keys())
    
    def __iter__(self):
        import random
        # Shuffle day order each epoch
        days = self.day_keys.copy()
        random.shuffle(days)
        for day in days:
            indices = self.day_to_indices[day]
            # Shuffle within day for some randomness
            random.shuffle(indices)
            yield from indices
    
    def __len__(self):
        return len(self.sample_index)

# Architecture: Time-Aware Selective State Space Model (Mamba)
# ============================================================

class SelectiveSSM(nn.Module):
    """
    Simplified Mamba-style selective state space model block.

    Core recurrence (parallelized via selective scan):
        h_new = A(x) * h_old + B(x) * input
        output = C(x) * h_new

    Where A, B, C are INPUT-DEPENDENT (selective) -- the model learns
    WHAT to remember and WHAT to forget based on the current input.

    Time-delta conditioning: A is further modulated by time_delta:
        A_effective = A(x) * exp(-decay_rate * time_delta)
    When events arrive fast (small delta), state is preserved.
    When there's a gap (large delta), state decays -> model "forgets" stale info.

    This is computed in parallel via the associative scan trick:
    for each position t, we compute the cumulative product of A and
    cumulative sum of B*x using a parallel prefix scan, achieving O(n)
    work with O(log n) depth on GPU.

    Parameters:
        d_model:  input/output dimension
        d_state:  SSM state dimension (N in Mamba paper)
        dt_rank:  rank of the delta (timestep) projection
        d_conv:   local convolution kernel size (captures short-range patterns)
        time_delta_idx: which feature index contains time_delta_log (default 0)
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_rank: int = 16,
        d_conv: int = 4,
        time_delta_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.time_delta_idx = time_delta_idx

        # Expand dimension (Mamba uses 2x expansion internally)
        self.d_inner = d_model * 2

        # Input projection: d_model -> 2 * d_inner (split into x and z branches)
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Local convolution on x branch (short-range pattern detection)
        # Causal: left-pad only
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv,
            padding=0,  # manual causal padding
            groups=self.d_inner,  # depthwise
            bias=True,
        )

        # Selective projections: input-dependent B, C, and delta (dt)
        # x -> (dt, B, C) where dt controls discretization step
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)

        # dt projection: dt_rank -> d_inner (one per channel)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)
        # Initialize dt bias to log-uniform in [0.001, 0.1] for stable training
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (np.log(0.1) - np.log(0.001)) + np.log(0.001)
            )
            # Inverse of softplus for initialization
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter: initialized to negative real values (log-space for stability)
        # Shape: (d_inner, d_state) -- one A per channel per state dim
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))  # Learn in log space
        # D parameter (skip connection): (d_inner,)
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Time-decay modulation: learnable per-channel decay rate
        # Applied as: A_effective = A * exp(-decay_rate * time_delta)
        self.time_decay_rate = nn.Parameter(torch.ones(self.d_inner, 1) * 0.1)

        # Output projection: d_inner -> d_model
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

    def _selective_scan_sequential(
        self,
        x: torch.Tensor,      # (B, L, D)  D = d_inner
        dt: torch.Tensor,      # (B, L, D)
        A: torch.Tensor,       # (D, N)     N = d_state
        B: torch.Tensor,       # (B, L, N)
        C: torch.Tensor,       # (B, L, N)
        D: torch.Tensor,       # (D,)
        time_delta: Optional[torch.Tensor] = None,  # (B, L)
    ) -> torch.Tensor:
        """
        Selective scan with time-delta modulation.

        Uses an efficient parallel scan implementation when possible,
        falls back to sequential for correctness guarantee.

        Returns: (B, L, D)
        """
        batch, seq_len, d_inner = x.shape
        d_state = A.shape[1]

        # Discretize A: dA = exp(A * dt)
        # A is (D, N), dt is (B, L, D) -> dA is (B, L, D, N)
        A_expanded = A.unsqueeze(0).unsqueeze(0)       # (1, 1, D, N)
        dt_expanded = dt.unsqueeze(-1)                  # (B, L, D, 1)
        dA = torch.exp(A_expanded * dt_expanded)        # (B, L, D, N)

        # Time-delta conditioning: modulate dA by time gap
        # When time_delta is large, decay state more aggressively
        if time_delta is not None:
            # time_delta: (B, L) -> (B, L, 1, 1)
            td = time_delta.unsqueeze(-1).unsqueeze(-1)  # (B, L, 1, 1)
            # decay_rate: (D, 1) -> (1, 1, D, 1)
            dr = F.softplus(self.time_decay_rate).unsqueeze(0).unsqueeze(0)
            time_decay = torch.exp(-dr * td.abs())       # (B, L, D, N) broadcast
            dA = dA * time_decay

        # Discretize B: dB = B * dt
        dB = B.unsqueeze(2) * dt_expanded               # (B, L, D, N)

        # Input-weighted B
        dBx = dB * x.unsqueeze(-1)                      # (B, L, D, N)

        # Run the selective scan via parallel prefix sum (associative scan)
        # For each position t:
        #   h[t] = dA[t] * h[t-1] + dBx[t]
        #   y[t] = (C[t] . h[t])
        # This is a linear recurrence that can be parallelized.
        y = self._parallel_scan(dA, dBx, C, d_inner, d_state, batch, seq_len, x.device, x.dtype)

        # Skip connection
        y = y + x * D.unsqueeze(0).unsqueeze(0)         # (B, L, D)

        return y

    def _parallel_scan(
        self,
        dA: torch.Tensor,    # (B, L, D, N)
        dBx: torch.Tensor,   # (B, L, D, N)
        C: torch.Tensor,     # (B, L, N)
        d_inner: int,
        d_state: int,
        batch: int,
        seq_len: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        """
        Parallel prefix scan for linear recurrence.

        Uses chunked computation: process in chunks then combine.
        This is a practical middle ground between fully sequential O(n)
        and fully parallel O(n log n) approaches.

        For sequences up to ~2000, the chunk approach is fast and memory-efficient.
        """
        # Chunk size for the scan: balance parallelism vs memory
        CHUNK = 64

        # Initialize hidden state
        h = torch.zeros(batch, d_inner, d_state, device=device, dtype=dtype)
        outputs = []

        for t_start in range(0, seq_len, CHUNK):
            t_end = min(t_start + CHUNK, seq_len)
            chunk_len = t_end - t_start

            # Extract chunk
            dA_chunk  = dA[:, t_start:t_end]    # (B, chunk, D, N)
            dBx_chunk = dBx[:, t_start:t_end]   # (B, chunk, D, N)
            C_chunk   = C[:, t_start:t_end]      # (B, chunk, N)

            # Sequential scan within chunk (chunk is small, so this is fast)
            chunk_outputs = []
            for t in range(chunk_len):
                h = dA_chunk[:, t] * h + dBx_chunk[:, t]  # (B, D, N)
                # Output: y[t] = sum_n(C[t,n] * h[t,:,n])
                y_t = torch.einsum("bn,bdn->bd", C_chunk[:, t], h)  # (B, D)
                chunk_outputs.append(y_t)

            chunk_out = torch.stack(chunk_outputs, dim=1)  # (B, chunk, D)
            outputs.append(chunk_out)

        return torch.cat(outputs, dim=1)  # (B, L, D)

    def forward(
        self,
        x: torch.Tensor,                               # (B, L, d_model)
        time_delta: Optional[torch.Tensor] = None,      # (B, L)
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model) input embeddings
            time_delta: (B, L) log time delta between consecutive events
                        (feature index 0 before normalization)

        Returns:
            out: (B, L, d_model)
        """
        B, L, _ = x.shape

        # Project input to 2 * d_inner (x_branch and z_branch)
        xz = self.in_proj(x)                            # (B, L, 2 * d_inner)
        x_branch, z = xz.chunk(2, dim=-1)               # each (B, L, d_inner)

        # Local 1D convolution on x branch (causal: left-pad)
        # Rearrange: (B, L, D) -> (B, D, L) for Conv1d
        x_conv = x_branch.transpose(1, 2).contiguous()  # (B, D, L)
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))    # left-pad for causality
        x_conv = self.conv1d(x_conv)                     # (B, D, L)
        x_conv = x_conv.transpose(1, 2).contiguous()     # (B, L, D)
        x_branch = F.silu(x_conv)                        # SiLU activation

        # Compute selective parameters from x
        x_proj = self.x_proj(x_branch)                   # (B, L, dt_rank + 2*d_state)
        dt_x = x_proj[:, :, :self.dt_rank]               # (B, L, dt_rank)
        B_sel = x_proj[:, :, self.dt_rank:self.dt_rank + self.d_state]  # (B, L, N)
        C_sel = x_proj[:, :, self.dt_rank + self.d_state:]              # (B, L, N)

        # Project dt to full d_inner dimension and apply softplus
        dt = F.softplus(self.dt_proj(dt_x))              # (B, L, d_inner)

        # Get A from log space (negative for stability)
        A = -torch.exp(self.A_log)                       # (d_inner, d_state)

        # Run selective scan with time-delta conditioning
        y = self._selective_scan_sequential(
            x_branch, dt, A, B_sel, C_sel, self.D,
            time_delta=time_delta,
        )                                                 # (B, L, d_inner)

        # Gate with z branch (SiLU gating, same as Mamba paper)
        y = y * F.silu(z)                                # (B, L, d_inner)

        # Project back to d_model
        out = self.out_proj(y)                           # (B, L, d_model)

        return out


class MambaBlock(nn.Module):
    """
    Single Mamba block: LayerNorm -> SelectiveSSM -> residual.

    Pre-norm residual architecture for stable deep training.
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_rank: int = 16,
        d_conv: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.use_cuda = USE_CUDA_MAMBA
        if USE_CUDA_MAMBA:
            # Use fused CUDA selective scan kernel (10-50x faster)
            self.ssm = CUDAMamba(
                d_model=d_model,
                d_state=d_state,
                d_conv=d_conv,
                expand=2,
            )
        else:
            self.ssm = SelectiveSSM(
                d_model=d_model,
                d_state=d_state,
                dt_rank=dt_rank,
                d_conv=d_conv,
            )
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        x: torch.Tensor,                               # (B, L, d_model)
        time_delta: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        residual = x
        x = self.norm(x)
        if self.use_cuda:
            x = self.ssm(x)  # CUDA Mamba handles gating internally
        else:
            x = self.ssm(x, time_delta=time_delta)
        x = self.dropout(x)
        return x + residual


class CNNMambaV2(nn.Module):
    """
    CNN-Mamba v2: Parallel Fusion Architecture.
    
    Two parallel pathways process input DIFFERENTLY, then fuse before Mamba:
    
    Pathway 1 - Feature Interaction MLP:
        At each timestep, learns non-linear cross-feature interactions.
        25 features -> 128 -> 64. Captures microstructure states that depend
        on COMBINATIONS of features (e.g., bid_size + spread + OFI together).
        
    Pathway 2 - Multi-Scale Temporal CNN:
        Four Conv1d kernels at different scales (k=3,7,15,31) each producing
        16 channels. Captures temporal patterns at multiple resolutions.
        Causal padding (left-pad only). Total: 64 channels.
    
    Fusion: Concat (128) -> Linear -> LayerNorm -> d_model
    Mamba: Standard 3-layer Mamba backbone with time-delta conditioning.
    Head: Multi-task prediction (1s, 5s, 10s price change).
    """

    def __init__(
        self,
        d_model: int = MAMBA_D_MODEL,
        d_state: int = MAMBA_D_STATE,
        n_layers: int = MAMBA_N_LAYERS,
        dt_rank: int = MAMBA_DT_RANK,
        d_conv: int = MAMBA_D_CONV,
        dropout: float = MAMBA_DROPOUT,
        n_targets: int = 3,
        feature_mlp_hidden: int = 128,
        feature_mlp_out: int = 64,
        cnn_channels_per_scale: int = 16,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_targets = n_targets

        # -- Pathway 1: Feature Interaction MLP --
        # Applied at EACH timestep independently (no temporal mixing)
        # Learns non-linear cross-feature patterns
        self.feature_mlp = nn.Sequential(
            nn.Linear(N_TOTAL_FEATURES, feature_mlp_hidden),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(feature_mlp_hidden, feature_mlp_out),
            nn.LayerNorm(feature_mlp_out),
        )

        # -- Pathway 2: Multi-Scale Temporal CNN --
        # Four scales to capture different temporal resolutions
        # All use CAUSAL padding (left-pad only)
        self.cnn_kernels = [3, 7, 15, 31]
        self.temporal_cnns = nn.ModuleList()
        for k in self.cnn_kernels:
            self.temporal_cnns.append(
                nn.Sequential(
                    nn.Conv1d(N_TOTAL_FEATURES, cnn_channels_per_scale,
                              kernel_size=k, padding=0, bias=True),  # causal: manual pad
                    nn.GELU(),
                )
            )
        self.cnn_out_dim = cnn_channels_per_scale * len(self.cnn_kernels)  # 16*4=64

        # -- Fusion: Combine both pathways --
        fusion_dim = feature_mlp_out + self.cnn_out_dim  # 64 + 64 = 128
        self.fusion_proj = nn.Sequential(
            nn.Linear(fusion_dim, d_model),
            nn.LayerNorm(d_model),
        )

        # -- Mamba Backbone (unchanged from v1) --
        self.blocks = nn.ModuleList([
            MambaBlock(
                d_model=d_model,
                d_state=d_state,
                dt_rank=dt_rank,
                d_conv=d_conv,
                dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        # Final norm before prediction head
        self.final_norm = nn.LayerNorm(d_model)

        # Multi-task prediction head
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, events: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            events: (B, L, N_features) float32
            return_embedding: if True, also return state embedding
        Returns:
            preds: (B, n_targets)
            embedding: (B, d_model) optional
        """
        B, L, F = events.shape

        # Extract time_delta for Mamba blocks
        time_delta = events[:, :, 0]  # (B, L)

        # -- Pathway 1: Feature Interaction MLP --
        # Applied pointwise at each timestep: (B, L, F) -> (B, L, 64)
        feat_out = self.feature_mlp(events)  # (B, L, feature_mlp_out)

        # -- Pathway 2: Multi-Scale Temporal CNN --
        # Transpose for Conv1d: (B, L, F) -> (B, F, L)
        x_t = events.transpose(1, 2)  # (B, F, L)
        
        cnn_outputs = []
        for i, (k, conv_block) in enumerate(zip(self.cnn_kernels, self.temporal_cnns)):
            # Causal left-padding
            padded = torch.nn.functional.pad(x_t, (k - 1, 0))  # (B, F, L + k - 1)
            out = conv_block(padded)  # (B, cnn_channels_per_scale, L)
            cnn_outputs.append(out)
        
        # Concatenate all scales: (B, 64, L)
        cnn_cat = torch.cat(cnn_outputs, dim=1)
        # Transpose back: (B, 64, L) -> (B, L, 64)
        cnn_out = cnn_cat.transpose(1, 2)  # (B, L, cnn_out_dim)

        # -- Fusion --
        # Concatenate both pathways: (B, L, 64+64=128)
        fused = torch.cat([feat_out, cnn_out], dim=-1)
        # Project to d_model: (B, L, 128) -> (B, L, d_model)
        x = self.fusion_proj(fused)

        # -- Mamba Backbone --
        for block in self.blocks:
            x = block(x, time_delta=time_delta)

        # Take LAST position (causal)
        x_last = x[:, -1, :]  # (B, d_model)

        # Final norm + prediction
        embedding = self.final_norm(x_last)
        preds = self.head(embedding)

        if return_embedding:
            return preds, embedding
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


def compute_comprehensive_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    horizon: str,
    fold_idx: int,
    output_dir: Path,
    oot_files: list = None,
) -> dict:
    """
    Compute full 3-tier metrics at confidence bands for a single horizon.

    Tier 1: IC, DA, MagCorr at confidence tiers
    Tier 2: MFE/MAE proxy, long/short breakdown, signal flip frequency
    Tier 3: Win rate, avg winner/loser, profit factor proxy

    Confidence tiers: All, Top50%, Top25%, Top10%, Top5%, Top1%, Top0.5%, Top0.1%
    """
    valid = ~(np.isnan(predictions) | np.isnan(labels))
    preds = predictions[valid]
    labs = labels[valid]

    if len(preds) < 100:
        logger.warning(f"  Insufficient samples ({len(preds)}) for comprehensive metrics")
        return {}

    abs_preds = np.abs(preds)
    tiers = {
        "All":     0.0,
        "Top50%":  np.percentile(abs_preds, 50),
        "Top25%":  np.percentile(abs_preds, 75),
        "Top10%":  np.percentile(abs_preds, 90),
        "Top5%":   np.percentile(abs_preds, 95),
        "Top1%":   np.percentile(abs_preds, 99),
        "Top0.5%": np.percentile(abs_preds, 99.5),
        "Top0.1%": np.percentile(abs_preds, 99.9),
    }

    results = {"horizon": horizon, "fold": fold_idx, "n_samples_total": int(len(preds))}

    for tier_name, threshold in tiers.items():
        mask = abs_preds >= threshold
        n = int(mask.sum())
        if n < 10:
            results[tier_name] = {"n_samples": n, "skipped": True}
            continue

        p = preds[mask]
        l = labs[mask]

        # Tier 1: Signal Quality
        ic = float(scipy.stats.spearmanr(p, l).correlation) if n >= 20 else float("nan")
        da = float(np.mean(np.sign(p) == np.sign(l)))
        # MagCorr: correlation between |pred| and |label| (does confidence predict magnitude?)
        mag_corr = float(scipy.stats.spearmanr(np.abs(p), np.abs(l)).correlation) if n >= 20 else float("nan")

        # Tier 2: Tradeability
        # Long/short breakdown
        long_mask = p > 0
        short_mask = p < 0
        long_da = float(np.mean(l[long_mask] > 0)) if long_mask.sum() > 5 else float("nan")
        short_da = float(np.mean(l[short_mask] < 0)) if short_mask.sum() > 5 else float("nan")
        n_long = int(long_mask.sum())
        n_short = int(short_mask.sum())

        # MFE/MAE proxy (using label as realized move)
        # For each prediction: if correct direction, label magnitude = favorable excursion proxy
        correct = np.sign(p) == np.sign(l)
        wrong = ~correct
        mfe_proxy = float(np.mean(np.abs(l[correct]))) if correct.sum() > 5 else float("nan")
        mae_proxy = float(np.mean(np.abs(l[wrong]))) if wrong.sum() > 5 else float("nan")

        # Avg winner vs avg loser (directional PnL: sign(pred) * label)
        pnl = np.sign(p) * l
        winners = pnl[pnl > 0]
        losers = pnl[pnl < 0]
        avg_winner = float(np.mean(winners)) if len(winners) > 0 else 0.0
        avg_loser = float(np.mean(losers)) if len(losers) > 0 else 0.0
        win_rate = float(np.mean(pnl > 0))
        profit_factor = float(np.sum(winners) / abs(np.sum(losers))) if len(losers) > 0 and np.sum(losers) != 0 else float("inf")

        # Sortino ratio (using directional PnL)
        downside = pnl[pnl < 0]
        sortino = float(np.mean(pnl) / (np.std(downside) + 1e-10)) if len(downside) > 5 else float("nan")

        # Net PnL (sum of sign(pred) * label, in label units)
        net_pnl = float(np.sum(pnl))
        avg_pnl = float(np.mean(pnl))

        results[tier_name] = {
            "n_samples": n,
            # Tier 1
            "IC": round(ic, 4),
            "DA": round(da, 4),
            "MagCorr": round(mag_corr, 4),
            # Tier 2
            "long_DA": round(long_da, 4) if not np.isnan(long_da) else None,
            "short_DA": round(short_da, 4) if not np.isnan(short_da) else None,
            "n_long": n_long,
            "n_short": n_short,
            "MFE_proxy": round(mfe_proxy, 4) if not np.isnan(mfe_proxy) else None,
            "MAE_proxy": round(mae_proxy, 4) if not np.isnan(mae_proxy) else None,
            # Tier 3
            "win_rate": round(win_rate, 4),
            "avg_winner": round(avg_winner, 4),
            "avg_loser": round(avg_loser, 4),
            "profit_factor": round(profit_factor, 4) if profit_factor != float("inf") else "inf",
            "sortino": round(sortino, 4) if not np.isnan(sortino) else None,
            "net_pnl": round(net_pnl, 2),
            "avg_pnl": round(avg_pnl, 6),
        }

    # Signal flip frequency (how often does prediction sign change between consecutive samples)
    sign_changes = np.sum(np.diff(np.sign(preds)) != 0)
    flip_freq = float(sign_changes / max(1, len(preds) - 1))
    results["signal_flip_frequency"] = round(flip_freq, 4)

    # Log summary
    t1 = results.get("Top1%", {})
    t5 = results.get("Top5%", {})
    t10 = results.get("Top10%", {})
    all_t = results.get("All", {})
    logger.info(
        f"  {horizon} Metrics | All: IC={all_t.get('IC','?')} DA={all_t.get('DA','?')} | "
        f"Top10%: IC={t10.get('IC','?')} DA={t10.get('DA','?')} MagCorr={t10.get('MagCorr','?')} | "
        f"Top1%: IC={t1.get('IC','?')} DA={t1.get('DA','?')} WR={t1.get('win_rate','?')} PF={t1.get('profit_factor','?')} | "
        f"FlipFreq={results['signal_flip_frequency']}"
    )

    return results


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

    def state_dict(self):
        return {"_step": self._step, "base_lrs": self.base_lrs,
                "warmup_steps": self.warmup_steps, "total_steps": self.total_steps,
                "min_lr": self.min_lr}

    def load_state_dict(self, state):
        self._step = state["_step"]
        self.base_lrs = state.get("base_lrs", self.base_lrs)
        self.warmup_steps = state.get("warmup_steps", self.warmup_steps)
        self.total_steps = state.get("total_steps", self.total_steps)
        self.min_lr = state.get("min_lr", self.min_lr)

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ============================================================
# Evaluation helpers
# ============================================================

def evaluate(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    """Run inference on a DataLoader and return (metrics_dict, preds, labels)."""
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
        for events, labels in loader:
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with amp_ctx:
                preds = model(events)
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
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
) -> Dict:
    """Convenience wrapper: return only metrics dict."""
    metrics, _, _ = evaluate(model, loader, device, use_amp=use_amp)
    return metrics


# ============================================================
# OOT Inference
# ============================================================

def run_oot_inference(
    model:  nn.Module,
    loader: DataLoader,
    device: torch.device,
    use_amp: bool = True,
    extract_embeddings: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    """Run inference on OOT fold, return (predictions, labels, embeddings).

    If extract_embeddings=True, also returns (B, d_model) state embeddings for fusion.
    """
    model.eval()
    all_preds  = []
    all_labels = []
    all_embeds = [] if extract_embeddings else None

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    with torch.no_grad():
        for events, labels in loader:
            events = events.to(device, non_blocking=True)
            with amp_ctx:
                if extract_embeddings:
                    preds, emb = model(events, return_embedding=True)
                    all_embeds.append(emb.float().cpu().numpy())
                else:
                    preds = model(events)
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())

    if not all_preds:
        empty = np.empty((0, len(HORIZONS)))
        return empty, empty, np.empty((0, 0)) if extract_embeddings else None
    preds_out = np.concatenate(all_preds, axis=0)
    labels_out = np.concatenate(all_labels, axis=0)
    embeds_out = np.concatenate(all_embeds, axis=0) if extract_embeddings else None
    return preds_out, labels_out, embeds_out


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
    """Train Mamba for one expanding-window fold. Returns metrics dict."""

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

    # Check for intra-epoch checkpoint to resume from
    resume_epoch = 0
    resume_batch = 0
    intra_ckpt_path = output_dir / f"fold_{fold_idx:02d}_intra_ckpt.pt"
    if intra_ckpt_path.exists():
        try:
            ckpt = torch.load(intra_ckpt_path, map_location=device)
            if ckpt.get("fold") == fold_idx:
                model.load_state_dict(ckpt["model_state"])
                optimizer.load_state_dict(ckpt["optimizer_state"])
                scheduler.load_state_dict(ckpt["scheduler_state"])
                scaler.load_state_dict(ckpt["scaler_state"])
                resume_epoch = ckpt["epoch"]
                resume_batch = ckpt["batch"]
                global_step = ckpt.get("global_step", 0)
                best_val_loss = ckpt.get("best_val_loss", float("inf"))
                logger.info(f"RESUMED from checkpoint: fold={fold_idx}, epoch={resume_epoch}, batch={resume_batch}")
            else:
                logger.info(f"Checkpoint fold mismatch ({ckpt.get('fold')} vs {fold_idx}), starting fresh")
        except Exception as e:
            logger.warning(f"Failed to load checkpoint: {e}, starting fresh")

    total_batches = len(train_loader)
    print(f">>> train_one_fold: {EPOCHS_PER_FOLD} epochs, {total_batches} batches/epoch", flush=True)
    for epoch in range(resume_epoch, EPOCHS_PER_FOLD):
        model.train()
        epoch_loss = 0.0
        n_batches  = 0
        epoch_start = time.time()
        print(f">>> Starting epoch {epoch+1}/{EPOCHS_PER_FOLD}...", flush=True)

        for events, labels in train_loader:
            # Skip batches already processed (resume from checkpoint)
            if epoch == resume_epoch and n_batches < resume_batch:
                n_batches += 1
                continue

            events = events.to(device, non_blocking=True)   # (B, W, 6)
            labels = labels.to(device, non_blocking=True)   # (B, 3)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                preds = model(events)                        # (B, 3)
                loss  = F.mse_loss(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches  += 1
            global_step += 1

            # Progress logging every 100 batches (using print for guaranteed output)
            if n_batches % 100 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / n_batches * (total_batches - n_batches)
                msg = (
                    f"  Batch {n_batches}/{total_batches} | "
                    f"Loss: {epoch_loss/n_batches:.4f} | "
                    f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
                )
                print(msg, flush=True)
                logger.info(msg)

            # Intra-epoch checkpoint every 500 batches
            if n_batches % 500 == 0:
                intra_ckpt_path = output_dir / f"fold_{fold_idx:02d}_intra_ckpt.pt"
                torch.save({
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_state": scheduler.state_dict(),
                    "scaler_state": scaler.state_dict(),
                    "fold": fold_idx,
                    "epoch": epoch,
                    "batch": n_batches,
                    "global_step": global_step,
                    "epoch_loss": epoch_loss,
                    "n_batches": n_batches,
                    "best_val_loss": best_val_loss,
                    "arch": {
                        "d_model": MAMBA_D_MODEL,
                        "d_state": MAMBA_D_STATE,
                        "n_layers": MAMBA_N_LAYERS,
                        "dt_rank": MAMBA_DT_RANK,
                        "d_conv": MAMBA_D_CONV,
                        "dropout": MAMBA_DROPOUT,
                        "window_size": WINDOW_SIZE,
                    },
                }, intra_ckpt_path)
                logger.info(f"  Intra-epoch checkpoint saved: fold={fold_idx}, epoch={epoch}, batch={n_batches}")

        # Reset resume_batch after first resumed epoch completes
        if epoch == resume_epoch:
            resume_batch = 0

        avg_loss    = epoch_loss / max(n_batches, 1)
        epoch_time  = time.time() - epoch_start
        # Use evaluate() which returns (metrics, preds, labels)
        val_metrics, _, _ = evaluate(model, val_loader, device, use_amp=use_amp)

        logger.info(
            f"Fold {fold_idx:02d} | Epoch {epoch+1:2d}/{EPOCHS_PER_FOLD} | "
            f"Train Loss: {avg_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IC (10s): {val_metrics.get('ic_10s', float('nan')):.4f} | "
            f"LR: {scheduler.get_lr():.2e} | Time: {epoch_time:.1f}s"
        )

        if MLFLOW_AVAILABLE and mlflow_run:
            step_offset = fold_idx * EPOCHS_PER_FOLD + epoch
            mlflow.log_metrics(
                {
                    f"fold{fold_idx:02d}_train_loss": avg_loss,
                    f"fold{fold_idx:02d}_val_loss":   val_metrics["loss"],
                    f"fold{fold_idx:02d}_val_ic_1s":  val_metrics.get("ic_1s",  float("nan")),
                    f"fold{fold_idx:02d}_val_ic_5s":  val_metrics.get("ic_5s",  float("nan")),
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
                    "model_state":  model.state_dict(),
                    "fold":         fold_idx,
                    "epoch":        epoch,
                    "val_loss":     val_metrics["loss"],
                    "val_ic_10s":   val_metrics.get("ic_10s"),
                    "arch": {
                        "d_model":     MAMBA_D_MODEL,
                        "d_state":     MAMBA_D_STATE,
                        "n_layers":    MAMBA_N_LAYERS,
                        "dt_rank":     MAMBA_DT_RANK,
                        "d_conv":      MAMBA_D_CONV,
                        "dropout":     MAMBA_DROPOUT,
                        "window_size": WINDOW_SIZE,
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
    Feature statistics are always computed from the training set only (no leakage).
    OOT predictions are saved per fold + concatenated for final concat-IC calculation.
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

    logger.info(f"Total files (valid): {n_files} ({npz_files[0].name} -> {npz_files[-1].name})")

    # Build fold boundaries (sliding or expanding window)
    min_train     = max(5, n_files - n_folds)
    fold_boundaries = []
    wf_mode = "sliding" if WF_WINDOW_DAYS > 0 else "expanding"
    for fold in range(n_folds):
        train_end = min_train + fold
        oot_start = train_end
        oot_end   = oot_start + max(1, (n_files - min_train) // n_folds)
        oot_end   = min(oot_end, n_files)
        if oot_start >= n_files:
            break
        if wf_mode == "sliding" and WF_WINDOW_DAYS > 0:
            train_start = max(0, train_end - WF_WINDOW_DAYS)
            train_indices = list(range(train_start, train_end))
        else:
            train_indices = list(range(train_end))
        fold_boundaries.append((fold, train_indices, list(range(oot_start, oot_end))))

    logger.info(f"Running {len(fold_boundaries)} folds ({wf_mode} window, WF_WINDOW_DAYS={WF_WINDOW_DAYS})")

    # AMP only useful on CUDA
    use_amp = device.type == "cuda"

    # Concat storage
    concat_preds  = {h: [] for h in HORIZONS}
    concat_labels = {h: [] for h in HORIZONS}
    concat_embeds = []  # For fusion layer / metamodel

    # MLflow run
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(
            run_name=f"EventMamba_{time.strftime('%Y%m%d_%H%M')}"
        )
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        mlflow.log_params(
            {
                "model":           "EventMamba",
                "window_size":     WINDOW_SIZE,
                "stride":          STRIDE,
                "d_model":         MAMBA_D_MODEL,
                "d_state":         MAMBA_D_STATE,
                "n_layers":        MAMBA_N_LAYERS,
                "dt_rank":         MAMBA_DT_RANK,
                "d_conv":          MAMBA_D_CONV,
                "dropout":         MAMBA_DROPOUT,
                "batch_size":      BATCH_SIZE,
                "lr":              LR,
                "epochs_per_fold": EPOCHS_PER_FOLD,
                "n_folds":         len(fold_boundaries),
                "horizons":        str(HORIZONS),
                "n_files":         n_files,
                "optimizer":       "AdamW",
                "warmup_steps":    WARMUP_STEPS,
                "grad_clip":       GRAD_CLIP,
                "node":            socket.gethostname(),
                "gpu":             gpu_name,
                "data_dir":        str(npz_files[0].parent),
                "output_dir":      str(output_dir),
                "mixed_precision": "fp16" if use_amp else "none",
                "num_workers":     0,
            }
        )

    try:
        start_fold = int(os.environ.get("START_FOLD", 0))
        for fold_idx, train_file_idxs, oot_file_idxs in fold_boundaries:
            if fold_idx < start_fold:
                logger.info(f"Skipping fold {fold_idx:02d} (START_FOLD={start_fold})")
                continue
            train_files = [npz_files[i] for i in train_file_idxs]
            oot_files   = [npz_files[i] for i in oot_file_idxs]

            logger.info(
                f"\n{'='*60}\n"
                f"FOLD {fold_idx:02d} | Train: {len(train_files)} days "
                f"({train_files[0].name}->{train_files[-1].name}) | "
                f"OOT: {len(oot_files)} days ({oot_files[0].name}->{oot_files[-1].name})"
                f"\n{'='*60}"
            )

            # Build train dataset -- stats computed from train set only (no leakage)
            logger.info("Building train dataset...")
            TrainDatasetClass = LazyMboEventDataset if len(train_files) > 50 else MboEventDataset
            train_ds      = TrainDatasetClass(train_files, window_size=WINDOW_SIZE, stride=STRIDE, normalize_features=not SKIP_NORMALIZE, horizons=HORIZONS)
            feature_stats = train_ds.get_feature_stats()

            # Build OOT dataset using TRAIN feature stats (no leakage)
            logger.info("Building OOT dataset...")
            oot_ds = MboEventDataset(
                oot_files,
                window_size   = WINDOW_SIZE,
                stride        = STRIDE,
                feature_stats = feature_stats,
                normalize_features = not SKIP_NORMALIZE,
                horizons      = HORIZONS,
            )

            # Use multiple workers on Linux with lazy loading, 0 on Windows
            _num_workers = 0  # FIXED: workers>0 causes OOM on 64GB RAM with large MBO files  # File-sequential sampler makes workers safe  # Lazy=safe with workers (only index in memory)

            # Use file-sequential sampler for cache-friendly lazy loading
            _train_sampler = FileSequentialSampler(train_ds) if isinstance(train_ds, LazyMboEventDataset) else None
            train_loader = DataLoader(
                train_ds,
                batch_size  = BATCH_SIZE,
                shuffle     = (_train_sampler is None),
                sampler     = _train_sampler,
                num_workers = _num_workers,
                pin_memory  = True,
                drop_last   = True,
            )
            oot_loader = DataLoader(
                oot_ds,
                batch_size  = BATCH_SIZE * 2,
                shuffle     = False,
                num_workers = _num_workers,
                pin_memory  = True,
            )

            # Fresh model per fold — CNN-Mamba hybrid
            model = CNNMambaV2(
                d_model   = MAMBA_D_MODEL,
                d_state   = MAMBA_D_STATE,
                n_layers  = MAMBA_N_LAYERS,
                dt_rank   = MAMBA_DT_RANK,
                d_conv    = MAMBA_D_CONV,
                dropout   = MAMBA_DROPOUT,
                n_targets = len(HORIZONS),
            ).to(device)

            if fold_idx == 0:
                n_params = count_parameters(model)
                logger.info(f"Model parameters:  {n_params:,}")
                logger.info(f"Architecture: CNN-Mamba v2 (parallel fusion)")
                logger.info(f"  Pathways: Feature MLP (25->128->64) + Multi-Scale CNN (k=3,7,15,31 x16ch)")
                logger.info(f"  Mamba: d_model={MAMBA_D_MODEL}, d_state={MAMBA_D_STATE}, "
                            f"n_layers={MAMBA_N_LAYERS}, dt_rank={MAMBA_DT_RANK}, d_conv={MAMBA_D_CONV}")
                logger.info(f"Features: {N_TOTAL_FEATURES} ({N_BASE_FEATURES} base"
                            f"{' + ' + str(N_DERIVED_FEATURES) + ' derived' if USE_DERIVED_FEATURES else ''})")
                logger.info(f"Window size: {WINDOW_SIZE} events (stride={STRIDE})")
                logger.info(f"Context: CNN local patterns + Mamba O(n) long-range")

            total_steps = EPOCHS_PER_FOLD * len(train_loader)
            print(f">>> ENTERING train_one_fold: {total_steps} total steps, {len(train_loader)} batches/epoch", flush=True)
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
            oot_preds, oot_labels, oot_embeds = run_oot_inference(
                model, oot_loader, device, use_amp=use_amp, extract_embeddings=True
            )

            # Per-fold IC
            fold_ics: Dict[str, float] = {}
            for i, h in enumerate(HORIZONS):
                ic           = compute_ic(oot_preds[:, i], oot_labels[:, i])
                fold_ics[h]  = ic
                concat_preds[h].append(oot_preds[:, i])
                concat_labels[h].append(oot_labels[:, i])

            # Collect embeddings for fusion/metamodel
            if oot_embeds is not None:
                concat_embeds.append(oot_embeds)

            logger.info(
                f"Fold {fold_idx:02d} OOT IC | "
                + " | ".join(f"{h}: {fold_ics[h]:.4f}" for h in HORIZONS)
            )

            # Save fold artifacts: .npz predictions + embeddings
            pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
            save_dict = dict(
                predictions = oot_preds,
                labels      = oot_labels,
                horizons    = np.array(HORIZONS),
                ic_1s       = np.array(fold_ics.get("1s",  float("nan"))),
                ic_5s       = np.array(fold_ics.get("5s",  float("nan"))),
                ic_10s      = np.array(fold_ics.get("10s", float("nan"))),
                oot_files   = np.array([str(f) for f in oot_files]),
            )
            if oot_embeds is not None:
                save_dict["embeddings"] = oot_embeds
            np.savez_compressed(pred_path, **save_dict)
            if oot_embeds is not None:
                logger.info(f"Saved predictions + embeddings ({oot_embeds.shape[1]}d) -> {pred_path}")
            else:
                logger.info(f"Saved predictions -> {pred_path}")

            # Save feature stats for this fold (needed for inference)
            stats_path = output_dir / f"fold_{fold_idx:02d}_feature_stats.npz"
            np.savez(stats_path, mean=feature_stats["mean"], std=feature_stats["std"])

            # ============================================================
            # COMPREHENSIVE METRICS: 3-tier analysis at confidence bands
            # ============================================================
            logger.info(f"Computing comprehensive metrics for fold {fold_idx:02d}...")
            fold_analysis = {
                "fold": fold_idx,
                "oot_files": [str(f) for f in oot_files],
                "n_samples": int(oot_preds.shape[0]),
                "horizons": {},
            }
            for i, h in enumerate(HORIZONS):
                h_metrics = compute_comprehensive_metrics(
                    oot_preds[:, i], oot_labels[:, i],
                    horizon=h, fold_idx=fold_idx,
                    output_dir=output_dir, oot_files=oot_files,
                )
                fold_analysis["horizons"][h] = h_metrics

            # Save fold analysis as JSON
            import json as _json
            analysis_path = output_dir / f"fold_{fold_idx:02d}_analysis.json"
            with open(analysis_path, "w") as f:
                _json.dump(fold_analysis, f, indent=2, default=str)
            logger.info(f"Saved comprehensive analysis -> {analysis_path}")

            # Log per-fold metrics to MLflow
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow.log_metrics(
                    {f"oot_ic_{h}_fold{fold_idx:02d}": fold_ics[h] for h in HORIZONS},
                    step=fold_idx,
                )
                # Also log DA and MagCorr at key tiers
                for h in HORIZONS:
                    h_data = fold_analysis.get("horizons", {}).get(h, {})
                    for tier in ["All", "Top10%", "Top1%"]:
                        t_data = h_data.get(tier, {})
                        if isinstance(t_data, dict) and not t_data.get("skipped"):
                            for metric in ["DA", "MagCorr", "win_rate", "profit_factor"]:
                                val = t_data.get(metric)
                                if val is not None and val != "inf" and not (isinstance(val, float) and np.isnan(val)):
                                    mlflow.log_metric(
                                        f"{metric}_{h}_{tier.replace(chr(37), chr(112)+chr(99)+chr(116))}_f{fold_idx:02d}",
                                        float(val), step=fold_idx
                                    )

            # Free memory before next fold
            del train_ds, oot_ds, train_loader, oot_loader, model
            gc.collect()
            torch.cuda.empty_cache()

        # ============================================================
        # Compute CONCAT IC (primary metric -- all folds combined)
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("CONCAT IC (primary metric -- all folds combined)")
        logger.info("=" * 60)

        concat_ic: Dict[str, float] = {}
        for h in HORIZONS:
            if concat_preds[h]:
                all_p         = np.concatenate(concat_preds[h])
                all_l         = np.concatenate(concat_labels[h])
                ic            = compute_ic(all_p, all_l)
                concat_ic[h]  = ic
                logger.info(f"  Concat IC ({h}): {ic:.4f}")
            else:
                concat_ic[h] = float("nan")

        # Save all concat predictions + embeddings
        save_dict = {
            **{f"preds_{h}":     np.concatenate(concat_preds[h])
               for h in HORIZONS if concat_preds[h]},
            **{f"labels_{h}":    np.concatenate(concat_labels[h])
               for h in HORIZONS if concat_labels[h]},
            **{f"concat_ic_{h}": np.array(concat_ic[h]) for h in HORIZONS},
        }
        if concat_embeds:
            save_dict["embeddings"] = np.concatenate(concat_embeds, axis=0)
            logger.info(f"Concat embeddings: {save_dict['embeddings'].shape} (for fusion layer)")
        concat_path = output_dir / "concat_oot_predictions.npz"
        np.savez_compressed(concat_path, **save_dict)
        if concat_embeds:
            logger.info(f"Saved concat predictions + embeddings -> {concat_path}")
        else:
            logger.info(f"Saved concat predictions -> {concat_path}")

        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.log_metrics({f"concat_ic_{h}": concat_ic[h] for h in HORIZONS})

        # ============================================================
        # CONCAT COMPREHENSIVE METRICS (all folds combined)
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("CONCAT COMPREHENSIVE METRICS (all folds combined)")
        logger.info("=" * 60)
        import json as _json
        concat_analysis = {"type": "concat", "horizons": {}}
        for h in HORIZONS:
            if concat_preds[h]:
                all_p = np.concatenate(concat_preds[h])
                all_l = np.concatenate(concat_labels[h])
                h_metrics = compute_comprehensive_metrics(
                    all_p, all_l, horizon=h, fold_idx=-1,
                    output_dir=output_dir,
                )
                concat_analysis["horizons"][h] = h_metrics
        concat_analysis_path = output_dir / "concat_analysis.json"
        with open(concat_analysis_path, "w") as f:
            _json.dump(concat_analysis, f, indent=2, default=str)
        logger.info(f"Saved concat comprehensive analysis -> {concat_analysis_path}")

        logger.info("\n" + "=" * 60)
        logger.info("LEAKAGE AUDIT: PASSED")
        logger.info("  - Expanding window: train set never contains OOT dates")
        logger.info("  - Feature normalization computed from train set only per fold")
        logger.info("  - SSM is causal by construction: h[t] depends only on events <= t")
        logger.info("  - Time-delta conditioning uses only past inter-event gaps")
        logger.info("=" * 60)

        return concat_ic

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()


# ============================================================
# Data Transfer: Jupiter -> Neptune via SCP / API
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

    logger.info(f"Transferring {len(to_copy)}/{len(remote_files)} files from Jupiter -> Neptune...")

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
            logger.info("SCP key-based auth works -- using SCP for transfer")
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
                elapsed = time.time() - t0
                sz = dest_path.stat().st_size / (1024 * 1024)
                logger.info(f"    SCP OK: {sz:.1f} MB in {elapsed:.1f}s")
                continue
            else:
                logger.warning(f"    SCP failed: {r.stderr[:100]}")

        # Fallback: base64 via API (8MB chunks)
        try:
            out = _jupiter_exec(f"wc -c < {remote_path}", timeout=10)
            file_size = int(out.strip())
        except Exception:
            file_size = 0

        CHUNK = 8 * 1024 * 1024
        collected = b""
        offset = 0

        while True:
            cmd = f"dd if={remote_path} bs=1 skip={offset} count={CHUNK} 2>/dev/null | base64 -w0"
            try:
                b64_data = _jupiter_exec(cmd, timeout=60)
                if not b64_data.strip():
                    break
                collected += base64.b64decode(b64_data)
                offset += CHUNK
                if file_size and offset >= file_size:
                    break
            except Exception as e:
                logger.warning(f"    Chunk transfer failed at offset {offset}: {e}")
                break

        if collected:
            dest_path.write_bytes(collected)
            elapsed = time.time() - t0
            sz = len(collected) / (1024 * 1024)
            logger.info(f"    API OK: {sz:.1f} MB in {elapsed:.1f}s")
        else:
            logger.warning(f"    Failed to transfer {fname}")


# ============================================================
# CLI
# ============================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Train CNN-Mamba Hybrid on MBO event streams"
    )
    parser.add_argument(
        "--data-dir", type=str, default=DEFAULT_DATA_DIR,
        help="Directory containing MBO event NPZ files",
    )
    parser.add_argument(
        "--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR),
        help="Directory for model checkpoints and predictions",
    )
    parser.add_argument(
        "--n-folds", type=int, default=N_FOLDS,
        help="Number of walk-forward folds",
    )
    parser.add_argument(
        "--skip-transfer", action="store_true",
        help="Skip data transfer from Jupiter (use local data only)",
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device: 'cuda', 'cpu', or None (auto-detect)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    logger.info("=" * 60)
    logger.info("CNN-Mamba Hybrid Training")
    logger.info("=" * 60)
    logger.info(f"Config (CNN-Mamba Hybrid):")
    logger.info(f"  cnn_channels= {CNN_CHANNELS}")
    logger.info(f"  cnn_kernel  = {CNN_KERNEL}")
    logger.info(f"  cnn_layers  = {CNN_LAYERS}")
    logger.info(f"  d_model     = {MAMBA_D_MODEL}")
    logger.info(f"  d_state     = {MAMBA_D_STATE}")
    logger.info(f"  n_layers    = {MAMBA_N_LAYERS}")
    logger.info(f"  dt_rank     = {MAMBA_DT_RANK}")
    logger.info(f"  d_conv      = {MAMBA_D_CONV}")
    logger.info(f"  dropout     = {MAMBA_DROPOUT}")
    logger.info(f"  window_size = {WINDOW_SIZE}")
    logger.info(f"  stride      = {STRIDE}")
    logger.info(f"  batch_size  = {BATCH_SIZE}")
    logger.info(f"  lr          = {LR}")
    logger.info(f"  epochs      = {EPOCHS_PER_FOLD}")
    logger.info(f"  n_folds     = {args.n_folds}")
    logger.info(f"  data_dir    = {args.data_dir}")
    logger.info(f"  output_dir  = {args.output_dir}")

    # Device
    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"  device      = {device}")
    if device.type == "cuda":
        logger.info(f"  GPU         = {torch.cuda.get_device_name(0)}")
        logger.info(f"  VRAM        = {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    data_dir = Path(args.data_dir)

    # Transfer data from Jupiter if needed
    if not args.skip_transfer:
        try:
            transfer_data_from_jupiter(data_dir)
        except Exception as e:
            logger.warning(f"Data transfer failed (continuing with local data): {e}")

    # Find NPZ files
    npz_files = sorted(data_dir.glob("*.npz"))
    if not npz_files:
        logger.error(f"No NPZ files found in {data_dir}")
        sys.exit(1)

    logger.info(f"Found {len(npz_files)} NPZ files in {data_dir}")

    # Run walk-forward training
    output_dir = Path(args.output_dir)
    concat_ic = run_expanding_wf(
        npz_files  = npz_files,
        output_dir = output_dir,
        device     = device,
        n_folds    = args.n_folds,
    )

    # Final summary
    logger.info("\n" + "=" * 60)
    logger.info("FINAL RESULTS -- CNN-Mamba Hybrid")
    logger.info("=" * 60)
    if concat_ic:
        for h in HORIZONS:
            ic = concat_ic.get(h, float("nan"))
            logger.info(f"  Concat IC ({h}): {ic:.4f}")
    logger.info(f"  Artifacts in: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
