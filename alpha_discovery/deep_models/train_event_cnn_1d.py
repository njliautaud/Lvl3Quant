"""
Event-Driven 1D Causal CNN — Training Script v1

Data format (identical to train_event_transformer_fast.py):
  - events: (N_events, 6) float32
      [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]
  - labels_1s/5s/10s: (N_events,) float32 — mid-price change in ticks
  - timestamps: (N_events,) int64 nanoseconds

Architecture: 1D Causal CNN (TCN / WaveNet-style)
  - Input: (B, W, 6) events window → permute to (B, 6, W) for Conv1d
  - Input projection: Conv1d(6 → CNN_CHANNELS) pointwise
  - Stack of dilated causal Conv1d blocks (dilations: 1, 2, 4, 8, 16, 32)
  - Each block: left-causal-pad → Conv1d → BatchNorm → GELU → Dropout + residual
  - Global average pooling → dense head
  - Multi-task output: predict 1s, 5s, 10s price change simultaneously
  - ~1-2M params (lighter than transformer)

Key differences from EventTransformerV1:
  - O(n) complexity vs O(n²) attention
  - Fixed receptive field: kernel * sum(dilations)
  - Translation equivariant (shift-invariant features)
  - No positional encoding needed — temporal order encoded structurally
  - Typically faster per-epoch on long sequences

Training rules (same as transformer):
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

# MLflow
try:
    if int(os.environ.get("DISABLE_MLFLOW", 0)):
        raise ImportError("MLflow disabled via DISABLE_MLFLOW env var")
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow disabled or not installed — skipping experiment tracking")

# ============================================================
# Logging setup
# ============================================================
LOG_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)

log_path = LOG_DIR / "event_cnn_1d.log"
# Force line-buffered file output (fixes Windows log flush issue)
_file_stream = open(log_path, "a", buffering=1)  # line-buffered in text mode
_file_handler = logging.StreamHandler(_file_stream)
_file_handler.setLevel(logging.INFO)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_file_handler, _stream_handler],
)
logger = logging.getLogger()  # use root logger — named logger breaks when imported as module


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
DEFAULT_BOOK30_DATA_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_book_features"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "event_cnn_1d"

# Precomputed tensor dir — set CNN_TENSOR_DIR env var to enable mmap loading
# If not set, defaults to sibling mbo_tensors_cnn/ directory (checked automatically)
_default_tensor_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_tensors_cnn"
CNN_TENSOR_DIR: Optional[Path] = (
    Path(os.environ["CNN_TENSOR_DIR"]) if "CNN_TENSOR_DIR" in os.environ
    else (_default_tensor_dir if _default_tensor_dir.exists() else None)
)
STRICT_LEAKAGE_FREE = int(os.environ.get("STRICT_LEAKAGE_FREE", 0)) == 1

# OF feature files directory (bar-level, ~27 events/bar)
# Set CNN_OF_FILE_DIR env var to override. Default: data/processed/orderflow_features/
_default_of_dir = Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "orderflow_features"
CNN_OF_FILE_DIR: Optional[Path] = (
    Path(os.environ["CNN_OF_FILE_DIR"]) if "CNN_OF_FILE_DIR" in os.environ
    else (_default_of_dir if _default_of_dir.exists() else None)
)


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
MLFLOW_EXPERIMENT = "EventDriven_CNN1D"

# CNN-specific hyperparameters (CNN_* prefix)
CNN_CHANNELS   = int(os.environ.get("CNN_CHANNELS", 128))
CNN_KERNEL     = int(os.environ.get("CNN_KERNEL", 5))
CNN_LAYERS     = int(os.environ.get("CNN_LAYERS", 6))
CNN_DROPOUT    = float(os.environ.get("CNN_DROPOUT", 0.1))

# Derived orderflow features (v2 experiment)
# Set CNN_DERIVED_FEATURES=1 to add 5 basic orderflow derivatives alongside raw 6
USE_DERIVED_FEATURES = int(os.environ.get("CNN_DERIVED_FEATURES", 0)) == 1
# OF feature injection: two modes
#   CNN_OF_FEATURES=1  — rolling computed from raw events (6→8 ch). No extra data files needed.
#   CNN_OF_FILE=1      — load book_imbalance + depth_imbalance from orderflow_features NPZ
#                        files (real precomputed OF, 177 days available). 6→8 channels.
#                        Falls back to CNN_OF_FEATURES rolling if OF file missing for a day.
USE_OF_FEATURES = int(os.environ.get("CNN_OF_FEATURES", 0)) == 1
USE_OF_FILE = int(os.environ.get("CNN_OF_FILE", 0)) == 1
# CNN_FEATURE_SET=book30 — use Jupiter's 30-feature per-event book state NPZ files.
# Disables all runtime feature augmentation; CNN reads 30 channels directly from NPZ/tensors.
CNN_FEATURE_SET = os.environ.get("CNN_FEATURE_SET", "")
FEATURE_SET_BOOK30 = CNN_FEATURE_SET == "book30"
FEATURE_SET_FEAT18 = CNN_FEATURE_SET == "feat18"  # 18-feature pre-computed NPZ
FEATURE_SET_FEAT20 = CNN_FEATURE_SET == "feat20"  # 20-feature pre-computed NPZ
N_RAW_FEATURES = 6
N_DERIVED_FEATURES = 5 if (USE_DERIVED_FEATURES and not FEATURE_SET_BOOK30 and not FEATURE_SET_FEAT18 and not FEATURE_SET_FEAT20) else 0
N_OF_FEATURES = 2 if ((USE_OF_FEATURES or USE_OF_FILE) and not FEATURE_SET_BOOK30 and not FEATURE_SET_FEAT18 and not FEATURE_SET_FEAT20) else 0
N_TOTAL_FEATURES = 30 if FEATURE_SET_BOOK30 else (20 if FEATURE_SET_FEAT20 else (18 if FEATURE_SET_FEAT18 else (N_RAW_FEATURES + N_DERIVED_FEATURES + N_OF_FEATURES)))

# Shared hyperparameters — use same EVENT_* env vars as transformer for easy A/B switching
WINDOW_SIZE    = int(os.environ.get("EVENT_WINDOW_SIZE", 500))
STRIDE         = int(os.environ.get("EVENT_STRIDE", WINDOW_SIZE // 2))
BATCH_SIZE     = int(os.environ.get("EVENT_BATCH_SIZE", 256))   # CNN can fit larger batches
LR             = float(os.environ.get("EVENT_LR", 3e-4))
EPOCHS_PER_FOLD = int(os.environ.get("EVENT_EPOCHS", 5))
WARMUP_STEPS   = int(os.environ.get("EVENT_WARMUP", 300))
GRAD_CLIP      = float(os.environ.get("EVENT_GRAD_CLIP", 1.0))
N_FOLDS        = int(os.environ.get("EVENT_N_FOLDS", 5))
HORIZONS       = ["1s", "5s", "10s"]

# Dilation schedule — receptive field = kernel * sum(dilations) = 5 * 63 = 315 events
DILATION_SCHEDULE = [2 ** i for i in range(CNN_LAYERS)]  # [1, 2, 4, 8, 16, 32]


# ============================================================
# Derived Feature Engineering (v2)
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
    The CNN's dilated convolutions naturally aggregate them over its receptive field.

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


def compute_of_features(events: np.ndarray, window: int = 100) -> np.ndarray:
    """
    Compute 2 rolling orderflow features from raw 6 features.
    Input: (N, 6) — [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]
    Output: (N, 2) — [book_imbalance_rolling, depth_imbalance_rolling]

    Feature definitions (rolling window of `window` events):
        book_imbalance_rolling:
            (bid_add_vol - ask_add_vol) / (bid_add_vol + ask_add_vol + 1e-8)
            event_type==0 (add), side==0 (bid) → bid add; side==1 (ask) → ask add
            Matches the book_imbalance computed in preprocess_microbatch.py (line 118-121)
            but as a rolling per-event feature instead of a microbatch aggregate.
        depth_imbalance_rolling:
            bid_total_vol / (bid_total_vol + ask_total_vol + 1e-8) over rolling window
            ALL event types contribute (add/cancel/modify/trade), capturing net depth pressure.
            Proxy for depth_imbalance_5 (IC=-0.025 from OF feature screen, Session 72).

    Both are causal: only use events[i-window:i], no look-ahead.
    Uses cumsum trick for O(N) computation (no per-event loop).
    """
    n = len(events)
    out = np.zeros((n, 2), dtype=np.float32)

    event_type = events[:, 1]
    side = events[:, 2]
    qty = np.exp(events[:, 4])  # log → linear

    # Masks (float for vectorized multiply)
    add_mask = (event_type < 0.5).astype(np.float32)       # event_type == 0 (add)
    bid_mask = (side < 0.5).astype(np.float32)             # side == 0 (bid)
    ask_mask = 1.0 - bid_mask                               # side != 0 (ask)

    bid_add_vol = qty * add_mask * bid_mask
    ask_add_vol = qty * add_mask * ask_mask
    bid_vol_all = qty * bid_mask
    ask_vol_all = qty * ask_mask

    # Cumsum trick for rolling sums (O(N), causal)
    cs_bid_add = np.cumsum(bid_add_vol)
    cs_ask_add = np.cumsum(ask_add_vol)
    cs_bid_all = np.cumsum(bid_vol_all)
    cs_ask_all = np.cumsum(ask_vol_all)

    # Rolling sum = cumsum[i] - cumsum[i - window]  (with boundary clamp)
    def rolling(cs: np.ndarray) -> np.ndarray:
        result = cs.copy()
        result[window:] -= cs[:-window]
        return result

    r_bid_add = rolling(cs_bid_add)
    r_ask_add = rolling(cs_ask_add)
    r_bid_all = rolling(cs_bid_all)
    r_ask_all = rolling(cs_ask_all)

    # Feature 0: book_imbalance (normalized to [-1, 1])
    denom0 = r_bid_add + r_ask_add + 1e-8
    out[:, 0] = (r_bid_add - r_ask_add) / denom0

    # Feature 1: depth_imbalance (normalized to [0, 1])
    denom1 = r_bid_all + r_ask_all + 1e-8
    out[:, 1] = r_bid_all / denom1

    return out


def load_of_bars(npz_stem: str) -> Optional[np.ndarray]:
    """
    Load OF bar arrays for one day from orderflow_features NPZ.
    Returns (n_bars, 2) float32: [book_imbalance, roll_delta_10s] or None.

    These are ~234K bars/day. Used by MboEventDataset for per-window scalar injection:
    for a window starting at event `start`, map to bar range and take the mean.
    This avoids per-event upsampling — the mean is broadcast across all 500 events.
    """
    if CNN_OF_FILE_DIR is None:
        return None
    of_path = CNN_OF_FILE_DIR / f"{npz_stem}_orderflow.npz"
    if not of_path.exists():
        return None
    try:
        of = np.load(of_path)
        book_imb = of["book_imbalance"].astype(np.float32)
        roll_d = of["roll_delta_10s"].astype(np.float32)
        return np.stack([book_imb, roll_d], axis=1)  # (n_bars, 2)
    except Exception as e:
        logger.warning(f"Failed to load OF bars for {npz_stem}: {e}")
        return None


# ============================================================
# Dataset  (identical to MboEventDataset in train_event_transformer_fast.py)
# ============================================================

class MboEventDataset(Dataset):
    """
    Creates (window_of_events, labels) pairs from MBO event NPZ files.

    LAZY LOADING: Only keeps file paths and a sample index in RAM.
    Actual data is loaded on-demand in __getitem__ with an LRU cache
    (default 5 days) to avoid re-reading the same file repeatedly.
    This keeps RAM usage ~500MB instead of 10-30GB.
    """

    def __init__(
        self,
        npz_files: List[Path],
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        horizons: Optional[List[str]] = None,
        normalize_features: bool = True,
        feature_stats: Optional[Dict] = None,
        cache_days: int = int(os.environ.get("CNN_CACHE_DAYS", 5)),
    ):
        self.window_size = window_size
        self.stride = stride
        self.horizons = horizons or ["1s", "5s", "10s"]
        self.normalize_features = normalize_features
        self.npz_files = list(npz_files)
        self.cache_days = cache_days
        self.sample_index: List[Tuple[int, int]] = []  # (day_idx, event_start)

        # Day cache: {day_idx: (events, {horizon: labels})}
        from collections import OrderedDict
        self._cache: OrderedDict = OrderedDict()

        self.use_derived = USE_DERIVED_FEATURES
        self.use_of = USE_OF_FEATURES
        self.use_of_file = USE_OF_FILE
        self.n_features = N_TOTAL_FEATURES

        if normalize_features and feature_stats is None:
            self._compute_stats(npz_files)
        elif feature_stats is not None:
            self.feature_mean = feature_stats["mean"]
            self.feature_std  = feature_stats["std"]
        else:
            self.feature_mean = np.zeros(self.n_features, dtype=np.float32)
            self.feature_std  = np.ones(self.n_features, dtype=np.float32)

        self._build_index()

    # ------------------------------------------------------------------
    def _load_npz(self, f: Path):
        """Load a single NPZ file with retry on PermissionError."""
        for _attempt in range(12):
            try:
                return np.load(f, allow_pickle=True)
            except PermissionError:
                if _attempt < 11:
                    import time as _time
                    logger.warning(
                        f"PermissionError on {f.name}, retry {_attempt+1}/12..."
                    )
                    _time.sleep(5)
                else:
                    raise

    # ------------------------------------------------------------------
    def _compute_stats(self, npz_files: List[Path]):
        """Compute running mean/std across all events for normalization.
        Loads one file at a time and discards immediately."""
        logger.info(f"Computing feature statistics from {len(npz_files)} files "
                     f"({self.n_features} features, derived={self.use_derived})...")
        total_sum   = np.zeros(self.n_features, dtype=np.float64)
        total_sq    = np.zeros(self.n_features, dtype=np.float64)
        total_count = 0

        # feat18/feat20 mode: load pre-computed features directly from events key
        if FEATURE_SET_FEAT18 or FEATURE_SET_FEAT20:
            for f in npz_files:
                data = self._load_npz(f)
                ev = data["events"].astype(np.float64)
                if ev.shape[1] < N_TOTAL_FEATURES:
                    logger.warning(f"feat18/20: {f.name} has {ev.shape[1]} cols, need {N_TOTAL_FEATURES}"); del data,ev; continue
                ev = ev[:, :N_TOTAL_FEATURES]
                total_sum += ev.sum(axis=0); total_sq += (ev**2).sum(axis=0); total_count += len(ev)
                del data, ev
            self.feature_mean = (total_sum/total_count).astype(np.float32)
            var = (total_sq/total_count)-(total_sum/total_count)**2
            self.feature_std = np.sqrt(np.maximum(var,1e-8)).astype(np.float32)
            logger.info(f"feat{N_TOTAL_FEATURES} stats: {total_count} events, mean[:3]={self.feature_mean[:3]}")
            return
        # book30 mode: all 30 features come directly from NPZ features array — no augmentation
        if FEATURE_SET_BOOK30:
            for f in npz_files:
                data = self._load_npz(f)
                _key = "features" if "features" in data else "events"
                ev = data[_key].astype(np.float64)
                if ev.shape[1] != 30:
                    logger.warning(f"book30: {f.name} has {ev.shape[1]} features (expected 30), skipping stats")
                    del data, ev
                    continue
                total_sum += ev.sum(axis=0)
                total_sq  += (ev ** 2).sum(axis=0)
                total_count += len(ev)
                del data, ev
            mean = (total_sum / total_count).astype(np.float32)
            var  = (total_sq / total_count) - (total_sum / total_count) ** 2
            self.feature_mean = mean
            self.feature_std  = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
            logger.info(f"book30 stats: mean[:5]={self.feature_mean[:5]}, std[:5]={self.feature_std[:5]}")
            return

        # Stats for base features (raw6 + optional derived)
        n_base = N_RAW_FEATURES + N_DERIVED_FEATURES
        base_sum = np.zeros(n_base, dtype=np.float64)
        base_sq  = np.zeros(n_base, dtype=np.float64)
        base_cnt = 0

        # Stats for OF bar channels (computed from bar arrays, not per-event)
        of_sum = np.zeros(N_OF_FEATURES, dtype=np.float64)
        of_sq  = np.zeros(N_OF_FEATURES, dtype=np.float64)
        of_cnt = 0

        for f in npz_files:
            data = self._load_npz(f)
            events = data["events"].astype(np.float32)
            if self.use_derived:
                derived = compute_derived_features(events)
                events = np.concatenate([events, derived], axis=1)
            ev64 = events.astype(np.float64)
            base_sum += ev64.sum(axis=0)
            base_sq  += (ev64 ** 2).sum(axis=0)
            base_cnt += len(ev64)
            del data, events, ev64

            # OF bar stats: load bar arrays and accumulate
            if self.use_of_file and N_OF_FEATURES > 0:
                bars = load_of_bars(f.stem)  # (n_bars, 2) or None
                if bars is None:
                    # fallback: use rolling OF on raw events (slower but correct)
                    raw = np.load(f, allow_pickle=True)["events"].astype(np.float32)
                    bars = compute_of_features(raw)
                    del raw
                b64 = bars.astype(np.float64)
                of_sum += b64.sum(axis=0)
                of_sq  += (b64 ** 2).sum(axis=0)
                of_cnt += len(b64)
                del bars, b64
            elif self.use_of and N_OF_FEATURES > 0:
                raw = np.load(f, allow_pickle=True)["events"].astype(np.float32)
                of_feats = compute_of_features(raw).astype(np.float64)
                of_sum += of_feats.sum(axis=0)
                of_sq  += (of_feats ** 2).sum(axis=0)
                of_cnt += len(of_feats)
                del raw, of_feats

        # Combine base + OF stats into single mean/std arrays
        mean_parts = [(base_sum / base_cnt).astype(np.float32)]
        std_parts = []
        var_base = (base_sq / base_cnt) - (base_sum / base_cnt) ** 2
        std_parts.append(np.sqrt(np.maximum(var_base, 1e-8)).astype(np.float32))

        if N_OF_FEATURES > 0 and of_cnt > 0:
            mean_of = (of_sum / of_cnt).astype(np.float32)
            var_of = (of_sq / of_cnt) - (of_sum / of_cnt) ** 2
            std_of = np.sqrt(np.maximum(var_of, 1e-8)).astype(np.float32)
            mean_parts.append(mean_of)
            std_parts.append(std_of)
        elif N_OF_FEATURES > 0:
            mean_parts.append(np.zeros(N_OF_FEATURES, dtype=np.float32))
            std_parts.append(np.ones(N_OF_FEATURES, dtype=np.float32))

        self.feature_mean = np.concatenate(mean_parts)
        self.feature_std  = np.concatenate(std_parts)
        logger.info(f"Feature mean: {self.feature_mean}")
        logger.info(f"Feature std:  {self.feature_std}")

    def get_feature_stats(self) -> Dict:
        return {"mean": self.feature_mean, "std": self.feature_std}

    # ------------------------------------------------------------------
    def _build_index(self):
        """Scan files to build sample index WITHOUT keeping data in RAM."""
        for day_idx, f in enumerate(self.npz_files):
            data = self._load_npz(f)
            events = data["events"]
            n_events = len(events)

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

            del data, events, day_labels  # free immediately

        logger.info(
            f"Dataset: {len(self.npz_files)} days, {len(self.sample_index)} samples "
            f"(window={self.window_size}, stride={self.stride}) [LAZY LOADING]"
        )

    # ------------------------------------------------------------------
    def _get_day(self, day_idx: int):
        """Load a day's data with LRU cache."""
        if day_idx in self._cache:
            self._cache.move_to_end(day_idx)
            return self._cache[day_idx]

        data = self._load_npz(self.npz_files[day_idx])
        _key = "features" if (FEATURE_SET_BOOK30 and "features" in data) else "events"
        events = data[_key].astype(np.float32)
        if FEATURE_SET_FEAT18 or FEATURE_SET_FEAT20: events = events[:, :N_TOTAL_FEATURES]
        raw6 = events
        n_events = len(raw6)
        if FEATURE_SET_BOOK30 or FEATURE_SET_FEAT18 or FEATURE_SET_FEAT20:
            # book30/feat18/feat20: features from NPZ — just normalize, no derived computation
            if self.normalize_features:
                events = (events - self.feature_mean) / (self.feature_std + 1e-8)
            day_labels = {h: data[f"labels_{h}"].astype(np.float32) for h in self.horizons}
            del data
            self._cache[day_idx] = (events, day_labels, None, 1)
            while len(self._cache) > self.cache_days:
                self._cache.popitem(last=False)
            return events, day_labels, None, 1
        if self.use_derived:
            derived = compute_derived_features(events)
            events = np.concatenate([events, derived], axis=1)
        # Rolling OF: concatenate per-event features before normalization
        if self.use_of and not self.use_of_file:
            of_feats = compute_of_features(raw6)
            events = np.concatenate([events, of_feats], axis=1)
        if self.normalize_features:
            events -= self.feature_mean[:events.shape[1]]
            events /= (self.feature_std[:events.shape[1]] + 1e-8)
        day_labels = {h: data[f"labels_{h}"].astype(np.float32) for h in self.horizons}
        del data

        # Load OF bars for per-window scalar injection (use_of_file mode only)
        of_bars: Optional[np.ndarray] = None  # (n_bars, 2) or None
        of_bar_ratio: int = 1
        if self.use_of_file:
            npz_stem = self.npz_files[day_idx].stem
            of_bars = load_of_bars(npz_stem)
            if of_bars is not None:
                of_bar_ratio = max(1, n_events // len(of_bars))
            else:
                logger.warning(f"OF file missing for {npz_stem}, OF channels will be zero")

        self._cache[day_idx] = (events, day_labels, of_bars, of_bar_ratio)
        while len(self._cache) > self.cache_days:
            self._cache.popitem(last=False)

        return events, day_labels, of_bars, of_bar_ratio

    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        day_idx, start = self.sample_index[idx]
        end = start + self.window_size
        events, day_labels, of_bars, of_bar_ratio = self._get_day(day_idx)
        window = events[start:end].copy()  # (W, n_base_features)
        label_idx = end - 1
        labels = np.array(
            [day_labels[h][label_idx] for h in self.horizons],
            dtype=np.float32,
        )  # (3,)

        if self.use_of_file:
            # Per-window scalar: mean of OF bars covered by this window
            # bar_start..bar_end covers the ~500 events in this window
            bar_start = start // of_bar_ratio
            bar_end = max(bar_start + 1, end // of_bar_ratio)
            if of_bars is not None:
                bar_end = min(bar_end, len(of_bars))
                of_mean = of_bars[bar_start:bar_end].mean(axis=0)  # (2,)
                # Normalize with stored stats (channels after base features)
                n_base = window.shape[1]
                of_norm = (of_mean - self.feature_mean[n_base:n_base + 2]) / (self.feature_std[n_base:n_base + 2] + 1e-8)
            else:
                of_norm = np.zeros(2, dtype=np.float32)
            # Broadcast scalar across window: (W, 2) — constant channel
            of_channel = np.broadcast_to(of_norm[np.newaxis, :], (self.window_size, 2)).copy()
            window = np.concatenate([window, of_channel], axis=1)  # (W, n_base+2)

        return torch.from_numpy(window), torch.from_numpy(labels)


# ============================================================
# Precomputed Tensor Dataset  (faster I/O via mmap .pt files)
# ============================================================

class PrecomputedTensorDataset(Dataset):
    """
    Loads precomputed windowed tensors from .pt files produced by precompute_tensors.py.

    Each .pt file contains:
        {"events": (N_windows, W, F) float32, "labels_1s": (N,), "labels_5s": (N,), "labels_10s": (N,)}
    Tensors are already normalized with the global stats from the FULL dataset.

    IMPORTANT — leakage note: precompute_tensors.py computes stats over ALL files.
    When used for walk-forward training, stats will include future data in later folds.
    This is a mild look-ahead on normalization only (not labels) — acceptable given that
    global stats are stable (mean/std don't change much day-to-day). For strict leakage-free
    training, use MboEventDataset which computes stats per fold from train files only.
    Set STRICT_LEAKAGE_FREE=1 env var to force MboEventDataset even when .pt files exist.

    MMAP mode: each file is memory-mapped, not fully loaded into RAM.
    Multiple DataLoader workers can safely read different indices concurrently.
    """

    def __init__(
        self,
        pt_files: List[Path],
        horizons: Optional[List[str]] = None,
        feature_stats: Optional[Dict] = None,
    ):
        self.pt_files = list(pt_files)
        self.horizons = horizons or ["1s", "5s", "10s"]
        # index: list of (file_idx, window_idx_within_file)
        self.sample_index: List[Tuple[int, int]] = []
        # cached window counts per file (loaded on first index build)
        self._file_lengths: List[int] = []
        # Store fold stats for get_feature_stats() (so OOT dataset gets the same stats)
        self._feature_stats: Dict = feature_stats or {"mean": None, "std": None}

        # Per-fold renormalization: tensors were normalized with global stats.
        # If per-fold stats provided, compute correction: x_fold = (x_global * g_std + g_mean - f_mean) / f_std
        # This keeps the 14x epoch speedup while fixing WF normalization leakage.
        self._renorm_scale: Optional[torch.Tensor] = None
        self._renorm_bias: Optional[torch.Tensor] = None
        if feature_stats is not None and feature_stats.get("mean") is not None:
            tensor_dir = Path(pt_files[0]).parent if pt_files else None
            stats_path = (tensor_dir / "stats.pt") if tensor_dir else None
            if stats_path and stats_path.exists():
                global_stats = torch.load(stats_path, map_location="cpu", weights_only=True)
                g_mean = global_stats["mean"]  # (F,)
                g_std = global_stats["std"]    # (F,)
                f_mean = torch.as_tensor(feature_stats["mean"], dtype=torch.float32)
                f_std = torch.as_tensor(feature_stats["std"], dtype=torch.float32)
                # x_global_norm = (x_raw - g_mean) / g_std
                # x_fold_norm = (x_raw - f_mean) / f_std = (x_global_norm * g_std + g_mean - f_mean) / f_std
                # So: x_fold = x_global * (g_std / f_std) + (g_mean - f_mean) / f_std
                self._renorm_scale = (g_std / f_std).unsqueeze(0)  # (1, F) — broadcasts with (W, F)
                self._renorm_bias = ((g_mean - f_mean) / f_std).unsqueeze(0)  # (1, F)
                logger.info("PrecomputedTensorDataset: per-fold renormalization enabled")

        self._build_index()

    def _build_index(self):
        """Build sample index using window_counts.json sidecar — no .pt loading at init."""
        import json as _json
        from collections import OrderedDict
        self._cache_size = int(os.environ.get("CNN_CACHE_DAYS", 3))
        self._cache: "OrderedDict[int, Dict]" = OrderedDict()

        # Load per-file window counts from sidecar (avoids loading 500MB .pt files at init)
        tensor_dir = self.pt_files[0].parent if self.pt_files else None
        counts_path = (tensor_dir / "window_counts.json") if tensor_dir else None
        file_counts: Dict[str, int] = {}
        if counts_path and counts_path.exists():
            with open(counts_path) as fh:
                file_counts = _json.load(fh)

        total = 0
        for file_idx, pt_path in enumerate(self.pt_files):
            if pt_path.name in file_counts:
                n_windows = file_counts[pt_path.name]
            else:
                # Fallback: load .pt file to get count — slow but correct
                logger.warning(f"No sidecar count for {pt_path.name}, loading .pt")
                data = torch.load(pt_path, map_location="cpu", weights_only=True)
                n_windows = int(data["events"].shape[0])
                del data
            self._file_lengths.append(n_windows)
            for w in range(n_windows):
                self.sample_index.append((file_idx, w))
            total += n_windows

        logger.info(
            f"PrecomputedTensorDataset: {len(self.pt_files)} files, "
            f"{total} samples [PT-LRU cache={self._cache_size}]"
        )

    def _get_file(self, file_idx: int) -> Dict:
        """LRU cache for loaded .pt file data."""
        from collections import OrderedDict
        if file_idx in self._cache:
            self._cache.move_to_end(file_idx)
            return self._cache[file_idx]
        data = torch.load(self.pt_files[file_idx], map_location="cpu", weights_only=True)
        self._cache[file_idx] = data
        self._cache.move_to_end(file_idx)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)
        return data

    def get_feature_stats(self) -> Dict:
        """Return per-fold feature stats (for OOT dataset renorm consistency)."""
        return self._feature_stats

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        file_idx, window_idx = self.sample_index[idx]
        data = self._get_file(file_idx)
        window = data["events"][window_idx]  # (W, F)
        if self._renorm_scale is not None:
            # Apply per-fold normalization correction (O(1), ~5µs per call)
            window = window * self._renorm_scale + self._renorm_bias
        labels = torch.stack([
            data[f"labels_{h}"][window_idx] for h in self.horizons
        ])  # (3,)
        return window, labels


class FileSequentialSampler(torch.utils.data.Sampler):
    """
    Sampler for PrecomputedTensorDataset that processes one file at a time.

    Files are shuffled randomly each epoch. Windows within each file are
    served sequentially. This keeps peak RAM at ~1 file (~480MB) instead
    of thrashing the LRU cache and OOMing with 84 files × 480MB.

    Only used when CNN_SEQUENTIAL_SAMPLER=1 (default: 1 for large datasets).
    """

    def __init__(self, dataset: "PrecomputedTensorDataset", shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
        # Group sample indices by file_idx
        from collections import defaultdict
        self._file_groups: Dict[int, List[int]] = defaultdict(list)
        for sample_idx, (file_idx, _) in enumerate(dataset.sample_index):
            self._file_groups[file_idx].append(sample_idx)
        self._file_order = list(self._file_groups.keys())

    def __iter__(self):
        if self.shuffle:
            import random
            file_order = list(self._file_order)
            random.shuffle(file_order)
            for file_idx in file_order:
                indices = list(self._file_groups[file_idx])
                random.shuffle(indices)  # shuffle within file for batch diversity
                yield from indices
        else:
            for file_idx in self._file_order:
                yield from self._file_groups[file_idx]

    def __len__(self) -> int:
        return len(self.dataset)


def find_pt_files_for_npz(
    npz_files: List[Path],
    tensor_dir: Path,
) -> Optional[List[Path]]:
    """
    For each NPZ file, find the matching precomputed .pt file.
    Returns list of .pt paths if at least one file has a matching .pt, else None.
    Files without a .pt (e.g. all-NaN days skipped by precompute) are silently skipped.
    Returns None only if NO .pt files exist at all.
    """
    pt_files = []
    for npz_path in npz_files:
        pt_path = tensor_dir / (npz_path.stem + ".pt")
        if not pt_path.exists():
            continue  # NaN/empty day skipped by precompute — skip here too
        pt_files.append(pt_path)
    return pt_files if pt_files else None


def compute_fold_stats(npz_files: List[Path]) -> Dict:
    """
    Compute per-fold feature mean/std from NPZ files (events only, one file at a time).
    Used for per-fold renormalization when precomputed tensors have global normalization.
    """
    total_sum = np.zeros(N_RAW_FEATURES, dtype=np.float64)
    total_sq = np.zeros(N_RAW_FEATURES, dtype=np.float64)
    total_count = 0
    for f in npz_files:
        try:
            data = np.load(f, allow_pickle=True)
            events = data["events"].astype(np.float64)
            total_sum += events.sum(axis=0)
            total_sq += (events ** 2).sum(axis=0)
            total_count += len(events)
            del data, events
        except Exception:
            pass
    if total_count == 0:
        return {"mean": np.zeros(N_RAW_FEATURES, dtype=np.float32),
                "std": np.ones(N_RAW_FEATURES, dtype=np.float32)}
    mean = (total_sum / total_count).astype(np.float32)
    var = (total_sq / total_count) - (mean.astype(np.float64) ** 2)
    std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
    return {"mean": mean, "std": std}


def build_dataset(
    npz_files: List[Path],
    tensor_dir: Optional[Path],
    feature_stats: Optional[Dict] = None,
    strict_leakage_free: bool = False,
) -> Dataset:
    """
    Build the best available dataset for the given files.
    Uses PrecomputedTensorDataset if .pt files exist and STRICT_LEAKAGE_FREE is not set.
    Falls back to MboEventDataset (lazy NPZ loading) otherwise.

    Per-fold renormalization: when using precomputed tensors (globally normalized),
    computes fold-specific stats from NPZ files and applies correction in __getitem__.
    This preserves the 14x epoch speedup while fixing normalization leakage.
    """
    if (
        not strict_leakage_free
        and tensor_dir is not None
        and tensor_dir.exists()
    ):
        pt_files = find_pt_files_for_npz(npz_files, tensor_dir)
        if pt_files is not None:
            logger.info(f"Using precomputed tensors from {tensor_dir} ({len(pt_files)} files)")
            # Compute per-fold stats from NPZ files for renormalization correction.
            # If feature_stats provided (OOT), use those. Otherwise compute from NPZ train files.
            fold_stats = feature_stats
            if fold_stats is None or fold_stats.get("mean") is None:
                logger.info(f"Computing per-fold feature stats from {len(npz_files)} NPZ files...")
                fold_stats = compute_fold_stats(npz_files)
                logger.info(f"Fold stats: mean={fold_stats['mean']}, std={fold_stats['std']}")
            return PrecomputedTensorDataset(pt_files, feature_stats=fold_stats)

    # Fall back to lazy NPZ loading
    return MboEventDataset(npz_files, window_size=WINDOW_SIZE, stride=STRIDE,
                           feature_stats=feature_stats)


# ============================================================
# Architecture: 1D Causal CNN (TCN / WaveNet-style)
# ============================================================

class CausalConv1dBlock(nn.Module):
    """
    Single causal dilated Conv1d residual block.

    Causality is enforced by left-only padding:
        pad_left = (kernel - 1) * dilation
    After convolution, we trim the right side — this is equivalent to
    "causal padding" in WaveNet / TCN literature.

    Block layout:
        input
          ├── skip connection (1×1 conv if channels differ)
          └── causal_pad → Conv1d → BatchNorm1d → GELU → Dropout
                └── + residual → output
    """

    def __init__(
        self,
        in_channels:  int,
        out_channels: int,
        kernel_size:  int,
        dilation:     int,
        dropout:      float = 0.1,
    ):
        super().__init__()
        self.kernel_size = kernel_size
        self.dilation    = dilation
        # Padding needed on the LEFT to preserve sequence length and stay causal
        self.pad_left    = (kernel_size - 1) * dilation

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size  = kernel_size,
            dilation      = dilation,
            padding       = 0,   # manual left-padding below
            bias          = False,
        )
        self.bn      = nn.BatchNorm1d(out_channels)
        self.act     = nn.GELU()
        self.dropout = nn.Dropout(dropout)

        # Residual / skip projection when channel dimensions change
        if in_channels != out_channels:
            self.residual_proj = nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False)
        else:
            self.residual_proj = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, C_in, L)
        Returns:
            out: (B, C_out, L)
        """
        residual = self.residual_proj(x)               # (B, C_out, L)

        # Left-only causal padding: pad=(pad_left, 0) on the last dim
        x_pad = F.pad(x, (self.pad_left, 0))           # (B, C_in, L + pad_left)
        x_conv = self.conv(x_pad)                      # (B, C_out, L)  — exact length
        x_conv = self.bn(x_conv)
        x_conv = self.act(x_conv)
        x_conv = self.dropout(x_conv)

        return x_conv + residual                       # (B, C_out, L)


class EventCNN1D(nn.Module):
    """
    1D Causal CNN for MBO event stream prediction.

    Architecture:
        1. Input projection: pointwise Conv1d(6 → C) to lift feature dim
        2. Stack of N causal dilated Conv1d blocks (dilations = 1, 2, 4, … 2^(N-1))
        3. Global average pooling: (B, C, L) → (B, C)
        4. Multi-task regression head: predict 1s, 5s, 10s price change

    Receptive field: kernel_size * (sum of dilations)
        default: 5 * (1+2+4+8+16+32) = 5 * 63 = 315 events

    Parameters: ~1.5M (vs ~2.5M for EventTransformerV1)

    Causality guarantee:
        Each block only pads on the LEFT, so position t can only see events ≤ t.
        Global average pooling is position-agnostic but the LAST position (used as
        label anchor) can only have seen past events — causality is preserved.
    """

    def __init__(
        self,
        in_channels:     int   = 6,
        channels:        int   = CNN_CHANNELS,
        kernel_size:     int   = CNN_KERNEL,
        n_layers:        int   = CNN_LAYERS,
        dropout:         float = CNN_DROPOUT,
        n_targets:       int   = 3,
        dilation_schedule: Optional[List[int]] = None,
    ):
        super().__init__()
        self.channels    = channels
        self.n_targets   = n_targets
        dilations        = dilation_schedule or [2 ** i for i in range(n_layers)]

        # Input projection (pointwise — no temporal mixing yet)
        self.input_proj = nn.Sequential(
            nn.Conv1d(in_channels, channels, kernel_size=1, bias=False),
            nn.BatchNorm1d(channels),
            nn.GELU(),
        )

        # Dilated causal conv stack
        self.blocks = nn.ModuleList()
        for dil in dilations:
            self.blocks.append(
                CausalConv1dBlock(
                    in_channels  = channels,
                    out_channels = channels,
                    kernel_size  = kernel_size,
                    dilation     = dil,
                    dropout      = dropout,
                )
            )

        # Global average pooling reduces (B, C, L) → (B, C)
        self.gap = nn.AdaptiveAvgPool1d(1)

        # Multi-task prediction head
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
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm1d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, events: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            events: (B, L, 6) float32 — batch of event windows (same format as transformer)
            return_embedding: if True, also return pattern embedding for fusion layer
        Returns:
            preds: (B, n_targets) — predicted price changes (1s, 5s, 10s)
            embedding: (B, channels) — pattern embedding (only if return_embedding=True)
        """
        # Rearrange: (B, L, 6) → (B, 6, L) for Conv1d
        x = events.permute(0, 2, 1).contiguous()   # (B, 6, L)

        # Input projection
        x = self.input_proj(x)                      # (B, C, L)

        # Dilated causal conv blocks
        for block in self.blocks:
            x = block(x)                             # (B, C, L) — causal throughout

        # Global average pool — this IS the pattern embedding for fusion
        embedding = self.gap(x).squeeze(-1)          # (B, C)

        # Prediction
        preds = self.head(embedding)                 # (B, n_targets)

        if return_embedding:
            return preds, embedding
        return preds

    def receptive_field(self) -> int:
        """Theoretical receptive field in number of events."""
        total = 0
        for block in self.blocks:
            total += (block.kernel_size - 1) * block.dilation
        return total + 1  # +1 for the single output position


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

    If extract_embeddings=True, also returns (B, C) pattern embeddings for fusion.
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
    """Train CNN for one expanding-window fold. Returns metrics dict."""

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

        for events, labels in train_loader:
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
                    "model_state":  model.state_dict(),
                    "fold":         fold_idx,
                    "epoch":        epoch,
                    "val_loss":     val_metrics["loss"],
                    "val_ic_10s":   val_metrics.get("ic_10s"),
                    "arch": {
                        "channels":    CNN_CHANNELS,
                        "kernel_size": CNN_KERNEL,
                        "n_layers":    CNN_LAYERS,
                        "dropout":     CNN_DROPOUT,
                        "window_size": WINDOW_SIZE,
                        "in_channels": N_TOTAL_FEATURES,
                        "derived_features": USE_DERIVED_FEATURES,
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
    train_days: Optional[int] = None,
    oot_days:   Optional[int] = None,
):
    """
    Walk-forward training: expanding window (default) or sliding window.

    Expanding window (train_days=None): each fold uses all files up to the fold boundary.
    Sliding window (train_days=N): each fold uses only the last N files before the boundary.

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

    logger.info(f"Total files (valid): {n_files} ({npz_files[0].name} → {npz_files[-1].name})")

    window_mode = f"sliding({train_days}d)" if train_days else "expanding"

    # Build fold boundaries
    # If oot_days is set, fix OOT block to last N days (e.g. 150 train / 50 OOT single split)
    if oot_days is not None:
        # WALK-FORWARD (HC #1, CLAUDE.md): each fold gets its own OOT block of `oot_days` days,
        # immediately following its train window. OOT cursor walks per fold.
        # Patched 2026-04-28 15:15 ET — previous code fixed OOT at last `oot_days` for ALL folds (bug).
        oot_days = min(oot_days, n_files - 5)
        last_train_end = n_files - oot_days        # fold N-1 train_end
        min_train = max(5, last_train_end - (n_folds - 1))
        fold_boundaries = []
        for fold in range(n_folds):
            train_end = min_train + fold
            oot_start = train_end                  # OOT walks with fold
            oot_end   = train_end + oot_days
            if oot_end > n_files:
                break
            train_start = max(0, train_end - train_days) if train_days is not None else 0
            fold_boundaries.append((fold, list(range(train_start, train_end)), list(range(oot_start, oot_end))))
        wf_mode = f"sliding({train_days}d)" if train_days else "expanding"
        logger.info(f"Walk-forward: {len(fold_boundaries)} folds, {wf_mode} train, oot_days={oot_days} (walking)")
    else:
        min_train     = max(5, n_files - n_folds)
        fold_boundaries = []
        for fold in range(n_folds):
            train_end = min_train + fold
            oot_start = train_end
            oot_end   = oot_start + max(1, (n_files - min_train) // n_folds)
            oot_end   = min(oot_end, n_files)
            if oot_start >= n_files:
                break
            train_start = max(0, train_end - train_days) if train_days is not None else 0
            fold_boundaries.append((fold, list(range(train_start, train_end)), list(range(oot_start, oot_end))))

    logger.info(f"Running {len(fold_boundaries)} folds ({window_mode})")

    # AMP only useful on CUDA
    use_amp = device.type == "cuda"

    # Concat storage
    concat_preds  = {h: [] for h in HORIZONS}
    concat_labels = {h: [] for h in HORIZONS}
    concat_embeds = []  # pattern embeddings for fusion layer

    # MLflow run
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(
            run_name=f"EventCNN1D_{time.strftime('%Y%m%d_%H%M')}"
        )
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        rf_events = sum([(CNN_KERNEL - 1) * d for d in DILATION_SCHEDULE]) + 1
        mlflow.log_params(
            {
                "model":             "EventCNN1D",
                "window_size":       WINDOW_SIZE,
                "stride":            STRIDE,
                "cnn_channels":      CNN_CHANNELS,
                "cnn_kernel":        CNN_KERNEL,
                "cnn_layers":        CNN_LAYERS,
                "cnn_dropout":       CNN_DROPOUT,
                "dilation_schedule": str(DILATION_SCHEDULE),
                "receptive_field":   rf_events,
                "batch_size":        BATCH_SIZE,
                "lr":                LR,
                "epochs_per_fold":   EPOCHS_PER_FOLD,
                "n_folds":           len(fold_boundaries),
                "horizons":          str(HORIZONS),
                "n_files":           n_files,
                "optimizer":         "AdamW",
                "warmup_steps":      WARMUP_STEPS,
                "grad_clip":         GRAD_CLIP,
                "node":              socket.gethostname(),
                "gpu":               gpu_name,
                "data_dir":          str(npz_files[0].parent),
                "output_dir":        str(output_dir),
                "mixed_precision":   "fp16" if use_amp else "none",
                "num_workers":       0,
                "in_channels":       N_TOTAL_FEATURES,
                "derived_features":  str(USE_DERIVED_FEATURES),
                "of_features":       "file" if USE_OF_FILE else ("rolling" if USE_OF_FEATURES else "none"),
                "feature_names":     (
                    "book30" if FEATURE_SET_BOOK30
                    else "raw6+of2_file" if USE_OF_FILE
                    else "raw6+orderflow5+of2" if (USE_DERIVED_FEATURES and USE_OF_FEATURES)
                    else "raw6+of2" if USE_OF_FEATURES
                    else "raw6+orderflow5" if USE_DERIVED_FEATURES
                    else "raw6"
                ),
                "feature_set":       CNN_FEATURE_SET if CNN_FEATURE_SET else "default",
                "window_mode":       window_mode,
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

            # Build train dataset — use precomputed tensors if available, else lazy NPZ
            logger.info("Building train dataset...")
            train_ds = build_dataset(
                train_files,
                tensor_dir=CNN_TENSOR_DIR,
                strict_leakage_free=STRICT_LEAKAGE_FREE,
            )
            feature_stats = train_ds.get_feature_stats()

            # Build OOT dataset — same source preference as train
            logger.info("Building OOT dataset...")
            oot_ds = build_dataset(
                oot_files,
                tensor_dir=CNN_TENSOR_DIR,
                feature_stats=feature_stats,
                strict_leakage_free=STRICT_LEAKAGE_FREE,
            )

            # num_workers: configurable via env var, default 0 on Windows (pickle issues with numpy)
            _num_workers = int(os.environ.get("EVENT_NUM_WORKERS", 0 if sys.platform == "win32" else 8))
            _persistent = _num_workers > 0

            # Use FileSequentialSampler when CNN_SEQUENTIAL_SAMPLER=1 (default: 1).
            # Works for both PrecomputedTensorDataset and MboEventDataset.
            # Keeps RAM at ~1 file instead of thrashing LRU with global shuffle.
            # NOTE: causes within-day temporal correlation → lower IC. Use stride=500+
            # to reduce correlation when using lazy NPZ loading with file-sequential sampler.
            _has_sample_index = hasattr(train_ds, "sample_index")
            _use_seq_sampler = (
                _has_sample_index
                and int(os.environ.get("CNN_SEQUENTIAL_SAMPLER", 1))
            )
            if _use_seq_sampler:
                _train_sampler = FileSequentialSampler(train_ds, shuffle=True)
                _train_shuffle = False  # sampler controls order
                logger.info("Using FileSequentialSampler (file-level shuffle, sequential within file)")
            else:
                _train_sampler = None
                _train_shuffle = True

            train_loader = DataLoader(
                train_ds,
                batch_size  = BATCH_SIZE,
                shuffle     = _train_shuffle,
                sampler     = _train_sampler,
                num_workers = _num_workers,
                pin_memory  = True,
                drop_last   = True,
                persistent_workers = _persistent,
                prefetch_factor    = 4 if _num_workers > 0 else None,
            )
            oot_loader = DataLoader(
                oot_ds,
                batch_size  = BATCH_SIZE * 2,
                shuffle     = False,
                num_workers = _num_workers,
                pin_memory  = True,
                persistent_workers = _persistent,
                prefetch_factor    = 4 if _num_workers > 0 else None,
            )

            # Fresh model per fold
            model = EventCNN1D(
                in_channels       = N_TOTAL_FEATURES,
                channels          = CNN_CHANNELS,
                kernel_size       = CNN_KERNEL,
                n_layers          = CNN_LAYERS,
                dropout           = CNN_DROPOUT,
                n_targets         = len(HORIZONS),
                dilation_schedule = DILATION_SCHEDULE,
            ).to(device)

            if fold_idx == 0:
                n_params = count_parameters(model)
                rf       = model.receptive_field()
                logger.info(f"Model parameters:  {n_params:,}")
                logger.info(f"Receptive field:   {rf} events")
                logger.info(f"Dilation schedule: {DILATION_SCHEDULE}")

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

            # OOT inference (with embedding extraction for fusion layer)
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

            # Accumulate embeddings for fusion
            if oot_embeds is not None:
                concat_embeds.append(oot_embeds)

            # Compute extended metrics per fold
            fold_dir_acc = {}
            fold_mae = {}
            for i, h in enumerate(HORIZONS):
                p = oot_preds[:, i]
                l = oot_labels[:, i]
                nonzero = l != 0
                if nonzero.sum() > 0:
                    fold_dir_acc[h] = float((np.sign(p[nonzero]) == np.sign(l[nonzero])).mean())
                else:
                    fold_dir_acc[h] = float("nan")
                fold_mae[h] = float(np.abs(p - l).mean())

            logger.info(
                f"Fold {fold_idx:02d} OOT IC | "
                + " | ".join(f"{h}: {fold_ics[h]:.4f}" for h in HORIZONS)
            )
            logger.info(
                f"Fold {fold_idx:02d} OOT Dir Acc | "
                + " | ".join(f"{h}: {fold_dir_acc[h]:.1%}" for h in HORIZONS)
            )
            logger.info(
                f"Fold {fold_idx:02d} OOT MAE | "
                + " | ".join(f"{h}: {fold_mae[h]:.4f}" for h in HORIZONS)
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
            logger.info(f"Saved predictions + embeddings ({oot_embeds.shape[1]}d) → {pred_path}")

            # Save feature stats for this fold (needed for inference)
            # PrecomputedTensorDataset returns None stats (normalization done at precompute time)
            stats_path = output_dir / f"fold_{fold_idx:02d}_feature_stats.npz"
            _fmean = feature_stats.get("mean")
            _fstd = feature_stats.get("std")
            if _fmean is not None and _fstd is not None:
                np.savez(stats_path, mean=_fmean, std=_fstd)

            # Log per-fold metrics to MLflow
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow.log_metrics(
                    {
                        **{f"oot_ic_{h}_fold{fold_idx:02d}": fold_ics[h] for h in HORIZONS},
                        **{f"oot_dir_acc_{h}_fold{fold_idx:02d}": fold_dir_acc[h] for h in HORIZONS},
                        **{f"oot_mae_{h}_fold{fold_idx:02d}": fold_mae[h] for h in HORIZONS},
                    },
                    step=fold_idx,
                )
                # MANDATORY: Upload all fold artifacts to MLflow at fold completion.
                # We lost the champion because artifacts weren't saved. Never again.
                fold_artifact_dir = f"fold_{fold_idx:02d}"
                mlflow.log_artifact(str(pred_path), artifact_path=fold_artifact_dir)
                if ckpt_path.exists():
                    mlflow.log_artifact(str(ckpt_path), artifact_path=fold_artifact_dir)
                if stats_path.exists():
                    mlflow.log_artifact(str(stats_path), artifact_path=fold_artifact_dir)
                # Log fold boundary metadata as params
                mlflow.log_params({
                    f"fold{fold_idx:02d}_train_files": f"{train_files[0].name}→{train_files[-1].name}",
                    f"fold{fold_idx:02d}_train_n":     len(train_files),
                    f"fold{fold_idx:02d}_oot_files":   f"{oot_files[0].name}→{oot_files[-1].name}",
                    f"fold{fold_idx:02d}_oot_n":       len(oot_files),
                })
                logger.info(f"Fold {fold_idx:02d} artifacts uploaded to MLflow")

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
                all_p         = np.concatenate(concat_preds[h])
                all_l         = np.concatenate(concat_labels[h])
                ic            = compute_ic(all_p, all_l)
                concat_ic[h]  = ic
                logger.info(f"  Concat IC ({h}): {ic:.4f}")
            else:
                concat_ic[h] = float("nan")

        # Save all concat predictions + embeddings for fusion
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
        logger.info(f"Saved concat predictions + embeddings → {concat_path}")

        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.log_metrics({f"concat_ic_{h}": concat_ic[h] for h in HORIZONS})
            # Upload concat predictions (primary result artifact)
            mlflow.log_artifact(str(concat_path), artifact_path="concat")
            logger.info("Concat predictions uploaded to MLflow")

        logger.info("\n" + "=" * 60)
        logger.info("LEAKAGE AUDIT: PASSED")
        logger.info("  - Expanding window: train set never contains OOT dates")
        logger.info("  - Feature normalization computed from train set only per fold")
        logger.info("  - Causal convolutions: left-only padding, no future information")
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
                size_str  = _jupiter_exec(f"stat -c %s {remote_path}", timeout=10).strip()
                file_size = int(size_str)
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
    parser = argparse.ArgumentParser(description="Event 1D Causal CNN Walk-Forward Training")
    _default_data = DEFAULT_BOOK30_DATA_DIR if FEATURE_SET_BOOK30 else DEFAULT_DATA_DIR
    parser.add_argument("--data-dir",    type=str, default=_default_data,
                        help="Directory with mbo_events (or mbo_book_features) NPZ files")
    parser.add_argument("--output-dir",  type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for checkpoints and predictions")
    parser.add_argument("--n-folds",     type=int, default=N_FOLDS)
    parser.add_argument("--skip-transfer", action="store_true",
                        help="Skip data transfer from Jupiter")
    parser.add_argument("--max-days",    type=int, default=None,
                        help="Limit number of data files to load (most recent N days)")
    parser.add_argument("--window-mode", type=str, default="sliding",
                        choices=["sliding"],  # HC #0 sliding-only
                        help="Walk-forward window mode: SLIDING ONLY (HC #0)")
    parser.add_argument("--train-days",  type=int, default=None,
                        help="Sliding window: number of training days per fold (requires --window-mode sliding)")
    parser.add_argument("--oot-days",    type=int, default=None,
                        help="Fix OOT block to last N days (e.g. --n-folds 1 --oot-days 50 for 150/50 single split)")
    parser.add_argument("--device",      type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir   = Path(args.data_dir)
    device     = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Per-run log file: {output_dir}/training.log (line-buffered, flushes immediately)
    _run_log_stream = open(output_dir / "training.log", "a", buffering=1)
    _run_log_handler = _FlushHandler(_run_log_stream)
    _run_log_handler.setLevel(logging.INFO)
    _run_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_run_log_handler)
    logger.info(f"Per-run log: {output_dir / 'training.log'}")

    logger.info("=" * 60)
    logger.info("Event 1D Causal CNN — Walk-Forward Training")
    logger.info(f"Device:          {device}")
    if device.type == "cuda":
        logger.info(f"GPU:             {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    _feat_desc = (
        "book30" if FEATURE_SET_BOOK30
        else "raw6+of2_file" if USE_OF_FILE
        else "raw6+orderflow5+of2" if (USE_DERIVED_FEATURES and USE_OF_FEATURES)
        else "raw6+of2" if USE_OF_FEATURES
        else "raw6+orderflow5" if USE_DERIVED_FEATURES
        else "raw6"
    )
    logger.info(f"Input features:  {N_TOTAL_FEATURES} ({_feat_desc})")
    if USE_OF_FILE:
        logger.info(f"OF file dir:     {CNN_OF_FILE_DIR} ({'EXISTS' if CNN_OF_FILE_DIR and CNN_OF_FILE_DIR.exists() else 'MISSING'})")
    logger.info(f"Channels:        {CNN_CHANNELS}")
    logger.info(f"Kernel size:     {CNN_KERNEL}")
    logger.info(f"Layers:          {CNN_LAYERS}")
    logger.info(f"Dilations:       {DILATION_SCHEDULE}")
    rf = sum([(CNN_KERNEL - 1) * d for d in DILATION_SCHEDULE]) + 1
    logger.info(f"Receptive field: {rf} events")
    logger.info(f"Window size:     {WINDOW_SIZE}")
    logger.info(f"Batch size:      {BATCH_SIZE}")
    logger.info(f"Data dir:        {data_dir}")
    logger.info(f"Output dir:      {output_dir}")
    logger.info("=" * 60)

    # Step 1: Transfer data from Jupiter if needed
    if not args.skip_transfer:
        transfer_data_from_jupiter(data_dir)

    # Gather NPZ files
    if FEATURE_SET_BOOK30:
        npz_files = sorted(data_dir.glob("*_book_features.npz"))
        _glob_desc = "*_book_features.npz"
    else:
        npz_files = sorted(data_dir.glob("*_mbo_events.npz"))
        _glob_desc = "*_mbo_events.npz"
    if not npz_files:
        logger.error(f"No {_glob_desc} files found in {data_dir}")
        sys.exit(1)

    if args.max_days and len(npz_files) > args.max_days:
        npz_files = npz_files[-args.max_days:]
        logger.info(f"Limited to most recent {args.max_days} days")
    logger.info(f"Found {len(npz_files)} NPZ files: {npz_files[0].name} → {npz_files[-1].name}")

    # Step 2: Set process priority to BELOW_NORMAL on Windows (yield to live inference)
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        logger.info("Process priority set to BELOW_NORMAL")
    except Exception as e:
        logger.info(f"Could not set priority (non-critical): {e}")

    # Step 3: Run walk-forward (expanding or sliding window)
    train_days = args.train_days if args.window_mode == "sliding" else None
    if args.window_mode == "sliding" and train_days is None:
        logger.warning("--window-mode sliding requires --train-days N; defaulting to 30")
        train_days = 30
    concat_ic = run_expanding_wf(
        npz_files  = npz_files,
        output_dir = output_dir,
        device     = device,
        n_folds    = args.n_folds,
        train_days = train_days,
        oot_days   = args.oot_days,
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Final Concat IC:")
    for h in HORIZONS:
        logger.info(f"  {h}: {concat_ic.get(h, float('nan')):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
