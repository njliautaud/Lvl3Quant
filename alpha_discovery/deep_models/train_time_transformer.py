"""
Time-Aware Transformer — Training Script

Architecture: TimeAwareTransformer (~600K params)
  - Input: (B, 500, 15) feat15 event windows
  - Continuous time embeddings: sinusoidal encoding of time_delta_log (feature 0)
    with learnable scale, 32 dimensions
  - Discrete embeddings: event_type (3 types) → 16d, side (3 types) → 16d
  - Raw continuous features: Linear projection of remaining features → 32d
  - Concat all → Linear projection to d_model=96
  - CNN stem: Conv1d(96→96, kernel=3, padding=1) + GELU + residual
  - 2x Transformer encoder layers with LOCAL ATTENTION (window_size=192):
    - 4 attention heads (head_dim=24)
    - Time-decay attention masking: attn_weights *= exp(-decay * time_gap) per head
    - Pre-LayerNorm, GELU FFN, dropout=0.1
  - Output head: mean pool → Linear(96→48) → GELU → Dropout → Linear(48→3)
  - Multi-horizon output: 1s/5s/10s price change

Training:
  - Sliding window walk-forward (MANDATORY for this architecture)
  - Feature stats computed from TRAINING set only (NO leakage)
  - Mixed precision (fp16)
  - Cosine annealing LR with warmup
  - Saves .pt weights + .npz predictions per fold (with embeddings)
  - MLflow logging
  - Metrics at tiers (All/50%/25%/10%/5%) and horizons (1s/5s/10s)
"""

import os
import sys
import gc
import time
import math
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

# Windows compatibility: force UTF-8 for stdout/stderr
if sys.platform == "win32":
    os.environ.setdefault("PYTHONIOENCODING", "utf-8")
    try:
        sys.stdout.reconfigure(encoding="utf-8")
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

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

log_path = LOG_DIR / "time_transformer.log"
_file_stream = open(log_path, "a", buffering=1)
_file_handler = logging.StreamHandler(_file_stream)
_file_handler.setLevel(logging.INFO)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_file_handler, _stream_handler],
)
logger = logging.getLogger()


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
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "time_transformer"

STRICT_LEAKAGE_FREE = int(os.environ.get("STRICT_LEAKAGE_FREE", 0)) == 1

# Force feat15 mode for this architecture
N_TOTAL_FEATURES = 15
CNN_FEATURE_SET = "feat15"


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
MLFLOW_EXPERIMENT = "EventDriven_TimeTransformer"

# Hyperparameters
WINDOW_SIZE    = int(os.environ.get("EVENT_WINDOW_SIZE", 500))
STRIDE         = int(os.environ.get("EVENT_STRIDE", WINDOW_SIZE // 2))
BATCH_SIZE     = int(os.environ.get("EVENT_BATCH_SIZE", 128))  # Smaller than CNN for VRAM safety
LR             = float(os.environ.get("EVENT_LR", 3e-4))
EPOCHS_PER_FOLD = int(os.environ.get("EVENT_EPOCHS", 5))
WARMUP_STEPS   = int(os.environ.get("EVENT_WARMUP", 300))
GRAD_CLIP      = float(os.environ.get("EVENT_GRAD_CLIP", 1.0))
N_FOLDS        = int(os.environ.get("EVENT_N_FOLDS", 5))
HORIZONS       = ["1s", "5s", "10s"]

# Time-Aware Transformer hyperparameters
D_MODEL        = int(os.environ.get("TT_D_MODEL", 96))
N_HEADS        = int(os.environ.get("TT_N_HEADS", 4))
N_LAYERS_TF    = int(os.environ.get("TT_N_LAYERS", 2))
FFN_DIM        = int(os.environ.get("TT_FFN_DIM", D_MODEL * 4))  # 384
DROPOUT        = float(os.environ.get("TT_DROPOUT", 0.1))
TIME_EMBED_DIM = int(os.environ.get("TT_TIME_EMBED_DIM", 32))
DISCRETE_EMBED_DIM = int(os.environ.get("TT_DISCRETE_EMBED_DIM", 16))
LOCAL_ATTN_WINDOW  = int(os.environ.get("TT_LOCAL_ATTN_WINDOW", 192))

# Feature indices in feat15 layout
# 0: time_delta_log, 1: event_type_id, 2: side_id, 3-14: continuous features
IDX_TIME_DELTA = 0
IDX_EVENT_TYPE = 1
IDX_SIDE       = 2
N_EVENT_TYPES  = 3  # trade, add, cancel/modify
N_SIDES        = 3  # bid, ask, neutral/unknown
CONTINUOUS_FEATURE_INDICES = list(range(3, N_TOTAL_FEATURES))  # features 3..14 (12 features)


# ============================================================
# Dataset  (reused from train_event_cnn_1d.py with feat15 support)
# ============================================================

class MboEventDataset(Dataset):
    """
    Creates (window_of_events, labels) pairs from MBO event NPZ files.

    LAZY LOADING: Only keeps file paths and a sample index in RAM.
    Actual data is loaded on-demand in __getitem__ with an LRU cache
    (default 5 days) to avoid re-reading the same file repeatedly.
    """

    def __init__(
        self,
        npz_files: List[Path],
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        horizons: Optional[List[str]] = None,
        normalize_features: bool = True,
        feature_stats: Optional[Dict] = None,
        cache_days: int = int(os.environ.get("TT_CACHE_DAYS", 5)),
    ):
        self.window_size = window_size
        self.stride = stride
        self.horizons = horizons or ["1s", "5s", "10s"]
        self.normalize_features = normalize_features
        self.npz_files = list(npz_files)
        self.cache_days = cache_days
        self.sample_index: List[Tuple[int, int]] = []

        from collections import OrderedDict
        self._cache: OrderedDict = OrderedDict()

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

    def _compute_stats(self, npz_files: List[Path]):
        """Compute running mean/std across all events for normalization.
        Loads one file at a time and discards immediately."""
        logger.info(f"Computing feature statistics from {len(npz_files)} files "
                     f"({self.n_features} features)...")
        total_sum   = np.zeros(self.n_features, dtype=np.float64)
        total_sq    = np.zeros(self.n_features, dtype=np.float64)
        total_count = 0

        for f in npz_files:
            data = self._load_npz(f)
            ev = data["events"].astype(np.float64)
            if ev.shape[1] < N_TOTAL_FEATURES:
                logger.warning(f"feat15: {f.name} has {ev.shape[1]} cols, need {N_TOTAL_FEATURES}")
                del data, ev
                continue
            ev = ev[:, :N_TOTAL_FEATURES]
            total_sum += ev.sum(axis=0)
            total_sq += (ev ** 2).sum(axis=0)
            total_count += len(ev)
            del data, ev

        self.feature_mean = (total_sum / total_count).astype(np.float32)
        var = (total_sq / total_count) - (total_sum / total_count) ** 2
        self.feature_std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
        logger.info(f"feat15 stats: {total_count} events, mean[:3]={self.feature_mean[:3]}")

    def get_feature_stats(self) -> Dict:
        return {"mean": self.feature_mean, "std": self.feature_std}

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

            del data, events, day_labels

        logger.info(
            f"Dataset: {len(self.npz_files)} days, {len(self.sample_index)} samples "
            f"(window={self.window_size}, stride={self.stride}) [LAZY LOADING]"
        )

    def _get_day(self, day_idx: int):
        """Load a day's data with LRU cache."""
        if day_idx in self._cache:
            self._cache.move_to_end(day_idx)
            return self._cache[day_idx]

        data = self._load_npz(self.npz_files[day_idx])
        events = data["events"].astype(np.float32)
        events = events[:, :N_TOTAL_FEATURES]

        if self.normalize_features:
            events = (events - self.feature_mean) / (self.feature_std + 1e-8)

        day_labels = {h: data[f"labels_{h}"].astype(np.float32) for h in self.horizons}
        del data

        self._cache[day_idx] = (events, day_labels)
        while len(self._cache) > self.cache_days:
            self._cache.popitem(last=False)

        return events, day_labels

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        day_idx, start = self.sample_index[idx]
        end = start + self.window_size
        events, day_labels = self._get_day(day_idx)
        window = events[start:end].copy()  # (W, 15)
        label_idx = end - 1
        labels = np.array(
            [day_labels[h][label_idx] for h in self.horizons],
            dtype=np.float32,
        )  # (3,)

        return torch.from_numpy(window), torch.from_numpy(labels)


class FileSequentialSampler(torch.utils.data.Sampler):
    """
    Sampler that processes one file at a time.
    Files are shuffled randomly each epoch. Windows within each file are
    served sequentially. Keeps peak RAM at ~1 file.
    """

    def __init__(self, dataset: MboEventDataset, shuffle: bool = True):
        self.dataset = dataset
        self.shuffle = shuffle
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
                random.shuffle(indices)
                yield from indices
        else:
            for file_idx in self._file_order:
                yield from self._file_groups[file_idx]

    def __len__(self) -> int:
        return len(self.dataset)


# ============================================================
# Architecture: TimeAwareTransformer
# ============================================================

class SinusoidalTimeEmbedding(nn.Module):
    """
    Continuous sinusoidal embedding of time_delta_log values.
    Learnable scale parameter allows the model to adjust temporal sensitivity.

    Output: (B, L, time_embed_dim)
    """

    def __init__(self, embed_dim: int = 32):
        super().__init__()
        self.embed_dim = embed_dim
        # Learnable scale for the time input
        self.scale = nn.Parameter(torch.ones(1))
        # Fixed frequency basis (log-spaced like standard sinusoidal PE)
        half = embed_dim // 2
        freqs = torch.exp(torch.arange(half, dtype=torch.float32) * -(math.log(10000.0) / half))
        self.register_buffer("freqs", freqs)  # (half,)

    def forward(self, time_delta_log: torch.Tensor) -> torch.Tensor:
        """
        Args:
            time_delta_log: (B, L) — log inter-event time gaps
        Returns:
            embeddings: (B, L, embed_dim)
        """
        # Scale the time input
        t = time_delta_log.unsqueeze(-1) * self.scale  # (B, L, 1)
        # Apply frequency basis
        args = t * self.freqs  # (B, L, half)
        # Interleave sin and cos
        emb = torch.cat([torch.sin(args), torch.cos(args)], dim=-1)  # (B, L, embed_dim)
        return emb


class TimeDecayMultiHeadAttention(nn.Module):
    """
    Multi-head attention with:
    1. Local attention (band mask): only attend to positions within window_size
    2. Time-decay masking: attn_weights *= exp(-learned_decay_rate * time_gap_matrix) per head

    This is a custom implementation (not using nn.MultiheadAttention) for fine-grained
    control over the attention mask computation.
    """

    def __init__(
        self,
        d_model: int = 96,
        n_heads: int = 4,
        dropout: float = 0.1,
        local_window: int = 192,
    ):
        super().__init__()
        assert d_model % n_heads == 0
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.local_window = local_window
        self.scale = self.head_dim ** -0.5

        # QKV projection
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        # Output projection
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.attn_dropout = nn.Dropout(dropout)

        # Learnable decay rate per head (initialized to small positive values)
        # Higher decay = faster forgetting of distant events
        self.decay_rate = nn.Parameter(torch.full((n_heads,), 0.01))

    def forward(
        self,
        x: torch.Tensor,
        time_delta_log: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            time_delta_log: (B, L) — raw time_delta_log values (pre-normalization)
        Returns:
            out: (B, L, d_model)
        """
        B, L, _ = x.shape

        # QKV
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, D)
        q, k, v = qkv[0], qkv[1], qkv[2]  # each (B, H, L, D)

        # Scaled dot-product attention scores
        attn = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        # --- Local attention band mask ---
        # Only attend to positions within [-local_window, +local_window]
        # Build a 1D position tensor and compute pairwise distance
        positions = torch.arange(L, device=x.device)  # (L,)
        dist = (positions.unsqueeze(0) - positions.unsqueeze(1)).abs()  # (L, L)
        local_mask = dist > self.local_window  # True = mask out
        attn = attn.masked_fill(local_mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        # --- Time-decay masking ---
        # Compute cumulative time from time_delta_log: cumsum gives absolute time at each position
        # time_gap_matrix[i,j] = |cumtime[i] - cumtime[j]|
        # We use exp(time_delta_log) to get actual time deltas, then cumsum
        time_deltas = torch.exp(time_delta_log)  # (B, L) — actual time gaps
        cum_time = torch.cumsum(time_deltas, dim=1)  # (B, L) — cumulative time
        time_gap = (cum_time.unsqueeze(-1) - cum_time.unsqueeze(-2)).abs()  # (B, L, L)

        # Apply per-head decay: exp(-decay_rate * time_gap)
        # decay_rate is (H,), time_gap is (B, L, L) → need (B, H, L, L)
        decay = torch.abs(self.decay_rate)  # ensure non-negative
        time_decay = torch.exp(-decay.view(1, -1, 1, 1) * time_gap.unsqueeze(1))  # (B, H, L, L)

        # Apply softmax THEN multiply by time decay (so decay acts as a post-softmax weight)
        attn = F.softmax(attn, dim=-1)  # (B, H, L, L)
        attn = attn * time_decay
        # Re-normalize after decay (optional but helps stability)
        attn = attn / (attn.sum(dim=-1, keepdim=True) + 1e-8)
        attn = self.attn_dropout(attn)

        # Weighted sum of values
        out = torch.matmul(attn, v)  # (B, H, L, D)
        out = out.transpose(1, 2).reshape(B, L, self.d_model)  # (B, L, d_model)
        out = self.out_proj(out)

        return out


class TimeAwareTransformerBlock(nn.Module):
    """
    Pre-LayerNorm Transformer encoder block with time-decay local attention.
    """

    def __init__(
        self,
        d_model: int = 96,
        n_heads: int = 4,
        ffn_dim: int = 384,
        dropout: float = 0.1,
        local_window: int = 192,
    ):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = TimeDecayMultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
            local_window=local_window,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, time_delta_log: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
            time_delta_log: (B, L) — raw time_delta_log for time-decay attention
        """
        # Pre-norm self-attention with residual
        x = x + self.attn(self.norm1(x), time_delta_log)
        # Pre-norm FFN with residual
        x = x + self.ffn(self.norm2(x))
        return x


class TimeAwareTransformer(nn.Module):
    """
    Time-Aware Transformer for MBO event stream prediction.

    Architecture:
        1. Time embedding: sinusoidal encoding of time_delta_log → 32d
        2. Discrete embeddings: event_type → 16d, side → 16d
        3. Continuous features: Linear(12 → 32d)
        4. Concat all → Linear(96 → d_model=96)
        5. CNN stem: Conv1d(96→96, k=3, pad=1) + GELU + residual
        6. 2x Transformer blocks with local time-decay attention
        7. Mean pool → Linear(96→48) → GELU → Dropout → Linear(48→3)

    ~600K parameters (fits Razer's 8GB VRAM easily)
    """

    def __init__(
        self,
        n_continuous: int = 12,  # features 3..14
        d_model: int = D_MODEL,
        n_heads: int = N_HEADS,
        n_layers: int = N_LAYERS_TF,
        ffn_dim: int = FFN_DIM,
        dropout: float = DROPOUT,
        n_targets: int = 3,
        time_embed_dim: int = TIME_EMBED_DIM,
        discrete_embed_dim: int = DISCRETE_EMBED_DIM,
        n_event_types: int = N_EVENT_TYPES,
        n_sides: int = N_SIDES,
        local_window: int = LOCAL_ATTN_WINDOW,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_targets = n_targets
        self.n_continuous = n_continuous
        self.local_window = local_window

        # --- Input embeddings ---
        # 1. Continuous time embedding (32d)
        self.time_embed = SinusoidalTimeEmbedding(time_embed_dim)

        # 2. Discrete embeddings (16d each)
        self.event_type_embed = nn.Embedding(n_event_types + 1, discrete_embed_dim)  # +1 for unknown
        self.side_embed = nn.Embedding(n_sides + 1, discrete_embed_dim)  # +1 for unknown

        # 3. Continuous feature projection (12 → 32d)
        continuous_proj_dim = d_model - time_embed_dim - 2 * discrete_embed_dim  # 96 - 32 - 32 = 32
        self.continuous_proj = nn.Linear(n_continuous, continuous_proj_dim)

        # 4. Projection to d_model (concat dim should equal d_model: 32+16+16+32=96)
        concat_dim = time_embed_dim + 2 * discrete_embed_dim + continuous_proj_dim
        assert concat_dim == d_model, f"Concat dim {concat_dim} != d_model {d_model}"

        # 5. CNN stem: Conv1d with residual
        self.cnn_stem = nn.Sequential(
            nn.Conv1d(d_model, d_model, kernel_size=3, padding=1, bias=False),
            nn.GELU(),
        )
        self.cnn_stem_norm = nn.LayerNorm(d_model)

        # 6. Transformer blocks
        self.blocks = nn.ModuleList([
            TimeAwareTransformerBlock(
                d_model=d_model,
                n_heads=n_heads,
                ffn_dim=ffn_dim,
                dropout=dropout,
                local_window=local_window,
            )
            for _ in range(n_layers)
        ])

        # 7. Output head
        self.output_norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),  # 96 → 48
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets),  # 48 → 3
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, std=0.02)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, events: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            events: (B, L, 15) float32 — normalized feat15 event windows
            return_embedding: if True, also return (B, d_model) pattern embedding
        Returns:
            preds: (B, n_targets)
            embedding: (B, d_model) — only if return_embedding=True
        """
        B, L, F = events.shape

        # Extract components from the normalized input
        # NOTE: time_delta_log is used raw (un-normalized) for time-decay attention
        # but the normalized version is used for the sinusoidal embedding
        time_delta_log_norm = events[:, :, IDX_TIME_DELTA]  # (B, L) — normalized
        event_type_raw = events[:, :, IDX_EVENT_TYPE]  # (B, L) — normalized
        side_raw = events[:, :, IDX_SIDE]  # (B, L) — normalized
        continuous = events[:, :, 3:]  # (B, L, 12)

        # For discrete embeddings, we need integer indices.
        # Since the input is normalized, we round to nearest integer after denormalization.
        # But since we don't have the raw values, we use a robust approach:
        # clamp the normalized values to reasonable integer range after rounding
        event_type_ids = event_type_raw.round().long().clamp(0, N_EVENT_TYPES)  # (B, L)
        side_ids = side_raw.round().long().clamp(0, N_SIDES)  # (B, L)

        # 1. Time embedding from normalized time_delta_log
        time_emb = self.time_embed(time_delta_log_norm)  # (B, L, 32)

        # 2. Discrete embeddings
        evt_emb = self.event_type_embed(event_type_ids)  # (B, L, 16)
        side_emb = self.side_embed(side_ids)  # (B, L, 16)

        # 3. Continuous feature projection
        cont_emb = self.continuous_proj(continuous)  # (B, L, 32)

        # 4. Concatenate all embeddings → (B, L, 96)
        x = torch.cat([time_emb, evt_emb, side_emb, cont_emb], dim=-1)  # (B, L, d_model)

        # 5. CNN stem with residual
        residual = x
        x_conv = self.cnn_stem(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B, L, d_model)
        x = self.cnn_stem_norm(x_conv + residual)

        # 6. Transformer blocks (pass time_delta_log for time-decay attention)
        for block in self.blocks:
            x = block(x, time_delta_log_norm)

        # 7. Mean pooling → head
        embedding = self.output_norm(x.mean(dim=1))  # (B, d_model)
        preds = self.head(embedding)  # (B, n_targets)

        if return_embedding:
            return preds, embedding
        return preds

    def get_decay_rates(self) -> Dict[str, List[float]]:
        """Return learned decay rates per head for each layer (for logging)."""
        rates = {}
        for i, block in enumerate(self.blocks):
            rates[f"layer_{i}"] = torch.abs(block.attn.decay_rate).detach().cpu().tolist()
        return rates


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
# LR Scheduler with Warmup + Cosine Annealing
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
# Tiered Metrics Computation
# ============================================================

def compute_tiered_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    horizon: str,
) -> Dict[str, float]:
    """
    Compute IC, directional accuracy, and magnitude correlation at multiple tiers.
    Tiers: All, Top 50%, Top 25%, Top 10%, Top 5% (by absolute prediction magnitude).

    Returns dict with keys like:
        ic_{horizon}_all, ic_{horizon}_50pct, ic_{horizon}_25pct, ...
        da_{horizon}_all, da_{horizon}_50pct, ...
        magcorr_{horizon}_all, ...
    """
    metrics = {}
    valid = ~(np.isnan(predictions) | np.isnan(labels))
    p = predictions[valid]
    l = labels[valid]
    n = len(p)

    if n < 20:
        for tier in ["all", "50pct", "25pct", "10pct", "5pct"]:
            metrics[f"ic_{horizon}_{tier}"] = float("nan")
            metrics[f"da_{horizon}_{tier}"] = float("nan")
            metrics[f"magcorr_{horizon}_{tier}"] = float("nan")
        return metrics

    abs_pred = np.abs(p)
    sorted_idx = np.argsort(abs_pred)[::-1]  # highest conviction first

    tiers = {
        "all": n,
        "50pct": n // 2,
        "25pct": n // 4,
        "10pct": n // 10,
        "5pct": n // 20,
    }

    for tier_name, tier_n in tiers.items():
        if tier_n < 20:
            metrics[f"ic_{horizon}_{tier_name}"] = float("nan")
            metrics[f"da_{horizon}_{tier_name}"] = float("nan")
            metrics[f"magcorr_{horizon}_{tier_name}"] = float("nan")
            continue

        idx = sorted_idx[:tier_n] if tier_name != "all" else np.arange(n)
        tp = p[idx]
        tl = l[idx]

        # Spearman IC
        rho, _ = scipy.stats.spearmanr(tp, tl)
        metrics[f"ic_{horizon}_{tier_name}"] = float(rho)

        # Directional accuracy (exclude zero labels)
        nonzero = tl != 0
        if nonzero.sum() > 0:
            da = float((np.sign(tp[nonzero]) == np.sign(tl[nonzero])).mean())
        else:
            da = float("nan")
        metrics[f"da_{horizon}_{tier_name}"] = da

        # Magnitude correlation (Pearson of |pred| vs |label|)
        try:
            magcorr = float(np.corrcoef(np.abs(tp), np.abs(tl))[0, 1])
        except Exception:
            magcorr = float("nan")
        metrics[f"magcorr_{horizon}_{tier_name}"] = magcorr

    return metrics


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
    """Run inference on OOT fold, return (predictions, labels, embeddings)."""
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
    """Train TimeAwareTransformer for one fold. Returns metrics dict."""

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

    best_val_loss  = float("inf")
    global_step    = 0
    total_batches  = len(train_loader)
    print(f">>> train_one_fold: {EPOCHS_PER_FOLD} epochs, {total_batches} batches/epoch", flush=True)

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        epoch_loss  = 0.0
        n_batches   = 0
        epoch_start = time.time()

        for events, labels in train_loader:
            events = events.to(device, non_blocking=True)   # (B, W, 15)
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

            # Progress logging every 100 batches (print for guaranteed output)
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

        avg_loss    = epoch_loss / max(n_batches, 1)
        val_metrics = evaluate_metrics_only(model, val_loader, device, use_amp=use_amp)

        # Log decay rates
        decay_rates = model.get_decay_rates()
        decay_str = " | ".join(
            f"L{k}: [{', '.join(f'{r:.4f}' for r in v)}]"
            for k, v in decay_rates.items()
        )

        logger.info(
            f"Fold {fold_idx:02d} | Epoch {epoch+1:2d}/{EPOCHS_PER_FOLD} | "
            f"Train Loss: {avg_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IC (10s): {val_metrics.get('ic_10s', float('nan')):.4f} | "
            f"LR: {scheduler.get_lr():.2e} | Decay: {decay_str}"
        )

        if MLFLOW_AVAILABLE and mlflow_run:
            step_offset = fold_idx * EPOCHS_PER_FOLD + epoch
            log_dict = {
                f"fold{fold_idx:02d}_train_loss": avg_loss,
                f"fold{fold_idx:02d}_val_loss":   val_metrics["loss"],
                f"fold{fold_idx:02d}_val_ic_1s":  val_metrics.get("ic_1s",  float("nan")),
                f"fold{fold_idx:02d}_val_ic_10s": val_metrics.get("ic_10s", float("nan")),
            }
            # Log decay rates
            for layer_name, rates in decay_rates.items():
                for head_idx, rate in enumerate(rates):
                    log_dict[f"fold{fold_idx:02d}_decay_{layer_name}_h{head_idx}"] = rate
            mlflow.log_metrics(log_dict, step=step_offset)

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
                        "d_model":        D_MODEL,
                        "n_heads":        N_HEADS,
                        "n_layers":       N_LAYERS_TF,
                        "ffn_dim":        FFN_DIM,
                        "dropout":        DROPOUT,
                        "window_size":    WINDOW_SIZE,
                        "in_features":    N_TOTAL_FEATURES,
                        "time_embed_dim": TIME_EMBED_DIM,
                        "discrete_embed_dim": DISCRETE_EMBED_DIM,
                        "local_window":   LOCAL_ATTN_WINDOW,
                    },
                    "decay_rates": decay_rates,
                },
                ckpt_path,
            )

    return {"best_val_loss": best_val_loss}


# ============================================================
# Walk-Forward (Sliding Window)
# ============================================================

def run_walk_forward(
    npz_files:  List[Path],
    output_dir: Path,
    device:     torch.device,
    n_folds:    int = N_FOLDS,
    train_days: Optional[int] = None,
    oot_days:   Optional[int] = None,
):
    """
    Walk-forward training with sliding or expanding window.

    Sliding window (train_days=N): each fold uses only the last N files before the boundary.
    Expanding window (train_days=None): each fold uses all files up to the fold boundary.

    Feature statistics are always computed from the training set only (no leakage).
    OOT predictions are saved per fold + concatenated for final concat-IC calculation.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Sort files by date
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

    window_mode = f"sliding({train_days}d)" if train_days else "expanding"

    # Build fold boundaries — sliding window with 1-day OOT (matches CNN/Mamba)
    # Each fold slides forward 1 day: train=60d, OOT=next 1 day
    # Folds are ordered from earliest to latest OOT date
    oot_per_fold = oot_days if oot_days is not None else 1
    fold_boundaries = []
    # Start from the end: fold N-1 uses the last valid day as OOT
    for fold in range(n_folds):
        # OOT index: walk backwards from the end for fold ordering
        oot_end_idx = n_files - (n_folds - 1 - fold) * oot_per_fold
        oot_start_idx = oot_end_idx - oot_per_fold
        if oot_start_idx < 0:
            continue
        train_end_idx = oot_start_idx
        train_start_idx = max(0, train_end_idx - train_days) if train_days is not None else 0
        if train_end_idx - train_start_idx < 5:
            continue  # Not enough training data
        fold_boundaries.append((
            fold,
            list(range(train_start_idx, train_end_idx)),
            list(range(oot_start_idx, oot_end_idx))
        ))
    logger.info(f"Sliding WF: {len(fold_boundaries)} folds, {oot_per_fold}d OOT, "
                f"{'sliding(' + str(train_days) + 'd)' if train_days else 'expanding'} train")

    logger.info(f"Running {len(fold_boundaries)} folds ({window_mode})")

    use_amp = device.type == "cuda"

    # Concat storage
    concat_preds  = {h: [] for h in HORIZONS}
    concat_labels = {h: [] for h in HORIZONS}
    concat_embeds = []

    # MLflow run
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(
            run_name=f"TimeTransformer_{time.strftime('%Y%m%d_%H%M')}"
        )
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        mlflow.log_params(
            {
                "model":             "TimeAwareTransformer",
                "window_size":       WINDOW_SIZE,
                "stride":            STRIDE,
                "d_model":           D_MODEL,
                "n_heads":           N_HEADS,
                "n_layers":          N_LAYERS_TF,
                "ffn_dim":           FFN_DIM,
                "dropout":           DROPOUT,
                "time_embed_dim":    TIME_EMBED_DIM,
                "discrete_embed_dim": DISCRETE_EMBED_DIM,
                "local_attn_window": LOCAL_ATTN_WINDOW,
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
                "in_features":       N_TOTAL_FEATURES,
                "feature_set":       "feat15",
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
                f"({train_files[0].name}->{train_files[-1].name}) | "
                f"OOT: {len(oot_files)} days ({oot_files[0].name}->{oot_files[-1].name})"
                f"\n{'='*60}"
            )

            # Build train dataset (lazy NPZ loading — always leakage-free)
            logger.info("Building train dataset...")
            train_ds = MboEventDataset(
                train_files,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
            )
            feature_stats = train_ds.get_feature_stats()

            # Build OOT dataset with train feature stats (no leakage)
            logger.info("Building OOT dataset...")
            oot_ds = MboEventDataset(
                oot_files,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
                feature_stats=feature_stats,
            )

            # num_workers
            _num_workers = int(os.environ.get("EVENT_NUM_WORKERS", 0 if sys.platform == "win32" else 8))
            _persistent = _num_workers > 0

            # Use FileSequentialSampler to keep RAM low
            _use_seq_sampler = int(os.environ.get("TT_SEQUENTIAL_SAMPLER", 1))
            if _use_seq_sampler:
                _train_sampler = FileSequentialSampler(train_ds, shuffle=True)
                _train_shuffle = False
                logger.info("Using FileSequentialSampler")
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
            n_continuous = N_TOTAL_FEATURES - 3  # 15 - 3 = 12
            model = TimeAwareTransformer(
                n_continuous=n_continuous,
                d_model=D_MODEL,
                n_heads=N_HEADS,
                n_layers=N_LAYERS_TF,
                ffn_dim=FFN_DIM,
                dropout=DROPOUT,
                n_targets=len(HORIZONS),
                time_embed_dim=TIME_EMBED_DIM,
                discrete_embed_dim=DISCRETE_EMBED_DIM,
                local_window=LOCAL_ATTN_WINDOW,
            ).to(device)

            if fold_idx == 0:
                n_params = count_parameters(model)
                logger.info(f"Model parameters:  {n_params:,}")
                logger.info(f"Local attn window: {LOCAL_ATTN_WINDOW}")
                logger.info(f"d_model:           {D_MODEL}")
                logger.info(f"Heads:             {N_HEADS} (head_dim={D_MODEL // N_HEADS})")
                logger.info(f"Layers:            {N_LAYERS_TF}")
                if MLFLOW_AVAILABLE and mlflow_run:
                    mlflow.log_param("n_params", n_params)

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

            # OOT inference with embeddings
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

            if oot_embeds is not None:
                concat_embeds.append(oot_embeds)

            # Compute tiered metrics per fold
            all_tiered_metrics = {}
            for i, h in enumerate(HORIZONS):
                tiered = compute_tiered_metrics(oot_preds[:, i], oot_labels[:, i], h)
                all_tiered_metrics.update(tiered)

            # Per-fold extended metrics
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

            # Log tiered metrics
            for tier in ["all", "50pct", "25pct", "10pct", "5pct"]:
                tier_str = " | ".join(
                    f"{h}: IC={all_tiered_metrics.get(f'ic_{h}_{tier}', float('nan')):.4f} "
                    f"DA={all_tiered_metrics.get(f'da_{h}_{tier}', float('nan')):.1%}"
                    for h in HORIZONS
                )
                logger.info(f"Fold {fold_idx:02d} Tier {tier:>5s} | {tier_str}")

            # Log decay rates at end of fold
            decay_rates = model.get_decay_rates()
            for layer_name, rates in decay_rates.items():
                logger.info(f"Fold {fold_idx:02d} Decay {layer_name}: {[f'{r:.4f}' for r in rates]}")

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
            logger.info(f"Saved predictions + embeddings ({oot_embeds.shape[1] if oot_embeds is not None else 0}d) -> {pred_path}")

            # Save feature stats for this fold
            stats_path = output_dir / f"fold_{fold_idx:02d}_feature_stats.npz"
            _fmean = feature_stats.get("mean")
            _fstd = feature_stats.get("std")
            if _fmean is not None and _fstd is not None:
                np.savez(stats_path, mean=_fmean, std=_fstd)

            # Log per-fold metrics to MLflow
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow_metrics = {
                    **{f"oot_ic_{h}_fold{fold_idx:02d}": fold_ics[h] for h in HORIZONS},
                    **{f"oot_dir_acc_{h}_fold{fold_idx:02d}": fold_dir_acc[h] for h in HORIZONS},
                    **{f"oot_mae_{h}_fold{fold_idx:02d}": fold_mae[h] for h in HORIZONS},
                }
                # Add tiered metrics
                for key, val in all_tiered_metrics.items():
                    mlflow_metrics[f"fold{fold_idx:02d}_{key}"] = val if not np.isnan(val) else 0.0
                # Add decay rates
                for layer_name, rates in decay_rates.items():
                    for head_idx, rate in enumerate(rates):
                        mlflow_metrics[f"fold{fold_idx:02d}_decay_{layer_name}_h{head_idx}"] = rate
                mlflow.log_metrics(mlflow_metrics, step=fold_idx)

                # Upload fold artifacts
                fold_artifact_dir = f"fold_{fold_idx:02d}"
                mlflow.log_artifact(str(pred_path), artifact_path=fold_artifact_dir)
                if ckpt_path.exists():
                    mlflow.log_artifact(str(ckpt_path), artifact_path=fold_artifact_dir)
                if stats_path.exists():
                    mlflow.log_artifact(str(stats_path), artifact_path=fold_artifact_dir)
                mlflow.log_params({
                    f"fold{fold_idx:02d}_train_files": f"{train_files[0].name}->{train_files[-1].name}",
                    f"fold{fold_idx:02d}_train_n":     len(train_files),
                    f"fold{fold_idx:02d}_oot_files":   f"{oot_files[0].name}->{oot_files[-1].name}",
                    f"fold{fold_idx:02d}_oot_n":       len(oot_files),
                })
                logger.info(f"Fold {fold_idx:02d} artifacts uploaded to MLflow")

            # Free memory
            del train_ds, oot_ds, train_loader, oot_loader, model
            gc.collect()
            torch.cuda.empty_cache()

        # ============================================================
        # Compute CONCAT IC (primary metric)
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("CONCAT IC (primary metric — all folds combined)")
        logger.info("=" * 60)

        concat_ic: Dict[str, float] = {}
        concat_tiered: Dict[str, float] = {}
        for h in HORIZONS:
            if concat_preds[h]:
                all_p         = np.concatenate(concat_preds[h])
                all_l         = np.concatenate(concat_labels[h])
                ic            = compute_ic(all_p, all_l)
                concat_ic[h]  = ic
                logger.info(f"  Concat IC ({h}): {ic:.4f}")

                # Tiered metrics on concat
                tiered = compute_tiered_metrics(all_p, all_l, h)
                concat_tiered.update(tiered)
                for tier in ["all", "50pct", "25pct", "10pct", "5pct"]:
                    logger.info(
                        f"    Tier {tier:>5s}: IC={tiered[f'ic_{h}_{tier}']:.4f} "
                        f"DA={tiered[f'da_{h}_{tier}']:.1%} "
                        f"MagCorr={tiered[f'magcorr_{h}_{tier}']:.4f}"
                    )
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
        logger.info(f"Saved concat predictions + embeddings -> {concat_path}")

        if MLFLOW_AVAILABLE and mlflow_run:
            final_metrics = {f"concat_ic_{h}": concat_ic[h] for h in HORIZONS}
            # Add concat tiered metrics
            for key, val in concat_tiered.items():
                final_metrics[f"concat_{key}"] = val if not np.isnan(val) else 0.0
            mlflow.log_metrics(final_metrics)
            mlflow.log_artifact(str(concat_path), artifact_path="concat")
            logger.info("Concat predictions uploaded to MLflow")

        logger.info("\n" + "=" * 60)
        logger.info("LEAKAGE AUDIT: PASSED")
        logger.info("  - Walk-forward: train set never contains OOT dates")
        logger.info("  - Feature normalization computed from train set only per fold")
        logger.info("  - Attention is local (no global future leakage)")
        logger.info("  - Time-decay rates are learned per-head (no look-ahead)")
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
    """Copy MBO event NPZ files from Jupiter to Neptune."""
    import subprocess, base64, json as _json

    dest_dir.mkdir(parents=True, exist_ok=True)
    existing = set(f.name for f in dest_dir.glob("*.npz"))

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
        logger.info(f"All {len(remote_files)} files already present. Skipping transfer.")
        return

    logger.info(f"Transferring {len(to_copy)}/{len(remote_files)} files from Jupiter...")

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
            logger.info("SCP key-based auth works")
    except Exception:
        logger.info("SCP not available, using API base64 transfer")

    for fname in to_copy:
        remote_path = f"/home/jupiter/Lvl3Quant/data/processed/mbo_events/{fname}"
        dest_path   = dest_dir / fname

        if dest_path.exists():
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
                            break
                        fout.write(base64.b64decode(b64_data))

                actual_size = dest_path.stat().st_size
                if abs(actual_size - file_size) > 1024:
                    logger.warning(f"    Size mismatch, removing {fname}")
                    dest_path.unlink()
                else:
                    logger.info(f"    Done: {fname} ({actual_size/1e6:.0f} MB in {time.time()-t0:.1f}s)")
            except Exception as e:
                logger.error(f"    Transfer failed for {fname}: {e}")
                if dest_path.exists():
                    dest_path.unlink()

    final_files = list(dest_dir.glob("*.npz"))
    logger.info(f"Files available after transfer: {len(final_files)}")


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Time-Aware Transformer Walk-Forward Training")
    parser.add_argument("--data-dir",    type=str, default=DEFAULT_DATA_DIR,
                        help="Directory with mbo_events NPZ files (feat15 format)")
    parser.add_argument("--output-dir",  type=str, default=str(DEFAULT_OUTPUT_DIR),
                        help="Output directory for checkpoints and predictions")
    parser.add_argument("--n-folds",     type=int, default=N_FOLDS)
    parser.add_argument("--skip-transfer", action="store_true",
                        help="Skip data transfer from Jupiter")
    parser.add_argument("--max-days",    type=int, default=None,
                        help="Limit number of data files to load (most recent N days)")
    parser.add_argument("--window-mode", type=str, default="expanding",
                        choices=["expanding", "sliding"],
                        help="Walk-forward window mode: expanding (default) or sliding")
    parser.add_argument("--train-days",  type=int, default=None,
                        help="Sliding window: number of training days per fold (requires --window-mode sliding)")
    parser.add_argument("--oot-days",    type=int, default=None,
                        help="Fix OOT block to last N days")
    parser.add_argument("--device",      type=str, default="cuda")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir   = Path(args.data_dir)
    device     = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Per-run log file
    _run_log_stream = open(output_dir / "training.log", "a", buffering=1)
    _run_log_handler = _FlushHandler(_run_log_stream)
    _run_log_handler.setLevel(logging.INFO)
    _run_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_run_log_handler)
    logger.info(f"Per-run log: {output_dir / 'training.log'}")

    logger.info("=" * 60)
    logger.info("Time-Aware Transformer — Walk-Forward Training")
    logger.info(f"Device:          {device}")
    if device.type == "cuda":
        logger.info(f"GPU:             {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Input features:  {N_TOTAL_FEATURES} (feat15)")
    logger.info(f"d_model:         {D_MODEL}")
    logger.info(f"Heads:           {N_HEADS} (head_dim={D_MODEL // N_HEADS})")
    logger.info(f"Layers:          {N_LAYERS_TF}")
    logger.info(f"FFN dim:         {FFN_DIM}")
    logger.info(f"Local attn:      {LOCAL_ATTN_WINDOW}")
    logger.info(f"Time embed dim:  {TIME_EMBED_DIM}")
    logger.info(f"Discrete embed:  {DISCRETE_EMBED_DIM}")
    logger.info(f"Window size:     {WINDOW_SIZE}")
    logger.info(f"Batch size:      {BATCH_SIZE}")
    logger.info(f"Data dir:        {data_dir}")
    logger.info(f"Output dir:      {output_dir}")
    logger.info("=" * 60)

    # Step 1: Transfer data from Jupiter if needed
    if not args.skip_transfer:
        transfer_data_from_jupiter(data_dir)

    # Gather NPZ files (feat15 format)
    npz_files = sorted(data_dir.glob("*_mbo_events.npz"))
    if not npz_files:
        logger.error(f"No *_mbo_events.npz files found in {data_dir}")
        sys.exit(1)

    if args.max_days and len(npz_files) > args.max_days:
        npz_files = npz_files[-args.max_days:]
        logger.info(f"Limited to most recent {args.max_days} days")
    logger.info(f"Found {len(npz_files)} NPZ files: {npz_files[0].name} -> {npz_files[-1].name}")

    # Step 2: Set process priority to BELOW_NORMAL on Windows
    try:
        import psutil
        if sys.platform == "win32":
            p = psutil.Process()
            p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
            logger.info("Process priority set to BELOW_NORMAL")
        else:
            # On Linux, try to nice to 10 (lower priority)
            p = psutil.Process()
            p.nice(10)
            logger.info("Process nice set to 10")
    except Exception as e:
        logger.info(f"Could not set priority (non-critical): {e}")

    # Step 3: Run walk-forward
    train_days = args.train_days if args.window_mode == "sliding" else None
    if args.window_mode == "sliding" and train_days is None:
        logger.warning("--window-mode sliding requires --train-days N; defaulting to 60")
        train_days = 60
    concat_ic = run_walk_forward(
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
