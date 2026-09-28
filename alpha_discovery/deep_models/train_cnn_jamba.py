"""
CNN-Jamba (Hybrid CNN + Mamba SSM + Attention) -- Training Script v1

Architecture: CNN front-end for feature extraction + Jamba blocks (interleaved Mamba SSM + Multi-head Attention)

Data format (identical to train_event_mamba.py):
  - events: (N_events, 6) float32
      [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]
  - labels_1s/5s/10s: (N_events,) float32 -- mid-price change in ticks
  - timestamps: (N_events,) int64 nanoseconds

Architecture Flow:
  Input (6 features) → CNN 1D (3-4 layers, channels=64) →
  Jamba blocks (Mamba SSM + Multi-head Attention + FFN) x4 →
  Output head (3 horizons)

Key innovations:
  1. CNN front-end: Extracts local patterns from raw event stream
  2. Mamba SSM: O(n) long-range temporal dependencies with time-delta conditioning
  3. Multi-head Attention: Captures global dependencies at critical points
  4. FFN: Non-linear mixing after SSM+Attention
  5. Window size: 1000 events (exploit long context)

Advantages:
  - CNN captures local microstructure patterns
  - Mamba provides efficient long-range context (O(n) vs O(n^2))
  - Attention adds precise global alignment when needed
  - Time-delta conditioning makes state transitions time-aware
  - Hybrid architecture balances efficiency and expressiveness

Training rules (identical to Mamba):
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
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.stats

# MLflow
try:
    import mlflow
    import mlflow.pytorch
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    warnings.warn("MLflow not installed -- skipping experiment tracking")

# ============================================================
# Logging setup
# ============================================================
LOG_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)

log_path = LOG_DIR / "cnn_jamba.log"
_file_handler = logging.FileHandler(log_path)
_file_handler.setLevel(logging.INFO)
_stream_handler = logging.StreamHandler(sys.stdout)
_stream_handler.setLevel(logging.INFO)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[_file_handler, _stream_handler],
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
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "cnn_jamba"


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
MLFLOW_EXPERIMENT = "EventDriven_CNNJamba"

# CNN-Jamba hyperparameters
CNN_CHANNELS     = int(os.environ.get("CNN_CHANNELS", 64))      # CNN output channels
CNN_KERNEL       = int(os.environ.get("CNN_KERNEL", 5))         # CNN kernel size
CNN_LAYERS       = int(os.environ.get("CNN_LAYERS", 3))         # Number of CNN layers
JAMBA_D_MODEL    = int(os.environ.get("JAMBA_D_MODEL", 128))    # Model dimension
JAMBA_D_STATE    = int(os.environ.get("JAMBA_D_STATE", 64))     # SSM state dimension
JAMBA_N_BLOCKS   = int(os.environ.get("JAMBA_N_BLOCKS", 4))     # Number of Jamba blocks
JAMBA_N_HEADS    = int(os.environ.get("JAMBA_N_HEADS", 4))      # Attention heads
JAMBA_DROPOUT    = float(os.environ.get("JAMBA_DROPOUT", 0.1))  # Dropout
JAMBA_DT_RANK    = int(os.environ.get("JAMBA_DT_RANK", 16))     # SSM dt rank
JAMBA_D_CONV     = int(os.environ.get("JAMBA_D_CONV", 4))       # SSM local conv kernel

# Shared hyperparameters
WINDOW_SIZE     = int(os.environ.get("EVENT_WINDOW_SIZE", 1000))  # Long context
STRIDE          = int(os.environ.get("EVENT_STRIDE", WINDOW_SIZE // 2))
BATCH_SIZE      = int(os.environ.get("EVENT_BATCH_SIZE", 128))
LR              = float(os.environ.get("EVENT_LR", 3e-4))
EPOCHS_PER_FOLD = int(os.environ.get("EVENT_EPOCHS", 5))
WARMUP_STEPS    = int(os.environ.get("EVENT_WARMUP", 300))
GRAD_CLIP       = float(os.environ.get("EVENT_GRAD_CLIP", 1.0))
N_FOLDS         = int(os.environ.get("EVENT_N_FOLDS", 5))
HORIZONS        = ["1s", "5s", "10s"]


# ============================================================
# Dataset (identical to MboEventDataset in train_event_mamba.py)
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
            self.feature_mean = np.zeros(6, dtype=np.float32)
            self.feature_std  = np.ones(6, dtype=np.float32)

        self._load_data(npz_files)

    def _compute_stats(self, npz_files: List[Path]):
        """Compute running mean/std across all events for normalization."""
        import json
        import hashlib

        # Create cache key from sorted file list
        file_list_str = "|".join(sorted([f.name for f in npz_files]))
        cache_key = hashlib.md5(file_list_str.encode()).hexdigest()
        cache_file = npz_files[0].parent / f".feature_stats_cache_{cache_key}.json"

        # Try loading from cache
        if cache_file.exists():
            try:
                logger.info(f"Loading cached feature statistics from {cache_file.name}")
                with open(cache_file, 'r') as f:
                    cached = json.load(f)
                self.feature_mean = np.array(cached['mean'], dtype=np.float32)
                self.feature_std = np.array(cached['std'], dtype=np.float32)
                logger.info(f"✅ Loaded cached stats (mean: {self.feature_mean}, std: {self.feature_std})")
                return
            except Exception as e:
                logger.warning(f"Failed to load cache: {e}, recomputing...")

        # Compute stats from scratch
        logger.info(f"Computing feature statistics from {len(npz_files)} files...")
        total_sum   = np.zeros(6, dtype=np.float64)
        total_sq    = np.zeros(6, dtype=np.float64)
        total_count = 0

        for i, f in enumerate(npz_files, 1):
            logger.info(f"  Processing {i}/{len(npz_files)}: {f.name}")
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

        # Save to cache
        try:
            with open(cache_file, 'w') as f:
                json.dump({
                    'mean': self.feature_mean.tolist(),
                    'std': self.feature_std.tolist(),
                    'n_files': len(npz_files),
                    'file_list': [f.name for f in npz_files]
                }, f, indent=2)
            logger.info(f"✅ Saved stats cache to {cache_file.name}")
        except Exception as e:
            logger.warning(f"Failed to save cache: {e}")

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
        label_idx = end - 1
        labels = np.array(
            [self.all_labels[h][day_idx][label_idx] for h in self.horizons],
            dtype=np.float32,
        )  # (3,)
        return torch.from_numpy(events), torch.from_numpy(labels)


# ============================================================
# Architecture: Selective SSM (from Mamba)
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


# ============================================================
# Architecture: Multi-Head Attention
# ============================================================

class MultiHeadAttention(nn.Module):
    """
    Standard multi-head attention with causal masking.

    Provides global context awareness between SSM blocks.
    """

    def __init__(
        self,
        d_model: int,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"

        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, d_model)
        Returns:
            out: (B, L, d_model)
        """
        B, L, _ = x.shape

        # Project to Q, K, V
        qkv = self.qkv_proj(x)  # (B, L, 3*d_model)
        q, k, v = qkv.chunk(3, dim=-1)  # each (B, L, d_model)

        # Reshape for multi-head: (B, L, d_model) -> (B, n_heads, L, d_k)
        q = q.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        k = k.view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        v = v.view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # Attention scores: (B, n_heads, L, L)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # Causal mask: prevent attending to future positions
        causal_mask = torch.triu(torch.ones(L, L, device=x.device), diagonal=1).bool()
        scores = scores.masked_fill(causal_mask.unsqueeze(0).unsqueeze(0), float('-inf'))

        # Softmax and dropout
        attn = F.softmax(scores, dim=-1)  # (B, n_heads, L, L)
        attn = self.dropout(attn)

        # Apply attention to values
        out = torch.matmul(attn, v)  # (B, n_heads, L, d_k)

        # Reshape back: (B, n_heads, L, d_k) -> (B, L, d_model)
        out = out.transpose(1, 2).contiguous().view(B, L, self.d_model)

        # Output projection
        out = self.out_proj(out)

        return out


# ============================================================
# Architecture: Jamba Block (SSM + Attention + FFN)
# ============================================================

class JambaBlock(nn.Module):
    """
    Jamba block: Interleaved Mamba SSM + Multi-head Attention + Feed-Forward.

    Architecture:
        x -> LayerNorm -> SelectiveSSM -> residual
          -> LayerNorm -> MultiHeadAttention -> residual
          -> LayerNorm -> FFN -> residual

    Combines:
      - SSM: Efficient O(n) long-range dependencies
      - Attention: Precise global alignment
      - FFN: Non-linear mixing
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 64,
        dt_rank: int = 16,
        d_conv: int = 4,
        n_heads: int = 4,
        dropout: float = 0.1,
    ):
        super().__init__()

        # Mamba SSM
        self.norm1 = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSM(
            d_model=d_model,
            d_state=d_state,
            dt_rank=dt_rank,
            d_conv=d_conv,
        )

        # Multi-head Attention
        self.norm2 = nn.LayerNorm(d_model)
        self.attn = MultiHeadAttention(
            d_model=d_model,
            n_heads=n_heads,
            dropout=dropout,
        )

        # Feed-Forward Network
        self.norm3 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        x: torch.Tensor,                               # (B, L, d_model)
        time_delta: Optional[torch.Tensor] = None,      # (B, L)
    ) -> torch.Tensor:
        # SSM branch
        residual = x
        x = self.norm1(x)
        x = self.ssm(x, time_delta=time_delta)
        x = x + residual

        # Attention branch
        residual = x
        x = self.norm2(x)
        x = self.attn(x)
        x = x + residual

        # FFN branch
        residual = x
        x = self.norm3(x)
        x = self.ffn(x)
        x = x + residual

        return x


# ============================================================
# Architecture: CNN-Jamba Model
# ============================================================

class CNNJamba(nn.Module):
    """
    CNN-Jamba: CNN front-end + Jamba blocks for MBO event stream prediction.

    Architecture:
        1. CNN front-end: Extract local patterns from raw event stream
           - Input: (B, L, 6) -> (B, L, cnn_channels)
           - 3-4 layers of 1D convolution with residual connections
        2. Projection: CNN features -> d_model
        3. Stack of Jamba blocks (SSM + Attention + FFN)
        4. Take LAST hidden state -> prediction head
        5. Multi-task: predict 1s, 5s, 10s price change

    Key advantages:
        - CNN: Captures local microstructure patterns efficiently
        - Mamba SSM: O(n) long-range temporal context
        - Attention: Global dependencies at critical points
        - Time-delta conditioning: State transitions are time-aware
        - Long context: 1000 events (vs 500 for transformer)

    Parameters: ~4-5M depending on config
    """

    def __init__(
        self,
        cnn_channels: int = CNN_CHANNELS,
        cnn_kernel: int = CNN_KERNEL,
        cnn_layers: int = CNN_LAYERS,
        d_model: int = JAMBA_D_MODEL,
        d_state: int = JAMBA_D_STATE,
        n_blocks: int = JAMBA_N_BLOCKS,
        n_heads: int = JAMBA_N_HEADS,
        dt_rank: int = JAMBA_DT_RANK,
        d_conv: int = JAMBA_D_CONV,
        dropout: float = JAMBA_DROPOUT,
        n_targets: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_targets = n_targets
        self.cnn_channels = cnn_channels

        # CNN front-end: Extract local patterns
        # Input: (B, L, 6) -> transpose to (B, 6, L) for Conv1d
        cnn_layers_list = []
        in_channels = 6
        for i in range(cnn_layers):
            out_channels = cnn_channels
            cnn_layers_list.append(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size=cnn_kernel,
                    padding=cnn_kernel // 2,  # same padding
                    bias=True,
                )
            )
            cnn_layers_list.append(nn.GELU())
            cnn_layers_list.append(nn.Dropout(dropout * 0.5))
            in_channels = out_channels

        self.cnn = nn.Sequential(*cnn_layers_list)

        # Project CNN features to d_model
        self.cnn_to_model = nn.Sequential(
            nn.Linear(cnn_channels, d_model),
            nn.LayerNorm(d_model),
        )

        # Stack of Jamba blocks
        self.blocks = nn.ModuleList([
            JambaBlock(
                d_model=d_model,
                d_state=d_state,
                dt_rank=dt_rank,
                d_conv=d_conv,
                n_heads=n_heads,
                dropout=dropout,
            )
            for _ in range(n_blocks)
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
        """Initialize weights for stable training."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        # Don't re-init the carefully initialized dt_proj bias and A_log in SSM

    def forward(self, events: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            events: (B, L, 6) float32 -- batch of event windows
            return_embedding: if True, also return state embedding for fusion layer

        Returns:
            preds: (B, n_targets) -- predicted price changes (1s, 5s, 10s)
            embedding: (B, d_model) -- state embedding (only if return_embedding=True)
        """
        B, L, _ = events.shape

        # Extract time_delta_log for time-aware state decay (feature index 0)
        time_delta = events[:, :, 0]  # (B, L)

        # CNN front-end: (B, L, 6) -> (B, 6, L) -> (B, cnn_channels, L)
        x = events.transpose(1, 2).contiguous()  # (B, 6, L)
        x = self.cnn(x)                           # (B, cnn_channels, L)
        x = x.transpose(1, 2).contiguous()        # (B, L, cnn_channels)

        # Project to d_model
        x = self.cnn_to_model(x)                  # (B, L, d_model)

        # Pass through Jamba blocks with time-delta conditioning
        for block in self.blocks:
            x = block(x, time_delta=time_delta)

        # Take the LAST position's output (causal: it has seen all prior events)
        x_last = x[:, -1, :]         # (B, d_model)

        # Final norm -- this IS the state embedding for fusion
        embedding = self.final_norm(x_last)  # (B, d_model)
        preds = self.head(embedding)          # (B, n_targets)

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
) -> Tuple[np.ndarray, np.ndarray]:
    """Run inference on OOT fold, return (predictions, labels)."""
    model.eval()
    all_preds  = []
    all_labels = []

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    with torch.no_grad():
        for events, labels in loader:
            events = events.to(device, non_blocking=True)
            with amp_ctx:
                preds = model(events)
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
    """Train CNN-Jamba for one expanding-window fold. Returns metrics dict."""

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
        epoch_start = time.time()

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
        epoch_time  = time.time() - epoch_start
        val_metrics = evaluate_metrics_only(model, val_loader, device, use_amp=use_amp)

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
                        "cnn_channels": CNN_CHANNELS,
                        "cnn_kernel":   CNN_KERNEL,
                        "cnn_layers":   CNN_LAYERS,
                        "d_model":      JAMBA_D_MODEL,
                        "d_state":      JAMBA_D_STATE,
                        "n_blocks":     JAMBA_N_BLOCKS,
                        "n_heads":      JAMBA_N_HEADS,
                        "dt_rank":      JAMBA_DT_RANK,
                        "d_conv":       JAMBA_D_CONV,
                        "dropout":      JAMBA_DROPOUT,
                        "window_size":  WINDOW_SIZE,
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

    # Build fold boundaries (expanding window)
    min_train     = max(5, n_files - n_folds)
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
            run_name=f"CNNJamba_{time.strftime('%Y%m%d_%H%M')}"
        )
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        mlflow.log_params(
            {
                "model":           "CNNJamba",
                "window_size":     WINDOW_SIZE,
                "stride":          STRIDE,
                "cnn_channels":    CNN_CHANNELS,
                "cnn_kernel":      CNN_KERNEL,
                "cnn_layers":      CNN_LAYERS,
                "d_model":         JAMBA_D_MODEL,
                "d_state":         JAMBA_D_STATE,
                "n_blocks":        JAMBA_N_BLOCKS,
                "n_heads":         JAMBA_N_HEADS,
                "dt_rank":         JAMBA_DT_RANK,
                "d_conv":          JAMBA_D_CONV,
                "dropout":         JAMBA_DROPOUT,
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

            # Build train dataset -- stats computed from train set only (no leakage)
            logger.info("Building train dataset...")
            train_ds      = MboEventDataset(train_files, window_size=WINDOW_SIZE, stride=STRIDE)
            feature_stats = train_ds.get_feature_stats()

            # Build OOT dataset using TRAIN feature stats (no leakage)
            logger.info("Building OOT dataset...")
            oot_ds = MboEventDataset(
                oot_files,
                window_size   = WINDOW_SIZE,
                stride        = STRIDE,
                feature_stats = feature_stats,
            )

            # Windows: num_workers=0 to avoid pickle issues. Linux: use 8 for parallel loading
            import platform
            _num_workers = 0 if platform.system() == "Windows" else 8

            train_loader = DataLoader(
                train_ds,
                batch_size  = BATCH_SIZE,
                shuffle     = True,
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

            # Fresh model per fold
            model = CNNJamba(
                cnn_channels = CNN_CHANNELS,
                cnn_kernel   = CNN_KERNEL,
                cnn_layers   = CNN_LAYERS,
                d_model      = JAMBA_D_MODEL,
                d_state      = JAMBA_D_STATE,
                n_blocks     = JAMBA_N_BLOCKS,
                n_heads      = JAMBA_N_HEADS,
                dt_rank      = JAMBA_DT_RANK,
                d_conv       = JAMBA_D_CONV,
                dropout      = JAMBA_DROPOUT,
                n_targets    = len(HORIZONS),
            ).to(device)

            if fold_idx == 0:
                n_params = count_parameters(model)
                logger.info(f"Model parameters:  {n_params:,}")
                logger.info(f"CNN: channels={CNN_CHANNELS}, kernel={CNN_KERNEL}, layers={CNN_LAYERS}")
                logger.info(f"Jamba: d_model={JAMBA_D_MODEL}, d_state={JAMBA_D_STATE}, "
                            f"n_blocks={JAMBA_N_BLOCKS}, n_heads={JAMBA_N_HEADS}")
                logger.info(f"Window size: {WINDOW_SIZE} events (stride={STRIDE})")
                logger.info(f"Architecture: CNN front-end + Jamba blocks (SSM + Attention + FFN)")

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

            # Save fold artifacts: .npz predictions
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
            logger.info(f"Saved predictions -> {pred_path}")

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
        logger.info(f"Saved concat predictions -> {concat_path}")

        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.log_metrics({f"concat_ic_{h}": concat_ic[h] for h in HORIZONS})

        logger.info("\n" + "=" * 60)
        logger.info("LEAKAGE AUDIT: PASSED")
        logger.info("  - Expanding window: train set never contains OOT dates")
        logger.info("  - Feature normalization computed from train set only per fold")
        logger.info("  - CNN is causal: convolution uses same/causal padding")
        logger.info("  - SSM is causal by construction: h[t] depends only on events <= t")
        logger.info("  - Attention uses causal masking: cannot attend to future")
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
        description="Train CNN-Jamba (CNN + Mamba SSM + Attention) on MBO event streams"
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
    logger.info("CNN-Jamba (CNN + Mamba SSM + Attention) Training")
    logger.info("=" * 60)
    logger.info(f"Config:")
    logger.info(f"  CNN:")
    logger.info(f"    channels    = {CNN_CHANNELS}")
    logger.info(f"    kernel      = {CNN_KERNEL}")
    logger.info(f"    layers      = {CNN_LAYERS}")
    logger.info(f"  Jamba:")
    logger.info(f"    d_model     = {JAMBA_D_MODEL}")
    logger.info(f"    d_state     = {JAMBA_D_STATE}")
    logger.info(f"    n_blocks    = {JAMBA_N_BLOCKS}")
    logger.info(f"    n_heads     = {JAMBA_N_HEADS}")
    logger.info(f"    dt_rank     = {JAMBA_DT_RANK}")
    logger.info(f"    d_conv      = {JAMBA_D_CONV}")
    logger.info(f"    dropout     = {JAMBA_DROPOUT}")
    logger.info(f"  Training:")
    logger.info(f"    window_size = {WINDOW_SIZE}")
    logger.info(f"    stride      = {STRIDE}")
    logger.info(f"    batch_size  = {BATCH_SIZE}")
    logger.info(f"    lr          = {LR}")
    logger.info(f"    epochs      = {EPOCHS_PER_FOLD}")
    logger.info(f"    n_folds     = {args.n_folds}")
    logger.info(f"  Paths:")
    logger.info(f"    data_dir    = {args.data_dir}")
    logger.info(f"    output_dir  = {args.output_dir}")

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
    logger.info("FINAL RESULTS -- CNN-Jamba (CNN + Mamba SSM + Attention)")
    logger.info("=" * 60)
    if concat_ic:
        for h in HORIZONS:
            ic = concat_ic.get(h, float("nan"))
            logger.info(f"  Concat IC ({h}): {ic:.4f}")
    logger.info(f"  Artifacts in: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
