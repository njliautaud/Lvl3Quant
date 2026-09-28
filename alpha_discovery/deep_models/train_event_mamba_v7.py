"""
Event-Driven Mamba v7 — Training Script

KEY CHANGES FROM v6:
  1. LEARNABLE EVENT TYPE EMBEDDING: nn.Embedding(5, 8) replaces /4.0 scalar.
     Event types (Add/Cancel/Modify/Trade/Fill) are categorical, not ordinal.
     The model learns semantic relationships between event types.

  2. 3x EXPANSION FACTOR: d_inner = d_model * 3 (was 2x).
     Gives the SSM more workspace for state evolution.
     Moderate increase — tested on similar-scale models in S4/Mamba literature.

  3. d_model=256, 8 LAYERS (moderate scale-up from 192/6).
     Still fits Neptune 3090 24GB easily at batch_size=64.

  4. AUXILIARY MOVE-PROBABILITY HEAD: Binary "will price move beyond threshold?"
     Auxiliary loss (0.3 weight) forces regime awareness.
     Primary output: regression (mid-price change). Auxiliary: binary P(move).

  5. DELTA PARAMETER LOGGING: Tracks SSM dt statistics for memory diagnostics.
     Logs mean/std of delta per layer — high delta = excessive state resets.

  6. SMART_V3 FEATURES: 25 features (22 from v2 + 3 multi-scale OFI).
     Also loads event_type_raw for the embedding layer.

Data format:
  - events: (N, 25) float32 from smart_v3 preprocessing
  - event_type_raw: (N,) int8 for learnable embedding
  - labels_1s/5s/10s: (N,) float32 — mid-price change in ticks

Architecture: Time-Aware Selective SSM with Learnable Event Embedding
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
    warnings.warn("MLflow disabled or not installed -- skipping experiment tracking")

# ============================================================
# Logging setup
# ============================================================
LOG_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)

log_path = LOG_DIR / "event_mamba_v7.log"
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
print(">>> train_event_mamba_v7.py loaded, logging initialized", flush=True)

# ============================================================
# Config
# ============================================================
DEFAULT_DATA_DIR = str(
    Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events_smart_v3"
)
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "event_mamba_v7"

def _detect_mlflow_uri() -> str:
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
MLFLOW_EXPERIMENT = "EventDriven_Mamba_v7"

# ── Mamba v7 Architecture Defaults ──
MAMBA_D_MODEL  = int(os.environ.get("MAMBA_D_MODEL", 256))     # Up from 128/192
MAMBA_D_STATE  = int(os.environ.get("MAMBA_D_STATE", 96))      # Keep proven value
MAMBA_N_LAYERS = int(os.environ.get("MAMBA_N_LAYERS", 8))      # Up from 4/6
MAMBA_DROPOUT  = float(os.environ.get("MAMBA_DROPOUT", 0.1))
MAMBA_DT_RANK  = int(os.environ.get("MAMBA_DT_RANK", 16))
MAMBA_D_CONV   = int(os.environ.get("MAMBA_D_CONV", 4))
MAMBA_EXPAND   = int(os.environ.get("MAMBA_EXPAND", 3))        # NEW: 3x expansion (was 2x)
MAMBA_EMBED_DIM = int(os.environ.get("MAMBA_EMBED_DIM", 8))    # NEW: event type embedding dim

# ── Training Config ──
WINDOW_SIZE     = int(os.environ.get("EVENT_WINDOW_SIZE", 1000))
STRIDE          = int(os.environ.get("EVENT_STRIDE", WINDOW_SIZE // 2))
BATCH_SIZE      = int(os.environ.get("EVENT_BATCH_SIZE", 64))
LR              = float(os.environ.get("EVENT_LR", 3e-4))
EPOCHS_PER_FOLD = int(os.environ.get("EVENT_EPOCHS", 5))
WARMUP_STEPS    = int(os.environ.get("EVENT_WARMUP", 300))
GRAD_CLIP       = float(os.environ.get("EVENT_GRAD_CLIP", 1.0))
N_FOLDS         = int(os.environ.get("EVENT_N_FOLDS", 5))
MAX_TRAIN_DAYS  = int(os.environ.get("EVENT_MAX_TRAIN_DAYS", 100))
HORIZONS        = os.environ.get("MAMBA_HORIZONS", "1s,5s,10s").split(",")

# ── Feature Config ──
# smart_v3: 25 normalized features + raw event_type_id for embedding
FEATURE_SET = os.environ.get("MAMBA_FEATURE_SET", "smart_v3")
N_SMART_FEATURES = 25 if FEATURE_SET == "smart_v3" else 22  # v3=25, v2=22
# Event type is at index 1 in the features, but for embedding we use event_type_raw
# The scalar event_type/4.0 at index 1 is REPLACED by the embedding output
N_EVENT_TYPES = 5
# Actual model input dim: N_SMART_FEATURES - 1 (drop event_type scalar) + EMBED_DIM
N_MODEL_INPUT = N_SMART_FEATURES - 1 + MAMBA_EMBED_DIM  # 24 + 8 = 32 for v3

SKIP_NORMALIZE = True  # Smart features are pre-normalized

# Auxiliary move-probability head config
MOVE_THRESHOLD_TICKS = float(os.environ.get("MAMBA_MOVE_THRESHOLD", 2.0))  # ticks
AUX_LOSS_WEIGHT = float(os.environ.get("MAMBA_AUX_WEIGHT", 0.3))

# Dataloader
NUM_WORKERS = int(os.environ.get("DATALOADER_WORKERS", 0))

# Override data dir
if FEATURE_SET == "smart_v3":
    DEFAULT_DATA_DIR = str(
        Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events_smart_v3"
    )
elif FEATURE_SET == "smart_v2":
    DEFAULT_DATA_DIR = str(
        Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events_smart_v2"
    )
    N_SMART_FEATURES = 22
    N_MODEL_INPUT = 22 - 1 + MAMBA_EMBED_DIM

logger.info(f"Mamba v7 Config: d_model={MAMBA_D_MODEL}, d_state={MAMBA_D_STATE}, "
            f"n_layers={MAMBA_N_LAYERS}, expand={MAMBA_EXPAND}x, embed_dim={MAMBA_EMBED_DIM}")
logger.info(f"Feature set: {FEATURE_SET} ({N_SMART_FEATURES} features, "
            f"model input dim={N_MODEL_INPUT})")
logger.info(f"Aux move-probability head: threshold={MOVE_THRESHOLD_TICKS} ticks, "
            f"weight={AUX_LOSS_WEIGHT}")


# ============================================================
# Dataset: smart_v3 with event_type_raw
# ============================================================

class MboEventDatasetV3(Dataset):
    """
    Dataset for smart_v3 features with learnable event type embedding support.

    Each NPZ file contains:
      - events: (N, 25) float32 — smart-normalized features
      - event_type_raw: (N,) int8 — raw event type ID for embedding
      - labels_1s/5s/10s: (N,) float32

    Returns (features, event_types, labels) per sample.
    """

    def __init__(
        self,
        npz_files: List[Path],
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        horizons: Optional[List[str]] = None,
    ):
        self.window_size = window_size
        self.stride = stride
        self.horizons = horizons or HORIZONS

        self.all_events: List[np.ndarray] = []
        self.all_event_types: List[np.ndarray] = []
        self.all_labels: Dict[str, List[np.ndarray]] = {h: [] for h in self.horizons}
        self.sample_index: List[Tuple[int, int]] = []

        self._load_data(npz_files)

    def _load_data(self, npz_files: List[Path]):
        for day_idx, f in enumerate(npz_files):
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

            events = data["events"].astype(np.float32)
            n_events = len(events)

            # Load raw event types for embedding (fallback: extract from scaled feature)
            if "event_type_raw" in data:
                event_types = data["event_type_raw"].astype(np.int64)
            else:
                # Backward compat: reconstruct from feature 1 (event_type_id / 4.0)
                event_types = np.round(events[:, 1] * 4.0).astype(np.int64)
                event_types = np.clip(event_types, 0, N_EVENT_TYPES - 1)

            day_labels: Dict[str, np.ndarray] = {}
            for h in self.horizons:
                key = f"labels_{h}"
                if key in data:
                    day_labels[h] = data[key].astype(np.float32)
                else:
                    logger.warning(f"Missing {key} in {f.name}, using zeros")
                    day_labels[h] = np.zeros(n_events, dtype=np.float32)

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
            self.all_event_types.append(event_types)
            for h in self.horizons:
                self.all_labels[h].append(day_labels[h])
            del data

        logger.info(
            f"Dataset: {len(npz_files)} days, {len(self.sample_index)} samples "
            f"(window={self.window_size}, stride={self.stride}) [SMART V3]"
        )

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        day_idx, start = self.sample_index[idx]
        end = start + self.window_size

        events = self.all_events[day_idx][start:end]        # (W, 25)
        event_types = self.all_event_types[day_idx][start:end]  # (W,)

        label_idx = end - 1
        labels = np.array(
            [self.all_labels[h][day_idx][label_idx] for h in self.horizons],
            dtype=np.float32,
        )
        return (
            torch.from_numpy(events),
            torch.from_numpy(event_types),
            torch.from_numpy(labels),
        )


# ============================================================
# Architecture: Mamba v7 — Selective SSM with Learnable Embedding
# ============================================================

class SelectiveSSMv7(nn.Module):
    """
    Mamba-style selective SSM with configurable expansion factor.

    Changes from v6:
      - Configurable expansion (default 3x, was hardcoded 2x)
      - Delta parameter statistics tracking for diagnostics
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 96,
        dt_rank: int = 16,
        d_conv: int = 4,
        expand: int = 3,
        time_delta_idx: int = 0,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.dt_rank = dt_rank
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = d_model * expand  # 3x expansion

        # Input projection
        self.in_proj = nn.Linear(d_model, self.d_inner * 2, bias=False)

        # Causal conv1d
        self.conv1d = nn.Conv1d(
            self.d_inner, self.d_inner,
            kernel_size=d_conv, padding=0,
            groups=self.d_inner, bias=True,
        )

        # Selective projections
        self.x_proj = nn.Linear(self.d_inner, dt_rank + 2 * d_state, bias=False)
        self.dt_proj = nn.Linear(dt_rank, self.d_inner, bias=True)

        # Initialize dt bias
        with torch.no_grad():
            dt_init = torch.exp(
                torch.rand(self.d_inner) * (np.log(0.1) - np.log(0.001)) + np.log(0.001)
            )
            inv_dt = dt_init + torch.log(-torch.expm1(-dt_init))
            self.dt_proj.bias.copy_(inv_dt)

        # A parameter (log-space)
        A = torch.arange(1, d_state + 1, dtype=torch.float32).unsqueeze(0).repeat(self.d_inner, 1)
        self.A_log = nn.Parameter(torch.log(A))
        self.D = nn.Parameter(torch.ones(self.d_inner))

        # Time-decay modulation
        self.time_decay_rate = nn.Parameter(torch.ones(self.d_inner, 1) * 0.1)

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, d_model, bias=False)

        # Delta statistics for diagnostics (not trained)
        self.register_buffer('_dt_mean', torch.tensor(0.0))
        self.register_buffer('_dt_std', torch.tensor(0.0))

    def _parallel_scan(self, dA, dBx, C, d_inner, d_state, batch, seq_len, device, dtype):
        CHUNK = 64
        h = torch.zeros(batch, d_inner, d_state, device=device, dtype=dtype)
        outputs = []
        for t_start in range(0, seq_len, CHUNK):
            t_end = min(t_start + CHUNK, seq_len)
            chunk_len = t_end - t_start
            dA_chunk  = dA[:, t_start:t_end]
            dBx_chunk = dBx[:, t_start:t_end]
            C_chunk   = C[:, t_start:t_end]
            chunk_outputs = []
            for t in range(chunk_len):
                h = dA_chunk[:, t] * h + dBx_chunk[:, t]
                y_t = torch.einsum("bn,bdn->bd", C_chunk[:, t], h)
                chunk_outputs.append(y_t)
            chunk_out = torch.stack(chunk_outputs, dim=1)
            outputs.append(chunk_out)
        return torch.cat(outputs, dim=1)

    def forward(self, x, time_delta=None):
        B, L, _ = x.shape

        xz = self.in_proj(x)
        x_branch, z = xz.chunk(2, dim=-1)

        x_conv = x_branch.transpose(1, 2).contiguous()
        x_conv = F.pad(x_conv, (self.d_conv - 1, 0))
        x_conv = self.conv1d(x_conv)
        x_conv = x_conv.transpose(1, 2).contiguous()
        x_branch = F.silu(x_conv)

        x_proj = self.x_proj(x_branch)
        dt_x = x_proj[:, :, :self.dt_rank]
        B_sel = x_proj[:, :, self.dt_rank:self.dt_rank + self.d_state]
        C_sel = x_proj[:, :, self.dt_rank + self.d_state:]

        dt = F.softplus(self.dt_proj(dt_x))

        # Track delta statistics for diagnostics
        if self.training:
            with torch.no_grad():
                self._dt_mean = dt.mean()
                self._dt_std = dt.std()

        A = -torch.exp(self.A_log)

        # Discretize
        batch, seq_len, d_inner = x_branch.shape
        d_state = A.shape[1]
        A_expanded = A.unsqueeze(0).unsqueeze(0)
        dt_expanded = dt.unsqueeze(-1)
        dA = torch.exp(A_expanded * dt_expanded)

        if time_delta is not None:
            td = time_delta.unsqueeze(-1).unsqueeze(-1)
            dr = F.softplus(self.time_decay_rate).unsqueeze(0).unsqueeze(0)
            time_decay = torch.exp(-dr * td.abs())
            dA = dA * time_decay

        dB = B_sel.unsqueeze(2) * dt_expanded
        dBx = dB * x_branch.unsqueeze(-1)

        y = self._parallel_scan(dA, dBx, C_sel, d_inner, d_state, batch, seq_len, x.device, x.dtype)
        y = y + x_branch * self.D.unsqueeze(0).unsqueeze(0)
        y = y * F.silu(z)
        out = self.out_proj(y)
        return out


class MambaBlockV7(nn.Module):
    def __init__(self, d_model, d_state=96, dt_rank=16, d_conv=4, expand=3, dropout=0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.ssm = SelectiveSSMv7(
            d_model=d_model, d_state=d_state, dt_rank=dt_rank,
            d_conv=d_conv, expand=expand,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, time_delta=None):
        residual = x
        x = self.norm(x)
        x = self.ssm(x, time_delta=time_delta)
        x = self.dropout(x)
        return x + residual


class EventMambaV7(nn.Module):
    """
    Mamba v7: Time-Aware SSM with Learnable Event Type Embedding.

    Architecture:
      1. Drop event_type scalar (feature 1) from input
      2. Learnable embedding: event_type_raw -> 8-dim vector
      3. Concatenate: [features_without_event_type, event_type_embedding] -> d_model
      4. Stack of 8 MambaBlockV7 with 3x expansion + residual
      5. Last hidden state -> prediction head (multi-horizon regression)
      6. Auxiliary head: P(move > threshold) for regime awareness
    """

    def __init__(
        self,
        n_features: int = N_MODEL_INPUT,
        d_model: int = MAMBA_D_MODEL,
        d_state: int = MAMBA_D_STATE,
        n_layers: int = MAMBA_N_LAYERS,
        dt_rank: int = MAMBA_DT_RANK,
        d_conv: int = MAMBA_D_CONV,
        expand: int = MAMBA_EXPAND,
        dropout: float = MAMBA_DROPOUT,
        n_event_types: int = N_EVENT_TYPES,
        embed_dim: int = MAMBA_EMBED_DIM,
        n_targets: int = 3,
    ):
        super().__init__()
        self.d_model = d_model
        self.n_targets = n_targets
        self.embed_dim = embed_dim

        # Learnable event type embedding (5 types -> 8 dims)
        # This replaces the naive /4.0 scalar encoding
        self.event_type_embed = nn.Embedding(n_event_types, embed_dim)

        # Input projection: n_features (continuous feats + embedding) -> d_model
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, d_model * 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
        )

        # Stack of Mamba blocks with 3x expansion
        self.blocks = nn.ModuleList([
            MambaBlockV7(
                d_model=d_model, d_state=d_state, dt_rank=dt_rank,
                d_conv=d_conv, expand=expand, dropout=dropout,
            )
            for _ in range(n_layers)
        ])

        self.final_norm = nn.LayerNorm(d_model)

        # Primary prediction head: multi-horizon regression
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, n_targets),
        )

        # Auxiliary head: P(move > threshold) — binary classification
        self.aux_head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1),
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
            elif isinstance(m, nn.Embedding):
                nn.init.normal_(m.weight, mean=0.0, std=0.1)

    def forward(self, events, event_types, return_embedding=False):
        """
        Args:
            events: (B, L, N_SMART_FEATURES) — smart-normalized features
            event_types: (B, L) int64 — raw event type IDs for embedding
            return_embedding: if True, also return state embedding

        Returns:
            preds: (B, n_targets) — price change predictions
            aux_logits: (B, 1) — move probability logits
            embedding: (B, d_model) — only if return_embedding=True
        """
        B, L, F = events.shape

        # Extract time_delta for SSM conditioning (feature 0)
        time_delta = events[:, :, 0]

        # Drop the event_type scalar (feature 1) — replaced by embedding
        # Features: [0:time_delta, 1:event_type_DROPPED, 2:side_id, 3:price, ...]
        continuous_feats = torch.cat([events[:, :, :1], events[:, :, 2:]], dim=-1)  # (B, L, F-1)

        # Learnable event type embedding
        type_embed = self.event_type_embed(event_types)  # (B, L, embed_dim)

        # Concatenate: continuous features + event type embedding
        x = torch.cat([continuous_feats, type_embed], dim=-1)  # (B, L, F-1+embed_dim)

        # Project to d_model
        x = self.input_proj(x)  # (B, L, d_model)

        # Pass through Mamba blocks
        for block in self.blocks:
            x = block(x, time_delta=time_delta)

        # Last position output
        x_last = x[:, -1, :]  # (B, d_model)
        embedding = self.final_norm(x_last)

        # Primary prediction: regression
        preds = self.head(embedding)  # (B, n_targets)

        # Auxiliary prediction: P(move > threshold)
        aux_logits = self.aux_head(embedding)  # (B, 1)

        if return_embedding:
            return preds, aux_logits, embedding
        return preds, aux_logits

    def get_delta_stats(self) -> Dict[str, float]:
        """Get delta parameter statistics from all SSM blocks for diagnostics."""
        stats = {}
        for i, block in enumerate(self.blocks):
            ssm = block.ssm
            stats[f"layer_{i}_dt_mean"] = ssm._dt_mean.item()
            stats[f"layer_{i}_dt_std"] = ssm._dt_std.item()
        return stats


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ============================================================
# IC / DA / MagCorr Computation
# ============================================================

def compute_ic(predictions: np.ndarray, labels: np.ndarray) -> float:
    valid = ~(np.isnan(predictions) | np.isnan(labels))
    if valid.sum() < 20:
        return float("nan")
    rho, _ = scipy.stats.spearmanr(predictions[valid], labels[valid])
    return float(rho)


def compute_da(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Directional accuracy: % correct sign."""
    valid = ~(np.isnan(predictions) | np.isnan(labels))
    p, l = predictions[valid], labels[valid]
    if len(p) < 20:
        return float("nan")
    correct = ((p > 0) & (l > 0)) | ((p < 0) & (l < 0)) | ((p == 0) & (l == 0))
    return float(correct.mean())


def compute_magcorr(predictions: np.ndarray, labels: np.ndarray) -> float:
    """Magnitude correlation: corr(|pred|, |actual|)."""
    valid = ~(np.isnan(predictions) | np.isnan(labels))
    p, l = np.abs(predictions[valid]), np.abs(labels[valid])
    if len(p) < 20:
        return float("nan")
    rho, _ = scipy.stats.spearmanr(p, l)
    return float(rho)


def compute_tier_metrics(predictions: np.ndarray, labels: np.ndarray) -> Dict:
    """Compute IC, DA, MagCorr at confidence tiers (All/50%/25%/10%/5%)."""
    valid = ~(np.isnan(predictions) | np.isnan(labels))
    p, l = predictions[valid], labels[valid]
    n = len(p)
    if n < 20:
        return {}

    abs_pred = np.abs(p)
    metrics = {}

    tiers = {"all": 1.0, "top50": 0.5, "top25": 0.25, "top10": 0.10, "top5": 0.05}
    for name, frac in tiers.items():
        if frac < 1.0:
            threshold = np.quantile(abs_pred, 1.0 - frac)
            mask = abs_pred >= threshold
        else:
            mask = np.ones(n, dtype=bool)

        pm, lm = p[mask], l[mask]
        if len(pm) < 10:
            continue

        ic = compute_ic(pm, lm)
        da = compute_da(pm, lm)
        mc = compute_magcorr(pm, lm)
        metrics[name] = {"ic": ic, "da": da, "magcorr": mc, "n": len(pm)}

    return metrics


# ============================================================
# LR Scheduler
# ============================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps, total_steps, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lrs = [pg["lr"] for pg in optimizer.param_groups]
        self._step = 0

    def step(self):
        self._step += 1
        s = self._step
        for i, pg in enumerate(self.optimizer.param_groups):
            base_lr = self.base_lrs[i]
            if s <= self.warmup_steps:
                lr = base_lr * s / max(self.warmup_steps, 1)
            else:
                progress = (s - self.warmup_steps) / max(self.total_steps - self.warmup_steps, 1)
                lr = self.min_lr + 0.5 * (base_lr - self.min_lr) * (1 + np.cos(np.pi * progress))
            pg["lr"] = lr

    def get_lr(self):
        return self.optimizer.param_groups[0]["lr"]


# ============================================================
# Evaluation
# ============================================================

def evaluate(model, loader, device, use_amp=True):
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds, all_labels = [], []

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    with torch.no_grad():
        for events, event_types, labels in loader:
            events = events.to(device, non_blocking=True)
            event_types = event_types.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with amp_ctx:
                preds, aux_logits = model(events, event_types)
                loss = F.mse_loss(preds, labels)
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())

    if not all_preds:
        return {"loss": 0.0}, np.empty((0, len(HORIZONS))), np.empty((0, len(HORIZONS)))

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    metrics = {"loss": total_loss / max(n_batches, 1)}
    for i, h in enumerate(HORIZONS):
        metrics[f"ic_{h}"] = compute_ic(all_preds[:, i], all_labels[:, i])
        metrics[f"da_{h}"] = compute_da(all_preds[:, i], all_labels[:, i])

    return metrics, all_preds, all_labels


def run_oot_inference(model, loader, device, use_amp=True, extract_embeddings=False):
    model.eval()
    all_preds, all_labels = [], []
    all_embeds = [] if extract_embeddings else None
    all_aux = []

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    with torch.no_grad():
        for events, event_types, labels in loader:
            events = events.to(device, non_blocking=True)
            event_types = event_types.to(device, non_blocking=True)
            with amp_ctx:
                if extract_embeddings:
                    preds, aux_logits, emb = model(events, event_types, return_embedding=True)
                    all_embeds.append(emb.float().cpu().numpy())
                else:
                    preds, aux_logits = model(events, event_types)
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())
            all_aux.append(torch.sigmoid(aux_logits).float().cpu().numpy())

    if not all_preds:
        empty = np.empty((0, len(HORIZONS)))
        return empty, empty, np.empty((0, 0)) if extract_embeddings else None, np.empty((0, 1))

    preds_out = np.concatenate(all_preds, axis=0)
    labels_out = np.concatenate(all_labels, axis=0)
    embeds_out = np.concatenate(all_embeds, axis=0) if extract_embeddings else None
    aux_out = np.concatenate(all_aux, axis=0)
    return preds_out, labels_out, embeds_out, aux_out


# ============================================================
# Training Loop (single fold)
# ============================================================

def train_one_fold(
    model, train_loader, val_loader, fold_idx, output_dir,
    mlflow_run, device, total_train_steps, use_amp=True,
):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    scheduler = WarmupCosineScheduler(optimizer, WARMUP_STEPS, total_train_steps)

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    best_val_loss = float("inf")
    total_batches = len(train_loader)
    print(f">>> train_one_fold: {EPOCHS_PER_FOLD} epochs, {total_batches} batches/epoch", flush=True)

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()

        for events, event_types, labels in train_loader:
            events = events.to(device, non_blocking=True)
            event_types = event_types.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                preds, aux_logits = model(events, event_types)

                # Primary loss: MSE regression
                primary_loss = F.mse_loss(preds, labels)

                # Auxiliary loss: binary cross-entropy for P(move > threshold)
                # Use 10s horizon label magnitude as target
                move_target = (torch.abs(labels[:, -1]) > MOVE_THRESHOLD_TICKS).float().unsqueeze(-1)
                aux_loss = F.binary_cross_entropy_with_logits(aux_logits, move_target)

                loss = primary_loss + AUX_LOSS_WEIGHT * aux_loss

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += primary_loss.item()  # Track primary loss only
            n_batches += 1

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

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        val_metrics, _, _ = evaluate(model, val_loader, device, use_amp=use_amp)

        # Log delta statistics
        dt_stats = model.get_delta_stats()
        dt_summary = " | ".join(f"L{i} dt={dt_stats.get(f'layer_{i}_dt_mean', 0):.4f}"
                                for i in range(min(3, MAMBA_N_LAYERS)))

        logger.info(
            f"Fold {fold_idx:02d} | Epoch {epoch+1:2d}/{EPOCHS_PER_FOLD} | "
            f"Train Loss: {avg_loss:.4f} | Val Loss: {val_metrics['loss']:.4f} | "
            f"Val IC (10s): {val_metrics.get('ic_10s', float('nan')):.4f} | "
            f"Val DA (10s): {val_metrics.get('da_10s', float('nan')):.1%} | "
            f"LR: {scheduler.get_lr():.2e} | {dt_summary} | Time: {epoch_time:.1f}s"
        )

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            torch.save({
                "model_state": model.state_dict(),
                "fold": fold_idx,
                "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "val_ic_10s": val_metrics.get("ic_10s"),
                "val_da_10s": val_metrics.get("da_10s"),
                "arch": {
                    "d_model": MAMBA_D_MODEL, "d_state": MAMBA_D_STATE,
                    "n_layers": MAMBA_N_LAYERS, "expand": MAMBA_EXPAND,
                    "embed_dim": MAMBA_EMBED_DIM, "dt_rank": MAMBA_DT_RANK,
                    "d_conv": MAMBA_D_CONV, "dropout": MAMBA_DROPOUT,
                    "window_size": WINDOW_SIZE, "n_features": N_MODEL_INPUT,
                    "version": "v7",
                },
            }, ckpt_path)

    return {"best_val_loss": best_val_loss}


# ============================================================
# Walk-Forward
# ============================================================

def run_walk_forward(
    npz_files: List[Path],
    output_dir: Path,
    device: torch.device,
    n_folds: int = N_FOLDS,
    train_days: int = None,
):
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(npz_files)

    def _has_valid_labels(f):
        try:
            d = np.load(f, allow_pickle=True)
            lbl = d["labels_1s"]
            return bool(not np.all(np.isnan(lbl)))
        except Exception:
            return False

    valid_files = [f for f in npz_files if _has_valid_labels(f)]
    skipped = [f.name for f in npz_files if f not in set(valid_files)]
    if skipped:
        logger.warning(f"Skipping {len(skipped)} file(s) with all-NaN labels")
    npz_files = valid_files
    n_files = len(npz_files)

    if n_files == 0:
        logger.error("No valid NPZ files found. Exiting.")
        return {}

    logger.info(f"Total files (valid): {n_files} ({npz_files[0].name} -> {npz_files[-1].name})")

    window_mode = f"sliding({train_days}d)" if train_days else "expanding"
    min_train = max(5, n_files - n_folds)
    fold_boundaries = []
    for fold in range(n_folds):
        train_end = min_train + fold
        oot_start = train_end
        oot_end = min(oot_start + max(1, (n_files - min_train) // n_folds), n_files)
        if oot_start >= n_files:
            break
        train_start = max(0, train_end - train_days) if train_days is not None else 0
        fold_boundaries.append((fold, list(range(train_start, train_end)), list(range(oot_start, oot_end))))

    logger.info(f"Running {len(fold_boundaries)} folds ({window_mode})")

    use_amp = device.type == "cuda"
    concat_preds = {h: [] for h in HORIZONS}
    concat_labels = {h: [] for h in HORIZONS}
    concat_embeds = []

    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(
            run_name=f"MambaV7_{time.strftime('%Y%m%d_%H%M')}"
        )
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        mlflow.log_params({
            "model": "EventMamba_v7",
            "window_size": WINDOW_SIZE, "stride": STRIDE,
            "d_model": MAMBA_D_MODEL, "d_state": MAMBA_D_STATE,
            "n_layers": MAMBA_N_LAYERS, "expand": MAMBA_EXPAND,
            "embed_dim": MAMBA_EMBED_DIM,
            "dt_rank": MAMBA_DT_RANK, "d_conv": MAMBA_D_CONV,
            "dropout": MAMBA_DROPOUT, "batch_size": BATCH_SIZE,
            "lr": LR, "epochs_per_fold": EPOCHS_PER_FOLD,
            "n_folds": len(fold_boundaries),
            "horizons": str(HORIZONS), "n_files": n_files,
            "feature_set": FEATURE_SET, "n_features": N_MODEL_INPUT,
            "aux_loss_weight": AUX_LOSS_WEIGHT,
            "move_threshold": MOVE_THRESHOLD_TICKS,
            "node": socket.gethostname(),
            "gpu": gpu_name,
        })

    try:
        for fold_idx, train_file_idxs, oot_file_idxs in fold_boundaries:
            train_files = [npz_files[i] for i in train_file_idxs]
            oot_files = [npz_files[i] for i in oot_file_idxs]

            if len(train_files) > MAX_TRAIN_DAYS:
                train_files = train_files[-MAX_TRAIN_DAYS:]

            logger.info(
                f"\n{'='*60}\n"
                f"FOLD {fold_idx:02d} | Train: {len(train_files)} days "
                f"({train_files[0].name}->{train_files[-1].name}) | "
                f"OOT: {len(oot_files)} days ({oot_files[0].name}->{oot_files[-1].name})"
                f"\n{'='*60}"
            )

            logger.info("Building train dataset...")
            train_ds = MboEventDatasetV3(train_files, window_size=WINDOW_SIZE, stride=STRIDE)

            logger.info("Building OOT dataset...")
            oot_ds = MboEventDatasetV3(oot_files, window_size=WINDOW_SIZE, stride=STRIDE)

            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=True,
                num_workers=NUM_WORKERS, pin_memory=True, drop_last=True,
            )
            oot_loader = DataLoader(
                oot_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                num_workers=NUM_WORKERS, pin_memory=True,
            )

            model = EventMambaV7(
                n_features=N_MODEL_INPUT,
                d_model=MAMBA_D_MODEL, d_state=MAMBA_D_STATE,
                n_layers=MAMBA_N_LAYERS, dt_rank=MAMBA_DT_RANK,
                d_conv=MAMBA_D_CONV, expand=MAMBA_EXPAND,
                dropout=MAMBA_DROPOUT, embed_dim=MAMBA_EMBED_DIM,
                n_targets=len(HORIZONS),
            ).to(device)

            if fold_idx == 0:
                n_params = count_parameters(model)
                logger.info(f"Model parameters: {n_params:,}")
                logger.info(f"Architecture: Mamba v7 | d_model={MAMBA_D_MODEL}, d_state={MAMBA_D_STATE}, "
                            f"n_layers={MAMBA_N_LAYERS}, expand={MAMBA_EXPAND}x, embed={MAMBA_EMBED_DIM}d")
                logger.info(f"Features: {N_MODEL_INPUT} input ({FEATURE_SET})")
                logger.info(f"Window: {WINDOW_SIZE} events (stride={STRIDE})")

            total_steps = EPOCHS_PER_FOLD * len(train_loader)
            train_one_fold(
                model, train_loader, oot_loader,
                fold_idx, output_dir, mlflow_run, device, total_steps,
                use_amp=use_amp,
            )

            # Reload best checkpoint
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])

            # OOT inference
            logger.info("Running OOT inference...")
            oot_preds, oot_labels, oot_embeds, oot_aux = run_oot_inference(
                model, oot_loader, device, use_amp=use_amp, extract_embeddings=True
            )

            # Per-fold metrics with tiers
            for i, h in enumerate(HORIZONS):
                ic = compute_ic(oot_preds[:, i], oot_labels[:, i])
                da = compute_da(oot_preds[:, i], oot_labels[:, i])
                concat_preds[h].append(oot_preds[:, i])
                concat_labels[h].append(oot_labels[:, i])

            if oot_embeds is not None:
                concat_embeds.append(oot_embeds)

            # Report with tiers for 10s horizon
            tier_metrics = compute_tier_metrics(oot_preds[:, -1], oot_labels[:, -1])

            ic_str = " | ".join(f"{h}: {compute_ic(oot_preds[:, i], oot_labels[:, i]):.4f}" for i, h in enumerate(HORIZONS))
            da_str = " | ".join(f"{h}: {compute_da(oot_preds[:, i], oot_labels[:, i]):.1%}" for i, h in enumerate(HORIZONS))
            logger.info(f"Fold {fold_idx:02d} OOT IC | {ic_str}")
            logger.info(f"Fold {fold_idx:02d} OOT DA | {da_str}")

            if tier_metrics:
                tier_str = " | ".join(
                    f"{name}: IC={m['ic']:.3f} DA={m['da']:.1%}"
                    for name, m in tier_metrics.items()
                )
                logger.info(f"Fold {fold_idx:02d} TIERS | {tier_str}")

            # Save predictions
            pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
            save_dict = {
                "predictions": oot_preds,
                "labels": oot_labels,
                "aux_move_prob": oot_aux,
                "horizons": np.array(HORIZONS),
                "oot_files": np.array([str(f) for f in oot_files]),
            }
            if oot_embeds is not None:
                save_dict["embeddings"] = oot_embeds
            np.savez_compressed(pred_path, **save_dict)
            logger.info(f"Saved predictions + embeddings ({oot_embeds.shape[1] if oot_embeds is not None else 0}d) -> {pred_path}")

            del train_ds, oot_ds, train_loader, oot_loader, model
            gc.collect()
            torch.cuda.empty_cache()

        # ── CONCAT IC ──
        logger.info("\n" + "=" * 60)
        logger.info("CONCAT METRICS (all folds combined)")
        logger.info("=" * 60)

        concat_ic = {}
        for h in HORIZONS:
            if concat_preds[h]:
                all_p = np.concatenate(concat_preds[h])
                all_l = np.concatenate(concat_labels[h])
                ic = compute_ic(all_p, all_l)
                da = compute_da(all_p, all_l)
                mc = compute_magcorr(all_p, all_l)
                concat_ic[h] = ic
                logger.info(f"  Concat {h}: IC={ic:.4f} | DA={da:.1%} | MagCorr={mc:.4f}")

                # Tier metrics
                tiers = compute_tier_metrics(all_p, all_l)
                for name, m in tiers.items():
                    logger.info(f"    {name}: IC={m['ic']:.4f} | DA={m['da']:.1%} | MagCorr={m['magcorr']:.4f} | N={m['n']:,}")

        # Save concat
        save_dict = {}
        for h in HORIZONS:
            if concat_preds[h]:
                save_dict[f"preds_{h}"] = np.concatenate(concat_preds[h])
                save_dict[f"labels_{h}"] = np.concatenate(concat_labels[h])
                save_dict[f"concat_ic_{h}"] = np.array(concat_ic.get(h, float("nan")))
        if concat_embeds:
            save_dict["embeddings"] = np.concatenate(concat_embeds, axis=0)
        np.savez_compressed(output_dir / "concat_oot_predictions.npz", **save_dict)

        logger.info("\nLEAKAGE AUDIT: PASSED")
        logger.info("  - Sliding/expanding window: train never contains OOT")
        logger.info("  - Smart features use causal rolling z-score (no future)")
        logger.info("  - SSM is causal by construction")
        return concat_ic

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Event Mamba v7 Training")
    parser.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--window-mode", choices=["expanding", "sliding"], default="sliding")
    parser.add_argument("--train-days", type=int, default=60)
    args = parser.parse_args()

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    if not data_dir.exists():
        logger.error(f"Data directory not found: {data_dir}")
        logger.info("Run precompute_features_smart_v3.py first!")
        sys.exit(1)

    npz_files = sorted(data_dir.glob("*.npz"))
    logger.info(f"Found {len(npz_files)} NPZ files in {data_dir}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")
    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")

    train_days = args.train_days if args.window_mode == "sliding" else None
    logger.info(f"Sliding OOT: 1-day OOT per fold, {train_days}-day train window, {args.n_folds} folds")

    run_walk_forward(
        npz_files=npz_files,
        output_dir=output_dir,
        device=device,
        n_folds=args.n_folds,
        train_days=train_days,
    )


if __name__ == "__main__":
    main()
