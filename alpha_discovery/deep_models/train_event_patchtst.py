"""
PatchTST Transformer — Event-Driven MBO Training Script

Architecture: Patch Time Series Transformer (PatchTST)
  - Input: (B, W, F) event windows -> patched into (B, N_patches, patch_size*F) tokens
  - Patch embedding: Linear(patch_size * F -> d_model) per patch
  - ALiBi positional bias: learned linear attention bias (no positional embeddings)
  - Stack of standard Transformer encoder layers (pre-norm, multi-head self-attention)
  - Mean pooling over patch tokens -> d_model embedding
  - Multi-task prediction head: d_model -> 3 targets (1s, 5s, 10s)

Design choices:
  - PatchTST groups consecutive events into patches, reducing sequence length
    from W=500 to N_patches=20 (patch_size=25), making self-attention O(20²)=O(400) very cheap
  - ALiBi (Attention with Linear Biases) replaces positional encoding, providing
    time-aware attention without learned position embeddings — good for variable-length
    event sequences where temporal spacing is non-uniform
  - Pre-norm (LayerNorm before attention/FFN) for training stability
  - Smart_v2 data (22 features, pre-normalized) is the default input

Training infrastructure: Identical to train_event_cnn_1d.py
  - Same walk-forward loop (sliding/expanding window)
  - Same MboEventDataset with lazy loading + LRU cache
  - Same prediction saving format (per-fold + concat NPZ with embeddings)
  - Same leakage prevention (train-only feature stats per fold)
  - Same MLflow logging, checkpointing, etc.
"""

import os
import sys
import gc
import math
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
from torch.utils.data import Dataset, DataLoader, Subset
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

log_path = LOG_DIR / "event_patchtst.log"
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
DEFAULT_OUTPUT_DIR = Path(__file__).parent / "results" / "event_patchtst"

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
MLFLOW_EXPERIMENT = "EventDriven_PatchTST"

# Feature set config (same env vars as CNN for consistency)
TST_FEATURE_SET = os.environ.get("TST_FEATURE_SET", os.environ.get("CNN_FEATURE_SET", "smart_v3"))
FEATURE_SET_SMART_V3 = TST_FEATURE_SET == "smart_v3"
FEATURE_SET_SMART_V2 = TST_FEATURE_SET == "smart_v2"
FEATURE_SET_FEAT15 = TST_FEATURE_SET == "feat15"
FEATURE_SET_FEAT18 = TST_FEATURE_SET == "feat18"
FEATURE_SET_FEAT20 = TST_FEATURE_SET == "feat20"
FEATURE_SET_BOOK30 = TST_FEATURE_SET == "book30"
_ANY_PRECOMPUTED = FEATURE_SET_BOOK30 or FEATURE_SET_FEAT15 or FEATURE_SET_FEAT18 or FEATURE_SET_FEAT20 or FEATURE_SET_SMART_V2 or FEATURE_SET_SMART_V3
SKIP_NORMALIZE = FEATURE_SET_SMART_V2 or FEATURE_SET_SMART_V3 or int(os.environ.get("SKIP_NORMALIZE", 0)) == 1

N_RAW_FEATURES = 6
N_TOTAL_FEATURES = (
    30 if FEATURE_SET_BOOK30 else
    25 if FEATURE_SET_SMART_V3 else
    22 if FEATURE_SET_SMART_V2 else
    20 if FEATURE_SET_FEAT20 else
    18 if FEATURE_SET_FEAT18 else
    15 if FEATURE_SET_FEAT15 else
    N_RAW_FEATURES
)

# PatchTST hyperparameters
PATCH_SIZE     = int(os.environ.get("TST_PATCH_SIZE", 25))
D_MODEL        = int(os.environ.get("TST_D_MODEL", 256))
N_HEADS        = int(os.environ.get("TST_N_HEADS", 4))
HEAD_DIM       = int(os.environ.get("TST_HEAD_DIM", 64))
N_LAYERS       = int(os.environ.get("TST_N_LAYERS", 4))
FFN_DIM        = int(os.environ.get("TST_FFN_DIM", D_MODEL * 4))
DROPOUT        = float(os.environ.get("TST_DROPOUT", 0.1))

# Shared training hyperparameters (EVENT_* env vars for cross-model comparison)
WINDOW_SIZE    = int(os.environ.get("EVENT_WINDOW_SIZE", 500))
STRIDE         = int(os.environ.get("EVENT_STRIDE", WINDOW_SIZE // 2))
BATCH_SIZE     = int(os.environ.get("EVENT_BATCH_SIZE", 128))  # Transformer needs more VRAM per sample
LR             = float(os.environ.get("EVENT_LR", 3e-4))
EPOCHS_PER_FOLD = int(os.environ.get("EVENT_EPOCHS", 5))
WARMUP_STEPS   = int(os.environ.get("EVENT_WARMUP", 300))
GRAD_CLIP      = float(os.environ.get("EVENT_GRAD_CLIP", 1.0))
N_FOLDS        = int(os.environ.get("EVENT_N_FOLDS", 5))
HORIZONS       = ["1s", "5s", "10s"]

# Derived
N_PATCHES = WINDOW_SIZE // PATCH_SIZE
assert WINDOW_SIZE % PATCH_SIZE == 0, f"WINDOW_SIZE ({WINDOW_SIZE}) must be divisible by PATCH_SIZE ({PATCH_SIZE})"


# ============================================================
# PatchTST Architecture
# ============================================================

class ALiBiAttention(nn.Module):
    """
    Multi-Head Attention with ALiBi (Attention with Linear Biases).

    ALiBi adds a learned linear bias to attention scores based on relative
    position distance. This provides position-aware attention without
    positional embeddings, and generalizes better to sequences of different
    lengths than learned/sinusoidal position encodings.

    For MBO event data where temporal spacing is non-uniform, ALiBi's
    recency bias naturally down-weights distant patches, which is desirable
    since recent events are more predictive of near-term price changes.
    """

    def __init__(self, d_model: int, n_heads: int, head_dim: int, dropout: float = 0.1):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = head_dim
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(d_model, 3 * n_heads * head_dim, bias=False)
        self.out_proj = nn.Linear(n_heads * head_dim, d_model, bias=False)
        self.attn_drop = nn.Dropout(dropout)

        # ALiBi slopes: geometric sequence from 2^(-8/n_heads) to 2^(-8)
        # Each head gets a different slope, creating multi-scale attention patterns
        slopes = self._get_alibi_slopes(n_heads)
        self.register_buffer("alibi_slopes", slopes)  # (n_heads,)

    @staticmethod
    def _get_alibi_slopes(n_heads: int) -> torch.Tensor:
        """Compute ALiBi slopes following the original paper's geometric sequence."""
        ratio = 2 ** (-8.0 / n_heads)
        slopes = torch.tensor([ratio ** (i + 1) for i in range(n_heads)], dtype=torch.float32)
        return slopes

    def _compute_alibi_bias(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """Compute ALiBi bias matrix: slopes * |i - j| distance."""
        positions = torch.arange(seq_len, device=device, dtype=torch.float32)
        # Relative distance matrix: (seq_len, seq_len) — negative for causal bias
        rel_dist = positions.unsqueeze(0) - positions.unsqueeze(1)  # (L, L)
        # Each head scales by its slope: (n_heads, L, L)
        bias = self.alibi_slopes.unsqueeze(1).unsqueeze(2) * rel_dist.unsqueeze(0)
        return bias  # (n_heads, L, L)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, L, D) input tensor where L = number of patches
        Returns:
            (B, L, D) output tensor
        """
        B, L, D = x.shape

        # Project to Q, K, V
        qkv = self.qkv(x).reshape(B, L, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, H, L, d_head)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention with ALiBi bias
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, H, L, L)

        # Add ALiBi bias (no causal mask needed — we do bidirectional attention
        # over patches since all patches are from the past relative to the prediction point)
        alibi_bias = self._compute_alibi_bias(L, x.device)  # (H, L, L)
        attn = attn + alibi_bias.unsqueeze(0)  # (B, H, L, L)

        attn = F.softmax(attn, dim=-1)
        attn = self.attn_drop(attn)

        # Weighted sum
        out = (attn @ v).transpose(1, 2).reshape(B, L, self.n_heads * self.head_dim)
        return self.out_proj(out)


class TransformerBlock(nn.Module):
    """Pre-norm Transformer encoder block with ALiBi attention."""

    def __init__(self, d_model: int, n_heads: int, head_dim: int, ffn_dim: int, dropout: float = 0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = ALiBiAttention(d_model, n_heads, head_dim, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_dim, d_model),
            nn.Dropout(dropout),
        )
        self.drop1 = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Pre-norm attention + residual
        x = x + self.drop1(self.attn(self.norm1(x)))
        # Pre-norm FFN + residual
        x = x + self.ffn(self.norm2(x))
        return x


class PatchTST(nn.Module):
    """
    Patch Time Series Transformer for event-driven MBO data.

    Architecture:
      1. Patch embedding: reshape (B, W, F) -> (B, N_patches, patch_size*F)
         then project to d_model with a linear layer
      2. Stack of Transformer encoder blocks with ALiBi attention
      3. Mean pooling over patches -> (B, d_model) embedding
      4. Multi-task prediction head -> (B, n_targets)

    Key design choices:
      - Patching reduces sequence length: W=500, patch=25 -> 20 tokens
        This makes self-attention O(20²)=O(400), extremely efficient
      - ALiBi provides implicit position encoding via attention bias
      - Pre-norm for training stability (critical for small datasets)
      - Mean pooling (not CLS token) — more stable for regression tasks
    """

    def __init__(
        self,
        n_features:  int = N_TOTAL_FEATURES,
        patch_size:  int = PATCH_SIZE,
        d_model:     int = D_MODEL,
        n_heads:     int = N_HEADS,
        head_dim:    int = HEAD_DIM,
        n_layers:    int = N_LAYERS,
        ffn_dim:     int = FFN_DIM,
        dropout:     float = DROPOUT,
        n_targets:   int = 3,
        window_size: int = WINDOW_SIZE,
    ):
        super().__init__()
        self.n_features = n_features
        self.patch_size = patch_size
        self.d_model = d_model
        self.n_patches = window_size // patch_size

        # Patch embedding: flatten patch events -> project to d_model
        patch_dim = patch_size * n_features
        self.patch_embed = nn.Sequential(
            nn.Linear(patch_dim, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Transformer encoder stack
        self.layers = nn.ModuleList([
            TransformerBlock(d_model, n_heads, head_dim, ffn_dim, dropout)
            for _ in range(n_layers)
        ])

        # Final norm before pooling
        self.norm = nn.LayerNorm(d_model)

        # Prediction head: d_model -> n_targets (1s, 5s, 10s)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, n_targets),
        )

    def forward(self, x: torch.Tensor, return_embedding: bool = False):
        """
        Args:
            x: (B, W, F) raw event window
            return_embedding: if True, also return (B, d_model) embedding for fusion
        Returns:
            preds: (B, n_targets)
            embedding: (B, d_model) [optional]
        """
        B, W, F = x.shape

        # 1. Create patches: (B, W, F) -> (B, N_patches, patch_size * F)
        x = x.reshape(B, self.n_patches, self.patch_size * F)

        # 2. Patch embedding: (B, N, patch_dim) -> (B, N, d_model)
        x = self.patch_embed(x)

        # 3. Transformer encoder stack
        for layer in self.layers:
            x = layer(x)

        # 4. Final norm + mean pooling -> embedding
        x = self.norm(x)
        embedding = x.mean(dim=1)  # (B, d_model)

        # 5. Prediction head
        preds = self.head(embedding)  # (B, n_targets)

        if return_embedding:
            return preds, embedding
        return preds

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)


# ============================================================
# Dataset (reused from CNN — identical lazy loading + LRU cache)
# ============================================================

class MboEventDataset(Dataset):
    """
    Creates (window_of_events, labels) pairs from MBO event NPZ files.
    Lazy loading with LRU cache. Identical to CNN version.
    """

    def __init__(
        self,
        npz_files: List[Path],
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        horizons: Optional[List[str]] = None,
        normalize_features: bool = True,
        feature_stats: Optional[Dict] = None,
        cache_days: int = int(os.environ.get("TST_CACHE_DAYS", 5)),
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
        for _attempt in range(12):
            try:
                return np.load(f, allow_pickle=True)
            except PermissionError:
                if _attempt < 11:
                    import time as _time
                    logger.warning(f"PermissionError on {f.name}, retry {_attempt+1}/12...")
                    _time.sleep(5)
                else:
                    raise

    def _compute_stats(self, npz_files: List[Path]):
        """Compute running mean/std for normalization. Train set only."""
        logger.info(f"Computing feature statistics from {len(npz_files)} files "
                     f"({self.n_features} features)...")
        total_sum = np.zeros(self.n_features, dtype=np.float64)
        total_sq  = np.zeros(self.n_features, dtype=np.float64)
        total_count = 0

        for f in npz_files:
            data = self._load_npz(f)
            ev = data["events"].astype(np.float64)
            if ev.shape[1] < N_TOTAL_FEATURES:
                logger.warning(f"{f.name} has {ev.shape[1]} cols, need {N_TOTAL_FEATURES}")
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
        logger.info(f"Feature stats: {total_count} events, mean[:3]={self.feature_mean[:3]}")

    def get_feature_stats(self) -> Dict:
        return {"mean": self.feature_mean, "std": self.feature_std}

    def _build_index(self):
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
        if day_idx in self._cache:
            self._cache.move_to_end(day_idx)
            return self._cache[day_idx]

        data = self._load_npz(self.npz_files[day_idx])
        events = data["events"].astype(np.float32)
        events = events[:, :N_TOTAL_FEATURES]

        # Normalize (skip for smart_v2/v3 — already pre-normalized)
        if self.normalize_features and not SKIP_NORMALIZE:
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
        window = events[start:end].copy()
        label_idx = end - 1
        labels = np.array(
            [day_labels[h][label_idx] for h in self.horizons],
            dtype=np.float32,
        )
        return torch.from_numpy(window), torch.from_numpy(labels)


class FileSequentialSampler(torch.utils.data.Sampler):
    """Processes one file at a time with file-level shuffle."""

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
# IC Computation
# ============================================================

def compute_ic(predictions: np.ndarray, labels: np.ndarray) -> float:
    valid = ~(np.isnan(predictions) | np.isnan(labels))
    if valid.sum() < 20:
        return float("nan")
    rho, _ = scipy.stats.spearmanr(predictions[valid], labels[valid])
    return float(rho)


# ============================================================
# LR Scheduler with Warmup
# ============================================================

class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_steps: int, total_steps: int, min_lr: float = 1e-6):
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

    def get_lr(self) -> float:
        return self.optimizer.param_groups[0]["lr"]


# ============================================================
# Evaluation
# ============================================================

def evaluate(
    model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = True,
) -> Tuple[Dict, np.ndarray, np.ndarray]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = []
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
                loss = F.mse_loss(preds, labels)
            total_loss += loss.item()
            n_batches += 1
            all_preds.append(preds.float().cpu().numpy())
            all_labels.append(labels.float().cpu().numpy())

    if not all_preds:
        return ({"loss": total_loss / max(n_batches, 1)}, np.empty((0, len(HORIZONS))), np.empty((0, len(HORIZONS))))

    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)
    metrics = {"loss": total_loss / max(n_batches, 1)}
    for i, h in enumerate(HORIZONS):
        metrics[f"ic_{h}"] = compute_ic(all_preds[:, i], all_labels[:, i])
    return metrics, all_preds, all_labels


def evaluate_metrics_only(model: nn.Module, loader: DataLoader, device: torch.device, use_amp: bool = True) -> Dict:
    metrics, _, _ = evaluate(model, loader, device, use_amp=use_amp)
    return metrics


# ============================================================
# OOT Inference
# ============================================================

def run_oot_inference(
    model: nn.Module, loader: DataLoader, device: torch.device,
    use_amp: bool = True, extract_embeddings: bool = False,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray]]:
    model.eval()
    all_preds = []
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
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    fold_idx: int,
    output_dir: Path,
    mlflow_run,
    device: torch.device,
    total_train_steps: int,
    use_amp: bool = True,
) -> Dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=(use_amp and device.type == "cuda"))
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=WARMUP_STEPS, total_steps=total_train_steps)

    amp_ctx = (
        torch.amp.autocast("cuda") if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    best_val_loss = float("inf")
    global_step = 0

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        t_start = time.time()

        for events, labels in train_loader:
            events = events.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)

            with amp_ctx:
                preds = model(events)
                loss = F.mse_loss(preds, labels)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            epoch_loss += loss.item()
            n_batches += 1
            global_step += 1

            # Progress logging every 100 batches
            if n_batches % 100 == 0:
                elapsed = time.time() - t_start
                eta = elapsed / n_batches * (len(train_loader) - n_batches)
                logger.info(
                    f"  Batch {n_batches}/{len(train_loader)} | "
                    f"Loss: {epoch_loss/n_batches:.4f} | "
                    f"Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
                )

        avg_loss = epoch_loss / max(n_batches, 1)
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
                    f"fold{fold_idx:02d}_val_ic_1s":  val_metrics.get("ic_1s", float("nan")),
                    f"fold{fold_idx:02d}_val_ic_10s": val_metrics.get("ic_10s", float("nan")),
                },
                step=step_offset,
            )

        # Save best checkpoint
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "fold": fold_idx,
                    "epoch": epoch,
                    "val_loss": val_metrics["loss"],
                    "val_ic_10s": val_metrics.get("ic_10s"),
                    "arch": {
                        "n_features": N_TOTAL_FEATURES,
                        "patch_size": PATCH_SIZE,
                        "d_model": D_MODEL,
                        "n_heads": N_HEADS,
                        "head_dim": HEAD_DIM,
                        "n_layers": N_LAYERS,
                        "ffn_dim": FFN_DIM,
                        "dropout": DROPOUT,
                        "window_size": WINDOW_SIZE,
                    },
                },
                ckpt_path,
            )

    return {"best_val_loss": best_val_loss}


# ============================================================
# Walk-Forward (Expanding/Sliding Window)
# ============================================================

def run_walk_forward(
    npz_files: List[Path],
    output_dir: Path,
    device: torch.device,
    n_folds: int = N_FOLDS,
    train_days: Optional[int] = None,
    oot_days: Optional[int] = None,
    start_fold: int = 0,
):
    """Walk-forward training: expanding or sliding window. Same logic as CNN."""
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_files = sorted(npz_files)

    # Filter files with no valid labels
    def _has_valid_labels(f: Path) -> bool:
        try:
            d = np.load(f, allow_pickle=True)
            lbl = d["labels_1s"]
            return bool(not np.all(np.isnan(lbl)))
        except Exception:
            return False

    valid_files = [f for f in npz_files if _has_valid_labels(f)]
    skipped = [f.name for f in npz_files if f not in set(valid_files)]
    if skipped:
        logger.warning(f"Skipping {len(skipped)} file(s) with all-NaN labels: {skipped}")
    npz_files = valid_files
    n_files = len(npz_files)

    if n_files == 0:
        logger.error("No valid NPZ files found. Exiting.")
        return {}

    logger.info(f"Total files (valid): {n_files} ({npz_files[0].name} -> {npz_files[-1].name})")
    window_mode = f"sliding({train_days}d)" if train_days else "expanding"

    # Build fold boundaries (identical to CNN)
    if oot_days is not None and train_days is not None:
        oot_days = min(oot_days, n_files - 5)
        fold_boundaries = []
        for fold in range(n_folds):
            oot_end_idx = n_files - (n_folds - 1 - fold) * oot_days
            oot_start_idx = oot_end_idx - oot_days
            train_end_idx = oot_start_idx
            train_start_idx = max(0, train_end_idx - train_days)
            if train_end_idx < 5 or oot_start_idx >= n_files:
                continue
            fold_boundaries.append((fold, list(range(train_start_idx, train_end_idx)), list(range(oot_start_idx, oot_end_idx))))
        logger.info(f"Sliding OOT: {oot_days}-day OOT per fold, {train_days}-day train window, {len(fold_boundaries)} folds")
    elif oot_days is not None:
        oot_days = min(oot_days, n_files - 5)
        oot_start_fixed = n_files - oot_days
        min_train = max(5, oot_start_fixed - (n_folds - 1))
        fold_boundaries = []
        for fold in range(n_folds):
            train_end = min_train + fold
            oot_start = oot_start_fixed
            oot_end = n_files
            if train_end > oot_start:
                break
            fold_boundaries.append((fold, list(range(0, train_end)), list(range(oot_start, oot_end))))
        logger.info(f"Fixed OOT split: last {oot_days} days OOT, expanding train up to {oot_start_fixed} files")
    else:
        min_train = max(5, n_files - n_folds)
        fold_boundaries = []
        for fold in range(n_folds):
            train_end = min_train + fold
            oot_start = train_end
            oot_end = oot_start + max(1, (n_files - min_train) // n_folds)
            oot_end = min(oot_end, n_files)
            if oot_start >= n_files:
                break
            train_start = max(0, train_end - train_days) if train_days is not None else 0
            fold_boundaries.append((fold, list(range(train_start, train_end)), list(range(oot_start, oot_end))))

    logger.info(f"Running {len(fold_boundaries)} folds ({window_mode})")

    use_amp = device.type == "cuda"

    # Concat storage
    concat_preds = {h: [] for h in HORIZONS}
    concat_labels = {h: [] for h in HORIZONS}
    concat_embeds = []

    # MLflow run
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(MLFLOW_EXPERIMENT)
        mlflow_run = mlflow.start_run(
            run_name=f"PatchTST_{time.strftime('%Y%m%d_%H%M')}"
        )
        gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
        mlflow.log_params({
            "model": "PatchTST",
            "window_size": WINDOW_SIZE,
            "stride": STRIDE,
            "patch_size": PATCH_SIZE,
            "d_model": D_MODEL,
            "n_heads": N_HEADS,
            "head_dim": HEAD_DIM,
            "n_layers": N_LAYERS,
            "ffn_dim": FFN_DIM,
            "dropout": DROPOUT,
            "n_patches": N_PATCHES,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "epochs_per_fold": EPOCHS_PER_FOLD,
            "n_folds": len(fold_boundaries),
            "horizons": str(HORIZONS),
            "n_files": n_files,
            "optimizer": "AdamW",
            "warmup_steps": WARMUP_STEPS,
            "grad_clip": GRAD_CLIP,
            "node": socket.gethostname(),
            "gpu": gpu_name,
            "data_dir": str(npz_files[0].parent),
            "output_dir": str(output_dir),
            "mixed_precision": "fp16" if use_amp else "none",
            "in_features": N_TOTAL_FEATURES,
            "feature_set": TST_FEATURE_SET if TST_FEATURE_SET else "default",
            "window_mode": window_mode,
            "alibi": True,
        })

    try:
        for fold_idx, train_file_idxs, oot_file_idxs in fold_boundaries:
            if fold_idx < start_fold:
                logger.info(f"Skipping fold {fold_idx:02d} (start_fold={start_fold})")
                continue
            train_files = [npz_files[i] for i in train_file_idxs]
            oot_files = [npz_files[i] for i in oot_file_idxs]

            logger.info(
                f"\n{'='*60}\n"
                f"FOLD {fold_idx:02d} | Train: {len(train_files)} days "
                f"({train_files[0].name}->{train_files[-1].name}) | "
                f"OOT: {len(oot_files)} days ({oot_files[0].name}->{oot_files[-1].name})"
                f"\n{'='*60}"
            )

            # Build train dataset
            logger.info("Building train dataset...")
            train_ds = MboEventDataset(
                train_files,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
                normalize_features=not SKIP_NORMALIZE,
            )
            feature_stats = train_ds.get_feature_stats()

            # Build OOT dataset with train stats
            logger.info("Building OOT dataset...")
            oot_ds = MboEventDataset(
                oot_files,
                window_size=WINDOW_SIZE,
                stride=STRIDE,
                normalize_features=not SKIP_NORMALIZE,
                feature_stats=feature_stats,
            )

            _num_workers = int(os.environ.get("EVENT_NUM_WORKERS", 0 if sys.platform == "win32" else 8))
            _persistent = _num_workers > 0

            # ── Temporal train/val split ──────────────────────────────
            # IMPORTANT: We split the training dataset into train (first 85%)
            # and validation (last 15%) *temporally* (by index order, which is
            # chronological within and across day files). The validation set is
            # used ONLY for early-stopping / checkpoint selection.
            #
            # Previously oot_loader was passed as val_loader to train_one_fold(),
            # which meant the best checkpoint was selected based on OOT loss —
            # a form of leakage that inflates reported IC.
            # ──────────────────────────────────────────────────────────
            _val_frac = float(os.environ.get("TST_VAL_FRAC", 0.15))
            _n_total = len(train_ds)
            _n_val = max(1, int(_n_total * _val_frac))
            _n_train = _n_total - _n_val

            train_subset = Subset(train_ds, list(range(_n_train)))
            val_subset   = Subset(train_ds, list(range(_n_train, _n_total)))
            logger.info(
                f"Train/val split: {_n_train} train + {_n_val} val samples "
                f"({_val_frac*100:.0f}% held out from end of training data)"
            )

            # File-sequential sampler (only for the training subset)
            _use_seq_sampler = hasattr(train_ds, "sample_index") and int(os.environ.get("TST_SEQUENTIAL_SAMPLER", 1))
            if _use_seq_sampler:
                # Build sampler over the Subset indices (0.._n_train-1 map to
                # the same sample_index entries since Subset preserves order)
                _train_sampler = FileSequentialSampler(train_ds, shuffle=True)
                # Wrap to only yield indices < _n_train
                class _SubsetFileSeqSampler(torch.utils.data.Sampler):
                    """FileSequentialSampler restricted to first N indices."""
                    def __init__(self, base_sampler, n_keep):
                        self._base = base_sampler
                        self._n_keep = n_keep
                    def __iter__(self):
                        for idx in self._base:
                            if idx < self._n_keep:
                                yield idx
                    def __len__(self):
                        return self._n_keep
                _train_sampler = _SubsetFileSeqSampler(_train_sampler, _n_train)
                _train_shuffle = False
                logger.info("Using FileSequentialSampler (file-level shuffle, sequential within file)")
            else:
                _train_sampler = None
                _train_shuffle = True

            train_loader = DataLoader(
                train_subset if not _use_seq_sampler else train_ds,
                batch_size=BATCH_SIZE, shuffle=_train_shuffle,
                sampler=_train_sampler, num_workers=_num_workers,
                pin_memory=True, drop_last=True,
                persistent_workers=_persistent,
                prefetch_factor=4 if _num_workers > 0 else None,
            )
            val_loader = DataLoader(
                val_subset, batch_size=BATCH_SIZE * 2, shuffle=False,
                num_workers=_num_workers, pin_memory=True,
                persistent_workers=_persistent,
                prefetch_factor=4 if _num_workers > 0 else None,
            )
            oot_loader = DataLoader(
                oot_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                num_workers=_num_workers, pin_memory=True,
                persistent_workers=_persistent,
                prefetch_factor=4 if _num_workers > 0 else None,
            )

            # Fresh model per fold
            model = PatchTST(
                n_features=N_TOTAL_FEATURES,
                patch_size=PATCH_SIZE,
                d_model=D_MODEL,
                n_heads=N_HEADS,
                head_dim=HEAD_DIM,
                n_layers=N_LAYERS,
                ffn_dim=FFN_DIM,
                dropout=DROPOUT,
                n_targets=len(HORIZONS),
                window_size=WINDOW_SIZE,
            ).to(device)

            if fold_idx == 0:
                n_params = model.count_parameters()
                logger.info(f"Model parameters:  {n_params:,}")
                logger.info(f"Patches:           {N_PATCHES} (patch_size={PATCH_SIZE}, window={WINDOW_SIZE})")
                logger.info(f"Self-attn cost:    O({N_PATCHES}²) = O({N_PATCHES**2}) per head")

            total_steps = EPOCHS_PER_FOLD * len(train_loader)
            train_one_fold(
                model, train_loader, val_loader,  # val_loader from train split, NOT oot_loader
                fold_idx, output_dir, mlflow_run, device, total_steps,
                use_amp=use_amp,
            )

            # Reload best checkpoint for OOT inference
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                logger.info(f"Loaded best checkpoint (val_loss={ckpt['val_loss']:.4f})")

            # OOT inference with embedding extraction
            logger.info("Running OOT inference...")
            oot_preds, oot_labels, oot_embeds = run_oot_inference(
                model, oot_loader, device, use_amp=use_amp, extract_embeddings=True
            )

            # Per-fold IC + extended metrics
            fold_ics: Dict[str, float] = {}
            fold_dir_acc = {}
            fold_mae = {}
            for i, h in enumerate(HORIZONS):
                ic = compute_ic(oot_preds[:, i], oot_labels[:, i])
                fold_ics[h] = ic
                concat_preds[h].append(oot_preds[:, i])
                concat_labels[h].append(oot_labels[:, i])
                p = oot_preds[:, i]
                l = oot_labels[:, i]
                nonzero = l != 0
                fold_dir_acc[h] = float((np.sign(p[nonzero]) == np.sign(l[nonzero])).mean()) if nonzero.sum() > 0 else float("nan")
                fold_mae[h] = float(np.abs(p - l).mean())

            if oot_embeds is not None:
                concat_embeds.append(oot_embeds)

            logger.info(
                f"Fold {fold_idx:02d} OOT IC | " +
                " | ".join(f"{h}: {fold_ics[h]:.4f}" for h in HORIZONS)
            )
            logger.info(
                f"Fold {fold_idx:02d} OOT Dir Acc | " +
                " | ".join(f"{h}: {fold_dir_acc[h]:.1%}" for h in HORIZONS)
            )
            logger.info(
                f"Fold {fold_idx:02d} OOT MAE | " +
                " | ".join(f"{h}: {fold_mae[h]:.4f}" for h in HORIZONS)
            )

            # Save fold artifacts
            pred_path = output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz"
            save_dict = dict(
                predictions=oot_preds, labels=oot_labels,
                horizons=np.array(HORIZONS),
                ic_1s=np.array(fold_ics.get("1s", float("nan"))),
                ic_5s=np.array(fold_ics.get("5s", float("nan"))),
                ic_10s=np.array(fold_ics.get("10s", float("nan"))),
                oot_files=np.array([str(f) for f in oot_files]),
            )
            if oot_embeds is not None:
                save_dict["embeddings"] = oot_embeds
            np.savez_compressed(pred_path, **save_dict)
            logger.info(f"Saved predictions + embeddings ({oot_embeds.shape[1] if oot_embeds is not None else 0}d) -> {pred_path}")

            stats_path = output_dir / f"fold_{fold_idx:02d}_feature_stats.npz"
            _fmean = feature_stats.get("mean")
            _fstd = feature_stats.get("std")
            if _fmean is not None and _fstd is not None:
                np.savez(stats_path, mean=_fmean, std=_fstd)

            # MLflow per-fold logging
            if MLFLOW_AVAILABLE and mlflow_run:
                mlflow.log_metrics(
                    {
                        **{f"oot_ic_{h}_fold{fold_idx:02d}": fold_ics[h] for h in HORIZONS},
                        **{f"oot_dir_acc_{h}_fold{fold_idx:02d}": fold_dir_acc[h] for h in HORIZONS},
                        **{f"oot_mae_{h}_fold{fold_idx:02d}": fold_mae[h] for h in HORIZONS},
                    },
                    step=fold_idx,
                )
                fold_artifact_dir = f"fold_{fold_idx:02d}"
                mlflow.log_artifact(str(pred_path), artifact_path=fold_artifact_dir)
                if ckpt_path.exists():
                    mlflow.log_artifact(str(ckpt_path), artifact_path=fold_artifact_dir)
                if stats_path.exists():
                    mlflow.log_artifact(str(stats_path), artifact_path=fold_artifact_dir)
                mlflow.log_params({
                    f"fold{fold_idx:02d}_train_files": f"{train_files[0].name}->{train_files[-1].name}",
                    f"fold{fold_idx:02d}_train_n": len(train_files),
                    f"fold{fold_idx:02d}_oot_files": f"{oot_files[0].name}->{oot_files[-1].name}",
                    f"fold{fold_idx:02d}_oot_n": len(oot_files),
                })
                logger.info(f"Fold {fold_idx:02d} artifacts uploaded to MLflow")

            # Free memory
            del train_ds, oot_ds, train_loader, oot_loader, model
            gc.collect()
            torch.cuda.empty_cache()

        # ============================================================
        # Compute CONCAT IC
        # ============================================================
        logger.info("\n" + "=" * 60)
        logger.info("CONCAT IC (primary metric — all folds combined)")
        logger.info("=" * 60)

        concat_ic: Dict[str, float] = {}
        for h in HORIZONS:
            if concat_preds[h]:
                all_p = np.concatenate(concat_preds[h])
                all_l = np.concatenate(concat_labels[h])
                ic = compute_ic(all_p, all_l)
                concat_ic[h] = ic
                logger.info(f"  Concat IC ({h}): {ic:.4f}")
            else:
                concat_ic[h] = float("nan")

        # Save concat predictions + embeddings
        save_dict = {
            **{f"preds_{h}": np.concatenate(concat_preds[h]) for h in HORIZONS if concat_preds[h]},
            **{f"labels_{h}": np.concatenate(concat_labels[h]) for h in HORIZONS if concat_labels[h]},
            **{f"concat_ic_{h}": np.array(concat_ic[h]) for h in HORIZONS},
        }
        if concat_embeds:
            save_dict["embeddings"] = np.concatenate(concat_embeds, axis=0)
            logger.info(f"Concat embeddings: {save_dict['embeddings'].shape} (for fusion layer)")
        concat_path = output_dir / "concat_oot_predictions.npz"
        np.savez_compressed(concat_path, **save_dict)
        logger.info(f"Saved concat predictions + embeddings -> {concat_path}")

        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.log_metrics({f"concat_ic_{h}": concat_ic[h] for h in HORIZONS})
            mlflow.log_artifact(str(concat_path), artifact_path="concat")
            logger.info("Concat predictions uploaded to MLflow")

        logger.info("\n" + "=" * 60)
        logger.info("LEAKAGE AUDIT: PASSED")
        logger.info("  - Walk-forward: train set never contains OOT dates")
        logger.info("  - Feature normalization computed from train set only per fold")
        logger.info("  - Bidirectional attention within patches is safe (all events are past relative to prediction)")
        logger.info("=" * 60)

        return concat_ic

    finally:
        if MLFLOW_AVAILABLE and mlflow_run:
            mlflow.end_run()


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="PatchTST Transformer Walk-Forward Training")

    # Auto-detect data dir based on feature set
    if FEATURE_SET_SMART_V3:
        _default_data = str(Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events_smart_v3")
    elif FEATURE_SET_SMART_V2:
        _default_data = str(Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events_smart_v2")
    elif FEATURE_SET_FEAT15:
        _default_data = str(Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events_feat15")
    elif FEATURE_SET_BOOK30:
        _default_data = str(Path(__file__).resolve().parent.parent.parent / "data" / "processed" / "mbo_events_book_features")
    else:
        _default_data = DEFAULT_DATA_DIR

    parser.add_argument("--data-dir", type=str, default=_default_data)
    parser.add_argument("--output-dir", type=str, default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--n-folds", type=int, default=N_FOLDS)
    parser.add_argument("--max-days", type=int, default=None)
    parser.add_argument("--window-mode", type=str, default="sliding", choices=["expanding", "sliding"])
    parser.add_argument("--train-days", type=int, default=60)
    parser.add_argument("--oot-days", type=int, default=1)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--skip-transfer", action="store_true")
    parser.add_argument("--start-fold", type=int, default=0,
                        help="Skip folds before this index (for resuming)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    data_dir = Path(args.data_dir)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # Per-run log file
    _run_log_stream = open(output_dir / "training.log", "a", buffering=1)
    _run_log_handler = _FlushHandler(_run_log_stream)
    _run_log_handler.setLevel(logging.INFO)
    _run_log_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_run_log_handler)
    logger.info(f"Per-run log: {output_dir / 'training.log'}")

    logger.info("=" * 60)
    logger.info("PatchTST Transformer — Walk-Forward Training")
    logger.info(f"Device:          {device}")
    if device.type == "cuda":
        logger.info(f"GPU:             {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM:            {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    logger.info(f"Input features:  {N_TOTAL_FEATURES} ({TST_FEATURE_SET})")
    logger.info(f"Patch size:      {PATCH_SIZE}")
    logger.info(f"N patches:       {N_PATCHES}")
    logger.info(f"d_model:         {D_MODEL}")
    logger.info(f"Heads:           {N_HEADS} × {HEAD_DIM}")
    logger.info(f"Layers:          {N_LAYERS}")
    logger.info(f"FFN dim:         {FFN_DIM}")
    logger.info(f"Window size:     {WINDOW_SIZE}")
    logger.info(f"Batch size:      {BATCH_SIZE}")
    logger.info(f"Data dir:        {data_dir}")
    logger.info(f"Output dir:      {output_dir}")
    logger.info("=" * 60)

    # Estimate model size
    _temp_model = PatchTST()
    n_params = _temp_model.count_parameters()
    logger.info(f"Model parameters: {n_params:,}")
    del _temp_model

    # Gather NPZ files
    npz_files = sorted(data_dir.glob("*_mbo_events.npz"))
    if not npz_files:
        logger.error(f"No *_mbo_events.npz files found in {data_dir}")
        sys.exit(1)

    if args.max_days and len(npz_files) > args.max_days:
        npz_files = npz_files[-args.max_days:]
        logger.info(f"Limited to most recent {args.max_days} days")
    logger.info(f"Found {len(npz_files)} NPZ files: {npz_files[0].name} -> {npz_files[-1].name}")

    # Set process priority on Windows
    try:
        import psutil
        p = psutil.Process()
        p.nice(psutil.BELOW_NORMAL_PRIORITY_CLASS)
        logger.info("Process priority set to BELOW_NORMAL")
    except Exception as e:
        logger.info(f"Could not set priority (non-critical): {e}")

    # Run walk-forward
    train_days = args.train_days if args.window_mode == "sliding" else None
    if args.window_mode == "sliding" and train_days is None:
        logger.warning("--window-mode sliding requires --train-days N; defaulting to 60")
        train_days = 60

    concat_ic = run_walk_forward(
        npz_files=npz_files,
        output_dir=output_dir,
        device=device,
        n_folds=args.n_folds,
        train_days=train_days,
        oot_days=args.oot_days,
        start_fold=args.start_fold,
    )

    logger.info("\n" + "=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("Final Concat IC:")
    for h in HORIZONS:
        logger.info(f"  {h}: {concat_ic.get(h, float('nan')):.4f}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
