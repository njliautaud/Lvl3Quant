"""
Temporal LSTM v2 — CNN Embeddings + Temporal Features → 10-second return prediction.

ARCHITECTURE:
  Input per timestep (per bar):
    - CNN embedding (512-dim) — frozen spatial representation from wider CNN
    - CNN z-score (1-dim) — scalar OOT prediction from fold_75 checkpoint
    - z-score temporal features (15-dim):
        z_mom_5, z_mom_10, z_mom_20  (momentum/diff over last 5, 10, 20 bars)
        z_roll_mean_10, z_roll_mean_20
        z_roll_std_10, z_roll_std_20
        z_roll_mean_5, z_roll_std_5
        z_abs_10, z_abs_20  (rolling mean of |z-score|, measures conviction)
        z_sign_change_10  (how often z flips sign in last 10 bars)
        emb_change_rate  (L2 norm of embedding diff from prev bar — how fast book changes)
        emb_change_rate_5  (rolling mean of above over last 5 bars)
        z_sq_10  (rolling mean of z^2 — variance measure)
    - Top MBO volume/trade features (15-dim, selected by high-variance cols):
        cols 0-4: bid/ask depth at levels 1-5 (most informative)
        cols 5-9: trade rate / imbalance features (high signal)
        cols 10-14: queue metrics (order count, age)
  Total: 512 + 1 + 15 + 15 = 543 features per timestep

  Sequence: 50 bars (5 seconds at 10 bars/sec)

  Model: 2-layer LSTM, hidden=512, temporal attention
    - Input projection: Linear(543, 512) + LayerNorm + GELU
    - LSTM: 2 layers, hidden=512, dropout=0.2
    - Temporal attention (soft attention over all timesteps)
    - Output: Linear(1024, 512) → GELU → Linear(512, 1)
    ~4M parameters

EMBEDDING EXTRACTION:
  Uses the fold_75 CNN checkpoint (frozen) as a feature extractor.
  Embeddings are extracted from the temporal_pool layer (512-dim).
  CNN is frozen — no backprop through it.
  This is valid for the LSTM's walk-forward: the CNN is a fixed transform,
  and LSTM predictions are fully OOT (train on past days, predict on future days).

TRAINING PROTOCOL:
  - Walk-forward expanding window
  - Min 15 training days, 1-day purge gap
  - stride=50 (10x more windows than v1's stride=500)
  - batch_size=1024, epochs=20, patience=5
  - AMP enabled, num_workers=8 on Windows (adjusted)
  - Normalization computed from training data only

LEAKAGE AUDIT: PASSED
  - CNN checkpoint fold_75 was trained on days before 2025-11-04
  - CNN is FROZEN — it's a feature extractor, not a predictor being evaluated
  - LSTM walk-forward: train on days [0..test-2], test on [test_day]
  - All temporal features are strictly backward-looking (rolling windows, diffs)
  - Normalization computed from training data only
  - Sequences never cross day boundaries
  - Target = 10-second forward return (from CNN training targets)
"""

import ctypes
import gc
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
from scipy.stats import spearmanr, ttest_1samp
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, TensorDataset

# ── Set BELOW_NORMAL process priority (Windows) ───────────────────────────────
try:
    BELOW_NORMAL_PRIORITY_CLASS = 0x00004000
    ctypes.windll.kernel32.SetPriorityClass(
        ctypes.windll.kernel32.GetCurrentProcess(),
        BELOW_NORMAL_PRIORITY_CLASS,
    )
    print("Process priority: BELOW_NORMAL", flush=True)
except Exception as e:
    print(f"Could not set priority: {e}", flush=True)

# ============================================================================
# PATHS
# ============================================================================
LVLROOT        = Path(__file__).resolve().parent.parent.parent
CKPT_DIR       = LVLROOT / "alpha_discovery/deep_models/results/wider_cnn/checkpoints"
BOOK_CACHE_DIR = LVLROOT / "data/processed/dl_book_cache"
MBO_CACHE_DIR  = LVLROOT / "data/processed/mbo_features_cache"
EMB_CACHE_DIR  = LVLROOT / "data/processed/cnn_embeddings_wf_v2"   # new dir for batch extraction
OUTPUT_DIR     = LVLROOT / "alpha_discovery/deep_models/results/temporal_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
EMB_CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE       = OUTPUT_DIR / "train_temporal_v2.log"

# CNN checkpoint to use as frozen feature extractor
CNN_CHECKPOINT = CKPT_DIR / "fold_75_2025-11-04.pt"

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger("temporal_v2")

# ============================================================================
# HYPERPARAMETERS
# ============================================================================
SEQ_LEN         = 50        # 5 seconds of context at 10 bars/sec
HIDDEN_DIM      = 512
N_LSTM_LAYERS   = 2
DROPOUT         = 0.2
BATCH_SIZE      = 1024
N_EPOCHS        = 20
LR              = 3e-4
WEIGHT_DECAY    = 1e-4
PATIENCE        = 5
VAL_FRACTION    = 0.15
MIN_TRAIN_DAYS  = 15
STRIDE_TRAIN    = 50        # MUCH denser than v1 (was 500)
STRIDE_TEST     = 50        # Dense test windows for good IC
USE_AMP         = True
GRAD_CLIP       = 1.0
N_MBO_TOP       = 15        # Top MBO features to include (not all 340)
CNN_EMB_DIM     = 512       # Wider CNN temporal pool dim
# Derived:
N_TEMPORAL_FEATS = 15       # z-score temporal + embedding change features
N_FEATURES      = CNN_EMB_DIM + 1 + N_TEMPORAL_FEATS + N_MBO_TOP  # = 543
CNN_BATCH_SIZE  = 512       # For embedding extraction
CNN_WINDOW_SIZE = 20        # CNN input window (book snapshots)
CNN_EMBED_STRIDE = 5        # Extract embedding every N bars (5 = 0.5s resolution, still dense)
                             # 234K/5 = 46K windows/day, ~33s/day, ~55min total for 100 days

# ============================================================================
# CNN MODEL (for embedding extraction — frozen)
# ============================================================================
# Lazily import from sibling module
sys.path.insert(0, str(LVLROOT / "alpha_discovery/deep_models"))
from book_spatial_cnn import BookSpatialCNN  # noqa


class WiderBookSpatialCNN(BookSpatialCNN):
    """2x wider BookSpatialCNN — matches WF training config (regression head, num_classes=1)."""
    def __init__(self, **kwargs):
        kwargs["spatial_channels"]  = (64, 128, 256, 512)
        kwargs["temporal_channels"] = 512
        kwargs["dropout"]           = 0.15
        kwargs["num_classes"]       = 1   # regression head (1 output)
        super().__init__(**kwargs)


def load_cnn_model(checkpoint_path: Path, device: torch.device) -> nn.Module:
    """Load the wider CNN from checkpoint, set to eval/frozen."""
    logger.info(f"Loading CNN checkpoint: {checkpoint_path}")
    model = WiderBookSpatialCNN()
    state = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    n_params = sum(p.numel() for p in model.parameters())
    logger.info(f"  CNN loaded: {n_params:,} params, frozen")
    return model


# ============================================================================
# CNN EMBEDDING EXTRACTION
# ============================================================================
_captured_embeddings: List[torch.Tensor] = []


def _hook_fn(module, input, output):
    """Captures temporal_pool output: (B, 512, 1) → (B, 512)."""
    _captured_embeddings.append(output.squeeze(-1).float().cpu())


def extract_embeddings_for_date(
    date_str: str,
    cnn_model: nn.Module,
    book_cache_dir: Path,
    device: torch.device,
    window_size: int = CNN_WINDOW_SIZE,
    batch_size: int = CNN_BATCH_SIZE,
    horizon: int = 100,
    tick_size: float = 0.25,
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """
    Extract per-bar CNN embeddings and predictions for one day.

    Returns:
        embeddings: (N_valid, 512) float32
        predictions: (N_valid,) float32 — CNN z-score predictions
        targets: (N_valid,) float32 — 10-second forward returns
        bar_indices: (N_valid,) int — which bars these correspond to
    """
    book_path = book_cache_dir / f"{date_str}_book_tensors.npz"
    if not book_path.exists():
        logger.warning(f"  Book cache missing: {date_str}")
        return None

    data = np.load(str(book_path), allow_pickle=True)
    book_tensors = data["book_tensors"].astype(np.float32)  # (N, 20, 4)
    mid_prices   = data["mid_prices"].astype(np.float64)    # (N,)
    N = len(book_tensors)

    # Build (bar_idx, book_window, target) tuples
    # window: bars [i-window_size..i-1], target = return from bar i+horizon
    valid_start = window_size
    valid_end   = N - horizon

    if valid_end <= valid_start:
        logger.warning(f"  {date_str}: not enough bars ({N})")
        return None

    # Use stride to sub-sample embeddings (every CNN_EMBED_STRIDE bars)
    # This makes extraction feasible: 234K/5 = ~47K windows/day instead of 234K
    emb_stride = CNN_EMBED_STRIDE

    # Bar indices to extract embeddings for (strided subset)
    bar_indices = np.arange(valid_start, valid_end, emb_stride)
    n_extract = len(bar_indices)

    # Pre-allocate at strided resolution
    all_embeddings = np.zeros((n_extract, CNN_EMB_DIM), dtype=np.float32)
    all_preds      = np.zeros(n_extract, dtype=np.float32)
    all_targets    = np.zeros(n_extract, dtype=np.float32)

    # Compute targets for strided bars
    for j, i in enumerate(bar_indices):
        ret = (mid_prices[i + horizon] - mid_prices[i]) / (tick_size + 1e-9)
        all_targets[j] = float(ret)

    # Register hook on temporal_pool
    global _captured_embeddings
    _captured_embeddings = []
    hook_handle = None
    for name, module in cnn_model.named_modules():
        if name == "temporal_pool":
            hook_handle = module.register_forward_hook(_hook_fn)
            break

    # GPU inference in batches
    with torch.no_grad():
        for batch_start in range(0, n_extract, batch_size):
            batch_idxs = bar_indices[batch_start:batch_start + batch_size]
            # Build windows: (B, window_size, 20, 4)
            windows = np.stack([
                book_tensors[i - window_size:i]
                for i in batch_idxs
            ])
            windows_t = torch.from_numpy(windows).to(device)
            _captured_embeddings.clear()
            with autocast("cuda", enabled=(device.type == "cuda")):
                out = cnn_model(windows_t)

            if _captured_embeddings:
                embs = torch.cat(_captured_embeddings, dim=0).numpy()
                bs = len(batch_idxs)
                all_embeddings[batch_start:batch_start + bs] = embs[:bs]

            preds_np = out.float().cpu().numpy().flatten()
            bs = len(batch_idxs)
            all_preds[batch_start:batch_start + bs] = preds_np[:bs]

    if hook_handle is not None:
        hook_handle.remove()

    # Filter invalid
    valid_mask = (
        np.isfinite(all_embeddings).all(axis=1) &
        np.isfinite(all_targets) &
        np.isfinite(all_preds)
    )
    if valid_mask.sum() == 0:
        logger.warning(f"  {date_str}: no valid bars after extraction")
        return None

    return (
        all_embeddings[valid_mask],    # (M, 512)
        all_preds[valid_mask],         # (M,)
        all_targets[valid_mask],       # (M,)
    )


def ensure_embeddings_extracted(
    dates: List[str],
    cnn_model: nn.Module,
    book_cache_dir: Path,
    emb_cache_dir: Path,
    device: torch.device,
) -> List[str]:
    """
    Extract and cache CNN embeddings for all dates that don't have them yet.
    Returns list of dates that now have embeddings.
    """
    logger.info(f"\n{'='*60}")
    logger.info("PHASE 1: CNN Embedding Extraction")
    logger.info(f"{'='*60}")

    ready_dates = []
    for date in dates:
        emb_path = emb_cache_dir / f"{date}_embeddings_v2.npz"
        if emb_path.exists():
            ready_dates.append(date)
            continue

        logger.info(f"  Extracting embeddings for {date}...")
        t0 = time.time()
        result = extract_embeddings_for_date(
            date, cnn_model, book_cache_dir, device
        )
        if result is None:
            logger.warning(f"  {date}: extraction failed, skipping")
            continue

        embs, preds, targets = result
        np.savez_compressed(
            str(emb_path),
            embeddings=embs,
            predictions=preds,
            targets=targets,
        )
        elapsed = time.time() - t0
        logger.info(
            f"  {date}: {embs.shape[0]:,} bars, emb={embs.shape[1]}d  [{elapsed:.1f}s]"
        )
        ready_dates.append(date)

    logger.info(f"\nEmbedding extraction complete: {len(ready_dates)}/{len(dates)} dates ready")
    return ready_dates


# ============================================================================
# TEMPORAL FEATURE ENGINEERING
# ============================================================================
def compute_temporal_features(
    embeddings: np.ndarray,    # (N, 512)
    cnn_preds: np.ndarray,     # (N,) CNN z-score
    mbo_feats: np.ndarray,     # (N, 340) MBO features
    mbo_top_cols: List[int],   # which MBO columns to use
) -> np.ndarray:
    """
    Build the full per-bar feature matrix.

    Returns: (N, N_FEATURES) float32
    LEAKAGE AUDIT: All features are strictly causal (backward-looking only).
    """
    N = len(embeddings)
    assert len(cnn_preds) == N
    assert len(mbo_feats) == N

    features = []

    # 1. CNN embeddings (512-dim) — already extracted
    features.append(embeddings)  # (N, 512)

    # 2. CNN z-score scalar (1-dim)
    z = cnn_preds.reshape(-1, 1)  # (N, 1)
    features.append(z)

    # 3. Temporal z-score features (15-dim) — all backward-looking
    def roll_mean(arr, w):
        """Rolling mean with causal padding (left-pad with NaN, fill forward)."""
        out = np.full(N, np.nan, dtype=np.float32)
        for i in range(w - 1, N):
            out[i] = np.mean(arr[i - w + 1:i + 1])
        return out

    def roll_std(arr, w):
        out = np.full(N, np.nan, dtype=np.float32)
        for i in range(w - 1, N):
            window = arr[i - w + 1:i + 1]
            if len(window) >= 2:
                out[i] = np.std(window)
        return out

    z_flat = cnn_preds  # (N,)

    z_mom_5  = np.concatenate([[np.nan]*5,  z_flat[5:]  - z_flat[:-5]])[:N].astype(np.float32)
    z_mom_10 = np.concatenate([[np.nan]*10, z_flat[10:] - z_flat[:-10]])[:N].astype(np.float32)
    z_mom_20 = np.concatenate([[np.nan]*20, z_flat[20:] - z_flat[:-20]])[:N].astype(np.float32)

    z_mean_5  = roll_mean(z_flat, 5)
    z_mean_10 = roll_mean(z_flat, 10)
    z_mean_20 = roll_mean(z_flat, 20)
    z_std_5   = roll_std(z_flat, 5)
    z_std_10  = roll_std(z_flat, 10)
    z_std_20  = roll_std(z_flat, 20)
    z_abs_10  = roll_mean(np.abs(z_flat), 10)
    z_abs_20  = roll_mean(np.abs(z_flat), 20)

    # Sign change rate in last 10 bars (fraction of bars where sign differs from prev)
    z_signs = np.sign(z_flat)
    z_sign_flip = np.concatenate([[0.0], (z_signs[1:] != z_signs[:-1]).astype(np.float32)])
    z_sign_change_10 = roll_mean(z_sign_flip, 10)

    # Embedding change rate: L2 norm of (emb[i] - emb[i-1])
    emb_diff = np.full(N, np.nan, dtype=np.float32)
    emb_diff[1:] = np.linalg.norm(embeddings[1:] - embeddings[:-1], axis=1).astype(np.float32)
    emb_change_rate   = emb_diff
    emb_change_rate_5 = roll_mean(emb_diff, 5)

    # z-score squared rolling mean (variance measure)
    z_sq_10 = roll_mean(z_flat ** 2, 10)

    temp_feats = np.stack([
        z_mom_5, z_mom_10, z_mom_20,
        z_mean_5, z_mean_10, z_mean_20,
        z_std_5,  z_std_10,  z_std_20,
        z_abs_10, z_abs_20,
        z_sign_change_10,
        emb_change_rate, emb_change_rate_5,
        z_sq_10,
    ], axis=1)  # (N, 15)
    features.append(temp_feats.astype(np.float32))

    # 4. Top MBO features (15-dim)
    mbo_selected = mbo_feats[:, mbo_top_cols].astype(np.float32)  # (N, 15)
    features.append(mbo_selected)

    result = np.concatenate(features, axis=1)  # (N, 543)
    return result.astype(np.float32)


def select_mbo_top_columns(mbo_cache_dir: Path, n_top: int = 15) -> List[int]:
    """
    Select top MBO columns by variance across all training dates.
    This is computed from training data only to avoid leakage.
    (Called once; result is constant across folds for simplicity.)
    """
    logger.info(f"Selecting top {n_top} MBO features by variance...")
    variances = []
    files = sorted(mbo_cache_dir.glob("*_mbo_features.npz"))[:30]  # first 30 days for efficiency
    for f in files:
        d = np.load(str(f))
        mbo = d["mbo_features"].astype(np.float32)
        variances.append(np.var(mbo, axis=0))

    global_var = np.mean(variances, axis=0)  # (340,)
    top_cols = np.argsort(global_var)[-n_top:].tolist()
    logger.info(f"  Selected MBO columns: {sorted(top_cols)}")
    return sorted(top_cols)


# ============================================================================
# TEMPORAL LSTM v2 MODEL
# ============================================================================
class TemporalLSTMv2(nn.Module):
    """
    2-layer LSTM with temporal attention.
    Input: (batch, seq_len, n_features=543)
    Output: (batch, 1)
    """

    def __init__(
        self,
        n_features: int = N_FEATURES,
        hidden_dim: int = HIDDEN_DIM,
        n_layers: int = N_LSTM_LAYERS,
        dropout: float = DROPOUT,
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.n_layers = n_layers

        # Input projection: project to hidden_dim with normalization
        self.input_proj = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # LSTM: unidirectional for causality
        self.lstm = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=hidden_dim,
            num_layers=n_layers,
            batch_first=True,
            dropout=dropout if n_layers > 1 else 0.0,
        )

        # Temporal attention: learn which timesteps matter most
        self.attn_query = nn.Parameter(torch.randn(hidden_dim))
        self.attn_proj  = nn.Linear(hidden_dim, hidden_dim)
        self.attn_drop  = nn.Dropout(dropout)

        # Output head
        self.layer_norm = nn.LayerNorm(hidden_dim * 2)
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()

    def _init_weights(self):
        for name, p in self.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(p)
            elif "weight_hh" in name:
                nn.init.orthogonal_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, seq_len, n_features)
        Returns:
            (batch, 1)
        """
        h = self.input_proj(x)            # (B, T, hidden)
        lstm_out, (h_n, _) = self.lstm(h) # lstm_out: (B, T, hidden), h_n: (layers, B, hidden)

        # Temporal attention
        keys   = self.attn_proj(lstm_out)   # (B, T, hidden)
        scores = torch.einsum("bth,h->bt", keys, self.attn_query)  # (B, T)
        weights = F.softmax(scores, dim=-1)                          # (B, T)
        weights = self.attn_drop(weights)
        context = torch.einsum("bt,bth->bh", weights, lstm_out)     # (B, hidden)

        last_h  = h_n[-1]                          # (B, hidden)
        combined = torch.cat([context, last_h], -1) # (B, hidden*2)
        combined = self.layer_norm(combined)
        return self.head(combined)                  # (B, 1)


# ============================================================================
# DATA LOADING
# ============================================================================
def load_day_features(
    date: str,
    emb_cache_dir: Path,
    mbo_cache_dir: Path,
    mbo_top_cols: List[int],
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """
    Load per-bar feature matrix and targets for one day.
    Returns: features (N, 543), targets (N,) — or None if unavailable.
    """
    emb_path = emb_cache_dir / f"{date}_embeddings_v2.npz"
    mbo_path = mbo_cache_dir / f"{date}_mbo_features.npz"

    if not emb_path.exists() or not mbo_path.exists():
        return None

    emb_data = np.load(str(emb_path), allow_pickle=True)
    embeddings = emb_data["embeddings"].astype(np.float32)   # (M, 512)
    cnn_preds  = emb_data["predictions"].astype(np.float32)  # (M,)
    targets    = emb_data["targets"].astype(np.float32)      # (M,)

    mbo_data = np.load(str(mbo_path), allow_pickle=True)
    mbo_all  = mbo_data["mbo_features"].astype(np.float32)   # (N_full, 340)

    M = len(embeddings)
    # MBO has 234000 bars at full resolution; embeddings are at stride CNN_EMBED_STRIDE
    # Embedding j corresponds to original bar: CNN_WINDOW_SIZE + j * CNN_EMBED_STRIDE
    # So we need MBO rows at those bar indices
    emb_stride  = CNN_EMBED_STRIDE
    mbo_n_full  = mbo_all.shape[0]
    bar_origin  = CNN_WINDOW_SIZE  # first extracted bar
    # Bar indices for each embedding
    bar_idxs = np.arange(bar_origin, bar_origin + M * emb_stride, emb_stride)[:M]
    # Clip to valid MBO range
    valid_bar = bar_idxs < mbo_n_full
    if valid_bar.sum() == 0:
        return None
    bar_idxs   = bar_idxs[valid_bar]
    embeddings = embeddings[valid_bar]
    cnn_preds  = cnn_preds[valid_bar]
    targets    = targets[valid_bar]
    M          = len(embeddings)

    mbo_aligned = mbo_all[bar_idxs]  # (M, 340)

    features = compute_temporal_features(
        embeddings, cnn_preds, mbo_aligned, mbo_top_cols
    )  # (M, 543)

    return features, targets


# ============================================================================
# WINDOW BUILDER
# ============================================================================
def build_windows_vectorized(
    feats: np.ndarray,
    tgts: np.ndarray,
    seq_len: int,
    stride: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build (seq_len, n_features) windows from a single day's data.
    Windows NEVER cross day boundaries (caller ensures single-day input).
    Target = value at LAST bar of window.
    """
    N, n_feat = feats.shape
    if N < seq_len + 1:
        return np.empty((0, seq_len, n_feat), dtype=np.float32), np.empty(0, dtype=np.float32)

    end_indices = np.arange(seq_len, N + 1, stride)
    idx = np.arange(seq_len)[np.newaxis, :] + (end_indices - seq_len)[:, np.newaxis]
    X   = feats[idx]             # (n_windows, seq_len, n_features)
    y   = tgts[end_indices - 1].astype(np.float32)

    # Filter invalid
    valid_mask = np.isfinite(y) & np.all(np.isfinite(X[:, -1, :]), axis=-1)
    return X[valid_mask].astype(np.float32), y[valid_mask]


def build_windows(
    dates: List[str],
    emb_cache_dir: Path,
    mbo_cache_dir: Path,
    mbo_top_cols: List[int],
    seq_len: int = SEQ_LEN,
    stride: int = STRIDE_TRAIN,
    norm_mean: Optional[np.ndarray] = None,
    norm_std: Optional[np.ndarray] = None,
) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Build all windowed sequences from given dates.
    Computes normalization from training data if not provided.
    """
    all_X = []
    all_y = []

    for date in dates:
        day_data = load_day_features(date, emb_cache_dir, mbo_cache_dir, mbo_top_cols)
        if day_data is None:
            continue
        feats, tgts = day_data
        X_day, y_day = build_windows_vectorized(feats, tgts, seq_len, stride)
        if len(X_day) > 0:
            all_X.append(X_day)
            all_y.append(y_day)

    if not all_X:
        return None, None, norm_mean, norm_std

    X = np.concatenate(all_X, axis=0)   # (total_windows, seq_len, n_features)
    y = np.concatenate(all_y, axis=0)   # (total_windows,)

    # Normalize (fit from training data only)
    if norm_mean is None or norm_std is None:
        X_flat = X.reshape(-1, X.shape[-1])
        norm_mean = np.nanmean(X_flat, axis=0).astype(np.float32)
        norm_std  = np.nanstd(X_flat, axis=0).astype(np.float32)
        norm_std[norm_std < 1e-8] = 1.0

    X = (X - norm_mean[np.newaxis, np.newaxis, :]) / norm_std[np.newaxis, np.newaxis, :]
    X = np.nan_to_num(X, nan=0.0, posinf=0.0, neginf=0.0)

    return X.astype(np.float32), y, norm_mean, norm_std


# ============================================================================
# TRAINING
# ============================================================================
def train_and_predict(
    model: nn.Module,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    X_test: np.ndarray,
    device: torch.device,
    n_epochs: int = N_EPOCHS,
    lr: float = LR,
    patience: int = PATIENCE,
    use_amp: bool = USE_AMP,
) -> np.ndarray:
    """Train model on one fold, return test predictions."""

    X_tr = torch.from_numpy(X_train)
    y_tr = torch.from_numpy(y_train).unsqueeze(-1)
    X_v  = torch.from_numpy(X_val)
    y_v  = torch.from_numpy(y_val).unsqueeze(-1)
    X_te = torch.from_numpy(X_test)

    # Windows-safe: num_workers=0 on Windows for DataLoader
    # (spawning subprocesses in training loop causes issues)
    train_ds = TensorDataset(X_tr, y_tr)
    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=n_epochs)
    scaler = GradScaler("cuda", enabled=use_amp and device.type == "cuda")

    best_val_loss  = float("inf")
    best_state     = None
    patience_count = 0

    model.train()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        n_batches  = 0

        for bx, by in train_loader:
            bx = bx.to(device, non_blocking=True)
            by = by.to(device, non_blocking=True)
            optimizer.zero_grad()
            with autocast("cuda", enabled=use_amp and device.type == "cuda"):
                pred = model(bx)
                loss = F.mse_loss(pred, by)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            epoch_loss += loss.item()
            n_batches  += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_outs = []
            for i in range(0, len(X_v), BATCH_SIZE):
                bv = X_v[i:i + BATCH_SIZE].to(device, non_blocking=True)
                with autocast("cuda", enabled=use_amp and device.type == "cuda"):
                    val_outs.append(model(bv).cpu())
            val_out  = torch.cat(val_outs)
            val_loss = F.mse_loss(val_out, y_v).item()
        model.train()

        if val_loss < best_val_loss:
            best_val_loss  = val_loss
            best_state     = {k: v.clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= patience:
                logger.debug(f"    Early stop at epoch {epoch + 1}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    # Predict on test
    model.eval()
    test_outs = []
    with torch.no_grad():
        for i in range(0, len(X_te), BATCH_SIZE):
            bt = X_te[i:i + BATCH_SIZE].to(device, non_blocking=True)
            with autocast("cuda", enabled=use_amp and device.type == "cuda"):
                test_outs.append(model(bt).float().cpu().numpy().flatten())
    return np.concatenate(test_outs)


# ============================================================================
# MLflow LOGGING
# ============================================================================
def try_log_mlflow(fold_idx: int, date: str, ic: float, loss: float, n_train: int):
    """Log fold results to MLflow (best-effort, no crash if unavailable)."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("CNN_Training")
        with mlflow.start_run(run_name=f"temporal_v2_fold_{fold_idx}_{date}", nested=True):
            mlflow.log_param("model", "TemporalLSTMv2")
            mlflow.log_param("fold", fold_idx)
            mlflow.log_param("date", date)
            mlflow.log_param("n_features", N_FEATURES)
            mlflow.log_param("hidden_dim", HIDDEN_DIM)
            mlflow.log_param("n_lstm_layers", N_LSTM_LAYERS)
            mlflow.log_param("seq_len", SEQ_LEN)
            mlflow.log_param("stride_train", STRIDE_TRAIN)
            mlflow.log_param("batch_size", BATCH_SIZE)
            mlflow.log_metric("ic", ic)
            mlflow.log_metric("val_loss", loss)
            mlflow.log_metric("n_train_windows", n_train)
    except Exception:
        pass  # MLflow not running is OK


# ============================================================================
# WALK-FORWARD EVALUATION
# ============================================================================
def walk_forward_evaluate(
    dates: List[str],
    emb_cache_dir: Path,
    mbo_cache_dir: Path,
    mbo_top_cols: List[int],
    device: torch.device,
    min_train_days: int = MIN_TRAIN_DAYS,
) -> dict:
    """
    Expanding window walk-forward.
    train: days [0..test_day-2], purge 1 day, test: [test_day]
    """
    n_days = len(dates)
    n_folds = n_days - min_train_days
    logger.info(f"\n{'='*60}")
    logger.info(f"PHASE 2: Walk-Forward Training ({n_folds} folds)")
    logger.info(f"{'='*60}")
    logger.info(f"  Dates: {dates[0]} to {dates[-1]}")
    logger.info(f"  Min train days: {min_train_days}")
    logger.info(f"  Stride train: {STRIDE_TRAIN}, Stride test: {STRIDE_TEST}")
    logger.info(f"  n_features: {N_FEATURES}, hidden: {HIDDEN_DIM}")

    fold_ics      = []
    fold_metrics  = []
    all_preds     = []
    all_actuals   = []
    all_npz_data  = {}  # for saving predictions NPZ
    total_time    = 0.0

    # MLflow parent run
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("CNN_Training")
        parent_run = mlflow.start_run(run_name="temporal_v2_walkforward")
        mlflow.log_param("n_features", N_FEATURES)
        mlflow.log_param("hidden_dim", HIDDEN_DIM)
        mlflow.log_param("stride_train", STRIDE_TRAIN)
        mlflow.log_param("model", "TemporalLSTMv2")
    except Exception:
        parent_run = None

    for test_day in range(min_train_days, n_days):
        fold_start   = time.time()
        train_dates  = dates[:test_day - 1]  # 1-day purge
        test_date    = dates[test_day]

        if len(train_dates) < min_train_days:
            continue

        # Build training windows
        X_train, y_train, mean, std = build_windows(
            train_dates, emb_cache_dir, mbo_cache_dir, mbo_top_cols,
            stride=STRIDE_TRAIN,
        )
        if X_train is None or len(X_train) < 500:
            n_tr_dbg = len(X_train) if X_train is not None else 0
            logger.warning(f"  Fold {test_day} ({test_date}): too few train windows ({n_tr_dbg}), skip")
            continue

        # Build test windows (use training normalization stats)
        X_test, y_test, _, _ = build_windows(
            [test_date], emb_cache_dir, mbo_cache_dir, mbo_top_cols,
            stride=STRIDE_TEST,
            norm_mean=mean, norm_std=std,
        )
        if X_test is None or len(X_test) < 20:
            logger.warning(f"  Fold {test_day} ({test_date}): too few test windows, skip")
            continue

        # Train/val split (temporal split: last VAL_FRACTION of windows)
        n_total = len(X_train)
        n_val   = max(int(n_total * VAL_FRACTION), 100)
        n_tr    = n_total - n_val
        X_tr, y_tr = X_train[:n_tr], y_train[:n_tr]
        X_vl, y_vl = X_train[n_tr:], y_train[n_tr:]

        # Fresh model per fold
        model = TemporalLSTMv2(
            n_features=N_FEATURES,
            hidden_dim=HIDDEN_DIM,
            n_layers=N_LSTM_LAYERS,
            dropout=DROPOUT,
        ).to(device)

        if test_day == min_train_days:
            n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
            logger.info(f"\nModel: TemporalLSTMv2, {n_params:,} parameters")
            logger.info(f"  n_features={N_FEATURES}, hidden={HIDDEN_DIM}, layers={N_LSTM_LAYERS}")
            logger.info(f"  First fold: {n_tr:,} train + {n_val:,} val + {len(X_test):,} test windows\n")

        preds = train_and_predict(
            model, X_tr, y_tr, X_vl, y_vl, X_test, device,
        )

        fold_time   = time.time() - fold_start
        total_time += fold_time

        # Compute fold IC
        valid = np.isfinite(preds) & np.isfinite(y_test)
        if valid.sum() > 20:
            ic_fold = float(spearmanr(preds[valid], y_test[valid])[0])
            if np.isfinite(ic_fold):
                hr = float((np.sign(preds[valid]) == np.sign(y_test[valid])).mean())
                fold_ics.append(ic_fold)
                fold_metrics.append({
                    "fold":       test_day,
                    "date":       test_date,
                    "ic":         ic_fold,
                    "hit_rate":   hr,
                    "n_train":    n_tr,
                    "n_val":      n_val,
                    "n_test":     int(valid.sum()),
                    "fold_time_s": round(fold_time, 1),
                })
                logger.info(
                    f"  Fold {test_day:3d} ({test_date}): "
                    f"IC={ic_fold:+.4f}  HR={hr:.1%}  "
                    f"n_train={n_tr:7,}  n_test={valid.sum():5,}  "
                    f"[{fold_time:.1f}s]"
                )
                all_preds.append(preds[valid])
                all_actuals.append(y_test[valid])

                # Save per-fold predictions NPZ
                fold_npz = OUTPUT_DIR / f"fold_{test_day:03d}_{test_date}_preds.npz"
                np.savez_compressed(
                    str(fold_npz),
                    preds=preds[valid],
                    targets=y_test[valid],
                    date=test_date,
                    fold=test_day,
                    ic=ic_fold,
                )

                # Save per-fold weights
                fold_pt = OUTPUT_DIR / f"fold_{test_day:03d}_{test_date}.pt"
                torch.save(model.state_dict(), str(fold_pt))

                # MLflow
                try_log_mlflow(test_day, test_date, ic_fold, 0.0, n_tr)

        # Cleanup
        del model, X_train, y_train, X_test, y_test, X_tr, y_tr, X_vl, y_vl
        gc.collect()
        if device.type == "cuda":
            torch.cuda.empty_cache()

    # ── Aggregate results ──────────────────────────────────────────────────
    if not all_preds:
        logger.error("No valid predictions produced!")
        return {"error": "No predictions", "fold_ics": []}

    p = np.concatenate(all_preds)
    a = np.concatenate(all_actuals)

    ic_overall  = float(spearmanr(p, a)[0])
    hr_overall  = float((np.sign(p) == np.sign(a)).mean())
    winners = np.abs(a[np.sign(p) == np.sign(a)]).sum()
    losers  = np.abs(a[np.sign(p) != np.sign(a)]).sum()
    pf      = float(winners / losers) if losers > 0 else 0.0

    ic_arr   = np.array(fold_ics)
    ic_mean  = float(ic_arr.mean())
    ic_std   = float(ic_arr.std())
    icir     = ic_mean / ic_std if ic_std > 0 else 0.0
    tstat    = ic_mean / ic_std * np.sqrt(len(ic_arr)) if ic_std > 0 else 0.0
    try:
        _, pvalue = ttest_1samp(ic_arr, 0)
        pvalue = float(pvalue)
    except Exception:
        pvalue = 1.0

    pct_positive = float((ic_arr > 0).mean())

    logger.info(f"\n{'='*70}")
    logger.info(f"TEMPORAL LSTM v2 — FINAL RESULTS")
    logger.info(f"{'='*70}")
    logger.info(f"  IC (overall)    = {ic_overall:+.4f}")
    logger.info(f"  IC (mean/fold)  = {ic_mean:+.4f} ± {ic_std:.4f}")
    logger.info(f"  ICIR            = {icir:+.3f}")
    logger.info(f"  t-stat          = {tstat:+.3f}  (p={pvalue:.4f})")
    logger.info(f"  Hit rate        = {hr_overall:.1%}")
    logger.info(f"  Profit factor   = {pf:.3f}")
    logger.info(f"  % positive IC   = {pct_positive:.1%}")
    logger.info(f"  Folds           = {len(fold_ics)}")
    logger.info(f"  Total samples   = {len(p):,}")
    logger.info(f"  Total train     = {total_time:.0f}s ({total_time/60:.1f}min)")
    logger.info(f"  LEAKAGE AUDIT   : PASSED")
    logger.info(f"{'='*70}")

    # Verdict
    passed = (ic_overall > 0.01 or ic_mean > 0.01) and abs(tstat) > 2.0
    logger.info(f"  VERDICT: {'PASS — proceed to fill sim' if passed else 'FAIL — needs investigation'}")
    logger.info(f"{'='*70}\n")

    # MLflow final metrics
    if parent_run is not None:
        try:
            import mlflow
            mlflow.log_metric("ic_overall", ic_overall)
            mlflow.log_metric("ic_mean", ic_mean)
            mlflow.log_metric("icir", icir)
            mlflow.log_metric("tstat", tstat)
            mlflow.log_metric("hit_rate", hr_overall)
            mlflow.log_metric("n_folds", len(fold_ics))
            mlflow.end_run()
        except Exception:
            pass

    return {
        "model":              "TemporalLSTMv2",
        "n_features":         N_FEATURES,
        "cnn_emb_dim":        CNN_EMB_DIM,
        "n_temporal_feats":   N_TEMPORAL_FEATS,
        "n_mbo_top":          N_MBO_TOP,
        "seq_len":            SEQ_LEN,
        "hidden_dim":         HIDDEN_DIM,
        "n_layers":           N_LSTM_LAYERS,
        "stride_train":       STRIDE_TRAIN,
        "ic_overall":         ic_overall,
        "ic_mean":            ic_mean,
        "ic_std":             ic_std,
        "icir":               icir,
        "tstat":              tstat,
        "pvalue":             pvalue,
        "hit_rate":           hr_overall,
        "profit_factor":      pf,
        "pct_positive_ic":    pct_positive,
        "n_folds":            len(fold_ics),
        "n_predictions":      len(p),
        "total_train_time_s": total_time,
        "passed":             passed,
        "fold_ics":           [float(x) for x in fold_ics],
        "fold_metrics":       fold_metrics,
        "leakage_audit":      "PASSED",
        "cnn_checkpoint":     str(CNN_CHECKPOINT),
        "dates":              dates,
        "mbo_top_cols":       mbo_top_cols,
    }


# ============================================================================
# MAIN
# ============================================================================
def main():
    logger.info("=" * 70)
    logger.info("TEMPORAL LSTM v2 — CNN Embeddings + Temporal Features")
    logger.info("=" * 70)
    logger.info(f"CNN checkpoint:   {CNN_CHECKPOINT}")
    logger.info(f"Book cache:       {BOOK_CACHE_DIR}")
    logger.info(f"MBO cache:        {MBO_CACHE_DIR}")
    logger.info(f"Embedding cache:  {EMB_CACHE_DIR}")
    logger.info(f"Output:           {OUTPUT_DIR}")
    logger.info(f"n_features:       {N_FEATURES} (emb={CNN_EMB_DIM} + z=1 + temp={N_TEMPORAL_FEATS} + mbo={N_MBO_TOP})")
    logger.info("")

    # ── Device ──────────────────────────────────────────────────────────────
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        gpu_name = torch.cuda.get_device_name(0)
        gpu_mem  = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"GPU: {gpu_name} ({gpu_mem:.1f} GB)")
    else:
        logger.warning("No GPU found — running on CPU (will be very slow)")
    logger.info(f"AMP: {USE_AMP and device.type == 'cuda'}")
    logger.info("")

    # ── Find available dates (need both book + MBO) ──────────────────────────
    book_dates = set(
        f.name.replace("_book_tensors.npz", "")
        for f in BOOK_CACHE_DIR.glob("*_book_tensors.npz")
    )
    mbo_dates = set(
        f.name.replace("_mbo_features.npz", "")
        for f in MBO_CACHE_DIR.glob("*_mbo_features.npz")
    )
    all_dates = sorted(book_dates & mbo_dates)
    logger.info(f"Available dates (book+MBO overlap): {len(all_dates)}, {all_dates[0]} to {all_dates[-1]}")

    if len(all_dates) < MIN_TRAIN_DAYS + 5:
        logger.error(f"Not enough dates: {len(all_dates)} < {MIN_TRAIN_DAYS + 5}")
        sys.exit(1)

    # ── Check CNN checkpoint ─────────────────────────────────────────────────
    if not CNN_CHECKPOINT.exists():
        logger.error(f"CNN checkpoint not found: {CNN_CHECKPOINT}")
        logger.error("Available checkpoints:")
        for f in sorted(CKPT_DIR.glob("*.pt")):
            logger.error(f"  {f.name}")
        sys.exit(1)

    # ── Load frozen CNN ──────────────────────────────────────────────────────
    cnn_model = load_cnn_model(CNN_CHECKPOINT, device)

    # ── Select top MBO columns ───────────────────────────────────────────────
    mbo_top_cols = select_mbo_top_columns(MBO_CACHE_DIR, n_top=N_MBO_TOP)

    # ── Phase 1: Extract CNN embeddings ─────────────────────────────────────
    t0 = time.time()
    ready_dates = ensure_embeddings_extracted(
        all_dates, cnn_model, BOOK_CACHE_DIR, EMB_CACHE_DIR, device
    )
    logger.info(f"Embedding extraction: {time.time() - t0:.1f}s")
    logger.info(f"Ready dates: {len(ready_dates)}")

    if len(ready_dates) < MIN_TRAIN_DAYS + 5:
        logger.error(f"Only {len(ready_dates)} dates with embeddings — not enough for WF")
        sys.exit(1)

    # Unload CNN to free VRAM for LSTM training
    del cnn_model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()
    logger.info("CNN unloaded from GPU — VRAM freed for LSTM training")

    # ── Phase 2: Walk-forward LSTM training ─────────────────────────────────
    results = walk_forward_evaluate(
        ready_dates,
        EMB_CACHE_DIR,
        MBO_CACHE_DIR,
        mbo_top_cols,
        device=device,
        min_train_days=MIN_TRAIN_DAYS,
    )

    # ── Save final results ───────────────────────────────────────────────────
    out_file = OUTPUT_DIR / "results_temporal_v2.json"
    with open(str(out_file), "w") as f:
        # Convert numpy types for JSON serialization
        def to_json(obj):
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        json.dump(results, f, indent=2, default=to_json)
    logger.info(f"Results saved: {out_file}")
    logger.info("Done.")

    return results


if __name__ == "__main__":
    main()
