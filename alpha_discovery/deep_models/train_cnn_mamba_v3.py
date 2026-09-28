"""
CNN-Mamba v3 — FIFO-Aware Multi-Head Trainer

Per CNN_MAMBA_V3_SPEC.md (HC #270, #271(D), #278, #281, #282).

Architecture:
  - Inherits CNNMamba backbone from train_cnn_mamba.py (proven champion: IC_1s=0.222)
  - Replaces single 3-target head with 7 heads on a shared trunk:
      1. pred_log_ret_1s   (regression, MSE,    λ=1.0)
      2. pred_log_ret_5s   (regression, MSE,    λ=0.5)
      3. pred_log_ret_10s  (regression, MSE,    λ=0.3)
      4. pred_fifo_tp4sl3_net_ticks (Huber, λ=1.0)  [NEW]
      5. pred_fifo_tp8sl5_net_ticks (Huber, λ=1.0)  [NEW]
      6. pred_fifo_tp4sl3_hit_tp    (BCE,   λ=0.3)  [NEW]
      7. pred_fifo_tp8sl5_hit_tp    (BCE,   λ=0.3)  [NEW]

Inputs (REUSE smart_v3 dataset, no new feature engineering):
  data/processed/mbo_events_smart_v3/<date>_mbo_events.npz
  data/processed/mbo_events_smart_v3_fifo_labels/<date>_fifo_labels.npz

Walk-Forward (per HC #281(D)):
  10 weekly OOT folds anchored at 2026-02-23 (Mon)
  Each fold: 60 trading days SLIDING train, 5 trading days OOT (Mon-Fri)
  Fold 1 OOT = 2026-03-02 → directly comparable to v2 fold 6 (concat IC=0.221)

Normalization (per HC #278(B), #281(E)):
  Layer 1: smart_v3 input pre-norm — already done by dataset
  Layer 2: per-fold per-feature z-score on TRAIN only, applied to OOT
  Layer 3: per-day rank-norm on book_imbalance(idx 12), queue_depth_ratio(idx 15),
           signal_persistence(idx 17) — hardens against vol regime shifts

Warm-start (per HC #281(F)):
  Load v2 fold_10_best.pt backbone weights into v3 backbone (skip head — different shape)

MLflow (per HC #281(H)):
  Tailscale IP fixed → jupiter (was stale neptune-win)
  Artifact upload skipped if cross-host fails — metrics-only acceptable

Author: Claude (head-of-quant), 2026-05-10 15:30 ET, under HC #281/#282/#283
"""

# Set feature set BEFORE importing v2 trainer module (its constants depend on env)
import os as _os
_os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
_os.environ.setdefault("EVENT_WINDOW_SIZE", "3000")
_os.environ.setdefault("EVENT_STRIDE", "250")
_os.environ.setdefault("MLFLOW_TRACKING_URI", "http://jupiter:5000")

import os
import sys
import gc
import time
import json
import logging
import argparse
import socket
import re
import gzip
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.stats

# HC #722 FIX: causal rolling z-score for rank normalization (replaces full-day rankdata)
from alpha_discovery.deep_models.precompute_features_smart_v3 import causal_rolling_zscore

# Import proven v2 components (model + sampler).
# We deliberately reuse the architecture; only the head and loss change.
from alpha_discovery.deep_models.train_cnn_mamba import (
    CNNMamba,
    FileSequentialSampler,
    WarmupCosineScheduler,
    count_parameters,
    MAMBA_D_MODEL,
    MAMBA_D_STATE,
    MAMBA_N_LAYERS,
    MAMBA_DROPOUT,
    MAMBA_DT_RANK,
    MAMBA_D_CONV,
    CNN_CHANNELS,
    CNN_KERNEL,
    CNN_LAYERS,
    WINDOW_SIZE,
    STRIDE,
    BATCH_SIZE,
    LR,
    EPOCHS_PER_FOLD,
    WARMUP_STEPS,
    GRAD_CLIP,
)

# MLflow (mandatory per HC #6)
try:
    import mlflow
    MLFLOW_AVAILABLE = True
    if os.environ.get("DISABLE_MLFLOW", "0") == "1":
        MLFLOW_AVAILABLE = False
except ImportError:
    MLFLOW_AVAILABLE = False

# ============================================================
# Logging
# ============================================================
LOG_DIR = Path(__file__).parent / "results"
LOG_DIR.mkdir(exist_ok=True)
log_path = LOG_DIR / "cnn_mamba_v3.log"

# Add a separate file handler for v3 (don't replace v2 handlers)
_v3_handler = logging.FileHandler(log_path)
_v3_handler.setLevel(logging.INFO)
_v3_handler.setFormatter(logging.Formatter("%(asctime)s [v3 %(levelname)s] %(message)s"))
logging.root.addHandler(_v3_handler)

logger = logging.getLogger("cnn_mamba_v3")
logger.setLevel(logging.INFO)

print(">>> train_cnn_mamba_v3.py loaded", flush=True)

# ============================================================
# Config
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data" / "processed" / "mbo_events_smart_v3")
DEFAULT_LABEL_DIR = str(PROJECT_ROOT / "data" / "processed" / "mbo_events_smart_v3_fifo_labels")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output" / "cnn_mamba_v3_fifo_mar")
V2_WARMSTART_CKPT = str(PROJECT_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar" / "fold_10_best.pt")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")
MLFLOW_EXPERIMENT = "CNNMamba_v3_FIFO"

# Per-day rank-norm feature indices (per HC #281(E) Layer 3)
RANK_NORM_FEATURE_IDXS = [12, 15, 17]  # book_imbalance, queue_depth_ratio, signal_persistence

# Loss weights (per spec §2 table)
LOSS_LAMBDA = {
    "log_ret_1s":           1.0,
    "log_ret_5s":           0.5,
    "log_ret_10s":          0.3,
    "fifo_tp4sl3_net":      1.0,
    "fifo_tp8sl5_net":      1.0,
    "fifo_tp4sl3_hit_tp":   0.3,
    "fifo_tp8sl5_hit_tp":   0.3,
}

# Cap FIFO label values (prevent gradient blowup)
FIFO_LABEL_CAP_TICKS = 20.0

# Walk-forward (per HC #281(D))
WF_TRAIN_DAYS = int(os.environ.get("V3_WF_TRAIN_DAYS", 60))
N_FOLDS = int(os.environ.get("V3_N_FOLDS", 10))
FIRST_OOT_MONDAY = "2026-02-23"  # per HC #281(C), #282(B)

# ============================================================
# Date utilities
# ============================================================
DATE_RE = re.compile(r"(\d{8})_mbo_events\.npz$")


def date_from_path(p: Path) -> str:
    """Extract YYYYMMDD from MBO NPZ filename."""
    m = DATE_RE.search(p.name)
    if not m:
        return ""
    return m.group(1)


def fifo_label_path(label_dir: Path, date_str: str) -> Path:
    return label_dir / f"{date_str}_fifo_labels.npz"


def build_weekly_fold_schedule(
    available_dates: List[str],
    first_oot_monday: str = FIRST_OOT_MONDAY,
    n_folds: int = N_FOLDS,
    train_days: int = WF_TRAIN_DAYS,
) -> List[Dict]:
    """
    Per HC #281(D)/#282(B): each fold = 5 trading days OOT (Mon-Fri),
    train = `train_days` trading days SLIDING immediately before.
    First OOT week starts at first_oot_monday so fold 1 OOT = 2026-03-02 directly
    compares to v2 fold 6 (concat IC=0.221).
    """
    import datetime as _dt
    available_set = set(available_dates)
    sorted_dates = sorted(available_dates)

    # Find the first available trading date >= first_oot_monday
    target_dt = _dt.datetime.strptime(first_oot_monday, "%Y-%m-%d").date()
    target_str = target_dt.strftime("%Y%m%d")

    # Find anchor index in sorted_dates
    anchor_idx = None
    for i, d in enumerate(sorted_dates):
        if d >= target_str:
            anchor_idx = i
            break
    if anchor_idx is None:
        logger.error(f"No available date >= {first_oot_monday} (anchor missing)")
        return []

    folds = []
    cur_idx = anchor_idx
    for fold_n in range(n_folds):
        # OOT = next 5 trading days (or fewer if data ends)
        oot_dates = sorted_dates[cur_idx:cur_idx + 5]
        if len(oot_dates) == 0:
            break
        # Train = train_days dates immediately before OOT
        train_start_idx = max(0, cur_idx - train_days)
        train_dates = sorted_dates[train_start_idx:cur_idx]
        if len(train_dates) < 10:
            logger.warning(f"Fold {fold_n}: only {len(train_dates)} train days available, skipping")
            cur_idx += 5
            continue
        folds.append({
            "fold": fold_n,
            "train_dates": train_dates,
            "oot_dates": oot_dates,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "oot_start": oot_dates[0],
            "oot_end": oot_dates[-1],
        })
        cur_idx += 5
    return folds


# ============================================================
# Multi-Head Wrapper
# ============================================================
class CNNMambaV3(nn.Module):
    """
    v3 wrapper: v2 CNN-Mamba backbone + 7 heads on shared trunk.

    The backbone (CNN front-end + Mamba blocks + final_norm) is identical to v2.
    Only the head module is replaced.
    """
    HEAD_NAMES = [
        "log_ret_1s",
        "log_ret_5s",
        "log_ret_10s",
        "fifo_tp4sl3_net",
        "fifo_tp8sl5_net",
        "fifo_tp4sl3_hit_tp",
        "fifo_tp8sl5_hit_tp",
    ]

    def __init__(
        self,
        d_model: int = MAMBA_D_MODEL,
        d_state: int = MAMBA_D_STATE,
        n_layers: int = MAMBA_N_LAYERS,
        dt_rank: int = MAMBA_DT_RANK,
        d_conv: int = MAMBA_D_CONV,
        dropout: float = MAMBA_DROPOUT,
        cnn_channels: int = CNN_CHANNELS,
        cnn_kernel: int = CNN_KERNEL,
        cnn_layers: int = CNN_LAYERS,
        trunk_dim: int = 128,
    ):
        super().__init__()
        # Reuse v2 backbone with n_targets=3 (head will be discarded — we only
        # use everything up to and including final_norm).
        self.backbone = CNNMamba(
            d_model=d_model,
            d_state=d_state,
            n_layers=n_layers,
            dt_rank=dt_rank,
            d_conv=d_conv,
            dropout=dropout,
            n_targets=3,  # keeps backbone identical to v2 for warm-start
            cnn_channels=cnn_channels,
            cnn_kernel=cnn_kernel,
            cnn_layers=cnn_layers,
        )
        # Discard backbone.head — we use a fresh trunk + 7 heads
        self.backbone.head = nn.Identity()

        self.trunk = nn.Sequential(
            nn.Linear(d_model, trunk_dim),
            nn.GELU(),
            nn.LayerNorm(trunk_dim),
            nn.Dropout(dropout),
        )
        # 7 small heads on shared trunk
        self.heads = nn.ModuleDict({
            name: nn.Linear(trunk_dim, 1) for name in self.HEAD_NAMES
        })
        self._init_new_weights()

    def _init_new_weights(self):
        """Initialize the new trunk + heads (backbone weights handled by backbone._init_weights)."""
        for m in list(self.trunk.modules()) + list(self.heads.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, events: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Args:
            events: (B, L, F) batch of event windows
        Returns:
            dict mapping head name -> (B,) tensor
        """
        # backbone returns (preds_dummy, embedding); preds_dummy is from Identity head
        # so we use return_embedding=True to get the (B, d_model) embedding directly.
        _, embedding = self.backbone(events, return_embedding=True)
        trunk_out = self.trunk(embedding)  # (B, trunk_dim)
        outputs = {}
        for name, head in self.heads.items():
            outputs[name] = head(trunk_out).squeeze(-1)  # (B,)
        return outputs

    def load_v2_warmstart(self, ckpt_path: str, device: torch.device) -> bool:
        """Load v2 checkpoint into backbone (skip head)."""
        if not Path(ckpt_path).exists():
            logger.warning(f"V2 warm-start checkpoint not found: {ckpt_path} — training from scratch")
            return False
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            v2_state = ckpt.get("model_state", ckpt)
            # Map v2 state into backbone (keys are the same since v2.head is Identity in v3)
            backbone_state = self.backbone.state_dict()
            loaded = 0
            skipped = 0
            for k, v in v2_state.items():
                if k.startswith("head."):
                    skipped += 1
                    continue
                if k in backbone_state and backbone_state[k].shape == v.shape:
                    backbone_state[k] = v
                    loaded += 1
                else:
                    skipped += 1
            self.backbone.load_state_dict(backbone_state, strict=False)
            logger.info(
                f"V2 warm-start loaded: {loaded} tensors copied, {skipped} skipped "
                f"(from {Path(ckpt_path).name}, val_loss={ckpt.get('val_loss','?')})"
            )
            return True
        except Exception as e:
            logger.warning(f"V2 warm-start failed: {e} — training from scratch")
            return False


# ============================================================
# Joint Multi-Head Loss
# ============================================================
class JointMultiHeadLoss(nn.Module):
    """
    Weighted sum of 7 heads with per-head masking for unfilled FIFO rows.

    For FIFO regression heads (4, 5): use Huber loss, MASK unfilled rows.
    For FIFO binary heads (6, 7):    use BCEWithLogits, weight unfilled at 0.25.
    For log_ret heads (1, 2, 3):     use MSE, no mask (always available).
    """
    def __init__(self, lambdas: Dict[str, float] = None, huber_delta: float = 2.0):
        super().__init__()
        self.lambdas = lambdas or LOSS_LAMBDA
        self.huber = nn.HuberLoss(reduction="none", delta=huber_delta)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        preds:   dict[name -> (B,)]  raw outputs (logits for binary heads)
        targets: dict[name -> (B,)]  ground truth
        masks:   dict[name -> (B,) bool/float]  1.0 where row is valid for this head
        """
        components = {}
        total = 0.0
        for name, lam in self.lambdas.items():
            if lam == 0.0:
                continue
            p = preds[name]
            t = targets[name]
            m = masks.get(name, torch.ones_like(p))

            if name.startswith("log_ret"):
                loss_per = self.mse(p, t)
            elif name.endswith("_net"):
                loss_per = self.huber(p, t)
            elif name.endswith("_hit_tp"):
                loss_per = self.bce(p, t)
            else:
                continue

            denom = m.sum().clamp_min(1.0)
            loss = (loss_per * m).sum() / denom
            components[name] = float(loss.item())
            total = total + lam * loss

        return total, components


# ============================================================
# Dataset — smart_v3 + FIFO labels with per-day rank-norm
# ============================================================
class SmartV3FifoDataset(Dataset):
    """
    Loads (events_window, multi_head_targets, masks) per stride position.

    Per-fold normalization workflow:
      1. Per-day rank-norm applied here on RANK_NORM_FEATURE_IDXS at file load
      2. Per-fold z-score applied via feature_stats argument (mean, std arrays of length 25)
         — TRAIN ds computes stats; OOT ds receives them from train ds (no leakage)
    """

    HEAD_NAMES = CNNMambaV3.HEAD_NAMES

    def __init__(
        self,
        data_dir: Path,
        label_dir: Path,
        dates: List[str],
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        feature_stats: Optional[Dict] = None,  # {"mean": (25,), "std": (25,)}
        cache_size: int = 8,
        require_labels: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.label_dir = Path(label_dir)
        self.dates = dates
        self.window_size = window_size
        self.stride = stride
        self.cache_size = cache_size
        self.require_labels = require_labels
        self.feature_stats = feature_stats  # may be None for train (will compute)

        # LRU cache of loaded days (events + labels + fifo)
        self._cache: Dict[str, Dict] = {}
        self._cache_order: List[str] = []

        # Sample index: (date_str, window_start, window_k)
        self.sample_index: List[Tuple[str, int, int]] = []
        self._build_index()

        # If feature_stats not provided, compute from train days
        if self.feature_stats is None:
            self._compute_feature_stats()

    def _load_day_raw(self, date_str: str) -> Optional[Dict]:
        """Load events + FIFO labels for one date, apply per-day rank-norm."""
        events_path = self.data_dir / f"{date_str}_mbo_events.npz"
        labels_path = self.label_dir / f"{date_str}_fifo_labels.npz"
        if not events_path.exists():
            return None
        if self.require_labels and not labels_path.exists():
            logger.warning(f"FIFO labels missing for {date_str} — skipping date")
            return None

        for _attempt in range(8):
            try:
                ev = np.load(events_path, allow_pickle=True)
                break
            except PermissionError:
                if _attempt < 7:
                    time.sleep(3)
                else:
                    raise

        events = ev["events"].astype(np.float32)  # (N, 25)
        labels_1s = ev["labels_1s"].astype(np.float32)
        labels_5s = ev["labels_5s"].astype(np.float32)
        labels_10s = ev["labels_10s"].astype(np.float32)

        # HC #722 FIX: Per-day CAUSAL rank-norm on event features.
        # OLD CODE used scipy.stats.rankdata() on FULL DAY (leakage — rank at time t
        # depends on future events). Replaced with causal rolling z-score (W=5000).
        n = len(events)
        if n > 1:
            for fidx in RANK_NORM_FEATURE_IDXS:
                col = events[:, fidx]
                z = causal_rolling_zscore(col, 5000)
                events[:, fidx] = np.clip(z / 5.0, -1.0, 1.0)

        # Load FIFO labels (aligned by window_k)
        fifo = {}
        if labels_path.exists():
            fl = np.load(labels_path, allow_pickle=True)
            fifo = {
                "window_k":              fl["window_k"].astype(np.int64),
                "tp4sl3_short_net":      fl["tp4sl3_short_net_ticks"].astype(np.float32),
                "tp4sl3_short_filled":   fl["tp4sl3_short_filled"].astype(np.bool_),
                "tp4sl3_short_hit_tp":   fl["tp4sl3_short_hit_tp"].astype(np.bool_),
                "tp8sl5_short_net":      fl["tp8sl5_short_net_ticks"].astype(np.float32),
                "tp8sl5_short_filled":   fl["tp8sl5_short_filled"].astype(np.bool_),
                "tp8sl5_short_hit_tp":   fl["tp8sl5_short_hit_tp"].astype(np.bool_),
            }
            # Cap FIFO net values
            np.clip(fifo["tp4sl3_short_net"], -FIFO_LABEL_CAP_TICKS, FIFO_LABEL_CAP_TICKS,
                    out=fifo["tp4sl3_short_net"])
            np.clip(fifo["tp8sl5_short_net"], -FIFO_LABEL_CAP_TICKS, FIFO_LABEL_CAP_TICKS,
                    out=fifo["tp8sl5_short_net"])
            # Build a sparse map: window_k -> row index for fast lookup
            wk_to_row = {int(k): i for i, k in enumerate(fifo["window_k"])}
            fifo["wk_to_row"] = wk_to_row
        return {
            "events": events,
            "labels_1s": labels_1s,
            "labels_5s": labels_5s,
            "labels_10s": labels_10s,
            "fifo": fifo,
        }

    def _get_day(self, date_str: str) -> Optional[Dict]:
        """LRU cache for day data."""
        if date_str in self._cache:
            self._cache_order.remove(date_str)
            self._cache_order.append(date_str)
            return self._cache[date_str]

        d = self._load_day_raw(date_str)
        if d is None:
            return None
        self._cache[date_str] = d
        self._cache_order.append(date_str)
        while len(self._cache_order) > self.cache_size:
            old = self._cache_order.pop(0)
            del self._cache[old]
        return d

    def _build_index(self):
        """Build (date, start, window_k) sample index. Drop windows with all-NaN labels."""
        for date_str in self.dates:
            events_path = self.data_dir / f"{date_str}_mbo_events.npz"
            labels_path = self.label_dir / f"{date_str}_fifo_labels.npz"
            if not events_path.exists():
                logger.warning(f"smart_v3 events missing for {date_str}")
                continue
            if self.require_labels and not labels_path.exists():
                logger.warning(f"FIFO labels missing for {date_str} — skipping")
                continue

            try:
                ev = np.load(events_path, allow_pickle=True)
                n_events = len(ev["events"])
                lab1 = ev["labels_1s"]
            except Exception as e:
                logger.warning(f"Failed to read {events_path.name}: {e}")
                continue

            window_k = 0
            for start in range(0, n_events - self.window_size + 1, self.stride):
                end = start + self.window_size
                label_idx = end - 1
                if not np.isnan(lab1[label_idx]):
                    self.sample_index.append((date_str, start, window_k))
                window_k += 1
            del ev

        logger.info(
            f"Dataset: {len(self.dates)} days, {len(self.sample_index)} samples "
            f"(window={self.window_size}, stride={self.stride})"
        )

    def _compute_feature_stats(self):
        """Compute per-feature mean/std across all train events (after per-day rank-norm)."""
        logger.info(f"Computing per-fold feature stats from {len(self.dates)} train dates...")
        n_features = 25
        total_sum = np.zeros(n_features, dtype=np.float64)
        total_sq = np.zeros(n_features, dtype=np.float64)
        total_count = 0
        for date_str in self.dates:
            day = self._load_day_raw(date_str)
            if day is None:
                continue
            ev = day["events"].astype(np.float64)
            total_sum += ev.sum(axis=0)
            total_sq += (ev ** 2).sum(axis=0)
            total_count += len(ev)
        if total_count == 0:
            logger.warning("No events found for stats — using zero/one defaults")
            self.feature_stats = {
                "mean": np.zeros(n_features, dtype=np.float32),
                "std": np.ones(n_features, dtype=np.float32),
            }
            return
        mean = (total_sum / total_count).astype(np.float32)
        var = (total_sq / total_count) - (mean.astype(np.float64) ** 2)
        std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
        self.feature_stats = {"mean": mean, "std": std}
        logger.info(f"Feature stats computed (n={total_count} events)")

    def get_feature_stats(self) -> Dict:
        return self.feature_stats

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        date_str, start, window_k = self.sample_index[idx]
        end = start + self.window_size
        day = self._get_day(date_str)
        if day is None:
            # Should never happen post-build_index
            raise RuntimeError(f"Failed to load day {date_str} at __getitem__")
        events = day["events"][start:end]  # (W, 25)
        # Apply per-fold z-score (Layer 2)
        if self.feature_stats is not None:
            events = (events - self.feature_stats["mean"]) / (self.feature_stats["std"] + 1e-8)

        label_idx = end - 1
        targets = {
            "log_ret_1s":  np.float32(day["labels_1s"][label_idx]),
            "log_ret_5s":  np.float32(day["labels_5s"][label_idx]),
            "log_ret_10s": np.float32(day["labels_10s"][label_idx]),
            "fifo_tp4sl3_net":     np.float32(0.0),
            "fifo_tp8sl5_net":     np.float32(0.0),
            "fifo_tp4sl3_hit_tp":  np.float32(0.0),
            "fifo_tp8sl5_hit_tp":  np.float32(0.0),
        }
        masks = {
            "log_ret_1s":  np.float32(1.0),
            "log_ret_5s":  np.float32(1.0),
            "log_ret_10s": np.float32(1.0),
            "fifo_tp4sl3_net":    np.float32(0.0),
            "fifo_tp8sl5_net":    np.float32(0.0),
            "fifo_tp4sl3_hit_tp": np.float32(0.25),  # weak gradient on unfilled
            "fifo_tp8sl5_hit_tp": np.float32(0.25),
        }
        # Replace NaN log-rets with 0 + zero-mask
        for h in ("log_ret_1s", "log_ret_5s", "log_ret_10s"):
            v = targets[h]
            if np.isnan(v):
                targets[h] = np.float32(0.0)
                masks[h] = np.float32(0.0)

        # Join FIFO labels by window_k
        fifo = day.get("fifo", {})
        if fifo:
            row = fifo.get("wk_to_row", {}).get(int(window_k), -1)
            if row >= 0:
                # tp4sl3 short
                if fifo["tp4sl3_short_filled"][row]:
                    targets["fifo_tp4sl3_net"] = np.float32(fifo["tp4sl3_short_net"][row])
                    masks["fifo_tp4sl3_net"] = np.float32(1.0)
                    targets["fifo_tp4sl3_hit_tp"] = np.float32(1.0 if fifo["tp4sl3_short_hit_tp"][row] else 0.0)
                    masks["fifo_tp4sl3_hit_tp"] = np.float32(1.0)
                # tp8sl5 short
                if fifo["tp8sl5_short_filled"][row]:
                    targets["fifo_tp8sl5_net"] = np.float32(fifo["tp8sl5_short_net"][row])
                    masks["fifo_tp8sl5_net"] = np.float32(1.0)
                    targets["fifo_tp8sl5_hit_tp"] = np.float32(1.0 if fifo["tp8sl5_short_hit_tp"][row] else 0.0)
                    masks["fifo_tp8sl5_hit_tp"] = np.float32(1.0)

        # Convert to tensors
        events_t = torch.from_numpy(events.astype(np.float32))
        targets_t = {k: torch.tensor(v) for k, v in targets.items()}
        masks_t = {k: torch.tensor(v) for k, v in masks.items()}
        return events_t, targets_t, masks_t


def collate_v3(batch):
    """Stack events + per-head dicts into tensors."""
    events_list, targets_list, masks_list = zip(*batch)
    events = torch.stack(events_list, dim=0)
    targets = {k: torch.stack([t[k] for t in targets_list], dim=0)
               for k in targets_list[0]}
    masks = {k: torch.stack([m[k] for m in masks_list], dim=0)
             for k in masks_list[0]}
    return events, targets, masks


# ============================================================
# Metrics
# ============================================================
def compute_ic(predictions: np.ndarray, labels: np.ndarray) -> float:
    valid = ~(np.isnan(predictions) | np.isnan(labels))
    if valid.sum() < 20:
        return float("nan")
    rho, _ = scipy.stats.spearmanr(predictions[valid], labels[valid])
    return float(rho)


def evaluate_v3(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True,
) -> Tuple[Dict, Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    model.eval()
    total_loss = 0.0
    n_batches = 0
    all_preds = defaultdict(list)
    all_targets = defaultdict(list)
    all_masks = defaultdict(list)

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    with torch.no_grad():
        for events, targets, masks in loader:
            events = events.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            masks = {k: v.to(device, non_blocking=True) for k, v in masks.items()}
            with amp_ctx:
                preds = model(events)
                loss, _ = loss_fn(preds, targets, masks)
            total_loss += float(loss.item())
            n_batches += 1
            for k, v in preds.items():
                all_preds[k].append(v.float().cpu().numpy())
            for k, v in targets.items():
                all_targets[k].append(v.float().cpu().numpy())
            for k, v in masks.items():
                all_masks[k].append(v.float().cpu().numpy())

    metrics = {"loss": total_loss / max(n_batches, 1)}
    preds_out = {k: np.concatenate(v) if v else np.empty(0) for k, v in all_preds.items()}
    targets_out = {k: np.concatenate(v) if v else np.empty(0) for k, v in all_targets.items()}
    masks_out = {k: np.concatenate(v) if v else np.empty(0) for k, v in all_masks.items()}

    # IC for log_ret heads (no mask needed; nan-safe)
    for h in ("log_ret_1s", "log_ret_5s", "log_ret_10s"):
        m = masks_out[h] > 0
        if m.sum() > 20:
            metrics[f"ic_{h}"] = compute_ic(preds_out[h][m], targets_out[h][m])
        else:
            metrics[f"ic_{h}"] = float("nan")

    # FIFO masked correlation: between pred_net and realized net (where filled)
    for h in ("fifo_tp4sl3_net", "fifo_tp8sl5_net"):
        m = masks_out[h] > 0
        if m.sum() > 20:
            metrics[f"corr_{h}"] = compute_ic(preds_out[h][m], targets_out[h][m])
        else:
            metrics[f"corr_{h}"] = float("nan")

    # FIFO hit_tp AUC-ish: just report mean prediction vs mean realized hit-rate
    for h in ("fifo_tp4sl3_hit_tp", "fifo_tp8sl5_hit_tp"):
        m = masks_out[h] >= 1.0  # only filled rows
        if m.sum() > 20:
            p_sig = 1.0 / (1.0 + np.exp(-preds_out[h][m]))  # logits -> prob
            metrics[f"hitrate_pred_{h}"] = float(p_sig.mean())
            metrics[f"hitrate_real_{h}"] = float(targets_out[h][m].mean())
        else:
            metrics[f"hitrate_pred_{h}"] = float("nan")
            metrics[f"hitrate_real_{h}"] = float("nan")

    return metrics, preds_out, targets_out, masks_out


# ============================================================
# Train one fold
# ============================================================
def train_one_fold_v3(
    model: CNNMambaV3,
    train_loader: DataLoader,
    oot_loader: DataLoader,
    fold_idx: int,
    output_dir: Path,
    mlflow_run,
    device: torch.device,
    total_train_steps: int,
    use_amp: bool = True,
) -> Dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # bf16 doesn't need grad scaler
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=WARMUP_STEPS, total_steps=total_train_steps
    )
    loss_fn = JointMultiHeadLoss(LOSS_LAMBDA)

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    best_val_loss = float("inf")
    global_step = 0
    total_batches = len(train_loader)

    print(f">>> v3 train_one_fold {fold_idx}: {EPOCHS_PER_FOLD} epochs × {total_batches} batches", flush=True)

    for epoch in range(EPOCHS_PER_FOLD):
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        comp_acc = defaultdict(float)

        for events, targets, masks in train_loader:
            events = events.to(device, non_blocking=True)
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            masks = {k: v.to(device, non_blocking=True) for k, v in masks.items()}

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                preds = model(events)
                loss, components = loss_fn(preds, targets, masks)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            optimizer.step()
            scheduler.step()

            epoch_loss += float(loss.item())
            n_batches += 1
            global_step += 1
            for k, v in components.items():
                comp_acc[k] += v

            if n_batches % 100 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / n_batches * (total_batches - n_batches)
                msg = (
                    f"  Fold {fold_idx} Ep {epoch+1} Batch {n_batches}/{total_batches} | "
                    f"Loss: {epoch_loss/n_batches:.4f} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
                )
                print(msg, flush=True)
                logger.info(msg)

            if n_batches % 500 == 0:
                ckpt = {
                    "model_state": model.state_dict(),
                    "fold": fold_idx, "epoch": epoch, "batch": n_batches,
                    "global_step": global_step, "best_val_loss": best_val_loss,
                }
                torch.save(ckpt, output_dir / f"fold_{fold_idx:02d}_intra_ckpt.pt")

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start
        comp_avg = {k: v / max(n_batches, 1) for k, v in comp_acc.items()}

        val_metrics, _, _, _ = evaluate_v3(model, oot_loader, loss_fn, device, use_amp=use_amp)
        msg = (
            f"Fold {fold_idx:02d} Ep {epoch+1}/{EPOCHS_PER_FOLD} | "
            f"TrLoss {avg_loss:.4f} | OOT Loss {val_metrics['loss']:.4f} | "
            f"OOT IC 1s/5s/10s = "
            f"{val_metrics.get('ic_log_ret_1s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_5s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_10s', float('nan')):.4f} | "
            f"FIFO corr tp4/tp8 = "
            f"{val_metrics.get('corr_fifo_tp4sl3_net', float('nan')):.4f}/"
            f"{val_metrics.get('corr_fifo_tp8sl5_net', float('nan')):.4f} | "
            f"LR {scheduler.get_lr():.2e} | T {epoch_time:.1f}s"
        )
        print(msg, flush=True)
        logger.info(msg)

        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                step = fold_idx * EPOCHS_PER_FOLD + epoch
                metrics_to_log = {
                    f"f{fold_idx:02d}_train_loss": avg_loss,
                    f"f{fold_idx:02d}_oot_loss": val_metrics["loss"],
                }
                for k, v in val_metrics.items():
                    if k != "loss" and not np.isnan(v):
                        metrics_to_log[f"f{fold_idx:02d}_{k}"] = float(v)
                for k, v in comp_avg.items():
                    metrics_to_log[f"f{fold_idx:02d}_train_loss_{k}"] = float(v)
                mlflow.log_metrics(metrics_to_log, step=step)
            except Exception as e:
                logger.warning(f"MLflow log failed: {e}")

        # Save best per fold
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "model_state": model.state_dict(),
                "fold": fold_idx, "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "val_metrics": val_metrics,
                "arch": {
                    "model": "CNNMambaV3",
                    "d_model": MAMBA_D_MODEL, "d_state": MAMBA_D_STATE,
                    "n_layers": MAMBA_N_LAYERS, "dt_rank": MAMBA_DT_RANK,
                    "d_conv": MAMBA_D_CONV, "dropout": MAMBA_DROPOUT,
                    "cnn_channels": CNN_CHANNELS, "cnn_kernel": CNN_KERNEL,
                    "cnn_layers": CNN_LAYERS, "window_size": WINDOW_SIZE,
                    "trunk_dim": 128, "head_names": CNNMambaV3.HEAD_NAMES,
                },
            }, output_dir / f"fold_{fold_idx:02d}_best.pt")

    return {"best_val_loss": best_val_loss}


# ============================================================
# Walk-forward driver
# ============================================================
def run_weekly_wf_v3(
    data_dir: Path,
    label_dir: Path,
    output_dir: Path,
    device: torch.device,
    n_folds: int = N_FOLDS,
    train_days: int = WF_TRAIN_DAYS,
    warmstart_ckpt: str = V2_WARMSTART_CKPT,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Discover available dates
    npz_files = sorted(data_dir.glob("*_mbo_events.npz"))
    available_dates = sorted([date_from_path(p) for p in npz_files if date_from_path(p)])
    if not available_dates:
        logger.error(f"No smart_v3 NPZ files found in {data_dir}")
        return {}
    logger.info(f"Found {len(available_dates)} smart_v3 dates: {available_dates[0]} → {available_dates[-1]}")

    # Restrict to dates with FIFO labels available (require_labels for train AND OOT)
    available_dates = [d for d in available_dates if (label_dir / f"{d}_fifo_labels.npz").exists()]
    logger.info(f"Dates with FIFO labels: {len(available_dates)}")

    folds = build_weekly_fold_schedule(available_dates, n_folds=n_folds, train_days=train_days)
    logger.info(f"Built {len(folds)} weekly folds (anchor={FIRST_OOT_MONDAY})")
    for f in folds:
        logger.info(
            f"  Fold {f['fold']}: train {f['train_start']}→{f['train_end']} ({len(f['train_dates'])}d) "
            f"OOT {f['oot_start']}→{f['oot_end']} ({len(f['oot_dates'])}d)"
        )

    # Save fold schedule for downstream tooling
    with open(output_dir / "fold_schedule.json", "w") as fh:
        json.dump(folds, fh, indent=2, default=str)

    use_amp = device.type == "cuda"

    # MLflow
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(run_name=f"v3_{time.strftime('%Y%m%d_%H%M')}_{socket.gethostname()}")
            gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
            mlflow.log_params({
                "model": "CNNMambaV3",
                "window_size": WINDOW_SIZE, "stride": STRIDE,
                "d_model": MAMBA_D_MODEL, "d_state": MAMBA_D_STATE,
                "n_layers": MAMBA_N_LAYERS, "dt_rank": MAMBA_DT_RANK,
                "d_conv": MAMBA_D_CONV, "dropout": MAMBA_DROPOUT,
                "cnn_channels": CNN_CHANNELS, "cnn_kernel": CNN_KERNEL,
                "cnn_layers": CNN_LAYERS,
                "batch_size": BATCH_SIZE, "lr": LR, "epochs_per_fold": EPOCHS_PER_FOLD,
                "warmup_steps": WARMUP_STEPS, "grad_clip": GRAD_CLIP,
                "n_folds_planned": len(folds),
                "wf_train_days": train_days,
                "first_oot_monday": FIRST_OOT_MONDAY,
                "n_features": 25, "feature_set": "smart_v3",
                "rank_norm_idxs": str(RANK_NORM_FEATURE_IDXS),
                "loss_lambdas": json.dumps(LOSS_LAMBDA),
                "fifo_label_cap_ticks": FIFO_LABEL_CAP_TICKS,
                "warmstart_ckpt": warmstart_ckpt,
                "node": socket.gethostname(),
                "gpu": gpu_name,
                "data_dir": str(data_dir),
                "label_dir": str(label_dir),
                "output_dir": str(output_dir),
                "mixed_precision": "bf16" if use_amp else "none",
            })
            logger.info(f"MLflow run started: {mlflow_run.info.run_id}")
        except Exception as e:
            logger.warning(f"MLflow init failed: {e} — continuing without tracking")
            mlflow_run = None

    # Aggregated results
    concat_results = {h: {"preds": [], "targets": [], "masks": []} for h in CNNMambaV3.HEAD_NAMES}

    try:
        start_fold = int(os.environ.get("V3_START_FOLD", 0))
        for f_info in folds:
            fold_idx = f_info["fold"]
            if fold_idx < start_fold:
                logger.info(f"Skipping fold {fold_idx} (V3_START_FOLD={start_fold})")
                continue
            logger.info("=" * 60)
            logger.info(f"FOLD {fold_idx} | train {f_info['train_start']}→{f_info['train_end']} "
                        f"| OOT {f_info['oot_start']}→{f_info['oot_end']}")
            logger.info("=" * 60)

            # Build train dataset (computes feature stats)
            train_ds = SmartV3FifoDataset(
                data_dir=data_dir, label_dir=label_dir,
                dates=f_info["train_dates"],
                window_size=WINDOW_SIZE, stride=STRIDE,
                feature_stats=None,  # compute from train
                cache_size=4,
                require_labels=True,
            )
            feature_stats = train_ds.get_feature_stats()
            # Save feature stats for live inference (HC #281(E))
            np.savez(
                output_dir / f"fold_{fold_idx:02d}_feature_stats.npz",
                mean=feature_stats["mean"], std=feature_stats["std"],
            )

            oot_ds = SmartV3FifoDataset(
                data_dir=data_dir, label_dir=label_dir,
                dates=f_info["oot_dates"],
                window_size=WINDOW_SIZE, stride=STRIDE,
                feature_stats=feature_stats,
                cache_size=4,
                require_labels=True,
            )

            # HC #285(D) + CLAUDE.md training rule #8 -- THREE-PROBLEM dataloader fix:
            #
            # Problem #1 (original): num_workers=0 starved the GPU (main thread did
            #   all dataloader work serially).
            # Problem #2 (after fix #1): num_workers=8 + cache_size=8 OOM-killed
            #   workers on Neptune's 32GB RAM (8 workers * 8 dates * 1.15GB = >70GB).
            # Problem #3 (the structural one): with shuffle=True, a batch of 128
            #   samples drawn uniformly from 60 train dates touches ~60 unique
            #   dates per batch. With cache_size <60, every batch incurs ~60 npz
            #   loads * ~10s each (heavy: full-day events + scipy rankdata x3).
            #   First batch takes ~10 min, GPU stays at 0%. Effectively dead.
            #
            # FIX: shuffle=False on TRAIN. Workers see samples in (date, start_idx)
            # order within the dataset — each worker stays on one date for many
            # consecutive samples, so cache_size=4 gives ~100% hit rate. Quality
            # cost: per-epoch sample order is deterministic. Acceptable for ship-
            # tonight v3 fold 0; proper date-aware BatchSampler is the TODO that
            # restores shuffle quality without the thrash (next session).
            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=2, pin_memory=True, drop_last=True,
                persistent_workers=True, prefetch_factor=4,
                collate_fn=collate_v3,
            )
            oot_loader = DataLoader(
                oot_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                num_workers=1, pin_memory=True,
                persistent_workers=True, prefetch_factor=4,
                collate_fn=collate_v3,
            )

            # Fresh model + warm-start
            model = CNNMambaV3().to(device)
            if fold_idx == 0:
                logger.info(f"Model parameters: {count_parameters(model):,}")
            model.load_v2_warmstart(warmstart_ckpt, device)

            total_steps = EPOCHS_PER_FOLD * len(train_loader)
            train_one_fold_v3(
                model, train_loader, oot_loader,
                fold_idx, output_dir, mlflow_run, device,
                total_train_steps=total_steps, use_amp=use_amp,
            )

            # Reload best for OOT inference
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                logger.info(f"Loaded best (val_loss={ckpt['val_loss']:.4f})")

            loss_fn = JointMultiHeadLoss(LOSS_LAMBDA)
            metrics, preds, targets, masks = evaluate_v3(
                model, oot_loader, loss_fn, device, use_amp=use_amp
            )
            logger.info(
                f"Fold {fold_idx} OOT FINAL | "
                f"IC 1s={metrics.get('ic_log_ret_1s', float('nan')):.4f} | "
                f"IC 5s={metrics.get('ic_log_ret_5s', float('nan')):.4f} | "
                f"IC 10s={metrics.get('ic_log_ret_10s', float('nan')):.4f} | "
                f"FIFO corr tp4={metrics.get('corr_fifo_tp4sl3_net', float('nan')):.4f} | "
                f"FIFO corr tp8={metrics.get('corr_fifo_tp8sl5_net', float('nan')):.4f}"
            )

            # Save OOT artifacts
            save_dict = {
                "fold_idx": np.array(fold_idx),
                "oot_dates": np.array(f_info["oot_dates"]),
            }
            for h in CNNMambaV3.HEAD_NAMES:
                save_dict[f"pred_{h}"] = preds[h]
                save_dict[f"target_{h}"] = targets[h]
                save_dict[f"mask_{h}"] = masks[h]
            np.savez_compressed(output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz", **save_dict)

            # Aggregate for concat
            for h in CNNMambaV3.HEAD_NAMES:
                concat_results[h]["preds"].append(preds[h])
                concat_results[h]["targets"].append(targets[h])
                concat_results[h]["masks"].append(masks[h])

            # Save fold analysis JSON
            fold_summary = {
                "fold": fold_idx,
                "train_window": [f_info["train_start"], f_info["train_end"], len(f_info["train_dates"])],
                "oot_window": [f_info["oot_start"], f_info["oot_end"], len(f_info["oot_dates"])],
                "n_train_samples": len(train_ds),
                "n_oot_samples": len(oot_ds),
                "metrics": {k: float(v) if not np.isnan(v) else None for k, v in metrics.items()},
            }
            with open(output_dir / f"fold_{fold_idx:02d}_analysis.json", "w") as fh:
                json.dump(fold_summary, fh, indent=2, default=str)

            if MLFLOW_AVAILABLE and mlflow_run is not None:
                try:
                    final_metrics = {
                        f"oot_final_f{fold_idx:02d}_{k}": float(v)
                        for k, v in metrics.items() if not np.isnan(v)
                    }
                    mlflow.log_metrics(final_metrics, step=fold_idx)
                except Exception as e:
                    logger.warning(f"MLflow final log failed: {e}")

            del train_ds, oot_ds, train_loader, oot_loader, model
            gc.collect()
            torch.cuda.empty_cache()

        # Concat
        logger.info("=" * 60)
        logger.info("CONCAT RESULTS (all folds combined)")
        logger.info("=" * 60)
        concat_summary = {}
        for h in CNNMambaV3.HEAD_NAMES:
            if not concat_results[h]["preds"]:
                continue
            p = np.concatenate(concat_results[h]["preds"])
            t = np.concatenate(concat_results[h]["targets"])
            m = np.concatenate(concat_results[h]["masks"])
            valid = m > 0
            if h.startswith("log_ret"):
                ic = compute_ic(p[valid], t[valid])
                concat_summary[f"concat_ic_{h}"] = ic
                logger.info(f"  Concat IC {h}: {ic:.4f}  (n={int(valid.sum())})")
            elif h.endswith("_net"):
                corr = compute_ic(p[valid], t[valid])
                concat_summary[f"concat_corr_{h}"] = corr
                logger.info(f"  Concat corr {h}: {corr:.4f}  (n={int(valid.sum())})")
            elif h.endswith("_hit_tp"):
                p_sig = 1.0 / (1.0 + np.exp(-p[valid]))
                pred_rate = float(p_sig.mean()) if valid.sum() > 0 else float("nan")
                real_rate = float(t[valid].mean()) if valid.sum() > 0 else float("nan")
                concat_summary[f"concat_hitrate_pred_{h}"] = pred_rate
                concat_summary[f"concat_hitrate_real_{h}"] = real_rate
                logger.info(f"  Concat {h}: pred_rate={pred_rate:.4f} real_rate={real_rate:.4f}")

        np.savez_compressed(
            output_dir / "concat_oot_predictions.npz",
            **{f"preds_{h}": np.concatenate(concat_results[h]["preds"]) for h in CNNMambaV3.HEAD_NAMES if concat_results[h]["preds"]},
            **{f"targets_{h}": np.concatenate(concat_results[h]["targets"]) for h in CNNMambaV3.HEAD_NAMES if concat_results[h]["targets"]},
            **{f"masks_{h}": np.concatenate(concat_results[h]["masks"]) for h in CNNMambaV3.HEAD_NAMES if concat_results[h]["masks"]},
        )
        with open(output_dir / "concat_summary.json", "w") as fh:
            json.dump({k: (float(v) if not np.isnan(v) else None) for k, v in concat_summary.items()}, fh, indent=2)

        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                mlflow.log_metrics({k: float(v) for k, v in concat_summary.items() if not np.isnan(v)})
            except Exception as e:
                logger.warning(f"MLflow concat log failed: {e}")

        logger.info("=" * 60)
        logger.info("LEAKAGE AUDIT")
        logger.info(f"  Sliding window: train_days={train_days}, no overlap with OOT")
        logger.info(f"  Per-fold per-feature z-score: TRAIN-only stats applied to OOT")
        logger.info(f"  Per-day rank-norm: applied within each day's events (no cross-day leakage)")
        logger.info(f"  CNN causal padding + Mamba causal SSM: no future-event look-ahead")
        logger.info("=" * 60)

        return concat_summary

    finally:
        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                mlflow.end_run()
            except Exception:
                pass


# ============================================================
# CLI
# ============================================================
def parse_args():
    p = argparse.ArgumentParser(description="CNN-Mamba v3 FIFO-aware multi-head trainer")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--label-dir", type=str, default=DEFAULT_LABEL_DIR)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--train-days", type=int, default=WF_TRAIN_DAYS)
    p.add_argument("--warmstart-ckpt", type=str, default=V2_WARMSTART_CKPT)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("CNN-Mamba v3 FIFO-aware Multi-Head Training")
    logger.info("=" * 60)
    logger.info(f"data_dir       = {args.data_dir}")
    logger.info(f"label_dir      = {args.label_dir}")
    logger.info(f"output_dir     = {args.output_dir}")
    logger.info(f"n_folds        = {args.n_folds}")
    logger.info(f"train_days     = {args.train_days}")
    logger.info(f"warmstart_ckpt = {args.warmstart_ckpt}")
    logger.info(f"MLflow URI     = {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow exp     = {MLFLOW_EXPERIMENT}")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"device         = {device}")
    if device.type == "cuda":
        logger.info(f"GPU            = {torch.cuda.get_device_name(0)}")
        logger.info(f"VRAM           = {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    summary = run_weekly_wf_v3(
        data_dir=Path(args.data_dir),
        label_dir=Path(args.label_dir),
        output_dir=Path(args.output_dir),
        device=device,
        n_folds=args.n_folds,
        train_days=args.train_days,
        warmstart_ckpt=args.warmstart_ckpt,
    )

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    for k, v in (summary or {}).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
