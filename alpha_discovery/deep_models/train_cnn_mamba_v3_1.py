"""
CNN-Mamba v3.1 — ALPHA-FIRST Multi-Head Trainer

Per HC #293 (alpha-first heads + new inputs). Replaces strategy-baked heads
(tp4sl3/tp8sl5 net+hit) with path-aware alpha targets.

NEW HEAD SET (23 heads):
  Directional (MSE, λ=1.0):
    log_ret_1s, log_ret_5s, log_ret_10s, log_ret_30s
  Directional prob (BCE, λ=0.5):
    p_up_5s, p_up_10s, p_up_30s
  Quantile (Pinball, λ=0.5):
    log_ret_10s_q10, _q50, _q90, log_ret_30s_q10, _q50, _q90
  Path (Huber, λ=1.0):
    pred_mfe_30s_ticks, pred_mae_30s_ticks
  Time (Huber, λ=0.5):
    pred_time_to_mfe_secs
  Reversal (BCE, λ=0.5):
    p_reversal_15s, p_reversal_30s
  Vol (Huber, λ=0.5):
    pred_realized_vol_30s_ticks
  Legacy aux (λ=0.1, kept per HC #293(F)):
    fifo_tp4sl3_net, fifo_tp8sl5_net, fifo_tp4sl3_hit_tp, fifo_tp8sl5_hit_tp

NEW INPUTS: 25 event features + 3 pt_pred (forward-filled, rank-normed) + 1 has_pt mask
            = 29 features

WARMSTART: v3 fold_00_best.pt (backbone reuses, new heads init fresh, strict=False)
FOLD ANCHOR: 2026-03-09 (first Monday with PatchTST coverage); 10 weekly folds
"""

# Set feature set BEFORE importing v2 trainer
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
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.stats

# v2 backbone (CNN-Mamba)
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
log_path = LOG_DIR / "cnn_mamba_v3_1.log"
_v31_handler = logging.FileHandler(log_path)
_v31_handler.setLevel(logging.INFO)
_v31_handler.setFormatter(logging.Formatter("%(asctime)s [v3.1 %(levelname)s] %(message)s"))
logging.root.addHandler(_v31_handler)
logger = logging.getLogger("cnn_mamba_v3_1")
logger.setLevel(logging.INFO)
print(">>> train_cnn_mamba_v3_1.py loaded", flush=True)


# ============================================================
# Config
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data" / "processed" / "mbo_events_smart_v3")
DEFAULT_FIFO_LABEL_DIR = str(PROJECT_ROOT / "data" / "processed" / "mbo_events_smart_v3_fifo_labels")
DEFAULT_ALPHA_LABEL_DIR = str(PROJECT_ROOT / "data" / "processed" / "mbo_events_smart_v3_alpha_labels")
DEFAULT_PT_PRED_DIR = str(PROJECT_ROOT / "data" / "processed" / "mbo_events_smart_v3_pt_pred")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output" / "cnn_mamba_v3_1_alpha_first")
V3_WARMSTART_CKPT = str(PROJECT_ROOT / "output" / "cnn_mamba_v3_smart_v3_fifo" / "fold_00_best.pt")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")
MLFLOW_EXPERIMENT = "CNNMamba_v3_1_alpha_first"

# Per-day rank-norm feature indices (only the 25 event features; pt_pred is already rank-normed)
RANK_NORM_FEATURE_IDXS = [12, 15, 17]

# Head set (23 heads)
ALPHA_HEAD_NAMES = [
    # Directional regression
    "log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s",
    # Directional probability (BCE on signed labels)
    "p_up_5s", "p_up_10s", "p_up_30s",
    # Quantiles (pinball)
    "log_ret_10s_q10", "log_ret_10s_q50", "log_ret_10s_q90",
    "log_ret_30s_q10", "log_ret_30s_q50", "log_ret_30s_q90",
    # Path
    "pred_mfe_30s_ticks", "pred_mae_30s_ticks",
    # Time
    "pred_time_to_mfe_secs",
    # Reversal
    "p_reversal_15s", "p_reversal_30s",
    # Vol
    "pred_realized_vol_30s_ticks",
]
LEGACY_AUX_HEADS = [
    "fifo_tp4sl3_net", "fifo_tp8sl5_net",
    "fifo_tp4sl3_hit_tp", "fifo_tp8sl5_hit_tp",
]
ALL_HEAD_NAMES = ALPHA_HEAD_NAMES + LEGACY_AUX_HEADS  # 23 total

# Quantile targets: maps head -> (target_head_name, quantile)
QUANTILE_TARGETS = {
    "log_ret_10s_q10": ("log_ret_10s", 0.10),
    "log_ret_10s_q50": ("log_ret_10s", 0.50),
    "log_ret_10s_q90": ("log_ret_10s", 0.90),
    "log_ret_30s_q10": ("log_ret_30s", 0.10),
    "log_ret_30s_q50": ("log_ret_30s", 0.50),
    "log_ret_30s_q90": ("log_ret_30s", 0.90),
}

# Loss weights
LOSS_LAMBDA = {
    # Directional regression
    "log_ret_1s":      1.0,
    "log_ret_5s":      1.0,
    "log_ret_10s":     1.0,
    "log_ret_30s":     1.0,
    # Directional prob
    "p_up_5s":         0.5,
    "p_up_10s":        0.5,
    "p_up_30s":        0.5,
    # Quantiles
    "log_ret_10s_q10": 0.5,
    "log_ret_10s_q50": 0.5,
    "log_ret_10s_q90": 0.5,
    "log_ret_30s_q10": 0.5,
    "log_ret_30s_q50": 0.5,
    "log_ret_30s_q90": 0.5,
    # Path
    "pred_mfe_30s_ticks": 1.0,
    "pred_mae_30s_ticks": 1.0,
    # Time
    "pred_time_to_mfe_secs": 0.5,
    # Reversal
    "p_reversal_15s":  0.5,
    "p_reversal_30s":  0.5,
    # Vol
    "pred_realized_vol_30s_ticks": 0.5,
    # Legacy aux
    "fifo_tp4sl3_net":    0.1,
    "fifo_tp8sl5_net":    0.1,
    "fifo_tp4sl3_hit_tp": 0.1,
    "fifo_tp8sl5_hit_tp": 0.1,
}

FIFO_LABEL_CAP_TICKS = 20.0
# Cap path heads at +/- 50 ticks (typical 30s MFE rarely exceeds; cap prevents gradient blowup)
PATH_LABEL_CAP_TICKS = 50.0
VOL_LABEL_CAP_TICKS = 50.0

# Walk-forward (per HC #293 — anchor 2026-03-09 for full PatchTST coverage)
WF_TRAIN_DAYS = int(os.environ.get("V31_WF_TRAIN_DAYS", 60))
N_FOLDS = int(os.environ.get("V31_N_FOLDS", 10))
FIRST_OOT_MONDAY = os.environ.get("V31_FIRST_OOT_MONDAY", "2026-03-09")

# Total input feature count
N_EVENT_FEATURES = 25
N_PT_FEATURES = 4  # pt_pred_1s/5s/10s + has_pt_pred
N_INPUT_FEATURES = N_EVENT_FEATURES + N_PT_FEATURES  # 29


# ============================================================
# Date utilities
# ============================================================
DATE_RE = re.compile(r"(\d{8})_mbo_events\.npz$")

def date_from_path(p: Path) -> str:
    m = DATE_RE.search(p.name)
    return m.group(1) if m else ""


def build_weekly_fold_schedule(
    available_dates: List[str],
    first_oot_monday: str = FIRST_OOT_MONDAY,
    n_folds: int = N_FOLDS,
    train_days: int = WF_TRAIN_DAYS,
) -> List[Dict]:
    import datetime as _dt
    sorted_dates = sorted(available_dates)
    target_dt = _dt.datetime.strptime(first_oot_monday, "%Y-%m-%d").date()
    target_str = target_dt.strftime("%Y%m%d")

    anchor_idx = None
    for i, d in enumerate(sorted_dates):
        if d >= target_str:
            anchor_idx = i
            break
    if anchor_idx is None:
        logger.error(f"No date >= {first_oot_monday}")
        return []

    folds = []
    cur_idx = anchor_idx
    for fold_n in range(n_folds):
        oot_dates = sorted_dates[cur_idx:cur_idx + 5]
        if len(oot_dates) == 0:
            break
        train_start_idx = max(0, cur_idx - train_days)
        train_dates = sorted_dates[train_start_idx:cur_idx]
        if len(train_dates) < 10:
            logger.warning(f"Fold {fold_n}: only {len(train_dates)} train days, skipping")
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
# Pinball loss
# ============================================================
class PinballLoss(nn.Module):
    """Quantile (pinball) regression loss."""
    def __init__(self, quantile: float):
        super().__init__()
        self.q = quantile

    def forward(self, pred, target):
        diff = target - pred
        # per-elem: max(q*diff, (q-1)*diff)
        return torch.maximum(self.q * diff, (self.q - 1.0) * diff)


# ============================================================
# Multi-Head Wrapper (29 input features, 23 heads)
# ============================================================
class CNNMambaV31(nn.Module):
    HEAD_NAMES = ALL_HEAD_NAMES

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
        n_input_features: int = N_INPUT_FEATURES,
    ):
        super().__init__()
        # Backbone takes (B, L, N_INPUT) — adapter projects pt_pred features into the CNN/Mamba pipeline
        # Strategy: keep backbone same architecture as v3 (which expected 25 input feats — it uses a Linear
        # input projection internally to d_model). Add an input adapter: Linear(29 -> 25) so backbone weights
        # transfer cleanly. The 4 new pt_pred features are mixed into the 25-d projection at adapter level.
        self.input_adapter = nn.Linear(n_input_features, N_EVENT_FEATURES)
        # Initialize adapter so the first 25 features pass through identity, last 4 mix to 0 initially.
        with torch.no_grad():
            W = torch.zeros(N_EVENT_FEATURES, n_input_features)
            W[:N_EVENT_FEATURES, :N_EVENT_FEATURES] = torch.eye(N_EVENT_FEATURES)
            # pt_pred features (idx 25-28): small random init so they have non-zero gradient
            W[:, N_EVENT_FEATURES:] = torch.randn(N_EVENT_FEATURES, N_PT_FEATURES) * 0.01
            self.input_adapter.weight.copy_(W)
            self.input_adapter.bias.zero_()

        self.backbone = CNNMamba(
            d_model=d_model, d_state=d_state, n_layers=n_layers,
            dt_rank=dt_rank, d_conv=d_conv, dropout=dropout,
            n_targets=3,  # head will be discarded
            cnn_channels=cnn_channels, cnn_kernel=cnn_kernel, cnn_layers=cnn_layers,
        )
        self.backbone.head = nn.Identity()

        self.trunk = nn.Sequential(
            nn.Linear(d_model, trunk_dim),
            nn.GELU(),
            nn.LayerNorm(trunk_dim),
            nn.Dropout(dropout),
        )
        self.heads = nn.ModuleDict({
            name: nn.Linear(trunk_dim, 1) for name in self.HEAD_NAMES
        })
        self._init_new_weights()

    def _init_new_weights(self):
        for m in list(self.trunk.modules()) + list(self.heads.modules()):
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, events: torch.Tensor) -> Dict[str, torch.Tensor]:
        # events: (B, L, n_input_features=29)
        x = self.input_adapter(events)  # -> (B, L, 25)
        _, embedding = self.backbone(x, return_embedding=True)
        trunk_out = self.trunk(embedding)
        return {name: head(trunk_out).squeeze(-1) for name, head in self.heads.items()}

    def load_v3_warmstart(self, ckpt_path: str, device: torch.device) -> bool:
        """Load v3 fold_00_best.pt: copy backbone + trunk; skip heads (different set)."""
        if not Path(ckpt_path).exists():
            logger.warning(f"v3 warmstart not found: {ckpt_path} — training fresh")
            return False
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            v3_state = ckpt.get("model_state", ckpt)
            own_state = self.state_dict()
            loaded = 0
            skipped = 0
            for k, v in v3_state.items():
                # Heads are different in v3.1 — skip them
                if k.startswith("heads."):
                    skipped += 1
                    continue
                # input_adapter doesn't exist in v3 — skip (will use init values)
                if k.startswith("input_adapter."):
                    skipped += 1
                    continue
                if k in own_state and own_state[k].shape == v.shape:
                    own_state[k] = v
                    loaded += 1
                else:
                    skipped += 1
            self.load_state_dict(own_state, strict=False)
            logger.info(
                f"v3 warmstart: {loaded} tensors copied, {skipped} skipped "
                f"(from {Path(ckpt_path).name}, val_loss={ckpt.get('val_loss','?')})"
            )
            return True
        except Exception as e:
            logger.warning(f"v3 warmstart failed: {e}")
            return False


# ============================================================
# Joint Multi-Head Loss (handles MSE / BCE / Huber / Pinball)
# ============================================================
class JointMultiHeadLossV31(nn.Module):
    def __init__(self, lambdas: Dict[str, float] = None, huber_delta: float = 2.0):
        super().__init__()
        self.lambdas = lambdas or LOSS_LAMBDA
        self.huber = nn.HuberLoss(reduction="none", delta=huber_delta)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.pinball = {
            name: PinballLoss(QUANTILE_TARGETS[name][1])
            for name in QUANTILE_TARGETS
        }

    def forward(
        self,
        preds: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        masks: Dict[str, torch.Tensor],
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        components = {}
        total = 0.0
        for name, lam in self.lambdas.items():
            if lam == 0.0 or name not in preds:
                continue
            p = preds[name]

            # Resolve target (quantile heads use parent log_ret target)
            if name in QUANTILE_TARGETS:
                tgt_name, _ = QUANTILE_TARGETS[name]
                t = targets[tgt_name]
                m = masks.get(tgt_name, torch.ones_like(p))
                loss_per = self.pinball[name](p, t)
            elif name.endswith("_hit_tp") or name.startswith("p_up") or name.startswith("p_reversal"):
                t = targets[name]
                m = masks.get(name, torch.ones_like(p))
                loss_per = self.bce(p, t)
            elif (name.startswith("pred_") or name.endswith("_net") or
                  name == "pred_realized_vol_30s_ticks" or name.endswith("_ticks") or
                  name == "pred_time_to_mfe_secs"):
                t = targets[name]
                m = masks.get(name, torch.ones_like(p))
                loss_per = self.huber(p, t)
            elif name.startswith("log_ret"):
                t = targets[name]
                m = masks.get(name, torch.ones_like(p))
                loss_per = self.mse(p, t)
            else:
                continue

            denom = m.sum().clamp_min(1.0)
            loss = (loss_per * m).sum() / denom
            components[name] = float(loss.item())
            total = total + lam * loss

        return total, components


# ============================================================
# Dataset
# ============================================================
class SmartV31Dataset(Dataset):
    HEAD_NAMES = ALL_HEAD_NAMES

    def __init__(
        self,
        data_dir: Path,
        fifo_label_dir: Path,
        alpha_label_dir: Path,
        pt_pred_dir: Path,
        dates: List[str],
        window_size: int = WINDOW_SIZE,
        stride: int = STRIDE,
        feature_stats: Optional[Dict] = None,
        cache_size: int = 4,
        require_alpha_labels: bool = True,
    ):
        self.data_dir = Path(data_dir)
        self.fifo_label_dir = Path(fifo_label_dir)
        self.alpha_label_dir = Path(alpha_label_dir)
        self.pt_pred_dir = Path(pt_pred_dir)
        self.dates = dates
        self.window_size = window_size
        self.stride = stride
        self.cache_size = cache_size
        self.require_alpha_labels = require_alpha_labels
        self.feature_stats = feature_stats

        self._cache: Dict[str, Dict] = {}
        self._cache_order: List[str] = []
        self.sample_index: List[Tuple[str, int, int]] = []
        self._build_index()
        if self.feature_stats is None:
            self._compute_feature_stats()

    def _load_day_raw(self, date_str: str) -> Optional[Dict]:
        events_path = self.data_dir / f"{date_str}_mbo_events.npz"
        alpha_path = self.alpha_label_dir / f"{date_str}_alpha_labels.npz"
        pt_path = self.pt_pred_dir / f"{date_str}_pt_pred_event_aligned.npz"
        fifo_path = self.fifo_label_dir / f"{date_str}_fifo_labels.npz"

        if not events_path.exists():
            return None
        if self.require_alpha_labels and not alpha_path.exists():
            logger.warning(f"alpha labels missing for {date_str} — skipping")
            return None

        ev = np.load(events_path, allow_pickle=True)
        events_25 = ev["events"].astype(np.float32)  # (N, 25)
        labels_1s = ev["labels_1s"].astype(np.float32)
        labels_5s = ev["labels_5s"].astype(np.float32)
        labels_10s = ev["labels_10s"].astype(np.float32)
        labels_30s = ev["labels_30s"].astype(np.float32) if "labels_30s" in ev else None

        # Per-day rank-norm on event features
        n = len(events_25)
        if n > 1:
            for fidx in RANK_NORM_FEATURE_IDXS:
                col = events_25[:, fidx]
                ranks = scipy.stats.rankdata(col, method="average") - 1.0
                events_25[:, fidx] = ((ranks / max(n - 1, 1)) - 0.5) * 2.0

        # Load pt_pred (forward-filled, already rank-normed)
        if pt_path.exists():
            pt = np.load(pt_path, allow_pickle=True)
            pt_1s = pt["pt_pred_1s"].astype(np.float32)
            pt_5s = pt["pt_pred_5s"].astype(np.float32)
            pt_10s = pt["pt_pred_10s"].astype(np.float32)
            has_pt = pt["has_pt_pred"].astype(np.float32)
        else:
            pt_1s = np.zeros(n, dtype=np.float32)
            pt_5s = np.zeros(n, dtype=np.float32)
            pt_10s = np.zeros(n, dtype=np.float32)
            has_pt = np.zeros(n, dtype=np.float32)

        # Concatenate to 29-dim events (25 + 4)
        events_29 = np.concatenate([
            events_25,
            pt_1s.reshape(-1, 1),
            pt_5s.reshape(-1, 1),
            pt_10s.reshape(-1, 1),
            has_pt.reshape(-1, 1),
        ], axis=1).astype(np.float32)

        # Load alpha labels
        alpha = {}
        if alpha_path.exists():
            al = np.load(alpha_path, allow_pickle=True)
            alpha = {k: al[k].astype(np.float32) for k in al.keys()}

        # Load FIFO labels (legacy aux)
        fifo = {}
        if fifo_path.exists():
            fl = np.load(fifo_path, allow_pickle=True)
            fifo = {
                "window_k": fl["window_k"].astype(np.int64),
                "tp4sl3_short_net": fl["tp4sl3_short_net_ticks"].astype(np.float32),
                "tp4sl3_short_filled": fl["tp4sl3_short_filled"].astype(np.bool_),
                "tp4sl3_short_hit_tp": fl["tp4sl3_short_hit_tp"].astype(np.bool_),
                "tp8sl5_short_net": fl["tp8sl5_short_net_ticks"].astype(np.float32),
                "tp8sl5_short_filled": fl["tp8sl5_short_filled"].astype(np.bool_),
                "tp8sl5_short_hit_tp": fl["tp8sl5_short_hit_tp"].astype(np.bool_),
            }
            np.clip(fifo["tp4sl3_short_net"], -FIFO_LABEL_CAP_TICKS, FIFO_LABEL_CAP_TICKS,
                    out=fifo["tp4sl3_short_net"])
            np.clip(fifo["tp8sl5_short_net"], -FIFO_LABEL_CAP_TICKS, FIFO_LABEL_CAP_TICKS,
                    out=fifo["tp8sl5_short_net"])
            fifo["wk_to_row"] = {int(k): i for i, k in enumerate(fifo["window_k"])}

        return {
            "events": events_29,
            "labels_1s": labels_1s,
            "labels_5s": labels_5s,
            "labels_10s": labels_10s,
            "labels_30s": labels_30s if labels_30s is not None else alpha.get("log_ret_30s"),
            "alpha": alpha,
            "fifo": fifo,
        }

    def _get_day(self, date_str: str) -> Optional[Dict]:
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
        for date_str in self.dates:
            events_path = self.data_dir / f"{date_str}_mbo_events.npz"
            alpha_path = self.alpha_label_dir / f"{date_str}_alpha_labels.npz"
            if not events_path.exists():
                continue
            if self.require_alpha_labels and not alpha_path.exists():
                logger.warning(f"alpha labels missing for {date_str} — skipping")
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
        logger.info(f"Computing feature stats from {len(self.dates)} train dates...")
        n_features = N_INPUT_FEATURES
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
            self.feature_stats = {
                "mean": np.zeros(n_features, dtype=np.float32),
                "std": np.ones(n_features, dtype=np.float32),
            }
            return
        mean = (total_sum / total_count).astype(np.float32)
        var = (total_sq / total_count) - (mean.astype(np.float64) ** 2)
        std = np.sqrt(np.maximum(var, 1e-8)).astype(np.float32)
        self.feature_stats = {"mean": mean, "std": std}
        logger.info(f"Feature stats: n={total_count} events, {n_features} features")

    def get_feature_stats(self) -> Dict:
        return self.feature_stats

    def __len__(self) -> int:
        return len(self.sample_index)

    def __getitem__(self, idx: int):
        date_str, start, window_k = self.sample_index[idx]
        end = start + self.window_size
        day = self._get_day(date_str)
        if day is None:
            raise RuntimeError(f"Failed to load {date_str}")
        events = day["events"][start:end]  # (W, 29)
        if self.feature_stats is not None:
            events = (events - self.feature_stats["mean"]) / (self.feature_stats["std"] + 1e-8)

        label_idx = end - 1
        # Pull alpha + log_ret + fifo labels at label_idx
        alpha = day.get("alpha", {})
        lab1 = day["labels_1s"][label_idx]
        lab5 = day["labels_5s"][label_idx]
        lab10 = day["labels_10s"][label_idx]
        labs30_arr = day.get("labels_30s")
        lab30 = labs30_arr[label_idx] if labs30_arr is not None else np.nan

        def _safe(v, default=0.0):
            return float(default) if (v is None or np.isnan(v)) else float(v)

        def _alpha_at(k):
            arr = alpha.get(k)
            if arr is None or label_idx >= len(arr):
                return np.nan
            return float(arr[label_idx])

        targets = {}
        masks = {}

        # Directional regression (raw ticks; the model learns scale from data)
        for k_name, v in (("log_ret_1s", lab1), ("log_ret_5s", lab5),
                          ("log_ret_10s", lab10), ("log_ret_30s", lab30)):
            if v is None or np.isnan(v):
                targets[k_name] = np.float32(0.0)
                masks[k_name] = np.float32(0.0)
            else:
                # Cap to prevent gradient blowup (vol heads can exceed PATH_LABEL_CAP)
                cap = PATH_LABEL_CAP_TICKS if "30s" in k_name else 100.0
                targets[k_name] = np.float32(np.clip(v, -cap, cap))
                masks[k_name] = np.float32(1.0)

        # Directional probability (sign of log_ret)
        for k_name, lr in (("p_up_5s", lab5), ("p_up_10s", lab10), ("p_up_30s", lab30)):
            if lr is None or np.isnan(lr):
                targets[k_name] = np.float32(0.0)
                masks[k_name] = np.float32(0.0)
            else:
                targets[k_name] = np.float32(1.0 if lr > 0 else 0.0)
                masks[k_name] = np.float32(1.0)

        # Quantile heads: target is parent log_ret, mask=1 (same mask as parent regression)
        for q_name, (parent, _) in QUANTILE_TARGETS.items():
            targets[q_name] = targets[parent]
            masks[q_name] = masks[parent]

        # Path heads
        mfe = _alpha_at("mfe_30s_ticks")
        mae = _alpha_at("mae_30s_ticks")
        if np.isnan(mfe):
            targets["pred_mfe_30s_ticks"] = np.float32(0.0); masks["pred_mfe_30s_ticks"] = np.float32(0.0)
        else:
            targets["pred_mfe_30s_ticks"] = np.float32(np.clip(mfe, -PATH_LABEL_CAP_TICKS, PATH_LABEL_CAP_TICKS))
            masks["pred_mfe_30s_ticks"] = np.float32(1.0)
        if np.isnan(mae):
            targets["pred_mae_30s_ticks"] = np.float32(0.0); masks["pred_mae_30s_ticks"] = np.float32(0.0)
        else:
            targets["pred_mae_30s_ticks"] = np.float32(np.clip(mae, -PATH_LABEL_CAP_TICKS, PATH_LABEL_CAP_TICKS))
            masks["pred_mae_30s_ticks"] = np.float32(1.0)

        # Time to MFE
        tmfe = _alpha_at("time_to_mfe_secs")
        if np.isnan(tmfe):
            targets["pred_time_to_mfe_secs"] = np.float32(0.0); masks["pred_time_to_mfe_secs"] = np.float32(0.0)
        else:
            targets["pred_time_to_mfe_secs"] = np.float32(tmfe)
            masks["pred_time_to_mfe_secs"] = np.float32(1.0)

        # Reversal
        for k_name, ak in (("p_reversal_15s", "p_reversal_15s"),
                           ("p_reversal_30s", "p_reversal_30s")):
            v = _alpha_at(ak)
            if np.isnan(v):
                targets[k_name] = np.float32(0.0); masks[k_name] = np.float32(0.0)
            else:
                targets[k_name] = np.float32(v); masks[k_name] = np.float32(1.0)

        # Vol
        vol = _alpha_at("realized_vol_30s_ticks")
        if np.isnan(vol):
            targets["pred_realized_vol_30s_ticks"] = np.float32(0.0)
            masks["pred_realized_vol_30s_ticks"] = np.float32(0.0)
        else:
            targets["pred_realized_vol_30s_ticks"] = np.float32(np.clip(vol, 0.0, VOL_LABEL_CAP_TICKS))
            masks["pred_realized_vol_30s_ticks"] = np.float32(1.0)

        # Legacy FIFO aux (kept low weight per HC #293(F))
        for k_name in LEGACY_AUX_HEADS:
            targets[k_name] = np.float32(0.0)
            masks[k_name] = np.float32(0.0)
        fifo = day.get("fifo", {})
        if fifo:
            row = fifo.get("wk_to_row", {}).get(int(window_k), -1)
            if row >= 0:
                if fifo["tp4sl3_short_filled"][row]:
                    targets["fifo_tp4sl3_net"] = np.float32(fifo["tp4sl3_short_net"][row])
                    masks["fifo_tp4sl3_net"] = np.float32(1.0)
                    targets["fifo_tp4sl3_hit_tp"] = np.float32(1.0 if fifo["tp4sl3_short_hit_tp"][row] else 0.0)
                    masks["fifo_tp4sl3_hit_tp"] = np.float32(1.0)
                if fifo["tp8sl5_short_filled"][row]:
                    targets["fifo_tp8sl5_net"] = np.float32(fifo["tp8sl5_short_net"][row])
                    masks["fifo_tp8sl5_net"] = np.float32(1.0)
                    targets["fifo_tp8sl5_hit_tp"] = np.float32(1.0 if fifo["tp8sl5_short_hit_tp"][row] else 0.0)
                    masks["fifo_tp8sl5_hit_tp"] = np.float32(1.0)

        events_t = torch.from_numpy(events.astype(np.float32))
        targets_t = {k: torch.tensor(v) for k, v in targets.items()}
        masks_t = {k: torch.tensor(v) for k, v in masks.items()}
        return events_t, targets_t, masks_t


def collate_v31(batch):
    events_list, targets_list, masks_list = zip(*batch)
    events = torch.stack(events_list, dim=0)
    targets = {k: torch.stack([t[k] for t in targets_list], dim=0) for k in targets_list[0]}
    masks = {k: torch.stack([m[k] for m in masks_list], dim=0) for k in masks_list[0]}
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


def compute_auc(preds: np.ndarray, labels: np.ndarray) -> float:
    """Simple AUC via rank-sum (Mann-Whitney) — no sklearn dep needed."""
    valid = ~(np.isnan(preds) | np.isnan(labels))
    p = preds[valid]; y = labels[valid]
    if len(p) < 20:
        return float("nan")
    pos = (y > 0.5)
    n_pos = int(pos.sum()); n_neg = int((~pos).sum())
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    ranks = scipy.stats.rankdata(p)
    rank_sum_pos = ranks[pos].sum()
    u = rank_sum_pos - n_pos * (n_pos + 1) / 2.0
    auc = u / (n_pos * n_neg)
    return float(auc)


def evaluate_v31(
    model: nn.Module,
    loader: DataLoader,
    loss_fn: nn.Module,
    device: torch.device,
    use_amp: bool = True,
):
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

    # IC for log_ret heads
    for h in ("log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s"):
        if h in masks_out:
            m = masks_out[h] > 0
            metrics[f"ic_{h}"] = compute_ic(preds_out[h][m], targets_out[h][m]) if m.sum() > 20 else float("nan")

    # AUC for BCE heads
    for h in ("p_up_5s", "p_up_10s", "p_up_30s", "p_reversal_15s", "p_reversal_30s",
              "fifo_tp4sl3_hit_tp", "fifo_tp8sl5_hit_tp"):
        if h in masks_out:
            m = masks_out[h] > 0
            metrics[f"auc_{h}"] = compute_auc(preds_out[h][m], targets_out[h][m]) if m.sum() > 20 else float("nan")

    # MAE for path heads
    for h in ("pred_mfe_30s_ticks", "pred_mae_30s_ticks", "pred_time_to_mfe_secs",
              "pred_realized_vol_30s_ticks"):
        if h in masks_out:
            m = masks_out[h] > 0
            if m.sum() > 20:
                metrics[f"mae_{h}"] = float(np.mean(np.abs(preds_out[h][m] - targets_out[h][m])))
                metrics[f"corr_{h}"] = compute_ic(preds_out[h][m], targets_out[h][m])

    # Quantile coverage: % of actuals between q10 and q90
    for horizon in ("10s", "30s"):
        q10_name = f"log_ret_{horizon}_q10"
        q90_name = f"log_ret_{horizon}_q90"
        tgt_name = f"log_ret_{horizon}"
        if q10_name in preds_out and q90_name in preds_out:
            t = targets_out[tgt_name]
            q10 = preds_out[q10_name]
            q90 = preds_out[q90_name]
            m = masks_out[tgt_name] > 0
            if m.sum() > 20:
                inside = ((t[m] >= q10[m]) & (t[m] <= q90[m])).mean()
                metrics[f"q_coverage_{horizon}_80pct"] = float(inside)

    return metrics, preds_out, targets_out, masks_out


# ============================================================
# Train one fold
# ============================================================
def train_one_fold_v31(
    model: CNNMambaV31,
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
    scheduler = WarmupCosineScheduler(optimizer, warmup_steps=WARMUP_STEPS, total_steps=total_train_steps)
    loss_fn = JointMultiHeadLossV31(LOSS_LAMBDA)

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    best_val_loss = float("inf")
    global_step = 0
    total_batches = len(train_loader)
    print(f">>> v3.1 train_one_fold {fold_idx}: {EPOCHS_PER_FOLD} ep × {total_batches} batches", flush=True)

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
                    f"  Fold {fold_idx} Ep {epoch+1} B {n_batches}/{total_batches} | "
                    f"Loss {epoch_loss/n_batches:.4f} | Elapsed {elapsed:.0f}s | ETA {eta:.0f}s"
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

        val_metrics, _, _, _ = evaluate_v31(model, oot_loader, loss_fn, device, use_amp=use_amp)
        msg = (
            f"Fold {fold_idx:02d} Ep {epoch+1}/{EPOCHS_PER_FOLD} | "
            f"TrLoss {avg_loss:.4f} | OOT Loss {val_metrics['loss']:.4f} | "
            f"IC 1s/5s/10s/30s = "
            f"{val_metrics.get('ic_log_ret_1s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_5s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_10s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_30s', float('nan')):.4f} | "
            f"AUC p_up_10s={val_metrics.get('auc_p_up_10s', float('nan')):.3f} "
            f"AUC rev_30s={val_metrics.get('auc_p_reversal_30s', float('nan')):.3f} | "
            f"corr_mfe={val_metrics.get('corr_pred_mfe_30s_ticks', float('nan')):.3f} | "
            f"LR {scheduler.get_lr():.2e} | T {epoch_time:.1f}s"
        )
        print(msg, flush=True)
        logger.info(msg)

        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                step = fold_idx * EPOCHS_PER_FOLD + epoch
                m2log = {
                    f"f{fold_idx:02d}_train_loss": avg_loss,
                    f"f{fold_idx:02d}_oot_loss": val_metrics["loss"],
                }
                for k, v in val_metrics.items():
                    if k != "loss" and not np.isnan(v):
                        m2log[f"f{fold_idx:02d}_{k}"] = float(v)
                for k, v in comp_avg.items():
                    m2log[f"f{fold_idx:02d}_train_loss_{k}"] = float(v)
                mlflow.log_metrics(m2log, step=step)
            except Exception as e:
                logger.warning(f"MLflow log failed: {e}")

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "model_state": model.state_dict(),
                "fold": fold_idx, "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "val_metrics": val_metrics,
                "arch": {
                    "model": "CNNMambaV31",
                    "d_model": MAMBA_D_MODEL, "d_state": MAMBA_D_STATE,
                    "n_layers": MAMBA_N_LAYERS, "dt_rank": MAMBA_DT_RANK,
                    "d_conv": MAMBA_D_CONV, "dropout": MAMBA_DROPOUT,
                    "cnn_channels": CNN_CHANNELS, "cnn_kernel": CNN_KERNEL,
                    "cnn_layers": CNN_LAYERS, "window_size": WINDOW_SIZE,
                    "trunk_dim": 128, "head_names": ALL_HEAD_NAMES,
                    "n_input_features": N_INPUT_FEATURES,
                },
            }, output_dir / f"fold_{fold_idx:02d}_best.pt")

    return {"best_val_loss": best_val_loss}


# ============================================================
# Walk-forward driver
# ============================================================
def run_weekly_wf_v31(
    data_dir: Path,
    fifo_label_dir: Path,
    alpha_label_dir: Path,
    pt_pred_dir: Path,
    output_dir: Path,
    device: torch.device,
    n_folds: int = N_FOLDS,
    train_days: int = WF_TRAIN_DAYS,
    warmstart_ckpt: str = V3_WARMSTART_CKPT,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("*_mbo_events.npz"))
    available_dates = sorted([date_from_path(p) for p in npz_files if date_from_path(p)])
    if not available_dates:
        logger.error(f"No smart_v3 NPZs in {data_dir}")
        return {}
    logger.info(f"Found {len(available_dates)} smart_v3 dates")

    # Filter to dates with alpha labels (required)
    available_dates = [d for d in available_dates if (alpha_label_dir / f"{d}_alpha_labels.npz").exists()]
    logger.info(f"Dates with alpha labels: {len(available_dates)}")

    folds = build_weekly_fold_schedule(available_dates, n_folds=n_folds, train_days=train_days)
    logger.info(f"Built {len(folds)} folds (anchor={FIRST_OOT_MONDAY})")
    for f in folds:
        logger.info(f"  Fold {f['fold']}: train {f['train_start']}→{f['train_end']} ({len(f['train_dates'])}d) "
                    f"OOT {f['oot_start']}→{f['oot_end']} ({len(f['oot_dates'])}d)")

    with open(output_dir / "fold_schedule.json", "w") as fh:
        json.dump(folds, fh, indent=2, default=str)

    use_amp = device.type == "cuda"

    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(run_name=f"v3_1_{time.strftime('%Y%m%d_%H%M')}_{socket.gethostname()}")
            gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
            mlflow.log_params({
                "model": "CNNMambaV31",
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
                "n_input_features": N_INPUT_FEATURES,
                "n_event_features": N_EVENT_FEATURES,
                "n_pt_features": N_PT_FEATURES,
                "n_heads": len(ALL_HEAD_NAMES),
                "head_names": ",".join(ALL_HEAD_NAMES),
                "loss_lambdas": json.dumps(LOSS_LAMBDA),
                "warmstart_ckpt": warmstart_ckpt,
                "node": socket.gethostname(),
                "gpu": gpu_name,
                "data_dir": str(data_dir),
                "alpha_label_dir": str(alpha_label_dir),
                "pt_pred_dir": str(pt_pred_dir),
                "output_dir": str(output_dir),
            })
            logger.info(f"MLflow run started: {mlflow_run.info.run_id}")
        except Exception as e:
            logger.warning(f"MLflow init failed: {e}")
            mlflow_run = None

    concat_results = {h: {"preds": [], "targets": [], "masks": []} for h in ALL_HEAD_NAMES}

    try:
        start_fold = int(os.environ.get("V31_START_FOLD", 0))
        for f_info in folds:
            fold_idx = f_info["fold"]
            if fold_idx < start_fold:
                continue
            logger.info("=" * 60)
            logger.info(f"FOLD {fold_idx} train {f_info['train_start']}→{f_info['train_end']} "
                        f"OOT {f_info['oot_start']}→{f_info['oot_end']}")
            logger.info("=" * 60)

            train_ds = SmartV31Dataset(
                data_dir=data_dir, fifo_label_dir=fifo_label_dir,
                alpha_label_dir=alpha_label_dir, pt_pred_dir=pt_pred_dir,
                dates=f_info["train_dates"],
                window_size=WINDOW_SIZE, stride=STRIDE,
                feature_stats=None, cache_size=4,
                require_alpha_labels=True,
            )
            feature_stats = train_ds.get_feature_stats()
            np.savez(
                output_dir / f"fold_{fold_idx:02d}_feature_stats.npz",
                mean=feature_stats["mean"], std=feature_stats["std"],
            )

            oot_ds = SmartV31Dataset(
                data_dir=data_dir, fifo_label_dir=fifo_label_dir,
                alpha_label_dir=alpha_label_dir, pt_pred_dir=pt_pred_dir,
                dates=f_info["oot_dates"],
                window_size=WINDOW_SIZE, stride=STRIDE,
                feature_stats=feature_stats, cache_size=4,
                require_alpha_labels=True,
            )

            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=2, pin_memory=True, drop_last=True,
                persistent_workers=True, prefetch_factor=4,
                collate_fn=collate_v31,
            )
            oot_loader = DataLoader(
                oot_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                num_workers=1, pin_memory=True,
                persistent_workers=True, prefetch_factor=4,
                collate_fn=collate_v31,
            )

            model = CNNMambaV31().to(device)
            if fold_idx == 0:
                logger.info(f"Model params: {count_parameters(model):,}")
            model.load_v3_warmstart(warmstart_ckpt, device)

            total_steps = EPOCHS_PER_FOLD * len(train_loader)
            train_one_fold_v31(
                model, train_loader, oot_loader,
                fold_idx, output_dir, mlflow_run, device,
                total_train_steps=total_steps, use_amp=use_amp,
            )

            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                logger.info(f"Loaded best (val_loss={ckpt['val_loss']:.4f})")

            loss_fn = JointMultiHeadLossV31(LOSS_LAMBDA)
            metrics, preds, targets, masks = evaluate_v31(
                model, oot_loader, loss_fn, device, use_amp=use_amp,
            )
            logger.info(
                f"Fold {fold_idx} OOT FINAL | "
                f"IC 1s={metrics.get('ic_log_ret_1s', float('nan')):.4f} | "
                f"IC 5s={metrics.get('ic_log_ret_5s', float('nan')):.4f} | "
                f"IC 10s={metrics.get('ic_log_ret_10s', float('nan')):.4f} | "
                f"IC 30s={metrics.get('ic_log_ret_30s', float('nan')):.4f}"
            )

            save_dict = {
                "fold_idx": np.array(fold_idx),
                "oot_dates": np.array(f_info["oot_dates"]),
            }
            for h in ALL_HEAD_NAMES:
                save_dict[f"pred_{h}"] = preds[h]
                save_dict[f"target_{h}"] = targets[h]
                save_dict[f"mask_{h}"] = masks[h]
            np.savez_compressed(output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz", **save_dict)

            for h in ALL_HEAD_NAMES:
                concat_results[h]["preds"].append(preds[h])
                concat_results[h]["targets"].append(targets[h])
                concat_results[h]["masks"].append(masks[h])

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
            if device.type == "cuda":
                torch.cuda.empty_cache()

        logger.info("=" * 60)
        logger.info("CONCAT RESULTS")
        logger.info("=" * 60)
        concat_summary = {}
        for h in ALL_HEAD_NAMES:
            if not concat_results[h]["preds"]:
                continue
            p = np.concatenate(concat_results[h]["preds"])
            t = np.concatenate(concat_results[h]["targets"])
            m = np.concatenate(concat_results[h]["masks"])
            valid = m > 0
            if h.startswith("log_ret") and "_q" not in h:
                ic = compute_ic(p[valid], t[valid])
                concat_summary[f"concat_ic_{h}"] = ic
                logger.info(f"  Concat IC {h}: {ic:.4f}  (n={int(valid.sum())})")
            elif h.startswith("p_up") or h.startswith("p_reversal") or h.endswith("_hit_tp"):
                auc = compute_auc(p[valid], t[valid])
                concat_summary[f"concat_auc_{h}"] = auc
                logger.info(f"  Concat AUC {h}: {auc:.4f}")
            elif h.startswith("pred_") or h.endswith("_ticks") or h.endswith("_net") or h.endswith("_secs"):
                corr = compute_ic(p[valid], t[valid])
                concat_summary[f"concat_corr_{h}"] = corr
                logger.info(f"  Concat corr {h}: {corr:.4f}")

        with open(output_dir / "concat_summary.json", "w") as fh:
            json.dump({k: (float(v) if not np.isnan(v) else None) for k, v in concat_summary.items()}, fh, indent=2)

        if MLFLOW_AVAILABLE and mlflow_run is not None:
            try:
                mlflow.log_metrics({k: float(v) for k, v in concat_summary.items() if not np.isnan(v)})
            except Exception as e:
                logger.warning(f"MLflow concat log failed: {e}")

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
    p = argparse.ArgumentParser(description="CNN-Mamba v3.1 alpha-first trainer")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--fifo-label-dir", type=str, default=DEFAULT_FIFO_LABEL_DIR)
    p.add_argument("--alpha-label-dir", type=str, default=DEFAULT_ALPHA_LABEL_DIR)
    p.add_argument("--pt-pred-dir", type=str, default=DEFAULT_PT_PRED_DIR)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--train-days", type=int, default=WF_TRAIN_DAYS)
    p.add_argument("--warmstart-ckpt", type=str, default=V3_WARMSTART_CKPT)
    p.add_argument("--device", type=str, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("CNN-Mamba v3.1 ALPHA-FIRST Multi-Head Training")
    logger.info("=" * 60)
    logger.info(f"data_dir         = {args.data_dir}")
    logger.info(f"fifo_label_dir   = {args.fifo_label_dir}")
    logger.info(f"alpha_label_dir  = {args.alpha_label_dir}")
    logger.info(f"pt_pred_dir      = {args.pt_pred_dir}")
    logger.info(f"output_dir       = {args.output_dir}")
    logger.info(f"warmstart_ckpt   = {args.warmstart_ckpt}")
    logger.info(f"first_oot_monday = {FIRST_OOT_MONDAY}")
    logger.info(f"MLflow URI       = {MLFLOW_TRACKING_URI}")
    logger.info(f"MLflow exp       = {MLFLOW_EXPERIMENT}")

    if args.device:
        device = torch.device(args.device)
    elif torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    logger.info(f"device           = {device}")
    if device.type == "cuda":
        logger.info(f"GPU              = {torch.cuda.get_device_name(0)}")

    summary = run_weekly_wf_v31(
        data_dir=Path(args.data_dir),
        fifo_label_dir=Path(args.fifo_label_dir),
        alpha_label_dir=Path(args.alpha_label_dir),
        pt_pred_dir=Path(args.pt_pred_dir),
        output_dir=Path(args.output_dir),
        device=device,
        n_folds=args.n_folds,
        train_days=args.train_days,
        warmstart_ckpt=args.warmstart_ckpt,
    )

    logger.info("=" * 60)
    logger.info("FINAL")
    for k, v in (summary or {}).items():
        logger.info(f"  {k}: {v}")


if __name__ == "__main__":
    main()
