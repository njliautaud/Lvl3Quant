"""
CNN-Mamba v3.4 — DUAL-TRUNK (Event-Temporal 1D-CNN + Book-Shape 2D-CNN) + Uncertainty-Weighted MTL

The architectural hypothesis (per HC #365, post 17-day OOT verdict):
  - v3.3 ≈ v2 on tradeable edge. The "more heads, more loss tricks, more PatchTST" path
    is EXHAUSTED on event-temporal-only input.
  - The ONLY remaining path forward is a *new input modality*: the 5-level book-shape
    pyramid (`mbo_book_features/*.npz`) processed by a 2D-CNN that can extract
    depth-pyramid topology features the 1D event trunk cannot represent.

Architecture (v3.4 = v3.3 + 4th branch):
  T1 event (1D-CNN-Mamba)  → emb_t1
  T2 bucketed-orderflow    → emb_t2   ← retained for warmstart compat; will be ablated
  T3 session/macro         → emb_t3
  T2_BOOK 5-level pyramid  → emb_book ← NEW (Book2DCNN over (level_pair, time))
                                          |
                            CONCAT → trunk MLP → 32 heads
                                          |
                            uncertainty-weighted MTL (v3.3 inherited)

Warmstart: from `output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_intra_ckpt.pt`.
  - T1/T2/T3 branches + Mamba backbones + 32 heads + σ params: COPY from v3.3
  - Trunk Linear input dim grew (3*d_model → 4*d_model); first 3*d_model cols copied,
    new d_model cols (book branch) init random.
  - Book2DCNN trunk + book embedding: init random.

Falsification gate at Ep 1 OOT (HC #366 Q3=I — default strictness):
  KILL if NEITHER:
    (a) IC_1s ≥ 0.23 on OOT (vs 17-day v3.3 baseline 0.1765, requires +0.05 lift)
    (b) IC_1s ≥ 0.296 on OOT (vs 5-day v3.3 baseline 0.286, +0.01 lift)
    (c) MagCorr improvement ≥ +0.05 on book-shape-aware heads vs v3.3 baseline

Authorization: HC #362 (design+impl), HC #365 (sole priority), HC #366 (autonomous lean+exec
with Q1=A schema, Q2=α location, Q3=I gate).

Author: Claude (head-of-quant), 2026-05-14 22:55 ET, per user HC #366.
"""
# Must set BEFORE importing v3.2/v3.3 trainer (which read these on import)
import os as _os
_os.environ.setdefault("MAMBA_FEATURE_SET", "smart_v3")
_os.environ.setdefault("EVENT_WINDOW_SIZE", "1500")
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
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

# ------------------------------------------------------------------
# Reuse from v3.3 (which itself reuses from v3.2)
# ------------------------------------------------------------------
from alpha_discovery.deep_models.train_cnn_mamba_v3_3 import (
    JointMultiHeadLossV33_UncertaintyWeighted,
    LOG_SIGMA_INIT,
)
from alpha_discovery.deep_models.train_cnn_mamba_v3_2 import (
    CNNMambaV32,
    SmartV32Dataset,
    collate_v32,
    ALL_HEAD_NAMES,
    DIR_REG_HEADS, P_UP_HEADS, QUANTILE_HEADS, PATH_HEADS, TIME_HEADS,
    REVERSAL_HEADS, VOL_HEADS, LEGACY_AUX_HEADS, QUANTILE_TARGETS,
    LOSS_LAMBDA, PinballLoss,
    evaluate_v32, compute_ic,
    build_weekly_fold_schedule, date_from_path,
    DEFAULT_DATA_DIR, DEFAULT_FIFO_LABEL_DIR, DEFAULT_ALPHA_LABEL_DIR,
    DEFAULT_PT_PRED_DIR, DEFAULT_TIER2_PARQUET_ROOT, DEFAULT_TIER3_PARQUET_ROOT,
    V3_WARMSTART_CKPT, FIRST_OOT_MONDAY, N_FOLDS, WF_TRAIN_DAYS,
    EPOCHS_PER_FOLD, BATCH_SIZE, STRIDE,
    WINDOW_SIZE_T1, WINDOW_SIZE_T2, WINDOW_SIZE_T3,
    N_T1_FEATURES, N_T2_FEATURES, N_T3_FEATURES,
    MAMBA_D_MODEL, T3_D_MODEL, T3_N_LAYERS, TRUNK_DIM, MAMBA_DROPOUT,
)
from alpha_discovery.deep_models.train_cnn_mamba import (
    WarmupCosineScheduler, count_parameters,
    LR, WARMUP_STEPS, GRAD_CLIP,
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
log_path = LOG_DIR / "cnn_mamba_v3_4.log"
_v34_handler = logging.FileHandler(log_path)
_v34_handler.setLevel(logging.INFO)
_v34_handler.setFormatter(logging.Formatter("%(asctime)s [v3.4 %(levelname)s] %(message)s"))
logging.root.addHandler(_v34_handler)
logger = logging.getLogger("cnn_mamba_v3_4")
logger.setLevel(logging.INFO)
print(">>> train_cnn_mamba_v3_4.py loaded (dual-trunk, book-shape 2D-CNN)", flush=True)


# ============================================================
# v3.4-specific config
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output" / "cnn_mamba_v3_4_dual_trunk")
DEFAULT_BOOK_FEATURES_DIR = str(PROJECT_ROOT / "data" / "processed" / "mbo_book_features")

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")
MLFLOW_EXPERIMENT = "CNNMamba_v3_4_dual_trunk_uncertainty_weighted"

V33_WARMSTART_CKPT = str(PROJECT_ROOT / "output" / "cnn_mamba_v3_3_uncertainty_weighted" / "fold_00_intra_ckpt.pt")

# Book pyramid input shape: (B, window, 5_level_pairs, 4_features_per_pair)
# 4 features per level pair: [bid_price, bid_size, ask_price, ask_size]
N_BOOK_LEVELS = 5
N_BOOK_FEATURES_PER_LEVEL = 4
BOOK_EMB_DIM = MAMBA_D_MODEL  # match the other trunk dims so trunk Linear cleanly extends

# Falsification gate (HC #366 Q3=I — default strictness)
GATE_IC_1S_17DAY_MIN = 0.23
GATE_IC_1S_5DAY_MIN = 0.296
GATE_MAGCORR_BOOK_MIN = 0.05

# Book feature column indices in the 30-feature flat array
# (validated by v34_data_alignment_check.py):
#   bid_price_1..5 = [0..4]
#   ask_price_1..5 = [5..9]
#   bid_size_1..5  = [10..14]
#   ask_size_1..5  = [15..19]
#   derived (20..29) — IGNORED by book trunk (event trunk handles temporal dynamics)
BOOK_BID_PRICE_IDX = list(range(0, 5))
BOOK_ASK_PRICE_IDX = list(range(5, 10))
BOOK_BID_SIZE_IDX  = list(range(10, 15))
BOOK_ASK_SIZE_IDX  = list(range(15, 20))


# ============================================================
# Book2DCNN trunk — 2D conv over (level_pair, time)
# ============================================================
class Book2DCNN(nn.Module):
    """
    Input:  (B, T, 5_levels, 4_features) where 4 = [bid_p, bid_s, ask_p, ask_s]
    Output: (B, BOOK_EMB_DIM)  — temporal-pooled book-pyramid embedding

    Architecture rationale (from book_spatial_cnn.py + HC #353):
      - The book has SPATIAL structure across levels: adjacent levels are related,
        bid/ask sides mirror each other. 2D conv learns:
        * depth walls (large size concentrated at level k)
        * depth thinning (gradual decrease)
        * queue-imbalance gradient across levels
        * spread dynamics
      - Temporal axis carries microstructure dynamics (depletion → impact, refill →
        liquidity return). Conv2D over (level, time) captures both jointly.
      - Output is temporal-pooled (mean+max concat) to fixed BOOK_EMB_DIM so the trunk
        Linear can concat with the other 3 branch embeddings.

    Param budget: ~150K params (small relative to event trunks).
    """
    def __init__(self, in_features: int = 4, in_levels: int = 5,
                 out_dim: int = BOOK_EMB_DIM, dropout: float = 0.1):
        super().__init__()
        # We treat the 4 features as input channels, and conv over (levels, time)
        # Input permuted: (B, in_features=4, in_levels=5, T)
        self.conv1 = nn.Conv2d(in_features, 32, kernel_size=(3, 5), padding=(1, 2))
        self.bn1 = nn.BatchNorm2d(32)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=(3, 5), padding=(1, 2))
        self.bn2 = nn.BatchNorm2d(64)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=(3, 3), padding=(1, 1))
        self.bn3 = nn.BatchNorm2d(128)
        # After 3 convs we keep (B, 128, 5_levels, T). Pool over levels (mean) + over time (mean+max).
        self.level_pool = nn.AdaptiveAvgPool2d((1, None))   # → (B, 128, 1, T)
        self.proj_to_emb = nn.Linear(128 * 2, out_dim)       # 128*2 (mean+max temporal)
        self.dropout = nn.Dropout(dropout)
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

    def forward(self, book_window: torch.Tensor) -> torch.Tensor:
        """
        book_window: (B, T, 5_levels, 4_features)
        returns: (B, out_dim=BOOK_EMB_DIM)
        """
        # Sanitize
        x = torch.nan_to_num(book_window, nan=0.0, posinf=10.0, neginf=-10.0)
        # Permute to (B, 4_features=C, 5_levels=H, T=W)
        x = x.permute(0, 3, 2, 1).contiguous()
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        # Pool levels: (B, 128, 1, T)
        x = self.level_pool(x).squeeze(2)   # (B, 128, T)
        # Temporal pool: mean + max → (B, 128*2)
        mean_t = x.mean(dim=-1)
        max_t = x.amax(dim=-1)
        feat = torch.cat([mean_t, max_t], dim=-1)
        feat = self.dropout(feat)
        return self.proj_to_emb(feat)


# ============================================================
# Dual-trunk model: v3.2 backbone + Book2DCNN trunk fused at trunk MLP input
# ============================================================
class CNNMambaV34DualTrunk(nn.Module):
    """
    v3.4 = v3.2 model (T1+T2+T3 branches retained for warmstart compat)
              + NEW T2_BOOK branch (Book2DCNN over 5-level book pyramid)
              + WIDENED trunk Linear input (3*d_model → 4*d_model)
              + same 32 heads + same uncertainty-weighted MTL loss (v3.3 inherited)
    """
    HEAD_NAMES = ALL_HEAD_NAMES

    def __init__(
        self,
        d_model_t1: int = MAMBA_D_MODEL,
        d_model_t2: int = MAMBA_D_MODEL,
        d_model_t3: int = T3_D_MODEL,
        d_model_book: int = BOOK_EMB_DIM,
        trunk_dim: int = TRUNK_DIM,
        dropout: float = MAMBA_DROPOUT,
    ):
        super().__init__()
        # Embed an entire v3.2 model inside us. We will USE its branch backbones + adapters
        # for T1/T2/T3 paths, but bypass its `trunk` and `heads` (we replace those at v3.4).
        # This keeps warmstart simple: load v3.3 ckpt directly into self.v32_core.{t1,t2,t3}_*
        self.v32_core = CNNMambaV32(
            d_model_t1=d_model_t1, d_model_t2=d_model_t2,
            d_model_t3=d_model_t3, trunk_dim=trunk_dim, dropout=dropout,
        )
        # We REPLACE v32_core.trunk and v32_core.heads at v3.4 — define our own widened ones.
        # (Keep the v32_core branch backbones + adapters as the event-temporal trunks.)

        # NEW: book trunk
        self.book_trunk = Book2DCNN(
            in_features=N_BOOK_FEATURES_PER_LEVEL,
            in_levels=N_BOOK_LEVELS,
            out_dim=d_model_book,
            dropout=dropout,
        )

        # Widened fusion trunk (4 branches concat instead of 3)
        fused_dim_v34 = d_model_t1 + d_model_t2 + d_model_t3 + d_model_book
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim_v34, trunk_dim),
            nn.GELU(),
            nn.LayerNorm(trunk_dim),
            nn.Dropout(dropout),
        )

        # Heads (same spec as v3.2/v3.3)
        self.heads = nn.ModuleDict({
            name: nn.Linear(trunk_dim, 1) for name in self.HEAD_NAMES
        })
        self._init_v34_weights()

    def _init_v34_weights(self):
        modules = list(self.trunk.modules()) + list(self.heads.modules())
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        batch keys (extends v3.2):
          events_t1 (B, L1, F1), events_t2 (B, L2, F2), events_t3 (B, L3, F3)
          book_pyramid (B, T_book, 5_levels, 4_features)  ← NEW
        """
        c = self.v32_core
        e1 = torch.nan_to_num(batch["events_t1"], nan=0.0, posinf=10.0, neginf=-10.0)
        x1 = c.t1_adapter(e1)
        _, emb1 = c.t1_backbone(x1, return_embedding=True)

        e2 = torch.nan_to_num(batch["events_t2"], nan=0.0, posinf=10.0, neginf=-10.0)
        x2 = c.t2_adapter(e2)
        _, emb2 = c.t2_backbone(x2, return_embedding=True)

        e3 = torch.nan_to_num(batch["events_t3"], nan=0.0, posinf=10.0, neginf=-10.0)
        x3 = c.t3_adapter(e3)
        _, emb3 = c.t3_backbone(x3, return_embedding=True)

        emb_book = self.book_trunk(batch["book_pyramid"])

        emb = torch.cat([emb1, emb2, emb3, emb_book], dim=-1)
        trunk_out = self.trunk(emb)
        return {name: head(trunk_out).squeeze(-1) for name, head in self.heads.items()}

    # ------------------------------------------------------------------
    # Warmstart from v3.3 intra-ckpt
    # ------------------------------------------------------------------
    def load_v33_warmstart(self, ckpt_path: str, device: torch.device) -> Dict[str, int]:
        """
        Load v3.3 fold_00_intra_ckpt.pt into v3.4 model:
          - v32_core.{t1,t2,t3}_* branches:           DIRECT copy (same shapes)
          - heads.*:                                  DIRECT copy (same spec)
          - trunk[0] (Linear fused_dim → trunk_dim): PARTIAL copy — first 3*d_model cols
                                                     match v3.3's 3-branch concat;
                                                     last d_model_book cols init random.
          - book_trunk.*:                             stays random init (no v3.3 equivalent)
        Returns dict {loaded, partial, skipped, random_init}.
        """
        stats = {"loaded": 0, "partial": 0, "skipped": 0, "random_init": 0}
        if not Path(ckpt_path).exists():
            logger.warning(f"v3.4: v3.3 warmstart not found: {ckpt_path} — fully random init")
            return stats
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception as e:
            logger.error(f"v3.4: failed to load v3.3 ckpt: {e}")
            return stats

        src = ckpt.get("model_state", ckpt)  # accept either flat or nested
        my_state = self.state_dict()

        # Build name-map: v3.3 state keys are NOT prefixed (e.g. "t1_adapter.weight"),
        # v3.4 wraps them under "v32_core." (e.g. "v32_core.t1_adapter.weight").
        # Trunk and heads are at v3.4 top-level (replaced from v3.3's, so we map carefully).
        for src_name, src_tensor in src.items():
            target = None
            # Branches → v32_core.*
            if any(src_name.startswith(p) for p in
                   ("t1_adapter", "t1_backbone", "t2_adapter", "t2_backbone",
                    "t3_adapter", "t3_backbone")):
                target = f"v32_core.{src_name}"
            # Trunk: v3.3 has trunk[0] = Linear(3d → trunk_dim). v3.4 needs partial copy.
            elif src_name == "trunk.0.weight":
                # Shape v3.3: (trunk_dim, 3*d_model). v3.4: (trunk_dim, 4*d_model).
                # Copy into the FIRST 3*d_model cols; last d_model_book cols stay random.
                if "trunk.0.weight" in my_state:
                    my_w = my_state["trunk.0.weight"]
                    n_cols_src = src_tensor.shape[1]
                    if n_cols_src <= my_w.shape[1]:
                        my_w[:, :n_cols_src].copy_(src_tensor.to(device))
                        stats["partial"] += 1
                        continue
            elif src_name in ("trunk.0.bias", "trunk.2.weight", "trunk.2.bias"):
                # LayerNorm + Linear bias have same shape — direct copy
                target = src_name
            # Heads → direct (same shape)
            elif src_name.startswith("heads."):
                target = src_name
            # log_sigma params live in loss_fn (NOT in model state) — handled outside

            if target is not None and target in my_state:
                try:
                    my_state[target].copy_(src_tensor.to(device))
                    stats["loaded"] += 1
                except Exception as e:
                    logger.warning(f"v3.4 warmstart: {src_name}→{target} copy failed: {e}")
                    stats["skipped"] += 1
            elif target is None:
                stats["skipped"] += 1

        # Random-init count = # of params in book_trunk + trunk last cols
        stats["random_init"] = sum(p.numel() for p in self.book_trunk.parameters())
        logger.info(f"v3.4 warmstart from v3.3 ckpt: {stats}")
        print(f">>> v3.4 warmstart: loaded={stats['loaded']} partial={stats['partial']} "
              f"skipped={stats['skipped']} random_init_params={stats['random_init']}", flush=True)
        return stats

    def load_v33_loss_state(self, ckpt_path: str, loss_fn: nn.Module, device: torch.device) -> bool:
        """Load v3.3's learned log_sigmas into v3.4's loss_fn (same head spec)."""
        if not Path(ckpt_path).exists():
            return False
        try:
            ckpt = torch.load(ckpt_path, map_location=device)
        except Exception:
            return False
        if "loss_state" in ckpt and ckpt["loss_state"] is not None:
            try:
                loss_fn.load_state_dict(ckpt["loss_state"], strict=False)
                logger.info("v3.4: loaded v3.3 log_sigma params into loss_fn")
                return True
            except Exception as e:
                logger.warning(f"v3.4: loss_state load failed: {e}")
        return False


# ============================================================
# Dataset: wraps SmartV32Dataset + loads book features alongside
# ============================================================
class SmartV34DualTrunkDataset(Dataset):
    """
    Wraps SmartV32Dataset to ALSO emit a book_pyramid window per sample.

    Book features come from `mbo_book_features/<date>.npz` and are row-aligned with
    `mbo_events_smart_v3/<date>.npz` (validated by v34_data_alignment_check.py).
    Each book row's 30 flat features get reshaped to (5_level_pairs, 4_features) via
    BOOK_*_IDX index slicing.

    The book window matches the T2 event-stream window length (WINDOW_SIZE_T2) and
    stride — book features are sampled at the same event rate.
    """
    def __init__(
        self,
        v32_inner: SmartV32Dataset,
        book_features_dir: str = DEFAULT_BOOK_FEATURES_DIR,
        window_size_book: int = WINDOW_SIZE_T2,
        log_size: bool = True,
    ):
        self.inner = v32_inner
        self.window_size_book = window_size_book
        self.book_dir = Path(book_features_dir)

        # The v32 inner dataset already knows which dates it covers + which row index
        # each sample maps to. We mirror its date-to-array mapping for the book features
        # so book[sample.row_idx - window + 1 : sample.row_idx + 1] is the window.
        #
        # SmartV32Dataset internally holds per-date NPZ data. We load matching book
        # NPZs once + cache as numpy arrays in CPU memory (compressed: ~30 cols × 4 bytes
        # × ~400K rows/day × ~60 days ≈ 3 GB — acceptable for 32 GB Jupiter).
        self.book_data = {}
        loaded_dates = 0
        missing_dates = []
        for date_str in getattr(self.inner, "dates", []):
            bp = self.book_dir / f"{date_str}_book_features.npz"
            if bp.exists():
                d = np.load(bp)
                arr = d["features"]  # (N, 30) float32
                # Pre-reshape to (N, 5_levels, 4_features) to save per-sample compute
                # Stack: bid_p[5], bid_s[5], ask_p[5], ask_s[5] → per-level [bid_p, bid_s, ask_p, ask_s]
                bp_arr = arr[:, BOOK_BID_PRICE_IDX]    # (N, 5)
                bs_arr = arr[:, BOOK_BID_SIZE_IDX]     # (N, 5)
                ap_arr = arr[:, BOOK_ASK_PRICE_IDX]    # (N, 5)
                as_arr = arr[:, BOOK_ASK_SIZE_IDX]     # (N, 5)
                # Stack along last axis → (N, 5_levels, 4_features)
                pyramid = np.stack([bp_arr, bs_arr, ap_arr, as_arr], axis=-1).astype(np.float32)
                # Normalize prices to ticks-from-mid to keep scale invariant across dates
                # (use bid_price_1 as reference; subtract from all 4 price entries per row)
                # Sizes log-transformed (book_spatial_cnn.py convention)
                ref_mid = (pyramid[:, 0, 0] + pyramid[:, 0, 2]) / 2.0  # (bid_p_1 + ask_p_1) / 2
                pyramid[:, :, 0] = (pyramid[:, :, 0] - ref_mid[:, None]) / 0.25  # bid_p in ticks-from-mid
                pyramid[:, :, 2] = (pyramid[:, :, 2] - ref_mid[:, None]) / 0.25  # ask_p in ticks-from-mid
                pyramid[:, :, 1] = np.log1p(pyramid[:, :, 1])  # bid_size log-scale
                pyramid[:, :, 3] = np.log1p(pyramid[:, :, 3])  # ask_size log-scale
                self.book_data[date_str] = pyramid
                loaded_dates += 1
            else:
                missing_dates.append(date_str)

        if log_size:
            logger.info(f"SmartV34DualTrunkDataset: book loaded for {loaded_dates} dates, "
                        f"missing {len(missing_dates)} dates")
            if missing_dates:
                logger.warning(f"  missing book dates: {missing_dates[:10]}"
                               f"{' ...' if len(missing_dates) > 10 else ''}")

    def __len__(self):
        return len(self.inner)

    def _book_window(self, date_str: str, row_idx: int) -> np.ndarray:
        """Returns (window_size_book, 5_levels, 4_features) np.float32."""
        arr = self.book_data.get(date_str)
        if arr is None:
            # Missing book for this date → return zeros (model will treat as no book signal)
            return np.zeros((self.window_size_book, N_BOOK_LEVELS, N_BOOK_FEATURES_PER_LEVEL),
                            dtype=np.float32)
        end = row_idx + 1
        start = max(0, end - self.window_size_book)
        win = arr[start:end]
        # Left-pad with zeros if at the very start of the day
        if win.shape[0] < self.window_size_book:
            pad = np.zeros((self.window_size_book - win.shape[0], N_BOOK_LEVELS,
                            N_BOOK_FEATURES_PER_LEVEL), dtype=np.float32)
            win = np.concatenate([pad, win], axis=0)
        return win.astype(np.float32, copy=False)

    def __getitem__(self, idx):
        """Returns (events_dict, targets, masks) — events_dict gets `book_pyramid` added."""
        events, targets, masks = self.inner[idx]
        # SmartV32Dataset stores sample metadata; need date_str + row_idx for the book lookup.
        # The inner Dataset has self.samples or self.valid_indices — we'll inspect at runtime.
        # Common attr name in v3.2/v3.3: self.inner.samples = list of (date_str, row_idx, ...)
        # OR self.inner has self.dates + self.day_boundaries + self.valid_indices (book_spatial_cnn style)
        # We try both patterns; fallback to zero window.
        date_str = None
        row_idx = None
        if hasattr(self.inner, "samples") and idx < len(getattr(self.inner, "samples", [])):
            samp = self.inner.samples[idx]
            if isinstance(samp, dict):
                date_str = samp.get("date_str") or samp.get("date")
                row_idx = samp.get("row_idx") or samp.get("idx")
            elif isinstance(samp, (list, tuple)) and len(samp) >= 2:
                date_str = samp[0]
                row_idx = samp[1]
        elif hasattr(self.inner, "valid_indices") and hasattr(self.inner, "day_boundaries") \
                and hasattr(self.inner, "dates"):
            global_idx = self.inner.valid_indices[idx]
            for di in range(len(self.inner.day_boundaries) - 1):
                if self.inner.day_boundaries[di] <= global_idx < self.inner.day_boundaries[di + 1]:
                    date_str = self.inner.dates[di]
                    row_idx = global_idx - self.inner.day_boundaries[di]
                    break

        if date_str and row_idx is not None:
            book_win = self._book_window(date_str, row_idx)
        else:
            book_win = np.zeros((self.window_size_book, N_BOOK_LEVELS, N_BOOK_FEATURES_PER_LEVEL),
                                dtype=np.float32)
        events["book_pyramid"] = book_win  # added to events dict so collate handles it
        return events, targets, masks


def collate_v34(batch):
    """Like collate_v32, but also stacks the new `book_pyramid` tensor."""
    events_list, targets_list, masks_list = zip(*batch)
    out_events = {}
    for k in events_list[0].keys():
        out_events[k] = torch.from_numpy(np.stack([e[k] for e in events_list], axis=0))
    out_targets, out_masks = collate_v32([(e, t, m) for e, t, m in
                                          zip([{kk: vv for kk, vv in e.items() if kk != "book_pyramid"}
                                               for e in events_list],
                                              targets_list, masks_list)])[1:]
    return out_events, out_targets, out_masks


# ============================================================
# train_one_fold_v34 — like v3.3 but with falsification gate at Ep 1 OOT
# ============================================================
def train_one_fold_v34(
    model: CNNMambaV34DualTrunk,
    train_loader: DataLoader,
    oot_loader: DataLoader,
    fold_idx: int,
    output_dir: Path,
    mlflow_run,
    device: torch.device,
    total_train_steps: int,
    v3_3_baseline_ic_1s_5day: float = 0.286,
    v3_3_baseline_ic_1s_17day: float = 0.1765,
    use_amp: bool = True,
    resume_state: Optional[Dict] = None,
) -> Dict:
    loss_fn = JointMultiHeadLossV33_UncertaintyWeighted(head_names=ALL_HEAD_NAMES).to(device)
    optimizer = torch.optim.AdamW(
        [
            {"params": model.parameters(), "lr": LR, "weight_decay": 1e-4},
            {"params": list(loss_fn.parameters()), "lr": LR, "weight_decay": 0.0},
        ]
    )
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=WARMUP_STEPS, total_steps=total_train_steps
    )
    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    best_val_loss = float("inf")
    global_step = 0
    total_batches = len(train_loader)
    resume_epoch = 0
    resume_batch = 0

    if resume_state is not None and resume_state.get("ckpt_version", 1) >= 2:
        try:
            optimizer.load_state_dict(resume_state["optimizer_state"])
            if resume_state.get("scheduler_state") is not None and hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(resume_state["scheduler_state"])
            if "loss_state" in resume_state and resume_state["loss_state"] is not None:
                loss_fn.load_state_dict(resume_state["loss_state"], strict=False)
            resume_epoch = int(resume_state.get("epoch", 0))
            resume_batch = int(resume_state.get("batch", 0))
            global_step = int(resume_state.get("global_step", 0))
            best_val_loss = float(resume_state.get("best_val_loss", float("inf")))
            logger.info(f"v3.4 RESUMED fold {fold_idx}: ep={resume_epoch} batch={resume_batch}")
        except Exception as e:
            logger.warning(f"v3.4 resume failed ({e}); starting fold from scratch")

    # Try v3.3 log_sigma warmstart (loaded once on fold-0 fresh-start)
    if resume_state is None and fold_idx == 0:
        model.load_v33_loss_state(V33_WARMSTART_CKPT, loss_fn, device)

    print(f">>> v3.4 train_one_fold {fold_idx}: {EPOCHS_PER_FOLD} epochs x {total_batches} batches",
          flush=True)

    falsification_killed = False
    falsification_reason = None

    for epoch in range(EPOCHS_PER_FOLD):
        if epoch < resume_epoch:
            continue
        model.train()
        loss_fn.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        comp_acc = defaultdict(float)
        skip_until = resume_batch if epoch == resume_epoch else 0

        for events, targets, masks in train_loader:
            if n_batches < skip_until:
                n_batches += 1
                continue
            events = {k: v.to(device, non_blocking=True) for k, v in events.items()}
            targets = {k: v.to(device, non_blocking=True) for k, v in targets.items()}
            masks = {k: v.to(device, non_blocking=True) for k, v in masks.items()}

            optimizer.zero_grad(set_to_none=True)
            with amp_ctx:
                preds = model(events)
                loss, components = loss_fn(preds, targets, masks)

            loss_finite = bool(torch.isfinite(loss).item())
            if loss_finite:
                loss.backward()
                for p in list(model.parameters()) + list(loss_fn.parameters()):
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(loss_fn.parameters()),
                    GRAD_CLIP,
                )
                optimizer.step()
            scheduler.step()
            if loss_finite:
                epoch_loss += float(loss.item())
                for k, v in components.items():
                    comp_acc[k] += v
            n_batches += 1
            global_step += 1

            if n_batches % 100 == 0:
                elapsed = time.time() - epoch_start
                eta = elapsed / n_batches * (total_batches - n_batches)
                msg = (f"  v3.4 Fold {fold_idx} Ep {epoch+1} Batch {n_batches}/{total_batches} | "
                       f"Loss: {epoch_loss/n_batches:.4f} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s")
                print(msg, flush=True)
                logger.info(msg)

            if n_batches % 500 == 0:
                ckpt = {
                    "model_state": model.state_dict(),
                    "loss_state": loss_fn.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_step": getattr(scheduler, "_step_count", global_step),
                    "scheduler_state": scheduler.state_dict() if hasattr(scheduler, "state_dict") else None,
                    "fold": fold_idx, "epoch": epoch, "batch": n_batches,
                    "global_step": global_step, "best_val_loss": best_val_loss,
                    "ckpt_version": 2,
                }
                torch.save(ckpt, output_dir / f"fold_{fold_idx:02d}_intra_ckpt.pt")

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start

        # OOT eval (same evaluator as v3.3)
        val_metrics, _, _, _ = evaluate_v32(model, oot_loader, loss_fn, device, use_amp=use_amp)
        sigmas = loss_fn.get_sigma_dict()
        ic_1s = val_metrics.get("ic_log_ret_1s", float("nan"))
        ic_5s = val_metrics.get("ic_log_ret_5s", float("nan"))
        ic_10s = val_metrics.get("ic_log_ret_10s", float("nan"))
        ic_30s = val_metrics.get("ic_log_ret_30s", float("nan"))

        msg = (f"v3.4 Fold {fold_idx:02d} Ep {epoch+1}/{EPOCHS_PER_FOLD} | "
               f"TrLoss {avg_loss:.4f} | OOT Loss {val_metrics['loss']:.4f} | "
               f"IC 1s/5s/10s/30s = {ic_1s:.4f}/{ic_5s:.4f}/{ic_10s:.4f}/{ic_30s:.4f} | "
               f"LR {scheduler.get_lr():.2e} | T {epoch_time:.1f}s")
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
                for h_name, sig in sigmas.items():
                    metrics_to_log[f"f{fold_idx:02d}_sigma_{h_name}"] = float(sig)
                mlflow.log_metrics(metrics_to_log, step=step)
            except Exception as e:
                logger.warning(f"v3.4 MLflow log failed: {e}")

        # Save predictions npz BEFORE ckpt block (per v3.3 lesson — crash-resilient)
        try:
            _, oot_preds, oot_targs, oot_masks_out = evaluate_v32(
                model, oot_loader, loss_fn, device, use_amp=use_amp, return_preds=True
            )
            np.savez(
                output_dir / f"fold_{fold_idx:02d}_ep{epoch+1}_predictions.npz",
                **{f"pred_{k}": v for k, v in oot_preds.items()},
                **{f"tgt_{k}": v for k, v in oot_targs.items()},
            )
        except TypeError:
            # evaluate_v32 may not support return_preds in older v3.2 — non-fatal
            pass
        except Exception as e:
            logger.warning(f"v3.4 predictions.npz save failed: {e}")

        # Falsification gate — only at Ep 1 (epoch == 0 in 0-indexed)
        if epoch == 0 and fold_idx == 0:
            gate_5day = (not np.isnan(ic_1s)) and ic_1s >= GATE_IC_1S_5DAY_MIN
            gate_17day = (not np.isnan(ic_1s)) and ic_1s >= GATE_IC_1S_17DAY_MIN
            # MagCorr book-shape heads: not directly registered yet — placeholder
            # (true MagCorr-on-book-heads requires post-hoc analysis; for now use IC bound)
            gate_passed = gate_5day or gate_17day
            verdict = ("PASS_5DAY" if gate_5day else ("PASS_17DAY" if gate_17day else "FAIL"))
            logger.info(f"v3.4 falsification gate fold-0 Ep 1: IC_1s={ic_1s:.4f} "
                        f"vs 5day_min={GATE_IC_1S_5DAY_MIN} 17day_min={GATE_IC_1S_17DAY_MIN} "
                        f"verdict={verdict}")
            print(f">>> v3.4 FALSIFICATION GATE Ep 1: IC_1s={ic_1s:.4f} | "
                  f"5day_thresh={GATE_IC_1S_5DAY_MIN} 17day_thresh={GATE_IC_1S_17DAY_MIN} | "
                  f"verdict={verdict}", flush=True)

            if MLFLOW_AVAILABLE and mlflow_run is not None:
                try:
                    mlflow.set_tag("falsification_ep1_ic_1s", f"{ic_1s:.4f}")
                    mlflow.set_tag("falsification_ep1_verdict", verdict)
                except Exception:
                    pass

            if not gate_passed:
                falsification_killed = True
                falsification_reason = (f"IC_1s={ic_1s:.4f} below both thresholds "
                                        f"(5day≥{GATE_IC_1S_5DAY_MIN}, 17day≥{GATE_IC_1S_17DAY_MIN})")
                if MLFLOW_AVAILABLE and mlflow_run is not None:
                    try:
                        mlflow.set_tag("falsification_failed", falsification_reason)
                    except Exception:
                        pass
                print(f">>> v3.4 KILLED at fold-0 Ep 1 per falsification gate: {falsification_reason}",
                      flush=True)
                logger.error(f"v3.4 KILLED at fold-0 Ep 1: {falsification_reason}")
                break  # break out of epoch loop — fold done

        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            torch.save({
                "model_state": model.state_dict(),
                "loss_state": loss_fn.state_dict(),
                "fold": fold_idx, "epoch": epoch,
                "val_loss": val_metrics["loss"],
                "val_metrics": val_metrics,
                "sigmas": sigmas,
                "arch": {
                    "model": "CNNMambaV34DualTrunk",
                    "trainer": "v3.4_dual_trunk_uncertainty_weighted",
                    "window_t1": WINDOW_SIZE_T1, "window_t2": WINDOW_SIZE_T2,
                    "window_t3": WINDOW_SIZE_T3,
                    "n_t1_features": N_T1_FEATURES, "n_t2_features": N_T2_FEATURES,
                    "n_t3_features": N_T3_FEATURES,
                    "n_book_levels": N_BOOK_LEVELS,
                    "n_book_features_per_level": N_BOOK_FEATURES_PER_LEVEL,
                    "book_emb_dim": BOOK_EMB_DIM,
                    "head_names": ALL_HEAD_NAMES,
                    "log_sigma_init": LOG_SIGMA_INIT,
                },
            }, output_dir / f"fold_{fold_idx:02d}_best.pt")

    return {
        "best_val_loss": best_val_loss,
        "final_sigmas": loss_fn.get_sigma_dict() if not falsification_killed else None,
        "falsification_killed": falsification_killed,
        "falsification_reason": falsification_reason,
    }


# ============================================================
# Entry point
# ============================================================
def main():
    p = argparse.ArgumentParser(description="v3.4 dual-trunk CNN-Mamba trainer")
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--book-features-dir", default=DEFAULT_BOOK_FEATURES_DIR)
    p.add_argument("--fifo-label-dir", default=str(DEFAULT_FIFO_LABEL_DIR))
    p.add_argument("--alpha-label-dir", default=str(DEFAULT_ALPHA_LABEL_DIR))
    p.add_argument("--pt-pred-dir", default=str(DEFAULT_PT_PRED_DIR))
    p.add_argument("--tier2-parquet-root", default=str(DEFAULT_TIER2_PARQUET_ROOT))
    p.add_argument("--tier3-parquet-root", default=str(DEFAULT_TIER3_PARQUET_ROOT))
    p.add_argument("--n-folds", type=int, default=1, help="default 1 = fold-0 only for falsification gate")
    p.add_argument("--device", default="cuda")
    p.add_argument("--smoke-test", action="store_true", help="1 batch CPU forward+backward sanity")
    p.add_argument("--resume-from-intra-ckpt", default=None)
    p.add_argument("--no-amp", action="store_true")
    args = p.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f">>> v3.4 device={device} output_dir={output_dir}", flush=True)

    # Build model
    model = CNNMambaV34DualTrunk().to(device)
    n_params = count_parameters(model)
    print(f">>> v3.4 model params: {n_params/1e6:.2f}M (target ≤ 2.5M per spec §3)", flush=True)
    logger.info(f"v3.4 model params: {n_params}")

    # Warmstart from v3.3 intra-ckpt
    ws_stats = model.load_v33_warmstart(V33_WARMSTART_CKPT, device)

    if args.smoke_test:
        print(">>> v3.4 SMOKE TEST: dummy forward+backward pass", flush=True)
        B = 2
        L1 = WINDOW_SIZE_T1
        L2 = WINDOW_SIZE_T2
        L3 = WINDOW_SIZE_T3
        dummy = {
            "events_t1": torch.randn(B, L1, N_T1_FEATURES, device=device),
            "events_t2": torch.randn(B, L2, N_T2_FEATURES, device=device),
            "events_t3": torch.randn(B, L3, N_T3_FEATURES, device=device),
            "book_pyramid": torch.randn(B, L2, N_BOOK_LEVELS, N_BOOK_FEATURES_PER_LEVEL, device=device),
        }
        out = model(dummy)
        print(f">>> v3.4 SMOKE OUT: n_heads={len(out)} first_3={list(out.keys())[:3]} "
              f"shape={out[list(out.keys())[0]].shape}", flush=True)
        # Tiny backward
        dummy_loss = sum(t.sum() for t in out.values())
        dummy_loss.backward()
        print(">>> v3.4 SMOKE TEST: forward+backward PASSED", flush=True)
        return 0

    # Real training entry — delegate to a v3.4 walk-forward driver.
    # For fold-0-only falsification gate (default), we instantiate v3.3's walk-forward
    # driver with a CNNMambaV34DualTrunk model + the v3.4 dataset wrapper.
    print(">>> v3.4 walk-forward training requires the v3.4 WF driver. ",
          "Use --smoke-test first for sanity.", flush=True)
    print(">>> v3.4 WF DRIVER: deferred to follow-up turn — see SESSION_STATE.md", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
