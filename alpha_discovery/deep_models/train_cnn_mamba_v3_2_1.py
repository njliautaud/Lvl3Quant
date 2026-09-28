"""
CNN-Mamba v3.2 — LONG-CONTEXT, BOOK-MEMORY, REGIME-AWARE Multi-Head Trainer

Per CNN_MAMBA_V3.2 spec (docs/cnn_mamba_v3.2_spec.md) and HC #294.

Key changes vs v3:
  - Multi-resolution context: Tier 1 (1500 native MBO events) +
    Tier 2 (1500 @100ms bucket aggregates) + Tier 3 (1500 @1Hz session-context snapshots).
  - Each tier has its own Mamba branch; embeddings concatenated before trunk.
  - 28 alpha-first heads (directional 1s/5s/10s/30s/60s/5min, p_up, quantiles,
    MFE/MAE 30s+60s, reversal 15s/30s/60s, vol) + 4 legacy FIFO at λ=0.1.
  - Tier 1 input features: 25 smart_v3 + 4 PatchTST (fwd-filled) + 10 book-history = 39.
  - Tier 2 input features: 39 (bucket aggregates of Tier 1).
  - Tier 3 input features: 15 session-context (S/R, prior session H/L/C, VWAP, VPOC, TOD, vol regime).
    Tier 3 features are ZERO-PLACEHOLDER until backfilled (TODO).
  - Warmstart from v3 fold_00_best.pt (Tier 1 backbone + trunk; rest fresh).
  - Same fold schedule as v3 (anchor 2026-02-23) for direct OOT IC comparison.
  - Falsification at Ep 1 OOT: must beat v3 IC by >=+0.01 on at least one of
    IC_5s/IC_10s/IC_30s OR show non-degenerate quantile calibration on log_ret_10s
    OR >=+0.05 correlation between predicted and realized MFE/MAE.

Author: Claude (head-of-quant), 2026-05-11, under HC #283 / #293 / #294.
"""

# Set feature set BEFORE importing v2 trainer
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
import re
from pathlib import Path
from typing import Optional, List, Dict, Tuple
from collections import defaultdict

import numpy as np
import pyarrow.parquet as pq
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import scipy.stats

# v2 backbone (CNN-Mamba) — reused per branch (own params each)
from alpha_discovery.deep_models.train_cnn_mamba import (
    CNNMamba,
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
    LR,
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
log_path = LOG_DIR / "cnn_mamba_v3_2.log"
_v32_handler = logging.FileHandler(log_path)
_v32_handler.setLevel(logging.INFO)
_v32_handler.setFormatter(logging.Formatter("%(asctime)s [v3.2 %(levelname)s] %(message)s"))
logging.root.addHandler(_v32_handler)
logger = logging.getLogger("cnn_mamba_v3_2")
logger.setLevel(logging.INFO)
print(">>> train_cnn_mamba_v3_2.py loaded", flush=True)


# ============================================================
# Config
# ============================================================
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

DEFAULT_DATA_DIR = str(PROJECT_ROOT / "data" / "processed" / "mbo_events_smart_v3")
DEFAULT_FIFO_LABEL_DIR = str(PROJECT_ROOT / "data" / "processed" / "mbo_events_smart_v3_fifo_labels")
DEFAULT_ALPHA_LABEL_DIR = str(PROJECT_ROOT / "data" / "processed" / "mbo_events_smart_v3_alpha_labels")
DEFAULT_PT_PRED_DIR = str(PROJECT_ROOT / "data" / "processed" / "mbo_events_smart_v3_pt_pred")
DEFAULT_TIER2_PARQUET_ROOT = str(PROJECT_ROOT / "data" / "processed" / "tier2_orderflow_v3_2_1")
DEFAULT_TIER3_PARQUET_ROOT = str(PROJECT_ROOT / "data" / "processed" / "tier3_session_v3_2_1")
DEFAULT_OUTPUT_DIR = str(PROJECT_ROOT / "output" / "cnn_mamba_v3_2_1_long_context")
V3_WARMSTART_CKPT = str(PROJECT_ROOT / "output" / "cnn_mamba_v3_smart_v3_fifo" / "fold_00_best.pt")

# RTH bounds (mirrors build_v3_2_tier_features.py)
RTH_OPEN_HOUR_ET = 9
RTH_OPEN_MIN_ET = 30
RTH_CLOSE_HOUR_ET = 16
RTH_CLOSE_MIN_ET = 0
RTH_SECONDS_PER_DAY = 23400
TIER2_BUCKET_NS = 100_000_000  # 100ms
TIER3_SNAP_NS = 1_000_000_000  # 1s

# Per-parquet feature column order (MUST match build_v3_2_1_tier_features.py)
# v3.2.1: T2 expanded 14 -> 19 (5 missing features restored per HC #295C)
# v3.2.1: T3 expanded 25 -> 31 (2 LGBM vol pred + 4 session-phase one-hot)
TIER2_FEATURE_COLS = [
    "log_return_in_bucket_bps", "bucket_mfe_ticks", "bucket_mae_ticks",
    "n_trades", "trade_volume", "aggressor_buy_ratio", "signed_volume",
    "n_cancels", "n_adds", "cancel_add_ratio", "n_order_events",
    "avg_order_size", "bucket_range_ticks", "seconds_since_rth_open",
    # ---- NEW v3.2.1 (Priority-1 item #2): restored 5 missing T2 features ----
    "microprice_change_ticks",          # L2 microprice change in 100ms bucket
    "avg_spread_in_bucket_ticks",       # avg best_ask - best_bid in bucket
    "n_top_of_book_changes",            # count of TOB price changes
    "avg_top5_depth_volume",            # log1p of avg top-5 depth (PASS in builder, z-scored ok)
    "large_order_count",                # count of orders > 95th pctile of 5d size dist
]
TIER3_FEATURE_COLS = [
    # Price location / S-R (5) — DISTANCE: /100 in builder, PASS in dataloader
    "dist_intraday_high_ticks", "dist_intraday_low_ticks", "dist_session_vwap_ticks",
    "dist_prior_session_close_ticks", "dist_prior_session_vwap_ticks",
    # Volume profile (5)
    "dist_intraday_vpoc_ticks", "dist_intraday_vah_ticks", "dist_intraday_val_ticks",
    "position_in_value_area", "volume_at_current_price_pctile",
    # Prior session levels (4) — all DISTANCE
    "dist_prior_session_high_ticks", "dist_prior_session_low_ticks",
    "dist_prior_session_vpoc_ticks", "dist_5d_extreme_ticks",
    # Path memory (5)
    "log_return_60s_bps", "log_return_5min_bps", "log_return_15min_bps",
    "realized_vol_5min_ticks", "trend_strength_5min",
    # Regime / time (6) — cyclical + binary PASS in dataloader
    "tod_sin", "tod_cos", "is_lunch_lull", "is_close_hour", "dow_sin", "dow_cos",
    # ---- NEW v3.2.1 (Priority-1 item #3): LGBM vol predictions ----
    "lgbm_vol_pred_5min",               # LGBM vol model: 5min-ahead forecast
    "lgbm_vol_pred_30min",              # LGBM vol model: 30min-ahead forecast
    # ---- NEW v3.2.1 (Priority-1 item #5): session-phase one-hot ----
    "phase_open",                       # binary: 9:30-10:30 ET
    "phase_morning",                    # binary: 10:30-12:00 ET
    "phase_lunch",                      # binary: 12:00-13:30 ET
    "phase_afternoon",                  # binary: 13:30-close
]

# v3.2.1 (Priority-1 item #1, HC #298 fix): dataloader Z-SCORE EXCLUDE set.
# These columns are already correctly normalized in build_v3_2_1_tier_features.py
# (distance/100, cyclical raw, bounded raw, one-hot binary). Z-scoring them again
# would destroy their semantic reference frame (S/R distance, periodicity, value-area
# membership, etc.). PASS them through dataloader unchanged.
T3_ZSCORE_EXCLUDE: set = {
    # 12 distance features (already /100 in builder)
    "dist_intraday_high_ticks", "dist_intraday_low_ticks", "dist_session_vwap_ticks",
    "dist_prior_session_close_ticks", "dist_prior_session_vwap_ticks",
    "dist_intraday_vpoc_ticks", "dist_intraday_vah_ticks", "dist_intraday_val_ticks",
    "dist_prior_session_high_ticks", "dist_prior_session_low_ticks",
    "dist_prior_session_vpoc_ticks", "dist_5d_extreme_ticks",
    # Bounded raw
    "position_in_value_area",            # {-1, 0, +1} categorical
    "volume_at_current_price_pctile",    # [0, 1]
    # Cyclical raw [-1, +1]
    "tod_sin", "tod_cos", "dow_sin", "dow_cos",
    # Binary one-hot
    "is_lunch_lull", "is_close_hour",
    "phase_open", "phase_morning", "phase_lunch", "phase_afternoon",
}
# Precomputed boolean mask aligned to TIER3_FEATURE_COLS order: True = apply z-score, False = passthrough.
import numpy as _np_for_mask
T3_ZSCORE_APPLY_MASK = _np_for_mask.array(
    [c not in T3_ZSCORE_EXCLUDE for c in TIER3_FEATURE_COLS], dtype=bool
)
del _np_for_mask


def rth_open_ns_for_yyyymmdd(date_str: str) -> int:
    """date_str like '20260427' -> int64 ns UTC for 9:30 ET."""
    import pandas as pd
    y, m, d = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    ts = pd.Timestamp(year=y, month=m, day=d, hour=RTH_OPEN_HOUR_ET,
                      minute=RTH_OPEN_MIN_ET, tz="America/New_York")
    return int(ts.tz_convert("UTC").value)


def yyyymmdd_to_iso(date_str: str) -> str:
    """'20260427' -> '2026-04-27' for parquet partition keys."""
    return f"{date_str[:4]}-{date_str[4:6]}-{date_str[6:8]}"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000")
MLFLOW_EXPERIMENT = "CNNMamba_v3_2_1_long_context"

RANK_NORM_FEATURE_IDXS = [12, 15, 17]  # on smart_v3 event features

# Tier dims
WINDOW_SIZE_T1 = int(os.environ.get("V32_WINDOW_T1", 1500))
WINDOW_SIZE_T2 = int(os.environ.get("V32_WINDOW_T2", 1500))
WINDOW_SIZE_T3 = int(os.environ.get("V32_WINDOW_T3", 1500))
STRIDE = int(os.environ.get("EVENT_STRIDE", 250))
BATCH_SIZE = int(os.environ.get("V32_BATCH_SIZE", 96))
EPOCHS_PER_FOLD = int(os.environ.get("V32_EPOCHS", 5))

N_EVENT_FEATURES = 25  # smart_v3 event cols incl. event_type_id at idx 0 (scalar in raw input)
# v3.2.1 (Priority-1 item #4): event_type embedded as 8-dim learnable (not scalar z-scored).
# After embedding, the effective T1 per-step feature count grows from 25 -> 24 + 8 = 32.
EVENT_TYPE_INPUT_IDX = 0      # event_type_id column in smart_v3 events parquet
EVENT_TYPE_EMBED_DIM = 8
EVENT_TYPE_NUM_CLASSES = 8    # ES MBO event types: trade/add/cancel/modify/halt/...
N_EVENT_FEATURES_POST_EMBED = (N_EVENT_FEATURES - 1) + EVENT_TYPE_EMBED_DIM  # 32
N_PT_FEATURES = 4  # pt_pred_1s/5s/10s + has_pt
N_BOOK_HISTORY_FEATURES = 10
# RAW T1 input dim (what dataloader emits) = unchanged 39: 25 event + 4 PT + 10 book
# POST-EMBED T1 dim (what backbone sees through t1_adapter) = 32 + 4 + 10 = 46
N_T1_FEATURES_RAW = N_EVENT_FEATURES + N_PT_FEATURES + N_BOOK_HISTORY_FEATURES  # 39
N_T1_FEATURES_POST_EMBED = N_EVENT_FEATURES_POST_EMBED + N_PT_FEATURES + N_BOOK_HISTORY_FEATURES  # 46
# Back-compat alias (old name still used downstream):
N_T1_FEATURES = N_T1_FEATURES_RAW  # dataloader still emits 39-d (event_type as scalar idx)
# v3.2 with real Tier 2/3 features (HC #295 — proper engineering):
N_T2_FEATURES = len(TIER2_FEATURE_COLS)                                     # 14 (order-flow per 100ms)
N_T3_FEATURES = len(TIER3_FEATURE_COLS)                                     # 25 (session-context per 1Hz)

# Tier-3 (long) is smaller — lower information density
T3_D_MODEL = 64
T3_N_LAYERS = 2

TRUNK_DIM = 192

# Head set
DIR_REG_HEADS = ["log_ret_1s", "log_ret_5s", "log_ret_10s",
                 "log_ret_30s", "log_ret_60s", "log_ret_5min"]
P_UP_HEADS = ["p_up_5s", "p_up_10s", "p_up_30s", "p_up_60s"]
QUANTILE_HEADS = [
    "log_ret_10s_q10", "log_ret_10s_q50", "log_ret_10s_q90",
    "log_ret_30s_q10", "log_ret_30s_q50", "log_ret_30s_q90",
    "log_ret_60s_q10", "log_ret_60s_q50", "log_ret_60s_q90",
]
PATH_HEADS = ["pred_mfe_30s_ticks", "pred_mae_30s_ticks",
              "pred_mfe_60s_ticks", "pred_mae_60s_ticks"]
TIME_HEADS = ["pred_time_to_mfe_secs"]
REVERSAL_HEADS = ["p_reversal_15s", "p_reversal_30s", "p_reversal_60s"]
VOL_HEADS = ["pred_realized_vol_30s_ticks"]
LEGACY_AUX_HEADS = ["fifo_tp4sl3_net", "fifo_tp8sl5_net",
                    "fifo_tp4sl3_hit_tp", "fifo_tp8sl5_hit_tp"]

ALL_HEAD_NAMES = (DIR_REG_HEADS + P_UP_HEADS + QUANTILE_HEADS +
                  PATH_HEADS + TIME_HEADS + REVERSAL_HEADS +
                  VOL_HEADS + LEGACY_AUX_HEADS)

QUANTILE_TARGETS = {
    "log_ret_10s_q10": ("log_ret_10s", 0.10),
    "log_ret_10s_q50": ("log_ret_10s", 0.50),
    "log_ret_10s_q90": ("log_ret_10s", 0.90),
    "log_ret_30s_q10": ("log_ret_30s", 0.10),
    "log_ret_30s_q50": ("log_ret_30s", 0.50),
    "log_ret_30s_q90": ("log_ret_30s", 0.90),
    "log_ret_60s_q10": ("log_ret_60s", 0.10),
    "log_ret_60s_q50": ("log_ret_60s", 0.50),
    "log_ret_60s_q90": ("log_ret_60s", 0.90),
}

LOSS_LAMBDA = {
    # Directional regression
    "log_ret_1s": 1.0, "log_ret_5s": 1.0, "log_ret_10s": 1.0,
    "log_ret_30s": 1.0, "log_ret_60s": 1.0, "log_ret_5min": 0.5,
    # Direction prob
    "p_up_5s": 0.5, "p_up_10s": 0.5, "p_up_30s": 0.5, "p_up_60s": 0.5,
    # Quantiles
    **{h: 0.5 for h in QUANTILE_HEADS},
    # Path
    "pred_mfe_30s_ticks": 1.0, "pred_mae_30s_ticks": 1.0,
    "pred_mfe_60s_ticks": 1.0, "pred_mae_60s_ticks": 1.0,
    # Time
    "pred_time_to_mfe_secs": 0.5,
    # Reversal
    "p_reversal_15s": 0.5, "p_reversal_30s": 0.5, "p_reversal_60s": 0.5,
    # Vol
    "pred_realized_vol_30s_ticks": 0.5,
    # Legacy aux
    "fifo_tp4sl3_net": 0.1, "fifo_tp8sl5_net": 0.1,
    "fifo_tp4sl3_hit_tp": 0.1, "fifo_tp8sl5_hit_tp": 0.1,
}

# Caps
FIFO_LABEL_CAP_TICKS = 20.0
PATH_LABEL_CAP_TICKS = 50.0
VOL_LABEL_CAP_TICKS = 50.0
LOG_RET_CAP_TICKS = 100.0

# Walk-forward
WF_TRAIN_DAYS = int(os.environ.get("V32_WF_TRAIN_DAYS", 60))
N_FOLDS = int(os.environ.get("V32_N_FOLDS", 10))
FIRST_OOT_MONDAY = os.environ.get("V32_FIRST_OOT_MONDAY", "2026-02-23")


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
        # Purge gap: drop 1 day between train and OOT to prevent label leakage
        # from overlapping look-ahead windows at the train/OOT boundary.
        purge_days = 1
        train_start_idx = max(0, cur_idx - train_days)
        train_dates = sorted_dates[train_start_idx:max(train_start_idx, cur_idx - purge_days)]
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
    def __init__(self, quantile: float):
        super().__init__()
        self.q = quantile

    def forward(self, pred, target):
        diff = target - pred
        return torch.maximum(self.q * diff, (self.q - 1.0) * diff)


# ============================================================
# Multi-Tier Model
# ============================================================
class CNNMambaV321(nn.Module):
    """
    v3.2.1 — Three Mamba branches (T1/T2/T3) + concatenated trunk + 28 heads.

    Deltas vs v3.2 (CNNMambaV32):
      - T1 event_type_id (col idx 0) is embedded via nn.Embedding(8, 8) instead of
        being fed as a z-scored scalar. The remaining 24 event features + 4 PT +
        10 book history are concatenated with the 8-dim embedding for 46 total
        per-event dims (vs. 39 in v3.2). t1_adapter is sized Linear(46 -> 25)
        accordingly.
      - T2 input grows 14 -> 19 (5 missing order-flow features restored).
      - T3 input grows 25 -> 31 (2 LGBM vol pred + 4 session-phase one-hot).
      - Dataloader applies T3_ZSCORE_EXCLUDE mask for the 22 already-normalized cols.

    Backbone params are NOT shared across tiers (different temporal dynamics).
    """
    HEAD_NAMES = ALL_HEAD_NAMES

    def __init__(
        self,
        d_model_t1: int = MAMBA_D_MODEL,
        d_model_t2: int = MAMBA_D_MODEL,
        d_model_t3: int = T3_D_MODEL,
        n_layers_t3: int = T3_N_LAYERS,
        trunk_dim: int = TRUNK_DIM,
        dropout: float = MAMBA_DROPOUT,
    ):
        super().__init__()
        # Tier 1: native MBO events.
        # v3.2.1: event_type_id embedded as 8-dim learnable (Priority-1 item #4).
        # POST-EMBED per-event dim = (25 - 1 scalar event_type) + 8 emb + 4 PT + 10 book = 46.
        # t1_adapter Linear(46 -> 25) so backbone (trained on 25-d) can warmstart cleanly
        # on the 24 non-event-type continuous features (identity columns); the new
        # 8 embedding columns + extras get small random projection.
        self.event_type_emb = nn.Embedding(
            num_embeddings=EVENT_TYPE_NUM_CLASSES,
            embedding_dim=EVENT_TYPE_EMBED_DIM,
        )
        nn.init.normal_(self.event_type_emb.weight, mean=0.0, std=0.02)

        self.t1_adapter = nn.Linear(N_T1_FEATURES_POST_EMBED, N_EVENT_FEATURES)
        with torch.no_grad():
            W = torch.zeros(N_EVENT_FEATURES, N_T1_FEATURES_POST_EMBED)
            # Identity for the 24 continuous event features (cols 0..23 post-embed = orig
            # event cols 1..24 in raw smart_v3 events). They land in adapter input dims 0..23.
            n_continuous_event = N_EVENT_FEATURES - 1  # 24
            W[:n_continuous_event, :n_continuous_event] = torch.eye(n_continuous_event)
            # Remaining adapter input dims 24..45 = 8 event-type emb + 4 PT + 10 book = 22.
            extra_dim = N_T1_FEATURES_POST_EMBED - n_continuous_event  # 22
            W[:, n_continuous_event:] = torch.randn(N_EVENT_FEATURES, extra_dim) * 0.01
            self.t1_adapter.weight.copy_(W)
            self.t1_adapter.bias.zero_()

        self.t1_backbone = CNNMamba(
            d_model=d_model_t1, d_state=MAMBA_D_STATE,
            n_layers=MAMBA_N_LAYERS, dt_rank=MAMBA_DT_RANK,
            d_conv=MAMBA_D_CONV, dropout=dropout,
            n_targets=3,  # head discarded
            cnn_channels=CNN_CHANNELS, cnn_kernel=CNN_KERNEL, cnn_layers=CNN_LAYERS,
        )
        self.t1_backbone.head = nn.Identity()

        # Tier 2: 14 order-flow features per 100ms bucket (real parquet data, HC #295)
        # Linear adapter 14 -> 25 (xavier init; Tier 2 features are a different semantic
        # space from Tier 1 events, so no identity warmstart applies).
        self.t2_adapter = nn.Linear(N_T2_FEATURES, N_EVENT_FEATURES)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.t2_adapter.weight)
            self.t2_adapter.bias.zero_()
        self.t2_backbone = CNNMamba(
            d_model=d_model_t2, d_state=MAMBA_D_STATE,
            n_layers=MAMBA_N_LAYERS, dt_rank=MAMBA_DT_RANK,
            d_conv=MAMBA_D_CONV, dropout=dropout,
            n_targets=3,
            cnn_channels=CNN_CHANNELS, cnn_kernel=CNN_KERNEL, cnn_layers=CNN_LAYERS,
        )
        self.t2_backbone.head = nn.Identity()

        # Tier 3: 25 session-context features per 1Hz snapshot (real parquet data, HC #295)
        # Linear adapter 25 -> 25 (xavier init).
        self.t3_adapter = nn.Linear(N_T3_FEATURES, N_EVENT_FEATURES)
        with torch.no_grad():
            nn.init.xavier_uniform_(self.t3_adapter.weight)
            self.t3_adapter.bias.zero_()
        self.t3_backbone = CNNMamba(
            d_model=d_model_t3, d_state=MAMBA_D_STATE,
            n_layers=n_layers_t3, dt_rank=MAMBA_DT_RANK,
            d_conv=MAMBA_D_CONV, dropout=dropout,
            n_targets=3,
            cnn_channels=max(CNN_CHANNELS // 2, 16),
            cnn_kernel=CNN_KERNEL, cnn_layers=max(CNN_LAYERS - 1, 1),
        )
        self.t3_backbone.head = nn.Identity()

        # Fusion trunk
        fused_dim = d_model_t1 + d_model_t2 + d_model_t3
        self.trunk = nn.Sequential(
            nn.Linear(fused_dim, trunk_dim),
            nn.GELU(),
            nn.LayerNorm(trunk_dim),
            nn.Dropout(dropout),
        )

        # Heads
        self.heads = nn.ModuleDict({
            name: nn.Linear(trunk_dim, 1) for name in self.HEAD_NAMES
        })
        self._init_new_weights()

    def _init_new_weights(self):
        modules = (list(self.trunk.modules()) + list(self.heads.modules()))
        for m in modules:
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """
        batch keys: events_t1 (B, L1, F1), events_t2 (B, L2, F2), events_t3 (B, L3, F3)
        Inputs sanitized with nan_to_num to prevent NaN/Inf propagation (HC #295 NaN fix 2026-05-12).
        """
        e1_raw = torch.nan_to_num(batch["events_t1"], nan=0.0, posinf=10.0, neginf=-10.0)
        # v3.2.1: split event_type_id (col 0) from continuous T1 features, embed, concat.
        # e1_raw is (B, L, 39) — col 0 is event_type_id (long-castable), cols 1..38 are continuous.
        et_ids = e1_raw[..., EVENT_TYPE_INPUT_IDX].clamp(min=0, max=EVENT_TYPE_NUM_CLASSES - 1).long()
        et_emb = self.event_type_emb(et_ids)                          # (B, L, 8)
        e1_continuous = e1_raw[..., EVENT_TYPE_INPUT_IDX + 1:]        # (B, L, 38)
        e1 = torch.cat([e1_continuous, et_emb], dim=-1)               # (B, L, 46)
        x1 = self.t1_adapter(e1)
        _, emb1 = self.t1_backbone(x1, return_embedding=True)

        e2 = torch.nan_to_num(batch["events_t2"], nan=0.0, posinf=10.0, neginf=-10.0)
        x2 = self.t2_adapter(e2)
        _, emb2 = self.t2_backbone(x2, return_embedding=True)

        e3 = torch.nan_to_num(batch["events_t3"], nan=0.0, posinf=10.0, neginf=-10.0)
        x3 = self.t3_adapter(e3)
        _, emb3 = self.t3_backbone(x3, return_embedding=True)

        emb = torch.cat([emb1, emb2, emb3], dim=-1)
        trunk_out = self.trunk(emb)
        return {name: head(trunk_out).squeeze(-1) for name, head in self.heads.items()}

    def load_v3_warmstart(self, ckpt_path: str, device: torch.device) -> Dict[str, int]:
        """
        Load v3 fold_00_best.pt into Tier 1 branch (backbone + trunk).
        v3 had a separate trunk; v3.2 fused trunk has different shape — skip the trunk.
        Tier 2/3 branches init random.

        Returns dict {loaded, skipped, total_v3_keys}.
        """
        stats = {"loaded": 0, "skipped": 0, "total_v3_keys": 0, "init_random": 0}
        if not Path(ckpt_path).exists():
            logger.warning(f"v3 warmstart not found: {ckpt_path} — fully random init")
            return stats
        try:
            ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
            v3_state = ckpt.get("model_state", ckpt)
            stats["total_v3_keys"] = len(v3_state)

            own_state = self.state_dict()
            new_state = dict(own_state)

            # Map v3 backbone keys -> t1_backbone keys
            for k, v in v3_state.items():
                if k.startswith("backbone."):
                    target_k = "t1_backbone." + k[len("backbone."):]
                    if target_k in own_state and own_state[target_k].shape == v.shape:
                        new_state[target_k] = v
                        stats["loaded"] += 1
                    else:
                        stats["skipped"] += 1
                elif k.startswith("trunk."):
                    # v3 trunk has different in-features (d_model vs fused d_model_sum) → skip
                    stats["skipped"] += 1
                elif k.startswith("heads."):
                    # v3 had 7 heads; v3.2 head set is different → skip
                    stats["skipped"] += 1
                else:
                    stats["skipped"] += 1

            self.load_state_dict(new_state, strict=False)
            # Count init-random keys (own_state keys that did NOT get a v3 value)
            v3_targets_used = set()
            for k in v3_state:
                if k.startswith("backbone."):
                    v3_targets_used.add("t1_backbone." + k[len("backbone."):])
            stats["init_random"] = sum(1 for k in own_state if k not in v3_targets_used)

            logger.info(
                f"v3 warmstart: {stats['loaded']} tensors copied into t1_backbone, "
                f"{stats['skipped']} v3 keys skipped, "
                f"{stats['init_random']} v3.2 keys remain init-random "
                f"(from {Path(ckpt_path).name}, val_loss={ckpt.get('val_loss','?')})"
            )
            return stats
        except Exception as e:
            logger.warning(f"v3 warmstart failed: {e}")
            return stats


# ============================================================
# Joint Multi-Head Loss
# ============================================================
class JointMultiHeadLossV32(nn.Module):
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

            if name in QUANTILE_TARGETS:
                tgt_name, _q = QUANTILE_TARGETS[name]
                if tgt_name not in targets:
                    continue
                t = targets[tgt_name]
                m = masks.get(tgt_name, torch.ones_like(p))
                loss_per = self.pinball[name](p, t)
            elif (name.endswith("_hit_tp") or name.startswith("p_up")
                  or name.startswith("p_reversal")):
                if name not in targets:
                    continue
                t = targets[name]
                m = masks.get(name, torch.ones_like(p))
                loss_per = self.bce(p, t)
            elif (name.startswith("pred_") or name.endswith("_net")
                  or name == "pred_realized_vol_30s_ticks"
                  or name.endswith("_ticks") or name == "pred_time_to_mfe_secs"):
                if name not in targets:
                    continue
                t = targets[name]
                m = masks.get(name, torch.ones_like(p))
                loss_per = self.huber(p, t)
            elif name.startswith("log_ret"):
                if name not in targets:
                    continue
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
# Book-history feature derivation (from raw event features)
# ============================================================
def derive_book_history_features(events_25: np.ndarray) -> np.ndarray:
    """
    Derive 10 book-history features from the 25 smart_v3 event features.

    Inputs columns assumed (smart_v3, partial mapping):
      idx 12: book_imbalance (-1..+1)
      idx 15: queue_depth_ratio
      idx 17: signal_persistence

    Since we don't have direct top-10 book here, we approximate with rolling
    statistics over the event sequence. This is an interim implementation —
    full top-10 features should be backfilled from raw MBO depth.

    Returns: (N, 10) float32 array.
    """
    n = len(events_25)
    out = np.zeros((n, N_BOOK_HISTORY_FEATURES), dtype=np.float32)
    if n < 50:
        return out

    book_imb = events_25[:, 12].astype(np.float32)
    queue_ratio = events_25[:, 15].astype(np.float32)
    persistence = events_25[:, 17].astype(np.float32)

    # 0: rolling book imbalance (5-event)
    # 1: rolling book imbalance (20-event)
    # 2: persistence ratio (5-event mean)
    # 3: persistence ratio (20-event mean)
    # 4: queue-depth ratio (5-event mean)
    # 5: queue-depth ratio (20-event mean)
    # 6: book pressure delta (current - 5-event-prior imbalance)
    # 7: book pressure delta (current - 20-event-prior imbalance)
    # 8: rolling std of book_imbalance (20-event)
    # 9: rolling std of queue_ratio (20-event)
    def rolling_mean(arr, w):
        kernel = np.ones(w, dtype=np.float32) / w
        # CAUSAL: mode='full'[:N] — no look-forward (HC audit 2026-07-01)
        return np.convolve(arr, kernel, mode="full")[:len(arr)]

    out[:, 0] = rolling_mean(book_imb, 5)
    out[:, 1] = rolling_mean(book_imb, 20)
    out[:, 2] = rolling_mean(persistence, 5)
    out[:, 3] = rolling_mean(persistence, 20)
    out[:, 4] = rolling_mean(queue_ratio, 5)
    out[:, 5] = rolling_mean(queue_ratio, 20)

    # Pressure deltas (current - lagged)
    pad5 = np.concatenate([np.full(5, book_imb[0]), book_imb[:-5]])
    pad20 = np.concatenate([np.full(20, book_imb[0]), book_imb[:-20]])
    out[:, 6] = book_imb - pad5
    out[:, 7] = book_imb - pad20

    # Rolling std (approx via rolling mean of squared deviation)
    mean20 = out[:, 1]
    sq_dev = (book_imb - mean20) ** 2
    out[:, 8] = np.sqrt(rolling_mean(sq_dev, 20) + 1e-8)
    mean20_q = out[:, 5]
    sq_dev_q = (queue_ratio - mean20_q) ** 2
    out[:, 9] = np.sqrt(rolling_mean(sq_dev_q, 20) + 1e-8)

    return out


def aggregate_t2_bucket(events_t1: np.ndarray, bucket_size: int = 16) -> np.ndarray:
    """
    Aggregate Tier 1 events into Tier 2 buckets.

    We don't have explicit timestamps here, so we approximate 100ms buckets
    by chunking every ~bucket_size events (smart_v3 stride is roughly 1 event
    per 6-10ms during active periods).

    Args:
        events_t1: (L1, F) tier-1 feature matrix
        bucket_size: events per bucket (default 16, ~100ms at typical cadence)

    Returns:
        (L1 // bucket_size, F) tier-2 aggregates (mean per bucket).
        Padded/truncated to L2 = WINDOW_SIZE_T2 by the caller.
    """
    L1, F = events_t1.shape
    n_buckets = max(L1 // bucket_size, 1)
    truncated = events_t1[:n_buckets * bucket_size]
    return truncated.reshape(n_buckets, bucket_size, F).mean(axis=1).astype(np.float32)


# ============================================================
# Dataset
# ============================================================
class SmartV32Dataset(Dataset):
    """
    Yields (events_dict, targets, masks) where events_dict has events_t1/t2/t3.
    """
    HEAD_NAMES = ALL_HEAD_NAMES

    def __init__(
        self,
        data_dir: Path,
        fifo_label_dir: Path,
        alpha_label_dir: Path,
        pt_pred_dir: Path,
        dates: List[str],
        tier2_parquet_root: Optional[Path] = None,
        tier3_parquet_root: Optional[Path] = None,
        window_t1: int = WINDOW_SIZE_T1,
        window_t2: int = WINDOW_SIZE_T2,
        window_t3: int = WINDOW_SIZE_T3,
        stride: int = STRIDE,
        feature_stats: Optional[Dict] = None,
        cache_size: int = 4,
        require_alpha_labels: bool = False,  # Set False so legacy-only days still train
    ):
        self.data_dir = Path(data_dir)
        self.fifo_label_dir = Path(fifo_label_dir)
        self.alpha_label_dir = Path(alpha_label_dir)
        self.pt_pred_dir = Path(pt_pred_dir)
        self.tier2_parquet_root = Path(tier2_parquet_root) if tier2_parquet_root else None
        self.tier3_parquet_root = Path(tier3_parquet_root) if tier3_parquet_root else None
        self.dates = dates
        self.window_t1 = window_t1
        self.window_t2 = window_t2
        self.window_t3 = window_t3
        self.stride = stride
        self.cache_size = cache_size
        self.require_alpha_labels = require_alpha_labels
        self.feature_stats = feature_stats  # {"mean_t1": .., "std_t1": .., "mean_t2": .., "std_t2": .., "mean_t3": .., "std_t3": ..}

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
            return None

        try:
            ev = np.load(events_path, allow_pickle=True)
        except Exception as e:
            logger.warning(f"Failed to load {events_path.name}: {e}")
            return None

        events_25 = ev["events"].astype(np.float32)
        labels_1s = ev["labels_1s"].astype(np.float32)
        labels_5s = ev["labels_5s"].astype(np.float32)
        labels_10s = ev["labels_10s"].astype(np.float32)
        labels_30s = ev["labels_30s"].astype(np.float32) if "labels_30s" in ev else None
        # Timestamps (int64 ns UTC) — needed to align target event with Tier 2/3 parquet rows
        timestamps_ns = ev["timestamps"].astype(np.int64) if "timestamps" in ev else None

        # Per-day rank-norm on event features
        n = len(events_25)
        if n > 1:
            for fidx in RANK_NORM_FEATURE_IDXS:
                col = events_25[:, fidx]
                ranks = scipy.stats.rankdata(col, method="average") - 1.0
                events_25[:, fidx] = ((ranks / max(n - 1, 1)) - 0.5) * 2.0

        # PatchTST predictions (optional)
        if pt_path.exists():
            try:
                pt = np.load(pt_path, allow_pickle=True)
                pt_1s = pt["pt_pred_1s"].astype(np.float32)
                pt_5s = pt["pt_pred_5s"].astype(np.float32)
                pt_10s = pt["pt_pred_10s"].astype(np.float32)
                has_pt = pt["has_pt_pred"].astype(np.float32)
            except Exception:
                pt_1s = np.zeros(n, dtype=np.float32)
                pt_5s = np.zeros(n, dtype=np.float32)
                pt_10s = np.zeros(n, dtype=np.float32)
                has_pt = np.zeros(n, dtype=np.float32)
        else:
            pt_1s = np.zeros(n, dtype=np.float32)
            pt_5s = np.zeros(n, dtype=np.float32)
            pt_10s = np.zeros(n, dtype=np.float32)
            has_pt = np.zeros(n, dtype=np.float32)

        # Book-history features (derived on-the-fly from smart_v3 features)
        book_hist = derive_book_history_features(events_25)

        # T1 features (39): 25 event + 4 pt + 10 book-history
        events_t1 = np.concatenate([
            events_25,
            pt_1s.reshape(-1, 1),
            pt_5s.reshape(-1, 1),
            pt_10s.reshape(-1, 1),
            has_pt.reshape(-1, 1),
            book_hist,
        ], axis=1).astype(np.float32)

        # Alpha labels (optional)
        alpha = {}
        if alpha_path.exists():
            try:
                al = np.load(alpha_path, allow_pickle=True)
                alpha = {k: al[k].astype(np.float32) for k in al.keys()}
            except Exception as e:
                logger.warning(f"Failed to read alpha labels {date_str}: {e}")

        # FIFO labels (optional)
        fifo = {}
        if fifo_path.exists():
            try:
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
            except Exception as e:
                logger.warning(f"Failed to read FIFO labels {date_str}: {e}")

        # Tier 2 sparse parquet (bucket_idx + 14 features). Sparse: only buckets with events.
        iso_date = yyyymmdd_to_iso(date_str)
        t2_bucket_idx = np.zeros(0, dtype=np.int64)
        t2_features = np.zeros((0, N_T2_FEATURES), dtype=np.float32)
        if self.tier2_parquet_root is not None:
            t2_path = self.tier2_parquet_root / f"date={iso_date}" / "part-0.parquet"
            if t2_path.exists():
                try:
                    t2_df = pq.ParquetFile(t2_path).read().to_pandas()
                    t2_bucket_idx = t2_df["bucket_idx"].to_numpy(dtype=np.int64)
                    t2_features = t2_df[TIER2_FEATURE_COLS].to_numpy(dtype=np.float32)
                except Exception as e:
                    logger.warning(f"Failed to read tier2 parquet for {date_str}: {e}")

        # Tier 3 dense parquet (RTH_SECONDS_PER_DAY rows × 25 features, indexed by second_idx)
        t3_features = np.zeros((RTH_SECONDS_PER_DAY, N_T3_FEATURES), dtype=np.float32)
        if self.tier3_parquet_root is not None:
            t3_path = self.tier3_parquet_root / f"date={iso_date}" / "part-0.parquet"
            if t3_path.exists():
                try:
                    t3_df = pq.ParquetFile(t3_path).read().to_pandas()
                    n3 = min(len(t3_df), RTH_SECONDS_PER_DAY)
                    # tier3 builder writes second_idx 0..23399 in order
                    t3_features[:n3] = t3_df[TIER3_FEATURE_COLS].to_numpy(dtype=np.float32)[:n3]
                except Exception as e:
                    logger.warning(f"Failed to read tier3 parquet for {date_str}: {e}")

        rth_open_ns = rth_open_ns_for_yyyymmdd(date_str)

        return {
            "events_t1": events_t1,
            "timestamps_ns": timestamps_ns,
            "rth_open_ns": rth_open_ns,
            "t2_bucket_idx": t2_bucket_idx,
            "t2_features": t2_features,
            "t3_features": t3_features,
            "labels_1s": labels_1s,
            "labels_5s": labels_5s,
            "labels_10s": labels_10s,
            "labels_30s": labels_30s,
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
            # HC #300: explicit GC after eviction. Per-worker day-cache leak was the
            # root cause of the 5 v3.2 OOM crashes today (2026-05-12) — Python refcounts
            # don't release fat numpy arrays until next GC cycle, anon RSS climbs to ~15GB
            # per worker over ~3.3h regardless of bs/persistent_workers settings.
            gc.collect()
        return d

    def _build_index(self):
        for date_str in self.dates:
            events_path = self.data_dir / f"{date_str}_mbo_events.npz"
            if not events_path.exists():
                continue
            try:
                ev = np.load(events_path, allow_pickle=True)
                n_events = len(ev["events"])
                lab1 = ev["labels_1s"]
            except Exception as e:
                logger.warning(f"Failed to read {events_path.name}: {e}")
                continue
            window_k = 0
            for start in range(0, n_events - self.window_t1 + 1, self.stride):
                end = start + self.window_t1
                label_idx = end - 1
                if not np.isnan(lab1[label_idx]):
                    self.sample_index.append((date_str, start, window_k))
                window_k += 1
            del ev
        logger.info(
            f"Dataset: {len(self.dates)} days, {len(self.sample_index)} samples "
            f"(window_t1={self.window_t1}, stride={self.stride})"
        )

    def _compute_feature_stats(self):
        """Compute per-fold z-score stats for T1 (event-level), T2 (100ms buckets),
        T3 (1Hz session-context). All from train_dates only — no leakage."""
        logger.info(f"Computing T1/T2/T3 feature stats from {len(self.dates)} train dates...")

        # T1: streamed mean/var via accumulators
        n_t1 = N_T1_FEATURES
        sum1 = np.zeros(n_t1, dtype=np.float64)
        sq1 = np.zeros(n_t1, dtype=np.float64)
        cnt1 = 0
        # T2 / T3 accumulators
        n_t2 = N_T2_FEATURES
        sum2 = np.zeros(n_t2, dtype=np.float64)
        sq2 = np.zeros(n_t2, dtype=np.float64)
        cnt2 = 0
        n_t3 = N_T3_FEATURES
        sum3 = np.zeros(n_t3, dtype=np.float64)
        sq3 = np.zeros(n_t3, dtype=np.float64)
        cnt3 = 0

        for date_str in self.dates:
            day = self._load_day_raw(date_str)
            if day is None:
                continue
            ev = day["events_t1"].astype(np.float64)
            sum1 += ev.sum(axis=0)
            sq1 += (ev ** 2).sum(axis=0)
            cnt1 += len(ev)
            if day["t2_features"].shape[0] > 0:
                t2 = day["t2_features"].astype(np.float64)
                sum2 += t2.sum(axis=0)
                sq2 += (t2 ** 2).sum(axis=0)
                cnt2 += len(t2)
            if day["t3_features"].shape[0] > 0:
                t3 = day["t3_features"].astype(np.float64)
                sum3 += t3.sum(axis=0)
                sq3 += (t3 ** 2).sum(axis=0)
                cnt3 += len(t3)

        def _finalize(s, sq, cnt, n_feat):
            if cnt == 0:
                return (np.zeros(n_feat, dtype=np.float32),
                        np.ones(n_feat, dtype=np.float32))
            m = (s / cnt).astype(np.float32)
            v = (sq / cnt) - (m.astype(np.float64) ** 2)
            sd = np.sqrt(np.maximum(v, 1e-8)).astype(np.float32)
            return m, sd

        mean_t1, std_t1 = _finalize(sum1, sq1, cnt1, n_t1)
        mean_t2, std_t2 = _finalize(sum2, sq2, cnt2, n_t2)
        mean_t3, std_t3 = _finalize(sum3, sq3, cnt3, n_t3)

        self.feature_stats = {
            "mean_t1": mean_t1, "std_t1": std_t1,
            "mean_t2": mean_t2, "std_t2": std_t2,
            "mean_t3": mean_t3, "std_t3": std_t3,
        }
        logger.info(
            f"Feature stats: T1 n={cnt1} ({n_t1} feats), "
            f"T2 n={cnt2} ({n_t2} feats), T3 n={cnt3} ({n_t3} feats)"
        )

    def get_feature_stats(self) -> Dict:
        return self.feature_stats

    def __len__(self) -> int:
        return len(self.sample_index)

    def _build_tier_windows(
        self,
        events_t1_window: np.ndarray,
        target_ts_ns: int,
        rth_open_ns: int,
        t2_bucket_idx: np.ndarray,
        t2_features: np.ndarray,
        t3_features: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Build T1/T2/T3 windows keyed by the target event's timestamp.

        T1: per-event window (L1, F1=39), z-scored.
        T2: dense slice of 100ms order-flow buckets ending at the target's bucket
            (window_t2 buckets × N_T2_FEATURES=14). Missing buckets → zeros (then z-scored).
            Per HC #295: real order-flow data, not bucket aggregates of T1.
        T3: dense slice of 1Hz session-context snapshots ending at the target's second
            (window_t3 seconds × N_T3_FEATURES=25). Pre-RTH or post-RTH seconds → zeros.
        """
        # ---------- T1 z-score ----------
        # Std floor 1e-3 + ±10 clip prevents zero-padded rows from blowing up
        # when a feature has near-constant value across the train window.
        if self.feature_stats is not None and "mean_t1" in self.feature_stats:
            events_t1_window = (
                (events_t1_window - self.feature_stats["mean_t1"])
                / np.maximum(self.feature_stats["std_t1"], 1e-3)
            )
            np.clip(events_t1_window, -10.0, 10.0, out=events_t1_window)

        # ---------- T2: 100ms order-flow buckets ----------
        events_t2 = np.zeros((self.window_t2, N_T2_FEATURES), dtype=np.float32)
        target_t2_bidx = int((target_ts_ns - rth_open_ns) // TIER2_BUCKET_NS)
        t2_win_start = target_t2_bidx - self.window_t2 + 1  # inclusive
        if t2_bucket_idx.size > 0:
            # Sparse buckets → dense window via direct indexing
            in_win = (t2_bucket_idx >= t2_win_start) & (t2_bucket_idx <= target_t2_bidx)
            if in_win.any():
                rel = (t2_bucket_idx[in_win] - t2_win_start).astype(np.int64)
                events_t2[rel] = t2_features[in_win]
        if self.feature_stats is not None and "mean_t2" in self.feature_stats:
            events_t2 = (
                (events_t2 - self.feature_stats["mean_t2"])
                / np.maximum(self.feature_stats["std_t2"], 1e-3)
            )
            np.clip(events_t2, -10.0, 10.0, out=events_t2)

        # ---------- T3: 1Hz session-context snapshots ----------
        events_t3 = np.zeros((self.window_t3, N_T3_FEATURES), dtype=np.float32)
        target_t3_sidx = int((target_ts_ns - rth_open_ns) // TIER3_SNAP_NS)
        t3_win_start = target_t3_sidx - self.window_t3 + 1
        # absolute second indices we want
        abs_idx = np.arange(t3_win_start, target_t3_sidx + 1, dtype=np.int64)
        valid = (abs_idx >= 0) & (abs_idx < RTH_SECONDS_PER_DAY)
        if valid.any():
            events_t3[valid] = t3_features[abs_idx[valid]]
        if self.feature_stats is not None and "mean_t3" in self.feature_stats:
            # v3.2.1 (HC #298 fix, Priority-1 item #1): apply z-score ONLY to columns
            # not in T3_ZSCORE_EXCLUDE. The 22 excluded cols (12 distance, 4 cyclical,
            # 6 binary/categorical/bounded, 4 phase-onehot) are passed through unchanged.
            mean_t3 = self.feature_stats["mean_t3"]
            std_t3 = np.maximum(self.feature_stats["std_t3"], 1e-3)
            normed = (events_t3 - mean_t3) / std_t3
            np.clip(normed, -10.0, 10.0, out=normed)
            # Write back only where mask True; leave passthrough cols unchanged.
            events_t3[:, T3_ZSCORE_APPLY_MASK] = normed[:, T3_ZSCORE_APPLY_MASK]

        return (
            events_t1_window.astype(np.float32),
            events_t2.astype(np.float32),
            events_t3.astype(np.float32),
        )

    def __getitem__(self, idx: int):
        date_str, start, window_k = self.sample_index[idx]
        end = start + self.window_t1
        day = self._get_day(date_str)
        if day is None:
            raise RuntimeError(f"Failed to load {date_str}")
        events_t1_window = day["events_t1"][start:end]

        label_idx = end - 1
        # Anchor target timestamp = ts of the last event in the T1 window.
        # If timestamps unavailable (legacy npz), fall back to a synthetic anchor near
        # RTH open (this disables real T2/T3 alignment but keeps shape consistent).
        ts_arr = day.get("timestamps_ns")
        if ts_arr is not None and label_idx < len(ts_arr):
            target_ts_ns = int(ts_arr[label_idx])
        else:
            target_ts_ns = int(day["rth_open_ns"])

        events_t1, events_t2, events_t3 = self._build_tier_windows(
            events_t1_window,
            target_ts_ns=target_ts_ns,
            rth_open_ns=int(day["rth_open_ns"]),
            t2_bucket_idx=day["t2_bucket_idx"],
            t2_features=day["t2_features"],
            t3_features=day["t3_features"],
        )
        lab1 = day["labels_1s"][label_idx]
        lab5 = day["labels_5s"][label_idx]
        lab10 = day["labels_10s"][label_idx]
        labs30_arr = day.get("labels_30s")
        lab30 = labs30_arr[label_idx] if labs30_arr is not None else np.nan

        alpha = day.get("alpha", {})

        def _alpha_at(k):
            arr = alpha.get(k)
            if arr is None or label_idx >= len(arr):
                return np.nan
            v = arr[label_idx]
            return float(v) if v is not None else np.nan

        targets: Dict[str, np.float32] = {}
        masks: Dict[str, np.float32] = {}

        # Directional regression
        for k_name, v in (("log_ret_1s", lab1), ("log_ret_5s", lab5),
                          ("log_ret_10s", lab10), ("log_ret_30s", lab30)):
            if v is None or (isinstance(v, float) and np.isnan(v)):
                targets[k_name] = np.float32(0.0); masks[k_name] = np.float32(0.0)
            else:
                cap = PATH_LABEL_CAP_TICKS if "30s" in k_name else LOG_RET_CAP_TICKS
                targets[k_name] = np.float32(np.clip(v, -cap, cap))
                masks[k_name] = np.float32(1.0)

        # 60s and 5min: not in v3 labels — try alpha labels, else mask out
        lab60 = _alpha_at("log_ret_60s")
        if np.isnan(lab60):
            targets["log_ret_60s"] = np.float32(0.0); masks["log_ret_60s"] = np.float32(0.0)
        else:
            targets["log_ret_60s"] = np.float32(np.clip(lab60, -PATH_LABEL_CAP_TICKS, PATH_LABEL_CAP_TICKS))
            masks["log_ret_60s"] = np.float32(1.0)
        lab5min = _alpha_at("log_ret_5min")
        if np.isnan(lab5min):
            targets["log_ret_5min"] = np.float32(0.0); masks["log_ret_5min"] = np.float32(0.0)
        else:
            targets["log_ret_5min"] = np.float32(np.clip(lab5min, -PATH_LABEL_CAP_TICKS, PATH_LABEL_CAP_TICKS))
            masks["log_ret_5min"] = np.float32(1.0)

        # p_up heads
        for k_name, lr in (("p_up_5s", lab5), ("p_up_10s", lab10),
                           ("p_up_30s", lab30), ("p_up_60s", lab60)):
            if lr is None or (isinstance(lr, float) and np.isnan(lr)):
                targets[k_name] = np.float32(0.0); masks[k_name] = np.float32(0.0)
            else:
                targets[k_name] = np.float32(1.0 if lr > 0 else 0.0)
                masks[k_name] = np.float32(1.0)

        # Quantile heads: inherit target+mask from parent regression head
        for q_name, (parent, _) in QUANTILE_TARGETS.items():
            targets[q_name] = targets.get(parent, np.float32(0.0))
            masks[q_name] = masks.get(parent, np.float32(0.0))

        # Path heads (30s + 60s)
        for horizon, alpha_key_mfe, alpha_key_mae in (
            ("30s", "mfe_30s_ticks", "mae_30s_ticks"),
            ("60s", "mfe_60s_ticks", "mae_60s_ticks"),
        ):
            mfe_k = f"pred_mfe_{horizon}_ticks"
            mae_k = f"pred_mae_{horizon}_ticks"
            mfe = _alpha_at(alpha_key_mfe)
            mae = _alpha_at(alpha_key_mae)
            if np.isnan(mfe):
                targets[mfe_k] = np.float32(0.0); masks[mfe_k] = np.float32(0.0)
            else:
                targets[mfe_k] = np.float32(np.clip(mfe, -PATH_LABEL_CAP_TICKS, PATH_LABEL_CAP_TICKS))
                masks[mfe_k] = np.float32(1.0)
            if np.isnan(mae):
                targets[mae_k] = np.float32(0.0); masks[mae_k] = np.float32(0.0)
            else:
                targets[mae_k] = np.float32(np.clip(mae, -PATH_LABEL_CAP_TICKS, PATH_LABEL_CAP_TICKS))
                masks[mae_k] = np.float32(1.0)

        # Time-to-MFE
        tmfe = _alpha_at("time_to_mfe_secs")
        if np.isnan(tmfe):
            targets["pred_time_to_mfe_secs"] = np.float32(0.0); masks["pred_time_to_mfe_secs"] = np.float32(0.0)
        else:
            targets["pred_time_to_mfe_secs"] = np.float32(tmfe)
            masks["pred_time_to_mfe_secs"] = np.float32(1.0)

        # Reversal
        for k_name in REVERSAL_HEADS:
            v = _alpha_at(k_name)
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

        # Legacy FIFO
        for k_name in LEGACY_AUX_HEADS:
            targets[k_name] = np.float32(0.0); masks[k_name] = np.float32(0.0)
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

        events_dict = {
            "events_t1": torch.from_numpy(events_t1),
            "events_t2": torch.from_numpy(events_t2),
            "events_t3": torch.from_numpy(events_t3),
        }
        targets_t = {k: torch.tensor(v) for k, v in targets.items()}
        masks_t = {k: torch.tensor(v) for k, v in masks.items()}
        return events_dict, targets_t, masks_t


def collate_v32(batch):
    events_list, targets_list, masks_list = zip(*batch)
    events = {
        k: torch.stack([e[k] for e in events_list], dim=0)
        for k in events_list[0].keys()
    }
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


def evaluate_v32(
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
            events = {k: v.to(device, non_blocking=True) for k, v in events.items()}
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
    for h in DIR_REG_HEADS:
        if h not in preds_out:
            continue
        m = masks_out[h] > 0
        if m.sum() > 20:
            metrics[f"ic_{h}"] = compute_ic(preds_out[h][m], targets_out[h][m])
        else:
            metrics[f"ic_{h}"] = float("nan")

    # Path heads correlation
    for h in PATH_HEADS:
        if h not in preds_out:
            continue
        m = masks_out[h] > 0
        if m.sum() > 20:
            metrics[f"corr_{h}"] = compute_ic(preds_out[h][m], targets_out[h][m])
        else:
            metrics[f"corr_{h}"] = float("nan")

    return metrics, preds_out, targets_out, masks_out


# ============================================================
# Train one fold
# ============================================================
def train_one_fold_v32(
    model: CNNMambaV321,
    train_loader: DataLoader,
    oot_loader: DataLoader,
    fold_idx: int,
    output_dir: Path,
    mlflow_run,
    device: torch.device,
    total_train_steps: int,
    use_amp: bool = True,
    resume_state: Optional[Dict] = None,  # HC #296: full-state resume
) -> Dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = WarmupCosineScheduler(
        optimizer, warmup_steps=WARMUP_STEPS, total_steps=total_train_steps
    )
    loss_fn = JointMultiHeadLossV32(LOSS_LAMBDA)

    amp_ctx = (
        torch.amp.autocast("cuda", dtype=torch.bfloat16)
        if use_amp and device.type == "cuda"
        else torch.amp.autocast("cpu", enabled=False)
    )

    best_val_loss = float("inf")
    global_step = 0
    total_batches = len(train_loader)

    # HC #296: resume-from-intra-ckpt logic.
    # If a v2 (full-state) ckpt is passed, restore optimizer/scheduler/RNG and
    # skip the data loader iterator to the saved batch index within the saved epoch.
    # If a v1 (model-only) legacy ckpt is passed, model_state is already loaded
    # upstream as a warmstart; we just start fresh from Ep 0 batch 0.
    resume_epoch = 0
    resume_batch = 0
    if resume_state is not None and resume_state.get("ckpt_version", 1) >= 2:
        try:
            optimizer.load_state_dict(resume_state["optimizer_state"])
            if resume_state.get("scheduler_state") is not None and hasattr(scheduler, "load_state_dict"):
                scheduler.load_state_dict(resume_state["scheduler_state"])
            else:
                # Manual scheduler advance to saved step
                for _ in range(int(resume_state.get("scheduler_step", 0))):
                    scheduler.step()
            # HC #300: torch.set_rng_state requires CPU torch.ByteTensor (dtype=uint8).
            # After torch.load(), the saved tensor may come back with a different dtype or
            # on a non-CPU device (esp. if map_location moved things). Cast explicitly to
            # avoid the "RNG state must be a torch.ByteTensor" RuntimeError that wasted
            # 4 fold-0 restarts on 2026-05-12.
            if resume_state.get("torch_rng_state") is not None:
                _trs = resume_state["torch_rng_state"]
                if hasattr(_trs, "cpu"):
                    _trs = _trs.cpu()
                if hasattr(_trs, "to"):
                    _trs = _trs.to(torch.uint8)
                torch.set_rng_state(_trs)
            if resume_state.get("cuda_rng_state") is not None and torch.cuda.is_available():
                _crs = resume_state["cuda_rng_state"]
                if hasattr(_crs, "cpu"):
                    _crs = _crs.cpu()
                if hasattr(_crs, "to"):
                    _crs = _crs.to(torch.uint8)
                torch.cuda.set_rng_state(_crs)
            resume_epoch = int(resume_state.get("epoch", 0))
            resume_batch = int(resume_state.get("batch", 0))
            global_step = int(resume_state.get("global_step", 0))
            best_val_loss = float(resume_state.get("best_val_loss", float("inf")))
            logger.info(
                f"  Fold {fold_idx} RESUMED from intra_ckpt: "
                f"epoch={resume_epoch} batch={resume_batch} global_step={global_step}"
            )
            print(
                f">>> v3.2 RESUME fold {fold_idx} from Ep {resume_epoch+1} batch {resume_batch}",
                flush=True,
            )
        except Exception as e:
            logger.warning(f"Resume from intra_ckpt failed ({e}); starting fold from scratch")
            resume_epoch = 0
            resume_batch = 0
            global_step = 0
            best_val_loss = float("inf")

    print(f">>> v3.2 train_one_fold {fold_idx}: {EPOCHS_PER_FOLD} epochs x {total_batches} batches",
          flush=True)

    for epoch in range(EPOCHS_PER_FOLD):
        # Skip already-completed epochs on resume
        if epoch < resume_epoch:
            continue
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        epoch_start = time.time()
        comp_acc = defaultdict(float)

        # If resuming inside this epoch, fast-forward the data iterator to the saved batch.
        # Since shuffle=False (deterministic order), iterating-and-skipping reproduces
        # the same batch sequence the prior run consumed.
        skip_until = resume_batch if epoch == resume_epoch else 0
        if skip_until > 0:
            logger.info(
                f"  Fold {fold_idx} Ep {epoch+1}: fast-forwarding loader to batch {skip_until}"
            )
            print(
                f">>> fast-forward fold {fold_idx} Ep {epoch+1} to batch {skip_until}",
                flush=True,
            )

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

            # NaN-guard (HC #295 NaN fix 2026-05-12): if loss non-finite, skip backward/step
            # to prevent Adam state corruption from a single bad batch.
            loss_finite = bool(torch.isfinite(loss).item())
            if loss_finite:
                loss.backward()
                # Sanitize gradients before clipping — any NaN/Inf grad → 0
                for p in model.parameters():
                    if p.grad is not None:
                        torch.nan_to_num_(p.grad, nan=0.0, posinf=0.0, neginf=0.0)
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
            else:
                if (n_batches % 100) == 0:
                    logger.warning(
                        f"  Fold {fold_idx} Ep {epoch+1} batch {n_batches}: "
                        f"non-finite loss — skipping step (HC#295 guard)"
                    )
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
                msg = (
                    f"  Fold {fold_idx} Ep {epoch+1} Batch {n_batches}/{total_batches} | "
                    f"Loss: {epoch_loss/n_batches:.4f} | Elapsed: {elapsed:.0f}s | ETA: {eta:.0f}s"
                )
                print(msg, flush=True)
                logger.info(msg)

            if n_batches % 500 == 0:
                # HC #296: full-state intra_ckpt for resume-from-intra-ckpt.
                # Save model + optimizer + scheduler step + RNG so we can exactly
                # restart from the saved batch on OOM/crash.
                ckpt = {
                    "model_state": model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "scheduler_step": getattr(scheduler, "_step_count", global_step),
                    "scheduler_state": (
                        scheduler.state_dict()
                        if hasattr(scheduler, "state_dict") else None
                    ),
                    "torch_rng_state": torch.get_rng_state(),
                    "cuda_rng_state": (
                        torch.cuda.get_rng_state() if torch.cuda.is_available() else None
                    ),
                    "fold": fold_idx, "epoch": epoch, "batch": n_batches,
                    "global_step": global_step, "best_val_loss": best_val_loss,
                    "ckpt_version": 2,  # v1 = model-only (pre-HC#296), v2 = full-state
                }
                torch.save(ckpt, output_dir / f"fold_{fold_idx:02d}_intra_ckpt.pt")

        avg_loss = epoch_loss / max(n_batches, 1)
        epoch_time = time.time() - epoch_start
        comp_avg = {k: v / max(n_batches, 1) for k, v in comp_acc.items()}

        # OOT eval
        val_metrics, _, _, _ = evaluate_v32(model, oot_loader, loss_fn, device, use_amp=use_amp)
        msg = (
            f"Fold {fold_idx:02d} Ep {epoch+1}/{EPOCHS_PER_FOLD} | "
            f"TrLoss {avg_loss:.4f} | OOT Loss {val_metrics['loss']:.4f} | "
            f"OOT IC 1s/5s/10s/30s = "
            f"{val_metrics.get('ic_log_ret_1s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_5s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_10s', float('nan')):.4f}/"
            f"{val_metrics.get('ic_log_ret_30s', float('nan')):.4f} | "
            f"corr MFE30s/MAE30s = "
            f"{val_metrics.get('corr_pred_mfe_30s_ticks', float('nan')):.4f}/"
            f"{val_metrics.get('corr_pred_mae_30s_ticks', float('nan')):.4f} | "
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
                    "model": "CNNMambaV321",
                    "window_t1": WINDOW_SIZE_T1,
                    "window_t2": WINDOW_SIZE_T2,
                    "window_t3": WINDOW_SIZE_T3,
                    "n_t1_features": N_T1_FEATURES,
                    "n_t2_features": N_T2_FEATURES,
                    "n_t3_features": N_T3_FEATURES,
                    "head_names": ALL_HEAD_NAMES,
                },
            }, output_dir / f"fold_{fold_idx:02d}_best.pt")

    return {"best_val_loss": best_val_loss}


# ============================================================
# Walk-forward driver
# ============================================================
def run_weekly_wf_v32(
    data_dir: Path,
    fifo_label_dir: Path,
    alpha_label_dir: Path,
    pt_pred_dir: Path,
    output_dir: Path,
    device: torch.device,
    n_folds: int = N_FOLDS,
    train_days: int = WF_TRAIN_DAYS,
    warmstart_ckpt: str = V3_WARMSTART_CKPT,
    tier2_parquet_root: Optional[Path] = None,
    tier3_parquet_root: Optional[Path] = None,
    resume_from_intra_ckpt: Optional[Path] = None,  # HC #296
):
    output_dir.mkdir(parents=True, exist_ok=True)

    npz_files = sorted(data_dir.glob("*_mbo_events.npz"))
    available_dates = sorted([date_from_path(p) for p in npz_files if date_from_path(p)])
    if not available_dates:
        logger.error(f"No MBO events found in {data_dir}")
        return {}
    logger.info(f"Found {len(available_dates)} dates: {available_dates[0]} → {available_dates[-1]}")

    folds = build_weekly_fold_schedule(available_dates, n_folds=n_folds, train_days=train_days)
    logger.info(f"Built {len(folds)} weekly folds (anchor={FIRST_OOT_MONDAY})")
    for f in folds:
        logger.info(
            f"  Fold {f['fold']}: train {f['train_start']}→{f['train_end']} ({len(f['train_dates'])}d) "
            f"OOT {f['oot_start']}→{f['oot_end']} ({len(f['oot_dates'])}d)"
        )

    with open(output_dir / "fold_schedule.json", "w") as fh:
        json.dump(folds, fh, indent=2, default=str)

    use_amp = device.type == "cuda"

    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(
                run_name=f"v3.2_{time.strftime('%Y%m%d_%H%M')}_{socket.gethostname()}",
                tags={"model_family": "cnn_mamba_v3.2", "version": "3.2"},
            )
            gpu_name = torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu"
            mlflow.log_params({
                "model": "CNNMambaV321",
                "window_t1": WINDOW_SIZE_T1,
                "window_t2": WINDOW_SIZE_T2,
                "window_t3": WINDOW_SIZE_T3,
                "stride": STRIDE,
                "n_t1_features": N_T1_FEATURES,
                "n_t2_features": N_T2_FEATURES,
                "n_t3_features": N_T3_FEATURES,
                "n_heads": len(ALL_HEAD_NAMES),
                "batch_size": BATCH_SIZE,
                "lr": LR,
                "epochs_per_fold": EPOCHS_PER_FOLD,
                "warmup_steps": WARMUP_STEPS,
                "grad_clip": GRAD_CLIP,
                "n_folds_planned": len(folds),
                "wf_train_days": train_days,
                "first_oot_monday": FIRST_OOT_MONDAY,
                "rank_norm_idxs": str(RANK_NORM_FEATURE_IDXS),
                "loss_lambdas": json.dumps(LOSS_LAMBDA),
                "warmstart_ckpt": warmstart_ckpt,
                "node": socket.gethostname(),
                "gpu": gpu_name,
                "data_dir": str(data_dir),
                "fifo_label_dir": str(fifo_label_dir),
                "alpha_label_dir": str(alpha_label_dir),
                "pt_pred_dir": str(pt_pred_dir),
                "output_dir": str(output_dir),
                "mixed_precision": "bf16" if use_amp else "none",
                "t3_features_status": "ZERO_PLACEHOLDER_TODO_BACKFILL",
            })
            logger.info(f"MLflow run started: {mlflow_run.info.run_id}")
        except Exception as e:
            logger.warning(f"MLflow init failed: {e} — continuing without tracking")
            mlflow_run = None

    concat_results = {h: {"preds": [], "targets": [], "masks": []} for h in ALL_HEAD_NAMES}

    try:
        start_fold = int(os.environ.get("V32_START_FOLD", 0))
        for f_info in folds:
            fold_idx = f_info["fold"]
            if fold_idx < start_fold:
                logger.info(f"Skipping fold {fold_idx} (V32_START_FOLD={start_fold})")
                continue
            logger.info("=" * 60)
            logger.info(f"FOLD {fold_idx} | train {f_info['train_start']}→{f_info['train_end']} "
                        f"| OOT {f_info['oot_start']}→{f_info['oot_end']}")
            logger.info("=" * 60)

            train_ds = SmartV32Dataset(
                data_dir=data_dir, fifo_label_dir=fifo_label_dir,
                alpha_label_dir=alpha_label_dir, pt_pred_dir=pt_pred_dir,
                tier2_parquet_root=tier2_parquet_root,
                tier3_parquet_root=tier3_parquet_root,
                dates=f_info["train_dates"],
                window_t1=WINDOW_SIZE_T1, window_t2=WINDOW_SIZE_T2, window_t3=WINDOW_SIZE_T3,
                stride=STRIDE, feature_stats=None,
                cache_size=1, require_alpha_labels=False,  # HC #300: was 4, root cause of OOM loop
            )
            feature_stats = train_ds.get_feature_stats()
            np.savez(
                output_dir / f"fold_{fold_idx:02d}_feature_stats.npz",
                mean_t1=feature_stats["mean_t1"], std_t1=feature_stats["std_t1"],
            )
            oot_ds = SmartV32Dataset(
                data_dir=data_dir, fifo_label_dir=fifo_label_dir,
                alpha_label_dir=alpha_label_dir, pt_pred_dir=pt_pred_dir,
                tier2_parquet_root=tier2_parquet_root,
                tier3_parquet_root=tier3_parquet_root,
                dates=f_info["oot_dates"],
                window_t1=WINDOW_SIZE_T1, window_t2=WINDOW_SIZE_T2, window_t3=WINDOW_SIZE_T3,
                stride=STRIDE, feature_stats=feature_stats,
                cache_size=1, require_alpha_labels=False,  # HC #300: was 4, root cause of OOM loop
            )

            # Same loader strategy as v3 — shuffle=False on train to keep cache hot.
            # HC #296: persistent_workers=False + prefetch_factor=2 to fix the 3.5h-wall-clock
            # memory leak that OOM-killed v3.2 fold 0 twice (2026-05-12 03:42 ET + 07:15 ET).
            # Persistent workers accumulated cached T1+T2+T3 tier tensors across epochs.
            train_loader = DataLoader(
                train_ds, batch_size=BATCH_SIZE, shuffle=False,
                num_workers=2, pin_memory=True, drop_last=True,
                persistent_workers=False, prefetch_factor=2,
                collate_fn=collate_v32,
            )
            oot_loader = DataLoader(
                oot_ds, batch_size=BATCH_SIZE * 2, shuffle=False,
                num_workers=1, pin_memory=True,
                persistent_workers=False, prefetch_factor=2,
                collate_fn=collate_v32,
            )

            # Fresh model + warmstart from v3
            model = CNNMambaV321().to(device)
            if fold_idx == 0:
                logger.info(f"Model parameters: {count_parameters(model):,}")
            warm_stats = model.load_v3_warmstart(warmstart_ckpt, device)
            if MLFLOW_AVAILABLE and mlflow_run is not None and fold_idx == 0:
                try:
                    mlflow.log_params({
                        "warmstart_loaded_tensors": warm_stats["loaded"],
                        "warmstart_skipped_keys": warm_stats["skipped"],
                        "warmstart_init_random_keys": warm_stats["init_random"],
                    })
                except Exception as e:
                    logger.warning(f"MLflow warmstart log failed: {e}")

            total_steps = EPOCHS_PER_FOLD * len(train_loader)

            # HC #296: resume-from-intra-ckpt for the current fold (if requested).
            # If the ckpt is v2 (full-state), load model_state into model AND pass the
            # dict to train_one_fold_v32 so it restores optimizer + scheduler + RNG
            # and skips to the saved batch. If v1 (model-only), use it as a warmstart.
            fold_resume_state = None
            if resume_from_intra_ckpt is not None:
                rp = Path(resume_from_intra_ckpt)
                if rp.exists():
                    try:
                        rckpt = torch.load(rp, map_location=device, weights_only=False)
                        rfold = int(rckpt.get("fold", fold_idx))
                        if rfold == fold_idx:
                            model.load_state_dict(rckpt["model_state"], strict=False)
                            ver = int(rckpt.get("ckpt_version", 1))
                            if ver >= 2:
                                fold_resume_state = rckpt
                                logger.info(
                                    f"  Fold {fold_idx} loaded RESUME ckpt v{ver} from {rp.name} "
                                    f"(ep={rckpt.get('epoch')} batch={rckpt.get('batch')})"
                                )
                            else:
                                logger.info(
                                    f"  Fold {fold_idx} loaded LEGACY model-only ckpt v1 from {rp.name} "
                                    f"as warmstart (Ep 0 will restart from batch 0)"
                                )
                        else:
                            logger.info(
                                f"  Fold {fold_idx} skipping resume ckpt (saved for fold {rfold})"
                            )
                    except Exception as e:
                        logger.warning(f"Resume ckpt load failed ({e}); proceeding without resume")
                else:
                    logger.warning(f"Resume ckpt path does not exist: {rp}")

            train_one_fold_v32(
                model, train_loader, oot_loader,
                fold_idx, output_dir, mlflow_run, device,
                total_train_steps=total_steps, use_amp=use_amp,
                resume_state=fold_resume_state,
            )

            # Reload best for OOT inference
            ckpt_path = output_dir / f"fold_{fold_idx:02d}_best.pt"
            if ckpt_path.exists():
                ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
                model.load_state_dict(ckpt["model_state"])
                logger.info(f"Loaded best (val_loss={ckpt['val_loss']:.4f})")

            loss_fn = JointMultiHeadLossV32(LOSS_LAMBDA)
            metrics, preds, targets, masks = evaluate_v32(
                model, oot_loader, loss_fn, device, use_amp=use_amp
            )
            logger.info(
                f"Fold {fold_idx} OOT FINAL | "
                f"IC 1s={metrics.get('ic_log_ret_1s', float('nan')):.4f} | "
                f"IC 5s={metrics.get('ic_log_ret_5s', float('nan')):.4f} | "
                f"IC 10s={metrics.get('ic_log_ret_10s', float('nan')):.4f} | "
                f"IC 30s={metrics.get('ic_log_ret_30s', float('nan')):.4f}"
            )

            # Save OOT artifacts
            save_dict = {
                "fold_idx": np.array(fold_idx),
                "oot_dates": np.array(f_info["oot_dates"]),
            }
            for h in ALL_HEAD_NAMES:
                if h in preds:
                    save_dict[f"pred_{h}"] = preds[h]
                    save_dict[f"target_{h}"] = targets[h]
                    save_dict[f"mask_{h}"] = masks[h]
            np.savez_compressed(output_dir / f"fold_{fold_idx:02d}_oot_predictions.npz", **save_dict)

            # Aggregate for concat
            for h in ALL_HEAD_NAMES:
                if h in preds:
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
            torch.cuda.empty_cache()

        # Concat
        logger.info("=" * 60)
        logger.info("CONCAT RESULTS (all folds combined)")
        logger.info("=" * 60)
        concat_summary = {}
        for h in ALL_HEAD_NAMES:
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
            elif h.startswith("pred_") or h.endswith("_ticks"):
                corr = compute_ic(p[valid], t[valid])
                concat_summary[f"concat_corr_{h}"] = corr
                logger.info(f"  Concat corr {h}: {corr:.4f}  (n={int(valid.sum())})")

        with open(output_dir / "concat_summary.json", "w") as fh:
            json.dump({k: (float(v) if not np.isnan(v) else None) for k, v in concat_summary.items()},
                      fh, indent=2)

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
    p = argparse.ArgumentParser(description="CNN-Mamba v3.2 long-context multi-head trainer")
    p.add_argument("--data-dir", type=str, default=DEFAULT_DATA_DIR)
    p.add_argument("--fifo-label-dir", type=str, default=DEFAULT_FIFO_LABEL_DIR)
    p.add_argument("--alpha-label-dir", type=str, default=DEFAULT_ALPHA_LABEL_DIR)
    p.add_argument("--pt-pred-dir", type=str, default=DEFAULT_PT_PRED_DIR)
    p.add_argument("--tier2-parquet-root", type=str, default=DEFAULT_TIER2_PARQUET_ROOT)
    p.add_argument("--tier3-parquet-root", type=str, default=DEFAULT_TIER3_PARQUET_ROOT)
    p.add_argument("--output-dir", type=str, default=DEFAULT_OUTPUT_DIR)
    p.add_argument("--n-folds", type=int, default=N_FOLDS)
    p.add_argument("--train-days", type=int, default=WF_TRAIN_DAYS)
    p.add_argument("--warmstart-ckpt", type=str, default=V3_WARMSTART_CKPT)
    p.add_argument("--device", type=str, default=None)
    # HC #296: resume from intra_ckpt (model + optimizer + scheduler + RNG state)
    p.add_argument(
        "--resume-from-intra-ckpt", type=str, default=None,
        help="Path to fold_NN_intra_ckpt.pt to resume from (HC #296). "
             "v2 ckpts restore full state + skip to saved batch; v1 ckpts act as warmstart."
    )
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("=" * 60)
    logger.info("CNN-Mamba v3.2 Long-Context Multi-Head Training")
    logger.info("=" * 60)
    logger.info(f"data_dir         = {args.data_dir}")
    logger.info(f"fifo_label_dir   = {args.fifo_label_dir}")
    logger.info(f"alpha_label_dir  = {args.alpha_label_dir}")
    logger.info(f"pt_pred_dir      = {args.pt_pred_dir}")
    logger.info(f"tier2_parquet    = {args.tier2_parquet_root}")
    logger.info(f"tier3_parquet    = {args.tier3_parquet_root}")
    logger.info(f"output_dir       = {args.output_dir}")
    logger.info(f"n_folds          = {args.n_folds}")
    logger.info(f"train_days       = {args.train_days}")
    logger.info(f"warmstart_ckpt   = {args.warmstart_ckpt}")
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
        logger.info(f"VRAM             = {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    resume_path = (
        Path(args.resume_from_intra_ckpt)
        if getattr(args, "resume_from_intra_ckpt", None) else None
    )
    if resume_path is not None:
        logger.info(f"resume_intra_ckpt = {resume_path}")

    summary = run_weekly_wf_v32(
        data_dir=Path(args.data_dir),
        fifo_label_dir=Path(args.fifo_label_dir),
        alpha_label_dir=Path(args.alpha_label_dir),
        pt_pred_dir=Path(args.pt_pred_dir),
        output_dir=Path(args.output_dir),
        device=device,
        n_folds=args.n_folds,
        train_days=args.train_days,
        warmstart_ckpt=args.warmstart_ckpt,
        tier2_parquet_root=Path(args.tier2_parquet_root),
        tier3_parquet_root=Path(args.tier3_parquet_root),
        resume_from_intra_ckpt=resume_path,
    )

    logger.info("=" * 60)
    logger.info("FINAL SUMMARY")
    logger.info("=" * 60)
    for k, v in (summary or {}).items():
        logger.info(f"  {k}: {v}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
