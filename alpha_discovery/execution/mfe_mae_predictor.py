#!/usr/bin/env python3
"""
mfe_mae_predictor.py — Adaptive TP/SL via MFE/MAE Prediction
=============================================================
Trains a lightweight MLP to predict per-trade MFE and MAE at 1s, 5s, 10s horizons.
Purpose: replace static TP/SL with adaptive levels set by predicted excursion.

Model:
  Input  : ~45 features (CNN-Mamba preds, PatchTST preds, microstructure, vol, ToD)
  Output : 6 values — MFE and MAE at 1s / 5s / 10s horizons (in ticks)
  Loss   : Asymmetric MSE — under-predicting MFE (leaves $$ on table) penalized 2x,
           over-predicting MAE (widens SL) penalized 1.5x
  Arch   : 4-layer MLP, 256 hidden, <500K params — fits in 8GB VRAM

Training:
  Walk-forward sliding window (HC #0) — 60d train, 1d OOT
  Labels: pre-computed per-fold MFE/MAE from mfe_mae_analysis/ directory
  MLflow: experiment "mfe_mae_predictor_razer" (mandatory)

Usage:
    # Jupiter (CPU, for testing — auto-detects):
    python -u mfe_mae_predictor.py

    # Razer (RTX 3070 GPU, full training):
    python -u mfe_mae_predictor.py --gpu --epochs 80 --batch-size 2048

    # Quick smoke test (2 folds only):
    python -u mfe_mae_predictor.py --max-folds 2 --epochs 5

Cost model:
  ES tick = $12.50 | commission RT = $4.70 = 0.376 ticks
  P&L = exit_tick - entry_tick - 0.376 ticks (NO separate spread cost)

Output files:
  {OUTPUT_DIR}/fold_{N:02d}_model.pt          — weights + norm params
  {OUTPUT_DIR}/fold_{N:02d}_predictions.npz   — predicted vs actual MFE/MAE
  {OUTPUT_DIR}/walk_forward_summary.json      — concat metrics across folds
"""

import argparse
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
from torch.utils.data import DataLoader, Dataset
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Path setup — works on both Jupiter (Linux) and Razer (Windows)
# ---------------------------------------------------------------------------
if sys.platform == "win32":
    LVL3_ROOT = Path("C:/Users/claude/Lvl3Quant")
else:
    LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")

# Adjust data paths per machine
# CNN-Mamba predictions (fold_XX_oot_predictions.npz)
CNN_PRED_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"

# Pre-computed MFE/MAE labels (fold_XX_mfe_mae_{horizon}.npz)
MFE_MAE_DIR = CNN_PRED_DIR / "mfe_mae_analysis"

# PatchTST predictions directory
PTST_PRED_DIR = LVL3_ROOT / "output" / "patchtst_smart_v3_mar"

# Vol LGBM predictions
VOL_PRED_DIR = LVL3_ROOT / "output" / "vol_lgbm_v3"

# MBO microstructure data
MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"

# Output
OUTPUT_DIR = LVL3_ROOT / "output" / "mfe_mae_predictor"

# ---------------------------------------------------------------------------
# Constants — canonical ES values
# ---------------------------------------------------------------------------
ES_TICK_SIZE = 0.25         # points per tick
ES_TICK_VALUE = 12.50       # dollars per tick
ES_RT_COMMISSION = 4.70     # dollars round-trip
ES_COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE  # 0.376 ticks

# Walk-forward config (HC #0: SLIDING window only)
N_FOLDS = 10                # total folds available
TRAIN_WINDOW = 8            # folds to train on (sliding 60d)
VAL_SIZE = 1                # folds for val (last fold before OOT)
# OOT = 1 fold (the last held-out fold in each window)

# MFE/MAE prediction horizons (seconds)
MFE_MAE_HORIZONS = [1.0, 5.0, 10.0]
HORIZON_LABELS = ["1s", "5s", "10s"]
# Output targets (6 values per prediction):
#   [MFE_ticks, MAE_ticks, pnl_1s, pnl_5s, pnl_10s, mfe_mae_ratio]
#   MFE_ticks:    max favorable excursion over 30s window (sets TP)
#   MAE_ticks:    max adverse excursion over 30s window (sets SL)
#   pnl_1s/5s/10s: directional P&L at snapshot (helps choose hold time)
#   mfe_mae_ratio: MFE/MAE ratio (signal quality indicator)
N_TARGETS = 6
TARGET_NAMES = ["MFE_ticks", "MAE_ticks", "pnl_1s", "pnl_5s", "pnl_10s", "mfe_mae_ratio"]

# Clamp labels (outlier protection)
MFE_MAX_TICKS = 50.0
MAE_MAX_TICKS = 20.0

# Training defaults
DEFAULT_HIDDEN_DIM = 256
DEFAULT_N_LAYERS = 4
DEFAULT_DROPOUT = 0.3
DEFAULT_LR = 3e-4
DEFAULT_BATCH_SIZE = 1024
DEFAULT_EPOCHS = 60
DEFAULT_PATIENCE = 15

# MLflow
MLFLOW_URI = os.environ.get("MLFLOW_TRACKING_URI", "http://jupiter:5000" if sys.platform == "win32" else "http://localhost:5000")
EXPERIMENT_NAME = "mfe_mae_predictor_razer"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
NUM_WORKERS = 4 if sys.platform != "win32" else 0
PIN_MEMORY = torch.cuda.is_available()

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [MFE_MAE] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("mfe_mae_predictor")

# ---------------------------------------------------------------------------
# MLflow (optional — always log if available)
# ---------------------------------------------------------------------------
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False
    log.warning("mlflow not installed — training without experiment tracking.")


# ===========================================================================
# Feature Engineering
# ===========================================================================

def rolling_zscore_fast(arr: np.ndarray, window: int = 500) -> np.ndarray:
    """Fast rolling z-score via cumulative sums."""
    n = len(arr)
    z = np.zeros(n, dtype=np.float32)
    cs = np.cumsum(arr.astype(np.float64))
    cs2 = np.cumsum(arr.astype(np.float64) ** 2)
    for i in range(n):
        s = max(0, i - window + 1)
        cnt = i - s + 1
        if cnt < 20:
            continue
        sm = cs[i] - (cs[s - 1] if s > 0 else 0.0)
        sm2 = cs2[i] - (cs2[s - 1] if s > 0 else 0.0)
        mean = sm / cnt
        var = max(sm2 / cnt - mean ** 2, 1e-12)
        z[i] = float((arr[i] - mean) / (var ** 0.5))
    return z


def rolling_mean_fast(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    cs = np.cumsum(arr.astype(np.float64))
    for i in range(n):
        s = max(0, i - window + 1)
        cnt = i - s + 1
        out[i] = float((cs[i] - (cs[s - 1] if s > 0 else 0.0)) / cnt)
    return out


def rolling_std_fast(arr: np.ndarray, window: int) -> np.ndarray:
    n = len(arr)
    out = np.zeros(n, dtype=np.float32)
    cs = np.cumsum(arr.astype(np.float64))
    cs2 = np.cumsum(arr.astype(np.float64) ** 2)
    for i in range(n):
        s = max(0, i - window + 1)
        cnt = i - s + 1
        if cnt < 5:
            continue
        sm = cs[i] - (cs[s - 1] if s > 0 else 0.0)
        sm2 = cs2[i] - (cs2[s - 1] if s > 0 else 0.0)
        mean = sm / cnt
        var = max(sm2 / cnt - mean ** 2, 0.0)
        out[i] = float(var ** 0.5)
    return out


def build_signal_features(
    cnn_preds: np.ndarray,   # (N, 3): 1s/5s/10s CNN-Mamba predictions
    embeddings: np.ndarray,  # (N, 96): CNN-Mamba internal embeddings
    ptst_preds: Optional[np.ndarray] = None,  # (N, 3): PatchTST predictions
) -> Tuple[np.ndarray, List[str]]:
    """
    Build signal-derived features.

    Features (total ~30 without embeddings, ~126 with embeddings):
      - CNN-Mamba raw preds (3)
      - CNN-Mamba abs preds = conviction (3)
      - CNN-Mamba sign agreement (1)
      - CNN-Mamba horizon disagreement / pred_std (1)
      - CNN-Mamba rolling z-scores (3)
      - CNN-Mamba short/long conviction trajectory (3)
      - CNN-Mamba pred range and skew (2)
      - CNN embeddings (96) — rich learned representation
      - PatchTST preds if available (3)
      - PatchTST-CNN confluence (agreement/disagreement) (3)
    """
    N = len(cnn_preds)
    parts, names = [], []

    # --- 1. Raw CNN-Mamba predictions (3) ---
    parts.append(cnn_preds)
    names.extend(["cnn_pred_1s", "cnn_pred_5s", "cnn_pred_10s"])

    # --- 2. Absolute predictions = conviction (3) ---
    abs_preds = np.abs(cnn_preds).astype(np.float32)
    parts.append(abs_preds)
    names.extend(["cnn_abs_1s", "cnn_abs_5s", "cnn_abs_10s"])

    # --- 3. Sign agreement across horizons (1) ---
    signs = np.sign(cnn_preds)
    sign_agreement = np.mean(signs == signs[:, 2:3], axis=1, keepdims=True).astype(np.float32)
    parts.append(sign_agreement)
    names.append("cnn_sign_agreement")

    # --- 4. Horizon disagreement: std of preds across horizons (1) ---
    pred_std = np.std(cnn_preds, axis=1, keepdims=True).astype(np.float32)
    parts.append(pred_std)
    names.append("cnn_horizon_disagree")

    # --- 5. Prediction range and skew (2) ---
    pred_range = (np.max(cnn_preds, axis=1) - np.min(cnn_preds, axis=1)).reshape(-1, 1).astype(np.float32)
    pred_skew = (cnn_preds[:, 2] - cnn_preds[:, 0]).reshape(-1, 1).astype(np.float32)
    parts.append(pred_range)
    parts.append(pred_skew)
    names.extend(["cnn_pred_range", "cnn_pred_skew_10_1"])

    # --- 6. Rolling z-scores per horizon (3) ---
    for hi, label in enumerate(["1s", "5s", "10s"]):
        z = rolling_zscore_fast(cnn_preds[:, hi], window=500)
        parts.append(z.reshape(-1, 1))
        names.append(f"cnn_zscore_{label}")

    # --- 7. Conviction trajectory: short vs long mean of |pred| (3) ---
    for hi, label in enumerate(["1s", "5s", "10s"]):
        short_m = rolling_mean_fast(np.abs(cnn_preds[:, hi]), window=50)
        long_m = rolling_mean_fast(np.abs(cnn_preds[:, hi]), window=500)
        traj = (short_m - long_m).reshape(-1, 1)
        parts.append(traj.astype(np.float32))
        names.append(f"cnn_conviction_traj_{label}")

    # --- 8. Embeddings (96) — deep learned features ---
    parts.append(embeddings.astype(np.float32))
    names.extend([f"emb_{i}" for i in range(embeddings.shape[1])])

    # --- 9. PatchTST predictions (3) + confluence features (3) ---
    if ptst_preds is not None:
        ptst = ptst_preds.astype(np.float32)
        parts.append(ptst)
        names.extend(["ptst_pred_1s", "ptst_pred_5s", "ptst_pred_10s"])

        # Confluence: sign agreement between CNN and PatchTST per horizon
        for hi, label in enumerate(["1s", "5s", "10s"]):
            agree = (np.sign(cnn_preds[:, hi]) == np.sign(ptst[:, hi])).astype(np.float32)
            parts.append(agree.reshape(-1, 1))
            names.append(f"confluence_{label}")
    else:
        # Placeholder zeros so input_dim is consistent
        parts.append(np.zeros((N, 6), dtype=np.float32))
        names.extend(["ptst_pred_1s", "ptst_pred_5s", "ptst_pred_10s",
                       "confluence_1s", "confluence_5s", "confluence_10s"])

    return np.concatenate(parts, axis=1).astype(np.float32), names


def build_microstructure_features(
    mbo_data: Optional[dict],
    n_samples: int,
    sample_indices: Optional[np.ndarray] = None,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build microstructure features from MBO event data.

    Key features for MFE/MAE prediction:
      - Book imbalance (short windows) — predicts order flow direction
      - Trade flow imbalance — actual execution flow
      - Realized vol at multiple windows — predicts excursion size
      - Event rate — predicts path volatility
      - Spread proxy — execution cost context

    Returns (n_samples, n_features), names
    """
    N_MICRO = 14  # fixed output dim
    zero_feats = np.zeros((n_samples, N_MICRO), dtype=np.float32)
    zero_names = [f"micro_{i}" for i in range(N_MICRO)]

    if mbo_data is None or len(mbo_data.get("timestamps", [])) < 200:
        return zero_feats, zero_names

    timestamps = mbo_data["timestamps"]
    events = mbo_data.get("events", None)

    if events is None or len(events) < 200:
        return zero_feats, zero_names

    n_events = len(timestamps)

    # Events are (N, 25) in smart_v3 format — see data loading for column layout
    # Columns (from inspection): [time_delta_log, event_type_id?, side_id?, price_rel_ticks, qty_log, ...]
    # We'll use columns that exist and are clearly interpretable
    try:
        prices = events[:, 3].astype(np.float32)       # price relative in ticks
        qty_log = events[:, 4].astype(np.float32)       # log(quantity)
        sizes = np.exp(np.clip(qty_log, -2, 10))        # actual quantity
        sides = events[:, 2].astype(np.float32)         # 0=bid, 1=ask, other=neutral
    except (IndexError, ValueError):
        return zero_feats, zero_names

    if sample_indices is None:
        sample_indices = np.linspace(200, n_events - 1, n_samples).astype(int)
    sample_indices = np.clip(sample_indices, 200, n_events - 1)

    # Derived arrays
    side_sign = np.where(sides == 1, 1.0, np.where(sides == 0, -1.0, 0.0))
    signed_flow = sizes * side_sign
    price_changes = np.diff(prices, prepend=prices[0])

    cs_sizes = np.cumsum(sizes.astype(np.float64))
    cs_flow = np.cumsum(signed_flow.astype(np.float64))
    cs_pc2 = np.cumsum(price_changes.astype(np.float64) ** 2)

    time_deltas = np.diff(timestamps.astype(np.float64), prepend=float(timestamps[0]))
    time_deltas_sec = np.clip(time_deltas / 1e9, 1e-6, 3600.0)
    cs_time = np.cumsum(time_deltas_sec)

    def _window_sum(cs, idx, w):
        s = max(0, int(idx) - w + 1)
        return float(cs[int(idx)] - (cs[s - 1] if s > 0 else 0.0)), int(idx) - s + 1

    # Feature arrays
    book_imb_50 = np.zeros(n_samples, dtype=np.float32)
    book_imb_200 = np.zeros(n_samples, dtype=np.float32)
    flow_imb_100 = np.zeros(n_samples, dtype=np.float32)
    flow_imb_500 = np.zeros(n_samples, dtype=np.float32)
    rvol_50 = np.zeros(n_samples, dtype=np.float32)
    rvol_200 = np.zeros(n_samples, dtype=np.float32)
    rvol_1000 = np.zeros(n_samples, dtype=np.float32)
    event_rate_100 = np.zeros(n_samples, dtype=np.float32)
    event_rate_500 = np.zeros(n_samples, dtype=np.float32)
    spread_proxy = np.zeros(n_samples, dtype=np.float32)
    spread_zscore = np.zeros(n_samples, dtype=np.float32)
    mom_50 = np.zeros(n_samples, dtype=np.float32)
    mom_200 = np.zeros(n_samples, dtype=np.float32)
    vol_of_vol = np.zeros(n_samples, dtype=np.float32)

    TICK = 0.25
    sp_history = []

    for i, eidx in enumerate(sample_indices):
        eidx = int(eidx)

        # Spread proxy: price range in last 20 events
        if eidx >= 20:
            wp = prices[eidx - 20:eidx + 1]
            sp = float((np.max(wp) - np.min(wp)) / TICK)
            spread_proxy[i] = sp
            sp_history.append(sp)
            if len(sp_history) > 50:
                sp_arr = np.array(sp_history[-50:])
                sp_m, sp_s = sp_arr.mean(), sp_arr.std()
                if sp_s > 1e-8:
                    spread_zscore[i] = (sp - sp_m) / sp_s

        # Book imbalance
        for w, arr in [(50, book_imb_50), (200, book_imb_200)]:
            if eidx >= w:
                rs = sides[eidx - w:eidx + 1]
                rz = sizes[eidx - w:eidx + 1]
                bv = rz[rs < 0.5].sum()
                av = rz[rs > 0.5].sum()
                tot = bv + av
                if tot > 0:
                    arr[i] = float((bv - av) / tot)

        # Flow imbalance
        for w, arr in [(100, flow_imb_100), (500, flow_imb_500)]:
            if eidx >= w:
                fs, _ = _window_sum(cs_flow, eidx, w)
                vs, _ = _window_sum(cs_sizes, eidx, w)
                if vs > 0:
                    arr[i] = float(fs / vs)

        # Realized vol
        for w, arr in [(50, rvol_50), (200, rvol_200), (1000, rvol_1000)]:
            if eidx >= w:
                pc2s, cnt = _window_sum(cs_pc2, eidx, w)
                arr[i] = float((pc2s / cnt) ** 0.5 / TICK)

        # Event rate
        for w, arr in [(100, event_rate_100), (500, event_rate_500)]:
            if eidx >= w:
                ts, _ = _window_sum(cs_time, eidx, w)
                if ts > 0:
                    arr[i] = float(w / ts)

        # Momentum
        for w, arr in [(50, mom_50), (200, mom_200)]:
            if eidx >= w:
                arr[i] = float((prices[eidx] - prices[eidx - w]) / TICK)

        # Vol of vol
        if eidx >= 500:
            rv_samples = []
            for k in range(10):
                kidx = eidx - k * 50
                if kidx >= 50:
                    pc2s, cnt = _window_sum(cs_pc2, kidx, 50)
                    rv_samples.append((pc2s / max(cnt, 1)) ** 0.5 / TICK)
            if len(rv_samples) >= 3:
                vol_of_vol[i] = float(np.std(rv_samples))

    feat_matrix = np.column_stack([
        book_imb_50, book_imb_200,
        flow_imb_100, flow_imb_500,
        rvol_50, rvol_200, rvol_1000,
        event_rate_100, event_rate_500,
        spread_proxy, spread_zscore,
        mom_50, mom_200,
        vol_of_vol,
    ]).astype(np.float32)

    feat_names = [
        "book_imb_50", "book_imb_200",
        "flow_imb_100", "flow_imb_500",
        "rvol_50", "rvol_200", "rvol_1000",
        "event_rate_100", "event_rate_500",
        "spread_proxy", "spread_zscore",
        "mom_50", "mom_200",
        "vol_of_vol",
    ]
    return feat_matrix, feat_names


def build_temporal_features(
    timestamps_ns: Optional[np.ndarray],
    n_samples: int,
) -> Tuple[np.ndarray, List[str]]:
    """
    Cyclical time-of-day and session indicator features.
    ES RTH: 9:30–16:00 ET. Pre-market and after-hours matter for execution.
    """
    if timestamps_ns is not None and len(timestamps_ns) == n_samples:
        utc_sec = timestamps_ns.astype(np.float64) / 1e9
        et_sec = utc_sec + (-5) * 3600   # ET = UTC-5 (simplified; doesn't handle DST)
        hours = (et_sec % 86400) / 3600.0
    else:
        hours = np.linspace(9.5, 16.0, n_samples)

    hours = hours.astype(np.float32)

    # Cyclical 24h and RTH (4)
    tod_sin = np.sin(2 * np.pi * hours / 24.0)
    tod_cos = np.cos(2 * np.pi * hours / 24.0)
    rth_frac = np.clip((hours - 9.5) / 6.5, 0.0, 1.0)
    rth_sin = np.sin(2 * np.pi * rth_frac)
    rth_cos = np.cos(2 * np.pi * rth_frac)

    # Session indicators (4) — execution profile differs per session
    is_open = ((hours >= 9.5) & (hours < 10.5)).astype(np.float32)
    is_core = ((hours >= 10.5) & (hours < 15.0)).astype(np.float32)
    is_close = ((hours >= 15.0) & (hours < 16.0)).astype(np.float32)
    is_pre = (hours < 9.5).astype(np.float32)

    # Minutes-based features (2)
    mins_open = np.clip((hours - 9.5) * 60, -60, 420)
    mins_close = np.clip((16.0 - hours) * 60, -60, 420)

    feat_matrix = np.column_stack([
        tod_sin, tod_cos,
        rth_sin, rth_cos,
        is_open, is_core, is_close, is_pre,
        mins_open, mins_close,
    ]).astype(np.float32)

    feat_names = [
        "tod_sin", "tod_cos",
        "rth_sin", "rth_cos",
        "session_open", "session_core", "session_close", "session_pre",
        "mins_since_open", "mins_until_close",
    ]
    return feat_matrix, feat_names


def build_vol_features(
    vol_pred: Optional[np.ndarray],
    cnn_preds: np.ndarray,
) -> Tuple[np.ndarray, List[str]]:
    """Volatility context features."""
    N = len(cnn_preds)
    parts, names = [], []

    if vol_pred is not None:
        vp = vol_pred if vol_pred.ndim == 2 else vol_pred.reshape(-1, 1)
        parts.append(vp.astype(np.float32))
        names.extend([f"vol_pred_{j}" for j in range(vp.shape[1])])
    else:
        parts.append(np.zeros((N, 1), dtype=np.float32))
        names.append("vol_pred_placeholder")

    # Realized vol from prediction variance
    pred_vol_short = rolling_std_fast(cnn_preds[:, 2], window=100)
    pred_vol_long = rolling_std_fast(cnn_preds[:, 2], window=1000)
    vol_regime = pred_vol_short / np.maximum(pred_vol_long, 1e-8)

    parts.extend([
        pred_vol_short.reshape(-1, 1),
        pred_vol_long.reshape(-1, 1),
        vol_regime.reshape(-1, 1),
    ])
    names.extend(["pred_vol_100", "pred_vol_1000", "vol_regime"])

    return np.concatenate(parts, axis=1).astype(np.float32), names


# ===========================================================================
# Data Loading
# ===========================================================================

def load_fold_data(fold_idx: int, mfe_horizon: str = "1s") -> Optional[dict]:
    """
    Load one OOT fold's features and MFE/MAE labels.

    Data sources (all on Jupiter or Razer's local paths):
      1. CNN-Mamba fold predictions + embeddings (fold_XX_oot_predictions.npz)
      2. Pre-computed MFE/MAE labels (fold_XX_mfe_mae_{horizon}.npz)
      3. PatchTST fold predictions (matched by date)
      4. Vol LGBM predictions (matched by date)
      5. MBO microstructure data (matched by date)

    Returns dict with all arrays aligned to CNN-Mamba prediction count.
    """
    # --- CNN-Mamba predictions ---
    cnn_file = CNN_PRED_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if not cnn_file.exists():
        # Try Neptune-style paths (when running on Razer with synced data)
        log.warning(f"Fold {fold_idx:02d}: CNN file not found at {cnn_file}")
        return None

    cnn_data = np.load(str(cnn_file), allow_pickle=True)
    predictions = cnn_data["predictions"].astype(np.float32)   # (N, 3)
    embeddings = cnn_data["embeddings"].astype(np.float32)     # (N, 96)
    labels = cnn_data["labels"].astype(np.float32)             # (N, 3) actual price changes
    N = len(predictions)

    # Extract date from oot_files
    oot_files = cnn_data["oot_files"]
    if hasattr(oot_files, "tolist"):
        oot_files = oot_files.tolist()
    raw_path = str(oot_files[0]).replace("\\", "/")
    date_str = raw_path.split("/")[-1][:8]  # YYYYMMDD

    # --- Pre-computed MFE/MAE labels ---
    # We use the 1s-horizon analysis file which contains the raw path arrays
    mfe_mae_file_1s = MFE_MAE_DIR / f"fold_{fold_idx:02d}_mfe_mae_1s.npz"
    mfe_mae_file_10s = MFE_MAE_DIR / f"fold_{fold_idx:02d}_mfe_mae_10s.npz"

    if not mfe_mae_file_1s.exists():
        log.warning(f"Fold {fold_idx:02d}: MFE/MAE labels not found at {mfe_mae_file_1s}")
        return None

    mfe_mae_1s = np.load(str(mfe_mae_file_1s), allow_pickle=True)
    mfe_mae_10s = np.load(str(mfe_mae_file_10s), allow_pickle=True) if mfe_mae_file_10s.exists() else None

    # MFE/MAE from 1s file — columns: mfe_ticks, mae_ticks, path_1s, path_5s, path_10s
    n_mfe = len(mfe_mae_1s["mfe_ticks"])

    # Align lengths (MFE analysis may have slightly fewer samples due to NaN filtering)
    n_use = min(N, n_mfe)
    if abs(N - n_mfe) > 100:
        log.warning(f"Fold {fold_idx:02d}: N mismatch CNN={N}, MFE={n_mfe}. Using {n_use}.")

    # Clamp and build MFE/MAE targets at each horizon
    # horizon approximation: use path_Xs as "MFE at Xs window"
    # For MFE/MAE AT 1s: use path_1s (directional P&L at 1s)
    # For MFE/MAE AT 5s: use path_5s
    # For MFE/MAE AT 10s: use path_10s
    # MFE = max favorable excursion = max(0, direction * path_Xs)
    # MAE = max adverse excursion = max(0, -direction * min over [0..Xs])
    # Since we only have snapshots, we use the pre-computed mfe/mae which are computed
    # over the full window (max excursion over all labeled horizons up to 30s)

    # Use actual pre-computed MFE/MAE for the overall window
    mfe_overall = np.clip(mfe_mae_1s["mfe_ticks"][:n_use], 0.0, MFE_MAX_TICKS).astype(np.float32)
    mae_overall = np.clip(mfe_mae_1s["mae_ticks"][:n_use], 0.0, MAE_MAX_TICKS).astype(np.float32)
    direction = mfe_mae_1s["direction"][:n_use].astype(np.float32)

    # Build targets: (N, 6) = [MFE_ticks, MAE_ticks, pnl_1s, pnl_5s, pnl_10s, mfe_mae_ratio]
    #
    # MFE_ticks: pre-computed max favorable excursion over 30s window (SETS TP)
    # MAE_ticks: pre-computed max adverse excursion over 30s window (SETS SL)
    # pnl_Xs: directional P&L at snapshot horizon = direction * path_Xs (for hold-time choice)
    # mfe_mae_ratio: signal quality (clamped to [0, 10])
    path_1s = mfe_mae_1s["path_1s"][:n_use].astype(np.float32)
    path_5s = mfe_mae_1s["path_5s"][:n_use].astype(np.float32)
    path_10s = mfe_mae_1s["path_10s"][:n_use].astype(np.float32)

    pnl_1s = np.clip(direction * path_1s, -MFE_MAX_TICKS, MFE_MAX_TICKS).astype(np.float32)
    pnl_5s = np.clip(direction * path_5s, -MFE_MAX_TICKS, MFE_MAX_TICKS).astype(np.float32)
    pnl_10s = np.clip(direction * path_10s, -MFE_MAX_TICKS, MFE_MAX_TICKS).astype(np.float32)

    # MFE/MAE ratio (quality of trade opportunity)
    mfe_mae_ratio = np.clip(
        mfe_overall / np.maximum(mae_overall, 0.1),
        0.0, 10.0
    ).astype(np.float32)

    # Stack targets: (N, 6) = [MFE_ticks, MAE_ticks, pnl_1s, pnl_5s, pnl_10s, mfe_mae_ratio]
    targets = np.stack([mfe_overall, mae_overall, pnl_1s, pnl_5s, pnl_10s, mfe_mae_ratio], axis=1)

    # Timestamps for temporal features
    timestamps_ns = mfe_mae_1s["timestamps_ns"][:n_use] if "timestamps_ns" in mfe_mae_1s else None

    # --- PatchTST predictions ---
    ptst_pred = None
    for ptst_file in sorted(PTST_PRED_DIR.glob("fold_*_oot_predictions.npz")):
        try:
            pd_raw = np.load(str(ptst_file), allow_pickle=True)
            ptst_oot = pd_raw.get("oot_files", [""])
            if hasattr(ptst_oot, "tolist"):
                ptst_oot = ptst_oot.tolist()
            ptst_date = str(ptst_oot[0]).replace("\\", "/").split("/")[-1][:8]
            if ptst_date == date_str:
                ptst_raw = pd_raw["predictions"].astype(np.float32)
                if len(ptst_raw) == n_use:
                    ptst_pred = ptst_raw
                else:
                    # Resample to match
                    from scipy.interpolate import interp1d
                    xp = np.linspace(0, 1, len(ptst_raw))
                    xq = np.linspace(0, 1, n_use)
                    ptst_pred = np.zeros((n_use, ptst_raw.shape[1]), dtype=np.float32)
                    for j in range(ptst_raw.shape[1]):
                        fi = interp1d(xp, ptst_raw[:, j], kind="nearest", fill_value="extrapolate")
                        ptst_pred[:, j] = fi(xq)
                break
        except Exception:
            continue

    # --- Vol LGBM predictions ---
    vol_pred = None
    vol_file = VOL_PRED_DIR / f"vol_v3_{date_str}_predictions.npz"
    if vol_file.exists():
        try:
            vd = np.load(str(vol_file), allow_pickle=True)
            for k in ["predictions", "vol_pred", "y_pred"]:
                if k in vd:
                    vp = vd[k]
                    if len(vp) == n_use:
                        vol_pred = vp.astype(np.float32)
                    else:
                        from scipy.interpolate import interp1d
                        xp = np.linspace(0, 1, len(vp))
                        xq = np.linspace(0, 1, n_use)
                        if vp.ndim == 1:
                            fi = interp1d(xp, vp, kind="nearest", fill_value="extrapolate")
                            vol_pred = fi(xq).astype(np.float32)
                        else:
                            vol_pred = np.zeros((n_use, vp.shape[1]), dtype=np.float32)
                            for j in range(vp.shape[1]):
                                fi = interp1d(xp, vp[:, j], kind="nearest", fill_value="extrapolate")
                                vol_pred[:, j] = fi(xq)
                    break
        except Exception as e:
            log.debug(f"Fold {fold_idx:02d}: vol pred error: {e}")

    # --- MBO microstructure ---
    mbo_data = None
    mbo_file = MBO_DIR / f"{date_str}_mbo_events.npz"
    if mbo_file.exists():
        try:
            mbo_raw = np.load(str(mbo_file), allow_pickle=True)
            mbo_data = {k: mbo_raw[k] for k in mbo_raw.files}
        except Exception as e:
            log.debug(f"Fold {fold_idx:02d}: MBO error: {e}")

    return {
        "fold_idx": fold_idx,
        "date": date_str,
        "predictions": predictions[:n_use],      # (N, 3)
        "embeddings": embeddings[:n_use],         # (N, 96)
        "labels": labels[:n_use],                 # (N, 3) actual price changes
        "targets": targets,                       # (N, 6) MFE/MAE at 3 horizons
        "direction": direction,                   # (N,) -1/+1
        "mfe_overall": mfe_overall,               # (N,) overall window MFE
        "mae_overall": mae_overall,               # (N,) overall window MAE
        "ptst_pred": ptst_pred,                   # (N, 3) or None
        "vol_pred": vol_pred,                     # (N,) or None
        "mbo_data": mbo_data,                     # dict or None
        "timestamps_ns": timestamps_ns,           # (N,) or None
        "n_samples": n_use,
    }


def build_features_for_fold(data: dict) -> Tuple[np.ndarray, List[str]]:
    """
    Assemble all features for one fold.
    Returns (N, total_features) float32 array and feature names.
    """
    N = data["n_samples"]
    all_parts, all_names = [], []

    # 1. Signal features (CNN + PatchTST)
    sig_feats, sig_names = build_signal_features(
        data["predictions"], data["embeddings"], data["ptst_pred"]
    )
    all_parts.append(sig_feats)
    all_names.extend(sig_names)

    # 2. Microstructure features
    timestamps = data.get("timestamps_ns")
    mbo = data.get("mbo_data")
    if mbo is not None and timestamps is not None:
        # Build sample indices from timestamps: find event index for each prediction timestamp
        mbo_ts = mbo.get("timestamps", np.array([]))
        if len(mbo_ts) > 0 and len(timestamps) > 0:
            # Map each prediction timestamp to nearest event index
            sample_idxs = np.searchsorted(mbo_ts, timestamps, side="left")
            sample_idxs = np.clip(sample_idxs, 200, len(mbo_ts) - 1)
        else:
            sample_idxs = None
        micro_feats, micro_names = build_microstructure_features(mbo, N, sample_idxs)
    else:
        micro_feats, micro_names = build_microstructure_features(None, N)
    all_parts.append(micro_feats)
    all_names.extend(micro_names)

    # 3. Temporal features
    temp_feats, temp_names = build_temporal_features(timestamps, N)
    all_parts.append(temp_feats)
    all_names.extend(temp_names)

    # 4. Vol features
    vol_feats, vol_names = build_vol_features(data["vol_pred"], data["predictions"])
    all_parts.append(vol_feats)
    all_names.extend(vol_names)

    feat_matrix = np.concatenate(all_parts, axis=1).astype(np.float32)
    feat_matrix = np.nan_to_num(feat_matrix, nan=0.0, posinf=5.0, neginf=-5.0)

    return feat_matrix, all_names


# ===========================================================================
# Dataset
# ===========================================================================

class MfeMaeDataset(Dataset):
    """Simple dataset for MFE/MAE prediction training."""

    def __init__(self, features: np.ndarray, targets: np.ndarray):
        self.features = torch.from_numpy(features.astype(np.float32))
        self.targets = torch.from_numpy(targets.astype(np.float32))

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        return self.features[idx], self.targets[idx]


# ===========================================================================
# Model
# ===========================================================================

class MfeMaeMLPSmall(nn.Module):
    """
    Lightweight 4-layer MLP for MFE/MAE prediction.

    Architecture:
      Input → BatchNorm → [Linear(D) → LayerNorm → GELU → Dropout] × N → 6 outputs

    6 outputs: [MFE_1s, MFE_5s, MFE_10s, MAE_1s, MAE_5s, MAE_10s]
    All outputs passed through Softplus (>=0, since MFE/MAE are non-negative).

    Size at D=256, N=4, input=130:
      ~130*256 + 256*256*3 + 256*6 ≈ 234K parameters
    Easily fits in 8GB VRAM even with large batch sizes.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 4,
        dropout: float = 0.3,
    ):
        super().__init__()
        self.input_bn = nn.BatchNorm1d(input_dim)

        layers = []
        prev_dim = input_dim
        for i in range(n_layers):
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.LayerNorm(hidden_dim))
            layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)

        # Head 1: MFE_ticks (overall window, sets TP) — non-negative
        self.mfe_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        # Head 2: MAE_ticks (overall window, sets SL) — non-negative
        self.mae_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
            nn.Softplus(),
        )
        # Head 3: P&L at 1s/5s/10s snapshots — can be negative
        self.pnl_head = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 3),  # pnl_1s, pnl_5s, pnl_10s (unbounded)
        )
        # Head 4: MFE/MAE ratio — non-negative quality signal
        self.ratio_head = nn.Sequential(
            nn.Linear(hidden_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
            nn.Softplus(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, input_dim) float32
        Returns:
            (B, 6) float32 — [MFE_ticks, MAE_ticks, pnl_1s, pnl_5s, pnl_10s, mfe_mae_ratio]
        """
        x = self.input_bn(x)
        shared = self.trunk(x)
        mfe = self.mfe_head(shared)    # (B, 1)
        mae = self.mae_head(shared)    # (B, 1)
        pnl = self.pnl_head(shared)    # (B, 3)
        ratio = self.ratio_head(shared) # (B, 1)
        return torch.cat([mfe, mae, pnl, ratio], dim=1)  # (B, 6)

    def param_count(self) -> int:
        return sum(p.numel() for p in self.parameters())


# ===========================================================================
# Loss Function
# ===========================================================================

class AsymmetricMfeMaeLoss(nn.Module):
    """
    Asymmetric MSE loss for MFE/MAE prediction.

    Trading rationale:
      - Under-predicting MFE: TP set too low → exit too early → leave money on table
        Penalty: 2.0x
      - Over-predicting MFE: TP set too high → trade never closes → held too long
        Penalty: 1.0x (standard)
      - Under-predicting MAE: SL set too tight → stopped out on noise → commission waste
        Penalty: 2.0x
      - Over-predicting MAE: SL set too wide → accept larger losses
        Penalty: 1.5x

    Loss = mean over [MFE_1s, MFE_5s, MFE_10s, MAE_1s, MAE_5s, MAE_10s]:
           asymmetric_mse(pred, target)
    """

    def __init__(
        self,
        mfe_under_weight: float = 2.0,   # penalty for MFE under-prediction
        mfe_over_weight: float = 1.0,    # penalty for MFE over-prediction
        mae_under_weight: float = 2.0,   # penalty for MAE under-prediction (tight SL)
        mae_over_weight: float = 1.5,    # penalty for MAE over-prediction (wide SL)
    ):
        super().__init__()
        self.mfe_under = mfe_under_weight
        self.mfe_over = mfe_over_weight
        self.mae_under = mae_under_weight
        self.mae_over = mae_over_weight

    def _asym_mse(self, pred: torch.Tensor, target: torch.Tensor,
                  under_w: float, over_w: float) -> torch.Tensor:
        """
        Asymmetric MSE: pred < target → under-prediction, weighted by under_w.
        """
        err = pred - target
        sq = err ** 2
        # Under-prediction: err < 0 (pred lower than actual)
        under_mask = (err < 0).float()
        over_mask = 1.0 - under_mask
        weight = under_mask * under_w + over_mask * over_w
        return (weight * sq).mean()

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pred:   (B, 6) — [MFE_ticks, MAE_ticks, pnl_1s, pnl_5s, pnl_10s, mfe_mae_ratio]
            target: (B, 6) — same layout
        Returns:
            scalar loss
        """
        # Col 0: MFE_ticks (TP) — asymmetric: under-predicting is worse
        mfe_loss = self._asym_mse(pred[:, 0], target[:, 0], self.mfe_under, self.mfe_over)

        # Col 1: MAE_ticks (SL) — asymmetric: under-predicting is worse (stops out on noise)
        mae_loss = self._asym_mse(pred[:, 1], target[:, 1], self.mae_under, self.mae_over)

        # Cols 2-4: pnl at 1s/5s/10s — standard Huber (can be negative, no asymmetry)
        pnl_loss = sum(
            F.smooth_l1_loss(pred[:, 2 + i], target[:, 2 + i])
            for i in range(3)
        ) / 3.0

        # Col 5: MFE/MAE ratio — standard MSE (quality signal)
        ratio_loss = F.mse_loss(pred[:, 5], target[:, 5])

        # Weight: MFE and MAE are primary (0.4 each), pnl secondary (0.15), ratio (0.05)
        return 0.4 * mfe_loss + 0.4 * mae_loss + 0.15 * pnl_loss + 0.05 * ratio_loss


# ===========================================================================
# Walk-Forward Training
# ===========================================================================

def compute_fold_metrics(
    pred: np.ndarray,   # (N, 6)
    target: np.ndarray, # (N, 6)
) -> dict:
    """
    Compute per-output metrics: MSE, MAE (L1), Spearman correlation.
    Targets layout: [MFE_ticks, MAE_ticks, pnl_1s, pnl_5s, pnl_10s, mfe_mae_ratio]
    """
    output_names = TARGET_NAMES  # ["MFE_ticks", "MAE_ticks", "pnl_1s", "pnl_5s", "pnl_10s", "mfe_mae_ratio"]
    metrics = {}
    for i, name in enumerate(output_names):
        p = pred[:, i]
        t = target[:, i]
        mse = float(np.mean((p - t) ** 2))
        mae_l1 = float(np.mean(np.abs(p - t)))
        # Spearman correlation (rank-based, robust to outliers)
        if len(np.unique(t)) > 2:
            corr, _ = spearmanr(p, t)
            corr = float(corr) if not np.isnan(corr) else 0.0
        else:
            corr = 0.0
        key = name.lower().replace(" ", "_")
        metrics[f"{key}_mse"] = mse
        metrics[f"{key}_l1"] = mae_l1
        metrics[f"{key}_corr"] = corr

    # Summary metrics — MFE and MAE are primary
    metrics["mfe_corr"] = metrics["mfe_ticks_corr"]
    metrics["mae_corr"] = metrics["mae_ticks_corr"]
    metrics["avg_mfe_corr"] = metrics["mfe_ticks_corr"]   # compat alias
    metrics["avg_mae_corr"] = metrics["mae_ticks_corr"]   # compat alias
    pnl_corrs = [metrics[f"pnl_{h}_corr"] for h in ["1s", "5s", "10s"]]
    metrics["avg_pnl_corr"] = float(np.mean(pnl_corrs))
    metrics["avg_corr"] = float(np.mean([
        metrics["mfe_ticks_corr"],
        metrics["mae_ticks_corr"],
        metrics["avg_pnl_corr"],
    ]))

    return metrics


def train_one_fold(
    train_feats: np.ndarray,   # (N_train, D)
    train_tgts: np.ndarray,    # (N_train, 6)
    val_feats: np.ndarray,     # (N_val, D)
    val_tgts: np.ndarray,      # (N_val, 6)
    fold_idx: int,
    args,
) -> Tuple[MfeMaeMLPSmall, dict, np.ndarray]:
    """
    Train MFE/MAE predictor on one walk-forward fold.

    Returns: (model, results_dict, oot_predictions)
    """
    # Normalize from train set stats
    mean = train_feats.mean(axis=0)
    std = train_feats.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)

    train_norm = np.nan_to_num((train_feats - mean) / std, nan=0.0)
    val_norm = np.nan_to_num((val_feats - mean) / std, nan=0.0)

    # Datasets
    train_ds = MfeMaeDataset(train_norm, train_tgts)
    val_ds = MfeMaeDataset(val_norm, val_tgts)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY, drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        num_workers=NUM_WORKERS, pin_memory=PIN_MEMORY,
    )

    # Model
    input_dim = train_feats.shape[1]
    model = MfeMaeMLPSmall(
        input_dim=input_dim,
        hidden_dim=args.hidden_dim,
        n_layers=args.n_layers,
        dropout=args.dropout,
    ).to(DEVICE)
    log.info(f"  Model params: {model.param_count():,} | input_dim={input_dim} | device={DEVICE}")

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Loss
    criterion = AsymmetricMfeMaeLoss()

    # Early stopping
    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0
    no_improve = 0
    patience = DEFAULT_PATIENCE

    train_losses, val_losses = [], []

    for epoch in range(args.epochs):
        # --- Train ---
        model.train()
        epoch_loss = 0.0
        n_batches = 0
        for feats, tgts in train_loader:
            feats, tgts = feats.to(DEVICE), tgts.to(DEVICE)
            pred = model(feats)
            loss = criterion(pred, tgts)
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
            n_batches += 1
        scheduler.step()
        avg_train_loss = epoch_loss / max(n_batches, 1)
        train_losses.append(avg_train_loss)

        # --- Validate ---
        model.eval()
        val_preds_list, val_tgts_list = [], []
        with torch.no_grad():
            for feats, tgts in val_loader:
                pred = model(feats.to(DEVICE))
                val_preds_list.append(pred.cpu().numpy())
                val_tgts_list.append(tgts.numpy())

        val_pred_np = np.concatenate(val_preds_list)
        val_tgt_np = np.concatenate(val_tgts_list)
        val_loss = float(criterion(
            torch.from_numpy(val_pred_np), torch.from_numpy(val_tgt_np)
        ).item())
        val_losses.append(val_loss)

        # Log every 5 epochs
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            vm = compute_fold_metrics(val_pred_np, val_tgt_np)
            log.info(
                f"  Epoch {epoch:>3d}/{args.epochs} | "
                f"Train: {avg_train_loss:.4f} | Val: {val_loss:.4f} | "
                f"MFE corr: {vm['mfe_corr']:.3f} | MAE corr: {vm['mae_corr']:.3f} | "
                f"pnl corr: {vm['avg_pnl_corr']:.3f}"
            )

            if MLFLOW_AVAILABLE:
                try:
                    mlflow.log_metrics({
                        f"fold{fold_idx}_train_loss": avg_train_loss,
                        f"fold{fold_idx}_val_loss": val_loss,
                        f"fold{fold_idx}_mfe_corr": vm["mfe_corr"],
                        f"fold{fold_idx}_mae_corr": vm["mae_corr"],
                        f"fold{fold_idx}_pnl_corr": vm["avg_pnl_corr"],
                        f"fold{fold_idx}_lr": optimizer.param_groups[0]["lr"],
                    }, step=epoch)
                except Exception:
                    pass

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            best_epoch = epoch
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                log.info(f"  Early stopping at epoch {epoch} (best={best_epoch})")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Final eval on val set
    model.eval()
    with torch.no_grad():
        val_norm_t = torch.from_numpy(val_norm.astype(np.float32)).to(DEVICE)
        final_preds_list = []
        for start in range(0, len(val_norm_t), 10000):
            end = min(start + 10000, len(val_norm_t))
            out = model(val_norm_t[start:end])
            final_preds_list.append(out.cpu().numpy())
    final_preds = np.concatenate(final_preds_list)
    final_metrics = compute_fold_metrics(final_preds, val_tgts)

    log.info(f"  Best epoch: {best_epoch} | Val loss: {best_val_loss:.4f}")
    log.info(f"  MFE corr: {final_metrics['mfe_corr']:.4f} | "
             f"MAE corr: {final_metrics['mae_corr']:.4f} | "
             f"pnl corr: {final_metrics['avg_pnl_corr']:.4f}")

    return model, {
        "best_epoch": best_epoch,
        "best_val_loss": best_val_loss,
        "metrics": final_metrics,
        "norm_mean": mean.tolist(),
        "norm_std": std.tolist(),
        "input_dim": int(input_dim),
        "train_losses": train_losses,
        "val_losses": val_losses,
    }, final_preds


def print_fold_summary(fold_idx: int, metrics: dict, date_oot: str):
    """Pretty print fold results."""
    fold_label = f"FOLD {fold_idx:02d}" if fold_idx >= 0 else "CONCAT"
    print(f"\n  {'─' * 65}")
    print(f"  {fold_label} OOT ({date_oot}) — MFE/MAE PREDICTION RESULTS")
    print(f"  {'─' * 65}")
    # Target layout: [MFE_ticks, MAE_ticks, pnl_1s, pnl_5s, pnl_10s, mfe_mae_ratio]
    output_display = [
        ("MFE_ticks", "mfe_ticks"),
        ("MAE_ticks", "mae_ticks"),
        ("pnl_1s",    "pnl_1s"),
        ("pnl_5s",    "pnl_5s"),
        ("pnl_10s",   "pnl_10s"),
        ("MFE/MAE r", "mfe_mae_ratio"),
    ]
    print(f"  {'Output':<12} {'MSE':>8} {'L1_err':>8} {'Spearman_r':>10}  (primary: MFE/MAE)")
    print(f"  {'─' * 45}")
    for name, pfx in output_display:
        pfx_key = pfx.lower().replace("/", "_")
        mse = metrics.get(f"{pfx_key}_mse", 0)
        l1 = metrics.get(f"{pfx_key}_l1", 0)
        corr = metrics.get(f"{pfx_key}_corr", 0)
        star = " *" if pfx in ("mfe_ticks", "mae_ticks") else ""
        print(f"  {name:<12} {mse:>8.3f} {l1:>8.3f} {corr:>10.4f}{star}")

    print(f"\n  MFE corr: {metrics.get('mfe_corr', 0):.4f} | "
          f"MAE corr: {metrics.get('mae_corr', 0):.4f} | "
          f"pnl corr: {metrics.get('avg_pnl_corr', 0):.4f} | "
          f"overall: {metrics.get('avg_corr', 0):.4f}")


# ===========================================================================
# Main
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description="MFE/MAE Predictor — Adaptive TP/SL for ES Futures Execution"
    )
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=DEFAULT_LR)
    parser.add_argument("--hidden-dim", type=int, default=DEFAULT_HIDDEN_DIM)
    parser.add_argument("--n-layers", type=int, default=DEFAULT_N_LAYERS)
    parser.add_argument("--dropout", type=float, default=DEFAULT_DROPOUT)
    parser.add_argument("--max-folds", type=int, default=None,
                        help="Limit number of OOT folds (for testing)")
    parser.add_argument("--gpu", action="store_true",
                        help="Force GPU training (ignored if CUDA unavailable)")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Override output directory")
    args = parser.parse_args()

    global DEVICE
    if args.gpu and not torch.cuda.is_available():
        log.warning("--gpu specified but CUDA unavailable, running on CPU")
    if not torch.cuda.is_available():
        log.info("No CUDA device found — running on CPU (use Razer for GPU training)")

    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 70)
    log.info("MFE/MAE Predictor — Adaptive TP/SL for ES Futures Execution")
    log.info("=" * 70)
    log.info(f"  Device:     {DEVICE}")
    log.info(f"  Epochs:     {args.epochs}")
    log.info(f"  Batch size: {args.batch_size}")
    log.info(f"  LR:         {args.lr}")
    log.info(f"  Hidden dim: {args.hidden_dim} x {args.n_layers} layers")
    log.info(f"  Dropout:    {args.dropout}")
    log.info(f"  Output:     {output_dir}")
    log.info(f"  HC #0:      Sliding window walk-forward (NEVER expanding)")
    log.info(f"  Cost:       {ES_COMMISSION_TICKS:.3f} ticks RT (${ES_RT_COMMISSION:.2f})")
    log.info(f"  Data root:  {LVL3_ROOT}")
    log.info("=" * 70)

    # --- MLflow setup ---
    mlflow_run = None
    if MLFLOW_AVAILABLE:
        try:
            mlflow.set_tracking_uri(MLFLOW_URI)
            mlflow.set_experiment(EXPERIMENT_NAME)
            mlflow_run = mlflow.start_run(
                run_name=f"mfe_mae_wf_{time.strftime('%Y%m%d_%H%M%S')}"
            )
            mlflow.log_params({
                "epochs": args.epochs,
                "batch_size": args.batch_size,
                "lr": args.lr,
                "hidden_dim": args.hidden_dim,
                "n_layers": args.n_layers,
                "dropout": args.dropout,
                "device": DEVICE,
                "n_targets": N_TARGETS,
                "target_layout": "MFE_1s,MFE_5s,MFE_10s,MAE_1s,MAE_5s,MAE_10s",
                "loss_fn": "AsymmetricMSE (mfe_under=2x,mfe_over=1x,mae_under=2x,mae_over=1.5x)",
                "walk_forward": "sliding_window_HC0",
                "commission_ticks": ES_COMMISSION_TICKS,
                "commission_usd": ES_RT_COMMISSION,
            })
            log.info("MLflow run started.")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")

    # --- Load all available fold data ---
    log.info("\n--- Loading fold data ---")
    all_folds = {}
    for fold_idx in range(N_FOLDS):
        t0 = time.time()
        d = load_fold_data(fold_idx)
        if d is None:
            log.warning(f"Fold {fold_idx:02d}: skipped (data not found)")
            continue
        elapsed = time.time() - t0
        has_ptst = "yes" if d["ptst_pred"] is not None else "no"
        has_vol = "yes" if d["vol_pred"] is not None else "no"
        has_mbo = "yes" if d["mbo_data"] is not None else "no"
        log.info(
            f"  Fold {fold_idx:02d}: date={d['date']}, n={d['n_samples']:,}, "
            f"PatchTST={has_ptst}, vol={has_vol}, MBO={has_mbo} ({elapsed:.1f}s)"
        )
        all_folds[fold_idx] = d

    if len(all_folds) < 3:
        log.error("Not enough folds loaded (need at least 3). Check data paths.")
        if mlflow_run:
            mlflow.end_run()
        return

    # --- Build features for all folds (done once) ---
    log.info("\n--- Building features ---")
    fold_feats = {}
    feature_names = None
    for fold_idx, data in all_folds.items():
        t0 = time.time()
        feats, names = build_features_for_fold(data)
        elapsed = time.time() - t0
        fold_feats[fold_idx] = feats
        if feature_names is None:
            feature_names = names
            log.info(f"  Total features: {len(names)}")
        log.info(f"  Fold {fold_idx:02d}: {feats.shape} ({elapsed:.1f}s)")

    # --- Walk-forward training (HC #0: sliding window) ---
    log.info("\n--- Walk-forward training (sliding window, HC #0) ---")

    sorted_folds = sorted(all_folds.keys())
    n_avail = len(sorted_folds)

    # Sliding window: train on TRAIN_WINDOW folds, OOT on next fold
    # The oldest fold is dropped as we advance
    wf_results = []
    concat_predictions = []
    concat_targets = []
    concat_dates = []

    max_oot_folds = args.max_folds if args.max_folds else n_avail
    n_oot_trained = 0

    for oot_pos in range(TRAIN_WINDOW, n_avail):
        if n_oot_trained >= max_oot_folds:
            break

        # Sliding window: folds [oot_pos - TRAIN_WINDOW .. oot_pos - 1] are train+val
        # Last fold in that window is val, rest are train
        # OOT = oot_pos (never used in normalization or training)
        window_folds = sorted_folds[oot_pos - TRAIN_WINDOW:oot_pos]
        oot_fold = sorted_folds[oot_pos]

        if len(window_folds) < 2:
            continue

        # Split: last fold in window is val
        train_folds_idx = window_folds[:-1]  # first TRAIN_WINDOW-1 folds
        val_fold_idx = window_folds[-1]      # most recent fold before OOT

        log.info(f"\n{'='*60}")
        log.info(f"OOT fold {oot_fold:02d} ({all_folds[oot_fold]['date']})")
        log.info(f"  Train folds: {train_folds_idx}")
        log.info(f"  Val fold:    {val_fold_idx}")

        # Assemble train data
        train_X = np.concatenate([fold_feats[i] for i in train_folds_idx])
        train_y = np.concatenate([all_folds[i]["targets"] for i in train_folds_idx])

        # Val data
        val_X = fold_feats[val_fold_idx]
        val_y = all_folds[val_fold_idx]["targets"]

        # OOT features (for inference only)
        oot_X = fold_feats[oot_fold]
        oot_y = all_folds[oot_fold]["targets"]

        log.info(f"  Train: {len(train_X):,} samples | Val: {len(val_X):,} | OOT: {len(oot_X):,}")
        log.info(f"  Target stats (train): MFE_1s mean={train_y[:,0].mean():.2f}, "
                 f"MAE_1s mean={train_y[:,3].mean():.2f}")

        # Train
        model, results, val_preds = train_one_fold(
            train_X, train_y,
            val_X, val_y,
            fold_idx=oot_fold,
            args=args,
        )

        # Inference on OOT
        norm_mean = np.array(results["norm_mean"])
        norm_std = np.array(results["norm_std"])
        oot_norm = np.nan_to_num((oot_X - norm_mean) / norm_std, nan=0.0)

        model.eval()
        with torch.no_grad():
            oot_t = torch.from_numpy(oot_norm.astype(np.float32)).to(DEVICE)
            oot_preds_list = []
            for start in range(0, len(oot_t), 10000):
                end = min(start + 10000, len(oot_t))
                oot_preds_list.append(model(oot_t[start:end]).cpu().numpy())
        oot_preds = np.concatenate(oot_preds_list)

        # OOT metrics
        oot_metrics = compute_fold_metrics(oot_preds, oot_y)
        print_fold_summary(oot_fold, oot_metrics, all_folds[oot_fold]["date"])

        # Log to MLflow
        if MLFLOW_AVAILABLE:
            try:
                mlflow.log_metrics({
                    f"oot{oot_fold}_mfe_corr": oot_metrics["mfe_corr"],
                    f"oot{oot_fold}_mae_corr": oot_metrics["mae_corr"],
                    f"oot{oot_fold}_avg_corr": oot_metrics["avg_corr"],
                    f"oot{oot_fold}_val_loss": results["best_val_loss"],
                })
            except Exception:
                pass

        # Save model checkpoint
        checkpoint_path = output_dir / f"fold_{oot_fold:02d}_model.pt"
        torch.save({
            "model_state_dict": model.state_dict(),
            "norm_mean": norm_mean,
            "norm_std": norm_std,
            "feature_names": feature_names,
            "input_dim": results["input_dim"],
            "hidden_dim": args.hidden_dim,
            "n_layers": args.n_layers,
            "dropout": args.dropout,
            "oot_date": all_folds[oot_fold]["date"],
            "oot_metrics": oot_metrics,
            "val_metrics": results["metrics"],
            "best_epoch": results["best_epoch"],
            "commission_ticks": ES_COMMISSION_TICKS,
            "target_layout": "MFE_1s,MFE_5s,MFE_10s,MAE_1s,MAE_5s,MAE_10s",
        }, str(checkpoint_path))
        log.info(f"  Model saved: {checkpoint_path}")

        # Save OOT predictions
        pred_path = output_dir / f"fold_{oot_fold:02d}_predictions.npz"
        np.savez_compressed(
            str(pred_path),
            oot_predictions=oot_preds,         # (N, 6)
            oot_targets=oot_y,                  # (N, 6)
            oot_date=all_folds[oot_fold]["date"],
            cnn_predictions=all_folds[oot_fold]["predictions"],  # (N, 3) raw signal
            direction=all_folds[oot_fold]["direction"],           # (N,) trade direction
            mfe_overall=all_folds[oot_fold]["mfe_overall"],
            mae_overall=all_folds[oot_fold]["mae_overall"],
        )
        log.info(f"  Predictions saved: {pred_path}")

        concat_predictions.append(oot_preds)
        concat_targets.append(oot_y)
        concat_dates.append(all_folds[oot_fold]["date"])

        wf_results.append({
            "fold": int(oot_fold),
            "date": all_folds[oot_fold]["date"],
            "n_oot": int(len(oot_X)),
            "best_epoch": int(results["best_epoch"]),
            "val_loss": float(results["best_val_loss"]),
            "oot_metrics": {k: float(v) for k, v in oot_metrics.items()},
        })

        n_oot_trained += 1

    if not wf_results:
        log.error("No OOT folds completed. Increase data or reduce --max-folds.")
        if mlflow_run:
            mlflow.end_run()
        return

    # --- Walk-forward summary ---
    all_pred = np.concatenate(concat_predictions)
    all_tgt = np.concatenate(concat_targets)
    concat_metrics = compute_fold_metrics(all_pred, all_tgt)

    print("\n" + "=" * 70)
    print("WALK-FORWARD SUMMARY (concatenated OOT)")
    print("=" * 70)
    print_fold_summary(-1, concat_metrics, f"concat ({len(concat_dates)} folds)")

    print("\n  Per-fold Spearman correlations:")
    for r in wf_results:
        print(f"    Fold {r['fold']:02d} ({r['date']}): "
              f"MFE={r['oot_metrics']['mfe_corr']:.4f} | "
              f"MAE={r['oot_metrics']['mae_corr']:.4f} | "
              f"pnl={r['oot_metrics']['avg_pnl_corr']:.4f}")

    print(f"\n  Concat MFE corr: {concat_metrics['mfe_corr']:.4f}")
    print(f"  Concat MAE corr: {concat_metrics['mae_corr']:.4f}")
    print(f"  Concat pnl corr: {concat_metrics['avg_pnl_corr']:.4f}")
    print(f"  Concat avg corr: {concat_metrics['avg_corr']:.4f}")
    print("=" * 70)

    # Save walk-forward summary
    summary = {
        "experiment": EXPERIMENT_NAME,
        "run_timestamp": time.strftime("%Y%m%d_%H%M%S"),
        "device": DEVICE,
        "n_oot_folds": len(wf_results),
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "lr": args.lr,
        "hidden_dim": args.hidden_dim,
        "n_layers": args.n_layers,
        "dropout": args.dropout,
        "commission_ticks": ES_COMMISSION_TICKS,
        "commission_usd": ES_RT_COMMISSION,
        "target_layout": "MFE_ticks,MAE_ticks,pnl_1s,pnl_5s,pnl_10s,mfe_mae_ratio",
        "loss_fn": "Composite: 0.4*AsymMSE(MFE)+0.4*AsymMSE(MAE)+0.15*Huber(pnl)+0.05*MSE(ratio)",
        "concat_metrics": {k: float(v) for k, v in concat_metrics.items()},
        "per_fold_results": wf_results,
    }
    summary_path = output_dir / "walk_forward_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"\nSummary saved: {summary_path}")

    # Save concat predictions
    concat_pred_path = output_dir / "concat_oot_predictions.npz"
    np.savez_compressed(
        str(concat_pred_path),
        predictions=all_pred,
        targets=all_tgt,
        dates=np.array(concat_dates),
    )
    log.info(f"Concat predictions saved: {concat_pred_path}")

    # MLflow summary metrics
    if MLFLOW_AVAILABLE:
        try:
            mlflow.log_metrics({
                "concat_mfe_corr": concat_metrics["mfe_corr"],
                "concat_mae_corr": concat_metrics["mae_corr"],
                "concat_pnl_corr": concat_metrics["avg_pnl_corr"],
                "concat_avg_corr": concat_metrics["avg_corr"],
                "n_oot_folds": len(wf_results),
            })
            mlflow.log_artifact(str(summary_path))
            mlflow.log_artifact(str(concat_pred_path))
            mlflow.end_run()
        except Exception:
            pass

    log.info("\n" + "=" * 70)
    log.info("TRAINING COMPLETE")
    log.info(f"  Output dir: {output_dir}")
    log.info(f"  Concat MFE Spearman: {concat_metrics['mfe_corr']:.4f}")
    log.info(f"  Concat MAE Spearman: {concat_metrics['mae_corr']:.4f}")
    log.info(f"  Concat pnl Spearman: {concat_metrics['avg_pnl_corr']:.4f}")
    log.info(f"  Models: {output_dir}/fold_XX_model.pt")
    log.info(f"  Predictions: {output_dir}/fold_XX_predictions.npz")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
