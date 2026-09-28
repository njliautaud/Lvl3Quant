#!/usr/bin/env python3
"""
Quantile Execution Model v2 — Regime-adaptive distribution-aware trade entry
======================================================================

Changes from v1:
- 30-day training window (was 60d) for faster regime adaptation
- 4 new regime features (46 total): realized_vol_5min, trend_indicator,
  hour_of_day, minute_bucket
- Early stopping on validation Spearman (patience=5) instead of val loss
- MLflow experiment: quantile_exec_v2_regime

Architecture: MLP with quantile regression heads (256 -> 128 -> 64 -> 5)
Loss: Pinball/quantile loss (asymmetric L1)
Walk-forward: SLIDING window, 30d train / 5d eval (mandatory per DIRECTIVES)

Author: Claude (Infrastructure Builder)
Date: 2026-05-27
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LVL3_ROOT = Path("/home/nick/Lvl3Quant")
MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
OUTPUT_DIR = LVL3_ROOT / "output" / "quantile_exec_v2"

# Prediction directories (checked in priority order)
PRED_DIRS = [
    LVL3_ROOT / "output" / "cnn_mamba_v2_all_oot",
    LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_oot",
    LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_inference",
    LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar",
]

ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE  # 0.376

# Quantile taus
QUANTILE_TAUS = [0.10, 0.25, 0.50, 0.75, 0.90]
N_QUANTILES = len(QUANTILE_TAUS)

# MBO event column indices
COL_PRICE_REL = 3
COL_SPREAD = 5
COL_SIDE = 2
COL_QTY_LOG = 4

# CNN-Mamba prediction parameters
PRED_STRIDE = 250   # events between predictions
PRED_WINDOW = 3000  # events per CNN window

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "quantile_exec_v2_regime"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_DIR = LVL3_ROOT / "logs"
LOG_FILE = LOG_DIR / "quantile_exec_v2.log"
os.makedirs(str(LOG_DIR), exist_ok=True)
logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [QUANT_EXEC_V2] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("quant_exec_v2")

# Ensure unbuffered stdout
sys.stdout.reconfigure(line_buffering=True) if hasattr(sys.stdout, 'reconfigure') else None

# ---------------------------------------------------------------------------
# Feature names (46 features = 42 original + 4 regime features)
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    "pred_1s", "pred_5s", "pred_10s",
    "abs_pred_1s", "abs_pred_5s", "abs_pred_10s",
    "signal_direction", "confidence_tier",
    "signal_agreement", "signal_strength",
    "pred_std", "pred_range",
    "pred_ratio_5s_1s", "pred_ratio_10s_1s",
    "book_imbalance", "bid_depth_log", "ask_depth_log",
    "spread", "depth_ratio",
    "vol_50", "vol_200", "vol_500", "vol_trend",
    "momentum_20", "momentum_50", "momentum_200",
    "vol_imb_50", "vol_imb_200", "vol_imb_500", "vol_imb_trend",
    "intensity_50", "intensity_200", "intensity_ratio",
    "time_of_day", "minutes_since_open", "is_first_30min", "is_last_30min",
    "conf_x_vol", "conf_x_imbalance", "conf_x_momentum",
    "aligned_imbalance", "aligned_momentum",
    # v2 regime features
    "realized_vol_5min", "trend_indicator", "hour_of_day", "minute_bucket",
]
N_FEATURES = len(FEATURE_NAMES)  # 46

HORIZONS = ["1s", "5s", "10s"]

# ---------------------------------------------------------------------------
# Vectorized rolling helpers
# ---------------------------------------------------------------------------

def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    out = np.zeros(n, dtype=np.float32)
    if n < window:
        return out
    vals = values.astype(np.float64)
    cs = np.cumsum(vals)
    cs2 = np.cumsum(vals ** 2)
    cs_pad = np.concatenate([[0.0], cs])
    cs2_pad = np.concatenate([[0.0], cs2])
    idx = np.arange(window, n + 1)
    sm = cs_pad[idx] - cs_pad[idx - window]
    sm2 = cs2_pad[idx] - cs2_pad[idx - window]
    mean = sm / window
    var = np.maximum(sm2 / window - mean ** 2, 0.0)
    out[window - 1:n] = np.sqrt(var).astype(np.float32)
    return out


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    n = len(values)
    out = np.zeros(n, dtype=np.float32)
    if n < window:
        return out
    vals = values.astype(np.float64)
    cs = np.cumsum(vals)
    cs_pad = np.concatenate([[0.0], cs])
    idx = np.arange(window, n + 1)
    out[window - 1:n] = ((cs_pad[idx] - cs_pad[idx - window]) / window).astype(np.float32)
    return out


def _ema(values: np.ndarray, span: int) -> np.ndarray:
    """Exponential moving average, vectorized."""
    alpha = 2.0 / (span + 1.0)
    n = len(values)
    out = np.zeros(n, dtype=np.float64)
    out[0] = values[0]
    for i in range(1, n):
        out[i] = alpha * values[i] + (1 - alpha) * out[i - 1]
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# Feature extraction (fully vectorized — 46 features including regime)
# ---------------------------------------------------------------------------

def extract_features_for_date(
    mbo_path: Path,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 46 features + net_ticks_passive target for one date.
    42 original features + 4 regime features.
    """
    data = np.load(str(mbo_path))
    events = data["events"]
    timestamps = data["timestamps"]

    n_events = len(events)
    n_preds = len(predictions)

    # Raw columns
    price_rel = events[:, COL_PRICE_REL].astype(np.float64)
    spread_arr = events[:, COL_SPREAD].astype(np.float32)
    side_arr = events[:, COL_SIDE].astype(np.float32)
    qty_log_arr = events[:, COL_QTY_LOG].astype(np.float32)
    ts_s = timestamps.astype(np.float64) / 1e9

    price_changes = np.diff(price_rel, prepend=price_rel[0]).astype(np.float64)

    # Rolling features (vectorized over all events)
    vol_50 = _rolling_std(price_changes, 50)
    vol_200 = _rolling_std(price_changes, 200)
    vol_500 = _rolling_std(price_changes, 500)
    mom_20 = _rolling_mean(price_changes, 20)
    mom_50 = _rolling_mean(price_changes, 50)
    mom_200 = _rolling_mean(price_changes, 200)

    # --- v2 regime features (computed over all events, indexed later) ---
    # realized_vol_5min: rolling std over ~1200 events (~5 min at ~4 events/s)
    vol_5min = _rolling_std(price_changes, 1200)

    # trend_indicator: sign of 30s EMA slope (~120 events)
    ema_30s = _ema(price_rel, 120)
    ema_slope = np.diff(ema_30s, prepend=ema_30s[0])
    trend_ind = np.sign(ema_slope).astype(np.float32)

    # Volume tracking
    qty = np.exp(np.clip(qty_log_arr, -10, 10))
    buy_mask = (side_arr > 0).astype(np.float32)
    sell_mask = (side_arr < 0).astype(np.float32)
    buy_cum = np.cumsum(qty * buy_mask)
    sell_cum = np.cumsum(qty * sell_mask)
    buy_cum_pad = np.concatenate([[0.0], buy_cum])
    sell_cum_pad = np.concatenate([[0.0], sell_cum])

    # Book features
    bid_depth_raw = events[:, 6] if events.shape[1] > 6 else np.ones(n_events, dtype=np.float32)
    ask_depth_raw = events[:, 7] if events.shape[1] > 7 else np.ones(n_events, dtype=np.float32)
    book_imb_raw = events[:, 8] if events.shape[1] > 8 else np.zeros(n_events, dtype=np.float32)

    # Event indices for each prediction
    pred_event_indices = PRED_WINDOW + np.arange(n_preds) * PRED_STRIDE

    # Validity mask
    valid_mask = pred_event_indices < (n_events - 1)
    pred_nonzero = ~((predictions[:, 0] == 0) & (predictions[:, 1] == 0) & (predictions[:, 2] == 0))
    labels_valid = ~np.any(np.isnan(labels), axis=1)
    combined_mask = valid_mask & pred_nonzero & labels_valid
    valid_indices = np.where(combined_mask)[0]

    n_valid = len(valid_indices)
    if n_valid == 0:
        return (
            np.zeros((0, N_FEATURES), dtype=np.float32),
            np.zeros(0, dtype=np.float32),
            np.zeros((0, 3), dtype=np.float32),
        )

    # Event indices for valid predictions
    eidx_all = pred_event_indices[valid_indices].astype(int)

    # ---- Vectorized signal features ----
    p1 = predictions[valid_indices, 0].astype(np.float32)
    p5 = predictions[valid_indices, 1].astype(np.float32)
    p10 = predictions[valid_indices, 2].astype(np.float32)
    a1, a5, a10 = np.abs(p1), np.abs(p5), np.abs(p10)
    sig_dir = np.where(p1 > 0, 1.0, -1.0).astype(np.float32)

    tier = np.where(a1 < 0.10, 0.0, np.where(a1 < 0.25, 1.0, np.where(a1 < 0.50, 2.0, 3.0))).astype(np.float32)
    signs_agree = (
        (np.sign(p1) == np.sign(p5)) & (np.sign(p5) == np.sign(p10)) & (np.sign(p1) != 0)
    ).astype(np.float32)
    strength = (a1 + a5 + a10) / 3.0

    preds_stack = np.stack([p1, p5, p10], axis=1)
    p_std = np.std(preds_stack, axis=1).astype(np.float32)
    p_range = (np.max(preds_stack, axis=1) - np.min(preds_stack, axis=1)).astype(np.float32)
    ratio_5_1 = np.where(np.abs(p1) > 1e-6, p5 / (p1 + 1e-8), 0.0).astype(np.float32)
    ratio_10_1 = np.where(np.abs(p1) > 1e-6, p10 / (p1 + 1e-8), 0.0).astype(np.float32)

    # ---- Vectorized book features ----
    bimb = book_imb_raw[eidx_all]
    bd_raw = np.maximum(bid_depth_raw[eidx_all], 0.0)
    ad_raw = np.maximum(ask_depth_raw[eidx_all], 0.0)
    bd = np.log1p(bd_raw).astype(np.float32)
    ad = np.log1p(ad_raw).astype(np.float32)
    spr = spread_arr[eidx_all]
    depth_rat = (np.exp(bd) / (np.exp(ad) + 1e-8)).astype(np.float32)

    # ---- Vectorized volatility/momentum ----
    v50 = vol_50[eidx_all]
    v200 = vol_200[eidx_all]
    v500 = vol_500[eidx_all]
    vol_trend_val = np.where(v200 > 1e-6, (v50 - v200) / (v200 + 1e-8), 0.0).astype(np.float32)

    m20 = mom_20[eidx_all]
    m50_v = mom_50[eidx_all]
    m200_v = mom_200[eidx_all]

    # ---- Volume imbalance (vectorized) ----
    def _vol_imb_vec(eidx_arr, window):
        starts = np.maximum(0, eidx_arr - window)
        bv = buy_cum_pad[eidx_arr + 1] - buy_cum_pad[starts + 1]
        sv = sell_cum_pad[eidx_arr + 1] - sell_cum_pad[starts + 1]
        tot = bv + sv
        return np.where(tot > 0, (bv - sv) / (tot + 1e-8), 0.0).astype(np.float32)

    vi_50 = _vol_imb_vec(eidx_all, 50)
    vi_200 = _vol_imb_vec(eidx_all, 200)
    vi_500 = _vol_imb_vec(eidx_all, 500)
    vi_trend = (vi_50 - vi_200).astype(np.float32)

    # ---- Trade intensity (vectorized) ----
    def _intensity_vec(eidx_arr, window):
        starts = np.maximum(0, eidx_arr - window)
        dt = ts_s[eidx_arr] - ts_s[starts]
        return np.where(dt > 0.01, window / (dt + 1e-8), 0.0).astype(np.float32)

    ti_50 = _intensity_vec(eidx_all, 50)
    ti_200 = _intensity_vec(eidx_all, 200)
    int_ratio = np.where(ti_200 > 1e-6, ti_50 / (ti_200 + 1e-8), 1.0).astype(np.float32)

    # ---- Time features (vectorized) ----
    cur_ts = ts_s[eidx_all]
    seconds_in_day = cur_ts % 86400
    et_seconds = (seconds_in_day - 4 * 3600) % 86400
    rth_start = 9.5 * 3600
    rth_end = 16.0 * 3600
    tod = np.clip((et_seconds - rth_start) / (rth_end - rth_start), 0.0, 1.0).astype(np.float32)
    minutes = (tod * 390.0).astype(np.float32)
    is_first_30 = (minutes < 30).astype(np.float32)
    is_last_30 = (minutes > 360).astype(np.float32)

    # ---- Interaction features ----
    conf_x_vol = (a1 * v200).astype(np.float32)
    conf_x_imb = (a1 * bimb * sig_dir).astype(np.float32)
    conf_x_mom = (a1 * m50_v * sig_dir).astype(np.float32)
    aligned_imb = (sig_dir * bimb > 0).astype(np.float32)
    aligned_mom = (sig_dir * m50_v > 0).astype(np.float32)

    # ---- v2 regime features ----
    # realized_vol_5min: 5-minute rolling volatility at each prediction point
    rv_5min = vol_5min[eidx_all]

    # trend_indicator: sign of 30s EMA slope
    trend_val = trend_ind[eidx_all]

    # hour_of_day: normalized 0-1 over 24h (ET hours / 24)
    hour_of_day = ((et_seconds / 3600.0) / 24.0).astype(np.float32)

    # minute_bucket: 15-min buckets within RTH (0-25), normalized 0-1
    rth_minutes = np.clip((et_seconds - rth_start) / 60.0, 0, 390).astype(np.float32)
    minute_bucket = (np.floor(rth_minutes / 15.0) / 26.0).astype(np.float32)  # 26 buckets max

    # ---- Assemble feature matrix (46 features) ----
    features_out = np.stack([
        p1, p5, p10, a1, a5, a10, sig_dir, tier,
        signs_agree, strength, p_std, p_range, ratio_5_1, ratio_10_1,
        bimb, bd, ad, spr, depth_rat,
        v50, v200, v500, vol_trend_val,
        m20, m50_v, m200_v,
        vi_50, vi_200, vi_500, vi_trend,
        ti_50, ti_200, int_ratio,
        tod, minutes, is_first_30, is_last_30,
        conf_x_vol, conf_x_imb, conf_x_mom,
        aligned_imb, aligned_mom,
        # v2 regime features
        rv_5min, trend_val, hour_of_day, minute_bucket,
    ], axis=1).astype(np.float32)

    # ---- Target: net_ticks_passive at 5s horizon ----
    l5 = labels[valid_indices, 1]  # 5s horizon raw label (ticks)
    dir_move_5s = (l5 * sig_dir).astype(np.float32)
    net_ticks_passive = (dir_move_5s - COMMISSION_TICKS).astype(np.float32)

    meta_out = np.stack([
        cur_ts,
        valid_indices.astype(np.float64),
        sig_dir.astype(np.float64),
    ], axis=1).astype(np.float32)

    return features_out, net_ticks_passive, meta_out


# ---------------------------------------------------------------------------
# Data loading: collect predictions from multiple directories
# ---------------------------------------------------------------------------

def build_date_pred_index() -> Dict[str, Path]:
    """Build date -> prediction file path, preferring all_oot > bulk_oot > bulk_inference > fold."""
    index = {}

    for pred_dir in PRED_DIRS:
        if not pred_dir.exists():
            continue
        for f in sorted(pred_dir.glob("*_predictions.npz")):
            stem = f.stem
            if stem.startswith("fold_") or stem.startswith("concat"):
                continue
            date_str = stem.replace("_oot_predictions", "").replace("_predictions", "")
            if len(date_str) == 8 and date_str.isdigit() and date_str not in index:
                index[date_str] = f

    # Also check fold files
    fold_dir = PRED_DIRS[-1]
    if fold_dir.exists():
        for f in sorted(fold_dir.glob("fold_*_oot_predictions.npz")):
            try:
                data = np.load(str(f), allow_pickle=True)
                oot_files = data.get("oot_files", [])
                if len(oot_files) > 0:
                    date_str = Path(str(oot_files[0])).stem.replace("_mbo_events", "")
                    if date_str not in index:
                        index[date_str] = f
            except Exception:
                pass

    return index


def load_all_dates() -> List[dict]:
    """Load features + net_ticks_passive targets for all dates."""
    pred_index = build_date_pred_index()
    mbo_files = {
        Path(f).stem.replace("_mbo_events", ""): f
        for f in sorted(MBO_DIR.glob("*_mbo_events.npz"))
    }

    log.info(f"Found {len(pred_index)} dates with predictions, {len(mbo_files)} MBO files")

    all_dates = []
    for date_str in sorted(pred_index.keys()):
        if date_str not in mbo_files:
            continue

        mbo_path = mbo_files[date_str]
        pred_path = pred_index[date_str]

        t0 = time.time()
        try:
            pred_data = np.load(str(pred_path), allow_pickle=True)
            predictions = pred_data["predictions"]
            labels = pred_data["labels"]

            features, targets, meta = extract_features_for_date(
                Path(mbo_path), predictions, labels
            )
            elapsed = time.time() - t0

            if len(features) < 10:
                log.warning(f"  {date_str}: only {len(features)} samples, skip")
                continue

            all_dates.append({
                "date": date_str,
                "features": features,
                "targets": targets,
                "meta": meta,
                "n_samples": len(features),
            })

            if len(all_dates) % 20 == 0:
                log.info(f"  {len(all_dates)} dates loaded ({date_str}: {len(features):,} in {elapsed:.1f}s)")

        except Exception as e:
            log.error(f"  {date_str}: {e}")
            traceback.print_exc()
            continue

        gc.collect()

    total_samples = sum(d["n_samples"] for d in all_dates)
    if total_samples > 0:
        all_targets = np.concatenate([d["targets"] for d in all_dates])
        pos_rate = float(np.mean(all_targets > 0))
        mean_ticks = float(np.mean(all_targets))
        log.info(f"Total: {len(all_dates)} dates, {total_samples:,} samples")
        log.info(f"Target stats: mean={mean_ticks:.4f} ticks, positive_rate={pos_rate:.1%}")
        log.info(f"Target quantiles: p10={np.percentile(all_targets, 10):.3f} "
                 f"p25={np.percentile(all_targets, 25):.3f} "
                 f"p50={np.percentile(all_targets, 50):.3f} "
                 f"p75={np.percentile(all_targets, 75):.3f} "
                 f"p90={np.percentile(all_targets, 90):.3f}")
    else:
        log.info(f"Total: {len(all_dates)} dates, 0 samples")

    return all_dates


# ---------------------------------------------------------------------------
# Quantile loss
# ---------------------------------------------------------------------------

def quantile_loss(pred: torch.Tensor, target: torch.Tensor, tau: float) -> torch.Tensor:
    """Pinball loss for quantile regression."""
    diff = target - pred
    return torch.mean(torch.max(tau * diff, (tau - 1) * diff))


def multi_quantile_loss(
    pred: torch.Tensor, target: torch.Tensor, taus: List[float]
) -> torch.Tensor:
    """Combined quantile loss across all taus. pred: (batch, n_quantiles), target: (batch,)."""
    total_loss = torch.tensor(0.0, device=pred.device)
    for i, tau in enumerate(taus):
        total_loss = total_loss + quantile_loss(pred[:, i], target, tau)
    return total_loss / len(taus)


# ---------------------------------------------------------------------------
# Model: Quantile Regression MLP
# ---------------------------------------------------------------------------

class QuantileExecMLP(nn.Module):
    """
    MLP with quantile regression heads.
    Architecture: 46 -> 256 -> 128 -> 64 -> 5 (one output per quantile)
    BatchNorm + Dropout between layers.
    """
    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dims: List[int] = [256, 128, 64],
        n_quantiles: int = N_QUANTILES,
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for h_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, h_dim),
                nn.BatchNorm1d(h_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = h_dim

        self.backbone = nn.Sequential(*layers)
        self.quantile_head = nn.Linear(prev_dim, n_quantiles)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.backbone(x)
        return self.quantile_head(h)


# ---------------------------------------------------------------------------
# Training one fold — early stopping on validation SPEARMAN (not loss)
# ---------------------------------------------------------------------------

def train_one_fold(
    model: QuantileExecMLP,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    fold_idx: int,
    taus: List[float],
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[dict, dict, np.ndarray, np.ndarray]:
    """
    Train one fold. Early stopping on validation Spearman(q50, actual) with patience=5.
    """
    from scipy.stats import spearmanr

    # Normalize using TRAIN stats only
    train_mean = np.nanmean(train_X, axis=0)
    train_std = np.nanstd(train_X, axis=0)
    train_std[train_std < 1e-8] = 1.0

    X_train = np.nan_to_num((train_X - train_mean) / train_std, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num((val_X - train_mean) / train_std, nan=0.0, posinf=0.0, neginf=0.0)

    log.info(f"  Train: {len(train_y):,} samples, target mean={np.mean(train_y):.4f}, "
             f"std={np.std(train_y):.4f}")

    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(train_y, dtype=torch.float32, device=DEVICE)

    X_v = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_v = torch.tensor(val_y, dtype=torch.float32, device=DEVICE)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    model = model.to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    best_val_spearman = -float("inf")
    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 5  # early stopping on Spearman

    q50_idx = taus.index(0.50)

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            q_preds = model(batch_x)
            loss = multi_quantile_loss(q_preds, batch_y, taus)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_q_preds = model(X_v)
            val_loss = multi_quantile_loss(val_q_preds, y_v, taus).item()
            val_q50 = val_q_preds[:, q50_idx].cpu().numpy()

        # Early stopping criterion: validation Spearman(q50, actual)
        val_spearman, _ = spearmanr(val_q50, val_y.cpu().numpy() if isinstance(val_y, torch.Tensor) else val_y)
        if np.isnan(val_spearman):
            val_spearman = 0.0

        if val_spearman > best_val_spearman:
            best_val_spearman = val_spearman
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info(f"  F{fold_idx} E{epoch+1}/{epochs}: "
                     f"train_qloss={np.mean(train_losses):.4f} val_qloss={val_loss:.4f} "
                     f"val_spearman={val_spearman:.4f} best_spearman={best_val_spearman:.4f}")
            sys.stdout.flush()

        if patience_counter >= patience:
            log.info(f"  Early stop at epoch {epoch+1} (best spearman={best_val_spearman:.4f})")
            sys.stdout.flush()
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()

    # Generate val predictions for evaluation
    with torch.no_grad():
        val_q_preds = model(X_v).cpu().numpy()

    metrics = evaluate_quantile_model(val_q_preds, val_y.cpu().numpy() if isinstance(val_y, torch.Tensor) else val_y, taus, fold_idx)
    metrics["best_val_spearman"] = round(best_val_spearman, 6)
    metrics["best_val_qloss"] = round(best_val_loss, 6)

    del X_t, y_t, dataset, loader
    gc.collect()
    torch.cuda.empty_cache()

    return metrics, best_state, train_mean, train_std


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_quantile_model(
    q_preds: np.ndarray,
    actuals: np.ndarray,
    taus: List[float],
    fold_idx: int,
) -> dict:
    from scipy.stats import spearmanr

    n = len(actuals)
    metrics = {
        "fold": fold_idx,
        "n_val_samples": n,
        "actual_mean": round(float(np.mean(actuals)), 6),
        "actual_std": round(float(np.std(actuals)), 6),
    }

    # 1. Mean quantile loss across all taus
    total_qloss = 0.0
    per_tau_loss = {}
    for i, tau in enumerate(taus):
        diff = actuals - q_preds[:, i]
        ql = float(np.mean(np.maximum(tau * diff, (tau - 1) * diff)))
        per_tau_loss[f"qloss_q{int(tau*100)}"] = round(ql, 6)
        total_qloss += ql
    mean_qloss = total_qloss / len(taus)
    metrics["mean_qloss"] = round(mean_qloss, 6)
    metrics.update(per_tau_loss)

    # 2. Calibration
    for i, tau in enumerate(taus):
        actual_coverage = float(np.mean(actuals < q_preds[:, i]))
        metrics[f"calib_q{int(tau*100)}"] = round(actual_coverage, 4)
        metrics[f"calib_err_q{int(tau*100)}"] = round(abs(actual_coverage - tau), 4)

    mean_calib_err = np.mean([metrics[f"calib_err_q{int(tau*100)}"] for tau in taus])
    metrics["mean_calib_err"] = round(float(mean_calib_err), 4)

    # 3. Spearman correlation of q50 vs actual
    q50_idx = taus.index(0.50)
    rho, pval = spearmanr(q_preds[:, q50_idx], actuals)
    metrics["spearman_q50"] = round(float(rho), 6)
    metrics["spearman_pval"] = float(pval)

    # 4. High-confidence trade rate: q10 > 0.376
    q10_idx = taus.index(0.10)
    high_conf_mask = q_preds[:, q10_idx] > COMMISSION_TICKS
    high_conf_rate = float(np.mean(high_conf_mask))
    metrics["high_conf_rate"] = round(high_conf_rate, 4)
    metrics["high_conf_count"] = int(high_conf_mask.sum())

    # 5. High-confidence edge
    if high_conf_mask.sum() > 0:
        hc_actual_mean = float(np.mean(actuals[high_conf_mask]))
        hc_actual_wr = float(np.mean(actuals[high_conf_mask] > 0))
        metrics["high_conf_edge"] = round(hc_actual_mean, 4)
        metrics["high_conf_wr"] = round(hc_actual_wr, 4)
    else:
        metrics["high_conf_edge"] = 0.0
        metrics["high_conf_wr"] = 0.0

    # q25 > 0 moderate confidence
    q25_idx = taus.index(0.25)
    mod_conf_mask = q_preds[:, q25_idx] > 0
    metrics["mod_conf_rate"] = round(float(np.mean(mod_conf_mask)), 4)
    if mod_conf_mask.sum() > 0:
        metrics["mod_conf_edge"] = round(float(np.mean(actuals[mod_conf_mask])), 4)
        metrics["mod_conf_wr"] = round(float(np.mean(actuals[mod_conf_mask] > 0)), 4)

    # Predicted quantile means
    for i, tau in enumerate(taus):
        metrics[f"pred_mean_q{int(tau*100)}"] = round(float(np.mean(q_preds[:, i])), 4)

    # Quantile crossing check
    crossings = 0
    for j in range(len(taus) - 1):
        crossings += int(np.sum(q_preds[:, j] > q_preds[:, j + 1]))
    metrics["quantile_crossings"] = crossings
    metrics["crossing_rate"] = round(crossings / (n * (len(taus) - 1)), 4)

    return metrics


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def run_walk_forward(
    all_dates: List[dict],
    n_train_days: int = 30,
    n_eval_days: int = 5,
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    hidden_dims: List[int] = [256, 128, 64],
    dropout: float = 0.3,
    taus: List[float] = QUANTILE_TAUS,
):
    n_dates = len(all_dates)
    if n_dates < n_train_days + n_eval_days:
        log.warning(f"Only {n_dates} dates, reducing train from {n_train_days}")
        n_train_days = max(5, n_dates - n_eval_days)

    n_folds = max(1, (n_dates - n_train_days) // n_eval_days)
    log.info(f"Walk-forward: {n_dates} dates, {n_train_days}d train + {n_eval_days}d eval = {n_folds} folds")

    # MLflow setup
    mlflow_run = None
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow_run = mlflow.start_run(
            run_name=f"quantile_exec_v2_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        mlflow.log_params({
            "model_type": "quantile_regression_mlp_v2_regime",
            "target": "net_ticks_passive_5s",
            "taus": str(taus),
            "n_quantiles": len(taus),
            "n_train_days": n_train_days,
            "n_eval_days": n_eval_days,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "weight_decay": 1e-4,
            "hidden_dims": str(hidden_dims),
            "dropout": dropout,
            "n_features": N_FEATURES,
            "device": str(DEVICE),
            "n_total_dates": n_dates,
            "n_folds": n_folds,
            "commission_ticks": COMMISSION_TICKS,
            "pred_stride": PRED_STRIDE,
            "pred_window": PRED_WINDOW,
            "loss": "pinball/quantile",
            "optimizer": "Adam",
            "scheduler": "CosineAnnealingLR",
            "early_stop_metric": "val_spearman_q50",
            "early_stop_patience": 5,
            "regime_features": "realized_vol_5min,trend_indicator,hour_of_day,minute_bucket",
            "v1_changes": "30d_window,regime_features,spearman_early_stop",
        })
        log.info(f"MLflow run: {mlflow_run.info.run_id}")
    except Exception as e:
        log.warning(f"MLflow: {e}")

    all_fold_metrics = []
    all_oot_q_preds = []
    all_oot_actuals = []
    all_oot_metas = []

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    for fold_idx in range(n_folds):
        fold_start = fold_idx * n_eval_days
        train_end = fold_start + n_train_days
        eval_end = min(train_end + n_eval_days, n_dates)

        if train_end >= n_dates:
            break

        train_dates = all_dates[fold_start:train_end]
        eval_dates = all_dates[train_end:eval_end]
        if not eval_dates:
            break

        log.info(f"\n{'='*60}")
        log.info(f"Fold {fold_idx}: train=[{train_dates[0]['date']}..{train_dates[-1]['date']}] "
                 f"({len(train_dates)}d), eval=[{eval_dates[0]['date']}..{eval_dates[-1]['date']}] "
                 f"({len(eval_dates)}d)")
        sys.stdout.flush()

        # Assemble train data
        train_X = np.concatenate([d["features"] for d in train_dates])
        train_y = np.concatenate([d["targets"] for d in train_dates])

        # Assemble eval data
        eval_X = np.concatenate([d["features"] for d in eval_dates])
        eval_y = np.concatenate([d["targets"] for d in eval_dates])
        eval_meta = np.concatenate([d["meta"] for d in eval_dates])

        log.info(f"  Train: {len(train_X):,} samples  Eval: {len(eval_X):,} samples")

        # Create model
        model = QuantileExecMLP(
            input_dim=N_FEATURES, hidden_dims=hidden_dims,
            n_quantiles=len(taus), dropout=dropout,
        )

        # Train
        fold_metrics, best_state, train_mean, train_std = train_one_fold(
            model, train_X, train_y, eval_X, eval_y,
            fold_idx, taus, epochs, batch_size, lr,
        )
        all_fold_metrics.append(fold_metrics)

        # Save fold artifacts (.pt weights + .npz predictions)
        fold_dir = OUTPUT_DIR / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, str(fold_dir / "model.pt"))
        np.savez(str(fold_dir / "norm_stats.npz"), mean=train_mean, std=train_std)

        # Generate OOT predictions
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
        model.eval()

        eval_X_norm = np.nan_to_num(
            (eval_X - train_mean) / train_std, nan=0.0, posinf=0.0, neginf=0.0
        )
        with torch.no_grad():
            X_v = torch.tensor(eval_X_norm, dtype=torch.float32, device=DEVICE)
            q_preds = model(X_v).cpu().numpy()

        # Save fold predictions as .npz
        np.savez(
            str(fold_dir / "oot_predictions.npz"),
            q_preds=q_preds,
            actuals=eval_y,
            features=eval_X,
            meta=eval_meta,
            taus=np.array(taus),
            train_dates=np.array([d["date"] for d in train_dates]),
            eval_dates=np.array([d["date"] for d in eval_dates]),
        )

        all_oot_q_preds.append(q_preds)
        all_oot_actuals.append(eval_y)
        all_oot_metas.append(eval_meta)

        # MLflow per-fold metrics
        try:
            import mlflow
            for k, v in fold_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"fold_{k}", v, step=fold_idx)
        except Exception:
            pass

        # Log fold summary
        ql = fold_metrics.get("mean_qloss", 0)
        ce = fold_metrics.get("mean_calib_err", 0)
        rho = fold_metrics.get("spearman_q50", 0)
        hcr = fold_metrics.get("high_conf_rate", 0)
        hce = fold_metrics.get("high_conf_edge", 0)
        hcw = fold_metrics.get("high_conf_wr", 0)
        hcc = fold_metrics.get("high_conf_count", 0)
        log.info(f"  FOLD {fold_idx}: qloss={ql:.4f} calib_err={ce:.4f} "
                 f"spearman={rho:.4f} hc_rate={hcr:.3f} hc_edge={hce:+.4f} "
                 f"hc_wr={hcw:.3f} hc_count={hcc}")
        sys.stdout.flush()

        del model, X_v, q_preds
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Aggregate concat metrics across all folds
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}")
    log.info("AGGREGATE CONCAT RESULTS (all OOT folds)")
    log.info(f"{'='*60}")
    sys.stdout.flush()

    if all_oot_q_preds:
        concat_q_preds = np.concatenate(all_oot_q_preds)
        concat_actuals = np.concatenate(all_oot_actuals)
        concat_meta = np.concatenate(all_oot_metas)

        # Save concat predictions
        np.savez(
            str(OUTPUT_DIR / "concat_oot_predictions.npz"),
            q_preds=concat_q_preds,
            actuals=concat_actuals,
            meta=concat_meta,
            taus=np.array(taus),
        )

        n_total = len(concat_actuals)
        log.info(f"Total OOT samples: {n_total:,}")
        log.info(f"Actual PnL: mean={np.mean(concat_actuals):.4f} std={np.std(concat_actuals):.4f}")

        concat_metrics = evaluate_quantile_model(
            concat_q_preds, concat_actuals, taus, fold_idx=-1
        )

        # ---- Calibration table ----
        log.info(f"\n--- Calibration (ideal: coverage = tau) ---")
        log.info(f"{'Tau':<10} {'Target':<10} {'Actual':<10} {'Error':<10}")
        log.info("-" * 40)
        for tau in taus:
            target = tau
            actual = concat_metrics.get(f"calib_q{int(tau*100)}", 0)
            err = concat_metrics.get(f"calib_err_q{int(tau*100)}", 0)
            log.info(f"{tau:<10.2f} {target:<10.2f} {actual:<10.4f} {err:<10.4f}")
        log.info(f"Mean calibration error: {concat_metrics['mean_calib_err']:.4f}")

        # ---- Predicted quantile means ----
        log.info(f"\n--- Predicted quantile means ---")
        for tau in taus:
            pm = concat_metrics.get(f"pred_mean_q{int(tau*100)}", 0)
            log.info(f"  q{int(tau*100)}: {pm:+.4f} ticks")

        # ---- Spearman ----
        log.info(f"\nSpearman(q50, actual): {concat_metrics['spearman_q50']:.4f} "
                 f"(p={concat_metrics['spearman_pval']:.2e})")

        # ---- High-confidence analysis ----
        log.info(f"\n--- High-confidence trade analysis ---")
        log.info(f"High-conf (q10 > {COMMISSION_TICKS:.3f} ticks): "
                 f"{concat_metrics['high_conf_rate']:.1%} of signals "
                 f"({concat_metrics['high_conf_count']:,})")
        log.info(f"High-conf edge (mean actual PnL): {concat_metrics['high_conf_edge']:+.4f} ticks")
        log.info(f"High-conf win rate: {concat_metrics['high_conf_wr']:.1%}")

        # ---- Threshold sweep ----
        log.info(f"\n--- Threshold sweep: q10 > threshold ---")
        log.info(f"{'Threshold':<12} {'Rate':<10} {'Count':<10} {'Edge':<12} {'WR':<10}")
        log.info("-" * 54)
        q10_idx = taus.index(0.10)
        for thresh in [-0.5, -0.25, 0.0, 0.25, COMMISSION_TICKS, 0.5, 0.75, 1.0]:
            mask = concat_q_preds[:, q10_idx] > thresh
            count = int(mask.sum())
            if count > 0:
                rate = float(mask.mean())
                edge = float(np.mean(concat_actuals[mask]))
                wr = float(np.mean(concat_actuals[mask] > 0))
                label = f"{thresh:.3f}" + (" *" if abs(thresh - COMMISSION_TICKS) < 0.01 else "")
                log.info(f"{label:<12} {rate:<10.3f} {count:<10d} {edge:<+12.4f} {wr:<10.3f}")

        # ---- Q50 bucket analysis ----
        log.info(f"\n--- Median (q50) prediction buckets ---")
        q50_idx = taus.index(0.50)
        q50_preds = concat_q_preds[:, q50_idx]
        percentiles = [0, 10, 25, 50, 75, 90, 100]
        boundaries = np.percentile(q50_preds, percentiles)
        log.info(f"{'Bucket':<20} {'Count':<10} {'Pred_mean':<12} {'Actual_mean':<12} {'WR':<10}")
        log.info("-" * 64)
        for k in range(len(boundaries) - 1):
            lo, hi = boundaries[k], boundaries[k + 1]
            if k == len(boundaries) - 2:
                mask = (q50_preds >= lo) & (q50_preds <= hi)
            else:
                mask = (q50_preds >= lo) & (q50_preds < hi)
            if mask.sum() > 0:
                pred_m = float(np.mean(q50_preds[mask]))
                act_m = float(np.mean(concat_actuals[mask]))
                wr = float(np.mean(concat_actuals[mask] > 0))
                log.info(f"p{percentiles[k]}-p{percentiles[k+1]:<14} {int(mask.sum()):<10d} "
                         f"{pred_m:<+12.4f} {act_m:<+12.4f} {wr:<10.3f}")

        # ---- Quantile crossings ----
        log.info(f"\nQuantile crossings: {concat_metrics['quantile_crossings']:,} "
                 f"({concat_metrics['crossing_rate']:.2%})")

        # ---- Moderate confidence ----
        if "mod_conf_rate" in concat_metrics:
            log.info(f"\nModerate-conf (q25 > 0): {concat_metrics['mod_conf_rate']:.1%}, "
                     f"edge={concat_metrics.get('mod_conf_edge', 0):+.4f}, "
                     f"WR={concat_metrics.get('mod_conf_wr', 0):.1%}")

        # ---- Per-fold summary table ----
        log.info(f"\n--- Per-fold metrics ---")
        log.info(f"{'Fold':<6} {'QLoss':<10} {'CalibErr':<10} {'Spearman':<10} "
                 f"{'HC_rate':<10} {'HC_edge':<10} {'HC_WR':<10} {'HC_count':<10} {'N_samples':<10}")
        log.info("-" * 86)
        pos_folds = 0
        for m in all_fold_metrics:
            is_pos = m.get("high_conf_edge", 0) > 0
            if is_pos:
                pos_folds += 1
            log.info(f"{m['fold']:<6d} {m.get('mean_qloss',0):<10.4f} "
                     f"{m.get('mean_calib_err',0):<10.4f} "
                     f"{m.get('spearman_q50',0):<10.4f} "
                     f"{m.get('high_conf_rate',0):<10.4f} "
                     f"{m.get('high_conf_edge',0):<+10.4f} "
                     f"{m.get('high_conf_wr',0):<10.4f} "
                     f"{m.get('high_conf_count',0):<10d} "
                     f"{m['n_val_samples']:<10d}")

        n_folds_actual = len(all_fold_metrics)
        log.info(f"\nPositive folds: {pos_folds}/{n_folds_actual} ({pos_folds/max(1,n_folds_actual):.0%})")

        # Mean across folds
        mean_ql = np.mean([m.get("mean_qloss", 0) for m in all_fold_metrics])
        mean_ce = np.mean([m.get("mean_calib_err", 0) for m in all_fold_metrics])
        mean_rho = np.mean([m.get("spearman_q50", 0) for m in all_fold_metrics])
        mean_hcr = np.mean([m.get("high_conf_rate", 0) for m in all_fold_metrics])
        mean_hce = np.mean([m.get("high_conf_edge", 0) for m in all_fold_metrics])
        mean_hcw = np.mean([m.get("high_conf_wr", 0) for m in all_fold_metrics])
        log.info(f"{'MEAN':<6} {mean_ql:<10.4f} {mean_ce:<10.4f} {mean_rho:<10.4f} "
                 f"{mean_hcr:<10.4f} {mean_hce:<+10.4f} {mean_hcw:<10.4f}")

        # MLflow aggregate
        try:
            import mlflow
            mlflow.log_metric("concat_mean_qloss", concat_metrics["mean_qloss"])
            mlflow.log_metric("concat_mean_calib_err", concat_metrics["mean_calib_err"])
            mlflow.log_metric("concat_spearman_q50", concat_metrics["spearman_q50"])
            mlflow.log_metric("concat_high_conf_rate", concat_metrics["high_conf_rate"])
            mlflow.log_metric("concat_high_conf_edge", concat_metrics["high_conf_edge"])
            mlflow.log_metric("concat_high_conf_wr", concat_metrics["high_conf_wr"])
            mlflow.log_metric("concat_crossing_rate", concat_metrics["crossing_rate"])
            mlflow.log_metric("mean_fold_qloss", mean_ql)
            mlflow.log_metric("mean_fold_calib_err", mean_ce)
            mlflow.log_metric("mean_fold_spearman", mean_rho)
            mlflow.log_metric("mean_fold_hc_rate", mean_hcr)
            mlflow.log_metric("mean_fold_hc_edge", mean_hce)
            mlflow.log_metric("total_oot_samples", n_total)
            mlflow.log_metric("total_folds", n_folds_actual)
            mlflow.log_metric("positive_folds", pos_folds)
            mlflow.log_metric("positive_fold_rate", pos_folds / max(1, n_folds_actual))

            for tau in taus:
                key = f"calib_q{int(tau*100)}"
                mlflow.log_metric(f"concat_{key}", concat_metrics[key])
        except Exception:
            pass

    # Save fold metrics JSON
    with open(str(OUTPUT_DIR / "fold_metrics.json"), "w") as f:
        json.dump(all_fold_metrics, f, indent=2, default=str)

    # End MLflow run
    try:
        import mlflow
        if mlflow_run:
            mlflow.end_run()
    except Exception:
        pass

    log.info(f"\nDone. Output saved.")
    sys.stdout.flush()
    return all_fold_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Quantile Execution Model v2 (Regime-adaptive)")
    parser.add_argument("--n-train-days", type=int, default=30)
    parser.add_argument("--n-eval-days", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128, 64])
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    log.info(f"=== Quantile Execution Model v2 (Regime-adaptive) ===")
    log.info(f"Device: {DEVICE}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log.info(f"Config: train={args.n_train_days}d eval={args.n_eval_days}d "
             f"epochs={args.epochs} bs={args.batch_size} lr={args.lr} wd={args.weight_decay}")
    log.info(f"Architecture: {args.hidden_dims}, dropout={args.dropout}")
    log.info(f"Features: {N_FEATURES} (42 original + 4 regime)")
    log.info(f"Regime features: realized_vol_5min, trend_indicator, hour_of_day, minute_bucket")
    log.info(f"Quantiles: {QUANTILE_TAUS}")
    log.info(f"Target: net_ticks_passive (5s horizon, cost={COMMISSION_TICKS:.3f} ticks)")
    log.info(f"Early stopping: patience=5 on validation Spearman(q50, actual)")
    log.info(f"Key v1 changes: 30d window (was 60d), regime features, Spearman early stop")
    sys.stdout.flush()

    log.info("\n--- Loading data ---")
    t0 = time.time()
    all_dates = load_all_dates()
    log.info(f"Data loading: {time.time() - t0:.1f}s")
    sys.stdout.flush()

    if not all_dates:
        log.error("No data loaded!")
        sys.exit(1)

    log.info("\n--- Walk-forward training ---")
    run_walk_forward(
        all_dates=all_dates,
        n_train_days=args.n_train_days,
        n_eval_days=args.n_eval_days,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        hidden_dims=args.hidden_dims,
        dropout=args.dropout,
        taus=QUANTILE_TAUS,
    )


if __name__ == "__main__":
    main()
