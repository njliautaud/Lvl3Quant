#!/usr/bin/env python3
"""
Supervised Execution MLP v2 — GPU-accelerated trade outcome predictor
======================================================================

Predicts whether a CNN-Mamba signal event will be profitable given
microstructure context. Dual-head: regression (P&L in ticks) +
classification (profitable yes/no).

Key differences from v1 (CPU-only, near-zero correlation):
  1. Runs on CUDA (RTX 3090)
  2. Richer features (42-dim, matching LGBM exec proven feature set)
  3. Deeper MLP with residual connections + BatchNorm
  4. Combined Huber + BCE loss
  5. Walk-forward sliding window: 60 train / 5 eval days
  6. Proper feature normalization per fold (train stats only)
  7. 135 dates of data (vs 10 in v1)
  8. Fully vectorized feature extraction (no Python loops per-event)

Data sources:
  - CNN-Mamba v2 predictions from multiple dirs (all_oot, bulk_oot, bulk_inference, fold)
  - MBO events: data/processed/mbo_events_smart_v3/*.npz
  - Stride=250, window_size=3000

Author: Claude (Infrastructure Builder)
Date: 2026-05-27
"""

from __future__ import annotations

import argparse
import gc
import json
import logging
import math
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
OUTPUT_DIR = LVL3_ROOT / "output" / "supervised_exec_v2"

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

# MBO event column indices
COL_PRICE_REL = 3
COL_SPREAD = 5
COL_SIDE = 2
COL_QTY_LOG = 4

# CNN-Mamba prediction parameters (from bulk inference)
PRED_STRIDE = 250   # events between predictions
PRED_WINDOW = 3000  # events per CNN window

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "supervised_exec_v2_gpu"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = LVL3_ROOT / "output" / "supervised_exec_v2.log"
os.makedirs(str(LOG_FILE.parent), exist_ok=True)
logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [SUP_EXEC_V2] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("sup_exec_v2")

# ---------------------------------------------------------------------------
# Feature names (42 features)
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
]
N_FEATURES = len(FEATURE_NAMES)  # 42

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


# ---------------------------------------------------------------------------
# Feature extraction (fully vectorized)
# ---------------------------------------------------------------------------

def extract_features_for_date(
    mbo_path: Path,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Extract 42 features + multi-horizon targets for one date.
    Fully vectorized - no Python loops over individual events.
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
        empty_targets = {
            f"{t}_{h}": np.zeros(0, dtype=np.float32)
            for h in HORIZONS for t in ["dir_move", "mfe", "mae", "pnl", "profitable"]
        }
        return np.zeros((0, N_FEATURES), dtype=np.float32), empty_targets, np.zeros((0, 3), dtype=np.float32)

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

    # ---- Assemble feature matrix ----
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
    ], axis=1).astype(np.float32)

    # ---- Targets ----
    l1 = labels[valid_indices, 0]
    l5 = labels[valid_indices, 1]
    l10 = labels[valid_indices, 2]

    targets = {}
    for h_name, lbl in zip(HORIZONS, [l1, l5, l10]):
        dir_move = lbl * sig_dir
        mfe_h = np.maximum(dir_move, 0.0)
        mae_h = np.maximum(-dir_move, 0.0)
        pnl_h = dir_move - COMMISSION_TICKS
        prof_h = (pnl_h > 0).astype(np.float32)

        targets[f"dir_move_{h_name}"] = np.clip(dir_move, -50, 50).astype(np.float32)
        targets[f"mfe_{h_name}"] = np.minimum(mfe_h, 50).astype(np.float32)
        targets[f"mae_{h_name}"] = np.minimum(mae_h, 50).astype(np.float32)
        targets[f"pnl_{h_name}"] = np.clip(pnl_h, -50, 50).astype(np.float32)
        targets[f"profitable_{h_name}"] = prof_h

    meta_out = np.stack([cur_ts, valid_indices.astype(np.float64), sig_dir.astype(np.float64)], axis=1).astype(np.float32)

    return features_out, targets, meta_out


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
    """Load features + targets for all dates with predictions + MBO data."""
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
    log.info(f"Total: {len(all_dates)} dates, {total_samples:,} samples")
    return all_dates


# ---------------------------------------------------------------------------
# Model: Dual-head MLP with residual connections
# ---------------------------------------------------------------------------

class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float = 0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.BatchNorm1d(dim),
        )
        self.act = nn.GELU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.act(x + self.dropout(self.net(x)))


class DualHeadMLP(nn.Module):
    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dims: List[int] = [256, 128, 64, 32],
        dropout: float = 0.3,
    ):
        super().__init__()

        layers = [
            nn.Linear(input_dim, hidden_dims[0]),
            nn.BatchNorm1d(hidden_dims[0]),
            nn.GELU(),
            nn.Dropout(dropout),
        ]

        for i in range(1, len(hidden_dims)):
            if hidden_dims[i] == hidden_dims[i - 1]:
                layers.append(ResidualBlock(hidden_dims[i], dropout))
            else:
                layers.extend([
                    nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])

        self.backbone = nn.Sequential(*layers)

        # Regression head: P&L per horizon (3 outputs)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )

        # Classification head: profitable per horizon (3 outputs)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.GELU(),
            nn.Linear(16, 3),
        )

    def forward(self, x):
        h = self.backbone(x)
        return self.reg_head(h), self.cls_head(h)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_one_fold(
    model: DualHeadMLP,
    train_X: np.ndarray,
    train_targets: Dict[str, np.ndarray],
    val_X: np.ndarray,
    val_targets: Dict[str, np.ndarray],
    fold_idx: int,
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[dict, dict, np.ndarray, np.ndarray]:

    # Normalize using TRAIN stats only
    train_mean = np.nanmean(train_X, axis=0)
    train_std = np.nanstd(train_X, axis=0)
    train_std[train_std < 1e-8] = 1.0

    X_train = np.nan_to_num((train_X - train_mean) / train_std, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num((val_X - train_mean) / train_std, nan=0.0, posinf=0.0, neginf=0.0)

    reg_train = np.stack([train_targets[f"pnl_{h}"] for h in HORIZONS], axis=1)
    reg_val = np.stack([val_targets[f"pnl_{h}"] for h in HORIZONS], axis=1)
    cls_train = np.stack([train_targets[f"profitable_{h}"] for h in HORIZONS], axis=1)
    cls_val = np.stack([val_targets[f"profitable_{h}"] for h in HORIZONS], axis=1)

    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    reg_t = torch.tensor(reg_train, dtype=torch.float32, device=DEVICE)
    cls_t = torch.tensor(cls_train, dtype=torch.float32, device=DEVICE)

    X_v = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    reg_v = torch.tensor(reg_val, dtype=torch.float32, device=DEVICE)
    cls_v = torch.tensor(cls_val, dtype=torch.float32, device=DEVICE)

    dataset = TensorDataset(X_t, reg_t, cls_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    huber_loss = nn.HuberLoss(delta=1.0)
    bce_loss = nn.BCEWithLogitsLoss()

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 7

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch_x, batch_reg, batch_cls in loader:
            optimizer.zero_grad()
            pred_reg, pred_cls = model(batch_x)

            loss_reg = huber_loss(pred_reg, batch_reg)
            loss_cls = bce_loss(pred_cls, batch_cls)
            loss = loss_reg + 0.5 * loss_cls

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        model.eval()
        with torch.no_grad():
            val_pred_reg, val_pred_cls = model(X_v)
            val_loss_reg = huber_loss(val_pred_reg, reg_v).item()
            val_loss_cls = bce_loss(val_pred_cls, cls_v).item()
            val_loss = val_loss_reg + 0.5 * val_loss_cls

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            log.info(f"  F{fold_idx} E{epoch+1}/{epochs}: "
                     f"train={np.mean(train_losses):.4f} val={val_loss:.4f} "
                     f"(reg={val_loss_reg:.4f} cls={val_loss_cls:.4f})")

        if patience_counter >= patience:
            log.info(f"  Early stop at epoch {epoch+1}")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    metrics = evaluate_model(model, X_v, reg_v, cls_v, val_targets, fold_idx)

    del X_t, reg_t, cls_t, dataset, loader
    gc.collect()
    torch.cuda.empty_cache()

    return metrics, best_state, train_mean, train_std


def evaluate_model(model, X_v, reg_v, cls_v, val_targets, fold_idx) -> dict:
    from scipy.stats import spearmanr
    try:
        from sklearn.metrics import roc_auc_score
    except ImportError:
        roc_auc_score = None

    with torch.no_grad():
        pred_reg, pred_cls = model(X_v)

    pred_reg_np = pred_reg.cpu().numpy()
    pred_cls_np = torch.sigmoid(pred_cls).cpu().numpy()
    actual_reg = reg_v.cpu().numpy()
    actual_cls = cls_v.cpu().numpy()

    metrics = {"fold": fold_idx, "n_val_samples": int(len(X_v))}

    for h_idx, h_name in enumerate(HORIZONS):
        pred_pnl = pred_reg_np[:, h_idx]
        actual_pnl = actual_reg[:, h_idx]
        pred_prob = pred_cls_np[:, h_idx]
        actual_prof = actual_cls[:, h_idx]

        mask = np.isfinite(pred_pnl) & np.isfinite(actual_pnl)
        if mask.sum() > 100:
            corr, pval = spearmanr(pred_pnl[mask], actual_pnl[mask])
            metrics[f"spearman_pnl_{h_name}"] = round(float(corr), 6)
        else:
            metrics[f"spearman_pnl_{h_name}"] = 0.0

        if roc_auc_score and len(np.unique(actual_prof)) > 1:
            metrics[f"auc_{h_name}"] = round(float(roc_auc_score(actual_prof, pred_prob)), 6)
        else:
            metrics[f"auc_{h_name}"] = 0.5

        n_total = len(pred_prob)
        for pct, pct_name in [(0.10, "top10"), (0.20, "top20")]:
            n_top = max(1, int(n_total * pct))
            top_idx = np.argsort(pred_prob)[-n_top:]
            metrics[f"precision_{pct_name}_{h_name}"] = round(float(np.mean(actual_prof[top_idx])), 4)
            metrics[f"net_ticks_{pct_name}_{h_name}"] = round(float(np.mean(actual_pnl[top_idx])), 4)

        take_mask = pred_prob > 0.5
        if take_mask.sum() > 10:
            metrics[f"precision_take_{h_name}"] = round(float(np.mean(actual_prof[take_mask])), 4)
            metrics[f"net_ticks_take_{h_name}"] = round(float(np.mean(actual_pnl[take_mask])), 4)
            metrics[f"n_trades_take_{h_name}"] = int(take_mask.sum())

        n_bottom = max(1, int(n_total * 0.10))
        bottom_idx = np.argsort(pred_prob)[:n_bottom]
        metrics[f"net_ticks_filtered_{h_name}"] = round(float(np.mean(actual_pnl[bottom_idx])), 4)

    return metrics


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def run_walk_forward(
    all_dates: List[dict],
    n_train_days: int = 60,
    n_eval_days: int = 5,
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    hidden_dims: List[int] = [256, 128, 64, 32],
    dropout: float = 0.3,
):
    n_dates = len(all_dates)
    if n_dates < n_train_days + n_eval_days:
        log.warning(f"Only {n_dates} dates, reducing train from {n_train_days}")
        n_train_days = max(5, n_dates - n_eval_days)

    n_folds = max(1, (n_dates - n_train_days) // n_eval_days)
    log.info(f"Walk-forward: {n_dates} dates, {n_train_days}d train + {n_eval_days}d eval = {n_folds} folds")

    # MLflow
    mlflow_run = None
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow_run = mlflow.start_run(
            run_name=f"sup_exec_v2_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        mlflow.log_params({
            "n_train_days": n_train_days,
            "n_eval_days": n_eval_days,
            "epochs": epochs,
            "batch_size": batch_size,
            "lr": lr,
            "hidden_dims": str(hidden_dims),
            "dropout": dropout,
            "n_features": N_FEATURES,
            "device": str(DEVICE),
            "n_total_dates": n_dates,
            "n_folds": n_folds,
            "commission_ticks": COMMISSION_TICKS,
            "pred_stride": PRED_STRIDE,
            "pred_window": PRED_WINDOW,
        })
        log.info(f"MLflow run: {mlflow_run.info.run_id}")
    except Exception as e:
        log.warning(f"MLflow: {e}")

    all_fold_metrics = []
    all_oot_preds = []
    all_oot_actuals = []
    all_oot_probs = []
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

        train_X = np.concatenate([d["features"] for d in train_dates])
        train_targets = {
            k: np.concatenate([d["targets"][k] for d in train_dates])
            for k in train_dates[0]["targets"]
        }

        eval_X = np.concatenate([d["features"] for d in eval_dates])
        eval_targets = {
            k: np.concatenate([d["targets"][k] for d in eval_dates])
            for k in eval_dates[0]["targets"]
        }
        eval_meta = np.concatenate([d["meta"] for d in eval_dates])

        log.info(f"  Train: {len(train_X):,}  Eval: {len(eval_X):,}")

        model = DualHeadMLP(input_dim=N_FEATURES, hidden_dims=hidden_dims, dropout=dropout)

        fold_metrics, best_state, train_mean, train_std = train_one_fold(
            model, train_X, train_targets, eval_X, eval_targets,
            fold_idx, epochs, batch_size, lr,
        )
        all_fold_metrics.append(fold_metrics)

        # Save fold artifacts
        fold_dir = OUTPUT_DIR / f"fold_{fold_idx:02d}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save(best_state, str(fold_dir / "model.pt"))
        np.savez(str(fold_dir / "norm_stats.npz"), mean=train_mean, std=train_std)

        # OOT predictions
        model.load_state_dict(best_state)
        model = model.to(DEVICE)
        model.eval()

        eval_X_norm = np.nan_to_num((eval_X - train_mean) / train_std, nan=0.0, posinf=0.0, neginf=0.0)
        with torch.no_grad():
            X_v = torch.tensor(eval_X_norm, dtype=torch.float32, device=DEVICE)
            pred_reg, pred_cls = model(X_v)

        pred_reg_np = pred_reg.cpu().numpy()
        pred_prob_np = torch.sigmoid(pred_cls).cpu().numpy()
        actual_pnl = np.stack([eval_targets[f"pnl_{h}"] for h in HORIZONS], axis=1)
        actual_prof = np.stack([eval_targets[f"profitable_{h}"] for h in HORIZONS], axis=1)

        np.savez(
            str(fold_dir / "oot_predictions.npz"),
            pred_pnl=pred_reg_np,
            pred_prob=pred_prob_np,
            actual_pnl=actual_pnl,
            actual_profitable=actual_prof,
            features=eval_X,
            meta=eval_meta,
            train_dates=np.array([d["date"] for d in train_dates]),
            eval_dates=np.array([d["date"] for d in eval_dates]),
        )

        all_oot_preds.append(pred_reg_np)
        all_oot_probs.append(pred_prob_np)
        all_oot_actuals.append(actual_pnl)
        all_oot_metas.append(eval_meta)

        # MLflow per-fold
        try:
            import mlflow
            for k, v in fold_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"fold_{k}", v, step=fold_idx)
        except Exception:
            pass

        for h in HORIZONS:
            sp = fold_metrics.get(f"spearman_pnl_{h}", 0)
            auc = fold_metrics.get(f"auc_{h}", 0)
            p10 = fold_metrics.get(f"precision_top10_{h}", 0)
            nt10 = fold_metrics.get(f"net_ticks_top10_{h}", 0)
            log.info(f"  {h}: Sp={sp:.4f} AUC={auc:.4f} P@10%={p10:.3f} NT@10%={nt10:+.3f}")

        del model, X_v, pred_reg, pred_cls
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Aggregate
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}")
    log.info("AGGREGATE RESULTS")
    log.info(f"{'='*60}")

    if all_oot_preds:
        concat_preds = np.concatenate(all_oot_preds)
        concat_probs = np.concatenate(all_oot_probs)
        concat_actuals = np.concatenate(all_oot_actuals)
        concat_meta = np.concatenate(all_oot_metas)

        np.savez(
            str(OUTPUT_DIR / "concat_oot_predictions.npz"),
            pred_pnl=concat_preds,
            pred_prob=concat_probs,
            actual_pnl=concat_actuals,
            meta=concat_meta,
        )

        from scipy.stats import spearmanr

        for h_idx, h_name in enumerate(HORIZONS):
            pred_col = concat_preds[:, h_idx]
            prob_col = concat_probs[:, h_idx]
            actual_col = concat_actuals[:, h_idx]
            mask = np.isfinite(pred_col) & np.isfinite(actual_col)

            corr, pval = spearmanr(pred_col[mask], actual_col[mask])
            log.info(f"\n  {h_name} concat Spearman: {corr:.4f} (p={pval:.2e})")

            for pct in [0.10, 0.20, 0.30, 0.50]:
                n_top = max(1, int(mask.sum() * pct))
                top_idx = np.argsort(prob_col[mask])[-n_top:]
                mean_pnl = float(np.mean(actual_col[mask][top_idx]))
                n_prof = int(np.sum(actual_col[mask][top_idx] > 0))
                wr = n_prof / n_top if n_top > 0 else 0
                log.info(f"    Top {pct*100:.0f}%: {n_top:,} trades, "
                         f"mean P&L={mean_pnl:+.4f} ticks, WR={wr:.1%}")

            n_bottom = max(1, int(mask.sum() * 0.10))
            bottom_idx = np.argsort(prob_col[mask])[:n_bottom]
            mean_pnl_bot = float(np.mean(actual_col[mask][bottom_idx]))
            log.info(f"    Bottom 10% (filtered): mean P&L={mean_pnl_bot:+.4f} ticks")

        # MLflow aggregate
        try:
            import mlflow
            for h_idx, h_name in enumerate(HORIZONS):
                pred_col = concat_preds[:, h_idx]
                actual_col = concat_actuals[:, h_idx]
                mask = np.isfinite(pred_col) & np.isfinite(actual_col)
                corr, _ = spearmanr(pred_col[mask], actual_col[mask])
                mlflow.log_metric(f"concat_spearman_{h_name}", corr)

            for key in all_fold_metrics[0]:
                if isinstance(all_fold_metrics[0][key], (int, float)):
                    vals = [m[key] for m in all_fold_metrics if key in m]
                    mlflow.log_metric(f"mean_{key}", float(np.mean(vals)))

            mlflow.log_metric("total_oot_samples", int(len(concat_preds)))
            mlflow.log_metric("total_folds", len(all_fold_metrics))
        except Exception:
            pass

    with open(str(OUTPUT_DIR / "fold_metrics.json"), "w") as f:
        json.dump(all_fold_metrics, f, indent=2, default=str)

    try:
        import mlflow
        if mlflow_run:
            mlflow.end_run()
    except Exception:
        pass

    log.info(f"\nDone. Output: {OUTPUT_DIR}")
    return all_fold_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train-days", type=int, default=60)
    parser.add_argument("--n-eval-days", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128, 64, 32])
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    log.info(f"Device: {DEVICE}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log.info(f"Config: train={args.n_train_days}d eval={args.n_eval_days}d "
             f"epochs={args.epochs} bs={args.batch_size} lr={args.lr}")
    log.info(f"Architecture: {args.hidden_dims}, dropout={args.dropout}")
    log.info(f"Features: {N_FEATURES}, Horizons: {HORIZONS}")
    log.info(f"Pred stride: {PRED_STRIDE}, Pred window: {PRED_WINDOW}")

    log.info("\n--- Loading data ---")
    t0 = time.time()
    all_dates = load_all_dates()
    log.info(f"Data loading: {time.time() - t0:.1f}s")

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
    )


if __name__ == "__main__":
    main()
