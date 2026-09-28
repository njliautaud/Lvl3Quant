#!/usr/bin/env python3
"""
Adverse Selection Predictor v1 — Binary classification for trade entry quality
=============================================================================

Predicts whether a trade entry at the current moment will experience adverse
price movement within 5 seconds. Binary classification:
  1 = adverse movement >= 1 tick against trade direction within 5s (MAE >= 1 tick)
  0 = no significant adverse movement

Architecture: MLP with residual connections (256→128→64→32→1)
Loss: Binary cross-entropy with class weighting (adverse ~40% of data)
Walk-forward: SLIDING window, 60d train / 5d eval (mandatory per DIRECTIVES)

Data sources (same as supervised_exec_v2.py):
  - CNN-Mamba v2 predictions from multiple dirs
  - MBO events: data/processed/mbo_events_smart_v3/*.npz
  - 42 microstructure features

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
OUTPUT_DIR = LVL3_ROOT / "output" / "adverse_selection_v1"

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

# Adverse selection threshold
ADVERSE_THRESHOLD_TICKS = 1.0  # MAE >= 1 tick = adverse

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
EXPERIMENT_NAME = "adverse_selection_v1"

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = LVL3_ROOT / "output" / "adverse_selection_v1.log"
os.makedirs(str(LOG_FILE.parent), exist_ok=True)
logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [ADV_SEL_V1] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("adv_sel_v1")

# ---------------------------------------------------------------------------
# Feature names (42 features — same as supervised_exec_v2)
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
# Vectorized rolling helpers (from supervised_exec_v2)
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
# Feature extraction (fully vectorized — from supervised_exec_v2)
# ---------------------------------------------------------------------------

def extract_features_for_date(
    mbo_path: Path,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Extract 42 features + adverse selection targets for one date.
    Fully vectorized - no Python loops over individual events.

    Returns:
        features: (N, 42) float32
        targets: dict with 'adverse_5s' binary target + supporting data
        meta: (N, 3) [timestamp, pred_index, signal_direction]
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
            "adverse_5s": np.zeros(0, dtype=np.float32),
            "mae_5s": np.zeros(0, dtype=np.float32),
            "mfe_5s": np.zeros(0, dtype=np.float32),
            "dir_move_5s": np.zeros(0, dtype=np.float32),
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

    # ---- Adverse selection target (5s horizon) ----
    # Labels contain directional moves at 1s, 5s, 10s horizons
    # dir_move = label * signal_direction (positive = favorable, negative = adverse)
    # MAE = max adverse excursion = max(0, -dir_move)
    # Adverse = MAE >= 1 tick
    l5 = labels[valid_indices, 1]  # 5s horizon raw label
    dir_move_5s = (l5 * sig_dir).astype(np.float32)
    mae_5s = np.maximum(-dir_move_5s, 0.0).astype(np.float32)
    mfe_5s = np.maximum(dir_move_5s, 0.0).astype(np.float32)
    adverse_5s = (mae_5s >= ADVERSE_THRESHOLD_TICKS).astype(np.float32)

    targets = {
        "adverse_5s": adverse_5s,
        "mae_5s": mae_5s,
        "mfe_5s": mfe_5s,
        "dir_move_5s": dir_move_5s,
    }

    meta_out = np.stack([
        cur_ts,
        valid_indices.astype(np.float64),
        sig_dir.astype(np.float64),
    ], axis=1).astype(np.float32)

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
    """Load features + adverse selection targets for all dates."""
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
    adverse_count = sum(d["targets"]["adverse_5s"].sum() for d in all_dates)
    adverse_rate = adverse_count / total_samples if total_samples > 0 else 0
    log.info(f"Total: {len(all_dates)} dates, {total_samples:,} samples")
    log.info(f"Adverse rate: {adverse_rate:.1%} ({int(adverse_count):,} / {total_samples:,})")
    return all_dates


# ---------------------------------------------------------------------------
# Model: Binary classification MLP with residual connections
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


class AdverseSelectionMLP(nn.Module):
    """
    Binary classifier: predicts P(adverse selection) for a trade entry.
    Architecture: 42 → 256 → 128 → 64 → 32 → 1
    With residual connections where dimensions match.
    """
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
                # Same dimension: use residual block
                layers.append(ResidualBlock(hidden_dims[i], dropout))
            else:
                # Dimension change: linear projection
                layers.extend([
                    nn.Linear(hidden_dims[i - 1], hidden_dims[i]),
                    nn.BatchNorm1d(hidden_dims[i]),
                    nn.GELU(),
                    nn.Dropout(dropout),
                ])

        self.backbone = nn.Sequential(*layers)

        # Single binary output (logit)
        self.head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        h = self.backbone(x)
        return self.head(h).squeeze(-1)  # (batch,) logits


# ---------------------------------------------------------------------------
# Training one fold
# ---------------------------------------------------------------------------

def train_one_fold(
    model: AdverseSelectionMLP,
    train_X: np.ndarray,
    train_y: np.ndarray,
    val_X: np.ndarray,
    val_y: np.ndarray,
    fold_idx: int,
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    weight_decay: float = 1e-4,
) -> Tuple[dict, dict, np.ndarray, np.ndarray]:
    """
    Train one fold of the adverse selection model.

    Returns:
        metrics: dict of evaluation metrics
        best_state: model state dict
        train_mean: feature means (for normalization)
        train_std: feature stds (for normalization)
    """
    # Normalize using TRAIN stats only
    train_mean = np.nanmean(train_X, axis=0)
    train_std = np.nanstd(train_X, axis=0)
    train_std[train_std < 1e-8] = 1.0

    X_train = np.nan_to_num((train_X - train_mean) / train_std, nan=0.0, posinf=0.0, neginf=0.0)
    X_val = np.nan_to_num((val_X - train_mean) / train_std, nan=0.0, posinf=0.0, neginf=0.0)

    # Class weighting for imbalanced data
    n_adverse = train_y.sum()
    n_safe = len(train_y) - n_adverse
    if n_adverse > 0 and n_safe > 0:
        # pos_weight = n_negative / n_positive (upweights the minority class)
        pos_weight = torch.tensor([n_safe / n_adverse], dtype=torch.float32, device=DEVICE)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float32, device=DEVICE)

    log.info(f"  Class balance: {n_adverse:.0f} adverse ({n_adverse/len(train_y):.1%}), "
             f"{n_safe:.0f} safe ({n_safe/len(train_y):.1%}), pos_weight={pos_weight.item():.3f}")

    X_t = torch.tensor(X_train, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(train_y, dtype=torch.float32, device=DEVICE)

    X_v = torch.tensor(X_val, dtype=torch.float32, device=DEVICE)
    y_v = torch.tensor(val_y, dtype=torch.float32, device=DEVICE)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=False)

    model = model.to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    bce_loss = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0
    patience = 7

    for epoch in range(epochs):
        model.train()
        train_losses = []

        for batch_x, batch_y in loader:
            optimizer.zero_grad()
            logits = model(batch_x)
            loss = bce_loss(logits, batch_y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_logits = model(X_v)
            val_loss = bce_loss(val_logits, y_v).item()

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or epoch == 0:
            val_probs = torch.sigmoid(val_logits).cpu().numpy()
            val_preds = (val_probs > 0.5).astype(float)
            acc = float(np.mean(val_preds == val_y))
            log.info(f"  F{fold_idx} E{epoch+1}/{epochs}: "
                     f"train_loss={np.mean(train_losses):.4f} val_loss={val_loss:.4f} "
                     f"val_acc={acc:.3f}")

        if patience_counter >= patience:
            log.info(f"  Early stop at epoch {epoch+1}")
            break

    # Restore best model
    if best_state is not None:
        model.load_state_dict(best_state)

    model.eval()
    metrics = evaluate_model(model, X_v, val_y, fold_idx)

    del X_t, y_t, dataset, loader
    gc.collect()
    torch.cuda.empty_cache()

    return metrics, best_state, train_mean, train_std


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(model, X_v, val_y: np.ndarray, fold_idx: int) -> dict:
    """Compute AUC, precision, recall, F1 for one fold."""
    try:
        from sklearn.metrics import (
            roc_auc_score, precision_score, recall_score, f1_score,
            precision_recall_curve, average_precision_score,
        )
    except ImportError:
        log.warning("sklearn not available, metrics will be limited")
        roc_auc_score = None

    if isinstance(X_v, np.ndarray):
        X_v_t = torch.tensor(X_v, dtype=torch.float32, device=DEVICE)
    else:
        X_v_t = X_v

    with torch.no_grad():
        logits = model(X_v_t)
        probs = torch.sigmoid(logits).cpu().numpy()

    if isinstance(val_y, torch.Tensor):
        y_true = val_y.cpu().numpy()
    else:
        y_true = val_y

    preds_05 = (probs > 0.5).astype(float)

    metrics = {
        "fold": fold_idx,
        "n_val_samples": int(len(y_true)),
        "adverse_rate": float(np.mean(y_true)),
        "pred_adverse_rate": float(np.mean(preds_05)),
    }

    if roc_auc_score is not None and len(np.unique(y_true)) > 1:
        metrics["auc"] = round(float(roc_auc_score(y_true, probs)), 6)
        metrics["avg_precision"] = round(float(average_precision_score(y_true, probs)), 6)
        metrics["precision"] = round(float(precision_score(y_true, preds_05, zero_division=0)), 4)
        metrics["recall"] = round(float(recall_score(y_true, preds_05, zero_division=0)), 4)
        metrics["f1"] = round(float(f1_score(y_true, preds_05, zero_division=0)), 4)

        # Metrics at different thresholds
        for thresh in [0.3, 0.4, 0.5, 0.6, 0.7]:
            preds_t = (probs > thresh).astype(float)
            n_flagged = int(preds_t.sum())
            if n_flagged > 0:
                prec = float(precision_score(y_true, preds_t, zero_division=0))
                rec = float(recall_score(y_true, preds_t, zero_division=0))
                f1 = float(f1_score(y_true, preds_t, zero_division=0))
                metrics[f"precision_t{thresh:.1f}"] = round(prec, 4)
                metrics[f"recall_t{thresh:.1f}"] = round(rec, 4)
                metrics[f"f1_t{thresh:.1f}"] = round(f1, 4)
                metrics[f"n_flagged_t{thresh:.1f}"] = n_flagged
    else:
        metrics["auc"] = 0.5
        metrics["avg_precision"] = float(np.mean(y_true))
        metrics["precision"] = 0.0
        metrics["recall"] = 0.0
        metrics["f1"] = 0.0

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

    # MLflow setup
    mlflow_run = None
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow_run = mlflow.start_run(
            run_name=f"adv_sel_v1_{time.strftime('%Y%m%d_%H%M%S')}"
        )
        mlflow.log_params({
            "model_type": "adverse_selection_binary",
            "target": "MAE_5s >= 1 tick",
            "adverse_threshold_ticks": ADVERSE_THRESHOLD_TICKS,
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
            "loss": "BCEWithLogits + class_weight",
            "optimizer": "AdamW",
            "scheduler": "CosineAnnealingLR",
        })
        log.info(f"MLflow run: {mlflow_run.info.run_id}")
    except Exception as e:
        log.warning(f"MLflow: {e}")

    all_fold_metrics = []
    all_oot_probs = []
    all_oot_labels = []
    all_oot_mae = []
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

        # Assemble train data
        train_X = np.concatenate([d["features"] for d in train_dates])
        train_y = np.concatenate([d["targets"]["adverse_5s"] for d in train_dates])

        # Assemble eval data
        eval_X = np.concatenate([d["features"] for d in eval_dates])
        eval_y = np.concatenate([d["targets"]["adverse_5s"] for d in eval_dates])
        eval_mae = np.concatenate([d["targets"]["mae_5s"] for d in eval_dates])
        eval_meta = np.concatenate([d["meta"] for d in eval_dates])

        log.info(f"  Train: {len(train_X):,} (adverse: {train_y.mean():.1%})  "
                 f"Eval: {len(eval_X):,} (adverse: {eval_y.mean():.1%})")

        # Create model
        model = AdverseSelectionMLP(
            input_dim=N_FEATURES, hidden_dims=hidden_dims, dropout=dropout
        )

        # Train
        fold_metrics, best_state, train_mean, train_std = train_one_fold(
            model, train_X, train_y, eval_X, eval_y,
            fold_idx, epochs, batch_size, lr,
        )
        all_fold_metrics.append(fold_metrics)

        # Save fold artifacts
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
            logits = model(X_v)
            probs = torch.sigmoid(logits).cpu().numpy()

        # Save fold predictions as .npz
        np.savez(
            str(fold_dir / "oot_predictions.npz"),
            probs=probs,
            labels=eval_y,
            mae=eval_mae,
            features=eval_X,
            meta=eval_meta,
            train_dates=np.array([d["date"] for d in train_dates]),
            eval_dates=np.array([d["date"] for d in eval_dates]),
        )

        all_oot_probs.append(probs)
        all_oot_labels.append(eval_y)
        all_oot_mae.append(eval_mae)
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
        auc = fold_metrics.get("auc", 0)
        prec = fold_metrics.get("precision", 0)
        rec = fold_metrics.get("recall", 0)
        f1 = fold_metrics.get("f1", 0)
        log.info(f"  FOLD {fold_idx}: AUC={auc:.4f} P={prec:.3f} R={rec:.3f} F1={f1:.3f}")

        del model, X_v, logits
        gc.collect()
        torch.cuda.empty_cache()

    # -----------------------------------------------------------------------
    # Aggregate concat metrics across all folds
    # -----------------------------------------------------------------------
    log.info(f"\n{'='*60}")
    log.info("AGGREGATE CONCAT RESULTS (all OOT folds)")
    log.info(f"{'='*60}")

    if all_oot_probs:
        concat_probs = np.concatenate(all_oot_probs)
        concat_labels = np.concatenate(all_oot_labels)
        concat_mae = np.concatenate(all_oot_mae)
        concat_meta = np.concatenate(all_oot_metas)

        # Save concat predictions
        np.savez(
            str(OUTPUT_DIR / "concat_oot_predictions.npz"),
            probs=concat_probs,
            labels=concat_labels,
            mae=concat_mae,
            meta=concat_meta,
        )

        n_total = len(concat_labels)
        n_adverse = int(concat_labels.sum())
        log.info(f"Total OOT samples: {n_total:,}")
        log.info(f"Adverse rate: {n_adverse/n_total:.1%} ({n_adverse:,} / {n_total:,})")

        # Concat AUC, precision, recall, F1
        try:
            from sklearn.metrics import (
                roc_auc_score, precision_score, recall_score, f1_score,
                average_precision_score, classification_report,
            )

            concat_auc = roc_auc_score(concat_labels, concat_probs)
            concat_ap = average_precision_score(concat_labels, concat_probs)
            log.info(f"\nConcat AUC: {concat_auc:.4f}")
            log.info(f"Concat Avg Precision: {concat_ap:.4f}")

            # Metrics at different thresholds
            log.info(f"\n{'Threshold':<12} {'Precision':<12} {'Recall':<12} {'F1':<12} {'N_flagged':<12} {'Adverse_WR':<12}")
            log.info("-" * 72)
            for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                preds_t = (concat_probs > thresh).astype(float)
                n_flagged = int(preds_t.sum())
                if n_flagged > 0:
                    prec = precision_score(concat_labels, preds_t, zero_division=0)
                    rec = recall_score(concat_labels, preds_t, zero_division=0)
                    f1 = f1_score(concat_labels, preds_t, zero_division=0)
                    # What fraction of flagged trades were truly adverse
                    adv_wr = float(concat_labels[concat_probs > thresh].mean())
                    log.info(f"{thresh:<12.1f} {prec:<12.4f} {rec:<12.4f} {f1:<12.4f} {n_flagged:<12d} {adv_wr:<12.4f}")

            # Practical analysis: what happens if we SKIP trades the model flags?
            log.info(f"\n--- Practical impact: skip trades flagged as adverse ---")
            log.info(f"{'Threshold':<12} {'Skipped':<12} {'Skip_rate':<12} {'Adverse_avoided':<16} {'Safe_lost':<12}")
            log.info("-" * 76)
            for thresh in [0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
                flagged = concat_probs > thresh
                n_flagged = int(flagged.sum())
                if n_flagged > 0:
                    skip_rate = n_flagged / n_total
                    adverse_in_flagged = int(concat_labels[flagged].sum())
                    safe_in_flagged = n_flagged - adverse_in_flagged
                    log.info(f"{thresh:<12.1f} {n_flagged:<12d} {skip_rate:<12.1%} "
                             f"{adverse_in_flagged:<16d} {safe_in_flagged:<12d}")

            # Mean MAE for flagged vs not-flagged
            log.info(f"\n--- MAE analysis by prediction ---")
            for thresh in [0.4, 0.5, 0.6]:
                flagged = concat_probs > thresh
                not_flagged = ~flagged
                if flagged.sum() > 0 and not_flagged.sum() > 0:
                    mae_flagged = float(concat_mae[flagged].mean())
                    mae_safe = float(concat_mae[not_flagged].mean())
                    log.info(f"  t={thresh:.1f}: MAE_flagged={mae_flagged:.3f} ticks, "
                             f"MAE_kept={mae_safe:.3f} ticks (delta={mae_flagged-mae_safe:+.3f})")

            # Per-fold summary table
            log.info(f"\n--- Per-fold metrics ---")
            log.info(f"{'Fold':<8} {'AUC':<10} {'Prec':<10} {'Recall':<10} {'F1':<10} {'N_samples':<12} {'Adv_rate':<10}")
            log.info("-" * 70)
            for m in all_fold_metrics:
                log.info(f"{m['fold']:<8d} {m.get('auc',0):<10.4f} {m.get('precision',0):<10.4f} "
                         f"{m.get('recall',0):<10.4f} {m.get('f1',0):<10.4f} "
                         f"{m['n_val_samples']:<12d} {m.get('adverse_rate',0):<10.3f}")

            # Mean across folds
            mean_auc = np.mean([m.get("auc", 0) for m in all_fold_metrics])
            mean_prec = np.mean([m.get("precision", 0) for m in all_fold_metrics])
            mean_rec = np.mean([m.get("recall", 0) for m in all_fold_metrics])
            mean_f1 = np.mean([m.get("f1", 0) for m in all_fold_metrics])
            log.info(f"{'MEAN':<8s} {mean_auc:<10.4f} {mean_prec:<10.4f} {mean_rec:<10.4f} {mean_f1:<10.4f}")

            # MLflow aggregate
            try:
                import mlflow
                mlflow.log_metric("concat_auc", concat_auc)
                mlflow.log_metric("concat_avg_precision", concat_ap)
                mlflow.log_metric("mean_fold_auc", mean_auc)
                mlflow.log_metric("mean_fold_precision", mean_prec)
                mlflow.log_metric("mean_fold_recall", mean_rec)
                mlflow.log_metric("mean_fold_f1", mean_f1)
                mlflow.log_metric("total_oot_samples", n_total)
                mlflow.log_metric("total_folds", len(all_fold_metrics))
                mlflow.log_metric("adverse_rate", n_adverse / n_total)

                # Log threshold-specific metrics
                for thresh in [0.4, 0.5, 0.6]:
                    preds_t = (concat_probs > thresh).astype(float)
                    n_flagged = int(preds_t.sum())
                    if n_flagged > 0:
                        prec = precision_score(concat_labels, preds_t, zero_division=0)
                        rec = recall_score(concat_labels, preds_t, zero_division=0)
                        f1 = f1_score(concat_labels, preds_t, zero_division=0)
                        mlflow.log_metric(f"concat_precision_t{thresh:.1f}", prec)
                        mlflow.log_metric(f"concat_recall_t{thresh:.1f}", rec)
                        mlflow.log_metric(f"concat_f1_t{thresh:.1f}", f1)
            except Exception:
                pass

        except ImportError:
            log.warning("sklearn not available for concat metrics")

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

    log.info(f"\nDone. Output: {OUTPUT_DIR}")
    return all_fold_metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Adverse Selection Predictor v1")
    parser.add_argument("--n-train-days", type=int, default=60)
    parser.add_argument("--n-eval-days", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--hidden-dims", nargs="+", type=int, default=[256, 128, 64, 32])
    parser.add_argument("--dropout", type=float, default=0.3)
    parser.add_argument("--adverse-threshold", type=float, default=1.0,
                        help="MAE threshold in ticks to classify as adverse (default: 1.0)")
    args = parser.parse_args()

    global ADVERSE_THRESHOLD_TICKS
    ADVERSE_THRESHOLD_TICKS = args.adverse_threshold

    log.info(f"Device: {DEVICE}")
    log.info(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
        log.info(f"GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    log.info(f"Config: train={args.n_train_days}d eval={args.n_eval_days}d "
             f"epochs={args.epochs} bs={args.batch_size} lr={args.lr} wd={args.weight_decay}")
    log.info(f"Architecture: {args.hidden_dims}, dropout={args.dropout}")
    log.info(f"Features: {N_FEATURES}, Target: adverse_5s (MAE >= {ADVERSE_THRESHOLD_TICKS} tick)")
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
