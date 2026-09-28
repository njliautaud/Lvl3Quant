#!/usr/bin/env python3
"""
Dynamic Hold Optimizer v1 — MLP predicting optimal hold duration per trade
==========================================================================

Instead of a fixed hold for all trades, this learns WHEN to hold longer vs
exit quickly, conditioned on entry microstructure state.

Classification target: which hold bucket {1s, 5s, 10s, 30s} maximizes net PnL
Regression target: PnL at the optimal hold duration

Architecture: MLP 256 → 128 → 64 → 4 (softmax) + 1 (regression)
Training: SLIDING walk-forward (20 train, 5 eval, drop oldest)

Data sources:
  - CNN-Mamba v2 OOT predictions: output/cnn_mamba_v2_bulk_oot/
  - MBO events (with labels_1s/5s/10s/30s): data/processed/mbo_events_smart_v3/

Author: Claude (Infrastructure Builder)
Date: 2026-05-27
"""

from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
import traceback
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
PRED_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_oot"
OUTPUT_DIR = LVL3_ROOT / "output" / "dynamic_hold_v1"

ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE  # 0.376

# MBO event column indices (from supervised_exec_v2)
COL_PRICE_REL = 3
COL_SPREAD = 5
COL_SIDE = 2
COL_QTY_LOG = 4

PRED_STRIDE = 250
PRED_WINDOW = 3000

HOLD_HORIZONS = ["1s", "5s", "10s", "30s"]
PRED_HORIZONS = ["1s", "5s", "10s"]
N_HOLD_CLASSES = 4

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "dynamic_hold_v1"

# Walk-forward config
WF_TRAIN_DAYS = 20
WF_EVAL_DAYS = 5
WF_SLIDE = 5  # slide by eval window size

# Training config
EPOCHS = 40
BATCH_SIZE = 4096
LR = 1e-3
WEIGHT_DECAY = 1e-4
DROPOUT = 0.3
CLS_WEIGHT = 1.0
REG_WEIGHT = 0.1

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
os.makedirs(str(OUTPUT_DIR), exist_ok=True)
LOG_FILE = OUTPUT_DIR / "training.log"
logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [DYN_HOLD_V1] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("dyn_hold_v1")

# ---------------------------------------------------------------------------
# Feature names (42 features — same proven set as supervised_exec_v2)
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


# ---------------------------------------------------------------------------
# Rolling helpers (vectorized)
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
# Feature extraction
# ---------------------------------------------------------------------------

def extract_features_for_date(
    mbo_path: Path,
    predictions: np.ndarray,
    labels_by_horizon: Dict[str, np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Extract 42 features + hold optimization targets for one date.

    Returns:
        features: (N, 42) float32
        hold_class: (N,) int64 — which horizon was best {0,1,2,3}
        optimal_pnl: (N,) float32 — PnL at optimal hold
        pnl_all: (N, 4) float32 — PnL at each horizon
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

    # Rolling features
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
    combined_mask = valid_mask & pred_nonzero
    valid_indices = np.where(combined_mask)[0]

    n_valid = len(valid_indices)
    if n_valid == 0:
        return (np.zeros((0, N_FEATURES), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32))

    eidx_all = pred_event_indices[valid_indices].astype(int)

    # ---- Signal features ----
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

    # ---- Book features ----
    bimb = book_imb_raw[eidx_all]
    bd_raw = np.maximum(bid_depth_raw[eidx_all], 0.0)
    ad_raw = np.maximum(ask_depth_raw[eidx_all], 0.0)
    bd = np.log1p(bd_raw).astype(np.float32)
    ad = np.log1p(ad_raw).astype(np.float32)
    spr = spread_arr[eidx_all]
    depth_rat = (np.exp(bd) / (np.exp(ad) + 1e-8)).astype(np.float32)

    # ---- Volatility/momentum ----
    v50 = vol_50[eidx_all]
    v200 = vol_200[eidx_all]
    v500 = vol_500[eidx_all]
    vol_trend_val = np.where(v200 > 1e-6, (v50 - v200) / (v200 + 1e-8), 0.0).astype(np.float32)

    m20 = mom_20[eidx_all]
    m50_v = mom_50[eidx_all]
    m200_v = mom_200[eidx_all]

    # ---- Volume imbalance ----
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

    # ---- Trade intensity ----
    def _intensity_vec(eidx_arr, window):
        starts = np.maximum(0, eidx_arr - window)
        dt = ts_s[eidx_arr] - ts_s[starts]
        return np.where(dt > 0.01, window / (dt + 1e-8), 0.0).astype(np.float32)

    ti_50 = _intensity_vec(eidx_all, 50)
    ti_200 = _intensity_vec(eidx_all, 200)
    int_ratio = np.where(ti_200 > 1e-6, ti_50 / (ti_200 + 1e-8), 1.0).astype(np.float32)

    # ---- Time features ----
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

    # ---- Hold optimization targets ----
    # Get labels at all 4 horizons, compute directional PnL after cost
    pnl_horizons = []
    any_nan = np.zeros(n_valid, dtype=bool)
    for h in HOLD_HORIZONS:
        lbl = labels_by_horizon[h][valid_indices]
        any_nan |= np.isnan(lbl)
        # Replace NaN with 0 for computation, will filter below
        lbl = np.nan_to_num(lbl, nan=0.0)
        dir_pnl = lbl * sig_dir - COMMISSION_TICKS  # net of passive commission
        pnl_horizons.append(np.clip(dir_pnl, -50, 50).astype(np.float32))

    pnl_all = np.stack(pnl_horizons, axis=1)  # (N, 4)

    # Filter out events where ANY horizon label was NaN
    valid_label_mask = ~any_nan
    features_out = features_out[valid_label_mask]
    pnl_all = pnl_all[valid_label_mask]

    if len(features_out) == 0:
        return (np.zeros((0, N_FEATURES), dtype=np.float32),
                np.zeros(0, dtype=np.int64),
                np.zeros(0, dtype=np.float32),
                np.zeros((0, 4), dtype=np.float32))

    # Optimal hold = argmax PnL across horizons
    hold_class = np.argmax(pnl_all, axis=1).astype(np.int64)
    optimal_pnl = np.max(pnl_all, axis=1).astype(np.float32)

    return features_out, hold_class, optimal_pnl, pnl_all


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_dates() -> List[dict]:
    """Load features + targets for all dates with predictions + MBO data."""
    pred_files = sorted(PRED_DIR.glob("*_predictions.npz"))
    mbo_lookup = {
        Path(f).stem.replace("_mbo_events", ""): f
        for f in sorted(MBO_DIR.glob("*_mbo_events.npz"))
    }

    log.info(f"Found {len(pred_files)} prediction files, {len(mbo_lookup)} MBO files")

    all_dates = []
    for pred_path in pred_files:
        date_str = pred_path.stem.replace("_predictions", "")
        if date_str not in mbo_lookup:
            continue

        mbo_path = mbo_lookup[date_str]
        t0 = time.time()
        try:
            pred_data = np.load(str(pred_path), allow_pickle=True)
            predictions = pred_data["predictions"]

            # Load MBO to get labels at all horizons
            mbo_data = np.load(str(mbo_path))
            labels_by_horizon = {}
            for h in HOLD_HORIZONS:
                key = f"labels_{h}"
                if key in mbo_data:
                    labels_by_horizon[h] = mbo_data[key].astype(np.float32)
                else:
                    log.warning(f"  {date_str}: missing {key}, skipping")
                    break
            else:
                features, hold_class, optimal_pnl, pnl_all = extract_features_for_date(
                    Path(mbo_path), predictions, labels_by_horizon
                )
                elapsed = time.time() - t0

                if len(features) < 10:
                    log.warning(f"  {date_str}: only {len(features)} samples, skip")
                    continue

                all_dates.append({
                    "date": date_str,
                    "features": features,
                    "hold_class": hold_class,
                    "optimal_pnl": optimal_pnl,
                    "pnl_all": pnl_all,
                    "n_samples": len(features),
                })

                if len(all_dates) % 10 == 0:
                    log.info(f"  {len(all_dates)} dates loaded ({date_str}: {len(features):,} in {elapsed:.1f}s)")

        except Exception as e:
            log.error(f"  {date_str}: {e}")
            traceback.print_exc()
            continue

        gc.collect()

    total_samples = sum(d["n_samples"] for d in all_dates)
    log.info(f"Total: {len(all_dates)} dates, {total_samples:,} samples")

    # Log class distribution
    all_classes = np.concatenate([d["hold_class"] for d in all_dates])
    for i, h in enumerate(HOLD_HORIZONS):
        pct = 100.0 * np.mean(all_classes == i)
        log.info(f"  Hold={h}: {pct:.1f}%")

    return all_dates


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

class DynamicHoldMLP(nn.Module):
    """
    Dual-head MLP: classification (which hold is best) + regression (PnL at best hold).
    """
    def __init__(
        self,
        input_dim: int = N_FEATURES,
        hidden_dims: List[int] = [256, 128, 64],
        n_classes: int = N_HOLD_CLASSES,
        dropout: float = DROPOUT,
    ):
        super().__init__()

        layers = []
        prev_dim = input_dim
        for hd in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hd),
                nn.BatchNorm1d(hd),
                nn.GELU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hd

        self.backbone = nn.Sequential(*layers)

        # Classification head: which hold horizon is optimal
        self.cls_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.GELU(),
            nn.Linear(32, n_classes),
        )

        # Regression head: PnL at optimal hold
        self.reg_head = nn.Sequential(
            nn.Linear(prev_dim, 32),
            nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        h = self.backbone(x)
        cls_logits = self.cls_head(h)    # (B, 4)
        reg_out = self.reg_head(h).squeeze(-1)  # (B,)
        return cls_logits, reg_out


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_fold(
    train_dates: List[dict],
    val_dates: List[dict],
    fold_idx: int,
) -> dict:
    """Train one fold, return metrics dict."""

    fold_dir = OUTPUT_DIR / f"fold_{fold_idx:02d}"
    os.makedirs(str(fold_dir), exist_ok=True)

    # Concatenate data
    train_X = np.concatenate([d["features"] for d in train_dates], axis=0)
    train_cls = np.concatenate([d["hold_class"] for d in train_dates], axis=0)
    train_pnl = np.concatenate([d["optimal_pnl"] for d in train_dates], axis=0)

    val_X = np.concatenate([d["features"] for d in val_dates], axis=0)
    val_cls = np.concatenate([d["hold_class"] for d in val_dates], axis=0)
    val_pnl = np.concatenate([d["optimal_pnl"] for d in val_dates], axis=0)
    val_pnl_all = np.concatenate([d["pnl_all"] for d in val_dates], axis=0)

    log.info(f"Fold {fold_idx}: train={len(train_X):,} val={len(val_X):,} "
             f"train_dates={[d['date'] for d in train_dates[:2]]}...{[d['date'] for d in train_dates[-1:]]} "
             f"val_dates={[d['date'] for d in val_dates]}")

    # Normalize using TRAIN stats only
    train_mean = np.nanmean(train_X, axis=0)
    train_std = np.nanstd(train_X, axis=0)
    train_std[train_std < 1e-8] = 1.0

    X_train_norm = np.nan_to_num((train_X - train_mean) / train_std, nan=0.0, posinf=0.0, neginf=0.0)
    X_val_norm = np.nan_to_num((val_X - train_mean) / train_std, nan=0.0, posinf=0.0, neginf=0.0)

    # Clip extreme normalized values to prevent NaN
    X_train_norm = np.clip(X_train_norm, -10.0, 10.0)
    X_val_norm = np.clip(X_val_norm, -10.0, 10.0)

    # Clip regression targets
    train_pnl = np.clip(train_pnl, -20.0, 20.0)
    val_pnl = np.clip(val_pnl, -20.0, 20.0)

    # Sanity check
    n_nan_train = np.isnan(X_train_norm).sum()
    n_nan_val = np.isnan(X_val_norm).sum()
    if n_nan_train > 0 or n_nan_val > 0:
        log.warning(f"  NaN remaining after cleanup: train={n_nan_train} val={n_nan_val}")
        X_train_norm = np.nan_to_num(X_train_norm, nan=0.0)
        X_val_norm = np.nan_to_num(X_val_norm, nan=0.0)

    # Save normalization stats
    np.savez(
        str(fold_dir / "norm_stats.npz"),
        mean=train_mean,
        std=train_std,
    )

    # Build tensors — keep on CPU for DataLoader, move to GPU in loop
    X_t = torch.tensor(X_train_norm, dtype=torch.float32)
    cls_t = torch.tensor(train_cls, dtype=torch.long)
    pnl_t = torch.tensor(train_pnl, dtype=torch.float32)

    # Validation tensors go straight to GPU (no DataLoader)
    X_v = torch.tensor(X_val_norm, dtype=torch.float32, device=DEVICE)
    cls_v = torch.tensor(val_cls, dtype=torch.long, device=DEVICE)
    pnl_v = torch.tensor(val_pnl, dtype=torch.float32, device=DEVICE)

    dataset = TensorDataset(X_t, cls_t, pnl_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True,
                        num_workers=4, pin_memory=True)

    # Model
    model = DynamicHoldMLP().to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-5)

    # Class weights (inverse frequency)
    class_counts = np.bincount(train_cls, minlength=N_HOLD_CLASSES).astype(np.float32)
    class_counts = np.maximum(class_counts, 1.0)
    class_weights = (1.0 / class_counts) * len(train_cls) / N_HOLD_CLASSES
    class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=DEVICE)

    ce_loss_fn = nn.CrossEntropyLoss(weight=class_weights_t)
    mse_loss_fn = nn.MSELoss()

    best_val_acc = 0.0
    best_epoch = 0

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0
        n_batches = 0

        for xb, cb, pb in loader:
            xb, cb, pb = xb.to(DEVICE), cb.to(DEVICE), pb.to(DEVICE)
            cls_logits, reg_out = model(xb)

            loss_cls = ce_loss_fn(cls_logits, cb)
            loss_reg = mse_loss_fn(reg_out, pb)
            loss = CLS_WEIGHT * loss_cls + REG_WEIGHT * loss_reg

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            total_loss += loss.item()
            n_batches += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_cls_logits, val_reg_out = model(X_v)
            val_loss_cls = ce_loss_fn(val_cls_logits, cls_v).item()
            val_loss_reg = mse_loss_fn(val_reg_out, pnl_v).item()
            val_loss = CLS_WEIGHT * val_loss_cls + REG_WEIGHT * val_loss_reg

            val_preds = val_cls_logits.argmax(dim=1).cpu().numpy()
            val_acc = np.mean(val_preds == val_cls) * 100.0

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_epoch = epoch
            torch.save(model.state_dict(), str(fold_dir / "model.pt"))

        if (epoch + 1) % 10 == 0 or epoch == 0:
            log.info(f"  Epoch {epoch+1}/{EPOCHS}: train_loss={total_loss/n_batches:.4f} "
                     f"val_loss={val_loss:.4f} val_acc={val_acc:.1f}% best={best_val_acc:.1f}%@{best_epoch+1}")

    # Load best model for final eval
    model.load_state_dict(torch.load(str(fold_dir / "model.pt"), weights_only=True))
    model.eval()
    with torch.no_grad():
        val_cls_logits, val_reg_out = model(X_v)
        val_preds = val_cls_logits.argmax(dim=1).cpu().numpy()
        val_probs = F.softmax(val_cls_logits, dim=1).cpu().numpy()
        val_reg_np = val_reg_out.cpu().numpy()

    # Save OOT predictions
    np.savez(
        str(fold_dir / "oot_predictions.npz"),
        predicted_hold_class=val_preds,
        hold_probabilities=val_probs,
        predicted_pnl=val_reg_np,
        true_hold_class=val_cls,
        true_optimal_pnl=val_pnl,
        pnl_all_horizons=val_pnl_all,
        val_dates=[d["date"] for d in val_dates],
        feature_names=FEATURE_NAMES,
    )

    # Compute detailed metrics
    val_acc_final = np.mean(val_preds == val_cls) * 100.0
    per_class_acc = {}
    for i, h in enumerate(HOLD_HORIZONS):
        mask_i = val_cls == i
        if mask_i.sum() > 0:
            per_class_acc[h] = np.mean(val_preds[mask_i] == i) * 100.0
        else:
            per_class_acc[h] = 0.0

    # PnL improvement: model-chosen hold vs fixed holds
    model_pnl = val_pnl_all[np.arange(len(val_preds)), val_preds]
    oracle_pnl = val_pnl  # best possible
    fixed_pnls = {h: val_pnl_all[:, i].mean() for i, h in enumerate(HOLD_HORIZONS)}
    model_mean_pnl = model_pnl.mean()

    log.info(f"Fold {fold_idx} RESULTS:")
    log.info(f"  Accuracy: {val_acc_final:.1f}%  Per-class: {per_class_acc}")
    log.info(f"  Model PnL: {model_mean_pnl:.4f} ticks  Oracle PnL: {oracle_pnl.mean():.4f} ticks")
    for h, fp in fixed_pnls.items():
        log.info(f"  Fixed {h}: {fp:.4f} ticks")

    metrics = {
        "fold": fold_idx,
        "val_accuracy": val_acc_final,
        "per_class_accuracy": per_class_acc,
        "model_pnl_mean": float(model_mean_pnl),
        "oracle_pnl_mean": float(oracle_pnl.mean()),
        "fixed_pnl": {h: float(v) for h, v in fixed_pnls.items()},
        "best_epoch": best_epoch + 1,
        "n_train": len(train_X),
        "n_val": len(val_X),
        "train_dates": [d["date"] for d in train_dates],
        "val_dates": [d["date"] for d in val_dates],
    }

    # Save metrics
    with open(str(fold_dir / "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Cleanup GPU
    del X_t, cls_t, pnl_t, X_v, cls_v, pnl_v, model
    torch.cuda.empty_cache()
    gc.collect()

    return metrics


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    log.info(f"Dynamic Hold Optimizer v1 — starting on {DEVICE}")
    log.info(f"Walk-forward: {WF_TRAIN_DAYS} train, {WF_EVAL_DAYS} eval, slide {WF_SLIDE}")

    # MLflow setup
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        mlflow_available = True
        log.info("MLflow connected")
    except Exception as e:
        log.warning(f"MLflow unavailable: {e} — continuing without tracking")
        mlflow_available = False

    # Start MLflow run
    if mlflow_available:
        mlflow.start_run(run_name=f"dynamic_hold_v1_{time.strftime('%Y%m%d_%H%M')}")
        mlflow.log_params({
            "model": "DynamicHoldMLP",
            "hidden_dims": "256-128-64",
            "n_features": N_FEATURES,
            "wf_train_days": WF_TRAIN_DAYS,
            "wf_eval_days": WF_EVAL_DAYS,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "lr": LR,
            "dropout": DROPOUT,
            "cls_weight": CLS_WEIGHT,
            "reg_weight": REG_WEIGHT,
            "commission_ticks": COMMISSION_TICKS,
        })

    # Load data
    log.info("Loading all dates...")
    all_dates = load_all_dates()
    if len(all_dates) < WF_TRAIN_DAYS + WF_EVAL_DAYS:
        log.error(f"Only {len(all_dates)} dates available, need {WF_TRAIN_DAYS + WF_EVAL_DAYS}. Aborting.")
        return

    # Walk-forward folds
    all_metrics = []
    fold_idx = 0
    start = 0

    while start + WF_TRAIN_DAYS + WF_EVAL_DAYS <= len(all_dates):
        train_dates = all_dates[start:start + WF_TRAIN_DAYS]
        val_dates = all_dates[start + WF_TRAIN_DAYS:start + WF_TRAIN_DAYS + WF_EVAL_DAYS]

        metrics = train_one_fold(train_dates, val_dates, fold_idx)
        all_metrics.append(metrics)

        if mlflow_available:
            mlflow.log_metrics({
                f"fold_{fold_idx}_accuracy": metrics["val_accuracy"],
                f"fold_{fold_idx}_model_pnl": metrics["model_pnl_mean"],
                f"fold_{fold_idx}_oracle_pnl": metrics["oracle_pnl_mean"],
            }, step=fold_idx)

        fold_idx += 1
        start += WF_SLIDE

    # Summary
    if all_metrics:
        avg_acc = np.mean([m["val_accuracy"] for m in all_metrics])
        avg_model_pnl = np.mean([m["model_pnl_mean"] for m in all_metrics])
        avg_oracle_pnl = np.mean([m["oracle_pnl_mean"] for m in all_metrics])

        avg_fixed = {}
        for h in HOLD_HORIZONS:
            avg_fixed[h] = np.mean([m["fixed_pnl"][h] for m in all_metrics])

        log.info("=" * 60)
        log.info(f"SUMMARY — {len(all_metrics)} folds")
        log.info(f"  Avg Accuracy: {avg_acc:.1f}%")
        log.info(f"  Avg Model PnL: {avg_model_pnl:.4f} ticks")
        log.info(f"  Avg Oracle PnL: {avg_oracle_pnl:.4f} ticks")
        log.info(f"  Capture ratio: {avg_model_pnl / (avg_oracle_pnl + 1e-8) * 100:.1f}%")
        for h, fp in avg_fixed.items():
            improvement = avg_model_pnl - fp
            log.info(f"  vs Fixed {h}: {improvement:+.4f} ticks ({fp:.4f} fixed)")

        if mlflow_available:
            mlflow.log_metrics({
                "avg_accuracy": avg_acc,
                "avg_model_pnl": avg_model_pnl,
                "avg_oracle_pnl": avg_oracle_pnl,
                "capture_ratio": avg_model_pnl / (avg_oracle_pnl + 1e-8) * 100,
                "n_folds": len(all_metrics),
            })

        # Save overall summary
        summary = {
            "n_folds": len(all_metrics),
            "avg_accuracy": float(avg_acc),
            "avg_model_pnl": float(avg_model_pnl),
            "avg_oracle_pnl": float(avg_oracle_pnl),
            "avg_fixed_pnl": {h: float(v) for h, v in avg_fixed.items()},
            "folds": all_metrics,
        }
        with open(str(OUTPUT_DIR / "summary.json"), "w") as f:
            json.dump(summary, f, indent=2)

    if mlflow_available:
        mlflow.end_run()

    log.info("Training complete.")


if __name__ == "__main__":
    main()
