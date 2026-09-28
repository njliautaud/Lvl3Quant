#!/usr/bin/env python3
"""
Fill Timing Model — Predicts optimal limit-order placement timing
=================================================================

Given a CNN-Mamba signal event + current microstructure state, predicts
the time (in seconds) until a passive limit order at the near-side would fill.

This is a REGRESSION model: target = time_to_fill (seconds).
Also includes a classification head: will it fill within T seconds? (binary)

Key insight: We don't need to predict IF a signal is profitable (that's the 
signal model's job). We need to predict WHEN to place the order to maximize
fill rate while minimizing adverse selection.

Architecture: MLP with residual connections (proven in supervised_exec_v2).
Features: 42 microstructure features (same as exec_v2) + 8 queue/fill features.
Target: Time-to-fill in seconds, computed from actual MBO book dynamics.

Walk-forward: 60d train, 5d eval, sliding window (mandatory per DIRECTIVES).
MLflow logging: mandatory.

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
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LVL3_ROOT = Path("/home/nick/Lvl3Quant")
MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"
OUTPUT_DIR = LVL3_ROOT / "output" / "fill_timing_v1"

PRED_DIRS = [
    LVL3_ROOT / "output" / "cnn_mamba_v2_all_oot",
    LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_oot",
    LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_inference",
    LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar",
]

ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE  # 0.376

COL_PRICE_REL = 3
COL_SPREAD = 5
COL_SIDE = 2
COL_QTY_LOG = 4

PRED_STRIDE = 250
PRED_WINDOW = 3000

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "fill_timing_v1"

# Fill timing constants
MAX_FILL_TIME_S = 30.0   # Cap fill time at 30s (matches signal decay horizon)
FILL_THRESHOLD_S = 5.0   # Binary: does it fill within 5s?

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
LOG_FILE = LVL3_ROOT / "output" / "fill_timing_v1.log"
os.makedirs(str(LOG_FILE.parent), exist_ok=True)
logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [FILL_TIMING] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.FileHandler(str(LOG_FILE), mode="w"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("fill_timing")

# ---------------------------------------------------------------------------
# Feature names (50 features = 42 base + 8 queue features)
# ---------------------------------------------------------------------------
BASE_FEATURE_NAMES = [
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

QUEUE_FEATURE_NAMES = [
    "near_depth_log",       # Depth at near side (bid for buy, ask for sell)
    "far_depth_log",        # Depth at far side
    "near_far_ratio",       # near_depth / far_depth
    "recent_fill_rate",     # Fills in last 200 events / total events
    "signed_pressure",      # Signed order flow pressure
    "price_at_near_edge",   # Is price at near-side edge? (binary)
    "queue_momentum",       # Change in near-side depth over recent events
    "cross_spread_rate",    # Rate of spread-crossing events
]

FEATURE_NAMES = BASE_FEATURE_NAMES + QUEUE_FEATURE_NAMES
N_FEATURES = len(FEATURE_NAMES)  # 50

HORIZONS = ["1s", "5s", "10s"]

# ---------------------------------------------------------------------------
# Rolling helpers (same as exec_v2)
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
# Feature extraction with fill timing targets
# ---------------------------------------------------------------------------

def extract_features_for_date(
    mbo_path: Path,
    predictions: np.ndarray,
    labels: np.ndarray,
) -> Tuple[np.ndarray, Dict[str, np.ndarray], np.ndarray]:
    """
    Extract 50 features + fill-timing targets for one date.
    
    Targets:
    - time_to_fill: seconds until a passive limit at near-side would fill
    - fill_within_5s: binary, does it fill within 5 seconds?
    - adverse_move_at_fill: how much the price moved against us at fill time
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

    # Event type tracking for fill rate
    event_type = data.get("event_type_raw", np.zeros(n_events, dtype=np.int8))
    is_fill = (event_type == 2).astype(np.float32)  # type 2 = trade/fill
    fill_cum = np.cumsum(is_fill)
    fill_cum_pad = np.concatenate([[0.0], fill_cum])

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
        return np.zeros((0, N_FEATURES), dtype=np.float32), {}, np.zeros((0, 3), dtype=np.float32)

    eidx_all = pred_event_indices[valid_indices].astype(int)

    # ---- Signal features (same as exec_v2) ----
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

    # ---- Queue features (NEW for fill timing) ----
    # Near depth: if signal is long, near side = ask; if short, near side = bid
    # We want the depth at the level where we'd place our passive order
    near_depth = np.where(sig_dir > 0, bd_raw, ad_raw)
    far_depth = np.where(sig_dir > 0, ad_raw, bd_raw)
    near_depth_log = np.log1p(near_depth).astype(np.float32)
    far_depth_log = np.log1p(far_depth).astype(np.float32)
    near_far_ratio = np.where(far_depth > 1e-6, near_depth / (far_depth + 1e-8), 1.0).astype(np.float32)

    # Recent fill rate
    def _fill_rate_vec(eidx_arr, window):
        starts = np.maximum(0, eidx_arr - window)
        fills = fill_cum_pad[eidx_arr + 1] - fill_cum_pad[starts + 1]
        return (fills / window).astype(np.float32)
    
    recent_fill_rate = _fill_rate_vec(eidx_all, 200)

    # Signed pressure: net order flow in signal direction
    signed_pressure = (vi_50 * sig_dir).astype(np.float32)

    # Price at near-side edge (spread = 1 tick = minimum)
    price_at_edge = (spr <= 1.01).astype(np.float32)

    # Queue momentum: change in near-side depth
    # Approximate by looking at depth change over recent events
    near_depth_200 = np.where(
        sig_dir > 0,
        np.maximum(bid_depth_raw[np.maximum(0, eidx_all - 200)], 0.0),
        np.maximum(ask_depth_raw[np.maximum(0, eidx_all - 200)], 0.0)
    )
    queue_momentum = np.where(
        near_depth_200 > 1e-6,
        (near_depth - near_depth_200) / (near_depth_200 + 1e-8),
        0.0
    ).astype(np.float32)

    # Cross-spread rate
    cross_spread_rate = _fill_rate_vec(eidx_all, 100)  # Fills are approximated by trade events

    # ---- Assemble feature matrix (50 features) ----
    features_out = np.stack([
        # 42 base features
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
        # 8 queue features
        near_depth_log, far_depth_log, near_far_ratio,
        recent_fill_rate, signed_pressure, price_at_edge,
        queue_momentum, cross_spread_rate,
    ], axis=1).astype(np.float32)

    # ---- Fill timing targets ----
    # For each prediction event, compute time until price crosses the near-side level
    # This simulates: "if I place a limit at bid (for buy) or ask (for sell), when does it fill?"
    
    targets = {}
    
    # Use the actual price movement data to compute fill times
    # A passive buy at bid fills when price drops to or below current bid
    # A passive sell at ask fills when price rises to or above current ask
    # We approximate: time until price moves in our direction by >= 0 ticks (any favorable touch)
    
    l1 = labels[valid_indices, 0]  # 1s horizon price move
    l5 = labels[valid_indices, 1]  # 5s horizon
    l10 = labels[valid_indices, 2]  # 10s horizon
    
    # Directional move in signal direction
    for h_name, lbl in zip(HORIZONS, [l1, l5, l10]):
        dir_move = lbl * sig_dir
        pnl_h = dir_move - COMMISSION_TICKS
        targets[f"pnl_{h_name}"] = np.clip(pnl_h, -50, 50).astype(np.float32)
        targets[f"profitable_{h_name}"] = (pnl_h > 0).astype(np.float32)
    
    # Time-to-fill proxy: use event-level price dynamics
    # For each signal event, scan forward in the event stream to find when
    # the price crosses back to or through the signal-side level
    
    fill_times = np.full(n_valid, MAX_FILL_TIME_S, dtype=np.float32)
    adverse_at_fill = np.zeros(n_valid, dtype=np.float32)
    
    # Batch processing: for events where we can look forward
    for vi in range(n_valid):
        ei = eidx_all[vi]
        sd = sig_dir[vi]
        ref_price = price_rel[ei]
        t_start = ts_s[ei]
        
        # Look forward up to 30s worth of events (but cap at n_events)
        # For efficiency, look at most 10000 events ahead
        end_ei = min(n_events, ei + 10000)
        future_prices = price_rel[ei:end_ei]
        future_ts = ts_s[ei:end_ei]
        dt = future_ts - t_start
        
        # Cap at MAX_FILL_TIME_S
        within_window = dt <= MAX_FILL_TIME_S
        if not np.any(within_window):
            continue
            
        fp = future_prices[within_window]
        ft = dt[within_window]
        
        # Fill condition: for a buy limit at current bid, price must come to us
        # For long signal: we're buying at bid. Fill when someone sells to us (price drops to bid).
        # Approximation: price moves below reference level
        # For short signal: we're selling at ask. Fill when price rises to ask.
        if sd > 0:  # Long signal
            # Buy at bid: fill when price drops to/below ref (someone hits our bid)
            fill_mask = fp <= ref_price
        else:  # Short signal
            # Sell at ask: fill when price rises to/above ref
            fill_mask = fp >= ref_price
        
        if np.any(fill_mask):
            first_fill_idx = np.argmax(fill_mask)
            fill_times[vi] = ft[first_fill_idx]
            # Adverse move at fill = how much price moved against us
            adverse_at_fill[vi] = np.abs(fp[first_fill_idx] - ref_price)
    
    targets["fill_time_s"] = fill_times
    targets["fill_within_5s"] = (fill_times <= FILL_THRESHOLD_S).astype(np.float32)
    targets["fill_within_10s"] = (fill_times <= 10.0).astype(np.float32)
    targets["adverse_at_fill"] = adverse_at_fill
    targets["log_fill_time"] = np.log1p(fill_times).astype(np.float32)  # Better target distribution
    
    meta_out = np.stack([cur_ts, valid_indices.astype(np.float64), sig_dir.astype(np.float64)], axis=1).astype(np.float32)

    return features_out, targets, meta_out


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def build_date_pred_index() -> Dict[str, Path]:
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
    return index


def load_all_dates() -> List[dict]:
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
# Model
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


class FillTimingMLP(nn.Module):
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

        # Regression head: log(fill_time) (1 output)
        self.reg_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

        # Classification head: fill_within_5s (1 output, sigmoid applied in loss)
        self.cls_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        
        # Adverse selection head: adverse_move_at_fill (1 output)
        self.adverse_head = nn.Sequential(
            nn.Linear(hidden_dims[-1], 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )

    def forward(self, x):
        h = self.backbone(x)
        reg_out = self.reg_head(h).squeeze(-1)
        cls_out = self.cls_head(h).squeeze(-1)
        adv_out = self.adverse_head(h).squeeze(-1)
        return reg_out, cls_out, adv_out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_fold(
    model: FillTimingMLP,
    train_X: torch.Tensor,
    train_fill_time: torch.Tensor,
    train_fill_cls: torch.Tensor,
    train_adverse: torch.Tensor,
    val_X: torch.Tensor,
    val_fill_time: torch.Tensor,
    val_fill_cls: torch.Tensor,
    val_adverse: torch.Tensor,
    epochs: int = 30,
    batch_size: int = 4096,
    lr: float = 1e-3,
    patience: int = 10,
) -> Dict:
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    
    huber_loss = nn.SmoothL1Loss()
    bce_loss = nn.BCEWithLogitsLoss()
    
    train_ds = TensorDataset(train_X, train_fill_time, train_fill_cls, train_adverse)
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    
    val_ds = TensorDataset(val_X, val_fill_time, val_fill_cls, val_adverse)
    val_loader = DataLoader(val_ds, batch_size=batch_size * 2, shuffle=False, num_workers=0)
    
    best_val_loss = float("inf")
    best_state = None
    wait = 0
    
    for epoch in range(1, epochs + 1):
        # Train
        model.train()
        train_losses = []
        for batch in train_loader:
            bx, bt, bc, ba = [b.to(DEVICE) for b in batch]
            reg_out, cls_out, adv_out = model(bx)
            
            loss_reg = huber_loss(reg_out, bt)
            loss_cls = bce_loss(cls_out, bc)
            loss_adv = huber_loss(adv_out, ba) * 0.5  # Lower weight for adverse selection
            loss = loss_reg + loss_cls + loss_adv
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_losses.append(loss.item())
        
        scheduler.step()
        
        # Validate
        model.eval()
        val_losses = []
        val_reg_losses = []
        val_cls_losses = []
        with torch.no_grad():
            for batch in val_loader:
                bx, bt, bc, ba = [b.to(DEVICE) for b in batch]
                reg_out, cls_out, adv_out = model(bx)
                loss_reg = huber_loss(reg_out, bt)
                loss_cls = bce_loss(cls_out, bc)
                loss_adv = huber_loss(adv_out, ba) * 0.5
                val_losses.append((loss_reg + loss_cls + loss_adv).item())
                val_reg_losses.append(loss_reg.item())
                val_cls_losses.append(loss_cls.item())
        
        avg_train = np.mean(train_losses)
        avg_val = np.mean(val_losses)
        avg_val_reg = np.mean(val_reg_losses)
        avg_val_cls = np.mean(val_cls_losses)
        
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            log.info(f"  E{epoch}/{epochs}: train={avg_train:.4f} val={avg_val:.4f} (reg={avg_val_reg:.4f} cls={avg_val_cls:.4f})")
        
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            wait = 0
        else:
            wait += 1
            if wait >= patience:
                log.info(f"  Early stop at epoch {epoch}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return {"best_val_loss": best_val_loss, "stopped_epoch": epoch}


# ---------------------------------------------------------------------------
# Walk-forward
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-train-days", type=int, default=60)
    parser.add_argument("--n-eval-days", type=int, default=5)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=4096)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden-dims", type=int, nargs="+", default=[256, 128, 64, 32])
    parser.add_argument("--dropout", type=float, default=0.3)
    args = parser.parse_args()

    log.info(f"Fill Timing Model v1 — {DEVICE}")
    log.info(f"Config: {vars(args)}")

    # Load data
    log.info("Loading all dates...")
    all_dates = load_all_dates()
    n_dates = len(all_dates)

    if n_dates < args.n_train_days + args.n_eval_days:
        log.error(f"Not enough dates: {n_dates} < {args.n_train_days + args.n_eval_days}")
        return

    n_folds = max(1, (n_dates - args.n_train_days) // args.n_eval_days)
    log.info(f"Walk-forward: {n_dates} dates, {args.n_train_days}d train + {args.n_eval_days}d eval = {n_folds} folds")

    os.makedirs(str(OUTPUT_DIR), exist_ok=True)

    # MLflow setup
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        run = mlflow.start_run(run_name=f"fill_timing_v1_{time.strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params(vars(args))
        mlflow.log_param("n_dates", n_dates)
        mlflow.log_param("n_folds", n_folds)
        mlflow.log_param("n_features", N_FEATURES)
        use_mlflow = True
        log.info(f"MLflow run: {run.info.run_id}")
    except Exception as e:
        log.warning(f"MLflow unavailable: {e}")
        use_mlflow = False

    all_fold_metrics = []

    for fold_idx in range(n_folds):
        train_start = fold_idx * args.n_eval_days
        train_end = train_start + args.n_train_days
        eval_start = train_end
        eval_end = min(eval_start + args.n_eval_days, n_dates)

        if eval_end <= eval_start:
            break

        train_dates = [d["date"] for d in all_dates[train_start:train_end]]
        eval_dates = [d["date"] for d in all_dates[eval_start:eval_end]]

        log.info(f"\n{'='*60}")
        log.info(f"Fold {fold_idx}: train=[{train_dates[0]}..{train_dates[-1]}] ({len(train_dates)}d), "
                 f"eval=[{eval_dates[0]}..{eval_dates[-1]}] ({len(eval_dates)}d)")

        # Assemble train/eval data
        train_features = np.concatenate([d["features"] for d in all_dates[train_start:train_end]])
        eval_features = np.concatenate([d["features"] for d in all_dates[eval_start:eval_end]])

        train_fill_time = np.concatenate([d["targets"]["log_fill_time"] for d in all_dates[train_start:train_end]])
        eval_fill_time = np.concatenate([d["targets"]["log_fill_time"] for d in all_dates[eval_start:eval_end]])
        
        train_fill_cls = np.concatenate([d["targets"]["fill_within_5s"] for d in all_dates[train_start:train_end]])
        eval_fill_cls = np.concatenate([d["targets"]["fill_within_5s"] for d in all_dates[eval_start:eval_end]])
        
        train_adverse = np.concatenate([d["targets"]["adverse_at_fill"] for d in all_dates[train_start:train_end]])
        eval_adverse = np.concatenate([d["targets"]["adverse_at_fill"] for d in all_dates[eval_start:eval_end]])

        log.info(f"  Train: {len(train_features):,}  Eval: {len(eval_features):,}")

        # Normalize features (train stats only)
        mean = np.nanmean(train_features, axis=0)
        std = np.nanstd(train_features, axis=0)
        std = np.where(std < 1e-6, 1.0, std)

        train_X = torch.from_numpy(((train_features - mean) / std).astype(np.float32)).to(DEVICE)
        eval_X = torch.from_numpy(((eval_features - mean) / std).astype(np.float32)).to(DEVICE)

        train_ft_t = torch.from_numpy(train_fill_time).to(DEVICE)
        eval_ft_t = torch.from_numpy(eval_fill_time).to(DEVICE)
        train_fc_t = torch.from_numpy(train_fill_cls).to(DEVICE)
        eval_fc_t = torch.from_numpy(eval_fill_cls).to(DEVICE)
        train_adv_t = torch.from_numpy(train_adverse).to(DEVICE)
        eval_adv_t = torch.from_numpy(eval_adverse).to(DEVICE)

        # Replace NaN/Inf in features
        train_X = torch.nan_to_num(train_X, nan=0.0, posinf=10.0, neginf=-10.0)
        eval_X = torch.nan_to_num(eval_X, nan=0.0, posinf=10.0, neginf=-10.0)

        # Train model
        model = FillTimingMLP(
            input_dim=N_FEATURES,
            hidden_dims=args.hidden_dims,
            dropout=args.dropout,
        ).to(DEVICE)

        train_result = train_fold(
            model, train_X, train_ft_t, train_fc_t, train_adv_t,
            eval_X, eval_ft_t, eval_fc_t, eval_adv_t,
            epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        )

        # Evaluate
        model.eval()
        with torch.no_grad():
            reg_pred, cls_pred, adv_pred = model(eval_X)
            reg_pred = reg_pred.cpu().numpy()
            cls_pred = torch.sigmoid(cls_pred).cpu().numpy()
            adv_pred = adv_pred.cpu().numpy()

        eval_ft_np = eval_fill_time
        eval_fc_np = eval_fill_cls

        # Spearman on fill time prediction
        spear_ft, _ = spearmanr(eval_ft_np, reg_pred)
        
        # AUC on fill-within-5s
        from sklearn.metrics import roc_auc_score
        auc_fc = roc_auc_score(eval_fc_np, cls_pred) if len(np.unique(eval_fc_np)) > 1 else float('nan')
        
        # Also evaluate profitability prediction (how well fill timing correlates with PnL)
        eval_pnl_1s = np.concatenate([d["targets"]["pnl_1s"] for d in all_dates[eval_start:eval_end]])
        # Idea: short predicted fill time + high confidence = good trade
        # Score = -predicted_fill_time (negative because shorter is better)
        trade_score = -reg_pred
        spear_pnl, _ = spearmanr(eval_pnl_1s, trade_score)

        log.info(f"  Fill Time: Spearman={spear_ft:.4f}, AUC(5s)={auc_fc:.4f}")
        log.info(f"  Fill-PnL correlation: Spearman={spear_pnl:.4f}")
        log.info(f"  Fill rate within 5s: {eval_fc_np.mean():.3f}")

        fold_metrics = {
            "fold": fold_idx,
            "train_dates": train_dates,
            "eval_dates": eval_dates,
            "spearman_fill_time": float(spear_ft),
            "auc_fill_5s": float(auc_fc),
            "spearman_pnl": float(spear_pnl),
            "best_val_loss": float(train_result["best_val_loss"]),
            "n_eval": len(eval_features),
            "fill_rate_5s": float(eval_fc_np.mean()),
        }
        all_fold_metrics.append(fold_metrics)

        if use_mlflow:
            for k, v in fold_metrics.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"fold_{fold_idx}_{k}", v)

        # Save fold outputs
        fold_dir = OUTPUT_DIR / f"fold_{fold_idx:02d}"
        os.makedirs(str(fold_dir), exist_ok=True)
        
        torch.save(model.state_dict(), str(fold_dir / "model.pt"))
        np.savez(str(fold_dir / "norm_stats.npz"), mean=mean, std=std)
        np.savez(
            str(fold_dir / "oot_predictions.npz"),
            pred_fill_time=reg_pred,
            pred_fill_prob=cls_pred,
            pred_adverse=adv_pred,
            actual_fill_time=eval_ft_np,
            actual_fill_cls=eval_fc_np,
            actual_pnl_1s=eval_pnl_1s,
            eval_dates=np.array(eval_dates),
        )

        # Free GPU memory
        del model, train_X, eval_X, train_ft_t, eval_ft_t, train_fc_t, eval_fc_t
        del train_adv_t, eval_adv_t
        torch.cuda.empty_cache()
        gc.collect()

    # Summary
    log.info(f"\n{'='*60}")
    log.info("SUMMARY")
    log.info(f"{'='*60}")
    
    if all_fold_metrics:
        avg_spear_ft = np.mean([m["spearman_fill_time"] for m in all_fold_metrics])
        avg_auc = np.mean([m["auc_fill_5s"] for m in all_fold_metrics])
        avg_spear_pnl = np.mean([m["spearman_pnl"] for m in all_fold_metrics])
        
        log.info(f"Folds: {len(all_fold_metrics)}")
        log.info(f"Avg Spearman (fill time): {avg_spear_ft:.4f}")
        log.info(f"Avg AUC (fill within 5s): {avg_auc:.4f}")
        log.info(f"Avg Spearman (fill-PnL): {avg_spear_pnl:.4f}")
        
        if use_mlflow:
            mlflow.log_metric("avg_spearman_fill_time", avg_spear_ft)
            mlflow.log_metric("avg_auc_fill_5s", avg_auc)
            mlflow.log_metric("avg_spearman_pnl", avg_spear_pnl)
            mlflow.log_metric("total_folds", len(all_fold_metrics))
            mlflow.end_run()
        
        # Save summary
        with open(str(OUTPUT_DIR / "summary.json"), "w") as f:
            json.dump({
                "all_fold_metrics": all_fold_metrics,
                "avg_spearman_fill_time": avg_spear_ft,
                "avg_auc_fill_5s": avg_auc,
                "avg_spearman_pnl": avg_spear_pnl,
            }, f, indent=2, default=str)

    log.info("Done!")


if __name__ == "__main__":
    main()
