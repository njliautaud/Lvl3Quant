#!/usr/bin/env python3
"""
LightGBM Execution Predictor — Gradient Boosted Trees for Trade Outcome Prediction
=====================================================================================

WHY THIS APPROACH:
- MLP-based supervised exec (v1-v3) showed near-zero correlation (Spearman 0.01-0.05)
- LightGBM is known to outperform MLPs on tabular data with <100 features
- Trees can find non-linear interactions (e.g., "high confidence + low spread + high volume = good trade")
- Feature importance built-in — tells us what actually matters
- Trains in seconds, not minutes — enables rapid iteration

WHAT'S DIFFERENT FROM MLP VERSIONS:
1. LightGBM instead of MLP — better for tabular data
2. Richer features: rolling windows at multiple timescales, interaction features
3. Multi-horizon analysis: separate models for 1s, 5s, 10s outcomes
4. Built-in feature importance ranking
5. Quantile regression option (predict confidence intervals, not just point estimates)

TARGETS:
- Primary: binary profitable (P&L > 0 at each horizon)
- Secondary: regression on directional P&L at each horizon
- Tertiary: MFE prediction (for exit timing)

Walk-forward sliding window (HC #0). Commission = 0.376 ticks RT only.

Author: Claude (Infrastructure Builder)
Date: 2026-05-08
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed. Run: pip install lightgbm")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Paths and constants
# ---------------------------------------------------------------------------
LVL3_ROOT = Path(os.environ.get("LVL3_ROOT", "/home/nick/Lvl3Quant"))

ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
COMMISSION_TICKS = ES_RT_COMMISSION / ES_TICK_VALUE  # 0.376

# MBO event column indices (from fifo_rl_env.py)
COL_PRICE_REL = 0
COL_SPREAD = 4
COL_SIDE = 1
COL_QTY_LOG = 3

# Prediction parameters
PRED_STRIDE = 64  # MBO events between predictions
PRED_WINDOW = 256  # MBO events per CNN window

LOG = logging.getLogger("LGBM_EXEC")

# ---------------------------------------------------------------------------
# Extended feature extraction with multi-timescale features
# ---------------------------------------------------------------------------

FEATURE_NAMES = [
    # Signal features (raw)
    "pred_1s", "pred_5s", "pred_10s",
    "abs_pred_1s", "abs_pred_5s", "abs_pred_10s",
    "signal_direction",  # +1/-1
    "confidence_tier",   # 0-3
    "signal_agreement",  # all horizons same sign
    "signal_strength",   # mean abs prediction
    "pred_std",          # cross-horizon disagreement
    "pred_range",        # max - min across horizons
    "pred_ratio_5s_1s",  # persistence: how much signal persists from 1s to 5s
    "pred_ratio_10s_1s", # persistence: how much signal persists from 1s to 10s

    # Book state features
    "book_imbalance",
    "bid_depth_log",
    "ask_depth_log",
    "spread",
    "depth_ratio",       # bid/ask depth ratio (>1 = more buy pressure)

    # Microstructure features - multiple timescales
    "recent_volatility_50",   # 50-event rolling
    "recent_volatility_200",  # 200-event rolling
    "recent_volatility_500",  # 500-event rolling
    "vol_trend",              # volatility increasing or decreasing

    "price_momentum_20",   # very short momentum
    "price_momentum_50",   # short momentum
    "price_momentum_200",  # medium momentum

    "volume_imbalance_50",   # buy-sell imbalance (50 events)
    "volume_imbalance_200",  # buy-sell imbalance (200 events)
    "volume_imbalance_500",  # buy-sell imbalance (500 events)
    "vol_imb_trend",         # volume imbalance trend (short vs long)

    "trade_intensity_50",    # events per second (50 events)
    "trade_intensity_200",   # events per second (200 events)
    "intensity_ratio",       # short/long intensity ratio (acceleration)

    # Time features
    "time_of_day",           # fractional 0=open, 1=close
    "minutes_since_open",    # raw minutes
    "is_first_30min",        # opening volatility flag
    "is_last_30min",         # closing volatility flag

    # Interaction features (pre-computed for trees)
    "conf_x_vol",            # confidence * volatility (high conf in high vol = different)
    "conf_x_imbalance",      # confidence * book imbalance alignment
    "conf_x_momentum",       # confidence * momentum alignment
    "signal_aligned_with_imbalance",  # signal direction matches book imbalance
    "signal_aligned_with_momentum",   # signal direction matches price momentum
]

N_FEATURES = len(FEATURE_NAMES)

# Multiple target horizons
HORIZONS = ["1s", "5s", "10s"]
TARGET_NAMES_PER_HORIZON = ["dir_move", "mfe", "mae", "pnl", "profitable"]


def _time_of_day_fraction(ts_s: float) -> float:
    seconds_in_day = ts_s % 86400
    et_seconds = (seconds_in_day - 4 * 3600) % 86400
    rth_start = 9.5 * 3600
    rth_end = 16.0 * 3600
    rth_duration = rth_end - rth_start
    frac = (et_seconds - rth_start) / rth_duration
    return max(0.0, min(1.0, frac))


def _rolling_std(values: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling std."""
    n = len(values)
    out = np.zeros(n, dtype=np.float32)
    if n < window:
        return out
    cs = np.cumsum(values.astype(np.float64))
    cs2 = np.cumsum(values.astype(np.float64) ** 2)
    cs_pad = np.concatenate([[0.0], cs])
    cs2_pad = np.concatenate([[0.0], cs2])
    valid = np.arange(window, n)
    sm = cs_pad[valid + 1] - cs_pad[valid - window + 1]
    sm2 = cs2_pad[valid + 1] - cs2_pad[valid - window + 1]
    mean = sm / window
    var = np.maximum(sm2 / window - mean ** 2, 0.0)
    out[valid] = np.sqrt(var).astype(np.float32)
    return out


def _rolling_mean(values: np.ndarray, window: int) -> np.ndarray:
    """Vectorized rolling mean."""
    n = len(values)
    out = np.zeros(n, dtype=np.float32)
    if n < window:
        return out
    cs = np.cumsum(values.astype(np.float64))
    cs_pad = np.concatenate([[0.0], cs])
    valid = np.arange(window, n)
    out[valid] = ((cs_pad[valid + 1] - cs_pad[valid - window + 1]) / window).astype(np.float32)
    return out


def extract_features_for_date(
    mbo_path: Path,
    predictions: np.ndarray,
    labels: np.ndarray,
    pred_stride: int = PRED_STRIDE,
    pred_window: int = PRED_WINDOW,
) -> Tuple[np.ndarray, dict, np.ndarray]:
    """
    Extract rich features and multi-horizon targets for one day.

    Returns:
        features: (N_valid, N_FEATURES) float32
        targets: dict with keys like "dir_move_1s", "profitable_5s", etc.
        meta: (N_valid, 3) float32 [timestamp_s, pred_idx, signal_direction]
    """
    data = np.load(str(mbo_path))
    events = data["events"]
    timestamps = data["timestamps"]

    n_events = len(events)
    n_preds = len(predictions)

    # Raw arrays
    price_rel = events[:, COL_PRICE_REL].astype(np.float64)
    spread = events[:, COL_SPREAD].astype(np.float32)
    side = events[:, COL_SIDE].astype(np.float32)
    qty_log = events[:, COL_QTY_LOG].astype(np.float32)
    ts_s = timestamps.astype(np.float64) / 1e9

    price_changes = np.diff(price_rel, prepend=price_rel[0])

    # Pre-compute rolling features at multiple timescales
    vol_50 = _rolling_std(price_changes, 50)
    vol_200 = _rolling_std(price_changes, 200)
    vol_500 = _rolling_std(price_changes, 500)

    mom_20 = _rolling_mean(price_changes, 20)
    mom_50 = _rolling_mean(price_changes, 50)
    mom_200 = _rolling_mean(price_changes, 200)

    # Volume tracking
    qty = np.exp(qty_log)
    buy_mask = (side > 0).astype(np.float32)
    sell_mask = (side < 0).astype(np.float32)
    buy_vol = qty * buy_mask
    sell_vol = qty * sell_mask
    buy_cum = np.cumsum(buy_vol)
    sell_cum = np.cumsum(sell_vol)
    buy_cum_pad = np.concatenate([[0.0], buy_cum])
    sell_cum_pad = np.concatenate([[0.0], sell_cum])

    # Book features
    bid_depth_raw = events[:, 6] if events.shape[1] > 6 else np.ones(n_events, dtype=np.float32)
    ask_depth_raw = events[:, 7] if events.shape[1] > 7 else np.ones(n_events, dtype=np.float32)
    book_imb_raw = events[:, 8] if events.shape[1] > 8 else np.zeros(n_events, dtype=np.float32)

    # Compute event indices for all predictions
    pred_event_indices = pred_window + np.arange(n_preds) * pred_stride

    # Valid mask
    valid_mask = pred_event_indices < (n_events - 1)
    pred_nonzero = ~((predictions[:, 0] == 0) & (predictions[:, 1] == 0) & (predictions[:, 2] == 0))
    pred_has_dir = predictions[:, 0] != 0
    labels_valid = ~np.any(np.isnan(labels), axis=1)
    combined_mask = valid_mask & pred_nonzero & pred_has_dir & labels_valid
    valid_indices = np.where(combined_mask)[0]

    n_valid = len(valid_indices)
    features_out = np.zeros((n_valid, N_FEATURES), dtype=np.float32)

    # Multi-horizon targets
    targets = {
        f"{tname}_{h}": np.zeros(n_valid, dtype=np.float32)
        for h in HORIZONS for tname in TARGET_NAMES_PER_HORIZON
    }
    meta_out = np.zeros((n_valid, 3), dtype=np.float32)

    for i, pred_idx in enumerate(valid_indices):
        eidx = int(pred_event_indices[pred_idx])

        p1 = float(predictions[pred_idx, 0])
        p5 = float(predictions[pred_idx, 1])
        p10 = float(predictions[pred_idx, 2])

        l1 = float(labels[pred_idx, 0])
        l5 = float(labels[pred_idx, 1])
        l10 = float(labels[pred_idx, 2])

        sig_dir = 1.0 if p1 > 0 else -1.0
        a1, a5, a10 = abs(p1), abs(p5), abs(p10)

        # --- Targets per horizon ---
        for h_idx, (h_name, label_val) in enumerate(zip(HORIZONS, [l1, l5, l10])):
            dir_move = label_val * sig_dir
            mfe_h = max(dir_move, 0.0)
            mae_h = max(-dir_move, 0.0)
            pnl_h = dir_move - COMMISSION_TICKS
            prof_h = 1.0 if pnl_h > 0 else 0.0

            targets[f"dir_move_{h_name}"][i] = min(max(dir_move, -50.0), 50.0)
            targets[f"mfe_{h_name}"][i] = min(mfe_h, 50.0)
            targets[f"mae_{h_name}"][i] = min(mae_h, 50.0)
            targets[f"pnl_{h_name}"][i] = min(max(pnl_h, -50.0), 50.0)
            targets[f"profitable_{h_name}"][i] = prof_h

        # --- Feature extraction ---
        cur_ts = float(ts_s[eidx])
        tod = _time_of_day_fraction(cur_ts)

        # Signal features
        tier = 0.0 if a1 < 0.10 else (1.0 if a1 < 0.25 else (2.0 if a1 < 0.50 else 3.0))
        signs_agree = 1.0 if (np.sign(p1) == np.sign(p5) == np.sign(p10)) and np.sign(p1) != 0 else 0.0
        strength = (a1 + a5 + a10) / 3.0
        preds = np.array([p1, p5, p10])
        p_std = float(np.std(preds))
        p_range = float(preds.max() - preds.min())
        ratio_5_1 = p5 / (p1 + 1e-8) if abs(p1) > 1e-6 else 0.0
        ratio_10_1 = p10 / (p1 + 1e-8) if abs(p1) > 1e-6 else 0.0

        # Book features
        bimb = float(book_imb_raw[eidx])
        bd = float(np.log1p(max(float(bid_depth_raw[eidx]), 0)))
        ad = float(np.log1p(max(float(ask_depth_raw[eidx]), 0)))
        spr = float(spread[eidx])
        depth_rat = np.exp(bd) / (np.exp(ad) + 1e-8)

        # Volatility at multiple scales
        v50 = float(vol_50[eidx])
        v200 = float(vol_200[eidx])
        v500 = float(vol_500[eidx])
        vol_trend = (v50 - v200) / (v200 + 1e-8) if v200 > 1e-6 else 0.0

        # Momentum at multiple scales
        m20 = float(mom_20[eidx])
        m50 = float(mom_50[eidx])
        m200 = float(mom_200[eidx])

        # Volume imbalance at multiple scales
        def _vol_imb(window):
            start = max(0, eidx - window)
            bv = float(buy_cum_pad[eidx + 1] - buy_cum_pad[start + 1])
            sv = float(sell_cum_pad[eidx + 1] - sell_cum_pad[start + 1])
            tot = bv + sv
            return (bv - sv) / (tot + 1e-8) if tot > 0 else 0.0

        vi_50 = _vol_imb(50)
        vi_200 = _vol_imb(200)
        vi_500 = _vol_imb(500)
        vi_trend = vi_50 - vi_200  # short-term vs medium-term

        # Trade intensity (events per second)
        def _intensity(window):
            start = max(0, eidx - window)
            dt = float(ts_s[eidx] - ts_s[start])
            return window / (dt + 1e-8) if dt > 0.01 else 0.0

        ti_50 = _intensity(50)
        ti_200 = _intensity(200)
        int_ratio = ti_50 / (ti_200 + 1e-8) if ti_200 > 1e-6 else 1.0

        # Time features
        minutes = tod * 390  # 6.5 hours = 390 minutes
        is_first_30 = 1.0 if minutes < 30 else 0.0
        is_last_30 = 1.0 if minutes > 360 else 0.0

        # Interaction features
        conf_x_vol = a1 * v200
        conf_x_imb = a1 * bimb * sig_dir  # positive when signal aligns with book
        conf_x_mom = a1 * m50 * sig_dir   # positive when signal aligns with momentum
        aligned_imb = 1.0 if sig_dir * bimb > 0 else 0.0
        aligned_mom = 1.0 if sig_dir * m50 > 0 else 0.0

        features_out[i] = [
            p1, p5, p10, a1, a5, a10, sig_dir, tier,
            signs_agree, strength, p_std, p_range, ratio_5_1, ratio_10_1,
            bimb, bd, ad, spr, depth_rat,
            v50, v200, v500, vol_trend,
            m20, m50, m200,
            vi_50, vi_200, vi_500, vi_trend,
            ti_50, ti_200, int_ratio,
            tod, minutes, is_first_30, is_last_30,
            conf_x_vol, conf_x_imb, conf_x_mom,
            aligned_imb, aligned_mom,
        ]
        meta_out[i] = [cur_ts, float(pred_idx), sig_dir]

    return features_out, targets, meta_out


def _extract_one_date(args) -> dict:
    """Worker function for parallel extraction."""
    mbo_path, pred_path, pred_stride, pred_window = args
    date_str = Path(mbo_path).stem.replace("_mbo_events", "")

    t0 = time.time()
    try:
        pred_data = np.load(str(pred_path), allow_pickle=True)
        predictions = pred_data["predictions"]
        labels = pred_data["labels"]

        features, targets, meta = extract_features_for_date(
            mbo_path=Path(mbo_path),
            predictions=predictions,
            labels=labels,
            pred_stride=pred_stride,
            pred_window=pred_window,
        )

        elapsed = time.time() - t0
        return {
            "date": date_str,
            "status": "ok",
            "n_samples": len(features),
            "features": features,
            "targets": targets,
            "meta": meta,
            "elapsed": elapsed,
        }
    except Exception as e:
        import traceback
        return {
            "date": date_str,
            "status": f"error: {e}\n{traceback.format_exc()}",
            "n_samples": 0,
            "features": np.zeros((0, N_FEATURES), dtype=np.float32),
            "targets": {f"{t}_{h}": np.zeros(0, dtype=np.float32)
                       for h in HORIZONS for t in TARGET_NAMES_PER_HORIZON},
            "meta": np.zeros((0, 3), dtype=np.float32),
            "elapsed": time.time() - t0,
        }


def build_date_pred_index(pred_dir: Path) -> Dict[str, Path]:
    """Build mapping: date_str -> pred_file_path."""
    index = {}

    # Per-date files: {date}_predictions.npz or {date}_oot_predictions.npz
    for pattern in ["*_predictions.npz", "*_oot_predictions.npz"]:
        for f in pred_dir.glob(pattern):
            stem = f.stem
            # Skip fold files
            if stem.startswith("fold_"):
                continue
            date = stem.replace("_oot_predictions", "").replace("_predictions", "")
            if len(date) == 8 and date.isdigit() and date not in index:
                index[date] = f

    # Walk-forward fold files: fold_XX_oot_predictions.npz
    for f in sorted(pred_dir.glob("fold_*_oot_predictions.npz")):
        try:
            data = np.load(str(f), allow_pickle=True)
            # Single-date fold files have a 'date' key
            if "date" in data:
                d_str = str(data["date"])
                if len(d_str) == 8 and d_str.isdigit() and d_str not in index:
                    index[d_str] = f
            elif "dates" in data:
                dates = data["dates"]
                preds = data["predictions"]
                labels_arr = data["labels"]
                if hasattr(dates, '__len__') and len(dates) > 0:
                    for d in np.unique(dates):
                        d_str = str(d)
                        if d_str not in index:
                            mask = dates == d
                            date_file = pred_dir / f"{d_str}_predictions.npz"
                            if not date_file.exists():
                                np.savez_compressed(
                                    str(date_file),
                                    predictions=preds[mask],
                                    labels=labels_arr[mask],
                                )
                            index[d_str] = date_file
        except Exception:
            pass

    return index


def load_all_data(
    data_dir: Path,
    pred_dir: Path,
    precomputed_dir: Optional[Path],
    n_workers: int = 8,
) -> Tuple[Dict[str, np.ndarray], Dict[str, dict], Dict[str, np.ndarray], List[str]]:
    """Load features and targets for all available dates."""

    # Build prediction index
    pred_index = build_date_pred_index(pred_dir)
    LOG.info(f"Prediction index: {len(pred_index)} dates")

    # Find matching MBO data files
    mbo_files = {}
    for d in sorted(data_dir.glob("*_mbo_events.npz")):
        date = d.stem.replace("_mbo_events", "")
        if date in pred_index:
            mbo_files[date] = d

    LOG.info(f"Dates with both MBO + predictions: {len(mbo_files)}")

    # Parallel extraction
    tasks = [
        (str(mbo_files[d]), str(pred_index[d]), PRED_STRIDE, PRED_WINDOW)
        for d in sorted(mbo_files.keys())
    ]

    features_by_date = {}
    targets_by_date = {}
    meta_by_date = {}
    all_dates = []
    total_samples = 0

    t0 = time.time()
    with ProcessPoolExecutor(max_workers=n_workers) as executor:
        futures = {executor.submit(_extract_one_date, t): t for t in tasks}
        for fut in as_completed(futures):
            result = fut.result()
            date = result["date"]
            if result["status"] == "ok" and result["n_samples"] > 0:
                features_by_date[date] = result["features"]
                targets_by_date[date] = result["targets"]
                meta_by_date[date] = result["meta"]
                all_dates.append(date)
                total_samples += result["n_samples"]
                LOG.info(f"  {date}: {result['n_samples']:,} samples in {result['elapsed']:.1f}s")
            elif result["n_samples"] == 0:
                LOG.warning(f"  {date}: 0 samples")
            else:
                LOG.error(f"  {date}: {result['status']}")

    all_dates.sort()
    LOG.info(f"Total: {total_samples:,} samples from {len(all_dates)} dates in {time.time()-t0:.1f}s")

    return features_by_date, targets_by_date, meta_by_date, all_dates


def train_lgbm_walk_forward(
    features_by_date: Dict[str, np.ndarray],
    targets_by_date: Dict[str, dict],
    meta_by_date: Dict[str, np.ndarray],
    all_dates: List[str],
    n_train_days: int = 40,
    n_oot_days: int = 5,
    output_dir: Path = Path("output/lgbm_exec_v1"),
) -> dict:
    """Walk-forward training with LightGBM."""

    output_dir.mkdir(parents=True, exist_ok=True)

    n_dates = len(all_dates)
    if n_dates < n_train_days + n_oot_days:
        LOG.error(f"Not enough dates: {n_dates} < {n_train_days} + {n_oot_days}")
        return {}

    # Generate walk-forward folds
    folds = []
    start = 0
    while start + n_train_days + n_oot_days <= n_dates:
        train_dates = all_dates[start:start + n_train_days]
        oot_dates = all_dates[start + n_train_days:start + n_train_days + n_oot_days]
        folds.append((train_dates, oot_dates))
        start += n_oot_days  # slide by OOT window

    LOG.info(f"Walk-forward: {len(folds)} folds ({n_train_days} train, {n_oot_days} OOT)")

    # Collect all OOT results
    all_oot_features = []
    all_oot_targets = {k: [] for k in list(targets_by_date[all_dates[0]].keys())}
    all_oot_meta = []
    all_oot_preds = {f"{target}": [] for target in ["profitable_1s", "profitable_5s", "profitable_10s",
                                                      "pnl_1s", "pnl_5s", "pnl_10s",
                                                      "mfe_1s", "mfe_5s", "mfe_10s"]}

    fold_results = []

    for fold_idx, (train_dates, oot_dates) in enumerate(folds):
        LOG.info(f"\n{'='*60}")
        LOG.info(f"FOLD {fold_idx}: Train [{train_dates[0]}..{train_dates[-1]}] "
                 f"Eval [{oot_dates[0]}..{oot_dates[-1]}]")
        LOG.info(f"{'='*60}")

        # Concatenate training data
        X_train = np.concatenate([features_by_date[d] for d in train_dates if d in features_by_date], axis=0)
        X_oot = np.concatenate([features_by_date[d] for d in oot_dates if d in features_by_date], axis=0)
        meta_oot = np.concatenate([meta_by_date[d] for d in oot_dates if d in meta_by_date], axis=0)

        if len(X_train) == 0 or len(X_oot) == 0:
            LOG.warning(f"Fold {fold_idx}: empty data, skipping")
            continue

        LOG.info(f"  Train: {len(X_train):,} samples | OOT: {len(X_oot):,} samples")

        fold_dir = output_dir / f"fold_{fold_idx}"
        fold_dir.mkdir(exist_ok=True)

        fold_preds = {}
        fold_importances = {}

        # Train models for each horizon and target type
        for horizon in HORIZONS:
            for target_type in ["profitable", "pnl", "mfe"]:
                target_key = f"{target_type}_{horizon}"

                y_train = np.concatenate([
                    targets_by_date[d][target_key]
                    for d in train_dates if d in targets_by_date
                ])
                y_oot = np.concatenate([
                    targets_by_date[d][target_key]
                    for d in oot_dates if d in targets_by_date
                ])

                if target_type == "profitable":
                    # Binary classification
                    params = {
                        "objective": "binary",
                        "metric": "auc",
                        "learning_rate": 0.05,
                        "num_leaves": 63,
                        "max_depth": 7,
                        "min_child_samples": 200,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.1,
                        "reg_lambda": 1.0,
                        "verbose": -1,
                        "n_jobs": 8,
                        "seed": 42 + fold_idx,
                    }

                    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
                    dval = lgb.Dataset(X_oot, label=y_oot, feature_name=FEATURE_NAMES, reference=dtrain)

                    model = lgb.train(
                        params,
                        dtrain,
                        num_boost_round=500,
                        valid_sets=[dval],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=30),
                            lgb.log_evaluation(period=100),
                        ],
                    )

                    preds = model.predict(X_oot)

                else:
                    # Regression
                    params = {
                        "objective": "regression",
                        "metric": "mae",
                        "learning_rate": 0.05,
                        "num_leaves": 63,
                        "max_depth": 7,
                        "min_child_samples": 200,
                        "subsample": 0.8,
                        "colsample_bytree": 0.8,
                        "reg_alpha": 0.1,
                        "reg_lambda": 1.0,
                        "verbose": -1,
                        "n_jobs": 8,
                        "seed": 42 + fold_idx,
                    }

                    dtrain = lgb.Dataset(X_train, label=y_train, feature_name=FEATURE_NAMES)
                    dval = lgb.Dataset(X_oot, label=y_oot, feature_name=FEATURE_NAMES, reference=dtrain)

                    model = lgb.train(
                        params,
                        dtrain,
                        num_boost_round=500,
                        valid_sets=[dval],
                        callbacks=[
                            lgb.early_stopping(stopping_rounds=30),
                            lgb.log_evaluation(period=100),
                        ],
                    )

                    preds = model.predict(X_oot)

                fold_preds[target_key] = preds
                fold_importances[target_key] = dict(zip(FEATURE_NAMES, model.feature_importance(importance_type="gain")))

                # Save model
                model.save_model(str(fold_dir / f"model_{target_key}.txt"))

        # Evaluate this fold
        fold_result = evaluate_fold(
            fold_idx, X_oot, fold_preds, targets_by_date, oot_dates, meta_oot,
            fold_importances, fold_dir,
        )
        fold_results.append(fold_result)

        # Accumulate OOT predictions
        all_oot_features.append(X_oot)
        all_oot_meta.append(meta_oot)
        for key in all_oot_preds:
            if key in fold_preds:
                all_oot_preds[key].append(fold_preds[key])
        for key in all_oot_targets:
            y = np.concatenate([targets_by_date[d][key] for d in oot_dates if d in targets_by_date])
            all_oot_targets[key].append(y)

    # Concatenate all OOT results
    LOG.info(f"\n{'='*70}")
    LOG.info(f"CONCAT WALK-FORWARD RESULTS ({len(folds)} folds)")
    LOG.info(f"{'='*70}")

    if not all_oot_features:
        LOG.error("No folds completed!")
        return {}

    X_all = np.concatenate(all_oot_features)
    meta_all = np.concatenate(all_oot_meta)
    concat_preds = {k: np.concatenate(v) for k, v in all_oot_preds.items() if v}
    concat_targets = {k: np.concatenate(v) for k, v in all_oot_targets.items() if v}

    # Compute concat metrics
    concat_results = evaluate_concat(
        X_all, concat_preds, concat_targets, meta_all,
        fold_results, output_dir,
    )

    return concat_results


def evaluate_fold(
    fold_idx: int,
    X_oot: np.ndarray,
    preds: dict,
    targets_by_date: dict,
    oot_dates: list,
    meta_oot: np.ndarray,
    importances: dict,
    fold_dir: Path,
) -> dict:
    """Evaluate a single fold's predictions."""
    from scipy.stats import spearmanr

    result = {"fold": fold_idx, "n_samples": len(X_oot)}

    y_oot = {}
    for key in preds:
        y = np.concatenate([targets_by_date[d][key] for d in oot_dates if d in targets_by_date])
        y_oot[key] = y

    for horizon in HORIZONS:
        # Classification metrics
        prob_key = f"profitable_{horizon}"
        if prob_key in preds and prob_key in y_oot:
            prob_preds = preds[prob_key]
            prob_actual = y_oot[prob_key]

            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(prob_actual, prob_preds)
            except:
                auc = 0.5

            # Threshold analysis
            for thresh in [0.5, 0.55, 0.6, 0.65, 0.7]:
                selected = prob_preds >= thresh
                n_sel = selected.sum()
                if n_sel > 0:
                    wr = prob_actual[selected].mean()
                    pnl_key = f"pnl_{horizon}"
                    if pnl_key in y_oot:
                        avg_pnl = y_oot[pnl_key][selected].mean()
                        total_pnl = y_oot[pnl_key][selected].sum()
                    else:
                        avg_pnl = 0.0
                        total_pnl = 0.0
                    LOG.info(f"  Fold {fold_idx} | {horizon} @{thresh:.2f}: "
                             f"n={n_sel:,} WR={wr:.3f} AvgPnL={avg_pnl:+.3f}tk TotalPnL={total_pnl:+.1f}tk AUC={auc:.4f}")

        # Regression metrics
        pnl_key = f"pnl_{horizon}"
        if pnl_key in preds and pnl_key in y_oot:
            sp, _ = spearmanr(preds[pnl_key], y_oot[pnl_key])
            result[f"spearman_pnl_{horizon}"] = sp
            LOG.info(f"  Fold {fold_idx} | {horizon} P&L Spearman: {sp:.4f}")

        mfe_key = f"mfe_{horizon}"
        if mfe_key in preds and mfe_key in y_oot:
            sp, _ = spearmanr(preds[mfe_key], y_oot[mfe_key])
            result[f"spearman_mfe_{horizon}"] = sp
            LOG.info(f"  Fold {fold_idx} | {horizon} MFE Spearman: {sp:.4f}")

    # Feature importance for this fold
    LOG.info(f"\n  Fold {fold_idx} Feature Importance (profitable_10s, top 15):")
    if "profitable_10s" in importances:
        imp = importances["profitable_10s"]
        sorted_imp = sorted(imp.items(), key=lambda x: -x[1])
        for fname, fval in sorted_imp[:15]:
            LOG.info(f"    {fname:30s}: {fval:.1f}")

    # Save fold results
    with open(fold_dir / "fold_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def evaluate_concat(
    X_all: np.ndarray,
    preds: dict,
    targets: dict,
    meta_all: np.ndarray,
    fold_results: list,
    output_dir: Path,
) -> dict:
    """Evaluate concatenated OOT results across all folds."""
    from scipy.stats import spearmanr

    result = {"n_samples": len(X_all), "n_folds": len(fold_results)}

    LOG.info(f"\nTotal OOT samples: {len(X_all):,}")

    for horizon in HORIZONS:
        LOG.info(f"\n--- {horizon} HORIZON ---")

        # Classification
        prob_key = f"profitable_{horizon}"
        if prob_key in preds and prob_key in targets:
            prob_p = preds[prob_key]
            prob_a = targets[prob_key]

            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(prob_a, prob_p)
            except:
                auc = 0.5

            result[f"auc_{horizon}"] = auc
            result[f"base_rate_{horizon}"] = float(prob_a.mean())
            LOG.info(f"  AUC: {auc:.4f} | Base rate: {prob_a.mean():.3f}")

            # Threshold analysis with trading simulation
            LOG.info(f"  Threshold analysis (simulated trading):")
            LOG.info(f"  {'Thresh':>7s}  {'Select%':>8s}  {'Trades':>7s}  {'WR':>6s}  {'AvgPnL':>8s}  {'TotalPnL':>10s}  {'PF':>6s}")

            pnl_key = f"pnl_{horizon}"
            pnl_actual = targets.get(pnl_key, np.zeros(len(X_all)))

            for thresh in [0.45, 0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80]:
                selected = prob_p >= thresh
                n_sel = int(selected.sum())
                if n_sel > 10:
                    sel_pct = n_sel / len(X_all) * 100
                    wr = float(prob_a[selected].mean())
                    avg_pnl = float(pnl_actual[selected].mean())
                    total_pnl = float(pnl_actual[selected].sum())
                    winners = pnl_actual[selected] > 0
                    losers = pnl_actual[selected] <= 0
                    gross_profit = float(pnl_actual[selected][winners].sum()) if winners.any() else 0
                    gross_loss = abs(float(pnl_actual[selected][losers].sum())) if losers.any() else 1e-8
                    pf = gross_profit / (gross_loss + 1e-8)

                    LOG.info(f"    {thresh:5.2f}    {sel_pct:6.1f}%  {n_sel:7,}  {wr:.3f}  {avg_pnl:+7.3f}tk  {total_pnl:+9.1f}tk  {pf:.2f}")

            # Top-N% analysis
            LOG.info(f"\n  Top decile analysis:")
            sorted_idx = np.argsort(-prob_p)
            for pct in [1, 5, 10, 20]:
                n_top = max(1, len(X_all) * pct // 100)
                top_idx = sorted_idx[:n_top]
                wr = float(prob_a[top_idx].mean())
                avg_pnl = float(pnl_actual[top_idx].mean())
                total_pnl = float(pnl_actual[top_idx].sum())
                winners = pnl_actual[top_idx] > 0
                losers = pnl_actual[top_idx] <= 0
                gp = float(pnl_actual[top_idx][winners].sum()) if winners.any() else 0
                gl = abs(float(pnl_actual[top_idx][losers].sum())) if losers.any() else 1e-8
                pf = gp / (gl + 1e-8)

                LOG.info(f"    Top {pct:2d}%: {n_top:6,} trades, WR={wr:.3f}, "
                         f"AvgPnL={avg_pnl:+.3f}tk, TotalPnL={total_pnl:+.1f}tk, PF={pf:.2f}")
                result[f"top{pct}pct_wr_{horizon}"] = wr
                result[f"top{pct}pct_pnl_{horizon}"] = avg_pnl
                result[f"top{pct}pct_pf_{horizon}"] = pf

        # Regression correlation
        pnl_key = f"pnl_{horizon}"
        if pnl_key in preds and pnl_key in targets:
            sp, _ = spearmanr(preds[pnl_key], targets[pnl_key])
            result[f"spearman_pnl_{horizon}"] = sp
            LOG.info(f"\n  P&L Spearman correlation: {sp:.4f}")

        mfe_key = f"mfe_{horizon}"
        if mfe_key in preds and mfe_key in targets:
            sp, _ = spearmanr(preds[mfe_key], targets[mfe_key])
            result[f"spearman_mfe_{horizon}"] = sp
            LOG.info(f"  MFE Spearman correlation: {sp:.4f}")

    # Aggregate feature importance across folds
    LOG.info(f"\n{'='*60}")
    LOG.info(f"AGGREGATE FEATURE IMPORTANCE (profitable_10s)")
    LOG.info(f"{'='*60}")

    # Check signal confidence vs outcome relationship in RAW data
    LOG.info(f"\n{'='*60}")
    LOG.info(f"SANITY CHECK: Signal Confidence → Outcome (Raw Data)")
    LOG.info(f"{'='*60}")

    abs_pred_1s = X_all[:, FEATURE_NAMES.index("abs_pred_1s")]
    for horizon in HORIZONS:
        pnl_key = f"pnl_{horizon}"
        prof_key = f"profitable_{horizon}"
        if pnl_key in targets and prof_key in targets:
            sp, _ = spearmanr(abs_pred_1s, targets[pnl_key])
            LOG.info(f"\n  {horizon}: Spearman(abs_pred_1s, P&L) = {sp:.4f}")

            # Binned analysis
            pctiles = [0, 25, 50, 75, 90, 95, 99, 100]
            thresholds = np.percentile(abs_pred_1s, pctiles)
            LOG.info(f"  {'Bin':>10s} | {'N':>7s} | {'AvgPnL':>8s} | {'WR':>6s} | {'AvgConf':>8s}")
            LOG.info(f"  {'-'*55}")
            for i in range(len(pctiles) - 1):
                mask = (abs_pred_1s >= thresholds[i]) & (abs_pred_1s < thresholds[i+1] + 1e-8)
                if i == len(pctiles) - 2:
                    mask = abs_pred_1s >= thresholds[i]
                n = mask.sum()
                if n > 0:
                    avg_pnl = targets[pnl_key][mask].mean()
                    wr = targets[prof_key][mask].mean()
                    avg_conf = abs_pred_1s[mask].mean()
                    label = f"{pctiles[i]}-{pctiles[i+1]}%"
                    LOG.info(f"  {label:>10s} | {n:7,} | {avg_pnl:+7.3f} | {wr:.3f} | {avg_conf:7.4f}")

    # Side analysis (short vs long)
    LOG.info(f"\n{'='*60}")
    LOG.info(f"SIDE ANALYSIS: Long vs Short Performance")
    LOG.info(f"{'='*60}")

    sig_dir = X_all[:, FEATURE_NAMES.index("signal_direction")]
    for horizon in HORIZONS:
        pnl_key = f"pnl_{horizon}"
        prof_key = f"profitable_{horizon}"
        if pnl_key in targets:
            long_mask = sig_dir > 0
            short_mask = sig_dir < 0
            n_long = long_mask.sum()
            n_short = short_mask.sum()

            if n_long > 0:
                wr_l = targets[prof_key][long_mask].mean()
                pnl_l = targets[pnl_key][long_mask].mean()
            else:
                wr_l, pnl_l = 0, 0

            if n_short > 0:
                wr_s = targets[prof_key][short_mask].mean()
                pnl_s = targets[pnl_key][short_mask].mean()
            else:
                wr_s, pnl_s = 0, 0

            LOG.info(f"  {horizon}: Long n={n_long:,} WR={wr_l:.3f} PnL={pnl_l:+.3f} | "
                     f"Short n={n_short:,} WR={wr_s:.3f} PnL={pnl_s:+.3f}")

    # Save results
    with open(output_dir / "concat_results.json", "w") as f:
        json.dump(result, f, indent=2, default=str)

    LOG.info(f"\nResults saved to {output_dir}")
    return result


def main():
    parser = argparse.ArgumentParser(description="LightGBM Execution Predictor")
    parser.add_argument("--data-dir", type=str,
                       default=str(LVL3_ROOT / "data/processed/mbo_events_smart_v3"))
    parser.add_argument("--pred-dir", type=str,
                       default=str(LVL3_ROOT / "output/cnn_mamba_v2_all_oot"))
    parser.add_argument("--precomputed-dir", type=str,
                       default=str(LVL3_ROOT / "data/precomputed_obs"))
    parser.add_argument("--output-dir", type=str,
                       default=str(LVL3_ROOT / "output/lgbm_exec_v1"))
    parser.add_argument("--n-train-days", type=int, default=40)
    parser.add_argument("--n-oot-days", type=int, default=5)
    parser.add_argument("--n-workers", type=int, default=8)
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler(args.output_dir + ".log", mode="w"),
        ],
    )

    LOG.info("=" * 70)
    LOG.info("LightGBM Execution Predictor — Walk-Forward Training")
    LOG.info("=" * 70)
    LOG.info(f"Data dir:       {args.data_dir}")
    LOG.info(f"Pred dir:       {args.pred_dir}")
    LOG.info(f"Output dir:     {args.output_dir}")
    LOG.info(f"Train days:     {args.n_train_days}")
    LOG.info(f"OOT days:       {args.n_oot_days}")
    LOG.info(f"Workers:        {args.n_workers}")
    LOG.info(f"N features:     {N_FEATURES}")
    LOG.info(f"Commission:     {COMMISSION_TICKS:.3f} ticks RT")

    data_dir = Path(args.data_dir)
    pred_dir = Path(args.pred_dir)
    precomputed_dir = Path(args.precomputed_dir) if args.precomputed_dir else None
    output_dir = Path(args.output_dir)

    # Load all data
    features_by_date, targets_by_date, meta_by_date, all_dates = load_all_data(
        data_dir=data_dir,
        pred_dir=pred_dir,
        precomputed_dir=precomputed_dir,
        n_workers=args.n_workers,
    )

    if not all_dates:
        LOG.error("No data loaded!")
        return

    # Train walk-forward
    results = train_lgbm_walk_forward(
        features_by_date=features_by_date,
        targets_by_date=targets_by_date,
        meta_by_date=meta_by_date,
        all_dates=all_dates,
        n_train_days=args.n_train_days,
        n_oot_days=args.n_oot_days,
        output_dir=output_dir,
    )

    LOG.info("\n" + "=" * 70)
    LOG.info("TRAINING COMPLETE")
    LOG.info("=" * 70)


if __name__ == "__main__":
    main()
