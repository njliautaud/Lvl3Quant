#!/usr/bin/env python3
"""
Longer-Horizon Directional Model v3 — Two-Stage Ensemble + Ranking Loss
========================================================================

Fixes v2 instability (IC_Sharpe -0.10 at 1h) via:
  1. LightGBM base: stable tabular learner on 15-min bar features
  2. Residual GRU: small NN captures temporal patterns LightGBM misses
  3. ListMLE ranking loss: optimizes prediction ORDERING, not MSE
  4. Side-specific heads: separate long/short processing
  5. Adaptive confidence: learned signal dampening under uncertainty
  6. 10-day validation window (vs v2's 5-day) for more stable fold eval

Architecture:
  Stage 1 — LightGBM trained on each fold, OOF predictions collected
  Stage 2 — Small GRU (hidden=64) predicts residual (actual - LightGBM pred)
  Final   — alpha * LightGBM + (1-alpha) * NN_residual, alpha tuned on val

Walk-Forward: 60d train, 10d val, slide 5d — SLIDING only (HC #0)
Horizons: 15min (primary), 1h (secondary), 2h (test)
Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission)

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python \\
      alpha_discovery/longer_horizon_v3_ensemble.py --device cuda --epochs 30

Author: Claude (autonomous research)
"""

import argparse
import gc
import json
import logging
import math
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
QUEUE_FEATURE_DIR = ROOT / "data" / "queue_augmented_features"
OUTPUT_DIR = ROOT / "output" / "longer_horizon_v3_ensemble"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [LH-v3-Ens] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "longer_horizon_v3_ensemble.log")),
    ],
)
log = logging.getLogger("LH-v3-Ens")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376
COST_RT_TICKS = 2.376  # market entry + market exit + RT commission

MIN_EDGE_TICKS = {
    "15min": 2.0,
    "1h": 3.0,
    "2h": 4.0,
}

HORIZON_BARS = {
    "15min": 1,
    "1h": 4,
    "2h": 8,
}

# ─────────────────────────────────────────────
#  DEFERRED IMPORTS
# ─────────────────────────────────────────────
torch = None
nn = None
lgb = None


def _import_torch():
    global torch, nn
    if torch is not None:
        return
    import torch as _torch
    import torch.nn as _nn
    torch = _torch
    nn = _nn


def _import_lightgbm():
    global lgb
    if lgb is not None:
        return
    import lightgbm as _lgb
    lgb = _lgb


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1: DATA LOADING (reused from v2, with fallback)
# ═══════════════════════════════════════════════════════════════════


def load_all_minute_bars(min_date: str = "20250714") -> pd.DataFrame:
    """Load all minute bar parquets into a single DataFrame."""
    files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
    frames = []
    for f in files:
        if f.stem < min_date:
            continue
        try:
            df = pd.read_parquet(f)
            df["date"] = f.stem
            frames.append(df)
        except Exception as e:
            log.warning(f"Skip {f.stem}: {e}")

    if not frames:
        raise RuntimeError(f"No minute bar files found in {MINUTE_BAR_DIR}")

    combined = pd.concat(frames, ignore_index=True)
    combined["ts_minute"] = pd.to_datetime(combined["ts_minute"], utc=True)
    combined = combined.sort_values("ts_minute").reset_index(drop=True)
    log.info(
        f"Loaded {len(combined):,} minute bars across {len(frames)} days "
        f"({frames[0]['date'].iloc[0]} -> {frames[-1]['date'].iloc[0]})"
    )
    return combined


def load_queue_features() -> Optional[pd.DataFrame]:
    """Load queue-augmented tick features if available."""
    if not QUEUE_FEATURE_DIR.exists():
        log.info("No queue_augmented_features directory — skipping")
        return None

    files = sorted(QUEUE_FEATURE_DIR.glob("features_*.parquet"))
    if not files:
        log.info("No queue feature files found — skipping")
        return None

    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            date_str = f.stem.replace("features_", "")
            df["date"] = date_str
            frames.append(df)
        except Exception as e:
            log.warning(f"Skip queue file {f.stem}: {e}")

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded queue features for {len(frames)} days, {len(combined):,} rows")
    return combined


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: 15-MINUTE BAR AGGREGATION + FEATURES (from v2)
# ═══════════════════════════════════════════════════════════════════


def _safe_polyfit_slope(arr: np.ndarray) -> float:
    """Linear regression slope, returns 0 on failure."""
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def aggregate_to_15min(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute bars into 15-minute bars with microstructure features."""
    df = minute_df.copy()
    df["bar_15m"] = df["ts_minute"].dt.floor("15min")
    df["return_1m"] = df.groupby("date")["close"].pct_change()
    df["abs_ofi"] = df["ofi_1min"].abs()
    sv_std = df.groupby("date")["signed_volume"].transform("std").replace(0, 1)
    df["sv_zscore"] = df["signed_volume"] / sv_std
    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["close"].clip(lower=1)

    records = []
    for (date_str, bar_key), grp in df.groupby(["date", "bar_15m"]):
        if len(grp) < 3:
            continue

        close_arr = grp["close"].values
        vol_arr = grp["volume"].values
        ofi_arr = grp["ofi_1min"].values
        sv_arr = grp["signed_volume"].values
        ret_arr = grp["return_1m"].fillna(0).values
        spread_arr = grp["spread_mean"].values
        tc_arr = grp["trade_count"].values

        rec = {
            "date": date_str,
            "bar_15m": bar_key,
            "ts": grp["ts_minute"].iloc[0],
            # Price action
            "open": close_arr[0],
            "high": close_arr.max(),
            "low": close_arr.min(),
            "close": close_arr[-1],
            "return_15m": (close_arr[-1] / close_arr[0] - 1) if close_arr[0] > 0 else 0,
            "range_ticks": (close_arr.max() - close_arr.min()) / 0.25,
            "close_position": (
                (close_arr[-1] - close_arr.min())
                / max(close_arr.max() - close_arr.min(), 0.25)
            ),
            # Volume profile
            "total_volume": vol_arr.sum(),
            "avg_volume": vol_arr.mean(),
            "volume_trend": _safe_polyfit_slope(vol_arr),
            "volume_concentration": vol_arr.max() / max(vol_arr.mean(), 1),
            # OFI
            "ofi_sum": ofi_arr.sum(),
            "ofi_mean": ofi_arr.mean(),
            "ofi_std": ofi_arr.std() if len(ofi_arr) > 1 else 0,
            "ofi_trend": _safe_polyfit_slope(ofi_arr),
            "ofi_consistency": (
                np.mean(np.sign(ofi_arr) == np.sign(ofi_arr.sum()))
                if ofi_arr.sum() != 0
                else 0.5
            ),
            "ofi_late_vs_early": (
                ofi_arr[len(ofi_arr) // 2 :].sum() - ofi_arr[: len(ofi_arr) // 2].sum()
            ),
            # Signed volume
            "signed_volume_sum": sv_arr.sum(),
            "signed_volume_ratio": sv_arr.sum() / max(vol_arr.sum(), 1),
            "buy_volume_frac": float(np.sum(sv_arr[sv_arr > 0])) / max(vol_arr.sum(), 1),
            "sell_volume_frac": float(-np.sum(sv_arr[sv_arr < 0])) / max(vol_arr.sum(), 1),
            # Sweep proxy
            "sweep_minutes": int(np.sum(np.abs(grp["sv_zscore"].values) > 2)),
            "max_sweep_intensity": float(np.abs(grp["sv_zscore"].values).max()),
            "sweep_direction": float(
                np.sign(sv_arr[np.abs(grp["sv_zscore"].values).argmax()])
            )
            if len(sv_arr) > 0
            else 0.0,
            # Spread & liquidity
            "spread_mean": spread_arr.mean(),
            "spread_max": spread_arr.max(),
            "spread_trend": _safe_polyfit_slope(spread_arr),
            # Trade intensity
            "trade_count_sum": tc_arr.sum(),
            "trade_count_trend": _safe_polyfit_slope(tc_arr),
            # Volatility
            "realized_vol": float(np.std(ret_arr) * np.sqrt(252 * 26)) if len(ret_arr) > 1 else 0,
            "vol_of_vol": float(np.std(np.abs(ret_arr))) if len(ret_arr) > 1 else 0,
            "up_vol": float(np.std(ret_arr[ret_arr > 0])) if np.sum(ret_arr > 0) > 1 else 0,
            "down_vol": float(np.std(ret_arr[ret_arr < 0])) if np.sum(ret_arr < 0) > 1 else 0,
            # VWAP
            "vwap_dev_mean": grp["vwap_dev"].mean(),
            "vwap_dev_trend": _safe_polyfit_slope(grp["vwap_dev"].values),
        }

        # Vol asymmetry
        if rec["up_vol"] > 0 and rec["down_vol"] > 0:
            rec["vol_asymmetry"] = rec["down_vol"] / rec["up_vol"] - 1
        elif rec["down_vol"] > 0:
            rec["vol_asymmetry"] = 1.0
        elif rec["up_vol"] > 0:
            rec["vol_asymmetry"] = -1.0
        else:
            rec["vol_asymmetry"] = 0.0

        records.append(rec)

    result = pd.DataFrame(records)
    log.info(f"Aggregated {len(result):,} 15-min bars with {len(result.columns)} columns")
    return result


def add_queue_features_to_bars(
    bars_df: pd.DataFrame, queue_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Merge queue-augmented tick features into 15min bars."""
    if queue_df is None or queue_df.empty:
        log.info("No queue features to merge")
        return bars_df

    queue_cols_mean = [
        "bid_qty_at_touch", "ask_qty_at_touch", "top_imbalance",
        "bid_q_ahead_p50", "ask_q_ahead_p50", "microprice_offset_ticks",
    ]
    queue_cols_sum = ["ofi_1s", "ofi_5s", "ofi_10s"]
    rate_cols = [
        "bid_add_rate_1s", "ask_add_rate_1s",
        "bid_cancel_rate_1s", "ask_cancel_rate_1s",
    ]

    available_mean = [c for c in queue_cols_mean if c in queue_df.columns]
    available_sum = [c for c in queue_cols_sum if c in queue_df.columns]
    available_rate = [c for c in rate_cols if c in queue_df.columns]

    if not available_mean and not available_sum and not available_rate:
        log.warning("Queue dataframe has no expected columns — skipping")
        return bars_df

    ts_col = None
    for candidate in ["ts", "timestamp", "ts_event"]:
        if candidate in queue_df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        log.warning("Queue dataframe has no timestamp column — skipping merge")
        return bars_df

    queue_df[ts_col] = pd.to_datetime(queue_df[ts_col], utc=True)
    queue_df["bar_15m"] = queue_df[ts_col].dt.floor("15min")

    agg_dict = {}
    for c in available_mean:
        agg_dict[c] = ["mean", "std"]
    for c in available_sum:
        agg_dict[c] = ["sum"]
    for c in available_rate:
        agg_dict[c] = ["mean"]

    agg = queue_df.groupby(["date", "bar_15m"]).agg(agg_dict)
    agg.columns = [f"q_{c}_{stat}" for c, stat in agg.columns]
    agg = agg.reset_index()

    if "bid_cancel_rate_1s" in queue_df.columns and "bid_add_rate_1s" in queue_df.columns:
        per_bar = queue_df.groupby(["date", "bar_15m"]).agg(
            bid_cancel_mean=("bid_cancel_rate_1s", "mean"),
            bid_add_mean=("bid_add_rate_1s", "mean"),
            ask_cancel_mean=("ask_cancel_rate_1s", "mean"),
            ask_add_mean=("ask_add_rate_1s", "mean"),
        ).reset_index()
        per_bar["q_bid_toxicity"] = per_bar["bid_cancel_mean"] / per_bar["bid_add_mean"].clip(lower=1e-6)
        per_bar["q_ask_toxicity"] = per_bar["ask_cancel_mean"] / per_bar["ask_add_mean"].clip(lower=1e-6)
        agg = agg.merge(
            per_bar[["date", "bar_15m", "q_bid_toxicity", "q_ask_toxicity"]],
            on=["date", "bar_15m"], how="left",
        )

    if "ofi_1s" in queue_df.columns:
        def _ofi_slope(g):
            vals = g["ofi_1s"].values
            return _safe_polyfit_slope(vals)
        ofi_slope = queue_df.groupby(["date", "bar_15m"]).apply(_ofi_slope).reset_index()
        ofi_slope.columns = ["date", "bar_15m", "q_ofi_slope"]
        agg = agg.merge(ofi_slope, on=["date", "bar_15m"], how="left")

    n_before = len(bars_df)
    bars_df = bars_df.merge(agg, on=["date", "bar_15m"], how="left")
    n_queue_dates = bars_df.dropna(subset=[agg.columns[2]]).date.nunique() if len(agg.columns) > 2 else 0
    log.info(
        f"Merged queue features: {len(agg.columns)-2} new columns, "
        f"available for {n_queue_dates} dates"
    )
    return bars_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: ROLLING CONTEXT + MOMENTUM FEATURES (from v2)
# ═══════════════════════════════════════════════════════════════════


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-bar rolling features using ONLY past data (causal)."""
    df = df.sort_values("ts").reset_index(drop=True)

    for w in [4, 8, 16, 32]:
        roll_mean = df["ofi_sum"].rolling(w, min_periods=1).mean()
        roll_std = df["ofi_sum"].rolling(w, min_periods=2).std().fillna(1).replace(0, 1)
        df[f"ofi_zscore_{w}bar"] = (df["ofi_sum"] - roll_mean) / roll_std

        vol_ma = df["total_volume"].rolling(w, min_periods=1).mean()
        df[f"vol_rel_{w}bar"] = df["total_volume"] / vol_ma.clip(lower=1)

        if "sweep_minutes" in df.columns:
            df[f"sweep_pct_{w}bar"] = (
                df["sweep_minutes"].rolling(w, min_periods=1).sum() / (w * 15)
            )

    for bars, label in [(4, "1h"), (8, "2h"), (16, "4h")]:
        df[f"ret_{label}_lb"] = df["close"].pct_change(bars)

    for w in [4, 8, 16]:
        df[f"rvol_{w}bar"] = df["return_15m"].rolling(w, min_periods=2).std()

    df["intraday_cum_ofi"] = df.groupby("date")["ofi_sum"].cumsum()
    df["intraday_cum_sv"] = df.groupby("date")["signed_volume_sum"].cumsum()

    df["ofi_sign_flip"] = (
        np.sign(df["ofi_sum"]) != np.sign(df["ofi_sum"].shift(1))
    ).astype(np.float32)

    df["absorption"] = df["total_volume"] / df["range_ticks"].clip(lower=1)

    day_stats = (
        df.groupby("date")
        .agg(
            day_ofi=("ofi_sum", "sum"),
            day_sv=("signed_volume_sum", "sum"),
            day_ret=("return_15m", "sum"),
            day_vol=("realized_vol", "mean"),
        )
        .reset_index()
    )
    day_stats["prev_day_ofi"] = day_stats["day_ofi"].shift(1)
    day_stats["prev_day_sv"] = day_stats["day_sv"].shift(1)
    day_stats["prev_day_ret"] = day_stats["day_ret"].shift(1)

    df = df.merge(
        day_stats[["date", "prev_day_ofi", "prev_day_sv", "prev_day_ret"]],
        on="date", how="left",
    )

    # Time-of-day encoding (sin/cos of session progress)
    session_start_hour = 13.5  # 13:30 UTC (09:30 ET)
    session_end_hour = 20.0    # 20:00 UTC (16:00 ET)
    session_len = session_end_hour - session_start_hour

    hour_frac = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    progress = ((hour_frac - session_start_hour) / session_len).clip(0, 1)
    df["tod_sin"] = np.sin(2 * np.pi * progress)
    df["tod_cos"] = np.cos(2 * np.pi * progress)
    df["tod_progress"] = progress.values

    df["bars_since_open"] = df.groupby("date").cumcount()

    log.info(f"Added rolling features: {len(df.columns)} total columns")
    return df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: FORWARD LABELS (STRICT NO-LEAKAGE)
# ═══════════════════════════════════════════════════════════════════


def add_forward_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward return labels per horizon with overnight gap protection."""
    df = df.sort_values("ts").reset_index(drop=True)

    for label, n_bars in HORIZON_BARS.items():
        fwd_close = df["close"].shift(-n_bars)
        fwd_return = fwd_close / df["close"] - 1
        fwd_ticks = (fwd_close - df["close"]) / 0.25

        # Null out overnight gaps
        ts_now = df["ts"].values
        ts_fwd = df["ts"].shift(-n_bars).values
        for i in range(len(df) - n_bars):
            if pd.isna(ts_fwd[i]):
                continue
            diff_s = (pd.Timestamp(ts_fwd[i]) - pd.Timestamp(ts_now[i])).total_seconds()
            if diff_s > 6 * 3600:
                fwd_return.iloc[i] = np.nan
                fwd_ticks.iloc[i] = np.nan

        df[f"fwd_return_{label}"] = fwd_return
        df[f"fwd_ticks_{label}"] = fwd_ticks

        min_ticks = MIN_EDGE_TICKS.get(label, 3.0)
        df[f"direction_{label}"] = 0
        df.loc[fwd_ticks > min_ticks, f"direction_{label}"] = 1
        df.loc[fwd_ticks < -min_ticks, f"direction_{label}"] = -1

        df[f"trade_quality_{label}"] = (fwd_ticks.abs() > 2 * COST_RT_TICKS).astype(np.float32)

    log.info(f"Added forward labels for horizons: {list(HORIZON_BARS.keys())}")
    return df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: LEAKAGE AUDIT
# ═══════════════════════════════════════════════════════════════════


def leakage_audit(
    df: pd.DataFrame,
    train_dates: List[str],
    val_dates: List[str],
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Explicit leakage audit. Returns dict of check results."""
    results = {}

    train_set = set(train_dates)
    val_set = set(val_dates)
    overlap = train_set & val_set
    results["date_overlap"] = len(overlap) == 0
    if overlap:
        log.error(f"LEAKAGE: Train/val date overlap: {overlap}")

    max_train = max(train_dates)
    min_val = min(val_dates)
    results["temporal_order"] = max_train < min_val
    if not results["temporal_order"]:
        log.error(f"LEAKAGE: Max train date {max_train} >= min val date {min_val}")

    fwd_leak = [c for c in feature_cols if c.startswith(("fwd_", "direction_", "trade_quality_"))]
    results["no_forward_features"] = len(fwd_leak) == 0
    if fwd_leak:
        log.error(f"LEAKAGE: Forward-looking columns in features: {fwd_leak}")

    price_cols = [c for c in feature_cols if c in ("close", "high", "low", "open")]
    results["no_raw_price_features"] = len(price_cols) == 0
    if price_cols:
        log.warning(f"WARNING: Raw price columns in features: {price_cols}")

    all_passed = all(results.values())
    results["all_passed"] = all_passed
    if all_passed:
        log.info("Leakage audit PASSED")
    else:
        log.error(f"Leakage audit FAILED: {results}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: FEATURE COLUMN SELECTION
# ═══════════════════════════════════════════════════════════════════


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select feature columns, excluding labels, metadata, raw prices."""
    exclude_prefixes = (
        "fwd_", "direction_", "trade_quality_",
        "date", "ts", "bar_15m",
    )
    raw_price_cols = {"open", "high", "low", "close"}

    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if c in raw_price_cols:
            continue
        if df[c].dtype in (np.float64, np.float32, np.int64, np.int32, np.float16, np.int16):
            cols.append(c)
    return cols


# ═══════════════════════════════════════════════════════════════════
#  SECTION 7: STAGE 1 — LightGBM BASE MODEL
# ═══════════════════════════════════════════════════════════════════


def train_lgbm_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    horizon: str,
    fold_idx: int,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Train LightGBM for one fold+horizon. Returns model, train OOF preds, val preds."""
    _import_lightgbm()

    # Remove NaN targets
    train_valid = ~np.isnan(y_train)
    val_valid = ~np.isnan(y_val)

    if train_valid.sum() < 50 or val_valid.sum() < 10:
        log.warning(f"  LGBM fold {fold_idx} {horizon}: too few valid samples "
                    f"(train={train_valid.sum()}, val={val_valid.sum()}) — skip")
        return None, np.full(len(y_train), np.nan), np.full(len(y_val), np.nan)

    X_tr = X_train[train_valid]
    y_tr = y_train[train_valid]
    X_v = X_val[val_valid]
    y_v = y_val[val_valid]

    params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_child_samples": 30,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 0.1,
        "lambda_l2": 1.0,
        "max_depth": 8,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42 + fold_idx,
    }

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    val_data = lgb.Dataset(X_v, label=y_v, feature_name=feature_names, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # Predict on full arrays (NaN positions will be overwritten)
    train_preds = np.full(len(y_train), np.nan)
    val_preds = np.full(len(y_val), np.nan)

    train_preds[train_valid] = model.predict(X_tr, num_iteration=model.best_iteration)
    val_preds[val_valid] = model.predict(X_v, num_iteration=model.best_iteration)

    # Quick IC check
    p_v = val_preds[val_valid]
    if len(p_v) > 5 and len(y_v) > 5:
        ic = np.corrcoef(p_v, y_v)[0, 1]
        log.info(f"  LGBM fold {fold_idx} {horizon}: IC={ic:.4f}, "
                 f"best_iter={model.best_iteration}, n_train={len(y_tr)}, n_val={len(y_v)}")
    else:
        log.info(f"  LGBM fold {fold_idx} {horizon}: too few samples for IC")

    return model, train_preds, val_preds


# ═══════════════════════════════════════════════════════════════════
#  SECTION 8: STAGE 2 — RESIDUAL GRU WITH RANKING LOSS
# ═══════════════════════════════════════════════════════════════════


def _build_residual_model_class():
    """Build the small residual GRU model class."""
    _import_torch()

    class ResidualGRU(nn.Module):
        """
        Small GRU that predicts the residual: actual_return - lgbm_prediction.
        Inputs: sequence of 15-min bar features + LightGBM prediction appended.
        Outputs per horizon: residual_score, confidence_logit, side_gate.

        Side-specific processing: separate linear projections for the long
        vs short pathways before final output, allowing the model to learn
        asymmetric behavior (short side is empirically stronger).
        """

        def __init__(
            self,
            n_features: int,
            hidden_dim: int = 64,
            n_gru_layers: int = 1,
            dropout: float = 0.15,
            horizons: Tuple[str, ...] = ("15min", "1h", "2h"),
        ):
            super().__init__()
            self.horizons = horizons
            self.hidden_dim = hidden_dim

            # Input: original features + 1 LightGBM prediction per horizon
            total_input = n_features + len(horizons)

            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(total_input, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # GRU encoder (unidirectional — causal)
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_gru_layers,
                batch_first=True,
                dropout=dropout if n_gru_layers > 1 else 0,
            )

            # Per-horizon heads with side-specific processing
            self.long_heads = nn.ModuleDict()
            self.short_heads = nn.ModuleDict()
            self.confidence_heads = nn.ModuleDict()
            self.side_gates = nn.ModuleDict()

            for h in horizons:
                # Long pathway
                self.long_heads[h] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
                # Short pathway
                self.short_heads[h] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 1),
                )
                # Confidence (scalar)
                self.confidence_heads[h] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 4, 1),
                )
                # Side gate: sigmoid output — 1.0 = use long head, 0.0 = use short head
                self.side_gates[h] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 4),
                    nn.GELU(),
                    nn.Linear(hidden_dim // 4, 1),
                    nn.Sigmoid(),
                )

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(
            self, x: "torch.Tensor"
        ) -> Dict[str, Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]]:
            """
            Args:
                x: (batch, seq_len, n_features + n_horizons)
            Returns:
                {horizon: (residual_score, confidence_logit, side_gate)}
                each shape (batch, 1)
            """
            h = self.input_proj(x)
            gru_out, _ = self.gru(h)
            # Use last timestep (causal)
            context = gru_out[:, -1, :]

            outputs = {}
            for h_name in self.horizons:
                long_out = self.long_heads[h_name](context)
                short_out = self.short_heads[h_name](context)
                gate = self.side_gates[h_name](context)
                # Blend: positive gate → long head, negative gate → short head
                residual = gate * long_out + (1.0 - gate) * short_out
                conf = self.confidence_heads[h_name](context)
                outputs[h_name] = (residual.squeeze(-1), conf.squeeze(-1), gate.squeeze(-1))

            return outputs

    return ResidualGRU


def listmle_loss(y_pred: "torch.Tensor", y_true: "torch.Tensor", mask: "torch.Tensor") -> "torch.Tensor":
    """
    ListMLE ranking loss — optimizes for correct ORDERING of predictions.

    This directly improves IC stability because IC = rank correlation.
    Reference: Xia et al. "Listwise Approach to Learning to Rank" (2008)

    Args:
        y_pred: (batch,) predicted scores
        y_true: (batch,) true values
        mask:   (batch,) 1.0 if valid, 0.0 if NaN
    Returns:
        scalar loss
    """
    _import_torch()

    # Only compute on valid samples
    valid_idx = mask.nonzero(as_tuple=True)[0]
    if len(valid_idx) < 4:
        return torch.tensor(0.0, device=y_pred.device, requires_grad=True)

    p = y_pred[valid_idx]
    t = y_true[valid_idx]

    # Sort by true values descending
    _, sorted_indices = torch.sort(t, descending=True)
    p_sorted = p[sorted_indices]

    # ListMLE: log-likelihood of the sorted permutation
    # P(permutation | scores) = prod_i softmax(remaining scores at position i)
    n = len(p_sorted)
    # Compute cumulative log-sum-exp from the end
    # For numerical stability, subtract max
    max_val = p_sorted.max()
    p_shifted = p_sorted - max_val

    # Cumulative logsumexp from the end
    cumsums = torch.logcumsumexp(p_shifted.flip(0), dim=0).flip(0)

    # Loss = -sum(p_sorted - cumsums)
    loss = -(p_shifted - cumsums).mean()

    return loss


def _build_residual_dataset_class():
    """Build dataset class for residual GRU training."""
    _import_torch()
    from torch.utils.data import Dataset

    class ResidualSequenceDataset(Dataset):
        """
        Each sample: seq_len consecutive 15min bars + LightGBM predictions appended.
        Target: residual = actual_ticks - lgbm_pred for each horizon.
        """

        def __init__(
            self,
            features: np.ndarray,
            lgbm_preds: Dict[str, np.ndarray],
            actuals: Dict[str, np.ndarray],
            seq_len: int = 16,
            dates: Optional[np.ndarray] = None,
        ):
            self.features = features.astype(np.float32)
            self.seq_len = seq_len
            self.horizons = list(lgbm_preds.keys())
            self.dates = dates

            # Stack LightGBM predictions as extra features
            lgbm_stack = np.column_stack([
                np.nan_to_num(lgbm_preds[h], nan=0.0).astype(np.float32)
                for h in self.horizons
            ])
            self.features_augmented = np.concatenate(
                [self.features, lgbm_stack], axis=1
            ).astype(np.float32)

            # Residual targets
            self.residuals = {}
            for h in self.horizons:
                actual = actuals[h].astype(np.float32)
                lgbm_p = np.nan_to_num(lgbm_preds[h], nan=0.0).astype(np.float32)
                self.residuals[h] = actual - lgbm_p

            self.actuals = {h: v.astype(np.float32) for h, v in actuals.items()}

            # Valid indices
            self.valid_indices = []
            if dates is not None:
                for i in range(len(features) - seq_len):
                    window_dates = dates[i : i + seq_len]
                    if window_dates[0] == window_dates[-1]:
                        target_idx = i + seq_len - 1
                        has_label = False
                        for h in self.horizons:
                            if not np.isnan(actuals[h][target_idx]):
                                has_label = True
                                break
                        if has_label:
                            self.valid_indices.append(i)
            else:
                self.valid_indices = list(range(len(features) - seq_len))

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            i = self.valid_indices[idx]
            target_idx = i + self.seq_len - 1

            x = torch.from_numpy(self.features_augmented[i : i + self.seq_len])

            residuals = {}
            actuals_out = {}
            masks = {}
            for h in self.horizons:
                r_val = self.residuals[h][target_idx]
                a_val = self.actuals[h][target_idx]
                valid = not np.isnan(a_val)
                residuals[h] = torch.tensor(r_val if valid else 0.0, dtype=torch.float32)
                actuals_out[h] = torch.tensor(a_val if valid else 0.0, dtype=torch.float32)
                masks[h] = torch.tensor(1.0 if valid else 0.0, dtype=torch.float32)

            return x, residuals, actuals_out, masks

    return ResidualSequenceDataset


def residual_collate_fn(batch):
    """Collate for residual dataset."""
    _import_torch()
    xs = torch.stack([b[0] for b in batch])
    residuals = {}
    actuals = {}
    masks = {}
    horizons = batch[0][1].keys()
    for h in horizons:
        residuals[h] = torch.stack([b[1][h] for b in batch])
        actuals[h] = torch.stack([b[2][h] for b in batch])
        masks[h] = torch.stack([b[3][h] for b in batch])
    return xs, residuals, actuals, masks


def train_residual_gru_fold(
    model,
    train_loader,
    val_loader,
    horizons: List[str],
    device: str,
    epochs: int,
    lr: float,
    patience: int,
    fold_idx: int,
) -> Tuple[Dict[str, float], float]:
    """Train residual GRU with ListMLE + MSE hybrid loss. Returns best val metrics and loss."""
    _import_torch()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    mse_loss_fn = nn.MSELoss(reduction="none")

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # ── TRAIN ──
        model.train()
        train_losses = []
        for batch in train_loader:
            x, residuals, actuals, masks = batch
            x = x.to(device)

            outputs = model(x)

            loss = torch.tensor(0.0, device=device)
            n_valid = 0

            for h in horizons:
                res_target = residuals[h].to(device)
                actual_target = actuals[h].to(device)
                msk = masks[h].to(device)

                if msk.sum() < 4:
                    continue

                res_pred, conf_logit, side_gate = outputs[h]

                # Loss 1: MSE on residual (weighted by mask)
                mse_l = (mse_loss_fn(res_pred, res_target) * msk).sum() / msk.sum()

                # Loss 2: ListMLE ranking loss on final prediction
                # (ensures ordering is correct, which is what IC measures)
                ranking_l = listmle_loss(res_pred, res_target, msk)

                # Loss 3: Confidence calibration — high confidence on correct-sign predictions
                correct_sign = (torch.sign(res_pred) == torch.sign(res_target)).float()
                conf_target = correct_sign * msk
                bce_l = nn.functional.binary_cross_entropy_with_logits(
                    conf_logit, conf_target, weight=msk, reduction="sum"
                ) / max(msk.sum().item(), 1.0)

                # Loss 4: Side gate regularization — push gate toward 1 when actual > 0
                # and toward 0 when actual < 0. This teaches the model to route correctly.
                side_target = (actual_target > 0).float() * msk
                side_l = nn.functional.binary_cross_entropy(
                    side_gate * msk, side_target, reduction="sum"
                ) / max(msk.sum().item(), 1.0)

                # Combined: MSE is primary, ranking for IC stability, conf+side are auxiliary
                horizon_loss = 1.0 * mse_l + 0.5 * ranking_l + 0.2 * bce_l + 0.1 * side_l
                loss = loss + horizon_loss
                n_valid += 1

            if n_valid > 0:
                loss = loss / n_valid
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

        scheduler.step()

        # ── VALIDATE ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, residuals, actuals, masks = batch
                x = x.to(device)

                outputs = model(x)

                loss = torch.tensor(0.0, device=device)
                n_valid = 0
                for h in horizons:
                    res_target = residuals[h].to(device)
                    msk = masks[h].to(device)
                    if msk.sum() < 2:
                        continue
                    res_pred, _, _ = outputs[h]
                    # Validate on ranking loss (what we care about: IC)
                    ranking_l = listmle_loss(res_pred, res_target, msk)
                    loss = loss + ranking_l
                    n_valid += 1

                if n_valid > 0:
                    val_losses.append((loss / n_valid).item())

        avg_train = np.mean(train_losses) if train_losses else float("inf")
        avg_val = np.mean(val_losses) if val_losses else float("inf")

        if epoch % 5 == 0 or epoch == epochs - 1:
            log.info(
                f"  ResGRU Fold {fold_idx} Epoch {epoch:3d}/{epochs}: "
                f"train_loss={avg_train:.6f}  val_loss={avg_val:.6f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"  ResGRU Fold {fold_idx}: Early stop at epoch {epoch}")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    return {}, best_val_loss


def predict_residual_fold(
    model,
    loader,
    horizons: List[str],
    device: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run residual GRU inference. Returns residual preds, confidences, side gates."""
    _import_torch()
    model.eval()

    preds = {h: [] for h in horizons}
    confs = {h: [] for h in horizons}
    gates = {h: [] for h in horizons}

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            outputs = model(x)

            for h in horizons:
                res_pred, conf_logit, side_gate = outputs[h]
                preds[h].append(res_pred.cpu().numpy())
                confs[h].append(torch.sigmoid(conf_logit).cpu().numpy())
                gates[h].append(side_gate.cpu().numpy())

    for h in horizons:
        preds[h] = np.concatenate(preds[h])
        confs[h] = np.concatenate(confs[h])
        gates[h] = np.concatenate(gates[h])

    return preds, confs, gates


# ═══════════════════════════════════════════════════════════════════
#  SECTION 9: ENSEMBLE BLENDING + ALPHA TUNING
# ═══════════════════════════════════════════════════════════════════


def tune_alpha(
    lgbm_preds: np.ndarray,
    nn_residual_preds: np.ndarray,
    actuals: np.ndarray,
    nn_confidence: np.ndarray,
) -> Tuple[float, float]:
    """
    Tune alpha on validation data.
    Final pred = alpha * lgbm + (1-alpha) * (lgbm + nn_residual * confidence)
              = lgbm + (1-alpha) * nn_residual * confidence

    Returns (alpha, best_ic).
    """
    valid = ~np.isnan(actuals) & ~np.isnan(lgbm_preds)
    if valid.sum() < 10:
        return 0.7, 0.0

    lgbm_v = lgbm_preds[valid]
    nn_res_v = nn_residual_preds[valid]
    conf_v = nn_confidence[valid]
    actual_v = actuals[valid]

    best_alpha = 0.7
    best_ic = -999.0

    for alpha_test in np.arange(0.0, 1.05, 0.05):
        # Blend: lgbm_base + (1-alpha) * nn_residual * confidence
        combined = alpha_test * lgbm_v + (1 - alpha_test) * (lgbm_v + nn_res_v * conf_v)
        # Simplify: combined = lgbm_v + (1-alpha) * nn_res_v * conf_v
        # Actually let's be more explicit:
        # ensemble = alpha * lgbm + (1 - alpha) * (lgbm + residual * conf)
        # = lgbm + (1 - alpha) * residual * conf
        if len(combined) > 5:
            ic = np.corrcoef(combined, actual_v)[0, 1]
            if not np.isnan(ic) and ic > best_ic:
                best_ic = ic
                best_alpha = alpha_test

    return best_alpha, best_ic


# ═══════════════════════════════════════════════════════════════════
#  SECTION 10: TRADE SIMULATION
# ═══════════════════════════════════════════════════════════════════


def simulate_trades(
    preds: np.ndarray,
    actuals: np.ndarray,
    confidence_pct: float = 0.10,
    cost_ticks: float = COST_RT_TICKS,
    confidences: Optional[np.ndarray] = None,
) -> Optional[Dict]:
    """Simulate long/short trades with per-side reporting."""
    if len(preds) < 20:
        return None

    upper = np.quantile(preds, 1 - confidence_pct)
    lower = np.quantile(preds, confidence_pct)

    trades = []
    for i in range(len(preds)):
        if preds[i] >= upper:
            pnl = actuals[i] - cost_ticks
            trades.append({"dir": "long", "pnl": pnl, "raw": actuals[i],
                           "conf": confidences[i] if confidences is not None else 1.0})
        elif preds[i] <= lower:
            pnl = -actuals[i] - cost_ticks
            trades.append({"dir": "short", "pnl": pnl, "raw": -actuals[i],
                           "conf": confidences[i] if confidences is not None else 1.0})

    if not trades:
        return None

    pnl_arr = np.array([t["pnl"] for t in trades])
    cum_pnl = np.cumsum(pnl_arr)

    sharpe = pnl_arr.mean() / max(pnl_arr.std(), 1e-6) * np.sqrt(252)
    downside = np.sqrt(np.mean(np.minimum(pnl_arr, 0) ** 2))
    sortino = pnl_arr.mean() / max(downside, 1e-6) * np.sqrt(252)
    wr = np.mean(pnl_arr > 0)
    pf = np.sum(pnl_arr[pnl_arr > 0]) / max(-np.sum(pnl_arr[pnl_arr < 0]), 1e-6)
    max_dd = float(np.min(cum_pnl - np.maximum.accumulate(cum_pnl)))

    long_pnl = np.array([t["pnl"] for t in trades if t["dir"] == "long"])
    short_pnl = np.array([t["pnl"] for t in trades if t["dir"] == "short"])

    return {
        "n_trades": len(trades),
        "total_pnl_ticks": float(pnl_arr.sum()),
        "total_pnl_dollars": float(pnl_arr.sum() * ES_TICK_VALUE),
        "avg_pnl_ticks": float(pnl_arr.mean()),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "win_rate": float(wr),
        "profit_factor": float(pf),
        "max_dd_ticks": float(max_dd),
        "max_dd_dollars": float(max_dd * ES_TICK_VALUE),
        "long_trades": int(len(long_pnl)),
        "long_wr": float(np.mean(long_pnl > 0)) if len(long_pnl) > 0 else 0,
        "long_avg": float(long_pnl.mean()) if len(long_pnl) > 0 else 0,
        "long_sharpe": float(
            long_pnl.mean() / max(long_pnl.std(), 1e-6) * np.sqrt(252)
        ) if len(long_pnl) > 2 else 0,
        "short_trades": int(len(short_pnl)),
        "short_wr": float(np.mean(short_pnl > 0)) if len(short_pnl) > 0 else 0,
        "short_avg": float(short_pnl.mean()) if len(short_pnl) > 0 else 0,
        "short_sharpe": float(
            short_pnl.mean() / max(short_pnl.std(), 1e-6) * np.sqrt(252)
        ) if len(short_pnl) > 2 else 0,
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 11: WALK-FORWARD ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════


def walk_forward_train(df: pd.DataFrame, args) -> Dict:
    """Full two-stage walk-forward training."""
    _import_torch()
    _import_lightgbm()
    from torch.utils.data import DataLoader

    horizons = list(HORIZON_BARS.keys())
    feature_cols = get_feature_columns(df)
    log.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:20]}...")

    dates = sorted(df["date"].unique())
    log.info(f"Total trading days: {len(dates)} ({dates[0]} -> {dates[-1]})")

    train_days = args.train_days
    val_days = args.val_days
    slide = args.slide_days
    seq_len = args.seq_len

    if len(dates) < train_days + val_days + slide:
        raise RuntimeError(
            f"Not enough days ({len(dates)}) for {train_days}+{val_days} walk-forward"
        )

    # Prepare arrays
    features_all = df[feature_cols].values.astype(np.float32)
    dates_all = df["date"].values

    label_arrays = {}
    for h in horizons:
        label_arrays[h] = df[f"fwd_ticks_{h}"].values.astype(np.float32)

    # Build model classes
    ResidualModelClass = _build_residual_model_class()
    ResidualDatasetClass = _build_residual_dataset_class()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # MLflow setup
    mlflow_client = None
    if args.mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.experiment_name)
            mlflow.start_run(
                run_name=f"lh_v3_ensemble_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            mlflow.log_params({
                "model": "LightGBM+ResidualGRU_Ensemble",
                "nn_hidden_dim": args.hidden_dim,
                "nn_gru_layers": args.n_gru_layers,
                "nn_dropout": args.dropout,
                "seq_len": seq_len,
                "train_days": train_days,
                "val_days": val_days,
                "slide_days": slide,
                "nn_epochs": args.epochs,
                "nn_batch_size": args.batch_size,
                "nn_lr": args.lr,
                "cost_rt_ticks": COST_RT_TICKS,
                "n_features": len(feature_cols),
                "n_dates": len(dates),
                "horizons": ",".join(horizons),
                "loss": "ListMLE+MSE+ConfBCE+SideGate",
            })
            mlflow_client = mlflow
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")

    # Walk-forward loop
    all_fold_results = []
    # Per-horizon accumulators for final analysis
    all_oot = {h: {"preds_lgbm": [], "preds_nn_res": [], "preds_ensemble": [],
                    "actuals": [], "confs": [], "gates": [], "dates": []}
                for h in horizons}
    fold_idx = 0

    start_idx = train_days
    for fold_start in range(start_idx, len(dates) - val_days + 1, slide):
        fold_train_dates = dates[fold_start - train_days : fold_start]
        fold_val_dates = dates[fold_start : fold_start + val_days]

        if len(fold_val_dates) < val_days:
            break

        fold_idx += 1
        log.info(
            f"\n{'='*60}\n"
            f"FOLD {fold_idx}: train {fold_train_dates[0]}->{fold_train_dates[-1]} "
            f"({len(fold_train_dates)}d), val {fold_val_dates[0]}->{fold_val_dates[-1]} "
            f"({len(fold_val_dates)}d)\n{'='*60}"
        )

        # ── LEAKAGE AUDIT ──
        audit = leakage_audit(df, fold_train_dates.tolist() if hasattr(fold_train_dates, 'tolist') else list(fold_train_dates),
                              fold_val_dates.tolist() if hasattr(fold_val_dates, 'tolist') else list(fold_val_dates),
                              feature_cols)
        if not audit["all_passed"]:
            log.error(f"FOLD {fold_idx}: Leakage audit FAILED — skipping fold")
            continue

        # ── SPLIT DATA ──
        train_mask = np.isin(dates_all, fold_train_dates)
        val_mask = np.isin(dates_all, fold_val_dates)

        train_features_raw = features_all[train_mask].copy()
        val_features_raw = features_all[val_mask].copy()

        # Robust scaling: median/IQR from TRAIN ONLY
        train_median = np.nanmedian(train_features_raw, axis=0)
        q75 = np.nanpercentile(train_features_raw, 75, axis=0)
        q25 = np.nanpercentile(train_features_raw, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0

        train_features = (train_features_raw - train_median) / iqr
        val_features = (val_features_raw - train_median) / iqr

        train_features = np.nan_to_num(train_features, nan=0.0, posinf=3.0, neginf=-3.0)
        val_features = np.nan_to_num(val_features, nan=0.0, posinf=3.0, neginf=-3.0)
        train_features = np.clip(train_features, -5, 5)
        val_features = np.clip(val_features, -5, 5)

        train_labels = {h: label_arrays[h][train_mask] for h in horizons}
        val_labels = {h: label_arrays[h][val_mask] for h in horizons}
        train_dates_fold = dates_all[train_mask]
        val_dates_fold = dates_all[val_mask]

        fold_result = {
            "fold": fold_idx,
            "train_start": fold_train_dates[0],
            "train_end": fold_train_dates[-1],
            "val_start": fold_val_dates[0],
            "val_end": fold_val_dates[-1],
            "train_samples": int(train_mask.sum()),
            "val_samples": int(val_mask.sum()),
        }

        # ════════════════════════════════
        #  STAGE 1: LightGBM
        # ════════════════════════════════
        log.info(f"  --- Stage 1: LightGBM ---")
        lgbm_models = {}
        lgbm_train_preds = {}
        lgbm_val_preds = {}

        for h in horizons:
            lgbm_model, tr_preds, v_preds = train_lgbm_fold(
                train_features, train_labels[h],
                val_features, val_labels[h],
                feature_names=feature_cols,
                horizon=h, fold_idx=fold_idx,
            )
            lgbm_models[h] = lgbm_model
            lgbm_train_preds[h] = tr_preds
            lgbm_val_preds[h] = v_preds

            # Log LGBM-only IC
            valid_v = ~np.isnan(val_labels[h]) & ~np.isnan(v_preds)
            if valid_v.sum() > 5:
                ic_lgbm = np.corrcoef(v_preds[valid_v], val_labels[h][valid_v])[0, 1]
                fold_result[f"lgbm_ic_{h}"] = float(ic_lgbm)
            else:
                fold_result[f"lgbm_ic_{h}"] = float("nan")

        # ════════════════════════════════
        #  STAGE 2: Residual GRU
        # ════════════════════════════════
        log.info(f"  --- Stage 2: Residual GRU ---")

        # Build residual datasets
        train_ds = ResidualDatasetClass(
            train_features, lgbm_train_preds, train_labels,
            seq_len=seq_len, dates=train_dates_fold,
        )
        val_ds = ResidualDatasetClass(
            val_features, lgbm_val_preds, val_labels,
            seq_len=seq_len, dates=val_dates_fold,
        )

        if len(train_ds) < 50 or len(val_ds) < 10:
            log.warning(f"Fold {fold_idx}: too few NN samples (train={len(train_ds)}, val={len(val_ds)}) — LGBM only")
            # Fall back to LGBM-only for this fold
            for h in horizons:
                val_target_idx = [vi + seq_len - 1 for vi in val_ds.valid_indices] if len(val_ds) > 0 else []
                if not val_target_idx:
                    continue
                v_preds = lgbm_val_preds[h][val_target_idx]
                v_actuals = val_labels[h][val_target_idx]
                valid = ~np.isnan(v_actuals) & ~np.isnan(v_preds)
                if valid.sum() > 5:
                    all_oot[h]["preds_lgbm"].append(v_preds[valid])
                    all_oot[h]["preds_ensemble"].append(v_preds[valid])
                    all_oot[h]["actuals"].append(v_actuals[valid])
                    all_oot[h]["confs"].append(np.ones(valid.sum()))
                    all_oot[h]["gates"].append(np.full(valid.sum(), 0.5))
                    all_oot[h]["dates"].extend(val_dates_fold[val_target_idx][valid].tolist())
            all_fold_results.append(fold_result)
            continue

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(args.num_workers, 8),
            pin_memory=(device == "cuda"),
            collate_fn=residual_collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(args.num_workers, 4),
            pin_memory=(device == "cuda"),
            collate_fn=residual_collate_fn,
        )

        # Build residual model
        n_input_features = len(feature_cols) + len(horizons)  # features + LGBM preds
        model = ResidualModelClass(
            n_features=len(feature_cols),
            hidden_dim=args.hidden_dim,
            n_gru_layers=args.n_gru_layers,
            dropout=args.dropout,
            horizons=tuple(horizons),
        ).to(device)

        # Train
        _, best_val_loss = train_residual_gru_fold(
            model, train_loader, val_loader,
            horizons, device,
            args.epochs, args.lr, args.patience,
            fold_idx,
        )
        fold_result["nn_val_loss"] = float(best_val_loss)

        # Predict on validation
        nn_res_preds, nn_confs, nn_gates = predict_residual_fold(
            model, val_loader, horizons, device
        )

        val_target_indices = [vi + seq_len - 1 for vi in val_ds.valid_indices]

        # ════════════════════════════════
        #  ENSEMBLE: Tune alpha + combine
        # ════════════════════════════════
        log.info(f"  --- Ensemble blending ---")

        for h in horizons:
            lgbm_v = lgbm_val_preds[h][val_target_indices]
            nn_res_v = nn_res_preds[h]
            conf_v = nn_confs[h]
            gate_v = nn_gates[h]
            actual_v = val_labels[h][val_target_indices]

            valid = ~np.isnan(actual_v) & ~np.isnan(lgbm_v)
            if valid.sum() < 10:
                fold_result[f"ensemble_ic_{h}"] = float("nan")
                fold_result[f"alpha_{h}"] = 0.7
                continue

            # Tune alpha on this fold's val data
            alpha, alpha_ic = tune_alpha(lgbm_v[valid], nn_res_v[valid], actual_v[valid], conf_v[valid])
            fold_result[f"alpha_{h}"] = float(alpha)

            # Final ensemble prediction
            ensemble_preds = lgbm_v.copy()
            ensemble_preds[valid] = (
                alpha * lgbm_v[valid]
                + (1 - alpha) * (lgbm_v[valid] + nn_res_v[valid] * conf_v[valid])
            )

            # IC of ensemble
            ens_ic = np.corrcoef(ensemble_preds[valid], actual_v[valid])[0, 1]
            fold_result[f"ensemble_ic_{h}"] = float(ens_ic)

            lgbm_ic = fold_result.get(f"lgbm_ic_{h}", float("nan"))
            log.info(
                f"  {h}: LGBM IC={lgbm_ic:.4f}, Ensemble IC={ens_ic:.4f}, "
                f"alpha={alpha:.2f}"
            )

            # Accumulate OOT
            all_oot[h]["preds_lgbm"].append(lgbm_v[valid])
            all_oot[h]["preds_nn_res"].append(nn_res_v[valid])
            all_oot[h]["preds_ensemble"].append(ensemble_preds[valid])
            all_oot[h]["actuals"].append(actual_v[valid])
            all_oot[h]["confs"].append(conf_v[valid])
            all_oot[h]["gates"].append(gate_v[valid])
            all_oot[h]["dates"].extend(val_dates_fold[val_target_indices][valid].tolist())

        all_fold_results.append(fold_result)

        # Save fold artifacts
        fold_dir = OUTPUT_DIR / f"fold_{fold_idx:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), fold_dir / "residual_gru_weights.pt")
        np.savez(fold_dir / "norm_params.npz", median=train_median, iqr=iqr)

        # Save LightGBM models
        for h in horizons:
            if lgbm_models[h] is not None:
                lgbm_models[h].save_model(str(fold_dir / f"lgbm_{h}.txt"))

        npz_data = {"dates": np.array(val_dates_fold[val_target_indices])}
        for h in horizons:
            lgbm_v = lgbm_val_preds[h][val_target_indices]
            valid = ~np.isnan(val_labels[h][val_target_indices]) & ~np.isnan(lgbm_v)
            if valid.sum() > 0:
                npz_data[f"preds_lgbm_{h}"] = lgbm_v[valid]
                npz_data[f"preds_nn_res_{h}"] = nn_res_preds[h][valid]
                npz_data[f"preds_ensemble_{h}"] = all_oot[h]["preds_ensemble"][-1]
                npz_data[f"actuals_{h}"] = val_labels[h][val_target_indices][valid]
                npz_data[f"confs_{h}"] = nn_confs[h][valid]
        np.savez_compressed(fold_dir / "predictions.npz", **npz_data)

        log.info(
            f"Fold {fold_idx} complete: " +
            ", ".join(
                f"{h} ens_IC={fold_result.get(f'ensemble_ic_{h}', float('nan')):.4f}"
                for h in horizons
            )
        )

        if mlflow_client:
            for h in horizons:
                for metric_name in [f"lgbm_ic_{h}", f"ensemble_ic_{h}", f"alpha_{h}"]:
                    val = fold_result.get(metric_name)
                    if val is not None and not (isinstance(val, float) and np.isnan(val)):
                        mlflow_client.log_metric(metric_name, val, step=fold_idx)

        # Cleanup
        del model, train_loader, val_loader, train_ds, val_ds
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    #  AGGREGATE OOT RESULTS
    # ═══════════════════════════════════════════════════════════════
    log.info(f"\n{'='*60}\nAGGREGATE OOT RESULTS ({fold_idx} folds)\n{'='*60}")

    concat_results = {}
    for h in horizons:
        if not all_oot[h]["preds_ensemble"]:
            log.warning(f"  {h}: No OOT predictions collected — skip")
            continue

        p_ens = np.concatenate(all_oot[h]["preds_ensemble"])
        p_lgbm = np.concatenate(all_oot[h]["preds_lgbm"])
        a = np.concatenate(all_oot[h]["actuals"])
        c = np.concatenate(all_oot[h]["confs"])

        if len(p_ens) < 10:
            continue

        ic_ens = np.corrcoef(p_ens, a)[0, 1]
        ic_lgbm = np.corrcoef(p_lgbm, a)[0, 1]
        rank_ic_ens = stats.spearmanr(p_ens, a)[0]
        rank_ic_lgbm = stats.spearmanr(p_lgbm, a)[0]
        dir_acc_ens = np.mean(np.sign(p_ens) == np.sign(a))
        dir_acc_lgbm = np.mean(np.sign(p_lgbm) == np.sign(a))

        # Per-fold IC for stability
        fold_ics_ens = [
            fr.get(f"ensemble_ic_{h}", np.nan) for fr in all_fold_results
            if not np.isnan(fr.get(f"ensemble_ic_{h}", np.nan))
        ]
        fold_ics_lgbm = [
            fr.get(f"lgbm_ic_{h}", np.nan) for fr in all_fold_results
            if not np.isnan(fr.get(f"lgbm_ic_{h}", np.nan))
        ]

        ic_mean_ens = np.mean(fold_ics_ens) if fold_ics_ens else 0
        ic_std_ens = np.std(fold_ics_ens) if len(fold_ics_ens) > 1 else 1e-6
        ic_sharpe_ens = ic_mean_ens / max(ic_std_ens, 1e-6)

        ic_mean_lgbm = np.mean(fold_ics_lgbm) if fold_ics_lgbm else 0
        ic_std_lgbm = np.std(fold_ics_lgbm) if len(fold_ics_lgbm) > 1 else 1e-6
        ic_sharpe_lgbm = ic_mean_lgbm / max(ic_std_lgbm, 1e-6)

        concat_results[h] = {
            # Ensemble metrics
            "ensemble_concat_ic": float(ic_ens),
            "ensemble_rank_ic": float(rank_ic_ens),
            "ensemble_dir_acc": float(dir_acc_ens),
            "ensemble_ic_mean": float(ic_mean_ens),
            "ensemble_ic_std": float(ic_std_ens),
            "ensemble_ic_sharpe": float(ic_sharpe_ens),
            # LGBM-only metrics (for comparison)
            "lgbm_concat_ic": float(ic_lgbm),
            "lgbm_rank_ic": float(rank_ic_lgbm),
            "lgbm_dir_acc": float(dir_acc_lgbm),
            "lgbm_ic_mean": float(ic_mean_lgbm),
            "lgbm_ic_std": float(ic_std_lgbm),
            "lgbm_ic_sharpe": float(ic_sharpe_lgbm),
            # Meta
            "n_predictions": int(len(p_ens)),
            "n_folds": len(fold_ics_ens),
        }

        log.info(
            f"\n  {h} — ENSEMBLE: IC={ic_ens:.4f}, RankIC={rank_ic_ens:.4f}, "
            f"DirAcc={dir_acc_ens:.1%}, IC_Sharpe={ic_sharpe_ens:.2f}"
        )
        log.info(
            f"  {h} — LGBM only: IC={ic_lgbm:.4f}, RankIC={rank_ic_lgbm:.4f}, "
            f"DirAcc={dir_acc_lgbm:.1%}, IC_Sharpe={ic_sharpe_lgbm:.2f}"
        )
        log.info(
            f"  {h} — IC improvement: {ic_ens - ic_lgbm:+.4f} "
            f"(Sharpe improvement: {ic_sharpe_ens - ic_sharpe_lgbm:+.2f})"
        )

        # Quantile analysis (per-side)
        for q_pct in [10, 20, 30]:
            q = q_pct / 100
            top_mask = p_ens >= np.quantile(p_ens, 1 - q)
            bot_mask = p_ens <= np.quantile(p_ens, q)
            top_actual = a[top_mask].mean()
            bot_actual = a[bot_mask].mean()
            top_wr = np.mean(a[top_mask] > 0)
            bot_wr = np.mean(a[bot_mask] < 0)

            concat_results[h][f"long_top{q_pct}_mean_ticks"] = float(top_actual)
            concat_results[h][f"short_bot{q_pct}_mean_ticks"] = float(bot_actual)
            concat_results[h][f"long_top{q_pct}_wr"] = float(top_wr)
            concat_results[h][f"short_bot{q_pct}_wr"] = float(bot_wr)
            concat_results[h][f"ls{q_pct}_spread"] = float(top_actual - bot_actual)

            log.info(
                f"    Q{q_pct}: LONG avg={top_actual:+.2f}tk WR={top_wr:.1%}, "
                f"SHORT avg={bot_actual:+.2f}tk WR={bot_wr:.1%}, "
                f"L/S spread={top_actual - bot_actual:.2f}tk"
            )

        # Fold stability breakdown
        log.info(f"  {h} — Per-fold ICs: {[f'{x:.3f}' for x in fold_ics_ens]}")

        if mlflow_client:
            mlflow_client.log_metrics({
                f"oot_ensemble_ic_{h}": ic_ens,
                f"oot_ensemble_rank_ic_{h}": rank_ic_ens,
                f"oot_ensemble_dir_acc_{h}": dir_acc_ens,
                f"oot_ensemble_ic_sharpe_{h}": ic_sharpe_ens,
                f"oot_lgbm_ic_{h}": ic_lgbm,
                f"oot_lgbm_ic_sharpe_{h}": ic_sharpe_lgbm,
            })

    # ═══════════════════════════════════════════════════════════════
    #  REGIME-STRATIFIED ANALYSIS (HC #428 R1)
    # ═══════════════════════════════════════════════════════════════
    log.info(f"\n{'='*60}\nREGIME-STRATIFIED ANALYSIS\n{'='*60}")

    regime_results = {}
    day_closes = df.groupby("date")["close"].last()
    if len(day_closes) > 1:
        day_returns = day_closes.pct_change()
        green_days = set(day_returns[day_returns > 0].index)
        red_days = set(day_returns[day_returns <= 0].index)

        for h in horizons:
            if h not in concat_results:
                continue
            p = np.concatenate(all_oot[h]["preds_ensemble"])
            a = np.concatenate(all_oot[h]["actuals"])
            oot_dates = np.array(all_oot[h]["dates"])

            for regime_name, regime_dates in [("green", green_days), ("red", red_days)]:
                rmask = np.isin(oot_dates[:len(p)], list(regime_dates))
                if rmask.sum() < 10:
                    continue
                r_ic = np.corrcoef(p[rmask], a[rmask])[0, 1] if rmask.sum() > 5 else 0
                r_dir = np.mean(np.sign(p[rmask]) == np.sign(a[rmask]))
                key = f"{h}_{regime_name}"
                regime_results[key] = {
                    "ic": float(r_ic),
                    "dir_acc": float(r_dir),
                    "n": int(rmask.sum()),
                }
                log.info(f"  Regime {key}: IC={r_ic:.4f}, DirAcc={r_dir:.1%}, N={rmask.sum()}")

            # HC #428 R1 check
            green_key = f"{h}_green"
            red_key = f"{h}_red"
            if green_key in regime_results and red_key in regime_results:
                g_ic = regime_results[green_key]["ic"]
                r_ic = regime_results[red_key]["ic"]
                max_ic = max(abs(g_ic), abs(r_ic))
                if max_ic > 0:
                    regime_gap = abs(g_ic - r_ic) / max_ic
                    passed = regime_gap <= 0.50
                    log.info(
                        f"  {h} regime gap: |{g_ic:.3f} - {r_ic:.3f}| / {max_ic:.3f} = {regime_gap:.2f} "
                        f"({'PASS' if passed else 'FAIL: >0.50'})"
                    )
                    regime_results[f"{h}_gap"] = {
                        "gap": float(regime_gap),
                        "passed": passed,
                        "green_ic": float(g_ic),
                        "red_ic": float(r_ic),
                    }

    # ═══════════════════════════════════════════════════════════════
    #  TRADING SIMULATION
    # ═══════════════════════════════════════════════════════════════
    log.info(f"\n{'='*60}\nTRADING SIMULATION\n{'='*60}")

    sim_results = {}
    for h in horizons:
        if h not in concat_results:
            continue
        p = np.concatenate(all_oot[h]["preds_ensemble"])
        a = np.concatenate(all_oot[h]["actuals"])
        c = np.concatenate(all_oot[h]["confs"])

        for conf_label, conf_thresh in [("top10", 0.10), ("top20", 0.20), ("top30", 0.30)]:
            sim = simulate_trades(p, a, confidence_pct=conf_thresh,
                                  cost_ticks=COST_RT_TICKS, confidences=c)
            if sim:
                key = f"{h}_{conf_label}"
                sim_results[key] = sim
                log.info(
                    f"  SIM {key}: {sim['n_trades']} trades, "
                    f"Sharpe={sim['sharpe']:.2f}, Sortino={sim['sortino']:.2f}, "
                    f"WR={sim['win_rate']:.1%}, PF={sim['profit_factor']:.2f}, "
                    f"PnL={sim['total_pnl_ticks']:.0f}tk (${sim['total_pnl_dollars']:,.0f})"
                )
                log.info(
                    f"    LONG: {sim['long_trades']} trades, "
                    f"WR={sim['long_wr']:.1%}, avg={sim['long_avg']:.2f}tk, "
                    f"Sharpe={sim['long_sharpe']:.2f}"
                )
                log.info(
                    f"    SHORT: {sim['short_trades']} trades, "
                    f"WR={sim['short_wr']:.1%}, avg={sim['short_avg']:.2f}tk, "
                    f"Sharpe={sim['short_sharpe']:.2f}"
                )

                if mlflow_client:
                    mlflow_client.log_metrics({
                        f"sim_{key}_sharpe": sim["sharpe"],
                        f"sim_{key}_sortino": sim["sortino"],
                        f"sim_{key}_wr": sim["win_rate"],
                        f"sim_{key}_pf": sim["profit_factor"],
                        f"sim_{key}_trades": sim["n_trades"],
                    })

    # ═══════════════════════════════════════════════════════════════
    #  SAVE CONCAT OOT PREDICTIONS
    # ═══════════════════════════════════════════════════════════════
    npz_concat = {}
    for h in horizons:
        if all_oot[h]["preds_ensemble"]:
            npz_concat[f"preds_ensemble_{h}"] = np.concatenate(all_oot[h]["preds_ensemble"])
            npz_concat[f"preds_lgbm_{h}"] = np.concatenate(all_oot[h]["preds_lgbm"])
            npz_concat[f"actuals_{h}"] = np.concatenate(all_oot[h]["actuals"])
            npz_concat[f"confs_{h}"] = np.concatenate(all_oot[h]["confs"])
            npz_concat[f"gates_{h}"] = np.concatenate(all_oot[h]["gates"])
            npz_concat[f"dates_{h}"] = np.array(all_oot[h]["dates"])
    if npz_concat:
        np.savez_compressed(OUTPUT_DIR / "concat_oot_predictions.npz", **npz_concat)
        log.info(f"Saved concat OOT predictions")

    # ═══════════════════════════════════════════════════════════════
    #  SUMMARY JSON
    # ═══════════════════════════════════════════════════════════════
    summary = {
        "run_time": datetime.now().isoformat(),
        "model": "LightGBM + Residual GRU Ensemble (v3)",
        "config": {
            "lgbm_params": "num_leaves=63, lr=0.05, max_depth=8, feature_fraction=0.7",
            "nn_hidden_dim": args.hidden_dim,
            "nn_gru_layers": args.n_gru_layers,
            "nn_dropout": args.dropout,
            "seq_len": args.seq_len,
            "train_days": args.train_days,
            "val_days": args.val_days,
            "slide_days": args.slide_days,
            "nn_epochs": args.epochs,
            "nn_batch_size": args.batch_size,
            "nn_lr": args.lr,
            "cost_rt_ticks": COST_RT_TICKS,
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
            "loss_fn": "ListMLE (0.5) + MSE (1.0) + ConfBCE (0.2) + SideGate (0.1)",
        },
        "concat_results": concat_results,
        "regime_results": regime_results,
        "sim_results": sim_results,
        "fold_results": [
            {k: (v if not isinstance(v, (np.floating, np.integer)) else float(v))
             for k, v in fr.items()}
            for fr in all_fold_results
        ],
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary saved to {summary_path}")

    if mlflow_client:
        try:
            mlflow_client.log_artifact(str(summary_path))
            mlflow_client.log_artifact(str(OUTPUT_DIR / "concat_oot_predictions.npz"))
            mlflow_client.end_run()
        except Exception as e:
            log.warning(f"MLflow artifact logging failed: {e}")

    return summary


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Longer-Horizon Directional Model v3 (LightGBM + Residual GRU Ensemble)"
    )

    # Data
    parser.add_argument("--min-date", default="20250714", help="Earliest date to load")

    # Walk-forward
    parser.add_argument("--train-days", type=int, default=60, help="Training window days")
    parser.add_argument("--val-days", type=int, default=10, help="Validation window days (v3: 10d)")
    parser.add_argument("--slide-days", type=int, default=5, help="Slide step days")

    # Residual GRU model
    parser.add_argument("--hidden-dim", type=int, default=64, help="GRU hidden dimension (smaller than v2)")
    parser.add_argument("--n-gru-layers", type=int, default=1, help="GRU layers (1 for residual)")
    parser.add_argument("--dropout", type=float, default=0.15, help="Dropout rate")
    parser.add_argument("--seq-len", type=int, default=16, help="Input sequence length (15min bars)")

    # Training
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--epochs", type=int, default=30, help="Max NN epochs per fold")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--patience", type=int, default=7, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")

    # MLflow
    parser.add_argument("--mlflow", action="store_true", help="Log to MLflow")
    parser.add_argument(
        "--mlflow-uri", default="http://neptune:5000", help="MLflow tracking URI"
    )
    parser.add_argument(
        "--experiment-name", default="longer_horizon_v3_ensemble", help="MLflow experiment"
    )

    args = parser.parse_args()

    log.info("=" * 60)
    log.info("LONGER-HORIZON DIRECTIONAL MODEL v3")
    log.info("Two-Stage Ensemble: LightGBM + Residual GRU")
    log.info("=" * 60)
    log.info(f"Architecture: LightGBM base + ResidualGRU(hidden={args.hidden_dim}, "
             f"layers={args.n_gru_layers})")
    log.info(f"Loss: ListMLE ranking + MSE + Confidence BCE + Side gate")
    log.info(f"Walk-forward: {args.train_days}d train, {args.val_days}d val, "
             f"slide {args.slide_days}d")
    log.info(f"Device: {args.device}")
    log.info(f"Horizons: {list(HORIZON_BARS.keys())}")

    # Import torch
    _import_torch()
    _import_lightgbm()
    log.info(f"PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")
    log.info(f"LightGBM {lgb.__version__}")

    # Step 1: Load data
    log.info("\nStep 1: Loading minute bars...")
    t0 = time.time()
    minute_df = load_all_minute_bars(min_date=args.min_date)
    log.info(f"  Loaded in {time.time()-t0:.1f}s")

    # Step 2: Queue features
    log.info("\nStep 2: Loading queue features...")
    queue_df = load_queue_features()

    # Step 3: Aggregate to 15min
    log.info("\nStep 3: Aggregating to 15-minute bars...")
    t0 = time.time()
    bars_df = aggregate_to_15min(minute_df)
    del minute_df
    gc.collect()
    log.info(f"  Aggregated in {time.time()-t0:.1f}s")

    # Step 4: Queue merge
    log.info("\nStep 4: Merging queue features...")
    bars_df = add_queue_features_to_bars(bars_df, queue_df)
    del queue_df
    gc.collect()

    # Step 5: Rolling features
    log.info("\nStep 5: Adding rolling features...")
    t0 = time.time()
    bars_df = add_rolling_features(bars_df)
    log.info(f"  Done in {time.time()-t0:.1f}s")

    # Step 6: Forward labels
    log.info("\nStep 6: Computing forward labels...")
    bars_df = add_forward_labels(bars_df)

    # Save processed dataset
    bars_df.to_parquet(OUTPUT_DIR / "bars_15min_features.parquet", index=False)
    log.info(f"  Saved processed dataset: {len(bars_df):,} rows, {len(bars_df.columns)} cols")

    # Step 7: Walk-forward training
    log.info("\nStep 7: Two-stage walk-forward training...")
    summary = walk_forward_train(bars_df, args)

    # Final report
    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE — v3 ENSEMBLE RESULTS")
    log.info("=" * 60)

    if summary and "concat_results" in summary:
        for h, r in summary["concat_results"].items():
            log.info(
                f"  {h}: ENS IC={r['ensemble_concat_ic']:.4f}, "
                f"RankIC={r['ensemble_rank_ic']:.4f}, "
                f"DirAcc={r['ensemble_dir_acc']:.1%}, "
                f"IC_Sharpe={r['ensemble_ic_sharpe']:.2f} | "
                f"LGBM IC={r['lgbm_concat_ic']:.4f}, "
                f"IC_Sharpe={r['lgbm_ic_sharpe']:.2f}"
            )

    return summary


if __name__ == "__main__":
    main()
