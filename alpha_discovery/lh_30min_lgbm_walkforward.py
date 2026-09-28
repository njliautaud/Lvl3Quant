#!/usr/bin/env python3
"""
30-Minute LightGBM Walk-Forward — CPU-Friendly concat_oot.npz Generator
========================================================================

Generates concat_oot.npz compatible with the Integrated Pipeline paper engine.
Uses the same data pipeline as lh_30min_lean_oot.py but runs pure walk-forward
with 60d sliding train, 1d OOT, producing predictions for every available date.

Output: /home/jupiter/Lvl3Quant/output/lh_30min_lgbm_wf/concat_oot.npz
  Keys: preds (float32), actuals (float32), dates (str), confs (float32)

Walk-Forward: 60d train, 1d OOT, slide 1d — SLIDING only (HC #0).
Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission).

Confidence: |prediction| / rolling_std(predictions) — simple certainty proxy,
then sigmoid-squashed to [0.45, 0.85] range to match deep model output.

Usage:
  cd /home/jupiter/Lvl3Quant && python -u alpha_discovery/lh_30min_lgbm_walkforward.py

Author: Claude (autonomous research)
"""

import gc
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
QUEUE_FEATURE_DIR = ROOT / "data" / "queue_augmented_features"
OUTPUT_DIR = ROOT / "output" / "lh_30min_lgbm_wf"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [LGBM-WF] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "lh_30min_lgbm_wf.log")),
    ],
)
log = logging.getLogger("LGBM-WF")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures -- AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376
COST_RT_TICKS = 2.376  # market entry + market exit + RT commission
MIN_EDGE_TICKS = 2.5

# Walk-forward config
TRAIN_DAYS = 60
OOT_DAYS = 1   # 1-day OOT per fold
SLIDE_DAYS = 1  # slide 1 day at a time for maximum coverage

# LightGBM hyperparameters
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 31,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_depth": 6,
    "verbose": -1,
    "n_jobs": -1,
    "seed": 42,
}
N_ESTIMATORS = 300

# Deferred import
lgb = None


def _import_lightgbm():
    global lgb
    if lgb is not None:
        return
    import lightgbm as _lgb
    lgb = _lgb


# ======================================================================
#  DATA LOADING (faithfully reused from lh_30min_lean_oot.py)
# ======================================================================


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
        log.info("No queue_augmented_features directory -- skipping")
        return None

    files = sorted(QUEUE_FEATURE_DIR.glob("features_*.parquet"))
    if not files:
        log.info("No queue feature files found -- skipping")
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


# ======================================================================
#  BAR AGGREGATION + FEATURES (faithfully reused from lh_30min_lean_oot.py)
# ======================================================================


def _safe_polyfit_slope(arr: np.ndarray) -> float:
    """Linear regression slope, returns 0 on failure."""
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def aggregate_to_bars(minute_df: pd.DataFrame, bar_size_min: int = 30) -> pd.DataFrame:
    """Aggregate 1-minute bars into 30-minute bars with microstructure features."""
    df = minute_df.copy()
    df["bar_key"] = df["ts_minute"].dt.floor(f"{bar_size_min}min")
    df["return_1m"] = df.groupby("date")["close"].pct_change()
    df["abs_ofi"] = df["ofi_1min"].abs()
    sv_std = df.groupby("date")["signed_volume"].transform("std").replace(0, 1)
    df["sv_zscore"] = df["signed_volume"] / sv_std
    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["close"].clip(lower=1)

    records = []
    for (date_str, bar_key), grp in df.groupby(["date", "bar_key"]):
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
            "bar_key": bar_key,
            "ts": grp["ts_minute"].iloc[0],
            # Price action
            "open": close_arr[0],
            "high": close_arr.max(),
            "low": close_arr.min(),
            "close": close_arr[-1],
            "return_bar": (close_arr[-1] / close_arr[0] - 1) if close_arr[0] > 0 else 0,
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
                ofi_arr[len(ofi_arr) // 2:].sum() - ofi_arr[:len(ofi_arr) // 2].sum()
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
            ) if len(sv_arr) > 0 else 0.0,
            # Spread & liquidity
            "spread_mean": spread_arr.mean(),
            "spread_max": spread_arr.max(),
            "spread_trend": _safe_polyfit_slope(spread_arr),
            # Trade intensity
            "trade_count_sum": tc_arr.sum(),
            "trade_count_trend": _safe_polyfit_slope(tc_arr),
            # Volatility
            "realized_vol": float(np.std(ret_arr) * np.sqrt(252 * (390 // bar_size_min))) if len(ret_arr) > 1 else 0,
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
    log.info(f"Aggregated {len(result):,} {bar_size_min}min bars")
    return result


def add_queue_features_to_bars(
    bars_df: pd.DataFrame, queue_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Merge queue-augmented tick features into bars."""
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
        log.warning("Queue dataframe has no expected columns -- skipping")
        return bars_df

    ts_col = None
    for candidate in ["ts", "timestamp", "ts_event"]:
        if candidate in queue_df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        log.warning("Queue dataframe has no timestamp column -- skipping merge")
        return bars_df

    queue_df = queue_df.copy()
    queue_df[ts_col] = pd.to_datetime(queue_df[ts_col], utc=True)
    queue_df["bar_key"] = queue_df[ts_col].dt.floor("30min")

    agg_dict = {}
    for c in available_mean:
        agg_dict[c] = ["mean", "std"]
    for c in available_sum:
        agg_dict[c] = ["sum"]
    for c in available_rate:
        agg_dict[c] = ["mean"]

    agg = queue_df.groupby(["date", "bar_key"]).agg(agg_dict)
    agg.columns = [f"q_{c}_{stat}" for c, stat in agg.columns]
    agg = agg.reset_index()

    if "bid_cancel_rate_1s" in queue_df.columns and "bid_add_rate_1s" in queue_df.columns:
        per_bar = queue_df.groupby(["date", "bar_key"]).agg(
            bid_cancel_mean=("bid_cancel_rate_1s", "mean"),
            bid_add_mean=("bid_add_rate_1s", "mean"),
            ask_cancel_mean=("ask_cancel_rate_1s", "mean"),
            ask_add_mean=("ask_add_rate_1s", "mean"),
        ).reset_index()
        per_bar["q_bid_toxicity"] = per_bar["bid_cancel_mean"] / per_bar["bid_add_mean"].clip(lower=1e-6)
        per_bar["q_ask_toxicity"] = per_bar["ask_cancel_mean"] / per_bar["ask_add_mean"].clip(lower=1e-6)
        agg = agg.merge(
            per_bar[["date", "bar_key", "q_bid_toxicity", "q_ask_toxicity"]],
            on=["date", "bar_key"], how="left",
        )

    bars_df = bars_df.merge(agg, on=["date", "bar_key"], how="left")
    n_queue_cols = len(agg.columns) - 2
    log.info(f"Merged queue features: {n_queue_cols} new columns")
    return bars_df


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
                df["sweep_minutes"].rolling(w, min_periods=1).sum() / (w * 30)
            )

    # Lookback returns at various horizons (in bar units)
    for bars, label in [(4, "lb_4bar"), (8, "lb_8bar"), (16, "lb_16bar")]:
        df[f"ret_{label}"] = df["close"].pct_change(bars)

    # Rolling realized vol
    for w in [4, 8, 16]:
        df[f"rvol_{w}bar"] = df["return_bar"].rolling(w, min_periods=2).std()

    # Intraday cumulative signals
    df["intraday_cum_ofi"] = df.groupby("date")["ofi_sum"].cumsum()
    df["intraday_cum_sv"] = df.groupby("date")["signed_volume_sum"].cumsum()

    # OFI sign flip indicator
    df["ofi_sign_flip"] = (
        np.sign(df["ofi_sum"]) != np.sign(df["ofi_sum"].shift(1))
    ).astype(np.float32)

    # Absorption: volume per tick of range
    df["absorption"] = df["total_volume"] / df["range_ticks"].clip(lower=1)

    # Previous day context (causal: uses ONLY completed prior day)
    day_stats = (
        df.groupby("date")
        .agg(
            day_ofi=("ofi_sum", "sum"),
            day_sv=("signed_volume_sum", "sum"),
            day_ret=("return_bar", "sum"),
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

    # Regime features
    df["regime_ret_16bar"] = df["close"].pct_change(16)
    df["regime_ret_32bar"] = df["close"].pct_change(32)

    rvol_short = df["return_bar"].rolling(4, min_periods=2).std()
    rvol_long = df["return_bar"].rolling(16, min_periods=4).std()
    df["regime_vol_ratio"] = rvol_short / rvol_long.clip(lower=1e-8)

    df["intraday_direction_strength"] = (
        df["intraday_cum_ofi"].abs()
        / df.groupby("date")["ofi_sum"].transform(
            lambda x: x.abs().cumsum()
        ).clip(lower=1)
    )

    log.info(f"Added rolling + regime features: {len(df.columns)} total columns")
    return df


def add_forward_labels(df: pd.DataFrame, horizon_bars: int = 1) -> pd.DataFrame:
    """Compute forward return labels with overnight gap protection."""
    df = df.sort_values("ts").reset_index(drop=True)

    fwd_close = df["close"].shift(-horizon_bars)
    fwd_ticks = (fwd_close - df["close"]) / 0.25

    ts_now = df["ts"].values
    ts_fwd = df["ts"].shift(-horizon_bars).values
    for i in range(len(df) - horizon_bars):
        if pd.isna(ts_fwd[i]):
            continue
        diff_s = (pd.Timestamp(ts_fwd[i]) - pd.Timestamp(ts_now[i])).total_seconds()
        if diff_s > 6 * 3600:
            fwd_ticks.iloc[i] = np.nan

    df["fwd_ticks"] = fwd_ticks
    df["direction"] = 0
    df.loc[fwd_ticks > MIN_EDGE_TICKS, "direction"] = 1
    df.loc[fwd_ticks < -MIN_EDGE_TICKS, "direction"] = -1

    log.info(
        f"Forward labels: {(~fwd_ticks.isna()).sum():,} valid, "
        f"{(df['direction'] == 1).sum():,} long, "
        f"{(df['direction'] == -1).sum():,} short"
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select feature columns, excluding labels, metadata, raw prices."""
    exclude_prefixes = (
        "fwd_", "direction", "trade_quality_",
        "date", "ts", "bar_key",
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


# ======================================================================
#  LEAKAGE AUDIT
# ======================================================================


def leakage_audit(
    train_dates: List[str],
    oot_dates: List[str],
    feature_cols: List[str],
) -> bool:
    """Explicit leakage audit. Returns True if all checks pass."""
    train_set = set(train_dates)
    oot_set = set(oot_dates)

    # No date overlap
    overlap = train_set & oot_set
    if overlap:
        log.error(f"LEAKAGE: Train/OOT date overlap: {overlap}")
        return False

    # Temporal order
    max_train = max(train_dates)
    min_oot = min(oot_dates)
    if max_train >= min_oot:
        log.error(f"LEAKAGE: Max train date {max_train} >= min OOT date {min_oot}")
        return False

    # No forward-looking features
    fwd_leak = [c for c in feature_cols if c.startswith(("fwd_", "direction", "trade_quality_"))]
    if fwd_leak:
        log.error(f"LEAKAGE: Forward-looking columns in features: {fwd_leak}")
        return False

    return True


# ======================================================================
#  CONFIDENCE COMPUTATION
# ======================================================================


def compute_confidence(preds: np.ndarray, window: int = 50) -> np.ndarray:
    """
    Compute confidence as sigmoid-squashed |prediction| / rolling_std.

    Higher |prediction| relative to recent prediction volatility = higher confidence.
    Output range is approximately [0.45, 0.85] to match the deep model's output.
    """
    abs_pred = np.abs(preds)

    # Rolling std of predictions (use expanding for first few samples)
    pred_std = np.full_like(preds, np.std(preds))  # fallback
    if len(preds) >= window:
        for i in range(len(preds)):
            start = max(0, i - window + 1)
            chunk = preds[start:i + 1]
            if len(chunk) >= 5:
                pred_std[i] = max(np.std(chunk), 1e-6)

    # Raw certainty: how extreme is this prediction relative to recent ones
    raw_cert = abs_pred / np.maximum(pred_std, 1e-6)

    # Sigmoid squash to [0.45, 0.85] range
    # sigmoid(x) maps (-inf, inf) -> (0, 1)
    # We want: raw_cert=0 -> ~0.5, raw_cert=2 -> ~0.73, raw_cert=4 -> ~0.82
    sigmoid = 1.0 / (1.0 + np.exp(-0.8 * (raw_cert - 1.0)))
    conf = 0.45 + 0.40 * sigmoid  # maps to [0.45, 0.85]

    return conf.astype(np.float32)


# ======================================================================
#  WALK-FORWARD ENGINE
# ======================================================================


def run_walkforward(
    features_all: np.ndarray,
    labels_all: np.ndarray,
    dates_all: np.ndarray,
    feature_cols: List[str],
    all_dates: List[str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Run pure walk-forward with 60d sliding train, 1d OOT.
    Returns: (preds, actuals, dates, confs) arrays for concat_oot.npz
    """
    _import_lightgbm()

    all_oot_preds = []
    all_oot_actuals = []
    all_oot_dates = []
    fold_count = 0
    skipped_count = 0

    total_folds = len(all_dates) - TRAIN_DAYS
    log.info(f"Walk-forward: {TRAIN_DAYS}d train, {OOT_DAYS}d OOT, {total_folds} potential folds")

    for fold_start in range(TRAIN_DAYS, len(all_dates)):
        # SLIDING window: always use exactly TRAIN_DAYS before this fold
        train_dates = all_dates[fold_start - TRAIN_DAYS: fold_start]
        oot_date = all_dates[fold_start]

        # Leakage audit
        if not leakage_audit(train_dates, [oot_date], feature_cols):
            skipped_count += 1
            continue

        # Build masks
        train_mask = np.isin(dates_all, train_dates)
        oot_mask = dates_all == oot_date

        X_train_raw = features_all[train_mask].copy()
        X_oot_raw = features_all[oot_mask].copy()
        y_train = labels_all[train_mask]
        y_oot = labels_all[oot_mask]

        # Filter valid labels
        train_valid = ~np.isnan(y_train)
        oot_valid = ~np.isnan(y_oot)

        if train_valid.sum() < 50 or oot_valid.sum() < 2:
            skipped_count += 1
            continue

        # Robust scaling: fit on train only
        train_median = np.nanmedian(X_train_raw, axis=0)
        q75 = np.nanpercentile(X_train_raw, 75, axis=0)
        q25 = np.nanpercentile(X_train_raw, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0

        X_train = np.clip(
            np.nan_to_num((X_train_raw - train_median) / iqr, nan=0.0, posinf=3.0, neginf=-3.0),
            -5, 5,
        )
        X_oot = np.clip(
            np.nan_to_num((X_oot_raw - train_median) / iqr, nan=0.0, posinf=3.0, neginf=-3.0),
            -5, 5,
        )

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]
        X_o = X_oot[oot_valid]
        y_o = y_oot[oot_valid]

        # Split last 15% of train as early stopping validation
        n_tr = len(X_tr)
        n_val_split = max(int(n_tr * 0.15), 30)
        X_tr_split = X_tr[:-n_val_split]
        y_tr_split = y_tr[:-n_val_split]
        X_val_split = X_tr[-n_val_split:]
        y_val_split = y_tr[-n_val_split:]

        # LightGBM training
        params = {**LGBM_PARAMS, "seed": 42 + fold_count}
        train_data = lgb.Dataset(X_tr_split, label=y_tr_split, feature_name=feature_cols)
        val_data = lgb.Dataset(
            X_val_split, label=y_val_split,
            feature_name=feature_cols, reference=train_data,
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        try:
            model = lgb.train(
                params, train_data, num_boost_round=N_ESTIMATORS,
                valid_sets=[val_data], callbacks=callbacks,
            )
        except Exception as e:
            log.warning(f"  Fold {fold_count} ({oot_date}) train failed: {e}")
            skipped_count += 1
            continue

        # Predict on OOT day
        oot_preds = model.predict(X_o, num_iteration=model.best_iteration)

        all_oot_preds.append(oot_preds)
        all_oot_actuals.append(y_o)
        all_oot_dates.append(np.array([oot_date] * len(y_o)))

        fold_count += 1

        # Progress logging every 20 folds
        if fold_count % 20 == 0 or fold_count == 1:
            ic_so_far = float("nan")
            if fold_count > 5:
                p_cat = np.concatenate(all_oot_preds)
                a_cat = np.concatenate(all_oot_actuals)
                if len(p_cat) > 10:
                    ic_so_far = np.corrcoef(p_cat, a_cat)[0, 1]
            log.info(
                f"  Fold {fold_count}/{total_folds}: OOT={oot_date}, "
                f"bars={len(y_o)}, running_IC={ic_so_far:.4f}"
            )

        del model, train_data, val_data
        gc.collect()

    log.info(
        f"Walk-forward complete: {fold_count} folds, {skipped_count} skipped"
    )

    if not all_oot_preds:
        raise RuntimeError("No valid walk-forward folds completed!")

    # Concatenate all OOT predictions
    concat_preds = np.concatenate(all_oot_preds).astype(np.float32)
    concat_actuals = np.concatenate(all_oot_actuals).astype(np.float32)
    concat_dates = np.concatenate(all_oot_dates)

    # Compute confidence
    concat_confs = compute_confidence(concat_preds)

    return concat_preds, concat_actuals, concat_dates, concat_confs


# ======================================================================
#  MAIN
# ======================================================================


def main():
    log.info("=" * 70)
    log.info("30-MINUTE LGBM WALK-FORWARD -- CPU-friendly concat_oot.npz generator")
    log.info("=" * 70)
    log.info(f"Walk-forward: {TRAIN_DAYS}d train, {OOT_DAYS}d OOT, slide {SLIDE_DAYS}d")
    log.info(f"LightGBM: n_estimators={N_ESTIMATORS}, lr={LGBM_PARAMS['learning_rate']}, "
             f"max_depth={LGBM_PARAMS['max_depth']}, num_leaves={LGBM_PARAMS['num_leaves']}")
    log.info(f"Output: {OUTPUT_DIR}")

    t0 = time.time()

    # ── Load data ──
    log.info("\n--- Loading data ---")
    minute_df = load_all_minute_bars()
    queue_df = load_queue_features()

    # ── Build 30-min bars with all features ──
    log.info("\n--- Building 30-min bars ---")
    bars_df = aggregate_to_bars(minute_df, bar_size_min=30)
    bars_df = add_queue_features_to_bars(bars_df, queue_df)
    bars_df = add_rolling_features(bars_df)
    bars_df = add_forward_labels(bars_df, horizon_bars=1)

    del minute_df, queue_df
    gc.collect()

    # ── Feature columns ──
    feature_cols = get_feature_columns(bars_df)
    log.info(f"Using {len(feature_cols)} features: {feature_cols[:10]}...")

    # ── Prepare arrays ──
    features_all = bars_df[feature_cols].values.astype(np.float32)
    labels_all = bars_df["fwd_ticks"].values.astype(np.float32)
    dates_all = bars_df["date"].values
    all_dates = sorted(bars_df["date"].unique())

    log.info(f"Total days: {len(all_dates)} ({all_dates[0]} -> {all_dates[-1]})")
    log.info(f"Total bars: {len(bars_df):,}")

    # ── Run walk-forward ──
    log.info("\n--- Running walk-forward ---")
    preds, actuals, dates, confs = run_walkforward(
        features_all, labels_all, dates_all, feature_cols, all_dates,
    )

    # ── Save concat_oot.npz ──
    npz_path = OUTPUT_DIR / "concat_oot.npz"
    np.savez_compressed(
        str(npz_path),
        preds=preds,
        actuals=actuals,
        dates=dates,
        confs=confs,
    )

    unique_dates = sorted(set(dates))
    log.info(f"\nSaved concat_oot.npz: {len(preds)} predictions across {len(unique_dates)} dates")
    log.info(f"  Preds range: [{preds.min():.3f}, {preds.max():.3f}]")
    log.info(f"  Confs range: [{confs.min():.3f}, {confs.max():.3f}]")
    log.info(f"  Date range: {unique_dates[0]} -> {unique_dates[-1]}")

    # ── Compute summary metrics ──
    concat_ic = float(np.corrcoef(preds, actuals)[0, 1])
    dir_acc = float(np.mean(np.sign(preds) == np.sign(actuals)))

    # Per-day IC
    per_day_ics = []
    for d in unique_dates:
        d_mask = dates == d
        if d_mask.sum() < 3:
            continue
        d_preds = preds[d_mask]
        d_actuals = actuals[d_mask]
        d_ic = np.corrcoef(d_preds, d_actuals)[0, 1]
        if not np.isnan(d_ic):
            per_day_ics.append(d_ic)

    ic_mean = np.mean(per_day_ics) if per_day_ics else 0.0
    ic_std = np.std(per_day_ics) if len(per_day_ics) > 2 else 1.0
    ic_sharpe = ic_mean / max(ic_std, 1e-6)

    log.info(f"\n--- CONCAT OOT METRICS ---")
    log.info(f"  Concat IC:  {concat_ic:.4f}")
    log.info(f"  IC Sharpe:  {ic_sharpe:.3f}")
    log.info(f"  Dir Acc:    {dir_acc:.3f}")
    log.info(f"  Per-day IC: mean={ic_mean:.4f}, std={ic_std:.4f}")
    log.info(f"  Predictions: {len(preds):,}")
    log.info(f"  OOT days:   {len(unique_dates)}")

    # ── Quick trade simulation ──
    for conf_pct in [0.10, 0.15, 0.20]:
        valid = ~np.isnan(preds) & ~np.isnan(actuals)
        p = preds[valid]
        a = actuals[valid]

        upper = np.quantile(p, 1 - conf_pct)
        lower = np.quantile(p, conf_pct)

        trade_pnls = []
        for i in range(len(p)):
            if p[i] >= upper:
                trade_pnls.append(a[i] - COST_RT_TICKS)
            elif p[i] <= lower:
                trade_pnls.append(-a[i] - COST_RT_TICKS)

        if trade_pnls:
            arr = np.array(trade_pnls)
            sharpe = arr.mean() / max(arr.std(), 1e-6) * np.sqrt(252)
            wr = np.mean(arr > 0)
            pf = np.sum(arr[arr > 0]) / max(-np.sum(arr[arr < 0]), 1e-6)
            log.info(
                f"  Trades top {int(conf_pct*100)}%: n={len(arr)}, "
                f"Sharpe={sharpe:.2f}, WR={wr:.1%}, PF={pf:.2f}, "
                f"avg={arr.mean():.2f}t, total=${arr.sum() * ES_TICK_VALUE:.0f}"
            )

    # ── MLflow logging ──
    use_mlflow = True
    try:
        import mlflow

        mlflow_uri = "http://localhost:5000"
        mlflow.set_tracking_uri(mlflow_uri)
        mlflow.set_experiment("lh_30min_lgbm_wf")

        run_name = f"lgbm_wf_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "model": "LightGBM",
                "mode": "walk-forward",
                "train_days": TRAIN_DAYS,
                "oot_days": OOT_DAYS,
                "slide_days": SLIDE_DAYS,
                "n_estimators": N_ESTIMATORS,
                "learning_rate": LGBM_PARAMS["learning_rate"],
                "max_depth": LGBM_PARAMS["max_depth"],
                "num_leaves": LGBM_PARAMS["num_leaves"],
                "subsample": LGBM_PARAMS["bagging_fraction"],
                "colsample_bytree": LGBM_PARAMS["feature_fraction"],
                "min_child_samples": LGBM_PARAMS["min_child_samples"],
                "reg_alpha": LGBM_PARAMS["lambda_l1"],
                "reg_lambda": LGBM_PARAMS["lambda_l2"],
                "n_features": len(feature_cols),
                "n_total_days": len(all_dates),
                "cost_rt_ticks": COST_RT_TICKS,
            })

            mlflow.log_metrics({
                "concat_ic": concat_ic,
                "ic_sharpe": ic_sharpe,
                "dir_acc": dir_acc,
                "n_predictions": len(preds),
                "n_oot_days": len(unique_dates),
                "ic_mean_daily": ic_mean,
                "ic_std_daily": ic_std,
            })

            # Log trade sim at 10%
            valid = ~np.isnan(preds) & ~np.isnan(actuals)
            p = preds[valid]
            a = actuals[valid]
            upper = np.quantile(p, 0.90)
            lower = np.quantile(p, 0.10)
            trade_pnls = []
            for i in range(len(p)):
                if p[i] >= upper:
                    trade_pnls.append(a[i] - COST_RT_TICKS)
                elif p[i] <= lower:
                    trade_pnls.append(-a[i] - COST_RT_TICKS)
            if trade_pnls:
                arr = np.array(trade_pnls)
                mlflow.log_metrics({
                    "trade_sharpe_10pct": float(arr.mean() / max(arr.std(), 1e-6) * np.sqrt(252)),
                    "trade_wr_10pct": float(np.mean(arr > 0)),
                    "trade_pf_10pct": float(np.sum(arr[arr > 0]) / max(-np.sum(arr[arr < 0]), 1e-6)),
                    "trade_n_10pct": len(arr),
                })

            mlflow.log_artifact(str(npz_path))
            log.info(f"MLflow run logged: {run_name}")

    except Exception as e:
        log.warning(f"MLflow logging failed (non-fatal): {e}")

    # ── Save summary JSON ──
    summary = {
        "model": "LightGBM",
        "mode": "walk-forward",
        "train_days": TRAIN_DAYS,
        "oot_days": OOT_DAYS,
        "slide_days": SLIDE_DAYS,
        "lgbm_params": LGBM_PARAMS,
        "n_estimators": N_ESTIMATORS,
        "n_features": len(feature_cols),
        "feature_cols": feature_cols,
        "concat_oot": {
            "ic": concat_ic,
            "ic_sharpe": ic_sharpe,
            "dir_acc": dir_acc,
            "n_predictions": int(len(preds)),
            "n_oot_days": len(unique_dates),
            "date_range": f"{unique_dates[0]} -> {unique_dates[-1]}",
            "preds_range": [float(preds.min()), float(preds.max())],
            "confs_range": [float(confs.min()), float(confs.max())],
        },
        "per_day_ic": {
            "mean": float(ic_mean),
            "std": float(ic_std),
            "n_days": len(per_day_ics),
        },
        "runtime_minutes": (time.time() - t0) / 60,
        "timestamp": datetime.now().isoformat(),
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    log.info(f"\nRuntime: {elapsed / 60:.1f} minutes")
    log.info(f"Output: {npz_path}")
    log.info("DONE")


if __name__ == "__main__":
    main()
