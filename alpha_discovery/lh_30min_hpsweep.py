#!/usr/bin/env python3
"""
30-Minute LightGBM Hyperparameter Sweep
========================================

Random search over ~100 configs for the champion 30-min LightGBM (Strategy B).
Tests LightGBM hyperparams AND feature subsets (all, OFI+queue, momentum+vol).

Walk-Forward: 60d train, 10d val, slide 5d — SLIDING only (HC #0).
Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission).

Usage:
  cd /home/nick/Lvl3Quant && \
  /home/nick/miniconda3/envs/py311-train/bin/python -u \
      alpha_discovery/lh_30min_hpsweep.py

Author: Claude (autonomous research)
"""

import argparse
import csv
import gc
import json
import logging
import os
import random
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
OUTPUT_DIR = ROOT / "output" / "lh_30min_hpsweep"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [HP-SWEEP] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "lh_30min_hpsweep.log")),
    ],
)
log = logging.getLogger("HP-SWEEP")

# ─────────────────────────────────────────────
#  COST CONSTANTS
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
COST_RT_TICKS = 2.376
MIN_EDGE_TICKS = 2.5

# ─────────────────────────────────────────────
#  HYPERPARAMETER SEARCH SPACE
# ─────────────────────────────────────────────
HP_SPACE = {
    "num_leaves": [31, 63, 127, 255],
    "max_depth": [6, 8, 12, -1],
    "feature_fraction": [0.5, 0.7, 0.9],
    "bagging_fraction": [0.7, 0.8, 0.9],
    "min_data_in_leaf": [20, 50, 100],
    "learning_rate": [0.01, 0.05, 0.1],
    "reg_alpha": [0, 0.1, 1.0],
    "reg_lambda": [0, 0.1, 1.0],
}

FEATURE_GROUPS = {
    "all": None,  # use all features
    "ofi_queue": None,  # populated after feature computation
    "momentum_vol": None,  # populated after feature computation
}

# Deferred import
lgb = None


def _import_lightgbm():
    global lgb
    if lgb is not None:
        return
    import lightgbm as _lgb
    lgb = _lgb


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING (reused from longer_horizon_v4_focused.py)
# ═══════════════════════════════════════════════════════════════════


def load_all_minute_bars(min_date: str = "20250714") -> pd.DataFrame:
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


# ═══════════════════════════════════════════════════════════════════
#  BAR AGGREGATION + FEATURES (reused from longer_horizon_v4_focused.py)
# ═══════════════════════════════════════════════════════════════════


def _safe_polyfit_slope(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def aggregate_to_bars(minute_df: pd.DataFrame, bar_size_min: int = 30) -> pd.DataFrame:
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
            "total_volume": vol_arr.sum(),
            "avg_volume": vol_arr.mean(),
            "volume_trend": _safe_polyfit_slope(vol_arr),
            "volume_concentration": vol_arr.max() / max(vol_arr.mean(), 1),
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
            "signed_volume_sum": sv_arr.sum(),
            "signed_volume_ratio": sv_arr.sum() / max(vol_arr.sum(), 1),
            "buy_volume_frac": float(np.sum(sv_arr[sv_arr > 0])) / max(vol_arr.sum(), 1),
            "sell_volume_frac": float(-np.sum(sv_arr[sv_arr < 0])) / max(vol_arr.sum(), 1),
            "sweep_minutes": int(np.sum(np.abs(grp["sv_zscore"].values) > 2)),
            "max_sweep_intensity": float(np.abs(grp["sv_zscore"].values).max()),
            "sweep_direction": float(
                np.sign(sv_arr[np.abs(grp["sv_zscore"].values).argmax()])
            ) if len(sv_arr) > 0 else 0.0,
            "spread_mean": spread_arr.mean(),
            "spread_max": spread_arr.max(),
            "spread_trend": _safe_polyfit_slope(spread_arr),
            "trade_count_sum": tc_arr.sum(),
            "trade_count_trend": _safe_polyfit_slope(tc_arr),
            "realized_vol": float(np.std(ret_arr) * np.sqrt(252 * (390 // bar_size_min))) if len(ret_arr) > 1 else 0,
            "vol_of_vol": float(np.std(np.abs(ret_arr))) if len(ret_arr) > 1 else 0,
            "up_vol": float(np.std(ret_arr[ret_arr > 0])) if np.sum(ret_arr > 0) > 1 else 0,
            "down_vol": float(np.std(ret_arr[ret_arr < 0])) if np.sum(ret_arr < 0) > 1 else 0,
            "vwap_dev_mean": grp["vwap_dev"].mean(),
            "vwap_dev_trend": _safe_polyfit_slope(grp["vwap_dev"].values),
        }

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
    log.info(f"Aggregated {len(result):,} 30min bars")
    return result


def add_queue_features_to_bars(
    bars_df: pd.DataFrame, queue_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
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

    for bars, label in [(4, "lb_4bar"), (8, "lb_8bar"), (16, "lb_16bar")]:
        df[f"ret_{label}"] = df["close"].pct_change(bars)

    for w in [4, 8, 16]:
        df[f"rvol_{w}bar"] = df["return_bar"].rolling(w, min_periods=2).std()

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

    session_start_hour = 13.5
    session_end_hour = 20.0
    session_len = session_end_hour - session_start_hour

    hour_frac = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    progress = ((hour_frac - session_start_hour) / session_len).clip(0, 1)
    df["tod_sin"] = np.sin(2 * np.pi * progress)
    df["tod_cos"] = np.cos(2 * np.pi * progress)
    df["tod_progress"] = progress.values

    df["bars_since_open"] = df.groupby("date").cumcount()

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


# ═══════════════════════════════════════════════════════════════════
#  FEATURE GROUP CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════


def classify_feature_groups(all_features: List[str]) -> Dict[str, List[str]]:
    """
    Classify features into subsets:
    - ofi_queue: OFI, signed volume, queue, sweep, absorption, toxicity features
    - momentum_vol: return, vol, regime, vwap, range, close_position features
    """
    ofi_queue_keywords = [
        "ofi", "signed_volume", "buy_volume", "sell_volume", "sweep",
        "absorption", "queue", "q_bid", "q_ask", "q_ofi", "q_top",
        "q_microprice", "toxicity", "trade_count", "spread",
    ]
    momentum_vol_keywords = [
        "return", "ret_", "rvol", "realized_vol", "vol_of_vol", "vol_rel",
        "vol_asymmetry", "regime", "range_ticks", "close_position",
        "vwap", "volume_trend", "volume_concentration", "total_volume",
        "avg_volume", "tod_", "bars_since", "prev_day",
        "intraday_direction", "intraday_cum",
    ]

    ofi_queue = []
    momentum_vol = []

    for f in all_features:
        f_lower = f.lower()
        is_ofi = any(kw in f_lower for kw in ofi_queue_keywords)
        is_mom = any(kw in f_lower for kw in momentum_vol_keywords)

        if is_ofi:
            ofi_queue.append(f)
        if is_mom:
            momentum_vol.append(f)

    log.info(f"Feature groups: all={len(all_features)}, ofi_queue={len(ofi_queue)}, momentum_vol={len(momentum_vol)}")
    return {
        "all": all_features,
        "ofi_queue": ofi_queue,
        "momentum_vol": momentum_vol,
    }


# ═══════════════════════════════════════════════════════════════════
#  RANDOM SEARCH CONFIG GENERATION
# ═══════════════════════════════════════════════════════════════════


def sample_configs(n_configs: int, seed: int = 42) -> List[Dict]:
    """Sample n_configs random hyperparameter configurations."""
    rng = random.Random(seed)
    configs = []

    # Distribute across feature groups: ~60% all, ~20% ofi_queue, ~20% momentum_vol
    n_all = int(n_configs * 0.6)
    n_ofi = int(n_configs * 0.2)
    n_mom = n_configs - n_all - n_ofi

    group_assignments = (
        ["all"] * n_all + ["ofi_queue"] * n_ofi + ["momentum_vol"] * n_mom
    )
    rng.shuffle(group_assignments)

    for i, feat_group in enumerate(group_assignments):
        cfg = {
            "config_id": i,
            "feature_group": feat_group,
            "num_leaves": rng.choice(HP_SPACE["num_leaves"]),
            "max_depth": rng.choice(HP_SPACE["max_depth"]),
            "feature_fraction": rng.choice(HP_SPACE["feature_fraction"]),
            "bagging_fraction": rng.choice(HP_SPACE["bagging_fraction"]),
            "min_data_in_leaf": rng.choice(HP_SPACE["min_data_in_leaf"]),
            "learning_rate": rng.choice(HP_SPACE["learning_rate"]),
            "reg_alpha": rng.choice(HP_SPACE["reg_alpha"]),
            "reg_lambda": rng.choice(HP_SPACE["reg_lambda"]),
        }
        configs.append(cfg)

    return configs


# ═══════════════════════════════════════════════════════════════════
#  WALK-FORWARD FOR ONE CONFIG
# ═══════════════════════════════════════════════════════════════════


def run_one_config(
    config: Dict,
    features_all: np.ndarray,
    labels_all: np.ndarray,
    dates_all: np.ndarray,
    feature_cols: List[str],
    feature_groups: Dict[str, List[str]],
    all_dates: List[str],
    day_returns: Dict[str, float],
    train_days: int = 60,
    val_days: int = 10,
    slide: int = 5,
) -> Dict:
    """Run walk-forward for a single config. Returns metrics dict."""
    _import_lightgbm()

    cfg_id = config["config_id"]
    feat_group = config["feature_group"]

    # Select feature subset
    group_features = feature_groups[feat_group]
    feat_indices = [feature_cols.index(f) for f in group_features if f in feature_cols]

    if len(feat_indices) < 5:
        return {
            "config_id": cfg_id,
            "error": f"too few features ({len(feat_indices)}) for group {feat_group}",
        }

    feat_subset = features_all[:, feat_indices]
    feat_names = [feature_cols[i] for i in feat_indices]

    # LightGBM params from config
    lgbm_params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": config["learning_rate"],
        "num_leaves": config["num_leaves"],
        "min_child_samples": config["min_data_in_leaf"],
        "feature_fraction": config["feature_fraction"],
        "bagging_fraction": config["bagging_fraction"],
        "bagging_freq": 5,
        "lambda_l1": config["reg_alpha"],
        "lambda_l2": config["reg_lambda"],
        "max_depth": config["max_depth"],
        "verbose": -1,
        "n_jobs": -1,
    }

    all_oot_preds = []
    all_oot_actuals = []
    all_oot_dates_list = []
    fold_ics = []
    fold_idx = 0

    for fold_start in range(train_days, len(all_dates) - val_days + 1, slide):
        fold_train_dates = all_dates[fold_start - train_days: fold_start]
        fold_val_dates = all_dates[fold_start: fold_start + val_days]

        if len(fold_val_dates) < val_days:
            break

        fold_idx += 1

        # Split
        train_mask = np.isin(dates_all, fold_train_dates)
        val_mask = np.isin(dates_all, fold_val_dates)

        X_train_raw = feat_subset[train_mask].copy()
        X_val_raw = feat_subset[val_mask].copy()

        # Robust scaling from train only
        train_median = np.nanmedian(X_train_raw, axis=0)
        q75 = np.nanpercentile(X_train_raw, 75, axis=0)
        q25 = np.nanpercentile(X_train_raw, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0

        X_train = np.clip(np.nan_to_num((X_train_raw - train_median) / iqr, nan=0.0, posinf=3.0, neginf=-3.0), -5, 5)
        X_val = np.clip(np.nan_to_num((X_val_raw - train_median) / iqr, nan=0.0, posinf=3.0, neginf=-3.0), -5, 5)

        y_train = labels_all[train_mask]
        y_val = labels_all[val_mask]
        val_dates_fold = dates_all[val_mask]

        # Remove NaN targets
        train_valid = ~np.isnan(y_train)
        val_valid = ~np.isnan(y_val)

        if train_valid.sum() < 50 or val_valid.sum() < 10:
            continue

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]
        X_v = X_val[val_valid]
        y_v = y_val[val_valid]

        params = {**lgbm_params, "seed": 42 + fold_idx}

        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names)
        val_data = lgb.Dataset(X_v, label=y_v, feature_name=feat_names, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        try:
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[val_data],
                callbacks=callbacks,
            )
        except Exception as e:
            log.warning(f"Config {cfg_id} fold {fold_idx} train failed: {e}")
            continue

        val_preds = model.predict(X_v, num_iteration=model.best_iteration)

        if len(val_preds) > 5:
            ic = np.corrcoef(val_preds, y_v)[0, 1]
            if not np.isnan(ic):
                fold_ics.append(ic)

        all_oot_preds.append(val_preds)
        all_oot_actuals.append(y_v)
        all_oot_dates_list.append(val_dates_fold[val_valid])

        del model, train_data, val_data
        gc.collect()

    if not all_oot_preds:
        return {"config_id": cfg_id, "error": "no valid folds"}

    # Concat OOT
    concat_preds = np.concatenate(all_oot_preds)
    concat_actuals = np.concatenate(all_oot_actuals)
    concat_dates = np.concatenate(all_oot_dates_list)

    concat_ic = float(np.corrcoef(concat_preds, concat_actuals)[0, 1])
    ic_sharpe = (
        float(np.mean(fold_ics) / max(np.std(fold_ics), 1e-6))
        if len(fold_ics) > 2 else float("nan")
    )

    # Trade sim at top 15%
    sharpe_15 = float("nan")
    sortino_15 = float("nan")
    wr_15 = float("nan")
    pf_15 = float("nan")
    n_trades_15 = 0
    avg_pnl_15 = float("nan")
    long_sharpe_15 = float("nan")
    short_sharpe_15 = float("nan")

    valid = ~np.isnan(concat_preds) & ~np.isnan(concat_actuals)
    p_v = concat_preds[valid]
    a_v = concat_actuals[valid]
    d_v = concat_dates[valid]

    if len(p_v) >= 20:
        upper = np.quantile(p_v, 0.85)
        lower = np.quantile(p_v, 0.15)

        trades_pnl = []
        trades_dir = []
        for i in range(len(p_v)):
            if p_v[i] >= upper:
                trades_pnl.append(a_v[i] - COST_RT_TICKS)
                trades_dir.append("long")
            elif p_v[i] <= lower:
                trades_pnl.append(-a_v[i] - COST_RT_TICKS)
                trades_dir.append("short")

        if trades_pnl:
            pnl_arr = np.array(trades_pnl)
            n_trades_15 = len(pnl_arr)
            avg_pnl_15 = float(pnl_arr.mean())
            wr_15 = float(np.mean(pnl_arr > 0))
            sharpe_15 = float(pnl_arr.mean() / max(pnl_arr.std(), 1e-6) * np.sqrt(252))
            downside = np.sqrt(np.mean(np.minimum(pnl_arr, 0) ** 2))
            sortino_15 = float(pnl_arr.mean() / max(downside, 1e-6) * np.sqrt(252))
            pf_15 = float(np.sum(pnl_arr[pnl_arr > 0]) / max(-np.sum(pnl_arr[pnl_arr < 0]), 1e-6))

            # Per-side sharpe
            long_pnl = np.array([p for p, d in zip(trades_pnl, trades_dir) if d == "long"])
            short_pnl = np.array([p for p, d in zip(trades_pnl, trades_dir) if d == "short"])
            if len(long_pnl) > 2:
                long_sharpe_15 = float(long_pnl.mean() / max(long_pnl.std(), 1e-6) * np.sqrt(252))
            if len(short_pnl) > 2:
                short_sharpe_15 = float(short_pnl.mean() / max(short_pnl.std(), 1e-6) * np.sqrt(252))

    # Regime gap at top 15%
    regime_gap = float("nan")
    regime_gap_pass = False
    if len(p_v) >= 20:
        day_class = {}
        for d, ret in day_returns.items():
            if ret > 0.001:
                day_class[d] = "green"
            elif ret < -0.001:
                day_class[d] = "red"
            else:
                day_class[d] = "flat"

        regime_arr = np.array([day_class.get(d, "flat") for d in d_v])
        regime_sharpes = {}

        for regime in ["green", "red"]:
            mask = regime_arr == regime
            if mask.sum() < 10:
                continue
            p_r = p_v[mask]
            a_r = a_v[mask]
            upper_r = np.quantile(p_r, 0.85)
            lower_r = np.quantile(p_r, 0.15)
            r_pnl = []
            for i in range(len(p_r)):
                if p_r[i] >= upper_r:
                    r_pnl.append(a_r[i] - COST_RT_TICKS)
                elif p_r[i] <= lower_r:
                    r_pnl.append(-a_r[i] - COST_RT_TICKS)
            if len(r_pnl) > 2:
                r_pnl_arr = np.array(r_pnl)
                regime_sharpes[regime] = float(r_pnl_arr.mean() / max(r_pnl_arr.std(), 1e-6) * np.sqrt(252))

        if "green" in regime_sharpes and "red" in regime_sharpes:
            s_g = regime_sharpes["green"]
            s_r = regime_sharpes["red"]
            denom = max(abs(s_g), abs(s_r), 1e-6)
            regime_gap = abs(s_g - s_r) / denom
            regime_gap_pass = regime_gap <= 0.50

    return {
        "config_id": cfg_id,
        "feature_group": feat_group,
        "n_features": len(feat_indices),
        "num_leaves": config["num_leaves"],
        "max_depth": config["max_depth"],
        "feature_fraction": config["feature_fraction"],
        "bagging_fraction": config["bagging_fraction"],
        "min_data_in_leaf": config["min_data_in_leaf"],
        "learning_rate": config["learning_rate"],
        "reg_alpha": config["reg_alpha"],
        "reg_lambda": config["reg_lambda"],
        "n_folds": fold_idx,
        "n_valid_folds": len(fold_ics),
        "concat_ic": concat_ic,
        "ic_sharpe": ic_sharpe,
        "sharpe_15pct": sharpe_15,
        "sortino_15pct": sortino_15,
        "wr_15pct": wr_15,
        "pf_15pct": pf_15,
        "n_trades_15pct": n_trades_15,
        "avg_pnl_15pct": avg_pnl_15,
        "long_sharpe_15pct": long_sharpe_15,
        "short_sharpe_15pct": short_sharpe_15,
        "regime_gap": regime_gap,
        "regime_gap_pass": regime_gap_pass,
    }


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="30-min LightGBM HP Sweep")
    parser.add_argument("--n-configs", type=int, default=100, help="Number of random configs")
    parser.add_argument("--train-days", type=int, default=60, help="Training window")
    parser.add_argument("--val-days", type=int, default=10, help="Validation window")
    parser.add_argument("--slide-days", type=int, default=5, help="Slide step")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("30-MINUTE LightGBM HYPERPARAMETER SWEEP")
    log.info("=" * 70)
    log.info(f"Configs: {args.n_configs}, WF: {args.train_days}d/{args.val_days}d/slide {args.slide_days}d")
    log.info(f"Cost: {COST_RT_TICKS} ticks RT")

    t0 = time.time()

    # ── Load data ──
    log.info("\n--- Loading data ---")
    minute_df = load_all_minute_bars()
    queue_df = load_queue_features()

    # ── Build 30-min bars with all features ──
    log.info("\n--- Building 30-min bars ---")
    bars_df = aggregate_to_bars(minute_df, bar_size_min=30)
    bars_df.attrs["bar_size_min"] = 30
    bars_df = add_queue_features_to_bars(bars_df, queue_df)
    bars_df = add_rolling_features(bars_df)
    bars_df = add_forward_labels(bars_df, horizon_bars=1)

    # ── Feature columns ──
    all_feature_cols = get_feature_columns(bars_df)
    log.info(f"Total features: {len(all_feature_cols)}")

    # ── Classify feature groups ──
    feature_groups = classify_feature_groups(all_feature_cols)

    # ── Prepare arrays ──
    features_all = bars_df[all_feature_cols].values.astype(np.float32)
    labels_all = bars_df["fwd_ticks"].values.astype(np.float32)
    dates_all = bars_df["date"].values
    all_dates = sorted(bars_df["date"].unique())

    day_close = bars_df.groupby("date")["close"].last()
    day_returns = day_close.pct_change().to_dict()

    log.info(f"Total days: {len(all_dates)} ({all_dates[0]} -> {all_dates[-1]})")

    # Free memory
    del minute_df, queue_df
    gc.collect()

    # ── Generate configs ──
    configs = sample_configs(args.n_configs, seed=args.seed)
    log.info(f"\nGenerated {len(configs)} random configs")

    # Feature group distribution
    group_counts = {}
    for c in configs:
        g = c["feature_group"]
        group_counts[g] = group_counts.get(g, 0) + 1
    log.info(f"Feature group distribution: {group_counts}")

    # ── CSV output ──
    csv_path = OUTPUT_DIR / "sweep_results.csv"
    csv_fields = [
        "config_id", "feature_group", "n_features",
        "num_leaves", "max_depth", "feature_fraction", "bagging_fraction",
        "min_data_in_leaf", "learning_rate", "reg_alpha", "reg_lambda",
        "n_folds", "n_valid_folds", "concat_ic", "ic_sharpe",
        "sharpe_15pct", "sortino_15pct", "wr_15pct", "pf_15pct",
        "n_trades_15pct", "avg_pnl_15pct",
        "long_sharpe_15pct", "short_sharpe_15pct",
        "regime_gap", "regime_gap_pass",
    ]

    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
        writer.writeheader()

    # ── Run sweep ──
    results = []
    for i, config in enumerate(configs):
        t_cfg = time.time()
        log.info(
            f"\n--- Config {i+1}/{len(configs)} (id={config['config_id']}) ---"
            f" feat_group={config['feature_group']}"
            f" leaves={config['num_leaves']} depth={config['max_depth']}"
            f" lr={config['learning_rate']} ff={config['feature_fraction']}"
        )

        result = run_one_config(
            config=config,
            features_all=features_all,
            labels_all=labels_all,
            dates_all=dates_all,
            feature_cols=all_feature_cols,
            feature_groups=feature_groups,
            all_dates=all_dates,
            day_returns=day_returns,
            train_days=args.train_days,
            val_days=args.val_days,
            slide=args.slide_days,
        )

        elapsed_cfg = time.time() - t_cfg
        results.append(result)

        if "error" in result:
            log.warning(f"  Config {config['config_id']} ERROR: {result['error']} ({elapsed_cfg:.1f}s)")
        else:
            log.info(
                f"  Config {config['config_id']}: "
                f"IC={result['concat_ic']:.4f}, IC_Sharpe={result['ic_sharpe']:.3f}, "
                f"Sharpe@15%={result['sharpe_15pct']:.2f}, "
                f"WR={result['wr_15pct']:.1%}, PF={result['pf_15pct']:.2f}, "
                f"regime_gap={result['regime_gap']:.2f} "
                f"({'PASS' if result['regime_gap_pass'] else 'FAIL'})"
                f" ({elapsed_cfg:.1f}s)"
            )

            # Append to CSV
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=csv_fields, extrasaction="ignore")
                writer.writerow(result)

        # Progress estimate
        elapsed_total = time.time() - t0
        avg_per_cfg = elapsed_total / (i + 1)
        remaining = avg_per_cfg * (len(configs) - i - 1)
        log.info(f"  Progress: {i+1}/{len(configs)}, ETA: {remaining/60:.0f}min")

        gc.collect()

    # ── Final report ──
    valid_results = [r for r in results if "error" not in r]
    log.info(f"\n{'=' * 70}")
    log.info(f"SWEEP COMPLETE: {len(valid_results)}/{len(configs)} configs succeeded")
    log.info(f"{'=' * 70}")

    if not valid_results:
        log.error("No valid results!")
        return

    # Sort by IC_Sharpe (primary), then Sharpe@15% (secondary)
    valid_results.sort(
        key=lambda r: (r.get("ic_sharpe", -999), r.get("sharpe_15pct", -999)),
        reverse=True,
    )

    # Top 5 overall
    log.info("\n--- TOP 5 CONFIGS (by IC_Sharpe) ---")
    for rank, r in enumerate(valid_results[:5], 1):
        log.info(
            f"  #{rank} [config {r['config_id']}] "
            f"feat_group={r['feature_group']} ({r['n_features']} feats) "
            f"| IC={r['concat_ic']:.4f} IC_Sharpe={r['ic_sharpe']:.3f} "
            f"| Sharpe@15%={r['sharpe_15pct']:.2f} Sortino={r['sortino_15pct']:.2f} "
            f"WR={r['wr_15pct']:.1%} PF={r['pf_15pct']:.2f} "
            f"| trades={r['n_trades_15pct']} avg_pnl={r['avg_pnl_15pct']:.2f}t "
            f"| long_sharpe={r['long_sharpe_15pct']:.2f} short_sharpe={r['short_sharpe_15pct']:.2f} "
            f"| regime_gap={r['regime_gap']:.2f} {'PASS' if r['regime_gap_pass'] else 'FAIL'}"
        )
        log.info(
            f"    params: leaves={r['num_leaves']} depth={r['max_depth']} "
            f"lr={r['learning_rate']} ff={r['feature_fraction']} "
            f"bf={r['bagging_fraction']} min_leaf={r['min_data_in_leaf']} "
            f"alpha={r['reg_alpha']} lambda={r['reg_lambda']}"
        )

    # Top 5 that pass regime gap
    passing = [r for r in valid_results if r.get("regime_gap_pass")]
    if passing:
        passing.sort(key=lambda r: r.get("sharpe_15pct", -999), reverse=True)
        log.info(f"\n--- TOP 5 REGIME-AGNOSTIC (gap<=0.50, by Sharpe@15%) --- [{len(passing)} total pass]")
        for rank, r in enumerate(passing[:5], 1):
            log.info(
                f"  #{rank} [config {r['config_id']}] "
                f"feat_group={r['feature_group']} ({r['n_features']} feats) "
                f"| IC={r['concat_ic']:.4f} IC_Sharpe={r['ic_sharpe']:.3f} "
                f"| Sharpe@15%={r['sharpe_15pct']:.2f} WR={r['wr_15pct']:.1%} PF={r['pf_15pct']:.2f} "
                f"| regime_gap={r['regime_gap']:.2f} PASS "
                f"| long={r['long_sharpe_15pct']:.2f} short={r['short_sharpe_15pct']:.2f}"
            )
    else:
        log.info("\n--- NO CONFIGS PASS REGIME GAP (<=0.50) ---")

    # Feature group analysis
    log.info("\n--- FEATURE GROUP SUMMARY ---")
    for group in ["all", "ofi_queue", "momentum_vol"]:
        group_results = [r for r in valid_results if r.get("feature_group") == group]
        if group_results:
            ics = [r["concat_ic"] for r in group_results]
            sharpes = [r["sharpe_15pct"] for r in group_results if not np.isnan(r["sharpe_15pct"])]
            log.info(
                f"  {group}: n={len(group_results)}, "
                f"IC mean={np.mean(ics):.4f} (std={np.std(ics):.4f}), "
                f"Sharpe@15% mean={np.mean(sharpes):.2f} (std={np.std(sharpes):.2f})"
            )

    # Compare to baseline
    log.info(
        "\n--- BASELINE (current champion) ---"
        "\n  IC_Sharpe=1.044, Sharpe@15%=2.55, regime_gap=0.27 PASS"
        "\n  params: leaves=63, depth=8, lr=0.05, ff=0.7, bf=0.8, min_leaf=30, alpha=0.1, lambda=1.0"
    )

    # Save JSON summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_configs": len(configs),
        "n_valid": len(valid_results),
        "baseline": {
            "ic_sharpe": 1.044, "sharpe_15pct": 2.55, "regime_gap": 0.27,
        },
        "top5": valid_results[:5],
        "top5_regime_pass": passing[:5] if passing else [],
        "feature_group_summary": {
            group: {
                "n": len([r for r in valid_results if r.get("feature_group") == group]),
                "mean_ic": float(np.mean([r["concat_ic"] for r in valid_results if r.get("feature_group") == group])),
                "mean_sharpe": float(np.mean([
                    r["sharpe_15pct"] for r in valid_results
                    if r.get("feature_group") == group and not np.isnan(r["sharpe_15pct"])
                ])) if any(
                    not np.isnan(r["sharpe_15pct"]) for r in valid_results if r.get("feature_group") == group
                ) else None,
            }
            for group in ["all", "ofi_queue", "momentum_vol"]
        },
    }

    json_path = OUTPUT_DIR / "sweep_summary.json"
    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)

    elapsed = time.time() - t0
    log.info(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    log.info(f"Results CSV: {csv_path}")
    log.info(f"Summary JSON: {json_path}")
    log.info("DONE")


if __name__ == "__main__":
    main()
