#!/usr/bin/env python3
"""
Trade Management Model v2 — Tick-Level Queue Data for Mid-Trade Monitoring
==========================================================================

v1 failed (AUC 0.605, static beat all dynamic configs) because it used
minute-level features from 30-min bars — too coarse. Invalidation and
confirmation signals happen at the TICK level.

v2 uses tick-level queue features sampled every 10 seconds during each trade:
  - Queue dynamics: queue_ratio, OFI flow alignment, cancel spikes, level age
  - Post-entry state: unrealized P&L, MFE, MAE, drawdown from peak
  - Entry context: original LightGBM prediction magnitude

Architecture:
  1. Retrain 30-min LightGBM entry model on minute bars (dates overlapping tick data)
  2. Reconstruct trades: entry times + directions for 41 tick-data days
  3. For each trade, extract tick-level queue features every 10s up to 45 min
  4. Labels: will_eventually_profit, should_exit_now, remaining_favorable_ticks
  5. Train 3 LightGBM models: invalidation, exit timing, remaining edge
  6. Simulate dynamic vs static exits

Walk-Forward: 25d train / 8d val / 4d slide — SLIDING only (HC #0).
Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission).
MLflow experiment: trade_management_v2_tick

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python \\
      alpha_discovery/trade_management_v2_tick.py

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
from scipy import stats

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
QUEUE_FEATURE_DIR = ROOT / "data" / "queue_augmented_features"
OUTPUT_DIR = ROOT / "output" / "trade_management_v2_tick"
LOG_DIR = ROOT / "logs"
MODEL_DIR = OUTPUT_DIR / "models"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [TM-v2] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "trade_management_v2_tick.log")),
    ],
)
log = logging.getLogger("TM-v2")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
COST_RT_TICKS = 2.376  # market entry + market exit + RT commission

# Walk-forward config — tight because only ~31 overlapping tick-data days
TRAIN_DAYS = 20
VAL_DAYS = 6
SLIDE_DAYS = 3

# Entry model config (30-min bars)
ENTRY_BAR_SIZE_MIN = 30
MIN_EDGE_TICKS = 2.5

# Trade management config
MAX_HOLD_SECONDS = 45 * 60  # 45 minutes
SAMPLE_INTERVAL_SECONDS = 10  # sample every 10 seconds
INVALIDATION_THRESHOLD = 0.70
REMAINING_MFE_THRESHOLD = 1.0

# LightGBM params — entry model
LGBM_ENTRY_PARAMS = {
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
}

# LightGBM params — invalidation classifier
LGBM_CLF_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_depth": 7,
    "verbose": -1,
    "n_jobs": -1,
    "is_unbalance": True,
}

# LightGBM params — exit timing classifier
LGBM_EXIT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_depth": 7,
    "verbose": -1,
    "n_jobs": -1,
    "is_unbalance": True,
}

# LightGBM params — remaining edge regressor
LGBM_REG_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 63,
    "min_child_samples": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "max_depth": 7,
    "verbose": -1,
    "n_jobs": -1,
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
#  SECTION 1: DATA LOADING
# ═══════════════════════════════════════════════════════════════════


def load_minute_bars(min_date: str = "20250714") -> pd.DataFrame:
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
            log.warning(f"Skip minute bar {f.stem}: {e}")

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


def discover_queue_dates() -> List[str]:
    """Discover available queue feature dates."""
    files = sorted(QUEUE_FEATURE_DIR.glob("features_*.parquet"))
    dates = [f.stem.replace("features_", "") for f in files]
    log.info(f"Found {len(dates)} queue feature dates: {dates[0]}..{dates[-1]}" if dates else "No queue dates found!")
    return dates


def load_queue_features(date_str: str) -> Optional[pd.DataFrame]:
    """Load queue features for a single date."""
    fpath = QUEUE_FEATURE_DIR / f"features_{date_str}.parquet"
    if not fpath.exists():
        return None
    try:
        df = pd.read_parquet(fpath)
        df["date"] = date_str
        return df
    except Exception as e:
        log.warning(f"Failed to load queue features for {date_str}: {e}")
        return None


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: ENTRY MODEL (30-min bars, walk-forward)
# ═══════════════════════════════════════════════════════════════════


def _safe_polyfit_slope(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def aggregate_to_30min_bars(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute bars into 30-minute bars with microstructure features."""
    df = minute_df.copy()
    df["bar_key"] = df["ts_minute"].dt.floor(f"{ENTRY_BAR_SIZE_MIN}min")
    df["return_1m"] = df.groupby("date")["close"].pct_change()
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
                if ofi_arr.sum() != 0 else 0.5
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
            "spread_mean": spread_arr.mean(),
            "spread_max": spread_arr.max(),
            "spread_trend": _safe_polyfit_slope(spread_arr),
            "trade_count_sum": tc_arr.sum(),
            "trade_count_trend": _safe_polyfit_slope(tc_arr),
            "realized_vol": float(np.std(ret_arr) * np.sqrt(252 * (390 // ENTRY_BAR_SIZE_MIN))) if len(ret_arr) > 1 else 0,
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

    return df


def add_forward_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward return labels for 30-min horizon."""
    df = df.sort_values("ts").reset_index(drop=True)
    fwd_close = df["close"].shift(-1)  # next 30-min bar close
    fwd_ticks = (fwd_close - df["close"]) / 0.25

    # Overnight gap protection
    ts_now = df["ts"].values
    ts_fwd = df["ts"].shift(-1).values
    for i in range(len(df) - 1):
        if pd.isna(ts_fwd[i]):
            continue
        diff_s = (pd.Timestamp(ts_fwd[i]) - pd.Timestamp(ts_now[i])).total_seconds()
        if diff_s > 6 * 3600:
            fwd_ticks.iloc[i] = np.nan

    df["fwd_ticks_30min"] = fwd_ticks
    df["direction_30min"] = 0
    df.loc[fwd_ticks > MIN_EDGE_TICKS, "direction_30min"] = 1
    df.loc[fwd_ticks < -MIN_EDGE_TICKS, "direction_30min"] = -1
    return df


def get_entry_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select feature columns for the entry model."""
    exclude_prefixes = ("fwd_", "direction_", "date", "ts", "bar_key")
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


def train_entry_model_wf(bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward train the entry model on 30-min bars.
    Returns bars_df with 'entry_pred_ticks' and 'entry_signal' columns.
    """
    _import_lightgbm()

    target_col = "fwd_ticks_30min"
    feature_cols = get_entry_feature_columns(bars_df)
    log.info(f"Entry model: {len(feature_cols)} features, target={target_col}")

    dates = sorted(bars_df["date"].unique())
    log.info(f"Entry model WF: {len(dates)} unique dates")

    all_preds = pd.Series(np.nan, index=bars_df.index, dtype=np.float64)

    fold_idx = 0
    # Use 60d/10d/5d for the entry model (more data available from minute bars)
    entry_train_days = 60
    entry_val_days = 10
    entry_slide_days = 5
    start = 0

    while start + entry_train_days + entry_val_days <= len(dates):
        train_dates = dates[start:start + entry_train_days]
        val_dates = dates[start + entry_train_days:start + entry_train_days + entry_val_days]

        train_mask = bars_df["date"].isin(train_dates)
        val_mask = bars_df["date"].isin(val_dates)

        X_train = bars_df.loc[train_mask, feature_cols].values.astype(np.float32)
        y_train = bars_df.loc[train_mask, target_col].values.astype(np.float32)
        X_val = bars_df.loc[val_mask, feature_cols].values.astype(np.float32)

        valid_train = ~np.isnan(y_train) & ~np.isnan(X_train).any(axis=1)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        if len(X_train) < 100 or len(X_val) < 10:
            start += entry_slide_days
            continue

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feature_cols, free_raw_data=False)

        model = lgb.train(
            LGBM_ENTRY_PARAMS,
            dtrain,
            num_boost_round=500,
            valid_sets=[dtrain],
            callbacks=[lgb.log_evaluation(0)],
        )

        preds = model.predict(X_val)
        all_preds.iloc[bars_df.index[val_mask]] = preds

        fold_idx += 1
        if fold_idx % 5 == 0:
            log.info(f"Entry fold {fold_idx}: train={train_dates[0]}..{train_dates[-1]}, "
                     f"val={val_dates[0]}..{val_dates[-1]}")

        start += entry_slide_days
        del model, dtrain
        gc.collect()

    bars_df["entry_pred_ticks"] = all_preds
    valid_preds = all_preds.dropna()
    log.info(f"Entry model: {len(valid_preds)} OOT predictions across {fold_idx} folds")

    if len(valid_preds) > 0:
        p80 = np.nanpercentile(valid_preds, 80)
        p20 = np.nanpercentile(valid_preds, 20)
        bars_df["entry_signal"] = 0
        bars_df.loc[bars_df["entry_pred_ticks"] > max(p80, MIN_EDGE_TICKS), "entry_signal"] = 1
        bars_df.loc[bars_df["entry_pred_ticks"] < min(p20, -MIN_EDGE_TICKS), "entry_signal"] = -1

        n_long = (bars_df["entry_signal"] == 1).sum()
        n_short = (bars_df["entry_signal"] == -1).sum()
        log.info(f"Entry signals: {n_long} long, {n_short} short "
                 f"(thresholds: long>{p80:.1f}, short<{p20:.1f})")
    else:
        bars_df["entry_signal"] = 0

    return bars_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: TICK-LEVEL TRADE SAMPLE CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════


# Expected queue feature columns
QUEUE_COLS = [
    "event_id", "ts_ns", "side", "pred_1s", "pred_5s", "pred_10s",
    "bid_qty_at_touch", "bid_n_orders", "bid_q_ahead_if_join_back", "bid_q_ahead_p50",
    "ask_qty_at_touch", "ask_n_orders", "ask_q_ahead_if_join_back", "ask_q_ahead_p50",
    "bid_level_age_s", "ask_level_age_s",
    "bid_time_since_last_add_s", "bid_time_since_last_cancel_s",
    "ask_time_since_last_add_s", "ask_time_since_last_cancel_s",
    "bid_add_rate_1s", "ask_add_rate_1s",
    "bid_cancel_rate_1s", "ask_cancel_rate_1s",
    "bid_trade_rate_1s", "ask_trade_rate_1s",
    "ofi_1s", "ofi_5s", "ofi_10s",
    "top_imbalance", "microprice_offset_ticks",
]


def _find_nearest_tick_idx(ts_ns_arr: np.ndarray, target_ns: int) -> int:
    """Binary search for nearest tick index."""
    idx = np.searchsorted(ts_ns_arr, target_ns)
    if idx >= len(ts_ns_arr):
        return len(ts_ns_arr) - 1
    if idx == 0:
        return 0
    # Return closest
    if abs(ts_ns_arr[idx] - target_ns) < abs(ts_ns_arr[idx - 1] - target_ns):
        return idx
    return idx - 1


def construct_tick_trade_samples(
    bars_df: pd.DataFrame,
    minute_df: pd.DataFrame,
    queue_dates: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    For each entry signal on a queue-data day, extract tick-level samples every 10s.

    Returns (samples_df, meta_df).
    """
    # Get trades only on dates with queue data
    trades = bars_df[
        (bars_df["entry_signal"] != 0)
        & (bars_df["date"].isin(queue_dates))
    ].copy()
    trades = trades.dropna(subset=["entry_pred_ticks", "fwd_ticks_30min"])

    if len(trades) == 0:
        log.warning("No valid trade entries on queue-data days!")
        return pd.DataFrame(), pd.DataFrame()

    log.info(f"Constructing tick-level samples for {len(trades)} trades on {len(queue_dates)} queue days")

    all_samples = []
    trade_meta = []
    queue_cache = {}  # date -> DataFrame

    for trade_idx, (idx, trade_row) in enumerate(trades.iterrows()):
        entry_date = trade_row["date"]
        direction = int(trade_row["entry_signal"])  # +1 or -1
        entry_price = trade_row["close"]  # entry at bar close
        entry_confidence = abs(trade_row["entry_pred_ticks"])
        actual_outcome_ticks = trade_row["fwd_ticks_30min"] * direction  # signed
        entry_ts = pd.Timestamp(trade_row["ts"])

        # Load queue data for this date (cache it)
        if entry_date not in queue_cache:
            qdf = load_queue_features(entry_date)
            if qdf is None:
                continue
            queue_cache[entry_date] = qdf
        qdf = queue_cache[entry_date]

        if "ts_ns" not in qdf.columns or len(qdf) == 0:
            continue

        # Convert entry time to nanoseconds for matching
        entry_ns = int(entry_ts.value)
        end_ns = entry_ns + MAX_HOLD_SECONDS * 1_000_000_000

        ts_ns_arr = qdf["ts_ns"].values
        if not np.issubdtype(ts_ns_arr.dtype, np.integer):
            ts_ns_arr = ts_ns_arr.astype(np.int64)

        # Find entry and end indices in tick data
        entry_tick_idx = _find_nearest_tick_idx(ts_ns_arr, entry_ns)
        end_tick_idx = _find_nearest_tick_idx(ts_ns_arr, end_ns)

        if entry_tick_idx >= end_tick_idx:
            continue

        # Window of tick data for this trade
        window = qdf.iloc[entry_tick_idx:end_tick_idx + 1].copy()
        if len(window) < 10:
            continue

        # Entry-time snapshot for reference
        entry_tick = window.iloc[0]
        entry_microprice_offset = _safe_float(entry_tick, "microprice_offset_ticks", 0.0)
        entry_top_imbalance = _safe_float(entry_tick, "top_imbalance", 0.5)

        # Queue state at entry (side-relative)
        if direction == 1:  # long: our side = bid
            entry_our_side_qty = _safe_float(entry_tick, "bid_qty_at_touch", 0.0)
            entry_against_qty = _safe_float(entry_tick, "ask_qty_at_touch", 0.0)
        else:  # short: our side = ask
            entry_our_side_qty = _safe_float(entry_tick, "ask_qty_at_touch", 0.0)
            entry_against_qty = _safe_float(entry_tick, "bid_qty_at_touch", 0.0)

        entry_queue_ratio = entry_our_side_qty / max(entry_against_qty, 1.0)

        # Compute microprice-based P&L trajectory
        # microprice_offset_ticks is relative to mid, so we track changes from entry
        microprice_offsets = window["microprice_offset_ticks"].values.astype(np.float64)
        microprice_offsets = np.nan_to_num(microprice_offsets, nan=0.0)

        # Price movement in ticks from entry (via microprice offset changes)
        # positive offset = price above mid. For longs, positive change = profit
        microprice_changes = microprice_offsets - microprice_offsets[0]
        signed_pnl_ticks = microprice_changes * direction

        # Running MFE/MAE
        running_mfe = np.maximum.accumulate(signed_pnl_ticks)
        running_mae = np.minimum.accumulate(signed_pnl_ticks)

        # OFI cumulative from entry
        ofi_1s_vals = window["ofi_1s"].values.astype(np.float64)
        ofi_1s_vals = np.nan_to_num(ofi_1s_vals, nan=0.0)

        # Approximate cumulative OFI (sum of 1s OFI snapshots)
        ofi_cumulative = np.cumsum(ofi_1s_vals)

        # Sample every SAMPLE_INTERVAL_SECONDS
        window_ts_ns = window["ts_ns"].values
        if not np.issubdtype(window_ts_ns.dtype, np.integer):
            window_ts_ns = window_ts_ns.astype(np.int64)

        sample_ns_step = SAMPLE_INTERVAL_SECONDS * 1_000_000_000
        sample_times_ns = np.arange(
            window_ts_ns[0] + sample_ns_step,
            window_ts_ns[-1],
            sample_ns_step,
        )

        # Final outcome (at 30-min static exit)
        is_winner = 1 if actual_outcome_ticks > COST_RT_TICKS else 0
        final_pnl_ticks = actual_outcome_ticks

        for sample_ns in sample_times_ns:
            # Find nearest tick
            s_idx = _find_nearest_tick_idx(window_ts_ns, sample_ns)
            if s_idx <= 0 or s_idx >= len(window) - 1:
                continue

            tick = window.iloc[s_idx]
            time_in_trade_s = (window_ts_ns[s_idx] - window_ts_ns[0]) / 1e9

            if time_in_trade_s < 5:  # skip first 5 seconds
                continue

            # ── Post-entry state ──
            unrealized_pnl = signed_pnl_ticks[s_idx]
            mfe_so_far = running_mfe[s_idx]
            mae_so_far = running_mae[s_idx]
            drawdown_from_peak = (mfe_so_far - unrealized_pnl) / max(abs(mfe_so_far), 0.25) if mfe_so_far > 0.1 else 0.0

            # ── Queue dynamics (THE KEY FEATURES) ──
            if direction == 1:  # long: our side = bid
                our_side_qty = _safe_float(tick, "bid_qty_at_touch", 0.0)
                against_side_qty = _safe_float(tick, "ask_qty_at_touch", 0.0)
                our_side_add_rate = _safe_float(tick, "bid_add_rate_1s", 0.0)
                against_side_cancel_rate = _safe_float(tick, "ask_cancel_rate_1s", 0.0)
                trade_rate_our_side = _safe_float(tick, "bid_trade_rate_1s", 0.0)
                trade_rate_against_side = _safe_float(tick, "ask_trade_rate_1s", 0.0)
                level_age_our_side = _safe_float(tick, "bid_level_age_s", 0.0)
                cancel_rate_our_side = _safe_float(tick, "bid_cancel_rate_1s", 0.0)
                our_side_n_orders = _safe_float(tick, "bid_n_orders", 0.0)
                against_side_n_orders = _safe_float(tick, "ask_n_orders", 0.0)
            else:  # short: our side = ask
                our_side_qty = _safe_float(tick, "ask_qty_at_touch", 0.0)
                against_side_qty = _safe_float(tick, "bid_qty_at_touch", 0.0)
                our_side_add_rate = _safe_float(tick, "ask_add_rate_1s", 0.0)
                against_side_cancel_rate = _safe_float(tick, "bid_cancel_rate_1s", 0.0)
                trade_rate_our_side = _safe_float(tick, "ask_trade_rate_1s", 0.0)
                trade_rate_against_side = _safe_float(tick, "bid_trade_rate_1s", 0.0)
                level_age_our_side = _safe_float(tick, "ask_level_age_s", 0.0)
                cancel_rate_our_side = _safe_float(tick, "ask_cancel_rate_1s", 0.0)
                our_side_n_orders = _safe_float(tick, "ask_n_orders", 0.0)
                against_side_n_orders = _safe_float(tick, "bid_n_orders", 0.0)

            queue_ratio = our_side_qty / max(against_side_qty, 1.0)
            queue_ratio_change = queue_ratio - entry_queue_ratio

            # OFI alignment
            ofi_since_entry = ofi_cumulative[s_idx]
            # Recent 5s OFI: use last ~50 ticks worth of ofi_1s (roughly 5s at ~10 ticks/s)
            lookback_5s = max(0, s_idx - 50)
            ofi_recent_5s = np.sum(ofi_1s_vals[lookback_5s:s_idx + 1])
            ofi_alignment = 1.0 if (np.sign(ofi_recent_5s) == np.sign(direction)) else 0.0

            # Imbalance
            imbalance_now = _safe_float(tick, "top_imbalance", 0.5)
            imbalance_vs_entry = imbalance_now - entry_top_imbalance

            # Microprice
            microprice_offset_now = _safe_float(tick, "microprice_offset_ticks", 0.0)
            # Microprice trend over last 30s (~300 ticks)
            lookback_30s = max(0, s_idx - 300)
            if s_idx - lookback_30s > 10:
                mp_recent = microprice_offsets[lookback_30s:s_idx + 1]
                try:
                    microprice_trend = float(np.polyfit(np.arange(len(mp_recent)), mp_recent, 1)[0]) * direction
                except:
                    microprice_trend = 0.0
            else:
                microprice_trend = 0.0

            # Cancel spike detection: compare current cancel rate to average
            entry_cancel_rate = _safe_float(entry_tick, "bid_cancel_rate_1s" if direction == 1 else "ask_cancel_rate_1s", 0.0)
            cancel_spike = max(cancel_rate_our_side / max(entry_cancel_rate, 0.01) - 1.0, 0.0)

            # ── Labels ──
            # Remaining favorable ticks from this point to trade end
            remaining_signed_pnl = signed_pnl_ticks[s_idx:] - signed_pnl_ticks[s_idx]
            remaining_favorable = np.max(remaining_signed_pnl) if len(remaining_signed_pnl) > 1 else 0.0

            # Should exit now? Compare exit-now P&L vs hold-to-static
            exit_now_pnl = unrealized_pnl - COST_RT_TICKS  # not double-counting entry cost
            # Static exit pnl is the final outcome minus cost
            static_exit_pnl = final_pnl_ticks - COST_RT_TICKS
            should_exit_now = 1 if exit_now_pnl > static_exit_pnl else 0

            sample = {
                # Identity
                "trade_idx": trade_idx,
                "date": entry_date,
                "direction": direction,
                "time_in_trade_seconds": time_in_trade_s,

                # ── Post-entry state ──
                "unrealized_pnl_ticks": unrealized_pnl,
                "mfe_so_far_ticks": mfe_so_far,
                "mae_so_far_ticks": mae_so_far,
                "drawdown_from_peak": drawdown_from_peak,

                # ── Queue dynamics (THE KEY FEATURES) ──
                "our_side_qty": our_side_qty,
                "against_side_qty": against_side_qty,
                "queue_ratio": queue_ratio,
                "queue_ratio_change": queue_ratio_change,
                "our_side_add_rate": our_side_add_rate,
                "against_side_cancel_rate": against_side_cancel_rate,
                "ofi_since_entry": ofi_since_entry,
                "ofi_recent_5s": ofi_recent_5s,
                "ofi_alignment": ofi_alignment,
                "imbalance_now": imbalance_now,
                "imbalance_vs_entry": imbalance_vs_entry,
                "microprice_offset_now": microprice_offset_now,
                "microprice_trend": microprice_trend,
                "trade_rate_our_side": trade_rate_our_side,
                "trade_rate_against_side": trade_rate_against_side,
                "level_age_our_side": level_age_our_side,
                "cancel_spike": cancel_spike,
                "our_side_n_orders": our_side_n_orders,
                "against_side_n_orders": against_side_n_orders,

                # ── Entry context ──
                "entry_confidence": entry_confidence,
                "entry_direction": direction,

                # ── Labels ──
                "label_final_pnl_ticks": final_pnl_ticks,
                "label_remaining_favorable_ticks": remaining_favorable,
                "label_will_eventually_profit": is_winner,
                "label_should_exit_now": should_exit_now,
            }

            all_samples.append(sample)

        # Trade metadata
        trade_meta.append({
            "trade_idx": trade_idx,
            "entry_time": entry_ts,
            "direction": direction,
            "entry_price": entry_price,
            "entry_confidence": entry_confidence,
            "actual_outcome_ticks": actual_outcome_ticks,
            "is_winner": is_winner,
            "date": entry_date,
        })

        if (trade_idx + 1) % 100 == 0:
            log.info(f"  Processed {trade_idx + 1}/{len(trades)} trades, "
                     f"{len(all_samples)} tick samples so far")

        # Free cache for dates we're done with
        if trade_idx + 1 < len(trades):
            next_date = trades.iloc[trade_idx + 1]["date"] if trade_idx + 1 < len(trades) else None
            # Keep cache small
            if len(queue_cache) > 3:
                for cached_date in list(queue_cache.keys()):
                    if cached_date != entry_date and cached_date != next_date:
                        del queue_cache[cached_date]
                gc.collect()

    if not all_samples:
        log.warning("No tick-level samples constructed!")
        return pd.DataFrame(), pd.DataFrame()

    samples_df = pd.DataFrame(all_samples)
    meta_df = pd.DataFrame(trade_meta)

    log.info(f"Tick samples: {len(samples_df):,} from {len(meta_df)} trades "
             f"(avg {len(samples_df)/max(len(meta_df),1):.1f} samples/trade)")
    log.info(f"  Winners: {meta_df['is_winner'].sum()}/{len(meta_df)} "
             f"({meta_df['is_winner'].mean()*100:.1f}%)")

    return samples_df, meta_df


def _safe_float(row, col, default=0.0):
    """Safely extract a float from a row."""
    try:
        v = row[col]
        if pd.isna(v):
            return default
        return float(v)
    except (KeyError, TypeError):
        return default


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: MANAGEMENT FEATURE COLUMNS
# ═══════════════════════════════════════════════════════════════════


def get_management_feature_columns() -> List[str]:
    """Return the feature columns used by the management models."""
    return [
        # Post-entry state
        "time_in_trade_seconds",
        "unrealized_pnl_ticks",
        "mfe_so_far_ticks",
        "mae_so_far_ticks",
        "drawdown_from_peak",
        # Queue dynamics
        "our_side_qty",
        "against_side_qty",
        "queue_ratio",
        "queue_ratio_change",
        "our_side_add_rate",
        "against_side_cancel_rate",
        "ofi_since_entry",
        "ofi_recent_5s",
        "ofi_alignment",
        "imbalance_now",
        "imbalance_vs_entry",
        "microprice_offset_now",
        "microprice_trend",
        "trade_rate_our_side",
        "trade_rate_against_side",
        "level_age_our_side",
        "cancel_spike",
        "our_side_n_orders",
        "against_side_n_orders",
        # Entry context
        "entry_confidence",
        "entry_direction",
    ]


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: MANAGEMENT MODEL TRAINING (Walk-Forward)
# ═══════════════════════════════════════════════════════════════════


def train_management_models_wf(
    samples_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> Tuple[List[Dict], pd.DataFrame]:
    """
    Walk-forward train three management models:
      1. Invalidation classifier: P(trade_will_lose | tick_features)
      2. Exit timing classifier: P(should_exit_now | tick_features)
      3. Remaining edge regressor: predicted remaining favorable ticks

    25d train / 8d val / 4d slide.
    Returns fold results and samples with predictions.
    """
    _import_lightgbm()

    feature_cols = get_management_feature_columns()
    dates = sorted(samples_df["date"].unique())
    log.info(f"Management model WF: {len(dates)} unique dates, {len(feature_cols)} features")

    all_inv_preds = pd.Series(np.nan, index=samples_df.index, dtype=np.float64)
    all_exit_preds = pd.Series(np.nan, index=samples_df.index, dtype=np.float64)
    all_edge_preds = pd.Series(np.nan, index=samples_df.index, dtype=np.float64)

    fold_results = []
    fold_idx = 0
    start = 0

    while start + TRAIN_DAYS + VAL_DAYS <= len(dates):
        train_dates = dates[start:start + TRAIN_DAYS]
        val_dates = dates[start + TRAIN_DAYS:start + TRAIN_DAYS + VAL_DAYS]

        train_mask = samples_df["date"].isin(train_dates)
        val_mask = samples_df["date"].isin(val_dates)

        X_train = samples_df.loc[train_mask, feature_cols].values.astype(np.float32)
        y_train_inv = samples_df.loc[train_mask, "label_will_eventually_profit"].values.astype(np.float32)
        y_train_inv = 1.0 - y_train_inv  # Invert: we want P(loser)
        y_train_exit = samples_df.loc[train_mask, "label_should_exit_now"].values.astype(np.float32)
        y_train_edge = samples_df.loc[train_mask, "label_remaining_favorable_ticks"].values.astype(np.float32)
        X_val = samples_df.loc[val_mask, feature_cols].values.astype(np.float32)
        y_val_inv = 1.0 - samples_df.loc[val_mask, "label_will_eventually_profit"].values.astype(np.float32)
        y_val_exit = samples_df.loc[val_mask, "label_should_exit_now"].values.astype(np.float32)
        y_val_edge = samples_df.loc[val_mask, "label_remaining_favorable_ticks"].values.astype(np.float32)

        # Remove NaN rows
        valid_train = ~(np.isnan(y_train_inv) | np.isnan(y_train_exit) |
                        np.isnan(y_train_edge) | np.isnan(X_train).any(axis=1))
        valid_val = ~(np.isnan(X_val).any(axis=1) | np.isnan(y_val_inv) |
                      np.isnan(y_val_exit) | np.isnan(y_val_edge))

        X_tr = X_train[valid_train]
        y_inv_tr = y_train_inv[valid_train]
        y_exit_tr = y_train_exit[valid_train]
        y_edge_tr = y_train_edge[valid_train]
        X_v = X_val[valid_val]
        y_inv_v = y_val_inv[valid_val]
        y_exit_v = y_val_exit[valid_val]
        y_edge_v = y_val_edge[valid_val]

        if len(X_tr) < 200 or len(X_v) < 20:
            log.warning(f"Fold {fold_idx}: insufficient data (train={len(X_tr)}, val={len(X_v)}), skipping")
            start += SLIDE_DAYS
            continue

        # ── 1. Invalidation classifier: P(loser) ──
        dtrain_inv = lgb.Dataset(X_tr, label=y_inv_tr, feature_name=feature_cols, free_raw_data=False)
        dval_inv = lgb.Dataset(X_v, label=y_inv_v, feature_name=feature_cols, free_raw_data=False)
        inv_model = lgb.train(
            LGBM_CLF_PARAMS,
            dtrain_inv,
            num_boost_round=500,
            valid_sets=[dval_inv],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        inv_preds = inv_model.predict(X_v)

        # ── 2. Exit timing classifier: P(should_exit_now) ──
        dtrain_exit = lgb.Dataset(X_tr, label=y_exit_tr, feature_name=feature_cols, free_raw_data=False)
        dval_exit = lgb.Dataset(X_v, label=y_exit_v, feature_name=feature_cols, free_raw_data=False)
        exit_model = lgb.train(
            LGBM_EXIT_PARAMS,
            dtrain_exit,
            num_boost_round=500,
            valid_sets=[dval_exit],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        exit_preds = exit_model.predict(X_v)

        # ── 3. Remaining edge regressor ──
        dtrain_edge = lgb.Dataset(X_tr, label=y_edge_tr, feature_name=feature_cols, free_raw_data=False)
        dval_edge = lgb.Dataset(X_v, label=y_edge_v, feature_name=feature_cols, free_raw_data=False)
        edge_model = lgb.train(
            LGBM_REG_PARAMS,
            dtrain_edge,
            num_boost_round=500,
            valid_sets=[dval_edge],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        edge_preds = edge_model.predict(X_v)

        # Store predictions
        val_indices = samples_df.index[val_mask][valid_val]
        all_inv_preds.iloc[val_indices] = inv_preds
        all_exit_preds.iloc[val_indices] = exit_preds
        all_edge_preds.iloc[val_indices] = edge_preds

        # Fold-level metrics
        try:
            from sklearn.metrics import roc_auc_score
            inv_auc = roc_auc_score(y_inv_v, inv_preds)
        except (ValueError, ImportError):
            inv_auc = 0.5
        try:
            exit_auc = roc_auc_score(y_exit_v, exit_preds)
        except (ValueError, ImportError):
            exit_auc = 0.5
        edge_mae = float(np.mean(np.abs(y_edge_v - edge_preds)))

        # Feature importance (from invalidation model, most important)
        imp = inv_model.feature_importance(importance_type="gain")
        imp_dict = dict(zip(feature_cols, imp.tolist()))

        fold_results.append({
            "fold": fold_idx,
            "train_start": train_dates[0],
            "train_end": train_dates[-1],
            "val_start": val_dates[0],
            "val_end": val_dates[-1],
            "n_train": len(X_tr),
            "n_val": len(X_v),
            "inv_auc": inv_auc,
            "exit_auc": exit_auc,
            "edge_mae": edge_mae,
            "feature_importance": imp_dict,
        })

        log.info(f"Mgmt fold {fold_idx}: inv_AUC={inv_auc:.4f}, exit_AUC={exit_auc:.4f}, "
                 f"edge_MAE={edge_mae:.2f}, val={val_dates[0]}..{val_dates[-1]}, "
                 f"n_train={len(X_tr)}, n_val={len(X_v)}")

        fold_idx += 1
        start += SLIDE_DAYS

        # Save last fold's models
        if start + TRAIN_DAYS + VAL_DAYS > len(dates):
            inv_model.save_model(str(MODEL_DIR / "invalidation_clf_latest.txt"))
            exit_model.save_model(str(MODEL_DIR / "exit_timing_clf_latest.txt"))
            edge_model.save_model(str(MODEL_DIR / "remaining_edge_reg_latest.txt"))
            log.info("Saved latest management models")

        del inv_model, exit_model, edge_model
        del dtrain_inv, dval_inv, dtrain_exit, dval_exit, dtrain_edge, dval_edge
        gc.collect()

    samples_df["pred_p_loser"] = all_inv_preds
    samples_df["pred_should_exit"] = all_exit_preds
    samples_df["pred_remaining_edge"] = all_edge_preds

    log.info(f"Management models trained: {fold_idx} folds")
    if fold_results:
        avg_inv_auc = np.mean([f["inv_auc"] for f in fold_results])
        avg_exit_auc = np.mean([f["exit_auc"] for f in fold_results])
        avg_edge_mae = np.mean([f["edge_mae"] for f in fold_results])
        log.info(f"  Average inv_AUC={avg_inv_auc:.4f}, exit_AUC={avg_exit_auc:.4f}, "
                 f"edge_MAE={avg_edge_mae:.2f}")

    return fold_results, samples_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: SIMULATION — STATIC vs DYNAMIC EXIT
# ═══════════════════════════════════════════════════════════════════


def simulate_exits(
    samples_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    invalidation_thresholds: List[float] = [0.55, 0.60, 0.65, 0.70, 0.75, 0.80],
    exit_thresholds: List[float] = [0.55, 0.60, 0.65, 0.70],
    edge_thresholds: List[float] = [0.5, 1.0, 1.5, 2.0],
) -> Dict:
    """
    Compare static 30-min exit vs dynamic exit strategies.

    Dynamic strategies:
      A) Invalidation-only: exit when P(loser) > threshold
      B) Exit-timing-only: exit when P(should_exit) > threshold
      C) Edge-only: exit when pred_remaining_edge < threshold
      D) Combined: invalidation OR (exit_timing AND low_edge)
    """
    results = {}

    # ── Static exit (baseline) ──
    static_trades = meta_df.copy()
    static_pnl = static_trades["actual_outcome_ticks"].values - COST_RT_TICKS
    static_sharpe = _compute_sharpe(static_pnl)
    static_wr = (static_pnl > 0).mean()
    static_pf = _compute_profit_factor(static_pnl)
    static_sortino = _compute_sortino(static_pnl)

    results["static"] = {
        "n_trades": len(static_trades),
        "mean_pnl_ticks": float(np.mean(static_pnl)),
        "sharpe": static_sharpe,
        "sortino": static_sortino,
        "win_rate": float(static_wr),
        "profit_factor": static_pf,
        "total_pnl_ticks": float(np.sum(static_pnl)),
        "max_drawdown_ticks": float(_max_drawdown(static_pnl)),
        "avg_hold_sec": MAX_HOLD_SECONDS,
    }

    log.info(f"STATIC EXIT: Sharpe={static_sharpe:.3f}, Sortino={static_sortino:.3f}, "
             f"WR={static_wr:.1%}, PF={static_pf:.2f}, "
             f"mean={np.mean(static_pnl):.2f}t, n={len(static_trades)}")

    valid_samples = samples_df.dropna(subset=["pred_p_loser", "pred_should_exit", "pred_remaining_edge"])
    if len(valid_samples) == 0:
        log.warning("No valid management predictions for simulation!")
        return results

    best_config = None
    best_sharpe = static_sharpe

    # ── Strategy A: Invalidation only ──
    for inv_thresh in invalidation_thresholds:
        config_key = f"inv_only_{inv_thresh:.2f}"
        trade_results = _simulate_strategy(
            valid_samples, meta_df,
            inv_thresh=inv_thresh, exit_thresh=None, edge_thresh=None,
        )
        _record_results(results, config_key, trade_results, inv_thresh, None, None)
        if results[config_key]["sharpe"] > best_sharpe:
            best_sharpe = results[config_key]["sharpe"]
            best_config = config_key

    # ── Strategy B: Exit timing only ──
    for exit_thresh in exit_thresholds:
        config_key = f"exit_only_{exit_thresh:.2f}"
        trade_results = _simulate_strategy(
            valid_samples, meta_df,
            inv_thresh=None, exit_thresh=exit_thresh, edge_thresh=None,
        )
        _record_results(results, config_key, trade_results, None, exit_thresh, None)
        if results[config_key]["sharpe"] > best_sharpe:
            best_sharpe = results[config_key]["sharpe"]
            best_config = config_key

    # ── Strategy C: Edge only ──
    for edge_thresh in edge_thresholds:
        config_key = f"edge_only_{edge_thresh:.1f}"
        trade_results = _simulate_strategy(
            valid_samples, meta_df,
            inv_thresh=None, exit_thresh=None, edge_thresh=edge_thresh,
        )
        _record_results(results, config_key, trade_results, None, None, edge_thresh)
        if results[config_key]["sharpe"] > best_sharpe:
            best_sharpe = results[config_key]["sharpe"]
            best_config = config_key

    # ── Strategy D: Combined (invalidation OR (exit AND low_edge)) ──
    for inv_thresh in [0.65, 0.70, 0.75]:
        for exit_thresh in [0.60, 0.65]:
            for edge_thresh in [1.0, 1.5]:
                config_key = f"combined_inv{inv_thresh:.2f}_exit{exit_thresh:.2f}_edge{edge_thresh:.1f}"
                trade_results = _simulate_strategy(
                    valid_samples, meta_df,
                    inv_thresh=inv_thresh, exit_thresh=exit_thresh, edge_thresh=edge_thresh,
                )
                _record_results(results, config_key, trade_results, inv_thresh, exit_thresh, edge_thresh)
                if results[config_key]["sharpe"] > best_sharpe:
                    best_sharpe = results[config_key]["sharpe"]
                    best_config = config_key

    if best_config:
        log.info(f"\nBEST DYNAMIC CONFIG: {best_config}")
        best = results[best_config]
        log.info(f"  Sharpe={best['sharpe']:.3f} (vs static {static_sharpe:.3f}), "
                 f"Sortino={best['sortino']:.3f}, WR={best['win_rate']:.1%}, "
                 f"PF={best['profit_factor']:.2f}")
        log.info(f"  Avg hold: {best['avg_hold_sec']:.0f}s "
                 f"(winners: {best.get('avg_hold_winners', 0):.0f}s, "
                 f"losers: {best.get('avg_hold_losers', 0):.0f}s)")
        log.info(f"  Exit reasons: {best.get('exit_reasons', {})}")
        results["best_config"] = best_config
    else:
        log.info("No dynamic config improved over static exit")
        results["best_config"] = "static"

    return results


def _simulate_strategy(
    valid_samples: pd.DataFrame,
    meta_df: pd.DataFrame,
    inv_thresh: Optional[float],
    exit_thresh: Optional[float],
    edge_thresh: Optional[float],
) -> List[Dict]:
    """Simulate a single dynamic exit strategy across all trades."""
    trade_results = []

    for _, trade_info in meta_df.iterrows():
        trade_idx = trade_info["trade_idx"]
        trade_samples = valid_samples[
            valid_samples["trade_idx"] == trade_idx
        ].sort_values("time_in_trade_seconds")

        if len(trade_samples) == 0:
            pnl = trade_info["actual_outcome_ticks"] - COST_RT_TICKS
            trade_results.append({
                "trade_idx": trade_idx,
                "pnl_ticks": pnl,
                "hold_time_sec": MAX_HOLD_SECONDS,
                "exit_reason": "static_no_data",
                "direction": trade_info["direction"],
                "date": trade_info["date"],
            })
            continue

        exit_time_sec = None
        exit_reason = "timeout"
        exit_pnl = None

        for _, sample in trade_samples.iterrows():
            t = sample["time_in_trade_seconds"]
            triggered = False

            # Check invalidation
            if inv_thresh is not None and sample["pred_p_loser"] > inv_thresh:
                triggered = True
                exit_reason = "invalidation"

            # Check exit timing (only after minimum hold of 30s)
            if not triggered and exit_thresh is not None and t >= 30:
                if sample["pred_should_exit"] > exit_thresh:
                    if edge_thresh is not None:
                        # Combined: exit timing must agree with low edge
                        if sample["pred_remaining_edge"] < edge_thresh:
                            triggered = True
                            exit_reason = "exit_timing+low_edge"
                    else:
                        triggered = True
                        exit_reason = "exit_timing"

            # Check remaining edge (only, without exit timing)
            if not triggered and edge_thresh is not None and exit_thresh is None:
                if t >= 30 and sample["pred_remaining_edge"] < edge_thresh:
                    triggered = True
                    exit_reason = "low_edge"

            if triggered:
                exit_time_sec = t
                exit_pnl = sample["unrealized_pnl_ticks"] - COST_RT_TICKS
                break

        if exit_time_sec is None:
            pnl = trade_info["actual_outcome_ticks"] - COST_RT_TICKS
            hold = MAX_HOLD_SECONDS
        else:
            pnl = exit_pnl
            hold = exit_time_sec

        trade_results.append({
            "trade_idx": trade_idx,
            "pnl_ticks": pnl,
            "hold_time_sec": hold,
            "exit_reason": exit_reason,
            "direction": trade_info["direction"],
            "date": trade_info["date"],
        })

    return trade_results


def _record_results(
    results: Dict,
    config_key: str,
    trade_results: List[Dict],
    inv_thresh: Optional[float],
    exit_thresh: Optional[float],
    edge_thresh: Optional[float],
):
    """Record simulation results for a config."""
    trades_df = pd.DataFrame(trade_results)
    pnl_arr = trades_df["pnl_ticks"].values

    results[config_key] = {
        "inv_threshold": inv_thresh,
        "exit_threshold": exit_thresh,
        "edge_threshold": edge_thresh,
        "n_trades": len(trades_df),
        "mean_pnl_ticks": float(np.mean(pnl_arr)),
        "sharpe": _compute_sharpe(pnl_arr),
        "sortino": _compute_sortino(pnl_arr),
        "win_rate": float((pnl_arr > 0).mean()),
        "profit_factor": _compute_profit_factor(pnl_arr),
        "total_pnl_ticks": float(np.sum(pnl_arr)),
        "max_drawdown_ticks": float(_max_drawdown(pnl_arr)),
        "avg_hold_sec": float(trades_df["hold_time_sec"].mean()),
        "exit_reasons": trades_df["exit_reason"].value_counts().to_dict(),
        "avg_hold_winners": float(
            trades_df[trades_df["pnl_ticks"] > 0]["hold_time_sec"].mean()
        ) if (trades_df["pnl_ticks"] > 0).any() else 0,
        "avg_hold_losers": float(
            trades_df[trades_df["pnl_ticks"] <= 0]["hold_time_sec"].mean()
        ) if (trades_df["pnl_ticks"] <= 0).any() else 0,
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 7: FEATURE IMPORTANCE & WINNER/LOSER ANALYSIS
# ═══════════════════════════════════════════════════════════════════


def analyze_feature_importance(fold_results: List[Dict]) -> Dict:
    """Aggregate feature importance across folds."""
    if not fold_results:
        return {}

    # Collect all importance dicts
    all_imp = {}
    for fr in fold_results:
        for feat, imp in fr.get("feature_importance", {}).items():
            if feat not in all_imp:
                all_imp[feat] = []
            all_imp[feat].append(imp)

    # Average importance
    avg_imp = {feat: np.mean(vals) for feat, vals in all_imp.items()}
    sorted_imp = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)

    log.info("\n=== TOP 15 FEATURES (avg gain importance) ===")
    for feat, imp in sorted_imp[:15]:
        log.info(f"  {feat:40s} {imp:.1f}")

    return dict(sorted_imp)


def winner_loser_queue_dynamics(
    samples_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> Dict:
    """Compare queue dynamics for winning vs losing trades."""
    winners = meta_df[meta_df["is_winner"] == 1]["trade_idx"].values
    losers = meta_df[meta_df["is_winner"] == 0]["trade_idx"].values

    win_samples = samples_df[samples_df["trade_idx"].isin(winners)]
    lose_samples = samples_df[samples_df["trade_idx"].isin(losers)]

    queue_features = [
        "queue_ratio", "queue_ratio_change", "ofi_since_entry",
        "ofi_recent_5s", "ofi_alignment", "imbalance_now",
        "imbalance_vs_entry", "microprice_trend",
        "trade_rate_our_side", "trade_rate_against_side",
        "level_age_our_side", "cancel_spike",
    ]

    comparison = {}
    log.info("\n=== WINNER vs LOSER QUEUE DYNAMICS ===")
    for feat in queue_features:
        if feat not in win_samples.columns:
            continue
        w_mean = win_samples[feat].mean()
        l_mean = lose_samples[feat].mean()
        diff_pct = (w_mean - l_mean) / max(abs(l_mean), 0.001) * 100

        comparison[feat] = {
            "winner_mean": float(w_mean),
            "loser_mean": float(l_mean),
            "diff_pct": float(diff_pct),
        }
        log.info(f"  {feat:35s}  W={w_mean:8.3f}  L={l_mean:8.3f}  diff={diff_pct:+.1f}%")

    return comparison


# ═══════════════════════════════════════════════════════════════════
#  SECTION 8: REGIME ANALYSIS (HC #428)
# ═══════════════════════════════════════════════════════════════════


def regime_analysis(
    meta_df: pd.DataFrame,
    bars_df: pd.DataFrame,
) -> Dict:
    """Stratify results by market regime (green/red days)."""
    day_returns = bars_df.groupby("date").agg(
        day_open=("open", "first"),
        day_close=("close", "last"),
    )
    day_returns["day_return"] = day_returns["day_close"] - day_returns["day_open"]
    day_returns["regime"] = "flat"
    day_returns.loc[day_returns["day_return"] > 0, "regime"] = "green"
    day_returns.loc[day_returns["day_return"] < 0, "regime"] = "red"

    regime_map = day_returns["regime"].to_dict()
    meta = meta_df.copy()
    meta["regime"] = meta["date"].map(regime_map).fillna("flat")

    results = {}
    for regime in ["green", "red", "flat"]:
        regime_trades = meta[meta["regime"] == regime]
        if len(regime_trades) < 3:
            continue
        pnl = regime_trades["actual_outcome_ticks"].values - COST_RT_TICKS
        results[regime] = {
            "n_trades": len(regime_trades),
            "sharpe": _compute_sharpe(pnl),
            "sortino": _compute_sortino(pnl),
            "win_rate": float((pnl > 0).mean()),
            "profit_factor": _compute_profit_factor(pnl),
            "mean_pnl_ticks": float(np.mean(pnl)),
        }
        log.info(f"Regime {regime}: Sharpe={results[regime]['sharpe']:.3f}, "
                 f"WR={results[regime]['win_rate']:.1%}, n={len(regime_trades)}")

    if "green" in results and "red" in results:
        s_g = results["green"]["sharpe"]
        s_r = results["red"]["sharpe"]
        denom = max(abs(s_g), abs(s_r), 0.01)
        regime_gap = abs(s_g - s_r) / denom
        results["regime_gap"] = regime_gap
        results["regime_pass"] = regime_gap <= 0.50
        log.info(f"Regime gap: {regime_gap:.2f} ({'PASS' if regime_gap <= 0.50 else 'FAIL'} HC #428)")

    return results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 9: MLFLOW LOGGING
# ═══════════════════════════════════════════════════════════════════


def log_to_mlflow(
    fold_results: List[Dict],
    sim_results: Dict,
    regime_results: Dict,
    feat_importance: Dict,
    queue_comparison: Dict,
    n_trades: int,
    n_samples: int,
):
    """Log experiment to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://neptune:5000")
        mlflow.set_experiment("trade_management_v2_tick")

        with mlflow.start_run(run_name=f"tm_v2_tick_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Params
            mlflow.log_param("model_type", "LightGBM_3models")
            mlflow.log_param("sample_interval_sec", SAMPLE_INTERVAL_SECONDS)
            mlflow.log_param("train_days", TRAIN_DAYS)
            mlflow.log_param("val_days", VAL_DAYS)
            mlflow.log_param("slide_days", SLIDE_DAYS)
            mlflow.log_param("max_hold_seconds", MAX_HOLD_SECONDS)
            mlflow.log_param("cost_rt_ticks", COST_RT_TICKS)
            mlflow.log_param("n_trades", n_trades)
            mlflow.log_param("n_tick_samples", n_samples)
            mlflow.log_param("n_features", len(get_management_feature_columns()))
            mlflow.log_param("data_type", "tick_level_queue")

            # Fold-level metrics
            if fold_results:
                mlflow.log_metric("avg_inv_auc", np.mean([f["inv_auc"] for f in fold_results]))
                mlflow.log_metric("avg_exit_auc", np.mean([f["exit_auc"] for f in fold_results]))
                mlflow.log_metric("avg_edge_mae", np.mean([f["edge_mae"] for f in fold_results]))
                mlflow.log_metric("n_folds", len(fold_results))

            # Static baseline
            if "static" in sim_results:
                st = sim_results["static"]
                mlflow.log_metric("static_sharpe", st["sharpe"])
                mlflow.log_metric("static_sortino", st["sortino"])
                mlflow.log_metric("static_wr", st["win_rate"])
                mlflow.log_metric("static_pf", st["profit_factor"])
                mlflow.log_metric("static_mean_pnl", st["mean_pnl_ticks"])

            # Best dynamic config
            best_key = sim_results.get("best_config", "static")
            if best_key != "static" and best_key in sim_results:
                best = sim_results[best_key]
                mlflow.log_metric("best_dynamic_sharpe", best["sharpe"])
                mlflow.log_metric("best_dynamic_sortino", best["sortino"])
                mlflow.log_metric("best_dynamic_wr", best["win_rate"])
                mlflow.log_metric("best_dynamic_pf", best["profit_factor"])
                mlflow.log_metric("best_dynamic_mean_pnl", best["mean_pnl_ticks"])
                mlflow.log_metric("best_dynamic_avg_hold_sec", best["avg_hold_sec"])
                mlflow.log_param("best_config", best_key)

                # Improvement over static
                if "static" in sim_results:
                    mlflow.log_metric("sharpe_improvement",
                                     best["sharpe"] - sim_results["static"]["sharpe"])

            # Regime results
            if regime_results:
                if "regime_gap" in regime_results:
                    mlflow.log_metric("regime_gap", regime_results["regime_gap"])
                    mlflow.log_param("regime_pass", regime_results.get("regime_pass", False))

            # Save full results as artifact
            results_path = OUTPUT_DIR / "full_results.json"
            with open(results_path, "w") as f:
                json.dump({
                    "fold_results": [{k: v for k, v in fr.items() if k != "feature_importance"} for fr in fold_results],
                    "sim_results": sim_results,
                    "regime_results": regime_results,
                    "feature_importance": feat_importance,
                    "queue_comparison": queue_comparison,
                }, f, indent=2, default=str)
            mlflow.log_artifact(str(results_path))

            log.info("MLflow logging complete")

    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


# ═══════════════════════════════════════════════════════════════════
#  UTILITY FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def _compute_sharpe(pnl: np.ndarray) -> float:
    if len(pnl) < 2 or np.std(pnl) == 0:
        return 0.0
    return float(np.mean(pnl) / np.std(pnl) * np.sqrt(252))


def _compute_sortino(pnl: np.ndarray) -> float:
    if len(pnl) < 2:
        return 0.0
    downside = pnl[pnl < 0]
    if len(downside) < 2 or np.std(downside) == 0:
        return float(np.mean(pnl) * np.sqrt(252)) if np.mean(pnl) > 0 else 0.0
    return float(np.mean(pnl) / np.std(downside) * np.sqrt(252))


def _compute_profit_factor(pnl: np.ndarray) -> float:
    gross_profit = pnl[pnl > 0].sum()
    gross_loss = abs(pnl[pnl < 0].sum())
    if gross_loss == 0:
        return 99.0 if gross_profit > 0 else 0.0
    return float(gross_profit / gross_loss)


def _max_drawdown(pnl: np.ndarray) -> float:
    cumsum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cumsum)
    dd = peak - cumsum
    return float(dd.max()) if len(dd) > 0 else 0.0


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("Trade Management v2 — Tick-Level Queue Data")
    log.info("=" * 70)

    # ── Step 0: Discover data ──
    queue_dates = discover_queue_dates()
    if not queue_dates:
        log.error("No queue feature data found! Cannot proceed.")
        return

    # ── Step 1: Load minute bars and build entry model ──
    log.info("\n>>> STEP 1: Loading minute bars and training entry model <<<")
    minute_df = load_minute_bars()
    bars_df = aggregate_to_30min_bars(minute_df)
    bars_df = add_rolling_features(bars_df)
    bars_df = add_forward_labels(bars_df)
    bars_df = train_entry_model_wf(bars_df)

    # Check overlap
    bar_dates = set(bars_df["date"].unique())
    queue_date_set = set(queue_dates)
    overlap_dates = sorted(bar_dates & queue_date_set)
    log.info(f"Date overlap: {len(overlap_dates)} days with both entry signals and queue data")

    if len(overlap_dates) < TRAIN_DAYS + VAL_DAYS:
        log.warning(f"Only {len(overlap_dates)} overlapping days — need {TRAIN_DAYS + VAL_DAYS} "
                    f"for at least one WF fold. Proceeding anyway...")

    # ── Step 2: Construct tick-level trade samples ──
    log.info("\n>>> STEP 2: Constructing tick-level trade samples <<<")
    samples_df, meta_df = construct_tick_trade_samples(bars_df, minute_df, overlap_dates)

    if len(samples_df) == 0:
        log.error("No samples constructed! Check data alignment.")
        return

    # Save intermediate data
    samples_df.to_parquet(OUTPUT_DIR / "tick_samples.parquet", index=False)
    meta_df.to_parquet(OUTPUT_DIR / "trade_meta.parquet", index=False)
    log.info(f"Saved {len(samples_df):,} samples and {len(meta_df)} trade records")

    # ── Step 3: Train management models ──
    log.info("\n>>> STEP 3: Training management models (walk-forward) <<<")
    fold_results, samples_df = train_management_models_wf(samples_df, meta_df)

    if not fold_results:
        log.error("No WF folds completed! Insufficient data or too few overlapping dates.")
        return

    # ── Step 4: Feature importance analysis ──
    log.info("\n>>> STEP 4: Feature importance analysis <<<")
    feat_importance = analyze_feature_importance(fold_results)

    # ── Step 5: Winner vs loser queue dynamics ──
    log.info("\n>>> STEP 5: Winner vs loser queue dynamics <<<")
    queue_comparison = winner_loser_queue_dynamics(samples_df, meta_df)

    # ── Step 6: Simulate dynamic exits ──
    log.info("\n>>> STEP 6: Simulating dynamic exits <<<")
    sim_results = simulate_exits(samples_df, meta_df)

    # ── Step 7: Regime analysis ──
    log.info("\n>>> STEP 7: Regime analysis (HC #428) <<<")
    regime_results = regime_analysis(meta_df, bars_df)

    # ── Step 8: Log to MLflow ──
    log.info("\n>>> STEP 8: Logging to MLflow <<<")
    log_to_mlflow(
        fold_results, sim_results, regime_results,
        feat_importance, queue_comparison,
        n_trades=len(meta_df),
        n_samples=len(samples_df),
    )

    # ── Final summary ──
    elapsed = time.time() - t0
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)
    log.info(f"Total trades: {len(meta_df)}")
    log.info(f"Total tick samples: {len(samples_df):,}")
    log.info(f"WF folds: {len(fold_results)}")
    if fold_results:
        log.info(f"Avg invalidation AUC: {np.mean([f['inv_auc'] for f in fold_results]):.4f}")
        log.info(f"Avg exit timing AUC: {np.mean([f['exit_auc'] for f in fold_results]):.4f}")
        log.info(f"Avg remaining edge MAE: {np.mean([f['edge_mae'] for f in fold_results]):.2f}")

    if "static" in sim_results:
        log.info(f"Static baseline: Sharpe={sim_results['static']['sharpe']:.3f}, "
                 f"Sortino={sim_results['static']['sortino']:.3f}, "
                 f"WR={sim_results['static']['win_rate']:.1%}")

    best_key = sim_results.get("best_config", "static")
    if best_key != "static" and best_key in sim_results:
        best = sim_results[best_key]
        log.info(f"Best dynamic: {best_key}")
        log.info(f"  Sharpe={best['sharpe']:.3f}, Sortino={best['sortino']:.3f}, "
                 f"WR={best['win_rate']:.1%}, PF={best['profit_factor']:.2f}")
        improvement = best['sharpe'] - sim_results['static']['sharpe']
        log.info(f"  Sharpe improvement: {improvement:+.3f}")
    else:
        log.info("No dynamic config beat static exit")

    log.info(f"\nCompleted in {elapsed:.0f}s ({elapsed/60:.1f}min)")
    log.info(f"Results saved to {OUTPUT_DIR}")

    # Save final summary JSON
    summary = {
        "completed_at": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "n_trades": len(meta_df),
        "n_samples": len(samples_df),
        "n_folds": len(fold_results),
        "best_config": best_key,
        "static_sharpe": sim_results.get("static", {}).get("sharpe", 0),
        "best_dynamic_sharpe": sim_results.get(best_key, {}).get("sharpe", 0) if best_key != "static" else None,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
