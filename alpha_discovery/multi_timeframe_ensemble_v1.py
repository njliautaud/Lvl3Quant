#!/usr/bin/env python3
"""
Multi-Timeframe Ensemble v1
============================

Combines two validated, uncorrelated strategies on different timeframes:
  1. Short-horizon (30-min LightGBM): Sharpe 5.87, WR 65%, SL10/TP20 ticks
  2. Long-horizon (4h+ flow): Sharpe 1.84, WR 49.3%, PF 1.36

Three combination approaches:
  A) Portfolio allocation — independent execution, combined equity curve
  B) Confluence gating — 30-min trades only when long-horizon agrees on direction
  C) Position sizing — scale 30-min size by long-horizon confidence

Walk-forward validated, SLIDING windows, FIFO costs, regime gate (gap <= 0.50).
Logs to MLflow experiment "multi_timeframe_ensemble_v1".

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python -u \\
      alpha_discovery/multi_timeframe_ensemble_v1.py

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
LONG_HORIZON_DIR = ROOT / "output" / "long_horizon_trading_v1"
OUTPUT_DIR = ROOT / "output" / "multi_timeframe_ensemble_v1"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [MTF-ENSEMBLE] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "multi_timeframe_ensemble_v1.log")),
    ],
)
log = logging.getLogger("MTF-ENSEMBLE")

# ─────────────────────────────────────────────
#  COST CONSTANTS (CANONICAL — HC #74)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376  # $4.70 / $12.50
SPREAD_CROSSING_TICKS = 1.0    # 1 tick spread in ES RTH

# Short-horizon: aggressive entry + aggressive exit (fast fills needed)
SHORT_COST_RT_TICKS = 2.376    # 2 * spread_crossing + commission
# Long-horizon: passive entry + passive exit (can wait, multi-hour hold)
LONG_COST_RT_TICKS = 0.376     # commission only, passive fills on 4h+ holds
# Blended for confluence: passive entry, aggressive exit
CONFLUENCE_COST_RT_TICKS = 1.376  # 1 spread_crossing + commission

MIN_EDGE_TICKS = 2.5  # minimum move to define direction label

# ─────────────────────────────────────────────
#  CHAMPION 30-MIN CONFIG (from hpsweep config_id 28)
# ─────────────────────────────────────────────
CHAMPION_30MIN_CONFIG = {
    "feature_group": "momentum_vol",
    "num_leaves": 31,
    "max_depth": -1,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.9,
    "min_data_in_leaf": 100,
    "learning_rate": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 0.1,
}

# Walk-forward params
WF_TRAIN_DAYS = 60
WF_VAL_DAYS = 5   # 5-day OOT per fold for granular walk-forward
WF_SLIDE = 5


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING
# ═══════════════════════════════════════════════════════════════════


def load_all_minute_bars(min_date: str = "20250714") -> pd.DataFrame:
    """Load all minute bar parquet files."""
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


def load_long_horizon_data() -> Dict:
    """Load all long-horizon strategy outputs."""
    data = {}

    # Daily features (has close prices, OFI, regime info)
    data["daily_features"] = pd.read_parquet(LONG_HORIZON_DIR / "daily_features.parquet")

    # Walk-forward predictions
    data["pred_1d"] = pd.read_parquet(LONG_HORIZON_DIR / "predictions_1d_regression.parquet")
    data["pred_3d"] = pd.read_parquet(LONG_HORIZON_DIR / "predictions_3d_direction.parquet")

    # Trade logs
    data["intraday_trades"] = pd.read_parquet(LONG_HORIZON_DIR / "best_intraday_trades.parquet")
    data["multiday_trades"] = pd.read_parquet(LONG_HORIZON_DIR / "best_multiday_trades.parquet")

    log.info(
        f"Long-horizon data: {len(data['pred_1d'])} 1d preds, "
        f"{len(data['pred_3d'])} 3d preds, "
        f"{len(data['multiday_trades'])} multiday trades"
    )
    return data


# ═══════════════════════════════════════════════════════════════════
#  30-MIN BAR FEATURES (from hpsweep.py)
# ═══════════════════════════════════════════════════════════════════


def _safe_polyfit_slope(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def aggregate_to_bars(minute_df: pd.DataFrame, bar_size_min: int = 30) -> pd.DataFrame:
    """Aggregate minute bars to 30-min bars with features."""
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
            "signed_volume_sum": sv_arr.sum(),
            "signed_volume_ratio": sv_arr.sum() / max(vol_arr.sum(), 1),
            "buy_volume_frac": float(np.sum(sv_arr[sv_arr > 0])) / max(vol_arr.sum(), 1),
            "sell_volume_frac": float(-np.sum(sv_arr[sv_arr < 0])) / max(vol_arr.sum(), 1),
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
    log.info(f"Aggregated {len(result):,} {bar_size_min}min bars")
    return result


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add rolling lookback features to bar dataframe."""
    df = df.sort_values("ts").reset_index(drop=True)

    for w in [4, 8, 16, 32]:
        roll_mean = df["ofi_sum"].rolling(w, min_periods=1).mean()
        roll_std = df["ofi_sum"].rolling(w, min_periods=2).std().fillna(1).replace(0, 1)
        df[f"ofi_zscore_{w}bar"] = (df["ofi_sum"] - roll_mean) / roll_std
        vol_ma = df["total_volume"].rolling(w, min_periods=1).mean()
        df[f"vol_rel_{w}bar"] = df["total_volume"] / vol_ma.clip(lower=1)

    for bars, label in [(4, "lb_4bar"), (8, "lb_8bar"), (16, "lb_16bar")]:
        df[f"ret_{label}"] = df["close"].pct_change(bars)

    for w in [4, 8, 16]:
        df[f"rvol_{w}bar"] = df["return_bar"].rolling(w, min_periods=2).std()

    df["intraday_cum_ofi"] = df.groupby("date")["ofi_sum"].cumsum()
    df["intraday_cum_sv"] = df.groupby("date")["signed_volume_sum"].cumsum()

    day_stats = (
        df.groupby("date")
        .agg(
            day_ofi=("ofi_sum", "sum"),
            day_sv=("signed_volume_sum", "sum"),
            day_ret=("return_bar", "sum"),
        )
        .reset_index()
    )
    day_stats["prev_day_ret"] = day_stats["day_ret"].shift(1)
    df = df.merge(day_stats[["date", "prev_day_ret"]], on="date", how="left")

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

    log.info(f"Added rolling features: {len(df.columns)} total columns")
    return df


def add_forward_labels(df: pd.DataFrame, horizon_bars: int = 1) -> pd.DataFrame:
    """Add forward return labels."""
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
    return df


def get_momentum_vol_features(df: pd.DataFrame) -> List[str]:
    """Get the momentum_vol feature subset (matching hpsweep champion)."""
    momentum_vol_keywords = [
        "return", "ret_", "rvol", "realized_vol", "vol_of_vol", "vol_rel",
        "vol_asymmetry", "regime", "range_ticks", "close_position",
        "vwap", "volume_trend", "volume_concentration", "total_volume",
        "avg_volume", "tod_", "bars_since", "prev_day",
        "intraday_direction", "intraday_cum",
    ]

    exclude_prefixes = ("fwd_", "direction", "date", "ts", "bar_key")
    raw_price_cols = {"open", "high", "low", "close"}

    all_numeric = []
    for c in df.columns:
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if c in raw_price_cols:
            continue
        if df[c].dtype in (np.float64, np.float32, np.int64, np.int32):
            all_numeric.append(c)

    features = []
    for f in all_numeric:
        f_lower = f.lower()
        if any(kw in f_lower for kw in momentum_vol_keywords):
            features.append(f)

    return features


# ═══════════════════════════════════════════════════════════════════
#  30-MIN WALK-FORWARD (REBUILD CHAMPION PREDICTIONS)
# ═══════════════════════════════════════════════════════════════════


def rebuild_30min_predictions(bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Run walk-forward LightGBM with champion config.
    Returns DataFrame with columns: [date, bar_idx, ts, pred, actual, close]
    """
    import lightgbm as lgb

    feature_cols = get_momentum_vol_features(bars_df)
    log.info(f"Champion 30-min features: {len(feature_cols)}")

    all_dates = sorted(bars_df["date"].unique())
    log.info(f"Total dates: {len(all_dates)} ({all_dates[0]} -> {all_dates[-1]})")

    features_all = bars_df[feature_cols].values.astype(np.float32)
    labels_all = bars_df["fwd_ticks"].values.astype(np.float32)
    dates_all = bars_df["date"].values
    ts_all = bars_df["ts"].values
    close_all = bars_df["close"].values

    cfg = CHAMPION_30MIN_CONFIG
    lgbm_params = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": cfg["learning_rate"],
        "num_leaves": cfg["num_leaves"],
        "min_child_samples": cfg["min_data_in_leaf"],
        "feature_fraction": cfg["feature_fraction"],
        "bagging_fraction": cfg["bagging_fraction"],
        "bagging_freq": 5,
        "lambda_l1": cfg["reg_alpha"],
        "lambda_l2": cfg["reg_lambda"],
        "max_depth": cfg["max_depth"],
        "verbose": -1,
        "n_jobs": -1,
    }

    oot_records = []
    fold_ics = []
    fold_idx = 0

    for fold_start in range(WF_TRAIN_DAYS, len(all_dates) - WF_VAL_DAYS + 1, WF_SLIDE):
        fold_train_dates = all_dates[fold_start - WF_TRAIN_DAYS: fold_start]
        fold_val_dates = all_dates[fold_start: fold_start + WF_VAL_DAYS]

        if len(fold_val_dates) < WF_VAL_DAYS:
            break

        fold_idx += 1

        train_mask = np.isin(dates_all, fold_train_dates)
        val_mask = np.isin(dates_all, fold_val_dates)

        X_train_raw = features_all[train_mask].copy()
        X_val_raw = features_all[val_mask].copy()

        # Robust scaling from train
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
        val_ts_fold = ts_all[val_mask]
        val_close_fold = close_all[val_mask]

        train_valid = ~np.isnan(y_train)
        val_valid = ~np.isnan(y_val)

        if train_valid.sum() < 50 or val_valid.sum() < 10:
            continue

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]

        params = {**lgbm_params, "seed": 42 + fold_idx}
        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)

        # Use a small held-out set from train end for early stopping
        n_es = min(200, len(X_tr) // 5)
        es_data = lgb.Dataset(X_tr[-n_es:], label=y_tr[-n_es:], feature_name=feature_cols, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        try:
            model = lgb.train(
                params,
                train_data,
                num_boost_round=500,
                valid_sets=[es_data],
                callbacks=callbacks,
            )
        except Exception as e:
            log.warning(f"Fold {fold_idx} train failed: {e}")
            continue

        # Predict on ALL val bars (including those with NaN targets)
        X_v_all = X_val
        val_preds_all = model.predict(X_v_all, num_iteration=model.best_iteration)

        for i in range(len(val_preds_all)):
            oot_records.append({
                "fold": fold_idx,
                "date": val_dates_fold[i],
                "ts": val_ts_fold[i],
                "close": val_close_fold[i],
                "pred": val_preds_all[i],
                "actual": y_val[i],
            })

        # Fold IC on valid targets only
        X_v = X_val[val_valid]
        y_v = y_val[val_valid]
        val_preds = model.predict(X_v, num_iteration=model.best_iteration)
        if len(val_preds) > 5:
            ic = np.corrcoef(val_preds, y_v)[0, 1]
            if not np.isnan(ic):
                fold_ics.append(ic)

        del model, train_data, es_data
        gc.collect()

        if fold_idx % 5 == 0:
            log.info(f"  Fold {fold_idx}: IC={fold_ics[-1]:.4f}" if fold_ics else f"  Fold {fold_idx}: no IC")

    oot_df = pd.DataFrame(oot_records)
    valid_mask = ~oot_df["actual"].isna()
    concat_ic = float(np.corrcoef(oot_df.loc[valid_mask, "pred"], oot_df.loc[valid_mask, "actual"])[0, 1])
    ic_sharpe = float(np.mean(fold_ics) / max(np.std(fold_ics), 1e-6)) if len(fold_ics) > 2 else float("nan")

    log.info(f"30-min WF complete: {fold_idx} folds, {len(oot_df)} OOT bars, "
             f"concat IC={concat_ic:.4f}, IC Sharpe={ic_sharpe:.2f}")

    return oot_df


# ═══════════════════════════════════════════════════════════════════
#  DAILY REGIME CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════


def classify_regimes(daily_features: pd.DataFrame) -> Dict[str, str]:
    """Classify each date as green/red/flat based on close-to-close return."""
    regimes = {}
    df = daily_features.copy()
    df["date_str"] = df["date"].astype(str).str.replace("-", "")

    for _, row in df.iterrows():
        cc = row.get("cc_return_ticks", 0)
        if pd.isna(cc):
            regimes[row["date_str"]] = "flat"
        elif cc > 4:   # > 1 point
            regimes[row["date_str"]] = "green"
        elif cc < -4:
            regimes[row["date_str"]] = "red"
        else:
            regimes[row["date_str"]] = "flat"

    return regimes


# ═══════════════════════════════════════════════════════════════════
#  LONG-HORIZON DAILY SIGNAL LOOKUP
# ═══════════════════════════════════════════════════════════════════


def build_long_horizon_daily_signals(lh_data: Dict) -> pd.DataFrame:
    """
    Build a daily signal DataFrame from long-horizon predictions.
    Returns: DataFrame with columns [date_str, lh_direction, lh_confidence, lh_pred_1d]
    """
    pred_3d = lh_data["pred_3d"].copy()
    pred_1d = lh_data["pred_1d"].copy()

    # Standardize date format
    pred_3d["date_str"] = pred_3d["date"].astype(str).str.replace("-", "").str[:8]
    pred_1d["date_str"] = pred_1d["date"].astype(str).str.replace("-", "").str[:8]

    # 3d direction: pred > 0.5 = bullish, pred < 0.5 = bearish
    pred_3d["lh_direction"] = np.where(pred_3d["pred"] > 0.5, 1, -1)
    pred_3d["lh_confidence"] = (pred_3d["pred"] - 0.5).abs() * 2  # 0-1 scale

    # 1d regression: sign = direction, magnitude = confidence
    pred_1d["lh_pred_1d"] = pred_1d["pred"]
    pred_1d["lh_1d_direction"] = np.sign(pred_1d["pred"])

    # Merge
    signals = pred_3d[["date_str", "lh_direction", "lh_confidence"]].merge(
        pred_1d[["date_str", "lh_pred_1d", "lh_1d_direction"]],
        on="date_str", how="outer"
    )

    # For the NEXT trading day, we use the PREVIOUS day's prediction
    # (predictions are made EOD for next day's action)
    signals = signals.sort_values("date_str").reset_index(drop=True)
    signals["signal_for_date"] = signals["date_str"].shift(-1)
    signals = signals.dropna(subset=["signal_for_date"])

    log.info(f"Long-horizon daily signals: {len(signals)} days")
    return signals


# ═══════════════════════════════════════════════════════════════════
#  TRADE SIMULATION HELPERS
# ═══════════════════════════════════════════════════════════════════


def simulate_short_horizon_trades(
    oot_df: pd.DataFrame,
    threshold_pct: float = 0.15,
    cost_rt: float = SHORT_COST_RT_TICKS,
    tp_ticks: float = 20.0,
    sl_ticks: float = 10.0,
) -> pd.DataFrame:
    """
    Simulate 30-min short-horizon trades.
    Takes top/bottom threshold_pct of predictions as long/short.
    Uses SL10/TP20 with 1-bar hold assumption for simplicity.
    Returns per-trade DataFrame.
    """
    valid = oot_df.dropna(subset=["actual"]).copy()
    if len(valid) < 20:
        return pd.DataFrame()

    upper = valid["pred"].quantile(1 - threshold_pct)
    lower = valid["pred"].quantile(threshold_pct)

    trades = []
    for _, row in valid.iterrows():
        if row["pred"] >= upper:
            direction = 1
            pnl = row["actual"] - cost_rt
        elif row["pred"] <= lower:
            direction = -1
            pnl = -row["actual"] - cost_rt
        else:
            continue

        # Apply SL/TP caps (approximate from 30-min bar actual move)
        actual_move = abs(row["actual"])
        if direction == 1:
            if row["actual"] > tp_ticks:
                pnl = tp_ticks - cost_rt
            elif row["actual"] < -sl_ticks:
                pnl = -sl_ticks - cost_rt
        else:
            if -row["actual"] > tp_ticks:
                pnl = tp_ticks - cost_rt
            elif -row["actual"] < -sl_ticks:
                pnl = -sl_ticks - cost_rt

        trades.append({
            "date": row["date"],
            "ts": row["ts"],
            "direction": direction,
            "pred": row["pred"],
            "actual": row["actual"],
            "pnl_ticks": pnl,
            "close": row["close"],
            "strategy": "short_horizon",
        })

    return pd.DataFrame(trades)


def simulate_long_horizon_trades(lh_data: Dict) -> pd.DataFrame:
    """
    Use the pre-computed long-horizon multiday trade log.
    Converts to the same format as short-horizon trades.
    """
    mt = lh_data["multiday_trades"].copy()
    trades = []

    for _, row in mt.iterrows():
        # Standardize dates
        sig_date = str(row["signal_date"]).replace("-", "")[:8]
        exit_date = str(row["exit_date"]).replace("-", "")[:8]

        trades.append({
            "date": sig_date,
            "exit_date": exit_date,
            "direction": row["direction"],
            "pred": row["pred"],
            "pnl_ticks": row["pnl_ticks"],
            "hold_days": row["hold_days"],
            "entry_price": row["entry_price"],
            "exit_price": row["exit_price"],
            "strategy": "long_horizon",
        })

    return pd.DataFrame(trades)


# ═══════════════════════════════════════════════════════════════════
#  METRICS COMPUTATION
# ═══════════════════════════════════════════════════════════════════


def compute_metrics(pnl_series: np.ndarray, name: str, regimes: Dict = None,
                    dates: np.ndarray = None) -> Dict:
    """Compute comprehensive metrics for a P&L array."""
    if len(pnl_series) < 3:
        return {"name": name, "error": "too few trades"}

    n = len(pnl_series)
    wr = float(np.mean(pnl_series > 0))
    total_pnl = float(np.sum(pnl_series))
    avg_pnl = float(np.mean(pnl_series))

    # Sharpe (annualized, assuming ~5 trades/day avg)
    daily_factor = np.sqrt(252)
    sharpe = float(np.mean(pnl_series) / max(np.std(pnl_series), 1e-6) * daily_factor)

    # Sortino
    downside = np.sqrt(np.mean(np.minimum(pnl_series, 0) ** 2))
    sortino = float(np.mean(pnl_series) / max(downside, 1e-6) * daily_factor)

    # Profit factor
    gross_profit = float(np.sum(pnl_series[pnl_series > 0]))
    gross_loss = float(-np.sum(pnl_series[pnl_series < 0]))
    pf = gross_profit / max(gross_loss, 1e-6)

    # Max drawdown (cumulative)
    cum_pnl = np.cumsum(pnl_series)
    peak = np.maximum.accumulate(cum_pnl)
    dd = peak - cum_pnl
    max_dd = float(np.max(dd)) if len(dd) > 0 else 0

    # Day-level concentration (HC #344: cap <= 0.70)
    day_conc = 0.0
    if dates is not None and len(dates) > 0:
        unique_dates = np.unique(dates)
        day_pnl = []
        for d in unique_dates:
            mask = dates == d
            day_pnl.append(np.sum(pnl_series[mask]))
        day_pnl = np.array(day_pnl)
        if len(day_pnl) > 0 and total_pnl > 0:
            day_conc = float(np.max(np.abs(day_pnl)) / max(total_pnl, 1e-6))

    result = {
        "name": name,
        "n_trades": n,
        "wr": round(wr, 4),
        "pf": round(pf, 3),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "total_pnl_ticks": round(total_pnl, 1),
        "total_pnl_dollars": round(total_pnl * ES_TICK_VALUE, 0),
        "avg_pnl_ticks": round(avg_pnl, 2),
        "max_dd_ticks": round(max_dd, 1),
        "max_dd_dollars": round(max_dd * ES_TICK_VALUE, 0),
        "day_concentration": round(day_conc, 3),
    }

    # Per-regime metrics
    if regimes is not None and dates is not None:
        regime_arr = np.array([regimes.get(str(d), "flat") for d in dates])
        regime_metrics = {}
        for regime in ["green", "red", "flat"]:
            mask = regime_arr == regime
            if mask.sum() < 3:
                continue
            r_pnl = pnl_series[mask]
            r_sharpe = float(np.mean(r_pnl) / max(np.std(r_pnl), 1e-6) * daily_factor)
            regime_metrics[regime] = {
                "n": int(mask.sum()),
                "sharpe": round(r_sharpe, 3),
                "wr": round(float(np.mean(r_pnl > 0)), 4),
                "mean_pnl": round(float(np.mean(r_pnl)), 2),
                "total_pnl": round(float(np.sum(r_pnl)), 1),
            }

        result["regime"] = regime_metrics

        # Regime gap check
        if "green" in regime_metrics and "red" in regime_metrics:
            s_g = regime_metrics["green"]["sharpe"]
            s_r = regime_metrics["red"]["sharpe"]
            denom = max(abs(s_g), abs(s_r), 1e-6)
            gap = abs(s_g - s_r) / denom
            result["regime_gap"] = round(gap, 4)
            result["regime_pass"] = gap <= 0.50
        else:
            result["regime_gap"] = None
            result["regime_pass"] = None

    return result


# ═══════════════════════════════════════════════════════════════════
#  APPROACH A: PORTFOLIO ALLOCATION (INDEPENDENT)
# ═══════════════════════════════════════════════════════════════════


def approach_a_portfolio(
    short_trades: pd.DataFrame,
    long_trades: pd.DataFrame,
    regimes: Dict,
    weight_short: float = 0.5,
    weight_long: float = 0.5,
) -> Dict:
    """
    Run both strategies independently with equal capital allocation.
    Returns combined metrics.
    """
    log.info("=== Approach A: Portfolio Allocation ===")

    # Build daily P&L for each strategy
    all_dates = set()

    short_daily = {}
    for _, row in short_trades.iterrows():
        d = row["date"]
        all_dates.add(d)
        short_daily[d] = short_daily.get(d, 0) + row["pnl_ticks"]

    long_daily = {}
    for _, row in long_trades.iterrows():
        d = row["date"]
        all_dates.add(d)
        # Long-horizon trades span multiple days; attribute P&L to signal date
        long_daily[d] = long_daily.get(d, 0) + row["pnl_ticks"]

    sorted_dates = sorted(all_dates)

    # Combined daily P&L (weighted)
    combined_daily = []
    combined_dates = []
    short_only_daily = []
    long_only_daily = []

    for d in sorted_dates:
        s_pnl = short_daily.get(d, 0) * weight_short
        l_pnl = long_daily.get(d, 0) * weight_long
        combined_daily.append(s_pnl + l_pnl)
        combined_dates.append(d)
        short_only_daily.append(short_daily.get(d, 0))
        long_only_daily.append(long_daily.get(d, 0))

    combined_daily = np.array(combined_daily)
    combined_dates = np.array(combined_dates)

    # Compute correlation between strategies
    short_arr = np.array(short_only_daily)
    long_arr = np.array(long_only_daily)
    corr = float(np.corrcoef(short_arr, long_arr)[0, 1]) if len(short_arr) > 5 else float("nan")

    # Combined trade-level for metrics
    all_pnl = np.concatenate([
        short_trades["pnl_ticks"].values * weight_short,
        long_trades["pnl_ticks"].values * weight_long,
    ])
    all_trade_dates = np.concatenate([
        short_trades["date"].values,
        long_trades["date"].values,
    ])

    metrics = compute_metrics(all_pnl, "portfolio_allocation", regimes, all_trade_dates)
    metrics["strategy_correlation"] = round(corr, 4)
    metrics["weight_short"] = weight_short
    metrics["weight_long"] = weight_long

    # Also compute daily-level Sharpe (more meaningful for portfolio)
    if len(combined_daily) > 5:
        daily_sharpe = float(np.mean(combined_daily) / max(np.std(combined_daily), 1e-6) * np.sqrt(252))
        daily_sortino_down = np.sqrt(np.mean(np.minimum(combined_daily, 0) ** 2))
        daily_sortino = float(np.mean(combined_daily) / max(daily_sortino_down, 1e-6) * np.sqrt(252))
        metrics["daily_sharpe"] = round(daily_sharpe, 3)
        metrics["daily_sortino"] = round(daily_sortino, 3)

        # Daily regime
        green_pnl = [p for p, d in zip(combined_daily, combined_dates) if regimes.get(d, "flat") == "green"]
        red_pnl = [p for p, d in zip(combined_daily, combined_dates) if regimes.get(d, "flat") == "red"]

        if len(green_pnl) > 3 and len(red_pnl) > 3:
            g_sharpe = float(np.mean(green_pnl) / max(np.std(green_pnl), 1e-6) * np.sqrt(252))
            r_sharpe = float(np.mean(red_pnl) / max(np.std(red_pnl), 1e-6) * np.sqrt(252))
            gap = abs(g_sharpe - r_sharpe) / max(abs(g_sharpe), abs(r_sharpe), 1e-6)
            metrics["daily_regime_gap"] = round(gap, 4)
            metrics["daily_regime_pass"] = gap <= 0.50
            metrics["daily_green_sharpe"] = round(g_sharpe, 3)
            metrics["daily_red_sharpe"] = round(r_sharpe, 3)

    # Short standalone metrics
    short_metrics = compute_metrics(
        short_trades["pnl_ticks"].values, "short_horizon_standalone",
        regimes, short_trades["date"].values
    )
    long_metrics = compute_metrics(
        long_trades["pnl_ticks"].values, "long_horizon_standalone",
        regimes, long_trades["date"].values
    )

    log.info(f"  Combined: Sharpe={metrics['sharpe']}, Sortino={metrics['sortino']}, "
             f"PF={metrics['pf']}, WR={metrics['wr']}")
    log.info(f"  Daily Sharpe={metrics.get('daily_sharpe','N/A')}, Correlation={corr:.4f}")

    return {
        "combined": metrics,
        "short_standalone": short_metrics,
        "long_standalone": long_metrics,
    }


# ═══════════════════════════════════════════════════════════════════
#  APPROACH B: CONFLUENCE GATING
# ═══════════════════════════════════════════════════════════════════


def approach_b_confluence(
    short_oot_df: pd.DataFrame,
    lh_signals: pd.DataFrame,
    regimes: Dict,
    threshold_pct: float = 0.15,
    cost_rt: float = CONFLUENCE_COST_RT_TICKS,
) -> Dict:
    """
    Only take 30-min trades when long-horizon flow agrees on direction.
    """
    log.info("=== Approach B: Confluence Gating ===")

    # Build lookup: date -> long-horizon direction
    lh_lookup = {}
    for _, row in lh_signals.iterrows():
        d = row.get("signal_for_date", "")
        if pd.notna(d) and d != "":
            lh_lookup[str(d)[:8]] = {
                "direction": row.get("lh_direction", 0),
                "confidence": row.get("lh_confidence", 0),
                "pred_1d": row.get("lh_pred_1d", 0),
            }

    valid = short_oot_df.dropna(subset=["actual"]).copy()
    if len(valid) < 20:
        return {"error": "too few bars"}

    upper = valid["pred"].quantile(1 - threshold_pct)
    lower = valid["pred"].quantile(threshold_pct)

    trades_gated = []
    trades_ungated = []

    for _, row in valid.iterrows():
        date_str = str(row["date"])[:8]

        if row["pred"] >= upper:
            short_dir = 1
        elif row["pred"] <= lower:
            short_dir = -1
        else:
            continue

        # Ungated trade (baseline)
        if short_dir == 1:
            pnl_ungated = row["actual"] - SHORT_COST_RT_TICKS
        else:
            pnl_ungated = -row["actual"] - SHORT_COST_RT_TICKS

        trades_ungated.append({
            "date": date_str,
            "direction": short_dir,
            "pnl_ticks": pnl_ungated,
            "pred": row["pred"],
            "actual": row["actual"],
        })

        # Confluence gate: only take if long-horizon agrees
        lh = lh_lookup.get(date_str, None)
        if lh is None:
            continue  # no long-horizon signal for this date

        lh_dir = lh["direction"]
        if lh_dir == short_dir:
            # Agreement! Take the trade with better cost (can use limits)
            if short_dir == 1:
                pnl = row["actual"] - cost_rt
            else:
                pnl = -row["actual"] - cost_rt

            trades_gated.append({
                "date": date_str,
                "direction": short_dir,
                "pnl_ticks": pnl,
                "pred": row["pred"],
                "actual": row["actual"],
                "lh_confidence": lh["confidence"],
            })

    trades_gated_df = pd.DataFrame(trades_gated)
    trades_ungated_df = pd.DataFrame(trades_ungated)

    result = {}

    # Gated metrics
    if len(trades_gated_df) > 5:
        gated_metrics = compute_metrics(
            trades_gated_df["pnl_ticks"].values, "confluence_gated",
            regimes, trades_gated_df["date"].values
        )
        gated_metrics["n_filtered_out"] = len(trades_ungated_df) - len(trades_gated_df)
        gated_metrics["filter_rate"] = round(
            1 - len(trades_gated_df) / max(len(trades_ungated_df), 1), 3
        )
        result["gated"] = gated_metrics
        log.info(f"  Gated: {len(trades_gated_df)} trades (filtered {gated_metrics['filter_rate']*100:.0f}%), "
                 f"Sharpe={gated_metrics['sharpe']}, PF={gated_metrics['pf']}")
    else:
        result["gated"] = {"error": f"only {len(trades_gated_df)} gated trades"}

    # Ungated baseline
    if len(trades_ungated_df) > 5:
        ungated_metrics = compute_metrics(
            trades_ungated_df["pnl_ticks"].values, "no_gate_baseline",
            regimes, trades_ungated_df["date"].values
        )
        result["ungated"] = ungated_metrics
    else:
        result["ungated"] = {"error": "too few ungated trades"}

    # Anti-confluence (disagreement only) - diagnostic
    trades_anti = []
    for _, row in valid.iterrows():
        date_str = str(row["date"])[:8]
        if row["pred"] >= upper:
            short_dir = 1
        elif row["pred"] <= lower:
            short_dir = -1
        else:
            continue
        lh = lh_lookup.get(date_str, None)
        if lh is not None and lh["direction"] != short_dir:
            if short_dir == 1:
                pnl = row["actual"] - SHORT_COST_RT_TICKS
            else:
                pnl = -row["actual"] - SHORT_COST_RT_TICKS
            trades_anti.append({
                "date": date_str,
                "direction": short_dir,
                "pnl_ticks": pnl,
            })

    if len(trades_anti) > 5:
        anti_df = pd.DataFrame(trades_anti)
        anti_metrics = compute_metrics(
            anti_df["pnl_ticks"].values, "anti_confluence",
            regimes, anti_df["date"].values
        )
        result["anti_confluence"] = anti_metrics
        log.info(f"  Anti-confluence (disagreement): Sharpe={anti_metrics['sharpe']}, "
                 f"PF={anti_metrics['pf']}, n={anti_metrics['n_trades']}")

    return result


# ═══════════════════════════════════════════════════════════════════
#  APPROACH C: POSITION SIZING
# ═══════════════════════════════════════════════════════════════════


def approach_c_position_sizing(
    short_oot_df: pd.DataFrame,
    lh_signals: pd.DataFrame,
    regimes: Dict,
    threshold_pct: float = 0.15,
) -> Dict:
    """
    Scale 30-min trade size by long-horizon confidence.
    High LH confidence + agreement = 1.5x. Low confidence / disagreement = 0.5x.
    """
    log.info("=== Approach C: Position Sizing ===")

    lh_lookup = {}
    for _, row in lh_signals.iterrows():
        d = row.get("signal_for_date", "")
        if pd.notna(d) and d != "":
            lh_lookup[str(d)[:8]] = {
                "direction": row.get("lh_direction", 0),
                "confidence": row.get("lh_confidence", 0),
            }

    valid = short_oot_df.dropna(subset=["actual"]).copy()
    if len(valid) < 20:
        return {"error": "too few bars"}

    upper = valid["pred"].quantile(1 - threshold_pct)
    lower = valid["pred"].quantile(threshold_pct)

    # Test multiple sizing schemes
    sizing_schemes = {
        "binary_gate": {"agree": 1.5, "disagree": 0.5, "no_signal": 1.0},
        "confidence_scaled": None,  # continuous scaling
        "agree_only_scaled": {"agree": None, "disagree": 0.0, "no_signal": 1.0},
    }

    results = {}

    for scheme_name, params in sizing_schemes.items():
        trades = []
        for _, row in valid.iterrows():
            date_str = str(row["date"])[:8]

            if row["pred"] >= upper:
                short_dir = 1
            elif row["pred"] <= lower:
                short_dir = -1
            else:
                continue

            if short_dir == 1:
                base_pnl = row["actual"] - SHORT_COST_RT_TICKS
            else:
                base_pnl = -row["actual"] - SHORT_COST_RT_TICKS

            lh = lh_lookup.get(date_str, None)

            if scheme_name == "binary_gate":
                if lh is None:
                    size = params["no_signal"]
                elif lh["direction"] == short_dir:
                    size = params["agree"]
                else:
                    size = params["disagree"]

            elif scheme_name == "confidence_scaled":
                if lh is None:
                    size = 1.0
                else:
                    conf = lh["confidence"]  # 0-1
                    if lh["direction"] == short_dir:
                        size = 1.0 + conf  # 1.0 to 2.0
                    else:
                        size = max(0.25, 1.0 - conf)  # 0.25 to 1.0

            elif scheme_name == "agree_only_scaled":
                if lh is None:
                    size = params["no_signal"]
                elif lh["direction"] != short_dir:
                    size = params["disagree"]
                else:
                    # Scale by LH confidence
                    size = 0.5 + lh["confidence"]  # 0.5 to 1.5

            else:
                size = 1.0

            # Cost scales with size (more contracts = proportional cost)
            scaled_pnl = base_pnl * size

            trades.append({
                "date": date_str,
                "direction": short_dir,
                "pnl_ticks": scaled_pnl,
                "size": size,
            })

        if len(trades) > 5:
            trades_df = pd.DataFrame(trades)
            metrics = compute_metrics(
                trades_df["pnl_ticks"].values, f"sizing_{scheme_name}",
                regimes, trades_df["date"].values
            )
            metrics["avg_size"] = round(float(trades_df["size"].mean()), 3)
            metrics["size_range"] = [
                round(float(trades_df["size"].min()), 3),
                round(float(trades_df["size"].max()), 3),
            ]
            results[scheme_name] = metrics
            log.info(f"  {scheme_name}: Sharpe={metrics['sharpe']}, PF={metrics['pf']}, "
                     f"avg_size={metrics['avg_size']}")
        else:
            results[scheme_name] = {"error": f"only {len(trades)} trades"}

    return results


# ═══════════════════════════════════════════════════════════════════
#  PER-DAY ANALYSIS (for regime gate R1)
# ═══════════════════════════════════════════════════════════════════


def per_day_analysis(trades_df: pd.DataFrame, regimes: Dict) -> Dict:
    """Compute per-day Sharpe/PF/WR and stratified regime metrics (R1 gate)."""
    if len(trades_df) < 5:
        return {}

    days = sorted(trades_df["date"].unique())
    day_records = []

    for d in days:
        mask = trades_df["date"] == d
        day_pnl = trades_df.loc[mask, "pnl_ticks"].values
        regime = regimes.get(str(d), "flat")

        day_records.append({
            "date": d,
            "regime": regime,
            "n_trades": len(day_pnl),
            "total_pnl": float(np.sum(day_pnl)),
            "mean_pnl": float(np.mean(day_pnl)),
            "wr": float(np.mean(day_pnl > 0)),
            "profitable": float(np.sum(day_pnl)) > 0,
        })

    day_df = pd.DataFrame(day_records)

    # Stratified Sharpe per regime
    regime_sharpes = {}
    for regime in ["green", "red", "flat"]:
        r_days = day_df[day_df["regime"] == regime]
        if len(r_days) >= 3:
            daily_pnl = r_days["total_pnl"].values
            regime_sharpes[regime] = {
                "n_days": len(r_days),
                "daily_sharpe": round(float(np.mean(daily_pnl) / max(np.std(daily_pnl), 1e-6) * np.sqrt(252)), 3),
                "wr": round(float(np.mean(r_days["profitable"])), 3),
                "total_pnl": round(float(np.sum(daily_pnl)), 1),
                "mean_daily_pnl": round(float(np.mean(daily_pnl)), 2),
            }

    return {
        "n_days": len(days),
        "n_profitable_days": int(day_df["profitable"].sum()),
        "day_wr": round(float(day_df["profitable"].mean()), 3),
        "per_regime": regime_sharpes,
        "per_day": day_records,
    }


# ═══════════════════════════════════════════════════════════════════
#  MLFLOW LOGGING
# ═══════════════════════════════════════════════════════════════════


def log_to_mlflow(results: Dict, approach_name: str):
    """Log results to MLflow on Jupiter."""
    try:
        import mlflow

        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("multi_timeframe_ensemble_v1")

        with mlflow.start_run(run_name=f"mtf_{approach_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Flatten and log metrics
            def _log_flat(d, prefix=""):
                for k, v in d.items():
                    if isinstance(v, (int, float)) and not isinstance(v, bool) and not np.isnan(v) if isinstance(v, float) else True:
                        try:
                            mlflow.log_metric(f"{prefix}{k}", v)
                        except Exception:
                            pass
                    elif isinstance(v, dict):
                        _log_flat(v, prefix=f"{prefix}{k}.")

            _log_flat(results)

            # Log params
            mlflow.log_param("approach", approach_name)
            mlflow.log_param("short_cost_rt", SHORT_COST_RT_TICKS)
            mlflow.log_param("long_cost_rt", LONG_COST_RT_TICKS)
            mlflow.log_param("wf_train_days", WF_TRAIN_DAYS)
            mlflow.log_param("wf_val_days", WF_VAL_DAYS)
            mlflow.log_param("wf_slide", WF_SLIDE)
            mlflow.log_param("champion_config", json.dumps(CHAMPION_30MIN_CONFIG))

        log.info(f"  Logged to MLflow: {approach_name}")
        return True
    except Exception as e:
        log.warning(f"  MLflow logging failed: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    log.info("=" * 70)
    log.info("MULTI-TIMEFRAME ENSEMBLE v1")
    log.info("=" * 70)
    log.info(f"Short cost: {SHORT_COST_RT_TICKS} ticks RT (aggressive)")
    log.info(f"Long cost: {LONG_COST_RT_TICKS} ticks RT (passive)")
    log.info(f"Confluence cost: {CONFLUENCE_COST_RT_TICKS} ticks RT (passive entry + aggressive exit)")
    log.info(f"WF: {WF_TRAIN_DAYS}d train, {WF_VAL_DAYS}d val, slide {WF_SLIDE}d")

    t0 = time.time()

    # ── Step 1: Load long-horizon data ──
    log.info("\n--- Loading long-horizon data ---")
    lh_data = load_long_horizon_data()
    lh_signals = build_long_horizon_daily_signals(lh_data)
    regimes = classify_regimes(lh_data["daily_features"])

    log.info(f"Regimes: {sum(1 for v in regimes.values() if v == 'green')} green, "
             f"{sum(1 for v in regimes.values() if v == 'red')} red, "
             f"{sum(1 for v in regimes.values() if v == 'flat')} flat")

    # ── Step 2: Rebuild 30-min WF predictions ──
    log.info("\n--- Loading minute bars and building 30-min features ---")
    minute_df = load_all_minute_bars()
    bars_df = aggregate_to_bars(minute_df, bar_size_min=30)
    bars_df = add_rolling_features(bars_df)
    bars_df = add_forward_labels(bars_df, horizon_bars=1)

    del minute_df
    gc.collect()

    log.info("\n--- Running 30-min walk-forward (champion config) ---")
    short_oot_df = rebuild_30min_predictions(bars_df)

    del bars_df
    gc.collect()

    # Save short OOT predictions
    short_oot_df.to_parquet(OUTPUT_DIR / "short_30min_oot_predictions.parquet", index=False)
    log.info(f"Saved {len(short_oot_df)} short OOT predictions")

    # ── Step 3: Simulate trades ──
    log.info("\n--- Simulating trades ---")
    short_trades = simulate_short_horizon_trades(short_oot_df, threshold_pct=0.15)
    long_trades = simulate_long_horizon_trades(lh_data)

    log.info(f"Short-horizon trades: {len(short_trades)}")
    log.info(f"Long-horizon trades: {len(long_trades)}")

    # Check date overlap
    short_dates = set(short_trades["date"].unique()) if len(short_trades) > 0 else set()
    long_dates = set(long_trades["date"].unique()) if len(long_trades) > 0 else set()
    overlap = short_dates & long_dates
    log.info(f"Date overlap: {len(overlap)} days (short has {len(short_dates)}, long has {len(long_dates)})")

    # ── Step 4: Run all three approaches ──
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "config": {
            "short_cost_rt": SHORT_COST_RT_TICKS,
            "long_cost_rt": LONG_COST_RT_TICKS,
            "confluence_cost_rt": CONFLUENCE_COST_RT_TICKS,
            "wf_train_days": WF_TRAIN_DAYS,
            "wf_val_days": WF_VAL_DAYS,
            "wf_slide": WF_SLIDE,
            "champion_30min_config": CHAMPION_30MIN_CONFIG,
            "threshold_pct": 0.15,
        },
    }

    # A) Portfolio allocation
    if len(short_trades) > 10 and len(long_trades) > 3:
        result_a = approach_a_portfolio(short_trades, long_trades, regimes)
        all_results["approach_a_portfolio"] = result_a

        # Test different allocations
        for ws, wl in [(0.7, 0.3), (0.3, 0.7), (0.6, 0.4)]:
            r = approach_a_portfolio(short_trades, long_trades, regimes, ws, wl)
            all_results[f"approach_a_w{int(ws*100)}_{int(wl*100)}"] = r
            log.info(f"  w={ws}/{wl}: Sharpe={r['combined']['sharpe']}")

        log_to_mlflow(result_a["combined"], "portfolio_allocation")

    # B) Confluence gating
    result_b = approach_b_confluence(short_oot_df, lh_signals, regimes)
    all_results["approach_b_confluence"] = result_b
    if "gated" in result_b and "error" not in result_b["gated"]:
        log_to_mlflow(result_b["gated"], "confluence_gated")

    # C) Position sizing
    result_c = approach_c_position_sizing(short_oot_df, lh_signals, regimes)
    all_results["approach_c_sizing"] = result_c
    for scheme, metrics in result_c.items():
        if isinstance(metrics, dict) and "error" not in metrics:
            log_to_mlflow(metrics, f"sizing_{scheme}")

    # ── Step 5: Per-day analysis for best approaches ──
    log.info("\n--- Per-day regime analysis ---")

    # Analyze the best combined approach
    if len(short_trades) > 10:
        short_perday = per_day_analysis(short_trades, regimes)
        all_results["short_horizon_perday"] = short_perday
        log.info(f"Short: {short_perday['n_profitable_days']}/{short_perday['n_days']} profitable days "
                 f"({short_perday['day_wr']*100:.0f}%)")

    if len(long_trades) > 3:
        long_perday = per_day_analysis(long_trades, regimes)
        all_results["long_horizon_perday"] = long_perday
        log.info(f"Long: {long_perday.get('n_profitable_days','?')}/{long_perday.get('n_days','?')} profitable days")

    # ── Step 6: Save results ──
    elapsed = time.time() - t0

    all_results["runtime_minutes"] = round(elapsed / 60, 2)

    with open(OUTPUT_DIR / "results.json", "w") as f:
        json.dump(all_results, f, indent=2, default=str)

    # Save trade logs
    if len(short_trades) > 0:
        short_trades.to_parquet(OUTPUT_DIR / "short_trades.parquet", index=False)
    if len(long_trades) > 0:
        long_trades.to_parquet(OUTPUT_DIR / "long_trades.parquet", index=False)

    # ── Summary ──
    log.info("\n" + "=" * 70)
    log.info("FINAL RESULTS SUMMARY")
    log.info("=" * 70)

    if "approach_a_portfolio" in all_results:
        a = all_results["approach_a_portfolio"]["combined"]
        log.info(f"\nA) Portfolio (50/50): Sharpe={a['sharpe']}, Sortino={a['sortino']}, "
                 f"PF={a['pf']}, WR={a['wr']}")
        if "regime_gap" in a and a["regime_gap"] is not None:
            log.info(f"   Regime gap={a['regime_gap']:.4f} {'PASS' if a.get('regime_pass') else 'FAIL'}")

    if "approach_b_confluence" in all_results:
        b = all_results["approach_b_confluence"]
        if "gated" in b and "error" not in b.get("gated", {}):
            bg = b["gated"]
            log.info(f"\nB) Confluence gated: Sharpe={bg['sharpe']}, Sortino={bg['sortino']}, "
                     f"PF={bg['pf']}, WR={bg['wr']}, filtered={bg.get('filter_rate',0)*100:.0f}%")
            if "regime_gap" in bg and bg["regime_gap"] is not None:
                log.info(f"   Regime gap={bg['regime_gap']:.4f} {'PASS' if bg.get('regime_pass') else 'FAIL'}")
        if "ungated" in b and "error" not in b.get("ungated", {}):
            bu = b["ungated"]
            log.info(f"   Ungated baseline: Sharpe={bu['sharpe']}, PF={bu['pf']}")

    if "approach_c_sizing" in all_results:
        log.info(f"\nC) Position sizing:")
        for scheme, m in all_results["approach_c_sizing"].items():
            if isinstance(m, dict) and "error" not in m:
                log.info(f"   {scheme}: Sharpe={m['sharpe']}, PF={m['pf']}, avg_size={m.get('avg_size','?')}")
                if "regime_gap" in m and m["regime_gap"] is not None:
                    log.info(f"     Regime gap={m['regime_gap']:.4f} {'PASS' if m.get('regime_pass') else 'FAIL'}")

    log.info(f"\nRuntime: {elapsed/60:.1f} minutes")
    log.info(f"Results saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
