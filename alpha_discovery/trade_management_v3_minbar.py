#!/usr/bin/env python3
"""
Trade Management Model v3 — Minute-Bar Dynamic Exit Optimization
================================================================

Builds on v2's proof-of-concept (dynamic exits beat static by +30% Sharpe)
but uses the FULL 197-day minute bar dataset instead of 41 tick-data days.

Architecture:
  Phase 1: Train lean 30-min entry model (reuse v4 code) via walk-forward
  Phase 2: For each high-confidence entry signal, extract minute-level
            management features during the 30-min trade window
  Phase 3: Train LightGBM binary classifier: should we EXIT NOW?
           Label: hindsight-optimal — at minute t, if remaining PnL < 0, EXIT=1
  Phase 4: Simulate static 30-min hold vs dynamic management model

Walk-Forward: 60d train, 10d val, 5d slide — SLIDING only (HC #0).
Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission).

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python -u \\
      alpha_discovery/trade_management_v3_minbar.py

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
OUTPUT_DIR = ROOT / "output" / "trade_management_v3"
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
    format="%(asctime)s [TM-v3] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "trade_management_v3.log")),
    ],
)
log = logging.getLogger("TM-v3")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
COST_RT_TICKS = 2.376  # market entry + market exit + RT commission

# Walk-forward config — entry model
TRAIN_DAYS = 60
VAL_DAYS = 10
SLIDE_DAYS = 5

# Walk-forward config — management model (shorter: simpler features, need more folds)
MGMT_TRAIN_DAYS = 30
MGMT_VAL_DAYS = 5
MGMT_SLIDE_DAYS = 3

# Entry model config
ENTRY_BAR_SIZE_MIN = 30
MIN_EDGE_TICKS = 2.5
CONFIDENCE_PCT = 0.15  # top/bottom 15% for entry signals
MAX_HOLD_MINUTES = 30  # trade hold window in minutes

# LightGBM params — entry model (regression)
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

# LightGBM params — management classifier (binary)
LGBM_MGMT_PARAMS = {
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


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: BAR AGGREGATION + FEATURES (30-min entry model)
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
            "sweep_direction": float(
                np.sign(sv_arr[np.abs(grp["sv_zscore"].values).argmax()])
            ) if len(sv_arr) > 0 else 0.0,
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
    log.info(f"Aggregated {len(result):,} 30-min bars with {len(result.columns)} cols")
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


def add_forward_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward return labels for 30-min horizon."""
    df = df.sort_values("ts").reset_index(drop=True)

    fwd_close = df["close"].shift(-1)  # 1 bar ahead = 30min
    fwd_ticks = (fwd_close - df["close"]) / 0.25

    # Null out overnight gaps
    ts_now = df["ts"].values
    ts_fwd = df["ts"].shift(-1).values
    for i in range(len(df) - 1):
        if pd.isna(ts_fwd[i]):
            continue
        diff_s = (pd.Timestamp(ts_fwd[i]) - pd.Timestamp(ts_now[i])).total_seconds()
        if diff_s > 6 * 3600:
            fwd_ticks.iloc[i] = np.nan

    df["fwd_ticks_30min"] = fwd_ticks

    log.info(
        f"Forward labels: {(~fwd_ticks.isna()).sum():,} valid, "
        f"mean={fwd_ticks.mean():.2f}, std={fwd_ticks.std():.2f}"
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select feature columns, excluding labels, metadata, raw prices."""
    exclude_prefixes = (
        "fwd_", "direction_", "trade_quality_",
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
#  SECTION 3: LEAKAGE AUDIT
# ═══════════════════════════════════════════════════════════════════


def leakage_audit(
    train_dates: List[str],
    val_dates: List[str],
    feature_cols: List[str],
    context: str = "",
) -> bool:
    """Explicit leakage audit. Returns True if all checks pass."""
    passed = True

    overlap = set(train_dates) & set(val_dates)
    if overlap:
        log.error(f"LEAKAGE [{context}]: Train/val date overlap: {overlap}")
        passed = False

    max_train = max(train_dates)
    min_val = min(val_dates)
    if max_train >= min_val:
        log.error(f"LEAKAGE [{context}]: Max train {max_train} >= min val {min_val}")
        passed = False

    fwd_leak = [c for c in feature_cols if c.startswith(("fwd_", "direction_", "trade_quality_"))]
    if fwd_leak:
        log.error(f"LEAKAGE [{context}]: Forward-looking cols in features: {fwd_leak}")
        passed = False

    price_cols = [c for c in feature_cols if c in ("close", "high", "low", "open")]
    if price_cols:
        log.warning(f"WARNING [{context}]: Raw price columns in features: {price_cols}")

    if passed:
        log.info(f"Leakage audit PASSED [{context}]")
    return passed


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: ENTRY MODEL WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════


def run_entry_model_wf(
    bars_df: pd.DataFrame,
    feature_cols: List[str],
) -> pd.DataFrame:
    """
    Run the lean 30-min entry model through walk-forward.
    Returns bars_df with 'entry_pred' column populated for OOT bars.
    """
    _import_lightgbm()

    dates = sorted(bars_df["date"].unique())
    log.info(f"Entry model WF: {len(dates)} days ({dates[0]} -> {dates[-1]})")

    features_all = bars_df[feature_cols].values.astype(np.float32)
    labels_all = bars_df["fwd_ticks_30min"].values.astype(np.float32)
    dates_all = bars_df["date"].values

    bars_df["entry_pred"] = np.nan
    fold_idx = 0
    total_oot = 0

    start_idx = TRAIN_DAYS
    for fold_start in range(start_idx, len(dates) - VAL_DAYS + 1, SLIDE_DAYS):
        fold_train_dates = dates[fold_start - TRAIN_DAYS: fold_start]
        fold_val_dates = dates[fold_start: fold_start + VAL_DAYS]

        if len(fold_val_dates) < VAL_DAYS:
            break

        fold_idx += 1

        if not leakage_audit(
            list(fold_train_dates), list(fold_val_dates),
            feature_cols, context=f"entry fold {fold_idx}"
        ):
            continue

        train_mask = np.isin(dates_all, fold_train_dates)
        val_mask = np.isin(dates_all, fold_val_dates)

        X_train = features_all[train_mask].copy()
        X_val = features_all[val_mask].copy()
        y_train = labels_all[train_mask]
        y_val = labels_all[val_mask]

        # Robust scaling from train only
        train_median = np.nanmedian(X_train, axis=0)
        q75 = np.nanpercentile(X_train, 75, axis=0)
        q25 = np.nanpercentile(X_train, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0

        X_train = np.clip(np.nan_to_num((X_train - train_median) / iqr, nan=0.0), -5, 5)
        X_val = np.clip(np.nan_to_num((X_val - train_median) / iqr, nan=0.0), -5, 5)

        # Remove NaN targets
        train_valid = ~np.isnan(y_train)
        val_valid = ~np.isnan(y_val)

        if train_valid.sum() < 50 or val_valid.sum() < 10:
            log.warning(f"  Entry fold {fold_idx}: too few samples, skip")
            continue

        params = {**LGBM_ENTRY_PARAMS, "seed": 42 + fold_idx}
        train_data = lgb.Dataset(
            X_train[train_valid], label=y_train[train_valid],
            feature_name=feature_cols,
        )
        val_data = lgb.Dataset(
            X_val[val_valid], label=y_val[val_valid],
            feature_name=feature_cols,
            reference=train_data,
        )

        model = lgb.train(
            params, train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # Predict OOT
        preds = model.predict(X_val[val_valid], num_iteration=model.best_iteration)

        # IC check
        ic = np.corrcoef(preds, y_val[val_valid])[0, 1] if len(preds) > 5 else float("nan")

        # Store predictions back into bars_df
        val_indices = bars_df.index[val_mask]
        valid_indices = val_indices[val_valid]
        bars_df.loc[valid_indices, "entry_pred"] = preds

        n_oot = val_valid.sum()
        total_oot += n_oot
        log.info(
            f"  Entry fold {fold_idx}: IC={ic:.4f}, n_oot={n_oot}, "
            f"train={fold_train_dates[0]}..{fold_train_dates[-1]}, "
            f"val={fold_val_dates[0]}..{fold_val_dates[-1]}"
        )

        del model, train_data, val_data
        gc.collect()

    log.info(f"Entry model complete: {fold_idx} folds, {total_oot} OOT predictions")
    return bars_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: TRADE SAMPLE GENERATION
# ═══════════════════════════════════════════════════════════════════


def generate_trade_samples(
    bars_df: pd.DataFrame,
    minute_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each high-confidence entry signal, extract minute-level management
    features for the next 30 minutes. Returns a DataFrame with one row
    per (trade, minute) observation.
    """
    # Filter to bars with OOT predictions
    has_pred = bars_df[bars_df["entry_pred"].notna()].copy()
    if len(has_pred) == 0:
        raise RuntimeError("No entry predictions available")

    preds = has_pred["entry_pred"].values
    upper_thresh = np.quantile(preds, 1 - CONFIDENCE_PCT)
    lower_thresh = np.quantile(preds, CONFIDENCE_PCT)

    # Select high-confidence signals
    long_mask = has_pred["entry_pred"] >= upper_thresh
    short_mask = has_pred["entry_pred"] <= lower_thresh
    signals = has_pred[long_mask | short_mask].copy()
    signals["trade_dir"] = 0
    signals.loc[long_mask[long_mask].index.intersection(signals.index), "trade_dir"] = 1
    signals.loc[short_mask[short_mask].index.intersection(signals.index), "trade_dir"] = -1

    log.info(
        f"High-confidence signals: {len(signals)} "
        f"(long={int((signals['trade_dir'] == 1).sum())}, "
        f"short={int((signals['trade_dir'] == -1).sum())}), "
        f"thresholds: upper={upper_thresh:.2f}, lower={lower_thresh:.2f}"
    )

    # Build minute-level index for fast lookup
    minute_df = minute_df.copy()
    minute_df = minute_df.set_index("ts_minute").sort_index()

    all_obs = []
    trade_id = 0

    for idx, row in signals.iterrows():
        entry_ts = row["ts"]
        entry_price = row["close"]
        trade_direction = row["trade_dir"]
        entry_confidence = row["entry_pred"]
        entry_date = row["date"]

        # Get the regime features at entry (causal — from bars_df)
        regime_vol = row.get("regime_vol_ratio", np.nan)
        regime_ret = row.get("regime_ret_16bar", np.nan)
        entry_spread = row.get("spread_mean", np.nan)

        # Extract minute bars for the next MAX_HOLD_MINUTES after entry
        # The entry bar ends at entry_ts + 30min. The trade starts at the
        # END of the entry bar (we observe the bar, then enter).
        trade_start = entry_ts + pd.Timedelta(minutes=ENTRY_BAR_SIZE_MIN)
        trade_end = trade_start + pd.Timedelta(minutes=MAX_HOLD_MINUTES)

        # Get minute bars in [trade_start, trade_end)
        mask = (minute_df.index >= trade_start) & (minute_df.index < trade_end)
        trade_minutes = minute_df.loc[mask]

        if len(trade_minutes) < 5:
            continue  # Skip if not enough minute data (e.g. near close)

        # Check that these minutes are on the same date (no overnight gap)
        if trade_minutes.iloc[0].get("date", entry_date) != entry_date:
            # Could happen at day boundary; check if gap is reasonable
            first_ts = trade_minutes.index[0]
            gap_s = (first_ts - trade_start).total_seconds()
            if abs(gap_s) > 3600:
                continue

        # Use the close of the last available minute as the exit price
        final_price = trade_minutes["close"].iloc[-1]
        final_pnl_ticks = trade_direction * (final_price - entry_price) / 0.25

        trade_id += 1

        # Build per-minute observations
        cum_ofi = 0.0
        cum_volume = 0.0
        prices_seen = []
        ofi_history = []
        volume_history = []

        for t_idx, (t_ts, t_row) in enumerate(trade_minutes.iterrows()):
            minutes_since = (t_ts - trade_start).total_seconds() / 60.0
            current_price = t_row["close"]
            current_ofi = t_row["ofi_1min"]
            current_volume = t_row["volume"]
            current_spread = t_row["spread_mean"]
            current_sv = t_row["signed_volume"]
            current_tc = t_row["trade_count"]

            cum_ofi += current_ofi
            cum_volume += current_volume
            prices_seen.append(current_price)
            ofi_history.append(current_ofi)
            volume_history.append(current_volume)

            # Price move since entry (signed by direction)
            price_move_ticks = trade_direction * (current_price - entry_price) / 0.25

            # MFE and MAE
            price_moves = [trade_direction * (p - entry_price) / 0.25 for p in prices_seen]
            mfe = max(price_moves)
            mae = min(price_moves)

            # Time-normalized excursion rates
            time_factor = max(minutes_since, 1.0)
            mfe_ratio = mfe / time_factor
            mae_ratio = mae / time_factor

            # Recent OFI (last 5 min and last 1 min)
            ofi_recent_5m = sum(ofi_history[-5:])
            ofi_recent_1m = ofi_history[-1]

            # Volume trend since entry
            if len(volume_history) >= 2:
                vol_trend = _safe_polyfit_slope(np.array(volume_history))
            else:
                vol_trend = 0.0

            # Trade intensity recent (5 min window)
            recent_minutes = trade_minutes.iloc[max(0, t_idx - 4):t_idx + 1]
            trade_intensity_recent = recent_minutes["trade_count"].mean()

            # Signed volume direction match
            sv_recent = recent_minutes["signed_volume"].sum()
            sv_direction_match = float(np.sign(sv_recent) == trade_direction)

            # Remaining PnL if we hold to end (hindsight label)
            remaining_pnl_ticks = trade_direction * (final_price - current_price) / 0.25

            # Drawdown from peak
            drawdown_from_mfe = mfe - price_move_ticks

            # Price momentum / acceleration
            if len(prices_seen) >= 3:
                recent_prices = prices_seen[-3:]
                price_accel = (recent_prices[-1] - 2 * recent_prices[-2] + recent_prices[-3]) / 0.25
                price_accel *= trade_direction  # positive = accelerating in our favor
            else:
                price_accel = 0.0

            # OFI momentum (change in OFI flow)
            if len(ofi_history) >= 3:
                ofi_momentum = ofi_history[-1] - ofi_history[-3]
                ofi_momentum_aligned = ofi_momentum * trade_direction  # positive = flow in our favor
            else:
                ofi_momentum_aligned = 0.0

            # Fraction of hold time elapsed
            time_fraction = minutes_since / MAX_HOLD_MINUTES

            # Is the trade underwater? (negative unrealized P&L)
            is_underwater = float(price_move_ticks < 0)

            # Ratio of favorable to total excursion (how one-sided)
            total_excursion = abs(mfe) + abs(mae)
            favorable_ratio = abs(mfe) / max(total_excursion, 0.01)

            obs = {
                "trade_id": trade_id,
                "date": entry_date,
                "entry_ts": trade_start,
                "obs_ts": t_ts,
                "trade_dir": trade_direction,
                # Management features
                "minutes_since_entry": minutes_since,
                "ofi_since_entry": cum_ofi,
                "ofi_recent_5m": ofi_recent_5m,
                "ofi_recent_1m": ofi_recent_1m,
                "volume_since_entry": cum_volume,
                "volume_trend_since_entry": vol_trend,
                "price_move_since_entry": price_move_ticks,
                "max_favorable_excursion": mfe,
                "max_adverse_excursion": mae,
                "mfe_ratio": mfe_ratio,
                "mae_ratio": mae_ratio,
                "drawdown_from_mfe": drawdown_from_mfe,
                "spread_current": current_spread,
                "spread_vs_entry": current_spread - entry_spread if not np.isnan(entry_spread) else 0.0,
                "trade_intensity_recent": trade_intensity_recent,
                "sv_direction_match": sv_direction_match,
                "entry_confidence": abs(entry_confidence),  # magnitude (always positive)
                "regime_vol_at_entry": regime_vol if not np.isnan(regime_vol) else 1.0,
                "regime_ret_at_entry": regime_ret if not np.isnan(regime_ret) else 0.0,
                # Additional discriminative features
                "price_acceleration": price_accel,
                "ofi_momentum_aligned": ofi_momentum_aligned,
                "time_fraction": time_fraction,
                "is_underwater": is_underwater,
                "favorable_ratio": favorable_ratio,
                # Labels (hindsight)
                "remaining_pnl_ticks": remaining_pnl_ticks,
                "final_pnl_ticks": final_pnl_ticks,
                "should_exit": int(remaining_pnl_ticks < 0),  # EXIT=1 if remaining is negative
            }
            all_obs.append(obs)

    if not all_obs:
        raise RuntimeError("No trade observations generated")

    result = pd.DataFrame(all_obs)
    n_trades = result["trade_id"].nunique()
    n_obs = len(result)
    exit_rate = result["should_exit"].mean()

    log.info(
        f"Generated {n_obs:,} observations from {n_trades} trades "
        f"(avg {n_obs / max(n_trades, 1):.0f} obs/trade, exit_rate={exit_rate:.2%})"
    )
    return result


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: MANAGEMENT MODEL WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════

MGMT_FEATURE_COLS = [
    "minutes_since_entry",
    "ofi_since_entry",
    "ofi_recent_5m",
    "ofi_recent_1m",
    "volume_since_entry",
    "volume_trend_since_entry",
    "price_move_since_entry",
    "max_favorable_excursion",
    "max_adverse_excursion",
    "mfe_ratio",
    "mae_ratio",
    "drawdown_from_mfe",
    "spread_current",
    "spread_vs_entry",
    "trade_intensity_recent",
    "sv_direction_match",
    "entry_confidence",
    "regime_vol_at_entry",
    "regime_ret_at_entry",
    # v3.1 additional features
    "price_acceleration",
    "ofi_momentum_aligned",
    "time_fraction",
    "is_underwater",
    "favorable_ratio",
]


def run_management_model_wf(
    obs_df: pd.DataFrame,
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Walk-forward training of the management (exit) classifier.
    Returns obs_df with 'exit_prob' column and list of fold results.
    """
    _import_lightgbm()

    dates = sorted(obs_df["date"].unique())
    log.info(f"Management model WF: {len(dates)} dates with trade observations")

    obs_df["exit_prob"] = np.nan
    fold_results = []
    fold_idx = 0

    # Shorter WF for management model — simpler features, need more OOT coverage
    start_idx = MGMT_TRAIN_DAYS
    for fold_start in range(start_idx, len(dates) - MGMT_VAL_DAYS + 1, MGMT_SLIDE_DAYS):
        fold_train_dates = dates[fold_start - MGMT_TRAIN_DAYS: fold_start]
        fold_val_dates = dates[fold_start: fold_start + MGMT_VAL_DAYS]

        if len(fold_val_dates) < MGMT_VAL_DAYS:
            break

        fold_idx += 1

        if not leakage_audit(
            list(fold_train_dates), list(fold_val_dates),
            MGMT_FEATURE_COLS, context=f"mgmt fold {fold_idx}"
        ):
            continue

        train_mask = obs_df["date"].isin(fold_train_dates)
        val_mask = obs_df["date"].isin(fold_val_dates)

        train_data = obs_df[train_mask]
        val_data_df = obs_df[val_mask]

        if len(train_data) < 100 or len(val_data_df) < 20:
            log.warning(
                f"  Mgmt fold {fold_idx}: too few samples "
                f"(train={len(train_data)}, val={len(val_data_df)}) -- skip"
            )
            continue

        X_train = train_data[MGMT_FEATURE_COLS].values.astype(np.float32)
        y_train = train_data["should_exit"].values.astype(np.float32)
        X_val = val_data_df[MGMT_FEATURE_COLS].values.astype(np.float32)
        y_val = val_data_df["should_exit"].values.astype(np.float32)

        # Robust scaling from train only
        train_median = np.nanmedian(X_train, axis=0)
        q75 = np.nanpercentile(X_train, 75, axis=0)
        q25 = np.nanpercentile(X_train, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0

        X_train = np.clip(np.nan_to_num((X_train - train_median) / iqr, nan=0.0), -5, 5)
        X_val = np.clip(np.nan_to_num((X_val - train_median) / iqr, nan=0.0), -5, 5)

        params = {**LGBM_MGMT_PARAMS, "seed": 42 + fold_idx}
        lgb_train = lgb.Dataset(
            X_train, label=y_train, feature_name=MGMT_FEATURE_COLS,
        )
        lgb_val = lgb.Dataset(
            X_val, label=y_val, feature_name=MGMT_FEATURE_COLS,
            reference=lgb_train,
        )

        model = lgb.train(
            params, lgb_train,
            num_boost_round=500,
            valid_sets=[lgb_val],
            callbacks=[
                lgb.early_stopping(stopping_rounds=30, verbose=False),
                lgb.log_evaluation(period=0),
            ],
        )

        # Predict exit probabilities
        preds = model.predict(X_val, num_iteration=model.best_iteration)
        obs_df.loc[val_data_df.index, "exit_prob"] = preds

        # AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_val, preds)
        except Exception:
            auc = float("nan")

        # Feature importance
        importance = model.feature_importance(importance_type="gain")
        feat_imp = sorted(
            zip(MGMT_FEATURE_COLS, importance), key=lambda x: x[1], reverse=True
        )[:10]

        fold_result = {
            "fold": fold_idx,
            "train_dates": f"{fold_train_dates[0]}..{fold_train_dates[-1]}",
            "val_dates": f"{fold_val_dates[0]}..{fold_val_dates[-1]}",
            "train_samples": len(train_data),
            "val_samples": len(val_data_df),
            "auc": float(auc),
            "val_exit_rate": float(y_val.mean()),
            "pred_mean": float(preds.mean()),
            "best_iter": model.best_iteration,
            "top_features": [{"name": n, "gain": float(g)} for n, g in feat_imp],
        }
        fold_results.append(fold_result)

        log.info(
            f"  Mgmt fold {fold_idx}: AUC={auc:.4f}, "
            f"n_train={len(train_data)}, n_val={len(val_data_df)}, "
            f"exit_rate={y_val.mean():.2%}"
        )

        del model, lgb_train, lgb_val
        gc.collect()

    log.info(
        f"Management model complete: {fold_idx} folds, "
        f"{obs_df['exit_prob'].notna().sum()} OOT predictions"
    )
    return obs_df, fold_results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 7: STRATEGY SIMULATION (STATIC vs DYNAMIC)
# ═══════════════════════════════════════════════════════════════════


def simulate_strategies(
    obs_df: pd.DataFrame,
    exit_threshold: float = 0.50,
) -> Dict[str, Any]:
    """
    Compare static 30-min hold vs dynamic management model exits.

    For dynamic: at each minute, if exit_prob > exit_threshold, exit immediately.
    PnL = price_move_since_entry - cost.
    For static: hold to end. PnL = final_pnl_ticks - cost.
    """
    # Only use observations with OOT exit predictions
    has_pred = obs_df[obs_df["exit_prob"].notna()].copy()
    if len(has_pred) == 0:
        return {"error": "no predictions available"}

    trade_ids = has_pred["trade_id"].unique()
    log.info(f"Simulating {len(trade_ids)} trades with management predictions")

    static_pnls = []
    dynamic_pnls = []
    dynamic_hold_times = []
    static_hold_times = []
    trade_details = []

    for tid in trade_ids:
        trade_obs = has_pred[has_pred["trade_id"] == tid].sort_values("minutes_since_entry")

        if len(trade_obs) == 0:
            continue

        final_pnl = trade_obs.iloc[-1]["final_pnl_ticks"]
        static_pnl = final_pnl - COST_RT_TICKS
        static_hold = trade_obs.iloc[-1]["minutes_since_entry"]

        static_pnls.append(static_pnl)
        static_hold_times.append(static_hold)

        # Dynamic: find first minute where exit_prob > threshold
        dynamic_pnl = static_pnl  # default: hold to end
        dynamic_hold = static_hold
        exited_early = False

        for _, obs in trade_obs.iterrows():
            if obs["exit_prob"] > exit_threshold:
                dynamic_pnl = obs["price_move_since_entry"] - COST_RT_TICKS
                dynamic_hold = obs["minutes_since_entry"]
                exited_early = True
                break

        dynamic_pnls.append(dynamic_pnl)
        dynamic_hold_times.append(dynamic_hold)

        trade_details.append({
            "trade_id": int(tid),
            "date": trade_obs.iloc[0]["date"],
            "direction": int(trade_obs.iloc[0]["trade_dir"]),
            "static_pnl": float(static_pnl),
            "dynamic_pnl": float(dynamic_pnl),
            "static_hold_min": float(static_hold),
            "dynamic_hold_min": float(dynamic_hold),
            "exited_early": exited_early,
        })

    static_arr = np.array(static_pnls)
    dynamic_arr = np.array(dynamic_pnls)

    def _metrics(pnl_arr, hold_arr, label):
        if len(pnl_arr) < 3:
            return {"error": f"too few trades for {label}"}
        sharpe = pnl_arr.mean() / max(pnl_arr.std(), 1e-6) * np.sqrt(252)
        downside = np.sqrt(np.mean(np.minimum(pnl_arr, 0) ** 2))
        sortino = pnl_arr.mean() / max(downside, 1e-6) * np.sqrt(252)
        wr = np.mean(pnl_arr > 0)
        winners = pnl_arr[pnl_arr > 0]
        losers = pnl_arr[pnl_arr < 0]
        pf = np.sum(winners) / max(-np.sum(losers), 1e-6)
        cum = np.cumsum(pnl_arr)
        max_dd = float(np.min(cum - np.maximum.accumulate(cum)))

        # Per-day analysis
        trade_df = pd.DataFrame(trade_details)
        day_pnl = trade_df.groupby("date")[f"{label.lower()}_pnl"].sum()

        return {
            "n_trades": len(pnl_arr),
            "total_pnl_ticks": float(pnl_arr.sum()),
            "total_pnl_dollars": float(pnl_arr.sum() * ES_TICK_VALUE),
            "avg_pnl_ticks": float(pnl_arr.mean()),
            "sharpe": float(sharpe),
            "sortino": float(sortino),
            "win_rate": float(wr),
            "profit_factor": float(pf),
            "max_dd_ticks": float(max_dd),
            "avg_hold_min": float(np.mean(hold_arr)),
            "n_trading_days": int(day_pnl.index.nunique()) if len(day_pnl) > 0 else 0,
            "daily_sharpe": float(
                day_pnl.mean() / max(day_pnl.std(), 1e-6) * np.sqrt(252)
            ) if len(day_pnl) > 2 else 0,
        }

    static_metrics = _metrics(static_arr, np.array(static_hold_times), "Static")
    dynamic_metrics = _metrics(dynamic_arr, np.array(dynamic_hold_times), "Dynamic")

    # Early exit statistics
    early_exits = [t for t in trade_details if t["exited_early"]]
    early_pct = len(early_exits) / max(len(trade_details), 1)

    # Improvement
    sharpe_improvement = 0
    if "sharpe" in static_metrics and "sharpe" in dynamic_metrics:
        if static_metrics["sharpe"] != 0:
            sharpe_improvement = (
                (dynamic_metrics["sharpe"] - static_metrics["sharpe"])
                / abs(static_metrics["sharpe"])
            )

    # Regime stratification
    trade_df = pd.DataFrame(trade_details)
    regime_results = {}
    if len(trade_df) > 0:
        # Get daily returns for regime classification
        dates_in_trades = trade_df["date"].unique()
        for regime_label, pnl_col in [("static", "static_pnl"), ("dynamic", "dynamic_pnl")]:
            long_trades = trade_df[trade_df["direction"] == 1]
            short_trades = trade_df[trade_df["direction"] == -1]
            if len(long_trades) > 0 and len(short_trades) > 0:
                long_sharpe = (
                    long_trades[pnl_col].mean()
                    / max(long_trades[pnl_col].std(), 1e-6)
                    * np.sqrt(252)
                )
                short_sharpe = (
                    short_trades[pnl_col].mean()
                    / max(short_trades[pnl_col].std(), 1e-6)
                    * np.sqrt(252)
                )
                denom = max(abs(long_sharpe), abs(short_sharpe), 1e-6)
                side_gap = abs(long_sharpe - short_sharpe) / denom
                regime_results[regime_label] = {
                    "long_sharpe": float(long_sharpe),
                    "short_sharpe": float(short_sharpe),
                    "side_gap": float(side_gap),
                    "long_trades": len(long_trades),
                    "short_trades": len(short_trades),
                }

    return {
        "static": static_metrics,
        "dynamic": dynamic_metrics,
        "sharpe_improvement_pct": float(sharpe_improvement * 100),
        "early_exit_rate": float(early_pct),
        "n_early_exits": len(early_exits),
        "regime_by_side": regime_results,
        "trade_details": trade_details,
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 8: THRESHOLD SWEEP
# ═══════════════════════════════════════════════════════════════════


def sweep_exit_thresholds(obs_df: pd.DataFrame) -> List[Dict]:
    """Sweep exit probability thresholds to find optimal."""
    thresholds = [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    results = []

    for thresh in thresholds:
        sim = simulate_strategies(obs_df, exit_threshold=thresh)
        if "error" in sim:
            continue

        results.append({
            "threshold": thresh,
            "dynamic_sharpe": sim["dynamic"].get("sharpe", 0),
            "dynamic_sortino": sim["dynamic"].get("sortino", 0),
            "dynamic_wr": sim["dynamic"].get("win_rate", 0),
            "dynamic_pf": sim["dynamic"].get("profit_factor", 0),
            "dynamic_avg_hold": sim["dynamic"].get("avg_hold_min", 0),
            "early_exit_rate": sim["early_exit_rate"],
            "static_sharpe": sim["static"].get("sharpe", 0),
            "sharpe_improvement_pct": sim["sharpe_improvement_pct"],
        })

        log.info(
            f"  Threshold {thresh:.2f}: "
            f"dynamic_sharpe={sim['dynamic'].get('sharpe', 0):.2f}, "
            f"static_sharpe={sim['static'].get('sharpe', 0):.2f}, "
            f"improvement={sim['sharpe_improvement_pct']:.1f}%, "
            f"early_exits={sim['early_exit_rate']:.1%}, "
            f"avg_hold={sim['dynamic'].get('avg_hold_min', 0):.1f}min"
        )

    return results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 9: MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("TRADE MANAGEMENT MODEL v3 — Minute-Bar Dynamic Exit")
    log.info("=" * 70)

    # ── Phase 1: Load data and build entry model ──
    log.info("\n" + "─" * 50)
    log.info("PHASE 1: Load minute bars + train entry model")
    log.info("─" * 50)

    minute_df = load_minute_bars()
    bars_df = aggregate_to_30min_bars(minute_df)
    bars_df = add_rolling_features(bars_df)
    bars_df = add_forward_labels(bars_df)

    feature_cols = get_feature_columns(bars_df)
    log.info(f"Entry model features ({len(feature_cols)}): {feature_cols[:10]}...")

    bars_df = run_entry_model_wf(bars_df, feature_cols)

    n_with_pred = bars_df["entry_pred"].notna().sum()
    log.info(f"Entry model produced {n_with_pred} OOT predictions")

    if n_with_pred < 50:
        log.error("Too few entry predictions to proceed")
        return

    # ── Phase 2: Generate trade samples with minute-level features ──
    log.info("\n" + "─" * 50)
    log.info("PHASE 2: Generate trade samples with per-minute features")
    log.info("─" * 50)

    obs_df = generate_trade_samples(bars_df, minute_df)

    # Free memory — minute_df is large
    del minute_df
    gc.collect()

    # ── Phase 3: Train management model ──
    log.info("\n" + "─" * 50)
    log.info("PHASE 3: Train exit management classifier (LightGBM binary)")
    log.info("─" * 50)

    obs_df, mgmt_fold_results = run_management_model_wf(obs_df)

    # Report aggregate AUC
    aucs = [f["auc"] for f in mgmt_fold_results if not np.isnan(f.get("auc", float("nan")))]
    if aucs:
        log.info(f"Management model aggregate AUC: {np.mean(aucs):.4f} (±{np.std(aucs):.4f})")

    # ── Phase 4: Simulate and compare strategies ──
    log.info("\n" + "─" * 50)
    log.info("PHASE 4: Simulate static vs dynamic exit strategies")
    log.info("─" * 50)

    # Best threshold sweep
    log.info("Sweeping exit thresholds...")
    threshold_results = sweep_exit_thresholds(obs_df)

    if threshold_results:
        best = max(threshold_results, key=lambda x: x["dynamic_sharpe"])
        log.info(f"\nBest threshold: {best['threshold']:.2f}")
        log.info(f"  Dynamic Sharpe: {best['dynamic_sharpe']:.2f}")
        log.info(f"  Static Sharpe:  {best['static_sharpe']:.2f}")
        log.info(f"  Improvement:    {best['sharpe_improvement_pct']:.1f}%")
        log.info(f"  Early exit rate: {best['early_exit_rate']:.1%}")
        log.info(f"  Avg hold time:  {best['dynamic_avg_hold']:.1f} min")

        # Run final simulation at best threshold for detailed results
        best_sim = simulate_strategies(obs_df, exit_threshold=best["threshold"])
    else:
        best = None
        best_sim = simulate_strategies(obs_df, exit_threshold=0.50)

    # ── Save results ──
    elapsed = time.time() - t0

    results = {
        "run_timestamp": datetime.utcnow().isoformat(),
        "elapsed_seconds": elapsed,
        "config": {
            "train_days": TRAIN_DAYS,
            "val_days": VAL_DAYS,
            "slide_days": SLIDE_DAYS,
            "confidence_pct": CONFIDENCE_PCT,
            "max_hold_minutes": MAX_HOLD_MINUTES,
            "cost_rt_ticks": COST_RT_TICKS,
        },
        "entry_model": {
            "n_oot_predictions": int(n_with_pred),
            "feature_count": len(feature_cols),
        },
        "management_model": {
            "n_features": len(MGMT_FEATURE_COLS),
            "feature_names": MGMT_FEATURE_COLS,
            "fold_results": mgmt_fold_results,
            "aggregate_auc": float(np.mean(aucs)) if aucs else None,
        },
        "threshold_sweep": threshold_results,
        "best_threshold": best,
        "final_simulation": {
            "static": best_sim.get("static", {}),
            "dynamic": best_sim.get("dynamic", {}),
            "sharpe_improvement_pct": best_sim.get("sharpe_improvement_pct", 0),
            "early_exit_rate": best_sim.get("early_exit_rate", 0),
            "regime_by_side": best_sim.get("regime_by_side", {}),
        },
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, "w") as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"Results saved to {results_path}")

    # Save trade details
    if "trade_details" in best_sim:
        trade_df = pd.DataFrame(best_sim["trade_details"])
        trade_path = OUTPUT_DIR / "trade_details.csv"
        trade_df.to_csv(trade_path, index=False)
        log.info(f"Trade details saved to {trade_path}")

    # Save observation data for analysis
    obs_path = OUTPUT_DIR / "observations.parquet"
    obs_df.to_parquet(obs_path, index=False)
    log.info(f"Observations saved to {obs_path}")

    # ── Final Summary ──
    log.info("\n" + "=" * 70)
    log.info("TRADE MANAGEMENT v3 — FINAL SUMMARY")
    log.info("=" * 70)

    static = best_sim.get("static", {})
    dynamic = best_sim.get("dynamic", {})

    log.info(f"  Trades: {static.get('n_trades', 0)}")
    log.info(f"  Trading days: {static.get('n_trading_days', 0)}")
    log.info("")
    log.info(f"  {'Metric':<20} {'Static 30min':>15} {'Dynamic Mgmt':>15}")
    log.info(f"  {'─' * 50}")
    log.info(f"  {'Sharpe':<20} {static.get('sharpe', 0):>15.2f} {dynamic.get('sharpe', 0):>15.2f}")
    log.info(f"  {'Sortino':<20} {static.get('sortino', 0):>15.2f} {dynamic.get('sortino', 0):>15.2f}")
    log.info(f"  {'Win Rate':<20} {static.get('win_rate', 0):>14.1%} {dynamic.get('win_rate', 0):>14.1%}")
    log.info(f"  {'Profit Factor':<20} {static.get('profit_factor', 0):>15.2f} {dynamic.get('profit_factor', 0):>15.2f}")
    log.info(f"  {'Avg PnL (ticks)':<20} {static.get('avg_pnl_ticks', 0):>15.2f} {dynamic.get('avg_pnl_ticks', 0):>15.2f}")
    log.info(f"  {'Total PnL ($)':<20} {static.get('total_pnl_dollars', 0):>15.0f} {dynamic.get('total_pnl_dollars', 0):>15.0f}")
    log.info(f"  {'Max DD (ticks)':<20} {static.get('max_dd_ticks', 0):>15.1f} {dynamic.get('max_dd_ticks', 0):>15.1f}")
    log.info(f"  {'Avg Hold (min)':<20} {static.get('avg_hold_min', 0):>15.1f} {dynamic.get('avg_hold_min', 0):>15.1f}")
    log.info("")
    log.info(f"  Sharpe improvement: {best_sim.get('sharpe_improvement_pct', 0):.1f}%")
    log.info(f"  Early exit rate: {best_sim.get('early_exit_rate', 0):.1%}")
    log.info(f"  Best exit threshold: {best['threshold'] if best else 'N/A'}")
    log.info(f"  Elapsed: {elapsed:.0f}s ({elapsed / 60:.1f}min)")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
