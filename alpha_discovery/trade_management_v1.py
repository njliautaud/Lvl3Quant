#!/usr/bin/env python3
"""
Trade Management Model v1 — Dynamic Exit via Trade Outcome Prediction
======================================================================

We have a strong 30-minute ENTRY model (LightGBM v4, Sharpe 2.55, regime-agnostic).
But it uses a static 30-minute time exit.

This model learns WHEN to exit by continuously monitoring the trade after entry:
  - Invalidation classifier: P(trade_will_lose | current state) → EXIT when > threshold
  - Remaining MFE regressor: expected favorable move remaining → EXIT when < cost

Architecture:
  1. Reconstruct entry signals from the v4 entry model (retrained inline)
  2. For each trade, sample minute-level features every 1 min for up to 45 min
  3. At each sample: compute post-entry features + real-time orderflow + market state
  4. Label: binary (will trade be profitable at close?) + remaining MFE
  5. Train LightGBM classifier + regressor with walk-forward (60d/10d/5d slide)
  6. Simulate: compare static 30-min exit vs dynamic exit

Walk-Forward: 60d train, 10d val, 5d slide — SLIDING only (HC #0).
Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission).

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python \\
      alpha_discovery/trade_management_v1.py --device cuda

Author: Claude (autonomous research)
"""

import argparse
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
OUTPUT_DIR = ROOT / "output" / "trade_management_v1"
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
    format="%(asctime)s [TM-v1] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "trade_management_v1.log")),
    ],
)
log = logging.getLogger("TM-v1")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376
COST_RT_TICKS = 2.376  # market entry + market exit + RT commission

# Walk-forward config
TRAIN_DAYS = 60
VAL_DAYS = 10
SLIDE_DAYS = 5

# Entry model config (30-min bars, both-sides)
ENTRY_BAR_SIZE_MIN = 30
ENTRY_HORIZON_BARS = 1  # 30min = 1 bar
ENTRY_HORIZON_LABEL = "30min"
MIN_EDGE_TICKS = 2.5

# Trade management config
MAX_HOLD_MINUTES = 45
SAMPLE_INTERVAL_MINUTES = 1
INVALIDATION_THRESHOLD = 0.70  # P(loser) > this → exit
REMAINING_MFE_THRESHOLD = 1.0  # expected remaining MFE < this (ticks) → exit

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

# LightGBM params — management classifier (invalidation)
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

# LightGBM params — management regressor (remaining MFE)
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


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: BAR AGGREGATION + ENTRY FEATURES (from v4)
# ═══════════════════════════════════════════════════════════════════


def _safe_polyfit_slope(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def aggregate_to_bars(minute_df: pd.DataFrame, bar_size_min: int = 30) -> pd.DataFrame:
    """Aggregate 1-minute bars into N-minute bars with microstructure features."""
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
    log.info(f"Aggregated {len(result):,} {bar_size_min}min bars with {len(result.columns)} cols")
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


def add_forward_labels(df: pd.DataFrame, horizon_bars: int, horizon_label: str) -> pd.DataFrame:
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

    df[f"fwd_ticks_{horizon_label}"] = fwd_ticks
    df[f"direction_{horizon_label}"] = 0
    df.loc[fwd_ticks > MIN_EDGE_TICKS, f"direction_{horizon_label}"] = 1
    df.loc[fwd_ticks < -MIN_EDGE_TICKS, f"direction_{horizon_label}"] = -1

    return df


def get_entry_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select feature columns for the entry model."""
    exclude_prefixes = ("fwd_", "direction_", "trade_quality_", "date", "ts", "bar_key")
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
#  SECTION 3: ENTRY MODEL TRAINING (Walk-Forward, reproducing v4)
# ═══════════════════════════════════════════════════════════════════


def train_entry_model_wf(bars_df: pd.DataFrame) -> pd.DataFrame:
    """
    Walk-forward train the entry model on 30-min bars.
    Returns bars_df with 'entry_pred_ticks' and 'entry_signal' columns.
    """
    _import_lightgbm()

    target_col = f"fwd_ticks_{ENTRY_HORIZON_LABEL}"
    feature_cols = get_entry_feature_columns(bars_df)
    log.info(f"Entry model: {len(feature_cols)} features, target={target_col}")

    dates = sorted(bars_df["date"].unique())
    log.info(f"Entry model WF: {len(dates)} unique dates")

    all_preds = pd.Series(np.nan, index=bars_df.index, dtype=np.float64)

    fold_idx = 0
    start = 0
    while start + TRAIN_DAYS + VAL_DAYS <= len(dates):
        train_dates = dates[start:start + TRAIN_DAYS]
        val_dates = dates[start + TRAIN_DAYS:start + TRAIN_DAYS + VAL_DAYS]

        train_mask = bars_df["date"].isin(train_dates)
        val_mask = bars_df["date"].isin(val_dates)

        X_train = bars_df.loc[train_mask, feature_cols].values.astype(np.float32)
        y_train = bars_df.loc[train_mask, target_col].values.astype(np.float32)
        X_val = bars_df.loc[val_mask, feature_cols].values.astype(np.float32)

        # Remove NaN targets from train
        valid_train = ~np.isnan(y_train)
        X_train = X_train[valid_train]
        y_train = y_train[valid_train]

        if len(X_train) < 100 or len(X_val) < 10:
            start += SLIDE_DAYS
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
                     f"val={val_dates[0]}..{val_dates[-1]}, n_train={len(X_train)}, n_val={len(X_val)}")

        start += SLIDE_DAYS
        del model, dtrain
        gc.collect()

    bars_df["entry_pred_ticks"] = all_preds
    valid_preds = all_preds.dropna()
    log.info(f"Entry model: {len(valid_preds)} OOT predictions across {fold_idx} folds")

    # Generate entry signals: top/bottom quantile predictions
    # Use percentile thresholds to get ~reasonable trade count
    if len(valid_preds) > 0:
        p80 = np.nanpercentile(valid_preds, 80)
        p20 = np.nanpercentile(valid_preds, 20)
        bars_df["entry_signal"] = 0
        bars_df.loc[bars_df["entry_pred_ticks"] > max(p80, MIN_EDGE_TICKS), "entry_signal"] = 1   # long
        bars_df.loc[bars_df["entry_pred_ticks"] < min(p20, -MIN_EDGE_TICKS), "entry_signal"] = -1  # short

        n_long = (bars_df["entry_signal"] == 1).sum()
        n_short = (bars_df["entry_signal"] == -1).sum()
        log.info(f"Entry signals: {n_long} long, {n_short} short (thresholds: long>{p80:.1f}, short<{p20:.1f})")
    else:
        bars_df["entry_signal"] = 0

    return bars_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: TRADE-LEVEL DATA CONSTRUCTION
# ═══════════════════════════════════════════════════════════════════


def construct_trade_samples(
    bars_df: pd.DataFrame,
    minute_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    For each entry signal, construct per-minute samples for trade management.

    For each trade:
      - Get entry time, direction, entry price, entry confidence
      - Sample minute bars for up to MAX_HOLD_MINUTES after entry
      - At each minute: compute post-entry features + orderflow + market state
      - Label: will this trade be profitable at its eventual close (30 min)?
              + what's the remaining MFE from this point?
    """
    # Get trades (rows with entry signals)
    trades = bars_df[bars_df["entry_signal"] != 0].copy()
    trades = trades.dropna(subset=["entry_pred_ticks", f"fwd_ticks_{ENTRY_HORIZON_LABEL}"])

    if len(trades) == 0:
        log.warning("No valid trade entries found!")
        return pd.DataFrame()

    log.info(f"Constructing trade samples for {len(trades)} entries...")

    # Build minute-level lookup indexed by timestamp
    min_df = minute_df.copy()
    min_df = min_df.sort_values("ts_minute").reset_index(drop=True)

    # Create a minute-level time index for fast lookup
    min_df["ts_idx"] = min_df["ts_minute"]
    min_df = min_df.set_index("ts_idx")

    all_samples = []
    trade_meta = []  # Store trade-level info for later simulation

    for trade_idx, (idx, trade_row) in enumerate(trades.iterrows()):
        entry_time = trade_row["ts"]
        entry_date = trade_row["date"]
        direction = int(trade_row["entry_signal"])  # +1 or -1
        entry_price = trade_row["close"]  # entry at bar close
        entry_confidence = abs(trade_row["entry_pred_ticks"])
        actual_outcome_ticks = trade_row[f"fwd_ticks_{ENTRY_HORIZON_LABEL}"]

        # Signed outcome: positive = trade was correct direction
        signed_outcome = actual_outcome_ticks * direction
        is_winner = 1 if signed_outcome > COST_RT_TICKS else 0

        # Get minute bars from entry_time to entry_time + MAX_HOLD_MINUTES
        try:
            entry_ts = pd.Timestamp(entry_time)
            end_ts = entry_ts + pd.Timedelta(minutes=MAX_HOLD_MINUTES)
        except Exception:
            continue

        # Select minute bars in this window (same date to avoid overnight)
        mask = (
            (minute_df["ts_minute"] >= entry_ts)
            & (minute_df["ts_minute"] <= end_ts)
            & (minute_df["date"] == entry_date)
        )
        window_minutes = minute_df.loc[mask].sort_values("ts_minute")

        if len(window_minutes) < 5:
            continue

        # Track running MFE/MAE from entry
        entry_close = entry_price
        running_mfe = 0.0
        running_mae = 0.0
        cumulative_ofi = 0.0
        cumulative_sv = 0.0
        sweep_count = 0
        entry_spread = window_minutes["spread_mean"].iloc[0]
        entry_volume_rate = window_minutes["volume"].iloc[0]

        # Compute full-horizon MFE for labeling: best signed move within entire window
        prices = window_minutes["close"].values
        if direction == 1:
            signed_moves = (prices - entry_close) / 0.25  # ticks
        else:
            signed_moves = (entry_close - prices) / 0.25

        full_mfe = np.max(signed_moves) if len(signed_moves) > 0 else 0

        for min_idx, (_, min_row) in enumerate(window_minutes.iterrows()):
            if min_idx == 0:
                continue  # skip entry minute itself

            current_price = min_row["close"]
            time_in_trade = min_idx  # minutes

            # Signed PnL in ticks from entry
            if direction == 1:
                unrealized_pnl_ticks = (current_price - entry_close) / 0.25
            else:
                unrealized_pnl_ticks = (entry_close - current_price) / 0.25

            # Update running MFE/MAE
            running_mfe = max(running_mfe, unrealized_pnl_ticks)
            running_mae = min(running_mae, unrealized_pnl_ticks)

            # Cumulative orderflow since entry
            cumulative_ofi += min_row["ofi_1min"]
            cumulative_sv += min_row["signed_volume"]

            # Sweep detection (|sv_zscore| > 2 proxy)
            sv_std_proxy = max(abs(min_row["signed_volume"]), 1)
            if abs(min_row["signed_volume"]) > 2 * entry_volume_rate:
                sweep_count += 1

            # OFI direction alignment: is orderflow going WITH our trade?
            ofi_direction_alignment = cumulative_ofi * direction

            # Remaining MFE: from THIS point forward, what's the best move still available?
            remaining_prices = window_minutes["close"].values[min_idx:]
            if direction == 1:
                remaining_signed = (remaining_prices - current_price) / 0.25
            else:
                remaining_signed = (current_price - remaining_prices) / 0.25
            remaining_mfe = np.max(remaining_signed) if len(remaining_signed) > 0 else 0

            # Build sample features
            sample = {
                # Trade identity
                "trade_idx": trade_idx,
                "entry_time": entry_ts,
                "sample_time": min_row["ts_minute"],
                "direction": direction,
                "date": entry_date,

                # ── Post-entry features ──
                "time_in_trade_minutes": time_in_trade,
                "unrealized_pnl_ticks": unrealized_pnl_ticks,
                "max_favorable_ticks": running_mfe,
                "max_adverse_ticks": running_mae,
                "pnl_vs_mfe": unrealized_pnl_ticks / max(running_mfe, 0.25),  # How much of peak given back
                "mfe_vs_time": running_mfe / max(time_in_trade, 1),  # Rate of favorable movement
                "mae_vs_time": running_mae / max(time_in_trade, 1),  # Rate of adverse movement
                "drawdown_from_peak": running_mfe - unrealized_pnl_ticks,  # Ticks given back from peak

                # ── Real-time orderflow features ──
                "ofi_since_entry": cumulative_ofi,
                "ofi_direction_alignment": ofi_direction_alignment,
                "ofi_current_minute": min_row["ofi_1min"],
                "ofi_alignment_current": min_row["ofi_1min"] * direction,
                "signed_vol_since_entry": cumulative_sv,
                "signed_vol_alignment": cumulative_sv * direction,
                "volume_rate_current": min_row["volume"],
                "volume_rate_vs_entry": min_row["volume"] / max(entry_volume_rate, 1),
                "sweep_count_since_entry": sweep_count,

                # ── Market state features ──
                "spread_now": min_row["spread_mean"],
                "spread_vs_entry": min_row["spread_mean"] - entry_spread,
                "trade_count_current": min_row["trade_count"],

                # ── Entry context ──
                "entry_confidence": entry_confidence,
                "entry_pred_ticks": trade_row["entry_pred_ticks"],

                # ── Labels (only for training, not used as features!) ──
                "label_is_winner": is_winner,
                "label_is_loser": 1 - is_winner,
                "label_remaining_mfe": remaining_mfe,
                "label_final_pnl_ticks": signed_outcome,
            }

            all_samples.append(sample)

        # Trade metadata
        trade_meta.append({
            "trade_idx": trade_idx,
            "entry_time": entry_ts,
            "direction": direction,
            "entry_price": entry_price,
            "entry_confidence": entry_confidence,
            "actual_outcome_ticks": signed_outcome,
            "is_winner": is_winner,
            "full_mfe": full_mfe,
            "date": entry_date,
        })

        if (trade_idx + 1) % 200 == 0:
            log.info(f"  Processed {trade_idx + 1}/{len(trades)} trades, "
                     f"{len(all_samples)} samples so far")

    samples_df = pd.DataFrame(all_samples)
    meta_df = pd.DataFrame(trade_meta)

    log.info(f"Trade samples: {len(samples_df)} total from {len(meta_df)} trades "
             f"(avg {len(samples_df)/max(len(meta_df),1):.1f} samples/trade)")
    log.info(f"  Winners: {meta_df['is_winner'].sum()}/{len(meta_df)} "
             f"({meta_df['is_winner'].mean()*100:.1f}%)")

    return samples_df, meta_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: MANAGEMENT FEATURES
# ═══════════════════════════════════════════════════════════════════


def get_management_feature_columns() -> List[str]:
    """Return the feature columns used by the management models."""
    return [
        "time_in_trade_minutes",
        "unrealized_pnl_ticks",
        "max_favorable_ticks",
        "max_adverse_ticks",
        "pnl_vs_mfe",
        "mfe_vs_time",
        "mae_vs_time",
        "drawdown_from_peak",
        "ofi_since_entry",
        "ofi_direction_alignment",
        "ofi_current_minute",
        "ofi_alignment_current",
        "signed_vol_since_entry",
        "signed_vol_alignment",
        "volume_rate_current",
        "volume_rate_vs_entry",
        "sweep_count_since_entry",
        "spread_now",
        "spread_vs_entry",
        "trade_count_current",
        "entry_confidence",
        "entry_pred_ticks",
    ]


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: MANAGEMENT MODEL TRAINING (Walk-Forward)
# ═══════════════════════════════════════════════════════════════════


def train_management_models_wf(
    samples_df: pd.DataFrame,
    meta_df: pd.DataFrame,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Walk-forward train the management models:
      1. Invalidation classifier: P(loser)
      2. Remaining MFE regressor

    Returns fold results and samples with predictions.
    """
    _import_lightgbm()

    feature_cols = get_management_feature_columns()
    dates = sorted(samples_df["date"].unique())
    log.info(f"Management model WF: {len(dates)} unique dates, {len(feature_cols)} features")

    all_clf_preds = pd.Series(np.nan, index=samples_df.index, dtype=np.float64)
    all_reg_preds = pd.Series(np.nan, index=samples_df.index, dtype=np.float64)

    fold_results = []
    fold_idx = 0
    start = 0

    while start + TRAIN_DAYS + VAL_DAYS <= len(dates):
        train_dates = dates[start:start + TRAIN_DAYS]
        val_dates = dates[start + TRAIN_DAYS:start + TRAIN_DAYS + VAL_DAYS]

        train_mask = samples_df["date"].isin(train_dates)
        val_mask = samples_df["date"].isin(val_dates)

        X_train = samples_df.loc[train_mask, feature_cols].values.astype(np.float32)
        y_train_clf = samples_df.loc[train_mask, "label_is_loser"].values.astype(np.float32)
        y_train_reg = samples_df.loc[train_mask, "label_remaining_mfe"].values.astype(np.float32)
        X_val = samples_df.loc[val_mask, feature_cols].values.astype(np.float32)
        y_val_clf = samples_df.loc[val_mask, "label_is_loser"].values.astype(np.float32)
        y_val_reg = samples_df.loc[val_mask, "label_remaining_mfe"].values.astype(np.float32)

        # Remove NaN
        valid_train = ~(np.isnan(y_train_clf) | np.isnan(y_train_reg) | np.isnan(X_train).any(axis=1))
        valid_val = ~(np.isnan(X_val).any(axis=1))
        X_train = X_train[valid_train]
        y_train_clf = y_train_clf[valid_train]
        y_train_reg = y_train_reg[valid_train]
        X_val = X_val[valid_val]

        if len(X_train) < 200 or len(X_val) < 20:
            start += SLIDE_DAYS
            continue

        # ── Classifier: P(loser) ──
        dtrain_clf = lgb.Dataset(X_train, label=y_train_clf, feature_name=feature_cols, free_raw_data=False)
        dval_clf = lgb.Dataset(X_val, label=y_val_clf[valid_val] if len(y_val_clf[valid_val]) == len(X_val) else y_val_clf[:len(X_val)],
                               feature_name=feature_cols, free_raw_data=False)

        clf_model = lgb.train(
            LGBM_CLF_PARAMS,
            dtrain_clf,
            num_boost_round=400,
            valid_sets=[dval_clf],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        clf_preds = clf_model.predict(X_val)

        # ── Regressor: remaining MFE ──
        dtrain_reg = lgb.Dataset(X_train, label=y_train_reg, feature_name=feature_cols, free_raw_data=False)
        dval_reg = lgb.Dataset(X_val, label=y_val_reg[valid_val] if len(y_val_reg[valid_val]) == len(X_val) else y_val_reg[:len(X_val)],
                               feature_name=feature_cols, free_raw_data=False)

        reg_model = lgb.train(
            LGBM_REG_PARAMS,
            dtrain_reg,
            num_boost_round=400,
            valid_sets=[dval_reg],
            callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)],
        )
        reg_preds = reg_model.predict(X_val)

        # Store predictions
        val_indices = samples_df.index[val_mask][valid_val]
        all_clf_preds.iloc[val_indices] = clf_preds
        all_reg_preds.iloc[val_indices] = reg_preds

        # Fold-level metrics
        if len(y_val_clf[valid_val]) > 0:
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(y_val_clf[valid_val], clf_preds)
            except ValueError:
                auc = 0.5
            reg_mae = np.mean(np.abs(y_val_reg[valid_val] - reg_preds))

            fold_results.append({
                "fold": fold_idx,
                "train_start": train_dates[0],
                "train_end": train_dates[-1],
                "val_start": val_dates[0],
                "val_end": val_dates[-1],
                "n_train": len(X_train),
                "n_val": len(X_val),
                "clf_auc": auc,
                "reg_mae": reg_mae,
            })

            if fold_idx % 3 == 0:
                log.info(f"Mgmt fold {fold_idx}: AUC={auc:.4f}, MAE={reg_mae:.2f}, "
                         f"val={val_dates[0]}..{val_dates[-1]}")

        fold_idx += 1
        start += SLIDE_DAYS

        # Save the last fold's model
        if start + TRAIN_DAYS + VAL_DAYS > len(dates):
            clf_model.save_model(str(MODEL_DIR / "invalidation_clf_latest.txt"))
            reg_model.save_model(str(MODEL_DIR / "remaining_mfe_reg_latest.txt"))
            log.info("Saved latest management models")

        del clf_model, reg_model, dtrain_clf, dtrain_reg, dval_clf, dval_reg
        gc.collect()

    samples_df["pred_p_loser"] = all_clf_preds
    samples_df["pred_remaining_mfe"] = all_reg_preds

    log.info(f"Management models trained: {fold_idx} folds")
    if fold_results:
        avg_auc = np.mean([f["clf_auc"] for f in fold_results])
        avg_mae = np.mean([f["reg_mae"] for f in fold_results])
        log.info(f"  Average AUC={avg_auc:.4f}, Average MAE={avg_mae:.2f}")

    return fold_results, samples_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 7: SIMULATION — STATIC vs DYNAMIC EXIT
# ═══════════════════════════════════════════════════════════════════


def simulate_exits(
    samples_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    invalidation_thresholds: List[float] = [0.60, 0.65, 0.70, 0.75, 0.80],
    mfe_thresholds: List[float] = [0.5, 1.0, 1.5, 2.0],
) -> Dict:
    """
    Compare static 30-min exit vs dynamic exit strategies.

    Dynamic exit: at each minute, check if P(loser) > threshold OR remaining_MFE < threshold.
    If either condition met → exit at that minute's price.
    """
    results = {}

    # ── Static exit (baseline) ──
    static_trades = meta_df.copy()
    static_pnl = static_trades["actual_outcome_ticks"] - COST_RT_TICKS
    static_sharpe = _compute_sharpe(static_pnl.values)
    static_wr = (static_pnl > 0).mean()
    static_pf = _compute_profit_factor(static_pnl.values)
    static_sortino = _compute_sortino(static_pnl.values)

    results["static"] = {
        "n_trades": len(static_trades),
        "mean_pnl_ticks": float(static_pnl.mean()),
        "sharpe": static_sharpe,
        "sortino": static_sortino,
        "win_rate": float(static_wr),
        "profit_factor": static_pf,
        "total_pnl_ticks": float(static_pnl.sum()),
        "max_drawdown_ticks": float(_max_drawdown(static_pnl.values)),
        "avg_hold_min": 30.0,
    }

    log.info(f"STATIC EXIT: Sharpe={static_sharpe:.3f}, Sortino={static_sortino:.3f}, "
             f"WR={static_wr:.1%}, PF={static_pf:.2f}, "
             f"mean={static_pnl.mean():.2f}t, n={len(static_trades)}")

    # ── Dynamic exits ──
    valid_samples = samples_df.dropna(subset=["pred_p_loser", "pred_remaining_mfe"])
    if len(valid_samples) == 0:
        log.warning("No valid management predictions for simulation!")
        return results

    best_config = None
    best_sharpe = static_sharpe

    for inv_thresh in invalidation_thresholds:
        for mfe_thresh in mfe_thresholds:
            config_key = f"dynamic_inv{inv_thresh:.2f}_mfe{mfe_thresh:.1f}"
            trade_results = []

            for trade_idx in meta_df["trade_idx"].unique():
                trade_info = meta_df[meta_df["trade_idx"] == trade_idx].iloc[0]
                trade_samples = valid_samples[valid_samples["trade_idx"] == trade_idx].sort_values("time_in_trade_minutes")

                if len(trade_samples) == 0:
                    # No management predictions → use static exit
                    pnl = trade_info["actual_outcome_ticks"] - COST_RT_TICKS
                    hold_time = 30
                    exit_reason = "static"
                else:
                    exit_minute = None
                    exit_reason = "timeout"  # default: held to max

                    for _, sample in trade_samples.iterrows():
                        t = sample["time_in_trade_minutes"]

                        # Check invalidation
                        if sample["pred_p_loser"] > inv_thresh:
                            exit_minute = t
                            exit_reason = "invalidation"
                            break

                        # Check remaining MFE
                        if t >= 5 and sample["pred_remaining_mfe"] < mfe_thresh:
                            exit_minute = t
                            exit_reason = "low_mfe"
                            break

                    if exit_minute is not None:
                        # Exit at this minute's unrealized PnL
                        exit_sample = trade_samples[trade_samples["time_in_trade_minutes"] == exit_minute]
                        if len(exit_sample) > 0:
                            pnl = exit_sample.iloc[0]["unrealized_pnl_ticks"] - COST_RT_TICKS
                        else:
                            pnl = trade_info["actual_outcome_ticks"] - COST_RT_TICKS
                        hold_time = exit_minute
                    else:
                        # Held to timeout/static exit
                        pnl = trade_info["actual_outcome_ticks"] - COST_RT_TICKS
                        hold_time = 30

                trade_results.append({
                    "trade_idx": trade_idx,
                    "pnl_ticks": pnl,
                    "hold_time_min": hold_time,
                    "exit_reason": exit_reason,
                    "direction": trade_info["direction"],
                    "date": trade_info["date"],
                })

            trades_df = pd.DataFrame(trade_results)
            pnl_arr = trades_df["pnl_ticks"].values
            sharpe = _compute_sharpe(pnl_arr)
            sortino = _compute_sortino(pnl_arr)
            wr = (pnl_arr > 0).mean()
            pf = _compute_profit_factor(pnl_arr)
            avg_hold = trades_df["hold_time_min"].mean()

            exit_reasons = trades_df["exit_reason"].value_counts().to_dict()

            results[config_key] = {
                "inv_threshold": inv_thresh,
                "mfe_threshold": mfe_thresh,
                "n_trades": len(trades_df),
                "mean_pnl_ticks": float(pnl_arr.mean()),
                "sharpe": sharpe,
                "sortino": sortino,
                "win_rate": float(wr),
                "profit_factor": pf,
                "total_pnl_ticks": float(pnl_arr.sum()),
                "max_drawdown_ticks": float(_max_drawdown(pnl_arr)),
                "avg_hold_min": float(avg_hold),
                "exit_reasons": exit_reasons,
                "avg_hold_winners": float(trades_df[trades_df["pnl_ticks"] > 0]["hold_time_min"].mean()) if (trades_df["pnl_ticks"] > 0).any() else 0,
                "avg_hold_losers": float(trades_df[trades_df["pnl_ticks"] <= 0]["hold_time_min"].mean()) if (trades_df["pnl_ticks"] <= 0).any() else 0,
            }

            if sharpe > best_sharpe:
                best_sharpe = sharpe
                best_config = config_key

    if best_config:
        log.info(f"\nBEST DYNAMIC CONFIG: {best_config}")
        best = results[best_config]
        log.info(f"  Sharpe={best['sharpe']:.3f} (vs static {static_sharpe:.3f}), "
                 f"Sortino={best['sortino']:.3f}, WR={best['win_rate']:.1%}, "
                 f"PF={best['profit_factor']:.2f}")
        log.info(f"  Avg hold: {best['avg_hold_min']:.1f}min "
                 f"(winners: {best['avg_hold_winners']:.1f}, losers: {best['avg_hold_losers']:.1f})")
        log.info(f"  Exit reasons: {best['exit_reasons']}")
        results["best_config"] = best_config
    else:
        log.info("No dynamic config improved over static exit")
        results["best_config"] = "static"

    return results


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
#  SECTION 8: REGIME STRATIFICATION
# ═══════════════════════════════════════════════════════════════════


def regime_analysis(
    samples_df: pd.DataFrame,
    meta_df: pd.DataFrame,
    bars_df: pd.DataFrame,
) -> Dict:
    """Stratify results by market regime (green/red days)."""
    # Compute daily returns from bars
    day_returns = bars_df.groupby("date").agg(
        day_open=("open", "first"),
        day_close=("close", "last"),
    )
    day_returns["day_return"] = day_returns["day_close"] - day_returns["day_open"]
    day_returns["regime"] = "flat"
    day_returns.loc[day_returns["day_return"] > 0, "regime"] = "green"
    day_returns.loc[day_returns["day_return"] < 0, "regime"] = "red"

    regime_map = day_returns["regime"].to_dict()
    meta_df = meta_df.copy()
    meta_df["regime"] = meta_df["date"].map(regime_map).fillna("flat")

    results = {}
    for regime in ["green", "red", "flat"]:
        regime_trades = meta_df[meta_df["regime"] == regime]
        if len(regime_trades) < 5:
            continue

        pnl = regime_trades["actual_outcome_ticks"].values - COST_RT_TICKS
        results[regime] = {
            "n_trades": len(regime_trades),
            "sharpe": _compute_sharpe(pnl),
            "sortino": _compute_sortino(pnl),
            "win_rate": float((pnl > 0).mean()),
            "profit_factor": _compute_profit_factor(pnl),
            "mean_pnl_ticks": float(pnl.mean()),
        }
        log.info(f"Regime {regime}: Sharpe={results[regime]['sharpe']:.3f}, "
                 f"WR={results[regime]['win_rate']:.1%}, n={len(regime_trades)}")

    # HC #428: check regime gap
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
    n_trades: int,
    n_samples: int,
):
    """Log experiment to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri("http://neptune:5000")
        mlflow.set_experiment("trade_management_v1")

        with mlflow.start_run(run_name=f"tm_v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            # Params
            mlflow.log_param("model_type", "LightGBM_clf+reg")
            mlflow.log_param("train_days", TRAIN_DAYS)
            mlflow.log_param("val_days", VAL_DAYS)
            mlflow.log_param("slide_days", SLIDE_DAYS)
            mlflow.log_param("max_hold_minutes", MAX_HOLD_MINUTES)
            mlflow.log_param("invalidation_threshold", INVALIDATION_THRESHOLD)
            mlflow.log_param("mfe_threshold", REMAINING_MFE_THRESHOLD)
            mlflow.log_param("cost_rt_ticks", COST_RT_TICKS)
            mlflow.log_param("n_trades", n_trades)
            mlflow.log_param("n_samples", n_samples)

            # Fold-level metrics
            if fold_results:
                avg_auc = np.mean([f["clf_auc"] for f in fold_results])
                avg_mae = np.mean([f["reg_mae"] for f in fold_results])
                mlflow.log_metric("avg_clf_auc", avg_auc)
                mlflow.log_metric("avg_reg_mae", avg_mae)
                mlflow.log_metric("n_folds", len(fold_results))

            # Static baseline
            if "static" in sim_results:
                s = sim_results["static"]
                mlflow.log_metric("static_sharpe", s["sharpe"])
                mlflow.log_metric("static_sortino", s["sortino"])
                mlflow.log_metric("static_wr", s["win_rate"])
                mlflow.log_metric("static_pf", s["profit_factor"])
                mlflow.log_metric("static_mean_pnl", s["mean_pnl_ticks"])

            # Best dynamic
            best_key = sim_results.get("best_config", "static")
            if best_key != "static" and best_key in sim_results:
                b = sim_results[best_key]
                mlflow.log_metric("dynamic_sharpe", b["sharpe"])
                mlflow.log_metric("dynamic_sortino", b["sortino"])
                mlflow.log_metric("dynamic_wr", b["win_rate"])
                mlflow.log_metric("dynamic_pf", b["profit_factor"])
                mlflow.log_metric("dynamic_mean_pnl", b["mean_pnl_ticks"])
                mlflow.log_metric("dynamic_avg_hold_min", b["avg_hold_min"])
                mlflow.log_metric("sharpe_improvement", b["sharpe"] - sim_results["static"]["sharpe"])

            # Regime
            if "regime_gap" in regime_results:
                mlflow.log_metric("regime_gap", regime_results["regime_gap"])
                mlflow.log_param("regime_pass", regime_results["regime_pass"])

            # Artifacts
            summary_path = str(OUTPUT_DIR / "summary.json")
            if os.path.exists(summary_path):
                mlflow.log_artifact(summary_path)

        log.info("MLflow logging complete")
    except Exception as e:
        log.warning(f"MLflow logging failed (non-fatal): {e}")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 10: MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Trade Management Model v1")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    args = parser.parse_args()

    start_time = time.time()
    log.info("=" * 70)
    log.info("Trade Management Model v1 — Dynamic Exit Learning")
    log.info(f"Device: {args.device}")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info("=" * 70)

    # ── Step 1: Load minute-level data ──
    log.info("\n[STEP 1/7] Loading minute bars...")
    minute_df = load_all_minute_bars()

    # ── Step 2: Aggregate to 30-min bars for entry model ──
    log.info("\n[STEP 2/7] Aggregating to 30-min bars...")
    bars_df = aggregate_to_bars(minute_df, bar_size_min=ENTRY_BAR_SIZE_MIN)
    bars_df = add_rolling_features(bars_df)
    bars_df = add_forward_labels(bars_df, ENTRY_HORIZON_BARS, ENTRY_HORIZON_LABEL)

    log.info(f"  Bars: {len(bars_df)}, Features: {len(get_entry_feature_columns(bars_df))}")

    # ── Step 3: Train entry model (walk-forward) to get trade entries ──
    log.info("\n[STEP 3/7] Training entry model (walk-forward)...")
    bars_df = train_entry_model_wf(bars_df)

    n_entries = (bars_df["entry_signal"] != 0).sum()
    log.info(f"  {n_entries} entry signals generated")

    if n_entries < 50:
        log.error(f"Only {n_entries} entries — not enough for management model. Aborting.")
        return

    # ── Step 4: Construct trade-level samples ──
    log.info("\n[STEP 4/7] Constructing trade-level samples...")
    result = construct_trade_samples(bars_df, minute_df)
    if isinstance(result, tuple):
        samples_df, meta_df = result
    else:
        log.error("Trade sample construction returned no data!")
        return

    if len(samples_df) < 500:
        log.error(f"Only {len(samples_df)} samples — not enough. Aborting.")
        return

    log.info(f"  {len(samples_df)} samples from {len(meta_df)} trades")

    # ── Step 5: Train management models (walk-forward) ──
    log.info("\n[STEP 5/7] Training management models (walk-forward)...")
    fold_results, samples_df = train_management_models_wf(samples_df, meta_df)

    # ── Step 6: Simulate static vs dynamic exits ──
    log.info("\n[STEP 6/7] Simulating exit strategies...")
    sim_results = simulate_exits(samples_df, meta_df)

    # ── Step 7: Regime analysis ──
    log.info("\n[STEP 7/7] Regime stratification...")
    regime_results = regime_analysis(samples_df, meta_df, bars_df)

    # ── Save results ──
    elapsed = time.time() - start_time

    summary = {
        "run_time": datetime.now().isoformat(),
        "elapsed_seconds": elapsed,
        "n_trades": len(meta_df),
        "n_samples": len(samples_df),
        "n_folds": len(fold_results),
        "entry_model": {
            "n_long": int((meta_df["direction"] == 1).sum()),
            "n_short": int((meta_df["direction"] == -1).sum()),
            "overall_wr": float(meta_df["is_winner"].mean()),
        },
        "management_model": {
            "avg_clf_auc": float(np.mean([f["clf_auc"] for f in fold_results])) if fold_results else 0,
            "avg_reg_mae": float(np.mean([f["reg_mae"] for f in fold_results])) if fold_results else 0,
        },
        "simulation": sim_results,
        "regime": regime_results,
        "cost_rt_ticks": COST_RT_TICKS,
    }

    # Save summary
    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary saved to {summary_path}")

    # Save fold results
    if fold_results:
        fold_df = pd.DataFrame(fold_results)
        fold_df.to_csv(OUTPUT_DIR / "fold_results.csv", index=False)

    # Save trade-level results
    meta_df.to_csv(OUTPUT_DIR / "trade_meta.csv", index=False)

    # MLflow
    log.info("\nLogging to MLflow...")
    log_to_mlflow(fold_results, sim_results, regime_results, len(meta_df), len(samples_df))

    # ── Final report ──
    log.info("\n" + "=" * 70)
    log.info("TRADE MANAGEMENT MODEL v1 — COMPLETE")
    log.info(f"Elapsed: {elapsed/60:.1f} minutes")
    log.info(f"Trades: {len(meta_df)}, Samples: {len(samples_df)}, Folds: {len(fold_results)}")

    if "static" in sim_results:
        s = sim_results["static"]
        log.info(f"\nSTATIC EXIT (baseline):")
        log.info(f"  Sharpe={s['sharpe']:.3f}, Sortino={s['sortino']:.3f}, "
                 f"WR={s['win_rate']:.1%}, PF={s['profit_factor']:.2f}")

    best_key = sim_results.get("best_config", "static")
    if best_key != "static" and best_key in sim_results:
        b = sim_results[best_key]
        log.info(f"\nBEST DYNAMIC EXIT ({best_key}):")
        log.info(f"  Sharpe={b['sharpe']:.3f}, Sortino={b['sortino']:.3f}, "
                 f"WR={b['win_rate']:.1%}, PF={b['profit_factor']:.2f}")
        log.info(f"  Avg hold: {b['avg_hold_min']:.1f}min "
                 f"(winners: {b['avg_hold_winners']:.1f}, losers: {b['avg_hold_losers']:.1f})")
        improvement = b['sharpe'] - sim_results['static']['sharpe']
        log.info(f"  Sharpe improvement: {improvement:+.3f}")
    else:
        log.info("\nNo dynamic config improved over static exit.")

    log.info("=" * 70)


if __name__ == "__main__":
    main()
