#!/usr/bin/env python3
"""
Trailing Stop Full OOT v1 — Maximum OOT Coverage + Full Regime Analysis
=========================================================================

PROBLEM: Previous trailing stop sweep (5,400 configs, all Sharpe > 8) only
used 84 OOT days out of 197 available. Root causes:
  1. 40-day training window means OOT starts at day 41
  2. Confluence gate (2/6) + top/bottom 10% filtering removes too many days
  3. Some days have no trades after filtering

FIXES APPLIED:
  1. 30d train window (vs 40d) — OOT starts at day 31, ~167 OOT days
  2. NO confluence gating — just the raw 30-min model (Sharpe 4.03 champion)
  3. Entry threshold sweep: 5%, 10%, 15%, 20%, 25% (more trades per day)
  4. Trailing stop trigger sweep: 2, 3, 4, 5 ticks

ARCHITECTURE:
  Phase 1: Train 30-min entry model walk-forward (30d train, 5d slide)
  Phase 2: Reconstruct trades using various entry thresholds (no confluence)
  Phase 3: Run trailing stop configs on all trades
  Phase 4: Sweep entry_threshold x trailing_trigger (20 configs)
  Phase 5: Full regime analysis on best config

CONSTRAINTS:
  - Walk-forward: 30d train, 5d slide — SLIDING only (HC #0)
  - Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission)
  - Regime-agnostic gate: |Sharpe_green - Sharpe_red| / max < 0.50 (HC #428)
  - Day-concentration cap <= 0.70 (HC #344)
  - MLflow logging mandatory

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python -u \\
      alpha_discovery/trailing_stop_full_oot_v1.py 2>&1 | \\
      tee logs/trailing_stop_full_oot_v1.log

Author: Claude (autonomous research)
"""

import gc
import json
import logging
import os
import sys
import time
import traceback
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
OUTPUT_DIR = ROOT / "output" / "trailing_stop_full_oot_v1"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [TS-FULL-OOT] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "trailing_stop_full_oot_v1.log")),
    ],
)
log = logging.getLogger("TS-FULL-OOT")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376
COST_RT_TICKS = 2.376  # market entry + market exit + RT commission

# ─────────────────────────────────────────────
#  WALK-FORWARD CONFIG — 30d for MAX OOT
# ─────────────────────────────────────────────
TRAIN_DAYS = 30         # 30d train → OOT starts day 31, ~167 OOT days
SLIDE_DAYS = 5

# ─────────────────────────────────────────────
#  TRADE CONFIG
# ─────────────────────────────────────────────
ENTRY_BAR_SIZE_MIN = 30        # entry signal bar resolution
MGMT_BAR_SIZE_MIN = 5          # management checkpoint resolution (5-min bars)
MAX_HOLD_MINUTES = 60          # max extension for trailing stop trades

# ─────────────────────────────────────────────
#  SWEEP GRID
# ─────────────────────────────────────────────
ENTRY_THRESHOLDS = [0.05, 0.10, 0.15, 0.20, 0.25]
TRAILING_TRIGGERS = [2, 3, 4, 5]  # MFE ticks to activate trailing stop

# Best proven config from v5 (baseline reference)
BEST_TRAILING = {
    "trigger1": 3, "buf1": 2.0,
    "trigger2": 6, "buf2": 1.5,
    "stale_bars": 4, "max_hold": 30,
}

# ─────────────────────────────────────────────
#  LGBM PARAMS
# ─────────────────────────────────────────────
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
#  SECTION 2: BAR AGGREGATION + FEATURES
# ═══════════════════════════════════════════════════════════════════


def _safe_polyfit_slope(arr: np.ndarray) -> float:
    """Linear regression slope, returns 0 on failure."""
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def aggregate_to_bars(minute_df: pd.DataFrame, bar_size_min: int = 30) -> pd.DataFrame:
    """Aggregate 1-minute bars into N-minute bars with microstructure features."""
    df = minute_df.copy()
    bar_label = f"{bar_size_min}min"
    df["bar_key"] = df["ts_minute"].dt.floor(f"{bar_size_min}min")
    df["return_1m"] = df.groupby("date")["close"].pct_change()
    df["abs_ofi"] = df["ofi_1min"].abs()
    sv_std = df.groupby("date")["signed_volume"].transform("std").replace(0, 1)
    df["sv_zscore"] = df["signed_volume"] / sv_std
    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["close"].clip(lower=1)

    records = []
    for (date_str, bar_key), grp in df.groupby(["date", "bar_key"]):
        if len(grp) < 2:
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
    log.info(f"Aggregated {len(result):,} {bar_label} bars with {len(result.columns)} columns")
    return result


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: ROLLING CONTEXT + REGIME FEATURES
# ═══════════════════════════════════════════════════════════════════


def add_rolling_features(df: pd.DataFrame, bar_size_min: int = 30) -> pd.DataFrame:
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
                df["sweep_minutes"].rolling(w, min_periods=1).sum() / (w * bar_size_min)
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

    # Previous day context (causal)
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

    # Time-of-day encoding
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


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: FEATURE COLUMN SELECTION
# ═══════════════════════════════════════════════════════════════════


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
#  SECTION 5: FORWARD LABELS
# ═══════════════════════════════════════════════════════════════════


def add_forward_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    horizon_label: str,
    min_edge_ticks: float = 2.5,
) -> pd.DataFrame:
    """Compute forward return labels with overnight gap protection."""
    df = df.sort_values("ts").reset_index(drop=True)

    fwd_close = df["close"].shift(-horizon_bars)
    fwd_return = fwd_close / df["close"] - 1
    fwd_ticks = (fwd_close - df["close"]) / 0.25

    # Null out overnight gaps
    ts_now = df["ts"].values
    ts_fwd = df["ts"].shift(-horizon_bars).values
    for i in range(len(df) - horizon_bars):
        if pd.isna(ts_fwd[i]):
            continue
        diff_s = (pd.Timestamp(ts_fwd[i]) - pd.Timestamp(ts_now[i])).total_seconds()
        if diff_s > 6 * 3600:
            fwd_return.iloc[i] = np.nan
            fwd_ticks.iloc[i] = np.nan

    df[f"fwd_return_{horizon_label}"] = fwd_return
    df[f"fwd_ticks_{horizon_label}"] = fwd_ticks

    df[f"direction_{horizon_label}"] = 0
    df.loc[fwd_ticks > min_edge_ticks, f"direction_{horizon_label}"] = 1
    df.loc[fwd_ticks < -min_edge_ticks, f"direction_{horizon_label}"] = -1

    log.info(
        f"Forward labels ({horizon_label}): "
        f"{(~fwd_ticks.isna()).sum():,} valid, "
        f"{(df[f'direction_{horizon_label}'] == 1).sum():,} long, "
        f"{(df[f'direction_{horizon_label}'] == -1).sum():,} short"
    )
    return df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: LEAKAGE AUDIT
# ═══════════════════════════════════════════════════════════════════


def leakage_audit(
    train_dates: List[str],
    val_dates: List[str],
    feature_cols: List[str],
) -> bool:
    """Explicit leakage audit. Returns True if all checks pass."""
    train_set = set(train_dates)
    val_set = set(val_dates)
    overlap = train_set & val_set
    if overlap:
        log.error(f"LEAKAGE: Train/val date overlap: {overlap}")
        return False

    max_train = max(train_dates)
    min_val = min(val_dates)
    if max_train >= min_val:
        log.error(f"LEAKAGE: Max train date {max_train} >= min val date {min_val}")
        return False

    fwd_leak = [c for c in feature_cols if c.startswith(("fwd_", "direction_", "trade_quality_"))]
    if fwd_leak:
        log.error(f"LEAKAGE: Forward-looking columns in features: {fwd_leak}")
        return False

    return True


# ═══════════════════════════════════════════════════════════════════
#  SECTION 7: ENTRY MODEL TRAINING
# ═══════════════════════════════════════════════════════════════════


def train_entry_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    fold_idx: int,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Train a LightGBM entry model for the 30-min horizon."""
    _import_lightgbm()

    train_valid = ~np.isnan(y_train)
    val_valid = ~np.isnan(y_val)

    if train_valid.sum() < 50 or val_valid.sum() < 10:
        log.warning(
            f"  Entry fold {fold_idx}: too few valid "
            f"(train={train_valid.sum()}, val={val_valid.sum()}) -- skip"
        )
        return None, np.full(len(y_train), np.nan), np.full(len(y_val), np.nan)

    X_tr = X_train[train_valid]
    y_tr = y_train[train_valid]
    X_v = X_val[val_valid]
    y_v = y_val[val_valid]

    params = {**LGBM_ENTRY_PARAMS, "seed": 42 + fold_idx}

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    val_data = lgb.Dataset(X_v, label=y_v, feature_name=feature_names, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[val_data], callbacks=callbacks,
    )

    train_preds = np.full(len(y_train), np.nan)
    val_preds = np.full(len(y_val), np.nan)
    train_preds[train_valid] = model.predict(X_tr, num_iteration=model.best_iteration)
    val_preds[val_valid] = model.predict(X_v, num_iteration=model.best_iteration)

    p_v = val_preds[val_valid]
    if len(p_v) > 5:
        ic = np.corrcoef(p_v, y_v)[0, 1]
        log.info(f"  Entry fold {fold_idx}: IC={ic:.4f}, best_iter={model.best_iteration}")

    return model, train_preds, val_preds


# ═══════════════════════════════════════════════════════════════════
#  SECTION 8: TRADE RECONSTRUCTION (NO CONFLUENCE)
# ═══════════════════════════════════════════════════════════════════


def reconstruct_trades_no_confluence(
    bars_30m: pd.DataFrame,
    minute_df: pd.DataFrame,
    pred_30m: np.ndarray,
    confidence_pct: float = 0.15,
    max_hold_minutes: int = MAX_HOLD_MINUTES,
) -> List[Dict]:
    """
    Reconstruct trades from entry signals WITHOUT confluence gating.

    For each 30-min bar where model prediction is in top/bottom confidence_pct,
    create a trade record with minute-level price path for trailing stop analysis.
    No confluence gate = maximum OOT day coverage.
    """
    valid_mask = ~np.isnan(pred_30m)
    valid_preds = pred_30m[valid_mask]
    if len(valid_preds) < 20:
        log.warning("Too few valid predictions for trade reconstruction")
        return []

    upper_thresh = np.nanquantile(pred_30m[valid_mask], 1 - confidence_pct)
    lower_thresh = np.nanquantile(pred_30m[valid_mask], confidence_pct)

    # Build minute-level lookup: date -> sorted minute bars
    minute_lookup = {}
    for date_str, grp in minute_df.groupby("date"):
        grp_sorted = grp.sort_values("ts_minute").reset_index(drop=True)
        minute_lookup[date_str] = grp_sorted

    trades = []
    bars_ts = bars_30m["ts"].values
    bars_dates = bars_30m["date"].values
    bars_close = bars_30m["close"].values

    for i in range(len(bars_30m)):
        if np.isnan(pred_30m[i]):
            continue

        # Check entry conditions — NO confluence gate
        direction = 0
        if pred_30m[i] >= upper_thresh:
            direction = 1
        elif pred_30m[i] <= lower_thresh:
            direction = -1
        else:
            continue

        date_str = bars_dates[i]
        entry_ts = pd.Timestamp(bars_ts[i])
        entry_price = bars_close[i]

        if date_str not in minute_lookup:
            continue

        day_minutes = minute_lookup[date_str]
        day_ts = day_minutes["ts_minute"].values

        # Find minute bars within the trade window (up to max_hold_minutes)
        entry_ts_np = np.datetime64(entry_ts)
        window_end = entry_ts + pd.Timedelta(minutes=max_hold_minutes)
        window_end_np = np.datetime64(window_end)

        # Get minute bars after entry within window
        mask = (day_ts >= entry_ts_np) & (day_ts <= window_end_np)
        trade_minutes = day_minutes[mask].copy()

        if len(trade_minutes) < 5:
            continue

        # Extract price path
        prices = trade_minutes["close"].values
        times = trade_minutes["ts_minute"].values

        # Compute P&L path in ticks
        pnl_path = (prices - entry_price) / 0.25 * direction

        # MFE and MAE at each minute
        mfe_path = np.maximum.accumulate(pnl_path)
        mae_path = np.minimum.accumulate(pnl_path)

        trade = {
            "idx": i,
            "date": date_str,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "direction": direction,
            "pred_30m": float(pred_30m[i]),
            # Minute-level data for trailing stop
            "prices": prices,
            "times": times,
            "pnl_path": pnl_path,
            "mfe_path": mfe_path,
            "mae_path": mae_path,
            "n_minutes": len(prices),
        }

        # Static hold outcomes
        for hold_min in [10, 15, 20, 25, 30, 45, 60]:
            idx_hold = min(hold_min, len(pnl_path) - 1)
            trade[f"pnl_{hold_min}min"] = float(pnl_path[idx_hold]) - COST_RT_TICKS

        trades.append(trade)

    log.info(f"Reconstructed {len(trades)} trades "
             f"(confidence={confidence_pct:.0%}, NO confluence gate)")

    if trades:
        directions = [t["direction"] for t in trades]
        log.info(f"  Long: {sum(1 for d in directions if d == 1)}, "
                 f"Short: {sum(1 for d in directions if d == -1)}")
        trade_dates = sorted(set(t["date"] for t in trades))
        log.info(f"  Trading days: {len(trade_dates)} "
                 f"({trade_dates[0]} -> {trade_dates[-1]})")

    return trades


# ═══════════════════════════════════════════════════════════════════
#  SECTION 9: ADAPTIVE TRAILING STOP
# ═══════════════════════════════════════════════════════════════════


def simulate_trailing_stop(
    trades: List[Dict],
    trigger1: float = 3.0,
    buf1: float = 2.0,
    trigger2: float = 6.0,
    buf2: float = 1.5,
    stale_bars: int = 4,
    max_hold: int = 30,
    checkpoint_interval: int = MGMT_BAR_SIZE_MIN,
) -> Dict:
    """
    Simulate adaptive trailing stop on a list of trades.

    Trailing stop logic:
      - MFE >= trigger2 ticks: floor = MFE - buf2 (lock in profits aggressively)
      - MFE >= trigger1 ticks: floor = MFE - buf1 (lock in some profit)
      - MFE < trigger1 ticks: no trailing floor
      - stale_bars mgmt bars (stale_bars * checkpoint_interval min) with no new MFE: exit
      - max_hold: max hold in minutes

    Returns comprehensive statistics.
    """
    pnls, dates, dirs, hold_durations = [], [], [], []
    exit_reasons = {"trailing_floor": 0, "stale_mfe": 0, "static_hold": 0, "max_hold": 0}

    for trade in trades:
        pnl_path = trade["pnl_path"]
        mfe_path = trade["mfe_path"]
        n_minutes = trade["n_minutes"]
        exited = False
        exit_minute = min(max_hold, n_minutes - 1)

        for ckpt in range(checkpoint_interval, n_minutes - 1, checkpoint_interval):
            if ckpt > max_hold:
                exit_minute = max_hold if max_hold < n_minutes else n_minutes - 1
                exited = True
                exit_reasons["max_hold"] += 1
                break

            mfe_now = mfe_path[ckpt]
            pnl_now = pnl_path[ckpt]

            # bars_since_mfe: how many mgmt bars since MFE was last extended
            bars_since = 0
            for lb in range(1, ckpt + 1):
                if mfe_path[ckpt - lb] < mfe_now:
                    break
                bars_since += 1
            bars_since_mgmt = bars_since / max(checkpoint_interval, 1)

            # Stale trade exit
            if bars_since_mgmt >= stale_bars and mfe_now > 1.0:
                exit_minute = ckpt
                exited = True
                exit_reasons["stale_mfe"] += 1
                break

            # Trailing stop floor
            floor = None
            if mfe_now >= trigger2:
                floor = mfe_now - buf2
            elif mfe_now >= trigger1:
                floor = mfe_now - buf1

            if floor is not None and pnl_now < floor:
                exit_minute = ckpt
                exited = True
                exit_reasons["trailing_floor"] += 1
                break

        if not exited:
            exit_minute = min(max_hold, n_minutes - 1)
            exit_reasons["static_hold"] += 1

        exit_minute = min(exit_minute, n_minutes - 1)
        trade_pnl = float(pnl_path[exit_minute]) - COST_RT_TICKS
        pnls.append(trade_pnl)
        dates.append(trade["date"])
        dirs.append(trade["direction"])
        hold_durations.append(exit_minute)

    if not pnls:
        return {"error": "no trades"}

    return _compute_trade_stats(
        np.array(pnls), np.array(dates), np.array(dirs),
        f"TS_t{trigger1}_b{buf1}_t2{trigger2}_b2{buf2}_s{stale_bars}_h{max_hold}",
        hold_durations=hold_durations, exit_reasons=exit_reasons,
    )


def simulate_static(trades: List[Dict], hold_minutes: int = 30) -> Dict:
    """STATIC hold baseline."""
    pnl_key = f"pnl_{hold_minutes}min"
    pnls, dates, dirs = [], [], []

    for trade in trades:
        if pnl_key in trade:
            pnls.append(trade[pnl_key])
            dates.append(trade["date"])
            dirs.append(trade["direction"])

    if not pnls:
        return {"error": "no trades"}

    return _compute_trade_stats(
        np.array(pnls), np.array(dates), np.array(dirs), f"STATIC_{hold_minutes}",
        hold_durations=[hold_minutes] * len(pnls),
    )


# ═══════════════════════════════════════════════════════════════════
#  SECTION 10: TRADE STATISTICS
# ═══════════════════════════════════════════════════════════════════


def _compute_trade_stats(
    pnl_arr: np.ndarray,
    dates: np.ndarray,
    directions: np.ndarray,
    label: str,
    hold_durations: Optional[List[float]] = None,
    exit_reasons: Optional[Dict] = None,
) -> Dict:
    """Compute standard trade statistics with per-side breakdown."""
    if len(pnl_arr) == 0:
        return {"error": "no trades"}

    cum_pnl = np.cumsum(pnl_arr)
    sharpe = pnl_arr.mean() / max(pnl_arr.std(), 1e-6) * np.sqrt(252)
    downside = np.sqrt(np.mean(np.minimum(pnl_arr, 0) ** 2))
    sortino = pnl_arr.mean() / max(downside, 1e-6) * np.sqrt(252)
    wr = np.mean(pnl_arr > 0)
    pf = np.sum(pnl_arr[pnl_arr > 0]) / max(-np.sum(pnl_arr[pnl_arr < 0]), 1e-6)
    max_dd = float(np.min(cum_pnl - np.maximum.accumulate(cum_pnl)))

    # Per-side breakdown
    long_mask = directions == 1
    short_mask = directions == -1
    long_pnl = pnl_arr[long_mask] if long_mask.any() else np.array([])
    short_pnl = pnl_arr[short_mask] if short_mask.any() else np.array([])

    # Per-day aggregation
    trade_df = pd.DataFrame({"pnl": pnl_arr, "date": dates})
    day_pnl = trade_df.groupby("date")["pnl"].agg(["sum", "count"]).reset_index()
    day_pnl.columns = ["date", "daily_pnl", "daily_trades"]

    daily_sharpe = 0.0
    if len(day_pnl) > 2:
        daily_sharpe = float(
            day_pnl["daily_pnl"].mean() / max(day_pnl["daily_pnl"].std(), 1e-6) * np.sqrt(252)
        )

    # Day concentration (HC #344)
    day_conc = 0.0
    if len(day_pnl) > 0:
        total_abs = day_pnl["daily_pnl"].abs().sum()
        if total_abs > 0:
            day_conc = float(day_pnl["daily_pnl"].abs().max() / total_abs)

    result = {
        "label": label,
        "n_trades": len(pnl_arr),
        "total_pnl_ticks": float(pnl_arr.sum()),
        "total_pnl_dollars": float(pnl_arr.sum() * ES_TICK_VALUE),
        "avg_pnl_ticks": float(pnl_arr.mean()),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "win_rate": float(wr),
        "profit_factor": float(pf),
        "max_dd_ticks": float(max_dd),
        "max_dd_dollars": float(max_dd * ES_TICK_VALUE),
        "daily_sharpe": float(daily_sharpe),
        "n_trading_days": int(len(day_pnl)),
        "day_concentration": float(day_conc),
        "day_concentration_pass": day_conc <= 0.70,
        "daily_pnl_list": day_pnl.to_dict("records"),
        # Per-side
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

    if hold_durations is not None and len(hold_durations) > 0:
        result["avg_hold_minutes"] = float(np.mean(hold_durations))
        result["median_hold_minutes"] = float(np.median(hold_durations))

    if exit_reasons is not None:
        result["exit_reasons"] = exit_reasons

    return result


# ═══════════════════════════════════════════════════════════════════
#  SECTION 11: REGIME ANALYSIS (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════


def full_regime_analysis(
    pnl_arr: np.ndarray,
    dates: np.ndarray,
    directions: np.ndarray,
    day_returns: Dict[str, float],
    bars_30m: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Full regime analysis as mandated by HC #428 R1.

    Includes:
      - Per-day P&L table
      - Green/Red/Flat day classification (ES close-to-close)
      - Per-regime Sharpe/Sortino/WR/PF
      - Regime gap test
      - Monthly breakdown
      - Per-day Sharpe/PF/WR
    """
    if len(pnl_arr) < 20:
        return {"error": "too few trades", "regime_gap_pass": False}

    # Classify days
    day_class = {}
    for d, ret in day_returns.items():
        if ret > 0.001:
            day_class[d] = "green"
        elif ret < -0.001:
            day_class[d] = "red"
        else:
            day_class[d] = "flat"

    # Build per-day results
    trade_df = pd.DataFrame({
        "pnl": pnl_arr,
        "date": dates,
        "direction": directions,
    })
    day_agg = trade_df.groupby("date").agg(
        daily_pnl=("pnl", "sum"),
        n_trades=("pnl", "count"),
        long_trades=("direction", lambda x: (x == 1).sum()),
        short_trades=("direction", lambda x: (x == -1).sum()),
        win_rate=("pnl", lambda x: (x > 0).mean()),
    ).reset_index()

    day_agg["regime"] = day_agg["date"].map(lambda d: day_class.get(d, "flat"))
    day_agg["month"] = day_agg["date"].str[:6]

    # Get ES day return from bar data
    day_close_first = bars_30m.groupby("date")["open"].first()
    day_close_last = bars_30m.groupby("date")["close"].last()
    day_ret_map = ((day_close_last - day_close_first) / day_close_first).to_dict()
    day_agg["es_day_return"] = day_agg["date"].map(lambda d: day_ret_map.get(d, 0.0))

    # Per-regime stats
    regime_results = {}
    regime_sharpes = {}

    for regime in ["green", "red", "flat"]:
        mask = day_agg["regime"] == regime
        if mask.sum() < 3:
            regime_results[regime] = {"n_days": int(mask.sum()), "skip": True}
            continue

        r_days = day_agg[mask]
        r_pnl = r_days["daily_pnl"].values

        r_sharpe = float(r_pnl.mean() / max(r_pnl.std(), 1e-6) * np.sqrt(252))
        r_downside = np.sqrt(np.mean(np.minimum(r_pnl, 0) ** 2))
        r_sortino = float(r_pnl.mean() / max(r_downside, 1e-6) * np.sqrt(252))
        r_wr = float(np.mean(r_pnl > 0))
        r_pf = float(np.sum(r_pnl[r_pnl > 0]) / max(-np.sum(r_pnl[r_pnl < 0]), 1e-6))

        regime_sharpes[regime] = r_sharpe
        regime_results[regime] = {
            "n_days": int(mask.sum()),
            "total_pnl_ticks": float(r_pnl.sum()),
            "avg_daily_pnl": float(r_pnl.mean()),
            "sharpe": r_sharpe,
            "sortino": r_sortino,
            "win_rate": r_wr,
            "profit_factor": r_pf,
            "total_trades": int(r_days["n_trades"].sum()),
        }

    # Regime gap test (HC #428)
    if "green" in regime_sharpes and "red" in regime_sharpes:
        s_green = regime_sharpes["green"]
        s_red = regime_sharpes["red"]
        denom = max(abs(s_green), abs(s_red), 1e-6)
        gap = abs(s_green - s_red) / denom
        regime_results["regime_gap"] = float(gap)
        regime_results["regime_gap_pass"] = gap <= 0.50
        regime_results["regime_gap_detail"] = (
            f"green_sharpe={s_green:.2f}, red_sharpe={s_red:.2f}, "
            f"gap={gap:.2f} {'PASS' if gap <= 0.50 else 'FAIL'}"
        )
    else:
        regime_results["regime_gap"] = float("nan")
        regime_results["regime_gap_pass"] = False
        regime_results["regime_gap_detail"] = "insufficient regime data"

    # Monthly breakdown
    monthly = {}
    for month, m_grp in day_agg.groupby("month"):
        m_pnl = m_grp["daily_pnl"].values
        if len(m_pnl) < 2:
            monthly[month] = {"n_days": len(m_pnl), "total_pnl": float(m_pnl.sum())}
            continue

        m_sharpe = float(m_pnl.mean() / max(m_pnl.std(), 1e-6) * np.sqrt(252))
        monthly[month] = {
            "n_days": len(m_pnl),
            "total_pnl_ticks": float(m_pnl.sum()),
            "total_pnl_dollars": float(m_pnl.sum() * ES_TICK_VALUE),
            "avg_daily_pnl": float(m_pnl.mean()),
            "sharpe": m_sharpe,
            "win_rate": float(np.mean(m_pnl > 0)),
            "total_trades": int(m_grp["n_trades"].sum()),
            "green_days": int((m_grp["regime"] == "green").sum()),
            "red_days": int((m_grp["regime"] == "red").sum()),
        }

    # Per-day P&L table (as list of dicts for output)
    per_day_table = []
    for _, row in day_agg.iterrows():
        per_day_table.append({
            "date": row["date"],
            "regime": row["regime"],
            "daily_pnl_ticks": float(row["daily_pnl"]),
            "daily_pnl_dollars": float(row["daily_pnl"] * ES_TICK_VALUE),
            "n_trades": int(row["n_trades"]),
            "long": int(row["long_trades"]),
            "short": int(row["short_trades"]),
            "win_rate": float(row["win_rate"]),
            "es_day_return": float(row["es_day_return"]),
        })

    return {
        "regimes": regime_results,
        "monthly": monthly,
        "per_day": per_day_table,
        "n_total_days": len(day_agg),
        "n_green_days": int((day_agg["regime"] == "green").sum()),
        "n_red_days": int((day_agg["regime"] == "red").sum()),
        "n_flat_days": int((day_agg["regime"] == "flat").sum()),
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 12: JSON HELPER
# ═══════════════════════════════════════════════════════════════════


def clean_for_json(obj):
    """Clean numpy types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: clean_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [clean_for_json(v) for v in obj]
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_,)):
        return bool(obj)
    elif isinstance(obj, pd.Timestamp):
        return str(obj)
    return obj


# ═══════════════════════════════════════════════════════════════════
#  SECTION 13: MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════


def run_trailing_stop_full_oot():
    """
    Run the full trailing stop analysis with maximum OOT coverage.

    Pipeline:
      1. Load minute bars, build 30-min bar features
      2. Walk-forward train entry model (30d train, 5d slide) — NO confluence
      3. For each entry threshold, reconstruct trades
      4. Run trailing stop sweep: entry_threshold x trailing_trigger (20 configs)
      5. Also run the v5-proven best trailing stop config
      6. Full regime analysis on the best config
      7. Save everything + MLflow
    """
    _import_lightgbm()

    # ── MLflow setup ──
    mlflow_active = False
    mlflow_run = None
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("trailing_stop_full_oot_v1")
        mlflow_run = mlflow.start_run(
            run_name=f"ts_full_oot_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        mlflow_active = True
        log.info(f"MLflow run started: {mlflow_run.info.run_id}")
    except Exception as e:
        log.warning(f"MLflow unavailable: {e} -- proceeding without tracking")

    t0 = time.time()

    # ════════════════════════════════════════
    #  PHASE 1: Load and prepare data
    # ════════════════════════════════════════
    log.info("=" * 70)
    log.info("PHASE 1: Loading raw minute bar data")
    log.info("=" * 70)

    minute_df = load_all_minute_bars()
    all_dates_raw = sorted(minute_df["date"].unique())

    log.info(f"Total trading days: {len(all_dates_raw)}")
    log.info(f"Date range: {all_dates_raw[0]} -> {all_dates_raw[-1]}")

    if mlflow_active:
        mlflow.log_param("n_trading_days", len(all_dates_raw))
        mlflow.log_param("date_range", f"{all_dates_raw[0]}-{all_dates_raw[-1]}")
        mlflow.log_param("train_days", TRAIN_DAYS)
        mlflow.log_param("slide_days", SLIDE_DAYS)
        mlflow.log_param("cost_rt_ticks", COST_RT_TICKS)
        mlflow.log_param("confluence", "NONE")
        mlflow.log_param("entry_thresholds", str(ENTRY_THRESHOLDS))
        mlflow.log_param("trailing_triggers", str(TRAILING_TRIGGERS))

    # ════════════════════════════════════════
    #  PHASE 1b: Build 30-min bar features
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 1b: Building 30-min bar features")
    log.info("=" * 70)

    bars_30m = aggregate_to_bars(minute_df, bar_size_min=30)
    bars_30m = add_rolling_features(bars_30m, bar_size_min=30)
    bars_30m = add_forward_labels(bars_30m, horizon_bars=1, horizon_label="30min",
                                   min_edge_ticks=2.5)

    feature_cols = get_feature_columns(bars_30m)
    log.info(f"30-min features: {len(feature_cols)}")

    if mlflow_active:
        mlflow.log_param("n_features_30m", len(feature_cols))

    # Per-day returns for regime classification
    day_close = bars_30m.groupby("date")["close"].last()
    day_returns_raw = day_close.pct_change()
    day_returns = day_returns_raw.to_dict()

    # ════════════════════════════════════════
    #  PHASE 2: Walk-forward entry model (30d train, NO confluence)
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: Walk-forward entry model training (30d train, NO confluence)")
    log.info("=" * 70)

    dates_30m = sorted(bars_30m["date"].unique())
    features_all = bars_30m[feature_cols].values.astype(np.float32)
    labels_all = bars_30m["fwd_ticks_30min"].values.astype(np.float32)
    dates_all = bars_30m["date"].values

    # Walk-forward: accumulate OOT predictions
    entry_preds = np.full(len(bars_30m), np.nan, dtype=np.float32)

    fold_idx = 0
    val_days = 5

    expected_oot_days = len(dates_30m) - TRAIN_DAYS
    log.info(f"Walk-Forward: {TRAIN_DAYS}d train, {val_days}d val, {SLIDE_DAYS}d slide")
    log.info(f"Dates available: {len(dates_30m)}")
    log.info(f"Expected OOT days: ~{expected_oot_days}")

    start_idx = TRAIN_DAYS
    for fold_start in range(start_idx, len(dates_30m) - val_days + 1, SLIDE_DAYS):
        fold_train_dates = dates_30m[fold_start - TRAIN_DAYS: fold_start]
        fold_val_dates = dates_30m[fold_start: fold_start + val_days]

        if len(fold_val_dates) < val_days:
            break

        fold_idx += 1

        train_mask = np.isin(dates_all, fold_train_dates)
        val_mask = np.isin(dates_all, fold_val_dates)

        if not leakage_audit(list(fold_train_dates), list(fold_val_dates), feature_cols):
            log.error(f"Fold {fold_idx}: leakage audit FAILED -- skip")
            continue

        # Robust scaling from train only
        tr = features_all[train_mask].copy()
        vl = features_all[val_mask].copy()
        med = np.nanmedian(tr, axis=0)
        q75 = np.nanpercentile(tr, 75, axis=0)
        q25 = np.nanpercentile(tr, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0
        tr = np.clip(np.nan_to_num((tr - med) / iqr, nan=0.0), -5, 5)
        vl = np.clip(np.nan_to_num((vl - med) / iqr, nan=0.0), -5, 5)

        _, _, val_preds = train_entry_model(
            tr, labels_all[train_mask],
            vl, labels_all[val_mask],
            feature_names=feature_cols,
            fold_idx=fold_idx,
        )
        entry_preds[val_mask] = val_preds

        if fold_idx % 5 == 0:
            n_valid = (~np.isnan(entry_preds)).sum()
            log.info(f"  Progress: fold {fold_idx}, valid predictions: {n_valid}")

    # Summary
    n_valid_preds = (~np.isnan(entry_preds)).sum()
    valid_dates = sorted(set(dates_all[~np.isnan(entry_preds)]))
    log.info(f"\nEntry model complete: {fold_idx} folds")
    log.info(f"  Valid predictions: {n_valid_preds} / {len(entry_preds)}")
    log.info(f"  OOT days with predictions: {len(valid_dates)}")
    log.info(f"  OOT coverage: {len(valid_dates)} / {len(all_dates_raw)} "
             f"({len(valid_dates)/len(all_dates_raw)*100:.1f}%)")

    if mlflow_active:
        mlflow.log_metric("n_folds", fold_idx)
        mlflow.log_metric("n_valid_preds", n_valid_preds)
        mlflow.log_metric("n_oot_days", len(valid_dates))
        mlflow.log_metric("oot_coverage_pct", len(valid_dates) / len(all_dates_raw) * 100)

    # Concat IC
    valid_mask = ~np.isnan(entry_preds) & ~np.isnan(labels_all)
    if valid_mask.sum() > 50:
        ic = np.corrcoef(entry_preds[valid_mask], labels_all[valid_mask])[0, 1]
        rank_ic = stats.spearmanr(entry_preds[valid_mask], labels_all[valid_mask]).correlation
        log.info(f"  Concat IC: {ic:.4f}, Rank IC: {rank_ic:.4f}")
        if mlflow_active:
            mlflow.log_metric("concat_ic", ic)
            mlflow.log_metric("rank_ic", rank_ic)

    # ════════════════════════════════════════
    #  PHASE 3: Entry model baseline evaluation
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 3: Entry model baseline (static hold, various thresholds)")
    log.info("=" * 70)

    baseline_results = {}
    for conf_pct in ENTRY_THRESHOLDS:
        trades = reconstruct_trades_no_confluence(
            bars_30m, minute_df, entry_preds,
            confidence_pct=conf_pct, max_hold_minutes=MAX_HOLD_MINUTES,
        )
        if not trades:
            log.warning(f"  No trades at {conf_pct:.0%} threshold")
            continue

        for hold_min in [15, 30]:
            sim = simulate_static(trades, hold_minutes=hold_min)
            if "error" not in sim:
                tag = f"static_{hold_min}m_top{int(conf_pct*100)}pct"
                baseline_results[tag] = sim
                log.info(
                    f"  {tag}: Sharpe={sim['sharpe']:.2f}, "
                    f"Sortino={sim['sortino']:.2f}, "
                    f"WR={sim['win_rate']:.1%}, PF={sim['profit_factor']:.2f}, "
                    f"N={sim['n_trades']}, Days={sim['n_trading_days']}"
                )
                if mlflow_active:
                    mlflow.log_metric(f"{tag}_sharpe", sim["sharpe"])
                    mlflow.log_metric(f"{tag}_n_days", sim["n_trading_days"])

    # ════════════════════════════════════════
    #  PHASE 4: Trailing stop sweep (20 configs)
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 4: Trailing stop sweep — entry_threshold x trailing_trigger")
    log.info(f"  Entry thresholds: {ENTRY_THRESHOLDS}")
    log.info(f"  Trailing triggers: {TRAILING_TRIGGERS}")
    log.info(f"  Total configs: {len(ENTRY_THRESHOLDS) * len(TRAILING_TRIGGERS)}")
    log.info("=" * 70)

    sweep_results = []
    best_sharpe = -999
    best_config = None
    best_trades = None
    best_result = None

    for conf_pct in ENTRY_THRESHOLDS:
        # Reconstruct trades once per threshold
        trades = reconstruct_trades_no_confluence(
            bars_30m, minute_df, entry_preds,
            confidence_pct=conf_pct, max_hold_minutes=MAX_HOLD_MINUTES,
        )
        if not trades:
            log.warning(f"  No trades at {conf_pct:.0%} threshold -- skip")
            continue

        for trigger in TRAILING_TRIGGERS:
            # Use v5-proven structure: trigger1=trigger, buf1=trigger-1,
            # trigger2=2*trigger, buf2=trigger/2, stale=4, max_hold=30
            buf1 = max(trigger - 1.0, 1.0)
            trigger2 = trigger * 2.0
            buf2 = trigger / 2.0

            result = simulate_trailing_stop(
                trades,
                trigger1=float(trigger),
                buf1=buf1,
                trigger2=trigger2,
                buf2=buf2,
                stale_bars=4,
                max_hold=30,
            )

            if "error" in result:
                continue

            config_tag = f"top{int(conf_pct*100)}_trig{trigger}"
            result["config"] = {
                "entry_threshold": conf_pct,
                "trigger1": float(trigger),
                "buf1": buf1,
                "trigger2": trigger2,
                "buf2": buf2,
                "stale_bars": 4,
                "max_hold": 30,
            }
            sweep_results.append(result)

            log.info(
                f"  {config_tag}: "
                f"Sharpe={result['sharpe']:.2f}, "
                f"Sortino={result['sortino']:.2f}, "
                f"WR={result['win_rate']:.1%}, "
                f"PF={result['profit_factor']:.2f}, "
                f"N={result['n_trades']}, "
                f"Days={result['n_trading_days']}, "
                f"$PnL={result['total_pnl_dollars']:.0f}"
            )

            if mlflow_active:
                mlflow.log_metric(f"sweep_{config_tag}_sharpe", result["sharpe"])
                mlflow.log_metric(f"sweep_{config_tag}_sortino", result["sortino"])
                mlflow.log_metric(f"sweep_{config_tag}_n_trades", result["n_trades"])
                mlflow.log_metric(f"sweep_{config_tag}_n_days", result["n_trading_days"])

            if result["sharpe"] > best_sharpe:
                best_sharpe = result["sharpe"]
                best_config = result["config"]
                best_trades = trades
                best_result = result

    # Also run the v5-proven best config at top 15% entry
    log.info("\n  Running v5-proven best config (trigger=3, buf=2, trigger2=6, buf2=1.5)")
    for conf_pct in ENTRY_THRESHOLDS:
        trades = reconstruct_trades_no_confluence(
            bars_30m, minute_df, entry_preds,
            confidence_pct=conf_pct, max_hold_minutes=MAX_HOLD_MINUTES,
        )
        if not trades:
            continue

        result = simulate_trailing_stop(
            trades,
            trigger1=BEST_TRAILING["trigger1"],
            buf1=BEST_TRAILING["buf1"],
            trigger2=BEST_TRAILING["trigger2"],
            buf2=BEST_TRAILING["buf2"],
            stale_bars=BEST_TRAILING["stale_bars"],
            max_hold=BEST_TRAILING["max_hold"],
        )
        if "error" in result:
            continue

        config_tag = f"v5best_top{int(conf_pct*100)}"
        result["config"] = {
            "entry_threshold": conf_pct,
            **BEST_TRAILING,
        }
        sweep_results.append(result)

        log.info(
            f"  {config_tag}: "
            f"Sharpe={result['sharpe']:.2f}, "
            f"Sortino={result['sortino']:.2f}, "
            f"WR={result['win_rate']:.1%}, "
            f"PF={result['profit_factor']:.2f}, "
            f"N={result['n_trades']}, "
            f"Days={result['n_trading_days']}, "
            f"$PnL={result['total_pnl_dollars']:.0f}"
        )

        if result["sharpe"] > best_sharpe:
            best_sharpe = result["sharpe"]
            best_config = result["config"]
            best_trades = trades
            best_result = result

    # Sort sweep by Sharpe
    sweep_results.sort(key=lambda x: x.get("sharpe", -999), reverse=True)

    log.info(f"\n{'=' * 70}")
    log.info("SWEEP LEADERBOARD (top 10)")
    log.info(f"{'=' * 70}")
    log.info(f"{'Rank':>4} {'Config':>30} {'Sharpe':>8} {'Sortino':>8} "
             f"{'WR':>6} {'PF':>6} {'N':>6} {'Days':>5} {'$PnL':>8}")
    log.info("-" * 90)
    for i, r in enumerate(sweep_results[:10]):
        cfg = r.get("config", {})
        cfg_str = f"top{int(cfg.get('entry_threshold', 0)*100)}_t{cfg.get('trigger1', 0)}"
        log.info(
            f"{i+1:4d} {cfg_str:>30} "
            f"{r['sharpe']:8.2f} {r['sortino']:8.2f} "
            f"{r['win_rate']:6.1%} {r['profit_factor']:6.2f} "
            f"{r['n_trades']:6d} {r['n_trading_days']:5d} "
            f"{r['total_pnl_dollars']:8.0f}"
        )

    if mlflow_active and best_result:
        mlflow.log_metric("best_sharpe", best_result["sharpe"])
        mlflow.log_metric("best_sortino", best_result["sortino"])
        mlflow.log_metric("best_n_trades", best_result["n_trades"])
        mlflow.log_metric("best_n_days", best_result["n_trading_days"])
        mlflow.log_metric("best_pnl_dollars", best_result["total_pnl_dollars"])
        mlflow.log_param("best_config", str(best_config))

    # ════════════════════════════════════════
    #  PHASE 5: Full regime analysis on best config
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 5: Full regime analysis on best config")
    log.info("=" * 70)

    if best_result is not None and best_trades is not None:
        log.info(f"  Best config: {best_config}")
        log.info(f"  Best Sharpe: {best_sharpe:.2f}")

        # Re-run best trailing stop to get trade-level arrays
        best_cfg = best_config
        pnls, dates, dirs, hold_durations = [], [], [], []

        for trade in best_trades:
            pnl_path = trade["pnl_path"]
            mfe_path = trade["mfe_path"]
            n_minutes = trade["n_minutes"]
            exited = False
            exit_minute = min(best_cfg.get("max_hold", 30), n_minutes - 1)
            trigger1 = best_cfg.get("trigger1", 3.0)
            buf1_val = best_cfg.get("buf1", 2.0)
            trigger2 = best_cfg.get("trigger2", 6.0)
            buf2_val = best_cfg.get("buf2", 1.5)
            stale_b = best_cfg.get("stale_bars", 4)
            max_h = best_cfg.get("max_hold", 30)

            for ckpt in range(MGMT_BAR_SIZE_MIN, n_minutes - 1, MGMT_BAR_SIZE_MIN):
                if ckpt > max_h:
                    exit_minute = min(max_h, n_minutes - 1)
                    exited = True
                    break

                mfe_now = mfe_path[ckpt]
                pnl_now = pnl_path[ckpt]

                bars_since = 0
                for lb in range(1, ckpt + 1):
                    if mfe_path[ckpt - lb] < mfe_now:
                        break
                    bars_since += 1
                bars_since_mgmt = bars_since / max(MGMT_BAR_SIZE_MIN, 1)

                if bars_since_mgmt >= stale_b and mfe_now > 1.0:
                    exit_minute = ckpt
                    exited = True
                    break

                floor = None
                if mfe_now >= trigger2:
                    floor = mfe_now - buf2_val
                elif mfe_now >= trigger1:
                    floor = mfe_now - buf1_val

                if floor is not None and pnl_now < floor:
                    exit_minute = ckpt
                    exited = True
                    break

            if not exited:
                exit_minute = min(max_h, n_minutes - 1)

            exit_minute = min(exit_minute, n_minutes - 1)
            trade_pnl = float(pnl_path[exit_minute]) - COST_RT_TICKS
            pnls.append(trade_pnl)
            dates.append(trade["date"])
            dirs.append(trade["direction"])
            hold_durations.append(exit_minute)

        pnl_arr = np.array(pnls)
        dates_arr = np.array(dates)
        dirs_arr = np.array(dirs)

        # Full regime analysis
        regime_analysis = full_regime_analysis(
            pnl_arr, dates_arr, dirs_arr, day_returns, bars_30m,
        )

        # Print regime results
        log.info(f"\n  OOT days with trades: {regime_analysis['n_total_days']}")
        log.info(f"  Green days: {regime_analysis['n_green_days']}, "
                 f"Red days: {regime_analysis['n_red_days']}, "
                 f"Flat days: {regime_analysis['n_flat_days']}")

        regimes = regime_analysis["regimes"]
        if "regime_gap_detail" in regimes:
            log.info(f"  REGIME GAP TEST: {regimes['regime_gap_detail']}")

        for regime in ["green", "red", "flat"]:
            if regime in regimes and "sharpe" in regimes[regime]:
                r = regimes[regime]
                log.info(
                    f"    {regime:6s}: Sharpe={r['sharpe']:.2f}, "
                    f"Sortino={r['sortino']:.2f}, "
                    f"WR={r['win_rate']:.1%}, "
                    f"PF={r['profit_factor']:.2f}, "
                    f"Days={r['n_days']}, "
                    f"Trades={r['total_trades']}, "
                    f"Avg daily PnL={r['avg_daily_pnl']:.2f} ticks"
                )

        # Monthly breakdown
        log.info(f"\n  MONTHLY BREAKDOWN:")
        log.info(f"  {'Month':>8} {'Days':>5} {'Trades':>7} {'PnL($)':>9} "
                 f"{'Sharpe':>8} {'WR':>6} {'Green':>5} {'Red':>5}")
        log.info("  " + "-" * 65)
        for month in sorted(regime_analysis["monthly"].keys()):
            m = regime_analysis["monthly"][month]
            log.info(
                f"  {month:>8} {m.get('n_days', 0):5d} "
                f"{m.get('total_trades', 0):7d} "
                f"{m.get('total_pnl_dollars', 0):9.0f} "
                f"{m.get('sharpe', 0):8.2f} "
                f"{m.get('win_rate', 0):6.1%} "
                f"{m.get('green_days', 0):5d} "
                f"{m.get('red_days', 0):5d}"
            )

        # Per-day P&L table (abbreviated)
        log.info(f"\n  PER-DAY P&L TABLE (first 20 days):")
        log.info(f"  {'Date':>10} {'Regime':>6} {'PnL($)':>8} {'Trades':>6} "
                 f"{'WR':>6} {'L':>3} {'S':>3} {'ES Ret':>8}")
        log.info("  " + "-" * 60)
        for day in regime_analysis["per_day"][:20]:
            log.info(
                f"  {day['date']:>10} {day['regime']:>6} "
                f"{day['daily_pnl_dollars']:8.0f} {day['n_trades']:6d} "
                f"{day['win_rate']:6.1%} {day['long']:3d} {day['short']:3d} "
                f"{day['es_day_return']:8.4f}"
            )
        if len(regime_analysis["per_day"]) > 20:
            log.info(f"  ... ({len(regime_analysis['per_day']) - 20} more days)")

        # Day concentration check (HC #344)
        if best_result.get("day_concentration_pass", False):
            log.info(f"\n  Day concentration: {best_result.get('day_concentration', 0):.3f} PASS")
        else:
            log.info(f"\n  Day concentration: {best_result.get('day_concentration', 0):.3f} FAIL")

        if mlflow_active:
            if "regime_gap" in regimes:
                mlflow.log_metric("regime_gap", regimes.get("regime_gap", 999))
                mlflow.log_metric("regime_gap_pass", int(regimes.get("regime_gap_pass", False)))
            for regime in ["green", "red", "flat"]:
                if regime in regimes and "sharpe" in regimes[regime]:
                    mlflow.log_metric(f"regime_{regime}_sharpe", regimes[regime]["sharpe"])
                    mlflow.log_metric(f"regime_{regime}_wr", regimes[regime]["win_rate"])
    else:
        log.warning("No best config found — sweep produced no valid results")
        regime_analysis = None

    # ════════════════════════════════════════
    #  PHASE 6: Save results
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 6: Saving results")
    log.info("=" * 70)

    # Save predictions
    np.savez_compressed(
        str(OUTPUT_DIR / "entry_predictions.npz"),
        entry_preds=entry_preds,
        labels_30m=labels_all,
        dates=dates_all,
    )
    log.info(f"  Saved predictions to {OUTPUT_DIR / 'entry_predictions.npz'}")

    # Save sweep results
    sweep_clean = clean_for_json(sweep_results)
    with open(str(OUTPUT_DIR / "sweep_results.json"), "w") as f:
        json.dump(sweep_clean, f, indent=2, default=str)

    # Save best config results
    if best_result is not None:
        best_output = clean_for_json({
            "best_config": best_config,
            "best_result": best_result,
            "regime_analysis": regime_analysis,
            "baselines": baseline_results,
        })
        with open(str(OUTPUT_DIR / "best_config_analysis.json"), "w") as f:
            json.dump(best_output, f, indent=2, default=str)

    if mlflow_active:
        try:
            mlflow.log_artifact(str(OUTPUT_DIR / "sweep_results.json"))
            if (OUTPUT_DIR / "best_config_analysis.json").exists():
                mlflow.log_artifact(str(OUTPUT_DIR / "best_config_analysis.json"))
        except Exception as e:
            log.warning(f"MLflow artifact logging failed: {e}")

    elapsed = time.time() - t0

    # ════════════════════════════════════════
    #  FINAL SUMMARY
    # ════════════════════════════════════════
    log.info(f"\n{'=' * 70}")
    log.info(f"TRAILING STOP FULL OOT v1 COMPLETE in {elapsed/60:.1f} minutes")
    log.info(f"{'=' * 70}")

    log.info(f"\n  Total trading days in data: {len(all_dates_raw)}")
    log.info(f"  OOT days with predictions: {len(valid_dates)}")
    if best_result:
        log.info(f"  OOT days with trades (best): {best_result.get('n_trading_days', 0)}")
        log.info(f"  OOT coverage improvement: 84 -> {best_result.get('n_trading_days', 0)} days")
    log.info(f"  Sweep configs tested: {len(sweep_results)}")

    if best_result:
        log.info(f"\n  BEST CONFIG: {best_config}")
        log.info(
            f"  Sharpe={best_result['sharpe']:.2f}, "
            f"Sortino={best_result['sortino']:.2f}, "
            f"WR={best_result['win_rate']:.1%}, "
            f"PF={best_result['profit_factor']:.2f}, "
            f"N={best_result['n_trades']}, "
            f"Days={best_result['n_trading_days']}, "
            f"$PnL={best_result['total_pnl_dollars']:.0f}"
        )

    if mlflow_active:
        mlflow.log_metric("total_runtime_min", elapsed / 60)
        mlflow.end_run()
        log.info(f"MLflow run ended: {mlflow_run.info.run_id}")

    return sweep_results


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("Trailing Stop Full OOT v1 starting...")
    log.info(f"  ROOT: {ROOT}")
    log.info(f"  OUTPUT: {OUTPUT_DIR}")
    log.info(f"  Cost: {COST_RT_TICKS} ticks RT")
    log.info(f"  Walk-forward: {TRAIN_DAYS}d train, {SLIDE_DAYS}d slide")
    log.info(f"  Entry: NO confluence gate, thresholds: {ENTRY_THRESHOLDS}")
    log.info(f"  Trailing triggers: {TRAILING_TRIGGERS}")
    log.info(f"  Sweep grid: {len(ENTRY_THRESHOLDS)} x {len(TRAILING_TRIGGERS)} "
             f"+ {len(ENTRY_THRESHOLDS)} v5-best = "
             f"{len(ENTRY_THRESHOLDS) * len(TRAILING_TRIGGERS) + len(ENTRY_THRESHOLDS)} configs")

    try:
        results = run_trailing_stop_full_oot()
    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        log.error(traceback.format_exc())
        try:
            import mlflow
            mlflow.end_run(status="FAILED")
        except Exception:
            pass
        sys.exit(1)
