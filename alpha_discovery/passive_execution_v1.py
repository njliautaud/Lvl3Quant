#!/usr/bin/env python3
"""
Passive Execution v1 — Fully Passive Entry + Passive Limit TP Exit
====================================================================

HYPOTHESIS: The 30-min LightGBM entry model has genuine signal (~1-2 ticks),
but market-order exits (trailing stop) consume the entire edge via 1-tick
slippage. If BOTH entry AND exit are passive limit orders, total cost drops
from ~1.376 ticks to ~0.376 ticks (commission only), potentially making
the signal profitable.

ENTRY (same FIFO v1):
  - Long: limit BUY at bid. Fill when bar low trades 1 tick through bid.
  - Short: limit SELL at ask. Fill when bar high trades 1 tick through ask.
  - Cancel if unfilled after cancel_window minutes.

EXIT (PASSIVE LIMIT TAKE-PROFIT + TIME STOP):
  - On fill, place limit TP order:
    - Long: limit SELL at (fill_price + TP_ticks * 0.25). Join the ask.
      Fill when bar HIGH trades 1 tick THROUGH our TP level.
    - Short: limit BUY at (fill_price - TP_ticks * 0.25). Join the bid.
      Fill when bar LOW trades 1 tick THROUGH our TP level.
  - Time stop: if TP not hit within max_hold minutes, exit at MARKET
    (1 tick adverse slippage).
  - COST BREAKDOWN:
    - Passive TP fill: 0.376 ticks (commission only)
    - Time stop market exit: 1.376 ticks (commission + 1 tick slippage)

SWEEP:
  - TP_ticks: [1, 2, 3, 4, 5]
  - cancel_window: [5, 10, 15, 30] minutes
  - max_hold: [15, 30, 60] minutes
  - entry_threshold: [0.05, 0.10, 0.15]
  - Total: 5 x 4 x 3 x 3 = 180 configs

CONSTRAINTS:
  - Walk-forward: 30d train, 5d slide — SLIDING only (HC #0)
  - FIFO fills only (HC #74)
  - Regime-agnostic gate: |Sharpe_green - Sharpe_red| / max < 0.50 (HC #428)
  - Day-concentration cap <= 0.70 (HC #344)
  - MLflow logging mandatory

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
from itertools import product
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
OUTPUT_DIR = ROOT / "output" / "passive_execution_v1"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [PASSIVE-V1] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "passive_execution_v1.log")),
    ],
)
log = logging.getLogger("PASSIVE-V1")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK = 0.25           # 1 tick = 0.25 points
ES_TICK_VALUE = 12.50    # 1 tick = $12.50
ES_RT_COMMISSION = 4.70  # Round-trip commission
ES_RT_COMMISSION_TICKS = 0.376  # $4.70 / $12.50
EXIT_SLIPPAGE_TICKS = 1.0  # Market exit = 1 tick adverse slippage

# ─────────────────────────────────────────────
#  WALK-FORWARD CONFIG
# ─────────────────────────────────────────────
TRAIN_DAYS = 30
SLIDE_DAYS = 5

# ─────────────────────────────────────────────
#  ENTRY BAR CONFIG
# ─────────────────────────────────────────────
ENTRY_BAR_SIZE_MIN = 30
MGMT_BAR_SIZE_MIN = 1

# ─────────────────────────────────────────────
#  SWEEP PARAMETERS
# ─────────────────────────────────────────────
TP_TICKS_SWEEP = [1, 2, 3, 4, 5]
CANCEL_WINDOW_SWEEP = [5, 10, 15, 30]  # minutes
MAX_HOLD_SWEEP = [15, 30, 60]  # minutes
ENTRY_THRESHOLD_SWEEP = [0.05, 0.10, 0.15]

# ─────────────────────────────────────────────
#  LGBM PARAMS (same as FIFO v1)
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

lgb = None


def _import_lightgbm():
    global lgb
    if lgb is not None:
        return
    import lightgbm as _lgb
    lgb = _lgb


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1: DATA LOADING (reused from FIFO v1)
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
#  SECTION 2: BAR AGGREGATION + FEATURES (reused from FIFO v1)
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
    log.info(f"Aggregated {len(result):,} {bar_size_min}min bars with {len(result.columns)} columns")
    return result


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: ROLLING CONTEXT + REGIME FEATURES (reused from FIFO v1)
# ═══════════════════════════════════════════════════════════════════


def add_rolling_features(df: pd.DataFrame, bar_size_min: int = 30) -> pd.DataFrame:
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


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: FEATURE COLUMN SELECTION (reused)
# ═══════════════════════════════════════════════════════════════════


def get_feature_columns(df: pd.DataFrame) -> List[str]:
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
#  SECTION 5: FORWARD LABELS (reused)
# ═══════════════════════════════════════════════════════════════════


def add_forward_labels(
    df: pd.DataFrame, horizon_bars: int, horizon_label: str,
    min_edge_ticks: float = 2.5,
) -> pd.DataFrame:
    df = df.sort_values("ts").reset_index(drop=True)
    fwd_close = df["close"].shift(-horizon_bars)
    fwd_return = fwd_close / df["close"] - 1
    fwd_ticks = (fwd_close - df["close"]) / 0.25

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
#  SECTION 6: LEAKAGE AUDIT (reused)
# ═══════════════════════════════════════════════════════════════════


def leakage_audit(
    train_dates: List[str], val_dates: List[str], feature_cols: List[str],
) -> bool:
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
#  SECTION 7: ENTRY MODEL TRAINING (reused)
# ═══════════════════════════════════════════════════════════════════


def train_entry_model(
    X_train, y_train, X_val, y_val, feature_names, fold_idx,
):
    _import_lightgbm()
    train_valid = ~np.isnan(y_train)
    val_valid = ~np.isnan(y_val)
    if train_valid.sum() < 50 or val_valid.sum() < 10:
        log.warning(f"  Entry fold {fold_idx}: too few valid -- skip")
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
#  SECTION 8: FIFO ENTRY FILL RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════


def reconstruct_entry_fills(
    bars_30m: pd.DataFrame,
    minute_df: pd.DataFrame,
    pred_30m: np.ndarray,
    confidence_pct: float,
    cancel_window_min: int,
) -> Tuple[List[Dict], Dict]:
    """
    Reconstruct FIFO entry fills. Returns trades with minute-level data
    attached for subsequent exit simulation.
    
    Each trade dict contains the fill info plus full minute-bar arrays
    from fill time through end of day for exit simulation.
    """
    valid_mask = ~np.isnan(pred_30m)
    valid_preds = pred_30m[valid_mask]
    if len(valid_preds) < 20:
        return [], {"error": "too few predictions"}

    upper_thresh = np.nanquantile(pred_30m[valid_mask], 1 - confidence_pct)
    lower_thresh = np.nanquantile(pred_30m[valid_mask], confidence_pct)

    # Build minute-level lookup
    minute_lookup = {}
    for date_str, grp in minute_df.groupby("date"):
        minute_lookup[date_str] = grp.sort_values("ts_minute").reset_index(drop=True)

    trades = []
    n_signals = 0
    n_filled = 0
    n_cancelled = 0
    n_no_data = 0
    fill_delays = []

    bars_ts = bars_30m["ts"].values
    bars_dates = bars_30m["date"].values
    bars_close = bars_30m["close"].values

    for i in range(len(bars_30m)):
        if np.isnan(pred_30m[i]):
            continue

        direction = 0
        if pred_30m[i] >= upper_thresh:
            direction = 1
        elif pred_30m[i] <= lower_thresh:
            direction = -1
        else:
            continue

        n_signals += 1
        date_str = bars_dates[i]
        signal_ts = pd.Timestamp(bars_ts[i])
        signal_price = bars_close[i]

        if date_str not in minute_lookup:
            n_no_data += 1
            continue

        day_minutes = minute_lookup[date_str]
        day_ts = day_minutes["ts_minute"].values
        day_high = day_minutes["high"].values
        day_low = day_minutes["low"].values

        limit_price = signal_price

        signal_bar_end = signal_ts + pd.Timedelta(minutes=ENTRY_BAR_SIZE_MIN)
        signal_bar_end_np = np.datetime64(signal_bar_end)
        cancel_ts = signal_ts + pd.Timedelta(minutes=cancel_window_min + ENTRY_BAR_SIZE_MIN)
        cancel_ts_np = np.datetime64(cancel_ts)

        fill_mask = (day_ts >= signal_bar_end_np) & (day_ts <= cancel_ts_np)
        fill_candidates = day_minutes[fill_mask]

        if len(fill_candidates) == 0:
            n_cancelled += 1
            continue

        filled = False
        fill_price = None
        fill_ts = None

        for j, (_, mbar) in enumerate(fill_candidates.iterrows()):
            if direction == 1:
                if mbar["low"] <= limit_price - ES_TICK:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar["ts_minute"]
                    fill_delays.append(j + 1)
                    break
            else:
                if mbar["high"] >= limit_price + ES_TICK:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar["ts_minute"]
                    fill_delays.append(j + 1)
                    break

        if not filled:
            n_cancelled += 1
            continue

        n_filled += 1

        # Attach ALL remaining minute bars from fill time onward (for exit sim)
        fill_ts_np = np.datetime64(fill_ts)
        remaining_mask = day_ts >= fill_ts_np
        remaining_minutes = day_minutes[remaining_mask]

        if len(remaining_minutes) < 2:
            n_filled -= 1
            n_cancelled += 1
            continue

        trade = {
            "idx": i,
            "date": date_str,
            "signal_ts": signal_ts,
            "fill_ts": pd.Timestamp(fill_ts),
            "fill_price": fill_price,
            "fill_delay_minutes": fill_delays[-1],
            "direction": direction,
            "pred_30m": float(pred_30m[i]),
            # Minute-level data for exit simulation
            "prices_close": remaining_minutes["close"].values.copy(),
            "prices_high": remaining_minutes["high"].values.copy(),
            "prices_low": remaining_minutes["low"].values.copy(),
            "times": remaining_minutes["ts_minute"].values.copy(),
            "n_remaining_minutes": len(remaining_minutes),
        }
        trades.append(trade)

    fill_stats = {
        "n_signals": n_signals,
        "n_filled": n_filled,
        "n_cancelled": n_cancelled,
        "n_no_data": n_no_data,
        "fill_rate": n_filled / max(n_signals, 1),
        "cancel_rate": n_cancelled / max(n_signals, 1),
        "avg_fill_delay_min": float(np.mean(fill_delays)) if fill_delays else 0,
        "median_fill_delay_min": float(np.median(fill_delays)) if fill_delays else 0,
        "p90_fill_delay_min": float(np.percentile(fill_delays, 90)) if fill_delays else 0,
    }

    log.info(f"  Entry fills (top/bottom {confidence_pct:.0%}, cancel={cancel_window_min}m): "
             f"signals={n_signals}, filled={n_filled} ({fill_stats['fill_rate']:.1%}), "
             f"cancelled={n_cancelled}")

    return trades, fill_stats


# ═══════════════════════════════════════════════════════════════════
#  SECTION 9: PASSIVE TP EXIT SIMULATION
# ═══════════════════════════════════════════════════════════════════


def simulate_passive_tp_exit(
    trades: List[Dict],
    tp_ticks: int,
    max_hold_minutes: int,
) -> Tuple[List[float], List[str], List[int], Dict]:
    """
    Simulate passive limit take-profit exit with time stop fallback.

    For each filled trade:
      - Place limit TP at fill_price + direction * tp_ticks * ES_TICK
      - TP fills passively: bar must trade 1 tick THROUGH our TP level
        (same FIFO back-of-queue logic as entry)
      - If TP not hit within max_hold minutes: exit at market (1 tick slippage)

    Returns:
      pnls: list of trade PnL in ticks (after costs)
      dates: list of trade dates
      directions: list of trade directions
      exit_stats: dict with TP fill rate, passive exit fraction, etc.
    """
    pnls = []
    dates = []
    directions = []
    hold_durations = []

    n_tp_fills = 0
    n_time_stops = 0
    n_skipped = 0

    for trade in trades:
        direction = trade["direction"]
        fill_price = trade["fill_price"]
        prices_close = trade["prices_close"]
        prices_high = trade["prices_high"]
        prices_low = trade["prices_low"]
        n_remaining = trade["n_remaining_minutes"]

        # Determine TP price level
        tp_price = fill_price + direction * tp_ticks * ES_TICK

        # Check each minute bar for TP fill (FIFO: 1 tick through)
        tp_filled = False
        tp_minute = None
        max_check = min(max_hold_minutes, n_remaining)

        for m in range(1, max_check):  # Start at minute 1 (after fill bar)
            if direction == 1:
                # Long TP: selling at tp_price (ask side).
                # Fill when HIGH trades 1 tick THROUGH our TP level.
                if prices_high[m] >= tp_price + ES_TICK:
                    tp_filled = True
                    tp_minute = m
                    break
            else:
                # Short TP: buying at tp_price (bid side).
                # Fill when LOW trades 1 tick THROUGH our TP level.
                if prices_low[m] <= tp_price - ES_TICK:
                    tp_filled = True
                    tp_minute = m
                    break

        if tp_filled:
            # Passive TP fill: exit at our limit price, NO slippage
            # Cost = commission only (0.376 ticks)
            raw_pnl_ticks = (tp_price - fill_price) / ES_TICK * direction
            trade_pnl = raw_pnl_ticks - ES_RT_COMMISSION_TICKS
            hold_min = tp_minute
            n_tp_fills += 1
        else:
            # Time stop: exit at market with 1 tick slippage
            exit_minute = min(max_hold_minutes, n_remaining - 1)
            exit_minute = max(exit_minute, 1)  # at least 1 minute
            exit_close = prices_close[exit_minute]

            if direction == 1:
                exit_fill = exit_close - ES_TICK  # 1 tick slippage down
            else:
                exit_fill = exit_close + ES_TICK  # 1 tick slippage up

            raw_pnl_ticks = (exit_fill - fill_price) / ES_TICK * direction
            trade_pnl = raw_pnl_ticks - ES_RT_COMMISSION_TICKS
            hold_min = exit_minute
            n_time_stops += 1

        pnls.append(trade_pnl)
        dates.append(trade["date"])
        directions.append(direction)
        hold_durations.append(hold_min)

    total_exits = n_tp_fills + n_time_stops
    exit_stats = {
        "n_tp_fills": n_tp_fills,
        "n_time_stops": n_time_stops,
        "tp_fill_rate": n_tp_fills / max(total_exits, 1),
        "passive_exit_fraction": n_tp_fills / max(total_exits, 1),
        "time_stop_fraction": n_time_stops / max(total_exits, 1),
        "avg_hold_minutes": float(np.mean(hold_durations)) if hold_durations else 0,
        "median_hold_minutes": float(np.median(hold_durations)) if hold_durations else 0,
    }

    return pnls, dates, directions, exit_stats


# ═══════════════════════════════════════════════════════════════════
#  SECTION 10: DAILY TRADE STATISTICS
# ═══════════════════════════════════════════════════════════════════


def compute_daily_stats(
    pnl_arr: np.ndarray,
    dates: np.ndarray,
    directions: np.ndarray,
    label: str,
    exit_stats: Optional[Dict] = None,
    fill_stats: Optional[Dict] = None,
) -> Dict:
    """Compute daily Sharpe, Sortino, WR, PF, regime stats."""
    if len(pnl_arr) == 0:
        return {"error": "no trades"}

    wr = np.mean(pnl_arr > 0)
    pf = np.sum(pnl_arr[pnl_arr > 0]) / max(-np.sum(pnl_arr[pnl_arr < 0]), 1e-6)

    long_mask = directions == 1
    short_mask = directions == -1
    long_pnl = pnl_arr[long_mask] if long_mask.any() else np.array([])
    short_pnl = pnl_arr[short_mask] if short_mask.any() else np.array([])

    trade_df = pd.DataFrame({"pnl": pnl_arr, "date": dates})
    day_pnl = trade_df.groupby("date")["pnl"].agg(["sum", "count"]).reset_index()
    day_pnl.columns = ["date", "daily_pnl", "daily_trades"]

    daily_sharpe = 0.0
    daily_sortino = 0.0
    if len(day_pnl) > 2:
        daily_mean = day_pnl["daily_pnl"].mean()
        daily_std = day_pnl["daily_pnl"].std()
        daily_sharpe = float(daily_mean / max(daily_std, 1e-6) * np.sqrt(252))
        daily_downside = np.sqrt(np.mean(np.minimum(day_pnl["daily_pnl"].values, 0) ** 2))
        daily_sortino = float(daily_mean / max(daily_downside, 1e-6) * np.sqrt(252))

    cum_pnl = np.cumsum(pnl_arr)
    max_dd = float(np.min(cum_pnl - np.maximum.accumulate(cum_pnl)))

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
        "daily_sharpe": float(daily_sharpe),
        "daily_sortino": float(daily_sortino),
        "win_rate": float(wr),
        "profit_factor": float(pf),
        "max_dd_ticks": float(max_dd),
        "max_dd_dollars": float(max_dd * ES_TICK_VALUE),
        "n_trading_days": int(len(day_pnl)),
        "trades_per_day": float(len(pnl_arr) / max(len(day_pnl), 1)),
        "day_concentration": float(day_conc),
        "day_concentration_pass": day_conc <= 0.70,
        # Per-side
        "long_trades": int(len(long_pnl)),
        "long_wr": float(np.mean(long_pnl > 0)) if len(long_pnl) > 0 else 0,
        "long_avg": float(long_pnl.mean()) if len(long_pnl) > 0 else 0,
        "short_trades": int(len(short_pnl)),
        "short_wr": float(np.mean(short_pnl > 0)) if len(short_pnl) > 0 else 0,
        "short_avg": float(short_pnl.mean()) if len(short_pnl) > 0 else 0,
    }

    if exit_stats:
        result["exit_stats"] = exit_stats
    if fill_stats:
        result["fill_stats"] = fill_stats

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
    if len(pnl_arr) < 20:
        return {"error": "too few trades", "regime_gap_pass": False}

    day_class = {}
    for d, ret in day_returns.items():
        if ret > 0.001:
            day_class[d] = "green"
        elif ret < -0.001:
            day_class[d] = "red"
        else:
            day_class[d] = "flat"

    trade_df = pd.DataFrame({"pnl": pnl_arr, "date": dates, "direction": directions})
    day_agg = trade_df.groupby("date").agg(
        daily_pnl=("pnl", "sum"),
        n_trades=("pnl", "count"),
        long_trades=("direction", lambda x: (x == 1).sum()),
        short_trades=("direction", lambda x: (x == -1).sum()),
        win_rate=("pnl", lambda x: (x > 0).mean()),
    ).reset_index()

    day_agg["regime"] = day_agg["date"].map(lambda d: day_class.get(d, "flat"))
    day_agg["month"] = day_agg["date"].str[:6]

    day_close_first = bars_30m.groupby("date")["open"].first()
    day_close_last = bars_30m.groupby("date")["close"].last()
    day_ret_map = ((day_close_last - day_close_first) / day_close_first).to_dict()
    day_agg["es_day_return"] = day_agg["date"].map(lambda d: day_ret_map.get(d, 0.0))

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
        }

    per_day_table = []
    for _, row in day_agg.iterrows():
        per_day_table.append({
            "date": row["date"],
            "regime": row["regime"],
            "daily_pnl_ticks": float(row["daily_pnl"]),
            "daily_pnl_dollars": float(row["daily_pnl"] * ES_TICK_VALUE),
            "n_trades": int(row["n_trades"]),
            "win_rate": float(row["win_rate"]),
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


def run_passive_execution():
    """
    Run passive execution experiment.

    Pipeline:
      1. Load minute bars, build 30-min bar features
      2. Walk-forward train entry model (30d train, 5d slide)
      3. For each (entry_threshold, cancel_window) combo:
         a. FIFO entry fill reconstruction
         b. For each (tp_ticks, max_hold) combo:
            - Simulate passive TP exit
            - Compute daily stats
      4. Rank all 180 configs by daily Sharpe
      5. Full regime analysis on top configs
      6. Save everything + MLflow
    """
    _import_lightgbm()

    # ── MLflow setup ──
    mlflow_active = False
    mlflow_run = None
    try:
        import mlflow
        mlflow.set_tracking_uri("http://neptune:5000")
        mlflow.set_experiment("passive_execution_v1")
        mlflow_run = mlflow.start_run(
            run_name=f"passive_exec_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        mlflow.log_param("entry_model", "FIFO_passive_limit")
        mlflow.log_param("exit_model", "passive_limit_TP_plus_time_stop")
        mlflow.log_param("commission_ticks", ES_RT_COMMISSION_TICKS)
        mlflow.log_param("tp_ticks_sweep", str(TP_TICKS_SWEEP))
        mlflow.log_param("cancel_window_sweep", str(CANCEL_WINDOW_SWEEP))
        mlflow.log_param("max_hold_sweep", str(MAX_HOLD_SWEEP))
        mlflow.log_param("entry_threshold_sweep", str(ENTRY_THRESHOLD_SWEEP))
        mlflow.log_param("total_configs", len(TP_TICKS_SWEEP) * len(CANCEL_WINDOW_SWEEP) * len(MAX_HOLD_SWEEP) * len(ENTRY_THRESHOLD_SWEEP))

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
    #  PHASE 2: Walk-forward entry model
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: Walk-forward entry model training (30d train, SLIDING)")
    log.info("=" * 70)

    dates_30m = sorted(bars_30m["date"].unique())
    features_all = bars_30m[feature_cols].values.astype(np.float32)
    labels_all = bars_30m["fwd_ticks_30min"].values.astype(np.float32)
    dates_all = bars_30m["date"].values

    entry_preds = np.full(len(bars_30m), np.nan, dtype=np.float32)

    fold_idx = 0
    val_days = 5

    log.info(f"Walk-Forward: {TRAIN_DAYS}d train, {val_days}d val, {SLIDE_DAYS}d slide")
    log.info(f"Dates available: {len(dates_30m)}")

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

        # Robust scaling
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

    if mlflow_active:
        mlflow.log_metric("n_folds", fold_idx)
        mlflow.log_metric("n_valid_preds", n_valid_preds)
        mlflow.log_metric("n_oot_days", len(valid_dates))

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
    #  PHASE 3: SWEEP — Passive TP execution configs
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 3: Sweeping passive TP execution configs (180 total)")
    log.info("=" * 70)

    all_results = []
    config_count = 0
    total_configs = len(TP_TICKS_SWEEP) * len(CANCEL_WINDOW_SWEEP) * len(MAX_HOLD_SWEEP) * len(ENTRY_THRESHOLD_SWEEP)

    # Cache entry fills per (entry_threshold, cancel_window) combo to avoid redundant recomputation
    entry_fill_cache = {}

    for entry_thresh in ENTRY_THRESHOLD_SWEEP:
        for cancel_window in CANCEL_WINDOW_SWEEP:
            cache_key = (entry_thresh, cancel_window)

            if cache_key not in entry_fill_cache:
                trades, fill_stats = reconstruct_entry_fills(
                    bars_30m, minute_df, entry_preds,
                    confidence_pct=entry_thresh,
                    cancel_window_min=cancel_window,
                )
                entry_fill_cache[cache_key] = (trades, fill_stats)
            else:
                trades, fill_stats = entry_fill_cache[cache_key]

            if not trades:
                log.warning(f"  No fills for thresh={entry_thresh}, cancel={cancel_window}")
                for tp_ticks in TP_TICKS_SWEEP:
                    for max_hold in MAX_HOLD_SWEEP:
                        config_count += 1
                continue

            for tp_ticks in TP_TICKS_SWEEP:
                for max_hold in MAX_HOLD_SWEEP:
                    config_count += 1

                    # Simulate passive TP exit
                    pnls, trade_dates, trade_dirs, exit_stats = simulate_passive_tp_exit(
                        trades, tp_ticks=tp_ticks, max_hold_minutes=max_hold,
                    )

                    if len(pnls) < 10:
                        continue

                    pnl_arr = np.array(pnls)
                    dates_arr = np.array(trade_dates)
                    dirs_arr = np.array(trade_dirs)

                    label = f"tp{tp_ticks}_cw{cancel_window}_mh{max_hold}_et{int(entry_thresh*100)}"

                    result = compute_daily_stats(
                        pnl_arr, dates_arr, dirs_arr, label,
                        exit_stats=exit_stats, fill_stats=fill_stats,
                    )

                    if "error" in result:
                        continue

                    result["config"] = {
                        "tp_ticks": tp_ticks,
                        "cancel_window": cancel_window,
                        "max_hold": max_hold,
                        "entry_threshold": entry_thresh,
                    }

                    all_results.append(result)

                    if config_count % 30 == 0:
                        log.info(f"  Progress: {config_count}/{total_configs} configs done")

    log.info(f"\n  Sweep complete: {config_count} configs evaluated, "
             f"{len(all_results)} produced valid results")

    # Sort by daily Sharpe
    all_results.sort(key=lambda x: x.get("daily_sharpe", -999), reverse=True)

    # ════════════════════════════════════════
    #  PHASE 4: LEADERBOARD
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PASSIVE EXECUTION LEADERBOARD (top 20)")
    log.info("=" * 70)
    log.info(
        f"{'Rk':>3} {'TP':>3} {'CW':>4} {'MH':>4} {'ET':>4} "
        f"{'Sharpe':>8} {'Sort':>8} {'WR':>6} {'PF':>6} "
        f"{'N':>5} {'Days':>5} {'$PnL':>9} {'TP%':>6} {'Fill%':>6} {'DayConc':>7}"
    )
    log.info("-" * 105)

    for i, r in enumerate(all_results[:20]):
        cfg = r.get("config", {})
        es = r.get("exit_stats", {})
        fs = r.get("fill_stats", {})
        log.info(
            f"{i+1:3d} "
            f"{cfg.get('tp_ticks', 0):3d} "
            f"{cfg.get('cancel_window', 0):4d} "
            f"{cfg.get('max_hold', 0):4d} "
            f"{int(cfg.get('entry_threshold', 0)*100):3d}% "
            f"{r['daily_sharpe']:8.2f} {r['daily_sortino']:8.2f} "
            f"{r['win_rate']:6.1%} {r['profit_factor']:6.2f} "
            f"{r['n_trades']:5d} {r['n_trading_days']:5d} "
            f"{r['total_pnl_dollars']:9.0f} "
            f"{es.get('tp_fill_rate', 0):6.1%} "
            f"{fs.get('fill_rate', 0):6.1%} "
            f"{r['day_concentration']:7.2f}"
        )

    # Also log worst 5 to see if there's systematic failure
    if len(all_results) > 25:
        log.info("\nBOTTOM 5 CONFIGS:")
        for r in all_results[-5:]:
            cfg = r.get("config", {})
            es = r.get("exit_stats", {})
            log.info(
                f"  tp={cfg.get('tp_ticks')}, cw={cfg.get('cancel_window')}, "
                f"mh={cfg.get('max_hold')}, et={cfg.get('entry_threshold')}: "
                f"Sharpe={r['daily_sharpe']:.2f}, TP%={es.get('tp_fill_rate', 0):.1%}, "
                f"N={r['n_trades']}, $PnL={r['total_pnl_dollars']:.0f}"
            )

    # ════════════════════════════════════════
    #  PHASE 5: Regime analysis on top 3 configs
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 5: Regime analysis on top configs")
    log.info("=" * 70)

    regime_results_top = {}
    for i, r in enumerate(all_results[:3]):
        cfg = r.get("config", {})
        label = r["label"]

        # Reconstruct trades for regime analysis
        cache_key = (cfg["entry_threshold"], cfg["cancel_window"])
        trades, fill_stats = entry_fill_cache[cache_key]

        pnls, trade_dates, trade_dirs, exit_stats = simulate_passive_tp_exit(
            trades, tp_ticks=cfg["tp_ticks"], max_hold_minutes=cfg["max_hold"],
        )

        pnl_arr = np.array(pnls)
        dates_arr = np.array(trade_dates)
        dirs_arr = np.array(trade_dirs)

        regime = full_regime_analysis(pnl_arr, dates_arr, dirs_arr, day_returns, bars_30m)
        regime_results_top[label] = regime

        log.info(f"\n  Config #{i+1}: {label}")
        log.info(f"    Daily Sharpe={r['daily_sharpe']:.2f}, "
                 f"Sortino={r['daily_sortino']:.2f}, "
                 f"WR={r['win_rate']:.1%}, PF={r['profit_factor']:.2f}")
        log.info(f"    N={r['n_trades']}, Days={r['n_trading_days']}, "
                 f"$PnL={r['total_pnl_dollars']:.0f}")
        log.info(f"    TP fill rate: {exit_stats.get('tp_fill_rate', 0):.1%} "
                 f"(passive exit fraction)")
        log.info(f"    Entry fill rate: {fill_stats.get('fill_rate', 0):.1%}")

        regimes = regime.get("regimes", {})
        if "regime_gap_detail" in regimes:
            log.info(f"    REGIME GAP: {regimes['regime_gap_detail']}")

        for reg in ["green", "red", "flat"]:
            if reg in regimes and "sharpe" in regimes[reg]:
                rr = regimes[reg]
                log.info(
                    f"      {reg:6s}: Sharpe={rr['sharpe']:.2f}, "
                    f"WR={rr['win_rate']:.1%}, PF={rr['profit_factor']:.2f}, "
                    f"Days={rr['n_days']}, Trades={rr['total_trades']}"
                )

        # Monthly
        if "monthly" in regime:
            log.info(f"    MONTHLY:")
            for month in sorted(regime["monthly"].keys()):
                m = regime["monthly"][month]
                log.info(
                    f"      {month}: days={m.get('n_days')}, "
                    f"pnl=${m.get('total_pnl_dollars', 0):.0f}, "
                    f"sharpe={m.get('sharpe', 0):.2f}, "
                    f"wr={m.get('win_rate', 0):.1%}"
                )

    # ════════════════════════════════════════
    #  PHASE 6: Aggregate fill statistics analysis
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 6: Fill statistics analysis")
    log.info("=" * 70)

    log.info("\n  TP FILL RATE BY TP_TICKS (averaging across other params):")
    for tp in TP_TICKS_SWEEP:
        tp_results = [r for r in all_results if r.get("config", {}).get("tp_ticks") == tp]
        if tp_results:
            avg_tp_rate = np.mean([r.get("exit_stats", {}).get("tp_fill_rate", 0) for r in tp_results])
            avg_sharpe = np.mean([r["daily_sharpe"] for r in tp_results])
            avg_wr = np.mean([r["win_rate"] for r in tp_results])
            log.info(f"    TP={tp} ticks: avg_tp_fill_rate={avg_tp_rate:.1%}, "
                     f"avg_sharpe={avg_sharpe:.2f}, avg_wr={avg_wr:.1%}, "
                     f"n_configs={len(tp_results)}")

    log.info("\n  ENTRY FILL RATE BY CANCEL_WINDOW:")
    for cw in CANCEL_WINDOW_SWEEP:
        cw_results = [r for r in all_results if r.get("config", {}).get("cancel_window") == cw]
        if cw_results:
            avg_fill = np.mean([r.get("fill_stats", {}).get("fill_rate", 0) for r in cw_results])
            avg_sharpe = np.mean([r["daily_sharpe"] for r in cw_results])
            log.info(f"    CW={cw}min: avg_entry_fill_rate={avg_fill:.1%}, "
                     f"avg_sharpe={avg_sharpe:.2f}, n_configs={len(cw_results)}")

    log.info("\n  AVG SHARPE BY ENTRY THRESHOLD:")
    for et in ENTRY_THRESHOLD_SWEEP:
        et_results = [r for r in all_results if r.get("config", {}).get("entry_threshold") == et]
        if et_results:
            avg_sharpe = np.mean([r["daily_sharpe"] for r in et_results])
            avg_pnl = np.mean([r["total_pnl_dollars"] for r in et_results])
            log.info(f"    ET={et:.0%}: avg_sharpe={avg_sharpe:.2f}, "
                     f"avg_$pnl={avg_pnl:.0f}, n_configs={len(et_results)}")

    # ════════════════════════════════════════
    #  PHASE 7: Save results + MLflow
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 7: Saving results")
    log.info("=" * 70)

    # Save predictions
    np.savez_compressed(
        str(OUTPUT_DIR / "entry_predictions.npz"),
        entry_preds=entry_preds,
        labels_30m=labels_all,
        dates=dates_all,
    )

    # Save all results (strip daily_pnl_list to save space)
    results_save = []
    for r in all_results:
        r_copy = {k: v for k, v in r.items() if k != "daily_pnl_list"}
        results_save.append(r_copy)

    results_clean = clean_for_json(results_save)
    with open(str(OUTPUT_DIR / "all_results.json"), "w") as f:
        json.dump(results_clean, f, indent=2, default=str)

    # Save top configs with regime analysis
    top_output = clean_for_json({
        "top_configs": results_save[:10],
        "regime_analyses": regime_results_top,
    })
    with open(str(OUTPUT_DIR / "top_configs_analysis.json"), "w") as f:
        json.dump(top_output, f, indent=2, default=str)

    # Save fill statistics summary
    fill_summary = {}
    for key, (trades, fs) in entry_fill_cache.items():
        fill_summary[f"et{int(key[0]*100)}_cw{key[1]}"] = clean_for_json(fs)
    with open(str(OUTPUT_DIR / "fill_statistics.json"), "w") as f:
        json.dump(fill_summary, f, indent=2, default=str)

    if mlflow_active:
        try:
            # Log best config metrics
            if all_results:
                best = all_results[0]
                best_cfg = best.get("config", {})
                best_es = best.get("exit_stats", {})
                best_fs = best.get("fill_stats", {})

                mlflow.log_metric("best_daily_sharpe", best["daily_sharpe"])
                mlflow.log_metric("best_daily_sortino", best["daily_sortino"])
                mlflow.log_metric("best_win_rate", best["win_rate"])
                mlflow.log_metric("best_profit_factor", best["profit_factor"])
                mlflow.log_metric("best_n_trades", best["n_trades"])
                mlflow.log_metric("best_n_days", best["n_trading_days"])
                mlflow.log_metric("best_pnl_dollars", best["total_pnl_dollars"])
                mlflow.log_metric("best_tp_fill_rate", best_es.get("tp_fill_rate", 0))
                mlflow.log_metric("best_entry_fill_rate", best_fs.get("fill_rate", 0))
                mlflow.log_metric("best_passive_exit_frac", best_es.get("passive_exit_fraction", 0))
                mlflow.log_param("best_tp_ticks", best_cfg.get("tp_ticks"))
                mlflow.log_param("best_cancel_window", best_cfg.get("cancel_window"))
                mlflow.log_param("best_max_hold", best_cfg.get("max_hold"))
                mlflow.log_param("best_entry_threshold", best_cfg.get("entry_threshold"))

                # Log regime analysis for best config
                if best["label"] in regime_results_top:
                    regime = regime_results_top[best["label"]]
                    regimes = regime.get("regimes", {})
                    if "regime_gap" in regimes:
                        mlflow.log_metric("best_regime_gap", regimes.get("regime_gap", 999))
                        mlflow.log_metric("best_regime_gap_pass", int(regimes.get("regime_gap_pass", False)))
                    for reg in ["green", "red", "flat"]:
                        if reg in regimes and "sharpe" in regimes[reg]:
                            mlflow.log_metric(f"best_regime_{reg}_sharpe", regimes[reg]["sharpe"])

            # Log artifacts
            for artifact_name in ["all_results.json", "top_configs_analysis.json", "fill_statistics.json"]:
                artifact_path = OUTPUT_DIR / artifact_name
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))
        except Exception as e:
            log.warning(f"MLflow logging failed: {e}")

    elapsed = time.time() - t0

    # ════════════════════════════════════════
    #  FINAL SUMMARY
    # ════════════════════════════════════════
    log.info(f"\n{'=' * 70}")
    log.info(f"PASSIVE EXECUTION v1 COMPLETE in {elapsed/60:.1f} minutes")
    log.info(f"{'=' * 70}")

    log.info(f"\n  ENTRY: Passive limit (FIFO back-of-queue, 1-tick-through)")
    log.info(f"  EXIT:  Passive limit TP (FIFO) + time stop fallback (market)")
    log.info(f"  COST:  Passive TP = {ES_RT_COMMISSION_TICKS} ticks | "
             f"Time stop = {ES_RT_COMMISSION_TICKS + EXIT_SLIPPAGE_TICKS} ticks")

    log.info(f"\n  Total configs swept: {total_configs}")
    log.info(f"  Valid results: {len(all_results)}")
    log.info(f"  OOT days: {len(valid_dates)}")

    if all_results:
        best = all_results[0]
        best_cfg = best.get("config", {})
        best_es = best.get("exit_stats", {})
        log.info(f"\n  === BEST CONFIG ===")
        log.info(f"  TP={best_cfg.get('tp_ticks')} ticks, "
                 f"CancelWindow={best_cfg.get('cancel_window')}min, "
                 f"MaxHold={best_cfg.get('max_hold')}min, "
                 f"EntryThresh={best_cfg.get('entry_threshold')}")
        log.info(
            f"  Sharpe={best['daily_sharpe']:.2f}, "
            f"Sortino={best['daily_sortino']:.2f}, "
            f"WR={best['win_rate']:.1%}, "
            f"PF={best['profit_factor']:.2f}"
        )
        log.info(
            f"  Trades={best['n_trades']}, "
            f"Days={best['n_trading_days']}, "
            f"$PnL={best['total_pnl_dollars']:.0f}"
        )
        log.info(
            f"  TP fill rate={best_es.get('tp_fill_rate', 0):.1%} (passive exit fraction)"
        )
        log.info(
            f"  CRITICAL INSIGHT: {best_es.get('tp_fill_rate', 0)*100:.0f}% of exits "
            f"are passive (cost={ES_RT_COMMISSION_TICKS} ticks), "
            f"{best_es.get('time_stop_fraction', 0)*100:.0f}% forced market "
            f"(cost={ES_RT_COMMISSION_TICKS + EXIT_SLIPPAGE_TICKS} ticks)"
        )

    if mlflow_active:
        mlflow.log_metric("total_runtime_min", elapsed / 60)
        mlflow.end_run()
        log.info(f"MLflow run ended: {mlflow_run.info.run_id}")

    return all_results


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("Passive Execution v1 starting...")
    log.info(f"  ROOT: {ROOT}")
    log.info(f"  OUTPUT: {OUTPUT_DIR}")
    log.info(f"  ENTRY: Passive limit (FIFO back-of-queue)")
    log.info(f"  EXIT: Passive limit TP + time stop fallback")
    log.info(f"  Commission: {ES_RT_COMMISSION_TICKS} ticks RT")
    log.info(f"  Sweep: TP={TP_TICKS_SWEEP}, CW={CANCEL_WINDOW_SWEEP}, "
             f"MH={MAX_HOLD_SWEEP}, ET={ENTRY_THRESHOLD_SWEEP}")
    total = len(TP_TICKS_SWEEP) * len(CANCEL_WINDOW_SWEEP) * len(MAX_HOLD_SWEEP) * len(ENTRY_THRESHOLD_SWEEP)
    log.info(f"  Total configs: {total}")
    log.info(f"  Walk-forward: {TRAIN_DAYS}d train, {SLIDE_DAYS}d slide")

    try:
        results = run_passive_execution()
    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        log.error(traceback.format_exc())
        try:
            import mlflow
            mlflow.end_run(status="FAILED")
        except Exception:
            pass
        sys.exit(1)
