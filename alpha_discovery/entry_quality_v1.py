#!/usr/bin/env python3
"""
Entry Quality Filter Experiment v1
===================================

Base strategy: 30-min LGBM entry model → passive entry/TP, market SL
  Config: TP=25, SL_L=4, SL_S=3, max_hold=60, cancel=30

This experiment tests FILTERS that reject signals before entry.
The model itself is unchanged — we only gate which signals we act on.

Filters tested:
  1. Time of Day — exclude midday (11:00-14:00 ET)
  2. Spread gate — skip wide spread bars (> threshold)
  3. Volume confirmation — require volume above rolling average
  4. OFI confluence — require orderflow in signal direction
  5. Volatility regime — exclude extreme vol regimes
  6. Entry confidence tier — top 2%/3%/5%/10%

Sweep: 2 × 3 × 2 × 2 × 4 = 96 filter configs (individual)
Then combine best filters for final combo.

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
OUTPUT_DIR = ROOT / "output" / "entry_quality_v1"
MFE_MAE_DIR = ROOT / "output" / "mfe_mae_analysis"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [ENTRY-QUAL] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "entry_quality_v1.log")),
    ],
)
log = logging.getLogger("ENTRY-QUAL")

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
TICK_SIZE = 1.0
TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376
MARKET_SLIPPAGE_TICKS = 1.0
TRAIN_DAYS = 30
SLIDE_DAYS = 5
ENTRY_BAR_SIZE_MIN = 30

# BASE STRATEGY PARAMS (from validated config: Sharpe 5.17)
BASE_TP_TICKS = 25
BASE_SL_LONG = 4
BASE_SL_SHORT = 3
BASE_MAX_HOLD = 60
BASE_CANCEL_WINDOW = 30
BASE_ENTRY_THRESHOLD = 0.05  # top/bottom 5%

# LGBM params
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
#  DATA LOADING (from passive_sl_v1.py)
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
    log.info(f"Loaded {len(combined):,} minute bars across {len(frames)} days")
    return combined


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
            "date": date_str, "bar_key": bar_key,
            "ts": grp["ts_minute"].iloc[0],
            "open": close_arr[0], "high": close_arr.max(),
            "low": close_arr.min(), "close": close_arr[-1],
            "return_bar": (close_arr[-1] / close_arr[0] - 1) if close_arr[0] > 0 else 0,
            "range_ticks": (close_arr.max() - close_arr.min()) / TICK_SIZE,
            "close_position": ((close_arr[-1] - close_arr.min()) / max(close_arr.max() - close_arr.min(), TICK_SIZE)),
            "total_volume": vol_arr.sum(), "avg_volume": vol_arr.mean(),
            "volume_trend": _safe_polyfit_slope(vol_arr),
            "volume_concentration": vol_arr.max() / max(vol_arr.mean(), 1),
            "ofi_sum": ofi_arr.sum(), "ofi_mean": ofi_arr.mean(),
            "ofi_std": ofi_arr.std() if len(ofi_arr) > 1 else 0,
            "ofi_trend": _safe_polyfit_slope(ofi_arr),
            "ofi_consistency": (np.mean(np.sign(ofi_arr) == np.sign(ofi_arr.sum())) if ofi_arr.sum() != 0 else 0.5),
            "ofi_late_vs_early": (ofi_arr[len(ofi_arr)//2:].sum() - ofi_arr[:len(ofi_arr)//2].sum()),
            "signed_volume_sum": sv_arr.sum(),
            "signed_volume_ratio": sv_arr.sum() / max(vol_arr.sum(), 1),
            "buy_volume_frac": float(np.sum(sv_arr[sv_arr > 0])) / max(vol_arr.sum(), 1),
            "sell_volume_frac": float(-np.sum(sv_arr[sv_arr < 0])) / max(vol_arr.sum(), 1),
            "sweep_minutes": int(np.sum(np.abs(grp["sv_zscore"].values) > 2)),
            "max_sweep_intensity": float(np.abs(grp["sv_zscore"].values).max()),
            "sweep_direction": float(np.sign(sv_arr[np.abs(grp["sv_zscore"].values).argmax()])) if len(sv_arr) > 0 else 0.0,
            "spread_mean": spread_arr.mean(), "spread_max": spread_arr.max(),
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
    log.info(f"Aggregated {len(result):,} {bar_size_min}min bars")
    return result


def add_rolling_features(df: pd.DataFrame, bar_size_min: int = 30) -> pd.DataFrame:
    df = df.sort_values("ts").reset_index(drop=True)
    for w in [4, 8, 16, 32]:
        roll_mean = df["ofi_sum"].rolling(w, min_periods=1).mean()
        roll_std = df["ofi_sum"].rolling(w, min_periods=2).std().fillna(1).replace(0, 1)
        df[f"ofi_zscore_{w}bar"] = (df["ofi_sum"] - roll_mean) / roll_std
        vol_ma = df["total_volume"].rolling(w, min_periods=1).mean()
        df[f"vol_rel_{w}bar"] = df["total_volume"] / vol_ma.clip(lower=1)
        if "sweep_minutes" in df.columns:
            df[f"sweep_pct_{w}bar"] = df["sweep_minutes"].rolling(w, min_periods=1).sum() / (w * bar_size_min)
    for bars, label in [(4, "lb_4bar"), (8, "lb_8bar"), (16, "lb_16bar")]:
        df[f"ret_{label}"] = df["close"].pct_change(bars)
    for w in [4, 8, 16]:
        df[f"rvol_{w}bar"] = df["return_bar"].rolling(w, min_periods=2).std()
    df["intraday_cum_ofi"] = df.groupby("date")["ofi_sum"].cumsum()
    df["intraday_cum_sv"] = df.groupby("date")["signed_volume_sum"].cumsum()
    df["ofi_sign_flip"] = (np.sign(df["ofi_sum"]) != np.sign(df["ofi_sum"].shift(1))).astype(np.float32)
    df["absorption"] = df["total_volume"] / df["range_ticks"].clip(lower=1)
    day_stats = df.groupby("date").agg(
        day_ofi=("ofi_sum", "sum"), day_sv=("signed_volume_sum", "sum"),
        day_ret=("return_bar", "sum"), day_vol=("realized_vol", "mean")
    ).reset_index()
    day_stats["prev_day_ofi"] = day_stats["day_ofi"].shift(1)
    day_stats["prev_day_sv"] = day_stats["day_sv"].shift(1)
    day_stats["prev_day_ret"] = day_stats["day_ret"].shift(1)
    df = df.merge(day_stats[["date", "prev_day_ofi", "prev_day_sv", "prev_day_ret"]], on="date", how="left")
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
        / df.groupby("date")["ofi_sum"].transform(lambda x: x.abs().cumsum()).clip(lower=1)
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
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


def add_forward_labels(df, horizon_bars, horizon_label, min_edge_ticks=2.5):
    df = df.sort_values("ts").reset_index(drop=True)
    fwd_close = df["close"].shift(-horizon_bars)
    fwd_ticks = (fwd_close - df["close"]) / TICK_SIZE
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
    df.loc[fwd_ticks > min_edge_ticks, f"direction_{horizon_label}"] = 1
    df.loc[fwd_ticks < -min_edge_ticks, f"direction_{horizon_label}"] = -1
    return df


def leakage_audit(train_dates, val_dates, feature_cols):
    overlap = set(train_dates) & set(val_dates)
    if overlap:
        return False
    if max(train_dates) >= min(val_dates):
        return False
    fwd_leak = [c for c in feature_cols if c.startswith(("fwd_", "direction_", "trade_quality_"))]
    if fwd_leak:
        return False
    return True


def train_entry_model(X_train, y_train, X_val, y_val, feature_names, fold_idx):
    _import_lightgbm()
    train_valid = ~np.isnan(y_train)
    val_valid = ~np.isnan(y_val)
    if train_valid.sum() < 50 or val_valid.sum() < 10:
        return None, np.full(len(y_train), np.nan), np.full(len(y_val), np.nan)
    X_tr, y_tr = X_train[train_valid], y_train[train_valid]
    X_v, y_v = X_val[val_valid], y_val[val_valid]
    params = {**LGBM_ENTRY_PARAMS, "seed": 42 + fold_idx}
    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_names)
    val_data = lgb.Dataset(X_v, label=y_v, feature_name=feature_names, reference=train_data)
    callbacks = [lgb.early_stopping(stopping_rounds=30, verbose=False), lgb.log_evaluation(period=0)]
    model = lgb.train(params, train_data, num_boost_round=500, valid_sets=[val_data], callbacks=callbacks)
    train_preds = np.full(len(y_train), np.nan)
    val_preds = np.full(len(y_val), np.nan)
    train_preds[train_valid] = model.predict(X_tr, num_iteration=model.best_iteration)
    val_preds[val_valid] = model.predict(X_v, num_iteration=model.best_iteration)
    return model, train_preds, val_preds


# ═══════════════════════════════════════════════════════════════════
#  ENTRY FILL RECONSTRUCTION (from passive_sl_v1.py)
#  Modified: returns extra bar-level metadata for filtering
# ═══════════════════════════════════════════════════════════════════

def reconstruct_entry_fills(
    bars_30m: pd.DataFrame,
    minute_df: pd.DataFrame,
    pred_30m: np.ndarray,
    confidence_pct: float,
    cancel_window_min: int,
) -> Tuple[List[Dict], Dict]:
    valid_mask = ~np.isnan(pred_30m)
    valid_preds = pred_30m[valid_mask]
    if len(valid_preds) < 20:
        return [], {"error": "too few predictions"}

    upper_thresh = np.nanquantile(pred_30m[valid_mask], 1 - confidence_pct)
    lower_thresh = np.nanquantile(pred_30m[valid_mask], confidence_pct)

    minute_lookup = {}
    for date_str, grp in minute_df.groupby("date"):
        minute_lookup[date_str] = grp.sort_values("ts_minute").reset_index(drop=True)

    trades = []
    n_signals = 0
    n_filled = 0
    n_cancelled = 0
    fill_delays = []

    bars_ts = bars_30m["ts"].values
    bars_dates = bars_30m["date"].values
    bars_close = bars_30m["close"].values

    # Pre-extract bar-level filter features from bars_30m
    bars_spread_mean = bars_30m["spread_mean"].values if "spread_mean" in bars_30m.columns else None
    bars_ofi_sum = bars_30m["ofi_sum"].values if "ofi_sum" in bars_30m.columns else None
    bars_total_volume = bars_30m["total_volume"].values if "total_volume" in bars_30m.columns else None

    # Compute rolling volume average (30-bar = ~15 hours = roughly "intraday average")
    if bars_total_volume is not None:
        vol_series = pd.Series(bars_total_volume)
        vol_rolling_avg = vol_series.rolling(window=2, min_periods=1).mean().values  # 2 bars = 1 hour
    else:
        vol_rolling_avg = None

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
            continue

        day_minutes = minute_lookup[date_str]
        day_ts = day_minutes["ts_minute"].values

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
                if mbar["low"] <= limit_price - TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar["ts_minute"]
                    fill_delays.append(j + 1)
                    break
            else:
                if mbar["high"] >= limit_price + TICK_SIZE:
                    filled = True
                    fill_price = limit_price
                    fill_ts = mbar["ts_minute"]
                    fill_delays.append(j + 1)
                    break

        if not filled:
            n_cancelled += 1
            continue

        n_filled += 1

        fill_ts_np = np.datetime64(fill_ts)
        remaining_mask = day_ts >= fill_ts_np
        remaining_minutes = day_minutes[remaining_mask]

        if len(remaining_minutes) < 2:
            n_filled -= 1
            n_cancelled += 1
            continue

        # Compute filter features for this signal bar
        # Time of day (UTC hour of signal)
        signal_hour_utc = signal_ts.hour + signal_ts.minute / 60.0
        # Convert UTC to ET: ET = UTC - 4 (EDT) or UTC - 5 (EST)
        # During summer (Jul-Nov data): EDT, so ET = UTC - 4
        signal_hour_et = signal_hour_utc - 4.0

        # Get 5-min volume around signal from minute bars
        lookback_5min_mask = (day_ts >= np.datetime64(signal_ts - pd.Timedelta(minutes=5))) & \
                             (day_ts < np.datetime64(signal_ts))
        recent_5min_vol = day_minutes.loc[lookback_5min_mask, "volume"].sum() if lookback_5min_mask.any() else 0

        # Get 30-min rolling avg volume
        lookback_30min_mask = (day_ts >= np.datetime64(signal_ts - pd.Timedelta(minutes=30))) & \
                              (day_ts < np.datetime64(signal_ts))
        recent_30min_vol = day_minutes.loc[lookback_30min_mask, "volume"].mean() if lookback_30min_mask.any() else 1

        # OFI from the signal bar
        bar_ofi = float(bars_ofi_sum[i]) if bars_ofi_sum is not None else 0.0

        # Spread from the signal bar
        bar_spread = float(bars_spread_mean[i]) if bars_spread_mean is not None else 1.0

        # Vol regime from minute bars around signal
        vol_regime_vals = day_minutes.loc[lookback_5min_mask, "vol_regime"].values if lookback_5min_mask.any() else []
        vol_regime = vol_regime_vals[-1] if len(vol_regime_vals) > 0 else "medium"

        # Volume relative to rolling average
        bar_vol = float(bars_total_volume[i]) if bars_total_volume is not None else 0
        bar_vol_avg = float(vol_rolling_avg[i]) if vol_rolling_avg is not None else 1
        vol_above_avg = bar_vol > bar_vol_avg

        trade = {
            "idx": i,
            "date": date_str,
            "signal_ts": signal_ts,
            "fill_ts": pd.Timestamp(fill_ts),
            "fill_price": fill_price,
            "fill_delay_minutes": fill_delays[-1],
            "direction": direction,
            "pred_30m": float(pred_30m[i]),
            "prices_close": remaining_minutes["close"].values.copy(),
            "prices_high": remaining_minutes["high"].values.copy(),
            "prices_low": remaining_minutes["low"].values.copy(),
            "times": remaining_minutes["ts_minute"].values.copy(),
            "n_remaining_minutes": len(remaining_minutes),
            # Filter features
            "signal_hour_et": signal_hour_et,
            "bar_spread": bar_spread,
            "bar_ofi": bar_ofi,
            "vol_above_avg": vol_above_avg,
            "vol_regime": vol_regime,
            "recent_5min_vol": recent_5min_vol,
            "recent_30min_avg_vol": recent_30min_vol,
        }
        trades.append(trade)

    fill_stats = {
        "n_signals": n_signals,
        "n_filled": n_filled,
        "n_cancelled": n_cancelled,
        "fill_rate": n_filled / max(n_signals, 1),
        "avg_fill_delay_min": float(np.mean(fill_delays)) if fill_delays else 0,
    }
    log.info(f"  Entry fills (top/bottom {confidence_pct:.0%}, cancel={cancel_window_min}m): "
             f"signals={n_signals}, filled={n_filled} ({fill_stats['fill_rate']:.1%})")
    return trades, fill_stats


# ═══════════════════════════════════════════════════════════════════
#  PASSIVE TP + STOP LOSS EXIT (asymmetric SL: long=4, short=3)
# ═══════════════════════════════════════════════════════════════════

def simulate_exit(
    trades: List[Dict],
    tp_ticks: int,
    sl_long: int,
    sl_short: int,
    max_hold_minutes: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, Dict]:
    pnls = []
    dates = []
    directions = []
    exit_types_list = []

    n_tp = 0
    n_sl = 0
    n_time = 0

    for trade in trades:
        direction = trade["direction"]
        fill_price = trade["fill_price"]
        prices_close = trade["prices_close"]
        prices_high = trade["prices_high"]
        prices_low = trade["prices_low"]
        n_remaining = trade["n_remaining_minutes"]

        sl_ticks = sl_long if direction == 1 else sl_short
        tp_price = fill_price + direction * tp_ticks * TICK_SIZE
        sl_price = fill_price - direction * sl_ticks * TICK_SIZE

        exit_found = False
        max_check = min(max_hold_minutes, n_remaining)

        for m in range(1, max_check):
            bar_high = prices_high[m]
            bar_low = prices_low[m]
            sl_hit = False
            tp_hit = False

            if direction == 1:
                if bar_low <= sl_price:
                    sl_hit = True
                if bar_high >= tp_price + TICK_SIZE:
                    tp_hit = True
            else:
                if bar_high >= sl_price:
                    sl_hit = True
                if bar_low <= tp_price - TICK_SIZE:
                    tp_hit = True

            if sl_hit and tp_hit:
                sl_hit = True
                tp_hit = False

            if sl_hit:
                trade_pnl = -(sl_ticks + MARKET_SLIPPAGE_TICKS + RT_COMMISSION_TICKS)
                pnls.append(trade_pnl)
                dates.append(trade["date"])
                directions.append(direction)
                exit_types_list.append("sl")
                n_sl += 1
                exit_found = True
                break

            if tp_hit:
                trade_pnl = tp_ticks - RT_COMMISSION_TICKS
                pnls.append(trade_pnl)
                dates.append(trade["date"])
                directions.append(direction)
                exit_types_list.append("tp")
                n_tp += 1
                exit_found = True
                break

        if not exit_found:
            exit_minute = min(max_hold_minutes, n_remaining - 1)
            exit_minute = max(exit_minute, 1)
            exit_close = prices_close[exit_minute]
            if direction == 1:
                exit_fill = exit_close - TICK_SIZE
            else:
                exit_fill = exit_close + TICK_SIZE
            raw_pnl_ticks = (exit_fill - fill_price) / TICK_SIZE * direction
            trade_pnl = raw_pnl_ticks - RT_COMMISSION_TICKS
            pnls.append(trade_pnl)
            dates.append(trade["date"])
            directions.append(direction)
            exit_types_list.append("time_stop")
            n_time += 1

    total = n_tp + n_sl + n_time
    exit_stats = {
        "n_tp": n_tp, "n_sl": n_sl, "n_time_stop": n_time,
        "tp_rate": n_tp / max(total, 1),
        "sl_rate": n_sl / max(total, 1),
    }

    return np.array(pnls), np.array(dates), np.array(directions), exit_stats


# ═══════════════════════════════════════════════════════════════════
#  FILTER LOGIC
# ═══════════════════════════════════════════════════════════════════

def apply_filters(
    trades: List[Dict],
    tod_exclude_midday: bool,
    spread_gate: Optional[float],
    volume_above_avg: bool,
    ofi_confluence: bool,
) -> List[Dict]:
    """Apply entry quality filters. Returns filtered trade list."""
    filtered = []
    for t in trades:
        # 1. Time of Day filter — exclude midday (11:00-14:00 ET)
        if tod_exclude_midday:
            h = t["signal_hour_et"]
            if 11.0 <= h < 14.0:
                continue

        # 2. Spread gate
        if spread_gate is not None:
            if t["bar_spread"] > spread_gate:
                continue

        # 3. Volume above average
        if volume_above_avg:
            if not t["vol_above_avg"]:
                continue

        # 4. OFI confluence
        if ofi_confluence:
            if t["direction"] == 1 and t["bar_ofi"] <= 0:
                continue
            if t["direction"] == -1 and t["bar_ofi"] >= 0:
                continue

        filtered.append(t)
    return filtered


# ═══════════════════════════════════════════════════════════════════
#  METRICS
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(pnl_arr, dates_arr, directions_arr, day_returns):
    if len(pnl_arr) == 0:
        return {"error": "no trades", "n_trades": 0}

    wr = float(np.mean(pnl_arr > 0))
    gross_win = float(np.sum(pnl_arr[pnl_arr > 0]))
    gross_loss = float(-np.sum(pnl_arr[pnl_arr < 0]))
    pf = gross_win / max(gross_loss, 1e-6)

    trade_df = pd.DataFrame({"pnl": pnl_arr, "date": dates_arr})
    day_pnl = trade_df.groupby("date")["pnl"].agg(["sum", "count"]).reset_index()
    day_pnl.columns = ["date", "daily_pnl", "daily_trades"]

    n_days = len(day_pnl)
    daily_sharpe = 0.0
    daily_sortino = 0.0
    if n_days > 2:
        daily_mean = day_pnl["daily_pnl"].mean()
        daily_std = day_pnl["daily_pnl"].std()
        daily_sharpe = float(daily_mean / max(daily_std, 1e-6) * np.sqrt(252))
        daily_downside = np.sqrt(np.mean(np.minimum(day_pnl["daily_pnl"].values, 0) ** 2))
        daily_sortino = float(daily_mean / max(daily_downside, 1e-6) * np.sqrt(252))

    # Day concentration
    total_abs = day_pnl["daily_pnl"].abs().sum()
    day_conc = float(day_pnl["daily_pnl"].abs().max() / max(total_abs, 1e-6))

    # Regime analysis
    day_class = {}
    for d, ret in day_returns.items():
        if pd.isna(ret):
            continue
        if ret > 0.001:
            day_class[d] = "green"
        elif ret < -0.001:
            day_class[d] = "red"
        else:
            day_class[d] = "flat"

    day_pnl["regime"] = day_pnl["date"].map(lambda d: day_class.get(d, "flat"))

    regime_sharpes = {}
    for regime in ["green", "red"]:
        mask = day_pnl["regime"] == regime
        if mask.sum() < 3:
            continue
        r_pnl = day_pnl.loc[mask, "daily_pnl"].values
        r_sharpe = float(r_pnl.mean() / max(r_pnl.std(), 1e-6) * np.sqrt(252))
        regime_sharpes[regime] = r_sharpe

    regime_gap = float("nan")
    regime_gap_pass = False
    if "green" in regime_sharpes and "red" in regime_sharpes:
        s_g = regime_sharpes["green"]
        s_r = regime_sharpes["red"]
        denom = max(abs(s_g), abs(s_r), 1e-6)
        regime_gap = abs(s_g - s_r) / denom
        regime_gap_pass = regime_gap <= 0.50

    # Long/short breakdown
    long_mask = directions_arr == 1
    short_mask = directions_arr == -1
    long_pnl = pnl_arr[long_mask]
    short_pnl = pnl_arr[short_mask]

    return {
        "n_trades": len(pnl_arr),
        "n_days": n_days,
        "trades_per_day": float(len(pnl_arr) / max(n_days, 1)),
        "total_pnl_ticks": float(pnl_arr.sum()),
        "total_pnl_dollars": float(pnl_arr.sum() * TICK_VALUE),
        "avg_pnl_ticks": float(pnl_arr.mean()),
        "win_rate": wr,
        "profit_factor": pf,
        "daily_sharpe": daily_sharpe,
        "daily_sortino": daily_sortino,
        "day_concentration": day_conc,
        "day_conc_pass": day_conc <= 0.70,
        "regime_gap": regime_gap,
        "regime_gap_pass": regime_gap_pass,
        "regime_sharpes": regime_sharpes,
        "long_trades": int(long_mask.sum()),
        "long_wr": float(np.mean(long_pnl > 0)) if len(long_pnl) > 0 else 0,
        "long_avg": float(long_pnl.mean()) if len(long_pnl) > 0 else 0,
        "short_trades": int(short_mask.sum()),
        "short_wr": float(np.mean(short_pnl > 0)) if len(short_pnl) > 0 else 0,
        "short_avg": float(short_pnl.mean()) if len(short_pnl) > 0 else 0,
    }


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
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    _import_lightgbm()
    t0 = time.time()

    # ── MLflow ──
    mlflow_active = False
    try:
        import mlflow
        mlflow.set_tracking_uri("http://neptune:5000")
        mlflow.set_experiment("entry_quality_v1")
        mlflow_run = mlflow.start_run(
            run_name=f"entry_qual_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        mlflow_active = True
        log.info(f"MLflow run started: {mlflow_run.info.run_id}")
    except Exception as e:
        log.warning(f"MLflow unavailable: {e} -- saving to disk only")

    # ══════════════════════════════════════
    #  PHASE 1: Data + Features + Predictions
    # ══════════════════════════════════════
    log.info("=" * 70)
    log.info("PHASE 1: Loading data and building features")
    log.info("=" * 70)

    minute_df = load_all_minute_bars()
    bars_30m = aggregate_to_bars(minute_df, bar_size_min=30)
    bars_30m = add_rolling_features(bars_30m, bar_size_min=30)
    bars_30m = add_forward_labels(bars_30m, horizon_bars=1, horizon_label="30min",
                                   min_edge_ticks=2.5)
    feature_cols = get_feature_columns(bars_30m)
    log.info(f"Features: {len(feature_cols)}")

    day_close = bars_30m.groupby("date")["close"].last()
    day_returns_raw = day_close.pct_change()
    day_returns = day_returns_raw.to_dict()

    # ── Predictions (cached or WF) ──
    dates_all = bars_30m["date"].values
    labels_all = bars_30m["fwd_ticks_30min"].values.astype(np.float32)
    dates_30m = sorted(bars_30m["date"].unique())

    cached_preds_path = MFE_MAE_DIR / "entry_predictions.npz"
    entry_preds = None

    if cached_preds_path.exists():
        try:
            cached = np.load(str(cached_preds_path), allow_pickle=True)
            cached_preds = cached["entry_preds"]
            cached_dates = cached["dates"]
            if len(cached_preds) == len(bars_30m) and np.array_equal(cached_dates, dates_all):
                entry_preds = cached_preds
                n_valid = (~np.isnan(entry_preds)).sum()
                log.info(f"Loaded cached predictions: {n_valid} valid")
            else:
                log.warning("Cached predictions don't align -- retraining")
        except Exception as e:
            log.warning(f"Failed to load cached predictions: {e}")

    if entry_preds is None:
        log.info("Training walk-forward entry model (30d train, SLIDING)...")
        features_all = bars_30m[feature_cols].values.astype(np.float32)
        entry_preds = np.full(len(bars_30m), np.nan, dtype=np.float32)
        fold_idx = 0

        for fold_start in range(TRAIN_DAYS, len(dates_30m) - SLIDE_DAYS + 1, SLIDE_DAYS):
            fold_train_dates = dates_30m[fold_start - TRAIN_DAYS: fold_start]
            fold_val_dates = dates_30m[fold_start: fold_start + SLIDE_DAYS]
            if len(fold_val_dates) < SLIDE_DAYS:
                break
            fold_idx += 1
            train_mask = np.isin(dates_all, fold_train_dates)
            val_mask = np.isin(dates_all, fold_val_dates)
            if not leakage_audit(list(fold_train_dates), list(fold_val_dates), feature_cols):
                continue

            X_tr = features_all[train_mask].copy()
            X_vl = features_all[val_mask].copy()
            y_tr = labels_all[train_mask].copy()
            y_vl = labels_all[val_mask].copy()

            # Impute NaNs
            med = np.nanmedian(X_tr, axis=0)
            for col_i in range(X_tr.shape[1]):
                nan_mask_tr = np.isnan(X_tr[:, col_i])
                X_tr[nan_mask_tr, col_i] = med[col_i]
                nan_mask_vl = np.isnan(X_vl[:, col_i])
                X_vl[nan_mask_vl, col_i] = med[col_i]

            model, _, val_preds = train_entry_model(X_tr, y_tr, X_vl, y_vl, feature_cols, fold_idx)
            if model is not None:
                entry_preds[val_mask] = val_preds

        n_valid = (~np.isnan(entry_preds)).sum()
        log.info(f"Walk-forward complete: {n_valid} valid OOT predictions across {fold_idx} folds")

        # Save predictions
        np.savez(str(OUTPUT_DIR / "entry_predictions.npz"),
                 entry_preds=entry_preds, labels_30m=labels_all, dates=dates_all)

    # ══════════════════════════════════════
    #  PHASE 2: Reconstruct fills for ALL thresholds
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: Reconstruct entry fills per confidence tier")
    log.info("=" * 70)

    entry_thresholds = [0.02, 0.03, 0.05, 0.10]
    trades_by_threshold = {}

    for thresh in entry_thresholds:
        log.info(f"\n--- Threshold: top/bottom {thresh:.0%} ---")
        trades, fill_stats = reconstruct_entry_fills(
            bars_30m, minute_df, entry_preds,
            confidence_pct=thresh,
            cancel_window_min=BASE_CANCEL_WINDOW,
        )
        trades_by_threshold[thresh] = trades
        log.info(f"  Filled trades: {len(trades)}")

    # ══════════════════════════════════════
    #  PHASE 3: Filter sweep
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 3: Entry quality filter sweep (96 configs)")
    log.info("=" * 70)

    # Sweep dimensions
    tod_options = [False, True]           # exclude midday
    spread_options = [None, 1.25, 1.5]    # spread gate
    vol_options = [False, True]           # volume above avg
    ofi_options = [False, True]           # OFI confluence

    all_results = []
    config_idx = 0
    total_configs = len(tod_options) * len(spread_options) * len(vol_options) * len(ofi_options) * len(entry_thresholds)

    for thresh in entry_thresholds:
        trades = trades_by_threshold[thresh]
        if len(trades) == 0:
            log.warning(f"No trades for threshold {thresh}")
            continue

        for tod_mid, spread_gate, vol_abv, ofi_conf in product(
            tod_options, spread_options, vol_options, ofi_options
        ):
            config_idx += 1

            # Apply filters
            filtered = apply_filters(trades, tod_mid, spread_gate, vol_abv, ofi_conf)

            # Config label
            label_parts = [f"t{thresh:.0%}"]
            if tod_mid:
                label_parts.append("noMid")
            if spread_gate is not None:
                label_parts.append(f"sp<{spread_gate}")
            if vol_abv:
                label_parts.append("volAbv")
            if ofi_conf:
                label_parts.append("ofiConf")
            label = "_".join(label_parts)

            if len(filtered) < 10:
                log.info(f"  [{config_idx}/{total_configs}] {label}: {len(filtered)} trades (too few, skip)")
                all_results.append({
                    "config": label, "threshold": thresh,
                    "tod_exclude_midday": tod_mid, "spread_gate": spread_gate,
                    "volume_above_avg": vol_abv, "ofi_confluence": ofi_conf,
                    "n_trades_before_filter": len(trades),
                    "n_trades_after_filter": len(filtered),
                    "filter_pct": 1 - len(filtered) / max(len(trades), 1),
                    "error": "too few trades after filter",
                })
                continue

            # Simulate exits
            pnl_arr, dates_arr, dir_arr, exit_stats = simulate_exit(
                filtered, BASE_TP_TICKS, BASE_SL_LONG, BASE_SL_SHORT, BASE_MAX_HOLD
            )

            # Compute metrics
            metrics = compute_metrics(pnl_arr, dates_arr, dir_arr, day_returns)

            result = {
                "config": label,
                "threshold": thresh,
                "tod_exclude_midday": tod_mid,
                "spread_gate": spread_gate,
                "volume_above_avg": vol_abv,
                "ofi_confluence": ofi_conf,
                "n_trades_before_filter": len(trades),
                "n_trades_after_filter": len(filtered),
                "filter_pct": 1 - len(filtered) / max(len(trades), 1),
                **metrics,
                "exit_stats": exit_stats,
            }
            all_results.append(result)

            if config_idx % 10 == 0 or config_idx == total_configs:
                log.info(f"  [{config_idx}/{total_configs}] {label}: "
                         f"trades={metrics.get('n_trades', 0)}, "
                         f"Sharpe={metrics.get('daily_sharpe', 0):.2f}, "
                         f"WR={metrics.get('win_rate', 0):.1%}, "
                         f"PF={metrics.get('profit_factor', 0):.2f}, "
                         f"regime_gap={metrics.get('regime_gap', float('nan')):.2f}")

    # ══════════════════════════════════════
    #  PHASE 4: Analysis & ranking
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 4: Analysis & ranking")
    log.info("=" * 70)

    # Filter to valid results
    valid_results = [r for r in all_results if "error" not in r and r.get("n_trades", 0) >= 10]

    if not valid_results:
        log.error("No valid results! Check data.")
        return

    # Sort by Sharpe
    valid_results.sort(key=lambda r: r.get("daily_sharpe", 0), reverse=True)

    # Find baseline (no filters, 5% threshold)
    baseline = None
    for r in valid_results:
        if (r["threshold"] == 0.05 and not r["tod_exclude_midday"]
            and r["spread_gate"] is None and not r["volume_above_avg"]
            and not r["ofi_confluence"]):
            baseline = r
            break

    baseline_sharpe = baseline["daily_sharpe"] if baseline else 0

    log.info(f"\nBaseline (no filters, 5%): Sharpe={baseline_sharpe:.2f}")
    log.info(f"Total valid configs: {len(valid_results)}")

    # ── Top 10 by Sharpe ──
    log.info("\n" + "─" * 50)
    log.info("TOP 10 CONFIGS BY SHARPE")
    log.info("─" * 50)
    for i, r in enumerate(valid_results[:10]):
        sharpe_delta = r["daily_sharpe"] - baseline_sharpe
        log.info(
            f"  #{i+1}: {r['config']:<40s} "
            f"Sharpe={r['daily_sharpe']:6.2f} (Δ{sharpe_delta:+.2f}) "
            f"Sortino={r['daily_sortino']:6.2f} "
            f"WR={r['win_rate']:.1%} PF={r['profit_factor']:.2f} "
            f"trades={r['n_trades']:4d} tpd={r['trades_per_day']:.1f} "
            f"regime_gap={r.get('regime_gap', float('nan')):.2f} "
            f"{'PASS' if r.get('regime_gap_pass', False) else 'FAIL'} "
            f"filtered={r['filter_pct']:.0%}"
        )

    # ── Per-filter impact analysis ──
    log.info("\n" + "─" * 50)
    log.info("PER-FILTER IMPACT (avg Sharpe improvement over same-threshold baseline)")
    log.info("─" * 50)

    # Group results by threshold for fair comparison
    for thresh in entry_thresholds:
        thresh_results = [r for r in valid_results if r["threshold"] == thresh]
        thresh_base = [r for r in thresh_results
                       if not r["tod_exclude_midday"] and r["spread_gate"] is None
                       and not r["volume_above_avg"] and not r["ofi_confluence"]]
        if not thresh_base:
            continue
        base_s = thresh_base[0]["daily_sharpe"]

        # ToD impact
        tod_on = [r["daily_sharpe"] for r in thresh_results if r["tod_exclude_midday"]]
        tod_off = [r["daily_sharpe"] for r in thresh_results if not r["tod_exclude_midday"]]
        if tod_on and tod_off:
            log.info(f"  [{thresh:.0%}] ToD_exclude_midday: ON avg={np.mean(tod_on):.2f}, OFF avg={np.mean(tod_off):.2f}, Δ={np.mean(tod_on)-np.mean(tod_off):+.2f}")

        # Spread impact
        for sg in [1.25, 1.5]:
            sp_on = [r["daily_sharpe"] for r in thresh_results if r["spread_gate"] == sg]
            sp_off = [r["daily_sharpe"] for r in thresh_results if r["spread_gate"] is None]
            if sp_on and sp_off:
                log.info(f"  [{thresh:.0%}] Spread<{sg}: ON avg={np.mean(sp_on):.2f}, OFF avg={np.mean(sp_off):.2f}, Δ={np.mean(sp_on)-np.mean(sp_off):+.2f}")

        # Volume impact
        vol_on = [r["daily_sharpe"] for r in thresh_results if r["volume_above_avg"]]
        vol_off = [r["daily_sharpe"] for r in thresh_results if not r["volume_above_avg"]]
        if vol_on and vol_off:
            log.info(f"  [{thresh:.0%}] Vol_above_avg: ON avg={np.mean(vol_on):.2f}, OFF avg={np.mean(vol_off):.2f}, Δ={np.mean(vol_on)-np.mean(vol_off):+.2f}")

        # OFI impact
        ofi_on = [r["daily_sharpe"] for r in thresh_results if r["ofi_confluence"]]
        ofi_off = [r["daily_sharpe"] for r in thresh_results if not r["ofi_confluence"]]
        if ofi_on and ofi_off:
            log.info(f"  [{thresh:.0%}] OFI_confluence: ON avg={np.mean(ofi_on):.2f}, OFF avg={np.mean(ofi_off):.2f}, Δ={np.mean(ofi_on)-np.mean(ofi_off):+.2f}")

    # ── Regime-passing configs ──
    regime_pass = [r for r in valid_results if r.get("regime_gap_pass", False)]
    log.info(f"\n{'─' * 50}")
    log.info(f"REGIME-PASSING CONFIGS: {len(regime_pass)} / {len(valid_results)}")
    log.info(f"{'─' * 50}")
    for i, r in enumerate(regime_pass[:10]):
        sharpe_delta = r["daily_sharpe"] - baseline_sharpe
        log.info(
            f"  #{i+1}: {r['config']:<40s} "
            f"Sharpe={r['daily_sharpe']:6.2f} (Δ{sharpe_delta:+.2f}) "
            f"Sortino={r['daily_sortino']:6.2f} "
            f"WR={r['win_rate']:.1%} PF={r['profit_factor']:.2f} "
            f"trades={r['n_trades']:4d} "
            f"gap={r.get('regime_gap', float('nan')):.2f}"
        )

    # ── Best regime-passing by Sharpe ──
    best_regime_pass = regime_pass[0] if regime_pass else None

    # ══════════════════════════════════════
    #  PHASE 5: Combination sweep of best individual filters
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 5: Best combo identification")
    log.info("=" * 70)

    # Find best config overall and best regime-passing
    best_overall = valid_results[0]
    log.info(f"\nBEST OVERALL: {best_overall['config']}")
    log.info(f"  Sharpe={best_overall['daily_sharpe']:.2f}, Sortino={best_overall['daily_sortino']:.2f}")
    log.info(f"  WR={best_overall['win_rate']:.1%}, PF={best_overall['profit_factor']:.2f}")
    log.info(f"  Trades={best_overall['n_trades']}, TPD={best_overall['trades_per_day']:.1f}")
    log.info(f"  Regime gap={best_overall.get('regime_gap', float('nan')):.2f} "
             f"{'PASS' if best_overall.get('regime_gap_pass', False) else 'FAIL'}")
    log.info(f"  Filtered out={best_overall['filter_pct']:.0%}")
    log.info(f"  Total PnL: {best_overall['total_pnl_ticks']:.1f} ticks (${best_overall['total_pnl_dollars']:.0f})")

    if best_regime_pass and best_regime_pass != best_overall:
        log.info(f"\nBEST REGIME-PASSING: {best_regime_pass['config']}")
        log.info(f"  Sharpe={best_regime_pass['daily_sharpe']:.2f}, Sortino={best_regime_pass['daily_sortino']:.2f}")
        log.info(f"  WR={best_regime_pass['win_rate']:.1%}, PF={best_regime_pass['profit_factor']:.2f}")
        log.info(f"  Trades={best_regime_pass['n_trades']}, TPD={best_regime_pass['trades_per_day']:.1f}")
        log.info(f"  Regime gap={best_regime_pass.get('regime_gap', float('nan')):.2f} PASS")
        log.info(f"  Filtered out={best_regime_pass['filter_pct']:.0%}")
        log.info(f"  Total PnL: {best_regime_pass['total_pnl_ticks']:.1f} ticks (${best_regime_pass['total_pnl_dollars']:.0f})")

    if baseline:
        log.info(f"\nBASELINE (no filters): Sharpe={baseline['daily_sharpe']:.2f}, "
                 f"WR={baseline['win_rate']:.1%}, PF={baseline['profit_factor']:.2f}, "
                 f"trades={baseline['n_trades']}")

    # ══════════════════════════════════════
    #  SAVE
    # ══════════════════════════════════════
    # Save all results
    results_file = OUTPUT_DIR / "filter_sweep_results.json"
    with open(str(results_file), "w") as f:
        json.dump(clean_for_json(all_results), f, indent=2, default=str)
    log.info(f"\nSaved {len(all_results)} results to {results_file}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "total_configs": total_configs,
        "valid_configs": len(valid_results),
        "regime_passing_configs": len(regime_pass),
        "baseline": clean_for_json(baseline) if baseline else None,
        "best_overall": clean_for_json(best_overall),
        "best_regime_passing": clean_for_json(best_regime_pass) if best_regime_pass else None,
        "top_10": clean_for_json(valid_results[:10]),
        "runtime_seconds": time.time() - t0,
    }
    summary_file = OUTPUT_DIR / "summary.json"
    with open(str(summary_file), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Saved summary to {summary_file}")

    # MLflow logging
    if mlflow_active:
        try:
            import mlflow
            mlflow.log_param("base_tp", BASE_TP_TICKS)
            mlflow.log_param("base_sl_long", BASE_SL_LONG)
            mlflow.log_param("base_sl_short", BASE_SL_SHORT)
            mlflow.log_param("base_max_hold", BASE_MAX_HOLD)
            mlflow.log_param("total_configs", total_configs)

            if baseline:
                mlflow.log_metric("baseline_sharpe", baseline["daily_sharpe"])
                mlflow.log_metric("baseline_wr", baseline["win_rate"])
                mlflow.log_metric("baseline_pf", baseline["profit_factor"])
                mlflow.log_metric("baseline_trades", baseline["n_trades"])

            mlflow.log_metric("best_sharpe", best_overall["daily_sharpe"])
            mlflow.log_metric("best_sortino", best_overall["daily_sortino"])
            mlflow.log_metric("best_wr", best_overall["win_rate"])
            mlflow.log_metric("best_pf", best_overall["profit_factor"])
            mlflow.log_metric("best_trades", best_overall["n_trades"])
            mlflow.log_metric("sharpe_improvement",
                              best_overall["daily_sharpe"] - baseline_sharpe)
            mlflow.log_metric("n_regime_passing", len(regime_pass))

            if best_regime_pass:
                mlflow.log_metric("best_regime_pass_sharpe", best_regime_pass["daily_sharpe"])

            mlflow.log_artifact(str(results_file))
            mlflow.log_artifact(str(summary_file))
            mlflow.end_run()
            log.info("MLflow run logged and closed.")
        except Exception as e:
            log.warning(f"MLflow logging error: {e}")

    elapsed = time.time() - t0
    log.info(f"\nTotal runtime: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info("DONE.")


if __name__ == "__main__":
    main()
