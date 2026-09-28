#!/usr/bin/env python3
"""
Passive Entry + Stop Loss v1 — The Missing Piece
=================================================

Execution model:
  - Entry: passive limit at bid/ask, FIFO back-of-queue (1-tick-through fill)
  - Take-profit: passive limit on opposite side, TP_ticks from entry (FIFO fill)
  - Stop loss: MARKET order if price moves SL_ticks against us (1 tick slippage)
  - Time stop: if neither TP nor SL hit within max_hold, exit at market (1 tick slippage)

Cost math:
  - TP fill (passive exit): +TP_ticks - 0.376 commission
  - SL fill (market exit): -(SL_ticks + 1 tick slippage + 0.376 commission)
  - Time stop (market exit): actual_move - 1 tick slippage - 0.376 commission

SWEEP: 6x6x2x2x3 = 432 configs
  TP_ticks: [2, 3, 4, 5, 6, 8]
  SL_ticks: [2, 3, 4, 5, 6, 8]
  cancel_window: [10, 30]
  max_hold: [30, 60]
  entry_threshold: [0.05, 0.10, 0.15]

Reuses WF entry model and data loading from mfe_mae_analysis.py.
Loads cached predictions from Phase 1 if available.

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
OUTPUT_DIR = ROOT / "output" / "passive_sl_v1"
MFE_MAE_DIR = ROOT / "output" / "mfe_mae_analysis"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [PASSIVE-SL] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "passive_sl_v1.log")),
    ],
)
log = logging.getLogger("PASSIVE-SL")

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────
TICK_SIZE = 1.0            # 1 data unit = 1 real tick
TICK_VALUE = 12.50
# RT commission = $4.70
RT_COMMISSION_TICKS = 0.376
MARKET_SLIPPAGE_TICKS = 1.0
TRAIN_DAYS = 30
SLIDE_DAYS = 5
ENTRY_BAR_SIZE_MIN = 30

# ─────────────────────────────────────────────
#  SWEEP PARAMETERS
# ─────────────────────────────────────────────
TP_TICKS_SWEEP = [8, 12, 16, 20, 25, 30]
SL_TICKS_SWEEP = [4, 6, 8, 10, 15, 20]
CANCEL_WINDOW_SWEEP = [10, 30]
MAX_HOLD_SWEEP = [30, 60]
ENTRY_THRESHOLD_SWEEP = [0.05, 0.10, 0.15]

TOTAL_CONFIGS = (len(TP_TICKS_SWEEP) * len(SL_TICKS_SWEEP) *
                 len(CANCEL_WINDOW_SWEEP) * len(MAX_HOLD_SWEEP) *
                 len(ENTRY_THRESHOLD_SWEEP))

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

lgb = None


def _import_lightgbm():
    global lgb
    if lgb is not None:
        return
    import lightgbm as _lgb
    lgb = _lgb


# ═══════════════════════════════════════════════════════════════════
#  DATA LOADING (identical to mfe_mae_analysis.py)
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
    log.info(f"Forward labels ({horizon_label}): {(~fwd_ticks.isna()).sum():,} valid")
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
    p_v = val_preds[val_valid]
    if len(p_v) > 5:
        ic = np.corrcoef(p_v, y_v)[0, 1]
        log.info(f"  Entry fold {fold_idx}: IC={ic:.4f}, best_iter={model.best_iteration}")
    return model, train_preds, val_preds


# ═══════════════════════════════════════════════════════════════════
#  ENTRY FILL RECONSTRUCTION (reused from passive_execution_v1.py)
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
#  PASSIVE TP + STOP LOSS EXIT SIMULATION (NEW)
# ═══════════════════════════════════════════════════════════════════


def simulate_passive_tp_sl_exit(
    trades: List[Dict],
    tp_ticks: int,
    sl_ticks: int,
    max_hold_minutes: int,
) -> Tuple[List[float], List[str], List[int], Dict]:
    """
    Simulate passive TP + market SL + time stop.

    Priority order per minute bar:
      1. Check if SL is hit (adverse excursion >= SL)
      2. Check if TP is hit (favorable excursion >= TP, FIFO 1-tick-through)
      3. If neither, continue to next bar
      4. If max_hold reached, exit at market

    Within a single bar, we must determine which was hit first.
    Conservative assumption: if BOTH could trigger in same bar,
    SL fires first (worst case for us). This prevents look-ahead bias.
    """
    pnls = []
    dates = []
    directions = []
    hold_durations = []
    exit_types = []  # 'tp', 'sl', 'time_stop'

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

        # TP and SL price levels
        tp_price = fill_price + direction * tp_ticks * TICK_SIZE
        sl_price = fill_price - direction * sl_ticks * TICK_SIZE

        # For FIFO TP fill: need 1-tick-through
        # For SL: triggered when price reaches SL level
        exit_found = False
        max_check = min(max_hold_minutes, n_remaining)

        for m in range(1, max_check):
            bar_high = prices_high[m]
            bar_low = prices_low[m]

            sl_hit = False
            tp_hit = False

            if direction == 1:
                # Long position
                # SL hit if low <= sl_price
                if bar_low <= sl_price:
                    sl_hit = True
                # TP hit if high >= tp_price + 1 tick (FIFO through)
                if bar_high >= tp_price + TICK_SIZE:
                    tp_hit = True
            else:
                # Short position
                # SL hit if high >= sl_price
                if bar_high >= sl_price:
                    sl_hit = True
                # TP hit if low <= tp_price - 1 tick (FIFO through)
                if bar_low <= tp_price - TICK_SIZE:
                    tp_hit = True

            if sl_hit and tp_hit:
                # Both triggered in same bar — conservative: SL fires first
                # Unless TP is closer to open (we can't know intra-bar order)
                # Conservative = SL first
                sl_hit = True
                tp_hit = False

            if sl_hit:
                # Market exit at SL: fill_price - direction * sl_ticks * tick + slippage
                # SL loss = sl_ticks + 1 tick slippage
                trade_pnl = -(sl_ticks + MARKET_SLIPPAGE_TICKS + RT_COMMISSION_TICKS)
                pnls.append(trade_pnl)
                dates.append(trade["date"])
                directions.append(direction)
                hold_durations.append(m)
                exit_types.append("sl")
                n_sl += 1
                exit_found = True
                break

            if tp_hit:
                # Passive TP fill at our limit price
                trade_pnl = tp_ticks - RT_COMMISSION_TICKS
                pnls.append(trade_pnl)
                dates.append(trade["date"])
                directions.append(direction)
                hold_durations.append(m)
                exit_types.append("tp")
                n_tp += 1
                exit_found = True
                break

        if not exit_found:
            # Time stop: exit at market
            exit_minute = min(max_hold_minutes, n_remaining - 1)
            exit_minute = max(exit_minute, 1)
            exit_close = prices_close[exit_minute]

            if direction == 1:
                exit_fill = exit_close - TICK_SIZE  # 1 tick slippage
            else:
                exit_fill = exit_close + TICK_SIZE

            raw_pnl_ticks = (exit_fill - fill_price) / TICK_SIZE * direction
            trade_pnl = raw_pnl_ticks - RT_COMMISSION_TICKS
            pnls.append(trade_pnl)
            dates.append(trade["date"])
            directions.append(direction)
            hold_durations.append(exit_minute)
            exit_types.append("time_stop")
            n_time += 1

    total = n_tp + n_sl + n_time
    exit_stats = {
        "n_tp": n_tp,
        "n_sl": n_sl,
        "n_time_stop": n_time,
        "tp_rate": n_tp / max(total, 1),
        "sl_rate": n_sl / max(total, 1),
        "time_stop_rate": n_time / max(total, 1),
        "avg_hold_min": float(np.mean(hold_durations)) if hold_durations else 0,
        "median_hold_min": float(np.median(hold_durations)) if hold_durations else 0,
    }

    # Avg winner/loser size
    pnl_arr = np.array(pnls)
    if len(pnl_arr) > 0:
        winners = pnl_arr[pnl_arr > 0]
        losers = pnl_arr[pnl_arr < 0]
        exit_stats["avg_winner"] = float(winners.mean()) if len(winners) > 0 else 0.0
        exit_stats["avg_loser"] = float(losers.mean()) if len(losers) > 0 else 0.0
        exit_stats["win_loss_ratio"] = (
            abs(exit_stats["avg_winner"] / exit_stats["avg_loser"])
            if exit_stats["avg_loser"] != 0 else float("inf")
        )

    return pnls, dates, directions, exit_stats


# ═══════════════════════════════════════════════════════════════════
#  DAILY STATS + REGIME ANALYSIS (reused)
# ═══════════════════════════════════════════════════════════════════


def compute_daily_stats(pnl_arr, dates, directions, label, exit_stats=None, fill_stats=None):
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

    day_conc = 0.0
    if len(day_pnl) > 0:
        total_abs = day_pnl["daily_pnl"].abs().sum()
        if total_abs > 0:
            day_conc = float(day_pnl["daily_pnl"].abs().max() / total_abs)

    result = {
        "label": label,
        "n_trades": len(pnl_arr),
        "total_pnl_ticks": float(pnl_arr.sum()),
        "total_pnl_dollars": float(pnl_arr.sum() * TICK_VALUE),
        "avg_pnl_ticks": float(pnl_arr.mean()),
        "daily_sharpe": float(daily_sharpe),
        "daily_sortino": float(daily_sortino),
        "win_rate": float(wr),
        "profit_factor": float(pf),
        "max_dd_ticks": float(max_dd),
        "max_dd_dollars": float(max_dd * TICK_VALUE),
        "n_trading_days": int(len(day_pnl)),
        "trades_per_day": float(len(pnl_arr) / max(len(day_pnl), 1)),
        "day_concentration": float(day_conc),
        "day_concentration_pass": day_conc <= 0.70,
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


def full_regime_analysis(pnl_arr, dates, directions, day_returns, bars_30m):
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
        win_rate=("pnl", lambda x: (x > 0).mean()),
    ).reset_index()
    day_agg["regime"] = day_agg["date"].map(lambda d: day_class.get(d, "flat"))

    regime_sharpes = {}
    regime_results = {}

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

    # Per-day table
    per_day = []
    for _, row in day_agg.iterrows():
        per_day.append({
            "date": row["date"],
            "regime": row["regime"],
            "daily_pnl_ticks": float(row["daily_pnl"]),
            "n_trades": int(row["n_trades"]),
            "win_rate": float(row["win_rate"]),
        })

    return {
        "regimes": regime_results,
        "per_day": per_day,
        "n_total_days": len(day_agg),
        "n_green_days": int((day_agg["regime"] == "green").sum()),
        "n_red_days": int((day_agg["regime"] == "red").sum()),
        "n_flat_days": int((day_agg["regime"] == "flat").sum()),
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
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════


def main():
    _import_lightgbm()
    t0 = time.time()

    # ── MLflow ──
    mlflow_active = False
    try:
        import mlflow
        mlflow.set_tracking_uri("http://neptune:5000")
        mlflow.set_experiment("passive_sl_v1")
        mlflow_run = mlflow.start_run(
            run_name=f"passive_sl_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        mlflow_active = True
        log.info(f"MLflow run started: {mlflow_run.info.run_id}")
    except Exception as e:
        log.warning(f"MLflow unavailable: {e} -- saving to disk only")

    # ══════════════════════════════════════
    #  PHASE 1: Data + Features
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

    # ══════════════════════════════════════
    #  PHASE 2: Try to load cached predictions, else train WF
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: Entry model predictions")
    log.info("=" * 70)

    dates_30m = sorted(bars_30m["date"].unique())
    labels_all = bars_30m["fwd_ticks_30min"].values.astype(np.float32)
    dates_all = bars_30m["date"].values

    # Try loading cached predictions from MFE/MAE analysis
    cached_preds_path = MFE_MAE_DIR / "entry_predictions.npz"
    entry_preds = None

    if cached_preds_path.exists():
        try:
            cached = np.load(str(cached_preds_path), allow_pickle=True)
            cached_preds = cached["entry_preds"]
            cached_dates = cached["dates"]
            # Verify alignment
            if len(cached_preds) == len(bars_30m) and np.array_equal(cached_dates, dates_all):
                entry_preds = cached_preds
                n_valid = (~np.isnan(entry_preds)).sum()
                log.info(f"Loaded cached predictions from MFE/MAE analysis: {n_valid} valid")
            else:
                log.warning("Cached predictions don't align -- retraining")
        except Exception as e:
            log.warning(f"Failed to load cached predictions: {e}")

    if entry_preds is None:
        log.info("Training walk-forward entry model (30d train, SLIDING)...")
        features_all = bars_30m[feature_cols].values.astype(np.float32)
        entry_preds = np.full(len(bars_30m), np.nan, dtype=np.float32)
        fold_idx = 0
        val_days = 5
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
                continue
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
                tr, labels_all[train_mask], vl, labels_all[val_mask],
                feature_names=feature_cols, fold_idx=fold_idx,
            )
            entry_preds[val_mask] = val_preds
            if fold_idx % 5 == 0:
                log.info(f"  Progress: fold {fold_idx}")

        log.info(f"Entry model: {fold_idx} folds, {(~np.isnan(entry_preds)).sum()} valid preds")

    # Concat IC
    valid_mask = ~np.isnan(entry_preds) & ~np.isnan(labels_all)
    if valid_mask.sum() > 50:
        ic = np.corrcoef(entry_preds[valid_mask], labels_all[valid_mask])[0, 1]
        rank_ic = stats.spearmanr(entry_preds[valid_mask], labels_all[valid_mask]).correlation
        log.info(f"Concat IC: {ic:.4f}, Rank IC: {rank_ic:.4f}")
        if mlflow_active:
            mlflow.log_metric("concat_ic", ic)
            mlflow.log_metric("rank_ic", rank_ic)

    # ══════════════════════════════════════
    #  PHASE 3: SWEEP — 432 configs
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info(f"PHASE 3: Sweeping {TOTAL_CONFIGS} passive TP + SL configs")
    log.info("=" * 70)

    if mlflow_active:
        mlflow.log_param("total_configs", TOTAL_CONFIGS)
        mlflow.log_param("tp_sweep", str(TP_TICKS_SWEEP))
        mlflow.log_param("sl_sweep", str(SL_TICKS_SWEEP))
        mlflow.log_param("cancel_sweep", str(CANCEL_WINDOW_SWEEP))
        mlflow.log_param("hold_sweep", str(MAX_HOLD_SWEEP))
        mlflow.log_param("entry_thresh_sweep", str(ENTRY_THRESHOLD_SWEEP))
        mlflow.log_param("commission_ticks", RT_COMMISSION_TICKS)
        mlflow.log_param("exit_slippage_ticks", MARKET_SLIPPAGE_TICKS)

    all_results = []
    config_count = 0
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
                for _ in product(TP_TICKS_SWEEP, SL_TICKS_SWEEP, MAX_HOLD_SWEEP):
                    config_count += 1
                continue

            for tp_ticks in TP_TICKS_SWEEP:
                for sl_ticks in SL_TICKS_SWEEP:
                    for max_hold in MAX_HOLD_SWEEP:
                        config_count += 1

                        pnls, trade_dates, trade_dirs, exit_stats = simulate_passive_tp_sl_exit(
                            trades, tp_ticks=tp_ticks, sl_ticks=sl_ticks,
                            max_hold_minutes=max_hold,
                        )

                        if len(pnls) < 10:
                            continue

                        pnl_arr = np.array(pnls)
                        dates_arr = np.array(trade_dates)
                        dirs_arr = np.array(trade_dirs)

                        label = (f"tp{tp_ticks}_sl{sl_ticks}_cw{cancel_window}_"
                                 f"mh{max_hold}_et{int(entry_thresh*100)}")

                        result = compute_daily_stats(
                            pnl_arr, dates_arr, dirs_arr, label,
                            exit_stats=exit_stats, fill_stats=fill_stats,
                        )

                        if "error" in result:
                            continue

                        result["config"] = {
                            "tp_ticks": tp_ticks,
                            "sl_ticks": sl_ticks,
                            "cancel_window": cancel_window,
                            "max_hold": max_hold,
                            "entry_threshold": entry_thresh,
                        }

                        # Breakeven WR for reference
                        be_wr = (sl_ticks + 1.376) / (tp_ticks + sl_ticks + 1.0)
                        result["breakeven_wr"] = float(be_wr)
                        result["wr_edge"] = float(result["win_rate"] - be_wr)

                        all_results.append(result)

                        if config_count % 50 == 0:
                            log.info(f"  Progress: {config_count}/{TOTAL_CONFIGS} "
                                     f"({len(all_results)} valid)")

    log.info(f"\nSweep complete: {config_count} configs evaluated, "
             f"{len(all_results)} produced valid results")

    # Sort by daily Sharpe
    all_results.sort(key=lambda x: x.get("daily_sharpe", -999), reverse=True)

    # ══════════════════════════════════════
    #  PHASE 4: LEADERBOARD
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PASSIVE TP + STOP LOSS LEADERBOARD (top 30)")
    log.info("=" * 70)
    log.info(
        f"{'Rk':>3} {'TP':>3} {'SL':>3} {'CW':>4} {'MH':>4} {'ET':>4} "
        f"{'Sharpe':>8} {'Sort':>8} {'WR':>6} {'BEWR':>6} {'Edge':>6} {'PF':>6} "
        f"{'N':>5} {'Days':>5} {'$PnL':>9} {'TP%':>6} {'SL%':>6} {'TO%':>6} "
        f"{'AvgW':>6} {'AvgL':>7} {'DayC':>5}"
    )
    log.info("-" * 130)

    for i, r in enumerate(all_results[:30]):
        cfg = r.get("config", {})
        es = r.get("exit_stats", {})
        log.info(
            f"{i+1:3d} "
            f"{cfg.get('tp_ticks', 0):3d} "
            f"{cfg.get('sl_ticks', 0):3d} "
            f"{cfg.get('cancel_window', 0):4d} "
            f"{cfg.get('max_hold', 0):4d} "
            f"{int(cfg.get('entry_threshold', 0)*100):3d}% "
            f"{r['daily_sharpe']:8.2f} {r['daily_sortino']:8.2f} "
            f"{r['win_rate']:6.1%} {r['breakeven_wr']:6.1%} {r['wr_edge']:+6.1%} "
            f"{r['profit_factor']:6.2f} "
            f"{r['n_trades']:5d} {r['n_trading_days']:5d} "
            f"{r['total_pnl_dollars']:9.0f} "
            f"{es.get('tp_rate', 0):6.1%} "
            f"{es.get('sl_rate', 0):6.1%} "
            f"{es.get('time_stop_rate', 0):6.1%} "
            f"{es.get('avg_winner', 0):6.2f} "
            f"{es.get('avg_loser', 0):7.2f} "
            f"{r['day_concentration']:5.2f}"
        )

    # ══════════════════════════════════════
    #  PHASE 5: Regime analysis on top 5
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 5: Regime analysis on top 5 configs")
    log.info("=" * 70)

    regime_results_top = {}
    for i, r in enumerate(all_results[:5]):
        cfg = r.get("config", {})
        label = r["label"]

        cache_key = (cfg["entry_threshold"], cfg["cancel_window"])
        trades, fill_stats = entry_fill_cache[cache_key]

        pnls, trade_dates, trade_dirs, exit_stats = simulate_passive_tp_sl_exit(
            trades, tp_ticks=cfg["tp_ticks"], sl_ticks=cfg["sl_ticks"],
            max_hold_minutes=cfg["max_hold"],
        )

        pnl_arr = np.array(pnls)
        dates_arr = np.array(trade_dates)
        dirs_arr = np.array(trade_dirs)

        regime = full_regime_analysis(pnl_arr, dates_arr, dirs_arr, day_returns, bars_30m)
        regime_results_top[label] = regime

        log.info(f"\n  Config #{i+1}: {label}")
        log.info(f"    Sharpe={r['daily_sharpe']:.2f}, Sortino={r['daily_sortino']:.2f}, "
                 f"WR={r['win_rate']:.1%} (BE: {r['breakeven_wr']:.1%}, Edge: {r['wr_edge']:+.1%})")
        log.info(f"    N={r['n_trades']}, Days={r['n_trading_days']}, "
                 f"$PnL={r['total_pnl_dollars']:.0f}")
        log.info(f"    TP={exit_stats.get('tp_rate', 0):.1%}, "
                 f"SL={exit_stats.get('sl_rate', 0):.1%}, "
                 f"TimeStop={exit_stats.get('time_stop_rate', 0):.1%}")
        log.info(f"    AvgWin={exit_stats.get('avg_winner', 0):.2f}, "
                 f"AvgLoss={exit_stats.get('avg_loser', 0):.2f}, "
                 f"W/L ratio={exit_stats.get('win_loss_ratio', 0):.2f}")

        regimes = regime.get("regimes", {})
        if "regime_gap_detail" in regimes:
            log.info(f"    REGIME GAP: {regimes['regime_gap_detail']}")

        for reg in ["green", "red", "flat"]:
            if reg in regimes and "sharpe" in regimes[reg]:
                rr = regimes[reg]
                log.info(f"      {reg:6s}: Sharpe={rr['sharpe']:.2f}, "
                         f"WR={rr['win_rate']:.1%}, PF={rr['profit_factor']:.2f}, "
                         f"Days={rr['n_days']}, Trades={rr['total_trades']}")

    # ══════════════════════════════════════
    #  PHASE 6: Cross-dimensional analysis
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 6: Cross-dimensional analysis")
    log.info("=" * 70)

    # Avg Sharpe by TP
    log.info("\n  AVG SHARPE BY TP_TICKS:")
    for tp in TP_TICKS_SWEEP:
        tp_r = [r for r in all_results if r["config"]["tp_ticks"] == tp]
        if tp_r:
            log.info(f"    TP={tp}: Sharpe={np.mean([r['daily_sharpe'] for r in tp_r]):.2f}, "
                     f"WR={np.mean([r['win_rate'] for r in tp_r]):.1%}, N={len(tp_r)}")

    log.info("\n  AVG SHARPE BY SL_TICKS:")
    for sl in SL_TICKS_SWEEP:
        sl_r = [r for r in all_results if r["config"]["sl_ticks"] == sl]
        if sl_r:
            log.info(f"    SL={sl}: Sharpe={np.mean([r['daily_sharpe'] for r in sl_r]):.2f}, "
                     f"WR={np.mean([r['win_rate'] for r in sl_r]):.1%}, "
                     f"SL%={np.mean([r['exit_stats']['sl_rate'] for r in sl_r]):.1%}, N={len(sl_r)}")

    log.info("\n  AVG SHARPE BY ENTRY THRESHOLD:")
    for et in ENTRY_THRESHOLD_SWEEP:
        et_r = [r for r in all_results if r["config"]["entry_threshold"] == et]
        if et_r:
            log.info(f"    ET={et:.0%}: Sharpe={np.mean([r['daily_sharpe'] for r in et_r]):.2f}, "
                     f"$PnL={np.mean([r['total_pnl_dollars'] for r in et_r]):.0f}, N={len(et_r)}")

    # Best TP/SL heatmap
    log.info("\n  TP/SL SHARPE HEATMAP (avg across other params):")
    log.info(f"  {'':>4} " + " ".join(f"SL={sl:2d}" for sl in SL_TICKS_SWEEP))
    for tp in TP_TICKS_SWEEP:
        row = []
        for sl in SL_TICKS_SWEEP:
            combo = [r for r in all_results
                     if r["config"]["tp_ticks"] == tp and r["config"]["sl_ticks"] == sl]
            if combo:
                row.append(f"{np.mean([r['daily_sharpe'] for r in combo]):6.2f}")
            else:
                row.append(f"{'N/A':>6}")
        log.info(f"  TP={tp:2d} " + " ".join(row))

    # ══════════════════════════════════════
    #  PHASE 7: Save results + MLflow
    # ══════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 7: Saving results")
    log.info("=" * 70)

    # Save predictions (in case not cached)
    np.savez_compressed(
        str(OUTPUT_DIR / "entry_predictions.npz"),
        entry_preds=entry_preds,
        labels_30m=labels_all,
        dates=dates_all,
    )

    # All results
    results_clean = clean_for_json([{k: v for k, v in r.items()} for r in all_results])
    with open(str(OUTPUT_DIR / "all_results.json"), "w") as f:
        json.dump(results_clean, f, indent=2, default=str)

    # Top configs with regime
    top_output = clean_for_json({
        "top_configs": [{k: v for k, v in r.items()} for r in all_results[:10]],
        "regime_analyses": regime_results_top,
    })
    with open(str(OUTPUT_DIR / "top_configs_analysis.json"), "w") as f:
        json.dump(top_output, f, indent=2, default=str)

    if mlflow_active:
        try:
            if all_results:
                best = all_results[0]
                best_cfg = best["config"]
                best_es = best.get("exit_stats", {})
                mlflow.log_metric("best_daily_sharpe", best["daily_sharpe"])
                mlflow.log_metric("best_daily_sortino", best["daily_sortino"])
                mlflow.log_metric("best_win_rate", best["win_rate"])
                mlflow.log_metric("best_profit_factor", best["profit_factor"])
                mlflow.log_metric("best_n_trades", best["n_trades"])
                mlflow.log_metric("best_pnl_dollars", best["total_pnl_dollars"])
                mlflow.log_metric("best_tp_rate", best_es.get("tp_rate", 0))
                mlflow.log_metric("best_sl_rate", best_es.get("sl_rate", 0))
                mlflow.log_metric("best_breakeven_wr", best["breakeven_wr"])
                mlflow.log_metric("best_wr_edge", best["wr_edge"])
                mlflow.log_param("best_tp_ticks", best_cfg["tp_ticks"])
                mlflow.log_param("best_sl_ticks", best_cfg["sl_ticks"])
                mlflow.log_param("best_cancel_window", best_cfg["cancel_window"])
                mlflow.log_param("best_max_hold", best_cfg["max_hold"])
                mlflow.log_param("best_entry_threshold", best_cfg["entry_threshold"])

                # Log regime for best
                if best["label"] in regime_results_top:
                    rr = regime_results_top[best["label"]].get("regimes", {})
                    if "regime_gap" in rr:
                        mlflow.log_metric("best_regime_gap", rr["regime_gap"])
                    for reg in ["green", "red", "flat"]:
                        if reg in rr and "sharpe" in rr[reg]:
                            mlflow.log_metric(f"best_{reg}_sharpe", rr[reg]["sharpe"])

            mlflow.log_artifact(str(OUTPUT_DIR / "all_results.json"))
            mlflow.log_artifact(str(OUTPUT_DIR / "top_configs_analysis.json"))
            mlflow.end_run()
            log.info("MLflow run completed successfully")
        except Exception as e:
            log.warning(f"MLflow logging failed: {e}")

    elapsed = time.time() - t0
    log.info(f"\nPassive SL v1 complete in {elapsed/60:.1f} minutes")
    log.info(f"Results saved to {OUTPUT_DIR}")

    # Final summary
    log.info("\n" + "=" * 70)
    log.info("EXECUTIVE SUMMARY")
    log.info("=" * 70)

    if all_results:
        best = all_results[0]
        cfg = best["config"]
        es = best.get("exit_stats", {})
        log.info(f"\nBEST CONFIG: TP={cfg['tp_ticks']}, SL={cfg['sl_ticks']}, "
                 f"CW={cfg['cancel_window']}, MH={cfg['max_hold']}, "
                 f"ET={cfg['entry_threshold']:.0%}")
        log.info(f"  Sharpe={best['daily_sharpe']:.2f}, Sortino={best['daily_sortino']:.2f}")
        log.info(f"  WR={best['win_rate']:.1%} (breakeven={best['breakeven_wr']:.1%}, "
                 f"edge={best['wr_edge']:+.1%})")
        log.info(f"  PF={best['profit_factor']:.2f}, N={best['n_trades']}, "
                 f"Days={best['n_trading_days']}")
        log.info(f"  $PnL={best['total_pnl_dollars']:.0f}, "
                 f"MaxDD=${best['max_dd_dollars']:.0f}")
        log.info(f"  TP%={es.get('tp_rate', 0):.1%}, SL%={es.get('sl_rate', 0):.1%}, "
                 f"TimeStop%={es.get('time_stop_rate', 0):.1%}")
        log.info(f"  AvgWin={es.get('avg_winner', 0):.2f}, AvgLoss={es.get('avg_loser', 0):.2f}")
        log.info(f"  DayConc={best['day_concentration']:.2f} "
                 f"({'PASS' if best['day_concentration_pass'] else 'FAIL'})")

        # Count profitable configs
        profitable = [r for r in all_results if r["total_pnl_dollars"] > 0]
        positive_sharpe = [r for r in all_results if r["daily_sharpe"] > 0]
        regime_pass = [r for r in all_results[:10]
                       if r["label"] in regime_results_top
                       and regime_results_top[r["label"]].get("regimes", {}).get("regime_gap_pass", False)]

        log.info(f"\n  Profitable configs: {len(profitable)}/{len(all_results)}")
        log.info(f"  Positive Sharpe configs: {len(positive_sharpe)}/{len(all_results)}")
        log.info(f"  Regime-agnostic (top 10): {len(regime_pass)}/min(10, {len(all_results)})")


if __name__ == "__main__":
    main()
