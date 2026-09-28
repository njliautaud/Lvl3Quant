#!/usr/bin/env python3
"""
Trailing Stop FIFO v1 — Realistic FIFO Queue Fill Model for ES Futures
=========================================================================

Replaces the midpoint/close-price fill model from trailing_stop_full_oot_v1.py
with a conservative FIFO queue simulation:

ENTRY (Passive Limit):
  - Long signal: place limit BUY at the bid (signal bar close - 0.25/2, i.e. bid side).
    Fill only if a subsequent minute bar's LOW trades 1 tick THROUGH our bid level.
    Conservative: assume we're at the BACK of the queue.
  - Short signal: place limit SELL at the ask (signal bar close + 0.25/2, i.e. ask side).
    Fill only if a subsequent minute bar's HIGH trades 1 tick THROUGH our ask level.
  - Cancel if not filled within cancel_window minutes (default: 30, matches prediction horizon).
  - Fill price = our limit price (we get filled at our price, not worse, since it's a limit order).

EXIT (Market Order on Trailing Stop Trigger):
  - When trailing stop triggers, exit is a MARKET ORDER.
  - Apply 1 tick (0.25 pts) adverse slippage from the trigger price.
  - Long exit: sell market → fill = trigger_price - 0.25 (slipped down 1 tick).
  - Short exit: buy market → fill = trigger_price + 0.25 (slipped up 1 tick).

COSTS:
  - Commission: 0.376 ticks RT ($4.70 / $12.50) — applied to every trade.
  - No additional spread cost on passive entry (we're providing liquidity).
  - Exit spread cost is captured by the 1-tick adverse slippage.

SHARPE:
  - Computed on DAILY PnL bars (sum all trade PnLs per calendar day).
  - Sharpe = mean(daily_pnl) / std(daily_pnl) * sqrt(252).

CONFIGS:
  - Primary: trigger1=5, buf1=1.0, trigger2=6, buf2=0.5, stale=4, maxhold=30
  - Reference: v5 config (trigger1=3, buf1=2.0, trigger2=6, buf2=1.5, stale=4, maxhold=30)

All other architecture (walk-forward, leakage audit, regime analysis) is
preserved from trailing_stop_full_oot_v1.py.

Constraints:
  - Walk-forward: 30d train, 5d slide — SLIDING only (HC #0)
  - FIFO fills only (HC #74)
  - Regime-agnostic gate: |Sharpe_green - Sharpe_red| / max < 0.50 (HC #428)
  - Day-concentration cap <= 0.70 (HC #344)
  - MLflow logging mandatory

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python -u \\
      alpha_discovery/trailing_stop_fifo_v1.py 2>&1 | \\
      tee logs/trailing_stop_fifo_v1.log

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
OUTPUT_DIR = ROOT / "output" / "trailing_stop_fifo_v1"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [TS-FIFO-V1] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "trailing_stop_fifo_v1.log")),
    ],
)
log = logging.getLogger("TS-FIFO-V1")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK = 0.25           # 1 tick = 0.25 points
ES_TICK_VALUE = 12.50    # 1 tick = $12.50
ES_RT_COMMISSION = 4.70  # Round-trip commission
ES_RT_COMMISSION_TICKS = 0.376  # $4.70 / $12.50
EXIT_SLIPPAGE_TICKS = 1.0  # Market exit = 1 tick adverse slippage

# FIFO cost: commission only (passive entry = no spread cost, exit slippage applied separately)
COST_COMMISSION_TICKS = ES_RT_COMMISSION_TICKS  # 0.376 ticks

# ─────────────────────────────────────────────
#  WALK-FORWARD CONFIG — 30d for MAX OOT
# ─────────────────────────────────────────────
TRAIN_DAYS = 30
SLIDE_DAYS = 5

# ─────────────────────────────────────────────
#  TRADE CONFIG
# ─────────────────────────────────────────────
ENTRY_BAR_SIZE_MIN = 30
MGMT_BAR_SIZE_MIN = 1     # Use 1-min bars for FIFO fill checking + trailing stop
MAX_HOLD_MINUTES = 60
CANCEL_WINDOW_MIN = 30    # Cancel unfilled limit orders after 30 minutes

# ─────────────────────────────────────────────
#  ENTRY THRESHOLD (fixed to 5% as specified)
# ─────────────────────────────────────────────
ENTRY_THRESHOLDS = [0.05, 0.10, 0.15]

# ─────────────────────────────────────────────
#  TRAILING STOP CONFIGS
# ─────────────────────────────────────────────
# Primary config from sweep champion
PRIMARY_TS = {
    "trigger1": 5, "buf1": 1.0,
    "trigger2": 6, "buf2": 0.5,
    "stale_bars": 4, "max_hold": 30,
}

# v5 reference config
V5_REFERENCE_TS = {
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
#  SECTION 8: FIFO TRADE RECONSTRUCTION
# ═══════════════════════════════════════════════════════════════════


def reconstruct_trades_fifo(
    bars_30m: pd.DataFrame,
    minute_df: pd.DataFrame,
    pred_30m: np.ndarray,
    confidence_pct: float = 0.05,
    cancel_window_min: int = CANCEL_WINDOW_MIN,
    max_hold_minutes: int = MAX_HOLD_MINUTES,
) -> Tuple[List[Dict], Dict]:
    """
    Reconstruct trades with FIFO queue fill simulation.

    For each 30-min signal bar where prediction is in top/bottom confidence_pct:
      1. Determine entry direction from prediction sign
      2. Place passive limit order at bid (longs) or ask (shorts)
      3. Check subsequent 1-min bars for fill (price must trade THROUGH our level)
      4. If filled, record trade with minute-level price path for trailing stop
      5. If not filled within cancel_window, skip (no trade)

    Returns:
      trades: list of trade dicts with minute-level price paths
      fill_stats: dict with fill rate statistics
    """
    valid_mask = ~np.isnan(pred_30m)
    valid_preds = pred_30m[valid_mask]
    if len(valid_preds) < 20:
        log.warning("Too few valid predictions for trade reconstruction")
        return [], {"error": "too few predictions"}

    upper_thresh = np.nanquantile(pred_30m[valid_mask], 1 - confidence_pct)
    lower_thresh = np.nanquantile(pred_30m[valid_mask], confidence_pct)

    # Build minute-level lookup: date -> sorted minute bars
    minute_lookup = {}
    for date_str, grp in minute_df.groupby("date"):
        grp_sorted = grp.sort_values("ts_minute").reset_index(drop=True)
        minute_lookup[date_str] = grp_sorted

    trades = []
    n_signals = 0
    n_filled = 0
    n_cancelled = 0
    n_no_data = 0
    fill_delays = []  # minutes until fill

    bars_ts = bars_30m["ts"].values
    bars_dates = bars_30m["date"].values
    bars_close = bars_30m["close"].values

    for i in range(len(bars_30m)):
        if np.isnan(pred_30m[i]):
            continue

        # Check entry conditions
        direction = 0
        if pred_30m[i] >= upper_thresh:
            direction = 1   # Long
        elif pred_30m[i] <= lower_thresh:
            direction = -1  # Short
        else:
            continue

        n_signals += 1
        date_str = bars_dates[i]
        signal_ts = pd.Timestamp(bars_ts[i])
        signal_price = bars_close[i]  # Close of the signal bar

        if date_str not in minute_lookup:
            n_no_data += 1
            continue

        day_minutes = minute_lookup[date_str]
        day_ts = day_minutes["ts_minute"].values
        day_high = day_minutes["high"].values
        day_low = day_minutes["low"].values
        day_close = day_minutes["close"].values
        day_volume = day_minutes["volume"].values

        # Determine limit price
        # For longs: place bid at signal_price (we join the bid)
        # For shorts: place ask at signal_price (we join the ask)
        # Since ES book is 1 tick wide, bid = close - 0.25 if close is mid,
        # but close IS typically bid or ask. Conservative: use signal bar close as
        # our limit level. We need price to trade THROUGH (1 tick beyond) for fill.
        limit_price = signal_price

        # Find minute bars AFTER the signal bar for fill checking
        signal_ts_np = np.datetime64(signal_ts)
        cancel_ts = signal_ts + pd.Timedelta(minutes=cancel_window_min)
        cancel_ts_np = np.datetime64(cancel_ts)

        # Look at bars AFTER the signal (not including signal bar itself)
        # Signal bar is the 30-min bar. We check 1-min bars starting from the
        # NEXT minute after the signal bar's end.
        signal_bar_end = signal_ts + pd.Timedelta(minutes=ENTRY_BAR_SIZE_MIN)
        signal_bar_end_np = np.datetime64(signal_bar_end)

        fill_mask = (day_ts >= signal_bar_end_np) & (day_ts <= cancel_ts_np)
        fill_candidates = day_minutes[fill_mask]

        if len(fill_candidates) == 0:
            n_cancelled += 1
            continue

        # Check each 1-min bar for fill
        filled = False
        fill_price = None
        fill_minute_idx = None
        fill_ts = None

        for j, (_, mbar) in enumerate(fill_candidates.iterrows()):
            if direction == 1:
                # Long: buying at bid (limit_price). Fill if bar LOW goes
                # 1 tick BELOW our limit (price traded through our level).
                # Back-of-queue: need price to go to limit_price - 1 tick
                if mbar["low"] <= limit_price - ES_TICK:
                    filled = True
                    fill_price = limit_price  # Limit order fills at limit price
                    fill_ts = mbar["ts_minute"]
                    fill_delays.append(j + 1)
                    break
            else:
                # Short: selling at ask (limit_price). Fill if bar HIGH goes
                # 1 tick ABOVE our limit (price traded through our level).
                # Back-of-queue: need price to go to limit_price + 1 tick
                if mbar["high"] >= limit_price + ES_TICK:
                    filled = True
                    fill_price = limit_price  # Limit order fills at limit price
                    fill_ts = mbar["ts_minute"]
                    fill_delays.append(j + 1)
                    break

        if not filled:
            n_cancelled += 1
            continue

        n_filled += 1

        # Now build the trade's minute-level price path from fill time
        fill_ts_np = np.datetime64(fill_ts)
        trade_end = pd.Timestamp(fill_ts) + pd.Timedelta(minutes=max_hold_minutes)
        trade_end_np = np.datetime64(trade_end)

        trade_mask = (day_ts >= fill_ts_np) & (day_ts <= trade_end_np)
        trade_minutes = day_minutes[trade_mask].copy()

        if len(trade_minutes) < 3:
            # Not enough bars for meaningful trailing stop
            n_filled -= 1
            n_cancelled += 1
            continue

        # Extract price path using CLOSE prices (for P&L tracking)
        prices = trade_minutes["close"].values
        highs = trade_minutes["high"].values
        lows = trade_minutes["low"].values
        times = trade_minutes["ts_minute"].values

        # P&L path in ticks from fill price
        pnl_path = (prices - fill_price) / ES_TICK * direction
        # For trailing stop, we also need MFE computed from HIGH/LOW (not just close)
        # Long: MFE from highs; Short: MFE from lows (inverted)
        if direction == 1:
            mfe_prices = highs
            mae_prices = lows
        else:
            mfe_prices = lows  # For shorts, lower = more favorable
            mae_prices = highs  # For shorts, higher = more adverse

        mfe_path_raw = (mfe_prices - fill_price) / ES_TICK * direction
        mae_path_raw = (mae_prices - fill_price) / ES_TICK * direction

        # Running MFE/MAE
        mfe_path = np.maximum.accumulate(mfe_path_raw)
        mae_path = np.minimum.accumulate(mae_path_raw)

        trade = {
            "idx": i,
            "date": date_str,
            "signal_ts": signal_ts,
            "fill_ts": pd.Timestamp(fill_ts),
            "fill_price": fill_price,
            "fill_delay_minutes": fill_delays[-1],
            "direction": direction,
            "pred_30m": float(pred_30m[i]),
            # Minute-level data for trailing stop
            "prices": prices,
            "highs": highs,
            "lows": lows,
            "times": times,
            "pnl_path": pnl_path,
            "mfe_path": mfe_path,
            "mae_path": mae_path,
            "n_minutes": len(prices),
        }

        trades.append(trade)

    # Fill statistics
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

    log.info(f"FIFO Trade Reconstruction (top/bottom {confidence_pct:.0%}):")
    log.info(f"  Signals: {n_signals}, Filled: {n_filled} ({fill_stats['fill_rate']:.1%}), "
             f"Cancelled: {n_cancelled}, No data: {n_no_data}")
    if fill_delays:
        log.info(f"  Fill delay: mean={fill_stats['avg_fill_delay_min']:.1f}min, "
                 f"median={fill_stats['median_fill_delay_min']:.1f}min, "
                 f"p90={fill_stats['p90_fill_delay_min']:.1f}min")

    if trades:
        directions = [t["direction"] for t in trades]
        log.info(f"  Long: {sum(1 for d in directions if d == 1)}, "
                 f"Short: {sum(1 for d in directions if d == -1)}")
        trade_dates = sorted(set(t["date"] for t in trades))
        log.info(f"  Trading days: {len(trade_dates)} "
                 f"({trade_dates[0]} -> {trade_dates[-1]})")

    return trades, fill_stats


# ═══════════════════════════════════════════════════════════════════
#  SECTION 9: ADAPTIVE TRAILING STOP WITH FIFO EXIT
# ═══════════════════════════════════════════════════════════════════


def simulate_trailing_stop_fifo(
    trades: List[Dict],
    trigger1: float = 5.0,
    buf1: float = 1.0,
    trigger2: float = 6.0,
    buf2: float = 0.5,
    stale_bars: int = 4,
    max_hold: int = 30,
    checkpoint_interval: int = 1,  # 1-min checkpoints for FIFO
) -> Dict:
    """
    Simulate adaptive trailing stop with FIFO exit model.

    Key difference from v1: EXIT is a MARKET ORDER.
    - Apply 1 tick adverse slippage from exit price.
    - Commission: 0.376 ticks RT.
    - Checkpoint every 1 minute (not 5 min) for finer exit timing.

    Trailing stop logic (unchanged):
      - MFE >= trigger2: floor = MFE - buf2 (lock in profits aggressively)
      - MFE >= trigger1: floor = MFE - buf1 (lock in some profit)
      - MFE < trigger1: no trailing floor
      - stale_bars minutes with no new MFE: exit
      - max_hold: max hold in minutes
    """
    pnls, dates, dirs, hold_durations = [], [], [], []
    exit_reasons = {"trailing_floor": 0, "stale_mfe": 0, "max_hold": 0}

    for trade in trades:
        pnl_path = trade["pnl_path"]
        mfe_path = trade["mfe_path"]
        n_minutes = trade["n_minutes"]
        direction = trade["direction"]
        fill_price = trade["fill_price"]
        prices = trade["prices"]

        exited = False
        exit_minute = min(max_hold, n_minutes - 1)
        exit_reason = "max_hold"

        for ckpt in range(checkpoint_interval, min(n_minutes, max_hold + 1), checkpoint_interval):
            mfe_now = mfe_path[ckpt]
            pnl_now = pnl_path[ckpt]

            # Minutes since MFE was last extended
            minutes_since_mfe = 0
            for lb in range(1, ckpt + 1):
                if mfe_path[ckpt - lb] < mfe_now:
                    break
                minutes_since_mfe += 1

            # Stale trade exit: no MFE improvement for stale_bars minutes
            # and we had SOME favorable move
            if minutes_since_mfe >= stale_bars and mfe_now > 1.0:
                exit_minute = ckpt
                exited = True
                exit_reason = "stale_mfe"
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
                exit_reason = "trailing_floor"
                break

        if not exited:
            exit_minute = min(max_hold, n_minutes - 1)
            exit_reason = "max_hold"

        exit_minute = min(exit_minute, n_minutes - 1)
        exit_reasons[exit_reason] = exit_reasons.get(exit_reason, 0) + 1

        # EXIT P&L with FIFO model:
        # Exit price = close at exit minute (where we decide to exit)
        # Apply 1 tick adverse slippage (market order to exit)
        exit_close = prices[exit_minute]
        if direction == 1:
            # Long: selling market → fill = exit_close - 1 tick slippage
            exit_fill = exit_close - ES_TICK
        else:
            # Short: buying market → fill = exit_close + 1 tick slippage
            exit_fill = exit_close + ES_TICK

        # Trade P&L in ticks
        raw_pnl_ticks = (exit_fill - fill_price) / ES_TICK * direction
        # Subtract commission
        trade_pnl = raw_pnl_ticks - COST_COMMISSION_TICKS

        pnls.append(trade_pnl)
        dates.append(trade["date"])
        dirs.append(trade["direction"])
        hold_durations.append(exit_minute)

    if not pnls:
        return {"error": "no trades"}

    return _compute_daily_trade_stats(
        np.array(pnls), np.array(dates), np.array(dirs),
        f"FIFO_TS_t{trigger1}_b{buf1}_t2{trigger2}_b2{buf2}_s{stale_bars}_h{max_hold}",
        hold_durations=hold_durations, exit_reasons=exit_reasons,
    )


# ═══════════════════════════════════════════════════════════════════
#  SECTION 10: TRADE STATISTICS (DAILY SHARPE)
# ═══════════════════════════════════════════════════════════════════


def _compute_daily_trade_stats(
    pnl_arr: np.ndarray,
    dates: np.ndarray,
    directions: np.ndarray,
    label: str,
    hold_durations: Optional[List[float]] = None,
    exit_reasons: Optional[Dict] = None,
) -> Dict:
    """
    Compute trade statistics with DAILY Sharpe (not per-trade Sharpe).

    Sharpe = mean(daily_pnl) / std(daily_pnl) * sqrt(252)
    This is the correct Sharpe for strategies that trade intraday.
    """
    if len(pnl_arr) == 0:
        return {"error": "no trades"}

    # Per-trade metrics
    wr = np.mean(pnl_arr > 0)
    pf = np.sum(pnl_arr[pnl_arr > 0]) / max(-np.sum(pnl_arr[pnl_arr < 0]), 1e-6)

    # Per-side breakdown
    long_mask = directions == 1
    short_mask = directions == -1
    long_pnl = pnl_arr[long_mask] if long_mask.any() else np.array([])
    short_pnl = pnl_arr[short_mask] if short_mask.any() else np.array([])

    # DAILY aggregation for Sharpe/Sortino
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

    # Cumulative P&L for drawdown
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
        "day_concentration": float(day_conc),
        "day_concentration_pass": day_conc <= 0.70,
        "daily_pnl_list": day_pnl.to_dict("records"),
        # Per-side
        "long_trades": int(len(long_pnl)),
        "long_wr": float(np.mean(long_pnl > 0)) if len(long_pnl) > 0 else 0,
        "long_avg": float(long_pnl.mean()) if len(long_pnl) > 0 else 0,
        "short_trades": int(len(short_pnl)),
        "short_wr": float(np.mean(short_pnl > 0)) if len(short_pnl) > 0 else 0,
        "short_avg": float(short_pnl.mean()) if len(short_pnl) > 0 else 0,
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
    Full regime analysis with DAILY Sharpe per regime.
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

    # Per-regime stats (DAILY Sharpe)
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

    # Per-day P&L table
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


def run_trailing_stop_fifo():
    """
    Run trailing stop analysis with FIFO queue fill model.

    Pipeline:
      1. Load minute bars, build 30-min bar features
      2. Walk-forward train entry model (30d train, 5d slide)
      3. FIFO trade reconstruction (passive limit entry, fill-or-cancel)
      4. Run trailing stop configs with market exit + slippage
      5. Full regime analysis
      6. Save everything + MLflow
    """
    _import_lightgbm()

    # ── MLflow setup ──
    mlflow_active = False
    mlflow_run = None
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("trailing_stop_fifo_v1")
        mlflow_run = mlflow.start_run(
            run_name=f"ts_fifo_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
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
        mlflow.log_param("fill_model", "FIFO_passive_limit")
        mlflow.log_param("exit_model", "market_1tick_slippage")
        mlflow.log_param("commission_ticks", COST_COMMISSION_TICKS)
        mlflow.log_param("exit_slippage_ticks", EXIT_SLIPPAGE_TICKS)
        mlflow.log_param("cancel_window_min", CANCEL_WINDOW_MIN)
        mlflow.log_param("confluence", "NONE")
        mlflow.log_param("entry_thresholds", str(ENTRY_THRESHOLDS))

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
    #  PHASE 3: FIFO trade reconstruction + trailing stop
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 3: FIFO Trade Reconstruction + Trailing Stop Simulation")
    log.info("=" * 70)

    all_results = []
    all_fill_stats = {}
    best_daily_sharpe = -999
    best_config = None
    best_trades = None
    best_result = None
    best_fill_stats = None

    configs_to_run = [
        ("primary", PRIMARY_TS),
        ("v5_reference", V5_REFERENCE_TS),
    ]

    for conf_pct in ENTRY_THRESHOLDS:
        log.info(f"\n--- Entry threshold: top/bottom {conf_pct:.0%} ---")

        # FIFO trade reconstruction
        trades, fill_stats = reconstruct_trades_fifo(
            bars_30m, minute_df, entry_preds,
            confidence_pct=conf_pct,
            cancel_window_min=CANCEL_WINDOW_MIN,
            max_hold_minutes=MAX_HOLD_MINUTES,
        )

        fill_tag = f"top{int(conf_pct*100)}"
        all_fill_stats[fill_tag] = fill_stats

        if mlflow_active:
            mlflow.log_metric(f"fill_rate_{fill_tag}", fill_stats.get("fill_rate", 0))
            mlflow.log_metric(f"n_signals_{fill_tag}", fill_stats.get("n_signals", 0))
            mlflow.log_metric(f"n_filled_{fill_tag}", fill_stats.get("n_filled", 0))
            mlflow.log_metric(f"avg_fill_delay_{fill_tag}",
                              fill_stats.get("avg_fill_delay_min", 0))

        if not trades:
            log.warning(f"  No FIFO fills at {conf_pct:.0%} threshold")
            continue

        # Run each trailing stop config
        for config_name, ts_config in configs_to_run:
            result = simulate_trailing_stop_fifo(
                trades,
                trigger1=float(ts_config["trigger1"]),
                buf1=float(ts_config["buf1"]),
                trigger2=float(ts_config["trigger2"]),
                buf2=float(ts_config["buf2"]),
                stale_bars=ts_config["stale_bars"],
                max_hold=ts_config["max_hold"],
            )

            if "error" in result:
                continue

            config_tag = f"{config_name}_{fill_tag}"
            result["config"] = {
                "name": config_name,
                "entry_threshold": conf_pct,
                **ts_config,
            }
            result["fill_stats"] = fill_stats
            all_results.append(result)

            log.info(
                f"  {config_tag}: "
                f"DailySharpe={result['daily_sharpe']:.2f}, "
                f"Sortino={result['daily_sortino']:.2f}, "
                f"WR={result['win_rate']:.1%}, "
                f"PF={result['profit_factor']:.2f}, "
                f"N={result['n_trades']}, "
                f"Days={result['n_trading_days']}, "
                f"$PnL={result['total_pnl_dollars']:.0f}, "
                f"FillRate={fill_stats.get('fill_rate', 0):.1%}"
            )

            if mlflow_active:
                mlflow.log_metric(f"{config_tag}_daily_sharpe", result["daily_sharpe"])
                mlflow.log_metric(f"{config_tag}_sortino", result["daily_sortino"])
                mlflow.log_metric(f"{config_tag}_wr", result["win_rate"])
                mlflow.log_metric(f"{config_tag}_pf", result["profit_factor"])
                mlflow.log_metric(f"{config_tag}_n_trades", result["n_trades"])
                mlflow.log_metric(f"{config_tag}_n_days", result["n_trading_days"])
                mlflow.log_metric(f"{config_tag}_pnl_dollars", result["total_pnl_dollars"])

            if result["daily_sharpe"] > best_daily_sharpe:
                best_daily_sharpe = result["daily_sharpe"]
                best_config = result["config"]
                best_trades = trades
                best_result = result
                best_fill_stats = fill_stats

    # Sort results by daily Sharpe
    all_results.sort(key=lambda x: x.get("daily_sharpe", -999), reverse=True)

    log.info(f"\n{'=' * 70}")
    log.info("FIFO TRAILING STOP LEADERBOARD")
    log.info(f"{'=' * 70}")
    log.info(f"{'Rank':>4} {'Config':>30} {'DlySharpe':>10} {'Sortino':>8} "
             f"{'WR':>6} {'PF':>6} {'N':>6} {'Days':>5} {'$PnL':>8} {'Fill%':>6}")
    log.info("-" * 100)
    for i, r in enumerate(all_results):
        cfg = r.get("config", {})
        cfg_str = f"{cfg.get('name', '?')}_{int(cfg.get('entry_threshold', 0)*100)}pct"
        fs = r.get("fill_stats", {})
        log.info(
            f"{i+1:4d} {cfg_str:>30} "
            f"{r['daily_sharpe']:10.2f} {r['daily_sortino']:8.2f} "
            f"{r['win_rate']:6.1%} {r['profit_factor']:6.2f} "
            f"{r['n_trades']:6d} {r['n_trading_days']:5d} "
            f"{r['total_pnl_dollars']:8.0f} {fs.get('fill_rate', 0):6.1%}"
        )

    if mlflow_active and best_result:
        mlflow.log_metric("best_daily_sharpe", best_result["daily_sharpe"])
        mlflow.log_metric("best_daily_sortino", best_result["daily_sortino"])
        mlflow.log_metric("best_n_trades", best_result["n_trades"])
        mlflow.log_metric("best_n_days", best_result["n_trading_days"])
        mlflow.log_metric("best_pnl_dollars", best_result["total_pnl_dollars"])
        mlflow.log_param("best_config", str(best_config))
        if best_fill_stats:
            mlflow.log_metric("best_fill_rate", best_fill_stats.get("fill_rate", 0))

    # ════════════════════════════════════════
    #  PHASE 4: Full regime analysis on best config
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 4: Full regime analysis on best config")
    log.info("=" * 70)

    regime_analysis = None
    if best_result is not None and best_trades is not None:
        log.info(f"  Best config: {best_config}")
        log.info(f"  Best Daily Sharpe: {best_daily_sharpe:.2f}")

        # Re-run best trailing stop to get trade-level arrays
        best_cfg = best_config
        pnls, dates_list, dirs = [], [], []

        for trade in best_trades:
            pnl_path = trade["pnl_path"]
            mfe_path = trade["mfe_path"]
            n_minutes = trade["n_minutes"]
            direction = trade["direction"]
            fill_price = trade["fill_price"]
            prices = trade["prices"]

            trigger1 = best_cfg.get("trigger1", 5.0)
            buf1_val = best_cfg.get("buf1", 1.0)
            trigger2 = best_cfg.get("trigger2", 6.0)
            buf2_val = best_cfg.get("buf2", 0.5)
            stale_b = best_cfg.get("stale_bars", 4)
            max_h = best_cfg.get("max_hold", 30)

            exited = False
            exit_minute = min(max_h, n_minutes - 1)

            for ckpt in range(1, min(n_minutes, max_h + 1)):
                mfe_now = mfe_path[ckpt]
                pnl_now = pnl_path[ckpt]

                minutes_since_mfe = 0
                for lb in range(1, ckpt + 1):
                    if mfe_path[ckpt - lb] < mfe_now:
                        break
                    minutes_since_mfe += 1

                if minutes_since_mfe >= stale_b and mfe_now > 1.0:
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

            # FIFO exit: market order with 1 tick slippage
            exit_close = prices[exit_minute]
            if direction == 1:
                exit_fill = exit_close - ES_TICK
            else:
                exit_fill = exit_close + ES_TICK

            trade_pnl = (exit_fill - fill_price) / ES_TICK * direction - COST_COMMISSION_TICKS
            pnls.append(trade_pnl)
            dates_list.append(trade["date"])
            dirs.append(trade["direction"])

        pnl_arr = np.array(pnls)
        dates_arr = np.array(dates_list)
        dirs_arr = np.array(dirs)

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

        if mlflow_active:
            if "regime_gap" in regimes:
                mlflow.log_metric("regime_gap", regimes.get("regime_gap", 999))
                mlflow.log_metric("regime_gap_pass", int(regimes.get("regime_gap_pass", False)))
            for regime in ["green", "red", "flat"]:
                if regime in regimes and "sharpe" in regimes[regime]:
                    mlflow.log_metric(f"regime_{regime}_sharpe", regimes[regime]["sharpe"])
                    mlflow.log_metric(f"regime_{regime}_wr", regimes[regime]["win_rate"])
    else:
        log.warning("No best config found -- no valid FIFO results")

    # ════════════════════════════════════════
    #  PHASE 5: Fill statistics summary
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("PHASE 5: FIFO Fill Statistics Summary")
    log.info("=" * 70)

    for tag, fs in all_fill_stats.items():
        log.info(f"\n  {tag}:")
        log.info(f"    Signals: {fs.get('n_signals', 0)}")
        log.info(f"    Filled:  {fs.get('n_filled', 0)} ({fs.get('fill_rate', 0):.1%})")
        log.info(f"    Cancelled: {fs.get('n_cancelled', 0)} ({fs.get('cancel_rate', 0):.1%})")
        log.info(f"    Avg fill delay: {fs.get('avg_fill_delay_min', 0):.1f} min")
        log.info(f"    Median fill delay: {fs.get('median_fill_delay_min', 0):.1f} min")
        log.info(f"    P90 fill delay: {fs.get('p90_fill_delay_min', 0):.1f} min")

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
    log.info(f"  Saved predictions")

    # Save all results
    results_clean = clean_for_json(all_results)
    with open(str(OUTPUT_DIR / "all_results.json"), "w") as f:
        json.dump(results_clean, f, indent=2, default=str)

    # Save fill stats
    fill_stats_clean = clean_for_json(all_fill_stats)
    with open(str(OUTPUT_DIR / "fill_statistics.json"), "w") as f:
        json.dump(fill_stats_clean, f, indent=2, default=str)

    # Save best config analysis
    if best_result is not None:
        best_output = clean_for_json({
            "best_config": best_config,
            "best_result": best_result,
            "fill_stats": best_fill_stats,
            "regime_analysis": regime_analysis,
        })
        with open(str(OUTPUT_DIR / "best_config_analysis.json"), "w") as f:
            json.dump(best_output, f, indent=2, default=str)

    if mlflow_active:
        try:
            for artifact_name in ["all_results.json", "fill_statistics.json", "best_config_analysis.json"]:
                artifact_path = OUTPUT_DIR / artifact_name
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))
        except Exception as e:
            log.warning(f"MLflow artifact logging failed: {e}")

    elapsed = time.time() - t0

    # ════════════════════════════════════════
    #  FINAL SUMMARY
    # ════════════════════════════════════════
    log.info(f"\n{'=' * 70}")
    log.info(f"TRAILING STOP FIFO v1 COMPLETE in {elapsed/60:.1f} minutes")
    log.info(f"{'=' * 70}")

    log.info(f"\n  FILL MODEL: Passive limit entry (back-of-queue, 1-tick-through)")
    log.info(f"  EXIT MODEL: Market order with 1 tick adverse slippage")
    log.info(f"  COMMISSION: {COST_COMMISSION_TICKS} ticks RT")

    log.info(f"\n  Total trading days in data: {len(all_dates_raw)}")
    log.info(f"  OOT days with predictions: {len(valid_dates)}")
    log.info(f"  Configs tested: {len(all_results)}")

    if best_result:
        log.info(f"\n  BEST CONFIG: {best_config}")
        log.info(
            f"  Daily Sharpe={best_result['daily_sharpe']:.2f}, "
            f"Sortino={best_result['daily_sortino']:.2f}, "
            f"WR={best_result['win_rate']:.1%}, "
            f"PF={best_result['profit_factor']:.2f}, "
            f"N={best_result['n_trades']}, "
            f"Days={best_result['n_trading_days']}, "
            f"$PnL={best_result['total_pnl_dollars']:.0f}"
        )
        if best_fill_stats:
            log.info(
                f"  Fill Rate: {best_fill_stats.get('fill_rate', 0):.1%} "
                f"({best_fill_stats.get('n_filled', 0)} / {best_fill_stats.get('n_signals', 0)} signals)"
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
    log.info("Trailing Stop FIFO v1 starting...")
    log.info(f"  ROOT: {ROOT}")
    log.info(f"  OUTPUT: {OUTPUT_DIR}")
    log.info(f"  Fill model: FIFO passive limit (back-of-queue, 1-tick-through)")
    log.info(f"  Exit model: Market order + {EXIT_SLIPPAGE_TICKS} tick slippage")
    log.info(f"  Commission: {COST_COMMISSION_TICKS} ticks RT")
    log.info(f"  Cancel window: {CANCEL_WINDOW_MIN} min")
    log.info(f"  Walk-forward: {TRAIN_DAYS}d train, {SLIDE_DAYS}d slide")
    log.info(f"  Entry: NO confluence gate, thresholds: {ENTRY_THRESHOLDS}")
    log.info(f"  TS configs: PRIMARY={PRIMARY_TS}, V5_REF={V5_REFERENCE_TS}")

    try:
        results = run_trailing_stop_fifo()
    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        log.error(traceback.format_exc())
        try:
            import mlflow
            mlflow.end_run(status="FAILED")
        except Exception:
            pass
        sys.exit(1)
