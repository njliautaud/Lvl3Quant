#!/usr/bin/env python3
"""
Trailing Stop Parameter Sweep v1
=================================

Verifies that the trailing stop finding from Trade Management v5 is NOT overfit
to specific parameter choices (MFE trigger=3/6, buffer=2/1.5, stale=4 bars).

Sweeps ALL trailing stop parameters across 5,760 configs:
  - MFE trigger 1:     [2, 3, 4, 5] ticks
  - Buffer 1:          [1.0, 1.5, 2.0, 2.5, 3.0] ticks
  - MFE trigger 2:     [5, 6, 8, 10] ticks (tighter trailing at higher MFE)
  - Buffer 2:          [0.5, 1.0, 1.5, 2.0] ticks
  - Stale bars:        [2, 3, 4, 6, 8, 999] (999 = disabled)
  - Max hold:          [30, 45, 60] minutes

Each config is a pure simulation pass (no retraining) on the same reconstructed
trades from the v5 entry model walk-forward. This isolates trailing stop
robustness from entry quality.

KEY OUTPUTS:
  - Full sweep CSV with all 5,760 configs and their metrics
  - Top/bottom 20 configs JSON
  - Robustness report: % of configs with Sharpe > 4.0
  - Stability heatmaps: marginal Sharpe over (trigger1, buffer1)
  - MLflow experiment: trailing_stop_sweep_v1

CONSTRAINTS:
  - Walk-forward: 40d train, 5d slide — SLIDING only (HC #0)
  - Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission)
  - Ranking: min(Sharpe, Sortino/2) to find ROBUST configs, not max-Sharpe

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python -u \\
      alpha_discovery/trailing_stop_sweep_v1.py 2>&1 | \\
      tee logs/trailing_stop_sweep_v1.log

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
QUEUE_FEATURE_DIR = ROOT / "data" / "queue_augmented_features"
OUTPUT_DIR = ROOT / "output" / "trailing_stop_sweep_v1"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [TS-SWEEP] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "trailing_stop_sweep_v1.log")),
    ],
)
log = logging.getLogger("TS-SWEEP")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376
COST_RT_TICKS = 2.376  # market entry + market exit + RT commission

# ─────────────────────────────────────────────
#  WALK-FORWARD CONFIG (matches v5 exactly)
# ─────────────────────────────────────────────
TRAIN_DAYS = 40
SLIDE_DAYS = 5

# ─────────────────────────────────────────────
#  ENTRY CONFIG (matches v5 exactly)
# ─────────────────────────────────────────────
ENTRY_BAR_SIZE_MIN = 30
MGMT_BAR_SIZE_MIN = 5          # management checkpoint resolution
STATIC_HOLD_MINUTES = 30
CONFLUENCE_MIN = 2
ENTRY_CONFIDENCE_PCT = 0.10
IMPROVE_THRESHOLD_TICKS = 0.5

# ─────────────────────────────────────────────
#  SWEEP GRID
# ─────────────────────────────────────────────
SWEEP_TRIGGER1 = [2, 3, 4, 5]                   # MFE ticks to activate tier 1
SWEEP_BUFFER1 = [1.0, 1.5, 2.0, 2.5, 3.0]       # ticks below MFE for tier 1 floor
SWEEP_TRIGGER2 = [5, 6, 8, 10]                   # MFE ticks to activate tier 2
SWEEP_BUFFER2 = [0.5, 1.0, 1.5, 2.0]             # ticks below MFE for tier 2 floor
SWEEP_STALE_BARS = [2, 3, 4, 6, 8, 999]          # 999 = disabled
SWEEP_MAX_HOLD = [30, 45, 60]                     # minutes
CHECKPOINT_INTERVAL = 5                           # fixed at 5 min

# V5 reference config for comparison
V5_REF_CONFIG = {
    "trigger1": 3, "buffer1": 2.0,
    "trigger2": 6, "buffer2": 1.5,
    "stale_bars": 4, "max_hold": 60,
}

# ─────────────────────────────────────────────
#  LGBM PARAMS (entry model only — matches v5)
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
#  DATA LOADING (copied from v5)
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
#  BAR AGGREGATION + FEATURES (copied from v5)
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
#  OFI EXHAUSTION SIGNAL (copied from v5)
# ═══════════════════════════════════════════════════════════════════


def compute_ofi_exhaustion_signal(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Compute the OFI exhaustion counter-trend signal on minute bars."""
    df = minute_df.copy()
    df["ts_minute"] = pd.to_datetime(df["ts_minute"], utc=True)
    df = df.sort_values("ts_minute").reset_index(drop=True)

    ofi_col = "ofi_1min"
    df["ofi_rmean"] = df.groupby("date")[ofi_col].transform(
        lambda x: x.rolling(60, min_periods=20).mean()
    )
    df["ofi_rstd"] = df.groupby("date")[ofi_col].transform(
        lambda x: x.rolling(60, min_periods=20).std()
    )
    df["ofi_z"] = (df[ofi_col] - df["ofi_rmean"]) / df["ofi_rstd"].clip(lower=1e-6)

    df["vol_rmean"] = df.groupby("date")["volume"].transform(
        lambda x: x.rolling(60, min_periods=20).mean()
    )
    df["vol_rstd"] = df.groupby("date")["volume"].transform(
        lambda x: x.rolling(60, min_periods=20).std()
    )
    df["vol_z"] = (df["volume"] - df["vol_rmean"]) / df["vol_rstd"].clip(lower=1e-6)

    df["ret_30m"] = df.groupby("date")["close"].transform(lambda x: x.pct_change(30))

    ofi_thresh = 2.5
    vol_thresh = 1.0
    signal = np.zeros(len(df), dtype=np.float32)

    bull_exhaust = (df["ofi_z"] > ofi_thresh) & (df["vol_z"] > vol_thresh) & (df["ret_30m"] > 0)
    signal[bull_exhaust.values] = -1.0

    bear_exhaust = (df["ofi_z"] < -ofi_thresh) & (df["vol_z"] > vol_thresh) & (df["ret_30m"] < 0)
    signal[bear_exhaust.values] = 1.0

    ofi_z_vals = df["ofi_z"].values
    for i in range(len(signal)):
        if signal[i] != 0:
            signal[i] *= min(abs(ofi_z_vals[i]) / ofi_thresh, 3.0)

    result = pd.DataFrame({
        "date": df["date"].values,
        "ts_minute": df["ts_minute"].values,
        "ofi_exhaust_signal": signal,
        "ofi_z": df["ofi_z"].values,
        "vol_z": df["vol_z"].values,
    })

    n_signals = (signal != 0).sum()
    log.info(f"OFI exhaustion: {n_signals} signal bars out of {len(df)} total "
             f"({n_signals/len(df)*100:.1f}%)")
    return result


# ═══════════════════════════════════════════════════════════════════
#  FEATURE COLUMNS + FORWARD LABELS (copied from v5)
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

    df[f"trade_quality_{horizon_label}"] = (
        fwd_ticks.abs() > 2 * COST_RT_TICKS
    ).astype(np.float32)

    log.info(
        f"Forward labels ({horizon_label}): "
        f"{(~fwd_ticks.isna()).sum():,} valid, "
        f"{(df[f'direction_{horizon_label}'] == 1).sum():,} long, "
        f"{(df[f'direction_{horizon_label}'] == -1).sum():,} short"
    )
    return df


# ═══════════════════════════════════════════════════════════════════
#  LEAKAGE AUDIT (copied from v5)
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
#  ENTRY MODEL TRAINING (copied from v5)
# ═══════════════════════════════════════════════════════════════════


def train_entry_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    fold_idx: int,
    horizon_label: str,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Train a Level-1 LightGBM entry model for a single horizon."""
    _import_lightgbm()

    train_valid = ~np.isnan(y_train)
    val_valid = ~np.isnan(y_val)

    if train_valid.sum() < 50 or val_valid.sum() < 10:
        log.warning(
            f"  L1 {horizon_label} fold {fold_idx}: too few valid "
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
        log.info(f"  L1 {horizon_label} fold {fold_idx}: IC={ic:.4f}, "
                 f"best_iter={model.best_iteration}")

    return model, train_preds, val_preds


# ═══════════════════════════════════════════════════════════════════
#  CONFLUENCE SCORING (copied from v5)
# ═══════════════════════════════════════════════════════════════════


def compute_confluence_scores(
    pred_30m: np.ndarray,
    pred_1h: np.ndarray,
    ofi_signal: np.ndarray,
    bars_30m: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Compute confluence scores (6 signals). See v5 for full docstring."""
    n = len(pred_30m)
    confluence_score = np.zeros(n, dtype=np.float32)
    confluence_direction = np.zeros(n, dtype=np.float32)
    confluence_detail = np.zeros(n, dtype=np.float32)

    pred_30m_abs = np.abs(pred_30m)
    pred_30m_pctile = np.zeros(n, dtype=np.float32)
    for i in range(100, n):
        window = pred_30m_abs[max(0, i - 200):i]
        valid_w = window[~np.isnan(window)]
        if len(valid_w) > 10:
            pred_30m_pctile[i] = np.searchsorted(np.sort(valid_w), pred_30m_abs[i]) / len(valid_w)

    ret_4bar = bars_30m["ret_lb_4bar"].values if "ret_lb_4bar" in bars_30m.columns else np.zeros(n)
    vol_rel = bars_30m["vol_rel_4bar"].values if "vol_rel_4bar" in bars_30m.columns else np.ones(n)
    rvol_4bar = bars_30m["rvol_4bar"].values if "rvol_4bar" in bars_30m.columns else np.ones(n)
    sv_ratio = bars_30m["signed_volume_ratio"].values if "signed_volume_ratio" in bars_30m.columns else np.zeros(n)

    rvol_pctile = np.zeros(n, dtype=np.float32)
    for i in range(50, n):
        window = rvol_4bar[max(0, i - 200):i]
        valid_w = window[~np.isnan(window)]
        if len(valid_w) > 10:
            rvol_pctile[i] = np.searchsorted(np.sort(valid_w), rvol_4bar[i]) / len(valid_w)

    for i in range(n):
        if np.isnan(pred_30m[i]):
            continue

        if pred_30m[i] > 0:
            candidate_dir = 1.0
        elif pred_30m[i] < 0:
            candidate_dir = -1.0
        else:
            continue

        signals = 0
        detail_bits = 0

        if pred_30m_pctile[i] >= 0.70:
            signals += 1
            detail_bits |= 1
        if not np.isnan(pred_1h[i]) and np.sign(pred_1h[i]) == candidate_dir:
            signals += 1
            detail_bits |= 2
        if ofi_signal[i] != 0 and np.sign(ofi_signal[i]) == candidate_dir:
            signals += 1
            detail_bits |= 4
        if not np.isnan(ret_4bar[i]) and np.sign(ret_4bar[i]) == candidate_dir:
            signals += 1
            detail_bits |= 8
        if vol_rel[i] > 1.0 and np.sign(sv_ratio[i]) == candidate_dir:
            signals += 1
            detail_bits |= 16
        if 0.15 <= rvol_pctile[i] <= 0.85:
            signals += 1
            detail_bits |= 32

        confluence_score[i] = signals
        confluence_direction[i] = candidate_dir
        confluence_detail[i] = detail_bits

    n_high = (confluence_score >= CONFLUENCE_MIN).sum()
    log.info(f"Confluence scoring: {n_high} bars with score >= {CONFLUENCE_MIN}")
    return confluence_score, confluence_direction, confluence_detail


# ═══════════════════════════════════════════════════════════════════
#  TRADE RECONSTRUCTION (copied from v5, extended max hold to 60)
# ═══════════════════════════════════════════════════════════════════


def reconstruct_trades(
    bars_30m: pd.DataFrame,
    minute_df: pd.DataFrame,
    pred_30m: np.ndarray,
    confluence_score: np.ndarray,
    confluence_direction: np.ndarray,
    min_confluence: int = CONFLUENCE_MIN,
    confidence_pct: float = ENTRY_CONFIDENCE_PCT,
    max_hold_minutes: int = 60,
) -> List[Dict]:
    """
    Reconstruct trades from entry signals. max_hold_minutes set to the
    largest sweep value so we have enough price path for all configs.
    """
    valid_mask = ~np.isnan(pred_30m)
    valid_preds = pred_30m[valid_mask]
    if len(valid_preds) < 20:
        log.warning("Too few valid predictions for trade reconstruction")
        return []

    upper_thresh = np.nanquantile(pred_30m[valid_mask], 1 - confidence_pct)
    lower_thresh = np.nanquantile(pred_30m[valid_mask], confidence_pct)

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

        direction = 0
        if pred_30m[i] >= upper_thresh:
            direction = 1
        elif pred_30m[i] <= lower_thresh:
            direction = -1
        else:
            continue

        if confluence_score[i] < min_confluence:
            continue

        if confluence_direction[i] != 0 and confluence_direction[i] != direction:
            continue

        date_str = bars_dates[i]
        entry_ts = pd.Timestamp(bars_ts[i])
        entry_price = bars_close[i]

        if date_str not in minute_lookup:
            continue

        day_minutes = minute_lookup[date_str]
        day_ts = day_minutes["ts_minute"].values

        entry_ts_np = np.datetime64(entry_ts)
        window_end = entry_ts + pd.Timedelta(minutes=max_hold_minutes)
        window_end_np = np.datetime64(window_end)

        mask = (day_ts >= entry_ts_np) & (day_ts <= window_end_np)
        trade_minutes = day_minutes[mask].copy()

        if len(trade_minutes) < 5:
            continue

        prices = trade_minutes["close"].values
        times = trade_minutes["ts_minute"].values
        ofi_vals = trade_minutes["ofi_1min"].values
        vol_vals = trade_minutes["volume"].values
        spread_vals = trade_minutes["spread_mean"].values
        sv_vals = trade_minutes["signed_volume"].values

        pnl_path = (prices - entry_price) / 0.25 * direction
        mfe_path = np.maximum.accumulate(pnl_path)
        mae_path = np.minimum.accumulate(pnl_path)

        trade = {
            "idx": i,
            "date": date_str,
            "entry_ts": entry_ts,
            "entry_price": entry_price,
            "direction": direction,
            "pred_30m": float(pred_30m[i]),
            "confluence_score": int(confluence_score[i]),
            "prices": prices,
            "times": times,
            "pnl_path": pnl_path,
            "mfe_path": mfe_path,
            "mae_path": mae_path,
            "ofi_vals": ofi_vals,
            "vol_vals": vol_vals,
            "spread_vals": spread_vals,
            "sv_vals": sv_vals,
            "n_minutes": len(prices),
        }

        # Precompute static hold outcomes
        for hold_min in [10, 15, 20, 25, 30, 45, 60]:
            idx_hold = min(hold_min, len(pnl_path) - 1)
            trade[f"pnl_{hold_min}min"] = float(pnl_path[idx_hold]) - COST_RT_TICKS

        trades.append(trade)

    log.info(f"Reconstructed {len(trades)} trades from entry signals")
    if trades:
        directions = [t["direction"] for t in trades]
        log.info(f"  Long: {sum(1 for d in directions if d == 1)}, "
                 f"Short: {sum(1 for d in directions if d == -1)}")
        trade_dates = sorted(set(t["date"] for t in trades))
        log.info(f"  Trading days: {len(trade_dates)} "
                 f"({trade_dates[0]} -> {trade_dates[-1]})")

    return trades


# ═══════════════════════════════════════════════════════════════════
#  TRAILING STOP SIMULATION (parameterized version)
# ═══════════════════════════════════════════════════════════════════


def simulate_trailing_config(
    trades: List[Dict],
    trigger1: float,
    buffer1: float,
    trigger2: float,
    buffer2: float,
    stale_bars: int,
    max_hold: int,
    checkpoint_interval: int = CHECKPOINT_INTERVAL,
) -> Dict:
    """
    Simulate a single trailing stop configuration across all trades.

    Returns a dict with all performance metrics.
    """
    pnls = []
    dates = []
    dirs = []
    hold_durations = []
    exit_reasons = {"trailing_t1": 0, "trailing_t2": 0, "stale_mfe": 0,
                    "static_hold": 0, "max_hold": 0}

    for trade in trades:
        pnl_path = trade["pnl_path"]
        mfe_path = trade["mfe_path"]
        n_minutes = trade["n_minutes"]
        exited = False
        exit_minute = min(STATIC_HOLD_MINUTES, n_minutes - 1)

        for ckpt in range(checkpoint_interval, n_minutes - 1, checkpoint_interval):
            mfe_now = float(mfe_path[ckpt])
            pnl_now = float(pnl_path[ckpt])

            # Compute bars_since_mfe in management-bar units
            bars_since = 0
            for lb in range(1, ckpt + 1):
                if mfe_path[ckpt - lb] < mfe_now:
                    break
                bars_since += 1
            bars_since_mgmt = bars_since / max(checkpoint_interval, 1)

            # Stale MFE exit
            if stale_bars < 999 and bars_since_mgmt >= stale_bars and mfe_now > 1.0:
                exit_minute = ckpt
                exited = True
                exit_reasons["stale_mfe"] += 1
                break

            # Trailing stop floor
            floor = None
            exit_reason_key = None
            if mfe_now >= trigger2:
                floor = mfe_now - buffer2
                exit_reason_key = "trailing_t2"
            elif mfe_now >= trigger1:
                floor = mfe_now - buffer1
                exit_reason_key = "trailing_t1"

            if floor is not None and pnl_now < floor:
                exit_minute = ckpt
                exited = True
                exit_reasons[exit_reason_key] += 1
                break

            # Max hold cap
            if ckpt >= max_hold:
                exit_minute = ckpt
                exited = True
                exit_reasons["max_hold"] += 1
                break

        if not exited:
            exit_minute = min(STATIC_HOLD_MINUTES, n_minutes - 1)
            exit_reasons["static_hold"] += 1

        exit_minute = min(exit_minute, n_minutes - 1)
        trade_pnl = float(pnl_path[exit_minute]) - COST_RT_TICKS
        pnls.append(trade_pnl)
        dates.append(trade["date"])
        dirs.append(trade["direction"])
        hold_durations.append(exit_minute)

    if not pnls:
        return {"error": "no trades"}

    pnl_arr = np.array(pnls)
    dates_arr = np.array(dates)
    dirs_arr = np.array(dirs)
    hold_arr = np.array(hold_durations, dtype=np.float64)

    # Core metrics
    cum_pnl = np.cumsum(pnl_arr)
    mean_pnl = pnl_arr.mean()
    std_pnl = max(pnl_arr.std(), 1e-6)
    sharpe = mean_pnl / std_pnl * np.sqrt(252)

    downside = np.sqrt(np.mean(np.minimum(pnl_arr, 0) ** 2))
    sortino = mean_pnl / max(downside, 1e-6) * np.sqrt(252)

    wr = float(np.mean(pnl_arr > 0))
    winners = pnl_arr[pnl_arr > 0].sum()
    losers = -pnl_arr[pnl_arr < 0].sum()
    pf = winners / max(losers, 1e-6)

    max_dd = float(np.min(cum_pnl - np.maximum.accumulate(cum_pnl)))

    # Per-day metrics
    trade_df = pd.DataFrame({"pnl": pnl_arr, "date": dates_arr})
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

    # Per-side
    long_mask = dirs_arr == 1
    short_mask = dirs_arr == -1
    long_pnl = pnl_arr[long_mask]
    short_pnl = pnl_arr[short_mask]

    long_sharpe = 0.0
    if len(long_pnl) > 2:
        long_sharpe = float(long_pnl.mean() / max(long_pnl.std(), 1e-6) * np.sqrt(252))

    short_sharpe = 0.0
    if len(short_pnl) > 2:
        short_sharpe = float(short_pnl.mean() / max(short_pnl.std(), 1e-6) * np.sqrt(252))

    return {
        "trigger1": trigger1,
        "buffer1": buffer1,
        "trigger2": trigger2,
        "buffer2": buffer2,
        "stale_bars": stale_bars,
        "max_hold": max_hold,
        "n_trades": len(pnl_arr),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "win_rate": wr,
        "profit_factor": float(pf),
        "total_pnl_ticks": float(pnl_arr.sum()),
        "total_pnl_dollars": float(pnl_arr.sum() * ES_TICK_VALUE),
        "avg_pnl_ticks": float(mean_pnl),
        "max_dd_ticks": float(max_dd),
        "max_dd_dollars": float(max_dd * ES_TICK_VALUE),
        "daily_sharpe": daily_sharpe,
        "day_concentration": day_conc,
        "day_conc_pass": day_conc <= 0.70,
        "avg_hold_min": float(hold_arr.mean()),
        "median_hold_min": float(np.median(hold_arr)),
        "n_trading_days": int(len(day_pnl)),
        "long_trades": int(long_mask.sum()),
        "short_trades": int(short_mask.sum()),
        "long_sharpe": long_sharpe,
        "short_sharpe": short_sharpe,
        "long_wr": float(np.mean(long_pnl > 0)) if len(long_pnl) > 0 else 0,
        "short_wr": float(np.mean(short_pnl > 0)) if len(short_pnl) > 0 else 0,
        "exit_reasons": exit_reasons,
        # Ranking metric: min(Sharpe, Sortino/2) for robustness
        "rank_metric": float(min(sharpe, sortino / 2)),
    }


# ═══════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════


def run_trailing_stop_sweep():
    """
    Main pipeline:
      1. Load data + build features (same as v5)
      2. Walk-forward entry model (same as v5)
      3. Reconstruct trades (same as v5)
      4. Sweep all trailing stop configs
      5. Analyze and report results
    """
    _import_lightgbm()

    # ── MLflow setup ──
    mlflow_active = False
    mlflow_run = None
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("trailing_stop_sweep_v1")
        mlflow_run = mlflow.start_run(
            run_name=f"ts_sweep_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        mlflow_active = True
        log.info(f"MLflow run started: {mlflow_run.info.run_id}")
    except Exception as e:
        log.warning(f"MLflow unavailable: {e} -- proceeding without tracking")

    t0 = time.time()

    # ════════════════════════════════════════
    #  STEP 1: Load data + build features
    # ════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 1: Loading raw minute bar data")
    log.info("=" * 70)

    minute_df = load_all_minute_bars()
    all_dates_raw = sorted(minute_df["date"].unique())
    log.info(f"Total trading days: {len(all_dates_raw)}")
    log.info(f"Date range: {all_dates_raw[0]} -> {all_dates_raw[-1]}")

    # ════════════════════════════════════════
    #  STEP 2: Build bar features
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 2: Building bar features for entry models")
    log.info("=" * 70)

    bars_30m = aggregate_to_bars(minute_df, bar_size_min=30)
    bars_30m = add_rolling_features(bars_30m, bar_size_min=30)
    bars_30m = add_forward_labels(bars_30m, horizon_bars=1, horizon_label="30min",
                                   min_edge_ticks=2.5)

    bars_15m = aggregate_to_bars(minute_df, bar_size_min=15)
    bars_15m = add_rolling_features(bars_15m, bar_size_min=15)
    bars_15m = add_forward_labels(bars_15m, horizon_bars=4, horizon_label="1h",
                                   min_edge_ticks=3.0)

    ofi_signal_df = compute_ofi_exhaustion_signal(minute_df)

    feature_cols_30m = get_feature_columns(bars_30m)
    feature_cols_15m = get_feature_columns(bars_15m)
    log.info(f"30-min features: {len(feature_cols_30m)}")
    log.info(f"15-min features: {len(feature_cols_15m)}")

    # ════════════════════════════════════════
    #  STEP 3: Walk-forward entry models (40d train, 5d slide)
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 3: Walk-forward Level-1 entry model training")
    log.info("=" * 70)

    dates_30m = sorted(bars_30m["date"].unique())
    dates_15m = sorted(bars_15m["date"].unique())

    features_30m_all = bars_30m[feature_cols_30m].values.astype(np.float32)
    labels_30m_all = bars_30m["fwd_ticks_30min"].values.astype(np.float32)
    dates_30m_all = bars_30m["date"].values

    features_15m_all = bars_15m[feature_cols_15m].values.astype(np.float32)
    labels_1h_all = bars_15m["fwd_ticks_1h"].values.astype(np.float32)
    dates_15m_all = bars_15m["date"].values

    # Per-day returns for regime classification
    day_close = bars_30m.groupby("date")["close"].last()
    day_returns_raw = day_close.pct_change()
    day_returns = day_returns_raw.to_dict()

    l1_preds_30m = np.full(len(bars_30m), np.nan, dtype=np.float32)
    l1_preds_1h_on_15m = np.full(len(bars_15m), np.nan, dtype=np.float32)

    fold_idx = 0
    val_days_l1 = 5

    log.info(f"L1 Walk-Forward: {TRAIN_DAYS}d train, {val_days_l1}d val, {SLIDE_DAYS}d slide")

    start_idx = TRAIN_DAYS
    for fold_start in range(start_idx, len(dates_30m) - val_days_l1 + 1, SLIDE_DAYS):
        fold_train_dates = dates_30m[fold_start - TRAIN_DAYS: fold_start]
        fold_val_dates = dates_30m[fold_start: fold_start + val_days_l1]

        if len(fold_val_dates) < val_days_l1:
            break

        fold_idx += 1

        # 30-min model
        train_mask_30m = np.isin(dates_30m_all, fold_train_dates)
        val_mask_30m = np.isin(dates_30m_all, fold_val_dates)

        if not leakage_audit(list(fold_train_dates), list(fold_val_dates), feature_cols_30m):
            continue

        tr_30m = features_30m_all[train_mask_30m].copy()
        vl_30m = features_30m_all[val_mask_30m].copy()
        med = np.nanmedian(tr_30m, axis=0)
        q75 = np.nanpercentile(tr_30m, 75, axis=0)
        q25 = np.nanpercentile(tr_30m, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0
        tr_30m = np.clip(np.nan_to_num((tr_30m - med) / iqr, nan=0.0), -5, 5)
        vl_30m = np.clip(np.nan_to_num((vl_30m - med) / iqr, nan=0.0), -5, 5)

        _, _, val_p_30m = train_entry_model(
            tr_30m, labels_30m_all[train_mask_30m],
            vl_30m, labels_30m_all[val_mask_30m],
            feature_names=feature_cols_30m,
            fold_idx=fold_idx, horizon_label="30min",
        )
        l1_preds_30m[val_mask_30m] = val_p_30m

        # 1h model (15-min bars)
        train_mask_15m = np.isin(dates_15m_all, fold_train_dates)
        val_mask_15m = np.isin(dates_15m_all, fold_val_dates)

        if not leakage_audit(list(fold_train_dates), list(fold_val_dates), feature_cols_15m):
            continue

        tr_15m = features_15m_all[train_mask_15m].copy()
        vl_15m = features_15m_all[val_mask_15m].copy()
        med15 = np.nanmedian(tr_15m, axis=0)
        q75_15 = np.nanpercentile(tr_15m, 75, axis=0)
        q25_15 = np.nanpercentile(tr_15m, 25, axis=0)
        iqr15 = q75_15 - q25_15
        iqr15[iqr15 < 1e-8] = 1.0
        tr_15m = np.clip(np.nan_to_num((tr_15m - med15) / iqr15, nan=0.0), -5, 5)
        vl_15m = np.clip(np.nan_to_num((vl_15m - med15) / iqr15, nan=0.0), -5, 5)

        _, _, val_p_1h = train_entry_model(
            tr_15m, labels_1h_all[train_mask_15m],
            vl_15m, labels_1h_all[val_mask_15m],
            feature_names=feature_cols_15m,
            fold_idx=fold_idx, horizon_label="1h",
        )
        l1_preds_1h_on_15m[val_mask_15m] = val_p_1h

        if fold_idx % 5 == 0:
            n_valid_30m = (~np.isnan(l1_preds_30m)).sum()
            log.info(f"  Progress: fold {fold_idx}, 30m preds: {n_valid_30m}")

    log.info(f"\nL1 complete: {fold_idx} folds")
    log.info(f"  30m predictions: {(~np.isnan(l1_preds_30m)).sum()} / {len(l1_preds_30m)}")
    log.info(f"  1h predictions: {(~np.isnan(l1_preds_1h_on_15m)).sum()} / {len(l1_preds_1h_on_15m)}")

    # ════════════════════════════════════════
    #  STEP 4: Align signals + confluence
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 4: Align signals + compute confluence")
    log.info("=" * 70)

    bars_15m_mapped = bars_15m[["date", "ts"]].copy()
    bars_15m_mapped["pred_1h"] = l1_preds_1h_on_15m
    bars_15m_mapped["bar_30m_key"] = bars_15m_mapped["ts"].dt.floor("30min")

    pred_1h_per_30m = bars_15m_mapped.groupby(
        ["date", "bar_30m_key"]
    )["pred_1h"].mean().reset_index()
    pred_1h_per_30m.columns = ["date", "bar_key", "pred_1h_aligned"]

    bars_30m_merged = bars_30m.merge(pred_1h_per_30m, on=["date", "bar_key"], how="left")
    pred_1h_aligned = bars_30m_merged["pred_1h_aligned"].values.astype(np.float32)

    ofi_ts = pd.to_datetime(ofi_signal_df["ts_minute"], utc=True)
    ofi_signal_df["bar_30m_key"] = ofi_ts.dt.floor("30min")
    ofi_per_30m = ofi_signal_df.groupby(["date", "bar_30m_key"]).agg(
        ofi_exhaust_max=("ofi_exhaust_signal",
                         lambda x: x[x.abs() == x.abs().max()].iloc[0] if len(x) > 0 else 0),
    ).reset_index()
    ofi_per_30m.columns = ["date", "bar_key", "ofi_exhaust_max"]

    bars_30m_merged = bars_30m_merged.merge(ofi_per_30m, on=["date", "bar_key"], how="left")
    ofi_aligned = bars_30m_merged["ofi_exhaust_max"].fillna(0).values.astype(np.float32)

    confluence_score, confluence_direction, confluence_detail = compute_confluence_scores(
        l1_preds_30m, pred_1h_aligned, ofi_aligned, bars_30m,
    )

    # ════════════════════════════════════════
    #  STEP 5: Reconstruct trades
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 5: Reconstruct trades (confluence >= 2, top/bottom 10%)")
    log.info("=" * 70)

    # Use max of sweep max_hold values for price path length
    max_hold_for_reconstruction = max(SWEEP_MAX_HOLD)

    all_trades = reconstruct_trades(
        bars_30m, minute_df, l1_preds_30m,
        confluence_score, confluence_direction,
        min_confluence=CONFLUENCE_MIN,
        confidence_pct=ENTRY_CONFIDENCE_PCT,
        max_hold_minutes=max_hold_for_reconstruction,
    )

    if not all_trades:
        log.error("No trades reconstructed -- cannot proceed")
        if mlflow_active:
            mlflow.end_run(status="FAILED")
        return

    log.info(f"Total trades for sweep: {len(all_trades)}")
    trade_dates = sorted(set(t["date"] for t in all_trades))
    log.info(f"Trading days: {len(trade_dates)} ({trade_dates[0]} -> {trade_dates[-1]})")

    # Free memory from data prep
    del minute_df, bars_15m, ofi_signal_df
    gc.collect()

    # ════════════════════════════════════════
    #  STEP 6: SWEEP all trailing stop configs
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 6: Sweeping trailing stop parameters")
    log.info("=" * 70)

    # Build grid
    grid = list(product(
        SWEEP_TRIGGER1, SWEEP_BUFFER1,
        SWEEP_TRIGGER2, SWEEP_BUFFER2,
        SWEEP_STALE_BARS, SWEEP_MAX_HOLD,
    ))

    # Filter invalid configs: trigger2 must be > trigger1
    valid_grid = []
    for t1, b1, t2, b2, stale, mh in grid:
        if t2 <= t1:
            continue  # tier 2 must be strictly above tier 1
        valid_grid.append((t1, b1, t2, b2, stale, mh))

    total_configs = len(valid_grid)
    log.info(f"Total valid configs to sweep: {total_configs} "
             f"(filtered from {len(grid)} where trigger2 > trigger1)")

    if mlflow_active:
        mlflow.log_param("total_configs", total_configs)
        mlflow.log_param("n_trades", len(all_trades))
        mlflow.log_param("n_trading_days", len(trade_dates))
        mlflow.log_param("cost_rt_ticks", COST_RT_TICKS)

    results = []
    t_sweep_start = time.time()

    for cfg_idx, (t1, b1, t2, b2, stale, mh) in enumerate(valid_grid):
        result = simulate_trailing_config(
            all_trades,
            trigger1=t1, buffer1=b1,
            trigger2=t2, buffer2=b2,
            stale_bars=stale, max_hold=mh,
        )

        if "error" not in result:
            results.append(result)

        if (cfg_idx + 1) % 500 == 0:
            elapsed = time.time() - t_sweep_start
            rate = (cfg_idx + 1) / elapsed
            remaining = (total_configs - cfg_idx - 1) / max(rate, 0.1)
            log.info(
                f"  Progress: {cfg_idx + 1}/{total_configs} configs "
                f"({elapsed:.0f}s elapsed, ~{remaining:.0f}s remaining, "
                f"{rate:.0f} configs/sec)"
            )

    sweep_time = time.time() - t_sweep_start
    log.info(f"\nSweep complete: {len(results)} valid configs in {sweep_time:.1f}s "
             f"({len(results)/max(sweep_time,0.1):.0f} configs/sec)")

    if not results:
        log.error("No valid sweep results")
        if mlflow_active:
            mlflow.end_run(status="FAILED")
        return

    # ════════════════════════════════════════
    #  STEP 7: Analyze results
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 7: Analyzing sweep results")
    log.info("=" * 70)

    sweep_df = pd.DataFrame(results)

    # Remove exit_reasons column for CSV (it's a dict)
    csv_cols = [c for c in sweep_df.columns if c != "exit_reasons"]
    sweep_csv = sweep_df[csv_cols].copy()

    # Sort by rank_metric (min(Sharpe, Sortino/2))
    sweep_csv = sweep_csv.sort_values("rank_metric", ascending=False)

    # ── Robustness metrics ──
    sharpe_gt4 = (sweep_df["sharpe"] > 4.0).sum()
    sharpe_gt6 = (sweep_df["sharpe"] > 6.0).sum()
    sharpe_gt8 = (sweep_df["sharpe"] > 8.0).sum()
    sharpe_positive = (sweep_df["sharpe"] > 0).sum()
    median_sharpe = sweep_df["sharpe"].median()
    mean_sharpe = sweep_df["sharpe"].mean()
    std_sharpe = sweep_df["sharpe"].std()

    log.info(f"\n  ROBUSTNESS SUMMARY:")
    log.info(f"  Total configs tested:          {len(sweep_df)}")
    log.info(f"  Sharpe > 0:                    {sharpe_positive} ({sharpe_positive/len(sweep_df)*100:.1f}%)")
    log.info(f"  Sharpe > 4.0:                  {sharpe_gt4} ({sharpe_gt4/len(sweep_df)*100:.1f}%)")
    log.info(f"  Sharpe > 6.0:                  {sharpe_gt6} ({sharpe_gt6/len(sweep_df)*100:.1f}%)")
    log.info(f"  Sharpe > 8.0:                  {sharpe_gt8} ({sharpe_gt8/len(sweep_df)*100:.1f}%)")
    log.info(f"  Median Sharpe:                 {median_sharpe:.2f}")
    log.info(f"  Mean Sharpe:                   {mean_sharpe:.2f} +/- {std_sharpe:.2f}")
    log.info(f"  Mean Win Rate:                 {sweep_df['win_rate'].mean():.1%}")
    log.info(f"  Mean Profit Factor:            {sweep_df['profit_factor'].mean():.2f}")
    log.info(f"  Day conc pass rate:            {sweep_df['day_conc_pass'].mean():.1%}")

    # ── V5 reference config performance ──
    ref_mask = (
        (sweep_df["trigger1"] == V5_REF_CONFIG["trigger1"]) &
        (sweep_df["buffer1"] == V5_REF_CONFIG["buffer1"]) &
        (sweep_df["trigger2"] == V5_REF_CONFIG["trigger2"]) &
        (sweep_df["buffer2"] == V5_REF_CONFIG["buffer2"]) &
        (sweep_df["stale_bars"] == V5_REF_CONFIG["stale_bars"]) &
        (sweep_df["max_hold"] == V5_REF_CONFIG["max_hold"])
    )
    ref_rows = sweep_df[ref_mask]
    if len(ref_rows) > 0:
        ref = ref_rows.iloc[0]
        ref_rank = sweep_csv.index.get_loc(ref.name) + 1
        log.info(f"\n  V5 REFERENCE CONFIG (trigger1=3, buffer1=2.0, trigger2=6, buffer2=1.5, stale=4, max_hold=60):")
        log.info(f"    Sharpe={ref['sharpe']:.2f}, Sortino={ref['sortino']:.2f}, "
                 f"WR={ref['win_rate']:.1%}, PF={ref['profit_factor']:.2f}, "
                 f"Rank={ref_rank}/{len(sweep_df)}")
        log.info(f"    Percentile: {(1 - ref_rank/len(sweep_df))*100:.1f}th")

    # ── Top 20 configs ──
    log.info(f"\n  TOP 20 CONFIGS (ranked by min(Sharpe, Sortino/2)):")
    log.info(f"  {'Rank':>4s} {'t1':>3s} {'b1':>4s} {'t2':>3s} {'b2':>4s} "
             f"{'stale':>5s} {'mh':>3s} {'Sharpe':>7s} {'Sortino':>8s} "
             f"{'WR':>5s} {'PF':>5s} {'N':>4s} {'AvgHold':>7s} {'$PnL':>8s}")
    log.info("  " + "-" * 95)

    top_20 = sweep_csv.head(20)
    for rank, (_, row) in enumerate(top_20.iterrows(), 1):
        log.info(
            f"  {rank:4d} {row['trigger1']:3.0f} {row['buffer1']:4.1f} "
            f"{row['trigger2']:3.0f} {row['buffer2']:4.1f} "
            f"{row['stale_bars']:5.0f} {row['max_hold']:3.0f} "
            f"{row['sharpe']:7.2f} {row['sortino']:8.2f} "
            f"{row['win_rate']:5.1%} {row['profit_factor']:5.2f} "
            f"{row['n_trades']:4.0f} {row['avg_hold_min']:7.1f} "
            f"{row['total_pnl_dollars']:8.0f}"
        )

    # ── Bottom 20 configs ──
    log.info(f"\n  BOTTOM 20 CONFIGS:")
    log.info(f"  {'Rank':>4s} {'t1':>3s} {'b1':>4s} {'t2':>3s} {'b2':>4s} "
             f"{'stale':>5s} {'mh':>3s} {'Sharpe':>7s} {'Sortino':>8s} "
             f"{'WR':>5s} {'PF':>5s} {'N':>4s} {'AvgHold':>7s} {'$PnL':>8s}")
    log.info("  " + "-" * 95)

    bottom_20 = sweep_csv.tail(20).iloc[::-1]
    for rank_from_bottom, (_, row) in enumerate(bottom_20.iterrows(), 1):
        rank = len(sweep_csv) - rank_from_bottom + 1
        log.info(
            f"  {rank:4d} {row['trigger1']:3.0f} {row['buffer1']:4.1f} "
            f"{row['trigger2']:3.0f} {row['buffer2']:4.1f} "
            f"{row['stale_bars']:5.0f} {row['max_hold']:3.0f} "
            f"{row['sharpe']:7.2f} {row['sortino']:8.2f} "
            f"{row['win_rate']:5.1%} {row['profit_factor']:5.2f} "
            f"{row['n_trades']:4.0f} {row['avg_hold_min']:7.1f} "
            f"{row['total_pnl_dollars']:8.0f}"
        )

    # ── Stability heatmaps: marginal Sharpe over (trigger1, buffer1) ──
    log.info(f"\n  STABILITY HEATMAP: Mean Sharpe by (trigger1, buffer1) — marginalizing over other params")

    heatmap_tb1 = sweep_df.groupby(["trigger1", "buffer1"])["sharpe"].agg(
        ["mean", "std", "count"]
    ).reset_index()
    heatmap_tb1.columns = ["trigger1", "buffer1", "mean_sharpe", "std_sharpe", "count"]
    heatmap_tb1 = heatmap_tb1.sort_values(["trigger1", "buffer1"])

    header = f"  {'':>8s}"
    for b1 in SWEEP_BUFFER1:
        header += f" b1={b1:<5.1f}"
    log.info(header)

    for t1 in SWEEP_TRIGGER1:
        row_str = f"  t1={t1:>3.0f}"
        for b1 in SWEEP_BUFFER1:
            mask = (heatmap_tb1["trigger1"] == t1) & (heatmap_tb1["buffer1"] == b1)
            rows = heatmap_tb1[mask]
            if len(rows) > 0:
                ms = rows.iloc[0]["mean_sharpe"]
                row_str += f" {ms:>7.2f}"
            else:
                row_str += f" {'N/A':>7s}"
        log.info(row_str)

    # ── Stability heatmap: Mean Sharpe by (trigger2, buffer2) ──
    log.info(f"\n  STABILITY HEATMAP: Mean Sharpe by (trigger2, buffer2)")

    heatmap_tb2 = sweep_df.groupby(["trigger2", "buffer2"])["sharpe"].agg(
        ["mean", "std"]
    ).reset_index()
    heatmap_tb2.columns = ["trigger2", "buffer2", "mean_sharpe", "std_sharpe"]

    header2 = f"  {'':>8s}"
    for b2 in SWEEP_BUFFER2:
        header2 += f" b2={b2:<5.1f}"
    log.info(header2)

    for t2 in SWEEP_TRIGGER2:
        row_str = f"  t2={t2:>3.0f}"
        for b2 in SWEEP_BUFFER2:
            mask = (heatmap_tb2["trigger2"] == t2) & (heatmap_tb2["buffer2"] == b2)
            rows = heatmap_tb2[mask]
            if len(rows) > 0:
                ms = rows.iloc[0]["mean_sharpe"]
                row_str += f" {ms:>7.2f}"
            else:
                row_str += f" {'N/A':>7s}"
        log.info(row_str)

    # ── Stability: Sharpe by stale_bars ──
    log.info(f"\n  Mean Sharpe by stale_bars:")
    stale_agg = sweep_df.groupby("stale_bars")["sharpe"].agg(["mean", "std", "count"])
    for stale_val, row in stale_agg.iterrows():
        label = "disabled" if stale_val >= 999 else f"{int(stale_val)} bars"
        log.info(f"    {label:>12s}: Sharpe={row['mean']:.2f} +/- {row['std']:.2f} (n={int(row['count'])})")

    # ── Stability: Sharpe by max_hold ──
    log.info(f"\n  Mean Sharpe by max_hold:")
    mh_agg = sweep_df.groupby("max_hold")["sharpe"].agg(["mean", "std", "count"])
    for mh_val, row in mh_agg.iterrows():
        log.info(f"    {int(mh_val):>4d} min: Sharpe={row['mean']:.2f} +/- {row['std']:.2f} (n={int(row['count'])})")

    # ── Regime analysis on top config ──
    log.info(f"\n  REGIME ANALYSIS on top-1 config:")
    top1 = sweep_csv.iloc[0]
    top1_result = simulate_trailing_config(
        all_trades,
        trigger1=top1["trigger1"], buffer1=top1["buffer1"],
        trigger2=top1["trigger2"], buffer2=top1["buffer2"],
        stale_bars=int(top1["stale_bars"]), max_hold=int(top1["max_hold"]),
    )

    # Re-simulate to get per-trade arrays for regime analysis
    pnls_regime = []
    dates_regime = []
    for trade in all_trades:
        pnl_path = trade["pnl_path"]
        mfe_path = trade["mfe_path"]
        n_minutes = trade["n_minutes"]
        exited = False
        exit_minute = min(STATIC_HOLD_MINUTES, n_minutes - 1)

        for ckpt in range(CHECKPOINT_INTERVAL, n_minutes - 1, CHECKPOINT_INTERVAL):
            mfe_now = float(mfe_path[ckpt])
            pnl_now = float(pnl_path[ckpt])
            bars_since = 0
            for lb in range(1, ckpt + 1):
                if mfe_path[ckpt - lb] < mfe_now:
                    break
                bars_since += 1
            bars_since_mgmt = bars_since / max(CHECKPOINT_INTERVAL, 1)

            stale_val = int(top1["stale_bars"])
            if stale_val < 999 and bars_since_mgmt >= stale_val and mfe_now > 1.0:
                exit_minute = ckpt
                exited = True
                break

            floor = None
            if mfe_now >= top1["trigger2"]:
                floor = mfe_now - top1["buffer2"]
            elif mfe_now >= top1["trigger1"]:
                floor = mfe_now - top1["buffer1"]
            if floor is not None and pnl_now < floor:
                exit_minute = ckpt
                exited = True
                break

            if ckpt >= int(top1["max_hold"]):
                exit_minute = ckpt
                exited = True
                break

        if not exited:
            exit_minute = min(STATIC_HOLD_MINUTES, n_minutes - 1)

        exit_minute = min(exit_minute, n_minutes - 1)
        trade_pnl = float(pnl_path[exit_minute]) - COST_RT_TICKS
        pnls_regime.append(trade_pnl)
        dates_regime.append(trade["date"])

    pnl_arr_regime = np.array(pnls_regime)
    dates_arr_regime = np.array(dates_regime)

    # Classify days
    for regime_name, check_fn in [
        ("green", lambda r: r > 0.001),
        ("red", lambda r: r < -0.001),
        ("flat", lambda r: -0.001 <= r <= 0.001),
    ]:
        mask = np.array([
            check_fn(day_returns.get(d, 0)) for d in dates_arr_regime
        ])
        if mask.sum() > 5:
            rpnl = pnl_arr_regime[mask]
            rsharpe = float(rpnl.mean() / max(rpnl.std(), 1e-6) * np.sqrt(252))
            rwr = float(np.mean(rpnl > 0))
            log.info(f"    {regime_name}: N={mask.sum()}, Sharpe={rsharpe:.2f}, WR={rwr:.1%}")

    # Check regime gap (HC #428)
    green_mask = np.array([day_returns.get(d, 0) > 0.001 for d in dates_arr_regime])
    red_mask = np.array([day_returns.get(d, 0) < -0.001 for d in dates_arr_regime])
    if green_mask.sum() > 5 and red_mask.sum() > 5:
        s_green = float(pnl_arr_regime[green_mask].mean() / max(pnl_arr_regime[green_mask].std(), 1e-6) * np.sqrt(252))
        s_red = float(pnl_arr_regime[red_mask].mean() / max(pnl_arr_regime[red_mask].std(), 1e-6) * np.sqrt(252))
        denom = max(abs(s_green), abs(s_red), 1e-6)
        gap = abs(s_green - s_red) / denom
        log.info(f"    Regime gap: {gap:.2f} ({'PASS' if gap <= 0.50 else 'FAIL'} — threshold 0.50)")

    # ════════════════════════════════════════
    #  STEP 8: Save results
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 8: Saving results")
    log.info("=" * 70)

    # Full sweep CSV
    csv_path = OUTPUT_DIR / "sweep_results.csv"
    sweep_csv.to_csv(str(csv_path), index=False)
    log.info(f"  Saved full sweep: {csv_path}")

    # Top 20 + bottom 20 as JSON
    top_configs = sweep_csv.head(20).to_dict("records")
    bottom_configs = sweep_csv.tail(20).to_dict("records")

    summary = {
        "sweep_timestamp": datetime.now().isoformat(),
        "total_configs": total_configs,
        "valid_results": len(results),
        "n_trades": len(all_trades),
        "n_trading_days": len(trade_dates),
        "date_range": f"{trade_dates[0]} -> {trade_dates[-1]}",
        "cost_rt_ticks": COST_RT_TICKS,
        "robustness": {
            "sharpe_gt_0_pct": float(sharpe_positive / len(sweep_df) * 100),
            "sharpe_gt_4_pct": float(sharpe_gt4 / len(sweep_df) * 100),
            "sharpe_gt_6_pct": float(sharpe_gt6 / len(sweep_df) * 100),
            "sharpe_gt_8_pct": float(sharpe_gt8 / len(sweep_df) * 100),
            "median_sharpe": float(median_sharpe),
            "mean_sharpe": float(mean_sharpe),
            "std_sharpe": float(std_sharpe),
        },
        "v5_reference_config": V5_REF_CONFIG,
        "v5_reference_rank": int(ref_rank) if len(ref_rows) > 0 else None,
        "top_20_configs": top_configs,
        "bottom_20_configs": bottom_configs,
    }

    summary_path = OUTPUT_DIR / "sweep_summary.json"
    with open(str(summary_path), "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"  Saved summary: {summary_path}")

    # MLflow logging
    if mlflow_active:
        mlflow.log_metric("sharpe_gt0_pct", float(sharpe_positive / len(sweep_df) * 100))
        mlflow.log_metric("sharpe_gt4_pct", float(sharpe_gt4 / len(sweep_df) * 100))
        mlflow.log_metric("sharpe_gt6_pct", float(sharpe_gt6 / len(sweep_df) * 100))
        mlflow.log_metric("sharpe_gt8_pct", float(sharpe_gt8 / len(sweep_df) * 100))
        mlflow.log_metric("median_sharpe", float(median_sharpe))
        mlflow.log_metric("mean_sharpe", float(mean_sharpe))
        mlflow.log_metric("best_sharpe", float(sweep_df["sharpe"].max()))
        mlflow.log_metric("worst_sharpe", float(sweep_df["sharpe"].min()))

        if len(ref_rows) > 0:
            mlflow.log_metric("v5_ref_sharpe", float(ref["sharpe"]))
            mlflow.log_metric("v5_ref_rank", int(ref_rank))
            mlflow.log_metric("v5_ref_percentile", float((1 - ref_rank / len(sweep_df)) * 100))

        # Top 1 config
        mlflow.log_metric("top1_sharpe", float(top1["sharpe"]))
        mlflow.log_metric("top1_sortino", float(top1["sortino"]))
        mlflow.log_metric("top1_wr", float(top1["win_rate"]))
        mlflow.log_metric("top1_pf", float(top1["profit_factor"]))
        mlflow.log_param("top1_trigger1", float(top1["trigger1"]))
        mlflow.log_param("top1_buffer1", float(top1["buffer1"]))
        mlflow.log_param("top1_trigger2", float(top1["trigger2"]))
        mlflow.log_param("top1_buffer2", float(top1["buffer2"]))
        mlflow.log_param("top1_stale_bars", int(top1["stale_bars"]))
        mlflow.log_param("top1_max_hold", int(top1["max_hold"]))

        try:
            mlflow.log_artifact(str(csv_path))
            mlflow.log_artifact(str(summary_path))
        except Exception:
            pass

        mlflow.log_metric("total_runtime_min", (time.time() - t0) / 60)
        mlflow.end_run()
        log.info(f"MLflow run ended: {mlflow_run.info.run_id}")

    # ── Executive Summary ──
    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"TRAILING STOP SWEEP v1 COMPLETE in {elapsed/60:.1f} minutes")
    log.info(f"{'=' * 70}")
    log.info(f"  Configs tested:           {len(results)}")
    log.info(f"  Trades per config:        {len(all_trades)}")
    log.info(f"  Trading days:             {len(trade_dates)}")
    log.info(f"  Sharpe > 4.0:             {sharpe_gt4}/{len(results)} ({sharpe_gt4/len(results)*100:.1f}%)")
    log.info(f"  Median Sharpe:            {median_sharpe:.2f}")
    log.info(f"  Best config Sharpe:       {sweep_df['sharpe'].max():.2f}")
    log.info(f"  Worst config Sharpe:      {sweep_df['sharpe'].min():.2f}")
    if len(ref_rows) > 0:
        log.info(f"  V5 reference Sharpe:      {ref['sharpe']:.2f} (rank {ref_rank}/{len(results)})")
    log.info(f"\n  VERDICT: {'ROBUST' if sharpe_gt4/len(results) > 0.25 else 'FRAGILE'} "
             f"— {sharpe_gt4/len(results)*100:.0f}% of configs have Sharpe > 4.0 "
             f"(threshold: >25% for robust)")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("Trailing Stop Parameter Sweep v1")
    log.info(f"  ROOT: {ROOT}")
    log.info(f"  OUTPUT: {OUTPUT_DIR}")
    log.info(f"  Cost: {COST_RT_TICKS} ticks RT")
    log.info(f"  Walk-forward: {TRAIN_DAYS}d train, {SLIDE_DAYS}d slide")
    log.info(f"  Entry: confluence >= {CONFLUENCE_MIN}, top/bottom {ENTRY_CONFIDENCE_PCT:.0%}")
    log.info(f"  Sweep grid: {len(SWEEP_TRIGGER1)}x{len(SWEEP_BUFFER1)}x"
             f"{len(SWEEP_TRIGGER2)}x{len(SWEEP_BUFFER2)}x"
             f"{len(SWEEP_STALE_BARS)}x{len(SWEEP_MAX_HOLD)} = "
             f"{len(SWEEP_TRIGGER1)*len(SWEEP_BUFFER1)*len(SWEEP_TRIGGER2)*len(SWEEP_BUFFER2)*len(SWEEP_STALE_BARS)*len(SWEEP_MAX_HOLD)} "
             f"raw configs")
    log.info(f"  V5 reference: trigger1={V5_REF_CONFIG['trigger1']}, "
             f"buffer1={V5_REF_CONFIG['buffer1']}, "
             f"trigger2={V5_REF_CONFIG['trigger2']}, "
             f"buffer2={V5_REF_CONFIG['buffer2']}, "
             f"stale={V5_REF_CONFIG['stale_bars']}, "
             f"max_hold={V5_REF_CONFIG['max_hold']}")
    log.info(f"  Ranking: min(Sharpe, Sortino/2) for robustness")

    try:
        run_trailing_stop_sweep()
    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        log.error(traceback.format_exc())
        try:
            import mlflow
            mlflow.end_run(status="FAILED")
        except Exception:
            pass
        sys.exit(1)
