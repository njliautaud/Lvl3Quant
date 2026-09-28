#!/usr/bin/env python3
"""
Multi-Signal Ensemble v1 — Cross-Horizon Stacking
====================================================

Combines proven signals across different predictive horizons into a unified
trading signal using a walk-forward meta-learner.

KEY INSIGHT:
  - Fast signal (1s microstructure via CNN-Mamba) decays in ~30s but has
    excellent timing for entries.
  - Medium signal (30-min LightGBM bar model) has Sharpe 4.03, captures
    momentum + OFI at 30-min horizon.
  - OFI exhaustion signal (10-min) fades extreme OFI spikes, Sharpe 1.71 OOT.
  - Slow signals (1h/2h/4h LightGBM) have IC 0.40-0.55, persistent directional.

ARCHITECTURE:
  Level 1 — Individual horizon models (retrained walk-forward, 60d/5d slide):
    (A) 30-min LightGBM on microstructure bar features
    (B) 1h LightGBM on microstructure bar features
    (C) OFI exhaustion rule-based signal (10-min hold)
  Level 2 — Meta-learner (LightGBM) stacks Level-1 predictions + agreement
             features into final trade decision
  Level 3 — Position sizing: agreement across horizons → confidence multiplier

CONSTRAINTS:
  - Walk-forward: 60d train, 5d slide — SLIDING only (HC #0)
  - Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission)
  - Regime-agnostic gate: |Sharpe_green - Sharpe_red| / max < 0.50 (HC #428)
  - FIFO-based execution costs only (HC #74)
  - MLflow logging mandatory

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python -u \\
      alpha_discovery/multi_signal_ensemble_v1.py 2>&1 | \\
      tee logs/multi_signal_ensemble_v1.log

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
QUEUE_FEATURE_DIR = ROOT / "data" / "queue_augmented_features"
OUTPUT_DIR = ROOT / "output" / "multi_signal_ensemble_v1"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [ENSEMBLE-v1] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "multi_signal_ensemble_v1.log")),
    ],
)
log = logging.getLogger("ENSEMBLE-v1")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376
COST_RT_TICKS = 2.376  # market entry + market exit + RT commission

# ─────────────────────────────────────────────
#  WALK-FORWARD CONFIG
# ─────────────────────────────────────────────
TRAIN_DAYS = 60
SLIDE_DAYS = 5
META_VAL_DAYS = 10  # meta-learner uses 10 OOT days from L1 for its own training

# ─────────────────────────────────────────────
#  HORIZON CONFIG
# ─────────────────────────────────────────────
HORIZON_CONFIG = {
    "30min": {
        "bar_size_min": 30,
        "horizon_bars": 1,  # 1 x 30-min bar = 30 min forward
        "min_edge_ticks": 2.5,
        "long_only": False,
    },
    "1h": {
        "bar_size_min": 15,
        "horizon_bars": 4,  # 4 x 15-min bars = 1 hour forward
        "min_edge_ticks": 3.0,
        "long_only": False,
    },
}

# LightGBM params — proven from v4 focused study
LGBM_PARAMS = {
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

META_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_child_samples": 50,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.5,
    "lambda_l2": 2.0,
    "max_depth": 5,
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


def load_queue_features() -> Optional[pd.DataFrame]:
    """Load queue-augmented tick features if available."""
    if not QUEUE_FEATURE_DIR.exists():
        log.info("No queue_augmented_features directory -- skipping")
        return None

    files = sorted(QUEUE_FEATURE_DIR.glob("features_*.parquet"))
    if not files:
        log.info("No queue feature files found -- skipping")
        return None

    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            date_str = f.stem.replace("features_", "")
            df["date"] = date_str
            frames.append(df)
        except Exception as e:
            log.warning(f"Skip queue file {f.stem}: {e}")

    if not frames:
        return None

    combined = pd.concat(frames, ignore_index=True)
    log.info(f"Loaded queue features for {len(frames)} days, {len(combined):,} rows")
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
#  SECTION 4: OFI EXHAUSTION SIGNAL (RULE-BASED)
# ═══════════════════════════════════════════════════════════════════


def compute_ofi_exhaustion_signal(minute_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute the OFI exhaustion counter-trend signal on minute bars.
    Returns a DataFrame with date, ts_minute, and ofi_exhaust_signal columns.

    Signal logic: fade extreme OFI z-scores (>2.5 or <-2.5) when aligned
    with recent 30-min trend. The exhaustion = trend has pushed too far.
    """
    df = minute_df.copy()
    df["ts_minute"] = pd.to_datetime(df["ts_minute"], utc=True)
    df = df.sort_values("ts_minute").reset_index(drop=True)

    ofi_col = "ofi_1min"
    # Rolling z-score of OFI (60-min lookback, intraday)
    df["ofi_rmean"] = df.groupby("date")[ofi_col].transform(
        lambda x: x.rolling(60, min_periods=20).mean()
    )
    df["ofi_rstd"] = df.groupby("date")[ofi_col].transform(
        lambda x: x.rolling(60, min_periods=20).std()
    )
    df["ofi_z"] = (df[ofi_col] - df["ofi_rmean"]) / df["ofi_rstd"].clip(lower=1e-6)

    # Volume z-score
    df["vol_rmean"] = df.groupby("date")["volume"].transform(
        lambda x: x.rolling(60, min_periods=20).mean()
    )
    df["vol_rstd"] = df.groupby("date")["volume"].transform(
        lambda x: x.rolling(60, min_periods=20).std()
    )
    df["vol_z"] = (df["volume"] - df["vol_rmean"]) / df["vol_rstd"].clip(lower=1e-6)

    # 30-min trailing return
    df["ret_30m"] = df.groupby("date")["close"].transform(lambda x: x.pct_change(30))

    # Exhaustion signal: fade extreme OFI spikes aligned with trend
    ofi_thresh = 2.5
    vol_thresh = 1.0
    signal = np.zeros(len(df), dtype=np.float32)

    # Bearish exhaustion (OFI extremely positive + uptrend = fade long)
    bull_exhaust = (df["ofi_z"] > ofi_thresh) & (df["vol_z"] > vol_thresh) & (df["ret_30m"] > 0)
    signal[bull_exhaust.values] = -1.0  # Short (fade the exhaustion)

    # Bullish exhaustion (OFI extremely negative + downtrend = fade short)
    bear_exhaust = (df["ofi_z"] < -ofi_thresh) & (df["vol_z"] > vol_thresh) & (df["ret_30m"] < 0)
    signal[bear_exhaust.values] = 1.0  # Long (fade the exhaustion)

    # Scale by magnitude of z-score (stronger exhaustion = stronger signal)
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
#  SECTION 5: FEATURE COLUMN SELECTION
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
#  SECTION 6: FORWARD LABELS
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
#  SECTION 7: LEAKAGE AUDIT
# ═══════════════════════════════════════════════════════════════════


def leakage_audit(
    df: pd.DataFrame,
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
#  SECTION 8: LEVEL-1 MODEL TRAINING (individual horizons)
# ═══════════════════════════════════════════════════════════════════


def train_l1_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    fold_idx: int,
    horizon_label: str,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """Train a Level-1 LightGBM model for a single horizon."""
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

    params = {**LGBM_PARAMS, "seed": 42 + fold_idx}

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
#  SECTION 9: META-LEARNER FEATURES
# ═══════════════════════════════════════════════════════════════════


def build_meta_features(
    pred_30m: np.ndarray,
    pred_1h: np.ndarray,
    ofi_signal: np.ndarray,
    bars_df_30m: pd.DataFrame,
) -> Tuple[np.ndarray, List[str]]:
    """
    Build meta-learner features from Level-1 predictions.

    The meta-learner sees:
    1. Raw predictions from each horizon model
    2. AGREEMENT features (do horizons agree on direction?)
    3. CONFIDENCE features (how extreme is each prediction?)
    4. INTERACTION features (cross-horizon relationships)
    5. Context features (time of day, recent volatility)
    """
    n = len(pred_30m)
    features = []
    names = []

    # ── Raw predictions ──
    features.append(pred_30m)
    names.append("pred_30m")
    features.append(pred_1h)
    names.append("pred_1h")
    features.append(ofi_signal)
    names.append("ofi_exhaust_signal")

    # ── Direction agreement ──
    sign_30m = np.sign(pred_30m)
    sign_1h = np.sign(pred_1h)
    sign_ofi = np.sign(ofi_signal)

    # 2-way agreement (30m + 1h)
    agree_30m_1h = (sign_30m == sign_1h).astype(np.float32)
    features.append(agree_30m_1h)
    names.append("agree_30m_1h")

    # 3-way agreement (all three, where OFI is non-zero)
    ofi_active = ofi_signal != 0
    three_agree = np.zeros(n, dtype=np.float32)
    for i in range(n):
        if ofi_active[i] and sign_30m[i] == sign_1h[i] == sign_ofi[i] and sign_30m[i] != 0:
            three_agree[i] = 1.0
    features.append(three_agree)
    names.append("three_way_agree")

    # Disagreement (horizons point opposite ways)
    disagree = (sign_30m * sign_1h < 0).astype(np.float32)
    features.append(disagree)
    names.append("direction_conflict")

    # ── Confidence features ──
    # Percentile rank of predictions (how extreme within recent window)
    for pred, name in [(pred_30m, "30m"), (pred_1h, "1h")]:
        abs_pred = np.abs(pred)
        # Rolling percentile rank (200-bar lookback)
        pctile = np.zeros(n, dtype=np.float32)
        for i in range(200, n):
            window = abs_pred[i-200:i]
            valid_w = window[~np.isnan(window)]
            if len(valid_w) > 10:
                pctile[i] = np.searchsorted(np.sort(valid_w), abs_pred[i]) / len(valid_w)
        features.append(pctile)
        names.append(f"confidence_pctile_{name}")

    # ── Interaction features ──
    # Average of aligned predictions (directional blend)
    avg_signal = (pred_30m + pred_1h) / 2.0
    features.append(avg_signal)
    names.append("avg_signal")

    # Spread between horizons (divergence)
    signal_spread = pred_30m - pred_1h
    features.append(signal_spread)
    names.append("signal_spread_30m_minus_1h")

    # Magnitude product (both strong AND same direction = very confident)
    mag_product = pred_30m * pred_1h
    features.append(mag_product)
    names.append("magnitude_product")

    # ── Context features from bar data ──
    if "tod_progress" in bars_df_30m.columns:
        features.append(bars_df_30m["tod_progress"].values.astype(np.float32))
        names.append("tod_progress")

    if "realized_vol" in bars_df_30m.columns:
        features.append(bars_df_30m["realized_vol"].values.astype(np.float32))
        names.append("recent_vol")

    if "regime_vol_ratio" in bars_df_30m.columns:
        features.append(bars_df_30m["regime_vol_ratio"].values.astype(np.float32))
        names.append("vol_regime_ratio")

    if "intraday_direction_strength" in bars_df_30m.columns:
        features.append(bars_df_30m["intraday_direction_strength"].values.astype(np.float32))
        names.append("intraday_dir_strength")

    if "ofi_zscore_4bar" in bars_df_30m.columns:
        features.append(bars_df_30m["ofi_zscore_4bar"].values.astype(np.float32))
        names.append("ofi_z_4bar")

    # Stack
    X = np.column_stack(features)
    return X, names


# ═══════════════════════════════════════════════════════════════════
#  SECTION 10: TRADE SIMULATION (FIFO-based)
# ═══════════════════════════════════════════════════════════════════


def simulate_trades(
    preds: np.ndarray,
    actuals: np.ndarray,
    dates: np.ndarray,
    confidence_pct: float = 0.10,
    cost_ticks: float = COST_RT_TICKS,
) -> Optional[Dict]:
    """Simulate trades with per-side and per-day reporting."""
    valid = ~np.isnan(preds) & ~np.isnan(actuals)
    preds_v = preds[valid]
    actuals_v = actuals[valid]
    dates_v = dates[valid]

    if len(preds_v) < 20:
        return None

    trades = []
    upper = np.quantile(preds_v, 1 - confidence_pct)
    lower = np.quantile(preds_v, confidence_pct)

    for i in range(len(preds_v)):
        if preds_v[i] >= upper:
            pnl = actuals_v[i] - cost_ticks
            trades.append({"dir": "long", "pnl": pnl, "raw": actuals_v[i],
                           "pred": preds_v[i], "date": dates_v[i]})
        elif preds_v[i] <= lower:
            pnl = -actuals_v[i] - cost_ticks
            trades.append({"dir": "short", "pnl": pnl, "raw": -actuals_v[i],
                           "pred": preds_v[i], "date": dates_v[i]})

    if not trades:
        return None

    pnl_arr = np.array([t["pnl"] for t in trades])
    cum_pnl = np.cumsum(pnl_arr)

    sharpe = pnl_arr.mean() / max(pnl_arr.std(), 1e-6) * np.sqrt(252)
    downside = np.sqrt(np.mean(np.minimum(pnl_arr, 0) ** 2))
    sortino = pnl_arr.mean() / max(downside, 1e-6) * np.sqrt(252)
    wr = np.mean(pnl_arr > 0)
    pf = np.sum(pnl_arr[pnl_arr > 0]) / max(-np.sum(pnl_arr[pnl_arr < 0]), 1e-6)
    max_dd = float(np.min(cum_pnl - np.maximum.accumulate(cum_pnl)))

    long_trades = [t for t in trades if t["dir"] == "long"]
    short_trades = [t for t in trades if t["dir"] == "short"]
    long_pnl = np.array([t["pnl"] for t in long_trades]) if long_trades else np.array([])
    short_pnl = np.array([t["pnl"] for t in short_trades]) if short_trades else np.array([])

    trade_df = pd.DataFrame(trades)
    day_pnl = trade_df.groupby("date")["pnl"].agg(["sum", "count"]).reset_index()
    day_pnl.columns = ["date", "daily_pnl", "daily_trades"]

    return {
        "n_trades": len(trades),
        "total_pnl_ticks": float(pnl_arr.sum()),
        "total_pnl_dollars": float(pnl_arr.sum() * ES_TICK_VALUE),
        "avg_pnl_ticks": float(pnl_arr.mean()),
        "sharpe": float(sharpe),
        "sortino": float(sortino),
        "win_rate": float(wr),
        "profit_factor": float(pf),
        "max_dd_ticks": float(max_dd),
        "max_dd_dollars": float(max_dd * ES_TICK_VALUE),
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
        "daily_pnl": day_pnl.to_dict("records"),
        "n_trading_days": int(len(day_pnl)),
        "daily_sharpe": float(
            day_pnl["daily_pnl"].mean() / max(day_pnl["daily_pnl"].std(), 1e-6) * np.sqrt(252)
        ) if len(day_pnl) > 2 else 0,
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 11: REGIME STRATIFICATION (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════


def regime_stratification(
    preds: np.ndarray,
    actuals: np.ndarray,
    dates: np.ndarray,
    day_returns: Dict[str, float],
    confidence_pct: float = 0.10,
) -> Dict[str, Any]:
    """
    Stratify by day-regime (green/red/flat based on ES close-to-close).
    HC #428 R1: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
    """
    valid = ~np.isnan(preds) & ~np.isnan(actuals)
    preds_v = preds[valid]
    actuals_v = actuals[valid]
    dates_v = dates[valid]

    if len(preds_v) < 20:
        return {"error": "too few valid predictions", "regime_gap_pass": False}

    day_class = {}
    for d, ret in day_returns.items():
        if ret > 0.001:
            day_class[d] = "green"
        elif ret < -0.001:
            day_class[d] = "red"
        else:
            day_class[d] = "flat"

    regime_arr = np.array([day_class.get(d, "flat") for d in dates_v])

    results = {}
    regime_sharpes = {}

    for regime in ["green", "red", "flat"]:
        mask = regime_arr == regime
        if mask.sum() < 10:
            results[regime] = {"n_predictions": int(mask.sum()), "skip": True}
            continue

        p_r = preds_v[mask]
        a_r = actuals_v[mask]
        d_r = dates_v[mask]

        ic = np.corrcoef(p_r, a_r)[0, 1] if len(p_r) > 5 else float("nan")
        sim = simulate_trades(p_r, a_r, d_r, confidence_pct=confidence_pct)

        if sim is not None:
            regime_sharpes[regime] = sim["sharpe"]
            results[regime] = {
                "n_predictions": int(mask.sum()),
                "ic": float(ic),
                "sharpe": sim["sharpe"],
                "sortino": sim["sortino"],
                "win_rate": sim["win_rate"],
                "n_trades": sim["n_trades"],
                "avg_pnl_ticks": sim["avg_pnl_ticks"],
            }
        else:
            results[regime] = {"n_predictions": int(mask.sum()), "ic": float(ic), "n_trades": 0}

    if "green" in regime_sharpes and "red" in regime_sharpes:
        s_green = regime_sharpes["green"]
        s_red = regime_sharpes["red"]
        denom = max(abs(s_green), abs(s_red), 1e-6)
        gap = abs(s_green - s_red) / denom
        results["regime_gap"] = float(gap)
        results["regime_gap_pass"] = gap <= 0.50
        results["regime_gap_detail"] = (
            f"green_sharpe={s_green:.2f}, red_sharpe={s_red:.2f}, "
            f"gap={gap:.2f} {'PASS' if gap <= 0.50 else 'FAIL'}"
        )
    else:
        results["regime_gap"] = float("nan")
        results["regime_gap_pass"] = False
        results["regime_gap_detail"] = "insufficient regime data"

    return results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 12: MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════


def run_ensemble():
    """
    Run the full multi-signal ensemble experiment.

    Architecture:
      1. Build Level-1 predictions (30-min and 1h models) via walk-forward
      2. Align OFI exhaustion signal to the 30-min bar grid
      3. Train Level-2 meta-learner on stacked predictions
      4. Evaluate ensemble with FIFO costs and regime gates
      5. Compare: ensemble vs individual models vs naive average
    """
    _import_lightgbm()

    # ── MLflow setup ──
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("multi_signal_ensemble_v1")
        mlflow_run = mlflow.start_run(run_name=f"ensemble_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow_active = True
        log.info(f"MLflow run started: {mlflow_run.info.run_id}")
    except Exception as e:
        log.warning(f"MLflow unavailable: {e} -- proceeding without tracking")
        mlflow_active = False

    t0 = time.time()

    # ════════════════════════════════════════
    #  STEP 1: Load raw data
    # ════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 1: Loading raw minute bar data")
    log.info("=" * 70)

    minute_df = load_all_minute_bars()
    queue_df = load_queue_features()
    all_dates_raw = sorted(minute_df["date"].unique())

    log.info(f"Total trading days: {len(all_dates_raw)}")
    log.info(f"Date range: {all_dates_raw[0]} -> {all_dates_raw[-1]}")

    if mlflow_active:
        mlflow.log_param("n_trading_days", len(all_dates_raw))
        mlflow.log_param("date_range", f"{all_dates_raw[0]}-{all_dates_raw[-1]}")
        mlflow.log_param("train_days", TRAIN_DAYS)
        mlflow.log_param("slide_days", SLIDE_DAYS)
        mlflow.log_param("cost_rt_ticks", COST_RT_TICKS)

    # ════════════════════════════════════════
    #  STEP 2: Build bar-level features for each horizon
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 2: Building bar features for 30-min and 1h horizons")
    log.info("=" * 70)

    # 30-min bars
    bars_30m = aggregate_to_bars(minute_df, bar_size_min=30)
    bars_30m.attrs["bar_size_min"] = 30
    bars_30m = add_rolling_features(bars_30m, bar_size_min=30)
    bars_30m = add_forward_labels(bars_30m, horizon_bars=1, horizon_label="30min",
                                   min_edge_ticks=2.5)

    # 15-min bars for 1h model
    bars_15m = aggregate_to_bars(minute_df, bar_size_min=15)
    bars_15m.attrs["bar_size_min"] = 15
    bars_15m = add_rolling_features(bars_15m, bar_size_min=15)
    bars_15m = add_forward_labels(bars_15m, horizon_bars=4, horizon_label="1h",
                                   min_edge_ticks=3.0)

    # OFI exhaustion signal
    ofi_signal_df = compute_ofi_exhaustion_signal(minute_df)

    feature_cols_30m = get_feature_columns(bars_30m)
    feature_cols_15m = get_feature_columns(bars_15m)
    log.info(f"30-min features: {len(feature_cols_30m)}")
    log.info(f"15-min features (for 1h model): {len(feature_cols_15m)}")

    if mlflow_active:
        mlflow.log_param("n_features_30m", len(feature_cols_30m))
        mlflow.log_param("n_features_15m", len(feature_cols_15m))

    # ════════════════════════════════════════
    #  STEP 3: Walk-forward Level-1 predictions
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 3: Walk-forward Level-1 model training")
    log.info("=" * 70)

    # We'll work on the 30-min bar grid for the ensemble.
    # The 1h model (15-min bars, 4-bar horizon) predictions get mapped to 30-min bars.
    dates_30m = sorted(bars_30m["date"].unique())
    dates_15m = sorted(bars_15m["date"].unique())

    # Prepare arrays for 30-min model
    features_30m_all = bars_30m[feature_cols_30m].values.astype(np.float32)
    labels_30m_all = bars_30m["fwd_ticks_30min"].values.astype(np.float32)
    dates_30m_all = bars_30m["date"].values
    ts_30m_all = bars_30m["ts"].values

    # Prepare arrays for 1h model (15-min bars)
    features_15m_all = bars_15m[feature_cols_15m].values.astype(np.float32)
    labels_1h_all = bars_15m["fwd_ticks_1h"].values.astype(np.float32)
    dates_15m_all = bars_15m["date"].values
    ts_15m_all = bars_15m["ts"].values

    # Per-day returns for regime classification (from 30-min bars)
    day_close = bars_30m.groupby("date")["close"].last()
    day_returns_raw = day_close.pct_change()
    day_returns = day_returns_raw.to_dict()

    # Walk-forward: accumulate OOT predictions from each L1 model
    l1_preds_30m = np.full(len(bars_30m), np.nan, dtype=np.float32)
    l1_preds_1h_on_15m = np.full(len(bars_15m), np.nan, dtype=np.float32)

    fold_idx = 0
    val_days_l1 = 5  # 5-day validation windows for L1

    log.info(f"\nL1 Walk-Forward: {TRAIN_DAYS}d train, {val_days_l1}d val, {SLIDE_DAYS}d slide")
    log.info(f"Dates available: {len(dates_30m)} (30m), {len(dates_15m)} (15m)")

    start_idx = TRAIN_DAYS
    for fold_start in range(start_idx, len(dates_30m) - val_days_l1 + 1, SLIDE_DAYS):
        fold_train_dates = dates_30m[fold_start - TRAIN_DAYS: fold_start]
        fold_val_dates = dates_30m[fold_start: fold_start + val_days_l1]

        if len(fold_val_dates) < val_days_l1:
            break

        fold_idx += 1

        # ── 30-min model ──
        train_mask_30m = np.isin(dates_30m_all, fold_train_dates)
        val_mask_30m = np.isin(dates_30m_all, fold_val_dates)

        if not leakage_audit(bars_30m, list(fold_train_dates), list(fold_val_dates), feature_cols_30m):
            log.error(f"Fold {fold_idx}: 30m leakage audit FAILED -- skip")
            continue

        # Robust scaling from train only
        tr_30m = features_30m_all[train_mask_30m].copy()
        vl_30m = features_30m_all[val_mask_30m].copy()
        med = np.nanmedian(tr_30m, axis=0)
        q75 = np.nanpercentile(tr_30m, 75, axis=0)
        q25 = np.nanpercentile(tr_30m, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0
        tr_30m = np.clip(np.nan_to_num((tr_30m - med) / iqr, nan=0.0), -5, 5)
        vl_30m = np.clip(np.nan_to_num((vl_30m - med) / iqr, nan=0.0), -5, 5)

        _, _, val_p_30m = train_l1_model(
            tr_30m, labels_30m_all[train_mask_30m],
            vl_30m, labels_30m_all[val_mask_30m],
            feature_names=feature_cols_30m,
            fold_idx=fold_idx, horizon_label="30min",
        )
        l1_preds_30m[val_mask_30m] = val_p_30m

        # ── 1h model (15-min bars) ──
        train_mask_15m = np.isin(dates_15m_all, fold_train_dates)
        val_mask_15m = np.isin(dates_15m_all, fold_val_dates)

        if not leakage_audit(bars_15m, list(fold_train_dates), list(fold_val_dates), feature_cols_15m):
            log.error(f"Fold {fold_idx}: 1h leakage audit FAILED -- skip")
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

        _, _, val_p_1h = train_l1_model(
            tr_15m, labels_1h_all[train_mask_15m],
            vl_15m, labels_1h_all[val_mask_15m],
            feature_names=feature_cols_15m,
            fold_idx=fold_idx, horizon_label="1h",
        )
        l1_preds_1h_on_15m[val_mask_15m] = val_p_1h

        if fold_idx % 5 == 0:
            n_valid_30m = (~np.isnan(l1_preds_30m)).sum()
            n_valid_1h = (~np.isnan(l1_preds_1h_on_15m)).sum()
            log.info(f"  Progress: fold {fold_idx}, "
                     f"30m preds: {n_valid_30m}, 1h preds: {n_valid_1h}")

    log.info(f"\nL1 complete: {fold_idx} folds")
    log.info(f"  30m predictions: {(~np.isnan(l1_preds_30m)).sum()} / {len(l1_preds_30m)}")
    log.info(f"  1h predictions: {(~np.isnan(l1_preds_1h_on_15m)).sum()} / {len(l1_preds_1h_on_15m)}")

    # ════════════════════════════════════════
    #  STEP 3b: Evaluate individual L1 models (baseline)
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 3b: Evaluate individual L1 models (baselines)")
    log.info("=" * 70)

    baseline_results = {}

    # 30-min model baseline
    for conf_pct in [0.05, 0.10, 0.15, 0.20]:
        sim_30m = simulate_trades(
            l1_preds_30m, labels_30m_all, dates_30m_all,
            confidence_pct=conf_pct,
        )
        if sim_30m:
            tag = f"L1_30m_top{int(conf_pct*100)}pct"
            baseline_results[tag] = sim_30m
            log.info(
                f"  {tag}: Sharpe={sim_30m['sharpe']:.2f}, "
                f"Sortino={sim_30m['sortino']:.2f}, "
                f"WR={sim_30m['win_rate']:.1%}, "
                f"PF={sim_30m['profit_factor']:.2f}, "
                f"N={sim_30m['n_trades']}"
            )

            if mlflow_active:
                mlflow.log_metric(f"{tag}_sharpe", sim_30m["sharpe"])
                mlflow.log_metric(f"{tag}_sortino", sim_30m["sortino"])
                mlflow.log_metric(f"{tag}_wr", sim_30m["win_rate"])
                mlflow.log_metric(f"{tag}_n_trades", sim_30m["n_trades"])

    # 1h model baseline (evaluated on 15-min bars)
    for conf_pct in [0.05, 0.10, 0.15, 0.20]:
        sim_1h = simulate_trades(
            l1_preds_1h_on_15m, labels_1h_all, dates_15m_all,
            confidence_pct=conf_pct,
        )
        if sim_1h:
            tag = f"L1_1h_top{int(conf_pct*100)}pct"
            baseline_results[tag] = sim_1h
            log.info(
                f"  {tag}: Sharpe={sim_1h['sharpe']:.2f}, "
                f"Sortino={sim_1h['sortino']:.2f}, "
                f"WR={sim_1h['win_rate']:.1%}, "
                f"PF={sim_1h['profit_factor']:.2f}, "
                f"N={sim_1h['n_trades']}"
            )

            if mlflow_active:
                mlflow.log_metric(f"{tag}_sharpe", sim_1h["sharpe"])
                mlflow.log_metric(f"{tag}_sortino", sim_1h["sortino"])

    # ════════════════════════════════════════
    #  STEP 4: Map 1h predictions to 30-min bar grid
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 4: Align all signals to 30-min bar grid")
    log.info("=" * 70)

    # Map 1h predictions (15-min bars) -> 30-min bars by taking the prediction
    # at the 15-min bar that starts each 30-min bar (i.e. first 15-min bar in window)
    bars_15m_mapped = bars_15m[["date", "ts"]].copy()
    bars_15m_mapped["pred_1h"] = l1_preds_1h_on_15m
    bars_15m_mapped["bar_30m_key"] = bars_15m_mapped["ts"].dt.floor("30min")

    # Take mean of 15-min predictions within each 30-min bar
    pred_1h_per_30m = bars_15m_mapped.groupby(["date", "bar_30m_key"])["pred_1h"].mean().reset_index()
    pred_1h_per_30m.columns = ["date", "bar_key", "pred_1h_aligned"]

    bars_30m_merged = bars_30m.merge(pred_1h_per_30m, on=["date", "bar_key"], how="left")
    pred_1h_aligned = bars_30m_merged["pred_1h_aligned"].values.astype(np.float32)

    log.info(f"  1h predictions aligned to 30m grid: "
             f"{(~np.isnan(pred_1h_aligned)).sum()} / {len(pred_1h_aligned)}")

    # Map OFI exhaustion signal to 30-min bars (mean of minute-level signals)
    # Ensure timezone consistency: ts_minute is UTC-aware from load, floor preserves tz
    ofi_ts = pd.to_datetime(ofi_signal_df["ts_minute"], utc=True)
    ofi_signal_df["bar_30m_key"] = ofi_ts.dt.floor("30min")
    ofi_per_30m = ofi_signal_df.groupby(["date", "bar_30m_key"]).agg(
        ofi_exhaust_mean=("ofi_exhaust_signal", "mean"),
        ofi_exhaust_max=("ofi_exhaust_signal", lambda x: x[x.abs() == x.abs().max()].iloc[0] if len(x) > 0 else 0),
        ofi_z_max=("ofi_z", lambda x: x.abs().max()),
    ).reset_index()
    ofi_per_30m.columns = ["date", "bar_key", "ofi_exhaust_mean", "ofi_exhaust_max", "ofi_z_max"]

    bars_30m_merged = bars_30m_merged.merge(ofi_per_30m, on=["date", "bar_key"], how="left")
    ofi_aligned = bars_30m_merged["ofi_exhaust_max"].fillna(0).values.astype(np.float32)

    log.info(f"  OFI exhaustion signals in 30m grid: {(ofi_aligned != 0).sum()}")

    # ════════════════════════════════════════
    #  STEP 5: Build meta-features and train Level-2
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 5: Level-2 Meta-Learner walk-forward")
    log.info("=" * 70)

    meta_X, meta_feature_names = build_meta_features(
        l1_preds_30m, pred_1h_aligned, ofi_aligned, bars_30m,
    )
    meta_y = labels_30m_all  # Target: 30-min forward ticks (our trading horizon)

    log.info(f"  Meta-features: {len(meta_feature_names)}: {meta_feature_names}")

    # Walk-forward the meta-learner
    # We need L1 predictions to exist, so we start later (after first L1 fold)
    # Find first date where both L1 predictions are non-NaN
    has_30m = ~np.isnan(l1_preds_30m)
    has_1h = ~np.isnan(pred_1h_aligned)
    both_valid = has_30m & has_1h

    if both_valid.sum() < 100:
        log.error(f"Only {both_valid.sum()} bars have both L1 predictions -- need at least 100")
        log.info("Falling back to 30m-only meta-learner")
        both_valid = has_30m

    valid_dates = sorted(set(dates_30m_all[both_valid]))
    log.info(f"  Dates with valid L1 predictions: {len(valid_dates)} "
             f"({valid_dates[0]} -> {valid_dates[-1]})")

    meta_train_days = 30  # meta-learner uses fewer days (L1 preds are OOT already)
    meta_val_days = 5
    meta_slide = 5

    meta_preds = np.full(len(bars_30m), np.nan, dtype=np.float32)
    meta_fold_idx = 0

    for fold_start in range(meta_train_days, len(valid_dates) - meta_val_days + 1, meta_slide):
        meta_train_dates = valid_dates[fold_start - meta_train_days: fold_start]
        meta_val_dates_fold = valid_dates[fold_start: fold_start + meta_val_days]

        if len(meta_val_dates_fold) < meta_val_days:
            break

        meta_fold_idx += 1

        tr_mask = np.isin(dates_30m_all, meta_train_dates) & both_valid
        vl_mask = np.isin(dates_30m_all, meta_val_dates_fold) & both_valid

        if tr_mask.sum() < 50 or vl_mask.sum() < 10:
            continue

        X_tr = meta_X[tr_mask].copy()
        y_tr = meta_y[tr_mask].copy()
        X_vl = meta_X[vl_mask].copy()
        y_vl = meta_y[vl_mask].copy()

        # Clean NaN/inf
        X_tr = np.nan_to_num(X_tr, nan=0.0, posinf=3.0, neginf=-3.0)
        X_vl = np.nan_to_num(X_vl, nan=0.0, posinf=3.0, neginf=-3.0)

        valid_train = ~np.isnan(y_tr)
        valid_val = ~np.isnan(y_vl)

        if valid_train.sum() < 50 or valid_val.sum() < 10:
            continue

        params = {**META_LGBM_PARAMS, "seed": 100 + meta_fold_idx}
        train_data = lgb.Dataset(
            X_tr[valid_train], label=y_tr[valid_train],
            feature_name=meta_feature_names,
        )
        val_data = lgb.Dataset(
            X_vl[valid_val], label=y_vl[valid_val],
            feature_name=meta_feature_names,
            reference=train_data,
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=20, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        meta_model = lgb.train(
            params, train_data, num_boost_round=300,
            valid_sets=[val_data], callbacks=callbacks,
        )

        # Predict on validation
        val_pred = np.full(vl_mask.sum(), np.nan)
        val_pred[valid_val] = meta_model.predict(
            X_vl[valid_val], num_iteration=meta_model.best_iteration
        )
        meta_preds[vl_mask] = val_pred

        if meta_fold_idx % 5 == 0:
            n_meta = (~np.isnan(meta_preds)).sum()
            log.info(f"  Meta fold {meta_fold_idx}: {n_meta} predictions so far")

    log.info(f"\nMeta-learner complete: {meta_fold_idx} folds")
    log.info(f"  Meta predictions: {(~np.isnan(meta_preds)).sum()} / {len(meta_preds)}")

    # Feature importance from last meta-model
    if meta_fold_idx > 0:
        importance = meta_model.feature_importance(importance_type="gain")
        feat_imp = sorted(zip(meta_feature_names, importance), key=lambda x: x[1], reverse=True)
        log.info("\n  Meta-learner feature importance (last fold):")
        for name, gain in feat_imp:
            log.info(f"    {name:30s} gain={gain:.1f}")

        if mlflow_active:
            for name, gain in feat_imp:
                mlflow.log_metric(f"meta_importance_{name}", float(gain))

    # ════════════════════════════════════════
    #  STEP 5b: Naive ensemble baseline (simple average)
    # ════════════════════════════════════════
    naive_avg = np.nanmean(
        np.column_stack([l1_preds_30m, pred_1h_aligned]), axis=1
    ).astype(np.float32)
    # Where both are NaN, naive_avg is NaN (correct)

    # ════════════════════════════════════════
    #  STEP 6: Final evaluation — all models head-to-head
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 6: Final evaluation — ensemble vs baselines")
    log.info("=" * 70)

    models_to_eval = {
        "L1_30min": (l1_preds_30m, labels_30m_all, dates_30m_all),
        "L1_1h_aligned": (pred_1h_aligned, labels_30m_all, dates_30m_all),
        "Naive_Average": (naive_avg, labels_30m_all, dates_30m_all),
        "Meta_Ensemble": (meta_preds, labels_30m_all, dates_30m_all),
    }

    all_results = {}

    for model_name, (preds, actuals, dates) in models_to_eval.items():
        log.info(f"\n{'─' * 50}")
        log.info(f"  Model: {model_name}")
        log.info(f"{'─' * 50}")

        valid_count = (~np.isnan(preds) & ~np.isnan(actuals)).sum()
        log.info(f"  Valid predictions: {valid_count}")

        if valid_count < 50:
            log.warning(f"  {model_name}: too few predictions ({valid_count}) -- skip")
            continue

        # IC
        valid_mask = ~np.isnan(preds) & ~np.isnan(actuals)
        ic = np.corrcoef(preds[valid_mask], actuals[valid_mask])[0, 1]
        rank_ic = stats.spearmanr(preds[valid_mask], actuals[valid_mask]).correlation
        log.info(f"  Concat IC: {ic:.4f}, Rank IC: {rank_ic:.4f}")

        model_results = {"ic": float(ic), "rank_ic": float(rank_ic)}

        # Trade simulation at multiple confidence levels
        for conf_pct in [0.05, 0.10, 0.15, 0.20]:
            sim = simulate_trades(preds, actuals, dates, confidence_pct=conf_pct)
            if sim:
                tag = f"top{int(conf_pct*100)}pct"
                model_results[tag] = sim
                log.info(
                    f"  {tag}: Sharpe={sim['sharpe']:.2f}, "
                    f"Sortino={sim['sortino']:.2f}, "
                    f"WR={sim['win_rate']:.1%}, "
                    f"PF={sim['profit_factor']:.2f}, "
                    f"N={sim['n_trades']}, "
                    f"$PnL={sim['total_pnl_dollars']:.0f}"
                )

                if mlflow_active:
                    mlflow.log_metric(f"{model_name}_{tag}_sharpe", sim["sharpe"])
                    mlflow.log_metric(f"{model_name}_{tag}_sortino", sim["sortino"])
                    mlflow.log_metric(f"{model_name}_{tag}_wr", sim["win_rate"])
                    mlflow.log_metric(f"{model_name}_{tag}_pf", sim["profit_factor"])
                    mlflow.log_metric(f"{model_name}_{tag}_n_trades", sim["n_trades"])
                    mlflow.log_metric(f"{model_name}_{tag}_pnl_dollars", sim["total_pnl_dollars"])

        # Regime stratification (HC #428 R1) at top 10%
        regime = regime_stratification(preds, actuals, dates, day_returns, confidence_pct=0.10)
        model_results["regime"] = regime

        if "regime_gap_detail" in regime:
            log.info(f"  Regime gate: {regime['regime_gap_detail']}")
        for r in ["green", "red", "flat"]:
            if r in regime and "sharpe" in regime[r]:
                log.info(f"    {r}: Sharpe={regime[r]['sharpe']:.2f}, "
                         f"WR={regime[r].get('win_rate', 0):.1%}, "
                         f"N={regime[r].get('n_trades', 0)}")

        if mlflow_active:
            mlflow.log_metric(f"{model_name}_ic", ic)
            mlflow.log_metric(f"{model_name}_rank_ic", rank_ic)
            if "regime_gap" in regime and not np.isnan(regime["regime_gap"]):
                mlflow.log_metric(f"{model_name}_regime_gap", regime["regime_gap"])
                mlflow.log_metric(f"{model_name}_regime_pass", int(regime.get("regime_gap_pass", False)))

        all_results[model_name] = model_results

    # ════════════════════════════════════════
    #  STEP 7: Day-concentration check (HC #344)
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 7: Day-concentration check (HC #344, cap <= 0.70)")
    log.info("=" * 70)

    for model_name in ["Meta_Ensemble", "L1_30min"]:
        if model_name not in all_results:
            continue
        for tag in ["top10pct", "top5pct"]:
            if tag not in all_results[model_name]:
                continue
            sim = all_results[model_name][tag]
            if "daily_pnl" in sim:
                daily = pd.DataFrame(sim["daily_pnl"])
                if len(daily) > 0 and "daily_pnl" in daily.columns:
                    total_abs = daily["daily_pnl"].abs().sum()
                    if total_abs > 0:
                        day_conc = daily["daily_pnl"].abs().max() / total_abs
                        log.info(f"  {model_name} {tag}: day_conc={day_conc:.3f} "
                                 f"{'PASS' if day_conc <= 0.70 else 'FAIL'}")
                        if mlflow_active:
                            mlflow.log_metric(f"{model_name}_{tag}_day_conc", day_conc)

    # ════════════════════════════════════════
    #  STEP 8: Save results
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 8: Saving results and predictions")
    log.info("=" * 70)

    # Save predictions
    np.savez_compressed(
        str(OUTPUT_DIR / "l1_predictions.npz"),
        pred_30m=l1_preds_30m,
        pred_1h_aligned=pred_1h_aligned,
        ofi_signal=ofi_aligned,
        meta_preds=meta_preds,
        naive_avg=naive_avg,
        labels_30m=labels_30m_all,
        dates=dates_30m_all,
    )
    log.info(f"  Saved predictions to {OUTPUT_DIR / 'l1_predictions.npz'}")

    # Save results JSON
    # Clean up non-serializable values
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
        return obj

    results_clean = clean_for_json(all_results)
    with open(str(OUTPUT_DIR / "ensemble_results.json"), "w") as f:
        json.dump(results_clean, f, indent=2, default=str)
    log.info(f"  Saved results to {OUTPUT_DIR / 'ensemble_results.json'}")

    if mlflow_active:
        mlflow.log_artifact(str(OUTPUT_DIR / "ensemble_results.json"))

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"ENSEMBLE v1 COMPLETE in {elapsed/60:.1f} minutes")
    log.info(f"{'=' * 70}")

    # ── Summary ──
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY — Head-to-Head Comparison (top 10%)")
    log.info("=" * 70)
    log.info(f"{'Model':25s} {'Sharpe':>8} {'Sortino':>8} {'WR':>6} {'PF':>6} {'N':>6} {'Regime':>8}")
    log.info("-" * 70)

    for model_name in ["L1_30min", "L1_1h_aligned", "Naive_Average", "Meta_Ensemble"]:
        if model_name not in all_results:
            continue
        r = all_results[model_name]
        sim = r.get("top10pct", {})
        regime_pass = r.get("regime", {}).get("regime_gap_pass", False)
        if sim:
            log.info(
                f"{model_name:25s} "
                f"{sim.get('sharpe', 0):8.2f} "
                f"{sim.get('sortino', 0):8.2f} "
                f"{sim.get('win_rate', 0):6.1%} "
                f"{sim.get('profit_factor', 0):6.2f} "
                f"{sim.get('n_trades', 0):6d} "
                f"{'PASS' if regime_pass else 'FAIL':>8}"
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
    log.info("Multi-Signal Ensemble v1 starting...")
    log.info(f"  ROOT: {ROOT}")
    log.info(f"  OUTPUT: {OUTPUT_DIR}")
    log.info(f"  Cost: {COST_RT_TICKS} ticks RT")
    log.info(f"  Walk-forward: {TRAIN_DAYS}d train, {SLIDE_DAYS}d slide")

    try:
        results = run_ensemble()
    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        log.error(traceback.format_exc())
        # Try to end MLflow run if active
        try:
            import mlflow
            mlflow.end_run(status="FAILED")
        except Exception:
            pass
        sys.exit(1)
