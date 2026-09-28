#!/usr/bin/env python3
"""
Dynamic Trade Management v5 — Adaptive Trailing Stops + ML Hybrid
===================================================================

Fixes v4 issues:
  - v4 had only 238 trades over 66 days (too restrictive confluence gate 3/6)
  - v4 static baseline Sharpe was -1.81 (below standalone entry model Sharpe 4.03)
  - v4 completed in 1.1 min (only 66 of 197 days had trades due to 60d warm-up)

KEY CHANGES FROM v4:
  1. Confluence floor lowered to 2/6 — score used as FEATURE, not hard gate
  2. 40d train window (not 60d) — OOT predictions start day 41, ~157 trading days
  3. Richer management features: mfe_ticks, mae_ticks, mfe_mae_ratio,
     pnl_velocity, bars_since_mfe, ofi_reversal, volume_surge,
     entry_confluence_score
  4. Adaptive trailing stop: locks in profits based on MFE progression
  5. Four exit strategies compared head-to-head:
     STATIC_30, TRAILING_ONLY, ML_ONLY, HYBRID (trailing + ML override)
  6. Management model can override trailing stop if P(improve) > 0.7

ARCHITECTURE:
  Layer 1 — Entry: walk-forward LightGBM on 30-min bars predicting fwd_ticks.
            Confluence scored 0-6 but gate is >= 2 (not 3).
            Confluence score fed as feature to management model.

  Layer 2 — Management: at every 5-min checkpoint during the trade:
            a) Adaptive trailing stop computes a floor based on MFE
            b) ML management model predicts P(trade will improve)
            c) HYBRID mode: trailing stop decides exit UNLESS ML overrides

  Layer 3 — Exit strategies:
            STATIC_30:    hold 30 min, exit at close
            TRAILING_ONLY: adaptive trailing stop, no ML
            ML_ONLY:      ML management model decides hold/exit
            HYBRID:       trailing stop + ML override (full system)

CONSTRAINTS:
  - Walk-forward: 40d train, 5d slide — SLIDING only (HC #0)
  - Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission)
  - Regime-agnostic gate: |Sharpe_green - Sharpe_red| / max < 0.50 (HC #428)
  - Day-concentration cap <= 0.70 (HC #344)
  - MLflow logging mandatory

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python -u \\
      alpha_discovery/trade_management_v5_adaptive.py 2>&1 | \\
      tee logs/trade_management_v5_adaptive.log

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
OUTPUT_DIR = ROOT / "output" / "trade_management_v5_adaptive"
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
    format="%(asctime)s [TM-v5] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "trade_management_v5_adaptive.log")),
    ],
)
log = logging.getLogger("TM-v5")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376
COST_RT_TICKS = 2.376  # market entry + market exit + RT commission

# ─────────────────────────────────────────────
#  WALK-FORWARD CONFIG — FASTER WARM-UP (40d not 60d)
# ─────────────────────────────────────────────
TRAIN_DAYS = 40         # v4 was 60 → only 66 OOT days; 40 → ~157 OOT days
SLIDE_DAYS = 5

# Management model uses shorter training window
MGMT_TRAIN_DAYS = 20    # v4 was 30; shorter = more responsive
MGMT_SLIDE_DAYS = 5

# ─────────────────────────────────────────────
#  TRADE CONFIG
# ─────────────────────────────────────────────
ENTRY_BAR_SIZE_MIN = 30        # entry signal bar resolution
MGMT_BAR_SIZE_MIN = 5          # management checkpoint resolution (5-min bars)
STATIC_HOLD_MINUTES = 30       # baseline static hold
MAX_HOLD_MINUTES = 60          # max extension for CONFIRMED trades
CONFLUENCE_MIN = 2             # v4 was 3 → too restrictive; 2 lets marginal entries through
ENTRY_CONFIDENCE_PCT = 0.10    # top/bottom 10% for entry model signal
IMPROVE_THRESHOLD_TICKS = 0.5  # minimum improvement to justify holding

# ─────────────────────────────────────────────
#  ADAPTIVE TRAILING STOP CONFIG
# ─────────────────────────────────────────────
TRAILING_MFE_TIER1 = 3.0       # MFE > 3 ticks: floor = MFE - 2
TRAILING_MFE_TIER2 = 6.0       # MFE > 6 ticks: floor = MFE - 1.5
TRAILING_OFFSET_TIER1 = 2.0    # ticks below MFE for tier 1
TRAILING_OFFSET_TIER2 = 1.5    # ticks below MFE for tier 2
BARS_SINCE_MFE_EXIT = 4        # exit after 4 mgmt bars (20 min) with no new MFE
ML_OVERRIDE_THRESHOLD = 0.70   # ML can override trailing stop if P(improve) > 0.70

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

LGBM_MGMT_PARAMS = {
    "objective": "binary",
    "metric": "auc",
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
#  SECTION 4: OFI EXHAUSTION SIGNAL (RULE-BASED)
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
#  SECTION 8: LEVEL-1 MODEL TRAINING (entry models)
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
#  SECTION 9: CONFLUENCE SCORING (Layer 1)
# ═══════════════════════════════════════════════════════════════════


def compute_confluence_scores(
    pred_30m: np.ndarray,
    pred_1h: np.ndarray,
    ofi_signal: np.ndarray,
    bars_30m: pd.DataFrame,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute confluence scores for each 30-min bar.

    6 signals checked for directional agreement:
      1. 30-min model prediction strength (top 30% magnitude)
      2. 1h model agreement (same direction)
      3. OFI exhaustion alignment
      4. Recent momentum direction (4-bar lookback return)
      5. Volume confirmation (above-average volume in direction)
      6. Volatility regime (not too high, not too low)

    Returns:
      confluence_score: float array (0-6, how many signals agree)
      confluence_direction: -1/0/+1 array (consensus direction)
      confluence_detail: array of per-bar 6-bit detail
    """
    n = len(pred_30m)
    confluence_score = np.zeros(n, dtype=np.float32)
    confluence_direction = np.zeros(n, dtype=np.float32)
    confluence_detail = np.zeros(n, dtype=np.float32)

    # Precompute thresholds from rolling windows (causal)
    pred_30m_abs = np.abs(pred_30m)
    pred_30m_pctile = np.zeros(n, dtype=np.float32)
    for i in range(100, n):
        window = pred_30m_abs[max(0, i - 200):i]
        valid_w = window[~np.isnan(window)]
        if len(valid_w) > 10:
            pred_30m_pctile[i] = np.searchsorted(np.sort(valid_w), pred_30m_abs[i]) / len(valid_w)

    # Extract bar features
    ret_4bar = bars_30m["ret_lb_4bar"].values if "ret_lb_4bar" in bars_30m.columns else np.zeros(n)
    vol_rel = bars_30m["vol_rel_4bar"].values if "vol_rel_4bar" in bars_30m.columns else np.ones(n)
    rvol_4bar = bars_30m["rvol_4bar"].values if "rvol_4bar" in bars_30m.columns else np.ones(n)
    sv_ratio = bars_30m["signed_volume_ratio"].values if "signed_volume_ratio" in bars_30m.columns else np.zeros(n)

    # Rolling volatility percentile for regime classification
    rvol_pctile = np.zeros(n, dtype=np.float32)
    for i in range(50, n):
        window = rvol_4bar[max(0, i - 200):i]
        valid_w = window[~np.isnan(window)]
        if len(valid_w) > 10:
            rvol_pctile[i] = np.searchsorted(np.sort(valid_w), rvol_4bar[i]) / len(valid_w)

    for i in range(n):
        if np.isnan(pred_30m[i]):
            continue

        # Determine candidate direction from 30m model
        if pred_30m[i] > 0:
            candidate_dir = 1.0
        elif pred_30m[i] < 0:
            candidate_dir = -1.0
        else:
            continue

        signals = 0
        detail_bits = 0

        # Signal 1: 30-min model prediction strength (top 30% magnitude)
        if pred_30m_pctile[i] >= 0.70:
            signals += 1
            detail_bits |= 1

        # Signal 2: 1h model agrees on direction
        if not np.isnan(pred_1h[i]) and np.sign(pred_1h[i]) == candidate_dir:
            signals += 1
            detail_bits |= 2

        # Signal 3: OFI exhaustion alignment
        if ofi_signal[i] != 0 and np.sign(ofi_signal[i]) == candidate_dir:
            signals += 1
            detail_bits |= 4

        # Signal 4: Recent momentum direction (4-bar return aligns)
        if not np.isnan(ret_4bar[i]) and np.sign(ret_4bar[i]) == candidate_dir:
            signals += 1
            detail_bits |= 8

        # Signal 5: Volume confirmation (above-average volume, signed volume aligns)
        if vol_rel[i] > 1.0 and np.sign(sv_ratio[i]) == candidate_dir:
            signals += 1
            detail_bits |= 16

        # Signal 6: Volatility regime (not extreme — between 15th and 85th percentile)
        if 0.15 <= rvol_pctile[i] <= 0.85:
            signals += 1
            detail_bits |= 32

        confluence_score[i] = signals
        confluence_direction[i] = candidate_dir
        confluence_detail[i] = detail_bits

    n_high = (confluence_score >= CONFLUENCE_MIN).sum()
    log.info(f"Confluence scoring: {n_high} bars with score >= {CONFLUENCE_MIN} "
             f"out of {(confluence_score > 0).sum()} scored bars")

    # Distribution
    for s in range(7):
        cnt = (confluence_score == s).sum()
        if cnt > 0:
            log.info(f"  Score {s}: {cnt} bars ({cnt / max(n, 1) * 100:.1f}%)")

    return confluence_score, confluence_direction, confluence_detail


# ═══════════════════════════════════════════════════════════════════
#  SECTION 10: TRADE RECONSTRUCTION FROM ENTRY SIGNALS
# ═══════════════════════════════════════════════════════════════════


def reconstruct_trades(
    bars_30m: pd.DataFrame,
    minute_df: pd.DataFrame,
    pred_30m: np.ndarray,
    confluence_score: np.ndarray,
    confluence_direction: np.ndarray,
    min_confluence: int = CONFLUENCE_MIN,
    confidence_pct: float = ENTRY_CONFIDENCE_PCT,
) -> List[Dict]:
    """
    Reconstruct trades from entry signals.

    For each 30-min bar where:
      - model prediction is in top/bottom confidence_pct
      - confluence score >= min_confluence (now 2, not 3)
      - direction matches confluence
    We create a trade record with minute-level price path for management.
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

        # Check entry conditions
        direction = 0
        if pred_30m[i] >= upper_thresh:
            direction = 1
        elif pred_30m[i] <= lower_thresh:
            direction = -1
        else:
            continue

        # Confluence gate — lowered to 2 (v4 was 3)
        if confluence_score[i] < min_confluence:
            continue

        # Direction must match confluence consensus
        if confluence_direction[i] != 0 and confluence_direction[i] != direction:
            continue

        date_str = bars_dates[i]
        entry_ts = pd.Timestamp(bars_ts[i])
        entry_price = bars_close[i]

        if date_str not in minute_lookup:
            continue

        day_minutes = minute_lookup[date_str]
        day_ts = day_minutes["ts_minute"].values

        # Find minute bars within the trade window (up to MAX_HOLD_MINUTES)
        entry_ts_np = np.datetime64(entry_ts)
        window_end = entry_ts + pd.Timedelta(minutes=MAX_HOLD_MINUTES)
        window_end_np = np.datetime64(window_end)

        # Get minute bars after entry within window
        mask = (day_ts >= entry_ts_np) & (day_ts <= window_end_np)
        trade_minutes = day_minutes[mask].copy()

        if len(trade_minutes) < 5:
            continue

        # Extract price path
        prices = trade_minutes["close"].values
        times = trade_minutes["ts_minute"].values
        ofi_vals = trade_minutes["ofi_1min"].values
        vol_vals = trade_minutes["volume"].values
        spread_vals = trade_minutes["spread_mean"].values
        sv_vals = trade_minutes["signed_volume"].values

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
            "confluence_score": int(confluence_score[i]),
            "confluence_detail": int(confluence_direction[i]),
            # Minute-level data for management model
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

        # Static hold outcomes (at various durations)
        for hold_min in [10, 15, 20, 25, 30, 45, 60]:
            idx_hold = min(hold_min, len(pnl_path) - 1)
            trade[f"pnl_{hold_min}min"] = float(pnl_path[idx_hold]) - COST_RT_TICKS

        trades.append(trade)

    log.info(f"Reconstructed {len(trades)} trades from entry signals "
             f"(confidence={confidence_pct:.0%}, min_confluence={min_confluence})")

    if trades:
        directions = [t["direction"] for t in trades]
        log.info(f"  Long: {sum(1 for d in directions if d == 1)}, "
                 f"Short: {sum(1 for d in directions if d == -1)}")
        confl_scores = [t["confluence_score"] for t in trades]
        log.info(f"  Confluence: mean={np.mean(confl_scores):.1f}, "
                 f"min={min(confl_scores)}, max={max(confl_scores)}")
        trade_dates = sorted(set(t["date"] for t in trades))
        log.info(f"  Trading days: {len(trade_dates)} "
                 f"({trade_dates[0]} -> {trade_dates[-1]})")

    return trades


# ═══════════════════════════════════════════════════════════════════
#  SECTION 11: MANAGEMENT FEATURES (Layer 2) — EXPANDED for v5
# ═══════════════════════════════════════════════════════════════════


def extract_management_features(
    trade: Dict,
    checkpoint_idx: int,
    prev_checkpoint_idx: int = -1,
) -> Optional[Dict]:
    """
    Extract management features at a specific checkpoint within the trade.

    v5 additions over v4:
      - mfe_ticks: max favorable excursion so far
      - mae_ticks: max adverse excursion so far
      - mfe_mae_ratio: mfe / max(mae, 1)
      - pnl_velocity: change in unrealized P&L over last 2 checkpoints
      - bars_since_mfe: how many checkpoints since peak MFE — stale trades
      - ofi_reversal: did OFI flip against our direction since entry?
      - volume_surge: current volume vs entry bar volume
      - entry_confluence_score: original confluence (higher = more patience)
    """
    pnl_path = trade["pnl_path"]
    n_minutes = trade["n_minutes"]

    if checkpoint_idx >= n_minutes or checkpoint_idx < 1:
        return None

    direction = trade["direction"]

    # Current state
    unrealized_pnl = pnl_path[checkpoint_idx]
    time_in_trade = checkpoint_idx
    mfe_so_far = float(trade["mfe_path"][checkpoint_idx])
    mae_so_far = float(trade["mae_path"][checkpoint_idx])

    # MFE/MAE ratio — v5 key feature
    mfe_mae_ratio = 0.0
    if abs(mae_so_far) > 0.1:
        mfe_mae_ratio = mfe_so_far / abs(mae_so_far)
    elif mfe_so_far > 0:
        mfe_mae_ratio = 5.0  # Strong positive, minimal drawdown

    # bars_since_mfe — v5 key feature: how long since we hit peak MFE?
    # Look back through the MFE path to find when MFE last increased
    mfe_current = trade["mfe_path"][checkpoint_idx]
    bars_since_mfe = 0
    for lookback in range(1, checkpoint_idx + 1):
        if trade["mfe_path"][checkpoint_idx - lookback] < mfe_current:
            break
        bars_since_mfe += 1
    # Convert to management-bar units (divide by MGMT_BAR_SIZE_MIN)
    bars_since_mfe_mgmt = bars_since_mfe / max(MGMT_BAR_SIZE_MIN, 1)

    # OFI direction alignment
    ofi_vals = trade["ofi_vals"]
    lookback_start = max(0, checkpoint_idx - 5)
    recent_ofi = ofi_vals[lookback_start:checkpoint_idx + 1]
    ofi_sum_recent = recent_ofi.sum()
    ofi_direction_aligned = float(np.sign(ofi_sum_recent) == direction)

    # ofi_reversal — v5: has OFI flipped against trade direction?
    # Compare first 5 min OFI direction vs recent 5 min
    entry_ofi = ofi_vals[:min(5, checkpoint_idx + 1)].sum()
    ofi_reversal = 0.0
    if checkpoint_idx >= 5:
        ofi_reversal = float(
            np.sign(entry_ofi) == direction and np.sign(ofi_sum_recent) != direction
        )

    # OFI momentum (slope over last 5 minutes)
    ofi_momentum = _safe_polyfit_slope(recent_ofi)

    # Volume trend
    vol_vals = trade["vol_vals"]
    recent_vol = vol_vals[lookback_start:checkpoint_idx + 1]
    volume_trend = _safe_polyfit_slope(recent_vol)

    # volume_surge — v5: current volume vs entry bar volume
    entry_vol = vol_vals[:min(5, checkpoint_idx + 1)].mean() if checkpoint_idx > 0 else vol_vals[0]
    current_vol = vol_vals[checkpoint_idx]
    volume_surge = current_vol / max(entry_vol, 1)

    # Spread behavior
    spread_vals = trade["spread_vals"]
    spread_current = spread_vals[checkpoint_idx]
    spread_entry = spread_vals[0]
    spread_change = spread_current - spread_entry

    # Signed volume alignment
    sv_vals = trade["sv_vals"]
    recent_sv = sv_vals[lookback_start:checkpoint_idx + 1]
    sv_aligned = float(np.sign(recent_sv.sum()) == direction)

    # Price momentum since entry
    prices = trade["prices"]
    entry_price = trade["entry_price"]
    current_price = prices[checkpoint_idx]
    price_momentum = (current_price - entry_price) / entry_price * direction

    # Close position in recent range
    if checkpoint_idx >= 5:
        recent_prices = prices[lookback_start:checkpoint_idx + 1]
        price_range = recent_prices.max() - recent_prices.min()
        if price_range > 0:
            close_position = (current_price - recent_prices.min()) / price_range
        else:
            close_position = 0.5
    else:
        close_position = 0.5

    # pnl_velocity — v5: change over last 2 management checkpoints
    if prev_checkpoint_idx > 0 and prev_checkpoint_idx < checkpoint_idx:
        pnl_velocity = (pnl_path[checkpoint_idx] - pnl_path[prev_checkpoint_idx]) / \
                        max(checkpoint_idx - prev_checkpoint_idx, 1)
    elif checkpoint_idx >= 5:
        pnl_velocity = (pnl_path[checkpoint_idx] - pnl_path[checkpoint_idx - 5]) / 5.0
    elif checkpoint_idx > 0:
        pnl_velocity = (pnl_path[checkpoint_idx] - pnl_path[0]) / max(checkpoint_idx, 1)
    else:
        pnl_velocity = 0.0

    # Drawdown from MFE
    drawdown_from_mfe = mfe_so_far - unrealized_pnl

    # Normalized time (0 = entry, 1 = max hold)
    time_normalized = time_in_trade / MAX_HOLD_MINUTES

    features = {
        "unrealized_pnl": float(unrealized_pnl),
        "time_in_trade": float(time_in_trade),
        "time_normalized": float(time_normalized),
        # v5 MFE/MAE features
        "mfe_ticks": float(mfe_so_far),
        "mae_ticks": float(mae_so_far),
        "mfe_mae_ratio": float(mfe_mae_ratio),
        "bars_since_mfe": float(bars_since_mfe_mgmt),
        "drawdown_from_mfe": float(drawdown_from_mfe),
        # v5 velocity
        "pnl_velocity": float(pnl_velocity),
        # OFI features
        "ofi_direction_aligned": float(ofi_direction_aligned),
        "ofi_momentum": float(ofi_momentum),
        "ofi_reversal": float(ofi_reversal),
        # Volume
        "volume_trend": float(volume_trend),
        "volume_surge": float(volume_surge),
        # Spread
        "spread_current": float(spread_current),
        "spread_change": float(spread_change),
        # Directional
        "sv_aligned": float(sv_aligned),
        "price_momentum": float(price_momentum),
        "close_position": float(close_position),
        # Entry context (static per trade)
        "entry_confluence_score": float(trade["confluence_score"]),
    }

    return features


MGMT_FEATURE_NAMES = [
    "unrealized_pnl", "time_in_trade", "time_normalized",
    "mfe_ticks", "mae_ticks", "mfe_mae_ratio",
    "bars_since_mfe", "drawdown_from_mfe",
    "pnl_velocity",
    "ofi_direction_aligned", "ofi_momentum", "ofi_reversal",
    "volume_trend", "volume_surge",
    "spread_current", "spread_change",
    "sv_aligned", "price_momentum", "close_position",
    "entry_confluence_score",
]


def build_management_dataset(
    trades: List[Dict],
    checkpoint_interval: int = MGMT_BAR_SIZE_MIN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[int]]:
    """
    Build the management model training dataset.

    For each trade, at every checkpoint_interval minutes:
      - Extract management features
      - Compute label: will_improve = (best future P&L > current + threshold)
    """
    X_rows = []
    y_rows = []
    date_rows = []
    trade_idx_rows = []

    for t_idx, trade in enumerate(trades):
        n_minutes = trade["n_minutes"]
        pnl_path = trade["pnl_path"]

        prev_ckpt = -1
        # Generate checkpoints every checkpoint_interval minutes
        for ckpt in range(checkpoint_interval, n_minutes - 1, checkpoint_interval):
            features = extract_management_features(trade, ckpt, prev_checkpoint_idx=prev_ckpt)
            if features is None:
                continue

            # Label: will holding longer improve by at least IMPROVE_THRESHOLD_TICKS?
            current_pnl = pnl_path[ckpt]
            remaining_path = pnl_path[ckpt:]
            best_remaining = remaining_path.max()

            will_improve = float(best_remaining > current_pnl + IMPROVE_THRESHOLD_TICKS)

            feat_vals = [features[fn] for fn in MGMT_FEATURE_NAMES]
            X_rows.append(feat_vals)
            y_rows.append(will_improve)
            date_rows.append(trade["date"])
            trade_idx_rows.append(t_idx)

            prev_ckpt = ckpt

    if not X_rows:
        return np.array([]), np.array([]), np.array([]), []

    X = np.array(X_rows, dtype=np.float32)
    y = np.array(y_rows, dtype=np.float32)
    dates = np.array(date_rows)
    trade_indices = trade_idx_rows

    pos_rate = y.mean()
    log.info(f"Management dataset: {len(X)} samples from {len(trades)} trades, "
             f"improve_rate={pos_rate:.1%}")

    return X, y, dates, trade_indices


# ═══════════════════════════════════════════════════════════════════
#  SECTION 12: MANAGEMENT MODEL TRAINING (Layer 2)
# ═══════════════════════════════════════════════════════════════════


def train_management_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    fold_idx: int,
) -> Tuple[Any, float]:
    """Train a management LightGBM binary classifier."""
    _import_lightgbm()

    if len(X_train) < 50 or len(X_val) < 10:
        log.warning(f"  Mgmt fold {fold_idx}: too few samples "
                    f"(train={len(X_train)}, val={len(X_val)}) -- skip")
        return None, 0.0

    params = {**LGBM_MGMT_PARAMS, "seed": 200 + fold_idx}

    train_data = lgb.Dataset(X_train, label=y_train, feature_name=MGMT_FEATURE_NAMES)
    val_data = lgb.Dataset(X_val, label=y_val, feature_name=MGMT_FEATURE_NAMES,
                           reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=20, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params, train_data, num_boost_round=300,
        valid_sets=[val_data], callbacks=callbacks,
    )

    val_pred = model.predict(X_val, num_iteration=model.best_iteration)
    from sklearn.metrics import roc_auc_score
    try:
        auc = roc_auc_score(y_val, val_pred)
    except ValueError:
        auc = 0.5

    log.info(f"  Mgmt fold {fold_idx}: AUC={auc:.4f}, best_iter={model.best_iteration}")
    return model, auc


# ═══════════════════════════════════════════════════════════════════
#  SECTION 13: ADAPTIVE TRAILING STOP LOGIC (v5 core innovation)
# ═══════════════════════════════════════════════════════════════════


def compute_trailing_stop_floor(mfe_ticks: float) -> Optional[float]:
    """
    Compute trailing stop floor based on MFE progression.

    Rules:
      - MFE > 6 ticks: floor = MFE - 1.5 ticks (lock in profits aggressively)
      - MFE > 3 ticks: floor = MFE - 2.0 ticks (lock in some profit)
      - MFE <= 3 ticks: no trailing floor (regular stop applies)

    Returns the floor P&L in ticks, or None if no trailing stop active.
    """
    if mfe_ticks >= TRAILING_MFE_TIER2:
        return mfe_ticks - TRAILING_OFFSET_TIER2
    elif mfe_ticks >= TRAILING_MFE_TIER1:
        return mfe_ticks - TRAILING_OFFSET_TIER1
    else:
        return None


def should_exit_trailing(
    mfe_ticks: float,
    current_pnl: float,
    bars_since_mfe: float,
) -> Tuple[bool, str]:
    """
    Determine if the adaptive trailing stop triggers an exit.

    Returns (should_exit, reason).
    """
    # Rule: stale trade — no new MFE for 4 management bars (20 minutes)
    if bars_since_mfe >= BARS_SINCE_MFE_EXIT and mfe_ticks > 1.0:
        return True, "stale_mfe"

    # Rule: trailing stop floor breached
    floor = compute_trailing_stop_floor(mfe_ticks)
    if floor is not None and current_pnl < floor:
        return True, f"trailing_floor_{floor:.1f}"

    return False, ""


# ═══════════════════════════════════════════════════════════════════
#  SECTION 14: TRADE SIMULATION — 4 EXIT STRATEGIES
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
        "daily_pnl": day_pnl.to_dict("records"),
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


def simulate_static(trades: List[Dict], hold_minutes: int = STATIC_HOLD_MINUTES) -> Dict:
    """STATIC_30: fixed hold baseline."""
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


def simulate_trailing_only(trades: List[Dict]) -> Dict:
    """TRAILING_ONLY: adaptive trailing stop, no ML model."""
    pnls, dates, dirs, hold_durations = [], [], [], []
    exit_reasons = {"trailing_floor": 0, "stale_mfe": 0, "static_hold": 0, "max_hold": 0}

    for trade in trades:
        pnl_path = trade["pnl_path"]
        mfe_path = trade["mfe_path"]
        n_minutes = trade["n_minutes"]
        exited = False
        exit_minute = min(STATIC_HOLD_MINUTES, n_minutes - 1)

        prev_ckpt = 0
        for ckpt in range(MGMT_BAR_SIZE_MIN, n_minutes - 1, MGMT_BAR_SIZE_MIN):
            mfe_now = mfe_path[ckpt]
            pnl_now = pnl_path[ckpt]

            # bars_since_mfe: how many mgmt bars since MFE was last extended
            bars_since = 0
            for lb in range(1, ckpt + 1):
                if mfe_path[ckpt - lb] < mfe_now:
                    break
                bars_since += 1
            bars_since_mgmt = bars_since / max(MGMT_BAR_SIZE_MIN, 1)

            should_exit, reason = should_exit_trailing(mfe_now, pnl_now, bars_since_mgmt)

            if should_exit:
                exit_minute = ckpt
                exited = True
                if "trailing_floor" in reason:
                    exit_reasons["trailing_floor"] += 1
                elif reason == "stale_mfe":
                    exit_reasons["stale_mfe"] += 1
                break

            # If past static hold and trailing stop hasn't triggered, allow extension
            # up to MAX_HOLD_MINUTES (trailing stop protects us)
            if ckpt >= MAX_HOLD_MINUTES:
                exit_minute = ckpt
                exited = True
                exit_reasons["max_hold"] += 1
                break

            prev_ckpt = ckpt

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

    return _compute_trade_stats(
        np.array(pnls), np.array(dates), np.array(dirs),
        "TRAILING_ONLY", hold_durations=hold_durations, exit_reasons=exit_reasons,
    )


def simulate_ml_only(
    trades: List[Dict],
    mgmt_models: Dict[str, Any],
    exit_threshold: float = 0.35,
    confirm_threshold: float = 0.70,
) -> Dict:
    """ML_ONLY: management model decides, no trailing stop."""
    pnls, dates, dirs, hold_durations = [], [], [], []
    exit_reasons = {"early_exit": 0, "static_hold": 0, "extended_hold": 0, "max_hold": 0}

    all_model_dates = sorted(mgmt_models.keys())

    for trade in trades:
        date_str = trade["date"]
        pnl_path = trade["pnl_path"]
        n_minutes = trade["n_minutes"]

        # Find the most recent model trained before this trade date
        model = None
        for model_date in reversed(all_model_dates):
            if model_date <= date_str:
                model = mgmt_models[model_date]
                break

        if model is None:
            # No model available — fall back to static hold
            if "pnl_30min" in trade:
                pnls.append(trade["pnl_30min"])
                dates.append(date_str)
                dirs.append(trade["direction"])
                exit_reasons["static_hold"] += 1
                hold_durations.append(STATIC_HOLD_MINUTES)
            continue

        confirmed = False
        exited = False
        exit_minute = STATIC_HOLD_MINUTES

        prev_ckpt = 0
        for ckpt in range(MGMT_BAR_SIZE_MIN, n_minutes - 1, MGMT_BAR_SIZE_MIN):
            features = extract_management_features(trade, ckpt, prev_checkpoint_idx=prev_ckpt)
            if features is None:
                prev_ckpt = ckpt
                continue

            feat_arr = np.array([[features[fn] for fn in MGMT_FEATURE_NAMES]], dtype=np.float32)
            feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=3.0, neginf=-3.0)

            prob_improve = model.predict(feat_arr, num_iteration=model.best_iteration)[0]

            if prob_improve < exit_threshold:
                exit_minute = ckpt
                exited = True
                exit_reasons["early_exit"] += 1
                break
            elif prob_improve > confirm_threshold:
                confirmed = True

            if ckpt >= STATIC_HOLD_MINUTES and not confirmed:
                exit_minute = STATIC_HOLD_MINUTES
                exit_reasons["static_hold"] += 1
                exited = True
                break

            prev_ckpt = ckpt

        if not exited:
            if confirmed:
                exit_minute = min(n_minutes - 1, MAX_HOLD_MINUTES)
                exit_reasons["extended_hold"] += 1
            else:
                exit_minute = min(STATIC_HOLD_MINUTES, n_minutes - 1)
                exit_reasons["static_hold"] += 1

        exit_minute = min(exit_minute, n_minutes - 1)
        trade_pnl = float(pnl_path[exit_minute]) - COST_RT_TICKS
        pnls.append(trade_pnl)
        dates.append(date_str)
        dirs.append(trade["direction"])
        hold_durations.append(exit_minute)

    if not pnls:
        return {"error": "no trades"}

    return _compute_trade_stats(
        np.array(pnls), np.array(dates), np.array(dirs),
        f"ML_ONLY_exit{int(exit_threshold*100)}_conf{int(confirm_threshold*100)}",
        hold_durations=hold_durations, exit_reasons=exit_reasons,
    )


def simulate_hybrid(
    trades: List[Dict],
    mgmt_models: Dict[str, Any],
    exit_threshold: float = 0.35,
    ml_override_threshold: float = ML_OVERRIDE_THRESHOLD,
) -> Dict:
    """
    HYBRID: trailing stop + ML override — the full system.

    Logic at each checkpoint:
      1. Compute trailing stop decision
      2. Compute ML model probability
      3. If trailing stop says EXIT:
         a. If ML says P(improve) > ml_override_threshold → HOLD (override)
         b. Else → EXIT
      4. If trailing stop says HOLD:
         a. If ML says P(improve) < exit_threshold → EXIT
         b. Else → HOLD
      5. If past static hold time, continue only if ML confirms or trailing allows
    """
    pnls, dates, dirs, hold_durations = [], [], [], []
    exit_reasons = {
        "trailing_exit": 0, "trailing_exit_ml_override": 0,
        "ml_exit": 0, "static_hold": 0, "extended_hold": 0, "max_hold": 0,
    }

    all_model_dates = sorted(mgmt_models.keys())

    for trade in trades:
        date_str = trade["date"]
        pnl_path = trade["pnl_path"]
        mfe_path = trade["mfe_path"]
        n_minutes = trade["n_minutes"]

        # Find the most recent model trained before this trade date
        model = None
        for model_date in reversed(all_model_dates):
            if model_date <= date_str:
                model = mgmt_models[model_date]
                break

        confirmed = False
        exited = False
        exit_minute = min(STATIC_HOLD_MINUTES, n_minutes - 1)

        prev_ckpt = 0
        for ckpt in range(MGMT_BAR_SIZE_MIN, n_minutes - 1, MGMT_BAR_SIZE_MIN):
            mfe_now = mfe_path[ckpt]
            pnl_now = pnl_path[ckpt]

            # bars_since_mfe
            bars_since = 0
            for lb in range(1, ckpt + 1):
                if mfe_path[ckpt - lb] < mfe_now:
                    break
                bars_since += 1
            bars_since_mgmt = bars_since / max(MGMT_BAR_SIZE_MIN, 1)

            trailing_exit, trailing_reason = should_exit_trailing(
                mfe_now, pnl_now, bars_since_mgmt,
            )

            # Get ML prediction if model available
            prob_improve = 0.5  # neutral if no model
            if model is not None:
                features = extract_management_features(trade, ckpt, prev_checkpoint_idx=prev_ckpt)
                if features is not None:
                    feat_arr = np.array(
                        [[features[fn] for fn in MGMT_FEATURE_NAMES]], dtype=np.float32,
                    )
                    feat_arr = np.nan_to_num(feat_arr, nan=0.0, posinf=3.0, neginf=-3.0)
                    prob_improve = model.predict(
                        feat_arr, num_iteration=model.best_iteration,
                    )[0]

            # Decision logic
            if trailing_exit:
                if prob_improve > ml_override_threshold:
                    # ML overrides trailing stop — continue holding
                    exit_reasons["trailing_exit_ml_override"] += 1
                    confirmed = True
                else:
                    # Trailing stop exits
                    exit_minute = ckpt
                    exited = True
                    exit_reasons["trailing_exit"] += 1
                    break
            else:
                # Trailing says hold — check ML
                if prob_improve < exit_threshold:
                    exit_minute = ckpt
                    exited = True
                    exit_reasons["ml_exit"] += 1
                    break
                elif prob_improve > ml_override_threshold:
                    confirmed = True

            # Past static hold — need confirmation to extend
            if ckpt >= STATIC_HOLD_MINUTES and not confirmed:
                exit_minute = STATIC_HOLD_MINUTES
                exit_reasons["static_hold"] += 1
                exited = True
                break

            # Hard cap at max hold
            if ckpt >= MAX_HOLD_MINUTES:
                exit_minute = ckpt
                exit_reasons["max_hold"] += 1
                exited = True
                break

            prev_ckpt = ckpt

        if not exited:
            if confirmed:
                exit_minute = min(n_minutes - 1, MAX_HOLD_MINUTES)
                exit_reasons["extended_hold"] += 1
            else:
                exit_minute = min(STATIC_HOLD_MINUTES, n_minutes - 1)
                exit_reasons["static_hold"] += 1

        exit_minute = min(exit_minute, n_minutes - 1)
        trade_pnl = float(pnl_path[exit_minute]) - COST_RT_TICKS
        pnls.append(trade_pnl)
        dates.append(date_str)
        dirs.append(trade["direction"])
        hold_durations.append(exit_minute)

    if not pnls:
        return {"error": "no trades"}

    return _compute_trade_stats(
        np.array(pnls), np.array(dates), np.array(dirs),
        f"HYBRID_exit{int(exit_threshold*100)}_override{int(ml_override_threshold*100)}",
        hold_durations=hold_durations, exit_reasons=exit_reasons,
    )


# ═══════════════════════════════════════════════════════════════════
#  SECTION 15: REGIME STRATIFICATION (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════


def regime_stratification_trades(
    pnl_arr: np.ndarray,
    dates: np.ndarray,
    day_returns: Dict[str, float],
) -> Dict[str, Any]:
    """
    Stratify trade results by day-regime (green/red/flat based on ES close-to-close).
    HC #428 R1: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
    """
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

    regime_arr = np.array([day_class.get(d, "flat") for d in dates])

    results = {}
    regime_sharpes = {}

    for regime in ["green", "red", "flat"]:
        mask = regime_arr == regime
        if mask.sum() < 5:
            results[regime] = {"n_trades": int(mask.sum()), "skip": True}
            continue

        pnl_r = pnl_arr[mask]
        sharpe = float(pnl_r.mean() / max(pnl_r.std(), 1e-6) * np.sqrt(252))
        wr = float(np.mean(pnl_r > 0))
        avg_pnl = float(pnl_r.mean())

        regime_sharpes[regime] = sharpe
        results[regime] = {
            "n_trades": int(mask.sum()),
            "sharpe": sharpe,
            "win_rate": wr,
            "avg_pnl_ticks": avg_pnl,
        }

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
#  SECTION 16: MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════


def run_trade_management_v5():
    """
    Run the full adaptive trade management experiment.

    Pipeline:
      1. Load minute bars, build 30-min and 15-min bar features
      2. Walk-forward train Level 1 entry models (30-min, 1h) with 40d window
      3. Compute confluence scores (gate at 2/6) and reconstruct trades
      4. Build management dataset from trade checkpoints (expanded features)
      5. Walk-forward train management model
      6. Simulate 4 exit strategies: STATIC_30, TRAILING_ONLY, ML_ONLY, HYBRID
      7. Sweep threshold variants for ML_ONLY and HYBRID
      8. Report with regime analysis, day-concentration, per-side breakdown
      9. Log to MLflow
    """
    _import_lightgbm()

    # ── MLflow setup ──
    mlflow_active = False
    mlflow_run = None
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("trade_management_v5_adaptive")
        mlflow_run = mlflow.start_run(
            run_name=f"tm_v5_adaptive_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        mlflow_active = True
        log.info(f"MLflow run started: {mlflow_run.info.run_id}")
    except Exception as e:
        log.warning(f"MLflow unavailable: {e} -- proceeding without tracking")

    t0 = time.time()

    # ════════════════════════════════════════
    #  STEP 1: Load raw data
    # ════════════════════════════════════════
    log.info("=" * 70)
    log.info("STEP 1: Loading raw minute bar data")
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
        mlflow.log_param("confluence_min", CONFLUENCE_MIN)
        mlflow.log_param("entry_confidence_pct", ENTRY_CONFIDENCE_PCT)
        mlflow.log_param("mgmt_checkpoint_min", MGMT_BAR_SIZE_MIN)
        mlflow.log_param("static_hold_min", STATIC_HOLD_MINUTES)
        mlflow.log_param("max_hold_min", MAX_HOLD_MINUTES)
        mlflow.log_param("trailing_tier1", TRAILING_MFE_TIER1)
        mlflow.log_param("trailing_tier2", TRAILING_MFE_TIER2)
        mlflow.log_param("trailing_offset1", TRAILING_OFFSET_TIER1)
        mlflow.log_param("trailing_offset2", TRAILING_OFFSET_TIER2)
        mlflow.log_param("bars_since_mfe_exit", BARS_SINCE_MFE_EXIT)
        mlflow.log_param("ml_override_threshold", ML_OVERRIDE_THRESHOLD)

    # ════════════════════════════════════════
    #  STEP 2: Build bar-level features
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 2: Building bar features for entry models")
    log.info("=" * 70)

    # 30-min bars for entry model + trade grid
    bars_30m = aggregate_to_bars(minute_df, bar_size_min=30)
    bars_30m = add_rolling_features(bars_30m, bar_size_min=30)
    bars_30m = add_forward_labels(bars_30m, horizon_bars=1, horizon_label="30min",
                                   min_edge_ticks=2.5)

    # 15-min bars for 1h model
    bars_15m = aggregate_to_bars(minute_df, bar_size_min=15)
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
    #  STEP 3: Walk-forward Level-1 entry models (40d train, 5d slide)
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 3: Walk-forward Level-1 entry model training (40d train)")
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

    # Walk-forward: accumulate OOT predictions
    l1_preds_30m = np.full(len(bars_30m), np.nan, dtype=np.float32)
    l1_preds_1h_on_15m = np.full(len(bars_15m), np.nan, dtype=np.float32)

    fold_idx = 0
    val_days_l1 = 5

    log.info(f"L1 Walk-Forward: {TRAIN_DAYS}d train, {val_days_l1}d val, {SLIDE_DAYS}d slide")
    log.info(f"Dates available: {len(dates_30m)} (30m), {len(dates_15m)} (15m)")
    log.info(f"Expected OOT days: ~{len(dates_30m) - TRAIN_DAYS} "
             f"(vs v4's ~{len(dates_30m) - 60} with 60d train)")

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

        if not leakage_audit(list(fold_train_dates), list(fold_val_dates), feature_cols_30m):
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

        _, _, val_p_30m = train_entry_model(
            tr_30m, labels_30m_all[train_mask_30m],
            vl_30m, labels_30m_all[val_mask_30m],
            feature_names=feature_cols_30m,
            fold_idx=fold_idx, horizon_label="30min",
        )
        l1_preds_30m[val_mask_30m] = val_p_30m

        # ── 1h model (15-min bars) ──
        train_mask_15m = np.isin(dates_15m_all, fold_train_dates)
        val_mask_15m = np.isin(dates_15m_all, fold_val_dates)

        if not leakage_audit(list(fold_train_dates), list(fold_val_dates), feature_cols_15m):
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

        _, _, val_p_1h = train_entry_model(
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

    if mlflow_active:
        mlflow.log_metric("l1_folds", fold_idx)
        mlflow.log_metric("l1_30m_valid_preds", int((~np.isnan(l1_preds_30m)).sum()))

    # ════════════════════════════════════════
    #  STEP 4: Align signals and compute confluence
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 4: Align signals to 30-min grid + compute confluence (gate=2/6)")
    log.info("=" * 70)

    # Map 1h predictions (15-min bars) -> 30-min bars
    bars_15m_mapped = bars_15m[["date", "ts"]].copy()
    bars_15m_mapped["pred_1h"] = l1_preds_1h_on_15m
    bars_15m_mapped["bar_30m_key"] = bars_15m_mapped["ts"].dt.floor("30min")

    pred_1h_per_30m = bars_15m_mapped.groupby(
        ["date", "bar_30m_key"]
    )["pred_1h"].mean().reset_index()
    pred_1h_per_30m.columns = ["date", "bar_key", "pred_1h_aligned"]

    bars_30m_merged = bars_30m.merge(pred_1h_per_30m, on=["date", "bar_key"], how="left")
    pred_1h_aligned = bars_30m_merged["pred_1h_aligned"].values.astype(np.float32)

    log.info(f"  1h predictions aligned to 30m grid: "
             f"{(~np.isnan(pred_1h_aligned)).sum()} / {len(pred_1h_aligned)}")

    # Map OFI exhaustion signal to 30-min bars
    ofi_ts = pd.to_datetime(ofi_signal_df["ts_minute"], utc=True)
    ofi_signal_df["bar_30m_key"] = ofi_ts.dt.floor("30min")
    ofi_per_30m = ofi_signal_df.groupby(["date", "bar_30m_key"]).agg(
        ofi_exhaust_max=("ofi_exhaust_signal",
                         lambda x: x[x.abs() == x.abs().max()].iloc[0] if len(x) > 0 else 0),
    ).reset_index()
    ofi_per_30m.columns = ["date", "bar_key", "ofi_exhaust_max"]

    bars_30m_merged = bars_30m_merged.merge(ofi_per_30m, on=["date", "bar_key"], how="left")
    ofi_aligned = bars_30m_merged["ofi_exhaust_max"].fillna(0).values.astype(np.float32)

    log.info(f"  OFI exhaustion signals in 30m grid: {(ofi_aligned != 0).sum()}")

    # Compute confluence scores
    confluence_score, confluence_direction, confluence_detail = compute_confluence_scores(
        l1_preds_30m, pred_1h_aligned, ofi_aligned, bars_30m,
    )

    if mlflow_active:
        for gate in [2, 3, 4, 5]:
            n_above = int((confluence_score >= gate).sum())
            mlflow.log_metric(f"n_confluence_ge{gate}", n_above)

    # ════════════════════════════════════════
    #  STEP 5: Reconstruct trades from entry signals (confluence >= 2)
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 5: Reconstruct trades (confluence >= 2, top/bottom 10%)")
    log.info("=" * 70)

    all_trades = reconstruct_trades(
        bars_30m, minute_df, l1_preds_30m,
        confluence_score, confluence_direction,
        min_confluence=CONFLUENCE_MIN,
        confidence_pct=ENTRY_CONFIDENCE_PCT,
    )

    if not all_trades:
        log.error("No trades reconstructed -- cannot proceed")
        if mlflow_active:
            mlflow.log_metric("n_trades", 0)
            mlflow.end_run(status="FAILED")
        return {}

    log.info(f"Total trades reconstructed: {len(all_trades)}")
    trade_dates = sorted(set(t["date"] for t in all_trades))
    log.info(f"Trading days covered: {len(trade_dates)} "
             f"({trade_dates[0]} -> {trade_dates[-1]})")

    if mlflow_active:
        mlflow.log_metric("n_trades_total", len(all_trades))
        mlflow.log_metric("n_trading_days_with_trades", len(trade_dates))

    # ════════════════════════════════════════
    #  STEP 5b: Static baselines
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 5b: Static hold baselines")
    log.info("=" * 70)

    static_results = {}
    for hold_min in [10, 15, 20, 25, 30, 45, 60]:
        result = simulate_static(all_trades, hold_minutes=hold_min)
        if "error" not in result:
            static_results[f"STATIC_{hold_min}"] = result
            log.info(
                f"  Static {hold_min:2d}min: Sharpe={result['sharpe']:.2f}, "
                f"Sortino={result['sortino']:.2f}, "
                f"WR={result['win_rate']:.1%}, "
                f"PF={result['profit_factor']:.2f}, "
                f"N={result['n_trades']}, "
                f"$PnL={result['total_pnl_dollars']:.0f}"
            )

            if mlflow_active:
                mlflow.log_metric(f"static_{hold_min}min_sharpe", result["sharpe"])
                mlflow.log_metric(f"static_{hold_min}min_sortino", result["sortino"])

    # ════════════════════════════════════════
    #  STEP 5c: TRAILING_ONLY baseline (no ML needed)
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 5c: TRAILING_ONLY baseline (adaptive trailing stop, no ML)")
    log.info("=" * 70)

    trailing_result = simulate_trailing_only(all_trades)
    if "error" not in trailing_result:
        log.info(
            f"  TRAILING_ONLY: Sharpe={trailing_result['sharpe']:.2f}, "
            f"Sortino={trailing_result['sortino']:.2f}, "
            f"WR={trailing_result['win_rate']:.1%}, "
            f"PF={trailing_result['profit_factor']:.2f}, "
            f"N={trailing_result['n_trades']}, "
            f"$PnL={trailing_result['total_pnl_dollars']:.0f}, "
            f"AvgHold={trailing_result.get('avg_hold_minutes', 0):.1f}min"
        )
        log.info(f"  Exit reasons: {trailing_result.get('exit_reasons', {})}")

        if mlflow_active:
            mlflow.log_metric("trailing_only_sharpe", trailing_result["sharpe"])
            mlflow.log_metric("trailing_only_sortino", trailing_result["sortino"])
            mlflow.log_metric("trailing_only_wr", trailing_result["win_rate"])

    # ════════════════════════════════════════
    #  STEP 6: Build management dataset + walk-forward train
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 6: Management model — dataset construction + walk-forward training")
    log.info("=" * 70)

    mgmt_X, mgmt_y, mgmt_dates, mgmt_trade_indices = build_management_dataset(
        all_trades, checkpoint_interval=MGMT_BAR_SIZE_MIN,
    )

    if len(mgmt_X) == 0:
        log.error("Empty management dataset -- only static/trailing results available")
        all_results = {
            "static": static_results,
            "trailing_only": trailing_result,
        }
        if mlflow_active:
            mlflow.log_metric("mgmt_dataset_size", 0)
            mlflow.end_run(status="FAILED")
        return all_results

    log.info(f"Management dataset: {mgmt_X.shape[0]} samples, {mgmt_X.shape[1]} features")

    if mlflow_active:
        mlflow.log_metric("mgmt_dataset_size", len(mgmt_X))
        mlflow.log_metric("mgmt_improve_rate", float(mgmt_y.mean()))

    # Walk-forward train management models
    unique_mgmt_dates = sorted(set(mgmt_dates))
    log.info(f"Management model dates: {len(unique_mgmt_dates)} "
             f"({unique_mgmt_dates[0]} -> {unique_mgmt_dates[-1]})")

    mgmt_models = {}
    mgmt_fold_idx = 0
    mgmt_aucs = []

    for fold_start in range(MGMT_TRAIN_DAYS, len(unique_mgmt_dates) - 5 + 1, MGMT_SLIDE_DAYS):
        fold_train_dates = unique_mgmt_dates[fold_start - MGMT_TRAIN_DAYS: fold_start]
        fold_val_dates = unique_mgmt_dates[fold_start: fold_start + 5]

        if len(fold_val_dates) < 5:
            break

        if not leakage_audit(fold_train_dates, fold_val_dates, MGMT_FEATURE_NAMES):
            continue

        mgmt_fold_idx += 1

        tr_mask = np.isin(mgmt_dates, fold_train_dates)
        vl_mask = np.isin(mgmt_dates, fold_val_dates)

        if tr_mask.sum() < 50 or vl_mask.sum() < 10:
            log.warning(f"  Mgmt fold {mgmt_fold_idx}: too few samples "
                        f"(train={tr_mask.sum()}, val={vl_mask.sum()}) -- skip")
            continue

        X_tr = mgmt_X[tr_mask]
        y_tr = mgmt_y[tr_mask]
        X_vl = mgmt_X[vl_mask]
        y_vl = mgmt_y[vl_mask]

        model, auc = train_management_model(X_tr, y_tr, X_vl, y_vl, mgmt_fold_idx)

        if model is not None:
            model_key = fold_val_dates[0]
            mgmt_models[model_key] = model
            mgmt_aucs.append(auc)

        if mgmt_fold_idx % 5 == 0:
            log.info(f"  Progress: mgmt fold {mgmt_fold_idx}, models={len(mgmt_models)}")

    log.info(f"\nManagement model training complete: {mgmt_fold_idx} folds, "
             f"{len(mgmt_models)} models saved")
    if mgmt_aucs:
        log.info(f"  Mean AUC: {np.mean(mgmt_aucs):.4f} "
                 f"(min={min(mgmt_aucs):.4f}, max={max(mgmt_aucs):.4f})")

    if mlflow_active:
        mlflow.log_metric("mgmt_n_folds", mgmt_fold_idx)
        mlflow.log_metric("mgmt_n_models", len(mgmt_models))
        if mgmt_aucs:
            mlflow.log_metric("mgmt_mean_auc", float(np.mean(mgmt_aucs)))
            mlflow.log_metric("mgmt_max_auc", float(max(mgmt_aucs)))

    # Feature importance from last management model
    if mgmt_models:
        last_model = list(mgmt_models.values())[-1]
        importance = last_model.feature_importance(importance_type="gain")
        feat_imp = sorted(zip(MGMT_FEATURE_NAMES, importance), key=lambda x: x[1], reverse=True)
        log.info("\n  Management model feature importance (last fold):")
        for name, gain in feat_imp:
            log.info(f"    {name:25s} gain={gain:.1f}")

        if mlflow_active:
            for name, gain in feat_imp:
                mlflow.log_metric(f"mgmt_importance_{name}", float(gain))

    # ════════════════════════════════════════
    #  STEP 7: Simulate all 4 exit strategies head-to-head
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 7: Exit strategy comparison — STATIC vs TRAILING vs ML vs HYBRID")
    log.info("=" * 70)

    all_results = {
        "static": static_results,
        "trailing_only": trailing_result,
    }

    if mgmt_models:
        # ML_ONLY — sweep thresholds
        log.info("\n  ML_ONLY variants:")
        ml_best_sharpe = -999
        ml_best_key = None
        for exit_th in [0.25, 0.30, 0.35, 0.40, 0.45]:
            for conf_th in [0.60, 0.65, 0.70, 0.75]:
                r = simulate_ml_only(
                    all_trades, mgmt_models,
                    exit_threshold=exit_th, confirm_threshold=conf_th,
                )
                if "error" not in r:
                    key = f"ML_exit{int(exit_th*100)}_conf{int(conf_th*100)}"
                    all_results[key] = r
                    if r["sharpe"] > ml_best_sharpe:
                        ml_best_sharpe = r["sharpe"]
                        ml_best_key = key
                    if exit_th == 0.35 and conf_th == 0.70:
                        log.info(
                            f"    ML_ONLY (default): Sharpe={r['sharpe']:.2f}, "
                            f"Sortino={r['sortino']:.2f}, WR={r['win_rate']:.1%}, "
                            f"N={r['n_trades']}, AvgHold={r.get('avg_hold_minutes', 0):.1f}min"
                        )

        if ml_best_key:
            r = all_results[ml_best_key]
            log.info(f"    Best ML_ONLY ({ml_best_key}): Sharpe={r['sharpe']:.2f}, "
                     f"Sortino={r['sortino']:.2f}")
            all_results["ml_only_best"] = {"key": ml_best_key, "result": r}

        # HYBRID — sweep thresholds
        log.info("\n  HYBRID variants:")
        hybrid_best_sharpe = -999
        hybrid_best_key = None
        for exit_th in [0.25, 0.30, 0.35, 0.40]:
            for override_th in [0.65, 0.70, 0.75, 0.80]:
                r = simulate_hybrid(
                    all_trades, mgmt_models,
                    exit_threshold=exit_th, ml_override_threshold=override_th,
                )
                if "error" not in r:
                    key = f"HYBRID_exit{int(exit_th*100)}_override{int(override_th*100)}"
                    all_results[key] = r
                    if r["sharpe"] > hybrid_best_sharpe:
                        hybrid_best_sharpe = r["sharpe"]
                        hybrid_best_key = key
                    if exit_th == 0.35 and override_th == 70:
                        log.info(
                            f"    HYBRID (default): Sharpe={r['sharpe']:.2f}, "
                            f"Sortino={r['sortino']:.2f}, WR={r['win_rate']:.1%}, "
                            f"N={r['n_trades']}, AvgHold={r.get('avg_hold_minutes', 0):.1f}min"
                        )

        if hybrid_best_key:
            r = all_results[hybrid_best_key]
            log.info(f"    Best HYBRID ({hybrid_best_key}): Sharpe={r['sharpe']:.2f}, "
                     f"Sortino={r['sortino']:.2f}")
            all_results["hybrid_best"] = {"key": hybrid_best_key, "result": r}
    else:
        log.warning("No management models trained -- only STATIC and TRAILING results available")

    # ════════════════════════════════════════
    #  STEP 8: Final comparison table
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("FINAL COMPARISON TABLE")
    log.info("=" * 70)
    log.info(f"{'Strategy':45s} {'Sharpe':>8} {'Sortino':>8} {'WR':>6} {'PF':>6} "
             f"{'N':>6} {'AvgPnL':>8} {'$Total':>10} {'AvgHold':>8}")
    log.info("-" * 115)

    # Static baselines
    for hold_min in [10, 15, 20, 25, 30, 45, 60]:
        key = f"STATIC_{hold_min}"
        if key in static_results:
            r = static_results[key]
            log.info(
                f"{key:45s} "
                f"{r['sharpe']:8.2f} {r['sortino']:8.2f} "
                f"{r['win_rate']:6.1%} {r['profit_factor']:6.2f} "
                f"{r['n_trades']:6d} {r['avg_pnl_ticks']:8.3f} "
                f"{r['total_pnl_dollars']:10.0f} {hold_min:>8d}"
            )

    # Trailing only
    if "error" not in trailing_result:
        r = trailing_result
        log.info(
            f"{'TRAILING_ONLY':45s} "
            f"{r['sharpe']:8.2f} {r['sortino']:8.2f} "
            f"{r['win_rate']:6.1%} {r['profit_factor']:6.2f} "
            f"{r['n_trades']:6d} {r['avg_pnl_ticks']:8.3f} "
            f"{r['total_pnl_dollars']:10.0f} {r.get('avg_hold_minutes', 0):8.1f}"
        )

    # ML_ONLY default
    ml_default_key = "ML_exit35_conf70"
    if ml_default_key in all_results and "error" not in all_results[ml_default_key]:
        r = all_results[ml_default_key]
        log.info(
            f"{'ML_ONLY (exit=35, conf=70)':45s} "
            f"{r['sharpe']:8.2f} {r['sortino']:8.2f} "
            f"{r['win_rate']:6.1%} {r['profit_factor']:6.2f} "
            f"{r['n_trades']:6d} {r['avg_pnl_ticks']:8.3f} "
            f"{r['total_pnl_dollars']:10.0f} {r.get('avg_hold_minutes', 0):8.1f}"
        )

    # Best ML
    if "ml_only_best" in all_results:
        bv = all_results["ml_only_best"]
        r = bv["result"]
        label = f"ML_ONLY BEST ({bv['key']})"
        log.info(
            f"{label:45s} "
            f"{r['sharpe']:8.2f} {r['sortino']:8.2f} "
            f"{r['win_rate']:6.1%} {r['profit_factor']:6.2f} "
            f"{r['n_trades']:6d} {r['avg_pnl_ticks']:8.3f} "
            f"{r['total_pnl_dollars']:10.0f} {r.get('avg_hold_minutes', 0):8.1f}"
        )

    # HYBRID default
    hybrid_default_key = "HYBRID_exit35_override70"
    if hybrid_default_key in all_results and "error" not in all_results[hybrid_default_key]:
        r = all_results[hybrid_default_key]
        log.info(
            f"{'HYBRID (exit=35, override=70)':45s} "
            f"{r['sharpe']:8.2f} {r['sortino']:8.2f} "
            f"{r['win_rate']:6.1%} {r['profit_factor']:6.2f} "
            f"{r['n_trades']:6d} {r['avg_pnl_ticks']:8.3f} "
            f"{r['total_pnl_dollars']:10.0f} {r.get('avg_hold_minutes', 0):8.1f}"
        )

    # Best HYBRID
    if "hybrid_best" in all_results:
        bv = all_results["hybrid_best"]
        r = bv["result"]
        label = f"HYBRID BEST ({bv['key']})"
        log.info(
            f"{label:45s} "
            f"{r['sharpe']:8.2f} {r['sortino']:8.2f} "
            f"{r['win_rate']:6.1%} {r['profit_factor']:6.2f} "
            f"{r['n_trades']:6d} {r['avg_pnl_ticks']:8.3f} "
            f"{r['total_pnl_dollars']:10.0f} {r.get('avg_hold_minutes', 0):8.1f}"
        )

    # ════════════════════════════════════════
    #  STEP 9: Regime analysis + day concentration for key strategies
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 9: Regime stratification (HC #428 R1) + day-concentration (HC #344)")
    log.info("=" * 70)

    # Build per-trade PnL for static 30-min
    regime_strategies = {}
    if "STATIC_30" in static_results:
        pnls_static = np.array([t["pnl_30min"] for t in all_trades if "pnl_30min" in t])
        dates_static = np.array([t["date"] for t in all_trades if "pnl_30min" in t])
        regime_strategies["STATIC_30"] = (pnls_static, dates_static)

    # For trailing — we need to re-simulate to get per-trade arrays
    # (already have them from the trailing result)
    # Simplification: use daily PnL for regime analysis
    for strat_name, strat_result in [
        ("TRAILING_ONLY", trailing_result),
    ]:
        if "error" in strat_result:
            continue
        if "daily_pnl" in strat_result:
            dp = pd.DataFrame(strat_result["daily_pnl"])
            if not dp.empty:
                log.info(f"  {strat_name}: day_conc={strat_result.get('day_concentration', 0):.3f} "
                         f"{'PASS' if strat_result.get('day_concentration_pass', False) else 'FAIL'}")

    for strat_name, (pnl_arr, date_arr) in regime_strategies.items():
        regime = regime_stratification_trades(pnl_arr, date_arr, day_returns)
        all_results[f"{strat_name}_regime"] = regime

        if "regime_gap_detail" in regime:
            log.info(f"  {strat_name} regime: {regime['regime_gap_detail']}")
        for r_name in ["green", "red", "flat"]:
            if r_name in regime and "sharpe" in regime.get(r_name, {}):
                r_data = regime[r_name]
                log.info(f"    {r_name}: Sharpe={r_data['sharpe']:.2f}, "
                         f"WR={r_data.get('win_rate', 0):.1%}, "
                         f"N={r_data.get('n_trades', 0)}")

    # Regime for best strategies
    for best_key in ["ml_only_best", "hybrid_best"]:
        if best_key in all_results and "result" in all_results[best_key]:
            r = all_results[best_key]["result"]
            if "day_concentration" in r:
                log.info(f"  {best_key}: day_conc={r['day_concentration']:.3f} "
                         f"{'PASS' if r.get('day_concentration_pass', False) else 'FAIL'}")

    # MLflow logging for key strategies
    if mlflow_active:
        for strat_name in ["trailing_only"]:
            r = all_results.get(strat_name, {})
            if isinstance(r, dict) and "error" not in r and "sharpe" in r:
                mlflow.log_metric(f"{strat_name}_sharpe", r["sharpe"])
                mlflow.log_metric(f"{strat_name}_sortino", r["sortino"])
                mlflow.log_metric(f"{strat_name}_wr", r["win_rate"])
                mlflow.log_metric(f"{strat_name}_n_trades", r["n_trades"])

        for key in ["STATIC_30"]:
            if key in static_results:
                r = static_results[key]
                mlflow.log_metric(f"{key.lower()}_sharpe", r["sharpe"])
                mlflow.log_metric(f"{key.lower()}_pnl_dollars", r["total_pnl_dollars"])

        for best_key in ["ml_only_best", "hybrid_best"]:
            if best_key in all_results and "result" in all_results[best_key]:
                r = all_results[best_key]["result"]
                mlflow.log_metric(f"{best_key}_sharpe", r["sharpe"])
                mlflow.log_metric(f"{best_key}_sortino", r["sortino"])
                mlflow.log_metric(f"{best_key}_wr", r["win_rate"])
                mlflow.log_metric(f"{best_key}_pf", r["profit_factor"])
                mlflow.log_metric(f"{best_key}_n_trades", r["n_trades"])
                mlflow.log_metric(f"{best_key}_pnl_dollars", r["total_pnl_dollars"])
                mlflow.log_metric(f"{best_key}_avg_hold", r.get("avg_hold_minutes", 0))

    # ════════════════════════════════════════
    #  STEP 10: Per-day stats for best strategy
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 10: Per-day performance (best strategy)")
    log.info("=" * 70)

    # Find best strategy
    best_strat_name = "STATIC_30"
    best_sharpe = static_results.get("STATIC_30", {}).get("sharpe", -999)

    candidates = [
        ("TRAILING_ONLY", trailing_result),
    ]
    if ml_default_key in all_results:
        candidates.append(("ML_ONLY", all_results[ml_default_key]))
    if hybrid_default_key in all_results:
        candidates.append(("HYBRID", all_results[hybrid_default_key]))
    if "ml_only_best" in all_results:
        candidates.append(("ML_BEST", all_results["ml_only_best"].get("result", {})))
    if "hybrid_best" in all_results:
        candidates.append(("HYBRID_BEST", all_results["hybrid_best"].get("result", {})))

    for name, r in candidates:
        if isinstance(r, dict) and "error" not in r:
            s = r.get("sharpe", -999)
            if s > best_sharpe:
                best_sharpe = s
                best_strat_name = name

    log.info(f"Best strategy: {best_strat_name} (Sharpe={best_sharpe:.2f})")

    # Find daily PnL from best strategy
    daily = pd.DataFrame()
    for name, r in candidates + [("STATIC_30", static_results.get("STATIC_30", {}))]:
        if name == best_strat_name and isinstance(r, dict) and "daily_pnl" in r:
            daily = pd.DataFrame(r["daily_pnl"])
            break

    if not daily.empty and "daily_pnl" in daily.columns:
        log.info(f"\n{'Date':>12s} {'PnL':>8s} {'Trades':>7s} {'Cum PnL':>10s}")
        log.info("-" * 40)
        cum = 0.0
        green_days = 0
        red_days = 0
        for _, row in daily.iterrows():
            cum += row["daily_pnl"]
            day_class = "+" if row["daily_pnl"] > 0 else ("-" if row["daily_pnl"] < 0 else " ")
            if row["daily_pnl"] > 0:
                green_days += 1
            elif row["daily_pnl"] < 0:
                red_days += 1
            log.info(
                f"{row['date']:>12s} {row['daily_pnl']:>+8.2f} {int(row['daily_trades']):>7d} "
                f"{cum:>+10.2f} {day_class}"
            )
        log.info(f"\nGreen days: {green_days}, Red days: {red_days}")
        log.info(f"Win rate (days): {green_days / max(green_days + red_days, 1):.1%}")

    # ════════════════════════════════════════
    #  STEP 11: Save results
    # ════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 11: Saving results")
    log.info("=" * 70)

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

    results_clean = clean_for_json(all_results)
    results_path = OUTPUT_DIR / "v5_adaptive_results.json"
    with open(str(results_path), "w") as f:
        json.dump(results_clean, f, indent=2, default=str)
    log.info(f"  Saved results to {results_path}")

    # Save predictions and confluence data
    np.savez_compressed(
        str(OUTPUT_DIR / "v5_predictions.npz"),
        pred_30m=l1_preds_30m,
        pred_1h_aligned=pred_1h_aligned,
        ofi_signal=ofi_aligned,
        confluence_score=confluence_score,
        confluence_direction=confluence_direction,
        labels_30m=labels_30m_all,
        dates=dates_30m_all,
    )
    log.info(f"  Saved predictions to {OUTPUT_DIR / 'v5_predictions.npz'}")

    if mlflow_active:
        try:
            mlflow.log_artifact(str(results_path))
        except Exception:
            pass

    elapsed = time.time() - t0
    log.info(f"\n{'=' * 70}")
    log.info(f"TRADE MANAGEMENT v5 ADAPTIVE COMPLETE in {elapsed/60:.1f} minutes")
    log.info(f"{'=' * 70}")

    # ── Executive Summary ──
    log.info("\n" + "=" * 70)
    log.info("EXECUTIVE SUMMARY")
    log.info("=" * 70)

    static_30_sharpe = static_results.get("STATIC_30", {}).get("sharpe", 0)
    trailing_sharpe = trailing_result.get("sharpe", 0) if "error" not in trailing_result else 0

    log.info(f"  Static 30-min baseline Sharpe:    {static_30_sharpe:.2f}")
    log.info(f"  Trailing-only Sharpe:             {trailing_sharpe:.2f}")

    if "ml_only_best" in all_results:
        bv = all_results["ml_only_best"]
        log.info(f"  Best ML-only Sharpe:              {bv['result']['sharpe']:.2f} "
                 f"({bv['key']})")

    if "hybrid_best" in all_results:
        bv = all_results["hybrid_best"]
        log.info(f"  Best Hybrid Sharpe:               {bv['result']['sharpe']:.2f} "
                 f"({bv['key']})")

    log.info(f"\n  Total trades: {len(all_trades)}")
    log.info(f"  Trading days: {len(trade_dates)}")
    log.info(f"  Expected improvement: 40d train gives {len(trade_dates)} OOT days "
             f"(v4 had 66 with 60d train)")

    if mlflow_active:
        mlflow.log_metric("total_runtime_min", elapsed / 60)
        mlflow.end_run()
        log.info(f"MLflow run ended: {mlflow_run.info.run_id}")

    return all_results


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    log.info("Trade Management v5 — Adaptive Trailing Stops + ML Hybrid")
    log.info(f"  ROOT: {ROOT}")
    log.info(f"  OUTPUT: {OUTPUT_DIR}")
    log.info(f"  Cost: {COST_RT_TICKS} ticks RT")
    log.info(f"  Walk-forward: {TRAIN_DAYS}d train, {SLIDE_DAYS}d slide (v4 was 60d)")
    log.info(f"  Confluence minimum: {CONFLUENCE_MIN} of 6 signals (v4 was 3)")
    log.info(f"  Management checkpoints: every {MGMT_BAR_SIZE_MIN} minutes")
    log.info(f"  Static hold: {STATIC_HOLD_MINUTES} min, max hold: {MAX_HOLD_MINUTES} min")
    log.info(f"  Trailing stop: tier1={TRAILING_MFE_TIER1}t offset={TRAILING_OFFSET_TIER1}t, "
             f"tier2={TRAILING_MFE_TIER2}t offset={TRAILING_OFFSET_TIER2}t")
    log.info(f"  Stale MFE exit: {BARS_SINCE_MFE_EXIT} bars ({BARS_SINCE_MFE_EXIT * MGMT_BAR_SIZE_MIN} min)")
    log.info(f"  ML override threshold: {ML_OVERRIDE_THRESHOLD}")
    log.info(f"  Exit strategies: STATIC_30, TRAILING_ONLY, ML_ONLY, HYBRID")

    try:
        results = run_trade_management_v5()
    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        log.error(traceback.format_exc())
        try:
            import mlflow
            mlflow.end_run(status="FAILED")
        except Exception:
            pass
        sys.exit(1)
