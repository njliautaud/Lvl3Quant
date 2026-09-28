#!/usr/bin/env python3
"""
Longer-Horizon Directional Model v4 — Focused Strategies
=========================================================

Lessons from v3 ensemble:
  - 1h LONG: Sharpe 1.39, WR 60.7% (275 trades) — STRONG
  - 1h SHORT: Sharpe 0.82 — drags combined down
  - 1h regime gap = 1.10 (FAILS HC #428) — long works green, short fails green
  - 15min: regime-agnostic but thin IC 0.025
  - GRU residual barely helped (alpha → 0.0) — drop it

Two focused strategies (pure LightGBM, no NN):

  Strategy A — Long-Only 1h Momentum
    Predict upward moves at 1h horizon. In hostile regimes, model outputs
    LOW confidence (don't trade) rather than shorting.

  Strategy B — 30-Minute Both-Sides
    New 30-min horizon that may capture MBO-driven moves the 1h aggregation
    washes out. Long + short with standard confidence gating.

Walk-Forward: 60d train, 10d val, slide 5d — SLIDING only (HC #0).
Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission).

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python \\
      alpha_discovery/longer_horizon_v4_focused.py --device cuda

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
OUTPUT_DIR = ROOT / "output" / "longer_horizon_v4_focused"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [LH-v4] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "longer_horizon_v4_focused.log")),
    ],
)
log = logging.getLogger("LH-v4")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376
COST_RT_TICKS = 2.376  # market entry + market exit + RT commission

# Min tick move to be a "real" trade label (must exceed cost)
MIN_EDGE_TICKS = {
    "30min": 2.5,
    "1h": 3.0,
}

# Bars per horizon (in terms of the bar_size used)
# Strategy A: 15-min bars, 1h = 4 bars
# Strategy B: 15-min bars, 30min = 2 bars  OR  30-min bars, 30min = 1 bar
HORIZON_CONFIG = {
    "strategy_a": {
        "name": "Long-Only 1h Momentum",
        "bar_size_min": 15,
        "horizon_label": "1h",
        "horizon_bars": 4,
        "long_only": True,
    },
    "strategy_b": {
        "name": "30-Minute Both-Sides",
        "bar_size_min": 30,
        "horizon_label": "30min",
        "horizon_bars": 1,
        "long_only": False,
    },
}

# LightGBM params — tuned from v3 experience
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


def aggregate_to_bars(minute_df: pd.DataFrame, bar_size_min: int = 15) -> pd.DataFrame:
    """
    Aggregate 1-minute bars into N-minute bars with microstructure features.

    Supports bar_size_min = 15 (Strategy A) or 30 (Strategy B).
    """
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
        # Require at least 3 minutes of data per bar
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


def add_queue_features_to_bars(
    bars_df: pd.DataFrame, queue_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Merge queue-augmented tick features into bars."""
    if queue_df is None or queue_df.empty:
        log.info("No queue features to merge")
        return bars_df

    queue_cols_mean = [
        "bid_qty_at_touch", "ask_qty_at_touch", "top_imbalance",
        "bid_q_ahead_p50", "ask_q_ahead_p50", "microprice_offset_ticks",
    ]
    queue_cols_sum = ["ofi_1s", "ofi_5s", "ofi_10s"]
    rate_cols = [
        "bid_add_rate_1s", "ask_add_rate_1s",
        "bid_cancel_rate_1s", "ask_cancel_rate_1s",
    ]

    available_mean = [c for c in queue_cols_mean if c in queue_df.columns]
    available_sum = [c for c in queue_cols_sum if c in queue_df.columns]
    available_rate = [c for c in rate_cols if c in queue_df.columns]

    if not available_mean and not available_sum and not available_rate:
        log.warning("Queue dataframe has no expected columns -- skipping")
        return bars_df

    ts_col = None
    for candidate in ["ts", "timestamp", "ts_event"]:
        if candidate in queue_df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        log.warning("Queue dataframe has no timestamp column -- skipping merge")
        return bars_df

    queue_df = queue_df.copy()
    queue_df[ts_col] = pd.to_datetime(queue_df[ts_col], utc=True)
    queue_df["bar_key"] = queue_df[ts_col].dt.floor(
        f"{bars_df.attrs.get('bar_size_min', 15)}min"
    )

    agg_dict = {}
    for c in available_mean:
        agg_dict[c] = ["mean", "std"]
    for c in available_sum:
        agg_dict[c] = ["sum"]
    for c in available_rate:
        agg_dict[c] = ["mean"]

    agg = queue_df.groupby(["date", "bar_key"]).agg(agg_dict)
    agg.columns = [f"q_{c}_{stat}" for c, stat in agg.columns]
    agg = agg.reset_index()

    if "bid_cancel_rate_1s" in queue_df.columns and "bid_add_rate_1s" in queue_df.columns:
        per_bar = queue_df.groupby(["date", "bar_key"]).agg(
            bid_cancel_mean=("bid_cancel_rate_1s", "mean"),
            bid_add_mean=("bid_add_rate_1s", "mean"),
            ask_cancel_mean=("ask_cancel_rate_1s", "mean"),
            ask_add_mean=("ask_add_rate_1s", "mean"),
        ).reset_index()
        per_bar["q_bid_toxicity"] = per_bar["bid_cancel_mean"] / per_bar["bid_add_mean"].clip(lower=1e-6)
        per_bar["q_ask_toxicity"] = per_bar["ask_cancel_mean"] / per_bar["ask_add_mean"].clip(lower=1e-6)
        agg = agg.merge(
            per_bar[["date", "bar_key", "q_bid_toxicity", "q_ask_toxicity"]],
            on=["date", "bar_key"], how="left",
        )

    n_before = len(bars_df)
    bars_df = bars_df.merge(agg, on=["date", "bar_key"], how="left")
    n_queue_cols = len(agg.columns) - 2
    log.info(f"Merged queue features: {n_queue_cols} new columns")
    return bars_df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: ROLLING CONTEXT + REGIME FEATURES
# ═══════════════════════════════════════════════════════════════════


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

    # Previous day context (causal: uses ONLY completed prior day)
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

    # Time-of-day encoding (sin/cos of session progress)
    session_start_hour = 13.5  # 13:30 UTC (09:30 ET)
    session_end_hour = 20.0    # 20:00 UTC (16:00 ET)
    session_len = session_end_hour - session_start_hour

    hour_frac = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    progress = ((hour_frac - session_start_hour) / session_len).clip(0, 1)
    df["tod_sin"] = np.sin(2 * np.pi * progress)
    df["tod_cos"] = np.cos(2 * np.pi * progress)
    df["tod_progress"] = progress.values

    df["bars_since_open"] = df.groupby("date").cumcount()

    # ── REGIME FEATURE ──
    # Trailing multi-day return as a regime indicator.
    # This is causal: uses close of PRIOR bars only.
    # Gives the model a sense of "are we in a green or red environment?"
    # Uses 4h lookback (16 bars for 15min, 8 bars for 30min)
    df["regime_ret_16bar"] = df["close"].pct_change(16)
    df["regime_ret_32bar"] = df["close"].pct_change(32)

    # Regime volatility (recent vs trailing)
    rvol_short = df["return_bar"].rolling(4, min_periods=2).std()
    rvol_long = df["return_bar"].rolling(16, min_periods=4).std()
    df["regime_vol_ratio"] = rvol_short / rvol_long.clip(lower=1e-8)

    # Intraday directional strength (how one-sided is today so far)
    df["intraday_direction_strength"] = (
        df["intraday_cum_ofi"].abs()
        / df.groupby("date")["ofi_sum"].transform(
            lambda x: x.abs().cumsum()
        ).clip(lower=1)
    )

    log.info(f"Added rolling + regime features: {len(df.columns)} total columns")
    return df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: FORWARD LABELS (STRICT NO-LEAKAGE)
# ═══════════════════════════════════════════════════════════════════


def add_forward_labels(
    df: pd.DataFrame,
    horizon_bars: int,
    horizon_label: str,
) -> pd.DataFrame:
    """Compute forward return labels with overnight gap protection."""
    df = df.sort_values("ts").reset_index(drop=True)

    fwd_close = df["close"].shift(-horizon_bars)
    fwd_return = fwd_close / df["close"] - 1
    fwd_ticks = (fwd_close - df["close"]) / 0.25

    # Null out overnight gaps (if gap > 6 hours, it's cross-day)
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

    # Directional label with minimum edge threshold
    min_ticks = MIN_EDGE_TICKS.get(horizon_label, 2.5)
    df[f"direction_{horizon_label}"] = 0
    df.loc[fwd_ticks > min_ticks, f"direction_{horizon_label}"] = 1
    df.loc[fwd_ticks < -min_ticks, f"direction_{horizon_label}"] = -1

    # Quality flag: move exceeds 2x cost
    df[f"trade_quality_{horizon_label}"] = (
        fwd_ticks.abs() > 2 * COST_RT_TICKS
    ).astype(np.float32)

    log.info(
        f"Forward labels ({horizon_label}): "
        f"{(~fwd_ticks.isna()).sum():,} valid, "
        f"{(df[f'direction_{horizon_label}'] == 1).sum():,} long signals, "
        f"{(df[f'direction_{horizon_label}'] == -1).sum():,} short signals"
    )
    return df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: LEAKAGE AUDIT
# ═══════════════════════════════════════════════════════════════════


def leakage_audit(
    df: pd.DataFrame,
    train_dates: List[str],
    val_dates: List[str],
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Explicit leakage audit. Returns dict of check results."""
    results = {}

    train_set = set(train_dates)
    val_set = set(val_dates)
    overlap = train_set & val_set
    results["date_overlap"] = len(overlap) == 0
    if overlap:
        log.error(f"LEAKAGE: Train/val date overlap: {overlap}")

    max_train = max(train_dates)
    min_val = min(val_dates)
    results["temporal_order"] = max_train < min_val
    if not results["temporal_order"]:
        log.error(f"LEAKAGE: Max train date {max_train} >= min val date {min_val}")

    fwd_leak = [c for c in feature_cols if c.startswith(("fwd_", "direction_", "trade_quality_"))]
    results["no_forward_features"] = len(fwd_leak) == 0
    if fwd_leak:
        log.error(f"LEAKAGE: Forward-looking columns in features: {fwd_leak}")

    price_cols = [c for c in feature_cols if c in ("close", "high", "low", "open")]
    results["no_raw_price_features"] = len(price_cols) == 0
    if price_cols:
        log.warning(f"WARNING: Raw price columns in features: {price_cols}")

    all_passed = all(results.values())
    results["all_passed"] = all_passed
    if all_passed:
        log.info("Leakage audit PASSED")
    else:
        log.error(f"Leakage audit FAILED: {results}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: FEATURE COLUMN SELECTION
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
#  SECTION 7: LightGBM TRAINING
# ═══════════════════════════════════════════════════════════════════


def train_lgbm_fold(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    feature_names: List[str],
    fold_idx: int,
    strategy_name: str,
    long_only: bool = False,
    sample_weights_train: Optional[np.ndarray] = None,
) -> Tuple[Any, np.ndarray, np.ndarray]:
    """
    Train LightGBM for one fold. Returns model, train preds, val preds.

    If long_only=True, we weight training samples so that:
    - Positive return samples: full weight
    - Negative return samples: reduced weight (model learns to predict
      magnitude of UP moves; for down moves, it outputs low values = low confidence)
    """
    _import_lightgbm()

    # Remove NaN targets
    train_valid = ~np.isnan(y_train)
    val_valid = ~np.isnan(y_val)

    if train_valid.sum() < 50 or val_valid.sum() < 10:
        log.warning(
            f"  LGBM {strategy_name} fold {fold_idx}: too few valid samples "
            f"(train={train_valid.sum()}, val={val_valid.sum()}) -- skip"
        )
        return None, np.full(len(y_train), np.nan), np.full(len(y_val), np.nan)

    X_tr = X_train[train_valid]
    y_tr = y_train[train_valid]
    X_v = X_val[val_valid]
    y_v = y_val[val_valid]

    # Build sample weights
    weights = None
    if long_only:
        # For long-only: upweight positive returns, downweight negative
        # Model learns: "how much will this go UP?" — low output = don't trade
        weights = np.ones(len(y_tr), dtype=np.float32)
        # Positive returns get full weight
        pos_mask = y_tr > 0
        neg_mask = y_tr <= 0
        # Reduce weight of negative samples by 3x — model should focus on
        # predicting upside magnitude accurately
        weights[neg_mask] = 0.33
        # Extra weight for large positive moves (the ones we want to catch)
        large_up = y_tr > np.percentile(y_tr[pos_mask], 75) if pos_mask.sum() > 10 else pos_mask
        weights[large_up] = 2.0
    elif sample_weights_train is not None:
        weights = sample_weights_train[train_valid]

    params = {**LGBM_PARAMS, "seed": 42 + fold_idx}

    train_data = lgb.Dataset(
        X_tr, label=y_tr, feature_name=feature_names,
        weight=weights,
    )
    val_data = lgb.Dataset(
        X_v, label=y_v, feature_name=feature_names,
        reference=train_data,
    )

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params,
        train_data,
        num_boost_round=500,
        valid_sets=[val_data],
        callbacks=callbacks,
    )

    # Predict on full arrays
    train_preds = np.full(len(y_train), np.nan)
    val_preds = np.full(len(y_val), np.nan)

    train_preds[train_valid] = model.predict(X_tr, num_iteration=model.best_iteration)
    val_preds[val_valid] = model.predict(X_v, num_iteration=model.best_iteration)

    # Quick IC check
    p_v = val_preds[val_valid]
    if len(p_v) > 5 and len(y_v) > 5:
        ic = np.corrcoef(p_v, y_v)[0, 1]
        log.info(
            f"  LGBM {strategy_name} fold {fold_idx}: IC={ic:.4f}, "
            f"best_iter={model.best_iteration}, n_train={len(y_tr)}, n_val={len(y_v)}"
        )
    else:
        log.info(f"  LGBM {strategy_name} fold {fold_idx}: too few samples for IC")

    return model, train_preds, val_preds


# ═══════════════════════════════════════════════════════════════════
#  SECTION 8: TRADE SIMULATION
# ═══════════════════════════════════════════════════════════════════


def simulate_trades(
    preds: np.ndarray,
    actuals: np.ndarray,
    dates: np.ndarray,
    confidence_pct: float = 0.10,
    cost_ticks: float = COST_RT_TICKS,
    long_only: bool = False,
) -> Optional[Dict]:
    """
    Simulate trades with per-side and per-day reporting.

    If long_only=True, only take long trades (top confidence_pct predictions).
    """
    valid = ~np.isnan(preds) & ~np.isnan(actuals)
    preds_v = preds[valid]
    actuals_v = actuals[valid]
    dates_v = dates[valid]

    if len(preds_v) < 20:
        return None

    trades = []
    if long_only:
        # Only go long on top N% predicted moves
        upper = np.quantile(preds_v, 1 - confidence_pct)
        for i in range(len(preds_v)):
            if preds_v[i] >= upper:
                pnl = actuals_v[i] - cost_ticks
                trades.append({
                    "dir": "long", "pnl": pnl, "raw": actuals_v[i],
                    "pred": preds_v[i], "date": dates_v[i],
                })
    else:
        # Long and short
        upper = np.quantile(preds_v, 1 - confidence_pct)
        lower = np.quantile(preds_v, confidence_pct)
        for i in range(len(preds_v)):
            if preds_v[i] >= upper:
                pnl = actuals_v[i] - cost_ticks
                trades.append({
                    "dir": "long", "pnl": pnl, "raw": actuals_v[i],
                    "pred": preds_v[i], "date": dates_v[i],
                })
            elif preds_v[i] <= lower:
                pnl = -actuals_v[i] - cost_ticks
                trades.append({
                    "dir": "short", "pnl": pnl, "raw": -actuals_v[i],
                    "pred": preds_v[i], "date": dates_v[i],
                })

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

    # Per-day PnL for drawdown analysis
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
        # Long side
        "long_trades": int(len(long_pnl)),
        "long_wr": float(np.mean(long_pnl > 0)) if len(long_pnl) > 0 else 0,
        "long_avg": float(long_pnl.mean()) if len(long_pnl) > 0 else 0,
        "long_sharpe": float(
            long_pnl.mean() / max(long_pnl.std(), 1e-6) * np.sqrt(252)
        ) if len(long_pnl) > 2 else 0,
        # Short side
        "short_trades": int(len(short_pnl)),
        "short_wr": float(np.mean(short_pnl > 0)) if len(short_pnl) > 0 else 0,
        "short_avg": float(short_pnl.mean()) if len(short_pnl) > 0 else 0,
        "short_sharpe": float(
            short_pnl.mean() / max(short_pnl.std(), 1e-6) * np.sqrt(252)
        ) if len(short_pnl) > 2 else 0,
        # Per-day analysis
        "daily_pnl": day_pnl.to_dict("records"),
        "n_trading_days": int(len(day_pnl)),
        "daily_sharpe": float(
            day_pnl["daily_pnl"].mean() / max(day_pnl["daily_pnl"].std(), 1e-6) * np.sqrt(252)
        ) if len(day_pnl) > 2 else 0,
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 9: REGIME STRATIFICATION (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════


def regime_stratification(
    preds: np.ndarray,
    actuals: np.ndarray,
    dates: np.ndarray,
    day_returns: Dict[str, float],
    confidence_pct: float = 0.10,
    long_only: bool = False,
) -> Dict[str, Any]:
    """
    Stratify predictions by day-regime (green/red/flat based on ES close-to-close).
    Returns per-regime metrics and the HC #428 R1 gap check.

    HC #428 R1: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
    """
    valid = ~np.isnan(preds) & ~np.isnan(actuals)
    preds_v = preds[valid]
    actuals_v = actuals[valid]
    dates_v = dates[valid]

    if len(preds_v) < 20:
        return {"error": "too few valid predictions", "regime_gap_pass": False}

    # Classify each day as green (>0.1%), red (<-0.1%), flat
    day_class = {}
    for d, ret in day_returns.items():
        if ret > 0.001:
            day_class[d] = "green"
        elif ret < -0.001:
            day_class[d] = "red"
        else:
            day_class[d] = "flat"

    # Build regime array
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

        # IC within regime
        ic = np.corrcoef(p_r, a_r)[0, 1] if len(p_r) > 5 else float("nan")

        # Simulate trades within regime
        sim = simulate_trades(p_r, a_r, d_r, confidence_pct=confidence_pct,
                              long_only=long_only)

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
            results[regime] = {
                "n_predictions": int(mask.sum()),
                "ic": float(ic),
                "n_trades": 0,
            }

    # HC #428 R1 gap check
    if "green" in regime_sharpes and "red" in regime_sharpes:
        s_green = regime_sharpes["green"]
        s_red = regime_sharpes["red"]
        denom = max(abs(s_green), abs(s_red), 1e-6)
        gap = abs(s_green - s_red) / denom
        results["regime_gap"] = float(gap)
        results["regime_gap_pass"] = gap <= 0.50
        results["regime_gap_detail"] = (
            f"green_sharpe={s_green:.2f}, red_sharpe={s_red:.2f}, "
            f"gap={gap:.2f} {'PASS' if gap <= 0.50 else 'FAIL'} (threshold 0.50)"
        )
    else:
        results["regime_gap"] = float("nan")
        results["regime_gap_pass"] = False
        results["regime_gap_detail"] = "insufficient regime data"

    return results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 10: WALK-FORWARD ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════


def run_strategy(
    minute_df: pd.DataFrame,
    queue_df: Optional[pd.DataFrame],
    strategy_key: str,
    args,
) -> Dict:
    """
    Run a single strategy through full walk-forward.
    Returns comprehensive results dict.
    """
    _import_lightgbm()

    cfg = HORIZON_CONFIG[strategy_key]
    strategy_name = cfg["name"]
    bar_size_min = cfg["bar_size_min"]
    horizon_label = cfg["horizon_label"]
    horizon_bars = cfg["horizon_bars"]
    long_only = cfg["long_only"]

    log.info(f"\n{'#' * 70}")
    log.info(f"# STRATEGY: {strategy_name}")
    log.info(f"# Bar size: {bar_size_min}min, Horizon: {horizon_label}, "
             f"Long-only: {long_only}")
    log.info(f"{'#' * 70}\n")

    # ── Step 1: Aggregate bars ──
    bars_df = aggregate_to_bars(minute_df, bar_size_min=bar_size_min)
    bars_df.attrs["bar_size_min"] = bar_size_min

    # ── Step 2: Add queue features ──
    bars_df = add_queue_features_to_bars(bars_df, queue_df)

    # ── Step 3: Rolling + regime features ──
    bars_df = add_rolling_features(bars_df)

    # ── Step 4: Forward labels ──
    bars_df = add_forward_labels(bars_df, horizon_bars, horizon_label)

    # ── Step 5: Feature selection ──
    feature_cols = get_feature_columns(bars_df)
    log.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:15]}...")

    # ── Step 6: Walk-forward ──
    dates = sorted(bars_df["date"].unique())
    log.info(f"Total trading days: {len(dates)} ({dates[0]} -> {dates[-1]})")

    train_days = args.train_days
    val_days = args.val_days
    slide = args.slide_days

    if len(dates) < train_days + val_days + slide:
        log.error(
            f"Not enough days ({len(dates)}) for {train_days}+{val_days} walk-forward"
        )
        return {"error": "insufficient data", "strategy": strategy_key}

    # Prepare arrays
    features_all = bars_df[feature_cols].values.astype(np.float32)
    labels_all = bars_df[f"fwd_ticks_{horizon_label}"].values.astype(np.float32)
    dates_all = bars_df["date"].values
    ts_all = bars_df["ts"].values

    # Compute per-day returns for regime classification
    day_close = bars_df.groupby("date")["close"].last()
    day_returns_raw = day_close.pct_change()
    day_returns = day_returns_raw.to_dict()

    # Accumulators
    all_oot_preds = []
    all_oot_actuals = []
    all_oot_dates = []
    all_fold_results = []
    fold_idx = 0

    start_idx = train_days
    for fold_start in range(start_idx, len(dates) - val_days + 1, slide):
        fold_train_dates = dates[fold_start - train_days: fold_start]
        fold_val_dates = dates[fold_start: fold_start + val_days]

        if len(fold_val_dates) < val_days:
            break

        fold_idx += 1
        log.info(
            f"\n{'=' * 60}\n"
            f"FOLD {fold_idx} [{strategy_name}]: "
            f"train {fold_train_dates[0]}->{fold_train_dates[-1]} ({len(fold_train_dates)}d), "
            f"val {fold_val_dates[0]}->{fold_val_dates[-1]} ({len(fold_val_dates)}d)"
            f"\n{'=' * 60}"
        )

        # ── Leakage audit ──
        audit = leakage_audit(
            bars_df,
            list(fold_train_dates),
            list(fold_val_dates),
            feature_cols,
        )
        if not audit["all_passed"]:
            log.error(f"FOLD {fold_idx}: Leakage audit FAILED -- skipping fold")
            continue

        # ── Split data ──
        train_mask = np.isin(dates_all, fold_train_dates)
        val_mask = np.isin(dates_all, fold_val_dates)

        train_features_raw = features_all[train_mask].copy()
        val_features_raw = features_all[val_mask].copy()

        # Robust scaling: median/IQR from TRAIN ONLY (no leakage)
        train_median = np.nanmedian(train_features_raw, axis=0)
        q75 = np.nanpercentile(train_features_raw, 75, axis=0)
        q25 = np.nanpercentile(train_features_raw, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0

        train_features = (train_features_raw - train_median) / iqr
        val_features = (val_features_raw - train_median) / iqr

        train_features = np.nan_to_num(train_features, nan=0.0, posinf=3.0, neginf=-3.0)
        val_features = np.nan_to_num(val_features, nan=0.0, posinf=3.0, neginf=-3.0)
        train_features = np.clip(train_features, -5, 5)
        val_features = np.clip(val_features, -5, 5)

        train_labels = labels_all[train_mask]
        val_labels = labels_all[val_mask]
        val_dates_fold = dates_all[val_mask]

        # ── Train LightGBM ──
        model, _, val_preds = train_lgbm_fold(
            train_features, train_labels,
            val_features, val_labels,
            feature_names=feature_cols,
            fold_idx=fold_idx,
            strategy_name=strategy_name,
            long_only=long_only,
        )

        if model is None:
            continue

        # ── Fold IC ──
        valid_v = ~np.isnan(val_labels) & ~np.isnan(val_preds)
        if valid_v.sum() > 5:
            ic = np.corrcoef(val_preds[valid_v], val_labels[valid_v])[0, 1]
            dir_acc = np.mean(
                np.sign(val_preds[valid_v]) == np.sign(val_labels[valid_v])
            )
        else:
            ic = float("nan")
            dir_acc = float("nan")

        fold_result = {
            "fold": fold_idx,
            "train_start": fold_train_dates[0],
            "train_end": fold_train_dates[-1],
            "val_start": fold_val_dates[0],
            "val_end": fold_val_dates[-1],
            "train_samples": int(train_mask.sum()),
            "val_samples": int(val_mask.sum()),
            "ic": float(ic) if not np.isnan(ic) else None,
            "dir_acc": float(dir_acc) if not np.isnan(dir_acc) else None,
            "lgbm_best_iter": model.best_iteration,
        }

        # Feature importance (top 10)
        importance = model.feature_importance(importance_type="gain")
        feat_imp = sorted(
            zip(feature_cols, importance), key=lambda x: x[1], reverse=True
        )[:10]
        fold_result["top_features"] = [
            {"name": n, "gain": float(g)} for n, g in feat_imp
        ]

        all_fold_results.append(fold_result)

        # ── Accumulate OOT predictions ──
        all_oot_preds.append(val_preds[valid_v])
        all_oot_actuals.append(val_labels[valid_v])
        all_oot_dates.append(val_dates_fold[valid_v])

        # Save fold predictions
        np.savez_compressed(
            str(OUTPUT_DIR / f"{strategy_key}_fold_{fold_idx:03d}.npz"),
            preds=val_preds,
            actuals=val_labels,
            dates=val_dates_fold,
            feature_cols=feature_cols,
        )

        log.info(
            f"  Fold {fold_idx} result: IC={ic:.4f}, dir_acc={dir_acc:.3f}, "
            f"n_valid={valid_v.sum()}"
        )

        gc.collect()

    # ── Concatenate OOT ──
    if not all_oot_preds:
        log.error(f"No valid folds for {strategy_name}")
        return {"error": "no valid folds", "strategy": strategy_key}

    concat_preds = np.concatenate(all_oot_preds)
    concat_actuals = np.concatenate(all_oot_actuals)
    concat_dates = np.concatenate(all_oot_dates)

    # Save concat OOT
    np.savez_compressed(
        str(OUTPUT_DIR / f"{strategy_key}_concat_oot.npz"),
        preds=concat_preds,
        actuals=concat_actuals,
        dates=concat_dates,
    )

    # ── Concat OOT metrics ──
    concat_ic = np.corrcoef(concat_preds, concat_actuals)[0, 1]
    fold_ics = [f["ic"] for f in all_fold_results if f["ic"] is not None]
    ic_sharpe = (
        np.mean(fold_ics) / max(np.std(fold_ics), 1e-6)
        if len(fold_ics) > 2 else float("nan")
    )
    concat_dir_acc = np.mean(np.sign(concat_preds) == np.sign(concat_actuals))

    log.info(
        f"\n{'=' * 60}\n"
        f"CONCAT OOT [{strategy_name}]: "
        f"IC={concat_ic:.4f}, IC_Sharpe={ic_sharpe:.3f}, "
        f"dir_acc={concat_dir_acc:.3f}, n={len(concat_preds):,}"
        f"\n{'=' * 60}"
    )

    # ── Trade simulation at multiple confidence thresholds ──
    trade_results = {}
    for conf_pct in [0.05, 0.10, 0.15, 0.20, 0.30]:
        sim = simulate_trades(
            concat_preds, concat_actuals, concat_dates,
            confidence_pct=conf_pct, long_only=long_only,
        )
        if sim is not None:
            trade_results[f"top_{int(conf_pct*100)}pct"] = sim
            log.info(
                f"  Trades (top {int(conf_pct*100)}%): "
                f"n={sim['n_trades']}, Sharpe={sim['sharpe']:.2f}, "
                f"Sortino={sim['sortino']:.2f}, WR={sim['win_rate']:.1%}, "
                f"PF={sim['profit_factor']:.2f}, avg_pnl={sim['avg_pnl_ticks']:.2f}t"
            )

    # ── Regime stratification (HC #428 R1) ──
    # Use top 10% as the primary threshold
    regime_results = regime_stratification(
        concat_preds, concat_actuals, concat_dates,
        day_returns=day_returns,
        confidence_pct=0.10,
        long_only=long_only,
    )

    log.info(f"\n  Regime analysis: {regime_results.get('regime_gap_detail', 'N/A')}")

    # ── Per-day IC analysis ──
    unique_dates = sorted(set(concat_dates))
    per_day_ics = []
    for d in unique_dates:
        d_mask = concat_dates == d
        if d_mask.sum() < 5:
            continue
        d_ic = np.corrcoef(concat_preds[d_mask], concat_actuals[d_mask])[0, 1]
        d_regime = "green" if day_returns.get(d, 0) > 0.001 else (
            "red" if day_returns.get(d, 0) < -0.001 else "flat"
        )
        per_day_ics.append({
            "date": d,
            "ic": float(d_ic) if not np.isnan(d_ic) else 0.0,
            "n_bars": int(d_mask.sum()),
            "regime": d_regime,
        })

    # Stratified IC by regime
    green_ics = [x["ic"] for x in per_day_ics if x["regime"] == "green"]
    red_ics = [x["ic"] for x in per_day_ics if x["regime"] == "red"]
    flat_ics = [x["ic"] for x in per_day_ics if x["regime"] == "flat"]

    log.info(
        f"  Per-day IC: green={np.mean(green_ics):.4f} ({len(green_ics)}d), "
        f"red={np.mean(red_ics):.4f} ({len(red_ics)}d), "
        f"flat={np.mean(flat_ics):.4f} ({len(flat_ics)}d)"
    )

    # ── Assemble final results ──
    summary = {
        "strategy": strategy_key,
        "strategy_name": strategy_name,
        "bar_size_min": bar_size_min,
        "horizon_label": horizon_label,
        "long_only": long_only,
        "n_folds": fold_idx,
        "n_valid_folds": len(all_fold_results),
        "concat_oot": {
            "ic": float(concat_ic),
            "ic_sharpe": float(ic_sharpe),
            "dir_acc": float(concat_dir_acc),
            "n_predictions": int(len(concat_preds)),
            "n_oot_days": len(unique_dates),
        },
        "trade_sims": trade_results,
        "regime": regime_results,
        "per_day_ic": per_day_ics,
        "per_day_ic_summary": {
            "green_mean": float(np.mean(green_ics)) if green_ics else None,
            "red_mean": float(np.mean(red_ics)) if red_ics else None,
            "flat_mean": float(np.mean(flat_ics)) if flat_ics else None,
            "green_n": len(green_ics),
            "red_n": len(red_ics),
            "flat_n": len(flat_ics),
        },
        "fold_results": all_fold_results,
        "feature_cols": feature_cols,
    }

    return summary


# ═══════════════════════════════════════════════════════════════════
#  SECTION 11: MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Longer Horizon v4 - Focused Strategies")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--train-days", type=int, default=60, help="Training window (days)")
    parser.add_argument("--val-days", type=int, default=10, help="Validation window (days)")
    parser.add_argument("--slide-days", type=int, default=5, help="Slide step (days)")
    parser.add_argument(
        "--strategies", nargs="+", default=["strategy_a", "strategy_b"],
        choices=["strategy_a", "strategy_b"],
        help="Which strategies to run",
    )
    parser.add_argument("--mlflow", action="store_true", default=True, help="Log to MLflow")
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow")
    parser.add_argument(
        "--mlflow-uri", default="http://neptune:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-name", default="longer_horizon_v4_focused",
        help="MLflow experiment name",
    )
    args = parser.parse_args()

    if args.no_mlflow:
        args.mlflow = False

    log.info("=" * 70)
    log.info("LONGER HORIZON v4 - FOCUSED STRATEGIES")
    log.info("=" * 70)
    log.info(f"Device: {args.device}")
    log.info(f"Walk-forward: {args.train_days}d train, {args.val_days}d val, slide {args.slide_days}d")
    log.info(f"Strategies: {args.strategies}")
    log.info(f"Cost: {COST_RT_TICKS} ticks RT")
    log.info(f"Output: {OUTPUT_DIR}")

    t0 = time.time()

    # ── Load data (shared across strategies) ──
    log.info("\n--- Loading data ---")
    minute_df = load_all_minute_bars()
    queue_df = load_queue_features()

    # ── MLflow setup ──
    mlflow_run = None
    if args.mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.experiment_name)
            mlflow_run = mlflow.start_run(
                run_name=f"lh_v4_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            mlflow.log_params({
                "model": "LightGBM_Focused",
                "train_days": args.train_days,
                "val_days": args.val_days,
                "slide_days": args.slide_days,
                "cost_rt_ticks": COST_RT_TICKS,
                "strategies": ",".join(args.strategies),
                "lgbm_num_leaves": LGBM_PARAMS["num_leaves"],
                "lgbm_lr": LGBM_PARAMS["learning_rate"],
                "lgbm_max_depth": LGBM_PARAMS["max_depth"],
            })
            log.info(f"MLflow run started: {mlflow_run.info.run_id}")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e} -- continuing without tracking")
            args.mlflow = False

    # ── Run strategies ──
    all_results = {}
    for strategy_key in args.strategies:
        try:
            result = run_strategy(minute_df, queue_df, strategy_key, args)
            all_results[strategy_key] = result

            # Log to MLflow
            if args.mlflow and "error" not in result:
                try:
                    import mlflow
                    prefix = strategy_key.replace("strategy_", "s")
                    oot = result["concat_oot"]
                    mlflow.log_metrics({
                        f"{prefix}_ic": oot["ic"],
                        f"{prefix}_ic_sharpe": oot["ic_sharpe"],
                        f"{prefix}_dir_acc": oot["dir_acc"],
                        f"{prefix}_n_predictions": oot["n_predictions"],
                        f"{prefix}_n_oot_days": oot["n_oot_days"],
                    })

                    # Log best trade sim
                    if result.get("trade_sims"):
                        best_key = max(
                            result["trade_sims"].keys(),
                            key=lambda k: result["trade_sims"][k].get("sharpe", -999),
                        )
                        best = result["trade_sims"][best_key]
                        mlflow.log_metrics({
                            f"{prefix}_best_sharpe": best["sharpe"],
                            f"{prefix}_best_sortino": best["sortino"],
                            f"{prefix}_best_wr": best["win_rate"],
                            f"{prefix}_best_pf": best["profit_factor"],
                            f"{prefix}_best_n_trades": best["n_trades"],
                            f"{prefix}_best_threshold": best_key,
                        })

                    # Regime gap
                    regime = result.get("regime", {})
                    if "regime_gap" in regime and not np.isnan(regime["regime_gap"]):
                        mlflow.log_metrics({
                            f"{prefix}_regime_gap": regime["regime_gap"],
                            f"{prefix}_regime_gap_pass": int(regime.get("regime_gap_pass", False)),
                        })
                except Exception as e:
                    log.warning(f"MLflow logging failed for {strategy_key}: {e}")

        except Exception as e:
            log.error(f"Strategy {strategy_key} FAILED: {e}", exc_info=True)
            all_results[strategy_key] = {"error": str(e), "strategy": strategy_key}

    # ── Save summary JSON ──
    # Strip non-serializable items for JSON
    summary_for_json = {}
    for k, v in all_results.items():
        # Deep copy and remove feature_cols list (too verbose)
        s = {kk: vv for kk, vv in v.items() if kk != "feature_cols"}
        summary_for_json[k] = s

    summary_path = OUTPUT_DIR / "v4_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary_for_json, f, indent=2, default=str)
    log.info(f"\nSummary saved to {summary_path}")

    # ── End MLflow ──
    if args.mlflow and mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_artifact(str(summary_path))
            mlflow.end_run()
        except Exception as e:
            log.warning(f"MLflow end_run failed: {e}")

    elapsed = time.time() - t0
    log.info(f"\nTotal runtime: {elapsed / 60:.1f} minutes")

    # ── Print final summary ──
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY")
    log.info("=" * 70)

    for strategy_key, result in all_results.items():
        if "error" in result:
            log.info(f"\n{strategy_key}: FAILED - {result['error']}")
            continue

        cfg = HORIZON_CONFIG[strategy_key]
        oot = result["concat_oot"]
        regime = result.get("regime", {})

        log.info(f"\n--- {cfg['name']} ---")
        log.info(f"  Concat IC: {oot['ic']:.4f}, IC_Sharpe: {oot['ic_sharpe']:.3f}, "
                 f"dir_acc: {oot['dir_acc']:.3f}")
        log.info(f"  OOT days: {oot['n_oot_days']}, predictions: {oot['n_predictions']:,}")

        # Best trade sim
        if result.get("trade_sims"):
            for thresh_key, sim in result["trade_sims"].items():
                log.info(
                    f"  {thresh_key}: n={sim['n_trades']}, "
                    f"Sharpe={sim['sharpe']:.2f}, Sortino={sim['sortino']:.2f}, "
                    f"WR={sim['win_rate']:.1%}, PF={sim['profit_factor']:.2f}, "
                    f"avg={sim['avg_pnl_ticks']:.2f}t, "
                    f"total=${sim['total_pnl_dollars']:.0f}"
                )
                if sim["long_trades"] > 0:
                    log.info(
                        f"    LONG: n={sim['long_trades']}, WR={sim['long_wr']:.1%}, "
                        f"avg={sim['long_avg']:.2f}t, Sharpe={sim['long_sharpe']:.2f}"
                    )
                if sim["short_trades"] > 0:
                    log.info(
                        f"    SHORT: n={sim['short_trades']}, WR={sim['short_wr']:.1%}, "
                        f"avg={sim['short_avg']:.2f}t, Sharpe={sim['short_sharpe']:.2f}"
                    )

        # Regime
        log.info(f"  Regime gap: {regime.get('regime_gap_detail', 'N/A')}")

        # Per-day IC
        pd_ic = result.get("per_day_ic_summary", {})
        log.info(
            f"  Per-day IC: green={pd_ic.get('green_mean', 'N/A')}, "
            f"red={pd_ic.get('red_mean', 'N/A')}, flat={pd_ic.get('flat_mean', 'N/A')}"
        )

    log.info("\n" + "=" * 70)
    log.info("DONE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
