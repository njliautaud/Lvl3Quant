#!/usr/bin/env python3
"""
Longer-Horizon Directional Model v2 — PyTorch Neural Network
=============================================================

Temporal Fusion-style GRU + Self-Attention model for multi-horizon
directional prediction on ES futures using MBO microstructure.

Features:
  - 15-minute bar aggregation (finer than v1's hourly)
  - OFI momentum, volume profile, sweep intensity, volatility, time encoding
  - Optional queue-augmented features (41 days overlap)
  - Multi-horizon output heads: 15min, 1h, 2h, 4h
  - Auxiliary trade quality head
  - Walk-forward sliding window (60d train, 5d val, slide 5d) — HC #0
  - Strict leakage audit
  - MLflow logging
  - Per-fold .npz weight + prediction saves

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python \\
      alpha_discovery/longer_horizon_nn_v2.py --device cuda --epochs 50 --hidden-dim 256

Author: Claude (autonomous research)
"""

import argparse
import gc
import json
import logging
import math
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
OUTPUT_DIR = ROOT / "output" / "longer_horizon_nn_v2"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

# Create dirs
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [LH-NN-v2] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "longer_horizon_nn_v2.log")),
    ],
)
log = logging.getLogger("LH-NN-v2")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376
# Market entry + market exit + RT commission = 2.376 ticks
COST_RT_TICKS = 2.376

# Minimum ticks of directional move to justify a trade per horizon
MIN_EDGE_TICKS = {
    "15min": 2.0,
    "1h": 3.0,
    "2h": 4.0,
    "4h": 5.0,
}

# ─────────────────────────────────────────────
#  PYTORCH IMPORTS (deferred so help works w/o torch)
# ─────────────────────────────────────────────
torch = None
nn = None


def _import_torch():
    global torch, nn
    if torch is not None:
        return
    import torch as _torch
    import torch.nn as _nn

    torch = _torch
    nn = _nn


# ═════════════════════════════════════════════
#  STEP 1: DATA LOADING
# ═════════════════════════════════════════════


def load_all_minute_bars(min_date: str = "20250714") -> pd.DataFrame:
    """Load all minute bar parquets into a single DataFrame.

    Reuses logic from alpha_discovery.longer_horizon_directional_v1.
    """
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
        log.info("No queue_augmented_features directory found — skipping queue features")
        return None

    files = sorted(QUEUE_FEATURE_DIR.glob("features_*.parquet"))
    if not files:
        log.info("No queue feature files found — skipping")
        return None

    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Extract date from filename: features_YYYYMMDD.parquet
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


# ═════════════════════════════════════════════
#  STEP 2: 15-MINUTE BAR AGGREGATION + FEATURES
# ═════════════════════════════════════════════


def _safe_polyfit_slope(arr: np.ndarray) -> float:
    """Linear regression slope, returns 0 on failure."""
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def aggregate_to_15min(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute bars into 15-minute bars with microstructure features."""
    df = minute_df.copy()

    # Create 15min period key
    df["bar_15m"] = df["ts_minute"].dt.floor("15min")

    # Per-minute derived
    df["return_1m"] = df.groupby("date")["close"].pct_change()
    df["abs_ofi"] = df["ofi_1min"].abs()
    sv_std = df.groupby("date")["signed_volume"].transform("std").replace(0, 1)
    df["sv_zscore"] = df["signed_volume"] / sv_std
    df["vwap_dev"] = (df["close"] - df["vwap"]) / df["close"].clip(lower=1)

    records = []
    for (date_str, bar_key), grp in df.groupby(["date", "bar_15m"]):
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
            "bar_15m": bar_key,
            "ts": grp["ts_minute"].iloc[0],
            # Price action
            "open": close_arr[0],
            "high": close_arr.max(),
            "low": close_arr.min(),
            "close": close_arr[-1],
            "return_15m": (close_arr[-1] / close_arr[0] - 1) if close_arr[0] > 0 else 0,
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
                ofi_arr[len(ofi_arr) // 2 :].sum() - ofi_arr[: len(ofi_arr) // 2].sum()
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
            )
            if len(sv_arr) > 0
            else 0.0,
            # Spread & liquidity
            "spread_mean": spread_arr.mean(),
            "spread_max": spread_arr.max(),
            "spread_trend": _safe_polyfit_slope(spread_arr),
            # Trade intensity
            "trade_count_sum": tc_arr.sum(),
            "trade_count_trend": _safe_polyfit_slope(tc_arr),
            # Volatility
            "realized_vol": float(np.std(ret_arr) * np.sqrt(252 * 26)) if len(ret_arr) > 1 else 0,
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
    log.info(f"Aggregated {len(result):,} 15-min bars with {len(result.columns)} columns")
    return result


def add_queue_features_to_bars(
    bars_df: pd.DataFrame, queue_df: Optional[pd.DataFrame]
) -> pd.DataFrame:
    """Merge queue-augmented tick features into 15min bars.

    For dates where queue data exists, aggregate tick-level features
    into per-15min-bar summaries.
    """
    if queue_df is None or queue_df.empty:
        log.info("No queue features to merge")
        return bars_df

    # Expected queue columns
    queue_cols_mean = [
        "bid_qty_at_touch",
        "ask_qty_at_touch",
        "top_imbalance",
        "bid_q_ahead_p50",
        "ask_q_ahead_p50",
        "microprice_offset_ticks",
    ]
    queue_cols_sum = ["ofi_1s", "ofi_5s", "ofi_10s"]
    rate_cols = [
        "bid_add_rate_1s",
        "ask_add_rate_1s",
        "bid_cancel_rate_1s",
        "ask_cancel_rate_1s",
    ]

    # Check which columns exist
    available_mean = [c for c in queue_cols_mean if c in queue_df.columns]
    available_sum = [c for c in queue_cols_sum if c in queue_df.columns]
    available_rate = [c for c in rate_cols if c in queue_df.columns]

    if not available_mean and not available_sum and not available_rate:
        log.warning("Queue dataframe has no expected columns — skipping")
        return bars_df

    # Need a timestamp column for grouping into 15min bars
    ts_col = None
    for candidate in ["ts", "timestamp", "ts_event"]:
        if candidate in queue_df.columns:
            ts_col = candidate
            break

    if ts_col is None:
        log.warning("Queue dataframe has no timestamp column — skipping merge")
        return bars_df

    queue_df[ts_col] = pd.to_datetime(queue_df[ts_col], utc=True)
    queue_df["bar_15m"] = queue_df[ts_col].dt.floor("15min")

    # Aggregate per (date, bar_15m)
    agg_dict = {}
    for c in available_mean:
        agg_dict[c] = ["mean", "std"]
    for c in available_sum:
        agg_dict[c] = ["sum"]
    for c in available_rate:
        agg_dict[c] = ["mean"]

    agg = queue_df.groupby(["date", "bar_15m"]).agg(agg_dict)
    # Flatten multi-level columns
    agg.columns = [f"q_{c}_{stat}" for c, stat in agg.columns]
    agg = agg.reset_index()

    # Derived queue features
    if "bid_cancel_rate_1s" in queue_df.columns and "bid_add_rate_1s" in queue_df.columns:
        per_bar = queue_df.groupby(["date", "bar_15m"]).agg(
            bid_cancel_mean=("bid_cancel_rate_1s", "mean"),
            bid_add_mean=("bid_add_rate_1s", "mean"),
            ask_cancel_mean=("ask_cancel_rate_1s", "mean"),
            ask_add_mean=("ask_add_rate_1s", "mean"),
        ).reset_index()
        per_bar["q_bid_toxicity"] = per_bar["bid_cancel_mean"] / per_bar["bid_add_mean"].clip(lower=1e-6)
        per_bar["q_ask_toxicity"] = per_bar["ask_cancel_mean"] / per_bar["ask_add_mean"].clip(lower=1e-6)
        agg = agg.merge(
            per_bar[["date", "bar_15m", "q_bid_toxicity", "q_ask_toxicity"]],
            on=["date", "bar_15m"],
            how="left",
        )

    # OFI slope within window
    if "ofi_1s" in queue_df.columns:
        def _ofi_slope(g):
            vals = g["ofi_1s"].values
            return _safe_polyfit_slope(vals)
        ofi_slope = queue_df.groupby(["date", "bar_15m"]).apply(_ofi_slope).reset_index()
        ofi_slope.columns = ["date", "bar_15m", "q_ofi_slope"]
        agg = agg.merge(ofi_slope, on=["date", "bar_15m"], how="left")

    n_before = len(bars_df)
    bars_df = bars_df.merge(agg, on=["date", "bar_15m"], how="left")
    n_queue_dates = bars_df.dropna(subset=[agg.columns[2]]).date.nunique() if len(agg.columns) > 2 else 0
    log.info(
        f"Merged queue features: {len(agg.columns)-2} new columns, "
        f"available for {n_queue_dates} dates"
    )
    return bars_df


# ═════════════════════════════════════════════
#  STEP 3: ROLLING CONTEXT + MOMENTUM FEATURES
# ═════════════════════════════════════════════


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-bar rolling features using ONLY past data (causal)."""
    df = df.sort_values("ts").reset_index(drop=True)

    # OFI momentum: rolling z-score across 4/8/16/32 bars
    for w in [4, 8, 16, 32]:
        roll_mean = df["ofi_sum"].rolling(w, min_periods=1).mean()
        roll_std = df["ofi_sum"].rolling(w, min_periods=2).std().fillna(1).replace(0, 1)
        df[f"ofi_zscore_{w}bar"] = (df["ofi_sum"] - roll_mean) / roll_std

        # Volume relative to rolling mean
        vol_ma = df["total_volume"].rolling(w, min_periods=1).mean()
        df[f"vol_rel_{w}bar"] = df["total_volume"] / vol_ma.clip(lower=1)

        # Sweep intensity rolling
        if "sweep_minutes" in df.columns:
            df[f"sweep_pct_{w}bar"] = (
                df["sweep_minutes"].rolling(w, min_periods=1).sum()
                / (w * 15)  # fraction of minutes with sweeps
            )

    # Price momentum (lookback returns)
    # 1h = 4 bars, 2h = 8, 4h = 16
    for bars, label in [(4, "1h"), (8, "2h"), (16, "4h")]:
        df[f"ret_{label}_lb"] = df["close"].pct_change(bars)

    # Realized vol over windows (past only)
    for w in [4, 8, 16]:
        df[f"rvol_{w}bar"] = df["return_15m"].rolling(w, min_periods=2).std()

    # Signed volume cumulative (intraday)
    df["intraday_cum_ofi"] = df.groupby("date")["ofi_sum"].cumsum()
    df["intraday_cum_sv"] = df.groupby("date")["signed_volume_sum"].cumsum()

    # OFI sign change from previous bar
    df["ofi_sign_flip"] = (
        np.sign(df["ofi_sum"]) != np.sign(df["ofi_sum"].shift(1))
    ).astype(np.float32)

    # Absorption: high volume + small range
    df["absorption"] = df["total_volume"] / df["range_ticks"].clip(lower=1)

    # Cross-day: previous day closing stats
    day_stats = (
        df.groupby("date")
        .agg(
            day_ofi=("ofi_sum", "sum"),
            day_sv=("signed_volume_sum", "sum"),
            day_ret=("return_15m", "sum"),
            day_vol=("realized_vol", "mean"),
        )
        .reset_index()
    )
    day_stats["prev_day_ofi"] = day_stats["day_ofi"].shift(1)
    day_stats["prev_day_sv"] = day_stats["day_sv"].shift(1)
    day_stats["prev_day_ret"] = day_stats["day_ret"].shift(1)

    df = df.merge(
        day_stats[["date", "prev_day_ofi", "prev_day_sv", "prev_day_ret"]],
        on="date",
        how="left",
    )

    # Time-of-day encoding (sin/cos of session progress)
    # ES RTH: 09:30-16:00 ET = 13:30-20:00 UTC
    session_start_hour = 13.5  # 13:30 UTC
    session_end_hour = 20.0    # 20:00 UTC
    session_len = session_end_hour - session_start_hour

    hour_frac = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    progress = ((hour_frac - session_start_hour) / session_len).clip(0, 1)
    df["tod_sin"] = np.sin(2 * np.pi * progress)
    df["tod_cos"] = np.cos(2 * np.pi * progress)
    df["tod_progress"] = progress.values

    # Bars since session open
    df["bars_since_open"] = df.groupby("date").cumcount()

    log.info(f"Added rolling features: {len(df.columns)} total columns")
    return df


# ═════════════════════════════════════════════
#  STEP 4: FORWARD LABELS (STRICT NO-LEAKAGE)
# ═════════════════════════════════════════════

HORIZON_BARS = {
    "15min": 1,
    "1h": 4,
    "2h": 8,
    "4h": 16,
}


def add_forward_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute forward return labels per horizon.

    LEAKAGE SAFETY: uses shift(-N) on sorted time series only.
    Labels that cross overnight gaps are set to NaN.
    """
    df = df.sort_values("ts").reset_index(drop=True)

    for label, n_bars in HORIZON_BARS.items():
        fwd_close = df["close"].shift(-n_bars)
        fwd_return = fwd_close / df["close"] - 1
        fwd_ticks = (fwd_close - df["close"]) / 0.25

        # Null out overnight gaps: if time gap > 6h between bar and bar+N, it's a gap
        ts_now = df["ts"].values
        ts_fwd = df["ts"].shift(-n_bars).values
        # Convert to seconds
        gap_mask = pd.isna(df["ts"].shift(-n_bars))
        for i in range(len(df) - n_bars):
            if pd.isna(ts_fwd[i]):
                continue
            diff_s = (pd.Timestamp(ts_fwd[i]) - pd.Timestamp(ts_now[i])).total_seconds()
            if diff_s > 6 * 3600:
                fwd_return.iloc[i] = np.nan
                fwd_ticks.iloc[i] = np.nan

        df[f"fwd_return_{label}"] = fwd_return
        df[f"fwd_ticks_{label}"] = fwd_ticks

        # Direction label
        min_ticks = MIN_EDGE_TICKS.get(label, 3.0)
        df[f"direction_{label}"] = 0
        df.loc[fwd_ticks > min_ticks, f"direction_{label}"] = 1
        df.loc[fwd_ticks < -min_ticks, f"direction_{label}"] = -1

        # Trade quality: did the move reach 2x cost before reversing?
        # Approximated as: |fwd_ticks| > 2 * COST_RT_TICKS
        df[f"trade_quality_{label}"] = (fwd_ticks.abs() > 2 * COST_RT_TICKS).astype(np.float32)

    log.info(f"Added forward labels for horizons: {list(HORIZON_BARS.keys())}")
    return df


# ═════════════════════════════════════════════
#  STEP 5: LEAKAGE AUDIT
# ═════════════════════════════════════════════


def leakage_audit(
    df: pd.DataFrame,
    train_dates: List[str],
    val_dates: List[str],
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Explicit leakage audit. Returns dict of check results."""
    results = {}

    # CHECK 1: No future dates in training set
    train_set = set(train_dates)
    val_set = set(val_dates)
    overlap = train_set & val_set
    results["date_overlap"] = len(overlap) == 0
    if overlap:
        log.error(f"LEAKAGE: Train/val date overlap: {overlap}")

    # CHECK 2: Train dates are ALL before val dates
    max_train = max(train_dates)
    min_val = min(val_dates)
    results["temporal_order"] = max_train < min_val
    if not results["temporal_order"]:
        log.error(
            f"LEAKAGE: Max train date {max_train} >= min val date {min_val}"
        )

    # CHECK 3: Feature columns don't include forward labels
    fwd_leak = [c for c in feature_cols if c.startswith(("fwd_", "direction_", "trade_quality_"))]
    results["no_forward_features"] = len(fwd_leak) == 0
    if fwd_leak:
        log.error(f"LEAKAGE: Forward-looking columns in features: {fwd_leak}")

    # CHECK 4: No price-level columns that could proxy future
    price_cols = [c for c in feature_cols if c in ("close", "high", "low", "open")]
    results["no_raw_price_features"] = len(price_cols) == 0
    if price_cols:
        log.warning(f"WARNING: Raw price columns in features (potential proxy leakage): {price_cols}")

    all_passed = all(results.values())
    results["all_passed"] = all_passed
    if all_passed:
        log.info("Leakage audit PASSED")
    else:
        log.error(f"Leakage audit FAILED: {results}")

    return results


# ═════════════════════════════════════════════
#  STEP 6: PYTORCH MODEL
# ═════════════════════════════════════════════


def _build_model_class():
    """Build model class after torch import."""
    _import_torch()

    class LongerHorizonModelV2(nn.Module):
        """
        Temporal Fusion-style model:
          - Input: (batch, seq_len, n_features) — sequences of 15min bars
          - Encoder: 2-layer bidirectional GRU + multi-head self-attention
          - Horizon heads: one regression head per horizon (15min, 1h, 2h, 4h)
          - Trade quality head: auxiliary binary classification
          - Output: direction scores + confidence
        """

        def __init__(
            self,
            n_features: int,
            hidden_dim: int = 256,
            n_heads: int = 4,
            n_gru_layers: int = 2,
            dropout: float = 0.2,
            horizons: Tuple[str, ...] = ("15min", "1h", "2h", "4h"),
        ):
            super().__init__()
            self.horizons = horizons
            self.hidden_dim = hidden_dim

            # Input projection
            self.input_proj = nn.Sequential(
                nn.Linear(n_features, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

            # Bidirectional GRU encoder
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=n_gru_layers,
                batch_first=True,
                bidirectional=True,
                dropout=dropout if n_gru_layers > 1 else 0,
            )

            # Self-attention over GRU outputs
            gru_out_dim = hidden_dim * 2  # bidirectional
            self.attn_proj = nn.Linear(gru_out_dim, hidden_dim)
            self.self_attn = nn.MultiheadAttention(
                embed_dim=hidden_dim,
                num_heads=n_heads,
                dropout=dropout,
                batch_first=True,
            )
            self.attn_norm = nn.LayerNorm(hidden_dim)

            # Aggregation: attention-weighted pooling
            self.pool_attn = nn.Linear(hidden_dim, 1)

            # Per-horizon regression heads
            self.horizon_heads = nn.ModuleDict()
            for h in horizons:
                self.horizon_heads[h] = nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim // 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim // 2, 2),  # [direction_score, confidence]
                )

            # Auxiliary trade quality head (binary: did trade reach 2x cost?)
            self.quality_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, len(horizons)),  # one logit per horizon
            )

            self._init_weights()

        def _init_weights(self):
            for m in self.modules():
                if isinstance(m, nn.Linear):
                    nn.init.xavier_uniform_(m.weight)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)

        def forward(
            self, x: "torch.Tensor"
        ) -> Tuple[Dict[str, "torch.Tensor"], "torch.Tensor"]:
            """
            Args:
                x: (batch, seq_len, n_features)
            Returns:
                horizon_outputs: {horizon: (batch, 2)} — [direction_score, confidence]
                quality_logits: (batch, n_horizons) — trade quality logits
            """
            # Input projection
            h = self.input_proj(x)  # (B, T, hidden)

            # GRU encoding
            gru_out, _ = self.gru(h)  # (B, T, hidden*2)

            # Project back for attention
            attn_in = self.attn_proj(gru_out)  # (B, T, hidden)

            # Self-attention
            attn_out, _ = self.self_attn(attn_in, attn_in, attn_in)
            attn_out = self.attn_norm(attn_out + attn_in)  # residual

            # Attention-weighted pooling
            pool_weights = torch.softmax(self.pool_attn(attn_out).squeeze(-1), dim=1)
            context = (attn_out * pool_weights.unsqueeze(-1)).sum(dim=1)  # (B, hidden)

            # Per-horizon outputs
            horizon_outputs = {}
            for h_name in self.horizons:
                horizon_outputs[h_name] = self.horizon_heads[h_name](context)

            # Quality head
            quality_logits = self.quality_head(context)

            return horizon_outputs, quality_logits

    return LongerHorizonModelV2


# ═════════════════════════════════════════════
#  STEP 7: DATASET + SEQUENCE CONSTRUCTION
# ═════════════════════════════════════════════


def _build_dataset_class():
    _import_torch()
    from torch.utils.data import Dataset

    class SequenceDataset(Dataset):
        """Sliding window dataset: each sample is seq_len consecutive 15min bars."""

        def __init__(
            self,
            features: np.ndarray,
            labels: Dict[str, np.ndarray],
            quality_labels: Dict[str, np.ndarray],
            seq_len: int = 24,
            dates: Optional[np.ndarray] = None,
        ):
            self.features = features.astype(np.float32)
            self.labels = {k: v.astype(np.float32) for k, v in labels.items()}
            self.quality_labels = {k: v.astype(np.float32) for k, v in quality_labels.items()}
            self.seq_len = seq_len
            self.dates = dates

            # Valid indices: sequence must not span overnight gap
            self.valid_indices = []
            if dates is not None:
                for i in range(len(features) - seq_len):
                    # Check all bars in window are from same or consecutive days (no gaps)
                    window_dates = dates[i : i + seq_len]
                    # Simple check: first and last date must be same day
                    # (15min bars, 24 bars = 6h, fits within a session)
                    if window_dates[0] == window_dates[-1]:
                        # Target bar is the one after the window
                        target_idx = i + seq_len - 1
                        # Check label is valid for at least one horizon
                        has_label = False
                        for h, lbl in self.labels.items():
                            if not np.isnan(lbl[target_idx]):
                                has_label = True
                                break
                        if has_label:
                            self.valid_indices.append(i)
            else:
                self.valid_indices = list(range(len(features) - seq_len))

        def __len__(self):
            return len(self.valid_indices)

        def __getitem__(self, idx):
            i = self.valid_indices[idx]
            target_idx = i + self.seq_len - 1

            x = torch.from_numpy(self.features[i : i + self.seq_len])

            label_dict = {}
            quality_dict = {}
            for h in self.labels:
                val = self.labels[h][target_idx]
                label_dict[h] = torch.tensor(val if not np.isnan(val) else 0.0, dtype=torch.float32)
                q_val = self.quality_labels[h][target_idx]
                quality_dict[h] = torch.tensor(
                    q_val if not np.isnan(q_val) else 0.0, dtype=torch.float32
                )

            # Mask: 1.0 if label is valid, 0.0 if NaN (don't train on this)
            mask_dict = {}
            for h in self.labels:
                val = self.labels[h][target_idx]
                mask_dict[h] = torch.tensor(0.0 if np.isnan(val) else 1.0, dtype=torch.float32)

            return x, label_dict, quality_dict, mask_dict

    return SequenceDataset


# ═════════════════════════════════════════════
#  STEP 8: TRAINING LOOP
# ═════════════════════════════════════════════


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select feature columns, excluding labels, metadata, and raw prices."""
    exclude_prefixes = (
        "fwd_",
        "direction_",
        "trade_quality_",
        "date",
        "ts",
        "bar_15m",
    )
    # Raw price levels can proxy for future — exclude
    raw_price_cols = {"open", "high", "low", "close"}

    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if c in raw_price_cols:
            continue
        # Must be numeric
        if df[c].dtype in (np.float64, np.float32, np.int64, np.int32, np.float16, np.int16):
            cols.append(c)
    return cols


def train_one_fold(
    model,
    train_loader,
    val_loader,
    horizons: List[str],
    device: str,
    epochs: int,
    lr: float,
    patience: int,
    fold_idx: int,
) -> Tuple[Dict[str, float], float]:
    """Train one fold with early stopping. Returns best val metrics and loss."""
    _import_torch()

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=lr * 0.01)

    mse_loss = nn.MSELoss(reduction="none")
    bce_loss = nn.BCEWithLogitsLoss(reduction="none")

    best_val_loss = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(epochs):
        # ── TRAIN ──
        model.train()
        train_losses = []
        for batch in train_loader:
            x, labels, quality_labels, masks = batch
            x = x.to(device)

            horizon_out, quality_logits = model(x)

            loss = torch.tensor(0.0, device=device)
            n_valid = 0

            for i, h in enumerate(horizons):
                lbl = labels[h].to(device)
                msk = masks[h].to(device)
                q_lbl = quality_labels[h].to(device)

                if msk.sum() < 1:
                    continue

                # Direction regression loss
                pred_direction = horizon_out[h][:, 0]
                reg_loss = (mse_loss(pred_direction, lbl) * msk).sum() / msk.sum()
                loss = loss + reg_loss

                # Quality head loss
                q_logit = quality_logits[:, i]
                q_loss = (bce_loss(q_logit, q_lbl) * msk).sum() / msk.sum()
                loss = loss + 0.3 * q_loss

                n_valid += 1

            if n_valid > 0:
                loss = loss / n_valid
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                train_losses.append(loss.item())

        scheduler.step()

        # ── VALIDATE ──
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                x, labels, quality_labels, masks = batch
                x = x.to(device)

                horizon_out, quality_logits = model(x)

                loss = torch.tensor(0.0, device=device)
                n_valid = 0
                for i, h in enumerate(horizons):
                    lbl = labels[h].to(device)
                    msk = masks[h].to(device)
                    if msk.sum() < 1:
                        continue
                    pred_direction = horizon_out[h][:, 0]
                    reg_loss = (mse_loss(pred_direction, lbl) * msk).sum() / msk.sum()
                    loss = loss + reg_loss
                    n_valid += 1

                if n_valid > 0:
                    val_losses.append((loss / n_valid).item())

        avg_train = np.mean(train_losses) if train_losses else float("inf")
        avg_val = np.mean(val_losses) if val_losses else float("inf")

        if epoch % 5 == 0 or epoch == epochs - 1:
            log.info(
                f"  Fold {fold_idx} Epoch {epoch:3d}/{epochs}: "
                f"train_loss={avg_train:.6f}  val_loss={avg_val:.6f}  "
                f"lr={scheduler.get_last_lr()[0]:.2e}"
            )

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log.info(f"  Fold {fold_idx}: Early stop at epoch {epoch} (patience={patience})")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    return {}, best_val_loss


def predict_fold(
    model,
    loader,
    horizons: List[str],
    device: str,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Run inference, return predictions and confidence per horizon."""
    _import_torch()
    model.eval()

    preds = {h: [] for h in horizons}
    confs = {h: [] for h in horizons}

    with torch.no_grad():
        for batch in loader:
            x = batch[0].to(device)
            horizon_out, quality_logits = model(x)

            for i, h in enumerate(horizons):
                out = horizon_out[h].cpu().numpy()
                preds[h].append(out[:, 0])  # direction score
                confs[h].append(torch.sigmoid(quality_logits[:, i]).cpu().numpy())

    for h in horizons:
        preds[h] = np.concatenate(preds[h])
        confs[h] = np.concatenate(confs[h])

    return preds, confs


# ═════════════════════════════════════════════
#  STEP 9: WALK-FORWARD ORCHESTRATOR
# ═════════════════════════════════════════════


def collate_fn(batch):
    """Custom collate that handles dict labels."""
    xs = torch.stack([b[0] for b in batch])
    labels = {}
    quality = {}
    masks = {}
    horizons = batch[0][1].keys()
    for h in horizons:
        labels[h] = torch.stack([b[1][h] for b in batch])
        quality[h] = torch.stack([b[2][h] for b in batch])
        masks[h] = torch.stack([b[3][h] for b in batch])
    return xs, labels, quality, masks


def walk_forward_train(
    df: pd.DataFrame,
    args,
) -> Dict:
    """Full walk-forward sliding window training."""
    _import_torch()
    from torch.utils.data import DataLoader

    horizons = list(HORIZON_BARS.keys())
    feature_cols = get_feature_columns(df)
    log.info(f"Feature columns ({len(feature_cols)}): {feature_cols[:20]}...")

    dates = sorted(df["date"].unique())
    log.info(f"Total trading days: {len(dates)} ({dates[0]} -> {dates[-1]})")

    train_days = args.train_days
    val_days = args.val_days
    slide = args.slide_days
    seq_len = args.seq_len

    if len(dates) < train_days + val_days + slide:
        raise RuntimeError(
            f"Not enough days ({len(dates)}) for {train_days}+{val_days} walk-forward"
        )

    # Prepare arrays
    features_all = df[feature_cols].values.astype(np.float32)
    dates_all = df["date"].values

    label_arrays = {}
    quality_arrays = {}
    for h in horizons:
        label_arrays[h] = df[f"fwd_ticks_{h}"].values.astype(np.float32)
        quality_arrays[h] = df[f"trade_quality_{h}"].values.astype(np.float32)

    # Build model class
    ModelClass = _build_model_class()
    DatasetClass = _build_dataset_class()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    # MLflow
    mlflow_client = None
    if args.mlflow:
        try:
            import mlflow

            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.experiment_name)
            mlflow.start_run(
                run_name=f"lh_nn_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            mlflow.log_params(
                {
                    "model": "GRU+SelfAttn",
                    "hidden_dim": args.hidden_dim,
                    "n_heads": args.n_heads,
                    "n_gru_layers": args.n_gru_layers,
                    "dropout": args.dropout,
                    "seq_len": seq_len,
                    "train_days": train_days,
                    "val_days": val_days,
                    "slide_days": slide,
                    "epochs": args.epochs,
                    "batch_size": args.batch_size,
                    "lr": args.lr,
                    "cost_rt_ticks": COST_RT_TICKS,
                    "n_features": len(feature_cols),
                    "n_dates": len(dates),
                    "horizons": ",".join(horizons),
                }
            )
            mlflow_client = mlflow
        except Exception as e:
            log.warning(f"MLflow setup failed: {e}")

    # Walk-forward loop
    all_fold_results = []
    all_oot_preds = {h: [] for h in horizons}
    all_oot_actuals = {h: [] for h in horizons}
    all_oot_confs = {h: [] for h in horizons}
    all_oot_dates = []
    fold_idx = 0

    start_idx = train_days
    for fold_start in range(start_idx, len(dates) - val_days + 1, slide):
        fold_train_dates = dates[fold_start - train_days : fold_start]
        fold_val_dates = dates[fold_start : fold_start + val_days]

        if len(fold_val_dates) < val_days:
            break

        fold_idx += 1
        log.info(
            f"\n{'='*50}\n"
            f"FOLD {fold_idx}: train {fold_train_dates[0]}->{fold_train_dates[-1]} "
            f"({len(fold_train_dates)}d), val {fold_val_dates[0]}->{fold_val_dates[-1]} "
            f"({len(fold_val_dates)}d)\n{'='*50}"
        )

        # ── LEAKAGE AUDIT ──
        audit = leakage_audit(df, fold_train_dates, fold_val_dates, feature_cols)
        if not audit["all_passed"]:
            log.error(f"FOLD {fold_idx}: Leakage audit FAILED — skipping fold")
            continue

        # ── FEATURE NORMALIZATION: fit on train, transform both ──
        train_mask = np.isin(dates_all, fold_train_dates)
        val_mask = np.isin(dates_all, fold_val_dates)

        train_features = features_all[train_mask].copy()
        val_features = features_all[val_mask].copy()

        # Robust scaling: median/IQR from training set ONLY
        train_median = np.nanmedian(train_features, axis=0)
        q75 = np.nanpercentile(train_features, 75, axis=0)
        q25 = np.nanpercentile(train_features, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0  # prevent div by zero

        # Apply normalization
        train_features = (train_features - train_median) / iqr
        val_features = (val_features - train_median) / iqr

        # Replace NaN/inf
        train_features = np.nan_to_num(train_features, nan=0.0, posinf=3.0, neginf=-3.0)
        val_features = np.nan_to_num(val_features, nan=0.0, posinf=3.0, neginf=-3.0)

        # Clip extremes
        train_features = np.clip(train_features, -5, 5)
        val_features = np.clip(val_features, -5, 5)

        # Extract labels for this fold
        train_labels = {h: label_arrays[h][train_mask] for h in horizons}
        val_labels = {h: label_arrays[h][val_mask] for h in horizons}
        train_quality = {h: quality_arrays[h][train_mask] for h in horizons}
        val_quality = {h: quality_arrays[h][val_mask] for h in horizons}
        train_dates_fold = dates_all[train_mask]
        val_dates_fold = dates_all[val_mask]

        # Build datasets
        train_ds = DatasetClass(
            train_features, train_labels, train_quality, seq_len, train_dates_fold
        )
        val_ds = DatasetClass(
            val_features, val_labels, val_quality, seq_len, val_dates_fold
        )

        if len(train_ds) < 50 or len(val_ds) < 10:
            log.warning(
                f"Fold {fold_idx}: too few samples (train={len(train_ds)}, val={len(val_ds)}) — skip"
            )
            continue

        train_loader = DataLoader(
            train_ds,
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=min(args.num_workers, 8),
            pin_memory=(device == "cuda"),
            collate_fn=collate_fn,
            drop_last=True,
        )
        val_loader = DataLoader(
            val_ds,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=min(args.num_workers, 4),
            pin_memory=(device == "cuda"),
            collate_fn=collate_fn,
        )

        # ── BUILD MODEL ──
        model = ModelClass(
            n_features=len(feature_cols),
            hidden_dim=args.hidden_dim,
            n_heads=args.n_heads,
            n_gru_layers=args.n_gru_layers,
            dropout=args.dropout,
            horizons=tuple(horizons),
        ).to(device)

        # ── TRAIN ──
        _, best_val_loss = train_one_fold(
            model,
            train_loader,
            val_loader,
            horizons,
            device,
            args.epochs,
            args.lr,
            args.patience,
            fold_idx,
        )

        # ── PREDICT ON VALIDATION ──
        preds, confs = predict_fold(model, val_loader, horizons, device)

        # Collect OOT predictions
        # We need actual labels aligned with predictions
        # val_ds.valid_indices tells us which bars in val set generated predictions
        val_target_indices = [vi + seq_len - 1 for vi in val_ds.valid_indices]

        fold_result = {
            "fold": fold_idx,
            "train_start": fold_train_dates[0],
            "train_end": fold_train_dates[-1],
            "val_start": fold_val_dates[0],
            "val_end": fold_val_dates[-1],
            "train_samples": len(train_ds),
            "val_samples": len(val_ds),
            "best_val_loss": best_val_loss,
        }

        for h in horizons:
            actuals_h = val_labels[h][val_target_indices]
            preds_h = preds[h]

            # Only evaluate where we have valid labels
            valid = ~np.isnan(actuals_h)
            if valid.sum() < 5:
                fold_result[f"ic_{h}"] = np.nan
                continue

            p = preds_h[valid]
            a = actuals_h[valid]

            ic = np.corrcoef(p, a)[0, 1] if len(p) > 5 else 0
            dir_acc = np.mean(np.sign(p) == np.sign(a))
            fold_result[f"ic_{h}"] = ic
            fold_result[f"dir_acc_{h}"] = dir_acc

            # Accumulate OOT
            all_oot_preds[h].append(p)
            all_oot_actuals[h].append(a)
            all_oot_confs[h].append(confs[h][valid])

        val_dates_for_fold = val_dates_fold[val_target_indices]
        all_oot_dates.extend(val_dates_for_fold.tolist())

        all_fold_results.append(fold_result)

        # Save fold weights + predictions
        fold_dir = OUTPUT_DIR / f"fold_{fold_idx:03d}"
        fold_dir.mkdir(parents=True, exist_ok=True)

        torch.save(model.state_dict(), fold_dir / "model_weights.pt")

        npz_data = {"dates": np.array(val_dates_for_fold)}
        for h in horizons:
            if len(preds[h]) > 0:
                actuals_h = val_labels[h][val_target_indices]
                valid = ~np.isnan(actuals_h)
                npz_data[f"preds_{h}"] = preds[h][valid]
                npz_data[f"actuals_{h}"] = actuals_h[valid]
                npz_data[f"confs_{h}"] = confs[h][valid]
        np.savez_compressed(fold_dir / "predictions.npz", **npz_data)

        # Save normalization params for this fold
        np.savez(fold_dir / "norm_params.npz", median=train_median, iqr=iqr)

        log.info(
            f"Fold {fold_idx} done: val_loss={best_val_loss:.6f}, "
            + ", ".join(
                f"IC_{h}={fold_result.get(f'ic_{h}', np.nan):.4f}" for h in horizons
            )
        )

        if mlflow_client:
            for h in horizons:
                ic_val = fold_result.get(f"ic_{h}", None)
                if ic_val is not None and not np.isnan(ic_val):
                    mlflow_client.log_metric(f"fold_ic_{h}", ic_val, step=fold_idx)
            mlflow_client.log_metric("fold_val_loss", best_val_loss, step=fold_idx)

        # Cleanup
        del model, train_loader, val_loader, train_ds, val_ds
        gc.collect()
        if device == "cuda":
            torch.cuda.empty_cache()

    # ═════════════════════════════════════════════
    #  AGGREGATE OOT RESULTS
    # ═════════════════════════════════════════════
    log.info(f"\n{'='*60}\nAGGREGATE OOT RESULTS ({fold_idx} folds)\n{'='*60}")

    concat_results = {}
    for h in horizons:
        if not all_oot_preds[h]:
            continue

        p = np.concatenate(all_oot_preds[h])
        a = np.concatenate(all_oot_actuals[h])
        c = np.concatenate(all_oot_confs[h])

        if len(p) < 10:
            continue

        ic = np.corrcoef(p, a)[0, 1]
        rank_ic = stats.spearmanr(p, a)[0]
        dir_acc = np.mean(np.sign(p) == np.sign(a))

        # Per-fold IC for Sharpe calculation
        fold_ics = [
            fr.get(f"ic_{h}", np.nan) for fr in all_fold_results if not np.isnan(fr.get(f"ic_{h}", np.nan))
        ]
        ic_mean = np.mean(fold_ics) if fold_ics else 0
        ic_std = np.std(fold_ics) if len(fold_ics) > 1 else 1e-6
        ic_sharpe = ic_mean / max(ic_std, 1e-6)

        concat_results[h] = {
            "concat_ic": float(ic),
            "rank_ic": float(rank_ic),
            "dir_acc": float(dir_acc),
            "ic_mean": float(ic_mean),
            "ic_std": float(ic_std),
            "ic_sharpe": float(ic_sharpe),
            "n_predictions": int(len(p)),
            "n_folds": len(fold_ics),
        }

        log.info(
            f"  {h}: IC={ic:.4f}, RankIC={rank_ic:.4f}, DirAcc={dir_acc:.1%}, "
            f"IC_Sharpe={ic_sharpe:.2f}, N={len(p):,}"
        )

        # Quantile analysis
        for q_pct in [10, 20, 30]:
            q = q_pct / 100
            top_mask = p >= np.quantile(p, 1 - q)
            bot_mask = p <= np.quantile(p, q)
            top_actual = a[top_mask].mean()
            bot_actual = a[bot_mask].mean()
            top_wr = np.mean(a[top_mask] > 0)
            bot_wr = np.mean(a[bot_mask] < 0)

            concat_results[h][f"top{q_pct}_mean_ticks"] = float(top_actual)
            concat_results[h][f"bot{q_pct}_mean_ticks"] = float(bot_actual)
            concat_results[h][f"top{q_pct}_wr"] = float(top_wr)
            concat_results[h][f"bot{q_pct}_wr"] = float(bot_wr)
            concat_results[h][f"ls{q_pct}_spread"] = float(top_actual - bot_actual)

            log.info(
                f"    Q{q_pct}: long={top_actual:+.2f}tk WR={top_wr:.1%}, "
                f"short={bot_actual:+.2f}tk WR={bot_wr:.1%}, "
                f"L/S spread={top_actual - bot_actual:.2f}tk"
            )

        if mlflow_client:
            mlflow_client.log_metrics(
                {
                    f"oot_ic_{h}": ic,
                    f"oot_rank_ic_{h}": rank_ic,
                    f"oot_dir_acc_{h}": dir_acc,
                    f"oot_ic_sharpe_{h}": ic_sharpe,
                }
            )

    # ═════════════════════════════════════════════
    #  REGIME-STRATIFIED ANALYSIS
    # ═════════════════════════════════════════════
    oot_dates_arr = np.array(all_oot_dates)
    regime_results = {}

    # Build day-level regime from daily returns
    day_closes = df.groupby("date")["close"].last()
    if len(day_closes) > 0:
        day_returns = day_closes.pct_change()
        # Green day = positive return, Red day = negative
        green_days = set(day_returns[day_returns > 0].index)
        red_days = set(day_returns[day_returns <= 0].index)

        for h in horizons:
            if h not in concat_results:
                continue
            p = np.concatenate(all_oot_preds[h])
            a = np.concatenate(all_oot_actuals[h])

            for regime_name, regime_dates in [("green", green_days), ("red", red_days)]:
                rmask = np.isin(oot_dates_arr[: len(p)], list(regime_dates))
                if rmask.sum() < 10:
                    continue
                r_ic = np.corrcoef(p[rmask], a[rmask])[0, 1] if rmask.sum() > 5 else 0
                r_dir = np.mean(np.sign(p[rmask]) == np.sign(a[rmask]))
                key = f"{h}_{regime_name}"
                regime_results[key] = {
                    "ic": float(r_ic),
                    "dir_acc": float(r_dir),
                    "n": int(rmask.sum()),
                }
                log.info(f"  Regime {key}: IC={r_ic:.4f}, DirAcc={r_dir:.1%}, N={rmask.sum()}")

        # HC #428 R1: check regime agnosticism
        for h in horizons:
            green_key = f"{h}_green"
            red_key = f"{h}_red"
            if green_key in regime_results and red_key in regime_results:
                g_ic = regime_results[green_key]["ic"]
                r_ic = regime_results[red_key]["ic"]
                max_ic = max(abs(g_ic), abs(r_ic))
                if max_ic > 0:
                    regime_gap = abs(g_ic - r_ic) / max_ic
                    log.info(
                        f"  {h} regime gap: |green-red|/max = {regime_gap:.2f} "
                        f"({'PASS' if regime_gap <= 0.50 else 'FAIL: >0.50'})"
                    )

    # ═════════════════════════════════════════════
    #  TRADING SIMULATION
    # ═════════════════════════════════════════════
    sim_results = {}
    for h in horizons:
        if h not in concat_results:
            continue
        p = np.concatenate(all_oot_preds[h])
        a = np.concatenate(all_oot_actuals[h])

        for conf_label, conf_thresh in [("top10", 0.10), ("top20", 0.20), ("top30", 0.30)]:
            sim = _simulate_trades(p, a, confidence_pct=conf_thresh, cost_ticks=COST_RT_TICKS)
            if sim:
                key = f"{h}_{conf_label}"
                sim_results[key] = sim
                log.info(
                    f"  SIM {key}: {sim['n_trades']} trades, "
                    f"Sharpe={sim['sharpe']:.2f}, Sortino={sim['sortino']:.2f}, "
                    f"WR={sim['win_rate']:.1%}, PF={sim['profit_factor']:.2f}, "
                    f"PnL={sim['total_pnl_ticks']:.0f}tk (${sim['total_pnl_dollars']:,.0f})"
                )

                if mlflow_client:
                    mlflow_client.log_metrics(
                        {
                            f"sim_{key}_sharpe": sim["sharpe"],
                            f"sim_{key}_sortino": sim["sortino"],
                            f"sim_{key}_wr": sim["win_rate"],
                            f"sim_{key}_pf": sim["profit_factor"],
                            f"sim_{key}_trades": sim["n_trades"],
                        }
                    )

    # ═════════════════════════════════════════════
    #  SAVE CONCAT OOT PREDICTIONS
    # ═════════════════════════════════════════════
    npz_concat = {"dates": oot_dates_arr}
    for h in horizons:
        if all_oot_preds[h]:
            npz_concat[f"preds_{h}"] = np.concatenate(all_oot_preds[h])
            npz_concat[f"actuals_{h}"] = np.concatenate(all_oot_actuals[h])
            npz_concat[f"confs_{h}"] = np.concatenate(all_oot_confs[h])
    np.savez_compressed(OUTPUT_DIR / "concat_oot_predictions.npz", **npz_concat)

    # ═════════════════════════════════════════════
    #  SUMMARY JSON
    # ═════════════════════════════════════════════
    summary = {
        "run_time": datetime.now().isoformat(),
        "config": {
            "model": "GRU+SelfAttn (LongerHorizonModelV2)",
            "hidden_dim": args.hidden_dim,
            "n_heads": args.n_heads,
            "n_gru_layers": args.n_gru_layers,
            "dropout": args.dropout,
            "seq_len": seq_len,
            "train_days": train_days,
            "val_days": val_days,
            "slide_days": slide,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "cost_rt_ticks": COST_RT_TICKS,
            "n_features": len(feature_cols),
            "feature_columns": feature_cols,
        },
        "concat_results": concat_results,
        "regime_results": regime_results,
        "sim_results": sim_results,
        "fold_results": [
            {k: v for k, v in fr.items() if not isinstance(v, np.ndarray)}
            for fr in all_fold_results
        ],
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Summary saved to {summary_path}")

    if mlflow_client:
        mlflow_client.log_artifact(str(summary_path))
        mlflow_client.log_artifact(str(OUTPUT_DIR / "concat_oot_predictions.npz"))
        mlflow_client.end_run()

    return summary


# ═════════════════════════════════════════════
#  TRADING SIMULATION
# ═════════════════════════════════════════════


def _simulate_trades(
    preds: np.ndarray,
    actuals: np.ndarray,
    confidence_pct: float = 0.10,
    cost_ticks: float = COST_RT_TICKS,
) -> Optional[Dict]:
    """Simulate long/short trades based on prediction confidence quantiles."""
    if len(preds) < 20:
        return None

    upper = np.quantile(preds, 1 - confidence_pct)
    lower = np.quantile(preds, confidence_pct)

    trades = []
    for i in range(len(preds)):
        if preds[i] >= upper:
            pnl = actuals[i] - cost_ticks
            trades.append({"dir": "long", "pnl": pnl, "raw": actuals[i]})
        elif preds[i] <= lower:
            pnl = -actuals[i] - cost_ticks
            trades.append({"dir": "short", "pnl": pnl, "raw": -actuals[i]})

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

    long_pnl = [t["pnl"] for t in trades if t["dir"] == "long"]
    short_pnl = [t["pnl"] for t in trades if t["dir"] == "short"]

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
        "long_trades": len(long_pnl),
        "long_wr": float(np.mean(np.array(long_pnl) > 0)) if long_pnl else 0,
        "long_avg": float(np.mean(long_pnl)) if long_pnl else 0,
        "short_trades": len(short_pnl),
        "short_wr": float(np.mean(np.array(short_pnl) > 0)) if short_pnl else 0,
        "short_avg": float(np.mean(short_pnl)) if short_pnl else 0,
    }


# ═════════════════════════════════════════════
#  MAIN
# ═════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(
        description="Longer-Horizon Directional Model v2 (PyTorch NN)"
    )

    # Data
    parser.add_argument("--min-date", default="20250714", help="Earliest date to load")

    # Walk-forward
    parser.add_argument("--train-days", type=int, default=60, help="Training window days")
    parser.add_argument("--val-days", type=int, default=5, help="Validation window days")
    parser.add_argument("--slide-days", type=int, default=5, help="Slide step days")

    # Model
    parser.add_argument("--hidden-dim", type=int, default=256, help="GRU hidden dimension")
    parser.add_argument("--n-heads", type=int, default=4, help="Self-attention heads")
    parser.add_argument("--n-gru-layers", type=int, default=2, help="GRU layers")
    parser.add_argument("--dropout", type=float, default=0.2, help="Dropout rate")
    parser.add_argument("--seq-len", type=int, default=24, help="Input sequence length (15min bars)")

    # Training
    parser.add_argument("--device", default="cpu", choices=["cpu", "cuda"], help="Device")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs per fold")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers")

    # MLflow
    parser.add_argument("--mlflow", action="store_true", help="Log to MLflow")
    parser.add_argument(
        "--mlflow-uri", default="http://neptune:5000", help="MLflow tracking URI"
    )
    parser.add_argument(
        "--experiment-name", default="longer_horizon_nn_v2", help="MLflow experiment"
    )

    args = parser.parse_args()

    log.info("=" * 60)
    log.info("LONGER-HORIZON DIRECTIONAL MODEL v2 (PyTorch NN)")
    log.info("=" * 60)
    log.info(f"Config: hidden={args.hidden_dim}, heads={args.n_heads}, "
             f"gru_layers={args.n_gru_layers}, seq_len={args.seq_len}")
    log.info(f"Training: {args.train_days}d train, {args.val_days}d val, "
             f"slide {args.slide_days}d, epochs={args.epochs}, lr={args.lr}")
    log.info(f"Device: {args.device}")

    # ── Import torch now ──
    _import_torch()
    log.info(f"PyTorch {torch.__version__}, CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        log.info(f"GPU: {torch.cuda.get_device_name(0)}")

    # ── Step 1: Load data ──
    log.info("\nStep 1: Loading minute bars...")
    t0 = time.time()
    minute_df = load_all_minute_bars(min_date=args.min_date)
    log.info(f"  Loaded in {time.time()-t0:.1f}s")

    # ── Step 2: Load queue features ──
    log.info("\nStep 2: Loading queue features...")
    queue_df = load_queue_features()

    # ── Step 3: Aggregate to 15min bars ──
    log.info("\nStep 3: Aggregating to 15-minute bars...")
    t0 = time.time()
    bars_df = aggregate_to_15min(minute_df)
    del minute_df
    gc.collect()
    log.info(f"  Aggregated in {time.time()-t0:.1f}s")

    # ── Step 4: Merge queue features ──
    log.info("\nStep 4: Merging queue features...")
    bars_df = add_queue_features_to_bars(bars_df, queue_df)
    del queue_df
    gc.collect()

    # ── Step 5: Rolling features ──
    log.info("\nStep 5: Adding rolling features...")
    t0 = time.time()
    bars_df = add_rolling_features(bars_df)
    log.info(f"  Done in {time.time()-t0:.1f}s")

    # ── Step 6: Forward labels ──
    log.info("\nStep 6: Computing forward labels...")
    bars_df = add_forward_labels(bars_df)

    # Save processed dataset
    bars_df.to_parquet(OUTPUT_DIR / "bars_15min_features.parquet", index=False)
    log.info(f"  Saved processed dataset: {len(bars_df):,} rows, {len(bars_df.columns)} cols")

    # ── Step 7: Walk-forward training ──
    log.info("\nStep 7: Walk-forward training...")
    summary = walk_forward_train(bars_df, args)

    # ── Final report ──
    log.info("\n" + "=" * 60)
    log.info("TRAINING COMPLETE")
    log.info("=" * 60)

    if summary and "concat_results" in summary:
        for h, r in summary["concat_results"].items():
            log.info(
                f"  {h}: IC={r['concat_ic']:.4f}, RankIC={r['rank_ic']:.4f}, "
                f"DirAcc={r['dir_acc']:.1%}, IC_Sharpe={r['ic_sharpe']:.2f}"
            )

    return summary


if __name__ == "__main__":
    main()
