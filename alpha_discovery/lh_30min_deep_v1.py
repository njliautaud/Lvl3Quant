#!/usr/bin/env python3
"""
30-Minute Deep Tabular Model v1 — DeepTab with Temporal Context
===============================================================

Hypothesis: LightGBM achieved IC=0.117 on 30-min bars seeing one bar at a time.
A small PyTorch model that sees the LAST 8 bars (4 hours) might capture
temporal patterns in feature sequences that LightGBM misses.

Architecture: DeepTab30min
  - Input: last 8 bars of 30-min features (4 hours of context)
  - Per-bar: Sparse attention (TabNet-inspired) to select important features
  - Temporal: 1-layer GRU to capture sequence patterns
  - Head: prediction of 30-min forward return (regression)
  - Confidence head: separate scalar output

Design principles:
  - Small model (hidden=128) to avoid overfitting with ~200 days of data
  - Aggressive dropout (0.3)
  - Ranking loss (ListMLE) for IC optimization
  - MSE as secondary loss
  - Early stopping with patience 10

Walk-Forward: 60d train, 10d val, slide 5d — SLIDING only (HC #0).
Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission).

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python \\
      alpha_discovery/lh_30min_deep_v1.py --device cuda --epochs 50 --hidden-dim 128

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
OUTPUT_DIR = ROOT / "output" / "lh_30min_deep_v1"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [DeepTab30m] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "lh_30min_deep_v1.log")),
    ],
)
log = logging.getLogger("DeepTab30m")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376
COST_RT_TICKS = 2.376  # market entry + market exit + RT commission

MIN_EDGE_TICKS = 2.5  # min tick move for 30min label

# Sequence length: look back 8 bars of 30-min = 4 hours of context
SEQ_LEN = 8

# ─────────────────────────────────────────────
#  PYTORCH IMPORTS (deferred)
# ─────────────────────────────────────────────
torch = None
nn = None
F = None


def _import_torch():
    global torch, nn, F
    if torch is not None:
        return
    import torch as _torch
    import torch.nn as _nn
    import torch.nn.functional as _F
    torch = _torch
    nn = _nn
    F = _F


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1: DATA LOADING (reused from v4_focused)
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
#  SECTION 2: BAR AGGREGATION + FEATURES (30-min bars from v4)
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
    """
    Aggregate 1-minute bars into N-minute bars with microstructure features.
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
    queue_df["bar_key"] = queue_df[ts_col].dt.floor("30min")

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
                df["sweep_minutes"].rolling(w, min_periods=1).sum() / (w * 30)
            )

    # Lookback returns at various horizons (in bar units)
    for bars, label in [(2, "lb_1h"), (4, "lb_2h"), (8, "lb_4h")]:
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
#  SECTION 4: FORWARD LABELS
# ═══════════════════════════════════════════════════════════════════


def add_forward_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Compute 30-min forward return labels with overnight gap protection."""
    df = df.sort_values("ts").reset_index(drop=True)

    horizon_bars = 1  # 1 bar of 30min = 30min forward
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

    df["fwd_return_30min"] = fwd_return
    df["fwd_ticks_30min"] = fwd_ticks

    # Directional label
    df["direction_30min"] = 0
    df.loc[fwd_ticks > MIN_EDGE_TICKS, "direction_30min"] = 1
    df.loc[fwd_ticks < -MIN_EDGE_TICKS, "direction_30min"] = -1

    df["trade_quality_30min"] = (
        fwd_ticks.abs() > 2 * COST_RT_TICKS
    ).astype(np.float32)

    log.info(
        f"Forward labels (30min): "
        f"{(~fwd_ticks.isna()).sum():,} valid, "
        f"{(df['direction_30min'] == 1).sum():,} long signals, "
        f"{(df['direction_30min'] == -1).sum():,} short signals"
    )
    return df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: LEAKAGE AUDIT
# ═══════════════════════════════════════════════════════════════════


def leakage_audit(
    train_dates: List[str],
    val_dates: List[str],
    feature_cols: List[str],
) -> Dict[str, Any]:
    """Explicit leakage audit."""
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

    all_passed = all(results.values())
    results["all_passed"] = all_passed
    if not all_passed:
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
#  SECTION 7: PyTorch MODEL — DeepTab30min
# ═══════════════════════════════════════════════════════════════════


def build_model(n_features: int, hidden_dim: int, dropout: float, device: str):
    """Build and return the DeepTab30min model."""
    _import_torch()

    class SparseAttentionBlock(nn.Module):
        """TabNet-inspired sparse feature attention for a single bar."""

        def __init__(self, n_features, hidden_dim, dropout):
            super().__init__()
            self.fc_shared = nn.Linear(n_features, hidden_dim)
            self.bn_shared = nn.BatchNorm1d(hidden_dim)

            # Attention: learns which features matter for this bar
            self.attn_fc = nn.Linear(hidden_dim, n_features)
            self.attn_bn = nn.BatchNorm1d(n_features)

            # After attention masking, project back to hidden
            self.fc_out = nn.Linear(n_features, hidden_dim)
            self.bn_out = nn.BatchNorm1d(hidden_dim)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            # x: (batch, n_features)
            h = self.fc_shared(x)
            h = self.bn_shared(h)
            h = F.relu(h)

            # Sparse attention mask
            attn = self.attn_fc(h)
            attn = self.attn_bn(attn)
            attn = F.softmax(attn, dim=-1)  # sparse-ish selection over features

            # Apply attention to original features
            masked = x * attn
            out = self.fc_out(masked)
            out = self.bn_out(out)
            out = F.relu(out)
            out = self.dropout(out)
            return out  # (batch, hidden_dim)

    class DeepTab30min(nn.Module):
        """
        DeepTab with temporal context for 30-min prediction.

        Input: (batch, seq_len=8, n_features) — last 8 bars of features
        Output: (batch, 1) prediction + (batch, 1) confidence
        """

        def __init__(self, n_features, hidden_dim, seq_len, dropout):
            super().__init__()
            self.seq_len = seq_len
            self.n_features = n_features
            self.hidden_dim = hidden_dim

            # Per-bar sparse attention (shared weights across time steps)
            self.bar_encoder = SparseAttentionBlock(n_features, hidden_dim, dropout)

            # Temporal: 1-layer GRU captures sequence patterns
            self.gru = nn.GRU(
                input_size=hidden_dim,
                hidden_size=hidden_dim,
                num_layers=1,
                batch_first=True,
                dropout=0,  # single layer, no internal dropout
            )
            self.gru_dropout = nn.Dropout(dropout)

            # Prediction head
            self.pred_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 2),
                nn.BatchNorm1d(hidden_dim // 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 2, 1),
            )

            # Confidence head (separate — learns when prediction is reliable)
            self.conf_head = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim // 4),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim // 4, 1),
                nn.Sigmoid(),
            )

        def forward(self, x):
            """
            x: (batch, seq_len, n_features)
            Returns: pred (batch, 1), conf (batch, 1)
            """
            batch_size = x.size(0)

            # Encode each bar through shared sparse attention
            # Reshape to (batch * seq_len, n_features)
            x_flat = x.view(batch_size * self.seq_len, self.n_features)
            bar_encoded = self.bar_encoder(x_flat)  # (batch * seq_len, hidden_dim)
            bar_encoded = bar_encoded.view(batch_size, self.seq_len, self.hidden_dim)

            # GRU over the sequence
            gru_out, _ = self.gru(bar_encoded)  # (batch, seq_len, hidden_dim)
            # Use last hidden state
            last_hidden = gru_out[:, -1, :]  # (batch, hidden_dim)
            last_hidden = self.gru_dropout(last_hidden)

            # Prediction and confidence
            pred = self.pred_head(last_hidden)  # (batch, 1)
            conf = self.conf_head(last_hidden)  # (batch, 1)

            return pred, conf

    model = DeepTab30min(n_features, hidden_dim, SEQ_LEN, dropout)
    model = model.to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    log.info(f"DeepTab30min: {n_params:,} trainable parameters, device={device}")
    return model


# ═══════════════════════════════════════════════════════════════════
#  SECTION 8: LOSS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════


def listmle_loss(y_pred, y_true):
    """
    ListMLE ranking loss — optimizes for IC directly.
    Differentiable surrogate for Spearman correlation.
    """
    _import_torch()

    # Sort by true values (descending)
    _, indices = torch.sort(y_true.squeeze(), descending=True)
    y_pred_sorted = y_pred.squeeze()[indices]

    # ListMLE: log-likelihood of the permutation
    n = y_pred_sorted.size(0)
    if n < 2:
        return torch.tensor(0.0, device=y_pred.device)

    # Compute cumulative logsumexp from the end
    max_val = y_pred_sorted.max()
    y_shifted = y_pred_sorted - max_val

    # For numerical stability, compute in reverse
    cumsum_exp = torch.zeros(n, device=y_pred.device)
    cumsum_exp[n - 1] = torch.exp(y_shifted[n - 1])
    for i in range(n - 2, -1, -1):
        cumsum_exp[i] = cumsum_exp[i + 1] + torch.exp(y_shifted[i])

    log_cumsum = torch.log(cumsum_exp + 1e-10) + max_val
    loss = -torch.mean(y_pred_sorted - log_cumsum)
    return loss


def combined_loss(y_pred, y_true, conf, alpha_rank=0.5, alpha_conf=0.1):
    """
    Combined loss:
    - MSE for magnitude prediction
    - ListMLE for ranking (IC optimization)
    - Confidence calibration: penalize high confidence on wrong predictions
    """
    _import_torch()

    # MSE loss
    mse = F.mse_loss(y_pred.squeeze(), y_true.squeeze())

    # ListMLE ranking loss
    rank_loss = listmle_loss(y_pred, y_true)

    # Confidence calibration loss
    # High confidence should correlate with low prediction error
    pred_error = (y_pred.squeeze() - y_true.squeeze()).abs()
    # Normalize error to [0, 1] range roughly
    error_norm = pred_error / (pred_error.max() + 1e-8)
    # Confidence should be LOW when error is HIGH
    conf_loss = F.mse_loss(conf.squeeze(), 1.0 - error_norm.detach())

    total = (1 - alpha_rank) * mse + alpha_rank * rank_loss + alpha_conf * conf_loss
    return total, mse.item(), rank_loss.item(), conf_loss.item()


# ═══════════════════════════════════════════════════════════════════
#  SECTION 9: SEQUENCE DATASET
# ═══════════════════════════════════════════════════════════════════


def build_sequences(
    features: np.ndarray,
    labels: np.ndarray,
    dates: np.ndarray,
    seq_len: int = SEQ_LEN,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Build sequences of (seq_len, n_features) for each bar.
    Each sample at time t uses features from [t-seq_len+1, ..., t].
    The label is for bar t (forward return from bar t).

    CAUSAL: only uses past bars. No leakage.
    Skips bars where we don't have enough history or label is NaN.
    Also skips sequences that cross overnight gaps (date changes within sequence).
    """
    n_samples, n_features = features.shape
    sequences = []
    seq_labels = []
    seq_dates = []

    for i in range(seq_len - 1, n_samples):
        if np.isnan(labels[i]):
            continue

        # Check that all bars in sequence are from a contiguous period
        # (no overnight gaps within the 8-bar lookback)
        seq_dates_window = dates[i - seq_len + 1: i + 1]
        # Allow at most 1 unique date in a 4-hour window (8 bars of 30min)
        # Actually, we might span 2 days at the open. Allow 2 unique dates max.
        unique_dates = len(set(seq_dates_window))
        if unique_dates > 2:
            continue

        seq = features[i - seq_len + 1: i + 1]  # (seq_len, n_features)

        # Check for NaN in features
        if np.any(np.isnan(seq)):
            seq = np.nan_to_num(seq, nan=0.0, posinf=3.0, neginf=-3.0)

        sequences.append(seq)
        seq_labels.append(labels[i])
        seq_dates.append(dates[i])

    if not sequences:
        return np.array([]), np.array([]), np.array([])

    return (
        np.array(sequences, dtype=np.float32),
        np.array(seq_labels, dtype=np.float32),
        np.array(seq_dates),
    )


# ═══════════════════════════════════════════════════════════════════
#  SECTION 10: TRAINING LOOP
# ═══════════════════════════════════════════════════════════════════


def train_one_fold(
    X_train_seq: np.ndarray,
    y_train: np.ndarray,
    X_val_seq: np.ndarray,
    y_val: np.ndarray,
    n_features: int,
    args,
    fold_idx: int,
) -> Tuple[Optional[Any], np.ndarray, np.ndarray]:
    """
    Train the DeepTab30min model for one fold.
    Returns: (model_state_dict, val_preds, val_confidences) or (None, nan, nan) on failure.
    """
    _import_torch()

    device = args.device
    if device == "cuda" and not torch.cuda.is_available():
        log.warning("CUDA not available, falling back to CPU")
        device = "cpu"

    if len(X_train_seq) < 100 or len(X_val_seq) < 20:
        log.warning(f"  Fold {fold_idx}: too few samples (train={len(X_train_seq)}, val={len(X_val_seq)}) -- skip")
        return None, np.full(len(y_val), np.nan), np.full(len(y_val), np.nan)

    model = build_model(n_features, args.hidden_dim, args.dropout, device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.lr * 0.01
    )

    # Convert to tensors
    X_train_t = torch.tensor(X_train_seq, dtype=torch.float32, device=device)
    y_train_t = torch.tensor(y_train, dtype=torch.float32, device=device).unsqueeze(1)
    X_val_t = torch.tensor(X_val_seq, dtype=torch.float32, device=device)
    y_val_t = torch.tensor(y_val, dtype=torch.float32, device=device).unsqueeze(1)

    batch_size = min(args.batch_size, len(X_train_seq))
    n_batches = (len(X_train_seq) + batch_size - 1) // batch_size

    best_val_loss = float("inf")
    best_val_ic = -1.0
    best_state = None
    patience_counter = 0

    for epoch in range(args.epochs):
        model.train()
        # Shuffle training data
        perm = torch.randperm(len(X_train_t), device=device)
        X_train_shuf = X_train_t[perm]
        y_train_shuf = y_train_t[perm]

        epoch_mse = 0.0
        epoch_rank = 0.0
        epoch_conf = 0.0
        n_batch_actual = 0

        for b in range(n_batches):
            start = b * batch_size
            end = min(start + batch_size, len(X_train_shuf))
            if end - start < 8:
                continue

            xb = X_train_shuf[start:end]
            yb = y_train_shuf[start:end]

            pred, conf = model(xb)
            loss, mse_v, rank_v, conf_v = combined_loss(
                pred, yb, conf,
                alpha_rank=args.alpha_rank,
                alpha_conf=args.alpha_conf,
            )

            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            epoch_mse += mse_v
            epoch_rank += rank_v
            epoch_conf += conf_v
            n_batch_actual += 1

        scheduler.step()

        # Validation
        model.eval()
        with torch.no_grad():
            val_pred, val_conf = model(X_val_t)
            val_loss, val_mse, val_rank, val_conf_loss = combined_loss(
                val_pred, y_val_t, val_conf,
                alpha_rank=args.alpha_rank,
                alpha_conf=args.alpha_conf,
            )

            val_pred_np = val_pred.squeeze().cpu().numpy()
            val_conf_np = val_conf.squeeze().cpu().numpy()
            y_val_np = y_val

            # Compute IC
            valid_mask = ~np.isnan(val_pred_np) & ~np.isnan(y_val_np)
            if valid_mask.sum() > 5:
                ic = np.corrcoef(val_pred_np[valid_mask], y_val_np[valid_mask])[0, 1]
            else:
                ic = 0.0

        # Early stopping on IC (primary) or val_loss (fallback)
        improved = False
        if ic > best_val_ic + 0.005:  # IC must improve by at least 0.005
            best_val_ic = ic
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = True
        elif val_loss.item() < best_val_loss - 1e-4:
            best_val_loss = val_loss.item()
            if best_state is None:  # only update if no IC-based best yet
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            improved = True
        else:
            patience_counter += 1

        if (epoch + 1) % 5 == 0 or improved or epoch == 0:
            avg_mse = epoch_mse / max(n_batch_actual, 1)
            log.info(
                f"  Fold {fold_idx} epoch {epoch+1}/{args.epochs}: "
                f"train_mse={avg_mse:.6f}, val_IC={ic:.4f}, "
                f"val_loss={val_loss.item():.6f}, best_IC={best_val_ic:.4f}, "
                f"patience={patience_counter}/{args.patience}"
            )

        if patience_counter >= args.patience:
            log.info(f"  Fold {fold_idx}: early stopping at epoch {epoch+1}")
            break

    # Restore best model and predict
    if best_state is not None:
        model.load_state_dict(best_state)
        model = model.to(device)

    model.eval()
    with torch.no_grad():
        final_pred, final_conf = model(X_val_t)
        val_preds = final_pred.squeeze().cpu().numpy()
        val_confs = final_conf.squeeze().cpu().numpy()

    log.info(
        f"  Fold {fold_idx} DONE: best_IC={best_val_ic:.4f}, "
        f"stopped_epoch={epoch+1}, val_samples={len(y_val)}"
    )

    return best_state, val_preds, val_confs


# ═══════════════════════════════════════════════════════════════════
#  SECTION 11: TRADE SIMULATION
# ═══════════════════════════════════════════════════════════════════


def simulate_trades(
    preds: np.ndarray,
    actuals: np.ndarray,
    dates: np.ndarray,
    confidence_pct: float = 0.10,
    cost_ticks: float = COST_RT_TICKS,
    confidences: Optional[np.ndarray] = None,
    use_confidence_gate: bool = False,
) -> Optional[Dict]:
    """
    Simulate trades with per-side and per-day reporting.

    If use_confidence_gate=True and confidences are provided,
    only take trades where model confidence > median.
    """
    valid = ~np.isnan(preds) & ~np.isnan(actuals)
    preds_v = preds[valid]
    actuals_v = actuals[valid]
    dates_v = dates[valid]
    conf_v = confidences[valid] if confidences is not None else None

    if len(preds_v) < 20:
        return None

    # Optional confidence gate
    if use_confidence_gate and conf_v is not None:
        conf_gate = conf_v > np.median(conf_v)
    else:
        conf_gate = np.ones(len(preds_v), dtype=bool)

    trades = []
    upper = np.quantile(preds_v, 1 - confidence_pct)
    lower = np.quantile(preds_v, confidence_pct)

    for i in range(len(preds_v)):
        if not conf_gate[i]:
            continue
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
#  SECTION 12: REGIME STRATIFICATION (HC #428 R1)
# ═══════════════════════════════════════════════════════════════════


def regime_stratification(
    preds: np.ndarray,
    actuals: np.ndarray,
    dates: np.ndarray,
    day_returns: Dict[str, float],
    confidence_pct: float = 0.10,
) -> Dict[str, Any]:
    """
    Stratify by day-regime (green/red/flat).
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
            f"gap={gap:.2f} {'PASS' if gap <= 0.50 else 'FAIL'} (threshold 0.50)"
        )
    else:
        results["regime_gap"] = float("nan")
        results["regime_gap_pass"] = False
        results["regime_gap_detail"] = "insufficient regime data"

    return results


# ═══════════════════════════════════════════════════════════════════
#  SECTION 13: WALK-FORWARD ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="30-Min DeepTab v1 — Temporal Deep Model")
    parser.add_argument("--device", default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--epochs", type=int, default=50, help="Max epochs per fold")
    parser.add_argument("--hidden-dim", type=int, default=128, help="Hidden dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout rate")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    parser.add_argument("--alpha-rank", type=float, default=0.5, help="Weight for ranking loss")
    parser.add_argument("--alpha-conf", type=float, default=0.1, help="Weight for confidence loss")
    parser.add_argument("--train-days", type=int, default=60, help="Training window (days)")
    parser.add_argument("--val-days", type=int, default=10, help="Validation window (days)")
    parser.add_argument("--slide-days", type=int, default=5, help="Slide step (days)")
    parser.add_argument(
        "--mlflow-uri", default="http://neptune:5000",
        help="MLflow tracking URI",
    )
    parser.add_argument(
        "--experiment-name", default="lh_30min_deep_v1",
        help="MLflow experiment name",
    )
    parser.add_argument("--no-mlflow", action="store_true", help="Disable MLflow")
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("30-MINUTE DEEP TABULAR MODEL v1 — DeepTab with Temporal Context")
    log.info("=" * 70)
    log.info(f"Device: {args.device}")
    log.info(f"Architecture: SparseAttention + GRU (seq_len={SEQ_LEN})")
    log.info(f"Hidden dim: {args.hidden_dim}, Dropout: {args.dropout}")
    log.info(f"Loss: {1-args.alpha_rank:.0%} MSE + {args.alpha_rank:.0%} ListMLE + {args.alpha_conf:.0%} Confidence")
    log.info(f"Walk-forward: {args.train_days}d train, {args.val_days}d val, slide {args.slide_days}d")
    log.info(f"Cost: {COST_RT_TICKS} ticks RT")
    log.info(f"Output: {OUTPUT_DIR}")

    _import_torch()
    t0 = time.time()

    # ── Load data ──
    log.info("\n--- Loading data ---")
    minute_df = load_all_minute_bars()
    queue_df = load_queue_features()

    # ── Aggregate to 30-min bars ──
    log.info("\n--- Aggregating to 30-min bars ---")
    bars_df = aggregate_to_bars(minute_df, bar_size_min=30)

    # ── Add queue features ──
    bars_df = add_queue_features_to_bars(bars_df, queue_df)

    # ── Rolling + regime features ──
    bars_df = add_rolling_features(bars_df)

    # ── Forward labels ──
    bars_df = add_forward_labels(bars_df)

    # ── Feature selection ──
    feature_cols = get_feature_columns(bars_df)
    n_features = len(feature_cols)
    log.info(f"Feature columns ({n_features}): {feature_cols[:15]}...")

    # ── Walk-forward setup ──
    dates = sorted(bars_df["date"].unique())
    log.info(f"Total trading days: {len(dates)} ({dates[0]} -> {dates[-1]})")

    train_days = args.train_days
    val_days = args.val_days
    slide = args.slide_days

    if len(dates) < train_days + val_days + slide:
        log.error(f"Not enough days ({len(dates)}) for {train_days}+{val_days} walk-forward")
        return

    # Prepare arrays
    features_all = bars_df[feature_cols].values.astype(np.float32)
    labels_all = bars_df["fwd_ticks_30min"].values.astype(np.float32)
    dates_all = bars_df["date"].values
    ts_all = bars_df["ts"].values

    # Day returns for regime classification
    day_close = bars_df.groupby("date")["close"].last()
    day_returns_raw = day_close.pct_change()
    day_returns = day_returns_raw.to_dict()

    # ── MLflow setup ──
    mlflow_run = None
    use_mlflow = not args.no_mlflow
    if use_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(args.mlflow_uri)
            mlflow.set_experiment(args.experiment_name)
            mlflow_run = mlflow.start_run(
                run_name=f"deeptab30m_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            )
            mlflow.log_params({
                "model": "DeepTab30min",
                "architecture": "SparseAttention+GRU",
                "seq_len": SEQ_LEN,
                "hidden_dim": args.hidden_dim,
                "dropout": args.dropout,
                "lr": args.lr,
                "weight_decay": args.weight_decay,
                "batch_size": args.batch_size,
                "epochs_max": args.epochs,
                "patience": args.patience,
                "alpha_rank": args.alpha_rank,
                "alpha_conf": args.alpha_conf,
                "train_days": train_days,
                "val_days": val_days,
                "slide_days": slide,
                "cost_rt_ticks": COST_RT_TICKS,
                "n_features": n_features,
                "n_total_days": len(dates),
            })
            log.info(f"MLflow run started: {mlflow_run.info.run_id}")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e} -- continuing without tracking")
            use_mlflow = False

    # ── Walk-Forward Loop ──
    all_oot_preds = []
    all_oot_actuals = []
    all_oot_dates = []
    all_oot_confs = []
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
            f"FOLD {fold_idx}: "
            f"train {fold_train_dates[0]}->{fold_train_dates[-1]} ({len(fold_train_dates)}d), "
            f"val {fold_val_dates[0]}->{fold_val_dates[-1]} ({len(fold_val_dates)}d)"
            f"\n{'=' * 60}"
        )

        # ── Leakage audit ──
        audit = leakage_audit(
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
        train_labels = labels_all[train_mask]
        val_labels = labels_all[val_mask]
        train_dates_fold = dates_all[train_mask]
        val_dates_fold = dates_all[val_mask]

        # ── Robust scaling: fit on train only ──
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

        # ── Build sequences ──
        X_train_seq, y_train_seq, d_train_seq = build_sequences(
            train_features, train_labels, train_dates_fold, seq_len=SEQ_LEN
        )
        X_val_seq, y_val_seq, d_val_seq = build_sequences(
            val_features, val_labels, val_dates_fold, seq_len=SEQ_LEN
        )

        log.info(
            f"  Sequences: train={len(X_train_seq)}, val={len(X_val_seq)} "
            f"(seq_len={SEQ_LEN}, n_features={n_features})"
        )

        if len(X_train_seq) < 100 or len(X_val_seq) < 10:
            log.warning(f"  Fold {fold_idx}: insufficient sequences -- skip")
            continue

        # ── Train model ──
        best_state, val_preds, val_confs = train_one_fold(
            X_train_seq, y_train_seq,
            X_val_seq, y_val_seq,
            n_features, args, fold_idx,
        )

        if best_state is None:
            continue

        # ── Fold IC ──
        valid_v = ~np.isnan(val_preds) & ~np.isnan(y_val_seq)
        if valid_v.sum() > 5:
            ic = np.corrcoef(val_preds[valid_v], y_val_seq[valid_v])[0, 1]
            dir_acc = np.mean(
                np.sign(val_preds[valid_v]) == np.sign(y_val_seq[valid_v])
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
            "train_sequences": int(len(X_train_seq)),
            "val_sequences": int(len(X_val_seq)),
            "ic": float(ic) if not np.isnan(ic) else None,
            "dir_acc": float(dir_acc) if not np.isnan(dir_acc) else None,
            "mean_confidence": float(np.mean(val_confs[valid_v])) if valid_v.sum() > 0 else None,
        }
        all_fold_results.append(fold_result)

        # Log to MLflow per fold
        if use_mlflow:
            try:
                import mlflow
                mlflow.log_metrics({
                    f"fold_{fold_idx}_ic": float(ic) if not np.isnan(ic) else 0,
                    f"fold_{fold_idx}_dir_acc": float(dir_acc) if not np.isnan(dir_acc) else 0,
                }, step=fold_idx)
            except Exception:
                pass

        # ── Accumulate OOT ──
        all_oot_preds.append(val_preds[valid_v])
        all_oot_actuals.append(y_val_seq[valid_v])
        all_oot_dates.append(d_val_seq[valid_v])
        all_oot_confs.append(val_confs[valid_v])

        # Save fold
        np.savez_compressed(
            str(OUTPUT_DIR / f"fold_{fold_idx:03d}.npz"),
            preds=val_preds,
            confs=val_confs,
            actuals=y_val_seq,
            dates=d_val_seq,
            feature_cols=feature_cols,
        )

        # Save model weights
        torch.save(best_state, str(OUTPUT_DIR / f"fold_{fold_idx:03d}_model.pt"))

        log.info(
            f"  Fold {fold_idx} result: IC={ic:.4f}, dir_acc={dir_acc:.3f}, "
            f"n_valid={valid_v.sum()}"
        )

        gc.collect()
        if args.device == "cuda" and torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ═══════════════════════════════════════════════════════════════
    #  CONCAT OOT ANALYSIS
    # ═══════════════════════════════════════════════════════════════

    if not all_oot_preds:
        log.error("No valid folds completed!")
        return

    concat_preds = np.concatenate(all_oot_preds)
    concat_actuals = np.concatenate(all_oot_actuals)
    concat_dates = np.concatenate(all_oot_dates)
    concat_confs = np.concatenate(all_oot_confs)

    # Save concat OOT
    np.savez_compressed(
        str(OUTPUT_DIR / "concat_oot.npz"),
        preds=concat_preds,
        actuals=concat_actuals,
        dates=concat_dates,
        confs=concat_confs,
    )

    # ── Concat metrics ──
    concat_ic = np.corrcoef(concat_preds, concat_actuals)[0, 1]
    fold_ics = [f["ic"] for f in all_fold_results if f["ic"] is not None]
    ic_sharpe = (
        np.mean(fold_ics) / max(np.std(fold_ics), 1e-6)
        if len(fold_ics) > 2 else float("nan")
    )
    concat_dir_acc = np.mean(np.sign(concat_preds) == np.sign(concat_actuals))

    log.info(
        f"\n{'=' * 60}\n"
        f"CONCAT OOT: "
        f"IC={concat_ic:.4f}, IC_Sharpe={ic_sharpe:.3f}, "
        f"dir_acc={concat_dir_acc:.3f}, n={len(concat_preds):,}"
        f"\n{'=' * 60}"
    )

    # ── Trade simulation at multiple thresholds ──
    trade_results = {}
    for conf_pct in [0.05, 0.10, 0.15, 0.20, 0.30]:
        # Without confidence gate
        sim = simulate_trades(
            concat_preds, concat_actuals, concat_dates,
            confidence_pct=conf_pct,
        )
        if sim is not None:
            trade_results[f"top_{int(conf_pct*100)}pct"] = sim
            log.info(
                f"  Trades (top {int(conf_pct*100)}%): "
                f"n={sim['n_trades']}, Sharpe={sim['sharpe']:.2f}, "
                f"Sortino={sim['sortino']:.2f}, WR={sim['win_rate']:.1%}, "
                f"PF={sim['profit_factor']:.2f}, avg_pnl={sim['avg_pnl_ticks']:.2f}t"
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

        # With confidence gate
        sim_gated = simulate_trades(
            concat_preds, concat_actuals, concat_dates,
            confidence_pct=conf_pct,
            confidences=concat_confs,
            use_confidence_gate=True,
        )
        if sim_gated is not None:
            trade_results[f"top_{int(conf_pct*100)}pct_conf_gated"] = sim_gated
            log.info(
                f"  Trades (top {int(conf_pct*100)}% + conf gate): "
                f"n={sim_gated['n_trades']}, Sharpe={sim_gated['sharpe']:.2f}, "
                f"WR={sim_gated['win_rate']:.1%}"
            )

    # ── Regime stratification (HC #428 R1) ──
    regime_results = regime_stratification(
        concat_preds, concat_actuals, concat_dates,
        day_returns=day_returns,
        confidence_pct=0.15,  # use 15% since LightGBM was best at 15%
    )
    log.info(f"\n  Regime analysis: {regime_results.get('regime_gap_detail', 'N/A')}")

    # ── Per-day IC analysis ──
    unique_dates = sorted(set(concat_dates))
    per_day_ics = []
    for d in unique_dates:
        d_mask = concat_dates == d
        if d_mask.sum() < 3:
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

    green_ics = [x["ic"] for x in per_day_ics if x["regime"] == "green"]
    red_ics = [x["ic"] for x in per_day_ics if x["regime"] == "red"]
    flat_ics = [x["ic"] for x in per_day_ics if x["regime"] == "flat"]

    log.info(
        f"  Per-day IC: green={np.mean(green_ics):.4f} ({len(green_ics)}d), "
        f"red={np.mean(red_ics):.4f} ({len(red_ics)}d), "
        f"flat={np.mean(flat_ics):.4f} ({len(flat_ics)}d)"
    )

    # ── MLflow final metrics ──
    if use_mlflow:
        try:
            import mlflow
            mlflow.log_metrics({
                "concat_ic": float(concat_ic),
                "ic_sharpe": float(ic_sharpe) if not np.isnan(ic_sharpe) else 0,
                "concat_dir_acc": float(concat_dir_acc),
                "n_predictions": len(concat_preds),
                "n_oot_days": len(unique_dates),
                "n_folds": fold_idx,
                "regime_gap": float(regime_results.get("regime_gap", 0)),
                "regime_gap_pass": int(regime_results.get("regime_gap_pass", False)),
            })

            # Best trade sim
            if trade_results:
                best_key = max(
                    [k for k in trade_results.keys() if "conf_gated" not in k],
                    key=lambda k: trade_results[k].get("sharpe", -999),
                    default=None,
                )
                if best_key:
                    best = trade_results[best_key]
                    mlflow.log_metrics({
                        "best_sharpe": best["sharpe"],
                        "best_sortino": best["sortino"],
                        "best_wr": best["win_rate"],
                        "best_pf": best["profit_factor"],
                        "best_n_trades": best["n_trades"],
                    })
        except Exception as e:
            log.warning(f"MLflow logging failed: {e}")

    # ── Save summary JSON ──
    summary = {
        "model": "DeepTab30min",
        "architecture": "SparseAttention+GRU",
        "seq_len": SEQ_LEN,
        "hidden_dim": args.hidden_dim,
        "n_features": n_features,
        "n_folds": fold_idx,
        "n_valid_folds": len(all_fold_results),
        "concat_oot": {
            "ic": float(concat_ic),
            "ic_sharpe": float(ic_sharpe) if not np.isnan(ic_sharpe) else None,
            "dir_acc": float(concat_dir_acc),
            "n_predictions": int(len(concat_preds)),
            "n_oot_days": len(unique_dates),
        },
        "trade_sims": trade_results,
        "regime": regime_results,
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
        "comparison_target": "LightGBM IC=0.117, IC_Sharpe=1.044, Sharpe_15pct=2.55",
        "runtime_minutes": (time.time() - t0) / 60,
    }

    summary_path = OUTPUT_DIR / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"\nSummary saved to {summary_path}")

    # ── End MLflow ──
    if use_mlflow and mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_artifact(str(summary_path))
            mlflow.end_run()
        except Exception as e:
            log.warning(f"MLflow end_run failed: {e}")

    elapsed = time.time() - t0

    # ── Final summary ──
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY — DeepTab30min v1")
    log.info("=" * 70)
    log.info(f"  Architecture: SparseAttention + GRU (seq_len={SEQ_LEN})")
    log.info(f"  Hidden dim: {args.hidden_dim}, Features: {n_features}")
    log.info(f"  Concat IC: {concat_ic:.4f} (LightGBM baseline: 0.117)")
    log.info(f"  IC_Sharpe: {ic_sharpe:.3f} (LightGBM baseline: 1.044)")
    log.info(f"  Dir accuracy: {concat_dir_acc:.3f}")
    log.info(f"  OOT days: {len(unique_dates)}, Predictions: {len(concat_preds):,}")

    if trade_results:
        for thresh_key, sim in trade_results.items():
            if "conf_gated" in thresh_key:
                continue
            log.info(
                f"  {thresh_key}: n={sim['n_trades']}, "
                f"Sharpe={sim['sharpe']:.2f}, Sortino={sim['sortino']:.2f}, "
                f"WR={sim['win_rate']:.1%}, PF={sim['profit_factor']:.2f}, "
                f"avg={sim['avg_pnl_ticks']:.2f}t, total=${sim['total_pnl_dollars']:.0f}"
            )

    log.info(f"  Regime: {regime_results.get('regime_gap_detail', 'N/A')}")
    log.info(f"  Runtime: {elapsed / 60:.1f} minutes")

    # Compare to LightGBM
    lgbm_ic = 0.117
    improvement = (concat_ic - lgbm_ic) / lgbm_ic * 100
    log.info(f"\n  vs LightGBM: IC {'improved' if concat_ic > lgbm_ic else 'regressed'} "
             f"by {improvement:+.1f}% ({lgbm_ic:.3f} -> {concat_ic:.4f})")

    log.info("\n" + "=" * 70)
    log.info("DONE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
