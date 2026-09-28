#!/usr/bin/env python3
"""
Meta-Confluence v1 — Multi-Signal Gating & Sizing Model (HC #646 R2)
=====================================================================

PROBLEM: We have multiple independent ES futures signals:
  - Short-horizon (30min): LightGBM on OFI/flow, Sharpe 2.49, 560 trades / 31 days
  - Long-horizon (4h): LightGBM on 3-day OFI, Sharpe 1.84, 150 trades / 197 days
  - Portfolio ensemble: simple allocation, Sharpe 1.96

HC #646 R2: "Don't enter on a single signal. Multiple independent signals must
agree — model prediction, orderflow confirmation, regime alignment, volatility
conditions. More confluence = higher conviction = better sizing."

GOAL: Build a meta-model that learns WHEN all conditions align for the best
trades. Instead of simple portfolio allocation, learn the optimal gating
and sizing function across 4 independent sub-signals.

ARCHITECTURE:
  1. Build daily feature set from minute bars (OFI, momentum, vol, spread)
  2. 4 independent signal sub-models (walk-forward, 40d sliding):
     A: 1-day direction  B: 3-day direction
     C: Volatility regime  D: Momentum continuation
  3. Meta-model (LightGBM) on signal probs + agreement + regime features
  4. Trading simulation: entry when meta confidence > threshold, 4h hold,
     80-tick trailing stop, FIFO costs

HC #0: SLIDING windows only.
HC #428: Regime-agnostic OOT validation (gap ≤ 0.50, day-conc ≤ 0.70).
FIFO costs: 0.376t passive entry, 1.376t market exit.

Usage:
  cd /home/nick/Lvl3Quant && \\
  /home/nick/miniconda3/envs/py311-train/bin/python \\
      alpha_discovery/meta_confluence_v1.py
"""

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

warnings.filterwarnings("ignore")

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
OUTPUT_DIR = ROOT / "output" / "meta_confluence_v1"
LOG_DIR = ROOT / "logs"

MINUTE_BARS_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
DAILY_FEATURES_V1 = ROOT / "output" / "long_horizon_trading_v1" / "daily_features.parquet"
ENHANCED_FEATURES = ROOT / "output" / "long_horizon_flow_v2" / "enhanced_daily_features.parquet"
V1_TRADES_PATH = ROOT / "output" / "long_horizon_trading_v1" / "best_intraday_trades.parquet"

for d in [OUTPUT_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [META-CONF] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_DIR / "meta_confluence_v1.log"), mode="w"),
    ],
)
log = logging.getLogger("META-CONF")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
TICK_SIZE = 0.25
COST_PASSIVE_TICKS = 0.376       # commission only (passive limit fill)
COST_MARKET_TICKS = 1.376        # commission + 1 tick spread crossing
TOTAL_RT_COST_TICKS = COST_PASSIVE_TICKS + COST_MARKET_TICKS  # 1.752 ticks

STOP_LOSS_TICKS = 80             # trailing stop
HOLD_MINUTES = 240               # 4h max hold

# Walk-forward config
WF_TRAIN_DAYS = 40               # HC #0: sliding window
WF_SLIDE = 1                     # slide by 1 day

# Meta-model confidence thresholds to test
META_THRESHOLDS = [0.50, 0.55, 0.60, 0.65, 0.70]

# Accumulation windows for OFI features
ACCUM_WINDOWS = [3, 5, 10, 20]

# ─────────────────────────────────────────────
#  JSON ENCODER
# ─────────────────────────────────────────────

class NumpyEncoder(json.JSONEncoder):
    """JSON encoder that handles numpy types."""
    def default(self, obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        if isinstance(obj, pd.Timestamp):
            return str(obj)
        return super().default(obj)


# ─────────────────────────────────────────────
#  COLUMN DETECTION HELPER
# ─────────────────────────────────────────────

def _find_col(df: pd.DataFrame, candidates: List[str]) -> Optional[str]:
    """Find first matching column name (case-insensitive fallback)."""
    for c in candidates:
        if c in df.columns:
            return c
    col_lower = {c.lower(): c for c in df.columns}
    for c in candidates:
        if c.lower() in col_lower:
            return col_lower[c.lower()]
    return None


def _get_price_cols(df: pd.DataFrame) -> Dict[str, Optional[str]]:
    """Get standard price/volume column names from a minute bar DataFrame."""
    return {
        "close": _find_col(df, ["close", "Close", "CLOSE"]),
        "open": _find_col(df, ["open", "Open", "OPEN"]),
        "high": _find_col(df, ["high", "High", "HIGH"]),
        "low": _find_col(df, ["low", "Low", "LOW"]),
        "volume": _find_col(df, ["volume", "Volume", "VOLUME", "vol", "Vol"]),
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 1: DAILY FEATURE ENGINEERING FROM MINUTE BARS
# ═══════════════════════════════════════════════════════════════════

def load_minute_bars(date_str: str) -> Optional[pd.DataFrame]:
    """Load minute-bar data for a specific date (YYYYMMDD)."""
    fpath = MINUTE_BARS_DIR / f"{date_str}.parquet"
    if not fpath.exists():
        return None
    try:
        return pd.read_parquet(fpath)
    except Exception as e:
        log.warning(f"Failed to load minute bars for {date_str}: {e}")
        return None


def compute_daily_features(df: pd.DataFrame, date_str: str) -> Optional[Dict[str, float]]:
    """
    Compute daily features from minute bars for one trading day.

    Features:
      - Morning/afternoon OFI (signed volume proxy)
      - Intraday momentum (close - open in ticks)
      - Volume profile (first half vs second half)
      - Realized volatility (5-min returns)
      - Spread proxy (high-low range / volume)
    """
    cols = _get_price_cols(df)
    close_col = cols["close"]
    open_col = cols["open"]
    high_col = cols["high"]
    low_col = cols["low"]
    vol_col = cols["volume"]

    if close_col is None:
        log.warning(f"{date_str}: No close column found")
        return None

    n_bars = len(df)
    if n_bars < 60:  # need at least 1 hour of data
        return None

    closes = df[close_col].values.astype(float)
    feats: Dict[str, float] = {}
    feats["date"] = date_str

    # ── 1. OFI proxy: signed volume ──
    # Use close-to-close direction * volume as OFI proxy
    if vol_col is not None:
        volumes = df[vol_col].values.astype(float)
        price_changes = np.diff(closes)
        signed_vol = np.sign(price_changes) * volumes[1:]

        # Replace NaN/inf with 0
        signed_vol = np.where(np.isfinite(signed_vol), signed_vol, 0.0)

        # Morning: first 120 bars (2 hours)
        morning_end = min(120, len(signed_vol))
        feats["morning_ofi"] = float(np.sum(signed_vol[:morning_end]))

        # Afternoon: last 120 bars (2 hours)
        afternoon_start = max(0, len(signed_vol) - 120)
        feats["afternoon_ofi"] = float(np.sum(signed_vol[afternoon_start:]))

        # Full day OFI
        feats["daily_ofi"] = float(np.sum(signed_vol))

        # Volume profile: first half vs second half
        mid = len(volumes) // 2
        first_half_vol = np.nansum(volumes[:mid])
        second_half_vol = np.nansum(volumes[mid:])
        denom = first_half_vol + second_half_vol
        feats["volume_profile_ratio"] = float(
            first_half_vol / denom if denom > 0 else 0.5
        )
        feats["total_volume"] = float(np.nansum(volumes))
    else:
        # No volume — use price changes only
        price_changes = np.diff(closes)
        morning_end = min(120, len(price_changes))
        afternoon_start = max(0, len(price_changes) - 120)
        feats["morning_ofi"] = float(np.sum(np.where(np.isfinite(price_changes[:morning_end]),
                                                      price_changes[:morning_end], 0.0)))
        feats["afternoon_ofi"] = float(np.sum(np.where(np.isfinite(price_changes[afternoon_start:]),
                                                        price_changes[afternoon_start:], 0.0)))
        feats["daily_ofi"] = float(np.sum(np.where(np.isfinite(price_changes),
                                                    price_changes, 0.0)))
        feats["volume_profile_ratio"] = 0.5
        feats["total_volume"] = 0.0

    # ── 2. Intraday momentum (close - open in ticks) ──
    if open_col is not None:
        day_open = float(df[open_col].iloc[0])
        day_close = float(closes[-1])
        feats["intraday_momentum_ticks"] = (day_close - day_open) / TICK_SIZE
    else:
        feats["intraday_momentum_ticks"] = (closes[-1] - closes[0]) / TICK_SIZE

    # ── 3. Realized volatility (5-min returns) ──
    # Subsample every 5 bars for 5-min returns
    subsample_idx = np.arange(0, n_bars, 5)
    sub_closes = closes[subsample_idx]
    sub_closes = sub_closes[np.isfinite(sub_closes)]
    if len(sub_closes) > 5:
        rets_5m = np.diff(sub_closes) / sub_closes[:-1]
        rets_5m = rets_5m[np.isfinite(rets_5m)]
        feats["realized_vol"] = float(np.std(rets_5m)) if len(rets_5m) > 2 else 0.0
        feats["realized_vol_ann"] = feats["realized_vol"] * np.sqrt(252 * 78)  # ~78 5-min bars/day
    else:
        feats["realized_vol"] = 0.0
        feats["realized_vol_ann"] = 0.0

    # ── 4. Spread proxy: high-low range / volume ──
    if high_col is not None and low_col is not None:
        highs = df[high_col].values.astype(float)
        lows = df[low_col].values.astype(float)
        daily_range = float(np.nanmax(highs) - np.nanmin(lows)) / TICK_SIZE
        feats["daily_range_ticks"] = daily_range
        if feats["total_volume"] > 0:
            feats["spread_proxy"] = daily_range / feats["total_volume"]
        else:
            feats["spread_proxy"] = 0.0

        # Intrabar range average
        bar_ranges = (highs - lows) / TICK_SIZE
        bar_ranges = bar_ranges[np.isfinite(bar_ranges)]
        feats["avg_bar_range_ticks"] = float(np.mean(bar_ranges)) if len(bar_ranges) > 0 else 0.0
    else:
        feats["daily_range_ticks"] = 0.0
        feats["spread_proxy"] = 0.0
        feats["avg_bar_range_ticks"] = 0.0

    # ── 5. Close-to-close return (for next-day label building) ──
    feats["close_price"] = float(closes[-1])
    feats["open_price"] = float(closes[0]) if open_col is None else float(df[open_col].iloc[0])

    # ── 6. Additional flow features ──
    # Trend strength: abs(momentum) / range
    if feats["daily_range_ticks"] > 0:
        feats["trend_strength"] = abs(feats["intraday_momentum_ticks"]) / feats["daily_range_ticks"]
    else:
        feats["trend_strength"] = 0.0

    # Close location within range (0=low, 1=high)
    if high_col is not None and low_col is not None:
        day_high = float(np.nanmax(highs))
        day_low = float(np.nanmin(lows))
        rng = day_high - day_low
        if rng > 0:
            feats["close_location"] = (float(closes[-1]) - day_low) / rng
        else:
            feats["close_location"] = 0.5
    else:
        feats["close_location"] = 0.5

    return feats


def build_daily_feature_matrix(
    date_strs: List[str],
    existing_features: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build the full daily feature matrix from minute bars.
    Optionally merge with existing daily features (v1/v2).
    """
    log.info(f"Building daily features from {len(date_strs)} days of minute bars...")
    records = []
    for i, ds in enumerate(date_strs):
        bars = load_minute_bars(ds)
        if bars is None:
            continue
        feats = compute_daily_features(bars, ds)
        if feats is not None:
            records.append(feats)
        del bars
        if (i + 1) % 50 == 0:
            log.info(f"  Processed {i+1}/{len(date_strs)} days")

    if not records:
        raise ValueError("No daily features computed — check minute bar data")

    df = pd.DataFrame(records)
    log.info(f"Computed features for {len(df)} days, {len(df.columns)} columns")

    # Add accumulated features (3d, 5d, 10d, 20d cumulative OFI)
    df = df.sort_values("date").reset_index(drop=True)
    for w in ACCUM_WINDOWS:
        for col in ["daily_ofi", "morning_ofi", "afternoon_ofi", "intraday_momentum_ticks"]:
            if col in df.columns:
                df[f"{col}_cum{w}d"] = df[col].rolling(window=w, min_periods=1).sum()
        # Rolling vol
        if "realized_vol" in df.columns:
            df[f"realized_vol_mean{w}d"] = df["realized_vol"].rolling(window=w, min_periods=1).mean()

    # Merge with existing features if available
    if existing_features is not None:
        log.info(f"Merging with existing features ({len(existing_features)} rows, "
                 f"{len(existing_features.columns)} cols)")
        # Find date column in existing
        date_col = _find_col(existing_features, ["date", "Date", "trade_date", "signal_date"])
        if date_col is not None:
            existing_features = existing_features.copy()
            existing_features["_merge_date"] = existing_features[date_col].astype(str).str.replace("-", "")
            df = df.merge(existing_features.drop(columns=[date_col], errors="ignore"),
                          left_on="date", right_on="_merge_date", how="left",
                          suffixes=("", "_ext"))
            df.drop(columns=["_merge_date"], errors="ignore", inplace=True)

    log.info(f"Final feature matrix: {len(df)} days x {len(df.columns)} columns")
    return df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 2: FORWARD LABELS (next-day, 3-day returns)
# ═══════════════════════════════════════════════════════════════════

def add_forward_labels(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add forward return labels. These are TARGETS only, never used as features.
    Prefix: fwd_ (excluded from feature sets automatically).
    """
    df = df.copy()
    closes = df["close_price"].values

    # 1-day forward return (in ticks)
    fwd_1d = np.full(len(closes), np.nan)
    fwd_1d[:-1] = (closes[1:] - closes[:-1]) / TICK_SIZE
    df["fwd_1d_ticks"] = fwd_1d

    # 1-day direction
    df["fwd_1d_dir"] = np.where(fwd_1d > 0, 1, np.where(fwd_1d < 0, -1, 0))

    # 3-day forward return
    fwd_3d = np.full(len(closes), np.nan)
    fwd_3d[:-3] = (closes[3:] - closes[:-3]) / TICK_SIZE
    df["fwd_3d_ticks"] = fwd_3d
    df["fwd_3d_dir"] = np.where(fwd_3d > 0, 1, np.where(fwd_3d < 0, -1, 0))

    # 5-day forward return
    fwd_5d = np.full(len(closes), np.nan)
    fwd_5d[:-5] = (closes[5:] - closes[:-5]) / TICK_SIZE
    df["fwd_5d_ticks"] = fwd_5d

    # Volatility regime label (next-day vol > median historical vol)
    if "realized_vol" in df.columns:
        vol_arr = df["realized_vol"].values
        fwd_vol = np.full(len(vol_arr), np.nan)
        fwd_vol[:-1] = vol_arr[1:]
        df["fwd_1d_vol"] = fwd_vol

    # Momentum continuation: did today's direction persist tomorrow?
    mom = df.get("intraday_momentum_ticks")
    if mom is not None:
        mom_arr = mom.values
        cont = np.full(len(mom_arr), np.nan)
        cont[:-1] = np.sign(mom_arr[:-1]) * np.sign(fwd_1d[:-1])
        df["fwd_mom_continuation"] = cont  # +1 = continued, -1 = reversed

    return df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 3: SUB-SIGNAL WALK-FORWARD MODELS
# ═══════════════════════════════════════════════════════════════════

def _get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Get feature columns — EXCLUDE all fwd_* columns (leakage!) and metadata."""
    exclude_prefixes = ("fwd_", "date", "close_price", "open_price")
    return [c for c in df.columns
            if not any(c.startswith(p) for p in exclude_prefixes)
            and c not in ("date",)
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)]


def _safe_finite(arr: np.ndarray) -> np.ndarray:
    """Replace NaN/inf with 0."""
    return np.where(np.isfinite(arr), arr, 0.0)


def train_sub_signal_wf(
    df: pd.DataFrame,
    target_col: str,
    signal_name: str,
    is_classifier: bool = True,
) -> np.ndarray:
    """
    Walk-forward train a sub-signal model using SLIDING 40-day window.

    Returns: array of OOT predictions (probability or regression value),
             same length as df, NaN where no prediction.
    """
    import lightgbm as lgb

    feature_cols = _get_feature_cols(df)
    n = len(df)
    predictions = np.full(n, np.nan)

    valid_target = df[target_col].notna()

    # LightGBM params
    if is_classifier:
        params = {
            "objective": "binary",
            "metric": "auc",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 10,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 2.0,
            "max_depth": 4,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }
    else:
        params = {
            "objective": "regression",
            "metric": "mse",
            "learning_rate": 0.03,
            "num_leaves": 15,
            "min_child_samples": 10,
            "feature_fraction": 0.7,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "lambda_l1": 0.5,
            "lambda_l2": 2.0,
            "max_depth": 4,
            "verbose": -1,
            "n_jobs": -1,
            "seed": 42,
        }

    n_predicted = 0
    for test_idx in range(WF_TRAIN_DAYS, n):
        # Sliding window: train on [test_idx - WF_TRAIN_DAYS, test_idx)
        train_start = test_idx - WF_TRAIN_DAYS
        train_end = test_idx

        train_mask = valid_target.iloc[train_start:train_end].values
        if train_mask.sum() < 15:
            continue

        X_train = df[feature_cols].iloc[train_start:train_end].values[train_mask]
        y_train = df[target_col].iloc[train_start:train_end].values[train_mask]

        X_train = _safe_finite(X_train)
        y_train = _safe_finite(y_train)

        if is_classifier:
            # Convert to binary: positive class = 1
            y_train = (y_train > 0).astype(float)

        # Skip if all same class
        if is_classifier and len(np.unique(y_train)) < 2:
            continue

        try:
            train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
            model = lgb.train(
                params, train_ds, num_boost_round=100,
                valid_sets=[train_ds], callbacks=[lgb.log_evaluation(0)],
            )

            X_test = _safe_finite(df[feature_cols].iloc[test_idx:test_idx + 1].values)
            pred = model.predict(X_test)[0]

            if is_classifier:
                predictions[test_idx] = float(pred)  # already probability
            else:
                predictions[test_idx] = float(pred)

            n_predicted += 1
            del model, train_ds
        except Exception as e:
            log.debug(f"Signal {signal_name} fold {test_idx}: {e}")
            continue

    log.info(f"  Signal {signal_name}: {n_predicted} OOT predictions "
             f"out of {n - WF_TRAIN_DAYS} possible")
    return predictions


def build_all_sub_signals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build all 4 sub-signals via walk-forward.

    Signal A: 1-day direction (classifier, target: fwd_1d_dir > 0)
    Signal B: 3-day direction (classifier, target: fwd_3d_dir > 0)
    Signal C: Volatility regime (classifier, target: fwd_1d_vol > median)
    Signal D: Momentum continuation (classifier, target: fwd_mom_continuation > 0)
    """
    log.info("Building sub-signals via walk-forward...")

    # Prepare classification targets
    df = df.copy()

    # Signal A: 1-day direction (prob of up)
    log.info("  Training Signal A: 1-day direction...")
    df["_target_a"] = (df["fwd_1d_dir"] > 0).astype(float)
    df.loc[df["fwd_1d_dir"] == 0, "_target_a"] = np.nan
    df["signal_a_prob"] = train_sub_signal_wf(df, "_target_a", "A_1d_dir", is_classifier=True)

    # Signal B: 3-day direction
    log.info("  Training Signal B: 3-day direction...")
    df["_target_b"] = (df["fwd_3d_dir"] > 0).astype(float)
    df.loc[df["fwd_3d_dir"] == 0, "_target_b"] = np.nan
    df["signal_b_prob"] = train_sub_signal_wf(df, "_target_b", "B_3d_dir", is_classifier=True)

    # Signal C: Volatility regime (high vol = above rolling median)
    log.info("  Training Signal C: Volatility regime...")
    if "fwd_1d_vol" in df.columns:
        # Rolling median vol over past 20 days as threshold
        vol_median = df["realized_vol"].rolling(window=20, min_periods=5).median()
        df["_target_c"] = np.where(
            df["fwd_1d_vol"].notna(),
            (df["fwd_1d_vol"] > vol_median).astype(float),
            np.nan,
        )
        df["signal_c_prob"] = train_sub_signal_wf(df, "_target_c", "C_vol_regime", is_classifier=True)
    else:
        df["signal_c_prob"] = np.nan
        log.warning("  No fwd_1d_vol — Signal C disabled")

    # Signal D: Momentum continuation
    log.info("  Training Signal D: Momentum continuation...")
    if "fwd_mom_continuation" in df.columns:
        df["_target_d"] = np.where(
            df["fwd_mom_continuation"].notna(),
            (df["fwd_mom_continuation"] > 0).astype(float),
            np.nan,
        )
        df["signal_d_prob"] = train_sub_signal_wf(df, "_target_d", "D_mom_cont", is_classifier=True)
    else:
        df["signal_d_prob"] = np.nan
        log.warning("  No fwd_mom_continuation — Signal D disabled")

    # Drop temp target columns
    df.drop(columns=[c for c in df.columns if c.startswith("_target_")], inplace=True)

    return df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 4: META-MODEL FEATURES
# ═══════════════════════════════════════════════════════════════════

def build_meta_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build meta-features from the 4 sub-signal probabilities.

    Meta-features:
      - All 4 signal probabilities (0-1)
      - Signal agreement count (how many agree on direction)
      - Signal confidence spread (max - min)
      - Mean signal probability
      - Regime features: recent vol, trend, OFI accumulation
    """
    df = df.copy()
    sig_cols = ["signal_a_prob", "signal_b_prob", "signal_c_prob", "signal_d_prob"]

    # Direction from each signal: prob > 0.5 = long, < 0.5 = short
    for sc in sig_cols:
        df[f"{sc}_dir"] = np.where(df[sc] > 0.5, 1, np.where(df[sc] < 0.5, -1, 0))

    # Agreement: how many directional signals agree (A and B only, C and D are regime/vol)
    dir_cols = ["signal_a_prob_dir", "signal_b_prob_dir"]
    dirs = df[dir_cols].values
    # Count how many agree with the majority
    majority = np.sign(np.nansum(dirs, axis=1))
    agreement = np.zeros(len(df))
    for i, dc in enumerate(dir_cols):
        agreement += (df[dc].values == majority).astype(float)
    # Also count if C and D support (vol + momentum confirm)
    if "signal_c_prob" in df.columns:
        # Low vol regime (signal_c < 0.5) is generally better for mean-reversion
        # High vol (signal_c > 0.5) is better for momentum — context-dependent
        pass
    if "signal_d_prob" in df.columns:
        # If momentum continuation (D) agrees with direction signals
        d_dir = df["signal_d_prob_dir"].values
        agreement += (d_dir == majority).astype(float)

    df["meta_agreement_count"] = agreement

    # Signal confidence spread
    sig_vals = df[sig_cols].values
    with np.errstate(invalid="ignore"):
        df["meta_conf_spread"] = np.nanmax(sig_vals, axis=1) - np.nanmin(sig_vals, axis=1)
        df["meta_mean_prob"] = np.nanmean(sig_vals, axis=1)
        df["meta_max_prob"] = np.nanmax(sig_vals, axis=1)
        df["meta_min_prob"] = np.nanmin(sig_vals, axis=1)

    # Directional conviction: distance of directional signals from 0.5
    for sc in ["signal_a_prob", "signal_b_prob"]:
        df[f"{sc}_conviction"] = np.abs(df[sc] - 0.5) * 2  # 0-1 scale

    # Mean directional conviction
    conv_cols = [f"{sc}_conviction" for sc in ["signal_a_prob", "signal_b_prob"]]
    df["meta_mean_conviction"] = df[conv_cols].mean(axis=1)

    # Clean up direction columns (not needed as meta features, just used for agreement)
    df.drop(columns=[f"{sc}_dir" for sc in sig_cols], errors="ignore", inplace=True)

    return df


# ═══════════════════════════════════════════════════════════════════
#  SECTION 5: META-MODEL WALK-FORWARD
# ═══════════════════════════════════════════════════════════════════

def _get_meta_feature_cols(df: pd.DataFrame) -> List[str]:
    """Get meta-model feature columns — signals + regime + base features.
    EXCLUDE all fwd_* columns and metadata."""
    exclude_prefixes = ("fwd_", "date", "close_price", "open_price")
    exclude_exact = {"date"}
    return [c for c in df.columns
            if not any(c.startswith(p) for p in exclude_prefixes)
            and c not in exclude_exact
            and df[c].dtype in (np.float64, np.float32, np.int64, np.int32, float, int)]


def train_meta_model_wf(df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward train the meta-model.

    Target: next-day return magnitude * direction correctness
            = fwd_1d_ticks (positive = made money on direction prediction)

    Returns: (meta_predictions, meta_directions) arrays, same length as df.
    """
    import lightgbm as lgb

    feature_cols = _get_meta_feature_cols(df)
    log.info(f"Meta-model features ({len(feature_cols)}): {feature_cols[:10]}...")

    # Target: signed next-day return (regression)
    target_col = "fwd_1d_ticks"
    n = len(df)
    meta_preds = np.full(n, np.nan)
    meta_dirs = np.full(n, np.nan)

    params = {
        "objective": "regression",
        "metric": "mse",
        "learning_rate": 0.02,
        "num_leaves": 15,
        "min_child_samples": 8,
        "feature_fraction": 0.7,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "lambda_l1": 1.0,
        "lambda_l2": 3.0,
        "max_depth": 4,
        "verbose": -1,
        "n_jobs": -1,
        "seed": 42,
    }

    # Need sub-signals to be available, so start after WF_TRAIN_DAYS * 2
    # (first WF_TRAIN_DAYS for sub-signals, then WF_TRAIN_DAYS for meta)
    meta_start = WF_TRAIN_DAYS * 2
    n_predicted = 0
    importances_accum = np.zeros(len(feature_cols))

    for test_idx in range(meta_start, n):
        train_start = test_idx - WF_TRAIN_DAYS
        train_end = test_idx

        # Only use rows where we have both sub-signals AND target
        train_df = df.iloc[train_start:train_end]
        valid = (
            train_df[target_col].notna()
            & train_df["signal_a_prob"].notna()
            & train_df["signal_b_prob"].notna()
        )
        if valid.sum() < 15:
            continue

        X_train = _safe_finite(train_df[feature_cols].values[valid.values])
        y_train = _safe_finite(train_df[target_col].values[valid.values])

        try:
            train_ds = lgb.Dataset(X_train, label=y_train, free_raw_data=True)
            model = lgb.train(
                params, train_ds, num_boost_round=100,
                valid_sets=[train_ds], callbacks=[lgb.log_evaluation(0)],
            )

            X_test = _safe_finite(df[feature_cols].iloc[test_idx:test_idx + 1].values)
            pred = float(model.predict(X_test)[0])
            meta_preds[test_idx] = pred
            meta_dirs[test_idx] = np.sign(pred)

            importances_accum += model.feature_importance(importance_type="gain")
            n_predicted += 1
            del model, train_ds
        except Exception as e:
            log.debug(f"Meta fold {test_idx}: {e}")
            continue

    log.info(f"Meta-model: {n_predicted} OOT predictions")

    # Log top features
    if n_predicted > 0:
        avg_imp = importances_accum / n_predicted
        top_idx = np.argsort(avg_imp)[::-1][:10]
        log.info("  Top meta features by gain:")
        for rank, fi in enumerate(top_idx):
            log.info(f"    {rank+1}. {feature_cols[fi]}: {avg_imp[fi]:.1f}")

    return meta_preds, meta_dirs


# ═══════════════════════════════════════════════════════════════════
#  SECTION 6: TRADING SIMULATION
# ═══════════════════════════════════════════════════════════════════

def simulate_trades(
    df: pd.DataFrame,
    meta_preds: np.ndarray,
    meta_dirs: np.ndarray,
    threshold: float,
) -> Dict[str, Any]:
    """
    Simulate intraday trades gated by meta-model confidence.

    Rules:
      - Enter when |meta_pred| > threshold (in ticks — larger = more confident)
      - Direction from meta_pred sign
      - 4h intraday hold, 80-tick trailing stop
      - FIFO costs: 0.376t passive entry, 1.376t market exit

    Since we don't have tick-level data for intraday sim, we approximate
    using minute bars: entry at open of next bar after signal, exit at
    min(stop hit, 4h hold, day end).
    """
    trades = []
    dates = df["date"].values

    for i in range(len(df)):
        pred = meta_preds[i]
        direction = meta_dirs[i]
        if not np.isfinite(pred) or not np.isfinite(direction):
            continue
        if abs(pred) < threshold:
            continue
        if direction == 0:
            continue

        # Signal is generated end-of-day i, trade executed on day i+1
        trade_day_idx = i + 1
        if trade_day_idx >= len(df):
            continue

        trade_date = dates[trade_day_idx]
        bars = load_minute_bars(str(trade_date))
        if bars is None:
            continue

        cols = _get_price_cols(bars)
        close_col = cols["close"]
        high_col = cols["high"]
        low_col = cols["low"]
        if close_col is None:
            continue

        closes = bars[close_col].values.astype(float)
        n_bars = len(closes)
        if n_bars < 10:
            continue

        # Entry at bar 0 (market open)
        entry_price = float(closes[0])
        if not np.isfinite(entry_price) or entry_price <= 0:
            continue

        direction = int(direction)
        max_hold = min(HOLD_MINUTES, n_bars - 1)

        # Simulate with trailing stop
        best_price = entry_price
        exit_bar = max_hold
        exit_reason = "time"
        exit_price = float(closes[min(exit_bar, n_bars - 1)])

        if high_col is not None and low_col is not None:
            highs = bars[high_col].values.astype(float)
            lows = bars[low_col].values.astype(float)
        else:
            highs = closes
            lows = closes

        for b in range(1, max_hold + 1):
            if b >= n_bars:
                exit_bar = b - 1
                exit_reason = "eod"
                exit_price = float(closes[min(exit_bar, n_bars - 1)])
                break

            bar_high = float(highs[b]) if np.isfinite(highs[b]) else float(closes[b])
            bar_low = float(lows[b]) if np.isfinite(lows[b]) else float(closes[b])

            if direction == 1:
                best_price = max(best_price, bar_high)
                drawdown = (best_price - bar_low) / TICK_SIZE
            else:
                best_price = min(best_price, bar_low)
                drawdown = (bar_high - best_price) / TICK_SIZE

            if drawdown >= STOP_LOSS_TICKS:
                exit_bar = b
                exit_reason = "stop"
                # Stop fill: entry_price + direction * (best_price offset - stop)
                if direction == 1:
                    exit_price = best_price - STOP_LOSS_TICKS * TICK_SIZE
                else:
                    exit_price = best_price + STOP_LOSS_TICKS * TICK_SIZE
                break
        else:
            exit_price = float(closes[min(exit_bar, n_bars - 1)])

        # P&L in ticks (FIFO)
        raw_pnl_ticks = direction * (exit_price - entry_price) / TICK_SIZE
        cost_ticks = TOTAL_RT_COST_TICKS
        net_pnl_ticks = raw_pnl_ticks - cost_ticks

        # MFE/MAE from bar data
        if direction == 1:
            mfe = (np.nanmax(highs[1:exit_bar + 1]) - entry_price) / TICK_SIZE if exit_bar > 0 else 0
            mae = (entry_price - np.nanmin(lows[1:exit_bar + 1])) / TICK_SIZE if exit_bar > 0 else 0
        else:
            mfe = (entry_price - np.nanmin(lows[1:exit_bar + 1])) / TICK_SIZE if exit_bar > 0 else 0
            mae = (np.nanmax(highs[1:exit_bar + 1]) - entry_price) / TICK_SIZE if exit_bar > 0 else 0

        trades.append({
            "signal_date": str(dates[i]),
            "trade_date": str(trade_date),
            "direction": direction,
            "meta_pred": float(pred),
            "meta_conf": float(abs(pred)),
            "entry_price": entry_price,
            "exit_price": exit_price,
            "hold_minutes": exit_bar,
            "exit_reason": exit_reason,
            "raw_pnl_ticks": raw_pnl_ticks,
            "cost_ticks": cost_ticks,
            "net_pnl_ticks": net_pnl_ticks,
            "net_pnl_dollars": net_pnl_ticks * ES_TICK_VALUE,
            "mfe_ticks": float(mfe) if np.isfinite(mfe) else 0.0,
            "mae_ticks": float(mae) if np.isfinite(mae) else 0.0,
        })

        del bars

    return trades


# ═══════════════════════════════════════════════════════════════════
#  SECTION 7: PERFORMANCE METRICS
# ═══════════════════════════════════════════════════════════════════

def compute_metrics(trades: List[Dict], label: str) -> Dict[str, Any]:
    """Compute risk-adjusted performance metrics for a trade list."""
    if not trades:
        return {"label": label, "n_trades": 0, "sharpe": 0.0}

    pnls = np.array([t["net_pnl_ticks"] for t in trades])
    n = len(pnls)
    total_pnl = float(np.sum(pnls))
    mean_pnl = float(np.mean(pnls))
    std_pnl = float(np.std(pnls)) if n > 1 else 1.0

    winners = pnls[pnls > 0]
    losers = pnls[pnls <= 0]
    wr = float(len(winners) / n) if n > 0 else 0.0

    # Sharpe (daily-ish, but per-trade is what we have)
    sharpe = float(mean_pnl / std_pnl * np.sqrt(252)) if std_pnl > 1e-10 else 0.0

    # Sortino (downside deviation)
    downside = pnls[pnls < 0]
    downside_std = float(np.std(downside)) if len(downside) > 1 else 1.0
    sortino = float(mean_pnl / downside_std * np.sqrt(252)) if downside_std > 1e-10 else 0.0

    # Profit factor
    gross_profit = float(np.sum(winners)) if len(winners) > 0 else 0.0
    gross_loss = float(np.abs(np.sum(losers))) if len(losers) > 0 else 1.0
    pf = gross_profit / gross_loss if gross_loss > 0 else float("inf")

    # Payoff ratio
    avg_win = float(np.mean(winners)) if len(winners) > 0 else 0.0
    avg_loss = float(np.abs(np.mean(losers))) if len(losers) > 0 else 1.0
    payoff = avg_win / avg_loss if avg_loss > 0 else float("inf")

    # Direction breakdown
    long_trades = [t for t in trades if t["direction"] == 1]
    short_trades = [t for t in trades if t["direction"] == -1]
    long_pnl = np.mean([t["net_pnl_ticks"] for t in long_trades]) if long_trades else 0.0
    short_pnl = np.mean([t["net_pnl_ticks"] for t in short_trades]) if short_trades else 0.0

    # Max drawdown (cumulative)
    cum_pnl = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdowns = running_max - cum_pnl
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    return {
        "label": label,
        "n_trades": n,
        "total_pnl_ticks": total_pnl,
        "total_pnl_dollars": total_pnl * ES_TICK_VALUE,
        "mean_pnl_ticks": mean_pnl,
        "std_pnl_ticks": std_pnl,
        "sharpe": sharpe,
        "sortino": sortino,
        "profit_factor": pf,
        "win_rate": wr,
        "payoff_ratio": payoff,
        "n_long": len(long_trades),
        "n_short": len(short_trades),
        "mean_pnl_long": float(long_pnl),
        "mean_pnl_short": float(short_pnl),
        "max_drawdown_ticks": max_dd,
        "avg_mfe_ticks": float(np.mean([t["mfe_ticks"] for t in trades])),
        "avg_mae_ticks": float(np.mean([t["mae_ticks"] for t in trades])),
        "avg_hold_minutes": float(np.mean([t["hold_minutes"] for t in trades])),
    }


# ═══════════════════════════════════════════════════════════════════
#  SECTION 8: REGIME-AGNOSTIC VALIDATION (HC #428)
# ═══════════════════════════════════════════════════════════════════

def regime_validation(
    trades: List[Dict],
    df: pd.DataFrame,
) -> Dict[str, Any]:
    """
    HC #428 regime-agnostic validation.

    - Classify each day as green/red/flat based on close-to-close
    - Compute per-regime Sharpe
    - Check regime gap: |Sharpe_green - Sharpe_red| / max(...) ≤ 0.50
    - Day concentration: max trades on any single day / total ≤ 0.70
    """
    if not trades:
        return {"regime_pass": False, "reason": "no trades"}

    # Classify trade days by regime
    date_to_regime = {}
    closes = df.set_index("date")["close_price"]
    dates_sorted = df["date"].values
    for i in range(1, len(dates_sorted)):
        d = str(dates_sorted[i])
        prev_close = closes.get(str(dates_sorted[i - 1]))
        curr_close = closes.get(d)
        if prev_close is not None and curr_close is not None:
            ret = (curr_close - prev_close) / prev_close
            if ret > 0.002:
                date_to_regime[d] = "green"
            elif ret < -0.002:
                date_to_regime[d] = "red"
            else:
                date_to_regime[d] = "flat"

    # Assign regime to each trade
    regime_pnls: Dict[str, List[float]] = {"green": [], "red": [], "flat": []}
    trade_day_counts: Dict[str, int] = {}

    for t in trades:
        td = t["trade_date"]
        regime = date_to_regime.get(td, "flat")
        regime_pnls[regime].append(t["net_pnl_ticks"])
        trade_day_counts[td] = trade_day_counts.get(td, 0) + 1

    # Per-regime Sharpe
    regime_sharpes = {}
    for regime, pnls in regime_pnls.items():
        if len(pnls) > 2:
            arr = np.array(pnls)
            s = float(np.mean(arr) / np.std(arr) * np.sqrt(252)) if np.std(arr) > 1e-10 else 0.0
            regime_sharpes[regime] = s
        else:
            regime_sharpes[regime] = 0.0

    # Regime gap check
    sg = regime_sharpes.get("green", 0.0)
    sr = regime_sharpes.get("red", 0.0)
    denom = max(abs(sg), abs(sr), 1e-10)
    regime_gap = abs(sg - sr) / denom

    # Day concentration
    max_day_count = max(trade_day_counts.values()) if trade_day_counts else 0
    day_conc = max_day_count / len(trades) if trades else 0.0

    regime_pass = regime_gap <= 0.50 and day_conc <= 0.70

    result = {
        "regime_pass": regime_pass,
        "regime_gap": float(regime_gap),
        "regime_gap_threshold": 0.50,
        "day_concentration": float(day_conc),
        "day_conc_threshold": 0.70,
        "regime_sharpes": {k: float(v) for k, v in regime_sharpes.items()},
        "regime_trade_counts": {k: len(v) for k, v in regime_pnls.items()},
    }

    if not regime_pass:
        reasons = []
        if regime_gap > 0.50:
            reasons.append(f"regime_gap={regime_gap:.2f} > 0.50")
        if day_conc > 0.70:
            reasons.append(f"day_conc={day_conc:.2f} > 0.70")
        result["reason"] = "; ".join(reasons)

    return result


# ═══════════════════════════════════════════════════════════════════
#  SECTION 9: MLFLOW LOGGING
# ═══════════════════════════════════════════════════════════════════

def log_to_mlflow(results: Dict, experiment_name: str = "meta_confluence_v1") -> None:
    """Log all results to MLflow."""
    try:
        import mlflow
    except ImportError:
        log.warning("MLflow not available, skipping logging")
        return

    mlflow_uri = "http://jupiter:5000"
    mlflow.set_tracking_uri(mlflow_uri)

    try:
        mlflow.set_experiment(experiment_name)
    except Exception as e:
        log.warning(f"Could not set MLflow experiment: {e}")
        return

    try:
        with mlflow.start_run(
            run_name=f"meta_conf_v1_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            description="Meta-confluence v1: multi-signal gating & sizing (HC #646 R2)",
        ):
            # Params
            mlflow.log_param("approach", "meta_confluence_gating")
            mlflow.log_param("wf_train_days", WF_TRAIN_DAYS)
            mlflow.log_param("window_type", "SLIDING_40D")
            mlflow.log_param("stop_loss_ticks", STOP_LOSS_TICKS)
            mlflow.log_param("hold_minutes", HOLD_MINUTES)
            mlflow.log_param("cost_passive_ticks", COST_PASSIVE_TICKS)
            mlflow.log_param("cost_market_ticks", COST_MARKET_TICKS)
            mlflow.log_param("accum_windows", str(ACCUM_WINDOWS))
            mlflow.log_param("thresholds_tested", str(META_THRESHOLDS))

            # Per-threshold metrics
            for thresh_key, thresh_data in results.get("thresholds", {}).items():
                safe = thresh_key.replace(".", "_")
                metrics = thresh_data.get("metrics", {})
                for metric in ["sharpe", "sortino", "profit_factor", "win_rate",
                               "mean_pnl_ticks", "total_pnl_ticks", "n_trades",
                               "max_drawdown_ticks", "payoff_ratio"]:
                    if metric in metrics:
                        val = metrics[metric]
                        if np.isfinite(val):
                            mlflow.log_metric(f"{safe}_{metric}", val)

                # Regime validation
                rv = thresh_data.get("regime_validation", {})
                if rv:
                    gap = rv.get("regime_gap", -1)
                    if np.isfinite(gap):
                        mlflow.log_metric(f"{safe}_regime_gap", gap)
                    mlflow.log_metric(f"{safe}_regime_pass", int(rv.get("regime_pass", False)))

            # Best threshold
            best = results.get("best_threshold", "")
            if best:
                mlflow.log_param("best_threshold", best)
                best_data = results["thresholds"].get(best, {}).get("metrics", {})
                for m in ["sharpe", "sortino", "profit_factor", "win_rate", "n_trades"]:
                    if m in best_data and np.isfinite(best_data[m]):
                        mlflow.log_metric(f"best_{m}", best_data[m])

            # Artifacts
            for fname in ["summary.json", "trades.parquet", "daily_features.parquet"]:
                artifact_path = OUTPUT_DIR / fname
                if artifact_path.exists():
                    mlflow.log_artifact(str(artifact_path))

            log.info("MLflow logging complete")

    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


# ═══════════════════════════════════════════════════════════════════
#  SECTION 10: MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    t_start = time.time()
    log.info("=" * 70)
    log.info("Meta-Confluence v1 — Multi-Signal Gating (HC #646 R2)")
    log.info("=" * 70)
    log.info(f"Goal: Learn WHEN multiple signals align for the best trades")
    log.info(f"Walk-forward: {WF_TRAIN_DAYS}d sliding window (HC #0)")
    log.info(f"Stop loss: {STOP_LOSS_TICKS} ticks, max hold: {HOLD_MINUTES} min")
    log.info(f"Costs: {COST_PASSIVE_TICKS}t passive entry + {COST_MARKET_TICKS}t market exit")
    log.info(f"Output: {OUTPUT_DIR}")

    # ── Step 1: Discover available minute bar days ──
    log.info("\n--- Step 1: Discover minute bar data ---")
    bar_files = sorted(MINUTE_BARS_DIR.glob("*.parquet"))
    date_strs = [f.stem for f in bar_files]
    log.info(f"Found {len(date_strs)} days of minute bars")
    if len(date_strs) < WF_TRAIN_DAYS * 2 + 10:
        log.error(f"Need at least {WF_TRAIN_DAYS * 2 + 10} days, have {len(date_strs)}")
        return

    # ── Step 2: Load existing features if available ──
    log.info("\n--- Step 2: Load existing features ---")
    existing_feats = None
    for feat_path in [ENHANCED_FEATURES, DAILY_FEATURES_V1]:
        if feat_path.exists():
            try:
                ef = pd.read_parquet(feat_path)
                # Exclude any fwd_* columns from existing features (leakage!)
                fwd_cols = [c for c in ef.columns if c.startswith("fwd_")]
                if fwd_cols:
                    log.info(f"  Dropping {len(fwd_cols)} fwd_* columns from existing features")
                    ef = ef.drop(columns=fwd_cols)
                existing_feats = ef
                log.info(f"  Loaded existing features from {feat_path.name}: "
                         f"{len(ef)} rows x {len(ef.columns)} cols")
                break
            except Exception as e:
                log.warning(f"  Could not load {feat_path.name}: {e}")

    # ── Step 3: Build daily feature matrix ──
    log.info("\n--- Step 3: Build daily feature matrix ---")
    df = build_daily_feature_matrix(date_strs, existing_feats)

    # ── Step 4: Add forward labels ──
    log.info("\n--- Step 4: Add forward labels ---")
    df = add_forward_labels(df)

    n_valid_1d = df["fwd_1d_ticks"].notna().sum()
    n_valid_3d = df["fwd_3d_ticks"].notna().sum()
    log.info(f"  Valid 1d labels: {n_valid_1d}, 3d labels: {n_valid_3d}")

    # Save features for inspection
    df.to_parquet(OUTPUT_DIR / "daily_features.parquet", index=False)
    log.info(f"  Saved daily features: {len(df)} rows x {len(df.columns)} cols")

    # ── Step 5: Build sub-signals ──
    log.info("\n--- Step 5: Build sub-signals (walk-forward) ---")
    t_sub = time.time()
    df = build_all_sub_signals(df)
    log.info(f"  Sub-signals complete in {time.time() - t_sub:.1f}s")

    # Report sub-signal quality
    for sig in ["signal_a_prob", "signal_b_prob", "signal_c_prob", "signal_d_prob"]:
        valid = df[sig].notna()
        if valid.sum() > 0:
            vals = df.loc[valid, sig].values
            log.info(f"  {sig}: {valid.sum()} predictions, "
                     f"mean={np.mean(vals):.3f}, std={np.std(vals):.3f}")

    # ── Step 6: Build meta-features ──
    log.info("\n--- Step 6: Build meta-features ---")
    df = build_meta_features(df)

    # ── Step 7: Train meta-model ──
    log.info("\n--- Step 7: Train meta-model (walk-forward) ---")
    t_meta = time.time()
    meta_preds, meta_dirs = train_meta_model_wf(df)
    log.info(f"  Meta-model complete in {time.time() - t_meta:.1f}s")

    n_meta_valid = np.sum(np.isfinite(meta_preds))
    log.info(f"  Meta predictions: {n_meta_valid} valid out of {len(meta_preds)}")

    if n_meta_valid < 20:
        log.error("Too few meta predictions — check data availability")
        return

    # ── Step 8: Trade simulation across thresholds ──
    log.info("\n--- Step 8: Trading simulation ---")
    all_results: Dict[str, Any] = {"thresholds": {}}
    best_sharpe = -999.0
    best_thresh = ""

    for thresh in META_THRESHOLDS:
        log.info(f"\n  Threshold: {thresh}")
        trades = simulate_trades(df, meta_preds, meta_dirs, thresh)
        metrics = compute_metrics(trades, f"meta_conf_t{thresh}")
        rv = regime_validation(trades, df)

        log.info(f"    Trades: {metrics['n_trades']}, WR: {metrics['win_rate']:.1%}, "
                 f"Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}, "
                 f"PF: {metrics['profit_factor']:.2f}")
        log.info(f"    Total P&L: {metrics['total_pnl_ticks']:.1f} ticks "
                 f"(${metrics.get('total_pnl_dollars', 0):.0f})")
        log.info(f"    Regime: gap={rv['regime_gap']:.2f}, "
                 f"day_conc={rv['day_concentration']:.2f}, "
                 f"pass={rv['regime_pass']}")
        if "regime_sharpes" in rv:
            for regime, s in rv["regime_sharpes"].items():
                log.info(f"      {regime}: Sharpe={s:.2f}, "
                         f"n={rv['regime_trade_counts'].get(regime, 0)}")

        thresh_key = f"t{thresh}"
        all_results["thresholds"][thresh_key] = {
            "threshold": thresh,
            "metrics": metrics,
            "regime_validation": rv,
            "n_trades": metrics["n_trades"],
        }

        # Save trades for best threshold
        if metrics["sharpe"] > best_sharpe and metrics["n_trades"] >= 20:
            best_sharpe = metrics["sharpe"]
            best_thresh = thresh_key
            # Save best trades
            if trades:
                trades_df = pd.DataFrame(trades)
                trades_df.to_parquet(OUTPUT_DIR / "trades.parquet", index=False)

    all_results["best_threshold"] = best_thresh
    log.info(f"\n  Best threshold: {best_thresh} (Sharpe={best_sharpe:.2f})")

    # ── Step 9: Baseline comparison ──
    log.info("\n--- Step 9: Baseline comparison ---")
    # Baseline: trade every day without meta-gating
    baseline_preds = np.where(np.isfinite(meta_preds), meta_preds, 0.0)
    baseline_dirs = np.sign(baseline_preds)
    baseline_trades = simulate_trades(df, baseline_preds, baseline_dirs, threshold=0.0)
    baseline_metrics = compute_metrics(baseline_trades, "baseline_no_gate")
    all_results["baseline"] = baseline_metrics

    log.info(f"  Baseline (no gate): {baseline_metrics['n_trades']} trades, "
             f"Sharpe={baseline_metrics['sharpe']:.2f}, WR={baseline_metrics['win_rate']:.1%}")

    if best_thresh:
        best_metrics = all_results["thresholds"][best_thresh]["metrics"]
        sharpe_improvement = best_metrics["sharpe"] - baseline_metrics["sharpe"]
        log.info(f"  Meta-gating improvement: Sharpe {baseline_metrics['sharpe']:.2f} → "
                 f"{best_metrics['sharpe']:.2f} ({sharpe_improvement:+.2f})")

    # ── Step 10: Save results ──
    log.info("\n--- Step 10: Save results ---")
    summary = {
        "experiment": "meta_confluence_v1",
        "timestamp": datetime.now().isoformat(),
        "config": {
            "wf_train_days": WF_TRAIN_DAYS,
            "window_type": "SLIDING",
            "stop_loss_ticks": STOP_LOSS_TICKS,
            "hold_minutes": HOLD_MINUTES,
            "cost_passive_ticks": COST_PASSIVE_TICKS,
            "cost_market_ticks": COST_MARKET_TICKS,
            "accum_windows": ACCUM_WINDOWS,
            "thresholds_tested": META_THRESHOLDS,
        },
        "data": {
            "n_days": len(df),
            "n_meta_predictions": int(n_meta_valid),
            "date_range": f"{df['date'].iloc[0]} to {df['date'].iloc[-1]}",
        },
        "results": all_results,
    }

    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, cls=NumpyEncoder)
    log.info(f"  Saved summary.json")

    # Save daily features with predictions
    df["meta_pred"] = meta_preds
    df["meta_dir"] = meta_dirs
    df.to_parquet(OUTPUT_DIR / "daily_features.parquet", index=False)
    log.info(f"  Saved daily_features.parquet with predictions")

    # ── Step 11: MLflow logging ──
    log.info("\n--- Step 11: MLflow logging ---")
    log_to_mlflow(all_results)

    elapsed = time.time() - t_start
    log.info(f"\n{'=' * 70}")
    log.info(f"Meta-Confluence v1 complete in {elapsed:.1f}s ({elapsed/60:.1f}min)")
    log.info(f"Output: {OUTPUT_DIR}")
    log.info(f"{'=' * 70}")


if __name__ == "__main__":
    main()
