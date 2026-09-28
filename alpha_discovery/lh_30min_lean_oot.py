#!/usr/bin/env python3
"""
30-Minute Lean Model OOT Validation
====================================

Strict out-of-time holdout comparison: lean (43 features) vs full (65 features).

The feature ablation study found that the lean set (momentum_vol + top-10 OFI)
outperformed the full model: Sharpe 3.23 vs 2.70. This script validates that
finding with a proper holdout test.

Design:
  - Walk-forward on first 160 days (60d train, 10d val, 5d slide) — SLIDING only
  - True OOT holdout on last N days — train final model on 60d before holdout
  - Compare lean vs full on the SAME holdout
  - Stability check: 3 holdout splits (last 37d, 30d, 25d)

Walk-Forward: 60d train, 10d val, slide 5d — SLIDING only (HC #0).
Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission).

Usage:
  cd /home/nick/Lvl3Quant && \
  /home/nick/miniconda3/envs/py311-train/bin/python -u \
      alpha_discovery/lh_30min_lean_oot.py

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
OUTPUT_DIR = ROOT / "output" / "lh_30min_lean_oot"
LOG_DIR = ROOT / "logs"
ABLATION_RESULTS = ROOT / "output" / "lh_30min_feature_ablation" / "ablation_results.json"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [LEAN-OOT] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "lh_30min_lean_oot.log")),
    ],
)
log = logging.getLogger("LEAN-OOT")

# ─────────────────────────────────────────────
#  COST CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION = 4.70
ES_RT_COMMISSION_TICKS = 0.376
COST_RT_TICKS = 2.376  # market entry + market exit + RT commission
MIN_EDGE_TICKS = 2.5

# Walk-forward config
TRAIN_DAYS = 60
VAL_DAYS = 10
SLIDE_DAYS = 5

# Holdout splits for stability check
HOLDOUT_SPLITS = [37, 30, 25]

# LightGBM params — champion params from v4 focused
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
#  DATA LOADING (faithfully reused from lh_30min_feature_ablation.py)
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
#  BAR AGGREGATION + FEATURES (faithfully reused from ablation script)
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
    """Aggregate 1-minute bars into 30-minute bars with microstructure features."""
    df = minute_df.copy()
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
    log.info(f"Aggregated {len(result):,} {bar_size_min}min bars")
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


def add_forward_labels(df: pd.DataFrame, horizon_bars: int = 1) -> pd.DataFrame:
    """Compute forward return labels with overnight gap protection."""
    df = df.sort_values("ts").reset_index(drop=True)

    fwd_close = df["close"].shift(-horizon_bars)
    fwd_ticks = (fwd_close - df["close"]) / 0.25

    ts_now = df["ts"].values
    ts_fwd = df["ts"].shift(-horizon_bars).values
    for i in range(len(df) - horizon_bars):
        if pd.isna(ts_fwd[i]):
            continue
        diff_s = (pd.Timestamp(ts_fwd[i]) - pd.Timestamp(ts_now[i])).total_seconds()
        if diff_s > 6 * 3600:
            fwd_ticks.iloc[i] = np.nan

    df["fwd_ticks"] = fwd_ticks
    df["direction"] = 0
    df.loc[fwd_ticks > MIN_EDGE_TICKS, "direction"] = 1
    df.loc[fwd_ticks < -MIN_EDGE_TICKS, "direction"] = -1

    log.info(
        f"Forward labels: {(~fwd_ticks.isna()).sum():,} valid, "
        f"{(df['direction'] == 1).sum():,} long, "
        f"{(df['direction'] == -1).sum():,} short"
    )
    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select feature columns, excluding labels, metadata, raw prices."""
    exclude_prefixes = (
        "fwd_", "direction", "trade_quality_",
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
#  LEAKAGE AUDIT
# ═══════════════════════════════════════════════════════════════════


def leakage_audit(
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

    fwd_leak = [c for c in feature_cols if c.startswith(("fwd_", "direction", "trade_quality_"))]
    results["no_forward_features"] = len(fwd_leak) == 0
    if fwd_leak:
        log.error(f"LEAKAGE: Forward-looking columns in features: {fwd_leak}")

    price_cols = [c for c in feature_cols if c in ("close", "high", "low", "open")]
    results["no_raw_price_features"] = len(price_cols) == 0
    if price_cols:
        log.warning(f"WARNING: Raw price columns in features: {price_cols}")

    all_passed = all(results.values())
    results["all_passed"] = all_passed
    if not all_passed:
        log.error(f"Leakage audit FAILED: {results}")

    return results


# ═══════════════════════════════════════════════════════════════════
#  TRADE SIMULATION (faithfully from ablation script)
# ═══════════════════════════════════════════════════════════════════


def simulate_trades(
    preds: np.ndarray,
    actuals: np.ndarray,
    dates: np.ndarray,
    confidence_pct: float = 0.10,
    cost_ticks: float = COST_RT_TICKS,
) -> Optional[Dict]:
    """Simulate both-sides trades with per-side and per-day reporting."""
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
        "n_trading_days": int(len(day_pnl)),
        "daily_sharpe": float(
            day_pnl["daily_pnl"].mean() / max(day_pnl["daily_pnl"].std(), 1e-6) * np.sqrt(252)
        ) if len(day_pnl) > 2 else 0,
        "daily_pnl": day_pnl.to_dict("records"),
    }


# ═══════════════════════════════════════════════════════════════════
#  REGIME STRATIFICATION (HC #428 R1)
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
            results[regime] = {
                "n_predictions": int(mask.sum()),
                "ic": float(ic),
                "n_trades": 0,
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
            f"gap={gap:.2f} {'PASS' if gap <= 0.50 else 'FAIL'} (threshold 0.50)"
        )
    else:
        results["regime_gap"] = float("nan")
        results["regime_gap_pass"] = False
        results["regime_gap_detail"] = "insufficient regime data"

    return results


# ═══════════════════════════════════════════════════════════════════
#  LOAD LEAN FEATURE LIST FROM ABLATION RESULTS
# ═══════════════════════════════════════════════════════════════════


def load_lean_features() -> List[str]:
    """Load the lean feature list from ablation results JSON."""
    if not ABLATION_RESULTS.exists():
        log.error(f"Ablation results not found at {ABLATION_RESULTS}")
        raise FileNotFoundError(f"Missing: {ABLATION_RESULTS}")

    with open(ABLATION_RESULTS) as f:
        results = json.load(f)

    lean_entry = results.get("lean", {})
    features = lean_entry.get("features", [])

    if not features:
        # Try _meta.feature_groups.lean
        meta = results.get("_meta", {})
        fg = meta.get("feature_groups", {})
        features = fg.get("lean", [])

    if not features:
        raise ValueError("Could not find lean feature list in ablation results")

    log.info(f"Loaded lean feature list: {len(features)} features")
    log.info(f"  Features: {features}")
    return features


# ═══════════════════════════════════════════════════════════════════
#  TRAIN + PREDICT FOR ONE MODEL ON HOLDOUT
# ═══════════════════════════════════════════════════════════════════


def train_and_predict_holdout(
    model_name: str,
    feature_subset: List[str],
    features_all: np.ndarray,
    labels_all: np.ndarray,
    dates_all: np.ndarray,
    feature_cols: List[str],
    train_dates: List[str],
    holdout_dates: List[str],
    day_returns: Dict[str, float],
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Train a model on train_dates, predict on holdout_dates.
    Returns comprehensive results dict.
    """
    _import_lightgbm()

    log.info(f"\n{'=' * 60}")
    log.info(f"MODEL: {model_name} ({len(feature_subset)} features)")
    log.info(f"Train: {train_dates[0]} -> {train_dates[-1]} ({len(train_dates)}d)")
    log.info(f"Holdout: {holdout_dates[0]} -> {holdout_dates[-1]} ({len(holdout_dates)}d)")
    log.info(f"{'=' * 60}")

    # Get column indices for this feature subset
    feat_indices = [feature_cols.index(f) for f in feature_subset if f in feature_cols]
    feat_names = [feature_cols[i] for i in feat_indices]
    feat_subset = features_all[:, feat_indices]

    if len(feat_indices) < 3:
        log.error(f"Too few features ({len(feat_indices)}) for {model_name}")
        return {"error": f"too few features ({len(feat_indices)})"}

    # Leakage audit
    audit = leakage_audit(train_dates, holdout_dates, feat_names)
    if not audit["all_passed"]:
        log.error(f"LEAKAGE AUDIT FAILED for {model_name}")
        return {"error": "leakage audit failed"}

    # Split
    train_mask = np.isin(dates_all, train_dates)
    holdout_mask = np.isin(dates_all, holdout_dates)

    X_train_raw = feat_subset[train_mask].copy()
    X_holdout_raw = feat_subset[holdout_mask].copy()

    # Robust scaling from train only
    train_median = np.nanmedian(X_train_raw, axis=0)
    q75 = np.nanpercentile(X_train_raw, 75, axis=0)
    q25 = np.nanpercentile(X_train_raw, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr < 1e-8] = 1.0

    X_train = np.clip(np.nan_to_num((X_train_raw - train_median) / iqr, nan=0.0, posinf=3.0, neginf=-3.0), -5, 5)
    X_holdout = np.clip(np.nan_to_num((X_holdout_raw - train_median) / iqr, nan=0.0, posinf=3.0, neginf=-3.0), -5, 5)

    y_train = labels_all[train_mask]
    y_holdout = labels_all[holdout_mask]
    holdout_dates_arr = dates_all[holdout_mask]

    train_valid = ~np.isnan(y_train)
    holdout_valid = ~np.isnan(y_holdout)

    if train_valid.sum() < 50 or holdout_valid.sum() < 10:
        log.error(f"Too few valid samples: train={train_valid.sum()}, holdout={holdout_valid.sum()}")
        return {"error": "too few samples"}

    X_tr = X_train[train_valid]
    y_tr = y_train[train_valid]
    X_h = X_holdout[holdout_valid]
    y_h = y_holdout[holdout_valid]
    dates_h = holdout_dates_arr[holdout_valid]

    log.info(f"  Train samples: {len(y_tr):,}, Holdout samples: {len(y_h):,}")

    # Train LightGBM
    params = {**LGBM_PARAMS, "seed": seed}

    # Use 20% of training data as validation for early stopping
    n_tr = len(X_tr)
    n_val_split = max(int(n_tr * 0.2), 50)
    X_tr_split = X_tr[:-n_val_split]
    y_tr_split = y_tr[:-n_val_split]
    X_val_split = X_tr[-n_val_split:]
    y_val_split = y_tr[-n_val_split:]

    train_data = lgb.Dataset(X_tr_split, label=y_tr_split, feature_name=feat_names)
    val_data = lgb.Dataset(X_val_split, label=y_val_split, feature_name=feat_names, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    try:
        model = lgb.train(
            params, train_data, num_boost_round=500,
            valid_sets=[val_data], callbacks=callbacks,
        )
    except Exception as e:
        log.error(f"Training failed for {model_name}: {e}")
        return {"error": str(e)}

    # Predict on holdout
    holdout_preds = model.predict(X_h, num_iteration=model.best_iteration)

    # ── Metrics ──
    ic = float(np.corrcoef(holdout_preds, y_h)[0, 1])
    dir_acc = float(np.mean(np.sign(holdout_preds) == np.sign(y_h)))

    log.info(f"  Holdout IC={ic:.4f}, dir_acc={dir_acc:.3f}")

    # Feature importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feat_names, importance), key=lambda x: x[1], reverse=True)
    top_features = [{"name": n, "gain": float(g)} for n, g in feat_imp[:15]]

    # Trade sim at 10% and 15%
    trade_results = {}
    for conf_pct in [0.10, 0.15]:
        sim = simulate_trades(holdout_preds, y_h, dates_h, confidence_pct=conf_pct)
        if sim is not None:
            trade_results[f"top_{int(conf_pct*100)}pct"] = sim
            log.info(
                f"  Trades (top {int(conf_pct*100)}%): "
                f"n={sim['n_trades']}, Sharpe={sim['sharpe']:.2f}, "
                f"Sortino={sim['sortino']:.2f}, WR={sim['win_rate']:.1%}, "
                f"PF={sim['profit_factor']:.2f}, avg_pnl={sim['avg_pnl_ticks']:.2f}t, "
                f"maxDD={sim['max_dd_ticks']:.1f}t"
            )
            # Long vs short
            if sim["long_trades"] > 0:
                log.info(
                    f"    LONG:  n={sim['long_trades']}, WR={sim['long_wr']:.1%}, "
                    f"avg={sim['long_avg']:.2f}t, Sharpe={sim['long_sharpe']:.2f}"
                )
            if sim["short_trades"] > 0:
                log.info(
                    f"    SHORT: n={sim['short_trades']}, WR={sim['short_wr']:.1%}, "
                    f"avg={sim['short_avg']:.2f}t, Sharpe={sim['short_sharpe']:.2f}"
                )

    # Regime stratification (HC #428 R1)
    regime_results = regime_stratification(
        holdout_preds, y_h, dates_h,
        day_returns=day_returns, confidence_pct=0.10,
    )
    log.info(f"  Regime: {regime_results.get('regime_gap_detail', 'N/A')}")

    # Per-day PnL breakdown
    unique_holdout_dates = sorted(set(dates_h))
    per_day = []
    for d in unique_holdout_dates:
        d_mask = dates_h == d
        if d_mask.sum() < 2:
            continue
        d_preds = holdout_preds[d_mask]
        d_actuals = y_h[d_mask]
        d_ic = np.corrcoef(d_preds, d_actuals)[0, 1] if d_mask.sum() > 3 else float("nan")
        d_regime = "green" if day_returns.get(d, 0) > 0.001 else (
            "red" if day_returns.get(d, 0) < -0.001 else "flat"
        )
        # Compute daily pnl from trade sim (top 10%)
        d_sim = simulate_trades(d_preds, d_actuals, np.array([d] * len(d_preds)), confidence_pct=0.10)
        d_pnl = d_sim["total_pnl_ticks"] if d_sim else 0.0
        d_trades = d_sim["n_trades"] if d_sim else 0

        per_day.append({
            "date": d,
            "regime": d_regime,
            "ic": float(d_ic) if not np.isnan(d_ic) else 0.0,
            "n_bars": int(d_mask.sum()),
            "pnl_ticks": float(d_pnl),
            "n_trades": d_trades,
        })

    # Save predictions
    np.savez_compressed(
        str(OUTPUT_DIR / f"{model_name}_holdout_preds.npz"),
        preds=holdout_preds,
        actuals=y_h,
        dates=dates_h,
        features=feat_names,
    )

    del model, train_data, val_data
    gc.collect()

    return {
        "model": model_name,
        "n_features": len(feat_indices),
        "features": feat_names,
        "train_days": len(train_dates),
        "holdout_days": len(holdout_dates),
        "holdout_start": holdout_dates[0],
        "holdout_end": holdout_dates[-1],
        "n_holdout_samples": int(len(y_h)),
        "ic": ic,
        "dir_acc": dir_acc,
        "best_iteration": int(model.best_iteration) if 'model' in dir() else 0,
        "trade_sims": trade_results,
        "regime": regime_results,
        "regime_gap": float(regime_results.get("regime_gap", float("nan"))),
        "regime_gap_pass": regime_results.get("regime_gap_pass", False),
        "regime_gap_detail": regime_results.get("regime_gap_detail", "N/A"),
        "per_day": per_day,
        "top_features": top_features,
    }


# ═══════════════════════════════════════════════════════════════════
#  WALK-FORWARD (for pre-holdout model quality check)
# ═══════════════════════════════════════════════════════════════════


def run_walkforward(
    model_name: str,
    feature_subset: List[str],
    features_all: np.ndarray,
    labels_all: np.ndarray,
    dates_all: np.ndarray,
    feature_cols: List[str],
    wf_dates: List[str],
    day_returns: Dict[str, float],
) -> Dict[str, Any]:
    """
    Run walk-forward on the non-holdout portion to verify model quality.
    60d train, 10d val, 5d slide — SLIDING only.
    """
    _import_lightgbm()

    log.info(f"\n--- Walk-Forward: {model_name} on {len(wf_dates)} days ---")

    feat_indices = [feature_cols.index(f) for f in feature_subset if f in feature_cols]
    feat_names = [feature_cols[i] for i in feat_indices]
    feat_subset = features_all[:, feat_indices]

    all_oot_preds = []
    all_oot_actuals = []
    all_oot_dates = []
    fold_ics = []
    fold_idx = 0

    for fold_start in range(TRAIN_DAYS, len(wf_dates) - VAL_DAYS + 1, SLIDE_DAYS):
        fold_train_dates = wf_dates[fold_start - TRAIN_DAYS: fold_start]
        fold_val_dates = wf_dates[fold_start: fold_start + VAL_DAYS]

        if len(fold_val_dates) < VAL_DAYS:
            break

        fold_idx += 1

        audit = leakage_audit(list(fold_train_dates), list(fold_val_dates), feat_names)
        if not audit["all_passed"]:
            continue

        train_mask = np.isin(dates_all, fold_train_dates)
        val_mask = np.isin(dates_all, fold_val_dates)

        X_train_raw = feat_subset[train_mask].copy()
        X_val_raw = feat_subset[val_mask].copy()

        train_median = np.nanmedian(X_train_raw, axis=0)
        q75 = np.nanpercentile(X_train_raw, 75, axis=0)
        q25 = np.nanpercentile(X_train_raw, 25, axis=0)
        iqr = q75 - q25
        iqr[iqr < 1e-8] = 1.0

        X_train = np.clip(np.nan_to_num((X_train_raw - train_median) / iqr, nan=0.0, posinf=3.0, neginf=-3.0), -5, 5)
        X_val = np.clip(np.nan_to_num((X_val_raw - train_median) / iqr, nan=0.0, posinf=3.0, neginf=-3.0), -5, 5)

        y_train = labels_all[train_mask]
        y_val = labels_all[val_mask]
        val_dates_fold = dates_all[val_mask]

        train_valid = ~np.isnan(y_train)
        val_valid = ~np.isnan(y_val)

        if train_valid.sum() < 50 or val_valid.sum() < 10:
            continue

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]
        X_v = X_val[val_valid]
        y_v = y_val[val_valid]

        params = {**LGBM_PARAMS, "seed": 42 + fold_idx}

        train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names)
        val_data = lgb.Dataset(X_v, label=y_v, feature_name=feat_names, reference=train_data)

        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        try:
            model = lgb.train(
                params, train_data, num_boost_round=500,
                valid_sets=[val_data], callbacks=callbacks,
            )
        except Exception as e:
            log.warning(f"  Fold {fold_idx} train failed: {e}")
            continue

        val_preds = model.predict(X_v, num_iteration=model.best_iteration)

        if len(val_preds) > 5:
            ic = float(np.corrcoef(val_preds, y_v)[0, 1])
            if not np.isnan(ic):
                fold_ics.append(ic)
        else:
            ic = float("nan")

        all_oot_preds.append(val_preds)
        all_oot_actuals.append(y_v)
        all_oot_dates.append(val_dates_fold[val_valid])

        del model, train_data, val_data
        gc.collect()

    if not all_oot_preds:
        return {"error": "no valid WF folds"}

    concat_preds = np.concatenate(all_oot_preds)
    concat_actuals = np.concatenate(all_oot_actuals)
    concat_dates = np.concatenate(all_oot_dates)

    concat_ic = float(np.corrcoef(concat_preds, concat_actuals)[0, 1])
    ic_sharpe = float(np.mean(fold_ics) / max(np.std(fold_ics), 1e-6)) if len(fold_ics) > 2 else float("nan")

    sim_10 = simulate_trades(concat_preds, concat_actuals, concat_dates, confidence_pct=0.10)

    log.info(
        f"  WF {model_name}: {fold_idx} folds, concat_IC={concat_ic:.4f}, IC_Sharpe={ic_sharpe:.3f}"
    )
    if sim_10:
        log.info(
            f"  WF trades (top 10%): Sharpe={sim_10['sharpe']:.2f}, "
            f"Sortino={sim_10['sortino']:.2f}, WR={sim_10['win_rate']:.1%}"
        )

    return {
        "model": model_name,
        "n_folds": fold_idx,
        "concat_ic": concat_ic,
        "ic_sharpe": ic_sharpe,
        "wf_sharpe_10pct": sim_10["sharpe"] if sim_10 else float("nan"),
        "wf_sortino_10pct": sim_10["sortino"] if sim_10 else float("nan"),
    }


# ═══════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════


def main():
    log.info("=" * 70)
    log.info("30-MINUTE LEAN MODEL OOT VALIDATION")
    log.info("=" * 70)
    log.info(f"WF: {TRAIN_DAYS}d train, {VAL_DAYS}d val, slide {SLIDE_DAYS}d")
    log.info(f"Cost: {COST_RT_TICKS} ticks RT")
    log.info(f"Holdout splits: {HOLDOUT_SPLITS} days")
    log.info(f"Output: {OUTPUT_DIR}")

    t0 = time.time()

    # ── Load lean feature list from ablation results ──
    log.info("\n--- Loading lean feature list ---")
    lean_features = load_lean_features()

    # ── Load data ──
    log.info("\n--- Loading data ---")
    minute_df = load_all_minute_bars()
    queue_df = load_queue_features()

    # ── Build 30-min bars with all features ──
    log.info("\n--- Building 30-min bars ---")
    bars_df = aggregate_to_bars(minute_df, bar_size_min=30)
    bars_df.attrs["bar_size_min"] = 30
    bars_df = add_queue_features_to_bars(bars_df, queue_df)
    bars_df = add_rolling_features(bars_df)
    bars_df = add_forward_labels(bars_df, horizon_bars=1)

    del minute_df, queue_df
    gc.collect()

    # ── Feature columns ──
    all_feature_cols = get_feature_columns(bars_df)
    log.info(f"Total features: {len(all_feature_cols)}")

    # Validate lean features exist in current dataset
    available_lean = [f for f in lean_features if f in all_feature_cols]
    missing_lean = [f for f in lean_features if f not in all_feature_cols]
    if missing_lean:
        log.warning(f"Missing lean features (will skip): {missing_lean}")
    lean_features = available_lean
    log.info(f"Lean features available: {len(lean_features)}")
    log.info(f"Full features: {len(all_feature_cols)}")

    # ── Prepare arrays ──
    features_all = bars_df[all_feature_cols].values.astype(np.float32)
    labels_all = bars_df["fwd_ticks"].values.astype(np.float32)
    dates_all = bars_df["date"].values
    all_dates = sorted(bars_df["date"].unique())

    day_close = bars_df.groupby("date")["close"].last()
    day_returns = day_close.pct_change().to_dict()

    log.info(f"Total days: {len(all_dates)} ({all_dates[0]} -> {all_dates[-1]})")

    # ═══════════════════════════════════════════════════════════════
    #  STABILITY CHECK: 3 holdout splits
    # ═══════════════════════════════════════════════════════════════

    all_holdout_results = {}

    for holdout_days in HOLDOUT_SPLITS:
        log.info(f"\n\n{'#' * 70}")
        log.info(f"# HOLDOUT SPLIT: last {holdout_days} days")
        log.info(f"{'#' * 70}")

        if len(all_dates) < TRAIN_DAYS + holdout_days:
            log.error(f"Not enough days for {holdout_days}-day holdout (have {len(all_dates)})")
            continue

        holdout_dates = all_dates[-holdout_days:]
        # Train on the 60 days immediately before holdout
        train_end_idx = len(all_dates) - holdout_days
        train_start_idx = max(0, train_end_idx - TRAIN_DAYS)
        final_train_dates = all_dates[train_start_idx:train_end_idx]

        # WF dates = everything before holdout (for walk-forward quality check)
        wf_dates = all_dates[:train_end_idx]

        log.info(f"Final train: {final_train_dates[0]} -> {final_train_dates[-1]} ({len(final_train_dates)}d)")
        log.info(f"Holdout: {holdout_dates[0]} -> {holdout_dates[-1]} ({len(holdout_dates)}d)")
        log.info(f"WF period: {wf_dates[0]} -> {wf_dates[-1]} ({len(wf_dates)}d)")

        split_results = {}

        # ── Walk-forward quality check (both models) ──
        for model_name, feature_list in [("lean", lean_features), ("full", all_feature_cols)]:
            wf_result = run_walkforward(
                model_name=f"{model_name}_wf_{holdout_days}d",
                feature_subset=feature_list,
                features_all=features_all,
                labels_all=labels_all,
                dates_all=dates_all,
                feature_cols=all_feature_cols,
                wf_dates=wf_dates,
                day_returns=day_returns,
            )
            split_results[f"{model_name}_wf"] = wf_result

        # ── True holdout test (both models) ──
        for model_name, feature_list in [("lean", lean_features), ("full", all_feature_cols)]:
            holdout_result = train_and_predict_holdout(
                model_name=f"{model_name}_{holdout_days}d",
                feature_subset=feature_list,
                features_all=features_all,
                labels_all=labels_all,
                dates_all=dates_all,
                feature_cols=all_feature_cols,
                train_dates=final_train_dates,
                holdout_dates=holdout_dates,
                day_returns=day_returns,
                seed=42,
            )
            split_results[f"{model_name}_holdout"] = holdout_result

        all_holdout_results[f"last_{holdout_days}d"] = split_results

    # ═══════════════════════════════════════════════════════════════
    #  COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════

    log.info(f"\n\n{'=' * 120}")
    log.info("OOT VALIDATION COMPARISON TABLE")
    log.info(f"{'=' * 120}")

    header = (
        f"{'Split':<12s} {'Model':<8s} {'#Feat':>5s} {'IC':>7s} "
        f"{'10%Sharpe':>10s} {'10%Sortino':>11s} {'10%WR':>7s} {'10%PF':>7s} {'10%MaxDD':>9s} "
        f"{'15%Sharpe':>10s} {'15%WR':>7s} "
        f"{'RegimeGap':>10s} {'Pass':>5s}"
    )
    log.info(header)
    log.info("-" * 120)

    lean_wins = 0
    total_splits = 0

    for split_name, split_results in all_holdout_results.items():
        for model_key in ["lean_holdout", "full_holdout"]:
            r = split_results.get(model_key, {})
            if "error" in r:
                log.info(f"  {split_name:<12s} {model_key.replace('_holdout',''):<8s} ERROR: {r['error']}")
                continue

            sim_10 = r.get("trade_sims", {}).get("top_10pct", {})
            sim_15 = r.get("trade_sims", {}).get("top_15pct", {})

            regime_gap_str = f"{r.get('regime_gap', float('nan')):.2f}" if not np.isnan(r.get("regime_gap", float("nan"))) else "N/A"
            pass_str = "PASS" if r.get("regime_gap_pass") else "FAIL"

            line = (
                f"{split_name:<12s} {model_key.replace('_holdout',''):<8s} "
                f"{r.get('n_features', 0):>5d} "
                f"{r.get('ic', 0):>7.4f} "
                f"{sim_10.get('sharpe', 0):>10.2f} "
                f"{sim_10.get('sortino', 0):>11.2f} "
                f"{sim_10.get('win_rate', 0):>7.1%} "
                f"{sim_10.get('profit_factor', 0):>7.2f} "
                f"{sim_10.get('max_dd_ticks', 0):>9.1f} "
                f"{sim_15.get('sharpe', 0):>10.2f} "
                f"{sim_15.get('win_rate', 0):>7.1%} "
                f"{regime_gap_str:>10s} {pass_str:>5s}"
            )
            log.info(line)

        # Compare lean vs full for this split
        lean_r = split_results.get("lean_holdout", {})
        full_r = split_results.get("full_holdout", {})
        if "error" not in lean_r and "error" not in full_r:
            lean_sim = lean_r.get("trade_sims", {}).get("top_10pct", {})
            full_sim = full_r.get("trade_sims", {}).get("top_10pct", {})
            lean_sharpe = lean_sim.get("sharpe", float("-inf"))
            full_sharpe = full_sim.get("sharpe", float("-inf"))
            if lean_sharpe > full_sharpe:
                lean_wins += 1
            total_splits += 1
            winner = "LEAN" if lean_sharpe > full_sharpe else "FULL"
            log.info(f"  >>> {split_name} winner: {winner} (lean={lean_sharpe:.2f} vs full={full_sharpe:.2f})")

        log.info("")

    log.info("-" * 120)

    # ── Stability verdict ──
    log.info(f"\n{'=' * 60}")
    log.info("STABILITY VERDICT")
    log.info(f"{'=' * 60}")
    log.info(f"Lean wins {lean_wins}/{total_splits} holdout splits")

    if total_splits > 0:
        if lean_wins == total_splits:
            log.info("VERDICT: Lean CONSISTENTLY beats full across all splits. STRONG evidence for feature reduction.")
        elif lean_wins > total_splits / 2:
            log.info("VERDICT: Lean beats full in majority of splits. MODERATE evidence for feature reduction.")
        elif lean_wins == total_splits / 2:
            log.info("VERDICT: Mixed results. No clear advantage to feature reduction.")
        else:
            log.info("VERDICT: Full model wins majority. Lean model advantage may have been in-sample artifact.")

    # ── WF consistency check ──
    log.info(f"\n--- Walk-Forward Consistency ---")
    for split_name, split_results in all_holdout_results.items():
        for model_key in ["lean_wf", "full_wf"]:
            wf = split_results.get(model_key, {})
            if "error" in wf:
                continue
            log.info(
                f"  {split_name} {model_key}: WF_IC={wf.get('concat_ic', 0):.4f}, "
                f"WF_IC_Sharpe={wf.get('ic_sharpe', 0):.3f}, "
                f"WF_Sharpe_10%={wf.get('wf_sharpe_10pct', 0):.2f}"
            )

    # ── Per-day PnL details for primary split (last 37d) ──
    primary_split = all_holdout_results.get("last_37d", {})
    for model_key in ["lean_holdout", "full_holdout"]:
        r = primary_split.get(model_key, {})
        if "error" in r or not r.get("per_day"):
            continue
        log.info(f"\n--- Per-Day PnL: {model_key} (last 37d holdout) ---")
        log.info(f"{'Date':<12s} {'Regime':<6s} {'IC':>7s} {'PnL_t':>8s} {'#Trades':>8s}")
        for d in r["per_day"]:
            log.info(
                f"  {d['date']:<12s} {d['regime']:<6s} "
                f"{d['ic']:>7.3f} {d['pnl_ticks']:>8.2f} {d['n_trades']:>8d}"
            )
        # Summary
        pnl_sum = sum(d["pnl_ticks"] for d in r["per_day"])
        green_pnl = sum(d["pnl_ticks"] for d in r["per_day"] if d["regime"] == "green")
        red_pnl = sum(d["pnl_ticks"] for d in r["per_day"] if d["regime"] == "red")
        flat_pnl = sum(d["pnl_ticks"] for d in r["per_day"] if d["regime"] == "flat")
        log.info(
            f"  TOTAL: {pnl_sum:.2f}t | green={green_pnl:.2f}t, red={red_pnl:.2f}t, flat={flat_pnl:.2f}t"
        )

    # ── Save all results ──
    save_results = {
        "meta": {
            "timestamp": datetime.now().isoformat(),
            "experiment": "30min_lean_oot_validation",
            "train_days": TRAIN_DAYS,
            "val_days": VAL_DAYS,
            "slide_days": SLIDE_DAYS,
            "cost_rt_ticks": COST_RT_TICKS,
            "holdout_splits": HOLDOUT_SPLITS,
            "lean_feature_count": len(lean_features),
            "full_feature_count": len(all_feature_cols),
            "total_days": len(all_dates),
            "date_range": f"{all_dates[0]} -> {all_dates[-1]}",
        },
        "lean_features": lean_features,
        "full_features": all_feature_cols,
        "stability": {
            "lean_wins": lean_wins,
            "total_splits": total_splits,
            "verdict": (
                "STRONG" if lean_wins == total_splits and total_splits > 0
                else "MODERATE" if lean_wins > total_splits / 2 and total_splits > 0
                else "MIXED" if lean_wins == total_splits / 2 and total_splits > 0
                else "WEAK"
            ),
        },
    }

    # Serialize holdout results (strip daily_pnl from trade_sims for compactness)
    for split_name, split_results in all_holdout_results.items():
        save_split = {}
        for k, v in split_results.items():
            if isinstance(v, dict):
                sv = {}
                for kk, vv in v.items():
                    if kk == "trade_sims" and isinstance(vv, dict):
                        sv[kk] = {
                            tk: {tkk: tvv for tkk, tvv in tv.items() if tkk != "daily_pnl"}
                            for tk, tv in vv.items()
                        }
                    elif kk == "regime" and isinstance(vv, dict):
                        sv["regime_summary"] = {
                            "regime_gap": vv.get("regime_gap"),
                            "regime_gap_pass": vv.get("regime_gap_pass"),
                            "regime_gap_detail": vv.get("regime_gap_detail"),
                        }
                    else:
                        sv[kk] = vv
                save_split[k] = sv
            else:
                save_split[k] = v
        save_results[split_name] = save_split

    json_path = OUTPUT_DIR / "lean_oot_results.json"
    with open(json_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {json_path}")

    elapsed = time.time() - t0
    log.info(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    log.info("DONE")


if __name__ == "__main__":
    main()
