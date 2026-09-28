#!/usr/bin/env python3
"""
30-Minute LightGBM Feature Ablation Study
==========================================

Tests 5 feature groups to determine if the 30-min champion model is
diluted by noisy features. HP sweep found momentum_vol alone has higher
mean IC (0.092) than all features (0.081).

Feature groups:
  1. all            — full 65 features (baseline reproduction)
  2. momentum_vol   — return/vol/regime/tod features only
  3. ofi_queue      — OFI + queue + signed volume features only
  4. top_importance  — top-20 features by LightGBM gain importance
  5. lean           — momentum_vol + top-10 OFI features by importance

Walk-Forward: 60d train, 10d val, slide 5d — SLIDING only (HC #0).
Cost: COST_RT_TICKS = 2.376 (market entry + market exit + RT commission).

Usage:
  cd /home/nick/Lvl3Quant && \
  /home/nick/miniconda3/envs/py311-train/bin/python -u \
      alpha_discovery/lh_30min_feature_ablation.py

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
OUTPUT_DIR = ROOT / "output" / "lh_30min_feature_ablation"
LOG_DIR = ROOT / "logs"

sys.path.insert(0, str(ROOT))

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_DIR.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [FEAT-ABLATION] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "lh_30min_feature_ablation.log")),
    ],
)
log = logging.getLogger("FEAT-ABLATION")

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
#  DATA LOADING (faithfully reused from longer_horizon_v4_focused.py)
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
#  BAR AGGREGATION + FEATURES (faithfully reused from v4_focused)
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
#  FEATURE GROUP CLASSIFICATION
# ═══════════════════════════════════════════════════════════════════


def classify_feature_groups(all_features: List[str]) -> Dict[str, List[str]]:
    """
    Classify features into momentum_vol and ofi_queue subsets.

    momentum_vol: ret_*, rvol_*, realized_vol, vol_*, regime_*, tod_*, bars_since_open
    ofi_queue: ofi_*, q_*, signed_volume_*, sweep_*, intraday_cum_*
    """
    momentum_vol_keywords = [
        "return_bar", "ret_", "rvol_", "realized_vol", "vol_of_vol", "vol_rel_",
        "vol_asymmetry", "regime_", "range_ticks", "close_position",
        "vwap_dev", "volume_trend", "volume_concentration", "total_volume",
        "avg_volume", "tod_", "bars_since_open", "prev_day_ret",
        "intraday_direction_strength", "up_vol", "down_vol",
    ]

    ofi_queue_keywords = [
        "ofi_", "signed_volume_", "buy_volume_frac", "sell_volume_frac",
        "sweep_", "absorption", "q_bid", "q_ask", "q_ofi", "q_top",
        "q_microprice", "toxicity", "trade_count", "spread_",
        "intraday_cum_ofi", "intraday_cum_sv", "prev_day_ofi", "prev_day_sv",
    ]

    momentum_vol = []
    ofi_queue = []

    for f in all_features:
        f_lower = f.lower()
        is_mom = any(kw in f_lower for kw in momentum_vol_keywords)
        is_ofi = any(kw in f_lower for kw in ofi_queue_keywords)

        if is_mom:
            momentum_vol.append(f)
        if is_ofi:
            ofi_queue.append(f)

    log.info(
        f"Feature groups: all={len(all_features)}, "
        f"momentum_vol={len(momentum_vol)}, ofi_queue={len(ofi_queue)}"
    )
    log.info(f"  momentum_vol: {momentum_vol}")
    log.info(f"  ofi_queue: {ofi_queue}")

    return {
        "all": all_features,
        "momentum_vol": momentum_vol,
        "ofi_queue": ofi_queue,
    }


# ═══════════════════════════════════════════════════════════════════
#  TRADE SIMULATION (faithfully from v4_focused)
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
#  IMPORTANCE EXTRACTION (first fold with all features)
# ═══════════════════════════════════════════════════════════════════


def extract_importance_features(
    features_all: np.ndarray,
    labels_all: np.ndarray,
    dates_all: np.ndarray,
    feature_cols: List[str],
    all_dates: List[str],
    ofi_queue_features: List[str],
) -> Tuple[List[str], List[str]]:
    """
    Train one fold with all features, extract importance.
    Returns (top_20_features, top_10_ofi_features).
    """
    _import_lightgbm()

    # Use first valid fold
    fold_train_dates = all_dates[:TRAIN_DAYS]
    fold_val_dates = all_dates[TRAIN_DAYS: TRAIN_DAYS + VAL_DAYS]

    train_mask = np.isin(dates_all, fold_train_dates)
    val_mask = np.isin(dates_all, fold_val_dates)

    X_train_raw = features_all[train_mask].copy()
    X_val_raw = features_all[val_mask].copy()

    # Robust scaling from train only
    train_median = np.nanmedian(X_train_raw, axis=0)
    q75 = np.nanpercentile(X_train_raw, 75, axis=0)
    q25 = np.nanpercentile(X_train_raw, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr < 1e-8] = 1.0

    X_train = np.clip(np.nan_to_num((X_train_raw - train_median) / iqr, nan=0.0, posinf=3.0, neginf=-3.0), -5, 5)
    X_val = np.clip(np.nan_to_num((X_val_raw - train_median) / iqr, nan=0.0, posinf=3.0, neginf=-3.0), -5, 5)

    y_train = labels_all[train_mask]
    y_val = labels_all[val_mask]

    train_valid = ~np.isnan(y_train)
    val_valid = ~np.isnan(y_val)

    X_tr = X_train[train_valid]
    y_tr = y_train[train_valid]
    X_v = X_val[val_valid]
    y_v = y_val[val_valid]

    params = {**LGBM_PARAMS, "seed": 42}

    train_data = lgb.Dataset(X_tr, label=y_tr, feature_name=feature_cols)
    val_data = lgb.Dataset(X_v, label=y_v, feature_name=feature_cols, reference=train_data)

    callbacks = [
        lgb.early_stopping(stopping_rounds=30, verbose=False),
        lgb.log_evaluation(period=0),
    ]

    model = lgb.train(
        params, train_data, num_boost_round=500,
        valid_sets=[val_data], callbacks=callbacks,
    )

    # Extract importance
    importance = model.feature_importance(importance_type="gain")
    feat_imp = sorted(zip(feature_cols, importance), key=lambda x: x[1], reverse=True)

    log.info("Feature importance (top 25):")
    for i, (name, gain) in enumerate(feat_imp[:25]):
        log.info(f"  {i+1:3d}. {name:40s} gain={gain:.1f}")

    # Top 20 overall
    top_20 = [name for name, _ in feat_imp[:20]]

    # Top 10 OFI features by importance
    ofi_set = set(ofi_queue_features)
    ofi_ranked = [(name, gain) for name, gain in feat_imp if name in ofi_set]
    top_10_ofi = [name for name, _ in ofi_ranked[:10]]

    log.info(f"\ntop_importance (top-20): {top_20}")
    log.info(f"top-10 OFI by importance: {top_10_ofi}")

    del model, train_data, val_data
    gc.collect()

    return top_20, top_10_ofi


# ═══════════════════════════════════════════════════════════════════
#  WALK-FORWARD FOR ONE FEATURE GROUP
# ═══════════════════════════════════════════════════════════════════


def run_feature_group(
    group_name: str,
    group_features: List[str],
    features_all: np.ndarray,
    labels_all: np.ndarray,
    dates_all: np.ndarray,
    feature_cols: List[str],
    all_dates: List[str],
    day_returns: Dict[str, float],
) -> Dict:
    """Run full walk-forward for one feature group."""
    _import_lightgbm()

    log.info(f"\n{'#' * 70}")
    log.info(f"# FEATURE GROUP: {group_name} ({len(group_features)} features)")
    log.info(f"# Features: {group_features[:15]}{'...' if len(group_features) > 15 else ''}")
    log.info(f"{'#' * 70}\n")

    # Get column indices for this feature group
    feat_indices = [feature_cols.index(f) for f in group_features if f in feature_cols]
    feat_subset = features_all[:, feat_indices]
    feat_names = [feature_cols[i] for i in feat_indices]

    if len(feat_indices) < 3:
        log.error(f"  Too few features ({len(feat_indices)}) for group {group_name}")
        return {"error": f"too few features ({len(feat_indices)})", "group": group_name}

    all_oot_preds = []
    all_oot_actuals = []
    all_oot_dates_list = []
    fold_ics = []
    fold_results = []
    fold_idx = 0

    for fold_start in range(TRAIN_DAYS, len(all_dates) - VAL_DAYS + 1, SLIDE_DAYS):
        fold_train_dates = all_dates[fold_start - TRAIN_DAYS: fold_start]
        fold_val_dates = all_dates[fold_start: fold_start + VAL_DAYS]

        if len(fold_val_dates) < VAL_DAYS:
            break

        fold_idx += 1

        # Leakage audit
        audit = leakage_audit(
            list(fold_train_dates),
            list(fold_val_dates),
            feat_names,
        )
        if not audit["all_passed"]:
            log.error(f"  FOLD {fold_idx}: Leakage audit FAILED -- skipping")
            continue

        # Split
        train_mask = np.isin(dates_all, fold_train_dates)
        val_mask = np.isin(dates_all, fold_val_dates)

        X_train_raw = feat_subset[train_mask].copy()
        X_val_raw = feat_subset[val_mask].copy()

        # Robust scaling from train only
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
            log.warning(f"  Fold {fold_idx}: too few samples -- skip")
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

        # Fold IC
        if len(val_preds) > 5:
            ic = float(np.corrcoef(val_preds, y_v)[0, 1])
            if not np.isnan(ic):
                fold_ics.append(ic)
        else:
            ic = float("nan")

        fold_results.append({
            "fold": fold_idx,
            "train_start": fold_train_dates[0],
            "train_end": fold_train_dates[-1],
            "val_start": fold_val_dates[0],
            "val_end": fold_val_dates[-1],
            "ic": ic if not np.isnan(ic) else None,
            "best_iter": model.best_iteration,
            "n_train": int(train_valid.sum()),
            "n_val": int(val_valid.sum()),
        })

        all_oot_preds.append(val_preds)
        all_oot_actuals.append(y_v)
        all_oot_dates_list.append(val_dates_fold[val_valid])

        log.info(
            f"  Fold {fold_idx}: IC={ic:.4f}, best_iter={model.best_iteration}, "
            f"n_val={val_valid.sum()}"
        )

        del model, train_data, val_data
        gc.collect()

    if not all_oot_preds:
        log.error(f"No valid folds for {group_name}")
        return {"error": "no valid folds", "group": group_name}

    # Concat OOT
    concat_preds = np.concatenate(all_oot_preds)
    concat_actuals = np.concatenate(all_oot_actuals)
    concat_dates = np.concatenate(all_oot_dates_list)

    # Save concat OOT
    np.savez_compressed(
        str(OUTPUT_DIR / f"{group_name}_concat_oot.npz"),
        preds=concat_preds,
        actuals=concat_actuals,
        dates=concat_dates,
        features=feat_names,
    )

    # Metrics
    concat_ic = float(np.corrcoef(concat_preds, concat_actuals)[0, 1])
    ic_sharpe = (
        float(np.mean(fold_ics) / max(np.std(fold_ics), 1e-6))
        if len(fold_ics) > 2 else float("nan")
    )
    concat_dir_acc = float(np.mean(np.sign(concat_preds) == np.sign(concat_actuals)))

    log.info(
        f"\n{'=' * 60}\n"
        f"CONCAT OOT [{group_name}]: "
        f"IC={concat_ic:.4f}, IC_Sharpe={ic_sharpe:.3f}, "
        f"dir_acc={concat_dir_acc:.3f}, n={len(concat_preds):,}"
        f"\n{'=' * 60}"
    )

    # Trade sim at multiple thresholds
    trade_results = {}
    for conf_pct in [0.05, 0.10, 0.15, 0.20]:
        sim = simulate_trades(concat_preds, concat_actuals, concat_dates, confidence_pct=conf_pct)
        if sim is not None:
            trade_results[f"top_{int(conf_pct*100)}pct"] = sim
            log.info(
                f"  Trades (top {int(conf_pct*100)}%): "
                f"n={sim['n_trades']}, Sharpe={sim['sharpe']:.2f}, "
                f"Sortino={sim['sortino']:.2f}, WR={sim['win_rate']:.1%}, "
                f"PF={sim['profit_factor']:.2f}, avg_pnl={sim['avg_pnl_ticks']:.2f}t"
            )

    # Regime stratification (HC #428 R1)
    regime_results = regime_stratification(
        concat_preds, concat_actuals, concat_dates,
        day_returns=day_returns, confidence_pct=0.10,
    )
    log.info(f"  Regime: {regime_results.get('regime_gap_detail', 'N/A')}")

    # Per-day IC
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
        per_day_ics.append({"date": d, "ic": float(d_ic) if not np.isnan(d_ic) else 0.0, "regime": d_regime})

    green_ics = [x["ic"] for x in per_day_ics if x["regime"] == "green"]
    red_ics = [x["ic"] for x in per_day_ics if x["regime"] == "red"]

    # Primary metrics for comparison (use top 10% as primary)
    primary_sim = trade_results.get("top_10pct", trade_results.get("top_15pct"))
    primary_sharpe = primary_sim["sharpe"] if primary_sim else float("nan")
    primary_sortino = primary_sim["sortino"] if primary_sim else float("nan")
    primary_wr = primary_sim["win_rate"] if primary_sim else float("nan")
    primary_pf = primary_sim["profit_factor"] if primary_sim else float("nan")

    return {
        "group": group_name,
        "n_features": len(feat_indices),
        "features": feat_names,
        "n_folds": fold_idx,
        "n_valid_folds": len(fold_ics),
        "concat_ic": concat_ic,
        "ic_sharpe": ic_sharpe,
        "dir_acc": concat_dir_acc,
        "n_predictions": int(len(concat_preds)),
        "n_oot_days": len(unique_dates),
        "primary_sharpe": float(primary_sharpe),
        "primary_sortino": float(primary_sortino),
        "primary_wr": float(primary_wr),
        "primary_pf": float(primary_pf),
        "regime_gap": float(regime_results.get("regime_gap", float("nan"))),
        "regime_gap_pass": regime_results.get("regime_gap_pass", False),
        "regime_gap_detail": regime_results.get("regime_gap_detail", "N/A"),
        "green_ic": float(np.mean(green_ics)) if green_ics else None,
        "red_ic": float(np.mean(red_ics)) if red_ics else None,
        "trade_sims": trade_results,
        "regime": regime_results,
        "fold_results": fold_results,
    }


# ═══════════════════════════════════════════════════════════════════
#  MAIN ORCHESTRATOR
# ═══════════════════════════════════════════════════════════════════


def main():
    log.info("=" * 70)
    log.info("30-MINUTE LightGBM FEATURE ABLATION STUDY")
    log.info("=" * 70)
    log.info(f"WF: {TRAIN_DAYS}d train, {VAL_DAYS}d val, slide {SLIDE_DAYS}d")
    log.info(f"Cost: {COST_RT_TICKS} ticks RT")
    log.info(f"Output: {OUTPUT_DIR}")

    t0 = time.time()

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

    # Free raw data
    del minute_df, queue_df
    gc.collect()

    # ── Feature columns ──
    all_feature_cols = get_feature_columns(bars_df)
    log.info(f"Total features: {len(all_feature_cols)}")
    log.info(f"All features: {all_feature_cols}")

    # ── Classify feature groups ──
    feature_groups = classify_feature_groups(all_feature_cols)

    # ── Prepare arrays ──
    features_all = bars_df[all_feature_cols].values.astype(np.float32)
    labels_all = bars_df["fwd_ticks"].values.astype(np.float32)
    dates_all = bars_df["date"].values
    all_dates = sorted(bars_df["date"].unique())

    day_close = bars_df.groupby("date")["close"].last()
    day_returns = day_close.pct_change().to_dict()

    log.info(f"Total days: {len(all_dates)} ({all_dates[0]} -> {all_dates[-1]})")

    # ── Step 1: Extract importance from first fold ──
    log.info("\n--- Extracting feature importance (first fold, all features) ---")
    top_20_features, top_10_ofi = extract_importance_features(
        features_all, labels_all, dates_all, all_feature_cols,
        all_dates, feature_groups["ofi_queue"],
    )

    # ── Build lean group: momentum_vol + top-10 OFI ──
    lean_features = list(set(feature_groups["momentum_vol"] + top_10_ofi))
    lean_features = [f for f in all_feature_cols if f in set(lean_features)]  # preserve order

    log.info(f"\nFeature group sizes:")
    log.info(f"  1. all:             {len(all_feature_cols)}")
    log.info(f"  2. momentum_vol:    {len(feature_groups['momentum_vol'])}")
    log.info(f"  3. ofi_queue:       {len(feature_groups['ofi_queue'])}")
    log.info(f"  4. top_importance:  {len(top_20_features)}")
    log.info(f"  5. lean:            {len(lean_features)}")

    # ── Define experiment groups ──
    experiment_groups = {
        "all": all_feature_cols,
        "momentum_vol": feature_groups["momentum_vol"],
        "ofi_queue": feature_groups["ofi_queue"],
        "top_importance": top_20_features,
        "lean": lean_features,
    }

    # ── Run all 5 groups sequentially ──
    all_results = {}
    for group_name, group_features in experiment_groups.items():
        t_group = time.time()
        result = run_feature_group(
            group_name=group_name,
            group_features=group_features,
            features_all=features_all,
            labels_all=labels_all,
            dates_all=dates_all,
            feature_cols=all_feature_cols,
            all_dates=all_dates,
            day_returns=day_returns,
        )
        elapsed_group = time.time() - t_group
        result["runtime_min"] = elapsed_group / 60.0
        all_results[group_name] = result
        log.info(f"  Group {group_name} completed in {elapsed_group/60:.1f} min")
        gc.collect()

    # ═══════════════════════════════════════════════════════════════════
    #  COMPARISON TABLE
    # ═══════════════════════════════════════════════════════════════════

    log.info(f"\n\n{'=' * 100}")
    log.info("FEATURE ABLATION COMPARISON TABLE")
    log.info(f"{'=' * 100}")

    header = (
        f"{'Group':<18s} {'#Feat':>5s} {'IC':>7s} {'IC_Sharpe':>10s} "
        f"{'Sharpe':>7s} {'Sortino':>8s} {'WR':>6s} {'PF':>6s} "
        f"{'RegimeGap':>10s} {'Pass':>5s} {'Runtime':>8s}"
    )
    log.info(header)
    log.info("-" * 100)

    # Collect for winner determination
    valid_results = []

    for group_name in ["all", "momentum_vol", "ofi_queue", "top_importance", "lean"]:
        r = all_results.get(group_name, {})
        if "error" in r:
            log.info(f"  {group_name:<18s} ERROR: {r['error']}")
            continue

        regime_gap_str = f"{r['regime_gap']:.2f}" if not np.isnan(r.get("regime_gap", float("nan"))) else "N/A"
        pass_str = "PASS" if r.get("regime_gap_pass") else "FAIL"

        line = (
            f"{group_name:<18s} {r['n_features']:>5d} "
            f"{r['concat_ic']:>7.4f} {r['ic_sharpe']:>10.3f} "
            f"{r['primary_sharpe']:>7.2f} {r['primary_sortino']:>8.2f} "
            f"{r['primary_wr']:>6.1%} {r['primary_pf']:>6.2f} "
            f"{regime_gap_str:>10s} {pass_str:>5s} "
            f"{r.get('runtime_min', 0):>7.1f}m"
        )
        log.info(line)
        valid_results.append((group_name, r))

    log.info("-" * 100)

    # ── Winner determination ──
    # Primary: regime_gap_pass=True AND highest IC_Sharpe
    # If no regime-pass candidates, pick highest IC_Sharpe overall
    passing = [(name, r) for name, r in valid_results if r.get("regime_gap_pass")]
    if passing:
        winner_name, winner = max(passing, key=lambda x: x[1].get("ic_sharpe", -999))
        log.info(f"\nWINNER (regime-agnostic): {winner_name}")
    else:
        log.info("\nNo groups pass regime gap check. Picking by IC_Sharpe:")
        winner_name, winner = max(valid_results, key=lambda x: x[1].get("ic_sharpe", -999))
        log.info(f"  Best IC_Sharpe: {winner_name} (but FAILS regime gap)")

    log.info(
        f"  IC={winner['concat_ic']:.4f}, IC_Sharpe={winner['ic_sharpe']:.3f}, "
        f"Sharpe={winner['primary_sharpe']:.2f}, Sortino={winner['primary_sortino']:.2f}, "
        f"WR={winner['primary_wr']:.1%}, PF={winner['primary_pf']:.2f}, "
        f"regime_gap={winner.get('regime_gap', 'N/A')}"
    )

    # ── Compare to baseline ──
    baseline = all_results.get("all", {})
    if "error" not in baseline and winner_name != "all":
        ic_delta = winner["concat_ic"] - baseline["concat_ic"]
        sharpe_delta = winner["primary_sharpe"] - baseline["primary_sharpe"]
        log.info(
            f"\n  vs baseline (all features): "
            f"IC delta={ic_delta:+.4f}, Sharpe delta={sharpe_delta:+.2f}"
        )
        if sharpe_delta > 0:
            log.info(f"  CONCLUSION: Feature reduction to '{winner_name}' IMPROVES performance")
        else:
            log.info(f"  CONCLUSION: Full feature set ('all') is still best by Sharpe")

    # ── Dilution analysis ──
    log.info(f"\n--- DILUTION ANALYSIS ---")
    for name, r in valid_results:
        if "error" in r or name == "all":
            continue
        if "error" in baseline:
            continue
        ic_ratio = r["concat_ic"] / max(abs(baseline["concat_ic"]), 1e-6)
        feat_ratio = r["n_features"] / max(baseline["n_features"], 1)
        efficiency = ic_ratio / feat_ratio if feat_ratio > 0 else 0
        log.info(
            f"  {name:<18s}: IC/baseline={ic_ratio:.2f}, "
            f"feat_ratio={feat_ratio:.2f}, IC_efficiency={efficiency:.2f}x"
        )

    # ── Save results JSON ──
    # Strip non-serializable items
    save_results = {}
    for name, r in all_results.items():
        sr = {k: v for k, v in r.items() if k not in ("trade_sims", "regime", "fold_results")}
        if "trade_sims" in r:
            sr["trade_sims_summary"] = {
                k: {kk: vv for kk, vv in v.items() if kk != "daily_pnl"}
                for k, v in r["trade_sims"].items()
            } if isinstance(r["trade_sims"], dict) else {}
        sr["regime_gap_detail"] = r.get("regime_gap_detail", "N/A")
        save_results[name] = sr

    save_results["_meta"] = {
        "timestamp": datetime.now().isoformat(),
        "experiment": "30min_lgbm_feature_ablation",
        "train_days": TRAIN_DAYS,
        "val_days": VAL_DAYS,
        "slide_days": SLIDE_DAYS,
        "cost_rt_ticks": COST_RT_TICKS,
        "winner": winner_name,
        "feature_groups": {k: v for k, v in experiment_groups.items()},
    }

    json_path = OUTPUT_DIR / "ablation_results.json"
    with open(json_path, "w") as f:
        json.dump(save_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {json_path}")

    elapsed = time.time() - t0
    log.info(f"\nTotal runtime: {elapsed / 60:.1f} minutes")
    log.info("DONE")


if __name__ == "__main__":
    main()
