#!/usr/bin/env python3
"""
30-Minute LightGBM Paper Trading Engine (LEAN v2)
==================================================

Paper trades the 30-minute LightGBM directional strategy using the LEAN
feature set (43 features instead of 65) identified by feature ablation.

Validation results (lean model):
  - OOT holdout (37d): Sharpe 3.90, Sortino 7.41, WR 59.5%, PF 2.01
  - Regime gap 0.00 (PASS), wins 3/3 holdout splits
  - Long Sharpe 5.81, Short Sharpe 1.46

Architecture:
  - Runs on Jupiter CPU as a PM2 process
  - Every 30 minutes during RTH (09:30-16:00 ET / 13:30-20:00 UTC):
    1. Loads latest minute bar data
    2. Aggregates into 30-min bars
    3. Computes features (identical to training pipeline)
    4. Runs LightGBM inference
    5. If top-15% confidence -> paper trade
    6. Checks existing positions for 30-min time exit
  - Retrains daily using latest 60 days (sliding window, HC #0)
  - Logs to /logs/lh_30min_paper.log
  - Saves trades to /output/lh_30min_paper/trades.csv

PM2:
  pm2 start /home/jupiter/Lvl3Quant/live_trading_linux/lh_30min_paper_engine.py \\
    --name lh-30min-paper --interpreter python3

Author: Claude (autonomous research)
"""

import csv
import gc
import json
import logging
import os
import sys
import time
import traceback
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
OUTPUT_DIR = ROOT / "output" / "lh_30min_paper"
STATE_DIR = OUTPUT_DIR
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = ROOT / "logs"

for d in [OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [LH-30m] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "lh_30min_paper.log")),
    ],
)
log = logging.getLogger("LH-30m")

# ─────────────────────────────────────────────
#  CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
BAR_SIZE_MINUTES = 30
PREDICTION_HORIZON_MINUTES = 30
CONFIDENCE_THRESHOLD_PCT = 5        # top/bottom 5% — optimal per confidence sweep (Sharpe 4.03 vs 2.35 at 15%)
COST_RT_TICKS = 2.376               # market entry + market exit + RT commission
ES_TICK_VALUE = 12.50
STARTING_CAPITAL = 100_000
MAX_POSITION = 1                    # 1 ES contract at a time
TRAIN_DAYS = 60                     # sliding training window

# RTH boundaries (UTC)
RTH_START_UTC = (13, 30)  # 09:30 ET
RTH_END_UTC = (20, 0)     # 16:00 ET

# LightGBM params (identical to training script)
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

# How many days of minute bars to load for feature context (need rolling windows)
LOOKBACK_DAYS = 75  # 60 train + buffer for rolling features

# ── LEAN FEATURE SET (43 features) ──
# Identified by feature ablation: momentum_vol + top-10 OFI by importance.
# OOT validated: Sharpe 3.90 on 37d holdout, wins 3/3 splits.
USE_LEAN_FEATURES = True  # Set False to revert to full 65-feature model
LEAN_FEATURES = [
    'return_bar', 'range_ticks', 'close_position', 'total_volume', 'avg_volume',
    'volume_trend', 'volume_concentration', 'ofi_sum', 'ofi_mean', 'ofi_std',
    'ofi_trend', 'ofi_consistency', 'buy_volume_frac', 'sell_volume_frac',
    'realized_vol', 'vol_of_vol', 'up_vol', 'down_vol', 'vwap_dev_mean',
    'vwap_dev_trend', 'vol_asymmetry', 'ofi_zscore_4bar', 'vol_rel_4bar',
    'ofi_zscore_8bar', 'vol_rel_8bar', 'vol_rel_16bar', 'vol_rel_32bar',
    'sweep_pct_32bar', 'ret_lb_4bar', 'ret_lb_8bar', 'ret_lb_16bar',
    'rvol_4bar', 'rvol_8bar', 'rvol_16bar', 'prev_day_ret', 'tod_sin',
    'tod_cos', 'tod_progress', 'bars_since_open', 'regime_ret_16bar',
    'regime_ret_32bar', 'regime_vol_ratio', 'intraday_direction_strength',
]

# Main loop sleep interval (seconds)
LOOP_SLEEP_SECONDS = 30


# ═══════════════════════════════════════════════════════════════════
#  FEATURE COMPUTATION — IDENTICAL TO longer_horizon_v4_focused.py
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
    IDENTICAL to Section 2 of longer_horizon_v4_focused.py.
    """
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
            "sweep_direction": (
                float(np.sign(sv_arr[np.abs(grp["sv_zscore"].values).argmax()]))
                if len(sv_arr) > 0
                else 0.0
            ),
            # Spread & liquidity
            "spread_mean": spread_arr.mean(),
            "spread_max": spread_arr.max(),
            "spread_trend": _safe_polyfit_slope(spread_arr),
            # Trade intensity
            "trade_count_sum": tc_arr.sum(),
            "trade_count_trend": _safe_polyfit_slope(tc_arr),
            # Volatility
            "realized_vol": (
                float(np.std(ret_arr) * np.sqrt(252 * (390 // bar_size_min)))
                if len(ret_arr) > 1
                else 0
            ),
            "vol_of_vol": float(np.std(np.abs(ret_arr))) if len(ret_arr) > 1 else 0,
            "up_vol": (
                float(np.std(ret_arr[ret_arr > 0])) if np.sum(ret_arr > 0) > 1 else 0
            ),
            "down_vol": (
                float(np.std(ret_arr[ret_arr < 0])) if np.sum(ret_arr < 0) > 1 else 0
            ),
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
    return result


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add multi-bar rolling features using ONLY past data (causal).
    IDENTICAL to Section 3 of longer_horizon_v4_focused.py.
    """
    df = df.sort_values("ts").reset_index(drop=True)

    for w in [4, 8, 16, 32]:
        roll_mean = df["ofi_sum"].rolling(w, min_periods=1).mean()
        roll_std = (
            df["ofi_sum"].rolling(w, min_periods=2).std().fillna(1).replace(0, 1)
        )
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
        on="date",
        how="left",
    )

    # Time-of-day encoding (sin/cos of session progress)
    session_start_hour = 13.5  # 13:30 UTC (09:30 ET)
    session_end_hour = 20.0  # 20:00 UTC (16:00 ET)
    session_len = session_end_hour - session_start_hour

    hour_frac = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    progress = ((hour_frac - session_start_hour) / session_len).clip(0, 1)
    df["tod_sin"] = np.sin(2 * np.pi * progress)
    df["tod_cos"] = np.cos(2 * np.pi * progress)
    df["tod_progress"] = progress.values

    df["bars_since_open"] = df.groupby("date").cumcount()

    # ── REGIME FEATURES ──
    df["regime_ret_16bar"] = df["close"].pct_change(16)
    df["regime_ret_32bar"] = df["close"].pct_change(32)

    rvol_short = df["return_bar"].rolling(4, min_periods=2).std()
    rvol_long = df["return_bar"].rolling(16, min_periods=4).std()
    df["regime_vol_ratio"] = rvol_short / rvol_long.clip(lower=1e-8)

    # Intraday directional strength
    df["intraday_direction_strength"] = df["intraday_cum_ofi"].abs() / (
        df.groupby("date")["ofi_sum"]
        .transform(lambda x: x.abs().cumsum())
        .clip(lower=1)
    )

    return df


def get_feature_columns(df: pd.DataFrame) -> List[str]:
    """Select feature columns, excluding labels, metadata, raw prices.
    If USE_LEAN_FEATURES is True, filters to the 43-feature lean set."""
    exclude_prefixes = (
        "fwd_",
        "direction_",
        "trade_quality_",
        "date",
        "ts",
        "bar_key",
    )
    raw_price_cols = {"open", "high", "low", "close"}

    cols = []
    for c in df.columns:
        if any(c.startswith(p) for p in exclude_prefixes):
            continue
        if c in raw_price_cols:
            continue
        if df[c].dtype in (
            np.float64,
            np.float32,
            np.int64,
            np.int32,
            np.float16,
            np.int16,
        ):
            cols.append(c)

    if USE_LEAN_FEATURES:
        lean_set = set(LEAN_FEATURES)
        filtered = [c for c in cols if c in lean_set]
        # Warn if any lean features are missing from data
        missing = lean_set - set(cols)
        if missing:
            log.warning(f"Lean features missing from data: {missing}")
        log.info(f"Using LEAN feature set: {len(filtered)} of {len(cols)} available")
        return filtered

    return cols


# ═══════════════════════════════════════════════════════════════════
#  PAPER TRADING ENGINE
# ═══════════════════════════════════════════════════════════════════


class LH30MinPaperEngine:
    """Paper trading engine for 30-minute LightGBM directional strategy."""

    def __init__(self):
        self.state_path = STATE_DIR / "state.json"
        self.trades_csv = OUTPUT_DIR / "trades.csv"
        self.model_path = MODEL_DIR / "lgbm_30min_latest.txt"
        self.scaler_path = MODEL_DIR / "scaler_30min_latest.npz"
        self.meta_path = MODEL_DIR / "meta_30min_latest.json"

        self.model = None
        self.feature_cols: Optional[List[str]] = None
        self.scaler_median: Optional[np.ndarray] = None
        self.scaler_iqr: Optional[np.ndarray] = None
        self.pred_quantiles: Optional[Dict[str, float]] = None

        # Load or initialize
        self.state = self._load_state()
        self._ensure_trades_csv()

        log.info(
            f"Engine initialized. Capital: ${self.state['capital']:.0f}, "
            f"Position: {self.state['position']}, "
            f"Total trades: {self.state['total_trades']}"
        )

    # ── State persistence ──

    def _load_state(self) -> Dict:
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to load state: {e} -- starting fresh")
        return {
            "capital": STARTING_CAPITAL,
            "position": 0,  # -1, 0, or +1
            "entry_price": 0.0,
            "entry_time": None,
            "entry_bar_key": None,
            "exit_target_time": None,
            "total_trades": 0,
            "total_pnl_ticks": 0.0,
            "total_pnl_dollars": 0.0,
            "wins": 0,
            "losses": 0,
            "last_signal_bar": None,
            "last_retrain_date": None,
            "created": datetime.now(timezone.utc).isoformat(),
        }

    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _ensure_trades_csv(self):
        """Create trades CSV with header if it doesn't exist."""
        if not self.trades_csv.exists():
            with open(self.trades_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "trade_id",
                        "entry_time",
                        "exit_time",
                        "direction",
                        "entry_price",
                        "exit_price",
                        "pnl_ticks",
                        "pnl_dollars",
                        "cost_ticks",
                        "prediction",
                        "confidence_pct",
                        "hold_minutes",
                        "capital_after",
                    ]
                )

    def _append_trade(self, trade: Dict):
        """Append a completed trade to the CSV."""
        with open(self.trades_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(
                [
                    trade.get("trade_id", self.state["total_trades"]),
                    trade.get("entry_time", ""),
                    trade.get("exit_time", ""),
                    trade.get("direction", ""),
                    trade.get("entry_price", 0),
                    trade.get("exit_price", 0),
                    trade.get("pnl_ticks", 0),
                    trade.get("pnl_dollars", 0),
                    trade.get("cost_ticks", COST_RT_TICKS),
                    trade.get("prediction", 0),
                    trade.get("confidence_pct", 0),
                    trade.get("hold_minutes", 0),
                    trade.get("capital_after", self.state["capital"]),
                ]
            )

    # ── Data loading ──

    def _load_minute_bars(self, n_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
        """Load recent minute bar parquets."""
        files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
        if not files:
            log.error(f"No minute bar files in {MINUTE_BAR_DIR}")
            return pd.DataFrame()

        recent = files[-n_days:] if len(files) >= n_days else files
        frames = []
        for f in recent:
            try:
                df = pd.read_parquet(f)
                df["date"] = f.stem
                frames.append(df)
            except Exception as e:
                log.warning(f"Skip {f.stem}: {e}")

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined["ts_minute"] = pd.to_datetime(combined["ts_minute"], utc=True)
        combined = combined.sort_values("ts_minute").reset_index(drop=True)
        log.info(
            f"Loaded {len(combined):,} minute bars across {len(frames)} days "
            f"({frames[0]['date'].iloc[0]} -> {frames[-1]['date'].iloc[0]})"
        )
        return combined

    # ── Feature pipeline ──

    def _build_feature_df(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """Full feature pipeline: aggregate -> rolling -> feature selection."""
        bars_df = aggregate_to_bars(minute_df, bar_size_min=BAR_SIZE_MINUTES)
        if bars_df.empty:
            return pd.DataFrame()
        bars_df = add_rolling_features(bars_df)
        return bars_df

    # ── Training ──

    def retrain(self, minute_df: pd.DataFrame) -> bool:
        """
        Retrain LightGBM on latest TRAIN_DAYS using sliding window.
        Returns True on success.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            log.error("LightGBM not installed -- pip install lightgbm")
            return False

        bars_df = self._build_feature_df(minute_df)
        if bars_df.empty:
            log.warning("No bars after aggregation")
            return False

        # Get all unique dates
        dates = sorted(bars_df["date"].unique())
        if len(dates) < TRAIN_DAYS + 5:
            log.warning(
                f"Not enough days for training: {len(dates)} < {TRAIN_DAYS + 5}"
            )
            return False

        # Use last TRAIN_DAYS for training
        train_dates = dates[-TRAIN_DAYS:]
        train_mask = bars_df["date"].isin(train_dates)
        train_df = bars_df[train_mask].copy()

        # Forward label: 1-bar forward ticks (30min horizon = 1 bar)
        train_df = train_df.sort_values("ts").reset_index(drop=True)
        fwd_close = train_df["close"].shift(-1)
        fwd_ticks = (fwd_close - train_df["close"]) / 0.25

        # Null out overnight gaps
        ts_now = train_df["ts"].values
        ts_fwd = train_df["ts"].shift(-1).values
        for i in range(len(train_df) - 1):
            if pd.isna(ts_fwd[i]):
                continue
            diff_s = (
                pd.Timestamp(ts_fwd[i]) - pd.Timestamp(ts_now[i])
            ).total_seconds()
            if diff_s > 6 * 3600:
                fwd_ticks.iloc[i] = np.nan

        train_df["fwd_ticks_30min"] = fwd_ticks

        # Feature columns
        self.feature_cols = get_feature_columns(train_df)
        log.info(f"Training features ({len(self.feature_cols)}): {self.feature_cols[:10]}...")

        # Prepare arrays
        X_raw = train_df[self.feature_cols].values.astype(np.float32)
        y = train_df["fwd_ticks_30min"].values.astype(np.float32)

        # Remove NaN targets
        valid = ~np.isnan(y)
        X_raw = X_raw[valid]
        y = y[valid]

        if len(y) < 100:
            log.warning(f"Too few training samples: {len(y)}")
            return False

        # Robust scaling: median/IQR (same as training script)
        self.scaler_median = np.nanmedian(X_raw, axis=0)
        q75 = np.nanpercentile(X_raw, 75, axis=0)
        q25 = np.nanpercentile(X_raw, 25, axis=0)
        self.scaler_iqr = q75 - q25
        self.scaler_iqr[self.scaler_iqr < 1e-8] = 1.0

        X = (X_raw - self.scaler_median) / self.scaler_iqr
        X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)
        X = np.clip(X, -5, 5)

        # Train LightGBM
        params = {**LGBM_PARAMS, "seed": 42}
        train_data = lgb.Dataset(X, label=y, feature_name=self.feature_cols)

        # Use last 10% as validation for early stopping
        n_val = max(int(len(X) * 0.1), 20)
        val_data = lgb.Dataset(
            X[-n_val:], label=y[-n_val:],
            feature_name=self.feature_cols,
            reference=train_data,
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        self.model = lgb.train(
            params,
            train_data,
            num_boost_round=500,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # Compute prediction quantiles on training data for thresholding
        train_preds = self.model.predict(X, num_iteration=self.model.best_iteration)
        conf = CONFIDENCE_THRESHOLD_PCT / 100.0
        self.pred_quantiles = {
            "upper": float(np.percentile(train_preds, 100 * (1 - conf))),
            "lower": float(np.percentile(train_preds, 100 * conf)),
            "p90": float(np.percentile(train_preds, 90)),
            "p10": float(np.percentile(train_preds, 10)),
            "p95": float(np.percentile(train_preds, 95)),
            "p05": float(np.percentile(train_preds, 5)),
            "mean": float(np.mean(train_preds)),
            "std": float(np.std(train_preds)),
        }

        # Save model, scaler, and metadata
        self.model.save_model(str(self.model_path))
        np.savez_compressed(
            str(self.scaler_path),
            median=self.scaler_median,
            iqr=self.scaler_iqr,
        )
        meta = {
            "feature_cols": self.feature_cols,
            "pred_quantiles": self.pred_quantiles,
            "train_dates": [train_dates[0], train_dates[-1]],
            "n_samples": len(y),
            "best_iteration": self.model.best_iteration,
            "retrain_time": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self.state["last_retrain_date"] = train_dates[-1]
        self._save_state()

        # Quick OOT IC on holdout
        ic_oot = np.corrcoef(train_preds[-n_val:], y[-n_val:])[0, 1]

        log.info(
            f"Retrained on {len(train_dates)} days "
            f"({train_dates[0]} -> {train_dates[-1]}), "
            f"{len(y)} samples, best_iter={self.model.best_iteration}, "
            f"holdout IC={ic_oot:.4f}, "
            f"thresholds: upper={self.pred_quantiles['upper']:.2f}, "
            f"lower={self.pred_quantiles['lower']:.2f}"
        )
        return True

    def load_model(self) -> bool:
        """Load a previously trained model from disk."""
        if not self.model_path.exists():
            return False
        if not self.scaler_path.exists():
            return False
        if not self.meta_path.exists():
            return False

        try:
            import lightgbm as lgb

            self.model = lgb.Booster(model_file=str(self.model_path))

            scaler = np.load(str(self.scaler_path))
            self.scaler_median = scaler["median"]
            self.scaler_iqr = scaler["iqr"]

            with open(self.meta_path) as f:
                meta = json.load(f)
            self.feature_cols = meta["feature_cols"]
            self.pred_quantiles = meta["pred_quantiles"]

            log.info(
                f"Loaded model from disk. "
                f"Features: {len(self.feature_cols)}, "
                f"Thresholds: upper={self.pred_quantiles['upper']:.2f}, "
                f"lower={self.pred_quantiles['lower']:.2f}"
            )
            return True
        except Exception as e:
            log.warning(f"Failed to load model: {e}")
            return False

    # ── Inference ──

    def predict(self, bars_df: pd.DataFrame) -> Optional[Dict]:
        """
        Generate prediction for the latest completed 30-min bar.
        Returns signal dict or None.
        """
        if self.model is None or self.feature_cols is None:
            return None
        if bars_df.empty:
            return None

        latest = bars_df.iloc[-1:]

        # Verify all features present
        missing = [c for c in self.feature_cols if c not in latest.columns]
        if missing:
            log.warning(f"Missing {len(missing)} features: {missing[:5]}...")
            return None

        # Extract and scale features
        X_raw = latest[self.feature_cols].values.astype(np.float32)
        X = (X_raw - self.scaler_median) / self.scaler_iqr
        X = np.nan_to_num(X, nan=0.0, posinf=3.0, neginf=-3.0)
        X = np.clip(X, -5, 5)

        pred = self.model.predict(X, num_iteration=self.model.best_iteration)[0]

        # Determine signal
        signal = 0
        confidence_label = "low"
        if pred >= self.pred_quantiles["upper"]:
            signal = 1  # Long
            if pred >= self.pred_quantiles.get("p95", self.pred_quantiles["upper"]):
                confidence_label = "very_high"
            elif pred >= self.pred_quantiles.get("p90", self.pred_quantiles["upper"]):
                confidence_label = "high"
            else:
                confidence_label = "medium"
        elif pred <= self.pred_quantiles["lower"]:
            signal = -1  # Short
            if pred <= self.pred_quantiles.get("p05", self.pred_quantiles["lower"]):
                confidence_label = "very_high"
            elif pred <= self.pred_quantiles.get("p10", self.pred_quantiles["lower"]):
                confidence_label = "high"
            else:
                confidence_label = "medium"

        bar_ts = latest["ts"].iloc[0]
        bar_close = latest["close"].iloc[0]
        bar_date = latest["date"].iloc[0]

        return {
            "prediction_ticks": float(pred),
            "signal": signal,
            "confidence": confidence_label,
            "bar_ts": str(bar_ts),
            "bar_date": bar_date,
            "bar_close": float(bar_close),
            "upper_threshold": self.pred_quantiles["upper"],
            "lower_threshold": self.pred_quantiles["lower"],
        }

    # ── Position management ──

    def _enter_position(self, signal_info: Dict):
        """Enter a new paper position."""
        sig = signal_info["signal"]
        if sig == 0:
            return
        if self.state["position"] != 0:
            log.info("Already positioned -- skip entry")
            return

        price = signal_info["bar_close"]
        entry_ts = signal_info["bar_ts"]

        self.state["position"] = sig
        self.state["entry_price"] = price
        self.state["entry_time"] = entry_ts
        self.state["entry_bar_key"] = entry_ts
        # Exit after 30 minutes (1 bar)
        entry_dt = pd.Timestamp(entry_ts)
        exit_dt = entry_dt + timedelta(minutes=PREDICTION_HORIZON_MINUTES)
        self.state["exit_target_time"] = str(exit_dt)
        self.state["entry_prediction"] = signal_info["prediction_ticks"]
        self.state["entry_confidence"] = signal_info["confidence"]

        direction = "LONG" if sig == 1 else "SHORT"
        log.info(
            f"ENTER {direction} @ {price:.2f} | "
            f"pred={signal_info['prediction_ticks']:.2f}t | "
            f"confidence={signal_info['confidence']} | "
            f"exit_target={exit_dt}"
        )
        self._save_state()

    def _check_exit(self, bars_df: pd.DataFrame) -> bool:
        """
        Check if current position should be exited (30-min time exit).
        Returns True if position was closed.
        """
        if self.state["position"] == 0:
            return False

        exit_target = self.state.get("exit_target_time")
        if exit_target is None:
            return False

        # Get the latest bar timestamp
        if bars_df.empty:
            return False

        latest_ts = bars_df["ts"].iloc[-1]
        exit_ts = pd.Timestamp(exit_target)

        if latest_ts < exit_ts:
            return False  # Not time yet

        # Find the bar closest to exit target for exit price
        # Use the bar at or after the exit target
        exit_candidates = bars_df[bars_df["ts"] >= exit_ts]
        if exit_candidates.empty:
            # Use latest available bar
            exit_bar = bars_df.iloc[-1]
        else:
            exit_bar = exit_candidates.iloc[0]

        exit_price = float(exit_bar["close"])
        entry_price = self.state["entry_price"]
        direction = self.state["position"]

        # Calculate P&L
        raw_ticks = (exit_price - entry_price) / 0.25 * direction
        pnl_ticks = raw_ticks - COST_RT_TICKS
        pnl_dollars = pnl_ticks * ES_TICK_VALUE

        # Update state
        self.state["capital"] += pnl_dollars
        self.state["total_trades"] += 1
        self.state["total_pnl_ticks"] += pnl_ticks
        self.state["total_pnl_dollars"] += pnl_dollars
        if pnl_ticks > 0:
            self.state["wins"] += 1
        else:
            self.state["losses"] += 1

        dir_str = "LONG" if direction == 1 else "SHORT"
        entry_time = self.state.get("entry_time", "")

        # Compute hold duration
        try:
            entry_dt = pd.Timestamp(entry_time)
            exit_dt = exit_bar["ts"]
            hold_minutes = (exit_dt - entry_dt).total_seconds() / 60
        except Exception:
            hold_minutes = PREDICTION_HORIZON_MINUTES

        log.info(
            f"EXIT {dir_str} @ {exit_price:.2f} | "
            f"entry={entry_price:.2f} | "
            f"raw={raw_ticks:.1f}t | "
            f"net={pnl_ticks:.2f}t (${pnl_dollars:.0f}) | "
            f"hold={hold_minutes:.0f}m | "
            f"capital=${self.state['capital']:.0f}"
        )

        # Append to trades CSV
        wr = (
            self.state["wins"] / max(self.state["total_trades"], 1) * 100
        )
        self._append_trade(
            {
                "trade_id": self.state["total_trades"],
                "entry_time": entry_time,
                "exit_time": str(exit_bar["ts"]),
                "direction": dir_str,
                "entry_price": entry_price,
                "exit_price": exit_price,
                "pnl_ticks": round(pnl_ticks, 3),
                "pnl_dollars": round(pnl_dollars, 2),
                "cost_ticks": COST_RT_TICKS,
                "prediction": self.state.get("entry_prediction", 0),
                "confidence_pct": self.state.get("entry_confidence", ""),
                "hold_minutes": round(hold_minutes, 1),
                "capital_after": round(self.state["capital"], 2),
            }
        )

        # Clear position
        self.state["position"] = 0
        self.state["entry_price"] = 0.0
        self.state["entry_time"] = None
        self.state["entry_bar_key"] = None
        self.state["exit_target_time"] = None
        self.state["entry_prediction"] = 0
        self.state["entry_confidence"] = ""
        self._save_state()

        return True

    # ── RTH check ──

    @staticmethod
    def _is_rth(dt: datetime) -> bool:
        """Check if datetime is within RTH (13:30-20:00 UTC)."""
        h, m = dt.hour, dt.minute
        t = h * 60 + m
        rth_start = RTH_START_UTC[0] * 60 + RTH_START_UTC[1]
        rth_end = RTH_END_UTC[0] * 60 + RTH_END_UTC[1]
        return rth_start <= t < rth_end

    @staticmethod
    def _is_30min_boundary(dt: datetime) -> bool:
        """Check if we're near a 30-min bar boundary."""
        return dt.minute in (0, 1, 2, 30, 31, 32)

    @staticmethod
    def _current_bar_key(dt: datetime) -> str:
        """Get the 30-min bar key for a given time."""
        # Floor to 30 minutes
        floored_min = (dt.minute // 30) * 30
        return dt.replace(minute=floored_min, second=0, microsecond=0).isoformat()

    # ── Summary ──

    def daily_summary(self) -> str:
        """Generate a daily summary string."""
        s = self.state
        n = s["total_trades"]
        if n == 0:
            return "No trades yet."

        wr = s["wins"] / n * 100
        avg_pnl = s["total_pnl_ticks"] / n

        lines = [
            f"30-Min LH Paper: {n} trades, WR {wr:.0f}%",
            f"Total PnL: {s['total_pnl_ticks']:.1f} ticks (${s['total_pnl_dollars']:.0f})",
            f"Avg trade: {avg_pnl:.2f} ticks",
            f"Capital: ${s['capital']:.0f}",
        ]
        if s["position"] != 0:
            dir_str = "LONG" if s["position"] == 1 else "SHORT"
            lines.append(f"Open: {dir_str} @ {s['entry_price']:.2f}")

        return " | ".join(lines)

    # ── Backtest mode ──

    def backtest(self, n_test_days: int = 5):
        """
        Run a quick backtest on the last N days to verify the engine works.
        Uses the last TRAIN_DAYS + n_test_days of data, trains on first
        TRAIN_DAYS, then paper-trades the rest bar by bar.
        """
        log.info(f"\n{'=' * 60}")
        log.info(f"BACKTEST MODE: last {n_test_days} trading days")
        log.info(f"{'=' * 60}\n")

        minute_df = self._load_minute_bars(n_days=LOOKBACK_DAYS)
        if minute_df.empty:
            log.error("No minute bar data for backtest")
            return

        # Build full feature set
        bars_df = self._build_feature_df(minute_df)
        if bars_df.empty:
            log.error("No bars after aggregation")
            return

        dates = sorted(bars_df["date"].unique())
        if len(dates) < TRAIN_DAYS + n_test_days:
            log.warning(
                f"Only {len(dates)} days available, need {TRAIN_DAYS + n_test_days}"
            )
            n_test_days = len(dates) - TRAIN_DAYS
            if n_test_days < 1:
                log.error("Not enough data for backtest")
                return

        train_dates = dates[-(TRAIN_DAYS + n_test_days) : -n_test_days]
        test_dates = dates[-n_test_days:]

        log.info(
            f"Train: {train_dates[0]} -> {train_dates[-1]} ({len(train_dates)} days)"
        )
        log.info(f"Test: {test_dates[0]} -> {test_dates[-1]} ({len(test_dates)} days)")

        # Train on the train period
        train_minute_dates = set(train_dates) | set(
            dates[: max(0, len(dates) - TRAIN_DAYS - n_test_days)]
        )
        # Actually just retrain using the minute_df which has all the context
        if not self.retrain(minute_df):
            log.error("Retrain failed -- cannot backtest")
            return

        # Now simulate bar-by-bar on test dates
        test_mask = bars_df["date"].isin(test_dates)
        test_bars = bars_df[test_mask].sort_values("ts").reset_index(drop=True)

        log.info(f"Test bars: {len(test_bars)}")

        # Reset state for backtest
        orig_state = self.state.copy()
        self.state = {
            "capital": STARTING_CAPITAL,
            "position": 0,
            "entry_price": 0.0,
            "entry_time": None,
            "entry_bar_key": None,
            "exit_target_time": None,
            "total_trades": 0,
            "total_pnl_ticks": 0.0,
            "total_pnl_dollars": 0.0,
            "wins": 0,
            "losses": 0,
            "last_signal_bar": None,
            "last_retrain_date": None,
            "entry_prediction": 0,
            "entry_confidence": "",
            "created": datetime.now(timezone.utc).isoformat(),
        }

        for i in range(len(test_bars)):
            # Build a view of all bars up to and including this one
            # (for rolling feature context — already computed)
            current_bars = bars_df[bars_df["ts"] <= test_bars.iloc[i]["ts"]]

            # Check for exit first
            self._check_exit(current_bars)

            # Skip if already positioned
            if self.state["position"] != 0:
                continue

            # Only trade during RTH
            bar_ts = test_bars.iloc[i]["ts"]
            if not self._is_rth(bar_ts):
                continue

            # Predict
            signal = self.predict(current_bars)
            if signal is None:
                continue

            if signal["signal"] != 0:
                self._enter_position(signal)

        # Close any remaining position at last bar
        if self.state["position"] != 0:
            self._check_exit(test_bars)

        # Report
        bt = self.state
        log.info(f"\n{'=' * 60}")
        log.info("BACKTEST RESULTS")
        log.info(f"{'=' * 60}")
        log.info(f"Period: {test_dates[0]} -> {test_dates[-1]} ({len(test_dates)} days)")
        log.info(f"Trades: {bt['total_trades']}")
        if bt["total_trades"] > 0:
            wr = bt["wins"] / bt["total_trades"] * 100
            avg = bt["total_pnl_ticks"] / bt["total_trades"]
            log.info(
                f"Win rate: {wr:.1f}% | "
                f"Avg PnL: {avg:.2f}t | "
                f"Total PnL: {bt['total_pnl_ticks']:.1f}t "
                f"(${bt['total_pnl_dollars']:.0f})"
            )
        log.info(f"Final capital: ${bt['capital']:.0f}")
        log.info(f"{'=' * 60}\n")

        # Restore original state (don't corrupt live state with backtest)
        self.state = orig_state
        self._save_state()

    # ── Main loop ──

    def run_once(self, minute_df: Optional[pd.DataFrame] = None) -> bool:
        """
        Execute one iteration of the engine.
        Returns True if a signal was processed (trade entered or skipped).
        """
        now = datetime.now(timezone.utc)

        # Check RTH
        if not self._is_rth(now):
            return False

        # Load data if not provided
        if minute_df is None:
            minute_df = self._load_minute_bars()
        if minute_df.empty:
            log.warning("No minute bar data available")
            return False

        # Build features
        bars_df = self._build_feature_df(minute_df)
        if bars_df.empty:
            log.warning("No bars after aggregation")
            return False

        # Check for exits first
        self._check_exit(bars_df)

        # Check if we need to retrain (daily)
        today_str = now.strftime("%Y%m%d")
        dates = sorted(bars_df["date"].unique())
        latest_data_date = dates[-1] if dates else None

        if (
            self.state.get("last_retrain_date") is None
            or self.state["last_retrain_date"] < (latest_data_date or "")
        ):
            log.info(f"Retraining (last: {self.state.get('last_retrain_date')}, latest data: {latest_data_date})")
            if not self.retrain(minute_df):
                # Try to load from disk
                if not self.load_model():
                    log.error("No model available -- skip")
                    return False

        # Ensure model is loaded
        if self.model is None:
            if not self.load_model():
                log.warning("No model -- attempting retrain")
                if not self.retrain(minute_df):
                    return False

        # Check if latest bar is new (don't re-signal same bar)
        latest_bar_key = self._current_bar_key(bars_df["ts"].iloc[-1])
        if self.state.get("last_signal_bar") == latest_bar_key:
            return False  # Already processed this bar

        # Skip if already positioned
        if self.state["position"] != 0:
            return False

        # Predict
        signal = self.predict(bars_df)
        if signal is None:
            return False

        self.state["last_signal_bar"] = latest_bar_key

        if signal["signal"] != 0:
            self._enter_position(signal)
            return True
        else:
            log.debug(
                f"No signal (pred={signal['prediction_ticks']:.2f}t, "
                f"range=[{signal['lower_threshold']:.2f}, {signal['upper_threshold']:.2f}])"
            )
            self._save_state()
            return False

    def run_loop(self):
        """Main event loop. Sleeps and wakes at 30-min boundaries."""
        log.info("Starting main loop")
        log.info(f"RTH: {RTH_START_UTC[0]}:{RTH_START_UTC[1]:02d} - "
                 f"{RTH_END_UTC[0]}:{RTH_END_UTC[1]:02d} UTC")
        log.info(f"Bar size: {BAR_SIZE_MINUTES}min, "
                 f"Confidence threshold: top/bottom {CONFIDENCE_THRESHOLD_PCT}%")
        log.info(f"Cost: {COST_RT_TICKS} ticks RT")

        # Try loading model from disk first
        if not self.load_model():
            log.info("No model on disk -- will retrain on first data load")

        minute_df = None
        last_data_load = None

        while True:
            try:
                now = datetime.now(timezone.utc)

                # Reload data every 30 minutes or on first run
                if last_data_load is None or (now - last_data_load).total_seconds() > 1700:
                    minute_df = self._load_minute_bars()
                    last_data_load = now
                    gc.collect()

                if self._is_rth(now):
                    # Near a 30-min boundary -- process
                    if self._is_30min_boundary(now):
                        self.run_once(minute_df)

                    # Also check exits outside boundaries
                    elif self.state["position"] != 0:
                        bars_df = self._build_feature_df(minute_df) if minute_df is not None and not minute_df.empty else pd.DataFrame()
                        if not bars_df.empty:
                            self._check_exit(bars_df)
                else:
                    # Outside RTH -- force close any position
                    if self.state["position"] != 0 and minute_df is not None and not minute_df.empty:
                        bars_df = self._build_feature_df(minute_df)
                        if not bars_df.empty:
                            log.info("RTH ended -- forcing position close")
                            # Override exit target to now
                            self.state["exit_target_time"] = str(
                                now - timedelta(minutes=1)
                            )
                            self._check_exit(bars_df)

                # Sleep until next check
                time.sleep(LOOP_SLEEP_SECONDS)

            except KeyboardInterrupt:
                log.info("Shutting down (KeyboardInterrupt)")
                break
            except Exception as e:
                log.error(f"Loop error: {e}\n{traceback.format_exc()}")
                time.sleep(60)  # Back off on error


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="30-Minute LightGBM Paper Trading Engine"
    )
    parser.add_argument(
        "--backtest",
        type=int,
        default=0,
        metavar="N",
        help="Run backtest on last N trading days (0 = live mode)",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run one iteration and exit (for cron/testing)",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print current state summary and exit",
    )
    args = parser.parse_args()

    engine = LH30MinPaperEngine()

    if args.summary:
        print(engine.daily_summary())
        return

    if args.backtest > 0:
        engine.backtest(n_test_days=args.backtest)
        return

    if args.once:
        engine.run_once()
        log.info(engine.daily_summary())
        return

    # Default: run the main loop
    engine.run_loop()


if __name__ == "__main__":
    main()
