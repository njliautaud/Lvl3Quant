#!/usr/bin/env python3
"""
2-Hour LightGBM Directional Paper Trading Engine
=================================================

Paper trades the 2-hour LightGBM directional model on ES futures.
Runs on Jupiter CPU as a PM2-managed process.

How it works:
  - On startup, loads minute bar data and aggregates to hourly bars
  - Trains a LightGBM regression model on the last 60 days of hourly
    features (with a 5-day purge gap to avoid label leakage)
  - Every hour during RTH (09:30 - 14:30 ET), generates a 2-hour
    directional prediction from the latest hourly bar features
  - If prediction > 0: paper long 1 ES contract at the bar close
  - If prediction < 0: paper short 1 ES contract at the bar close
  - Positions are held for exactly 2 hours then exited at the close
    of the bar 2 hours later
  - All trades, equity, and daily P&L are logged to JSON and CSV

Trading window is 09:30-14:30 ET (13:30-18:30 UTC) because:
  - 2-hour hold means last entry at 14:30 exits at 16:30 (near close)
  - Hours 19-20 UTC (15:00-16:00 ET) are excluded per intraday-clean
    label rules (no entries in the last 2 hours before close)

Model hyperparameters:
  num_leaves=15, max_depth=4, lr=0.02, feature_fraction=0.5,
  bagging_fraction=0.7, min_child_samples=50, lambda_l1=1.0,
  lambda_l2=5.0, n_estimators=500

Cost assumptions: 1.376 ticks RT (market order + AMP commission)

PM2:
  pm2 start lh_2h_paper_ecosystem.config.js

Author: Claude (paper trading engine)
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
from scipy import stats

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
STATE_DIR = ROOT / "live_trading_linux" / "lh_2h_paper_state"
OUTPUT_DIR = ROOT / "output" / "lh_2h_paper"
MODEL_DIR = OUTPUT_DIR / "models"
LOG_DIR = ROOT / "live_trading_linux" / "logs"

for d in [STATE_DIR, OUTPUT_DIR, MODEL_DIR, LOG_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
logging.basicConfig(
    format="%(asctime)s [LH-2h] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_DIR / "lh_2h_paper_engine.out.log")),
    ],
)
log = logging.getLogger("LH-2h")

# ─────────────────────────────────────────────
#  CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
COST_RT_TICKS = 1.376          # market order + commission (canonical)
STARTING_CAPITAL = 100_000
MAX_POSITION = 1               # 1 ES contract at a time

# Safeguards (HC #664 adversarial audit)
MIN_CONFIDENCE_TICKS = 16.0    # Only trade when |pred| > 16t (bottom 20% loses money)
MAX_DAILY_LOSS_TICKS = 100.0   # Stop trading for the day if loss exceeds 100 ticks ($1,250)
STALE_DATA_MAX_DAYS = 2        # Refuse to trade if most recent bar is > 2 trading days old
CONSTANT_PRED_HOURS = 3        # Flag as broken if same prediction for 3+ consecutive hours

# Model / training
TRAIN_DAYS = 60                # sliding training window
PURGE_DAYS = 5                 # purge gap between train and OOT
HORIZON_BARS = 2               # 2 hourly bars forward
LOOKBACK_DAYS = 80             # days of minute bars to load (60 + purge + buffer)

# Trading hours (UTC)
# RTH: 13:30 UTC (09:30 ET) to 20:00 UTC (16:00 ET)
# Last entry: 18:30 UTC (14:30 ET) so 2h hold exits by 20:30 UTC (~16:30 ET)
# We only generate signals at the top of hours 14-18 UTC (10:00-14:00 ET entry,
# plus 13:30 which is partial first hour). Simpler: signal hours are 14..18 UTC.
SIGNAL_HOURS_UTC = [14, 15, 16, 17, 18]  # 10:00-14:00 ET entries
# (13 UTC = 09:00 ET is pre-market, 19-20 UTC = excluded per intraday-clean)

# LightGBM hyperparameters (from lh_2h_enhanced_ic_push.py)
LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mae",
    "verbosity": -1,
    "num_leaves": 15,
    "max_depth": 4,
    "learning_rate": 0.02,
    "feature_fraction": 0.5,
    "bagging_fraction": 0.7,
    "bagging_freq": 5,
    "min_child_samples": 50,
    "lambda_l1": 1.0,
    "lambda_l2": 5.0,
    "n_estimators": 500,
}

# Main loop sleep (seconds)
LOOP_SLEEP_SECONDS = 30


# =============================================================================
#  FEATURE ENGINEERING
#  (Matches lh_2h_enhanced_ic_push.py exactly)
# =============================================================================

def _safe_polyfit_slope(arr: np.ndarray, deg: int = 1) -> float:
    """Linear regression slope, returns 0 on failure."""
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, deg)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def compute_enhanced_hourly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-min bars into hourly bars with enhanced feature set.
    Identical to lh_2h_enhanced_ic_push.py::compute_enhanced_hourly()."""
    df = df.copy()
    df["hour"] = df["ts_minute"].dt.hour
    df["date_str"] = df["date"]
    df["return_1m"] = df.groupby("date_str")["close"].pct_change()

    records = []
    for (date_str, hour), g in df.groupby(["date_str", "hour"]):
        if len(g) < 5:
            continue
        c = g["close"].values.astype(float)
        v = g["volume"].values.astype(float)
        ofi = g["ofi_1min"].values.astype(float)
        sv = g["signed_volume"].values.astype(float)
        ret = g["return_1m"].fillna(0).values.astype(float)
        sp = g["spread_mean"].values.astype(float)
        tc = g["trade_count"].values.astype(float)
        vwap = g["vwap"].values.astype(float)
        mp = g["microprice_close"].values.astype(float)

        # --- STANDARD features ---
        rec = {
            "date": date_str, "hour": hour, "ts": g["ts_minute"].iloc[0],
            "open": c[0], "high": c.max(), "low": c.min(), "close": c[-1],
            "return_1h": (c[-1] / c[0] - 1) if c[0] > 0 else 0,
            "range_ticks": (c.max() - c.min()),
            "close_position": (c[-1] - c.min()) / max(c.max() - c.min(), 1),
            "total_volume": v.sum(),
            "avg_volume": v.mean(),
            "volume_trend": _safe_polyfit_slope(v),
            "volume_concentration": v.max() / max(v.mean(), 1),
            "ofi_sum": ofi.sum(),
            "ofi_mean": ofi.mean(),
            "ofi_trend": _safe_polyfit_slope(ofi),
            "ofi_consistency": np.mean(np.sign(ofi) == np.sign(ofi.sum())) if ofi.sum() != 0 else 0.5,
            "ofi_late_vs_early": ofi[len(ofi) // 2:].sum() - ofi[:len(ofi) // 2].sum(),
            "signed_volume_sum": sv.sum(),
            "signed_volume_ratio": sv.sum() / max(v.sum(), 1),
            "buy_volume_fraction": np.sum(sv[sv > 0]) / max(v.sum(), 1),
            "sell_volume_fraction": -np.sum(sv[sv < 0]) / max(v.sum(), 1),
            "sweep_minutes": int(np.sum(np.abs(sv) > 2 * sv.std())) if sv.std() > 0 else 0,
            "spread_mean": sp.mean(),
            "spread_max": sp.max(),
            "trade_count_sum": tc.sum(),
            "trade_intensity": tc.mean(),
            "return_std": ret.std(),
            "return_skew": float(stats.skew(ret)) if len(ret) > 3 else 0,
            "realized_vol": ret.std() * np.sqrt(60),
        }

        # --- ENHANCED features ---

        # 1. Microprice deviation
        mp_valid = mp[mp > 0]
        c_valid = c[:len(mp_valid)] if len(mp_valid) > 0 else c
        if len(mp_valid) > 0 and len(c_valid) > 0:
            mp_dev = (mp_valid - c_valid[:len(mp_valid)]) / np.maximum(c_valid[:len(mp_valid)], 1)
            rec["microprice_dev_mean"] = mp_dev.mean()
            rec["microprice_dev_trend"] = _safe_polyfit_slope(mp_dev)
            rec["microprice_dev_late"] = mp_dev[-len(mp_dev) // 3:].mean() if len(mp_dev) >= 3 else mp_dev.mean()
        else:
            rec["microprice_dev_mean"] = 0
            rec["microprice_dev_trend"] = 0
            rec["microprice_dev_late"] = 0

        # 2. VWAP deviation
        vwap_valid = vwap[vwap > 0]
        if len(vwap_valid) > 0:
            vwap_dev = (c[:len(vwap_valid)] - vwap_valid) / np.maximum(vwap_valid, 1)
            rec["vwap_dev_final"] = vwap_dev[-1] if len(vwap_dev) > 0 else 0
            rec["vwap_dev_trend"] = _safe_polyfit_slope(vwap_dev)
        else:
            rec["vwap_dev_final"] = 0
            rec["vwap_dev_trend"] = 0

        # 3. Volume-weighted return
        if v.sum() > 0:
            rec["vw_return"] = np.sum(ret * v[:len(ret)]) / v[:len(ret)].sum() if len(ret) <= len(v) else 0
        else:
            rec["vw_return"] = 0

        # 4. Return autocorrelation
        if len(ret) > 10:
            rec["return_autocorr_1"] = np.corrcoef(ret[:-1], ret[1:])[0, 1] if ret[:-1].std() > 0 and ret[1:].std() > 0 else 0
            rec["return_autocorr_5"] = np.corrcoef(ret[:-5], ret[5:])[0, 1] if ret[:-5].std() > 0 and ret[5:].std() > 0 else 0
        else:
            rec["return_autocorr_1"] = 0
            rec["return_autocorr_5"] = 0

        # 5. Trade size distribution
        if tc.sum() > 0 and v.sum() > 0:
            rec["avg_trade_size"] = v.sum() / tc.sum()
            v_sorted = np.sort(v)
            mid = len(v_sorted) // 2
            rec["volume_top_half_ratio"] = v_sorted[mid:].sum() / max(v.sum(), 1)
        else:
            rec["avg_trade_size"] = 0
            rec["volume_top_half_ratio"] = 0.5

        # 6. OFI acceleration
        if len(ofi) > 5:
            ofi_first_half = ofi[:len(ofi) // 2].sum()
            ofi_second_half = ofi[len(ofi) // 2:].sum()
            rec["ofi_acceleration"] = ofi_second_half - ofi_first_half
            if len(ofi) > 3:
                coeffs = np.polyfit(np.arange(len(ofi)), ofi, 2)
                rec["ofi_curvature"] = coeffs[0]
            else:
                rec["ofi_curvature"] = 0
        else:
            rec["ofi_acceleration"] = 0
            rec["ofi_curvature"] = 0

        # 7. Time-of-day
        rec["hour_sin"] = np.sin(2 * np.pi * hour / 24)
        rec["hour_cos"] = np.cos(2 * np.pi * hour / 24)

        # 8. High-low range position
        rng = c.max() - c.min()
        rec["hl_range_position"] = (c[-1] - c.min()) / max(rng, 1)

        # 9. Volume-price divergence
        if len(v) > 5 and v.std() > 0 and c.std() > 0:
            price_direction = np.sign(c[-1] - c[0])
            volume_trend_dir = np.sign(_safe_polyfit_slope(v))
            rec["vol_price_divergence"] = float(price_direction != volume_trend_dir)
        else:
            rec["vol_price_divergence"] = 0

        # 10. Realized vol of OFI
        if ofi.std() > 0:
            rec["ofi_vol"] = ofi.std()
            rec["ofi_vol_normalized"] = ofi.std() / max(abs(ofi.mean()), 1)
        else:
            rec["ofi_vol"] = 0
            rec["ofi_vol_normalized"] = 0

        records.append(rec)

    hourly = pd.DataFrame(records)
    hourly = hourly.sort_values("ts").reset_index(drop=True)
    log.info(f"Computed {len(hourly)} hourly bars with {len(hourly.columns)} columns")
    return hourly


def add_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-hour rolling features (matches lh_2h_enhanced_ic_push.py)."""
    for w in [2, 4, 6]:
        lbl = f"{w}h"
        df[f"ofi_sum_{lbl}"] = df["ofi_sum"].rolling(w, min_periods=1).sum()
        df[f"ofi_trend_{lbl}"] = df["ofi_trend"].rolling(w, min_periods=1).mean()
        df[f"sv_sum_{lbl}"] = df["signed_volume_sum"].rolling(w, min_periods=1).sum()
        df[f"volume_ma_{lbl}"] = df["total_volume"].rolling(w, min_periods=1).mean()
        df[f"volume_vs_ma_{lbl}"] = df["total_volume"] / df[f"volume_ma_{lbl}"].clip(lower=1)
        df[f"vol_trend_{lbl}"] = df["realized_vol"].rolling(w, min_periods=1).apply(
            lambda x: _safe_polyfit_slope(x.values), raw=False,
        )

    # Enhanced rolling features
    for w in [2, 4, 6]:
        lbl = f"{w}h"
        df[f"mpdev_sum_{lbl}"] = df["microprice_dev_mean"].rolling(w, min_periods=1).sum()
        df[f"vwapdev_sum_{lbl}"] = df["vwap_dev_final"].rolling(w, min_periods=1).sum()
        df[f"ofi_accel_{lbl}"] = df["ofi_acceleration"].rolling(w, min_periods=1).sum()
        df[f"autocorr_mean_{lbl}"] = df["return_autocorr_1"].rolling(w, min_periods=1).mean()

    # Price momentum
    df["mom_2h"] = df["close"].pct_change(2)
    df["mom_4h"] = df["close"].pct_change(4)
    df["mom_6h"] = df["close"].pct_change(6)

    return df


def add_regime_context(df: pd.DataFrame) -> pd.DataFrame:
    """Regime features (matches lh_2h_enhanced_ic_push.py)."""
    df["vol_20h"] = df["realized_vol"].rolling(20, min_periods=5).mean()
    # Use expanding quantile rank to avoid look-ahead bias (HC #664 audit fix).
    # Previously used pd.qcut on full dataset — leaked future vol distribution.
    # Now: at each row, rank is computed only vs. data seen so far.
    expanding_rank = df["vol_20h"].expanding(min_periods=5).rank(pct=True)
    df["vol_regime_f"] = pd.cut(
        expanding_rank, bins=[0, 1/3, 2/3, 1.0], labels=[0, 1, 2], include_lowest=True
    ).astype(float)
    df["trend_8h"] = df["close"].pct_change(8)
    df["trend_20h"] = df["close"].pct_change(20)
    df["trend_regime"] = np.where(
        df["trend_20h"] > 0.005, 1, np.where(df["trend_20h"] < -0.005, -1, 0)
    )
    return df


def get_feature_cols(df: pd.DataFrame) -> List[str]:
    """Get all feature columns (exclude metadata and labels)."""
    exclude = {"date", "hour", "ts", "open", "high", "low", "close",
               "fwd_ticks", "date_str"}
    return [
        c for c in df.columns
        if c not in exclude and df[c].dtype in ["float64", "float32", "int64", "int32"]
    ]


# =============================================================================
#  PAPER TRADING ENGINE
# =============================================================================

class LH2hPaperEngine:
    """Paper trading engine for 2-hour LightGBM directional strategy."""

    def __init__(self):
        self.state_path = STATE_DIR / "state.json"
        self.trades_csv = OUTPUT_DIR / "trades.csv"
        self.equity_csv = OUTPUT_DIR / "equity.csv"
        self.model_path = MODEL_DIR / "lgbm_2h_latest.txt"
        self.meta_path = MODEL_DIR / "meta_2h_latest.json"

        self.model = None
        self.feature_cols: Optional[List[str]] = None
        self.hourly_df: Optional[pd.DataFrame] = None

        # Load or initialize state
        self.state = self._load_state()
        self._ensure_trades_csv()
        self._ensure_equity_csv()

        log.info(
            f"Engine initialized. Capital: ${self.state['capital']:,.0f}, "
            f"Position: {self.state['position']}, "
            f"Total trades: {self.state['total_trades']}, "
            f"Total P&L: {self.state['total_pnl_ticks']:.2f} ticks "
            f"(${self.state['total_pnl_dollars']:,.2f})"
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
            "position": 0,           # -1, 0, or +1
            "entry_price": 0.0,
            "entry_time": None,
            "entry_hour_utc": None,
            "exit_target_hour_utc": None,
            "entry_prediction": 0.0,
            "total_trades": 0,
            "total_pnl_ticks": 0.0,
            "total_pnl_dollars": 0.0,
            "wins": 0,
            "losses": 0,
            "daily_pnl": {},         # date -> pnl_dollars
            "last_signal_hour": None, # "YYYYMMDD-HH" to avoid double-trading
            "last_retrain_date": None,
            "created": datetime.now(timezone.utc).isoformat(),
        }

    def _save_state(self):
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _ensure_trades_csv(self):
        if not self.trades_csv.exists():
            with open(self.trades_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "trade_id", "entry_time", "exit_time", "direction",
                    "entry_price", "exit_price", "gross_pnl_ticks",
                    "cost_ticks", "net_pnl_ticks", "net_pnl_dollars",
                    "prediction", "hold_hours", "capital_after",
                    "exit_reason", "confidence_decile", "hour_of_day_utc",
                ])

    def _ensure_equity_csv(self):
        if not self.equity_csv.exists():
            with open(self.equity_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "timestamp", "capital", "position", "total_trades",
                    "total_pnl_dollars", "win_rate",
                ])

    def _append_trade(self, trade: Dict):
        with open(self.trades_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trade.get("trade_id", self.state["total_trades"]),
                trade.get("entry_time", ""),
                trade.get("exit_time", ""),
                trade.get("direction", ""),
                trade.get("entry_price", 0),
                trade.get("exit_price", 0),
                trade.get("gross_pnl_ticks", 0),
                trade.get("cost_ticks", COST_RT_TICKS),
                trade.get("net_pnl_ticks", 0),
                trade.get("net_pnl_dollars", 0),
                trade.get("prediction", 0),
                trade.get("hold_hours", HORIZON_BARS),
                trade.get("capital_after", self.state["capital"]),
                trade.get("exit_reason", "hold_expired"),
                trade.get("confidence_decile", self._get_confidence_decile(trade.get("prediction", 0))),
                trade.get("hour_of_day_utc", ""),
            ])

    def _log_equity(self):
        wr = self.state["wins"] / max(self.state["total_trades"], 1)
        with open(self.equity_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                datetime.now(timezone.utc).isoformat(),
                self.state["capital"],
                self.state["position"],
                self.state["total_trades"],
                self.state["total_pnl_dollars"],
                f"{wr:.4f}",
            ])

    # ── Data loading ──

    def _load_minute_bars(self, n_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
        """Load recent minute bar parquets."""
        files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
        if not files:
            log.error(f"No minute bar files found")
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

    def _build_hourly_df(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        """Full feature pipeline: minute bars -> hourly features."""
        hourly = compute_enhanced_hourly(minute_df)
        if hourly.empty:
            return pd.DataFrame()
        hourly = add_rolling_features(hourly)
        hourly = add_regime_context(hourly)
        return hourly

    # ── Training ──

    def retrain(self, minute_df: pd.DataFrame) -> bool:
        """
        Retrain LightGBM on latest TRAIN_DAYS with PURGE_DAYS gap.
        Uses sliding window (HC #0). Returns True on success.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            log.error("LightGBM not installed -- pip install lightgbm")
            return False

        hourly = self._build_hourly_df(minute_df)
        if hourly.empty:
            log.warning("No hourly bars after aggregation")
            return False

        # Add forward labels (2-hour forward ticks)
        # close is stored in tick units (1 unit = 1 tick = 0.25 ES pts).
        # Raw difference is already in ticks — no division needed.
        hourly = hourly.sort_values("ts").reset_index(drop=True)
        hourly["fwd_ticks"] = hourly["close"].shift(-HORIZON_BARS) - hourly["close"]

        # Null out overnight gaps (if 2 bars ahead is a different day with >6h gap)
        for i in range(len(hourly) - HORIZON_BARS):
            ts_now = hourly["ts"].iloc[i]
            ts_fwd = hourly["ts"].iloc[i + HORIZON_BARS]
            diff_s = (ts_fwd - ts_now).total_seconds()
            if diff_s > 8 * 3600:  # overnight gap
                hourly.loc[hourly.index[i], "fwd_ticks"] = np.nan

        # Filter to intraday-clean: exclude hours 19-20 UTC (last 2h before close)
        hourly_clean = hourly[~hourly["hour"].isin([19, 20])].copy()
        hourly_clean = hourly_clean.dropna(subset=["fwd_ticks"])

        dates = sorted(hourly_clean["date"].unique())
        if len(dates) < TRAIN_DAYS + PURGE_DAYS:
            log.warning(f"Not enough days: {len(dates)} < {TRAIN_DAYS + PURGE_DAYS}")
            return False

        # Use last TRAIN_DAYS (excluding PURGE_DAYS most recent)
        train_dates = dates[-(TRAIN_DAYS + PURGE_DAYS):-PURGE_DAYS]
        train_df = hourly_clean[hourly_clean["date"].isin(train_dates)].copy()

        self.feature_cols = get_feature_cols(train_df)
        log.info(f"Training with {len(self.feature_cols)} features on {len(train_dates)} days")

        X_train = train_df[self.feature_cols].fillna(0).values.astype(np.float32)
        y_train = train_df["fwd_ticks"].values.astype(np.float32)

        if len(y_train) < 100:
            log.warning(f"Too few training samples: {len(y_train)}")
            return False

        # Split last 20% for early stopping validation
        split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split], X_train[split:]
        y_tr, y_val = y_train[:split], y_train[split:]

        params = {**LGBM_PARAMS, "seed": 42}
        model = lgb.LGBMRegressor(**params, early_stopping_rounds=50)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(0)],
        )

        self.model = model

        # Save model and metadata
        model.booster_.save_model(str(self.model_path))
        meta = {
            "feature_cols": self.feature_cols,
            "train_dates": [train_dates[0], train_dates[-1]],
            "purge_days": PURGE_DAYS,
            "n_samples": len(y_train),
            "best_iteration": model.best_iteration_,
            "retrain_time": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        # Quick training IC
        train_preds = model.predict(X_train)
        ic = float(stats.spearmanr(train_preds, y_train)[0])

        self.state["last_retrain_date"] = train_dates[-1]
        self._save_state()

        # Cache full hourly df for live inference
        self.hourly_df = hourly

        log.info(
            f"Retrained: {len(train_dates)} days ({train_dates[0]} -> {train_dates[-1]}), "
            f"{len(y_train)} samples, best_iter={model.best_iteration_}, "
            f"train IC={ic:.4f}"
        )
        return True

    def load_model(self) -> bool:
        """Load a previously trained model from disk."""
        if not self.model_path.exists() or not self.meta_path.exists():
            return False
        try:
            import lightgbm as lgb
            self.model = lgb.Booster(model_file=str(self.model_path))
            with open(self.meta_path) as f:
                meta = json.load(f)
            self.feature_cols = meta["feature_cols"]
            log.info(f"Loaded model from disk. Features: {len(self.feature_cols)}")
            return True
        except Exception as e:
            log.warning(f"Failed to load model: {e}")
            return False

    # ── Inference ──

    def predict_latest(self) -> Optional[Dict]:
        """
        Generate prediction for the latest completed hourly bar.
        Returns signal dict or None.
        """
        if self.model is None or self.feature_cols is None:
            log.warning("No model loaded")
            return None
        if self.hourly_df is None or self.hourly_df.empty:
            log.warning("No hourly data available")
            return None

        latest = self.hourly_df.iloc[-1]

        # Check we have all features
        missing = [c for c in self.feature_cols if c not in self.hourly_df.columns]
        if missing:
            log.warning(f"Missing {len(missing)} features: {missing[:5]}")
            return None

        X = self.hourly_df.iloc[[-1]][self.feature_cols].fillna(0).values.astype(np.float32)

        # Handle LGBMRegressor vs Booster
        if hasattr(self.model, "predict"):
            if hasattr(self.model, "best_iteration_"):
                pred = self.model.predict(X)[0]
            else:
                pred = self.model.predict(X, num_iteration=self.model.best_iteration)[0]
        else:
            pred = float(self.model.predict(X)[0])

        # Confidence threshold: |pred| >= 16 ticks (HC #664 adversarial audit).
        # Bottom 20% of predictions loses money. 16t threshold per deep validation.

        return {
            "prediction_ticks": float(pred),
            "signal": (1 if pred > 0 else -1) if abs(pred) >= MIN_CONFIDENCE_TICKS else 0,
            "bar_ts": str(latest["ts"]),
            "bar_date": str(latest["date"]),
            "bar_hour": int(latest["hour"]),
            "bar_close": float(latest["close"]),
        }

    # ── Position management ──

    def _enter_position(self, signal_info: Dict):
        """Enter a new paper position."""
        sig = signal_info["signal"]
        if sig == 0:
            log.info("Prediction is 0 -- no trade")
            return
        if self.state["position"] != 0:
            log.info("Already positioned -- skip entry")
            return

        price = signal_info["bar_close"]
        entry_ts = signal_info["bar_ts"]
        entry_hour = signal_info["bar_hour"]

        self.state["position"] = sig
        self.state["entry_price"] = price
        self.state["entry_time"] = entry_ts
        self.state["entry_hour_utc"] = entry_hour
        self.state["exit_target_hour_utc"] = entry_hour + HORIZON_BARS
        self.state["entry_prediction"] = signal_info["prediction_ticks"]

        direction = "LONG" if sig == 1 else "SHORT"
        log.info(
            f"ENTER {direction} @ {price:.2f} | "
            f"pred={signal_info['prediction_ticks']:+.3f} ticks | "
            f"exit target: hour {entry_hour + HORIZON_BARS} UTC"
        )
        self._save_state()

    def _close_position(self, exit_reason: str = "hold_expired") -> bool:
        """
        Close the current position and record the trade.
        Called by both scheduled exit and mid-trade invalidation.
        exit_reason: 'hold_expired' | 'thesis_invalidated'
        Returns True if position was closed.
        """
        if self.state["position"] == 0:
            return False
        if self.hourly_df is None or self.hourly_df.empty:
            return False

        latest = self.hourly_df.iloc[-1]
        latest_date = str(latest["date"])

        exit_price = float(latest["close"])
        entry_price = self.state["entry_price"]
        direction = self.state["position"]

        # Calculate P&L
        # close prices are stored in tick units (1 unit = 1 tick = 0.25 ES pts),
        # so the raw price difference IS already in ticks. No division needed.
        if direction == 1:  # long
            gross_ticks = exit_price - entry_price
        else:  # short
            gross_ticks = entry_price - exit_price

        net_ticks = gross_ticks - COST_RT_TICKS
        net_dollars = net_ticks * ES_TICK_VALUE

        # Update state
        self.state["total_trades"] += 1
        self.state["total_pnl_ticks"] += net_ticks
        self.state["total_pnl_dollars"] += net_dollars
        self.state["capital"] += net_dollars

        if net_ticks > 0:
            self.state["wins"] += 1
        else:
            self.state["losses"] += 1

        # Track daily P&L
        trade_date = latest_date
        if trade_date not in self.state["daily_pnl"]:
            self.state["daily_pnl"][trade_date] = 0.0
        self.state["daily_pnl"][trade_date] += net_dollars

        # Track mid-trade invalidation stats
        if exit_reason == "thesis_invalidated":
            self.state.setdefault("thesis_invalidations", 0)
            self.state["thesis_invalidations"] += 1

        dir_str = "LONG" if direction == 1 else "SHORT"
        wr = self.state["wins"] / max(self.state["total_trades"], 1)

        # Compute hold duration
        entry_hour = self.state.get("entry_hour_utc", 0)
        actual_hold = int(latest["hour"]) - entry_hour if entry_hour else HORIZON_BARS

        reason_tag = " [THESIS INVALIDATED]" if exit_reason == "thesis_invalidated" else ""
        log.info(
            f"EXIT {dir_str}{reason_tag} @ {exit_price:.2f} | "
            f"gross={gross_ticks:+.2f}t, cost={COST_RT_TICKS:.3f}t, "
            f"net={net_ticks:+.2f}t (${net_dollars:+,.2f}) | "
            f"held {actual_hold}h of {HORIZON_BARS}h | "
            f"Capital: ${self.state['capital']:,.2f} | "
            f"WR: {wr:.1%} ({self.state['total_trades']} trades)"
        )

        # Log trade
        self._append_trade({
            "trade_id": self.state["total_trades"],
            "entry_time": self.state["entry_time"],
            "exit_time": str(latest["ts"]),
            "direction": dir_str,
            "entry_price": entry_price,
            "exit_price": exit_price,
            "gross_pnl_ticks": gross_ticks,
            "cost_ticks": COST_RT_TICKS,
            "net_pnl_ticks": net_ticks,
            "net_pnl_dollars": net_dollars,
            "prediction": self.state.get("entry_prediction", 0),
            "hold_hours": actual_hold,
            "capital_after": self.state["capital"],
            "exit_reason": exit_reason,
            "confidence_decile": self._get_confidence_decile(self.state.get("entry_prediction", 0)),
            "hour_of_day_utc": self.state.get("entry_hour_utc", ""),
        })

        # Clear position
        self.state["position"] = 0
        self.state["entry_price"] = 0.0
        self.state["entry_time"] = None
        self.state["entry_hour_utc"] = None
        self.state["exit_target_hour_utc"] = None
        self.state["entry_prediction"] = 0.0

        self._save_state()
        self._log_equity()
        return True

    def _check_exit(self) -> bool:
        """
        Check if current position should be exited (2-hour hold expired).
        Returns True if position was closed.
        """
        if self.state["position"] == 0:
            return False
        if self.hourly_df is None or self.hourly_df.empty:
            return False

        latest = self.hourly_df.iloc[-1]
        latest_hour = int(latest["hour"])
        latest_date = str(latest["date"])
        entry_date = None

        # Parse entry time to get entry date
        entry_time_str = self.state.get("entry_time", "")
        if entry_time_str:
            try:
                entry_dt = pd.Timestamp(entry_time_str)
                entry_date = entry_dt.strftime("%Y%m%d")
            except Exception:
                pass

        exit_target = self.state.get("exit_target_hour_utc")
        if exit_target is None:
            return False

        # Exit if we've reached or passed the target hour (same day)
        # or if it's a new day (force exit)
        should_exit = False
        if entry_date and latest_date > entry_date:
            should_exit = True  # new day, force exit
        elif latest_hour >= exit_target:
            should_exit = True

        if not should_exit:
            return False

        return self._close_position(exit_reason="hold_expired")

    def _check_mid_trade_invalidation(self) -> bool:
        """
        Mid-trade thesis validation (HC #648).

        While in position, re-run the model on the latest hourly bar features.
        If the updated prediction FLIPS SIGN relative to the entry direction
        AND the magnitude exceeds the confidence threshold, exit early.

        This catches cases where post-entry order flow contradicts the thesis:
        e.g., entered LONG because OFI was strongly positive, but in the next
        hour OFI reversed negative — the directional thesis is invalidated.

        Returns True if position was closed due to invalidation.
        """
        if self.state["position"] == 0:
            return False
        if self.model is None or self.hourly_df is None:
            return False

        # Only check once we have a new bar since entry (1h into the hold)
        latest_hour = int(self.hourly_df.iloc[-1]["hour"])
        entry_hour = self.state.get("entry_hour_utc")
        exit_target = self.state.get("exit_target_hour_utc")

        if entry_hour is None or exit_target is None:
            return False

        # Don't invalidate on the same bar as entry (need fresh data)
        if latest_hour <= entry_hour:
            return False

        # Don't invalidate on the exit bar (normal exit handles that)
        if latest_hour >= exit_target:
            return False

        # Re-run prediction on latest features
        signal = self.predict_latest()
        if signal is None:
            return False

        new_pred = signal["prediction_ticks"]
        entry_direction = self.state["position"]  # +1 or -1
        entry_pred = self.state.get("entry_prediction", 0)

        # Thesis invalidation: prediction sign flipped with confidence
        pred_flipped = (
            (entry_direction == 1 and new_pred < -MIN_CONFIDENCE_TICKS) or
            (entry_direction == -1 and new_pred > MIN_CONFIDENCE_TICKS)
        )

        if not pred_flipped:
            # Thesis still intact (or prediction is weak/neutral)
            log.debug(
                f"Mid-trade check: entry_dir={entry_direction}, "
                f"new_pred={new_pred:+.1f}t — thesis INTACT"
            )
            return False

        # Thesis invalidated — exit early
        log.info(
            f"⚠️ THESIS INVALIDATED: entry_dir={'LONG' if entry_direction == 1 else 'SHORT'}, "
            f"entry_pred={entry_pred:+.1f}t, new_pred={new_pred:+.1f}t — EXITING EARLY"
        )
        return self._close_position(exit_reason="thesis_invalidated")

    # ── Trade logging helpers ──

    def _get_confidence_decile(self, pred_ticks: float) -> int:
        """Map prediction magnitude to confidence decile (1=weakest, 10=strongest)."""
        abs_pred = abs(pred_ticks)
        # Approximate decile boundaries from walkforward analysis
        # These should be calibrated from actual prediction distribution
        boundaries = [0, 5, 10, 16, 22, 30, 40, 52, 68, 90]
        for i, b in enumerate(boundaries):
            if abs_pred < b:
                return i
        return 10

    # ── Safeguards (HC #664 adversarial audit) ──

    def _check_stale_data(self) -> bool:
        """
        STALE DATA GUARD: Refuse to trade if the most recent minute bar file
        is more than STALE_DATA_MAX_DAYS trading days old.
        Returns True if data is STALE (should NOT trade).
        """
        files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
        if not files:
            log.error("STALE DATA GUARD: No minute bar files found. REFUSING TO TRADE.")
            return True

        latest_file = files[-1]
        # Parse date from filename (format: YYYYMMDD.parquet)
        try:
            latest_date_str = latest_file.stem
            latest_date = datetime.strptime(latest_date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        except (ValueError, AttributeError):
            log.error(f"STALE DATA GUARD: Cannot parse date from {latest_file.name}")
            return True

        now = datetime.now(timezone.utc)
        # Calendar days difference (trading days ~ calendar * 5/7)
        delta_days = (now - latest_date).days
        # Approximate trading days (exclude weekends)
        trading_days_approx = delta_days * 5 / 7

        if trading_days_approx > STALE_DATA_MAX_DAYS:
            log.warning(
                f"STALE DATA GUARD: Latest bar file is {latest_file.stem} "
                f"({delta_days} calendar days / ~{trading_days_approx:.0f} trading days old). "
                f"Threshold: {STALE_DATA_MAX_DAYS} trading days. REFUSING TO TRADE."
            )
            return True

        return False

    def _check_constant_predictions(self, new_pred: float) -> bool:
        """
        PREDICTION SANITY CHECK: If model prediction is constant (same value)
        for CONSTANT_PRED_HOURS consecutive hours, flag as broken.
        Returns True if predictions appear BROKEN (should NOT trade).
        """
        # Track recent predictions in state
        recent_preds = self.state.get("recent_predictions", [])
        recent_preds.append(round(new_pred, 2))

        # Keep only last N predictions
        if len(recent_preds) > CONSTANT_PRED_HOURS + 1:
            recent_preds = recent_preds[-(CONSTANT_PRED_HOURS + 1):]
        self.state["recent_predictions"] = recent_preds

        if len(recent_preds) >= CONSTANT_PRED_HOURS:
            last_n = recent_preds[-CONSTANT_PRED_HOURS:]
            if len(set(last_n)) == 1:
                log.error(
                    f"PREDICTION SANITY CHECK: Model produced identical prediction "
                    f"({last_n[0]}) for {CONSTANT_PRED_HOURS} consecutive hours. "
                    f"Model appears BROKEN. REFUSING TO TRADE."
                )
                return True

        return False

    def _check_daily_loss_limit(self) -> bool:
        """
        MAX DAILY LOSS: Stop trading for the day if cumulative loss exceeds
        MAX_DAILY_LOSS_TICKS. Prevents runaway losing.
        Returns True if daily loss limit HIT (should NOT trade).
        """
        today = datetime.now(timezone.utc).strftime("%Y%m%d")
        daily_pnl = self.state.get("daily_pnl", {})
        today_pnl_dollars = daily_pnl.get(today, 0.0)
        today_pnl_ticks = today_pnl_dollars / ES_TICK_VALUE

        if today_pnl_ticks < -MAX_DAILY_LOSS_TICKS:
            log.warning(
                f"MAX DAILY LOSS: Today's loss is {today_pnl_ticks:.1f} ticks "
                f"(${today_pnl_dollars:,.0f}), exceeds limit of {MAX_DAILY_LOSS_TICKS} ticks "
                f"(${MAX_DAILY_LOSS_TICKS * ES_TICK_VALUE:,.0f}). STOP TRADING FOR TODAY."
            )
            return True

        return False

    # ── Hourly tick ──

    def _should_trade_this_hour(self, hour_utc: int, date_str: str) -> bool:
        """Check if we should generate a signal this hour."""
        if hour_utc not in SIGNAL_HOURS_UTC:
            return False

        # Avoid double-trading the same hour
        hour_key = f"{date_str}-{hour_utc:02d}"
        if self.state.get("last_signal_hour") == hour_key:
            return False

        return True

    def hourly_tick(self, minute_df: pd.DataFrame):
        """
        Called every loop iteration. Checks if a new hourly bar has completed
        and processes signals / exits accordingly.
        """
        now_utc = datetime.now(timezone.utc)
        current_hour = now_utc.hour
        current_date = now_utc.strftime("%Y%m%d")

        # Rebuild hourly features from latest minute bars
        self.hourly_df = self._build_hourly_df(minute_df)
        if self.hourly_df is None or self.hourly_df.empty:
            return

        # Check for exits first (scheduled 2h hold expiry)
        self._check_exit()

        # Mid-trade thesis validation: re-run model while in position
        # and exit early if prediction flips sign (HC #648)
        self._check_mid_trade_invalidation()

        # Check for new entry signals
        if self._should_trade_this_hour(current_hour, current_date):
            # ── SAFEGUARDS (HC #664) ──
            # 1. Stale data guard
            if self._check_stale_data():
                return  # Refuse to trade on old data

            # 2. Daily loss limit
            if self._check_daily_loss_limit():
                return  # Stop for the day

            signal = self.predict_latest()
            if signal is not None:
                # 3. Prediction sanity check
                if self._check_constant_predictions(signal["prediction_ticks"]):
                    return  # Model appears broken

                log.info(
                    f"Signal at hour {current_hour} UTC: "
                    f"pred={signal['prediction_ticks']:+.3f}t, "
                    f"signal={'LONG' if signal['signal'] == 1 else 'SHORT' if signal['signal'] == -1 else 'FLAT'}, "
                    f"confidence_decile={self._get_confidence_decile(signal['prediction_ticks'])}"
                )

                # Record that we processed this hour
                hour_key = f"{current_date}-{current_hour:02d}"
                self.state["last_signal_hour"] = hour_key

                # Enter if flat
                if self.state["position"] == 0:
                    self._enter_position(signal)
                else:
                    log.info(
                        f"Already in position ({self.state['position']}), "
                        f"skipping new entry"
                    )

                self._save_state()

    # ── Backfill (simulate historical trades on startup) ──

    def backfill(self, minute_df: pd.DataFrame) -> int:
        """
        Run the model over historical data to populate the trade log.
        This trains on each walk-forward fold and generates paper trades.
        Returns the number of trades generated.
        """
        try:
            import lightgbm as lgb
        except ImportError:
            log.error("LightGBM not installed")
            return 0

        hourly = self._build_hourly_df(minute_df)
        if hourly.empty:
            return 0

        # Add forward labels
        # close is stored in tick units (1 unit = 1 tick = 0.25 ES pts).
        # Raw difference is already in ticks — no division needed.
        hourly = hourly.sort_values("ts").reset_index(drop=True)
        hourly["fwd_ticks"] = hourly["close"].shift(-HORIZON_BARS) - hourly["close"]

        # Null overnight gaps
        for i in range(len(hourly) - HORIZON_BARS):
            ts_now = hourly["ts"].iloc[i]
            ts_fwd = hourly["ts"].iloc[i + HORIZON_BARS]
            diff_s = (ts_fwd - ts_now).total_seconds()
            if diff_s > 8 * 3600:
                hourly.loc[hourly.index[i], "fwd_ticks"] = np.nan

        # Filter intraday-clean
        hourly_clean = hourly[~hourly["hour"].isin([19, 20])].copy()

        dates = sorted(hourly_clean["date"].unique())
        feature_cols = get_feature_cols(hourly_clean)

        trades_generated = 0

        for i in range(TRAIN_DAYS + PURGE_DAYS, len(dates)):
            oot_date = dates[i]
            train_end_idx = i - PURGE_DAYS
            train_start_idx = max(0, train_end_idx - TRAIN_DAYS)
            train_dates = dates[train_start_idx:train_end_idx]

            train = hourly_clean[hourly_clean["date"].isin(train_dates)].dropna(subset=["fwd_ticks"])
            oot = hourly_clean[hourly_clean["date"] == oot_date]

            # Only trade signal hours
            oot_tradeable = oot[oot["hour"].isin(SIGNAL_HOURS_UTC)]
            if len(train) < 100 or len(oot_tradeable) == 0:
                continue

            X_train = train[feature_cols].fillna(0).values.astype(np.float32)
            y_train = train["fwd_ticks"].values.astype(np.float32)

            # Split for validation
            split = int(len(X_train) * 0.8)
            try:
                model = lgb.LGBMRegressor(**LGBM_PARAMS, early_stopping_rounds=50, seed=42)
                model.fit(
                    X_train[:split], y_train[:split],
                    eval_set=[(X_train[split:], y_train[split:])],
                    callbacks=[lgb.log_evaluation(0)],
                )
            except Exception as e:
                log.warning(f"Training failed for {oot_date}: {e}")
                continue

            # Predict on ALL OOT bars (not just signal hours) for mid-trade checks
            oot_all = hourly_clean[hourly_clean["date"] == oot_date]
            X_oot_all = oot_all[feature_cols].fillna(0).values.astype(np.float32)
            preds_all = model.predict(X_oot_all)
            oot_all = oot_all.copy()
            oot_all["pred"] = preds_all

            X_oot = oot_tradeable[feature_cols].fillna(0).values.astype(np.float32)
            preds = model.predict(X_oot)

            for j, (idx, row) in enumerate(oot_tradeable.iterrows()):
                pred = preds[j]
                # Apply confidence filter (same as live)
                if abs(pred) < MIN_CONFIDENCE_TICKS:
                    continue
                signal = 1 if pred > 0 else -1
                entry_price = row["close"]
                entry_hour = int(row["hour"])

                # Mid-trade thesis validation: check prediction at entry+1h
                # If prediction flips sign with confidence, exit at 1h bar instead of 2h
                exit_hour = entry_hour + HORIZON_BARS
                mid_hour = entry_hour + 1
                exit_reason = "hold_expired"
                actual_hold = HORIZON_BARS

                # Check mid-trade bar prediction
                mid_bar = oot_all[oot_all["hour"] == mid_hour]
                if not mid_bar.empty:
                    mid_pred = float(mid_bar.iloc[0]["pred"])
                    thesis_invalidated = (
                        (signal == 1 and mid_pred < -MIN_CONFIDENCE_TICKS) or
                        (signal == -1 and mid_pred > MIN_CONFIDENCE_TICKS)
                    )
                    if thesis_invalidated:
                        # Exit at the mid bar close instead of the 2h bar
                        exit_hour = mid_hour
                        exit_reason = "thesis_invalidated"
                        actual_hold = 1
                        self.state.setdefault("thesis_invalidations", 0)
                        self.state["thesis_invalidations"] += 1

                # Find exit bar
                exit_candidates = hourly[
                    (hourly["date"] == oot_date) &
                    (hourly["hour"] == exit_hour)
                ]
                if exit_candidates.empty:
                    continue

                exit_price = exit_candidates.iloc[0]["close"]

                # close prices are in tick units — raw diff IS already in ticks
                if signal == 1:
                    gross_ticks = exit_price - entry_price
                else:
                    gross_ticks = entry_price - exit_price

                net_ticks = gross_ticks - COST_RT_TICKS
                net_dollars = net_ticks * ES_TICK_VALUE

                self.state["total_trades"] += 1
                self.state["total_pnl_ticks"] += net_ticks
                self.state["total_pnl_dollars"] += net_dollars
                self.state["capital"] += net_dollars

                if net_ticks > 0:
                    self.state["wins"] += 1
                else:
                    self.state["losses"] += 1

                if oot_date not in self.state["daily_pnl"]:
                    self.state["daily_pnl"][oot_date] = 0.0
                self.state["daily_pnl"][oot_date] += net_dollars

                dir_str = "LONG" if signal == 1 else "SHORT"
                self._append_trade({
                    "trade_id": self.state["total_trades"],
                    "entry_time": str(row["ts"]),
                    "exit_time": str(exit_candidates.iloc[0]["ts"]),
                    "direction": dir_str,
                    "entry_price": entry_price,
                    "exit_price": exit_price,
                    "gross_pnl_ticks": gross_ticks,
                    "cost_ticks": COST_RT_TICKS,
                    "net_pnl_ticks": net_ticks,
                    "net_pnl_dollars": net_dollars,
                    "prediction": pred,
                    "hold_hours": actual_hold,
                    "capital_after": self.state["capital"],
                })

                trades_generated += 1

            del model
            gc.collect()

        self._save_state()
        self._log_equity()

        wr = self.state["wins"] / max(self.state["total_trades"], 1)
        log.info(
            f"Backfill complete: {trades_generated} trades, "
            f"net P&L: {self.state['total_pnl_ticks']:+.2f} ticks "
            f"(${self.state['total_pnl_dollars']:+,.2f}), "
            f"WR: {wr:.1%}, Capital: ${self.state['capital']:,.2f}"
        )
        return trades_generated

    # ── Summary ──

    def summary(self) -> Dict:
        """Return summary metrics."""
        wr = self.state["wins"] / max(self.state["total_trades"], 1)
        daily_pnl = self.state.get("daily_pnl", {})
        daily_vals = list(daily_pnl.values()) if daily_pnl else [0]

        avg_daily = np.mean(daily_vals) if daily_vals else 0
        std_daily = np.std(daily_vals) if len(daily_vals) > 1 else 1
        sharpe = (avg_daily / max(std_daily, 1)) * np.sqrt(252) if std_daily > 0 else 0

        # Sortino (downside deviation only)
        neg_daily = [d for d in daily_vals if d < 0]
        downside_std = np.std(neg_daily) if len(neg_daily) > 1 else std_daily
        sortino = (avg_daily / max(downside_std, 1)) * np.sqrt(252) if downside_std > 0 else 0

        # Profit factor
        gross_wins = sum(d for d in daily_vals if d > 0)
        gross_losses = abs(sum(d for d in daily_vals if d < 0))
        pf = gross_wins / max(gross_losses, 1)

        return {
            "total_trades": self.state["total_trades"],
            "wins": self.state["wins"],
            "losses": self.state["losses"],
            "win_rate": wr,
            "total_pnl_ticks": self.state["total_pnl_ticks"],
            "total_pnl_dollars": self.state["total_pnl_dollars"],
            "capital": self.state["capital"],
            "avg_daily_pnl": avg_daily,
            "sharpe": sharpe,
            "sortino": sortino,
            "profit_factor": pf,
            "trading_days": len(daily_vals),
            "thesis_invalidations": self.state.get("thesis_invalidations", 0),
        }


# =============================================================================
#  MAIN LOOP
# =============================================================================

def main():
    log.info("=" * 60)
    log.info("2-Hour LightGBM Paper Trading Engine starting")
    log.info(f"Capital: ${STARTING_CAPITAL:,} | Cost: {COST_RT_TICKS} ticks RT")
    log.info(f"Signal hours (UTC): {SIGNAL_HOURS_UTC}")
    log.info(f"Train window: {TRAIN_DAYS}d, Purge: {PURGE_DAYS}d")
    log.info(f"Confidence threshold: {MIN_CONFIDENCE_TICKS} ticks (HC #664)")
    log.info(f"Max daily loss: {MAX_DAILY_LOSS_TICKS} ticks (${MAX_DAILY_LOSS_TICKS * ES_TICK_VALUE:,.0f})")
    log.info(f"Stale data guard: {STALE_DATA_MAX_DAYS} trading days")
    log.info("=" * 60)

    engine = LH2hPaperEngine()

    # Load minute bars
    log.info("Loading minute bar data...")
    minute_df = engine._load_minute_bars()
    if minute_df.empty:
        log.error("No minute bar data available. Exiting.")
        sys.exit(1)

    # Try to load existing model, otherwise retrain
    if not engine.load_model():
        log.info("No saved model found. Training from scratch...")
        if not engine.retrain(minute_df):
            log.error("Initial training failed. Exiting.")
            sys.exit(1)
    else:
        # Rebuild hourly features for inference even if model was loaded
        engine.hourly_df = engine._build_hourly_df(minute_df)

    # If no trades yet, run backfill on historical data
    if engine.state["total_trades"] == 0:
        log.info("No trades in history. Running backfill on historical data...")
        n_trades = engine.backfill(minute_df)
        log.info(f"Backfill produced {n_trades} trades")

        # Print summary
        s = engine.summary()
        log.info(
            f"Backfill summary: {s['total_trades']} trades, "
            f"WR {s['win_rate']:.1%}, Sharpe {s['sharpe']:.2f}, "
            f"Sortino {s['sortino']:.2f}, PF {s['profit_factor']:.2f}, "
            f"Net P&L ${s['total_pnl_dollars']:+,.2f}"
        )

    # Main loop: check for new signals every LOOP_SLEEP_SECONDS
    last_retrain_date = engine.state.get("last_retrain_date", "")
    last_data_reload = time.time()
    DATA_RELOAD_INTERVAL = 3600  # reload minute bars every hour

    log.info("Entering main trading loop...")

    while True:
        try:
            now_utc = datetime.now(timezone.utc)
            current_date = now_utc.strftime("%Y%m%d")

            # Reload minute bars periodically (pick up new data)
            if time.time() - last_data_reload > DATA_RELOAD_INTERVAL:
                log.info("Reloading minute bar data...")
                minute_df = engine._load_minute_bars()
                last_data_reload = time.time()

            # Retrain daily (once per day at first signal hour)
            if current_date != last_retrain_date and now_utc.hour >= SIGNAL_HOURS_UTC[0]:
                log.info(f"Daily retrain triggered for {current_date}")
                if engine.retrain(minute_df):
                    last_retrain_date = current_date
                else:
                    log.warning("Daily retrain failed, using existing model")

            # Run hourly tick (signal generation + position management)
            engine.hourly_tick(minute_df)

            # Periodic summary (every 4 hours)
            if now_utc.hour % 4 == 0 and now_utc.minute < 1:
                s = engine.summary()
                log.info(
                    f"Status: {s['total_trades']} trades, "
                    f"WR {s['win_rate']:.1%}, "
                    f"P&L ${s['total_pnl_dollars']:+,.2f}, "
                    f"Capital ${s['capital']:,.2f}"
                )

        except KeyboardInterrupt:
            log.info("Shutdown requested. Saving state...")
            engine._save_state()
            break
        except Exception as e:
            log.error(f"Error in main loop: {e}")
            log.error(traceback.format_exc())
            # Save state on error to avoid data loss
            try:
                engine._save_state()
            except Exception:
                pass

        time.sleep(LOOP_SLEEP_SECONDS)

    # Final summary
    s = engine.summary()
    log.info("=" * 60)
    log.info("FINAL SUMMARY")
    log.info(f"  Trades: {s['total_trades']} (W:{s['wins']} / L:{s['losses']})")
    log.info(f"  Win Rate: {s['win_rate']:.1%}")
    log.info(f"  Sharpe: {s['sharpe']:.2f}")
    log.info(f"  Sortino: {s['sortino']:.2f}")
    log.info(f"  Profit Factor: {s['profit_factor']:.2f}")
    log.info(f"  Net P&L: ${s['total_pnl_dollars']:+,.2f}")
    log.info(f"  Capital: ${s['capital']:,.2f}")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
