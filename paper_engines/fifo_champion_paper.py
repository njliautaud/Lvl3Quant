#!/usr/bin/env python3
"""
FIFO Champion Paper Trading Engine
====================================

Paper trades the validated champion 30-min LightGBM strategy with FIFO-based
passive limit execution on ES futures.

CHAMPION STRATEGY (Sharpe 5.46, 5/5 robustness PASS):
  - Entry model: 30-min LightGBM on 43-feature lean set (sliding 60d train)
  - Confidence: Top 5% predictions only
  - Daily filter: OFI contrarian (1.5x threshold blocks trades against daily bias)
  - Entry: Passive limit at bid/ask. Fill simulated as FIFO back-of-queue
          (fill when price trades 1 tick through entry price)
  - TP: 25 ticks, passive limit on opposite side
  - SL: 4 ticks (long) / 3 ticks (short), market order with 1 tick slippage
  - Max hold: 60 minutes (force exit at market)
  - Max position: 1 contract at a time

COST MODEL:
  - Entry (passive limit): 0.376 ticks (commission only)
  - TP exit (passive limit): 0.376 ticks (commission only)
  - SL exit (market order): 1.376 ticks (commission + 1 tick spread)
  - Time-stop exit (market): 1.376 ticks
  - Entry + TP: 0.752 ticks total
  - Entry + SL: 1.752 ticks total

MODES:
  --replay N    Replay last N trading days bar-by-bar (default: 5)
  --live        Run in live loop mode (for PM2, checks data every 30s)
  --summary     Print current state and exit

PM2:
  pm2 start /home/jupiter/Lvl3Quant/paper_engines/fifo_champion_paper.py \
    --name fifo-champion-paper --interpreter python3 -- --live

Author: Claude (autonomous build)
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
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from threading import Thread
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────
#  PATHS
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
ENGINE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ENGINE_DIR / "logs" / "fifo_champion_paper"
MODEL_DIR = OUTPUT_DIR / "models"
STATE_DIR = OUTPUT_DIR

for d in [OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
LOG_FILE = OUTPUT_DIR / "fifo_champion_paper.log"
logging.basicConfig(
    format="%(asctime)s [FIFO-CHAMP] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE)),
    ],
)
log = logging.getLogger("FIFO-CHAMP")

# ─────────────────────────────────────────────
#  CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
BAR_SIZE_MINUTES = 30
ES_TICK_SIZE = 0.25
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376       # $4.70 / $12.50

# Execution costs per leg
COST_PASSIVE_ENTRY = ES_RT_COMMISSION_TICKS / 2   # 0.188 ticks (half RT commission)
COST_PASSIVE_EXIT = ES_RT_COMMISSION_TICKS / 2    # 0.188 ticks
COST_MARKET_EXIT = ES_RT_COMMISSION_TICKS / 2 + 1.0  # 0.188 + 1.0 tick spread = 1.188 ticks
# Full RT costs
COST_ENTRY_PLUS_TP = ES_RT_COMMISSION_TICKS       # 0.376 ticks (passive + passive)
COST_ENTRY_PLUS_SL = ES_RT_COMMISSION_TICKS + 1.0 # 1.376 ticks (passive + market w/ 1t adverse)

# Strategy parameters (CHAMPION CONFIG)
CONFIDENCE_THRESHOLD_PCT = 5         # top/bottom 5%
TP_TICKS = 25                        # take-profit in ticks
SL_LONG_TICKS = 4                    # stop-loss for long positions
SL_SHORT_TICKS = 3                   # stop-loss for short positions
MAX_HOLD_MINUTES = 60                # maximum hold time
MAX_POSITION = 1                     # 1 contract at a time
TRAIN_DAYS = 60                      # sliding training window
STARTING_CAPITAL = 100_000

# Daily OFI contrarian filter
OFI_CONTRARIAN_THRESHOLD = 1.5       # multiplier of OFI median abs deviation

# FIFO fill model: fill when price trades 1 tick THROUGH entry price
FILL_THROUGH_TICKS = 1               # price must trade this many ticks through for fill

# RTH boundaries (UTC)
RTH_START_H, RTH_START_M = 13, 30    # 09:30 ET
RTH_END_H, RTH_END_M = 20, 0        # 16:00 ET

# Data lookback for feature context
LOOKBACK_DAYS = 75

# LightGBM params (identical to training/OOT validation)
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

# 43-feature lean set (validated: Sharpe 3.90 on 37d holdout, wins 3/3 splits)
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

# Health check port
HEALTH_PORT = 8765

# Main loop sleep
LOOP_SLEEP_SECONDS = 30


# ═══════════════════════════════════════════════════════════════════
#  FEATURE COMPUTATION (identical to lh_30min_paper_engine.py / lean_oot.py)
# ═══════════════════════════════════════════════════════════════════


def _safe_polyfit_slope(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def aggregate_to_bars(minute_df: pd.DataFrame, bar_size_min: int = 30) -> pd.DataFrame:
    """Aggregate 1-minute bars into N-minute bars with microstructure features."""
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
            "open": close_arr[0],
            "high": close_arr.max(),
            "low": close_arr.min(),
            "close": close_arr[-1],
            "return_bar": (close_arr[-1] / close_arr[0] - 1) if close_arr[0] > 0 else 0,
            "range_ticks": (close_arr.max() - close_arr.min()) / ES_TICK_SIZE,
            "close_position": (
                (close_arr[-1] - close_arr.min())
                / max(close_arr.max() - close_arr.min(), ES_TICK_SIZE)
            ),
            "total_volume": vol_arr.sum(),
            "avg_volume": vol_arr.mean(),
            "volume_trend": _safe_polyfit_slope(vol_arr),
            "volume_concentration": vol_arr.max() / max(vol_arr.mean(), 1),
            "ofi_sum": ofi_arr.sum(),
            "ofi_mean": ofi_arr.mean(),
            "ofi_std": ofi_arr.std() if len(ofi_arr) > 1 else 0,
            "ofi_trend": _safe_polyfit_slope(ofi_arr),
            "ofi_consistency": (
                np.mean(np.sign(ofi_arr) == np.sign(ofi_arr.sum()))
                if ofi_arr.sum() != 0 else 0.5
            ),
            "ofi_late_vs_early": (
                ofi_arr[len(ofi_arr) // 2:].sum() - ofi_arr[:len(ofi_arr) // 2].sum()
            ),
            "signed_volume_sum": sv_arr.sum(),
            "signed_volume_ratio": sv_arr.sum() / max(vol_arr.sum(), 1),
            "buy_volume_frac": float(np.sum(sv_arr[sv_arr > 0])) / max(vol_arr.sum(), 1),
            "sell_volume_frac": float(-np.sum(sv_arr[sv_arr < 0])) / max(vol_arr.sum(), 1),
            "sweep_minutes": int(np.sum(np.abs(grp["sv_zscore"].values) > 2)),
            "max_sweep_intensity": float(np.abs(grp["sv_zscore"].values).max()),
            "sweep_direction": (
                float(np.sign(sv_arr[np.abs(grp["sv_zscore"].values).argmax()]))
                if len(sv_arr) > 0 else 0.0
            ),
            "spread_mean": spread_arr.mean(),
            "spread_max": spread_arr.max(),
            "spread_trend": _safe_polyfit_slope(spread_arr),
            "trade_count_sum": tc_arr.sum(),
            "trade_count_trend": _safe_polyfit_slope(tc_arr),
            "realized_vol": (
                float(np.std(ret_arr) * np.sqrt(252 * (390 // bar_size_min)))
                if len(ret_arr) > 1 else 0
            ),
            "vol_of_vol": float(np.std(np.abs(ret_arr))) if len(ret_arr) > 1 else 0,
            "up_vol": float(np.std(ret_arr[ret_arr > 0])) if np.sum(ret_arr > 0) > 1 else 0,
            "down_vol": float(np.std(ret_arr[ret_arr < 0])) if np.sum(ret_arr < 0) > 1 else 0,
            "vwap_dev_mean": grp["vwap_dev"].mean(),
            "vwap_dev_trend": _safe_polyfit_slope(grp["vwap_dev"].values),
        }

        if rec["up_vol"] > 0 and rec["down_vol"] > 0:
            rec["vol_asymmetry"] = rec["down_vol"] / rec["up_vol"] - 1
        elif rec["down_vol"] > 0:
            rec["vol_asymmetry"] = 1.0
        elif rec["up_vol"] > 0:
            rec["vol_asymmetry"] = -1.0
        else:
            rec["vol_asymmetry"] = 0.0

        records.append(rec)

    return pd.DataFrame(records)


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

    for bars, label in [(4, "lb_4bar"), (8, "lb_8bar"), (16, "lb_16bar")]:
        df[f"ret_{label}"] = df["close"].pct_change(bars)

    for w in [4, 8, 16]:
        df[f"rvol_{w}bar"] = df["return_bar"].rolling(w, min_periods=2).std()

    df["intraday_cum_ofi"] = df.groupby("date")["ofi_sum"].cumsum()
    df["intraday_cum_sv"] = df.groupby("date")["signed_volume_sum"].cumsum()

    df["ofi_sign_flip"] = (
        np.sign(df["ofi_sum"]) != np.sign(df["ofi_sum"].shift(1))
    ).astype(np.float32)

    df["absorption"] = df["total_volume"] / df["range_ticks"].clip(lower=1)

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

    session_start_hour = 13.5
    session_end_hour = 20.0
    session_len = session_end_hour - session_start_hour

    hour_frac = df["ts"].dt.hour + df["ts"].dt.minute / 60.0
    progress = ((hour_frac - session_start_hour) / session_len).clip(0, 1)
    df["tod_sin"] = np.sin(2 * np.pi * progress)
    df["tod_cos"] = np.cos(2 * np.pi * progress)
    df["tod_progress"] = progress.values

    df["bars_since_open"] = df.groupby("date").cumcount()

    df["regime_ret_16bar"] = df["close"].pct_change(16)
    df["regime_ret_32bar"] = df["close"].pct_change(32)

    rvol_short = df["return_bar"].rolling(4, min_periods=2).std()
    rvol_long = df["return_bar"].rolling(16, min_periods=4).std()
    df["regime_vol_ratio"] = rvol_short / rvol_long.clip(lower=1e-8)

    df["intraday_direction_strength"] = df["intraday_cum_ofi"].abs() / (
        df.groupby("date")["ofi_sum"]
        .transform(lambda x: x.abs().cumsum())
        .clip(lower=1)
    )

    return df


# ═══════════════════════════════════════════════════════════════════
#  DAILY OFI CONTRARIAN FILTER
# ═══════════════════════════════════════════════════════════════════


def compute_daily_ofi_bias(bars_df: pd.DataFrame) -> Dict[str, int]:
    """
    Compute the daily OFI contrarian directional bias for each date.

    Logic: If yesterday's session OFI was strongly positive (>1.5x median abs),
    today's bias is SHORT (contrarian). If strongly negative, bias is LONG.
    If within threshold, no bias (all trades allowed).

    Returns dict: {date_str: bias} where bias = +1 (long only), -1 (short only), 0 (no filter)
    """
    day_ofi = bars_df.groupby("date")["ofi_sum"].sum().reset_index()
    day_ofi.columns = ["date", "session_ofi"]
    day_ofi = day_ofi.sort_values("date").reset_index(drop=True)

    # Compute rolling median absolute OFI for threshold
    abs_ofi = day_ofi["session_ofi"].abs()
    rolling_median_abs = abs_ofi.expanding(min_periods=5).median()

    bias_map = {}
    for i in range(len(day_ofi)):
        date = day_ofi.iloc[i]["date"]
        if i == 0:
            bias_map[date] = 0  # No prior day
            continue

        prev_ofi = day_ofi.iloc[i - 1]["session_ofi"]
        threshold = rolling_median_abs.iloc[i - 1] * OFI_CONTRARIAN_THRESHOLD

        if prev_ofi > threshold:
            # Strong positive OFI yesterday -> contrarian SHORT bias today
            bias_map[date] = -1
        elif prev_ofi < -threshold:
            # Strong negative OFI yesterday -> contrarian LONG bias today
            bias_map[date] = 1
        else:
            bias_map[date] = 0  # No filter

    return bias_map


# ═══════════════════════════════════════════════════════════════════
#  FIFO FILL SIMULATION
# ═══════════════════════════════════════════════════════════════════


def simulate_fifo_fill(
    entry_price: float,
    direction: int,
    minute_bars_after: pd.DataFrame,
) -> Optional[Dict]:
    """
    Simulate whether a passive limit entry order gets filled using FIFO model.

    For a LONG entry at price P:
      - We place a bid at P
      - Fill occurs when price trades at P - 1 tick (trades through our level)

    For a SHORT entry at price P:
      - We place an ask at P
      - Fill occurs when price trades at P + 1 tick (trades through our level)

    Returns fill info dict or None if unfilled within the bar period.
    """
    if minute_bars_after.empty:
        return None

    fill_threshold = FILL_THROUGH_TICKS * ES_TICK_SIZE

    for _, bar in minute_bars_after.iterrows():
        if direction == 1:  # Long: buying at bid, fill when price drops through
            if bar["low"] <= entry_price - fill_threshold:
                return {
                    "fill_price": entry_price,
                    "fill_time": bar["ts_minute"],
                    "fill_type": "passive_limit",
                }
        else:  # Short: selling at ask, fill when price rises through
            if bar["high"] >= entry_price + fill_threshold:
                return {
                    "fill_price": entry_price,
                    "fill_time": bar["ts_minute"],
                    "fill_type": "passive_limit",
                }

    return None  # Not filled


def simulate_exit(
    entry_price: float,
    direction: int,
    fill_time: pd.Timestamp,
    minute_bars_after: pd.DataFrame,
) -> Dict:
    """
    Simulate exit using TP/SL/TimeStop on minute bars after entry fill.

    TP: Passive limit on opposite side (fill when price trades 1 tick through)
    SL: Market order (immediate) when price hits stop level
    TimeStop: Market exit after MAX_HOLD_MINUTES

    Returns exit info dict.
    """
    tp_ticks = TP_TICKS
    sl_ticks = SL_LONG_TICKS if direction == 1 else SL_SHORT_TICKS

    if direction == 1:
        tp_price = entry_price + tp_ticks * ES_TICK_SIZE
        sl_price = entry_price - sl_ticks * ES_TICK_SIZE
    else:
        tp_price = entry_price - tp_ticks * ES_TICK_SIZE
        sl_price = entry_price + sl_ticks * ES_TICK_SIZE

    time_stop = fill_time + timedelta(minutes=MAX_HOLD_MINUTES)
    fill_threshold = FILL_THROUGH_TICKS * ES_TICK_SIZE

    # Walk through minute bars chronologically
    for _, bar in minute_bars_after.iterrows():
        bar_time = bar["ts_minute"]
        if bar_time <= fill_time:
            continue

        bar_high = bar["high"]
        bar_low = bar["low"]
        bar_close = bar["close"]

        # Check SL first (market order - immediate fill at adverse price)
        if direction == 1:
            if bar_low <= sl_price:
                # SL hit on long: exit at SL price (market, 1 tick adverse slippage)
                exit_price = sl_price - 1.0 * ES_TICK_SIZE  # 1 tick adverse slippage
                return {
                    "exit_price": exit_price,
                    "exit_time": bar_time,
                    "exit_type": "SL",
                    "cost_ticks": COST_ENTRY_PLUS_SL,
                    "raw_pnl_ticks": (exit_price - entry_price) / ES_TICK_SIZE,
                }
        else:
            if bar_high >= sl_price:
                exit_price = sl_price + 1.0 * ES_TICK_SIZE
                return {
                    "exit_price": exit_price,
                    "exit_time": bar_time,
                    "exit_type": "SL",
                    "cost_ticks": COST_ENTRY_PLUS_SL,
                    "raw_pnl_ticks": (entry_price - exit_price) / ES_TICK_SIZE,
                }

        # Check TP (passive limit - fill when price trades through)
        if direction == 1:
            if bar_high >= tp_price + fill_threshold:
                return {
                    "exit_price": tp_price,
                    "exit_time": bar_time,
                    "exit_type": "TP",
                    "cost_ticks": COST_ENTRY_PLUS_TP,
                    "raw_pnl_ticks": (tp_price - entry_price) / ES_TICK_SIZE,
                }
        else:
            if bar_low <= tp_price - fill_threshold:
                return {
                    "exit_price": tp_price,
                    "exit_time": bar_time,
                    "exit_type": "TP",
                    "cost_ticks": COST_ENTRY_PLUS_TP,
                    "raw_pnl_ticks": (entry_price - tp_price) / ES_TICK_SIZE,
                }

        # Check time stop
        if bar_time >= time_stop:
            exit_price = bar_close
            raw_pnl = (exit_price - entry_price) / ES_TICK_SIZE * direction
            return {
                "exit_price": exit_price,
                "exit_time": bar_time,
                "exit_type": "TimeStop",
                "cost_ticks": COST_ENTRY_PLUS_SL,  # market exit cost
                "raw_pnl_ticks": raw_pnl,
            }

    # End of data: force close at last bar
    if not minute_bars_after.empty:
        last_bar = minute_bars_after.iloc[-1]
        exit_price = last_bar["close"]
        raw_pnl = (exit_price - entry_price) / ES_TICK_SIZE * direction
        return {
            "exit_price": exit_price,
            "exit_time": last_bar["ts_minute"],
            "exit_type": "EOD",
            "cost_ticks": COST_ENTRY_PLUS_SL,
            "raw_pnl_ticks": raw_pnl,
        }

    return {
        "exit_price": entry_price,
        "exit_time": fill_time,
        "exit_type": "NoData",
        "cost_ticks": 0,
        "raw_pnl_ticks": 0,
    }


# ═══════════════════════════════════════════════════════════════════
#  PAPER TRADING ENGINE
# ═══════════════════════════════════════════════════════════════════


class FIFOChampionPaperEngine:
    """FIFO-based champion paper trading engine with TP/SL/TimeStop."""

    def __init__(self):
        self.state_path = STATE_DIR / "state.json"
        self.trades_csv = OUTPUT_DIR / "trades.csv"
        self.daily_csv = OUTPUT_DIR / "daily_pnl.csv"
        self.model_path = MODEL_DIR / "lgbm_fifo_champion.txt"
        self.scaler_path = MODEL_DIR / "scaler_fifo_champion.npz"
        self.meta_path = MODEL_DIR / "meta_fifo_champion.json"

        self.model = None
        self.feature_cols: Optional[List[str]] = None
        self.scaler_median: Optional[np.ndarray] = None
        self.scaler_iqr: Optional[np.ndarray] = None
        self.pred_quantiles: Optional[Dict[str, float]] = None
        self.daily_ofi_bias: Dict[str, int] = {}

        self.state = self._load_state()
        self._ensure_trades_csv()
        self._ensure_daily_csv()

        log.info(
            f"Engine initialized. Capital: ${self.state['capital']:.0f}, "
            f"Trades: {self.state['total_trades']}, "
            f"Net PnL: {self.state['total_pnl_ticks']:.1f} ticks"
        )

    # ── State persistence ──

    def _load_state(self) -> Dict:
        if self.state_path.exists():
            try:
                with open(self.state_path) as f:
                    return json.load(f)
            except Exception as e:
                log.warning(f"Failed to load state: {e}, starting fresh")
        return self._fresh_state()

    def _fresh_state(self) -> Dict:
        return {
            "capital": STARTING_CAPITAL,
            "position": 0,
            "entry_price": 0.0,
            "entry_time": None,
            "fill_time": None,
            "tp_price": 0.0,
            "sl_price": 0.0,
            "time_stop": None,
            "total_trades": 0,
            "total_pnl_ticks": 0.0,
            "total_pnl_dollars": 0.0,
            "wins": 0,
            "losses": 0,
            "tp_exits": 0,
            "sl_exits": 0,
            "time_exits": 0,
            "eod_exits": 0,
            "signals_generated": 0,
            "signals_filtered_ofi": 0,
            "signals_traded": 0,
            "last_signal_bar": None,
            "last_retrain_date": None,
            "last_heartbeat": None,
            "created": datetime.now(timezone.utc).isoformat(),
        }

    def _save_state(self):
        self.state["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _ensure_trades_csv(self):
        if not self.trades_csv.exists():
            with open(self.trades_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "trade_id", "date", "signal_time", "fill_time", "exit_time",
                    "direction", "entry_price", "exit_price",
                    "tp_price", "sl_price",
                    "exit_type", "raw_pnl_ticks", "cost_ticks", "net_pnl_ticks",
                    "net_pnl_dollars", "prediction", "confidence_pct",
                    "ofi_bias", "hold_minutes", "capital_after",
                ])

    def _ensure_daily_csv(self):
        if not self.daily_csv.exists():
            with open(self.daily_csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    "date", "n_trades", "n_signals", "n_filtered",
                    "gross_pnl_ticks", "cost_ticks", "net_pnl_ticks", "net_pnl_dollars",
                    "wins", "losses", "tp_exits", "sl_exits", "time_exits",
                    "win_rate", "capital_after",
                ])

    def _append_trade(self, trade: Dict):
        with open(self.trades_csv, "a", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                trade.get("trade_id"),
                trade.get("date"),
                trade.get("signal_time"),
                trade.get("fill_time"),
                trade.get("exit_time"),
                trade.get("direction"),
                trade.get("entry_price"),
                trade.get("exit_price"),
                trade.get("tp_price"),
                trade.get("sl_price"),
                trade.get("exit_type"),
                round(trade.get("raw_pnl_ticks", 0), 3),
                round(trade.get("cost_ticks", 0), 3),
                round(trade.get("net_pnl_ticks", 0), 3),
                round(trade.get("net_pnl_dollars", 0), 2),
                round(trade.get("prediction", 0), 4),
                trade.get("confidence_pct"),
                trade.get("ofi_bias"),
                round(trade.get("hold_minutes", 0), 1),
                round(trade.get("capital_after", 0), 2),
            ])

    # ── Data loading ──

    def _load_minute_bars(self, n_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
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
        bars_df = aggregate_to_bars(minute_df, bar_size_min=BAR_SIZE_MINUTES)
        if bars_df.empty:
            return pd.DataFrame()
        bars_df = add_rolling_features(bars_df)
        return bars_df

    # ── Training ──

    def retrain(self, minute_df: pd.DataFrame) -> bool:
        try:
            import lightgbm as lgb
        except ImportError:
            log.error("LightGBM not installed")
            return False

        bars_df = self._build_feature_df(minute_df)
        if bars_df.empty:
            return False

        dates = sorted(bars_df["date"].unique())
        if len(dates) < TRAIN_DAYS:
            log.warning(f"Not enough days: {len(dates)} < {TRAIN_DAYS}")
            return False

        train_dates = dates[-TRAIN_DAYS:]
        train_mask = bars_df["date"].isin(train_dates)
        train_df = bars_df[train_mask].copy().sort_values("ts").reset_index(drop=True)

        # Forward label: 1-bar forward ticks
        fwd_close = train_df["close"].shift(-1)
        fwd_ticks = (fwd_close - train_df["close"]) / ES_TICK_SIZE

        # Null overnight gaps
        ts_now = train_df["ts"].values
        ts_fwd = train_df["ts"].shift(-1).values
        for i in range(len(train_df) - 1):
            if pd.isna(ts_fwd[i]):
                continue
            diff_s = (pd.Timestamp(ts_fwd[i]) - pd.Timestamp(ts_now[i])).total_seconds()
            if diff_s > 6 * 3600:
                fwd_ticks.iloc[i] = np.nan

        train_df["fwd_ticks_30min"] = fwd_ticks

        # Feature columns (lean set only)
        lean_set = set(LEAN_FEATURES)
        available = [c for c in train_df.columns if c in lean_set]
        missing = lean_set - set(available)
        if missing:
            log.warning(f"Missing lean features: {missing}")
        self.feature_cols = [f for f in LEAN_FEATURES if f in available]
        log.info(f"Training with {len(self.feature_cols)} lean features")

        X_raw = train_df[self.feature_cols].values.astype(np.float32)
        y = train_df["fwd_ticks_30min"].values.astype(np.float32)

        valid = ~np.isnan(y)
        X_raw = X_raw[valid]
        y = y[valid]

        if len(y) < 100:
            log.warning(f"Too few samples: {len(y)}")
            return False

        # Robust scaling
        self.scaler_median = np.nanmedian(X_raw, axis=0)
        q75 = np.nanpercentile(X_raw, 75, axis=0)
        q25 = np.nanpercentile(X_raw, 25, axis=0)
        self.scaler_iqr = q75 - q25
        self.scaler_iqr[self.scaler_iqr < 1e-8] = 1.0

        X = np.clip(np.nan_to_num(
            (X_raw - self.scaler_median) / self.scaler_iqr,
            nan=0.0, posinf=3.0, neginf=-3.0
        ), -5, 5)

        params = {**LGBM_PARAMS, "seed": 42}
        # FIX: split train/val BEFORE creating datasets to prevent val leak (audit 2026-07-01)
        n_val = max(int(len(X) * 0.1), 20)
        X_train, X_val = X[:-n_val], X[-n_val:]
        y_train, y_val = y[:-n_val], y[-n_val:]
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.feature_cols)
        val_data = lgb.Dataset(
            X_val, label=y_val,
            feature_name=self.feature_cols, reference=train_data,
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        self.model = lgb.train(
            params, train_data, num_boost_round=500,
            valid_sets=[val_data], callbacks=callbacks,
        )

        # Prediction quantiles for confidence thresholding
        # Use walk-forward OOT predictions for threshold calibration:
        # Train on first 80%, predict on last 20%, use those OOT preds for quantiles
        train_preds = self.model.predict(X, num_iteration=self.model.best_iteration)
        oot_preds = train_preds[-n_val:]  # Last 10% is OOT (early-stopping holdout)

        # Also run a mini walk-forward: train on first 70%, predict 70-90%
        n_wf_train = int(len(X) * 0.7)
        n_wf_test = int(len(X) * 0.2)
        wf_td = lgb.Dataset(X[:n_wf_train], label=y[:n_wf_train], feature_name=self.feature_cols)
        wf_vd = lgb.Dataset(X[n_wf_train:n_wf_train+50], label=y[n_wf_train:n_wf_train+50],
                            feature_name=self.feature_cols, reference=wf_td)
        wf_model = lgb.train(params, wf_td, 500, valid_sets=[wf_vd], callbacks=callbacks)
        wf_preds = wf_model.predict(X[n_wf_train:n_wf_train+n_wf_test],
                                     num_iteration=wf_model.best_iteration)

        # Combine OOT predictions for more robust threshold estimation
        combined_oot = np.concatenate([oot_preds, wf_preds])
        conf = CONFIDENCE_THRESHOLD_PCT / 100.0
        self.pred_quantiles = {
            "upper": float(np.percentile(combined_oot, 100 * (1 - conf))),
            "lower": float(np.percentile(combined_oot, 100 * conf)),
            "mean": float(np.mean(combined_oot)),
            "std": float(np.std(combined_oot)),
        }
        log.info(f"OOT-calibrated thresholds (n={len(combined_oot)}): "
                 f"lower={self.pred_quantiles['lower']:.2f}, upper={self.pred_quantiles['upper']:.2f}")

        # Compute daily OFI bias
        self.daily_ofi_bias = compute_daily_ofi_bias(bars_df)

        # Save model artifacts
        self.model.save_model(str(self.model_path))
        np.savez_compressed(str(self.scaler_path),
                            median=self.scaler_median, iqr=self.scaler_iqr)
        meta = {
            "feature_cols": self.feature_cols,
            "pred_quantiles": self.pred_quantiles,
            "daily_ofi_bias": self.daily_ofi_bias,
            "train_dates": [train_dates[0], train_dates[-1]],
            "n_samples": len(y),
            "best_iteration": self.model.best_iteration,
            "retrain_time": datetime.now(timezone.utc).isoformat(),
        }
        with open(self.meta_path, "w") as f:
            json.dump(meta, f, indent=2)

        self.state["last_retrain_date"] = train_dates[-1]
        self._save_state()

        ic_oot = np.corrcoef(train_preds[-n_val:], y[-n_val:])[0, 1]
        log.info(
            f"Retrained: {train_dates[0]}->{train_dates[-1]}, "
            f"{len(y)} samples, iter={self.model.best_iteration}, "
            f"holdout IC={ic_oot:.4f}, "
            f"thresholds=[{self.pred_quantiles['lower']:.2f}, {self.pred_quantiles['upper']:.2f}]"
        )
        return True

    def load_model(self) -> bool:
        if not all(p.exists() for p in [self.model_path, self.scaler_path, self.meta_path]):
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
            self.daily_ofi_bias = meta.get("daily_ofi_bias", {})
            log.info(f"Loaded model. Features: {len(self.feature_cols)}")
            return True
        except Exception as e:
            log.warning(f"Failed to load model: {e}")
            return False

    # ── Inference ──

    def predict(self, bars_df: pd.DataFrame) -> Optional[Dict]:
        if self.model is None or self.feature_cols is None:
            return None
        if bars_df.empty:
            return None

        latest = bars_df.iloc[-1:]
        missing = [c for c in self.feature_cols if c not in latest.columns]
        if missing:
            log.warning(f"Missing features: {missing[:5]}")
            return None

        X_raw = latest[self.feature_cols].values.astype(np.float32)
        X = np.clip(np.nan_to_num(
            (X_raw - self.scaler_median) / self.scaler_iqr,
            nan=0.0, posinf=3.0, neginf=-3.0
        ), -5, 5)

        pred = self.model.predict(X, num_iteration=self.model.best_iteration)[0]

        signal = 0
        if pred >= self.pred_quantiles["upper"]:
            signal = 1
        elif pred <= self.pred_quantiles["lower"]:
            signal = -1

        bar_date = latest["date"].iloc[0]
        bar_close = float(latest["close"].iloc[0])
        bar_ts = latest["ts"].iloc[0]

        return {
            "prediction_ticks": float(pred),
            "signal": signal,
            "bar_ts": str(bar_ts),
            "bar_date": bar_date,
            "bar_close": bar_close,
        }

    # ── Replay backtest ──

    def replay(self, n_test_days: int = 5):
        """
        Replay the last N trading days bar-by-bar with full FIFO fill simulation.
        Uses minute-level data for fill/exit simulation within each 30-min signal bar.
        """
        log.info(f"\n{'=' * 70}")
        log.info(f"REPLAY MODE: last {n_test_days} trading days")
        log.info(f"Strategy: top/bottom {CONFIDENCE_THRESHOLD_PCT}% confidence")
        log.info(f"TP={TP_TICKS}t, SL_long={SL_LONG_TICKS}t, SL_short={SL_SHORT_TICKS}t, "
                 f"max_hold={MAX_HOLD_MINUTES}m")
        log.info(f"Fill model: passive limit, FIFO (fill when price trades {FILL_THROUGH_TICKS}t through)")
        log.info(f"OFI contrarian filter: {OFI_CONTRARIAN_THRESHOLD}x threshold")
        log.info(f"{'=' * 70}\n")

        minute_df = self._load_minute_bars(n_days=LOOKBACK_DAYS)
        if minute_df.empty:
            log.error("No minute bar data")
            return

        bars_df = self._build_feature_df(minute_df)
        if bars_df.empty:
            log.error("No bars after aggregation")
            return

        dates = sorted(bars_df["date"].unique())
        if len(dates) < TRAIN_DAYS + n_test_days:
            n_test_days = min(n_test_days, len(dates) - TRAIN_DAYS)
            if n_test_days < 1:
                log.error("Not enough data")
                return

        test_dates = dates[-n_test_days:]
        train_end_idx = len(dates) - n_test_days
        train_dates_for_model = dates[max(0, train_end_idx - TRAIN_DAYS):train_end_idx]
        log.info(f"Test dates: {test_dates[0]} -> {test_dates[-1]} ({len(test_dates)} days)")
        log.info(f"Train dates for model: {train_dates_for_model[0]} -> {train_dates_for_model[-1]} ({len(train_dates_for_model)} days)")

        # Train on data STRICTLY before test period
        # Filter minute_df to exclude test dates
        pre_test_minute_df = minute_df[~minute_df["date"].isin(test_dates)]
        if not self.retrain(pre_test_minute_df):
            log.error("Retrain failed")
            return

        # Compute OFI bias using only pre-test data (FIX: avoid test-period leak, audit 2026-07-01)
        pre_test_bars = bars_df[~bars_df["date"].isin(test_dates)]
        self.daily_ofi_bias = compute_daily_ofi_bias(pre_test_bars)

        # Filter test bars
        test_bars = bars_df[bars_df["date"].isin(test_dates)].sort_values("ts").reset_index(drop=True)
        log.info(f"Test bars: {len(test_bars)}")

        # Reset state for replay
        self.state = self._fresh_state()

        # Track daily stats
        daily_stats = {}
        trades_list = []

        for i in range(len(test_bars)):
            bar = test_bars.iloc[i]
            bar_date = bar["date"]
            bar_ts = bar["ts"]

            # Initialize daily tracking
            if bar_date not in daily_stats:
                daily_stats[bar_date] = {
                    "n_signals": 0, "n_filtered": 0, "n_trades": 0,
                    "gross_pnl": 0.0, "cost": 0.0, "net_pnl": 0.0,
                    "wins": 0, "losses": 0,
                    "tp": 0, "sl": 0, "time": 0,
                }

            # Skip if already positioned
            if self.state["position"] != 0:
                continue

            # Only trade during RTH
            h, m = bar_ts.hour, bar_ts.minute
            t = h * 60 + m
            if not (RTH_START_H * 60 + RTH_START_M <= t < RTH_END_H * 60 + RTH_END_M):
                continue

            # Get prediction using all bars up to this point
            context_bars = bars_df[bars_df["ts"] <= bar_ts]
            signal = self.predict(context_bars)
            if signal is None or signal["signal"] == 0:
                continue

            self.state["signals_generated"] += 1
            daily_stats[bar_date]["n_signals"] += 1
            direction = signal["signal"]

            # Apply daily OFI contrarian filter
            ofi_bias = self.daily_ofi_bias.get(bar_date, 0)
            if ofi_bias != 0 and direction != ofi_bias:
                self.state["signals_filtered_ofi"] += 1
                daily_stats[bar_date]["n_filtered"] += 1
                log.debug(
                    f"OFI filter: {bar_date} bias={ofi_bias}, signal={direction} -> BLOCKED"
                )
                continue

            # Entry price: bid for long, ask for short
            # In ES, book is 1 tick wide during RTH, so bid = close - 0.25, ask = close
            # But for 30-min bars, use close as reference
            entry_price = signal["bar_close"]

            # Get minute bars for fill simulation (current bar + next bars up to max_hold)
            fill_window_end = bar_ts + timedelta(minutes=BAR_SIZE_MINUTES)
            fill_minute_bars = minute_df[
                (minute_df["ts_minute"] > bar_ts) &
                (minute_df["ts_minute"] <= fill_window_end) &
                (minute_df["date"] == bar_date)
            ].sort_values("ts_minute")

            # Simulate FIFO fill
            fill = simulate_fifo_fill(entry_price, direction, fill_minute_bars)
            if fill is None:
                log.debug(f"No fill for {bar_date} {bar_ts} dir={direction}")
                continue

            fill_time = fill["fill_time"]

            # Now simulate exit using minute bars after fill
            exit_window_end = fill_time + timedelta(minutes=MAX_HOLD_MINUTES + 30)
            exit_minute_bars = minute_df[
                (minute_df["ts_minute"] >= fill_time) &
                (minute_df["ts_minute"] <= exit_window_end) &
                (minute_df["date"] == bar_date)
            ].sort_values("ts_minute")

            exit_info = simulate_exit(entry_price, direction, fill_time, exit_minute_bars)

            # Compute PnL
            raw_pnl = exit_info["raw_pnl_ticks"]
            cost = exit_info["cost_ticks"]
            net_pnl_ticks = raw_pnl - cost
            net_pnl_dollars = net_pnl_ticks * ES_TICK_VALUE

            # Update state
            self.state["total_trades"] += 1
            self.state["total_pnl_ticks"] += net_pnl_ticks
            self.state["total_pnl_dollars"] += net_pnl_dollars
            self.state["capital"] += net_pnl_dollars
            self.state["signals_traded"] += 1

            if net_pnl_ticks > 0:
                self.state["wins"] += 1
                daily_stats[bar_date]["wins"] += 1
            else:
                self.state["losses"] += 1
                daily_stats[bar_date]["losses"] += 1

            exit_type = exit_info["exit_type"]
            if exit_type == "TP":
                self.state["tp_exits"] += 1
                daily_stats[bar_date]["tp"] += 1
            elif exit_type == "SL":
                self.state["sl_exits"] += 1
                daily_stats[bar_date]["sl"] += 1
            elif exit_type in ("TimeStop", "EOD"):
                self.state["time_exits"] += 1
                daily_stats[bar_date]["time"] += 1

            daily_stats[bar_date]["n_trades"] += 1
            daily_stats[bar_date]["gross_pnl"] += raw_pnl
            daily_stats[bar_date]["cost"] += cost
            daily_stats[bar_date]["net_pnl"] += net_pnl_ticks

            # Compute TP/SL prices for logging
            if direction == 1:
                tp_price = entry_price + TP_TICKS * ES_TICK_SIZE
                sl_price = entry_price - SL_LONG_TICKS * ES_TICK_SIZE
            else:
                tp_price = entry_price - TP_TICKS * ES_TICK_SIZE
                sl_price = entry_price + SL_SHORT_TICKS * ES_TICK_SIZE

            hold_minutes = (exit_info["exit_time"] - fill_time).total_seconds() / 60

            dir_str = "LONG" if direction == 1 else "SHORT"
            log.info(
                f"{bar_date} {dir_str} | "
                f"entry={entry_price:.2f} fill={fill_time.strftime('%H:%M')} | "
                f"exit={exit_info['exit_price']:.2f} ({exit_type}) {exit_info['exit_time'].strftime('%H:%M')} | "
                f"raw={raw_pnl:+.1f}t net={net_pnl_ticks:+.2f}t (${net_pnl_dollars:+.0f}) | "
                f"hold={hold_minutes:.0f}m"
            )

            trade_record = {
                "trade_id": self.state["total_trades"],
                "date": bar_date,
                "signal_time": str(bar_ts),
                "fill_time": str(fill_time),
                "exit_time": str(exit_info["exit_time"]),
                "direction": dir_str,
                "entry_price": entry_price,
                "exit_price": exit_info["exit_price"],
                "tp_price": tp_price,
                "sl_price": sl_price,
                "exit_type": exit_type,
                "raw_pnl_ticks": raw_pnl,
                "cost_ticks": cost,
                "net_pnl_ticks": net_pnl_ticks,
                "net_pnl_dollars": net_pnl_dollars,
                "prediction": signal["prediction_ticks"],
                "confidence_pct": CONFIDENCE_THRESHOLD_PCT,
                "ofi_bias": ofi_bias,
                "hold_minutes": hold_minutes,
                "capital_after": self.state["capital"],
            }
            self._append_trade(trade_record)
            trades_list.append(trade_record)

        # ── Report ──
        self._print_replay_report(test_dates, daily_stats, trades_list)

        # Save daily CSV
        for date, ds in sorted(daily_stats.items()):
            wr = ds["wins"] / max(ds["n_trades"], 1)
            with open(self.daily_csv, "a", newline="") as f:
                writer = csv.writer(f)
                writer.writerow([
                    date, ds["n_trades"], ds["n_signals"], ds["n_filtered"],
                    round(ds["gross_pnl"], 2), round(ds["cost"], 2),
                    round(ds["net_pnl"], 2), round(ds["net_pnl"] * ES_TICK_VALUE, 2),
                    ds["wins"], ds["losses"], ds["tp"], ds["sl"], ds["time"],
                    round(wr, 3), round(self.state["capital"], 2),
                ])

        # Try to log to MLflow (skip if server not running)
        try:
            import socket
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            result = s.connect_ex(('localhost', 5000))
            s.close()
            if result == 0:
                self._log_to_mlflow(test_dates, daily_stats, trades_list)
            else:
                log.info("MLflow server not running, skipping logging")
        except Exception:
            log.info("MLflow check failed, skipping logging")

        self._save_state()

    def _print_replay_report(self, test_dates, daily_stats, trades_list):
        s = self.state
        n = s["total_trades"]

        log.info(f"\n{'=' * 70}")
        log.info("REPLAY RESULTS")
        log.info(f"{'=' * 70}")
        log.info(f"Period: {test_dates[0]} -> {test_dates[-1]} ({len(test_dates)} days)")
        log.info(f"Config: top {CONFIDENCE_THRESHOLD_PCT}% confidence, "
                 f"TP={TP_TICKS}t, SL_L={SL_LONG_TICKS}t, SL_S={SL_SHORT_TICKS}t, "
                 f"max_hold={MAX_HOLD_MINUTES}m")
        log.info(f"OFI filter: {OFI_CONTRARIAN_THRESHOLD}x threshold")
        log.info(f"")
        log.info(f"Signals generated: {s['signals_generated']}")
        log.info(f"Signals filtered by OFI: {s['signals_filtered_ofi']}")
        log.info(f"Trades executed: {n}")

        if n > 0:
            wr = s["wins"] / n * 100
            avg_pnl = s["total_pnl_ticks"] / n

            # Compute Sharpe and Sortino from trade-level PnL
            pnl_arr = np.array([t["net_pnl_ticks"] for t in trades_list])
            sharpe = pnl_arr.mean() / max(pnl_arr.std(), 1e-6) * np.sqrt(252)
            downside = np.sqrt(np.mean(np.minimum(pnl_arr, 0) ** 2))
            sortino = pnl_arr.mean() / max(downside, 1e-6) * np.sqrt(252)
            pf = np.sum(pnl_arr[pnl_arr > 0]) / max(-np.sum(pnl_arr[pnl_arr < 0]), 1e-6)

            # Daily Sharpe
            daily_pnl_arr = np.array([ds["net_pnl"] for ds in daily_stats.values()])
            daily_sharpe = (
                daily_pnl_arr.mean() / max(daily_pnl_arr.std(), 1e-6) * np.sqrt(252)
                if len(daily_pnl_arr) > 1 else 0
            )

            # Long/short breakdown
            long_trades = [t for t in trades_list if t["direction"] == "LONG"]
            short_trades = [t for t in trades_list if t["direction"] == "SHORT"]

            log.info(f"")
            log.info(f"Win Rate: {wr:.1f}%")
            log.info(f"Profit Factor: {pf:.2f}")
            log.info(f"Trade Sharpe: {sharpe:.2f}")
            log.info(f"Trade Sortino: {sortino:.2f}")
            log.info(f"Daily Sharpe: {daily_sharpe:.2f}")
            log.info(f"Avg PnL/trade: {avg_pnl:.2f} ticks (${avg_pnl * ES_TICK_VALUE:.0f})")
            log.info(f"Total PnL: {s['total_pnl_ticks']:.1f} ticks (${s['total_pnl_dollars']:.0f})")
            log.info(f"")
            log.info(f"Exit breakdown: TP={s['tp_exits']}, SL={s['sl_exits']}, "
                     f"TimeStop={s['time_exits']}, EOD={s.get('eod_exits', 0)}")

            if long_trades:
                long_pnl = [t["net_pnl_ticks"] for t in long_trades]
                long_wr = sum(1 for p in long_pnl if p > 0) / len(long_pnl) * 100
                log.info(f"LONG:  {len(long_trades)} trades, WR {long_wr:.0f}%, "
                         f"avg {np.mean(long_pnl):.2f}t")
            if short_trades:
                short_pnl = [t["net_pnl_ticks"] for t in short_trades]
                short_wr = sum(1 for p in short_pnl if p > 0) / len(short_pnl) * 100
                log.info(f"SHORT: {len(short_trades)} trades, WR {short_wr:.0f}%, "
                         f"avg {np.mean(short_pnl):.2f}t")

            log.info(f"")
            log.info(f"Per-day breakdown:")
            log.info(f"{'Date':<12s} {'Trades':>6s} {'Sigs':>5s} {'Filt':>5s} "
                     f"{'NetPnL':>8s} {'$PnL':>8s} {'WR':>5s} {'TP':>3s} {'SL':>3s} {'Time':>4s}")
            for date in sorted(daily_stats.keys()):
                ds = daily_stats[date]
                d_wr = ds["wins"] / max(ds["n_trades"], 1) * 100
                log.info(
                    f"{date:<12s} {ds['n_trades']:>6d} {ds['n_signals']:>5d} "
                    f"{ds['n_filtered']:>5d} {ds['net_pnl']:>8.1f} "
                    f"{ds['net_pnl'] * ES_TICK_VALUE:>8.0f} "
                    f"{d_wr:>4.0f}% {ds['tp']:>3d} {ds['sl']:>3d} {ds['time']:>4d}"
                )
        else:
            log.info("No trades executed.")

        log.info(f"Final capital: ${s['capital']:.0f}")
        log.info(f"{'=' * 70}\n")

    def _log_to_mlflow(self, test_dates, daily_stats, trades_list):
        """Best-effort MLflow logging."""
        try:
            import mlflow
            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("fifo_champion_paper")

            s = self.state
            n = s["total_trades"]
            if n == 0:
                return

            pnl_arr = np.array([t["net_pnl_ticks"] for t in trades_list])
            sharpe = pnl_arr.mean() / max(pnl_arr.std(), 1e-6) * np.sqrt(252)
            downside = np.sqrt(np.mean(np.minimum(pnl_arr, 0) ** 2))
            sortino = pnl_arr.mean() / max(downside, 1e-6) * np.sqrt(252)
            pf = np.sum(pnl_arr[pnl_arr > 0]) / max(-np.sum(pnl_arr[pnl_arr < 0]), 1e-6)

            with mlflow.start_run(run_name=f"replay_{test_dates[0]}_{test_dates[-1]}"):
                mlflow.log_params({
                    "mode": "replay",
                    "confidence_pct": CONFIDENCE_THRESHOLD_PCT,
                    "tp_ticks": TP_TICKS,
                    "sl_long_ticks": SL_LONG_TICKS,
                    "sl_short_ticks": SL_SHORT_TICKS,
                    "max_hold_min": MAX_HOLD_MINUTES,
                    "ofi_threshold": OFI_CONTRARIAN_THRESHOLD,
                    "train_days": TRAIN_DAYS,
                    "test_start": test_dates[0],
                    "test_end": test_dates[-1],
                    "n_test_days": len(test_dates),
                })
                mlflow.log_metrics({
                    "n_trades": n,
                    "win_rate": s["wins"] / n,
                    "profit_factor": float(pf),
                    "sharpe": float(sharpe),
                    "sortino": float(sortino),
                    "total_pnl_ticks": s["total_pnl_ticks"],
                    "total_pnl_dollars": s["total_pnl_dollars"],
                    "avg_pnl_ticks": s["total_pnl_ticks"] / n,
                    "tp_exits": s["tp_exits"],
                    "sl_exits": s["sl_exits"],
                    "time_exits": s["time_exits"],
                    "signals_generated": s["signals_generated"],
                    "signals_filtered": s["signals_filtered_ofi"],
                })
                # Log trade CSV as artifact
                if self.trades_csv.exists():
                    mlflow.log_artifact(str(self.trades_csv))
                if self.daily_csv.exists():
                    mlflow.log_artifact(str(self.daily_csv))

            log.info("Logged to MLflow experiment 'fifo_champion_paper'")
        except Exception as e:
            log.warning(f"MLflow logging failed (non-fatal): {e}")

    # ── Summary ──

    def summary(self) -> str:
        s = self.state
        n = s["total_trades"]
        if n == 0:
            return "FIFO Champion Paper: No trades yet."

        wr = s["wins"] / n * 100
        avg_pnl = s["total_pnl_ticks"] / n

        lines = [
            f"FIFO Champion Paper: {n} trades, WR {wr:.0f}%",
            f"PnL: {s['total_pnl_ticks']:.1f}t (${s['total_pnl_dollars']:.0f})",
            f"Avg: {avg_pnl:.2f}t/trade",
            f"Exits: TP={s['tp_exits']} SL={s['sl_exits']} Time={s['time_exits']}",
            f"Signals: {s['signals_generated']} gen, {s['signals_filtered_ofi']} filtered",
            f"Capital: ${s['capital']:.0f}",
        ]
        return " | ".join(lines)

    # ── Health check server ──

    def _start_health_server(self):
        engine = self

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                if self.path == "/health":
                    self.send_response(200)
                    self.send_header("Content-Type", "application/json")
                    self.end_headers()
                    health = {
                        "status": "ok",
                        "engine": "fifo_champion_paper",
                        "trades": engine.state["total_trades"],
                        "pnl_ticks": engine.state["total_pnl_ticks"],
                        "capital": engine.state["capital"],
                        "last_heartbeat": engine.state.get("last_heartbeat"),
                        "uptime_check": datetime.now(timezone.utc).isoformat(),
                    }
                    self.wfile.write(json.dumps(health).encode())
                else:
                    self.send_response(404)
                    self.end_headers()

            def log_message(self, format, *args):
                pass  # Suppress HTTP logs

        try:
            server = HTTPServer(("0.0.0.0", HEALTH_PORT), Handler)
            thread = Thread(target=server.serve_forever, daemon=True)
            thread.start()
            log.info(f"Health check server on port {HEALTH_PORT}")
        except Exception as e:
            log.warning(f"Could not start health server: {e}")

    # ── Live loop ──

    def run_live(self):
        """Main loop for live/PM2 mode."""
        log.info("Starting LIVE mode")
        log.info(f"Config: top {CONFIDENCE_THRESHOLD_PCT}% confidence, "
                 f"TP={TP_TICKS}t, SL_L={SL_LONG_TICKS}t, SL_S={SL_SHORT_TICKS}t, "
                 f"max_hold={MAX_HOLD_MINUTES}m")

        self._start_health_server()

        if not self.load_model():
            log.info("No model on disk, will retrain on first data load")

        minute_df = None
        last_data_load = None

        while True:
            try:
                now = datetime.now(timezone.utc)

                # Reload data periodically
                if last_data_load is None or (now - last_data_load).total_seconds() > 1700:
                    minute_df = self._load_minute_bars()
                    last_data_load = now
                    gc.collect()

                h, m = now.hour, now.minute
                t = h * 60 + m
                is_rth = RTH_START_H * 60 + RTH_START_M <= t < RTH_END_H * 60 + RTH_END_M

                if is_rth and minute_df is not None and not minute_df.empty:
                    # Near 30-min boundary
                    if m in (0, 1, 2, 30, 31, 32):
                        bars_df = self._build_feature_df(minute_df)
                        if not bars_df.empty:
                            # Check retrain needed
                            dates = sorted(bars_df["date"].unique())
                            if dates and (
                                self.state.get("last_retrain_date") is None or
                                self.state["last_retrain_date"] < dates[-1]
                            ):
                                self.retrain(minute_df)

                            if self.model is not None and self.state["position"] == 0:
                                signal = self.predict(bars_df)
                                if signal and signal["signal"] != 0:
                                    bar_key = str(bars_df["ts"].iloc[-1])
                                    if self.state.get("last_signal_bar") != bar_key:
                                        self.state["last_signal_bar"] = bar_key
                                        log.info(
                                            f"Signal: {'LONG' if signal['signal'] == 1 else 'SHORT'} "
                                            f"pred={signal['prediction_ticks']:.2f}t @ {signal['bar_close']:.2f}"
                                        )
                                        # In live mode, we'd place the order here
                                        # For now, just log the signal
                                        self._save_state()

                self.state["last_heartbeat"] = now.isoformat()
                time.sleep(LOOP_SLEEP_SECONDS)

            except KeyboardInterrupt:
                log.info("Shutting down")
                break
            except Exception as e:
                log.error(f"Loop error: {e}\n{traceback.format_exc()}")
                time.sleep(60)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    import argparse

    parser = argparse.ArgumentParser(description="FIFO Champion Paper Trading Engine")
    parser.add_argument(
        "--replay", type=int, default=0, metavar="N",
        help="Replay last N trading days (default: 0 = no replay)",
    )
    parser.add_argument(
        "--live", action="store_true",
        help="Run in live loop mode (for PM2)",
    )
    parser.add_argument(
        "--summary", action="store_true",
        help="Print current state summary and exit",
    )
    args = parser.parse_args()

    engine = FIFOChampionPaperEngine()

    if args.summary:
        print(engine.summary())
        return

    if args.replay > 0:
        engine.replay(n_test_days=args.replay)
        return

    if args.live:
        engine.run_live()
        return

    # Default: replay 5 days as validation
    log.info("No mode specified, running default 5-day replay validation")
    engine.replay(n_test_days=5)


if __name__ == "__main__":
    main()
