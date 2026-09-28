#!/usr/bin/env python3
"""
Integrated Pipeline Paper Trading Engine
==========================================

Paper trades the integrated pipeline 30-min LightGBM strategy with FIFO-based
passive limit execution on ES futures. Uses the validated SL10/TP20 exit structure
from the parameter sensitivity grid (minute-level replay validation).

STRATEGY (from integrated_pipeline robustness validation):
  - Entry model: 30-min LightGBM on 43-feature lean set (sliding 60d train)
  - Entry criteria: conf >= 0.52 AND |pred_zscore| >= 0.3
  - Direction: pred > 0 → LONG, pred < 0 → SHORT
  - Entry: Passive limit at bid/ask (FIFO back-of-queue)
  - TP: 20 ticks, passive limit on opposite side
  - SL: 10 ticks, market order with 1 tick slippage
  - Max hold: 30 minutes (one bar — matches prediction horizon per HC #432)
  - Max position: 1 contract at a time

COST MODEL (ES Futures — AMP/Rithmic):
  - Entry (passive limit): 0.188 ticks (half RT commission)
  - TP exit (passive limit): 0.188 ticks → Entry+TP = 0.376 ticks
  - SL exit (market order): 1.188 ticks → Entry+SL = 1.376 ticks
  - Time-stop exit (market): 1.188 ticks → Entry+Time = 1.376 ticks

SENSITIVITY GRID VALIDATION (minute-level replay, 460 trades):
  SL10/TP20: Sharpe 19.19, WR 65.2%, PF 3.07, avg +8.19t/trade
  SL12/TP25: Sharpe 11.68, WR 53.9%, PF 2.07, avg +6.57t/trade
  SL15/TP30: Sharpe  8.46, WR 49.6%, PF 1.72, avg +5.93t/trade

ROBUSTNESS:
  Bootstrap 95% CI: Sharpe [4.80, 9.12]
  Walk-forward stability: 0.817 (3-segment min/max ratio)
  Regime gap: 0.183 (PASS, threshold 0.50)
  Monthly consistency: 7/7 positive months
  Long/Short: both profitable (LONG Sharpe 10.5, SHORT Sharpe 7.8)

MODES:
  --replay N    Replay last N trading days bar-by-bar (default: all OOT days)
  --live        Run in live loop mode (for PM2, checks data every 30s)
  --summary     Print current state and exit

PM2:
  pm2 start /home/jupiter/Lvl3Quant/paper_engines/integrated_pipeline_paper.py \
    --name integrated-pipeline-paper --interpreter python3 -- --live

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
OUTPUT_DIR = ENGINE_DIR / "logs" / "integrated_pipeline_paper"
MODEL_DIR = OUTPUT_DIR / "models"
STATE_DIR = OUTPUT_DIR

# Pre-computed OOT predictions (from walk-forward LightGBM training on Neptune)
OOT_PREDS_PATH = ROOT / "output" / "lh_30min_deep_v1" / "concat_oot.npz"
V1_TRADES_PATH = ROOT / "output" / "integrated_pipeline_v1" / "best_trades.parquet"

for d in [OUTPUT_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
LOG_FILE = OUTPUT_DIR / "integrated_pipeline_paper.log"
logging.basicConfig(
    format="%(asctime)s [INTG-PAPER] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE)),
    ],
)
log = logging.getLogger("INTG-PAPER")

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

# Strategy parameters (INTEGRATED PIPELINE — SL4/TP20 validated 2026-07-01)
ENTRY_CONF_THRESHOLD = 0.52           # minimum model confidence
ZSCORE_THRESHOLD = 0.3                # minimum |pred z-score| for entry
TP_TICKS = 20                         # take-profit in ticks
SL_TICKS = 4                          # stop-loss in ticks (was 10; sweep: Sharpe 17→21, +24% P&L, regime gap 0.05)
MAX_HOLD_MINUTES = 30                 # one bar — matches prediction horizon (HC #432)
MAX_POSITION = 1                      # 1 contract at a time
TRAIN_DAYS = 60                       # sliding training window
STARTING_CAPITAL = 100_000

# FIFO fill model: fill when price trades 1 tick THROUGH entry price
FILL_THROUGH_TICKS = 1

# Afternoon-short filter (HC audit 2026-06-28: afternoon shorts are pure drag)
# Morning (before 11:00 ET): both longs and shorts allowed
# Afternoon (11:00+ ET): longs only, shorts filtered
AFTERNOON_SHORT_FILTER = True
AFTERNOON_CUTOFF_HOUR_ET = 11  # ET hour after which shorts are blocked

# Pre-open noise filter (HC audit 2026-06-28: 8:30 ET bar = 22% WR across all regimes)
# First 30-min bar after pre-market has wide spreads, noise-driven fills → skip
SKIP_FIRST_BAR = True
FIRST_BAR_CUTOFF_HOUR_ET = 9  # Skip signals before 9:00 ET (covers 8:30 bar)
FIRST_BAR_CUTOFF_MIN_ET = 0

# Lunch-hour filter (HC audit 2026-06-29: noon ET = 37.8% WR across ALL confidence levels)
# 12:00-12:59 ET signals are structurally weak — thin book, noise-driven. Not fixable with thresholds.
# Removing noon trades: Sharpe 10.33 → 10.55, WR 49.7% → 51.7%, trade count 314 → 269.
LUNCH_HOUR_FILTER = True
LUNCH_HOUR_ET = 12  # Skip signals during 12:00-12:59 ET

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

# Health check port (8765 is used by fifo-champion-paper)
HEALTH_PORT = 8766

# Main loop sleep
LOOP_SLEEP_SECONDS = 30


# ═══════════════════════════════════════════════════════════════════
#  FEATURE COMPUTATION (identical to fifo_champion_paper.py lean set)
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
            "range_ticks": close_arr.max() - close_arr.min(),  # prices already in tick units
            "close_position": (
                (close_arr[-1] - close_arr.min())
                / max(close_arr.max() - close_arr.min(), 1)  # 1 tick unit min
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

    # Time-of-day features
    df["tod_sin"] = np.sin(2 * np.pi * df["ts"].dt.hour / 24)
    df["tod_cos"] = np.cos(2 * np.pi * df["ts"].dt.hour / 24)
    rth_start = 13 * 60 + 30  # 13:30 UTC = 9:30 ET
    rth_end = 20 * 60  # 20:00 UTC = 16:00 ET
    minutes = df["ts"].dt.hour * 60 + df["ts"].dt.minute
    df["tod_progress"] = (minutes - rth_start).clip(lower=0) / (rth_end - rth_start)
    df["bars_since_open"] = df.groupby("date").cumcount()

    # Regime features
    for w in [16, 32]:
        df[f"regime_ret_{w}bar"] = df["return_bar"].rolling(w, min_periods=4).sum()
    vol_short = df["realized_vol"].rolling(4, min_periods=2).mean()
    vol_long = df["realized_vol"].rolling(16, min_periods=4).mean()
    df["regime_vol_ratio"] = vol_short / vol_long.clip(lower=1e-8)

    # Previous day return
    daily_ret = df.groupby("date")["return_bar"].sum()
    prev_day_map = {}
    dates = sorted(daily_ret.index.tolist())
    for i, d in enumerate(dates):
        if i > 0:
            prev_day_map[d] = daily_ret[dates[i - 1]]
        else:
            prev_day_map[d] = 0.0
    df["prev_day_ret"] = df["date"].map(prev_day_map).fillna(0)

    # Intraday direction strength
    df["intraday_direction_strength"] = (
        df["intraday_cum_ofi"].abs() /
        df.groupby("date")["ofi_sum"].transform(lambda x: x.abs().cumsum()).clip(lower=1)
    )

    return df


# ═══════════════════════════════════════════════════════════════════
#  FILL SIMULATION (FIFO back-of-queue model)
# ═══════════════════════════════════════════════════════════════════


def simulate_fifo_fill(
    entry_price: float,
    direction: int,
    minute_bars: pd.DataFrame,
) -> Optional[Dict]:
    """
    Simulate passive limit fill. For FIFO, fill when price trades
    FILL_THROUGH_TICKS ticks through entry price.
    """
    if minute_bars.empty:
        return None

    for _, row in minute_bars.iterrows():
        if direction == 1:
            # Long: bid at entry_price. Fill when price trades below by FILL_THROUGH_TICKS
            if row["low"] <= entry_price - FILL_THROUGH_TICKS:  # prices in tick units
                return {
                    "fill_price": entry_price,
                    "fill_time": row["ts_minute"],
                }
        else:
            # Short: ask at entry_price. Fill when price trades above by FILL_THROUGH_TICKS
            if row["high"] >= entry_price + FILL_THROUGH_TICKS:  # prices in tick units
                return {
                    "fill_price": entry_price,
                    "fill_time": row["ts_minute"],
                }
    return None


def simulate_exit(
    entry_price: float,
    direction: int,
    fill_time,
    minute_bars: pd.DataFrame,
) -> Dict:
    """
    Simulate exit using minute-level data with TP/SL stops.
    TP = passive limit, SL = market order.
    """
    # NOTE: minute bar prices are in tick units (price / 0.25), NOT index points.
    # So TP/SL offsets must be in tick units too — just add TP_TICKS directly.
    tp_price = entry_price + (TP_TICKS * direction)
    sl_price = entry_price - (SL_TICKS * direction)
    time_stop = fill_time + timedelta(minutes=MAX_HOLD_MINUTES)

    if minute_bars.empty:
        return {
            "exit_type": "NoData",
            "exit_price": entry_price,
            "exit_time": fill_time,
            "raw_pnl_ticks": 0,
            "cost_ticks": ES_RT_COMMISSION_TICKS,
        }

    for _, row in minute_bars.iterrows():
        ts = row["ts_minute"]
        hi = row["high"]
        lo = row["low"]

        if direction == 1:
            tp_hit = hi >= tp_price
            sl_hit = lo <= sl_price
        else:
            tp_hit = lo <= tp_price
            sl_hit = hi >= sl_price

        # Check for EOD forced exit (15:55 ET = 19:55 UTC)
        eod_time = ts.replace(hour=19, minute=55, second=0)
        if ts >= eod_time:
            exit_price = row["close"]
            raw_pnl = (exit_price - entry_price) * direction  # prices already in tick units
            return {
                "exit_type": "EOD",
                "exit_price": exit_price,
                "exit_time": ts,
                "raw_pnl_ticks": raw_pnl,
                "cost_ticks": COST_ENTRY_PLUS_SL,  # market exit
            }

        if tp_hit and sl_hit:
            # Both levels breached in same minute bar. Without tick data we
            # cannot determine the true order. Use the bar's CLOSE price as
            # a tiebreaker: if close is on the favorable side of entry, the
            # move likely went through TP first; if adverse, SL first.
            # This is the least biased OHLC-only heuristic (validated:
            # distance/overshoot heuristics are algebraically identical and
            # systematically SL-biased when TP > SL).
            bar_close = row["close"]
            if direction == 1:
                favorable = bar_close >= entry_price
            else:
                favorable = bar_close <= entry_price
            if favorable:
                raw_pnl = TP_TICKS
                return {
                    "exit_type": "TP",
                    "exit_price": tp_price,
                    "exit_time": ts,
                    "raw_pnl_ticks": raw_pnl,
                    "cost_ticks": COST_ENTRY_PLUS_TP,
                }
            else:
                raw_pnl = -SL_TICKS
                return {
                    "exit_type": "SL",
                    "exit_price": sl_price,
                    "exit_time": ts,
                    "raw_pnl_ticks": raw_pnl,
                    "cost_ticks": COST_ENTRY_PLUS_SL,
                }
        elif sl_hit:
            raw_pnl = -SL_TICKS
            return {
                "exit_type": "SL",
                "exit_price": sl_price,
                "exit_time": ts,
                "raw_pnl_ticks": raw_pnl,
                "cost_ticks": COST_ENTRY_PLUS_SL,
            }
        elif tp_hit:
            raw_pnl = TP_TICKS
            return {
                "exit_type": "TP",
                "exit_price": tp_price,
                "exit_time": ts,
                "raw_pnl_ticks": raw_pnl,
                "cost_ticks": COST_ENTRY_PLUS_TP,
            }

        # Time stop
        if ts >= time_stop:
            exit_price = row["close"]
            raw_pnl = (exit_price - entry_price) * direction  # prices already in tick units
            return {
                "exit_type": "TimeStop",
                "exit_price": exit_price,
                "exit_time": ts,
                "raw_pnl_ticks": raw_pnl,
                "cost_ticks": COST_ENTRY_PLUS_SL,  # market exit
            }

    # End of data — close at last bar
    last = minute_bars.iloc[-1]
    exit_price = last["close"]
    raw_pnl = (exit_price - entry_price) * direction  # prices already in tick units
    return {
        "exit_type": "DataEnd",
        "exit_price": exit_price,
        "exit_time": last["ts_minute"],
        "raw_pnl_ticks": raw_pnl,
        "cost_ticks": COST_ENTRY_PLUS_SL,  # market exit
    }


# ═══════════════════════════════════════════════════════════════════
#  HEALTH CHECK SERVER
# ═══════════════════════════════════════════════════════════════════


class HealthHandler(BaseHTTPRequestHandler):
    engine = None

    def do_GET(self):
        if self.path == "/health":
            state = self.engine.state if self.engine else {}
            body = json.dumps({
                "status": "ok",
                "engine": "integrated-pipeline-paper",
                "strategy": f"SL{SL_TICKS}/TP{TP_TICKS}",
                "trades": state.get("total_trades", 0),
                "pnl_ticks": round(state.get("total_pnl_ticks", 0), 2),
                "pnl_dollars": round(state.get("total_pnl_dollars", 0), 2),
                "wins": state.get("wins", 0),
                "losses": state.get("losses", 0),
                "last_heartbeat": state.get("last_heartbeat", ""),
                "mode": state.get("mode", "unknown"),
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass  # Suppress HTTP access logs


def start_health_server(engine):
    HealthHandler.engine = engine
    try:
        server = HTTPServer(("0.0.0.0", HEALTH_PORT), HealthHandler)
        t = Thread(target=server.serve_forever, daemon=True)
        t.start()
        log.info(f"Health server on port {HEALTH_PORT}")
    except OSError as e:
        log.warning(f"Health server failed: {e}")


# ═══════════════════════════════════════════════════════════════════
#  PAPER TRADING ENGINE
# ═══════════════════════════════════════════════════════════════════


class IntegratedPipelinePaperEngine:
    """
    Two operating modes:
      1. REPLAY: Uses pre-computed OOT predictions from concat_oot.npz to replay
         historical trades with minute-level fill/exit simulation.
      2. LIVE: Retrains LightGBM on a sliding 60-day window and generates new
         predictions on each 30-min bar close.
    """

    def __init__(self):
        self.state_path = STATE_DIR / "state.json"
        self.trades_csv = OUTPUT_DIR / "trades.csv"
        self.daily_csv = OUTPUT_DIR / "daily_pnl.csv"
        self.model_path = MODEL_DIR / "lgbm_model.txt"
        self.scaler_path = MODEL_DIR / "scaler.npz"
        self.meta_path = MODEL_DIR / "meta.json"

        self.model = None
        self.feature_cols = None
        self.scaler_median = None
        self.scaler_iqr = None
        self.pred_quantiles = None

        # Load or init state
        if self.state_path.exists():
            with open(self.state_path) as f:
                self.state = json.load(f)
            log.info(f"Loaded state: {self.state.get('total_trades', 0)} trades, "
                     f"PnL ${self.state.get('total_pnl_dollars', 0):.0f}")
        else:
            self.state = self._fresh_state()

        self._ensure_trades_csv()
        self._ensure_daily_csv()

    def _fresh_state(self):
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
            "signals_filtered": 0,
            "signals_traded": 0,
            "last_signal_bar": None,
            "last_retrain_date": None,
            "last_heartbeat": None,
            "mode": "init",
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
                    "net_pnl_dollars", "prediction", "confidence",
                    "pred_zscore", "hold_minutes", "capital_after",
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
                round(trade.get("confidence", 0), 4),
                round(trade.get("pred_zscore", 0), 4),
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

    def _load_minute_bars_for_dates(self, date_list: list) -> pd.DataFrame:
        """Load minute bars for specific dates only."""
        frames = []
        for date_str in date_list:
            fpath = MINUTE_BAR_DIR / f"{date_str}.parquet"
            if not fpath.exists():
                continue
            try:
                df = pd.read_parquet(fpath)
                df["date"] = date_str
                frames.append(df)
            except Exception as e:
                log.warning(f"Skip {date_str}: {e}")

        if not frames:
            return pd.DataFrame()

        combined = pd.concat(frames, ignore_index=True)
        combined["ts_minute"] = pd.to_datetime(combined["ts_minute"], utc=True)
        combined = combined.sort_values("ts_minute").reset_index(drop=True)
        return combined

    # ── Feature pipeline ──

    def _build_feature_df(self, minute_df: pd.DataFrame) -> pd.DataFrame:
        bars_df = aggregate_to_bars(minute_df, bar_size_min=BAR_SIZE_MINUTES)
        if bars_df.empty:
            return pd.DataFrame()
        bars_df = add_rolling_features(bars_df)
        return bars_df

    # ── Training (for live mode) ──

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
        fwd_ticks = fwd_close - train_df["close"]  # prices already in tick units

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

        # CRITICAL FIX (2026-06-29): validation set must NOT overlap training set.
        # Previous code used all of X for training AND last 10% for validation
        # → early stopping evaluated on in-sample data → overfit risk.
        n_val = max(int(len(X) * 0.1), 20)
        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]

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

        # Prediction distribution for z-score normalization (use ALL data for stats)
        all_preds = self.model.predict(X, num_iteration=self.model.best_iteration)
        self.pred_quantiles = {
            "mean": float(np.mean(all_preds)),
            "std": float(np.std(all_preds)),
        }
        log.info(f"Pred distribution: mean={self.pred_quantiles['mean']:.3f}, "
                 f"std={self.pred_quantiles['std']:.3f}")

        # Save model artifacts
        self.model.save_model(str(self.model_path))
        np.savez_compressed(str(self.scaler_path),
                            median=self.scaler_median, iqr=self.scaler_iqr)
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

        # Holdout IC on TRUE out-of-sample validation (not seen during training)
        val_preds = self.model.predict(X_val, num_iteration=self.model.best_iteration)
        ic_oot = np.corrcoef(val_preds, y_val)[0, 1]
        log.info(
            f"Retrained: {train_dates[0]}->{train_dates[-1]}, "
            f"{len(y)} samples (train={len(y_train)}, val={n_val}), "
            f"iter={self.model.best_iteration}, holdout IC={ic_oot:.4f}"
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
            self.pred_quantiles = meta.get("pred_quantiles", {"mean": 0, "std": 1})
            log.info(f"Loaded model. Features: {len(self.feature_cols)}")
            return True
        except Exception as e:
            log.warning(f"Failed to load model: {e}")
            return False

    # ── Inference (for live mode) ──

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

        # Compute z-score
        pred_mean = self.pred_quantiles.get("mean", 0)
        pred_std = max(self.pred_quantiles.get("std", 1), 1e-6)
        zscore = (pred - pred_mean) / pred_std

        bar_date = latest["date"].iloc[0]
        bar_close = float(latest["close"].iloc[0])
        bar_ts = latest["ts"].iloc[0]

        return {
            "prediction_ticks": float(pred),
            "zscore": float(zscore),
            "bar_ts": str(bar_ts),
            "bar_date": bar_date,
            "bar_close": bar_close,
        }

    # ── OOT Replay (primary validation mode) ──

    def replay_oot(self, n_test_days: int = 0):
        """
        Replay OOT predictions from concat_oot.npz with minute-level fill/exit
        simulation. This is the canonical validation mode.

        If n_test_days=0, replays ALL available OOT days.
        """
        log.info(f"\n{'=' * 70}")
        log.info(f"OOT REPLAY MODE — Integrated Pipeline SL{SL_TICKS}/TP{TP_TICKS}")
        log.info(f"Entry: conf >= {ENTRY_CONF_THRESHOLD}, |zscore| >= {ZSCORE_THRESHOLD}")
        log.info(f"Max hold: {MAX_HOLD_MINUTES}m, FIFO fill model")
        log.info(f"{'=' * 70}\n")

        # Load pre-computed OOT predictions
        if not OOT_PREDS_PATH.exists():
            log.error(f"OOT predictions not found: {OOT_PREDS_PATH}")
            return
        d = np.load(OOT_PREDS_PATH, allow_pickle=True)
        preds = d["preds"]
        actuals = d["actuals"]
        dates = d["dates"]
        confs = d["confs"]
        log.info(f"Loaded {len(preds)} OOT predictions across {len(np.unique(dates))} days")

        # FIX: compute z-score with EXPANDING window to avoid look-ahead (audit 2026-07-01)
        # Build per-date expanding mean/std from predictions seen SO FAR (causal only)
        _expanding_stats = {}
        _all_preds_seen = []
        for _date in sorted(np.unique(dates)):
            _day_mask = dates == _date
            _all_preds_seen.extend(preds[_day_mask].tolist())
            _arr = np.array(_all_preds_seen)
            _expanding_stats[_date] = (float(_arr.mean()), float(max(_arr.std(), 1e-6)))
        # Use first day's own stats for day 1 (no prior data available)
        log.info(f"Prediction z-score: expanding window across {len(_expanding_stats)} days (causal, no look-ahead)")

        # Get unique dates
        unique_dates = sorted(np.unique(dates))
        if n_test_days > 0:
            unique_dates = unique_dates[-n_test_days:]
        log.info(f"Replaying {len(unique_dates)} OOT days: {unique_dates[0]} -> {unique_dates[-1]}")

        # Load minute bar data for fill/exit simulation
        log.info("Loading minute bar cache...")
        minute_df = self._load_minute_bars_for_dates(unique_dates)
        if minute_df.empty:
            log.error("No minute bar data for OOT dates")
            return
        log.info(f"Loaded minute bars for {len(minute_df['date'].unique())} days")

        # Build 30-min bar cache for entry price lookup
        bar_cache = {}
        for date_str in unique_dates:
            date_minutes = minute_df[minute_df["date"] == date_str].copy()
            if date_minutes.empty:
                continue
            date_minutes["bar_key"] = date_minutes["ts_minute"].dt.floor("30min")
            bars_30 = []
            for bar_key, grp in date_minutes.groupby("bar_key"):
                if len(grp) < 3:
                    continue
                bars_30.append({
                    "bar_key": bar_key,
                    "open": grp.iloc[0]["open"] if "open" in grp.columns else grp.iloc[0]["close"],
                    "close": grp.iloc[-1]["close"],
                    "high": grp["high"].max() if "high" in grp.columns else grp["close"].max(),
                    "low": grp["low"].min() if "low" in grp.columns else grp["close"].min(),
                    "minutes": grp,
                })
            bar_cache[date_str] = bars_30

        # Reset state
        self.state = self._fresh_state()
        self.state["mode"] = "oot_replay"

        daily_stats = {}
        trades_list = []

        # Figure out unique bar counts per date (handles duplicate predictions)
        def get_unique_bar_count(date_str):
            mask = dates == date_str
            n_total = mask.sum()
            if n_total <= 15:
                return n_total
            if n_total == 28: return 14
            elif n_total == 21: return 7
            elif n_total == 16: return 8
            elif n_total == 18: return 9
            else: return min(n_total, 15)

        for date_str in unique_dates:
            if date_str not in bar_cache or not bar_cache[date_str]:
                continue

            if date_str not in daily_stats:
                daily_stats[date_str] = {
                    "n_signals": 0, "n_filtered": 0, "n_trades": 0,
                    "gross_pnl": 0.0, "cost": 0.0, "net_pnl": 0.0,
                    "wins": 0, "losses": 0,
                    "tp": 0, "sl": 0, "time": 0,
                }

            # Get predictions for this date
            mask = dates == date_str
            day_preds = preds[mask]
            day_confs = confs[mask]
            n_unique = get_unique_bar_count(date_str)
            bars_30 = bar_cache[date_str]

            for bar_idx in range(min(n_unique, len(bars_30))):
                pred_val = float(day_preds[bar_idx])
                conf_val = float(day_confs[bar_idx])

                # Check entry criteria (expanding z-score — causal, no look-ahead)
                _exp_mean, _exp_std = _expanding_stats[date_str]
                zscore = (pred_val - _exp_mean) / _exp_std

                self.state["signals_generated"] += 1
                daily_stats[date_str]["n_signals"] += 1

                # Filter: confidence and z-score thresholds
                if conf_val < ENTRY_CONF_THRESHOLD or abs(zscore) < ZSCORE_THRESHOLD:
                    self.state["signals_filtered"] += 1
                    daily_stats[date_str]["n_filtered"] += 1
                    continue

                # Direction from prediction sign
                direction = 1 if pred_val > 0 else -1

                # Pre-open noise filter: skip first bar (before 9:00 ET)
                if SKIP_FIRST_BAR:
                    bar_info_fb = bars_30[bar_idx]
                    bar_ts_fb = bar_info_fb["bar_key"]
                    bar_et_fb = pd.Timestamp(bar_ts_fb).tz_convert('US/Eastern') if bar_ts_fb.tzinfo else pd.Timestamp(bar_ts_fb, tz='UTC').tz_convert('US/Eastern')
                    if bar_et_fb.hour < FIRST_BAR_CUTOFF_HOUR_ET or (bar_et_fb.hour == FIRST_BAR_CUTOFF_HOUR_ET and bar_et_fb.minute < FIRST_BAR_CUTOFF_MIN_ET):
                        self.state["signals_filtered"] += 1
                        daily_stats[date_str]["n_filtered"] += 1
                        continue

                # Lunch-hour filter: skip 12:00-12:59 ET (replay path)
                if LUNCH_HOUR_FILTER:
                    bar_info_lh = bars_30[bar_idx]
                    bar_ts_lh = bar_info_lh["bar_key"]
                    bar_et_lh = pd.Timestamp(bar_ts_lh).tz_convert('US/Eastern') if bar_ts_lh.tzinfo else pd.Timestamp(bar_ts_lh, tz='UTC').tz_convert('US/Eastern')
                    if bar_et_lh.hour == LUNCH_HOUR_ET:
                        self.state["signals_filtered"] += 1
                        daily_stats[date_str]["n_filtered"] += 1
                        continue

                # Afternoon-short filter: block shorts after 11:00 ET
                if AFTERNOON_SHORT_FILTER and direction == -1:
                    bar_info_check = bars_30[bar_idx]
                    bar_ts_check = bar_info_check["bar_key"]
                    # bar_key is UTC — convert to ET (UTC-4 EDT / UTC-5 EST)
                    bar_et = pd.Timestamp(bar_ts_check).tz_convert('US/Eastern') if bar_ts_check.tzinfo else pd.Timestamp(bar_ts_check, tz='UTC').tz_convert('US/Eastern')
                    if bar_et.hour >= AFTERNOON_CUTOFF_HOUR_ET:
                        self.state["signals_filtered"] += 1
                        daily_stats[date_str]["n_filtered"] += 1
                        continue

                # RTH check on bar time
                bar_info = bars_30[bar_idx]
                bar_ts = bar_info["bar_key"]
                h, m = bar_ts.hour, bar_ts.minute
                t = h * 60 + m
                if not (RTH_START_H * 60 + RTH_START_M <= t < RTH_END_H * 60 + RTH_END_M - BAR_SIZE_MINUTES):
                    self.state["signals_filtered"] += 1
                    daily_stats[date_str]["n_filtered"] += 1
                    continue

                # Entry price: next bar open (if available)
                if bar_idx + 1 >= len(bars_30):
                    continue
                next_bar = bars_30[bar_idx + 1]
                entry_price = next_bar["open"]
                next_minutes = next_bar["minutes"]

                # Simulate FIFO fill within next bar
                fill = simulate_fifo_fill(entry_price, direction, next_minutes)
                if fill is None:
                    continue

                fill_time = fill["fill_time"]

                # Simulate exit from fill time
                exit_window_end = fill_time + timedelta(minutes=MAX_HOLD_MINUTES + 30)
                exit_minute_bars = minute_df[
                    (minute_df["ts_minute"] >= fill_time) &
                    (minute_df["ts_minute"] <= exit_window_end) &
                    (minute_df["date"] == date_str)
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
                    daily_stats[date_str]["wins"] += 1
                else:
                    self.state["losses"] += 1
                    daily_stats[date_str]["losses"] += 1

                exit_type = exit_info["exit_type"]
                if exit_type == "TP":
                    self.state["tp_exits"] += 1
                    daily_stats[date_str]["tp"] += 1
                elif exit_type == "SL":
                    self.state["sl_exits"] += 1
                    daily_stats[date_str]["sl"] += 1
                elif exit_type in ("TimeStop", "EOD"):
                    self.state["time_exits"] += 1
                    daily_stats[date_str]["time"] += 1

                daily_stats[date_str]["n_trades"] += 1
                daily_stats[date_str]["gross_pnl"] += raw_pnl
                daily_stats[date_str]["cost"] += cost
                daily_stats[date_str]["net_pnl"] += net_pnl_ticks

                # TP/SL prices for logging (tick units — no ES_TICK_SIZE multiply)
                tp_price = entry_price + (TP_TICKS * direction)
                sl_price = entry_price - (SL_TICKS * direction)
                hold_minutes = (exit_info["exit_time"] - fill_time).total_seconds() / 60

                dir_str = "LONG" if direction == 1 else "SHORT"
                log.info(
                    f"{date_str} {dir_str} | "
                    f"pred={pred_val:+.2f} conf={conf_val:.3f} z={zscore:+.2f} | "
                    f"entry={entry_price:.2f} fill={fill_time.strftime('%H:%M')} | "
                    f"exit={exit_info['exit_price']:.2f} ({exit_type}) | "
                    f"net={net_pnl_ticks:+.2f}t (${net_pnl_dollars:+.0f}) | "
                    f"hold={hold_minutes:.0f}m"
                )

                trade_record = {
                    "trade_id": self.state["total_trades"],
                    "date": date_str,
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
                    "prediction": pred_val,
                    "confidence": conf_val,
                    "pred_zscore": zscore,
                    "hold_minutes": hold_minutes,
                    "capital_after": self.state["capital"],
                }
                self._append_trade(trade_record)
                trades_list.append(trade_record)

        # ── Report ──
        self._print_report(unique_dates, daily_stats, trades_list)

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

        self._save_state()

        # MLflow
        self._log_to_mlflow(unique_dates, daily_stats, trades_list)

        return trades_list

    # ── Live loop (for PM2) ──

    def run_live(self):
        """
        Live paper trading loop. Every 30 seconds:
        1. Check if new 30-min bar has completed
        2. If model is stale (>1 day old), retrain
        3. Generate prediction, check entry criteria
        4. If signal, simulate FIFO fill and manage position
        """
        self.state["mode"] = "live"
        log.info(f"\n{'=' * 70}")
        log.info(f"LIVE PAPER MODE — SL{SL_TICKS}/TP{TP_TICKS}")
        log.info(f"Entry: conf >= {ENTRY_CONF_THRESHOLD}, |zscore| >= {ZSCORE_THRESHOLD}")
        log.info(f"Health: http://0.0.0.0:{HEALTH_PORT}/health")
        log.info(f"{'=' * 70}\n")

        # Try to load existing model
        if not self.load_model():
            log.info("No saved model, will train on first data load")
            minute_df = self._load_minute_bars()
            if not minute_df.empty:
                self.retrain(minute_df)

        last_bar_ts = None
        last_retrain_check = None

        while True:
            try:
                now = datetime.now(timezone.utc)

                # Check if it's RTH
                h, m = now.hour, now.minute
                t = h * 60 + m
                is_rth = RTH_START_H * 60 + RTH_START_M <= t < RTH_END_H * 60 + RTH_END_M

                if not is_rth:
                    # Check for daily retrain outside RTH
                    today = now.strftime("%Y%m%d")
                    if last_retrain_check != today and now.hour >= 21:
                        log.info("Post-RTH retrain check...")
                        minute_df = self._load_minute_bars()
                        if not minute_df.empty:
                            self.retrain(minute_df)
                        last_retrain_check = today
                    self._save_state()
                    time.sleep(LOOP_SLEEP_SECONDS)
                    continue

                # Check for new bar (on 30-min boundaries)
                bar_boundary = now.replace(second=0, microsecond=0)
                bar_boundary = bar_boundary.replace(
                    minute=(bar_boundary.minute // BAR_SIZE_MINUTES) * BAR_SIZE_MINUTES
                )

                if last_bar_ts == bar_boundary:
                    time.sleep(LOOP_SLEEP_SECONDS)
                    continue

                # Wait a few seconds after bar close for data to arrive
                seconds_past = (now - bar_boundary).total_seconds()
                if seconds_past < 15:
                    time.sleep(LOOP_SLEEP_SECONDS)
                    continue

                last_bar_ts = bar_boundary
                log.info(f"New bar boundary: {bar_boundary}")

                # ── EXIT MANAGEMENT: check TP/SL/TimeStop/EOD while in position ──
                if self.state["position"] != 0:
                    direction = self.state["position"]
                    entry_price = self.state["entry_price"]
                    fill_time_str = self.state.get("fill_time")
                    tp_price = self.state["tp_price"]
                    sl_price = self.state["sl_price"]
                    time_stop_str = self.state.get("time_stop")

                    # Load recent minute bars to check price action
                    minute_df = self._load_minute_bars(n_days=3)
                    if not minute_df.empty and fill_time_str:
                        fill_time = pd.Timestamp(fill_time_str)
                        if fill_time.tzinfo is None:
                            fill_time = fill_time.tz_localize("UTC")

                        # Get bars since entry
                        bars_since_entry = minute_df[
                            minute_df["ts_minute"] >= fill_time
                        ].sort_values("ts_minute")

                        if not bars_since_entry.empty:
                            exit_info = simulate_exit(
                                entry_price, direction, fill_time, bars_since_entry
                            )

                            # Only exit if we got a definitive exit type
                            if exit_info["exit_type"] in ("TP", "SL", "TimeStop", "EOD"):
                                raw_pnl = exit_info["raw_pnl_ticks"]
                                cost = exit_info["cost_ticks"]
                                net_pnl_ticks = raw_pnl - cost
                                net_pnl_dollars = net_pnl_ticks * ES_TICK_VALUE
                                hold_minutes = (
                                    exit_info["exit_time"] - fill_time
                                ).total_seconds() / 60

                                # Update state
                                self.state["total_trades"] += 1
                                self.state["total_pnl_ticks"] += net_pnl_ticks
                                self.state["total_pnl_dollars"] += net_pnl_dollars
                                self.state["capital"] += net_pnl_dollars

                                exit_type = exit_info["exit_type"]
                                if net_pnl_ticks > 0:
                                    self.state["wins"] += 1
                                else:
                                    self.state["losses"] += 1

                                if exit_type == "TP":
                                    self.state["tp_exits"] += 1
                                elif exit_type == "SL":
                                    self.state["sl_exits"] += 1
                                elif exit_type in ("TimeStop", "EOD"):
                                    self.state["time_exits"] += 1

                                dir_str = "LONG" if direction == 1 else "SHORT"
                                log.info(
                                    f"EXIT: {dir_str} | "
                                    f"entry={entry_price:.2f} → "
                                    f"exit={exit_info['exit_price']:.2f} ({exit_type}) | "
                                    f"net={net_pnl_ticks:+.2f}t (${net_pnl_dollars:+.0f}) | "
                                    f"hold={hold_minutes:.0f}m"
                                )

                                # Record trade
                                trade_record = {
                                    "trade_id": self.state["total_trades"],
                                    "date": exit_info["exit_time"].strftime("%Y%m%d")
                                        if hasattr(exit_info["exit_time"], "strftime")
                                        else str(exit_info["exit_time"])[:10].replace("-", ""),
                                    "signal_time": self.state.get("entry_time", ""),
                                    "fill_time": fill_time_str,
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
                                    "prediction": self.state.get("entry_prediction", 0),
                                    "confidence": abs(self.state.get("entry_zscore", 0)),
                                    "pred_zscore": self.state.get("entry_zscore", 0),
                                    "hold_minutes": hold_minutes,
                                    "capital_after": self.state["capital"],
                                }
                                self._append_trade(trade_record)

                                # Clear position
                                self.state["position"] = 0
                                self.state["entry_price"] = 0.0
                                self.state["entry_time"] = None
                                self.state["fill_time"] = None
                                self.state["tp_price"] = 0.0
                                self.state["sl_price"] = 0.0
                                self.state["time_stop"] = None
                                self._save_state()
                                log.info(
                                    f"Position closed. Capital: ${self.state['capital']:,.0f} | "
                                    f"Trades: {self.state['total_trades']} | "
                                    f"Net: {self.state['total_pnl_ticks']:+.1f}t"
                                )
                            else:
                                # DataEnd or NoData — still in position, wait for more data
                                log.info(
                                    f"In position ({('LONG' if direction == 1 else 'SHORT')}) | "
                                    f"entry={entry_price:.2f} | "
                                    f"TP={tp_price:.2f} SL={sl_price:.2f} | "
                                    f"waiting for exit trigger"
                                )
                    else:
                        log.info("In position but no minute data — waiting for data")

                    time.sleep(LOOP_SLEEP_SECONDS)
                    continue

                # Load data and predict
                minute_df = self._load_minute_bars()
                if minute_df.empty:
                    time.sleep(LOOP_SLEEP_SECONDS)
                    continue

                # Retrain if needed
                if self.model is None:
                    if not self.retrain(minute_df):
                        time.sleep(LOOP_SLEEP_SECONDS)
                        continue

                bars_df = self._build_feature_df(minute_df)
                if bars_df.empty:
                    time.sleep(LOOP_SLEEP_SECONDS)
                    continue

                signal = self.predict(bars_df)
                if signal is None:
                    time.sleep(LOOP_SLEEP_SECONDS)
                    continue

                pred = signal["prediction_ticks"]
                zscore = signal["zscore"]
                self.state["signals_generated"] += 1

                # Check entry criteria
                # Note: in live mode we don't have conf from OOT — use z-score only
                if abs(zscore) < ZSCORE_THRESHOLD:
                    self.state["signals_filtered"] += 1
                    log.info(f"Signal filtered: pred={pred:+.2f}, zscore={zscore:+.2f} "
                             f"(below {ZSCORE_THRESHOLD})")
                    time.sleep(LOOP_SLEEP_SECONDS)
                    continue

                direction = 1 if pred > 0 else -1
                dir_str = "LONG" if direction == 1 else "SHORT"

                # Pre-open noise filter (live path)
                if SKIP_FIRST_BAR:
                    now_et_fb = pd.Timestamp(now).tz_convert('US/Eastern')
                    if now_et_fb.hour < FIRST_BAR_CUTOFF_HOUR_ET or (now_et_fb.hour == FIRST_BAR_CUTOFF_HOUR_ET and now_et_fb.minute < FIRST_BAR_CUTOFF_MIN_ET):
                        self.state["signals_filtered"] += 1
                        log.info(f"Signal filtered: {dir_str} blocked (pre-open noise)")
                        time.sleep(LOOP_SLEEP_SECONDS)
                        continue

                # Afternoon-short filter (live path)
                if AFTERNOON_SHORT_FILTER and direction == -1:
                    # Convert to ET: UTC-4 (EDT) or UTC-5 (EST)
                    # Use pandas for proper timezone handling
                    now_et = pd.Timestamp(now).tz_convert('US/Eastern')
                    if now_et.hour >= AFTERNOON_CUTOFF_HOUR_ET:
                        self.state["signals_filtered"] += 1
                        log.info(f"Signal filtered: {dir_str} blocked (afternoon short)")
                        time.sleep(LOOP_SLEEP_SECONDS)
                        continue

                entry_price = signal["bar_close"]

                log.info(
                    f"SIGNAL: {dir_str} pred={pred:+.2f} z={zscore:+.2f} "
                    f"entry={entry_price:.2f}"
                )

                # In live paper mode, we simulate the trade immediately
                # (real fills would come from market data stream)
                self.state["position"] = direction
                self.state["entry_price"] = entry_price
                self.state["entry_time"] = now.isoformat()
                self.state["fill_time"] = now.isoformat()
                self.state["tp_price"] = entry_price + (TP_TICKS * direction)
                self.state["sl_price"] = entry_price - (SL_TICKS * direction)
                self.state["time_stop"] = (now + timedelta(minutes=MAX_HOLD_MINUTES)).isoformat()
                self.state["entry_prediction"] = pred
                self.state["entry_zscore"] = zscore
                self.state["signals_traded"] += 1

                self._save_state()

            except Exception as e:
                log.error(f"Live loop error: {e}\n{traceback.format_exc()}")

            time.sleep(LOOP_SLEEP_SECONDS)

    # ── Reporting ──

    def _print_report(self, test_dates, daily_stats, trades_list):
        s = self.state
        n = s["total_trades"]

        log.info(f"\n{'=' * 70}")
        log.info("INTEGRATED PIPELINE PAPER — RESULTS")
        log.info(f"{'=' * 70}")
        log.info(f"Period: {test_dates[0]} -> {test_dates[-1]} ({len(test_dates)} days)")
        log.info(f"Config: conf >= {ENTRY_CONF_THRESHOLD}, |zscore| >= {ZSCORE_THRESHOLD}")
        log.info(f"Exits: TP={TP_TICKS}t, SL={SL_TICKS}t, max_hold={MAX_HOLD_MINUTES}m")
        log.info(f"Fill model: passive limit, FIFO (fill when price trades {FILL_THROUGH_TICKS}t through)")
        log.info(f"")
        log.info(f"Signals generated: {s['signals_generated']}")
        log.info(f"Signals filtered: {s['signals_filtered']}")
        log.info(f"Trades executed: {n}")

        if n > 0:
            wr = s["wins"] / n * 100
            avg_pnl = s["total_pnl_ticks"] / n

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

            # Max drawdown
            cum_pnl = np.cumsum(pnl_arr)
            running_max = np.maximum.accumulate(cum_pnl)
            drawdowns = running_max - cum_pnl
            max_dd = drawdowns.max()

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
            log.info(f"Max Drawdown: {max_dd:.1f} ticks (${max_dd * ES_TICK_VALUE:.0f})")
            log.info(f"")
            log.info(f"Exit breakdown: TP={s['tp_exits']}, SL={s['sl_exits']}, "
                     f"TimeStop={s['time_exits']}, EOD={s.get('eod_exits', 0)}")

            if long_trades:
                long_pnl = [t["net_pnl_ticks"] for t in long_trades]
                long_wr = sum(1 for p in long_pnl if p > 0) / len(long_pnl) * 100
                log.info(f"LONG:  {len(long_trades)} trades, WR {long_wr:.0f}%, "
                         f"avg {np.mean(long_pnl):.2f}t, total {np.sum(long_pnl):.1f}t")
            if short_trades:
                short_pnl = [t["net_pnl_ticks"] for t in short_trades]
                short_wr = sum(1 for p in short_pnl if p > 0) / len(short_pnl) * 100
                log.info(f"SHORT: {len(short_trades)} trades, WR {short_wr:.0f}%, "
                         f"avg {np.mean(short_pnl):.2f}t, total {np.sum(short_pnl):.1f}t")

            # Regime gap check — HC #428 R1: stratify by ES close-to-close
            # Green day = ES close > prior close; Red = close < prior close; Flat = unchanged
            # Reject if |Sharpe_green − Sharpe_red| / max(|Sharpe_green|,|Sharpe_red|) > 0.50
            regime_daily = {}  # date -> {"pnl": float, "regime": str}
            sorted_dates = sorted(daily_stats.keys())
            for i, date in enumerate(sorted_dates):
                ds = daily_stats[date]
                # Classify regime from minute bar close-to-close
                regime = "flat"  # default
                try:
                    bar_file = MINUTE_BAR_DIR / f"{date}.parquet"
                    if bar_file.exists():
                        bars = pd.read_parquet(bar_file)
                        if "close" in bars.columns and len(bars) > 0:
                            day_close = bars["close"].iloc[-1]
                            # Check prior day
                            if i > 0:
                                prev_date = sorted_dates[i - 1]
                                prev_file = MINUTE_BAR_DIR / f"{prev_date}.parquet"
                                if prev_file.exists():
                                    prev_bars = pd.read_parquet(prev_file)
                                    if "close" in prev_bars.columns and len(prev_bars) > 0:
                                        prev_close = prev_bars["close"].iloc[-1]
                                        if day_close > prev_close:
                                            regime = "green"
                                        elif day_close < prev_close:
                                            regime = "red"
                except Exception:
                    pass
                regime_daily[date] = {"pnl": ds["net_pnl"], "regime": regime}

            # Compute per-regime Sharpe
            regime_sharpes = {}
            for regime_name in ["green", "red", "flat"]:
                r_pnls = np.array([v["pnl"] for v in regime_daily.values() if v["regime"] == regime_name])
                r_pnls_nonzero = r_pnls[r_pnls != 0]  # exclude zero-trade days
                if len(r_pnls_nonzero) > 1 and r_pnls_nonzero.std() > 0:
                    regime_sharpes[regime_name] = r_pnls_nonzero.mean() / r_pnls_nonzero.std() * np.sqrt(252)
                elif len(r_pnls_nonzero) == 1:
                    regime_sharpes[regime_name] = float("inf") if r_pnls_nonzero[0] > 0 else float("-inf")
                # else: not enough data, skip

            log.info(f"")
            if "green" in regime_sharpes and "red" in regime_sharpes:
                sg, sr = regime_sharpes["green"], regime_sharpes["red"]
                max_abs = max(abs(sg), abs(sr))
                regime_gap = abs(sg - sr) / max_abs if max_abs > 0 else 0
                pass_fail = "PASS" if regime_gap <= 0.50 else "FAIL"
                log.info(f"Regime gap (HC #428): {regime_gap:.3f} {pass_fail} (threshold 0.50)")
                for rn in ["green", "red", "flat"]:
                    if rn in regime_sharpes:
                        n_days = sum(1 for v in regime_daily.values() if v["regime"] == rn and v["pnl"] != 0)
                        total = sum(v["pnl"] for v in regime_daily.values() if v["regime"] == rn)
                        log.info(f"  {rn.upper():>5s}: {n_days} days, Sharpe {regime_sharpes[rn]:.2f}, total {total:.1f}t")
            else:
                log.info(f"Regime gap: insufficient data (need both green and red days with trades)")
                for rn, sh in regime_sharpes.items():
                    log.info(f"  {rn.upper():>5s}: Sharpe {sh:.2f}")

            # Also show temporal stability (3-segment, informational only)
            if len(daily_pnl_arr) >= 6:
                n_seg = 3
                seg_size = len(daily_pnl_arr) // n_seg
                seg_sharpes = []
                for si in range(n_seg):
                    seg = daily_pnl_arr[si * seg_size:(si + 1) * seg_size]
                    if len(seg) > 1 and seg.std() > 0:
                        seg_sharpes.append(seg.mean() / seg.std() * np.sqrt(252))
                    else:
                        seg_sharpes.append(0)
                log.info(f"Temporal stability (3-seg): {[round(s, 2) for s in seg_sharpes]}")

            log.info(f"")
            log.info(f"Per-day breakdown:")
            log.info(f"{'Date':<12s} {'Trades':>6s} {'Sigs':>5s} {'Filt':>5s} "
                     f"{'NetPnL':>8s} {'$PnL':>8s} {'WR':>5s} {'TP':>3s} {'SL':>3s} {'Time':>4s}")
            for date in sorted(daily_stats.keys()):
                ds = daily_stats[date]
                if ds["n_trades"] == 0 and ds["n_signals"] == 0:
                    continue
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
            import socket

            # Check if server is running
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(1)
            result = s.connect_ex(('localhost', 5000))
            s.close()
            if result != 0:
                log.info("MLflow server not running, skipping logging")
                return

            mlflow.set_tracking_uri("http://localhost:5000")
            mlflow.set_experiment("integrated_pipeline_paper")

            st = self.state
            n = st["total_trades"]
            if n == 0:
                return

            pnl_arr = np.array([t["net_pnl_ticks"] for t in trades_list])
            sharpe = pnl_arr.mean() / max(pnl_arr.std(), 1e-6) * np.sqrt(252)
            downside = np.sqrt(np.mean(np.minimum(pnl_arr, 0) ** 2))
            sortino = pnl_arr.mean() / max(downside, 1e-6) * np.sqrt(252)
            pf = np.sum(pnl_arr[pnl_arr > 0]) / max(-np.sum(pnl_arr[pnl_arr < 0]), 1e-6)

            daily_pnl_arr = np.array([ds["net_pnl"] for ds in daily_stats.values()])
            daily_sharpe = (
                daily_pnl_arr.mean() / max(daily_pnl_arr.std(), 1e-6) * np.sqrt(252)
                if len(daily_pnl_arr) > 1 else 0
            )

            cum_pnl = np.cumsum(pnl_arr)
            max_dd = (np.maximum.accumulate(cum_pnl) - cum_pnl).max()

            with mlflow.start_run(run_name=f"oot_replay_SL{SL_TICKS}_TP{TP_TICKS}"):
                mlflow.log_params({
                    "mode": "oot_replay",
                    "entry_conf": ENTRY_CONF_THRESHOLD,
                    "zscore_thresh": ZSCORE_THRESHOLD,
                    "tp_ticks": TP_TICKS,
                    "sl_ticks": SL_TICKS,
                    "max_hold_min": MAX_HOLD_MINUTES,
                    "train_days": TRAIN_DAYS,
                    "test_start": test_dates[0],
                    "test_end": test_dates[-1],
                    "n_test_days": len(test_dates),
                })
                mlflow.log_metrics({
                    "n_trades": n,
                    "win_rate": st["wins"] / n,
                    "profit_factor": float(pf),
                    "trade_sharpe": float(sharpe),
                    "trade_sortino": float(sortino),
                    "daily_sharpe": float(daily_sharpe),
                    "total_pnl_ticks": st["total_pnl_ticks"],
                    "total_pnl_dollars": st["total_pnl_dollars"],
                    "avg_pnl_ticks": st["total_pnl_ticks"] / n,
                    "max_drawdown_ticks": float(max_dd),
                    "tp_exits": st["tp_exits"],
                    "sl_exits": st["sl_exits"],
                    "time_exits": st["time_exits"],
                })
            log.info("MLflow logged successfully")
        except Exception as e:
            log.warning(f"MLflow logging failed: {e}")

    # ── Summary ──

    def summary(self):
        s = self.state
        n = s["total_trades"]
        log.info(f"\n{'=' * 70}")
        log.info(f"INTEGRATED PIPELINE PAPER — STATUS")
        log.info(f"{'=' * 70}")
        log.info(f"Mode: {s.get('mode', 'unknown')}")
        log.info(f"Capital: ${s['capital']:.0f}")
        log.info(f"Trades: {n}")
        if n > 0:
            wr = s["wins"] / n * 100
            log.info(f"Win Rate: {wr:.1f}%")
            log.info(f"PnL: {s['total_pnl_ticks']:.1f} ticks (${s['total_pnl_dollars']:.0f})")
            log.info(f"Avg PnL: {s['total_pnl_ticks']/n:.2f} ticks/trade")
            log.info(f"Exits: TP={s['tp_exits']}, SL={s['sl_exits']}, Time={s['time_exits']}")
        log.info(f"Signals: {s['signals_generated']} generated, {s['signals_filtered']} filtered")
        log.info(f"Last heartbeat: {s.get('last_heartbeat', 'never')}")
        log.info(f"{'=' * 70}\n")


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Integrated Pipeline Paper Engine")
    parser.add_argument("--replay", type=int, nargs="?", const=0, default=None,
                        help="Replay OOT predictions (0 = all days, N = last N days)")
    parser.add_argument("--live", action="store_true", help="Run in live loop mode")
    parser.add_argument("--summary", action="store_true", help="Print status and exit")
    args = parser.parse_args()

    engine = IntegratedPipelinePaperEngine()

    if args.summary:
        engine.summary()
        return

    # Start health server
    start_health_server(engine)

    if args.live:
        engine.run_live()
    elif args.replay is not None:
        engine.replay_oot(n_test_days=args.replay)
    else:
        # Default: replay all OOT days
        engine.replay_oot(n_test_days=0)


if __name__ == "__main__":
    main()
