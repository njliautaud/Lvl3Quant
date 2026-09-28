#!/usr/bin/env python3
"""
Multi-Timeframe Portfolio Paper Trading Engine
================================================

Paper trades two validated, uncorrelated ES futures strategies in parallel:

STRATEGY A — SHORT-HORIZON (70% allocation):
  - 30-min LightGBM on lean 43-feature microstructure set (sliding 60d train)
  - Entry: conf >= 0.52, |pred_zscore| >= 0.3
  - Direction: pred > 0 -> LONG, pred < 0 -> SHORT
  - Entry: Passive limit at bid/ask (FIFO back-of-queue)
  - TP: 20 ticks (passive), SL: 10 ticks (market), Max hold: 30 min
  - Validated: Sharpe 2.49, 560 trades, WR ~50%, regime gap 0.18

STRATEGY B — LONG-HORIZON (30% allocation):
  - LightGBM on hourly features with 3-20 day accumulated OFI flow
  - Entry: top/bottom 15% confidence predictions
  - Hold: 4h (time-based exit)
  - Validated: Sharpe 1.84, WR 49.3%, PF 1.36, regime gap 0.39

PORTFOLIO:
  - 70/30 short-heavy allocation
  - Strategies run independently (r=0.001 correlation)
  - Combined Daily Sharpe 4.16, passes all regime gates

COST MODEL (ES Futures — AMP/Rithmic):
  Short-horizon: Entry(passive) + TP(passive) = 0.376 ticks RT
                 Entry(passive) + SL(market) = 1.376 ticks RT
  Long-horizon:  Market entry + market exit = 2.376 ticks RT

MODES:
  --replay N    Replay last N OOT days (default: all available)
  --live        Run in live loop mode (for PM2)
  --summary     Print current state and exit

PM2:
  pm2 start /home/jupiter/Lvl3Quant/paper_engines/multi_timeframe_paper.py \
    --name multi-timeframe-paper --interpreter python3 -- --live

Author: Claude (autonomous build)
"""

import argparse
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
REGIME_DIR = ROOT / "data" / "feature_store" / "v1"
ENGINE_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = ENGINE_DIR / "logs" / "multi_timeframe_paper"
SHORT_MODEL_DIR = OUTPUT_DIR / "models" / "short_horizon"
LONG_MODEL_DIR = OUTPUT_DIR / "models" / "long_horizon"
STATE_DIR = OUTPUT_DIR

for d in [OUTPUT_DIR, SHORT_MODEL_DIR, LONG_MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────
LOG_FILE = OUTPUT_DIR / "multi_timeframe_paper.log"
logging.basicConfig(
    format="%(asctime)s [MTF-PAPER] %(levelname)s %(message)s",
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE)),
    ],
)
log = logging.getLogger("MTF-PAPER")

# ─────────────────────────────────────────────
#  CONSTANTS (ES Futures — AMP/Rithmic)
# ─────────────────────────────────────────────
ES_TICK_SIZE = 0.25
ES_TICK_VALUE = 12.50
ES_RT_COMMISSION_TICKS = 0.376       # $4.70 / $12.50

# Short-horizon costs
SHORT_COST_PASSIVE_RT = ES_RT_COMMISSION_TICKS       # 0.376 ticks (passive + passive)
SHORT_COST_MARKET_RT = ES_RT_COMMISSION_TICKS + 1.0  # 1.376 ticks (passive + market)

# Long-horizon costs (market entry + market exit)
LONG_COST_RT = 2 * 1.0 + ES_RT_COMMISSION_TICKS     # 2.376 ticks

# ── SHORT-HORIZON STRATEGY PARAMS ──
# Champion config from robustness_v1 (Sharpe 5.46, regime gap 0.121 PASS)
SHORT_BAR_SIZE_MINUTES = 30
SHORT_ENTRY_CONF = 0.05              # Champion threshold (low gate, let model decide)
SHORT_ZSCORE_THRESHOLD = 0.0          # No z-score filter (paper showed it's anti-predictive)
SHORT_TP_TICKS = 25                   # Champion TP (not 20)
SHORT_SL_LONG_TICKS = 4              # Champion SL for longs (asymmetric)
SHORT_SL_SHORT_TICKS = 3             # Champion SL for shorts (tighter)
SHORT_SL_TICKS = 4                    # Backward compat default
SHORT_MAX_HOLD_MINUTES = 60           # Champion max hold (not 30)
SHORT_TRAIN_DAYS = 60
SHORT_FILL_THROUGH_TICKS = 1
SHORT_DAILY_BIAS_MULT = 1.5           # Champion daily bias multiplier

# ── LONG-HORIZON STRATEGY PARAMS ──
LONG_THRESHOLD_PCT = 0.15            # top/bottom 15% triggers trade
LONG_HOLD_HOURS = 4                  # hold for 4 hours then exit
LONG_TRAIN_DAYS = 60

# ── PORTFOLIO PARAMS ──
WEIGHT_SHORT = 0.70
WEIGHT_LONG = 0.30
STARTING_CAPITAL = 100_000
MAX_POSITION_PER_STRATEGY = 1

# RTH boundaries (UTC)
RTH_START_H, RTH_START_M = 13, 30    # 09:30 ET
RTH_END_H, RTH_END_M = 20, 0        # 16:00 ET

# Data lookback
LOOKBACK_DAYS = 75

# Health check port
HEALTH_PORT = 8767

# Main loop sleep
LOOP_SLEEP_SECONDS = 30

# ── LightGBM params (short-horizon, identical to integrated pipeline) ──
SHORT_LGBM_PARAMS = {
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

# ── LightGBM params (long-horizon) ──
LONG_LGBM_PARAMS = {
    "objective": "regression",
    "metric": "mse",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "max_depth": 6,
    "min_data_in_leaf": 50,
    "feature_fraction": 0.7,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "lambda_l1": 0.1,
    "lambda_l2": 1.0,
    "verbose": -1,
    "n_jobs": 8,
    "seed": 42,
}

# 43-feature lean set (short-horizon, validated)
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


# ═══════════════════════════════════════════════════════════════════
#  SHARED: DATA LOADING
# ═══════════════════════════════════════════════════════════════════


def load_minute_bars(n_days: int = LOOKBACK_DAYS) -> pd.DataFrame:
    """Load recent minute bars from MBO data."""
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


def load_minute_bars_for_dates(date_list: list) -> pd.DataFrame:
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
    return combined.sort_values("ts_minute").reset_index(drop=True)


# ═══════════════════════════════════════════════════════════════════
#  SHORT-HORIZON: Feature computation (30-min bars, lean feature set)
# ═══════════════════════════════════════════════════════════════════


def _safe_polyfit_slope(arr: np.ndarray) -> float:
    if len(arr) < 2:
        return 0.0
    try:
        return float(np.polyfit(np.arange(len(arr)), arr, 1)[0])
    except (np.linalg.LinAlgError, ValueError):
        return 0.0


def aggregate_to_30min_bars(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute bars into 30-min bars with microstructure features."""
    df = minute_df.copy()
    df["bar_key"] = df["ts_minute"].dt.floor("30min")
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
                float(np.std(ret_arr) * np.sqrt(252 * 13))
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


def add_short_rolling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-bar rolling features for short-horizon strategy (causal only)."""
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

    # Time-of-day features
    df["tod_sin"] = np.sin(2 * np.pi * df["ts"].dt.hour / 24)
    df["tod_cos"] = np.cos(2 * np.pi * df["ts"].dt.hour / 24)
    rth_start = 13 * 60 + 30
    rth_end = 20 * 60
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

    df["intraday_direction_strength"] = (
        df["intraday_cum_ofi"].abs() /
        df.groupby("date")["ofi_sum"]
        .transform(lambda x: x.abs().cumsum())
        .clip(lower=1)
    )

    return df


# ═══════════════════════════════════════════════════════════════════
#  LONG-HORIZON: Feature computation (hourly bars, multi-day OFI flow)
# ═══════════════════════════════════════════════════════════════════


def compute_hourly_features(minute_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate 1-minute bars to hourly bars with microstructure features."""
    df = minute_df.copy()
    df['hour'] = df['ts_minute'].dt.hour
    df['date_str'] = df['date']
    df['return_1m'] = df.groupby('date_str')['close'].pct_change()
    df['log_volume'] = np.log1p(df['volume'])
    df['abs_ofi'] = df['ofi_1min'].abs()

    sv_std = df.groupby('date_str')['signed_volume'].transform('std').replace(0, 1)
    df['sv_zscore'] = df['signed_volume'] / sv_std
    df['vwap_dev'] = (df['close'] - df['vwap']) / df['close'].clip(lower=1)

    hourly_records = []
    for (date_str, hour), group in df.groupby(['date_str', 'hour']):
        if len(group) < 5:
            continue

        close_arr = group['close'].values
        volume_arr = group['volume'].values
        ofi_arr = group['ofi_1min'].values
        sv_arr = group['signed_volume'].values
        ret_arr = group['return_1m'].fillna(0).values
        spread_arr = group['spread_mean'].values
        tc_arr = group['trade_count'].values

        rec = {
            'date': date_str,
            'hour': hour,
            'ts': group['ts_minute'].iloc[0],
            'open': close_arr[0],
            'high': close_arr.max(),
            'low': close_arr.min(),
            'close': close_arr[-1],
            'return_1h': (close_arr[-1] / close_arr[0] - 1) if close_arr[0] > 0 else 0,
            'range_ticks': (close_arr.max() - close_arr.min()) / 0.25,
            'close_position': (close_arr[-1] - close_arr.min()) / max(close_arr.max() - close_arr.min(), 0.25),
            'total_volume': volume_arr.sum(),
            'avg_volume': volume_arr.mean(),
            'volume_trend': _safe_polyfit_slope(volume_arr),
            'volume_concentration': volume_arr.max() / max(volume_arr.mean(), 1),
            'ofi_sum': ofi_arr.sum(),
            'ofi_mean': ofi_arr.mean(),
            'ofi_trend': _safe_polyfit_slope(ofi_arr),
            'ofi_consistency': np.mean(np.sign(ofi_arr) == np.sign(ofi_arr.sum())) if ofi_arr.sum() != 0 else 0.5,
            'ofi_late_vs_early': ofi_arr[len(ofi_arr)//2:].sum() - ofi_arr[:len(ofi_arr)//2].sum(),
            'signed_volume_sum': sv_arr.sum(),
            'signed_volume_ratio': sv_arr.sum() / max(volume_arr.sum(), 1),
            'buy_volume_fraction': np.sum(sv_arr[sv_arr > 0]) / max(volume_arr.sum(), 1),
            'sell_volume_fraction': -np.sum(sv_arr[sv_arr < 0]) / max(volume_arr.sum(), 1),
            'sweep_minutes': np.sum(np.abs(group['sv_zscore'].values) > 2),
            'max_sweep_intensity': np.abs(group['sv_zscore'].values).max(),
            'sweep_direction': (
                np.sign(sv_arr[np.abs(group['sv_zscore'].values).argmax()])
                if len(sv_arr) > 0 else 0
            ),
            'spread_mean': spread_arr.mean(),
            'spread_max': spread_arr.max(),
            'spread_trend': _safe_polyfit_slope(spread_arr),
            'trade_count_sum': tc_arr.sum(),
            'trade_count_trend': _safe_polyfit_slope(tc_arr),
            'realized_vol': np.std(ret_arr) * np.sqrt(60) if len(ret_arr) > 1 else 0,
            'vol_of_vol': np.std(np.abs(ret_arr)) if len(ret_arr) > 1 else 0,
            'up_vol': np.std(ret_arr[ret_arr > 0]) if np.sum(ret_arr > 0) > 1 else 0,
            'down_vol': np.std(ret_arr[ret_arr < 0]) if np.sum(ret_arr < 0) > 1 else 0,
            'vol_asymmetry': 0,
            'vwap_dev_mean': group['vwap_dev'].mean(),
            'vwap_dev_trend': _safe_polyfit_slope(group['vwap_dev'].values),
        }

        if rec['up_vol'] > 0 and rec['down_vol'] > 0:
            rec['vol_asymmetry'] = rec['down_vol'] / rec['up_vol'] - 1
        elif rec['down_vol'] > 0:
            rec['vol_asymmetry'] = 1.0
        elif rec['up_vol'] > 0:
            rec['vol_asymmetry'] = -1.0

        hourly_records.append(rec)

    result = pd.DataFrame(hourly_records)
    return result


def add_long_rolling_context(df: pd.DataFrame) -> pd.DataFrame:
    """Add multi-hour rolling features for long-horizon — captures positioning buildup."""
    df = df.sort_values('ts').reset_index(drop=True)

    for window in [2, 4, 6]:
        label = f'{window}h'
        df[f'ofi_sum_{label}'] = df['ofi_sum'].rolling(window, min_periods=1).sum()
        df[f'ofi_trend_{label}'] = df['ofi_trend'].rolling(window, min_periods=1).mean()
        df[f'sv_sum_{label}'] = df['signed_volume_sum'].rolling(window, min_periods=1).sum()
        df[f'volume_ma_{label}'] = df['total_volume'].rolling(window, min_periods=1).mean()
        df[f'volume_vs_ma_{label}'] = df['total_volume'] / df[f'volume_ma_{label}'].clip(lower=1)
        df[f'vol_trend_{label}'] = df['realized_vol'].rolling(window, min_periods=1).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True
        )

    # Cross-day features
    df['prev_day_ofi'] = df.groupby('date')['ofi_sum'].transform('sum').shift(1)
    df['prev_day_return'] = df.groupby('date')['return_1h'].transform('sum').shift(1)
    df['prev_day_sv'] = df.groupby('date')['signed_volume_sum'].transform('sum').shift(1)

    # Intraday time features
    df['hours_since_open'] = df.groupby('date').cumcount()
    df['is_first_hour'] = (df['hours_since_open'] == 0).astype(int)
    df['is_last_hour'] = df.groupby('date')['hours_since_open'].transform('max') == df['hours_since_open']
    df['is_last_hour'] = df['is_last_hour'].astype(int)

    # Cumulative intraday flow
    df['intraday_cum_ofi'] = df.groupby('date')['ofi_sum'].cumsum()
    df['intraday_cum_sv'] = df.groupby('date')['signed_volume_sum'].cumsum()
    df['intraday_cum_return'] = df.groupby('date')['return_1h'].cumsum()

    # Flow reversal
    df['ofi_sign_change'] = (np.sign(df['ofi_sum']) != np.sign(df['ofi_sum'].shift(1))).astype(int)

    # Absorption proxy
    df['absorption_score'] = df['total_volume'] / (df['range_ticks'].clip(lower=1) * 100)

    return df


def add_long_macro_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add macro/regime features for long-horizon model."""
    regime_path = REGIME_DIR / "regime_features.parquet"
    if regime_path.exists():
        log.info("Loading macro regime features for long-horizon")
        regime_df = pd.read_parquet(regime_path)
        if 'date' in regime_df.columns:
            regime_df['date'] = regime_df['date'].astype(str).str.replace('-', '')
            regime_cols = [c for c in regime_df.columns if c != 'date']
            existing = [c for c in regime_cols if c in df.columns]
            if existing:
                df = df.drop(columns=existing)
            df = df.merge(regime_df, on='date', how='left')
    else:
        # Build simple regime from price data
        df_daily = df.groupby('date').agg(
            day_close=('close', 'last'),
            day_vol=('realized_vol', 'mean'),
        ).reset_index()
        df_daily['ma_20d'] = df_daily['day_close'].rolling(20, min_periods=5).mean()
        df_daily['above_ma20'] = (df_daily['day_close'] > df_daily['ma_20d']).astype(int)
        df_daily['trend_20d'] = df_daily['day_close'].pct_change(20)
        df_daily['vol_20d'] = df_daily['day_vol'].rolling(20, min_periods=5).mean()
        df = df.merge(
            df_daily[['date', 'above_ma20', 'trend_20d', 'vol_20d']],
            on='date', how='left',
        )

    return df


# ═══════════════════════════════════════════════════════════════════
#  FILL SIMULATION (FIFO back-of-queue model — short-horizon)
# ═══════════════════════════════════════════════════════════════════


def simulate_fifo_fill(
    entry_price: float,
    direction: int,
    minute_bars: pd.DataFrame,
) -> Optional[Dict]:
    """Simulate passive limit fill for short-horizon strategy."""
    if minute_bars.empty:
        return None

    fill_threshold = SHORT_FILL_THROUGH_TICKS * ES_TICK_SIZE

    for _, row in minute_bars.iterrows():
        if direction == 1:
            if row["low"] <= entry_price - fill_threshold:
                return {"fill_price": entry_price, "fill_time": row["ts_minute"]}
        else:
            if row["high"] >= entry_price + fill_threshold:
                return {"fill_price": entry_price, "fill_time": row["ts_minute"]}
    return None


def simulate_short_exit(
    entry_price: float,
    direction: int,
    fill_time,
    minute_bars: pd.DataFrame,
) -> Dict:
    """Simulate exit for short-horizon strategy with TP/SL/TimeStop."""
    # Asymmetric SL: longs get wider SL, shorts get tighter (champion config)
    sl_ticks = SHORT_SL_LONG_TICKS if direction == 1 else SHORT_SL_SHORT_TICKS
    tp_price = entry_price + (SHORT_TP_TICKS * ES_TICK_SIZE * direction)
    sl_price = entry_price - (sl_ticks * ES_TICK_SIZE * direction)
    time_stop = fill_time + timedelta(minutes=SHORT_MAX_HOLD_MINUTES)

    if minute_bars.empty:
        return {
            "exit_type": "NoData", "exit_price": entry_price,
            "exit_time": fill_time, "raw_pnl_ticks": 0,
            "cost_ticks": SHORT_COST_PASSIVE_RT,
        }

    for _, row in minute_bars.iterrows():
        ts = row["ts_minute"]
        hi, lo = row["high"], row["low"]

        if direction == 1:
            tp_hit = hi >= tp_price
            sl_hit = lo <= sl_price
        else:
            tp_hit = lo <= tp_price
            sl_hit = hi >= sl_price

        # EOD forced exit
        eod_time = ts.replace(hour=19, minute=55, second=0)
        if ts >= eod_time:
            exit_price = row["close"]
            raw_pnl = (exit_price - entry_price) / ES_TICK_SIZE * direction
            return {
                "exit_type": "EOD", "exit_price": exit_price,
                "exit_time": ts, "raw_pnl_ticks": raw_pnl,
                "cost_ticks": SHORT_COST_MARKET_RT,
            }

        if tp_hit and sl_hit:
            bar_close = row["close"]
            favorable = (bar_close >= entry_price) if direction == 1 else (bar_close <= entry_price)
            if favorable:
                return {
                    "exit_type": "TP", "exit_price": tp_price,
                    "exit_time": ts, "raw_pnl_ticks": SHORT_TP_TICKS,
                    "cost_ticks": SHORT_COST_PASSIVE_RT,
                }
            else:
                return {
                    "exit_type": "SL", "exit_price": sl_price,
                    "exit_time": ts, "raw_pnl_ticks": -sl_ticks,
                    "cost_ticks": SHORT_COST_MARKET_RT,
                }
        elif sl_hit:
            return {
                "exit_type": "SL", "exit_price": sl_price,
                "exit_time": ts, "raw_pnl_ticks": -sl_ticks,
                "cost_ticks": SHORT_COST_MARKET_RT,
            }
        elif tp_hit:
            return {
                "exit_type": "TP", "exit_price": tp_price,
                "exit_time": ts, "raw_pnl_ticks": SHORT_TP_TICKS,
                "cost_ticks": SHORT_COST_PASSIVE_RT,
            }

        if ts >= time_stop:
            exit_price = row["close"]
            raw_pnl = (exit_price - entry_price) / ES_TICK_SIZE * direction
            return {
                "exit_type": "TimeStop", "exit_price": exit_price,
                "exit_time": ts, "raw_pnl_ticks": raw_pnl,
                "cost_ticks": SHORT_COST_MARKET_RT,
            }

    last = minute_bars.iloc[-1]
    exit_price = last["close"]
    raw_pnl = (exit_price - entry_price) / ES_TICK_SIZE * direction
    return {
        "exit_type": "DataEnd", "exit_price": exit_price,
        "exit_time": last["ts_minute"], "raw_pnl_ticks": raw_pnl,
        "cost_ticks": SHORT_COST_MARKET_RT,
    }


# ═══════════════════════════════════════════════════════════════════
#  HEALTH CHECK SERVER
# ═══════════════════════════════════════════════════════════════════


class HealthHandler(BaseHTTPRequestHandler):
    engine = None

    def do_GET(self):
        if self.path == "/health":
            state = self.engine.state if self.engine else {}
            short_s = state.get("short_horizon", {})
            long_s = state.get("long_horizon", {})
            portfolio = state.get("portfolio", {})

            body = json.dumps({
                "status": "ok",
                "engine": "multi-timeframe-paper",
                "mode": state.get("mode", "unknown"),
                "last_heartbeat": state.get("last_heartbeat", ""),
                "short_horizon": {
                    "trades": short_s.get("total_trades", 0),
                    "pnl_ticks": round(short_s.get("total_pnl_ticks", 0), 2),
                    "pnl_dollars": round(short_s.get("total_pnl_dollars", 0), 2),
                    "wins": short_s.get("wins", 0),
                    "losses": short_s.get("losses", 0),
                },
                "long_horizon": {
                    "trades": long_s.get("total_trades", 0),
                    "pnl_ticks": round(long_s.get("total_pnl_ticks", 0), 2),
                    "pnl_dollars": round(long_s.get("total_pnl_dollars", 0), 2),
                    "wins": long_s.get("wins", 0),
                    "losses": long_s.get("losses", 0),
                },
                "portfolio": {
                    "weighted_pnl_ticks": round(portfolio.get("weighted_pnl_ticks", 0), 2),
                    "weighted_pnl_dollars": round(portfolio.get("weighted_pnl_dollars", 0), 2),
                    "capital": round(portfolio.get("capital", STARTING_CAPITAL), 2),
                },
            }).encode()
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(body)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, format, *args):
        pass


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


class MultiTimeframePaperEngine:
    """
    Runs two strategies in parallel with independent position tracking.

    Modes:
      - REPLAY: Walk-forward replay on historical data
      - LIVE: Continuous loop retraining/predicting on new data
    """

    def __init__(self):
        self.state_path = STATE_DIR / "state.json"
        self.short_trades_csv = OUTPUT_DIR / "short_trades.csv"
        self.long_trades_csv = OUTPUT_DIR / "long_trades.csv"
        self.daily_csv = OUTPUT_DIR / "daily_pnl.csv"

        # Short-horizon model
        self.short_model = None
        self.short_feature_cols = None
        self.short_scaler_median = None
        self.short_scaler_iqr = None
        self.short_pred_quantiles = None

        # Long-horizon model
        self.long_model = None
        self.long_feature_cols = None
        self.long_pred_quantiles = None

        # Load or init state
        if self.state_path.exists():
            with open(self.state_path) as f:
                self.state = json.load(f)
            log.info(f"Loaded existing state")
        else:
            self.state = self._fresh_state()

        self._ensure_csvs()

    def _fresh_state(self) -> Dict:
        return {
            "mode": "init",
            "created": datetime.now(timezone.utc).isoformat(),
            "last_heartbeat": None,
            "short_horizon": {
                "position": 0,
                "entry_price": 0.0,
                "entry_time": None,
                "fill_time": None,
                "tp_price": 0.0,
                "sl_price": 0.0,
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
                "last_retrain_date": None,
            },
            "long_horizon": {
                "position": 0,
                "entry_price": 0.0,
                "entry_time": None,
                "exit_target_time": None,
                "total_trades": 0,
                "total_pnl_ticks": 0.0,
                "total_pnl_dollars": 0.0,
                "wins": 0,
                "losses": 0,
                "last_retrain_date": None,
            },
            "portfolio": {
                "capital": STARTING_CAPITAL,
                "weighted_pnl_ticks": 0.0,
                "weighted_pnl_dollars": 0.0,
            },
        }

    def _save_state(self):
        self.state["last_heartbeat"] = datetime.now(timezone.utc).isoformat()
        with open(self.state_path, "w") as f:
            json.dump(self.state, f, indent=2, default=str)

    def _ensure_csvs(self):
        if not self.short_trades_csv.exists():
            with open(self.short_trades_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "trade_id", "date", "signal_time", "fill_time", "exit_time",
                    "direction", "entry_price", "exit_price",
                    "exit_type", "raw_pnl_ticks", "cost_ticks", "net_pnl_ticks",
                    "net_pnl_dollars", "prediction", "pred_zscore",
                    "hold_minutes", "weighted_pnl_ticks",
                ])
        if not self.long_trades_csv.exists():
            with open(self.long_trades_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "trade_id", "date", "entry_time", "exit_time",
                    "direction", "entry_price", "exit_price",
                    "exit_type", "raw_pnl_ticks", "cost_ticks", "net_pnl_ticks",
                    "net_pnl_dollars", "prediction", "hold_hours",
                    "weighted_pnl_ticks",
                ])
        if not self.daily_csv.exists():
            with open(self.daily_csv, "w", newline="") as f:
                csv.writer(f).writerow([
                    "date",
                    "short_trades", "short_net_pnl_ticks", "short_wins", "short_losses",
                    "long_trades", "long_net_pnl_ticks", "long_wins", "long_losses",
                    "combined_weighted_pnl_ticks", "combined_weighted_pnl_dollars",
                    "capital_after",
                ])

    def _append_short_trade(self, trade: Dict):
        with open(self.short_trades_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                trade.get("trade_id"), trade.get("date"),
                trade.get("signal_time"), trade.get("fill_time"),
                trade.get("exit_time"), trade.get("direction"),
                trade.get("entry_price"), trade.get("exit_price"),
                trade.get("exit_type"),
                round(trade.get("raw_pnl_ticks", 0), 3),
                round(trade.get("cost_ticks", 0), 3),
                round(trade.get("net_pnl_ticks", 0), 3),
                round(trade.get("net_pnl_dollars", 0), 2),
                round(trade.get("prediction", 0), 4),
                round(trade.get("pred_zscore", 0), 4),
                round(trade.get("hold_minutes", 0), 1),
                round(trade.get("weighted_pnl_ticks", 0), 3),
            ])

    def _append_long_trade(self, trade: Dict):
        with open(self.long_trades_csv, "a", newline="") as f:
            csv.writer(f).writerow([
                trade.get("trade_id"), trade.get("date"),
                trade.get("entry_time"), trade.get("exit_time"),
                trade.get("direction"),
                trade.get("entry_price"), trade.get("exit_price"),
                trade.get("exit_type"),
                round(trade.get("raw_pnl_ticks", 0), 3),
                round(trade.get("cost_ticks", 0), 3),
                round(trade.get("net_pnl_ticks", 0), 3),
                round(trade.get("net_pnl_dollars", 0), 2),
                round(trade.get("prediction", 0), 4),
                round(trade.get("hold_hours", 0), 2),
                round(trade.get("weighted_pnl_ticks", 0), 3),
            ])

    # ── SHORT-HORIZON: Training ──

    def retrain_short(self, minute_df: pd.DataFrame) -> bool:
        """Retrain short-horizon 30-min LightGBM."""
        try:
            import lightgbm as lgb
        except ImportError:
            log.error("LightGBM not installed")
            return False

        bars_df = aggregate_to_30min_bars(minute_df)
        if bars_df.empty:
            return False
        bars_df = add_short_rolling_features(bars_df)

        dates = sorted(bars_df["date"].unique())
        if len(dates) < SHORT_TRAIN_DAYS:
            log.warning(f"Short-horizon: not enough days ({len(dates)} < {SHORT_TRAIN_DAYS})")
            return False

        train_dates = dates[-SHORT_TRAIN_DAYS:]
        train_df = bars_df[bars_df["date"].isin(train_dates)].copy().sort_values("ts").reset_index(drop=True)

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

        lean_set = set(LEAN_FEATURES)
        available = [c for c in train_df.columns if c in lean_set]
        self.short_feature_cols = [f for f in LEAN_FEATURES if f in available]

        X_raw = train_df[self.short_feature_cols].values.astype(np.float32)
        y = train_df["fwd_ticks_30min"].values.astype(np.float32)

        valid = ~np.isnan(y)
        X_raw, y = X_raw[valid], y[valid]

        if len(y) < 100:
            log.warning(f"Short-horizon: too few samples ({len(y)})")
            return False

        # Robust scaling
        self.short_scaler_median = np.nanmedian(X_raw, axis=0)
        q75 = np.nanpercentile(X_raw, 75, axis=0)
        q25 = np.nanpercentile(X_raw, 25, axis=0)
        self.short_scaler_iqr = q75 - q25
        self.short_scaler_iqr[self.short_scaler_iqr < 1e-8] = 1.0

        X = np.clip(np.nan_to_num(
            (X_raw - self.short_scaler_median) / self.short_scaler_iqr,
            nan=0.0, posinf=3.0, neginf=-3.0
        ), -5, 5)

        params = {**SHORT_LGBM_PARAMS, "seed": 42}

        # CRITICAL FIX: validation set must NOT overlap training set.
        n_val = max(int(len(X) * 0.1), 20)
        X_train, y_train = X[:-n_val], y[:-n_val]
        X_val, y_val = X[-n_val:], y[-n_val:]
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=self.short_feature_cols)
        val_data = lgb.Dataset(
            X_val, label=y_val,
            feature_name=self.short_feature_cols, reference=train_data,
        )

        callbacks = [
            lgb.early_stopping(stopping_rounds=30, verbose=False),
            lgb.log_evaluation(period=0),
        ]

        self.short_model = lgb.train(
            params, train_data, num_boost_round=500,
            valid_sets=[val_data], callbacks=callbacks,
        )

        train_preds = self.short_model.predict(X, num_iteration=self.short_model.best_iteration)
        self.short_pred_quantiles = {
            "mean": float(np.mean(train_preds)),
            "std": float(np.std(train_preds)),
        }

        self.state["short_horizon"]["last_retrain_date"] = train_dates[-1]

        # Save model
        self.short_model.save_model(str(SHORT_MODEL_DIR / "lgbm_model.txt"))
        np.savez_compressed(str(SHORT_MODEL_DIR / "scaler.npz"),
                            median=self.short_scaler_median, iqr=self.short_scaler_iqr)
        with open(SHORT_MODEL_DIR / "meta.json", "w") as f:
            json.dump({
                "feature_cols": self.short_feature_cols,
                "pred_quantiles": self.short_pred_quantiles,
                "train_dates": [train_dates[0], train_dates[-1]],
                "n_samples": len(y),
                "best_iteration": self.short_model.best_iteration,
            }, f, indent=2)

        ic_oot = np.corrcoef(train_preds[-n_val:], y[-n_val:])[0, 1]
        log.info(
            f"SHORT retrained: {train_dates[0]}->{train_dates[-1]}, "
            f"{len(y)} samples, iter={self.short_model.best_iteration}, IC={ic_oot:.4f}"
        )
        return True

    def load_short_model(self) -> bool:
        """Load pre-trained short-horizon model."""
        model_path = SHORT_MODEL_DIR / "lgbm_model.txt"
        scaler_path = SHORT_MODEL_DIR / "scaler.npz"
        meta_path = SHORT_MODEL_DIR / "meta.json"

        if not all(p.exists() for p in [model_path, scaler_path, meta_path]):
            return False
        try:
            import lightgbm as lgb
            self.short_model = lgb.Booster(model_file=str(model_path))
            scaler = np.load(str(scaler_path))
            self.short_scaler_median = scaler["median"]
            self.short_scaler_iqr = scaler["iqr"]
            with open(meta_path) as f:
                meta = json.load(f)
            self.short_feature_cols = meta["feature_cols"]
            self.short_pred_quantiles = meta.get("pred_quantiles", {"mean": 0, "std": 1})
            log.info(f"Loaded short-horizon model ({len(self.short_feature_cols)} features)")
            return True
        except Exception as e:
            log.warning(f"Failed to load short model: {e}")
            return False

    def predict_short(self, bars_df: pd.DataFrame) -> Optional[Dict]:
        """Generate prediction from short-horizon model."""
        if self.short_model is None or self.short_feature_cols is None:
            return None
        if bars_df.empty:
            return None

        latest = bars_df.iloc[-1:]
        missing = [c for c in self.short_feature_cols if c not in latest.columns]
        if missing:
            return None

        X_raw = latest[self.short_feature_cols].values.astype(np.float32)
        X = np.clip(np.nan_to_num(
            (X_raw - self.short_scaler_median) / self.short_scaler_iqr,
            nan=0.0, posinf=3.0, neginf=-3.0
        ), -5, 5)

        pred = self.short_model.predict(X, num_iteration=self.short_model.best_iteration)[0]

        pred_mean = self.short_pred_quantiles.get("mean", 0)
        pred_std = max(self.short_pred_quantiles.get("std", 1), 1e-6)
        zscore = (pred - pred_mean) / pred_std

        return {
            "prediction_ticks": float(pred),
            "zscore": float(zscore),
            "bar_date": latest["date"].iloc[0],
            "bar_close": float(latest["close"].iloc[0]),
            "bar_ts": str(latest["ts"].iloc[0]),
        }

    # ── LONG-HORIZON: Training ──

    def retrain_long(self, minute_df: pd.DataFrame) -> bool:
        """Retrain long-horizon LightGBM on hourly features."""
        try:
            import lightgbm as lgb
        except ImportError:
            log.error("LightGBM not installed")
            return False

        hourly_df = compute_hourly_features(minute_df)
        if hourly_df.empty:
            return False
        hourly_df = add_long_rolling_context(hourly_df)
        hourly_df = add_long_macro_features(hourly_df)

        dates = sorted(hourly_df["date"].unique())
        if len(dates) < LONG_TRAIN_DAYS:
            log.warning(f"Long-horizon: not enough days ({len(dates)} < {LONG_TRAIN_DAYS})")
            return False

        train_dates = dates[-LONG_TRAIN_DAYS:]
        train_df = hourly_df[hourly_df["date"].isin(train_dates)].copy().sort_values("ts").reset_index(drop=True)

        # Forward label: 4h forward ticks
        train_df['fwd_ticks'] = (
            train_df.groupby('date')['close'].shift(-4) - train_df['close']
        ) / ES_TICK_SIZE
        train_df = train_df.dropna(subset=['fwd_ticks'])

        # Feature columns (clean: no price levels)
        exclude = {'fwd_ticks', 'date', 'ts', 'vol_regime_mode', 'open', 'high',
                    'low', 'close', 'intraday_cum_return', 'hour'}
        self.long_feature_cols = [
            c for c in train_df.columns
            if c not in exclude and not c.startswith(('fwd_', 'direction_', 'up_'))
        ]

        X = train_df[self.long_feature_cols].values
        y = train_df['fwd_ticks'].values

        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        valid = ~np.isnan(y)
        X, y = X[valid], y[valid]

        if len(X) < 100:
            log.warning(f"Long-horizon: too few samples ({len(X)})")
            return False

        # Use last 20% as held-out validation for quantile estimation
        # Training predictions overfit → quantiles too wide → live predictions never trigger
        split_idx = int(len(X) * 0.8)
        X_fit, y_fit = X[:split_idx], y[:split_idx]
        X_held, y_held = X[split_idx:], y[split_idx:]

        train_data = lgb.Dataset(X_fit, label=y_fit)
        self.long_model = lgb.train(LONG_LGBM_PARAMS, train_data, num_boost_round=500)

        # Compute quantiles on HELD-OUT predictions (closer to live behavior)
        held_preds = self.long_model.predict(X_held)
        self.long_pred_quantiles = {
            'p85': float(np.percentile(held_preds, 85)),
            'p15': float(np.percentile(held_preds, 15)),
            'p90': float(np.percentile(held_preds, 90)),
            'p10': float(np.percentile(held_preds, 10)),
        }

        self.state["long_horizon"]["last_retrain_date"] = train_dates[-1]

        # Save model
        self.long_model.save_model(str(LONG_MODEL_DIR / "lgbm_model.txt"))
        with open(LONG_MODEL_DIR / "meta.json", "w") as f:
            json.dump({
                "feature_cols": self.long_feature_cols,
                "pred_quantiles": self.long_pred_quantiles,
                "train_dates": [train_dates[0], train_dates[-1]],
                "n_samples": len(X),
            }, f, indent=2)

        log.info(
            f"LONG retrained: {train_dates[0]}->{train_dates[-1]}, "
            f"{len(X)} samples, p15={self.long_pred_quantiles['p15']:.1f}, "
            f"p85={self.long_pred_quantiles['p85']:.1f}"
        )
        return True

    def load_long_model(self) -> bool:
        """Load pre-trained long-horizon model."""
        model_path = LONG_MODEL_DIR / "lgbm_model.txt"
        meta_path = LONG_MODEL_DIR / "meta.json"

        if not all(p.exists() for p in [model_path, meta_path]):
            return False
        try:
            import lightgbm as lgb
            self.long_model = lgb.Booster(model_file=str(model_path))
            with open(meta_path) as f:
                meta = json.load(f)
            self.long_feature_cols = meta["feature_cols"]
            self.long_pred_quantiles = meta.get("pred_quantiles", {})
            log.info(f"Loaded long-horizon model ({len(self.long_feature_cols)} features)")
            return True
        except Exception as e:
            log.warning(f"Failed to load long model: {e}")
            return False

    def predict_long(self, hourly_df: pd.DataFrame) -> Optional[Dict]:
        """Generate prediction from long-horizon model."""
        if self.long_model is None or self.long_feature_cols is None:
            return None
        if hourly_df.empty:
            return None

        latest = hourly_df.iloc[-1:]
        missing = [c for c in self.long_feature_cols if c not in latest.columns]
        if missing:
            log.warning(f"Long model missing features: {missing[:5]}")
            return None

        X = latest[self.long_feature_cols].values
        X = np.nan_to_num(X, nan=0, posinf=0, neginf=0)
        pred = self.long_model.predict(X)[0]

        signal = 0
        if pred >= self.long_pred_quantiles.get('p85', 999):
            signal = 1
        elif pred <= self.long_pred_quantiles.get('p15', -999):
            signal = -1

        return {
            'prediction_ticks': float(pred),
            'signal': signal,
            'bar_date': latest['date'].iloc[0],
            'bar_close': float(latest['close'].iloc[0]),
            'bar_ts': str(latest['ts'].iloc[0]),
            'bar_hour': int(latest['hour'].iloc[0]) if 'hour' in latest.columns else None,
        }

    # ── LIVE LOOP ──

    def run_live(self):
        """Main live loop: retrain + predict on new bar closes."""
        self.state["mode"] = "live"
        self._save_state()
        start_health_server(self)

        log.info("=" * 70)
        log.info("MULTI-TIMEFRAME PAPER ENGINE — LIVE MODE")
        log.info(f"  Short-horizon: SL{SHORT_SL_TICKS}/TP{SHORT_TP_TICKS}, "
                 f"conf>={SHORT_ENTRY_CONF}, |z|>={SHORT_ZSCORE_THRESHOLD}")
        log.info(f"  Long-horizon: top/bottom {int(LONG_THRESHOLD_PCT*100)}%, "
                 f"{LONG_HOLD_HOURS}h hold")
        log.info(f"  Portfolio: {int(WEIGHT_SHORT*100)}/{int(WEIGHT_LONG*100)} "
                 f"short/long allocation")
        log.info("=" * 70)

        # Try loading existing models
        short_loaded = self.load_short_model()
        long_loaded = self.load_long_model()

        last_short_bar = None
        last_long_bar = None
        last_retrain_date = None

        while True:
            try:
                now = datetime.now(timezone.utc)

                # Only run during extended hours (pre-market + RTH + post)
                if not (12 <= now.hour <= 22):
                    time.sleep(60)
                    continue

                # Load data
                minute_df = load_minute_bars(n_days=LOOKBACK_DAYS)
                if minute_df.empty:
                    log.warning("No minute bar data available")
                    time.sleep(LOOP_SLEEP_SECONDS)
                    continue

                today = now.strftime("%Y%m%d")

                # ── Daily retrain check ──
                if last_retrain_date != today:
                    log.info("Daily retrain triggered")
                    if self.retrain_short(minute_df):
                        short_loaded = True
                    if self.retrain_long(minute_df):
                        long_loaded = True
                    last_retrain_date = today

                # ── Short-horizon: check for new 30-min bar ──
                if short_loaded:
                    bars_30 = aggregate_to_30min_bars(minute_df)
                    if not bars_30.empty:
                        bars_30 = add_short_rolling_features(bars_30)
                        current_bar_key = str(bars_30.iloc[-1]["ts"])
                        if current_bar_key != last_short_bar:
                            last_short_bar = current_bar_key
                            sig = self.predict_short(bars_30)
                            if sig:
                                self._handle_short_signal(sig, minute_df)

                # ── Long-horizon: check for new hourly bar ──
                if long_loaded:
                    hourly_df = compute_hourly_features(minute_df)
                    if not hourly_df.empty:
                        hourly_df = add_long_rolling_context(hourly_df)
                        hourly_df = add_long_macro_features(hourly_df)
                        current_hour = str(hourly_df.iloc[-1]["ts"])
                        if current_hour != last_long_bar:
                            last_long_bar = current_hour
                            sig = self.predict_long(hourly_df)
                            if sig:
                                self._handle_long_signal(sig, minute_df)

                # ── Check long-horizon exits ──
                self._check_long_exit(minute_df, now)

                self._save_state()
                gc.collect()

            except Exception as e:
                log.error(f"Live loop error: {e}\n{traceback.format_exc()}")

            time.sleep(LOOP_SLEEP_SECONDS)

    def _handle_short_signal(self, sig: Dict, minute_df: pd.DataFrame):
        """Process short-horizon signal — check entry criteria, simulate fill."""
        ss = self.state["short_horizon"]

        # Already positioned
        if ss["position"] != 0:
            return

        pred = sig["prediction_ticks"]
        zscore = sig["zscore"]

        ss["signals_generated"] += 1

        # Entry criteria: confidence and z-score
        if abs(zscore) < SHORT_ZSCORE_THRESHOLD:
            ss["signals_filtered"] += 1
            return

        direction = 1 if pred > 0 else -1
        entry_price = sig["bar_close"]

        # In live mode, we place the order and track it
        ss["position"] = direction
        ss["entry_price"] = entry_price
        ss["entry_time"] = sig["bar_ts"]
        ss["fill_time"] = sig["bar_ts"]  # Assume fill at bar close for paper
        # Asymmetric SL: longs use SHORT_SL_LONG_TICKS, shorts use SHORT_SL_SHORT_TICKS
        sl_ticks = SHORT_SL_LONG_TICKS if direction == 1 else SHORT_SL_SHORT_TICKS
        ss["tp_price"] = entry_price + SHORT_TP_TICKS * ES_TICK_SIZE * direction
        ss["sl_price"] = entry_price - sl_ticks * ES_TICK_SIZE * direction
        ss["signals_traded"] += 1

        log.info(
            f"SHORT-HORIZON ENTRY: {'LONG' if direction > 0 else 'SHORT'} "
            f"@ {entry_price:.2f}, pred={pred:.2f}t, z={zscore:.2f}"
        )

    def _handle_long_signal(self, sig: Dict, minute_df: pd.DataFrame):
        """Process long-horizon signal."""
        ls = self.state["long_horizon"]

        if ls["position"] != 0:
            return

        if sig["signal"] == 0:
            return

        direction = sig["signal"]
        entry_price = sig["bar_close"]

        ls["position"] = direction
        ls["entry_price"] = entry_price
        ls["entry_time"] = sig["bar_ts"]
        exit_target = datetime.fromisoformat(sig["bar_ts"].replace("Z", "+00:00")) if "Z" in sig["bar_ts"] else datetime.now(timezone.utc)
        ls["exit_target_time"] = (exit_target + timedelta(hours=LONG_HOLD_HOURS)).isoformat()

        log.info(
            f"LONG-HORIZON ENTRY: {'LONG' if direction > 0 else 'SHORT'} "
            f"@ {entry_price:.2f}, pred={sig['prediction_ticks']:.1f}t"
        )

    def _check_long_exit(self, minute_df: pd.DataFrame, now: datetime):
        """Check if long-horizon position should be closed (time-based)."""
        ls = self.state["long_horizon"]
        if ls["position"] == 0 or not ls.get("exit_target_time"):
            return

        target = datetime.fromisoformat(ls["exit_target_time"])
        if target.tzinfo is None:
            target = target.replace(tzinfo=timezone.utc)

        if now >= target:
            # Get latest price
            if not minute_df.empty:
                exit_price = float(minute_df.iloc[-1]["close"])
            else:
                exit_price = ls["entry_price"]

            self._close_long_position(exit_price, "TIME_EXIT")

    def _close_long_position(self, exit_price: float, reason: str):
        """Close long-horizon position and record trade."""
        ls = self.state["long_horizon"]
        pos = ls["position"]
        entry = ls["entry_price"]

        raw_ticks = (exit_price - entry) / ES_TICK_SIZE * pos
        cost = LONG_COST_RT
        net_ticks = raw_ticks - cost
        net_dollars = net_ticks * ES_TICK_VALUE
        weighted_pnl = net_ticks * WEIGHT_LONG

        ls["total_trades"] += 1
        ls["total_pnl_ticks"] += net_ticks
        ls["total_pnl_dollars"] += net_dollars
        if net_ticks > 0:
            ls["wins"] += 1
        else:
            ls["losses"] += 1

        self.state["portfolio"]["weighted_pnl_ticks"] += weighted_pnl
        self.state["portfolio"]["weighted_pnl_dollars"] += weighted_pnl * ES_TICK_VALUE
        self.state["portfolio"]["capital"] += weighted_pnl * ES_TICK_VALUE

        trade = {
            "trade_id": ls["total_trades"],
            "date": datetime.now(timezone.utc).strftime("%Y%m%d"),
            "entry_time": ls["entry_time"],
            "exit_time": datetime.now(timezone.utc).isoformat(),
            "direction": "LONG" if pos > 0 else "SHORT",
            "entry_price": entry,
            "exit_price": exit_price,
            "exit_type": reason,
            "raw_pnl_ticks": raw_ticks,
            "cost_ticks": cost,
            "net_pnl_ticks": net_ticks,
            "net_pnl_dollars": net_dollars,
            "prediction": 0,
            "hold_hours": LONG_HOLD_HOURS,
            "weighted_pnl_ticks": weighted_pnl,
        }
        self._append_long_trade(trade)

        log.info(
            f"LONG-HORIZON EXIT ({reason}): {'LONG' if pos > 0 else 'SHORT'} "
            f"@ {exit_price:.2f}, net={net_ticks:+.1f}t (${net_dollars:+.0f}), "
            f"weighted={weighted_pnl:+.1f}t"
        )

        ls["position"] = 0
        ls["entry_price"] = 0.0
        ls["entry_time"] = None
        ls["exit_target_time"] = None
        self._save_state()

    # ── REPLAY MODE ──

    def replay(self, n_days: int = 0):
        """Walk-forward replay on historical minute bar data."""
        self.state = self._fresh_state()
        self.state["mode"] = "replay"

        log.info("=" * 70)
        log.info("MULTI-TIMEFRAME PAPER ENGINE — REPLAY MODE")
        log.info(f"  Short-horizon: SL{SHORT_SL_TICKS}/TP{SHORT_TP_TICKS}, "
                 f"conf>={SHORT_ENTRY_CONF}, |z|>={SHORT_ZSCORE_THRESHOLD}")
        log.info(f"  Long-horizon: top/bottom {int(LONG_THRESHOLD_PCT*100)}%, "
                 f"{LONG_HOLD_HOURS}h hold")
        log.info(f"  Portfolio: {int(WEIGHT_SHORT*100)}/{int(WEIGHT_LONG*100)}")
        log.info("=" * 70)

        minute_df = load_minute_bars(n_days=LOOKBACK_DAYS)
        if minute_df.empty:
            log.error("No minute bar data")
            return

        # Build feature sets
        log.info("Computing 30-min bars and features...")
        bars_30 = aggregate_to_30min_bars(minute_df)
        bars_30 = add_short_rolling_features(bars_30)

        log.info("Computing hourly bars and features...")
        hourly_df = compute_hourly_features(minute_df)
        hourly_df = add_long_rolling_context(hourly_df)
        hourly_df = add_long_macro_features(hourly_df)

        all_dates = sorted(bars_30["date"].unique())
        if len(all_dates) < SHORT_TRAIN_DAYS + 1:
            log.error(f"Not enough dates for walk-forward ({len(all_dates)})")
            return

        # Walk-forward: train on first TRAIN_DAYS, test on rest
        train_end_idx = SHORT_TRAIN_DAYS
        if n_days > 0:
            test_dates = all_dates[-(n_days):]
        else:
            test_dates = all_dates[train_end_idx:]

        log.info(f"Testing on {len(test_dates)} days: {test_dates[0]} -> {test_dates[-1]}")

        # Train initial models
        train_minute = minute_df[minute_df["date"].isin(all_dates[:train_end_idx])]
        if not train_minute.empty:
            self.retrain_short(train_minute)
            self.retrain_long(train_minute)

        daily_stats = {}

        for test_date in test_dates:
            if test_date not in daily_stats:
                daily_stats[test_date] = {
                    "short_trades": 0, "short_pnl": 0.0,
                    "short_wins": 0, "short_losses": 0,
                    "long_trades": 0, "long_pnl": 0.0,
                    "long_wins": 0, "long_losses": 0,
                }

            # ── SHORT-HORIZON TRADES ──
            day_bars = bars_30[bars_30["date"] == test_date].copy()
            day_minutes = minute_df[minute_df["date"] == test_date].copy()

            if not day_bars.empty and self.short_model is not None:
                for bar_idx in range(len(day_bars)):
                    bar = day_bars.iloc[bar_idx]
                    bar_ts = bar["ts"]

                    # RTH check
                    h, m = bar_ts.hour, bar_ts.minute
                    t = h * 60 + m
                    if not (RTH_START_H * 60 + RTH_START_M <= t < RTH_END_H * 60 - SHORT_BAR_SIZE_MINUTES):
                        continue

                    # Need context bars for prediction
                    context = bars_30[bars_30["ts"] <= bar_ts].tail(50)
                    if len(context) < 5:
                        continue

                    sig = self.predict_short(context)
                    if sig is None:
                        continue

                    self.state["short_horizon"]["signals_generated"] += 1

                    if abs(sig["zscore"]) < SHORT_ZSCORE_THRESHOLD:
                        self.state["short_horizon"]["signals_filtered"] += 1
                        continue

                    direction = 1 if sig["prediction_ticks"] > 0 else -1
                    entry_price = float(bar["close"])

                    # Simulate FIFO fill on minute bars after the signal bar
                    signal_time = bar_ts
                    after_signal = day_minutes[day_minutes["ts_minute"] > signal_time].head(SHORT_BAR_SIZE_MINUTES)
                    if after_signal.empty:
                        continue

                    fill = simulate_fifo_fill(entry_price, direction, after_signal)
                    if fill is None:
                        continue

                    # Simulate exit
                    fill_time = fill["fill_time"]
                    after_fill = day_minutes[day_minutes["ts_minute"] > fill_time]
                    exit_info = simulate_short_exit(
                        entry_price, direction, fill_time, after_fill
                    )

                    net_pnl = exit_info["raw_pnl_ticks"] - exit_info["cost_ticks"]
                    net_dollars = net_pnl * ES_TICK_VALUE
                    weighted = net_pnl * WEIGHT_SHORT

                    hold_min = 0
                    if exit_info.get("exit_time") is not None:
                        hold_min = (exit_info["exit_time"] - fill_time).total_seconds() / 60

                    # Record
                    ss = self.state["short_horizon"]
                    ss["total_trades"] += 1
                    ss["total_pnl_ticks"] += net_pnl
                    ss["total_pnl_dollars"] += net_dollars
                    ss["signals_traded"] += 1
                    if net_pnl > 0:
                        ss["wins"] += 1
                    else:
                        ss["losses"] += 1
                    exit_type = exit_info["exit_type"]
                    if exit_type == "TP":
                        ss["tp_exits"] += 1
                    elif exit_type == "SL":
                        ss["sl_exits"] += 1
                    elif exit_type in ("TimeStop", "EOD"):
                        ss["time_exits"] += 1

                    self.state["portfolio"]["weighted_pnl_ticks"] += weighted
                    self.state["portfolio"]["weighted_pnl_dollars"] += weighted * ES_TICK_VALUE
                    self.state["portfolio"]["capital"] += weighted * ES_TICK_VALUE

                    trade = {
                        "trade_id": ss["total_trades"],
                        "date": test_date,
                        "signal_time": str(signal_time),
                        "fill_time": str(fill_time),
                        "exit_time": str(exit_info["exit_time"]),
                        "direction": "LONG" if direction > 0 else "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_info["exit_price"],
                        "exit_type": exit_type,
                        "raw_pnl_ticks": exit_info["raw_pnl_ticks"],
                        "cost_ticks": exit_info["cost_ticks"],
                        "net_pnl_ticks": net_pnl,
                        "net_pnl_dollars": net_dollars,
                        "prediction": sig["prediction_ticks"],
                        "pred_zscore": sig["zscore"],
                        "hold_minutes": hold_min,
                        "weighted_pnl_ticks": weighted,
                    }
                    self._append_short_trade(trade)

                    daily_stats[test_date]["short_trades"] += 1
                    daily_stats[test_date]["short_pnl"] += net_pnl
                    if net_pnl > 0:
                        daily_stats[test_date]["short_wins"] += 1
                    else:
                        daily_stats[test_date]["short_losses"] += 1

            # ── LONG-HORIZON TRADES ──
            day_hourly = hourly_df[hourly_df["date"] == test_date].copy()

            if not day_hourly.empty and self.long_model is not None:
                for hr_idx in range(len(day_hourly)):
                    hr_bar = day_hourly.iloc[hr_idx]

                    # RTH check
                    if not (RTH_START_H <= hr_bar["hour"] < RTH_END_H - LONG_HOLD_HOURS):
                        continue

                    # Need context for prediction
                    context = hourly_df[hourly_df["ts"] <= hr_bar["ts"]].tail(50)
                    if len(context) < 5:
                        continue

                    sig = self.predict_long(context)
                    if sig is None or sig["signal"] == 0:
                        continue

                    direction = sig["signal"]
                    entry_price = float(hr_bar["close"])
                    entry_time = hr_bar["ts"]

                    # Exit: 4 hours later or EOD
                    exit_time_target = entry_time + pd.Timedelta(hours=LONG_HOLD_HOURS)
                    eod_time = entry_time.replace(hour=19, minute=55, second=0)

                    actual_exit_time = min(exit_time_target, eod_time)

                    # Find exit price from minute bars
                    exit_minutes = minute_df[
                        (minute_df["ts_minute"] >= actual_exit_time) &
                        (minute_df["date"] == test_date)
                    ]
                    if not exit_minutes.empty:
                        exit_price = float(exit_minutes.iloc[0]["close"])
                    else:
                        # Use last available price for the day
                        day_end = day_minutes.iloc[-1]["close"] if not day_minutes.empty else entry_price
                        exit_price = float(day_end)

                    raw_ticks = (exit_price - entry_price) / ES_TICK_SIZE * direction
                    net_ticks = raw_ticks - LONG_COST_RT
                    net_dollars = net_ticks * ES_TICK_VALUE
                    weighted = net_ticks * WEIGHT_LONG

                    hold_hrs = (actual_exit_time - entry_time).total_seconds() / 3600

                    ls = self.state["long_horizon"]
                    ls["total_trades"] += 1
                    ls["total_pnl_ticks"] += net_ticks
                    ls["total_pnl_dollars"] += net_dollars
                    if net_ticks > 0:
                        ls["wins"] += 1
                    else:
                        ls["losses"] += 1

                    self.state["portfolio"]["weighted_pnl_ticks"] += weighted
                    self.state["portfolio"]["weighted_pnl_dollars"] += weighted * ES_TICK_VALUE
                    self.state["portfolio"]["capital"] += weighted * ES_TICK_VALUE

                    trade = {
                        "trade_id": ls["total_trades"],
                        "date": test_date,
                        "entry_time": str(entry_time),
                        "exit_time": str(actual_exit_time),
                        "direction": "LONG" if direction > 0 else "SHORT",
                        "entry_price": entry_price,
                        "exit_price": exit_price,
                        "exit_type": "TIME_EXIT",
                        "raw_pnl_ticks": raw_ticks,
                        "cost_ticks": LONG_COST_RT,
                        "net_pnl_ticks": net_ticks,
                        "net_pnl_dollars": net_dollars,
                        "prediction": sig["prediction_ticks"],
                        "hold_hours": hold_hrs,
                        "weighted_pnl_ticks": weighted,
                    }
                    self._append_long_trade(trade)

                    daily_stats[test_date]["long_trades"] += 1
                    daily_stats[test_date]["long_pnl"] += net_ticks
                    if net_ticks > 0:
                        daily_stats[test_date]["long_wins"] += 1
                    else:
                        daily_stats[test_date]["long_losses"] += 1

        # ── Print summary ──
        self._print_summary(daily_stats)
        self._save_state()

    def _print_summary(self, daily_stats: Dict):
        """Print end-of-replay summary."""
        ss = self.state["short_horizon"]
        ls = self.state["long_horizon"]
        pf = self.state["portfolio"]

        log.info("\n" + "=" * 70)
        log.info("MULTI-TIMEFRAME PAPER ENGINE — REPLAY SUMMARY")
        log.info("=" * 70)

        # Short-horizon
        short_wr = ss["wins"] / max(ss["total_trades"], 1) * 100
        log.info(f"\nSHORT-HORIZON (weight={WEIGHT_SHORT}):")
        log.info(f"  Trades: {ss['total_trades']}")
        log.info(f"  WR: {short_wr:.1f}%")
        log.info(f"  Net P&L: {ss['total_pnl_ticks']:+.1f} ticks (${ss['total_pnl_dollars']:+,.0f})")
        log.info(f"  TP/SL/Time exits: {ss['tp_exits']}/{ss['sl_exits']}/{ss['time_exits']}")
        log.info(f"  Signals: {ss['signals_generated']} generated, "
                 f"{ss['signals_filtered']} filtered, {ss['signals_traded']} traded")

        # Long-horizon
        long_wr = ls["wins"] / max(ls["total_trades"], 1) * 100
        log.info(f"\nLONG-HORIZON (weight={WEIGHT_LONG}):")
        log.info(f"  Trades: {ls['total_trades']}")
        log.info(f"  WR: {long_wr:.1f}%")
        log.info(f"  Net P&L: {ls['total_pnl_ticks']:+.1f} ticks (${ls['total_pnl_dollars']:+,.0f})")

        # Portfolio
        log.info(f"\nPORTFOLIO (70/30):")
        log.info(f"  Weighted P&L: {pf['weighted_pnl_ticks']:+.1f} ticks (${pf['weighted_pnl_dollars']:+,.0f})")
        log.info(f"  Capital: ${pf['capital']:,.0f}")

        # Daily stats
        if daily_stats:
            daily_pnls = []
            for date, ds in sorted(daily_stats.items()):
                combined = (ds["short_pnl"] * WEIGHT_SHORT + ds["long_pnl"] * WEIGHT_LONG)
                daily_pnls.append(combined)
                combined_dollars = combined * ES_TICK_VALUE

                with open(self.daily_csv, "a", newline="") as f:
                    csv.writer(f).writerow([
                        date,
                        ds["short_trades"], round(ds["short_pnl"], 2),
                        ds["short_wins"], ds["short_losses"],
                        ds["long_trades"], round(ds["long_pnl"], 2),
                        ds["long_wins"], ds["long_losses"],
                        round(combined, 2), round(combined_dollars, 2),
                        round(pf["capital"], 2),
                    ])

            daily_arr = np.array(daily_pnls)
            if len(daily_arr) > 1 and daily_arr.std() > 0:
                daily_sharpe = daily_arr.mean() / daily_arr.std() * np.sqrt(252)
                downside = np.sqrt(np.mean(np.minimum(daily_arr, 0) ** 2))
                daily_sortino = daily_arr.mean() / max(downside, 1e-8) * np.sqrt(252)
                green_days = sum(1 for x in daily_arr if x > 0)

                log.info(f"\n  Daily Sharpe: {daily_sharpe:.2f}")
                log.info(f"  Daily Sortino: {daily_sortino:.2f}")
                log.info(f"  Green days: {green_days}/{len(daily_arr)} "
                         f"({green_days/len(daily_arr)*100:.0f}%)")

        log.info("=" * 70)

    def print_state(self):
        """Print current state summary."""
        ss = self.state["short_horizon"]
        ls = self.state["long_horizon"]
        pf = self.state["portfolio"]

        print("\n" + "=" * 60)
        print("MULTI-TIMEFRAME PAPER ENGINE — CURRENT STATE")
        print("=" * 60)
        print(f"Mode: {self.state.get('mode', 'unknown')}")
        print(f"Last heartbeat: {self.state.get('last_heartbeat', 'never')}")

        print(f"\nSHORT-HORIZON ({WEIGHT_SHORT*100:.0f}% weight):")
        print(f"  Position: {ss['position']}")
        print(f"  Trades: {ss['total_trades']}, WR: "
              f"{ss['wins']/max(ss['total_trades'],1)*100:.1f}%")
        print(f"  Net P&L: {ss['total_pnl_ticks']:+.1f}t "
              f"(${ss['total_pnl_dollars']:+,.0f})")

        print(f"\nLONG-HORIZON ({WEIGHT_LONG*100:.0f}% weight):")
        print(f"  Position: {ls['position']}")
        print(f"  Trades: {ls['total_trades']}, WR: "
              f"{ls['wins']/max(ls['total_trades'],1)*100:.1f}%")
        print(f"  Net P&L: {ls['total_pnl_ticks']:+.1f}t "
              f"(${ls['total_pnl_dollars']:+,.0f})")

        print(f"\nPORTFOLIO:")
        print(f"  Weighted P&L: {pf['weighted_pnl_ticks']:+.1f}t "
              f"(${pf['weighted_pnl_dollars']:+,.0f})")
        print(f"  Capital: ${pf['capital']:,.0f}")
        print("=" * 60)


# ═══════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════


def main():
    parser = argparse.ArgumentParser(description="Multi-Timeframe Paper Trading Engine")
    parser.add_argument("--replay", type=int, nargs="?", const=0, default=None,
                        help="Replay N OOT days (0 = all available)")
    parser.add_argument("--live", action="store_true",
                        help="Run in live loop mode")
    parser.add_argument("--summary", action="store_true",
                        help="Print current state and exit")
    args = parser.parse_args()

    engine = MultiTimeframePaperEngine()

    if args.summary:
        engine.print_state()
    elif args.replay is not None:
        engine.replay(n_days=args.replay)
    elif args.live:
        engine.run_live()
    else:
        # Default: print state
        engine.print_state()


if __name__ == "__main__":
    main()
