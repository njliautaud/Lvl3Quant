#!/usr/bin/env python3
"""
dynamic_execution_v8.py — Dynamic SL/TP + Mid-Trade Management for ES Futures
==============================================================================
30-min LightGBM entry model on minute bars with:
  Phase 1: Volatility-adaptive SL/TP optimization (ATR-based + percentile-based)
  Phase 2: Mid-trade minute-bar management (LightGBM exit classifier)
  Phase 3: Regime-stratified validation (HC #428 gate)
  Phase 4: MLflow logging

Walk-forward: SLIDING 60d train, 1d OOT, drop oldest (HC #0).
Costs: passive entry = 0.376 ticks, market exit = 1.376 ticks (HC constants).

Neptune GPU (RTX 3090): /home/nick/Lvl3Quant/
Data: /home/nick/Lvl3Quant/data/processed/mbo_minute_bars_v1/*.parquet

Usage:
    python alpha_discovery/dynamic_execution_v8.py
    python alpha_discovery/dynamic_execution_v8.py --quick   # fewer configs for testing
"""

import os
import gc
import sys
import json
import time
import logging
import argparse
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict, field

import numpy as np
import pandas as pd
from scipy.stats import spearmanr, zscore

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# ---------------------------------------------------------------------------
# Path setup — works on both Jupiter and Neptune
# ---------------------------------------------------------------------------
HOSTNAME = os.uname().nodename.lower()
if "neptune" in HOSTNAME or "nick" in str(Path.home()):
    LVL3_ROOT = Path("/home/nick/Lvl3Quant")
else:
    LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")

MINUTE_BAR_DIR = LVL3_ROOT / "data" / "processed" / "mbo_minute_bars_v1"
ENHANCED_FEATURES = LVL3_ROOT / "output" / "long_horizon_flow_v2" / "enhanced_daily_features.parquet"
OUTPUT_DIR = LVL3_ROOT / "output" / "dynamic_execution_v8"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_ts = datetime.now().strftime("%Y%m%d_%H%M%S")
_log_file = OUTPUT_DIR / f"dynamic_execution_v8_{_ts}.log"

logging.basicConfig(
    format="%(asctime)s [dyn_exec_v8] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.INFO,
    handlers=[
        logging.FileHandler(str(_log_file), mode="w", encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
log = logging.getLogger("dyn_exec_v8")

# ---------------------------------------------------------------------------
# Constants (ES Futures — canonical from CLAUDE.md)
# ---------------------------------------------------------------------------
TICK_SIZE = 0.25           # ES tick = 0.25 points
TICK_VALUE = 12.50         # $12.50 per tick
ES_RT_COMMISSION = 4.70   # AMP round-trip
COMMISSION_TICKS = ES_RT_COMMISSION / TICK_VALUE  # 0.376 ticks

# Cost model: passive entry + market exit
ENTRY_COST_TICKS = COMMISSION_TICKS / 2   # 0.188 (half RT for entry leg)
EXIT_COST_TICKS = COMMISSION_TICKS / 2 + 1.0  # 1.188 (half RT + 1 tick spread for market exit)
TOTAL_RT_COST_TICKS = ENTRY_COST_TICKS + EXIT_COST_TICKS  # ~1.376 ticks

# Walk-forward parameters
TRAIN_DAYS = 60
SLIDE_DAYS = 1

# Trade parameters
HOLD_BARS = 30             # 30-minute hold horizon (1 bar = 1 minute)
PREDICTION_HORIZON = 30   # predict 30 min ahead

# ---------------------------------------------------------------------------
# Feature engineering from minute bars
# ---------------------------------------------------------------------------
FEATURE_NAMES = [
    # Price returns
    "ret_1m", "ret_5m", "ret_10m", "ret_15m", "ret_30m",
    # Volatility
    "realized_vol_5m", "realized_vol_10m", "realized_vol_30m",
    "atr_5m", "atr_10m",
    "high_low_range_5m", "high_low_range_10m",
    # Volume
    "volume_1m", "volume_ratio_5m", "volume_ratio_10m",
    "volume_zscore_30m",
    # Order flow
    "ofi_1m", "ofi_cum_5m", "ofi_cum_10m",
    "ofi_zscore_30m",
    "signed_volume_ratio_5m",
    # Microstructure
    "spread_mean_1m", "spread_zscore_10m",
    "vwap_deviation",
    "microprice_deviation",
    # Price position
    "price_position_5m", "price_position_10m", "price_position_30m",
    # Momentum / mean-reversion
    "rsi_14", "momentum_score",
    # Trade intensity
    "trade_count_ratio_5m",
    # Vol regime (encoded)
    "vol_regime_encoded",
]

NUM_FEATURES = len(FEATURE_NAMES)


def compute_features(df: pd.DataFrame) -> np.ndarray:
    """
    Compute feature matrix from a single day's minute bar DataFrame.
    Returns float32 array of shape (n_bars, n_features).
    """
    n = len(df)
    feats = np.full((n, NUM_FEATURES), np.nan, dtype=np.float32)

    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    opn = df["open"].values.astype(np.float64)
    volume = df["volume"].values.astype(np.float64)
    vwap = df["vwap"].values.astype(np.float64)
    trade_count = df["trade_count"].values.astype(np.float64)
    signed_vol = df["signed_volume"].values.astype(np.float64)
    spread = df["spread_mean"].values.astype(np.float64)
    ofi = df["ofi_1min"].values.astype(np.float64)
    microprice = df["microprice_close"].values.astype(np.float64)

    # Vol regime encoding
    vol_regime = df["vol_regime"].values
    vol_enc = np.zeros(n, dtype=np.float32)
    for i, v in enumerate(vol_regime):
        if v == "high":
            vol_enc[i] = 1.0
        elif v == "med":
            vol_enc[i] = 0.0
        elif v == "low":
            vol_enc[i] = -1.0

    fi = 0  # feature index

    # --- Price returns ---
    for lag, name in [(1, "ret_1m"), (5, "ret_5m"), (10, "ret_10m"),
                      (15, "ret_15m"), (30, "ret_30m")]:
        ret = np.full(n, np.nan)
        if lag < n:
            ret[lag:] = (close[lag:] - close[:-lag]) / (close[:-lag] + 1e-10)
        feats[:, fi] = ret
        fi += 1

    # --- Realized volatility (rolling std of 1-min returns) ---
    ret_1m = np.full(n, np.nan)
    ret_1m[1:] = (close[1:] - close[:-1]) / (close[:-1] + 1e-10)
    for window in [5, 10, 30]:
        rv = np.full(n, np.nan)
        for i in range(window, n):
            rv[i] = np.nanstd(ret_1m[i - window:i])
        feats[:, fi] = rv
        fi += 1

    # --- ATR (Average True Range) ---
    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    for window in [5, 10]:
        atr = np.full(n, np.nan)
        for i in range(window, n):
            atr[i] = np.mean(tr[i - window:i])
        feats[:, fi] = atr
        fi += 1

    # --- High-low range ---
    for window in [5, 10]:
        hlr = np.full(n, np.nan)
        for i in range(window, n):
            hlr[i] = np.max(high[i - window:i]) - np.min(low[i - window:i])
        feats[:, fi] = hlr
        fi += 1

    # --- Volume features ---
    feats[:, fi] = volume
    fi += 1
    for window in [5, 10]:
        vr = np.full(n, np.nan)
        for i in range(window, n):
            avg = np.mean(volume[i - window:i])
            vr[i] = volume[i] / (avg + 1e-10)
        feats[:, fi] = vr
        fi += 1
    # Volume z-score
    vz = np.full(n, np.nan)
    for i in range(30, n):
        window_vol = volume[i - 30:i]
        mu, sig = np.mean(window_vol), np.std(window_vol)
        vz[i] = (volume[i] - mu) / (sig + 1e-10)
    feats[:, fi] = vz
    fi += 1

    # --- OFI features ---
    feats[:, fi] = ofi
    fi += 1
    for window in [5, 10]:
        oc = np.full(n, np.nan)
        for i in range(window, n):
            oc[i] = np.sum(ofi[i - window:i])
        feats[:, fi] = oc
        fi += 1
    # OFI z-score
    oz = np.full(n, np.nan)
    for i in range(30, n):
        window_ofi = ofi[i - 30:i]
        mu, sig = np.mean(window_ofi), np.std(window_ofi)
        oz[i] = (ofi[i] - mu) / (sig + 1e-10)
    feats[:, fi] = oz
    fi += 1
    # Signed volume ratio
    for window in [5]:
        svr = np.full(n, np.nan)
        for i in range(window, n):
            total = np.sum(np.abs(signed_vol[i - window:i]))
            svr[i] = np.sum(signed_vol[i - window:i]) / (total + 1e-10)
        feats[:, fi] = svr
        fi += 1

    # --- Microstructure ---
    feats[:, fi] = spread
    fi += 1
    # Spread z-score
    sz = np.full(n, np.nan)
    for i in range(10, n):
        ws = spread[i - 10:i]
        mu, sig = np.mean(ws), np.std(ws)
        sz[i] = (spread[i] - mu) / (sig + 1e-10)
    feats[:, fi] = sz
    fi += 1
    # VWAP deviation
    feats[:, fi] = (close - vwap) / (TICK_SIZE + 1e-10)
    fi += 1
    # Microprice deviation
    feats[:, fi] = (close - microprice) / (TICK_SIZE + 1e-10)
    fi += 1

    # --- Price position in range ---
    for window in [5, 10, 30]:
        pp = np.full(n, np.nan)
        for i in range(window, n):
            hi = np.max(high[i - window:i])
            lo = np.min(low[i - window:i])
            rng = hi - lo
            pp[i] = (close[i] - lo) / (rng + 1e-10)
        feats[:, fi] = pp
        fi += 1

    # --- RSI(14) ---
    rsi = np.full(n, np.nan)
    if n > 15:
        delta = np.diff(close)
        gain = np.where(delta > 0, delta, 0.0)
        loss = np.where(delta < 0, -delta, 0.0)
        avg_gain = np.full(n - 1, np.nan)
        avg_loss = np.full(n - 1, np.nan)
        avg_gain[13] = np.mean(gain[:14])
        avg_loss[13] = np.mean(loss[:14])
        for i in range(14, len(gain)):
            avg_gain[i] = (avg_gain[i - 1] * 13 + gain[i]) / 14
            avg_loss[i] = (avg_loss[i - 1] * 13 + loss[i]) / 14
        for i in range(13, len(avg_gain)):
            if avg_loss[i] > 0:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i + 1] = 100.0 - 100.0 / (1.0 + rs)
            else:
                rsi[i + 1] = 100.0
    feats[:, fi] = rsi
    fi += 1

    # --- Momentum score (sum of signed returns over multiple horizons) ---
    mom = np.zeros(n)
    for lag in [1, 5, 10]:
        r = np.zeros(n)
        if lag < n:
            r[lag:] = np.sign(close[lag:] - close[:-lag])
        mom += r
    feats[:, fi] = mom
    fi += 1

    # --- Trade count ratio ---
    for window in [5]:
        tcr = np.full(n, np.nan)
        for i in range(window, n):
            avg = np.mean(trade_count[i - window:i])
            tcr[i] = trade_count[i] / (avg + 1e-10)
        feats[:, fi] = tcr
        fi += 1

    # --- Vol regime encoded ---
    feats[:, fi] = vol_enc
    fi += 1

    assert fi == NUM_FEATURES, f"Feature count mismatch: {fi} vs {NUM_FEATURES}"

    # Clean NaN/Inf
    np.nan_to_num(feats, copy=False, nan=0.0, posinf=1e6, neginf=-1e6)
    np.clip(feats, -1e6, 1e6, out=feats)

    return feats


def compute_labels(close: np.ndarray, horizon: int) -> np.ndarray:
    """
    Label = (close[i+horizon] - close[i]) / TICK_SIZE (in ticks).
    NaN for last `horizon` bars (no future data).
    """
    n = len(close)
    labels = np.full(n, np.nan, dtype=np.float32)
    if horizon < n:
        labels[:n - horizon] = (close[horizon:] - close[:n - horizon]) / TICK_SIZE
    return labels


def compute_atr_series(df: pd.DataFrame, window: int = 5) -> np.ndarray:
    """Compute rolling ATR in ticks from minute bars."""
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    close = df["close"].values.astype(np.float64)

    tr = np.maximum(high - low,
                    np.maximum(np.abs(high - np.roll(close, 1)),
                               np.abs(low - np.roll(close, 1))))
    tr[0] = high[0] - low[0]
    tr_ticks = tr / TICK_SIZE

    atr = np.full(len(tr), np.nan)
    for i in range(window, len(tr)):
        atr[i] = np.mean(tr_ticks[i - window:i])
    return atr


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
@dataclass
class DayData:
    date_str: str      # YYYYMMDD
    df: pd.DataFrame   # raw minute bars
    features: np.ndarray  # (n_bars, n_features)
    labels: np.ndarray    # (n_bars,) label in ticks
    close: np.ndarray     # (n_bars,) close prices
    high: np.ndarray
    low: np.ndarray
    atr: np.ndarray       # (n_bars,) rolling ATR in ticks


def load_day(parquet_path: Path) -> Optional[DayData]:
    """Load and process a single day's minute bars."""
    try:
        df = pd.read_parquet(str(parquet_path))
    except Exception as e:
        log.warning(f"Failed to load {parquet_path.name}: {e}")
        return None

    if len(df) < 60:  # need at least 60 bars for meaningful features
        return None

    date_str = parquet_path.stem  # e.g., "20260401"
    features = compute_features(df)
    close = df["close"].values.astype(np.float64)
    high = df["high"].values.astype(np.float64)
    low = df["low"].values.astype(np.float64)
    labels = compute_labels(close, PREDICTION_HORIZON)
    atr = compute_atr_series(df, window=5)

    return DayData(
        date_str=date_str, df=df, features=features,
        labels=labels, close=close, high=high, low=low, atr=atr,
    )


def load_all_days() -> List[DayData]:
    """Load all available minute-bar days, sorted chronologically."""
    parquet_files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
    log.info(f"Found {len(parquet_files)} parquet files in {MINUTE_BAR_DIR}")

    days = []
    for pf in parquet_files:
        dd = load_day(pf)
        if dd is not None:
            days.append(dd)
        if len(days) % 50 == 0 and len(days) > 0:
            log.info(f"  Loaded {len(days)} days...")

    log.info(f"Successfully loaded {len(days)} days "
             f"({days[0].date_str} to {days[-1].date_str})")
    return days


# ---------------------------------------------------------------------------
# Regime classification
# ---------------------------------------------------------------------------
def classify_regimes(days: List[DayData]) -> Dict[str, str]:
    """
    Classify each day as green/red/flat based on close-to-close return.
    Uses first bar close vs last bar close within each day.
    Falls back to enhanced_daily_features.parquet if available.
    """
    regime_map = {}

    # Try to load enhanced daily features first
    if ENHANCED_FEATURES.exists():
        try:
            edf = pd.read_parquet(str(ENHANCED_FEATURES))
            # Extract daily close-to-close from enhanced features
            if "es_close_to_close" in edf.columns:
                for _, row in edf.iterrows():
                    date_str = str(row.get("date", ""))[:8].replace("-", "")
                    cc = row["es_close_to_close"]
                    if cc > 0.5:
                        regime_map[date_str] = "green"
                    elif cc < -0.5:
                        regime_map[date_str] = "red"
                    else:
                        regime_map[date_str] = "flat"
                log.info(f"Loaded {len(regime_map)} regime classifications from enhanced features")
        except Exception as e:
            log.warning(f"Could not load enhanced features: {e}")

    # Compute from minute bars for any days not yet classified
    for dd in days:
        if dd.date_str not in regime_map:
            daily_return = (dd.close[-1] - dd.close[0]) / TICK_SIZE
            if daily_return > 2.0:  # >0.5 points = green
                regime_map[dd.date_str] = "green"
            elif daily_return < -2.0:
                regime_map[dd.date_str] = "red"
            else:
                regime_map[dd.date_str] = "flat"

    return regime_map


# ---------------------------------------------------------------------------
# LightGBM model training
# ---------------------------------------------------------------------------
def train_entry_model(
    train_days: List[DayData],
    val_days: Optional[List[DayData]] = None,
) -> "lgb.LGBMRegressor":
    """Train a LightGBM regression model for 30-min price change prediction."""
    import lightgbm as lgb

    # Concatenate training data
    X_list, y_list = [], []
    for dd in train_days:
        valid = np.isfinite(dd.labels)
        if valid.sum() > 10:
            X_list.append(dd.features[valid])
            y_list.append(dd.labels[valid])

    X_train = np.concatenate(X_list).astype(np.float32)
    y_train = np.concatenate(y_list).astype(np.float32)

    # Subsample if too large
    MAX_ROWS = 500_000
    if len(X_train) > MAX_ROWS:
        rng = np.random.default_rng(42)
        idx = rng.choice(len(X_train), MAX_ROWS, replace=False)
        idx.sort()
        X_train = X_train[idx]
        y_train = y_train[idx]

    # Validation set
    X_val, y_val = None, None
    if val_days:
        xv, yv = [], []
        for dd in val_days:
            valid = np.isfinite(dd.labels)
            if valid.sum() > 10:
                xv.append(dd.features[valid])
                yv.append(dd.labels[valid])
        if xv:
            X_val = np.concatenate(xv).astype(np.float32)
            y_val = np.concatenate(yv).astype(np.float32)

    params = dict(
        n_estimators=300,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.4,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=100,
        verbose=-1,
        n_jobs=-1,
        device="cpu",
        max_bin=127,
        objective="regression",
        metric="rmse",
    )

    model = lgb.LGBMRegressor(**params)

    fit_kwargs = {}
    if X_val is not None:
        fit_kwargs["eval_set"] = [(X_val, y_val)]
        fit_kwargs["callbacks"] = [lgb.early_stopping(30, verbose=False)]

    model.fit(X_train, y_train, **fit_kwargs)
    return model


# ---------------------------------------------------------------------------
# Trade simulation with MFE/MAE tracking
# ---------------------------------------------------------------------------
@dataclass
class Trade:
    date_str: str
    entry_bar: int
    entry_price: float
    direction: int       # +1 long, -1 short
    prediction: float    # model prediction in ticks
    pred_zscore: float   # z-scored prediction
    atr_at_entry: float  # ATR in ticks at entry time

    # Filled after simulation
    exit_bar: int = 0
    exit_price: float = 0.0
    exit_reason: str = ""
    pnl_ticks: float = 0.0
    pnl_net_ticks: float = 0.0  # after costs
    hold_time: int = 0          # bars held
    mfe_ticks: float = 0.0      # max favorable excursion
    mae_ticks: float = 0.0      # max adverse excursion


def simulate_trades_fixed(
    day: DayData,
    predictions: np.ndarray,
    tp_ticks: float,
    sl_ticks: float,
    conf_threshold: float = 0.0,
) -> List[Trade]:
    """
    Simulate trades with FIXED TP/SL.
    Entry: when |pred_zscore| > conf_threshold.
    """
    trades = []
    close = day.close
    high = day.high
    low = day.low
    n = len(close)

    # Z-score predictions within this day
    valid = np.isfinite(predictions) & (predictions != 0)
    if valid.sum() < 10:
        return trades

    pred_mean = np.nanmean(predictions[valid])
    pred_std = np.nanstd(predictions[valid])
    if pred_std < 1e-8:
        return trades

    zscores = (predictions - pred_mean) / pred_std

    # Minimum bar index to have features computed
    min_bar = 30

    in_trade = False
    cooldown = 0

    for i in range(min_bar, n - HOLD_BARS):
        if cooldown > 0:
            cooldown -= 1
            continue

        if in_trade:
            continue

        if not np.isfinite(predictions[i]):
            continue

        z = zscores[i]
        if abs(z) < conf_threshold:
            continue

        direction = 1 if predictions[i] > 0 else -1
        entry_price = close[i]
        atr_val = day.atr[i] if np.isfinite(day.atr[i]) else 20.0

        trade = Trade(
            date_str=day.date_str,
            entry_bar=i,
            entry_price=entry_price,
            direction=direction,
            prediction=float(predictions[i]),
            pred_zscore=float(z),
            atr_at_entry=float(atr_val),
        )

        # Simulate the trade bar-by-bar
        mfe = 0.0
        mae = 0.0
        exited = False

        for j in range(1, HOLD_BARS + 1):
            bar_idx = i + j
            if bar_idx >= n:
                break

            if direction == 1:
                favorable = (high[bar_idx] - entry_price) / TICK_SIZE
                adverse = (entry_price - low[bar_idx]) / TICK_SIZE
            else:
                favorable = (entry_price - low[bar_idx]) / TICK_SIZE
                adverse = (high[bar_idx] - entry_price) / TICK_SIZE

            mfe = max(mfe, favorable)
            mae = max(mae, adverse)

            # Check SL hit (using close as proxy; in reality intrabar)
            unrealized = direction * (close[bar_idx] - entry_price) / TICK_SIZE
            if adverse >= sl_ticks:
                trade.exit_bar = bar_idx
                trade.exit_price = entry_price - direction * sl_ticks * TICK_SIZE
                trade.exit_reason = "stop_loss"
                trade.pnl_ticks = -sl_ticks
                exited = True
                break

            if favorable >= tp_ticks:
                trade.exit_bar = bar_idx
                trade.exit_price = entry_price + direction * tp_ticks * TICK_SIZE
                trade.exit_reason = "take_profit"
                trade.pnl_ticks = tp_ticks
                exited = True
                break

        if not exited:
            # Exit at end of hold period
            last_bar = min(i + HOLD_BARS, n - 1)
            trade.exit_bar = last_bar
            trade.exit_price = close[last_bar]
            trade.exit_reason = "timeout"
            trade.pnl_ticks = direction * (close[last_bar] - entry_price) / TICK_SIZE

        trade.hold_time = trade.exit_bar - trade.entry_bar
        trade.mfe_ticks = mfe
        trade.mae_ticks = mae
        trade.pnl_net_ticks = trade.pnl_ticks - TOTAL_RT_COST_TICKS

        trades.append(trade)
        in_trade = False
        cooldown = max(1, trade.hold_time)  # avoid overlapping trades

    return trades


def simulate_trades_adaptive(
    day: DayData,
    predictions: np.ndarray,
    sl_atr_mult: float,
    tp_atr_mult: float,
    conf_threshold: float = 0.0,
) -> List[Trade]:
    """
    Simulate trades with VOLATILITY-ADAPTIVE TP/SL.
    SL = sl_atr_mult * ATR, TP = tp_atr_mult * ATR.
    """
    trades = []
    close = day.close
    high = day.high
    low = day.low
    n = len(close)

    valid = np.isfinite(predictions) & (predictions != 0)
    if valid.sum() < 10:
        return trades

    pred_mean = np.nanmean(predictions[valid])
    pred_std = np.nanstd(predictions[valid])
    if pred_std < 1e-8:
        return trades

    zscores = (predictions - pred_mean) / pred_std
    min_bar = 30
    cooldown = 0

    for i in range(min_bar, n - HOLD_BARS):
        if cooldown > 0:
            cooldown -= 1
            continue

        if not np.isfinite(predictions[i]):
            continue

        z = zscores[i]
        if abs(z) < conf_threshold:
            continue

        direction = 1 if predictions[i] > 0 else -1
        entry_price = close[i]

        atr_val = day.atr[i]
        if not np.isfinite(atr_val) or atr_val < 2.0:
            atr_val = 20.0  # fallback

        sl_ticks = sl_atr_mult * atr_val
        tp_ticks = tp_atr_mult * atr_val

        # Floor/cap SL and TP
        sl_ticks = max(4.0, min(sl_ticks, 40.0))
        tp_ticks = max(4.0, min(tp_ticks, 60.0))

        trade = Trade(
            date_str=day.date_str,
            entry_bar=i,
            entry_price=entry_price,
            direction=direction,
            prediction=float(predictions[i]),
            pred_zscore=float(z),
            atr_at_entry=float(atr_val),
        )

        mfe = 0.0
        mae = 0.0
        exited = False

        for j in range(1, HOLD_BARS + 1):
            bar_idx = i + j
            if bar_idx >= n:
                break

            if direction == 1:
                favorable = (high[bar_idx] - entry_price) / TICK_SIZE
                adverse = (entry_price - low[bar_idx]) / TICK_SIZE
            else:
                favorable = (entry_price - low[bar_idx]) / TICK_SIZE
                adverse = (high[bar_idx] - entry_price) / TICK_SIZE

            mfe = max(mfe, favorable)
            mae = max(mae, adverse)

            if adverse >= sl_ticks:
                trade.exit_bar = bar_idx
                trade.exit_price = entry_price - direction * sl_ticks * TICK_SIZE
                trade.exit_reason = "stop_loss"
                trade.pnl_ticks = -sl_ticks
                exited = True
                break

            if favorable >= tp_ticks:
                trade.exit_bar = bar_idx
                trade.exit_price = entry_price + direction * tp_ticks * TICK_SIZE
                trade.exit_reason = "take_profit"
                trade.pnl_ticks = tp_ticks
                exited = True
                break

        if not exited:
            last_bar = min(i + HOLD_BARS, n - 1)
            trade.exit_bar = last_bar
            trade.exit_price = close[last_bar]
            trade.exit_reason = "timeout"
            trade.pnl_ticks = direction * (close[last_bar] - entry_price) / TICK_SIZE

        trade.hold_time = trade.exit_bar - trade.entry_bar
        trade.mfe_ticks = mfe
        trade.mae_ticks = mae
        trade.pnl_net_ticks = trade.pnl_ticks - TOTAL_RT_COST_TICKS

        trades.append(trade)
        cooldown = max(1, trade.hold_time)

    return trades


def simulate_trades_percentile(
    day: DayData,
    predictions: np.ndarray,
    mfe_distribution: np.ndarray,
    mae_distribution: np.ndarray,
    sl_pct: float = 25,
    tp_pct: float = 75,
    conf_threshold: float = 0.0,
) -> List[Trade]:
    """
    Simulate trades with PERCENTILE-BASED TP/SL.
    SL = p(sl_pct) of historical MAE, TP = p(tp_pct) of historical MFE.
    """
    sl_ticks = max(4.0, np.percentile(mae_distribution, sl_pct)) if len(mae_distribution) > 10 else 10.0
    tp_ticks = max(4.0, np.percentile(mfe_distribution, tp_pct)) if len(mfe_distribution) > 10 else 20.0

    return simulate_trades_fixed(day, predictions, tp_ticks, sl_ticks, conf_threshold)


# ---------------------------------------------------------------------------
# Phase 2: Mid-Trade Management
# ---------------------------------------------------------------------------
@dataclass
class MidTradeState:
    """State of a trade at a given minute bar during the trade."""
    unrealized_pnl: float      # ticks from entry
    time_in_trade: int         # bars since entry
    time_remaining: int        # bars until hold timeout
    pnl_velocity: float        # 1-bar change in unrealized P&L
    ofi_aligned: float         # OFI alignment with trade direction
    volume_ratio: float        # current volume vs entry bar volume
    price_momentum_1m: float   # 1-min return aligned with direction
    price_momentum_5m: float   # 5-min return aligned with direction
    mfe_so_far: float          # max favorable excursion so far
    mae_so_far: float          # max adverse excursion so far
    pnl_to_mfe_ratio: float    # current P&L / MFE (how much given back)
    spread_change: float       # spread vs entry spread
    atr_at_entry: float        # ATR context
    pred_zscore: float         # original prediction confidence


def build_midtrade_features(
    day: DayData,
    trade: Trade,
    bar_offset: int,
) -> Optional[MidTradeState]:
    """Build mid-trade state features at a specific bar during the trade."""
    cur_bar = trade.entry_bar + bar_offset
    if cur_bar >= len(day.close) or bar_offset < 1:
        return None

    close = day.close
    direction = trade.direction
    entry_price = trade.entry_price

    unrealized = direction * (close[cur_bar] - entry_price) / TICK_SIZE

    # P&L velocity
    if bar_offset >= 2:
        prev_pnl = direction * (close[cur_bar - 1] - entry_price) / TICK_SIZE
        pnl_velocity = unrealized - prev_pnl
    else:
        pnl_velocity = unrealized

    # MFE/MAE so far
    mfe_so_far = 0.0
    mae_so_far = 0.0
    for k in range(1, bar_offset + 1):
        bi = trade.entry_bar + k
        if bi >= len(close):
            break
        if direction == 1:
            fav = (day.high[bi] - entry_price) / TICK_SIZE
            adv = (entry_price - day.low[bi]) / TICK_SIZE
        else:
            fav = (entry_price - day.low[bi]) / TICK_SIZE
            adv = (day.high[bi] - entry_price) / TICK_SIZE
        mfe_so_far = max(mfe_so_far, fav)
        mae_so_far = max(mae_so_far, adv)

    pnl_to_mfe = unrealized / (mfe_so_far + 1e-8) if mfe_so_far > 0 else 0.0

    # OFI alignment
    ofi_val = day.df["ofi_1min"].iloc[cur_bar] if cur_bar < len(day.df) else 0
    ofi_aligned = float(ofi_val * direction)

    # Volume ratio
    entry_vol = max(1, day.df["volume"].iloc[trade.entry_bar])
    cur_vol = day.df["volume"].iloc[cur_bar] if cur_bar < len(day.df) else entry_vol
    volume_ratio = float(cur_vol) / float(entry_vol)

    # Price momentum
    mom_1m = 0.0
    if cur_bar >= 1:
        mom_1m = direction * (close[cur_bar] - close[cur_bar - 1]) / TICK_SIZE

    mom_5m = 0.0
    if cur_bar >= 5:
        mom_5m = direction * (close[cur_bar] - close[cur_bar - 5]) / TICK_SIZE

    # Spread change
    entry_spread = day.df["spread_mean"].iloc[trade.entry_bar]
    cur_spread = day.df["spread_mean"].iloc[cur_bar] if cur_bar < len(day.df) else entry_spread
    spread_change = float(cur_spread - entry_spread)

    return MidTradeState(
        unrealized_pnl=unrealized,
        time_in_trade=bar_offset,
        time_remaining=HOLD_BARS - bar_offset,
        pnl_velocity=pnl_velocity,
        ofi_aligned=ofi_aligned,
        volume_ratio=volume_ratio,
        price_momentum_1m=mom_1m,
        price_momentum_5m=mom_5m,
        mfe_so_far=mfe_so_far,
        mae_so_far=mae_so_far,
        pnl_to_mfe_ratio=pnl_to_mfe,
        spread_change=spread_change,
        atr_at_entry=trade.atr_at_entry,
        pred_zscore=trade.pred_zscore,
    )


MIDTRADE_FEATURE_NAMES = [
    "unrealized_pnl", "time_in_trade", "time_remaining", "pnl_velocity",
    "ofi_aligned", "volume_ratio", "price_momentum_1m", "price_momentum_5m",
    "mfe_so_far", "mae_so_far", "pnl_to_mfe_ratio", "spread_change",
    "atr_at_entry", "pred_zscore",
]


def midtrade_state_to_array(state: MidTradeState) -> np.ndarray:
    """Convert MidTradeState to feature array."""
    return np.array([
        state.unrealized_pnl, state.time_in_trade, state.time_remaining,
        state.pnl_velocity, state.ofi_aligned, state.volume_ratio,
        state.price_momentum_1m, state.price_momentum_5m,
        state.mfe_so_far, state.mae_so_far, state.pnl_to_mfe_ratio,
        state.spread_change, state.atr_at_entry, state.pred_zscore,
    ], dtype=np.float32)


def build_midtrade_dataset(
    days_data: Dict[str, DayData],
    trades: List[Trade],
    check_interval: int = 1,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Build training dataset for mid-trade exit classifier.
    Label: 1 if trade ends profitable (net of costs), 0 otherwise.
    Sample at every `check_interval` bars during each trade.
    """
    X_list, y_list = [], []

    for trade in trades:
        if trade.date_str not in days_data:
            continue
        dd = days_data[trade.date_str]

        # Label: will this trade be profitable from this point?
        final_pnl_net = trade.pnl_net_ticks

        for offset in range(check_interval, trade.hold_time, check_interval):
            state = build_midtrade_features(dd, trade, offset)
            if state is None:
                continue

            # Remaining P&L from this point
            cur_bar = trade.entry_bar + offset
            if cur_bar >= len(dd.close):
                continue
            remaining_pnl = trade.direction * (dd.close[trade.exit_bar] - dd.close[cur_bar]) / TICK_SIZE
            remaining_pnl_net = remaining_pnl - EXIT_COST_TICKS  # cost of exiting now vs later

            label = 1.0 if remaining_pnl_net > 0 else 0.0

            X_list.append(midtrade_state_to_array(state))
            y_list.append(label)

    if not X_list:
        return np.array([]).reshape(0, len(MIDTRADE_FEATURE_NAMES)), np.array([])

    return np.array(X_list, dtype=np.float32), np.array(y_list, dtype=np.float32)


def train_exit_classifier(X: np.ndarray, y: np.ndarray) -> Optional["lgb.LGBMClassifier"]:
    """Train LightGBM classifier for mid-trade exit decisions."""
    import lightgbm as lgb

    if len(X) < 100:
        return None

    params = dict(
        n_estimators=200,
        max_depth=4,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.5,
        reg_alpha=0.1,
        reg_lambda=1.0,
        min_child_samples=50,
        verbose=-1,
        n_jobs=-1,
        objective="binary",
        metric="auc",
        is_unbalance=True,
    )

    model = lgb.LGBMClassifier(**params)

    # Simple train/val split (last 20%)
    split = int(len(X) * 0.8)
    X_tr, X_val = X[:split], X[split:]
    y_tr, y_val = y[:split], y[split:]

    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        callbacks=[lgb.early_stopping(20, verbose=False)],
    )
    return model


def simulate_with_midtrade_exit(
    day: DayData,
    predictions: np.ndarray,
    exit_model: "lgb.LGBMClassifier",
    sl_atr_mult: float,
    tp_atr_mult: float,
    exit_threshold: float = 0.35,
    conf_threshold: float = 0.0,
    check_interval: int = 3,
) -> List[Trade]:
    """
    Simulate trades with adaptive SL/TP AND mid-trade exit classifier.
    If exit classifier confidence < exit_threshold at any check bar, exit immediately.
    """
    trades = []
    close = day.close
    high = day.high
    low = day.low
    n = len(close)

    valid = np.isfinite(predictions) & (predictions != 0)
    if valid.sum() < 10:
        return trades

    pred_mean = np.nanmean(predictions[valid])
    pred_std = np.nanstd(predictions[valid])
    if pred_std < 1e-8:
        return trades

    zscores = (predictions - pred_mean) / pred_std
    min_bar = 30
    cooldown = 0

    for i in range(min_bar, n - HOLD_BARS):
        if cooldown > 0:
            cooldown -= 1
            continue

        if not np.isfinite(predictions[i]):
            continue

        z = zscores[i]
        if abs(z) < conf_threshold:
            continue

        direction = 1 if predictions[i] > 0 else -1
        entry_price = close[i]

        atr_val = day.atr[i]
        if not np.isfinite(atr_val) or atr_val < 2.0:
            atr_val = 20.0

        sl_ticks = max(4.0, min(sl_atr_mult * atr_val, 40.0))
        tp_ticks = max(4.0, min(tp_atr_mult * atr_val, 60.0))

        trade = Trade(
            date_str=day.date_str,
            entry_bar=i,
            entry_price=entry_price,
            direction=direction,
            prediction=float(predictions[i]),
            pred_zscore=float(z),
            atr_at_entry=float(atr_val),
        )

        mfe = 0.0
        mae = 0.0
        exited = False

        for j in range(1, HOLD_BARS + 1):
            bar_idx = i + j
            if bar_idx >= n:
                break

            if direction == 1:
                favorable = (high[bar_idx] - entry_price) / TICK_SIZE
                adverse = (entry_price - low[bar_idx]) / TICK_SIZE
            else:
                favorable = (entry_price - low[bar_idx]) / TICK_SIZE
                adverse = (high[bar_idx] - entry_price) / TICK_SIZE

            mfe = max(mfe, favorable)
            mae = max(mae, adverse)

            # SL check
            if adverse >= sl_ticks:
                trade.exit_bar = bar_idx
                trade.exit_price = entry_price - direction * sl_ticks * TICK_SIZE
                trade.exit_reason = "stop_loss"
                trade.pnl_ticks = -sl_ticks
                exited = True
                break

            # TP check
            if favorable >= tp_ticks:
                trade.exit_bar = bar_idx
                trade.exit_price = entry_price + direction * tp_ticks * TICK_SIZE
                trade.exit_reason = "take_profit"
                trade.pnl_ticks = tp_ticks
                exited = True
                break

            # Mid-trade exit check
            if j >= check_interval and j % check_interval == 0:
                state = build_midtrade_features(day, trade, j)
                if state is not None:
                    feat = midtrade_state_to_array(state).reshape(1, -1)
                    prob_profit = exit_model.predict_proba(feat)[0, 1]
                    if prob_profit < exit_threshold:
                        trade.exit_bar = bar_idx
                        trade.exit_price = close[bar_idx]
                        trade.exit_reason = "exit_classifier"
                        trade.pnl_ticks = direction * (close[bar_idx] - entry_price) / TICK_SIZE
                        exited = True
                        break

        if not exited:
            last_bar = min(i + HOLD_BARS, n - 1)
            trade.exit_bar = last_bar
            trade.exit_price = close[last_bar]
            trade.exit_reason = "timeout"
            trade.pnl_ticks = direction * (close[last_bar] - entry_price) / TICK_SIZE

        trade.hold_time = trade.exit_bar - trade.entry_bar
        trade.mfe_ticks = mfe
        trade.mae_ticks = mae
        trade.pnl_net_ticks = trade.pnl_ticks - TOTAL_RT_COST_TICKS

        trades.append(trade)
        cooldown = max(1, trade.hold_time)

    return trades


# ---------------------------------------------------------------------------
# Metrics computation
# ---------------------------------------------------------------------------
def compute_metrics(trades: List[Trade]) -> Dict:
    """Compute comprehensive trading metrics."""
    if not trades:
        return {
            "n_trades": 0, "win_rate": 0, "sharpe": 0, "sortino": 0,
            "profit_factor": 0, "avg_pnl_net": 0, "total_pnl_net": 0,
            "avg_mfe": 0, "avg_mae": 0, "max_dd_ticks": 0,
        }

    pnls = np.array([t.pnl_net_ticks for t in trades])
    winners = pnls > 0
    losers = pnls < 0

    n_trades = len(trades)
    win_rate = float(winners.sum()) / n_trades if n_trades > 0 else 0.0

    total_pnl = float(pnls.sum())
    avg_pnl = float(pnls.mean())

    # Sharpe (annualized, assume ~252 trading days)
    if pnls.std() > 1e-8:
        daily_sharpe = pnls.mean() / pnls.std()
        # Rough annualization: multiply by sqrt(trades_per_year / n_trades * 252)
        sharpe = daily_sharpe * np.sqrt(min(252, n_trades))
    else:
        sharpe = 0.0

    # Sortino (downside deviation)
    downside = pnls[pnls < 0]
    if len(downside) > 1:
        downside_std = np.std(downside)
        if downside_std > 1e-8:
            sortino = (pnls.mean() / downside_std) * np.sqrt(min(252, n_trades))
        else:
            sortino = 0.0
    else:
        sortino = sharpe  # no losers = same as sharpe

    # Profit factor
    gross_profit = float(pnls[winners].sum()) if winners.any() else 0.0
    gross_loss = float(np.abs(pnls[losers].sum())) if losers.any() else 1e-8
    profit_factor = gross_profit / gross_loss if gross_loss > 1e-8 else (10.0 if gross_profit > 0 else 0.0)

    # MFE/MAE
    avg_mfe = float(np.mean([t.mfe_ticks for t in trades]))
    avg_mae = float(np.mean([t.mae_ticks for t in trades]))

    # Max drawdown
    cumulative = np.cumsum(pnls)
    running_max = np.maximum.accumulate(cumulative)
    drawdowns = running_max - cumulative
    max_dd = float(drawdowns.max()) if len(drawdowns) > 0 else 0.0

    # Exit reason breakdown
    exit_reasons = {}
    for t in trades:
        exit_reasons[t.exit_reason] = exit_reasons.get(t.exit_reason, 0) + 1

    # Long vs short breakdown
    long_trades = [t for t in trades if t.direction == 1]
    short_trades = [t for t in trades if t.direction == -1]
    long_wr = float(np.mean([t.pnl_net_ticks > 0 for t in long_trades])) if long_trades else 0.0
    short_wr = float(np.mean([t.pnl_net_ticks > 0 for t in short_trades])) if short_trades else 0.0

    return {
        "n_trades": n_trades,
        "win_rate": round(win_rate, 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(profit_factor, 3),
        "avg_pnl_net": round(avg_pnl, 3),
        "total_pnl_net": round(total_pnl, 2),
        "total_pnl_dollars": round(total_pnl * TICK_VALUE, 2),
        "avg_mfe": round(avg_mfe, 2),
        "avg_mae": round(avg_mae, 2),
        "max_dd_ticks": round(max_dd, 2),
        "avg_hold_time": round(float(np.mean([t.hold_time for t in trades])), 1),
        "exit_reasons": exit_reasons,
        "n_long": len(long_trades),
        "n_short": len(short_trades),
        "long_wr": round(long_wr, 4),
        "short_wr": round(short_wr, 4),
    }


def compute_regime_metrics(
    trades: List[Trade],
    regime_map: Dict[str, str],
) -> Dict:
    """Compute per-regime metrics and HC #428 gate check."""
    regime_trades = {"green": [], "red": [], "flat": []}
    for t in trades:
        regime = regime_map.get(t.date_str, "flat")
        regime_trades[regime].append(t)

    regime_metrics = {}
    for regime, rtrades in regime_trades.items():
        if rtrades:
            regime_metrics[regime] = compute_metrics(rtrades)
        else:
            regime_metrics[regime] = {"n_trades": 0, "sharpe": 0, "win_rate": 0}

    # HC #428 regime gate
    sharpe_green = regime_metrics.get("green", {}).get("sharpe", 0)
    sharpe_red = regime_metrics.get("red", {}).get("sharpe", 0)
    max_sharpe = max(abs(sharpe_green), abs(sharpe_red))

    if max_sharpe > 0:
        regime_divergence = abs(sharpe_green - sharpe_red) / max_sharpe
    else:
        regime_divergence = 0.0

    passes_regime_gate = regime_divergence <= 0.50

    return {
        "per_regime": regime_metrics,
        "regime_divergence": round(regime_divergence, 4),
        "passes_hc428_gate": passes_regime_gate,
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
    }


# ---------------------------------------------------------------------------
# Per-day metrics for granular reporting
# ---------------------------------------------------------------------------
def compute_per_day_metrics(trades: List[Trade]) -> List[Dict]:
    """Compute metrics per OOT day."""
    from collections import defaultdict
    day_trades = defaultdict(list)
    for t in trades:
        day_trades[t.date_str].append(t)

    per_day = []
    for date_str in sorted(day_trades.keys()):
        dt = day_trades[date_str]
        pnls = [t.pnl_net_ticks for t in dt]
        per_day.append({
            "date": date_str,
            "n_trades": len(dt),
            "total_pnl_ticks": round(sum(pnls), 2),
            "win_rate": round(float(np.mean([p > 0 for p in pnls])), 4) if pnls else 0,
            "avg_pnl": round(float(np.mean(pnls)), 3) if pnls else 0,
        })
    return per_day


# ---------------------------------------------------------------------------
# Main walk-forward engine
# ---------------------------------------------------------------------------
def run_walkforward(
    days: List[DayData],
    train_window: int = TRAIN_DAYS,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Sliding walk-forward: train on [i : i+train_window], predict day i+train_window.
    Returns dict of date_str -> predictions array, and list of OOT dates.
    """
    import lightgbm as lgb

    n_days = len(days)
    if n_days < train_window + 1:
        log.error(f"Not enough days for WF: {n_days} < {train_window + 1}")
        return {}, []

    predictions = {}
    oot_dates = []
    n_folds = n_days - train_window

    log.info(f"Starting walk-forward: {n_folds} folds, "
             f"train={train_window}d, slide=1d")

    for fold_idx in range(n_folds):
        train_start = fold_idx
        train_end = fold_idx + train_window
        test_idx = train_end

        if test_idx >= n_days:
            break

        train_days_data = days[train_start:train_end]
        test_day = days[test_idx]

        # Use last 5 days of training window as validation
        val_split = max(1, train_window // 12)  # ~5 days
        train_subset = train_days_data[:-val_split]
        val_subset = train_days_data[-val_split:]

        try:
            model = train_entry_model(train_subset, val_subset)
        except Exception as e:
            log.warning(f"Fold {fold_idx} ({test_day.date_str}) training failed: {e}")
            continue

        # Predict on test day
        test_preds = model.predict(test_day.features.astype(np.float32))
        predictions[test_day.date_str] = test_preds.astype(np.float32)
        oot_dates.append(test_day.date_str)

        if (fold_idx + 1) % 20 == 0:
            # Quick IC check
            valid = np.isfinite(test_day.labels) & np.isfinite(test_preds)
            if valid.sum() > 10:
                ic, _ = spearmanr(test_preds[valid], test_day.labels[valid])
            else:
                ic = 0.0
            log.info(f"  Fold {fold_idx + 1}/{n_folds} "
                     f"({test_day.date_str}) IC={ic:.4f}")

        # Memory cleanup
        del model
        if (fold_idx + 1) % 50 == 0:
            gc.collect()

    log.info(f"Walk-forward complete: {len(predictions)} OOT days")
    return predictions, oot_dates


# ---------------------------------------------------------------------------
# Phase 1: Dynamic SL/TP Sweep
# ---------------------------------------------------------------------------
def phase1_dynamic_sltp(
    days: List[DayData],
    predictions: Dict[str, np.ndarray],
    oot_dates: List[str],
    quick: bool = False,
) -> Dict:
    """
    Phase 1: Test fixed, ATR-adaptive, and percentile-based SL/TP configs.
    """
    log.info("=" * 70)
    log.info("PHASE 1: Dynamic SL/TP Optimization")
    log.info("=" * 70)

    days_by_date = {d.date_str: d for d in days}

    # Collect MFE/MAE from a baseline run (fixed SL=20, TP=20, no conf gate)
    log.info("Collecting baseline MFE/MAE distributions...")
    baseline_trades = []
    for date_str in oot_dates:
        if date_str not in days_by_date or date_str not in predictions:
            continue
        dd = days_by_date[date_str]
        preds = predictions[date_str]
        trades = simulate_trades_fixed(dd, preds, tp_ticks=40.0, sl_ticks=40.0, conf_threshold=0.0)
        baseline_trades.extend(trades)

    if not baseline_trades:
        log.error("No baseline trades generated!")
        return {}

    mfe_dist = np.array([t.mfe_ticks for t in baseline_trades])
    mae_dist = np.array([t.mae_ticks for t in baseline_trades])
    log.info(f"Baseline: {len(baseline_trades)} trades, "
             f"MFE p50={np.median(mfe_dist):.1f} p75={np.percentile(mfe_dist, 75):.1f} "
             f"p90={np.percentile(mfe_dist, 90):.1f}, "
             f"MAE p25={np.percentile(mae_dist, 25):.1f} p50={np.median(mae_dist):.1f} "
             f"p75={np.percentile(mae_dist, 75):.1f}")

    # Build config grid
    configs = []

    # --- Fixed SL/TP configs ---
    fixed_sls = [8, 10, 15, 20] if not quick else [10, 15]
    fixed_tps = [10, 15, 20, 30] if not quick else [15, 20]
    conf_thresholds = [0.0, 0.3, 0.5, 0.7] if not quick else [0.0, 0.5]

    for sl in fixed_sls:
        for tp in fixed_tps:
            for ct in conf_thresholds:
                configs.append({
                    "type": "fixed",
                    "sl_ticks": sl,
                    "tp_ticks": tp,
                    "conf_threshold": ct,
                    "name": f"fixed_SL{sl}_TP{tp}_conf{ct}",
                })

    # --- ATR-adaptive configs ---
    sl_mults = [0.5, 0.75, 1.0, 1.5] if not quick else [0.75, 1.0]
    tp_mults = [1.0, 1.5, 2.0, 2.5] if not quick else [1.5, 2.0]

    for sl_m in sl_mults:
        for tp_m in tp_mults:
            for ct in conf_thresholds:
                configs.append({
                    "type": "atr_adaptive",
                    "sl_atr_mult": sl_m,
                    "tp_atr_mult": tp_m,
                    "conf_threshold": ct,
                    "name": f"atr_SL{sl_m}_TP{tp_m}_conf{ct}",
                })

    # --- Percentile-based configs ---
    pct_configs = [(25, 75), (30, 70), (20, 80), (25, 90)] if not quick else [(25, 75)]
    for sl_pct, tp_pct in pct_configs:
        for ct in conf_thresholds:
            configs.append({
                "type": "percentile",
                "sl_pct": sl_pct,
                "tp_pct": tp_pct,
                "conf_threshold": ct,
                "name": f"pct_SL_p{sl_pct}_TP_p{tp_pct}_conf{ct}",
                "mfe_dist": mfe_dist,
                "mae_dist": mae_dist,
            })

    log.info(f"Testing {len(configs)} SL/TP configurations...")

    results = []
    for ci, config in enumerate(configs):
        all_trades = []

        for date_str in oot_dates:
            if date_str not in days_by_date or date_str not in predictions:
                continue
            dd = days_by_date[date_str]
            preds = predictions[date_str]

            ct = config["conf_threshold"]

            if config["type"] == "fixed":
                trades = simulate_trades_fixed(dd, preds, config["tp_ticks"],
                                                config["sl_ticks"], ct)
            elif config["type"] == "atr_adaptive":
                trades = simulate_trades_adaptive(dd, preds, config["sl_atr_mult"],
                                                   config["tp_atr_mult"], ct)
            elif config["type"] == "percentile":
                trades = simulate_trades_percentile(
                    dd, preds, config["mfe_dist"], config["mae_dist"],
                    config.get("sl_pct", 25), config.get("tp_pct", 75), ct)

            all_trades.extend(trades)

        metrics = compute_metrics(all_trades)
        config_clean = {k: v for k, v in config.items()
                       if k not in ("mfe_dist", "mae_dist")}
        result = {"config": config_clean, "metrics": metrics}
        results.append(result)

        if (ci + 1) % 10 == 0:
            log.info(f"  Config {ci + 1}/{len(configs)}: {config['name']} "
                     f"-> WR={metrics['win_rate']:.1%} PF={metrics['profit_factor']:.2f} "
                     f"Sharpe={metrics['sharpe']:.2f} N={metrics['n_trades']}")

    # Sort by Sharpe
    results.sort(key=lambda r: r["metrics"]["sharpe"], reverse=True)

    # Log top 10
    log.info("\n" + "=" * 70)
    log.info("PHASE 1 RESULTS — Top 10 Configs by Sharpe")
    log.info("=" * 70)
    for i, r in enumerate(results[:10]):
        m = r["metrics"]
        log.info(f"  #{i + 1}: {r['config']['name']}")
        log.info(f"      Sharpe={m['sharpe']:.2f} Sortino={m['sortino']:.2f} "
                 f"PF={m['profit_factor']:.2f} WR={m['win_rate']:.1%} "
                 f"N={m['n_trades']} Total=${m.get('total_pnl_dollars', 0):.0f}")

    return {
        "all_results": results,
        "mfe_distribution": {
            "p25": float(np.percentile(mfe_dist, 25)),
            "p50": float(np.median(mfe_dist)),
            "p75": float(np.percentile(mfe_dist, 75)),
            "p90": float(np.percentile(mfe_dist, 90)),
        },
        "mae_distribution": {
            "p25": float(np.percentile(mae_dist, 25)),
            "p50": float(np.median(mae_dist)),
            "p75": float(np.percentile(mae_dist, 75)),
            "p90": float(np.percentile(mae_dist, 90)),
        },
        "n_baseline_trades": len(baseline_trades),
    }


# ---------------------------------------------------------------------------
# Phase 2: Mid-Trade Management
# ---------------------------------------------------------------------------
def phase2_midtrade_management(
    days: List[DayData],
    predictions: Dict[str, np.ndarray],
    oot_dates: List[str],
    best_config: Dict,
    quick: bool = False,
) -> Dict:
    """
    Phase 2: Train mid-trade exit classifier and test early exit rules.
    """
    log.info("\n" + "=" * 70)
    log.info("PHASE 2: Mid-Trade Management")
    log.info("=" * 70)

    days_by_date = {d.date_str: d for d in days}

    # Split OOT dates: first 70% for training exit model, last 30% for testing
    n_oot = len(oot_dates)
    split_idx = int(n_oot * 0.7)
    train_oot = oot_dates[:split_idx]
    test_oot = oot_dates[split_idx:]

    log.info(f"Exit model training: {len(train_oot)} days, testing: {len(test_oot)} days")

    # Extract best config params
    cfg = best_config.get("config", {})
    cfg_type = cfg.get("type", "fixed")
    conf_threshold = cfg.get("conf_threshold", 0.0)

    # Generate training trades using best Phase 1 config
    log.info("Generating training trades for exit classifier...")
    train_trades = []
    for date_str in train_oot:
        if date_str not in days_by_date or date_str not in predictions:
            continue
        dd = days_by_date[date_str]
        preds = predictions[date_str]

        if cfg_type == "atr_adaptive":
            trades = simulate_trades_adaptive(
                dd, preds, cfg.get("sl_atr_mult", 1.0),
                cfg.get("tp_atr_mult", 2.0), conf_threshold)
        elif cfg_type == "percentile":
            # Use reasonable defaults since we don't have the distributions here
            trades = simulate_trades_fixed(dd, preds, 20.0, 15.0, conf_threshold)
        else:
            trades = simulate_trades_fixed(
                dd, preds, cfg.get("tp_ticks", 20.0),
                cfg.get("sl_ticks", 15.0), conf_threshold)

        train_trades.extend(trades)

    log.info(f"Training trades: {len(train_trades)}")

    if len(train_trades) < 50:
        log.warning("Too few training trades for exit classifier, skipping Phase 2")
        return {"skipped": True, "reason": "insufficient_trades"}

    # Build mid-trade training dataset
    log.info("Building mid-trade feature dataset...")
    X_exit, y_exit = build_midtrade_dataset(days_by_date, train_trades, check_interval=2)
    log.info(f"Exit dataset: {X_exit.shape[0]} samples, "
             f"positive rate: {y_exit.mean():.3f}")

    if len(X_exit) < 100:
        log.warning("Too few mid-trade samples, skipping Phase 2")
        return {"skipped": True, "reason": "insufficient_midtrade_samples"}

    # Train exit classifier
    log.info("Training exit classifier...")
    exit_model = train_exit_classifier(X_exit, y_exit)
    if exit_model is None:
        return {"skipped": True, "reason": "training_failed"}

    # Test with different exit thresholds
    exit_thresholds = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50] if not quick else [0.30, 0.40]
    check_intervals = [2, 3, 5] if not quick else [3]

    results = []

    # Baseline: same config WITHOUT exit classifier
    baseline_trades = []
    for date_str in test_oot:
        if date_str not in days_by_date or date_str not in predictions:
            continue
        dd = days_by_date[date_str]
        preds = predictions[date_str]

        if cfg_type == "atr_adaptive":
            trades = simulate_trades_adaptive(
                dd, preds, cfg.get("sl_atr_mult", 1.0),
                cfg.get("tp_atr_mult", 2.0), conf_threshold)
        else:
            trades = simulate_trades_fixed(
                dd, preds, cfg.get("tp_ticks", 20.0),
                cfg.get("sl_ticks", 15.0), conf_threshold)

        baseline_trades.extend(trades)

    baseline_metrics = compute_metrics(baseline_trades)
    results.append({
        "config": {"type": "baseline_no_exit_model"},
        "metrics": baseline_metrics,
    })
    log.info(f"Baseline (no exit model): WR={baseline_metrics['win_rate']:.1%} "
             f"PF={baseline_metrics['profit_factor']:.2f} "
             f"Sharpe={baseline_metrics['sharpe']:.2f}")

    # Test exit classifier configs
    for exit_thresh in exit_thresholds:
        for check_int in check_intervals:
            test_trades = []
            for date_str in test_oot:
                if date_str not in days_by_date or date_str not in predictions:
                    continue
                dd = days_by_date[date_str]
                preds = predictions[date_str]

                if cfg_type == "atr_adaptive":
                    trades = simulate_with_midtrade_exit(
                        dd, preds, exit_model,
                        cfg.get("sl_atr_mult", 1.0),
                        cfg.get("tp_atr_mult", 2.0),
                        exit_threshold=exit_thresh,
                        conf_threshold=conf_threshold,
                        check_interval=check_int)
                else:
                    # Wrap fixed SL/TP in adaptive with equivalent multipliers
                    sl_t = cfg.get("sl_ticks", 15.0)
                    tp_t = cfg.get("tp_ticks", 20.0)
                    # Approximate ATR mult from fixed ticks (assume avg ATR ~15)
                    trades = simulate_with_midtrade_exit(
                        dd, preds, exit_model,
                        sl_atr_mult=sl_t / 15.0,
                        tp_atr_mult=tp_t / 15.0,
                        exit_threshold=exit_thresh,
                        conf_threshold=conf_threshold,
                        check_interval=check_int)

                test_trades.extend(trades)

            metrics = compute_metrics(test_trades)
            config_name = f"exit_thresh{exit_thresh}_check{check_int}"
            results.append({
                "config": {
                    "type": "midtrade_exit",
                    "exit_threshold": exit_thresh,
                    "check_interval": check_int,
                    "name": config_name,
                },
                "metrics": metrics,
            })

            # Count early exits
            n_early = sum(1 for t in test_trades if t.exit_reason == "exit_classifier")
            pct_early = n_early / max(1, len(test_trades))
            log.info(f"  {config_name}: WR={metrics['win_rate']:.1%} "
                     f"PF={metrics['profit_factor']:.2f} "
                     f"Sharpe={metrics['sharpe']:.2f} "
                     f"N={metrics['n_trades']} "
                     f"early_exits={pct_early:.0%}")

    # Sort by Sharpe
    results.sort(key=lambda r: r["metrics"]["sharpe"], reverse=True)

    log.info("\nPhase 2 Results — Top configs by Sharpe:")
    for i, r in enumerate(results[:5]):
        m = r["metrics"]
        log.info(f"  #{i + 1}: {r['config'].get('name', r['config']['type'])} "
                 f"Sharpe={m['sharpe']:.2f} WR={m['win_rate']:.1%} "
                 f"PF={m['profit_factor']:.2f}")

    return {
        "results": results,
        "baseline_metrics": baseline_metrics,
        "exit_dataset_size": len(X_exit),
        "exit_positive_rate": float(y_exit.mean()),
    }


# ---------------------------------------------------------------------------
# Phase 3: Regime-Stratified Validation
# ---------------------------------------------------------------------------
def phase3_regime_validation(
    days: List[DayData],
    predictions: Dict[str, np.ndarray],
    oot_dates: List[str],
    top_configs: List[Dict],
    regime_map: Dict[str, str],
) -> Dict:
    """
    Phase 3: Validate top configs across regimes and apply HC #428 gate.
    """
    log.info("\n" + "=" * 70)
    log.info("PHASE 3: Regime-Stratified Validation (HC #428)")
    log.info("=" * 70)

    days_by_date = {d.date_str: d for d in days}

    # Count regime distribution
    regime_counts = {"green": 0, "red": 0, "flat": 0}
    for d in oot_dates:
        r = regime_map.get(d, "flat")
        regime_counts[r] += 1
    log.info(f"Regime distribution: {regime_counts}")

    validated_results = []

    for config_entry in top_configs:
        cfg = config_entry.get("config", {})
        cfg_type = cfg.get("type", "fixed")
        conf_threshold = cfg.get("conf_threshold", 0.0)
        config_name = cfg.get("name", str(cfg))

        all_trades = []
        for date_str in oot_dates:
            if date_str not in days_by_date or date_str not in predictions:
                continue
            dd = days_by_date[date_str]
            preds = predictions[date_str]

            if cfg_type == "atr_adaptive":
                trades = simulate_trades_adaptive(
                    dd, preds, cfg.get("sl_atr_mult", 1.0),
                    cfg.get("tp_atr_mult", 2.0), conf_threshold)
            elif cfg_type == "percentile":
                trades = simulate_trades_fixed(dd, preds, 20.0, 15.0, conf_threshold)
            else:
                trades = simulate_trades_fixed(
                    dd, preds, cfg.get("tp_ticks", 20.0),
                    cfg.get("sl_ticks", 15.0), conf_threshold)

            all_trades.extend(trades)

        if not all_trades:
            continue

        overall_metrics = compute_metrics(all_trades)
        regime_analysis = compute_regime_metrics(all_trades, regime_map)
        per_day = compute_per_day_metrics(all_trades)

        # Day concentration check (HC #344): max single-day P&L contribution
        if per_day:
            day_pnls = [d["total_pnl_ticks"] for d in per_day]
            total = sum(abs(p) for p in day_pnls)
            if total > 0:
                day_conc = max(abs(p) for p in day_pnls) / total
            else:
                day_conc = 0.0
            passes_day_conc = day_conc <= 0.70
        else:
            day_conc = 0.0
            passes_day_conc = True

        result = {
            "config": cfg,
            "config_name": config_name,
            "overall_metrics": overall_metrics,
            "regime_analysis": regime_analysis,
            "per_day_metrics": per_day,
            "day_concentration": round(day_conc, 4),
            "passes_day_conc_gate": passes_day_conc,
            "passes_regime_gate": regime_analysis["passes_hc428_gate"],
            "ACCEPTED": regime_analysis["passes_hc428_gate"] and passes_day_conc,
        }

        validated_results.append(result)

        status = "ACCEPTED" if result["ACCEPTED"] else "REJECTED"
        log.info(f"\n  Config: {config_name} -> {status}")
        log.info(f"    Overall: Sharpe={overall_metrics['sharpe']:.2f} "
                 f"WR={overall_metrics['win_rate']:.1%} "
                 f"PF={overall_metrics['profit_factor']:.2f} "
                 f"N={overall_metrics['n_trades']}")
        log.info(f"    Regime gate: divergence={regime_analysis['regime_divergence']:.3f} "
                 f"(max 0.50) -> {'PASS' if regime_analysis['passes_hc428_gate'] else 'FAIL'}")
        log.info(f"      Green: Sharpe={regime_analysis['sharpe_green']:.2f} "
                 f"Red: Sharpe={regime_analysis['sharpe_red']:.2f}")
        log.info(f"    Day conc: {day_conc:.3f} (max 0.70) -> "
                 f"{'PASS' if passes_day_conc else 'FAIL'}")

    # Sort: accepted first (by Sharpe), then rejected
    accepted = [r for r in validated_results if r["ACCEPTED"]]
    rejected = [r for r in validated_results if not r["ACCEPTED"]]
    accepted.sort(key=lambda r: r["overall_metrics"]["sharpe"], reverse=True)
    rejected.sort(key=lambda r: r["overall_metrics"]["sharpe"], reverse=True)

    log.info(f"\n{'=' * 70}")
    log.info(f"PHASE 3 SUMMARY: {len(accepted)} ACCEPTED, {len(rejected)} REJECTED "
             f"out of {len(validated_results)} configs")
    if accepted:
        best = accepted[0]
        log.info(f"BEST ACCEPTED: {best['config_name']}")
        log.info(f"  Sharpe={best['overall_metrics']['sharpe']:.2f} "
                 f"Sortino={best['overall_metrics']['sortino']:.2f} "
                 f"PF={best['overall_metrics']['profit_factor']:.2f} "
                 f"WR={best['overall_metrics']['win_rate']:.1%}")

    return {
        "accepted": accepted,
        "rejected": rejected,
        "regime_distribution": regime_counts,
        "n_accepted": len(accepted),
        "n_rejected": len(rejected),
    }


# ---------------------------------------------------------------------------
# Phase 4: MLflow Logging
# ---------------------------------------------------------------------------
def phase4_mlflow_logging(
    phase1_results: Dict,
    phase2_results: Dict,
    phase3_results: Dict,
    total_time: float,
) -> bool:
    """Log all results to MLflow."""
    log.info("\n" + "=" * 70)
    log.info("PHASE 4: MLflow Logging")
    log.info("=" * 70)

    try:
        import mlflow
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("dynamic_execution_v8")

        with mlflow.start_run(run_name=f"v8_full_{_ts}"):
            # Log parameters
            mlflow.log_param("train_days", TRAIN_DAYS)
            mlflow.log_param("slide_days", SLIDE_DAYS)
            mlflow.log_param("hold_bars", HOLD_BARS)
            mlflow.log_param("prediction_horizon", PREDICTION_HORIZON)
            mlflow.log_param("entry_cost_ticks", ENTRY_COST_TICKS)
            mlflow.log_param("exit_cost_ticks", EXIT_COST_TICKS)
            mlflow.log_param("total_rt_cost_ticks", TOTAL_RT_COST_TICKS)
            mlflow.log_param("n_features", NUM_FEATURES)

            # Phase 1 metrics
            if phase1_results.get("all_results"):
                best_p1 = phase1_results["all_results"][0]
                mlflow.log_metric("p1_best_sharpe", best_p1["metrics"]["sharpe"])
                mlflow.log_metric("p1_best_sortino", best_p1["metrics"]["sortino"])
                mlflow.log_metric("p1_best_pf", best_p1["metrics"]["profit_factor"])
                mlflow.log_metric("p1_best_wr", best_p1["metrics"]["win_rate"])
                mlflow.log_metric("p1_best_n_trades", best_p1["metrics"]["n_trades"])
                mlflow.log_param("p1_best_config", best_p1["config"]["name"])

                # MFE/MAE distributions
                for k, v in phase1_results.get("mfe_distribution", {}).items():
                    mlflow.log_metric(f"mfe_{k}", v)
                for k, v in phase1_results.get("mae_distribution", {}).items():
                    mlflow.log_metric(f"mae_{k}", v)

            # Phase 2 metrics
            if not phase2_results.get("skipped"):
                if phase2_results.get("results"):
                    best_p2 = phase2_results["results"][0]
                    mlflow.log_metric("p2_best_sharpe", best_p2["metrics"]["sharpe"])
                    mlflow.log_metric("p2_best_pf", best_p2["metrics"]["profit_factor"])
                    mlflow.log_metric("p2_best_wr", best_p2["metrics"]["win_rate"])
                    mlflow.log_param("p2_best_config",
                                     best_p2["config"].get("name", str(best_p2["config"])))

                    baseline = phase2_results.get("baseline_metrics", {})
                    mlflow.log_metric("p2_baseline_sharpe", baseline.get("sharpe", 0))
                    mlflow.log_metric("p2_sharpe_improvement",
                                     best_p2["metrics"]["sharpe"] - baseline.get("sharpe", 0))

            # Phase 3 metrics
            if phase3_results.get("accepted"):
                best_p3 = phase3_results["accepted"][0]
                mlflow.log_metric("p3_best_sharpe", best_p3["overall_metrics"]["sharpe"])
                mlflow.log_metric("p3_best_sortino", best_p3["overall_metrics"]["sortino"])
                mlflow.log_metric("p3_best_pf", best_p3["overall_metrics"]["profit_factor"])
                mlflow.log_metric("p3_best_wr", best_p3["overall_metrics"]["win_rate"])
                mlflow.log_metric("p3_regime_divergence",
                                 best_p3["regime_analysis"]["regime_divergence"])
                mlflow.log_metric("p3_day_concentration", best_p3["day_concentration"])
                mlflow.log_param("p3_best_config", best_p3["config_name"])

            mlflow.log_metric("n_accepted_configs", phase3_results.get("n_accepted", 0))
            mlflow.log_metric("n_rejected_configs", phase3_results.get("n_rejected", 0))
            mlflow.log_metric("total_runtime_minutes", total_time / 60)

            # Log artifacts
            artifacts_dir = OUTPUT_DIR / "mlflow_artifacts"
            artifacts_dir.mkdir(exist_ok=True)

            # Save summary JSON
            summary_path = artifacts_dir / "results_summary.json"
            summary = {
                "phase1_top5": phase1_results.get("all_results", [])[:5],
                "phase2_results": phase2_results if not phase2_results.get("skipped") else {"skipped": True},
                "phase3_accepted": phase3_results.get("accepted", [])[:3],
                "phase3_rejected_count": phase3_results.get("n_rejected", 0),
            }
            with open(summary_path, "w") as f:
                json.dump(summary, f, indent=2, default=str)
            mlflow.log_artifact(str(summary_path))

        log.info("MLflow logging complete")
        return True

    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")
        log.info("Results saved locally — MLflow logging is non-blocking")
        return False


# ---------------------------------------------------------------------------
# Save results to disk
# ---------------------------------------------------------------------------
def save_results(
    phase1_results: Dict,
    phase2_results: Dict,
    phase3_results: Dict,
    predictions: Dict[str, np.ndarray],
    oot_dates: List[str],
) -> None:
    """Save all results to JSON and parquet files."""
    log.info("Saving results to disk...")

    # Phase 1 results (JSON)
    p1_path = OUTPUT_DIR / "phase1_sltp_results.json"
    p1_save = {
        "all_results": phase1_results.get("all_results", []),
        "mfe_distribution": phase1_results.get("mfe_distribution", {}),
        "mae_distribution": phase1_results.get("mae_distribution", {}),
    }
    with open(p1_path, "w") as f:
        json.dump(p1_save, f, indent=2, default=str)
    log.info(f"  Phase 1: {p1_path}")

    # Phase 2 results (JSON)
    p2_path = OUTPUT_DIR / "phase2_midtrade_results.json"
    with open(p2_path, "w") as f:
        json.dump(phase2_results, f, indent=2, default=str)
    log.info(f"  Phase 2: {p2_path}")

    # Phase 3 results (JSON)
    p3_path = OUTPUT_DIR / "phase3_regime_results.json"
    # Reduce size: drop per_day_metrics from rejected
    p3_save = {
        "accepted": phase3_results.get("accepted", []),
        "rejected": [{k: v for k, v in r.items() if k != "per_day_metrics"}
                     for r in phase3_results.get("rejected", [])],
        "regime_distribution": phase3_results.get("regime_distribution", {}),
        "n_accepted": phase3_results.get("n_accepted", 0),
        "n_rejected": phase3_results.get("n_rejected", 0),
    }
    with open(p3_path, "w") as f:
        json.dump(p3_save, f, indent=2, default=str)
    log.info(f"  Phase 3: {p3_path}")

    # Save predictions as .npz
    pred_path = OUTPUT_DIR / "wf_predictions.npz"
    np.savez_compressed(str(pred_path), **{k: v for k, v in predictions.items()})
    log.info(f"  Predictions: {pred_path}")

    # Save per-day trade detail for best accepted config as parquet
    if phase3_results.get("accepted"):
        best = phase3_results["accepted"][0]
        if best.get("per_day_metrics"):
            df = pd.DataFrame(best["per_day_metrics"])
            df_path = OUTPUT_DIR / "best_config_per_day.parquet"
            df.to_parquet(str(df_path))
            log.info(f"  Best per-day: {df_path}")

    # Save comprehensive summary
    summary_path = OUTPUT_DIR / "experiment_summary.json"
    summary = {
        "experiment": "dynamic_execution_v8",
        "timestamp": _ts,
        "walk_forward": {"train_days": TRAIN_DAYS, "slide_days": SLIDE_DAYS},
        "costs": {
            "entry_cost_ticks": ENTRY_COST_TICKS,
            "exit_cost_ticks": EXIT_COST_TICKS,
            "total_rt_cost_ticks": TOTAL_RT_COST_TICKS,
        },
        "n_oot_days": len(oot_dates),
        "n_features": NUM_FEATURES,
        "phase1_n_configs": len(phase1_results.get("all_results", [])),
        "phase3_n_accepted": phase3_results.get("n_accepted", 0),
        "phase3_n_rejected": phase3_results.get("n_rejected", 0),
    }

    if phase3_results.get("accepted"):
        best = phase3_results["accepted"][0]
        summary["best_config"] = best["config_name"]
        summary["best_metrics"] = best["overall_metrics"]
        summary["best_regime"] = best["regime_analysis"]
    elif phase1_results.get("all_results"):
        best = phase1_results["all_results"][0]
        summary["best_config"] = best["config"]["name"]
        summary["best_metrics"] = best["metrics"]
        summary["note"] = "No config passed regime gate — best Phase 1 result shown"

    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"  Summary: {summary_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Dynamic Execution v8")
    parser.add_argument("--quick", action="store_true",
                       help="Quick mode: fewer configs for faster testing")
    parser.add_argument("--max-days", type=int, default=0,
                       help="Limit number of days loaded (0=all)")
    parser.add_argument("--skip-phase2", action="store_true",
                       help="Skip mid-trade management phase")
    parser.add_argument("--top-n", type=int, default=10,
                       help="Number of top Phase 1 configs to validate in Phase 3")
    args = parser.parse_args()

    t0 = time.time()

    log.info("=" * 70)
    log.info("Dynamic Execution v8 — Starting")
    log.info(f"  Output: {OUTPUT_DIR}")
    log.info(f"  Data: {MINUTE_BAR_DIR}")
    log.info(f"  Quick mode: {args.quick}")
    log.info(f"  Cost model: entry={ENTRY_COST_TICKS:.3f} + exit={EXIT_COST_TICKS:.3f} "
             f"= {TOTAL_RT_COST_TICKS:.3f} ticks RT")
    log.info("=" * 70)

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    log.info("\nLoading minute bar data...")
    days = load_all_days()
    if args.max_days > 0:
        days = days[:args.max_days]

    if len(days) < TRAIN_DAYS + 10:
        log.error(f"Insufficient data: {len(days)} days < {TRAIN_DAYS + 10} minimum")
        sys.exit(1)

    # Classify regimes
    regime_map = classify_regimes(days)
    regime_counts = {"green": 0, "red": 0, "flat": 0}
    for d in days:
        r = regime_map.get(d.date_str, "flat")
        regime_counts[r] += 1
    log.info(f"Regime distribution across all days: {regime_counts}")

    # -----------------------------------------------------------------------
    # Walk-forward entry model
    # -----------------------------------------------------------------------
    log.info("\nRunning walk-forward entry model training...")
    t_wf = time.time()
    predictions, oot_dates = run_walkforward(days, train_window=TRAIN_DAYS)
    log.info(f"Walk-forward completed in {(time.time() - t_wf) / 60:.1f} minutes")

    if not predictions:
        log.error("No predictions generated!")
        sys.exit(1)

    # Quick IC summary across all OOT days
    all_preds, all_labels = [], []
    days_by_date = {d.date_str: d for d in days}
    for date_str in oot_dates:
        if date_str in predictions and date_str in days_by_date:
            dd = days_by_date[date_str]
            p = predictions[date_str]
            valid = np.isfinite(dd.labels) & np.isfinite(p)
            if valid.sum() > 10:
                all_preds.extend(p[valid])
                all_labels.extend(dd.labels[valid])

    if all_preds:
        concat_ic, _ = spearmanr(all_preds, all_labels)
        log.info(f"Concat IC across {len(oot_dates)} OOT days: {concat_ic:.4f}")
    else:
        concat_ic = 0.0
        log.warning("Could not compute concat IC")

    # -----------------------------------------------------------------------
    # Phase 1: Dynamic SL/TP
    # -----------------------------------------------------------------------
    t_p1 = time.time()
    phase1_results = phase1_dynamic_sltp(days, predictions, oot_dates, quick=args.quick)
    log.info(f"Phase 1 completed in {(time.time() - t_p1) / 60:.1f} minutes")

    # -----------------------------------------------------------------------
    # Phase 2: Mid-Trade Management
    # -----------------------------------------------------------------------
    phase2_results = {}
    if not args.skip_phase2 and phase1_results.get("all_results"):
        t_p2 = time.time()
        best_p1_config = phase1_results["all_results"][0]
        phase2_results = phase2_midtrade_management(
            days, predictions, oot_dates, best_p1_config, quick=args.quick)
        log.info(f"Phase 2 completed in {(time.time() - t_p2) / 60:.1f} minutes")
    else:
        phase2_results = {"skipped": True, "reason": "phase2_disabled_or_no_p1_results"}

    # -----------------------------------------------------------------------
    # Phase 3: Regime Validation
    # -----------------------------------------------------------------------
    t_p3 = time.time()
    # Take top N configs from Phase 1 for regime validation
    top_n = min(args.top_n, len(phase1_results.get("all_results", [])))
    top_configs = phase1_results.get("all_results", [])[:top_n]
    phase3_results = phase3_regime_validation(
        days, predictions, oot_dates, top_configs, regime_map)
    log.info(f"Phase 3 completed in {(time.time() - t_p3) / 60:.1f} minutes")

    # -----------------------------------------------------------------------
    # Phase 4: MLflow Logging
    # -----------------------------------------------------------------------
    total_time = time.time() - t0
    phase4_mlflow_logging(phase1_results, phase2_results, phase3_results, total_time)

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    save_results(phase1_results, phase2_results, phase3_results, predictions, oot_dates)

    # -----------------------------------------------------------------------
    # Final Summary
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("EXPERIMENT COMPLETE — FINAL SUMMARY")
    log.info("=" * 70)
    log.info(f"Total runtime: {total_time / 60:.1f} minutes")
    log.info(f"OOT days: {len(oot_dates)}")
    log.info(f"Concat IC: {concat_ic:.4f}")
    log.info(f"Phase 1: {len(phase1_results.get('all_results', []))} configs tested")
    log.info(f"Phase 2: {'completed' if not phase2_results.get('skipped') else 'skipped'}")
    log.info(f"Phase 3: {phase3_results.get('n_accepted', 0)} accepted, "
             f"{phase3_results.get('n_rejected', 0)} rejected")

    if phase3_results.get("accepted"):
        best = phase3_results["accepted"][0]
        log.info(f"\nBEST ACCEPTED CONFIG: {best['config_name']}")
        m = best["overall_metrics"]
        log.info(f"  Sharpe: {m['sharpe']:.2f}")
        log.info(f"  Sortino: {m['sortino']:.2f}")
        log.info(f"  Profit Factor: {m['profit_factor']:.2f}")
        log.info(f"  Win Rate: {m['win_rate']:.1%}")
        log.info(f"  Total P&L: ${m.get('total_pnl_dollars', 0):.0f} "
                 f"({m['total_pnl_net']:.1f} ticks)")
        log.info(f"  N Trades: {m['n_trades']}")
        log.info(f"  Regime Divergence: {best['regime_analysis']['regime_divergence']:.3f}")
        log.info(f"  Day Concentration: {best['day_concentration']:.3f}")
    else:
        log.info("\nNO CONFIG PASSED REGIME GATE — check Phase 1 results for best unconstrained config")
        if phase1_results.get("all_results"):
            best = phase1_results["all_results"][0]
            log.info(f"  Best unconstrained: {best['config']['name']} "
                     f"Sharpe={best['metrics']['sharpe']:.2f}")

    log.info(f"\nResults saved to: {OUTPUT_DIR}")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
