#!/usr/bin/env python3
"""Champion config extended validation across ALL 197 days of minute bar data.

Validates TP=25, SL_long=4, SL_short=3, max_hold=60min, daily_bias_mult=1.5
using walk-forward LightGBM (60d train, 1d slide) on 30-min bars built from
minute bars. Tests multiple config variants + regime analysis (HC #428).

Run on Neptune: /home/nick/miniconda3/envs/py311-train/bin/python champion_extended_validation.py
"""

import logging
import json
import csv
import time
import warnings
from pathlib import Path
from datetime import datetime, timedelta
from collections import defaultdict

import numpy as np
import pandas as pd
import lightgbm as lgb

warnings.filterwarnings("ignore", category=UserWarning)
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
LVL3_ROOT = Path("/home/nick/Lvl3Quant")
MINUTE_BAR_DIR = LVL3_ROOT / "data" / "processed" / "mbo_minute_bars_v1"
ENHANCED_DAILY = LVL3_ROOT / "output" / "long_horizon_flow_v2" / "enhanced_daily_features.parquet"
OUT_DIR = LVL3_ROOT / "output" / "champion_extended_validation"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ES_TICK_SIZE = 0.25      # 1 tick = 0.25 points
ES_TICK_VALUE = 12.50    # $12.50 per tick
COST_ENTRY_TICKS = 0.376   # passive entry (commission only)
COST_EXIT_TICKS = 1.376    # market exit (commission + spread)
COST_RT_TICKS = COST_ENTRY_TICKS + COST_EXIT_TICKS  # 1.752 ticks round-trip

TRAIN_DAYS = 60
SLIDE_DAYS = 1
BAR_MINUTES = 30

MLFLOW_URI = "http://jupiter:5000"
EXPERIMENT_NAME = "champion_extended_validation"

# LightGBM params (canonical)
LGB_PARAMS = dict(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.5,
    min_child_samples=50,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_jobs=-1,
    verbose=-1,
    random_state=42,
)

# Champion config variants
CONFIGS = {
    "champion": dict(
        tp_ticks=25, sl_long_ticks=4, sl_short_ticks=3,
        max_hold_minutes=60, threshold=0.05, daily_bias=True, bias_mult=1.5,
        adaptive_sl=False,
    ),
    "wider_sl": dict(
        tp_ticks=25, sl_long_ticks=6, sl_short_ticks=5,
        max_hold_minutes=60, threshold=0.05, daily_bias=True, bias_mult=1.5,
        adaptive_sl=False,
    ),
    "adaptive_sl": dict(
        tp_ticks=25, sl_long_ticks=4, sl_short_ticks=3,
        max_hold_minutes=60, threshold=0.05, daily_bias=True, bias_mult=1.5,
        adaptive_sl=True,
    ),
    "thresh_010": dict(
        tp_ticks=25, sl_long_ticks=4, sl_short_ticks=3,
        max_hold_minutes=60, threshold=0.10, daily_bias=True, bias_mult=1.5,
        adaptive_sl=False,
    ),
    "thresh_015": dict(
        tp_ticks=25, sl_long_ticks=4, sl_short_ticks=3,
        max_hold_minutes=60, threshold=0.15, daily_bias=True, bias_mult=1.5,
        adaptive_sl=False,
    ),
    "thresh_020": dict(
        tp_ticks=25, sl_long_ticks=4, sl_short_ticks=3,
        max_hold_minutes=60, threshold=0.20, daily_bias=True, bias_mult=1.5,
        adaptive_sl=False,
    ),
    "no_daily_bias": dict(
        tp_ticks=25, sl_long_ticks=4, sl_short_ticks=3,
        max_hold_minutes=60, threshold=0.05, daily_bias=False, bias_mult=1.0,
        adaptive_sl=False,
    ),
}

# Feature column names
FEATURE_COLS = [
    "ofi_sum", "volume", "vwap_bar", "signed_volume_sum",
    "mom_3bar", "mom_5bar", "vol_3bar", "vol_5bar",
    "ofi_trend_3bar", "ofi_trend_5bar",
    "spread_mean_bar", "vol_regime_mode",
    "bar_range", "bar_body", "bar_upper_wick", "bar_lower_wick",
    "volume_ratio_3bar", "volume_ratio_5bar",
    "close_vs_vwap", "trade_intensity",
]


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_all_minute_bars() -> pd.DataFrame:
    """Load all minute bar parquets into a single DataFrame."""
    files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {MINUTE_BAR_DIR}")
    log.info(f"Loading {len(files)} minute bar files...")
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            # Ensure date column exists
            if "date" not in df.columns:
                # Try to extract from filename or ts_minute
                fname = f.stem
                date_str = "".join(c for c in fname if c.isdigit())[:8]
                if len(date_str) == 8:
                    df["date"] = date_str
                elif "ts_minute" in df.columns:
                    df["date"] = pd.to_datetime(df["ts_minute"]).dt.strftime("%Y%m%d")
            dfs.append(df)
        except Exception as e:
            log.warning(f"Skip {f.name}: {e}")
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.sort_values(["date", "ts_minute"], inplace=True)
    df_all.reset_index(drop=True, inplace=True)
    log.info(f"Loaded {len(df_all):,} minute bars across {df_all['date'].nunique()} days "
             f"({df_all['date'].min()} to {df_all['date'].max()})")
    return df_all


def aggregate_to_30min(df_min: pd.DataFrame) -> pd.DataFrame:
    """Aggregate minute bars to 30-min bars per day."""
    df = df_min.copy()
    df["ts_minute"] = pd.to_datetime(df["ts_minute"])
    df["bar_30"] = df["ts_minute"].dt.floor("30min")

    agg_funcs = {
        "open": ("open", "first"),
        "high": ("high", "max"),
        "low": ("low", "min"),
        "close": ("close", "last"),
        "volume": ("volume", "sum"),
        "trade_count": ("trade_count", "sum"),
    }

    # Standard OHLCV
    grouped = df.groupby(["date", "bar_30"])
    bars = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        trade_count=("trade_count", "sum"),
    )

    # VWAP: volume-weighted average price
    df["dollar_volume"] = df["close"] * df["volume"]
    bars["vwap_bar"] = grouped["dollar_volume"].sum() / (grouped["volume"].sum() + 1e-8)

    # Sum-based columns
    if "signed_volume" in df.columns:
        bars["signed_volume_sum"] = grouped["signed_volume"].sum()
    else:
        bars["signed_volume_sum"] = 0.0

    if "ofi_1min" in df.columns:
        bars["ofi_sum"] = grouped["ofi_1min"].sum()
    else:
        bars["ofi_sum"] = 0.0

    if "spread_mean" in df.columns:
        bars["spread_mean_bar"] = grouped["spread_mean"].mean()
    else:
        bars["spread_mean_bar"] = 0.0

    if "vol_regime" in df.columns:
        # Map string vol_regime to numeric: low=0, medium=1, high=2
        vol_map = {'low': 0, 'medium': 1, 'high': 2, 0: 0, 1: 1, 2: 2}
        df["vol_regime_num"] = df["vol_regime"].map(vol_map).fillna(1).astype(int)
        bars["vol_regime_mode"] = grouped["vol_regime_num"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 1
        )
    else:
        bars["vol_regime_mode"] = 1

    bars.reset_index(inplace=True)
    bars.sort_values(["date", "bar_30"], inplace=True)
    bars.reset_index(drop=True, inplace=True)
    log.info(f"Aggregated to {len(bars):,} 30-min bars")
    return bars


# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
def engineer_features(bars: pd.DataFrame) -> pd.DataFrame:
    """Add rolling features to 30-min bars."""
    df = bars.copy()

    # Bar anatomy
    df["bar_range"] = (df["high"] - df["low"]) / ES_TICK_SIZE
    df["bar_body"] = abs(df["close"] - df["open"]) / ES_TICK_SIZE
    df["bar_upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / ES_TICK_SIZE
    df["bar_lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / ES_TICK_SIZE

    # Close vs VWAP
    df["close_vs_vwap"] = (df["close"] - df["vwap_bar"]) / ES_TICK_SIZE

    # Trade intensity
    df["trade_intensity"] = df["trade_count"] / (df["volume"] + 1e-8)

    # Rolling features (within each day group to avoid cross-day leakage)
    for window, suffix in [(3, "3bar"), (5, "5bar")]:
        # Momentum: close change over window
        df[f"mom_{suffix}"] = df.groupby("date")["close"].transform(
            lambda s: s.diff(window) / ES_TICK_SIZE
        )
        # Volatility: rolling std of close changes
        df[f"vol_{suffix}"] = df.groupby("date")["close"].transform(
            lambda s: s.diff().rolling(window, min_periods=1).std() / ES_TICK_SIZE
        )
        # OFI trend
        df[f"ofi_trend_{suffix}"] = df.groupby("date")["ofi_sum"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        # Volume ratio vs rolling mean
        vol_rm = df.groupby("date")["volume"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"volume_ratio_{suffix}"] = df["volume"] / (vol_rm + 1e-8)

    # Fill NaN from rolling with 0
    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)

    return df


def compute_label(bars: pd.DataFrame) -> pd.DataFrame:
    """Label: mid-price change 30 minutes ahead (1 bar forward), in ticks."""
    df = bars.copy()
    # Forward return in ticks (next bar's close - this bar's close)
    df["label"] = df.groupby("date")["close"].transform(lambda s: s.shift(-1)) - df["close"]
    df["label"] = df["label"] / ES_TICK_SIZE  # convert to ticks
    return df


# ---------------------------------------------------------------------------
# Daily classification (regime)
# ---------------------------------------------------------------------------
def classify_days(bars: pd.DataFrame) -> dict:
    """Classify each day as green/red/flat. Returns {date: regime}."""
    # Try enhanced daily features first
    daily_close = {}
    if ENHANCED_DAILY.exists():
        try:
            edf = pd.read_parquet(ENHANCED_DAILY)
            if "ES_close" in edf.columns and "date" in edf.columns:
                for _, row in edf.iterrows():
                    d = str(row["date"])[:8] if not isinstance(row["date"], str) else row["date"]
                    daily_close[d] = float(row["ES_close"])
                log.info(f"Using enhanced_daily_features for regime classification ({len(daily_close)} days)")
        except Exception as e:
            log.warning(f"Could not load enhanced daily features: {e}")

    # Fall back to bar data
    if not daily_close:
        for date, grp in bars.groupby("date"):
            if len(grp) > 0:
                daily_close[date] = float(grp.iloc[-1]["close"])

    dates_sorted = sorted(daily_close.keys())
    regimes = {}
    for i, d in enumerate(dates_sorted):
        if i == 0:
            regimes[d] = "flat"
            continue
        prev_d = dates_sorted[i - 1]
        chg = daily_close[d] - daily_close[prev_d]
        if chg > 5.0:
            regimes[d] = "green"
        elif chg < -5.0:
            regimes[d] = "red"
        else:
            regimes[d] = "flat"

    counts = defaultdict(int)
    for r in regimes.values():
        counts[r] += 1
    log.info(f"Day regimes: green={counts['green']}, red={counts['red']}, flat={counts['flat']}")
    return regimes


def get_prior_day_bias(bars: pd.DataFrame) -> dict:
    """For each day, determine if prior day was up or down (for daily bias multiplier)."""
    daily = {}
    for date, grp in bars.groupby("date"):
        if len(grp) >= 2:
            daily[date] = {"open": float(grp.iloc[0]["open"]),
                           "close": float(grp.iloc[-1]["close"])}
        elif len(grp) == 1:
            daily[date] = {"open": float(grp.iloc[0]["open"]),
                           "close": float(grp.iloc[0]["close"])}

    dates_sorted = sorted(daily.keys())
    # prior_day_up: True if prior day close > open
    prior_day_up = {}
    for i, d in enumerate(dates_sorted):
        if i == 0:
            prior_day_up[d] = None  # no prior day
        else:
            prev = daily[dates_sorted[i - 1]]
            prior_day_up[d] = prev["close"] > prev["open"]
    return prior_day_up


# ---------------------------------------------------------------------------
# Walk-forward engine
# ---------------------------------------------------------------------------
def walk_forward(bars: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward LightGBM: 60d train, 1d slide. Returns predictions DataFrame."""
    dates = sorted(bars["date"].unique())
    log.info(f"Walk-forward over {len(dates)} unique days, {TRAIN_DAYS}d train, {SLIDE_DAYS}d slide")

    all_preds = []
    n_folds = 0

    for i in range(TRAIN_DAYS, len(dates)):
        oot_date = dates[i]
        train_dates = dates[max(0, i - TRAIN_DAYS):i]

        # Train data
        train_mask = bars["date"].isin(train_dates)
        df_train = bars[train_mask].dropna(subset=["label"])
        if len(df_train) < 100:
            continue

        # OOT data
        oot_mask = bars["date"] == oot_date
        df_oot = bars[oot_mask].copy()
        if len(df_oot) == 0:
            continue

        X_train = df_train[FEATURE_COLS].values.astype(np.float32)
        y_train = df_train["label"].values.astype(np.float32)
        X_oot = df_oot[FEATURE_COLS].values.astype(np.float32)

        # Train
        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train)

        # Predict
        preds = model.predict(X_oot)
        df_oot = df_oot.copy()
        df_oot["prediction"] = preds
        all_preds.append(df_oot)

        n_folds += 1
        if n_folds % 20 == 0:
            log.info(f"  Fold {n_folds}: OOT date {oot_date}, "
                     f"train {len(df_train)} bars, OOT {len(df_oot)} bars")

    if not all_preds:
        raise RuntimeError("No OOT predictions generated!")

    result = pd.concat(all_preds, ignore_index=True)
    log.info(f"Walk-forward complete: {n_folds} folds, {len(result):,} OOT predictions")
    return result


# ---------------------------------------------------------------------------
# Trade simulation
# ---------------------------------------------------------------------------
def simulate_trades(
    predictions: pd.DataFrame,
    minute_bars: pd.DataFrame,
    config: dict,
    prior_day_up: dict,
    daily_atr: dict | None = None,
) -> list[dict]:
    """Simulate trades using minute-bar replay.

    For each 30-min bar prediction exceeding threshold:
      - Enter at bar close price
      - Walk forward through minute bars checking SL/TP/timeout
      - SL checked before TP in same bar (conservative for tight SL)
    """
    tp_ticks = config["tp_ticks"]
    sl_long_ticks = config["sl_long_ticks"]
    sl_short_ticks = config["sl_short_ticks"]
    max_hold_min = config["max_hold_minutes"]
    threshold = config["threshold"]
    use_daily_bias = config["daily_bias"]
    bias_mult = config["bias_mult"]
    adaptive_sl = config["adaptive_sl"]

    # Index minute bars by (date, ts_minute) for fast lookup
    minute_bars = minute_bars.copy()
    minute_bars["ts_minute"] = pd.to_datetime(minute_bars["ts_minute"])
    minute_bars_by_date = {
        date: grp.sort_values("ts_minute").reset_index(drop=True)
        for date, grp in minute_bars.groupby("date")
    }

    trades = []
    for _, row in predictions.iterrows():
        pred = row["prediction"]
        date = row["date"]
        bar_time = pd.to_datetime(row["bar_30"])
        entry_price = row["close"]

        # Apply daily bias multiplier
        adjusted_pred = pred
        if use_daily_bias and date in prior_day_up and prior_day_up[date] is not None:
            if prior_day_up[date]:
                # Prior day was up -> boost SHORT signals
                if pred < 0:
                    adjusted_pred = pred * bias_mult
            else:
                # Prior day was down -> boost LONG signals
                if pred > 0:
                    adjusted_pred = pred * bias_mult

        # Check threshold
        if abs(adjusted_pred) < threshold:
            continue

        direction = 1 if adjusted_pred > 0 else -1  # 1=LONG, -1=SHORT

        # Determine SL for this trade
        if direction == 1:
            sl_ticks = sl_long_ticks
        else:
            sl_ticks = sl_short_ticks

        # Adaptive SL: max(champion_SL, 0.5 * daily_ATR_in_ticks)
        if adaptive_sl and daily_atr and date in daily_atr:
            sl_ticks = max(sl_ticks, 0.5 * daily_atr[date])

        sl_pts = sl_ticks * ES_TICK_SIZE
        tp_pts = tp_ticks * ES_TICK_SIZE

        # Get minute bars for this day
        if date not in minute_bars_by_date:
            continue
        day_mins = minute_bars_by_date[date]

        # Find the first minute bar AFTER the 30-min bar close
        entry_time = bar_time + pd.Timedelta(minutes=30)  # bar close = end of 30-min bar
        mask_after = day_mins["ts_minute"] > bar_time
        future_mins = day_mins[mask_after]

        if len(future_mins) == 0:
            continue

        # Simulate through minute bars
        exit_price = None
        exit_reason = None
        exit_time = None
        bars_held = 0

        for _, mbar in future_mins.iterrows():
            bars_held += 1

            if direction == 1:  # LONG
                # SL check first (conservative for tight SL)
                if mbar["low"] <= entry_price - sl_pts:
                    exit_price = entry_price - sl_pts
                    exit_reason = "SL"
                    exit_time = mbar["ts_minute"]
                    break
                # TP check
                if mbar["high"] >= entry_price + tp_pts:
                    exit_price = entry_price + tp_pts
                    exit_reason = "TP"
                    exit_time = mbar["ts_minute"]
                    break
            else:  # SHORT
                # SL check first
                if mbar["high"] >= entry_price + sl_pts:
                    exit_price = entry_price + sl_pts
                    exit_reason = "SL"
                    exit_time = mbar["ts_minute"]
                    break
                # TP check
                if mbar["low"] <= entry_price - tp_pts:
                    exit_price = entry_price - tp_pts
                    exit_reason = "TP"
                    exit_time = mbar["ts_minute"]
                    break

            # Timeout
            if bars_held >= max_hold_min:
                exit_price = mbar["close"]
                exit_reason = "TIMEOUT"
                exit_time = mbar["ts_minute"]
                break

        if exit_price is None:
            # End of day exit
            exit_price = future_mins.iloc[-1]["close"]
            exit_reason = "EOD"
            exit_time = future_mins.iloc[-1]["ts_minute"]
            bars_held = len(future_mins)

        # P&L in ticks
        raw_pnl_pts = (exit_price - entry_price) * direction
        raw_pnl_ticks = raw_pnl_pts / ES_TICK_SIZE
        net_pnl_ticks = raw_pnl_ticks - COST_RT_TICKS
        net_pnl_dollars = net_pnl_ticks * ES_TICK_VALUE

        trades.append({
            "date": date,
            "entry_time": str(bar_time),
            "exit_time": str(exit_time),
            "direction": "LONG" if direction == 1 else "SHORT",
            "entry_price": float(entry_price),
            "exit_price": float(exit_price),
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            "prediction": float(pred),
            "adjusted_pred": float(adjusted_pred),
            "raw_pnl_ticks": round(float(raw_pnl_ticks), 4),
            "cost_ticks": COST_RT_TICKS,
            "net_pnl_ticks": round(float(net_pnl_ticks), 4),
            "net_pnl_dollars": round(float(net_pnl_dollars), 2),
        })

    return trades


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def compute_metrics(trades: list[dict], label: str = "") -> dict:
    """Compute risk-adjusted performance metrics."""
    if not trades:
        return {"label": label, "n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0}

    pnls = np.array([t["net_pnl_ticks"] for t in trades])
    n = len(pnls)
    gross_wins = pnls[pnls > 0].sum()
    gross_losses = abs(pnls[pnls < 0].sum())
    win_rate = (pnls > 0).mean()
    avg_win = pnls[pnls > 0].mean() if (pnls > 0).any() else 0
    avg_loss = abs(pnls[pnls < 0].mean()) if (pnls < 0).any() else 0

    # Sharpe (daily)
    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t["date"]] += t["net_pnl_ticks"]
    daily_arr = np.array(list(daily_pnl.values()))
    sharpe = float(daily_arr.mean() / (daily_arr.std() + 1e-8) * np.sqrt(252)) if len(daily_arr) > 1 else 0

    # Sortino (daily)
    downside = daily_arr[daily_arr < 0]
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-8
    sortino = float(daily_arr.mean() / (downside_std + 1e-8) * np.sqrt(252)) if len(daily_arr) > 1 else 0

    # Profit factor
    pf = float(gross_wins / (gross_losses + 1e-8))

    # Max drawdown (cumulative ticks)
    cum_pnl = np.cumsum(pnls)
    peak = np.maximum.accumulate(cum_pnl)
    dd = peak - cum_pnl
    max_dd = float(dd.max()) if len(dd) > 0 else 0

    # Exit reason breakdown
    exit_counts = defaultdict(int)
    for t in trades:
        exit_counts[t["exit_reason"]] += 1

    # Direction breakdown
    long_trades = [t for t in trades if t["direction"] == "LONG"]
    short_trades = [t for t in trades if t["direction"] == "SHORT"]
    long_pnl = sum(t["net_pnl_ticks"] for t in long_trades) if long_trades else 0
    short_pnl = sum(t["net_pnl_ticks"] for t in short_trades) if short_trades else 0

    metrics = {
        "label": label,
        "n_trades": n,
        "n_days": len(daily_pnl),
        "trades_per_day": round(n / max(len(daily_pnl), 1), 2),
        "total_pnl_ticks": round(float(pnls.sum()), 2),
        "total_pnl_dollars": round(float(pnls.sum() * ES_TICK_VALUE), 2),
        "avg_pnl_ticks": round(float(pnls.mean()), 4),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "profit_factor": round(pf, 3),
        "win_rate": round(float(win_rate), 4),
        "avg_win_ticks": round(float(avg_win), 3),
        "avg_loss_ticks": round(float(avg_loss), 3),
        "payoff_ratio": round(float(avg_win / (avg_loss + 1e-8)), 3),
        "max_drawdown_ticks": round(max_dd, 2),
        "n_long": len(long_trades),
        "n_short": len(short_trades),
        "long_pnl_ticks": round(long_pnl, 2),
        "short_pnl_ticks": round(short_pnl, 2),
        "exit_SL": exit_counts.get("SL", 0),
        "exit_TP": exit_counts.get("TP", 0),
        "exit_TIMEOUT": exit_counts.get("TIMEOUT", 0),
        "exit_EOD": exit_counts.get("EOD", 0),
    }
    return metrics


def regime_analysis(trades: list[dict], regimes: dict, label: str = "") -> dict:
    """Per-regime Sharpe, PF, WR + HC #428 gate check."""
    regime_trades = defaultdict(list)
    for t in trades:
        r = regimes.get(t["date"], "unknown")
        regime_trades[r].append(t)

    regime_metrics = {}
    for regime in ["green", "red", "flat"]:
        rt = regime_trades.get(regime, [])
        if rt:
            m = compute_metrics(rt, label=f"{label}_{regime}")
            regime_metrics[regime] = m
        else:
            regime_metrics[regime] = {"sharpe": 0, "n_trades": 0}

    # HC #428 regime gate: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|) <= 0.50
    s_green = regime_metrics.get("green", {}).get("sharpe", 0)
    s_red = regime_metrics.get("red", {}).get("sharpe", 0)
    denom = max(abs(s_green), abs(s_red), 1e-8)
    regime_skew = abs(s_green - s_red) / denom
    gate_pass = regime_skew <= 0.50

    return {
        "regime_metrics": regime_metrics,
        "sharpe_green": s_green,
        "sharpe_red": s_red,
        "regime_skew": round(regime_skew, 4),
        "hc428_gate_pass": gate_pass,
    }


def compute_per_day_stats(trades: list[dict]) -> list[dict]:
    """Per-day breakdown of P&L."""
    daily = defaultdict(list)
    for t in trades:
        daily[t["date"]].append(t)

    rows = []
    for date in sorted(daily.keys()):
        dt = daily[date]
        pnls = [t["net_pnl_ticks"] for t in dt]
        rows.append({
            "date": date,
            "n_trades": len(dt),
            "pnl_ticks": round(sum(pnls), 4),
            "pnl_dollars": round(sum(pnls) * ES_TICK_VALUE, 2),
            "wr": round(sum(1 for p in pnls if p > 0) / max(len(pnls), 1), 4),
            "n_long": sum(1 for t in dt if t["direction"] == "LONG"),
            "n_short": sum(1 for t in dt if t["direction"] == "SHORT"),
            "n_sl": sum(1 for t in dt if t["exit_reason"] == "SL"),
            "n_tp": sum(1 for t in dt if t["exit_reason"] == "TP"),
        })
    return rows


def compute_daily_atr(bars: pd.DataFrame) -> dict:
    """Compute daily ATR in ticks for adaptive SL."""
    daily_atr = {}
    for date, grp in bars.groupby("date"):
        ranges = (grp["high"] - grp["low"]).values / ES_TICK_SIZE
        daily_atr[date] = float(np.mean(ranges)) if len(ranges) > 0 else 4.0
    return daily_atr


# ---------------------------------------------------------------------------
# MLflow logging
# ---------------------------------------------------------------------------
def log_to_mlflow(all_results: dict):
    """Log results to MLflow."""
    try:
        import mlflow
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)

        with mlflow.start_run(run_name=f"champion_extended_{datetime.now():%Y%m%d_%H%M%S}"):
            # Log overall params
            mlflow.log_param("n_configs", len(all_results["configs"]))
            mlflow.log_param("train_days", TRAIN_DAYS)
            mlflow.log_param("bar_minutes", BAR_MINUTES)
            mlflow.log_param("total_dates", all_results.get("n_total_dates", 0))
            mlflow.log_param("oot_dates", all_results.get("n_oot_dates", 0))
            mlflow.log_param("cost_rt_ticks", COST_RT_TICKS)

            # Log champion metrics
            champ = all_results["configs"].get("champion", {})
            if "metrics" in champ:
                m = champ["metrics"]
                mlflow.log_metric("champion_sharpe", m.get("sharpe", 0))
                mlflow.log_metric("champion_sortino", m.get("sortino", 0))
                mlflow.log_metric("champion_pf", m.get("profit_factor", 0))
                mlflow.log_metric("champion_wr", m.get("win_rate", 0))
                mlflow.log_metric("champion_n_trades", m.get("n_trades", 0))
                mlflow.log_metric("champion_total_pnl_ticks", m.get("total_pnl_ticks", 0))

            if "regime" in champ:
                r = champ["regime"]
                mlflow.log_metric("regime_skew", r.get("regime_skew", 0))
                mlflow.log_metric("hc428_gate_pass", int(r.get("hc428_gate_pass", False)))

            # Log all config summaries
            for cname, cdata in all_results["configs"].items():
                if "metrics" in cdata:
                    m = cdata["metrics"]
                    mlflow.log_metric(f"{cname}_sharpe", m.get("sharpe", 0))
                    mlflow.log_metric(f"{cname}_n_trades", m.get("n_trades", 0))

            # Log artifacts
            summary_path = OUT_DIR / "results_summary.json"
            if summary_path.exists():
                mlflow.log_artifact(str(summary_path))

        log.info("MLflow logging complete")
    except Exception as e:
        log.warning(f"MLflow logging failed (non-fatal): {e}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log.info("=" * 80)
    log.info("CHAMPION CONFIG EXTENDED VALIDATION")
    log.info(f"TP=25, SL_L=4, SL_S=3, max_hold=60min, daily_bias=1.5")
    log.info(f"Walk-forward: {TRAIN_DAYS}d train, {SLIDE_DAYS}d slide, 30-min bars")
    log.info(f"Cost: entry {COST_ENTRY_TICKS} ticks, exit {COST_EXIT_TICKS} ticks, RT {COST_RT_TICKS} ticks")
    log.info("=" * 80)

    # 1. Load minute bars
    df_min = load_all_minute_bars()

    # 2. Aggregate to 30-min bars
    bars_30 = aggregate_to_30min(df_min)

    # 3. Feature engineering
    bars_30 = engineer_features(bars_30)

    # 4. Compute labels
    bars_30 = compute_label(bars_30)

    # 5. Day classification
    regimes = classify_days(bars_30)
    prior_day_up = get_prior_day_bias(bars_30)
    daily_atr = compute_daily_atr(bars_30)

    # 6. Walk-forward predictions
    predictions = walk_forward(bars_30)
    n_total_dates = bars_30["date"].nunique()
    n_oot_dates = predictions["date"].nunique()
    log.info(f"OOT predictions cover {n_oot_dates} days (of {n_total_dates} total)")

    # 7. Simulate trades for each config variant
    all_results = {
        "timestamp": datetime.now().isoformat(),
        "n_total_dates": n_total_dates,
        "n_oot_dates": n_oot_dates,
        "n_total_predictions": len(predictions),
        "cost_entry_ticks": COST_ENTRY_TICKS,
        "cost_exit_ticks": COST_EXIT_TICKS,
        "cost_rt_ticks": COST_RT_TICKS,
        "configs": {},
    }

    for config_name, config in CONFIGS.items():
        log.info(f"\n{'='*60}")
        log.info(f"Simulating config: {config_name}")
        log.info(f"  TP={config['tp_ticks']}, SL_L={config['sl_long_ticks']}, "
                 f"SL_S={config['sl_short_ticks']}, hold={config['max_hold_minutes']}min, "
                 f"thresh={config['threshold']}, bias={config['daily_bias']}")

        trades = simulate_trades(
            predictions, df_min, config, prior_day_up,
            daily_atr=daily_atr if config["adaptive_sl"] else None,
        )

        if not trades:
            log.warning(f"  No trades generated for {config_name}!")
            all_results["configs"][config_name] = {"metrics": {"n_trades": 0}, "config": config}
            continue

        metrics = compute_metrics(trades, label=config_name)
        regime = regime_analysis(trades, regimes, label=config_name)
        per_day = compute_per_day_stats(trades)

        log.info(f"  Results: {metrics['n_trades']} trades over {metrics['n_days']} days")
        log.info(f"  Sharpe={metrics['sharpe']:.3f}  Sortino={metrics['sortino']:.3f}  "
                 f"PF={metrics['profit_factor']:.3f}  WR={metrics['win_rate']:.1%}")
        log.info(f"  Total P&L: {metrics['total_pnl_ticks']:.1f} ticks "
                 f"(${metrics['total_pnl_dollars']:,.0f})")
        log.info(f"  Exits: SL={metrics['exit_SL']} TP={metrics['exit_TP']} "
                 f"TIMEOUT={metrics['exit_TIMEOUT']} EOD={metrics['exit_EOD']}")
        log.info(f"  Long={metrics['n_long']} ({metrics['long_pnl_ticks']:.1f}t), "
                 f"Short={metrics['n_short']} ({metrics['short_pnl_ticks']:.1f}t)")
        log.info(f"  Regime gate: skew={regime['regime_skew']:.4f} "
                 f"{'PASS' if regime['hc428_gate_pass'] else 'FAIL'} "
                 f"(green Sharpe={regime['sharpe_green']:.3f}, "
                 f"red Sharpe={regime['sharpe_red']:.3f})")

        # Save per-trade CSV
        trades_csv = OUT_DIR / f"trades_{config_name}.csv"
        if trades:
            with open(trades_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=trades[0].keys())
                writer.writeheader()
                writer.writerows(trades)

        # Save per-day CSV
        daily_csv = OUT_DIR / f"daily_{config_name}.csv"
        if per_day:
            with open(daily_csv, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=per_day[0].keys())
                writer.writeheader()
                writer.writerows(per_day)

        all_results["configs"][config_name] = {
            "config": config,
            "metrics": metrics,
            "regime": regime,
        }

    # 8. Summary comparison
    log.info(f"\n{'='*80}")
    log.info("CROSS-CONFIG COMPARISON")
    log.info(f"{'Config':<16} {'Trades':>7} {'Sharpe':>8} {'Sortino':>8} {'PF':>6} "
             f"{'WR':>6} {'P&L($)':>10} {'Gate':>6}")
    log.info("-" * 80)
    for cname, cdata in all_results["configs"].items():
        m = cdata.get("metrics", {})
        r = cdata.get("regime", {})
        gate = "PASS" if r.get("hc428_gate_pass", False) else "FAIL"
        if m.get("n_trades", 0) == 0:
            gate = "N/A"
        log.info(f"{cname:<16} {m.get('n_trades',0):>7} {m.get('sharpe',0):>8.3f} "
                 f"{m.get('sortino',0):>8.3f} {m.get('profit_factor',0):>6.3f} "
                 f"{m.get('win_rate',0):>6.1%} {m.get('total_pnl_dollars',0):>10,.0f} "
                 f"{gate:>6}")

    # 9. Save JSON summary
    summary_path = OUT_DIR / "results_summary.json"

    # Make regime metrics JSON-serializable
    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(i) for i in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.bool_,)):
            return bool(obj)
        return obj

    with open(summary_path, "w") as f:
        json.dump(make_serializable(all_results), f, indent=2, default=str)
    log.info(f"\nSaved summary to {summary_path}")

    # 10. Log to MLflow
    log_to_mlflow(all_results)

    elapsed = time.time() - t0
    log.info(f"\nTotal runtime: {elapsed/60:.1f} minutes")
    log.info("Done.")


if __name__ == "__main__":
    main()
