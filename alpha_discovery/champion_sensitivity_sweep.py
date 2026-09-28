#!/usr/bin/env python3
"""Champion config parameter sensitivity sweep.

Trains the walk-forward LightGBM ONCE, then sweeps a dense grid of
TP/SL/threshold/hold combinations through the trade simulator.
This answers: how robust is the champion config to parameter perturbations?

Run on Neptune: python champion_sensitivity_sweep.py
Output: output/champion_sensitivity_sweep/

Leakage audit: CLEAN (same WF as champion_extended_validation.py, audited 2026-06-25)
"""

import logging
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict
from itertools import product

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
OUT_DIR = LVL3_ROOT / "output" / "champion_sensitivity_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ES_TICK_SIZE = 0.25
ES_TICK_VALUE = 12.50
COST_ENTRY_TICKS = 0.376
COST_EXIT_TICKS = 1.376
COST_RT_TICKS = COST_ENTRY_TICKS + COST_EXIT_TICKS  # 1.752

TRAIN_DAYS = 60
SLIDE_DAYS = 1
BAR_MINUTES = 30

# LightGBM params (same as champion)
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
# Sweep grid
# ---------------------------------------------------------------------------
TP_GRID = [15, 18, 20, 22, 25, 28, 30, 35, 40]
SL_LONG_GRID = [2, 3, 4, 5, 6, 8, 10]
SL_SHORT_GRID = [2, 3, 4, 5, 6, 8, 10]
THRESHOLD_GRID = [0.03, 0.05, 0.08, 0.10, 0.15, 0.20]
HOLD_GRID = [30, 45, 60, 90, 120]

# To keep runtime manageable, we fix some params and sweep others:
# Phase 1: TP x SL sweep (hold=60, thresh=0.05, bias=True)
# Phase 2: Threshold sweep on champion TP/SL
# Phase 3: Hold time sweep on champion TP/SL/thresh

# ---------------------------------------------------------------------------
# Data loading (same as champion_extended_validation.py)
# ---------------------------------------------------------------------------
def load_all_minute_bars() -> pd.DataFrame:
    files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files in {MINUTE_BAR_DIR}")
    log.info(f"Loading {len(files)} minute bar files...")
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if "date" not in df.columns:
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
    log.info(f"Loaded {len(df_all):,} minute bars across {df_all['date'].nunique()} days")
    return df_all


def aggregate_to_30min(df_min: pd.DataFrame) -> pd.DataFrame:
    df = df_min.copy()
    df["ts_minute"] = pd.to_datetime(df["ts_minute"])
    df["bar_30"] = df["ts_minute"].dt.floor("30min")

    grouped = df.groupby(["date", "bar_30"])
    bars = grouped.agg(
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        trade_count=("trade_count", "sum"),
    )

    df["dollar_volume"] = df["close"] * df["volume"]
    bars["vwap_bar"] = grouped["dollar_volume"].sum() / (grouped["volume"].sum() + 1e-8)

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


def engineer_features(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df["bar_range"] = (df["high"] - df["low"]) / ES_TICK_SIZE
    df["bar_body"] = abs(df["close"] - df["open"]) / ES_TICK_SIZE
    df["bar_upper_wick"] = (df["high"] - df[["open", "close"]].max(axis=1)) / ES_TICK_SIZE
    df["bar_lower_wick"] = (df[["open", "close"]].min(axis=1) - df["low"]) / ES_TICK_SIZE
    df["close_vs_vwap"] = (df["close"] - df["vwap_bar"]) / ES_TICK_SIZE
    df["trade_intensity"] = df["trade_count"] / (df["volume"] + 1e-8)

    for window, suffix in [(3, "3bar"), (5, "5bar")]:
        df[f"mom_{suffix}"] = df.groupby("date")["close"].transform(
            lambda s: s.diff(window) / ES_TICK_SIZE
        )
        df[f"vol_{suffix}"] = df.groupby("date")["close"].transform(
            lambda s: s.diff().rolling(window, min_periods=1).std() / ES_TICK_SIZE
        )
        df[f"ofi_trend_{suffix}"] = df.groupby("date")["ofi_sum"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        vol_rm = df.groupby("date")["volume"].transform(
            lambda s: s.rolling(window, min_periods=1).mean()
        )
        df[f"volume_ratio_{suffix}"] = df["volume"] / (vol_rm + 1e-8)

    df[FEATURE_COLS] = df[FEATURE_COLS].fillna(0.0)
    return df


def compute_label(bars: pd.DataFrame) -> pd.DataFrame:
    df = bars.copy()
    df["label"] = df.groupby("date")["close"].transform(lambda s: s.shift(-1)) - df["close"]
    df["label"] = df["label"] / ES_TICK_SIZE
    return df


def get_prior_day_bias(bars: pd.DataFrame) -> dict:
    daily = {}
    for date, grp in bars.groupby("date"):
        if len(grp) >= 2:
            daily[date] = {"open": float(grp.iloc[0]["open"]),
                           "close": float(grp.iloc[-1]["close"])}
        elif len(grp) == 1:
            daily[date] = {"open": float(grp.iloc[0]["open"]),
                           "close": float(grp.iloc[0]["close"])}
    dates_sorted = sorted(daily.keys())
    prior_day_up = {}
    for i, d in enumerate(dates_sorted):
        if i == 0:
            prior_day_up[d] = None
        else:
            prev = daily[dates_sorted[i - 1]]
            prior_day_up[d] = prev["close"] > prev["open"]
    return prior_day_up


def classify_days(bars: pd.DataFrame) -> dict:
    daily_close = {}
    if ENHANCED_DAILY.exists():
        try:
            edf = pd.read_parquet(ENHANCED_DAILY)
            if "ES_close" in edf.columns and "date" in edf.columns:
                for _, row in edf.iterrows():
                    d = str(row["date"])[:8] if not isinstance(row["date"], str) else row["date"]
                    daily_close[d] = float(row["ES_close"])
        except Exception:
            pass
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
        chg = daily_close[d] - daily_close[dates_sorted[i - 1]]
        if chg > 5.0:
            regimes[d] = "green"
        elif chg < -5.0:
            regimes[d] = "red"
        else:
            regimes[d] = "flat"
    return regimes


# ---------------------------------------------------------------------------
# Walk-forward (train once, save predictions)
# ---------------------------------------------------------------------------
def walk_forward(bars: pd.DataFrame) -> pd.DataFrame:
    """Walk-forward LightGBM: 60d train, 1d slide. Train ONCE, reuse predictions."""
    dates = sorted(bars["date"].unique())
    log.info(f"Walk-forward over {len(dates)} days, {TRAIN_DAYS}d train")

    all_preds = []
    n_folds = 0

    for i in range(TRAIN_DAYS, len(dates)):
        oot_date = dates[i]
        train_dates = dates[max(0, i - TRAIN_DAYS):i]

        train_mask = bars["date"].isin(train_dates)
        df_train = bars[train_mask].dropna(subset=["label"])
        if len(df_train) < 100:
            continue

        oot_mask = bars["date"] == oot_date
        df_oot = bars[oot_mask].copy()
        if len(df_oot) == 0:
            continue

        X_train = df_train[FEATURE_COLS].values.astype(np.float32)
        y_train = df_train["label"].values.astype(np.float32)
        X_oot = df_oot[FEATURE_COLS].values.astype(np.float32)

        model = lgb.LGBMRegressor(**LGB_PARAMS)
        model.fit(X_train, y_train)

        preds = model.predict(X_oot)
        df_oot = df_oot.copy()
        df_oot["prediction"] = preds
        all_preds.append(df_oot)

        n_folds += 1
        if n_folds % 20 == 0:
            log.info(f"  Fold {n_folds}: OOT date {oot_date}")

    if not all_preds:
        raise RuntimeError("No folds produced predictions")

    result = pd.concat(all_preds, ignore_index=True)
    log.info(f"Walk-forward complete: {n_folds} folds, {len(result):,} predictions")
    return result


# ---------------------------------------------------------------------------
# Trade simulation (parameterized)
# ---------------------------------------------------------------------------
def simulate_trades(
    predictions: pd.DataFrame,
    minute_bars: pd.DataFrame,
    tp_ticks: int,
    sl_long_ticks: int,
    sl_short_ticks: int,
    max_hold_minutes: int,
    threshold: float,
    daily_bias: bool,
    bias_mult: float,
    prior_day_up: dict,
) -> list[dict]:
    """Simulate trades with given parameters."""
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

        adjusted_pred = pred
        if daily_bias and date in prior_day_up and prior_day_up[date] is not None:
            if prior_day_up[date]:
                if pred < 0:
                    adjusted_pred = pred * bias_mult
            else:
                if pred > 0:
                    adjusted_pred = pred * bias_mult

        if abs(adjusted_pred) < threshold:
            continue

        direction = 1 if adjusted_pred > 0 else -1
        sl_ticks = sl_long_ticks if direction == 1 else sl_short_ticks
        sl_pts = sl_ticks * ES_TICK_SIZE
        tp_pts = tp_ticks * ES_TICK_SIZE

        if date not in minute_bars_by_date:
            continue
        day_mins = minute_bars_by_date[date]

        mask_after = day_mins["ts_minute"] > bar_time
        future_mins = day_mins[mask_after]
        if len(future_mins) == 0:
            continue

        exit_price = None
        exit_reason = None
        bars_held = 0

        for _, mbar in future_mins.iterrows():
            bars_held += 1

            if direction == 1:
                if mbar["low"] <= entry_price - sl_pts:
                    exit_price = entry_price - sl_pts
                    exit_reason = "SL"
                    break
                if mbar["high"] >= entry_price + tp_pts:
                    exit_price = entry_price + tp_pts
                    exit_reason = "TP"
                    break
            else:
                if mbar["high"] >= entry_price + sl_pts:
                    exit_price = entry_price + sl_pts
                    exit_reason = "SL"
                    break
                if mbar["low"] <= entry_price - tp_pts:
                    exit_price = entry_price - tp_pts
                    exit_reason = "TP"
                    break

            if bars_held >= max_hold_minutes:
                exit_price = mbar["close"]
                exit_reason = "TIMEOUT"
                break

        if exit_price is None:
            exit_price = future_mins.iloc[-1]["close"]
            exit_reason = "EOD"
            bars_held = len(future_mins)

        raw_pnl_pts = (exit_price - entry_price) * direction
        raw_pnl_ticks = raw_pnl_pts / ES_TICK_SIZE
        net_pnl_ticks = raw_pnl_ticks - COST_RT_TICKS

        trades.append({
            "date": date,
            "direction": "LONG" if direction == 1 else "SHORT",
            "exit_reason": exit_reason,
            "bars_held": bars_held,
            "raw_pnl_ticks": round(float(raw_pnl_ticks), 4),
            "net_pnl_ticks": round(float(net_pnl_ticks), 4),
        })

    return trades


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def compute_metrics(trades: list[dict], regimes: dict = None) -> dict:
    if not trades:
        return {"n_trades": 0, "sharpe": 0, "sortino": 0, "pf": 0, "wr": 0,
                "regime_gap": 999, "avg_win": 0, "avg_loss": 0, "trades_per_day": 0}

    pnls = np.array([t["net_pnl_ticks"] for t in trades])
    n = len(pnls)
    gross_wins = pnls[pnls > 0].sum()
    gross_losses = abs(pnls[pnls < 0].sum())
    win_rate = float((pnls > 0).mean())
    avg_win = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0
    avg_loss = float(abs(pnls[pnls < 0].mean())) if (pnls < 0).any() else 0

    # Daily Sharpe
    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t["date"]] += t["net_pnl_ticks"]
    daily_arr = np.array(list(daily_pnl.values()))
    n_days = len(daily_arr)
    sharpe = float(daily_arr.mean() / (daily_arr.std() + 1e-8) * np.sqrt(252)) if n_days > 1 else 0
    trades_per_day = n / max(n_days, 1)

    # Sortino
    downside = daily_arr[daily_arr < 0]
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-8
    sortino = float(daily_arr.mean() / (downside_std + 1e-8) * np.sqrt(252)) if n_days > 1 else 0

    pf = float(gross_wins / (gross_losses + 1e-8))

    # Regime analysis
    regime_gap = 999.0
    sharpe_green = 0
    sharpe_red = 0
    if regimes:
        green_pnl = defaultdict(float)
        red_pnl = defaultdict(float)
        for t in trades:
            r = regimes.get(t["date"], "flat")
            if r == "green":
                green_pnl[t["date"]] += t["net_pnl_ticks"]
            elif r == "red":
                red_pnl[t["date"]] += t["net_pnl_ticks"]

        if len(green_pnl) > 1 and len(red_pnl) > 1:
            g_arr = np.array(list(green_pnl.values()))
            r_arr = np.array(list(red_pnl.values()))
            sharpe_green = float(g_arr.mean() / (g_arr.std() + 1e-8) * np.sqrt(252))
            sharpe_red = float(r_arr.mean() / (r_arr.std() + 1e-8) * np.sqrt(252))
            max_abs = max(abs(sharpe_green), abs(sharpe_red), 1e-8)
            regime_gap = abs(sharpe_green - sharpe_red) / max_abs

    # Max drawdown
    cum = np.cumsum(daily_arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max()) if len(dd) > 0 else 0

    return {
        "n_trades": n,
        "n_days": n_days,
        "trades_per_day": round(trades_per_day, 1),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(win_rate * 100, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "regime_gap": round(regime_gap, 3),
        "sharpe_green": round(sharpe_green, 3),
        "sharpe_red": round(sharpe_red, 3),
        "max_dd_ticks": round(max_dd, 1),
        "total_pnl_ticks": round(float(pnls.sum()), 1),
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------
def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("CHAMPION SENSITIVITY SWEEP — Parameter Robustness Analysis")
    log.info("=" * 70)

    # Load and prepare data
    log.info("Phase 0: Loading data...")
    minute_bars = load_all_minute_bars()
    bars_30 = aggregate_to_30min(minute_bars)
    bars_30 = engineer_features(bars_30)
    bars_30 = compute_label(bars_30)
    prior_day_up = get_prior_day_bias(bars_30)
    regimes = classify_days(bars_30)

    # Walk-forward: train ONCE
    log.info("Phase 1: Walk-forward training (one-time)...")
    predictions = walk_forward(bars_30)

    # Save predictions for reuse
    pred_path = OUT_DIR / "wf_predictions.parquet"
    predictions.to_parquet(pred_path)
    log.info(f"Saved {len(predictions):,} predictions to {pred_path}")

    # -----------------------------------------------------------------------
    # Phase 2: TP x SL sweep (champion threshold=0.05, hold=60, bias=True)
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("Phase 2: TP x SL sensitivity sweep")
    log.info("=" * 70)

    results = []
    total_combos = len(TP_GRID) * len(SL_LONG_GRID) * len(SL_SHORT_GRID)
    combo_i = 0

    for tp in TP_GRID:
        for sl_l in SL_LONG_GRID:
            for sl_s in SL_SHORT_GRID:
                combo_i += 1
                if combo_i % 50 == 0:
                    log.info(f"  TP/SL combo {combo_i}/{total_combos}...")

                trades = simulate_trades(
                    predictions, minute_bars,
                    tp_ticks=tp, sl_long_ticks=sl_l, sl_short_ticks=sl_s,
                    max_hold_minutes=60, threshold=0.05,
                    daily_bias=True, bias_mult=1.5,
                    prior_day_up=prior_day_up,
                )
                m = compute_metrics(trades, regimes)
                m["config"] = f"TP{tp}_SL{sl_l}_{sl_s}"
                m["tp"] = tp
                m["sl_long"] = sl_l
                m["sl_short"] = sl_s
                m["threshold"] = 0.05
                m["hold_min"] = 60
                m["phase"] = "tp_sl_sweep"
                results.append(m)

    log.info(f"TP/SL sweep: {len(results)} configs tested")

    # -----------------------------------------------------------------------
    # Phase 3: Threshold sweep on best TP/SL combos
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("Phase 3: Threshold sensitivity sweep")
    log.info("=" * 70)

    # Find top 5 TP/SL combos by Sharpe (regime-passing only)
    passing = [r for r in results if r["regime_gap"] < 0.50 and r["n_trades"] >= 50]
    passing.sort(key=lambda x: x["sharpe"], reverse=True)
    top_configs = passing[:5] if passing else sorted(results, key=lambda x: x["sharpe"], reverse=True)[:5]

    for cfg in top_configs:
        for thresh in THRESHOLD_GRID:
            trades = simulate_trades(
                predictions, minute_bars,
                tp_ticks=cfg["tp"], sl_long_ticks=cfg["sl_long"],
                sl_short_ticks=cfg["sl_short"],
                max_hold_minutes=60, threshold=thresh,
                daily_bias=True, bias_mult=1.5,
                prior_day_up=prior_day_up,
            )
            m = compute_metrics(trades, regimes)
            m["config"] = f"TP{cfg['tp']}_SL{cfg['sl_long']}_{cfg['sl_short']}_T{thresh}"
            m["tp"] = cfg["tp"]
            m["sl_long"] = cfg["sl_long"]
            m["sl_short"] = cfg["sl_short"]
            m["threshold"] = thresh
            m["hold_min"] = 60
            m["phase"] = "threshold_sweep"
            results.append(m)

    # -----------------------------------------------------------------------
    # Phase 4: Hold time sweep on champion config
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("Phase 4: Hold time sensitivity sweep")
    log.info("=" * 70)

    for hold in HOLD_GRID:
        for thresh in [0.05, 0.10, 0.15]:
            trades = simulate_trades(
                predictions, minute_bars,
                tp_ticks=25, sl_long_ticks=4, sl_short_ticks=3,
                max_hold_minutes=hold, threshold=thresh,
                daily_bias=True, bias_mult=1.5,
                prior_day_up=prior_day_up,
            )
            m = compute_metrics(trades, regimes)
            m["config"] = f"TP25_SL4_3_T{thresh}_H{hold}"
            m["tp"] = 25
            m["sl_long"] = 4
            m["sl_short"] = 3
            m["threshold"] = thresh
            m["hold_min"] = hold
            m["phase"] = "hold_sweep"
            results.append(m)

    # -----------------------------------------------------------------------
    # Phase 5: Bias vs no-bias comparison
    # -----------------------------------------------------------------------
    log.info("\n" + "=" * 70)
    log.info("Phase 5: Daily bias ablation")
    log.info("=" * 70)

    for thresh in [0.05, 0.10, 0.15]:
        trades = simulate_trades(
            predictions, minute_bars,
            tp_ticks=25, sl_long_ticks=4, sl_short_ticks=3,
            max_hold_minutes=60, threshold=thresh,
            daily_bias=False, bias_mult=1.0,
            prior_day_up=prior_day_up,
        )
        m = compute_metrics(trades, regimes)
        m["config"] = f"TP25_SL4_3_T{thresh}_NoBias"
        m["tp"] = 25
        m["sl_long"] = 4
        m["sl_short"] = 3
        m["threshold"] = thresh
        m["hold_min"] = 60
        m["phase"] = "bias_ablation"
        results.append(m)

    # -----------------------------------------------------------------------
    # Save results + summary
    # -----------------------------------------------------------------------
    results_df = pd.DataFrame(results)
    results_path = OUT_DIR / "sweep_results.csv"
    results_df.to_csv(results_path, index=False)
    log.info(f"\nSaved {len(results)} configs to {results_path}")

    # Print summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY — Top 20 configs by Sharpe (regime-passing only)")
    log.info("=" * 70)

    regime_pass = results_df[results_df["regime_gap"] < 0.50].copy()
    regime_pass = regime_pass[regime_pass["n_trades"] >= 30]
    regime_pass = regime_pass.sort_values("sharpe", ascending=False).head(20)

    for _, r in regime_pass.iterrows():
        log.info(
            f"  {r['config']:35s} | Sharpe {r['sharpe']:6.2f} | Sortino {r['sortino']:6.2f} | "
            f"PF {r['pf']:5.2f} | WR {r['wr']:5.1f}% | {r['n_trades']:4d} trades | "
            f"gap {r['regime_gap']:5.3f} | G {r['sharpe_green']:5.2f} R {r['sharpe_red']:5.2f}"
        )

    # Champion specifically
    champ = results_df[results_df["config"] == "TP25_SL4_3"]
    if len(champ) > 0:
        c = champ.iloc[0]
        log.info(f"\n  CHAMPION: Sharpe {c['sharpe']:.2f}, PF {c['pf']:.2f}, WR {c['wr']:.1f}%, "
                 f"{c['n_trades']} trades, regime_gap {c['regime_gap']:.3f}")

    # Robustness: how many nearby configs also pass?
    nearby = results_df[
        (results_df["phase"] == "tp_sl_sweep") &
        (results_df["tp"].between(20, 30)) &
        (results_df["sl_long"].between(3, 6)) &
        (results_df["sl_short"].between(2, 5))
    ]
    n_passing = len(nearby[nearby["regime_gap"] < 0.50])
    log.info(f"\n  ROBUSTNESS: {n_passing}/{len(nearby)} nearby TP/SL configs pass regime gate")

    profitable = nearby[nearby["sharpe"] > 0]
    log.info(f"  ROBUSTNESS: {len(profitable)}/{len(nearby)} nearby configs have positive Sharpe")

    elapsed = time.time() - t0
    log.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info("Done.")


if __name__ == "__main__":
    main()
