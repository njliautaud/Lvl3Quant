#!/usr/bin/env python3
"""Bar-timing audit: compare BUGGY (mask > bar_start) vs FIXED (mask > bar_end).

The champion_extended_validation.py has a bar timing bug:
  bar_30 = ts_minute.floor("30min") = bar START (e.g., 10:00)
  entry_price = close of bar = close at 10:29
  mask = ts_minute > bar_time  → includes bars 10:01-10:29 BEFORE entry

FIX: mask = ts_minute >= bar_time + 30min → only bars from 10:30 onward

This script runs both versions side-by-side on the saved predictions.
"""

import logging
import json
import time
import warnings
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s",
    level=logging.INFO,
)
log = logging.getLogger(__name__)

LVL3_ROOT = Path("/home/nick/Lvl3Quant")
MINUTE_BAR_DIR = LVL3_ROOT / "data" / "processed" / "mbo_minute_bars_v1"
PRED_PATH = LVL3_ROOT / "output" / "champion_sensitivity_sweep" / "wf_predictions.parquet"
ENHANCED_DAILY = LVL3_ROOT / "output" / "long_horizon_flow_v2" / "enhanced_daily_features.parquet"
OUT_DIR = LVL3_ROOT / "output" / "champion_bartiming_audit"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ES_TICK_SIZE = 0.25
ES_TICK_VALUE = 12.50
COST_RT_TICKS = 0.376 + 1.376  # 1.752

TP_TICKS = 25
SL_LONG_TICKS = 4
SL_SHORT_TICKS = 3
THRESHOLD = 0.05
MAX_HOLD_MIN = 60


def load_minute_bars():
    files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            if "date" not in df.columns:
                date_str = "".join(c for c in f.stem if c.isdigit())[:8]
                if len(date_str) == 8:
                    df["date"] = date_str
            dfs.append(df)
        except Exception:
            pass
    df_all = pd.concat(dfs, ignore_index=True)
    df_all.sort_values(["date", "ts_minute"], inplace=True)
    df_all["ts_minute"] = pd.to_datetime(df_all["ts_minute"])
    log.info(f"Loaded {len(df_all):,} minute bars, {df_all['date'].nunique()} days")
    return df_all


def classify_days(minute_bars):
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
        for date, grp in minute_bars.groupby("date"):
            if len(grp) > 0:
                daily_close[date] = float(grp.iloc[-1]["close"])
    dates_sorted = sorted(daily_close.keys())
    regimes = {}
    for i, d in enumerate(dates_sorted):
        if i == 0:
            regimes[d] = "flat"
            continue
        chg = daily_close[d] - daily_close[dates_sorted[i - 1]]
        regimes[d] = "green" if chg > 5.0 else ("red" if chg < -5.0 else "flat")
    return regimes


def simulate(predictions, minute_bars, use_fixed_timing=False, label=""):
    """Run trade simulation.

    use_fixed_timing=False: BUGGY (mask > bar_start, includes intra-bar data)
    use_fixed_timing=True:  FIXED (mask >= bar_start + 30min, only post-entry data)
    """
    minute_bars_by_date = {
        date: grp.sort_values("ts_minute").reset_index(drop=True)
        for date, grp in minute_bars.groupby("date")
    }

    preds = predictions.copy()
    preds["bar_30"] = pd.to_datetime(preds["bar_30"])
    preds = preds.sort_values(["date", "bar_30"]).reset_index(drop=True)

    trades = []
    for _, row in preds.iterrows():
        pred = row["prediction"]
        date = row["date"]
        bar_time = row["bar_30"]
        entry_price = row["close"]

        if abs(pred) < THRESHOLD:
            continue

        direction = 1 if pred > 0 else -1
        sl_ticks = SL_LONG_TICKS if direction == 1 else SL_SHORT_TICKS
        sl_pts = sl_ticks * ES_TICK_SIZE
        tp_pts = TP_TICKS * ES_TICK_SIZE

        if date not in minute_bars_by_date:
            continue
        day_mins = minute_bars_by_date[date]

        # KEY DIFFERENCE: bar timing
        if use_fixed_timing:
            # FIXED: only bars AFTER the 30-min bar closes
            bar_end = bar_time + pd.Timedelta(minutes=30)
            mask_after = day_mins["ts_minute"] >= bar_end
        else:
            # BUGGY: includes intra-bar minute data before entry
            mask_after = day_mins["ts_minute"] > bar_time

        future_mins = day_mins[mask_after]
        if len(future_mins) == 0:
            continue

        exit_price = None
        exit_reason = None
        exit_time = None
        bars_held = 0
        mfe_ticks = 0.0
        mae_ticks = 0.0

        for _, mbar in future_mins.iterrows():
            bars_held += 1

            # MFE/MAE
            if direction == 1:
                fav = (mbar["high"] - entry_price) / ES_TICK_SIZE
                adv = (entry_price - mbar["low"]) / ES_TICK_SIZE
            else:
                fav = (entry_price - mbar["low"]) / ES_TICK_SIZE
                adv = (mbar["high"] - entry_price) / ES_TICK_SIZE
            mfe_ticks = max(mfe_ticks, fav)
            mae_ticks = max(mae_ticks, adv)

            # SL check first
            if direction == 1:
                if mbar["low"] <= entry_price - sl_pts:
                    exit_price = entry_price - sl_pts
                    exit_reason = "SL"
                    exit_time = mbar["ts_minute"]
                    break
                if mbar["high"] >= entry_price + tp_pts:
                    exit_price = entry_price + tp_pts
                    exit_reason = "TP"
                    exit_time = mbar["ts_minute"]
                    break
            else:
                if mbar["high"] >= entry_price + sl_pts:
                    exit_price = entry_price + sl_pts
                    exit_reason = "SL"
                    exit_time = mbar["ts_minute"]
                    break
                if mbar["low"] <= entry_price - tp_pts:
                    exit_price = entry_price - tp_pts
                    exit_reason = "TP"
                    exit_time = mbar["ts_minute"]
                    break

            if bars_held >= MAX_HOLD_MIN:
                exit_price = mbar["close"]
                exit_reason = "TIMEOUT"
                exit_time = mbar["ts_minute"]
                break

        if exit_price is None:
            exit_price = future_mins.iloc[-1]["close"]
            exit_reason = "EOD"
            exit_time = future_mins.iloc[-1]["ts_minute"]
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
            "mfe_ticks": round(float(mfe_ticks), 2),
            "mae_ticks": round(float(mae_ticks), 2),
        })

    return trades


def compute_metrics(trades, regimes=None, label=""):
    if not trades:
        return {"label": label, "n_trades": 0}

    pnls = np.array([t["net_pnl_ticks"] for t in trades])
    n = len(pnls)
    win_rate = float((pnls > 0).mean())
    gross_wins = pnls[pnls > 0].sum()
    gross_losses = abs(pnls[pnls < 0].sum())
    pf = float(gross_wins / (gross_losses + 1e-8))
    avg_win = float(pnls[pnls > 0].mean()) if (pnls > 0).any() else 0
    avg_loss = float(abs(pnls[pnls < 0].mean())) if (pnls < 0).any() else 0

    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t["date"]] += t["net_pnl_ticks"]
    daily_arr = np.array(list(daily_pnl.values()))
    n_days = len(daily_arr)
    sharpe = float(daily_arr.mean() / (daily_arr.std() + 1e-8) * np.sqrt(252)) if n_days > 1 else 0

    downside = daily_arr[daily_arr < 0]
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-8
    sortino = float(daily_arr.mean() / (downside_std + 1e-8) * np.sqrt(252)) if n_days > 1 else 0

    cum = np.cumsum(daily_arr)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = float(dd.max()) if len(dd) > 0 else 0

    green_days = (daily_arr > 0).sum()
    red_days = (daily_arr < 0).sum()
    flat_days = (daily_arr == 0).sum()

    # Regime
    regime_gap = 999.0
    sharpe_green = sharpe_red = 0
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

    mfes = np.array([t["mfe_ticks"] for t in trades])
    maes = np.array([t["mae_ticks"] for t in trades])

    # Exit reason counts
    reasons = defaultdict(int)
    for t in trades:
        reasons[t["exit_reason"]] += 1

    return {
        "label": label,
        "n_trades": n,
        "n_days": n_days,
        "trades_per_day": round(n / max(n_days, 1), 1),
        "sharpe": round(sharpe, 2),
        "sortino": round(sortino, 2),
        "pf": round(pf, 2),
        "wr": round(win_rate * 100, 1),
        "avg_win": round(avg_win, 2),
        "avg_loss": round(avg_loss, 2),
        "total_pnl_ticks": round(float(pnls.sum()), 1),
        "total_pnl_dollars": round(float(pnls.sum() * ES_TICK_VALUE), 2),
        "max_dd_ticks": round(max_dd, 1),
        "green_days": int(green_days),
        "red_days": int(red_days),
        "regime_gap": round(regime_gap, 3),
        "sharpe_green": round(sharpe_green, 2),
        "sharpe_red": round(sharpe_red, 2),
        "mfe_mean": round(float(mfes.mean()), 2),
        "mfe_p50": round(float(np.median(mfes)), 2),
        "mfe_p90": round(float(np.percentile(mfes, 90)), 2),
        "mae_mean": round(float(maes.mean()), 2),
        "mae_p50": round(float(np.median(maes)), 2),
        "mae_p90": round(float(np.percentile(maes, 90)), 2),
        "exit_TP": reasons.get("TP", 0),
        "exit_SL": reasons.get("SL", 0),
        "exit_TIMEOUT": reasons.get("TIMEOUT", 0),
        "exit_EOD": reasons.get("EOD", 0),
    }


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("BAR-TIMING AUDIT: Buggy vs Fixed simulation")
    log.info("=" * 70)

    predictions = pd.read_parquet(PRED_PATH)
    log.info(f"Predictions: {len(predictions):,}")
    minute_bars = load_minute_bars()
    regimes = classify_days(minute_bars)

    # BUGGY version (current)
    log.info("\n--- BUGGY (mask > bar_start, includes intra-bar data) ---")
    trades_buggy = simulate(predictions, minute_bars, use_fixed_timing=False)
    m_buggy = compute_metrics(trades_buggy, regimes, "BUGGY")

    # FIXED version
    log.info("--- FIXED (mask >= bar_end, only post-entry data) ---")
    trades_fixed = simulate(predictions, minute_bars, use_fixed_timing=True)
    m_fixed = compute_metrics(trades_fixed, regimes, "FIXED")

    # Print comparison
    log.info("\n" + "=" * 70)
    log.info("COMPARISON: BUGGY vs FIXED bar timing")
    log.info("=" * 70)

    for key in ["n_trades", "trades_per_day", "sharpe", "sortino", "pf", "wr",
                "avg_win", "avg_loss", "total_pnl_ticks", "total_pnl_dollars",
                "max_dd_ticks", "green_days", "red_days",
                "regime_gap", "sharpe_green", "sharpe_red",
                "mfe_mean", "mfe_p50", "mfe_p90", "mae_mean", "mae_p50", "mae_p90",
                "exit_TP", "exit_SL", "exit_TIMEOUT", "exit_EOD"]:
        v_b = m_buggy.get(key, "N/A")
        v_f = m_fixed.get(key, "N/A")
        marker = " <<<" if v_b != v_f else ""
        log.info(f"  {key:20s}: {str(v_b):>12s} → {str(v_f):>12s}{marker}")

    # Direction breakdown for FIXED
    log.info("\n--- FIXED: Direction breakdown ---")
    for side in ["LONG", "SHORT"]:
        side_trades = [t for t in trades_fixed if t["direction"] == side]
        if side_trades:
            m_side = compute_metrics(side_trades, regimes, f"FIXED_{side}")
            log.info(f"  {side}: {m_side['n_trades']} trades, Sharpe {m_side['sharpe']}, "
                     f"PF {m_side['pf']}, WR {m_side['wr']}%, "
                     f"MFE_mean {m_side['mfe_mean']}t, MAE_mean {m_side['mae_mean']}t, "
                     f"TP:{m_side['exit_TP']} SL:{m_side['exit_SL']} TO:{m_side['exit_TIMEOUT']} EOD:{m_side['exit_EOD']}")

    # Save
    summary = {"buggy": m_buggy, "fixed": m_fixed}
    with open(OUT_DIR / "bartiming_audit.json", "w") as f:
        json.dump(summary, f, indent=2)

    trades_fixed_df = pd.DataFrame(trades_fixed)
    trades_fixed_df.to_csv(OUT_DIR / "trades_fixed_timing.csv", index=False)

    elapsed = time.time() - t0
    log.info(f"\nTotal time: {elapsed:.0f}s")
    log.info("Done.")


if __name__ == "__main__":
    main()
