#!/usr/bin/env python3
"""Corrected sensitivity sweep with FIXED bar timing.

The champion_extended_validation.py had a critical bar-timing bug where
mask > bar_start checked intra-bar data before the actual entry point.
This script uses the corrected mask >= bar_end.

Also tests wider SL ranges since the tight SL (3-4 ticks) was likely
only "working" due to the bug.

Reuses saved walk-forward predictions from the original sweep.
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
OUT_DIR = LVL3_ROOT / "output" / "champion_sweep_fixed_timing"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ES_TICK_SIZE = 0.25
ES_TICK_VALUE = 12.50
COST_RT_TICKS = 0.376 + 1.376  # 1.752

# Wider grids — the old tight SLs were only "working" due to the bug
TP_GRID = [4, 6, 8, 10, 12, 15, 20, 25, 30, 40]
SL_GRID = [4, 6, 8, 10, 12, 15, 20, 25, 30, 40]  # symmetric for simplicity
THRESHOLD_GRID = [0.05, 0.10, 0.15, 0.20, 0.30, 0.50]
HOLD_GRID = [15, 30, 45, 60, 90, 120]


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


def simulate_fixed(predictions, minute_bars, tp_ticks, sl_ticks, threshold, max_hold_min):
    """Trade simulation with FIXED bar timing (only post-entry bars)."""
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

        if abs(pred) < threshold:
            continue

        direction = 1 if pred > 0 else -1
        sl_pts = sl_ticks * ES_TICK_SIZE
        tp_pts = tp_ticks * ES_TICK_SIZE

        if date not in minute_bars_by_date:
            continue
        day_mins = minute_bars_by_date[date]

        # FIXED: only bars AFTER bar close (bar_time + 30min)
        bar_end = bar_time + pd.Timedelta(minutes=30)
        mask_after = day_mins["ts_minute"] >= bar_end
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

            if bars_held >= max_hold_min:
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
            "net_pnl_ticks": round(float(net_pnl_ticks), 4),
        })

    return trades


def compute_metrics(trades, regimes=None):
    if not trades:
        return {"n_trades": 0, "sharpe": -999, "pf": 0, "wr": 0, "regime_gap": 999}

    pnls = np.array([t["net_pnl_ticks"] for t in trades])
    n = len(pnls)
    if n < 10:
        return {"n_trades": n, "sharpe": -999, "pf": 0, "wr": 0, "regime_gap": 999}

    win_rate = float((pnls > 0).mean())
    gross_wins = pnls[pnls > 0].sum()
    gross_losses = abs(pnls[pnls < 0].sum())
    pf = float(gross_wins / (gross_losses + 1e-8))

    daily_pnl = defaultdict(float)
    for t in trades:
        daily_pnl[t["date"]] += t["net_pnl_ticks"]
    daily_arr = np.array(list(daily_pnl.values()))
    n_days = len(daily_arr)
    sharpe = float(daily_arr.mean() / (daily_arr.std() + 1e-8) * np.sqrt(252)) if n_days > 1 else 0

    downside = daily_arr[daily_arr < 0]
    downside_std = np.sqrt((downside ** 2).mean()) if len(downside) > 0 else 1e-8
    sortino = float(daily_arr.mean() / (downside_std + 1e-8) * np.sqrt(252)) if n_days > 1 else 0

    green_days = int((daily_arr > 0).sum())
    red_days = int((daily_arr < 0).sum())

    # Exit reasons
    reasons = defaultdict(int)
    for t in trades:
        reasons[t["exit_reason"]] += 1

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

    return {
        "n_trades": n,
        "n_days": n_days,
        "trades_per_day": round(n / max(n_days, 1), 1),
        "sharpe": round(sharpe, 3),
        "sortino": round(sortino, 3),
        "pf": round(pf, 3),
        "wr": round(win_rate * 100, 1),
        "total_pnl_ticks": round(float(pnls.sum()), 1),
        "green_days": green_days,
        "red_days": red_days,
        "regime_gap": round(regime_gap, 3),
        "sharpe_green": round(sharpe_green, 2),
        "sharpe_red": round(sharpe_red, 2),
        "exit_TP": reasons.get("TP", 0),
        "exit_SL": reasons.get("SL", 0),
        "exit_TIMEOUT": reasons.get("TIMEOUT", 0),
        "exit_EOD": reasons.get("EOD", 0),
    }


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("CORRECTED SENSITIVITY SWEEP (fixed bar timing)")
    log.info("=" * 70)

    predictions = pd.read_parquet(PRED_PATH)
    log.info(f"Predictions: {len(predictions):,}")
    minute_bars = load_minute_bars()
    regimes = classify_days(minute_bars)

    results = []

    # Phase 1: TP x SL sweep (symmetric SL, threshold=0.05, hold=60)
    log.info("\n--- Phase 1: TP x SL sweep (threshold=0.05, hold=60) ---")
    total = len(TP_GRID) * len(SL_GRID)
    i = 0
    for tp in TP_GRID:
        for sl in SL_GRID:
            i += 1
            trades = simulate_fixed(predictions, minute_bars, tp, sl, 0.05, 60)
            m = compute_metrics(trades, regimes)
            m["config"] = f"TP{tp}_SL{sl}"
            m["tp"] = tp
            m["sl"] = sl
            m["threshold"] = 0.05
            m["hold_min"] = 60
            m["phase"] = "tp_sl"
            results.append(m)
            if i % 20 == 0:
                log.info(f"  {i}/{total} combos...")

    # Phase 2: Threshold sweep on best configs from phase 1
    log.info("\n--- Phase 2: Threshold sweep on best configs ---")
    df_p1 = pd.DataFrame([r for r in results if r["phase"] == "tp_sl"])
    profitable = df_p1[df_p1["sharpe"] > 0].sort_values("sharpe", ascending=False)
    if len(profitable) > 0:
        top_configs = profitable.head(10)
        for _, cfg in top_configs.iterrows():
            for thresh in THRESHOLD_GRID:
                trades = simulate_fixed(predictions, minute_bars, int(cfg["tp"]), int(cfg["sl"]), thresh, 60)
                m = compute_metrics(trades, regimes)
                m["config"] = f"TP{int(cfg['tp'])}_SL{int(cfg['sl'])}_T{thresh}"
                m["tp"] = int(cfg["tp"])
                m["sl"] = int(cfg["sl"])
                m["threshold"] = thresh
                m["hold_min"] = 60
                m["phase"] = "threshold"
                results.append(m)
    else:
        log.info("  NO profitable configs found in Phase 1 — skipping threshold sweep")

    # Phase 3: Hold time sweep
    log.info("\n--- Phase 3: Hold time sweep ---")
    if len(profitable) > 0:
        top3 = profitable.head(3)
        for _, cfg in top3.iterrows():
            for hold in HOLD_GRID:
                trades = simulate_fixed(predictions, minute_bars, int(cfg["tp"]), int(cfg["sl"]), 0.05, hold)
                m = compute_metrics(trades, regimes)
                m["config"] = f"TP{int(cfg['tp'])}_SL{int(cfg['sl'])}_H{hold}"
                m["tp"] = int(cfg["tp"])
                m["sl"] = int(cfg["sl"])
                m["threshold"] = 0.05
                m["hold_min"] = hold
                m["phase"] = "hold"
                results.append(m)
    else:
        log.info("  Skipping (no profitable configs)")

    # Save
    results_df = pd.DataFrame(results)
    results_df.to_csv(OUT_DIR / "sweep_results_fixed.csv", index=False)

    # Summary
    log.info("\n" + "=" * 70)
    log.info("SUMMARY — TP x SL heatmap (Sharpe)")
    log.info("=" * 70)

    p1 = results_df[results_df["phase"] == "tp_sl"]
    pivot = p1.pivot_table(values="sharpe", index="sl", columns="tp", aggfunc="first")
    log.info(f"\n{pivot.to_string()}")

    log.info("\n--- PF heatmap ---")
    pivot_pf = p1.pivot_table(values="pf", index="sl", columns="tp", aggfunc="first")
    log.info(f"\n{pivot_pf.to_string()}")

    log.info("\n--- WR heatmap ---")
    pivot_wr = p1.pivot_table(values="wr", index="sl", columns="tp", aggfunc="first")
    log.info(f"\n{pivot_wr.to_string()}")

    log.info("\n--- Trades/day heatmap ---")
    pivot_tpd = p1.pivot_table(values="trades_per_day", index="sl", columns="tp", aggfunc="first")
    log.info(f"\n{pivot_tpd.to_string()}")

    # Top profitable configs
    profitable_all = results_df[results_df["sharpe"] > 0].sort_values("sharpe", ascending=False)
    if len(profitable_all) > 0:
        log.info(f"\n--- Top 20 profitable configs (Sharpe > 0) ---")
        for _, r in profitable_all.head(20).iterrows():
            rg = f"gap {r['regime_gap']:.3f}" if r['regime_gap'] < 100 else "gap N/A"
            log.info(f"  {r['config']:30s} S={r['sharpe']:6.3f} PF={r['pf']:5.3f} WR={r['wr']:5.1f}% "
                     f"N={int(r['n_trades']):4d} TP:{int(r['exit_TP'])} SL:{int(r['exit_SL'])} "
                     f"TO:{int(r['exit_TIMEOUT'])} {rg}")

        # Regime-passing subset
        regime_pass = profitable_all[profitable_all["regime_gap"] < 0.50]
        if len(regime_pass) > 0:
            log.info(f"\n--- Top 10 REGIME-PASSING configs (gap < 0.50) ---")
            for _, r in regime_pass.head(10).iterrows():
                log.info(f"  {r['config']:30s} S={r['sharpe']:6.3f} PF={r['pf']:5.3f} WR={r['wr']:5.1f}% "
                         f"N={int(r['n_trades']):4d} gap={r['regime_gap']:.3f} "
                         f"G={r['sharpe_green']:.2f} R={r['sharpe_red']:.2f}")
        else:
            log.info("\n  NO regime-passing profitable configs found")
    else:
        log.info("\n  NO profitable configs found at all — model has no tradeable edge at 30-min bars with fixed timing")

    n_profitable = len(results_df[results_df["sharpe"] > 0])
    n_total = len(results_df)
    log.info(f"\n  OVERALL: {n_profitable}/{n_total} configs profitable ({n_profitable/n_total*100:.0f}%)")

    elapsed = time.time() - t0
    log.info(f"\nTotal time: {elapsed:.0f}s ({elapsed/60:.1f} min)")
    log.info("Done.")


if __name__ == "__main__":
    main()
