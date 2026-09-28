#!/usr/bin/env python3
"""macro_swing_v1 NAAIM addendum.

Re-uses the prior harness (load_daily, perf_metrics, run_signal_series,
equity_from_daily_pnl, COST_RT) by importing from the original backtest.py
in /home/jupiter/Lvl3Quant/output/macro_swing_v1/backtest.py.

6 cells to test (NAAIM-gated swing on SPY daily, 2010-01-04 to 2026-06-04):
  1. NAAIM<30 long, hold 5d   (1w mean-reversion long)
  2. NAAIM<30 long, hold 10d  (2w mean-reversion long)
  3. NAAIM<30 long, hold 20d  (4w mean-reversion long)
  4. NAAIM>90 short, hold 5d  (1w mean-reversion short)
  5. NAAIM>90 short, hold 10d (2w mean-reversion short)
  6. NAAIM divergence: 4w NAAIM change vs 4w SPY change, counter-trend, hold 10d

Gates: Sharpe>1.0, PF>1.4, WR>0.50, MaxDD>-20%, n_trades>=30, cal_year_sharpe_gap<=0.5
Cost: SPY market RT = 1.8 bps (round-trip applied at exit)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime

import mlflow

# Import the harness from the prior cell run
sys.path.insert(0, "/home/jupiter/Lvl3Quant/output/macro_swing_v1")
from backtest import (  # noqa: E402
    load_daily,
    perf_metrics,
    run_signal_series,
    equity_from_daily_pnl,
    COST_RT,
)

OUT = "/home/jupiter/Lvl3Quant/output/macro_swing_v1_naaim_addendum"
NAAIM_PATH = "/home/jupiter/Lvl3Quant/data/derived/naaim_weekly.parquet"

BT_START = pd.Timestamp("2010-01-04")
BT_END = pd.Timestamp("2026-06-04")


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "run_log.txt"), "a") as f:
        f.write(f"[{datetime.utcnow().isoformat()}] {msg}\n")


def load_with_naaim():
    df = load_daily()
    df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    # restrict to backtest window
    df = df[(df["date"] >= BT_START) & (df["date"] <= BT_END)].reset_index(drop=True)

    naaim = pd.read_parquet(NAAIM_PATH)
    naaim["date"] = pd.to_datetime(naaim["date"]).dt.normalize()
    naaim = naaim[["date", "naaim", "naaim_change_1w", "naaim_change_4w", "naaim_zscore_52w"]]

    d = df.merge(naaim, on="date", how="left").sort_values("date").reset_index(drop=True)
    # data already daily ffilled in source; do a safety ffill
    d["naaim"] = d["naaim"].ffill(limit=10)
    d["naaim_change_4w"] = d["naaim_change_4w"].ffill(limit=10)
    return d


# ----------------------------- STRATEGIES -----------------------------

def naaim_low_long(df, low_thresh, hold):
    """Long SPY when NAAIM < low_thresh; fixed-hold exit."""
    sig = df["naaim"] < low_thresh
    return run_signal_series(df, sig, hold_days=hold, side="long")


def naaim_high_short(df, high_thresh, hold):
    """Short SPY when NAAIM > high_thresh; fixed-hold exit."""
    sig = df["naaim"] > high_thresh
    return run_signal_series(df, sig, hold_days=hold, side="short")


def naaim_divergence_countertrend(df, hold):
    """Divergence fade:
       - SPY falling 4w (ret_4w < -2%) AND NAAIM rising (naaim_change_4w > +10) -> LONG (managers buying weakness)
       - SPY rising 4w (ret_4w > +2%)  AND NAAIM falling (naaim_change_4w < -10) -> SHORT (managers selling strength)
       We can't trade two sides in a single run_signal_series call, so we do two passes
       and combine the daily_ret series and the trade logs.
    """
    d = df.copy()
    d["spy_ret_4w"] = d["spy_adj"].pct_change(20)

    sig_long = (d["spy_ret_4w"] < -0.02) & (d["naaim_change_4w"] > 10)
    sig_short = (d["spy_ret_4w"] > 0.02) & (d["naaim_change_4w"] < -10)

    tl_long, dr_long = run_signal_series(d, sig_long, hold_days=hold, side="long")
    tl_short, dr_short = run_signal_series(d, sig_short, hold_days=hold, side="short")

    # combine
    tl = pd.concat([tl_long, tl_short], ignore_index=True).sort_values("entry_date").reset_index(drop=True)
    dr = dr_long.add(dr_short, fill_value=0.0)
    return tl, dr


# ----------------------------- RUNNER -----------------------------

def run_cells(df):
    rows = []
    equity_long = []

    def record(cell, trade_log, dr):
        if trade_log is None:
            return
        dr = dr.copy()
        dr.index = pd.to_datetime(dr.index)
        m = perf_metrics(dr, trade_log)
        m["cell"] = cell
        rows.append(m)
        eq = equity_from_daily_pnl(dr)
        for dt, v in eq.items():
            equity_long.append({"cell": cell, "date": dt, "equity": float(v)})

    # --- Cells 1-3: NAAIM < 30 LONG, holds 5/10/20d
    for hold in [5, 10, 20]:
        cell = f"naaim_lt30_long_hold{hold}"
        tl, dr = naaim_low_long(df, 30, hold)
        log(f"{cell}: n_trades={len(tl)}")
        record(cell, tl, dr)

    # --- Cells 4-5: NAAIM > 90 SHORT, holds 5/10d
    for hold in [5, 10]:
        cell = f"naaim_gt90_short_hold{hold}"
        tl, dr = naaim_high_short(df, 90, hold)
        log(f"{cell}: n_trades={len(tl)}")
        record(cell, tl, dr)

    # --- Cell 6: divergence fade, hold 10d (2w)
    cell = "naaim_divergence_fade_hold10"
    tl, dr = naaim_divergence_countertrend(df, 10)
    log(f"{cell}: n_trades={len(tl)}")
    record(cell, tl, dr)

    results = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    eq_df = pd.DataFrame(equity_long)
    return results, eq_df


def passes_gates(r):
    try:
        return (r["sharpe"] > 1.0 and r["pf"] > 1.4 and r["wr"] > 0.50
                and r["max_dd"] > -0.20 and r["n_trades"] >= 30
                and (np.isnan(r["calendar_year_sharpe_gap"]) or r["calendar_year_sharpe_gap"] <= 0.5))
    except Exception:
        return False


def main():
    os.makedirs(OUT, exist_ok=True)
    # reset run_log
    with open(os.path.join(OUT, "run_log.txt"), "w") as f:
        f.write("")
    log("=== backtest_addendum.py START ===")
    log(f"COST_RT (from prior harness) = {COST_RT}  (1.8 bps RT confirmed)")

    df = load_with_naaim()
    log(f"data rows={len(df)}  range={df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}")
    log(f"NAAIM non-null rows: {df['naaim'].notna().sum()}")

    # MLflow
    mlflow_on = False
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("macro_swing_v1")
        mlflow.start_run(run_name="macro_swing_v1_naaim_addendum")
        mlflow.log_param("cost_rt_bps", 1.8)
        mlflow.log_param("data_start", str(df["date"].iloc[0].date()))
        mlflow.log_param("data_end", str(df["date"].iloc[-1].date()))
        mlflow.log_param("naaim_path", NAAIM_PATH)
        mlflow.log_param("n_cells", 6)
        mlflow_on = True
    except Exception as e:
        log(f"MLflow unavailable: {e}")

    results, equity_df = run_cells(df)
    results["passes_gates"] = results.apply(passes_gates, axis=1)

    cols_order = ["cell", "sharpe", "sortino", "pf", "wr", "max_dd", "cagr",
                  "n_trades", "ann_vol", "ann_ret", "total_ret",
                  "calendar_year_sharpe_gap", "passes_gates", "sharpe_by_year_json"]
    results = results[[c for c in cols_order if c in results.columns]]

    results_path = os.path.join(OUT, "results.csv")
    eq_path = os.path.join(OUT, "equity_curves.csv")
    results.to_csv(results_path, index=False)
    equity_df.to_csv(eq_path, index=False)
    log(f"wrote {results_path}  ({len(results)} cells)")
    log(f"wrote {eq_path}  ({len(equity_df)} rows)")

    txt_path = os.path.join(OUT, "top_strategies.txt")
    with open(txt_path, "w") as f:
        f.write("macro_swing_v1 NAAIM addendum — 6 cells\n")
        f.write("=" * 72 + "\n")
        f.write(f"Data: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}\n")
        f.write(f"Cost: SPY market RT = 1.8 bps\n")
        f.write(f"Gates: Sharpe>1, PF>1.4, WR>0.50, DD>-20%, n_trades>=30, cy_gap<=0.5\n\n")
        for i, r in results.iterrows():
            f.write(f"--- #{i+1} : {r['cell']}\n")
            f.write(f"  Sharpe={r['sharpe']:.2f}  Sortino={r['sortino']:.2f}  PF={r['pf']:.2f}  "
                    f"WR={r['wr']:.2%}  MaxDD={r['max_dd']:.2%}  CAGR={r['cagr']:.2%}  "
                    f"n_trades={int(r['n_trades'])}  cy_gap={r['calendar_year_sharpe_gap']:.2f}\n")
            f.write(f"  passes_gates={r['passes_gates']}\n")
            f.write(f"  by_year_sharpe={r['sharpe_by_year_json']}\n\n")
        passers = results[results["passes_gates"]]
        f.write(f"\nCells PASSING all gates: {len(passers)}\n")
        for _, r in passers.iterrows():
            f.write(f"  {r['cell']}  Sharpe={r['sharpe']:.2f}  PF={r['pf']:.2f}  WR={r['wr']:.2%}\n")
    log(f"wrote {txt_path}")

    if mlflow_on:
        try:
            mlflow.log_artifact(results_path)
            mlflow.log_artifact(eq_path)
            mlflow.log_artifact(txt_path)
            mlflow.log_artifact(os.path.join(OUT, "run_log.txt"))
            if len(results) > 0:
                best = results.iloc[0]
                mlflow.log_metric("best_sharpe", float(best["sharpe"]) if np.isfinite(best["sharpe"]) else 0.0)
                mlflow.log_metric("best_pf", float(best["pf"]) if np.isfinite(best["pf"]) else 0.0)
                mlflow.log_metric("best_wr", float(best["wr"]) if np.isfinite(best["wr"]) else 0.0)
                mlflow.log_metric("best_max_dd", float(best["max_dd"]) if np.isfinite(best["max_dd"]) else 0.0)
                mlflow.log_metric("best_cagr", float(best["cagr"]) if np.isfinite(best["cagr"]) else 0.0)
                mlflow.log_metric("n_cells", len(results))
                mlflow.log_metric("n_passers", int(results["passes_gates"].sum()))
            mlflow.end_run()
            log("MLflow run logged")
        except Exception as e:
            log(f"MLflow log failed: {e}")

    log("=== backtest_addendum.py DONE ===")
    print("\nResults by Sharpe:")
    print(results.to_string(index=False))


if __name__ == "__main__":
    main()
