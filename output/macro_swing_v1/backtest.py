#!/usr/bin/env python3
"""macro_swing_v1 backtest grid.

Strategies (cells):
  1. NAAIM extremes mean-reversion        -- placeholder if naaim missing
  2. VIX-spike mean reversion             -- VIX > p{90,95,99}, hold {5,10,20}d
  3. Trend + low-vol                      -- 200d MA up & VIX<{15,20,25}; exit MA flat or VIX>{25,30,35}
  4. NAAIM divergence                     -- placeholder
  5. Monthly seasonality (turn-of-month)  -- T-1..T+3 long
  6. VIX term structure                   -- VIX/VIX3M backwardation regime

Cost model (HC #541 R1): SPY market RT = 1.8 bps  (= 9 bps each side  >> we use 1.8 bps RT applied at exit)
We apply ROUND-TRIP cost on the exit bar = 0.00018 * notional.

All trades: long-only or short-only SPY shares, full notional, no leverage.
Equity tracked in arithmetic returns. Daily mark-to-market.
"""

import os
import json
import math
import numpy as np
import pandas as pd
from datetime import datetime
from itertools import product

import mlflow

OUT = "/home/jupiter/Lvl3Quant/output/macro_swing_v1"
DATA = "/home/jupiter/Lvl3Quant/data"
NAAIM_PATH = os.path.join(DATA, "naaim_weekly.parquet")

COST_RT = 0.00018  # 1.8 bps RT (HC #541 R1)
ANNUALIZATION_DAYS = 252


def log(msg):
    print(msg, flush=True)
    with open(os.path.join(OUT, "run_log.txt"), "a") as f:
        f.write(f"[{datetime.utcnow().isoformat()}] {msg}\n")


# ----------------------------- DATA LOAD -----------------------------

def load_daily():
    spy = pd.read_parquet(os.path.join(OUT, "spy_daily.parquet"))
    vix = pd.read_parquet(os.path.join(OUT, "vix_daily.parquet"))
    vix3m = pd.read_parquet(os.path.join(OUT, "vix3m_daily.parquet"))
    for df in (spy, vix, vix3m):
        df["date"] = pd.to_datetime(df["date"]).dt.normalize()
    spy = spy[["date", "open", "high", "low", "close", "adj_close", "volume"]].rename(
        columns={"close": "spy_close", "adj_close": "spy_adj", "open": "spy_open",
                 "high": "spy_high", "low": "spy_low", "volume": "spy_vol"})
    vix = vix[["date", "close"]].rename(columns={"close": "vix_close"})
    vix3m = vix3m[["date", "close"]].rename(columns={"close": "vix3m_close"})
    df = spy.merge(vix, on="date", how="left").merge(vix3m, on="date", how="left")
    df = df.sort_values("date").reset_index(drop=True)
    # Use adj_close for returns
    df["ret"] = df["spy_adj"].pct_change()
    df["ma200"] = df["spy_adj"].rolling(200).mean()
    df["ma200_slope_20d"] = df["ma200"].diff(20)
    df["vix_pct_252d"] = df["vix_close"].rolling(252).rank(pct=True)
    df["vix_term"] = df["vix3m_close"] / df["vix_close"]  # >1 = contango, <1 = backwardation
    df["dom"] = df["date"].dt.day
    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["is_last_trading_day_of_month"] = (df.groupby([df["date"].dt.year, df["date"].dt.month])["date"]
                                              .transform("max") == df["date"])
    return df


def load_naaim():
    if not os.path.exists(NAAIM_PATH):
        return None
    n = pd.read_parquet(NAAIM_PATH)
    # Expected cols (best-effort): date, naaim
    return n


# ----------------------------- METRICS -----------------------------

def equity_from_daily_pnl(pnl):
    """pnl: daily arithmetic returns (decimals). Returns equity curve starting at 1.0."""
    return (1.0 + pnl.fillna(0)).cumprod()


def perf_metrics(daily_ret, trade_log):
    """daily_ret: pd.Series of daily strategy returns (decimal).
       trade_log: DataFrame with entry_date, exit_date, ret_gross, ret_net, side, hold_days, mfe, mae
    """
    daily_ret = daily_ret.fillna(0).astype(float)
    n_days = (daily_ret != 0).sum()
    if n_days < 5 or len(trade_log) == 0:
        return {"sharpe": np.nan, "sortino": np.nan, "pf": np.nan, "wr": np.nan,
                "max_dd": np.nan, "cagr": np.nan, "n_trades": len(trade_log),
                "ann_vol": np.nan, "ann_ret": np.nan, "total_ret": np.nan,
                "calendar_year_sharpe_gap": np.nan}

    mu = daily_ret.mean()
    sd = daily_ret.std(ddof=1)
    sharpe = (mu / sd) * math.sqrt(ANNUALIZATION_DAYS) if sd > 0 else 0.0

    downside = daily_ret[daily_ret < 0]
    dsd = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (mu / dsd) * math.sqrt(ANNUALIZATION_DAYS) if (dsd and dsd > 0) else np.nan

    wins = trade_log[trade_log["ret_net"] > 0]["ret_net"].sum()
    losses = -trade_log[trade_log["ret_net"] < 0]["ret_net"].sum()
    pf = (wins / losses) if losses > 0 else (np.inf if wins > 0 else np.nan)

    wr = (trade_log["ret_net"] > 0).mean()

    eq = equity_from_daily_pnl(daily_ret)
    peak = eq.cummax()
    dd = (eq / peak - 1.0)
    max_dd = dd.min()

    total_ret = eq.iloc[-1] - 1.0
    days = max(1, len(daily_ret))
    years = days / ANNUALIZATION_DAYS
    cagr = (eq.iloc[-1]) ** (1 / years) - 1 if eq.iloc[-1] > 0 else np.nan

    # calendar-year Sharpe gap
    cy = daily_ret.groupby(daily_ret.index.year)
    sharpe_by_year = {}
    for y, s in cy:
        if s.std(ddof=1) > 0 and len(s) > 20:
            sharpe_by_year[y] = (s.mean() / s.std(ddof=1)) * math.sqrt(ANNUALIZATION_DAYS)
    if len(sharpe_by_year) >= 2:
        vals = list(sharpe_by_year.values())
        gap = max(vals) - min(vals)
    else:
        gap = np.nan

    return {
        "sharpe": sharpe, "sortino": sortino, "pf": pf, "wr": wr,
        "max_dd": max_dd, "cagr": cagr, "n_trades": len(trade_log),
        "ann_vol": sd * math.sqrt(ANNUALIZATION_DAYS), "ann_ret": mu * ANNUALIZATION_DAYS,
        "total_ret": total_ret, "calendar_year_sharpe_gap": gap,
        "sharpe_by_year_json": json.dumps({int(k): round(float(v), 3) for k, v in sharpe_by_year.items()}),
    }


# ----------------------------- BACKTEST CORE -----------------------------

def run_signal_series(df, entry_signal, exit_signal=None, hold_days=None, side="long",
                      cooldown_days=0, max_concurrent=1):
    """
    Generic event-driven backtest.

    df: daily dataframe with 'date','ret','spy_adj' indexed 0..N-1
    entry_signal: bool Series aligned to df (True on the bar we decide to enter; entry executes at NEXT bar's open).
    exit_signal: bool Series aligned to df, optional (exits on NEXT bar's open). If None, use hold_days.
    hold_days: int (fixed-holding-period exit, exits at open of entry+hold_days+1)
    side: 'long' or 'short'
    cooldown_days: minimum gap between exit and next entry
    max_concurrent: cap on simultaneously open trades (default 1, no pyramiding)

    Returns: trade_log DataFrame, daily_ret Series (indexed by date)
    """
    df = df.reset_index(drop=True).copy()
    n = len(df)
    daily_ret = pd.Series(0.0, index=df["date"])

    trades = []
    open_trades = []  # list of dicts: entry_idx, side
    last_exit_idx = -10**9
    sign = 1 if side == "long" else -1

    # We must enter on day t+1 open after signal on day t. Exit similarly.
    # For simplicity, use close-to-close return on adj_close, with entry on signal+1 close and exit on signal+exit+1 close.
    # This avoids needing same-day open/close handling; one-bar delay is realistic for swing.

    for t in range(n):
        # process exits at start of bar t (means: trade entered earlier, exit decision was triggered on t-1, exit closes at t)
        still_open = []
        for tr in open_trades:
            exit_now = False
            if hold_days is not None and (t - tr["entry_idx"]) >= hold_days:
                exit_now = True
            if exit_signal is not None and t > 0 and bool(exit_signal.iloc[t - 1]):
                exit_now = True
            # Force exit at the last bar
            if t == n - 1:
                exit_now = True
            if exit_now:
                # exit at close of bar t (adj_close)
                entry_px = df["spy_adj"].iloc[tr["entry_idx"]]
                exit_px = df["spy_adj"].iloc[t]
                gross_ret = sign * (exit_px / entry_px - 1.0)
                net_ret = gross_ret - COST_RT
                # Apply the net P&L spread across the holding days via daily MTM with cost subtracted at exit
                # Daily MTM:
                for k in range(tr["entry_idx"] + 1, t + 1):
                    daily_ret.iloc[k] = daily_ret.iloc[k] + sign * df["ret"].iloc[k]
                # subtract cost on exit day
                daily_ret.iloc[t] = daily_ret.iloc[t] - COST_RT
                # MFE/MAE within trade window
                window = df["spy_adj"].iloc[tr["entry_idx"]:t + 1].values
                if len(window) > 1:
                    rels = (window / entry_px - 1.0) * sign
                    mfe = float(rels.max())
                    mae = float(rels.min())
                else:
                    mfe = mae = 0.0
                trades.append({
                    "entry_date": df["date"].iloc[tr["entry_idx"]],
                    "exit_date": df["date"].iloc[t],
                    "side": side,
                    "hold_days": t - tr["entry_idx"],
                    "entry_px": float(entry_px), "exit_px": float(exit_px),
                    "ret_gross": float(gross_ret), "ret_net": float(net_ret),
                    "mfe": mfe, "mae": mae,
                })
                last_exit_idx = t
            else:
                still_open.append(tr)
        open_trades = still_open

        # process entries: signal on t-1 -> enter on t (open trade for bar t+1 onwards)
        if t > 0 and bool(entry_signal.iloc[t - 1]) and len(open_trades) < max_concurrent:
            if (t - last_exit_idx) >= cooldown_days:
                open_trades.append({"entry_idx": t})

    trade_log = pd.DataFrame(trades)
    return trade_log, daily_ret


# ----------------------------- STRATEGIES -----------------------------

def strat_vix_spike(df, pct_thresh, hold):
    sig_entry = df["vix_pct_252d"] > pct_thresh
    return run_signal_series(df, sig_entry, hold_days=hold, side="long")


def strat_trend_lowvol(df, vix_enter, vix_exit, slope_lookback=20):
    sig_entry = (df["spy_adj"] > df["ma200"]) & (df["ma200_slope_20d"] > 0) & (df["vix_close"] < vix_enter)
    sig_exit = (df["ma200_slope_20d"] <= 0) | (df["vix_close"] > vix_exit)
    return run_signal_series(df, sig_entry, exit_signal=sig_exit, side="long")


def strat_tom(df, days_before=1, days_after=3):
    """Turn-of-month: enter days_before trading days before month end, hold days_before+days_after."""
    n = len(df)
    sig_entry = pd.Series(False, index=df.index)
    # find last-trading-day-of-month indices
    eom_idx = df.index[df["is_last_trading_day_of_month"]].tolist()
    for i in eom_idx:
        signal_bar = i - days_before  # signal fires here -> entry at i-days_before+1
        if signal_bar >= 0 and signal_bar < n:
            sig_entry.iloc[signal_bar] = True
    hold = days_before + days_after
    return run_signal_series(df, sig_entry, hold_days=hold, side="long")


def strat_vix_term(df, ratio_thresh, hold):
    """Backwardation regime: VIX > VIX3M (ratio < ratio_thresh) -> long SPY (mean-reversion of fear)."""
    sig_entry = df["vix_term"] < ratio_thresh
    return run_signal_series(df, sig_entry, hold_days=hold, side="long")


def strat_naaim_extreme(df, naaim_join, low, high, hold):
    """Long when naaim<low, short when naaim>high."""
    # naaim_join: df with date,naaim. Forward-fill weekly to daily.
    if naaim_join is None:
        return None, None
    d = df.merge(naaim_join, on="date", how="left").sort_values("date").reset_index(drop=True)
    d["naaim"] = d["naaim"].ffill(limit=10)
    sig_long = d["naaim"] < low
    return run_signal_series(d, sig_long, hold_days=hold, side="long")


def strat_naaim_divergence(df, naaim_join, hold):
    if naaim_join is None:
        return None, None
    d = df.merge(naaim_join, on="date", how="left").sort_values("date").reset_index(drop=True)
    d["naaim"] = d["naaim"].ffill(limit=10)
    d["naaim_4w_chg"] = d["naaim"].diff(20)
    d["spy_4w_chg"] = d["spy_adj"].pct_change(20)
    # bearish divergence: SPY up, NAAIM down -> short
    sig_short = (d["spy_4w_chg"] > 0.02) & (d["naaim_4w_chg"] < -10)
    return run_signal_series(d, sig_short, hold_days=hold, side="short")


# ----------------------------- GRID RUNNER -----------------------------

def run_all(df, naaim_df):
    daily_ret = lambda s: s
    rows = []
    equity_long = []  # for equity_curves.csv

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

    # --- VIX-spike mean reversion: 3 pct thresholds × 3 holds = 9 cells
    for pct, hold in product([0.90, 0.95, 0.99], [5, 10, 20]):
        cell = f"vix_spike_p{int(pct*100)}_hold{hold}"
        tl, dr = strat_vix_spike(df, pct, hold)
        record(cell, tl, dr)

    # --- Trend + low-vol: 3 entry × 3 exit = 9 cells
    for vin, vout in product([15, 20, 25], [25, 30, 35]):
        cell = f"trend_lowvol_inV{vin}_outV{vout}"
        tl, dr = strat_trend_lowvol(df, vin, vout)
        record(cell, tl, dr)

    # --- Turn-of-month: small grid
    for db, da in [(1, 3), (1, 5), (2, 3), (0, 5)]:
        cell = f"tom_pre{db}_post{da}"
        tl, dr = strat_tom(df, db, da)
        record(cell, tl, dr)

    # --- VIX term structure backwardation
    for ratio, hold in product([1.0, 0.98, 0.95], [5, 10, 20]):
        cell = f"vixterm_ratio{ratio:.2f}_hold{hold}"
        tl, dr = strat_vix_term(df, ratio, hold)
        record(cell, tl, dr)

    # --- NAAIM cells (only if data present)
    if naaim_df is not None:
        for low, high, hold in product([30], [80], [5, 10, 20]):
            cell = f"naaim_extreme_lo{low}_hi{high}_hold{hold}_long"
            tl, dr = strat_naaim_extreme(df, naaim_df, low, high, hold)
            record(cell, tl, dr)
        for hold in [5, 10, 20]:
            cell = f"naaim_divergence_short_hold{hold}"
            tl, dr = strat_naaim_divergence(df, naaim_df, hold)
            record(cell, tl, dr)
    else:
        log("[grid] NAAIM cells SKIPPED (data not yet ingested)")

    results = pd.DataFrame(rows).sort_values("sharpe", ascending=False).reset_index(drop=True)
    eq_df = pd.DataFrame(equity_long)
    return results, eq_df


# ----------------------------- MAIN -----------------------------

def main():
    log("=== backtest.py START ===")
    df = load_daily()
    log(f"loaded daily: rows={len(df)}  date_range={df['date'].iloc[0]} -> {df['date'].iloc[-1]}")
    naaim = load_naaim()
    log(f"naaim loaded: {'YES' if naaim is not None else 'NO (skip cells)'}")

    # MLflow tracking
    try:
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("macro_swing_v1")
        active = mlflow.start_run(run_name="macro_swing_v1")
        mlflow.log_param("cost_rt_bps", 1.8)
        mlflow.log_param("data_start", str(df["date"].iloc[0].date()))
        mlflow.log_param("data_end", str(df["date"].iloc[-1].date()))
        mlflow.log_param("naaim_available", naaim is not None)
        mlflow_on = True
    except Exception as e:
        log(f"MLflow unavailable: {e}")
        mlflow_on = False

    results, equity_df = run_all(df, naaim)

    # write outputs
    results_path = os.path.join(OUT, "strategy_grid_results.csv")
    eq_path = os.path.join(OUT, "equity_curves.csv")
    cols_order = ["cell", "sharpe", "sortino", "pf", "wr", "max_dd", "cagr",
                  "n_trades", "ann_vol", "ann_ret", "total_ret",
                  "calendar_year_sharpe_gap", "sharpe_by_year_json"]
    results = results[[c for c in cols_order if c in results.columns]]
    results.to_csv(results_path, index=False)
    equity_df.to_csv(eq_path, index=False)
    log(f"wrote {results_path}  ({len(results)} cells)")
    log(f"wrote {eq_path}  ({len(equity_df)} rows)")

    # gate evaluation
    def passes_gates(r):
        try:
            return (r["sharpe"] > 1.0 and r["pf"] > 1.4 and r["wr"] > 0.50
                    and r["max_dd"] > -0.20 and r["n_trades"] >= 30
                    and (np.isnan(r["calendar_year_sharpe_gap"]) or r["calendar_year_sharpe_gap"] <= 0.5))
        except Exception:
            return False
    results["passes_gates"] = results.apply(passes_gates, axis=1)
    results.to_csv(results_path, index=False)

    # top 5
    top5 = results.head(5)
    txt_path = os.path.join(OUT, "top5_swing_strategies.txt")
    with open(txt_path, "w") as f:
        f.write("TOP 5 macro_swing_v1 cells by Sharpe\n")
        f.write("=" * 70 + "\n")
        f.write(f"Data: {df['date'].iloc[0].date()} -> {df['date'].iloc[-1].date()}\n")
        f.write(f"Cost: SPY market RT = 1.8 bps (HC #541 R1)\n")
        f.write(f"Cells run: {len(results)}\n")
        f.write(f"NAAIM data available: {naaim is not None}\n\n")
        for i, r in top5.iterrows():
            f.write(f"--- #{i+1} : {r['cell']}\n")
            f.write(f"  Sharpe={r['sharpe']:.2f}  Sortino={r['sortino']:.2f}  PF={r['pf']:.2f}  "
                    f"WR={r['wr']:.2%}  MaxDD={r['max_dd']:.2%}  CAGR={r['cagr']:.2%}  "
                    f"n_trades={int(r['n_trades'])}  cal_year_sharpe_gap={r['calendar_year_sharpe_gap']:.2f}\n")
            f.write(f"  by_year_sharpe={r['sharpe_by_year_json']}\n")
            f.write(f"  passes_gates={r['passes_gates']}\n\n")
        # gate-pass summary
        passers = results[results["passes_gates"]]
        f.write(f"\nCells PASSING all HC #428 R1 gates (Sharpe>1, PF>1.4, WR>0.50, DD>-20%, n>=30, gap<=0.5): {len(passers)}\n")
        for _, r in passers.iterrows():
            f.write(f"  {r['cell']}  Sharpe={r['sharpe']:.2f}  PF={r['pf']:.2f}  WR={r['wr']:.2%}\n")
    log(f"wrote {txt_path}")

    # MLflow log
    if mlflow_on:
        try:
            mlflow.log_artifact(results_path)
            mlflow.log_artifact(eq_path)
            mlflow.log_artifact(txt_path)
            mlflow.log_artifact(os.path.join(OUT, "run_log.txt"))
            if len(top5) > 0:
                best = top5.iloc[0]
                mlflow.log_metric("best_sharpe", float(best["sharpe"]))
                mlflow.log_metric("best_pf", float(best["pf"]) if np.isfinite(best["pf"]) else 0.0)
                mlflow.log_metric("best_wr", float(best["wr"]))
                mlflow.log_metric("best_max_dd", float(best["max_dd"]))
                mlflow.log_metric("best_cagr", float(best["cagr"]))
                mlflow.log_metric("n_cells", len(results))
                mlflow.log_metric("n_passers", len(results[results["passes_gates"]]))
            mlflow.end_run()
            log("MLflow run logged")
        except Exception as e:
            log(f"MLflow log failed: {e}")

    log("=== backtest.py DONE ===")
    print("\nTOP 5 by Sharpe:")
    print(top5.to_string(index=False))


if __name__ == "__main__":
    main()
