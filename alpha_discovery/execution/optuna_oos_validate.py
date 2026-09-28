#!/usr/bin/env python3
"""OOS validation for Optuna top config.

Splits cached fill_sim trades by DATE (not random — preserves temporal structure):
  - Train period: first 70% of unique dates
  - OOS period:   last 30% of unique dates

Re-runs Optuna 1500 trials on TRAIN, then evaluates the best config on OOS.
Reports train vs OOS Sortino delta to detect overfit.
"""
from __future__ import annotations
import argparse, logging
from pathlib import Path
import numpy as np, pandas as pd, optuna

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [OOS] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

COMMISSION_TICKS = 0.376


def load_trades(p):
    df = pd.read_parquet(p)
    df = df[df["exit_reason"].isin(["TakeProfit","StopLoss","HoldTimeout"])].copy()
    df["abs_signal"] = df["signal_strength"].abs()
    df["pnl_net"] = df["pnl_ticks"].astype(float) - COMMISSION_TICKS
    df["fill_latency_s"] = df["fill_latency_ns"].astype(float)/1e9
    df["hold_duration_s"] = df["hold_duration_ns"].astype(float)/1e9
    df["side_lc"] = df["side"].str.upper().map({"BUY":"long","B":"long","LONG":"long","L":"long",
                                                "SELL":"short","S":"short","SHORT":"short"})
    df["config_family"] = df["source_file"].str.extract(r"(qpos_top\d+_tp\d_sl\d)")[0]
    df["queue_position_at_post"] = df["queue_position_at_post"].astype(float)
    df["date_str"] = df["date"].astype(str)
    return df


def filter_apply(df, p):
    f = df[(df["abs_signal"] >= p["conf_min"]) & (df["abs_signal"] <= p["conf_max"])]
    f = f[f["queue_position_at_post"] <= p["queue_max"]]
    f = f[f["fill_latency_s"] <= p["fill_max_s"]]
    f = f[(f["hold_duration_s"] >= p["hold_min_s"]) & (f["hold_duration_s"] <= p["hold_max_s"])]
    if p["side"] != "both": f = f[f["side_lc"] == p["side"]]
    if p["config_subset"] != "all":
        f = f[f["config_family"].str.contains(p["config_subset"], na=False)]
    return f


def sortino(returns):
    if len(returns) < 5: return 0.0
    mean = returns.mean()
    dn = returns[returns < 0]
    if len(dn) == 0: return mean*10
    dd = np.sqrt((dn**2).mean())
    return float(mean/dd*np.sqrt(252)) if dd > 0 else 0.0


def stats(f):
    if len(f) < 5:
        return {"n":len(f), "wr":np.nan, "avg_net":np.nan, "total_net":np.nan,
                "pf":np.nan, "sortino":np.nan}
    pnl = f["pnl_net"].to_numpy()
    pnlg = f["pnl_ticks"].to_numpy()
    win_sum = pnlg[pnlg>0].sum(); loss_sum = -pnlg[pnlg<0].sum()
    return {
        "n": len(f),
        "wr": float((pnlg>0).mean()),
        "avg_net": float(pnl.mean()),
        "total_net": float(pnl.sum()),
        "pf": float(win_sum/loss_sum) if loss_sum>0 else 99.0,
        "sortino": sortino(pnl),
    }


def objective(trial, df):
    p = {
        "conf_min": trial.suggest_float("conf_min", 0.50, 2.50),
        "conf_max": trial.suggest_float("conf_max", 0.60, 3.50),
        "queue_max": trial.suggest_int("queue_max", 0, 50),
        "fill_max_s": trial.suggest_float("fill_max_s", 0.05, 10.0, log=True),
        "hold_min_s": trial.suggest_float("hold_min_s", 0.0, 30.0),
        "hold_max_s": trial.suggest_float("hold_max_s", 5.0, 120.0),
        "side": trial.suggest_categorical("side", ["long","short","both"]),
        "config_subset": trial.suggest_categorical("config_subset", ["tp3_sl3","tp4_sl4","all"]),
        "min_n": trial.suggest_int("min_n", 30, 300),
    }
    if p["conf_min"] >= p["conf_max"]-0.01: return -10
    if p["hold_min_s"] >= p["hold_max_s"]-1: return -10
    f = filter_apply(df, p)
    if len(f) < p["min_n"]: return -10
    return sortino(f["pnl_net"].to_numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-trials", type=int, default=1500)
    ap.add_argument("--train-frac", type=float, default=0.70)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_trades(args.parquet)

    dates = sorted(df["date_str"].unique())
    split = int(len(dates)*args.train_frac)
    train_dates = dates[:split]
    oos_dates = dates[split:]
    train_df = df[df["date_str"].isin(train_dates)]
    oos_df = df[df["date_str"].isin(oos_dates)]
    log.info(f"Train: {len(train_dates)} dates, {len(train_df):,} trades")
    log.info(f"OOS:   {len(oos_dates)} dates, {len(oos_df):,} trades")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42, multivariate=True))
    study.optimize(lambda t: objective(t, train_df), n_trials=args.n_trials, show_progress_bar=False)

    bp = study.best_params
    log.info(f"\nBest TRAIN params: {bp}")
    log.info(f"Best TRAIN Sortino: {study.best_value:.4f}")

    train_stats = stats(filter_apply(train_df, bp))
    oos_stats = stats(filter_apply(oos_df, bp))
    log.info("\n" + "="*70)
    log.info("TRAIN vs OOS COMPARISON")
    log.info("="*70)
    for k in ["n","wr","avg_net","total_net","pf","sortino"]:
        tv = train_stats[k]; ov = oos_stats[k]
        log.info(f"  {k:12s}  train={tv}  oos={ov}")

    # Robust check: also look at top-10 trial configs averaged
    top10 = sorted([t for t in study.trials if t.value is not None and t.value > 0],
                    key=lambda t: t.value, reverse=True)[:10]
    log.info(f"\nTop-10 configs OOS performance:")
    log.info(f"{'rank':<5} {'train_sortino':<15} {'oos_sortino':<15} {'oos_wr':<8} {'oos_n':<6} {'oos_total':<10}")
    rows = []
    for i, t in enumerate(top10):
        os_stats = stats(filter_apply(oos_df, t.params))
        rows.append({"rank": i+1, "train_sortino": t.value,
                    "oos_sortino": os_stats["sortino"],
                    "oos_wr": os_stats["wr"],
                    "oos_n": os_stats["n"],
                    "oos_total_net": os_stats["total_net"],
                    "oos_pf": os_stats["pf"],
                    **t.params})
        log.info(f"{i+1:<5} {t.value:<15.4f} {os_stats['sortino']:<15.4f} "
                f"{os_stats['wr']:<8.3f} {os_stats['n']:<6} {os_stats['total_net']:<10.2f}")
    pd.DataFrame(rows).to_csv(out/"train_vs_oos_top10.csv", index=False)
    log.info(f"\nSaved -> {out}/train_vs_oos_top10.csv")


if __name__ == "__main__":
    main()
