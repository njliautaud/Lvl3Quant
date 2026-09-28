#!/usr/bin/env python3
"""
High-Dimensional Optuna Sweep over cached fill-sim trades.
Per HC #251: static 4-axis grids are too narrow. This searches 9+ axes with TPE
sampler over 1500 trials. Optimizes Sortino on net P&L (gross - 0.376 commission).

Search axes (continuous + categorical):
  1. confidence_min (abs_signal lower bound)        [continuous, 0.50–2.50]
  2. confidence_max (abs_signal upper bound)        [continuous, 0.60–3.50]
  3. queue_max (queue_position_at_post upper bound) [int, 0–50]
  4. fill_max_s (fill_latency_s upper bound)        [continuous, 0.05–10.0]
  5. hold_min_s / hold_max_s (hold_duration window) [continuous, 0–60]
  6. side filter (long / short / both)              [categorical]
  7. config_subset (which TP/SL config family)      [categorical: tp3_sl3, tp4_sl4, all]
  8. tier (top1/5/10/20/50/all)                     [categorical]
  9. min_n_trades (reject sparse configs)           [int, 50–500]

Outputs:
  - top_trials.csv  (best 100 by Sortino)
  - study.pkl       (full Optuna study)
  - param_importance.csv

Usage:
  python3 optuna_high_dim_sweep.py \\
    --parquet output/fill_wait_mfe_v1/all_trades.parquet \\
    --output-dir output/optuna_v1 --n-trials 1500
"""
from __future__ import annotations
import argparse, logging, pickle
from pathlib import Path
import numpy as np, pandas as pd, optuna

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [OPTUNA] %(levelname)s: %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

COMMISSION_TICKS = 0.376  # ES_RT_COMMISSION


def load_trades(parquet_path: str) -> pd.DataFrame:
    df = pd.read_parquet(parquet_path)
    df = df[df["exit_reason"].isin(["TakeProfit", "StopLoss", "HoldTimeout"])].copy()
    df["abs_signal"] = df["signal_strength"].abs()
    df["pnl_net"] = df["pnl_ticks"].astype(float) - COMMISSION_TICKS
    df["fill_latency_s"] = df["fill_latency_ns"].astype(float) / 1e9
    df["hold_duration_s"] = df["hold_duration_ns"].astype(float) / 1e9
    df["side_lc"] = df["side"].str.upper().map({"BUY": "long", "B": "long",
                                                "LONG": "long", "L": "long",
                                                "SELL": "short", "S": "short",
                                                "SHORT": "short"})
    # extract config family from source_file (qpos_top10_tp3_sl3_20260306.json)
    df["config_family"] = df["source_file"].str.extract(r"(qpos_top\d+_tp\d_sl\d)")[0]
    df["queue_position_at_post"] = df["queue_position_at_post"].astype(float)
    return df


def filter_apply(df: pd.DataFrame, params: dict) -> pd.DataFrame:
    f = df
    f = f[(f["abs_signal"] >= params["conf_min"]) & (f["abs_signal"] <= params["conf_max"])]
    f = f[f["queue_position_at_post"] <= params["queue_max"]]
    f = f[f["fill_latency_s"] <= params["fill_max_s"]]
    f = f[(f["hold_duration_s"] >= params["hold_min_s"]) &
          (f["hold_duration_s"] <= params["hold_max_s"])]
    if params["side"] != "both":
        f = f[f["side_lc"] == params["side"]]
    if params["config_subset"] != "all":
        f = f[f["config_family"].str.contains(params["config_subset"], na=False)]
    return f


def sortino(returns: np.ndarray) -> float:
    if len(returns) < 5: return 0.0
    mean = returns.mean()
    downside = returns[returns < 0]
    if len(downside) == 0: return mean * 10
    dd = np.sqrt((downside ** 2).mean())
    if dd == 0: return 0.0
    return float(mean / dd * np.sqrt(252))


def objective(trial, df: pd.DataFrame) -> float:
    params = {
        "conf_min": trial.suggest_float("conf_min", 0.50, 2.50),
        "conf_max": trial.suggest_float("conf_max", 0.60, 3.50),
        "queue_max": trial.suggest_int("queue_max", 0, 50),
        "fill_max_s": trial.suggest_float("fill_max_s", 0.05, 10.0, log=True),
        "hold_min_s": trial.suggest_float("hold_min_s", 0.0, 30.0),
        "hold_max_s": trial.suggest_float("hold_max_s", 5.0, 120.0),
        "side": trial.suggest_categorical("side", ["long", "short", "both"]),
        "config_subset": trial.suggest_categorical("config_subset", ["tp3_sl3", "tp4_sl4", "all"]),
        "min_n": trial.suggest_int("min_n", 50, 500),
    }
    if params["conf_min"] >= params["conf_max"] - 0.01: return -10.0
    if params["hold_min_s"] >= params["hold_max_s"] - 1: return -10.0

    f = filter_apply(df, params)
    if len(f) < params["min_n"]: return -10.0

    pnl = f["pnl_net"].to_numpy()
    s = sortino(pnl)
    trial.set_user_attr("n_trades", len(f))
    trial.set_user_attr("avg_net_ticks", float(pnl.mean()))
    trial.set_user_attr("total_net_ticks", float(pnl.sum()))
    trial.set_user_attr("wr", float((f["pnl_ticks"] > 0).mean()))
    trial.set_user_attr("pf", float(f.loc[f["pnl_ticks"] > 0, "pnl_ticks"].sum() /
                                     -f.loc[f["pnl_ticks"] < 0, "pnl_ticks"].sum())
                        if (f["pnl_ticks"] < 0).any() else 99.0)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--parquet", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-trials", type=int, default=1500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    out = Path(args.output_dir); out.mkdir(parents=True, exist_ok=True)
    df = load_trades(args.parquet)
    log.info(f"Loaded {len(df):,} trades for sweep")
    log.info(f"Side dist: {df['side_lc'].value_counts().to_dict()}")
    log.info(f"Config family dist: {df['config_family'].value_counts().to_dict()}")

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=args.seed, multivariate=True),
        pruner=optuna.pruners.NopPruner(),
    )
    study.optimize(lambda t: objective(t, df),
                   n_trials=args.n_trials, show_progress_bar=False,
                   n_jobs=1)

    log.info(f"\nBEST: Sortino={study.best_value:.4f}")
    log.info(f"  params: {study.best_params}")
    log.info(f"  attrs:  {study.best_trial.user_attrs}")

    rows = []
    for t in study.trials:
        if t.value is None or t.value < -5: continue
        r = {"trial": t.number, "sortino": t.value, **t.params, **t.user_attrs}
        rows.append(r)
    top_df = pd.DataFrame(rows).sort_values("sortino", ascending=False)
    top_df.head(100).to_csv(out / "top_trials.csv", index=False)
    log.info(f"Saved top 100 trials -> {out}/top_trials.csv")

    try:
        imp = optuna.importance.get_param_importances(study)
        pd.DataFrame([{"param": k, "importance": v} for k, v in imp.items()]
                     ).sort_values("importance", ascending=False
                     ).to_csv(out / "param_importance.csv", index=False)
        log.info(f"Param importance: {imp}")
    except Exception as e:
        log.warning(f"Importance calc failed: {e}")

    with open(out / "study.pkl", "wb") as f:
        pickle.dump(study, f)
    log.info(f"DONE — {len(study.trials)} trials")


if __name__ == "__main__":
    main()
