"""
run_ga.py — pyGAD driver for the macro-exposure GA.

Usage:
  python3 ga/run_ga.py --smoke              # pop=10 gen=5
  python3 ga/run_ga.py --pop 80 --gen 40    # full
"""
from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import pygad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ga.chromosome import GENE_NAMES, GENE_BOUNDS, N_GENES, decode, FEATURE_COLUMN_MAP
from ga.fitness import compute_fitness
from backtest.exposure_engine import run_exposure_backtest

CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------------- features

def _zscore(s: pd.Series, window: int = 252) -> pd.Series:
    m = s.rolling(window, min_periods=60).mean()
    sd = s.rolling(window, min_periods=60).std()
    return (s - m) / sd.replace(0, np.nan)


def build_feature_panel(smoke: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (features_df indexed by date, prices_wide_df indexed by date).
    features_df has the columns referenced by FEATURE_COLUMN_MAP, all standardized
    so cross-feature weights are sane.
    """
    sfx = "_smoke" if smoke else ""
    macro = pd.read_parquet(CACHE / f"macro_features{sfx}.parquet")
    sent = pd.read_parquet(CACHE / f"sentiment{sfx}.parquet")
    breadth = pd.read_parquet(CACHE / f"breadth{sfx}.parquet")
    market = pd.read_parquet(CACHE / f"market{sfx}.parquet")

    macro["date"] = pd.to_datetime(macro["date"])
    sent["date"] = pd.to_datetime(sent["date"])
    breadth["date"] = pd.to_datetime(breadth["date"])
    market["date"] = pd.to_datetime(market["date"])

    f = macro.set_index("date").sort_index()
    sent = sent.set_index("date").sort_index()
    breadth = breadth.set_index("date").sort_index()

    # Forward fill weekly / sparse signals on a daily spine
    f = f.join(sent[["naaim", "aaii_bullbear"]], how="left")
    f["naaim"] = f["naaim"].ffill()
    f["aaii_bullbear"] = f["aaii_bullbear"].ffill()
    f = f.join(breadth[["pct_above_200dma"]], how="left")
    f["pct_above_200dma"] = f["pct_above_200dma"].ffill()

    # Derived / standardized features
    f["naaim_z"] = _zscore(f["naaim"], 252) if f["naaim"].notna().any() else 0.0
    f["aaii_bullbear_z"] = (_zscore(f["aaii_bullbear"], 252)
                            if f["aaii_bullbear"].notna().any() else 0.0)
    # VIX percentile -> centered around 0 (negative when calm, positive when stressed)
    f["vix_pct_20d_centered"] = (f["vix_pct_20d"] - 0.5) * 2.0
    # VIX TS: > 1 = contango (calm), < 1 = backwardation (stress)
    f["vix_ts_slope_centered"] = (f["vix_ts_slope"].fillna(1.0) - 1.0)
    f["spy_above_200dma_sign"] = np.sign(f["spy_dma200_dist"].fillna(0))
    f["breadth_centered"] = (f["pct_above_200dma"].fillna(0.5) - 0.5) * 2.0
    f["gold_copper_z"] = _zscore(f["gold_copper_ratio"], 252)

    # Final feature dataframe
    feature_cols = list(FEATURE_COLUMN_MAP.values())
    for c in feature_cols:
        if c not in f.columns:
            f[c] = 0.0
    features = f[feature_cols].fillna(0.0)

    # Prices wide (SPY/QQQ/IWM) for the backtest engine
    px = market.pivot_table(index="date", columns="ticker", values="close", aggfunc="last").sort_index()
    px = px.ffill()
    # Restrict feature index to dates where SPY exists
    common = features.index.intersection(px.index)
    features = features.loc[common]
    px = px.loc[common]
    return features, px


def cadence_dates(idx: pd.DatetimeIndex, cadence: str) -> pd.DatetimeIndex:
    if cadence == "weekly":
        return idx[idx.weekday == 0]  # Mondays
    if cadence == "biweekly":
        mondays = idx[idx.weekday == 0]
        return mondays[::2]
    if cadence == "monthly":
        return idx[(idx.is_month_start) | (idx.to_series().shift(1).dt.month != idx.month)]
    return idx


def build_allocation(features: pd.DataFrame, cfg: dict) -> pd.Series:
    score = pd.Series(0.0, index=features.index)
    for gene, col in FEATURE_COLUMN_MAP.items():
        w = cfg["feature_weights"].get(gene, 0.0)
        if w == 0.0:
            continue
        score = score + w * features[col]

    long_th = cfg["long_threshold"]
    short_th = cfg["short_threshold"]
    half_band = max(cfg["flat_band_width"], 0.0) / 2.0
    eff_long = long_th + half_band
    eff_short = short_th - half_band

    alloc = pd.Series(0.0, index=features.index)
    alloc[score >= eff_long] = cfg["long_strength"]
    if cfg["allow_short"]:
        alloc[score <= eff_short] = -cfg["short_strength"]
    return alloc.clip(lower=-cfg["max_leverage"], upper=cfg["max_leverage"])


# ----------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--gen", type=int, default=40)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--starting-cash", type=float, default=100_000.0)
    ap.add_argument("--out-tag", default="full")
    args = ap.parse_args()
    if args.smoke:
        args.pop, args.gen, args.out_tag = 10, 5, "smoke"

    print(f"[ga] loading panel smoke={args.smoke}", flush=True)
    features, prices = build_feature_panel(args.smoke)
    print(f"[ga] features {features.shape}  prices {prices.shape}  "
          f"span {features.index.min()} -> {features.index.max()}", flush=True)

    rows = []
    eval_n = {"n": 0}

    def fitness_fn(ga_instance, solution, idx):
        eval_n["n"] += 1
        cfg = decode(solution)
        try:
            alloc = build_allocation(features, cfg)
            rebal = cadence_dates(features.index, cfg["cadence"])
            res = run_exposure_backtest(
                allocation=alloc,
                basket_w=cfg["basket_weights"],
                prices=prices,
                rebalance_dates=rebal,
                starting_cash=args.starting_cash,
                allow_short=cfg["allow_short"],
                max_leverage=cfg["max_leverage"],
            )
            fit = compute_fitness(res.metrics)
        except Exception as e:
            print(f"[ga] eval err: {e}", flush=True)
            return -1.0

        row = dict(
            fitness=fit,
            cagr=res.metrics["cagr"],
            sortino=res.metrics["sortino"],
            sharpe=res.metrics["sharpe"],
            max_dd_pct=res.metrics["max_dd_pct"],
            worst_month_pct=res.metrics["worst_month_pct"],
            turnover_per_year=res.metrics["turnover_per_year"],
            avg_leverage=res.metrics["avg_leverage"],
            pct_long=res.metrics["pct_long"],
            pct_flat=res.metrics["pct_flat"],
            pct_short=res.metrics["pct_short"],
            cadence=cfg["cadence"],
            allow_short=int(cfg["allow_short"]),
            max_leverage=cfg["max_leverage"],
            long_threshold=cfg["long_threshold"],
            short_threshold=cfg["short_threshold"],
            flat_band_width=cfg["flat_band_width"],
            long_strength=cfg["long_strength"],
            short_strength=cfg["short_strength"],
            bw_spy=cfg["basket_weights"]["SPY"],
            bw_qqq=cfg["basket_weights"]["QQQ"],
            bw_iwm=cfg["basket_weights"]["IWM"],
        )
        row.update(cfg["feature_weights"])
        rows.append(row)
        if eval_n["n"] % 10 == 0:
            print(f"[ga] eval {eval_n['n']}  fit={fit:.4f}  cagr={res.metrics['cagr']*100:.1f}%  "
                  f"dd={res.metrics['max_dd_pct']:.1f}%  to/yr={res.metrics['turnover_per_year']:.1f}",
                  flush=True)
        return fit

    gene_space = [{"low": lo, "high": hi} for lo, hi in GENE_BOUNDS]
    ga = pygad.GA(
        num_generations=args.gen,
        sol_per_pop=args.pop,
        num_parents_mating=max(args.pop // 4, 2),
        num_genes=N_GENES,
        gene_space=gene_space,
        gene_type=float,
        parent_selection_type="tournament",
        K_tournament=3,
        crossover_type="uniform",
        mutation_type="random",
        mutation_percent_genes=15,
        random_seed=args.seed,
        fitness_func=fitness_fn,
        suppress_warnings=True,
    )

    t0 = time.time()
    ga.run()
    print(f"[ga] done {eval_n['n']} evals in {time.time()-t0:.1f}s", flush=True)

    df = pd.DataFrame(rows)
    out = RESULTS / args.out_tag
    out.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out / "all_evals.parquet", index=False)

    # Pareto front on (CAGR maximise, max_DD minimise, turnover minimise)
    pareto_mask = []
    arr = df[["cagr", "max_dd_pct", "turnover_per_year"]].to_numpy()
    for i in range(len(arr)):
        dom = False
        for j in range(len(arr)):
            if i == j:
                continue
            if (arr[j, 0] >= arr[i, 0] and arr[j, 1] <= arr[i, 1] and arr[j, 2] <= arr[i, 2]
                    and (arr[j, 0] > arr[i, 0] or arr[j, 1] < arr[i, 1] or arr[j, 2] < arr[i, 2])):
                dom = True
                break
        pareto_mask.append(not dom)
    pareto = df[pareto_mask].copy().sort_values("fitness", ascending=False)
    pareto.to_parquet(out / "pareto.parquet", index=False)

    best = df.sort_values("fitness", ascending=False).head(20)
    best.to_parquet(out / "top20.parquet", index=False)

    print(f"[ga] wrote {len(df)} evals, {len(pareto)} pareto -> {out}", flush=True)
    if len(best):
        b = best.iloc[0]
        print(f"[ga] best fit={b['fitness']:.4f} cagr={b['cagr']*100:.1f}% "
              f"dd={b['max_dd_pct']:.1f}% sortino={b['sortino']:.2f} "
              f"to/yr={b['turnover_per_year']:.1f}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
