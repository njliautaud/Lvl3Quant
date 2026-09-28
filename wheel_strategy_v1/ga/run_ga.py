"""
run_ga.py — Genetic algorithm driver (pyGAD).

Usage:
  python3 ga/run_ga.py --smoke
  python3 ga/run_ga.py --pop 100 --gen 50
"""
from __future__ import annotations
import sys
import time
import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import pygad

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from ga.chromosome import GENE_BOUNDS, GENE_NAMES, N_GENES, decode
from ga.fitness import compute_metrics
from backtest.wheel_engine import WheelConfig, run_wheel

CACHE = ROOT / "data" / "cache"
RESULTS = ROOT / "results"
RESULTS.mkdir(parents=True, exist_ok=True)


def _load(smoke: bool):
    sfx = "_smoke" if smoke else ""
    prices = pd.read_parquet(CACHE / f"prices{sfx}.parquet")
    iv = pd.read_parquet(CACHE / f"iv_features{sfx}.parquet")
    macro = pd.read_parquet(CACHE / f"macro{sfx}.parquet")
    fundamentals = pd.read_parquet(CACHE / f"fundamentals{sfx}.parquet")
    universe = pd.read_parquet(CACHE / "universe.parquet")
    return prices, iv, macro, fundamentals, universe


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--pop", type=int, default=100)
    ap.add_argument("--gen", type=int, default=50)
    ap.add_argument("--starting-cash", type=float, default=100_000.0)
    ap.add_argument("--start", default=None)
    ap.add_argument("--end", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-tag", default="full")
    args = ap.parse_args()

    if args.smoke:
        args.pop = 10
        args.gen = 5
        args.out_tag = "smoke"

    print(f"[ga] loading data smoke={args.smoke}", flush=True)
    prices, iv, macro, fundamentals, universe = _load(args.smoke)
    print(f"[ga] prices={len(prices)} iv={len(iv)} macro={len(macro)} "
          f"fund={len(fundamentals)} uni={len(universe)}", flush=True)

    pareto_rows = []   # store every evaluated config for later Pareto extraction
    eval_count = {"n": 0}

    def fitness_func(ga_instance, solution, solution_idx):
        eval_count["n"] += 1
        cfg = decode(solution)
        wc = WheelConfig(**cfg)
        try:
            result = run_wheel(wc, prices, iv, macro, fundamentals, universe,
                               starting_cash=args.starting_cash,
                               start=args.start, end=args.end)
            m = compute_metrics(result)
        except Exception as e:
            print(f"[ga] eval err: {e}", flush=True)
            return -1.0
        row = {**cfg, **m}
        pareto_rows.append(row)
        if eval_count["n"] % 5 == 0:
            print(f"[ga] eval {eval_count['n']} fit={m['fitness']:.4f} "
                  f"yield={m['ann_premium_yield']:.3f} dd={m['max_dd_pct']:.1f}% "
                  f"sortino={m['sortino']:.2f} trades={m['n_trades']}", flush=True)
        return m["fitness"]

    gene_space = [{"low": lo, "high": hi} for (lo, hi) in GENE_BOUNDS]

    ga = pygad.GA(
        num_generations=args.gen,
        num_parents_mating=max(2, args.pop // 4),
        fitness_func=fitness_func,
        sol_per_pop=args.pop,
        num_genes=N_GENES,
        gene_space=gene_space,
        parent_selection_type="tournament",
        K_tournament=3,
        crossover_type="single_point",
        mutation_type="random",
        mutation_percent_genes=20,
        random_seed=args.seed,
        keep_elitism=2,
        suppress_warnings=True,
    )

    t0 = time.time()
    ga.run()
    dt = time.time() - t0
    print(f"[ga] done in {dt/60:.1f}min, {eval_count['n']} evals", flush=True)

    df = pd.DataFrame(pareto_rows)
    if df.empty:
        print("[ga] no rows recorded", file=sys.stderr); sys.exit(3)

    # Pareto front on (yield UP, dd DOWN, assignment_rate DOWN)
    def is_dominated(i, df):
        a = df.iloc[i]
        for j in range(len(df)):
            if j == i:
                continue
            b = df.iloc[j]
            if (b["ann_premium_yield"] >= a["ann_premium_yield"] and
                b["max_dd_pct"] <= a["max_dd_pct"] and
                b["assignment_rate"] <= a["assignment_rate"] and
                (b["ann_premium_yield"] > a["ann_premium_yield"] or
                 b["max_dd_pct"] < a["max_dd_pct"] or
                 b["assignment_rate"] < a["assignment_rate"])):
                return True
        return False

    # only consider rows with non-negative fitness for Pareto
    feas = df[df["n_trades"] > 0].reset_index(drop=True)
    if feas.empty:
        feas = df.copy().reset_index(drop=True)
    mask = [not is_dominated(i, feas) for i in range(len(feas))]
    pareto = feas[mask].reset_index(drop=True)
    pareto["is_pareto"] = True
    df["is_pareto"] = False

    out_all = RESULTS / f"all_evals_{args.out_tag}.parquet"
    out_par = RESULTS / f"pareto_{args.out_tag}.parquet"
    df.to_parquet(out_all, index=False)
    pareto.to_parquet(out_par, index=False)
    print(f"[ga] wrote {len(df)} evals -> {out_all}")
    print(f"[ga] wrote {len(pareto)} Pareto configs -> {out_par}")

    best = df.sort_values("fitness", ascending=False).iloc[0].to_dict()
    summary = {
        "tag": args.out_tag,
        "pop": args.pop,
        "gen": args.gen,
        "n_evals": eval_count["n"],
        "elapsed_min": dt / 60,
        "best_fitness": float(best["fitness"]),
        "best_yield": float(best["ann_premium_yield"]),
        "best_dd_pct": float(best["max_dd_pct"]),
        "best_sortino": float(best["sortino"]),
        "best_trades": int(best["n_trades"]),
    }
    with open(RESULTS / f"summary_{args.out_tag}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
