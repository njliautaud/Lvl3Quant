"""
ga_formula_pit_sparse.py — Cardinality-constrained variant of ga_formula_pit.

Problem: With 44 features and ~14-40 tickers per sector, the unconstrained GA
overfits massively (Energy TRAIN +1.95 -> OOT -1.12 Sharpe; Healthcare TRAIN
+1.66 -> OOT -0.67). The rich PIT feature shelf gives the GA more rope.

Fix: at every chromosome, project to top-K abs(weight) features (default K=10).
This is a hard L0 constraint — equivalent to saying "the formula may use no
more than K features at a time". Forces parsimony, prevents in-sample noise
fitting, and the resulting formulas are also more interpretable.

Reuses ga_formula machinery; only the GA loop is overridden. The PIT load_data
+ DEFAULT_FEATURES_PIT come from ga_formula_pit.

Run:
    cd /home/jupiter/Lvl3Quant
    python3 -m wheel_strategy_v1.strategy.macro_picker.ga_formula_pit_sparse \
        --sector dead_sectors --pop 80 --generations 40 --k 10 \
        --out-tag v2_pit_k10
"""
from __future__ import annotations
import argparse
import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from wheel_strategy_v1.strategy.macro_picker.ga_formula import (
    HARD_CALMAR_FLOOR,
    build_feature_matrix,
    evaluate_formula,
    format_expression,
    precompute,
    scalar_fitness,
    OUT_DIR,
    OOT_START,
)
from wheel_strategy_v1.strategy.macro_picker.ga_formula_pit import (
    DEFAULT_FEATURES_PIT,
    DEAD_SECTORS,
    load_data_pit,
)


def _project_top_k(w: np.ndarray, k: int) -> np.ndarray:
    """Zero all but the top-k by absolute magnitude."""
    if k >= len(w):
        return w
    idx = np.argpartition(-np.abs(w), k)[:k]
    out = np.zeros_like(w)
    out[idx] = w[idx]
    return out


def ga_search_sparse(
    Xz: np.ndarray,
    df: pd.DataFrame,
    features: List[str],
    pop: int = 80,
    gens: int = 40,
    seed: int = 13,
    elite: int = 8,
    sigma: float = 0.3,
    k: int = 10,
) -> Tuple[np.ndarray, Dict[str, float], List[float]]:
    rng = np.random.default_rng(seed)
    F = Xz.shape[1]
    pc = precompute(df)

    # Init random sparse chromosomes
    population = rng.normal(0, 0.5, size=(pop, F))
    population = np.array([_project_top_k(p, k) for p in population])

    best_score = -np.inf
    best_w = None
    best_metrics = None
    history = []

    for gen in range(gens):
        scores = np.zeros(pop)
        metrics_list = []
        for i in range(pop):
            m = evaluate_formula(population[i], Xz, df, pc=pc)
            metrics_list.append(m)
            scores[i] = scalar_fitness(m)

        order = np.argsort(scores)[::-1]
        scores = scores[order]
        population = population[order]
        metrics_list = [metrics_list[i] for i in order]

        if scores[0] > best_score:
            best_score = scores[0]
            best_w = population[0].copy()
            best_metrics = metrics_list[0]

        history.append({"gen": gen, "best": float(scores[0]),
                        "median": float(np.median(scores)),
                        "calmar": metrics_list[0]["calmar"],
                        "sharpe": metrics_list[0]["sharpe"],
                        "nonzero": int((np.abs(best_w) > 1e-9).sum())})

        # Breed
        new_pop = list(population[:elite])
        while len(new_pop) < pop:
            i1, i2 = rng.integers(0, pop // 2, size=2)
            p1, p2 = population[i1], population[i2]
            mask_x = rng.random(F) < 0.5
            child = np.where(mask_x, p1, p2)
            child = child + rng.normal(0, sigma, size=F) * (rng.random(F) < 0.3)
            # Project to top-k AFTER mutation (the hard L0 constraint)
            child = _project_top_k(child, k)
            new_pop.append(child)
        population = np.array(new_pop)

        sigma = max(0.05, sigma * 0.97)

        if gen % 5 == 0 or gen == gens - 1:
            nz = int((np.abs(best_w) > 1e-9).sum())
            print(f"  gen {gen:3d}: best={scores[0]:+.3f} calmar={metrics_list[0]['calmar']:+.2f} "
                  f"sharpe={metrics_list[0]['sharpe']:+.2f} dd={metrics_list[0]['maxdd']*100:+.1f}% "
                  f"nz={nz}", flush=True)

    return best_w, best_metrics, history


def run_sector_sparse(sector: str, pop: int, gens: int, seed: int,
                       k: int,
                       oot_start: Optional[pd.Timestamp],
                       fstore_version: str,
                       universe_file: str,
                       prices_file: str,
                       out_tag: str) -> Dict:
    print(f"\n=== GA-PIT-SPARSE: sector={sector} k={k} pop={pop} gens={gens} ===", flush=True)
    panel, _, _ = load_data_pit(sector, fstore_version=fstore_version,
                                 universe_file=universe_file, prices_file=prices_file)
    print(f"  panel rows={len(panel)} tickers={panel['ticker'].nunique()} dates={panel['date'].nunique()}", flush=True)

    if oot_start is not None:
        train_panel = panel[panel["date"] < oot_start].copy()
        oot_panel = panel[panel["date"] >= oot_start].copy()
    else:
        train_panel, oot_panel = panel, pd.DataFrame()

    Xz_train, _, df_train, used_features = build_feature_matrix(train_panel, DEFAULT_FEATURES_PIT)
    print(f"  train shape={Xz_train.shape}  features={len(used_features)}  k={k}", flush=True)

    t0 = time.time()
    best_w, best_m, hist = ga_search_sparse(Xz_train, df_train, used_features,
                                              pop=pop, gens=gens, seed=seed, k=k)
    dt = time.time() - t0
    print(f"  GA done in {dt:.1f}s", flush=True)

    expr = format_expression(best_w, used_features)
    nz_train = int((np.abs(best_w) > 1e-9).sum())
    print(f"  formula ({nz_train} non-zero terms): {expr}", flush=True)
    print(f"  TRAIN: sharpe={best_m['sharpe']:+.2f} calmar={best_m['calmar']:+.2f} "
          f"cagr={best_m['cagr']*100:+.1f}% maxdd={best_m['maxdd']*100:+.1f}%", flush=True)

    oot_metrics = None
    if not oot_panel.empty:
        Xz_oot, _, df_oot, oot_feats = build_feature_matrix(oot_panel, DEFAULT_FEATURES_PIT)
        if oot_feats == used_features:
            oot_metrics = evaluate_formula(best_w, Xz_oot, df_oot)
            print(f"  OOT  : sharpe={oot_metrics['sharpe']:+.2f} calmar={oot_metrics['calmar']:+.2f} "
                  f"cagr={oot_metrics['cagr']*100:+.1f}% maxdd={oot_metrics['maxdd']*100:+.1f}%", flush=True)

    artifact = {
        "sector": sector,
        "k": k,
        "features": used_features,
        "weights": best_w.tolist(),
        "expression": expr,
        "fitness_train": best_m,
        "fitness_oot": oot_metrics,
        "n_features_nonzero": nz_train,
        "trained_on": {"start": str(train_panel["date"].min().date()),
                        "end": str(train_panel["date"].max().date()),
                        "n_tickers": int(train_panel["ticker"].nunique()),
                        "n_days": int(train_panel["date"].nunique())},
        "oot_window": ({"start": str(oot_panel["date"].min().date()),
                         "end": str(oot_panel["date"].max().date()),
                         "n_days": int(oot_panel["date"].nunique())} if not oot_panel.empty else None),
        "deployable": bool(oot_metrics and oot_metrics["calmar"] >= HARD_CALMAR_FLOOR
                            and oot_metrics["regime_drift"] <= 0.5
                            and oot_metrics["sharpe"] > 0.7),
        "feature_shelf": f"v2_pit_k{k}",
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    out_path = OUT_DIR / f"formula_{out_tag}_{sector.replace(' ', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"  wrote {out_path}", flush=True)
    return artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", default="dead_sectors")
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--k", type=int, default=10,
                    help="Max non-zero features per formula (cardinality cap).")
    ap.add_argument("--fstore-version", default="v2")
    ap.add_argument("--universe-file", default="universe_v2.parquet")
    ap.add_argument("--prices-file", default="prices_v2.parquet")
    ap.add_argument("--out-tag", default=None,
                    help="Output tag — defaults to v2_pit_k<K>")
    ap.add_argument("--oot-start", default=str(OOT_START.date()))
    args = ap.parse_args()

    if args.out_tag is None:
        args.out_tag = f"v2_pit_k{args.k}"
    oot_start = None if args.oot_start.lower() == "none" else pd.Timestamp(args.oot_start)

    if args.sector == "dead_sectors":
        sectors = DEAD_SECTORS
    elif args.sector == "all":
        u = pd.read_parquet("/home/jupiter/Lvl3Quant/wheel_strategy_v1/data/cache/" + args.universe_file)
        sectors = [s for s in u["sector"].unique() if s != "ETF"]
    else:
        sectors = [args.sector]

    results = []
    for s in sectors:
        try:
            art = run_sector_sparse(s, pop=args.pop, gens=args.generations, seed=args.seed,
                                      k=args.k, oot_start=oot_start,
                                      fstore_version=args.fstore_version,
                                      universe_file=args.universe_file, prices_file=args.prices_file,
                                      out_tag=args.out_tag)
            results.append((s, art))
        except Exception as e:
            print(f"  ERR {s}: {e}", flush=True)

    print(f"\n=== SUMMARY (PIT + K={args.k} cardinality cap) ===")
    print(f"{'Sector':25s} {'OOT_Sharpe':>10s} {'OOT_Calmar':>10s} {'OOT_CAGR':>10s} {'NonZero':>8s} {'Deploy':>8s}")
    print("-" * 80)
    sum_path = OUT_DIR / f"summary_{args.out_tag}.json"
    payload = []
    for s, art in results:
        m = art.get("fitness_oot") or {}
        sh = m.get("sharpe", float("nan"))
        ca = m.get("calmar", float("nan"))
        cg = (m.get("cagr") or 0) * 100
        nz = art.get("n_features_nonzero", 0)
        dep = "YES" if art.get("deployable") else "no"
        print(f"{s:25s} {sh:>+10.2f} {ca:>+10.2f} {cg:>+9.1f}% {nz:>8d} {dep:>8s}")
        payload.append({"sector": s, "fitness_oot": m, "deployable": art.get("deployable"),
                         "n_features_nonzero": nz, "k": args.k})
    with open(sum_path, "w") as f:
        json.dump(payload, f, indent=2, default=str)
    print(f"\nWrote {sum_path}")


if __name__ == "__main__":
    main()
