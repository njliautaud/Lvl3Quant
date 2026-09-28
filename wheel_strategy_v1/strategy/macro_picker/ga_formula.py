"""
ga_formula.py — Per-sector genetic algorithm for interpretable ranking formulas.

Implements HC #559 R3 (closed-form weighted expression), R4 (multi-objective
fitness with Calmar≥1.0 hard floor), R5 (per-GICS-sector formula).

Output artifact (per HC #559 R3):
    strategy/macro_picker/formulas/formula_v1_<sector>.json
    {
        "sector": "Technology",
        "features": [...],
        "weights": [...],
        "expression": "score = 0.42*flow_sectorRet_r20 + 0.31*factor_momentum_load - 0.18*fund_evEbitda_z + ...",
        "fitness": {"sharpe": .., "calmar": .., "cagr": .., "maxdd": .., "spy_beat": ..},
        "trained_on": {"start": "...", "end": "...", "n_tickers": N, "n_days": M},
        "regime_breakdown": {"green_sharpe": .., "red_sharpe": ..},
    }

Fitness (HC #559 R4):
    Multi-objective scalar = w1*(Sharpe - 1.5*SPY_Sharpe) + w2*Calmar + w3*(CAGR - 1.5*SPY_CAGR)
                              - w4*turnover - w5*max_sector_concentration
    Hard floor: Calmar < 1.0 ⇒ fitness = -1e9 (rejected in selection).

GA: simple (μ,λ) ES with crossover + Gaussian mutation. No DEAP dependency.

Usage:
    cd /home/jupiter/Lvl3Quant
    python3 -m wheel_strategy_v1.strategy.macro_picker.ga_formula --sector Technology --generations 40 --pop 80
    python3 -m wheel_strategy_v1.strategy.macro_picker.ga_formula --sector all --generations 40 --pop 80
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import warnings
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/home/jupiter/Lvl3Quant")
WHEEL = ROOT / "wheel_strategy_v1"
FSTORE = ROOT / "data" / "feature_store" / "v1"
CACHE = WHEEL / "data" / "cache"
OUT_DIR = WHEEL / "strategy" / "macro_picker" / "formulas"
OUT_DIR.mkdir(parents=True, exist_ok=True)

START = pd.Timestamp("2020-01-01")
END = pd.Timestamp("2026-07-01")
# OOT split (HC #428 R1: ≥40 OOT days, all regimes, stratified per regime).
# Default: train on 2020-01 → 2025-04 (~5.3y), OOT 2025-04 → 2026-07 (~14 months trading days).
OOT_START = pd.Timestamp("2025-04-01")
TRADING_DAYS = 252
HARD_CALMAR_FLOOR = 1.0  # HC #559 R1

# Feature columns we let the GA pick weights over.
# Excluded: ticker, date, *_asof, regime_* (regime enters via fitness stratification, not formula).
DEFAULT_FEATURES = [
    # Flow
    "flow_sectorRet_r20", "flow_sectorRet_r60", "flow_sectorAUM_z20",
    "flow_sectorAUM_z60", "flow_sectorRel_r20", "flow_sectorRel_r60",
    # Factor
    "factor_momentum_load", "factor_quality_load", "factor_value_load",
    "factor_lowvol_load", "factor_momentum_rank", "factor_quality_rank",
    # Fundamentals
    "fund_debtEquity_z", "fund_earningsYield_z", "fund_evEbitda_z",
    "fund_fcfMargin_z", "fund_fcfYield_z", "fund_gross_margin_z",
    "fund_net_margin_z", "fund_revScale_log_z", "fund_roicProxy_z",
]


def load_data(sector: str,
              fstore_version: str = "v1",
              universe_file: str = "universe.parquet",
              prices_file: str = "prices.parquet") -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load joined feature panel + forward returns + regime."""
    universe = pd.read_parquet(CACHE / universe_file)
    if sector != "all":
        tickers = universe[universe["sector"] == sector]["ticker"].tolist()
    else:
        tickers = universe["ticker"].tolist()
    if not tickers:
        raise ValueError(f"No tickers in sector {sector!r}. Known: {universe['sector'].unique().tolist()}")

    fstore_dir = ROOT / "data" / "feature_store" / fstore_version
    fund = pd.read_parquet(fstore_dir / "fund_features.parquet")
    flow = pd.read_parquet(fstore_dir / "flow_features.parquet")
    factor = pd.read_parquet(fstore_dir / "factor_features.parquet")
    regime = pd.read_parquet(fstore_dir / "regime_features.parquet")

    panel = (
        fund.merge(flow, on=["ticker", "date"], how="outer")
            .merge(factor, on=["ticker", "date"], how="outer")
    )
    panel = panel[panel["ticker"].isin(tickers)].copy()
    panel = panel[(panel["date"] >= START) & (panel["date"] < END)].copy()
    panel = panel.merge(regime, on="date", how="left")

    prices = pd.read_parquet(CACHE / prices_file)
    prices = prices[prices["ticker"].isin(tickers)].copy()
    prices["fwd_ret_5d"] = prices.groupby("ticker")["close"].pct_change(5).shift(-5)
    prices["fwd_ret_1d"] = prices.groupby("ticker")["close"].pct_change(1).shift(-1)
    prices = prices[(prices["date"] >= START) & (prices["date"] < END)][
        ["ticker", "date", "close", "fwd_ret_1d", "fwd_ret_5d"]
    ]

    # Build SPY proxy for excess metrics
    if "SPY" in universe["ticker"].values:
        spy = prices[prices["ticker"] == "SPY"][["date", "close"]].copy()
    else:
        # Use universe equal-weight as proxy (good enough for fitness ranking)
        spy = prices.groupby("date")["close"].mean().reset_index()
    spy["spy_ret_1d"] = spy["close"].pct_change(1)
    spy = spy[["date", "spy_ret_1d"]]

    panel = panel.merge(prices[["ticker", "date", "fwd_ret_1d", "fwd_ret_5d"]],
                        on=["ticker", "date"], how="inner")
    panel = panel.merge(spy, on="date", how="left")
    panel = panel.sort_values(["date", "ticker"]).reset_index(drop=True)

    return panel, prices, spy


def build_feature_matrix(panel: pd.DataFrame, features: List[str]) -> Tuple[np.ndarray, np.ndarray, pd.DataFrame]:
    """Return (X[N×F], y[N], idx_df[date,ticker,fwd_ret_1d,spy_ret_1d])."""
    use = [c for c in features if c in panel.columns]
    if not use:
        raise ValueError("None of the requested features exist in panel.")
    X = panel[use].values.astype(np.float32)
    # Cross-section z-score by date to make weights comparable
    df = panel[["date", "ticker", "fwd_ret_1d", "spy_ret_1d"]].copy()
    df["__row"] = np.arange(len(df))
    # Fill NaN with 0 AFTER cross-sectional z (so missing → neutral)
    Xz = np.zeros_like(X)
    for c_i in range(X.shape[1]):
        col = pd.Series(X[:, c_i], index=panel["date"].values)
        mu = col.groupby(col.index).transform("mean")
        sd = col.groupby(col.index).transform("std").replace(0, np.nan)
        z = ((col - mu) / sd).fillna(0.0).values
        # Winsorize
        z = np.clip(z, -3.0, 3.0)
        Xz[:, c_i] = z
    y = panel["fwd_ret_1d"].fillna(0.0).values.astype(np.float32)
    return Xz, y, df, use


@dataclass
class Precomputed:
    """Per-day vectorized indices for fast evaluation."""
    date_starts: np.ndarray  # offsets into sorted rows
    date_lens: np.ndarray    # group sizes
    fwd: np.ndarray          # forward returns aligned with Xz rows
    spy_per_date: np.ndarray # spy daily return per unique date
    n_dates: int
    ticker_per_row: np.ndarray  # ticker id (int) per row
    n_tickers: int


def precompute(df: pd.DataFrame) -> Precomputed:
    """Caller has already sorted Xz/df by date,ticker."""
    df = df.reset_index(drop=True)
    dates = df["date"].values
    # Run-length groups (df is sorted by date)
    chg = np.concatenate(([True], dates[1:] != dates[:-1]))
    date_starts = np.flatnonzero(chg)
    date_lens = np.diff(np.append(date_starts, len(df)))
    spy_per_date = df.iloc[date_starts]["spy_ret_1d"].fillna(0).values.astype(np.float64)
    fwd = df["fwd_ret_1d"].fillna(0).values.astype(np.float64)
    tickers = df["ticker"].values
    uniq_t, ticker_idx = np.unique(tickers, return_inverse=True)
    return Precomputed(
        date_starts=date_starts.astype(np.int64),
        date_lens=date_lens.astype(np.int64),
        fwd=fwd,
        spy_per_date=spy_per_date,
        n_dates=len(date_starts),
        ticker_per_row=ticker_idx.astype(np.int32),
        n_tickers=len(uniq_t),
    )


def evaluate_formula(
    weights: np.ndarray,
    Xz: np.ndarray,
    df: pd.DataFrame,
    pc: Optional[Precomputed] = None,
    long_q: float = 0.20,
    short_q: float = 0.20,
) -> Dict[str, float]:
    """Apply weights → score → daily long-short portfolio → return Sharpe/Calmar/etc."""
    if pc is None:
        pc = precompute(df)
    scores = Xz @ weights  # (N,)
    port_ret = np.zeros(pc.n_dates, dtype=np.float64)
    # Track turnover via weight changes per ticker
    prev_w = np.zeros(pc.n_tickers, dtype=np.float64)
    turnover_sum = 0.0
    turnover_n = 0
    fwd = pc.fwd
    ticker_idx = pc.ticker_per_row

    for d in range(pc.n_dates):
        s = pc.date_starts[d]
        n = pc.date_lens[d]
        e = s + n
        if n < 5:
            cur_w = np.zeros(pc.n_tickers, dtype=np.float64)
            turnover_sum += np.abs(cur_w - prev_w).sum()
            turnover_n += 1
            prev_w = cur_w
            continue
        day_scores = scores[s:e]
        # Per-day ranks
        order = np.argsort(day_scores)
        ranks = np.empty(n, dtype=np.float64)
        ranks[order] = np.arange(n) / max(n - 1, 1)
        long_mask = ranks >= (1 - long_q)
        short_mask = ranks <= short_q
        w = np.zeros(n, dtype=np.float64)
        n_long = long_mask.sum()
        n_short = short_mask.sum()
        if n_long > 0:
            w[long_mask] = 0.5 / n_long
        if n_short > 0:
            w[short_mask] = -0.5 / n_short
        port_ret[d] = float(np.dot(w, fwd[s:e]))
        # Map to ticker-space for turnover
        cur_w = np.zeros(pc.n_tickers, dtype=np.float64)
        np.add.at(cur_w, ticker_idx[s:e], w)
        turnover_sum += np.abs(cur_w - prev_w).sum()
        turnover_n += 1
        prev_w = cur_w

    # Build daily DataFrame just for metrics
    daily = pd.DataFrame({"port_ret": port_ret, "spy_ret": pc.spy_per_date})
    if len(daily) < 30:
        return _bad()

    ret = daily["port_ret"].values
    spy = daily["spy_ret"].fillna(0).values

    mu = ret.mean() * TRADING_DAYS
    sd = ret.std() * np.sqrt(TRADING_DAYS)
    sharpe = mu / sd if sd > 1e-9 else 0.0

    equity = (1 + ret).cumprod()
    n_years = len(ret) / TRADING_DAYS
    cagr = equity[-1] ** (1 / n_years) - 1 if n_years > 0 else 0.0
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1
    maxdd = dd.min()
    calmar = cagr / abs(maxdd) if abs(maxdd) > 1e-6 else 0.0

    spy_mu = spy.mean() * TRADING_DAYS
    spy_sd = spy.std() * np.sqrt(TRADING_DAYS)
    spy_sharpe = spy_mu / spy_sd if spy_sd > 1e-9 else 0.0
    spy_equity = (1 + spy).cumprod()
    spy_cagr = spy_equity[-1] ** (1 / n_years) - 1 if n_years > 0 else 0.0

    # Turnover (precomputed in main loop)
    turnover = turnover_sum / max(turnover_n, 1)

    # Regime breakdown (classify by SPY 5d return forward at portfolio dates)
    spy_5d = pd.Series(spy).rolling(5).sum().fillna(0).values
    green = ret[spy_5d > 0.005]
    red = ret[spy_5d < -0.005]
    g_sharpe = (green.mean() * TRADING_DAYS) / (green.std() * np.sqrt(TRADING_DAYS) + 1e-9) if len(green) > 5 else 0.0
    r_sharpe = (red.mean() * TRADING_DAYS) / (red.std() * np.sqrt(TRADING_DAYS) + 1e-9) if len(red) > 5 else 0.0

    return {
        "sharpe": float(sharpe),
        "calmar": float(calmar),
        "cagr": float(cagr),
        "maxdd": float(maxdd),
        "spy_sharpe": float(spy_sharpe),
        "spy_cagr": float(spy_cagr),
        "spy_beat_sharpe": float(sharpe - 1.5 * spy_sharpe),
        "spy_beat_cagr": float(cagr - 1.5 * spy_cagr),
        "turnover": float(turnover),
        "green_sharpe": float(g_sharpe),
        "red_sharpe": float(r_sharpe),
        "regime_drift": float(abs(g_sharpe - r_sharpe) / max(abs(g_sharpe), abs(r_sharpe), 1e-6)),
        "n_days": int(len(daily)),
    }


def _bad() -> Dict[str, float]:
    return {k: -9.99 for k in ["sharpe", "calmar", "cagr", "maxdd", "spy_sharpe", "spy_cagr",
                                "spy_beat_sharpe", "spy_beat_cagr", "turnover",
                                "green_sharpe", "red_sharpe", "regime_drift"]} | {"n_days": 0}


def scalar_fitness(m: Dict[str, float]) -> float:
    """HC #559 R4 multi-objective scalarization. Calmar<1.0 → hard reject."""
    if m["calmar"] < HARD_CALMAR_FLOOR:
        return -1e6 + m["calmar"]  # rank rejects but allow gradient
    if m["regime_drift"] > 0.5:  # HC #428 R1: regime-tailored, not edge
        return -1e5 + m["sharpe"]
    score = (
        1.0 * m["spy_beat_sharpe"]
        + 0.7 * m["calmar"]
        + 0.5 * m["spy_beat_cagr"]
        - 0.05 * m["turnover"]
        - 0.3 * m["regime_drift"]
    )
    return score


# --------- GA ---------

def ga_search(
    Xz: np.ndarray,
    df: pd.DataFrame,
    features: List[str],
    pop: int = 80,
    gens: int = 40,
    seed: int = 13,
    elite: int = 8,
    sigma: float = 0.3,
    sparsity_p: float = 0.4,  # prob of zeroing a weight (sparse formulas)
) -> Tuple[np.ndarray, Dict[str, float], List[float]]:
    rng = np.random.default_rng(seed)
    F = Xz.shape[1]
    pc = precompute(df)

    # Init
    population = rng.normal(0, 0.5, size=(pop, F))
    mask = rng.random((pop, F)) > sparsity_p
    population *= mask

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
                        "sharpe": metrics_list[0]["sharpe"]})

        # Selection: keep elites, breed rest via tournament + crossover + mutation
        new_pop = list(population[:elite])
        while len(new_pop) < pop:
            i1, i2 = rng.integers(0, pop // 2, size=2)
            p1, p2 = population[i1], population[i2]
            # Uniform crossover
            mask_x = rng.random(F) < 0.5
            child = np.where(mask_x, p1, p2)
            # Gaussian mutation
            child = child + rng.normal(0, sigma, size=F) * (rng.random(F) < 0.3)
            # Sparsity mutation: occasionally zero a weight or revive one
            sparsity_flip = rng.random(F) < 0.05
            child = np.where(sparsity_flip, 0.0, child)
            new_pop.append(child)
        population = np.array(new_pop)

        # Adaptive sigma decay
        sigma = max(0.05, sigma * 0.97)

        if gen % 5 == 0 or gen == gens - 1:
            print(f"  gen {gen:3d}: best={scores[0]:+.3f} calmar={metrics_list[0]['calmar']:+.2f} "
                  f"sharpe={metrics_list[0]['sharpe']:+.2f} dd={metrics_list[0]['maxdd']*100:+.1f}% "
                  f"drift={metrics_list[0]['regime_drift']:.2f}", flush=True)

    return best_w, best_metrics, history


def format_expression(weights: np.ndarray, features: List[str], threshold: float = 0.05) -> str:
    terms = []
    order = np.argsort(-np.abs(weights))
    for idx in order:
        w = weights[idx]
        if abs(w) < threshold:
            continue
        sign = "+" if w > 0 else "-"
        terms.append(f"{sign}{abs(w):.3f}*{features[idx]}")
    if not terms:
        return "score = 0"
    expr = "score = " + " ".join(terms)
    if expr.startswith("score = +"):
        expr = "score = " + expr[9:]
    return expr


def run_sector(sector: str, pop: int, gens: int, seed: int,
               oot_start: Optional[pd.Timestamp] = None,
               fstore_version: str = "v1",
               universe_file: str = "universe.parquet",
               prices_file: str = "prices.parquet",
               out_tag: str = "v1") -> Dict:
    print(f"\n=== GA: sector={sector} pop={pop} gens={gens} fstore={fstore_version} oot_start={oot_start} ===", flush=True)
    panel, prices, spy = load_data(sector, fstore_version=fstore_version,
                                    universe_file=universe_file, prices_file=prices_file)
    print(f"  panel rows={len(panel)} tickers={panel['ticker'].nunique()} dates={panel['date'].nunique()}", flush=True)

    # OOT split
    if oot_start is not None:
        train_panel = panel[panel["date"] < oot_start].copy()
        oot_panel = panel[panel["date"] >= oot_start].copy()
        print(f"  TRAIN: rows={len(train_panel)} dates={train_panel['date'].nunique()}", flush=True)
        print(f"  OOT  : rows={len(oot_panel)} dates={oot_panel['date'].nunique()}", flush=True)
        if oot_panel["date"].nunique() < 40:
            print(f"  WARN: OOT has only {oot_panel['date'].nunique()} days < 40 (HC #428 R1 minimum)", flush=True)
    else:
        train_panel = panel
        oot_panel = pd.DataFrame()

    Xz_train, y_train, df_train, used_features = build_feature_matrix(train_panel, DEFAULT_FEATURES)
    print(f"  train feature_matrix shape={Xz_train.shape} features={len(used_features)}", flush=True)

    t0 = time.time()
    best_w, best_m, hist = ga_search(Xz_train, df_train, used_features, pop=pop, gens=gens, seed=seed)
    dt = time.time() - t0
    print(f"  GA done in {dt:.1f}s (TRAIN fitness)", flush=True)

    expr = format_expression(best_w, used_features)
    print(f"  formula: {expr}", flush=True)
    print(f"  TRAIN metrics: sharpe={best_m['sharpe']:+.2f} calmar={best_m['calmar']:+.2f} "
          f"cagr={best_m['cagr']*100:+.1f}% maxdd={best_m['maxdd']*100:+.1f}% "
          f"green_S={best_m['green_sharpe']:+.2f} red_S={best_m['red_sharpe']:+.2f}", flush=True)

    # Evaluate on OOT
    oot_metrics = None
    if not oot_panel.empty:
        Xz_oot, _, df_oot, oot_feats = build_feature_matrix(oot_panel, DEFAULT_FEATURES)
        # Realign weights to oot_feats (should be identical since DEFAULT_FEATURES same)
        if oot_feats == used_features:
            oot_metrics = evaluate_formula(best_w, Xz_oot, df_oot)
            print(f"  OOT   metrics: sharpe={oot_metrics['sharpe']:+.2f} calmar={oot_metrics['calmar']:+.2f} "
                  f"cagr={oot_metrics['cagr']*100:+.1f}% maxdd={oot_metrics['maxdd']*100:+.1f}% "
                  f"green_S={oot_metrics['green_sharpe']:+.2f} red_S={oot_metrics['red_sharpe']:+.2f}", flush=True)
        else:
            print(f"  WARN: OOT feature set differs from TRAIN; skipping OOT eval", flush=True)

    artifact = {
        "sector": sector,
        "features": used_features,
        "weights": best_w.tolist(),
        "expression": expr,
        "fitness_train": best_m,
        "fitness_oot": oot_metrics,
        "trained_on": {
            "start": str(train_panel["date"].min().date()),
            "end": str(train_panel["date"].max().date()),
            "n_tickers": int(train_panel["ticker"].nunique()),
            "n_days": int(train_panel["date"].nunique()),
            "n_rows": int(len(train_panel)),
        },
        "oot_window": ({
            "start": str(oot_panel["date"].min().date()) if not oot_panel.empty else None,
            "end": str(oot_panel["date"].max().date()) if not oot_panel.empty else None,
            "n_days": int(oot_panel["date"].nunique()) if not oot_panel.empty else 0,
        } if not oot_panel.empty else None),
        "ga_history_tail": hist[-5:],
        "ga_config": {"pop": pop, "gens": gens, "seed": seed},
        "calmar_floor_pass_train": bool(best_m["calmar"] >= HARD_CALMAR_FLOOR),
        "regime_drift_pass_train": bool(best_m["regime_drift"] <= 0.5),
        "calmar_floor_pass_oot": bool(oot_metrics["calmar"] >= HARD_CALMAR_FLOOR) if oot_metrics else None,
        "regime_drift_pass_oot": bool(oot_metrics["regime_drift"] <= 0.5) if oot_metrics else None,
        "deployable": bool(oot_metrics and oot_metrics["calmar"] >= HARD_CALMAR_FLOOR
                           and oot_metrics["regime_drift"] <= 0.5
                           and oot_metrics["sharpe"] > 0.7),
        "fstore_version": fstore_version,
        "generated_at": pd.Timestamp.now().isoformat(),
    }
    out_path = OUT_DIR / f"formula_{out_tag}_{sector.replace(' ', '_')}.json"
    with open(out_path, "w") as f:
        json.dump(artifact, f, indent=2, default=str)
    print(f"  wrote {out_path}", flush=True)
    return artifact


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sector", default="Technology",
                    help="GICS sector name from universe, or 'all'.")
    ap.add_argument("--pop", type=int, default=80)
    ap.add_argument("--generations", type=int, default=40)
    ap.add_argument("--seed", type=int, default=13)
    ap.add_argument("--fstore-version", default="v1",
                    help="Feature store version subdir (v1, v2, ...)")
    ap.add_argument("--universe-file", default="universe.parquet",
                    help="Filename in CACHE for universe (universe.parquet, universe_v2.parquet)")
    ap.add_argument("--prices-file", default="prices.parquet",
                    help="Filename in CACHE for prices")
    ap.add_argument("--out-tag", default="v1",
                    help="Output filename tag (formula_<tag>_<sector>.json)")
    ap.add_argument("--oot-start", default=str(OOT_START.date()),
                    help="OOT split start date (YYYY-MM-DD). 'none' to disable.")
    args = ap.parse_args()
    oot_start = None if args.oot_start.lower() == "none" else pd.Timestamp(args.oot_start)

    kw = dict(pop=args.pop, gens=args.generations, seed=args.seed,
              oot_start=oot_start, fstore_version=args.fstore_version,
              universe_file=args.universe_file, prices_file=args.prices_file,
              out_tag=args.out_tag)

    if args.sector == "all":
        universe = pd.read_parquet(CACHE / args.universe_file)
        sectors = [s for s in universe["sector"].unique()
                   if s != "ETF" and (universe["sector"] == s).sum() >= 5]
        results = {}
        for s in sectors:
            try:
                results[s] = run_sector(s, **kw)
            except Exception as e:
                print(f"  sector {s} FAILED: {e}", flush=True)
                import traceback; traceback.print_exc()
                results[s] = {"error": str(e)}
        with open(OUT_DIR / f"summary_{args.out_tag}.json", "w") as f:
            summary = {k: ({
                "expression": v.get("expression"),
                "train_calmar": v.get("fitness_train", {}).get("calmar"),
                "train_sharpe": v.get("fitness_train", {}).get("sharpe"),
                "train_cagr": v.get("fitness_train", {}).get("cagr"),
                "oot_calmar": (v.get("fitness_oot") or {}).get("calmar"),
                "oot_sharpe": (v.get("fitness_oot") or {}).get("sharpe"),
                "oot_cagr": (v.get("fitness_oot") or {}).get("cagr"),
                "deployable": v.get("deployable"),
                "calmar_pass_train": v.get("calmar_floor_pass_train"),
                "calmar_pass_oot": v.get("calmar_floor_pass_oot"),
                "regime_pass_train": v.get("regime_drift_pass_train"),
                "regime_pass_oot": v.get("regime_drift_pass_oot"),
            } if "error" not in v else v) for k, v in results.items()}
            json.dump(summary, f, indent=2, default=str)
        print(f"\nSummary written: {OUT_DIR/f'summary_{args.out_tag}.json'}", flush=True)
    else:
        run_sector(args.sector, **kw)


if __name__ == "__main__":
    main()
