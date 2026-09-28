"""
HC #561 R3 — Walk-forward validation harness for the macro picker (GA v2).

WHY THIS EXISTS
---------------
The prior GA v2 results (ga_sector_fit_v2.py) used *median OOT-fold Sharpe* as
the GA fitness function, i.e. parameters were SELECTED on the very OOT windows
later reported.  That is in-sample selection and blocks any deployability claim.

WHAT THIS DOES
--------------
Rolling SLIDING walk-forward (HC #0 — NEVER expanding):
    3-year train window  /  1-year out-of-time test window  /  6-month step.

Per fold:
    1. GA v2 optimization (same operators / population mechanics / cost model /
       portfolio construction as ga_sector_fit_v2.py) with fitness computed on
       the TRAIN window ONLY (annualized net Sharpe).
    2. Best weights FROZEN, evaluated once on the following 1y OOT window.
Then all OOT windows are concatenated into a pooled OOT return stream
(non-overlapping by construction: step=6m vs oot=12m gives 2x-overlapping
fold schedule, so pooling uses EVERY-OTHER fold's OOT segment to keep the
concatenated stream non-overlapping; per-fold metrics still cover all folds).

Metrics per fold and pooled (HC #428 R1): Sharpe, Sortino, PF, WR, MaxDD,
Calmar + regime-stratified Sharpe (green/red/flat days classified by the
underlying universe equal-weight close-to-close return; SPY is not in the
price cache).  Regime symmetry gate:
    |S_green - S_red| / max(|S_green|, |S_red|) <= 0.50  -> PASS.

Everything is logged to MLflow (http://localhost:5000) under experiment
`macro_picker_walkforward`.  Fold params + OOT equity curves are saved to
output/macro_picker/walkforward/.

PERFORMANCE NOTE
----------------
ga_sector_fit_v2._build_portfolio_returns loops Python-level over every
trading day and is far too slow for per-fold GA re-optimization
(~15 folds x 60 pop x 20 gens).  This module re-implements the SAME portfolio
semantics (weekly rebalance, long top decile / short bottom decile, 0.5/0.5
gross split, 5/30 bps ADV-bucketed turnover costs, weights effective the day
AFTER the rebalance print) in vectorized numpy.  Data loading, GA operators,
GAConfig and the metric/verdict helpers are reused, not reinvented.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
OUT_DIR = ROOT / "output/macro_picker/walkforward"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = ROOT / "logs/macro_picker_walkforward.log"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [wf] %(message)s",
    handlers=[logging.FileHandler(LOG_PATH), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("wf")

# --- reuse: metric helpers (research/walk_forward.py) ----------------------
sys.path.insert(0, str(ROOT / "research"))
from walk_forward import (  # noqa: E402
    _annualize_sharpe, _annualize_sortino, _cagr, _max_dd, _calmar,
    _profit_factor, _win_rate, _verdict_for_metrics,
)

# --- reuse: GA v2 data loading + GA operators -------------------------------
sys.path.insert(0, str(ROOT / "strategy/macro_picker"))
from ga_sector_fit_v2 import (  # noqa: E402
    load_features, load_prices_and_adv, get_universe_sectors,
    GAConfig, _random_candidate, _mutate, _crossover,
)

MLFLOW_URI = "http://localhost:5000"
MLFLOW_EXPERIMENT = "macro_picker_walkforward"

TC_LOW_BPS = 5.0
TC_HIGH_BPS = 30.0
ADV_HIGH_THRESHOLD = 20e6
FLAT_BAND = 0.001  # |underlying c2c ret| <= 0.1% -> flat day
REGIME_SYMMETRY_MAX = 0.50  # HC #428 R1


# ===========================================================================
# Vectorized sector panel (precomputed once per sector)
# ===========================================================================
class SectorPanel:
    """Precomputed cross-sectional rank tensor + return/ADV matrices for one
    sector so a GA candidate can be evaluated in milliseconds.

    Semantics match ga_sector_fit_v2:
      * features pct-ranked cross-sectionally per date (rank - 0.5), NaN -> 0
      * score = ranks @ weights
      * universe per date = tickers with BOTH a feature row and a price row
    """

    def __init__(self, sector: str, features: pd.DataFrame, prices: pd.DataFrame,
                 tickers: list[str]):
        self.sector = sector
        self.tickers = sorted(tickers)
        t_idx = {t: i for i, t in enumerate(self.tickers)}

        f = features[features["ticker"].isin(self.tickers)].copy()
        p = prices[prices["ticker"].isin(self.tickers)].copy()

        drop = {"ticker", "date"}
        self.feat_cols = [c for c in f.columns
                          if c not in drop and pd.api.types.is_numeric_dtype(f[c])]

        # cross-sectional pct rank per date (same as ga_sector_fit_v2._score_panel)
        for c in self.feat_cols:
            f[c] = f.groupby("date")[c].rank(pct=True) - 0.5

        # date axis = intersection-friendly union of price dates (mtm needs prices)
        self.dates = pd.DatetimeIndex(sorted(p["date"].unique()))
        d_idx = {d: i for i, d in enumerate(self.dates)}
        nd, nt = len(self.dates), len(self.tickers)

        # return + ADV matrices
        self.R = np.zeros((nd, nt))            # NaN ret treated as 0 (v2 fillna(0))
        self.has_price = np.zeros((nd, nt), bool)
        self.ADV = np.full((nd, nt), np.nan)
        di = p["date"].map(d_idx).values
        ti = p["ticker"].map(t_idx).values
        self.R[di, ti] = np.nan_to_num(p["ret"].values, nan=0.0)
        self.has_price[di, ti] = True
        self.ADV[di, ti] = p["adv_60d"].values

        # rank tensor X: (nd, nt, nf); NaN feature -> 0 after ranking (v2 fillna(0))
        nf = len(self.feat_cols)
        self.X = np.zeros((nd, nt, nf))
        self.has_feat = np.zeros((nd, nt), bool)
        f = f[f["date"].isin(d_idx)]
        di = f["date"].map(d_idx).values
        ti = f["ticker"].map(t_idx).values
        self.X[di, ti, :] = np.nan_to_num(f[self.feat_cols].values, nan=0.0)
        self.has_feat[di, ti] = True

        self.present = self.has_price & self.has_feat  # tradable universe per date

        # weekly rebalance flags: first trading day of each ISO week
        weeks = self.dates.to_period("W")
        self.is_rebal = np.zeros(nd, bool)
        self.is_rebal[pd.Series(np.arange(nd)).groupby(weeks).min().values] = True

        log.info(f"[{sector}] panel: {nd} dates x {nt} tickers x {nf} feats")

    # ------------------------------------------------------------------
    def simulate(self, weights: np.ndarray, i0: int, i1: int) -> pd.Series:
        """Net daily portfolio returns on dates[i0:i1], starting flat.

        Weekly rebalance: long top decile / short bottom decile (0.5/0.5 gross),
        weights effective the day AFTER the rebalance print; turnover cost
        charged on the rebalance day (5 bps if 60d ADV >= $20M else 30 bps).
        """
        S = self.X[i0:i1] @ weights              # (n, nt) scores
        R = self.R[i0:i1]
        ADV = self.ADV[i0:i1]
        present = self.present[i0:i1]
        is_rb = self.is_rebal[i0:i1]
        n, nt = S.shape

        w_state = np.zeros(nt)
        net = np.zeros(n)
        for t in range(n):
            cost = 0.0
            new_w = None
            if is_rb[t]:
                mask = present[t]
                if mask.sum() >= 10:
                    sc = S[t][mask]
                    q_lo, q_hi = np.quantile(sc, 0.1), np.quantile(sc, 0.9)
                    longs = mask & (S[t] >= q_hi) & present[t]
                    shorts = mask & (S[t] <= q_lo) & present[t]
                    nl, ns = int(longs.sum()), int(shorts.sum())
                    if nl > 0 and ns > 0:
                        new_w = np.zeros(nt)
                        new_w[longs] = 0.5 / nl
                        new_w[shorts] = -0.5 / ns
                        delta = np.abs(new_w - w_state)
                        adv = ADV[t]
                        rate = np.where(np.nan_to_num(adv, nan=0.0) >= ADV_HIGH_THRESHOLD,
                                        TC_LOW_BPS / 1e4, TC_HIGH_BPS / 1e4)
                        cost = float((delta * rate).sum())
            # mark-to-market with PRIOR weights, then roll
            net[t] = float(w_state @ R[t]) - cost
            if new_w is not None:
                w_state = new_w
        return pd.Series(net, index=self.dates[i0:i1])


# ===========================================================================
# Fold schedule — SLIDING (HC #0)
# ===========================================================================
def fold_schedule(dates: pd.DatetimeIndex, train_months: int = 36,
                  oot_months: int = 12, step_months: int = 6) -> list[dict]:
    folds = []
    cursor = pd.Timestamp(dates.min())
    end = pd.Timestamp(dates.max())
    while True:
        tr_start = cursor                                   # SLIDES forward
        tr_end = tr_start + pd.DateOffset(months=train_months)
        oot_end = tr_end + pd.DateOffset(months=oot_months)
        if oot_end > end + pd.Timedelta(days=1):
            break
        folds.append({"train_start": tr_start, "train_end": tr_end,
                      "oot_start": tr_end, "oot_end": oot_end})
        cursor = cursor + pd.DateOffset(months=step_months)
    return folds


def _slice_idx(dates: pd.DatetimeIndex, start, end) -> tuple[int, int]:
    return int(dates.searchsorted(start, "left")), int(dates.searchsorted(end, "left"))


# ===========================================================================
# Metrics incl. regime stratification (HC #428 R1)
# ===========================================================================
def regime_labels(market_ret: pd.Series) -> pd.Series:
    lab = pd.Series("flat", index=market_ret.index)
    lab[market_ret > FLAT_BAND] = "green"
    lab[market_ret < -FLAT_BAND] = "red"
    return lab


def full_metrics(rets: pd.Series, market_ret: pd.Series) -> dict:
    m = {
        "sharpe": _annualize_sharpe(rets),
        "sortino": _annualize_sortino(rets),
        "cagr": _cagr(rets),
        "max_dd": _max_dd(rets),
        "calmar": _calmar(rets),
        "pf": _profit_factor(rets),
        "wr": _win_rate(rets),
        "n_days": int(rets.dropna().shape[0]),
    }
    lab = regime_labels(market_ret.reindex(rets.index).fillna(0.0))
    for reg in ("green", "red", "flat"):
        sub = rets[lab == reg]
        m[f"sharpe_{reg}"] = _annualize_sharpe(sub) if len(sub) >= 10 else float("nan")
        m[f"n_days_{reg}"] = int(len(sub))
    sg, sr = m["sharpe_green"], m["sharpe_red"]
    if np.isfinite(sg) and np.isfinite(sr) and max(abs(sg), abs(sr)) > 0:
        m["regime_asym"] = abs(sg - sr) / max(abs(sg), abs(sr))
    else:
        m["regime_asym"] = float("nan")
    return m


# ===========================================================================
# Per-fold GA (fitness on TRAIN window ONLY — fixes the in-sample selection)
# ===========================================================================
def ga_optimize_train(panel: SectorPanel, i0: int, i1: int, cfg: GAConfig,
                      seed: int) -> tuple[np.ndarray, float]:
    rng = np.random.default_rng(seed)
    nf = len(panel.feat_cols)

    def fitness(w: np.ndarray) -> float:
        rets = panel.simulate(w, i0, i1)
        if rets.empty or rets.std() == 0:
            return -999.0
        s = _annualize_sharpe(rets)
        return float(s) if np.isfinite(s) else -999.0

    population = [_random_candidate(nf, cfg.max_active, rng) for _ in range(cfg.pop)]
    best_fit, best_w = -np.inf, None
    for gen in range(cfg.gens):
        scored = sorted(((fitness(w), w) for w in population),
                        key=lambda x: x[0], reverse=True)
        if scored[0][0] > best_fit:
            best_fit, best_w = scored[0][0], scored[0][1].copy()
        elite = [w for _, w in scored[:cfg.elite]]
        new_pop = list(elite)
        while len(new_pop) < cfg.pop:
            p1, p2 = rng.choice(len(elite), size=2, replace=True)
            child = _crossover(elite[p1], elite[p2], rng)
            if rng.random() < cfg.mut_rate:
                child = _mutate(child, cfg.max_active, rng)
            active = np.where(child != 0)[0]
            if len(active) > cfg.max_active:
                drop_idx = rng.choice(active, size=len(active) - cfg.max_active,
                                      replace=False)
                child[drop_idx] = 0.0
            new_pop.append(child)
        population = new_pop
    return best_w, best_fit


# ===========================================================================
# MLflow helpers
# ===========================================================================
def _mlflow_setup():
    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    return mlflow


def _log_metrics(mlflow, m: dict):
    for k, v in m.items():
        if isinstance(v, (int, float)) and np.isfinite(v):
            mlflow.log_metric(k, float(v))


# ===========================================================================
# Main driver
# ===========================================================================
def run_sector(panel: SectorPanel, folds: list[dict], cfg: GAConfig,
               market_ret: pd.Series, mlflow) -> dict:
    sector_dir = OUT_DIR / panel.sector.replace(" ", "_").lower()
    sector_dir.mkdir(parents=True, exist_ok=True)

    per_fold, oot_segments, pooled_segments = [], [], []
    for k, f in enumerate(folds):
        t0 = time.time()
        tr0, tr1 = _slice_idx(panel.dates, f["train_start"], f["train_end"])
        oo0, oo1 = _slice_idx(panel.dates, f["oot_start"], f["oot_end"])
        if tr1 - tr0 < 250 or oo1 - oo0 < 20:
            log.info(f"[{panel.sector}] fold {k}: too few days, skipped")
            continue

        w, train_fit = ga_optimize_train(panel, tr0, tr1, cfg, seed=cfg.seed + k)
        oot_rets = panel.simulate(w, oo0, oo1)
        m = full_metrics(oot_rets, market_ret)
        m_train = full_metrics(panel.simulate(w, tr0, tr1), market_ret)

        active = np.where(w != 0)[0]
        weights = {panel.feat_cols[i]: float(w[i]) for i in active}
        fold_rec = {
            "fold": k,
            "train_start": str(f["train_start"].date()),
            "train_end": str(f["train_end"].date()),
            "oot_start": str(f["oot_start"].date()),
            "oot_end": str(f["oot_end"].date()),
            "train_sharpe": train_fit,
            "weights": dict(sorted(weights.items(), key=lambda kv: -abs(kv[1]))),
            "oot_metrics": m,
        }
        per_fold.append(fold_rec)
        oot_segments.append((k, oot_rets))
        if k % 2 == 0:  # every-other fold -> non-overlapping pooled OOT stream
            pooled_segments.append(oot_rets)

        # save params + equity curve
        with open(sector_dir / f"fold_{k:02d}_params.json", "w") as fh:
            json.dump(fold_rec, fh, indent=2, default=str)
        eq = (1.0 + oot_rets).cumprod().rename("equity")
        eq.to_frame().assign(ret=oot_rets).to_csv(sector_dir / f"fold_{k:02d}_oot_equity.csv")

        # MLflow per-fold run
        with mlflow.start_run(run_name=f"{panel.sector}_fold{k:02d}"):
            mlflow.log_params({
                "sector": panel.sector, "fold": k,
                "train_start": fold_rec["train_start"], "train_end": fold_rec["train_end"],
                "oot_start": fold_rec["oot_start"], "oot_end": fold_rec["oot_end"],
                "window": "sliding_36m_train_12m_oot_6m_step",
                "ga_pop": cfg.pop, "ga_gens": cfg.gens, "ga_max_active": cfg.max_active,
                "n_active_weights": len(weights),
            })
            _log_metrics(mlflow, {f"oot_{k2}": v2 for k2, v2 in m.items()})
            mlflow.log_metric("train_sharpe", train_fit)
            mlflow.log_metric("train_oot_sharpe_gap",
                              train_fit - m["sharpe"] if np.isfinite(m["sharpe"]) else 999)
            mlflow.log_artifact(str(sector_dir / f"fold_{k:02d}_params.json"))

        log.info(f"[{panel.sector}] fold {k}: train Sharpe {train_fit:+.2f} -> "
                 f"OOT Sharpe {m['sharpe']:+.2f} PF {m['pf']:.2f} "
                 f"({time.time() - t0:.0f}s)")

    if not per_fold:
        return {"sector": panel.sector, "skipped": True}

    pooled = pd.concat(pooled_segments).sort_index()
    pooled = pooled[~pooled.index.duplicated(keep="first")]
    if pooled.std() == 0 or pooled.abs().sum() == 0:
        log.info(f"[{panel.sector}] pooled OOT degenerate (never traded — "
                 f"cross-section < 10 names) — excluded from combined")
        return {"sector": panel.sector, "skipped": True}
    pm = full_metrics(pooled, market_ret)
    eq = (1.0 + pooled).cumprod().rename("equity")
    eq.to_frame().assign(ret=pooled).to_csv(sector_dir / "pooled_oot_equity.csv")

    with mlflow.start_run(run_name=f"{panel.sector}_pooled_oot"):
        mlflow.log_params({
            "sector": panel.sector, "n_folds": len(per_fold),
            "pooling": "every_other_fold_non_overlapping",
            "window": "sliding_36m_train_12m_oot_6m_step",
        })
        _log_metrics(mlflow, {f"pooled_{k2}": v2 for k2, v2 in pm.items()})
        mlflow.log_artifact(str(sector_dir / "pooled_oot_equity.csv"))

    log.info(f"[{panel.sector}] POOLED OOT: Sharpe {pm['sharpe']:+.2f} "
             f"Sortino {pm['sortino']:+.2f} PF {pm['pf']:.2f} WR {pm['wr']:.2f} "
             f"MaxDD {pm['max_dd']:.2%} Calmar {pm['calmar']:.2f} "
             f"asym {pm['regime_asym']:.2f}")
    return {"sector": panel.sector, "per_fold": per_fold, "pooled_metrics": pm,
            "pooled_rets": pooled}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-tickers", type=int, default=8)
    ap.add_argument("--sectors", type=str, default="",
                    help="comma-separated; empty = all sectors with >= min-tickers")
    args = ap.parse_args()

    cfg = GAConfig(pop=args.pop, gens=args.gens, seed=args.seed, n_jobs=1)
    log.info(f"HC #561 R3 walk-forward start | cfg={cfg.__dict__}")

    features = load_features()
    prices = load_prices_and_adv()
    sectors_map = get_universe_sectors()

    # market proxy for regime classification (SPY not in price cache):
    # equal-weight universe close-to-close mean daily return
    market_ret = prices.groupby("date")["ret"].mean().sort_index()

    counts = sectors_map["sector"].value_counts()
    if args.sectors:
        targets = [s.strip() for s in args.sectors.split(",") if s.strip()]
    else:
        targets = counts[counts >= args.min_tickers].index.tolist()
    log.info(f"target sectors: {targets}")

    mlflow = _mlflow_setup()
    results = []
    for sector in targets:
        tickers = sectors_map[sectors_map["sector"] == sector]["ticker"].tolist()
        if len(tickers) < args.min_tickers:
            log.info(f"skip {sector}: {len(tickers)} tickers")
            continue
        panel = SectorPanel(sector, features, prices, tickers)
        folds = fold_schedule(panel.dates)
        log.info(f"[{sector}] {len(folds)} folds")
        res = run_sector(panel, folds, cfg, market_ret, mlflow)
        if not res.get("skipped"):
            results.append(res)

    if not results:
        log.error("no sector produced folds — aborting summary")
        return

    # combined macro picker = equal-weight across sector sleeves (pooled OOT)
    combined = pd.concat([r["pooled_rets"].rename(r["sector"]) for r in results],
                         axis=1).mean(axis=1, skipna=True).dropna().sort_index()
    cm = full_metrics(combined, market_ret)
    eq = (1.0 + combined).cumprod().rename("equity")
    eq.to_frame().assign(ret=combined).to_csv(OUT_DIR / "combined_pooled_oot_equity.csv")

    with mlflow.start_run(run_name="combined_pooled_oot"):
        mlflow.log_params({
            "sectors": ",".join(r["sector"] for r in results),
            "window": "sliding_36m_train_12m_oot_6m_step",
            "pooling": "every_other_fold_non_overlapping",
        })
        _log_metrics(mlflow, {f"pooled_{k}": v for k, v in cm.items()})
        mlflow.log_artifact(str(OUT_DIR / "combined_pooled_oot_equity.csv"))

    # ---------------- SUMMARY.md ----------------
    sym_pass = (np.isfinite(cm["regime_asym"]) and cm["regime_asym"] <= REGIME_SYMMETRY_MAX)
    verdict_gates = _verdict_for_metrics(cm)
    deploy_grade = (verdict_gates in ("TARGET MET", "STRETCH MET")) and sym_pass \
        and np.isfinite(cm["sharpe"]) and cm["sharpe"] > 0
    verdict = "DEPLOY-GRADE" if deploy_grade else "R&D ONLY — NOT DEPLOYABLE"

    def fm(x, p=3):
        return "n/a" if x is None or not np.isfinite(x) else f"{x:.{p}f}"

    lines = [
        "# Macro Picker — HC #561 R3 Walk-Forward Validation (GA v2 re-run)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "Design: SLIDING 36m train / 12m OOT / 6m step (HC #0). GA fitness on the",
        "TRAIN window ONLY; frozen params evaluated on the following 1y OOT window.",
        "Pooled OOT stream uses every-other fold so segments are non-overlapping.",
        "Costs: 5/30 bps ADV-bucketed round-trip turnover. Weekly rebalance,",
        "long top decile / short bottom decile.",
        "",
        "## Pooled OOT — combined macro picker (equal-weight sector sleeves)",
        "",
        "| Metric | Value |",
        "|---|---|",
        f"| Sharpe | {fm(cm['sharpe'])} |",
        f"| Sortino | {fm(cm['sortino'])} |",
        f"| PF | {fm(cm['pf'])} |",
        f"| WR | {fm(cm['wr'])} |",
        f"| MaxDD | {fm(cm['max_dd'])} |",
        f"| Calmar | {fm(cm['calmar'])} |",
        f"| CAGR | {fm(cm['cagr'])} |",
        f"| Sharpe (green days) | {fm(cm['sharpe_green'])} |",
        f"| Sharpe (red days) | {fm(cm['sharpe_red'])} |",
        f"| Sharpe (flat days) | {fm(cm['sharpe_flat'])} |",
        f"| Regime asymmetry | {fm(cm['regime_asym'])} (gate <= 0.50: "
        f"{'PASS' if sym_pass else 'FAIL'}) |",
        f"| OOT days pooled | {cm['n_days']} |",
        "",
        f"## Verdict: **{verdict}**",
        "",
        f"- HC #561 R4 gates on pooled OOT: {verdict_gates}",
        f"- HC #428 R1 regime symmetry: {'PASS' if sym_pass else 'FAIL'}",
        "",
        "## Per-sector pooled OOT",
        "",
        "| Sector | folds | Sharpe | Sortino | PF | WR | MaxDD | Calmar | RegAsym |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        pmm = r["pooled_metrics"]
        lines.append(
            f"| {r['sector']} | {len(r['per_fold'])} | {fm(pmm['sharpe'])} | "
            f"{fm(pmm['sortino'])} | {fm(pmm['pf'])} | {fm(pmm['wr'])} | "
            f"{fm(pmm['max_dd'])} | {fm(pmm['calmar'])} | {fm(pmm['regime_asym'])} |")
    lines += ["", "## Per-fold OOT Sharpe (train Sharpe -> OOT Sharpe)", ""]
    for r in results:
        lines.append(f"### {r['sector']}")
        for f in r["per_fold"]:
            m = f["oot_metrics"]
            lines.append(
                f"- fold {f['fold']:02d} OOT {f['oot_start']}..{f['oot_end']}: "
                f"train {f['train_sharpe']:+.2f} -> OOT {fm(m['sharpe'], 2)} "
                f"(PF {fm(m['pf'], 2)}, WR {fm(m['wr'], 2)}, MaxDD {fm(m['max_dd'], 2)})")
        lines.append("")
    lines += [
        "## Context",
        "",
        "Prior GA v2 numbers were in-sample-selected (fitness = median OOT-fold",
        "Sharpe). This harness removes that selection bias; the pooled OOT result",
        "above is the honest deployability evidence for the macro picker.",
        "",
    ]
    with open(OUT_DIR / "SUMMARY.md", "w") as fh:
        fh.write("\n".join(lines))
    log.info(f"SUMMARY written -> {OUT_DIR / 'SUMMARY.md'}")
    log.info(f"VERDICT: {verdict} | pooled Sharpe {fm(cm['sharpe'])} "
             f"Sortino {fm(cm['sortino'])} PF {fm(cm['pf'])} WR {fm(cm['wr'])} "
             f"MaxDD {fm(cm['max_dd'])} Calmar {fm(cm['calmar'])}")


if __name__ == "__main__":
    main()
