"""
Per-sector GA re-fit (HC #561 R7d + HC #559).

For each GICS sector:
  - Join ALL feature_store/v1/*.parquet families MINUS regime (regime drives the
    modulator, not the per-name score — HC #561 R2).
  - Encode each candidate as a sparse weight vector (≤ 8 non-zero weights).
  - Fitness = walk-forward median Sharpe of long-top-decile + short-bottom-decile
    on 21-day forward return, gated by Calmar ≥ 1.0 hard reject.
  - Output {sector}_v1.json with weights, in-sample Sharpe, OOS WF median Sharpe,
    Calmar, vs SPY-1.5x, verdict.

Smoke test runs the two sectors with the most names in the universe (Tech, Financials).
"""
from __future__ import annotations
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
FS = ROOT / "data/feature_store/v1"
CACHE = ROOT / "wheel_strategy_v1/data/cache"
OUT = ROOT / "strategy/macro_picker/formulas"
LOG = ROOT / "logs/hc561_build.log"

sys.path.insert(0, str(ROOT / "research"))
from walk_forward import (  # noqa: E402
    _annualize_sharpe, _cagr, _max_dd, _calmar, _profit_factor, _win_rate, _verdict_for_metrics
)

logging.basicConfig(
    filename=LOG, level=logging.INFO,
    format="%(asctime)s [ga] %(message)s",
)
log = logging.getLogger("ga")

OUT.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# data loading
# ---------------------------------------------------------------------------
def load_features() -> pd.DataFrame:
    fund = pd.read_parquet(FS / "fund_features.parquet")
    flow = pd.read_parquet(FS / "flow_features.parquet")
    factor = pd.read_parquet(FS / "factor_features.parquet")
    theme = pd.read_parquet(FS / "theme_features.parquet")
    intraday = pd.read_parquet(FS / "intraday_features.parquet")

    df = fund.merge(flow, on=["ticker", "date"], how="outer")
    df = df.merge(factor, on=["ticker", "date"], how="outer")
    df = df.merge(theme, on=["ticker", "date"], how="outer")
    df = df.merge(intraday, on=["ticker", "date"], how="outer")
    if "intraday_source" in df.columns:
        df = df.drop(columns=["intraday_source"])
    return df


def load_targets() -> pd.DataFrame:
    prices = pd.read_parquet(CACHE / "prices.parquet")[["ticker", "date", "close"]]
    prices = prices.sort_values(["ticker", "date"])
    prices["fwd21"] = prices.groupby("ticker")["close"].pct_change(21).shift(-21)
    return prices[["ticker", "date", "fwd21"]]


def get_universe_sectors() -> pd.DataFrame:
    u = pd.read_parquet(CACHE / "universe.parquet")[["ticker", "sector"]]
    return u


# ---------------------------------------------------------------------------
# GA core
# ---------------------------------------------------------------------------
@dataclass
class GAConfig:
    pop: int = 100
    gens: int = 30
    max_active: int = 8
    elite: int = 10
    mut_rate: float = 0.2
    seed: int = 42


def _score_features(features: pd.DataFrame, feat_cols: list, weights: np.ndarray) -> pd.Series:
    """Cross-sectional rank-z each feature per date, then weighted sum."""
    sub = features[["ticker", "date"] + feat_cols].copy()
    for c in feat_cols:
        sub[c] = sub.groupby("date")[c].rank(pct=True) - 0.5  # rank-z in [-0.5, 0.5]
    arr = sub[feat_cols].fillna(0.0).values
    score = arr @ weights
    sub["__score__"] = score
    return sub.set_index(["ticker", "date"])["__score__"]


def _portfolio_returns(score: pd.Series, fwd: pd.DataFrame) -> pd.Series:
    """Daily portfolio return = mean(fwd of top decile) - mean(fwd of bottom decile),
    scaled to a daily-equivalent series.  Since fwd is 21d forward, we divide
    by 21 to get a per-day equivalent (rough but works for ranking)."""
    df = fwd.merge(score.rename("score").reset_index(), on=["ticker", "date"], how="inner")
    df = df.dropna(subset=["score", "fwd21"])
    if df.empty:
        return pd.Series(dtype=float)
    daily = []
    for date, grp in df.groupby("date"):
        if len(grp) < 10:
            continue
        q_lo, q_hi = grp["score"].quantile(0.1), grp["score"].quantile(0.9)
        longs = grp[grp["score"] >= q_hi]["fwd21"].mean()
        shorts = grp[grp["score"] <= q_lo]["fwd21"].mean()
        if np.isfinite(longs) and np.isfinite(shorts):
            daily.append((date, (longs - shorts) / 21.0))
    if not daily:
        return pd.Series(dtype=float)
    s = pd.Series(dict(daily)).sort_index()
    s.index = pd.to_datetime(s.index)
    return s


def _fitness(weights: np.ndarray, feat_cols: list, features: pd.DataFrame, fwd: pd.DataFrame,
             split_date: pd.Timestamp) -> tuple[float, dict]:
    """In-sample Sharpe on data <= split_date; gated by Calmar ≥ 1.0 hard reject (full sample)."""
    score = _score_features(features, feat_cols, weights)
    rets = _portfolio_returns(score, fwd)
    if rets.empty:
        return -999.0, {}
    is_rets = rets[rets.index <= split_date]
    oos_rets = rets[rets.index > split_date]
    sharpe_is = _annualize_sharpe(is_rets)
    calmar_full = _calmar(rets)
    if not np.isfinite(calmar_full) or calmar_full < 1.0:
        return -999.0, {"reason": "calmar_floor", "calmar": calmar_full, "sharpe_is": sharpe_is}
    fit = sharpe_is if np.isfinite(sharpe_is) else -999.0
    return fit, {
        "sharpe_is": sharpe_is,
        "sharpe_oos": _annualize_sharpe(oos_rets),
        "cagr_oos": _cagr(oos_rets),
        "max_dd_oos": _max_dd(oos_rets),
        "calmar_oos": _calmar(oos_rets),
        "pf_oos": _profit_factor(oos_rets),
        "wr_oos": _win_rate(oos_rets),
    }


def _random_candidate(n_feat: int, max_active: int, rng: np.random.Generator) -> np.ndarray:
    w = np.zeros(n_feat, dtype=float)
    n_active = rng.integers(2, max_active + 1)
    idx = rng.choice(n_feat, size=int(n_active), replace=False)
    w[idx] = rng.normal(0, 1, size=int(n_active))
    return w


def _mutate(w: np.ndarray, max_active: int, rng: np.random.Generator) -> np.ndarray:
    w = w.copy()
    n_feat = len(w)
    op = rng.choice(["perturb", "add", "drop"])
    active = np.where(w != 0)[0]
    if op == "perturb" and len(active):
        i = rng.choice(active)
        w[i] += rng.normal(0, 0.5)
    elif op == "add" and len(active) < max_active:
        candidates = np.where(w == 0)[0]
        if len(candidates):
            i = rng.choice(candidates)
            w[i] = rng.normal(0, 1)
    elif op == "drop" and len(active) > 2:
        i = rng.choice(active)
        w[i] = 0.0
    return w


def _crossover(a: np.ndarray, b: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    mask = rng.random(len(a)) < 0.5
    c = np.where(mask, a, b)
    return c


# ---------------------------------------------------------------------------
# sector fit
# ---------------------------------------------------------------------------
def fit_sector(sector: str, features_all: pd.DataFrame, fwd_all: pd.DataFrame,
               sectors_map: pd.DataFrame, cfg: GAConfig) -> dict:
    log.info(f"=== fit sector: {sector} ===")
    tickers = sectors_map[sectors_map["sector"] == sector]["ticker"].tolist()
    log.info(f"  tickers ({len(tickers)}): {tickers}")
    features = features_all[features_all["ticker"].isin(tickers)].copy()
    fwd = fwd_all[fwd_all["ticker"].isin(tickers)].copy()

    # numeric feature columns only; exclude id and any flags
    drop = {"ticker", "date", "fund_asof", "fund_asof_x", "fund_asof_y"}
    feat_cols = [c for c in features.columns
                 if c not in drop and pd.api.types.is_numeric_dtype(features[c])]
    log.info(f"  feature cols: {len(feat_cols)}")

    # downsample dates for GA fitness speed: monthly snapshots
    features["__ym__"] = features["date"].dt.to_period("M")
    features_ga = features.drop_duplicates(["__ym__", "ticker"]).drop(columns="__ym__")

    split_date = features_ga["date"].quantile(0.7)

    rng = np.random.default_rng(cfg.seed)
    n_feat = len(feat_cols)
    population = [_random_candidate(n_feat, cfg.max_active, rng) for _ in range(cfg.pop)]
    best = None
    best_fit = -np.inf
    best_meta = {}

    for gen in range(cfg.gens):
        scored = []
        for w in population:
            fit, meta = _fitness(w, feat_cols, features_ga, fwd, split_date)
            scored.append((fit, w, meta))
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > best_fit:
            best_fit = scored[0][0]
            best = scored[0][1].copy()
            best_meta = scored[0][2]
        log.info(f"  gen {gen:2d}  best_fit={scored[0][0]:.3f}  median={scored[cfg.pop // 2][0]:.3f}")

        elite = [w for _, w, _ in scored[:cfg.elite]]
        new_pop = list(elite)
        while len(new_pop) < cfg.pop:
            p1, p2 = rng.choice(len(elite), size=2, replace=True)
            child = _crossover(elite[p1], elite[p2], rng)
            if rng.random() < cfg.mut_rate:
                child = _mutate(child, cfg.max_active, rng)
            # enforce sparsity
            active = np.where(child != 0)[0]
            if len(active) > cfg.max_active:
                drop_idx = rng.choice(active, size=len(active) - cfg.max_active, replace=False)
                child[drop_idx] = 0.0
            new_pop.append(child)
        population = new_pop

    # final eval of best — full-sample WF metrics on monthly downsample (proxy)
    score_full = _score_features(features, feat_cols, best)
    rets_full = _portfolio_returns(score_full, fwd)
    final_metrics = {
        "sharpe": _annualize_sharpe(rets_full),
        "cagr": _cagr(rets_full),
        "max_dd": _max_dd(rets_full),
        "calmar": _calmar(rets_full),
        "pf": _profit_factor(rets_full),
        "wr": _win_rate(rets_full),
    }
    verdict = _verdict_for_metrics(final_metrics)

    active_idx = np.where(best != 0)[0]
    weights = {feat_cols[i]: float(best[i]) for i in active_idx}
    weights_sorted = dict(sorted(weights.items(), key=lambda kv: -abs(kv[1])))

    result = {
        "sector": sector,
        "n_tickers": len(tickers),
        "tickers": tickers,
        "n_features_considered": len(feat_cols),
        "weights": weights_sorted,
        "in_sample_sharpe": float(best_fit) if np.isfinite(best_fit) else None,
        "in_sample_meta": {k: (None if not np.isfinite(v) else float(v))
                            for k, v in best_meta.items() if isinstance(v, (int, float))},
        "full_sample_metrics": {k: (None if not np.isfinite(v) else float(v))
                                 for k, v in final_metrics.items()},
        "verdict": verdict,
        "ga_config": {
            "pop": cfg.pop, "gens": cfg.gens, "max_active": cfg.max_active,
            "elite": cfg.elite, "mut_rate": cfg.mut_rate, "seed": cfg.seed,
        },
        "feature_families_used": ["fund", "flow", "factor", "theme", "intraday"],
        "regime_excluded": True,
        "hc": ["#561", "#559"],
    }

    out_path = OUT / f"{sector.replace(' ', '_').lower()}_v1.json"
    with open(out_path, "w") as f:
        json.dump(result, f, indent=2, default=str)
    log.info(f"  wrote {out_path}")
    return result


# ---------------------------------------------------------------------------
# main: smoke test on 2 sectors with most names
# ---------------------------------------------------------------------------
def main(smoke: bool = True):
    log.info("loading features + targets + universe")
    features = load_features()
    fwd = load_targets()
    sectors = get_universe_sectors()
    log.info(f"features shape: {features.shape}  fwd shape: {fwd.shape}")

    counts = sectors["sector"].value_counts()
    log.info(f"sector counts: {counts.to_dict()}")

    if smoke:
        target_sectors = counts.head(2).index.tolist()
    else:
        target_sectors = counts.index.tolist()
    log.info(f"target_sectors: {target_sectors}")

    cfg = GAConfig(pop=60, gens=20, max_active=8, elite=8, mut_rate=0.3, seed=42)

    results = {}
    for s in target_sectors:
        try:
            results[s] = fit_sector(s, features, fwd, sectors, cfg)
        except Exception as e:
            log.exception(f"sector {s} failed: {e}")
            results[s] = {"error": str(e)}

    # print summary
    print("\n=== GA SMOKE TEST RESULTS ===")
    for s, r in results.items():
        print(f"\n--- {s} (n={r.get('n_tickers', '?')} names) ---")
        if "error" in r:
            print(f"  ERROR: {r['error']}")
            continue
        print(f"  verdict: {r['verdict']}")
        m = r.get("full_sample_metrics", {})
        print(f"  full-sample: sharpe={m.get('sharpe')}  cagr={m.get('cagr')}  calmar={m.get('calmar')}  mdd={m.get('max_dd')}")
        w = r.get("weights", {})
        top3 = list(w.items())[:3]
        print(f"  top-3 features:")
        for f, val in top3:
            print(f"    {f:40s}  {val:+.3f}")
    return results


if __name__ == "__main__":
    main(smoke=True)
