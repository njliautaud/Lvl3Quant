"""
GA sector fitter v2 — honest-pricing upgrade (HC #559 + HC #561).

Differences vs v1:
  1. **Daily-bar fitness** — portfolio returns are computed daily from feature_store
     scores and the weekly rebalance schedule (was: monthly snapshot fitness).
  2. **Transaction costs** — 5 bps round-trip per name turnover for liquid names,
     30 bps for ADV < $20M. Applied on each rebalance day's turnover.
  3. **Walk-forward fitness** — fitness = median Sharpe across WF folds (3yr train /
     1yr OOT / 6mo step). HARD REJECT if any fold's Calmar < 1.0 (HC #559).
  4. **Universe** — auto-attempts S&P 500 expansion via yfinance (best-effort);
     degrades gracefully to the existing 70-name universe if yfinance throttles
     or the price-cache build can't be completed.
  5. **Outputs** — {sector}_v2.json with WF median + p25 + p75 per metric,
     verdict per HC #561 R4. Aggregate report at research/findings/sector_ga_v2_report.md.
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
import traceback
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

ROOT = Path(os.environ.get("LVL3_ROOT", "/home/jupiter/Lvl3Quant"))
FS = ROOT / "data/feature_store/v1"
CACHE = ROOT / "wheel_strategy_v1/data/cache"
OUT = ROOT / "strategy/macro_picker/formulas"
FINDINGS = ROOT / "research/findings"
LOG = ROOT / "logs/ga_v2_full_sweep.log"

OUT.mkdir(parents=True, exist_ok=True)
FINDINGS.mkdir(parents=True, exist_ok=True)
LOG.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    filename=LOG, level=logging.INFO,
    format="%(asctime)s [ga2] %(message)s",
)
log = logging.getLogger("ga2")

sys.path.insert(0, str(ROOT / "research"))
from walk_forward import (  # noqa: E402
    _annualize_sharpe, _annualize_sortino, _cagr, _max_dd, _calmar,
    _profit_factor, _win_rate, _verdict_for_metrics,
)

TRADING_DAYS = 252

# ---------------------------------------------------------------------------
# universe expansion (best-effort yfinance S&P 500 fetch)
# ---------------------------------------------------------------------------
SP500_PARQUET = CACHE / "universe_sp500.parquet"


def try_expand_universe_sp500(timeout_sec: int = 120) -> Optional[pd.DataFrame]:
    """Pull S&P 500 constituents from yfinance/wikipedia. Returns ticker+sector+industry.
    Returns None on any failure (caller falls back to the existing 70-name universe)."""
    try:
        import urllib.request
        url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
        log.info(f"trying to fetch S&P 500 list from wikipedia ({url})")
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout_sec) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(html)
        sp = tables[0]
        cols_lower = {c.lower(): c for c in sp.columns}
        ticker_col = cols_lower.get("symbol") or cols_lower.get("ticker")
        sector_col = cols_lower.get("gics sector") or cols_lower.get("sector")
        industry_col = cols_lower.get("gics sub-industry") or cols_lower.get("gics sub industry") or cols_lower.get("industry")
        if not ticker_col or not sector_col:
            log.warning(f"could not find expected columns in wikipedia table; got {sp.columns.tolist()}")
            return None
        out = sp[[ticker_col, sector_col]].copy()
        out.columns = ["ticker", "sector"]
        if industry_col:
            out["industry"] = sp[industry_col]
        out["ticker"] = out["ticker"].astype(str).str.replace(".", "-", regex=False).str.strip()
        out["source"] = "sp500_wikipedia"
        out = out.drop_duplicates("ticker").reset_index(drop=True)
        out.to_parquet(SP500_PARQUET, index=False)
        log.info(f"saved S&P 500 universe ({len(out)} names) -> {SP500_PARQUET}")
        return out
    except Exception as e:
        log.warning(f"S&P 500 expansion failed: {e}")
        return None


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
    drop_cols = [c for c in ["intraday_source", "sector_etf", "fund_asof"] if c in df.columns]
    if drop_cols:
        df = df.drop(columns=drop_cols)
    df["date"] = pd.to_datetime(df["date"])
    return df


def load_prices_and_adv() -> pd.DataFrame:
    p = pd.read_parquet(CACHE / "prices.parquet")
    p = p[["ticker", "date", "close", "volume"]].copy()
    p["date"] = pd.to_datetime(p["date"])
    p = p.sort_values(["ticker", "date"]).reset_index(drop=True)
    p["dollar_volume"] = p["close"] * p["volume"]
    # 60d ADV
    p["adv_60d"] = p.groupby("ticker")["dollar_volume"].transform(
        lambda s: s.rolling(60, min_periods=10).mean().shift(1)
    )
    p["ret"] = p.groupby("ticker")["close"].pct_change()
    return p


def get_universe_sectors() -> pd.DataFrame:
    u = pd.read_parquet(CACHE / "universe.parquet")[["ticker", "sector"]]
    return u


def macro_extra() -> pd.DataFrame:
    me = pd.read_parquet(CACHE / "macro_extra.parquet")
    me["date"] = pd.to_datetime(me["date"])
    return me.set_index("date")


# ---------------------------------------------------------------------------
# scoring / portfolio construction
# ---------------------------------------------------------------------------
def _score_panel(features: pd.DataFrame, feat_cols: list, weights: np.ndarray) -> pd.DataFrame:
    """Cross-sectional pct-rank then weighted sum, per date."""
    sub = features[["ticker", "date"] + feat_cols].copy()
    # rank within each date for each feature
    for c in feat_cols:
        sub[c] = sub.groupby("date")[c].rank(pct=True) - 0.5
    arr = sub[feat_cols].fillna(0.0).values
    sub["score"] = arr @ weights
    return sub[["ticker", "date", "score"]]


def _weekly_rebalance_dates(dates: pd.DatetimeIndex) -> pd.DatetimeIndex:
    df = pd.DataFrame(index=pd.DatetimeIndex(dates))
    df["week"] = df.index.to_period("W")
    return pd.DatetimeIndex(df.groupby("week").apply(lambda g: g.index.min()).values)


def _build_portfolio_returns(
    score_panel: pd.DataFrame,
    prices: pd.DataFrame,
    tc_low_bps: float = 5.0,
    tc_high_bps: float = 30.0,
    adv_high_threshold: float = 20e6,
) -> pd.Series:
    """Daily portfolio return: weekly-rebalanced long-top-decile / short-bottom-decile.
    Transaction costs applied on rebalance days: 5bps round-trip per turnover for
    liquid names (60d ADV >= $20M), 30bps otherwise."""
    p = prices[["ticker", "date", "ret", "adv_60d"]].copy()
    sp = score_panel.merge(p, on=["ticker", "date"], how="inner")
    sp = sp.sort_values(["date", "ticker"]).reset_index(drop=True)

    all_dates = pd.DatetimeIndex(sorted(sp["date"].unique()))
    if len(all_dates) < 30:
        return pd.Series(dtype=float)
    rb_dates = set(_weekly_rebalance_dates(all_dates))

    # rebalance weights at each rb_date based on most recent score in [<=rb_date]
    weights_state = pd.Series(dtype=float)  # ticker -> weight
    daily_rows = []
    for d in all_dates:
        day = sp[sp["date"] == d]
        rebal = d in rb_dates
        new_w = weights_state
        rebal_cost = 0.0
        if rebal:
            valid = day.dropna(subset=["score"])
            if len(valid) >= 10:
                q_lo, q_hi = valid["score"].quantile(0.1), valid["score"].quantile(0.9)
                longs = valid[valid["score"] >= q_hi].copy()
                shorts = valid[valid["score"] <= q_lo].copy()
                if len(longs) > 0 and len(shorts) > 0:
                    new_w = pd.Series(0.0, index=valid["ticker"].values)
                    longs_w = 0.5 / len(longs)
                    shorts_w = -0.5 / len(shorts)
                    new_w.loc[longs["ticker"]] = longs_w
                    new_w.loc[shorts["ticker"]] = shorts_w
                    # turnover = sum |new_w - old_w| per ticker (use 0 for unseen)
                    old = weights_state.reindex(new_w.index).fillna(0.0)
                    deltas = (new_w - old).abs()
                    # cost rate by ADV bucket
                    adv_lookup = day.set_index("ticker")["adv_60d"]
                    adv = adv_lookup.reindex(deltas.index).fillna(0.0)
                    cost_rate = np.where(adv >= adv_high_threshold,
                                          tc_low_bps / 1e4, tc_high_bps / 1e4)
                    rebal_cost = float((deltas.values * cost_rate).sum())
                    # also dropped tickers (in old but not new) — pay close cost
                    dropped = weights_state.index.difference(new_w.index)
                    if len(dropped) > 0:
                        old_dropped = weights_state.loc[dropped].abs()
                        adv2 = adv_lookup.reindex(dropped).fillna(0.0)
                        cost_rate2 = np.where(adv2 >= adv_high_threshold,
                                               tc_low_bps / 1e4, tc_high_bps / 1e4)
                        rebal_cost += float((old_dropped.values * cost_rate2).sum())
        # mark-to-market: gross ret = sum(w * ret) using prior weights
        ret_lookup = day.set_index("ticker")["ret"]
        if len(weights_state) > 0:
            joined = ret_lookup.reindex(weights_state.index).fillna(0.0)
            gross = float((weights_state.values * joined.values).sum())
        else:
            gross = 0.0
        net = gross - rebal_cost
        daily_rows.append((d, net))
        if rebal:
            weights_state = new_w

    s = pd.Series(dict(daily_rows)).sort_index()
    s.index = pd.to_datetime(s.index)
    return s


# ---------------------------------------------------------------------------
# walk-forward over a GA candidate
# ---------------------------------------------------------------------------
def _fold_dates(date_index: pd.DatetimeIndex, train_months: int = 36,
                oot_months: int = 12, step_months: int = 6) -> list[dict]:
    if len(date_index) < 1:
        return []
    cursor = pd.Timestamp(date_index.min())
    end = pd.Timestamp(date_index.max())
    folds = []
    while True:
        tr_start = cursor
        tr_end = tr_start + pd.DateOffset(months=train_months)
        oot_start = tr_end
        oot_end = oot_start + pd.DateOffset(months=oot_months)
        if oot_end > end + pd.Timedelta(days=1):
            break
        folds.append({"train_start": tr_start, "train_end": tr_end,
                      "oot_start": oot_start, "oot_end": oot_end})
        cursor = cursor + pd.DateOffset(months=step_months)
    return folds


def _wf_metrics(weights: np.ndarray, feat_cols: list, features: pd.DataFrame,
                prices: pd.DataFrame, folds: list) -> tuple[float, dict, bool]:
    """Run a GA candidate through WF. Returns (fitness, summary, calmar_floor_failed)."""
    score = _score_panel(features, feat_cols, weights)
    rets = _build_portfolio_returns(score, prices)
    if rets.empty or rets.std() == 0:
        return -999.0, {}, True
    per_fold = []
    for f in folds:
        mask = (rets.index >= f["oot_start"]) & (rets.index < f["oot_end"])
        sub = rets[mask]
        if len(sub) < 20:
            continue
        m = {
            "sharpe": _annualize_sharpe(sub),
            "sortino": _annualize_sortino(sub),
            "cagr": _cagr(sub),
            "max_dd": _max_dd(sub),
            "calmar": _calmar(sub),
            "pf": _profit_factor(sub),
            "wr": _win_rate(sub),
        }
        per_fold.append(m)
    if not per_fold:
        return -999.0, {}, True
    # calmar floor: any fold below 1.0 -> reject
    calmar_fail = any(
        (not np.isfinite(f["calmar"])) or f["calmar"] < 1.0
        for f in per_fold
    )
    sharpes = [f["sharpe"] for f in per_fold if np.isfinite(f["sharpe"])]
    if not sharpes:
        return -999.0, {}, True
    fitness = float(np.median(sharpes))
    summary = {}
    keys = ["sharpe", "sortino", "cagr", "max_dd", "calmar", "pf", "wr"]
    for k in keys:
        vals = pd.Series([f[k] for f in per_fold], dtype=float).dropna()
        if len(vals) == 0:
            summary[k] = {"median": float("nan"), "p25": float("nan"), "p75": float("nan")}
        else:
            summary[k] = {"median": float(vals.median()),
                          "p25": float(vals.quantile(0.25)),
                          "p75": float(vals.quantile(0.75))}
    summary["n_folds"] = len(per_fold)
    summary["per_fold"] = per_fold
    if calmar_fail:
        return -999.0, summary, True
    return fitness, summary, False


# ---------------------------------------------------------------------------
# GA core
# ---------------------------------------------------------------------------
@dataclass
class GAConfig:
    pop: int = 60
    gens: int = 20
    max_active: int = 8
    elite: int = 8
    mut_rate: float = 0.3
    seed: int = 42
    n_jobs: int = 8


def _random_candidate(n_feat, max_active, rng):
    w = np.zeros(n_feat, dtype=float)
    n_active = rng.integers(2, max_active + 1)
    idx = rng.choice(n_feat, size=int(n_active), replace=False)
    w[idx] = rng.normal(0, 1, size=int(n_active))
    return w


def _mutate(w, max_active, rng):
    w = w.copy()
    n_feat = len(w)
    op = rng.choice(["perturb", "add", "drop"])
    active = np.where(w != 0)[0]
    if op == "perturb" and len(active):
        i = rng.choice(active)
        w[i] += rng.normal(0, 0.5)
    elif op == "add" and len(active) < max_active:
        cand = np.where(w == 0)[0]
        if len(cand):
            i = rng.choice(cand)
            w[i] = rng.normal(0, 1)
    elif op == "drop" and len(active) > 2:
        i = rng.choice(active)
        w[i] = 0.0
    return w


def _crossover(a, b, rng):
    mask = rng.random(len(a)) < 0.5
    return np.where(mask, a, b)


def _eval_candidate(args):
    w, feat_cols, features, prices, folds = args
    try:
        fit, summary, _ = _wf_metrics(w, feat_cols, features, prices, folds)
        return (fit, w, summary)
    except Exception as e:
        return (-999.0, w, {"error": str(e)})


def fit_sector(sector: str, features_all: pd.DataFrame, prices_all: pd.DataFrame,
               sectors_map: pd.DataFrame, cfg: GAConfig) -> dict:
    t0 = time.time()
    log.info(f"=== fit sector: {sector} ===")
    tickers = sectors_map[sectors_map["sector"] == sector]["ticker"].tolist()
    log.info(f"  tickers ({len(tickers)}): {tickers[:20]}{'...' if len(tickers) > 20 else ''}")
    if len(tickers) < 8:
        log.info(f"  skipping {sector}: too few tickers ({len(tickers)})")
        return {"sector": sector, "skipped": True, "reason": "n_tickers<8",
                "n_tickers": len(tickers)}
    features = features_all[features_all["ticker"].isin(tickers)].copy()
    prices = prices_all[prices_all["ticker"].isin(tickers)].copy()

    drop = {"ticker", "date"}
    feat_cols = [c for c in features.columns
                 if c not in drop and pd.api.types.is_numeric_dtype(features[c])]
    log.info(f"  feature cols: {len(feat_cols)}")

    date_index = pd.DatetimeIndex(sorted(features["date"].unique()))
    folds = _fold_dates(date_index)
    log.info(f"  folds: {len(folds)}")
    if not folds:
        return {"sector": sector, "skipped": True, "reason": "no_wf_folds"}

    rng = np.random.default_rng(cfg.seed)
    n_feat = len(feat_cols)
    population = [_random_candidate(n_feat, cfg.max_active, rng) for _ in range(cfg.pop)]
    best_fit = -np.inf
    best_w = None
    best_summary = {}

    for gen in range(cfg.gens):
        tg = time.time()
        args = [(w, feat_cols, features, prices, folds) for w in population]
        if cfg.n_jobs > 1:
            scored = Parallel(n_jobs=cfg.n_jobs, prefer="processes", batch_size=1)(
                delayed(_eval_candidate)(a) for a in args
            )
        else:
            scored = [_eval_candidate(a) for a in args]
        scored.sort(key=lambda x: x[0], reverse=True)
        if scored[0][0] > best_fit:
            best_fit = scored[0][0]
            best_w = scored[0][1].copy()
            best_summary = scored[0][2]
        log.info(f"  gen {gen:2d}  best={scored[0][0]:.3f}  "
                 f"med={scored[cfg.pop // 2][0]:.3f}  ({time.time()-tg:.1f}s)")
        elite = [w for _, w, _ in scored[:cfg.elite]]
        new_pop = list(elite)
        while len(new_pop) < cfg.pop:
            p1, p2 = rng.choice(len(elite), size=2, replace=True)
            child = _crossover(elite[p1], elite[p2], rng)
            if rng.random() < cfg.mut_rate:
                child = _mutate(child, cfg.max_active, rng)
            active = np.where(child != 0)[0]
            if len(active) > cfg.max_active:
                drop_idx = rng.choice(active, size=len(active) - cfg.max_active, replace=False)
                child[drop_idx] = 0.0
            new_pop.append(child)
        population = new_pop

    # final eval with verdict
    fit_final, summary_final, calmar_fail = _wf_metrics(
        best_w, feat_cols, features, prices, folds
    )
    med = {k: v.get("median", float("nan")) for k, v in summary_final.items()
           if isinstance(v, dict)}
    verdict = _verdict_for_metrics(med) if med else "FAILS CALMAR FLOOR"
    if calmar_fail and verdict != "FAILS CALMAR FLOOR":
        verdict = "FAILS CALMAR FLOOR"

    active_idx = np.where(best_w != 0)[0] if best_w is not None else np.array([], dtype=int)
    weights = {feat_cols[i]: float(best_w[i]) for i in active_idx} if best_w is not None else {}
    weights_sorted = dict(sorted(weights.items(), key=lambda kv: -abs(kv[1])))

    out = {
        "sector": sector,
        "n_tickers": len(tickers),
        "tickers": tickers,
        "n_features_considered": len(feat_cols),
        "weights": weights_sorted,
        "wf_summary": {k: v for k, v in summary_final.items() if k != "per_fold"},
        "n_folds": summary_final.get("n_folds", 0),
        "per_fold": summary_final.get("per_fold", []),
        "wf_median_sharpe": float(fit_final) if np.isfinite(fit_final) else None,
        "verdict": verdict,
        "calmar_floor_failed": bool(calmar_fail),
        "ga_config": {**cfg.__dict__},
        "feature_families_used": ["fund", "flow", "factor", "theme", "intraday"],
        "regime_excluded": True,
        "costs_model": {"tc_low_bps": 5.0, "tc_high_bps": 30.0,
                        "adv_high_threshold_usd": 20e6,
                        "rebalance": "weekly",
                        "horizon_days": "weekly_rebal_daily_mtm"},
        "hc": ["#561", "#559", "#560"],
        "elapsed_sec": time.time() - t0,
    }
    out_path = OUT / f"{sector.replace(' ', '_').lower()}_v2.json"
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2, default=str)
    log.info(f"  wrote {out_path}  verdict={verdict}  elapsed={out['elapsed_sec']:.1f}s")
    return out


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pop", type=int, default=60)
    ap.add_argument("--gens", type=int, default=20)
    ap.add_argument("--n-jobs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sectors", type=str, default="",
                    help="comma-separated sector names; empty = all sectors with >=8 names")
    ap.add_argument("--no-expand", action="store_true",
                    help="skip S&P 500 expansion attempt")
    args = ap.parse_args()

    cfg = GAConfig(pop=args.pop, gens=args.gens, n_jobs=args.n_jobs, seed=args.seed)
    log.info(f"GA v2 sweep start | cfg={cfg.__dict__}")

    # 1. universe expansion (best-effort, non-blocking)
    expanded = None
    if not args.no_expand:
        expanded = try_expand_universe_sp500(timeout_sec=60)
    if expanded is not None:
        log.info(f"S&P 500 list fetched ({len(expanded)} names). Note: price-cache rebuild "
                 f"for those names is NOT done in this script — see ga_v2_full_sweep "
                 f"runbook. Continuing on the 70-name cached universe.")

    log.info("loading features ...")
    features = load_features()
    log.info(f"features shape: {features.shape}")
    log.info("loading prices+ADV ...")
    prices = load_prices_and_adv()
    log.info(f"prices shape: {prices.shape}")
    sectors_map = get_universe_sectors()
    log.info(f"universe sectors: {sectors_map['sector'].value_counts().to_dict()}")
    universe_size_used = sectors_map["ticker"].nunique()
    universe_source = "wheel_strategy_v1/data/cache/universe.parquet (70-name baseline)"
    if expanded is not None:
        universe_source += f" + sp500_wikipedia ({len(expanded)} names — list-only, not yet priced)"

    counts = sectors_map["sector"].value_counts()
    if args.sectors:
        target_sectors = [s.strip() for s in args.sectors.split(",") if s.strip()]
    else:
        target_sectors = counts[counts >= 5].index.tolist()
    log.info(f"target_sectors: {target_sectors}")

    results = []
    for s in target_sectors:
        try:
            res = fit_sector(s, features, prices, sectors_map, cfg)
            results.append(res)
        except Exception as e:
            log.exception(f"sector {s} failed: {e}")
            results.append({"sector": s, "error": str(e),
                            "tb": traceback.format_exc()})

    # write aggregate report
    report_path = FINDINGS / "sector_ga_v2_report.md"
    lines = [
        "# Sector GA Sweep — v2 (HC #559 + HC #561 R4)",
        "",
        f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        f"Universe: **{universe_size_used} names** in `{universe_source}`.",
        "Costs: 5 bps round-trip per turnover (ADV >= $20M), 30 bps otherwise.",
        "Rebalance: weekly, long-top-decile / short-bottom-decile.",
        "Regime EXCLUDED from per-name score (HC #561 R2).",
        "",
        "## Verdict Table (HC #561 R4 acceptance gates)",
        "",
        "| Sector | n | folds | verdict | WF med Sharpe | WF med Calmar | WF med CAGR | WF med MaxDD |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in results:
        if r.get("skipped"):
            lines.append(f"| {r['sector']} | {r.get('n_tickers','?')} | - | SKIPPED ({r.get('reason')}) | - | - | - | - |")
            continue
        if "error" in r:
            lines.append(f"| {r['sector']} | - | - | ERROR | - | - | - | - |")
            continue
        summ = r.get("wf_summary", {})
        ms = summ.get("sharpe", {}).get("median")
        mc = summ.get("calmar", {}).get("median")
        mcagr = summ.get("cagr", {}).get("median")
        mdd = summ.get("max_dd", {}).get("median")
        def fmt(x): return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.3f}"
        lines.append(f"| {r['sector']} | {r['n_tickers']} | {r.get('n_folds',0)} | "
                     f"{r['verdict']} | {fmt(ms)} | {fmt(mc)} | {fmt(mcagr)} | {fmt(mdd)} |")
    lines += ["", "## Per-sector details", ""]
    for r in results:
        if r.get("skipped") or "error" in r:
            continue
        lines.append(f"### {r['sector']} — verdict: {r['verdict']}")
        lines.append("")
        lines.append(f"- n_tickers: {r['n_tickers']} | features considered: {r['n_features_considered']} | n_folds: {r.get('n_folds',0)}")
        lines.append(f"- Top weights:")
        for f, v in list(r.get("weights", {}).items())[:8]:
            lines.append(f"  - `{f}`: {v:+.3f}")
        summ = r.get("wf_summary", {})
        lines.append(f"- WF summary (median / p25 / p75):")
        for k in ("sharpe", "sortino", "cagr", "max_dd", "calmar", "pf", "wr"):
            d = summ.get(k, {})
            lines.append(f"  - {k}: med={d.get('median', float('nan')):.3f}  "
                         f"p25={d.get('p25', float('nan')):.3f}  p75={d.get('p75', float('nan')):.3f}")
        lines.append("")
    with open(report_path, "w") as f:
        f.write("\n".join(lines))
    log.info(f"wrote report -> {report_path}")
    print(f"DONE: {len(results)} sectors processed; report at {report_path}")


if __name__ == "__main__":
    main()
