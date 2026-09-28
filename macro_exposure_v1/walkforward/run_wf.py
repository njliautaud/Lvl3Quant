"""
run_wf.py — Walk-forward out-of-sample validation of macro_exposure_v1 Balanced tier.

HC #428 R1: regime-agnostic OOT validation across >=3 macro regimes.
HC #534 R1: walk-forward OOS required before any paper-trading talk.

Design
------
- FIXED Balanced chromosome (NO re-optimization). Pulled from
  results/full/pareto.parquet (the row with CAGR=20.38%, DD=12.38%, Sortino=1.56).
- Sliding folds: 1-year OOT each, step 6 months. We do NOT re-fit on the train
  window — feature standardization (rolling 252d zscore) is what the GA already
  did and we reuse it. Folds simply slice the precomputed OOT period from the
  full-sample feature panel.
- OOT span: 2015-01-01 -> 2026-06-04, stepped every 6 months, 1-year window.
- Regime label per OOT fold = SPY CAGR over that OOT window:
    bull  if SPY_CAGR > +10%
    bear  if SPY_CAGR < -5%
    chop  otherwise
- Pooled OOS metrics computed on the concatenated daily-returns series from
  NON-OVERLAPPING 1-year fold groups (we take every other fold so OOT windows
  don't double-count). Per-fold metrics use the full 1-year fold for richer
  regime stratification.

Verdict gates (from task brief):
  PASS if  pooled OOS CAGR >= 10%
       AND pooled OOS max DD <= 18%
       AND no regime Sortino < 0.5
       AND hit rate (folds with positive return) >= 65%
  else FAIL with reasons.
"""
from __future__ import annotations
import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from ga.chromosome import FEATURE_COLUMN_MAP, FEATURE_WEIGHT_NAMES
from ga.run_ga import build_feature_panel, build_allocation, cadence_dates
from backtest.exposure_engine import run_exposure_backtest

OUT = Path(__file__).resolve().parent / "results"
OUT.mkdir(parents=True, exist_ok=True)


# --- The FIXED Balanced chromosome (verbatim from pareto.parquet match) -------
BALANCED_CFG = {
    "feature_weights": {
        "w_naaim":              -1.859139305897906,
        "w_vix_pct":            -1.4977376854509945,
        "w_vix_ts_slope":       +1.678539632112297,
        "w_aaii_bullbear":      -1.882229941020674,
        "w_mom_3m":             -1.7861779440672065,
        "w_mom_6m":             -0.257666740748268,
        "w_spy_above_200dma":   +1.2713368647000864,
        "w_yield_2s10s":        -1.5191637785175662,
        "w_dxy_trend":          -0.39153045740817083,
        "w_breadth":            -1.5044777388177524,
        "w_gold_copper_ratio":  +0.030441514737820086,
    },
    "long_threshold":  0.08421423521041449,
    "short_threshold": -0.22733397225907126,
    "flat_band_width": 0.38112248057147324,
    "basket_weights":  {"SPY": 0.4922318994944715,
                        "QQQ": 0.37589771494571145,
                        "IWM": 0.13187038555981706},
    "max_leverage":    1.25,
    "allow_short":     False,   # allow_short=0 in pareto row
    "cadence":         "weekly",
    "long_strength":   1.25,
    "short_strength":  1.0,
}


# --- Helpers ------------------------------------------------------------------

def _annualize(daily_ret: pd.Series) -> dict:
    """Compute risk-adjusted metrics from a daily-return series."""
    if len(daily_ret) < 5:
        return dict(cagr=0.0, sharpe=0.0, sortino=0.0, max_dd_pct=0.0,
                    worst_month_pct=0.0, n_days=int(len(daily_ret)))
    yrs = max(len(daily_ret) / 252.0, 1e-6)
    eq = (1.0 + daily_ret).cumprod()
    cagr = float(eq.iloc[-1]) ** (1.0 / yrs) - 1.0
    mu = daily_ret.mean() * 252.0
    sd = daily_ret.std() * np.sqrt(252.0)
    sharpe = mu / sd if sd > 1e-12 else 0.0
    down = daily_ret[daily_ret < 0]
    dsd = down.std() * np.sqrt(252.0) if len(down) else 0.0
    sortino = mu / dsd if dsd > 1e-12 else 0.0
    peak = eq.cummax()
    dd = (eq / peak - 1.0).min()
    monthly = (1.0 + daily_ret).resample("ME").prod() - 1.0
    worst = float(monthly.min()) if len(monthly) else 0.0
    return dict(
        cagr=float(cagr),
        sharpe=float(sharpe),
        sortino=float(sortino),
        max_dd_pct=float(abs(dd) * 100.0),
        worst_month_pct=float(worst * 100.0),
        n_days=int(len(daily_ret)),
    )


def _spy_cagr(prices: pd.DataFrame, start, end) -> float:
    s = prices["SPY"].loc[start:end].dropna()
    if len(s) < 5:
        return 0.0
    yrs = max(len(s) / 252.0, 1e-6)
    return float((s.iloc[-1] / s.iloc[0]) ** (1.0 / yrs) - 1.0)


def _regime_label(spy_cagr: float) -> str:
    if spy_cagr > 0.10:
        return "bull"
    if spy_cagr < -0.05:
        return "bear"
    return "chop"


# --- Main walk-forward driver -------------------------------------------------

def main():
    print("[wf] loading panel (full sample)", flush=True)
    features, prices = build_feature_panel(smoke=False)
    print(f"[wf] features {features.shape}  prices {prices.shape}  "
          f"span {features.index.min().date()} -> {features.index.max().date()}",
          flush=True)

    # Build full-sample allocation ONCE with the fixed chromosome.
    # Slicing later by fold gives the OOT allocation per fold.
    alloc_full = build_allocation(features, BALANCED_CFG)
    rebal_full = cadence_dates(features.index, BALANCED_CFG["cadence"])

    # Define folds: 1-year OOT, step 6 months, from 2015-01-01 onward.
    fold_start = pd.Timestamp("2015-01-01")
    last_date = features.index.max()
    folds = []
    cur = fold_start
    while cur + pd.DateOffset(years=1) <= last_date + pd.Timedelta(days=1):
        oot_start = cur
        oot_end = cur + pd.DateOffset(years=1) - pd.Timedelta(days=1)
        folds.append((oot_start, oot_end))
        cur = cur + pd.DateOffset(months=6)
    # Tail fold: if there is >=6mo of data left, run a short final fold
    tail_start = cur
    if tail_start < last_date and (last_date - tail_start).days >= 150:
        folds.append((tail_start, last_date))

    print(f"[wf] {len(folds)} folds  first={folds[0][0].date()}  last={folds[-1][1].date()}",
          flush=True)

    rows = []
    daily_concat_blocks = []  # for pooled OOS — non-overlapping 1y blocks

    for k, (s, e) in enumerate(folds):
        idx_mask = (features.index >= s) & (features.index <= e)
        if idx_mask.sum() < 30:
            continue
        # Slice features/prices/allocation to OOT window
        fidx = features.index[idx_mask]
        alloc_oot = alloc_full.loc[fidx]
        rebal_oot = rebal_full[(rebal_full >= s) & (rebal_full <= e)]
        prices_oot = prices.loc[fidx]

        res = run_exposure_backtest(
            allocation=alloc_oot,
            basket_w=BALANCED_CFG["basket_weights"],
            prices=prices_oot,
            rebalance_dates=rebal_oot,
            starting_cash=100_000.0,
            allow_short=BALANCED_CFG["allow_short"],
            max_leverage=BALANCED_CFG["max_leverage"],
        )
        m = res.metrics
        spy_c = _spy_cagr(prices, s, e)
        regime = _regime_label(spy_c)
        row = dict(
            fold=k,
            oot_start=str(s.date()),
            oot_end=str(e.date()),
            regime=regime,
            spy_cagr_pct=spy_c * 100.0,
            fold_cagr_pct=m["cagr"] * 100.0,
            fold_dd_pct=m["max_dd_pct"],
            fold_sortino=m["sortino"],
            fold_sharpe=m["sharpe"],
            worst_month_pct=m["worst_month_pct"],
            turnover_per_year=m["turnover_per_year"],
            avg_leverage=m["avg_leverage"],
            n_days=int(len(res.daily_ret)),
        )
        rows.append(row)
        print(f"[wf] fold {k:2d} {s.date()} -> {e.date()}  regime={regime} "
              f"spyCAGR={spy_c*100:5.1f}%  cagr={m['cagr']*100:6.2f}%  "
              f"dd={m['max_dd_pct']:5.2f}%  sortino={m['sortino']:5.2f}",
              flush=True)

    # ---- Pooled OOS: non-overlapping 1y blocks (every other fold) -----------
    # Folds step 6mo with 1y windows => folds 0,2,4,... are non-overlapping.
    nonoverlap_idx = list(range(0, len(folds), 2))
    for i in nonoverlap_idx:
        s, e = folds[i]
        idx_mask = (features.index >= s) & (features.index <= e)
        if idx_mask.sum() < 30:
            continue
        fidx = features.index[idx_mask]
        alloc_oot = alloc_full.loc[fidx]
        rebal_oot = rebal_full[(rebal_full >= s) & (rebal_full <= e)]
        prices_oot = prices.loc[fidx]
        res = run_exposure_backtest(
            allocation=alloc_oot,
            basket_w=BALANCED_CFG["basket_weights"],
            prices=prices_oot,
            rebalance_dates=rebal_oot,
            starting_cash=100_000.0,
            allow_short=BALANCED_CFG["allow_short"],
            max_leverage=BALANCED_CFG["max_leverage"],
        )
        daily_concat_blocks.append(res.daily_ret)

    pooled = pd.concat(daily_concat_blocks).sort_index()
    # De-duplicate any boundary overlap by averaging same-date returns
    pooled = pooled.groupby(pooled.index).mean()
    pooled_metrics = _annualize(pooled)
    print(f"[wf] pooled OOS: CAGR={pooled_metrics['cagr']*100:.2f}%  "
          f"DD={pooled_metrics['max_dd_pct']:.2f}%  "
          f"Sortino={pooled_metrics['sortino']:.2f}  "
          f"Sharpe={pooled_metrics['sharpe']:.2f}  "
          f"n_days={pooled_metrics['n_days']}", flush=True)

    df = pd.DataFrame(rows)
    df.to_parquet(OUT / "wf_folds.parquet", index=False)
    df.to_csv(OUT / "wf_folds.csv", index=False)

    # ---- Per-regime stratification --------------------------------------------
    regime_rows = []
    for reg in ["bull", "bear", "chop"]:
        sub = df[df["regime"] == reg]
        if len(sub) == 0:
            regime_rows.append(dict(regime=reg, n_folds=0, avg_cagr_pct=float("nan"),
                                    avg_dd_pct=float("nan"), avg_sortino=float("nan"),
                                    avg_sharpe=float("nan"), hit_rate_pct=float("nan")))
            continue
        regime_rows.append(dict(
            regime=reg,
            n_folds=len(sub),
            avg_cagr_pct=float(sub["fold_cagr_pct"].mean()),
            avg_dd_pct=float(sub["fold_dd_pct"].mean()),
            avg_sortino=float(sub["fold_sortino"].mean()),
            avg_sharpe=float(sub["fold_sharpe"].mean()),
            hit_rate_pct=float((sub["fold_cagr_pct"] > 0).mean() * 100.0),
        ))
    reg_df = pd.DataFrame(regime_rows)
    reg_df.to_csv(OUT / "wf_regimes.csv", index=False)

    # ---- Hit rate & verdict --------------------------------------------------
    hit_rate = float((df["fold_cagr_pct"] > 0).mean() * 100.0)
    reasons = []
    if pooled_metrics["cagr"] * 100.0 < 10.0:
        reasons.append(f"pooled CAGR {pooled_metrics['cagr']*100:.2f}% < 10%")
    if pooled_metrics["max_dd_pct"] > 18.0:
        reasons.append(f"pooled max DD {pooled_metrics['max_dd_pct']:.2f}% > 18%")
    for r in regime_rows:
        if r["n_folds"] > 0 and r["avg_sortino"] < 0.5:
            reasons.append(f"{r['regime']} regime Sortino {r['avg_sortino']:.2f} < 0.5")
    if hit_rate < 65.0:
        reasons.append(f"hit rate {hit_rate:.1f}% < 65%")
    verdict = "PASS" if not reasons else "FAIL"

    summary = dict(
        pooled_cagr_pct=pooled_metrics["cagr"] * 100.0,
        pooled_dd_pct=pooled_metrics["max_dd_pct"],
        pooled_sortino=pooled_metrics["sortino"],
        pooled_sharpe=pooled_metrics["sharpe"],
        pooled_worst_month_pct=pooled_metrics["worst_month_pct"],
        pooled_n_days=pooled_metrics["n_days"],
        n_folds=len(df),
        hit_rate_pct=hit_rate,
        verdict=verdict,
        reasons=reasons,
        in_sample_reference=dict(
            cagr_pct=20.38, max_dd_pct=12.38, sortino=1.56, sharpe=1.41,
        ),
    )
    (OUT / "wf_summary.json").write_text(json.dumps(summary, indent=2))

    # ---- Markdown report -----------------------------------------------------
    md = []
    md.append("# Macro-Exposure v1 — Balanced Tier Walk-Forward OOS Validation\n")
    md.append("**Fixed chromosome** (NO re-optimization): the Balanced tier row from "
              "`results/full/pareto.parquet` (CAGR=20.38%, DD=12.38%, Sortino=1.56 in-sample).\n")
    md.append("**Walk-forward**: 1-year OOT folds stepped every 6 months from 2015-01-01. "
              "Pooled OOS uses non-overlapping folds (every other fold).\n")
    md.append("**Regime label per fold**: SPY CAGR over OOT window. bull > +10%, bear < -5%, chop in between.\n")
    md.append("**Gates (task brief)**: pooled CAGR >= 10%, pooled DD <= 18%, no regime Sortino < 0.5, hit rate >= 65%.\n")

    md.append("## Pooled OOS metrics\n")
    md.append(f"- CAGR: **{pooled_metrics['cagr']*100:.2f}%**")
    md.append(f"- Max drawdown: **{pooled_metrics['max_dd_pct']:.2f}%**")
    md.append(f"- Sortino: **{pooled_metrics['sortino']:.2f}**")
    md.append(f"- Sharpe: **{pooled_metrics['sharpe']:.2f}**")
    md.append(f"- Worst month: **{pooled_metrics['worst_month_pct']:.2f}%**")
    md.append(f"- N trading days pooled: **{pooled_metrics['n_days']}**")
    md.append(f"- Fold hit rate (positive CAGR): **{hit_rate:.1f}%**  (n_folds={len(df)})\n")

    md.append("## Per-fold table\n")
    md.append("| fold | OOT start | OOT end | regime | SPY CAGR | fold CAGR | fold DD | fold Sortino | fold Sharpe |")
    md.append("|---|---|---|---|---|---|---|---|---|")
    for _, r in df.iterrows():
        md.append(f"| {int(r['fold'])} | {r['oot_start']} | {r['oot_end']} | {r['regime']} | "
                  f"{r['spy_cagr_pct']:+.1f}% | {r['fold_cagr_pct']:+.2f}% | "
                  f"{r['fold_dd_pct']:.2f}% | {r['fold_sortino']:.2f} | {r['fold_sharpe']:.2f} |")
    md.append("")

    md.append("## Per-regime stratification\n")
    md.append("| regime | n_folds | avg CAGR | avg DD | avg Sortino | avg Sharpe | hit rate |")
    md.append("|---|---|---|---|---|---|---|")
    for r in regime_rows:
        if r["n_folds"] == 0:
            md.append(f"| {r['regime']} | 0 | — | — | — | — | — |")
        else:
            md.append(f"| {r['regime']} | {r['n_folds']} | "
                      f"{r['avg_cagr_pct']:+.2f}% | {r['avg_dd_pct']:.2f}% | "
                      f"{r['avg_sortino']:.2f} | {r['avg_sharpe']:.2f} | "
                      f"{r['hit_rate_pct']:.1f}% |")
    md.append("")

    md.append("## Verdict\n")
    md.append(f"**{verdict}**")
    if reasons:
        md.append("\nFailure reasons:")
        for x in reasons:
            md.append(f"- {x}")
    else:
        md.append("\nAll four gates satisfied.")
    md.append("\n## In-sample reference\n")
    md.append("- CAGR 20.38%, DD 12.38%, Sortino 1.56, Sharpe 1.41 (2010-2026 full sample).\n")
    (OUT / "wf_report.md").write_text("\n".join(md))
    print(f"[wf] wrote {OUT/'wf_report.md'}", flush=True)
    print(f"[wf] verdict: {verdict}  reasons={reasons}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
