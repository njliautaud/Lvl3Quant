"""
Sector picker v6 — Sparse-K cardinality-constrained per-sector fit.

Motivation (2026-06-08): v5 ran a dense ridge over the full 76-feature pool on
master_panel_v2 and produced pooled Sharpe 0.77 / Calmar 0.30 / 0-of-11
deployable sectors. The horizon sweep this morning showed hold-window is NOT
the bottleneck (10d hold was best at Calmar 0.42, still 0/11). The remaining
hypothesis: the dense ridge is washing out real signal with 76 noisy features.

v6 keeps everything else identical to v5 (same panel, same walk-forward
windows, same long-short top3/bot3 books, same costs) but replaces the dense
fit with a per-fold SPARSE pipeline:

    1. Inside each train fold, compute univariate Spearman IC of every
       candidate feature vs forward return (train-only — no OOT leakage).
    2. Keep the top-K features by |IC|.
    3. Fit ridge on ONLY those K features.
    4. Score OOT using the same K features (z-scored on OOT cross-section).

Why univariate IC filter (not L1 / group-LASSO):
    - L1 path solvers add a nested CV layer (alpha-per-fold) on top of an
      already nested walk-forward — slow + harder to debug.
    - Group-LASSO needs clean feature groupings; our pool is heterogeneous
      (tech / fundamentals / flows / insider / 10K text) and groups would
      be hand-defined and brittle.
    - Univariate IC is what the diagnosis pointed at ("top 5 ridge weights
      are all fundamentals — real signal exists, dense model is washing it
      out"). Hard-threshold by IC honors that diagnosis directly.
    - Spearman (rank corr) is robust to outliers and monotone non-linearity
      — better than Pearson for cross-sectional signal scoring.

CLI:
    python3 sector_picker_v6_sparseK.py [--K 10] [--hold-days 10] [--out DIR]

Parallelism (HC #565 R1): across SECTORS via joblib, not across folds.
Sectors are independent (11 sectors), each one is a few-second ridge — joblib
n_jobs=-1 gives near-linear speedup on Jupiter's 64-core CPU.
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from scipy.stats import spearmanr

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "strategy/macro_picker"))

import sector_picker_v4 as v4  # noqa: E402  type: ignore
from walk_forward import _metrics  # noqa: E402  type: ignore

PANEL_V2 = ROOT / "data/feature_store/master_panel/master_panel_v2.parquet"
DEFAULT_HOLD = 10
DEFAULT_K = 10
TOP_N = 3
BOT_N = 3
TXN_COST_BPS = 5
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# Feature pool — identical shape to v5 (76 features on master_panel_v2)
# ---------------------------------------------------------------------------
FUND_PIT = [
    "fp_revenue_ttm", "fp_fcf_ttm", "fp_ni_ttm", "fp_eps_ttm", "fp_ebitda_ttm",
    "fp_gross_margin", "fp_net_margin", "fp_ebitda_margin",
    "fp_roe", "fp_debt_to_equity", "fp_current_ratio",
    "fp_market_cap_pit", "fp_fcf_yield",
    "fp_rev_yoy_growth", "fp_ni_yoy_growth", "fp_eps_yoy_growth",
    "fp_fcf_yoy_growth", "fp_margin_trend_4q", "fp_beat_rate_4q",
]
SECTOR_FLOWS = [
    "sf_dv_z_20d", "sf_dv_chg_60d",
    "sf_rs_60d_spy", "sf_rs_252d_spy",
    "sf_px_200dma", "sf_vol_252d",
    "sf_corr_tlt_60d", "sf_corr_hyg_60d", "sf_corr_uup_60d", "sf_corr_gld_60d",
]
INSIDER_V2 = [
    "ins_net_insider_usd", "ins_gross_buy_usd", "ins_gross_sell_usd",
    "ins_n_buys", "ins_n_sells", "ins_n_buyers", "ins_n_sellers",
    "ins_mean_price",
]
# 2026-06-08: shelf-completeness expansion (HC #565 R2). Reddit hype + Google
# Trends are now wired into master_panel.py; expose them to the IC ranker so
# sparse-K can pick them when they outscore fundamentals/flows.
REDDIT_HYPE = [
    "rh_mention_count", "rh_upvote_sum", "rh_comment_count",
    "rh_sentiment_polarity", "rh_sentiment_volatility",
]
GOOGLE_TRENDS = [
    "gt_search_interest", "gt_search_interest_z", "gt_related_breakout_count",
]
ANALYST_REVS = [
    "ar_strong_buy", "ar_buy", "ar_hold", "ar_sell", "ar_strong_sell",
    "ar_net_score", "ar_net_score_delta_qoq",
]
EXTRA = FUND_PIT + SECTOR_FLOWS + INSIDER_V2 + REDDIT_HYPE + GOOGLE_TRENDS + ANALYST_REVS
FEATURE_POOL = list(dict.fromkeys(v4.FEATURE_POOL_BASE + v4.TEXT_FEATURES + EXTRA))


# ---------------------------------------------------------------------------
# Univariate IC ranking inside a train fold
# ---------------------------------------------------------------------------
def _train_fold_top_k(train_z: pd.DataFrame, feats: list[str], K: int) -> list[str]:
    """Return top-K features by |Spearman IC| of feature vs y_fwd on train data.

    train_z is already cross-sectionally z-scored. NaNs are filled with 0
    (consistent with v4 ridge pipeline). Features with constant value get
    IC=0 and fall to the bottom of the ranking.
    """
    y = train_z["y_fwd"].values.astype(float)
    ics = []
    for f in feats:
        x = train_z[f].values.astype(float)
        # spearmanr is robust; nan_policy=omit handles residual NaNs
        try:
            rho, _ = spearmanr(x, y, nan_policy="omit")
        except Exception:
            rho = 0.0
        if rho is None or not np.isfinite(rho):
            rho = 0.0
        ics.append((f, float(rho)))
    ics.sort(key=lambda kv: -abs(kv[1]))
    chosen = [f for f, _ in ics[:K]]
    return chosen, dict(ics)


# ---------------------------------------------------------------------------
# Per-sector walk-forward with sparse-K fit
# ---------------------------------------------------------------------------
def portfolio_for_sector_sparseK(
    sp: pd.DataFrame,
    feats: list[str],
    K: int,
    hold_days: int,
):
    sp = sp.sort_values(["date", "ticker"]).reset_index(drop=True)
    sp["ret_raw"] = sp["ret"].astype(float)

    start, end = sp["date"].min(), sp["date"].max()
    cursor = start
    daily_pnl = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    folds = []
    chosen_counts: dict[str, int] = {}
    ic_history: dict[str, list[float]] = {}
    coef_history: dict[str, list[float]] = {}

    while True:
        tr_start = cursor
        tr_end = tr_start + pd.DateOffset(months=36)
        oot_start = tr_end
        oot_end = oot_start + pd.DateOffset(months=12)
        if oot_end > end + pd.Timedelta(days=1):
            break

        train = sp[(sp["date"] >= tr_start) & (sp["date"] < tr_end)]
        oot = sp[(sp["date"] >= oot_start) & (sp["date"] < oot_end)]
        if len(train) < 1000 or len(oot) < 100:
            cursor = cursor + pd.DateOffset(months=6)
            continue

        train_z = v4._xs_zscore(train, feats)
        oot_z = v4._xs_zscore(oot, feats)
        for f in feats:
            train_z[f] = train_z[f].fillna(0.0)
            oot_z[f] = oot_z[f].fillna(0.0)
        train_z = train_z.dropna(subset=["y_fwd"])
        if train_z.empty:
            cursor = cursor + pd.DateOffset(months=6); continue

        # ---- SPARSE-K STEP: pick K features by |IC| on TRAIN ONLY ----
        chosen, ic_map = _train_fold_top_k(train_z, feats, K)
        for f in chosen:
            chosen_counts[f] = chosen_counts.get(f, 0) + 1
        for f, ic in ic_map.items():
            ic_history.setdefault(f, []).append(ic)

        # ---- Fit ridge on ONLY chosen K features ----
        X_tr = train_z[chosen].values
        y_tr = train_z["y_fwd"].values
        coef, intercept, alpha = v4._fit_ridge(X_tr, y_tr)
        if coef is None:
            cursor = cursor + pd.DateOffset(months=6); continue
        for f, c in zip(chosen, coef):
            coef_history.setdefault(f, []).append(float(c))

        # ---- Score OOT on the same K features ----
        X_oot = oot_z[chosen].fillna(0.0).values
        oot_z = oot_z.copy()
        oot_z["score"] = X_oot @ coef + intercept

        unique_dates = sorted(oot_z["date"].unique())
        rebal_dates = unique_dates[::hold_days]
        fold_daily = []
        for rd in rebal_dates:
            snap = oot_z[oot_z["date"] == rd].dropna(subset=["score"])
            if len(snap) < (TOP_N + BOT_N):
                continue
            longs = snap.nlargest(TOP_N, "score")["ticker"].tolist()
            shorts = snap.nsmallest(BOT_N, "score")["ticker"].tolist()
            hold_win = oot_z[(oot_z["date"] > rd)
                             & (oot_z["date"] <= rd + pd.Timedelta(days=hold_days))]
            for d, g in hold_win.groupby("date"):
                lret = g[g["ticker"].isin(longs)]["ret_raw"].mean() if longs else 0.0
                sret = g[g["ticker"].isin(shorts)]["ret_raw"].mean() if shorts else 0.0
                day_ret = float(np.clip(
                    0.5 * (lret if pd.notna(lret) else 0)
                    - 0.5 * (sret if pd.notna(sret) else 0),
                    -0.25, 0.25))
                fold_daily.append((d, day_ret))
            tc = TXN_COST_BPS / 10000.0
            fold_daily.append((rd, -tc))

        if fold_daily:
            s = pd.Series(dict(fold_daily))
            s.index = pd.to_datetime(s.index)
            s = s.groupby(level=0).sum()
            daily_pnl = pd.concat([daily_pnl, s])

        folds.append({
            "oot_start": str(oot_start.date()),
            "oot_end": str(oot_end.date()),
            "alpha": alpha,
            "chosen_features": chosen,
            "coef": dict(zip(chosen, [float(c) for c in coef])),
        })
        cursor = cursor + pd.DateOffset(months=6)

    daily_pnl = daily_pnl.sort_index()
    daily_pnl = daily_pnl[~daily_pnl.index.duplicated(keep="last")]

    avg_coef = {f: float(np.mean(v)) for f, v in coef_history.items()}
    avg_ic = {f: float(np.mean(v)) for f, v in ic_history.items()}
    return {
        "daily_pnl": daily_pnl,
        "folds": folds,
        "avg_coef": avg_coef,
        "avg_ic": avg_ic,
        "chosen_counts": chosen_counts,
    }


# ---------------------------------------------------------------------------
# Sector-level worker for joblib
# ---------------------------------------------------------------------------
def _run_one_sector(sector: str, panel: pd.DataFrame, K: int, hold_days: int):
    sp = panel[panel["sector"] == sector].copy()
    if sp.empty or sp["ticker"].nunique() < 5:
        return sector, None
    feats_present = [f for f in FEATURE_POOL if f in sp.columns]
    res = portfolio_for_sector_sparseK(sp, feats_present, K=K, hold_days=hold_days)
    pnl = res["daily_pnl"]
    if pnl.empty:
        return sector, None
    return sector, {
        "n_tickers": int(sp["ticker"].nunique()),
        "n_candidate_features": len(feats_present),
        "n_folds": len(res["folds"]),
        "avg_coef": res["avg_coef"],
        "avg_ic": res["avg_ic"],
        "chosen_counts": res["chosen_counts"],
        "folds": res["folds"],
        "daily_pnl": pnl,
        "metrics": _metrics(pnl),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--K", type=int, default=DEFAULT_K,
                    help="Cardinality: number of features kept per fold (default 10)")
    ap.add_argument("--hold-days", type=int, default=DEFAULT_HOLD,
                    help="Hold horizon in trading days (default 10 — best from horizon sweep)")
    ap.add_argument("--out", type=str, default=None,
                    help="Output dir (default: output/macro_picker/sparseK_<ts>/K<K>_H<H>)")
    ap.add_argument("--n-jobs", type=int, default=-1,
                    help="joblib n_jobs across sectors (default -1 = all cores)")
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.out is None:
        out_dir = ROOT / f"output/macro_picker/sparseK_{ts}/K{args.K:02d}_H{args.hold_days:02d}d"
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v6 sparseK] K={args.K}  hold_days={args.hold_days}  out={out_dir}")
    print(f"[v6 sparseK] panel={PANEL_V2.name}  feature_pool={len(FEATURE_POOL)} candidates")

    # Patch v4 globals so its helpers see the right panel + horizon
    v4.PANEL = PANEL_V2
    v4.HOLD_DAYS = args.hold_days

    print("loading panel ...")
    panel = v4.load_panel()
    print(f"  shape {panel.shape}")
    spy_ret, fund = v4.load_spy_and_funding()
    panel = v4.build_target(panel, hold_days=args.hold_days)

    sectors = sorted([s for s in panel["sector"].dropna().unique() if s != "ETF"])
    print(f"sectors: {sectors}")

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(_run_one_sector)(s, panel, args.K, args.hold_days) for s in sectors
    )
    wall = time.time() - t0
    print(f"  parallel wall: {wall:.1f}s across {len(sectors)} sectors")

    per_sector = {s: r for s, r in results if r is not None}
    if not per_sector:
        print("no sector results, aborting"); return

    # Combined book — equal-weight across sectors per day
    pnl_frames = {s: r["daily_pnl"] for s, r in per_sector.items()}
    combined_df = pd.concat(pnl_frames.values(), axis=1, keys=pnl_frames.keys())
    combined_pnl = combined_df.mean(axis=1, skipna=True).dropna()

    strat_m = _metrics(combined_pnl)
    aligned = pd.concat([combined_pnl.rename("s"),
                         spy_ret.rename("spy"),
                         fund.rename("f")], axis=1, join="inner").dropna(subset=["s", "spy"])
    aligned["f"] = aligned["f"].ffill().bfill()
    spy1 = v4._spy_levered(aligned["spy"], 1.0, aligned["f"])
    spy15 = v4._spy_levered(aligned["spy"], 1.5, aligned["f"])
    spy2 = v4._spy_levered(aligned["spy"], 2.0, aligned["f"])
    spy1_m = _metrics(spy1)
    spy15_m = _metrics(spy15)
    spy2_m = _metrics(spy2)

    per_sector_metrics = {s: r["metrics"] for s, r in per_sector.items()}
    n_deploy = sum(1 for m in per_sector_metrics.values() if m.get("calmar", -1e9) >= 1.0)

    # Feature picks: how often each feature was selected across folds-x-sectors
    pick_counts: dict[str, int] = {}
    for r in per_sector.values():
        for f, c in r["chosen_counts"].items():
            pick_counts[f] = pick_counts.get(f, 0) + c
    pick_counts_sorted = dict(sorted(pick_counts.items(), key=lambda kv: -kv[1]))

    # Avg |coef| across selected sectors
    feat_imp: dict[str, list[float]] = {}
    for r in per_sector.values():
        for f, c in r["avg_coef"].items():
            feat_imp.setdefault(f, []).append(abs(c))
    feat_imp_avg = {f: float(np.mean(v)) for f, v in feat_imp.items()}
    feat_imp_sorted = dict(sorted(feat_imp_avg.items(), key=lambda kv: -kv[1]))

    # ---- Per-sector formulas — write a compact JSON ----
    formulas = {}
    for s, r in per_sector.items():
        formulas[s] = {
            "avg_coef_on_selected": r["avg_coef"],
            "chosen_counts": r["chosen_counts"],
            "per_fold": r["folds"],
        }
    (out_dir / "per_sector_formulas.json").write_text(
        json.dumps(formulas, indent=2, default=str))

    out = {
        "config": {
            "K": args.K,
            "hold_days": args.hold_days,
            "panel": str(PANEL_V2),
            "candidate_pool_size": len(FEATURE_POOL),
            "filter": "univariate Spearman IC, top-K by |IC|, train-only",
            "fit": "ridge over K selected features (v4 _fit_ridge)",
        },
        "pooled_oot_combined": strat_m,
        "spy_1x": spy1_m,
        "spy_15x": spy15_m,
        "spy_2x": spy2_m,
        "per_sector": per_sector_metrics,
        "n_sectors_calmar_pass": n_deploy,
        "feature_pick_counts": pick_counts_sorted,
        "feature_importance_avg_abs_coef": feat_imp_sorted,
        "n_trading_days": int(len(combined_pnl)),
        "date_range": [str(combined_pnl.index.min().date()),
                       str(combined_pnl.index.max().date())] if len(combined_pnl) else [None, None],
        "sectors_used": list(per_sector.keys()),
        "wall_sec": round(wall, 1),
    }
    (out_dir / "report.json").write_text(json.dumps(out, indent=2, default=str))

    # Per-day PnL parquet
    pnl_df = pd.DataFrame({"date": combined_pnl.index,
                           "combined_pnl": combined_pnl.values})
    for s, p in pnl_frames.items():
        pnl_df = pnl_df.merge(
            pd.DataFrame({"date": p.index, s: p.values}), on="date", how="left")
    pnl_df.to_parquet(out_dir / "per_day_pnl.parquet", index=False)

    # Markdown summary
    md = [f"# Sector picker v6 — Sparse-K (K={args.K}, hold={args.hold_days}d)"]
    md.append("")
    md.append(f"**Filter**: univariate Spearman IC, top-{args.K} per train fold (no OOT leakage)")
    md.append(f"**Fit**: ridge on selected K features")
    md.append(f"**Pool**: {len(FEATURE_POOL)} candidate features (76-feature master_panel_v2 shelf)")
    md.append(f"**Walk-forward**: 36mo train / 12mo OOT, 6mo step")
    md.append("")
    md.append("## Pooled-OOT")
    md.append("")
    md.append("| Metric | Strategy | SPY 1x | SPY 1.5x | SPY 2x |")
    md.append("|---|---:|---:|---:|---:|")
    md.append(f"| Sharpe | {strat_m['sharpe']:.2f} | {spy1_m['sharpe']:.2f} | {spy15_m['sharpe']:.2f} | {spy2_m['sharpe']:.2f} |")
    md.append(f"| Sortino | {strat_m['sortino']:.2f} | {spy1_m['sortino']:.2f} | {spy15_m['sortino']:.2f} | {spy2_m['sortino']:.2f} |")
    md.append(f"| CAGR | {strat_m['cagr']*100:.1f}% | {spy1_m['cagr']*100:.1f}% | {spy15_m['cagr']*100:.1f}% | {spy2_m['cagr']*100:.1f}% |")
    md.append(f"| MaxDD | {strat_m['max_dd']*100:.1f}% | {spy1_m['max_dd']*100:.1f}% | {spy15_m['max_dd']*100:.1f}% | {spy2_m['max_dd']*100:.1f}% |")
    md.append(f"| Calmar | {strat_m['calmar']:.2f} | {spy1_m['calmar']:.2f} | {spy15_m['calmar']:.2f} | {spy2_m['calmar']:.2f} |")
    md.append("")
    md.append(f"**Deployable sectors (Calmar >= 1.0)**: {n_deploy} / {len(per_sector)}")
    md.append("")
    md.append("## Per-sector pooled-OOT")
    md.append("")
    md.append("| Sector | Sharpe | CAGR | MaxDD | Calmar | PF | WR |")
    md.append("|---|---:|---:|---:|---:|---:|---:|")
    for s, m in sorted(per_sector_metrics.items(), key=lambda kv: -kv[1]["sharpe"]):
        md.append(f"| {s} | {m['sharpe']:.2f} | {m['cagr']*100:.1f}% | "
                  f"{m['max_dd']*100:.1f}% | {m['calmar']:.2f} | "
                  f"{m['pf']:.2f} | {m['wr']*100:.1f}% |")
    md.append("")
    md.append("## Top-20 most-frequently-picked features (across folds x sectors)")
    md.append("")
    md.append("| Rank | Feature | Pick count |")
    md.append("|---:|---|---:|")
    for i, (f, c) in enumerate(list(pick_counts_sorted.items())[:20], 1):
        md.append(f"| {i} | `{f}` | {c} |")
    md.append("")
    (out_dir / "report.md").write_text("\n".join(md))

    print()
    print("\n".join(md))
    print()
    print(f"WROTE:")
    print(f"  {out_dir / 'report.json'}")
    print(f"  {out_dir / 'report.md'}")
    print(f"  {out_dir / 'per_sector_formulas.json'}")
    print(f"  {out_dir / 'per_day_pnl.parquet'}")


if __name__ == "__main__":
    main()
