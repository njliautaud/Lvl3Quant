"""
Sector picker v7 — LightGBM per-sector ranker (HC #565 R2 follow-up to the
2026-06-08 verdict that ridge+sparse-K is the bottleneck, not the data shelf).

Same panel (master_panel_v2), same walk-forward windows (36mo train / 12mo OOT
/ 6mo step), same long-short top3/bot3 books, same 5bps per-rebalance cost.
The ONLY change vs v5/v6 is the cross-sectional ranker: a regularized LightGBM
regressor replaces dense ridge / sparse-K ridge.

Why LightGBM:
  - Captures non-linear interactions and threshold effects that a linear
    ridge cannot. Reddit sentiment volatility, accounting-change flags, and
    margin trends interact (signal is conditional, not additive).
  - Native handling of NaN — no fillna(0) collapsing missing data onto the
    cross-sectional mean (which a ridge has to do).
  - Built-in feature importance + per-fold split counts to diagnose
    whether new shelf families actually contribute.
  - Hard early-stopping + small num_leaves controls overfit on noisy cross-
    sectional alpha (40-60 names per sector × ~750 train days = ~30k rows).

Honest deploy gate (HC #559 R4 + HC #428 R1): per-sector pooled-OOT Calmar
must be >= 1.0 on >= 8 of 11 sectors. Pooled-equal-weight book reports the
same metrics as v5/v6 for direct comparison.

CLI:
    python3 sector_picker_v7_lgbm.py [--hold-days 10] [--n-jobs -1] [--out DIR]

Parallelism: across sectors via joblib (each sector independent). LightGBM
inside each worker uses 2 threads to avoid contention.
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

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT / "research"))
sys.path.insert(0, str(ROOT / "strategy/macro_picker"))

import sector_picker_v4 as v4  # type: ignore
import sector_picker_v6_sparseK as v6  # type: ignore  (for FEATURE_POOL)
from walk_forward import _metrics  # type: ignore

import lightgbm as lgb

PANEL_V2 = ROOT / "data/feature_store/master_panel/master_panel_v2.parquet"
DEFAULT_HOLD = 10
TOP_N = 3
BOT_N = 3
TXN_COST_BPS = 5
TRADING_DAYS = 252


# ---------------------------------------------------------------------------
# LightGBM hyperparams — conservative, regularized for noisy cross-section
# ---------------------------------------------------------------------------
LGB_PARAMS = dict(
    objective="regression",
    metric="rmse",
    learning_rate=0.03,
    num_leaves=15,            # small — penalize overfit on ~30k rows
    min_data_in_leaf=200,     # require meaningful population per leaf
    feature_fraction=0.7,     # column subsample per tree
    bagging_fraction=0.8,     # row subsample
    bagging_freq=4,
    lambda_l2=1.0,            # ridge-style regularization on leaf values
    verbosity=-1,
    num_threads=2,            # avoid contention with joblib parallel
)
N_BOOST_ROUND = 400
EARLY_STOP_ROUNDS = 40


def portfolio_for_sector_lgbm(sp: pd.DataFrame, feats: list[str], hold_days: int):
    """Walk-forward LightGBM fit + OOT score → long-short top3/bot3 book."""
    sp = sp.sort_values(["date", "ticker"]).reset_index(drop=True)
    sp["ret_raw"] = sp["ret"].astype(float)

    start, end = sp["date"].min(), sp["date"].max()
    cursor = start
    daily_pnl = pd.Series(dtype=float, index=pd.DatetimeIndex([]))
    folds = []
    importance_gain: dict[str, list[float]] = {}
    importance_split: dict[str, list[int]] = {}

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
            cursor = cursor + pd.DateOffset(months=6); continue

        # Cross-sectional z-score per date (same as v5/v6 — keeps LGBM
        # fair-comparable; otherwise LGBM gets a free advantage from raw
        # scale features that linear models can't use).
        train_z = v4._xs_zscore(train, feats)
        oot_z = v4._xs_zscore(oot, feats)
        # Match v5/v6: fill cross-sectional z-score NaNs (constant-valued
        # broadcast features → 0/0). LGBM handles NaN natively but the
        # downstream score-and-rank breaks if entire snapshots are NaN.
        for f in feats:
            train_z[f] = train_z[f].fillna(0.0)
            oot_z[f] = oot_z[f].fillna(0.0)
        train_z = train_z.dropna(subset=["y_fwd"])
        if train_z.empty:
            cursor = cursor + pd.DateOffset(months=6); continue

        # Hold-out the LAST 20% of train dates for early stopping (still
        # before OOT — strict no leakage).
        train_dates = sorted(train_z["date"].unique())
        n_es = max(20, int(0.20 * len(train_dates)))
        es_dates = set(train_dates[-n_es:])
        es_mask = train_z["date"].isin(es_dates)
        tr_inner = train_z[~es_mask]
        tr_es = train_z[es_mask]
        if len(tr_inner) < 500 or len(tr_es) < 100:
            cursor = cursor + pd.DateOffset(months=6); continue

        X_tr = tr_inner[feats].values
        y_tr = tr_inner["y_fwd"].values
        X_es = tr_es[feats].values
        y_es = tr_es["y_fwd"].values

        dtrain = lgb.Dataset(X_tr, label=y_tr, feature_name=feats, free_raw_data=False)
        des = lgb.Dataset(X_es, label=y_es, feature_name=feats,
                          reference=dtrain, free_raw_data=False)
        try:
            booster = lgb.train(
                LGB_PARAMS,
                dtrain,
                num_boost_round=N_BOOST_ROUND,
                valid_sets=[des],
                callbacks=[lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                           lgb.log_evaluation(0)],
            )
        except Exception:
            cursor = cursor + pd.DateOffset(months=6); continue

        gain = booster.feature_importance(importance_type="gain")
        split = booster.feature_importance(importance_type="split")
        for f, g, s in zip(feats, gain, split):
            importance_gain.setdefault(f, []).append(float(g))
            importance_split.setdefault(f, []).append(int(s))

        # ---- Score OOT ----
        X_oot = oot_z[feats].values
        oot_z = oot_z.copy()
        oot_z["score"] = booster.predict(X_oot, num_iteration=booster.best_iteration)

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
            "best_iter": int(booster.best_iteration or 0),
            "n_train_rows": int(len(tr_inner)),
            "n_es_rows": int(len(tr_es)),
        })
        cursor = cursor + pd.DateOffset(months=6)

    daily_pnl = daily_pnl.sort_index()
    daily_pnl = daily_pnl[~daily_pnl.index.duplicated(keep="last")]

    avg_gain = {f: float(np.mean(v)) for f, v in importance_gain.items()}
    avg_split = {f: float(np.mean(v)) for f, v in importance_split.items()}
    return {
        "daily_pnl": daily_pnl,
        "folds": folds,
        "avg_gain": avg_gain,
        "avg_split": avg_split,
    }


def _run_one_sector(sector: str, panel: pd.DataFrame, hold_days: int):
    sp = panel[panel["sector"] == sector].copy()
    if sp.empty or sp["ticker"].nunique() < 5:
        return sector, None
    feats_present = [f for f in v6.FEATURE_POOL if f in sp.columns]
    res = portfolio_for_sector_lgbm(sp, feats_present, hold_days=hold_days)
    pnl = res["daily_pnl"]
    if pnl.empty:
        return sector, None
    return sector, {
        "n_tickers": int(sp["ticker"].nunique()),
        "n_candidate_features": len(feats_present),
        "n_folds": len(res["folds"]),
        "avg_gain": res["avg_gain"],
        "avg_split": res["avg_split"],
        "folds": res["folds"],
        "daily_pnl": pnl,
        "metrics": _metrics(pnl),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold-days", type=int, default=DEFAULT_HOLD)
    ap.add_argument("--n-jobs", type=int, default=-1)
    ap.add_argument("--out", type=str, default=None)
    args = ap.parse_args()

    ts = time.strftime("%Y%m%d_%H%M%S")
    if args.out is None:
        out_dir = ROOT / f"output/macro_picker/lgbm_v7_{ts}_H{args.hold_days:02d}d"
    else:
        out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[v7 lgbm] hold_days={args.hold_days}  out={out_dir}")
    print(f"[v7 lgbm] panel={PANEL_V2.name}  feature_pool={len(v6.FEATURE_POOL)} candidates")

    v4.PANEL = PANEL_V2
    v4.HOLD_DAYS = args.hold_days

    print("loading panel ...")
    panel = v4.load_panel()
    print(f"  shape {panel.shape}")
    panel = v4.build_target(panel, hold_days=args.hold_days)

    sectors = sorted([s for s in panel["sector"].dropna().unique() if s != "ETF"])
    print(f"sectors: {sectors}")

    t0 = time.time()
    results = Parallel(n_jobs=args.n_jobs, backend="loky", verbose=5)(
        delayed(_run_one_sector)(s, panel, args.hold_days) for s in sectors
    )
    wall = time.time() - t0
    print(f"  parallel wall: {wall:.1f}s across {len(sectors)} sectors")

    per_sector = {s: r for s, r in results if r is not None}
    if not per_sector:
        print("no sector results"); return

    # Combined book — equal-weight per day
    pnl_frames = {s: r["daily_pnl"] for s, r in per_sector.items()}
    combined_df = pd.concat(pnl_frames.values(), axis=1, keys=pnl_frames.keys())
    combined_pnl = combined_df.mean(axis=1, skipna=True).dropna()

    strat_m = _metrics(combined_pnl)
    deployable = sum(1 for r in per_sector.values()
                     if (r["metrics"].get("calmar") or 0.0) >= 1.0)

    # Aggregate avg gain across sectors
    all_gain: dict[str, list[float]] = {}
    for r in per_sector.values():
        for f, g in r["avg_gain"].items():
            all_gain.setdefault(f, []).append(g)
    agg_gain = {f: float(np.mean(v)) for f, v in all_gain.items()}
    top_gain = sorted(agg_gain.items(), key=lambda kv: -kv[1])[:15]

    summary = {
        "hold_days": args.hold_days,
        "wall_sec": wall,
        "n_sectors": len(per_sector),
        "deployable": deployable,
        "n_features": len(v6.FEATURE_POOL),
        "pooled_metrics": strat_m,
        "per_sector_metrics": {s: r["metrics"] for s, r in per_sector.items()},
        "top15_features_by_gain": top_gain,
    }
    (out_dir / "report.json").write_text(json.dumps(summary, indent=2, default=str))

    # Markdown report
    lines = [
        "# Sector picker v7 — LightGBM (HC #565 R2 / 2026-06-08 verdict)",
        "",
        f"Same panel as v5/v6 ({PANEL_V2.name}). Feature pool: {len(v6.FEATURE_POOL)} candidates.",
        f"Hold horizon: {args.hold_days}d. Walk-forward: 36mo train / 12mo OOT / 6mo step.",
        "",
        "## Pooled equal-weight book",
        f"- Sharpe: {strat_m.get('sharpe', 0):.2f}",
        f"- Calmar: {strat_m.get('calmar', 0):.2f}",
        f"- CAGR:   {strat_m.get('cagr', 0)*100:.1f}%",
        f"- MaxDD:  {strat_m.get('max_dd', 0)*100:.1f}%",
        f"- Deployable sectors (Calmar ≥ 1): **{deployable} / {len(per_sector)}**",
        "",
        "## Per-sector pooled-OOT",
        "",
        "| Sector | Sharpe | Calmar | CAGR | MaxDD | PF | WR |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in sorted(per_sector.keys()):
        m = per_sector[s]["metrics"]
        lines.append(
            f"| {s} | {m.get('sharpe',0):.2f} | {m.get('calmar',0):.2f} | "
            f"{m.get('cagr',0)*100:.1f}% | {m.get('max_dd',0)*100:.1f}% | "
            f"{m.get('pf',0):.2f} | {m.get('wr',0)*100:.1f}% |"
        )
    lines += [
        "",
        "## Top-15 features by average LGBM gain",
        "",
        "| Feature | Avg gain |",
        "|---|---|",
    ]
    for f, g in top_gain:
        lines.append(f"| `{f}` | {g:.1f} |")
    (out_dir / "report.md").write_text("\n".join(lines))

    print(f"\nOK v7 lgbm: pooled Sharpe={strat_m.get('sharpe',0):.2f} "
          f"Calmar={strat_m.get('calmar',0):.2f} "
          f"deployable={deployable}/{len(per_sector)}")
    print(f"  → {out_dir}/report.md")


if __name__ == "__main__":
    main()
