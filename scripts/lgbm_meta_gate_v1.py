#!/usr/bin/env python3
"""
LightGBM Meta-Gate v1 (HC #486 R4 prototype)
============================================
Trains a LightGBM gate on v3.4.2 CNN-Mamba OOT prediction heads (47 days available, 34 with data)
to predict which signal moments will be PROFITABLE in FIFO replay (tp4sl3 and tp8sl5 brackets).

Inputs:
  /home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_YYYYMMDD.npz

Each day has 49.5k samples and 30+ prediction heads (the CNN-Mamba multi-task outputs).
Features = all `pred_*` heads.
Label = sign(target_fifo_tp4sl3_net) > 0  (profitable after FIFO + commissions)

Method (HC #428 R2, HC #0):
  - SLIDING walk-forward: train on N-day window ending at day t-1, OOT on day t, slide by 1 day.
  - Anchored chronological order; no shuffling within a day either.
  - Per-day OOT metrics: Sharpe (per-trade), PF, WR, gross/net ticks, count.
  - Cost convention (CLAUDE.md cost table): FIFO net targets already include commissions;
    no further cost subtraction needed for the tp4sl3_net target.
  - Gate decision: predicted P(profitable) >= threshold => take trade; else skip.
  - Compare gated vs. ungated (baseline = take every signal where pred_log_ret_5s > 0 for long,
    < 0 for short -- traditional sign-of-prediction baseline).

Outputs:
  /home/jupiter/Lvl3Quant/output/lgbm_meta_gate_v1/
    per_day_metrics.csv     -- per-day Sharpe/PF/WR/count for gated vs ungated
    feature_importance.csv  -- gain-importance ranked
    verdict.md              -- markdown summary (HC #486 R4 acceptance test)
    config.json             -- run config
    train_log.jsonl         -- per-fold training events
"""
import argparse, json, os, sys, glob, time, traceback
from pathlib import Path
import numpy as np
import pandas as pd
import lightgbm as lgb

OOT_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/lgbm_meta_gate_v1")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Cost constants (CLAUDE.md cost table) -- tp4sl3_net already includes commissions per the
# CNN-Mamba multi-task target definition.
ES_RT_COMMISSION_TICKS = 0.376
ES_TICK_VALUE = 12.50


def load_day(path):
    """Load one OOT day. Returns (DataFrame, feat_cols) or (None, None) if day is empty/incomplete."""
    z = np.load(path, allow_pickle=True)
    required = ["target_fifo_tp4sl3_net", "target_fifo_tp8sl5_net",
                "target_log_ret_5s", "pred_log_ret_5s", "sample_dates"]
    if any(k not in z.files for k in required):
        return None, None
    feat_cols = [k for k in z.keys() if k.startswith("pred_")]
    df = pd.DataFrame({k: z[k] for k in feat_cols})
    df["target_fifo_tp4sl3_net"] = z["target_fifo_tp4sl3_net"]
    df["target_fifo_tp8sl5_net"] = z["target_fifo_tp8sl5_net"]
    df["target_fifo_tp4sl3_hit_tp"] = z["target_fifo_tp4sl3_hit_tp"]
    df["target_log_ret_5s"] = z["target_log_ret_5s"]
    df["pred_log_ret_5s_raw"] = z["pred_log_ret_5s"]
    df["date"] = str(z["sample_dates"][0])
    return df, feat_cols


def day_metrics(net_ticks, name="all"):
    """Per-day Sharpe (per-trade), PF, WR, mean ticks, count."""
    n = len(net_ticks)
    if n == 0:
        return dict(n=0, sharpe=np.nan, pf=np.nan, wr=np.nan, mean_ticks=np.nan, sum_ticks=0.0, name=name)
    pos = net_ticks[net_ticks > 0].sum()
    neg = -net_ticks[net_ticks < 0].sum()
    pf = pos / neg if neg > 1e-9 else np.inf
    wr = (net_ticks > 0).mean()
    sharpe = (net_ticks.mean() / net_ticks.std()) * np.sqrt(n) if net_ticks.std() > 1e-9 else 0.0
    return dict(
        n=int(n),
        sharpe=float(sharpe),
        pf=float(pf) if np.isfinite(pf) else 999.0,
        wr=float(wr),
        mean_ticks=float(net_ticks.mean()),
        sum_ticks=float(net_ticks.sum()),
        name=name,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train_window_days", type=int, default=15,
                    help="Sliding train window length (days). HC #428 R1 prefers >=15.")
    ap.add_argument("--target", default="target_fifo_tp4sl3_net",
                    choices=["target_fifo_tp4sl3_net", "target_fifo_tp8sl5_net"])
    ap.add_argument("--gate_threshold", type=float, default=0.55,
                    help="Primary probability threshold for taking a signal (gated mode).")
    ap.add_argument("--threshold_sweep", type=str, default="0.50,0.52,0.55,0.58,0.60,0.65",
                    help="Comma-separated list of additional thresholds to evaluate per fold.")
    ap.add_argument("--num_boost", type=int, default=300)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--num_leaves", type=int, default=63)
    ap.add_argument("--min_data_leaf", type=int, default=200)
    ap.add_argument("--feature_frac", type=float, default=0.8)
    ap.add_argument("--bagging_frac", type=float, default=0.8)
    args = ap.parse_args()

    cfg = vars(args).copy()
    cfg["script"] = __file__
    cfg["start_ts"] = time.strftime("%Y-%m-%d %H:%M:%S")
    (OUT_DIR / "config.json").write_text(json.dumps(cfg, indent=2))

    # Discover OOT days in order
    files = sorted(OOT_DIR.glob("oot_*.npz"))
    if len(files) < args.train_window_days + 5:
        print(f"FATAL: only {len(files)} OOT days available, need >= {args.train_window_days + 5}", file=sys.stderr)
        sys.exit(1)
    print(f"[init] {len(files)} OOT days available")

    # Pre-load all days
    print("[load] loading all days into memory...")
    days = []
    feat_cols = None
    skipped = []
    for f in files:
        df, fc = load_day(f)
        if df is None:
            skipped.append(f.stem)
            continue
        days.append((f.stem, df))
        if feat_cols is None:
            feat_cols = fc
    if skipped:
        print(f"[load] skipped {len(skipped)} empty/incomplete days: {skipped}")
    print(f"[load] loaded {len(days)} days; {len(feat_cols)} feature heads; "
          f"total samples = {sum(len(d) for _, d in days):,}")

    log_path = OUT_DIR / "train_log.jsonl"
    log_f = open(log_path, "w")
    metrics_rows = []
    sweep_all = []
    importance_accum = {c: 0.0 for c in feat_cols}
    n_folds_trained = 0

    # Walk-forward: for each OOT day t, train on days [t-W : t-1]
    W = args.train_window_days
    for t in range(W, len(days)):
        oot_name, oot_df = days[t]
        train_dfs = [days[i][1] for i in range(t - W, t)]
        train_df = pd.concat(train_dfs, ignore_index=True)

        y_train = (train_df[args.target].values > 0).astype(np.int32)
        y_oot = (oot_df[args.target].values > 0).astype(np.int32)
        X_train = train_df[feat_cols].values.astype(np.float32)
        X_oot = oot_df[feat_cols].values.astype(np.float32)

        # Skip degenerate folds
        if y_train.sum() < 100 or y_train.sum() > len(y_train) - 100:
            continue

        dtrain = lgb.Dataset(X_train, label=y_train, feature_name=feat_cols)
        params = dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=args.lr,
            num_leaves=args.num_leaves,
            min_data_in_leaf=args.min_data_leaf,
            feature_fraction=args.feature_frac,
            bagging_fraction=args.bagging_frac,
            bagging_freq=5,
            verbose=-1,
            num_threads=8,
        )
        booster = lgb.train(params, dtrain, num_boost_round=args.num_boost)
        proba = booster.predict(X_oot)

        # Gated metric: trade only when proba >= threshold
        net = oot_df[args.target].values
        take_gated = proba >= args.gate_threshold
        # Ungated baseline: take everything (the model is allowed to take any signal)
        take_baseline_all = np.ones_like(net, dtype=bool)
        # Directional baseline: take when CNN-Mamba says move is in the favorable direction
        # (since target_fifo_tp4sl3_net is a long-side simulated outcome, take when pred_log_ret_5s > 0)
        take_baseline_dir = oot_df["pred_log_ret_5s_raw"].values > 0

        m_gated = day_metrics(net[take_gated], name="gated")
        m_all = day_metrics(net[take_baseline_all], name="all_signals")
        m_dir = day_metrics(net[take_baseline_dir], name="dir_baseline")

        # Threshold sweep (per-fold)
        sweep_thresholds = [float(x) for x in args.threshold_sweep.split(",")]
        sweep_rows_this_fold = []
        for th in sweep_thresholds:
            take = proba >= th
            m = day_metrics(net[take], name=f"th{th:.2f}")
            sweep_rows_this_fold.append(dict(
                fold=t - W, oot_day=oot_name, threshold=th,
                n=m["n"], sharpe=m["sharpe"], pf=m["pf"], wr=m["wr"],
                sum_ticks=m["sum_ticks"], mean_ticks=m["mean_ticks"],
            ))
        # Append to a global sweep list (declared below before loop)
        sweep_all.extend(sweep_rows_this_fold)

        # Feature importance
        gi = booster.feature_importance(importance_type="gain")
        for c, g in zip(feat_cols, gi):
            importance_accum[c] += float(g)
        n_folds_trained += 1

        row = dict(
            fold=t - W,
            oot_day=oot_name,
            train_days=f"{days[t-W][0]}..{days[t-1][0]}",
            n_train=int(len(y_train)),
            n_oot=int(len(y_oot)),
            base_rate_train=float(y_train.mean()),
            base_rate_oot=float(y_oot.mean()),
            **{f"gated_{k}": v for k, v in m_gated.items() if k != "name"},
            **{f"allsig_{k}": v for k, v in m_all.items() if k != "name"},
            **{f"dir_{k}": v for k, v in m_dir.items() if k != "name"},
        )
        metrics_rows.append(row)
        log_f.write(json.dumps(row) + "\n")
        log_f.flush()
        print(f"[fold {t-W:02d}] oot={oot_name} | "
              f"gated n={m_gated['n']:>5} Sh={m_gated['sharpe']:.2f} PF={m_gated['pf']:.2f} WR={m_gated['wr']:.2%} | "
              f"all n={m_all['n']:>5} Sh={m_all['sharpe']:.2f} PF={m_all['pf']:.2f} | "
              f"dir n={m_dir['n']:>5} Sh={m_dir['sharpe']:.2f} PF={m_dir['pf']:.2f}",
              flush=True)

    log_f.close()

    if not metrics_rows:
        print("[fatal] no folds trained", file=sys.stderr)
        sys.exit(2)

    df_m = pd.DataFrame(metrics_rows)
    df_m.to_csv(OUT_DIR / "per_day_metrics.csv", index=False)

    # Threshold sweep table
    if sweep_all:
        df_sw = pd.DataFrame(sweep_all)
        df_sw.to_csv(OUT_DIR / "threshold_sweep_per_day.csv", index=False)
        # Aggregated per threshold
        agg_sw = df_sw.groupby("threshold").agg(
            days_with_trades=("n", lambda s: int((s > 0).sum())),
            total_trades=("n", "sum"),
            total_net_ticks=("sum_ticks", "sum"),
            mean_per_day_sharpe=("sharpe", "mean"),
            median_per_day_sharpe=("sharpe", "median"),
        ).reset_index()
        agg_sw["total_net_dollars"] = agg_sw["total_net_ticks"] * ES_TICK_VALUE
        agg_sw.to_csv(OUT_DIR / "threshold_sweep_summary.csv", index=False)

    # Feature importance (gain averaged across folds)
    imp = sorted(importance_accum.items(), key=lambda x: -x[1])
    pd.DataFrame(imp, columns=["feature", "total_gain"]).to_csv(OUT_DIR / "feature_importance.csv", index=False)

    # Aggregate verdict
    def agg(col_prefix):
        sub = df_m[df_m[f"{col_prefix}_n"] > 0]
        if len(sub) == 0:
            return dict(days=0, total_trades=0, total_net_ticks=0.0, total_net_dollars=0.0,
                        mean_per_day_sharpe=float("nan"), median_per_day_sharpe=float("nan"),
                        mean_per_day_pf=float("nan"), weighted_wr=float("nan"))
        sums = sub[f"{col_prefix}_sum_ticks"].sum()
        ns = sub[f"{col_prefix}_n"].sum()
        mean_sh = sub[f"{col_prefix}_sharpe"].mean()
        med_sh = sub[f"{col_prefix}_sharpe"].median()
        mean_pf = sub[f"{col_prefix}_pf"].replace(999.0, np.nan).mean()
        wr_w = (sub[f"{col_prefix}_wr"] * sub[f"{col_prefix}_n"]).sum() / max(ns, 1)
        return dict(
            days=int(len(sub)),
            total_trades=int(ns),
            total_net_ticks=float(sums),
            total_net_dollars=float(sums * ES_TICK_VALUE),
            mean_per_day_sharpe=float(mean_sh),
            median_per_day_sharpe=float(med_sh),
            mean_per_day_pf=float(mean_pf) if not pd.isna(mean_pf) else None,
            weighted_wr=float(wr_w),
        )

    verdict = dict(
        config=cfg,
        n_folds_trained=n_folds_trained,
        target=args.target,
        gate_threshold=args.gate_threshold,
        gated=agg("gated"),
        all_signals=agg("allsig"),
        dir_baseline=agg("dir"),
        top10_features=imp[:10],
    )
    (OUT_DIR / "verdict.json").write_text(json.dumps(verdict, indent=2, default=str))

    # Markdown verdict
    g = verdict["gated"]
    a = verdict["all_signals"]
    d = verdict["dir_baseline"]
    md = f"""# LightGBM Meta-Gate v1 -- Verdict (HC #486 R4 prototype)

Run finished {time.strftime('%Y-%m-%d %H:%M:%S')}.
Folds trained: **{n_folds_trained}** (sliding {args.train_window_days}-day train, 1-day OOT, slide by 1)
Target: `{args.target}` (1 if FIFO-net > 0)
Gate threshold: P(profitable) >= **{args.gate_threshold}**

## Summary table

| Mode              | Days | Trades  | Net ticks | $        | Mean per-day Sh | Median Sh | Mean PF | WR    |
|-------------------|------|---------|-----------|----------|-----------------|-----------|---------|-------|
| Gated (LGBM)      | {g['days']:>4} | {g['total_trades']:>7,} | {g['total_net_ticks']:>9.1f} | {g['total_net_dollars']:>8.0f} | {g['mean_per_day_sharpe']:>15.3f} | {g['median_per_day_sharpe']:>9.3f} | {g['mean_per_day_pf']:>7.3f} | {g['weighted_wr']:>5.1%} |
| All signals       | {a['days']:>4} | {a['total_trades']:>7,} | {a['total_net_ticks']:>9.1f} | {a['total_net_dollars']:>8.0f} | {a['mean_per_day_sharpe']:>15.3f} | {a['median_per_day_sharpe']:>9.3f} | {a['mean_per_day_pf']:>7.3f} | {a['weighted_wr']:>5.1%} |
| Direction baseline| {d['days']:>4} | {d['total_trades']:>7,} | {d['total_net_ticks']:>9.1f} | {d['total_net_dollars']:>8.0f} | {d['mean_per_day_sharpe']:>15.3f} | {d['median_per_day_sharpe']:>9.3f} | {d['mean_per_day_pf']:>7.3f} | {d['weighted_wr']:>5.1%} |

## Top-10 meta-gate features (gain-summed across folds)

""" + "\n".join(f"- `{f}` -- gain {g_:.0f}" for f, g_ in imp[:10]) + """

## Verdict
"""
    # Accept/reject heuristic
    acceptable = (
        g["mean_per_day_sharpe"] > a["mean_per_day_sharpe"]
        and g["mean_per_day_sharpe"] > 0
        and g["total_net_ticks"] > 0
    )
    md += f"\n**{'ACCEPT' if acceptable else 'REJECT'}** -- Gated mode "
    md += "improves on the take-all baseline." if acceptable else "does not beat the take-all baseline."
    md += "\n\nNext steps: regime-stratified breakdown (HC #428 R1) and threshold sweep.\n"

    (OUT_DIR / "verdict.md").write_text(md)
    print("\n" + md)
    print(f"[done] outputs -> {OUT_DIR}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        sys.exit(99)
