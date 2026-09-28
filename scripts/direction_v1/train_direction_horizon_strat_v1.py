#!/usr/bin/env python
"""
HORIZON-STRATIFIED DIRECTION CLASSIFIER v1 (Razer GPU XGB).

Tests the brief hypothesis: direction edge concentrates at the shortest realized
first-passage horizons. We train ONE XGB binary direction classifier PER bucket
on the same v1 + CNN-Mamba-v2-join 12-feature set used by direction_v1.

Bucketing axis: realized first-passage time (FPT) = min(tp1_dt_ns, sl1_dt_ns)/1e9.
Brief asked for {<5s, 5-15s, >=15s} but ~99% of walks hit a 1-tick barrier in
<5s. We translate to the empirical log-scale equivalent and document:
  - h_short:  FPT < 0.1s
  - h_mid:    0.1s <= FPT < 1.0s
  - h_long:   FPT >= 1.0s   (includes longest-walk tail)

Reuses train_direction_v1.add_engineered, join_cnn_mamba_v2, train_walk_forward.

HC: #428 R1 + R2 (regime-agnostic + within-horizon), #500 OOT >= 20260317,
    #501 R3 MLflow, #504 (no SSM), #506 R5 (FIFO regrade gate).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

# import sibling module
sys.path.insert(0, str(Path(__file__).parent))
from train_direction_v1 import (
    add_engineered, join_cnn_mamba_v2, train_walk_forward,
    V1_FEATURES, V2_FEATURES,
)


BUCKETS = [
    ("h_short",  0.0,  0.1),
    ("h_mid",    0.1,  1.0),
    ("h_long",   1.0,  float("inf")),
]


def compute_fpt_seconds(df: pd.DataFrame) -> np.ndarray:
    tp = df["tp1_dt_ns"].values
    sl = df["sl1_dt_ns"].values
    tp_f = np.where(np.isnan(tp), np.inf, tp)
    sl_f = np.where(np.isnan(sl), np.inf, sl)
    fpt = np.minimum(tp_f, sl_f) / 1e9
    fpt = np.clip(fpt, 0.0, None)
    return fpt


def conditional_metrics_at_top_q(oos: pd.DataFrame, top_qs=(0.01, 0.05, 0.10)):
    """Top-q gate: take |pred-0.5| top-q most-confident events, P&L = sign(pred-0.5) * dmid.
    Returns list of {q, n, mean_ticks_net (after BE 1.376t passive? -> we report gross),
    wr, mean_ticks_gross}."""
    out = []
    if len(oos) == 0:
        return out
    conf = (oos["y_pred"] - 0.5).abs().values
    pnl_gross = (oos["mid_change_hold_ticks"] * np.sign(oos["y_pred"] - 0.5)).values
    for q in top_qs:
        thr = np.quantile(conf, 1 - q)
        sel = conf >= thr
        n = int(sel.sum())
        if n == 0:
            out.append({"q": q, "n": 0, "wr": None, "mean_ticks_gross": None,
                        "mean_ticks_net_taker": None})
            continue
        wr = float((pnl_gross[sel] > 0).mean())
        mg = float(pnl_gross[sel].mean())
        # Taker cost: commission 0.376t + 1.0t cross = 1.376t per round-trip
        mn_taker = mg - 1.376
        out.append({"q": q, "n": n, "wr": wr, "mean_ticks_gross": mg,
                    "mean_ticks_net_taker": mn_taker})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk-parquet", required=True)
    ap.add_argument("--v2-pred-dir", required=True)
    ap.add_argument("--mbo-events-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--subsample", type=float, default=0.85)
    ap.add_argument("--colsample", type=float, default=0.85)
    ap.add_argument("--min-abs-ticks", type=float, default=1.0)
    ap.add_argument("--train-window-days", type=int, default=10)
    ap.add_argument("--coverage-floor", type=float, default=0.70)
    ap.add_argument("--oot-min", type=int, default=20260317)
    ap.add_argument("--oot-max", type=int, default=20260429)
    ap.add_argument("--mlflow-uri", default="http://neptune-win:5000")
    ap.add_argument("--mlflow-experiment", default="razer_direction_horizon_strat_v1")
    ap.add_argument("--mlflow-run-name", default=None)
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[{time.strftime('%H:%M:%S')}] load walks {args.walk_parquet}", flush=True)
    df = pd.read_parquet(args.walk_parquet)
    df["oot_date"] = df["oot_date"].astype(str)
    df = add_engineered(df)
    df["fpt_s"] = compute_fpt_seconds(df)

    df["_date_int"] = df["oot_date"].astype(int)
    before = len(df)
    df = df[(df["_date_int"] >= args.oot_min) & (df["_date_int"] <= args.oot_max)].copy()
    df.drop(columns=["_date_int"], inplace=True)
    print(f"  OOT window [{args.oot_min},{args.oot_max}]: kept {len(df)}/{before} rows dates={df['oot_date'].nunique()}", flush=True)

    # Join CNN-Mamba v2 ONCE across all buckets
    print(f"[{time.strftime('%H:%M:%S')}] join CNN-Mamba v2 per-event", flush=True)
    df_joined, cov, cov_stats = join_cnn_mamba_v2(df, args.v2_pred_dir, args.mbo_events_dir)
    if cov < args.coverage_floor:
        print(f"!! join coverage {cov:.3f} < floor {args.coverage_floor}; abort", flush=True)
        (out_dir / "join_coverage.json").write_text(json.dumps({
            "overall_coverage": cov, "per_date": cov_stats, "aborted": True}, indent=2))
        sys.exit(2)
    pre = len(df_joined)
    df_joined = df_joined[df_joined["cnn_mamba_v2_pred"].notna()].reset_index(drop=True)
    print(f"  dropped {pre - len(df_joined)} unjoined; kept {len(df_joined)}", flush=True)
    df_joined["cnn_mamba_v2_x_side"] = df_joined["cnn_mamba_v2_pred"] * df_joined["side"]
    df_joined["cnn_mamba_v2_conf"]   = np.abs(df_joined["cnn_mamba_v2_pred"])

    feature_cols = V1_FEATURES + V2_FEATURES  # 9 + 3 = 12, the "no_book" set
    print(f"  features ({len(feature_cols)}): {feature_cols}", flush=True)

    # MLflow parent run
    mlflow = None; run_id = None
    try:
        import mlflow as _mlf
        _mlf.set_tracking_uri(args.mlflow_uri)
        _mlf.set_experiment(args.mlflow_experiment)
        rname = args.mlflow_run_name or f"horizon_strat_v1_{time.strftime('%Y%m%d_%H%M%S')}"
        run = _mlf.start_run(run_name=rname)
        run_id = run.info.run_id
        _mlf.log_params({
            "device": args.device, "n_estimators": args.n_estimators,
            "max_depth": args.max_depth, "learning_rate": args.learning_rate,
            "min_abs_ticks": args.min_abs_ticks,
            "train_window_days": args.train_window_days,
            "feature_count": len(feature_cols), "features_mode": "v1+v2_join_12f",
            "v2_coverage_overall": round(cov, 4),
            "oot_min": args.oot_min, "oot_max": args.oot_max,
            "bucket_axis": "first_passage_time_seconds",
            "buckets": str(BUCKETS),
        })
        mlflow = _mlf
        print(f"  MLflow run {run_id}", flush=True)
    except Exception as e:
        print(f"  MLflow init failed: {e} — continuing", flush=True)

    oot_dates = sorted(df_joined["oot_date"].unique().tolist())
    print(f"  oot_dates ({len(oot_dates)}): {oot_dates}", flush=True)

    bucket_results = []
    for bname, lo, hi in BUCKETS:
        if np.isinf(hi):
            mask = df_joined["fpt_s"] >= lo
        else:
            mask = (df_joined["fpt_s"] >= lo) & (df_joined["fpt_s"] < hi)
        sub = df_joined[mask].copy().reset_index(drop=True)
        n_total = len(sub)
        print(f"\n[{time.strftime('%H:%M:%S')}] === bucket {bname} (FPT in [{lo}, {hi})) n_total={n_total} ===", flush=True)
        if n_total < 3000:
            print(f"  skip — too few rows", flush=True)
            bucket_results.append({"bucket": bname, "n_total": n_total, "skipped": True})
            continue

        oos = train_walk_forward(
            sub, oot_dates, feature_cols, args.device,
            args.n_estimators, args.max_depth, args.learning_rate, args.reg_lambda,
            args.subsample, args.colsample, args.min_abs_ticks, args.train_window_days,
        )
        if oos.empty:
            print(f"  empty OOS for {bname}", flush=True)
            bucket_results.append({"bucket": bname, "n_total": n_total, "n_oos": 0, "empty": True})
            continue

        oos_path = out_dir / f"per_event_oos_{bname}.parquet"
        oos.to_parquet(oos_path, index=False)

        from sklearn.metrics import roc_auc_score, brier_score_loss
        try:
            overall_auc = float(roc_auc_score(oos["y_true"], oos["y_pred"]))
        except Exception:
            overall_auc = float("nan")
        overall_brier = float(brier_score_loss(oos["y_true"], oos["y_pred"]))

        # per-fold AUC
        fold_aucs = []
        for f_idx, gdf in oos.groupby("fold_idx"):
            if gdf["y_true"].nunique() > 1:
                try:
                    fold_aucs.append(float(roc_auc_score(gdf["y_true"], gdf["y_pred"])))
                except Exception:
                    pass
        fold_auc_mean = float(np.mean(fold_aucs)) if fold_aucs else float("nan")
        fold_auc_std  = float(np.std(fold_aucs)) if fold_aucs else float("nan")

        topq = conditional_metrics_at_top_q(oos)

        bucket_results.append({
            "bucket": bname, "lo_s": lo, "hi_s": hi,
            "n_total": int(n_total), "n_oos": int(len(oos)),
            "pos_rate": float(oos["y_true"].mean()),
            "overall_auc": overall_auc,
            "overall_brier": overall_brier,
            "fold_auc_mean": fold_auc_mean,
            "fold_auc_std": fold_auc_std,
            "n_folds": len(fold_aucs),
            "top_q_metrics": topq,
        })

        print(f"  bucket={bname} AUC={overall_auc:.4f} Brier={overall_brier:.4f} n_oos={len(oos)} pos={oos['y_true'].mean():.4f} fold_auc={fold_auc_mean:.4f}+-{fold_auc_std:.4f}", flush=True)
        for tq in topq:
            print(f"    top {tq['q']*100:>4.1f}% n={tq['n']:>6d} wr={tq['wr']} mean_gross_t={tq['mean_ticks_gross']} mean_net_taker_t={tq['mean_ticks_net_taker']}", flush=True)

        if mlflow is not None:
            try:
                prefix = bname
                mlflow.log_metrics({
                    f"{prefix}_auc": overall_auc,
                    f"{prefix}_brier": overall_brier,
                    f"{prefix}_n_oos": float(len(oos)),
                    f"{prefix}_pos_rate": float(oos["y_true"].mean()),
                    f"{prefix}_fold_auc_mean": fold_auc_mean,
                    f"{prefix}_fold_auc_std": fold_auc_std,
                })
                for tq in topq:
                    if tq["mean_ticks_gross"] is not None:
                        mlflow.log_metric(f"{prefix}_top{int(tq['q']*100)}pct_wr", tq["wr"])
                        mlflow.log_metric(f"{prefix}_top{int(tq['q']*100)}pct_net_taker_t", tq["mean_ticks_net_taker"])
            except Exception as e:
                print(f"  mlflow log_metrics failed: {e}", flush=True)

    # ---- verdict ----
    # ACCEPT criterion: any bucket with AUC >= 0.58 AND top-1% net_taker > 0
    accepted = []
    for r in bucket_results:
        if r.get("overall_auc", 0) and r["overall_auc"] >= 0.58:
            top1 = next((t for t in r.get("top_q_metrics", []) if abs(t["q"] - 0.01) < 1e-6), None)
            if top1 and top1["mean_ticks_net_taker"] is not None and top1["mean_ticks_net_taker"] > 0:
                accepted.append(r["bucket"])

    verdict = "GO — Jupiter label-rebuild justified" if accepted else "NO-GO — direction-at-short-horizon hypothesis dead too"

    log = {
        "device": args.device, "feature_cols": feature_cols,
        "v2_coverage_overall": cov,
        "wf_style": "sliding", "train_window_days": args.train_window_days,
        "min_abs_ticks": args.min_abs_ticks,
        "n_rows_after_join": len(df_joined),
        "mlflow_run_id": run_id,
        "buckets": BUCKETS,
        "bucket_axis": "first_passage_time_seconds  min(tp1,sl1)/1e9",
        "bucket_results": bucket_results,
        "accepted_buckets": accepted,
        "verdict": verdict,
        "hyperparams": {
            "n_estimators": args.n_estimators, "max_depth": args.max_depth,
            "learning_rate": args.learning_rate, "reg_lambda": args.reg_lambda,
            "subsample": args.subsample, "colsample_bytree": args.colsample,
        },
        "total_wall_s": time.time() - t0,
    }
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2, default=str))
    (out_dir / "join_coverage.json").write_text(json.dumps({
        "overall_coverage": cov,
        "per_date": [{"date": d, "n_events": n, "n_matched": m} for d, n, m in cov_stats],
    }, indent=2))

    # ---- verdict.md ----
    md = []
    md.append("# Razer Direction Horizon-Stratified v1 — Verdict\n")
    md.append(f"Run timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    md.append(f"OOT window: [{args.oot_min}, {args.oot_max}]  ({len(oot_dates)} dates)\n")
    md.append(f"Features (12): {feature_cols}\n")
    md.append(f"v2 join coverage: {cov*100:.2f}%\n")
    md.append("\n## Bucket axis\n")
    md.append("First-passage time (FPT) = min(tp1_dt_ns, sl1_dt_ns)/1e9.  ")
    md.append("Brief asked for {<5s, 5-15s, ≥15s} but ~99% of walks hit a 1-tick barrier in <5s.  ")
    md.append("Empirically log-scale-equivalent bins used:\n")
    md.append("  - h_short:  FPT <  0.1s\n  - h_mid:    0.1s ≤ FPT < 1.0s\n  - h_long:   FPT ≥ 1.0s\n")
    md.append("\n## Results table\n")
    md.append("| bucket | n_total | n_oos (after |Δmid|≥1t) | OOS AUC | OOS Brier | pos_rate | fold AUC mean ± std (n_folds) |\n")
    md.append("|---|---:|---:|---:|---:|---:|---|\n")
    for r in bucket_results:
        if r.get("skipped") or r.get("empty"):
            md.append(f"| {r['bucket']} | {r.get('n_total','?')} | — | — | — | — | (skipped/empty) |\n")
            continue
        md.append(f"| {r['bucket']} | {r['n_total']} | {r['n_oos']} | {r['overall_auc']:.4f} | {r['overall_brier']:.4f} | {r['pos_rate']:.4f} | {r['fold_auc_mean']:.4f} ± {r['fold_auc_std']:.4f}  (n={r['n_folds']}) |\n")
    md.append("\n## Top-q confidence regrade (gross ticks, and net after taker cost 1.376t round-trip)\n")
    md.append("BE for taker: WR > 57.6% at K=1 implied by 1.376t cost vs +/-1t move.\n\n")
    md.append("| bucket | q | n | WR | mean gross (ticks) | mean net taker (ticks) |\n")
    md.append("|---|---:|---:|---:|---:|---:|\n")
    for r in bucket_results:
        if r.get("skipped") or r.get("empty"):
            continue
        for tq in r.get("top_q_metrics", []):
            wr_s = f"{tq['wr']:.4f}" if tq["wr"] is not None else "—"
            mg_s = f"{tq['mean_ticks_gross']:+.4f}" if tq["mean_ticks_gross"] is not None else "—"
            mn_s = f"{tq['mean_ticks_net_taker']:+.4f}" if tq["mean_ticks_net_taker"] is not None else "—"
            md.append(f"| {r['bucket']} | {tq['q']*100:.1f}% | {tq['n']} | {wr_s} | {mg_s} | {mn_s} |\n")
    md.append(f"\n## Verdict\n\n**{verdict}**\n\n")
    md.append(f"Accepted buckets (AUC ≥ 0.58 AND top-1% net_taker > 0): `{accepted}`\n")
    md.append(f"\nMLflow run id: `{run_id}`  experiment: `{args.mlflow_experiment}`\n")
    md.append(f"Total wall: {log['total_wall_s']:.1f}s\n")
    (out_dir / "verdict.md").write_text("".join(md))

    if mlflow is not None:
        try:
            mlflow.log_metric("n_accepted_buckets", len(accepted))
            mlflow.log_param("verdict", verdict)
            mlflow.log_artifact(str(out_dir / "training_log.json"))
            mlflow.log_artifact(str(out_dir / "verdict.md"))
            mlflow.log_artifact(str(out_dir / "join_coverage.json"))
            mlflow.end_run()
        except Exception as e:
            print(f"  mlflow artifact log failed: {e}", flush=True)

    print(f"\n[{time.strftime('%H:%M:%S')}] DONE  wall={log['total_wall_s']:.1f}s  VERDICT: {verdict}", flush=True)


if __name__ == "__main__":
    main()
