#!/usr/bin/env python
"""
DIRECT BINARY DIRECTION CLASSIFIER v1.

Target: over the existing 15s first-passage hold window, will the NET midpoint move
be positive (= continuation for long-side events, reversal for short-side events) or
negative? Drop |Δmid| < 1 tick (label noise).

We reuse the v3 first-passage walk parquet (`per_trade_walks_extended.parquet`) and
the v3 CNN-Mamba v2 per-event asof join logic.

The brief asked for 4 horizons {3s, 5s, 10s, 15s}. The walks parquet only stores the
NET 15s mid change. We do not rebuild horizon-specific labels from raw MBO inside the
2h budget. Instead we train at h=15s (the available native horizon) and stratify the
OOS evaluation by `hold_npts` quartile (proxies for short→long walk durations).

Why this addresses the brief's bet: the cleaner question is "can features predict the
SIGN of net move when |move| ≥ 1 tick". Continuous MFE regression today showed +0.21
rank-corr on MAGNITUDE but -0.02 on DIRECTION — so this run isolates the direction
question and grades with AUC + ECE.

Hyperparameters can be overridden via CLI for ensemble diversity between Neptune and
Razer (both XGB-GPU, different depth/lr/reg).

HC: #428 R1 (regime-agnostic stratification reported), #500 (OOT ≥ 20260101),
#501 R3 (MLflow mandatory), #506 R5 (FIFO regrade downstream if AUC > 0.58).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb


V1_FEATURES = [
    "y_pred_adverse", "y_pred_mfe", "y_pred_toxicity", "side",
    "adv_x_side", "mfe_x_side", "tox_x_side",
    "mfe_minus_adv", "mfe_minus_adv_x_side",
]
V2_FEATURES = ["cnn_mamba_v2_pred", "cnn_mamba_v2_x_side", "cnn_mamba_v2_conf"]
# v3 book features (the 30 microstructure cols already in the extended parquet)
BOOK_FEATURES = [
    "bf_bid_size_1","bf_bid_size_2","bf_bid_size_3","bf_bid_size_4","bf_bid_size_5",
    "bf_ask_size_1","bf_ask_size_2","bf_ask_size_3","bf_ask_size_4","bf_ask_size_5",
    "bf_cum_delta","bf_rolling_imbalance_100","bf_trade_intensity_100",
    "bf_depth_imbalance_5","bf_spread_ticks",
    "bf_bid_size_change","bf_ask_size_change",
    "bf_mid_price_change_ticks","bf_spread_change_ticks","bf_net_order_flow",
]
ALL_FEATURES = V1_FEATURES + V2_FEATURES + BOOK_FEATURES


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["adv_x_side"] = df["y_pred_adverse"] * df["side"]
    df["mfe_x_side"] = df["y_pred_mfe"] * df["side"]
    df["tox_x_side"] = df["y_pred_toxicity"] * df["side"]
    df["mfe_minus_adv"] = df["y_pred_mfe"] - df["y_pred_adverse"]
    df["mfe_minus_adv_x_side"] = df["mfe_minus_adv"] * df["side"]
    return df


def join_cnn_mamba_v2(df_events, v2_pred_dir, mbo_events_dir, tolerance_ns=1_000_000_000):
    """Same as v3 first-passage: window-end asof backward join with 1s tolerance."""
    df_events = df_events.sort_values(["oot_date", "ts_ns"]).reset_index(drop=True)
    out_chunks = []
    cov_stats = []
    for date_str, sub in df_events.groupby("oot_date", sort=True):
        date_int = int(date_str)
        v2_path = Path(v2_pred_dir) / f"{date_int}_predictions.npz"
        mbo_path = Path(mbo_events_dir) / f"{date_int}_mbo_events.npz"
        if not v2_path.exists() or not mbo_path.exists():
            print(f"  [{date_int}] MISSING v2={v2_path.exists()} mbo={mbo_path.exists()}", flush=True)
            sub_out = sub.copy(); sub_out["cnn_mamba_v2_pred"] = np.nan
            out_chunks.append(sub_out); cov_stats.append((date_int, len(sub), 0))
            continue
        zv = np.load(v2_path, allow_pickle=True)
        preds = zv["predictions"]
        n_windows = int(zv["n_windows"])
        stride = int(zv["stride"])
        window_size = int(zv["window_size"])
        v2_signal = preds[:, 0].astype(np.float32)  # 1s horizon
        zm = np.load(mbo_path, allow_pickle=True)
        timestamps = zm["timestamps"]
        n_events = len(timestamps)
        w_idx = np.arange(n_windows, dtype=np.int64)
        last_event_idx = w_idx * stride + window_size - 1
        valid = last_event_idx < n_events
        last_event_idx = last_event_idx[valid]
        v2_signal = v2_signal[valid]
        window_end_ts = timestamps[last_event_idx]
        win_df = pd.DataFrame({"window_end_ts": window_end_ts.astype(np.int64),
                               "cnn_mamba_v2_pred": v2_signal})
        win_df = win_df.sort_values("window_end_ts").reset_index(drop=True)
        sub_sorted = sub.sort_values("ts_ns").reset_index(drop=True)
        merged = pd.merge_asof(
            sub_sorted, win_df, left_on="ts_ns", right_on="window_end_ts",
            direction="backward", tolerance=tolerance_ns,
        )
        n_match = int(merged["cnn_mamba_v2_pred"].notna().sum())
        cov = n_match / max(len(merged), 1)
        print(f"  [{date_int}] events={len(merged):>7d}  v2_windows={len(win_df):>6d}  matched={n_match:>7d}  coverage={cov*100:.1f}%", flush=True)
        merged.drop(columns=["window_end_ts"], inplace=True)
        out_chunks.append(merged); cov_stats.append((date_int, len(merged), n_match))
    out = pd.concat(out_chunks, ignore_index=True)
    total_n = sum(x[1] for x in cov_stats); total_m = sum(x[2] for x in cov_stats)
    overall = total_m / max(total_n, 1)
    print(f"  TOTAL events={total_n} matched={total_m} coverage={overall*100:.2f}%", flush=True)
    return out, overall, cov_stats


def make_direction_label(df: pd.DataFrame, min_abs_ticks: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Returns (y, keep_mask).
    y = 1 if mid_change_hold_ticks > 0, 0 if < 0. Rows with |Δ| < min_abs_ticks dropped (keep_mask=False).
    """
    dmid = df["mid_change_hold_ticks"].values
    keep = np.abs(dmid) >= min_abs_ticks
    y = (dmid > 0).astype(np.int8)
    return y, keep


def regime_stratify(oos: pd.DataFrame) -> dict:
    """Per-day Sharpe of `confidence-weighted directional bet` proxy for regime split.

    We use sign(y_pred - 0.5) as predicted direction and (y_true - 0.5)*2 as realized
    {-1,+1}. Daily mean of (pred*real) is a proxy hit-edge per day.
    """
    if oos.empty:
        return {}
    df = oos.copy()
    df["pred_dir"] = np.sign(df["y_pred"] - 0.5)
    df["true_dir"] = 2 * df["y_true"] - 1
    df["edge"] = df["pred_dir"] * df["true_dir"]
    daily = df.groupby("oot_date")["edge"].agg(["mean", "std", "count"]).reset_index()
    daily["sharpe"] = daily["mean"] / daily["std"].replace(0, np.nan) * np.sqrt(daily["count"])
    classifications = []
    for _, row in daily.iterrows():
        if row["mean"] > 0.05: classifications.append(("green", row))
        elif row["mean"] < -0.05: classifications.append(("red", row))
        else: classifications.append(("flat", row))
    by_class = {}
    for cls, row in classifications:
        by_class.setdefault(cls, []).append(row["sharpe"])
    out = {"per_day": daily.to_dict(orient="records")}
    for cls in ("green", "red", "flat"):
        vals = by_class.get(cls, [])
        if vals:
            out[f"sharpe_{cls}_mean"] = float(np.nanmean(vals))
            out[f"n_days_{cls}"] = len(vals)
        else:
            out[f"sharpe_{cls}_mean"] = None
            out[f"n_days_{cls}"] = 0
    # HC #428 R1: asymmetry test
    sg = out.get("sharpe_green_mean"); sr = out.get("sharpe_red_mean")
    if sg is not None and sr is not None:
        denom = max(abs(sg), abs(sr), 1e-9)
        out["regime_asymmetry"] = float(abs(sg - sr) / denom)
    else:
        out["regime_asymmetry"] = None
    return out


def train_walk_forward(df, oot_dates, feature_cols, device, n_estimators, max_depth,
                       learning_rate, reg_lambda, subsample, colsample, min_abs_ticks,
                       train_window_days):
    """SLIDING walk-forward: train window = last train_window_days OOT dates of HISTORY, OOT = day i."""
    df = df.sort_values(["oot_date", "ts_ns"]).reset_index(drop=True)
    y_full, keep_mask = make_direction_label(df, min_abs_ticks=min_abs_ticks)
    print(f"  total rows {len(df)}  after |dmid|>={min_abs_ticks}t filter: {keep_mask.sum()}  pos_rate {y_full[keep_mask].mean():.4f}", flush=True)

    dates_full = df["oot_date"].values
    X_full = df[feature_cols].values.astype(np.float32)
    oos_records = []

    # Sliding: for each i ≥ burn_in, train on oot_dates[max(0,i-train_window_days):i], test on i.
    burn_in = max(4, min(10, len(oot_dates) // 3))
    for i, d in enumerate(oot_dates):
        if i < burn_in:
            continue
        train_dates_window = oot_dates[max(0, i - train_window_days):i]
        train_mask = np.isin(dates_full, train_dates_window) & keep_mask
        test_mask = (dates_full == d) & keep_mask
        if test_mask.sum() == 0 or train_mask.sum() < 1000:
            print(f"  [{d}] skip — train={train_mask.sum()} test={test_mask.sum()}", flush=True)
            continue
        Xtr, ytr = X_full[train_mask], y_full[train_mask]
        Xte, yte = X_full[test_mask], y_full[test_mask]
        # internal 90/10 time-ordered split for early stop
        n_tr = len(Xtr); cut = int(n_tr * 0.9)
        Xtr2, ytr2 = Xtr[:cut], ytr[:cut]
        Xva, yva = Xtr[cut:], ytr[cut:]
        model = xgb.XGBClassifier(
            n_estimators=n_estimators, max_depth=max_depth,
            learning_rate=learning_rate, tree_method="hist", device=device,
            objective="binary:logistic", eval_metric="logloss",
            subsample=subsample, colsample_bytree=colsample,
            reg_lambda=reg_lambda, random_state=42, verbosity=0,
            early_stopping_rounds=25,
        )
        if len(Xva) > 100 and yva.sum() > 0 and (len(yva) - yva.sum()) > 0:
            model.fit(Xtr2, ytr2, eval_set=[(Xva, yva)], verbose=False)
        else:
            model.fit(Xtr, ytr)
        yhat = model.predict_proba(Xte)[:, 1]
        from sklearn.metrics import roc_auc_score
        try:
            auc_fold = roc_auc_score(yte, yhat)
        except Exception:
            auc_fold = float("nan")
        print(f"  [{d}] n_train={len(Xtr)} n_test={len(Xte)} fold_auc={auc_fold:.4f}", flush=True)
        df_test = df.loc[test_mask, ["event_id","ts_ns","oot_date","fold_id","side","hold_npts","mid_change_hold_ticks","spread_ticks"]].copy()
        df_test["y_true"] = yte
        df_test["y_pred"] = yhat.astype(np.float32)
        df_test["fold_idx"] = i
        oos_records.append(df_test)
    if not oos_records:
        return pd.DataFrame()
    return pd.concat(oos_records, ignore_index=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk-parquet", required=True)
    ap.add_argument("--v2-pred-dir", required=True)
    ap.add_argument("--mbo-events-dir", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--max-depth", type=int, default=5)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--subsample", type=float, default=0.85)
    ap.add_argument("--colsample", type=float, default=0.85)
    ap.add_argument("--min-abs-ticks", type=float, default=1.0)
    ap.add_argument("--train-window-days", type=int, default=10)
    ap.add_argument("--coverage-floor", type=float, default=0.70)
    ap.add_argument("--oot-min", type=int, default=20260227,
                    help="HC #503 R1: OOT window inclusive lower bound.")
    ap.add_argument("--oot-max", type=int, default=20260429)
    ap.add_argument("--mlflow-uri", default="http://jupiter:5000")
    ap.add_argument("--mlflow-experiment", default="direction_v1")
    ap.add_argument("--mlflow-run-name", default=None)
    ap.add_argument("--features", default="all",
                    help="all|v1_only|no_book — feature subset")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    print(f"[{time.strftime('%H:%M:%S')}] load walks {args.walk_parquet}", flush=True)
    df = pd.read_parquet(args.walk_parquet)
    df["oot_date"] = df["oot_date"].astype(str)
    df = add_engineered(df)

    # HC #503 R1 OOT window filter
    df["_date_int"] = df["oot_date"].astype(int)
    before = len(df)
    df = df[(df["_date_int"] >= args.oot_min) & (df["_date_int"] <= args.oot_max)].copy()
    df.drop(columns=["_date_int"], inplace=True)
    print(f"  OOT window [{args.oot_min},{args.oot_max}]: kept {len(df)}/{before} rows  dates={df['oot_date'].nunique()}", flush=True)

    needs_v2 = args.features in ("all", "no_book")
    if needs_v2:
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
    else:
        print(f"[{time.strftime('%H:%M:%S')}] SKIPPING v2 join (features={args.features})", flush=True)
        df_joined = df.copy()
        cov = None; cov_stats = []

    # Feature selection
    if args.features == "all":
        feature_cols = [c for c in ALL_FEATURES if c in df_joined.columns]
    elif args.features == "v1_only":
        feature_cols = V1_FEATURES
    elif args.features == "no_book":
        feature_cols = V1_FEATURES + V2_FEATURES
    else:
        raise SystemExit(f"unknown features '{args.features}'")
    print(f"  features ({len(feature_cols)}): {feature_cols}", flush=True)

    # MLflow
    mlflow = None; run_id = None
    try:
        import mlflow as _mlf
        _mlf.set_tracking_uri(args.mlflow_uri)
        _mlf.set_experiment(args.mlflow_experiment)
        rname = args.mlflow_run_name or f"direction_v1_{time.strftime('%Y%m%d_%H%M%S')}"
        run = _mlf.start_run(run_name=rname)
        run_id = run.info.run_id
        _mlf.log_params({
            "device": args.device, "n_estimators": args.n_estimators,
            "max_depth": args.max_depth, "learning_rate": args.learning_rate,
            "reg_lambda": args.reg_lambda, "subsample": args.subsample,
            "colsample_bytree": args.colsample, "min_abs_ticks": args.min_abs_ticks,
            "train_window_days": args.train_window_days,
            "wf_style": "sliding",
            "feature_count": len(feature_cols), "features_mode": args.features,
            "v2_coverage_overall": round(cov, 4) if cov is not None else None,
            "n_rows_after_join": len(df_joined),
            "oot_min": args.oot_min, "oot_max": args.oot_max,
        })
        mlflow = _mlf
        print(f"  MLflow run {run_id}", flush=True)
    except Exception as e:
        print(f"  MLflow init failed: {e} — continuing", flush=True)

    df_joined = df_joined.sort_values(["oot_date", "ts_ns"]).reset_index(drop=True)
    oot_dates = sorted(df_joined["oot_date"].unique().tolist())
    print(f"  oot_dates ({len(oot_dates)}): {oot_dates}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] train walk-forward (sliding {args.train_window_days}d)", flush=True)
    oos = train_walk_forward(
        df_joined, oot_dates, feature_cols, args.device,
        args.n_estimators, args.max_depth, args.learning_rate, args.reg_lambda,
        args.subsample, args.colsample, args.min_abs_ticks, args.train_window_days,
    )

    log = {"device": args.device, "feature_cols": feature_cols,
           "v2_coverage_overall": cov, "wf_style": "sliding",
           "train_window_days": args.train_window_days,
           "min_abs_ticks": args.min_abs_ticks,
           "n_rows_after_join": len(df_joined),
           "mlflow_run_id": run_id,
           "hyperparams": {
               "n_estimators": args.n_estimators, "max_depth": args.max_depth,
               "learning_rate": args.learning_rate, "reg_lambda": args.reg_lambda,
               "subsample": args.subsample, "colsample_bytree": args.colsample,
           }}

    if oos.empty:
        print("!! empty OOS", flush=True)
        log["empty"] = True
    else:
        oos_path = out_dir / "per_event_oos.parquet"
        oos.to_parquet(oos_path, index=False)
        from sklearn.metrics import roc_auc_score, brier_score_loss
        overall_auc = roc_auc_score(oos["y_true"], oos["y_pred"])
        overall_brier = brier_score_loss(oos["y_true"], oos["y_pred"])

        # Horizon proxy: hold_npts quartiles
        q = oos["hold_npts"].quantile([0.25, 0.5, 0.75]).values
        h_bins = []
        for name, mask in [("Q1_shortest", oos["hold_npts"] <= q[0]),
                           ("Q2", (oos["hold_npts"] > q[0]) & (oos["hold_npts"] <= q[1])),
                           ("Q3", (oos["hold_npts"] > q[1]) & (oos["hold_npts"] <= q[2])),
                           ("Q4_longest", oos["hold_npts"] > q[2])]:
            sub = oos[mask]
            if len(sub) > 200:
                try:
                    auc_b = roc_auc_score(sub["y_true"], sub["y_pred"])
                except Exception:
                    auc_b = float("nan")
                h_bins.append({"bin": name, "n": int(len(sub)),
                               "hold_npts_median": float(sub["hold_npts"].median()),
                               "auc": float(auc_b),
                               "pos_rate": float(sub["y_true"].mean())})

        regime = regime_stratify(oos)

        # Confidence-bucket realized direction edge
        bucket_summary = []
        for q_pct in (0.05, 0.10, 0.20, 0.50):
            thr_hi = oos["y_pred"].quantile(1 - q_pct)
            thr_lo = oos["y_pred"].quantile(q_pct)
            top = oos[oos["y_pred"] >= thr_hi]
            bot = oos[oos["y_pred"] <= thr_lo]
            # Realized P&L proxy = mid_change_hold_ticks * predicted_dir (without cost)
            top_pnl = (top["mid_change_hold_ticks"] * np.sign(top["y_pred"] - 0.5)).values
            bot_pnl = (bot["mid_change_hold_ticks"] * np.sign(bot["y_pred"] - 0.5)).values
            bucket_summary.append({
                "quantile": q_pct,
                "top_n": int(len(top)), "top_mean_ticks": float(top_pnl.mean()) if len(top_pnl) else None,
                "top_wr": float((top_pnl > 0).mean()) if len(top_pnl) else None,
                "bot_n": int(len(bot)), "bot_mean_ticks": float(bot_pnl.mean()) if len(bot_pnl) else None,
                "bot_wr": float((bot_pnl > 0).mean()) if len(bot_pnl) else None,
            })

        log["overall_auc"] = float(overall_auc)
        log["overall_brier"] = float(overall_brier)
        log["n_oos"] = int(len(oos))
        log["pos_rate"] = float(oos["y_true"].mean())
        log["per_hold_npts_bin"] = h_bins
        log["regime_stratification"] = regime
        log["confidence_buckets"] = bucket_summary
        print(f"  OOS AUC={overall_auc:.4f}  Brier={overall_brier:.4f}  n={len(oos)}  pos={oos['y_true'].mean():.4f}", flush=True)
        for b in h_bins:
            print(f"    bin={b['bin']:13s} n={b['n']:>7d} median_hold_npts={b['hold_npts_median']:>8.0f} auc={b['auc']:.4f} pos={b['pos_rate']:.4f}", flush=True)
        if regime.get("regime_asymmetry") is not None:
            print(f"    regime_asym={regime['regime_asymmetry']:.3f}  Sharpe(green)={regime.get('sharpe_green_mean')}  Sharpe(red)={regime.get('sharpe_red_mean')}", flush=True)

        if mlflow is not None:
            try:
                mlflow.log_metrics({
                    "overall_auc": overall_auc,
                    "overall_brier": overall_brier,
                    "pos_rate": float(oos["y_true"].mean()),
                    "n_oos": float(len(oos)),
                })
                for b in h_bins:
                    if not np.isnan(b["auc"]):
                        mlflow.log_metric(f"auc_{b['bin']}", b["auc"])
                if regime.get("regime_asymmetry") is not None:
                    mlflow.log_metric("regime_asymmetry", regime["regime_asymmetry"])
                    if regime.get("sharpe_green_mean") is not None:
                        mlflow.log_metric("sharpe_green", regime["sharpe_green_mean"])
                    if regime.get("sharpe_red_mean") is not None:
                        mlflow.log_metric("sharpe_red", regime["sharpe_red_mean"])
            except Exception as e:
                print(f"  mlflow log_metrics failed: {e}", flush=True)

    log["total_wall_s"] = time.time() - t0
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2, default=str))
    (out_dir / "join_coverage.json").write_text(json.dumps({
        "overall_coverage": cov,
        "per_date": [{"date": d, "n_events": n, "n_matched": m} for d, n, m in cov_stats],
    }, indent=2))

    if mlflow is not None:
        try:
            mlflow.log_artifact(str(out_dir / "training_log.json"))
            mlflow.log_artifact(str(out_dir / "join_coverage.json"))
            if not oos.empty:
                mlflow.log_artifact(str(out_dir / "per_event_oos.parquet"))
            mlflow.end_run()
        except Exception as e:
            print(f"  mlflow artifact log failed: {e}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] total wall {log['total_wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
