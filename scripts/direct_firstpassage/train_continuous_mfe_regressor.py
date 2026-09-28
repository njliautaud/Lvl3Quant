#!/usr/bin/env python
"""
Continuous MFE Regressor v1 — replaces direct_firstpassage K-framework.

Predicts continuous expected MFE (signed by side) per trade. Per-horizon model:
since walks parquet has only ONE realized MFE per trade (with variable hold_npts),
we bucket trades by hold_npts quartile (proxy for horizon) AND train one global
model targeting signed_mid_change_at_hold. Per-bucket regrade is downstream.

Targets (two heads, in parallel):
  - signed_terminal:   mid_change_hold_ticks * side  (realistic P&L at hold horizon)
  - mfe_magnitude:     y_true_mfe                    (unsigned max favorable excursion)

Features (12, same as v3): 9 v1 features + 3 cnn_mamba_v2 features.
WF: expanding-bucket, burn_in=4, canonical OOT [20260227..20260429] subset present.
XGB GPU regressor, n_est=600, lr=0.05, max_depth=5, early_stopping=20,
objective=reg:squarederror.

HC: #500 (train data ≥20260101), #503 R1 (OOT in [20260227,20260429]), #504 (Neptune-only),
    #506 R5 (regrade gates downstream), #428 (regime-aware downstream).
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
ALL_FEATURES = V1_FEATURES + V2_FEATURES


def add_engineered(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["adv_x_side"] = df["y_pred_adverse"] * df["side"]
    df["mfe_x_side"] = df["y_pred_mfe"] * df["side"]
    df["tox_x_side"] = df["y_pred_toxicity"] * df["side"]
    df["mfe_minus_adv"] = df["y_pred_mfe"] - df["y_pred_adverse"]
    df["mfe_minus_adv_x_side"] = df["mfe_minus_adv"] * df["side"]
    return df


def join_cnn_mamba_v2(df_events, v2_pred_dir, mbo_events_dir, tolerance_ns=1_000_000_000):
    """Same join as v3. Returns (joined_df, overall_coverage, per_date_stats)."""
    df_events = df_events.sort_values(["oot_date", "ts_ns"]).reset_index(drop=True)
    out_chunks, coverage_stats = [], []
    for date_str, sub in df_events.groupby("oot_date", sort=True):
        date_int = int(date_str)
        v2_path = v2_pred_dir / f"{date_int}_predictions.npz"
        mbo_path = mbo_events_dir / f"{date_int}_mbo_events.npz"
        if not v2_path.exists() or not mbo_path.exists():
            print(f"  [{date_int}] MISSING — v2={v2_path.exists()} mbo={mbo_path.exists()}", flush=True)
            sub_out = sub.copy()
            sub_out["cnn_mamba_v2_pred"] = np.nan
            out_chunks.append(sub_out)
            coverage_stats.append((date_int, len(sub), 0))
            continue
        zv = np.load(v2_path, allow_pickle=True)
        preds = zv["predictions"]
        n_windows = int(zv["n_windows"])
        stride = int(zv["stride"])
        window_size = int(zv["window_size"])
        v2_signal = preds[:, 0].astype(np.float32)
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
            sub_sorted, win_df,
            left_on="ts_ns", right_on="window_end_ts",
            direction="backward", tolerance=tolerance_ns,
        )
        n_match = int(merged["cnn_mamba_v2_pred"].notna().sum())
        cov = n_match / max(len(merged), 1)
        print(f"  [{date_int}] events={len(merged):>7d}  matched={n_match:>7d}  cov={cov*100:.1f}%", flush=True)
        merged.drop(columns=["window_end_ts"], inplace=True)
        out_chunks.append(merged)
        coverage_stats.append((date_int, len(merged), n_match))
    out = pd.concat(out_chunks, ignore_index=True)
    tot_e = sum(x[1] for x in coverage_stats)
    tot_m = sum(x[2] for x in coverage_stats)
    overall = tot_m / max(tot_e, 1)
    print(f"\n  TOTAL: events={tot_e}  matched={tot_m}  coverage={overall*100:.2f}%", flush=True)
    return out, overall, coverage_stats


def train_reg(df_full, target_col, oot_dates, device, feature_cols, n_estimators=600):
    """Expanding-bucket WF regression; burn_in=4."""
    y_full = df_full[target_col].values.astype(np.float32)
    X_full = df_full[feature_cols].values.astype(np.float32)
    dates_full = df_full["oot_date"].values
    oos_records = []
    burn_in = 4
    for i, d in enumerate(oot_dates):
        if i < burn_in:
            continue
        train_mask = np.isin(dates_full, oot_dates[:i])
        test_mask = dates_full == d
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue
        Xtr, ytr = X_full[train_mask], y_full[train_mask]
        Xte, yte = X_full[test_mask], y_full[test_mask]
        n_tr = len(Xtr)
        cut = int(n_tr * 0.9)
        Xtr2, ytr2 = Xtr[:cut], ytr[:cut]
        Xva, yva = Xtr[cut:], ytr[cut:]
        model = xgb.XGBRegressor(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.05,
            tree_method="hist",
            device=device,
            objective="reg:squarederror",
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=20,
        )
        if len(Xva) > 100:
            model.fit(Xtr2, ytr2, eval_set=[(Xva, yva)], verbose=False)
        else:
            model.fit(Xtr, ytr)
        yhat = model.predict(Xte)
        df_test = df_full.loc[test_mask, ["event_id","ts_ns","oot_date","fold_id","side","hold_npts"]].copy()
        df_test[f"y_true_{target_col}"] = yte
        df_test[f"y_pred_{target_col}"] = yhat.astype(np.float32)
        df_test["fold_oot_idx"] = i
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
    ap.add_argument("--coverage-floor", type=float, default=0.70)
    ap.add_argument("--mlflow-uri", default="http://jupiter:5000")
    ap.add_argument("--mlflow-experiment", default="continuous_mfe_regressor_v1")
    ap.add_argument("--oot-start", default="20260227")
    ap.add_argument("--oot-end",   default="20260429")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading {args.walk_parquet}...", flush=True)
    df = pd.read_parquet(args.walk_parquet)
    df = add_engineered(df)
    df["oot_date"] = df["oot_date"].astype(str)
    print(f"  pre-filter rows={len(df)} dates={df['oot_date'].nunique()}", flush=True)

    # HC #503 R1: OOT window
    df = df[(df["oot_date"] >= args.oot_start) & (df["oot_date"] <= args.oot_end)]
    print(f"  after OOT window [{args.oot_start}..{args.oot_end}]: rows={len(df)}", flush=True)

    # Drop rows with NA terminal move (can't compute target)
    df = df.dropna(subset=["mid_change_hold_ticks"]).reset_index(drop=True)
    print(f"  after NA-drop on mid_change_hold_ticks: rows={len(df)}", flush=True)

    # Targets
    df["signed_terminal"] = df["mid_change_hold_ticks"].astype("float32") * df["side"].astype("float32")
    df["mfe_magnitude"] = df["y_true_mfe"].astype("float32")
    # Also useful: signed MFE = mfe_magnitude in side-favored direction (it's defined as max favorable
    # excursion in the side direction, so positive numbers already favor side). We treat magnitude as our target.

    print(f"[{time.strftime('%H:%M:%S')}] joining CNN-Mamba v2...", flush=True)
    df_j, cov, cov_stats = join_cnn_mamba_v2(
        df, Path(args.v2_pred_dir), Path(args.mbo_events_dir), tolerance_ns=1_000_000_000,
    )
    pre = len(df_j)
    df_j = df_j[df_j["cnn_mamba_v2_pred"].notna()].reset_index(drop=True)
    print(f"  dropped {pre-len(df_j)} unjoined; kept {len(df_j)}", flush=True)

    fallback_v1_only = False
    if cov < args.coverage_floor:
        print(f"!! coverage {cov:.3f} < floor; FALLBACK to v1-only features", flush=True)
        fallback_v1_only = True

    if fallback_v1_only:
        feature_cols = V1_FEATURES
        # Use original df without v2 join, refill NA on v2 to allow regression
        df_j = df  # use the unjoined df
    else:
        df_j["cnn_mamba_v2_x_side"] = df_j["cnn_mamba_v2_pred"] * df_j["side"]
        df_j["cnn_mamba_v2_conf"] = np.abs(df_j["cnn_mamba_v2_pred"])
        feature_cols = ALL_FEATURES

    print(f"  features ({len(feature_cols)}): {feature_cols}", flush=True)

    # MLflow init
    mlflow = None
    run_id = None
    try:
        import mlflow as _mlf
        _mlf.set_tracking_uri(args.mlflow_uri)
        _mlf.set_experiment(args.mlflow_experiment)
        run = _mlf.start_run(run_name=f"cmr_v1_{time.strftime('%Y%m%d_%H%M%S')}")
        run_id = run.info.run_id
        _mlf.log_params({
            "n_estimators": args.n_estimators, "device": args.device,
            "feature_count": len(feature_cols),
            "v2_coverage_overall": round(cov, 4),
            "n_rows": len(df_j),
            "fallback_v1_only": fallback_v1_only,
            "wf_style": "expanding_bucket",
            "target_primary": "signed_terminal",
            "target_secondary": "mfe_magnitude",
            "oot_start": args.oot_start,
            "oot_end": args.oot_end,
        })
        mlflow = _mlf
        print(f"  MLflow run: {run_id}", flush=True)
    except Exception as e:
        print(f"  MLflow init failed: {e} — continuing", flush=True)

    df_j = df_j.sort_values(["oot_date","ts_ns"]).reset_index(drop=True)
    oot_dates = sorted(df_j["oot_date"].unique().tolist())
    print(f"  oot_dates ({len(oot_dates)}): {oot_dates}", flush=True)

    log = {"feature_cols": feature_cols, "wf_style":"expanding_bucket",
           "v2_coverage_overall": cov, "fallback_v1_only": fallback_v1_only,
           "n_rows": len(df_j), "mlflow_run_id": run_id, "targets": []}

    all_outs = {}
    for tgt in ["signed_terminal", "mfe_magnitude"]:
        t_h = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] === target {tgt} ===", flush=True)
        oos = train_reg(df_j, tgt, oot_dates, args.device, feature_cols, n_estimators=args.n_estimators)
        if oos.empty:
            print(f"  EMPTY oos for {tgt}", flush=True)
            continue
        # Quick metrics
        from scipy.stats import spearmanr
        sp = spearmanr(oos[f"y_pred_{tgt}"], oos[f"y_true_{tgt}"]).correlation
        rmse = float(np.sqrt(((oos[f"y_pred_{tgt}"] - oos[f"y_true_{tgt}"])**2).mean()))
        wall = time.time() - t_h
        print(f"  done {wall:.1f}s  n_oos={len(oos)}  spearman={sp:.4f}  rmse={rmse:.3f}", flush=True)
        log["targets"].append({"target": tgt, "n_oos": int(len(oos)),
                               "spearman": float(sp), "rmse": rmse, "wall_s": wall})
        if mlflow is not None:
            try:
                mlflow.log_metrics({f"spearman_{tgt}": sp, f"rmse_{tgt}": rmse})
            except Exception:
                pass
        all_outs[tgt] = oos

    # Merge both targets into single per-trade frame
    if "signed_terminal" in all_outs and "mfe_magnitude" in all_outs:
        a = all_outs["signed_terminal"]
        b = all_outs["mfe_magnitude"][["event_id","ts_ns","oot_date","side",
                                       "y_true_mfe_magnitude","y_pred_mfe_magnitude"]]
        merged = a.merge(b, on=["event_id","ts_ns","oot_date","side"], how="inner")
        merged.to_parquet(out_dir / "per_trade_oos.parquet", index=False)
        print(f"  wrote per_trade_oos.parquet rows={len(merged)}", flush=True)
    elif all_outs:
        for k,v in all_outs.items():
            v.to_parquet(out_dir / f"per_trade_oos_{k}.parquet", index=False)

    log["total_wall_s"] = time.time() - t0
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    (out_dir / "join_coverage.json").write_text(json.dumps({
        "overall_coverage": cov, "fallback_v1_only": fallback_v1_only,
        "per_date": [{"date": d, "n_events": n, "n_matched": m} for d,n,m in cov_stats],
    }, indent=2))

    if mlflow is not None:
        try:
            mlflow.log_artifact(str(out_dir / "training_log.json"))
            mlflow.log_artifact(str(out_dir / "join_coverage.json"))
            if (out_dir / "per_trade_oos.parquet").exists():
                mlflow.log_artifact(str(out_dir / "per_trade_oos.parquet"))
            mlflow.end_run()
        except Exception as e:
            print(f"  MLflow log_artifact failed: {e}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] total wall {log['total_wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
