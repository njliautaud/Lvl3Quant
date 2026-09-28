#!/usr/bin/env python
"""
Direct first-passage v3: v1's 9 features + per-event CNN-Mamba v2 prediction (Option B asof join).

Pipeline:
1. Load per_trade_walks_extended.parquet (v1 features + ts_ns per event).
2. For each oot_date in the parquet, load the matching cnn_mamba_v2 bulk_oot npz
   AND the underlying mbo_events npz (to recover per-window end timestamps).
3. Build a per-window dataframe (window_end_ts, v2_pred_1s) for that date.
4. merge_asof on ts_ns with tolerance=1.0s, direction='backward'.
5. Add v2_pred, v2_pred_x_side, v2_confidence (=|2p-1|). Coverage logged.
6. Drop events without v2 join (or impute median; choose drop for cleanliness).
7. Train 8 (K,S) heads with expanding-bucket WF on OOT dates (same as v1).
   Same feature set across all folds (no per-fold MI selection).
8. Write per_head_oos.parquet + training_log.json.

HC: #428 R2 (no architecture noise added), #506 R5 (will be regraded), #500 (≥20260101 data).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb

HEADS = [(2,1),(3,1),(4,1),(5,1),(3,2),(4,2),(5,2),(5,4)]
HOLD_CAP_NS = 15_000_000_000

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


def label_first_passage(df: pd.DataFrame, K: int, S: int, hold_cap_ns: int = HOLD_CAP_NS) -> np.ndarray:
    tp = df[f"tp{K}_dt_ns"].values
    sl = df[f"sl{S}_dt_ns"].values
    tp_hit = (tp > 0) & (tp <= hold_cap_ns)
    sl_hit = (sl > 0) & (sl <= hold_cap_ns)
    win = tp_hit & ((~sl_hit) | (tp < sl))
    return win.astype(np.int8)


def join_cnn_mamba_v2(
    df_events: pd.DataFrame,
    v2_pred_dir: Path,
    mbo_events_dir: Path,
    tolerance_ns: int = 1_000_000_000,
) -> pd.DataFrame:
    """
    For each oot_date in df_events, load:
      - {v2_pred_dir}/{date}_predictions.npz  → predictions[n_windows, 3], stride, window_size
      - {mbo_events_dir}/{date}_mbo_events.npz → timestamps[n_events]

    Map window index w → event index e = w*stride + window_size - 1 (last event in window).
    window_end_ts = timestamps[e].
    Use predictions[:,0] (1s horizon) as the scalar signal.

    Then merge_asof on ts_ns with tolerance, direction='backward'.
    Append columns: cnn_mamba_v2_pred (NaN if no join).
    """
    df_events = df_events.sort_values(["oot_date", "ts_ns"]).reset_index(drop=True)
    out_chunks = []
    coverage_stats = []

    for date_str, sub in df_events.groupby("oot_date", sort=True):
        date_int = int(date_str)
        v2_path = v2_pred_dir / f"{date_int}_predictions.npz"
        mbo_path = mbo_events_dir / f"{date_int}_mbo_events.npz"

        if not v2_path.exists() or not mbo_path.exists():
            print(f"  [{date_int}] MISSING — v2={v2_path.exists()} mbo={mbo_path.exists()}; skip-join", flush=True)
            sub_out = sub.copy()
            sub_out["cnn_mamba_v2_pred"] = np.nan
            out_chunks.append(sub_out)
            coverage_stats.append((date_int, len(sub), 0))
            continue

        zv = np.load(v2_path, allow_pickle=True)
        preds = zv["predictions"]   # [n_windows, 3]  cols = 1s, 5s, 10s
        n_windows = int(zv["n_windows"])
        stride = int(zv["stride"])
        window_size = int(zv["window_size"])

        # Use 1s horizon prediction (idx 0 per horizons array).
        v2_signal = preds[:, 0].astype(np.float32)

        # Load timestamps WITHOUT loading the events array (huge).
        zm = np.load(mbo_path, allow_pickle=True)
        timestamps = zm["timestamps"]  # int64
        n_events = len(timestamps)

        # Build per-window end timestamps.
        # window w covers events [w*stride : w*stride + window_size).
        # last event index = w*stride + window_size - 1.
        w_idx = np.arange(n_windows, dtype=np.int64)
        last_event_idx = w_idx * stride + window_size - 1
        # Some predictions may correspond to indices beyond the events array (unlikely but safe-guard).
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
        coverage = n_match / max(len(merged), 1)
        print(f"  [{date_int}] events={len(merged):>7d}  v2_windows={len(win_df):>6d}  matched={n_match:>7d}  coverage={coverage*100:.1f}%", flush=True)
        merged.drop(columns=["window_end_ts"], inplace=True)
        out_chunks.append(merged)
        coverage_stats.append((date_int, len(merged), n_match))

    out = pd.concat(out_chunks, ignore_index=True)
    total_events = sum(x[1] for x in coverage_stats)
    total_matched = sum(x[2] for x in coverage_stats)
    overall = total_matched / max(total_events, 1)
    print(f"\n  TOTAL: events={total_events}  matched={total_matched}  coverage={overall*100:.2f}%", flush=True)
    return out, overall, coverage_stats


def train_head(df_full: pd.DataFrame, K: int, S: int, oot_dates: list[str], device: str,
               feature_cols: list[str], n_estimators: int = 600,
               save_models_dir: Path | None = None) -> pd.DataFrame:
    """Expanding-bucket WF (v1 style). burn_in=4, train on all prior OOT rows, predict on oot_dates[i]."""
    y_full = label_first_passage(df_full, K, S)
    X_full = df_full[feature_cols].values.astype(np.float32)
    dates_full = df_full["oot_date"].values

    oos_records = []
    burn_in = 4
    pos_weight_cache = []

    for i, d in enumerate(oot_dates):
        if i < burn_in:
            continue
        train_mask = np.isin(dates_full, oot_dates[:i])
        test_mask = dates_full == d
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        Xtr, ytr = X_full[train_mask], y_full[train_mask]
        Xte, yte = X_full[test_mask], y_full[test_mask]

        pos = int(ytr.sum())
        neg = len(ytr) - pos
        spw = (neg / max(pos, 1)) if pos > 0 else 1.0
        pos_weight_cache.append(spw)

        # 90/10 internal split for early stopping
        n_tr = len(Xtr)
        cut = int(n_tr * 0.9)
        # Time-ordered split: use last 10% of training (already date-sorted via isin).
        # We rely on df_full being already sorted by (oot_date, ts_ns) before calling.
        Xtr2, ytr2 = Xtr[:cut], ytr[:cut]
        Xva, yva = Xtr[cut:], ytr[cut:]

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=5,
            learning_rate=0.05,
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            verbosity=0,
            early_stopping_rounds=20,
        )
        if len(Xva) > 100 and yva.sum() > 0 and (len(yva) - yva.sum()) > 0:
            model.fit(Xtr2, ytr2, eval_set=[(Xva, yva)], verbose=False)
        else:
            model.fit(Xtr, ytr)

        yhat = model.predict_proba(Xte)[:, 1]

        df_test = df_full.loc[test_mask, ["event_id", "ts_ns", "oot_date", "fold_id", "side"]].copy()
        df_test["K"] = K
        df_test["S"] = S
        df_test["y_true_fp"] = yte
        df_test["y_pred_fp"] = yhat.astype(np.float32)
        df_test["fold_oot_idx"] = i
        oos_records.append(df_test)

        if save_models_dir is not None:
            save_models_dir.mkdir(parents=True, exist_ok=True)
            model.save_model(str(save_models_dir / f"K{K}_S{S}_oot{i:02d}_{d}.json"))

    if not oos_records:
        return pd.DataFrame()
    out = pd.concat(oos_records, ignore_index=True)
    out.attrs["mean_spw"] = float(np.mean(pos_weight_cache)) if pos_weight_cache else 1.0
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walk-parquet", required=True)
    ap.add_argument("--v2-pred-dir", required=True, help="Dir with {date}_predictions.npz files")
    ap.add_argument("--mbo-events-dir", required=True, help="Dir with {date}_mbo_events.npz files")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-estimators", type=int, default=600)
    ap.add_argument("--coverage-floor", type=float, default=0.70,
                    help="If overall v2 join coverage < floor, abort training (caller fallback).")
    ap.add_argument("--mlflow-uri", default="http://jupiter:5000")
    ap.add_argument("--mlflow-experiment", default="direct_firstpassage_v3_cnn_mamba")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading walk parquet {args.walk_parquet}...", flush=True)
    df = pd.read_parquet(args.walk_parquet)
    df = add_engineered(df)
    df["oot_date"] = df["oot_date"].astype(str)
    print(f"  rows={len(df)}  dates={df['oot_date'].nunique()}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] joining CNN-Mamba v2 predictions...", flush=True)
    df_joined, coverage, cov_stats = join_cnn_mamba_v2(
        df, Path(args.v2_pred_dir), Path(args.mbo_events_dir),
        tolerance_ns=1_000_000_000,
    )

    # Drop rows without v2 join (cleanest signal; HC-aligned: avoid imputed noise).
    pre = len(df_joined)
    df_joined = df_joined[df_joined["cnn_mamba_v2_pred"].notna()].reset_index(drop=True)
    print(f"  dropped {pre - len(df_joined)} rows without v2 join; kept {len(df_joined)}", flush=True)

    if coverage < args.coverage_floor:
        print(f"!! coverage {coverage:.3f} < floor {args.coverage_floor}; aborting", flush=True)
        (out_dir / "join_coverage.json").write_text(json.dumps({
            "overall_coverage": coverage, "per_date": cov_stats, "aborted": True,
        }, indent=2))
        sys.exit(2)

    # v2-derived features
    df_joined["cnn_mamba_v2_x_side"] = df_joined["cnn_mamba_v2_pred"] * df_joined["side"]
    df_joined["cnn_mamba_v2_conf"] = np.abs(df_joined["cnn_mamba_v2_pred"])  # already signed in [-1,1]-ish

    feature_cols = ALL_FEATURES
    print(f"  features ({len(feature_cols)}): {feature_cols}", flush=True)

    # MLflow
    mlflow = None
    run_id = None
    try:
        import mlflow as _mlf
        _mlf.set_tracking_uri(args.mlflow_uri)
        _mlf.set_experiment(args.mlflow_experiment)
        run = _mlf.start_run(run_name=f"v3_join_train_{time.strftime('%Y%m%d_%H%M%S')}")
        run_id = run.info.run_id
        _mlf.log_params({
            "n_estimators": args.n_estimators,
            "device": args.device,
            "feature_count": len(feature_cols),
            "v2_coverage_overall": round(coverage, 4),
            "n_rows_after_join": len(df_joined),
            "wf_style": "expanding_bucket_v1",
        })
        mlflow = _mlf
        print(f"  MLflow run: {run_id}", flush=True)
    except Exception as e:
        print(f"  MLflow init failed: {e} — continuing without", flush=True)

    # Sort for time-ordered internal split
    df_joined = df_joined.sort_values(["oot_date", "ts_ns"]).reset_index(drop=True)

    oot_dates = sorted(df_joined["oot_date"].unique().tolist())
    print(f"  oot_dates ({len(oot_dates)}): {oot_dates}", flush=True)

    all_oos = []
    log = {"heads": [], "device": args.device, "n_estimators": args.n_estimators,
           "n_rows": len(df_joined), "feature_cols": feature_cols,
           "v2_coverage_overall": coverage, "wf_style": "expanding_bucket_v1",
           "mlflow_run_id": run_id}

    for K, S in HEADS:
        t_h = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] === head (K={K}, S={S}) ===", flush=True)
        oos = train_head(df_joined, K, S, oot_dates, args.device, feature_cols,
                         n_estimators=args.n_estimators)
        if oos.empty:
            print(f"  EMPTY oos for (K={K},S={S})", flush=True)
            continue
        from sklearn.metrics import roc_auc_score, average_precision_score
        try:
            auc = roc_auc_score(oos["y_true_fp"], oos["y_pred_fp"])
            ap_ = average_precision_score(oos["y_true_fp"], oos["y_pred_fp"])
        except Exception:
            auc = float("nan"); ap_ = float("nan")
        base = float(oos["y_true_fp"].mean())
        wall = time.time() - t_h
        print(f"  done in {wall:.1f}s  n_oos={len(oos)}  base_rate={base:.4f}  AUC={auc:.4f}  AP={ap_:.4f}", flush=True)
        log["heads"].append({"K": K, "S": S, "n_oos": int(len(oos)),
                             "base_rate": base, "auc": float(auc), "ap": float(ap_), "wall_s": wall})
        if mlflow is not None:
            try:
                mlflow.log_metrics({f"auc_K{K}_S{S}": auc, f"ap_K{K}_S{S}": ap_, f"base_K{K}_S{S}": base})
            except Exception:
                pass
        all_oos.append(oos)

    if all_oos:
        oos_all = pd.concat(all_oos, ignore_index=True)
        oos_path = out_dir / "per_head_oos.parquet"
        oos_all.to_parquet(oos_path, index=False)
        print(f"[{time.strftime('%H:%M:%S')}] wrote {oos_path}  rows={len(oos_all)}", flush=True)

    log["total_wall_s"] = time.time() - t0
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2))
    (out_dir / "join_coverage.json").write_text(json.dumps({
        "overall_coverage": coverage,
        "per_date": [{"date": d, "n_events": n, "n_matched": m} for d, n, m in cov_stats],
    }, indent=2))

    if mlflow is not None:
        try:
            mlflow.log_artifact(str(out_dir / "training_log.json"))
            mlflow.log_artifact(str(out_dir / "join_coverage.json"))
            if all_oos:
                mlflow.log_artifact(str(out_dir / "per_head_oos.parquet"))
            mlflow.end_run()
        except Exception as e:
            print(f"  MLflow log_artifact failed: {e}", flush=True)

    print(f"[{time.strftime('%H:%M:%S')}] total wall {log['total_wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
