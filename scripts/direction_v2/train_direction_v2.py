#!/usr/bin/env python3
"""
PHASE 2 — DIRECTION v2 NATIVE-HORIZON trainer.

Three independent XGBoost-GPU binary classifiers, one per horizon h in {1s, 5s, 10s}.
Target: (dmid_h_ticks > 0) where |dmid_h_ticks| >= min_abs_ticks (default 1 tick).

NOTE on horizon substitution: brief asked for {3s, 5s, 10s}; existing book_features
already have labels at native h in {1s, 5s, 10s}. 1s replaces 3s — finer horizon
that the v1 diagnostic already flagged as the strongest bin (Q1 AUC 0.5526).

Features (no v1 head-engineered preds — those are per-trade-entry, not per-event):
- 30 book features from mbo_book_features (L5 prices, sizes, imbalance, OFI, etc.)
- CNN-Mamba v2 1s pred (asof join, backward, 1s tolerance)
- v2_conf = abs(v2_pred)
- minute_of_day, sec_in_minute (TOD effects)

Walk-forward: SLIDING train_window_days, OOT one day at a time. OOT in [oot_min, oot_max].

Per-event subsampling: random 100k events/day to keep training fast.
Liquidity filter: spread_ticks <= 2 (HC: tradable only on tight book).

MLflow: experiment direction_v2_native_horizons, one run per horizon.

HC: #428 R1 (regime stratification), #500 (2026 only), #503 (OOT window),
    #506 R5 (gates downstream if AUC >= 0.58).
"""
from __future__ import annotations
import argparse, json, os, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb

BOOK_FEAT_DIR = "/home/nick/Lvl3Quant/data/processed/mbo_book_features"
LABELS_DIR = "/home/nick/Lvl3Quant/data/direction_labels_native"
V2_PRED_DIR = "/home/nick/Lvl3Quant/output/cnn_mamba_v2_all_oot"
DEFAULT_OUTPUT = "/home/nick/Lvl3Quant/output/direction_v2_neptune"

BOOK_FEATURE_NAMES = [
    "bid_price_1","bid_price_2","bid_price_3","bid_price_4","bid_price_5",
    "ask_price_1","ask_price_2","ask_price_3","ask_price_4","ask_price_5",
    "bid_size_1","bid_size_2","bid_size_3","bid_size_4","bid_size_5",
    "ask_size_1","ask_size_2","ask_size_3","ask_size_4","ask_size_5",
    "cum_delta","rolling_imbalance_100","trade_intensity_100",
    "depth_imbalance_5","spread_ticks",
    "bid_size_change","ask_size_change",
    "mid_price_change_ticks","spread_change_ticks","net_order_flow",
]

# subset we use as features (skip raw bid/ask prices — they're price-level, will not generalize)
USE_BOOK_FEATURES = [
    "bid_size_1","bid_size_2","bid_size_3","bid_size_4","bid_size_5",
    "ask_size_1","ask_size_2","ask_size_3","ask_size_4","ask_size_5",
    "cum_delta","rolling_imbalance_100","trade_intensity_100",
    "depth_imbalance_5","spread_ticks",
    "bid_size_change","ask_size_change",
    "mid_price_change_ticks","spread_change_ticks","net_order_flow",
]
V2_FEATURES = ["cnn_mamba_v2_pred", "cnn_mamba_v2_conf"]
TOD_FEATURES = ["minute_of_day"]
ALL_FEATURES = USE_BOOK_FEATURES + V2_FEATURES + TOD_FEATURES


def load_day_features(date_str: str, max_spread_ticks: int = 2,
                      n_subsample: int = 100_000, rng_seed: int = 42):
    """Load per-event features+labels for one day. Returns DataFrame or None.

    OPTIMIZED: subsample at the np-array level BEFORE building DataFrame.
    A single book_features.npz is ~1.7GB on disk; we only keep n_subsample rows.
    """
    book_path = os.path.join(BOOK_FEAT_DIR, f"{date_str}_book_features.npz")
    lab_path = os.path.join(LABELS_DIR, f"{date_str}.parquet")
    if not os.path.exists(book_path) or not os.path.exists(lab_path):
        return None
    # Use mmap_mode for the npz feature array — avoids loading full into RAM
    z = np.load(book_path, allow_pickle=False, mmap_mode="r")
    fn = list(z["feature_names"])
    feats_arr = z["features"]  # mmap'd, shape (N, 30)
    ts_full = np.asarray(z["timestamps"], dtype=np.int64)
    spread_idx = fn.index("spread_ticks")
    spread_full = np.asarray(feats_arr[:, spread_idx], dtype=np.float32)

    # Liquidity filter at array level (avoids materializing DataFrame for ~95% of rows)
    liq_mask = (spread_full <= max_spread_ticks) & np.isfinite(spread_full)
    liq_idx = np.flatnonzero(liq_mask)
    if len(liq_idx) == 0:
        return None
    # Random subsample among liquid events
    if len(liq_idx) > n_subsample:
        rng = np.random.default_rng(rng_seed + int(date_str))
        sel = rng.choice(len(liq_idx), size=n_subsample, replace=False)
        sel.sort()
        liq_idx = liq_idx[sel]

    # Now materialize ONLY the selected rows from mmap'd feature matrix
    col_idxs = [fn.index(name) for name in USE_BOOK_FEATURES]
    sel_feats = np.asarray(feats_arr[liq_idx[:, None], np.array(col_idxs)], dtype=np.float32)
    sel_ts = ts_full[liq_idx]
    df = pd.DataFrame(sel_feats, columns=USE_BOOK_FEATURES)
    df.insert(0, "ts_ns", sel_ts)

    # Load native-horizon labels (much smaller parquet)
    lab = pd.read_parquet(lab_path, columns=["ts_ns","dmid_1s_ticks","dmid_5s_ticks","dmid_10s_ticks","mid_now"])
    df = df.merge(lab, on="ts_ns", how="inner")
    if len(df) == 0:
        return None

    # CNN-Mamba v2 asof join
    v2_path = os.path.join(V2_PRED_DIR, f"{date_str}_predictions.npz")
    if os.path.exists(v2_path):
        try:
            zv = np.load(v2_path, allow_pickle=True)
            preds = zv["predictions"]
            n_windows = int(zv["n_windows"])
            stride = int(zv["stride"])
            window_size = int(zv["window_size"])
            v2_signal = preds[:, 0].astype(np.float32)  # 1s horizon
            w_idx = np.arange(n_windows, dtype=np.int64)
            last_event_idx = w_idx * stride + window_size - 1
            valid = last_event_idx < len(ts_full)
            last_event_idx = last_event_idx[valid]
            v2_signal = v2_signal[valid]
            window_end_ts = ts_full[last_event_idx]
            win_df = pd.DataFrame({
                "window_end_ts": window_end_ts,
                "cnn_mamba_v2_pred": v2_signal,
            }).sort_values("window_end_ts").reset_index(drop=True)
            df_sorted = df.sort_values("ts_ns").reset_index(drop=True)
            df_sorted = pd.merge_asof(df_sorted, win_df,
                                      left_on="ts_ns", right_on="window_end_ts",
                                      direction="backward", tolerance=1_000_000_000)
            df_sorted["cnn_mamba_v2_pred"] = df_sorted["cnn_mamba_v2_pred"].fillna(0.0)
            df_sorted["cnn_mamba_v2_conf"] = df_sorted["cnn_mamba_v2_pred"].abs()
            df_sorted = df_sorted.drop(columns=["window_end_ts"])
            df = df_sorted
        except Exception as e:
            print(f"  [{date_str}] v2 join failed: {e}; filling 0", flush=True)
            df["cnn_mamba_v2_pred"] = 0.0
            df["cnn_mamba_v2_conf"] = 0.0
    else:
        df["cnn_mamba_v2_pred"] = 0.0
        df["cnn_mamba_v2_conf"] = 0.0

    # TOD feature: minute of day (UTC seconds in day / 60)
    sec_in_day = (df["ts_ns"].values % (24*3600*int(1e9))) // int(1e9)
    df["minute_of_day"] = (sec_in_day // 60).astype(np.int32)

    df["oot_date"] = date_str
    return df


def _load_helper(d, max_spread_ticks, n_subsample):
    df = load_day_features(str(d), max_spread_ticks=max_spread_ticks, n_subsample=n_subsample)
    return (d, df)


def make_label(df: pd.DataFrame, horizon_col: str, min_abs_ticks: float = 1.0):
    dmid = df[horizon_col].values.astype(np.float32)
    keep = (np.abs(dmid) >= min_abs_ticks) & np.isfinite(dmid)
    y = (dmid > 0).astype(np.int8)
    return y, keep


def regime_stratify(oos: pd.DataFrame) -> dict:
    if oos.empty: return {}
    df = oos.copy()
    df["pred_dir"] = np.sign(df["y_pred"] - 0.5)
    df["true_dir"] = 2 * df["y_true"].astype(int) - 1
    df["edge"] = df["pred_dir"] * df["true_dir"]
    daily = df.groupby("oot_date")["edge"].agg(["mean","std","count"]).reset_index()
    daily["sharpe"] = daily["mean"] / daily["std"].replace(0, np.nan) * np.sqrt(daily["count"])
    classifications = []
    for _, row in daily.iterrows():
        if row["mean"] > 0.05: classifications.append(("green", row))
        elif row["mean"] < -0.05: classifications.append(("red", row))
        else: classifications.append(("flat", row))
    by_cls = {}
    for cls, row in classifications:
        by_cls.setdefault(cls, []).append(row["sharpe"])
    out = {"per_day": daily.to_dict(orient="records")}
    for cls in ("green","red","flat"):
        vals = by_cls.get(cls, [])
        out[f"sharpe_{cls}_mean"] = float(np.nanmean(vals)) if vals else None
        out[f"n_days_{cls}"] = len(vals)
    sg, sr = out.get("sharpe_green_mean"), out.get("sharpe_red_mean")
    if sg is not None and sr is not None:
        denom = max(abs(sg), abs(sr), 1e-9)
        out["regime_asymmetry"] = float(abs(sg-sr)/denom)
    else:
        out["regime_asymmetry"] = None
    return out


def train_horizon(df_all, horizon_col, horizon_name, oot_dates, train_window_days,
                  device, hp, min_abs_ticks, early_stop_folds, kill_auc_thresh,
                  out_dir, mlflow_uri, mlflow_exp, run_prefix):
    print(f"\n========= TRAIN h={horizon_name} =========", flush=True)
    t0 = time.time()
    feature_cols = ALL_FEATURES
    print(f"  feature_cols ({len(feature_cols)}): {feature_cols}", flush=True)

    y_full, keep = make_label(df_all, horizon_col, min_abs_ticks)
    print(f"  total rows {len(df_all)}  kept |dmid|>={min_abs_ticks}t: {keep.sum()}  pos_rate {y_full[keep].mean():.4f}", flush=True)

    dates_full = df_all["oot_date"].values
    X_full = df_all[feature_cols].values.astype(np.float32)
    # XGBoost handles NaN natively
    X_full = np.where(np.isfinite(X_full), X_full, np.nan).astype(np.float32)

    # MLflow start
    mlflow = run_id = None
    try:
        import mlflow as _mlf
        _mlf.set_tracking_uri(mlflow_uri)
        _mlf.set_experiment(mlflow_exp)
        run = _mlf.start_run(run_name=f"{run_prefix}_h={horizon_name}_{time.strftime('%H%M%S')}")
        run_id = run.info.run_id
        _mlf.set_tag("horizon", horizon_name)
        _mlf.log_params({**hp, "horizon": horizon_name, "device": device,
                         "train_window_days": train_window_days,
                         "min_abs_ticks": min_abs_ticks,
                         "wf_style": "sliding",
                         "n_rows_total": int(len(df_all)),
                         "n_oot_dates": len(oot_dates)})
        mlflow = _mlf
        print(f"  MLflow run {run_id}", flush=True)
    except Exception as e:
        print(f"  MLflow init failed: {e}", flush=True)

    burn_in = max(4, min(10, len(oot_dates)//3))
    oos_records = []
    fold_aucs = []
    killed_early = False
    for i, d in enumerate(oot_dates):
        if i < burn_in:
            continue
        train_dates_win = oot_dates[max(0, i - train_window_days):i]
        train_mask = np.isin(dates_full, train_dates_win) & keep
        test_mask = (dates_full == d) & keep
        if test_mask.sum() == 0 or train_mask.sum() < 5000:
            print(f"  [{d}] skip train={train_mask.sum()} test={test_mask.sum()}", flush=True)
            continue
        Xtr, ytr = X_full[train_mask], y_full[train_mask]
        Xte, yte = X_full[test_mask], y_full[test_mask]
        cut = int(len(Xtr) * 0.9)
        Xtr2, ytr2 = Xtr[:cut], ytr[:cut]
        Xva, yva = Xtr[cut:], ytr[cut:]
        model = xgb.XGBClassifier(
            n_estimators=hp["n_estimators"], max_depth=hp["max_depth"],
            learning_rate=hp["learning_rate"], tree_method="hist", device=device,
            objective="binary:logistic", eval_metric="logloss",
            subsample=hp["subsample"], colsample_bytree=hp["colsample"],
            reg_lambda=hp["reg_lambda"], random_state=42, verbosity=0,
            early_stopping_rounds=25,
        )
        if len(Xva) > 100 and yva.sum() > 0 and (len(yva)-yva.sum()) > 0:
            model.fit(Xtr2, ytr2, eval_set=[(Xva,yva)], verbose=False)
        else:
            model.fit(Xtr, ytr)
        yhat = model.predict_proba(Xte)[:, 1]
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(yte, yhat)
        except Exception:
            auc = float("nan")
        fold_aucs.append(auc)
        print(f"  [{d}] n_train={len(Xtr)} n_test={len(Xte)} fold_auc={auc:.4f}", flush=True)
        if mlflow is not None:
            try: mlflow.log_metric("fold_auc", auc, step=i)
            except Exception: pass

        df_test = df_all.loc[test_mask, ["ts_ns","oot_date","mid_now",
                                         "dmid_1s_ticks","dmid_5s_ticks","dmid_10s_ticks",
                                         "spread_ticks"]].copy()
        df_test["y_true"] = yte
        df_test["y_pred"] = yhat.astype(np.float32)
        df_test["fold_idx"] = i
        oos_records.append(df_test)

        # Kill-early check: after early_stop_folds, if max AUC < threshold, abort this horizon
        if len(fold_aucs) >= early_stop_folds and max(fold_aucs) < kill_auc_thresh:
            print(f"  ** KILL EARLY: {len(fold_aucs)} folds, max AUC {max(fold_aucs):.4f} < {kill_auc_thresh}", flush=True)
            killed_early = True
            break

    result = {
        "horizon": horizon_name, "killed_early": killed_early,
        "n_folds_done": len(fold_aucs),
        "fold_aucs": [float(a) for a in fold_aucs],
        "wall_s": time.time()-t0,
        "mlflow_run_id": run_id,
    }

    if oos_records:
        oos = pd.concat(oos_records, ignore_index=True)
        from sklearn.metrics import roc_auc_score, brier_score_loss
        try: overall_auc = roc_auc_score(oos["y_true"], oos["y_pred"])
        except Exception: overall_auc = float("nan")
        try: overall_brier = brier_score_loss(oos["y_true"], oos["y_pred"])
        except Exception: overall_brier = float("nan")
        regime = regime_stratify(oos)
        result["overall_auc"] = float(overall_auc)
        result["overall_brier"] = float(overall_brier)
        result["n_oos"] = int(len(oos))
        result["pos_rate"] = float(oos["y_true"].mean())
        result["regime"] = regime
        oos_path = os.path.join(out_dir, f"per_event_oos_h={horizon_name}.parquet")
        oos.to_parquet(oos_path, index=False)
        print(f"  OOS AUC={overall_auc:.4f} Brier={overall_brier:.4f} n={len(oos)} pos={oos['y_true'].mean():.4f}", flush=True)
        if mlflow is not None:
            try:
                mlflow.log_metrics({"overall_auc": overall_auc, "overall_brier": overall_brier,
                                    "pos_rate": float(oos["y_true"].mean()), "n_oos": float(len(oos))})
                if regime.get("regime_asymmetry") is not None:
                    mlflow.log_metric("regime_asymmetry", regime["regime_asymmetry"])
                mlflow.log_artifact(oos_path)
            except Exception as e:
                print(f"  mlflow log failed: {e}", flush=True)
    else:
        result["overall_auc"] = None

    if mlflow is not None:
        try: mlflow.end_run()
        except Exception: pass

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-estimators", type=int, default=500)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--learning-rate", type=float, default=0.05)
    ap.add_argument("--reg-lambda", type=float, default=1.0)
    ap.add_argument("--subsample", type=float, default=0.85)
    ap.add_argument("--colsample", type=float, default=0.85)
    ap.add_argument("--min-abs-ticks", type=float, default=1.0)
    ap.add_argument("--train-window-days", type=int, default=30)
    ap.add_argument("--oot-min", type=int, default=20260227)
    ap.add_argument("--oot-max", type=int, default=20260429)
    ap.add_argument("--max-spread-ticks", type=int, default=2)
    ap.add_argument("--n-subsample-per-day", type=int, default=100_000)
    ap.add_argument("--load-workers", type=int, default=8)
    ap.add_argument("--horizons", default="1s,5s,10s")
    ap.add_argument("--early-stop-folds", type=int, default=2,
                    help="After this many folds, kill horizon if max AUC < kill-auc-thresh")
    ap.add_argument("--kill-auc-thresh", type=float, default=0.55)
    ap.add_argument("--mlflow-uri", default="http://jupiter:5000")
    ap.add_argument("--mlflow-experiment", default="direction_v2_native_horizons")
    ap.add_argument("--run-prefix", default="direction_v2_neptune")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    # Discover available label days within OOT window AND ALL train days (no upper bound on train)
    all_days = sorted([int(f.replace(".parquet","")) for f in os.listdir(LABELS_DIR)
                       if f.endswith(".parquet") and f[0].isdigit()])
    print(f"[main] {len(all_days)} total label-days available", flush=True)
    # We need ALL days up to oot_max for training context. OOT days are window-restricted.
    use_days = [d for d in all_days if d <= args.oot_max]
    oot_days_in_window = [d for d in all_days if args.oot_min <= d <= args.oot_max]
    print(f"[main] training pool: {len(use_days)} days (<= {args.oot_max})", flush=True)
    print(f"[main] OOT eval days in window [{args.oot_min},{args.oot_max}]: {len(oot_days_in_window)}", flush=True)

    print(f"[main] loading + asof-joining per-event features for {len(use_days)} days (parallel)...", flush=True)
    from multiprocessing import Pool
    def _wrap(d):
        return d, load_day_features(str(d), max_spread_ticks=args.max_spread_ticks,
                                    n_subsample=args.n_subsample_per_day)
    # Pool can't pickle local functions — use module-level helper via starmap
    import functools
    fn = functools.partial(_load_helper,
                           max_spread_ticks=args.max_spread_ticks,
                           n_subsample=args.n_subsample_per_day)
    dfs = []
    with Pool(args.load_workers) as pool:
        for i, (d, df) in enumerate(pool.imap_unordered(fn, use_days, chunksize=1)):
            if df is None:
                print(f"  [{d}] skip", flush=True)
                continue
            dfs.append(df)
            if (i+1) % 10 == 0:
                print(f"  loaded {i+1}/{len(use_days)} days  rows_so_far={sum(len(x) for x in dfs)}", flush=True)
    df_all = pd.concat(dfs, ignore_index=True)
    df_all = df_all.sort_values(["oot_date","ts_ns"]).reset_index(drop=True)
    print(f"[main] total rows: {len(df_all)}  total days: {df_all['oot_date'].nunique()}", flush=True)
    print(f"[main] load wall: {time.time()-t0:.1f}s", flush=True)

    hp = dict(n_estimators=args.n_estimators, max_depth=args.max_depth,
              learning_rate=args.learning_rate, reg_lambda=args.reg_lambda,
              subsample=args.subsample, colsample=args.colsample)

    oot_dates_str = sorted(df_all["oot_date"].unique().tolist())
    # restrict OOT evaluation set to the [oot_min, oot_max] window
    oot_str_in_window = [d for d in oot_dates_str if args.oot_min <= int(d) <= args.oot_max]
    # We pass ALL oot_dates_str (including pre-oot) into walk-forward because train_window_days
    # may need days before oot_min as training context. But we'll filter test_mask to in-window
    # implicitly because for i corresponding to a pre-window day, we don't care — the OOS records
    # for ANY date will be aggregated, but for the headline AUC we'll filter to in-window dates.

    horizon_map = {"1s":"dmid_1s_ticks","5s":"dmid_5s_ticks","10s":"dmid_10s_ticks"}
    horizons = [h.strip() for h in args.horizons.split(",")]

    all_results = {}
    max_auc_across = 0.0
    for h in horizons:
        col = horizon_map[h]
        res = train_horizon(
            df_all, col, h, oot_dates_str, args.train_window_days,
            args.device, hp, args.min_abs_ticks,
            args.early_stop_folds, args.kill_auc_thresh,
            str(out_dir), args.mlflow_uri, args.mlflow_experiment, args.run_prefix,
        )
        all_results[h] = res
        if res.get("overall_auc") is not None:
            max_auc_across = max(max_auc_across, res["overall_auc"])

    summary = {
        "max_overall_auc_across_horizons": max_auc_across,
        "per_horizon": all_results,
        "total_wall_s": time.time()-t0,
        "args": vars(args),
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n[main] DONE in {time.time()-t0:.1f}s", flush=True)
    print(f"[main] MAX AUC across horizons: {max_auc_across:.4f}", flush=True)
    for h in horizons:
        r = all_results.get(h, {})
        print(f"  h={h}: AUC={r.get('overall_auc')}  n_folds={r.get('n_folds_done')}  killed={r.get('killed_early')}", flush=True)


if __name__ == "__main__":
    main()
