#!/usr/bin/env python
"""
Direct first-passage v2 — XGB GPU on extended feature set.

Same WF protocol as v1 (sliding train, 1-day OOT, 4-day burn-in), but features =
  9 engineered head crosses
+ top-K (default 10) book microstructure features ranked by MI on TRAIN ONLY (per head, per fold)

Outputs:
- per_head_oos.parquet
- training_log.json
- mlflow run per head (experiment: direct_firstpassage_v2_xgb)
"""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import mutual_info_classif
from sklearn.metrics import roc_auc_score, average_precision_score

try:
    import mlflow
    HAS_MLFLOW = True
except Exception:
    HAS_MLFLOW = False

HEADS = [(2,1),(3,1),(4,1),(5,1),(3,2),(4,2),(5,2),(5,4)]
HOLD_CAP_NS = 15_000_000_000

ENG_COLS = ["y_pred_adverse","y_pred_mfe","y_pred_toxicity","side",
            "adv_x_side","mfe_x_side","tox_x_side","mfe_minus_adv","mfe_minus_adv_x_side"]


def label_first_passage(df: pd.DataFrame, K: int, S: int, hold_cap_ns: int = HOLD_CAP_NS) -> np.ndarray:
    tp = df[f"tp{K}_dt_ns"].values
    sl = df[f"sl{S}_dt_ns"].values
    tp_hit = (tp > 0) & (tp <= hold_cap_ns)
    sl_hit = (sl > 0) & (sl <= hold_cap_ns)
    win = tp_hit & ((~sl_hit) | (tp < sl))
    return win.astype(np.int8)


def select_features(Xtr: np.ndarray, ytr: np.ndarray, names: list[str], top_k: int, eng_count: int, rng: int = 42) -> list[int]:
    """Always keep the first eng_count engineered cols. Pick top_k by MI from the rest."""
    n_features = Xtr.shape[1]
    if n_features <= eng_count:
        return list(range(n_features))
    eng_idx = list(range(eng_count))
    bf_idx = list(range(eng_count, n_features))
    # Subsample for MI speed
    n = len(ytr)
    take = min(n, 30000)
    sel = np.random.default_rng(rng).choice(n, size=take, replace=False)
    Xs = Xtr[sel][:, bf_idx]
    ys = ytr[sel]
    mi = mutual_info_classif(Xs, ys, random_state=rng, n_neighbors=3)
    order = np.argsort(mi)[::-1][:top_k]
    chosen_bf = [bf_idx[i] for i in order]
    return eng_idx + chosen_bf


def train_head(df_full: pd.DataFrame, K: int, S: int, oot_dates: list[str], device: str,
               feature_cols: list[str], n_estimators: int, top_k_bf: int, eng_count: int,
               early_stop: int = 30, lr: float = 0.03, max_depth: int = 6) -> tuple[pd.DataFrame, dict]:
    y_full = label_first_passage(df_full, K, S)
    X_full = df_full[feature_cols].values.astype(np.float32)
    dates_full = df_full["oot_date"].values

    burn_in = 4
    oos_records = []
    fold_logs = []

    for i, d in enumerate(oot_dates):
        if i < burn_in:
            continue
        train_mask = np.isin(dates_full, oot_dates[:i])
        # sliding window: only last 20 dates
        train_dates_window = oot_dates[max(0, i - 20):i]
        train_mask = np.isin(dates_full, train_dates_window)
        test_mask = dates_full == d
        if test_mask.sum() == 0 or train_mask.sum() == 0:
            continue

        Xtr_all, ytr = X_full[train_mask], y_full[train_mask]
        Xte_all, yte = X_full[test_mask], y_full[test_mask]

        sel = select_features(Xtr_all, ytr, feature_cols, top_k=top_k_bf, eng_count=eng_count)
        Xtr = Xtr_all[:, sel]
        Xte = Xte_all[:, sel]
        sel_names = [feature_cols[k] for k in sel]

        # Hold-out 15% of train for early stopping
        n_tr = len(Xtr)
        rng = np.random.default_rng(42 + i)
        perm = rng.permutation(n_tr)
        n_val = max(int(0.15 * n_tr), 2000)
        val_idx = perm[:n_val]
        tr_idx = perm[n_val:]
        Xtr_, ytr_ = Xtr[tr_idx], ytr[tr_idx]
        Xval, yval = Xtr[val_idx], ytr[val_idx]

        pos = ytr_.sum(); neg = len(ytr_) - pos
        spw = float(neg / max(pos, 1)) if pos > 0 else 1.0

        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=lr,
            tree_method="hist",
            device=device,
            objective="binary:logistic",
            eval_metric="logloss",
            scale_pos_weight=spw,
            subsample=0.85,
            colsample_bytree=0.85,
            reg_lambda=1.0,
            random_state=42,
            early_stopping_rounds=early_stop,
            verbosity=0,
        )
        model.fit(Xtr_, ytr_, eval_set=[(Xval, yval)], verbose=False)
        yhat = model.predict_proba(Xte)[:, 1]

        df_test = df_full.loc[test_mask, ["event_id","ts_ns","oot_date","fold_id","side"]].copy()
        df_test["K"] = K
        df_test["S"] = S
        df_test["y_true_fp"] = yte
        df_test["y_pred_fp"] = yhat.astype(np.float32)
        df_test["fold_oot_idx"] = i
        oos_records.append(df_test)

        try:
            auc = roc_auc_score(yte, yhat)
        except Exception:
            auc = float("nan")
        fold_logs.append({"oot_idx": i, "date": d, "n_train": int(train_mask.sum()), "n_test": int(test_mask.sum()),
                          "best_iter": int(getattr(model, "best_iteration", -1) or -1),
                          "fold_auc": float(auc), "selected_features": sel_names})

    if not oos_records:
        return pd.DataFrame(), {"folds": fold_logs}
    out = pd.concat(oos_records, ignore_index=True)
    return out, {"folds": fold_logs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--walks-extended", required=True)
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--n-estimators", type=int, default=1000)
    ap.add_argument("--early-stop", type=int, default=30)
    ap.add_argument("--lr", type=float, default=0.03)
    ap.add_argument("--max-depth", type=int, default=6)
    ap.add_argument("--top-k-bf", type=int, default=10, help="top K book features by MI")
    ap.add_argument("--mlflow-uri", default="")
    ap.add_argument("--mlflow-experiment", default="direct_firstpassage_v2_xgb")
    ap.add_argument("--heads", default="all")
    args = ap.parse_args()

    out_dir = Path(args.output_dir); out_dir.mkdir(parents=True, exist_ok=True)

    if HAS_MLFLOW and args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
        mlflow.set_experiment(args.mlflow_experiment)

    t0 = time.time()
    print(f"[{time.strftime('%H:%M:%S')}] loading extended walks parquet...", flush=True)
    df = pd.read_parquet(args.walks_extended)
    bf_cols = [c for c in df.columns if c.startswith("bf_")]
    feature_cols = ENG_COLS + bf_cols
    eng_count = len(ENG_COLS)
    print(f"  rows={len(df)} eng={eng_count} bf={len(bf_cols)} total_feats_avail={len(feature_cols)}", flush=True)

    oot_dates = sorted(df["oot_date"].unique().tolist())
    print(f"  oot_dates n={len(oot_dates)}: {oot_dates}", flush=True)

    if args.heads == "all":
        head_list = HEADS
    else:
        head_list = [tuple(int(x) for x in t.split("_")) for t in args.heads.split(",")]

    all_oos = []
    log = {"heads": [], "device": args.device, "n_estimators": args.n_estimators, "lr": args.lr,
           "max_depth": args.max_depth, "top_k_bf": args.top_k_bf, "early_stop": args.early_stop,
           "n_rows": len(df), "feature_cols_available": feature_cols}

    for K, S in head_list:
        t_h = time.time()
        print(f"[{time.strftime('%H:%M:%S')}] === head K={K} S={S} ===", flush=True)
        run_ctx = mlflow.start_run(run_name=f"K{K}_S{S}") if (HAS_MLFLOW and args.mlflow_uri) else None
        try:
            if run_ctx is not None:
                mlflow.log_params({"K": K, "S": S, "n_estimators": args.n_estimators, "lr": args.lr,
                                   "max_depth": args.max_depth, "top_k_bf": args.top_k_bf,
                                   "early_stop": args.early_stop, "device": args.device})
            oos, head_log = train_head(df, K, S, oot_dates, args.device, feature_cols,
                                       args.n_estimators, args.top_k_bf, eng_count,
                                       args.early_stop, args.lr, args.max_depth)
            if oos.empty:
                print("  EMPTY oos", flush=True); continue
            try:
                auc = roc_auc_score(oos["y_true_fp"], oos["y_pred_fp"])
                ap_ = average_precision_score(oos["y_true_fp"], oos["y_pred_fp"])
            except Exception:
                auc = float("nan"); ap_ = float("nan")
            base = float(oos["y_true_fp"].mean())
            wall = time.time() - t_h
            print(f"  done {wall:.1f}s  n={len(oos)}  base={base:.4f}  AUC={auc:.4f}  AP={ap_:.4f}", flush=True)
            entry = {"K": K, "S": S, "n_oos": int(len(oos)), "base_rate": base, "auc": float(auc),
                     "ap": float(ap_), "wall_s": wall, "folds": head_log["folds"]}
            log["heads"].append(entry)
            all_oos.append(oos)
            if run_ctx is not None:
                mlflow.log_metrics({"auc_oos": float(auc), "ap_oos": float(ap_), "base_rate": float(base),
                                    "n_oos": int(len(oos)), "wall_s": float(wall)})
        finally:
            if run_ctx is not None:
                mlflow.end_run()

    if all_oos:
        out_all = pd.concat(all_oos, ignore_index=True)
        out_all.to_parquet(out_dir / "per_head_oos.parquet", index=False)
    log["total_wall_s"] = time.time() - t0
    (out_dir / "training_log.json").write_text(json.dumps(log, indent=2, default=float))
    print(f"[{time.strftime('%H:%M:%S')}] done wall={log['total_wall_s']:.1f}s", flush=True)


if __name__ == "__main__":
    main()
