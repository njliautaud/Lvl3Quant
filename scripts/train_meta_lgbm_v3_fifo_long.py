#!/usr/bin/env python3
"""
Meta-LGBM v3 FIFO — HC #282(C) parallel training on Jupiter CPU.

Trains LightGBM on top of CNN-Mamba v2 OOT predictions to predict FIFO net
P&L. Uses 2:1 stride downsample (verified ratio=2.00x across all 11 folds).

Target: tp4sl3_long_net_ticks (HC #282(C): long side parallel sanity).
Features: pred_1s, pred_5s, pred_10s, plus 96-dim CNN-Mamba embedding.
CV: walk-forward by fold (train folds 0..N, eval fold N+1). Folds 0..10 → 10 evals.
Logged to MLflow experiment "MetaLGBM_v3_FIFO" on http://localhost:5000.
"""
import os, sys, time, json
from pathlib import Path
import numpy as np
import lightgbm as lgb
from scipy.stats import spearmanr
import mlflow

PRED_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar")
FIFO_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_fifo_labels")
OUT_DIR  = Path("/home/jupiter/Lvl3Quant/output/meta_lgbm_v3_fifo_long")
OUT_DIR.mkdir(parents=True, exist_ok=True)

CAP_TICKS = 20.0
TARGET_KEY = "tp4sl3_long_net_ticks"
FILLED_KEY = "tp4sl3_long_filled"

mlflow.set_tracking_uri("http://localhost:5000")
mlflow.set_experiment("MetaLGBM_v3_FIFO")


def load_fold(fold_idx):
    p = PRED_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if not p.exists():
        return None
    d = np.load(p)
    src = str(d["oot_files"][0])
    date = os.path.basename(src).split("_")[0]
    fifo_p = FIFO_DIR / f"{date}_fifo_labels.npz"
    if not fifo_p.exists():
        return None
    fifo = np.load(fifo_p)
    n_pred = d["predictions"].shape[0]
    # 2:1 stride alignment (verified)
    fifo_idx = np.arange(n_pred) * 2
    if fifo_idx.max() >= len(fifo["window_k"]):
        fifo_idx = fifo_idx[fifo_idx < len(fifo["window_k"])]
        n_pred = len(fifo_idx)
    preds = d["predictions"][:n_pred]      # (N, 3)
    embed = d["embeddings"][:n_pred]       # (N, 96)
    target = fifo[TARGET_KEY][fifo_idx].astype(np.float32)
    filled = fifo[FILLED_KEY][fifo_idx].astype(bool)
    target = np.clip(target, -CAP_TICKS, CAP_TICKS)
    # Restrict to filled-only (unfilled = no signal info)
    keep = filled
    X = np.concatenate([preds[keep], embed[keep]], axis=1).astype(np.float32)
    y = target[keep]
    return X, y, date, int(keep.sum())


def main():
    folds = []
    for i in range(11):
        r = load_fold(i)
        if r is not None:
            folds.append((i, *r))
    print(f"Loaded {len(folds)} folds with FIFO labels")
    feat_names = [f"pred_{h}" for h in ["1s", "5s", "10s"]] + [f"emb_{i}" for i in range(96)]

    with mlflow.start_run(run_name=f"meta_lgbm_v3_fifo_long_{int(time.time())}"):
        mlflow.log_params({
            "target": TARGET_KEY, "cap_ticks": CAP_TICKS,
            "n_folds": len(folds), "n_features": 99,
            "stride_alignment": "2:1 (v2 stride=250 -> fifo stride=125)",
            "model": "LightGBM regressor", "node": "Jupiter CPU",
            "filled_only": True,
        })
        all_results = []
        for split_idx in range(2, len(folds)):
            train_folds = folds[:split_idx]
            eval_fold = folds[split_idx]
            X_tr = np.concatenate([f[1] for f in train_folds])
            y_tr = np.concatenate([f[2] for f in train_folds])
            X_ev, y_ev = eval_fold[1], eval_fold[2]
            params = {
                "objective": "regression_l1", "metric": "mae",
                "num_leaves": 31, "learning_rate": 0.05,
                "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 5,
                "min_data_in_leaf": 100, "verbose": -1, "num_threads": 16,
            }
            ds_tr = lgb.Dataset(X_tr, label=y_tr, feature_name=feat_names)
            ds_ev = lgb.Dataset(X_ev, label=y_ev, reference=ds_tr)
            model = lgb.train(params, ds_tr, num_boost_round=500,
                              valid_sets=[ds_ev], valid_names=["eval"],
                              callbacks=[lgb.early_stopping(30), lgb.log_evaluation(0)])
            preds = model.predict(X_ev)
            ic, _ = spearmanr(preds, y_ev)
            top10_mask = preds >= np.quantile(preds, 0.9)
            top10_mean = float(y_ev[top10_mask].mean())
            top10_n = int(top10_mask.sum())
            rmse = float(np.sqrt(np.mean((preds - y_ev) ** 2)))
            print(f"split={split_idx:2d}  eval_date={eval_fold[3]}  "
                  f"n_tr={len(y_tr):>6}  n_ev={len(y_ev):>5}  "
                  f"IC={ic:+.4f}  top10_mean={top10_mean:+.3f}t (n={top10_n})  RMSE={rmse:.3f}")
            mlflow.log_metric(f"ic_split_{split_idx}", float(ic), step=split_idx)
            mlflow.log_metric(f"top10_split_{split_idx}", top10_mean, step=split_idx)
            mlflow.log_metric(f"rmse_split_{split_idx}", rmse, step=split_idx)
            all_results.append({"split": split_idx, "eval_date": eval_fold[3],
                                "ic": float(ic), "top10_mean": top10_mean, "rmse": rmse,
                                "n_train": int(len(y_tr)), "n_eval": int(len(y_ev))})
            model.save_model(str(OUT_DIR / f"meta_lgbm_split_{split_idx:02d}.txt"))
        with open(OUT_DIR / "results.json", "w") as f:
            json.dump(all_results, f, indent=2)
        ics = [r["ic"] for r in all_results]
        mlflow.log_metric("ic_mean", float(np.mean(ics)))
        mlflow.log_metric("ic_median", float(np.median(ics)))
        print(f"\nDONE  ic_mean={np.mean(ics):+.4f}  ic_median={np.median(ics):+.4f}")
        mlflow.log_artifact(str(OUT_DIR / "results.json"))


if __name__ == "__main__":
    main()
