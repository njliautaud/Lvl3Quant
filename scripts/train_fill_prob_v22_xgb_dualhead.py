#!/usr/bin/env python3
"""
Fill-Probability v2.2 — DUAL-HEAD XGB on combined 83-dim features (HC #497 R5)
==============================================================================
History:
  - v1 (XGB, 41 micro feats): concat AUC=0.86 (May 1, 2026).
  - v2 (MLP, 42 signal feats): AUC=0.59 — regressed.
  - v2.1 (MLP, 83 combined): AUC=0.57 — regressed further.
  Conclusion: MLP is wrong architecture. Switch back to gradient boosting,
  but keep the v2 capability of also predicting adverse cost (regression head).

v2.2 design:
  - Reuse v2.1's data loader (combined 83-dim: 41 micro + 42 signal) EXACTLY.
  - Two separate XGB models per fold trained in parallel:
       head1_fill = XGBClassifier (binary:logistic, eval_metric=auc)
       head2_adv  = XGBRegressor  (reg:squarederror, eval_metric=rmse)
  - tree_method='hist', device='cuda:0' (RTX 3090 on Neptune).
  - 13-fold sliding WF identical to v2.1 (HC #0).
  - MLflow experiment: fill_prob_v22_xgb_dualhead.

Honest reporting (HC #493 R4): if concat AUC < 0.80, do NOT claim success.
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

LVL3_ROOT = Path("/home/nick/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT / "experiments"))
sys.path.insert(0, str(LVL3_ROOT / "scripts"))

# Reuse v2.1 loader (which itself reuses v1 + v2). This gives us:
#   - load_all_dates_combined()
#   - N_FEATURES (83), N_SIGNAL_FEATURES (42), N_MICRO_FEATURES (41)
#   - FILL_THRESHOLD_TICKS
import train_fill_prob_v21_combined as v21  # noqa: E402

OUTPUT_DIR = LVL3_ROOT / "output" / "fill_prob_v22_xgb_dualhead"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "training.log"

logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [FILL_V22] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(str(LOG_FILE), mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("fill_v22")

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "fill_prob_v22_xgb_dualhead"

N_FEATURES = v21.N_FEATURES
N_SIGNAL_FEATURES = v21.N_SIGNAL_FEATURES
N_MICRO_FEATURES = v21.N_MICRO_FEATURES
FILL_THRESHOLD_TICKS = v21.FILL_THRESHOLD_TICKS

# XGB hyperparameters — start from v1's proven config, add GPU + regressor variant.
XGB_PARAMS_FILL = dict(
    objective="binary:logistic",
    eval_metric="auc",
    tree_method="hist",
    device="cuda:0",
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_estimators=2000,  # cap; early stopping will trim
    early_stopping_rounds=50,
    verbosity=0,
    random_state=42,
)

XGB_PARAMS_ADV = dict(
    objective="reg:squarederror",
    eval_metric="rmse",
    tree_method="hist",
    device="cuda:0",
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    min_child_weight=10,
    gamma=0.1,
    reg_alpha=0.1,
    reg_lambda=1.0,
    n_estimators=2000,
    early_stopping_rounds=50,
    verbosity=0,
    random_state=42,
)


def train_one_fold(fold_idx, train_X, train_yf, train_ya, eval_X, eval_yf, eval_ya,
                   smoke: bool = False):
    """Train fill classifier + adverse regressor in parallel threads (each XGB call
    is GPU-bound; the threads serialize on the GPU but it still saves the
    classifier-then-regressor wall-clock equivalent due to GIL release in xgb)."""
    import xgboost as xgb

    fill_params = dict(XGB_PARAMS_FILL)
    adv_params = dict(XGB_PARAMS_ADV)
    if smoke:
        fill_params["n_estimators"] = 50
        adv_params["n_estimators"] = 50
        fill_params["early_stopping_rounds"] = 10
        adv_params["early_stopping_rounds"] = 10

    # pos/neg balance for fill head
    pos = float(train_yf.sum())
    neg = float(len(train_yf) - pos)
    spw = neg / max(pos, 1.0)
    fill_params["scale_pos_weight"] = spw

    log.info(
        f"  fold{fold_idx} fitting: train={len(train_X):,} eval={len(eval_X):,} "
        f"pos_rate={pos/max(len(train_yf),1):.3%} spw={spw:.2f}"
    )

    t0 = time.time()

    def fit_fill():
        clf = xgb.XGBClassifier(**fill_params)
        clf.fit(train_X, train_yf, eval_set=[(eval_X, eval_yf)], verbose=False)
        return clf

    def fit_adv():
        reg = xgb.XGBRegressor(**adv_params)
        reg.fit(train_X, train_ya, eval_set=[(eval_X, eval_ya)], verbose=False)
        return reg

    with ThreadPoolExecutor(max_workers=2) as ex:
        fut_fill = ex.submit(fit_fill)
        fut_adv = ex.submit(fit_adv)
        clf = fut_fill.result()
        reg = fut_adv.result()

    dt = time.time() - t0

    p_fill = clf.predict_proba(eval_X)[:, 1]
    yhat_adv = reg.predict(eval_X)

    # metrics
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        auc = float(roc_auc_score(eval_yf, p_fill)) if 0 < eval_yf.sum() < len(eval_yf) else float("nan")
        brier = float(brier_score_loss(eval_yf, p_fill))
    except Exception:
        auc = float("nan"); brier = float("nan")
    rmse = float(np.sqrt(np.mean((yhat_adv - eval_ya) ** 2)))

    n_best_fill = int(getattr(clf, "best_iteration", clf.n_estimators) or clf.n_estimators)
    n_best_adv = int(getattr(reg, "best_iteration", reg.n_estimators) or reg.n_estimators)

    log.info(
        f"  fold{fold_idx} done dt={dt:.1f}s  "
        f"AUC={auc:.4f} Brier={brier:.4f} RMSE={rmse:.4f}  "
        f"best_iter_fill={n_best_fill} best_iter_adv={n_best_adv}"
    )

    return clf, reg, {
        "fold": fold_idx,
        "fit_seconds": dt,
        "n_train": int(len(train_X)),
        "n_eval": int(len(eval_X)),
        "fill_rate_train": float(train_yf.mean()),
        "fill_rate_eval": float(eval_yf.mean()),
        "auc": auc,
        "brier": brier,
        "rmse_adv": rmse,
        "adv_mean_eval": float(eval_ya.mean()),
        "adv_std_eval": float(eval_ya.std()),
        "best_iter_fill": n_best_fill,
        "best_iter_adv": n_best_adv,
    }, p_fill, yhat_adv


def walk_forward(all_dates, train_window=60, eval_window=5, max_folds=None, smoke=False):
    fold_metrics = []
    fold_idx = 0
    start = 0
    n = len(all_dates)
    while start + train_window + eval_window <= n:
        if max_folds is not None and fold_idx >= max_folds:
            break
        train_slice = all_dates[start:start + train_window]
        eval_slice = all_dates[start + train_window:start + train_window + eval_window]
        train_X = np.concatenate([d["features"] for d in train_slice])
        train_yf = np.concatenate([d["targets"]["fill_label"] for d in train_slice])
        train_ya = np.concatenate([d["targets"]["adverse_cost"] for d in train_slice])
        eval_X = np.concatenate([d["features"] for d in eval_slice])
        eval_yf = np.concatenate([d["targets"]["fill_label"] for d in eval_slice])
        eval_ya = np.concatenate([d["targets"]["adverse_cost"] for d in eval_slice])

        for arr in (train_X, eval_X):
            np.nan_to_num(arr, copy=False)

        log.info(
            f"FOLD {fold_idx}: train {train_slice[0]['date']}..{train_slice[-1]['date']} "
            f"({len(train_X):,}) eval {eval_slice[0]['date']}..{eval_slice[-1]['date']} "
            f"({len(eval_X):,})"
        )
        clf, reg, m, p_fill, yhat_adv = train_one_fold(
            fold_idx, train_X, train_yf, train_ya, eval_X, eval_yf, eval_ya, smoke=smoke
        )
        m["train_start"] = train_slice[0]["date"]
        m["train_end"] = train_slice[-1]["date"]
        m["eval_start"] = eval_slice[0]["date"]
        m["eval_end"] = eval_slice[-1]["date"]
        fold_metrics.append(m)

        # save artifacts
        clf.save_model(str(OUTPUT_DIR / f"fold_{fold_idx:03d}_fill.json"))
        reg.save_model(str(OUTPUT_DIR / f"fold_{fold_idx:03d}_adv.json"))
        np.savez(
            str(OUTPUT_DIR / f"fold_{fold_idx:03d}_oot.npz"),
            p_fill=p_fill, yhat_adv=yhat_adv,
            y_fill=eval_yf, y_adv=eval_ya,
        )

        # log to mlflow per fold
        try:
            import mlflow
            mlflow.log_metric("fold_auc", m["auc"], step=fold_idx)
            mlflow.log_metric("fold_brier", m["brier"], step=fold_idx)
            mlflow.log_metric("fold_rmse_adv", m["rmse_adv"], step=fold_idx)
            mlflow.log_metric("fold_fit_seconds", m["fit_seconds"], step=fold_idx)
            mlflow.log_metric("fold_best_iter_fill", m["best_iter_fill"], step=fold_idx)
            mlflow.log_metric("fold_best_iter_adv", m["best_iter_adv"], step=fold_idx)
        except Exception:
            pass

        start += eval_window
        fold_idx += 1
        gc.collect()
    return fold_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-window", type=int, default=60)
    ap.add_argument("--eval-window", type=int, default=5)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="Run a tiny 3-fold sanity check with reduced n_estimators")
    args = ap.parse_args()

    log.info(f"Output dir: {OUTPUT_DIR}")
    log.info(f"Features: {N_FEATURES} (signal={N_SIGNAL_FEATURES} + micro={N_MICRO_FEATURES})")
    log.info(f"FILL_THRESHOLD_TICKS: {FILL_THRESHOLD_TICKS}")

    import xgboost as xgb
    log.info(f"XGBoost: {xgb.__version__}")

    # GPU sanity check
    try:
        import subprocess
        out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.free", "--format=csv,noheader"], text=True).strip()
        log.info(f"GPU: {out}")
    except Exception as e:
        log.warning(f"nvidia-smi failed: {e}")

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"fill_v22_xgb_{time.strftime('%Y%m%d_%H%M')}"
    if args.smoke:
        run_name += "_SMOKE"
    run = mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        "n_features": N_FEATURES,
        "n_signal_features": N_SIGNAL_FEATURES,
        "n_micro_features": N_MICRO_FEATURES,
        "fill_threshold_ticks": FILL_THRESHOLD_TICKS,
        "fill_horizon": "1s",
        "adverse_horizon": "5s",
        "model": "XGB_dualhead",
        "tree_method": "hist",
        "device": "cuda:0",
        "n_estimators_cap": XGB_PARAMS_FILL["n_estimators"],
        "early_stopping_rounds": XGB_PARAMS_FILL["early_stopping_rounds"],
        "max_depth": XGB_PARAMS_FILL["max_depth"],
        "learning_rate": XGB_PARAMS_FILL["learning_rate"],
        "subsample": XGB_PARAMS_FILL["subsample"],
        "colsample_bytree": XGB_PARAMS_FILL["colsample_bytree"],
        "train_window_days": args.train_window,
        "eval_window_days": args.eval_window,
        "hc": "497_R5",
        "version": "v22_xgb_dualhead",
        "smoke": args.smoke,
    })
    log.info(f"MLflow experiment: {EXPERIMENT_NAME}")
    log.info(f"MLflow run id: {run.info.run_id}")
    log.info(f"MLflow run name: {run_name}")

    t0 = time.time()
    all_dates = v21.load_all_dates_combined()
    log.info(f"Loaded {len(all_dates)} dates in {time.time()-t0:.1f}s")
    if len(all_dates) < args.train_window + args.eval_window:
        log.error(f"Not enough dates ({len(all_dates)}) for WF (need {args.train_window+args.eval_window})")
        mlflow.end_run(status="FAILED")
        sys.exit(2)

    max_folds = args.max_folds
    if args.smoke and max_folds is None:
        max_folds = 3

    fold_metrics = walk_forward(
        all_dates,
        train_window=args.train_window,
        eval_window=args.eval_window,
        max_folds=max_folds,
        smoke=args.smoke,
    )

    # Concat metrics
    all_pf, all_yf, all_pa, all_ya = [], [], [], []
    for fm in fold_metrics:
        d = np.load(str(OUTPUT_DIR / f"fold_{fm['fold']:03d}_oot.npz"))
        all_pf.append(d["p_fill"]); all_yf.append(d["y_fill"])
        all_pa.append(d["yhat_adv"]); all_ya.append(d["y_adv"])

    concat_auc = float("nan"); concat_brier = float("nan"); concat_rmse = float("nan")
    if all_pf:
        pf = np.concatenate(all_pf); yf = np.concatenate(all_yf)
        pa = np.concatenate(all_pa); ya = np.concatenate(all_ya)
        try:
            from sklearn.metrics import roc_auc_score, brier_score_loss
            concat_auc = float(roc_auc_score(yf, pf)) if 0 < yf.sum() < len(yf) else float("nan")
            concat_brier = float(brier_score_loss(yf, pf))
        except Exception:
            pass
        concat_rmse = float(np.sqrt(np.mean((pa - ya) ** 2)))
        log.info(
            f"CONCAT n={len(pf):,}  fill_AUC={concat_auc:.4f}  "
            f"fill_Brier={concat_brier:.4f}  adv_RMSE={concat_rmse:.4f}"
        )
        try:
            import mlflow
            mlflow.log_metric("concat_fill_auc", concat_auc)
            mlflow.log_metric("concat_fill_brier", concat_brier)
            mlflow.log_metric("concat_adv_rmse", concat_rmse)
            mlflow.log_metric("n_folds", len(fold_metrics))
            mlflow.log_metric("n_total_oot", len(pf))
        except Exception:
            pass

    # Honest reporting (HC #493 R4)
    if not (concat_auc != concat_auc) and concat_auc < 0.80:
        log.warning(
            f"AUC {concat_auc:.4f} BELOW 0.80 target. NOT a success. "
            f"Report this honestly to the user."
        )
    elif concat_auc >= 0.80:
        log.info(f"AUC {concat_auc:.4f} meets 0.80 target.")

    summary = {
        "hc": "497_R5",
        "model": "XGB_dualhead",
        "n_features": N_FEATURES,
        "n_signal_features": N_SIGNAL_FEATURES,
        "n_micro_features": N_MICRO_FEATURES,
        "fill_threshold_ticks": FILL_THRESHOLD_TICKS,
        "fill_horizon": "1s",
        "adverse_horizon": "5s",
        "xgb_params_fill": XGB_PARAMS_FILL,
        "xgb_params_adv": XGB_PARAMS_ADV,
        "train_window_days": args.train_window,
        "eval_window_days": args.eval_window,
        "n_folds": len(fold_metrics),
        "concat_fill_auc": concat_auc,
        "concat_fill_brier": concat_brier,
        "concat_adv_rmse": concat_rmse,
        "smoke": args.smoke,
        "fold_metrics": fold_metrics,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Wrote {OUTPUT_DIR/'summary.json'}")
    try:
        import mlflow
        mlflow.log_artifact(str(OUTPUT_DIR / "summary.json"))
        mlflow.end_run()
    except Exception:
        pass
    log.info("DONE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log.error(f"FATAL: {e}")
        traceback.print_exc()
        try:
            import mlflow
            mlflow.end_run(status="FAILED")
        except Exception:
            pass
        sys.exit(1)
