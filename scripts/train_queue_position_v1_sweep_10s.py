#!/usr/bin/env python3
"""
Queue-Position v1 SWEEP — 10s fill classifier hyperparameter sweep
==================================================================
Goal: see if better hyperparams push v1's 10s fill AUC from 0.6984 to >=0.72.

REUSES v1's data loader (load_all_dates_joined) verbatim — no feature changes.
Trains FILL classifier ONLY (queue regression head dropped — R²=0.01 baseline).
ONLY 10s horizon.
Same sliding WF schedule as v1 (train_window=25, eval_window=1) → ~3 folds with current data.

5 hyperparam configs:
  A. baseline  — replicates v1 exactly (sanity check)
  B. deeper    — max_depth=10, n_est=3000, mcw=5
  C. slow      — lr=0.02, n_est=5000, early_stop=100
  D. more_reg  — alpha=1.0, lambda=5.0, gamma=1.0
  E. low_samp  — subsample=0.6, colsample=0.6

HC constraints:
  - HC #0:   sliding window only (no expanding)
  - HC #420: authorized codebase
  - HC #495: causal features only (uses v1's existing 83-dim, no new features)
  - HC #498 R4: honest reporting, all 5 configs side-by-side
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import traceback
from pathlib import Path

import numpy as np

LVL3_ROOT = Path("/home/nick/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT / "experiments"))
sys.path.insert(0, str(LVL3_ROOT / "scripts"))

# Reuse v1 loader and join logic verbatim
import train_queue_position_v1_xgb_dualhead as v1

OUTPUT_DIR = LVL3_ROOT / "output" / "queue_position_v1_sweep_10s"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "training.log"

logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [QPOS_SWEEP] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(str(LOG_FILE), mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("qpos_sweep")

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "queue_position_v1_sweep_10s"
H = 10  # only 10s horizon
TS_TOL_NS = v1.TS_TOL_NS

# ---------- 5 hyperparam configs (fill classifier only) ----------
def _base_params():
    return dict(
        objective="binary:logistic", eval_metric="auc",
        tree_method="hist", device="cuda:0",
        verbosity=0, random_state=42,
    )

CONFIGS = {
    "A_baseline": dict(_base_params(),
        max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        n_estimators=2000, early_stopping_rounds=50,
    ),
    "B_deeper": dict(_base_params(),
        max_depth=10, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=5, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        n_estimators=3000, early_stopping_rounds=50,
    ),
    "C_slow": dict(_base_params(),
        max_depth=6, learning_rate=0.02,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        n_estimators=5000, early_stopping_rounds=100,
    ),
    "D_more_reg": dict(_base_params(),
        max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        min_child_weight=10, gamma=1.0, reg_alpha=1.0, reg_lambda=5.0,
        n_estimators=2000, early_stopping_rounds=50,
    ),
    "E_low_samp": dict(_base_params(),
        max_depth=6, learning_rate=0.05,
        subsample=0.6, colsample_bytree=0.6,
        min_child_weight=10, gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
        n_estimators=2000, early_stopping_rounds=50,
    ),
}


def train_fill_only_fold(fold_idx, train_packs, eval_packs, params, smoke=False):
    """Train ONLY the 10s fill classifier on the given fold. Returns metrics + OOT preds."""
    import xgboost as xgb
    from sklearn.metrics import roc_auc_score, brier_score_loss

    train_X = np.concatenate([d["X"] for d in train_packs])
    eval_X = np.concatenate([d["X"] for d in eval_packs])
    np.nan_to_num(train_X, copy=False)
    np.nan_to_num(eval_X, copy=False)
    train_y = np.concatenate([d["ys"][H]["fill"] for d in train_packs]).astype(np.int8)
    eval_y = np.concatenate([d["ys"][H]["fill"] for d in eval_packs]).astype(np.int8)

    p = dict(params)
    if smoke:
        p["n_estimators"] = min(p["n_estimators"], 200)
        p["early_stopping_rounds"] = min(p["early_stopping_rounds"], 20)
    pos = float(train_y.sum())
    neg = float(len(train_y) - pos)
    p["scale_pos_weight"] = neg / max(pos, 1.0)

    t0 = time.time()
    clf = xgb.XGBClassifier(**p)
    clf.fit(train_X, train_y, eval_set=[(eval_X, eval_y)], verbose=False)
    yhat = clf.predict_proba(eval_X)[:, 1]
    dt = time.time() - t0

    try:
        auc = float(roc_auc_score(eval_y, yhat)) if 0 < eval_y.sum() < len(eval_y) else float("nan")
        brier = float(brier_score_loss(eval_y, yhat))
    except Exception:
        auc, brier = float("nan"), float("nan")
    best_iter = int(getattr(clf, "best_iteration", clf.n_estimators) or clf.n_estimators)

    return dict(
        fold=fold_idx, auc=auc, brier=brier, best_iter=best_iter, dt_s=dt,
        n_train=int(len(train_X)), n_eval=int(len(eval_X)),
        pos_rate_train=float(train_y.mean()), pos_rate_eval=float(eval_y.mean()),
    ), yhat, eval_y


def run_one_config(cfg_name, params, joined, train_window, eval_window, max_folds, smoke):
    import mlflow
    fold_metrics = []
    all_yhat = []
    all_y = []

    n = len(joined)
    start = 0
    fold_idx = 0
    while start + train_window + eval_window <= n:
        if max_folds is not None and fold_idx >= max_folds:
            break
        train_packs = joined[start:start + train_window]
        eval_packs = joined[start + train_window:start + train_window + eval_window]
        log.info(
            f"  [{cfg_name}] FOLD {fold_idx}: train {train_packs[0]['date']}..{train_packs[-1]['date']} "
            f"eval {eval_packs[0]['date']}..{eval_packs[-1]['date']}"
        )
        fm, yhat, y = train_fill_only_fold(fold_idx, train_packs, eval_packs, params, smoke=smoke)
        log.info(
            f"  [{cfg_name}] fold{fold_idx} dt={fm['dt_s']:.1f}s AUC={fm['auc']:.4f} "
            f"Brier={fm['brier']:.4f} best_iter={fm['best_iter']} n_eval={fm['n_eval']:,}"
        )
        fold_metrics.append(fm)
        all_yhat.append(yhat)
        all_y.append(y)
        try:
            mlflow.log_metric(f"fold_auc", fm["auc"], step=fold_idx)
            mlflow.log_metric(f"fold_brier", fm["brier"], step=fold_idx)
            mlflow.log_metric(f"fold_best_iter", fm["best_iter"], step=fold_idx)
        except Exception:
            pass
        start += eval_window
        fold_idx += 1
        gc.collect()

    # Concat across folds
    pf = np.concatenate(all_yhat) if all_yhat else np.array([])
    yf = np.concatenate(all_y) if all_y else np.array([])
    from sklearn.metrics import roc_auc_score, brier_score_loss
    if len(yf) > 0 and 0 < yf.sum() < len(yf):
        concat_auc = float(roc_auc_score(yf, pf))
        concat_brier = float(brier_score_loss(yf, pf))
    else:
        concat_auc, concat_brier = float("nan"), float("nan")

    # Persist OOT preds for the config
    np.savez(
        str(OUTPUT_DIR / f"{cfg_name}_oot.npz"),
        p_fill_10s=pf.astype(np.float32),
        y_fill_10s=yf.astype(np.int8),
    )
    return dict(
        cfg=cfg_name, params=params,
        n_folds=len(fold_metrics),
        concat_auc=concat_auc, concat_brier=concat_brier,
        n_oot=int(len(pf)), pos_rate_oot=float(yf.mean()) if len(yf) else float("nan"),
        fold_metrics=fold_metrics,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-window", type=int, default=25)
    ap.add_argument("--eval-window", type=int, default=1)
    ap.add_argument("--max-folds", type=int, default=None,
                    help="Cap on number of folds (smoke uses 2)")
    ap.add_argument("--smoke", action="store_true",
                    help="2 folds, capped rounds, all 5 configs — ~5 min sanity check")
    ap.add_argument("--configs", type=str, default="A_baseline,B_deeper,C_slow,D_more_reg,E_low_samp",
                    help="Comma list of config keys to run")
    args = ap.parse_args()

    log.info(f"Output dir: {OUTPUT_DIR}")
    import xgboost as xgb
    log.info(f"XGBoost: {xgb.__version__}")
    try:
        import subprocess
        out = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name,memory.free,utilization.gpu", "--format=csv,noheader"],
            text=True).strip()
        log.info(f"GPU: {out}")
    except Exception as e:
        log.warning(f"nvidia-smi failed: {e}")

    cfg_keys = [c.strip() for c in args.configs.split(",") if c.strip()]
    for k in cfg_keys:
        if k not in CONFIGS:
            log.error(f"Unknown config: {k}. Available: {list(CONFIGS)}")
            sys.exit(2)

    # Load data ONCE (reuse v1 loader)
    t0 = time.time()
    joined, join_rate = v1.load_all_dates_joined()
    log.info(f"Loaded {len(joined)} joined dates in {time.time()-t0:.1f}s (join_rate={join_rate:.2%})")
    if join_rate < 0.80:
        log.error(f"JOIN RATE {join_rate:.2%} < 80% — ABORTING")
        sys.exit(3)
    if len(joined) < args.train_window + args.eval_window:
        log.error(f"Not enough joined dates ({len(joined)}) for WF")
        sys.exit(2)

    max_folds = args.max_folds
    if args.smoke:
        max_folds = 2

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    all_results = {}
    overall_t0 = time.time()
    for cfg_name in cfg_keys:
        params = CONFIGS[cfg_name]
        run_name = f"{cfg_name}_{time.strftime('%Y%m%d_%H%M')}"
        if args.smoke:
            run_name += "_SMOKE"
        with mlflow.start_run(run_name=run_name) as run:
            log.info(f"=== CONFIG {cfg_name} === run_id={run.info.run_id}")
            # log params (drop non-scalar)
            log_params = {k: v for k, v in params.items() if isinstance(v, (int, float, str, bool))}
            log_params.update(dict(
                cfg=cfg_name, horizon_s=H,
                train_window_days=args.train_window,
                eval_window_days=args.eval_window,
                smoke=args.smoke,
                hc="498_R4_sweep",
            ))
            mlflow.log_params(log_params)
            mlflow.set_tag("cfg", cfg_name)
            mlflow.set_tag("sweep", "queue_position_v1_sweep_10s")

            t1 = time.time()
            res = run_one_config(cfg_name, params, joined,
                                 args.train_window, args.eval_window, max_folds, args.smoke)
            dt = time.time() - t1
            res["wall_s"] = dt
            log.info(f"=== {cfg_name} CONCAT AUC={res['concat_auc']:.4f} "
                     f"Brier={res['concat_brier']:.4f} n_oot={res['n_oot']:,} "
                     f"wall={dt:.1f}s")
            mlflow.log_metric("concat_fill_auc_10s", res["concat_auc"])
            mlflow.log_metric("concat_fill_brier_10s", res["concat_brier"])
            mlflow.log_metric("wall_s", dt)
            all_results[cfg_name] = res

    overall_wall = time.time() - overall_t0
    log.info(f"=== SWEEP DONE wall={overall_wall:.1f}s ===")

    # Build leaderboard
    leaderboard = sorted(
        [(k, v["concat_auc"], v["concat_brier"], v["n_oot"]) for k, v in all_results.items()],
        key=lambda r: (-r[1] if r[1] == r[1] else 1e9),
    )
    log.info("LEADERBOARD (concat AUC, desc):")
    for k, auc, brier, n in leaderboard:
        log.info(f"  {k:14s}  AUC={auc:.4f}  Brier={brier:.4f}  n_oot={n:,}")

    # V1 baseline reference
    v1_baseline_auc = 0.6984
    best_cfg, best_auc = leaderboard[0][0], leaderboard[0][1]
    if best_auc >= 0.72:
        verdict = f"MEANINGFUL: {best_cfg} AUC={best_auc:.4f} >= 0.72 — recommend productionize"
    elif best_auc >= 0.70:
        verdict = f"MARGINAL: {best_cfg} AUC={best_auc:.4f} (v1 was {v1_baseline_auc}) — document and stop"
    else:
        verdict = f"NULL: best AUC={best_auc:.4f} <= 0.70 — v1 hyperparams near-optimal"
    log.info(f"VERDICT: {verdict}")

    summary = {
        "hc": "498_R4_sweep",
        "horizon_s": H,
        "head": "fill_classifier_only",
        "train_window_days": args.train_window,
        "eval_window_days": args.eval_window,
        "n_joined_dates": len(joined),
        "join_rate_of_signal": join_rate,
        "smoke": args.smoke,
        "wall_s": overall_wall,
        "v1_baseline_concat_fill_auc_10s": v1_baseline_auc,
        "leaderboard": [
            dict(cfg=k, concat_auc=auc, concat_brier=brier, n_oot=n)
            for k, auc, brier, n in leaderboard
        ],
        "results": all_results,
        "verdict": verdict,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Wrote {OUTPUT_DIR/'summary.json'}")


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
