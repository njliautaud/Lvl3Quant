#!/usr/bin/env python3
"""
Queue-Position v2 — DUAL-HEAD XGB on MBO-walker labels + QUEUE-AUGMENTED features
=================================================================================
HC #498 R4 follow-up: v1 had broken queue regression (R^2 ~0.01) because v2.1
features had NO direct queue / level-age / order-flow info. We built those
features (queue_augmented_features/) and now refit.

Inputs
------
v2.1 combined features (83-dim):    via train_fill_prob_v21_combined.load_all_dates_combined()
walker labels (per date):           /home/nick/Lvl3Quant/output/mbo_walker_labels/labels_YYYYMMDD.parquet
queue-augmented features (per date):/home/nick/Lvl3Quant/data/queue_augmented_features/features_YYYYMMDD.parquet

Key fact (verified empirically 2026-05-30):
  For every date, the augmented-features parquet and the walker-labels parquet
  are 1:1 row-aligned (same event_id and ts_ns, same length, same order).
  So after we do the v1 ts-tolerance join (v2.1 features -> labels nearest ts_ns),
  we have `matched_label_rows` indices that index BOTH the labels df AND the
  augmented-features df. Use those same indices to slice augmented feats.

Final feature vector = 83 (v2.1) + ~28 numeric augmented = ~111 dim.
We exclude these augmented cols from the feature matrix because they are not
predictors: event_id, ts_ns, side, pred_1s, pred_5s, pred_10s
(pred_*s are CNN-Mamba predictions also embedded in v2.1; side/event_id/ts_ns
are identifiers).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
import traceback
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np

LVL3_ROOT = Path("/home/nick/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT / "experiments"))
sys.path.insert(0, str(LVL3_ROOT / "scripts"))

import train_fill_prob_v21_combined as v21  # reuses v1 + v2 loaders

LABELS_DIR = LVL3_ROOT / "output" / "mbo_walker_labels"
AUG_DIR = LVL3_ROOT / "data" / "queue_augmented_features"
OUTPUT_DIR = LVL3_ROOT / "output" / "queue_position_v2_xgb_dualhead_augmented"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = OUTPUT_DIR / "training.log"

logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [QPOS_V2] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[logging.FileHandler(str(LOG_FILE), mode="w"), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("qpos_v2")

MLFLOW_URI = "http://localhost:5000"
EXPERIMENT_NAME = "queue_position_v2_xgb_dualhead_augmented"
HORIZONS = [1, 5, 10]
TS_TOL_NS = 500_000_000  # 0.5s tolerance for ts_ns join (same as v1)

# Columns to drop from augmented parquet before concatenating into feature matrix:
# event_id, ts_ns are identifiers; side is constant-ish; pred_* are already in v2.1.
DROP_AUG_COLS = ["event_id", "ts_ns", "side", "pred_1s", "pred_5s", "pred_10s"]

# XGB hyperparameters (v1 baseline + slight reg bump from sweep findings)
XGB_PARAMS_FILL = dict(
    objective="binary:logistic", eval_metric="auc",
    tree_method="hist", device="cuda:0",
    max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=10,
    gamma=0.5, reg_alpha=0.5, reg_lambda=2.0,
    n_estimators=2000, early_stopping_rounds=50,
    verbosity=0, random_state=42,
)
XGB_PARAMS_QUEUE = dict(
    objective="reg:squarederror", eval_metric="rmse",
    tree_method="hist", device="cuda:0",
    max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    min_child_weight=10,
    gamma=0.5, reg_alpha=0.5, reg_lambda=2.0,
    n_estimators=2000, early_stopping_rounds=50,
    verbosity=0, random_state=42,
)


def load_walker_labels_for_date(date_str: str):
    p = LABELS_DIR / f"labels_{date_str}.parquet"
    if not p.exists():
        return None
    import pandas as pd
    df = pd.read_parquet(p)
    df = df.dropna(subset=["ts_ns"]).reset_index(drop=True)
    return df


def load_augmented_features_for_date(date_str: str):
    """Returns (df, X_aug, aug_feat_names) or (None, None, None) if missing."""
    p = AUG_DIR / f"features_{date_str}.parquet"
    if not p.exists():
        return None, None, None
    import pandas as pd
    df = pd.read_parquet(p)
    keep_cols = [c for c in df.columns if c not in DROP_AUG_COLS]
    X_aug = df[keep_cols].to_numpy(dtype=np.float32, copy=False)
    return df, X_aug, keep_cols


def join_features_labels(date_pack, labels_df, aug_df, X_aug, date_str: str):
    """
    Join v2.1 combined features with walker labels by ts_ns (nearest within TS_TOL_NS).
    Then ALSO slice augmented features by the same matched_label_rows indices
    (valid because labels_df and aug_df are 1:1 row-aligned per build invariant).
    """
    feats = date_pack["features"]   # (Nf, 83)
    meta = date_pack["meta"]
    if len(feats) == 0 or labels_df is None or len(labels_df) == 0:
        return None

    feat_ts_ns = (meta[:, 0].astype(np.float64) * 1e9).round().astype(np.int64)
    lab_ts = labels_df["ts_ns"].to_numpy().astype(np.int64)

    order = np.argsort(lab_ts, kind="stable")
    lab_ts_sorted = lab_ts[order]
    idx_right = np.searchsorted(lab_ts_sorted, feat_ts_ns)
    idx_left = np.clip(idx_right - 1, 0, len(lab_ts_sorted) - 1)
    idx_right = np.clip(idx_right, 0, len(lab_ts_sorted) - 1)
    d_left = np.abs(feat_ts_ns - lab_ts_sorted[idx_left])
    d_right = np.abs(feat_ts_ns - lab_ts_sorted[idx_right])
    pick_right = d_right < d_left
    nearest_sorted = np.where(pick_right, idx_right, idx_left)
    nearest_orig = order[nearest_sorted]
    nearest_dt = np.where(pick_right, d_right, d_left)
    keep = nearest_dt <= TS_TOL_NS
    n_signal = len(feat_ts_ns)
    n_label = len(lab_ts)
    n_match = int(keep.sum())

    if n_match == 0:
        return None

    matched_label_rows = nearest_orig[keep]

    X_v21 = feats[keep].astype(np.float32)

    # Slice aug features by matched_label_rows.
    # Sanity: assert aug_df and labels_df are 1:1 row-aligned (verified empirically).
    if aug_df is None or X_aug is None:
        # No augmented features for this date -> skip (treat as join failure for v2)
        return None
    if len(aug_df) != len(labels_df):
        log.warning(
            f"  {date_str}: aug rows {len(aug_df)} != labels rows {len(labels_df)} — SKIP"
        )
        return None
    # Random spot check of event_id alignment on first matched row
    if len(matched_label_rows) > 0:
        i = int(matched_label_rows[0])
        a_eid = int(aug_df["event_id"].iat[i])
        l_eid = int(labels_df["event_id"].iat[i])
        if a_eid != l_eid:
            log.warning(
                f"  {date_str}: event_id mismatch at row {i} (aug={a_eid} lab={l_eid}) — SKIP"
            )
            return None

    X_aug_matched = X_aug[matched_label_rows]
    # nan/inf guard
    X_aug_matched = np.nan_to_num(X_aug_matched, nan=0.0, posinf=0.0, neginf=0.0)
    X_combined = np.concatenate([X_v21, X_aug_matched], axis=1).astype(np.float32)

    ys = {}
    for h in HORIZONS:
        fill_col = f"filled_{h}s"
        rank_col = f"queue_rank_at_{h}s"
        y_fill = labels_df[fill_col].to_numpy()[matched_label_rows].astype(np.int8)
        y_queue = labels_df[rank_col].to_numpy()[matched_label_rows].astype(np.float32)
        ys[h] = {"fill": y_fill, "queue": y_queue}
    mean_dt_ms = float(nearest_dt[keep].mean()) / 1e6
    return X_combined, ys, n_signal, n_label, n_match, mean_dt_ms


def load_all_dates_joined():
    log.info("Loading v2.1 combined features (will take a few minutes)...")
    t0 = time.time()
    all_dates = v21.load_all_dates_combined()
    log.info(f"v2.1 loader returned {len(all_dates)} dates in {time.time()-t0:.1f}s")

    joined = []
    total_sig = 0
    total_lab = 0
    total_match = 0
    n_aug_avail = 0
    n_aug_joined = 0
    for pack in all_dates:
        date_str = pack["date"]
        labels_df = load_walker_labels_for_date(date_str)
        if labels_df is None:
            log.info(f"  {date_str}: no walker labels — SKIP")
            continue
        aug_df, X_aug, aug_cols = load_augmented_features_for_date(date_str)
        if aug_df is None:
            log.info(f"  {date_str}: no augmented features — SKIP (v2 requires aug)")
            continue
        n_aug_avail += 1
        result = join_features_labels(pack, labels_df, aug_df, X_aug, date_str)
        if result is None:
            log.info(f"  {date_str}: 0 matches after join — SKIP")
            continue
        X, ys, n_sig, n_lab, n_match, mean_dt_ms = result
        total_sig += n_sig
        total_lab += n_lab
        total_match += n_match
        n_aug_joined += 1
        log.info(
            f"  {date_str}: sig={n_sig:,} lab={n_lab:,} match={n_match:,} "
            f"({n_match/max(n_sig,1):.1%} of sig) mean_dt={mean_dt_ms:.1f}ms "
            f"X_dim={X.shape[1]}"
        )
        if n_match < 10:
            continue
        joined.append({"date": date_str, "X": X, "ys": ys, "n": n_match})
        gc.collect()

    join_rate = total_match / max(total_sig, 1)
    aug_join_rate = n_aug_joined / max(n_aug_avail, 1)
    log.info(
        f"JOIN SUMMARY: total signal_rows={total_sig:,} label_rows={total_lab:,} "
        f"matched={total_match:,} join_rate_of_signal={join_rate:.2%} "
        f"aug_dates_avail={n_aug_avail} aug_dates_joined={n_aug_joined} "
        f"aug_join_rate={aug_join_rate:.2%}"
    )
    return joined, join_rate, aug_join_rate


def train_fill_head(train_X, train_y, eval_X, eval_y, params, smoke_rounds=None):
    import xgboost as xgb
    p = dict(params)
    if smoke_rounds is not None:
        p["n_estimators"] = smoke_rounds
        p["early_stopping_rounds"] = min(20, smoke_rounds // 4 or 1)
    pos = float(train_y.sum())
    neg = float(len(train_y) - pos)
    p["scale_pos_weight"] = neg / max(pos, 1.0)
    clf = xgb.XGBClassifier(**p)
    clf.fit(train_X, train_y, eval_set=[(eval_X, eval_y)], verbose=False)
    y_pred = clf.predict_proba(eval_X)[:, 1]
    try:
        from sklearn.metrics import roc_auc_score, brier_score_loss
        auc = float(roc_auc_score(eval_y, y_pred)) if 0 < eval_y.sum() < len(eval_y) else float("nan")
        brier = float(brier_score_loss(eval_y, y_pred))
    except Exception:
        auc, brier = float("nan"), float("nan")
    n_best = int(getattr(clf, "best_iteration", clf.n_estimators) or clf.n_estimators)
    return clf, y_pred, dict(auc=auc, brier=brier, best_iter=n_best,
                              n_train=int(len(train_X)), n_eval=int(len(eval_X)),
                              pos_rate=float(train_y.mean()))


def train_queue_head(train_X, train_y, eval_X, eval_y, params, smoke_rounds=None):
    import xgboost as xgb
    p = dict(params)
    if smoke_rounds is not None:
        p["n_estimators"] = smoke_rounds
        p["early_stopping_rounds"] = min(20, smoke_rounds // 4 or 1)
    tr_mask = ~np.isnan(train_y)
    ev_mask = ~np.isnan(eval_y)
    if tr_mask.sum() < 10 or ev_mask.sum() < 10:
        return None, np.full(len(eval_X), np.nan), dict(r2=float("nan"), rmse=float("nan"),
                                                          best_iter=0, n_train=int(tr_mask.sum()),
                                                          n_eval=int(ev_mask.sum()))
    tX, ty = train_X[tr_mask], train_y[tr_mask]
    eX, ey = eval_X[ev_mask], eval_y[ev_mask]
    reg = xgb.XGBRegressor(**p)
    reg.fit(tX, ty, eval_set=[(eX, ey)], verbose=False)
    y_pred_eval_mask = reg.predict(eX)
    y_pred_full = np.full(len(eval_X), np.nan, dtype=np.float32)
    y_pred_full[ev_mask] = y_pred_eval_mask
    rmse = float(np.sqrt(np.mean((y_pred_eval_mask - ey) ** 2)))
    ss_res = float(np.sum((y_pred_eval_mask - ey) ** 2))
    ss_tot = float(np.sum((ey - ey.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    n_best = int(getattr(reg, "best_iteration", reg.n_estimators) or reg.n_estimators)
    return reg, y_pred_full, dict(r2=r2, rmse=rmse, best_iter=n_best,
                                    n_train=int(len(tX)), n_eval=int(len(eX)))


def fold_train(fold_idx, train_packs, eval_packs, smoke_rounds=None):
    train_X = np.concatenate([d["X"] for d in train_packs])
    eval_X = np.concatenate([d["X"] for d in eval_packs])
    np.nan_to_num(train_X, copy=False)
    np.nan_to_num(eval_X, copy=False)

    fold_res = {"fold": fold_idx,
                "train_dates": [d["date"] for d in train_packs],
                "eval_dates": [d["date"] for d in eval_packs],
                "n_train": int(len(train_X)), "n_eval": int(len(eval_X)),
                "feat_dim": int(train_X.shape[1])}
    preds_per_h = {}

    for h in HORIZONS:
        log.info(f"  fold{fold_idx} h={h}s — fitting fill+queue heads (X_dim={train_X.shape[1]})")
        t0 = time.time()
        train_yf = np.concatenate([d["ys"][h]["fill"] for d in train_packs])
        train_yq = np.concatenate([d["ys"][h]["queue"] for d in train_packs])
        eval_yf = np.concatenate([d["ys"][h]["fill"] for d in eval_packs])
        eval_yq = np.concatenate([d["ys"][h]["queue"] for d in eval_packs])

        with ThreadPoolExecutor(max_workers=2) as ex:
            fut_fill = ex.submit(train_fill_head, train_X, train_yf, eval_X, eval_yf,
                                 XGB_PARAMS_FILL, smoke_rounds)
            fut_q = ex.submit(train_queue_head, train_X, train_yq, eval_X, eval_yq,
                              XGB_PARAMS_QUEUE, smoke_rounds)
            clf_fill, p_fill, fill_m = fut_fill.result()
            reg_q, p_queue, q_m = fut_q.result()

        clf_fill.save_model(str(OUTPUT_DIR / f"fold_{fold_idx:03d}_fill_{h}s.json"))
        if reg_q is not None:
            reg_q.save_model(str(OUTPUT_DIR / f"fold_{fold_idx:03d}_queue_{h}s.json"))

        dt = time.time() - t0
        log.info(
            f"  fold{fold_idx} h={h}s dt={dt:.1f}s | "
            f"fill AUC={fill_m['auc']:.4f} Brier={fill_m['brier']:.4f} | "
            f"queue R2={q_m['r2']:.4f} RMSE={q_m['rmse']:.4f} | "
            f"best_iter fill={fill_m['best_iter']} q={q_m['best_iter']}"
        )
        fold_res[f"fill_{h}s"] = fill_m
        fold_res[f"queue_{h}s"] = q_m
        preds_per_h[h] = dict(p_fill=p_fill, y_fill=eval_yf,
                              p_queue=p_queue, y_queue=eval_yq)

    npz_kwargs = {}
    for h in HORIZONS:
        npz_kwargs[f"p_fill_{h}s"] = preds_per_h[h]["p_fill"]
        npz_kwargs[f"y_fill_{h}s"] = preds_per_h[h]["y_fill"]
        npz_kwargs[f"p_queue_{h}s"] = preds_per_h[h]["p_queue"]
        npz_kwargs[f"y_queue_{h}s"] = preds_per_h[h]["y_queue"]
    np.savez(str(OUTPUT_DIR / f"fold_{fold_idx:03d}_oot.npz"), **npz_kwargs)

    try:
        import mlflow
        for h in HORIZONS:
            mlflow.log_metric(f"fold_fill_auc_{h}s", fold_res[f"fill_{h}s"]["auc"], step=fold_idx)
            mlflow.log_metric(f"fold_fill_brier_{h}s", fold_res[f"fill_{h}s"]["brier"], step=fold_idx)
            mlflow.log_metric(f"fold_queue_r2_{h}s", fold_res[f"queue_{h}s"]["r2"], step=fold_idx)
            mlflow.log_metric(f"fold_queue_rmse_{h}s", fold_res[f"queue_{h}s"]["rmse"], step=fold_idx)
    except Exception:
        pass
    return fold_res


def walk_forward(joined, train_window, eval_window, max_folds=None, smoke_rounds=None):
    fold_metrics = []
    n = len(joined)
    start = 0
    fold_idx = 0
    while start + train_window + eval_window <= n:
        if max_folds is not None and fold_idx >= max_folds:
            break
        train_packs = joined[start:start + train_window]
        eval_packs = joined[start + train_window:start + train_window + eval_window]
        log.info(
            f"FOLD {fold_idx}: train {train_packs[0]['date']}..{train_packs[-1]['date']} "
            f"eval {eval_packs[0]['date']}..{eval_packs[-1]['date']}"
        )
        m = fold_train(fold_idx, train_packs, eval_packs, smoke_rounds=smoke_rounds)
        fold_metrics.append(m)
        start += eval_window
        fold_idx += 1
        gc.collect()
    return fold_metrics


def concat_metrics(fold_metrics):
    out = {}
    for h in HORIZONS:
        pf_list, yf_list, pq_list, yq_list = [], [], [], []
        for fm in fold_metrics:
            d = np.load(str(OUTPUT_DIR / f"fold_{fm['fold']:03d}_oot.npz"))
            pf_list.append(d[f"p_fill_{h}s"]); yf_list.append(d[f"y_fill_{h}s"])
            pq_list.append(d[f"p_queue_{h}s"]); yq_list.append(d[f"y_queue_{h}s"])
        pf = np.concatenate(pf_list); yf = np.concatenate(yf_list)
        pq = np.concatenate(pq_list); yq = np.concatenate(yq_list)
        try:
            from sklearn.metrics import roc_auc_score, brier_score_loss
            auc = float(roc_auc_score(yf, pf)) if 0 < yf.sum() < len(yf) else float("nan")
            brier = float(brier_score_loss(yf, pf))
        except Exception:
            auc, brier = float("nan"), float("nan")
        q_mask = ~np.isnan(pq) & ~np.isnan(yq)
        if q_mask.sum() > 10:
            pq_v = pq[q_mask]; yq_v = yq[q_mask]
            rmse = float(np.sqrt(np.mean((pq_v - yq_v) ** 2)))
            ss_res = float(np.sum((pq_v - yq_v) ** 2))
            ss_tot = float(np.sum((yq_v - yq_v.mean()) ** 2))
            r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
        else:
            rmse, r2 = float("nan"), float("nan")
        out[f"fill_auc_{h}s"] = auc
        out[f"fill_brier_{h}s"] = brier
        out[f"queue_rmse_{h}s"] = rmse
        out[f"queue_r2_{h}s"] = r2
        out[f"n_oot_{h}s"] = int(len(pf))
        out[f"n_oot_queue_{h}s"] = int(q_mask.sum())
        log.info(
            f"CONCAT h={h}s n={len(pf):,} | fill AUC={auc:.4f} Brier={brier:.4f} | "
            f"queue R2={r2:.4f} RMSE={rmse:.4f} (n_q={q_mask.sum():,})"
        )
    return out


# v1 baseline for honest reporting (from queue_position_v1_xgb_dualhead/summary.json)
V1_BASELINE = {
    "fill_auc_1s": 0.6157, "queue_r2_1s": 0.0221,
    "fill_auc_5s": 0.6312, "queue_r2_5s": 0.0294,
    "fill_auc_10s": 0.6984, "queue_r2_10s": 0.0101,
}


def main():
    global HORIZONS
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-window", type=int, default=25)
    ap.add_argument("--eval-window", type=int, default=1)
    ap.add_argument("--max-folds", type=int, default=None)
    ap.add_argument("--smoke", action="store_true",
                    help="Run 2-fold sanity check, h=5s only, 200 rounds")
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

    import mlflow
    mlflow.set_tracking_uri(MLFLOW_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    run_name = f"qpos_v2_{time.strftime('%Y%m%d_%H%M')}"
    if args.smoke:
        run_name += "_SMOKE"
    run = mlflow.start_run(run_name=run_name)
    mlflow.log_params({
        "horizons_s": HORIZONS,
        "model": "XGB_dualhead_per_horizon_v2_aug",
        "tree_method": "hist",
        "device": "cuda:0",
        "n_estimators_cap": XGB_PARAMS_FILL["n_estimators"],
        "early_stopping_rounds": XGB_PARAMS_FILL["early_stopping_rounds"],
        "max_depth": XGB_PARAMS_FILL["max_depth"],
        "learning_rate": XGB_PARAMS_FILL["learning_rate"],
        "reg_alpha": XGB_PARAMS_FILL["reg_alpha"],
        "reg_lambda": XGB_PARAMS_FILL["reg_lambda"],
        "gamma": XGB_PARAMS_FILL["gamma"],
        "train_window_days": args.train_window,
        "eval_window_days": args.eval_window,
        "hc": "498_R4_aug",
        "ts_join_tol_ns": TS_TOL_NS,
        "smoke": args.smoke,
        "augmented_features": True,
    })
    log.info(f"MLflow exp: {EXPERIMENT_NAME}, run_id={run.info.run_id}, name={run_name}")

    t0 = time.time()
    joined, join_rate, aug_join_rate = load_all_dates_joined()
    log.info(f"Loaded {len(joined)} joined dates in {time.time()-t0:.1f}s")
    mlflow.log_metric("join_rate_of_signal", join_rate)
    mlflow.log_metric("aug_join_rate", aug_join_rate)
    mlflow.log_metric("n_joined_dates", len(joined))
    if len(joined) > 0:
        mlflow.log_metric("feat_dim", int(joined[0]["X"].shape[1]))

    # GATE: augmented-feature attach rate must be >= 95% per task spec
    if aug_join_rate < 0.95:
        log.error(
            f"AUG JOIN RATE {aug_join_rate:.2%} < 95% — STOP per task spec. "
            f"Suspect missing dates or ts mismatch."
        )
        mlflow.end_run(status="FAILED")
        sys.exit(3)

    if len(joined) < args.train_window + args.eval_window:
        log.error(
            f"Not enough joined dates ({len(joined)}) for WF "
            f"(need {args.train_window+args.eval_window})"
        )
        mlflow.end_run(status="FAILED")
        sys.exit(2)

    max_folds = args.max_folds
    smoke_rounds = None
    if args.smoke:
        max_folds = 2
        HORIZONS = [5]
        smoke_rounds = 200
        log.info(f"SMOKE: HORIZONS={HORIZONS} max_folds={max_folds} rounds={smoke_rounds}")

    fold_metrics = walk_forward(
        joined, train_window=args.train_window, eval_window=args.eval_window,
        max_folds=max_folds, smoke_rounds=smoke_rounds,
    )

    concat = concat_metrics(fold_metrics)
    for k, v in concat.items():
        try:
            import mlflow as mf
            mf.log_metric(f"concat_{k}", v if v == v else 0.0)
        except Exception:
            pass

    # SMOKE GATES (h=5s only)
    if args.smoke:
        auc5 = concat.get("fill_auc_5s", float("nan"))
        r25 = concat.get("queue_r2_5s", float("nan"))
        smoke_pass = (auc5 == auc5 and auc5 > 0.55) and (r25 == r25 and r25 > -0.05)
        log.info(f"SMOKE GATES: fill_auc_5s={auc5:.4f} (need>0.55) queue_r2_5s={r25:.4f} (need>-0.05) -> {'PASS' if smoke_pass else 'FAIL'}")
        mlflow.log_metric("smoke_pass", 1.0 if smoke_pass else 0.0)

    # Delta vs v1
    delta_vs_v1 = {}
    for h in HORIZONS:
        for metric in ("fill_auc", "queue_r2"):
            key = f"{metric}_{h}s"
            if key in concat and key in V1_BASELINE:
                delta_vs_v1[key] = float(concat[key]) - float(V1_BASELINE[key])
                try:
                    mlflow.log_metric(f"delta_v1_{key}", delta_vs_v1[key])
                except Exception:
                    pass

    # HONEST reporting
    warnings = []
    for h in HORIZONS:
        a = concat.get(f"fill_auc_{h}s", float("nan"))
        r = concat.get(f"queue_r2_{h}s", float("nan"))
        if a == a and a < 0.65:
            warnings.append(f"fill_AUC_{h}s={a:.4f} < 0.65 target")
        if r == r and r < 0.20:
            warnings.append(f"queue_R2_{h}s={r:.4f} < 0.20 target")
        # Noise warning
        d_auc = delta_vs_v1.get(f"fill_auc_{h}s", 0.0)
        if abs(d_auc) < 0.05:
            warnings.append(
                f"fill_AUC_{h}s delta_vs_v1={d_auc:+.4f} within fold noise (~0.05) — NOT a clear improvement"
            )
    if warnings:
        log.warning("HONEST FLAGS: " + "; ".join(warnings))
        log.warning("HC #498 R4: no spin. Report deltas as-is.")

    summary = {
        "hc": "498_R4_aug",
        "model": "XGB_dualhead_per_horizon_v2_aug",
        "horizons_s": HORIZONS,
        "xgb_params_fill": XGB_PARAMS_FILL,
        "xgb_params_queue": XGB_PARAMS_QUEUE,
        "train_window_days": args.train_window,
        "eval_window_days": args.eval_window,
        "n_folds": len(fold_metrics),
        "join_rate_of_signal": join_rate,
        "aug_join_rate": aug_join_rate,
        "ts_join_tol_ns": TS_TOL_NS,
        "feat_dim": int(joined[0]["X"].shape[1]) if joined else None,
        "concat_metrics": concat,
        "v1_baseline": V1_BASELINE,
        "delta_vs_v1": delta_vs_v1,
        "warnings": warnings,
        "smoke": args.smoke,
        "fold_metrics": fold_metrics,
    }
    with open(OUTPUT_DIR / "summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Wrote {OUTPUT_DIR/'summary.json'}")
    try:
        import mlflow as mf
        mf.log_artifact(str(OUTPUT_DIR / "summary.json"))
        mf.end_run()
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
