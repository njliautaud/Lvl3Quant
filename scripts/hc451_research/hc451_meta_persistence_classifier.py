#!/usr/bin/env python3
"""HC #451 meta-persistence classifier.

Trains a LightGBM binary classifier to predict whether a v3.4.2 CNN-Mamba
event will exhibit favorable 1s->10s sign persistence, using ONLY features
computable AT TRADE TIME (no look-ahead).

Inputs (read-only):
  output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_<DATE>.npz
    keys used (predictions/targets/masks at multiple horizons + sample_dates)

Target (binary):
  persistence_1s_10s == +1  i.e. sign(target_log_ret_1s) == sign(target_log_ret_10s)
  with a neutrality band of 0.5 ticks on the 1s leg AND
  with the trade in the FAVORABLE direction relative to the model's predicted side.

Features (per event, NO look-ahead):
  - pred_log_ret_1s, pred_log_ret_5s, pred_log_ret_10s (raw signed)
  - |pred_log_ret_1s|  (confidence)
  - pred_p_up_5s, pred_p_up_10s
  - pred_pred_mfe_30s_ticks, pred_pred_mae_30s_ticks (MFE/MAE head outputs)
  - pred_pred_realized_vol_30s_ticks
  - pred_p_reversal_15s, pred_p_reversal_30s
  - sign_1s (-1/0/+1)
  - within-day percentile rank of |pred_log_ret_1s|
  - causal sign-agreement (last 10 preds same sign as current)
  - time-of-day bucket (fraction of way through the day's samples)

Walk-forward split: first 80% of OOT days -> train, last 20% -> validation.
Early stopping on validation logloss.

Outputs:
  output/hc451_meta_persistence/meta_model.txt    (LGBM booster)
  output/hc451_meta_persistence/meta_oot_predictions/<date>.npz
        keys: meta_prob (float32), sample_idx (int32),
              pred_1s (float32), abs_pred_1s (float32)
  output/hc451_meta_persistence/training_metrics.json
"""
from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb
from sklearn.metrics import roc_auc_score, log_loss

LVL3 = Path("/home/jupiter/Lvl3Quant")
OOT_DIR = LVL3 / "output" / "cnn_mamba_v3_4_2_fixedmtl" / "oot_47day_perdate"
OUT_DIR = LVL3 / "output" / "hc451_meta_persistence"
PRED_OUT = OUT_DIR / "meta_oot_predictions"
PRED_OUT.mkdir(parents=True, exist_ok=True)

NEUT_BAND = 0.5  # ticks
RECENT_K = 10    # autocorr window

FEATURE_KEYS_RAW = [
    "pred_log_ret_1s",
    "pred_log_ret_5s",
    "pred_log_ret_10s",
    "pred_p_up_5s",
    "pred_p_up_10s",
    "pred_pred_mfe_30s_ticks",
    "pred_pred_mae_30s_ticks",
    "pred_pred_realized_vol_30s_ticks",
    "pred_p_reversal_15s",
    "pred_p_reversal_30s",
]
ENG_FEATURE_NAMES = [
    "abs_pred_1s",
    "sign_1s",
    "pct_rank_abs_1s_within_day",
    "recent_sign_agree_k10",
    "tod_frac",
    "pred_1s_minus_5s",
    "pred_5s_minus_10s",
    "mfe_minus_mae_30s",
]
FEATURE_NAMES = FEATURE_KEYS_RAW + ENG_FEATURE_NAMES


def _causal_recent_sign_agree(sign_arr: np.ndarray, k: int) -> np.ndarray:
    """Per-event: fraction of last k preds (excluding current) whose sign equals current.

    Strictly causal (no current/future leak): uses indices [i-k, i-1].
    Returns 0.0 for first k samples (warm-up).
    """
    n = sign_arr.size
    out = np.zeros(n, dtype=np.float32)
    if n <= k:
        return out
    # rolling: compare past k signs to current; count matches
    s = sign_arr.astype(np.int8)
    for i in range(k, n):
        past = s[i - k:i]
        # match count where past != 0 and equals current
        cur = s[i]
        if cur == 0:
            out[i] = 0.0
        else:
            out[i] = float((past == cur).sum()) / k
    return out


def _within_day_pct_rank(x: np.ndarray) -> np.ndarray:
    """Percentile rank in [0,1] of x within the day (ties broken by index)."""
    n = x.size
    order = np.argsort(x, kind="stable")
    ranks = np.empty(n, dtype=np.float32)
    ranks[order] = np.arange(n, dtype=np.float32)
    return ranks / max(1, n - 1)


def build_features_and_target(npz_path: Path) -> Tuple[pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """Load one day's NPZ -> (features_df, y, mask, sample_idx).

    y = 1 if persistence in FAVORABLE direction (sign(pred_1s) == sign(target_1s)
    == sign(target_10s) AND |target_1s| > NEUT_BAND), else 0.
    mask: True where both 1s and 10s targets are valid AND |pred_1s| > 0
          (zero-pred excluded — they wouldn't trade).
    """
    z = np.load(npz_path, allow_pickle=False)
    if "pred_log_ret_1s" not in z.files:
        z.close()
        return pd.DataFrame(columns=FEATURE_NAMES), np.empty(0, np.int8), np.empty(0, bool), np.empty(0, np.int32)

    feats: Dict[str, np.ndarray] = {}
    for k in FEATURE_KEYS_RAW:
        feats[k] = z[k].astype(np.float32)

    pred_1s = feats["pred_log_ret_1s"]
    pred_5s = feats["pred_log_ret_5s"]
    pred_10s = feats["pred_log_ret_10s"]
    mfe = feats["pred_pred_mfe_30s_ticks"]
    mae = feats["pred_pred_mae_30s_ticks"]

    abs_p1 = np.abs(pred_1s)
    sign_1s = np.where(pred_1s > 0, 1, np.where(pred_1s < 0, -1, 0)).astype(np.int8)
    pct_rank = _within_day_pct_rank(abs_p1)
    recent_agree = _causal_recent_sign_agree(sign_1s, RECENT_K)
    n = pred_1s.size
    tod_frac = np.arange(n, dtype=np.float32) / max(1, n - 1)

    eng = {
        "abs_pred_1s": abs_p1,
        "sign_1s": sign_1s.astype(np.float32),
        "pct_rank_abs_1s_within_day": pct_rank,
        "recent_sign_agree_k10": recent_agree,
        "tod_frac": tod_frac,
        "pred_1s_minus_5s": pred_1s - pred_5s,
        "pred_5s_minus_10s": pred_5s - pred_10s,
        "mfe_minus_mae_30s": mfe - mae,
    }
    df = pd.DataFrame({**feats, **eng})

    # target
    t1 = z["target_log_ret_1s"].astype(np.float32)
    t10 = z["target_log_ret_10s"].astype(np.float32)
    m1 = z["mask_log_ret_1s"].astype(np.float32) > 0
    m10 = z["mask_log_ret_10s"].astype(np.float32) > 0
    z.close()

    # favorable persistence: pred side == 1s realized side == 10s realized side, AND |t1| > band
    s_pred = sign_1s
    s_t1 = np.where(t1 > NEUT_BAND, 1, np.where(t1 < -NEUT_BAND, -1, 0)).astype(np.int8)
    s_t10 = np.where(t10 > NEUT_BAND, 1, np.where(t10 < -NEUT_BAND, -1, 0)).astype(np.int8)
    y_fav = (
        (s_pred != 0) & (s_pred == s_t1) & (s_t1 == s_t10)
    ).astype(np.int8)

    mask = m1 & m10 & (s_pred != 0)
    sample_idx = np.arange(n, dtype=np.int32)
    return df, y_fav, mask, sample_idx


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    print(f"[hc451-meta] scanning {OOT_DIR}", flush=True)

    files = sorted(f for f in OOT_DIR.iterdir() if f.name.startswith("oot_") and f.suffix == ".npz")
    dates = [f.stem.replace("oot_", "") for f in files]

    # ─── Load features per day ───
    per_day: Dict[str, dict] = {}
    for fp, dt in zip(files, dates):
        df, y, mask, idx = build_features_and_target(fp)
        if df.empty or mask.sum() == 0:
            print(f"[hc451-meta]   {dt}: empty/skipped", flush=True)
            continue
        per_day[dt] = {"df": df, "y": y, "mask": mask, "idx": idx}
        print(f"[hc451-meta]   {dt}: n={len(df)} valid={int(mask.sum())} pos_rate={float(y[mask].mean()):.3f}", flush=True)

    if not per_day:
        print("[hc451-meta] FATAL: no data loaded", file=sys.stderr)
        return 1

    dates_used = sorted(per_day.keys())
    n_train = max(1, int(len(dates_used) * 0.8))
    train_dates = dates_used[:n_train]
    val_dates = dates_used[n_train:]
    print(f"[hc451-meta] train_dates={len(train_dates)} ({train_dates[0]}..{train_dates[-1]}) val_dates={len(val_dates)} ({val_dates[0] if val_dates else '-'}..{val_dates[-1] if val_dates else '-'})", flush=True)

    def stack(dts):
        Xs, ys = [], []
        for d in dts:
            r = per_day[d]
            m = r["mask"]
            Xs.append(r["df"].loc[m, FEATURE_NAMES].values.astype(np.float32))
            ys.append(r["y"][m])
        if not Xs:
            return np.zeros((0, len(FEATURE_NAMES)), np.float32), np.zeros(0, np.int8)
        return np.concatenate(Xs, axis=0), np.concatenate(ys, axis=0)

    X_tr, y_tr = stack(train_dates)
    X_va, y_va = stack(val_dates)
    print(f"[hc451-meta] X_tr={X_tr.shape} pos_rate={y_tr.mean():.3f}  X_va={X_va.shape} pos_rate={y_va.mean():.3f}", flush=True)

    train_set = lgb.Dataset(X_tr, label=y_tr, feature_name=FEATURE_NAMES, free_raw_data=False)
    val_set = lgb.Dataset(X_va, label=y_va, feature_name=FEATURE_NAMES, reference=train_set, free_raw_data=False)

    params = dict(
        objective="binary",
        metric=["binary_logloss", "auc"],
        learning_rate=0.05,
        num_leaves=63,
        max_depth=-1,
        feature_fraction=0.85,
        bagging_fraction=0.85,
        bagging_freq=5,
        min_data_in_leaf=200,
        seed=451,
        verbosity=-1,
        n_jobs=8,
    )

    print(f"[hc451-meta] training LightGBM ...", flush=True)
    booster = lgb.train(
        params,
        train_set,
        num_boost_round=1000,
        valid_sets=[train_set, val_set],
        valid_names=["train", "val"],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=True),
                   lgb.log_evaluation(period=50)],
    )

    # OOT validation metrics
    p_va = booster.predict(X_va, num_iteration=booster.best_iteration)
    auc_va = float(roc_auc_score(y_va, p_va)) if len(np.unique(y_va)) > 1 else float("nan")
    ll_va = float(log_loss(y_va, np.clip(p_va, 1e-6, 1 - 1e-6)))
    base_va = float(y_va.mean())

    # Calibration in high-prob bins
    cal_bins = [0.5, 0.6, 0.7, 0.8, 0.9]
    cal = {}
    for lo in cal_bins:
        sel = p_va >= lo
        if sel.sum() > 100:
            cal[f"p_ge_{lo}"] = {"n": int(sel.sum()),
                                 "actual_pos_rate": float(y_va[sel].mean()),
                                 "mean_pred": float(p_va[sel].mean())}
        else:
            cal[f"p_ge_{lo}"] = {"n": int(sel.sum()), "actual_pos_rate": None, "mean_pred": None}

    # Overlap with top |pred_1s| confidence (the raw-confidence selector)
    abs_p1_va = np.abs(X_va[:, FEATURE_NAMES.index("pred_log_ret_1s")])
    overlap = {}
    for top_pct in [0.01, 0.02, 0.05]:
        k = max(1, int(len(p_va) * top_pct))
        if k < 5:
            continue
        top_meta = np.argpartition(-p_va, k - 1)[:k]
        top_conf = np.argpartition(-abs_p1_va, k - 1)[:k]
        inter = len(set(top_meta.tolist()) & set(top_conf.tolist()))
        overlap[f"top_{int(top_pct*100)}pct"] = {
            "k": k,
            "jaccard": float(inter / (2 * k - inter)) if (2 * k - inter) > 0 else 0.0,
            "meta_in_conf_pct": float(inter / k),
        }

    metrics = {
        "n_train": int(X_tr.shape[0]), "n_val": int(X_va.shape[0]),
        "train_dates": train_dates, "val_dates": val_dates,
        "auc_val": auc_va, "logloss_val": ll_va, "base_pos_rate_val": base_va,
        "best_iter": int(booster.best_iteration),
        "calibration_val": cal,
        "overlap_meta_vs_rawconf_val": overlap,
        "feature_names": FEATURE_NAMES,
        "feature_importance_gain": dict(zip(FEATURE_NAMES,
                                            booster.feature_importance(importance_type="gain").tolist())),
        "elapsed_sec": time.time() - t0,
    }
    (OUT_DIR / "training_metrics.json").write_text(json.dumps(metrics, indent=2, default=str))
    booster.save_model(str(OUT_DIR / "meta_model.txt"))

    # ─── Score every OOT day and dump per-day prediction NPZ ───
    print(f"[hc451-meta] scoring all OOT days for FIFO consumption ...", flush=True)
    for d, r in per_day.items():
        df_full = r["df"]
        # Score ALL rows (including masked ones) so indices align 1:1 with v3.4.2 OOT NPZ
        Xd = df_full.loc[:, FEATURE_NAMES].values.astype(np.float32)
        # Replace nan with 0 for inference safety (LightGBM handles nan natively, but be explicit)
        prob = booster.predict(Xd, num_iteration=booster.best_iteration).astype(np.float32)
        np.savez(PRED_OUT / f"{d}.npz",
                 meta_prob=prob,
                 sample_idx=r["idx"],
                 pred_1s=df_full["pred_log_ret_1s"].values.astype(np.float32),
                 abs_pred_1s=df_full["abs_pred_1s"].values.astype(np.float32),
                 mask=r["mask"].astype(np.bool_),
                 y_true=r["y"].astype(np.int8))

    print(f"[hc451-meta] done. AUC_val={auc_va:.4f}  base={base_va:.3f}  elapsed={time.time()-t0:.1f}s", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
