#!/usr/bin/env python3
"""
razer_meta_train.py — Train 3 meta-models (XGBoost-GPU, LightGBM-GPU, PyTorch MLP)
that combine 32 CNN-Mamba v4 prediction heads into a single learned alpha signal.

Target: target_log_ret_5s (5-second realized log return, in ES ticks).

Walk-forward: sliding folds. For each ordered date d in the OOT, train on the
prior K_TRAIN_DAYS dates and predict day d. K_TRAIN_DAYS=15 (need at least one
training fold; v4 OOT is 32 days). Drop rows with NaN target.

NO LEAKAGE — features are only the 32 pred_* heads (NEVER targets).

Outputs per-date per-model NPZs:
  output/razer_meta/meta_preds_<model>_<YYYYMMDD>.npz
Plus a tree feature importances JSON for XGB and LGBM.

Usage on Razer (Windows): python razer_meta_train.py
On Jupiter (fallback): python3 razer_meta_train.py --cpu

Auto-detects GPU. If CUDA available, uses it; falls back to CPU otherwise.
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------------------
# CONFIG
# --------------------------------------------------------------------------------------
# Cross-platform path handling — Razer is Windows, Jupiter is Linux.
if os.name == "nt":
    DATA_FILE = Path(r"C:\Users\claude\Lvl3Quant\data\meta_features.parquet")
    OUT_DIR = Path(r"C:\Users\claude\Lvl3Quant\output\razer_meta")
else:
    DATA_FILE = Path("/home/jupiter/Lvl3Quant/output/razer_meta/meta_features.parquet")
    OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/razer_meta")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# 32 prediction heads (exclude pred_k which is just the row index)
EXCLUDE_FEATURES = {"pred_k"}
TARGET = "target_log_ret_5s"

# Walk-forward: train on prior K_TRAIN_DAYS, predict next day. Sliding.
K_TRAIN_DAYS = 15


# --------------------------------------------------------------------------------------
# DATA
# --------------------------------------------------------------------------------------
def load_data():
    print(f"[meta_train] loading {DATA_FILE}")
    df = pd.read_parquet(DATA_FILE)
    feat_cols = sorted([c for c in df.columns if c.startswith("pred_") and c not in EXCLUDE_FEATURES])
    print(f"[meta_train] features: {len(feat_cols)}")
    if TARGET not in df.columns:
        raise RuntimeError(f"Target {TARGET} not in parquet columns: {df.columns.tolist()}")
    # Drop NaN targets
    before = len(df)
    df = df.dropna(subset=[TARGET]).reset_index(drop=True)
    print(f"[meta_train] rows: {before} -> {len(df)} after dropping NaN target")
    print(f"[meta_train] unique dates: {df['date'].nunique()}")
    return df, feat_cols


def build_folds(unique_dates: List[str], k_train: int):
    """Sliding walk-forward folds. Returns list of (train_dates, test_date)."""
    folds = []
    for i in range(k_train, len(unique_dates)):
        train_dates = unique_dates[i - k_train:i]
        test_date = unique_dates[i]
        folds.append((train_dates, test_date))
    return folds


# --------------------------------------------------------------------------------------
# MODELS
# --------------------------------------------------------------------------------------
def train_xgb(X_train, y_train, X_test, use_gpu: bool):
    import xgboost as xgb
    params = dict(
        n_estimators=400,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="reg:squarederror",
        verbosity=0,
    )
    if use_gpu:
        # xgboost 2.0+ API: device='cuda', tree_method='hist'
        params["device"] = "cuda"
        params["tree_method"] = "hist"
    else:
        params["tree_method"] = "hist"
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    importances = model.feature_importances_
    return preds, importances


def train_lgbm(X_train, y_train, X_test, use_gpu: bool):
    import lightgbm as lgb
    params = dict(
        n_estimators=400,
        num_leaves=64,
        learning_rate=0.05,
        feature_fraction=0.8,
        bagging_fraction=0.8,
        bagging_freq=5,
        min_data_in_leaf=50,
        reg_alpha=0.1,
        reg_lambda=1.0,
        objective="regression",
        verbose=-1,
    )
    if use_gpu:
        # LightGBM GPU build; falls back if not available
        params["device_type"] = "gpu"
    model = lgb.LGBMRegressor(**params)
    try:
        model.fit(X_train, y_train)
    except Exception as e:
        # GPU build sometimes errors on Windows — fall back to CPU.
        if use_gpu:
            print(f"  [lgbm] GPU failed ({e}), falling back to CPU")
            params["device_type"] = "cpu"
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
        else:
            raise
    preds = model.predict(X_test)
    importances = model.feature_importances_
    return preds, importances


def train_mlp(X_train, y_train, X_test, use_gpu: bool):
    import torch
    import torch.nn as nn

    device = torch.device("cuda" if (use_gpu and torch.cuda.is_available()) else "cpu")

    # Standardize features (use train stats)
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-8
    Xtr = ((X_train - mu) / sd).astype(np.float32)
    Xte = ((X_test - mu) / sd).astype(np.float32)
    ytr = y_train.astype(np.float32)

    Xtr_t = torch.from_numpy(Xtr).to(device)
    Xte_t = torch.from_numpy(Xte).to(device)
    ytr_t = torch.from_numpy(ytr).to(device)

    n_feat = Xtr.shape[1]
    model = nn.Sequential(
        nn.Linear(n_feat, 64), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.2),
        nn.Linear(32, 1),
    ).to(device)

    opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-5)
    loss_fn = nn.MSELoss()

    n = Xtr_t.shape[0]
    batch_size = 4096
    n_epochs = 8
    for epoch in range(n_epochs):
        perm = torch.randperm(n, device=device)
        total_loss = 0.0
        n_batches = 0
        model.train()
        for start in range(0, n, batch_size):
            idx = perm[start:start + batch_size]
            xb = Xtr_t[idx]
            yb = ytr_t[idx]
            opt.zero_grad()
            pred = model(xb).squeeze(-1)
            loss = loss_fn(pred, yb)
            loss.backward()
            opt.step()
            total_loss += float(loss.item())
            n_batches += 1
        if epoch == 0 or epoch == n_epochs - 1:
            print(f"  [mlp] epoch {epoch+1}/{n_epochs} loss={total_loss/max(n_batches,1):.4f}")

    model.eval()
    with torch.no_grad():
        preds = model(Xte_t).squeeze(-1).cpu().numpy()
    return preds


# --------------------------------------------------------------------------------------
# MAIN
# --------------------------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cpu", action="store_true", help="Force CPU (no GPU)")
    ap.add_argument("--models", default="xgb,lgbm,mlp",
                    help="Comma-separated subset: xgb,lgbm,mlp")
    args = ap.parse_args()

    # GPU detection
    try:
        import torch
        cuda_ok = torch.cuda.is_available()
        if cuda_ok:
            print(f"[meta_train] CUDA OK: {torch.cuda.get_device_name(0)}")
        else:
            print(f"[meta_train] CUDA not available")
    except Exception:
        cuda_ok = False
    use_gpu = (cuda_ok and not args.cpu)
    print(f"[meta_train] use_gpu={use_gpu}")

    df, feat_cols = load_data()
    unique_dates = sorted(df["date"].unique().tolist())
    folds = build_folds(unique_dates, K_TRAIN_DAYS)
    print(f"[meta_train] folds: {len(folds)} (first test date: {folds[0][1]}, last: {folds[-1][1]})")

    models_to_train = [m.strip() for m in args.models.split(",")]
    print(f"[meta_train] models: {models_to_train}")

    feat_imps_xgb = {f: 0.0 for f in feat_cols}
    feat_imps_lgbm = {f: 0.0 for f in feat_cols}
    n_imp_folds = 0

    t_start = time.time()
    for fi, (train_dates, test_date) in enumerate(folds, start=1):
        t_fold = time.time()
        train_mask = df["date"].isin(train_dates)
        test_mask = df["date"] == test_date
        X_train = df.loc[train_mask, feat_cols].values.astype(np.float32)
        y_train = df.loc[train_mask, TARGET].values.astype(np.float32)
        X_test = df.loc[test_mask, feat_cols].values.astype(np.float32)
        y_test = df.loc[test_mask, TARGET].values.astype(np.float32)
        if len(X_test) < 100:
            print(f"  [fold {fi}/{len(folds)}] {test_date}: too few test rows ({len(X_test)}), skip.")
            continue
        print(f"\n[fold {fi}/{len(folds)}] test={test_date} train_n={len(X_train)} test_n={len(X_test)}")

        results_for_day = {"date": test_date, "y_test": y_test, "feat_cols": np.array(feat_cols)}

        if "xgb" in models_to_train:
            t = time.time()
            preds_xgb, imp_xgb = train_xgb(X_train, y_train, X_test, use_gpu)
            print(f"  [xgb] {time.time()-t:.1f}s")
            results_for_day["preds_xgb"] = preds_xgb
            for f, w in zip(feat_cols, imp_xgb):
                feat_imps_xgb[f] += float(w)
            # Save per-day NPZ
            out_path = OUT_DIR / f"meta_preds_xgb_{test_date}.npz"
            np.savez_compressed(out_path, date=test_date, preds=preds_xgb,
                                y_test=y_test, feat_cols=np.array(feat_cols),
                                feat_importances=imp_xgb)
            print(f"  [xgb] wrote {out_path.name}")

        if "lgbm" in models_to_train:
            t = time.time()
            preds_lgbm, imp_lgbm = train_lgbm(X_train, y_train, X_test, use_gpu)
            print(f"  [lgbm] {time.time()-t:.1f}s")
            results_for_day["preds_lgbm"] = preds_lgbm
            for f, w in zip(feat_cols, imp_lgbm):
                feat_imps_lgbm[f] += float(w)
            out_path = OUT_DIR / f"meta_preds_lgbm_{test_date}.npz"
            np.savez_compressed(out_path, date=test_date, preds=preds_lgbm,
                                y_test=y_test, feat_cols=np.array(feat_cols),
                                feat_importances=imp_lgbm)
            print(f"  [lgbm] wrote {out_path.name}")

        if "mlp" in models_to_train:
            t = time.time()
            preds_mlp = train_mlp(X_train, y_train, X_test, use_gpu)
            print(f"  [mlp] {time.time()-t:.1f}s")
            out_path = OUT_DIR / f"meta_preds_mlp_{test_date}.npz"
            np.savez_compressed(out_path, date=test_date, preds=preds_mlp,
                                y_test=y_test, feat_cols=np.array(feat_cols))
            print(f"  [mlp] wrote {out_path.name}")

        n_imp_folds += 1
        print(f"  [fold {fi}] elapsed {time.time()-t_fold:.1f}s | total {time.time()-t_start:.1f}s")

    # Aggregate feature importances
    if n_imp_folds > 0:
        for f in feat_cols:
            feat_imps_xgb[f] /= n_imp_folds
            feat_imps_lgbm[f] /= n_imp_folds
        imp_summary = {
            "xgb": sorted(feat_imps_xgb.items(), key=lambda x: -x[1]),
            "lgbm": sorted(feat_imps_lgbm.items(), key=lambda x: -x[1]),
            "n_folds": n_imp_folds,
            "target": TARGET,
            "features": feat_cols,
        }
        out_path = OUT_DIR / "feature_importances.json"
        out_path.write_text(json.dumps(imp_summary, indent=2))
        print(f"\n[meta_train] wrote {out_path}")
        print(f"\n[meta_train] Top 10 XGB features:")
        for f, w in imp_summary["xgb"][:10]:
            print(f"   {f:40s} {w:.5f}")
        print(f"\n[meta_train] Top 10 LGBM features:")
        for f, w in imp_summary["lgbm"][:10]:
            print(f"   {f:40s} {w:.5f}")

    print(f"\n[meta_train] DONE in {time.time()-t_start:.1f}s")


if __name__ == "__main__":
    main()
