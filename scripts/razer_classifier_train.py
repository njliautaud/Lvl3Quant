#!/usr/bin/env python3
"""
razer_classifier_train.py — Train 2 meta-classifiers (PyTorch MLP-Classifier,
XGBoost-GPU Classifier) that LEARN to detect the confluence-trigger event.

Target: y_profitable_trigger (binary, ~1.6% of events).
Features: the 32 CNN-Mamba v4 prediction heads (NEVER the targets or trigger flags).

Walk-forward: sliding 15-day-train / 1-day-test, same scheme as razer_meta_train.py.
Output per-date NPZ per model with classifier probability:
  output/razer_classifier/cls_preds_<model>_<YYYYMMDD>.npz
    arrays: prob (shape: n_events), date (str)

Auto-detects GPU. Runs on Razer (Windows) or Jupiter (Linux).
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

# Cross-platform paths
if os.name == "nt":
    DATA_FILE = Path(r"C:\Users\claude\Lvl3Quant\data\class_features.parquet")
    OUT_DIR = Path(r"C:\Users\claude\Lvl3Quant\output\razer_classifier")
else:
    DATA_FILE = Path("/home/jupiter/Lvl3Quant/output/razer_classifier/class_features.parquet")
    OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/razer_classifier")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_FEATURES = {"pred_k"}
# Never use these as features
EXCLUDE_COLS = {"date", "pred_k",
                "y_trigger", "y_profitable_trigger",
                "trig_1s_x_5s_pup", "trig_30s_q90_x_1s",
                "trig_30s_x_1s", "trig_mfe_x_1s",
                "target_log_ret_1s", "target_log_ret_5s",
                "target_log_ret_10s", "target_log_ret_30s"}

TARGET = "y_profitable_trigger"
K_TRAIN_DAYS = 15


def load_data():
    print(f"[cls_train] loading {DATA_FILE}", flush=True)
    df = pd.read_parquet(DATA_FILE)
    feat_cols = sorted([c for c in df.columns
                        if c.startswith("pred_") and c not in EXCLUDE_COLS])
    print(f"[cls_train] features: {len(feat_cols)}", flush=True)
    print(f"[cls_train] target: {TARGET}  positives: {df[TARGET].sum():,} ({df[TARGET].mean()*100:.3f}%)", flush=True)
    print(f"[cls_train] rows: {len(df):,}  dates: {df['date'].nunique()}", flush=True)
    return df, feat_cols


def build_folds(unique_dates: List[str], k_train: int) -> List[Tuple[List[str], str]]:
    folds = []
    for i in range(k_train, len(unique_dates)):
        train_dates = unique_dates[i - k_train:i]
        test_date = unique_dates[i]
        folds.append((train_dates, test_date))
    return folds


# --------------------------------------------------------------------------------------
# XGBoost-GPU Classifier
# --------------------------------------------------------------------------------------
def train_xgb_cls(df: pd.DataFrame, feat_cols: List[str], folds, device: str):
    import xgboost as xgb
    print(f"[cls_train] xgboost version: {xgb.__version__}", flush=True)

    importances_acc = {f: 0.0 for f in feat_cols}
    importance_folds = 0
    per_day_preds = {}

    for fold_i, (train_dates, test_date) in enumerate(folds):
        t0 = time.time()
        tr_mask = df["date"].isin(train_dates)
        te_mask = df["date"] == test_date
        X_tr = df.loc[tr_mask, feat_cols].values.astype(np.float32)
        y_tr = df.loc[tr_mask, TARGET].values.astype(np.int32)
        X_te = df.loc[te_mask, feat_cols].values.astype(np.float32)

        pos = int(y_tr.sum())
        neg = int(len(y_tr) - pos)
        spw = max(1.0, neg / max(1, pos))

        # xgboost 2.x: device='cuda', tree_method='hist'
        # xgboost 1.x: tree_method='gpu_hist'
        try:
            major = int(xgb.__version__.split(".")[0])
        except Exception:
            major = 2
        if major >= 2:
            params = dict(
                objective="binary:logistic",
                tree_method="hist",
                device=device,  # 'cuda' or 'cpu'
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw,
                eval_metric="logloss",
                verbosity=0,
            )
        else:
            params = dict(
                objective="binary:logistic",
                tree_method="gpu_hist" if device == "cuda" else "hist",
                n_estimators=200,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                scale_pos_weight=spw,
                eval_metric="logloss",
                verbosity=0,
            )

        model = xgb.XGBClassifier(**params)
        model.fit(X_tr, y_tr)
        prob = model.predict_proba(X_te)[:, 1]
        per_day_preds[test_date] = prob

        # Accumulate importances
        imp = dict(zip(feat_cols, model.feature_importances_))
        for k, v in imp.items():
            importances_acc[k] += float(v)
        importance_folds += 1

        print(f"  [xgb][fold {fold_i+1}/{len(folds)}] test={test_date}  n_tr={len(y_tr):,}  pos={pos:,}  spw={spw:.1f}  n_te={X_te.shape[0]:,}  took={time.time()-t0:.1f}s", flush=True)

    # Average importances
    if importance_folds > 0:
        importances_acc = {k: v / importance_folds for k, v in importances_acc.items()}
    return per_day_preds, importances_acc


# --------------------------------------------------------------------------------------
# PyTorch MLP Classifier
# --------------------------------------------------------------------------------------
def train_mlp_cls(df: pd.DataFrame, feat_cols: List[str], folds, device: str):
    import torch
    import torch.nn as nn

    torch_device = torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    print(f"[cls_train] torch device: {torch_device}  CUDA available: {torch.cuda.is_available()}", flush=True)
    if torch_device.type == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}", flush=True)

    per_day_preds = {}

    class MLP(nn.Module):
        def __init__(self, d_in):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(d_in, 64), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(64, 32), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(32, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)

    for fold_i, (train_dates, test_date) in enumerate(folds):
        t0 = time.time()
        tr_mask = df["date"].isin(train_dates)
        te_mask = df["date"] == test_date
        X_tr = df.loc[tr_mask, feat_cols].values.astype(np.float32)
        y_tr = df.loc[tr_mask, TARGET].values.astype(np.float32)
        X_te = df.loc[te_mask, feat_cols].values.astype(np.float32)

        # Z-normalize per-fold using train stats
        mu = X_tr.mean(axis=0, keepdims=True)
        sd = X_tr.std(axis=0, keepdims=True)
        sd[sd < 1e-8] = 1.0
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd

        # Class weight for BCE
        pos = float(y_tr.sum())
        neg = float(len(y_tr) - pos)
        pos_weight = torch.tensor([max(1.0, neg / max(1.0, pos))], device=torch_device)

        model = MLP(d_in=X_tr.shape[1]).to(torch_device)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        Xt = torch.from_numpy(X_tr).to(torch_device)
        yt = torch.from_numpy(y_tr).to(torch_device)

        batch = 65536
        n_epochs = 10
        n = Xt.shape[0]
        model.train()
        for ep in range(n_epochs):
            perm = torch.randperm(n, device=torch_device)
            ep_loss = 0.0
            steps = 0
            for s in range(0, n, batch):
                idx = perm[s:s + batch]
                logits = model(Xt[idx])
                loss = loss_fn(logits, yt[idx])
                opt.zero_grad()
                loss.backward()
                opt.step()
                ep_loss += float(loss.item())
                steps += 1
            # only print last epoch loss
        model.eval()
        with torch.no_grad():
            Xe = torch.from_numpy(X_te).to(torch_device)
            logits_te = model(Xe).cpu().numpy()
            prob = 1.0 / (1.0 + np.exp(-logits_te))
        per_day_preds[test_date] = prob
        print(f"  [mlp][fold {fold_i+1}/{len(folds)}] test={test_date}  n_tr={n:,}  pos={int(pos):,}  pw={pos_weight.item():.1f}  n_te={X_te.shape[0]:,}  took={time.time()-t0:.1f}s", flush=True)

    return per_day_preds


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------
def detect_gpu_device():
    # Prefer CUDA. Both XGB and torch will use 'cuda' when available.
    try:
        import torch
        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    # XGB can use GPU even without torch — try a quick probe
    try:
        import xgboost as xgb
        dtest = xgb.DMatrix(np.zeros((4, 4), dtype=np.float32))
        # crude probe — actual check would be GPU memory query
        return "cuda" if os.environ.get("CUDA_VISIBLE_DEVICES", "") or os.path.exists("/usr/lib/x86_64-linux-gnu/libcuda.so") or os.name == "nt" else "cpu"
    except Exception:
        return "cpu"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default=None, help="'cuda' or 'cpu' (auto-detect if not set)")
    ap.add_argument("--models", default="xgb,mlp", help="comma-separated subset of xgb,mlp")
    args = ap.parse_args()

    device = args.device or detect_gpu_device()
    print(f"[cls_train] device: {device}", flush=True)
    print(f"[cls_train] OUT_DIR: {OUT_DIR}", flush=True)

    df, feat_cols = load_data()
    unique_dates = sorted(df["date"].unique().tolist())
    folds = build_folds(unique_dates, K_TRAIN_DAYS)
    print(f"[cls_train] {len(folds)} walk-forward folds", flush=True)

    models = args.models.split(",")
    importances = {}

    if "xgb" in models:
        print("\n========== XGB CLASSIFIER ==========", flush=True)
        t0 = time.time()
        preds_xgb, imp_xgb = train_xgb_cls(df, feat_cols, folds, device)
        importances["xgb"] = imp_xgb
        print(f"[cls_train] xgb total: {time.time()-t0:.1f}s", flush=True)
        # Save per-day
        for d, p in preds_xgb.items():
            np.savez(OUT_DIR / f"cls_preds_xgb_{d}.npz", prob=p, date=d)

    if "mlp" in models:
        print("\n========== MLP CLASSIFIER ==========", flush=True)
        t0 = time.time()
        preds_mlp = train_mlp_cls(df, feat_cols, folds, device)
        print(f"[cls_train] mlp total: {time.time()-t0:.1f}s", flush=True)
        for d, p in preds_mlp.items():
            np.savez(OUT_DIR / f"cls_preds_mlp_{d}.npz", prob=p, date=d)

    # Save importances
    if importances:
        with open(OUT_DIR / "cls_feature_importances.json", "w") as f:
            json.dump(importances, f, indent=2)
        print(f"\n[cls_train] saved feature importances", flush=True)

    print("[cls_train] done.", flush=True)


if __name__ == "__main__":
    main()
