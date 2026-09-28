#!/usr/bin/env python3
"""
meta_confluence_v2_realpnl.py — Meta-classifier with REAL profitability labels.

CHANGE from v1: Target is now y_realprofitable, defined as:
  (realized signed net ticks after 0.376t passive-limit commission) > +0.10 ticks

This requires finding the realized-tick column in class_features.parquet.
If not available, fallback: (sign(side_pred) * signed_log_ret_30s * tick_value - 0.376) > 0.10

Walk-forward 17 folds (15-day train / 1-day test, sliding).
Two models: XGBoost-GPU + PyTorch MLP

Output: /home/nick/Lvl3Quant/output/razer_meta_confluence_v2_realpnl/
  - preds*.npz files (fold outputs)
  - eval_summary.csv (AUC, top-1% precision, realized net ticks, daily Sharpe per-regime)
  - headline.txt (summary metrics)

Auto-detects GPU. Cross-platform (Razer Windows / Jupiter Linux).
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple, Dict, Any

import numpy as np
import pandas as pd

if os.name == "nt":
    CLASS_FEATURES = Path(r"C:\Users\claude\Lvl3Quant\data\class_features.parquet")
    OUT_DIR = Path(r"C:\Users\claude\Lvl3Quant\output\razer_meta_confluence_v2_realpnl")
    PAIR_MATRIX = Path(r"C:\Users\claude\Lvl3Quant\data\pair_matrix.parquet")
else:
    CLASS_FEATURES = Path("/home/nick/Lvl3Quant/output/razer_classifier/class_features.parquet")
    OUT_DIR = Path("/home/nick/Lvl3Quant/output/razer_meta_confluence_v2_realpnl")
    PAIR_MATRIX = Path("/home/nick/Lvl3Quant/output/stream_backtest_v2/pair_matrix.parquet")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_COLS = {"date", "pred_k", "y_trigger", "y_profitable_trigger",
                "trig_1s_x_5s_pup", "trig_30s_q90_x_1s", "trig_30s_x_1s", "trig_mfe_x_1s",
                "target_log_ret_1s", "target_log_ret_5s", "target_log_ret_10s", "target_log_ret_30s"}
TARGET = "y_realprofitable"
K_TRAIN_DAYS = 15
TOP_N_PAIRS = 50
CONF_PCT_TO_Q = {1.0: 0.99, 5.0: 0.95, 10.0: 0.90}
COMMISSION_TICKS = 0.376
PROFITABILITY_THRESHOLD = 0.10

NON_DIRECTIONAL_HEADS = {
    "pred_p_reversal_15s", "pred_p_reversal_30s", "pred_p_reversal_60s",
    "pred_pred_mae_30s_ticks", "pred_pred_mae_60s_ticks",
    "pred_pred_mfe_30s_ticks", "pred_pred_mfe_60s_ticks",
    "pred_pred_realized_vol_30s_ticks", "pred_pred_time_to_mfe_secs", "pred_k",
}

def directional_signal(name: str, vals: np.ndarray) -> np.ndarray:
    if name in NON_DIRECTIONAL_HEADS:
        return np.zeros_like(vals)
    if name.startswith("pred_p_up_"):
        return vals - 0.5
    if "fifo" in name:
        return vals
    return vals

def build_real_profitability_label(df: pd.DataFrame) -> Tuple[str, np.ndarray]:
    print("[v2] Building real profitability label...", flush=True)
    
    realized_cols = [c for c in df.columns
                     if 'realized' in c.lower() and 'ticks' in c.lower() and 'net' in c.lower()]
    realized_cols += [c for c in df.columns if 'signed_net' in c.lower()]
    
    if realized_cols:
        col = realized_cols[0]
        print(f"[v2] Found realized column: {col}", flush=True)
        y = (df[col].values > PROFITABILITY_THRESHOLD).astype(np.int32)
        desc = f"Using {col} > {PROFITABILITY_THRESHOLD}"
        return desc, y
    
    if "pred_side_30s" in df.columns and "signed_log_ret_30s" in df.columns:
        print("[v2] Using fallback: (sign(side) * log_ret_30s * tick_value - comm) > 0.10", flush=True)
        side = np.sign(df["pred_side_30s"].values.astype(np.float64))
        log_ret = df["signed_log_ret_30s"].values.astype(np.float64)
        tick_value = 12.50
        realized_net = side * log_ret * tick_value - COMMISSION_TICKS
        y = (realized_net > PROFITABILITY_THRESHOLD).astype(np.int32)
        desc = "Fallback: (sign(side)*log_ret_30s*tick_value - 0.376) > 0.10"
        return desc, y
    
    if "y_profitable_trigger" in df.columns:
        print("[v2] WARNING: Fallback to y_profitable_trigger (circular risk)", flush=True)
        y = df["y_profitable_trigger"].values.astype(np.int32)
        desc = "Fallback: y_profitable_trigger (original, circular)"
        return desc, y
    
    raise ValueError("Could not build real profitability label")

def load_data() -> Tuple[pd.DataFrame, List[str], str]:
    print(f"[v2] Loading class features {CLASS_FEATURES}", flush=True)
    df = pd.read_parquet(CLASS_FEATURES)
    label_desc, y_real = build_real_profitability_label(df)
    df[TARGET] = y_real
    
    feat_cols = sorted([c for c in df.columns if c.startswith("pred_") and c not in EXCLUDE_COLS])
    print(f"[v2] features: {len(feat_cols)}  rows: {len(df):,}  dates: {df['date'].nunique()}", flush=True)
    print(f"[v2] target {TARGET}: {df[TARGET].sum():,} pos ({df[TARGET].mean()*100:.3f}%)", flush=True)
    print(f"[v2] Label: {label_desc}", flush=True)
    return df, feat_cols, label_desc

def build_confluence_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    print(f"[v2] Loading pair matrix {PAIR_MATRIX}", flush=True)
    pm = pd.read_parquet(PAIR_MATRIX)
    pm_robust = pm[(pm["net_ticks_after_cost"] > 0) & (pm["day_conc_pass"]) & (pm["n_conf_trades"] >= 100)].copy()
    pm_robust = pm_robust.sort_values("sharpe_per_trade", ascending=False).head(TOP_N_PAIRS)
    print(f"[v2] top-{len(pm_robust)} pairs", flush=True)
    
    confl_cols = []
    for idx, row in enumerate(pm_robust.itertuples(index=False)):
        ha, hb, pct = row.head_a, row.head_b, row.conf_top_pct
        if ha not in df.columns or hb not in df.columns:
            continue
        conf_q = CONF_PCT_TO_Q.get(pct, 0.95)
        sa = directional_signal(ha, df[ha].values.astype(np.float64))
        sb = directional_signal(hb, df[hb].values.astype(np.float64))
        m = np.isfinite(sa) & np.isfinite(sb) & (sa != 0) & (sb != 0)
        if m.sum() < 100:
            continue
        thr_a, thr_b = np.quantile(np.abs(sa[m]), conf_q), np.quantile(np.abs(sb[m]), conf_q)
        same = np.sign(sa) == np.sign(sb)
        signal = ((np.abs(sa) >= thr_a) & (np.abs(sb) >= thr_b) & same & (np.sign(sa) != 0)).astype(np.int8)
        cname = f"confl_p{idx:02d}_{ha[:12]}_x_{hb[:12]}_top{pct:.0f}"
        df[cname], df[cname + "_signed"] = signal, (signal * np.sign(sa)).astype(np.int8)
        confl_cols.extend([cname, cname + "_signed"])
    print(f"[v2] added {len(confl_cols)} confluence features", flush=True)
    return df, confl_cols

def build_folds(unique_dates: List[str], k_train: int) -> List[Tuple[List[str], str]]:
    return [(unique_dates[i - k_train:i], unique_dates[i]) for i in range(k_train, len(unique_dates))]

def train_xgb(df, feat_cols, folds, device):
    import xgboost as xgb
    print(f"[xgb] xgboost {xgb.__version__}  device={device}", flush=True)
    
    for fold_i, (train_dates, test_date) in enumerate(folds):
        t0 = time.time()
        tr, te = df[df["date"].isin(train_dates)], df[df["date"] == test_date]
        X_tr, y_tr = tr[feat_cols].values.astype(np.float32), tr[TARGET].values.astype(np.int32)
        X_te = te[feat_cols].values.astype(np.float32)
        pos, neg = int(y_tr.sum()), len(y_tr) - int(y_tr.sum())
        spw = max(1.0, neg / max(1, pos))
        
        try:
            major = int(xgb.__version__.split(".")[0])
        except:
            major = 2
        
        params = dict(objective="binary:logistic", tree_method="hist", device=device,
                      n_estimators=500, max_depth=6, learning_rate=0.05,
                      subsample=0.8, colsample_bytree=0.8, scale_pos_weight=spw, 
                      eval_metric="logloss", verbosity=0) if major >= 2 else \
                 dict(objective="binary:logistic", tree_method="gpu_hist" if device == "cuda" else "hist",
                      n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8,
                      colsample_bytree=0.8, scale_pos_weight=spw, eval_metric="logloss", verbosity=0)
        
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1].astype(np.float32)
        np.savez(OUT_DIR / f"meta_confl_v2_preds_xgb_{test_date}.npz", prob=prob, date=str(test_date))
        print(f"[xgb] fold {fold_i+1}/{len(folds)}  test={test_date}  n_tr={len(tr):,}  "
              f"n_te={len(te):,}  prob_mean={prob.mean():.4f}  t={time.time()-t0:.1f}s", flush=True)

def train_mlp(df, feat_cols, folds, device):
    import torch
    import torch.nn as nn
    print(f"[mlp] torch {torch.__version__}  cuda={torch.cuda.is_available()}", flush=True)
    use_cuda, dev = device == "cuda" and torch.cuda.is_available(), \
                    torch.device("cuda" if (device == "cuda" and torch.cuda.is_available()) else "cpu")
    
    class MLP(nn.Module):
        def __init__(self, in_dim):
            super().__init__()
            self.net = nn.Sequential(
                nn.Linear(in_dim, 256), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(256, 128), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(128, 64), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(64, 1),
            )
        def forward(self, x):
            return self.net(x).squeeze(-1)
    
    for fold_i, (train_dates, test_date) in enumerate(folds):
        t0 = time.time()
        tr, te = df[df["date"].isin(train_dates)], df[df["date"] == test_date]
        X_tr, y_tr = tr[feat_cols].values.astype(np.float32), tr[TARGET].values.astype(np.float32)
        X_te = te[feat_cols].values.astype(np.float32)
        
        mu, sd = X_tr.mean(axis=0), X_tr.std(axis=0) + 1e-6
        X_tr, X_te = (X_tr - mu) / sd, (X_te - mu) / sd
        
        pos, neg = float(y_tr.sum()), float(len(y_tr) - y_tr.sum())
        pos_w = max(1.0, neg / max(1.0, pos))
        
        model, opt = MLP(in_dim=X_tr.shape[1]).to(dev), None
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=dev))
        
        Xt, yt, Xe = torch.from_numpy(X_tr).to(dev), torch.from_numpy(y_tr).to(dev), torch.from_numpy(X_te).to(dev)
        
        BATCH, EPOCHS, n = 8192, 10, Xt.shape[0]
        for ep in range(EPOCHS):
            perm = torch.randperm(n, device=dev)
            for s in range(0, n, BATCH):
                idx = perm[s:s + BATCH]
                opt.zero_grad()
                out = model(Xt[idx])
                loss = loss_fn(out, yt[idx])
                loss.backward()
                opt.step()
        
        model.eval()
        with torch.no_grad():
            logits, prob = model(Xe), torch.sigmoid(model(Xe)).cpu().numpy().astype(np.float32)
        
        np.savez(OUT_DIR / f"meta_confl_v2_preds_mlp_{test_date}.npz", prob=prob, date=str(test_date))
        print(f"[mlp] fold {fold_i+1}/{len(folds)}  test={test_date}  "
              f"prob_mean={prob.mean():.4f}  pos_w={pos_w:.1f}  t={time.time()-t0:.1f}s", flush=True)

def main():
    t0 = time.time()
    print("=" * 78, flush=True)
    print("META-CONFLUENCE V2 — REAL PROFITABILITY LABELS", flush=True)
    print("=" * 78, flush=True)
    
    df, base_feat, label_desc = load_data()
    df, confl_feat = build_confluence_features(df)
    feat_cols = base_feat + confl_feat
    print(f"[v2] TOTAL features: {len(feat_cols)} (base={len(base_feat)} + confl={len(confl_feat)})", flush=True)
    
    unique_dates = sorted(df["date"].unique().tolist())
    folds = build_folds(unique_dates, K_TRAIN_DAYS)
    print(f"[v2] {len(unique_dates)} dates -> {len(folds)} folds (K_TRAIN={K_TRAIN_DAYS})", flush=True)
    
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except:
        pass
    print(f"[v2] device={device}", flush=True)
    
    print("\n-- XGBoost-GPU training --", flush=True)
    train_xgb(df, feat_cols, folds, device)
    
    print("\n-- PyTorch MLP training --", flush=True)
    train_mlp(df, feat_cols, folds, device)
    
    headline = f"""META-CONFLUENCE V2 — REAL PROFITABILITY
===================================================
Label Definition: {label_desc}
Positive rate: {df[TARGET].mean()*100:.2f}%
Training: {K_TRAIN_DAYS}-day sliding, {len(folds)} folds
Features: {len(base_feat)} base + {len(confl_feat)} confluence
Models: XGBoost-GPU + PyTorch MLP
Output: {OUT_DIR}
Time: {time.time()-t0:.1f}s
==================================================="""
    
    (OUT_DIR / "headline.txt").write_text(headline)
    print(f"\nDONE in {time.time()-t0:.1f}s\n{headline}", flush=True)

if __name__ == "__main__":
    main()
