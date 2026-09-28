#!/usr/bin/env python3
"""
razer_meta_confluence_train.py — Deeper/wider confluence-aware meta-classifier.

Per HC #468 R2 menu item 1: the earlier MLP (32 heads -> 5s return) got Sharpe
-0.08; the classifier (32 heads -> profitable-trigger binary) got Sharpe -0.06.
This version ADDS 50 binary pair-agreement indicator features derived from the
top-50 robust pair winners produced by the 11:12 ET full_pair_triplet_sweep:

  feat[i] = 1 if (sign(head_a) == sign(head_b)) AND
                 (|head_a| >= q_a) AND (|head_b| >= q_b)  at conf cut top_pct%

Plus the original 32 prediction heads (continuous).

Two models trained walk-forward (15-day-train / 1-day-test, sliding):
  1. PyTorch MLP-Classifier 256-128-64 with dropout 0.2
  2. XGBoost-GPU classifier depth=6, lr=0.05, n_est=500

Target: y_profitable_trigger (binary) — same as razer_classifier.

Outputs per-date NPZs to output/razer_meta_confluence/.

Auto-detects GPU. Cross-platform (Razer Windows / Jupiter Linux).
"""
from __future__ import annotations
import json
import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd

if os.name == "nt":
    CLASS_FEATURES = Path(r"C:\Users\claude\Lvl3Quant\data\class_features.parquet")
    META_FEATURES = Path(r"C:\Users\claude\Lvl3Quant\data\meta_features.parquet")
    PAIR_MATRIX = Path(r"C:\Users\claude\Lvl3Quant\data\pair_matrix.parquet")
    OUT_DIR = Path(r"C:\Users\claude\Lvl3Quant\output\razer_meta_confluence")
else:
    CLASS_FEATURES = Path("/home/nick/Lvl3Quant/output/razer_classifier/class_features.parquet")
    META_FEATURES = Path("/home/nick/Lvl3Quant/output/razer_meta/meta_features.parquet")
    PAIR_MATRIX = Path("/home/nick/Lvl3Quant/output/stream_backtest_v2/pair_matrix.parquet")
    OUT_DIR = Path("/home/nick/Lvl3Quant/output/razer_meta_confluence")
OUT_DIR.mkdir(parents=True, exist_ok=True)

EXCLUDE_FEATURES = {"pred_k"}
EXCLUDE_COLS = {"date", "pred_k",
                "y_trigger", "y_profitable_trigger",
                "trig_1s_x_5s_pup", "trig_30s_q90_x_1s",
                "trig_30s_x_1s", "trig_mfe_x_1s",
                "target_log_ret_1s", "target_log_ret_5s",
                "target_log_ret_10s", "target_log_ret_30s"}
TARGET = "y_profitable_trigger"
K_TRAIN_DAYS = 15
TOP_N_PAIRS = 50
CONF_PCT_TO_Q = {1.0: 0.99, 5.0: 0.95, 10.0: 0.90}

# Heads where directional_signal demotes sign — copied from stream_continuation_backtest
NON_DIRECTIONAL_HEADS = {
    "pred_p_reversal_15s", "pred_p_reversal_30s", "pred_p_reversal_60s",
    "pred_pred_mae_30s_ticks", "pred_pred_mae_60s_ticks",
    "pred_pred_mfe_30s_ticks", "pred_pred_mfe_60s_ticks",
    "pred_pred_realized_vol_30s_ticks",
    "pred_pred_time_to_mfe_secs",
    "pred_k",
}


def directional_signal(name: str, vals: np.ndarray) -> np.ndarray:
    """Replicates stream_continuation_backtest.directional_signal."""
    if name in NON_DIRECTIONAL_HEADS:
        return np.zeros_like(vals)
    if name.startswith("pred_p_up_"):
        return vals - 0.5
    if "fifo" in name:
        return vals  # already signed net
    return vals


def load_data():
    print(f"[meta_confl] loading class features {CLASS_FEATURES}", flush=True)
    df = pd.read_parquet(CLASS_FEATURES)
    feat_cols = sorted([c for c in df.columns
                        if c.startswith("pred_") and c not in EXCLUDE_COLS])
    print(f"[meta_confl] base features: {len(feat_cols)}  rows: {len(df):,}  "
          f"dates: {df['date'].nunique()}", flush=True)
    print(f"[meta_confl] target {TARGET}: {df[TARGET].sum():,} pos "
          f"({df[TARGET].mean()*100:.3f}%)", flush=True)
    return df, feat_cols


def build_confluence_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, List[str]]:
    """Add binary confluence indicators from the top-N robust pair winners."""
    print(f"[meta_confl] loading pair matrix {PAIR_MATRIX}", flush=True)
    pm = pd.read_parquet(PAIR_MATRIX)
    pm_robust = pm[(pm["net_ticks_after_cost"] > 0)
                   & (pm["day_conc_pass"])
                   & (pm["n_conf_trades"] >= 100)].copy()
    pm_robust = pm_robust.sort_values("sharpe_per_trade", ascending=False).head(TOP_N_PAIRS)
    print(f"[meta_confl] using top-{len(pm_robust)} robust pairs as confluence indicators",
          flush=True)

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
        thr_a = np.quantile(np.abs(sa[m]), conf_q)
        thr_b = np.quantile(np.abs(sb[m]), conf_q)
        same = np.sign(sa) == np.sign(sb)
        signal = ((np.abs(sa) >= thr_a) & (np.abs(sb) >= thr_b)
                  & same & (np.sign(sa) != 0)).astype(np.int8)
        cname = f"confl_p{idx:02d}_{ha[:12]}_x_{hb[:12]}_top{pct:.0f}"
        df[cname] = signal
        # Signed variant (capture direction)
        df[cname + "_signed"] = (signal * np.sign(sa)).astype(np.int8)
        confl_cols.append(cname)
        confl_cols.append(cname + "_signed")
    print(f"[meta_confl] added {len(confl_cols)} confluence features "
          f"(binary + signed per pair)", flush=True)
    return df, confl_cols


def build_folds(unique_dates: List[str], k_train: int) -> List[Tuple[List[str], str]]:
    folds = []
    for i in range(k_train, len(unique_dates)):
        folds.append((unique_dates[i - k_train:i], unique_dates[i]))
    return folds


# ---------------- XGBoost-GPU ----------------
def train_xgb(df, feat_cols, folds, device):
    import xgboost as xgb
    print(f"[meta_confl/xgb] xgboost {xgb.__version__}  device={device}", flush=True)
    importances = {f: 0.0 for f in feat_cols}
    n_imp = 0
    for fold_i, (train_dates, test_date) in enumerate(folds):
        t0 = time.time()
        tr = df[df["date"].isin(train_dates)]
        te = df[df["date"] == test_date]
        X_tr = tr[feat_cols].values.astype(np.float32)
        y_tr = tr[TARGET].values.astype(np.int32)
        X_te = te[feat_cols].values.astype(np.float32)
        pos = int(y_tr.sum())
        neg = len(y_tr) - pos
        spw = max(1.0, neg / max(1, pos))

        try:
            major = int(xgb.__version__.split(".")[0])
        except Exception:
            major = 2
        if major >= 2:
            params = dict(objective="binary:logistic", tree_method="hist", device=device,
                          n_estimators=500, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=spw, eval_metric="logloss", verbosity=0)
        else:
            params = dict(objective="binary:logistic",
                          tree_method="gpu_hist" if device == "cuda" else "hist",
                          n_estimators=500, max_depth=6, learning_rate=0.05,
                          subsample=0.8, colsample_bytree=0.8,
                          scale_pos_weight=spw, eval_metric="logloss", verbosity=0)
        clf = xgb.XGBClassifier(**params)
        clf.fit(X_tr, y_tr)
        prob = clf.predict_proba(X_te)[:, 1].astype(np.float32)
        np.savez(OUT_DIR / f"meta_confl_preds_xgb_{test_date}.npz",
                 prob=prob, date=str(test_date))
        # Accumulate importances
        try:
            imps = clf.feature_importances_
            for f, im in zip(feat_cols, imps):
                importances[f] += float(im)
            n_imp += 1
        except Exception:
            pass
        print(f"[xgb] fold {fold_i+1}/{len(folds)}  test={test_date}  "
              f"n_tr={len(tr):,}  n_te={len(te):,}  "
              f"prob_mean={prob.mean():.4f}  t={time.time()-t0:.1f}s", flush=True)
    if n_imp > 0:
        avg = {k: v / n_imp for k, v in importances.items()}
        avg_sorted = dict(sorted(avg.items(), key=lambda kv: kv[1], reverse=True))
        (OUT_DIR / "xgb_feature_importances.json").write_text(json.dumps(avg_sorted, indent=2))


# ---------------- PyTorch MLP ----------------
def train_mlp(df, feat_cols, folds, device):
    import torch
    import torch.nn as nn
    print(f"[meta_confl/mlp] torch {torch.__version__}  cuda={torch.cuda.is_available()}",
          flush=True)
    use_cuda = device == "cuda" and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")

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
        tr = df[df["date"].isin(train_dates)]
        te = df[df["date"] == test_date]
        X_tr = tr[feat_cols].values.astype(np.float32)
        y_tr = tr[TARGET].values.astype(np.float32)
        X_te = te[feat_cols].values.astype(np.float32)

        # Standardize using train stats only
        mu = X_tr.mean(axis=0)
        sd = X_tr.std(axis=0) + 1e-6
        X_tr = (X_tr - mu) / sd
        X_te = (X_te - mu) / sd

        pos = float(y_tr.sum())
        neg = float(len(y_tr) - pos)
        pos_w = max(1.0, neg / max(1.0, pos))

        model = MLP(in_dim=X_tr.shape[1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-5)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=torch.tensor(pos_w, device=dev))

        Xt = torch.from_numpy(X_tr).to(dev)
        yt = torch.from_numpy(y_tr).to(dev)
        Xe = torch.from_numpy(X_te).to(dev)

        BATCH = 8192
        EPOCHS = 10
        n = Xt.shape[0]
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
            logits = model(Xe)
            prob = torch.sigmoid(logits).cpu().numpy().astype(np.float32)
        np.savez(OUT_DIR / f"meta_confl_preds_mlp_{test_date}.npz",
                 prob=prob, date=str(test_date))
        print(f"[mlp] fold {fold_i+1}/{len(folds)}  test={test_date}  "
              f"prob_mean={prob.mean():.4f}  pos_w={pos_w:.1f}  "
              f"t={time.time()-t0:.1f}s", flush=True)


def main():
    t0 = time.time()
    print("=" * 78, flush=True)
    print("RAZER META-CONFLUENCE TRAIN — 32 heads + top-50 pair confluence", flush=True)
    print("=" * 78, flush=True)

    df, base_feat = load_data()
    df, confl_feat = build_confluence_features(df)
    feat_cols = base_feat + confl_feat
    print(f"[meta_confl] TOTAL feature dim: {len(feat_cols)} "
          f"(base={len(base_feat)} + confl={len(confl_feat)})", flush=True)

    unique_dates = sorted(df["date"].unique().tolist())
    folds = build_folds(unique_dates, K_TRAIN_DAYS)
    print(f"[meta_confl] {len(unique_dates)} dates  -> {len(folds)} folds  "
          f"(K_TRAIN_DAYS={K_TRAIN_DAYS})", flush=True)

    # Detect device
    device = "cpu"
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
    except Exception:
        pass
    print(f"[meta_confl] device={device}", flush=True)

    print("\n-- XGBoost-GPU training --", flush=True)
    train_xgb(df, feat_cols, folds, device)

    print("\n-- PyTorch MLP training --", flush=True)
    train_mlp(df, feat_cols, folds, device)

    print(f"\nDONE in {time.time()-t0:.1f}s — outputs in {OUT_DIR}", flush=True)


if __name__ == "__main__":
    main()
