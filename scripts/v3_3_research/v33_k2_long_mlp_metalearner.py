#!/usr/bin/env python3
"""
K=2 LONG MLP META-LEARNER  (Work Item 2, HC #372)

Goal: learn a go/no-go classifier + regression head on K=2 LONG entry events.
The K=2 LONG entry rule (mask):
    pred_log_ret_60s  in TOP 20%  of valid universe
    pred_log_ret_5min in BOT 20%  of valid universe

For each K=2 event we build a feature vector and a label:
  Features:
    - All 32 v3.3 head predictions at signal time
    - Pre-event 30s / 60s realized vol (tick std) + drift (ticks)
    - Pre-event 30s / 60s mean spread (ticks)
    - Time-of-day (sin/cos of minute-of-day)
    - Day-of-week one-hot (5 dims)
  Label (binary classification): tp4sl3_long_net_ticks > 0   (1 = trade)
  Label (regression):            tp4sl3_long_net_ticks       (NaN if not filled => skipped)

Cross-validation: rolling 3-train / 1-val / 1-test across the 5 OOT days
                 (5 rotations covering each day as test once).

Outputs:
  /home/jupiter/Lvl3Quant/output/v3_3_full_execution_analysis_20260514/k2_mlp_metalearner/
    - k2_long_events.csv
    - mlp_features.npz
    - mlp_results.json
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

LVL3 = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = LVL3 / "output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz"
FIFO_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
BOOK_DIR = LVL3 / "data/processed/mbo_book_features"
OUT_DIR = LVL3 / "output/v3_3_full_execution_analysis_20260514/k2_mlp_metalearner"
OUT_DIR.mkdir(parents=True, exist_ok=True)

OOT_DATES = ["20260223", "20260224", "20260225", "20260226", "20260227"]
BAND_FRAC = 0.20
PRE_WINDOWS_S = [30, 60]
RTH_OPEN_MIN = 9 * 60 + 30
RTH_CLOSE_MIN = 16 * 60

COL_BID1 = 0
COL_ASK1 = 5
COL_SPREAD = 24


def L(msg: str, buf=None):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    if buf is not None:
        buf.append(line)


def ns_to_et(ts_ns):
    return pd.to_datetime(ts_ns, utc=True).tz_convert("America/New_York")


def ns_to_mod(ts_ns):
    et = ns_to_et(ts_ns)
    return np.asarray(et.hour) * 60 + np.asarray(et.minute)


def load_book_rth(date: str):
    f = BOOK_DIR / f"{date}_book_features.npz"
    if not f.exists():
        return None
    d = np.load(f)
    feat = d["features"]; ts = d["timestamps"]
    bid = feat[:, COL_BID1].astype(np.float64)
    ask = feat[:, COL_ASK1].astype(np.float64)
    spr = feat[:, COL_SPREAD].astype(np.float64)
    sane = (np.abs(bid) < 10_000) & (np.abs(ask) < 10_000) & (bid != 0) & (ask != 0)
    ts = ts[sane]; bid = bid[sane]; ask = ask[sane]; spr = spr[sane]
    if len(ts) == 0:
        return None
    mid = 0.5 * (bid + ask)
    mod = ns_to_mod(ts)
    rth = (mod >= RTH_OPEN_MIN) & (mod < RTH_CLOSE_MIN)
    return {"ts": ts[rth], "mid": mid[rth], "spread": spr[rth]}


def load_fifo(date: str):
    f = FIFO_DIR / f"{date}_fifo_labels.npz"
    if not f.exists():
        return None
    d = np.load(f)
    return {k: d[k][:] for k in d.files}


# ============================================================================
# MLP — pure NumPy, CPU, with Adam optimizer
# ============================================================================
class MLP:
    def __init__(self, d_in, d_h=64, d_out=1, seed=0, l2=1e-5):
        rng = np.random.default_rng(seed)
        self.W1 = rng.normal(0, math.sqrt(2 / d_in), (d_in, d_h)).astype(np.float32)
        self.b1 = np.zeros(d_h, dtype=np.float32)
        self.W2 = rng.normal(0, math.sqrt(2 / d_h), (d_h, d_h)).astype(np.float32)
        self.b2 = np.zeros(d_h, dtype=np.float32)
        self.W3 = rng.normal(0, math.sqrt(2 / d_h), (d_h, d_out)).astype(np.float32)
        self.b3 = np.zeros(d_out, dtype=np.float32)
        self.l2 = l2

    def forward(self, X):
        self.X = X
        self.z1 = X @ self.W1 + self.b1
        self.h1 = np.maximum(0, self.z1)
        self.z2 = self.h1 @ self.W2 + self.b2
        self.h2 = np.maximum(0, self.z2)
        self.z3 = self.h2 @ self.W3 + self.b3
        return self.z3

    def predict_proba(self, X):
        z = self.forward(X)
        return 1.0 / (1.0 + np.exp(-np.clip(z, -30, 30)))

    def predict_reg(self, X):
        return self.forward(X)

    def backward_bce(self, p, y):
        # p shape (N,1), y shape (N,1)
        n = p.shape[0]
        dZ3 = (p - y) / n
        dW3 = self.h2.T @ dZ3 + self.l2 * self.W3
        db3 = dZ3.sum(0)
        dh2 = dZ3 @ self.W3.T
        dz2 = dh2 * (self.z2 > 0)
        dW2 = self.h1.T @ dz2 + self.l2 * self.W2
        db2 = dz2.sum(0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (self.z1 > 0)
        dW1 = self.X.T @ dz1 + self.l2 * self.W1
        db1 = dz1.sum(0)
        return dW1, db1, dW2, db2, dW3, db3

    def backward_mse(self, yhat, y):
        n = yhat.shape[0]
        dZ3 = 2 * (yhat - y) / n
        dW3 = self.h2.T @ dZ3 + self.l2 * self.W3
        db3 = dZ3.sum(0)
        dh2 = dZ3 @ self.W3.T
        dz2 = dh2 * (self.z2 > 0)
        dW2 = self.h1.T @ dz2 + self.l2 * self.W2
        db2 = dz2.sum(0)
        dh1 = dz2 @ self.W2.T
        dz1 = dh1 * (self.z1 > 0)
        dW1 = self.X.T @ dz1 + self.l2 * self.W1
        db1 = dz1.sum(0)
        return dW1, db1, dW2, db2, dW3, db3


def adam_init(params):
    return [{"m": np.zeros_like(p), "v": np.zeros_like(p), "t": 0} for p in params]


def adam_step(params, grads, state, lr=1e-3, b1=0.9, b2=0.999, eps=1e-8):
    for p, g, s in zip(params, grads, state):
        s["t"] += 1
        s["m"] = b1 * s["m"] + (1 - b1) * g
        s["v"] = b2 * s["v"] + (1 - b2) * (g * g)
        m_hat = s["m"] / (1 - b1 ** s["t"])
        v_hat = s["v"] / (1 - b2 ** s["t"])
        p -= lr * m_hat / (np.sqrt(v_hat) + eps)


def train_clf(X_tr, y_tr, X_va, y_va, d_h=64, lr=1e-3, epochs=150,
              batch=256, patience=20, seed=0, log_buf=None):
    n, d = X_tr.shape
    mlp = MLP(d, d_h, 1, seed=seed)
    state = adam_init([mlp.W1, mlp.b1, mlp.W2, mlp.b2, mlp.W3, mlp.b3])
    best = None
    best_val = -1e9
    bad = 0
    rng = np.random.default_rng(seed)
    y_tr_c = y_tr.reshape(-1, 1).astype(np.float32)
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            p = mlp.predict_proba(X_tr[b])
            grads = mlp.backward_bce(p, y_tr_c[b])
            adam_step(
                [mlp.W1, mlp.b1, mlp.W2, mlp.b2, mlp.W3, mlp.b3],
                grads, state, lr=lr,
            )
        # validate
        pv = mlp.predict_proba(X_va).ravel()
        # AUC-ish: just use balanced accuracy at 0.5 + log-loss
        from math import log as _log  # noqa
        bce = -float(np.mean(y_va * np.log(np.clip(pv, 1e-7, 1)) +
                             (1 - y_va) * np.log(np.clip(1 - pv, 1e-7, 1))))
        acc = float(np.mean((pv > 0.5) == (y_va > 0.5)))
        score = -bce  # higher better
        if score > best_val + 1e-5:
            best_val = score; bad = 0
            best = {k: getattr(mlp, k).copy() for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
        else:
            bad += 1
            if bad >= patience:
                break
    # restore best
    for k, v in best.items():
        setattr(mlp, k, v)
    return mlp, {"val_acc": acc, "val_bce": bce, "epochs_run": ep + 1}


def train_reg(X_tr, y_tr, X_va, y_va, d_h=64, lr=1e-3, epochs=150,
              batch=256, patience=20, seed=0):
    n, d = X_tr.shape
    mlp = MLP(d, d_h, 1, seed=seed)
    state = adam_init([mlp.W1, mlp.b1, mlp.W2, mlp.b2, mlp.W3, mlp.b3])
    best = None
    best_val = 1e18
    bad = 0
    rng = np.random.default_rng(seed)
    y_tr_c = y_tr.reshape(-1, 1).astype(np.float32)
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            yhat = mlp.predict_reg(X_tr[b])
            grads = mlp.backward_mse(yhat, y_tr_c[b])
            adam_step(
                [mlp.W1, mlp.b1, mlp.W2, mlp.b2, mlp.W3, mlp.b3],
                grads, state, lr=lr,
            )
        yv = mlp.predict_reg(X_va).ravel()
        mse = float(np.mean((yv - y_va) ** 2))
        if mse < best_val - 1e-6:
            best_val = mse; bad = 0
            best = {k: getattr(mlp, k).copy() for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
        else:
            bad += 1
            if bad >= patience:
                break
    for k, v in best.items():
        setattr(mlp, k, v)
    return mlp, {"val_mse": best_val, "epochs_run": ep + 1}


# ============================================================================
# Feature importance via permutation
# ============================================================================
def perm_importance_clf(mlp, X, y, feat_names, n_repeats=3, seed=0):
    rng = np.random.default_rng(seed)
    base_p = mlp.predict_proba(X).ravel()
    base_acc = float(np.mean((base_p > 0.5) == (y > 0.5)))
    imps = {}
    for j, name in enumerate(feat_names):
        deltas = []
        for r in range(n_repeats):
            Xp = X.copy()
            rng.shuffle(Xp[:, j])
            p = mlp.predict_proba(Xp).ravel()
            acc = float(np.mean((p > 0.5) == (y > 0.5)))
            deltas.append(base_acc - acc)
        imps[name] = float(np.mean(deltas))
    return base_acc, imps


# ============================================================================
# MAIN
# ============================================================================
def main():
    log_buf = []
    L(f"PID {os.getpid()} | v33_k2_long_mlp_metalearner", log_buf)
    L(f"OOT: {OOT_DATES}", log_buf)
    t0 = time.time()

    # Load predictions
    L(f"Loading predictions {PRED_NPZ}", log_buf)
    D = np.load(PRED_NPZ, allow_pickle=True)
    p60 = D["pred_log_ret_60s"][:].astype(np.float64)
    p5m = D["pred_log_ret_5min"][:].astype(np.float64)
    pred_keys = sorted([k for k in D.keys() if k.startswith("pred_")])
    L(f"  preds n={len(p60)}, head_count={len(pred_keys)}", log_buf)
    # All head features
    HEADS = np.stack([D[k][:].astype(np.float32) for k in pred_keys], axis=1)
    HEADS = np.where(np.isfinite(HEADS), HEADS, 0.0).astype(np.float32)
    L(f"  HEADS shape {HEADS.shape}", log_buf)

    # Load FIFO & book per day
    L("Loading FIFO + book...", log_buf)
    fifo, books = {}, {}
    for date in OOT_DATES:
        fifo[date] = load_fifo(date)
        books[date] = load_book_rth(date)
        if fifo[date] is None:
            L(f"  WARN no FIFO {date}", log_buf)
        if books[date] is None:
            L(f"  WARN no book {date}", log_buf)

    # Build ts/date alignment
    ts_chunks, date_chunks = [], []
    offsets = {}
    cursor = 0
    for date in OOT_DATES:
        f = fifo.get(date)
        if f is None:
            continue
        n = len(f["ts_ns"])
        offsets[date] = cursor
        ts_chunks.append(f["ts_ns"])
        date_chunks.append(np.array([date] * n))
        cursor += n
    ts_all = np.concatenate(ts_chunks)
    date_all = np.concatenate(date_chunks)
    n_align = min(len(p60), len(ts_all))
    p60 = p60[:n_align]; p5m = p5m[:n_align]
    HEADS = HEADS[:n_align]
    ts_all = ts_all[:n_align]; date_all = date_all[:n_align]
    L(f"  aligned n={n_align}", log_buf)

    # Build K=2 LONG signal mask (on all valid universe, per user's 247-event count)
    valid = np.isfinite(p60) & np.isfinite(p5m)
    p60v = p60[valid]; p5mv = p5m[valid]
    thr60_top = float(np.quantile(p60v, 1 - BAND_FRAC))
    thr5m_bot = float(np.quantile(p5mv, BAND_FRAC))
    sig_mask = valid & (p60 >= thr60_top) & (p5m <= thr5m_bot)
    sig_idx = np.where(sig_mask)[0]
    L(f"  K=2 thresholds: p60_top20={thr60_top:.6f}  p5m_bot20={thr5m_bot:.6f}", log_buf)
    L(f"  K=2 LONG events: n={len(sig_idx)}", log_buf)

    # Build per-event feature/label table
    rows = []
    pre_feats = []  # list of dicts
    for gi in sig_idx:
        date = str(date_all[gi])
        f = fifo.get(date)
        if f is None:
            continue
        within = gi - offsets[date]
        filled = bool(f["tp4sl3_long_filled"][within])
        net_t = float(f["tp4sl3_long_net_ticks"][within]) if filled else np.nan
        ts = int(ts_all[gi])
        rows.append({
            "global_idx": int(gi),
            "date": date,
            "ts_ns": ts,
            "p60": float(p60[gi]),
            "p5m": float(p5m[gi]),
            "filled": filled,
            "net_ticks": net_t,
        })
        # Pre-signal microstructure
        b = books.get(date)
        pf = {"vol_30s_t": np.nan, "vol_60s_t": np.nan,
              "drift_30s_t": np.nan, "drift_60s_t": np.nan,
              "spread_30s_t": np.nan, "spread_60s_t": np.nan}
        if b is not None:
            j = np.searchsorted(b["ts"], ts, side="right") - 1
            if j >= 1:
                for w, key_v, key_d, key_s in [
                    (30, "vol_30s_t", "drift_30s_t", "spread_30s_t"),
                    (60, "vol_60s_t", "drift_60s_t", "spread_60s_t"),
                ]:
                    s = np.searchsorted(b["ts"], ts - w * 1_000_000_000, side="left")
                    if s < j:
                        wm = b["mid"][s:j + 1]
                        ws = b["spread"][s:j + 1]
                        wts = b["ts"][s:j + 1]
                        secs = (wts // 1_000_000_000).astype(np.int64)
                        tmp = (pd.DataFrame({"x": secs, "m": wm})
                               .groupby("x")["m"].last().values)
                        if tmp.size > 1:
                            d = np.diff(tmp)
                            pf[key_v] = float(np.std(d)) if d.size > 1 else 0.0
                        else:
                            pf[key_v] = 0.0
                        pf[key_d] = float(wm[-1] - wm[0])
                        pf[key_s] = float(np.nanmean(ws))
        pre_feats.append(pf)

    ev = pd.DataFrame(rows)
    pre_df = pd.DataFrame(pre_feats)
    ev = pd.concat([ev.reset_index(drop=True), pre_df.reset_index(drop=True)], axis=1)
    L(f"  Events DF: n={len(ev)}", log_buf)
    L(f"  Per-day events: {ev.groupby('date').size().to_dict()}", log_buf)
    L(f"  Filled events:  {int(ev.filled.sum())}", log_buf)
    L(f"  Per-day fills:  {ev[ev.filled].groupby('date').size().to_dict()}", log_buf)
    if int(ev.filled.sum()) > 0:
        for date, sub in ev[ev.filled].groupby("date"):
            n = len(sub); mt = float(sub.net_ticks.mean())
            wr = float((sub.net_ticks > 0).mean())
            L(f"    {date}: nfill={n}  mean_t={mt:+.3f}  WR={wr:.3f}", log_buf)

    # Time-of-day + day-of-week features
    mod = ns_to_mod(ev.ts_ns.values)
    sin_tod = np.sin(2 * np.pi * mod / 1440.0)
    cos_tod = np.cos(2 * np.pi * mod / 1440.0)
    et = ns_to_et(ev.ts_ns.values)
    dow = np.asarray(et.dayofweek)  # 0=Mon ... 4=Fri (OOT all weekdays)
    dow_oh = np.zeros((len(ev), 5), dtype=np.float32)
    for i, d in enumerate(dow):
        if 0 <= d < 5:
            dow_oh[i, d] = 1.0
    ev["min_of_day"] = mod
    ev["dow"] = dow

    # Build feature matrix
    head_X = HEADS[ev.global_idx.values]
    pre_cols = ["vol_30s_t", "vol_60s_t", "drift_30s_t", "drift_60s_t",
                "spread_30s_t", "spread_60s_t"]
    pre_X = ev[pre_cols].fillna(0.0).values.astype(np.float32)
    tod_X = np.stack([sin_tod, cos_tod], axis=1).astype(np.float32)
    X_all = np.concatenate([head_X, pre_X, tod_X, dow_oh], axis=1)
    feat_names = list(pred_keys) + pre_cols + ["sin_tod", "cos_tod"] + \
                 [f"dow_{i}" for i in range(5)]
    L(f"  Feature matrix: {X_all.shape}, n_features={len(feat_names)}", log_buf)

    # Save event CSV
    ev_csv = OUT_DIR / "k2_long_events.csv"
    ev.to_csv(ev_csv, index=False)
    L(f"  Saved {ev_csv}", log_buf)

    # ===========================================================
    # Train/Val/Test by day (5 rotations)
    # ===========================================================
    if int(ev.filled.sum()) < 10:
        L("FATAL: insufficient filled events for MLP training", log_buf)
        # Still save what we have
        np.savez(OUT_DIR / "mlp_features.npz",
                 X=X_all, feat_names=np.array(feat_names),
                 dates=ev.date.values, filled=ev.filled.values,
                 net_ticks=ev.net_ticks.values)
        with open(OUT_DIR / "mlp_results.json", "w") as f:
            json.dump({"status": "skipped_insufficient_fills",
                       "n_events": len(ev),
                       "n_filled": int(ev.filled.sum()),
                       "per_day_fills": ev[ev.filled].groupby("date").size().to_dict(),
                       }, f, indent=2)
        (OUT_DIR / "mlp_metalearner.log").write_text("\n".join(log_buf) + "\n")
        return 0

    # Restrict to FILLED events for label
    keep = ev.filled.values
    Xf = X_all[keep]
    df = ev[keep].reset_index(drop=True)
    yc = (df.net_ticks.values > 0).astype(np.float32)
    yr = df.net_ticks.values.astype(np.float32)
    L(f"  Training on n={len(df)} filled events, win_rate={yc.mean():.3f}", log_buf)

    # 5-fold rolling: each rotation -> [train: 3 days] [val: 1 day] [test: 1 day]
    # Rotation k:  test = OOT[k], val = OOT[(k-1) mod 5], train = the other 3
    fold_results = []
    all_test_proba = np.full(len(df), np.nan)
    all_test_pred_reg = np.full(len(df), np.nan)
    all_test_label = np.full(len(df), np.nan)
    all_test_net = np.full(len(df), np.nan)
    feat_imps_acc = {fn: [] for fn in feat_names}

    for k in range(len(OOT_DATES)):
        test_date = OOT_DATES[k]
        val_date = OOT_DATES[(k - 1) % len(OOT_DATES)]
        train_dates = [d for d in OOT_DATES if d != test_date and d != val_date]
        tr_mask = df.date.isin(train_dates).values
        va_mask = (df.date.values == val_date)
        te_mask = (df.date.values == test_date)
        if tr_mask.sum() < 5 or te_mask.sum() < 1:
            L(f"  Fold {k} test={test_date}: insufficient samples — skip "
              f"(tr={tr_mask.sum()} va={va_mask.sum()} te={te_mask.sum()})", log_buf)
            continue
        # Standardize on train
        mu = Xf[tr_mask].mean(0, keepdims=True)
        sd = Xf[tr_mask].std(0, keepdims=True) + 1e-6
        Xtr = (Xf[tr_mask] - mu) / sd
        Xva = (Xf[va_mask] - mu) / sd if va_mask.sum() > 0 else Xtr[:1]
        Xte = (Xf[te_mask] - mu) / sd

        ytr_c = yc[tr_mask]; yva_c = yc[va_mask] if va_mask.sum() > 0 else ytr_c[:1]
        yte_c = yc[te_mask]
        ytr_r = yr[tr_mask]; yva_r = yr[va_mask] if va_mask.sum() > 0 else ytr_r[:1]
        yte_r = yr[te_mask]

        L(f"  Fold {k}: train={train_dates} (n={tr_mask.sum()})  "
          f"val={val_date} (n={va_mask.sum()})  test={test_date} (n={te_mask.sum()})",
          log_buf)

        # Classifier
        clf, clf_info = train_clf(Xtr, ytr_c, Xva, yva_c, d_h=48, lr=1e-3,
                                  epochs=200, batch=64, patience=25, seed=42 + k)
        pte = clf.predict_proba(Xte).ravel()
        acc = float(np.mean((pte > 0.5) == (yte_c > 0.5)))
        # Recall vs base-rate
        baserate = float(yte_c.mean()) if len(yte_c) else 0.0
        base_acc = max(baserate, 1 - baserate)

        # Regression
        reg, reg_info = train_reg(Xtr, ytr_r, Xva, yva_r, d_h=48, lr=1e-3,
                                  epochs=200, batch=64, patience=25, seed=99 + k)
        rte = reg.predict_reg(Xte).ravel()
        mse_te = float(np.mean((rte - yte_r) ** 2))
        mae_te = float(np.mean(np.abs(rte - yte_r)))

        # Trade decision: take if proba > 0.5
        take = pte > 0.5
        n_take = int(take.sum())
        if n_take > 0:
            pnl_taken = float(yte_r[take].sum())
            wr_taken = float((yte_r[take] > 0).mean())
            tpf_taken = float(yte_r[take].mean())
        else:
            pnl_taken = wr_taken = tpf_taken = 0.0
        # Baseline (take all)
        pnl_all = float(yte_r.sum())
        wr_all = float((yte_r > 0).mean())
        tpf_all = float(yte_r.mean())

        # Permutation importance on test (small)
        base_acc_p, imps = perm_importance_clf(clf, Xte, yte_c, feat_names,
                                               n_repeats=3, seed=k)
        for fn, v in imps.items():
            feat_imps_acc[fn].append(v)

        fold_results.append({
            "fold": k,
            "test_date": test_date,
            "val_date": val_date,
            "train_dates": train_dates,
            "n_train": int(tr_mask.sum()),
            "n_val": int(va_mask.sum()),
            "n_test": int(te_mask.sum()),
            "clf": {
                "test_acc": acc,
                "baserate_acc": base_acc,
                "uplift_vs_baserate": acc - base_acc,
                "val_acc": clf_info["val_acc"],
                "val_bce": clf_info["val_bce"],
            },
            "reg": {"test_mse": mse_te, "test_mae": mae_te,
                    "val_mse": reg_info["val_mse"]},
            "trade_sim": {
                "n_take": n_take,
                "pnl_taken_ticks": pnl_taken,
                "tpf_taken": tpf_taken,
                "wr_taken": wr_taken,
                "pnl_all_ticks": pnl_all,
                "tpf_all": tpf_all,
                "wr_all": wr_all,
                "trade_pct": n_take / len(yte_c) if len(yte_c) else 0.0,
            },
        })
        all_test_proba[te_mask] = pte
        all_test_pred_reg[te_mask] = rte
        all_test_label[te_mask] = yte_c
        all_test_net[te_mask] = yte_r

    # Aggregate OOS
    oos_mask = ~np.isnan(all_test_proba)
    n_oos = int(oos_mask.sum())
    L(f"OOS aggregate: n={n_oos}", log_buf)
    if n_oos > 0:
        proba = all_test_proba[oos_mask]
        label = all_test_label[oos_mask]
        net = all_test_net[oos_mask]
        take = proba > 0.5
        oos_acc = float(np.mean((proba > 0.5) == (label > 0.5)))
        oos_take_n = int(take.sum())
        oos_take_pnl = float(net[take].sum()) if oos_take_n else 0.0
        oos_take_wr = float((net[take] > 0).mean()) if oos_take_n else 0.0
        oos_take_tpf = float(net[take].mean()) if oos_take_n else 0.0
        oos_all_pnl = float(net.sum())
        oos_all_wr = float((net > 0).mean())
        oos_all_tpf = float(net.mean())
        # Per-day decomposition
        oos_df = df[oos_mask].copy()
        oos_df["proba"] = proba; oos_df["take"] = take.astype(int)
        oos_df["net"] = net
        per_day = (oos_df.groupby("date")
                   .apply(lambda g: pd.Series({
                       "n": len(g),
                       "n_take": int(g.take.sum()),
                       "pnl_take_t": float(g[g.take == 1].net.sum()),
                       "pnl_all_t": float(g.net.sum()),
                       "wr_take": float((g[g.take == 1].net > 0).mean()) if g.take.sum() else 0.0,
                       "wr_all": float((g.net > 0).mean()),
                   }))
                   .reset_index()).to_dict(orient="records")
        # Critical: would MLP have blocked the 0226 losses?
        d226 = oos_df[oos_df.date == "20260226"]
        if len(d226):
            losses_226 = d226[d226.net <= 0]
            blocked = losses_226[losses_226.take == 0]
            block_pct = float(len(blocked) / len(losses_226)) if len(losses_226) else 0.0
        else:
            block_pct = None
        oos_summary = {
            "n": n_oos, "acc": oos_acc, "n_take": oos_take_n,
            "pnl_take_ticks": oos_take_pnl, "tpf_take": oos_take_tpf,
            "wr_take": oos_take_wr, "pnl_all_ticks": oos_all_pnl,
            "tpf_all": oos_all_tpf, "wr_all": oos_all_wr,
            "trade_pct": oos_take_n / n_oos,
            "per_day": per_day,
            "would_have_blocked_0226_losses_pct": block_pct,
        }
        L(f"  OOS acc={oos_acc:.3f}  take={oos_take_n}/{n_oos} "
          f"({oos_take_n / n_oos:.2%})", log_buf)
        L(f"  OOS taken P&L: {oos_take_pnl:+.1f}t  tpf={oos_take_tpf:+.3f}  "
          f"WR={oos_take_wr:.3f}", log_buf)
        L(f"  OOS all-take baseline P&L: {oos_all_pnl:+.1f}t  tpf={oos_all_tpf:+.3f} "
          f" WR={oos_all_wr:.3f}", log_buf)
        if block_pct is not None:
            L(f"  Would have blocked {block_pct:.1%} of 0226 LONG losses", log_buf)
    else:
        oos_summary = {"n": 0}

    # Feature importance (mean across folds)
    imp_means = {fn: float(np.mean(vs)) if vs else 0.0
                 for fn, vs in feat_imps_acc.items()}
    top = sorted(imp_means.items(), key=lambda kv: kv[1], reverse=True)[:15]
    L("Top 15 features by permutation importance (mean dropped-acc):", log_buf)
    for fn, v in top:
        L(f"  {fn}: {v:+.4f}", log_buf)

    # Save artifacts
    np.savez(
        OUT_DIR / "mlp_features.npz",
        X=X_all, feat_names=np.array(feat_names),
        dates=ev.date.values, filled=ev.filled.values,
        net_ticks=ev.net_ticks.values,
        all_test_proba=all_test_proba, all_test_label=all_test_label,
        all_test_net=all_test_net, all_test_pred_reg=all_test_pred_reg,
    )

    results = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "rule": "K=2 LONG entry on K=2 SHORT mask",
        "n_total_events": int(len(ev)),
        "n_filled_events": int(ev.filled.sum()),
        "per_day_events": ev.groupby("date").size().to_dict(),
        "per_day_fills": ev[ev.filled].groupby("date").size().to_dict(),
        "thresholds": {"thr_p60_top20": thr60_top, "thr_p5m_bot20": thr5m_bot},
        "feature_names": feat_names,
        "fold_results": fold_results,
        "oos_summary": oos_summary,
        "top_feature_importances": [{"name": fn, "mean_delta_acc": v} for fn, v in top],
        "all_feature_importances": imp_means,
        "elapsed_sec": time.time() - t0,
        "notes": [
            "Pure NumPy MLP (2-layer, 48 hidden units) with Adam optimizer.",
            "5-fold rotation: 3 train days, 1 val day, 1 test day.",
            "Label: tp4sl3_long_net_ticks > 0 (binary). Only filled events used.",
            "Trade decision: take if classifier proba > 0.5.",
            "Includes regression head predicting net ticks.",
            "Commission already netted in tp4sl3_long_net_ticks.",
        ],
    }
    with open(OUT_DIR / "mlp_results.json", "w") as f:
        json.dump(results, f, indent=2,
                  default=lambda o: float(o) if isinstance(o, np.floating)
                  else (int(o) if isinstance(o, np.integer) else str(o)))
    L(f"Saved {OUT_DIR / 'mlp_results.json'}", log_buf)
    L(f"DONE in {time.time() - t0:.1f}s", log_buf)

    log_path = Path(os.environ.get(
        "MLP_LOG",
        str(OUT_DIR / f"mlp_metalearner_{datetime.now():%Y%m%d_%H%M%S}.log")))
    log_path.parent.mkdir(parents=True, exist_ok=True)
    log_path.write_text("\n".join(log_buf) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
