"""
HC #334 step 2 — JUPITER MLP META-LEARNER on v3.2's 32-head prediction vector.

Inputs: full prediction vector at each timestep (all pred_* heads).
Output: P(profitable trade in next 30s under TP8/SL5 FIFO).

Per HC #334: small MLP, < 1M params, CPU-trainable in <2h.
Per HC #320: FIFO labels only (target_fifo_tp8sl5_net > 0 = win).
Per HC #322: lead with trading-system perf, not signal IC.

TRAIN/TEST split: OOT 5 days = Feb 23-27. Use FIRST 3 OOT days for meta-train,
LAST 2 OOT days for meta-test. This is 2-stage WF — CNN-Mamba already trained
pre-OOT, meta MLP trained on OOT-portion-A, tested on OOT-portion-B. NO future
leakage within meta-train.

Per HC #333 leakage discipline:
  - target labels are forward-looking (future MBO bid/ask outcomes)
  - inputs are CNN-Mamba prediction outputs at time t
  - meta-test split is temporally AFTER meta-train

OUT: output/v3_2_deep_sim_20260512/mlp_meta_hc334.{json,csv}
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

PREDS = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512")
OUT_JSON = OUT_DIR / "mlp_meta_hc334.json"
OUT_CSV = OUT_DIR / "mlp_meta_hc334.csv"

ES_TICK_USD = 12.50

print(f"[load] {PREDS}")
Z = dict(np.load(PREDS))
N = Z["target_log_ret_1s"].shape[0]
print(f"[load] n_samples={N}, oot_dates={list(Z['oot_dates'])}")

# Build feature matrix from ALL pred_* heads (excluding pred_pred_* which are model's
# self-target predictions of MFE/MAE etc; keep those too actually).
PRED_KEYS = sorted([k for k in Z.keys() if k.startswith("pred_")])
print(f"[features] using {len(PRED_KEYS)} pred heads")
X = np.stack([Z[k].astype(np.float32) for k in PRED_KEYS], axis=1)  # (N, K)
print(f"[features] X.shape={X.shape}")

# Replace NaN/Inf with 0
X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)

# Targets:
#   y_long_win  = (target_fifo_tp8sl5_net > 0)
#   y_short_win = (-target_fifo_tp8sl5_net > 0)  # sign-flip per HC #320 caveat
TGT8 = Z["target_fifo_tp8sl5_net"].astype(np.float32)
TGT_TIME = Z["target_pred_time_to_mfe_secs"].astype(np.float32)
TGT_MFE = Z["target_pred_mfe_30s_ticks"].astype(np.float32)
TGT_MAE = Z["target_pred_mae_30s_ticks"].astype(np.float32)

# We'll train two MLPs: one for "long is profitable", one for "short is profitable"
y_long = (TGT8 > 0).astype(np.float32)
y_short = (TGT8 < 0).astype(np.float32)
print(f"[targets] long_win_rate={float(y_long.mean()):.4f}, short_win_rate={float(y_short.mean()):.4f}")

# Time split: 5 days, ~48k/day. First 3 days = meta-train (~144k), last 2 = meta-test (~97k).
# Use first 60% of indices as train, last 40% as test.
SPLIT = int(N * 0.6)
print(f"[split] train_idx [0:{SPLIT}), test_idx [{SPLIT}:{N})")

X_tr, X_te = X[:SPLIT], X[SPLIT:]
yL_tr, yL_te = y_long[:SPLIT], y_long[SPLIT:]
yS_tr, yS_te = y_short[:SPLIT], y_short[SPLIT:]

# Standardize on TRAIN ONLY per HC #333 item 3
mu = X_tr.mean(axis=0, keepdims=True)
sd = X_tr.std(axis=0, keepdims=True) + 1e-6
X_tr_n = (X_tr - mu) / sd
X_te_n = (X_te - mu) / sd


# --- Tiny MLP in NumPy (CPU, no torch needed) ----------------------------------
def init_mlp(d_in: int, d_h: int = 64, seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {
        "W1": rng.normal(0, math.sqrt(2 / d_in), size=(d_in, d_h)).astype(np.float32),
        "b1": np.zeros(d_h, dtype=np.float32),
        "W2": rng.normal(0, math.sqrt(2 / d_h), size=(d_h, d_h)).astype(np.float32),
        "b2": np.zeros(d_h, dtype=np.float32),
        "W3": rng.normal(0, math.sqrt(2 / d_h), size=(d_h, 1)).astype(np.float32),
        "b3": np.zeros(1, dtype=np.float32),
    }


def relu(x):
    return np.maximum(0.0, x)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


def fwd(p, x):
    h1 = relu(x @ p["W1"] + p["b1"])
    h2 = relu(h1 @ p["W2"] + p["b2"])
    z = h2 @ p["W3"] + p["b3"]
    return sigmoid(z).squeeze(-1), (x, h1, h2)


def train_mlp(X_tr, y_tr, X_val, y_val, d_h=64, epochs=30, batch=4096, lr=1e-3, seed=0):
    d_in = X_tr.shape[1]
    p = init_mlp(d_in, d_h, seed)
    n = X_tr.shape[0]
    rng = np.random.default_rng(seed + 1)
    best_auc = -1
    best_p = {k: v.copy() for k, v in p.items()}
    for ep in range(epochs):
        idx = rng.permutation(n)
        for i in range(0, n, batch):
            b = idx[i:i + batch]
            xb, yb = X_tr[b], y_tr[b]
            yhat, (x, h1, h2) = fwd(p, xb)
            # BCE gradient
            g = (yhat - yb) / xb.shape[0]
            g = g[:, None]
            gW3 = h2.T @ g
            gb3 = g.sum(axis=0)
            gh2 = g @ p["W3"].T
            gh2 = gh2 * (h2 > 0)
            gW2 = h1.T @ gh2
            gb2 = gh2.sum(axis=0)
            gh1 = gh2 @ p["W2"].T
            gh1 = gh1 * (h1 > 0)
            gW1 = x.T @ gh1
            gb1 = gh1.sum(axis=0)
            for k, g in [("W1", gW1), ("b1", gb1), ("W2", gW2), ("b2", gb2),
                          ("W3", gW3), ("b3", gb3)]:
                p[k] -= lr * g
        # Validation AUC + loss
        yv, _ = fwd(p, X_val)
        auc = roc_auc(y_val, yv)
        loss = float(-(y_val * np.log(np.clip(yv, 1e-7, 1)) +
                       (1 - y_val) * np.log(np.clip(1 - yv, 1e-7, 1))).mean())
        if auc > best_auc:
            best_auc = auc
            best_p = {k: v.copy() for k, v in p.items()}
        if ep % 5 == 0 or ep == epochs - 1:
            print(f"  [ep{ep:02d}] val_loss={loss:.4f} val_auc={auc:.4f} (best={best_auc:.4f})")
    return best_p, best_auc


def roc_auc(y_true, y_score):
    # Mann-Whitney U style AUC
    pos = y_score[y_true == 1]
    neg = y_score[y_true == 0]
    if len(pos) == 0 or len(neg) == 0:
        return 0.5
    # Sample if too large
    if len(pos) > 5000:
        pos = np.random.default_rng(0).choice(pos, 5000, replace=False)
    if len(neg) > 5000:
        neg = np.random.default_rng(0).choice(neg, 5000, replace=False)
    return float((pos[:, None] > neg[None, :]).mean())


# --- Train both MLPs -----------------------------------------------------------
print("\n[train] LONG MLP")
pL, aucL = train_mlp(X_tr_n, yL_tr, X_te_n, yL_te, d_h=64, epochs=20)
print(f"[done] LONG val AUC = {aucL:.4f}")

print("\n[train] SHORT MLP")
pS, aucS = train_mlp(X_tr_n, yS_tr, X_te_n, yS_te, d_h=64, epochs=20)
print(f"[done] SHORT val AUC = {aucS:.4f}")

# --- Inference on test split ---------------------------------------------------
pL_pred, _ = fwd(pL, X_te_n)
pS_pred, _ = fwd(pS, X_te_n)
TGT8_te = TGT8[SPLIT:]
TGT_MFE_te = TGT_MFE[SPLIT:]
TGT_MAE_te = TGT_MAE[SPLIT:]
TGT_TIME_te = TGT_TIME[SPLIT:]

# Trading rule:
#   At each step, take the higher of P(long_win), P(short_win) above threshold.
#   If neither above threshold, no trade.
def evaluate_meta(thresh: float, min_gap: int = 11) -> dict:
    long_signal = pL_pred > thresh
    short_signal = pS_pred > thresh
    # Decide direction
    direction = np.zeros(len(pL_pred), dtype=np.int8)
    take = (long_signal | short_signal)
    direction[long_signal & ~short_signal] = 1
    direction[short_signal & ~long_signal] = -1
    # Ties: pick the higher one
    both = long_signal & short_signal
    direction[both] = np.where(pL_pred[both] >= pS_pred[both], 1, -1)
    sel = np.where(take)[0]
    # Min gap
    taken = []
    last = -10**9
    for i in sel:
        if i - last < min_gap:
            continue
        taken.append(int(i))
        last = int(i)
    sel = np.asarray(taken, dtype=np.int64)
    if len(sel) < 20:
        return {"n_trades": int(len(sel)), "ok": False}
    long_pnl = TGT8_te[sel]
    dirs = direction[sel].astype(np.float32)
    pnl = np.where(dirs >= 0, long_pnl, -long_pnl).astype(np.float32)
    n = len(pnl)
    mean = float(pnl.mean())
    std = float(pnl.std(ddof=1))
    sharpe = mean / std * math.sqrt(252) if std > 1e-9 else float("nan")
    neg = pnl[pnl < 0]
    dn = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
    sortino = mean / dn * math.sqrt(252) if dn > 1e-9 else float("nan")
    gw = float(pnl[pnl > 0].sum())
    gl = -float(pnl[pnl < 0].sum())
    pf = gw / gl if gl > 1e-9 else float("inf")
    wr = float((pnl > 0).mean() * 100.0)
    return {
        "thresh": thresh,
        "n_trades": n,
        "n_long": int((dirs >= 0).sum()),
        "n_short": int((dirs < 0).sum()),
        "wr_pct": wr,
        "mean_ticks": mean,
        "sharpe": float(sharpe) if math.isfinite(sharpe) else None,
        "sortino": float(sortino) if math.isfinite(sortino) else None,
        "pf": float(pf) if math.isfinite(pf) else None,
        "total_ticks": float(pnl.sum()),
        "total_usd": float(pnl.sum() * ES_TICK_USD),
        "hold_secs_mean": float(TGT_TIME_te[sel].mean()),
        "mfe_mean": float(TGT_MFE_te[sel].mean()),
        "mae_mean": float(TGT_MAE_te[sel].mean()),
    }


print("\n[trade] sweeping threshold ...")
rows = []
for t in [0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90, 0.95]:
    r = evaluate_meta(t)
    rows.append(r)
    if r.get("ok", True) and r.get("n_trades", 0) > 20:
        print(f"  thresh={t:.2f} n={r['n_trades']:5d} WR={r['wr_pct']:5.1f}% "
              f"Sharpe={r.get('sharpe', 0):.2f} mean={r['mean_ticks']:+.2f}t "
              f"PF={r.get('pf', 0):.2f}")
    else:
        print(f"  thresh={t:.2f} n={r.get('n_trades', 0)} insufficient")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps({
    "n_features": int(X.shape[1]),
    "feature_keys": PRED_KEYS,
    "n_train": int(SPLIT),
    "n_test": int(N - SPLIT),
    "long_mlp_auc": float(aucL),
    "short_mlp_auc": float(aucS),
    "long_win_rate_pop": float(y_long.mean()),
    "short_win_rate_pop": float(y_short.mean()),
    "threshold_sweep": rows,
    "leakage_audit": {
        "fold_idx": int(Z["fold_idx"]),
        "oot_dates": list(map(str, Z["oot_dates"])),
        "split_method": "temporal 60/40 within OOT only",
        "feature_stats_train_only": True,
        "pnl_source": "target_fifo_tp8sl5_net (MBO bid/ask FIFO labels)",
        "midpoint_used": False,
    },
}, indent=2))

if rows:
    keys = list(rows[0].keys())
    with OUT_CSV.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

print(f"\n[wrote] {OUT_JSON}")
print(f"[wrote] {OUT_CSV}")
