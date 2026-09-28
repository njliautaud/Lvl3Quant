"""
HC #334 step 3 — TINY CONTEXTUAL BANDIT on top-confidence regime.

Per HC #334: tiny RL over {long, short, no-trade} × bracket choice with
state = head predictions, reward = realized FIFO PnL.

Per the MLP step 2 NEGATIVE-RESULT finding: train on TOP-1% confidence regime
ONLY (not full population) — avoids the distribution-shift trap that broke
the MLP meta-learner.

Algorithm: per-action linear regressor on standardized head vector → predict
expected FIFO PnL (in ticks). Action = argmax E[PnL]. Closed-form ridge solve,
no SGD needed. This is a contextual bandit (one-step), not full RL.

Action space:
  0 = no_trade  (reward = 0)
  1 = long_tp4sl3   (reward = +target_fifo_tp4sl3_net)
  2 = short_tp4sl3  (reward = -target_fifo_tp4sl3_net)
  3 = long_tp8sl5   (reward = +target_fifo_tp8sl5_net)
  4 = short_tp8sl5  (reward = -target_fifo_tp8sl5_net)

Training: ridge regression per arm on TRAIN-half (first 60% OOT in top-1%).
Inference: argmax over 5 arms on TEST-half (last 40% OOT in top-1%).

Per HC #320 FIFO labels only. Per HC #321 mandatory fields. Per HC #322 lead
with trading-system perf.

OUT: output/v3_2_deep_sim_20260512/bandit_executor_hc334.{json,csv}
"""
from __future__ import annotations
import json
import math
from pathlib import Path
import numpy as np

PREDS = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512")
OUT_JSON = OUT_DIR / "bandit_executor_hc334.json"
OUT_CSV = OUT_DIR / "bandit_executor_hc334.csv"

ES_TICK_USD = 12.50
TOP_PCT = 0.05  # widen the regime — top 5% (optuna found 0.1% best for rules)
MIN_GAP = 11    # 2.75s anti-churn at 250ms stride
RIDGE = 1.0     # ridge regularization

print(f"[load] {PREDS}")
Z = dict(np.load(PREDS))
N = Z["target_log_ret_1s"].shape[0]
print(f"[load] n_samples={N}")

# Features
PRED_KEYS = sorted([k for k in Z.keys() if k.startswith("pred_")])
X = np.stack([Z[k].astype(np.float32) for k in PRED_KEYS], axis=1)
X = np.where(np.isfinite(X), X, 0.0).astype(np.float32)
print(f"[features] X.shape={X.shape}, n_features={X.shape[1]}")

# FIFO targets
TGT4 = Z["target_fifo_tp4sl3_net"].astype(np.float32)
TGT8 = Z["target_fifo_tp8sl5_net"].astype(np.float32)
TGT_MFE = Z["target_pred_mfe_30s_ticks"].astype(np.float32)
TGT_MAE = Z["target_pred_mae_30s_ticks"].astype(np.float32)
TGT_TIME = Z["target_pred_time_to_mfe_secs"].astype(np.float32)

# Filter to top-pct confidence regime
ABS_LR1 = np.abs(Z["pred_log_ret_1s"]).astype(np.float32)
k = int(N * TOP_PCT)
top_idx_all = np.argpartition(-ABS_LR1, k - 1)[:k]
top_idx_all = np.sort(top_idx_all)
print(f"[regime] top {TOP_PCT*100:.1f}% = {len(top_idx_all)} samples")

# Reward matrix R: (n_top, 5_actions)
n_top = len(top_idx_all)
R = np.zeros((n_top, 5), dtype=np.float32)
R[:, 0] = 0.0                     # no_trade
R[:, 1] = TGT4[top_idx_all]       # long tp4sl3
R[:, 2] = -TGT4[top_idx_all]      # short tp4sl3 (sign-flip approx per HC #320)
R[:, 3] = TGT8[top_idx_all]       # long tp8sl5
R[:, 4] = -TGT8[top_idx_all]      # short tp8sl5 (sign-flip approx)
print(f"[rewards] action-wise mean: no={R[:,0].mean():+.3f} L4={R[:,1].mean():+.3f} "
      f"S4={R[:,2].mean():+.3f} L8={R[:,3].mean():+.3f} S8={R[:,4].mean():+.3f}")

# Feature matrix in top regime
X_top = X[top_idx_all]

# Temporal split: first 60% of TOP regime = meta-train, last 40% = meta-test
SPLIT = int(n_top * 0.6)
print(f"[split] meta-train [0:{SPLIT}), meta-test [{SPLIT}:{n_top})")
X_tr, X_te = X_top[:SPLIT], X_top[SPLIT:]
R_tr, R_te = R[:SPLIT], R[SPLIT:]

# Standardize on train only (HC #333)
mu = X_tr.mean(axis=0, keepdims=True)
sd = X_tr.std(axis=0, keepdims=True) + 1e-6
X_tr_n = (X_tr - mu) / sd
X_te_n = (X_te - mu) / sd

# Add bias column
X_tr_b = np.hstack([X_tr_n, np.ones((X_tr_n.shape[0], 1), dtype=np.float32)])
X_te_b = np.hstack([X_te_n, np.ones((X_te_n.shape[0], 1), dtype=np.float32)])

# Closed-form ridge per arm:  w = (X^T X + λI)^-1 X^T r
d = X_tr_b.shape[1]
A = X_tr_b.T @ X_tr_b + RIDGE * np.eye(d, dtype=np.float32)
A_inv = np.linalg.inv(A).astype(np.float32)
W = np.zeros((d, 5), dtype=np.float32)
for a in range(5):
    W[:, a] = A_inv @ (X_tr_b.T @ R_tr[:, a])
print(f"[fit] ridge solved for 5 arms")

# Inference on test set
Q_te = X_te_b @ W  # (n_test, 5)

# Strategy 1: pure argmax
actions_argmax = Q_te.argmax(axis=1)
# Strategy 2: argmax with "skip if best Q < threshold"
def evaluate_policy(actions: np.ndarray, min_q: float | None = None,
                    Q: np.ndarray | None = None) -> dict:
    sel_mask = (actions != 0)  # filter out no_trade
    if min_q is not None and Q is not None:
        best_q = Q.max(axis=1)
        sel_mask = sel_mask & (best_q > min_q)
    # min_gap on top-regime indices
    sel_local = np.where(sel_mask)[0]
    # gap in the original timestep space
    sel_global = top_idx_all[SPLIT:][sel_local]
    taken_local = []
    last_global = -10**9
    for i_local, i_global in zip(sel_local, sel_global):
        if i_global - last_global < MIN_GAP:
            continue
        taken_local.append(int(i_local))
        last_global = int(i_global)
    if len(taken_local) < 20:
        return {"n_trades": int(len(taken_local)), "ok": False}
    taken_local = np.asarray(taken_local, dtype=np.int64)
    a = actions[taken_local]
    realized = R_te[taken_local, a]
    n = len(realized)
    mean = float(realized.mean())
    std = float(realized.std(ddof=1))
    sharpe = mean / std * math.sqrt(252) if std > 1e-9 else float("nan")
    neg = realized[realized < 0]
    dn = float(neg.std(ddof=1)) if len(neg) > 1 else 0.0
    sortino = mean / dn * math.sqrt(252) if dn > 1e-9 else float("nan")
    gw = float(realized[realized > 0].sum())
    gl = -float(realized[realized < 0].sum())
    pf = gw / gl if gl > 1e-9 else float("inf")
    wr = float((realized > 0).mean() * 100.0)
    # action distribution
    action_names = ["no_trade", "L_tp4sl3", "S_tp4sl3", "L_tp8sl5", "S_tp8sl5"]
    action_counts = {action_names[i]: int((a == i).sum()) for i in range(5)}
    # gather slice for HC #321 fields
    test_global_taken = top_idx_all[SPLIT:][taken_local]
    return {
        "n_trades": n,
        "wr_pct": wr,
        "mean_ticks": mean,
        "sharpe": float(sharpe) if math.isfinite(sharpe) else None,
        "sortino": float(sortino) if math.isfinite(sortino) else None,
        "pf": float(pf) if math.isfinite(pf) else None,
        "total_ticks": float(realized.sum()),
        "total_usd": float(realized.sum() * ES_TICK_USD),
        "hold_secs_mean": float(TGT_TIME[test_global_taken].mean()),
        "mfe_mean": float(TGT_MFE[test_global_taken].mean()),
        "mae_mean": float(TGT_MAE[test_global_taken].mean()),
        "action_counts": action_counts,
        "min_q_filter": min_q,
        "ok": True,
    }

print("\n[eval] policy sweep with min_q filter")
rows = []
results = []
for min_q in [None, 0.0, 0.25, 0.5, 1.0, 1.5, 2.0]:
    r = evaluate_policy(actions_argmax, min_q=min_q, Q=Q_te)
    results.append(r)
    if r.get("ok"):
        ac = r.get("action_counts", {})
        print(f"  min_q={str(min_q):>5} n={r['n_trades']:5d} WR={r['wr_pct']:5.1f}% "
              f"Sharpe={r.get('sharpe', 0):+.2f} mean={r['mean_ticks']:+.3f}t "
              f"PF={r.get('pf', 0):.2f}  actions: "
              f"L4={ac.get('L_tp4sl3',0)} S4={ac.get('S_tp4sl3',0)} "
              f"L8={ac.get('L_tp8sl5',0)} S8={ac.get('S_tp8sl5',0)}")
    else:
        print(f"  min_q={str(min_q):>5} n={r.get('n_trades', 0)} insufficient")

OUT_DIR.mkdir(parents=True, exist_ok=True)
OUT_JSON.write_text(json.dumps({
    "algorithm": "contextual_bandit_ridge",
    "regime": f"top_{TOP_PCT*100:.1f}_pct_by_abs_pred_log_ret_1s",
    "n_features": int(X.shape[1]),
    "n_train_in_regime": int(SPLIT),
    "n_test_in_regime": int(n_top - SPLIT),
    "ridge_lambda": float(RIDGE),
    "min_gap_steps": int(MIN_GAP),
    "action_space": ["no_trade", "L_tp4sl3", "S_tp4sl3", "L_tp8sl5", "S_tp8sl5"],
    "results": results,
    "leakage_audit": {
        "fold_idx": int(Z["fold_idx"]),
        "oot_dates": list(map(str, Z["oot_dates"])),
        "split_method": "temporal 60/40 within top-1% OOT subset",
        "feature_stats_train_only": True,
        "pnl_source": "target_fifo_{tp4sl3,tp8sl5}_net (MBO bid/ask FIFO labels)",
        "midpoint_used": False,
        "short_side_caveat": "sign-flip approximation per HC #320",
    },
}, indent=2))

if results:
    flat_rows = []
    for r in results:
        flat = {k: v for k, v in r.items() if not isinstance(v, dict)}
        ac = r.get("action_counts", {})
        for k, v in ac.items():
            flat[f"action_{k}"] = v
        flat_rows.append(flat)
    keys = list(flat_rows[0].keys())
    with OUT_CSV.open("w") as f:
        f.write(",".join(keys) + "\n")
        for r in flat_rows:
            f.write(",".join(str(r.get(k, "")) for k in keys) + "\n")

print(f"\n[wrote] {OUT_JSON}")
print(f"[wrote] {OUT_CSV}")
