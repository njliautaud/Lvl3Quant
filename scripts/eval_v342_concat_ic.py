"""Compute concat IC for CNN-Mamba v3.4.2 (first 3 folds)."""
import numpy as np
from scipy.stats import spearmanr

BASE = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_hc477fix_v2"
FOLDS = [0, 1, 2]
HORIZONS = ["1s", "5s", "10s", "30s"]

# Load all folds
fold_data = {}
for f in FOLDS:
    path = f"{BASE}/fold_{f:02d}_oot_predictions.npz"
    fold_data[f] = np.load(path, allow_pickle=True)

print("="*60)
print("CNN-Mamba v3.4.2 — Concat IC (folds 0-2)")
print("="*60)

# Per-fold info
for f in FOLDS:
    d = fold_data[f]
    n = d['pred_log_ret_1s'].shape[0]
    dates = d['oot_dates']
    print(f"\nFold {f}: {n:,} samples, OOT dates: {list(dates)}")

# Compute per-fold and concat IC
print("\n" + "-"*60)
print(f"{'Horizon':<10} | {'Fold 0':>8} {'Fold 1':>8} {'Fold 2':>8} | {'Concat IC':>10} | {'v2 baseline':>12}")
print("-"*60)

v2_baseline = {"1s": 0.222, "5s": 0.141, "10s": 0.106, "30s": None}

for h in HORIZONS:
    pred_key = f"pred_log_ret_{h}"
    tgt_key = f"target_log_ret_{h}"
    mask_key = f"mask_log_ret_{h}"

    per_fold_ic = []
    all_preds = []
    all_tgts = []

    for f in FOLDS:
        d = fold_data[f]
        mask = d[mask_key].astype(bool)
        p = d[pred_key][mask]
        t = d[tgt_key][mask]

        # Remove NaN/Inf
        valid = np.isfinite(p) & np.isfinite(t)
        p, t = p[valid], t[valid]

        if len(p) > 10:
            ic, _ = spearmanr(p, t)
            per_fold_ic.append(ic)
        else:
            per_fold_ic.append(float('nan'))

        all_preds.append(p)
        all_tgts.append(t)

    # Concat IC
    cat_p = np.concatenate(all_preds)
    cat_t = np.concatenate(all_tgts)
    concat_ic, _ = spearmanr(cat_p, cat_t)

    baseline_str = f"{v2_baseline[h]:.3f}" if v2_baseline[h] else "N/A"
    delta = ""
    if v2_baseline[h] is not None:
        d_val = concat_ic - v2_baseline[h]
        delta = f" ({d_val:+.3f})"

    print(f"{h:<10} | {per_fold_ic[0]:>8.3f} {per_fold_ic[1]:>8.3f} {per_fold_ic[2]:>8.3f} | {concat_ic:>10.4f} | {baseline_str:>8}{delta}")

print("-"*60)
print(f"\nTotal concat samples: {len(cat_p):,}")

# Also check p_up heads at key horizons
print("\n" + "="*60)
print("Auxiliary heads — concat IC (Spearman)")
print("="*60)
aux_heads = [
    ("p_up_5s", "P(up) 5s"),
    ("p_up_10s", "P(up) 10s"),
    ("p_up_30s", "P(up) 30s"),
    ("fifo_tp4sl3_net", "FIFO tp4sl3 net"),
    ("fifo_tp8sl5_net", "FIFO tp8sl5 net"),
    ("fifo_tp4sl3_hit_tp", "FIFO tp4sl3 hit_tp"),
    ("fifo_tp8sl5_hit_tp", "FIFO tp8sl5 hit_tp"),
    ("pred_mfe_30s_ticks", "MFE 30s"),
    ("pred_mae_30s_ticks", "MAE 30s"),
]

for key, label in aux_heads:
    pred_key = f"pred_{key}"
    tgt_key = f"target_{key}"
    mask_key = f"mask_{key}"

    all_p, all_t = [], []
    for f in FOLDS:
        d = fold_data[f]
        if pred_key not in d:
            continue
        mask = d[mask_key].astype(bool)
        p = d[pred_key][mask]
        t = d[tgt_key][mask]
        valid = np.isfinite(p) & np.isfinite(t)
        all_p.append(p[valid])
        all_t.append(t[valid])

    if all_p:
        cp = np.concatenate(all_p)
        ct = np.concatenate(all_t)
        ic, _ = spearmanr(cp, ct)
        print(f"  {label:<25}: {ic:.4f}  (n={len(cp):,})")
