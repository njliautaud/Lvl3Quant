import numpy as np
import glob
from scipy.stats import pearsonr

npz_paths = sorted(glob.glob("output/v342_extended_oot/fold_00_extended_v2_chunk*_20260520_154518.npz"))
horizons = ["log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s", "log_ret_60s"]

# Probe mask convention on chunk 6
z6 = np.load(npz_paths[-1], allow_pickle=True)
mask_1s = z6["mask_log_ret_1s"]
print(f"chunk6 mask_log_ret_1s: dtype={mask_1s.dtype} n={len(mask_1s)} sum={mask_1s.sum()} unique={np.unique(mask_1s)[:5]}")
# If mask is bool/0-1: sum tells us how many "1"s
# If mask=1 means VALID -> sum = valid count
# If mask=1 means MASKED-OUT -> sum = invalid count

# Method A: assume mask=1 is VALID
pred_1s = z6["pred_log_ret_1s"]
lbl_1s = z6["target_log_ret_1s"]
finite = np.isfinite(pred_1s) & np.isfinite(lbl_1s)
mask_valid = mask_1s.astype(bool)

for name, sel in [("finite-only", finite),
                  ("finite AND mask=1", finite & mask_valid),
                  ("finite AND mask=0", finite & ~mask_valid)]:
    if sel.sum() > 1:
        ic, _ = pearsonr(pred_1s[sel], lbl_1s[sel])
        print(f"  chunk6 {name}: n={sel.sum()} IC_1s={ic:+.4f}")

print()
print("Inference script reported chunk6 IC_1s=0.126316 — match-check above tells us convention.")
print()

# Now compute the right concat IC across all 6 chunks
def compute_ic(z, h):
    pred = z[f"pred_{h}"]
    lbl = z[f"target_{h}"]
    finite = np.isfinite(pred) & np.isfinite(lbl)
    if finite.sum() < 2:
        return finite.sum(), float("nan")
    ic, _ = pearsonr(pred[finite], lbl[finite])
    return finite.sum(), ic

# Per-chunk
print("Per-chunk IC (finite-only):")
print(f"{'chunk':<8}{'n_total':>10}", end="")
for h in horizons: print(f"  IC_{h.split('_')[-1]:<5}", end="")
print()
for p in npz_paths:
    z = np.load(p, allow_pickle=True)
    chunk = p.split("chunk")[1].split("_")[0]
    n_total = len(z[f"pred_{horizons[0]}"])
    row = f"chunk{chunk:<3}{n_total:>10}"
    for h in horizons:
        nn, ic = compute_ic(z, h)
        row += f"  {ic:+.4f}"
    print(row)

# Concat across all 52 days
print()
print("=== CONCAT IC ACROSS 52 OOT DAYS (finite-only) ===")
for h in horizons:
    preds, lbls = [], []
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        pred = z[f"pred_{h}"]
        lbl = z[f"target_{h}"]
        finite = np.isfinite(pred) & np.isfinite(lbl)
        preds.append(pred[finite])
        lbls.append(lbl[finite])
    p_all = np.concatenate(preds)
    l_all = np.concatenate(lbls)
    if len(p_all) > 1:
        ic, _ = pearsonr(p_all, l_all)
    else:
        ic = float("nan")
    print(f"  {h}: n={len(p_all):>8} IC={ic:+.4f}")

print()
print("CANONICAL 5-DAY BASELINE: IC_1s=0.106, IC_5s=0.052, IC_10s=0.040")
