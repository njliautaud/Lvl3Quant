import numpy as np
import glob
from scipy.stats import pearsonr, spearmanr

npz_paths = sorted(glob.glob("output/v342_extended_oot/fold_00_extended_v2_chunk*_20260520_154518.npz"))

z1 = np.load(npz_paths[0], allow_pickle=True)
metric_keys = sorted(k for k in z1.files if k.startswith("metric_ic"))
print("Metric IC keys in chunk1 NPZ:", metric_keys)
print()

print("Per-chunk NATIVE metric_ic (saved by inference script — same as inference log = canonical baseline metric):")
hdr = "chunk".ljust(8)
for h in ["1s","5s","10s","30s"]: hdr += "  m_IC_" + h.ljust(6)
print(hdr)
for p in npz_paths:
    z = np.load(p, allow_pickle=True)
    chunk = p.split("chunk")[1].split("_")[0]
    row = ("chunk" + chunk).ljust(8)
    for h in ["1s","5s","10s","30s"]:
        key = f"metric_ic_log_ret_{h}"
        v = float(z[key]) if key in z.files else float("nan")
        row += f"  {v:+.4f}      "
    print(row)
print()

# Mean of per-chunk native metric (weighted by chunk size — approximates concat)
print("Sample-size-weighted mean of native metric IC (across 6 chunks):")
for h in ["1s","5s","10s","30s"]:
    weights, vals = [], []
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        n = len(z["pred_log_ret_1s"])
        v = float(z[f"metric_ic_log_ret_{h}"])
        weights.append(n)
        vals.append(v)
    weighted = np.average(vals, weights=weights)
    print(f"  {h}: weighted_mean_native_IC = {weighted:+.4f}  (per-chunk: {[f'{v:+.3f}' for v in vals]})")
print()

print("=== Exploratory: Pearson vs Spearman concat across 52 days ===")
for h in ["log_ret_1s", "log_ret_5s", "log_ret_10s"]:
    preds, lbls = [], []
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        pred = z[f"pred_{h}"]
        lbl = z[f"target_{h}"]
        f_msk = np.isfinite(pred) & np.isfinite(lbl)
        preds.append(pred[f_msk])
        lbls.append(lbl[f_msk])
    p_all = np.concatenate(preds)
    l_all = np.concatenate(lbls)
    ic_p, _ = pearsonr(p_all, l_all)
    ic_s, _ = spearmanr(p_all, l_all)
    print(f"  {h}: n={len(p_all):>8} pearson={ic_p:+.4f}  spearman={ic_s:+.4f}")
