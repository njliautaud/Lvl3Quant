import numpy as np
import glob
from scipy.stats import pearsonr

npz_paths = sorted(glob.glob("output/v342_extended_oot/fold_00_extended_v2_chunk*_20260520_154518.npz"))
horizons = ["log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s", "log_ret_60s"]

all_pred = {h: [] for h in horizons}
all_lbl = {h: [] for h in horizons}
print(f"{'chunk':<8}{'n':>10}", end="")
for h in horizons: print(f"  IC_{h.split('_')[-1]:<5}", end="")
print()

per_chunk_ic = {}
for p in npz_paths:
    z = np.load(p, allow_pickle=True)
    chunk = p.split("chunk")[1].split("_")[0]
    n = len(z[f"pred_{horizons[0]}"])
    row = f"chunk{chunk:<3}{n:>10}"
    per_chunk_ic[chunk] = {}
    for h in horizons:
        pred = z[f"pred_{h}"]
        lbl = z[f"target_{h}"]
        msk = z[f"mask_{h}"].astype(bool)
        valid = msk & np.isfinite(pred) & np.isfinite(lbl)
        if valid.sum() > 10:
            ic, _ = pearsonr(pred[valid], lbl[valid])
        else:
            ic = float("nan")
        per_chunk_ic[chunk][h] = ic
        row += f"  {ic:+.4f}"
        all_pred[h].append(pred[valid])
        all_lbl[h].append(lbl[valid])
    print(row)

print()
print("=== CONCAT IC ACROSS ALL 52 OOT DAYS (chunks 1-6) ===")
print(f"{'horizon':<10}{'n_valid':>10}{'IC':>10}")
concat_results = {}
for h in horizons:
    p_all = np.concatenate(all_pred[h])
    l_all = np.concatenate(all_lbl[h])
    ic, _ = pearsonr(p_all, l_all)
    concat_results[h] = (len(p_all), ic)
    print(f"{h:<10}{len(p_all):>10}{ic:>+10.4f}")

print()
print("=== vs CANONICAL 5-DAY BASELINE (May 19 14:32 ckpt) ===")
print("baseline: IC_1s=0.106, IC_5s=0.052, IC_10s=0.040")

# Save report
report_path = "output/v342_extended_oot/CONCAT_IC_52DAY_REPORT.txt"
with open(report_path, "w") as f:
    f.write("v3.4.2 Extended OOT - 52-day Concat IC Report\n")
    f.write("=" * 60 + "\n")
    f.write(f"NPZs: {len(npz_paths)} chunks, 52 OOT days (20260301-20260429)\n")
    f.write("Ckpt: fold_00_intra_ckpt.pt (May 19 14:32)\n\n")
    f.write("PER-CHUNK IC:\n")
    for c, ics in sorted(per_chunk_ic.items()):
        f.write(f"  chunk{c}: " + " ".join(f"{h.split('_')[-1]}={v:+.4f}" for h, v in ics.items()) + "\n")
    f.write("\nCONCAT IC (52 days):\n")
    for h, (n, ic) in concat_results.items():
        f.write(f"  {h}: n={n:>8} IC={ic:+.4f}\n")
    f.write("\nBaseline (5-day): IC_1s=0.106, IC_5s=0.052, IC_10s=0.040\n")
print(f"Report written to {report_path}")
