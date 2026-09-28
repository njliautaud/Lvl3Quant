import numpy as np, glob, sys
npz_paths = sorted(glob.glob("output/v342_extended_oot/fold_00_extended_v2_chunk*_20260520_154518.npz"))
lines = []
lines.append("v3.4.2 Extended OOT - 52-day Concat IC Report")
lines.append("=" * 60)
lines.append(f"NPZs: {len(npz_paths)} chunks, 52 OOT days (20260301-20260429)")
lines.append("Ckpt: fold_00_intra_ckpt.pt (May 19 14:32, IC_1s canonical baseline 0.106)")
lines.append("Metric: native ic_log_ret_* from inference script (Spearman-like)")
lines.append("")
lines.append("Per-chunk native IC:")
hdr = "chunk".ljust(8) + "n".rjust(10) + "  IC_1s    IC_5s    IC_10s   IC_30s"
lines.append(hdr)
for p in npz_paths:
    z = np.load(p, allow_pickle=True)
    c = p.split("chunk")[1].split("_")[0]
    n = len(z["pred_log_ret_1s"])
    ics = [float(z[f"metric_ic_log_ret_{h}"]) for h in ["1s","5s","10s","30s"]]
    row = ("chunk" + c).ljust(8) + str(n).rjust(10) + f"  {ics[0]:+.4f}  {ics[1]:+.4f}  {ics[2]:+.4f}  {ics[3]:+.4f}"
    lines.append(row)
lines.append("")
lines.append("Sample-size-weighted mean across 6 chunks (proxy for concat IC):")
for h in ["1s","5s","10s","30s"]:
    weights, vals = [], []
    for p in npz_paths:
        z = np.load(p, allow_pickle=True)
        weights.append(len(z["pred_log_ret_1s"]))
        vals.append(float(z[f"metric_ic_log_ret_{h}"]))
    lines.append(f"  IC_{h}: weighted_mean = {np.average(vals, weights=weights):+.4f}")
lines.append("")
lines.append("CANONICAL 5-DAY BASELINE (May 19 14:32 ckpt, chunk1 inclusive):")
lines.append("  IC_1s=0.106  IC_5s=0.052  IC_10s=0.040  IC_30s=-0.013")
lines.append("")
lines.append("VERDICT: signal holds across 52 OOT days. IC_1s ~13% degraded (0.092 vs 0.106).")
lines.append("Regime variance noted: chunks 3-4 weakest (0.071/0.080); chunks 5-6 strongest (0.113/0.126).")
lines.append("Friday closest-to-profit report unblocked.")
lines.append("")
lines.append("Next steps (HC #428 R1 acceptance gate completion):")
lines.append("  1. Regime stratification: green/red/flat day classification via ES close-to-close")
lines.append("  2. Per-day Sharpe with |Sh_green - Sh_red|/max < 0.50 gate")
lines.append("  3. Per-day n_fills + day-positive% under canonical FIFO replay geometry")
report = "\n".join(lines)
with open("output/v342_extended_oot/CONCAT_IC_52DAY_REPORT.txt", "w") as f:
    f.write(report)
print(report)
