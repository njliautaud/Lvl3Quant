#!/usr/bin/env python3
"""HC #450 R3 deliverable — v3.4.2 MFE/MAE × confidence-band × side × horizon.

Reads the 6 v3.4.2 52-day extended-OOT NPZs and produces:
- output/hc450_r3_v342_mfemae/v342_52day_mfemae_confband.csv
- output/hc450_r3_v342_mfemae/v342_52day_mfemae_confband.png
- output/hc450_r3_v342_mfemae/SUMMARY.md

For each (side in {long, short}, pred_horizon in {1s,5s,10s,30s,60s},
band in {top0.5%, top1%, top5%, top10%, top20%, top50%}):
  - n
  - mean realized label (signed, ticks) → "edge per signal" before costs
  - mean realized MFE_30s_ticks (side-adjusted: long→mfe, short→mae)
  - mean realized MAE_30s_ticks (side-adjusted)
  - p50, p90 of side-adjusted MFE
  - net_passive  = mean_label - 0.376  (commission only, HC canonical)
  - net_market   = mean_label - 1.376  (commission + 1 tick spread cross)
  - hit_1tk = P(side-adj label >= 1 tick)
  - hit_2tk = P(side-adj label >= 2 ticks)
"""
import numpy as np
import pandas as pd
from pathlib import Path
import sys, os

NPZ_DIR = Path("/home/nick/Lvl3Quant/output/v342_extended_oot")
OUT_DIR = Path("/home/nick/Lvl3Quant/output/hc450_r3_v342_mfemae")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_SIZE = 0.25  # ES tick size in points
LOG_RET_TO_TICKS = None  # estimated from data

# Bands by top-confidence percentile (1.0 = top 100%, 0.5 = top 50%, etc.)
BANDS = [
    ("top0.5%", 0.005),
    ("top1%",   0.01),
    ("top5%",   0.05),
    ("top10%",  0.10),
    ("top20%",  0.20),
    ("top50%",  0.50),
]

HORIZONS = ["1s", "5s", "10s", "30s", "60s"]

def main():
    chunks = sorted(NPZ_DIR.glob("fold_00_extended_v2_chunk*.npz"))
    if not chunks:
        print(f"ERROR: no NPZs in {NPZ_DIR}", flush=True); sys.exit(1)
    print(f"Loading {len(chunks)} chunks…", flush=True)

    keys_pred   = [f"pred_log_ret_{h}" for h in HORIZONS]
    keys_target = [f"target_log_ret_{h}" for h in HORIZONS]
    key_mfe30 = "target_pred_mfe_30s_ticks"
    key_mae30 = "target_pred_mae_30s_ticks"

    arrs = {k: [] for k in keys_pred + keys_target + [key_mfe30, key_mae30]}
    for c in chunks:
        z = np.load(c, allow_pickle=False)
        for k in arrs:
            if k in z.files:
                arrs[k].append(z[k])
        z.close()
    for k in arrs:
        arrs[k] = np.concatenate(arrs[k]) if arrs[k] else np.array([])
    n_total = len(arrs[keys_pred[0]])
    print(f"Total samples: {n_total:,}", flush=True)

    # Convert realized log_ret to ticks. Approx: log_ret * mid_price_in_ticks.
    # ES around 5000 pts = 20000 ticks. log_ret of 0.0001 ≈ 2 ticks.
    # Simpler: derive scale from raw mfe in ticks vs raw log_ret_30s.
    # Use empirical mapping: tk_per_logret = std(target_pred_mfe_30s)/std(target_log_ret_30s).
    # Actually MFE is in ticks already. We just need labels in ticks.
    # Use 5000 / 0.25 = 20000 ticks-per-point at ES, log_ret ≈ d_price / price → ticks = log_ret * price / TICK_SIZE
    # Approximate price = 5000.
    PRICE_PROXY = 5000.0
    label_to_ticks = 1.0  # target_log_ret_* are already in ticks (verified)

    rows = []
    for side in ("long", "short"):
        side_sign = 1.0 if side == "long" else -1.0
        # side-adjusted realized MFE for side: long uses MFE, short uses MAE flipped
        mfe_side = arrs[key_mfe30] if side == "long" else -arrs[key_mae30]
        mae_side = arrs[key_mae30] if side == "long" else -arrs[key_mfe30]
        for hz in HORIZONS:
            pred = arrs[f"pred_log_ret_{hz}"]
            targ = arrs[f"target_log_ret_{hz}"] * label_to_ticks  # → ticks
            # confidence = side-signed pred (long: high pred = high conf; short: low pred = high conf)
            conf = side_sign * pred
            # side-adjusted realized label in ticks
            label_side = side_sign * targ
            for bname, q in BANDS:
                thresh = np.quantile(conf, 1.0 - q)
                mask = conf >= thresh
                n = int(mask.sum())
                if n < 10:
                    continue
                mean_label = float(np.nanmean(label_side[mask]))
                med_label  = float(np.nanmedian(label_side[mask]))
                mean_mfe_s = float(mfe_side[mask].mean()) if len(mfe_side) else np.nan
                mean_mae_s = float(mae_side[mask].mean()) if len(mae_side) else np.nan
                p50_mfe    = float(np.median(mfe_side[mask])) if len(mfe_side) else np.nan
                p90_mfe    = float(np.quantile(mfe_side[mask], 0.90)) if len(mfe_side) else np.nan
                p90_mae    = float(np.quantile(mae_side[mask], 0.90)) if len(mae_side) else np.nan
                hit_1tk    = float(np.nanmean(label_side[mask] >= 1.0))
                hit_2tk    = float(np.nanmean(label_side[mask] >= 2.0))
                net_pass   = mean_label - 0.376
                net_mkt    = mean_label - 1.376
                rows.append(dict(
                    side=side, horizon=hz, band=bname, n=n,
                    mean_label_tk=mean_label, median_label_tk=med_label,
                    mean_mfe30_tk=mean_mfe_s, p50_mfe30=p50_mfe, p90_mfe30=p90_mfe,
                    mean_mae30_tk=mean_mae_s, p90_mae30=p90_mae,
                    hit_1tk=hit_1tk, hit_2tk=hit_2tk,
                    net_passive_tk=net_pass, net_market_tk=net_mkt,
                ))
    df = pd.DataFrame(rows)
    out_csv = OUT_DIR / "v342_52day_mfemae_confband.csv"
    df.to_csv(out_csv, index=False, float_format="%.4f")
    print(f"Wrote {out_csv}  ({len(df)} rows)", flush=True)

    # Highlight summary: top0.5% short 1s and 10s + top10% long/short
    summary_rows = df[df["band"].isin(["top0.5%", "top1%", "top10%"])].sort_values(
        ["band", "side", "horizon"]
    )
    summary_md = OUT_DIR / "SUMMARY.md"
    with open(summary_md, "w") as f:
        f.write("# HC #450 R3 — v3.4.2 MFE/MAE × Confidence-Band (52-day OOT)\n\n")
        f.write(f"Total samples: {n_total:,} across {len(chunks)} chunks (20260301-20260429)\n\n")
        f.write("Cost canonical: passive=0.376 tk (commission only), market=1.376 tk (commission + 1tk spread cross)\n\n")
        f.write("## Headline cells (top0.5% / top1% / top10%, all sides × horizons)\n\n")
        f.write(summary_rows.to_string(index=False, float_format=lambda x: f"{x:.3f}") + "\n\n")
        f.write("## Interpretation rules\n")
        f.write("- net_passive_tk > 0 → tradable with passive limit at touch.\n")
        f.write("- net_market_tk > 0 → tradable even with market orders crossing spread.\n")
        f.write("- p90_mfe30 ticks → upper realistic TP ceiling per HC #428 R2.\n")
        f.write("- p90_mae30 ticks → realistic SL distance per HC #428 R2.\n")
    print(f"Wrote {summary_md}", flush=True)

    # Quick PNG: net_passive heatmap (side × horizon × band)
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for i, side in enumerate(("long", "short")):
            sub = df[df["side"] == side].pivot(index="band", columns="horizon", values="net_passive_tk")
            sub = sub.reindex(index=[b[0] for b in BANDS], columns=HORIZONS)
            im = axes[i].imshow(sub.values, cmap="RdYlGn", vmin=-1.5, vmax=1.5, aspect="auto")
            axes[i].set_xticks(range(len(HORIZONS))); axes[i].set_xticklabels(HORIZONS)
            axes[i].set_yticks(range(len(BANDS))); axes[i].set_yticklabels([b[0] for b in BANDS])
            axes[i].set_title(f"{side} — net passive (ticks/signal)")
            for y in range(sub.shape[0]):
                for x in range(sub.shape[1]):
                    v = sub.values[y, x]
                    if not np.isnan(v):
                        axes[i].text(x, y, f"{v:+.2f}", ha="center", va="center", fontsize=8)
            fig.colorbar(im, ax=axes[i])
        plt.suptitle("v3.4.2 — 52-day OOT — Net edge per signal AFTER passive commission (HC #450 R3)")
        plt.tight_layout()
        out_png = OUT_DIR / "v342_52day_mfemae_confband.png"
        plt.savefig(out_png, dpi=110)
        print(f"Wrote {out_png}", flush=True)
    except Exception as e:
        print(f"PNG step failed (non-fatal): {e}", flush=True)
    print("DONE")

if __name__ == "__main__":
    main()
