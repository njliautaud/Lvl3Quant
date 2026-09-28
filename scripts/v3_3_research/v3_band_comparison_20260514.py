"""
v2 vs v3.2 vs v3.3 per-confidence-band comparison on common OOT 20260223.
Per HC #348 (just added).

Outputs:
  - output/v3_3_vs_v3_2_vs_v2_band_comparison_20260514/comparison.csv
  - output/v3_3_vs_v3_2_vs_v2_band_comparison_20260514/comparison.md
  - output/v3_3_vs_v3_2_vs_v2_band_comparison_20260514/comparison.png

Per malware-guard (HC #307D): NEW analysis script ONLY, no trainer code touched.

DATA LIMITATIONS / GAPS (honest reporting):
  - v3.3: fold_00 still training (Ep4 at time of write).  fold_00_intra_ckpt.pt
    exists but no prediction NPZ was emitted by the trainer.  v33_intra_ckpt_band_eval.py
    (HC #343 spec) does NOT exist on disk.  v3.3 is EXCLUDED from this comparison.
    Aggregate IC numbers from 04:33 ET Discord post for reference only:
      IC_1s=0.2694, IC_5s=0.126, IC_10s=0.083, IC_30s=0.0446
  - v3.2: fold_00_oot_predictions.npz covers FIVE days (20260223-20260227)
    concatenated with no per-day boundary index.  We report v3.2 metrics over
    the full 5-day OOT window (not 20260223 alone).  v2 metrics are 20260223 only.
    This asymmetry is documented in the output caption.
  - v2: cnn_mamba_v2_all_oot/fold_00_oot_predictions.npz IS 20260223 only.
    Has 3 horizons {1s,5s,10s}, no 30s.  30s row for v2 is N/A.

LABEL SCALE:
  - v2 labels are integer ticks (verified: max=23, min=-34, integer)
  - v3.2 targets are log returns (verified: target_log_ret_1s std=1.63 -- looks
    like ticks too actually). HC convention for v3.2 trainer labels: ticks.
  - Both treated as ticks. realized = labels[h] in ticks.

MFE/MAE:
  - v3.2 has true per-trade MFE_30s/MAE_30s in ticks (target_pred_mfe_30s_ticks).
  - v2 does NOT have MFE/MAE -- we approximate by horizon-h realized return:
      MFE_h_approx = max(0, realized_h)
      MAE_h_approx = min(0, realized_h)
    Documented limitation; we use the same approximation for v3.2 short-horizon
    MFE/MAE for apples-to-apples cross-model comparison.

MAGCORR per HC #348:
  - MagCorr_raw = pearson(|pred|, |realized|)
  - MagCorr_ticks = MagCorr_raw * sd(realized_h)  (within the selected band)
"""
import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


OUT_DIR = "/home/jupiter/Lvl3Quant/output/v3_3_vs_v3_2_vs_v2_band_comparison_20260514"
os.makedirs(OUT_DIR, exist_ok=True)

V2_NPZ = "/tmp/v2_fold00_20260223.npz"  # rsync'd from neptune:/home/nick/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar/fold_00_oot_predictions.npz (the local symlink target is broken)
V32_NPZ = "/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"

BANDS = [0.001, 0.005, 0.01, 0.05, 0.10]  # top 0.1%, 0.5%, 1%, 5%, 10%
BAND_LABELS = ["0.1%", "0.5%", "1%", "5%", "10%"]
HORIZONS = ["1s", "5s", "10s", "30s"]
SIDES = ["long", "short"]


def load_v2():
    """v2 -> dict per horizon of (pred, realized) in ticks. 20260223 only."""
    d = np.load(V2_NPZ)
    preds = d["predictions"]  # (N, 3) for 1s/5s/10s
    labels = d["labels"]      # (N, 3) ticks
    horizons = list(d["horizons"])  # ['1s','5s','10s']
    out = {}
    for i, h in enumerate(horizons):
        out[h] = {
            "pred": preds[:, i].astype(np.float64),
            "realized": labels[:, i].astype(np.float64),
        }
    return out, str(d["oot_files"][0]) if "oot_files" in d.files else "?"


def load_v32():
    """v3.2 -> dict per horizon of (pred, realized) in ticks. 5-day OOT concat."""
    d = np.load(V32_NPZ)
    out = {}
    for h in HORIZONS:
        pkey = f"pred_log_ret_{h}"
        tkey = f"target_log_ret_{h}"
        mkey = f"mask_log_ret_{h}"
        if pkey not in d.files:
            continue
        p = d[pkey].astype(np.float64)
        t = d[tkey].astype(np.float64)
        m = d[mkey].astype(np.float64) > 0
        # drop nan
        valid = m & np.isfinite(p) & np.isfinite(t)
        out[h] = {"pred": p[valid], "realized": t[valid]}
    return out


def compute_bands_for(pred, realized, model, horizon):
    """Return list of rows for each (band, side)."""
    rows = []
    N = len(pred)
    # Aggregate (whole-day) IC
    if N > 1 and np.std(pred) > 0 and np.std(realized) > 0:
        ic_agg, _ = spearmanr(pred, realized)
    else:
        ic_agg = np.nan
    sd_realized = float(np.std(realized))

    for band_frac, band_lbl in zip(BANDS, BAND_LABELS):
        K = max(1, int(round(N * band_frac)))
        # absolute sort top K
        abs_pred = np.abs(pred)
        # top K by |pred|
        top_idx = np.argpartition(-abs_pred, K - 1)[:K]
        sub_pred = pred[top_idx]
        sub_real = realized[top_idx]

        for side in SIDES:
            if side == "long":
                side_mask = sub_pred > 0
            else:
                side_mask = sub_pred < 0
            n_side = int(side_mask.sum())
            if n_side < 5:
                rows.append({
                    "model": model, "horizon": horizon, "band": band_lbl,
                    "side": side, "n": n_side,
                    "IC": np.nan, "DA_pct": np.nan, "MagCorr_ticks": np.nan,
                    "MFE_ticks": np.nan, "MAE_ticks": np.nan,
                    "avg_move_ticks": np.nan,
                })
                continue

            sp = sub_pred[side_mask]
            sr = sub_real[side_mask]

            # IC (Spearman) within selected subset
            if np.std(sp) > 0 and np.std(sr) > 0:
                ic_sub, _ = spearmanr(sp, sr)
            else:
                ic_sub = np.nan

            # DA% per HC #313: fraction of correct sign
            da = float(np.mean(np.sign(sp) == np.sign(sr)) * 100.0)

            # MagCorr per HC #348: pearson(|pred|, |real|) * sd(realized) -> ticks
            if np.std(np.abs(sp)) > 0 and np.std(np.abs(sr)) > 0:
                mc_raw, _ = pearsonr(np.abs(sp), np.abs(sr))
            else:
                mc_raw = np.nan
            mc_ticks = mc_raw * float(np.std(sr)) if np.isfinite(mc_raw) else np.nan

            # MFE / MAE approximation from horizon-h realized return
            #   long side: MFE = max(0, realized), MAE = min(0, realized)
            #   short side: signal says price will go down; MFE = max(0, -realized)
            #                                                MAE = min(0, -realized)
            if side == "long":
                mfe = np.maximum(0.0, sr).mean()
                mae = np.minimum(0.0, sr).mean()
                avg_move = float(sr.mean())  # signed in direction of pred
            else:
                mfe = np.maximum(0.0, -sr).mean()
                mae = np.minimum(0.0, -sr).mean()
                avg_move = float((-sr).mean())  # signed in direction of pred (short)

            rows.append({
                "model": model, "horizon": horizon, "band": band_lbl,
                "side": side, "n": n_side,
                "IC": ic_sub, "DA_pct": da, "MagCorr_ticks": mc_ticks,
                "MFE_ticks": float(mfe), "MAE_ticks": float(mae),
                "avg_move_ticks": avg_move,
            })

    return rows, ic_agg, sd_realized


def main():
    v2_data, v2_src = load_v2()
    v32_data = load_v32()

    print(f"[v2  ] source: {v2_src}")
    print(f"[v2  ] horizons available: {list(v2_data.keys())}")
    print(f"[v32 ] horizons available: {list(v32_data.keys())}")

    all_rows = []
    summary = {}

    for h in HORIZONS:
        if h in v2_data:
            rows, ic_agg, sd_r = compute_bands_for(
                v2_data[h]["pred"], v2_data[h]["realized"], "v2", h
            )
            all_rows.extend(rows)
            summary[("v2", h)] = {"ic_agg": ic_agg, "sd_realized": sd_r,
                                  "n": len(v2_data[h]["pred"])}
        if h in v32_data:
            rows, ic_agg, sd_r = compute_bands_for(
                v32_data[h]["pred"], v32_data[h]["realized"], "v3.2", h
            )
            all_rows.extend(rows)
            summary[("v3.2", h)] = {"ic_agg": ic_agg, "sd_realized": sd_r,
                                    "n": len(v32_data[h]["pred"])}

    df = pd.DataFrame(all_rows)

    csv_path = os.path.join(OUT_DIR, "comparison.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"[OK] wrote {csv_path}")

    # Markdown table
    md_path = os.path.join(OUT_DIR, "comparison.md")
    with open(md_path, "w") as f:
        f.write("# v2 vs v3.2 vs v3.3 per-confidence-band -- OOT 20260223\n\n")
        f.write("**v3.3 EXCLUDED**: fold 0 still training, no prediction NPZ.\n\n")
        f.write("**v2**: 20260223 only (24665 samples, 3 horizons).  \n")
        f.write("**v3.2**: 5-day OOT window 20260223-20260227 (241351 samples, 4 horizons).  \n")
        f.write("MFE/MAE are horizon-h realized-return approximations (true per-trade MFE/MAE not in v2 NPZ).\n\n")
        f.write("## Aggregate (whole-set) numbers\n\n")
        f.write("| model | horizon | IC_agg | sd_realized_ticks | n |\n")
        f.write("|-------|---------|--------|-------------------|----|\n")
        for (m, h), info in sorted(summary.items()):
            f.write(f"| {m} | {h} | {info['ic_agg']:.4f} | {info['sd_realized']:.3f} | {info['n']} |\n")
        f.write("\n## Per-band breakdown\n\n")
        # Simple pipe-table without tabulate dependency
        cols = list(df.columns)
        f.write("| " + " | ".join(cols) + " |\n")
        f.write("|" + "|".join(["---"]*len(cols)) + "|\n")
        for _, row in df.iterrows():
            cells = []
            for c in cols:
                v = row[c]
                if isinstance(v, float):
                    cells.append(f"{v:.3f}" if np.isfinite(v) else "nan")
                else:
                    cells.append(str(v))
            f.write("| " + " | ".join(cells) + " |\n")
    print(f"[OK] wrote {md_path}")

    # ---------- PNG ----------
    # Layout: 3 rows (DA%, MFE, MAE) x 4 cols (horizons 1s/5s/10s/30s)
    # Each subplot: grouped bars over 5 bands; 2 models (v2, v3.2) x 2 sides (long/short)
    fig, axes = plt.subplots(3, 4, figsize=(16, 11), dpi=150)
    metrics = [("DA_pct", "DA %"), ("MFE_ticks", "MFE (ticks)"),
               ("MAE_ticks", "MAE (ticks)")]
    models = ["v2", "v3.2"]
    model_colors = {"v2": "#1f77b4", "v3.2": "#d62728"}

    band_x = np.arange(len(BAND_LABELS))
    bar_w = 0.2

    for col, h in enumerate(HORIZONS):
        for row, (metric, mlabel) in enumerate(metrics):
            ax = axes[row, col]
            offsets = {("v2","long"): -1.5, ("v2","short"): -0.5,
                       ("v3.2","long"): 0.5, ("v3.2","short"): 1.5}
            hatches = {"long": "", "short": "//"}
            for m in models:
                if (m, h) not in summary:
                    continue
                for side in SIDES:
                    sub = df[(df.model==m) & (df.horizon==h) & (df.side==side)]
                    if sub.empty:
                        continue
                    vals = sub.sort_values("band", key=lambda s: s.map({b:i for i,b in enumerate(BAND_LABELS)}))[metric].values
                    if len(vals) != len(BAND_LABELS):
                        continue
                    color = model_colors[m]
                    alpha = 1.0 if side == "long" else 0.55
                    off = offsets[(m,side)]
                    ax.bar(band_x + off*bar_w, vals, bar_w,
                           color=color, alpha=alpha,
                           hatch=hatches[side], edgecolor="black", linewidth=0.3,
                           label=f"{m} {side}" if (row==0 and col==0) else None)
            if row == 0:
                ax.set_title(f"horizon {h}", fontsize=11)
            if col == 0:
                ax.set_ylabel(mlabel, fontsize=10)
            ax.set_xticks(band_x)
            ax.set_xticklabels(BAND_LABELS, fontsize=8)
            ax.axhline(0, color="k", lw=0.5)
            if metric == "DA_pct":
                ax.axhline(50, color="grey", lw=0.5, ls="--")
                ax.set_ylim(0, 100)
            ax.grid(True, alpha=0.25, axis="y")
            if row == 2:
                ax.set_xlabel("top-pct band", fontsize=9)

    # Legend at top
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("v2 vs v3.2 -- Per-Confidence-Band Performance -- OOT 20260223\n"
                 "(v3.3 EXCLUDED: fold 0 still training; no prediction NPZ available)",
                 fontsize=12, y=0.96)
    fig.text(0.01, 0.005,
             "v2: 20260223 only, 3 horizons.   v3.2: 5-day OOT concat 20260223-20260227.\n"
             "MFE/MAE = max(0,realized)/min(0,realized) at horizon h (true per-trade MFE/MAE unavailable for v2).",
             fontsize=7, color="grey")
    plt.tight_layout(rect=[0, 0.02, 1, 0.94])

    png_path = os.path.join(OUT_DIR, "comparison.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[OK] wrote {png_path}")

    # ---------- Console summary ----------
    print("\n=== Quick winners (DA% top-0.1% short side, by horizon) ===")
    for h in HORIZONS:
        sub = df[(df.horizon==h) & (df.band=="0.1%") & (df.side=="short")]
        for _, r in sub.iterrows():
            print(f"  {h} {r.model:5s} short top-0.1%: DA={r.DA_pct:.1f}% MFE={r.MFE_ticks:.2f} MAE={r.MAE_ticks:.2f} avg={r.avg_move_ticks:+.2f}")

    # Save a small json with metadata
    meta = {
        "v2_source": v2_src,
        "v32_source": V32_NPZ,
        "v33_status": "EXCLUDED -- fold 0 training (Ep4 51% as of 07:30 ET); no prediction NPZ written; v33_intra_ckpt_band_eval.py does not exist",
        "v33_aggregate_ic_from_discord_0433ET": {
            "IC_1s": 0.2694, "IC_5s": 0.126, "IC_10s": 0.083, "IC_30s": 0.0446
        },
        "n_v2": int(summary.get(("v2","1s"),{}).get("n",0)),
        "n_v32": int(summary.get(("v3.2","1s"),{}).get("n",0)),
        "bands": BAND_LABELS,
        "horizons": HORIZONS,
    }
    with open(os.path.join(OUT_DIR, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)


if __name__ == "__main__":
    main()
