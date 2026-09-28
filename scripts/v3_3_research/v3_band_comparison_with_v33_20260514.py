"""
v2 vs v3.2 vs v3.3 per-confidence-band comparison on common OOT 20260223.
Per HC #343 + HC #348. Companion to v3_band_comparison_20260514.py (this
file ADDS v3.3 from the new v33_intra_ckpt_band_eval.py predictions NPZ).

Per malware-guard (HC #307D): NEW analysis script ONLY, no trainer code
touched, prior comparison script not modified.

NOTES / LIMITATIONS:
  - v3.3 NPZ is the intra_ckpt at Epoch 3 / batch 11500 (~51% trained).
    Final fold weights expected ~14-15 ET. This comparison is honest about
    that gap in the caption.
  - v3.2 NPZ covers 5 OOT days concatenated (20260223-20260227); we report
    over the full window for v3.2 because the per-day boundary index is not
    in the saved NPZ. Asymmetric vs v2 (which is 20260223 only) — same as
    the prior comparison agent's caveat. v3.3 here is 20260223 only.
  - MagCorr per HC #348: pearson(|pred|,|real|) * sd(realized) -> ticks.

Outputs:
  - output/v3_3_vs_v3_2_vs_v2_band_comparison_20260514/comparison_with_v33.csv
  - output/v3_3_vs_v3_2_vs_v2_band_comparison_20260514/comparison_with_v33.md
  - output/v3_3_vs_v3_2_vs_v2_band_comparison_20260514/comparison_v2_with_v33.png
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import spearmanr, pearsonr


OUT_DIR = "/home/jupiter/Lvl3Quant/output/v3_3_vs_v3_2_vs_v2_band_comparison_20260514"
os.makedirs(OUT_DIR, exist_ok=True)

V2_NPZ = "/tmp/v2_fold00_20260223.npz"
V32_NPZ = "/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"
V33_NPZ = "/home/jupiter/Lvl3Quant/output/v3_3_oot_20260223/predictions.npz"

BANDS = [0.001, 0.005, 0.01, 0.05, 0.10]
BAND_LABELS = ["0.1%", "0.5%", "1%", "5%", "10%"]
HORIZONS = ["1s", "5s", "10s", "30s"]
SIDES = ["long", "short"]


def load_v2():
    d = np.load(V2_NPZ)
    preds = d["predictions"]
    labels = d["labels"]
    horizons = list(d["horizons"])
    out = {}
    for i, h in enumerate(horizons):
        # v2 labels are stored as bytes/str sometimes
        hkey = h.decode() if isinstance(h, bytes) else str(h)
        out[hkey] = {
            "pred": preds[:, i].astype(np.float64),
            "realized": labels[:, i].astype(np.float64),
        }
    return out, "v2 fold_00 20260223"


def _load_v3_style(npz_path):
    d = np.load(npz_path)
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
        valid = m & np.isfinite(p) & np.isfinite(t)
        out[h] = {"pred": p[valid], "realized": t[valid]}
    return out


def load_v32():
    return _load_v3_style(V32_NPZ)


def load_v33():
    return _load_v3_style(V33_NPZ)


def compute_bands_for(pred, realized, model, horizon):
    rows = []
    N = len(pred)
    if N > 1 and np.std(pred) > 0 and np.std(realized) > 0:
        ic_agg, _ = spearmanr(pred, realized)
    else:
        ic_agg = np.nan
    sd_realized = float(np.std(realized))

    for band_frac, band_lbl in zip(BANDS, BAND_LABELS):
        K = max(1, int(round(N * band_frac)))
        abs_pred = np.abs(pred)
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
            ic_sub = spearmanr(sp, sr)[0] if (np.std(sp) > 0 and np.std(sr) > 0) else np.nan
            da = float(np.mean(np.sign(sp) == np.sign(sr)) * 100.0)
            if np.std(np.abs(sp)) > 0 and np.std(np.abs(sr)) > 0:
                mc_raw = pearsonr(np.abs(sp), np.abs(sr))[0]
            else:
                mc_raw = np.nan
            mc_ticks = mc_raw * float(np.std(sr)) if np.isfinite(mc_raw) else np.nan
            if side == "long":
                mfe = float(np.maximum(0.0, sr).mean())
                mae = float(np.minimum(0.0, sr).mean())
                avg_move = float(sr.mean())
            else:
                mfe = float(np.maximum(0.0, -sr).mean())
                mae = float(np.minimum(0.0, -sr).mean())
                avg_move = float((-sr).mean())
            rows.append({
                "model": model, "horizon": horizon, "band": band_lbl,
                "side": side, "n": n_side,
                "IC": ic_sub, "DA_pct": da, "MagCorr_ticks": mc_ticks,
                "MFE_ticks": mfe, "MAE_ticks": mae,
                "avg_move_ticks": avg_move,
            })
    return rows, ic_agg, sd_realized


def main():
    v2_data, v2_src = load_v2()
    v32_data = load_v32()
    v33_data = load_v33()

    print(f"[v2  ] horizons: {list(v2_data.keys())}")
    print(f"[v32 ] horizons: {list(v32_data.keys())}")
    print(f"[v33 ] horizons: {list(v33_data.keys())}")

    all_rows = []
    summary = {}

    for h in HORIZONS:
        for label, data in [("v2", v2_data), ("v3.2", v32_data), ("v3.3", v33_data)]:
            if h in data:
                rows, ic_agg, sd_r = compute_bands_for(
                    data[h]["pred"], data[h]["realized"], label, h
                )
                all_rows.extend(rows)
                summary[(label, h)] = {
                    "ic_agg": ic_agg, "sd_realized": sd_r,
                    "n": len(data[h]["pred"])
                }

    df = pd.DataFrame(all_rows)
    csv_path = os.path.join(OUT_DIR, "comparison_with_v33.csv")
    df.to_csv(csv_path, index=False, float_format="%.4f")
    print(f"[OK] {csv_path}")

    # ---------- Markdown ----------
    md_path = os.path.join(OUT_DIR, "comparison_with_v33.md")
    with open(md_path, "w") as f:
        f.write("# v2 vs v3.2 vs v3.3 per-confidence-band -- OOT 20260223\n\n")
        f.write("**v3.3**: intra_ckpt fold 0 Epoch 3 batch 11500 (~51% trained). NOT final fold weights.\n")
        f.write("**v2**: 20260223 only (3 horizons).  **v3.2**: 5-day OOT 20260223-20260227 (4 horizons).  \n")
        f.write("**v3.3**: 20260223 only (4 horizons).\n\n")
        f.write("## Aggregate (whole-set) numbers\n\n")
        f.write("| model | horizon | IC_agg | sd_realized_ticks | n |\n")
        f.write("|-------|---------|--------|-------------------|----|\n")
        for (m, h), info in sorted(summary.items()):
            f.write(f"| {m} | {h} | {info['ic_agg']:.4f} | {info['sd_realized']:.3f} | {info['n']} |\n")
        f.write("\n## Per-band breakdown\n\n")
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
    print(f"[OK] {md_path}")

    # ---------- PNG: 3 rows (DA%, MFE, MAE) x 4 cols (horizons) ----------
    # Within each subplot: 3 models x 2 sides (long/short) grouped over 5 bands.
    fig, axes = plt.subplots(3, 4, figsize=(18, 12), dpi=150)
    metrics = [("DA_pct", "DA %"), ("MFE_ticks", "MFE (ticks)"),
               ("MAE_ticks", "MAE (ticks)")]
    models = ["v2", "v3.2", "v3.3"]
    model_colors = {"v2": "#1f77b4", "v3.2": "#d62728", "v3.3": "#2ca02c"}

    band_x = np.arange(len(BAND_LABELS))
    bar_w = 0.14
    # 6 bars per band slot: (v2 long, v2 short, v3.2 long, v3.2 short, v3.3 long, v3.3 short)
    offsets = {
        ("v2","long"):   -2.5, ("v2","short"):   -1.5,
        ("v3.2","long"): -0.5, ("v3.2","short"):  0.5,
        ("v3.3","long"):  1.5, ("v3.3","short"):  2.5,
    }
    hatches = {"long": "", "short": "//"}

    for col, h in enumerate(HORIZONS):
        for row, (metric, mlabel) in enumerate(metrics):
            ax = axes[row, col]
            for m in models:
                if (m, h) not in summary:
                    continue
                for side in SIDES:
                    sub = df[(df.model==m) & (df.horizon==h) & (df.side==side)]
                    if sub.empty:
                        continue
                    vals = sub.sort_values(
                        "band",
                        key=lambda s: s.map({b:i for i,b in enumerate(BAND_LABELS)})
                    )[metric].values
                    if len(vals) != len(BAND_LABELS):
                        continue
                    color = model_colors[m]
                    alpha = 1.0 if side == "long" else 0.55
                    off = offsets[(m, side)]
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

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=6, fontsize=10,
               bbox_to_anchor=(0.5, 0.985))
    fig.suptitle("v2 vs v3.2 vs v3.3 -- Per-Band Comparison -- OOT 20260223\n"
                 "(v3.3 = intra_ckpt Ep3 b11500 ~51% trained; NOT final fold weights -- re-run at fold end ~14-15 ET)",
                 fontsize=12, y=0.965)
    fig.text(0.01, 0.005,
             "v2: 20260223 only, 3 horizons.  v3.2: 5-day OOT concat 20260223-20260227.  v3.3: 20260223 only.\n"
             "MFE/MAE = max(0,realized)/min(0,realized) at horizon h (true per-trade MFE/MAE not present in v2 NPZ).",
             fontsize=7, color="grey")
    plt.tight_layout(rect=[0, 0.02, 1, 0.93])

    png_path = os.path.join(OUT_DIR, "comparison_v2_with_v33.png")
    plt.savefig(png_path, dpi=150, bbox_inches="tight")
    print(f"[OK] {png_path}")

    # ---------- Console quick winners ----------
    print("\n=== DA% Top-0.1% short side, by horizon and model ===")
    for h in HORIZONS:
        sub = df[(df.horizon==h) & (df.band=="0.1%") & (df.side=="short")]
        for _, r in sub.iterrows():
            print(f"  {h:3s} {r.model:5s} short top-0.1%: "
                  f"DA={r.DA_pct:.1f}% MFE={r.MFE_ticks:.2f} MAE={r.MAE_ticks:.2f} "
                  f"avg={r.avg_move_ticks:+.2f} MagCorr_ticks={r.MagCorr_ticks:.3f} n={r.n}")

    print("\n=== Per-horizon winner at Top-0.1% (DA% short) ===")
    for h in HORIZONS:
        sub = df[(df.horizon==h) & (df.band=="0.1%") & (df.side=="short") & df.DA_pct.notna()]
        if not sub.empty:
            win = sub.loc[sub.DA_pct.idxmax()]
            print(f"  {h}: {win.model} (DA={win.DA_pct:.1f}%)")

    meta = {
        "v2_source": v2_src,
        "v32_source": V32_NPZ,
        "v33_source": V33_NPZ,
        "v33_status": "intra_ckpt Ep3 b11500 (~51% trained) on Jupiter CPU",
        "models_included": models,
        "bands": BAND_LABELS,
        "horizons": HORIZONS,
        "summary": {f"{m}_{h}": v for (m, h), v in summary.items()},
    }
    with open(os.path.join(OUT_DIR, "meta_with_v33.json"), "w") as f:
        json.dump(meta, f, indent=2, default=str)
    print(f"[OK] {os.path.join(OUT_DIR, 'meta_with_v33.json')}")


if __name__ == "__main__":
    main()
