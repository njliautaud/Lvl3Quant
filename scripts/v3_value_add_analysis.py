#!/usr/bin/env python3
"""
v3 VALUE-ADD CONFIDENCE/TIMING ANALYSIS — HC #288(D).

Generates markdown tables with these metrics broken out by:
  - confidence band: Top0.1% / Top0.5% / Top1% / Top5% / Top10% / Bottom10% / All
  - horizon: 1s / 5s / 10s (where artifacts exist; v2 has 1s + 10s)
  - time-of-day band: open (09:30-10:00) / mid (10:00-15:00) / close (15:00-16:00)
Metrics:
  - DA (Directional Accuracy = % sign(pred)==sign(realized label))
  - MagCorr (Pearson(|pred|, |realized|))
  - Avg realized label ticks
  - Price-after drift at +1s, +5s, +10s, +30s
  - MFE / MAE (favorable / adverse excursion in ticks, from path data)
  - n_predictions in band

Inputs (per fold):
  - mfe_mae_analysis/fold_NN_mfe_mae_{1s,10s}.npz
    - pred_values, label_values, mfe_ticks, mae_ticks,
      path_1s, path_5s, path_10s, path_30s, timestamps_ns

Output: markdown to stdout + JSON to disk for record-keeping.
Use this on v2 NOW for BASELINE; rerun on v3 once v3 fold 0 completes.
"""
import os, sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from datetime import datetime
import pytz

# --- CLI ---
MODEL_DIR = sys.argv[1] if len(sys.argv) > 1 else \
    "/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar"
FOLD     = int(sys.argv[2]) if len(sys.argv) > 2 else 0
TAG      = sys.argv[3] if len(sys.argv) > 3 else "v2_baseline"
OUT_DIR  = Path(MODEL_DIR) / "value_add_analysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Confidence bands as quantile cutoffs on |pred|
BANDS = [
    ("Top0.1%",  0.999, 1.0,  "top"),
    ("Top0.5%",  0.995, 1.0,  "top"),
    ("Top1%",    0.99,  1.0,  "top"),
    ("Top5%",    0.95,  1.0,  "top"),
    ("Top10%",   0.90,  1.0,  "top"),
    ("Bottom10%",0.0,   0.10, "bot"),  # smallest |pred|
    ("All",      0.0,   1.0,  "all"),
]

ET = pytz.timezone("America/New_York")
def tod_bin(ts_ns: int) -> str:
    dt = datetime.fromtimestamp(ts_ns/1e9, tz=pytz.utc).astimezone(ET)
    hm = dt.hour*60 + dt.minute
    if hm < 9*60 + 30:       return "preopen"
    if hm < 10*60:           return "open"      # 09:30-10:00
    if hm < 15*60:           return "mid"       # 10:00-15:00
    if hm < 16*60:           return "close"     # 15:00-16:00
    return "afterhours"


def compute_band_metrics(preds, labels, mfe, mae, paths, ts_ns, band_lo, band_hi, kind):
    """Return dict of metrics for predictions whose |pred| falls in the band."""
    abs_pred = np.abs(preds)
    if kind == "all":
        mask = np.ones(len(preds), dtype=bool)
    elif kind == "top":
        lo_q = np.quantile(abs_pred, band_lo)
        mask = abs_pred >= lo_q
    else:  # "bot" — bottom by |pred|
        hi_q = np.quantile(abs_pred, band_hi)
        mask = abs_pred <= hi_q
    n = int(mask.sum())
    if n == 0:
        return {"n": 0}
    p = preds[mask]; y = labels[mask]; m = mfe[mask]; a = mae[mask]
    # DA: sign(pred)==sign(label), excluding zero-label rows
    nz = y != 0
    if nz.sum() > 0:
        da = float((np.sign(p[nz]) == np.sign(y[nz])).mean()) * 100
    else:
        da = float("nan")
    # MagCorr: Pearson(|pred|, |y|)
    if n >= 2 and np.std(np.abs(p)) > 0 and np.std(np.abs(y)) > 0:
        magcorr = float(np.corrcoef(np.abs(p), np.abs(y))[0, 1])
    else:
        magcorr = float("nan")
    out = {
        "n": n,
        "DA_pct": round(da, 1),
        "MagCorr": round(magcorr, 4),
        "label_mean_ticks": round(float(y.mean()), 3),
        "label_median_ticks": round(float(np.median(y)), 3),
        "MFE_mean": round(float(m.mean()), 3),
        "MFE_p75": round(float(np.percentile(m, 75)), 2),
        "MAE_mean": round(float(a.mean()), 3),
        "MAE_p75": round(float(np.percentile(a, 75)), 2),
        "MFE_MAE_ratio_mean": round(float((m / np.maximum(a, 1.0)).mean()), 2),
    }
    # Signed price drift: multiply by sign(pred) so "good drift" is positive
    # (i.e. price moved in the direction we predicted).
    sgn = np.sign(p)
    sgn[sgn == 0] = 1.0
    for h in ("1s", "5s", "10s", "30s"):
        if paths.get(h) is None: continue
        pp = paths[h][mask] * sgn
        out[f"drift_{h}_mean"] = round(float(pp.mean()), 3)
        out[f"drift_{h}_p25"] = round(float(np.percentile(pp, 25)), 2)
        out[f"drift_{h}_p75"] = round(float(np.percentile(pp, 75)), 2)
    return out


def load_artifacts(model_dir: str, fold: int, horizon: str):
    fp = Path(model_dir) / "mfe_mae_analysis" / f"fold_{fold:02d}_mfe_mae_{horizon}.npz"
    if not fp.exists():
        return None
    d = np.load(fp, allow_pickle=True)
    return {
        "preds":  d["pred_values"].astype(np.float32),
        "labels": d["label_values"].astype(np.float32),
        "mfe":    d["mfe_ticks"].astype(np.float32),
        "mae":    d["mae_ticks"].astype(np.float32),
        "ts_ns":  d["timestamps_ns"].astype(np.int64),
        "paths":  {h: d[f"path_{h}"].astype(np.float32) if f"path_{h}" in d.files else None
                   for h in ("1s","5s","10s","30s")},
    }


def render_band_table(bands_data, horizon, tag):
    """Render markdown table for a single horizon, rows=bands."""
    cols = ["Band","n","DA%","MagCorr","Lbl_mean","MFE_mean","MFE_p75","MAE_mean","MAE_p75","MFE/MAE",
            "drift1s","drift5s","drift10s","drift30s"]
    lines = [f"### {tag} — fold 0 — horizon {horizon} — confidence bands",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"]*len(cols)) + "|"]
    for band_name, m in bands_data:
        if m.get("n", 0) == 0:
            lines.append(f"| {band_name} | 0 | — | — | — | — | — | — | — | — | — | — | — | — |")
            continue
        lines.append("| " + " | ".join([
            band_name,
            str(m["n"]),
            f"{m.get('DA_pct','—')}",
            f"{m.get('MagCorr','—')}",
            f"{m.get('label_mean_ticks','—')}",
            f"{m.get('MFE_mean','—')}",
            f"{m.get('MFE_p75','—')}",
            f"{m.get('MAE_mean','—')}",
            f"{m.get('MAE_p75','—')}",
            f"{m.get('MFE_MAE_ratio_mean','—')}",
            f"{m.get('drift_1s_mean','—')}",
            f"{m.get('drift_5s_mean','—')}",
            f"{m.get('drift_10s_mean','—')}",
            f"{m.get('drift_30s_mean','—')}",
        ]) + " |")
    return "\n".join(lines)


def render_tod_table(tod_data, tag):
    """Render markdown table broken out by ToD x band for the strongest bands only."""
    cols = ["ToD","Band","n","DA%","Lbl_mean","drift1s","drift5s","drift10s","MFE_mean","MAE_mean"]
    lines = [f"### {tag} — fold 0 — ToD × Confidence Band (Top0.5%, Top1%, Top10% only — 1s horizon)",
             "| " + " | ".join(cols) + " |",
             "|" + "|".join(["---"]*len(cols)) + "|"]
    for (tod, band), m in tod_data:
        if m.get("n", 0) == 0:
            lines.append(f"| {tod} | {band} | 0 | — | — | — | — | — | — | — |")
            continue
        lines.append("| " + " | ".join([
            tod, band, str(m["n"]),
            f"{m.get('DA_pct','—')}",
            f"{m.get('label_mean_ticks','—')}",
            f"{m.get('drift_1s_mean','—')}",
            f"{m.get('drift_5s_mean','—')}",
            f"{m.get('drift_10s_mean','—')}",
            f"{m.get('MFE_mean','—')}",
            f"{m.get('MAE_mean','—')}",
        ]) + " |")
    return "\n".join(lines)


def main():
    print(f"# v3 VALUE-ADD ANALYSIS — {TAG} — fold {FOLD}")
    print(f"# Source: {MODEL_DIR}/mfe_mae_analysis/fold_{FOLD:02d}_mfe_mae_*.npz")
    print(f"# Run timestamp: {datetime.now().isoformat()}")
    all_out = {}
    for horizon in ("1s", "5s", "10s"):
        art = load_artifacts(MODEL_DIR, FOLD, horizon)
        if art is None:
            print(f"\n## horizon {horizon}: SKIPPED (no artifact)")
            continue
        # Per-band on ALL data
        band_metrics = []
        for name, lo, hi, kind in BANDS:
            m = compute_band_metrics(
                art["preds"], art["labels"], art["mfe"], art["mae"],
                art["paths"], art["ts_ns"], lo, hi, kind
            )
            band_metrics.append((name, m))
        print()
        print(render_band_table(band_metrics, horizon, TAG))
        all_out[f"horizon_{horizon}"] = {n: m for n, m in band_metrics}
        if horizon != "1s":
            continue
        # ToD x band breakout — only for 1s horizon
        ts = art["ts_ns"]
        tod_labels = np.array([tod_bin(t) for t in ts])
        tod_table = []
        for tod in ("open", "mid", "close"):
            tod_mask_full = tod_labels == tod
            if tod_mask_full.sum() == 0:
                for band_name in ("Top0.5%", "Top1%", "Top10%"):
                    tod_table.append(((tod, band_name), {"n": 0}))
                continue
            preds_t = art["preds"][tod_mask_full]
            labels_t = art["labels"][tod_mask_full]
            mfe_t = art["mfe"][tod_mask_full]
            mae_t = art["mae"][tod_mask_full]
            ts_t  = ts[tod_mask_full]
            paths_t = {k: (v[tod_mask_full] if v is not None else None) for k, v in art["paths"].items()}
            for band_name, lo, hi, kind in BANDS:
                if band_name not in ("Top0.5%", "Top1%", "Top10%"):
                    continue
                m = compute_band_metrics(preds_t, labels_t, mfe_t, mae_t, paths_t, ts_t, lo, hi, kind)
                tod_table.append(((tod, band_name), m))
        print()
        print(render_tod_table(tod_table, TAG))
        all_out["tod_x_band_1s"] = {f"{tod}_{band}": m for (tod, band), m in tod_table}
    # Persist
    out_json = OUT_DIR / f"fold_{FOLD:02d}_value_add_{TAG}.json"
    with open(out_json, "w") as f:
        json.dump(all_out, f, indent=2)
    print(f"\n# Saved JSON: {out_json}")


if __name__ == "__main__":
    main()
