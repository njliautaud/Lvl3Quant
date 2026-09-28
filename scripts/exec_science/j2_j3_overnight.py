#!/usr/bin/env python3
"""
Overnight exec science J2 + J3 on v3.3 5-day OOT predictions.

HC #388: "performance is deeper than raw IC there's tons of outputs now that
are very crucial. And at confidence bands only"

J2: Per-head confidence-band ladder
    For each head H with sigma_H available:
        bands = top {50, 25, 10, 5, 1, 0.5, 0.1} % by 1/sigma (confidence)
        for each band:
            n, IC (Spearman), Pearson, DA (sign accuracy), MagCorr (|pred|<->|tgt|),
            signed_long: WR + mean target where pred>0
            signed_short: WR + mean target where pred<0

J3: Head-agreement matrix at top-10% confidence
    For each head pair (H_i, H_j):
        agreement = P(sign(pred_i) == sign(pred_j) | both in top-10% conf)
    Find the most confluent heads.

Outputs:
    j2_confidence_ladder.json   — full ladder per head
    j2_confidence_ladder.csv    — flat table
    j3_head_agreement.csv       — pairwise agreement matrix (top-10%)
    j3_high_agreement_pairs.csv — pairs with >0.65 agreement
    summary.txt                 — bottom-line readable digest
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy import stats


CONF_BANDS = [50.0, 25.0, 10.0, 5.0, 1.0, 0.5, 0.1]  # top-N% by confidence

def safe_corr(x, y):
    if len(x) < 5 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return float("nan"), float("nan")
    try:
        spr = stats.spearmanr(x, y).statistic
    except Exception:
        spr = float("nan")
    try:
        pea = stats.pearsonr(x, y).statistic
    except Exception:
        pea = float("nan")
    return float(spr), float(pea)


def head_band_metrics(pred, target, mask, sigma, band_pct):
    valid = (mask > 0)
    if sigma is not None:
        valid &= np.isfinite(sigma) & (sigma > 0)
    valid &= np.isfinite(pred) & np.isfinite(target)
    if valid.sum() < 50:
        return None
    p = pred[valid]
    t = target[valid]
    s = sigma[valid] if sigma is not None else np.abs(p)  # fallback: |pred|
    # confidence = 1/sigma (high conf = low uncertainty); fallback uses |pred| magnitude
    conf = 1.0 / np.clip(s, 1e-12, None) if sigma is not None else np.abs(p)
    # top-X% by confidence
    cutoff = np.percentile(conf, 100.0 - band_pct)
    in_band = conf >= cutoff
    if in_band.sum() < 30:
        return None
    pp = p[in_band]
    tt = t[in_band]
    ic_sp, ic_pe = safe_corr(pp, tt)
    da = float(np.mean(np.sign(pp) == np.sign(tt)))
    mag_sp, _ = safe_corr(np.abs(pp), np.abs(tt))
    # signed-direction stats
    long_mask = pp > 0
    short_mask = pp < 0
    long_wr = float(np.mean(tt[long_mask] > 0)) if long_mask.sum() >= 5 else float("nan")
    short_wr = float(np.mean(tt[short_mask] < 0)) if short_mask.sum() >= 5 else float("nan")
    long_mean_tgt = float(np.mean(tt[long_mask])) if long_mask.sum() >= 5 else float("nan")
    short_mean_tgt = float(np.mean(tt[short_mask])) if short_mask.sum() >= 5 else float("nan")
    return {
        "n": int(in_band.sum()),
        "band_pct": band_pct,
        "ic_spearman": ic_sp,
        "ic_pearson": ic_pe,
        "directional_acc": da,
        "magnitude_corr_spearman": mag_sp,
        "long_n": int(long_mask.sum()),
        "long_wr": long_wr,
        "long_mean_target": long_mean_tgt,
        "short_n": int(short_mask.sum()),
        "short_wr": short_wr,
        "short_mean_target": short_mean_tgt,
        "pred_mean": float(np.mean(pp)),
        "pred_std": float(np.std(pp)),
        "target_mean": float(np.mean(tt)),
        "target_std": float(np.std(tt)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--npz", default="/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/fold_00_predictions.npz")
    ap.add_argument("--outdir", default="/home/jupiter/Lvl3Quant/output/exec_science_v3_3_overnight")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print(f"[J2/J3] loading {args.npz}", flush=True)
    d = np.load(args.npz, allow_pickle=True)
    keys = set(d.keys())

    # discover heads
    heads = sorted({k[5:] for k in keys if k.startswith("pred_") and f"target_{k[5:]}" in keys})
    print(f"[J2/J3] discovered {len(heads)} heads", flush=True)

    # ---------- J2 ----------
    j2 = {}
    rows = []
    for h in heads:
        pred = np.asarray(d[f"pred_{h}"], dtype=np.float64)
        target = np.asarray(d[f"target_{h}"], dtype=np.float64)
        mask_key = f"mask_{h}"
        mask = np.asarray(d[mask_key], dtype=np.float64) if mask_key in keys else np.ones_like(pred)
        sigma_key = f"sigma_{h}"
        sigma = np.asarray(d[sigma_key], dtype=np.float64) if sigma_key in keys else None
        has_sigma = sigma is not None
        j2[h] = {"has_sigma": has_sigma, "bands": {}}
        # ALL: baseline (100% in mask)
        for band in [100.0] + CONF_BANDS:
            metrics = head_band_metrics(pred, target, mask, sigma if band < 100.0 else None, band)
            if metrics is None:
                continue
            j2[h]["bands"][f"top_{band}pct"] = metrics
            row = {"head": h, "has_sigma": has_sigma}
            row.update(metrics)
            rows.append(row)
        print(f"[J2] {h:32s}  sigma={has_sigma}  bands={len(j2[h]['bands'])}", flush=True)

    (out / "j2_confidence_ladder.json").write_text(json.dumps(j2, indent=2, default=float))

    # CSV
    if rows:
        cols = ["head", "has_sigma", "band_pct", "n",
                "ic_spearman", "ic_pearson", "directional_acc", "magnitude_corr_spearman",
                "long_n", "long_wr", "long_mean_target",
                "short_n", "short_wr", "short_mean_target",
                "pred_mean", "pred_std", "target_mean", "target_std"]
        with open(out / "j2_confidence_ladder.csv", "w") as f:
            f.write(",".join(cols) + "\n")
            for r in rows:
                f.write(",".join(f"{r.get(c, '')}" for c in cols) + "\n")

    # ---------- J3: head-agreement at top-10% conf ----------
    BAND_PCT = 10.0
    sig_heads = []
    sign_arrays = {}
    for h in heads:
        pred = np.asarray(d[f"pred_{h}"], dtype=np.float64)
        mask_key = f"mask_{h}"
        mask = np.asarray(d[mask_key], dtype=np.float64) if mask_key in keys else np.ones_like(pred)
        sigma_key = f"sigma_{h}"
        sigma = np.asarray(d[sigma_key], dtype=np.float64) if sigma_key in keys else None
        valid = (mask > 0) & np.isfinite(pred)
        if sigma is not None:
            valid &= np.isfinite(sigma) & (sigma > 0)
            conf = np.where(valid, 1.0 / np.clip(sigma, 1e-12, None), -np.inf)
        else:
            conf = np.where(valid, np.abs(pred), -np.inf)
        if (conf > -np.inf).sum() < 100:
            continue
        cutoff = np.percentile(conf[conf > -np.inf], 100.0 - BAND_PCT)
        sgn = np.where(conf >= cutoff, np.sign(pred), 0).astype(np.int8)
        if (sgn != 0).sum() < 100:
            continue
        sig_heads.append(h)
        sign_arrays[h] = sgn

    print(f"[J3] head-agreement on {len(sig_heads)} heads at top-{BAND_PCT}% conf", flush=True)
    H = len(sig_heads)
    agree = np.full((H, H), np.nan)
    cooccur_n = np.zeros((H, H), dtype=np.int64)
    for i, h1 in enumerate(sig_heads):
        s1 = sign_arrays[h1]
        a1 = (s1 != 0)
        for j, h2 in enumerate(sig_heads):
            s2 = sign_arrays[h2]
            both = a1 & (s2 != 0)
            n = int(both.sum())
            cooccur_n[i, j] = n
            if n < 50:
                continue
            agree[i, j] = float((s1[both] == s2[both]).mean())

    with open(out / "j3_head_agreement.csv", "w") as f:
        f.write("head," + ",".join(sig_heads) + "\n")
        for i, h1 in enumerate(sig_heads):
            row = [h1]
            for j in range(H):
                v = agree[i, j]
                row.append(f"{v:.4f}" if np.isfinite(v) else "")
            f.write(",".join(row) + "\n")

    pairs = []
    for i in range(H):
        for j in range(i + 1, H):
            v = agree[i, j]
            if np.isfinite(v):
                pairs.append((sig_heads[i], sig_heads[j], v, cooccur_n[i, j]))
    pairs.sort(key=lambda x: x[2], reverse=True)
    with open(out / "j3_high_agreement_pairs.csv", "w") as f:
        f.write("head_a,head_b,agreement,cooccur_n\n")
        for p in pairs:
            f.write(f"{p[0]},{p[1]},{p[2]:.4f},{p[3]}\n")

    # ---------- summary ----------
    lines = []
    lines.append("=" * 90)
    lines.append("EXEC-SCIENCE OVERNIGHT (v3.3 5-day OOT, 241,351 samples)")
    lines.append("=" * 90)
    lines.append("")
    lines.append("J2 — CONFIDENCE-BAND LADDER (top heads by directional acc at top-10% conf)")
    lines.append("-" * 90)
    sortable = []
    for h, info in j2.items():
        if "top_10.0pct" in info["bands"]:
            m = info["bands"]["top_10.0pct"]
            sortable.append((h, m["directional_acc"], m["ic_spearman"], m["n"], info["has_sigma"]))
    sortable.sort(key=lambda x: x[1], reverse=True)
    lines.append(f"{'head':35s} {'DA@10%':>8s} {'IC@10%':>8s} {'n':>8s} {'sigma':>6s}")
    for h, da, ic, n, sg in sortable[:15]:
        lines.append(f"{h:35s} {da:>8.4f} {ic:>8.4f} {n:>8d} {str(sg):>6s}")

    lines.append("")
    lines.append("J2 — TOP HEADS @ TOP-1% CONF (most concentrated edge)")
    lines.append("-" * 90)
    sortable_p1 = []
    for h, info in j2.items():
        if "top_1.0pct" in info["bands"]:
            m = info["bands"]["top_1.0pct"]
            sortable_p1.append((h, m["directional_acc"], m["ic_spearman"], m["short_wr"], m["long_wr"], m["n"]))
    sortable_p1.sort(key=lambda x: x[1], reverse=True)
    lines.append(f"{'head':35s} {'DA@1%':>8s} {'IC@1%':>8s} {'shortWR':>8s} {'longWR':>8s} {'n':>6s}")
    for h, da, ic, swr, lwr, n in sortable_p1[:15]:
        lines.append(f"{h:35s} {da:>8.4f} {ic:>8.4f} {swr:>8.4f} {lwr:>8.4f} {n:>6d}")

    lines.append("")
    lines.append("J3 — TOP HEAD-AGREEMENT PAIRS (sign agreement @ top-10% conf, n>=50)")
    lines.append("-" * 90)
    lines.append(f"{'head_a':35s} {'head_b':35s} {'agree':>8s} {'n':>6s}")
    for p in pairs[:25]:
        lines.append(f"{p[0]:35s} {p[1]:35s} {p[2]:>8.4f} {p[3]:>6d}")

    lines.append("")
    lines.append("=" * 90)
    lines.append("DONE — see j2_confidence_ladder.{json,csv}, j3_head_agreement.csv, j3_high_agreement_pairs.csv")
    lines.append("=" * 90)

    summary_text = "\n".join(lines)
    (out / "summary.txt").write_text(summary_text)
    print(summary_text, flush=True)


if __name__ == "__main__":
    main()
