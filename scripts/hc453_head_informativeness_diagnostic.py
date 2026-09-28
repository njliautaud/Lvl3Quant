#!/usr/bin/env python3
"""
HC #453 R6b — Head Informativeness + Smoothness JOINT Diagnostic.

R6a (the prior diagnostic) ranked heads by lag-1 autocorrelation and found 7 heads
"passing" the 0.30 floor. Sanity-checking those heads on dynamic-range showed they
are SMOOTH-BUT-DEAD: pred_fifo_tp4sl3_net is 100% negative on every OOT date
(p10-p90 of 0.06-0.51, full range 0.94-2.03). The model collapsed those heads to
near-constants due to low gradient signal at FixedWeightMultiHeadLoss default
weight = 0.1. High autocorr without dynamic range = dead-signal artifact, not
stream-coherence.

This diagnostic re-ranks heads on a JOINT informativeness + smoothness gate:
  - smoothness: mean lag-1 autocorr (same as R6a)
  - informativeness: median per-day std AND median per-day p10-p90 spread
  - actionability: fraction of days where sign(x) is meaningfully balanced
                   (i.e. neither all-positive nor all-negative)

A head is "actionable + smooth" only if BOTH gates pass. The R6a 0.30 autocorr
threshold remains a diagnostic checkpoint (per HC #453 R4 amended, principle-first).
"""

import csv
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


INPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/hc453_head_smoothness")
WORKERS = 8

HEADS_TO_TEST = [
    "pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_10s", "pred_log_ret_30s",
    "pred_log_ret_60s", "pred_log_ret_5min",
    "pred_p_up_5s", "pred_p_up_10s", "pred_p_up_30s", "pred_p_up_60s",
    "pred_log_ret_10s_q50", "pred_log_ret_30s_q50", "pred_log_ret_60s_q50",
    "pred_pred_mfe_30s_ticks", "pred_pred_mae_30s_ticks",
    "pred_pred_mfe_60s_ticks", "pred_pred_mae_60s_ticks",
    "pred_p_reversal_15s", "pred_p_reversal_30s", "pred_p_reversal_60s",
    "pred_pred_realized_vol_30s_ticks",
    "pred_fifo_tp4sl3_net", "pred_fifo_tp8sl5_net",
]


def autocorr_at_lag(x, lag):
    if lag <= 0 or x.size <= lag + 1:
        return float("nan")
    a, b = x[:-lag], x[lag:]
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 100:
        return float("nan")
    a, b = a[m], b[m]
    a, b = a - a.mean(), b - b.mean()
    den = np.sqrt((a * a).sum() * (b * b).sum())
    if den <= 0:
        return float("nan")
    return float((a * b).sum() / den)


def per_date_stats(npz_path):
    """Compute per-head stats for one date."""
    try:
        z = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        return [{"date": npz_path.stem, "head": "_ERROR_", "error": str(e)}]
    date_str = npz_path.stem.replace("oot_", "")
    rows = []
    for head in HEADS_TO_TEST:
        if head not in z.files:
            continue
        x = z[head].astype(np.float64)
        x = x[~np.isnan(x)]
        if x.size < 200:
            continue
        ac1 = autocorr_at_lag(x, 1)
        std = float(x.std())
        p10, p90 = float(np.quantile(x, 0.1)), float(np.quantile(x, 0.9))
        spread_iqr80 = p90 - p10
        pos_frac = float((x > 0).mean())
        # balanced = within [0.05, 0.95] -- both signs meaningfully represented
        balanced = 1.0 if (0.05 <= pos_frac <= 0.95) else 0.0
        # normalize spread by full-range to detect "stuck near one value"
        full_range = float(x.max() - x.min())
        spread_ratio = spread_iqr80 / max(full_range, 1e-9)
        rows.append({
            "date": date_str, "head": head, "n": int(x.size),
            "ac_lag1": ac1, "std": std,
            "p10_p90_spread": spread_iqr80, "full_range": full_range,
            "spread_ratio_iqr80_over_range": spread_ratio,
            "pos_frac": pos_frac, "balanced_sign": balanced,
        })
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob("oot_*.npz"))
    if not files:
        print(f"NO INPUT FILES in {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"[hc453-r6b] processing {len(files)} OOT NPZ files")
    all_rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(per_date_stats, p): p for p in files}
        for i, fut in enumerate(as_completed(futs)):
            all_rows.extend(fut.result())
            if (i + 1) % 10 == 0:
                print(f"  done {i+1}/{len(files)}")

    # Aggregate by head
    by_head = {}
    for r in all_rows:
        if r.get("head") == "_ERROR_":
            continue
        by_head.setdefault(r["head"], []).append(r)

    ranking = []
    for h in sorted(by_head.keys()):
        rs = by_head[h]
        ac1s = [r["ac_lag1"] for r in rs if not np.isnan(r["ac_lag1"])]
        stds = [r["std"] for r in rs]
        spreads = [r["p10_p90_spread"] for r in rs]
        spread_ratios = [r["spread_ratio_iqr80_over_range"] for r in rs]
        bal = [r["balanced_sign"] for r in rs]
        pos = [r["pos_frac"] for r in rs]
        # JOINT GATE: smooth AND informative AND balanced
        mean_ac1 = float(np.mean(ac1s)) if ac1s else float("nan")
        med_std = float(np.median(stds))
        med_spread = float(np.median(spreads))
        med_sr = float(np.median(spread_ratios))
        bal_frac = float(np.mean(bal))
        # actionability score = smoothness × informativeness × balanced-fraction
        # informativeness proxy: spread_ratio (live signal uses most of its dynamic range)
        # smoothness: clamp ac1 to [0,1]
        smoothness = max(0.0, min(1.0, mean_ac1))
        informativeness = med_sr  # already in [0,1]
        actionable = smoothness * informativeness * bal_frac
        ranking.append({
            "head": h,
            "n_days": len(rs),
            "mean_ac_lag1": mean_ac1,
            "median_std": med_std,
            "median_p10_p90_spread": med_spread,
            "median_spread_ratio": med_sr,
            "balanced_sign_frac": bal_frac,
            "median_pos_frac": float(np.median(pos)),
            "actionable_score": actionable,
        })
    ranking.sort(key=lambda r: r["actionable_score"], reverse=True)

    out_csv = OUTPUT_DIR / "head_ranking_r6b_joint.csv"
    cols = ["head","n_days","mean_ac_lag1","median_std","median_p10_p90_spread",
            "median_spread_ratio","balanced_sign_frac","median_pos_frac","actionable_score"]
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in ranking:
            w.writerow(r)
    print(f"[hc453-r6b] wrote {out_csv}")

    md = ["# HC #453 R6b — Head JOINT (smoothness AND informativeness AND sign-balance)", "",
          "## Why this exists",
          "",
          "R6a ranked by lag-1 autocorr only. Top 7 heads passed the 0.30 floor — but",
          "spot-checking showed pred_fifo_tp4sl3_net is 100% negative on every OOT date,",
          "with p10-p90 spread of 0.06-0.51 in a full range of 0.94-2.03. The 'smoothness'",
          "was a DEAD-SIGNAL artifact: model collapsed heads to near-constants because",
          "FixedWeightMultiHeadLoss assigned them default weight 0.1. High autocorr",
          "without dynamic range != stream-coherent signal.",
          "",
          "Per HC #453 R4-amended (principle-first, number-second), the right gate is",
          "JOINT: smooth AND informative AND sign-balanced. Score = ac_lag1 × spread_ratio × balanced_frac.",
          "",
          "## Re-ranked top 10 by joint actionability score", "",
          "| head | days | mean_ac1 | med_spread_ratio | balanced% | pos% | ACTIONABLE |",
          "|---|---|---|---|---|---|---|"]
    for r in ranking[:10]:
        md.append(f"| {r['head']} | {r['n_days']} | {r['mean_ac_lag1']:.3f} | "
                  f"{r['median_spread_ratio']:.3f} | {r['balanced_sign_frac']:.2f} | "
                  f"{r['median_pos_frac']:.3f} | **{r['actionable_score']:.4f}** |")
    md += ["", "## R6a 'smooth' heads now reassessed", ""]
    md.append("| head | mean_ac1 | spread_ratio | balanced% | verdict |")
    md.append("|---|---|---|---|---|")
    r6a_smooth = ["pred_fifo_tp4sl3_net","pred_fifo_tp8sl5_net","pred_p_up_60s",
                  "pred_p_reversal_15s","pred_p_reversal_60s","pred_p_reversal_30s",
                  "pred_pred_realized_vol_30s_ticks"]
    by_h = {r["head"]: r for r in ranking}
    for h in r6a_smooth:
        if h not in by_h: continue
        r = by_h[h]
        verdict = ("DEAD (constant)" if r["balanced_sign_frac"] < 0.1
                   else "narrow" if r["median_spread_ratio"] < 0.2
                   else "OK")
        md.append(f"| {h} | {r['mean_ac_lag1']:.3f} | {r['median_spread_ratio']:.3f} | "
                  f"{r['balanced_sign_frac']:.2f} | {verdict} |")
    md += ["", "## Conclusion", "",
           "If the top actionable_score across all 23 heads is still < ~0.05, then NO existing",
           "v3.4.2 head is both stream-coherent AND informative. The HC #454 Phase 2 trainer",
           "patch (new pressure/persistence/MFE-MAE heads with weight=1.0 and a smoothness",
           "regularizer) is then the ONLY path forward — confirms user's reframing instinct.",
           ""]
    (OUTPUT_DIR / "SUMMARY_R6B.md").write_text("\n".join(md))
    print(f"[hc453-r6b] wrote SUMMARY_R6B.md")
    print(f"\nTop 5 by actionable_score:")
    for r in ranking[:5]:
        print(f"  {r['head']:38s} ac1={r['mean_ac_lag1']:.3f} "
              f"sr={r['median_spread_ratio']:.3f} bal={r['balanced_sign_frac']:.2f} "
              f"score={r['actionable_score']:.4f}")


if __name__ == "__main__":
    main()
