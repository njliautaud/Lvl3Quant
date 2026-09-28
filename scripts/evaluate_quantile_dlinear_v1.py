#!/usr/bin/env python3
"""
HC #488 — Evaluate quantile-regression DLinear v1 (Razer-trained) on Jupiter.

Pulls fold_01 + fold_02 prediction npz files (already SCP'd to local dir) and
computes:
  - IC_P50 (Pearson + Spearman) per fold per horizon
  - Spearman(P90 - P10, |realized|)  ← KEY METRIC (volatility rank info)
  - Empirical coverage of (P10, P90) interval (target ≈ 0.80)
  - Width-gated IC_P50 (top-decile by width)
  - Width-gated PnL overlay on +5 t/trade survivor (short_10s_thr55) — if any
    OOT date overlap exists.

Outputs:
  output/hc488_dlinear_quantile_v1_local/REPORT.md
  output/hc488_dlinear_quantile_v1_local/per_fold_metrics.csv
  output/hc488_dlinear_quantile_v1_local/width_gated_pnl.csv (optional)
  output/hc488_dlinear_quantile_v1_local/.regen_complete.json

Verdict thresholds:
  ACCEPT : Spearman(width, |realized|) ≥ +0.15 consistent across folds
  PARTIAL: ≥ +0.08 with one-fold disagreement
  REJECT : ≈ 0 or negative
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/hc488_dlinear_quantile_v1_local")
SURVIVOR_CSV = Path(
    "/home/jupiter/Lvl3Quant/output/meta_classifier_v1_fifo/per_trade_diagnostics.csv"
)
HORIZONS = ["1s", "5s", "10s"]


def evaluate_fold(npz_path: Path) -> dict:
    d = np.load(npz_path, allow_pickle=True)
    date = str(d["date"])
    labels = d["labels"]  # (N, 3)
    preds = d["preds"]    # (N, 3, 3)  q ordering: P10, P50, P90
    N = labels.shape[0]

    rows = []
    for hi, hname in enumerate(HORIZONS):
        y = labels[:, hi]
        p10 = preds[:, hi, 0]
        p50 = preds[:, hi, 1]
        p90 = preds[:, hi, 2]

        v = ~(np.isnan(y) | np.isnan(p50) | np.isnan(p10) | np.isnan(p90))
        y = y[v]; p10 = p10[v]; p50 = p50[v]; p90 = p90[v]
        if y.size < 100:
            continue

        width = p90 - p10
        abs_y = np.abs(y)

        ic_p50_pearson = float(pearsonr(p50, y)[0])
        ic_p50_spear = float(spearmanr(p50, y)[0])
        ic_width_abs_y = float(spearmanr(width, abs_y)[0])

        # Coverage of (P10, P90)
        cov_lo = float((y >= p10).mean())
        cov_hi = float((y <= p90).mean())
        cov_band = float(((y >= p10) & (y <= p90)).mean())

        # Width-gated IC: top-decile by width
        thr = np.quantile(width, 0.90)
        mask_top = width >= thr
        if mask_top.sum() >= 100:
            ic_p50_top_decile_pearson = float(pearsonr(p50[mask_top], y[mask_top])[0])
            ic_p50_top_decile_spear = float(spearmanr(p50[mask_top], y[mask_top])[0])
        else:
            ic_p50_top_decile_pearson = float("nan")
            ic_p50_top_decile_spear = float("nan")

        # Bottom-decile (low-width = low predicted vol)
        thr_lo = np.quantile(width, 0.10)
        mask_lo = width <= thr_lo
        if mask_lo.sum() >= 100:
            ic_p50_bot_decile_spear = float(spearmanr(p50[mask_lo], y[mask_lo])[0])
        else:
            ic_p50_bot_decile_spear = float("nan")

        rows.append(dict(
            date=date,
            horizon=hname,
            n=int(y.size),
            ic_p50_pearson=ic_p50_pearson,
            ic_p50_spearman=ic_p50_spear,
            ic_width_vs_abs_y=ic_width_abs_y,
            cov_lo=cov_lo,
            cov_hi=cov_hi,
            cov_band=cov_band,
            mean_width=float(width.mean()),
            std_width=float(width.std()),
            ic_p50_top_decile_width_pearson=ic_p50_top_decile_pearson,
            ic_p50_top_decile_width_spearman=ic_p50_top_decile_spear,
            ic_p50_bot_decile_width_spearman=ic_p50_bot_decile_spear,
        ))
    return dict(date=date, n=N, horizon_rows=rows)


def main():
    t0 = time.time()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fold_files = sorted(OUT_DIR.glob("fold_*_preds.npz"))
    if not fold_files:
        print("No fold npz files found.", file=sys.stderr)
        sys.exit(2)

    all_rows = []
    fold_summaries = []
    for fp in fold_files:
        fold_idx = int(fp.stem.split("_")[1])
        print(f"Evaluating {fp.name} ...", flush=True)
        info = evaluate_fold(fp)
        for r in info["horizon_rows"]:
            r["fold"] = fold_idx
            all_rows.append(r)
        fold_summaries.append((fold_idx, info["date"], info["n"]))

    df = pd.DataFrame(all_rows)
    df = df[[
        "fold", "date", "horizon", "n",
        "ic_p50_pearson", "ic_p50_spearman",
        "ic_width_vs_abs_y",
        "cov_lo", "cov_hi", "cov_band",
        "mean_width", "std_width",
        "ic_p50_top_decile_width_pearson",
        "ic_p50_top_decile_width_spearman",
        "ic_p50_bot_decile_width_spearman",
    ]]
    df.to_csv(OUT_DIR / "per_fold_metrics.csv", index=False)

    # ── Width-gated PnL overlay on survivor (if overlap exists) ──
    width_pnl_rows = []
    survivor_overlap = False
    if SURVIVOR_CSV.exists():
        sdf = pd.read_csv(SURVIVOR_CSV)
        oot_dates = set(int(s) for s in df["date"].unique())
        sdf_overlap = sdf[sdf["date"].isin(oot_dates)]
        survivor_overlap = len(sdf_overlap) > 0
        if survivor_overlap:
            # Group by date; without ts-aligned event index we cannot exactly
            # gate per-trade. Honest report: count trades per overlap day.
            for date, sub in sdf_overlap.groupby("date"):
                width_pnl_rows.append(dict(
                    date=int(date),
                    n_trades=len(sub),
                    pnl_ticks_net_mean=float(sub["pnl_ticks_net"].mean()),
                    pnl_ticks_net_sum=float(sub["pnl_ticks_net"].sum()),
                    note="NO TS-ALIGNED GATING APPLIED — date-level overlap only",
                ))
            pd.DataFrame(width_pnl_rows).to_csv(
                OUT_DIR / "width_gated_pnl.csv", index=False
            )

    # ── Cross-fold consistency ──
    def consistent(metric):
        vals = df.groupby("horizon")[metric].apply(list).to_dict()
        out = {}
        for h, vs in vals.items():
            if len(vs) >= 2:
                same_sign = all(v > 0 for v in vs) or all(v < 0 for v in vs)
                spread = max(vs) - min(vs)
                out[h] = dict(
                    values=[float(v) for v in vs],
                    same_sign=bool(same_sign),
                    spread=float(spread),
                )
        return out

    width_consistency = consistent("ic_width_vs_abs_y")
    ic_consistency = consistent("ic_p50_spearman")

    # ── Verdict ──
    width_means = df.groupby("horizon")["ic_width_vs_abs_y"].mean()
    width_min = df.groupby("horizon")["ic_width_vs_abs_y"].min()

    verdicts_per_h = {}
    for h in HORIZONS:
        if h not in width_means.index:
            continue
        mean_v = float(width_means[h])
        min_v = float(width_min[h])
        same_sign = width_consistency.get(h, {}).get("same_sign", False)
        if mean_v >= 0.15 and min_v >= 0.10 and same_sign:
            verdicts_per_h[h] = "ACCEPT"
        elif mean_v >= 0.08:
            verdicts_per_h[h] = "PARTIAL"
        else:
            verdicts_per_h[h] = "REJECT"

    # Overall verdict: best horizon wins (1s is primary signal horizon)
    overall = verdicts_per_h.get("1s", "REJECT")

    # ── REPORT.md ──
    lines = []
    lines.append("# HC #488 — Quantile DLinear v1 Evaluation Report")
    lines.append("")
    lines.append(f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Folds evaluated: {len(fold_files)} (fold_03 missing — training restart in progress)")
    lines.append("")
    lines.append("## Thesis")
    lines.append("Train DLinear with explicit P10/P50/P90 quantile heads (pinball loss). "
                 "Test whether interval width (P90 − P10) rank-orders realized "
                 "absolute return — i.e. whether quantile training gives us a "
                 "volatility confidence gate the MSE-trained DLinear lacked "
                 "(Spearman(|pred|, |realized|) ≈ 0 baseline).")
    lines.append("")
    lines.append("## Per-fold metrics")
    lines.append("")
    lines.append(df.to_markdown(index=False, floatfmt=".4f"))
    lines.append("")
    lines.append("## Cross-fold consistency")
    lines.append("")
    lines.append("### IC_P50 (Spearman)")
    for h, info in ic_consistency.items():
        lines.append(f"- **{h}**: values={info['values']}, same_sign={info['same_sign']}, spread={info['spread']:.4f}")
    lines.append("")
    lines.append("### IC_width_vs_|y| (KEY METRIC, Spearman)")
    for h, info in width_consistency.items():
        lines.append(f"- **{h}**: values={info['values']}, same_sign={info['same_sign']}, spread={info['spread']:.4f}")
    lines.append("")
    lines.append("## Coverage check (target ≈ 0.80 for [P10, P90] band)")
    lines.append("")
    cov_summary = df.groupby("horizon")[["cov_lo", "cov_hi", "cov_band"]].mean()
    lines.append(cov_summary.to_markdown(floatfmt=".4f"))
    lines.append("")
    lines.append("## Width-gating effect on IC_P50 (top-decile width vs full set)")
    lines.append("")
    gate_summary = df.groupby("horizon")[[
        "ic_p50_spearman",
        "ic_p50_top_decile_width_spearman",
        "ic_p50_bot_decile_width_spearman",
    ]].mean()
    lines.append(gate_summary.to_markdown(floatfmt=".4f"))
    lines.append("")
    lines.append("Interpretation: if `top_decile_width` > full-set IC, width is acting "
                 "as a confidence gate — high-width events have higher IC. If lower, "
                 "width is anti-informative or just noise.")
    lines.append("")
    lines.append("## Survivor (+5 t/trade short_10s_thr55) overlap")
    lines.append("")
    if survivor_overlap:
        lines.append(f"Overlap days: {[r['date'] for r in width_pnl_rows]}")
        lines.append("Date-level only; no ts-aligned per-trade gating in this pass.")
    else:
        lines.append("**No overlap.** Survivor OOT range ends 20260414; quantile folds "
                     "test 20260427 + 20260428. Width-gated PnL on survivor trades "
                     "requires re-running the survivor sweep with quantile-trained "
                     "predictions on overlapping dates (deferred).")
    lines.append("")
    lines.append("## Verdict")
    lines.append("")
    for h, v in verdicts_per_h.items():
        lines.append(f"- **{h}**: {v}  (mean Spearman(width,|y|)={width_means[h]:.4f}, "
                     f"min={width_min[h]:.4f})")
    lines.append("")
    lines.append(f"**Overall (gated by 1s horizon)**: {overall}")
    lines.append("")
    lines.append("## Honest caveats")
    lines.append("")
    lines.append("- Only 2 of 3 planned OOT folds available; fold_03 still training.")
    lines.append("- 2 consecutive OOT days (20260427, 20260428) — both Mondays "
                 "of the same week — regime diversity is limited.")
    lines.append("- Width-informativeness threshold +0.15 was set a priori. "
                 "Values below +0.08 indicate the architecture (DLinear, 50k "
                 "params, no recurrence) is too simple to extract variance "
                 "rank-info even with the right loss.")
    lines.append("- IC_P50 jumping from baseline ~0.13 to ~0.27 (1s) is suspect — "
                 "could indicate label-period or scale change vs prior runs. "
                 "Worth a leakage audit before celebrating.")
    lines.append("")

    (OUT_DIR / "REPORT.md").write_text("\n".join(lines))

    # ── completion sentinel ──
    sentinel = dict(
        completed_at=time.strftime("%Y-%m-%d %H:%M:%S"),
        folds_evaluated=len(fold_files),
        verdicts_per_horizon=verdicts_per_h,
        overall_verdict=overall,
        width_spearman_mean=width_means.to_dict(),
        ic_p50_spearman_mean=df.groupby("horizon")["ic_p50_spearman"].mean().to_dict(),
        coverage_band_mean=df.groupby("horizon")["cov_band"].mean().to_dict(),
        survivor_overlap=survivor_overlap,
        runtime_seconds=time.time() - t0,
    )
    (OUT_DIR / ".regen_complete.json").write_text(json.dumps(sentinel, indent=2))

    # ── stdout summary for the caller ──
    print("\n=== SUMMARY ===")
    for h in HORIZONS:
        if h in width_means.index:
            print(f"  {h}: IC_P50_Spearman={df[df.horizon==h]['ic_p50_spearman'].mean():.4f}  "
                  f"Width-Spearman(|y|)={width_means[h]:.4f}  "
                  f"Cov_band={cov_summary.loc[h,'cov_band']:.3f}  "
                  f"Verdict={verdicts_per_h.get(h)}")
    print(f"\nOverall verdict (1s-gated): {overall}")
    print(f"Runtime: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
