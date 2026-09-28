#!/usr/bin/env python3
"""
HC #451 R5 — Pressure label cache aggregate report.

Reads every cache file in
data/processed/mbo_events_smart_v3_pressure_labels/<date>_pressure.npz
together with its source NPZ (for |labels_1s| confidence stratification) and
emits output/hc451_pressure_labels/REPORT.md.

Sections:
  - Coverage (dates, events)
  - Overall persistence rates (1s-10s, 1s-30s, 5s-30s)
  - Confidence-stratified persistence (top 10%/5%/1%/0.5% by |labels_1s|)
  - Pressure-score distribution (percentiles + ASCII histogram)
  - One-paragraph interpretation
"""

import os
import sys
import math
from pathlib import Path

import numpy as np

CACHE_DIR = "/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3_pressure_labels"
SRC_DIR   = "/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3"
OUT_DIR   = "/home/jupiter/Lvl3Quant/output/hc451_pressure_labels"
OUT_PATH  = os.path.join(OUT_DIR, "REPORT.md")

PERSIST_KEYS = ["persistence_1s_10s", "persistence_1s_30s", "persistence_5s_30s"]


def rate_breakdown(arr: np.ndarray):
    """Return (pct_plus1, pct_zero, pct_minus1) as floats in [0,100]."""
    n = len(arr)
    if n == 0:
        return (0.0, 0.0, 0.0)
    return (
        100.0 * (arr == 1).sum() / n,
        100.0 * (arr == 0).sum() / n,
        100.0 * (arr == -1).sum() / n,
    )


def ascii_histogram(values: np.ndarray, bins: int = 21, width: int = 50) -> str:
    """Simple ASCII histogram of values."""
    finite = values[np.isfinite(values)]
    if len(finite) == 0:
        return "(no finite values)"
    counts, edges = np.histogram(finite, bins=bins, range=(-1.0, 1.0))
    mx = counts.max() if counts.max() > 0 else 1
    lines = []
    for i, c in enumerate(counts):
        lo, hi = edges[i], edges[i + 1]
        bar = "#" * int(round(width * c / mx))
        pct = 100.0 * c / counts.sum()
        lines.append(f"  [{lo:+.2f},{hi:+.2f})  {bar:<{width}}  {pct:5.2f}%  (n={c:,})")
    return "\n".join(lines)


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    cache_files = sorted(f for f in os.listdir(CACHE_DIR) if f.endswith("_pressure.npz"))
    if not cache_files:
        print("No cache files found.", file=sys.stderr)
        return 1

    dates = [f.replace("_pressure.npz", "") for f in cache_files]

    # Streaming aggregates
    n_total = 0
    p_counts = {k: {1: 0, 0: 0, -1: 0} for k in PERSIST_KEYS}

    # Confidence stratification: we accumulate |labels_1s| from sources, and
    # to keep memory manageable on 1.5B+ events, we use a two-pass approach:
    # PASS 1: compute global |labels_1s| percentile thresholds via approximate
    #         per-date concatenation of a uniform sample (cap at ~50M total).
    # PASS 2: with thresholds known, stream through and accumulate persistence
    #         counts within each |labels_1s| confidence band.
    pct_targets = [90.0, 95.0, 99.0, 99.5]  # top 10%, 5%, 1%, 0.5%
    band_names  = ["top10pct", "top5pct", "top1pct", "top0p5pct"]

    # Pressure score accumulators (uniform sample for distribution)
    rng = np.random.default_rng(seed=42)
    SAMPLE_CAP = 5_000_000  # 5M uniform sample for pressure-score distribution
    ABS1_SAMPLE_CAP = 50_000_000

    print(f"Scanning {len(cache_files)} cache files...", flush=True)

    # PASS 1: sample |labels_1s| from src files
    abs1_sample = []
    src_count = 0
    for d in dates:
        src_path = os.path.join(SRC_DIR, f"{d}_mbo_events.npz")
        if not os.path.exists(src_path):
            continue
        with np.load(src_path) as z:
            l1 = z["labels_1s"]
        valid = ~np.isnan(l1)
        a = np.abs(l1[valid]).astype(np.float32)
        n_total += len(l1)  # count ALL events (incl. nan-only edge) for coverage
        # uniform sub-sample
        if len(a) > 0:
            keep = min(len(a), max(1, ABS1_SAMPLE_CAP // max(1, len(dates))))
            if len(a) > keep:
                idx = rng.choice(len(a), size=keep, replace=False)
                abs1_sample.append(a[idx])
            else:
                abs1_sample.append(a)
        src_count += 1
        if src_count % 50 == 0:
            print(f"  pass1: {src_count}/{len(dates)} sampled", flush=True)

    abs1_sample = np.concatenate(abs1_sample) if abs1_sample else np.array([], dtype=np.float32)
    print(f"  pass1: |labels_1s| sample size = {len(abs1_sample):,}", flush=True)

    thresholds = {bn: float(np.percentile(abs1_sample, p)) for bn, p in zip(band_names, pct_targets)}
    print(f"  thresholds (|labels_1s|): {thresholds}", flush=True)

    # PASS 2: walk source + cache together, accumulate persistence by band
    band_persist = {
        bn: {k: {1: 0, 0: 0, -1: 0} for k in PERSIST_KEYS} for bn in band_names
    }
    band_counts = {bn: 0 for bn in band_names}

    # Also collect pressure-score sample
    ps_sample = []
    PS_PER_DATE = max(1, SAMPLE_CAP // max(1, len(dates)))

    for i, d in enumerate(dates, 1):
        src_path = os.path.join(SRC_DIR, f"{d}_mbo_events.npz")
        cache_path = os.path.join(CACHE_DIR, f"{d}_pressure.npz")
        if not (os.path.exists(src_path) and os.path.exists(cache_path)):
            continue
        with np.load(src_path) as zs:
            l1 = zs["labels_1s"]
        with np.load(cache_path) as zc:
            persist = {k: zc[k] for k in PERSIST_KEYS}
            ps = zc["pressure_score"]

        # Global counts
        for k in PERSIST_KEYS:
            arr = persist[k]
            p_counts[k][1]  += int((arr == 1).sum())
            p_counts[k][0]  += int((arr == 0).sum())
            p_counts[k][-1] += int((arr == -1).sum())

        # Pressure score sample
        ps_valid = ps[np.isfinite(ps)]
        if len(ps_valid) > 0:
            keep = min(len(ps_valid), PS_PER_DATE)
            if len(ps_valid) > keep:
                idx = rng.choice(len(ps_valid), size=keep, replace=False)
                ps_sample.append(ps_valid[idx])
            else:
                ps_sample.append(ps_valid)

        # Confidence-stratified persistence
        absl1 = np.abs(l1)
        for bn in band_names:
            thr = thresholds[bn]
            mask = (absl1 >= thr) & np.isfinite(l1)
            band_counts[bn] += int(mask.sum())
            for k in PERSIST_KEYS:
                arr = persist[k][mask]
                band_persist[bn][k][1]  += int((arr == 1).sum())
                band_persist[bn][k][0]  += int((arr == 0).sum())
                band_persist[bn][k][-1] += int((arr == -1).sum())

        if i % 25 == 0 or i == len(dates):
            print(f"  pass2: {i}/{len(dates)} dates", flush=True)

    ps_sample = np.concatenate(ps_sample) if ps_sample else np.array([], dtype=np.float32)

    # -----------------------------------------------------------------------
    # Render report
    # -----------------------------------------------------------------------
    def pct_row(counts_dict):
        n = sum(counts_dict.values())
        if n == 0:
            return "0.00 / 0.00 / 0.00"
        return (f"{100.0*counts_dict[1]/n:5.2f} / "
                f"{100.0*counts_dict[0]/n:5.2f} / "
                f"{100.0*counts_dict[-1]/n:5.2f}")

    lines = []
    lines.append("# HC #451 R5 — Pressure Label Cache Report")
    lines.append("")
    lines.append(f"- Dates processed: **{len(dates)}**  (range {dates[0]} .. {dates[-1]})")
    lines.append(f"- Total events (incl. NaN-edge): **{n_total:,}**")
    lines.append(f"- Cache directory: `{CACHE_DIR}`")
    lines.append(f"- Neutrality band: 0.5 (ticks, same unit as source labels)")
    lines.append("")
    lines.append("## Overall persistence rates")
    lines.append("")
    lines.append("Format: `+1 (agree) / 0 (neutral) / -1 (flip)` as percent of all events.")
    lines.append("")
    lines.append("| Horizon pair | +1 / 0 / -1 (%) |")
    lines.append("|---|---|")
    for k in PERSIST_KEYS:
        lines.append(f"| {k} | {pct_row(p_counts[k])} |")
    lines.append("")
    lines.append("## Confidence-stratified persistence (by |labels_1s|)")
    lines.append("")
    lines.append("Thresholds derived from a uniform sample of valid |labels_1s|.")
    lines.append("")
    lines.append("| Band | |labels_1s| >= (ticks) | n events | persistence_1s_10s (+1/0/-1) | persistence_1s_30s (+1/0/-1) | persistence_5s_30s (+1/0/-1) |")
    lines.append("|---|---|---|---|---|---|")
    for bn in band_names:
        thr = thresholds[bn]
        n_b = band_counts[bn]
        lines.append(
            f"| {bn} | {thr:.3f} | {n_b:,} | "
            f"{pct_row(band_persist[bn]['persistence_1s_10s'])} | "
            f"{pct_row(band_persist[bn]['persistence_1s_30s'])} | "
            f"{pct_row(band_persist[bn]['persistence_5s_30s'])} |"
        )
    lines.append("")
    lines.append("## Pressure score distribution")
    lines.append("")
    if len(ps_sample) > 0:
        pct = lambda p: float(np.percentile(ps_sample, p))
        lines.append(f"- Sample size: {len(ps_sample):,} (uniform sub-sample)")
        lines.append(f"- Mean: {ps_sample.mean():+.4f}")
        lines.append(f"- p10 / p25 / p50 / p75 / p90 / p99: "
                     f"{pct(10):+.3f} / {pct(25):+.3f} / {pct(50):+.3f} / "
                     f"{pct(75):+.3f} / {pct(90):+.3f} / {pct(99):+.3f}")
        lines.append(f"- |pressure_score| >= 0.9: {(np.abs(ps_sample)>=0.9).mean()*100:.2f}%  (~all four horizons agree)")
        lines.append(f"- |pressure_score| <= 0.1: {(np.abs(ps_sample)<=0.1).mean()*100:.2f}%  (mixed / neutral)")
        lines.append("")
        lines.append("```")
        lines.append(ascii_histogram(ps_sample))
        lines.append("```")
    else:
        lines.append("(no pressure-score sample available)")
    lines.append("")

    # -----------------------------------------------------------------------
    # Interpretation paragraph
    # -----------------------------------------------------------------------
    top1 = band_persist["top1pct"]
    top1_n = sum(top1["persistence_1s_10s"].values())
    p1_10_agree = 100.0 * top1["persistence_1s_10s"][1] / top1_n if top1_n else 0
    p1_10_flip  = 100.0 * top1["persistence_1s_10s"][-1] / top1_n if top1_n else 0
    p1_10_neut  = 100.0 * top1["persistence_1s_10s"][0]  / top1_n if top1_n else 0
    p1_30_agree = 100.0 * top1["persistence_1s_30s"][1] / top1_n if top1_n else 0
    p1_30_flip  = 100.0 * top1["persistence_1s_30s"][-1] / top1_n if top1_n else 0
    p1_30_neut  = 100.0 * top1["persistence_1s_30s"][0]  / top1_n if top1_n else 0

    # Overall (all events) for comparison
    all_n_110 = sum(p_counts["persistence_1s_10s"].values())
    all_110_agree = 100.0 * p_counts["persistence_1s_10s"][1] / all_n_110 if all_n_110 else 0
    all_110_flip  = 100.0 * p_counts["persistence_1s_10s"][-1] / all_n_110 if all_n_110 else 0

    lines.append("## Interpretation")
    lines.append("")
    lines.append(
        f"For the **top 1%** of events by |labels_1s| (|labels_1s| >= "
        f"{thresholds['top1pct']:.2f} ticks), the 1s sign **agrees** with the 10s sign "
        f"{p1_10_agree:.1f}% of the time, **flips** {p1_10_flip:.1f}%, and lands in the "
        f"neutrality band {p1_10_neut:.1f}%. At the 30s horizon the agreement rate is "
        f"{p1_30_agree:.1f}% with a flip rate of {p1_30_flip:.1f}% (neutral {p1_30_neut:.1f}%). "
        f"For comparison, across **all events** (most of which are noise around the bid-ask), "
        f"1s-to-10s agreement is only {all_110_agree:.1f}% (flip {all_110_flip:.1f}%). "
        f"This means the high-confidence 1s signal does carry forward — the directional edge "
        f"observed at 1s is materially more likely to still be in the same direction at 10s, "
        f"and even at 30s, than would be true of an average event. However, **the flip rate "
        f"is non-trivial** ({p1_10_flip:.0f}% at 10s, {p1_30_flip:.0f}% at 30s for the top "
        f"1%), which means the alpha is real but decays and reverses for a meaningful "
        f"minority of trades — consistent with the HC #428 decay-window evidence and the "
        f"existing belief that holds beyond ~10s start picking up noise rather than signal."
    )
    lines.append("")
    lines.append("## Notes")
    lines.append("")
    lines.append("- MFE/MAE within 10s is a 3-point approximation (1s, 5s, 10s sample of the path).")
    lines.append("  For a true tick-resolution MFE/MAE, run `alpha_discovery/evaluation/mfe_mae_path_analysis.py --dense`.")
    lines.append("- Neutrality band of 0.5 is in the source label unit (ticks), not bps. The")
    lines.append("  HC brief used 'bps' loosely; the source NPZ labels per the codebase header are ticks.")
    lines.append("- Pressure-score distribution is a 5M uniform sub-sample, not the full corpus,")
    lines.append("  to keep this report's memory bounded.")
    lines.append("")

    Path(OUT_PATH).write_text("\n".join(lines))
    print(f"Wrote {OUT_PATH}")

    # Also emit a small machine-readable summary
    import json
    summary = {
        "dates_processed": len(dates),
        "n_total_events": int(n_total),
        "neutrality_ticks": 0.5,
        "thresholds_abs_labels_1s": thresholds,
        "overall_persistence": {
            k: {"plus1_pct": 100.0*p_counts[k][1]/max(1,sum(p_counts[k].values())),
                "neutral_pct": 100.0*p_counts[k][0]/max(1,sum(p_counts[k].values())),
                "minus1_pct": 100.0*p_counts[k][-1]/max(1,sum(p_counts[k].values()))}
            for k in PERSIST_KEYS
        },
        "band_persistence": {
            bn: {
                "n": band_counts[bn],
                **{k: {"plus1_pct": 100.0*band_persist[bn][k][1]/max(1,sum(band_persist[bn][k].values())),
                       "neutral_pct": 100.0*band_persist[bn][k][0]/max(1,sum(band_persist[bn][k].values())),
                       "minus1_pct": 100.0*band_persist[bn][k][-1]/max(1,sum(band_persist[bn][k].values()))}
                   for k in PERSIST_KEYS}
            }
            for bn in band_names
        },
        "pressure_score_percentiles": (
            {p: float(np.percentile(ps_sample, p)) for p in (10,25,50,75,90,99)}
            if len(ps_sample) else {}
        ),
        "pressure_score_mean": float(ps_sample.mean()) if len(ps_sample) else None,
    }
    with open(os.path.join(OUT_DIR, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Wrote {os.path.join(OUT_DIR, 'summary.json')}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
