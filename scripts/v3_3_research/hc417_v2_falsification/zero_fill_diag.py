"""
HC #417 falsification — Phase 4: zero-fill day diagnosis.

For cell v2_1s_short_top05 (Top0.5% short side @ 1s):
  - Zero-fill dates from v3.4.2-borrowed run: Apr 19, 21, 22, 23, 24, 26, 28, 29
  - High-fill dates: Mar 17, 18

Compare:
  (a) Top0.5% short threshold per date (the *signed* pred percentile threshold).
      Is the magnitude similar across dates, or is the GLOBAL Top0.5% threshold
      simply not reached on quiet days?
  (b) The GLOBAL Top0.5% short threshold (the one actually used by the backtester
      ranks across ALL samples, not per-day) — how many samples per date have
      signed_pred >= that global threshold?
  (c) Direction: are the highest-magnitude predictions on the zero-fill dates
      LONG-side rather than SHORT-side? (i.e. is the signal flipping direction?)
  (d) Data quality: n_samples per date.

Output: output/hc417_zero_fill_diagnosis.md
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

LVL3 = Path("/home/jupiter/Lvl3Quant")
NPZ_PATH = LVL3 / "output/hc417_v2_full_oot_wrapped_for_hc413.npz"
OUT_MD = LVL3 / "output/hc417_zero_fill_diagnosis.md"

ZERO_FILL_DATES = ["20260419", "20260421", "20260422", "20260423",
                   "20260424", "20260426", "20260428", "20260429"]
HIGH_FILL_DATES = ["20260317", "20260318"]


def main():
    d = np.load(NPZ_PATH, allow_pickle=True)
    n = int(d["n_samples"])
    oot_dates = [str(x) for x in d["oot_dates"]]
    pred_1s = d["pred_log_ret_1s"][:n].astype(np.float64)
    mask_1s = d["mask_log_ret_1s"][:n].astype(bool) & np.isfinite(pred_1s)

    # Need to map each sample to its date_idx. The wrapped NPZ has no day_index field,
    # so we use FIFO labels' per-day count to reconstruct it (the backtester does the same).
    sys.path.insert(0, str(LVL3 / "scripts/hc413_scalping_backtester"))
    from fill_sim import load_fifo_for_dates
    LABELS_DIR = LVL3 / "data/processed/mbo_events_smart_v3_fifo_labels"
    fifo = load_fifo_for_dates(LABELS_DIR, oot_dates)
    fifo_n_per_day = list(fifo["_n_per_day"])
    cum = np.cumsum([0] + fifo_n_per_day)
    date_idx_of_sample = np.searchsorted(cum, np.arange(n), side="right") - 1

    # Short-side signed pred = -pred_1s (a more negative pred_1s -> more bullish short signal)
    signed_short = -pred_1s
    valid = mask_1s & np.isfinite(signed_short)
    # GLOBAL Top0.5% threshold (this is what the backtester uses)
    global_thr = np.percentile(signed_short[valid], 100.0 - 0.5)

    # Equivalently, top0.5% LONG threshold
    signed_long = pred_1s
    global_thr_long = np.percentile(signed_long[valid], 100.0 - 0.5)

    print(f"GLOBAL Top0.5% short threshold: signed_short >= {global_thr:.6f}")
    print(f"GLOBAL Top0.5% long  threshold: signed_long  >= {global_thr_long:.6f}")
    print(f"Total samples passing global short threshold: {int((signed_short[valid] >= global_thr).sum())}")

    lines = []
    lines.append("# HC #417 Phase 4 — Zero-fill day diagnosis for v2_1s_short_top05\n")
    lines.append(f"NPZ: `{NPZ_PATH.name}`")
    lines.append(f"Total samples n={n:,}, OOT dates={len(oot_dates)}")
    lines.append(f"GLOBAL Top0.5% short threshold: signed_short = -pred_1s >= **{global_thr:.5f}**")
    lines.append(f"  - i.e. pred_1s <= {-global_thr:.5f} qualifies as a Top0.5% short signal\n")

    lines.append("## Per-date diagnosis (TARGETED dates)\n")
    lines.append("Columns:")
    lines.append("- `n_samples`: samples present on the date in the wrapped NPZ")
    lines.append("- `n_global_short_top05`: # samples on the date with signed_short >= global Top0.5% threshold")
    lines.append("- `n_global_long_top05`: # samples on the date with signed_long >= global Top0.5% threshold (direction check)")
    lines.append("- `pred_1s_min/max/mean/std`: distribution of pred_1s on the date (negative = bullish short)")
    lines.append("- `local_top05_short_thr`: the date-internal Top0.5% short threshold (what would gate if ranked PER-DAY)")
    lines.append("- `local_top05_pred_1s_at_cutoff`: pred_1s value at the local Top0.5% short cutoff (negative if bearish predictions present)")
    lines.append("")
    lines.append("| date | bucket | n_samples | global_short | global_long | pred_min | pred_max | pred_mean | pred_std | local_short_thr | pred@local_cutoff |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|")

    target = [(dt, "zero-fill") for dt in ZERO_FILL_DATES] + [(dt, "high-fill") for dt in HIGH_FILL_DATES]
    for dt, bucket in target:
        if dt not in oot_dates:
            lines.append(f"| {dt} | {bucket} | (not in NPZ) | | | | | | | | |")
            continue
        di = oot_dates.index(dt)
        mask_d = (date_idx_of_sample == di) & valid
        n_d = int(mask_d.sum())
        if n_d == 0:
            lines.append(f"| {dt} | {bucket} | 0 | 0 | 0 | n/a | n/a | n/a | n/a | n/a | n/a |")
            continue
        sshort_d = signed_short[mask_d]
        slong_d = signed_long[mask_d]
        pred_d = pred_1s[mask_d]
        n_short_g = int((sshort_d >= global_thr).sum())
        n_long_g = int((slong_d >= global_thr_long).sum())
        # Local top0.5% short threshold
        local_thr_short = np.percentile(sshort_d, 100.0 - 0.5)
        local_cutoff_pred = -local_thr_short
        lines.append(
            f"| {dt} | {bucket} | {n_d:,} | {n_short_g} | {n_long_g} | "
            f"{pred_d.min():+.5f} | {pred_d.max():+.5f} | {pred_d.mean():+.5f} | {pred_d.std():.5f} | "
            f"{local_thr_short:+.5f} | {local_cutoff_pred:+.5f} |"
        )

    # Overall per-date table for context (all OOT dates)
    lines.append("\n## All-dates context table (per-date global Top0.5% short count)\n")
    lines.append("| date | n_samples | global_short_top05 | global_long_top05 | pred_min | pred_max | pred_mean |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for di, dt in enumerate(oot_dates):
        mask_d = (date_idx_of_sample == di) & valid
        n_d = int(mask_d.sum())
        if n_d == 0:
            lines.append(f"| {dt} | 0 | 0 | 0 | n/a | n/a | n/a |")
            continue
        sshort_d = signed_short[mask_d]
        slong_d = signed_long[mask_d]
        pred_d = pred_1s[mask_d]
        lines.append(
            f"| {dt} | {n_d:,} | {int((sshort_d >= global_thr).sum())} | "
            f"{int((slong_d >= global_thr_long).sum())} | "
            f"{pred_d.min():+.5f} | {pred_d.max():+.5f} | {pred_d.mean():+.5f} |"
        )

    # Diagnostic verdict
    lines.append("\n## Diagnostic verdict\n")
    # Re-compute with mask breakdown to separate "no valid samples" from "valid but no short top05 qualifiers"
    pred_finite = np.isfinite(pred_1s)
    n_no_data = 0; n_mask_all_false = 0
    n_threshold_artifact = 0; n_filter_issue = 0; n_long_skewed = 0
    for dt in ZERO_FILL_DATES:
        if dt not in oot_dates:
            continue
        di = oot_dates.index(dt)
        m_d = (date_idx_of_sample == di)
        n_s = int(m_d.sum())
        n_mt = int((mask_1s & m_d).sum())
        n_fp = int((pred_finite & m_d).sum())
        if n_s == 0:
            n_no_data += 1
            continue
        if n_mt == 0 and n_fp > 0:
            # Predictions exist but target mask is all-False → no 1s target labels for this date
            n_mask_all_false += 1
            continue
        n_v = int(((mask_1s & pred_finite) & m_d).sum())
        if n_v == 0:
            n_no_data += 1
            continue
        sshort_d = signed_short[m_d & mask_1s & pred_finite]
        slong_d = signed_long[m_d & mask_1s & pred_finite]
        n_short_g = int((sshort_d >= global_thr).sum())
        n_long_g = int((slong_d >= global_thr_long).sum())
        if n_short_g == 0:
            n_threshold_artifact += 1
        else:
            n_filter_issue += 1
        if n_long_g > 5 * max(n_short_g, 1):
            n_long_skewed += 1

    lines.append(f"### Root cause breakdown across {len(ZERO_FILL_DATES)} zero-fill dates")
    lines.append("")
    lines.append(f"- **{n_no_data}** dates have ZERO samples in the wrapped NPZ at all (no data)")
    lines.append(f"- **{n_mask_all_false}** dates have predictions but `mask_log_ret_1s` is uniformly False")
    lines.append(f"  (1s target labels missing → entire date filtered out by HC #408 mask gate)")
    lines.append(f"- **{n_threshold_artifact}** dates have valid samples but ZERO meet global Top0.5% short threshold")
    lines.append(f"  (predictions too small on quiet days — threshold-too-strict artifact)")
    lines.append(f"- **{n_filter_issue}** dates have Top0.5% short qualifiers that FIFO didn't fill (rare)")
    lines.append(f"- **{n_long_skewed}** dates had >>5x more long-side signals than short-side (direction-flip evidence)\n")

    lines.append("### Final interpretation\n")
    primary = max(
        ("mask_filtered", n_mask_all_false),
        ("threshold_artifact", n_threshold_artifact),
        ("direction_flip", n_long_skewed),
        ("no_data", n_no_data),
        key=lambda kv: kv[1],
    )
    if primary[0] == "mask_filtered":
        lines.append("**PRIMARY CAUSE: missing 1s target labels (mask_log_ret_1s == False).**")
        lines.append("These dates have v2 predictions but the canonical FIFO target labelling did NOT")
        lines.append("produce 1s realised returns (likely truncated MBO recordings near session close or")
        lines.append("missing label-builder coverage for those dates). This is a DATA-PIPELINE issue,")
        lines.append("not a signal-decay issue. The model still has signal on those dates; we just")
        lines.append("can't evaluate it because labels are missing.\n")
        lines.append("Implication for HC #415 rule 2:")
        lines.append("- Per_day_pass_rate denominator currently includes these dates as `n_fills==0`")
        lines.append("- The v3.4.2-borrowed run reported `per_day_pass_rate=0.84` for v3.4.2_1s_short_top05")
        lines.append("- The v2-native MFE run reports `per_day_pass_rate=1.00` for v2_1s_short_top05")
        lines.append("  because the verdict generator counts only `active` dates (n_fills > 0)")
        lines.append("  → ZERO-FILL DATES ARE NOT PENALISING THE 1.00 score; they're correctly excluded.")
    elif primary[0] == "threshold_artifact":
        lines.append("**PRIMARY CAUSE: threshold-too-strict on quiet days.**")
        lines.append("Predictions on the late-April dates were uniformly small; the GLOBAL Top0.5%")
        lines.append("short threshold (signed_short >= {:.5f}) was not reached.".format(global_thr))
    elif primary[0] == "direction_flip":
        lines.append("**PRIMARY CAUSE: signal flipped direction (long-side dominant on those dates).**")
    else:
        lines.append("**PRIMARY CAUSE: data missing for these dates in the NPZ.**")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"wrote {OUT_MD}")


if __name__ == "__main__":
    main()
