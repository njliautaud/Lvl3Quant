#!/usr/bin/env python3
"""
HC #346 PASS 7 — Robustness validation of pass 6's positive pocket

Pass 6 found ONE statistically positive config:
  S_Top0.5% × agree_15 × no_reversal × golden-ToD × vol_mid
  n_fill=19, mean +1.29 t/fill, Sharpe 2.70, 95% CI [+0.29, +2.13]

Concerns:
  - Sample size 19 fills
  - Days 0 & 4 contributed ZERO fills (only days 1-3)
  - "Golden ToD buckets" were SELECTED FROM the same data → in-sample bias
  - "vol_mid" cut also derived from same data

Pass 7 stress-tests:
  R1. LEAVE-ONE-DAY-OUT validation: rebuild config on 4 days, test on 5th
  R2. PERMUTATION test: shuffle pred-side labels 1000× to get null distribution
      of the pocket's Sharpe
  R3. ToD bucket sensitivity: shift golden buckets ±15min, drop one bucket at
      a time
  R4. Looser thresholds: relax band to Top1%/Top5%, drop one filter at a time —
      see how edge degrades
  R5. Tp4sl3 vs Tp8sl5 cross-check: does the pocket appear in BOTH FIFO label
      types?
  R6. Effective N — Newey-West-style autocorrelation correction (signals close
      in time aren't independent)

Outputs: output/v3_2_allnight_research_20260514/pass7_robustness/
"""
from __future__ import annotations

import json
import math
import sys
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"
OUT = ROOT / "output/v3_2_allnight_research_20260514/pass7_robustness"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "pass7.log"

PASSIVE_COST = 0.376


def log(msg):
    ts = datetime.utcnow().isoformat(timespec="seconds")
    line = f"[{ts}Z] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def safe_sharpe(arr):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 2:
        return 0.0
    sd = a.std(ddof=1)
    if sd <= 1e-12:
        return 0.0
    return float(a.mean() / sd * math.sqrt(a.size))


def main():
    log("PASS 7 — Robustness validation of pass 6 pocket")
    d = np.load(PRED_NPZ, allow_pickle=True)
    n = int(d["n_samples"])

    pred_1s = np.asarray(d["pred_log_ret_1s"]).flatten()
    pred_5s = np.asarray(d["pred_log_ret_5s"]).flatten()
    rev15 = np.asarray(d["pred_p_reversal_15s"]).flatten()
    rev30 = np.asarray(d["pred_p_reversal_30s"]).flatten()
    pred_vol30 = np.asarray(d["pred_pred_realized_vol_30s_ticks"]).flatten()
    mask_vol = np.asarray(d["mask_pred_realized_vol_30s_ticks"]).flatten().astype(bool)
    fifo43 = np.asarray(d["target_fifo_tp4sl3_net"]).flatten()
    mask_fifo43 = np.asarray(d["mask_fifo_tp4sl3_net"]).flatten().astype(bool)
    fifo85 = np.asarray(d["target_fifo_tp8sl5_net"]).flatten()
    mask_fifo85 = np.asarray(d["mask_fifo_tp8sl5_net"]).flatten().astype(bool)
    mask_5s = np.asarray(d["mask_log_ret_5s"]).flatten().astype(bool)

    # synth ToD & day idx
    n_days = 5
    spd = n // n_days
    day_idx = np.zeros(n, dtype=np.int32)
    sec_from_open = np.zeros(n, dtype=np.float32)
    rth_seconds = int(6.5 * 3600)
    for di in range(n_days):
        s = di * spd
        e = (di + 1) * spd if di < n_days - 1 else n
        day_idx[s:e] = di
        sec_from_open[s:e] = np.linspace(0, rth_seconds, e - s, dtype=np.float32)
    bucket = np.clip((sec_from_open // 1800).astype(int), 0, 12)
    rth_open_sec = 9 * 3600 + 30 * 60
    bucket_labels = []
    for b in range(13):
        sh, sm = divmod(rth_open_sec + b * 1800, 3600)
        eh, em = divmod(rth_open_sec + (b + 1) * 1800, 3600)
        bucket_labels.append(f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}")

    valid_5s = mask_5s
    a5 = np.where(valid_5s, pred_5s, np.nan)

    def percentile_band_short(pct, mask=None):
        m = valid_5s if mask is None else (valid_5s & mask)
        a = np.where(m, pred_5s, np.nan)
        cut = np.nanquantile(a, pct / 100.0)
        return (a <= cut) & m

    agree_15 = (np.sign(pred_1s) == np.sign(pred_5s))
    no_rev = (rev15 < 0.4) & (rev30 < 0.4)

    # vol_mid: 33-67%ile of pred_vol30 over valid samples
    vol_valid = mask_vol & np.isfinite(pred_vol30)
    vlo = np.nanquantile(np.where(vol_valid, pred_vol30, np.nan), 0.33)
    vhi = np.nanquantile(np.where(vol_valid, pred_vol30, np.nan), 0.67)
    vol_mid = vol_valid & (pred_vol30 >= vlo) & (pred_vol30 <= vhi)

    golden_buckets = [4, 6, 8]  # 11:18-12:00, 12:18-13:00, 13:18-14:00 (offset by 15min wrap)
    # Actually rebuild with proper indices: bucket=0 is 09:30-10:00, bucket=4 is 11:30-12:00...
    # The "golden" ones were 11:18-12:00 (b=4), 12:18-13:00 (b=6), 13:18-14:00 (b=8)
    in_golden = np.isin(bucket, golden_buckets)

    def compute_pocket(em, fifo_arr=fifo43, mask_arr=mask_fifo43, side_sign=-1):
        m = em & mask_arr
        if int(m.sum()) == 0:
            return None
        raw = side_sign * fifo_arr[m]
        fm = raw != 0
        if int(fm.sum()) == 0:
            return None
        pnl = raw[fm] - PASSIVE_COST
        return pnl

    results = {}

    # ─────────────── R1. Leave-one-day-out ───────────────
    log("R1. Leave-one-day-out validation")
    r1_rows = []
    pocket_em = percentile_band_short(0.5) & agree_15 & no_rev & vol_mid & in_golden
    for du in range(n_days):
        # Train: derive band threshold from "other 4 days", apply to held-out day
        train_mask = day_idx != du
        test_mask = day_idx == du
        # Re-derive top0.5% short threshold using training days
        a_train = np.where(valid_5s & train_mask, pred_5s, np.nan)
        if int(np.isfinite(a_train).sum()) < 100:
            r1_rows.append({"holdout": du, "n_fill": 0, "mean_t": float("nan"),
                            "wr": float("nan"), "sharpe": float("nan")})
            continue
        cut_train = np.nanquantile(a_train, 0.005)
        em_test = (a5 <= cut_train) & valid_5s & test_mask & agree_15 & no_rev & vol_mid & in_golden
        pnl = compute_pocket(em_test)
        if pnl is None:
            r1_rows.append({"holdout": du, "n_fill": 0, "mean_t": float("nan"),
                            "wr": float("nan"), "sharpe": float("nan")})
            continue
        r1_rows.append({
            "holdout": du, "n_fill": int(pnl.size),
            "mean_t": float(pnl.mean()), "wr": float((pnl > 0).mean() * 100),
            "sharpe": safe_sharpe(pnl),
            "total_t": float(pnl.sum()),
        })
    results["R1_lodo"] = r1_rows

    # ─────────────── R2. Permutation test ───────────────
    log("R2. Permutation test on pocket Sharpe")
    # Shuffle pred_5s sign labels (keeping magnitudes), recompute pocket Sharpe 1000x
    pnl_orig = compute_pocket(pocket_em)
    if pnl_orig is not None:
        sharpe_orig = safe_sharpe(pnl_orig)
        rng = np.random.default_rng(2026)
        n_perm = 1000
        sharpes_null = np.zeros(n_perm)
        # Permute the SIGN of pred_5s only on valid_5s rows
        valid_idx = np.where(valid_5s)[0]
        signs_orig = np.sign(pred_5s[valid_idx])
        mags = np.abs(pred_5s[valid_idx])
        for i in range(n_perm):
            shuffled_signs = rng.permutation(signs_orig)
            shuffled_pred5 = pred_5s.copy()
            shuffled_pred5[valid_idx] = shuffled_signs * mags
            a5_perm = np.where(valid_5s, shuffled_pred5, np.nan)
            cut_perm = np.nanquantile(a5_perm, 0.005)
            em_perm = (a5_perm <= cut_perm) & valid_5s & agree_15 & no_rev & vol_mid & in_golden
            pnl_perm = compute_pocket(em_perm)
            if pnl_perm is None:
                sharpes_null[i] = 0.0
            else:
                sharpes_null[i] = safe_sharpe(pnl_perm)
        p_value = float((sharpes_null >= sharpe_orig).mean())
        results["R2_permutation"] = {
            "observed_sharpe": sharpe_orig,
            "null_mean_sharpe": float(sharpes_null.mean()),
            "null_p95_sharpe": float(np.percentile(sharpes_null, 95)),
            "null_p99_sharpe": float(np.percentile(sharpes_null, 99)),
            "p_value": p_value,
            "n_perm": n_perm,
        }
        log(f"  R2: observed Sharpe={sharpe_orig:.2f}, null mean={sharpes_null.mean():.2f}, p={p_value:.4f}")
    else:
        results["R2_permutation"] = {"error": "no pocket fills"}

    # ─────────────── R3. ToD bucket sensitivity ───────────────
    log("R3. ToD bucket sensitivity")
    r3 = {}
    for bset_name, bset in [("original_4_6_8", [4, 6, 8]),
                             ("shift_back_3_5_7", [3, 5, 7]),
                             ("shift_fwd_5_7_9", [5, 7, 9]),
                             ("only_4_6", [4, 6]),
                             ("only_6_8", [6, 8]),
                             ("only_4_8", [4, 8]),
                             ("expanded_3_4_6_8", [3, 4, 6, 8]),
                             ("expanded_4_6_8_9", [4, 6, 8, 9])]:
        em = percentile_band_short(0.5) & agree_15 & no_rev & vol_mid & np.isin(bucket, bset)
        pnl = compute_pocket(em)
        if pnl is None:
            r3[bset_name] = {"n_fill": 0, "mean_t": float("nan"), "wr": float("nan"), "sharpe": float("nan")}
            continue
        r3[bset_name] = {"n_fill": int(pnl.size), "mean_t": float(pnl.mean()),
                         "wr": float((pnl > 0).mean() * 100), "sharpe": safe_sharpe(pnl)}
    results["R3_tod_sensitivity"] = r3

    # ─────────────── R4. Drop-one-filter ablation ───────────────
    log("R4. Drop-one-filter ablation")
    r4 = {}
    base_filters = {
        "all_filters": percentile_band_short(0.5) & agree_15 & no_rev & vol_mid & in_golden,
        "drop_agree_15": percentile_band_short(0.5) & no_rev & vol_mid & in_golden,
        "drop_no_rev": percentile_band_short(0.5) & agree_15 & vol_mid & in_golden,
        "drop_vol_mid": percentile_band_short(0.5) & agree_15 & no_rev & in_golden,
        "drop_golden_tod": percentile_band_short(0.5) & agree_15 & no_rev & vol_mid,
        "loose_band_Top1": percentile_band_short(1.0) & agree_15 & no_rev & vol_mid & in_golden,
        "loose_band_Top5": percentile_band_short(5.0) & agree_15 & no_rev & vol_mid & in_golden,
        "tight_band_Top0p1": percentile_band_short(0.1) & agree_15 & no_rev & vol_mid & in_golden,
    }
    for name, em in base_filters.items():
        pnl = compute_pocket(em)
        if pnl is None:
            r4[name] = {"n_fill": 0, "mean_t": float("nan"), "wr": float("nan"), "sharpe": float("nan")}
            continue
        r4[name] = {"n_fill": int(pnl.size), "mean_t": float(pnl.mean()),
                    "wr": float((pnl > 0).mean() * 100), "sharpe": safe_sharpe(pnl)}
    results["R4_drop_one"] = r4

    # ─────────────── R5. Tp4sl3 vs Tp8sl5 cross-check ───────────────
    log("R5. Tp4sl3 vs Tp8sl5 cross-check on pocket")
    r5 = {}
    for name, fifo_a, mask_a, label in [("tp4sl3", fifo43, mask_fifo43, "tp4sl3"),
                                          ("tp8sl5", fifo85, mask_fifo85, "tp8sl5")]:
        pnl = compute_pocket(pocket_em, fifo_a, mask_a)
        if pnl is None:
            r5[label] = None
            continue
        r5[label] = {
            "n_fill": int(pnl.size), "mean_t": float(pnl.mean()),
            "wr": float((pnl > 0).mean() * 100), "sharpe": safe_sharpe(pnl)
        }
    results["R5_fifo_cross"] = r5

    # ─────────────── R6. Autocorrelation-corrected effective N ───────────────
    log("R6. Effective N via Newey-West")
    pnl_orig = compute_pocket(pocket_em)
    if pnl_orig is not None and pnl_orig.size > 4:
        # crude: positive correlation between consecutive trades shrinks N_eff
        x = pnl_orig - pnl_orig.mean()
        # lag-1 autocorrelation (approx — trades not necessarily ordered in time but
        # we use sample-index order which approximates time)
        if x.size >= 2 and x.std() > 0:
            rho1 = float(np.corrcoef(x[:-1], x[1:])[0, 1])
        else:
            rho1 = 0.0
        # Effective sample size: N / (1 + 2*rho_1) for AR(1) approximation
        n_eff = pnl_orig.size / max(1.0, 1 + 2 * max(0, rho1))
        sharpe = safe_sharpe(pnl_orig)
        sharpe_corrected = pnl_orig.mean() / pnl_orig.std(ddof=1) * math.sqrt(n_eff) if pnl_orig.std(ddof=1) > 0 else 0.0
        results["R6_neweywest"] = {
            "n_obs": int(pnl_orig.size), "rho_lag1": rho1,
            "n_eff": float(n_eff), "sharpe_naive": sharpe,
            "sharpe_corrected": sharpe_corrected,
        }
    else:
        results["R6_neweywest"] = {"error": "no fills"}

    # ─────────────── Write report ───────────────
    md = OUT / "PASS7.md"
    with open(md, "w") as f:
        f.write("# PASS 7 — Robustness Validation of Pass 6 Pocket\n\n")
        f.write("Pocket = S_Top0.5% × agree_15 × no_reversal × golden-ToD × vol_mid (n=19, mean +1.29t)\n\n")

        f.write("## R1. Leave-One-Day-Out CV\n\n")
        f.write("| Holdout day | n_fill | mean_t | WR% | Sharpe | Total_t |\n|---|---:|---:|---:|---:|---:|\n")
        for r in r1_rows:
            tt = r.get("total_t", float("nan"))
            f.write(f"| {r['holdout']} | {r['n_fill']} | {r['mean_t']:.3f} | "
                    f"{r['wr']:.1f} | {r['sharpe']:.2f} | {tt:.2f} |\n")
        f.write("\n")

        f.write("## R2. Permutation Test (1000 random sign-shuffles)\n\n")
        if "p_value" in results.get("R2_permutation", {}):
            p2 = results["R2_permutation"]
            f.write(f"- Observed pocket Sharpe: **{p2['observed_sharpe']:.3f}**\n")
            f.write(f"- Null mean Sharpe: {p2['null_mean_sharpe']:.3f}\n")
            f.write(f"- Null 95th percentile: {p2['null_p95_sharpe']:.3f}\n")
            f.write(f"- Null 99th percentile: {p2['null_p99_sharpe']:.3f}\n")
            f.write(f"- **p-value (one-tail) = {p2['p_value']:.4f}**\n\n")
        else:
            f.write("- Could not compute (no pocket fills)\n\n")

        f.write("## R3. ToD Bucket Sensitivity\n\n")
        f.write("| Bucket set | n_fill | mean_t | WR% | Sharpe |\n|---|---:|---:|---:|---:|\n")
        for k, v in r3.items():
            f.write(f"| {k} | {v['n_fill']} | {v['mean_t']:.3f} | {v['wr']:.1f} | {v['sharpe']:.2f} |\n")
        f.write("\n")

        f.write("## R4. Drop-One-Filter Ablation\n\n")
        f.write("| Variant | n_fill | mean_t | WR% | Sharpe |\n|---|---:|---:|---:|---:|\n")
        for k, v in r4.items():
            f.write(f"| {k} | {v['n_fill']} | {v['mean_t']:.3f} | {v['wr']:.1f} | {v['sharpe']:.2f} |\n")
        f.write("\n")

        f.write("## R5. tp4sl3 vs tp8sl5 Cross-Check\n\n")
        f.write("| FIFO | n_fill | mean_t | WR% | Sharpe |\n|---|---:|---:|---:|---:|\n")
        for k, v in r5.items():
            if v is None:
                continue
            f.write(f"| {k} | {v['n_fill']} | {v['mean_t']:.3f} | {v['wr']:.1f} | {v['sharpe']:.2f} |\n")
        f.write("\n")

        f.write("## R6. Newey-West Effective Sample Size\n\n")
        if "n_obs" in results.get("R6_neweywest", {}):
            r6 = results["R6_neweywest"]
            f.write(f"- n_obs = {r6['n_obs']}\n")
            f.write(f"- lag-1 autocorr = {r6['rho_lag1']:.3f}\n")
            f.write(f"- n_eff = {r6['n_eff']:.1f}\n")
            f.write(f"- naive Sharpe = {r6['sharpe_naive']:.3f}\n")
            f.write(f"- AR(1)-corrected Sharpe = {r6['sharpe_corrected']:.3f}\n\n")

    with open(OUT / "pass7_results.json", "w") as f:
        json.dump(results, f, default=str, indent=2)
    log(f"PASS 7 complete — wrote {md}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL")
        log(traceback.format_exc())
        sys.exit(1)
