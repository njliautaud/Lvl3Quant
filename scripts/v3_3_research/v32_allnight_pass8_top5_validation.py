#!/usr/bin/env python3
"""
HC #346 PASS 8 — Validate the Top5% loose-band candidate

Pass 7 R4 surfaced: S_Top5% × agree_15 × no_reversal × golden-ToD × vol_mid
gives n=262 fills, mean +0.64t, Sharpe 3.85. This pass stress-tests it
the same way pass 7 stress-tested the (failed) Top0.5% pocket.

Tests:
  T1. Per-day breakdown (5 OOT days)
  T2. Leave-one-day-out CV
  T3. Time-of-day bucket coverage (which buckets contribute the trades?)
  T4. Permutation test
  T5. Subset Sharpe — half-split (first 2 days vs last 3)
  T6. Equity curve & max drawdown
  T7. Comparison: same config WITHOUT golden-ToD (any ToD)
  T8. Strict Top5% with NO confluence — does the edge survive?
  T9. Sensitivity to band: Top2%, Top3%, Top5%, Top10%, Top20%
  T10. Day-by-day equity AND if you'd traded ONE contract per fill,
       what's the daily/weekly P&L distribution

Outputs: output/v3_2_allnight_research_20260514/pass8_top5/
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
OUT = ROOT / "output/v3_2_allnight_research_20260514/pass8_top5"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "pass8.log"

PASSIVE_COST = 0.376
TICK_VALUE = 12.50


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
    log("PASS 8 — Validate Top5% loose-band candidate")
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

    def short_band_mask(pct, valid_extra=None):
        v = valid_5s if valid_extra is None else (valid_5s & valid_extra)
        a = np.where(v, pred_5s, np.nan)
        cut = np.nanquantile(a, pct / 100.0)
        return (a <= cut) & v

    agree_15 = (np.sign(pred_1s) == np.sign(pred_5s))
    no_rev = (rev15 < 0.4) & (rev30 < 0.4)
    vol_valid = mask_vol & np.isfinite(pred_vol30)
    vlo = np.nanquantile(np.where(vol_valid, pred_vol30, np.nan), 0.33)
    vhi = np.nanquantile(np.where(vol_valid, pred_vol30, np.nan), 0.67)
    vol_mid = vol_valid & (pred_vol30 >= vlo) & (pred_vol30 <= vhi)
    golden_buckets = [4, 6, 8]
    in_golden = np.isin(bucket, golden_buckets)

    def fifo_pnl(em, fifo_arr=fifo43, mask_arr=mask_fifo43):
        m = em & mask_arr
        raw = -fifo_arr[m]  # short-side
        fm = raw != 0
        if int(fm.sum()) == 0:
            return None, m
        return raw[fm] - PASSIVE_COST, m

    # base candidate config
    cand_em = short_band_mask(5.0) & agree_15 & no_rev & vol_mid & in_golden
    cand_pnl, cand_m = fifo_pnl(cand_em)
    log(f"Candidate: n_fill={cand_pnl.size if cand_pnl is not None else 0}")

    results = {}

    # T1. Per-day
    log("T1. Per-day")
    t1 = []
    for du in range(n_days):
        em_d = cand_em & (day_idx == du)
        pnl_d, _ = fifo_pnl(em_d)
        if pnl_d is None:
            t1.append({"day": du, "n_fill": 0, "mean_t": float("nan"),
                       "wr": float("nan"), "sharpe": float("nan"), "total_t": 0.0})
            continue
        t1.append({"day": du, "n_fill": int(pnl_d.size),
                   "mean_t": float(pnl_d.mean()),
                   "wr": float((pnl_d > 0).mean() * 100),
                   "sharpe": safe_sharpe(pnl_d),
                   "total_t": float(pnl_d.sum())})
    results["T1_per_day"] = t1

    # T2. LODO
    log("T2. LODO")
    t2 = []
    for du in range(n_days):
        train_m = day_idx != du
        test_m = day_idx == du
        a_train = np.where(valid_5s & train_m, pred_5s, np.nan)
        if int(np.isfinite(a_train).sum()) < 100:
            t2.append({"holdout": du, "n_fill": 0, "mean_t": float("nan"),
                       "wr": float("nan"), "sharpe": float("nan"), "total_t": 0.0})
            continue
        cut_train = np.nanquantile(a_train, 0.05)  # Top5% short
        em_test = (a5 <= cut_train) & valid_5s & test_m & agree_15 & no_rev & vol_mid & in_golden
        pnl_t, _ = fifo_pnl(em_test)
        if pnl_t is None:
            t2.append({"holdout": du, "n_fill": 0, "mean_t": float("nan"),
                       "wr": float("nan"), "sharpe": float("nan"), "total_t": 0.0})
            continue
        t2.append({"holdout": du, "n_fill": int(pnl_t.size),
                   "mean_t": float(pnl_t.mean()),
                   "wr": float((pnl_t > 0).mean() * 100),
                   "sharpe": safe_sharpe(pnl_t),
                   "total_t": float(pnl_t.sum())})
    results["T2_lodo"] = t2

    # T3. ToD bucket coverage
    log("T3. ToD bucket coverage")
    t3 = []
    for b in range(13):
        em_b = cand_em & (bucket == b)
        pnl_b, _ = fifo_pnl(em_b)
        if pnl_b is None:
            continue
        t3.append({"bucket": bucket_labels[b], "n_fill": int(pnl_b.size),
                   "mean_t": float(pnl_b.mean()),
                   "wr": float((pnl_b > 0).mean() * 100),
                   "sharpe": safe_sharpe(pnl_b)})
    results["T3_tod"] = t3

    # T4. Permutation test
    log("T4. Permutation 1000x")
    if cand_pnl is not None and cand_pnl.size > 0:
        sharpe_orig = safe_sharpe(cand_pnl)
        rng = np.random.default_rng(42)
        n_perm = 1000
        sharpes_null = np.zeros(n_perm)
        valid_idx = np.where(valid_5s)[0]
        signs_orig = np.sign(pred_5s[valid_idx])
        mags = np.abs(pred_5s[valid_idx])
        for i in range(n_perm):
            shuf_signs = rng.permutation(signs_orig)
            shuf = pred_5s.copy()
            shuf[valid_idx] = shuf_signs * mags
            a5p = np.where(valid_5s, shuf, np.nan)
            cut = np.nanquantile(a5p, 0.05)
            em_p = (a5p <= cut) & valid_5s & agree_15 & no_rev & vol_mid & in_golden
            mp = em_p & mask_fifo43
            raw = -fifo43[mp]
            fm = raw != 0
            if fm.sum() < 2:
                sharpes_null[i] = 0.0
                continue
            pnl_p = raw[fm] - PASSIVE_COST
            sharpes_null[i] = safe_sharpe(pnl_p)
        p_value = float((sharpes_null >= sharpe_orig).mean())
        results["T4_permutation"] = {
            "observed_sharpe": sharpe_orig,
            "null_mean": float(sharpes_null.mean()),
            "null_p95": float(np.percentile(sharpes_null, 95)),
            "null_p99": float(np.percentile(sharpes_null, 99)),
            "p_value": p_value,
        }
        log(f"  T4: obs={sharpe_orig:.2f}, null_mean={sharpes_null.mean():.2f}, p={p_value:.4f}")

    # T5. Half-split (first 2 days vs last 3)
    log("T5. Half-split")
    em_first = cand_em & (day_idx <= 1)
    em_last = cand_em & (day_idx >= 2)
    pnl_first, _ = fifo_pnl(em_first)
    pnl_last, _ = fifo_pnl(em_last)
    t5 = {}
    if pnl_first is not None:
        t5["first_2_days"] = {"n_fill": int(pnl_first.size),
                              "mean_t": float(pnl_first.mean()),
                              "wr": float((pnl_first > 0).mean() * 100),
                              "sharpe": safe_sharpe(pnl_first)}
    if pnl_last is not None:
        t5["last_3_days"] = {"n_fill": int(pnl_last.size),
                             "mean_t": float(pnl_last.mean()),
                             "wr": float((pnl_last > 0).mean() * 100),
                             "sharpe": safe_sharpe(pnl_last)}
    results["T5_halfsplit"] = t5

    # T6. Equity curve & DD
    log("T6. Equity curve")
    if cand_pnl is not None:
        # order trades by sample index (proxy for time)
        idx_sorted = np.where(cand_m & (-fifo43 != 0) & mask_fifo43)[0]
        # rebuild PnL with same ordering
        raw_all = -fifo43[idx_sorted]
        fm_all = raw_all != 0
        idx_filled = idx_sorted[fm_all]
        pnl_seq = raw_all[fm_all] - PASSIVE_COST
        equity = np.cumsum(pnl_seq)
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        results["T6_equity"] = {
            "n": int(pnl_seq.size),
            "total_t": float(pnl_seq.sum()),
            "total_$": float(pnl_seq.sum() * TICK_VALUE),
            "max_dd_t": float(dd.min()),
            "max_dd_$": float(dd.min() * TICK_VALUE),
            "max_dd_idx": int(np.argmin(dd)),
            "calmar_proxy": float(pnl_seq.sum() / abs(dd.min())) if abs(dd.min()) > 0 else float("nan"),
        }

    # T7. Without golden_ToD (any time)
    log("T7. Without ToD filter")
    em_no_tod = short_band_mask(5.0) & agree_15 & no_rev & vol_mid
    pnl_nt, _ = fifo_pnl(em_no_tod)
    if pnl_nt is not None:
        results["T7_no_tod"] = {
            "n_fill": int(pnl_nt.size), "mean_t": float(pnl_nt.mean()),
            "wr": float((pnl_nt > 0).mean() * 100), "sharpe": safe_sharpe(pnl_nt)
        }

    # T8. Top5% raw (no confluence at all)
    log("T8. Top5% raw")
    em_raw = short_band_mask(5.0)
    pnl_raw, _ = fifo_pnl(em_raw)
    if pnl_raw is not None:
        results["T8_top5_raw"] = {
            "n_fill": int(pnl_raw.size), "mean_t": float(pnl_raw.mean()),
            "wr": float((pnl_raw > 0).mean() * 100), "sharpe": safe_sharpe(pnl_raw)
        }

    # T9. Band sensitivity within full filter set + golden ToD
    log("T9. Band sensitivity")
    t9 = {}
    for pct in [0.5, 1.0, 2.0, 3.0, 5.0, 10.0, 20.0]:
        em_x = short_band_mask(pct) & agree_15 & no_rev & vol_mid & in_golden
        pnl_x, _ = fifo_pnl(em_x)
        if pnl_x is None:
            continue
        t9[f"Top{pct}"] = {"n_fill": int(pnl_x.size), "mean_t": float(pnl_x.mean()),
                           "wr": float((pnl_x > 0).mean() * 100), "sharpe": safe_sharpe(pnl_x)}
    results["T9_band_sweep"] = t9

    # T10. Per-day total $
    log("T10. Per-day $ summary")
    t10 = []
    for du in range(n_days):
        em_d = cand_em & (day_idx == du)
        pnl_d, _ = fifo_pnl(em_d)
        if pnl_d is None:
            t10.append({"day": du, "n": 0, "total_t": 0.0, "total_$": 0.0,
                        "max_drawdown_t": 0.0})
            continue
        eq = np.cumsum(pnl_d)
        peak = np.maximum.accumulate(eq)
        dd = eq - peak
        t10.append({"day": du, "n": int(pnl_d.size),
                    "total_t": float(pnl_d.sum()),
                    "total_$": float(pnl_d.sum() * TICK_VALUE),
                    "max_drawdown_t": float(dd.min())})
    results["T10_daily_dollars"] = t10

    # ─── Write report ───
    md = OUT / "PASS8.md"
    with open(md, "w") as f:
        f.write("# PASS 8 — Top5% Loose-Band Validation\n\n")
        f.write("Candidate: S_Top5% × agree_15 × no_reversal × golden-ToD × vol_mid\n\n")

        f.write("## T1. Per-Day Breakdown\n\n")
        f.write("| day | n_fill | mean_t | WR% | Sharpe | total_t |\n|---|---:|---:|---:|---:|---:|\n")
        for r in t1:
            f.write(f"| {r['day']} | {r['n_fill']} | {r['mean_t']:.3f} | {r['wr']:.1f} | "
                    f"{r['sharpe']:.2f} | {r['total_t']:.2f} |\n")
        f.write("\n")

        f.write("## T2. Leave-One-Day-Out CV\n\n")
        f.write("| Holdout | n_fill | mean_t | WR% | Sharpe | total_t |\n|---|---:|---:|---:|---:|---:|\n")
        for r in t2:
            f.write(f"| day {r['holdout']} | {r['n_fill']} | {r['mean_t']:.3f} | "
                    f"{r['wr']:.1f} | {r['sharpe']:.2f} | {r['total_t']:.2f} |\n")
        f.write("\n")

        f.write("## T3. ToD Bucket Coverage\n\n")
        f.write("| Bucket | n_fill | mean_t | WR% | Sharpe |\n|---|---:|---:|---:|---:|\n")
        for r in t3:
            f.write(f"| {r['bucket']} | {r['n_fill']} | {r['mean_t']:.3f} | "
                    f"{r['wr']:.1f} | {r['sharpe']:.2f} |\n")
        f.write("\n")

        if "T4_permutation" in results:
            r = results["T4_permutation"]
            f.write("## T4. Permutation Test (1000 random sign-shuffles)\n\n")
            f.write(f"- Observed Sharpe: **{r['observed_sharpe']:.3f}**\n")
            f.write(f"- Null mean: {r['null_mean']:.3f}\n")
            f.write(f"- Null 95th: {r['null_p95']:.3f}\n")
            f.write(f"- Null 99th: {r['null_p99']:.3f}\n")
            f.write(f"- **p-value = {r['p_value']:.4f}**\n\n")

        f.write("## T5. Half-Split\n\n")
        f.write("| Split | n_fill | mean_t | WR% | Sharpe |\n|---|---:|---:|---:|---:|\n")
        for k, v in results.get("T5_halfsplit", {}).items():
            f.write(f"| {k} | {v['n_fill']} | {v['mean_t']:.3f} | {v['wr']:.1f} | {v['sharpe']:.2f} |\n")
        f.write("\n")

        if "T6_equity" in results:
            r = results["T6_equity"]
            f.write("## T6. Equity Curve & Drawdown\n\n")
            f.write(f"- Total trades: {r['n']}\n")
            f.write(f"- Total PnL: **{r['total_t']:.1f} ticks = ${r['total_$']:.0f}**\n")
            f.write(f"- Max drawdown: **{r['max_dd_t']:.1f} ticks = ${r['max_dd_$']:.0f}**\n")
            f.write(f"- Calmar proxy (total / |maxDD|): {r['calmar_proxy']:.2f}\n\n")

        if "T7_no_tod" in results:
            r = results["T7_no_tod"]
            f.write("## T7. WITHOUT ToD filter (any time)\n\n")
            f.write(f"- n={r['n_fill']}, mean_t={r['mean_t']:.3f}, WR={r['wr']:.1f}%, Sharpe={r['sharpe']:.2f}\n\n")

        if "T8_top5_raw" in results:
            r = results["T8_top5_raw"]
            f.write("## T8. Top5% RAW (no confluence at all)\n\n")
            f.write(f"- n={r['n_fill']}, mean_t={r['mean_t']:.3f}, WR={r['wr']:.1f}%, Sharpe={r['sharpe']:.2f}\n\n")

        f.write("## T9. Band Sensitivity (full filters + golden ToD)\n\n")
        f.write("| Band | n_fill | mean_t | WR% | Sharpe |\n|---|---:|---:|---:|---:|\n")
        for k, v in results.get("T9_band_sweep", {}).items():
            f.write(f"| {k} | {v['n_fill']} | {v['mean_t']:.3f} | {v['wr']:.1f} | {v['sharpe']:.2f} |\n")
        f.write("\n")

        f.write("## T10. Per-Day Dollars (assuming 1 contract/fill)\n\n")
        f.write("| day | n | total_t | total_$ | max_dd_t |\n|---|---:|---:|---:|---:|\n")
        for r in results.get("T10_daily_dollars", []):
            f.write(f"| {r['day']} | {r['n']} | {r['total_t']:.2f} | ${r['total_$']:.0f} | {r['max_drawdown_t']:.2f} |\n")
        f.write("\n")

    with open(OUT / "pass8_results.json", "w") as f:
        json.dump(results, f, default=str, indent=2)
    log(f"PASS 8 complete — wrote {md}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL")
        log(traceback.format_exc())
        sys.exit(1)
