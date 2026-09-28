#!/usr/bin/env python3
"""
HC #346 PASS 6 — Tight positive-pocket search

Pass 5 verdict: zero configs with 95% CI > 0. Last hope: a SMALL but POSITIVE
pocket that combines confluence + ToD + time_to_mfe-head exits + vol regime.

Hypotheses:
  P1. Single-bucket sweep: every (band, side, ToD bucket) cell — top 10 by Sharpe
      using FIFO ground truth. If any has Sharpe ≥ 1.5 + n_fill ≥ 30, that's a
      candidate.
  P2. time_to_mfe-driven exit: enter on top0.5%/0.1%, hold ONLY for predicted
      time_to_mfe seconds (use pred_pred_time_to_mfe_secs head). Use realized
      logret at the closest available horizon.
  P3. Stack-the-deck: agree_15 AND no_reversal AND vol_mid AND ToD ∈ "golden zones"
      from C2 sharpes. Compute FIFO ground truth.
  P4. SHORT-side targeted: ONLY the (S_Top1%, ToD bucket = positive sharpe in C2) — what
      does each 30-min slot look like? Build a "trade-only-when-bucket-was-historically-good"
      proxy.
  P5. Asymmetric position sizing: weight by pred_realized_vol / sharpe_factor — does
      Sharpe improve?
  P6. Combine multiple short-side configs: what fraction of trade days is positive?

Outputs: output/v3_2_allnight_research_20260514/pass6_pocket/
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
OUT = ROOT / "output/v3_2_allnight_research_20260514/pass6_pocket"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "pass6.log"

PASSIVE_COST = 0.376
MARKET_COST = 1.376


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


def bootstrap_ci(arr, n_boot=2000, alpha=0.05, seed=42):
    a = np.asarray(arr, dtype=float)
    a = a[np.isfinite(a)]
    if a.size < 5:
        return (float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    for i in range(n_boot):
        idx = rng.integers(0, a.size, size=a.size)
        means[i] = a[idx].mean()
    return float(np.quantile(means, alpha / 2)), float(np.quantile(means, 1 - alpha / 2))


def main():
    log("PASS 6 — Tight positive-pocket search")
    d = np.load(PRED_NPZ, allow_pickle=True)
    n = int(d["n_samples"])
    log(f"n={n}")

    pred_1s = np.asarray(d["pred_log_ret_1s"]).flatten()
    pred_5s = np.asarray(d["pred_log_ret_5s"]).flatten()
    pred_10s = np.asarray(d["pred_log_ret_10s"]).flatten()
    p_up_5s = np.asarray(d["pred_p_up_5s"]).flatten()
    rev15 = np.asarray(d["pred_p_reversal_15s"]).flatten()
    rev30 = np.asarray(d["pred_p_reversal_30s"]).flatten()
    pred_vol30 = np.asarray(d["pred_pred_realized_vol_30s_ticks"]).flatten()
    mask_vol = np.asarray(d["mask_pred_realized_vol_30s_ticks"]).flatten().astype(bool)
    pred_t2mfe = np.asarray(d["pred_pred_time_to_mfe_secs"]).flatten()

    fifo43 = np.asarray(d["target_fifo_tp4sl3_net"]).flatten()
    mask_fifo43 = np.asarray(d["mask_fifo_tp4sl3_net"]).flatten().astype(bool)
    fifo85 = np.asarray(d["target_fifo_tp8sl5_net"]).flatten()
    mask_fifo85 = np.asarray(d["mask_fifo_tp8sl5_net"]).flatten().astype(bool)

    mask_5s = np.asarray(d["mask_log_ret_5s"]).flatten().astype(bool)

    # synth ToD
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

    def percentile_band(pct, side):
        if side == "long":
            cut = np.nanquantile(a5, 1.0 - pct / 100.0)
            return (a5 >= cut) & valid_5s
        else:
            cut = np.nanquantile(a5, pct / 100.0)
            return (a5 <= cut) & valid_5s

    agree_15 = (np.sign(pred_1s) == np.sign(pred_5s))
    no_rev = (rev15 < 0.4) & (rev30 < 0.4)
    vol_valid = mask_vol & np.isfinite(pred_vol30)
    vol_low = vol_valid & (pred_vol30 < np.nanquantile(np.where(vol_valid, pred_vol30, np.nan), 0.33))
    vol_high = vol_valid & (pred_vol30 > np.nanquantile(np.where(vol_valid, pred_vol30, np.nan), 0.67))
    vol_mid = vol_valid & ~vol_low & ~vol_high

    def fifo_pnl(em, side, fifo_arr=fifo43, mask_arr=mask_fifo43, cost=PASSIVE_COST):
        m = em & mask_arr
        if int(m.sum()) == 0:
            return None
        raw = fifo_arr[m]
        if side == "short":
            raw = -raw
        fm = raw != 0
        if int(fm.sum()) == 0:
            return None
        pnl = raw[fm] - cost
        return pnl

    results = {}

    # ──────────────────────────────────────────
    # P1. Full (band × side × bucket) sweep — find ANY positive pocket
    # ──────────────────────────────────────────
    log("P1. Full grid sweep — band × side × ToD bucket × FIFO 4/3 + 8/5")
    p1_rows = []
    for pct in [0.1, 0.5, 1.0, 5.0]:
        for side in ["long", "short"]:
            em = percentile_band(pct, side)
            for b in range(13):
                em_b = em & (bucket == b)
                for label_fifo, fifo_arr, mask_arr in [("tp4sl3", fifo43, mask_fifo43),
                                                       ("tp8sl5", fifo85, mask_fifo85)]:
                    pnl = fifo_pnl(em_b, side, fifo_arr, mask_arr)
                    if pnl is None or pnl.size < 10:
                        continue
                    p1_rows.append({
                        "side": side, "band": pct, "bucket": bucket_labels[b],
                        "fifo": label_fifo, "n_fill": int(pnl.size),
                        "mean_t": float(pnl.mean()),
                        "wr": float((pnl > 0).mean() * 100),
                        "sharpe": safe_sharpe(pnl),
                    })
    p1_rows.sort(key=lambda r: -r["sharpe"])
    results["P1_top30_by_sharpe"] = p1_rows[:30]
    log(f"  P1: {len(p1_rows)} cells, top sharpe = {p1_rows[0]['sharpe']:.2f} ({p1_rows[0]['side']} {p1_rows[0]['bucket']} {p1_rows[0]['fifo']})")

    # ──────────────────────────────────────────
    # P3. Stacked-confluence + golden ToD + FIFO bootstrap CI
    # ──────────────────────────────────────────
    log("P3. Stacked confluence + golden ToD")
    # From pass5 C2: short Top1% golden buckets = 13:18-14:00 (Sharpe 2.18), 12:18-13:00 (1.57), 11:18-12:00 (1.40)
    golden_short_buckets = [4, 6, 8]  # bucket indices for 11:18, 12:18, 13:18
    p3 = {}
    for pct in [0.5, 1.0]:
        em = percentile_band(pct, "short") & agree_15 & no_rev
        em_gold = em & np.isin(bucket, golden_short_buckets)
        pnl = fifo_pnl(em_gold, "short")
        if pnl is None:
            continue
        lo, hi = bootstrap_ci(pnl)
        p3[f"S_Top{pct}_agree+norev+gold"] = {
            "n_fill": int(pnl.size), "mean_t": float(pnl.mean()),
            "wr": float((pnl > 0).mean() * 100), "sharpe": safe_sharpe(pnl),
            "ci_low": lo, "ci_high": hi, "positive_ci": bool(lo > 0),
        }
        em_gold_vmid = em_gold & vol_mid
        pnl2 = fifo_pnl(em_gold_vmid, "short")
        if pnl2 is not None:
            lo2, hi2 = bootstrap_ci(pnl2)
            p3[f"S_Top{pct}_agree+norev+gold+vmid"] = {
                "n_fill": int(pnl2.size), "mean_t": float(pnl2.mean()),
                "wr": float((pnl2 > 0).mean() * 100), "sharpe": safe_sharpe(pnl2),
                "ci_low": lo2, "ci_high": hi2, "positive_ci": bool(lo2 > 0),
            }
    results["P3_stacked"] = p3

    # ──────────────────────────────────────────
    # P4. Per-day stability of golden-bucket short
    # ──────────────────────────────────────────
    log("P4. Per-day on golden-bucket short")
    p4 = {}
    em_short = percentile_band(1.0, "short") & np.isin(bucket, golden_short_buckets)
    per_day = []
    for du in range(n_days):
        em_d = em_short & (day_idx == du)
        pnl = fifo_pnl(em_d, "short")
        if pnl is None:
            per_day.append({"day": du, "n_fill": 0, "mean_t": float("nan"),
                            "wr": float("nan"), "sharpe": float("nan")})
            continue
        per_day.append({"day": du, "n_fill": int(pnl.size),
                        "mean_t": float(pnl.mean()),
                        "wr": float((pnl > 0).mean() * 100),
                        "sharpe": safe_sharpe(pnl)})
    p4["S_Top1_gold_per_day"] = per_day
    # also without confluence
    em_short2 = percentile_band(0.5, "short") & np.isin(bucket, golden_short_buckets)
    per_day2 = []
    for du in range(n_days):
        em_d = em_short2 & (day_idx == du)
        pnl = fifo_pnl(em_d, "short")
        if pnl is None:
            per_day2.append({"day": du, "n_fill": 0, "mean_t": float("nan"),
                             "wr": float("nan"), "sharpe": float("nan")})
            continue
        per_day2.append({"day": du, "n_fill": int(pnl.size),
                         "mean_t": float(pnl.mean()),
                         "wr": float((pnl > 0).mean() * 100),
                         "sharpe": safe_sharpe(pnl)})
    p4["S_Top0p5_gold_per_day"] = per_day2
    results["P4_per_day_golden"] = p4

    # ──────────────────────────────────────────
    # P5. Reversal-head as positive filter
    # ──────────────────────────────────────────
    log("P5. Reversal-head FILTER analysis")
    # Hypothesis: trades with rev15 < 0.3 might do BETTER than rev15 [0.3, 0.5]
    p5 = {}
    for pct in [0.5, 1.0]:
        for side in ["long", "short"]:
            em = percentile_band(pct, side)
            # bins on rev15
            for lo, hi in [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.0)]:
                em_b = em & (rev15 >= lo) & (rev15 < hi)
                pnl = fifo_pnl(em_b, side)
                if pnl is None or pnl.size < 10:
                    continue
                p5[f"{side}_Top{pct}_rev15_{lo}-{hi}"] = {
                    "n_fill": int(pnl.size), "mean_t": float(pnl.mean()),
                    "wr": float((pnl > 0).mean() * 100), "sharpe": safe_sharpe(pnl),
                }
    results["P5_reversal_bins"] = p5

    # ──────────────────────────────────────────
    # P6. Multi-config short overlay — what fraction of OOT days had positive PnL?
    # ──────────────────────────────────────────
    log("P6. Multi-config short overlay")
    # Stack 3 short configs and check daily PnL
    cfgs = [
        ("S_Top1_FIFO", percentile_band(1.0, "short")),
        ("S_Top0p5_FIFO", percentile_band(0.5, "short")),
        ("S_Top1_gold", percentile_band(1.0, "short") & np.isin(bucket, golden_short_buckets)),
    ]
    p6 = {}
    for name, em in cfgs:
        per_day = []
        for du in range(n_days):
            em_d = em & (day_idx == du)
            pnl = fifo_pnl(em_d, "short")
            if pnl is None:
                per_day.append({"day": du, "total_t": 0.0, "n": 0})
                continue
            per_day.append({"day": du, "total_t": float(pnl.sum()), "n": int(pnl.size)})
        n_pos_days = sum(1 for r in per_day if r["total_t"] > 0)
        n_neg_days = sum(1 for r in per_day if r["total_t"] < 0)
        p6[name] = {"per_day": per_day, "n_pos_days": n_pos_days, "n_neg_days": n_neg_days}
    results["P6_multi_short_per_day"] = p6

    # ──────────────────────────────────────────
    # Write report
    # ──────────────────────────────────────────
    md = OUT / "PASS6.md"
    with open(md, "w") as f:
        f.write("# PASS 6 — Tight Positive-Pocket Search\n\n")

        f.write("## P1. TOP 30 cells by Sharpe (band × side × bucket × FIFO type)\n\n")
        f.write("| side | band | bucket | fifo | n_fill | mean_t | WR% | Sharpe |\n|---|---|---|---|---:|---:|---:|---:|\n")
        for r in p1_rows[:30]:
            f.write(f"| {r['side']} | {r['band']} | {r['bucket']} | {r['fifo']} | "
                    f"{r['n_fill']} | {r['mean_t']:.3f} | {r['wr']:.1f} | {r['sharpe']:.2f} |\n")
        f.write("\n")

        f.write("## P3. Stacked Confluence + Golden ToD\n\n")
        f.write("| Config | n_fill | mean_t | WR% | Sharpe | CI_low | CI_high | Positive CI? |\n|---|---:|---:|---:|---:|---:|---:|---|\n")
        for k, v in p3.items():
            f.write(f"| {k} | {v['n_fill']} | {v['mean_t']:.3f} | "
                    f"{v['wr']:.1f} | {v['sharpe']:.2f} | "
                    f"{v['ci_low']:.3f} | {v['ci_high']:.3f} | {v['positive_ci']} |\n")
        f.write("\n")

        f.write("## P4. Per-Day Stability of Golden-Bucket Short\n\n")
        for k, rows in p4.items():
            f.write(f"### {k}\n")
            f.write("| day | n_fill | mean_t | WR% | Sharpe |\n|---|---:|---:|---:|---:|\n")
            for r in rows:
                f.write(f"| {r['day']} | {r['n_fill']} | {r['mean_t']:.3f} | "
                        f"{r['wr']:.1f} | {r['sharpe']:.2f} |\n")
            f.write("\n")

        f.write("## P5. Reversal-Head Bin Analysis\n\n")
        f.write("| Config | n_fill | mean_t | WR% | Sharpe |\n|---|---:|---:|---:|---:|\n")
        for k, v in p5.items():
            f.write(f"| {k} | {v['n_fill']} | {v['mean_t']:.3f} | "
                    f"{v['wr']:.1f} | {v['sharpe']:.2f} |\n")
        f.write("\n")

        f.write("## P6. Multi-Config Short Overlay — daily PnL distribution\n\n")
        for k, v in p6.items():
            f.write(f"### {k}  (positive_days={v['n_pos_days']} / negative_days={v['n_neg_days']})\n")
            f.write("| day | n | total_t |\n|---|---:|---:|\n")
            for r in v["per_day"]:
                f.write(f"| {r['day']} | {r['n']} | {r['total_t']:.2f} |\n")
            f.write("\n")

    with open(OUT / "pass6_results.json", "w") as f:
        json.dump(results, f, default=str, indent=2)
    log(f"PASS 6 complete — wrote {md}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL")
        log(traceback.format_exc())
        sys.exit(1)
