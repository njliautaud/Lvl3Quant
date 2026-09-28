#!/usr/bin/env python3
"""
HC #345 v3 follow-up — Confluence + ToD slicing on the 4 SURVIVORS from
v32_per_head_permutation_test.

The 4 survivors at p < 0.05:
    A) log_ret_60s   SHORT Top0.5%   n=76   obs +1.119 t/fill
    B) log_ret_60s   SHORT Top1%     n=137  obs +0.803
    C) log_ret_60s_q10 SHORT Top10%  n=171  obs +0.587
    D) log_ret_1s    SHORT Top0.1%   n=35   obs +1.390

Questions:
  1. CONFLUENCE: when A AND D both fire (60s + 1s SHORT agreement at top bands),
     is the joint signal stronger?
  2. TOD: which ToD bucket concentrates the survivor fills? Same 11:18-12:00
     pattern as 8-pass research, or different?
  3. Per-day: is the edge 5-day stable or 1-day artifact?
  4. Survives commission + spread? (passive vs market order)

Output:
  output/v3_2_survivor_confluence_tod_20260514/
    confluence_results.json
    tod_distribution.csv
    per_day.csv
    SUMMARY.md

NO trainer code modified. Pure analysis. HC #307D.
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"
OUT_DIR = ROOT / "output/v3_2_survivor_confluence_tod_20260514"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "build.log"

COMM = 0.376


def log(msg: str):
    ts = datetime.utcnow().isoformat(timespec="seconds")
    print(f"[{ts}Z] {msg}", flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"[{ts}Z] {msg}\n")


def select_top_band(score: np.ndarray, side_sign: int, band_pct: float) -> np.ndarray:
    """Return boolean mask of top-band side-correct events."""
    sign = np.sign(score)
    magnitude = np.abs(score)
    side_mask = sign == side_sign
    if side_mask.sum() < 10:
        return np.zeros_like(side_mask)
    mag_in_side = magnitude[side_mask]
    k = max(1, int(side_mask.sum() * band_pct))
    if k < 5:
        return np.zeros_like(side_mask)
    thr = np.partition(mag_in_side, -k)[-k]
    return side_mask & (magnitude >= thr)


def stats(pnl: np.ndarray) -> dict:
    if len(pnl) < 2:
        return {"n": int(len(pnl)), "mean": float("nan"), "median": float("nan"),
                "sharpe": float("nan"), "wr_pos": float("nan"),
                "passive_net": float("nan"), "market_net": float("nan")}
    return {
        "n": int(len(pnl)),
        "mean": float(pnl.mean()),
        "median": float(np.median(pnl)),
        "sharpe": float(pnl.mean() / max(pnl.std(), 1e-9)),
        "wr_pos": float((pnl > 0).mean()),
        "passive_net": float(pnl.mean() - COMM),
        "market_net": float(pnl.mean() - COMM - 1.0),
    }


def main():
    t0 = time.time()
    log("V32 SURVIVOR CONFLUENCE + ToD ANALYSIS START")
    log(f"Loading {PRED_NPZ}")
    d = np.load(PRED_NPZ)

    n = int(d["n_samples"])
    log(f"n={n:,}")

    # Score arrays
    score_60s = d["pred_log_ret_60s"].astype(np.float32)
    score_60s_q10 = d["pred_log_ret_60s_q10"].astype(np.float32)
    score_1s = d["pred_log_ret_1s"].astype(np.float32)
    fifo = d["target_fifo_tp4sl3_net"].astype(np.float32)
    fifo_mask = d["mask_fifo_tp4sl3_net"].astype(bool)

    # ---- Define each survivor ----
    sel_A = select_top_band(score_60s, -1, 0.005)        # log_ret_60s SHORT Top0.5%
    sel_B = select_top_band(score_60s, -1, 0.010)        # log_ret_60s SHORT Top1%
    sel_C = select_top_band(score_60s_q10, -1, 0.10)     # log_ret_60s_q10 SHORT Top10%
    sel_D = select_top_band(score_1s, -1, 0.001)         # log_ret_1s SHORT Top0.1%

    log(f"  A=log_ret_60s SHORT Top0.5%:  {sel_A.sum():,} signals")
    log(f"  B=log_ret_60s SHORT Top1%:    {sel_B.sum():,} signals")
    log(f"  C=log_ret_60s_q10 SHORT T10%: {sel_C.sum():,} signals")
    log(f"  D=log_ret_1s SHORT Top0.1%:   {sel_D.sum():,} signals")

    # PnL helper for a selection mask
    def pnl_for(sel: np.ndarray) -> np.ndarray:
        return -fifo[sel & fifo_mask]  # short side: PnL = -tp4_net

    # ---- 1. INDIVIDUAL VERIFICATION ----
    indiv = {
        "A_log_ret_60s_SHORT_Top0.5": stats(pnl_for(sel_A)),
        "B_log_ret_60s_SHORT_Top1":   stats(pnl_for(sel_B)),
        "C_log_ret_60s_q10_SHORT_Top10": stats(pnl_for(sel_C)),
        "D_log_ret_1s_SHORT_Top0.1":  stats(pnl_for(sel_D)),
    }
    log("INDIVIDUAL VERIFICATION:")
    for k, v in indiv.items():
        log(f"  {k}: n={v['n']} mean={v['mean']:.3f} sharpe={v['sharpe']:.2f} WR={v['wr_pos']*100:.1f}% passive_net={v['passive_net']:.3f}")

    # ---- 2. PAIRWISE CONFLUENCE ----
    confl = {}
    pairs = [
        ("A", sel_A, "B", sel_B),
        ("A", sel_A, "C", sel_C),
        ("A", sel_A, "D", sel_D),
        ("B", sel_B, "C", sel_C),
        ("B", sel_B, "D", sel_D),
        ("C", sel_C, "D", sel_D),
    ]
    log("PAIRWISE CONFLUENCE (intersection):")
    for n1, s1, n2, s2 in pairs:
        sel = s1 & s2
        if sel.sum() < 5:
            log(f"  {n1}×{n2}: n_intersect={sel.sum()} (TOO SMALL)")
            confl[f"{n1}x{n2}"] = {"n": int(sel.sum())}
            continue
        st = stats(pnl_for(sel))
        st["n_intersect_signals"] = int(sel.sum())
        confl[f"{n1}x{n2}"] = st
        log(f"  {n1}×{n2}: n_intersect_sigs={sel.sum()} n_fills={st['n']} mean={st['mean']:.3f} "
            f"sharpe={st['sharpe']:.2f} WR={st['wr_pos']*100:.1f}% passive_net={st['passive_net']:.3f}")

    # All-4 intersection
    sel_all = sel_A & sel_B & sel_C & sel_D
    if sel_all.sum() >= 5:
        st = stats(pnl_for(sel_all))
        st["n_intersect_signals"] = int(sel_all.sum())
        confl["AxBxCxD"] = st
        log(f"  A×B×C×D: n_sigs={sel_all.sum()} n_fills={st['n']} mean={st['mean']:.3f} sharpe={st['sharpe']:.2f}")
    else:
        confl["AxBxCxD"] = {"n": int(sel_all.sum())}

    # ---- 3. TOD ANALYSIS ----
    # No explicit ToD field in npz. We approximate by event-index (events span the
    # RTH session ~6.5h × 5 days). Per the all-night research, the "11:18-12:00"
    # bucket showed up. We slice by 30-min buckets across the full event index
    # using uniform-event-rate assumption (NOT real ToD — caveat).
    # An OOT day spans roughly 9:30-16:00 ET = 390 min. With 5 days and ~241k events,
    # ~48270 events/day, ~123 events/min, ~3700 events per 30-min bucket.
    # For a more honest mapping we'd need timestamps in the npz — TODO use file_indexes.
    log("TOD ANALYSIS (approx via event-index buckets, 30min ≈ 3700 events/day):")
    BUCKETS = [
        ("09:30-10:00", 0,   3700),
        ("10:00-10:30", 3700, 7400),
        ("10:30-11:00", 7400, 11100),
        ("11:00-11:30", 11100, 14800),
        ("11:30-12:00", 14800, 18500),
        ("12:00-12:30", 18500, 22200),
        ("12:30-13:00", 22200, 25900),
        ("13:00-13:30", 25900, 29600),
        ("13:30-14:00", 29600, 33300),
        ("14:00-14:30", 33300, 37000),
        ("14:30-15:00", 37000, 40700),
        ("15:00-15:30", 40700, 44400),
        ("15:30-16:00", 44400, 48270),
    ]
    chunk = n // 5
    tod_results = {}
    for bn, lo, hi in BUCKETS:
        # For each of 5 days, extract the bucket
        bucket_mask = np.zeros(n, dtype=bool)
        for di in range(5):
            day_start = di * chunk
            day_end = day_start + chunk if di < 4 else n
            day_lo = day_start + lo
            day_hi = min(day_start + hi, day_end)
            bucket_mask[day_lo:day_hi] = True

        # Stats per survivor + per ToD bucket
        bd = {}
        for label, sel in [("A", sel_A), ("B", sel_B), ("C", sel_C), ("D", sel_D)]:
            sel_b = sel & bucket_mask
            n_sigs = int(sel_b.sum())
            pnl = pnl_for(sel_b)
            if len(pnl) >= 5:
                bd[label] = {"n_sigs": n_sigs, "n_fills": len(pnl), "mean": float(pnl.mean()),
                             "wr": float((pnl > 0).mean()), "passive_net": float(pnl.mean() - COMM)}
            else:
                bd[label] = {"n_sigs": n_sigs, "n_fills": len(pnl)}
        tod_results[bn] = bd
        log(f"  {bn}: A_n={bd['A'].get('n_fills', 0):4d} mean={bd['A'].get('mean', float('nan')):+.3f}"
            f" | B_n={bd['B'].get('n_fills', 0):4d} mean={bd['B'].get('mean', float('nan')):+.3f}"
            f" | C_n={bd['C'].get('n_fills', 0):4d} mean={bd['C'].get('mean', float('nan')):+.3f}"
            f" | D_n={bd['D'].get('n_fills', 0):4d} mean={bd['D'].get('mean', float('nan')):+.3f}")

    # ---- 4. PER-DAY ROBUSTNESS ----
    log("PER-DAY ROBUSTNESS:")
    per_day = {}
    for di in range(5):
        day_start = di * chunk
        day_end = day_start + chunk if di < 4 else n
        day_mask = np.zeros(n, dtype=bool)
        day_mask[day_start:day_end] = True
        date_str = str(d["oot_dates"][di])
        dd = {}
        for label, sel in [("A", sel_A), ("B", sel_B), ("C", sel_C), ("D", sel_D)]:
            sel_d = sel & day_mask
            pnl = pnl_for(sel_d)
            if len(pnl) >= 1:
                dd[label] = {"n_fills": len(pnl), "mean": float(pnl.mean()) if len(pnl) >= 2 else float("nan"),
                             "total_t": float(pnl.sum())}
            else:
                dd[label] = {"n_fills": 0, "mean": float("nan"), "total_t": 0.0}
        per_day[date_str] = dd
        log(f"  Day {di} ({date_str}): A_n={dd['A']['n_fills']:3d} total={dd['A']['total_t']:+.1f}t"
            f" | B_n={dd['B']['n_fills']:3d} total={dd['B']['total_t']:+.1f}t"
            f" | C_n={dd['C']['n_fills']:3d} total={dd['C']['total_t']:+.1f}t"
            f" | D_n={dd['D']['n_fills']:3d} total={dd['D']['total_t']:+.1f}t")

    # ---- SAVE ----
    with open(OUT_DIR / "confluence_results.json", "w") as f:
        json.dump({"individual": indiv, "confluence": confl,
                   "tod_buckets": tod_results, "per_day": per_day,
                   "generated": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    log(f"Wrote {OUT_DIR / 'confluence_results.json'}")

    # ---- MARKDOWN SUMMARY ----
    sm = OUT_DIR / "SUMMARY.md"
    with open(sm, "w") as f:
        f.write("# v3.2 Survivor Confluence + ToD Analysis\n\n")
        f.write(f"_{datetime.utcnow().isoformat(timespec='seconds')}Z_\n\n")
        f.write("## Survivor letter codes\n")
        f.write("- **A** = log_ret_60s SHORT Top0.5%\n")
        f.write("- **B** = log_ret_60s SHORT Top1%  (B is a superset of A)\n")
        f.write("- **C** = log_ret_60s_q10 SHORT Top10%  (different head: q10 quantile)\n")
        f.write("- **D** = log_ret_1s SHORT Top0.1%\n\n")
        f.write("## 1. Individual verification\n\n")
        f.write("| ID | Head | Side | Band | n | mean t/fill | Sharpe | WR | passive_net |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for k, v in indiv.items():
            f.write(f"| {k.split('_')[0]} | {k} | SHORT | - | {v['n']} | "
                    f"{v['mean']:+.3f} | {v['sharpe']:.2f} | {v['wr_pos']*100:.1f}% | {v['passive_net']:+.3f} |\n")
        f.write("\n## 2. Pairwise confluence (do agreeing signals stack?)\n\n")
        f.write("| Pair | n_sigs | n_fills | mean | Sharpe | WR | passive_net |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for k, v in confl.items():
            if "n_intersect_signals" in v:
                f.write(f"| {k} | {v['n_intersect_signals']} | {v['n']} | {v['mean']:+.3f} | "
                        f"{v['sharpe']:.2f} | {v['wr_pos']*100:.1f}% | {v['passive_net']:+.3f} |\n")
            else:
                f.write(f"| {k} | n={v['n']} | TOO SMALL | - | - | - | - |\n")
        f.write("\n## 3. ToD distribution (approximate via event-index buckets)\n\n")
        f.write("**CAVEAT**: NPZ does not carry per-event timestamps. ToD is approximated by event "
                "index assuming uniform RTH event rate. Real ToD requires re-running deep-sim with "
                "timestamps preserved.\n\n")
        f.write("| Bucket | A_fills | A_mean | B_fills | B_mean | C_fills | C_mean | D_fills | D_mean |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for bn, bd in tod_results.items():
            f.write(f"| {bn} | "
                    f"{bd['A'].get('n_fills',0)} | {bd['A'].get('mean', float('nan')):+.3f} | "
                    f"{bd['B'].get('n_fills',0)} | {bd['B'].get('mean', float('nan')):+.3f} | "
                    f"{bd['C'].get('n_fills',0)} | {bd['C'].get('mean', float('nan')):+.3f} | "
                    f"{bd['D'].get('n_fills',0)} | {bd['D'].get('mean', float('nan')):+.3f} |\n")
        f.write("\n## 4. Per-day robustness\n\n")
        f.write("| Date | A_n | A_total | B_n | B_total | C_n | C_total | D_n | D_total |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for date, dd in per_day.items():
            f.write(f"| {date} | "
                    f"{dd['A']['n_fills']} | {dd['A']['total_t']:+.1f} | "
                    f"{dd['B']['n_fills']} | {dd['B']['total_t']:+.1f} | "
                    f"{dd['C']['n_fills']} | {dd['C']['total_t']:+.1f} | "
                    f"{dd['D']['n_fills']} | {dd['D']['total_t']:+.1f} |\n")
        f.write("\n## Interpretation guide\n\n")
        f.write("- If **B is essentially the same as A**: the signal isn't tightly localized → loose-band tradable\n")
        f.write("- If **A×D > A**: confluence with the 1s-horizon adds value (different timescales agree)\n")
        f.write("- If **all fills concentrate in ToD bucket X**: that's the actual edge window — static rule\n")
        f.write("- If **per-day shows 1 day with 80% of fills**: same day-2-only artifact as the all-night research\n")
    log(f"Wrote {sm}")
    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
