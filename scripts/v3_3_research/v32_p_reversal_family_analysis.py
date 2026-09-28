#!/usr/bin/env python3
"""
HC #345 v3 follow-up — Confluence + ToD + per-day analysis on the
p_reversal_15s SHORT family + log_ret_60s SHORT loose-band, the BIG NEW
survivors from the top-120 permutation sweep.

The 13-survivor sweep surfaced these large-n new candidates beyond the
original A/B/C/D:
  E) p_reversal_15s SHORT Top20%   n=8143  obs +0.604  passive_net +0.228
  F) p_reversal_15s SHORT Top10%   n=4431  obs +0.667  passive_net +0.291
  G) p_reversal_15s SHORT Top5%    n=2173  obs +0.647  passive_net +0.271
  H) p_reversal_15s SHORT Top1%    n=562   obs +0.601  passive_net +0.225
  I) log_ret_60s    SHORT Top5%    n=531   obs +0.511  passive_net +0.135
  J) log_ret_60s    SHORT Top10%   n=998   obs +0.539  passive_net +0.163
  K) log_ret_60s    SHORT Top20%   n=1989  obs +0.547  passive_net +0.171

Questions:
  1. Are F and original A independent or overlapping? (If F captures A, F is
     strictly better — bigger n, similar passive_net dollars.)
  2. CONFLUENCE F × A: when both fire, is per-fill edge meaningfully higher?
  3. Per-day: is F's edge 5-day stable like A?
  4. ToD: F's huge n_fills should let us identify peak ToD windows.

NO trainer code modified. Pure analysis. HC #307D. Output:
  output/v3_2_p_reversal_family_20260514/SUMMARY.md
"""
from __future__ import annotations

import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"
OUT_DIR = ROOT / "output/v3_2_p_reversal_family_20260514"
OUT_DIR.mkdir(parents=True, exist_ok=True)
LOG_PATH = OUT_DIR / "build.log"

COMM = 0.376


def log(msg: str):
    ts = datetime.utcnow().isoformat(timespec="seconds")
    print(f"[{ts}Z] {msg}", flush=True)
    with open(LOG_PATH, "a") as f:
        f.write(f"[{ts}Z] {msg}\n")


def select_top_band(score: np.ndarray, side_sign: int, band_pct: float) -> np.ndarray:
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
        return {"n": int(len(pnl)), "mean": float("nan"), "sharpe": float("nan"),
                "wr": float("nan"), "passive_net": float("nan"), "total_t": float(pnl.sum()) if len(pnl) else 0.0,
                "dollar_pnl": float("nan")}
    mean = float(pnl.mean())
    return {
        "n": int(len(pnl)),
        "mean": mean,
        "sharpe": float(mean / max(pnl.std(), 1e-9)),
        "wr": float((pnl > 0).mean()),
        "passive_net": mean - COMM,
        "total_t": float(pnl.sum()),
        "dollar_pnl": float(len(pnl) * (mean - COMM) * 12.50),  # net of $4.70 RT comm
    }


def main():
    t0 = time.time()
    log("V32 p_reversal_15s FAMILY ANALYSIS START")
    log(f"Loading {PRED_NPZ}")
    d = np.load(PRED_NPZ)
    n = int(d["n_samples"])
    log(f"n={n:,}")

    # p_reversal head: prediction is probability — center around 0.5
    score_pr15 = d["pred_p_reversal_15s"].astype(np.float32) - 0.5
    score_60s = d["pred_log_ret_60s"].astype(np.float32)
    score_1s = d["pred_log_ret_1s"].astype(np.float32)

    fifo = d["target_fifo_tp4sl3_net"].astype(np.float32)
    fifo_mask = d["mask_fifo_tp4sl3_net"].astype(bool)

    # Define survivors
    sels = {
        "A_60s_SHORT_T0.5%": select_top_band(score_60s, -1, 0.005),
        "F_pr15s_SHORT_T10%": select_top_band(score_pr15, -1, 0.10),
        "G_pr15s_SHORT_T5%":  select_top_band(score_pr15, -1, 0.05),
        "H_pr15s_SHORT_T1%":  select_top_band(score_pr15, -1, 0.01),
        "E_pr15s_SHORT_T20%": select_top_band(score_pr15, -1, 0.20),
        "J_60s_SHORT_T10%":   select_top_band(score_60s, -1, 0.10),
        "I_60s_SHORT_T5%":    select_top_band(score_60s, -1, 0.05),
        "K_60s_SHORT_T20%":   select_top_band(score_60s, -1, 0.20),
        "D_1s_SHORT_T0.1%":   select_top_band(score_1s, -1, 0.001),
    }

    log("Selection sizes:")
    for k, v in sels.items():
        log(f"  {k}: {int(v.sum()):>6,} signals")

    def pnl_for(sel: np.ndarray) -> np.ndarray:
        return -fifo[sel & fifo_mask]  # SHORT side

    # ---- 1. INDIVIDUAL ----
    log("\nINDIVIDUAL VERIFICATION:")
    indiv = {}
    for name, sel in sels.items():
        st = stats(pnl_for(sel))
        indiv[name] = st
        log(f"  {name}: n={st['n']:>5} mean={st['mean']:+.3f} sharpe={st['sharpe']:.2f} "
            f"WR={st['wr']*100:.1f}% passive_net={st['passive_net']:+.3f} ${st['dollar_pnl']:+,.0f}")

    # ---- 2. CONFLUENCE: A × F (60s SHORT × p_reversal SHORT) ----
    log("\nCONFLUENCE PAIRS (intersection):")
    confluence_pairs = [
        ("A_60s_SHORT_T0.5%", "F_pr15s_SHORT_T10%"),
        ("A_60s_SHORT_T0.5%", "G_pr15s_SHORT_T5%"),
        ("I_60s_SHORT_T5%",   "F_pr15s_SHORT_T10%"),
        ("I_60s_SHORT_T5%",   "G_pr15s_SHORT_T5%"),
        ("J_60s_SHORT_T10%",  "F_pr15s_SHORT_T10%"),
        ("K_60s_SHORT_T20%",  "E_pr15s_SHORT_T20%"),
        ("D_1s_SHORT_T0.1%",  "F_pr15s_SHORT_T10%"),
    ]
    confl = {}
    for n1, n2 in confluence_pairs:
        s1, s2 = sels[n1], sels[n2]
        sel = s1 & s2
        if sel.sum() < 5:
            log(f"  {n1.split('_')[0]}×{n2.split('_')[0]}: n_intersect={sel.sum()} (TOO SMALL)")
            continue
        st = stats(pnl_for(sel))
        st["intersect_signals"] = int(sel.sum())
        st["frac_of_smaller"] = float(sel.sum() / min(s1.sum(), s2.sum()))
        key = f"{n1.split('_')[0]}x{n2.split('_')[0]}"
        confl[key] = st
        log(f"  {key}: n_sigs={sel.sum()} n_fills={st['n']} mean={st['mean']:+.3f} "
            f"sharpe={st['sharpe']:.2f} WR={st['wr']*100:.1f}% passive_net={st['passive_net']:+.3f} "
            f"${st['dollar_pnl']:+,.0f} (frac_of_smaller={st['frac_of_smaller']:.2f})")

    # ---- 3. OVERLAP MATRIX (does F encompass A?) ----
    log("\nOVERLAP MATRIX (Jaccard of signal sets):")
    keys = list(sels.keys())
    overlap = {}
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            inter = (sels[k1] & sels[k2]).sum()
            union = (sels[k1] | sels[k2]).sum()
            jacc = float(inter / max(union, 1))
            overlap[f"{k1.split('_')[0]}_vs_{k2.split('_')[0]}"] = {
                "intersect": int(inter),
                "union": int(union),
                "jaccard": jacc,
                "n1": int(sels[k1].sum()),
                "n2": int(sels[k2].sum()),
            }
    # Print key overlaps
    for k in ["A_vs_F", "A_vs_G", "F_vs_G", "F_vs_J", "I_vs_F", "D_vs_F"]:
        if k in overlap:
            o = overlap[k]
            log(f"  {k}: inter={o['intersect']} jacc={o['jaccard']:.3f} "
                f"(n1={o['n1']}, n2={o['n2']})")

    # ---- 4. PER-DAY ROBUSTNESS ----
    log("\nPER-DAY ROBUSTNESS (5 OOT days):")
    chunk = n // 5
    per_day = {}
    for di in range(5):
        day_lo = di * chunk
        day_hi = day_lo + chunk if di < 4 else n
        day_mask = np.zeros(n, dtype=bool)
        day_mask[day_lo:day_hi] = True
        date_str = str(d["oot_dates"][di])
        dd = {}
        for name, sel in sels.items():
            pnl = pnl_for(sel & day_mask)
            if len(pnl) >= 1:
                dd[name] = {"n_fills": int(len(pnl)),
                            "mean": float(pnl.mean()) if len(pnl) >= 2 else float("nan"),
                            "total_t": float(pnl.sum()),
                            "dollar_pnl": float(len(pnl) * (pnl.mean() - COMM) * 12.50) if len(pnl) >= 2 else 0.0}
            else:
                dd[name] = {"n_fills": 0, "mean": float("nan"), "total_t": 0.0, "dollar_pnl": 0.0}
        per_day[date_str] = dd
        # Compact print of headline survivors
        log(f"  Day{di} {date_str}: "
            f"A_n={dd['A_60s_SHORT_T0.5%']['n_fills']:>3} ${dd['A_60s_SHORT_T0.5%']['dollar_pnl']:+,.0f} | "
            f"F_n={dd['F_pr15s_SHORT_T10%']['n_fills']:>4} ${dd['F_pr15s_SHORT_T10%']['dollar_pnl']:+,.0f} | "
            f"G_n={dd['G_pr15s_SHORT_T5%']['n_fills']:>4} ${dd['G_pr15s_SHORT_T5%']['dollar_pnl']:+,.0f}")

    # ---- 5. ToD APPROXIMATION (event-index buckets) ----
    log("\nToD ANALYSIS (approx via event index):")
    BUCKETS = [
        ("09:30-10:30", 0,    7400),
        ("10:30-11:30", 7400, 14800),
        ("11:30-12:30", 14800, 22200),
        ("12:30-13:30", 22200, 29600),
        ("13:30-14:30", 29600, 37000),
        ("14:30-15:30", 37000, 44400),
        ("15:30-16:00", 44400, 48270),
    ]
    tod_results = {}
    for bn, lo, hi in BUCKETS:
        bucket_mask = np.zeros(n, dtype=bool)
        for di in range(5):
            day_start = di * chunk
            day_end = day_start + chunk if di < 4 else n
            day_lo = day_start + lo
            day_hi = min(day_start + hi, day_end)
            bucket_mask[day_lo:day_hi] = True
        bd = {}
        for name in ["F_pr15s_SHORT_T10%", "G_pr15s_SHORT_T5%", "A_60s_SHORT_T0.5%", "I_60s_SHORT_T5%"]:
            sel = sels[name]
            pnl = pnl_for(sel & bucket_mask)
            if len(pnl) >= 5:
                bd[name] = {"n_fills": int(len(pnl)), "mean": float(pnl.mean()),
                            "passive_net": float(pnl.mean() - COMM),
                            "dollar_pnl": float(len(pnl) * (pnl.mean() - COMM) * 12.50)}
            else:
                bd[name] = {"n_fills": int(len(pnl)), "mean": float("nan"),
                            "passive_net": float("nan"), "dollar_pnl": 0.0}
        tod_results[bn] = bd
        f_n = bd["F_pr15s_SHORT_T10%"]["n_fills"]
        f_d = bd["F_pr15s_SHORT_T10%"]["dollar_pnl"]
        a_n = bd["A_60s_SHORT_T0.5%"]["n_fills"]
        a_d = bd["A_60s_SHORT_T0.5%"]["dollar_pnl"]
        log(f"  {bn}: F_n={f_n:>4} ${f_d:+,.0f} | A_n={a_n:>3} ${a_d:+,.0f}")

    # ---- SAVE JSON ----
    out_json = {"individual": indiv, "confluence": confl, "overlap": overlap,
                "per_day": per_day, "tod_buckets": tod_results,
                "generated": datetime.utcnow().isoformat() + "Z"}
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump(out_json, f, indent=2)
    log(f"Wrote {OUT_DIR / 'results.json'}")

    # ---- MARKDOWN SUMMARY ----
    sm = OUT_DIR / "SUMMARY.md"
    with open(sm, "w") as f:
        f.write("# v3.2 p_reversal_15s Family Analysis — Confluence + Per-Day + ToD\n\n")
        f.write(f"_{datetime.utcnow().isoformat(timespec='seconds')}Z — fold 0 OOT, 5 days_\n\n")
        f.write("## Survivor codes\n")
        f.write("- **A** = log_ret_60s   SHORT Top0.5%  (original headline survivor)\n")
        f.write("- **D** = log_ret_1s    SHORT Top0.1%\n")
        f.write("- **E** = p_reversal_15s SHORT Top20%  ← BIG new\n")
        f.write("- **F** = p_reversal_15s SHORT Top10%  ← BIG new (best $/edge balance)\n")
        f.write("- **G** = p_reversal_15s SHORT Top5%\n")
        f.write("- **H** = p_reversal_15s SHORT Top1%\n")
        f.write("- **I** = log_ret_60s   SHORT Top5%\n")
        f.write("- **J** = log_ret_60s   SHORT Top10%\n")
        f.write("- **K** = log_ret_60s   SHORT Top20%\n\n")

        f.write("## 1. Individual verification\n\n")
        f.write("| ID | Head | Band | n_fills | mean t | Sharpe | WR | passive_net | $ P&L (5d) |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for name, st in indiv.items():
            id_, head_part = name.split("_", 1)
            f.write(f"| **{id_}** | {head_part} | - | {st['n']} | {st['mean']:+.3f} | "
                    f"{st['sharpe']:.2f} | {st['wr']*100:.1f}% | {st['passive_net']:+.3f} | "
                    f"${st['dollar_pnl']:+,.0f} |\n")

        f.write("\n## 2. Pairwise confluence (does combining heads help?)\n\n")
        f.write("| Pair | n_signals | n_fills | mean | Sharpe | WR | passive_net | $ P&L | frac_of_smaller |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for k, st in confl.items():
            f.write(f"| {k} | {st['intersect_signals']} | {st['n']} | {st['mean']:+.3f} | "
                    f"{st['sharpe']:.2f} | {st['wr']*100:.1f}% | {st['passive_net']:+.3f} | "
                    f"${st['dollar_pnl']:+,.0f} | {st['frac_of_smaller']:.2f} |\n")

        f.write("\n## 3. Overlap matrix (Jaccard of signal sets)\n\n")
        f.write("Key question: does F (large-n) encompass A (small-n)?\n\n")
        f.write("| Pair | intersect | union | Jaccard | n1 | n2 |\n")
        f.write("|---|---|---|---|---|---|\n")
        for k in ["A_vs_F", "A_vs_G", "A_vs_E", "F_vs_G", "F_vs_J", "I_vs_F", "D_vs_F"]:
            if k in overlap:
                o = overlap[k]
                f.write(f"| {k} | {o['intersect']} | {o['union']} | {o['jaccard']:.3f} | "
                        f"{o['n1']} | {o['n2']} |\n")

        f.write("\n## 4. Per-day robustness (5 OOT days, $ P&L)\n\n")
        f.write("| Date | A_n | A_$ | F_n | F_$ | G_n | G_$ | I_n | I_$ |\n")
        f.write("|---|---|---|---|---|---|---|---|---|\n")
        for date, dd in per_day.items():
            f.write(f"| {date} | {dd['A_60s_SHORT_T0.5%']['n_fills']} | "
                    f"${dd['A_60s_SHORT_T0.5%']['dollar_pnl']:+,.0f} | "
                    f"{dd['F_pr15s_SHORT_T10%']['n_fills']} | "
                    f"${dd['F_pr15s_SHORT_T10%']['dollar_pnl']:+,.0f} | "
                    f"{dd['G_pr15s_SHORT_T5%']['n_fills']} | "
                    f"${dd['G_pr15s_SHORT_T5%']['dollar_pnl']:+,.0f} | "
                    f"{dd['I_60s_SHORT_T5%']['n_fills']} | "
                    f"${dd['I_60s_SHORT_T5%']['dollar_pnl']:+,.0f} |\n")

        f.write("\n## 5. ToD distribution (approx via event-index buckets, $ P&L)\n\n")
        f.write("**CAVEAT**: Same caveat as confluence_tod analysis — NPZ has no timestamps; "
                "ToD is approximated by event index assuming uniform RTH event rate.\n\n")
        f.write("| Bucket | F_n | F_passive | F_$ | A_n | A_passive | A_$ |\n")
        f.write("|---|---|---|---|---|---|---|\n")
        for bn, bd in tod_results.items():
            f.write(f"| {bn} | {bd['F_pr15s_SHORT_T10%']['n_fills']} | "
                    f"{bd['F_pr15s_SHORT_T10%']['passive_net']:+.3f} | "
                    f"${bd['F_pr15s_SHORT_T10%']['dollar_pnl']:+,.0f} | "
                    f"{bd['A_60s_SHORT_T0.5%']['n_fills']} | "
                    f"{bd['A_60s_SHORT_T0.5%']['passive_net']:+.3f} | "
                    f"${bd['A_60s_SHORT_T0.5%']['dollar_pnl']:+,.0f} |\n")

        f.write("\n## Decision-quality interpretation\n\n")
        f.write("- If **F dominates A in $ P&L AND has 4-of-5 days positive** → F is the deployment-ready candidate (not A)\n")
        f.write("- If **F × A confluence has higher per-fill edge than F alone** → run BOTH, but only enter on confluence\n")
        f.write("- If **F is concentrated in same ToD bucket as A** → ToD-gate for the strategy is real\n")
        f.write("- If **F and A have <0.10 Jaccard** → they're orthogonal, BOTH should be deployed\n")
    log(f"Wrote {sm}")
    log(f"Done in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
