#!/usr/bin/env python3
"""
Sweep band combinations for the (log_ret_60s SHORT × p_reversal_15s SHORT)
confluence. K×E (T20%/T20%) was robust 3/3 days; check tighter and looser
variants for the best $ P&L per day-robustness trade-off.

NO trainer code modified. Pure analysis. HC #307D.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"
OUT_DIR = ROOT / "output/v3_2_kxe_band_sweep_20260514"
OUT_DIR.mkdir(parents=True, exist_ok=True)
COMM = 0.376


def select_top_band(score, side_sign, band_pct):
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


def main():
    d = np.load(PRED_NPZ)
    n = int(d["n_samples"])
    score_60s = d["pred_log_ret_60s"].astype(np.float32)
    score_pr15 = d["pred_p_reversal_15s"].astype(np.float32) - 0.5
    fifo = d["target_fifo_tp4sl3_net"].astype(np.float32)
    fifo_mask = d["mask_fifo_tp4sl3_net"].astype(bool)

    BANDS = [0.005, 0.01, 0.02, 0.03, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.50]
    chunk = n // 5
    out_dates = [str(x) for x in d["oot_dates"]]

    print(f"n={n:,}, chunk={chunk:,}, dates={out_dates}\n")
    print(f"{'b60s':>6} {'bpr15':>6} {'n':>5} {'mean_t':>7} {'pNet':>7} ${'P&L':>7}  D0  D1  D2  D3  D4  verdict")

    rows = []
    for b60 in BANDS:
        for bpr in BANDS:
            sel60 = select_top_band(score_60s, -1, b60)
            selpr = select_top_band(score_pr15, -1, bpr)
            sel = sel60 & selpr & fifo_mask
            n_total = int(sel.sum())
            if n_total < 10:
                continue
            pnl = -fifo[sel]
            mean_t = float(pnl.mean())
            net_t = mean_t - COMM
            net_dollars = float(n_total * net_t * 12.50)

            # Per-day
            per_day = []
            for di in range(5):
                day_lo = di * chunk
                day_hi = day_lo + chunk if di < 4 else n
                day_mask = np.zeros(n, dtype=bool)
                day_mask[day_lo:day_hi] = True
                pnl_d = -fifo[(sel60 & selpr) & day_mask & fifo_mask]
                if len(pnl_d) >= 1:
                    per_day.append({"n": int(len(pnl_d)),
                                    "net_$": float(len(pnl_d) * (pnl_d.mean() - COMM) * 12.50)
                                    if len(pnl_d) >= 2 else float(pnl_d[0] - COMM) * 12.50})
                else:
                    per_day.append({"n": 0, "net_$": 0.0})

            # Verdict
            fillable = [pd for pd in per_day if pd["n"] >= 5]
            n_pos = sum(1 for pd in fillable if pd["net_$"] > 0)
            n_total_fillable = len(fillable)
            d_max_share = max((pd["net_$"] for pd in per_day), default=0) / max(net_dollars, 1)
            if not fillable:
                verdict = "NO_FILL"
            elif net_dollars <= 0:
                verdict = "NEG_TOTAL"
            elif d_max_share >= 0.85:
                verdict = f"DAY_ART({d_max_share*100:.0f}%)"
            elif n_pos == n_total_fillable:
                verdict = f"ROBUST_{n_pos}/{n_total_fillable}"
            elif n_pos >= n_total_fillable - 1:
                verdict = f"MOSTLY_{n_pos}/{n_total_fillable}"
            else:
                verdict = f"MIXED_{n_pos}/{n_total_fillable}"

            rows.append({"b60s": b60, "bpr15": bpr, "n": n_total, "mean_t": mean_t,
                         "passive_net": net_t, "dollar_pnl": net_dollars,
                         "per_day": per_day, "verdict": verdict})

            day_str = ' '.join(f"{pd['n']:>3}" for pd in per_day)
            print(f"{b60:>6.3f} {bpr:>6.3f} {n_total:>5} {mean_t:>+7.3f} {net_t:>+7.3f} "
                  f"${net_dollars:>+8,.0f}  {day_str}  {verdict}")

    # Save
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"rows": rows, "generated": datetime.utcnow().isoformat() + "Z"},
                  f, indent=2)

    # Best by category
    print("\n=== TOP 10 ROBUST/MOSTLY by $ P&L ===")
    robust = [r for r in rows if "ROBUST" in r["verdict"] or "MOSTLY" in r["verdict"]]
    robust.sort(key=lambda r: -r["dollar_pnl"])
    for r in robust[:10]:
        print(f"  b60={r['b60s']:.3f} bpr={r['bpr15']:.3f} n={r['n']:>4} ${r['dollar_pnl']:>+9,.0f} {r['verdict']}")

    print("\n=== TOP 10 ALL by $ P&L (ignoring verdict) ===")
    rows.sort(key=lambda r: -r["dollar_pnl"])
    for r in rows[:10]:
        print(f"  b60={r['b60s']:.3f} bpr={r['bpr15']:.3f} n={r['n']:>5} "
              f"${r['dollar_pnl']:>+9,.0f} {r['verdict']}")


if __name__ == "__main__":
    main()
