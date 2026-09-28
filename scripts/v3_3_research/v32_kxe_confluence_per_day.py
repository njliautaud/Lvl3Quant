#!/usr/bin/env python3
"""
HC #345 — Per-day robustness for the K×E confluence (60s SHORT T20% AND
p_reversal_15s SHORT T20%), plus all other promising confluence pairs.

Ground truth: F alone is 96%-Day-2 artifact. Need to know if confluence
pairs ALSO concentrate on Day 2 or genuinely spread across days.

NO trainer code modified. Pure analysis. HC #307D.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"
OUT_DIR = ROOT / "output/v3_2_kxe_confluence_per_day_20260514"
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
    print("Loading...")
    d = np.load(PRED_NPZ)
    n = int(d["n_samples"])
    score_60s = d["pred_log_ret_60s"].astype(np.float32)
    score_pr15 = d["pred_p_reversal_15s"].astype(np.float32) - 0.5
    fifo = d["target_fifo_tp4sl3_net"].astype(np.float32)
    fifo_mask = d["mask_fifo_tp4sl3_net"].astype(bool)

    sels = {
        "K_60s_T20%": select_top_band(score_60s, -1, 0.20),
        "E_pr_T20%":  select_top_band(score_pr15, -1, 0.20),
        "J_60s_T10%": select_top_band(score_60s, -1, 0.10),
        "F_pr_T10%":  select_top_band(score_pr15, -1, 0.10),
        "I_60s_T5%":  select_top_band(score_60s, -1, 0.05),
        "G_pr_T5%":   select_top_band(score_pr15, -1, 0.05),
        "A_60s_T0.5%": select_top_band(score_60s, -1, 0.005),
    }

    confluences = {
        "KxE_T20_T20": sels["K_60s_T20%"] & sels["E_pr_T20%"],
        "JxF_T10_T10": sels["J_60s_T10%"] & sels["F_pr_T10%"],
        "IxG_T5_T5":   sels["I_60s_T5%"] & sels["G_pr_T5%"],
        "JxE_T10_T20": sels["J_60s_T10%"] & sels["E_pr_T20%"],
        "KxF_T20_T10": sels["K_60s_T20%"] & sels["F_pr_T10%"],
        "IxF_T5_T10":  sels["I_60s_T5%"] & sels["F_pr_T10%"],
        "AxF_T0.5_T10": sels["A_60s_T0.5%"] & sels["F_pr_T10%"],
        "AxE_T0.5_T20": sels["A_60s_T0.5%"] & sels["E_pr_T20%"],
    }

    chunk = n // 5
    print(f"\nn={n:,}, chunk={chunk:,}")

    print("\n=== PER-DAY P&L ($) FOR ALL CONFLUENCE PAIRS ===\n")
    print(f"{'Pair':<18} {'TOT_$':>10} {'D0_n':>5} {'D0_$':>9} {'D1_n':>5} {'D1_$':>9} "
          f"{'D2_n':>5} {'D2_$':>9} {'D3_n':>5} {'D3_$':>9} {'D4_n':>5} {'D4_$':>9} {'verdict':>14}")

    results = {}
    for name, mask in confluences.items():
        per_day = []
        total_pnl_dollars = 0.0
        total_n_fills = 0
        for di in range(5):
            day_lo = di * chunk
            day_hi = day_lo + chunk if di < 4 else n
            day_mask = np.zeros(n, dtype=bool)
            day_mask[day_lo:day_hi] = True
            sel = mask & day_mask & fifo_mask
            pnl = -fifo[sel]
            n_fills = len(pnl)
            net = (pnl - COMM).sum() * 12.50  # commission-adjusted dollars
            per_day.append({"n": int(n_fills), "net_$": float(net),
                            "mean_t": float(pnl.mean()) if n_fills >= 1 else float("nan")})
            total_pnl_dollars += net
            total_n_fills += n_fills

        # Verdict: positive in how many of fillable days?
        fillable = [pd for pd in per_day if pd["n"] >= 5]
        if not fillable:
            verdict = "NO_FILL"
        else:
            n_pos = sum(1 for pd in fillable if pd["net_$"] > 0)
            n_total = len(fillable)
            # Day-concentration: if any single day = >50% of total $
            d_max_share = max((pd["net_$"] for pd in per_day), default=0) / max(total_pnl_dollars, 1)
            if total_pnl_dollars <= 0:
                verdict = "NEG_TOTAL"
            elif d_max_share >= 0.80:
                verdict = f"DAY_ARTIFACT({d_max_share*100:.0f}%)"
            elif n_pos == n_total:
                verdict = f"ROBUST_{n_pos}/{n_total}"
            elif n_pos >= n_total - 1:
                verdict = f"MOSTLY_{n_pos}/{n_total}"
            else:
                verdict = f"MIXED_{n_pos}/{n_total}"

        results[name] = {"total_n_fills": int(total_n_fills),
                         "total_net_dollars": float(total_pnl_dollars),
                         "per_day": per_day, "verdict": verdict}

        print(f"{name:<18} ${total_pnl_dollars:>+9,.0f} "
              f"{per_day[0]['n']:>5} ${per_day[0]['net_$']:>+8,.0f} "
              f"{per_day[1]['n']:>5} ${per_day[1]['net_$']:>+8,.0f} "
              f"{per_day[2]['n']:>5} ${per_day[2]['net_$']:>+8,.0f} "
              f"{per_day[3]['n']:>5} ${per_day[3]['net_$']:>+8,.0f} "
              f"{per_day[4]['n']:>5} ${per_day[4]['net_$']:>+8,.0f} "
              f"{verdict:>14}")

    # Save JSON
    with open(OUT_DIR / "results.json", "w") as f:
        json.dump({"results": results, "generated": datetime.utcnow().isoformat() + "Z"}, f, indent=2)
    print(f"\nWrote {OUT_DIR / 'results.json'}")

    # ROBUST sorted summary
    print("\n=== ROBUST PAIRS RANKED BY $ P&L ===\n")
    robust = [(name, r) for name, r in results.items() if "ROBUST" in r["verdict"] or "MOSTLY" in r["verdict"]]
    robust.sort(key=lambda x: -x[1]["total_net_dollars"])
    for name, r in robust:
        print(f"  {name:<18} ${r['total_net_dollars']:>+9,.0f}  n={r['total_n_fills']:>4}  {r['verdict']}")
    if not robust:
        print("  (none — all are 1-day artifacts or insufficient sample)")


if __name__ == "__main__":
    main()
