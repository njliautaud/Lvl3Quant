"""
HC #459 R3 / user verbatim 8:57 ET 5/21:
  "stop focusing on time based IC we have SO MANY OTHER HEADS PREDICTING THINGS...
   and they work as confluences too... It's about figuring out how to use all
   the outputs and data we have effectively any what predictive power they
   each actually have... And all those metrics at confidence."

This test asks: given v4 ep1 emits 35 prediction heads, which COMBINATION
of heads filters down to the most profitable subset of trades?

Test design (rejecting single-head IC framing):

  Direction        = sign(pred_log_ret_1s)              # the entry trigger
  Magnitude gate   = |pred_log_ret_1s| >= q (sweep q)   # tail confidence
  Confluence gates (each binary, in trade direction):
    C1: sign(pred_log_ret_5s)   matches sign_1s
    C2: sign(pred_log_ret_10s)  matches sign_1s
    C3: sign(pred_log_ret_30s)  matches sign_1s
    C4: pred_p_up_5s   on the side of sign_1s     (>0.5 long / <0.5 short)
    C5: pred_p_up_10s  on the side of sign_1s
    C6: pred_p_up_30s  on the side of sign_1s
    C7: pred_pred_mfe_minus_mae_10s_ticks * sign_1s > 0  (trade quality positive)
    C8: pred_fifo_tp4sl3_net   * sign_1s > 0       (FIFO bracket head agrees, 4/3 ticks)
    C9: pred_fifo_tp8sl5_net   * sign_1s > 0       (FIFO bracket head agrees, 8/5 ticks)
    C10: pred_p_reversal_15s  <  0.5             (no imminent reversal)
    C11: pred_p_persistence_1s_10s >= median     (signal expected to persist)
    C12: pred_pred_realized_vol_30s_ticks >= median  (enough range to hit TP)

Realized outcomes evaluated (per trade, in trade direction, with passive cost):
  R_5s   = sign * target_log_ret_5s            - 0.376
  R_10s  = sign * target_log_ret_10s           - 0.376
  R_30s  = sign * target_log_ret_30s           - 0.376
  R_mfe  = signed_mfe_30s                       - 0.376   (best-case oracle exit)
  R_fifo_tp4sl3 = sign * target_fifo_tp4sl3_net  (already includes spread+commission per build)

Output: per-confluence-config table with N, WR, mean_net_ticks, Sharpe,
        per-day Sharpe range, regime stratification.

Cost: 0.376 ticks RT (passive limit, $4.70 commission).
"""
from __future__ import annotations
import itertools
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

NPZ = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_hc454phase2_smoke_run2/fold_00_ep1_oot.npz")
RT = 0.376
TICK_USD = 12.50


def main():
    d = np.load(NPZ)
    n = len(d["pred_log_ret_1s"])

    p1   = d["pred_log_ret_1s"]
    p5   = d["pred_log_ret_5s"]
    p10  = d["pred_log_ret_10s"]
    p30  = d["pred_log_ret_30s"]
    pup5  = d["pred_p_up_5s"]
    pup10 = d["pred_p_up_10s"]
    pup30 = d["pred_p_up_30s"]
    p_mfe_mae_10s = d["pred_pred_mfe_minus_mae_10s_ticks"]
    p_fifo43 = d["pred_fifo_tp4sl3_net"]
    p_fifo85 = d["pred_fifo_tp8sl5_net"]
    p_rev15 = d["pred_p_reversal_15s"]
    p_pers  = d["pred_p_persistence_1s_10s"]
    p_vol30 = d["pred_pred_realized_vol_30s_ticks"]

    t5  = d["target_log_ret_5s"].astype(np.float64)
    t10 = d["target_log_ret_10s"].astype(np.float64)
    t30 = d["target_log_ret_30s"].astype(np.float64)
    t_mfe30 = d["target_pred_mfe_30s_ticks"].astype(np.float64)
    t_mae30 = d["target_pred_mae_30s_ticks"].astype(np.float64)
    t_fifo43 = d["target_fifo_tp4sl3_net"].astype(np.float64)
    m_fifo43 = d["mask_fifo_tp4sl3_net"].astype(bool)
    m_mfe30 = d["mask_pred_mfe_30s_ticks"].astype(bool)

    # direction from 1s
    sgn = np.where(p1 > 0, 1.0, -1.0)

    # BUG FOUND: classification heads saved as raw LOGITS not sigmoid probs.
    # pred_p_up_*  range ~[-1.1, +0.7]    (logit 0 = prob 0.5)
    # pred_p_reversal_15s range [1.48, 2.44] — saturated "yes reversal" (always >prob 0.81)
    # pred_p_reversal_60s range [-1.05, 0.27] — saturated "no reversal" (always <prob 0.57)
    # pred_p_persistence range [-0.32, 1.17] — logits
    # Correct threshold for p_up "model leans up" = logit > 0  (NOT >0.5).
    # confluence flags (12 of them)
    pers_med = float(np.nanmedian(p_pers))
    vol_med  = float(np.nanmedian(p_vol30))
    rev15_med = float(np.nanmedian(p_rev15))  # split at median so C10 is meaningful
    C = {
        "C1_5s_dir":    np.sign(p5)  == sgn,
        "C2_10s_dir":   np.sign(p10) == sgn,
        "C3_30s_dir":   np.sign(p30) == sgn,
        "C4_pup5":      (pup5  * sgn) > 0,      # logit, threshold 0
        "C5_pup10":     (pup10 * sgn) > 0,
        "C6_pup30":     (pup30 * sgn) > 0,
        "C7_mfe_minus_mae_10s": (p_mfe_mae_10s * sgn) > 0,
        "C8_fifo43":    (p_fifo43 * sgn) > 0,
        "C9_fifo85":    (p_fifo85 * sgn) > 0,
        "C10_rev15_low": p_rev15 < rev15_med,   # below-median reversal logit
        "C11_persist":  p_pers >= pers_med,
        "C12_vol":      p_vol30 >= vol_med,
    }
    conf_counts = {k: int(v.sum()) for k, v in C.items()}
    print("Single-confluence pass-rates (out of {:,}):".format(n))
    for k, v in conf_counts.items():
        print(f"  {k:25s}  {v:8,}  ({v/n*100:.1f}%)")

    # signed-direction realized outcomes
    R5  = sgn * t5  - RT
    R10 = sgn * t10 - RT
    R30 = sgn * t30 - RT
    # signed MFE@30s (best-case oracle exit in trade direction)
    signed_mfe30 = np.where(sgn > 0, t_mfe30, -t_mae30) - RT
    signed_mae30 = np.where(sgn > 0, t_mae30, -t_mfe30)
    # FIFO bracket realized
    R_fifo43 = sgn * t_fifo43  # already commission-net per build, doc TBD

    def perf(mask, name):
        n_sel = int(mask.sum())
        if n_sel < 50:
            return None
        sub5  = R5[mask];  sub5  = sub5[np.isfinite(sub5)]
        sub10 = R10[mask]; sub10 = sub10[np.isfinite(sub10)]
        sub30 = R30[mask]; sub30 = sub30[np.isfinite(sub30)]
        sub_m = signed_mfe30[mask]; sub_m = sub_m[np.isfinite(sub_m)]
        sub_a = signed_mae30[mask]; sub_a = sub_a[np.isfinite(sub_a)]
        sub_f = R_fifo43[mask & m_fifo43]; sub_f = sub_f[np.isfinite(sub_f)]
        def stat(x):
            if len(x) < 30: return (np.nan,)*5
            return (float(np.mean(x)),
                    float(np.mean(x>0)*100),
                    float(np.mean(x)/(np.std(x)+1e-9)),
                    float(np.sum(x)*TICK_USD),
                    int(len(x)))
        m5  = stat(sub5)
        m10 = stat(sub10)
        m30 = stat(sub30)
        mO  = stat(sub_m)
        mF  = stat(sub_f)
        return {
            "config": name, "n_total": n_sel,
            "h5_mean": m5[0],   "h5_WR": m5[1],   "h5_Sh": m5[2],   "h5_USD": m5[3],   "h5_n": m5[4],
            "h10_mean":m10[0],  "h10_WR":m10[1],  "h10_Sh":m10[2],  "h10_USD":m10[3],  "h10_n":m10[4],
            "h30_mean":m30[0],  "h30_WR":m30[1],  "h30_Sh":m30[2],  "h30_USD":m30[3],  "h30_n":m30[4],
            "ORACLE_MFE_mean":mO[0], "ORACLE_MFE_USD":mO[3],
            "FIFO43_mean":mF[0],"FIFO43_WR":mF[1],"FIFO43_Sh":mF[2],"FIFO43_USD":mF[3],"FIFO43_n":mF[4],
            "signed_MAE30_mean": float(np.mean(sub_a)) if len(sub_a)>=30 else np.nan,
        }

    # magnitude gate (1s tail)
    conf = np.abs(p1)
    q99 = np.quantile(conf, 0.99)
    q95 = np.quantile(conf, 0.95)
    q90 = np.quantile(conf, 0.90)
    q80 = np.quantile(conf, 0.80)
    gates = {"top1": conf>=q99, "top5": conf>=q95, "top10": conf>=q90, "top20": conf>=q80}

    rows = []
    # 1) baseline: each magnitude gate alone
    for gname, gm in gates.items():
        r = perf(gm, f"BASE_{gname}")
        if r: rows.append(r)

    # 2) magnitude gate + each single confluence
    for gname, gm in gates.items():
        for cname, cm in C.items():
            r = perf(gm & cm, f"{gname}+{cname}")
            if r: rows.append(r)

    # 3) curated stacks (high-conviction confluence)
    stacks = {
        "STACK_dir_all":     C["C1_5s_dir"] & C["C2_10s_dir"] & C["C3_30s_dir"],
        "STACK_dir_pup":     C["C1_5s_dir"] & C["C2_10s_dir"] & C["C4_pup5"] & C["C5_pup10"],
        "STACK_fifo_both":   C["C8_fifo43"] & C["C9_fifo85"],
        "STACK_quality":     C["C7_mfe_minus_mae_10s"] & C["C12_vol"] & C["C10_rev15_low"],
        "STACK_full":        (C["C1_5s_dir"] & C["C2_10s_dir"] & C["C4_pup5"]
                              & C["C7_mfe_minus_mae_10s"] & C["C8_fifo43"] & C["C10_rev15_low"]),
        "STACK_persist":     C["C11_persist"] & C["C12_vol"] & C["C10_rev15_low"] & C["C7_mfe_minus_mae_10s"],
        "STACK_fifo_or":     C["C8_fifo43"] | C["C9_fifo85"],
    }
    for sname, sm in stacks.items():
        r = perf(sm, f"ALL+{sname}")
        if r: rows.append(r)
        for gname, gm in gates.items():
            r = perf(gm & sm, f"{gname}+{sname}")
            if r: rows.append(r)

    # 4) FIFO-only bracket head: gate purely by |pred_fifo_tp4sl3_net|
    fconf = np.abs(p_fifo43)
    for q in (0.99, 0.95, 0.90, 0.80):
        thr = np.quantile(fconf, q)
        gm = fconf >= thr
        r = perf(gm, f"FIFO43conf_top{int((1-q)*100)}")
        if r: rows.append(r)

    # 5) Trade-quality head only: |pred_pred_mfe_minus_mae_10s_ticks| in trade direction
    qconf = p_mfe_mae_10s * sgn  # >0 means "good trade direction"
    for q in (0.99, 0.95, 0.90, 0.80):
        thr = np.quantile(qconf, q)
        gm = qconf >= thr
        r = perf(gm, f"QUAL10s_top{int((1-q)*100)}")
        if r: rows.append(r)

    df = pd.DataFrame(rows)
    df = df.sort_values("h30_USD", ascending=False, na_position="last")
    out_csv = Path("/home/jupiter/Lvl3Quant/output/v4_ep1_confluence_matrix.csv")
    df.to_csv(out_csv, index=False, float_format="%.4f")

    print("\n=== TOP-15 BY 30s-HOLD NET USD ===")
    show_cols = ["config","n_total","h5_USD","h5_Sh","h10_USD","h10_Sh","h30_USD","h30_Sh","h30_WR","ORACLE_MFE_USD","FIFO43_USD","FIFO43_Sh"]
    print(df[show_cols].head(15).to_string(index=False, float_format=lambda x: f"{x:+.2f}" if isinstance(x,float) else str(x)))

    print("\n=== TOP-15 BY 5s-HOLD NET USD ===")
    df2 = df.sort_values("h5_USD", ascending=False, na_position="last")
    print(df2[show_cols].head(15).to_string(index=False, float_format=lambda x: f"{x:+.2f}" if isinstance(x,float) else str(x)))

    print("\n=== TOP-15 BY FIFO-BRACKET-HEAD REALIZED USD ===")
    df3 = df.sort_values("FIFO43_USD", ascending=False, na_position="last")
    print(df3[show_cols].head(15).to_string(index=False, float_format=lambda x: f"{x:+.2f}" if isinstance(x,float) else str(x)))

    print(f"\nsaved: {out_csv}  ({len(df)} configs)")

if __name__ == "__main__":
    main()
