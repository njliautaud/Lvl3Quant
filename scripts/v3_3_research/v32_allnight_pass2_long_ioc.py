#!/usr/bin/env python3
"""
HC #346 PASS 2 — Drill into LONG-side IOC market entry profitability

Pass 1 finding: Long-side IOC market entry on Top1-10% picks generates +6-7 ticks NET
(hold-to-MFE assumption). Short-side loses -5 ticks. Need to:
  1. Confirm with realistic exit logic (NOT oracle hold-to-MFE).
  2. Find best EXIT rule (fixed-N-secs, p_reversal threshold, MFE-trail).
  3. Stratify by day, time-of-day, vol regime, signal-strength to confirm robustness.
  4. Optuna over (entry_threshold, hold_secs, reversal_exit, max_hold_secs).
  5. Train tiny MLP for EXIT decision (state=heads+time-since-entry, action=hold/exit).

Outputs: output/v3_2_allnight_research_20260514/pass2_long_ioc/
"""
from __future__ import annotations

import json
import math
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"
OUT = ROOT / "output/v3_2_allnight_research_20260514/pass2_long_ioc"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "pass2.log"

MARKET_COST_TICKS = 1.376
PASSIVE_COST_TICKS = 0.376


def log(msg):
    ts = datetime.utcnow().isoformat(timespec="seconds")
    line = f"[{ts}Z] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def synth_day_idx(n, n_days=5):
    spd = n // n_days
    out = np.zeros(n, dtype=int)
    for di in range(n_days):
        s = di * spd; e = (di+1) * spd if di < n_days-1 else n
        out[s:e] = di
    return out


def synth_tod(n, n_days=5):
    spd = n // n_days
    out = np.zeros(n, dtype=np.float32)
    for di in range(n_days):
        s = di*spd; e = (di+1)*spd if di < n_days-1 else n
        out[s:e] = np.linspace(0, 6.5*3600, e-s, dtype=np.float32)
    return out


def main():
    log("PASS 2 — Long-IOC drill-down START")
    d = np.load(PRED_NPZ, allow_pickle=True)
    n = int(d["n_samples"])
    log(f"loaded n={n}")

    p1 = d["pred_log_ret_1s"]
    msk1 = d["mask_log_ret_1s"].astype(bool) & np.isfinite(p1)

    # MFE/MAE in ticks (real units — these are the truth)
    mfe30 = d["target_pred_mfe_30s_ticks"]
    mae30 = d["target_pred_mae_30s_ticks"]
    mfe60 = d["target_pred_mfe_60s_ticks"]
    mae60 = d["target_pred_mae_60s_ticks"]
    t2mfe = d["target_pred_time_to_mfe_secs"]
    mfe30_m = d["mask_pred_mfe_30s_ticks"].astype(bool) & np.isfinite(mfe30)
    mae30_m = d["mask_pred_mae_30s_ticks"].astype(bool) & np.isfinite(mae30)
    mfe60_m = d["mask_pred_mfe_60s_ticks"].astype(bool) & np.isfinite(mfe60)
    mae60_m = d["mask_pred_mae_60s_ticks"].astype(bool) & np.isfinite(mae60)
    t2mfe_m = d["mask_pred_time_to_mfe_secs"].astype(bool) & np.isfinite(t2mfe)

    rev15 = d["pred_p_reversal_15s"]
    rev30 = d["pred_p_reversal_30s"]
    rev60 = d["pred_p_reversal_60s"]

    pred_mfe = d["pred_pred_mfe_30s_ticks"]
    pred_mae = d["pred_pred_mae_30s_ticks"]
    pred_vol = d["pred_pred_realized_vol_30s_ticks"]

    day_idx = synth_day_idx(n)
    tod = synth_tod(n)

    # FIFO labels (ground-truth tick PnL)
    fnet43 = d["target_fifo_tp4sl3_net"]
    fmsk43 = d["mask_fifo_tp4sl3_net"].astype(bool)
    fnet85 = d["target_fifo_tp8sl5_net"]
    fmsk85 = d["mask_fifo_tp8sl5_net"].astype(bool)

    out_summary = {}

    # ─────── A. LONG ONLY: per-band realistic exit comparison ───────
    log("A. exit-rule comparison for long-side")
    a_rows = []
    long_mask = msk1 & (p1 > 0)
    absp = np.abs(p1[long_mask])
    idx_long = np.where(long_mask)[0]
    for bname, bfrac in [("Top0.1%", 0.001), ("Top0.5%", 0.005), ("Top1%", 0.01),
                          ("Top5%", 0.05), ("Top10%", 0.10)]:
        n_take = max(5, int(idx_long.size * bfrac))
        order = np.argsort(-np.abs(p1[idx_long]))
        cell = idx_long[order[:n_take]]

        # Use 30s MFE/MAE (in ticks, in price-direction = positive=up). For long: gain = mfe, loss = -mae
        mm = mfe30_m[cell] & mae30_m[cell]
        mfe = mfe30[cell][mm]
        mae = mae30[cell][mm]
        nn = mfe.size
        if nn < 5: continue

        # Strategies:
        # (1) Hold-to-MFE (oracle): exit at peak. PnL = mfe - cost
        s1 = mfe - MARKET_COST_TICKS
        # (2) Hold-fixed-30s: exit at 30s. Approx with avg(mfe, mae) - cost
        s2 = (mfe + mae)/2 - MARKET_COST_TICKS
        # (3) Stop-loss-N-ticks: exit at stop OR at MFE peak. PnL = max(-stop, min(mfe, ...))
        for stop_t in [2, 3, 4, 5]:
            # If MAE >= -stop, we got stopped: PnL = -stop - cost
            # If MAE < -stop (i.e., didn't hit stop), assume we exit at avg of MFE
            stopped = mae <= -stop_t
            pnl = np.where(stopped, -stop_t - MARKET_COST_TICKS,
                           (mfe + 0)/2 - MARKET_COST_TICKS)
            row_key = f"stop_{stop_t}_takeavg"
            a_rows.append({
                "band": bname, "strategy": row_key, "n": int(nn),
                "mean_pnl_t": float(np.mean(pnl)),
                "median_pnl_t": float(np.median(pnl)),
                "win_rate_pct": float(np.mean(pnl > 0) * 100),
                "sharpe_toy": float(np.mean(pnl) / np.std(pnl) * math.sqrt(nn)) if np.std(pnl) > 0 else None,
                "stop_hit_pct": float(np.mean(stopped) * 100)
            })
        # (4) Take-profit-N + stop-loss-M: exit at first hit
        for tp_t, sl_t in [(2, 2), (3, 3), (4, 3), (5, 3), (8, 5)]:
            hit_tp = mfe >= tp_t
            hit_sl = mae <= -sl_t
            # Both hit → unknown order, assume worse case (sl first)
            # Only TP → +tp-cost. Only SL → -sl-cost. Neither → exit at expected midpoint (mfe+mae)/2
            both = hit_tp & hit_sl
            only_tp = hit_tp & ~hit_sl
            only_sl = ~hit_tp & hit_sl
            neither = ~hit_tp & ~hit_sl
            pnl = np.zeros(nn)
            pnl[only_tp] = tp_t - MARKET_COST_TICKS
            pnl[only_sl] = -sl_t - MARKET_COST_TICKS
            pnl[both] = -sl_t - MARKET_COST_TICKS  # conservative
            pnl[neither] = (mfe[neither] + mae[neither])/2 - MARKET_COST_TICKS
            a_rows.append({
                "band": bname, "strategy": f"tp{tp_t}_sl{sl_t}",
                "n": int(nn), "mean_pnl_t": float(np.mean(pnl)),
                "median_pnl_t": float(np.median(pnl)),
                "win_rate_pct": float(np.mean(pnl > 0) * 100),
                "sharpe_toy": float(np.mean(pnl) / np.std(pnl) * math.sqrt(nn)) if np.std(pnl) > 0 else None,
                "tp_hit_pct": float(np.mean(only_tp | both) * 100),
                "sl_hit_pct": float(np.mean(only_sl | both) * 100),
            })
        # (5) Hold-to-MFE oracle
        a_rows.append({
            "band": bname, "strategy": "hold_to_MFE_oracle",
            "n": int(nn), "mean_pnl_t": float(np.mean(s1)),
            "median_pnl_t": float(np.median(s1)),
            "win_rate_pct": float(np.mean(s1 > 0) * 100),
            "sharpe_toy": float(np.mean(s1) / np.std(s1) * math.sqrt(nn)) if np.std(s1) > 0 else None,
        })
        # (6) Hold-30s
        a_rows.append({
            "band": bname, "strategy": "hold_30s_avgexit",
            "n": int(nn), "mean_pnl_t": float(np.mean(s2)),
            "median_pnl_t": float(np.median(s2)),
            "win_rate_pct": float(np.mean(s2 > 0) * 100),
            "sharpe_toy": float(np.mean(s2) / np.std(s2) * math.sqrt(nn)) if np.std(s2) > 0 else None,
        })

    out_summary["A_exit_strategies"] = a_rows
    log(f"  A: {len(a_rows)} (band × strategy) rows")

    # ─────── B. Per-day robustness on the BEST strategy (TP/SL) ───────
    log("B. per-day robustness")
    b_rows = []
    # Use best TP/SL combo from A (will pick after — for now use tp4_sl3 across days)
    for tp_t, sl_t in [(2, 2), (3, 3), (4, 3), (5, 3), (8, 5)]:
        for bname, bfrac in [("Top0.5%", 0.005), ("Top1%", 0.01), ("Top5%", 0.05)]:
            for di in range(5):
                day_mask = (day_idx == di) & long_mask
                idx_d = np.where(day_mask)[0]
                if idx_d.size < 50: continue
                n_take = max(5, int(idx_d.size * bfrac))
                cell = idx_d[np.argsort(-np.abs(p1[idx_d]))[:n_take]]
                mm = mfe30_m[cell] & mae30_m[cell]
                if mm.sum() < 5: continue
                mfe = mfe30[cell][mm]; mae = mae30[cell][mm]
                hit_tp = mfe >= tp_t; hit_sl = mae <= -sl_t
                both = hit_tp & hit_sl
                pnl = np.where(hit_tp & ~hit_sl, tp_t - MARKET_COST_TICKS,
                       np.where(hit_sl, -sl_t - MARKET_COST_TICKS,
                                (mfe + mae)/2 - MARKET_COST_TICKS))
                b_rows.append({
                    "tp": tp_t, "sl": sl_t, "band": bname, "day": di,
                    "n": int(mm.sum()),
                    "mean_pnl_t": float(np.mean(pnl)),
                    "win_rate_pct": float(np.mean(pnl > 0) * 100),
                    "sharpe_toy": float(np.mean(pnl) / np.std(pnl) * math.sqrt(mm.sum())) if np.std(pnl) > 0 else None,
                })
    out_summary["B_per_day"] = b_rows
    log(f"  B: {len(b_rows)} (tp,sl,band,day) rows")

    # ─────── C. ToD stratification on best strategy ───────
    log("C. ToD stratification (long, tp4_sl3, Top1%)")
    c_rows = []
    n_buckets = 13  # 30-min buckets
    for ti in range(n_buckets):
        in_b = (tod >= ti*1800) & (tod < (ti+1)*1800) & long_mask
        idx_b = np.where(in_b)[0]
        if idx_b.size < 30: continue
        n_take = max(5, int(idx_b.size * 0.01))
        cell = idx_b[np.argsort(-np.abs(p1[idx_b]))[:n_take]]
        mm = mfe30_m[cell] & mae30_m[cell]
        if mm.sum() < 3: continue
        mfe = mfe30[cell][mm]; mae = mae30[cell][mm]
        tp, sl = 4, 3
        hit_tp = mfe >= tp; hit_sl = mae <= -sl
        pnl = np.where(hit_tp & ~hit_sl, tp - MARKET_COST_TICKS,
               np.where(hit_sl, -sl - MARKET_COST_TICKS,
                        (mfe + mae)/2 - MARKET_COST_TICKS))
        c_rows.append({
            "bucket": ti,
            "label": f"{9 + (ti*30 + 30)//60:02d}:{(ti*30 + 30)%60:02d}",
            "n": int(mm.sum()),
            "mean_pnl_t": float(np.mean(pnl)),
            "win_rate_pct": float(np.mean(pnl > 0) * 100),
            "sharpe_toy": float(np.mean(pnl) / np.std(pnl) * math.sqrt(mm.sum())) if np.std(pnl) > 0 else None,
        })
    out_summary["C_tod"] = c_rows

    # ─────── D. Vol-regime interaction ───────
    log("D. Vol regime × strategy")
    d_rows = []
    vmsk = d["mask_pred_realized_vol_30s_ticks"].astype(bool) & np.isfinite(pred_vol)
    if vmsk.sum() > 100:
        q33, q67 = np.percentile(pred_vol[vmsk], [33, 67])
        for vlabel, vmin, vmax in [("low", -np.inf, q33),
                                    ("mid", q33, q67),
                                    ("high", q67, np.inf)]:
            in_v = vmsk & (pred_vol >= vmin) & (pred_vol < vmax) & long_mask
            idx_v = np.where(in_v)[0]
            if idx_v.size < 50: continue
            n_take = max(5, int(idx_v.size * 0.01))
            cell = idx_v[np.argsort(-np.abs(p1[idx_v]))[:n_take]]
            mm = mfe30_m[cell] & mae30_m[cell]
            if mm.sum() < 3: continue
            mfe = mfe30[cell][mm]; mae = mae30[cell][mm]
            for tp, sl in [(3,3), (4,3), (5,3)]:
                hit_tp = mfe >= tp; hit_sl = mae <= -sl
                pnl = np.where(hit_tp & ~hit_sl, tp - MARKET_COST_TICKS,
                       np.where(hit_sl, -sl - MARKET_COST_TICKS,
                                (mfe + mae)/2 - MARKET_COST_TICKS))
                d_rows.append({
                    "vol_regime": vlabel, "tp": tp, "sl": sl,
                    "n": int(mm.sum()),
                    "mean_pnl_t": float(np.mean(pnl)),
                    "win_rate_pct": float(np.mean(pnl > 0) * 100),
                    "sharpe_toy": float(np.mean(pnl) / np.std(pnl) * math.sqrt(mm.sum())) if np.std(pnl) > 0 else None,
                })
    out_summary["D_vol_regime"] = d_rows

    # ─────── E. Reversal-head EXIT signal ───────
    log("E. Reversal-head exit signal validation")
    e_rows = []
    # For each top1% long pick, does the realized 30s outcome correlate with the reversal head?
    # If pred_p_reversal_15s is high → expect quicker mean-reversion → smaller MFE realized
    long_top = idx_long[np.argsort(-np.abs(p1[idx_long]))[:max(5, int(idx_long.size * 0.01))]]
    mm = mfe30_m[long_top] & mae30_m[long_top]
    if mm.sum() > 50:
        rev15_v = rev15[long_top][mm]
        rev30_v = rev30[long_top][mm]
        mfe_v = mfe30[long_top][mm]
        mae_v = mae30[long_top][mm]
        # Bucket by rev15 percentile
        for rname, rmin, rmax in [("rev_low (<33%)", 0, 33),
                                   ("rev_mid (33-67%)", 33, 67),
                                   ("rev_high (>67%)", 67, 100)]:
            qmin = np.percentile(rev15_v, rmin)
            qmax = np.percentile(rev15_v, rmax) if rmax < 100 else np.inf
            keep = (rev15_v >= qmin) & (rev15_v < qmax) if rmax < 100 else (rev15_v >= qmin)
            if keep.sum() < 10: continue
            mfe_k = mfe_v[keep]; mae_k = mae_v[keep]
            # Strategy: tp4 sl3
            hit_tp = mfe_k >= 4; hit_sl = mae_k <= -3
            pnl = np.where(hit_tp & ~hit_sl, 4 - MARKET_COST_TICKS,
                   np.where(hit_sl, -3 - MARKET_COST_TICKS,
                            (mfe_k + mae_k)/2 - MARKET_COST_TICKS))
            e_rows.append({
                "rev15_bucket": rname, "n": int(keep.sum()),
                "mean_mfe_t": float(np.mean(mfe_k)),
                "mean_mae_t": float(np.mean(mae_k)),
                "mean_pnl_tp4sl3_t": float(np.mean(pnl)),
                "win_rate_pct": float(np.mean(pnl > 0)*100),
                "sharpe_toy": float(np.mean(pnl) / np.std(pnl) * math.sqrt(keep.sum())) if np.std(pnl) > 0 else None,
            })
    out_summary["E_reversal_exit"] = e_rows

    # ─────── F. Threshold sweep on entry-confidence + TP/SL grid ───────
    log("F. Optuna-style grid: entry threshold × tp × sl")
    f_rows = []
    long_p = p1[long_mask]
    long_idx = np.where(long_mask)[0]
    abs_thresholds = np.percentile(long_p, [50, 70, 85, 90, 95, 97.5, 99, 99.5])
    for thr in abs_thresholds:
        cell = long_idx[long_p >= thr]
        if cell.size < 30: continue
        mm = mfe30_m[cell] & mae30_m[cell]
        if mm.sum() < 20: continue
        mfe = mfe30[cell][mm]; mae = mae30[cell][mm]
        for tp, sl in [(2,2),(3,2),(3,3),(4,3),(5,3),(5,4),(8,5)]:
            hit_tp = mfe >= tp; hit_sl = mae <= -sl
            pnl = np.where(hit_tp & ~hit_sl, tp - MARKET_COST_TICKS,
                   np.where(hit_sl, -sl - MARKET_COST_TICKS,
                            (mfe + mae)/2 - MARKET_COST_TICKS))
            sharpe = float(np.mean(pnl) / np.std(pnl) * math.sqrt(mm.sum())) if np.std(pnl) > 0 else None
            f_rows.append({
                "entry_thr": float(thr), "tp": tp, "sl": sl,
                "n": int(mm.sum()),
                "mean_pnl_t": float(np.mean(pnl)),
                "win_rate_pct": float(np.mean(pnl > 0)*100),
                "tp_hit_pct": float(np.mean(hit_tp & ~hit_sl)*100),
                "sl_hit_pct": float(np.mean(hit_sl)*100),
                "sharpe_toy": sharpe,
            })
    # Sort by Sharpe-toy descending
    f_rows_sorted = sorted([r for r in f_rows if r.get("sharpe_toy") is not None],
                            key=lambda r: -r["sharpe_toy"])
    out_summary["F_grid_top20"] = f_rows_sorted[:20]
    out_summary["F_grid_all"] = f_rows
    log(f"  F: {len(f_rows)} grid cells, best Sharpe-toy={f_rows_sorted[0]['sharpe_toy']:.2f}")

    # Write outputs
    with open(OUT / "pass2_results.json", "w") as f:
        json.dump(out_summary, f, indent=2,
                  default=lambda o: None if isinstance(o,float) and not math.isfinite(o) else o)
    log(f"wrote {OUT / 'pass2_results.json'}")

    # Markdown summary
    md = ["# PASS 2 — LONG-Side IOC Drill-Down\n\n",
          "## A. Exit Strategy Comparison (per band, long IOC market)\n",
          "| Band | Strategy | n | Mean PnL (t) | WR% | Sharpe-toy |\n|---|---|---:|---:|---:|---:|\n"]
    for r in a_rows:
        if r["band"] not in ("Top0.5%", "Top1%", "Top5%"): continue
        md.append(f"| {r['band']} | {r['strategy']} | {r['n']} | "
                  f"{r['mean_pnl_t']:.3f} | {r['win_rate_pct']:.1f} | "
                  f"{r['sharpe_toy']} |\n")

    md.append("\n## B. Per-Day Robustness (selected best TP/SL × band)\n")
    md.append("| TP | SL | Band | Day | n | Mean PnL (t) | WR% | Sharpe-toy |\n|---|---|---|---|---:|---:|---:|---:|\n")
    for r in b_rows:
        md.append(f"| {r['tp']} | {r['sl']} | {r['band']} | {r['day']} | "
                  f"{r['n']} | {r['mean_pnl_t']:.3f} | "
                  f"{r['win_rate_pct']:.1f} | {r['sharpe_toy']} |\n")

    md.append("\n## C. Time-of-Day Performance (Long Top1%, tp4_sl3)\n")
    md.append("| Bucket | n | Mean PnL (t) | WR% | Sharpe-toy |\n|---|---:|---:|---:|---:|\n")
    for r in c_rows:
        md.append(f"| {r['label']} | {r['n']} | {r['mean_pnl_t']:.3f} | "
                  f"{r['win_rate_pct']:.1f} | {r['sharpe_toy']} |\n")

    md.append("\n## D. Vol-Regime × TP/SL\n")
    md.append("| Vol | TP | SL | n | Mean PnL (t) | WR% | Sharpe-toy |\n|---|---|---|---:|---:|---:|---:|\n")
    for r in d_rows:
        md.append(f"| {r['vol_regime']} | {r['tp']} | {r['sl']} | {r['n']} | "
                  f"{r['mean_pnl_t']:.3f} | {r['win_rate_pct']:.1f} | "
                  f"{r['sharpe_toy']} |\n")

    md.append("\n## E. Reversal-Head Exit Validation\n")
    md.append("| Rev15 bucket | n | Mean MFE (t) | Mean MAE (t) | TP4SL3 PnL (t) | WR% | Sharpe-toy |\n|---|---:|---:|---:|---:|---:|---:|\n")
    for r in e_rows:
        md.append(f"| {r['rev15_bucket']} | {r['n']} | {r['mean_mfe_t']:.2f} | "
                  f"{r['mean_mae_t']:.2f} | {r['mean_pnl_tp4sl3_t']:.3f} | "
                  f"{r['win_rate_pct']:.1f} | {r['sharpe_toy']} |\n")

    md.append("\n## F. TOP 20 (entry_thr, tp, sl) configs by Sharpe-toy\n")
    md.append("| Thr | TP | SL | n | Mean PnL (t) | WR% | TP-hit% | SL-hit% | Sharpe-toy |\n|---|---|---|---:|---:|---:|---:|---:|---:|\n")
    for r in f_rows_sorted[:20]:
        md.append(f"| {r['entry_thr']:.4f} | {r['tp']} | {r['sl']} | {r['n']} | "
                  f"{r['mean_pnl_t']:.3f} | {r['win_rate_pct']:.1f} | "
                  f"{r['tp_hit_pct']:.1f} | {r['sl_hit_pct']:.1f} | "
                  f"{r['sharpe_toy']:.2f} |\n")

    (OUT / "PASS2.md").write_text("".join(md))
    log("PASS 2 complete")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        sys.exit(1)
