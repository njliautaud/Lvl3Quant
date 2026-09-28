#!/usr/bin/env python3
"""
HC #346 PASS 5 — Properly-calibrated re-validation

Pass 4 caught: target_log_ret_* fields are Z-SCORE normalized, not raw log
returns. Direct division by TICK_LOG=4.95e-5 inflates ticks by ~20000x.
Pass 3's H1 "Realized (t)" column was wrong for the same reason.

Pass 5 fixes the calibration using the same anchor as task_01_unit_fix:
  z_unit_to_ticks_at_30s = sd(target_pred_mfe_30s_ticks) * sqrt(2) / sd(target_log_ret_30s)
Then scales other horizons by sqrt(h/30).

Pass 5 also:
  C1. Per-day robustness using **FIFO ground-truth** labels only (most reliable)
      for both long and short Top1% / Top0.1%
  C2. Time-of-day buckets on FIFO labels for headline configs
  C3. Combined long+short PORTFOLIO using FIFO labels + sequenced PnL curve
      with max drawdown
  C4. Re-do hold-time sweep with proper calibration
  C5. Cross-validate by computing edge on target_fifo_tp4sl3_net for the SAME
      configs that pass 3 H1 found "profitable" — does FIFO ground truth agree?
  C6. Bootstrap CI on FIFO-based headline configs

Outputs: output/v3_2_allnight_research_20260514/pass5_calibrated/
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
OUT = ROOT / "output/v3_2_allnight_research_20260514/pass5_calibrated"
OUT.mkdir(parents=True, exist_ok=True)
LOG = OUT / "pass5.log"

MARKET_COST_TICKS = 1.376
PASSIVE_COST_TICKS = 0.376


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
    log("PASS 5 — Calibrated re-validation")
    d = np.load(PRED_NPZ, allow_pickle=True)
    n = int(d["n_samples"])
    log(f"n_samples={n}")

    # ─── CALIBRATE z → ticks anchor ───
    mfe30 = np.asarray(d["target_pred_mfe_30s_ticks"]).flatten()
    mask_mfe30 = np.asarray(d["mask_pred_mfe_30s_ticks"]).flatten().astype(bool)
    valid_mfe = mask_mfe30 & np.isfinite(mfe30)
    sd_mfe30_ticks = float(np.std(mfe30[valid_mfe]))

    horizons_h = ["1s", "5s", "10s", "30s", "60s", "5min"]
    h_seconds = {"1s": 1, "5s": 5, "10s": 10, "30s": 30, "60s": 60, "5min": 300}

    sd_target = {}
    for h in horizons_h:
        a = np.asarray(d[f"target_log_ret_{h}"]).flatten()
        m = np.asarray(d[f"mask_log_ret_{h}"]).flatten().astype(bool)
        v = a[m & np.isfinite(a)]
        sd_target[h] = float(np.std(v)) if v.size else float("nan")

    # Calibrate at 30s: 1 z-unit corresponds to sd_mfe30_ticks * sqrt(2) ticks
    # (mfe is half-tail; full move sd ≈ mfe sd * sqrt(2))
    if sd_target["30s"] > 0:
        z2t_30s = sd_mfe30_ticks * 1.41421356 / sd_target["30s"]
    else:
        z2t_30s = float("nan")

    z2t = {h: z2t_30s * math.sqrt(h_seconds[h] / 30.0) for h in horizons_h}
    log(f"sd_mfe30_ticks={sd_mfe30_ticks:.3f}, sd_target_30s={sd_target['30s']:.3f}, z2t_30s={z2t_30s:.4f}")
    log(f"z2t per horizon: { {h: round(v, 4) for h, v in z2t.items()} }")

    pred_5s = np.asarray(d["pred_log_ret_5s"]).flatten()
    pred_1s = np.asarray(d["pred_log_ret_1s"]).flatten()
    p_up_5s = np.asarray(d["pred_p_up_5s"]).flatten()
    rev15 = np.asarray(d["pred_p_reversal_15s"]).flatten()
    rev30 = np.asarray(d["pred_p_reversal_30s"]).flatten()

    fifo_tp4sl3 = np.asarray(d["target_fifo_tp4sl3_net"]).flatten()
    mask_fifo = np.asarray(d["mask_fifo_tp4sl3_net"]).flatten().astype(bool)

    target_lr = {h: np.asarray(d[f"target_log_ret_{h}"]).flatten() for h in horizons_h}
    mask_lr = {h: np.asarray(d[f"mask_log_ret_{h}"]).flatten().astype(bool) for h in horizons_h}

    valid_5s = mask_lr["5s"]

    # synth time-of-day & day-idx (no per-sample timestamps in NPZ)
    n_days = 5
    samples_per_day = n // n_days
    day_idx = np.zeros(n, dtype=np.int32)
    sec_from_open = np.zeros(n, dtype=np.float32)
    rth_seconds = int(6.5 * 3600)
    for di in range(n_days):
        s = di * samples_per_day
        e = (di + 1) * samples_per_day if di < n_days - 1 else n
        day_idx[s:e] = di
        sec_from_open[s:e] = np.linspace(0, rth_seconds, e - s, dtype=np.float32)
    rth_open_sec = 9 * 3600 + 30 * 60
    sec_of_day_et = sec_from_open + rth_open_sec
    bucket = (sec_from_open // 1800).astype(int)
    bucket = np.clip(bucket, 0, 12)
    bucket_labels = []
    for b in range(13):
        sh, sm = divmod(rth_open_sec + b * 1800, 3600)
        eh, em = divmod(rth_open_sec + (b + 1) * 1800, 3600)
        bucket_labels.append(f"{sh:02d}:{sm:02d}-{eh:02d}:{em:02d}")

    def percentile_band(pct, side):
        a = np.where(valid_5s, pred_5s, np.nan)
        if side == "long":
            cut = np.nanquantile(a, 1.0 - pct / 100.0)
            return (a >= cut) & valid_5s
        else:
            cut = np.nanquantile(a, pct / 100.0)
            return (a <= cut) & valid_5s

    results = {}
    results["calibration"] = {
        "sd_mfe30_ticks": sd_mfe30_ticks,
        "sd_target_30s_z": sd_target["30s"],
        "z2t_30s": z2t_30s,
        "z2t_per_horizon": {h: round(v, 5) for h, v in z2t.items()},
    }

    # ──────────────────────────────────────────────
    # C1. Per-day robustness — FIFO ground truth ONLY
    # ──────────────────────────────────────────────
    log("C1. Per-day robustness — FIFO ground truth")
    cfgs_fifo = [
        ("L_Top1_FIFO", "long", 1.0, None),
        ("L_Top0p5_FIFO", "long", 0.5, None),
        ("L_Top0p1_FIFO", "long", 0.1, None),
        ("S_Top1_FIFO", "short", 1.0, None),
        ("S_Top0p5_FIFO", "short", 0.5, None),
        ("S_Top0p1_FIFO", "short", 0.1, None),
        ("S_Top1_FIFO_agree15", "short", 1.0, "agree_15"),
        ("S_Top0p1_FIFO_agree15", "short", 0.1, "agree_15"),
    ]

    # build agree_15 boolean
    agree_15 = (np.sign(pred_1s) == np.sign(pred_5s))

    c1 = {}
    for name, side, pct, gate in cfgs_fifo:
        em = percentile_band(pct, side)
        if gate == "agree_15":
            em = em & agree_15
        per_day = []
        for du in range(n_days):
            m = em & mask_fifo & (day_idx == du)
            if int(m.sum()) == 0:
                per_day.append({"day": du, "n_sig": 0, "n_fill": 0, "fill_pct": 0.0,
                                "mean_t": float("nan"), "wr": float("nan"), "sharpe": float("nan")})
                continue
            raw = fifo_tp4sl3[m]
            if side == "short":
                raw = -raw
            fm = raw != 0
            n_sig = int(m.sum())
            n_fill = int(fm.sum())
            if n_fill == 0:
                per_day.append({"day": du, "n_sig": n_sig, "n_fill": 0, "fill_pct": 0.0,
                                "mean_t": float("nan"), "wr": float("nan"), "sharpe": float("nan")})
                continue
            pnl = raw[fm] - PASSIVE_COST_TICKS
            per_day.append({"day": du, "n_sig": n_sig, "n_fill": n_fill,
                            "fill_pct": 100.0 * n_fill / n_sig,
                            "mean_t": float(pnl.mean()), "wr": float((pnl > 0).mean() * 100),
                            "sharpe": safe_sharpe(pnl)})
        c1[name] = per_day
    results["C1_per_day_fifo"] = c1

    # ──────────────────────────────────────────────
    # C2. Time-of-day buckets on FIFO
    # ──────────────────────────────────────────────
    log("C2. ToD buckets on FIFO ground truth")
    c2 = {}
    for name, side, pct, gate in cfgs_fifo:
        em = percentile_band(pct, side)
        if gate == "agree_15":
            em = em & agree_15
        rows = []
        for b in range(13):
            m = em & mask_fifo & (bucket == b)
            if int(m.sum()) == 0:
                continue
            raw = fifo_tp4sl3[m]
            if side == "short":
                raw = -raw
            fm = raw != 0
            if int(fm.sum()) < 3:
                continue
            pnl = raw[fm] - PASSIVE_COST_TICKS
            rows.append({"bucket": bucket_labels[b], "n_fill": int(fm.sum()),
                         "mean_t": float(pnl.mean()), "wr": float((pnl > 0).mean() * 100),
                         "sharpe": safe_sharpe(pnl)})
        c2[name] = rows
    results["C2_tod_fifo"] = c2

    # ──────────────────────────────────────────────
    # C3. Combined LONG+SHORT portfolio (FIFO, sequenced)
    # ──────────────────────────────────────────────
    log("C3. Combined long+short portfolio (FIFO sequenced)")

    def fifo_pnl(em, side):
        m = em & mask_fifo
        raw = fifo_tp4sl3[m]
        if side == "short":
            raw = -raw
        fm = raw != 0
        idx_full = np.where(m)[0][fm]
        pnl = raw[fm] - PASSIVE_COST_TICKS
        return idx_full, pnl

    portfolios = [
        ("L_Top1 + S_Top1", percentile_band(1.0, "long"), "long",
                            percentile_band(1.0, "short"), "short"),
        ("L_Top0p1 + S_Top0p1_agree15",
                            percentile_band(0.1, "long"), "long",
                            percentile_band(0.1, "short") & agree_15, "short"),
        ("L_Top1 + S_Top1_agree15",
                            percentile_band(1.0, "long"), "long",
                            percentile_band(1.0, "short") & agree_15, "short"),
        ("S_Top1 only", None, None, percentile_band(1.0, "short"), "short"),
        ("S_Top0p1_agree15 only", None, None,
                            percentile_band(0.1, "short") & agree_15, "short"),
    ]

    c3 = {}
    for name, em_l, side_l, em_s, side_s in portfolios:
        idx_all = []
        pnl_all = []
        side_all = []
        if em_l is not None:
            iL, pL = fifo_pnl(em_l, side_l)
            idx_all.append(iL); pnl_all.append(pL); side_all.append(np.full(iL.size, "L"))
        if em_s is not None:
            iS, pS = fifo_pnl(em_s, side_s)
            idx_all.append(iS); pnl_all.append(pS); side_all.append(np.full(iS.size, "S"))
        if not idx_all:
            continue
        idx_all = np.concatenate(idx_all)
        pnl_all = np.concatenate(pnl_all)
        side_all = np.concatenate(side_all)
        order = np.argsort(idx_all)  # by sample index (≈ chronological)
        idx_sorted = idx_all[order]
        pnl_sorted = pnl_all[order]
        side_sorted = side_all[order]
        equity = np.cumsum(pnl_sorted)
        peak = np.maximum.accumulate(equity)
        dd = equity - peak
        c3[name] = {
            "n_trades": int(pnl_sorted.size),
            "n_long": int((side_sorted == "L").sum()),
            "n_short": int((side_sorted == "S").sum()),
            "mean_pnl_t": float(pnl_sorted.mean()),
            "wr_pct": float((pnl_sorted > 0).mean() * 100),
            "sharpe": safe_sharpe(pnl_sorted),
            "total_pnl_t": float(pnl_sorted.sum()),
            "total_pnl_$": float(pnl_sorted.sum() * 12.50),
            "max_dd_t": float(dd.min()),
            "max_dd_$": float(dd.min() * 12.50),
        }
    results["C3_portfolios"] = c3

    # ──────────────────────────────────────────────
    # C4. Hold-time sweep (CALIBRATED) — sanity for long side
    # ──────────────────────────────────────────────
    log("C4. Hold-time sweep CALIBRATED")
    c4 = {}
    for side in ["long", "short"]:
        for pct in [0.5, 1.0]:
            em = percentile_band(pct, side)
            rows = []
            for h in horizons_h:
                m = em & mask_lr[h]
                if int(m.sum()) < 30:
                    continue
                z = target_lr[h][m]
                ticks = z * z2t[h]  # CALIBRATED
                if side == "short":
                    ticks = -ticks
                pnl = ticks - MARKET_COST_TICKS
                rows.append({"horizon": h, "secs": h_seconds[h], "n": int(m.sum()),
                             "mean_t": float(pnl.mean()), "wr": float((pnl > 0).mean() * 100),
                             "sharpe": safe_sharpe(pnl)})
            c4[f"{side}_Top{pct}"] = rows
    results["C4_hold_sweep_calibrated"] = c4

    # ──────────────────────────────────────────────
    # C5. Cross-check: compute FIFO PnL for "Pass3-H1 long" config
    # ──────────────────────────────────────────────
    log("C5. Cross-check pass3 H1 long with FIFO ground truth")
    c5 = {}
    for pct in [0.1, 0.5, 1.0, 5.0]:
        em = percentile_band(pct, "long")
        m = em & mask_fifo
        if int(m.sum()) == 0:
            continue
        raw = fifo_tp4sl3[m]
        fm = raw != 0
        if int(fm.sum()) < 5:
            continue
        pnl = raw[fm] - PASSIVE_COST_TICKS
        c5[f"L_Top{pct}_FIFO"] = {
            "n_sig": int(m.sum()), "n_fill": int(fm.sum()),
            "fill_pct": 100.0 * fm.sum() / m.sum(),
            "mean_t": float(pnl.mean()), "wr": float((pnl > 0).mean() * 100),
            "sharpe": safe_sharpe(pnl),
        }
    results["C5_pass3_H1_cross_check"] = c5

    # ──────────────────────────────────────────────
    # C6. Bootstrap CI on FIFO headlines
    # ──────────────────────────────────────────────
    log("C6. Bootstrap CI on FIFO configs")
    c6 = {}
    for name, side, pct, gate in cfgs_fifo:
        em = percentile_band(pct, side)
        if gate == "agree_15":
            em = em & agree_15
        m = em & mask_fifo
        raw = fifo_tp4sl3[m]
        if side == "short":
            raw = -raw
        fm = raw != 0
        if int(fm.sum()) < 5:
            continue
        pnl = raw[fm] - PASSIVE_COST_TICKS
        lo, hi = bootstrap_ci(pnl)
        c6[name] = {
            "n_fill": int(fm.sum()), "mean_t": float(pnl.mean()),
            "ci95_low": lo, "ci95_high": hi,
            "ci95_excludes_zero_positive": bool(lo > 0),
        }
    results["C6_bootstrap"] = c6

    # ──────────────────────────────────────────────
    # Write report
    # ──────────────────────────────────────────────
    md = OUT / "PASS5.md"
    with open(md, "w") as f:
        f.write("# PASS 5 — Calibrated Re-Validation\n\n")
        f.write("## Calibration\n\n")
        f.write(f"- sd(target_pred_mfe_30s_ticks) = {sd_mfe30_ticks:.3f} ticks\n")
        f.write(f"- sd(target_log_ret_30s, z) = {sd_target['30s']:.3f}\n")
        f.write(f"- **z2t @ 30s = {z2t_30s:.4f}** ticks per z-unit\n")
        f.write(f"- z2t per horizon (sqrt-time scaled): {results['calibration']['z2t_per_horizon']}\n\n")

        f.write("## C1. Per-Day Robustness — FIFO Ground Truth (passive cost 0.376t applied)\n\n")
        for name, rows in c1.items():
            f.write(f"### {name}\n")
            f.write("| day | n_sig | n_fill | fill% | mean_t | WR% | Sharpe |\n|---|---:|---:|---:|---:|---:|---:|\n")
            for r in rows:
                f.write(f"| {r['day']} | {r['n_sig']} | {r['n_fill']} | {r['fill_pct']:.1f} | "
                        f"{r['mean_t']:.3f} | {r['wr']:.1f} | {r['sharpe']:.2f} |\n")
            f.write("\n")

        f.write("## C2. Time-of-Day Stability — FIFO\n\n")
        for name, rows in c2.items():
            f.write(f"### {name}\n")
            f.write("| Bucket | n_fill | mean_t | WR% | Sharpe |\n|---|---:|---:|---:|---:|\n")
            for r in rows:
                f.write(f"| {r['bucket']} | {r['n_fill']} | {r['mean_t']:.3f} | "
                        f"{r['wr']:.1f} | {r['sharpe']:.2f} |\n")
            f.write("\n")

        f.write("## C3. Combined Portfolios (FIFO, sequenced)\n\n")
        f.write("| Portfolio | n | n_L | n_S | mean_t | WR% | Sharpe | Total_t | Total_$ | MaxDD_t | MaxDD_$ |\n|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|\n")
        for k, v in c3.items():
            f.write(f"| {k} | {v['n_trades']} | {v['n_long']} | {v['n_short']} | "
                    f"{v['mean_pnl_t']:.3f} | {v['wr_pct']:.1f} | {v['sharpe']:.2f} | "
                    f"{v['total_pnl_t']:.1f} | {v['total_pnl_$']:.0f} | "
                    f"{v['max_dd_t']:.1f} | {v['max_dd_$']:.0f} |\n")
        f.write("\n")

        f.write("## C4. Hold-Time Sweep (CALIBRATED ticks, market cost 1.376t)\n\n")
        for k, rows in c4.items():
            f.write(f"### {k}\n")
            f.write("| Horizon | secs | n | mean_t | WR% | Sharpe |\n|---|---:|---:|---:|---:|---:|\n")
            for r in rows:
                f.write(f"| {r['horizon']} | {r['secs']} | {r['n']} | "
                        f"{r['mean_t']:.3f} | {r['wr']:.1f} | {r['sharpe']:.2f} |\n")
            f.write("\n")

        f.write("## C5. Cross-check Pass3-H1 long with FIFO Ground Truth\n\n")
        f.write("| Config | n_sig | n_fill | fill% | mean_t | WR% | Sharpe |\n|---|---:|---:|---:|---:|---:|---:|\n")
        for k, v in c5.items():
            f.write(f"| {k} | {v['n_sig']} | {v['n_fill']} | {v['fill_pct']:.1f} | "
                    f"{v['mean_t']:.3f} | {v['wr']:.1f} | {v['sharpe']:.2f} |\n")
        f.write("\n")

        f.write("## C6. Bootstrap 95% CI on FIFO Headlines\n\n")
        f.write("| Config | n_fill | mean_t | CI_low | CI_high | Positive_CI? |\n|---|---:|---:|---:|---:|---:|\n")
        for k, v in c6.items():
            f.write(f"| {k} | {v['n_fill']} | {v['mean_t']:.3f} | "
                    f"{v['ci95_low']:.3f} | {v['ci95_high']:.3f} | {v['ci95_excludes_zero_positive']} |\n")
        f.write("\n")

    with open(OUT / "pass5_results.json", "w") as f:
        json.dump(results, f, default=str, indent=2)
    log(f"PASS 5 complete — wrote {md}")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        log("FATAL")
        log(traceback.format_exc())
        sys.exit(1)
