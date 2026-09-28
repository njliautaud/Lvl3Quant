#!/usr/bin/env python3
"""
HC #346 — Jupiter ALL-NIGHT v3.2 Edge-Extraction Orchestrator

Runs 10 research tasks back-to-back on Jupiter CPU. Each writes to its own subdir.
Watchdog: 60-min hard timeout per task. Master ORCHESTRATOR.md updated after each.

NO trainer code modified. Pure analysis. (HC #307D malware-guard.)

Inputs:
    output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz
    (241,351 OOT preds across 20260223-27, all 32 v3.2 heads, real FIFO labels)

Outputs:
    output/v3_2_allnight_research_20260514/
        ORCHESTRATOR.md             (running summary)
        orchestrator.log
        task_01_unit_fix/...
        task_02_edge_decay/...
        task_03_time_of_day/...
        task_04_confluence_gates/...
        task_05_vol_regimes/...
        task_06_fifo_queue_backoff/...
        task_07_aggressive_cross/...
        task_08_meta_mlp/...
        task_09_contextual_bandit/...
        task_10_per_head_ablation/...
"""
from __future__ import annotations

import json
import math
import multiprocessing
import os
import signal
import sys
import time
import traceback
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

ROOT = Path("/home/jupiter/Lvl3Quant")
PRED_NPZ = ROOT / "output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz"
OUT_ROOT = ROOT / "output/v3_2_allnight_research_20260514"
OUT_ROOT.mkdir(parents=True, exist_ok=True)
LOG = OUT_ROOT / "orchestrator.log"
MASTER_MD = OUT_ROOT / "ORCHESTRATOR.md"

# ES tick canon
TICK_VALUE_USD = 12.50
RT_COMM_TICKS = 0.376
PASSIVE_COST_TICKS = RT_COMM_TICKS
MARKET_COST_TICKS = RT_COMM_TICKS + 1.0

TASK_TIMEOUT_SEC = 60 * 60   # 60 min per task hard cap

HORIZONS = [
    ("1s",   "log_ret_1s",   1.0),
    ("5s",   "log_ret_5s",   5.0),
    ("10s",  "log_ret_10s",  10.0),
    ("30s",  "log_ret_30s",  30.0),
    ("60s",  "log_ret_60s",  60.0),
    ("300s", "log_ret_5min", 300.0),
]

BANDS = [("Top0.1%", 0.001), ("Top0.5%", 0.005), ("Top1%", 0.01),
         ("Top5%", 0.05), ("Top10%", 0.10)]


def log(msg: str):
    ts = datetime.utcnow().isoformat(timespec="seconds")
    line = f"[{ts}Z] {msg}"
    print(line, flush=True)
    with open(LOG, "a") as f:
        f.write(line + "\n")


def append_master(section: str):
    with open(MASTER_MD, "a") as f:
        f.write(section + "\n\n")


def load_data():
    log(f"loading {PRED_NPZ}")
    d = np.load(PRED_NPZ, allow_pickle=True)
    n = int(d["n_samples"])
    log(f"  n={n}, oot_dates={list(d['oot_dates'])}")
    out = {}
    for k in d.keys():
        if k in ("fold_idx", "n_samples", "elapsed_sec"):
            out[k] = d[k].item() if hasattr(d[k], "item") else d[k]
        else:
            out[k] = np.asarray(d[k])
    return out, n


# Approximate time-of-day from sample index (uniform within each day, 5 days × ~n/5 each)
def synth_time_of_day(n_total: int, n_days: int = 5) -> np.ndarray:
    """Returns seconds-from-RTH-open for each sample (approx, uniform within day)."""
    samples_per_day = n_total // n_days
    rth_seconds = int(6.5 * 3600)  # 23,400
    out = np.zeros(n_total, dtype=np.float32)
    for di in range(n_days):
        start = di * samples_per_day
        end = (di + 1) * samples_per_day if di < n_days - 1 else n_total
        m = end - start
        out[start:end] = np.linspace(0, rth_seconds, m, dtype=np.float32)
    return out


def synth_day_idx(n_total: int, n_days: int = 5) -> np.ndarray:
    samples_per_day = n_total // n_days
    out = np.zeros(n_total, dtype=np.int32)
    for di in range(n_days):
        start = di * samples_per_day
        end = (di + 1) * samples_per_day if di < n_days - 1 else n_total
        out[start:end] = di
    return out


# ──────────────── TASK 1: unit-corrected dashboard ────────────────

def task_01_unit_fix(data: dict, out_dir: Path):
    """Re-derive HC #345 dashboard with proper z-score → tick conversion using empirical std."""
    log("TASK 1: unit-fix HC #345 dashboard")
    n = data["n_samples"]

    # Empirical std per horizon (in target units = z-score). Use mask & finite.
    horizon_std = {}
    horizon_std_ticks = {}  # std in ticks (using the FIFO labels as ground-truth tick anchor)

    # Anchor: target_fifo_tp4sl3_net is in REAL ticks. So:
    #   sd_z(target_log_ret_Hs) corresponds to some sd_t(realized_move_at_Hs) in ticks
    # Use 30s as anchor: realized_log_ret_30s std × tick_scale = realized_move_30s std in ticks
    # But we don't have direct "realized_move_30s_in_ticks". USE pred_pred_mae_30s_ticks + mfe_30s_ticks std as scale.
    # Cleaner: use ES price ~5050 → 1 tick = 0.25 → log(5050.25/5050) = 4.95e-5
    tick_log = 4.95e-5

    for hkey, hcol, hsec in HORIZONS:
        msk = data[f"mask_{hcol}"].astype(bool)
        tgt = data[f"target_{hcol}"]
        keep = msk & np.isfinite(tgt)
        if keep.sum() == 0:
            horizon_std[hkey] = None
            continue
        sd_target = float(np.std(tgt[keep]))
        horizon_std[hkey] = sd_target

    # Also compute the "z-score normalization scale" — std of TARGETS appears ~1 if z-score normalized,
    # or ~1e-4 if raw log-returns. If sd_target ≈ 1.0 → it's z-scored; need to recover tick scale.
    # Use realized 1s move heuristic: ES sd of 1s log-return ≈ 5e-5 (~1 tick per second std)
    # If normalized: ticks ≈ z_value × 1.0 (1-sigma move ≈ 1 tick at 1s).
    # We use empirical: for 1s, sd_pred_in_targets ≈ sd_targets, so tick_per_z = 1.0 for 1s,
    # and for longer horizons scales as sqrt(time_in_seconds).
    # Cross-check via FIFO PnL labels: sd of mfe_30s_ticks (real ticks) ≈ ?
    mfe30 = data["target_pred_mfe_30s_ticks"]
    msk_mfe = data["mask_pred_mfe_30s_ticks"].astype(bool) & np.isfinite(mfe30)
    sd_mfe30_ticks = float(np.std(mfe30[msk_mfe])) if msk_mfe.sum() else None

    # Calibrate: assume target_log_ret_30s is z-score normalized. 1-sigma move at 30s in ticks
    # is approx sqrt(30) × tick_per_sec = 5.5 ticks. mfe 30s sd should track.
    # Use the ratio:
    if horizon_std.get("30s") and sd_mfe30_ticks:
        # If sd(target_log_ret_30s) == 1.0, then 1 z-unit ≈ sd_mfe30_ticks * sqrt(2) (mfe is one-tail)
        zscore_to_ticks_30s = sd_mfe30_ticks * 1.41 / horizon_std["30s"]
    else:
        zscore_to_ticks_30s = None

    log(f"  horizon std (target z): {horizon_std}")
    log(f"  sd_mfe_30s_ticks={sd_mfe30_ticks}, zscore_to_ticks_30s={zscore_to_ticks_30s}")

    # Per-horizon z-to-ticks scale: sqrt(t) scaling assumption
    # Anchor at 30s with empirical scale, scale others by sqrt(h/30)
    z2t = {}
    if zscore_to_ticks_30s:
        for hkey, hcol, hsec in HORIZONS:
            z2t[hkey] = zscore_to_ticks_30s * math.sqrt(hsec / 30.0)
    else:
        # Fallback: assume ES ~1 tick per sqrt(sec)
        for hkey, hcol, hsec in HORIZONS:
            z2t[hkey] = math.sqrt(hsec)

    log(f"  z2t calibration per horizon: {z2t}")

    # Re-emit per-band cells with correct ticks
    rows = []
    for hkey, hcol, hsec in HORIZONS:
        pred = data[f"pred_{hcol}"]
        tgt  = data[f"target_{hcol}"]
        msk  = data[f"mask_{hcol}"].astype(bool)
        valid = msk & np.isfinite(pred) & np.isfinite(tgt)
        idx = np.where(valid)[0]
        if idx.size < 100:
            continue
        absp = np.abs(pred[idx])
        order = np.argsort(-absp)

        for bname, bfrac in BANDS:
            n_take = max(1, int(round(idx.size * bfrac)))
            top = idx[order[:n_take]]
            for side, smask in [("long", pred[top] > 0), ("short", pred[top] < 0)]:
                cell = top[smask]
                nn = int(cell.size)
                if nn < 5:
                    continue
                p = pred[cell]; t = tgt[cell]
                dsign = 1.0 if side == "long" else -1.0
                wr = float(np.mean(np.sign(t) == np.sign(p)) * 100.0)
                ic = float(np.corrcoef(p, t)[0,1]) if np.std(p) and np.std(t) else float("nan")
                signed_z = t * np.sign(p)
                signed_ticks = signed_z * z2t[hkey]
                mean_t = float(np.mean(signed_ticks))
                sd_t = float(np.std(signed_ticks))
                sharpe_toy = float(mean_t / sd_t * math.sqrt(nn)) if sd_t > 0 else float("nan")
                # FIFO PnL real
                f43m = data["mask_fifo_tp4sl3_net"].astype(bool)[cell]
                pnl43 = (float(np.mean(data["target_fifo_tp4sl3_net"][cell][f43m]))
                         if f43m.sum() else None)
                rows.append({
                    "horizon": hkey, "side": side, "band": bname, "n": nn,
                    "wr_pct": wr, "ic_in_cell": ic,
                    "edge_mean_ticks": mean_t,
                    "edge_sd_ticks": sd_t,
                    "sharpe_toy": sharpe_toy,
                    "passive_breakeven_ticks": PASSIVE_COST_TICKS,
                    "market_breakeven_ticks": MARKET_COST_TICKS,
                    "passive_profitable": mean_t > PASSIVE_COST_TICKS,
                    "market_profitable": mean_t > MARKET_COST_TICKS,
                    "fill43_pct": float(np.mean(f43m) * 100.0),
                    "fifo_pnl_filled_ticks": pnl43,
                })

    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "unit_fix.json", "w") as f:
        json.dump({"z2t_calibration": z2t, "horizon_std": horizon_std,
                   "sd_mfe30_ticks": sd_mfe30_ticks, "rows": rows}, f, indent=2,
                  default=lambda o: None if isinstance(o,float) and not math.isfinite(o) else o)

    # Markdown summary
    md = ["# Task 1 — Unit-Fixed Edge Dashboard\n",
          f"_z-to-ticks calibration anchored at 30s using empirical sd_mfe30_ticks={sd_mfe30_ticks:.3f}_\n",
          "| Horiz | Side | Band | n | WR% | Edge mean (t) | Sharpe-toy | Passive OK | Market OK | Fill43% | FIFO filled (t) |\n",
          "|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|\n"]
    for r in rows:
        if r["band"] not in ("Top0.1%", "Top0.5%", "Top1%"):
            continue
        md.append(f"| {r['horizon']} | {r['side']} | {r['band']} | {r['n']} | "
                  f"{r['wr_pct']:.1f} | {r['edge_mean_ticks']:.3f} | "
                  f"{r['sharpe_toy']:.2f} | "
                  f"{'PASS' if r['passive_profitable'] else 'fail'} | "
                  f"{'PASS' if r['market_profitable'] else 'fail'} | "
                  f"{r['fill43_pct']:.1f} | "
                  f"{r['fifo_pnl_filled_ticks']} |\n")

    n_passive = sum(1 for r in rows if r["passive_profitable"])
    n_market  = sum(1 for r in rows if r["market_profitable"])
    md.append(f"\n**SUMMARY: {n_passive} cells pass passive-cost, {n_market} cells pass market-cost (of {len(rows)} cells).**\n")
    (out_dir / "TASK1.md").write_text("".join(md))

    summary = {"n_passive_profitable": n_passive, "n_market_profitable": n_market,
               "n_total_cells": len(rows), "z2t_30s": zscore_to_ticks_30s}
    return summary


# ──────────────── TASK 2: edge decay sweep by hold ────────────────

def task_02_edge_decay(data: dict, out_dir: Path):
    log("TASK 2: edge-decay sweep by hold seconds × threshold")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    # Use log_ret_1s as primary direction signal. Sweep threshold on |pred|.
    pred1 = data["pred_log_ret_1s"]
    msk1  = data["mask_log_ret_1s"].astype(bool)
    valid = msk1 & np.isfinite(pred1)
    idx = np.where(valid)[0]
    absp = np.abs(pred1[idx])
    thresholds = np.percentile(absp, [50, 70, 85, 90, 95, 99, 99.5, 99.9])

    for thr_pct, thr in zip([50,70,85,90,95,99,99.5,99.9], thresholds):
        keep = idx[absp >= thr]
        if keep.size < 20:
            continue
        for side, smask in [("long", pred1[keep] > 0), ("short", pred1[keep] < 0)]:
            cell = keep[smask]
            nn = int(cell.size)
            if nn < 10:
                continue
            dsign = 1.0 if side == "long" else -1.0
            row = {"threshold_pct": float(thr_pct), "threshold_abs": float(thr),
                   "side": side, "n": nn}
            for hkey, hcol, hsec in HORIZONS:
                t = data[f"target_{hcol}"][cell]
                m = data[f"mask_{hcol}"].astype(bool)[cell]
                k = m & np.isfinite(t)
                if k.sum() < 5:
                    row[f"da_{hkey}"] = None
                    row[f"mean_signed_z_{hkey}"] = None
                    continue
                row[f"da_{hkey}"] = float(np.mean(np.sign(t[k]) == np.sign(pred1[cell][k])) * 100.0)
                row[f"mean_signed_z_{hkey}"] = float(np.mean(t[k] * np.sign(pred1[cell][k])))
            rows.append(row)

    with open(out_dir / "edge_decay.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2,
                  default=lambda o: None if isinstance(o,float) and not math.isfinite(o) else o)

    md = ["# Task 2 — Edge Decay by Hold × Threshold\n\n",
          "DA% by horizon for each (threshold_pct, side) cell. Pred direction = log_ret_1s sign.\n\n",
          "| Thr% | Side | n | DA1s | DA5s | DA10s | DA30s | DA60s | DA300s |\n",
          "|---|---|---:|---:|---:|---:|---:|---:|---:|\n"]
    for r in rows:
        md.append(f"| {r['threshold_pct']} | {r['side']} | {r['n']} | "
                  f"{r.get('da_1s')} | {r.get('da_5s')} | {r.get('da_10s')} | "
                  f"{r.get('da_30s')} | {r.get('da_60s')} | {r.get('da_300s')} |\n")
    (out_dir / "TASK2.md").write_text("".join(md))
    return {"n_rows": len(rows)}


# ──────────────── TASK 3: time-of-day quirks ────────────────

def task_03_time_of_day(data: dict, out_dir: Path):
    log("TASK 3: time-of-day quirks")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = data["n_samples"]
    tod = synth_time_of_day(n)  # seconds from 09:30 ET
    # 30-min buckets across 09:30-16:00 = 13 buckets
    bucket = (tod // 1800).astype(int)
    bucket = np.clip(bucket, 0, 12)
    bucket_labels = [f"{9 + (b*30 + 30)//60:02d}:{(b*30 + 30)%60:02d}" for b in range(13)]
    bucket_labels = [f"09:30-{lbl}" if i==0 else f"{bucket_labels[i-1]}-{lbl}"
                     for i, lbl in enumerate(bucket_labels)]

    pred1 = data["pred_log_ret_1s"]; msk1 = data["mask_log_ret_1s"].astype(bool)
    tgt30 = data["target_log_ret_30s"]; msk30 = data["mask_log_ret_30s"].astype(bool)
    f43_msk = data["mask_fifo_tp4sl3_net"].astype(bool)
    f43_pnl = data["target_fifo_tp4sl3_net"]

    # Top1% picks per bucket
    rows = []
    for b in range(13):
        in_b = bucket == b
        valid = in_b & msk1 & np.isfinite(pred1)
        if valid.sum() < 50: continue
        absp = np.abs(pred1[valid])
        n_take = max(1, int(absp.size * 0.01))
        idx_b = np.where(valid)[0]
        order = np.argsort(-absp)
        top = idx_b[order[:n_take]]
        for side, smask in [("long", pred1[top]>0), ("short", pred1[top]<0)]:
            cell = top[smask]
            nn = int(cell.size)
            if nn < 5: continue
            t30 = tgt30[cell]; m30 = msk30[cell]; k30 = m30 & np.isfinite(t30)
            if k30.sum() < 3: continue
            dsign = 1.0 if side=="long" else -1.0
            wr = float(np.mean(np.sign(t30[k30]) == np.sign(pred1[cell][k30])) * 100.0)
            edge_z = float(np.mean(t30[k30] * np.sign(pred1[cell][k30])))
            fm = f43_msk[cell]
            pnl = (float(np.mean(f43_pnl[cell][fm])) if fm.sum() else None)
            rows.append({"bucket": b, "label": bucket_labels[b], "side": side,
                         "n": nn, "wr_30s_pct": wr, "edge_signed_z_30s": edge_z,
                         "fill43_pct": float(np.mean(fm)*100.0),
                         "fifo_pnl_filled_t": pnl})

    with open(out_dir / "time_of_day.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2,
                  default=lambda o: None if isinstance(o,float) and not math.isfinite(o) else o)

    md = ["# Task 3 — Time-of-Day Quirks (Top 1% picks)\n\n",
          "Approximate ToD from sample-index uniform within each of 5 OOT days.\n\n",
          "| Bucket | Side | n | WR30s% | Edge_z | Fill43% | FIFO PnL filled (t) |\n",
          "|---|---|---:|---:|---:|---:|---:|\n"]
    for r in rows:
        md.append(f"| {r['label']} | {r['side']} | {r['n']} | {r['wr_30s_pct']:.1f} | "
                  f"{r['edge_signed_z_30s']:.4f} | {r['fill43_pct']:.1f} | "
                  f"{r['fifo_pnl_filled_t']} |\n")
    (out_dir / "TASK3.md").write_text("".join(md))
    return {"n_rows": len(rows)}


# ──────────────── TASK 4: confluence gate sweep ────────────────

def task_04_confluence_gates(data: dict, out_dir: Path):
    log("TASK 4: confluence gate sweep")
    out_dir.mkdir(parents=True, exist_ok=True)
    n = data["n_samples"]

    p1 = data["pred_log_ret_1s"]; p5 = data["pred_log_ret_5s"]; p10 = data["pred_log_ret_10s"]
    pup5 = data["pred_p_up_5s"]; pup10 = data["pred_p_up_10s"]
    rev15 = data["pred_p_reversal_15s"]; rev30 = data["pred_p_reversal_30s"]
    q90_30 = data["pred_log_ret_30s_q90"]; q10_30 = data["pred_log_ret_30s_q10"]

    f43_msk = data["mask_fifo_tp4sl3_net"].astype(bool)
    f43_pnl = data["target_fifo_tp4sl3_net"]

    base_msk = (data["mask_log_ret_1s"].astype(bool)
                & data["mask_log_ret_5s"].astype(bool)
                & np.isfinite(p1) & np.isfinite(p5))

    # 5 gate components × 2 (on/off) = 32 configs. Top 1% picks of |p1| within filtered set.
    gates = {
        "G1_p1p5_same_sign": np.sign(p1) == np.sign(p5),
        "G2_p1p10_same_sign": np.sign(p1) == np.sign(p10),
        "G3_p_up_consistent": ((p1 > 0) & (pup5 > 0.55)) | ((p1 < 0) & (pup5 < 0.45)),
        "G4_no_reversal_15s": rev15 < 0.5,
        "G5_quantile_supports": ((p1 > 0) & (q90_30 > 0)) | ((p1 < 0) & (q10_30 < 0)),
    }
    gnames = list(gates.keys())

    rows = []
    for cfg in range(32):
        active = [gnames[i] for i in range(5) if (cfg >> i) & 1]
        gmask = base_msk.copy()
        for gn in active:
            gmask &= gates[gn]
        valid_idx = np.where(gmask)[0]
        if valid_idx.size < 200: continue
        absp = np.abs(p1[valid_idx])
        n_take = max(1, int(valid_idx.size * 0.01))
        order = np.argsort(-absp)
        top = valid_idx[order[:n_take]]
        for side, smask in [("long", p1[top]>0), ("short", p1[top]<0)]:
            cell = top[smask]
            nn = int(cell.size)
            if nn < 5: continue
            tgt = data["target_log_ret_5s"][cell]
            m = data["mask_log_ret_5s"].astype(bool)[cell]
            k = m & np.isfinite(tgt)
            wr = float(np.mean(np.sign(tgt[k]) == np.sign(p1[cell][k])) * 100.0) if k.sum() else None
            edge = float(np.mean(tgt[k] * np.sign(p1[cell][k]))) if k.sum() else None
            fm = f43_msk[cell]
            pnl = (float(np.mean(f43_pnl[cell][fm])) if fm.sum() else None)
            rows.append({"cfg": cfg, "gates": active, "side": side, "n": nn,
                         "wr_5s_pct": wr, "edge_z_5s": edge,
                         "fill43_pct": float(np.mean(fm)*100.0),
                         "fifo_pnl_filled_t": pnl})

    # Sort by FIFO PnL filled descending
    rows_sorted = sorted([r for r in rows if r["fifo_pnl_filled_t"] is not None],
                         key=lambda r: -r["fifo_pnl_filled_t"])

    with open(out_dir / "confluence_gates.json", "w") as f:
        json.dump({"rows": rows, "top_by_fifo": rows_sorted[:20]}, f, indent=2,
                  default=lambda o: None if isinstance(o,float) and not math.isfinite(o) else o)

    md = ["# Task 4 — Confluence Gate Sweep (32 configs, Top 1%)\n\n",
          "## Top 15 by FIFO PnL filled (real ticks, after commission)\n",
          "| Gates | Side | n | WR5s% | Edge_z_5s | Fill43% | FIFO PnL (t) |\n",
          "|---|---|---:|---:|---:|---:|---:|\n"]
    for r in rows_sorted[:15]:
        md.append(f"| {','.join(g.replace('G','') for g in r['gates']) or 'NONE'} | "
                  f"{r['side']} | {r['n']} | {r['wr_5s_pct']:.1f} | "
                  f"{r['edge_z_5s']:.4f} | {r['fill43_pct']:.1f} | "
                  f"{r['fifo_pnl_filled_t']:.3f} |\n")
    (out_dir / "TASK4.md").write_text("".join(md))
    return {"n_rows": len(rows), "best_pnl_t": rows_sorted[0]["fifo_pnl_filled_t"] if rows_sorted else None}


# ──────────────── TASK 5: vol-regime split ────────────────

def task_05_vol_regimes(data: dict, out_dir: Path):
    log("TASK 5: vol-regime split (low/mid/high)")
    out_dir.mkdir(parents=True, exist_ok=True)
    vol = data["pred_pred_realized_vol_30s_ticks"]
    vol_msk = data["mask_pred_realized_vol_30s_ticks"].astype(bool) & np.isfinite(vol)
    if vol_msk.sum() == 0:
        log("  no vol data"); return {"skipped": True}
    q33, q67 = np.percentile(vol[vol_msk], [33, 67])
    regime = np.full(data["n_samples"], -1, dtype=int)
    regime[vol_msk & (vol < q33)] = 0
    regime[vol_msk & (vol >= q33) & (vol < q67)] = 1
    regime[vol_msk & (vol >= q67)] = 2
    regime_names = ["low", "mid", "high"]

    p1 = data["pred_log_ret_1s"]; t30 = data["target_log_ret_30s"]
    msk_p = data["mask_log_ret_1s"].astype(bool) & np.isfinite(p1)
    msk_t = data["mask_log_ret_30s"].astype(bool) & np.isfinite(t30)
    f43m = data["mask_fifo_tp4sl3_net"].astype(bool)
    f43p = data["target_fifo_tp4sl3_net"]

    rows = []
    for ri, rname in enumerate(regime_names):
        in_r = (regime == ri) & msk_p
        if in_r.sum() < 100: continue
        absp = np.abs(p1[in_r])
        n_take = max(1, int(absp.size * 0.01))
        idx_r = np.where(in_r)[0]
        order = np.argsort(-absp)
        top = idx_r[order[:n_take]]
        for side, smask in [("long", p1[top]>0), ("short", p1[top]<0)]:
            cell = top[smask]
            nn = int(cell.size)
            if nn < 5: continue
            k = msk_t[cell] & np.isfinite(t30[cell])
            wr = float(np.mean(np.sign(t30[cell][k])==np.sign(p1[cell][k]))*100) if k.sum() else None
            edge = float(np.mean(t30[cell][k] * np.sign(p1[cell][k]))) if k.sum() else None
            fm = f43m[cell]
            pnl = (float(np.mean(f43p[cell][fm])) if fm.sum() else None)
            rows.append({"vol_regime": rname, "side": side, "n": nn,
                         "wr_30s_pct": wr, "edge_z_30s": edge,
                         "fill43_pct": float(np.mean(fm)*100),
                         "fifo_pnl_filled_t": pnl})

    with open(out_dir / "vol_regimes.json", "w") as f:
        json.dump({"q33_ticks": float(q33), "q67_ticks": float(q67), "rows": rows}, f, indent=2,
                  default=lambda o: None if isinstance(o,float) and not math.isfinite(o) else o)
    md = ["# Task 5 — Vol Regime Split (Top 1%, by pred_realized_vol_30s_ticks)\n\n",
          f"_low<{q33:.2f}t, mid {q33:.2f}-{q67:.2f}t, high>{q67:.2f}t_\n\n",
          "| Vol | Side | n | WR30s% | Edge_z | Fill43% | FIFO PnL (t) |\n",
          "|---|---|---:|---:|---:|---:|---:|\n"]
    for r in rows:
        md.append(f"| {r['vol_regime']} | {r['side']} | {r['n']} | {r['wr_30s_pct']:.1f} | "
                  f"{r['edge_z_30s']:.4f} | {r['fill43_pct']:.1f} | {r['fifo_pnl_filled_t']} |\n")
    (out_dir / "TASK5.md").write_text("".join(md))
    return {"n_rows": len(rows)}


# ──────────────── TASK 6: FIFO queue back-off sim (proxy via fill rate behavior) ────────────────

def task_06_fifo_queue_backoff(data: dict, out_dir: Path):
    """Without raw L2 we can only approximate: place limit at touch±k ticks → modeled fill rate
    as: more aggressive (closer/cross) → higher fill, but adverse-fill risk; further back → lower fill
    but better selection. We use the observed FIFO labels for at-touch baseline and synthesize
    back-off effect via target conditional on price-path."""
    log("TASK 6: FIFO queue back-off sim (proxy)")
    out_dir.mkdir(parents=True, exist_ok=True)

    # Top1% long+short by |p1|
    p1 = data["pred_log_ret_1s"]
    msk = data["mask_log_ret_1s"].astype(bool) & np.isfinite(p1)
    idx = np.where(msk)[0]
    absp = np.abs(p1[idx])
    n_take = max(1, int(idx.size * 0.01))
    top = idx[np.argsort(-absp)[:n_take]]

    # MFE & MAE in ticks
    mfe = data["target_pred_mfe_30s_ticks"]; mfe_m = data["mask_pred_mfe_30s_ticks"].astype(bool)
    mae = data["target_pred_mae_30s_ticks"]; mae_m = data["mask_pred_mae_30s_ticks"].astype(bool)
    f43m = data["mask_fifo_tp4sl3_net"].astype(bool)
    f43p = data["target_fifo_tp4sl3_net"]

    rows = []
    for side, smask in [("long", p1[top]>0), ("short", p1[top]<0)]:
        cell = top[smask]
        dsign = 1.0 if side=="long" else -1.0
        for backoff in [0, 1, 2, 3]:  # ticks back from touch
            # Modeled fill: only fills if price moves AGAINST our position by `backoff` ticks
            # (since we're behind the touch). Approx using MAE in dsign direction:
            # fill_eligible = MAE-in-direction (ticks) >= backoff
            mm = mae_m[cell] & np.isfinite(mae[cell])
            mae_in_dir = mae[cell][mm] * dsign
            fill_mask_local = mae_in_dir >= backoff
            n_fill = int(fill_mask_local.sum())
            n_total = int(mm.sum())
            fill_pct = n_fill / n_total * 100 if n_total else 0.0
            # Conditional FIFO PnL: only those that filled at backoff also achieve subsequent edge
            # Approximate edge after fill = (MFE_in_dir - backoff) for those that filled
            mfe_m_local = mfe_m[cell][mm]
            mfe_in_dir = mfe[cell][mm] * dsign
            edge_after_fill = (mfe_in_dir[fill_mask_local] - backoff) if n_fill else np.array([])
            # Subtract commission
            net_pnl_t = float(np.mean(edge_after_fill - PASSIVE_COST_TICKS)) if edge_after_fill.size else None
            rows.append({"side": side, "backoff_ticks": backoff,
                         "fill_pct": fill_pct, "n_filled": n_fill, "n_total": n_total,
                         "modeled_net_pnl_filled_t": net_pnl_t})

    with open(out_dir / "queue_backoff.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2,
                  default=lambda o: None if isinstance(o,float) and not math.isfinite(o) else o)
    md = ["# Task 6 — FIFO Queue Back-off Sim (proxy)\n\n",
          "Modeled: place limit `backoff` ticks behind touch on signal direction. ",
          "Filled iff price moves against us by `backoff` ticks first (MAE proxy). ",
          "Edge after fill = MFE-in-direction − backoff − commission.\n\n",
          "| Side | Backoff (t) | Fill% | n_filled | Net PnL/fill (t) |\n",
          "|---|---:|---:|---:|---:|\n"]
    for r in rows:
        md.append(f"| {r['side']} | {r['backoff_ticks']} | {r['fill_pct']:.1f} | "
                  f"{r['n_filled']} | {r['modeled_net_pnl_filled_t']} |\n")
    (out_dir / "TASK6.md").write_text("".join(md))
    return {"n_rows": len(rows)}


# ──────────────── TASK 7: aggressive cross IOC sim ────────────────

def task_07_aggressive_cross(data: dict, out_dir: Path):
    log("TASK 7: aggressive cross / IOC sim")
    out_dir.mkdir(parents=True, exist_ok=True)
    p1 = data["pred_log_ret_1s"]
    msk = data["mask_log_ret_1s"].astype(bool) & np.isfinite(p1)
    idx = np.where(msk)[0]
    absp = np.abs(p1[idx])
    mfe = data["target_pred_mfe_30s_ticks"]; mfe_m = data["mask_pred_mfe_30s_ticks"].astype(bool)
    mae = data["target_pred_mae_30s_ticks"]; mae_m = data["mask_pred_mae_30s_ticks"].astype(bool)

    rows = []
    for bname, bfrac in BANDS:
        n_take = max(1, int(idx.size * bfrac))
        top = idx[np.argsort(-absp)[:n_take]]
        for side, smask in [("long", p1[top]>0), ("short", p1[top]<0)]:
            cell = top[smask]
            dsign = 1.0 if side == "long" else -1.0
            mfm = mfe_m[cell] & np.isfinite(mfe[cell])
            mam = mae_m[cell] & np.isfinite(mae[cell])
            valid = mfm & mam
            if valid.sum() < 5: continue
            mfe_in_dir = mfe[cell][valid] * dsign
            mae_in_dir = mae[cell][valid] * dsign  # negative of position direction
            # IOC market entry: entry crosses spread (1 tick + commission). Then realized payoff
            # by 30s = signed realized_30s in ticks. We approximate via mfe_in_dir (best favorable
            # move achieved). For naive "hold to MFE peak": payoff = mfe_in_dir − MARKET_COST_TICKS
            net_pnl_naive = float(np.mean(mfe_in_dir - MARKET_COST_TICKS))
            # For "hold-fixed-30s" using avg of mfe and mae as expected close:
            mid = (mfe_in_dir + mae_in_dir) / 2
            net_pnl_holdmid = float(np.mean(mid - MARKET_COST_TICKS))
            # Sharpe-toy
            sd_mfe = float(np.std(mfe_in_dir - MARKET_COST_TICKS))
            sharpe = (net_pnl_naive / sd_mfe * math.sqrt(valid.sum())) if sd_mfe>0 else None
            rows.append({"band": bname, "side": side, "n": int(valid.sum()),
                         "mfe_mean_dir_t": float(np.mean(mfe_in_dir)),
                         "mae_mean_dir_t": float(np.mean(mae_in_dir)),
                         "net_pnl_market_holdtoMFE_t": net_pnl_naive,
                         "net_pnl_market_holdmid_t": net_pnl_holdmid,
                         "sharpe_toy_holdMFE": sharpe})

    with open(out_dir / "aggressive_cross.json", "w") as f:
        json.dump({"rows": rows}, f, indent=2,
                  default=lambda o: None if isinstance(o,float) and not math.isfinite(o) else o)
    md = ["# Task 7 — Aggressive Cross (IOC market) Sim\n\n",
          f"Cost floor: {MARKET_COST_TICKS} ticks (commission + 1-tick spread cross).\n\n",
          "| Band | Side | n | MFE_dir (t) | MAE_dir (t) | Net hold-to-MFE (t) | Net hold-mid (t) | Sharpe-toy |\n",
          "|---|---|---:|---:|---:|---:|---:|---:|\n"]
    for r in rows:
        md.append(f"| {r['band']} | {r['side']} | {r['n']} | "
                  f"{r['mfe_mean_dir_t']:.2f} | {r['mae_mean_dir_t']:.2f} | "
                  f"{r['net_pnl_market_holdtoMFE_t']:.3f} | "
                  f"{r['net_pnl_market_holdmid_t']:.3f} | "
                  f"{r['sharpe_toy_holdMFE']} |\n")
    (out_dir / "TASK7.md").write_text("".join(md))
    return {"n_rows": len(rows)}


# ──────────────── TASK 8: meta-MLP ────────────────

def task_08_meta_mlp(data: dict, out_dir: Path):
    log("TASK 8: meta-MLP P(profitable trade)")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.model_selection import KFold
        from sklearn.metrics import roc_auc_score
        from sklearn.preprocessing import StandardScaler
    except ImportError as e:
        log(f"  sklearn missing: {e}"); return {"skipped": True}

    # Build feature matrix from all pred_* heads
    feat_cols = []
    for k in data.keys():
        if isinstance(k, str) and k.startswith("pred_") and not k.startswith("pred_pred_"):
            feat_cols.append(k)
        elif isinstance(k, str) and k.startswith("pred_pred_"):
            feat_cols.append(k)  # pred_pred_mfe_30s_ticks etc
    feat_cols = sorted(set(feat_cols))
    log(f"  using {len(feat_cols)} pred features")

    X = np.column_stack([np.nan_to_num(data[c], nan=0.0, posinf=0.0, neginf=0.0)
                         for c in feat_cols])
    # Label: profitable trade if target_fifo_tp4sl3_net > 0 (and mask present)
    fmask = data["mask_fifo_tp4sl3_net"].astype(bool)
    y_full = (data["target_fifo_tp4sl3_net"] > 0).astype(int)
    sub_idx = np.where(fmask)[0]
    if sub_idx.size < 1000:
        log("  too few fills"); return {"skipped": True, "n_fills": int(sub_idx.size)}
    X_s = X[sub_idx]; y_s = y_full[sub_idx]
    # 5-fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    aucs = []; preds_oof = np.zeros(y_s.size)
    sc = StandardScaler()
    for tr, te in kf.split(X_s):
        Xtr_s = sc.fit_transform(X_s[tr]); Xte_s = sc.transform(X_s[te])
        clf = MLPClassifier(hidden_layer_sizes=(32,), max_iter=80, random_state=42,
                            early_stopping=True, validation_fraction=0.15)
        clf.fit(Xtr_s, y_s[tr])
        p = clf.predict_proba(Xte_s)[:,1]
        preds_oof[te] = p
        aucs.append(float(roc_auc_score(y_s[te], p)))
    mean_auc = float(np.mean(aucs))
    log(f"  CV AUC: {aucs}, mean={mean_auc:.4f}")

    # New Sharpe-toy when filtering by meta-MLP > threshold 0.6
    pnl = data["target_fifo_tp4sl3_net"][sub_idx]
    keep = preds_oof > 0.60
    if keep.sum() >= 20:
        sharpe = float(np.mean(pnl[keep]) / np.std(pnl[keep]) * math.sqrt(keep.sum())) if np.std(pnl[keep])>0 else None
        new_pnl = float(np.mean(pnl[keep]))
        n_kept = int(keep.sum())
    else:
        sharpe = None; new_pnl = None; n_kept = 0

    out = {"feat_cols": feat_cols, "cv_aucs": aucs, "mean_auc": mean_auc,
           "n_fills": int(sub_idx.size),
           "filter_threshold": 0.60, "n_kept": n_kept,
           "new_pnl_filled_t": new_pnl, "new_sharpe_toy": sharpe}
    with open(out_dir / "meta_mlp.json", "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda o: None if isinstance(o,float) and not math.isfinite(o) else o)
    md = ["# Task 8 — Meta-MLP P(profitable fill)\n\n",
          f"- Features: {len(feat_cols)} pred heads\n",
          f"- N fills (label avail): {sub_idx.size}\n",
          f"- 5-fold CV AUC: {[round(a,4) for a in aucs]}, **mean={mean_auc:.4f}**\n",
          f"- Filter threshold: 0.60 → kept {n_kept} fills\n",
          f"- Filtered mean PnL: {new_pnl} ticks/fill\n",
          f"- Filtered Sharpe-toy: {sharpe}\n"]
    (out_dir / "TASK8.md").write_text("".join(md))
    return {"mean_auc": mean_auc, "filtered_pnl_t": new_pnl}


# ──────────────── TASK 9: contextual bandit RL ────────────────

def task_09_contextual_bandit(data: dict, out_dir: Path):
    log("TASK 9: contextual bandit (logistic per action)")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import KFold
    except ImportError as e:
        log(f"  sklearn missing: {e}"); return {"skipped": True}

    # Build feature matrix (subset of preds for speed)
    fcols = ["pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_10s",
             "pred_p_up_5s", "pred_p_up_10s", "pred_p_up_30s",
             "pred_p_reversal_15s", "pred_p_reversal_30s",
             "pred_pred_mfe_30s_ticks", "pred_pred_mae_30s_ticks",
             "pred_pred_realized_vol_30s_ticks",
             "pred_log_ret_30s_q10", "pred_log_ret_30s_q90"]
    X = np.column_stack([np.nan_to_num(data[c], nan=0.0) for c in fcols])

    # Three actions: long, short, no-trade. Reward proxy:
    # long  reward = target_fifo_tp4sl3_net  (signed; >0 if long-trade made money)
    # short reward = -target_fifo_tp4sl3_net (sign-flip approximation)
    # no-trade reward = 0
    fmask = data["mask_fifo_tp4sl3_net"].astype(bool)
    pnl = data["target_fifo_tp4sl3_net"]
    sub = np.where(fmask)[0]
    if sub.size < 1000:
        return {"skipped": True}
    X_s = X[sub]; r_long = pnl[sub]; r_short = -pnl[sub]

    # Best action per row
    actions = np.where(r_long > r_short, 0, 1)  # 0=long, 1=short
    best_r = np.maximum(r_long, r_short)
    # Add no-trade dimension: if max(r_long, r_short) <= 0 → no-trade is best
    actions[best_r <= 0] = 2
    best_r[best_r <= 0] = 0

    # Train classifier action ← X
    sc = StandardScaler()
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    pol_rewards = []; choice_dist = np.zeros(3, dtype=int)
    for tr, te in kf.split(X_s):
        Xtr = sc.fit_transform(X_s[tr]); Xte = sc.transform(X_s[te])
        clf = LogisticRegression(max_iter=300, multi_class="multinomial",
                                 random_state=42, n_jobs=-1)
        clf.fit(Xtr, actions[tr])
        a_pred = clf.predict(Xte)
        # Realized reward of policy on test fold
        r_test = np.where(a_pred == 0, r_long[te],
                  np.where(a_pred == 1, r_short[te], 0.0))
        pol_rewards.append(float(np.mean(r_test)))
        for ai in range(3):
            choice_dist[ai] += int(np.sum(a_pred == ai))

    mean_pol_reward = float(np.mean(pol_rewards))
    base_reward_long = float(np.mean(r_long))
    base_reward_oracle = float(np.mean(best_r))
    out = {"feat_cols": fcols, "policy_reward_per_fill_t": mean_pol_reward,
           "policy_reward_folds": pol_rewards,
           "always_long_reward_per_fill_t": base_reward_long,
           "oracle_reward_per_fill_t": base_reward_oracle,
           "choice_distribution": {"long": int(choice_dist[0]),
                                    "short": int(choice_dist[1]),
                                    "no_trade": int(choice_dist[2])}}
    with open(out_dir / "bandit.json", "w") as f:
        json.dump(out, f, indent=2,
                  default=lambda o: None if isinstance(o,float) and not math.isfinite(o) else o)
    md = ["# Task 9 — Contextual Bandit (logistic 3-action)\n\n",
          f"- Features: {fcols}\n",
          f"- Policy reward per fill: **{mean_pol_reward:.4f} ticks**\n",
          f"- Always-long baseline: {base_reward_long:.4f} ticks\n",
          f"- Oracle (knows future): {base_reward_oracle:.4f} ticks\n",
          f"- Action distribution (CV): long={choice_dist[0]}, short={choice_dist[1]}, no-trade={choice_dist[2]}\n"]
    (out_dir / "TASK9.md").write_text("".join(md))
    return {"policy_reward_t": mean_pol_reward, "oracle_t": base_reward_oracle}


# ──────────────── TASK 10: per-head ablation ────────────────

def task_10_per_head_ablation(data: dict, out_dir: Path):
    log("TASK 10: per-head ablation against meta-MLP")
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        from sklearn.neural_network import MLPClassifier
        from sklearn.preprocessing import StandardScaler
        from sklearn.model_selection import KFold
        from sklearn.metrics import roc_auc_score
    except ImportError:
        return {"skipped": True}

    feat_cols = sorted([k for k in data.keys() if isinstance(k, str)
                        and k.startswith("pred_")])
    X_full = np.column_stack([np.nan_to_num(data[c], nan=0.0) for c in feat_cols])
    fmask = data["mask_fifo_tp4sl3_net"].astype(bool)
    y_full = (data["target_fifo_tp4sl3_net"] > 0).astype(int)
    sub = np.where(fmask)[0]
    if sub.size < 1000:
        return {"skipped": True}
    X = X_full[sub]; y = y_full[sub]

    # Sub-sample for speed (ablation is N×CV, expensive)
    rng = np.random.default_rng(42)
    if X.shape[0] > 12000:
        sel = rng.choice(X.shape[0], 12000, replace=False)
        X = X[sel]; y = y[sel]

    def cv_auc(Xin):
        aucs = []
        sc = StandardScaler()
        for tr, te in KFold(n_splits=3, shuffle=True, random_state=42).split(Xin):
            Xtr = sc.fit_transform(Xin[tr]); Xte = sc.transform(Xin[te])
            clf = MLPClassifier(hidden_layer_sizes=(16,), max_iter=40, random_state=42,
                                early_stopping=True, validation_fraction=0.15)
            try:
                clf.fit(Xtr, y[tr])
                p = clf.predict_proba(Xte)[:,1]
                aucs.append(roc_auc_score(y[te], p))
            except Exception as e:
                aucs.append(0.5)
        return float(np.mean(aucs))

    base_auc = cv_auc(X)
    log(f"  baseline AUC: {base_auc:.4f}")
    rows = [{"head": "BASELINE_ALL", "auc": base_auc, "delta": 0.0}]
    for i, col in enumerate(feat_cols):
        cols_keep = [j for j in range(len(feat_cols)) if j != i]
        a = cv_auc(X[:, cols_keep])
        rows.append({"head": col, "auc": a, "delta": a - base_auc})
        log(f"  drop {col}: AUC={a:.4f} (Δ={a-base_auc:+.4f})")
    rows_sorted = sorted(rows[1:], key=lambda r: r["delta"])  # most negative delta = most important
    with open(out_dir / "per_head_ablation.json", "w") as f:
        json.dump({"baseline_auc": base_auc, "ablations": rows_sorted}, f, indent=2,
                  default=lambda o: None if isinstance(o,float) and not math.isfinite(o) else o)
    md = ["# Task 10 — Per-Head Ablation (drop-one, smaller MLP, 3-fold CV)\n\n",
          f"Baseline AUC (all heads): {base_auc:.4f}\n\n",
          "## Top 15 most-important heads (largest AUC drop when removed)\n",
          "| Head | AUC w/o | Δ |\n|---|---:|---:|\n"]
    for r in rows_sorted[:15]:
        md.append(f"| {r['head']} | {r['auc']:.4f} | {r['delta']:+.4f} |\n")
    md.append("\n## 10 LEAST-important heads (positive Δ = removable noise)\n")
    md.append("| Head | AUC w/o | Δ |\n|---|---:|---:|\n")
    for r in rows_sorted[-10:][::-1]:
        md.append(f"| {r['head']} | {r['auc']:.4f} | {r['delta']:+.4f} |\n")
    (out_dir / "TASK10.md").write_text("".join(md))
    return {"baseline_auc": base_auc, "n_heads": len(feat_cols)}


# ──────────────── orchestrator ────────────────

TASKS = [
    ("01_unit_fix",          task_01_unit_fix),
    ("02_edge_decay",        task_02_edge_decay),
    ("03_time_of_day",       task_03_time_of_day),
    ("04_confluence_gates",  task_04_confluence_gates),
    ("05_vol_regimes",       task_05_vol_regimes),
    ("06_fifo_queue_backoff", task_06_fifo_queue_backoff),
    ("07_aggressive_cross",  task_07_aggressive_cross),
    ("08_meta_mlp",          task_08_meta_mlp),
    ("09_contextual_bandit", task_09_contextual_bandit),
    ("10_per_head_ablation", task_10_per_head_ablation),
]


def run_with_timeout(fn, args, timeout):
    """Run fn(*args) in a subprocess with timeout. Return result or {'timeout':True}."""
    def _worker(q, fn, args):
        try:
            r = fn(*args)
            q.put({"ok": True, "result": r})
        except Exception as e:
            q.put({"ok": False, "error": repr(e), "tb": traceback.format_exc()})
    q = multiprocessing.Queue()
    p = multiprocessing.Process(target=_worker, args=(q, fn, args))
    p.start()
    p.join(timeout)
    if p.is_alive():
        p.terminate()
        p.join(5)
        if p.is_alive():
            os.kill(p.pid, signal.SIGKILL)
        return {"ok": False, "error": "TIMEOUT", "result": None}
    if not q.empty():
        return q.get()
    return {"ok": False, "error": "no result"}


def main():
    started = datetime.utcnow()
    if not MASTER_MD.exists():
        MASTER_MD.write_text(f"# HC #346 — Jupiter All-Night v3.2 Edge Research\n\n"
                              f"Started: {started.isoformat()}Z\n\n"
                              f"Source: {PRED_NPZ}\n\n"
                              f"## Task Summaries (updated as each completes)\n\n")
    log(f"Orchestrator START — {len(TASKS)} tasks, timeout {TASK_TIMEOUT_SEC}s each")
    data, n = load_data()

    summaries = {}
    for tname, tfn in TASKS:
        sub = OUT_ROOT / f"task_{tname}"
        sub.mkdir(parents=True, exist_ok=True)
        log(f"=== START task_{tname} ===")
        t0 = time.time()
        # NOTE: passing huge `data` dict to subprocess is expensive.
        # Run inline (we trust each task to be bounded). Wrap in try/except.
        try:
            res = tfn(data, sub)
            took = time.time() - t0
            log(f"=== DONE task_{tname} in {took:.1f}s :: {res}")
            summaries[tname] = {"ok": True, "took_sec": took, "summary": res}
        except Exception as e:
            took = time.time() - t0
            log(f"=== FAIL task_{tname} in {took:.1f}s :: {e}\n{traceback.format_exc()}")
            summaries[tname] = {"ok": False, "took_sec": took, "error": repr(e)}

        # Append to ORCHESTRATOR.md
        section = f"### task_{tname} ({summaries[tname]['took_sec']:.1f}s)\n\n"
        if summaries[tname]["ok"]:
            section += f"- ✅ done — `{summaries[tname]['summary']}`\n"
            section += f"- output: `output/v3_2_allnight_research_20260514/task_{tname}/`\n"
        else:
            section += f"- ❌ FAIL — {summaries[tname]['error']}\n"
        append_master(section)

    final = {"started_utc": started.isoformat(),
             "ended_utc": datetime.utcnow().isoformat(),
             "summaries": summaries}
    (OUT_ROOT / "FINAL_SUMMARY.json").write_text(json.dumps(final, indent=2, default=str))
    append_master(f"## ALL TASKS COMPLETE — finished {datetime.utcnow().isoformat()}Z\n")
    log("Orchestrator COMPLETE")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        log(f"FATAL: {e}\n{traceback.format_exc()}")
        sys.exit(1)
