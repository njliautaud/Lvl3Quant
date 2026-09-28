#!/usr/bin/env python3
"""
HC #453 R6a — Head Smoothness Diagnostic.

CNN-Mamba v3.4.2 is already a multi-head model. We have been trading on
pred_log_ret_1s (the snapshot regression head) which HC #450 R4 measured at
lag-1 autocorr ~= 0.02 = flicker. The other heads (pred_p_up_*,
pred_pred_mfe_*, pred_p_reversal_*, pred_fifo_tp4sl3_net, pred_fifo_tp8sl5_net)
already exist as trained outputs. Some of them may be naturally smoother by
construction (longer-horizon targets, classification heads with sigmoid output).

This script measures, per-head per-day, the lag-1..lag-40 autocorrelation and
sign-flip rate so we can identify the SMOOTHEST existing head as the candidate
new trading signal -- without any retrain. Output is one CSV summarizing all
heads x dates and one ranked summary picking the top-3 smoothest heads.

Input:  /home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate/oot_YYYYMMDD.npz
Output: /home/jupiter/Lvl3Quant/output/hc453_head_smoothness/
            per_head_per_day.csv
            head_ranking.csv
            SUMMARY.md
"""

import csv
import os
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


INPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/hc453_head_smoothness")
LAGS = [1, 2, 4, 8, 20, 40, 120]  # 250ms strides => 0.25s, 0.5s, 1s, 2s, 5s, 10s, 30s
WORKERS = 8

HEADS_TO_TEST = [
    "pred_log_ret_1s",
    "pred_log_ret_5s",
    "pred_log_ret_10s",
    "pred_log_ret_30s",
    "pred_log_ret_60s",
    "pred_log_ret_5min",
    "pred_p_up_5s",
    "pred_p_up_10s",
    "pred_p_up_30s",
    "pred_p_up_60s",
    "pred_log_ret_10s_q50",
    "pred_log_ret_30s_q50",
    "pred_log_ret_60s_q50",
    "pred_pred_mfe_30s_ticks",
    "pred_pred_mae_30s_ticks",
    "pred_pred_mfe_60s_ticks",
    "pred_pred_mae_60s_ticks",
    "pred_p_reversal_15s",
    "pred_p_reversal_30s",
    "pred_p_reversal_60s",
    "pred_pred_realized_vol_30s_ticks",
    "pred_fifo_tp4sl3_net",
    "pred_fifo_tp8sl5_net",
]


def autocorr_at_lag(x: np.ndarray, lag: int) -> float:
    """Pearson autocorr at fixed lag, ignoring NaN."""
    if lag <= 0 or x.size <= lag + 1:
        return float("nan")
    a = x[:-lag]
    b = x[lag:]
    m = ~(np.isnan(a) | np.isnan(b))
    if m.sum() < 100:
        return float("nan")
    a = a[m]
    b = b[m]
    a = a - a.mean()
    b = b - b.mean()
    denom = np.sqrt((a * a).sum() * (b * b).sum())
    if denom <= 0:
        return float("nan")
    return float((a * b).sum() / denom)


def sign_flip_rate_per_min(x: np.ndarray, stride_hz: float = 4.0) -> float:
    """Fraction of consecutive ticks where sign flips, scaled to per-minute count.
    stride_hz=4 means 4 samples per second (stride 250ms)."""
    if x.size < 2:
        return float("nan")
    valid = ~np.isnan(x)
    x = x[valid]
    if x.size < 2:
        return float("nan")
    s = np.sign(x - np.median(x))  # demean to make zero-crossings meaningful for one-sided heads
    flips = (s[1:] != s[:-1]).sum()
    flips_per_sample = flips / max(1, x.size - 1)
    samples_per_min = stride_hz * 60.0
    return float(flips_per_sample * samples_per_min)


def top_decile_persistence(x: np.ndarray, lookahead: int = 4) -> float:
    """Given the signal is in top decile at t, what fraction of next `lookahead` ticks
    is it still in top decile? Tests 'high-confidence persistence'."""
    if x.size < lookahead + 100:
        return float("nan")
    valid = ~np.isnan(x)
    xv = x[valid]
    if xv.size < lookahead + 100:
        return float("nan")
    threshold = np.quantile(xv, 0.9)
    in_top = (x >= threshold).astype(np.int8)
    # at every t where in_top=1, check fraction of t+1..t+lookahead that are also in_top
    idx = np.where(in_top[:-lookahead] == 1)[0]
    if idx.size == 0:
        return float("nan")
    hits = 0
    total = 0
    for k in range(1, lookahead + 1):
        hits += in_top[idx + k].sum()
        total += idx.size
    return float(hits / total)


def process_one_file(npz_path: Path) -> list[dict]:
    """Compute diagnostic rows for one OOT NPZ. Returns list of dicts (one per head)."""
    try:
        z = np.load(npz_path, allow_pickle=False)
    except Exception as e:
        return [{"date": npz_path.stem, "head": "_ERROR_", "error": str(e)}]
    date_str = npz_path.stem.replace("oot_", "")
    rows = []
    for head in HEADS_TO_TEST:
        if head not in z.files:
            continue
        x = z[head].astype(np.float64)
        row = {"date": date_str, "head": head, "n": int(x.size)}
        for lag in LAGS:
            row[f"ac_lag{lag}"] = autocorr_at_lag(x, lag)
        row["flip_rate_per_min"] = sign_flip_rate_per_min(x)
        row["top10pct_persist_1s"] = top_decile_persistence(x, lookahead=4)
        row["top10pct_persist_10s"] = top_decile_persistence(x, lookahead=40)
        row["mean"] = float(np.nanmean(x))
        row["std"] = float(np.nanstd(x))
        rows.append(row)
    return rows


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob("oot_*.npz"))
    if not files:
        print(f"NO INPUT FILES in {INPUT_DIR}", file=sys.stderr)
        sys.exit(1)
    print(f"[hc453] processing {len(files)} OOT NPZ files with {WORKERS} workers")
    all_rows = []
    with ProcessPoolExecutor(max_workers=WORKERS) as ex:
        futs = {ex.submit(process_one_file, p): p for p in files}
        for i, fut in enumerate(as_completed(futs)):
            try:
                rows = fut.result()
                all_rows.extend(rows)
                if (i + 1) % 5 == 0 or (i + 1) == len(futs):
                    print(f"[hc453] done {i+1}/{len(futs)} files, rows so far={len(all_rows)}")
            except Exception as e:
                print(f"[hc453] worker failed for {futs[fut]}: {e}", file=sys.stderr)

    # Write per_head_per_day.csv
    if not all_rows:
        print("[hc453] no rows produced — abort", file=sys.stderr)
        sys.exit(2)
    cols = sorted({k for r in all_rows for k in r.keys()}, key=lambda c: (
        0 if c == "date" else 1 if c == "head" else 2 if c == "n" else 3 if c.startswith("ac_") else 4
    ))
    out_csv = OUTPUT_DIR / "per_head_per_day.csv"
    with out_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in all_rows:
            w.writerow(r)
    print(f"[hc453] wrote {out_csv} ({len(all_rows)} rows)")

    # Rank heads by mean lag-1 autocorr across all OOT days
    by_head: dict[str, list[float]] = {}
    flip_by_head: dict[str, list[float]] = {}
    pers_by_head: dict[str, list[float]] = {}
    for r in all_rows:
        if r.get("head") == "_ERROR_":
            continue
        h = r["head"]
        v = r.get("ac_lag1")
        if v is not None and not (isinstance(v, float) and np.isnan(v)):
            by_head.setdefault(h, []).append(v)
        fv = r.get("flip_rate_per_min")
        if fv is not None and not (isinstance(fv, float) and np.isnan(fv)):
            flip_by_head.setdefault(h, []).append(fv)
        pv = r.get("top10pct_persist_10s")
        if pv is not None and not (isinstance(pv, float) and np.isnan(pv)):
            pers_by_head.setdefault(h, []).append(pv)
    ranking = []
    for h in sorted(by_head.keys()):
        ranking.append({
            "head": h,
            "n_days": len(by_head[h]),
            "mean_ac_lag1": float(np.mean(by_head[h])),
            "mean_flip_per_min": float(np.mean(flip_by_head.get(h, [float("nan")]))),
            "mean_top10_persist_10s": float(np.mean(pers_by_head.get(h, [float("nan")]))),
        })
    ranking.sort(key=lambda r: r["mean_ac_lag1"], reverse=True)
    rank_csv = OUTPUT_DIR / "head_ranking.csv"
    with rank_csv.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["head", "n_days", "mean_ac_lag1", "mean_flip_per_min", "mean_top10_persist_10s"])
        w.writeheader()
        for r in ranking:
            w.writerow(r)
    print(f"[hc453] wrote {rank_csv}")

    # SUMMARY.md
    md = ["# HC #453 R6a — Head Smoothness Diagnostic", "", f"Source: {INPUT_DIR} ({len(files)} OOT NPZs)", "", "## Top 5 smoothest heads by mean lag-1 autocorr", "", "| head | n_days | mean ac_lag1 | flip/min | top10pct persist 10s |", "|---|---|---|---|---|"]
    for r in ranking[:5]:
        md.append(f"| {r['head']} | {r['n_days']} | {r['mean_ac_lag1']:.4f} | {r['mean_flip_per_min']:.2f} | {r['mean_top10_persist_10s']:.3f} |")
    md.append("")
    md.append("## Bottom 5 (most flickery)")
    md.append("")
    md.append("| head | n_days | mean ac_lag1 | flip/min | top10pct persist 10s |")
    md.append("|---|---|---|---|---|")
    for r in ranking[-5:]:
        md.append(f"| {r['head']} | {r['n_days']} | {r['mean_ac_lag1']:.4f} | {r['mean_flip_per_min']:.2f} | {r['mean_top10_persist_10s']:.3f} |")
    md.append("")
    md.append("## HC #453 R4 gate: lag-1 autocorr >= 0.30 required")
    md.append("")
    above = [r for r in ranking if r["mean_ac_lag1"] >= 0.30]
    if above:
        md.append("**Heads passing R4 gate (candidates for new trading signal)**:")
        for r in above:
            md.append(f"- {r['head']} (ac_lag1={r['mean_ac_lag1']:.4f})")
    else:
        md.append("**NO head passes R4 gate.** Confirms HC #453 R3 fallback: retrain with smoothness loss regularizer + autoregressive feedback required.")
    md.append("")
    (OUTPUT_DIR / "SUMMARY.md").write_text("\n".join(md))
    print(f"[hc453] wrote SUMMARY.md")
    print("[hc453] DONE.")


if __name__ == "__main__":
    main()
