"""
v3.2 stratification audit — where does the 1s edge actually live?

Splits the v3.2 OOT predictions.npz across three orthogonal axes:
  1. PER-DAY (5 OOT days, equal-size proxy)
  2. INTRA-DAY position (sample-index within day quartiles — early/mid/late session)
  3. VOLATILITY REGIME (target_log_ret_5s abs-value quartiles)
  4. SIDE (predicted long vs predicted short)

For each cell, computes IC and DA on log_ret_1s, log_ret_5s, log_ret_10s, log_ret_30s.

This tells us:
  - Is the edge concentrated in one OOT day (overfit risk)?
  - Does the edge weaken at session open/close (time-of-day bias)?
  - Is the edge only in high-vol regimes (vol-conditional)?
  - Is the long-DA win at top1% a real asymmetry or a sample-size artifact?

Output: output/v3_2_deep_sim_20260512/stratification_audit.csv
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PREDS_PATH = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OUT_CSV = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/stratification_audit.csv")
OUT_JSON = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/stratification_audit.json")

HORIZONS = ["log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s"]


def ic_and_da(pred: np.ndarray, target: np.ndarray, mask: np.ndarray) -> tuple[float, float, int]:
    m = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)
    if m.sum() < 50:
        return float("nan"), float("nan"), int(m.sum())
    p = pred[m]
    t = target[m]
    try:
        ic, _ = spearmanr(p, t)
    except Exception:
        ic = float("nan")
    da = float(((p > 0) == (t > 0)).mean())
    return float(ic), float(da), int(m.sum())


def main() -> None:
    d = np.load(PREDS_PATH, allow_pickle=True)
    oot_dates = d["oot_dates"].tolist()
    n_samples = int(d["n_samples"])
    print(f"Loaded {n_samples:,} samples over {len(oot_dates)} OOT days: {oot_dates}")

    rows = []
    summary = {}

    # === Axis 1: PER-DAY (equal-size proxy) ===
    n_days = len(oot_dates)
    day_edges = np.linspace(0, n_samples, n_days + 1, dtype=int)
    for di in range(n_days):
        lo, hi = day_edges[di], day_edges[di + 1]
        for h in HORIZONS:
            ic, da, n = ic_and_da(d[f"pred_{h}"][lo:hi], d[f"target_{h}"][lo:hi], d[f"mask_{h}"][lo:hi])
            rows.append({"axis": "per_day", "bucket": oot_dates[di], "horizon": h, "n": n, "IC": ic, "DA": da})

    # === Axis 2: INTRA-DAY position (assumed roughly uniform; quartile within day) ===
    for q in range(4):
        for h in HORIZONS:
            mask_q = np.zeros(n_samples, dtype=bool)
            for di in range(n_days):
                lo, hi = day_edges[di], day_edges[di + 1]
                inner = np.linspace(lo, hi, 5, dtype=int)
                mask_q[inner[q]:inner[q + 1]] = True
            ic, da, n = ic_and_da(
                d[f"pred_{h}"][mask_q],
                d[f"target_{h}"][mask_q],
                d[f"mask_{h}"][mask_q].astype(bool),
            )
            rows.append({"axis": "intraday_q", "bucket": f"Q{q+1}_of_day", "horizon": h, "n": n, "IC": ic, "DA": da})

    # === Axis 3: VOLATILITY REGIME (target_log_ret_5s abs quartile) ===
    target_5s = d["target_log_ret_5s"]
    abs_5s = np.abs(target_5s)
    finite = np.isfinite(abs_5s)
    quartiles = np.quantile(abs_5s[finite], [0.25, 0.5, 0.75])
    vol_buckets = {
        "vol_low_q1": abs_5s <= quartiles[0],
        "vol_mid_q2": (abs_5s > quartiles[0]) & (abs_5s <= quartiles[1]),
        "vol_mid_q3": (abs_5s > quartiles[1]) & (abs_5s <= quartiles[2]),
        "vol_high_q4": abs_5s > quartiles[2],
    }
    for name, bmask in vol_buckets.items():
        for h in HORIZONS:
            ic, da, n = ic_and_da(
                d[f"pred_{h}"][bmask],
                d[f"target_{h}"][bmask],
                d[f"mask_{h}"][bmask].astype(bool),
            )
            rows.append({"axis": "vol_regime", "bucket": name, "horizon": h, "n": n, "IC": ic, "DA": da})

    # === Axis 4: PREDICTED SIDE (pred_1s > 0 vs < 0) ===
    pred_1s = d["pred_log_ret_1s"]
    long_mask = pred_1s > 0
    short_mask = pred_1s < 0
    for side_name, side_mask in [("pred_long", long_mask), ("pred_short", short_mask)]:
        for h in HORIZONS:
            ic, da, n = ic_and_da(
                d[f"pred_{h}"][side_mask],
                d[f"target_{h}"][side_mask],
                d[f"mask_{h}"][side_mask].astype(bool),
            )
            rows.append({"axis": "pred_side", "bucket": side_name, "horizon": h, "n": n, "IC": ic, "DA": da})

    # === Axis 5: TOP-CONFIDENCE CONDITIONAL — does the top1% live in one day or spread? ===
    abs_pred_1s = np.abs(pred_1s)
    top1_thresh = np.quantile(abs_pred_1s[np.isfinite(abs_pred_1s)], 0.99)
    top1_mask = abs_pred_1s >= top1_thresh
    # Per-day breakdown of top1% signals
    for di in range(n_days):
        lo, hi = day_edges[di], day_edges[di + 1]
        day_top1 = np.zeros(n_samples, dtype=bool)
        day_top1[lo:hi] = top1_mask[lo:hi]
        for h in HORIZONS:
            ic, da, n = ic_and_da(
                d[f"pred_{h}"][day_top1],
                d[f"target_{h}"][day_top1],
                d[f"mask_{h}"][day_top1].astype(bool),
            )
            rows.append({"axis": "top1pct_per_day", "bucket": oot_dates[di], "horizon": h, "n": n, "IC": ic, "DA": da})

    # === Write CSV ===
    import csv
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["axis", "bucket", "horizon", "n", "IC", "DA"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    # === Build a summary for quick inspection ===
    summary["per_day_IC_1s"] = {
        oot_dates[di]: next(r["IC"] for r in rows if r["axis"] == "per_day" and r["bucket"] == oot_dates[di] and r["horizon"] == "log_ret_1s")
        for di in range(n_days)
    }
    summary["pred_long_vs_short_IC_1s"] = {
        "long": next(r["IC"] for r in rows if r["axis"] == "pred_side" and r["bucket"] == "pred_long" and r["horizon"] == "log_ret_1s"),
        "short": next(r["IC"] for r in rows if r["axis"] == "pred_side" and r["bucket"] == "pred_short" and r["horizon"] == "log_ret_1s"),
    }
    summary["vol_regime_IC_1s"] = {
        b: next(r["IC"] for r in rows if r["axis"] == "vol_regime" and r["bucket"] == b and r["horizon"] == "log_ret_1s")
        for b in ["vol_low_q1", "vol_mid_q2", "vol_mid_q3", "vol_high_q4"]
    }
    summary["intraday_IC_1s"] = {
        f"Q{q+1}": next(r["IC"] for r in rows if r["axis"] == "intraday_q" and r["bucket"] == f"Q{q+1}_of_day" and r["horizon"] == "log_ret_1s")
        for q in range(4)
    }

    with open(OUT_JSON, "w") as f:
        json.dump(summary, f, indent=2, default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x)

    print(f"Wrote {len(rows)} rows to {OUT_CSV}")
    print(f"Wrote summary to {OUT_JSON}")
    print("\n--- SUMMARY ---")
    print(json.dumps(summary, indent=2, default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x))


if __name__ == "__main__":
    main()
