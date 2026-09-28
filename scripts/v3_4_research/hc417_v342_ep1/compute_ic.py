#!/usr/bin/env python3
"""HC #417 — Concat IC for v3.4.2 ep-1 NPZ across all available heads.

Vs v2 baseline (full-OOT 46 dates): IC_1s=0.236, IC_5s=0.127, IC_10s=0.090.
v3.4.2 is on 5 dates only (Feb 23-27); ICs are not strictly comparable due to
different date scope. Reported here for tracking.
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

NPZ = Path("/home/jupiter/Lvl3Quant/output/v342_ep1_eval/fold_00_ep1_oot_wrapped.npz")
OUT_CSV = Path("/home/jupiter/Lvl3Quant/output/v342_ep1_eval/ic_summary.csv")

# (pred_key, target_key, mask_key, label)
PAIRS = [
    ("pred_log_ret_1s", "target_log_ret_1s", "mask_log_ret_1s", "log_ret_1s"),
    ("pred_log_ret_5s", "target_log_ret_5s", "mask_log_ret_5s", "log_ret_5s"),
    ("pred_log_ret_10s", "target_log_ret_10s", "mask_log_ret_10s", "log_ret_10s"),
    ("pred_log_ret_30s", "target_log_ret_30s", "mask_log_ret_30s", "log_ret_30s"),
    ("pred_log_ret_60s", "target_log_ret_60s", "mask_log_ret_60s", "log_ret_60s"),
    ("pred_log_ret_5min", "target_log_ret_5min", "mask_log_ret_5min", "log_ret_5min"),
    ("pred_p_up_5s", "target_p_up_5s", "mask_p_up_5s", "p_up_5s"),
    ("pred_p_up_10s", "target_p_up_10s", "mask_p_up_10s", "p_up_10s"),
    ("pred_p_up_30s", "target_p_up_30s", "mask_p_up_30s", "p_up_30s"),
    ("pred_p_up_60s", "target_p_up_60s", "mask_p_up_60s", "p_up_60s"),
    ("pred_p_reversal_15s", "target_p_reversal_15s", "mask_p_reversal_15s", "p_reversal_15s"),
    ("pred_p_reversal_30s", "target_p_reversal_30s", "mask_p_reversal_30s", "p_reversal_30s"),
    ("pred_p_reversal_60s", "target_p_reversal_60s", "mask_p_reversal_60s", "p_reversal_60s"),
    ("pred_pred_mfe_30s_ticks", "target_pred_mfe_30s_ticks", "mask_pred_mfe_30s_ticks", "mfe_30s_ticks"),
    ("pred_pred_mae_30s_ticks", "target_pred_mae_30s_ticks", "mask_pred_mae_30s_ticks", "mae_30s_ticks"),
    ("pred_pred_mfe_60s_ticks", "target_pred_mfe_60s_ticks", "mask_pred_mfe_60s_ticks", "mfe_60s_ticks"),
    ("pred_pred_mae_60s_ticks", "target_pred_mae_60s_ticks", "mask_pred_mae_60s_ticks", "mae_60s_ticks"),
    ("pred_pred_time_to_mfe_secs", "target_pred_time_to_mfe_secs", "mask_pred_time_to_mfe_secs", "time_to_mfe_secs"),
    ("pred_pred_realized_vol_30s_ticks", "target_pred_realized_vol_30s_ticks", "mask_pred_realized_vol_30s_ticks", "realized_vol_30s"),
    ("pred_fifo_tp4sl3_net", "target_fifo_tp4sl3_net", "mask_fifo_tp4sl3_net", "fifo_tp4sl3_net"),
    ("pred_fifo_tp4sl3_hit_tp", "target_fifo_tp4sl3_hit_tp", "mask_fifo_tp4sl3_hit_tp", "fifo_tp4sl3_hit_tp"),
    ("pred_fifo_tp8sl5_net", "target_fifo_tp8sl5_net", "mask_fifo_tp8sl5_net", "fifo_tp8sl5_net"),
    ("pred_fifo_tp8sl5_hit_tp", "target_fifo_tp8sl5_hit_tp", "mask_fifo_tp8sl5_hit_tp", "fifo_tp8sl5_hit_tp"),
]

V2_BASELINE = {
    "log_ret_1s": 0.236,
    "log_ret_5s": 0.127,
    "log_ret_10s": 0.090,
}


def pearson_safe(a: np.ndarray, b: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.size < 5:
        return float("nan")
    if a.std() == 0 or b.std() == 0:
        return float("nan")
    return float(np.corrcoef(a, b)[0, 1])


def main() -> int:
    d = np.load(NPZ, allow_pickle=True)
    n = int(d["n_samples"])
    print(f"[ic] NPZ n={n}, dates={list(d['oot_dates'])}")

    rows = []
    for pk, tk, mk, label in PAIRS:
        if pk not in d.files or tk not in d.files:
            continue
        pred = np.asarray(d[pk][:n], dtype=np.float64)
        target = np.asarray(d[tk][:n], dtype=np.float64)
        mask_raw = np.asarray(d[mk][:n], dtype=np.float64) if mk in d.files else np.ones(n)
        mask = (mask_raw > 0) & np.isfinite(pred) & np.isfinite(target)
        # Filter outliers in target (sentinel values like -100, +50)
        # but keep them in for fair concat IC unless they're clearly sentinels.
        # For log_ret_*: filter |target| <= 50 ticks (sentinels are -100, +46)
        if label.startswith("log_ret_"):
            mask &= np.abs(target) < 50.0
        n_valid = int(mask.sum())
        if n_valid < 10:
            ic = float("nan")
        else:
            ic = pearson_safe(pred[mask], target[mask])
        delta = ""
        if label in V2_BASELINE:
            delta = f"{ic - V2_BASELINE[label]:+.4f}"
        rows.append({
            "head": label,
            "n_valid": n_valid,
            "concat_ic": round(ic, 4) if np.isfinite(ic) else float("nan"),
            "v2_baseline_ic": V2_BASELINE.get(label, ""),
            "delta_vs_v2": delta,
        })
        print(f"[ic] {label:24s} n={n_valid:>8} ic={ic:+.4f} delta_v2={delta}")

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["head", "n_valid", "concat_ic",
                                           "v2_baseline_ic", "delta_vs_v2"])
        w.writeheader()
        for r in rows:
            w.writerow(r)
    print(f"[ic] wrote {OUT_CSV}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
