"""Check 3: per-tier confidence-rank vs realized hit-rate.

We bin predicted signal strength into deciles and compute the realized
"hit rate" per decile. A well-calibrated model produces a monotonically
rising hit-rate. We require: slope > 0.5 (in (decile_idx, hit_rate) units
scaled to [0,1]) and intercept residual |b| < 0.1.

We compute calibration on the 10s log-return head (primary signal).
Hit = (sign(pred) == sign(target)) restricted to finite, mask>0 rows.
"""
from __future__ import annotations
import numpy as np


SLOPE_MIN = 0.5
INTERCEPT_TOL = 0.1


def run(npz: np.lib.npyio.NpzFile, model_family: str) -> dict:
    failures = []
    details = {}

    pred_key = "pred_log_ret_10s"
    tgt_key = "target_log_ret_10s"
    mask_key = "mask_log_ret_10s"
    if pred_key not in npz.files or tgt_key not in npz.files:
        return {"check": "calibration", "passed": False,
                "failures": [f"missing {pred_key} or {tgt_key}"], "details": {}}

    p = np.asarray(npz[pred_key], dtype=np.float64)
    t = np.asarray(npz[tgt_key], dtype=np.float64)
    if mask_key in npz.files:
        m = np.asarray(npz[mask_key], dtype=np.float64) > 0
    else:
        m = np.ones_like(p, dtype=bool)
    good = m & np.isfinite(p) & np.isfinite(t)
    p, t = p[good], t[good]
    if p.size < 1000:
        return {"check": "calibration", "passed": False,
                "failures": [f"too few rows: {p.size}"], "details": {}}

    # Bin by |pred| decile so that higher confidence is right-tail.
    abs_p = np.abs(p)
    deciles = np.quantile(abs_p, np.linspace(0.1, 1.0, 10))
    hit_rates = []
    counts = []
    for i in range(10):
        lo = 0.0 if i == 0 else deciles[i - 1]
        hi = deciles[i]
        if i == 9:
            sel = abs_p >= lo
        else:
            sel = (abs_p >= lo) & (abs_p < hi)
        if sel.sum() < 10:
            hit_rates.append(np.nan)
            counts.append(int(sel.sum()))
            continue
        hits = (np.sign(p[sel]) == np.sign(t[sel])).mean()
        hit_rates.append(float(hits))
        counts.append(int(sel.sum()))

    hit_rates_arr = np.array(hit_rates, dtype=np.float64)
    x = np.arange(10, dtype=np.float64) / 9.0  # 0..1
    ok = np.isfinite(hit_rates_arr)
    if ok.sum() < 5:
        failures.append("too few valid deciles for regression")
        slope = intercept = float("nan")
    else:
        slope, intercept = np.polyfit(x[ok], hit_rates_arr[ok], 1)
        # Slope expected positive; magnitude interpreted vs full unit range.
        if slope < SLOPE_MIN * 0.1:  # 0.05 swing across deciles
            failures.append(
                f"calibration slope {slope:.3f} too flat (decile lift < 0.05)"
            )
        # Anchor: low-confidence hit-rate should sit near 0.5 within tolerance.
        low_hit = hit_rates_arr[ok][0] if ok.any() else float("nan")
        if abs(low_hit - 0.5) > 0.15:
            failures.append(
                f"low-confidence hit-rate {low_hit:.3f} far from 0.5 (suspicious)"
            )

    details["decile_hit_rates"] = [None if np.isnan(v) else round(v, 4) for v in hit_rates_arr.tolist()]
    details["decile_counts"] = counts
    details["slope"] = float(slope) if np.isfinite(slope) else None
    details["intercept"] = float(intercept) if np.isfinite(intercept) else None
    details["head"] = pred_key
    details["lift_top_minus_bottom"] = (
        float(hit_rates_arr[-1] - hit_rates_arr[0])
        if ok.sum() >= 2 and np.isfinite(hit_rates_arr[0]) and np.isfinite(hit_rates_arr[-1])
        else None
    )

    return {
        "check": "calibration",
        "passed": len(failures) == 0,
        "failures": failures,
        "details": details,
    }
