"""Check 4: direction balance — long/short signal counts within 5x of each other,
and predicted-up-prob histogram not bunched.
"""
from __future__ import annotations
import numpy as np


RATIO_MAX = 5.0
MIN_TAIL_FRAC = 0.02  # at least 2% of mass in each tail of p_up


def run(npz: np.lib.npyio.NpzFile, model_family: str) -> dict:
    failures = []
    details = {}

    # 1) Sign balance via log_ret_10s
    if "pred_log_ret_10s" in npz.files:
        p = np.asarray(npz["pred_log_ret_10s"], dtype=np.float64)
        p = p[np.isfinite(p)]
        n_long = int((p > 0).sum())
        n_short = int((p < 0).sum())
        n_zero = int((p == 0).sum())
        details["sign_counts"] = {"long": n_long, "short": n_short, "zero": n_zero}
        if min(n_long, n_short) == 0:
            failures.append("all signals on one side")
        else:
            ratio = max(n_long, n_short) / max(1, min(n_long, n_short))
            details["long_short_ratio"] = round(ratio, 3)
            if ratio > RATIO_MAX:
                failures.append(f"long/short ratio {ratio:.2f} > {RATIO_MAX}")
    else:
        failures.append("missing pred_log_ret_10s for direction check")

    # 2) Probability head bunching
    for k in ["pred_p_up_10s", "pred_p_up_30s"]:
        if k not in npz.files:
            continue
        a = np.asarray(npz[k], dtype=np.float64)
        a = a[np.isfinite(a)]
        if a.size == 0:
            failures.append(f"{k}: empty")
            continue
        # If outputs are logits (range outside [0,1]) we map via sigmoid for histogram only.
        lo, hi = float(a.min()), float(a.max())
        if lo < 0.0 or hi > 1.0:
            sig = 1.0 / (1.0 + np.exp(-a))
        else:
            sig = a
        below = float((sig < 0.4).mean())
        above = float((sig > 0.6).mean())
        details[k] = {
            "raw_min": lo,
            "raw_max": hi,
            "frac_below_0.4": round(below, 4),
            "frac_above_0.6": round(above, 4),
            "mean": float(sig.mean()),
        }
        if below < MIN_TAIL_FRAC or above < MIN_TAIL_FRAC:
            failures.append(
                f"{k}: bunched — frac<0.4={below:.3f} frac>0.6={above:.3f} (need >= {MIN_TAIL_FRAC})"
            )

    return {
        "check": "direction",
        "passed": len(failures) == 0,
        "failures": failures,
        "details": details,
    }
