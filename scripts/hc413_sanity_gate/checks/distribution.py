"""Check 2: per-horizon-head distribution. Not degenerate, not all-zero, not constant."""
from __future__ import annotations
import numpy as np


HORIZON_KEYS = [
    "pred_log_ret_1s",
    "pred_log_ret_5s",
    "pred_log_ret_10s",
    "pred_log_ret_30s",
    "pred_log_ret_60s",
    "pred_log_ret_5min",
]

STD_FLOOR = 1e-2


def _stats(a: np.ndarray) -> dict:
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": None, "std": None, "min": None, "max": None, "n": 0}
    return {
        "mean": float(a.mean()),
        "std": float(a.std()),
        "min": float(a.min()),
        "max": float(a.max()),
        "n": int(a.size),
    }


def run(npz: np.lib.npyio.NpzFile, model_family: str) -> dict:
    failures = []
    per_head = {}
    for k in HORIZON_KEYS:
        if k not in npz.files:
            failures.append(f"missing head: {k}")
            continue
        arr = npz[k]
        # For log returns, the natural scale is ~1e-5..1e-3 per step.
        # Use a relative std floor: predictions must be non-constant AND have
        # variance comparable to a reasonable fraction of target variance.
        s = _stats(arr)
        per_head[k] = s
        if s["n"] == 0:
            failures.append(f"{k}: empty after finite filter")
            continue
        if s["std"] is None or s["std"] <= 0:
            failures.append(f"{k}: zero std (constant predictions)")
            continue
        # Relative degeneracy vs target std
        tgt_key = "target_" + k[len("pred_"):]
        if tgt_key in npz.files:
            t = _stats(npz[tgt_key])
            per_head[k]["target_std"] = t["std"]
            if t["std"] and s["std"] < 0.01 * t["std"]:
                failures.append(
                    f"{k}: pred std {s['std']:.3e} < 1% of target std {t['std']:.3e} (degenerate)"
                )
        # Absolute floor for normalized prob-style heads handled in direction check.

    # Probability heads — must not be all 0 or all 1
    for k in ["pred_p_up_5s", "pred_p_up_10s", "pred_p_up_30s", "pred_p_up_60s"]:
        if k in npz.files:
            arr = npz[k]
            s = _stats(arr)
            per_head[k] = s
            if s["n"] == 0:
                failures.append(f"{k}: empty")
                continue
            if s["std"] is None or s["std"] < STD_FLOOR:
                failures.append(f"{k}: std {s['std']} < {STD_FLOOR} (degenerate prob head)")
            if s["min"] is not None and s["min"] == s["max"]:
                failures.append(f"{k}: constant (min==max=={s['min']})")

    return {
        "check": "distribution",
        "passed": len(failures) == 0,
        "failures": failures,
        "details": {"per_head": per_head},
    }
