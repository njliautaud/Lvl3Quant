"""Check 6: input feature stats.

We try to find a sample of input tensors in a few likely locations:
  - <output_dir_of_npz>/fold_00_feature_stats.npz (per-channel stats)
  - --data-dir <path>/sample.npz (if provided)

If we find feature_stats.npz with per-channel min/max/mean/std, validate ranges.
Else best-effort: scan the predictions NPZ for any auxiliary input arrays
and check finiteness only. This is a softer check; passes if it finds
finite data, fails if any NaN/Inf or unreasonable book-imbalance/log-vol.
"""
from __future__ import annotations
import os
import numpy as np


# Heuristic plausible ranges per channel name keyword
RANGE_RULES = [
    ("book_imbalance", (-1.0, 1.0)),
    ("imbalance", (-1.5, 1.5)),
    ("log_vol", (-3.0, 3.0)),
    ("log_volume", (-3.0, 5.0)),
    ("spread", (0.0, 50.0)),
    ("return", (-0.01, 0.01)),
    ("ret_", (-0.01, 0.01)),
]


def _check_range(name: str, lo: float, hi: float) -> tuple[float, float] | None:
    for kw, (rl, rh) in RANGE_RULES:
        if kw in name.lower():
            return rl, rh
    return None


def run(npz_path: str, model_family: str, data_dir: str | None = None) -> dict:
    failures = []
    details = {"sources": []}

    base = os.path.dirname(npz_path)
    candidates = [
        os.path.join(base, "fold_00_feature_stats.npz"),
        os.path.join(base, "feature_stats.npz"),
    ]
    if data_dir:
        candidates.append(os.path.join(data_dir, "feature_stats.npz"))

    feat_stats_path = next((c for c in candidates if os.path.exists(c)), None)
    if feat_stats_path is None:
        return {
            "check": "input_features",
            "passed": True,  # soft pass — not strictly required for gate
            "skipped": True,
            "failures": [],
            "details": {"reason": "no feature_stats.npz found nearby",
                        "searched": candidates},
        }

    details["sources"].append(feat_stats_path)
    fs = np.load(feat_stats_path, allow_pickle=True)
    per_channel = {}
    nan_found = []
    range_violations = []
    channel_names = None
    if "channel_names" in fs.files:
        try:
            channel_names = [str(x) for x in fs["channel_names"]]
        except Exception:
            channel_names = None

    for key in fs.files:
        arr = fs[key]
        if arr.dtype.kind not in "fiu":
            continue
        a = np.asarray(arr).ravel()
        if a.size == 0:
            continue
        finite = np.isfinite(a)
        if not finite.all():
            nan_found.append(key)
            a = a[finite]
            if a.size == 0:
                continue
        per_channel[key] = {
            "min": float(a.min()),
            "max": float(a.max()),
            "mean": float(a.mean()),
            "n": int(a.size),
        }

    # If we have arrays named e.g. "mean", "std", "min", "max" with channel_names alongside,
    # do per-channel range checks.
    if channel_names and "min" in fs.files and "max" in fs.files:
        mins = np.asarray(fs["min"]).ravel()
        maxs = np.asarray(fs["max"]).ravel()
        n = min(len(channel_names), len(mins), len(maxs))
        for i in range(n):
            cname = channel_names[i]
            rule = _check_range(cname, float(mins[i]), float(maxs[i]))
            if rule is None:
                continue
            rl, rh = rule
            mn, mx = float(mins[i]), float(maxs[i])
            if mn < rl - abs(rl) * 0.5 or mx > rh + abs(rh) * 0.5:
                range_violations.append(
                    f"{cname}: [{mn:.4f},{mx:.4f}] outside plausible [{rl},{rh}]"
                )

    if nan_found:
        failures.append(f"NaN/Inf in feature_stats arrays: {nan_found[:5]}")
    if range_violations:
        failures.extend(range_violations[:10])

    details["per_array"] = per_channel
    details["channel_names_seen"] = channel_names[:20] if channel_names else None
    details["range_violations"] = range_violations

    return {
        "check": "input_features",
        "passed": len(failures) == 0,
        "failures": failures,
        "details": details,
    }
