"""Check 1: NPZ load + NaN/Inf + shape consistency."""
from __future__ import annotations
import numpy as np


def run(npz: np.lib.npyio.NpzFile, model_family: str) -> dict:
    pred_keys = [k for k in npz.files if k.startswith("pred_")]
    tgt_keys = [k for k in npz.files if k.startswith("target_")]
    mask_keys = [k for k in npz.files if k.startswith("mask_")]

    n_pred = len(pred_keys)
    n_tgt = len(tgt_keys)
    n_mask = len(mask_keys)

    failures = []
    details = {
        "n_pred_heads": n_pred,
        "n_target_heads": n_tgt,
        "n_mask_heads": n_mask,
    }

    # All pred arrays must have a matching target
    pred_bases = {k[len("pred_"):] for k in pred_keys}
    tgt_bases = {k[len("target_"):] for k in tgt_keys}
    missing_targets = sorted(pred_bases - tgt_bases)
    if missing_targets:
        failures.append(f"pred heads without targets: {missing_targets[:5]}")

    # Reference shape
    ref_key = pred_keys[0]
    ref_shape = tuple(npz[ref_key].shape)
    details["ref_shape"] = ref_shape

    nan_counts: dict[str, int] = {}
    inf_counts: dict[str, int] = {}
    shape_mismatch: list[str] = []
    for k in pred_keys + tgt_keys:
        a = npz[k]
        if tuple(a.shape) != ref_shape:
            shape_mismatch.append(f"{k}={a.shape}")
            continue
        if np.issubdtype(a.dtype, np.floating):
            nans = int(np.isnan(a).sum())
            infs = int(np.isinf(a).sum())
            if nans:
                nan_counts[k] = nans
            if infs:
                inf_counts[k] = infs

    if shape_mismatch:
        failures.append(f"shape mismatch: {shape_mismatch[:5]}")
    if nan_counts:
        failures.append(f"NaN in {len(nan_counts)} heads (sample: {list(nan_counts.items())[:3]})")
    if inf_counts:
        failures.append(f"Inf in {len(inf_counts)} heads (sample: {list(inf_counts.items())[:3]})")

    details["nan_counts_n"] = len(nan_counts)
    details["inf_counts_n"] = len(inf_counts)
    details["total_rows"] = int(ref_shape[0]) if ref_shape else 0

    return {
        "check": "npz_health",
        "passed": len(failures) == 0,
        "failures": failures,
        "details": details,
    }
