#!/usr/bin/env python3
"""
HC #454 R3(b)(1) — PatchTST Joint-Gate Smoothness Diagnostic.

Runs the HC #455 R1 joint informativeness + smoothness gate against existing
PatchTST per-day OOT predictions in /home/jupiter/Lvl3Quant/output/patchtst_bulk_oot/.

Each prediction NPZ has shape (N_samples, 3) over horizons (presumed 1s/5s/10s
matching the v3.4.2 PatchTST training schema). We treat each horizon column as
one head and apply the joint-gate per HC #455 R1:

  (a) mean_ac1 = mean lag-1 autocorr across days (smoothness — stream-coherence proxy)
  (b) median_spread_ratio = median(p90-p10) / median(full_range) per day, then median
        (informativeness — head uses its dynamic range, not stuck at one value)
  (c) balanced_sign_frac = fraction of days where pos_frac ∈ [0.05, 0.95]
        (actionability — both signs meaningfully represented)

  actionable_score = mean_ac1 * median_spread_ratio * balanced_sign_frac
  PASS threshold: actionable_score >= 0.05 (HC #455 R1 loose floor)

Gates HC #454 R3 Razer GPU dispatch:
  - If at least one horizon PASSES → dispatch Razer to densify PatchTST inference
    (test HC #453 R8 stream principle at lower stride)
  - If ALL horizons FAIL → HC #453 R8 transformer-demotion is principled,
    dispatch Razer to 2-layer pure-SSM probe (HC #454 R3(b)(3)) instead.
"""

import json
from pathlib import Path

import numpy as np


INPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/patchtst_bulk_oot")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/hc454_patchtst_joint_gate")
HORIZON_NAMES = ["pred_log_ret_1s", "pred_log_ret_5s", "pred_log_ret_10s"]
ACTIONABLE_FLOOR = 0.05  # HC #455 R1
MIN_SAMPLES_PER_DAY = 1000  # exclude tiny smoke-test days


def autocorr_lag1(x: np.ndarray) -> float:
    x = x[~np.isnan(x)]
    if x.size < 100:
        return float("nan")
    a, b = x[:-1], x[1:]
    a, b = a - a.mean(), b - b.mean()
    den = float(np.sqrt((a * a).sum() * (b * b).sum()))
    if den <= 0:
        return float("nan")
    return float((a * b).sum() / den)


def per_day_stats(pred_col: np.ndarray) -> dict:
    """Stats for one (day, head). pred_col shape: (N,)."""
    p = pred_col[~np.isnan(pred_col)]
    if p.size < MIN_SAMPLES_PER_DAY:
        return None
    ac1 = autocorr_lag1(pred_col)
    p10, p90 = float(np.percentile(p, 10)), float(np.percentile(p, 90))
    pmin, pmax = float(p.min()), float(p.max())
    spread = p90 - p10
    full_range = max(pmax - pmin, 1e-12)
    spread_ratio = spread / full_range
    pos_frac = float((p > 0).mean())
    return {
        "n": int(p.size),
        "ac1": ac1,
        "p10_p90_spread": spread,
        "full_range": pmax - pmin,
        "spread_ratio": spread_ratio,
        "pos_frac": pos_frac,
        "std": float(p.std()),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    files = sorted(INPUT_DIR.glob("*_predictions.npz"))
    print(f"Found {len(files)} PatchTST day NPZs in {INPUT_DIR}")

    # head -> list of per-day stats dicts
    head_days = {h: [] for h in HORIZON_NAMES}
    days_skipped = []

    for f in files:
        d = np.load(f)
        if "predictions" not in d:
            continue
        preds = d["predictions"]
        if preds.ndim != 2 or preds.shape[1] != 3:
            continue
        date = str(d.get("date", f.stem))
        any_kept = False
        for i, hname in enumerate(HORIZON_NAMES):
            stats = per_day_stats(preds[:, i])
            if stats is None:
                continue
            stats["date"] = date
            head_days[hname].append(stats)
            any_kept = True
        if not any_kept:
            days_skipped.append((date, preds.shape[0]))

    print(f"Skipped {len(days_skipped)} days with <{MIN_SAMPLES_PER_DAY} samples per head")

    # Aggregate per head per HC #455 R1
    summary = []
    for hname in HORIZON_NAMES:
        days = head_days[hname]
        if not days:
            summary.append({"head": hname, "n_days": 0, "verdict": "NO_DATA"})
            continue
        ac1s = np.array([d["ac1"] for d in days if not np.isnan(d["ac1"])])
        spreads = np.array([d["p10_p90_spread"] for d in days])
        ranges = np.array([d["full_range"] for d in days])
        pos_fracs = np.array([d["pos_frac"] for d in days])

        mean_ac1 = float(ac1s.mean()) if ac1s.size else float("nan")
        # HC #455 R1: median(p90-p10) / median(full_range)
        median_spread_ratio = float(np.median(spreads) / max(np.median(ranges), 1e-12))
        balanced_sign_frac = float(((pos_fracs >= 0.05) & (pos_fracs <= 0.95)).mean())
        actionable_score = mean_ac1 * median_spread_ratio * balanced_sign_frac
        verdict = "PASS" if (actionable_score >= ACTIONABLE_FLOOR and not np.isnan(actionable_score)) else "FAIL"

        summary.append({
            "head": hname,
            "n_days": len(days),
            "mean_ac1": mean_ac1,
            "median_spread_ratio": median_spread_ratio,
            "balanced_sign_frac": balanced_sign_frac,
            "actionable_score": actionable_score,
            "verdict": verdict,
            "per_day_pos_frac_p10_p90": [float(np.percentile(pos_fracs, 10)), float(np.percentile(pos_fracs, 90))],
            "per_day_ac1_p10_p90": [float(np.percentile(ac1s, 10)), float(np.percentile(ac1s, 90))] if ac1s.size else None,
        })

    out_json = OUTPUT_DIR / "joint_gate_summary.json"
    with open(out_json, "w") as fh:
        json.dump({"floor": ACTIONABLE_FLOOR, "input_dir": str(INPUT_DIR),
                   "n_files": len(files), "summary": summary,
                   "days_skipped": days_skipped[:50]}, fh, indent=2)

    print(f"\n=== HC #454 R3(b)(1) PatchTST Joint-Gate Diagnostic — VERDICT ===")
    print(f"Floor: actionable_score >= {ACTIONABLE_FLOOR}")
    print(f"{'head':<22} {'n_days':>6} {'mean_ac1':>10} {'spread':>8} {'sign_bal':>9} {'score':>8} {'verdict':>8}")
    for s in summary:
        if s["n_days"] == 0:
            print(f"{s['head']:<22} {s['n_days']:>6}  {'(no data)':>40}")
            continue
        print(f"{s['head']:<22} {s['n_days']:>6} {s['mean_ac1']:>10.4f} {s['median_spread_ratio']:>8.4f} "
              f"{s['balanced_sign_frac']:>9.4f} {s['actionable_score']:>8.4f} {s['verdict']:>8}")

    n_pass = sum(1 for s in summary if s.get("verdict") == "PASS")
    print(f"\n{n_pass}/{len(summary)} horizons PASS joint gate")
    if n_pass > 0:
        print("→ HC #454 R3(b) DECISION: dispatch Razer to DENSIFY PatchTST inference (stride < 250)")
        print("    to test whether stream cadence reduces flicker — HC #453 R8 may be overturned.")
    else:
        print("→ HC #454 R3(b) DECISION: HC #453 R8 transformer-demotion is PRINCIPLED.")
        print("    Dispatch Razer to 2-layer pure-SSM probe per HC #454 R3(b)(3) instead.")
    print(f"\nFull JSON: {out_json}")


if __name__ == "__main__":
    main()
