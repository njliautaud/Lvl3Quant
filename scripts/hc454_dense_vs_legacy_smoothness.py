#!/usr/bin/env python3
"""HC #454 R3(b)(2) — Dense vs Legacy PatchTST smoothness comparison.

Compares lag-1 autocorr (smoothness) of new dense (stride=25) Razer
inference against existing legacy (stride=250) bulk OOT predictions on
overlapping days. If dense AC1 >> legacy AC1, HC #453 R8 transformer
demotion was an artifact of sparse stride, not architecture failure.
"""
import json
from pathlib import Path
import numpy as np

DENSE = Path("/home/jupiter/Lvl3Quant/output/hc454_dense_patchtst")
LEGACY = Path("/home/jupiter/Lvl3Quant/output/patchtst_bulk_oot")
OUT = Path("/home/jupiter/Lvl3Quant/output/hc454_dense_patchtst/smoothness_compare.json")
HEADS = ["1s", "5s", "10s"]
MIN_N = 1000


def ac1(x):
    x = x[~np.isnan(x)]
    if x.size < 100:
        return float("nan")
    a, b = x[:-1] - x[:-1].mean(), x[1:] - x[1:].mean()
    d = float(np.sqrt((a * a).sum() * (b * b).sum()))
    return float((a * b).sum() / d) if d > 0 else float("nan")


def load_preds(f):
    d = np.load(f)
    p = d.get("predictions")
    if p is None or p.ndim != 2 or p.shape[1] != 3 or p.shape[0] < MIN_N:
        return None
    return p


def summarize(files):
    """Return dict day -> [ac1_1s, ac1_5s, ac1_10s] and aggregates."""
    out = {}
    for f in files:
        p = load_preds(f)
        if p is None:
            continue
        day = f.stem.split("_")[0]
        out[day] = [ac1(p[:, i]) for i in range(3)]
    return out


def main():
    dense_files = sorted(DENSE.glob("*_dense_predictions.npz"))
    legacy_files = sorted(LEGACY.glob("*_predictions.npz"))
    dense = summarize(dense_files)
    legacy = summarize(legacy_files)

    overlap = sorted(set(dense) & set(legacy))
    print(f"Dense days: {len(dense)} | Legacy days: {len(legacy)} | Overlap: {len(overlap)}")

    rows = []
    for day in overlap:
        d = dense[day]
        l = legacy[day]
        rows.append({"day": day, "dense_ac1": d, "legacy_ac1": l,
                     "delta": [d[i] - l[i] for i in range(3)]})

    # aggregates
    def agg(idx, kind):
        vals = np.array([r[kind + "_ac1"][idx] for r in rows
                         if not np.isnan(r[kind + "_ac1"][idx])])
        return float(vals.mean()) if vals.size else float("nan")

    means = {}
    for i, h in enumerate(HEADS):
        means[h] = {"dense": agg(i, "dense"), "legacy": agg(i, "legacy")}
        means[h]["delta"] = means[h]["dense"] - means[h]["legacy"]

    result = {"overlap_days": overlap, "per_day": rows, "means": means,
              "n_overlap": len(overlap)}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(result, fh, indent=2)

    print(f"\n=== HC #454 R3(b)(2) Smoothness: Dense (stride=25) vs Legacy (stride=250) ===")
    print(f"{'horizon':<8} {'dense_ac1':>10} {'legacy_ac1':>11} {'delta':>8}")
    for h in HEADS:
        m = means[h]
        print(f"{h:<8} {m['dense']:>10.4f} {m['legacy']:>11.4f} {m['delta']:>+8.4f}")
    print(f"\nN overlap days: {len(overlap)}")
    if all(means[h]["delta"] > 0.10 for h in HEADS):
        print("\n>>> VERDICT: Dense stride MATERIALLY smoother across all horizons.")
        print(">>> HC #453 R8 transformer demotion was a STRIDE artifact, not architecture failure.")
    elif means["5s"]["delta"] > 0.05 or means["10s"]["delta"] > 0.05:
        print("\n>>> VERDICT: Dense stride PARTIALLY smoother on 5s/10s.")
        print(">>> HC #453 R8 demotion is OVERSTATED for the slower heads.")
    else:
        print("\n>>> VERDICT: Dense stride did NOT materially improve smoothness.")
        print(">>> HC #453 R8 transformer demotion STANDS.")
    print(f"\nFull JSON: {OUT}")


if __name__ == "__main__":
    main()
