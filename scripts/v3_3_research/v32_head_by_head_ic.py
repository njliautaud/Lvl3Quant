"""
Head-by-head IC breakdown for v3.2 OOT predictions.

Compute IC + DA + n for every prediction head present in predictions.npz.
Output: ranked list of heads by IC magnitude.

This is the v3.2 ground-truth for which heads "won" — to be compared with v3.3's
learned log_sigma values after it trains. If a head has high IC in v3.2 but high
σ in v3.3 (meaning v3.3 down-weighted it), that's diagnostic. If a head has low
IC in v3.2 and low σ in v3.3 (v3.3 up-weighted it), that's the multi-task fix
working as expected.
"""
from __future__ import annotations
import json, csv
from pathlib import Path
import numpy as np
from scipy.stats import spearmanr

PREDS = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/fold_00_oot_predictions.npz")
OUT_CSV = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/head_by_head_ic.csv")
OUT_JSON = Path("/home/jupiter/Lvl3Quant/output/v3_2_deep_sim_20260512/head_by_head_ic.json")


def head_metrics(pred, target, mask):
    m = mask.astype(bool) & np.isfinite(pred) & np.isfinite(target)
    if m.sum() < 50:
        return float("nan"), float("nan"), float("nan"), int(m.sum())
    p, t = pred[m], target[m]
    try:
        ic, _ = spearmanr(p, t)
    except Exception:
        ic = float("nan")
    da = float(((p > 0) == (t > 0)).mean())
    # Magnitude correlation (Pearson on abs values)
    try:
        mag = float(np.corrcoef(np.abs(p), np.abs(t))[0, 1])
    except Exception:
        mag = float("nan")
    return float(ic), float(da), mag, int(m.sum())


def main():
    d = np.load(PREDS, allow_pickle=True)
    keys = list(d.keys())
    # Pair each pred_X with target_X / mask_X
    head_names = sorted({k[5:] for k in keys if k.startswith("pred_")})

    rows = []
    for h in head_names:
        pk = f"pred_{h}"
        tk = f"target_{h}"
        mk = f"mask_{h}"
        if pk not in d or tk not in d or mk not in d:
            continue
        ic, da, mag, n = head_metrics(d[pk], d[tk], d[mk])
        rows.append({"head": h, "n": n, "IC": ic, "DA": da, "MagCorr": mag})

    # Sort by abs(IC) descending (treat NaN as 0)
    rows.sort(key=lambda r: -abs(r["IC"]) if r["IC"] == r["IC"] else 0)

    with open(OUT_CSV, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["head", "n", "IC", "DA", "MagCorr"])
        w.writeheader()
        for r in rows:
            w.writerow(r)

    with open(OUT_JSON, "w") as f:
        json.dump(rows, f, indent=2, default=lambda x: None if (isinstance(x, float) and not np.isfinite(x)) else x)

    print(f"Wrote {len(rows)} heads to {OUT_CSV} and {OUT_JSON}")
    print("\nTop 12 heads by |IC|:")
    print(f"{'rank':>4}  {'head':<28} {'n':>10} {'IC':>8} {'DA':>7} {'MagCorr':>9}")
    for i, r in enumerate(rows[:12], 1):
        ic = r["IC"] if r["IC"] == r["IC"] else float("nan")
        mag = r["MagCorr"] if r["MagCorr"] == r["MagCorr"] else float("nan")
        print(f"{i:>4}  {r['head']:<28} {r['n']:>10,} {ic:>8.4f} {r['DA']:>7.4f} {mag:>9.4f}")
    print("\nBottom 12 heads by |IC|:")
    for i, r in enumerate(rows[-12:], 1):
        ic = r["IC"] if r["IC"] == r["IC"] else float("nan")
        mag = r["MagCorr"] if r["MagCorr"] == r["MagCorr"] else float("nan")
        print(f"{i:>4}  {r['head']:<28} {r['n']:>10,} {ic:>8.4f} {r['DA']:>7.4f} {mag:>9.4f}")


if __name__ == "__main__":
    main()
