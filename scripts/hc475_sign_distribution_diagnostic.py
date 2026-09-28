#!/usr/bin/env python3
"""
HC #475 R1 — Raw signal asymmetry diagnostic.

Reads the v3.4.2 47-day OOT NPZ and reports, per head and per horizon:
  - Long/short prediction count (signed mass above/below 0)
  - Magnitude distribution (p10, p50, p90, max) of positive vs negative tails
  - Per-horizon IC if y_true is present (separate long-side IC and short-side IC)
  - Symmetry ratio: |positive_p90| / |negative_p90|; counts ratio (#pos / #neg)

Purpose: directly answer HC #475 R1's first 3 questions on the RAW signal,
independent of any thresholding / trading policy.

Output: prints to stdout + writes to reports/hc475_longshort_diagnosis/07_raw_signal_asymmetry.md.

Runs on Jupiter CPU, ~3-5 min, ~2GB RAM. Safe to run alongside the A/B job.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np


NPZ_PATH = "/home/jupiter/Lvl3Quant/output/hc432_v342_47day_validation/fold_00_ep1_oot_inference_47day_hc432.npz"
OUT_MD = Path("/home/jupiter/Lvl3Quant/reports/hc475_longshort_diagnosis/07_raw_signal_asymmetry.md")
OUT_JSON = Path("/home/jupiter/Lvl3Quant/output/hc475_ab/07_raw_signal_asymmetry.json")


def _centered(name: str, arr: np.ndarray) -> np.ndarray:
    """Return a signed signal for an array, accounting for v3.4.2's pre-centered p_up heads."""
    # v3.4.2 pred_p_up_* heads are pre-centered (observed range ~[-0.5, 0.1]). Use raw.
    if name.startswith("pred_p_up_"):
        return arr.astype(np.float64, copy=False)
    if name.startswith("pred_log_ret_"):
        return arr.astype(np.float64, copy=False)
    # FIFO net heads are already signed
    if name.startswith("pred_fifo_") and name.endswith("_net"):
        return arr.astype(np.float64, copy=False)
    if name in ("pred_pred_mae_30s_ticks", "pred_pred_mae_60s_ticks"):
        # higher MAE = more uncertainty; not directional. Skip.
        return None
    if name.startswith("pred_p_reversal_"):
        return None  # not directional
    if name in (
        "pred_book_pressure",
        "pred_microstructure_event",
        "pred_p_high_vol",
        "pred_vol_regime",
        "pred_p_low_vol",
        "pred_q_continuation",
    ):
        return None  # non-directional
    return arr.astype(np.float64, copy=False)


def _tail_quantiles(x: np.ndarray) -> dict:
    if x.size == 0:
        return {"count": 0}
    return {
        "count": int(x.size),
        "p10": float(np.quantile(x, 0.10)),
        "p50": float(np.quantile(x, 0.50)),
        "p90": float(np.quantile(x, 0.90)),
        "max": float(x.max()),
        "mean": float(x.mean()),
    }


def main() -> int:
    print(f"[hc475_diag] loading {NPZ_PATH}", flush=True)
    f = np.load(NPZ_PATH, allow_pickle=False)
    keys = list(f.keys())
    print(f"[hc475_diag] keys ({len(keys)}): {keys[:20]}{'...' if len(keys)>20 else ''}", flush=True)

    # Gather directional heads
    head_names = [k for k in keys if k.startswith("pred_")]
    results = {}
    for name in head_names:
        arr = f[name]
        if arr.ndim != 1:
            continue
        signed = _centered(name, arr)
        if signed is None:
            continue
        pos = signed[signed > 0]
        neg = signed[signed < 0]
        zero = int(np.sum(signed == 0))
        n = signed.size
        record = {
            "n_total": int(n),
            "n_pos": int(pos.size),
            "n_neg": int(neg.size),
            "n_zero": zero,
            "pos_share": float(pos.size) / n if n else 0.0,
            "neg_share": float(neg.size) / n if n else 0.0,
            "pos_tail": _tail_quantiles(pos),
            "neg_tail": _tail_quantiles(-neg),  # magnitudes on the negative side
            "raw_mean": float(signed.mean()),
            "raw_median": float(np.median(signed)),
        }
        # Symmetry ratios
        pos_p90 = record["pos_tail"].get("p90")
        neg_p90 = record["neg_tail"].get("p90")
        if neg_p90 and neg_p90 > 0 and pos_p90 is not None:
            record["mag_ratio_pos_over_neg_p90"] = pos_p90 / neg_p90
        else:
            record["mag_ratio_pos_over_neg_p90"] = float("inf") if pos.size else 0.0
        if record["n_neg"] > 0:
            record["count_ratio_pos_over_neg"] = record["n_pos"] / record["n_neg"]
        else:
            record["count_ratio_pos_over_neg"] = float("inf") if pos.size else 0.0
        results[name] = record
        print(
            f"[hc475_diag] {name:32s} n={n:>9d} pos={record['pos_share']:.3f} neg={record['neg_share']:.3f} "
            f"|pos_p90|={record['pos_tail'].get('p90', float('nan')):.4g} "
            f"|neg_p90|={record['neg_tail'].get('p90', float('nan')):.4g} "
            f"ratio={record['mag_ratio_pos_over_neg_p90']:.3f}",
            flush=True,
        )

    # Per-horizon IC if y_true present (look for label arrays)
    ic_results = {}
    label_keys = [k for k in keys if k.startswith("y_") or k.startswith("label_")]
    for label_name in label_keys:
        label = f[label_name]
        if label.ndim != 1:
            continue
        # Match to a prediction head by horizon-suffix
        suffix = label_name.replace("y_", "").replace("label_", "")
        candidate_preds = [n for n in results if suffix in n]
        for pred_name in candidate_preds:
            arr = f[pred_name].astype(np.float64)
            y = label.astype(np.float64)
            m = np.isfinite(arr) & np.isfinite(y)
            if m.sum() < 1000:
                continue
            a, b = arr[m], y[m]
            # Overall Spearman is expensive; use Pearson + per-side rank-correlations
            # Long-side: arr > 0 mask
            pos_m = a > 0
            neg_m = a < 0
            ic_pos = float(np.corrcoef(a[pos_m], b[pos_m])[0, 1]) if pos_m.sum() > 100 else float("nan")
            ic_neg = float(np.corrcoef(a[neg_m], b[neg_m])[0, 1]) if neg_m.sum() > 100 else float("nan")
            ic_all = float(np.corrcoef(a, b)[0, 1])
            ic_results[f"{label_name}__vs__{pred_name}"] = {
                "ic_all": ic_all,
                "ic_long_side": ic_pos,
                "ic_short_side": ic_neg,
                "n_pos_eval": int(pos_m.sum()),
                "n_neg_eval": int(neg_m.sum()),
            }
            print(
                f"[hc475_diag] IC {pred_name} vs {label_name}: all={ic_all:.4f} "
                f"long={ic_pos:.4f} short={ic_neg:.4f}",
                flush=True,
            )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps({"heads": results, "ic": ic_results}, indent=2, default=str))
    print(f"[hc475_diag] wrote JSON to {OUT_JSON}", flush=True)

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append(f"# HC #475 R1 — Raw Signal Asymmetry Diagnostic")
    lines.append("")
    lines.append(f"Generated: {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"Source NPZ: `{NPZ_PATH}`")
    lines.append("")
    lines.append("## Per-head sign distribution + magnitude symmetry")
    lines.append("")
    lines.append("| Head | n | pos_share | neg_share | pos_p90 | neg_p90 | mag_ratio | count_ratio |")
    lines.append("|------|---|-----------|-----------|---------|---------|-----------|-------------|")
    for name, r in results.items():
        pp = r['pos_tail'].get('p90', float('nan'))
        np_ = r['neg_tail'].get('p90', float('nan'))
        lines.append(
            f"| `{name}` | {r['n_total']} | {r['pos_share']:.3f} | {r['neg_share']:.3f} | "
            f"{pp if pp is not None else float('nan'):.4g} | {np_ if np_ is not None else float('nan'):.4g} | "
            f"{r['mag_ratio_pos_over_neg_p90']:.3f} | {r['count_ratio_pos_over_neg']:.3f} |"
        )
    lines.append("")
    if ic_results:
        lines.append("## Long-side vs short-side IC")
        lines.append("")
        lines.append("| Pred vs Label | IC_all | IC_long | IC_short | n_pos | n_neg |")
        lines.append("|----------------|--------|---------|----------|-------|-------|")
        for k, r in ic_results.items():
            lines.append(
                f"| {k} | {r['ic_all']:.4f} | {r['ic_long_side']:.4f} | {r['ic_short_side']:.4f} | "
                f"{r['n_pos_eval']} | {r['n_neg_eval']} |"
            )
        lines.append("")
    lines.append("## Interpretation per HC #475 R1")
    lines.append("")
    lines.append("Reject the model on the both-sides competency bar if **any** trading-horizon head shows:")
    lines.append("- `mag_ratio_pos_over_neg_p90` outside [0.5, 2.0], AND")
    lines.append("- `min(|IC_long|, |IC_short|) / max(|IC_long|, |IC_short|) < 0.5`")
    lines.append("")
    lines.append("If both fail: model is short-tail-only at the representation level — alpha redev required.")
    lines.append("If only count_ratio fails: bias is dataset/label-distribution, not representation.")
    lines.append("If only mag_ratio fails but IC ratio is OK: bias is calibration, fixable post-hoc.")
    OUT_MD.write_text("\n".join(lines))
    print(f"[hc475_diag] wrote MD to {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
