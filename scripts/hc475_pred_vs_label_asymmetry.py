#!/usr/bin/env python3
"""
HC #475 R1 question 4 — Compare PREDICTION asymmetry vs LABEL asymmetry.

The previous diagnostic showed model predictions are wildly skewed (p_up_5s only
2.4% positive, log_ret_60s only 0.4% positive). The decisive next question is:

   Are the LABELS the model trained on also skewed?

If yes  -> the model is faithfully reflecting the label distribution. Fix the labels.
If no   -> the model has a representational defect. Fix the model / loss.

Outputs a side-by-side table and writes to reports/hc475_longshort_diagnosis/08_pred_vs_label_asymmetry.md.
Runs in <5s on Jupiter CPU.
"""
from __future__ import annotations
import json
from pathlib import Path
from datetime import datetime
import numpy as np

NPZ = "/home/jupiter/Lvl3Quant/output/hc432_v342_47day_validation/fold_00_ep1_oot_inference_47day_hc432.npz"
OUT_MD = Path("/home/jupiter/Lvl3Quant/reports/hc475_longshort_diagnosis/08_pred_vs_label_asymmetry.md")
OUT_JSON = Path("/home/jupiter/Lvl3Quant/output/hc475_ab/08_pred_vs_label_asymmetry.json")

DIRECTIONAL_SUFFIXES = [
    "log_ret_1s", "log_ret_5s", "log_ret_10s", "log_ret_30s", "log_ret_60s", "log_ret_5min",
    "p_up_5s", "p_up_10s", "p_up_30s", "p_up_60s",
    "log_ret_10s_q10", "log_ret_10s_q50", "log_ret_10s_q90",
    "log_ret_30s_q10", "log_ret_30s_q50", "log_ret_30s_q90",
    "log_ret_60s_q10", "log_ret_60s_q50", "log_ret_60s_q90",
    "fifo_tp4sl3_net", "fifo_tp8sl5_net",
]


def stats(arr: np.ndarray) -> dict:
    a = arr.astype(np.float64)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"n": 0}
    pos = a[a > 0]
    neg = a[a < 0]
    return {
        "n": int(a.size),
        "pos_share": float(pos.size) / a.size,
        "neg_share": float(neg.size) / a.size,
        "raw_mean": float(a.mean()),
        "raw_median": float(np.median(a)),
        "pos_p90": float(np.quantile(pos, 0.90)) if pos.size > 100 else None,
        "neg_p90_mag": float(np.quantile(-neg, 0.90)) if neg.size > 100 else None,
    }


def main() -> int:
    f = np.load(NPZ, allow_pickle=False)
    rows = []
    json_out = {}
    for suf in DIRECTIONAL_SUFFIXES:
        pkey = f"pred_{suf}"
        # Some labels have a 'target_' prefix; p_up_* heads may have a target named 'target_p_up_Xs'
        lkey = f"target_{suf}"
        if pkey not in f.files or lkey not in f.files:
            continue
        ps = stats(f[pkey])
        ls = stats(f[lkey])
        rows.append({"suf": suf, "pred": ps, "label": ls})
        json_out[suf] = {"pred": ps, "label": ls}
        ppos = ps.get("pos_share", float("nan"))
        lpos = ls.get("pos_share", float("nan"))
        gap = ppos - lpos if (ppos is not None and lpos is not None) else float("nan")
        verdict = "FAITHFUL" if abs(gap) <= 0.10 else ("MODEL-DEFECT" if abs(gap) > 0.10 else "?")
        print(
            f"[hc475_pvl] {suf:24s} pred_pos={ppos:.3f} label_pos={lpos:.3f} gap={gap:+.3f} -> {verdict}",
            flush=True,
        )

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(json_out, indent=2))

    lines = [
        "# HC #475 R1 Q4 — Pred vs Label Asymmetry",
        "",
        f"Generated: {datetime.now().isoformat(timespec='seconds')}",
        f"Source: `{NPZ}`",
        "",
        "Reading: if `gap` = `pred_pos_share - label_pos_share` is within ±0.10, the model is FAITHFULLY",
        "reflecting the label distribution → the labels are the lever to pull (HC #475 R4 R-bullet 2).",
        "If `|gap| > 0.10`, the model is amplifying or dampening label asymmetry → representational defect,",
        "requires loss-function / architecture-level fix.",
        "",
        "| Head | pred_pos% | label_pos% | gap | verdict |",
        "|------|-----------|------------|-----|---------|",
    ]
    for r in rows:
        suf = r["suf"]
        ppos = r["pred"].get("pos_share")
        lpos = r["label"].get("pos_share")
        if ppos is None or lpos is None:
            continue
        gap = ppos - lpos
        verdict = "FAITHFUL" if abs(gap) <= 0.10 else "MODEL-DEFECT"
        lines.append(
            f"| `{suf}` | {ppos:.3f} | {lpos:.3f} | {gap:+.3f} | {verdict} |"
        )
    lines.append("")
    lines.append("## Strategic implication per HC #475")
    lines.append("")
    lines.append("- All `FAITHFUL` rows = labeling redesign is the right lever (HC #475 R4 'better labeling' clause).")
    lines.append("- Any `MODEL-DEFECT` rows = loss-function or architecture-level fix needed for that head.")
    lines.append("- The action plan splits between the two: keep usable heads, retrain or drop defective heads.")
    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(lines))
    print(f"[hc475_pvl] wrote {OUT_MD}", flush=True)
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
