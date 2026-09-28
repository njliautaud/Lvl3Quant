#!/usr/bin/env python3
"""h30s alpha-ceiling precomputation.

For each OOT day in mbo_events_smart_v3, compute the distribution of
realized labels at horizons 1s/5s/10s/30s. The p90 of |label| at h=30s
is the THEORETICAL CEILING for a perfect h=30s signal model. If that
ceiling < commission (0.376 ticks) the h=30s axis is dead before
Neptune's retrain even lands. This is HC #428 R2 pre-flight.

Writes: output/h30s_label_ceiling_v1/by_day.csv + summary.md

NOTE: scalar labels here are endpoint returns (price@t+h - price@t),
not MFE. They are a conservative LOWER BOUND on tradeable edge at
that horizon. ES tick = 0.25 pts. Labels appear to be in price-pt
units; converted to ticks (multiply by 4) for reporting.
"""
import json
from pathlib import Path
import numpy as np
import pandas as pd

DATA = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_smart_v3")
OUT = Path("/home/jupiter/Lvl3Quant/output/h30s_label_ceiling_v1")
OUT.mkdir(parents=True, exist_ok=True)

HORIZONS = ["1s", "5s", "10s", "30s"]
COMMISSION_TICKS = 0.376
MARKET_COST_TICKS = 1.376
# OOT window: late Feb -> April 29 per HC #462. Use 20260224 onward.
OOT_START = "20260224"
OOT_END = "20260429"


def main():
    rows = []
    npz_files = sorted(DATA.glob("*_mbo_events.npz"))
    print(f"Found {len(npz_files)} NPZ files in dataset")
    for fp in npz_files:
        day = fp.stem.split("_")[0]
        if day < OOT_START or day > OOT_END:
            continue
        try:
            f = np.load(fp, allow_pickle=True)
        except Exception as e:
            print(f"  skip {day}: {e}")
            continue
        n = len(f["timestamps"])
        if n < 1000:
            continue
        row = {"day": day, "n_events": n}
        for h in HORIZONS:
            key = f"labels_{h}"
            if key not in f.files:
                continue
            lab = f[key]
            # drop NaN
            valid = lab[~np.isnan(lab)]
            if len(valid) < 100:
                continue
            # labels are ALREADY in ticks (verified: p50abs_1s=1.0, p90abs_30s=11)
            ticks = valid * 1.0
            absticks = np.abs(ticks)
            row[f"std_{h}_tk"] = float(np.std(ticks))
            row[f"p50abs_{h}_tk"] = float(np.percentile(absticks, 50))
            row[f"p75abs_{h}_tk"] = float(np.percentile(absticks, 75))
            row[f"p90abs_{h}_tk"] = float(np.percentile(absticks, 90))
            row[f"p95abs_{h}_tk"] = float(np.percentile(absticks, 95))
            row[f"p99abs_{h}_tk"] = float(np.percentile(absticks, 99))
            row[f"frac_gt_commish_{h}"] = float((absticks > COMMISSION_TICKS).mean())
            row[f"frac_gt_market_{h}"] = float((absticks > MARKET_COST_TICKS).mean())
        rows.append(row)
        if len(rows) % 5 == 0:
            print(f"  processed {len(rows)} days...")
        f.close()
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "by_day.csv", index=False)

    # Summary across days
    lines = []
    lines.append("# h30s label-ceiling pre-flight (HC #428 R2)\n")
    lines.append(f"OOT window: {OOT_START} -> {OOT_END}, {len(df)} days\n")
    lines.append(f"Commission cost: {COMMISSION_TICKS} ticks. Market cost: {MARKET_COST_TICKS} ticks.\n\n")
    lines.append("## Cross-day means of |label| percentiles (ticks)\n\n")
    lines.append("| horizon | p50 | p75 | p90 | p95 | p99 | %|lab|>0.376 | %|lab|>1.376 |\n")
    lines.append("|---|---|---|---|---|---|---|---|\n")
    for h in HORIZONS:
        if f"p50abs_{h}_tk" not in df.columns:
            continue
        lines.append(
            f"| {h} | {df[f'p50abs_{h}_tk'].mean():.3f} | "
            f"{df[f'p75abs_{h}_tk'].mean():.3f} | "
            f"{df[f'p90abs_{h}_tk'].mean():.3f} | "
            f"{df[f'p95abs_{h}_tk'].mean():.3f} | "
            f"{df[f'p99abs_{h}_tk'].mean():.3f} | "
            f"{df[f'frac_gt_commish_{h}'].mean():.3f} | "
            f"{df[f'frac_gt_market_{h}'].mean():.3f} |\n"
        )
    lines.append("\n## Verdict\n")
    if "p90abs_30s_tk" in df.columns:
        p90 = df["p90abs_30s_tk"].mean()
        lines.append(f"- h=30s p90 |label| = {p90:.3f} ticks across-day mean\n")
        if p90 < COMMISSION_TICKS:
            lines.append(f"- VERDICT: BELOW commission ({COMMISSION_TICKS}). h=30s axis DEAD even with perfect signal.\n")
        elif p90 < MARKET_COST_TICKS:
            lines.append(f"- VERDICT: Above commission, below market cost. h=30s viable only with passive fills.\n")
        else:
            lines.append(f"- VERDICT: Above market cost. h=30s viable with market orders if signal selects top-decile moves.\n")
    (OUT / "summary.md").write_text("".join(lines))
    print(f"Wrote {OUT}/summary.md ({len(df)} days)")
    print("".join(lines))


if __name__ == "__main__":
    main()
