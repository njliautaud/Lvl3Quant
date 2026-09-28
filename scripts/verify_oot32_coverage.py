#!/usr/bin/env python3
"""
verify_oot32_coverage.py

HC #538 R3 — produce coverage_report.json showing per-day LGBM-vol and
cross-asset feature availability across the full 32-day OOT window
(actually 34 days in the oot_47day_perdate index).

Output:
  output/data_coverage_v1/coverage_report.json

Console summary printed at end. Idempotent — safe to re-run.
"""
from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/home/jupiter/Lvl3Quant")
OOT_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
VOL_DIR = ROOT / "output/vol_lgbm_v3"
XA_PARQUET = ROOT / "output/cross_asset_day_classifier_v1/combined_features.parquet"
COVERAGE_DIR = ROOT / "output/data_coverage_v1"
COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = COVERAGE_DIR / "coverage_report.json"


def main() -> int:
    oot_dates = sorted(
        p.stem.replace("oot_", "") for p in OOT_DIR.glob("oot_*.npz")
    )
    vol_dates = sorted(
        p.stem.split("_")[2] for p in VOL_DIR.glob("vol_v3_*_predictions.npz")
    )

    if XA_PARQUET.exists():
        xa_df = pd.read_parquet(XA_PARQUET)
        xa_dates = sorted(xa_df["date"].astype(str).tolist())
    else:
        xa_df = pd.DataFrame()
        xa_dates = []

    xa_feature_cols = [
        "VIX_close_prev",
        "VIX_change_5d",
        "VIX_zscore_20d",
        "NQ_overnight_return",
        "YM_overnight_return",
        "NQ_vs_ES_5d_corr",
        "SPX_5d_return",
        "DXY_5d_return",
    ]

    per_day = []
    for d in oot_dates:
        has_vol = d in vol_dates
        has_xa = d in xa_dates
        xa_full = False
        n_xa_nan = None
        if has_xa and len(xa_df) > 0:
            row = xa_df[xa_df["date"].astype(str) == d]
            if len(row) == 1:
                cols_present = [c for c in xa_feature_cols if c in row.columns]
                n_xa_nan = int(row[cols_present].isna().sum(axis=1).iloc[0])
                xa_full = (n_xa_nan == 0)
        # Quick vol sanity: load npz to make sure predictions array non-empty
        n_vol_preds = None
        if has_vol:
            try:
                vp = np.load(
                    VOL_DIR / f"vol_v3_{d}_predictions.npz", allow_pickle=False
                )
                n_vol_preds = int(vp["predictions"].shape[0])
            except Exception as e:
                n_vol_preds = -1
        per_day.append(
            {
                "date": d,
                "has_lgbm_vol": has_vol,
                "n_vol_predictions": n_vol_preds,
                "has_cross_asset_row": has_xa,
                "cross_asset_features_full": xa_full,
                "n_cross_asset_nan_cols": n_xa_nan,
            }
        )

    n_total = len(oot_dates)
    n_vol_ok = sum(1 for r in per_day if r["has_lgbm_vol"])
    n_xa_ok = sum(1 for r in per_day if r["has_cross_asset_row"])
    n_xa_full = sum(1 for r in per_day if r["cross_asset_features_full"])
    n_full_coverage = sum(
        1 for r in per_day if r["has_lgbm_vol"] and r["cross_asset_features_full"]
    )

    missing_vol = [r["date"] for r in per_day if not r["has_lgbm_vol"]]
    missing_xa = [r["date"] for r in per_day if not r["has_cross_asset_row"]]
    xa_partial = [
        r["date"]
        for r in per_day
        if r["has_cross_asset_row"] and not r["cross_asset_features_full"]
    ]

    report = {
        "generated_at": datetime.now().isoformat(),
        "oot_dates_total": n_total,
        "oot_dates_range": [oot_dates[0], oot_dates[-1]] if oot_dates else [None, None],
        "lgbm_vol": {
            "n_covered": n_vol_ok,
            "n_missing": len(missing_vol),
            "missing_dates": missing_vol,
        },
        "cross_asset": {
            "n_rows_present": n_xa_ok,
            "n_rows_full_features": n_xa_full,
            "n_rows_missing": len(missing_xa),
            "missing_dates": missing_xa,
            "partial_dates": xa_partial,
        },
        "full_coverage_days": n_full_coverage,
        "full_coverage_ratio": (
            n_full_coverage / n_total if n_total else None
        ),
        "per_day": per_day,
    }

    with open(REPORT_PATH, "w") as fh:
        json.dump(report, fh, indent=2)

    print("=" * 72)
    print(f"Coverage report: {REPORT_PATH}")
    print(f"OOT days: {n_total}")
    print(f"  LGBM-vol covered  : {n_vol_ok}/{n_total}")
    if missing_vol:
        print(f"     missing       : {missing_vol}")
    print(f"  Cross-asset rows  : {n_xa_ok}/{n_total}")
    print(f"  XA full features  : {n_xa_full}/{n_total}")
    if missing_xa:
        print(f"     missing       : {missing_xa}")
    if xa_partial:
        print(f"     partial       : {xa_partial}")
    print(f"  Full coverage     : {n_full_coverage}/{n_total} "
          f"({100.0 * n_full_coverage / n_total:.1f}%)")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
