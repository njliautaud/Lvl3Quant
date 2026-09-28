#!/usr/bin/env python3
"""
backfill_cross_asset_oot32.py

HC #538 R3 backfill — extend
output/cross_asset_day_classifier_v1/combined_features.parquet from its
current 15-day coverage (20260316..20260414, less 20260405) to the full
32-day OOT window used by confluence_filter_v1.

Reuses:
  - day_classifier_v1.build_day_features  (ES early-session features)
  - cross_asset_day_classifier_v1.build_cross_asset_features  (XA features)
  - cross_asset_day_classifier_v1._try_fetch_yf  (panel fetch)

Output schema EXACTLY matches the existing parquet so the confluence agent
ingests without code change. Existing rows are NOT overwritten; new rows
are appended. Rows that lack ground-truth labels (no per_day_fifo entry)
get `label_profitable = -1` and NaN P&L columns — flagged as unlabeled so
no future evaluator silently treats them as "negative".

Run:
  python3 scripts/backfill_cross_asset_oot32.py

CPU-only.
"""
from __future__ import annotations

import json
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=RuntimeWarning)

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT))

OOT_DIR = ROOT / "output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate"
XA_OUT_DIR = ROOT / "output/cross_asset_day_classifier_v1"
XA_PARQUET = XA_OUT_DIR / "combined_features.parquet"
PER_DAY_FIFO = ROOT / "output/meta_classifier_v1_fifo/per_day_fifo.csv"
ECON_CAL_PATH = ROOT / "data/external/economic_calendar_2023_2026.json"
MBO_DIR_NEW = ROOT / "data/processed/mbo_events_smart_v3"

COVERAGE_DIR = ROOT / "output/data_coverage_v1"
COVERAGE_DIR.mkdir(parents=True, exist_ok=True)
BACKFILL_LOG = COVERAGE_DIR / "backfill_xa.log"

# Import upstream scripts. day_classifier_v1 hardcodes MBO_DIR to the
# legacy `mbo_events` path that no longer exists; we monkey-patch it.
sys.path.insert(0, str(ROOT / "scripts"))
import day_classifier_v1 as dcv1  # type: ignore  # noqa: E402
import cross_asset_day_classifier_v1 as xacv1  # type: ignore  # noqa: E402

dcv1.MBO_DIR = MBO_DIR_NEW


def discover_oot_dates() -> list[int]:
    return sorted(
        int(p.stem.replace("oot_", "")) for p in OOT_DIR.glob("oot_*.npz")
    )


def load_econ_calendar() -> List[Dict]:
    if ECON_CAL_PATH.exists():
        try:
            return json.load(open(ECON_CAL_PATH))["events"]
        except Exception as e:
            print(f"[warn] econ calendar load failed: {e}", flush=True)
    return []


def load_existing_labels() -> Dict[int, Dict]:
    """Return {date_int: {label_profitable, day_mean_net_realized,
    day_sum_net_realized, n_filled}} from per_day_fifo (short_10s_thr55)."""
    if not PER_DAY_FIFO.exists():
        return {}
    pd_df = pd.read_csv(PER_DAY_FIFO)
    sub = pd_df[pd_df["candidate"] == "short_10s_thr55"].copy()
    out: Dict[int, Dict] = {}
    for _, r in sub.iterrows():
        dmn = float(r.get("day_mean_net_realized", float("nan")))
        if not (np.isfinite(dmn) and r.get("n_filled", 0) > 0):
            continue
        di = int(r["date"])
        out[di] = {
            "label_profitable": int(dmn > 0),
            "day_mean_net_realized": dmn,
            "day_sum_net_realized": float(r.get("day_sum_net_realized", float("nan"))),
            "n_filled": float(r.get("n_filled", 0.0)),
        }
    return out


def main() -> int:
    t0 = time.time()
    log_fh = open(BACKFILL_LOG, "w")

    def _log(msg: str) -> None:
        line = f"{datetime.now().isoformat()} {msg}"
        print(line, flush=True)
        log_fh.write(line + "\n")
        log_fh.flush()

    _log("[backfill_xa] start")
    _log(f"  MBO dir : {MBO_DIR_NEW}")
    _log(f"  XA out  : {XA_OUT_DIR}")

    if not XA_PARQUET.exists():
        _log(f"[fatal] missing baseline parquet: {XA_PARQUET}")
        return 2

    existing = pd.read_parquet(XA_PARQUET)
    existing_dates = set(existing["date"].astype(int).tolist())
    _log(f"  existing rows: {len(existing)} dates")

    oot_dates = discover_oot_dates()
    missing = [d for d in oot_dates if d not in existing_dates]
    _log(f"  OOT total: {len(oot_dates)}  to backfill: {len(missing)}")
    if not missing:
        _log("[plan] nothing to do.")
        return 0
    _log(f"  missing dates: {missing}")

    econ_events = load_econ_calendar()
    labels = load_existing_labels()
    _log(f"  econ events loaded: {len(econ_events)}  labeled OOT days: {len(labels)}")

    # 1) Build ES early-session features per missing day
    es_rows: List[Dict] = []
    failures: List[Dict] = []
    for di in missing:
        try:
            feats = dcv1.build_day_features(di, econ_events)
        except Exception as e:
            _log(f"  [ES-feat ERR] {di}: {e}")
            failures.append({"date": di, "stage": "es_features", "error": str(e)})
            continue
        feats["date"] = di
        lab = labels.get(di)
        if lab is not None:
            feats["label_profitable"] = lab["label_profitable"]
            feats["day_mean_net_realized"] = lab["day_mean_net_realized"]
            feats["day_sum_net_realized"] = lab["day_sum_net_realized"]
            feats["n_filled"] = lab["n_filled"]
        else:
            # Unlabeled: sentinel -1 + NaN P&L. Downstream filter only reads
            # feature columns; label_profitable is for diagnostics only.
            feats["label_profitable"] = -1
            feats["day_mean_net_realized"] = float("nan")
            feats["day_sum_net_realized"] = float("nan")
            feats["n_filled"] = float("nan")
        es_rows.append(feats)
        _log(f"  [ES-feat ok] {di}  n_events_early={feats.get('n_events_early', 0):.0f}  labeled={lab is not None}")

    if not es_rows:
        _log("[fatal] no ES rows built — abort")
        return 3

    es_new = pd.DataFrame(es_rows)

    # 2) Fetch yfinance panel once, build XA features for missing days.
    panel, have_xa, msg = xacv1._try_fetch_yf()
    _log(f"  yfinance: have_xa={have_xa} ({msg}) panel shape={panel.shape}")
    xa_new = xacv1.build_cross_asset_features(
        [int(r["date"]) for r in es_rows], panel, have_xa
    )
    combined_new = es_new.merge(xa_new, on="date", how="left")
    _log(f"  built XA features for {len(combined_new)} new rows")

    # 3) Align columns to existing parquet, append, sort, dedupe (existing wins).
    missing_cols = [c for c in existing.columns if c not in combined_new.columns]
    extra_cols = [c for c in combined_new.columns if c not in existing.columns]
    if missing_cols:
        _log(f"  WARNING: new rows missing cols {missing_cols} — filling NaN")
        for c in missing_cols:
            combined_new[c] = np.nan
    if extra_cols:
        _log(f"  WARNING: new rows have extra cols {extra_cols} — dropping")
        combined_new = combined_new.drop(columns=extra_cols)
    combined_new = combined_new[existing.columns]

    # Convert label_profitable to a nullable Int64 / promote to allow -1 if needed.
    # The existing dtype is int64; -1 fits.
    appended = pd.concat([existing, combined_new], ignore_index=True)
    appended = appended.sort_values("date").reset_index(drop=True)
    # Drop any accidental duplicate dates, keep FIRST (i.e. existing).
    pre_n = len(appended)
    appended = appended.drop_duplicates(subset=["date"], keep="first").reset_index(drop=True)
    if len(appended) != pre_n:
        _log(f"  deduped {pre_n - len(appended)} duplicate-date rows (kept existing)")

    # 4) Write parquet (do NOT overwrite — write to temp + rename atomically).
    tmp_path = XA_PARQUET.with_suffix(".parquet.tmp")
    appended.to_parquet(tmp_path, index=False)
    os.replace(tmp_path, XA_PARQUET)
    _log(f"  wrote {XA_PARQUET}  rows={len(appended)}")

    # 5) Write backfill manifest (NOT overwriting REPORT.md or .regen_complete.json).
    manifest = {
        "completed_at": datetime.now().isoformat(),
        "runtime_seconds": time.time() - t0,
        "oot_total": len(oot_dates),
        "existing_rows_before": len(existing),
        "rows_appended": len(combined_new),
        "rows_after": len(appended),
        "have_yfinance": bool(have_xa),
        "yfinance_status": msg,
        "labeled_appended": int(sum(int(r.get("label_profitable", -1)) >= 0 for r in es_rows)),
        "unlabeled_appended": int(sum(int(r.get("label_profitable", -1)) < 0 for r in es_rows)),
        "appended_dates": [int(r["date"]) for r in es_rows],
        "failures": failures,
    }
    out_path = XA_OUT_DIR / f"backfill_manifest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as fh:
        json.dump(manifest, fh, indent=2)
    _log(f"  manifest: {out_path}")
    _log(f"[done] {(time.time()-t0):.1f}s")
    log_fh.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
