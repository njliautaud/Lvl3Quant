#!/usr/bin/env python3
"""
HC #271(A) — Per-OOT-date regime/trend labeler.

For each OOT date, scan the raw MBO dbn.zst file for trade events on the
most-active ES instrument and emit:
    - session_open_price (first trade after RTH open 13:30 UTC)
    - session_close_price (last trade before RTH close 20:00 UTC)
    - rth_high, rth_low
    - close_minus_open_ticks (signed)
    - rth_range_ticks  (high-low)
    - rth_realized_vol_ticks  (stdev of 1-min mid changes, RTH only)
    - n_trades_rth
    - trend_label  : 'up' / 'down' / 'flat'  (cutoff ±10 ticks default)
    - vol_bucket   : 'low' / 'med' / 'high'  (computed in a 2nd pass after all dates)

Output: output/regime_labels/oot_dates_regime.parquet

Usage:
    python3 scripts/regime_label_oot_dates.py
    python3 scripts/regime_label_oot_dates.py --dates 20260406 20260407 ...
"""
from __future__ import annotations
import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
RAW_MBO_DIR = LVL3_ROOT / "data" / "raw" / "mbo"
OUT_DIR = LVL3_ROOT / "output" / "regime_labels"
ES_TICK_SIZE = 0.25  # points per tick

# RTH window in UTC (US Eastern 09:30-16:00 ET → 13:30-20:00 UTC during DST)
# We use a slightly-permissive window to handle DST/non-DST shifts.
RTH_START_UTC_HR = 13.5  # 13:30 UTC
RTH_END_UTC_HR   = 20.0  # 20:00 UTC

DEFAULT_OOT_DATES_FILE = LVL3_ROOT / "scripts" / "_oot_dates_56.json"

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("regime")


def load_oot_dates_default() -> list[str]:
    """Default OOT date list = all dates with a meta_lgbm_labels parquet OR the 14 tp8sl5 dates."""
    label_dir = LVL3_ROOT / "output" / "meta_lgbm_labels"
    if label_dir.exists():
        dates = sorted({p.name.split("_")[0] for p in label_dir.glob("2026*_signals_labeled.parquet")})
        if len(dates) >= 5:
            return dates
    # Fallback: 14 tp8sl5 dates
    return ["20260406", "20260407", "20260408", "20260409", "20260410",
            "20260412", "20260413", "20260414", "20260415", "20260416",
            "20260417", "20260419", "20260420", "20260426", "20260427"]


def label_one_date(date_str: str) -> dict | None:
    """Scan MBO for one date and return regime stats."""
    import databento as db
    p = RAW_MBO_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    if not p.exists():
        log.warning(f"[{date_str}] missing {p.name}")
        return None
    t0 = time.time()
    store = db.DBNStore.from_file(str(p))
    df = store.to_df()
    if "action" not in df.columns:
        log.warning(f"[{date_str}] no action col")
        return None
    trades = df[df["action"] == "T"].copy()
    if len(trades) == 0:
        log.warning(f"[{date_str}] no trade events")
        return None
    # Pick most-active instrument
    inst_counts = trades["instrument_id"].value_counts()
    main_inst = int(inst_counts.idxmax())
    trades = trades[trades["instrument_id"] == main_inst].copy()
    # databento.to_df() already returns prices in points (not nano-fixed). No scaling needed.
    trades["price_pts"] = trades["price"].astype(float)
    # Sanity: ES prices should be ~4000-9000 in 2026 era; if everything is way off, skip
    pmed = trades["price_pts"].median()
    if not (3000 < pmed < 10000):
        log.warning(f"[{date_str}] suspicious median price {pmed:.2f} — skipping")
        return None

    # Build hour-of-day from ts_event UTC
    if "ts_event" in trades.columns:
        ts = pd.to_datetime(trades["ts_event"], utc=True)
    else:
        ts = pd.to_datetime(trades.index, utc=True)
    hod = ts.dt.hour + ts.dt.minute / 60.0 + ts.dt.second / 3600.0
    rth_mask = (hod >= RTH_START_UTC_HR) & (hod < RTH_END_UTC_HR)
    rth = trades[rth_mask.values]
    if len(rth) < 100:
        log.warning(f"[{date_str}] only {len(rth)} RTH trades — partial session?")
        rth = trades  # fall back to whole session

    open_pts = float(rth["price_pts"].iloc[0])
    close_pts = float(rth["price_pts"].iloc[-1])
    high_pts = float(rth["price_pts"].max())
    low_pts = float(rth["price_pts"].min())
    n_trades_rth = int(len(rth))

    # 1-min realized vol: bucket by minute, take last price per minute, diff
    rth_ts = pd.to_datetime(rth["ts_event"] if "ts_event" in rth.columns else rth.index, utc=True)
    rth = rth.assign(_min_bucket=rth_ts.dt.floor("min").values)
    per_min = rth.groupby("_min_bucket")["price_pts"].last()
    diffs = per_min.diff().dropna()
    if len(diffs) > 1:
        rv_pts = float(diffs.std())
    else:
        rv_pts = 0.0

    close_minus_open_ticks = (close_pts - open_pts) / ES_TICK_SIZE
    range_ticks = (high_pts - low_pts) / ES_TICK_SIZE
    rv_ticks = rv_pts / ES_TICK_SIZE

    # Trend label (will be re-binned in 2nd pass; ±10 ticks default)
    if close_minus_open_ticks > 10:
        trend = "up"
    elif close_minus_open_ticks < -10:
        trend = "down"
    else:
        trend = "flat"

    elapsed = time.time() - t0
    log.info(f"[{date_str}] inst={main_inst} O={open_pts:.2f} C={close_pts:.2f} "
             f"net={close_minus_open_ticks:+.1f}t range={range_ticks:.1f}t rv={rv_ticks:.2f}t "
             f"trades_rth={n_trades_rth} trend={trend} ({elapsed:.1f}s)")

    return {
        "date": date_str,
        "instrument_id": main_inst,
        "open_pts": open_pts,
        "close_pts": close_pts,
        "high_pts": high_pts,
        "low_pts": low_pts,
        "close_minus_open_ticks": close_minus_open_ticks,
        "range_ticks": range_ticks,
        "rv_1min_ticks": rv_ticks,
        "n_trades_rth": n_trades_rth,
        "trend_label": trend,
    }


def assign_vol_bucket(df: pd.DataFrame) -> pd.DataFrame:
    """Assign low/med/high vol bucket based on rv_1min_ticks tertiles."""
    if len(df) < 3:
        df["vol_bucket"] = "med"
        return df
    qs = df["rv_1min_ticks"].quantile([1/3, 2/3]).values
    def _b(v):
        if v < qs[0]:
            return "low"
        if v < qs[1]:
            return "med"
        return "high"
    df["vol_bucket"] = df["rv_1min_ticks"].apply(_b)
    return df


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="*", default=None,
                    help="Specific dates YYYYMMDD; default = all OOT dates")
    ap.add_argument("--out", default=str(OUT_DIR / "oot_dates_regime.parquet"))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    dates = args.dates or load_oot_dates_default()
    log.info(f"Labeling {len(dates)} dates: {dates[:5]}{'...' if len(dates) > 5 else ''}")

    # Parallelize across dates (each ~50s loading time on its own process)
    from multiprocessing import Pool
    rows = []
    # HC #434 — use available CPU overhead. Default 12 workers on 16-core Jupiter
    # (leaves 4 cores for system/SSH/monitoring). Override with REGIME_LABEL_WORKERS env var.
    n_workers = min(int(os.environ.get("REGIME_LABEL_WORKERS", "12")), len(dates))
    log.info(f"using {n_workers} parallel workers")
    with Pool(n_workers) as pool:
        for r in pool.imap_unordered(label_one_date, dates):
            if r is not None:
                rows.append(r)

    if not rows:
        log.error("No regime labels produced.")
        return 1

    df = pd.DataFrame(rows)
    df = assign_vol_bucket(df)
    df = df.sort_values("date").reset_index(drop=True)
    out_path = Path(args.out)
    df.to_parquet(out_path, index=False)
    log.info(f"WROTE {len(df)} rows → {out_path}")

    # Headline summary
    cnt = df["trend_label"].value_counts().to_dict()
    log.info(f"trend distribution: {cnt}")
    cnt_v = df["vol_bucket"].value_counts().to_dict()
    log.info(f"vol distribution: {cnt_v}")
    log.info(f"% red days: {100.0 * cnt.get('down', 0) / len(df):.1f}% "
             f"| % green: {100.0 * cnt.get('up', 0) / len(df):.1f}% "
             f"| % flat: {100.0 * cnt.get('flat', 0) / len(df):.1f}%")
    return 0


if __name__ == "__main__":
    sys.exit(main())
