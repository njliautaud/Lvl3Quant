#!/usr/bin/env python3
"""
Smoketest the patched FIFO harness on 3 OOT v7 days.
Captures entry_time_ns, microprice_at_entry, microprice_dir (HC #491 R5 / HC #492 R3).
Date selection: green=20260413, red=20260403, flat=20260420.
"""
from __future__ import annotations

import sys
import time
import logging
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

LVL3_ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(LVL3_ROOT))

OUT_DIR = LVL3_ROOT / "output" / "v7_fifo_smoketest_2026-05-28"
OUT_DIR.mkdir(parents=True, exist_ok=True)

V7_PRED_NPZ = LVL3_ROOT / "output" / "meta_v7_prod" / "concat_oot_predictions.npz"
V2_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_bulk_oot_v2"
MBO_EVENT_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events_smart_v3"

H_SEC = 1.0
HOLD_S = 1.5
CANCEL_S = 1.0
TP_TICKS = 2.0
SL_TICKS = 1.0
PCT = 0.05

DATES = ["20260403", "20260413", "20260420"]  # red, green, flat

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s",
                    handlers=[logging.FileHandler(OUT_DIR / "run.log"), logging.StreamHandler()])
log = logging.getLogger("smoketest")


def load_v7_for_dates(target_dates):
    d = np.load(V7_PRED_NPZ, allow_pickle=False)
    preds = d["predictions"].astype(np.float32)
    dates = d["dates"].astype(str)
    perday = {}
    for date_str in target_dates:
        mask = dates == date_str
        v7_preds_d = preds[mask]
        if v7_preds_d.size == 0:
            log.warning(f"{date_str}: no v7 preds")
            continue
        v2_npz = V2_DIR / f"{date_str}_predictions.npz"
        if not v2_npz.exists():
            log.warning(f"{date_str}: no v2 NPZ")
            continue
        v2 = np.load(v2_npz, allow_pickle=False)
        ws = int(v2["window_size"])
        st = int(v2["stride"])
        nw = int(v2["n_windows"])
        if v7_preds_d.size > nw:
            log.warning(f"{date_str}: trim mismatch v7={v7_preds_d.size} v2={nw}, SKIP")
            continue
        perday[date_str] = {"preds": v7_preds_d, "window_size": ws, "stride": st}
        log.info(f"{date_str}: preds={v7_preds_d.size} ws={ws} st={st}")
    return perday


def select_topk(rec, side, pct):
    x = rec["preds"]
    if side == "long":
        mask = x > 0
        strength = x
    else:
        mask = x < 0
        strength = -x
    side_s = strength[mask]
    if side_s.size == 0:
        return None
    k = max(1, int(side_s.size * pct))
    thresh = np.partition(side_s, -k)[-k]
    selected = mask & (strength >= thresh)
    idx = np.where(selected)[0]
    if idx.size == 0:
        return None
    return idx, strength[idx]


def map_idx_to_ts(date_str, idx_in_day, window_size, stride):
    mbo_path = MBO_EVENT_DIR / f"{date_str}_mbo_events.npz"
    if not mbo_path.exists():
        return None
    mbo = np.load(mbo_path, allow_pickle=False)
    ts_events = mbo["timestamps"].astype(np.int64)
    n_events = len(ts_events)
    event_idx = np.minimum(idx_in_day * stride + window_size - 1, n_events - 1)
    return ts_events[event_idx]


def run_one(date_str, idx_in_day, direction, strength, window_size, stride):
    from alpha_discovery.deep_models.fifo_market_replay import FIFOReplayEngine

    ts_ns = map_idx_to_ts(date_str, idx_in_day, window_size, stride)
    if ts_ns is None:
        return [{"date": date_str, "error": "missing_mbo_events"}]

    signals = [{"ts_ns": int(t), "direction": direction, "strength": float(strength[i])}
               for i, t in enumerate(ts_ns)]
    if not signals:
        return []

    try:
        engine = FIFOReplayEngine(
            date=date_str,
            cancel_after_ns=int(CANCEL_S * 1_000_000_000),
            max_hold_ns=int(HOLD_S * 1_000_000_000),
        )
    except Exception as e:
        return [{"date": date_str, "direction": direction, "error": f"engine_init: {e}"}]

    try:
        trades = engine.simulate(signals=signals, tp_ticks=TP_TICKS, sl_ticks=SL_TICKS, order_type="limit")
    except Exception as e:
        return [{"date": date_str, "direction": direction, "error": f"simulate: {e}"}]

    rows = []
    for t in trades:
        hold_s = (t.exit_ts_ns - t.entry_ts_ns) / 1e9 if (t.entry_ts_ns and t.exit_ts_ns) else 0.0
        rows.append({
            "date": date_str,
            "direction": t.direction,
            "hold_s": hold_s,
            "fill_type": t.exit_reason,
            "net_ticks": float(t.pnl_ticks_net),
            "net_dollars": float(t.pnl_dollars),
            "queue_ahead": int(t.queue_ahead),
            "queue_wait_ns": int(t.queue_wait_ns),
            "slippage_ticks": float(t.slippage_ticks),
            "pred_strength": float(t.pred_strength),
            # New patched columns
            "entry_time_ns": int(getattr(t, "entry_time_ns", 0) or 0),
            "microprice_at_entry": float(getattr(t, "microprice_at_entry", 0.0) or 0.0),
            "microprice_dir": int(getattr(t, "microprice_dir", 0) or 0),
        })
    return rows


def main():
    t0 = time.time()
    perday = load_v7_for_dates(DATES)
    if not perday:
        log.error("No dates aligned, abort.")
        sys.exit(1)

    jobs = []
    for date_str, rec in perday.items():
        for side in ["short", "long"]:
            sel = select_topk(rec, side, PCT)
            if sel is None:
                continue
            idx, strength = sel
            jobs.append((date_str, idx, side, strength, rec["window_size"], rec["stride"]))
    log.info(f"Total jobs: {len(jobs)}")

    all_rows = []
    ctx = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=6, mp_context=ctx) as ex:
        futures = {ex.submit(run_one, *j): (j[0], j[2]) for j in jobs}
        for fut in as_completed(futures):
            d, side = futures[fut]
            try:
                rows = fut.result()
            except Exception as e:
                log.error(f"{d}/{side}: crash {e}")
                continue
            all_rows.extend(rows)
            log.info(f"{d}/{side}: {len(rows)} rows ({(time.time()-t0):.0f}s elapsed)")

    if not all_rows:
        log.error("No rows produced.")
        sys.exit(1)

    df = pd.DataFrame(all_rows)
    if "error" in df.columns:
        errs = df[df["error"].notna()]
        if len(errs):
            log.warning(f"Errors: {len(errs)} - {errs[['date','error']].to_dict('records')}")
        df = df[df["error"].isna()].drop(columns=["error"])

    df.to_parquet(OUT_DIR / "fills.parquet")
    log.info(f"Wrote fills.parquet: {len(df):,} rows in {(time.time()-t0):.0f}s")


if __name__ == "__main__":
    main()
