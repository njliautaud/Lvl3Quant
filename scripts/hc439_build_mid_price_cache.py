#!/usr/bin/env python3
"""HC #439 — Build per-day cached trade-print arrays (ts_ns, price_raw)
from raw DBN files for fast MFE/MAE walks.

Produces: /home/jupiter/Lvl3Quant/data/derived/mid_price_cache_hc439/<DATE>_trades.npz
Keys: ts_ns (int64), price_raw (int64, databento fixed-point /1e9 = points)

Uses to_ndarray() for ~2.4s/day vs 60s for to_df. Trade prints (action='T')
only — using last trade price as the mid proxy (matches existing
raw_trajectory_mfe_mae.py convention).
"""
import argparse
import sys
import time
from pathlib import Path
import numpy as np

import databento as db

RAW_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
CACHE_DIR = Path("/home/jupiter/Lvl3Quant/data/derived/mid_price_cache_hc439")
CACHE_DIR.mkdir(parents=True, exist_ok=True)

A_TRADE = ord('T')  # action byte for Trade


def build_one(date_str: str, force: bool = False) -> dict:
    """Build cache for one date. Returns status dict.

    Filters to ES front-month outright by selecting the single instrument_id with
    the most trade prints (heuristic — verified valid on 20260224 = ESH6 with
    430k trades vs 1.2k for ESM6 back month and 230 for spreads).
    """
    out_path = CACHE_DIR / f"{date_str}_trades.npz"
    if out_path.exists() and not force:
        return {"date": date_str, "status": "skipped_exists"}
    dbn_path = RAW_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    if not dbn_path.exists():
        return {"date": date_str, "status": "missing_dbn"}
    t0 = time.time()
    store = db.DBNStore.from_file(str(dbn_path))
    arr = store.to_ndarray()
    actions = arr['action'].view(np.uint8)
    trades_mask = actions == A_TRADE
    tr_ids = arr['instrument_id'][trades_mask]
    tr_ts = arr['ts_event'][trades_mask].astype(np.int64)
    tr_px = arr['price'][trades_mask].astype(np.int64)
    # Find front-month: dominant instrument_id among ES outright trades (price > 5000 pts)
    es_mask = (tr_px > 5_000_000_000_000) & (tr_px < 8_000_000_000_000)
    es_ids = tr_ids[es_mask]
    if len(es_ids) == 0:
        return {"date": date_str, "status": "no_es_trades"}
    uniq, cnt = np.unique(es_ids, return_counts=True)
    front_id = uniq[np.argmax(cnt)]
    sel = (tr_ids == front_id)
    ts = tr_ts[sel]
    px = tr_px[sel]
    order = np.argsort(ts)
    ts = ts[order]
    px = px[order]
    np.savez(out_path,
             ts_ns=ts, price_raw=px,
             instrument_id=np.int64(front_id))
    elapsed = time.time() - t0
    return {"date": date_str, "status": "ok",
            "front_id": int(front_id),
            "n_trades": int(len(ts)),
            "elapsed": round(elapsed, 1),
            "px_min_pts": float(px.min() / 1e9),
            "px_max_pts": float(px.max() / 1e9)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dates", nargs="+", help="YYYYMMDD list, default = all OOT v2 days")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    if args.dates:
        dates = args.dates
    else:
        import glob
        oot_npz = sorted(glob.glob(
            "/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_bulk_oot_v2/*.npz"))
        dates = []
        for f in oot_npz:
            d = np.load(f, allow_pickle=False)
            dates.append(str(d['date']))
    print(f"Building cache for {len(dates)} dates with {args.workers} workers")

    if args.workers == 1:
        for dstr in dates:
            r = build_one(dstr, force=args.force)
            print(r, flush=True)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(build_one, d, args.force): d for d in dates}
            for fut in as_completed(futs):
                try:
                    print(fut.result(), flush=True)
                except Exception as e:
                    print({"date": futs[fut], "error": str(e)}, flush=True)


if __name__ == "__main__":
    main()
