"""
Volume Profile Feature Engineering — Jupiter/Saturn CPU compute.

Reads feat15 NPZ files (or raw mbo_events NPZ) and appends 7 volume profile features.
Output: mbo_events_vp/ directory with (N, 22) feature arrays.

New features (indices 15-21):
  15: session_poc_dist      — price_rel_ticks minus session POC price (signed ticks)
  16: session_va_pos        — normalized position within session value area [0=VAL, 1=VAH]
  17: session_above_poc_vol — fraction of cumulative session volume above POC
  18: rolling_poc_dist      — rolling 2000-event window POC distance
  19: rolling_va_pos        — rolling 2000-event window value area position
  20: vol_at_price_ratio    — volume at current price tick / session POC volume (density ratio)
  21: time_of_day_norm      — normalized time within trading session [0,1]

Usage:
  python compute_vp_features.py [--src mbo_events_feat15] [--dst mbo_events_vp] [--workers 8]

Run on Jupiter: 64GB RAM, 209 files, all CPU.
"""

import os, sys, argparse, time, logging
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger(__name__)

# ── Column indices in source array ─────────────────────────────────────────────
COL_TIME     = 0   # time_delta_log (cumulative log-seconds from session start)
COL_ETYPE    = 1   # event_type_id
COL_SIDE     = 2   # side_id (1=bid, 2=ask)
COL_PRICE    = 3   # price_rel_ticks (float, signed relative ticks from session open)
COL_QTY      = 4   # qty_log
COL_SPREAD   = 5   # spread_ticks

# Value area threshold
VALUE_AREA_PCT = 0.70
ROLLING_WINDOW = 2000      # events for rolling VP
PRICE_BUCKET   = 0.5       # tick resolution for VP binning (0.5 = half-tick buckets)


def compute_vp_for_file(src_path: Path, dst_path: Path) -> dict:
    """Compute VP features for a single NPZ file. Returns stats dict."""
    t0 = time.time()

    data = np.load(src_path, allow_pickle=False)
    ev = data["events"].astype(np.float32)   # (N, 15+)
    N = len(ev)

    price  = ev[:, COL_PRICE].astype(np.float64)   # price_rel_ticks
    qty    = np.exp(np.clip(ev[:, COL_QTY].astype(np.float64), -10, 10))  # back to linear qty
    time_d = ev[:, COL_TIME].astype(np.float64)     # time_delta_log

    # ── Price binning ──────────────────────────────────────────────────────────
    # Bucket prices to half-tick resolution for VP
    price_bucket = np.round(price / PRICE_BUCKET).astype(np.int32)
    p_min, p_max = price_bucket.min(), price_bucket.max()
    n_buckets = p_max - p_min + 1

    # ── Session cumulative volume profile ──────────────────────────────────────
    # session_vol[i] = total qty at price bucket i (cumulative from start of file)
    session_vol = np.zeros(n_buckets, dtype=np.float64)
    for i in range(N):
        b = price_bucket[i] - p_min
        session_vol[b] += qty[i]
    total_session_vol = session_vol.sum() + 1e-8

    # Session POC
    poc_bucket = np.argmax(session_vol)
    poc_price  = (poc_bucket + p_min) * PRICE_BUCKET

    # Session Value Area (70% of volume)
    sorted_idx = np.argsort(-session_vol)  # descending by volume
    cumvol = 0.0
    va_buckets = set()
    for idx in sorted_idx:
        cumvol += session_vol[idx]
        va_buckets.add(idx)
        if cumvol >= VALUE_AREA_PCT * total_session_vol:
            break
    va_lo = min(va_buckets) + p_min
    va_hi = max(va_buckets) + p_min
    vah_price = va_hi * PRICE_BUCKET
    val_price = va_lo * PRICE_BUCKET
    va_range  = max(vah_price - val_price, 1e-4)

    # Volume above POC
    above_poc_vol = session_vol[poc_bucket + 1:].sum() if poc_bucket + 1 < n_buckets else 0.0
    session_above_poc_ratio = above_poc_vol / total_session_vol

    # POC volume (for density ratio)
    poc_vol = session_vol[poc_bucket] + 1e-8

    # ── Per-event session VP features ──────────────────────────────────────────
    feat15 = np.zeros((N, 1), dtype=np.float32)   # session_poc_dist
    feat16 = np.zeros((N, 1), dtype=np.float32)   # session_va_pos
    feat17 = np.full((N, 1), session_above_poc_ratio, dtype=np.float32)
    feat20 = np.zeros((N, 1), dtype=np.float32)   # vol_at_price_ratio

    feat15[:, 0] = (price - poc_price).astype(np.float32)
    feat16[:, 0] = ((price - val_price) / va_range).astype(np.float32)

    # Volume at each event's price bucket vs POC volume
    vol_at_event = np.array([session_vol[max(0, min(n_buckets-1, price_bucket[i] - p_min))]
                              for i in range(N)], dtype=np.float32)
    feat20[:, 0] = vol_at_event / poc_vol

    # ── Rolling VP features (2000-event window) ────────────────────────────────
    # Optimized: recompute POC/VA every RECOMPUTE_INTERVAL events (not every event)
    RECOMPUTE_INTERVAL = 200
    rolling_vol = np.zeros(n_buckets, dtype=np.float64)
    feat18 = np.zeros(N, dtype=np.float32)   # rolling_poc_dist
    feat19 = np.zeros(N, dtype=np.float32)   # rolling_va_pos

    r_poc_p = poc_price; r_val_p = val_price; r_vah_p = vah_price  # init from session

    for i in range(N):
        b_new = price_bucket[i] - p_min
        rolling_vol[b_new] += qty[i]
        if i >= ROLLING_WINDOW:
            b_old = price_bucket[i - ROLLING_WINDOW] - p_min
            rolling_vol[b_old] = max(0.0, rolling_vol[b_old] - qty[i - ROLLING_WINDOW])

        # Recompute POC/VA every RECOMPUTE_INTERVAL events
        if i % RECOMPUTE_INTERVAL == 0:
            r_total = rolling_vol.sum() + 1e-8
            r_poc_b = int(np.argmax(rolling_vol))
            r_poc_p = (r_poc_b + p_min) * PRICE_BUCKET
            # Value area via cumulative sort
            r_sorted = np.argsort(-rolling_vol)
            r_cumvol = 0.0; r_va_lo = r_poc_b; r_va_hi = r_poc_b
            for idx in r_sorted:
                r_cumvol += rolling_vol[idx]
                r_va_lo = min(r_va_lo, idx); r_va_hi = max(r_va_hi, idx)
                if r_cumvol >= VALUE_AREA_PCT * r_total:
                    break
            r_vah_p = (r_va_hi + p_min) * PRICE_BUCKET
            r_val_p = (r_va_lo + p_min) * PRICE_BUCKET

        feat18[i] = price[i] - r_poc_p
        feat19[i] = (price[i] - r_val_p) / max(r_vah_p - r_val_p, 1e-4)

    # ── Time-of-day normalization ──────────────────────────────────────────────
    # time_delta_log is log(1+seconds). Recover approximate absolute time offset.
    raw_time = np.expm1(np.clip(time_d, 0, 20))
    session_len = max(raw_time[-1] - raw_time[0], 1.0)
    feat21 = ((raw_time - raw_time[0]) / session_len).astype(np.float32)

    # ── Concatenate and save ───────────────────────────────────────────────────
    vp_features = np.stack([
        feat15[:, 0],   # 15: session_poc_dist
        feat16[:, 0],   # 16: session_va_pos
        feat17[:, 0],   # 17: session_above_poc_vol
        feat18,         # 18: rolling_poc_dist
        feat19,         # 19: rolling_va_pos
        feat20[:, 0],   # 20: vol_at_price_ratio
        feat21,         # 21: time_of_day_norm
    ], axis=1)  # (N, 7)

    out = np.concatenate([ev, vp_features], axis=1)  # (N, 22)

    dst_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dst_path, events=out)

    elapsed = time.time() - t0
    return {"file": src_path.name, "n_events": N, "poc_price": poc_price,
            "vah": vah_price, "val": val_price, "elapsed": elapsed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src",     default="mbo_events_feat15")
    ap.add_argument("--dst",     default="mbo_events_vp")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit",   type=int, default=0, help="process only first N files (debug)")
    args = ap.parse_args()

    data_root = Path(os.environ.get("DATA_ROOT", "/home/jupiter/Lvl3Quant/data/processed"))
    src_dir   = data_root / args.src
    dst_dir   = data_root / args.dst
    dst_dir.mkdir(parents=True, exist_ok=True)

    files = sorted(src_dir.glob("*.npz"))
    if args.limit:
        files = files[:args.limit]

    # Skip already-done files (resume-aware)
    todo = [f for f in files if not (dst_dir / f.name).exists()]
    log.info(f"Volume profile precompute: {len(todo)}/{len(files)} files to process")
    log.info(f"src={src_dir}  dst={dst_dir}  workers={args.workers}")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(compute_vp_for_file, f, dst_dir / f.name): f for f in todo}
        for fut in as_completed(futures):
            try:
                r = fut.result()
                done += 1
                log.info(f"[{done}/{len(todo)}] {r['file']} | {r['n_events']:,} events | "
                         f"POC={r['poc_price']:.1f} VAH={r['vah']:.1f} VAL={r['val']:.1f} | "
                         f"{r['elapsed']:.1f}s")
            except Exception as e:
                log.error(f"FAILED {futures[fut].name}: {e}")

    log.info(f"Done. {done}/{len(todo)} files written to {dst_dir}")


if __name__ == "__main__":
    main()
