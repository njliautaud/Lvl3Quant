#!/usr/bin/env python3
"""HC #451 R2 — Event-salience tag precompute.

Two boolean tags per MBO event, aligned to source event index:
  - sweep_tag       : aggressive multi-level sweeps in same direction within K_MS
  - large_print_tag : single trade size > p99 of last 60s rolling trade-size dist

For non-trade events both tags are False (sweep/large-print only defined on T rows).
We still emit one row per source event so downstream training datasets can join
on event index without re-aligning.

Output schema (per date):
  parquet columns: [ts_ns:i64, sweep_tag:bool, large_print_tag:bool]
  one row per source MBO event (full file, all instruments)

Filter to ES front-month outright (max-trade-count instrument_id) for tag
computation, matching hc439_build_mid_price_cache.py convention. Tags are
False on non-front-month events.

Run:
  python precompute_salience_tags.py --date 20260429 --output-dir /home/jupiter/Lvl3Quant/data/derived/hc451_salience_tags
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import polars as pl
import databento as db


RAW_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
TICK_RAW = 250_000_000  # 0.25 pt = 1 tick in databento fixed-point /1e9

A_TRADE = ord('T')

# Defaults from HC #451 R2 sweep_tag spec
DEFAULT_N = 5      # >= N aggressive orders
DEFAULT_M = 2      # crossing >= M price levels
DEFAULT_K_MS = 200 # within K ms

LARGE_PRINT_LOOKBACK_NS = 60 * 1_000_000_000  # 60s
LARGE_PRINT_QUANTILE = 0.99


def _pick_front_month(arr: np.ndarray) -> int:
    """Return instrument_id with the most ES outright trade prints."""
    actions = arr['action'].view(np.uint8)
    t_mask = actions == A_TRADE
    px = arr['price'][t_mask].astype(np.int64)
    iid = arr['instrument_id'][t_mask]
    es_mask = (px > 5_000_000_000_000) & (px < 8_000_000_000_000)
    iid = iid[es_mask]
    if len(iid) == 0:
        raise RuntimeError("no ES trades in file")
    uniq, cnt = np.unique(iid, return_counts=True)
    return int(uniq[np.argmax(cnt)])


def _compute_sweep_tag(trade_ts: np.ndarray,
                       trade_px: np.ndarray,
                       trade_side: np.ndarray,
                       n_min: int, m_min: int, k_ms: int) -> np.ndarray:
    """For each trade, look back K_MS. If >= N trades same side and >= M distinct
    price levels are seen in window → tag=True on that trade.

    trade_side: uint8 byte ('A' or 'B'). 'B' side means trade hit the bid order
    (so aggressor was a SELLER); 'A' means trade hit ask order (aggressor BUYER).
    The "direction" of a sweep is the aggressor direction. We tag separately per
    side and OR them.
    """
    k_ns = np.int64(k_ms) * 1_000_000
    n_tr = len(trade_ts)
    out = np.zeros(n_tr, dtype=bool)
    # Two-pointer: left advances as ts[left] < ts[i] - k_ns
    left = 0
    for i in range(n_tr):
        lim = trade_ts[i] - k_ns
        while left < i and trade_ts[left] < lim:
            left += 1
        # Window [left .. i] inclusive
        side_i = trade_side[i]
        # Count same-side trades and distinct levels in window
        # NB: small windows (<=K_MS=200ms) typically have <50 trades, brute force is fine
        cnt_same = 0
        # Track distinct prices via small set (max ~32 distinct levels in 200ms)
        seen_lo = np.int64(np.iinfo(np.int64).max)
        seen_hi = np.int64(np.iinfo(np.int64).min)
        # Use a tiny stack of distinct prices for set membership
        distinct = []
        for j in range(left, i + 1):
            if trade_side[j] != side_i:
                continue
            cnt_same += 1
            p = trade_px[j]
            if p not in distinct:
                distinct.append(p)
        if cnt_same >= n_min and len(distinct) >= m_min:
            out[i] = True
    return out


# Numba JIT version for speed (~50x faster than pure python loop)
try:
    from numba import njit

    @njit(cache=True, fastmath=False)
    def _compute_sweep_tag_nb(trade_ts, trade_px, trade_side, n_min, m_min, k_ns):
        n_tr = trade_ts.shape[0]
        out = np.zeros(n_tr, dtype=np.bool_)
        # distinct prices buffer (max 64 levels)
        buf = np.empty(64, dtype=np.int64)
        left = 0
        for i in range(n_tr):
            lim = trade_ts[i] - k_ns
            while left < i and trade_ts[left] < lim:
                left += 1
            side_i = trade_side[i]
            cnt_same = 0
            nbuf = 0
            for j in range(left, i + 1):
                if trade_side[j] != side_i:
                    continue
                cnt_same += 1
                p = trade_px[j]
                found = False
                for k in range(nbuf):
                    if buf[k] == p:
                        found = True
                        break
                if not found:
                    if nbuf < 64:
                        buf[nbuf] = p
                        nbuf += 1
            if cnt_same >= n_min and nbuf >= m_min:
                out[i] = True
        return out

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


def _compute_large_print_tag(trade_ts: np.ndarray,
                             trade_size: np.ndarray,
                             lookback_ns: int,
                             q: float) -> np.ndarray:
    """For each trade, tag=True if size > p99 of trade sizes in [t-lookback, t)."""
    n_tr = len(trade_ts)
    out = np.zeros(n_tr, dtype=bool)
    # Two-pointer window
    left = 0
    for i in range(n_tr):
        lim = trade_ts[i] - lookback_ns
        while left < i and trade_ts[left] < lim:
            left += 1
        if i - left < 50:
            # not enough history for stable p99 (require 50 prints)
            continue
        # p99 of trade_size[left:i]  (exclude current)
        thr = np.quantile(trade_size[left:i], q)
        if trade_size[i] > thr:
            out[i] = True
    return out


try:
    from numba import njit as _njit2

    @_njit2(cache=True, fastmath=False)
    def _compute_large_print_tag_nb(trade_ts, trade_size, lookback_ns, q):
        n_tr = trade_ts.shape[0]
        out = np.zeros(n_tr, dtype=np.bool_)
        # We'll compute p99 via partial sort of a copy each step. For speed we use
        # a histogram approach: trade sizes are integers, usually <500. Bucket count.
        # Maintain rolling histogram of trade sizes in window.
        MAX_SIZE = 5000
        hist = np.zeros(MAX_SIZE + 1, dtype=np.int64)
        n_in_window = 0
        left = 0
        for i in range(n_tr):
            lim = trade_ts[i] - lookback_ns
            while left < i and trade_ts[left] < lim:
                s = trade_size[left]
                if s <= MAX_SIZE:
                    hist[s] -= 1
                else:
                    hist[MAX_SIZE] -= 1
                n_in_window -= 1
                left += 1
            if n_in_window >= 50:
                # find threshold value v such that cumulative >= q*N from low side
                target = q * n_in_window
                cum = 0
                thr = 0
                for v in range(MAX_SIZE + 1):
                    cum += hist[v]
                    if cum >= target:
                        thr = v
                        break
                if trade_size[i] > thr:
                    out[i] = True
            # add current to window (will be looked-back for next i)
            s = trade_size[i]
            if s <= MAX_SIZE:
                hist[s] += 1
            else:
                hist[MAX_SIZE] += 1
            n_in_window += 1
        return out

except ImportError:
    pass


def process_date(date_str: str, output_dir: Path,
                 n_min=DEFAULT_N, m_min=DEFAULT_M, k_ms=DEFAULT_K_MS,
                 force=False) -> dict:
    out_path = output_dir / f"{date_str}_salience.parquet"
    stats_path = output_dir / f"{date_str}_stats.json"
    if out_path.exists() and not force:
        return {"date": date_str, "status": "skipped_exists"}

    dbn_path = RAW_DIR / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    if not dbn_path.exists():
        return {"date": date_str, "status": "missing_dbn"}

    t0 = time.time()
    store = db.DBNStore.from_file(str(dbn_path))
    arr = store.to_ndarray()
    t_load = time.time() - t0

    n_events = len(arr)
    actions = arr['action'].view(np.uint8)
    sides = arr['side'].view(np.uint8)
    iids = arr['instrument_id']
    ts_event = arr['ts_event'].astype(np.int64)
    prices = arr['price'].astype(np.int64)
    sizes = arr['size'].astype(np.int64)

    front_id = _pick_front_month(arr)

    # Trade events on front month only
    t_mask = (actions == A_TRADE) & (iids == front_id)
    tr_idx = np.flatnonzero(t_mask)
    tr_ts = ts_event[tr_idx]
    tr_px = prices[tr_idx]
    tr_side = sides[tr_idx]
    tr_size = sizes[tr_idx]

    # ensure trades are sorted by ts (DBN files are time-ordered)
    if not np.all(np.diff(tr_ts) >= 0):
        order = np.argsort(tr_ts, kind='stable')
        tr_idx = tr_idx[order]
        tr_ts = tr_ts[order]
        tr_px = tr_px[order]
        tr_side = tr_side[order]
        tr_size = tr_size[order]

    t1 = time.time()
    k_ns = np.int64(k_ms) * 1_000_000
    if HAS_NUMBA:
        sweep_tr = _compute_sweep_tag_nb(tr_ts, tr_px, tr_side,
                                         np.int64(n_min), np.int64(m_min), k_ns)
        large_tr = _compute_large_print_tag_nb(tr_ts, tr_size,
                                               np.int64(LARGE_PRINT_LOOKBACK_NS),
                                               np.float64(LARGE_PRINT_QUANTILE))
    else:
        sweep_tr = _compute_sweep_tag(tr_ts, tr_px, tr_side, n_min, m_min, k_ms)
        large_tr = _compute_large_print_tag(tr_ts, tr_size,
                                            LARGE_PRINT_LOOKBACK_NS,
                                            LARGE_PRINT_QUANTILE)
    t_tags = time.time() - t1

    # Scatter back to full event index
    sweep_full = np.zeros(n_events, dtype=bool)
    large_full = np.zeros(n_events, dtype=bool)
    sweep_full[tr_idx] = sweep_tr
    large_full[tr_idx] = large_tr

    # Write parquet
    t2 = time.time()
    df = pl.DataFrame({
        "ts_ns": ts_event,
        "sweep_tag": sweep_full,
        "large_print_tag": large_full,
    })
    df.write_parquet(out_path, compression="zstd")
    t_write = time.time() - t2

    elapsed = time.time() - t0
    stats = {
        "date": date_str,
        "status": "ok",
        "front_instrument_id": front_id,
        "n_events": int(n_events),
        "n_trades_front": int(len(tr_idx)),
        "sweep_tag_count": int(sweep_full.sum()),
        "large_print_tag_count": int(large_full.sum()),
        "sweep_tag_rate_per_trade": round(float(sweep_tr.mean()), 4),
        "large_print_tag_rate_per_trade": round(float(large_tr.mean()), 4),
        "params": {"N": n_min, "M": m_min, "K_ms": k_ms,
                   "large_lookback_s": 60, "large_q": LARGE_PRINT_QUANTILE},
        "elapsed_total_s": round(elapsed, 2),
        "elapsed_load_s": round(t_load, 2),
        "elapsed_tags_s": round(t_tags, 2),
        "elapsed_write_s": round(t_write, 2),
        "used_numba": HAS_NUMBA,
    }
    stats_path.write_text(json.dumps(stats, indent=2))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True, help="YYYYMMDD")
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--n-min", type=int, default=DEFAULT_N)
    ap.add_argument("--m-min", type=int, default=DEFAULT_M)
    ap.add_argument("--k-ms", type=int, default=DEFAULT_K_MS)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    stats = process_date(args.date, out_dir,
                         n_min=args.n_min, m_min=args.m_min, k_ms=args.k_ms,
                         force=args.force)
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
