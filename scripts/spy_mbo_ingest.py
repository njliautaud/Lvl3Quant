#!/usr/bin/env python3
"""spy_mbo_ingest.py — Decode XNAS.ITCH SPY MBO into the 6-col ES-compatible NPZ.

Input:  /home/nick/Lvl3Quant/data/raw/spy_mbo/xnas-itch-YYYYMMDD.mbo.dbn.zst
Output: /home/nick/Lvl3Quant/data/processed/spy_mbo_events/YYYYMMDD_mbo_events.npz

Per-event NPZ schema (matches ES mbo_event_pipeline.py output):
  events      (N, 6) float32   [time_delta_log, event_type_id, side_id,
                                 price_rel_ticks, qty_log, spread_ticks]
  timestamps  (N,)  int64      ns since epoch
  labels_1s   (N,)  float32    forward mid change (ticks) over next 1s
  labels_5s   (N,)  float32
  labels_10s  (N,)  float32
  labels_30s  (N,)  float32
  metadata    object (dict)    date, n_events, tick_size, instrument_id

Encoding (must match alpha_discovery/.../precompute_features_smart_v3.py):
  event_type_id: A=0, C=1, M=2, T=3, F=4
  side_id:       B=0, A=1, N=2

Top-of-book tracking: maintain a sorted price ladder of (bid prices, ask prices)
with net quantities; best_bid = max bid level with positive qty, best_ask = min
ask with positive qty. This is more honest than the SchemaAdapter's naive
top-of-book heuristic, since SPY ITCH has many away-from-NBBO levels (hidden,
2500-share liquidity-provider quotes, etc.).
"""
from __future__ import annotations

import os
import sys
import time
import json
import math
from pathlib import Path
from collections import Counter
import logging

import numpy as np
import databento as db
from sortedcontainers import SortedDict

# ─── Config ───────────────────────────────────────────────────────────────
RAW_DIR = Path("/home/nick/Lvl3Quant/data/raw/spy_mbo")
OUT_DIR = Path("/home/nick/Lvl3Quant/data/processed/spy_mbo_events")
STATS_DIR = Path("/home/nick/Lvl3Quant/data/processed/spy_mbo_events/stats")
OUT_DIR.mkdir(parents=True, exist_ok=True)
STATS_DIR.mkdir(parents=True, exist_ok=True)

TICK_SIZE_FIXED = 10_000_000          # 0.01 USD in 1e-9 fixed-point units
TICK_SIZE_FLOAT = 0.01
INVALID_PRICE = 9_223_372_036_854_775_807

# RTH 14:30–21:00 UTC = 09:30–16:00 ET (DST handling: Mar 9 2026 = DST starts,
# but Databento timestamps are UTC, so we compare to RTH in UTC. Mar 2-6 are
# pre-DST (EST=UTC-5 → 14:30–21:00 UTC); Mar 9-12 are post-DST (EDT=UTC-4 →
# 13:30–20:00 UTC). We pass dt_start_utc/dt_end_utc explicitly per date.)
RTH_BY_DATE = {
    # pre-DST (Mar 2-6): 14:30–21:00 UTC
    "20260302": (14*3600+30*60, 21*3600),
    "20260303": (14*3600+30*60, 21*3600),
    "20260304": (14*3600+30*60, 21*3600),
    "20260305": (14*3600+30*60, 21*3600),
    "20260306": (14*3600+30*60, 21*3600),
    # DST started Sun Mar 8 2026 → Mon Mar 9 onward 13:30–20:00 UTC
    "20260309": (13*3600+30*60, 20*3600),
    "20260310": (13*3600+30*60, 20*3600),
    "20260311": (13*3600+30*60, 20*3600),
    "20260312": (13*3600+30*60, 20*3600),
}

LABEL_HORIZONS_NS = {
    "labels_1s":  1_000_000_000,
    "labels_5s":  5_000_000_000,
    "labels_10s": 10_000_000_000,
    "labels_30s": 30_000_000_000,
}

MAX_TIME_DELTA_MS = 60_000.0      # cap time_delta_ms at 60s before log
MAX_PRICE_TICKS = 50              # clip price_rel to ±50 ticks
LOG_INTERVAL = 2_000_000

ACTION_ENCODING = {"A": 0, "C": 1, "M": 2, "T": 3, "F": 4}
SIDE_ENCODING = {"B": 0, "A": 1, "N": 2}

logger = logging.getLogger(__name__)


# ─── LOB tracker ──────────────────────────────────────────────────────────
class TopOfBook:
    """Maintain best bid / best ask using sorted price ladders.

    Levels can go negative if the ITCH stream has fills/cancels that exceed
    our observed adds (we only see adds *after* connection; pre-existing
    book is unknown). Clamp to 0.
    """
    def __init__(self):
        self.bid_levels = SortedDict()  # price_fixed -> qty (descending best=max key)
        self.ask_levels = SortedDict()  # price_fixed -> qty (ascending best=min key)
        self.best_bid = 0
        self.best_ask = 0
        self.mid = 0.0
        self.spread_ticks = 0.0
        self.last_trade_price = 0

    def _refresh_best(self):
        # Pop empty levels off the top.
        while self.bid_levels and self.bid_levels.peekitem(-1)[1] <= 0:
            self.bid_levels.popitem(-1)
        while self.ask_levels and self.ask_levels.peekitem(0)[1] <= 0:
            self.ask_levels.popitem(0)
        self.best_bid = self.bid_levels.peekitem(-1)[0] if self.bid_levels else 0
        self.best_ask = self.ask_levels.peekitem(0)[0]  if self.ask_levels else 0
        if self.best_bid > 0 and self.best_ask > 0 and self.best_ask >= self.best_bid:
            self.mid = (self.best_bid + self.best_ask) / 2.0
            self.spread_ticks = (self.best_ask - self.best_bid) / TICK_SIZE_FIXED
        elif self.last_trade_price > 0:
            self.mid = float(self.last_trade_price)

    def process(self, act: str, side: str, price: int, qty: int):
        if price == INVALID_PRICE or price <= 0:
            return
        if act == 'A':
            if side == 'B':
                self.bid_levels[price] = self.bid_levels.get(price, 0) + qty
            elif side == 'A':
                self.ask_levels[price] = self.ask_levels.get(price, 0) + qty
        elif act == 'C':
            if side == 'B' and price in self.bid_levels:
                self.bid_levels[price] -= qty
            elif side == 'A' and price in self.ask_levels:
                self.ask_levels[price] -= qty
        elif act == 'M':
            # Treat modify as cancel+add (we don't track order ids).
            if side == 'B':
                self.bid_levels[price] = self.bid_levels.get(price, 0) + qty
            elif side == 'A':
                self.ask_levels[price] = self.ask_levels.get(price, 0) + qty
        elif act in ('T', 'F'):
            # Trade/fill — reduce resting qty on the opposite side of aggressor.
            # In ITCH, side= aggressor side. The resting order was on the other.
            self.last_trade_price = price
            if side == 'B' and price in self.ask_levels:
                self.ask_levels[price] -= qty
            elif side == 'A' and price in self.bid_levels:
                self.bid_levels[price] -= qty
        # 'R' = clear, but we filter those out at the caller.
        self._refresh_best()


# ─── Compute labels ───────────────────────────────────────────────────────
def compute_labels(ts_ns: np.ndarray, mid_prices_fixed: np.ndarray) -> dict:
    """For each event i: future mid change in TICKS over horizon h."""
    N = len(ts_ns)
    out = {k: np.full(N, np.nan, dtype=np.float32) for k in LABEL_HORIZONS_NS}
    for hkey, hns in LABEL_HORIZONS_NS.items():
        arr = out[hkey]
        target = ts_ns + hns
        # j[i] = first index j > i with ts_ns[j] >= target[i].
        # searchsorted on ts_ns is monotonic.
        j_idx = np.searchsorted(ts_ns, target, side='left')
        # Anywhere j_idx >= N: no future data, leave NaN.
        valid = j_idx < N
        if valid.any():
            cur_mid  = mid_prices_fixed.astype(np.float64)
            fut_mid  = np.where(valid, mid_prices_fixed[np.minimum(j_idx, N-1)].astype(np.float64), 0.0)
            ok = valid & (cur_mid > 0) & (fut_mid > 0)
            arr[ok] = ((fut_mid[ok] - cur_mid[ok]) / TICK_SIZE_FIXED).astype(np.float32)
    return out


# ─── Single-file processor ────────────────────────────────────────────────
def process_file(date_str: str, force: bool = False) -> dict:
    filepath = RAW_DIR / f"xnas-itch-{date_str}.mbo.dbn.zst"
    out_path = OUT_DIR / f"{date_str}_mbo_events.npz"
    stats_path = STATS_DIR / f"{date_str}_stats.json"

    if out_path.exists() and not force:
        logger.info(f"[SKIP] {date_str} — already exists")
        return {"date": date_str, "status": "skipped"}

    if not filepath.exists():
        logger.warning(f"[MISSING] {filepath}")
        return {"date": date_str, "status": "missing"}

    t0 = time.time()
    logger.info(f"[START] {date_str}")

    store = db.DBNStore.from_file(str(filepath))
    lob = TopOfBook()

    ts_list, act_list, sid_list = [], [], []
    price_list, qty_list, mid_list, spr_list = [], [], [], []
    action_counts = Counter()
    raw_total = 0

    rth_start_sec, rth_end_sec = RTH_BY_DATE[date_str]

    for r in store:
        raw_total += 1
        if raw_total % LOG_INTERVAL == 0:
            logger.info(f"  {date_str}: scanned {raw_total:,}")

        act = chr(r.action) if isinstance(r.action, int) else str(r.action)
        if act == 'R':
            # 'R' = clear; reset book.
            lob = TopOfBook()
            continue

        sid = chr(r.side) if isinstance(r.side, int) else str(r.side)
        act_id = ACTION_ENCODING.get(act)
        sid_id = SIDE_ENCODING.get(sid)
        if act_id is None or sid_id is None:
            continue

        # Update book FIRST (use post-event mid for price_rel as in ES pipeline).
        lob.process(act, sid, r.price, r.size)

        ts_list.append(int(r.ts_event))
        act_list.append(act)
        sid_list.append(sid)
        price_list.append(int(r.price) if r.price != INVALID_PRICE else 0)
        qty_list.append(int(r.size))
        mid_list.append(lob.mid)
        spr_list.append(lob.spread_ticks if lob.best_bid > 0 and lob.best_ask > 0 else 0.0)
        action_counts[act] += 1

    N = len(ts_list)
    logger.info(f"  {date_str}: {N:,} events from {raw_total:,} raw records, actions={dict(action_counts)}")
    if N < 1000:
        return {"date": date_str, "status": "thin", "n": N}

    ts_arr = np.array(ts_list, dtype=np.int64)
    mid_arr = np.array(mid_list, dtype=np.float64)
    spr_arr = np.array(spr_list, dtype=np.float32)
    price_arr = np.array(price_list, dtype=np.int64)
    qty_arr = np.array(qty_list, dtype=np.int32)
    # act/side ids
    act_ids = np.array([ACTION_ENCODING[a] for a in act_list], dtype=np.int8)
    sid_ids = np.array([SIDE_ENCODING[s] for s in sid_list], dtype=np.int8)
    del ts_list, mid_list, spr_list, price_list, qty_list, act_list, sid_list

    # Forward-fill mid for pre-book-init events.
    last = 0.0
    for i in range(len(mid_arr)):
        if mid_arr[i] > 0:
            last = mid_arr[i]
        elif last > 0:
            mid_arr[i] = last

    # RTH filter.
    sec_in_day = (ts_arr // 1_000_000_000) % 86400
    rth_mask = (sec_in_day >= rth_start_sec) & (sec_in_day < rth_end_sec)
    rth_idx = np.where(rth_mask)[0]
    n_rth = len(rth_idx)
    pct_rth = 100*n_rth/N if N else 0
    logger.info(f"  {date_str}: {n_rth:,} RTH events ({pct_rth:.1f}%)")
    if n_rth < 100:
        return {"date": date_str, "status": "no_rth", "n_rth": n_rth}

    ts_r = ts_arr[rth_idx]
    mid_r = mid_arr[rth_idx]
    spr_r = spr_arr[rth_idx]
    price_r = price_arr[rth_idx]
    qty_r = qty_arr[rth_idx]
    act_r = act_ids[rth_idx]
    sid_r = sid_ids[rth_idx]

    # Build 6-col feature matrix.
    M = n_rth
    feats = np.zeros((M, 6), dtype=np.float32)

    # [0] time_delta_log
    dts = np.diff(ts_r, prepend=ts_r[0]).astype(np.float64)
    dts_ms = np.clip(dts / 1_000_000.0, 0.0, MAX_TIME_DELTA_MS)
    feats[:, 0] = np.log1p(dts_ms).astype(np.float32)

    # [1] event_type_id, [2] side_id
    feats[:, 1] = act_r.astype(np.float32)
    feats[:, 2] = sid_r.astype(np.float32)

    # [3] price_rel_ticks = (price - mid)/tick, clipped ±50 ticks.
    # NOTE: SPY xnas-itch has many away-from-NBBO orders (hidden, deep book
    # liquidity quotes). Clipping at ±50 ticks (50 cents) is reasonable but
    # truncates a meaningful tail; document in caveats.
    ok_mid = mid_r > 0
    price_rel = np.zeros(M, dtype=np.float64)
    price_rel[ok_mid] = (price_r[ok_mid] - mid_r[ok_mid]) / TICK_SIZE_FIXED
    feats[:, 3] = np.clip(price_rel, -MAX_PRICE_TICKS, MAX_PRICE_TICKS).astype(np.float32)

    # [4] qty_log
    feats[:, 4] = np.log(np.maximum(1, qty_r).astype(np.float64)).astype(np.float32)

    # [5] spread_ticks  (already in ticks; clip absurd outliers from crossed/locked books)
    feats[:, 5] = np.clip(spr_r, 0.0, 200.0)

    # Labels — use mid in fixed-point units.
    labels = compute_labels(ts_r, mid_r.astype(np.int64))

    # Save.
    save_dict = {
        "events": feats,
        "timestamps": ts_r,
        **labels,
    }
    np.savez(out_path, **save_dict)

    # Stats JSON.
    label_stats = {k: {
        "n_valid": int(np.sum(~np.isnan(v))),
        "mean":    float(np.nanmean(v)) if np.any(~np.isnan(v)) else None,
        "std":     float(np.nanstd(v))  if np.any(~np.isnan(v)) else None,
        "p50_abs": float(np.nanpercentile(np.abs(v), 50)) if np.any(~np.isnan(v)) else None,
        "p99_abs": float(np.nanpercentile(np.abs(v), 99)) if np.any(~np.isnan(v)) else None,
    } for k, v in labels.items()}

    finite_mid = mid_r[mid_r > 0]
    finite_spr = spr_r[spr_r > 0]
    stats = {
        "date": date_str,
        "tick_size": TICK_SIZE_FLOAT,
        "n_raw": raw_total,
        "n_events": int(N),
        "n_rth": int(n_rth),
        "actions": dict(action_counts),
        "mid_min": float(finite_mid.min()/1e9) if len(finite_mid) else None,
        "mid_max": float(finite_mid.max()/1e9) if len(finite_mid) else None,
        "mid_mean": float(finite_mid.mean()/1e9) if len(finite_mid) else None,
        "spread_p50": float(np.percentile(finite_spr, 50)) if len(finite_spr) else None,
        "spread_p99": float(np.percentile(finite_spr, 99)) if len(finite_spr) else None,
        "labels": label_stats,
        "elapsed_sec": time.time() - t0,
    }
    with open(stats_path, "w") as f:
        json.dump(stats, f, indent=2)

    logger.info(f"[DONE] {date_str} in {stats['elapsed_sec']:.1f}s "
                f"mid={stats['mid_mean']:.2f} spread_p50={stats['spread_p50']:.2f}t")
    return {"date": date_str, "status": "ok", **stats}


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    dates = sorted(RTH_BY_DATE.keys())
    results = []
    for d in dates:
        try:
            results.append(process_file(d))
        except Exception as e:
            logger.exception(f"[ERROR] {d}: {e}")
            results.append({"date": d, "status": "error", "error": str(e)})
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
