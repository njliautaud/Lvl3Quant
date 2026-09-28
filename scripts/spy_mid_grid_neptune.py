#!/usr/bin/env python3
"""spy_mid_grid_neptune.py — Build SPY mid-price 250ms grid time-series per day.

Runs ON NEPTUNE (has the raw .dbn.zst files + databento + sortedcontainers).
Reads from /home/nick/Lvl3Quant/data/raw/spy_mbo/xnas-itch-YYYYMMDD.mbo.dbn.zst
Writes to /home/nick/Lvl3Quant/data/processed/spy_mid_grid/YYYYMMDD_mid_250ms.npz

Per-file schema:
  grid_ts_ns  (T,)  int64    ns since epoch, RTH-only, 250-ms-aligned
  mid_price   (T,)  float64  SPY mid in dollars at grid timestamp (FWD-filled from latest book)
  bid_price   (T,)  float64
  ask_price   (T,)  float64
  spread_t    (T,)  float32  ticks (1 tick = $0.01)

Rationale: Phase A cross-asset IC + execution PnL needs SPY mid at lagged moments.
A 250-ms grid (4Hz) aligns with the ES prediction stride (per HC #74 / CLAUDE.md).
RTH only (US equities 09:30–16:00 ET, DST-aware as per spy_mbo_ingest.py).
"""
from __future__ import annotations
import sys, time, json, logging
from pathlib import Path
import numpy as np
import databento as db
from sortedcontainers import SortedDict

RAW_DIR  = Path("/home/nick/Lvl3Quant/data/raw/spy_mbo")
OUT_DIR  = Path("/home/nick/Lvl3Quant/data/processed/spy_mid_grid")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_SIZE_FIXED = 10_000_000  # 0.01 USD in 1e-9 fixed-point
TICK_SIZE_FLOAT = 0.01
INVALID_PRICE = 9_223_372_036_854_775_807
GRID_MS = 250
GRID_NS = GRID_MS * 1_000_000

# Same RTH map as spy_mbo_ingest.py (pre-DST vs post-DST handling)
RTH_BY_DATE = {
    "20260302": (14*3600+30*60, 21*3600),
    "20260303": (14*3600+30*60, 21*3600),
    "20260304": (14*3600+30*60, 21*3600),
    "20260305": (14*3600+30*60, 21*3600),
    "20260306": (14*3600+30*60, 21*3600),
    "20260309": (13*3600+30*60, 20*3600),
    "20260310": (13*3600+30*60, 20*3600),
    "20260311": (13*3600+30*60, 20*3600),
    "20260312": (13*3600+30*60, 20*3600),
}

logger = logging.getLogger(__name__)


class TopOfBook:
    """Same LOB tracker as spy_mbo_ingest.py."""
    def __init__(self):
        self.bid_levels = SortedDict()
        self.ask_levels = SortedDict()
        self.best_bid = 0
        self.best_ask = 0
        self.mid = 0.0
        self.spread_ticks = 0.0
        self.last_trade_price = 0

    def _refresh_best(self):
        while self.bid_levels and self.bid_levels.peekitem(-1)[1] <= 0:
            self.bid_levels.popitem(-1)
        while self.ask_levels and self.ask_levels.peekitem(0)[1] <= 0:
            self.ask_levels.popitem(0)
        self.best_bid = self.bid_levels.peekitem(-1)[0] if self.bid_levels else 0
        self.best_ask = self.ask_levels.peekitem(0)[0] if self.ask_levels else 0
        if self.best_bid > 0 and self.best_ask > 0 and self.best_ask >= self.best_bid:
            self.mid = (self.best_bid + self.best_ask) / 2.0
            self.spread_ticks = (self.best_ask - self.best_bid) / TICK_SIZE_FIXED
        elif self.last_trade_price > 0:
            self.mid = float(self.last_trade_price)

    def process(self, act, side, price, qty):
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
            if side == 'B':
                self.bid_levels[price] = self.bid_levels.get(price, 0) + qty
            elif side == 'A':
                self.ask_levels[price] = self.ask_levels.get(price, 0) + qty
        elif act in ('T', 'F'):
            self.last_trade_price = price
            if side == 'B' and price in self.ask_levels:
                self.ask_levels[price] -= qty
            elif side == 'A' and price in self.bid_levels:
                self.bid_levels[price] -= qty
        self._refresh_best()


def process_day(date_str: str, force: bool = False):
    filepath = RAW_DIR / f"xnas-itch-{date_str}.mbo.dbn.zst"
    out_path = OUT_DIR / f"{date_str}_mid_250ms.npz"
    if out_path.exists() and not force:
        logger.info(f"[SKIP] {date_str}")
        return {"date": date_str, "status": "skipped"}
    if not filepath.exists():
        logger.warning(f"[MISSING] {filepath}")
        return {"date": date_str, "status": "missing"}

    t0 = time.time()
    logger.info(f"[START] {date_str}")
    store = db.DBNStore.from_file(str(filepath))
    lob = TopOfBook()

    rth_start_sec, rth_end_sec = RTH_BY_DATE[date_str]

    # Establish day's epoch-day-start (UTC midnight ns)
    # Parse date to compute UTC start of day
    import datetime as dt
    d_obj = dt.datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=dt.timezone.utc)
    day_start_ns = int(d_obj.timestamp() * 1e9)
    rth_start_ns = day_start_ns + rth_start_sec * 1_000_000_000
    rth_end_ns   = day_start_ns + rth_end_sec * 1_000_000_000

    # Build grid timestamps (250-ms aligned, RTH-only)
    n_grid = (rth_end_ns - rth_start_ns) // GRID_NS
    grid_ts = rth_start_ns + np.arange(n_grid, dtype=np.int64) * GRID_NS

    mid_arr = np.zeros(n_grid, dtype=np.float64)   # in dollars
    bid_arr = np.zeros(n_grid, dtype=np.float64)
    ask_arr = np.zeros(n_grid, dtype=np.float64)
    spr_arr = np.zeros(n_grid, dtype=np.float32)   # ticks
    # Cursor into grid for forward-fill assignment as we stream events.
    grid_cur = 0  # next grid index awaiting fill
    last_mid_fixed = 0
    last_bid_fixed = 0
    last_ask_fixed = 0
    last_spr_ticks = 0.0

    raw_total = 0
    in_rth_total = 0
    pre_rth_total = 0
    LOG_INT = 2_000_000

    for r in store:
        raw_total += 1
        if raw_total % LOG_INT == 0:
            logger.info(f"  {date_str}: scanned {raw_total:,}  grid_cur={grid_cur}/{n_grid}")
        act = chr(r.action) if isinstance(r.action, int) else str(r.action)
        if act == 'R':
            lob = TopOfBook()
            continue
        side = chr(r.side) if isinstance(r.side, int) else str(r.side)
        if act not in ('A', 'C', 'M', 'T', 'F'):
            continue
        ts_ns = int(r.ts_event)
        # Process event into book FIRST.
        lob.process(act, side, r.price, r.size)

        if lob.best_bid > 0 and lob.best_ask > 0 and lob.best_ask >= lob.best_bid:
            last_mid_fixed = (lob.best_bid + lob.best_ask) // 2
            last_bid_fixed = lob.best_bid
            last_ask_fixed = lob.best_ask
            last_spr_ticks = lob.spread_ticks
        # Pre-RTH: just keep building book, don't write grid.
        if ts_ns < rth_start_ns:
            pre_rth_total += 1
            continue
        if ts_ns >= rth_end_ns:
            break
        in_rth_total += 1

        # Forward-fill all grid points STRICTLY BEFORE ts_ns with the prior state.
        # i.e. grid points with grid_ts < ts_ns and >= grid_cur's position.
        # Our prior state at the moment of THIS grid point is the LOB state just before this event.
        # But we already updated lob above. To be honest: we should record the post-event mid for
        # grid points >= ts_ns (i.e. the latest known mid at or after the grid time). So:
        # 1) For grid indices where grid_ts[i] < ts_ns: fill with PREVIOUS last_mid (i.e. the state
        #    from the prior event). But we've overwritten last_mid_fixed already.
        # SIMPLER CONVENTION: at any grid time t, mid_t = the LAST POST-EVENT mid with ts_event <= t.
        # So we fill all grid indices with grid_ts <= ts_ns up to the current grid_cur using PRE-EVENT
        # state, and then this event's post-state becomes the value for grid indices > grid_ts of this
        # event up to the NEXT event.
        # Implementation: at each event ts_ns, advance grid_cur while grid_ts[grid_cur] <= ts_ns,
        # writing the PREVIOUS (pre-event) state. THEN update "previous" to the current post-event state.
        # We'll track pre-event state in scratch vars BEFORE calling lob.process — restructure below.
        pass

    # The above structure is awkward — restart with a cleaner two-pass design.
    # PASS 1: replay events, recording (ts_ns, mid_fixed, bid_fixed, ask_fixed, spread_ticks) snapshots
    #         every state-change event. We'll then bin-assign onto the grid via searchsorted (last <= grid_ts).
    # This is memory-heavier but correct and clear.
    logger.info(f"  {date_str}: restarting with clean pass2 design")
    store2 = db.DBNStore.from_file(str(filepath))
    lob = TopOfBook()
    snap_ts = []
    snap_mid = []
    snap_bid = []
    snap_ask = []
    snap_spr = []
    raw_total = 0
    for r in store2:
        raw_total += 1
        if raw_total % LOG_INT == 0:
            logger.info(f"  {date_str}: pass2 scanned {raw_total:,}")
        act = chr(r.action) if isinstance(r.action, int) else str(r.action)
        if act == 'R':
            lob = TopOfBook()
            continue
        side = chr(r.side) if isinstance(r.side, int) else str(r.side)
        if act not in ('A', 'C', 'M', 'T', 'F'):
            continue
        ts_ns = int(r.ts_event)
        # Skip events outside an extended window (pre-RTH-15min .. post-RTH) for memory savings.
        if ts_ns < rth_start_ns - 15*60*1_000_000_000:
            # Still process to build book pre-RTH, but don't snapshot.
            lob.process(act, side, r.price, r.size)
            continue
        if ts_ns >= rth_end_ns + 60*1_000_000_000:
            break
        lob.process(act, side, r.price, r.size)
        if lob.best_bid > 0 and lob.best_ask > 0 and lob.best_ask >= lob.best_bid:
            snap_ts.append(ts_ns)
            snap_mid.append((lob.best_bid + lob.best_ask) / 2.0)
            snap_bid.append(float(lob.best_bid))
            snap_ask.append(float(lob.best_ask))
            snap_spr.append(float(lob.spread_ticks))

    n_snap = len(snap_ts)
    logger.info(f"  {date_str}: {n_snap:,} snapshots inside extended window")
    if n_snap < 100:
        return {"date": date_str, "status": "thin", "n_snap": n_snap}

    snap_ts = np.array(snap_ts, dtype=np.int64)
    snap_mid = np.array(snap_mid, dtype=np.float64) / 1e9          # to dollars
    snap_bid = np.array(snap_bid, dtype=np.float64) / 1e9
    snap_ask = np.array(snap_ask, dtype=np.float64) / 1e9
    snap_spr = np.array(snap_spr, dtype=np.float32)

    # Assign to grid: for each grid_ts[i], find LAST snap j with snap_ts[j] <= grid_ts[i].
    # searchsorted with side='right' then -1.
    idx = np.searchsorted(snap_ts, grid_ts, side='right') - 1
    valid = idx >= 0
    mid_arr[valid] = snap_mid[idx[valid]]
    bid_arr[valid] = snap_bid[idx[valid]]
    ask_arr[valid] = snap_ask[idx[valid]]
    spr_arr[valid] = snap_spr[idx[valid]]
    # For invalid (pre-first-snap) leave as 0 (we'll mark these later).

    # Stats
    nz = mid_arr > 0
    stats = {
        "date": date_str,
        "n_grid": int(n_grid),
        "n_valid_grid": int(nz.sum()),
        "rth_start_ns": int(rth_start_ns),
        "rth_end_ns": int(rth_end_ns),
        "mid_min": float(mid_arr[nz].min()) if nz.any() else None,
        "mid_max": float(mid_arr[nz].max()) if nz.any() else None,
        "mid_mean": float(mid_arr[nz].mean()) if nz.any() else None,
        "spread_p50_ticks": float(np.percentile(spr_arr[nz], 50)) if nz.any() else None,
        "spread_p99_ticks": float(np.percentile(spr_arr[nz], 99)) if nz.any() else None,
        "elapsed_sec": time.time() - t0,
        "tick_size": TICK_SIZE_FLOAT,
        "grid_ms": GRID_MS,
    }
    np.savez(out_path,
             grid_ts_ns=grid_ts,
             mid_price=mid_arr,
             bid_price=bid_arr,
             ask_price=ask_arr,
             spread_ticks=spr_arr,
             stats=np.array([json.dumps(stats)]))
    logger.info(f"[DONE] {date_str} in {stats['elapsed_sec']:.1f}s  "
                f"valid_grid={stats['n_valid_grid']}/{stats['n_grid']}  mid_mean={stats['mid_mean']:.2f}")
    return {"date": date_str, "status": "ok", **stats}


def main():
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s",
                        handlers=[logging.StreamHandler(sys.stdout)])
    dates = sorted(RTH_BY_DATE.keys())
    results = []
    force = "--force" in sys.argv
    for d in dates:
        try:
            results.append(process_day(d, force=force))
        except Exception as e:
            logger.exception(f"[ERROR] {d}: {e}")
            results.append({"date": d, "status": "error", "error": str(e)})
    print(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
