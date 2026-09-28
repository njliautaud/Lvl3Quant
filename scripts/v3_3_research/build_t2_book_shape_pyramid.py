"""
build_t2_book_shape_pyramid.py
==============================
Build the v3.4 / spec §5 compliant 20-level book-shape pyramid from raw MBO DBN.

OUTPUT SHAPE PER EVENT: (20_levels, 6_features) flat = 120 channels
  Levels: mid-10, mid-9, ..., mid-1 (bid side), mid+1, ..., mid+10 (ask side)
          (mid-rounded to nearest tick at each event; levels indexed in tick offsets)

  Per-level features (spec §5):
    [0] size                       — total size at that price level
    [1] n_orders                   — count of distinct live orders at that price
    [2] mean_order_age             — mean(now - order_placed_ts) over live orders at this price, in seconds
    [3] cancel_rate                — # cancel events at this price in rolling W-event window
    [4] add_rate                   — # add events at this price in rolling W-event window
    [5] executed_size_in_window    — sum of trade sizes at this price in rolling W-event window

OUTPUT FORMAT: per-day parquet at data/derived/tier2_book_shape_pyramid_v1.parquet/<date>.parquet
  Columns: timestamp_ns (int64), then 120 float32 columns named L{-10..-1,+1..+10}_{size,n_orders,age,cxl,add,exec}
  Row-aligned 1:1 with the existing mbo_events.npz reference timestamps (same N rows per day).

HC compliance:
  - HC #356: replaces bucketed T2 with spec-compliant 20-level pyramid (the "T2 redefined" mandate).
  - HC #329: 6-feature spec preserved exactly.
  - HC #307D: NEW script under scripts/v3_3_research/, malware-guard compliant (does NOT modify existing v3.2 T2 builder).
  - HC #383: spec compliance is the v3.4.2 fix — this is the data prep half.

Usage:
  python build_t2_book_shape_pyramid.py --date 20260306        # single date
  python build_t2_book_shape_pyramid.py --workers 2            # all available dates
  python build_t2_book_shape_pyramid.py --workers 2 --skip-existing
  python build_t2_book_shape_pyramid.py --rolling-window 100   # default 100 events

Status: NEW BUILD. Estimated runtime per date: ~5-8 min on Jupiter 1 CPU (single date),
        ~20-30h for all 238 dates with 2 workers per HC #383 + HC spec §43.
"""

import sys, os, time, argparse, logging, json
from pathlib import Path
from typing import Optional, Dict, List, Tuple
from collections import deque

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    handlers=[logging.StreamHandler(sys.stdout)])
logger = logging.getLogger(__name__)

PROJECT_ROOT  = Path('/home/jupiter/Lvl3Quant')
RAW_DIR       = PROJECT_ROOT / 'data' / 'raw' / 'mbo'
MBO_EVENTS_DIR = PROJECT_ROOT / 'data' / 'processed' / 'mbo_events'
OUT_DIR       = PROJECT_ROOT / 'data' / 'derived' / 'tier2_book_shape_pyramid_v1.parquet'

TICK_SIZE        = 0.25
N_LEVELS_PER_SIDE = 10          # mid±1..±10 → 20 levels total
N_TOTAL_LEVELS    = 2 * N_LEVELS_PER_SIDE
N_FEATURES_PER_LEVEL = 6        # size, n_orders, age, cxl_rate, add_rate, exec_size
ROLLING_W        = 100          # events in rolling window (tunable via CLI)
CHUNK_SIZE       = 100_000      # DBN read chunk

# Output column names: -10..-1 (bid side), +1..+10 (ask side)
LEVEL_OFFSETS = list(range(-N_LEVELS_PER_SIDE, 0)) + list(range(1, N_LEVELS_PER_SIDE + 1))
FEATURE_SUFFIX = ['size', 'n_orders', 'age', 'cxl', 'add', 'exec']

def make_column_names() -> List[str]:
    cols = []
    for off in LEVEL_OFFSETS:
        sign = '-' if off < 0 else '+'
        for feat in FEATURE_SUFFIX:
            cols.append(f'L{sign}{abs(off):02d}_{feat}')
    return cols

COL_NAMES = make_column_names()
assert len(COL_NAMES) == N_TOTAL_LEVELS * N_FEATURES_PER_LEVEL == 120


# =============================================================
# Per-tick rolling counters
# =============================================================
class TickRollingStats:
    """
    Maintains rolling W-event counters per absolute tick:
      cxl_count, add_count, exec_size_sum
    Updates incrementally as events enter/leave the window.

    Uses a deque of (tick, action_type, size) for the window history,
    plus a dict mapping tick -> [cxl, add, exec] for fast O(1) lookup.
    """
    __slots__ = ('window_size', 'history', 'per_tick')

    def __init__(self, window_size: int):
        self.window_size = window_size
        self.history: deque = deque()   # (tick, action_code, size); action_code: 0=cxl, 1=add, 2=exec
        # per_tick[tick] = [cxl_count, add_count, exec_size_sum]
        self.per_tick: Dict[int, List[float]] = {}

    def record(self, tick: int, action_code: int, size: float):
        """Add new event to window, evict oldest if window full."""
        self.history.append((tick, action_code, size))
        bucket = self.per_tick.setdefault(tick, [0.0, 0.0, 0.0])
        if action_code == 0:
            bucket[0] += 1.0
        elif action_code == 1:
            bucket[1] += 1.0
        elif action_code == 2:
            bucket[2] += size

        # Evict
        while len(self.history) > self.window_size:
            old_tick, old_ac, old_sz = self.history.popleft()
            old_bucket = self.per_tick.get(old_tick)
            if old_bucket is None:
                continue
            if old_ac == 0:
                old_bucket[0] -= 1.0
            elif old_ac == 1:
                old_bucket[1] -= 1.0
            elif old_ac == 2:
                old_bucket[2] -= old_sz
            # GC empty buckets to keep dict size sane
            if old_bucket[0] <= 0 and old_bucket[1] <= 0 and old_bucket[2] <= 0:
                self.per_tick.pop(old_tick, None)

    def get(self, tick: int) -> Tuple[float, float, float]:
        b = self.per_tick.get(tick)
        if b is None:
            return (0.0, 0.0, 0.0)
        return (b[0], b[1], b[2])


# =============================================================
# Main per-day worker
# =============================================================
def process_day(dbn_path: Path, npz_path: Optional[Path] = None,
                out_dir: Optional[Path] = None,
                rolling_window: int = ROLLING_W,
                skip_existing: bool = True) -> Optional[Path]:
    import databento as db

    _out_dir = out_dir or OUT_DIR
    date_str = dbn_path.name.split('-')[-1].replace('.mbo.dbn.zst', '')
    out_path = _out_dir / f'{date_str}.parquet'

    if skip_existing and out_path.exists():
        logger.info(f'[{date_str}] exists, skip')
        return out_path

    if npz_path is None:
        npz_path = MBO_EVENTS_DIR / f'{date_str}_mbo_events.npz'
    if not npz_path.exists():
        logger.warning(f'[{date_str}] no mbo_events npz at {npz_path}, skipping')
        return None

    npz = np.load(npz_path, allow_pickle=True)
    ref_ts: np.ndarray = npz['timestamps']     # (N,) int64 ns
    N = len(ref_ts)
    logger.info(f'[{date_str}] N={N} events; loading DBN ({dbn_path.stat().st_size/1e6:.0f} MB)')

    target_iid = int(json.loads(npz['metadata'].item())['instrument_id'])

    t0 = time.time()

    # ----- Book state -----
    # bids/asks: tick_idx -> total size
    bids: Dict[int, int] = {}
    asks: Dict[int, int] = {}
    # bids_orders/asks_orders: tick_idx -> set of order_ids
    bids_orders: Dict[int, set] = {}
    asks_orders: Dict[int, set] = {}
    # orders: order_id -> (is_bid, tick_idx, size, placed_ts_ns)
    orders: Dict[int, Tuple[bool, int, int, int]] = {}

    # Rolling cancel/add/exec stats — separate for bid and ask side
    bid_stats = TickRollingStats(rolling_window)
    ask_stats = TickRollingStats(rolling_window)

    # Output buffer: (N, 120) float32
    feat = np.zeros((N, N_TOTAL_LEVELS * N_FEATURES_PER_LEVEL), dtype=np.float32)
    n_out = 0
    n_records = 0

    store = db.DBNStore.from_file(str(dbn_path))

    def price_to_tick(p: float) -> int:
        return int(round(p / TICK_SIZE))

    def compute_level_features(side_book: Dict[int, int],
                               side_orders: Dict[int, set],
                               side_stats: TickRollingStats,
                               level_ticks: List[int],
                               cur_ts_ns: int,
                               out_slice: np.ndarray):
        """Fill 60 floats (10 levels × 6 feats) into out_slice."""
        for li, tk in enumerate(level_ticks):
            base = li * N_FEATURES_PER_LEVEL
            size = side_book.get(tk, 0)
            order_ids = side_orders.get(tk)
            if order_ids and size > 0:
                n_ord = len(order_ids)
                # Mean age in seconds
                ages_ns_sum = 0
                for oid in order_ids:
                    o = orders.get(oid)
                    if o is not None:
                        ages_ns_sum += (cur_ts_ns - o[3])
                mean_age_s = (ages_ns_sum / n_ord) / 1e9 if n_ord > 0 else 0.0
            else:
                n_ord = 0
                mean_age_s = 0.0
            cxl, add, exec_sz = side_stats.get(tk)
            out_slice[base + 0] = float(size)
            out_slice[base + 1] = float(n_ord)
            out_slice[base + 2] = float(mean_age_s)
            out_slice[base + 3] = float(cxl)
            out_slice[base + 4] = float(add)
            out_slice[base + 5] = float(exec_sz)

    # Iterate DBN — populate book + emit row PER mbo_events ref row.
    # Critical: rows in features MUST align 1:1 with ref_ts.
    # The existing v3 builder iterates DBN events with action != 'R' and writes
    # one feature row per event up to N. We follow the same protocol so alignment
    # holds with the existing mbo_events npz indexing.
    for chunk in store.to_df(count=CHUNK_SIZE):
        valid = chunk[(chunk['instrument_id'] == target_iid) & (chunk['action'] != 'R')]
        if valid.empty:
            continue

        actions   = valid['action'].values
        sides     = valid['side'].values
        prices    = valid['price'].values
        sizes     = valid['size'].values.astype(np.int32)
        order_ids = valid['order_id'].values.astype(np.int64)
        ts_ns_arr = valid['ts_recv'].values.astype('int64') if 'ts_recv' in valid.columns else \
                    valid['ts_event'].values.astype('int64') if 'ts_event' in valid.columns else \
                    np.zeros(len(valid), dtype=np.int64)

        for i in range(len(valid)):
            if n_out >= N:
                break
            action = actions[i]
            side   = sides[i]
            price  = prices[i]
            size   = int(sizes[i])
            oid    = int(order_ids[i])
            ts_ns  = int(ts_ns_arr[i])
            n_records += 1

            is_bid    = (side == 'B')
            is_trade  = (action == 'T' or action == 'F')
            is_add    = (action == 'A')
            is_cancel = (action == 'C')
            is_modify = (action == 'M')

            if price != price:  # NaN guard
                tk = 0
            else:
                tk = price_to_tick(price)

            book = bids if is_bid else asks
            book_orders = bids_orders if is_bid else asks_orders
            stats = bid_stats if is_bid else ask_stats

            if is_add:
                orders[oid] = (is_bid, tk, size, ts_ns)
                book[tk] = book.get(tk, 0) + size
                s = book_orders.setdefault(tk, set())
                s.add(oid)
                stats.record(tk, 1, float(size))

            elif is_cancel:
                if oid in orders:
                    o_bid, o_tk, o_sz, _ = orders.pop(oid)
                    b = bids if o_bid else asks
                    bo = bids_orders if o_bid else asks_orders
                    st = bid_stats if o_bid else ask_stats
                    cur = b.get(o_tk, 0) - o_sz
                    if cur <= 0:
                        b.pop(o_tk, None)
                    else:
                        b[o_tk] = cur
                    s = bo.get(o_tk)
                    if s is not None:
                        s.discard(oid)
                        if not s:
                            bo.pop(o_tk, None)
                    st.record(o_tk, 0, float(o_sz))

            elif is_modify:
                if oid in orders:
                    o_bid, o_tk, o_sz, o_ts = orders[oid]
                    b = bids if o_bid else asks
                    bo = bids_orders if o_bid else asks_orders
                    # Remove from old tick
                    cur = b.get(o_tk, 0) - o_sz
                    if cur <= 0:
                        b.pop(o_tk, None)
                    else:
                        b[o_tk] = cur
                    s = bo.get(o_tk)
                    if s is not None:
                        s.discard(oid)
                        if not s:
                            bo.pop(o_tk, None)
                    # Add to new tick (preserve original placed_ts for age)
                    orders[oid] = (o_bid, tk, size, o_ts)
                    b2 = bids if o_bid else asks
                    bo2 = bids_orders if o_bid else asks_orders
                    b2[tk] = b2.get(tk, 0) + size
                    s2 = bo2.setdefault(tk, set())
                    s2.add(oid)
                    # Treat modify as cancel-then-add for rolling counters
                    (bid_stats if o_bid else ask_stats).record(o_tk, 0, float(o_sz))
                    (bid_stats if o_bid else ask_stats).record(tk, 1, float(size))

            elif is_trade:
                # Trade hits opposite side at tick
                if is_bid:  # buy aggressor hits ask
                    cur = asks.get(tk, 0) - size
                    if cur <= 0:
                        asks.pop(tk, None)
                    else:
                        asks[tk] = cur
                    ask_stats.record(tk, 2, float(size))
                    # Note: order_id on trades is the *resting* order being hit;
                    # if we track it, decrement the order. For simplicity follow
                    # the existing v3 builder: don't try to identify which order
                    # was hit (would require full match logic).
                else:        # sell aggressor hits bid
                    cur = bids.get(tk, 0) - size
                    if cur <= 0:
                        bids.pop(tk, None)
                    else:
                        bids[tk] = cur
                    bid_stats.record(tk, 2, float(size))

            # --- Emit feature row for this event ---
            # Mid = (best_bid + best_ask) / 2 in tick space, rounded.
            best_bid_tk = max(bids.keys()) if bids else 0
            best_ask_tk = min(asks.keys()) if asks else 0
            if best_bid_tk > 0 and best_ask_tk > 0 and best_ask_tk > best_bid_tk:
                mid_tk_2x = best_bid_tk + best_ask_tk  # 2x mid in tick units
                # mid_tk_2x is always even or odd; use rounding to the nearest tick:
                mid_tk_floor = mid_tk_2x // 2
                # Bid levels: mid-1, mid-2, ..., mid-10 → in tick: (mid_tk_floor - k)
                # Ask levels: mid+1, mid+2, ..., mid+10 → in tick: (mid_tk_floor + k)  (with +0.5 tick offset)
                # When spread = 1 tick (most common), best_bid = mid_floor, best_ask = mid_floor+1
                # so bid_levels span (best_bid, best_bid-1, ...) — i.e. mid-1 = best_bid_tk
                bid_level_ticks = [best_bid_tk - k for k in range(N_LEVELS_PER_SIDE)]  # mid-1, mid-2 ... mid-10
                ask_level_ticks = [best_ask_tk + k for k in range(N_LEVELS_PER_SIDE)]  # mid+1, mid+2 ... mid+10
            else:
                bid_level_ticks = [0] * N_LEVELS_PER_SIDE
                ask_level_ticks = [0] * N_LEVELS_PER_SIDE

            row = feat[n_out]
            # Bid side fills first 60 floats (levels -10..-1 from spec layout — reverse order so closest-to-mid is last)
            # COL_NAMES expects: L-10, L-09, ..., L-01, L+01, ..., L+10
            # bid_level_ticks[0] = mid-1 (closest), bid_level_ticks[9] = mid-10 (farthest)
            # → write to indices reversed: L-10 at slot 0, L-01 at slot 9
            bid_levels_reversed = list(reversed(bid_level_ticks))  # now [mid-10, mid-9, ..., mid-1]
            compute_level_features(bids, bids_orders, bid_stats,
                                   bid_levels_reversed,
                                   ts_ns,
                                   row[0:N_LEVELS_PER_SIDE * N_FEATURES_PER_LEVEL])
            compute_level_features(asks, asks_orders, ask_stats,
                                   ask_level_ticks,
                                   ts_ns,
                                   row[N_LEVELS_PER_SIDE * N_FEATURES_PER_LEVEL:])
            n_out += 1

        if n_records % 500_000 < CHUNK_SIZE:
            logger.info(f'[{date_str}] {n_records/1e6:.1f}M records, '
                        f'{n_out}/{N} events ({time.time()-t0:.0f}s, '
                        f'rate={n_out/max(time.time()-t0,1):.0f}/s)')

        if n_out >= N:
            break

    elapsed = time.time() - t0
    logger.info(f'[{date_str}] DONE {n_records} records → {n_out}/{N} events in {elapsed:.0f}s')

    if n_out < N:
        logger.warning(f'[{date_str}] short by {N - n_out}, padding zeros')

    # ---- Build dataframe ----
    df = pd.DataFrame(feat, columns=COL_NAMES)
    df.insert(0, 'timestamp_ns', ref_ts.astype('int64'))

    _out_dir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, compression='zstd', index=False)
    logger.info(f'[{date_str}] wrote {out_path} '
                f'({out_path.stat().st_size/1e6:.1f} MB, shape={df.shape})')
    return out_path


# =============================================================
# Driver
# =============================================================
def list_dates() -> List[Path]:
    return sorted(RAW_DIR.glob('glbx-mdp3-*.mbo.dbn.zst'))

def worker_run(dbn_paths: List[Path], rolling_window: int, skip_existing: bool):
    for p in dbn_paths:
        try:
            process_day(p, rolling_window=rolling_window, skip_existing=skip_existing)
        except Exception as e:
            logger.exception(f'[{p.name}] FAILED: {e}')

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--date', type=str, default=None, help='YYYYMMDD single date')
    parser.add_argument('--workers', type=int, default=1)
    parser.add_argument('--rolling-window', type=int, default=ROLLING_W)
    parser.add_argument('--skip-existing', action='store_true', default=True)
    parser.add_argument('--no-skip-existing', dest='skip_existing', action='store_false')
    parser.add_argument('--out-dir', type=str, default=None)
    args = parser.parse_args()

    out_dir = Path(args.out_dir) if args.out_dir else OUT_DIR
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.date:
        candidates = sorted(RAW_DIR.glob(f'glbx-mdp3-{args.date}.mbo.dbn.zst'))
        if not candidates:
            logger.error(f'No DBN for date {args.date}')
            sys.exit(1)
        process_day(candidates[0], out_dir=out_dir,
                    rolling_window=args.rolling_window,
                    skip_existing=args.skip_existing)
        return

    paths = list_dates()
    logger.info(f'Found {len(paths)} dates')

    if args.workers <= 1:
        worker_run(paths, args.rolling_window, args.skip_existing)
        return

    # Round-robin assignment
    shards = [[] for _ in range(args.workers)]
    for i, p in enumerate(paths):
        shards[i % args.workers].append(p)

    from multiprocessing import Process
    procs = []
    for shard in shards:
        proc = Process(target=worker_run, args=(shard, args.rolling_window, args.skip_existing))
        proc.start()
        procs.append(proc)
    for proc in procs:
        proc.join()

if __name__ == '__main__':
    main()
