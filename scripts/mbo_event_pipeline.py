"""
MBO Event-Driven Data Pipeline — Phase 0
=========================================
Converts raw Databento MBO .dbn.zst files into event sequences for neural network training.

Output format per day: .npz with keys:
  - events:     (N_events, 6) float32  — [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]
  - labels_1s:  (N_events,) float32    — mid-price change (in ticks) over next 1s
  - labels_5s:  (N_events,) float32    — mid-price change over next 5s
  - labels_10s: (N_events,) float32    — mid-price change over next 10s
  - labels_30s: (N_events,) float32    — mid-price change over next 30s
  - timestamps: (N_events,) int64      — nanosecond ts_event for each event
  - metadata:   dict (pickled) — date, instrument_id, tick_size, n_events, n_rth_events, stats

Feature vector (6 dims):
  [0] time_delta_log:   log1p(milliseconds since prev event), clipped at 10s
  [1] event_type_id:    A=0, C=1, M=2, T=3, F=4  (R/clear excluded)
  [2] side_id:          B=0, A=1, N=2
  [3] price_rel_ticks:  (price - mid) / tick_size, clipped ±50 ticks
  [4] qty_log:          log1p(size), clipped at log1p(10000)
  [5] spread_ticks:     best_ask - best_bid in ticks (from LOB reconstruction), clipped ±20

Target instrument: ES front-month (instrument_id 294973 for Dec 2025 contract)
                   Will auto-detect dominant instrument per file.

RTH window: 09:30–16:00 ET (UTC-5 in winter, UTC-4 in summer)
            Uses loose RTH: 13:30–21:00 UTC (covers both EST and EDT)
            Tight label window: events must have 30s of future data

Usage:
  python3 mbo_event_pipeline.py [--input-dir DIR] [--output-dir DIR] [--workers N]
                                [--date-range START END] [--dry-run] [--stats-only]
"""

import argparse
import databento as dbn
import numpy as np
import os
import sys
import json
import time
import traceback
import logging
from pathlib import Path
from collections import Counter, defaultdict
from datetime import datetime, timezone, timedelta
import math
import multiprocessing as mp

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

RAW_DIR   = "/home/jupiter/Lvl3Quant/data/raw/mbo"
OUT_DIR   = "/home/jupiter/Lvl3Quant/data/processed/mbo_events"
STATS_DIR = "/home/jupiter/Lvl3Quant/data/processed/mbo_events_stats"

# ES/MES tick size = 0.25 index points = 250_000_000 in Databento fixed-point (1e9 scale)
TICK_SIZE_FIXED = 250_000_000   # 0.25 * 1e9
TICK_SIZE_FLOAT = 0.25

# Action encoding
ACTION_MAP = {
    'A': 0,   # Add
    'C': 1,   # Cancel
    'M': 2,   # Modify
    'T': 3,   # Trade
    'F': 4,   # Fill
}
# R = Clear/reset — excluded from sequences

# Side encoding
SIDE_MAP = {
    'B': 0,   # Bid
    'A': 1,   # Ask
    'N': 2,   # None (for clears)
}

# RTH window in UTC nanoseconds offsets from midnight
# RTH: 09:30–16:00 ET
# Winter (EST = UTC-5): 14:30–21:00 UTC
# Summer (EDT = UTC-4): 13:30–20:00 UTC
# Use conservative: 13:30–21:00 UTC to cover both
RTH_START_UTC_SEC = 13 * 3600 + 30 * 60   # 13:30:00 UTC
RTH_END_UTC_SEC   = 21 * 3600              # 21:00:00 UTC

# Label horizons in nanoseconds
LABEL_HORIZONS_NS = {
    '1s':  1_000_000_000,
    '5s':  5_000_000_000,
    '10s': 10_000_000_000,
    '30s': 30_000_000_000,
}

# Feature clipping
MAX_TIME_DELTA_MS = 10_000   # 10 seconds max gap (log-scaled)
MAX_PRICE_TICKS   = 50       # ±50 ticks from mid
MAX_QTY           = 10_000   # log1p(10000)
MAX_SPREAD_TICKS  = 20       # ±20 ticks


# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

def setup_logging(log_dir: str = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_dir:
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, f"pipeline_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  LOB (Level-1 Order Book) TRACKER — Lightweight
# ─────────────────────────────────────────────

class LOBTracker:
    """
    Memory-efficient Level-1 order book tracker for MBO events.

    Strategy:
    - Track best bid/ask using a small bounded price-level window (±100 ticks from last trade)
    - Use Trade/Fill events as anchors for the actual execution price
    - DO NOT track individual orders (avoids O(N_orders) memory growth)
    - Tracks price levels in sorted dict, pruned when > MAX_LEVELS

    This gives us accurate enough mid-price and spread for feature engineering
    without storing millions of order IDs in memory.
    """

    MAX_LEVELS = 20    # keep only ±10 ticks on each side
    PRUNE_EVERY = 5000  # prune stale levels every N events

    def __init__(self, tick_size_fixed: int = TICK_SIZE_FIXED):
        self.tick_size_fixed = tick_size_fixed
        # Price level qty: bid_levels {price -> net_qty}, ask_levels
        # We use int keys (fixed-point prices), values are signed qty (can go negative from fills, clamped at 0)
        self._bid_levels: dict = {}
        self._ask_levels: dict = {}
        self._best_bid: int = 0
        self._best_ask: int = 0
        self._mid: float = 0.0
        self._spread: float = float('nan')
        self._last_trade_price: int = 0
        self._event_count: int = 0

    def _recompute(self):
        valid_bids = [k for k, v in self._bid_levels.items() if v > 0]
        valid_asks = [k for k, v in self._ask_levels.items() if v > 0]

        self._best_bid = max(valid_bids) if valid_bids else 0
        self._best_ask = min(valid_asks) if valid_asks else 0

        if self._best_bid > 0 and self._best_ask > 0 and self._best_ask > self._best_bid:
            self._mid = (self._best_bid + self._best_ask) / 2.0
            self._spread = (self._best_ask - self._best_bid) / self.tick_size_fixed
        elif self._last_trade_price > 0:
            self._mid = float(self._last_trade_price)
            self._spread = float('nan')
        # else: keep previous mid

    def _prune_levels(self):
        """Remove stale levels far from current mid to keep memory bounded."""
        if self._mid <= 0:
            return
        # Keep only levels within ±50 ticks of current mid
        radius = 50 * self.tick_size_fixed
        lo = self._mid - radius
        hi = self._mid + radius
        self._bid_levels = {k: v for k, v in self._bid_levels.items() if lo <= k <= hi and v > 0}
        self._ask_levels = {k: v for k, v in self._ask_levels.items() if lo <= k <= hi and v > 0}

    def process(self, action: str, side: str, price: int, qty: int, order_id: int):
        """Update LOB with incoming event. Returns (mid_price_fixed, spread_ticks)."""
        INVALID_PRICE = 9_223_372_036_854_775_807

        self._event_count += 1

        # LEAKAGE FIX: Capture LOB state BEFORE processing event
        pre_mid = self._mid
        pre_spread = self._spread

        if action == 'R':
            # Clear — reset book state
            self._bid_levels.clear()
            self._ask_levels.clear()
            self._best_bid = 0
            self._best_ask = 0
            self._spread = float('nan')
            return pre_mid, pre_spread

        if price == INVALID_PRICE or price <= 0:
            return pre_mid, pre_spread

        if action == 'A':
            # Add order at price level
            if side == 'B':
                self._bid_levels[price] = self._bid_levels.get(price, 0) + qty
            elif side == 'A':
                self._ask_levels[price] = self._ask_levels.get(price, 0) + qty

        elif action == 'C':
            # Cancel — reduce at price level (we don't track order_ids, use qty directly)
            if side == 'B':
                self._bid_levels[price] = max(0, self._bid_levels.get(price, 0) - qty)
            elif side == 'A':
                self._ask_levels[price] = max(0, self._ask_levels.get(price, 0) - qty)

        elif action == 'M':
            # Modify — treated as cancel+add at new price
            # We don't know the old price without order_id tracking,
            # so just add at new price (slight inaccuracy, acceptable)
            if side == 'B':
                self._bid_levels[price] = self._bid_levels.get(price, 0) + qty
            elif side == 'A':
                self._ask_levels[price] = self._ask_levels.get(price, 0) + qty

        elif action in ('T', 'F'):
            # Trade/Fill — update last trade price, reduce qty at execution price
            self._last_trade_price = price
            if side == 'B':
                # Buyer-initiated trade fills at ask
                self._ask_levels[price] = max(0, self._ask_levels.get(price, 0) - qty)
            elif side == 'A':
                # Seller-initiated fills at bid
                self._bid_levels[price] = max(0, self._bid_levels.get(price, 0) - qty)

        # Periodic prune to keep memory bounded
        if self._event_count % self.PRUNE_EVERY == 0:
            self._prune_levels()

        self._recompute()
        return pre_mid, pre_spread

    @property
    def mid(self) -> float:
        return self._mid

    @property
    def spread_ticks(self) -> float:
        return self._spread


# ─────────────────────────────────────────────
#  PROCESS ONE FILE
# ─────────────────────────────────────────────

def is_rth_ns(ts_event_ns: int, date_str: str) -> bool:
    """
    Check if a nanosecond UTC timestamp falls within RTH.
    Uses loose RTH: 13:30–21:00 UTC (covers EST and EDT).
    """
    # Seconds within the day
    sec_in_day = (ts_event_ns // 1_000_000_000) % 86400
    return RTH_START_UTC_SEC <= sec_in_day < RTH_END_UTC_SEC


def compute_labels(timestamps_ns: np.ndarray, mid_prices: np.ndarray,
                   tick_size_fixed: int) -> dict:
    """
    For each event index i, compute mid-price change over each label horizon.
    mid_prices are in Databento fixed-point (divide by 1e9 for float, divide by tick_size_fixed for ticks).

    Returns dict horizon -> np.ndarray (N_events,) in TICKS, NaN if no future data.
    """
    N = len(timestamps_ns)
    labels = {h: np.full(N, np.nan, dtype=np.float32) for h in LABEL_HORIZONS_NS}

    for h_name, h_ns in LABEL_HORIZONS_NS.items():
        arr = labels[h_name]
        j = 0  # two-pointer: j is always >= i+1
        for i in range(N):
            t0 = timestamps_ns[i]
            t_target = t0 + h_ns
            # Advance j to at least i+1
            if j <= i:
                j = i + 1
            # If the last timestamp is before t_target, no label for this event
            if j >= N or timestamps_ns[N-1] < t_target:
                continue
            # Advance j until timestamps_ns[j] >= t_target
            while j < N and timestamps_ns[j] < t_target:
                j += 1
            if j < N:
                future_mid = mid_prices[j]
                current_mid = mid_prices[i]
                if not (math.isnan(future_mid) or math.isnan(current_mid) or
                        current_mid == 0 or future_mid == 0):
                    delta_fixed = future_mid - current_mid
                    arr[i] = delta_fixed / tick_size_fixed  # in ticks

    return labels


def detect_dominant_instrument(filepath: str, sample_size: int = 20000) -> int:
    """
    Fast scan of first sample_size records to detect dominant instrument_id.
    For ES/MES data, the dominant instrument is the front-month contract.
    """
    inst_counter = Counter()
    store = dbn.DBNStore.from_file(filepath)
    import itertools
    for r in itertools.islice(store, sample_size):
        if str(r.action) != 'R':
            inst_counter[r.instrument_id] += 1
    if not inst_counter:
        return -1
    return inst_counter.most_common(1)[0][0]


def process_file(filepath: str, output_dir: str, stats_dir: str,
                 dry_run: bool = False) -> dict:
    """
    Process one .dbn.zst file using streaming (single pass). Returns stats dict.

    Strategy:
    1. Quick 20k-record scan to detect dominant instrument
    2. Single streaming pass: filter to dominant instrument, build LOB, collect features
    3. Compute labels from the collected mid-price series
    4. Save .npz
    """
    filename = os.path.basename(filepath)
    date_str = filename.replace('glbx-mdp3-', '').replace('.mbo.dbn.zst', '')
    out_path = os.path.join(output_dir, f"{date_str}_mbo_events.npz")
    stats_path = os.path.join(stats_dir, f"{date_str}_stats.json")

    logger = logging.getLogger(__name__)

    # Skip if already processed
    if os.path.exists(out_path) and not dry_run:
        logger.info(f"[SKIP] {date_str} — already exists at {out_path}")
        existing = {}
        if os.path.exists(stats_path):
            with open(stats_path) as f:
                existing = json.load(f)
        return {'date': date_str, 'status': 'skipped', **existing}

    t_start = time.time()
    logger.info(f"[START] {date_str}")

    try:
        # ── Quick scan: detect dominant instrument ──
        dominant_instrument = detect_dominant_instrument(filepath)
        if dominant_instrument == -1:
            logger.warning(f"[EMPTY] {date_str} — no valid records in sample")
            return {'date': date_str, 'status': 'empty', 'n_records': 0}
        logger.info(f"  {date_str}: dominant instrument = {dominant_instrument}")

        # ── Single streaming pass ──
        store = dbn.DBNStore.from_file(filepath)
        lob = LOBTracker(tick_size_fixed=TICK_SIZE_FIXED)

        # Pre-allocate with dynamic lists (convert to numpy at end)
        ts_list    = []   # int64 nanoseconds
        act_list   = []   # str action code
        sid_list   = []   # str side code
        price_list = []   # int fixed-point price
        qty_list   = []   # int quantity
        mid_list   = []   # float fixed-point mid (from LOB)
        spr_list   = []   # float spread ticks (from LOB)
        oid_list   = []   # int64 order_id (for FIFO book reconstruction)

        action_counts = Counter()
        inst_total = 0
        LOG_INTERVAL = 1_000_000  # log every 1M records

        for r in store:
            act = str(r.action)
            if act == 'R':
                lob.process('R', str(r.side), r.price, r.size, r.order_id)
                continue

            inst_total += 1
            if inst_total % LOG_INTERVAL == 0:
                logger.info(f"  {date_str}: {inst_total:,} records scanned, {len(ts_list):,} for dominant instr so far")

            if r.instrument_id != dominant_instrument:
                continue

            sid = str(r.side)
            action_counts[act] += 1

            mid, spread = lob.process(act, sid, r.price, r.size, r.order_id)

            ts_list.append(r.ts_event)
            act_list.append(act)
            sid_list.append(sid)
            price_list.append(r.price)
            qty_list.append(r.size)
            mid_list.append(mid if not math.isnan(mid) else 0.0)
            spr_list.append(spread if not math.isnan(spread) else float('nan'))
            oid_list.append(r.order_id)

        N = len(ts_list)
        logger.info(f"  {date_str}: {N} events for instrument {dominant_instrument}, actions={dict(action_counts)}")

        if N < 100:
            logger.warning(f"[THIN] {date_str} — only {N} events, skipping")
            return {'date': date_str, 'status': 'thin', 'n_events': N}

        # Convert to numpy arrays
        ts_arr    = np.array(ts_list,    dtype=np.int64)
        mid_arr   = np.array(mid_list,   dtype=np.float64)
        spr_arr   = np.array(spr_list,   dtype=np.float32)
        price_arr = np.array(price_list, dtype=np.int64)
        qty_arr   = np.array(qty_list,   dtype=np.int32)
        oid_arr   = np.array(oid_list,   dtype=np.int64)
        # Free lists
        del ts_list, mid_list, spr_list, price_list, qty_list, oid_list

        # Forward-fill mid prices (for events before book initializes)
        for i in range(1, len(mid_arr)):
            if mid_arr[i] == 0.0 and mid_arr[i-1] != 0.0:
                mid_arr[i] = mid_arr[i-1]

        # ── RTH filter ──
        # Vectorized: compute seconds-in-day for all timestamps at once
        sec_in_day = (ts_arr // 1_000_000_000) % 86400
        rth_mask = (sec_in_day >= RTH_START_UTC_SEC) & (sec_in_day < RTH_END_UTC_SEC)
        rth_idx = np.where(rth_mask)[0]
        logger.info(f"  {date_str}: {len(rth_idx)} RTH events ({100*len(rth_idx)/N:.1f}%)")

        if len(rth_idx) < 50:
            logger.warning(f"[NO_RTH] {date_str} — only {len(rth_idx)} RTH events")
            if len(rth_idx) == 0:
                rth_idx = np.arange(N)  # fallback: use all events

        # ── Build feature matrix for RTH events ──
        ts_rth    = ts_arr[rth_idx]
        mid_rth   = mid_arr[rth_idx]
        spr_rth   = spr_arr[rth_idx]
        price_rth = price_arr[rth_idx]
        qty_rth   = qty_arr[rth_idx]
        oid_rth   = oid_arr[rth_idx]
        act_rth   = [act_list[i] for i in rth_idx]
        sid_rth   = [sid_list[i] for i in rth_idx]
        del act_list, sid_list

        M = len(rth_idx)
        features = np.zeros((M, 6), dtype=np.float32)

        INVALID_PRICE = 9_223_372_036_854_775_807

        # [0] time_delta_log — vectorized
        time_deltas_ns = np.diff(ts_rth, prepend=ts_rth[0])
        time_deltas_ms = time_deltas_ns / 1_000_000.0
        np.clip(time_deltas_ms, 0.0, MAX_TIME_DELTA_MS, out=time_deltas_ms)
        features[:, 0] = np.log1p(time_deltas_ms).astype(np.float32)

        # [1] event_type_id
        for i, act in enumerate(act_rth):
            features[i, 1] = float(ACTION_MAP.get(act, -1))

        # [2] side_id
        for i, sid in enumerate(sid_rth):
            features[i, 2] = float(SIDE_MAP.get(sid, 2))

        # [3] price_rel_ticks — vectorized where valid
        valid_price = (price_rth != INVALID_PRICE) & (mid_rth != 0.0)
        rel_ticks = np.where(valid_price,
                             (price_rth.astype(np.float64) - mid_rth) / TICK_SIZE_FIXED,
                             0.0)
        np.clip(rel_ticks, -MAX_PRICE_TICKS, MAX_PRICE_TICKS, out=rel_ticks)
        features[:, 3] = rel_ticks.astype(np.float32)

        # [4] qty_log — vectorized
        qty_clipped = np.minimum(qty_rth.astype(np.float64), MAX_QTY)
        features[:, 4] = np.log1p(qty_clipped).astype(np.float32)

        # [5] spread_ticks — vectorized
        spr_valid = ~np.isnan(spr_rth)
        spr_clipped = np.where(spr_valid, np.clip(spr_rth, 0.0, MAX_SPREAD_TICKS), 0.0)
        features[:, 5] = spr_clipped.astype(np.float32)

        # ── Compute labels ──
        labels = compute_labels(ts_rth, mid_rth, TICK_SIZE_FIXED)

        # ── Compute statistics ──
        inter_event_ms = np.diff(ts_rth) / 1_000_000.0
        events_per_sec_dist = 1000.0 / (inter_event_ms[inter_event_ms > 0]) if len(inter_event_ms) > 0 else np.array([0.0])

        total_act = sum(action_counts.values())
        stats = {
            'date': date_str,
            'status': 'ok',
            'instrument_id': dominant_instrument,
            'n_total_records': N,
            'n_rth_events': M,
            'rth_frac': float(M / N) if N > 0 else 0.0,
            'action_counts': {k: int(v) for k, v in action_counts.items()},
            'action_pct': {k: float(100 * v / total_act) for k, v in action_counts.items()},
            'inter_event_ms_mean': float(np.mean(inter_event_ms)) if len(inter_event_ms) > 0 else 0.0,
            'inter_event_ms_p50': float(np.percentile(inter_event_ms, 50)) if len(inter_event_ms) > 0 else 0.0,
            'inter_event_ms_p99': float(np.percentile(inter_event_ms, 99)) if len(inter_event_ms) > 0 else 0.0,
            'events_per_sec_mean': float(np.mean(events_per_sec_dist)) if len(events_per_sec_dist) > 0 else 0.0,
            'label_1s_mean': float(np.nanmean(labels['1s'])) if M > 0 else 0.0,
            'label_1s_std': float(np.nanstd(labels['1s'])) if M > 0 else 0.0,
            'label_1s_valid_frac': float(np.mean(~np.isnan(labels['1s']))) if M > 0 else 0.0,
            'label_30s_valid_frac': float(np.mean(~np.isnan(labels['30s']))) if M > 0 else 0.0,
            'processing_sec': time.time() - t_start,
        }

        # ── Build raw MBO arrays for proper FIFO book reconstruction ──
        # These enable downstream consumers (RL env, replay) to reconstruct
        # the full per-price-level FIFO order book from raw events.
        # action_raw: 0=Add, 1=Cancel, 2=Modify, 3=Trade, 4=Fill (matches ACTION_MAP)
        # side_raw:   0=Bid, 1=Ask, 2=None (matches SIDE_MAP)
        action_raw_arr = np.array(
            [ACTION_MAP.get(a, -1) for a in act_rth], dtype=np.int8
        )
        side_raw_arr = np.array(
            [SIDE_MAP.get(s, 2) for s in sid_rth], dtype=np.int8
        )

        if not dry_run:
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(stats_dir, exist_ok=True)

            np.savez_compressed(
                out_path,
                events=features,
                labels_1s=labels['1s'],
                labels_5s=labels['5s'],
                labels_10s=labels['10s'],
                labels_30s=labels['30s'],
                timestamps=ts_rth,
                # Raw MBO fields for FIFO book reconstruction
                order_ids=oid_rth,          # int64 — order_id from MBO feed
                prices_raw=price_rth,       # int64 — absolute fixed-point price
                sizes_raw=qty_rth,          # int32 — raw quantity
                sides_raw=side_raw_arr,     # int8 — 0=Bid, 1=Ask, 2=None
                actions_raw=action_raw_arr, # int8 — 0=Add, 1=Cancel, 2=Modify, 3=Trade, 4=Fill
                metadata=np.array([json.dumps({
                    'date': date_str,
                    'instrument_id': int(dominant_instrument),
                    'tick_size': TICK_SIZE_FLOAT,
                    'n_events': M,
                    'n_total_records': N,
                    'feature_names': ['time_delta_log', 'event_type_id', 'side_id',
                                      'price_rel_ticks', 'qty_log', 'spread_ticks'],
                    'action_encoding': ACTION_MAP,
                    'side_encoding': SIDE_MAP,
                    'label_horizons': list(LABEL_HORIZONS_NS.keys()),
                    'raw_field_encoding': {
                        'actions_raw': {v: k for k, v in ACTION_MAP.items()},
                        'sides_raw': {v: k for k, v in SIDE_MAP.items()},
                    },
                })], dtype=object)
            )
            with open(stats_path, 'w') as f:
                json.dump(stats, f, indent=2)

            elapsed = time.time() - t_start
            logger.info(f"[DONE] {date_str}: {M} RTH events → {out_path} ({elapsed:.1f}s)")

        return stats

    except Exception as e:
        logger.error(f"[ERROR] {date_str}: {traceback.format_exc()}")
        return {'date': date_str, 'status': 'error', 'error': str(e)}


# ─────────────────────────────────────────────
#  AGGREGATED STATISTICS
# ─────────────────────────────────────────────

def print_aggregate_stats(all_stats: list, logger):
    ok_stats = [s for s in all_stats if s.get('status') == 'ok']
    if not ok_stats:
        logger.info("No successfully processed files to aggregate")
        return

    n_events_list = [s['n_rth_events'] for s in ok_stats]
    eps_list = [s['events_per_sec_mean'] for s in ok_stats if s['events_per_sec_mean'] > 0]
    iet_list = [s['inter_event_ms_mean'] for s in ok_stats if s['inter_event_ms_mean'] > 0]

    # Aggregate action counts
    total_actions = Counter()
    for s in ok_stats:
        for k, v in s.get('action_counts', {}).items():
            total_actions[k] += v

    total = sum(total_actions.values())

    logger.info("=" * 60)
    logger.info("AGGREGATE STATISTICS")
    logger.info("=" * 60)
    logger.info(f"Files processed:           {len(ok_stats)}")
    logger.info(f"Total RTH events:          {sum(n_events_list):,}")
    logger.info(f"Events/day (mean):         {np.mean(n_events_list):,.0f}")
    logger.info(f"Events/day (min):          {np.min(n_events_list):,.0f}")
    logger.info(f"Events/day (max):          {np.max(n_events_list):,.0f}")
    logger.info(f"Events/sec (mean):         {np.mean(eps_list):.1f}")
    logger.info(f"Inter-event ms (mean):     {np.mean(iet_list):.3f}")

    logger.info("Action distribution (across all days):")
    for act_name, enc in sorted(ACTION_MAP.items(), key=lambda x: x[1]):
        ct = total_actions.get(act_name, 0)
        pct = 100 * ct / total if total > 0 else 0
        logger.info(f"  {act_name} ({['Add','Cancel','Modify','Trade','Fill'][enc]}): {ct:,} ({pct:.1f}%)")

    logger.info("=" * 60)


# ─────────────────────────────────────────────
#  MAIN
# ─────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="MBO Event-Driven Data Pipeline")
    parser.add_argument('--input-dir',  default=RAW_DIR,   help="Directory with .mbo.dbn.zst files")
    parser.add_argument('--output-dir', default=OUT_DIR,   help="Output directory for .npz event files")
    parser.add_argument('--stats-dir',  default=STATS_DIR, help="Output directory for stats JSON files")
    parser.add_argument('--workers',    type=int, default=4, help="Parallel workers (default 4)")
    parser.add_argument('--date-range', nargs=2, metavar=('START', 'END'),
                        help="Process only dates in range e.g. 20251201 20260101")
    parser.add_argument('--dry-run',    action='store_true', help="Process but don't write output files")
    parser.add_argument('--stats-only', action='store_true', help="Print stats from existing outputs and exit")
    args = parser.parse_args()

    logger = setup_logging(args.stats_dir)

    # Stats-only mode
    if args.stats_only:
        if not os.path.exists(args.stats_dir):
            logger.error(f"Stats dir not found: {args.stats_dir}")
            sys.exit(1)
        all_stats = []
        for f in sorted(Path(args.stats_dir).glob('*_stats.json')):
            with open(f) as jf:
                all_stats.append(json.load(jf))
        print_aggregate_stats(all_stats, logger)
        return

    # Find input files
    input_dir = Path(args.input_dir)
    if not input_dir.exists():
        logger.error(f"Input dir not found: {input_dir}")
        sys.exit(1)

    files = sorted(input_dir.glob('*.mbo.dbn.zst'))
    logger.info(f"Found {len(files)} .mbo.dbn.zst files in {input_dir}")

    # Date filter
    if args.date_range:
        start_d, end_d = args.date_range
        def in_range(fp):
            d = fp.name.replace('glbx-mdp3-', '').replace('.mbo.dbn.zst', '')
            return start_d <= d <= end_d
        files = [f for f in files if in_range(f)]
        logger.info(f"After date filter [{start_d}–{end_d}]: {len(files)} files")

    if not files:
        logger.error("No files to process")
        sys.exit(1)

    # Create output directories
    if not args.dry_run:
        os.makedirs(args.output_dir, exist_ok=True)
        os.makedirs(args.stats_dir, exist_ok=True)

    logger.info(f"Output dir: {args.output_dir}")
    logger.info(f"Stats dir:  {args.stats_dir}")
    logger.info(f"Workers:    {args.workers}")
    logger.info(f"Dry run:    {args.dry_run}")
    logger.info("")

    # Process files — sequential (parallel would need spawn method on Linux)
    # With 4 workers using ProcessPoolExecutor we get ~4x speedup
    all_stats = []
    t_total = time.time()

    if args.workers == 1:
        for fp in files:
            s = process_file(str(fp), args.output_dir, args.stats_dir, args.dry_run)
            all_stats.append(s)
    else:
        from concurrent.futures import ProcessPoolExecutor, as_completed
        from functools import partial

        process_fn = partial(process_file,
                             output_dir=args.output_dir,
                             stats_dir=args.stats_dir,
                             dry_run=args.dry_run)

        with ProcessPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(process_fn, str(fp)): fp for fp in files}
            for future in as_completed(futures):
                fp = futures[future]
                try:
                    s = future.result()
                    all_stats.append(s)
                    status = s.get('status', '?')
                    n = s.get('n_rth_events', 0)
                    logger.info(f"  Completed: {fp.name} → {status}, {n:,} RTH events")
                except Exception as e:
                    logger.error(f"  FAILED: {fp.name}: {e}")
                    all_stats.append({'date': fp.name, 'status': 'error', 'error': str(e)})

    elapsed_total = time.time() - t_total
    ok_count = sum(1 for s in all_stats if s.get('status') == 'ok')
    skip_count = sum(1 for s in all_stats if s.get('status') == 'skipped')
    err_count = sum(1 for s in all_stats if s.get('status') == 'error')

    logger.info("")
    logger.info(f"Pipeline complete: {ok_count} ok, {skip_count} skipped, {err_count} errors in {elapsed_total:.0f}s")

    print_aggregate_stats(all_stats, logger)

    # Save aggregate stats
    if not args.dry_run:
        agg_path = os.path.join(args.stats_dir, 'aggregate_stats.json')
        with open(agg_path, 'w') as f:
            json.dump({
                'generated': datetime.now().isoformat(),
                'n_files': len(files),
                'n_ok': ok_count,
                'n_skipped': skip_count,
                'n_error': err_count,
                'per_day': all_stats,
            }, f, indent=2)
        logger.info(f"Aggregate stats → {agg_path}")


if __name__ == '__main__':
    main()
