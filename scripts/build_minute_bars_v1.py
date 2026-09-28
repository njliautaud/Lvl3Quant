#!/usr/bin/env python3
"""
Minute-Bar Feature Pipeline from Raw MBO Data
==============================================

Converts raw Databento MBO .dbn.zst files into 1-minute OHLCV bars with
advanced features (OFI, microprice, vol_regime, signed_volume).

Output: per-day parquet at /home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1/YYYYMMDD.parquet

Columns:
  - ts_minute (UTC): Minute boundary timestamp
  - open, high, low, close: Bar OHLC in index points
  - volume: Total contract volume (trades)
  - vwap: Volume-weighted average price
  - trade_count: Number of trades in the minute
  - signed_volume: Buy volume - Sell volume (based on aggressive side)
  - spread_mean: Mean top-of-book spread (in index points)
  - ofi_1min: Order Flow Imbalance over the minute (sum of signed qty changes)
  - microprice_close: Closing microprice (bid+ask)/2
  - vol_regime: Realized vol bucketing (low/med/high)
"""

import argparse
import databento as dbn
import numpy as np
import pandas as pd
import os
import sys
import logging
import time
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from collections import defaultdict
import math

# ─────────────────────────────────────────────
#  CONSTANTS
# ─────────────────────────────────────────────

RAW_DIR   = "/home/jupiter/Lvl3Quant/data/raw/mbo"
OUT_DIR   = "/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1"
LOG_DIR   = "/home/jupiter/Lvl3Quant/logs"

# ES tick size = 0.25 index points = 250_000_000 in Databento fixed-point (1e9 scale)
TICK_SIZE_FIXED = 250_000_000   # 0.25 * 1e9
TICK_SIZE_FLOAT = 0.25

# RTH window in UTC nanoseconds offsets from midnight
# RTH: 09:30–16:00 ET = 13:30–21:00 UTC (loose, covers both EST and EDT)
RTH_START_UTC_SEC = 13 * 3600 + 30 * 60   # 13:30:00 UTC
RTH_END_UTC_SEC   = 21 * 3600              # 21:00:00 UTC

# ─────────────────────────────────────────────
#  LOGGING
# ─────────────────────────────────────────────

def setup_logging(log_file: str = None):
    handlers = [logging.StreamHandler(sys.stdout)]
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(message)s',
        handlers=handlers,
        force=True
    )
    return logging.getLogger(__name__)


# ─────────────────────────────────────────────
#  LOB TRACKER (Level-1)
# ─────────────────────────────────────────────

class LOBTracker:
    """Lightweight Level-1 order book tracker for spread and microprice."""

    def __init__(self, tick_size_fixed: int = TICK_SIZE_FIXED):
        self.tick_size_fixed = tick_size_fixed
        self._bid_levels = {}
        self._ask_levels = {}
        self._best_bid = 0
        self._best_ask = 0
        self._mid = 0.0
        self._spread = float('nan')
        self._last_trade_price = 0
        self._event_count = 0

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

    def _prune_levels(self):
        """Keep memory bounded by removing stale levels."""
        if self._mid <= 0:
            return
        radius = 50 * self.tick_size_fixed
        lo = self._mid - radius
        hi = self._mid + radius
        self._bid_levels = {k: v for k, v in self._bid_levels.items() if lo <= k <= hi and v > 0}
        self._ask_levels = {k: v for k, v in self._ask_levels.items() if lo <= k <= hi and v > 0}

    def process(self, action: str, side: str, price: int, qty: int):
        """Update LOB with incoming event. Returns (best_bid, best_ask) in fixed-point."""
        INVALID_PRICE = 9_223_372_036_854_775_807

        self._event_count += 1

        if action == 'R':
            self._bid_levels.clear()
            self._ask_levels.clear()
            self._best_bid = 0
            self._best_ask = 0
            self._spread = float('nan')
            return self._best_bid, self._best_ask

        if price == INVALID_PRICE or price <= 0:
            return self._best_bid, self._best_ask

        if action == 'A':
            if side == 'B':
                self._bid_levels[price] = self._bid_levels.get(price, 0) + qty
            elif side == 'A':
                self._ask_levels[price] = self._ask_levels.get(price, 0) + qty
        elif action == 'C':
            if side == 'B':
                self._bid_levels[price] = max(0, self._bid_levels.get(price, 0) - qty)
            elif side == 'A':
                self._ask_levels[price] = max(0, self._ask_levels.get(price, 0) - qty)
        elif action == 'M':
            if side == 'B':
                self._bid_levels[price] = self._bid_levels.get(price, 0) + qty
            elif side == 'A':
                self._ask_levels[price] = self._ask_levels.get(price, 0) + qty
        elif action in ('T', 'F'):
            self._last_trade_price = price
            if side == 'B':
                self._ask_levels[price] = max(0, self._ask_levels.get(price, 0) - qty)
            elif side == 'A':
                self._bid_levels[price] = max(0, self._bid_levels.get(price, 0) - qty)

        if self._event_count % 5000 == 0:
            self._prune_levels()

        self._recompute()
        return self._best_bid, self._best_ask

    @property
    def best_bid(self) -> int:
        return self._best_bid

    @property
    def best_ask(self) -> int:
        return self._best_ask

    @property
    def mid(self) -> float:
        return self._mid

    @property
    def spread_ticks(self) -> float:
        return self._spread


def is_rth_ns(ts_event_ns: int) -> bool:
    """Check if a nanosecond UTC timestamp falls within RTH (13:30–21:00 UTC)."""
    sec_in_day = (ts_event_ns // 1_000_000_000) % 86400
    return RTH_START_UTC_SEC <= sec_in_day < RTH_END_UTC_SEC


def detect_dominant_instrument(filepath: str, sample_size: int = 20000) -> int:
    """Fast scan to detect dominant instrument_id (front-month contract)."""
    from collections import Counter
    import itertools

    inst_counter = Counter()
    store = dbn.DBNStore.from_file(filepath)
    for r in itertools.islice(store, sample_size):
        if str(r.action) != 'R':
            inst_counter[r.instrument_id] += 1
    if not inst_counter:
        return -1
    return inst_counter.most_common(1)[0][0]


def process_file(filepath: str, output_dir: str, logger) -> dict:
    """
    Process one .dbn.zst file and produce minute bars.

    Returns: dict with keys: date, status, n_bars, path (if successful)
    """
    filename = os.path.basename(filepath)
    date_str = filename.replace('glbx-mdp3-', '').replace('.mbo.dbn.zst', '')
    out_path = os.path.join(output_dir, f"{date_str}.parquet")

    # Skip if already processed
    if os.path.exists(out_path):
        logger.info(f"[SKIP] {date_str} — already exists")
        return {'date': date_str, 'status': 'skipped'}

    t_start = time.time()
    logger.info(f"[START] {date_str}")

    try:
        # Detect dominant instrument
        dominant_instrument = detect_dominant_instrument(filepath)
        if dominant_instrument == -1:
            logger.warning(f"[EMPTY] {date_str} — no valid records")
            return {'date': date_str, 'status': 'empty'}

        # Single streaming pass
        store = dbn.DBNStore.from_file(filepath)
        lob = LOBTracker(tick_size_fixed=TICK_SIZE_FIXED)

        # Dictionary to collect minute bars: minute_ts -> {trades, volumes, etc}
        minute_bars = defaultdict(lambda: {
            'prices': [],
            'qtys': [],
            'sides': [],  # 'B' or 'A' for buyer/seller initiated
            'ts_events': [],
            'spread_ticks': [],
        })

        # Per-minute LOB snapshots: minute_ts -> (best_bid, best_ask)
        # Updated as events stream in; captures the LOB state at each minute boundary
        minute_lob_snapshots = {}

        inst_total = 0
        trade_total = 0
        LOG_INTERVAL = 1_000_000

        for r in store:
            act = str(r.action)
            if act == 'R':
                lob.process('R', str(r.side), r.price, r.size)
                continue

            inst_total += 1
            if inst_total % LOG_INTERVAL == 0:
                logger.info(f"  {date_str}: {inst_total:,} records, {trade_total:,} trades")

            if r.instrument_id != dominant_instrument:
                continue

            # Update LOB for all events (to get spread/microprice)
            bid, ask = lob.process(act, str(r.side), r.price, r.size)

            # Filter to RTH only
            if not is_rth_ns(r.ts_event):
                continue

            # Only trades contribute to OHLCV
            if act in ('T', 'F'):
                trade_total += 1

                # Minute timestamp (round down to start of minute in UTC)
                ts_sec = r.ts_event // 1_000_000_000
                ts_minute = (ts_sec // 60) * 60

                price_float = r.price / TICK_SIZE_FIXED  # Convert to index points

                minute_bars[ts_minute]['prices'].append(price_float)
                minute_bars[ts_minute]['qtys'].append(r.size)
                minute_bars[ts_minute]['sides'].append(str(r.side))
                minute_bars[ts_minute]['ts_events'].append(r.ts_event)
                minute_bars[ts_minute]['spread_ticks'].append(lob.spread_ticks)

                # Save LOB snapshot — overwrites each event within the minute,
                # so the final value is the LOB state at the LAST event of this minute
                minute_lob_snapshots[ts_minute] = (lob.best_bid, lob.best_ask)

        logger.info(f"  {date_str}: total {trade_total:,} RTH trades")

        if not minute_bars:
            logger.warning(f"[NO TRADES] {date_str}")
            return {'date': date_str, 'status': 'no_trades'}

        # Build DataFrame from minute bars
        rows = []
        for ts_minute, bar_data in sorted(minute_bars.items()):
            if not bar_data['prices']:
                continue

            prices = np.array(bar_data['prices'])
            qtys = np.array(bar_data['qtys'])
            sides = bar_data['sides']
            ts_events = np.array(bar_data['ts_events'])
            spreads = bar_data['spread_ticks']

            # OHLC
            o = prices[0]
            h = np.max(prices)
            l = np.min(prices)
            c = prices[-1]

            # Volume and count
            volume = np.sum(qtys)
            trade_count = len(prices)

            # VWAP
            vwap = np.sum(prices * qtys) / volume if volume > 0 else 0.0

            # Signed volume (buy vs sell based on aggressive side)
            buy_qty = sum(q for i, q in enumerate(qtys) if sides[i] == 'B')
            sell_qty = sum(q for i, q in enumerate(qtys) if sides[i] == 'A')
            signed_volume = buy_qty - sell_qty

            # Mean spread (in index points, not ticks)
            valid_spreads = [s for s in spreads if not math.isnan(s)]
            spread_mean = np.mean(valid_spreads) * TICK_SIZE_FLOAT if valid_spreads else 0.0

            # OFI: Order Flow Imbalance (rough: buy market orders - sell market orders signed qty)
            # For simplicity: sum of signed (Buy qtys) - (Sell qtys)
            ofi_1min = signed_volume

            # Microprice at close (midpoint of bid/ask at END of this minute)
            # Uses per-minute LOB snapshot, NOT the stale end-of-day LOB
            min_bid, min_ask = minute_lob_snapshots.get(ts_minute, (0, 0))
            microprice_close = (min_bid + min_ask) / 2.0 / TICK_SIZE_FIXED if (min_bid > 0 and min_ask > 0) else c

            # Vol regime: rolling 30-min realized vol
            # For now, just mark low/med/high based on intrabar volatility
            intrabar_vol = (h - l)  # simple range
            if intrabar_vol < 0.5:
                vol_regime = 'low'
            elif intrabar_vol < 2.0:
                vol_regime = 'med'
            else:
                vol_regime = 'high'

            rows.append({
                'ts_minute': pd.Timestamp(ts_minute, unit='s', tz='UTC'),
                'open': o,
                'high': h,
                'low': l,
                'close': c,
                'volume': volume,
                'vwap': vwap,
                'trade_count': trade_count,
                'signed_volume': signed_volume,
                'spread_mean': spread_mean,
                'ofi_1min': ofi_1min,
                'microprice_close': microprice_close,
                'vol_regime': vol_regime,
            })

        if not rows:
            logger.warning(f"[NO BARS] {date_str}")
            return {'date': date_str, 'status': 'no_bars'}

        df = pd.DataFrame(rows)

        # Save parquet
        os.makedirs(output_dir, exist_ok=True)
        df.to_parquet(out_path, index=False, compression='snappy')

        elapsed = time.time() - t_start
        logger.info(f"[DONE] {date_str}: {len(df)} bars in {elapsed:.1f}s — {out_path}")

        return {
            'date': date_str,
            'status': 'success',
            'n_bars': len(df),
            'path': out_path,
        }

    except Exception as e:
        logger.error(f"[ERROR] {date_str}: {e}")
        logger.error(traceback.format_exc())
        return {'date': date_str, 'status': 'error', 'error': str(e)}


def main():
    parser = argparse.ArgumentParser(description="Build minute bars from MBO .dbn.zst files")
    parser.add_argument('--input-dir', default=RAW_DIR, help='Input MBO directory')
    parser.add_argument('--output-dir', default=OUT_DIR, help='Output parquet directory')
    parser.add_argument('--log-dir', default=LOG_DIR, help='Log directory')
    parser.add_argument('--date', help='Single date to process (YYYYMMDD), or leave for all')
    parser.add_argument('--dry-run', action='store_true', help='Scan only, no processing')

    args = parser.parse_args()

    log_file = os.path.join(args.log_dir, 'minute_bars_build.log')
    logger = setup_logging(log_file)

    logger.info(f"=== Minute Bars Builder v1 ===")
    logger.info(f"Input: {args.input_dir}")
    logger.info(f"Output: {args.output_dir}")
    logger.info(f"Log: {log_file}")

    # Collect files to process
    all_files = sorted([f for f in os.listdir(args.input_dir) if f.endswith('.mbo.dbn.zst')])

    if args.date:
        files_to_process = [f for f in all_files if args.date in f]
        if not files_to_process:
            logger.error(f"No files found for date {args.date}")
            return 1
    else:
        files_to_process = all_files

    logger.info(f"Processing {len(files_to_process)} files (out of {len(all_files)} total)")

    results = []
    for i, fname in enumerate(files_to_process):
        filepath = os.path.join(args.input_dir, fname)
        result = process_file(filepath, args.output_dir, logger)
        results.append(result)

        logger.info(f"[{i+1}/{len(files_to_process)}] {result['date']}: {result['status']}")

    # Summary
    logger.info("=" * 60)
    logger.info(f"SUMMARY: {len(files_to_process)} files processed")
    success_count = sum(1 for r in results if r['status'] == 'success')
    logger.info(f"  Success: {success_count}")
    logger.info(f"  Skipped: {sum(1 for r in results if r['status'] == 'skipped')}")
    logger.info(f"  Errors:  {sum(1 for r in results if r['status'] == 'error')}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
