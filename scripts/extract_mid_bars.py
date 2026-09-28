#!/home/nick/miniconda3/envs/py311-train/bin/python3
"""
Extract 100ms mid-price bars from raw MBO data.
================================================

Produces 234,000 mid-price bars per RTH day (9:30-16:00 ET, 100ms intervals)
aligned with CNN-Mamba v2 prediction timestamps for pressure exit backtesting.

Reads Databento MBO .dbn.zst files, maintains a Level-1 LOB tracker to derive
best bid/ask, then samples mid = (best_bid + best_ask) / 2 at 100ms boundaries.

Output: /home/nick/Lvl3Quant/data/derived/mid_price_bars/YYYYMMDD.npz
  Arrays: mid_prices (float32, 234000), bid_prices (float32, 234000),
          ask_prices (float32, 234000), ts_ns (int64, 234000)

Usage:
    python extract_mid_bars.py                  # process all available dates
    python extract_mid_bars.py 20260301 20260302  # specific dates
    python extract_mid_bars.py --force 20260301   # overwrite existing

Authorization: HC #420 — user's own legitimate quant research codebase.
"""

import argparse
import os
import sys
import time as time_mod
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

try:
    import databento as dbn
except ImportError:
    print("ERROR: databento library not found. Run with: "
          "/home/nick/miniconda3/envs/py311-train/bin/python3", file=sys.stderr)
    sys.exit(1)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

MBO_DIRS = [
    Path("/home/nick/Lvl3Quant/data/raw/mbo"),
    Path("/home/jupiter/Lvl3Quant/data/raw/mbo"),
    Path("/home/jupiter/Lvl3Quant/data/raw_mbo"),
]
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/data/derived/mid_price_bars")

N_BARS = 234_000          # 23,400s RTH × 10 bars/s = 234,000 bars
BAR_NS = 100_000_000      # 100ms in nanoseconds

# ES tick size in Databento fixed-point (1e9 scale)
TICK_SIZE_FIXED = 250_000_000   # 0.25 index pts × 1e9
PRICE_SCALE = 1e-9              # fixed-point → actual price

# RTH boundaries (EDT: 9:30 ET = 13:30 UTC, 16:00 ET = 20:00 UTC)
# For EST dates (Nov-Mar): 9:30 ET = 14:30 UTC — handled by compute_rth_open_ns()


# ─────────────────────────────────────────────────────────────────────────────
# LOB Tracker (Level-1) — adapted from build_minute_bars_v1.py
# ─────────────────────────────────────────────────────────────────────────────

class LOBTracker:
    """Lightweight Level-1 order book tracker for BBO / mid-price."""

    __slots__ = ('_bid_levels', '_ask_levels', '_best_bid', '_best_ask',
                 '_last_trade_price', '_event_count')

    def __init__(self):
        self._bid_levels = {}
        self._ask_levels = {}
        self._best_bid = 0
        self._best_ask = 0
        self._last_trade_price = 0
        self._event_count = 0

    def _recompute(self):
        valid_bids = [k for k, v in self._bid_levels.items() if v > 0]
        valid_asks = [k for k, v in self._ask_levels.items() if v > 0]
        self._best_bid = max(valid_bids) if valid_bids else 0
        self._best_ask = min(valid_asks) if valid_asks else 0

    def _prune_levels(self):
        """Keep memory bounded by removing distant levels."""
        mid = (self._best_bid + self._best_ask) / 2.0 if (
            self._best_bid > 0 and self._best_ask > 0) else self._last_trade_price
        if mid <= 0:
            return
        radius = 50 * TICK_SIZE_FIXED
        lo, hi = mid - radius, mid + radius
        self._bid_levels = {k: v for k, v in self._bid_levels.items()
                           if lo <= k <= hi and v > 0}
        self._ask_levels = {k: v for k, v in self._ask_levels.items()
                           if lo <= k <= hi and v > 0}

    def process(self, action: str, side: str, price: int, qty: int):
        """Update LOB with one MBO event. Returns (best_bid, best_ask) in fixed-point."""
        INVALID_PRICE = 9_223_372_036_854_775_807
        self._event_count += 1

        if action == 'R':
            self._bid_levels.clear()
            self._ask_levels.clear()
            self._best_bid = 0
            self._best_ask = 0
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
            # Modify: set new qty (could be up or down)
            if side == 'B':
                self._bid_levels[price] = self._bid_levels.get(price, 0) + qty
            elif side == 'A':
                self._ask_levels[price] = self._ask_levels.get(price, 0) + qty
        elif action in ('T', 'F'):
            self._last_trade_price = price
            # Trade/Fill: aggressive side consumes passive side
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
    def mid_float(self) -> float:
        """Mid-price in actual dollars (float). 0.0 if BBO not valid."""
        if self._best_bid > 0 and self._best_ask > 0 and self._best_ask > self._best_bid:
            return (self._best_bid + self._best_ask) / 2.0 * PRICE_SCALE
        elif self._last_trade_price > 0:
            return self._last_trade_price * PRICE_SCALE
        return 0.0

    @property
    def bid_float(self) -> float:
        return self._best_bid * PRICE_SCALE if self._best_bid > 0 else 0.0

    @property
    def ask_float(self) -> float:
        return self._best_ask * PRICE_SCALE if self._best_ask > 0 else 0.0


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def is_edt(date_str: str) -> bool:
    """Rough EDT check: Mar second Sun through Nov first Sun."""
    d = datetime.strptime(date_str, "%Y%m%d")
    # EDT roughly applies Mar 10 – Nov 3 range; good enough for trading days
    if d.month >= 4 and d.month <= 10:
        return True
    if d.month >= 12 or d.month <= 2:
        return False
    # March/November: approximate
    if d.month == 3:
        return d.day >= 10  # second Sunday is 8-14; conservative
    if d.month == 11:
        return d.day < 4
    return True


def compute_rth_open_ns(date_str: str) -> int:
    """Compute RTH open (9:30 ET) as epoch nanoseconds."""
    d = datetime.strptime(date_str, "%Y%m%d")
    midnight_utc = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    if is_edt(date_str):
        # EDT: 9:30 ET = 13:30 UTC
        rth_open = midnight_utc + timedelta(hours=13, minutes=30)
    else:
        # EST: 9:30 ET = 14:30 UTC
        rth_open = midnight_utc + timedelta(hours=14, minutes=30)
    return int(rth_open.timestamp() * 1_000_000_000)


def find_mbo_files() -> list:
    """Find all MBO .dbn.zst files across known directories."""
    files = []
    for d in MBO_DIRS:
        if d.exists():
            files.extend(sorted(d.glob("glbx-mdp3-*.mbo.dbn.zst")))
    return sorted(set(files), key=lambda p: p.name)


def extract_date(filepath: Path) -> str:
    """Extract YYYYMMDD from filename like glbx-mdp3-20260301.mbo.dbn.zst"""
    name = filepath.name  # glbx-mdp3-20260301.mbo.dbn.zst
    # Split on 'glbx-mdp3-' prefix and '.mbo' suffix
    date_part = name.replace('glbx-mdp3-', '').split('.')[0]
    return date_part


def detect_dominant_instrument(filepath: str, sample_size: int = 20000) -> int:
    """Detect the front-month contract (most active instrument_id)."""
    import itertools
    inst_counter = Counter()
    store = dbn.DBNStore.from_file(filepath)
    for r in itertools.islice(store, sample_size):
        if str(r.action) != 'R':
            inst_counter[r.instrument_id] += 1
    if not inst_counter:
        return -1
    return inst_counter.most_common(1)[0][0]


# ─────────────────────────────────────────────────────────────────────────────
# Core extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_mid_bars(mbo_file: str, date_str: str) -> tuple:
    """
    Extract 100ms mid-price bars from one MBO file.

    Returns: (mid_prices, bid_prices, ask_prices, ts_ns) — all numpy arrays
    """
    rth_open_ns = compute_rth_open_ns(date_str)
    rth_close_ns = rth_open_ns + N_BARS * BAR_NS

    # Output arrays
    mid_prices = np.zeros(N_BARS, dtype=np.float32)
    bid_prices = np.zeros(N_BARS, dtype=np.float32)
    ask_prices = np.zeros(N_BARS, dtype=np.float32)
    ts_ns_arr = np.arange(N_BARS, dtype=np.int64) * BAR_NS + rth_open_ns

    # Detect dominant instrument (front-month ES)
    dominant_id = detect_dominant_instrument(mbo_file)
    if dominant_id == -1:
        print(f" no valid records", flush=True)
        return mid_prices, bid_prices, ask_prices, ts_ns_arr

    lob = LOBTracker()
    last_bar = -1
    record_count = 0
    rth_count = 0

    # Pre-RTH warmup: process events before RTH to have valid BBO at open
    # Then sample during RTH
    store = dbn.DBNStore.from_file(mbo_file)

    for r in store:
        action = str(r.action)
        side = str(r.side)
        record_count += 1

        # Filter to dominant instrument only
        if r.instrument_id != dominant_id:
            continue

        # Update LOB for all events (including pre-RTH warmup)
        lob.process(action, side, r.price, r.size)

        ts = r.ts_event

        # Skip pre-RTH (but we already processed the LOB update above)
        if ts < rth_open_ns:
            continue

        # Past RTH close — done
        if ts >= rth_close_ns:
            break

        rth_count += 1
        bar_idx = int((ts - rth_open_ns) // BAR_NS)
        bar_idx = min(bar_idx, N_BARS - 1)

        mid = lob.mid_float
        if mid <= 0:
            continue

        # Forward-fill from last_bar+1 to current bar_idx
        fill_start = max(last_bar + 1, 0)
        if fill_start <= bar_idx:
            mid_prices[fill_start:bar_idx + 1] = mid
            bid_prices[fill_start:bar_idx + 1] = lob.bid_float
            ask_prices[fill_start:bar_idx + 1] = lob.ask_float
            last_bar = bar_idx

        if record_count % 2_000_000 == 0:
            pct = 100.0 * bar_idx / N_BARS if N_BARS > 0 else 0
            print(f" {pct:.0f}%", end="", flush=True)

    # Forward-fill any remaining bars at end of day
    if 0 <= last_bar < N_BARS - 1:
        mid_prices[last_bar + 1:] = mid_prices[last_bar]
        bid_prices[last_bar + 1:] = bid_prices[last_bar]
        ask_prices[last_bar + 1:] = ask_prices[last_bar]

    return mid_prices, bid_prices, ask_prices, ts_ns_arr


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Extract 100ms mid-price bars from MBO data")
    parser.add_argument('dates', nargs='*',
                        help="Specific YYYYMMDD dates to process (default: all)")
    parser.add_argument('--force', action='store_true',
                        help="Overwrite existing output files")
    parser.add_argument('--output-dir', type=str, default=str(OUTPUT_DIR),
                        help=f"Output directory (default: {OUTPUT_DIR})")
    args = parser.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    mbo_files = find_mbo_files()
    print(f"Found {len(mbo_files)} MBO files across {[str(d) for d in MBO_DIRS if d.exists()]}")

    dates_filter = set(args.dates) if args.dates else None
    processed = 0
    skipped = 0
    errors = 0

    for mbo_file in mbo_files:
        date_str = extract_date(mbo_file)

        if dates_filter and date_str not in dates_filter:
            continue

        out_file = out_dir / f"{date_str}.npz"
        if out_file.exists() and not args.force:
            print(f"  {date_str}: already exists, skipping")
            skipped += 1
            continue

        t0 = time_mod.time()
        print(f"  {date_str}: extracting...", end="", flush=True)

        try:
            mid, bid, ask, ts = extract_mid_bars(str(mbo_file), date_str)

            nonzero = np.count_nonzero(mid)
            fill_pct = 100.0 * nonzero / N_BARS

            # Sanity checks
            if nonzero == 0:
                print(f" WARNING: 0 bars with data — skipping save")
                errors += 1
                continue

            if fill_pct < 50:
                print(f" WARNING: only {fill_pct:.1f}% fill — check RTH boundaries")

            # Price sanity: ES should be in ~4000-6500 range (2024-2026)
            valid_mid = mid[mid > 0]
            if len(valid_mid) > 0:
                px_min, px_max = valid_mid.min(), valid_mid.max()
                if px_min < 1000 or px_max > 10000:
                    print(f" WARNING: price range [{px_min:.2f}, {px_max:.2f}] looks wrong")

            np.savez_compressed(
                out_file,
                mid_prices=mid,
                bid_prices=bid,
                ask_prices=ask,
                ts_ns=ts,
            )

            elapsed = time_mod.time() - t0
            print(f" done — {nonzero}/{N_BARS} bars ({fill_pct:.1f}%), "
                  f"price=[{valid_mid.min():.2f}, {valid_mid.max():.2f}], "
                  f"{elapsed:.1f}s")
            processed += 1

        except Exception as e:
            elapsed = time_mod.time() - t0
            print(f" ERROR after {elapsed:.1f}s: {e}")
            errors += 1

    print(f"\nDone: {processed} processed, {skipped} skipped, {errors} errors")


if __name__ == '__main__':
    main()
