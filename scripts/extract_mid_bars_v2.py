#!/home/nick/miniconda3/envs/py311-train/bin/python3
"""
Extract 100ms mid-price bars from MBO data using TRADE PRICES.
v2: Uses trade prints instead of LOB tracking (simpler, more robust).

In liquid ES markets during RTH, the spread is almost always 1 tick (0.25 pts).
Trade prices alternate between bid and ask. The mid is well-approximated as
the last trade price ± 0.125 pts. For our purposes (measuring P&L at a given
bar), the last trade price IS the mid to within half a tick.

Authorization: HC #420 — user's own codebase.
"""

import argparse
import sys
import time as time_mod
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

try:
    import databento as dbn
except ImportError:
    print("ERROR: need databento. Use /home/nick/miniconda3/envs/py311-train/bin/python3")
    sys.exit(1)

MBO_DIRS = [
    Path("/home/nick/Lvl3Quant/data/raw/mbo"),
    Path("/home/jupiter/Lvl3Quant/data/raw/mbo"),
]
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/data/derived/mid_price_bars")
N_BARS = 234_000
BAR_NS = 100_000_000  # 100ms


def compute_rth_ns(date_str):
    """Return (rth_open_ns, rth_close_ns) for a given date in EDT."""
    d = datetime.strptime(date_str, "%Y%m%d")
    midnight = datetime(d.year, d.month, d.day, tzinfo=timezone.utc)
    # EDT (Mar-Nov): 9:30 ET = 13:30 UTC
    # All our dates (March-April 2026) are EDT
    month = d.month
    if month >= 3 and month <= 10:
        rth_open = midnight + timedelta(hours=13, minutes=30)
    else:
        rth_open = midnight + timedelta(hours=14, minutes=30)
    rth_open_ns = int(rth_open.timestamp() * 1_000_000_000)
    rth_close_ns = rth_open_ns + N_BARS * BAR_NS
    return rth_open_ns, rth_close_ns


def detect_dominant(filepath, date_str, sample=100000):
    """Detect the instrument the fill_sim uses (near-expiry ES front month).

    The MBO data contains multiple ES contracts. Near quarterly expiry (3rd Friday),
    the NEXT quarter has more volume but the fill_sim trades the CURRENT expiring contract.
    We identify the correct instrument by finding the one with the LOWEST price among
    the top-2 most-traded instruments (the front month is always cheaper due to
    basis/cost of carry).
    """
    store = dbn.DBNStore.from_file(filepath)
    inst_trades = Counter()
    inst_prices = {}

    for i, r in enumerate(store):
        if i >= sample:
            break
        action = getattr(r, 'action', '')
        is_trade = (str(action) == 'T' or
                   (hasattr(action, 'name') and action.name == 'TRADE'))

        if is_trade and hasattr(r, 'price') and r.price > 0:
            px = r.price * 1e-9
            if 1000 < px < 20000:
                inst_trades[r.instrument_id] += 1
                if r.instrument_id not in inst_prices:
                    inst_prices[r.instrument_id] = []
                inst_prices[r.instrument_id].append(px)

    if not inst_trades:
        return -1

    # Get top instruments by trade count
    top_insts = inst_trades.most_common(5)
    print(f"    Top instruments by trades:")
    for iid, count in top_insts[:3]:
        avg_px = sum(inst_prices[iid]) / len(inst_prices[iid]) if iid in inst_prices else 0
        print(f"      id={iid}: {count} trades, avg_price={avg_px:.2f}")

    # The fill_sim uses the LOWEST-PRICED of the top-2 instruments
    # (near-expiry contract is cheaper due to basis convergence)
    if len(top_insts) >= 2:
        top2 = top_insts[:2]
        avg_prices = {}
        for iid, _ in top2:
            if iid in inst_prices and inst_prices[iid]:
                avg_prices[iid] = sum(inst_prices[iid]) / len(inst_prices[iid])

        if avg_prices:
            # Pick the one with the LOWER average price
            chosen = min(avg_prices, key=avg_prices.get)
            print(f"    → Choosing {chosen} (lowest price = near-month)")
            return chosen

    return top_insts[0][0]


def extract_trade_mid_bars(mbo_file, date_str):
    """Extract 100ms bars using trade prices from dominant instrument."""
    rth_open_ns, rth_close_ns = compute_rth_ns(date_str)

    # Detect dominant instrument
    dominant = detect_dominant(mbo_file, date_str)
    if dominant == -1:
        raise ValueError("No dominant instrument found")

    mid_prices = np.zeros(N_BARS, dtype=np.float32)
    ts_ns_arr = np.arange(N_BARS, dtype=np.int64) * BAR_NS + rth_open_ns

    # Track last known trade price and recent high/low bid/ask for mid estimation
    last_trade_price = 0.0
    last_bar_filled = -1
    n_trades = 0
    n_records = 0
    pct_printed = set()

    store = dbn.DBNStore.from_file(mbo_file)
    total_est = 50_000_000  # rough estimate for progress

    for r in store:
        n_records += 1

        # Progress
        pct = n_records * 100 // total_est
        if pct % 10 == 0 and pct not in pct_printed and pct <= 100:
            print(f" {pct}%", end="", flush=True)
            pct_printed.add(pct)

        # Filter to dominant instrument
        if r.instrument_id != dominant:
            continue

        ts = r.ts_recv
        if ts < rth_open_ns or ts >= rth_close_ns:
            # Pre-RTH: still track trades for warmup
            if hasattr(r, 'action'):
                action = getattr(r, 'action', '')
                is_trade = (str(action) == 'T' or
                           (hasattr(action, 'name') and action.name == 'TRADE'))
                if is_trade and hasattr(r, 'price') and r.price > 0:
                    px = r.price * 1e-9
                    if 1000 < px < 20000:
                        last_trade_price = px
            continue

        bar_idx = int((ts - rth_open_ns) // BAR_NS)
        bar_idx = min(bar_idx, N_BARS - 1)

        # Check if this is a trade
        action = getattr(r, 'action', '')
        is_trade = (str(action) == 'T' or
                   (hasattr(action, 'name') and action.name == 'TRADE'))

        if is_trade and hasattr(r, 'price') and r.price > 0:
            px = r.price * 1e-9
            if 1000 < px < 20000:
                last_trade_price = px
                n_trades += 1

        # Fill bars with last trade price
        if last_trade_price > 0:
            # Forward fill from last_bar_filled+1 to bar_idx
            for b in range(max(last_bar_filled + 1, 0), bar_idx + 1):
                mid_prices[b] = last_trade_price
            last_bar_filled = bar_idx

    # Forward fill remaining bars
    if last_bar_filled >= 0 and last_bar_filled < N_BARS - 1:
        mid_prices[last_bar_filled + 1:] = mid_prices[last_bar_filled]

    return mid_prices, ts_ns_arr, n_trades


def find_mbo_file(date_str):
    """Find MBO file across candidate directories."""
    for d in MBO_DIRS:
        f = d / f"glbx-mdp3-{date_str}.mbo.dbn.zst"
        if f.exists():
            return str(f)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('dates', nargs='*', help='Specific dates (YYYYMMDD)')
    parser.add_argument('--force', action='store_true', help='Overwrite existing')
    args = parser.parse_args()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Find all MBO files
    all_files = []
    for d in MBO_DIRS:
        if d.exists():
            all_files.extend(d.glob("glbx-mdp3-*.mbo.dbn.zst"))

    dates_avail = sorted(set(
        f.stem.replace("glbx-mdp3-", "").split(".")[0] for f in all_files
    ))
    print(f"Found {len(dates_avail)} MBO files")

    if args.dates:
        dates = [d for d in args.dates if d in set(dates_avail)]
    else:
        dates = dates_avail

    processed = 0
    errors = 0

    for date_str in dates:
        out_file = OUTPUT_DIR / f"{date_str}.npz"
        if out_file.exists() and not args.force:
            continue

        mbo_file = find_mbo_file(date_str)
        if not mbo_file:
            continue

        print(f"  {date_str}: extracting...", end="", flush=True)
        t0 = time_mod.time()
        try:
            mid, ts, n_trades = extract_trade_mid_bars(mbo_file, date_str)
            nonzero = np.count_nonzero(mid)
            elapsed = time_mod.time() - t0

            # Sanity check
            if nonzero < N_BARS * 0.5:
                print(f" WARNING: only {nonzero}/{N_BARS} bars filled")
            if n_trades < 100:
                print(f" WARNING: only {n_trades} trades")

            # Save (overwrite)
            np.savez_compressed(str(out_file),
                               mid_prices=mid,
                               bid_prices=mid - 0.125,  # approximate bid = mid - half tick
                               ask_prices=mid + 0.125,  # approximate ask = mid + half tick
                               ts_ns=ts)
            print(f" done — {nonzero}/{N_BARS} bars, {n_trades} trades, "
                  f"price=[{mid[mid>0].min():.2f}, {mid.max():.2f}], {elapsed:.0f}s")
            processed += 1
        except Exception as e:
            print(f" ERROR: {e}")
            errors += 1

    print(f"\nDone: {processed} processed, {errors} errors")


if __name__ == '__main__':
    main()
