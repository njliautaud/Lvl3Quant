#!/usr/bin/env python3
"""
Finish v16 extraction — extract only the remaining uncached days.
"""
import sys, os, time, gc
import numpy as np
sys.path.insert(0, '/home/jupiter/Lvl3Quant/engines')

MBO_DIR = '/home/jupiter/Lvl3Quant/data/raw/mbo'
PRED_DIR = '/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_4_2_fixedmtl/oot_47day_perdate'
V16_CACHE = '/home/jupiter/Lvl3Quant/output/tick_replay_v16/day_cache'

# Import the extraction function from v16
sys.path.insert(0, '/home/jupiter/Lvl3Quant/scripts')

def extract_day_trade_bbo(mbo_path, date_key):
    """Extract trade-based BBO from MBO data (same logic as v16)."""
    import databento as db

    store = db.DBNStore.from_file(mbo_path)

    # Determine front-month contract
    from collections import Counter
    symbol_counts = Counter()
    count = 0
    for rec in store:
        if hasattr(rec, 'instrument_id'):
            sym = getattr(rec, 'symbol', '') or ''
            if sym.startswith('ES') and len(sym) <= 5:
                symbol_counts[sym] += 1
        count += 1
        if count > 100000:
            break

    if not symbol_counts:
        print(f"  {date_key}: no ES symbols found")
        return None

    front_month = symbol_counts.most_common(1)[0][0]

    # Re-read and extract
    store = db.DBNStore.from_file(mbo_path)

    ts_list = []
    is_trade_list = []
    trade_price_list = []
    trade_side_list = []
    bbo_bid_list = []
    bbo_ask_list = []

    current_bid = 0.0
    current_ask = 0.0
    n_events = 0
    n_valid = 0
    n_good_spread = 0

    for rec in store:
        sym = getattr(rec, 'symbol', '') or ''
        if sym != front_month:
            continue

        ts = rec.ts_event
        action = getattr(rec, 'action', '')
        side = getattr(rec, 'side', '')
        price = rec.price / 1e9 if hasattr(rec, 'price') else 0

        is_trade = (action == 'T')

        # Update BBO from trades
        if is_trade and price > 0:
            if side == 'A':  # Aggressive seller hit bid
                current_bid = price
                if current_ask <= 0 or current_ask < price:
                    current_ask = price + 0.25
            elif side == 'B':  # Aggressive buyer lifted ask
                current_ask = price
                if current_bid <= 0 or current_bid > price:
                    current_bid = price - 0.25

        ts_list.append(ts)
        is_trade_list.append(is_trade)
        trade_price_list.append(price if is_trade else 0)
        trade_side_list.append(side if is_trade else '')
        bbo_bid_list.append(current_bid)
        bbo_ask_list.append(current_ask)

        n_events += 1
        if current_bid > 0 and current_ask > 0:
            n_valid += 1
            spread = (current_ask - current_bid) / 0.25
            if 0 <= spread <= 2:
                n_good_spread += 1

        if n_events % (len(ts_list) // 10 + 1) == 0:
            pct = n_events / max(n_events, 1) * 100

    if n_events == 0:
        return None

    print(f"    {date_key}: filtered to {front_month} ({n_events:,} events)")

    # Save to cache
    cache_path = os.path.join(V16_CACHE, f'{date_key}_extracted.npz')

    # Convert side to bytes for storage
    side_arr = np.array(trade_side_list, dtype='U1')

    np.savez_compressed(cache_path,
        n_events=np.array(n_events),
        ts_event=np.array(ts_list, dtype=np.int64),
        is_trade=np.array(is_trade_list, dtype=np.bool_),
        trade_price=np.array(trade_price_list, dtype=np.float64),
        trade_side=side_arr,
        bbo_bid=np.array(bbo_bid_list, dtype=np.float64),
        bbo_ask=np.array(bbo_ask_list, dtype=np.float64),
    )

    valid_pct = n_valid / n_events * 100
    good_pct = n_good_spread / n_events * 100
    print(f"    {date_key}: BBO valid {valid_pct:.1f}%, good spread {good_pct:.1f}%")

    del ts_list, is_trade_list, trade_price_list, trade_side_list, bbo_bid_list, bbo_ask_list
    gc.collect()

    return cache_path


def main():
    print("=" * 70)
    print("V16 FINISH — Extract remaining uncached days")
    print("=" * 70)

    # Find all prediction dates
    pred_dates = set()
    for f in os.listdir(PRED_DIR):
        if f.startswith('oot_') and f.endswith('.npz'):
            pred_dates.add(f.replace('oot_', '').replace('.npz', ''))

    # Find all MBO dates
    import glob
    mbo_files = {}
    for f in glob.glob(os.path.join(MBO_DIR, 'glbx-mdp3-*.mbo.dbn.zst')):
        date8 = os.path.basename(f).split('-')[2].split('.')[0]
        if date8 in pred_dates:
            mbo_files[date8] = f

    # Find already cached
    cached = set()
    for f in os.listdir(V16_CACHE):
        if f.endswith('_extracted.npz'):
            cached.add(f.replace('_extracted.npz', ''))

    # Remaining
    remaining = sorted(set(mbo_files.keys()) - cached)
    print(f"  Total matched: {len(mbo_files)}")
    print(f"  Already cached: {len(cached)}")
    print(f"  Remaining: {len(remaining)}")
    print(f"  Dates: {remaining}")

    for date in remaining:
        print(f"\n  Extracting {date}...")
        t0 = time.time()
        result = extract_day_trade_bbo(mbo_files[date], date)
        elapsed = time.time() - t0
        if result:
            print(f"    Done in {elapsed:.0f}s")
        else:
            print(f"    FAILED")
        gc.collect()

    # Final count
    final_cached = len([f for f in os.listdir(V16_CACHE) if f.endswith('_extracted.npz')])
    print(f"\n  DONE. Total cached days: {final_cached}")


if __name__ == '__main__':
    main()
