#!/usr/bin/env python3
"""
process_mbo_for_trades.py — Process raw MBO data for champion trade dates only.

Reads raw .dbn.zst files, extracts RTH events, saves as .npz.
Only processes dates that have champion strategy trades.

Run on Jupiter (where raw MBO files live).
"""

import os, sys, json, time, logging
from pathlib import Path
from datetime import datetime
from collections import Counter
from multiprocessing import Pool
import numpy as np
import pandas as pd

try:
    import databento as dbn
except ImportError:
    print("ERROR: databento not installed. Run: pip install databento")
    sys.exit(1)

# ── Paths ──
RAW_DIR = Path("/home/jupiter/Lvl3Quant/data/raw/mbo")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events")
OUT_DIR.mkdir(parents=True, exist_ok=True)

# MBO event pipeline constants (from mbo_event_pipeline.py)
TICK_SIZE_FIXED = 250_000_000   # 0.25 * 1e9
TICK_SIZE_FLOAT = 0.25

ACTION_MAP = {'A': 0, 'C': 1, 'M': 2, 'T': 3, 'F': 4}
SIDE_MAP = {'B': 0, 'A': 1, 'N': 2}

# RTH window: 09:30–16:00 ET = 13:30–20:00 UTC (loose: up to 21:00)
RTH_START_NS = 13 * 3600 * 10**9 + 30 * 60 * 10**9  # 13:30 UTC
RTH_END_NS = 21 * 3600 * 10**9  # 21:00 UTC

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [MBO-PROC] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
log = logging.getLogger(__name__)


def get_trade_dates():
    """Get unique dates that have champion strategy trades.
    Only process the 156 OOT dates (Sept 2025 - Apr 2026) to save time."""
    # OOT dates from multi_scale_combo_v1 daily_bias
    oot_dates = [
        "20250909","20250910","20250911","20250912","20250915","20250916","20250917",
        "20250918","20250919","20250922","20250923","20250924","20250925","20250926",
        "20250929","20250930","20251001","20251002","20251003","20251006","20251007",
        "20251008","20251009","20251010","20251013","20251014","20251015","20251016",
        "20251017","20251020","20251021","20251022","20251023","20251024","20251027",
        "20251028","20251029","20251030","20251031","20251103","20251104","20251105",
        "20251106","20251107","20251110","20251111","20251112","20251113","20251114",
        "20251117","20251118","20251119","20251120","20251121","20251124","20251125",
        "20251126","20251127","20251128","20251201","20251202","20251203","20251204",
        "20251205","20251208","20251209","20251210","20251211","20251212","20251215",
        "20251216","20251217","20251218","20251219","20251222","20251223","20251224",
        "20251226","20251229","20251230","20251231","20260102","20260105","20260106",
        "20260107","20260108","20260109","20260112","20260113","20260114","20260115",
        "20260116","20260119","20260120","20260121","20260122","20260123","20260126",
        "20260127","20260128","20260129","20260130","20260202","20260203","20260204",
        "20260205","20260206","20260209","20260210","20260211","20260212","20260213",
        "20260216","20260217","20260218","20260219","20260220","20260223","20260224",
        "20260225","20260226","20260227","20260302","20260303","20260304","20260305",
        "20260306","20260309","20260310","20260311","20260312","20260313","20260316",
        "20260317","20260318","20260319","20260401","20260402","20260406","20260407",
        "20260408","20260409","20260410","20260413","20260414","20260415","20260416",
        "20260417","20260420","20260421","20260422","20260423","20260424","20260427",
        "20260428","20260429",
    ]
    # Filter to only dates where we have raw files
    available = set()
    for f in RAW_DIR.glob("*.dbn.zst"):
        parts = f.stem.split('-')
        if len(parts) >= 3:
            date_str = parts[2].split('.')[0]
            available.add(date_str)

    dates = [d for d in oot_dates if d in available]
    log.info(f"OOT dates: {len(oot_dates)}, with raw MBO: {len(dates)}")
    return dates


def process_single_day(date_str):
    """Process a single day's raw MBO file into events."""
    out_path = OUT_DIR / f"{date_str}.npz"
    if out_path.exists():
        log.info(f"  {date_str}: already processed, skipping")
        return True

    # Find raw file
    raw_pattern = f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    raw_path = RAW_DIR / raw_pattern
    if not raw_path.exists():
        log.warning(f"  {date_str}: raw file not found")
        return False

    try:
        store = dbn.DBNStore.from_file(str(raw_path))
        df = store.to_df()

        if len(df) == 0:
            log.warning(f"  {date_str}: empty file")
            return False

        # Find dominant instrument (ES front month)
        if 'instrument_id' in df.columns:
            instr_counts = df['instrument_id'].value_counts()
            dominant_id = instr_counts.index[0]
            df = df[df['instrument_id'] == dominant_id]

        # Get timestamps
        if 'ts_event' in df.columns:
            ts_col = 'ts_event'
        elif df.index.name == 'ts_event':
            df = df.reset_index()
            ts_col = 'ts_event'
        else:
            ts_col = df.columns[0]

        timestamps = df[ts_col].astype(np.int64)

        # Filter to RTH
        midnight = (timestamps // (24 * 3600 * 10**9)) * (24 * 3600 * 10**9)
        time_of_day = timestamps - midnight
        rth_mask = (time_of_day >= RTH_START_NS) & (time_of_day <= RTH_END_NS)
        df = df[rth_mask].copy()
        timestamps = timestamps[rth_mask].values

        if len(df) < 100:
            log.warning(f"  {date_str}: too few RTH events ({len(df)})")
            return False

        # Build LOB for spread calculation (simplified)
        # Track best bid/ask from the order book
        prices = df['price'].values if 'price' in df.columns else np.zeros(len(df))
        actions = df['action'].values if 'action' in df.columns else np.full(len(df), 'A')
        sides = df['side'].values if 'side' in df.columns else np.full(len(df), 'N')
        sizes = df['size'].values if 'size' in df.columns else np.ones(len(df))

        # Convert prices from fixed-point
        prices_float = prices.astype(np.float64) / 1e9 if prices.max() > 1e6 else prices.astype(np.float64)

        # Build event feature vectors — VECTORIZED for speed
        n = len(df)
        events = np.zeros((n, 6), dtype=np.float32)

        # [0] time_delta_log
        ts_diff = np.diff(timestamps, prepend=timestamps[0])
        ts_diff_ms = ts_diff.astype(np.float64) / 1e6
        ts_diff_ms = np.clip(ts_diff_ms, 0, 10000)
        events[:, 0] = np.log1p(ts_diff_ms)

        # [1] event_type_id — VECTORIZED via pandas map
        action_series = pd.Series(actions).str.strip().str.upper()
        action_map_series = action_series.map(ACTION_MAP).fillna(2)
        events[:, 1] = action_map_series.values.astype(np.float32)

        # [2] side_id — VECTORIZED via pandas map
        side_series = pd.Series(sides).str.strip().str.upper()
        side_map_series = side_series.map(SIDE_MAP).fillna(2)
        events[:, 2] = side_map_series.values.astype(np.float32)

        # Precompute action/side arrays for fast LOB tracking
        action_arr = action_map_series.values.astype(np.int8)  # 0=A,1=C,2=M,3=T,4=F
        side_arr = side_map_series.values.astype(np.int8)  # 0=B,1=A,2=N

        # [3] price_rel_ticks — use trade prices as mid proxy (much faster)
        # Instead of full LOB tracking, use a rolling mid from trade prices
        trade_mask = (action_arr == 3) | (action_arr == 4)
        bid_trade_mask = trade_mask & (side_arr == 0)
        ask_trade_mask = trade_mask & (side_arr == 1)

        # Forward-fill last trade prices per side to get running BBO estimate
        bid_prices = np.where(bid_trade_mask, prices_float, np.nan)
        ask_prices = np.where(ask_trade_mask, prices_float, np.nan)

        # Forward fill
        last_bid = np.nan
        last_ask = np.nan
        for i in range(n):
            if not np.isnan(bid_prices[i]):
                last_bid = bid_prices[i]
            else:
                bid_prices[i] = last_bid
            if not np.isnan(ask_prices[i]):
                last_ask = ask_prices[i]
            else:
                ask_prices[i] = last_ask

        # Mid price from trade-based BBO
        has_both = ~np.isnan(bid_prices) & ~np.isnan(ask_prices)
        mid_prices = np.where(has_both, (bid_prices + ask_prices) / 2,
                     np.where(~np.isnan(bid_prices), bid_prices,
                     np.where(~np.isnan(ask_prices), ask_prices, prices_float)))

        price_rel_ticks = (prices_float - mid_prices) / TICK_SIZE_FLOAT
        price_rel_ticks = np.clip(price_rel_ticks, -50, 50)
        events[:, 3] = price_rel_ticks

        # [4] qty_log
        qty_arr = sizes.astype(np.float64)
        events[:, 4] = np.log1p(np.clip(qty_arr, 0, 10000))

        # [5] spread_ticks — from trade-based BBO
        spread = np.where(has_both,
                         (ask_prices - bid_prices) / TICK_SIZE_FLOAT,
                         1.0)  # default 1 tick
        spread = np.clip(spread, 0, 20)
        events[:, 5] = spread.astype(np.float32)

        # Save
        np.savez_compressed(
            str(out_path),
            events=events,
            timestamps=timestamps,
            metadata={
                'date': date_str,
                'n_events': n,
                'dominant_instrument': int(dominant_id) if 'instrument_id' in df.columns else 0,
            }
        )

        log.info(f"  {date_str}: {n:,} RTH events processed")
        return True

    except Exception as e:
        log.error(f"  {date_str}: FAILED — {e}")
        return False


def main():
    start = time.time()
    log.info("Processing raw MBO files for champion trade dates")

    dates = get_trade_dates()
    log.info(f"Total dates to process: {len(dates)}")

    # Check which are already done
    done = [d for d in dates if (OUT_DIR / f"{d}.npz").exists()]
    todo = [d for d in dates if not (OUT_DIR / f"{d}.npz").exists()]
    log.info(f"Already done: {len(done)}, todo: {len(todo)}")

    # Use multiprocessing for speed (4 workers on Jupiter's CPU)
    n_workers = 4
    log.info(f"Processing with {n_workers} workers")

    success = 0
    fail = 0
    with Pool(n_workers) as pool:
        results = pool.map(process_single_day, todo)
    success = sum(1 for r in results if r)
    fail = sum(1 for r in results if not r)

    elapsed = time.time() - start
    log.info(f"\nDone in {elapsed:.0f}s. Success: {success}, Failed: {fail}")
    log.info(f"Total events files: {len(list(OUT_DIR.glob('*.npz')))}")


if __name__ == '__main__':
    main()
