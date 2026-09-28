#!/usr/bin/env python3
"""
MFE/MAE Within-Horizon Relabeling Pipeline v2
==============================================
Per HC #428 R2 / HC #464 R2: compute Maximum Favorable Excursion (MFE) and
Maximum Adverse Excursion (MAE) within each prediction horizon for every
event in the smart_v3 dataset.

For each event at time t and each horizon h in {1s, 5s, 10s, 30s}:
  - long_mfe_h:   max(mid[t..t+h] - mid[t])   best case if you went long
  - short_mfe_h:  max(mid[t] - mid[t..t+h])    best case if you went short
  - long_mae_h:   max(mid[t] - mid[t..t+h])    worst drawdown if long
  - short_mae_h:  max(mid[t..t+h] - mid[t])    worst drawup if short
  - mfe_ratio_h:  long_mfe / (long_mfe + long_mae + eps)  favorability ratio
  - time_to_mfe_h: fraction of horizon when MFE was reached (for longs)

Data flow:
  1. Read raw .dbn.zst -> reconstruct LOB -> extract (ts_ns, mid_price) for RTH
  2. Load processed smart_v3 npz -> get event timestamps
  3. For each event, find all mid_prices in [t, t+h] window
  4. Compute MFE/MAE/ratio/timing from the mid_price path
  5. Save per-date npz + summary JSON

Run on Neptune (CPU-only, uses multiprocessing):
  /home/nick/Lvl3Quant/venv_training/bin/python3 /home/nick/Lvl3Quant/experiments/mfe_mae_relabel_v2.py

Output: /home/nick/Lvl3Quant/data/processed/mfe_mae_labels/{date}_mfe_mae.npz
"""

import os
import sys
import glob
import json
import time
import math
import logging
import argparse
import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count
from pathlib import Path

# ---------------------------------------------------------------------------
#  Configuration
# ---------------------------------------------------------------------------

RAW_MBO_DIRS = [
    "/home/nick/Lvl3Quant/data/raw/mbo_files",
    "/home/nick/Lvl3Quant/data/raw/mbo",
]
PROCESSED_DIR = "/home/nick/Lvl3Quant/data/processed/mbo_events_smart_v3"
OUTPUT_DIR = "/home/nick/Lvl3Quant/data/processed/mfe_mae_labels"
SUMMARY_PATH = "/home/nick/Lvl3Quant/data/processed/mfe_mae_labels/summary_stats.json"
LOG_DIR = "/home/nick/Lvl3Quant/logs"
LOG_FILE = os.path.join(LOG_DIR, "mfe_mae_relabel_v2.log")

MLFLOW_TRACKING_URI = "http://jupiter:5000"
MLFLOW_EXPERIMENT = "mfe_mae_relabeling_v2"

# ES tick size in Databento fixed-point (price * 1e9)
TICK_SIZE_FIXED = 250_000_000   # 0.25 * 1e9
TICK_SIZE_FLOAT = 0.25

# RTH window (conservative: covers both EST and EDT)
RTH_START_UTC_SEC = 13 * 3600 + 30 * 60   # 13:30 UTC
RTH_END_UTC_SEC   = 21 * 3600             # 21:00 UTC

# Label horizons in nanoseconds
HORIZONS = {
    "1s":  1_000_000_000,
    "5s":  5_000_000_000,
    "10s": 10_000_000_000,
    "30s": 30_000_000_000,
}

# Multiprocessing
N_WORKERS = max(1, min(cpu_count() - 1, 6))  # leave 1 core free, cap at 6

# Epsilon for ratio computation
EPS = 1e-9

# ---------------------------------------------------------------------------
#  LOB Tracker (same logic as mbo_event_pipeline.py)
# ---------------------------------------------------------------------------

class LOBTracker:
    """Lightweight Level-of-Book tracker for mid-price extraction."""

    PRUNE_EVERY = 5000

    def __init__(self, tick_size_fixed: int = TICK_SIZE_FIXED):
        self.tick_size_fixed = tick_size_fixed
        self._bid_levels: dict = {}
        self._ask_levels: dict = {}
        self._best_bid: int = 0
        self._best_ask: int = 0
        self._mid: float = 0.0
        self._last_trade_price: int = 0
        self._event_count: int = 0

    def _recompute(self):
        valid_bids = [k for k, v in self._bid_levels.items() if v > 0]
        valid_asks = [k for k, v in self._ask_levels.items() if v > 0]
        self._best_bid = max(valid_bids) if valid_bids else 0
        self._best_ask = min(valid_asks) if valid_asks else 0

        if (self._best_bid > 0 and self._best_ask > 0
                and self._best_ask > self._best_bid):
            self._mid = (self._best_bid + self._best_ask) / 2.0
        elif self._last_trade_price > 0:
            self._mid = float(self._last_trade_price)

    def _prune_levels(self):
        if self._mid <= 0:
            return
        radius = 50 * self.tick_size_fixed
        lo = self._mid - radius
        hi = self._mid + radius
        self._bid_levels = {k: v for k, v in self._bid_levels.items()
                           if lo <= k <= hi and v > 0}
        self._ask_levels = {k: v for k, v in self._ask_levels.items()
                           if lo <= k <= hi and v > 0}

    def process(self, action: str, side: str, price: int, qty: int):
        """Update LOB. Returns mid_price in fixed-point."""
        INVALID_PRICE = 9_223_372_036_854_775_807
        self._event_count += 1

        if action == 'R':
            self._bid_levels.clear()
            self._ask_levels.clear()
            self._best_bid = 0
            self._best_ask = 0
            return self._mid

        if price == INVALID_PRICE or price <= 0:
            return self._mid

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

        if self._event_count % self.PRUNE_EVERY == 0:
            self._prune_levels()

        self._recompute()
        return self._mid


# ---------------------------------------------------------------------------
#  Raw file discovery
# ---------------------------------------------------------------------------

def find_raw_file(date_str: str) -> str:
    """Find the raw .dbn.zst file for a given date across all raw dirs."""
    fname = f"glbx-mdp3-{date_str}.mbo.dbn.zst"
    for d in RAW_MBO_DIRS:
        path = os.path.join(d, fname)
        if os.path.exists(path):
            # Resolve symlinks
            return os.path.realpath(path)
    return ""


def detect_dominant_instrument(filepath: str, sample_size: int = 20000) -> int:
    """Fast scan to detect the most common instrument_id."""
    import databento as db
    import itertools
    inst_counter = Counter()
    store = db.DBNStore.from_file(filepath)
    for r in itertools.islice(store, sample_size):
        if str(r.action) != 'R':
            inst_counter[r.instrument_id] += 1
    if not inst_counter:
        return -1
    return inst_counter.most_common(1)[0][0]


# ---------------------------------------------------------------------------
#  Extract mid-price series from raw MBO data
# ---------------------------------------------------------------------------

def extract_mid_prices(raw_path: str) -> tuple:
    """
    Stream raw .dbn.zst, rebuild LOB, return (timestamps_ns, mid_prices_ticks)
    for RTH events of the dominant instrument.

    Returns:
        ts_ns: np.ndarray int64 — nanosecond timestamps
        mid_ticks: np.ndarray float32 — mid prices in tick units
    """
    import databento as db

    dominant = detect_dominant_instrument(raw_path)
    if dominant == -1:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    store = db.DBNStore.from_file(raw_path)
    lob = LOBTracker()

    ts_list = []
    mid_list = []

    for r in store:
        act = str(r.action)
        if act == 'R':
            lob.process('R', str(r.side), r.price, r.size)
            continue

        if r.instrument_id != dominant:
            continue

        mid = lob.process(act, str(r.side), r.price, r.size)

        # Include ALL events (not just RTH) because the smart_v3 npz
        # contains events across the full session with valid labels.
        # MFE/MAE needs mid_prices at every event timestamp.
        if mid > 0:
            ts_list.append(r.ts_event)
            mid_list.append(mid)

    if not ts_list:
        return np.array([], dtype=np.int64), np.array([], dtype=np.float32)

    ts_ns = np.array(ts_list, dtype=np.int64)
    # Convert fixed-point mid to ticks: mid_fixed / tick_size_fixed
    mid_ticks = np.array(mid_list, dtype=np.float64) / TICK_SIZE_FIXED
    mid_ticks = mid_ticks.astype(np.float32)

    # Forward-fill zeros (pre-book-init events)
    for i in range(1, len(mid_ticks)):
        if mid_ticks[i] == 0.0 and mid_ticks[i - 1] != 0.0:
            mid_ticks[i] = mid_ticks[i - 1]

    return ts_ns, mid_ticks


# ---------------------------------------------------------------------------
#  Compute MFE/MAE for one horizon (vectorized with searchsorted)
# ---------------------------------------------------------------------------

def compute_mfe_mae_horizon(event_ts: np.ndarray, mid_ts: np.ndarray,
                            mid_prices: np.ndarray, h_ns: int) -> dict:
    """
    For each event timestamp, compute MFE/MAE within the forward window [t, t+h].

    Uses the full mid_price series (from raw data) to find max/min in each window.
    Events are aligned to the mid_price series via timestamp matching.

    Args:
        event_ts: (N,) int64 — event timestamps from smart_v3 npz
        mid_ts: (M,) int64 — mid-price timestamps from raw LOB reconstruction
        mid_prices: (M,) float32 — mid prices in ticks
        h_ns: int — horizon in nanoseconds

    Returns:
        dict with keys: long_mfe, short_mfe, long_mae, short_mae,
                        mfe_ratio, time_to_mfe  (all shape (N,) float32)
    """
    N = len(event_ts)
    M = len(mid_ts)

    long_mfe = np.full(N, np.nan, dtype=np.float32)
    short_mfe = np.full(N, np.nan, dtype=np.float32)
    long_mae = np.full(N, np.nan, dtype=np.float32)
    short_mae = np.full(N, np.nan, dtype=np.float32)
    mfe_ratio = np.full(N, np.nan, dtype=np.float32)
    time_to_mfe = np.full(N, np.nan, dtype=np.float32)

    if M == 0:
        return {
            "long_mfe": long_mfe, "short_mfe": short_mfe,
            "long_mae": long_mae, "short_mae": short_mae,
            "mfe_ratio": mfe_ratio, "time_to_mfe": time_to_mfe,
        }

    # For each event, find its position in the mid-price series
    # event_ts[i] should match (or be very close to) some mid_ts[j]
    # Use searchsorted to find the start index
    start_idx = np.searchsorted(mid_ts, event_ts, side='left')
    end_ts = event_ts + h_ns
    end_idx = np.searchsorted(mid_ts, end_ts, side='right')

    # Process in chunks for memory efficiency
    CHUNK = 50000
    for chunk_start in range(0, N, CHUNK):
        chunk_end = min(chunk_start + CHUNK, N)

        for i in range(chunk_start, chunk_end):
            si = start_idx[i]
            ei = end_idx[i]

            # Need at least 2 points in window for meaningful MFE/MAE
            if si >= M or ei <= si or ei - si < 2:
                continue

            # Ensure end of window is actually within the horizon
            # (handles truncation at end of day)
            if mid_ts[ei - 1] < event_ts[i]:
                continue

            window = mid_prices[si:ei]
            p0 = window[0]  # price at event time

            if p0 == 0.0:
                continue

            deltas = window - p0  # price changes relative to entry

            max_up = np.max(deltas)    # best upward move
            max_down = np.min(deltas)  # worst downward move (negative)

            # For LONG position:
            #   MFE = max favorable = max upward move
            #   MAE = max adverse = max downward move (as positive number)
            long_mfe[i] = max(0.0, max_up)
            long_mae[i] = max(0.0, -max_down)

            # For SHORT position:
            #   MFE = max favorable = max downward move (as positive number)
            #   MAE = max adverse = max upward move
            short_mfe[i] = max(0.0, -max_down)
            short_mae[i] = max(0.0, max_up)

            # MFE ratio (for longs): how much of total range was favorable
            lmfe = long_mfe[i]
            lmae = long_mae[i]
            mfe_ratio[i] = lmfe / (lmfe + lmae + EPS)

            # Time to MFE (for longs): fraction of horizon when max was reached
            if max_up > 0:
                mfe_idx = np.argmax(deltas)
                t_mfe = mid_ts[si + mfe_idx]
                t_span = mid_ts[min(ei - 1, M - 1)] - mid_ts[si]
                if t_span > 0:
                    time_to_mfe[i] = float(t_mfe - mid_ts[si]) / float(t_span)
                else:
                    time_to_mfe[i] = 0.0

    return {
        "long_mfe": long_mfe, "short_mfe": short_mfe,
        "long_mae": long_mae, "short_mae": short_mae,
        "mfe_ratio": mfe_ratio, "time_to_mfe": time_to_mfe,
    }


# ---------------------------------------------------------------------------
#  Process one date
# ---------------------------------------------------------------------------

def process_date(date_str: str) -> dict:
    """
    Process a single date: extract mid-prices from raw data, compute MFE/MAE
    for all events across all horizons, save output npz.

    Returns stats dict.
    """
    t_start = time.time()
    result = {"date": date_str, "status": "unknown"}

    try:
        # Check if already processed
        out_path = os.path.join(OUTPUT_DIR, f"{date_str}_mfe_mae.npz")
        if os.path.exists(out_path):
            result["status"] = "skipped"
            return result

        # Find raw file
        raw_path = find_raw_file(date_str)
        if not raw_path:
            result["status"] = "no_raw_file"
            return result

        # Load processed npz to get event timestamps
        proc_path = os.path.join(PROCESSED_DIR, f"{date_str}_mbo_events.npz")
        if not os.path.exists(proc_path):
            result["status"] = "no_processed_file"
            return result

        proc = np.load(proc_path)
        event_ts = proc["timestamps"]
        N_events = len(event_ts)

        if N_events == 0:
            result["status"] = "empty"
            return result

        # Extract mid-price series from raw data
        mid_ts, mid_prices = extract_mid_prices(raw_path)

        if len(mid_ts) == 0:
            result["status"] = "no_mid_prices"
            return result

        result["n_events"] = N_events
        result["n_mid_prices"] = len(mid_ts)

        # Compute MFE/MAE for each horizon
        save_dict = {}
        horizon_stats = {}

        for h_name, h_ns in HORIZONS.items():
            mfe_mae = compute_mfe_mae_horizon(event_ts, mid_ts, mid_prices, h_ns)

            # Store arrays
            for key, arr in mfe_mae.items():
                save_dict[f"{key}_{h_name}"] = arr

            # Compute summary stats for this horizon
            valid = ~np.isnan(mfe_mae["long_mfe"])
            n_valid = int(valid.sum())

            if n_valid > 0:
                lmfe = mfe_mae["long_mfe"][valid]
                smfe = mfe_mae["short_mfe"][valid]
                lmae = mfe_mae["long_mae"][valid]
                smae = mfe_mae["short_mae"][valid]

                horizon_stats[h_name] = {
                    "n_valid": n_valid,
                    "valid_frac": round(n_valid / N_events, 4),
                    "long_mfe_p50": round(float(np.median(lmfe)), 4),
                    "long_mfe_p90": round(float(np.percentile(lmfe, 90)), 4),
                    "long_mfe_p99": round(float(np.percentile(lmfe, 99)), 4),
                    "short_mfe_p50": round(float(np.median(smfe)), 4),
                    "short_mfe_p90": round(float(np.percentile(smfe, 90)), 4),
                    "short_mfe_p99": round(float(np.percentile(smfe, 99)), 4),
                    "long_mae_p50": round(float(np.median(lmae)), 4),
                    "long_mae_p90": round(float(np.percentile(lmae, 90)), 4),
                    "short_mae_p50": round(float(np.median(smae)), 4),
                    "short_mae_p90": round(float(np.percentile(smae, 90)), 4),
                    "mfe_ratio_mean": round(float(np.nanmean(mfe_mae["mfe_ratio"][valid])), 4),
                    "time_to_mfe_mean": round(float(np.nanmean(mfe_mae["time_to_mfe"][valid])), 4),
                }
            else:
                horizon_stats[h_name] = {"n_valid": 0, "valid_frac": 0.0}

        # Save output
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        np.savez_compressed(out_path, **save_dict)

        elapsed = time.time() - t_start
        result["status"] = "ok"
        result["elapsed_sec"] = round(elapsed, 1)
        result["horizon_stats"] = horizon_stats
        result["output_size_mb"] = round(os.path.getsize(out_path) / 1e6, 1)

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        import traceback
        result["traceback"] = traceback.format_exc()

    return result


# ---------------------------------------------------------------------------
#  Main
# ---------------------------------------------------------------------------

def setup_logging():
    os.makedirs(LOG_DIR, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(LOG_FILE),
            logging.StreamHandler(sys.stdout),
        ],
    )


def main():
    parser = argparse.ArgumentParser(description="MFE/MAE Within-Horizon Relabeling v2")
    parser.add_argument("--workers", type=int, default=N_WORKERS,
                        help=f"Number of parallel workers (default: {N_WORKERS})")
    parser.add_argument("--dates", nargs="*", default=None,
                        help="Specific dates to process (default: all)")
    parser.add_argument("--no-mlflow", action="store_true",
                        help="Skip MLflow logging")
    parser.add_argument("--dry-run", action="store_true",
                        help="Just probe data, don't process")
    args = parser.parse_args()

    setup_logging()
    log = logging.getLogger(__name__)

    log.info("=" * 70)
    log.info("MFE/MAE Within-Horizon Relabeling Pipeline v2")
    log.info("=" * 70)

    # Discover dates
    proc_files = sorted(glob.glob(os.path.join(PROCESSED_DIR, "*_mbo_events.npz")))
    all_dates = [os.path.basename(f).replace("_mbo_events.npz", "") for f in proc_files]

    if args.dates:
        dates = [d for d in args.dates if d in all_dates]
        log.info(f"Processing {len(dates)} specified dates")
    else:
        dates = all_dates
        log.info(f"Found {len(dates)} processed dates")

    # Check which are already done
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    done = set(
        f.replace("_mfe_mae.npz", "")
        for f in os.listdir(OUTPUT_DIR)
        if f.endswith("_mfe_mae.npz")
    )
    remaining = [d for d in dates if d not in done]
    log.info(f"Already processed: {len(done)}, remaining: {len(remaining)}")

    if not remaining:
        log.info("All dates already processed. Nothing to do.")
        return

    # Check raw file availability
    raw_available = sum(1 for d in remaining if find_raw_file(d))
    log.info(f"Raw files available for {raw_available}/{len(remaining)} remaining dates")

    if args.dry_run:
        log.info("DRY RUN — probing first date only")
        # Probe one file
        probe_date = remaining[0]
        raw_path = find_raw_file(probe_date)
        log.info(f"Probing {probe_date} from {raw_path}")

        proc = np.load(os.path.join(PROCESSED_DIR, f"{probe_date}_mbo_events.npz"))
        log.info(f"  Processed npz keys: {list(proc.keys())}")
        log.info(f"  Event timestamps shape: {proc['timestamps'].shape}")
        log.info(f"  Events shape: {proc['events'].shape}")

        t0 = time.time()
        mid_ts, mid_prices = extract_mid_prices(raw_path)
        log.info(f"  Mid-price extraction took {time.time()-t0:.1f}s")
        log.info(f"  Mid prices: {len(mid_prices)} points, "
                 f"range [{mid_prices.min():.2f}, {mid_prices.max():.2f}] ticks")
        log.info(f"  Event ts range: [{proc['timestamps'][0]}, {proc['timestamps'][-1]}]")
        log.info(f"  Mid ts range: [{mid_ts[0]}, {mid_ts[-1]}]")
        return

    # MLflow setup
    mlflow_run = None
    if not args.no_mlflow:
        try:
            import mlflow
            mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
            mlflow.set_experiment(MLFLOW_EXPERIMENT)
            mlflow_run = mlflow.start_run(run_name=f"relabel_v2_{len(remaining)}dates")
            mlflow.log_params({
                "n_dates_total": len(dates),
                "n_dates_remaining": len(remaining),
                "n_workers": args.workers,
                "horizons": ",".join(HORIZONS.keys()),
            })
            log.info(f"MLflow run started: {mlflow_run.info.run_id}")
        except Exception as e:
            log.warning(f"MLflow setup failed: {e} — continuing without tracking")
            mlflow_run = None

    # Process dates
    t_total_start = time.time()
    results = []
    errors = []

    log.info(f"Starting processing with {args.workers} workers...")

    if args.workers <= 1:
        # Sequential for debugging
        for i, date_str in enumerate(remaining):
            r = process_date(date_str)
            results.append(r)
            if r["status"] == "error":
                errors.append(r)
                log.error(f"[{i+1}/{len(remaining)}] {date_str}: ERROR - {r.get('error','?')}")
            elif r["status"] == "ok":
                log.info(f"[{i+1}/{len(remaining)}] {date_str}: OK "
                         f"({r['n_events']} events, {r['elapsed_sec']}s, "
                         f"{r['output_size_mb']}MB)")
            else:
                log.info(f"[{i+1}/{len(remaining)}] {date_str}: {r['status']}")
    else:
        # Parallel processing
        with Pool(processes=args.workers) as pool:
            for i, r in enumerate(pool.imap_unordered(process_date, remaining)):
                results.append(r)
                if r["status"] == "error":
                    errors.append(r)
                    log.error(f"[{i+1}/{len(remaining)}] {r['date']}: ERROR - "
                              f"{r.get('error','?')}")
                elif r["status"] == "ok":
                    if (i + 1) % 10 == 0 or i == 0:
                        log.info(f"[{i+1}/{len(remaining)}] {r['date']}: OK "
                                 f"({r['n_events']} events, {r['elapsed_sec']}s)")
                # Log progress every 10 dates
                if (i + 1) % 10 == 0:
                    elapsed = time.time() - t_total_start
                    rate = (i + 1) / elapsed * 60  # dates per minute
                    eta_min = (len(remaining) - i - 1) / max(rate, 0.01)
                    log.info(f"  Progress: {i+1}/{len(remaining)} "
                             f"({rate:.1f} dates/min, ETA {eta_min:.0f} min)")

    total_elapsed = time.time() - t_total_start
    n_ok = sum(1 for r in results if r["status"] == "ok")
    n_err = len(errors)
    n_skip = sum(1 for r in results if r["status"] == "skipped")

    log.info("=" * 70)
    log.info(f"COMPLETED: {n_ok} ok, {n_skip} skipped, {n_err} errors "
             f"in {total_elapsed/60:.1f} min")

    # Aggregate summary statistics across all dates
    agg_stats = {}
    for h_name in HORIZONS:
        all_lmfe, all_smfe, all_lmae, all_smae = [], [], [], []

        for r in results:
            if r["status"] != "ok" or "horizon_stats" not in r:
                continue
            hs = r["horizon_stats"].get(h_name, {})
            if hs.get("n_valid", 0) == 0:
                continue

            # We only have per-date percentiles, not raw values.
            # Collect per-date medians for the aggregate summary.
            all_lmfe.append(hs.get("long_mfe_p50", 0))
            all_smfe.append(hs.get("short_mfe_p50", 0))
            all_lmae.append(hs.get("long_mae_p50", 0))
            all_smae.append(hs.get("short_mae_p50", 0))

        if all_lmfe:
            agg_stats[h_name] = {
                "n_dates": len(all_lmfe),
                "long_mfe_median_of_medians": round(float(np.median(all_lmfe)), 4),
                "short_mfe_median_of_medians": round(float(np.median(all_smfe)), 4),
                "long_mae_median_of_medians": round(float(np.median(all_lmae)), 4),
                "short_mae_median_of_medians": round(float(np.median(all_smae)), 4),
            }

    # Also compute exact aggregate stats by re-reading a sample of output files
    log.info("Computing exact aggregate stats from output files...")
    agg_exact = compute_aggregate_stats(OUTPUT_DIR)

    summary = {
        "run_timestamp": time.strftime("%Y-%m-%d %H:%M:%S UTC", time.gmtime()),
        "n_dates_processed": n_ok,
        "n_dates_skipped": n_skip,
        "n_dates_error": n_err,
        "total_elapsed_min": round(total_elapsed / 60, 1),
        "per_date_stats_approx": agg_stats,
        "aggregate_stats_exact": agg_exact,
        "errors": [{"date": e["date"], "error": e.get("error", "?")}
                   for e in errors],
    }

    # Save summary
    with open(SUMMARY_PATH, "w") as f:
        json.dump(summary, f, indent=2)
    log.info(f"Summary saved to {SUMMARY_PATH}")

    # Print summary statistics
    log.info("")
    log.info("=" * 70)
    log.info("AGGREGATE MFE/MAE STATISTICS (exact, all dates pooled)")
    log.info("=" * 70)
    log.info(f"{'Horizon':>8} | {'LongMFE p50':>11} {'p90':>8} {'p99':>8} | "
             f"{'ShortMFE p50':>12} {'p90':>8} | "
             f"{'LongMAE p50':>11} {'p90':>8}")
    log.info("-" * 95)
    for h_name in HORIZONS:
        if h_name in agg_exact:
            s = agg_exact[h_name]
            log.info(f"{h_name:>8} | "
                     f"{s['long_mfe_p50']:>11.4f} {s['long_mfe_p90']:>8.4f} "
                     f"{s['long_mfe_p99']:>8.4f} | "
                     f"{s['short_mfe_p50']:>12.4f} {s['short_mfe_p90']:>8.4f} | "
                     f"{s['long_mae_p50']:>11.4f} {s['long_mae_p90']:>8.4f}")
    log.info("")
    log.info("KEY FOR HC #428 R2 — TP SETTING:")
    for h_name in HORIZONS:
        if h_name in agg_exact:
            s = agg_exact[h_name]
            log.info(f"  {h_name}: TP <= {s['long_mfe_p90']:.3f} ticks (p90 long MFE), "
                     f"short TP <= {s['short_mfe_p90']:.3f} ticks (p90 short MFE)")

    # Log to MLflow
    if mlflow_run is not None:
        try:
            import mlflow
            mlflow.log_metrics({
                "n_dates_ok": n_ok,
                "n_dates_error": n_err,
                "total_elapsed_min": round(total_elapsed / 60, 1),
            })
            for h_name, stats in agg_exact.items():
                for k, v in stats.items():
                    mlflow.log_metric(f"{h_name}_{k}", v)
            mlflow.log_artifact(SUMMARY_PATH)
            mlflow.end_run()
            log.info("MLflow run completed")
        except Exception as e:
            log.warning(f"MLflow logging failed: {e}")

    if errors:
        log.warning(f"\n{n_err} dates had errors:")
        for e in errors[:10]:
            log.warning(f"  {e['date']}: {e.get('error','?')}")


def compute_aggregate_stats(output_dir: str, max_files: int = 0) -> dict:
    """
    Read all output npz files and compute exact aggregate percentiles
    across all dates (pooled).

    Args:
        output_dir: directory with *_mfe_mae.npz files
        max_files: if >0, limit to this many files (for speed)

    Returns:
        dict: horizon -> {long_mfe_p50, long_mfe_p90, ...}
    """
    files = sorted(glob.glob(os.path.join(output_dir, "*_mfe_mae.npz")))
    if max_files > 0:
        files = files[:max_files]

    if not files:
        return {}

    # Accumulate per-horizon
    accum = {h: {"lmfe": [], "smfe": [], "lmae": [], "smae": [],
                 "ratio": [], "ttmfe": []}
             for h in HORIZONS}

    for fpath in files:
        try:
            d = np.load(fpath)
            for h_name in HORIZONS:
                lmfe_key = f"long_mfe_{h_name}"
                if lmfe_key not in d:
                    continue

                lmfe = d[f"long_mfe_{h_name}"]
                smfe = d[f"short_mfe_{h_name}"]
                lmae = d[f"long_mae_{h_name}"]
                smae = d[f"short_mae_{h_name}"]
                ratio = d[f"mfe_ratio_{h_name}"]
                ttmfe = d[f"time_to_mfe_{h_name}"]

                valid = ~np.isnan(lmfe)
                if valid.sum() == 0:
                    continue

                # Subsample large files to keep memory bounded
                # (~6.5M events/day * 248 days = 1.6B — too much to pool)
                n_valid = int(valid.sum())
                if n_valid > 100000:
                    # Random subsample of 100k
                    idx = np.where(valid)[0]
                    rng = np.random.RandomState(42)
                    idx = rng.choice(idx, size=100000, replace=False)
                    accum[h_name]["lmfe"].append(lmfe[idx])
                    accum[h_name]["smfe"].append(smfe[idx])
                    accum[h_name]["lmae"].append(lmae[idx])
                    accum[h_name]["smae"].append(smae[idx])
                    accum[h_name]["ratio"].append(ratio[idx])
                    accum[h_name]["ttmfe"].append(ttmfe[idx])
                else:
                    accum[h_name]["lmfe"].append(lmfe[valid])
                    accum[h_name]["smfe"].append(smfe[valid])
                    accum[h_name]["lmae"].append(lmae[valid])
                    accum[h_name]["smae"].append(smae[valid])
                    accum[h_name]["ratio"].append(ratio[valid])
                    accum[h_name]["ttmfe"].append(ttmfe[valid])
        except Exception:
            continue

    result = {}
    for h_name in HORIZONS:
        a = accum[h_name]
        if not a["lmfe"]:
            continue

        lmfe = np.concatenate(a["lmfe"])
        smfe = np.concatenate(a["smfe"])
        lmae = np.concatenate(a["lmae"])
        smae = np.concatenate(a["smae"])
        ratio = np.concatenate(a["ratio"])
        ttmfe = np.concatenate(a["ttmfe"])

        result[h_name] = {
            "n_samples": len(lmfe),
            "long_mfe_p50": round(float(np.nanmedian(lmfe)), 4),
            "long_mfe_p90": round(float(np.nanpercentile(lmfe, 90)), 4),
            "long_mfe_p99": round(float(np.nanpercentile(lmfe, 99)), 4),
            "long_mfe_mean": round(float(np.nanmean(lmfe)), 4),
            "short_mfe_p50": round(float(np.nanmedian(smfe)), 4),
            "short_mfe_p90": round(float(np.nanpercentile(smfe, 90)), 4),
            "short_mfe_p99": round(float(np.nanpercentile(smfe, 99)), 4),
            "short_mfe_mean": round(float(np.nanmean(smfe)), 4),
            "long_mae_p50": round(float(np.nanmedian(lmae)), 4),
            "long_mae_p90": round(float(np.nanpercentile(lmae, 90)), 4),
            "long_mae_p99": round(float(np.nanpercentile(lmae, 99)), 4),
            "short_mae_p50": round(float(np.nanmedian(smae)), 4),
            "short_mae_p90": round(float(np.nanpercentile(smae, 90)), 4),
            "short_mae_p99": round(float(np.nanpercentile(smae, 99)), 4),
            "mfe_ratio_mean": round(float(np.nanmean(ratio)), 4),
            "mfe_ratio_p50": round(float(np.nanmedian(ratio)), 4),
            "time_to_mfe_mean": round(float(np.nanmean(ttmfe)), 4),
            "time_to_mfe_p50": round(float(np.nanmedian(ttmfe)), 4),
        }

    return result


if __name__ == "__main__":
    main()
