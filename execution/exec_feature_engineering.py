#!/usr/bin/env python3
"""
Execution Feature Engineering — Jupiter CPU Pipeline
=====================================================
Builds RICH execution-specific features from raw MBO data that Neptune's
RL/AI execution model can consume.

Neptune's RL v5 currently uses only 10 basic book features. This script
engineers 40+ sophisticated execution features covering:

1. FILL PROBABILITY FEATURES
   - P(fill within N seconds) at current queue depth
   - Queue consumption velocity
   - Historical fill rates by spread regime

2. ADVERSE SELECTION INDICATORS
   - Price move after trades at bid/ask (toxicity)
   - Informed flow detection (large aggressive orders)
   - Post-fill drift by time-of-day

3. ORDERFLOW DYNAMICS
   - Trade imbalance momentum & acceleration
   - Volume-weighted imbalance (big vs small trades)
   - Cancellation velocity (pulling = informed exit)
   - Add/cancel ratio at BBO (refreshing vs pulling)

4. SPREAD REGIME FEATURES
   - Spread state (tight/wide/transitioning)
   - Time-in-spread (how long at current spread)
   - Spread volatility (choppy vs stable)

5. QUEUE DYNAMICS
   - Queue replenishment rate after trades
   - Depth restoration speed (resilience)
   - Layering detection (depth far from BBO)

6. TIME-OF-DAY EXECUTION QUALITY
   - Hour/minute encoding
   - Historical fill quality by TOD
   - Volatility regime by TOD

All features are computed per-event-window (matching RL v5's DECISION_STRIDE=5000)
and saved as .npz files Neptune can directly consume.

Usage:
    python exec_feature_engineering.py --workers 16
    python exec_feature_engineering.py --dates 20260301 20260302
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

# ── Paths ──
LVL3_ROOT = Path(__file__).resolve().parent.parent
MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events"
OUTPUT_DIR = LVL3_ROOT / "output" / "exec_features_v1"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
TICK_SIZE = 0.25
TICK_VALUE = 12.50
DECISION_STRIDE = 5000  # Match RL v5's decision frequency

# MBO event encoding
ACTION_ADD = 0
ACTION_CANCEL = 1
ACTION_MODIFY = 2
ACTION_TRADE = 3
ACTION_FILL = 4
SIDE_BID = 0
SIDE_ASK = 1

# Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(OUTPUT_DIR / "exec_features.log"), mode="a"),
    ]
)
log = logging.getLogger("exec_features")


# ═══════════════════════════════════════════════════════════════
# FEATURE EXTRACTORS
# ═══════════════════════════════════════════════════════════════

class ExecutionFeatureExtractor:
    """Extracts 44 execution-specific features from MBO event windows."""

    FEATURE_NAMES = [
        # Fill probability (6)
        "fill_prob_1s", "fill_prob_3s", "fill_prob_10s",
        "queue_consumption_velocity_bid", "queue_consumption_velocity_ask",
        "queue_replenish_ratio",
        # Adverse selection (6)
        "post_trade_drift_bid_1k", "post_trade_drift_ask_1k",
        "toxicity_imbalance", "large_trade_fraction",
        "informed_flow_score", "adverse_select_asymmetry",
        # Orderflow dynamics (8)
        "trade_imb_momentum", "trade_imb_acceleration",
        "volume_weighted_imbalance", "small_vs_large_imbalance",
        "cancel_velocity_bid", "cancel_velocity_ask",
        "add_cancel_ratio_bid", "add_cancel_ratio_ask",
        # Spread regime (6)
        "spread_ticks", "spread_volatility",
        "time_in_spread", "spread_state",  # 0=tight, 0.5=normal, 1=wide
        "spread_mean_10k", "spread_widening_trend",
        # Queue dynamics (8)
        "bid_depth_l1", "ask_depth_l1",
        "bid_depth_l2_l5", "ask_depth_l2_l5",
        "depth_restoration_speed", "depth_imbalance_l1",
        "layering_score_bid", "layering_score_ask",
        # Time of day (4)
        "tod_sin", "tod_cos",  # Circular encoding
        "minutes_from_open", "session_progress",
        # Meta (6)
        "event_rate", "trade_rate",
        "cancel_rate", "add_rate",
        "book_turnover", "price_volatility_window",
    ]

    NUM_FEATURES = len(FEATURE_NAMES)  # 44

    def __init__(self):
        self.reset()

    def reset(self):
        """Reset all state for new day."""
        # Running state
        self.bid_depth_l1 = 0.0
        self.ask_depth_l1 = 0.0
        self.bid_depth_far = 0.0
        self.ask_depth_far = 0.0
        self.spread_history = []
        self.trade_prices = []
        self.trade_sides = []
        self.trade_sizes = []

        # Per-window accumulators
        self._reset_window()

    def _reset_window(self):
        """Reset per-window accumulators."""
        self.w_adds_bid = 0
        self.w_adds_ask = 0
        self.w_cancels_bid = 0
        self.w_cancels_ask = 0
        self.w_trades_bid = 0  # aggressive buys
        self.w_trades_ask = 0  # aggressive sells
        self.w_trade_vol_bid = 0.0
        self.w_trade_vol_ask = 0.0
        self.w_large_trade_vol = 0.0
        self.w_small_trade_vol = 0.0
        self.w_spreads = []
        self.w_trade_prices_bid = []  # prices where buyer aggressor
        self.w_trade_prices_ask = []  # prices where seller aggressor
        self.w_queue_consumed_bid = 0.0
        self.w_queue_consumed_ask = 0.0
        self.w_queue_added_bid = 0.0
        self.w_queue_added_ask = 0.0
        self.w_cancel_vol_bid = 0.0
        self.w_cancel_vol_ask = 0.0
        self.w_n_events = 0
        self.w_depth_snapshots = []

        # Previous window values for momentum
        self.prev_trade_imb = 0.0
        self.prev_prev_trade_imb = 0.0

    def process_window(self, events: np.ndarray, timestamps_ns: np.ndarray,
                       window_idx: int) -> np.ndarray:
        """
        Process a window of DECISION_STRIDE events and return features.

        events: (N, 6) [time_delta_log, event_type_id, side_id, price_rel_ticks, qty_log, spread_ticks]
        timestamps_ns: (N,) absolute nanosecond timestamps

        Returns: (NUM_FEATURES,) feature vector
        """
        if len(events) == 0:
            return np.zeros(self.NUM_FEATURES, dtype=np.float32)

        actions = events[:, 1].astype(np.int32)
        sides = events[:, 2].astype(np.int32)
        prices = events[:, 3]  # price relative to reference in ticks
        qty_logs = events[:, 4]
        spreads = events[:, 5]

        # Decode quantities
        qtys = np.where(qty_logs > 0, np.exp(np.clip(qty_logs, 0, 10)), 1.0)

        # Large trade threshold (> 5 contracts)
        LARGE_THRESHOLD = 5.0

        # ── Masks ──
        add_mask = actions == ACTION_ADD
        cancel_mask = actions == ACTION_CANCEL
        trade_mask = (actions == ACTION_TRADE) | (actions == ACTION_FILL)
        bid_mask = sides == SIDE_BID
        ask_mask = sides == SIDE_ASK

        # ── Count events ──
        n_add_bid = (add_mask & bid_mask).sum()
        n_add_ask = (add_mask & ask_mask).sum()
        n_cancel_bid = (cancel_mask & bid_mask).sum()
        n_cancel_ask = (cancel_mask & ask_mask).sum()
        n_trade_bid = (trade_mask & bid_mask).sum()  # aggressive buys
        n_trade_ask = (trade_mask & ask_mask).sum()   # aggressive sells

        # ── Volumes ──
        vol_add_bid = qtys[add_mask & bid_mask].sum()
        vol_add_ask = qtys[add_mask & ask_mask].sum()
        vol_cancel_bid = qtys[cancel_mask & bid_mask].sum()
        vol_cancel_ask = qtys[cancel_mask & ask_mask].sum()
        vol_trade_bid = qtys[trade_mask & bid_mask].sum()
        vol_trade_ask = qtys[trade_mask & ask_mask].sum()

        # Large vs small trades
        trade_qtys = qtys[trade_mask]
        large_vol = trade_qtys[trade_qtys >= LARGE_THRESHOLD].sum()
        small_vol = trade_qtys[trade_qtys < LARGE_THRESHOLD].sum()
        total_trade_vol = vol_trade_bid + vol_trade_ask + 1e-8

        # ── Spread features ──
        valid_spreads = spreads[spreads > 0]
        if len(valid_spreads) > 0:
            current_spread = float(valid_spreads[-1])
            spread_vol = float(np.std(valid_spreads)) if len(valid_spreads) > 1 else 0.0
            spread_mean = float(np.mean(valid_spreads))
        else:
            current_spread = 1.0
            spread_vol = 0.0
            spread_mean = 1.0

        # Track spread history for trend
        self.spread_history.append(current_spread)
        if len(self.spread_history) > 20:
            self.spread_history = self.spread_history[-20:]

        spread_trend = 0.0
        if len(self.spread_history) >= 5:
            recent = np.mean(self.spread_history[-3:])
            older = np.mean(self.spread_history[-10:-3]) if len(self.spread_history) >= 10 else np.mean(self.spread_history[:-3])
            spread_trend = np.clip((recent - older) / (older + 1e-8), -1, 1)

        # Spread state: tight (1 tick) / normal (2) / wide (3+)
        spread_state = 0.0 if current_spread <= 1.0 else (0.5 if current_spread <= 2.0 else 1.0)

        # Time in current spread state
        same_spread_count = 0
        for s in reversed(self.spread_history):
            if abs(s - current_spread) < 0.1:
                same_spread_count += 1
            else:
                break
        time_in_spread = min(same_spread_count, 20) / 20.0

        # ── Fill probability estimation ──
        # Based on queue depth and trade velocity
        window_duration_s = 0.0
        if len(timestamps_ns) > 1:
            window_duration_s = max(1e-9, (timestamps_ns[-1] - timestamps_ns[0]) / 1e9)

        trade_rate_per_s = (n_trade_bid + n_trade_ask) / max(window_duration_s, 0.001)
        total_queue_depth = self.bid_depth_l1 + self.ask_depth_l1 + 1e-8

        # P(fill) ~ trade_rate * time / queue_depth
        fill_prob_1s = np.clip(trade_rate_per_s * 1.0 / total_queue_depth, 0, 1)
        fill_prob_3s = np.clip(trade_rate_per_s * 3.0 / total_queue_depth, 0, 1)
        fill_prob_10s = np.clip(trade_rate_per_s * 10.0 / total_queue_depth, 0, 1)

        # Queue consumption velocity
        q_consume_bid = vol_trade_ask / max(window_duration_s, 0.001)  # sells consume bid queue
        q_consume_ask = vol_trade_bid / max(window_duration_s, 0.001)  # buys consume ask queue

        # Queue replenishment: adds vs consumed
        total_added = vol_add_bid + vol_add_ask + 1e-8
        total_consumed = vol_trade_bid + vol_trade_ask + 1e-8
        queue_replenish_ratio = np.clip(total_added / total_consumed, 0, 5) / 5.0

        # ── Adverse selection features ──
        # Post-trade price drift (using prices within this window)
        trade_bid_prices = prices[trade_mask & bid_mask]
        trade_ask_prices = prices[trade_mask & ask_mask]

        # Mean drift after aggressive buy/sell
        post_drift_bid = float(np.mean(trade_bid_prices)) if len(trade_bid_prices) > 0 else 0.0
        post_drift_ask = float(np.mean(trade_ask_prices)) if len(trade_ask_prices) > 0 else 0.0

        # Toxicity: how much price moves against passive side
        toxicity_imb = np.clip((post_drift_bid + post_drift_ask) / 2.0, -5, 5) / 5.0

        # Large trade fraction (informed flow proxy)
        large_trade_frac = large_vol / total_trade_vol

        # Informed flow: large trades in one direction
        large_mask = (trade_mask) & (qtys >= LARGE_THRESHOLD)
        large_bid_vol = qtys[large_mask & bid_mask].sum()
        large_ask_vol = qtys[large_mask & ask_mask].sum()
        informed_flow = np.clip(
            (large_bid_vol - large_ask_vol) / (large_bid_vol + large_ask_vol + 1e-8), -1, 1
        )

        # Adverse selection asymmetry: are fills more toxic on one side?
        adverse_asym = np.clip(post_drift_bid - post_drift_ask, -5, 5) / 5.0

        # ── Orderflow dynamics ──
        current_imb = np.clip(
            (vol_trade_bid - vol_trade_ask) / total_trade_vol, -1, 1
        )

        # Momentum = current - previous
        trade_imb_momentum = current_imb - self.prev_trade_imb
        # Acceleration = momentum change
        prev_momentum = self.prev_trade_imb - self.prev_prev_trade_imb
        trade_imb_accel = trade_imb_momentum - prev_momentum

        # Update history
        self.prev_prev_trade_imb = self.prev_trade_imb
        self.prev_trade_imb = current_imb

        # Volume-weighted imbalance (accounts for trade size)
        vw_imb = np.clip(
            (vol_trade_bid - vol_trade_ask) / total_trade_vol, -1, 1
        )

        # Small vs large trade imbalance
        small_bid = qtys[(trade_mask & bid_mask) & (qtys < LARGE_THRESHOLD)].sum()
        small_ask = qtys[(trade_mask & ask_mask) & (qtys < LARGE_THRESHOLD)].sum()
        small_imb = (small_bid - small_ask) / (small_bid + small_ask + 1e-8)
        large_imb = (large_bid_vol - large_ask_vol) / (large_bid_vol + large_ask_vol + 1e-8)
        small_vs_large = np.clip(small_imb - large_imb, -2, 2) / 2.0

        # Cancel velocity (pulling liquidity = informed)
        cancel_vel_bid = vol_cancel_bid / max(window_duration_s, 0.001)
        cancel_vel_ask = vol_cancel_ask / max(window_duration_s, 0.001)

        # Add/cancel ratio at BBO
        ac_ratio_bid = vol_add_bid / (vol_cancel_bid + 1e-8)
        ac_ratio_ask = vol_add_ask / (vol_cancel_ask + 1e-8)

        # ── Update depth state (simple tracking) ──
        self.bid_depth_l1 = max(0, self.bid_depth_l1 + vol_add_bid - vol_cancel_bid - vol_trade_ask)
        self.ask_depth_l1 = max(0, self.ask_depth_l1 + vol_add_ask - vol_cancel_ask - vol_trade_bid)

        # L2-L5 depth (orders further from BBO)
        at_bbo = np.abs(prices) < 0.5
        far_bid = add_mask & bid_mask & ~at_bbo
        far_ask = add_mask & ask_mask & ~at_bbo
        self.bid_depth_far += qtys[far_bid].sum()
        self.ask_depth_far += qtys[far_ask].sum()
        # Decay far depth
        self.bid_depth_far *= 0.95
        self.ask_depth_far *= 0.95

        # Depth features
        total_depth = self.bid_depth_l1 + self.ask_depth_l1 + 1e-8
        depth_imb_l1 = (self.bid_depth_l1 - self.ask_depth_l1) / total_depth

        # Depth restoration speed (how fast does depth recover after trades)
        depth_restoration = np.clip(
            (vol_add_bid + vol_add_ask) / (vol_trade_bid + vol_trade_ask + 1e-8), 0, 5
        ) / 5.0

        # Layering score (lots of depth far from BBO = layering/spoofing proxy)
        layering_bid = np.clip(self.bid_depth_far / (self.bid_depth_l1 + 1e-8), 0, 10) / 10.0
        layering_ask = np.clip(self.ask_depth_far / (self.ask_depth_l1 + 1e-8), 0, 10) / 10.0

        # ── Time of day features ──
        if len(timestamps_ns) > 0:
            # Convert nanoseconds to time of day
            ts_s = timestamps_ns[len(timestamps_ns)//2] / 1e9
            # Assume UTC, convert to ET (UTC-4 or UTC-5)
            et_offset = 4 * 3600  # EDT
            tod_s = (ts_s - et_offset) % 86400
            tod_hours = tod_s / 3600.0

            # Circular encoding
            tod_sin = np.sin(2 * np.pi * tod_hours / 24.0)
            tod_cos = np.cos(2 * np.pi * tod_hours / 24.0)

            # Minutes from open (9:30 ET = 9.5 * 60 = 570 min)
            minutes_from_open = max(0, (tod_hours - 9.5) * 60)
            minutes_from_open = np.clip(minutes_from_open / 390.0, 0, 1)  # 6.5hr session

            # Session progress
            session_progress = minutes_from_open
        else:
            tod_sin, tod_cos, minutes_from_open, session_progress = 0.0, 0.0, 0.0, 0.0

        # ── Meta features ──
        event_rate = len(events) / max(window_duration_s, 0.001)
        trade_rate = (n_trade_bid + n_trade_ask) / max(window_duration_s, 0.001)
        cancel_rate = (n_cancel_bid + n_cancel_ask) / max(window_duration_s, 0.001)
        add_rate = (n_add_bid + n_add_ask) / max(window_duration_s, 0.001)

        # Book turnover: total volume flow relative to depth
        book_turnover = (vol_add_bid + vol_add_ask + vol_cancel_bid + vol_cancel_ask +
                        vol_trade_bid + vol_trade_ask) / total_depth

        # Price volatility within window
        if len(valid_spreads) > 2:
            price_vol = float(np.std(prices[trade_mask])) if trade_mask.sum() > 2 else 0.0
        else:
            price_vol = 0.0

        # ── Normalize and assemble ──
        features = np.array([
            # Fill probability (6)
            fill_prob_1s, fill_prob_3s, fill_prob_10s,
            np.clip(q_consume_bid / 100, 0, 1),
            np.clip(q_consume_ask / 100, 0, 1),
            queue_replenish_ratio,
            # Adverse selection (6)
            np.clip(post_drift_bid / 5, -1, 1),
            np.clip(post_drift_ask / 5, -1, 1),
            toxicity_imb,
            large_trade_frac,
            informed_flow,
            adverse_asym,
            # Orderflow dynamics (8)
            np.clip(trade_imb_momentum, -1, 1),
            np.clip(trade_imb_accel, -1, 1),
            vw_imb,
            small_vs_large,
            np.clip(cancel_vel_bid / 100, 0, 1),
            np.clip(cancel_vel_ask / 100, 0, 1),
            np.clip(ac_ratio_bid / 5, 0, 1),
            np.clip(ac_ratio_ask / 5, 0, 1),
            # Spread regime (6)
            np.clip(current_spread / 5, 0, 1),
            np.clip(spread_vol / 2, 0, 1),
            time_in_spread,
            spread_state,
            np.clip(spread_mean / 5, 0, 1),
            spread_trend,
            # Queue dynamics (8)
            np.clip(self.bid_depth_l1 / 500, 0, 1),
            np.clip(self.ask_depth_l1 / 500, 0, 1),
            np.clip(self.bid_depth_far / 1000, 0, 1),
            np.clip(self.ask_depth_far / 1000, 0, 1),
            depth_restoration,
            np.clip(depth_imb_l1, -1, 1),
            layering_bid,
            layering_ask,
            # Time of day (4)
            tod_sin, tod_cos,
            minutes_from_open,
            session_progress,
            # Meta (6)
            np.clip(event_rate / 5000, 0, 1),
            np.clip(trade_rate / 500, 0, 1),
            np.clip(cancel_rate / 2000, 0, 1),
            np.clip(add_rate / 2000, 0, 1),
            np.clip(book_turnover / 10, 0, 1),
            np.clip(price_vol / 5, 0, 1),
        ], dtype=np.float32)

        assert len(features) == self.NUM_FEATURES, f"Expected {self.NUM_FEATURES}, got {len(features)}"
        return features


def process_date(date_str: str) -> Optional[Dict]:
    """Process a single date's MBO data and extract execution features."""
    mbo_file = MBO_DIR / f"{date_str}_mbo_events.npz"
    out_file = OUTPUT_DIR / f"{date_str}_exec_features.npz"

    if out_file.exists():
        log.info(f"  {date_str}: already processed, skipping")
        return {"date": date_str, "status": "skipped"}

    if not mbo_file.exists():
        log.warning(f"  {date_str}: MBO file not found")
        return {"date": date_str, "status": "missing"}

    try:
        t0 = time.time()
        data = np.load(str(mbo_file))

        # Get event data — handle different array names
        events = None
        for key in ["events", "data", "arr_0"]:
            if key in data:
                events = data[key]
                break

        if events is None:
            log.warning(f"  {date_str}: No events array found (keys: {list(data.keys())})")
            return {"date": date_str, "status": "no_events"}

        # Get timestamps if available
        timestamps = None
        for key in ["timestamps", "timestamps_ns", "ts"]:
            if key in data:
                timestamps = data[key]
                break

        n_events = len(events)
        n_windows = n_events // DECISION_STRIDE

        if n_windows == 0:
            log.warning(f"  {date_str}: Too few events ({n_events})")
            return {"date": date_str, "status": "too_few"}

        extractor = ExecutionFeatureExtractor()
        extractor.reset()

        all_features = np.zeros((n_windows, ExecutionFeatureExtractor.NUM_FEATURES), dtype=np.float32)

        for i in range(n_windows):
            start = i * DECISION_STRIDE
            end = start + DECISION_STRIDE

            window_events = events[start:end]
            window_ts = timestamps[start:end] if timestamps is not None else np.arange(start, end)

            # Ensure correct shape (N, 6)
            if window_events.ndim == 1:
                continue
            if window_events.shape[1] < 6:
                # Pad with zeros if fewer columns
                padded = np.zeros((len(window_events), 6), dtype=window_events.dtype)
                padded[:, :window_events.shape[1]] = window_events
                window_events = padded

            all_features[i] = extractor.process_window(
                window_events[:, :6], window_ts, i
            )

        # Save
        np.savez_compressed(
            str(out_file),
            features=all_features,
            feature_names=ExecutionFeatureExtractor.FEATURE_NAMES,
            date=date_str,
            n_events=n_events,
            n_windows=n_windows,
            decision_stride=DECISION_STRIDE,
        )

        elapsed = time.time() - t0

        # Basic stats
        stats = {
            "date": date_str,
            "status": "ok",
            "n_events": int(n_events),
            "n_windows": int(n_windows),
            "elapsed_s": round(elapsed, 1),
            "features_mean": {
                name: round(float(all_features[:, j].mean()), 4)
                for j, name in enumerate(ExecutionFeatureExtractor.FEATURE_NAMES[:10])
            },
        }

        log.info(f"  {date_str}: {n_events:,} events → {n_windows} windows in {elapsed:.1f}s")
        del data, events, timestamps, all_features
        gc.collect()

        return stats

    except Exception as e:
        log.error(f"  {date_str}: ERROR — {e}")
        return {"date": date_str, "status": "error", "error": str(e)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=0,
                       help="Number of parallel workers (0=auto)")
    parser.add_argument("--dates", nargs="+", default=None,
                       help="Specific dates to process")
    args = parser.parse_args()

    if args.workers == 0:
        args.workers = max(1, os.cpu_count() - 2)

    # Find all dates
    if args.dates:
        dates = args.dates
    else:
        dates = sorted([
            f.stem.replace("_mbo_events", "")
            for f in MBO_DIR.glob("*_mbo_events.npz")
        ])

    log.info(f"═══ Execution Feature Engineering v1 ═══")
    log.info(f"  Dates: {len(dates)}")
    log.info(f"  Workers: {args.workers}")
    log.info(f"  Features: {ExecutionFeatureExtractor.NUM_FEATURES}")
    log.info(f"  Decision stride: {DECISION_STRIDE}")
    log.info(f"  Output: {OUTPUT_DIR}")

    t0 = time.time()
    results = []

    if args.workers == 1:
        for d in dates:
            results.append(process_date(d))
    else:
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            futures = {pool.submit(process_date, d): d for d in dates}
            for future in as_completed(futures):
                result = future.result()
                if result:
                    results.append(result)

    elapsed = time.time() - t0

    # Summary
    ok = [r for r in results if r and r.get("status") == "ok"]
    skipped = [r for r in results if r and r.get("status") == "skipped"]
    failed = [r for r in results if r and r.get("status") not in ("ok", "skipped")]

    log.info(f"\n═══ DONE ═══")
    log.info(f"  Processed: {len(ok)} dates")
    log.info(f"  Skipped: {len(skipped)} (already done)")
    log.info(f"  Failed: {len(failed)}")
    log.info(f"  Total time: {elapsed:.0f}s")

    if ok:
        total_events = sum(r["n_events"] for r in ok)
        total_windows = sum(r["n_windows"] for r in ok)
        log.info(f"  Total events: {total_events:,}")
        log.info(f"  Total windows: {total_windows:,}")

    if failed:
        log.warning(f"  Failed dates: {[r['date'] for r in failed]}")

    # Save summary
    summary = {
        "timestamp": datetime.now().isoformat(),
        "n_dates_processed": len(ok),
        "n_dates_skipped": len(skipped),
        "n_dates_failed": len(failed),
        "total_time_s": round(elapsed, 1),
        "feature_count": ExecutionFeatureExtractor.NUM_FEATURES,
        "feature_names": ExecutionFeatureExtractor.FEATURE_NAMES,
        "results": results,
    }

    with open(OUTPUT_DIR / "processing_summary.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)

    log.info(f"  Summary saved to {OUTPUT_DIR / 'processing_summary.json'}")


if __name__ == "__main__":
    main()
