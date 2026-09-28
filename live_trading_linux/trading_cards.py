#!/usr/bin/env python3
"""
trading_cards.py — Regime-conditional signal quality analysis for LGBM live trading.

Determines WHEN and HOW to trade by bucketing prediction quality across:
  1. Time of day  (30-min buckets, 9:30–16:00 ET)
  2. Spread regime (tight / normal / wide)
  3. Volatility regime (low / medium / high — rolling 5-min price range)
  4. Event rate regime (slow / normal / fast — events per second)

Outputs:
  - trading_cards_config.json   (machine-readable filters for live engine)
  - Human-readable summary to stdout

Usage:
  python -m live_trading_linux.trading_cards [--n-files 10] [--out-dir .]

Data source: Walk-forward NPZ files from mbo_events_LEAKY_PRE_APR19_DO_NOT_USE.
    NOTE: "LEAKY" refers to label construction leakage in the training pipeline,
    not to event/timestamp corruption. The raw events and timestamps are valid
    for this analysis; we only use labels for IC measurement (not for retraining).
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("trading_cards")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_handler)
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
DEFAULT_DATA_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_events_LEAKY_PRE_APR19_DO_NOT_USE")
DEFAULT_MODEL    = Path("/home/jupiter/Lvl3Quant/live_trading_linux/models/labels_10s_lgbm.pkl")
DEFAULT_CALIB    = Path("/home/jupiter/Lvl3Quant/live_trading_linux/models/labels_10s_calibration.json")
DEFAULT_OUT_DIR  = Path("/home/jupiter/Lvl3Quant/live_trading_linux")

# ---------------------------------------------------------------------------
# Event column indices  (NPZ 'events' array, shape (N, 6))
# ---------------------------------------------------------------------------
COL_TD_LOG   = 0   # time_delta_log
COL_ETYPE    = 1   # event_type_id (0=add, 1=cancel, 2=modify?, 3=trade, 4=?)
COL_SIDE     = 2   # side_id (0=bid, 1=ask)
COL_PRICE    = 3   # price_rel_ticks
COL_QTY_LOG  = 4   # qty_log
COL_SPREAD   = 5   # spread_ticks

# ---------------------------------------------------------------------------
# Regime thresholds
# ---------------------------------------------------------------------------
SPREAD_TIGHT  = 0.50   # spread_ticks < 0.50  → tight
SPREAD_WIDE   = 0.75   # spread_ticks > 0.75  → wide
                        # else                  → normal

EVENT_RATE_SLOW = 50    # events/sec < 50  → slow
EVENT_RATE_FAST = 200   # events/sec > 200 → fast

# Time-of-day buckets: 30-min intervals from 9:30 to 16:00 ET
TOD_BUCKETS = []
for h in range(9, 16):
    for m in (0, 30):
        start_min = h * 60 + m
        if start_min < 9 * 60 + 30:
            continue
        if start_min >= 16 * 60:
            continue
        TOD_BUCKETS.append(f"{h:02d}:{m:02d}")

# IC threshold for "tradeable" bucket
IC_TRADEABLE = 0.05


def load_test_day(npz_path: Path) -> Optional[dict]:
    """Load an NPZ file and extract only the last calendar day's events.

    Returns dict with keys: events, timestamps, labels_10s, date_str
    or None if the file has no valid labels or too few events.
    """
    d = np.load(npz_path, allow_pickle=True)
    events = d["events"]
    labels = d["labels_10s"]
    timestamps = d["timestamps"]

    if events.shape[0] < 10_000:
        return None
    if np.all(np.isnan(labels)):
        return None

    # Identify the last calendar day in the file
    last_ts = timestamps[-1]
    last_dt = datetime.datetime.fromtimestamp(last_ts / 1e9)
    day_start = datetime.datetime(last_dt.year, last_dt.month, last_dt.day, 0, 0)
    day_start_ns = int(day_start.timestamp() * 1e9)

    mask = timestamps >= day_start_ns
    if mask.sum() < 5_000:
        return None

    return {
        "events":     events[mask],
        "timestamps": timestamps[mask],
        "labels_10s": labels[mask],
        "date_str":   last_dt.strftime("%Y-%m-%d"),
    }


def compute_features_batch(events: np.ndarray) -> np.ndarray:
    """Compute 21-dim features via the batch reference implementation.

    Uses the same compute_derived() that shadow_batch uses, which is proven
    to match StreamingFeatures within float32 tolerance.
    """
    from live_trading_linux.streaming_features import _compute_derived_reference
    return _compute_derived_reference(events.astype(np.float64))


def compute_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rank correlation (IC) between predictions and labels.

    Skips NaN labels. Returns 0.0 if fewer than 100 valid pairs.
    """
    valid = ~np.isnan(labels) & ~np.isnan(preds)
    if valid.sum() < 100:
        return 0.0
    rho, _ = scipy_stats.spearmanr(preds[valid], labels[valid])
    return float(rho) if np.isfinite(rho) else 0.0


def compute_directional_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of predictions where sign(pred) == sign(label).

    Skips NaN labels and zero predictions/labels.
    """
    valid = ~np.isnan(labels) & ~np.isnan(preds) & (preds != 0) & (labels != 0)
    if valid.sum() < 100:
        return 0.0
    return float(np.mean(np.sign(preds[valid]) == np.sign(labels[valid])))


def timestamps_to_tod_bucket(timestamps: np.ndarray) -> np.ndarray:
    """Convert nanosecond timestamps to 30-min time-of-day bucket labels.

    Returns array of strings like '09:30', '10:00', etc.
    Assumes timestamps are in local time (ET).
    """
    # Convert ns → seconds, then extract hour and minute
    secs = timestamps / 1e9
    # Use vectorized datetime extraction
    # datetime.fromtimestamp is slow for millions; use modular arithmetic instead
    # Timestamps are already in ET (the data was recorded in ET)
    N = len(secs)
    buckets = np.empty(N, dtype="U5")

    # Batch convert: get time-of-day in seconds since midnight
    # Use the first timestamp to find the midnight offset
    first_dt = datetime.datetime.fromtimestamp(secs[0])
    midnight = datetime.datetime(first_dt.year, first_dt.month, first_dt.day)
    midnight_s = midnight.timestamp()

    tod_s = secs - midnight_s  # seconds since midnight

    for bucket_label in TOD_BUCKETS:
        h, m = int(bucket_label[:2]), int(bucket_label[3:])
        start_s = h * 3600 + m * 60
        end_s = start_s + 1800  # 30 min
        mask = (tod_s >= start_s) & (tod_s < end_s)
        buckets[mask] = bucket_label

    # Events outside 9:30-16:00 get labeled "other"
    buckets[buckets == ""] = "other"
    return buckets


def compute_event_rate(timestamps: np.ndarray, window_events: int = 500) -> np.ndarray:
    """Compute local event rate (events/sec) using a rolling window.

    For each event i, event_rate = window_events / (ts[i] - ts[max(0, i-window_events)])
    """
    N = len(timestamps)
    rates = np.zeros(N, dtype=np.float32)
    ts_sec = timestamps / 1e9

    # Vectorized: for i >= window_events, rate = window / (ts[i] - ts[i - window])
    if N > window_events:
        dt = ts_sec[window_events:] - ts_sec[:-window_events]
        dt = np.maximum(dt, 1e-6)  # avoid div-by-zero
        rates[window_events:] = window_events / dt

    # For early events, use available window
    for i in range(1, min(window_events, N)):
        dt = ts_sec[i] - ts_sec[0]
        if dt > 1e-6:
            rates[i] = (i + 1) / dt

    return rates


def compute_rolling_vol(prices: np.ndarray, timestamps: np.ndarray,
                        window_sec: float = 300.0) -> np.ndarray:
    """Compute rolling 5-min price range (max - min) as a volatility proxy.

    Uses a sliding window of `window_sec` seconds. Returns price range in ticks.
    """
    N = len(prices)
    vol = np.zeros(N, dtype=np.float32)
    ts_sec = timestamps / 1e9

    # For efficiency, use a pointer-based approach
    left = 0
    # Track running min/max with a deque would be ideal but for simplicity
    # we'll use a chunked approach: compute in blocks
    # Actually let's use a simple approach: for each event, find events in [t-300, t]
    # This is O(N) with two pointers since timestamps are sorted

    from collections import deque
    min_deque = deque()  # indices of potential minimums
    max_deque = deque()  # indices of potential maximums

    for i in range(N):
        t_cutoff = ts_sec[i] - window_sec

        # Remove elements outside window from front
        while min_deque and ts_sec[min_deque[0]] < t_cutoff:
            min_deque.popleft()
        while max_deque and ts_sec[max_deque[0]] < t_cutoff:
            max_deque.popleft()

        # Maintain monotonic deques
        while min_deque and prices[min_deque[-1]] >= prices[i]:
            min_deque.pop()
        while max_deque and prices[max_deque[-1]] <= prices[i]:
            max_deque.pop()

        min_deque.append(i)
        max_deque.append(i)

        vol[i] = prices[max_deque[0]] - prices[min_deque[0]]

    return vol


def classify_vol_regime(vol: np.ndarray) -> np.ndarray:
    """Classify volatility into low/medium/high using terciles."""
    p33 = np.percentile(vol[vol > 0], 33) if np.any(vol > 0) else 0
    p67 = np.percentile(vol[vol > 0], 67) if np.any(vol > 0) else 0
    regime = np.full(len(vol), "low", dtype="U6")
    regime[vol >= p33] = "medium"
    regime[vol >= p67] = "high"
    return regime, float(p33), float(p67)


def classify_spread_regime(spread_ticks: np.ndarray) -> np.ndarray:
    """Classify spread into tight/normal/wide."""
    regime = np.full(len(spread_ticks), "normal", dtype="U6")
    regime[spread_ticks < SPREAD_TIGHT] = "tight"
    regime[spread_ticks > SPREAD_WIDE] = "wide"
    return regime


def classify_event_rate_regime(rates: np.ndarray) -> np.ndarray:
    """Classify event rate into slow/normal/fast."""
    regime = np.full(len(rates), "normal", dtype="U6")
    regime[rates < EVENT_RATE_SLOW] = "slow"
    regime[rates > EVENT_RATE_FAST] = "fast"
    return regime


def analyze_regime(preds: np.ndarray, labels: np.ndarray,
                   regime_labels: np.ndarray, regime_name: str) -> dict:
    """Compute IC and directional accuracy per regime bucket."""
    unique_regimes = sorted(set(regime_labels))
    results = {}
    for r in unique_regimes:
        mask = regime_labels == r
        n = mask.sum()
        if n < 500:
            results[r] = {"ic": 0.0, "dir_acc": 0.0, "n": int(n), "pct": 0.0}
            continue
        ic = compute_ic(preds[mask], labels[mask])
        da = compute_directional_accuracy(preds[mask], labels[mask])
        results[r] = {
            "ic":      round(ic, 4),
            "dir_acc": round(da, 4),
            "n":       int(n),
            "pct":     round(100.0 * n / len(preds), 1),
        }
    return results


def process_day(day_data: dict, inf) -> dict:
    """Process one day's data: features → predictions → regime analysis."""
    events = day_data["events"]
    timestamps = day_data["timestamps"]
    labels = day_data["labels_10s"]
    date_str = day_data["date_str"]
    N = len(events)

    log.info("Processing %s: %d events", date_str, N)

    # 1. Compute 21-dim features (batch)
    t0 = time.time()
    feats = compute_features_batch(events)
    log.info("  Features computed in %.1fs", time.time() - t0)

    # 2. Batch predict
    t0 = time.time()
    preds = inf.predict_batch(feats).astype(np.float32)
    log.info("  Predictions computed in %.1fs", time.time() - t0)

    # 3. Overall IC
    overall_ic = compute_ic(preds, labels)
    overall_da = compute_directional_accuracy(preds, labels)
    log.info("  Overall IC=%.4f, DirAcc=%.4f", overall_ic, overall_da)

    # 4. Regime classification
    t0 = time.time()

    # Time of day
    tod_buckets = timestamps_to_tod_bucket(timestamps)

    # Spread regime (column 5 = spread_ticks)
    spread_regimes = classify_spread_regime(events[:, COL_SPREAD])

    # Event rate
    event_rates = compute_event_rate(timestamps)
    erate_regimes = classify_event_rate_regime(event_rates)

    # Volatility (rolling 5-min price range)
    # price_rel_ticks is relative, so accumulate for absolute price path
    prices_rel = events[:, COL_PRICE]
    prices_abs = np.cumsum(prices_rel)  # reconstruct price level from relative ticks
    vol = compute_rolling_vol(prices_abs, timestamps, window_sec=300.0)
    vol_regimes, vol_p33, vol_p67 = classify_vol_regime(vol)

    log.info("  Regimes classified in %.1fs", time.time() - t0)

    # 5. Analyze each regime dimension
    tod_results    = analyze_regime(preds, labels, tod_buckets,    "time_of_day")
    spread_results = analyze_regime(preds, labels, spread_regimes, "spread")
    erate_results  = analyze_regime(preds, labels, erate_regimes,  "event_rate")
    vol_results    = analyze_regime(preds, labels, vol_regimes,    "volatility")

    return {
        "date":       date_str,
        "n_events":   int(N),
        "overall_ic": round(overall_ic, 4),
        "overall_da": round(overall_da, 4),
        "pred_mean":  round(float(np.mean(preds)), 6),
        "pred_std":   round(float(np.std(preds)), 6),
        "time_of_day":  tod_results,
        "spread":       spread_results,
        "event_rate":   erate_results,
        "volatility":   vol_results,
        "vol_thresholds": {"p33": round(vol_p33, 2), "p67": round(vol_p67, 2)},
        "event_rate_stats": {
            "mean": round(float(np.mean(event_rates)), 1),
            "p10":  round(float(np.percentile(event_rates, 10)), 1),
            "p90":  round(float(np.percentile(event_rates, 90)), 1),
        },
    }


def aggregate_days(day_results: list[dict]) -> dict:
    """Aggregate per-day regime results into cross-day statistics."""
    agg = {
        "n_days":      len(day_results),
        "dates":       [d["date"] for d in day_results],
        "overall_ic":  round(np.mean([d["overall_ic"] for d in day_results]), 4),
        "overall_da":  round(np.mean([d["overall_da"] for d in day_results]), 4),
    }

    # For each regime dimension, average IC and dir_acc across days
    for dim in ["time_of_day", "spread", "event_rate", "volatility"]:
        # Collect all bucket keys across days
        all_keys = set()
        for d in day_results:
            all_keys.update(d[dim].keys())

        dim_agg = {}
        for key in sorted(all_keys):
            ics = []
            das = []
            ns  = []
            for d in day_results:
                if key in d[dim] and d[dim][key]["n"] >= 500:
                    ics.append(d[dim][key]["ic"])
                    das.append(d[dim][key]["dir_acc"])
                    ns.append(d[dim][key]["n"])
            if ics:
                dim_agg[key] = {
                    "ic_mean":     round(float(np.mean(ics)), 4),
                    "ic_std":      round(float(np.std(ics)), 4),
                    "ic_min":      round(float(np.min(ics)), 4),
                    "ic_max":      round(float(np.max(ics)), 4),
                    "dir_acc_mean": round(float(np.mean(das)), 4),
                    "n_days":      len(ics),
                    "avg_events":  int(np.mean(ns)),
                }
        agg[dim] = dim_agg

    return agg


def build_config(agg: dict, ic_threshold: float = IC_TRADEABLE) -> dict:
    """Build the machine-readable trading config from aggregated analysis."""
    config = {}

    # 1. Time-of-day filter: only buckets where mean IC > threshold
    tradeable_buckets = []
    if "time_of_day" in agg:
        for bucket, stats in sorted(agg["time_of_day"].items()):
            if bucket == "other":
                continue
            if stats["ic_mean"] > ic_threshold and stats["n_days"] >= 3:
                tradeable_buckets.append(bucket)

    config["time_of_day_filter"] = tradeable_buckets if tradeable_buckets else TOD_BUCKETS
    config["time_of_day_filter_active"] = bool(tradeable_buckets)

    # 2. Spread filter: max spread to trade in
    #    Find the widest spread regime that still has IC > threshold
    spread_max = 20.0  # default: no filter
    if "spread" in agg:
        # If wide spread regime has poor IC, cap at SPREAD_WIDE
        wide_stats = agg["spread"].get("wide", {})
        if wide_stats and wide_stats.get("ic_mean", 0) < ic_threshold:
            spread_max = SPREAD_WIDE
            # If normal also poor, cap at tight
            normal_stats = agg["spread"].get("normal", {})
            if normal_stats and normal_stats.get("ic_mean", 0) < ic_threshold:
                spread_max = SPREAD_TIGHT
    config["spread_max"] = spread_max

    # 3. Volatility regime filter
    vol_ok = []
    if "volatility" in agg:
        for regime, stats in agg["volatility"].items():
            if stats.get("ic_mean", 0) > ic_threshold and stats.get("n_days", 0) >= 3:
                vol_ok.append(regime)
    config["vol_regime_filter"] = vol_ok if vol_ok else ["low", "medium", "high"]
    config["vol_regime_filter_active"] = bool(vol_ok) and len(vol_ok) < 3

    # 4. Event rate filter
    erate_min = 0.0
    erate_max = 99999.0
    if "event_rate" in agg:
        slow_stats = agg["event_rate"].get("slow", {})
        if slow_stats and slow_stats.get("ic_mean", 0) < ic_threshold:
            erate_min = EVENT_RATE_SLOW
        fast_stats = agg["event_rate"].get("fast", {})
        if fast_stats and fast_stats.get("ic_mean", 0) < ic_threshold:
            erate_max = EVENT_RATE_FAST
    config["event_rate_min"] = erate_min
    config["event_rate_max"] = erate_max

    # Metadata
    config["generated_at"] = datetime.datetime.now().isoformat()
    config["n_days_analyzed"] = agg["n_days"]
    config["dates_analyzed"] = agg["dates"]
    config["overall_ic"] = agg["overall_ic"]
    config["overall_da"] = agg["overall_da"]
    config["ic_threshold"] = ic_threshold

    return config


def format_summary(agg: dict, config: dict) -> str:
    """Build a human-readable summary of the trading card analysis."""
    lines = []
    lines.append("=" * 72)
    lines.append("TRADING CARDS ANALYSIS — LGBM 10s Horizon")
    lines.append("=" * 72)
    lines.append(f"Days analyzed: {agg['n_days']}  ({', '.join(agg['dates'])})")
    lines.append(f"Overall IC (mean): {agg['overall_ic']:.4f}")
    lines.append(f"Overall Dir Accuracy: {agg['overall_da']:.4f}")
    lines.append("")

    # Time of day
    lines.append("-" * 72)
    lines.append("TIME OF DAY (30-min buckets, ET)")
    lines.append(f"{'Bucket':>8}  {'IC mean':>8}  {'IC std':>7}  {'DirAcc':>7}  {'Days':>4}  {'Avg N':>9}  {'Trade?':>6}")
    lines.append("-" * 72)
    if "time_of_day" in agg:
        for bucket in TOD_BUCKETS:
            if bucket in agg["time_of_day"]:
                s = agg["time_of_day"][bucket]
                tradeable = "YES" if bucket in config["time_of_day_filter"] else "no"
                lines.append(
                    f"{bucket:>8}  {s['ic_mean']:>8.4f}  {s['ic_std']:>7.4f}  "
                    f"{s['dir_acc_mean']:>7.4f}  {s['n_days']:>4}  {s['avg_events']:>9,}  "
                    f"{tradeable:>6}"
                )
    lines.append("")

    # Spread regime
    lines.append("-" * 72)
    lines.append(f"SPREAD REGIME  (tight < {SPREAD_TIGHT}, normal, wide > {SPREAD_WIDE} ticks)")
    lines.append(f"{'Regime':>8}  {'IC mean':>8}  {'IC std':>7}  {'DirAcc':>7}  {'Days':>4}  {'Avg N':>9}")
    lines.append("-" * 72)
    if "spread" in agg:
        for regime in ["tight", "normal", "wide"]:
            if regime in agg["spread"]:
                s = agg["spread"][regime]
                lines.append(
                    f"{regime:>8}  {s['ic_mean']:>8.4f}  {s['ic_std']:>7.4f}  "
                    f"{s['dir_acc_mean']:>7.4f}  {s['n_days']:>4}  {s['avg_events']:>9,}"
                )
    lines.append(f"  >> Max spread to trade: {config['spread_max']} ticks")
    lines.append("")

    # Volatility regime
    lines.append("-" * 72)
    lines.append("VOLATILITY REGIME  (rolling 5-min price range, tercile split)")
    lines.append(f"{'Regime':>8}  {'IC mean':>8}  {'IC std':>7}  {'DirAcc':>7}  {'Days':>4}  {'Avg N':>9}")
    lines.append("-" * 72)
    if "volatility" in agg:
        for regime in ["low", "medium", "high"]:
            if regime in agg["volatility"]:
                s = agg["volatility"][regime]
                lines.append(
                    f"{regime:>8}  {s['ic_mean']:>8.4f}  {s['ic_std']:>7.4f}  "
                    f"{s['dir_acc_mean']:>7.4f}  {s['n_days']:>4}  {s['avg_events']:>9,}"
                )
    active = config["vol_regime_filter_active"]
    lines.append(f"  >> Vol filter active: {active}")
    if active:
        lines.append(f"     Trade only in: {config['vol_regime_filter']}")
    lines.append("")

    # Event rate regime
    lines.append("-" * 72)
    lines.append(f"EVENT RATE REGIME  (slow < {EVENT_RATE_SLOW}, normal, fast > {EVENT_RATE_FAST} events/sec)")
    lines.append(f"{'Regime':>8}  {'IC mean':>8}  {'IC std':>7}  {'DirAcc':>7}  {'Days':>4}  {'Avg N':>9}")
    lines.append("-" * 72)
    if "event_rate" in agg:
        for regime in ["slow", "normal", "fast"]:
            if regime in agg["event_rate"]:
                s = agg["event_rate"][regime]
                lines.append(
                    f"{regime:>8}  {s['ic_mean']:>8.4f}  {s['ic_std']:>7.4f}  "
                    f"{s['dir_acc_mean']:>7.4f}  {s['n_days']:>4}  {s['avg_events']:>9,}"
                )
    lines.append(f"  >> Event rate filter: [{config['event_rate_min']}, {config['event_rate_max']}] events/sec")
    lines.append("")

    # Config summary
    lines.append("=" * 72)
    lines.append("LIVE TRADING CONFIG")
    lines.append("=" * 72)
    if config["time_of_day_filter_active"]:
        lines.append(f"  Trade hours:     {', '.join(config['time_of_day_filter'])}")
    else:
        lines.append(f"  Trade hours:     ALL (no bucket had IC > {config['ic_threshold']} consistently)")
    lines.append(f"  Max spread:      {config['spread_max']} ticks")
    lines.append(f"  Vol regimes:     {', '.join(config['vol_regime_filter'])}")
    lines.append(f"  Event rate:      [{config['event_rate_min']}, {config['event_rate_max']}]")
    lines.append("=" * 72)

    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Trading cards — regime-conditional signal analysis")
    ap.add_argument("--n-files", type=int, default=10,
                    help="Number of most recent NPZ files to analyze (default: 10)")
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--calibration", type=str, default=str(DEFAULT_CALIB))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--ic-threshold", type=float, default=IC_TRADEABLE,
                    help="Min IC for a regime bucket to be 'tradeable'")
    args = ap.parse_args()

    ic_threshold = args.ic_threshold

    data_dir = Path(args.data_dir)
    out_dir  = Path(args.out_dir)

    # Find NPZ files
    npz_files = sorted(data_dir.glob("*_mbo_events.npz"))
    if not npz_files:
        log.error("No NPZ files found in %s", data_dir)
        return 1

    # Take the last N files
    npz_files = npz_files[-args.n_files:]
    log.info("Found %d NPZ files, using last %d", len(list(data_dir.glob("*.npz"))), len(npz_files))

    # Load model
    from live_trading_linux.lgbm_inference import LGBMInference
    inf = LGBMInference(model_path=args.model, calibration_path=args.calibration)
    log.info("Loaded model: %s", args.model)

    # Process each day
    day_results = []
    for npz_path in npz_files:
        try:
            day_data = load_test_day(npz_path)
            if day_data is None:
                log.warning("Skipping %s (too few events or no valid labels)", npz_path.name)
                continue
            result = process_day(day_data, inf)
            day_results.append(result)
        except Exception as e:
            log.error("Failed to process %s: %s", npz_path.name, e, exc_info=True)
            continue

    if not day_results:
        log.error("No days processed successfully")
        return 1

    log.info("Processed %d days successfully", len(day_results))

    # Aggregate across days
    agg = aggregate_days(day_results)

    # Build config
    config = build_config(agg, ic_threshold=ic_threshold)

    # Write outputs
    config_path = out_dir / "trading_cards_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    log.info("Config written to %s", config_path)

    # Write detailed results
    detail_path = out_dir / "trading_cards_detail.json"
    output = {
        "aggregated": agg,
        "config":     config,
        "per_day":    day_results,
    }
    with open(detail_path, "w") as f:
        json.dump(output, f, indent=2, default=float)
    log.info("Detailed results written to %s", detail_path)

    # Print human-readable summary
    summary = format_summary(agg, config)
    print("\n" + summary)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
