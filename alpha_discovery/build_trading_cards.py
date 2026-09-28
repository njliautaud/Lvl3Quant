#!/usr/bin/env python3
"""
build_trading_cards.py — Comprehensive regime-conditional signal analysis.

Builds 4 "trading cards" that describe WHEN the LGBM model works best:

  1. Time-of-Day Card:     IC by 30-min bucket (9:30-16:00 ET)
  2. Spread Regime Card:   IC by bid-ask spread (tight / normal / wide)
  3. Volatility Regime Card: IC by rolling 5-min realized vol (low / med / high)
  4. Confidence Calibration Card: IC, dir accuracy, avg PnL per confidence tier

Data source:
  NPZ files from mbo_events_LEAKY_PRE_APR19_DO_NOT_USE.  Despite the name,
  the raw events and timestamps are valid for regime analysis — only the label
  construction had leakage (which doesn't affect IC measurement here since we
  use the existing labels as ground truth, not for model retraining).

Outputs:
  - trading_cards.json         Machine-readable, all card data
  - trading_cards_summary.txt  Human-readable summary
  Both saved to: /home/jupiter/Lvl3Quant/live_trading_linux/cards/

Usage:
  python alpha_discovery/build_trading_cards.py [--n-files 20] [--horizon labels_10s]
"""

from __future__ import annotations

import argparse
import datetime
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Optional

import numpy as np
from scipy import stats as scipy_stats

# Ensure live_trading_linux is importable
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
log = logging.getLogger("build_trading_cards")
_handler = logging.StreamHandler(sys.stdout)
_handler.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
log.addHandler(_handler)
log.setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path("/home/jupiter/Lvl3Quant")
DEFAULT_DATA_DIR = ROOT / "data" / "processed" / "mbo_events_LEAKY_PRE_APR19_DO_NOT_USE"
DEFAULT_MODEL = ROOT / "live_trading_linux" / "models" / "labels_10s_lgbm.pkl"
DEFAULT_CALIB = ROOT / "live_trading_linux" / "models" / "labels_10s_calibration.json"
DEFAULT_OUT_DIR = ROOT / "live_trading_linux" / "cards"

# ---------------------------------------------------------------------------
# Event column indices (NPZ 'events' array, shape (N, 6))
# ---------------------------------------------------------------------------
COL_TD_LOG = 0   # time_delta_log
COL_ETYPE = 1    # event_type_id
COL_SIDE = 2     # side_id
COL_PRICE = 3    # price_rel_ticks
COL_QTY_LOG = 4  # qty_log
COL_SPREAD = 5   # spread_ticks

# ---------------------------------------------------------------------------
# Regime thresholds
# ---------------------------------------------------------------------------
SPREAD_TIGHT = 0.50
SPREAD_WIDE = 0.75

TOD_BUCKETS = []
for _h in range(9, 16):
    for _m in (0, 30):
        start_min = _h * 60 + _m
        if start_min < 9 * 60 + 30 or start_min >= 16 * 60:
            continue
        TOD_BUCKETS.append(f"{_h:02d}:{_m:02d}")

# Confidence tiers: name -> (lower_percentile, upper_percentile)
CONFIDENCE_TIERS = {
    "all":   (0, 100),
    "top50": (50, 100),
    "top25": (75, 100),
    "top10": (90, 100),
    "top5":  (95, 100),
    "top1":  (99, 100),
}

IC_TRADEABLE = 0.05


# ===================================================================
# Helpers
# ===================================================================

def compute_ic(preds: np.ndarray, labels: np.ndarray) -> float:
    """Spearman rank IC, skipping NaN."""
    valid = ~np.isnan(labels) & ~np.isnan(preds)
    if valid.sum() < 100:
        return 0.0
    rho, _ = scipy_stats.spearmanr(preds[valid], labels[valid])
    return float(rho) if np.isfinite(rho) else 0.0


def compute_directional_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    """Fraction where sign(pred) == sign(label), skipping zeros."""
    valid = ~np.isnan(labels) & ~np.isnan(preds) & (preds != 0) & (labels != 0)
    if valid.sum() < 100:
        return 0.0
    return float(np.mean(np.sign(preds[valid]) == np.sign(labels[valid])))


def compute_win_rate(preds: np.ndarray, labels: np.ndarray) -> float:
    """Fraction of trades where pred*label > 0 (profitable direction).

    Unlike dir_acc, this counts zero-label outcomes as losses (no PnL)
    and only considers events where we would actually trade (pred != 0).
    """
    tradeable = ~np.isnan(labels) & ~np.isnan(preds) & (preds != 0)
    if tradeable.sum() < 100:
        return 0.0
    return float(np.mean((preds[tradeable] * labels[tradeable]) > 0))


def compute_sharpe_estimate(pnl: np.ndarray) -> float:
    """Annualized Sharpe from per-event PnL proxy (label * sign(pred))."""
    if len(pnl) < 100 or np.std(pnl) < 1e-10:
        return 0.0
    # Assume ~4M events/day, 252 trading days
    daily_factor = np.sqrt(252)
    return float(np.mean(pnl) / np.std(pnl) * daily_factor)


# ===================================================================
# Data loading
# ===================================================================

def load_day(npz_path: Path, horizon: str = "labels_10s") -> Optional[dict]:
    """Load an NPZ file and extract the last calendar day's events."""
    d = np.load(npz_path, allow_pickle=True)
    events = d["events"]
    labels = d[horizon]

    if "timestamps" not in d:
        log.warning("No timestamps in %s — skipping", npz_path.name)
        return None
    timestamps = d["timestamps"]

    if events.shape[0] < 10_000:
        return None
    if np.all(np.isnan(labels)):
        return None

    # Use the last calendar day in the file
    last_ts = timestamps[-1]
    last_dt = datetime.datetime.fromtimestamp(last_ts / 1e9)
    day_start = datetime.datetime(last_dt.year, last_dt.month, last_dt.day)
    day_start_ns = int(day_start.timestamp() * 1e9)

    mask = timestamps >= day_start_ns
    if mask.sum() < 5_000:
        return None

    return {
        "events": events[mask],
        "timestamps": timestamps[mask],
        "labels": labels[mask],
        "date_str": last_dt.strftime("%Y-%m-%d"),
    }


# ===================================================================
# Feature computation
# ===================================================================

def compute_features_batch(events: np.ndarray) -> np.ndarray:
    """Compute 21-dim features via the batch reference implementation."""
    from live_trading_linux.streaming_features import _compute_derived_reference
    return _compute_derived_reference(events.astype(np.float64))


# ===================================================================
# Regime classifiers
# ===================================================================

def timestamps_to_tod_bucket(timestamps: np.ndarray) -> np.ndarray:
    """Convert ns timestamps to 30-min time-of-day bucket labels."""
    secs = timestamps / 1e9
    N = len(secs)
    buckets = np.empty(N, dtype="U5")

    first_dt = datetime.datetime.fromtimestamp(secs[0])
    midnight = datetime.datetime(first_dt.year, first_dt.month, first_dt.day)
    midnight_s = midnight.timestamp()
    tod_s = secs - midnight_s

    for bucket_label in TOD_BUCKETS:
        h, m = int(bucket_label[:2]), int(bucket_label[3:])
        start_s = h * 3600 + m * 60
        end_s = start_s + 1800
        mask = (tod_s >= start_s) & (tod_s < end_s)
        buckets[mask] = bucket_label

    buckets[buckets == ""] = "other"
    return buckets


def classify_spread_regime(spread_ticks: np.ndarray) -> np.ndarray:
    regime = np.full(len(spread_ticks), "normal", dtype="U6")
    regime[spread_ticks < SPREAD_TIGHT] = "tight"
    regime[spread_ticks > SPREAD_WIDE] = "wide"
    return regime


def compute_rolling_vol(prices: np.ndarray, timestamps: np.ndarray,
                        window_sec: float = 300.0) -> np.ndarray:
    """Rolling 5-min price range (max - min) as volatility proxy.

    For large arrays (>1M events), uses a chunked approach that bins events
    into 1-second buckets and computes the rolling range over those buckets.
    This is O(N + B*W) instead of O(N*W) where B = num buckets, W = window.
    """
    N = len(prices)
    ts_sec = timestamps / 1e9

    if N > 500_000:
        # Fast path: bin into 1-second buckets, then use scipy rolling max/min
        t0 = ts_sec[0]
        bucket_ids = ((ts_sec - t0)).astype(np.int32)
        n_buckets = int(bucket_ids[-1]) + 1

        # Compute min/max price per bucket
        bucket_max = np.full(n_buckets, -np.inf, dtype=np.float32)
        bucket_min = np.full(n_buckets, np.inf, dtype=np.float32)
        np.maximum.at(bucket_max, bucket_ids, prices)
        np.minimum.at(bucket_min, bucket_ids, prices)

        # Fill empty buckets with previous values (forward fill)
        has_data = bucket_max > -np.inf
        for i in range(1, n_buckets):
            if not has_data[i]:
                bucket_max[i] = bucket_max[i - 1]
                bucket_min[i] = bucket_min[i - 1]

        # Causal rolling max/min over w=window_sec 1-second buckets
        # Use scipy maximum_filter1d with origin to make it causal (backward-looking)
        w = int(window_sec)
        from scipy.ndimage import maximum_filter1d, minimum_filter1d
        # origin = (w-1)//2 makes it look backward w samples
        # Actually for causal: pad front, filter, then trim
        pad_max = np.concatenate([np.full(w - 1, bucket_max[0]), bucket_max])
        pad_min = np.concatenate([np.full(w - 1, bucket_min[0]), bucket_min])
        roll_max = maximum_filter1d(pad_max, size=w)[w - 1:]
        roll_min = minimum_filter1d(pad_min, size=w)[w - 1:]

        vol_buckets = np.maximum(roll_max - roll_min, 0.0).astype(np.float32)

        # Map back to events
        vol = vol_buckets[bucket_ids]
        return vol
    else:
        # Small array: exact deque-based approach
        vol = np.zeros(N, dtype=np.float32)
        from collections import deque
        min_deque = deque()
        max_deque = deque()
        for i in range(N):
            t_cutoff = ts_sec[i] - window_sec
            while min_deque and ts_sec[min_deque[0]] < t_cutoff:
                min_deque.popleft()
            while max_deque and ts_sec[max_deque[0]] < t_cutoff:
                max_deque.popleft()
            while min_deque and prices[min_deque[-1]] >= prices[i]:
                min_deque.pop()
            while max_deque and prices[max_deque[-1]] <= prices[i]:
                max_deque.pop()
            min_deque.append(i)
            max_deque.append(i)
            vol[i] = prices[max_deque[0]] - prices[min_deque[0]]
        return vol


def classify_vol_regime(vol: np.ndarray) -> tuple:
    """Classify volatility into low/medium/high using terciles."""
    valid = vol[vol > 0]
    p33 = float(np.percentile(valid, 33)) if len(valid) > 0 else 0.0
    p67 = float(np.percentile(valid, 67)) if len(valid) > 0 else 0.0
    regime = np.full(len(vol), "low", dtype="U6")
    regime[vol >= p33] = "medium"
    regime[vol >= p67] = "high"
    return regime, p33, p67


# ===================================================================
# Card builders
# ===================================================================

def analyze_regime_card(preds, labels, regime_labels, regime_name):
    """Compute IC, dir accuracy, win rate per regime bucket."""
    unique = sorted(set(regime_labels))
    results = {}
    for r in unique:
        mask = regime_labels == r
        n = int(mask.sum())
        if n < 500:
            results[r] = {"ic": 0.0, "dir_acc": 0.0, "win_rate": 0.0, "n": n, "pct": 0.0}
            continue
        ic = compute_ic(preds[mask], labels[mask])
        da = compute_directional_accuracy(preds[mask], labels[mask])
        wr = compute_win_rate(preds[mask], labels[mask])
        results[r] = {
            "ic": round(ic, 4),
            "dir_acc": round(da, 4),
            "win_rate": round(wr, 4),
            "n": n,
            "pct": round(100.0 * n / len(preds), 1),
        }
    return results


def build_confidence_card(preds, labels):
    """Card 4: Confidence calibration — IC and PnL per tier."""
    abs_preds = np.abs(preds)
    results = {}

    for tier, (lo_pct, hi_pct) in CONFIDENCE_TIERS.items():
        lo_thresh = np.percentile(abs_preds, lo_pct) if lo_pct > 0 else 0.0
        mask = abs_preds >= lo_thresh
        n = int(mask.sum())

        if n < 100:
            results[tier] = {
                "ic": 0.0, "dir_acc": 0.0, "win_rate": 0.0,
                "avg_pnl_proxy": 0.0, "sharpe_est": 0.0,
                "n": n, "pct": round(100.0 * n / len(preds), 1),
                "abs_pred_threshold": round(float(lo_thresh), 6),
            }
            continue

        p = preds[mask]
        l = labels[mask]

        ic = compute_ic(p, l)
        da = compute_directional_accuracy(p, l)
        wr = compute_win_rate(p, l)

        # PnL proxy: label * sign(pred) — what you'd earn by trading in pred direction
        valid = ~np.isnan(l) & ~np.isnan(p) & (p != 0)
        if valid.sum() > 100:
            pnl = l[valid] * np.sign(p[valid])
            avg_pnl = float(np.mean(pnl))
            sharpe = compute_sharpe_estimate(pnl)
        else:
            avg_pnl = 0.0
            sharpe = 0.0

        results[tier] = {
            "ic": round(ic, 4),
            "dir_acc": round(da, 4),
            "win_rate": round(wr, 4),
            "avg_pnl_proxy": round(avg_pnl, 4),
            "sharpe_est": round(sharpe, 2),
            "n": n,
            "pct": round(100.0 * n / len(preds), 1),
            "abs_pred_threshold": round(float(lo_thresh), 6),
        }

    return results


# ===================================================================
# Per-day processing
# ===================================================================

def process_day(day_data: dict, inf) -> dict:
    """Run all 4 cards on one day's data."""
    events = day_data["events"]
    timestamps = day_data["timestamps"]
    labels = day_data["labels"]
    date_str = day_data["date_str"]
    N = len(events)

    log.info("Processing %s: %d events", date_str, N)

    # 1. Compute features and predict
    t0 = time.time()
    feats = compute_features_batch(events)
    log.info("  Features: %.1fs", time.time() - t0)

    t0 = time.time()
    preds = inf.predict_batch(feats).astype(np.float32)
    log.info("  Predictions: %.1fs", time.time() - t0)

    # Overall stats
    overall_ic = compute_ic(preds, labels)
    overall_da = compute_directional_accuracy(preds, labels)
    log.info("  Overall IC=%.4f  DirAcc=%.4f", overall_ic, overall_da)

    # 2. Regime classification
    t0 = time.time()

    tod_buckets = timestamps_to_tod_bucket(timestamps)
    spread_regimes = classify_spread_regime(events[:, COL_SPREAD])

    prices_abs = np.cumsum(events[:, COL_PRICE])
    vol = compute_rolling_vol(prices_abs, timestamps, window_sec=300.0)
    vol_regimes, vol_p33, vol_p67 = classify_vol_regime(vol)

    log.info("  Regimes: %.1fs", time.time() - t0)

    # 3. Build cards
    tod_card = analyze_regime_card(preds, labels, tod_buckets, "time_of_day")
    spread_card = analyze_regime_card(preds, labels, spread_regimes, "spread")
    vol_card = analyze_regime_card(preds, labels, vol_regimes, "volatility")
    conf_card = build_confidence_card(preds, labels)

    return {
        "date": date_str,
        "n_events": N,
        "overall_ic": round(overall_ic, 4),
        "overall_da": round(overall_da, 4),
        "pred_mean": round(float(np.mean(preds)), 6),
        "pred_std": round(float(np.std(preds)), 6),
        "time_of_day": tod_card,
        "spread": spread_card,
        "volatility": vol_card,
        "confidence": conf_card,
        "vol_thresholds": {"p33": round(vol_p33, 2), "p67": round(vol_p67, 2)},
    }


# ===================================================================
# Cross-day aggregation
# ===================================================================

def aggregate_days(day_results: list[dict]) -> dict:
    """Aggregate per-day results into cross-day statistics."""
    agg = {
        "n_days": len(day_results),
        "dates": [d["date"] for d in day_results],
        "overall_ic": round(float(np.mean([d["overall_ic"] for d in day_results])), 4),
        "overall_da": round(float(np.mean([d["overall_da"] for d in day_results])), 4),
    }

    # Regime cards
    for dim in ["time_of_day", "spread", "volatility"]:
        all_keys = set()
        for d in day_results:
            all_keys.update(d[dim].keys())

        dim_agg = {}
        for key in sorted(all_keys):
            ics, das, wrs, ns = [], [], [], []
            for d in day_results:
                if key in d[dim] and d[dim][key]["n"] >= 500:
                    ics.append(d[dim][key]["ic"])
                    das.append(d[dim][key]["dir_acc"])
                    wrs.append(d[dim][key]["win_rate"])
                    ns.append(d[dim][key]["n"])
            if ics:
                dim_agg[key] = {
                    "ic_mean": round(float(np.mean(ics)), 4),
                    "ic_std": round(float(np.std(ics)), 4),
                    "ic_min": round(float(np.min(ics)), 4),
                    "ic_max": round(float(np.max(ics)), 4),
                    "dir_acc_mean": round(float(np.mean(das)), 4),
                    "win_rate_mean": round(float(np.mean(wrs)), 4),
                    "n_days": len(ics),
                    "avg_events": int(np.mean(ns)),
                }
        agg[dim] = dim_agg

    # Confidence card
    tier_keys = list(CONFIDENCE_TIERS.keys())
    conf_agg = {}
    for tier in tier_keys:
        ics, das, wrs, pnls, sharpes, ns, threshs = [], [], [], [], [], [], []
        for d in day_results:
            if tier in d["confidence"] and d["confidence"][tier]["n"] >= 100:
                c = d["confidence"][tier]
                ics.append(c["ic"])
                das.append(c["dir_acc"])
                wrs.append(c["win_rate"])
                pnls.append(c["avg_pnl_proxy"])
                sharpes.append(c["sharpe_est"])
                ns.append(c["n"])
                threshs.append(c["abs_pred_threshold"])

        if ics:
            conf_agg[tier] = {
                "ic_mean": round(float(np.mean(ics)), 4),
                "ic_std": round(float(np.std(ics)), 4),
                "dir_acc_mean": round(float(np.mean(das)), 4),
                "win_rate_mean": round(float(np.mean(wrs)), 4),
                "avg_pnl_proxy": round(float(np.mean(pnls)), 4),
                "sharpe_est_mean": round(float(np.mean(sharpes)), 2),
                "sharpe_est_std": round(float(np.std(sharpes)), 2),
                "n_days": len(ics),
                "avg_events": int(np.mean(ns)),
                "avg_threshold": round(float(np.mean(threshs)), 6),
            }
    agg["confidence"] = conf_agg

    return agg


# ===================================================================
# Config builder (recommended live filters)
# ===================================================================

def build_recommended_config(agg: dict) -> dict:
    """Build machine-readable recommended filters from aggregated analysis."""
    config = {}

    # 1. Time-of-day: which buckets have IC > threshold?
    tradeable = []
    if "time_of_day" in agg:
        for bucket, stats in sorted(agg["time_of_day"].items()):
            if bucket == "other":
                continue
            if stats["ic_mean"] > IC_TRADEABLE and stats["n_days"] >= 3:
                tradeable.append(bucket)
    config["active_hours"] = tradeable if tradeable else TOD_BUCKETS
    config["active_hours_filter_on"] = bool(tradeable)

    # 2. Spread filter
    spread_max = 20.0
    if "spread" in agg:
        wide = agg["spread"].get("wide", {})
        if wide and wide.get("ic_mean", 0) < IC_TRADEABLE:
            spread_max = SPREAD_WIDE
            normal = agg["spread"].get("normal", {})
            if normal and normal.get("ic_mean", 0) < IC_TRADEABLE:
                spread_max = SPREAD_TIGHT
    config["spread_max_ticks"] = spread_max

    # 3. Volatility filter
    vol_ok = []
    if "volatility" in agg:
        for regime, stats in agg["volatility"].items():
            if stats.get("ic_mean", 0) > IC_TRADEABLE and stats.get("n_days", 0) >= 3:
                vol_ok.append(regime)
    config["vol_regimes_allowed"] = vol_ok if vol_ok else ["low", "medium", "high"]
    config["vol_filter_on"] = bool(vol_ok) and len(vol_ok) < 3

    # 4. Confidence recommendation: highest tier where Sharpe > 1
    best_tier = "all"
    if "confidence" in agg:
        for tier in ["top1", "top5", "top10", "top25", "top50"]:
            if tier in agg["confidence"]:
                if agg["confidence"][tier].get("sharpe_est_mean", 0) > 1.0:
                    best_tier = tier
                    break
    config["recommended_confidence_tier"] = best_tier
    if "confidence" in agg and best_tier in agg["confidence"]:
        config["recommended_tier_threshold"] = agg["confidence"][best_tier].get("avg_threshold", 0.0)
    else:
        config["recommended_tier_threshold"] = 0.0

    config["generated_at"] = datetime.datetime.now().isoformat()
    config["n_days_analyzed"] = agg["n_days"]
    config["dates_analyzed"] = agg["dates"]
    config["overall_ic"] = agg["overall_ic"]
    config["ic_threshold_used"] = IC_TRADEABLE

    return config


# ===================================================================
# Human-readable summary
# ===================================================================

def format_summary(agg: dict, config: dict) -> str:
    lines = []
    w = 76

    lines.append("=" * w)
    lines.append("TRADING CARDS — LGBM Signal Regime Analysis")
    lines.append("=" * w)
    lines.append(f"Days analyzed: {agg['n_days']}  ({', '.join(agg['dates'])})")
    lines.append(f"Overall IC (mean across days): {agg['overall_ic']:.4f}")
    lines.append(f"Overall Directional Accuracy:  {agg['overall_da']:.4f}")
    lines.append("")

    # ---- Card 1: Time of Day ----
    lines.append("-" * w)
    lines.append("CARD 1: TIME OF DAY  (30-min buckets, ET)")
    lines.append(f"{'Bucket':>8}  {'IC':>7}  {'IC std':>7}  {'DirAcc':>7}  "
                 f"{'WinR':>6}  {'Days':>4}  {'Avg N':>9}  {'Trade?':>6}")
    lines.append("-" * w)
    if "time_of_day" in agg:
        for bucket in TOD_BUCKETS:
            if bucket in agg["time_of_day"]:
                s = agg["time_of_day"][bucket]
                trade = "YES" if bucket in config["active_hours"] else "no"
                lines.append(
                    f"{bucket:>8}  {s['ic_mean']:>7.4f}  {s['ic_std']:>7.4f}  "
                    f"{s['dir_acc_mean']:>7.4f}  {s['win_rate_mean']:>6.4f}  "
                    f"{s['n_days']:>4}  {s['avg_events']:>9,}  {trade:>6}"
                )
    if config["active_hours_filter_on"]:
        lines.append(f"  >> Recommended active hours: {', '.join(config['active_hours'])}")
    else:
        lines.append(f"  >> No time filter (no bucket consistently above IC {IC_TRADEABLE})")
    lines.append("")

    # ---- Card 2: Spread Regime ----
    lines.append("-" * w)
    lines.append(f"CARD 2: SPREAD REGIME  (tight < {SPREAD_TIGHT}, wide > {SPREAD_WIDE} ticks)")
    lines.append(f"{'Regime':>8}  {'IC':>7}  {'IC std':>7}  {'DirAcc':>7}  "
                 f"{'WinR':>6}  {'Days':>4}  {'Avg N':>9}")
    lines.append("-" * w)
    if "spread" in agg:
        for regime in ["tight", "normal", "wide"]:
            if regime in agg["spread"]:
                s = agg["spread"][regime]
                lines.append(
                    f"{regime:>8}  {s['ic_mean']:>7.4f}  {s['ic_std']:>7.4f}  "
                    f"{s['dir_acc_mean']:>7.4f}  {s['win_rate_mean']:>6.4f}  "
                    f"{s['n_days']:>4}  {s['avg_events']:>9,}"
                )
    lines.append(f"  >> Max spread to trade: {config['spread_max_ticks']} ticks")
    lines.append("")

    # ---- Card 3: Volatility Regime ----
    lines.append("-" * w)
    lines.append("CARD 3: VOLATILITY REGIME  (rolling 5-min price range, tercile split)")
    lines.append(f"{'Regime':>8}  {'IC':>7}  {'IC std':>7}  {'DirAcc':>7}  "
                 f"{'WinR':>6}  {'Days':>4}  {'Avg N':>9}")
    lines.append("-" * w)
    if "volatility" in agg:
        for regime in ["low", "medium", "high"]:
            if regime in agg["volatility"]:
                s = agg["volatility"][regime]
                lines.append(
                    f"{regime:>8}  {s['ic_mean']:>7.4f}  {s['ic_std']:>7.4f}  "
                    f"{s['dir_acc_mean']:>7.4f}  {s['win_rate_mean']:>6.4f}  "
                    f"{s['n_days']:>4}  {s['avg_events']:>9,}"
                )
    if config["vol_filter_on"]:
        lines.append(f"  >> Trade only in: {', '.join(config['vol_regimes_allowed'])}")
    else:
        lines.append("  >> No vol filter active")
    lines.append("")

    # ---- Card 4: Confidence Calibration ----
    lines.append("-" * w)
    lines.append("CARD 4: CONFIDENCE CALIBRATION  (by |pred| percentile)")
    lines.append(f"{'Tier':>8}  {'IC':>7}  {'IC std':>7}  {'DirAcc':>7}  "
                 f"{'WinR':>6}  {'PnL':>7}  {'Sharpe':>7}  {'Thresh':>8}  "
                 f"{'Days':>4}  {'Avg N':>9}")
    lines.append("-" * w)
    if "confidence" in agg:
        for tier in ["all", "top50", "top25", "top10", "top5", "top1"]:
            if tier in agg["confidence"]:
                s = agg["confidence"][tier]
                lines.append(
                    f"{tier:>8}  {s['ic_mean']:>7.4f}  {s['ic_std']:>7.4f}  "
                    f"{s['dir_acc_mean']:>7.4f}  {s['win_rate_mean']:>6.4f}  "
                    f"{s['avg_pnl_proxy']:>7.4f}  {s['sharpe_est_mean']:>7.2f}  "
                    f"{s['avg_threshold']:>8.4f}  "
                    f"{s['n_days']:>4}  {s['avg_events']:>9,}"
                )
    lines.append(f"  >> Recommended tier: {config['recommended_confidence_tier']}  "
                 f"(threshold: {config['recommended_tier_threshold']:.4f})")
    lines.append("")

    # ---- Summary ----
    lines.append("=" * w)
    lines.append("RECOMMENDED LIVE CONFIG")
    lines.append("=" * w)
    if config["active_hours_filter_on"]:
        lines.append(f"  Trade hours:      {', '.join(config['active_hours'])}")
    else:
        lines.append("  Trade hours:      ALL RTH")
    lines.append(f"  Max spread:       {config['spread_max_ticks']} ticks")
    lines.append(f"  Vol regimes:      {', '.join(config['vol_regimes_allowed'])}")
    lines.append(f"  Confidence tier:  {config['recommended_confidence_tier']}")
    lines.append("=" * w)

    return "\n".join(lines)


# ===================================================================
# Main
# ===================================================================

def main() -> int:
    ap = argparse.ArgumentParser(description="Build trading cards — regime signal analysis")
    ap.add_argument("--n-files", type=int, default=20,
                    help="Number of most recent NPZ files to analyze (default: 20)")
    ap.add_argument("--data-dir", type=str, default=str(DEFAULT_DATA_DIR))
    ap.add_argument("--model", type=str, default=str(DEFAULT_MODEL))
    ap.add_argument("--calibration", type=str, default=str(DEFAULT_CALIB))
    ap.add_argument("--out-dir", type=str, default=str(DEFAULT_OUT_DIR))
    ap.add_argument("--horizon", type=str, default="labels_10s",
                    help="Label horizon to analyze (default: labels_10s)")
    args = ap.parse_args()

    data_dir = Path(args.data_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Find NPZ files
    npz_files = sorted(data_dir.glob("*_mbo_events.npz"))
    if not npz_files:
        log.error("No NPZ files found in %s", data_dir)
        return 1

    npz_files = npz_files[-args.n_files:]
    log.info("Using last %d of %d total NPZ files from %s",
             len(npz_files), len(list(data_dir.glob("*.npz"))), data_dir)

    # Load model
    from live_trading_linux.lgbm_inference import LGBMInference
    inf = LGBMInference(model_path=args.model, calibration_path=args.calibration)
    log.info("Model loaded: %s", args.model)

    # Process each day
    day_results = []
    for npz_path in npz_files:
        try:
            day_data = load_day(npz_path, horizon=args.horizon)
            if day_data is None:
                log.warning("Skipping %s", npz_path.name)
                continue
            result = process_day(day_data, inf)
            day_results.append(result)
        except Exception as e:
            log.error("Failed %s: %s", npz_path.name, e, exc_info=True)

    if not day_results:
        log.error("No days processed successfully")
        return 1

    log.info("Processed %d days", len(day_results))

    # Aggregate
    agg = aggregate_days(day_results)

    # Build config
    config = build_recommended_config(agg)

    # Write outputs
    cards_json = {
        "aggregated": agg,
        "recommended_config": config,
        "per_day": day_results,
    }

    json_path = out_dir / "trading_cards.json"
    with open(json_path, "w") as f:
        json.dump(cards_json, f, indent=2, default=float)
    log.info("Written: %s", json_path)

    summary = format_summary(agg, config)
    summary_path = out_dir / "trading_cards_summary.txt"
    with open(summary_path, "w") as f:
        f.write(summary + "\n")
    log.info("Written: %s", summary_path)

    print("\n" + summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
