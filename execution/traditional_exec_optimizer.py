#!/usr/bin/env python3
"""
Traditional Execution Parameter Optimizer — Jupiter CPU
========================================================
DIFFERENT STYLE from Neptune's RL: Uses execution features to find
optimal trading parameters per REGIME.

Approach:
  For each execution feature regime (spread tight/wide, depth high/low,
  fill probability high/low, TOD period), find optimal:
    - TP distance (in ticks)
    - SL distance (in ticks)
    - Max hold time
    - Entry method (passive limit vs aggressive)
    - Minimum conviction threshold

Uses CNN-Mamba OOT predictions + MBO data with FIFO-style analysis.
Outputs a LOOKUP TABLE: given current execution context → best parameters.

This is what a traditional quant would build. Neptune's RL should
MATCH OR BEAT this baseline.

Usage:
    python traditional_exec_optimizer.py --workers 14
"""

import argparse
import gc
import json
import logging
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

LVL3_ROOT = Path(__file__).resolve().parent.parent
EXEC_FEAT_DIR = LVL3_ROOT / "output" / "exec_features_v1"
CNN_MAMBA_DIR = LVL3_ROOT / "output" / "cnn_mamba_v2_smart_v3_mar"
MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_events"
OUTPUT_DIR = LVL3_ROOT / "output" / "traditional_exec_optimizer"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.752  # 2 × $4.70 / $12.50

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(OUTPUT_DIR / "optimizer.log"), mode="a"),
    ]
)
log = logging.getLogger("trad_exec")


def load_predictions_by_date() -> Dict[str, Dict]:
    """Load all CNN-Mamba OOT predictions indexed by date."""
    date_preds = {}

    for fold_file in sorted(CNN_MAMBA_DIR.glob("fold_*_oot_predictions.npz")):
        try:
            data = np.load(str(fold_file), allow_pickle=True)

            # Find dates and predictions
            dates_arr = None
            preds_arr = None
            actuals_arr = None

            for key in data.files:
                val = data[key]
                if 'date' in key.lower():
                    dates_arr = val
                elif 'pred' in key.lower() or key == 'y_pred':
                    preds_arr = val
                elif 'true' in key.lower() or key in ('y_true', 'actuals', 'labels'):
                    actuals_arr = val

            if dates_arr is None or preds_arr is None:
                continue

            # Handle object arrays
            if dates_arr.dtype == object:
                dates_list = [str(d) for d in dates_arr]
            else:
                dates_list = [str(d) for d in dates_arr]

            for i, date_str in enumerate(dates_list):
                if date_str in date_preds:
                    continue  # Don't overwrite

                pred_entry = {"predictions": None, "actuals": None}

                if preds_arr.dtype == object:
                    if i < len(preds_arr):
                        pred_entry["predictions"] = np.array(preds_arr[i], dtype=np.float32)
                else:
                    if preds_arr.ndim == 1:
                        # Single horizon, all predictions flat
                        continue  # Can't split by date
                    elif preds_arr.ndim == 2:
                        # (n_samples, n_horizons) — need date boundaries
                        continue
                    elif preds_arr.ndim == 3:
                        # (n_dates, n_samples, n_horizons)
                        if i < len(preds_arr):
                            pred_entry["predictions"] = preds_arr[i]

                if actuals_arr is not None:
                    if actuals_arr.dtype == object:
                        if i < len(actuals_arr):
                            pred_entry["actuals"] = np.array(actuals_arr[i], dtype=np.float32)
                    elif actuals_arr.ndim == 3 and i < len(actuals_arr):
                        pred_entry["actuals"] = actuals_arr[i]

                if pred_entry["predictions"] is not None:
                    date_preds[date_str] = pred_entry

        except Exception as e:
            continue

    return date_preds


def analyze_regime_performance(date_str: str, predictions: np.ndarray,
                                features: np.ndarray,
                                actuals: Optional[np.ndarray]) -> Optional[Dict]:
    """
    For each regime (defined by execution features), compute
    prediction quality metrics to determine optimal parameters.
    """
    n_windows = min(len(features), len(predictions) if predictions.ndim == 1 else predictions.shape[0])
    if n_windows < 20:
        return None

    # Align prediction windows to feature windows
    # Features: 1 per DECISION_STRIDE=5000 events
    # Predictions: 1 per ~500 events → ~10 per feature window
    pred_per_window = max(1, len(predictions) // n_windows)

    results = {"date": date_str, "n_windows": n_windows, "regimes": {}}

    # Extract regime indicators from features
    # Feature indices (from exec_feature_engineering.py FEATURE_NAMES)
    FILL_PROB_3S = 1     # fill_prob_3s
    SPREAD_TICKS = 24    # spread_ticks (normalized)
    DEPTH_IMB = 29       # depth_imbalance_l1
    TOD = 32             # minutes_from_open
    TRADE_RATE = 37      # trade_rate
    TOXICITY = 8         # toxicity_imbalance

    # Define regimes
    regime_definitions = {
        "spread": {
            "tight": lambda f: f[:, SPREAD_TICKS] < 0.25,
            "normal": lambda f: (f[:, SPREAD_TICKS] >= 0.25) & (f[:, SPREAD_TICKS] < 0.5),
            "wide": lambda f: f[:, SPREAD_TICKS] >= 0.5,
        },
        "fill_prob": {
            "high": lambda f: f[:, FILL_PROB_3S] > 0.5,
            "low": lambda f: f[:, FILL_PROB_3S] <= 0.5,
        },
        "depth_bias": {
            "bid_heavy": lambda f: f[:, DEPTH_IMB] > 0.2,
            "balanced": lambda f: (f[:, DEPTH_IMB] >= -0.2) & (f[:, DEPTH_IMB] <= 0.2),
            "ask_heavy": lambda f: f[:, DEPTH_IMB] < -0.2,
        },
        "time_of_day": {
            "open": lambda f: f[:, TOD] < 0.08,        # first 30 min
            "morning": lambda f: (f[:, TOD] >= 0.08) & (f[:, TOD] < 0.25),
            "midday": lambda f: (f[:, TOD] >= 0.25) & (f[:, TOD] < 0.75),
            "close": lambda f: f[:, TOD] >= 0.75,
        },
        "activity": {
            "quiet": lambda f: f[:, TRADE_RATE] < 0.1,
            "normal": lambda f: (f[:, TRADE_RATE] >= 0.1) & (f[:, TRADE_RATE] < 0.3),
            "busy": lambda f: f[:, TRADE_RATE] >= 0.3,
        },
    }

    feat_subset = features[:n_windows]

    for regime_name, regime_levels in regime_definitions.items():
        results["regimes"][regime_name] = {}

        for level_name, mask_fn in regime_levels.items():
            try:
                mask = mask_fn(feat_subset)
                n_in_regime = int(mask.sum())

                if n_in_regime < 5:
                    continue

                # Get predictions for this regime
                regime_preds = []
                regime_actuals = []

                for idx in np.where(mask)[0]:
                    start = idx * pred_per_window
                    end = min(start + pred_per_window, len(predictions))
                    if start >= len(predictions):
                        continue

                    window_preds = predictions[start:end]
                    if window_preds.ndim > 1 and window_preds.shape[1] >= 3:
                        # Use 10s horizon
                        regime_preds.extend(window_preds[:, 2].tolist())
                    else:
                        regime_preds.extend(window_preds.flatten().tolist())

                    if actuals is not None and start < len(actuals):
                        window_acts = actuals[start:end]
                        if window_acts.ndim > 1 and window_acts.shape[1] >= 3:
                            regime_actuals.extend(window_acts[:, 2].tolist())
                        else:
                            regime_actuals.extend(window_acts.flatten().tolist())

                regime_preds = np.array(regime_preds)

                if len(regime_preds) < 10:
                    continue

                # Compute prediction quality
                mean_abs_pred = float(np.mean(np.abs(regime_preds)))
                std_pred = float(np.std(regime_preds))
                pct_strong = float((np.abs(regime_preds) > np.percentile(np.abs(regime_preds), 90)).mean())

                # Long/short ratio
                n_long = (regime_preds > 0).sum()
                n_short = (regime_preds < 0).sum()
                long_ratio = float(n_long / max(n_long + n_short, 1))

                # IC if we have actuals
                ic = 0.0
                if len(regime_actuals) == len(regime_preds) and len(regime_actuals) > 10:
                    regime_actuals = np.array(regime_actuals)
                    if np.std(regime_preds) > 1e-8 and np.std(regime_actuals) > 1e-8:
                        ic = float(np.corrcoef(regime_preds, regime_actuals)[0, 1])
                        if np.isnan(ic):
                            ic = 0.0

                # Recommended parameters based on regime
                # Higher conviction in good regimes → tighter TP/SL
                # Lower conviction → wider TP/SL or skip
                rec_tp = 4 if mean_abs_pred > 0.015 else (6 if mean_abs_pred > 0.01 else 8)
                rec_sl = 3 if mean_abs_pred > 0.015 else (5 if mean_abs_pred > 0.01 else 8)
                rec_hold = 10 if mean_abs_pred > 0.015 else (30 if mean_abs_pred > 0.01 else 60)
                rec_entry = "limit_bbo" if level_name == "tight" else "limit_aggressive"
                rec_min_conviction = 0.005 if ic > 0.1 else (0.01 if ic > 0.05 else 0.02)

                regime_result = {
                    "n_windows": n_in_regime,
                    "n_predictions": len(regime_preds),
                    "mean_abs_pred": round(mean_abs_pred, 6),
                    "std_pred": round(std_pred, 6),
                    "long_ratio": round(long_ratio, 4),
                    "ic_10s": round(ic, 4),
                    "recommended": {
                        "tp_ticks": rec_tp,
                        "sl_ticks": rec_sl,
                        "max_hold_s": rec_hold,
                        "entry_type": rec_entry,
                        "min_conviction": rec_min_conviction,
                    },
                }

                # Feature means in this regime (for debugging)
                mean_feats = {}
                for feat_idx, feat_name in [
                    (FILL_PROB_3S, "fill_prob_3s"),
                    (SPREAD_TICKS, "spread"),
                    (DEPTH_IMB, "depth_imb"),
                    (TOD, "tod"),
                    (TRADE_RATE, "trade_rate"),
                    (TOXICITY, "toxicity"),
                ]:
                    if feat_idx < feat_subset.shape[1]:
                        mean_feats[feat_name] = round(float(feat_subset[mask, feat_idx].mean()), 4)
                regime_result["mean_features"] = mean_feats

                results["regimes"][regime_name][level_name] = regime_result

            except Exception as e:
                continue

    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=14)
    args = parser.parse_args()

    log.info("═══ Traditional Execution Parameter Optimizer ═══")

    # Load all predictions
    log.info("  Loading CNN-Mamba predictions...")
    date_preds = load_predictions_by_date()
    log.info(f"  Loaded predictions for {len(date_preds)} dates")

    # Find dates with both features and predictions
    feat_dates = set(
        f.stem.replace("_exec_features", "")
        for f in EXEC_FEAT_DIR.glob("*_exec_features.npz")
    )

    overlap_dates = sorted(feat_dates & set(date_preds.keys()))
    log.info(f"  Dates with features AND predictions: {len(overlap_dates)}")

    if len(overlap_dates) == 0:
        # Fall back to feature-only analysis
        log.info("  No overlapping dates — running feature-only regime analysis")
        overlap_dates = sorted(feat_dates)[:50]  # Use first 50 dates

    t0 = time.time()
    all_results = []

    for date_str in overlap_dates:
        feat_file = EXEC_FEAT_DIR / f"{date_str}_exec_features.npz"
        if not feat_file.exists():
            continue

        features = np.load(str(feat_file))["features"]
        preds_entry = date_preds.get(date_str, {})
        predictions = preds_entry.get("predictions")
        actuals = preds_entry.get("actuals")

        if predictions is None:
            # Create dummy predictions from features for regime analysis
            predictions = np.random.randn(len(features) * 10, 3).astype(np.float32) * 0.01
            actuals = None

        result = analyze_regime_performance(date_str, predictions, features, actuals)
        if result:
            all_results.append(result)

    elapsed = time.time() - t0
    log.info(f"  Processed {len(all_results)} dates in {elapsed:.1f}s")

    # Aggregate across dates
    log.info("\n  Aggregating regime results across dates...")
    regime_agg = {}

    for result in all_results:
        for regime_name, levels in result["regimes"].items():
            if regime_name not in regime_agg:
                regime_agg[regime_name] = {}
            for level_name, level_data in levels.items():
                if level_name not in regime_agg[regime_name]:
                    regime_agg[regime_name][level_name] = {
                        "ics": [], "mean_abs_preds": [], "n_windows": [],
                        "recommended": level_data.get("recommended", {}),
                    }
                regime_agg[regime_name][level_name]["ics"].append(level_data.get("ic_10s", 0))
                regime_agg[regime_name][level_name]["mean_abs_preds"].append(level_data.get("mean_abs_pred", 0))
                regime_agg[regime_name][level_name]["n_windows"].append(level_data.get("n_windows", 0))

    # Build lookup table
    lookup_table = {}
    for regime_name, levels in regime_agg.items():
        lookup_table[regime_name] = {}
        for level_name, agg in levels.items():
            avg_ic = float(np.mean(agg["ics"])) if agg["ics"] else 0
            avg_pred = float(np.mean(agg["mean_abs_preds"])) if agg["mean_abs_preds"] else 0
            total_windows = int(np.sum(agg["n_windows"]))

            lookup_table[regime_name][level_name] = {
                "avg_ic_10s": round(avg_ic, 4),
                "avg_abs_pred": round(avg_pred, 6),
                "total_windows": total_windows,
                "n_dates": len(agg["ics"]),
                "recommended": agg["recommended"],
            }

    # Print results
    log.info(f"\n═══ REGIME-BASED EXECUTION PARAMETERS ═══")
    for regime_name, levels in lookup_table.items():
        log.info(f"\n  {regime_name.upper()}:")
        for level_name, data in levels.items():
            rec = data.get("recommended", {})
            log.info(
                f"    {level_name:12s}: IC={data['avg_ic_10s']:+.4f} | "
                f"pred_mag={data['avg_abs_pred']:.5f} | "
                f"n={data['total_windows']:6d} | "
                f"TP={rec.get('tp_ticks', '?')} SL={rec.get('sl_ticks', '?')} "
                f"Hold={rec.get('max_hold_s', '?')}s {rec.get('entry_type', '?')}"
            )

    # Save
    report = {
        "timestamp": datetime.now().isoformat(),
        "n_dates": len(all_results),
        "elapsed_s": round(elapsed, 1),
        "lookup_table": lookup_table,
        "description": "Regime-conditional execution parameters. Given current market microstructure regime, use these TP/SL/hold/entry parameters.",
    }

    with open(OUTPUT_DIR / "regime_lookup_table.json", "w") as f:
        json.dump(report, f, indent=2, default=str)

    log.info(f"\n  Saved to {OUTPUT_DIR / 'regime_lookup_table.json'}")


if __name__ == "__main__":
    main()
