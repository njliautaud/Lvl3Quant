#!/usr/bin/env python3
"""
Signal-Conditional Retrace Analysis v1
Hypothesis: High-confidence short signals have HIGHER retrace rates (price drops 1 tick)
than the unconditional average (~85%).
"""

import numpy as np
from pathlib import Path

# Razer paths
PRED_DIR = r"C:\Users\claude\Lvl3Quant\output\cnn_mamba_v2_bulk_oot"
EVENT_DIR = r"C:\Users\claude\Lvl3Quant\data\processed\mbo_events_smart_v3"

def load_predictions(date_str):
    """Load predictions for a date. Returns (preds_1s, labels_1s) or (None, None)"""
    pred_file = Path(PRED_DIR) / f"{date_str}_predictions.npz"
    if not pred_file.exists():
        return None, None

    data = np.load(pred_file)

    # predictions shape: (N, 3) for [1s, 5s, 10s]
    preds_1s = data['predictions'][:, 0]  # Column 0 is 1s horizon
    labels_1s = data['labels'][:, 0]      # Column 0 is 1s horizon labels

    return preds_1s, labels_1s

def load_events(date_str):
    """Load MBO events for a date. Returns labels_1s or None"""
    event_file = Path(EVENT_DIR) / f"{date_str}_mbo_events.npz"
    if not event_file.exists():
        return None

    data = np.load(event_file)
    return data['labels_1s']

def compute_retrace_rates(preds, labels):
    """
    For SHORT signals (preds < 0), compute retrace rates across confidence buckets.
    Retrace = does price drop (labels[i:i+H].min() < 0) within H steps?
    Returns dict: {confidence_bucket: {horizon: retrace_rate}}
    """

    # Filter to SHORT signals (predictions < 0, expecting downward price move)
    short_mask = preds < 0
    short_preds = preds[short_mask]
    short_labels = labels[short_mask]

    if len(short_preds) == 0:
        return None

    # Confidence buckets (based on prediction magnitude, which correlates with model confidence)
    # Higher |prediction| = higher confidence
    abs_preds = np.abs(short_preds)

    buckets = {
        "top_1%": np.percentile(abs_preds, 99),
        "top_2%": np.percentile(abs_preds, 98),
        "top_5%": np.percentile(abs_preds, 95),
        "top_10%": np.percentile(abs_preds, 90),
        "top_50%": np.percentile(abs_preds, 50),
    }

    # Horizon definitions (in steps, ~250ms per step = 4 Hz)
    horizons = {
        "5s": int(5 * 4),      # 20 steps
        "10s": int(10 * 4),    # 40 steps
        "30s": int(30 * 4),    # 120 steps
        "60s": int(60 * 4),    # 240 steps
    }

    results = {}

    for bucket_name, conf_threshold in buckets.items():
        bucket_mask = (abs_preds >= conf_threshold)
        bucket_labels = short_labels[bucket_mask]

        if len(bucket_labels) == 0:
            results[bucket_name] = {h: None for h in horizons}
            continue

        retrace_rates = {}
        for horizon_name, horizon_steps in horizons.items():
            # For each event, check if price retraces down 1+ ticks within horizon
            retraces = 0
            valid_count = 0

            for i in range(len(bucket_labels)):
                end_idx = min(i + horizon_steps, len(bucket_labels))

                # If we don't have enough steps, count it as valid but might not retrace
                if end_idx - i < horizon_steps:
                    continue

                price_path = bucket_labels[i:end_idx]
                min_move = np.nanmin(price_path) if np.any(~np.isnan(price_path)) else np.nan

                if not np.isnan(min_move) and min_move <= -1.0:  # 1-tick downward retrace
                    retraces += 1
                    valid_count += 1
                elif not np.isnan(min_move):
                    valid_count += 1

            retrace_rates[horizon_name] = (retraces / valid_count * 100) if valid_count > 0 else None

        results[bucket_name] = retrace_rates

    # Add "all short signals" baseline
    all_short_rates = {}
    for horizon_name, horizon_steps in horizons.items():
        retraces = 0
        valid_count = 0
        for i in range(len(short_labels)):
            end_idx = min(i + horizon_steps, len(short_labels))
            if end_idx - i < horizon_steps:
                continue
            price_path = short_labels[i:end_idx]
            min_move = np.nanmin(price_path) if np.any(~np.isnan(price_path)) else np.nan
            if not np.isnan(min_move) and min_move <= -1.0:
                retraces += 1
                valid_count += 1
            elif not np.isnan(min_move):
                valid_count += 1
        all_short_rates[horizon_name] = (retraces / valid_count * 100) if valid_count > 0 else None

    results["all_shorts"] = all_short_rates

    return results

def main():
    pred_dates = sorted([f.stem.replace("_predictions", "") for f in Path(PRED_DIR).glob("*_predictions.npz")])

    if not pred_dates:
        print("❌ No prediction files found.")
        return

    print(f"\n📊 Signal-Conditional Retrace Analysis")
    print(f"   Found {len(pred_dates)} prediction dates\n")

    all_results = {}

    for date_str in pred_dates[:10]:  # Limit to first 10 dates
        preds, pred_labels = load_predictions(date_str)

        if preds is None:
            continue

        retrace_results = compute_retrace_rates(preds, pred_labels)

        if retrace_results:
            all_results[date_str] = retrace_results
            short_count = np.sum(preds < 0)
            print(f"  ✓ {date_str}: {short_count:,} short signals analyzed")

    if not all_results:
        print("❌ No results computed.")
        return

    # Aggregate across all dates
    print("\n" + "="*80)
    print("RETRACE RATES BY CONFIDENCE BUCKET (Short Signals Only)")
    print("="*80)
    print(f"{'Confidence':<15} {'5s':<10} {'10s':<10} {'30s':<10} {'60s':<10}")
    print("-"*80)

    agg_results = {}
    for bucket in ["all_shorts", "top_50%", "top_10%", "top_5%", "top_2%", "top_1%"]:
        rates = {h: [] for h in ["5s", "10s", "30s", "60s"]}

        for date_results in all_results.values():
            if bucket in date_results:
                for horizon in ["5s", "10s", "30s", "60s"]:
                    rate = date_results[bucket].get(horizon)
                    if rate is not None:
                        rates[horizon].append(rate)

        avg_rates = {h: np.mean(rates[h]) if rates[h] else None for h in ["5s", "10s", "30s", "60s"]}
        agg_results[bucket] = avg_rates

        rate_5s = f"{avg_rates['5s']:.1f}%" if avg_rates['5s'] else "N/A"
        rate_10s = f"{avg_rates['10s']:.1f}%" if avg_rates['10s'] else "N/A"
        rate_30s = f"{avg_rates['30s']:.1f}%" if avg_rates['30s'] else "N/A"
        rate_60s = f"{avg_rates['60s']:.1f}%" if avg_rates['60s'] else "N/A"

        print(f"{bucket:<15} {rate_5s:<10} {rate_10s:<10} {rate_30s:<10} {rate_60s:<10}")

    # Check hypothesis: does confidence correlate with retrace rate?
    print("\n" + "="*80)
    print("HYPOTHESIS CHECK: Confidence → Higher Retrace Rate?")
    print("="*80)

    baseline = agg_results.get("all_shorts", {}).get("30s")
    top1 = agg_results.get("top_1%", {}).get("30s")

    if baseline and top1:
        delta = top1 - baseline
        pct_change = (delta / baseline) * 100
        print(f"All short signals (30s retrace): {baseline:.1f}%")
        print(f"Top 1% confidence (30s retrace): {top1:.1f}%")
        print(f"Absolute change: {delta:+.1f}% ({pct_change:+.1f}%)")

        if pct_change > 5:
            print(f"\n✅ CONFIRMED: High-confidence signals have significantly higher retrace rates")
        elif pct_change > 0:
            print(f"\n⚠️  MARGINAL: Small improvement, confidence correlates weakly with retrace")
        else:
            print(f"\n❌ HYPOTHESIS REJECTED: Confidence does NOT predict higher retrace rates")

if __name__ == "__main__":
    main()
