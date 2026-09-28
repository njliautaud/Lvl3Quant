#!/usr/bin/env python3
"""
Signal Accuracy Diagnostic: Does the model predict direction correctly?
========================================================================

After v9's catastrophic failure (all 64 configs Sharpe -15 to -40), we need
to answer the fundamental question: does the model predict direction at all?

Method:
  1. Load pred_log_ret_1s predictions (every ~250ms stride)
  2. For each high-confidence signal, look forward in the tick stream
  3. Measure ACTUAL price change at +250ms, +500ms, +1s, +2s, +3s, +5s, +10s
  4. Report directional accuracy and average move in ticks

Key question: If we had zero-latency market entry, would we make money?
If not → model is dead. If yes → passive FIFO is the problem.

Author: Claude (diagnostic, 2026-07-03)
"""

import numpy as np
import os
import sys
import time
import glob
from collections import defaultdict

sys.path.insert(0, '/home/jupiter/Lvl3Quant/engines')
from tick_replay_fast import load_predictions, find_preprocessed, match_dates

PREPROC_DIR = "/home/jupiter/Lvl3Quant/data/preprocessed_mbo"
PRED_DIR = "/home/jupiter/Lvl3Quant/output/cnn_mamba_v3_3_uncertainty_weighted/oot_47day_perdate"

TICK_SIZE = 0.25
ACT_TRADE = 3
SIDE_BID = 0
SIDE_ASK = 1

# Horizons to measure (nanoseconds)
HORIZONS_NS = [
    250_000_000,      # 250ms
    500_000_000,      # 500ms
    1_000_000_000,    # 1s
    2_000_000_000,    # 2s
    3_000_000_000,    # 3s
    5_000_000_000,    # 5s
    10_000_000_000,   # 10s
    30_000_000_000,   # 30s
]
HORIZON_LABELS = ['250ms', '500ms', '1s', '2s', '3s', '5s', '10s', '30s']

THRESHOLDS = [0.2, 0.3, 0.4, 0.5, 0.6, 0.8, 1.0, 1.2]


def analyze_day(preproc_path, predictions):
    """
    For one day: at each prediction event, measure future price changes.

    The predictions are sampled every PRED_STRIDE=250 events from the MBO stream.
    So prediction[i] corresponds to event index i*250 in the MBO data.

    Returns: (pred_values, future_moves_ticks[n_preds x n_horizons])
    """
    mbo = np.load(preproc_path)
    ts_ns = mbo['ts_ns']
    action = mbo['action']
    price = mbo['price']

    n_events = len(ts_ns)
    n_preds = len(predictions)

    # Prediction event indices (every 250 events, window of 1500 events)
    PRED_STRIDE = 250
    PRED_WINDOW = 1500
    pred_event_indices = np.arange(n_preds) * PRED_STRIDE + PRED_WINDOW

    # Extract trade-only timestamps and prices for fast lookup
    trade_mask = (action == ACT_TRADE)
    trade_ts = ts_ns[trade_mask]
    trade_px = price[trade_mask]

    if len(trade_ts) == 0:
        return None

    # For each prediction, find the trade price at signal time and at future horizons
    n_horizons = len(HORIZONS_NS)
    future_moves = np.full((n_preds, n_horizons), np.nan)
    signal_prices = np.full(n_preds, np.nan)

    for i in range(n_preds):
        evt_idx = pred_event_indices[i]
        if evt_idx >= n_events:
            continue

        sig_ts = ts_ns[evt_idx]

        # Price at signal time = last trade price at or before signal event
        tidx = np.searchsorted(trade_ts, sig_ts, side='right') - 1
        if tidx < 0:
            continue

        sig_px = trade_px[tidx]
        signal_prices[i] = sig_px

        # Future prices at each horizon
        for h_idx, h_ns in enumerate(HORIZONS_NS):
            target_ts = sig_ts + h_ns
            fut_tidx = np.searchsorted(trade_ts, target_ts, side='right') - 1
            if fut_tidx > tidx:
                future_moves[i, h_idx] = (trade_px[fut_tidx] - sig_px) / TICK_SIZE

    return predictions, signal_prices, future_moves


def main():
    print("=" * 70)
    print("SIGNAL ACCURACY DIAGNOSTIC")
    print("Does the model predict direction correctly at signal time?")
    print("=" * 70)
    print()

    # Load data using engine's helpers
    preproc_map = find_preprocessed(PREPROC_DIR)
    pred_map = load_predictions(PRED_DIR, head='pred_log_ret_1s')
    matched = match_dates(preproc_map, pred_map)

    print(f"  Matched dates: {len(matched)}")
    if not matched:
        print("ERROR: No matched dates!")
        return

    # Process all days
    all_preds = []
    all_moves = []

    t0 = time.time()
    for di, (date_str, preproc_path, preds) in enumerate(matched):
        result = analyze_day(preproc_path, preds)
        if result is None:
            continue

        pred_vals, sig_prices, fut_moves = result
        valid = ~np.isnan(sig_prices)
        all_preds.append(pred_vals[valid])
        all_moves.append(fut_moves[valid])

        if (di + 1) % 5 == 0 or di == len(matched) - 1:
            elapsed = time.time() - t0
            print(f"  Processed {di+1}/{len(matched)} days [{elapsed:.1f}s]")

    elapsed = time.time() - t0
    print(f"\n  Total time: {elapsed:.1f}s")

    # Stack results
    preds = np.concatenate(all_preds)
    moves = np.vstack(all_moves)

    print(f"  Total valid predictions: {len(preds):,}")
    print(f"  Pred distribution: mean={preds.mean():.4f}, std={preds.std():.4f}")
    print()

    # =================================================================
    # ANALYSIS 1: SHORT signal directional accuracy
    # =================================================================
    print("=" * 70)
    print("ANALYSIS 1: DIRECTIONAL ACCURACY — SHORT signals (pred < -threshold)")
    print("  'Accuracy' = % of times price moved DOWN after signal")
    print("=" * 70)
    print()

    header = f"{'Thr':<6} {'N':>9}"
    for h in HORIZON_LABELS:
        header += f" {h:>7}"
    print(header)
    print("-" * len(header))

    for thr in THRESHOLDS:
        mask = preds < -thr
        n = mask.sum()
        if n < 100:
            continue

        row = f" {thr:<5.1f} {n:>9,}"
        for h_idx in range(len(HORIZONS_NS)):
            h_valid = ~np.isnan(moves[mask, h_idx])
            if h_valid.sum() < 50:
                row += f" {'N/A':>7}"
            else:
                # For shorts: accuracy = price went down (move < 0)
                acc = (moves[mask][h_valid, h_idx] < 0).mean()
                row += f" {acc:>6.1%}"
        print(row)

    print()

    # =================================================================
    # ANALYSIS 2: LONG signal directional accuracy
    # =================================================================
    print("=" * 70)
    print("ANALYSIS 2: DIRECTIONAL ACCURACY — LONG signals (pred > threshold)")
    print("  'Accuracy' = % of times price moved UP after signal")
    print("=" * 70)
    print()

    header = f"{'Thr':<6} {'N':>9}"
    for h in HORIZON_LABELS:
        header += f" {h:>7}"
    print(header)
    print("-" * len(header))

    for thr in THRESHOLDS:
        mask = preds > thr
        n = mask.sum()
        if n < 100:
            continue

        row = f" {thr:<5.1f} {n:>9,}"
        for h_idx in range(len(HORIZONS_NS)):
            h_valid = ~np.isnan(moves[mask, h_idx])
            if h_valid.sum() < 50:
                row += f" {'N/A':>7}"
            else:
                acc = (moves[mask][h_valid, h_idx] > 0).mean()
                row += f" {acc:>6.1%}"
        print(row)

    print()

    # =================================================================
    # ANALYSIS 3: Average move in ticks (favorable direction)
    # =================================================================
    print("=" * 70)
    print("ANALYSIS 3: AVG MOVE IN PREDICTED DIRECTION (ticks)")
    print("  SHORT: positive = price went down (good)")
    print("  LONG: positive = price went up (good)")
    print("  Need > +1.376t to cover market RT cost")
    print("=" * 70)
    print()

    for side_label in ["SHORT", "LONG"]:
        print(f"  --- {side_label} ---")
        header = f"  {'Thr':<5} {'N':>9}"
        for h in HORIZON_LABELS:
            header += f" {h:>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for thr in THRESHOLDS:
            if side_label == "SHORT":
                mask = preds < -thr
            else:
                mask = preds > thr
            n = mask.sum()
            if n < 100:
                continue

            row = f"  {thr:<5.1f} {n:>9,}"
            for h_idx in range(len(HORIZONS_NS)):
                h_valid = ~np.isnan(moves[mask, h_idx])
                if h_valid.sum() < 50:
                    row += f" {'N/A':>7}"
                else:
                    if side_label == "SHORT":
                        avg = -moves[mask][h_valid, h_idx].mean()  # Negate: down is good
                    else:
                        avg = moves[mask][h_valid, h_idx].mean()
                    row += f" {avg:>+6.3f}"
            print(row)
        print()

    # =================================================================
    # ANALYSIS 4: Net edge after costs
    # =================================================================
    print("=" * 70)
    print("ANALYSIS 4: NET EDGE AFTER MARKET RT COST (1.376 ticks)")
    print("  Positive = profitable with immediate market entry + market exit")
    print("=" * 70)
    print()

    COST_RT = 1.376  # market entry + market exit: spread + commission

    for side_label in ["SHORT", "LONG"]:
        print(f"  --- {side_label} ---")
        header = f"  {'Thr':<5} {'N':>9} {'N/day':>6}"
        for h in HORIZON_LABELS:
            header += f" {h:>7}"
        print(header)
        print("  " + "-" * (len(header) - 2))

        for thr in THRESHOLDS:
            if side_label == "SHORT":
                mask = preds < -thr
            else:
                mask = preds > thr
            n = mask.sum()
            n_per_day = n / len(matched)
            if n < 100:
                continue

            row = f"  {thr:<5.1f} {n:>9,} {n_per_day:>6.0f}"
            for h_idx in range(len(HORIZONS_NS)):
                h_valid = ~np.isnan(moves[mask, h_idx])
                if h_valid.sum() < 50:
                    row += f" {'N/A':>7}"
                else:
                    if side_label == "SHORT":
                        avg = -moves[mask][h_valid, h_idx].mean()
                    else:
                        avg = moves[mask][h_valid, h_idx].mean()
                    net = avg - COST_RT
                    row += f" {net:>+6.3f}"
            print(row)
        print()

    # =================================================================
    # ANALYSIS 5: MFE distribution (max favorable excursion across horizons)
    # =================================================================
    print("=" * 70)
    print("ANALYSIS 5: MFE DISTRIBUTION (approximate from measured horizons)")
    print("  How far does price move in our favor before reverting?")
    print("=" * 70)
    print()

    for thr in [0.4, 0.6, 0.8, 1.0]:
        for side_label in ["SHORT", "LONG"]:
            if side_label == "SHORT":
                mask = preds < -thr
            else:
                mask = preds > thr
            n = mask.sum()
            if n < 200:
                continue

            sub_moves = moves[mask]
            if side_label == "SHORT":
                # Favorable = negative moves (price down)
                favorable = -sub_moves
            else:
                favorable = sub_moves

            # MFE = max across horizons
            mfe = np.nanmax(favorable, axis=1)
            mfe_valid = mfe[~np.isnan(mfe)]

            if len(mfe_valid) == 0:
                continue

            print(f"  {side_label} |pred|>{thr} (N={n:,}, {n/len(matched):.0f}/day):")
            print(f"    MFE percentiles: "
                  f"p25={np.percentile(mfe_valid, 25):+.2f}t, "
                  f"p50={np.percentile(mfe_valid, 50):+.2f}t, "
                  f"p75={np.percentile(mfe_valid, 75):+.2f}t, "
                  f"p90={np.percentile(mfe_valid, 90):+.2f}t, "
                  f"p95={np.percentile(mfe_valid, 95):+.2f}t")
            print(f"    >= 2t: {(mfe_valid >= 2).mean()*100:.1f}%  "
                  f">= 3t: {(mfe_valid >= 3).mean()*100:.1f}%  "
                  f">= 4t: {(mfe_valid >= 4).mean()*100:.1f}%  "
                  f">= 6t: {(mfe_valid >= 6).mean()*100:.1f}%")
            print()

    # =================================================================
    # ANALYSIS 6: Adverse selection test
    # For PASSIVE entry: what's the move AFTER being filled?
    # Approximation: FIFO fill happens when price crosses our level,
    # meaning price was already moving against us. After fill, the adverse
    # move continues. This is WHY passive entry fails.
    # =================================================================
    print("=" * 70)
    print("ANALYSIS 6: COST STRUCTURE ANALYSIS")
    print("=" * 70)
    print()
    print("  Market entry costs: 1 tick spread + 0.376 tick commission")
    print("  Market exit costs:  1 tick spread + 0.376 tick commission")
    print("  Passive exit costs: 0.376 tick commission only")
    print()
    print("  Scenario A (market in + passive TP out):")
    print("    Entry cost = 1.0t, TP exit cost = 0.376t")
    print("    If avg favorable move at 1s = X ticks:")
    print("    Net per trade = X - 1.0 - 0.376 = X - 1.376")
    print()
    print("  Scenario B (market in + market out at time stop):")
    print("    Cost = 1.0 + 1.376 = 2.376 ticks")
    print("    Need avg move > 2.376t to break even")
    print()

    # Final verdict
    print("=" * 70)
    print("VERDICT")
    print("=" * 70)
    print()

    # Check best case: highest threshold, 1s horizon, shorts
    best_thr = 1.0
    mask = preds < -best_thr
    n = mask.sum()
    if n >= 100:
        h_1s = 2  # index for 1s horizon
        h_valid = ~np.isnan(moves[mask, h_1s])
        if h_valid.sum() > 50:
            avg_move_1s = -moves[mask][h_valid, h_1s].mean()
            acc_1s = (moves[mask][h_valid, h_1s] < 0).mean()
            print(f"  Best case (SHORT, |pred|>1.0, 1s horizon):")
            print(f"    N signals: {n:,} ({n/len(matched):.0f}/day)")
            print(f"    Directional accuracy: {acc_1s:.1%}")
            print(f"    Avg favorable move: {avg_move_1s:+.3f} ticks")
            print(f"    After market RT cost (1.376t): {avg_move_1s - 1.376:+.3f} ticks/trade")
            if avg_move_1s > 1.376:
                print(f"    ✅ MODEL HAS REAL EDGE — market entry viable!")
            elif avg_move_1s > 0:
                print(f"    ⚠️  Model predicts direction but edge < costs")
                print(f"    Need passive exit (TP) to capture, not market exit")
            else:
                print(f"    ❌ MODEL HAS NO DIRECTIONAL EDGE AT ALL")
            print()


if __name__ == "__main__":
    main()
