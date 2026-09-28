#!/usr/bin/env python3
"""
Big Move Analysis — Is there a signal for large price movements?
================================================================
Instead of predicting continuous returns (where edge is too small),
analyze whether our model predictions correlate with LARGE moves.

Key question: when the model predicts a large absolute return,
does the REALIZED absolute return also tend to be larger?

If yes → we can build a "big move imminent" detector and trade
only when large moves are predicted, potentially covering costs.

If no → the model's magnitude predictions are noise and we need
a fundamentally different approach.

Tests:
1. Does predicted magnitude correlate with realized magnitude?
2. Are the largest predicted moves also the largest realized moves?
3. If we trade only the top N% by |prediction|, does the average
   realized move scale enough to cover costs?
4. Is this effect regime-dependent (green vs red days)?

Author: Claude (autonomous research, 2026-07-02)
"""

import json
import logging
import sys
import time
from pathlib import Path
from collections import defaultdict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tick_level_replay import (
    ROOT, SMART_V3_DIR, OOT_PRED_DIR, OUTPUT_DIR,
    ES_TICK_VALUE, ES_RT_COMMISSION_TICKS,
    WINDOW_SIZE, STRIDE,
    get_oot_dates,
)

logging.basicConfig(
    format="%(asctime)s [BIG-MOVE] %(levelname)s %(message)s",
    level=logging.INFO,
)
log = logging.getLogger("BIG-MOVE")

COST_PASSIVE = ES_RT_COMMISSION_TICKS  # 0.376 ticks


def main():
    dates = get_oot_dates()
    log.info(f"Loading {len(dates)} days...")

    all_pred_1s = []
    all_pred_10s = []
    all_pred_30s = []
    all_label_1s = []
    all_label_10s = []
    all_label_30s = []
    all_dates = []
    all_vol = []

    for date_str in dates:
        mbo_path = SMART_V3_DIR / f"{date_str}_mbo_events.npz"
        pred_path = OOT_PRED_DIR / f"oot_{date_str}.npz"
        if not mbo_path.exists() or not pred_path.exists():
            continue

        try:
            mbo = np.load(str(mbo_path), allow_pickle=True)
            pred = np.load(str(pred_path), allow_pickle=True)

            if 'pred_log_ret_1s' not in pred:
                continue

            p1s = pred['pred_log_ret_1s'].astype(np.float64)
            n_pred = len(p1s)
            n_events = len(mbo['timestamps'])
            pred_indices = np.arange(WINDOW_SIZE - 1, n_events, STRIDE)[:n_pred]

            l1s = mbo['labels_1s'][pred_indices].astype(np.float64)
            l10s = mbo['labels_10s'][pred_indices].astype(np.float64)
            l30s = mbo['labels_30s'][pred_indices].astype(np.float64)

            # Optional: load multi-horizon predictions
            p10s = pred.get('pred_log_ret_10s', np.full(n_pred, np.nan)).astype(np.float64)[:n_pred]
            p30s = pred.get('pred_log_ret_30s', np.full(n_pred, np.nan)).astype(np.float64)[:n_pred]

            # Trailing vol
            vol_window = 240
            l1s_all = mbo['labels_1s'].astype(np.float64)
            trailing_vol = np.full(n_pred, np.nan)
            for i in range(vol_window, n_pred):
                start_idx = pred_indices[i - vol_window]
                end_idx = pred_indices[i]
                if end_idx < len(l1s_all):
                    segment = l1s_all[start_idx:end_idx]
                    valid_seg = segment[~np.isnan(segment)]
                    if len(valid_seg) > 10:
                        trailing_vol[i] = np.std(valid_seg)

            all_pred_1s.append(p1s)
            all_pred_10s.append(p10s)
            all_pred_30s.append(p30s)
            all_label_1s.append(l1s)
            all_label_10s.append(l10s)
            all_label_30s.append(l30s)
            all_vol.append(trailing_vol)
            all_dates.extend([date_str] * n_pred)

            log.info(f"  {date_str}: {n_pred} predictions")

        except Exception as e:
            log.error(f"  {date_str}: {e}")

    # Concatenate all
    pred_1s = np.concatenate(all_pred_1s)
    pred_10s = np.concatenate(all_pred_10s)
    pred_30s = np.concatenate(all_pred_30s)
    label_1s = np.concatenate(all_label_1s)
    label_10s = np.concatenate(all_label_10s)
    label_30s = np.concatenate(all_label_30s)
    vol = np.concatenate(all_vol)
    dates_arr = np.array(all_dates)

    # Valid mask
    valid = ~np.isnan(pred_1s) & ~np.isnan(label_10s) & ~np.isnan(vol)
    log.info(f"Total valid: {valid.sum():,} / {len(valid):,}")

    p1v = pred_1s[valid]
    p10v = pred_10s[valid]
    p30v = pred_30s[valid]
    l10v = label_10s[valid]
    l30v = label_30s[valid]
    vol_v = vol[valid]
    dates_v = dates_arr[valid]

    # ================================================================
    # TEST 1: Prediction magnitude vs realized magnitude correlation
    # ================================================================
    print(f"\n{'='*80}")
    print("TEST 1: MAGNITUDE CORRELATION (|prediction| vs |realized|)")
    print(f"{'='*80}")

    abs_p1 = np.abs(p1v)
    abs_l10 = np.abs(l10v)
    abs_l30 = np.abs(l30v)

    corr_1s_10s = np.corrcoef(abs_p1, abs_l10)[0, 1]
    corr_1s_30s = np.corrcoef(abs_p1, abs_l30)[0, 1]
    print(f"  |pred_1s| vs |label_10s|: r = {corr_1s_10s:.4f}")
    print(f"  |pred_1s| vs |label_30s|: r = {corr_1s_30s:.4f}")

    if not np.all(np.isnan(p10v)):
        valid10 = ~np.isnan(p10v)
        corr_10s = np.corrcoef(np.abs(p10v[valid10]), abs_l10[valid10])[0, 1]
        print(f"  |pred_10s| vs |label_10s|: r = {corr_10s:.4f}")

    if not np.all(np.isnan(p30v)):
        valid30 = ~np.isnan(p30v)
        corr_30s = np.corrcoef(np.abs(p30v[valid30]), abs_l30[valid30])[0, 1]
        print(f"  |pred_30s| vs |label_30s|: r = {corr_30s:.4f}")

    # ================================================================
    # TEST 2: Quantile analysis — do biggest predictions = biggest moves?
    # ================================================================
    print(f"\n{'='*80}")
    print("TEST 2: QUANTILE ANALYSIS (10s horizon)")
    print("        When we sort by |prediction|, does realized |move| scale?")
    print(f"{'='*80}")
    print(f"{'Quantile':>12} {'N':>8} {'Avg|pred|':>10} {'Avg|real|':>10} {'AvgDirectional':>15} {'WR':>6}")
    print("-" * 70)

    # Sort by |pred_1s|
    quantiles = [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 99]
    for q in range(len(quantiles) - 1):
        lo = np.percentile(abs_p1, quantiles[q])
        hi = np.percentile(abs_p1, quantiles[q + 1])
        mask = (abs_p1 >= lo) & (abs_p1 < hi)
        if mask.sum() < 100:
            continue

        avg_pred = np.mean(abs_p1[mask])
        avg_real = np.mean(abs_l10[mask])

        # Directional PnL: trade in predicted direction
        dirs = np.sign(p1v[mask])
        pnl = dirs * l10v[mask]
        avg_dir = np.mean(pnl)
        wr = np.mean(pnl > 0)

        print(f"  {quantiles[q]}-{quantiles[q+1]}%  {mask.sum():>7} "
              f"{avg_pred:>9.4f} {avg_real:>9.4f} {avg_dir:>+14.4f} {wr:>5.1%}")

    # Top 1%
    top1 = abs_p1 >= np.percentile(abs_p1, 99)
    if top1.sum() > 10:
        avg_pred_t1 = np.mean(abs_p1[top1])
        avg_real_t1 = np.mean(abs_l10[top1])
        dirs_t1 = np.sign(p1v[top1])
        pnl_t1 = dirs_t1 * l10v[top1]
        avg_dir_t1 = np.mean(pnl_t1)
        wr_t1 = np.mean(pnl_t1 > 0)
        print(f"  {'99-100%':>8}  {top1.sum():>7} "
              f"{avg_pred_t1:>9.4f} {avg_real_t1:>9.4f} {avg_dir_t1:>+14.4f} {wr_t1:>5.1%}")

    # ================================================================
    # TEST 3: "Big move imminent" — binary target feasibility
    # ================================================================
    print(f"\n{'='*80}")
    print("TEST 3: BIG MOVE BINARY TARGET FEASIBILITY")
    print("        If we define 'big move' as |label_10s| > T ticks,")
    print("        does the model predict these more often when they occur?")
    print(f"{'='*80}")

    for threshold in [2, 3, 4, 5, 8, 10]:
        big_move = abs_l10 > threshold
        rate = np.mean(big_move)
        if big_move.sum() < 50:
            continue

        # Can the model predict big moves?
        # Split by prediction magnitude percentile
        for pct in [50, 75, 90, 95, 99]:
            cutoff = np.percentile(abs_p1, pct)
            high_pred = abs_p1 >= cutoff
            big_given_high = np.mean(big_move[high_pred])
            big_given_low = np.mean(big_move[~high_pred])
            lift = big_given_high / big_given_low if big_given_low > 0 else 0

            if pct == 90:
                print(f"  |move|>{threshold}t: base={rate:.1%}, "
                      f"top-{100-pct}%={big_given_high:.1%}, "
                      f"bottom-{pct}%={big_given_low:.1%}, "
                      f"lift={lift:.2f}x")

    # ================================================================
    # TEST 4: Vol as big-move predictor (is trailing vol enough?)
    # ================================================================
    print(f"\n{'='*80}")
    print("TEST 4: TRAILING VOL AS BIG-MOVE PREDICTOR")
    print("        Does high vol predict big moves better than the model?")
    print(f"{'='*80}")

    for threshold in [3, 4, 5]:
        big_move = abs_l10 > threshold
        if big_move.sum() < 50:
            continue

        # Vol quantiles
        for vpct in [50, 75, 90, 95]:
            cutoff = np.percentile(vol_v, vpct)
            high_vol = vol_v >= cutoff
            big_given_hv = np.mean(big_move[high_vol])
            big_given_lv = np.mean(big_move[~high_vol])
            vol_lift = big_given_hv / big_given_lv if big_given_lv > 0 else 0

            if vpct == 90:
                print(f"  |move|>{threshold}t: top-10% vol → {big_given_hv:.1%} "
                      f"(vs {big_given_lv:.1%} baseline), lift={vol_lift:.2f}x")

    # ================================================================
    # TEST 5: COMBINED model + vol for big move detection
    # ================================================================
    print(f"\n{'='*80}")
    print("TEST 5: COMBINED MODEL + VOL FOR BIG MOVE DETECTION")
    print("        Top 10% |prediction| AND top 10% vol → big move rate?")
    print(f"{'='*80}")

    for threshold in [3, 4, 5]:
        big_move = abs_l10 > threshold
        if big_move.sum() < 50:
            continue

        # Model only (top 10%)
        model_cutoff = np.percentile(abs_p1, 90)
        vol_cutoff = np.percentile(vol_v, 90)

        model_only = abs_p1 >= model_cutoff
        vol_only = vol_v >= vol_cutoff
        both = model_only & vol_only

        base_rate = np.mean(big_move)
        model_rate = np.mean(big_move[model_only])
        vol_rate = np.mean(big_move[vol_only])
        both_rate = np.mean(big_move[both]) if both.sum() > 10 else 0
        both_n = both.sum()

        # If we trade when big move predicted correctly, what's the PnL?
        if both.sum() > 10:
            dirs = np.sign(p1v[both])
            pnl = dirs * l10v[both]
            avg_pnl = np.mean(pnl)
            wr = np.mean(pnl > 0)
            net = avg_pnl - COST_PASSIVE
            print(f"  |move|>{threshold}t: base={base_rate:.1%}, "
                  f"model_top10%={model_rate:.1%}, vol_top10%={vol_rate:.1%}, "
                  f"BOTH={both_rate:.1%} (n={both_n})")
            print(f"    → Trade PnL: gross={avg_pnl:+.3f}t, net={net:+.3f}t, WR={wr:.1%}")

    # ================================================================
    # TEST 6: Directional edge BY |prediction| magnitude
    # ================================================================
    print(f"\n{'='*80}")
    print("TEST 6: DIRECTIONAL EDGE BY PREDICTION MAGNITUDE")
    print("        Is the directional edge larger for bigger |predictions|?")
    print(f"{'='*80}")
    print(f"{'Bucket':>15} {'N':>7} {'Gross':>8} {'Net':>8} {'WR':>6} {'|Real|':>7}")
    print("-" * 55)

    for pct_lo, pct_hi in [(0,50), (50,75), (75,90), (90,95), (95,99), (99,100)]:
        lo = np.percentile(abs_p1, pct_lo)
        hi = np.percentile(abs_p1, pct_hi) if pct_hi < 100 else np.inf
        mask = (abs_p1 >= lo) & (abs_p1 < hi)
        if mask.sum() < 100:
            continue

        dirs = np.sign(p1v[mask])
        pnl = dirs * l10v[mask]
        avg_pnl = np.mean(pnl)
        net = avg_pnl - COST_PASSIVE
        wr = np.mean(pnl > 0)
        avg_abs_real = np.mean(abs_l10[mask])

        prof = '💰' if net > 0 else ''
        print(f"  {pct_lo}-{pct_hi}%  {mask.sum():>7} "
              f"{avg_pnl:>+7.3f}t {net:>+7.3f}t {wr:>5.1%} {avg_abs_real:>6.3f}t {prof}")

    # ================================================================
    # SUMMARY
    # ================================================================
    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")
    print(f"  Total predictions analyzed: {valid.sum():,}")
    print(f"  |pred_1s| vs |label_10s| correlation: {corr_1s_10s:.4f}")
    print(f"  Passive RT cost: {COST_PASSIVE:.3f} ticks")

    # Save
    output_path = OUTPUT_DIR / "big_move_analysis.json"
    with open(str(output_path), 'w') as f:
        json.dump({
            'n_valid': int(valid.sum()),
            'magnitude_corr_1s_10s': round(corr_1s_10s, 4),
            'magnitude_corr_1s_30s': round(corr_1s_30s, 4),
        }, f, indent=2)
    log.info(f"Saved to {output_path}")


if __name__ == '__main__':
    main()
