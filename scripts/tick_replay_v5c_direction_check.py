#!/usr/bin/env python3
"""
V5c — Quick direction analysis of k=20 conviction signals.
Why are ALL 274 trades long? Are there no short streaks? Or are there
short streaks but the strategy was secretly filtering them?
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from tick_level_replay import (
    SMART_V3_DIR, OOT_PRED_DIR,
    WINDOW_SIZE, STRIDE,
    get_oot_dates,
)

dates = get_oot_dates()
print(f"Checking {len(dates)} dates...\n")

total_pos_20 = 0
total_neg_20 = 0
total_pos_trades = 0  # would-be long entries after k=20 positive streak
total_neg_trades = 0  # would-be short entries after k=20 negative streak
pos_pnl_sum = 0.0
neg_pnl_sum = 0.0

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

        preds = pred['pred_log_ret_1s'].astype(np.float64)
        n_pred = len(preds)
        n_events = len(mbo['timestamps'])
        pred_indices = np.arange(WINDOW_SIZE - 1, n_events, STRIDE)[:n_pred]
        labels_10s = mbo['labels_10s'][pred_indices].astype(np.float64)

        # Compute trailing vol
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

        directions = np.sign(preds)

        # Count streaks
        streak_count = 0
        streak_dir = 0
        last_trade_idx = -40
        day_pos = 0
        day_neg = 0

        for i in range(n_pred):
            if np.isnan(trailing_vol[i]) or np.isnan(labels_10s[i]):
                streak_count = 0
                streak_dir = 0
                continue

            d = directions[i]
            if d == 0:
                streak_count = 0
                streak_dir = 0
                continue

            if d == streak_dir:
                streak_count += 1
            else:
                streak_dir = d
                streak_count = 1

            if streak_count >= 20 and trailing_vol[i] >= 1.5 and i - last_trade_idx >= 20:
                if streak_dir == 1:
                    day_pos += 1
                    pos_pnl_sum += labels_10s[i]
                else:
                    day_neg += 1
                    neg_pnl_sum -= labels_10s[i]  # short profits from negative return
                last_trade_idx = i
                streak_count = 0

        total_pos_20 += day_pos
        total_neg_20 += day_neg
        if day_pos + day_neg > 0:
            print(f"  {date_str}: L={day_pos}, S={day_neg}")
    except Exception as e:
        print(f"  {date_str}: error - {e}")

print(f"\nSUMMARY (k=20, vol≥1.5, 10s horizon):")
print(f"  Long entries:  {total_pos_20}")
print(f"  Short entries: {total_neg_20}")
print(f"  Long avg PnL:  {pos_pnl_sum/max(total_pos_20,1):+.3f} ticks/trade")
print(f"  Short avg PnL: {neg_pnl_sum/max(total_neg_20,1):+.3f} ticks/trade")

# Also check prediction distribution
all_preds = []
for date_str in dates:
    pred_path = OOT_PRED_DIR / f"oot_{date_str}.npz"
    if not pred_path.exists():
        continue
    try:
        pred = np.load(str(pred_path), allow_pickle=True)
        if 'pred_log_ret_1s' in pred:
            all_preds.extend(pred['pred_log_ret_1s'].tolist())
    except:
        pass

all_preds = np.array(all_preds)
print(f"\nPrediction distribution:")
print(f"  N: {len(all_preds):,}")
print(f"  Mean: {np.mean(all_preds):+.4f}")
print(f"  Std: {np.std(all_preds):.4f}")
print(f"  % positive: {100 * np.mean(all_preds > 0):.1f}%")
print(f"  % negative: {100 * np.mean(all_preds < 0):.1f}%")
print(f"  Median: {np.median(all_preds):+.4f}")

# Check if bimodal positive mode dominates
print(f"\n  p10: {np.percentile(all_preds, 10):+.4f}")
print(f"  p25: {np.percentile(all_preds, 25):+.4f}")
print(f"  p50: {np.percentile(all_preds, 50):+.4f}")
print(f"  p75: {np.percentile(all_preds, 75):+.4f}")
print(f"  p90: {np.percentile(all_preds, 90):+.4f}")

# Fraction of consecutive-same-direction streaks ≥20 by direction
dirs = np.sign(all_preds)
streak_count = 0
streak_dir = 0
pos_streaks_20 = 0
neg_streaks_20 = 0

for i in range(len(dirs)):
    d = dirs[i]
    if d == 0:
        if streak_count >= 20:
            if streak_dir == 1:
                pos_streaks_20 += 1
            else:
                neg_streaks_20 += 1
        streak_count = 0
        streak_dir = 0
        continue
    if d == streak_dir:
        streak_count += 1
    else:
        if streak_count >= 20:
            if streak_dir == 1:
                pos_streaks_20 += 1
            else:
                neg_streaks_20 += 1
        streak_dir = d
        streak_count = 1

print(f"\n  Positive 20-streaks: {pos_streaks_20}")
print(f"  Negative 20-streaks: {neg_streaks_20}")
print(f"  Ratio: {pos_streaks_20/(neg_streaks_20+0.001):.1f}x more positive")
