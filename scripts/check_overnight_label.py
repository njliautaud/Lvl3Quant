#!/usr/bin/env python3
"""
Check overnight label contamination in lh_2h_enhanced_ic_push.py
================================================================
The forward label `df['close'].shift(-horizon_bars) - df['close']` is computed
on hourly bars sorted globally by timestamp. The last 1-2 bars of each trading
day will reference the NEXT day's bars, incorporating the overnight gap.

This script:
1. Loads data the same way as lh_2h_enhanced_ic_push.py
2. Identifies cross-day labels
3. Reports % of bars affected and label magnitude differences
4. Trains LGBM with and without cross-day bars to compare IC
"""

import sys
import gc
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.lh_2h_enhanced_ic_push import (
    load_minute_bars,
    compute_enhanced_hourly,
    add_rolling_features,
    add_regime_context,
    get_feature_cols,
    HORIZON_BARS,
    TRAIN_DAYS,
    PURGE_DAYS,
)


def main():
    print("=" * 70)
    print("OVERNIGHT LABEL CONTAMINATION CHECK")
    print("=" * 70)

    # ── Load & build hourly bars (same pipeline) ──
    print("\nLoading minute bars...")
    minutes = load_minute_bars()

    print("Computing hourly bars + features...")
    hourly = compute_enhanced_hourly(minutes)
    hourly = add_rolling_features(hourly)
    hourly = add_regime_context(hourly)
    del minutes; gc.collect()

    # ── Add forward labels (same as original) ──
    hourly['fwd_ticks'] = hourly['close'].shift(-HORIZON_BARS) - hourly['close']

    # ── Identify which bars have cross-day labels ──
    # The label at row i uses close at row i+HORIZON_BARS.
    # If the date at row i differs from the date at row i+HORIZON_BARS, the label
    # spans an overnight gap.
    hourly['label_source_date'] = hourly['date'].shift(-HORIZON_BARS)
    hourly['is_cross_day'] = hourly['date'] != hourly['label_source_date']

    # Drop rows where label is NaN (last HORIZON_BARS rows)
    labeled = hourly.dropna(subset=['fwd_ticks']).copy()

    n_total = len(labeled)
    n_cross = labeled['is_cross_day'].sum()
    n_within = n_total - n_cross
    pct_cross = n_cross / n_total * 100

    print(f"\n{'─' * 50}")
    print(f"CROSS-DAY LABEL STATISTICS")
    print(f"{'─' * 50}")
    print(f"Total labeled bars:     {n_total}")
    print(f"Within-day labels:      {n_within}  ({100 - pct_cross:.1f}%)")
    print(f"Cross-day labels:       {n_cross}  ({pct_cross:.1f}%)")
    print(f"Horizon bars:           {HORIZON_BARS}")

    cross = labeled[labeled['is_cross_day']]
    within = labeled[~labeled['is_cross_day']]

    print(f"\n{'─' * 50}")
    print(f"LABEL MAGNITUDE COMPARISON")
    print(f"{'─' * 50}")
    print(f"{'Metric':<30s} {'Within-day':>12s} {'Cross-day':>12s} {'Ratio':>8s}")
    print(f"{'─' * 62}")

    w_mean = within['fwd_ticks'].abs().mean()
    c_mean = cross['fwd_ticks'].abs().mean()
    print(f"{'Mean |label| (ticks)':<30s} {w_mean:>12.3f} {c_mean:>12.3f} {c_mean/w_mean:>8.2f}x")

    w_std = within['fwd_ticks'].std()
    c_std = cross['fwd_ticks'].std()
    print(f"{'Label std (ticks)':<30s} {w_std:>12.3f} {c_std:>12.3f} {c_std/w_std:>8.2f}x")

    w_med = within['fwd_ticks'].abs().median()
    c_med = cross['fwd_ticks'].abs().median()
    print(f"{'Median |label| (ticks)':<30s} {w_med:>12.3f} {c_med:>12.3f} {c_med/w_med:>8.2f}x")

    w_p90 = within['fwd_ticks'].abs().quantile(0.9)
    c_p90 = cross['fwd_ticks'].abs().quantile(0.9)
    print(f"{'P90 |label| (ticks)':<30s} {w_p90:>12.3f} {c_p90:>12.3f} {c_p90/w_p90:>8.2f}x")

    w_p99 = within['fwd_ticks'].abs().quantile(0.99)
    c_p99 = cross['fwd_ticks'].abs().quantile(0.99)
    print(f"{'P99 |label| (ticks)':<30s} {w_p99:>12.3f} {c_p99:>12.3f} {c_p99/w_p99:>8.2f}x")

    # Show which hours are affected
    print(f"\n{'─' * 50}")
    print(f"CROSS-DAY BARS BY HOUR")
    print(f"{'─' * 50}")
    if n_cross > 0:
        hour_dist = cross.groupby('hour').size()
        total_by_hour = labeled.groupby('hour').size()
        for h in sorted(hour_dist.index):
            cnt = hour_dist[h]
            tot = total_by_hour.get(h, cnt)
            print(f"  Hour {h:2d}:  {cnt:4d} cross-day bars out of {tot:4d}  ({cnt/tot*100:.1f}%)")

    # ── Train LGBM: all bars vs excluding cross-day ──
    print(f"\n{'=' * 70}")
    print("IC COMPARISON: ALL BARS vs WITHIN-DAY ONLY")
    print(f"{'=' * 70}")

    feature_cols = get_feature_cols(labeled)
    print(f"Features: {len(feature_cols)}")

    import lightgbm as lgb

    def run_wf(data, label, tag):
        """Sliding walk-forward, returns IC."""
        dates = sorted(data['date'].unique())
        all_preds, all_actuals = [], []

        for i in range(TRAIN_DAYS + PURGE_DAYS, len(dates)):
            oot_date = dates[i]
            train_end = i - PURGE_DAYS
            train_start = max(0, train_end - TRAIN_DAYS)
            train_dates = dates[train_start:train_end]

            train = data[data['date'].isin(train_dates)]
            oot = data[data['date'] == oot_date]

            if len(train) < 100 or len(oot) == 0:
                continue

            X_tr = train[feature_cols].fillna(0).values
            y_tr = train['fwd_ticks'].values
            X_ot = oot[feature_cols].fillna(0).values
            y_ot = oot['fwd_ticks'].values

            split = int(len(X_tr) * 0.8)
            if split < 20 or len(X_tr) - split < 5:
                continue

            model = lgb.LGBMRegressor(
                num_leaves=15, max_depth=4, learning_rate=0.02,
                feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=5,
                min_child_samples=50, lambda_l1=1.0, lambda_l2=5.0,
                n_estimators=500, early_stopping_rounds=50, verbosity=-1,
            )
            model.fit(X_tr[:split], y_tr[:split],
                      eval_set=[(X_tr[split:], y_tr[split:])],
                      callbacks=[lgb.log_evaluation(0)])

            preds = model.predict(X_ot)
            all_preds.extend(preds)
            all_actuals.extend(y_ot)

        preds = np.array(all_preds)
        actuals = np.array(all_actuals)
        if len(preds) < 50:
            return 0.0, 0
        ic = float(stats.spearmanr(preds, actuals)[0])
        return ic, len(preds)

    # Run 1: ALL bars (same as original)
    print("\n[1/3] Walk-forward on ALL bars (original)...")
    ic_all, n_all = run_wf(labeled, 'all', 'ALL')
    print(f"      IC = {ic_all:.4f}  (n={n_all})")

    # Run 2: WITHIN-DAY only (exclude cross-day from both train AND test)
    print("\n[2/3] Walk-forward EXCLUDING cross-day bars (train+test)...")
    ic_clean, n_clean = run_wf(within, 'within', 'CLEAN')
    print(f"      IC = {ic_clean:.4f}  (n={n_clean})")

    # Run 3: Train on within-day only, but test on ALL OOT bars
    # (to see if cross-day bars in test inflate/deflate IC)
    print("\n[3/3] Walk-forward: train within-day, test ALL bars...")
    dates_all = sorted(labeled['date'].unique())
    all_preds3, all_actuals3, all_is_cross3 = [], [], []

    for i in range(TRAIN_DAYS + PURGE_DAYS, len(dates_all)):
        oot_date = dates_all[i]
        train_end = i - PURGE_DAYS
        train_start = max(0, train_end - TRAIN_DAYS)
        train_dates = dates_all[train_start:train_end]

        # Train on within-day only
        train = within[within['date'].isin(train_dates)]
        oot = labeled[labeled['date'] == oot_date]

        if len(train) < 100 or len(oot) == 0:
            continue

        X_tr = train[feature_cols].fillna(0).values
        y_tr = train['fwd_ticks'].values
        X_ot = oot[feature_cols].fillna(0).values
        y_ot = oot['fwd_ticks'].values

        split = int(len(X_tr) * 0.8)
        if split < 20 or len(X_tr) - split < 5:
            continue

        model = lgb.LGBMRegressor(
            num_leaves=15, max_depth=4, learning_rate=0.02,
            feature_fraction=0.5, bagging_fraction=0.7, bagging_freq=5,
            min_child_samples=50, lambda_l1=1.0, lambda_l2=5.0,
            n_estimators=500, early_stopping_rounds=50, verbosity=-1,
        )
        model.fit(X_tr[:split], y_tr[:split],
                  eval_set=[(X_tr[split:], y_tr[split:])],
                  callbacks=[lgb.log_evaluation(0)])

        preds = model.predict(X_ot)
        all_preds3.extend(preds)
        all_actuals3.extend(y_ot)
        all_is_cross3.extend(oot['is_cross_day'].values)

    preds3 = np.array(all_preds3)
    actuals3 = np.array(all_actuals3)
    is_cross3 = np.array(all_is_cross3)

    if len(preds3) >= 50:
        ic_hybrid = float(stats.spearmanr(preds3, actuals3)[0])
        ic_hybrid_within = float(stats.spearmanr(preds3[~is_cross3], actuals3[~is_cross3])[0]) if (~is_cross3).sum() > 50 else float('nan')
        ic_hybrid_cross = float(stats.spearmanr(preds3[is_cross3], actuals3[is_cross3])[0]) if is_cross3.sum() > 50 else float('nan')
        print(f"      IC (all test):      {ic_hybrid:.4f}  (n={len(preds3)})")
        print(f"      IC (within-day):    {ic_hybrid_within:.4f}  (n={(~is_cross3).sum()})")
        print(f"      IC (cross-day):     {ic_hybrid_cross:.4f}  (n={is_cross3.sum()})")

    # ── Summary ──
    print(f"\n{'=' * 70}")
    print("SUMMARY")
    print(f"{'=' * 70}")
    print(f"Cross-day bars:         {pct_cross:.1f}% of all labeled bars")
    print(f"Cross-day |label| mean: {c_mean:.3f} ticks  (vs within-day: {w_mean:.3f})")
    print(f"Overnight gap inflates labels by: {c_mean/w_mean:.2f}x")
    print()
    print(f"IC with ALL bars:       {ic_all:.4f}")
    print(f"IC without cross-day:   {ic_clean:.4f}")
    delta = ic_clean - ic_all
    print(f"IC delta (clean-all):   {delta:+.4f}")
    if abs(ic_all) > 0:
        print(f"IC change:              {delta/abs(ic_all)*100:+.1f}%")
    print()
    if c_mean / w_mean > 1.5:
        print("WARNING: Overnight gaps make cross-day labels dramatically larger.")
        print("         These bars inject noise/bias — the model trains on labels")
        print("         that include gap moves it can never predict from intraday features.")
    elif c_mean / w_mean > 1.1:
        print("CAUTION: Cross-day labels are moderately larger than within-day.")
    else:
        print("OK: Cross-day and within-day label magnitudes are similar.")


if __name__ == '__main__':
    main()
