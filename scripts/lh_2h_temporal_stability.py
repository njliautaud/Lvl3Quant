#!/usr/bin/env python3
"""
2h LGBM Model — Temporal Stability Analysis
============================================

HC #662 R4: Research continues. Key question: Is the 2h model's signal
STABLE over time or degrading? This is critical for deployment confidence.

Approach:
- Run 132-fold walk-forward (same as permutation test script)
- Track per-fold IC, WR, PnL
- Compute rolling 20-fold windows to see trends
- Test for statistically significant IC decay (OLS slope)
- Break into quartiles (earliest → latest) for comparison
"""

import numpy as np
import pandas as pd
import sys
import os
import json
from pathlib import Path
from datetime import datetime
from scipy import stats

sys.path.insert(0, '/home/jupiter/Lvl3Quant/live_trading_linux')

from lh_2h_paper_engine import (
    compute_enhanced_hourly, add_rolling_features, add_regime_context,
    get_feature_cols, LGBM_PARAMS, SIGNAL_HOURS_UTC, HORIZON_BARS,
    COST_RT_TICKS, ES_TICK_VALUE
)

TRAIN_DAYS = 60
PURGE_DAYS = 5
MINUTE_BAR_DIR = Path("/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1")
OUT_DIR = Path("/home/jupiter/Lvl3Quant/output/lh_2h_temporal_stability")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def load_all_minute_bars():
    files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["date"] = f.stem
            frames.append(df)
        except Exception as e:
            print(f"  Skip {f.stem}: {e}")
    combined = pd.concat(frames, ignore_index=True)
    combined["ts_minute"] = pd.to_datetime(combined["ts_minute"], utc=True)
    combined = combined.sort_values("ts_minute").reset_index(drop=True)
    print(f"Loaded {len(combined):,} minute bars across {len(frames)} days")
    return combined


def run_stability_analysis():
    import lightgbm as lgb

    print("=" * 70)
    print("2h LGBM — TEMPORAL STABILITY ANALYSIS")
    print("=" * 70)

    minute_df = load_all_minute_bars()
    hourly = compute_enhanced_hourly(minute_df)
    hourly = add_rolling_features(hourly)
    hourly = add_regime_context(hourly)

    hourly = hourly.sort_values("ts").reset_index(drop=True)
    hourly["fwd_ticks"] = hourly["close"].shift(-HORIZON_BARS) - hourly["close"]

    # Null overnight gaps
    for i in range(len(hourly) - HORIZON_BARS):
        ts_now = hourly["ts"].iloc[i]
        ts_fwd = hourly["ts"].iloc[i + HORIZON_BARS]
        diff_s = (ts_fwd - ts_now).total_seconds()
        if diff_s > 8 * 3600:
            hourly.loc[hourly.index[i], "fwd_ticks"] = np.nan

    hourly_clean = hourly[~hourly["hour"].isin([19, 20])].copy()
    dates = sorted(hourly_clean["date"].unique())
    feature_cols = get_feature_cols(hourly_clean)

    n_oot = len(dates) - TRAIN_DAYS - PURGE_DAYS
    print(f"Total dates: {len(dates)}, OOT folds: {n_oot}")
    print(f"Features: {len(feature_cols)}")
    print()

    # ── Walk-forward collecting per-fold metrics ──
    fold_results = []

    for i in range(TRAIN_DAYS + PURGE_DAYS, len(dates)):
        oot_date = dates[i]
        train_end_idx = i - PURGE_DAYS
        train_start_idx = max(0, train_end_idx - TRAIN_DAYS)
        train_dates = dates[train_start_idx:train_end_idx]

        train = hourly_clean[hourly_clean["date"].isin(train_dates)].dropna(subset=["fwd_ticks"])
        oot = hourly_clean[hourly_clean["date"] == oot_date]
        oot_tradeable = oot[oot["hour"].isin(SIGNAL_HOURS_UTC)]

        if len(train) < 100 or len(oot_tradeable) == 0:
            continue

        X_train = train[feature_cols].fillna(0).values.astype(np.float32)
        y_train = train["fwd_ticks"].values.astype(np.float32)
        X_oot = oot_tradeable[feature_cols].fillna(0).values.astype(np.float32)
        actual_fwd = oot_tradeable["fwd_ticks"].values

        split = int(len(X_train) * 0.8)
        try:
            model = lgb.LGBMRegressor(**LGBM_PARAMS, early_stopping_rounds=50, seed=42, verbose=-1)
            model.fit(X_train[:split], y_train[:split],
                      eval_set=[(X_train[split:], y_train[split:])])
        except Exception:
            continue

        preds = model.predict(X_oot)

        # Per-fold metrics
        valid_mask = ~np.isnan(actual_fwd)
        if valid_mask.sum() < 2:
            continue

        p = preds[valid_mask]
        a = actual_fwd[valid_mask]

        # Spearman IC
        ic, ic_pval = stats.spearmanr(p, a)

        # Trade-level PnL
        directions = np.sign(p)
        raw_ticks = a / 0.25
        gross = raw_ticks * directions
        net = gross - COST_RT_TICKS

        n_trades = len(p)
        wins = (net > 0).sum()
        total_pnl = net.sum()
        avg_pnl = net.mean()

        # Prediction magnitude (avg |pred|)
        avg_abs_pred = np.abs(p).mean()

        # Feature importance (top 5 mean)
        if hasattr(model, 'feature_importances_'):
            fi = model.feature_importances_
            top5_fi = np.sort(fi)[-5:].mean()
        else:
            top5_fi = 0

        fold_results.append({
            'date': oot_date,
            'fold_idx': len(fold_results),
            'ic': float(ic),
            'ic_pval': float(ic_pval),
            'n_trades': int(n_trades),
            'wr': float(wins / n_trades) if n_trades > 0 else 0,
            'total_pnl_ticks': float(total_pnl),
            'avg_pnl_ticks': float(avg_pnl),
            'avg_abs_pred': float(avg_abs_pred),
            'top5_feature_importance': float(top5_fi),
        })

        if len(fold_results) % 30 == 0:
            print(f"  Fold {len(fold_results)}: {oot_date}, IC={ic:.3f}, WR={wins/n_trades:.0%}, PnL={total_pnl:+.1f}t")

    print(f"\nCompleted {len(fold_results)} folds")

    # ── Analysis ──
    df = pd.DataFrame(fold_results)

    # 1. Overall trend (OLS regression of IC on fold index)
    slope_ic, intercept_ic, r_ic, p_ic, se_ic = stats.linregress(df['fold_idx'], df['ic'])
    slope_wr, intercept_wr, r_wr, p_wr, se_wr = stats.linregress(df['fold_idx'], df['wr'])
    slope_pnl, intercept_pnl, r_pnl, p_pnl, se_pnl = stats.linregress(df['fold_idx'], df['avg_pnl_ticks'])

    print(f"\n{'='*70}")
    print("TREND ANALYSIS (OLS slope per fold)")
    print(f"{'='*70}")
    print(f"IC slope:  {slope_ic:.5f}/fold (p={p_ic:.4f}) — {'SIGNIFICANT' if p_ic < 0.05 else 'not significant'}")
    print(f"WR slope:  {slope_wr:.5f}/fold (p={p_wr:.4f}) — {'SIGNIFICANT' if p_wr < 0.05 else 'not significant'}")
    print(f"PnL slope: {slope_pnl:.4f}t/fold (p={p_pnl:.4f}) — {'SIGNIFICANT' if p_pnl < 0.05 else 'not significant'}")

    # 2. Quartile comparison
    q_size = len(df) // 4
    quartiles = {}
    for qi, label in enumerate(['Q1 (oldest)', 'Q2', 'Q3', 'Q4 (newest)']):
        start = qi * q_size
        end = start + q_size if qi < 3 else len(df)
        qdf = df.iloc[start:end]
        quartiles[label] = {
            'dates': f"{qdf['date'].iloc[0]} to {qdf['date'].iloc[-1]}",
            'n_folds': len(qdf),
            'mean_ic': float(qdf['ic'].mean()),
            'median_ic': float(qdf['ic'].median()),
            'mean_wr': float(qdf['wr'].mean()),
            'mean_pnl_ticks': float(qdf['avg_pnl_ticks'].mean()),
            'total_pnl_ticks': float(qdf['total_pnl_ticks'].sum()),
            'pct_positive_ic': float((qdf['ic'] > 0).mean()),
        }

    print(f"\n{'='*70}")
    print("QUARTILE COMPARISON")
    print(f"{'='*70}")
    for label, q in quartiles.items():
        print(f"\n{label} ({q['dates']}):")
        print(f"  IC:  mean={q['mean_ic']:.3f}, median={q['median_ic']:.3f}, pct>0={q['pct_positive_ic']:.0%}")
        print(f"  WR:  {q['mean_wr']:.1%}")
        print(f"  PnL: avg={q['mean_pnl_ticks']:.1f}t, total={q['total_pnl_ticks']:.0f}t")

    # 3. Rolling 20-fold metrics
    roll_window = 20
    rolling = []
    for start in range(0, len(df) - roll_window + 1, 5):
        window = df.iloc[start:start + roll_window]
        rolling.append({
            'center_date': window['date'].iloc[roll_window // 2],
            'mean_ic': float(window['ic'].mean()),
            'mean_wr': float(window['wr'].mean()),
            'mean_pnl': float(window['avg_pnl_ticks'].mean()),
            'total_pnl': float(window['total_pnl_ticks'].sum()),
        })

    print(f"\n{'='*70}")
    print(f"ROLLING {roll_window}-FOLD WINDOWS (step 5)")
    print(f"{'='*70}")
    for r in rolling:
        bar = '█' * max(0, int(r['mean_ic'] * 10))
        print(f"  {r['center_date']}: IC={r['mean_ic']:.3f} {bar}  WR={r['mean_wr']:.0%}  PnL={r['mean_pnl']:+.1f}t")

    # 4. Prediction magnitude trend (model overconfidence?)
    slope_pred, _, _, p_pred, _ = stats.linregress(df['fold_idx'], df['avg_abs_pred'])
    print(f"\nPrediction magnitude slope: {slope_pred:.5f}/fold (p={p_pred:.4f})")

    # 5. IC autocorrelation (does a bad fold predict next bad fold?)
    ic_autocorr = df['ic'].autocorr(lag=1)
    print(f"IC lag-1 autocorrelation: {ic_autocorr:.3f}")

    # 6. Worst streak analysis
    bad_streaks = []
    current_streak = 0
    for _, row in df.iterrows():
        if row['ic'] < 0:
            current_streak += 1
        else:
            if current_streak > 0:
                bad_streaks.append(current_streak)
            current_streak = 0
    if current_streak > 0:
        bad_streaks.append(current_streak)

    max_bad_streak = max(bad_streaks) if bad_streaks else 0
    print(f"Worst streak of negative-IC folds: {max_bad_streak} consecutive days")

    # 7. Structural break test (Chow test equivalent — compare first vs second half)
    half = len(df) // 2
    first_half = df.iloc[:half]
    second_half = df.iloc[half:]
    t_stat, t_pval = stats.ttest_ind(first_half['ic'], second_half['ic'])
    print(f"\nFirst-half vs second-half IC t-test: t={t_stat:.3f}, p={t_pval:.4f}")
    print(f"  First half mean IC:  {first_half['ic'].mean():.3f}")
    print(f"  Second half mean IC: {second_half['ic'].mean():.3f}")

    # ── Verdict ──
    print(f"\n{'='*70}")
    print("VERDICT")
    print(f"{'='*70}")

    ic_stable = p_ic > 0.05
    wr_stable = p_wr > 0.05
    halves_same = t_pval > 0.05
    newest_q = list(quartiles.values())[-1]
    newest_positive = newest_q['mean_ic'] > 0.1

    if ic_stable and newest_positive and halves_same:
        verdict = "STABLE — No significant IC decay. Model signal persists in recent data."
    elif not ic_stable and slope_ic < 0:
        verdict = "DEGRADING — Statistically significant IC decay detected. Signal may be decaying."
    elif not newest_positive:
        verdict = "RECENT WEAKNESS — Latest quartile IC below 0.1. Monitor closely."
    else:
        verdict = "MIXED — Some instability detected but not conclusive."

    print(verdict)

    # ── Save ──
    results = {
        'n_folds': len(df),
        'date_range': f"{df['date'].iloc[0]} to {df['date'].iloc[-1]}",
        'overall': {
            'mean_ic': float(df['ic'].mean()),
            'median_ic': float(df['ic'].median()),
            'std_ic': float(df['ic'].std()),
            'mean_wr': float(df['wr'].mean()),
            'mean_pnl_ticks': float(df['avg_pnl_ticks'].mean()),
        },
        'trend': {
            'ic_slope_per_fold': float(slope_ic),
            'ic_slope_pval': float(p_ic),
            'wr_slope_per_fold': float(slope_wr),
            'wr_slope_pval': float(p_wr),
            'pnl_slope_per_fold': float(slope_pnl),
            'pnl_slope_pval': float(p_pnl),
            'pred_magnitude_slope': float(slope_pred),
            'pred_magnitude_pval': float(p_pred),
        },
        'structural_break': {
            'first_half_mean_ic': float(first_half['ic'].mean()),
            'second_half_mean_ic': float(second_half['ic'].mean()),
            't_stat': float(t_stat),
            'p_val': float(t_pval),
        },
        'quartiles': quartiles,
        'rolling_20fold': rolling,
        'diagnostics': {
            'ic_autocorr_lag1': float(ic_autocorr),
            'max_negative_ic_streak': int(max_bad_streak),
            'pct_positive_ic_folds': float((df['ic'] > 0).mean()),
        },
        'per_fold': fold_results,
        'verdict': verdict,
        'timestamp': datetime.now().isoformat(),
    }

    out_path = OUT_DIR / "stability_results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")

    return results


if __name__ == "__main__":
    run_stability_analysis()
