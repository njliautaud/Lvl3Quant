"""
2H LGBM Overfitting Diagnosis
==============================
Tests 5 hypotheses for why Sharpe=16 is impossibly good.

Data: close prices are in TICKS (1 unit = 1 tick = 0.25 ES points)
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
import sys
from collections import defaultdict

sys.stdout.reconfigure(line_buffering=True)

DATA_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1/'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/lh_2h_overfitting_diagnosis/'

TRAIN_DAYS = 60
PURGE_DAYS = 5
HORIZON_MINUTES = 120
TOTAL_RT_COST_TICKS = 1.376

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mse',
    'boosting_type': 'gbdt',
    'num_leaves': 31,
    'learning_rate': 0.05,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}
LGB_NUM_ROUNDS = 500
LGB_EARLY_STOPPING = 50


def load_all_data():
    """Load all minute bar data."""
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    all_dfs = {}
    for f in files:
        date_str = f.replace('.parquet', '')
        df = pd.read_parquet(os.path.join(DATA_DIR, f))
        df['ts_minute'] = pd.to_datetime(df['ts_minute'])
        df = df.reset_index(drop=True)
        all_dfs[date_str] = df
    return all_dfs


def compute_features_and_labels(all_dfs):
    """Compute features for all prediction points."""
    all_records = []

    for date_str, df in sorted(all_dfs.items()):
        if len(df) < 180:
            continue

        max_pred_idx = len(df) - HORIZON_MINUTES

        for pred_idx in range(60, max_pred_idx + 1, 60):
            if pred_idx + HORIZON_MINUTES > len(df):
                continue

            window = df.iloc[pred_idx-60:pred_idx]
            close_now = df.iloc[pred_idx]['close']
            close_future = df.iloc[pred_idx + HORIZON_MINUTES - 1]['close']
            label = close_future - close_now

            feat = {}

            # Order flow
            ofi = window['ofi_1min']
            feat['ofi_sum'] = ofi.sum()
            feat['ofi_mean'] = ofi.mean()
            feat['ofi_std'] = ofi.std()
            feat['ofi_last10'] = ofi.iloc[-10:].sum()
            feat['ofi_first10'] = ofi.iloc[:10].sum()
            feat['ofi_trend'] = feat['ofi_last10'] - feat['ofi_first10']

            # Volume
            feat['signed_vol_sum'] = window['signed_volume'].sum()
            feat['signed_vol_mean'] = window['signed_volume'].mean()
            feat['volume_sum'] = window['volume'].sum()
            feat['volume_mean'] = window['volume'].mean()
            feat['volume_std'] = window['volume'].std()
            v_first = window['volume'].iloc[:30].sum()
            feat['vol_ratio'] = window['volume'].iloc[-30:].sum() / max(v_first, 1)

            # Price
            feat['close_change'] = window['close'].iloc[-1] - window['close'].iloc[0]
            feat['close_change_last30'] = window['close'].iloc[-1] - window['close'].iloc[-30]
            feat['high_low_range'] = window['high'].max() - window['low'].min()
            feat['close_vs_high'] = window['close'].iloc[-1] - window['high'].max()
            feat['close_vs_low'] = window['close'].iloc[-1] - window['low'].min()
            bar_range = window['high'] - window['low']
            feat['bar_range_mean'] = bar_range.mean()
            feat['bar_range_std'] = bar_range.std()

            # Microprice
            feat['microprice_last'] = window['microprice_close'].iloc[-1]
            feat['microprice_vs_close'] = window['microprice_close'].iloc[-1] - window['close'].iloc[-1]
            feat['microprice_trend'] = window['microprice_close'].iloc[-1] - window['microprice_close'].iloc[0]

            # Spread
            feat['spread_mean'] = window['spread_mean'].mean()
            feat['spread_last'] = window['spread_mean'].iloc[-1]
            feat['spread_max'] = window['spread_mean'].max()

            # VWAP
            feat['vwap_vs_close'] = window['vwap'].iloc[-1] - window['close'].iloc[-1]
            feat['vwap_trend'] = window['vwap'].iloc[-1] - window['vwap'].iloc[0]

            # Trade count
            feat['trade_count_sum'] = window['trade_count'].sum()
            feat['trade_count_mean'] = window['trade_count'].mean()
            tc_first = window['trade_count'].iloc[:30].sum()
            feat['trade_count_ratio'] = window['trade_count'].iloc[-30:].sum() / max(tc_first, 1)

            # Vol regime
            vol_regime_map = {'low': 0, 'medium': 1, 'high': 2}
            feat['vol_regime'] = vol_regime_map.get(window['vol_regime'].iloc[-1], 1)

            # Time of day
            feat['hour_of_day'] = pred_idx / 60.0

            # Metadata
            feat['date'] = date_str
            feat['pred_minute_idx'] = pred_idx
            feat['label'] = label
            feat['close_at_pred'] = close_now
            feat['close_at_exit'] = close_future

            # Extra for diagnosis: return over last 1h (BEFORE pred point)
            feat['return_last_1h'] = window['close'].iloc[-1] - window['close'].iloc[0]
            # This is same as close_change but we keep it explicit

            all_records.append(feat)

    return pd.DataFrame(all_records)


def compute_sharpe(trades_pnl_series, dates_series):
    """Compute annualized Sharpe from per-trade PnL."""
    df = pd.DataFrame({'pnl': trades_pnl_series, 'date': dates_series})
    daily = df.groupby('date')['pnl'].sum()
    if daily.std() == 0:
        return 0.0
    return float(daily.mean() / daily.std() * np.sqrt(252))


def run_walkforward_with_features(all_features_df, feature_cols, label_col='label'):
    """Generic walkforward that returns trades."""
    meta_cols = ['date', 'pred_minute_idx', 'label', 'close_at_pred', 'close_at_exit', 'return_last_1h']

    dates = sorted(all_features_df['date'].unique())
    start_oot_idx = TRAIN_DAYS + PURGE_DAYS

    all_trades = []

    for oot_idx in range(start_oot_idx, len(dates)):
        oot_date = dates[oot_idx]
        train_end_idx = oot_idx - PURGE_DAYS
        train_start_idx = train_end_idx - TRAIN_DAYS

        if train_start_idx < 0:
            continue

        train_dates = set(dates[train_start_idx:train_end_idx])

        train_mask = all_features_df['date'].isin(train_dates)
        oot_mask = all_features_df['date'] == oot_date

        train_data = all_features_df[train_mask]
        oot_data = all_features_df[oot_mask]

        if len(train_data) < 50 or len(oot_data) == 0:
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data[label_col].values

        split_idx = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            LGB_PARAMS, dtrain,
            num_boost_round=LGB_NUM_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False)],
        )

        X_oot = oot_data[feature_cols].values
        preds = model.predict(X_oot)

        for pred, (_, row) in zip(preds, oot_data.iterrows()):
            gross_pnl = row['label'] if pred > 0 else -row['label']
            net_pnl = gross_pnl - TOTAL_RT_COST_TICKS
            all_trades.append({
                'date': row['date'],
                'prediction': float(pred),
                'actual_move': float(row['label']),
                'direction': 'LONG' if pred > 0 else 'SHORT',
                'gross_pnl_ticks': float(gross_pnl),
                'net_pnl_ticks': float(net_pnl),
            })

    return pd.DataFrame(all_trades)


# ============================================================
# TEST 1: AUTOCORRELATION / NAIVE MOMENTUM BASELINE
# ============================================================
def test_1_momentum_baseline(all_features_df):
    """Does last-1h return alone predict next-2h return?"""
    print("\n" + "="*70)
    print("TEST 1: MOMENTUM / AUTOCORRELATION BASELINE")
    print("="*70)

    df = all_features_df.copy()

    # Simple correlation
    corr = df['return_last_1h'].corr(df['label'])
    print(f"\n  Correlation(last_1h_return, next_2h_return) = {corr:.4f}")

    # IC per day
    daily_ic = df.groupby('date').apply(
        lambda x: x['return_last_1h'].corr(x['label']) if len(x) > 2 else np.nan
    ).dropna()
    print(f"  Mean daily IC (last_1h → next_2h): {daily_ic.mean():.4f} ± {daily_ic.std():.4f}")

    # Simple momentum strategy: if last 1h up, go long next 2h
    df['momentum_signal'] = df['return_last_1h']
    df['momentum_pnl'] = np.where(df['momentum_signal'] > 0, df['label'], -df['label']) - TOTAL_RT_COST_TICKS

    wr = (df['momentum_pnl'] > 0).mean()
    avg_pnl = df['momentum_pnl'].mean()

    daily_pnl = df.groupby('date')['momentum_pnl'].sum()
    sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252) if daily_pnl.std() > 0 else 0

    print(f"\n  NAIVE MOMENTUM STRATEGY (go with last-1h direction):")
    print(f"    Win Rate: {wr:.1%}")
    print(f"    Avg Net P&L: {avg_pnl:.3f} ticks")
    print(f"    Annualized Sharpe: {sharpe:.2f}")

    # Now check: what's the autocorrelation structure of 2h returns?
    # Get all returns in sequence
    dates = sorted(df['date'].unique())
    returns_seq = []
    for d in dates:
        day_data = df[df['date'] == d].sort_values('pred_minute_idx')
        returns_seq.extend(day_data['label'].values)

    returns_seq = np.array(returns_seq)

    # Autocorrelation of labels
    if len(returns_seq) > 10:
        from numpy import correlate
        r = returns_seq - returns_seq.mean()
        norm = np.sum(r**2)
        acf_1 = np.sum(r[:-1] * r[1:]) / norm if norm > 0 else 0
        acf_2 = np.sum(r[:-2] * r[2:]) / norm if norm > 0 else 0
        print(f"\n  LABEL AUTOCORRELATION:")
        print(f"    ACF(lag=1 prediction): {acf_1:.4f}")
        print(f"    ACF(lag=2 predictions): {acf_2:.4f}")

    # KEY: Check if the label distribution is biased (mean != 0)
    print(f"\n  LABEL STATISTICS:")
    print(f"    Mean: {df['label'].mean():.3f} ticks")
    print(f"    Std: {df['label'].std():.3f} ticks")
    print(f"    Median: {df['label'].median():.3f} ticks")
    print(f"    % positive: {(df['label'] > 0).mean():.1%}")
    print(f"    Skewness: {df['label'].skew():.3f}")

    # CRITICAL: If mean label is large positive, the model just needs to predict "long"
    # and it wins by default because ES trended up during this period
    if abs(df['label'].mean()) > 5:
        print(f"\n  *** MAJOR FINDING: Labels have large mean bias ({df['label'].mean():.1f} ticks) ***")
        print(f"  *** A model that always goes LONG would capture this drift ***")

        # Always-long strategy
        always_long_pnl = df['label'] - TOTAL_RT_COST_TICKS
        always_long_wr = (always_long_pnl > 0).mean()
        daily_al = df.groupby('date').apply(lambda x: (x['label'] - TOTAL_RT_COST_TICKS).sum())
        sharpe_al = daily_al.mean() / daily_al.std() * np.sqrt(252) if daily_al.std() > 0 else 0

        print(f"\n  ALWAYS-LONG STRATEGY:")
        print(f"    Win Rate: {always_long_wr:.1%}")
        print(f"    Avg Net P&L: {always_long_pnl.mean():.3f} ticks")
        print(f"    Annualized Sharpe: {sharpe_al:.2f}")

    return {
        'correlation_1h_vs_2h': float(corr),
        'mean_daily_ic': float(daily_ic.mean()),
        'momentum_sharpe': float(sharpe),
        'momentum_wr': float(wr),
        'label_mean': float(df['label'].mean()),
        'label_std': float(df['label'].std()),
        'label_pct_positive': float((df['label'] > 0).mean()),
        'acf_lag1': float(acf_1) if 'acf_1' in dir() else None,
    }


# ============================================================
# TEST 2: REDUCED FEATURES (5 core only)
# ============================================================
def test_2_reduced_features(all_features_df):
    """Retrain with only 5 core features."""
    print("\n" + "="*70)
    print("TEST 2: REDUCED FEATURES (5 core only)")
    print("="*70)

    core_features = ['ofi_sum', 'microprice_last', 'signed_vol_sum', 'spread_mean', 'close_change']

    # Check which features exist
    available = [f for f in core_features if f in all_features_df.columns]
    print(f"  Using features: {available}")

    trades_df = run_walkforward_with_features(all_features_df, available)

    if len(trades_df) == 0:
        print("  No trades generated!")
        return {}

    wr = (trades_df['net_pnl_ticks'] > 0).mean()
    avg_pnl = trades_df['net_pnl_ticks'].mean()
    sharpe = compute_sharpe(trades_df['net_pnl_ticks'], trades_df['date'])

    print(f"  Trades: {len(trades_df)}")
    print(f"  Win Rate: {wr:.1%}")
    print(f"  Avg Net P&L: {avg_pnl:.3f} ticks")
    print(f"  Annualized Sharpe: {sharpe:.2f}")

    return {
        'n_trades': len(trades_df),
        'win_rate': float(wr),
        'avg_pnl': float(avg_pnl),
        'sharpe': float(sharpe),
    }


# ============================================================
# TEST 3: RANDOM FEATURES
# ============================================================
def test_3_random_features(all_features_df):
    """Replace all features with random noise. If still profitable, it's label autocorrelation."""
    print("\n" + "="*70)
    print("TEST 3: RANDOM FEATURES (noise)")
    print("="*70)

    np.random.seed(42)

    # Create random features with same shape
    n_features = 30
    random_cols = [f'random_{i}' for i in range(n_features)]

    df_random = all_features_df[['date', 'pred_minute_idx', 'label', 'close_at_pred',
                                  'close_at_exit', 'return_last_1h']].copy()
    for col in random_cols:
        df_random[col] = np.random.randn(len(df_random))

    trades_df = run_walkforward_with_features(df_random, random_cols)

    if len(trades_df) == 0:
        print("  No trades generated!")
        return {}

    wr = (trades_df['net_pnl_ticks'] > 0).mean()
    avg_pnl = trades_df['net_pnl_ticks'].mean()
    sharpe = compute_sharpe(trades_df['net_pnl_ticks'], trades_df['date'])

    print(f"  Trades: {len(trades_df)}")
    print(f"  Win Rate: {wr:.1%}")
    print(f"  Avg Net P&L: {avg_pnl:.3f} ticks")
    print(f"  Annualized Sharpe: {sharpe:.2f}")

    if sharpe > 3:
        print(f"\n  *** CRITICAL: Random features produce Sharpe {sharpe:.1f}! ***")
        print(f"  *** This PROVES the model is exploiting label structure, NOT features ***")

    return {
        'n_trades': len(trades_df),
        'win_rate': float(wr),
        'avg_pnl': float(avg_pnl),
        'sharpe': float(sharpe),
    }


# ============================================================
# TEST 4: NON-OVERLAPPING LABELS ONLY
# ============================================================
def test_4_non_overlapping(all_features_df):
    """Only use predictions at T, T+2h, T+4h (no overlap)."""
    print("\n" + "="*70)
    print("TEST 4: NON-OVERLAPPING LABELS (every 2h only)")
    print("="*70)

    # Current setup: predictions every 60 min, labels span 120 min
    # So pred at minute 60 covers [60, 180], pred at minute 120 covers [120, 240]
    # These OVERLAP by 60 minutes!
    # Non-overlapping: only keep pred at 60, 180, 300 (every 120 min)

    # Filter to non-overlapping predictions only
    df_nonoverlap = all_features_df[all_features_df['pred_minute_idx'] % 120 == 60].copy()
    print(f"  Original predictions: {len(all_features_df)}")
    print(f"  Non-overlapping predictions: {len(df_nonoverlap)}")
    print(f"  Predictions per day: ~{df_nonoverlap.groupby('date').size().mean():.1f}")

    meta_cols = ['date', 'pred_minute_idx', 'label', 'close_at_pred', 'close_at_exit', 'return_last_1h']
    feature_cols = [c for c in all_features_df.columns if c not in meta_cols]

    trades_df = run_walkforward_with_features(df_nonoverlap, feature_cols)

    if len(trades_df) == 0:
        print("  No trades generated!")
        return {}

    wr = (trades_df['net_pnl_ticks'] > 0).mean()
    avg_pnl = trades_df['net_pnl_ticks'].mean()
    sharpe = compute_sharpe(trades_df['net_pnl_ticks'], trades_df['date'])

    print(f"  Trades: {len(trades_df)}")
    print(f"  Win Rate: {wr:.1%}")
    print(f"  Avg Net P&L: {avg_pnl:.3f} ticks")
    print(f"  Annualized Sharpe: {sharpe:.2f}")

    return {
        'n_trades': len(trades_df),
        'win_rate': float(wr),
        'avg_pnl': float(avg_pnl),
        'sharpe': float(sharpe),
    }


# ============================================================
# TEST 5: ENTRY PRICE VERIFICATION
# ============================================================
def test_5_entry_price(all_features_df, all_dfs):
    """Verify that entry price is correct and test for look-ahead."""
    print("\n" + "="*70)
    print("TEST 5: ENTRY PRICE & LOOK-AHEAD VERIFICATION")
    print("="*70)

    # The script uses: close_now = df.iloc[pred_idx]['close']
    # This is the CLOSE of the minute bar AT pred_idx
    # And: close_future = df.iloc[pred_idx + HORIZON_MINUTES - 1]['close']
    # = close of bar 119 minutes later

    # Question: Is pred_idx the CURRENT bar (whose close we already know)?
    # The features use window = df.iloc[pred_idx-60:pred_idx]
    # This means features use bars [pred_idx-60, pred_idx-1] (60 bars, exclusive of pred_idx)
    # But close_now = df.iloc[pred_idx]['close'] ← this is the NEXT bar AFTER the feature window!

    # Wait - let's be precise about pandas iloc slicing:
    # df.iloc[pred_idx-60:pred_idx] = bars at indices pred_idx-60 through pred_idx-1 (exclusive end)
    # So the feature window ends at bar pred_idx-1
    # But entry is at bar pred_idx (the NEXT bar after features end)

    # This means: at the moment we make the prediction, bar pred_idx hasn't happened yet!
    # We're using data through bar pred_idx-1, then entering at the CLOSE of bar pred_idx.
    # That's look-ahead! We can't know bar pred_idx's close when we make the prediction.
    #
    # BUT WAIT: in practice, the close of bar pred_idx IS known at time pred_idx+1 minute.
    # If we make the prediction at the END of bar pred_idx-1, and we enter at the OPEN of
    # bar pred_idx... but the script uses CLOSE of pred_idx as entry.
    #
    # Actually, re-reading: the convention might be that we make the prediction AFTER
    # observing bar pred_idx (inclusive). Let me check: features use bars [pred_idx-60:pred_idx]
    # which is [pred_idx-60, ..., pred_idx-1]. The close_change feature is
    # window['close'].iloc[-1] - window['close'].iloc[0] = close at pred_idx-1 minus close at pred_idx-60.
    #
    # Entry: close at pred_idx. Exit: close at pred_idx + 119.
    #
    # THE BUG: We predict at end of bar pred_idx-1, but enter at close of bar pred_idx.
    # This means we're giving the model 1 FREE bar of "future information" in the entry price.
    # Actually no — this is the entry EXECUTION price. We predict after seeing pred_idx-1,
    # then we enter at the next available price which is... the open of bar pred_idx?
    # No, the script uses close of pred_idx.
    #
    # In reality this is probably fine — we're saying "after this minute completes, we decide,
    # and we measure P&L from current close to future close." The entry IS bar pred_idx's close.
    # But we don't KNOW bar pred_idx's close until it happens. So effectively:
    # - We should enter at OPEN of bar pred_idx (or close of bar pred_idx-1, same thing)
    # - Instead we enter at close of bar pred_idx (1 bar later)
    #
    # This gives us 1 minute of free drift! Let me quantify this.

    print("\n  ENTRY PRICE ANALYSIS:")
    print(f"  Feature window: bars [pred_idx-60 : pred_idx] (exclusive end = pred_idx-1)")
    print(f"  Entry price: df.iloc[pred_idx]['close'] = close of bar AFTER feature window")
    print(f"  Exit price: df.iloc[pred_idx + 119]['close']")
    print()

    # The CORRECT entry should be the last known price = close of pred_idx - 1
    # Let's compute what the label should be with correct entry
    diffs = []
    for date_str, df in sorted(all_dfs.items()):
        if len(df) < 180:
            continue
        max_pred_idx = len(df) - HORIZON_MINUTES
        for pred_idx in range(60, max_pred_idx + 1, 60):
            if pred_idx + HORIZON_MINUTES > len(df):
                continue
            # Current (WRONG) entry: close at pred_idx
            entry_wrong = df.iloc[pred_idx]['close']
            # Correct entry: close at pred_idx - 1 (last bar in feature window)
            entry_correct = df.iloc[pred_idx - 1]['close']
            exit_price = df.iloc[pred_idx + HORIZON_MINUTES - 1]['close']

            label_wrong = exit_price - entry_wrong
            label_correct = exit_price - entry_correct

            diffs.append({
                'entry_diff': entry_wrong - entry_correct,  # How much the 1 extra bar adds
                'label_wrong': label_wrong,
                'label_correct': label_correct,
                'label_diff': label_wrong - label_correct,  # Should be negative if market trends up
            })

    diffs_df = pd.DataFrame(diffs)

    print(f"  Entry price difference (close[pred_idx] - close[pred_idx-1]):")
    print(f"    Mean: {diffs_df['entry_diff'].mean():.4f} ticks")
    print(f"    This represents the 1-bar drift between decision and entry")
    print()
    print(f"  Label with CURRENT entry (close at pred_idx):")
    print(f"    Mean: {diffs_df['label_wrong'].mean():.3f} ticks")
    print(f"    Std: {diffs_df['label_wrong'].std():.3f} ticks")
    print()
    print(f"  Label with CORRECT entry (close at pred_idx - 1):")
    print(f"    Mean: {diffs_df['label_correct'].mean():.3f} ticks")
    print(f"    Std: {diffs_df['label_correct'].std():.3f} ticks")
    print()

    # BIGGER ISSUE: The entry at close[pred_idx] means the feature 'close_change'
    # (which = close[pred_idx-1] - close[pred_idx-60]) is correlated with the
    # SUBSEQUENT move because of momentum. But the label starts at close[pred_idx],
    # not close[pred_idx-1]. So there's actually 1 bar of "gap" where the model
    # can't benefit from momentum in the entry price.
    #
    # Actually the real issue might be different. Let me check the ACTUAL correlation
    # between features and labels in the OOT period.

    return {
        'mean_entry_diff': float(diffs_df['entry_diff'].mean()),
        'label_mean_current': float(diffs_df['label_wrong'].mean()),
        'label_mean_correct': float(diffs_df['label_correct'].mean()),
    }


# ============================================================
# TEST 6 (BONUS): THE TREND BIAS TEST
# ============================================================
def test_6_detrended(all_features_df):
    """Test with DEMEANED labels (remove market drift)."""
    print("\n" + "="*70)
    print("TEST 6 (BONUS): DEMEANED LABELS (remove market drift)")
    print("="*70)

    # The market trended UP during the OOT period (Oct 2025 - Apr 2026).
    # ES went from ~25000 to ~28000 ticks = +3000 ticks over ~130 days.
    # That's ~23 ticks/day. With 5 predictions/day over 2h windows,
    # the expected drift per 2h window = 23 * (2/6.5) = ~7 ticks.
    # That's not enough to explain 46 ticks/trade avg.
    # But let's check with rolling demean.

    df = all_features_df.copy()

    # Rolling 20-day label mean (expanding from past only)
    dates = sorted(df['date'].unique())
    date_means = {}
    for i, d in enumerate(dates):
        # Use last 20 days of labels as estimate of drift
        lookback_dates = dates[max(0, i-20):i]
        if len(lookback_dates) > 0:
            past = df[df['date'].isin(lookback_dates)]
            date_means[d] = past['label'].mean()
        else:
            date_means[d] = 0

    df['label_drift'] = df['date'].map(date_means)
    df['label_demeaned'] = df['label'] - df['label_drift']

    print(f"  Average drift estimate: {df['label_drift'].mean():.3f} ticks per prediction")
    print(f"  Demeaned label mean: {df['label_demeaned'].mean():.3f}")
    print(f"  Original label mean: {df['label'].mean():.3f}")

    # Now run walkforward with demeaned labels
    meta_cols = ['date', 'pred_minute_idx', 'label', 'close_at_pred', 'close_at_exit',
                 'return_last_1h', 'label_drift', 'label_demeaned']
    feature_cols = [c for c in all_features_df.columns if c not in meta_cols]

    # Train on demeaned labels, but evaluate P&L on ACTUAL price moves
    dates_list = sorted(df['date'].unique())
    start_oot_idx = TRAIN_DAYS + PURGE_DAYS

    all_trades = []
    for oot_idx in range(start_oot_idx, len(dates_list)):
        oot_date = dates_list[oot_idx]
        train_end_idx = oot_idx - PURGE_DAYS
        train_start_idx = train_end_idx - TRAIN_DAYS
        if train_start_idx < 0:
            continue

        train_dates = set(dates_list[train_start_idx:train_end_idx])
        train_mask = df['date'].isin(train_dates)
        oot_mask = df['date'] == oot_date

        train_data = df[train_mask]
        oot_data = df[oot_mask]

        if len(train_data) < 50 or len(oot_data) == 0:
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data['label_demeaned'].values  # Train on DEMEANED labels

        split_idx = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            LGB_PARAMS, dtrain,
            num_boost_round=LGB_NUM_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False)],
        )

        X_oot = oot_data[feature_cols].values
        preds = model.predict(X_oot)

        for pred, (_, row) in zip(preds, oot_data.iterrows()):
            # Evaluate on ACTUAL move (not demeaned)
            gross_pnl = row['label'] if pred > 0 else -row['label']
            net_pnl = gross_pnl - TOTAL_RT_COST_TICKS
            all_trades.append({
                'date': row['date'],
                'net_pnl_ticks': float(net_pnl),
            })

    trades_df = pd.DataFrame(all_trades)

    if len(trades_df) == 0:
        print("  No trades!")
        return {}

    wr = (trades_df['net_pnl_ticks'] > 0).mean()
    avg_pnl = trades_df['net_pnl_ticks'].mean()
    sharpe = compute_sharpe(trades_df['net_pnl_ticks'], trades_df['date'])

    print(f"\n  DEMEANED LABEL TRAINING (drift-corrected):")
    print(f"    Trades: {len(trades_df)}")
    print(f"    Win Rate: {wr:.1%}")
    print(f"    Avg Net P&L: {avg_pnl:.3f} ticks")
    print(f"    Annualized Sharpe: {sharpe:.2f}")

    return {
        'n_trades': len(trades_df),
        'win_rate': float(wr),
        'avg_pnl': float(avg_pnl),
        'sharpe': float(sharpe),
    }


# ============================================================
# TEST 7: LABEL OVERLAP QUANTIFICATION
# ============================================================
def test_7_label_overlap(all_features_df):
    """Quantify how much overlapping labels inflates apparent performance."""
    print("\n" + "="*70)
    print("TEST 7: LABEL OVERLAP INFLATION ANALYSIS")
    print("="*70)

    # With 60-min prediction spacing and 120-min labels:
    # pred at t=60 → label covers [60, 180]
    # pred at t=120 → label covers [120, 240]
    # These overlap by [120, 180] = 60 minutes!
    #
    # If market moves +40 ticks in [120, 180], BOTH labels benefit.
    # With 5 predictions per day, adjacent predictions share 50% of their move.
    # This means trade-level P&L is NOT independent.
    #
    # A SINGLE large intraday move inflates 2 consecutive predictions.
    # Daily Sharpe is computed on SUM of (typically) 5 correlated predictions.
    # The daily std is lower than it should be because moves are double-counted.

    df = all_features_df.copy()
    dates = sorted(df['date'].unique())

    # For OOT days, compute correlation between adjacent labels
    start_oot_idx = TRAIN_DAYS + PURGE_DAYS
    oot_dates = dates[start_oot_idx:]

    adj_corrs = []
    for d in oot_dates:
        day_data = df[df['date'] == d].sort_values('pred_minute_idx')
        if len(day_data) < 2:
            continue
        labels = day_data['label'].values
        for i in range(len(labels) - 1):
            adj_corrs.append((labels[i], labels[i+1]))

    adj_corrs = np.array(adj_corrs)
    if len(adj_corrs) > 5:
        corr = np.corrcoef(adj_corrs[:, 0], adj_corrs[:, 1])[0, 1]
        print(f"\n  Adjacent prediction label correlation: {corr:.4f}")
        print(f"  (Expected ~0.5 if labels share 50% of price path)")

        # Effective number of independent observations
        # With correlation rho between adjacent obs, effective N ≈ N / (1 + 2*rho)
        n_total = len(adj_corrs)
        n_effective = n_total / (1 + 2 * corr) if corr > 0 else n_total
        inflation_factor = np.sqrt(n_total / n_effective)

        print(f"\n  Total trade-pairs: {n_total}")
        print(f"  Effective independent observations: {n_effective:.0f}")
        print(f"  Sharpe INFLATION factor from overlap: {inflation_factor:.2f}x")
        print(f"  Corrected Sharpe: {16.19 / inflation_factor:.2f}")

    return {
        'adjacent_label_corr': float(corr) if len(adj_corrs) > 5 else None,
        'inflation_factor': float(inflation_factor) if len(adj_corrs) > 5 else None,
    }


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("2H LGBM OVERFITTING DIAGNOSIS")
    print("=" * 70)

    print("\nLoading data...")
    all_dfs = load_all_data()
    print(f"  Loaded {len(all_dfs)} days")

    print("\nComputing features...")
    all_features = compute_features_and_labels(all_dfs)
    n_dates = all_features['date'].nunique()
    print(f"  {len(all_features)} prediction points across {n_dates} days")
    print(f"  ~{len(all_features)/n_dates:.1f} predictions per day")

    results = {}

    # Test 1: Momentum baseline
    results['test_1_momentum'] = test_1_momentum_baseline(all_features)

    # Test 7: Label overlap (quick, doesn't need model)
    results['test_7_overlap'] = test_7_label_overlap(all_features)

    # Test 5: Entry price
    results['test_5_entry'] = test_5_entry_price(all_features, all_dfs)

    # Test 3: Random features (fastest model test)
    results['test_3_random'] = test_3_random_features(all_features)

    # Test 2: Reduced features
    results['test_2_reduced'] = test_2_reduced_features(all_features)

    # Test 4: Non-overlapping
    results['test_4_nonoverlap'] = test_4_non_overlapping(all_features)

    # Test 6: Demeaned
    results['test_6_demeaned'] = test_6_detrended(all_features)

    # Save
    with open(os.path.join(OUTPUT_DIR, 'diagnosis_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Final summary
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)
    print(f"\n  Original claimed Sharpe: 16.19")
    print(f"\n  Test 1 - Momentum baseline Sharpe: {results['test_1_momentum'].get('momentum_sharpe', 'N/A'):.2f}")
    print(f"  Test 1 - Label mean bias: {results['test_1_momentum'].get('label_mean', 'N/A'):.1f} ticks")
    print(f"  Test 2 - 5 features Sharpe: {results['test_2_reduced'].get('sharpe', 'N/A'):.2f}")
    print(f"  Test 3 - Random features Sharpe: {results['test_3_random'].get('sharpe', 'N/A'):.2f}")
    print(f"  Test 4 - Non-overlapping Sharpe: {results['test_4_nonoverlap'].get('sharpe', 'N/A'):.2f}")
    print(f"  Test 6 - Demeaned Sharpe: {results['test_6_demeaned'].get('sharpe', 'N/A'):.2f}")
    if results['test_7_overlap'].get('inflation_factor'):
        print(f"  Test 7 - Overlap inflation factor: {results['test_7_overlap']['inflation_factor']:.2f}x")
        print(f"  Test 7 - Overlap-corrected Sharpe: {16.19 / results['test_7_overlap']['inflation_factor']:.2f}")

    print(f"\n  Results saved to: {OUTPUT_DIR}diagnosis_results.json")
    print("=" * 70)
