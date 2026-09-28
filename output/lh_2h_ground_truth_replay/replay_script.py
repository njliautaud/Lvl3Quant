"""
GROUND-UP 2H ES LGBM LIVE REPLAY (v2 - Optimized)
====================================================
Complete from-scratch validation. Pre-computes all features ONCE,
then slices for walk-forward. No redundant computation.

Author: Claude (validation audit)
Date: 2026-07-12
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
import sys

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)

# ============================================================
# CONSTANTS
# ============================================================
DATA_DIR = '/home/jupiter/Lvl3Quant/data/processed/mbo_minute_bars_v1/'
OUTPUT_DIR = '/home/jupiter/Lvl3Quant/output/lh_2h_ground_truth_replay/'

TRAIN_DAYS = 60
PURGE_DAYS = 5
HORIZON_MINUTES = 120  # 2 hours

# Cost assumptions (ES futures, AMP/Rithmic)
SPREAD_TICKS = 1.0
COMMISSION_RT_TICKS = 0.376
TOTAL_RT_COST_TICKS = SPREAD_TICKS + COMMISSION_RT_TICKS  # 1.376 ticks
TICK_VALUE_USD = 12.50

# LightGBM parameters
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


# ============================================================
# PRE-COMPUTE ALL FEATURES (once)
# ============================================================
def precompute_all_features():
    """Load all data and compute features for every valid prediction point."""
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])

    all_records = []

    for f in files:
        date_str = f.replace('.parquet', '')
        df = pd.read_parquet(os.path.join(DATA_DIR, f))
        df['ts_minute'] = pd.to_datetime(df['ts_minute'])

        if len(df) < 180:  # Need 60 lookback + 120 forward minimum
            continue

        df = df.reset_index(drop=True)
        max_pred_idx = len(df) - HORIZON_MINUTES

        # Prediction every 60 minutes, starting at minute 60
        for pred_idx in range(60, max_pred_idx + 1, 60):
            if pred_idx + HORIZON_MINUTES > len(df):
                continue

            window = df.iloc[pred_idx-60:pred_idx]
            close_now = df.iloc[pred_idx]['close']
            close_future = df.iloc[pred_idx + HORIZON_MINUTES - 1]['close']
            label = close_future - close_now

            # Compute features
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

            # Time of day (hour since market open, fractional)
            feat['hour_of_day'] = pred_idx / 60.0

            # Metadata
            feat['date'] = date_str
            feat['pred_minute_idx'] = pred_idx
            feat['pred_time_utc'] = str(df.iloc[pred_idx]['ts_minute'])
            feat['label'] = label
            feat['close_at_pred'] = close_now
            feat['close_at_exit'] = close_future

            all_records.append(feat)

    return pd.DataFrame(all_records)


# ============================================================
# WALK-FORWARD
# ============================================================
def run_walkforward(all_features_df):
    """Sliding walk-forward on pre-computed features."""

    meta_cols = ['date', 'pred_minute_idx', 'pred_time_utc', 'label',
                 'close_at_pred', 'close_at_exit']
    feature_cols = [c for c in all_features_df.columns if c not in meta_cols]

    dates = sorted(all_features_df['date'].unique())

    start_oot_idx = TRAIN_DAYS + PURGE_DAYS

    print(f"Total trading days with valid predictions: {len(dates)}")
    print(f"First OOT day index: {start_oot_idx} = {dates[start_oot_idx]}")
    print(f"Last OOT day: {dates[-1]}")
    print(f"OOT days: {len(dates) - start_oot_idx}")
    print(f"Feature columns ({len(feature_cols)}): {feature_cols[:8]}...")
    print()

    all_trades = []

    for oot_idx in range(start_oot_idx, len(dates)):
        oot_date = dates[oot_idx]

        # Training window
        train_end_idx = oot_idx - PURGE_DAYS
        train_start_idx = train_end_idx - TRAIN_DAYS

        if train_start_idx < 0:
            continue

        train_dates = set(dates[train_start_idx:train_end_idx])
        last_train_date = dates[train_end_idx - 1]

        # Get data slices
        train_mask = all_features_df['date'].isin(train_dates)
        oot_mask = all_features_df['date'] == oot_date

        train_data = all_features_df[train_mask]
        oot_data = all_features_df[oot_mask]

        if len(train_data) < 50 or len(oot_data) == 0:
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data['label'].values

        # 80/20 chronological split for early stopping
        split_idx = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split_idx], X_train[split_idx:]
        y_tr, y_val = y_train[:split_idx], y_train[split_idx:]

        # Train
        dtrain = lgb.Dataset(X_tr, label=y_tr)
        dval = lgb.Dataset(X_val, label=y_val, reference=dtrain)

        model = lgb.train(
            LGB_PARAMS,
            dtrain,
            num_boost_round=LGB_NUM_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False)],
        )

        # Predict
        X_oot = oot_data[feature_cols].values
        preds = model.predict(X_oot)

        for pred, (_, row) in zip(preds, oot_data.iterrows()):
            direction = 'LONG' if pred > 0 else 'SHORT'
            gross_pnl = row['label'] if pred > 0 else -row['label']
            net_pnl = gross_pnl - TOTAL_RT_COST_TICKS

            all_trades.append({
                'date': row['date'],
                'pred_time_utc': row['pred_time_utc'],
                'pred_minute_idx': int(row['pred_minute_idx']),
                'prediction': float(pred),
                'confidence': abs(float(pred)),
                'actual_move': float(row['label']),
                'close_at_pred': float(row['close_at_pred']),
                'close_at_exit': float(row['close_at_exit']),
                'direction': direction,
                'gross_pnl_ticks': float(gross_pnl),
                'net_pnl_ticks': float(net_pnl),
                'net_pnl_usd': float(net_pnl * TICK_VALUE_USD),
                'last_train_date': last_train_date,
            })

        if oot_idx % 20 == 0:
            n = len(all_trades)
            cum_pnl = sum(t['net_pnl_ticks'] for t in all_trades)
            print(f"  Day {oot_idx-start_oot_idx+1}/{len(dates)-start_oot_idx}: "
                  f"{oot_date} | trades={n} | cum_pnl={cum_pnl:.1f} ticks")

    return pd.DataFrame(all_trades), feature_cols


# ============================================================
# ANALYSIS
# ============================================================
def analyze_results(trades_df, confidence_threshold=0.0):
    """Compute comprehensive metrics."""
    df = trades_df[trades_df['confidence'] > confidence_threshold].copy()

    if len(df) == 0:
        return {'n_trades': 0, 'threshold': confidence_threshold}

    r = {
        'threshold': confidence_threshold,
        'n_trades': len(df),
        'n_long': int((df['direction'] == 'LONG').sum()),
        'n_short': int((df['direction'] == 'SHORT').sum()),
        'win_rate': float((df['net_pnl_ticks'] > 0).mean()),
        'avg_net_pnl_ticks': float(df['net_pnl_ticks'].mean()),
        'avg_net_pnl_usd': float(df['net_pnl_usd'].mean()),
        'total_net_pnl_ticks': float(df['net_pnl_ticks'].sum()),
        'total_net_pnl_usd': float(df['net_pnl_usd'].sum()),
        'avg_gross_pnl_ticks': float(df['gross_pnl_ticks'].mean()),
        'median_net_pnl_ticks': float(df['net_pnl_ticks'].median()),
        'std_net_pnl_ticks': float(df['net_pnl_ticks'].std()),
    }

    # Daily Sharpe
    daily_pnl = df.groupby('date')['net_pnl_ticks'].sum()
    if daily_pnl.std() > 0:
        r['daily_sharpe'] = float(daily_pnl.mean() / daily_pnl.std())
        r['annualized_sharpe'] = float(r['daily_sharpe'] * np.sqrt(252))
    else:
        r['daily_sharpe'] = 0.0
        r['annualized_sharpe'] = 0.0

    # Sortino
    downside = daily_pnl[daily_pnl < 0]
    if len(downside) > 0 and downside.std() > 0:
        r['daily_sortino'] = float(daily_pnl.mean() / downside.std())
        r['annualized_sortino'] = float(r['daily_sortino'] * np.sqrt(252))
    else:
        r['daily_sortino'] = 0.0
        r['annualized_sortino'] = 0.0

    # Profit Factor
    wins = df[df['net_pnl_ticks'] > 0]['net_pnl_ticks'].sum()
    losses = abs(df[df['net_pnl_ticks'] < 0]['net_pnl_ticks'].sum())
    r['profit_factor'] = float(wins / max(losses, 0.001))

    # Max Drawdown
    cum = df['net_pnl_ticks'].cumsum()
    r['max_drawdown_ticks'] = float((cum - cum.cummax()).min())
    r['max_drawdown_usd'] = float(r['max_drawdown_ticks'] * TICK_VALUE_USD)

    # Max consecutive losses
    is_loss = (df['net_pnl_ticks'] < 0).astype(int).values
    max_consec = current = 0
    for v in is_loss:
        if v:
            current += 1
            max_consec = max(max_consec, current)
        else:
            current = 0
    r['max_consecutive_losses'] = max_consec

    # Long vs Short
    longs = df[df['direction'] == 'LONG']
    shorts = df[df['direction'] == 'SHORT']
    r['long_win_rate'] = float((longs['net_pnl_ticks'] > 0).mean()) if len(longs) > 0 else 0
    r['short_win_rate'] = float((shorts['net_pnl_ticks'] > 0).mean()) if len(shorts) > 0 else 0
    r['long_avg_pnl'] = float(longs['net_pnl_ticks'].mean()) if len(longs) > 0 else 0
    r['short_avg_pnl'] = float(shorts['net_pnl_ticks'].mean()) if len(shorts) > 0 else 0

    # Per prediction-hour breakdown
    df_copy = df.copy()
    df_copy['pred_hour'] = df_copy['pred_minute_idx'] // 60
    hour_stats = {}
    for h, grp in df_copy.groupby('pred_hour'):
        hour_stats[str(int(h))] = {
            'n_trades': len(grp),
            'avg_pnl': float(grp['net_pnl_ticks'].mean()),
            'win_rate': float((grp['net_pnl_ticks'] > 0).mean()),
            'total_pnl': float(grp['net_pnl_ticks'].sum()),
        }
    r['per_hour'] = hour_stats

    # Monthly breakdown
    df_copy['month'] = df_copy['date'].str[:6]
    monthly = {}
    for m, grp in df_copy.groupby('month'):
        monthly[m] = {
            'n_trades': len(grp),
            'total_pnl': float(grp['net_pnl_ticks'].sum()),
            'avg_pnl': float(grp['net_pnl_ticks'].mean()),
            'win_rate': float((grp['net_pnl_ticks'] > 0).mean()),
        }
    r['monthly'] = monthly

    return r


def confidence_decile_analysis(trades_df):
    """Performance by confidence decile."""
    df = trades_df.copy()
    df['decile'] = pd.qcut(df['confidence'], 10, labels=False, duplicates='drop')

    results = []
    for d in sorted(df['decile'].unique()):
        sub = df[df['decile'] == d]
        results.append({
            'decile': int(d),
            'n_trades': len(sub),
            'conf_range': f"{sub['confidence'].min():.3f} - {sub['confidence'].max():.3f}",
            'avg_net_pnl': float(sub['net_pnl_ticks'].mean()),
            'win_rate': float((sub['net_pnl_ticks'] > 0).mean()),
            'total_pnl': float(sub['net_pnl_ticks'].sum()),
        })
    return results


# ============================================================
# MAIN
# ============================================================
if __name__ == '__main__':
    print("=" * 70)
    print("GROUND-UP 2H ES LGBM LIVE REPLAY")
    print("=" * 70)
    print()

    # Step 1: Pre-compute all features
    print("Step 1: Pre-computing features from raw minute bars...")
    all_features = precompute_all_features()
    n_dates = all_features['date'].nunique()
    print(f"  Computed {len(all_features)} prediction points across {n_dates} days")
    print(f"  Predictions per day: ~{len(all_features)/n_dates:.1f}")
    print(f"  Date range: {all_features['date'].min()} to {all_features['date'].max()}")
    print(f"  Label stats: mean={all_features['label'].mean():.3f}, "
          f"std={all_features['label'].std():.3f}, "
          f"min={all_features['label'].min():.1f}, max={all_features['label'].max():.1f}")
    print()

    # Step 2: Walk-forward
    print("Step 2: Running walk-forward simulation...")
    print(f"  Train={TRAIN_DAYS}d, Purge={PURGE_DAYS}d, Horizon={HORIZON_MINUTES}min")
    print(f"  RT cost={TOTAL_RT_COST_TICKS} ticks (${TOTAL_RT_COST_TICKS * TICK_VALUE_USD:.2f})")
    print()

    trades_df, feature_cols = run_walkforward(all_features)

    print(f"\nTotal trades generated: {len(trades_df)}")

    # Save trades
    trades_df.to_csv(os.path.join(OUTPUT_DIR, 'trades.csv'), index=False)

    # Step 3: Analysis
    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    def pr(r):
        if r['n_trades'] == 0:
            print("  No trades")
            return
        print(f"  Trades: {r['n_trades']} ({r['n_long']} long, {r['n_short']} short)")
        print(f"  Win Rate: {r['win_rate']:.1%}")
        print(f"  Avg Net P&L: {r['avg_net_pnl_ticks']:.3f} ticks (${r['avg_net_pnl_usd']:.2f})")
        print(f"  Total Net P&L: {r['total_net_pnl_ticks']:.1f} ticks (${r['total_net_pnl_usd']:.2f})")
        print(f"  Avg Gross P&L: {r['avg_gross_pnl_ticks']:.3f} ticks")
        print(f"  Daily Sharpe: {r['daily_sharpe']:.3f} (Ann: {r['annualized_sharpe']:.2f})")
        print(f"  Daily Sortino: {r['daily_sortino']:.3f} (Ann: {r['annualized_sortino']:.2f})")
        print(f"  Profit Factor: {r['profit_factor']:.2f}")
        print(f"  Max Drawdown: {r['max_drawdown_ticks']:.1f} ticks (${r['max_drawdown_usd']:.2f})")
        print(f"  Max Consecutive Losses: {r['max_consecutive_losses']}")
        print(f"  Long: WR={r['long_win_rate']:.1%}, Avg={r['long_avg_pnl']:.3f} ticks")
        print(f"  Short: WR={r['short_win_rate']:.1%}, Avg={r['short_avg_pnl']:.3f} ticks")

    all_results = {}

    # All trades
    print("\n--- ALL TRADES (no confidence filter) ---")
    r = analyze_results(trades_df, 0.0)
    all_results['all_trades'] = r
    pr(r)

    # Thresholds
    for t in [0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0]:
        n = (trades_df['confidence'] > t).sum()
        if n > 10:
            print(f"\n--- CONFIDENCE > {t} ({n} trades) ---")
            r = analyze_results(trades_df, t)
            all_results[f'threshold_{t}'] = r
            pr(r)

    # Decile analysis
    print("\n--- CONFIDENCE DECILE ANALYSIS ---")
    deciles = confidence_decile_analysis(trades_df)
    all_results['decile_analysis'] = deciles
    for d in deciles:
        print(f"  D{d['decile']}: conf={d['conf_range']}, n={d['n_trades']}, "
              f"avg={d['avg_net_pnl']:.3f}, WR={d['win_rate']:.1%}, tot={d['total_pnl']:.1f}")

    # Monthly
    print("\n--- MONTHLY BREAKDOWN (all trades) ---")
    if 'monthly' in all_results['all_trades']:
        for m, stats in sorted(all_results['all_trades']['monthly'].items()):
            print(f"  {m}: n={stats['n_trades']}, pnl={stats['total_pnl']:.1f} ticks, "
                  f"avg={stats['avg_pnl']:.3f}, WR={stats['win_rate']:.1%}")

    # Per hour
    print("\n--- PER PREDICTION HOUR ---")
    if 'per_hour' in all_results['all_trades']:
        for h, stats in sorted(all_results['all_trades']['per_hour'].items()):
            # Convert UTC hour offset to ET time
            utc_hour = int(h) + 9  # minute 60 = 14:30 UTC = hour 1 after 13:30
            et_hour = utc_hour - 4  # UTC to ET (summer)
            print(f"  Hour {h} (~{et_hour}:30 ET): n={stats['n_trades']}, "
                  f"avg={stats['avg_pnl']:.3f}, WR={stats['win_rate']:.1%}, tot={stats['total_pnl']:.1f}")

    # Save
    with open(os.path.join(OUTPUT_DIR, 'replay_results.json'), 'w') as f:
        json.dump(all_results, f, indent=2, default=str)

    print(f"\n{'='*70}")
    print("Files saved to:", OUTPUT_DIR)
    print("  - trades.csv")
    print("  - replay_results.json")
    print("  - replay_script.py")
    print("=" * 70)
