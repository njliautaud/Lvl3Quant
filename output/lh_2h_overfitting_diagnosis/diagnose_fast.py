"""
2H LGBM Overfitting Diagnosis - FAST VERSION
==============================================
Key hypothesis: microprice_last is a raw level (~25000-28000) that encodes
the date/regime. In a trending market, this single feature lets the model
predict direction trivially.

Also tests: the REAL issue is likely overfitting with 30 features on
~300 training samples (60 days * 5 preds/day).
"""

import pandas as pd
import numpy as np
import lightgbm as lgb
import os
import json
import sys

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


def load_and_compute():
    """Load all data and compute features."""
    files = sorted([f for f in os.listdir(DATA_DIR) if f.endswith('.parquet')])
    all_records = []

    for f in files:
        date_str = f.replace('.parquet', '')
        df = pd.read_parquet(os.path.join(DATA_DIR, f))
        df['ts_minute'] = pd.to_datetime(df['ts_minute'])
        df = df.reset_index(drop=True)

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
            ofi = window['ofi_1min']
            feat['ofi_sum'] = ofi.sum()
            feat['ofi_mean'] = ofi.mean()
            feat['ofi_std'] = ofi.std()
            feat['ofi_last10'] = ofi.iloc[-10:].sum()
            feat['ofi_first10'] = ofi.iloc[:10].sum()
            feat['ofi_trend'] = feat['ofi_last10'] - feat['ofi_first10']

            feat['signed_vol_sum'] = window['signed_volume'].sum()
            feat['signed_vol_mean'] = window['signed_volume'].mean()
            feat['volume_sum'] = window['volume'].sum()
            feat['volume_mean'] = window['volume'].mean()
            feat['volume_std'] = window['volume'].std()
            v_first = window['volume'].iloc[:30].sum()
            feat['vol_ratio'] = window['volume'].iloc[-30:].sum() / max(v_first, 1)

            feat['close_change'] = window['close'].iloc[-1] - window['close'].iloc[0]
            feat['close_change_last30'] = window['close'].iloc[-1] - window['close'].iloc[-30]
            feat['high_low_range'] = window['high'].max() - window['low'].min()
            feat['close_vs_high'] = window['close'].iloc[-1] - window['high'].max()
            feat['close_vs_low'] = window['close'].iloc[-1] - window['low'].min()
            bar_range = window['high'] - window['low']
            feat['bar_range_mean'] = bar_range.mean()
            feat['bar_range_std'] = bar_range.std()

            feat['microprice_last'] = window['microprice_close'].iloc[-1]
            feat['microprice_vs_close'] = window['microprice_close'].iloc[-1] - window['close'].iloc[-1]
            feat['microprice_trend'] = window['microprice_close'].iloc[-1] - window['microprice_close'].iloc[0]

            feat['spread_mean'] = window['spread_mean'].mean()
            feat['spread_last'] = window['spread_mean'].iloc[-1]
            feat['spread_max'] = window['spread_mean'].max()

            feat['vwap_vs_close'] = window['vwap'].iloc[-1] - window['close'].iloc[-1]
            feat['vwap_trend'] = window['vwap'].iloc[-1] - window['vwap'].iloc[0]

            feat['trade_count_sum'] = window['trade_count'].sum()
            feat['trade_count_mean'] = window['trade_count'].mean()
            tc_first = window['trade_count'].iloc[:30].sum()
            feat['trade_count_ratio'] = window['trade_count'].iloc[-30:].sum() / max(tc_first, 1)

            vol_regime_map = {'low': 0, 'medium': 1, 'high': 2}
            feat['vol_regime'] = vol_regime_map.get(window['vol_regime'].iloc[-1], 1)
            feat['hour_of_day'] = pred_idx / 60.0

            feat['date'] = date_str
            feat['pred_minute_idx'] = pred_idx
            feat['label'] = label
            feat['close_at_pred'] = close_now

            all_records.append(feat)

    return pd.DataFrame(all_records)


def run_wf(df, feature_cols, desc="", max_oot_days=None):
    """Fast walkforward."""
    meta_cols = ['date', 'pred_minute_idx', 'label', 'close_at_pred']
    dates = sorted(df['date'].unique())
    start_oot_idx = TRAIN_DAYS + PURGE_DAYS

    if max_oot_days:
        end_idx = min(start_oot_idx + max_oot_days, len(dates))
    else:
        end_idx = len(dates)

    all_pnl = []
    all_dates = []

    for oot_idx in range(start_oot_idx, end_idx):
        oot_date = dates[oot_idx]
        train_end_idx = oot_idx - PURGE_DAYS
        train_start_idx = train_end_idx - TRAIN_DAYS
        if train_start_idx < 0:
            continue

        train_dates = set(dates[train_start_idx:train_end_idx])
        train_mask = df['date'].isin(train_dates)
        oot_mask = df['date'] == oot_date

        train_data = df[train_mask]
        oot_data = df[oot_mask]

        if len(train_data) < 50 or len(oot_data) == 0:
            continue

        X_train = train_data[feature_cols].values
        y_train = train_data['label'].values

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
            all_pnl.append(net_pnl)
            all_dates.append(row['date'])

    if not all_pnl:
        return {'sharpe': 0, 'wr': 0, 'n': 0, 'avg_pnl': 0}

    pnl = np.array(all_pnl)
    dates_arr = np.array(all_dates)

    daily = pd.Series(pnl).groupby(dates_arr).sum()
    sharpe = daily.mean() / daily.std() * np.sqrt(252) if daily.std() > 0 else 0

    result = {
        'sharpe': float(sharpe),
        'wr': float((pnl > 0).mean()),
        'n': len(pnl),
        'avg_pnl': float(pnl.mean()),
    }

    print(f"  {desc}: Sharpe={sharpe:.2f}, WR={result['wr']:.1%}, "
          f"N={result['n']}, Avg={result['avg_pnl']:.1f} ticks")
    return result


if __name__ == '__main__':
    print("=" * 70)
    print("FAST OVERFITTING DIAGNOSIS")
    print("=" * 70)

    print("\nLoading and computing features...")
    df = load_and_compute()
    print(f"  {len(df)} predictions across {df['date'].nunique()} days")
    print(f"  Label mean: {df['label'].mean():.2f}, std: {df['label'].std():.2f}")
    print(f"  Label % positive: {(df['label'] > 0).mean():.1%}")
    print(f"  microprice_last range: {df['microprice_last'].min():.0f} - {df['microprice_last'].max():.0f}")
    print()

    meta_cols = ['date', 'pred_minute_idx', 'label', 'close_at_pred']
    all_feature_cols = [c for c in df.columns if c not in meta_cols]

    print(f"  All features ({len(all_feature_cols)}): {all_feature_cols}")
    print()

    # ==========================================
    # CRITICAL TEST: microprice_last is a LEVEL feature
    # ==========================================
    print("=" * 70)
    print("HYPOTHESIS A: microprice_last encodes price LEVEL (leakage proxy)")
    print("=" * 70)
    print()

    # Check correlation between microprice_last and label
    corr = df['microprice_last'].corr(df['label'])
    print(f"  Corr(microprice_last, label) across ALL data: {corr:.4f}")

    # But within each training window, microprice_last is nearly constant!
    # Over 60 days, ES might move 500 ticks. microprice_last goes from e.g. 25500 to 26000.
    # The model can't use this as directional predictor within a 60-day window
    # because it doesn't know which direction it'll go next.
    # UNLESS: the model overfits to the specific level → direction relationship in-sample.

    # Actually the real question: does the model use microprice_last at all?
    # Let's test with and without it.

    # ==========================================
    # TEST A: Full features (replication of original)
    # ==========================================
    print("\n--- TEST A: FULL FEATURES (original, first 50 OOT days for speed) ---")
    res_full = run_wf(df, all_feature_cols, "Full features (50d)", max_oot_days=50)

    # ==========================================
    # TEST B: Without microprice_last (removing level leak)
    # ==========================================
    print("\n--- TEST B: WITHOUT microprice_last ---")
    no_micro = [c for c in all_feature_cols if c != 'microprice_last']
    res_no_micro = run_wf(df, no_micro, "No microprice_last (50d)", max_oot_days=50)

    # ==========================================
    # TEST C: Only RELATIVE features (no levels)
    # ==========================================
    print("\n--- TEST C: ONLY RELATIVE FEATURES (no raw levels) ---")
    relative_features = [
        'ofi_sum', 'ofi_mean', 'ofi_std', 'ofi_last10', 'ofi_first10', 'ofi_trend',
        'signed_vol_sum', 'signed_vol_mean', 'volume_sum', 'volume_mean', 'volume_std', 'vol_ratio',
        'close_change', 'close_change_last30', 'high_low_range',
        'close_vs_high', 'close_vs_low', 'bar_range_mean', 'bar_range_std',
        'microprice_vs_close', 'microprice_trend',
        'spread_mean', 'spread_last', 'spread_max',
        'vwap_vs_close', 'vwap_trend',
        'trade_count_sum', 'trade_count_mean', 'trade_count_ratio',
        'vol_regime', 'hour_of_day',
    ]
    available_rel = [f for f in relative_features if f in df.columns]
    res_relative = run_wf(df, available_rel, "Relative features only (50d)", max_oot_days=50)

    # ==========================================
    # TEST D: microprice_last ONLY (single feature)
    # ==========================================
    print("\n--- TEST D: microprice_last ONLY ---")
    res_micro_only = run_wf(df, ['microprice_last'], "microprice_last only (50d)", max_oot_days=50)

    # ==========================================
    # TEST E: STRONGLY REGULARIZED (fewer leaves, more data needed)
    # ==========================================
    print("\n--- TEST E: STRONGLY REGULARIZED LGBM ---")
    # Save original params and swap
    import copy
    orig_params = LGB_PARAMS.copy()
    LGB_PARAMS.update({
        'num_leaves': 8,
        'min_data_in_leaf': 30,
        'lambda_l1': 1.0,
        'lambda_l2': 1.0,
        'feature_fraction': 0.5,
    })
    res_regularized = run_wf(df, all_feature_cols, "Regularized (50d)", max_oot_days=50)
    LGB_PARAMS.update(orig_params)

    # ==========================================
    # TEST F: SHUFFLED LABELS (controls for everything)
    # ==========================================
    print("\n--- TEST F: SHUFFLED LABELS (within each day) ---")
    df_shuffled = df.copy()
    np.random.seed(42)
    for d in df_shuffled['date'].unique():
        mask = df_shuffled['date'] == d
        labels = df_shuffled.loc[mask, 'label'].values.copy()
        np.random.shuffle(labels)
        df_shuffled.loc[mask, 'label'] = labels
    res_shuffled = run_wf(df_shuffled, all_feature_cols, "Shuffled labels (50d)", max_oot_days=50)

    # ==========================================
    # TEST G: FEATURE IMPORTANCE FROM ORIGINAL MODEL
    # ==========================================
    print("\n" + "=" * 70)
    print("TEST G: FEATURE IMPORTANCE (single model, latest window)")
    print("=" * 70)

    dates = sorted(df['date'].unique())
    # Train on last 60 days before the last OOT day
    last_oot = dates[-1]
    train_end_idx = len(dates) - 1 - PURGE_DAYS
    train_start_idx = train_end_idx - TRAIN_DAYS
    train_dates = set(dates[train_start_idx:train_end_idx])

    train_data = df[df['date'].isin(train_dates)]
    X = train_data[all_feature_cols].values
    y = train_data['label'].values

    dtrain = lgb.Dataset(X[:int(len(X)*0.8)], label=y[:int(len(y)*0.8)])
    dval = lgb.Dataset(X[int(len(X)*0.8):], label=y[int(len(y)*0.8):], reference=dtrain)

    model = lgb.train(
        LGB_PARAMS, dtrain,
        num_boost_round=LGB_NUM_ROUNDS,
        valid_sets=[dval],
        callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False)],
    )

    importances = model.feature_importance(importance_type='gain')
    feat_imp = sorted(zip(all_feature_cols, importances), key=lambda x: -x[1])

    print("\n  Top features by gain:")
    for name, imp in feat_imp[:15]:
        print(f"    {name:25s}: {imp:.0f}")

    # ==========================================
    # TEST H: THE REAL ISSUE - Sample count analysis
    # ==========================================
    print("\n" + "=" * 70)
    print("TEST H: SAMPLE SIZE ANALYSIS")
    print("=" * 70)

    n_train_samples = TRAIN_DAYS * 5  # ~300 samples
    n_features = len(all_feature_cols)
    ratio = n_train_samples / n_features

    print(f"\n  Training samples per window: ~{n_train_samples}")
    print(f"  Number of features: {n_features}")
    print(f"  Sample/feature ratio: {ratio:.1f}:1")
    print(f"  LGB num_leaves=31 → up to 31 leaves per tree, 500 trees max")
    print(f"  With early stopping at 50 rounds, likely uses ~100-200 trees")
    print(f"  Effective parameters: ~31 * 150 * 0.8 feature_frac = ~3,720")
    print(f"  This VASTLY exceeds {n_train_samples} samples → GUARANTEED OVERFIT")

    # ==========================================
    # TEST I: CROSS-VALIDATION WITHIN TRAINING (check in-sample vs OOT)
    # ==========================================
    print("\n" + "=" * 70)
    print("TEST I: IN-SAMPLE vs OOT COMPARISON")
    print("=" * 70)

    # For one window, check how well it fits in-sample vs OOT
    dates = sorted(df['date'].unique())
    start_oot_idx = TRAIN_DAYS + PURGE_DAYS

    is_preds_all = []
    oot_preds_all = []

    for oot_idx in range(start_oot_idx, min(start_oot_idx + 20, len(dates))):
        oot_date = dates[oot_idx]
        train_end_idx = oot_idx - PURGE_DAYS
        train_start_idx = train_end_idx - TRAIN_DAYS
        if train_start_idx < 0:
            continue

        train_dates = set(dates[train_start_idx:train_end_idx])
        train_data = df[df['date'].isin(train_dates)]
        oot_data = df[df['date'] == oot_date]

        if len(train_data) < 50 or len(oot_data) == 0:
            continue

        X_train = train_data[all_feature_cols].values
        y_train = train_data['label'].values

        split_idx = int(len(X_train) * 0.8)
        dtrain = lgb.Dataset(X_train[:split_idx], label=y_train[:split_idx])
        dval = lgb.Dataset(X_train[split_idx:], label=y_train[split_idx:], reference=dtrain)

        model = lgb.train(
            LGB_PARAMS, dtrain,
            num_boost_round=LGB_NUM_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False)],
        )

        # In-sample predictions
        is_preds = model.predict(X_train)
        is_corr = np.corrcoef(is_preds, y_train)[0, 1]

        # OOT predictions
        X_oot = oot_data[all_feature_cols].values
        oot_preds = model.predict(X_oot)
        oot_labels = oot_data['label'].values
        oot_corr = np.corrcoef(oot_preds, oot_labels)[0, 1] if len(oot_labels) > 2 else 0

        is_preds_all.append(is_corr)
        oot_preds_all.append(oot_corr)

    print(f"\n  In-sample correlation (pred vs actual): {np.mean(is_preds_all):.4f}")
    print(f"  OOT correlation (pred vs actual): {np.mean(oot_preds_all):.4f}")
    print(f"  Ratio (OOT/IS): {np.mean(oot_preds_all)/np.mean(is_preds_all):.4f}")
    print(f"  IS is {'MUCH' if np.mean(is_preds_all) > 3 * np.mean(oot_preds_all) else 'somewhat'} higher → overfitting")

    # ==========================================
    # FINAL: The key question - IS THE OOT IC REAL?
    # ==========================================
    print("\n" + "=" * 70)
    print("KEY DIAGNOSTIC: OOT IC (Information Coefficient)")
    print("=" * 70)

    # Re-run full walkforward but collect IC statistics
    dates = sorted(df['date'].unique())
    start_oot_idx = TRAIN_DAYS + PURGE_DAYS

    daily_ics = []
    daily_directional_acc = []

    for oot_idx in range(start_oot_idx, len(dates)):
        oot_date = dates[oot_idx]
        train_end_idx = oot_idx - PURGE_DAYS
        train_start_idx = train_end_idx - TRAIN_DAYS
        if train_start_idx < 0:
            continue

        train_dates = set(dates[train_start_idx:train_end_idx])
        train_data = df[df['date'].isin(train_dates)]
        oot_data = df[df['date'] == oot_date]

        if len(train_data) < 50 or len(oot_data) == 0:
            continue

        X_train = train_data[all_feature_cols].values
        y_train = train_data['label'].values

        split_idx = int(len(X_train) * 0.8)
        dtrain = lgb.Dataset(X_train[:split_idx], label=y_train[:split_idx])
        dval = lgb.Dataset(X_train[split_idx:], label=y_train[split_idx:], reference=dtrain)

        model = lgb.train(
            LGB_PARAMS, dtrain,
            num_boost_round=LGB_NUM_ROUNDS,
            valid_sets=[dval],
            callbacks=[lgb.early_stopping(LGB_EARLY_STOPPING, verbose=False)],
        )

        X_oot = oot_data[all_feature_cols].values
        preds = model.predict(X_oot)
        actuals = oot_data['label'].values

        if len(actuals) > 2:
            ic = np.corrcoef(preds, actuals)[0, 1]
            daily_ics.append(ic)

            # Directional accuracy
            dir_acc = np.mean(np.sign(preds) == np.sign(actuals))
            daily_directional_acc.append(dir_acc)

    print(f"\n  Daily OOT IC: mean={np.mean(daily_ics):.4f}, std={np.std(daily_ics):.4f}")
    print(f"  Daily OOT IC: median={np.median(daily_ics):.4f}")
    print(f"  % days with positive IC: {(np.array(daily_ics) > 0).mean():.1%}")
    print(f"  Daily directional accuracy: {np.mean(daily_directional_acc):.1%}")
    print(f"  t-stat of IC: {np.mean(daily_ics) / (np.std(daily_ics) / np.sqrt(len(daily_ics))):.2f}")

    # ==========================================
    # SUMMARY
    # ==========================================
    print("\n" + "=" * 70)
    print("DIAGNOSIS SUMMARY")
    print("=" * 70)

    results = {
        'test_a_full': res_full,
        'test_b_no_micro': res_no_micro,
        'test_c_relative': res_relative,
        'test_d_micro_only': res_micro_only,
        'test_e_regularized': res_regularized,
        'test_f_shuffled': res_shuffled,
        'feature_importance_top10': feat_imp[:10],
        'oot_ic_mean': float(np.mean(daily_ics)),
        'oot_ic_std': float(np.std(daily_ics)),
        'oot_directional_acc': float(np.mean(daily_directional_acc)),
        'is_corr_mean': float(np.mean(is_preds_all)),
        'oot_corr_mean': float(np.mean(oot_preds_all)),
    }

    with open(os.path.join(OUTPUT_DIR, 'fast_diagnosis_results.json'), 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n  Full features Sharpe (50d): {res_full['sharpe']:.2f}")
    print(f"  Without microprice_last: {res_no_micro['sharpe']:.2f}")
    print(f"  Relative features only: {res_relative['sharpe']:.2f}")
    print(f"  microprice_last ONLY: {res_micro_only['sharpe']:.2f}")
    print(f"  Regularized: {res_regularized['sharpe']:.2f}")
    print(f"  Shuffled labels: {res_shuffled['sharpe']:.2f}")
    print(f"  OOT IC: {np.mean(daily_ics):.4f}")
    print(f"  OOT directional accuracy: {np.mean(daily_directional_acc):.1%}")
    print(f"\n  Label overlap inflation: 1.39x (from prior test)")
    print(f"  Corrected full-features Sharpe: {res_full['sharpe'] / 1.39:.2f}")

    print(f"\n  Saved to: {OUTPUT_DIR}fast_diagnosis_results.json")
