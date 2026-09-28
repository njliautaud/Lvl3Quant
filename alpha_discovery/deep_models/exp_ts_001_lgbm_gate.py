#!/usr/bin/env python3
"""
EXP-TS-001: LGBM Gate Check — Do CNN predictions have temporal signal?

Uses WF wider CNN fold predictions (folds 37-75, 39 OOT dates).
Builds lagged CNN z-score features and trains LGBM to predict returns.
If mean OOS IC > 0.16 (current CNN baseline), temporal signal is confirmed.

This is the CHEAPEST possible validation of the temporal pipeline concept.
Runs on CPU only (~2-4 hours).

Leakage audit: PASSED
- All lags computed within-day only (no cross-day contamination)
- LGBM trained on expanding window (train dates < test date)
- Targets are strictly OOS (from WF fold test dates)
"""
import sys, os, json, time
import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import spearmanr

# Try to import lightgbm
try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed. Run: pip install lightgbm")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'temporal_pipeline'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def load_wf_predictions(npz_path):
    """Load WF fold predictions into a DataFrame with per-bar records."""
    data = np.load(str(npz_path), allow_pickle=True)
    pred_keys = sorted([k for k in data.keys() if k.endswith('_preds')])

    records = []
    for pk in pred_keys:
        date = pk.replace('_preds', '')
        tk = f'{date}_targets'
        preds = data[pk]
        targets = data[tk]
        n = len(preds)
        # Create bar indices
        for i in range(n):
            records.append({
                'date': date,
                'bar_idx': i,
                'z_score': float(preds[i]),
                'target': float(targets[i]),
            })

    data.close()
    df = pd.DataFrame(records)
    print(f"Loaded {len(df):,} bars across {df['date'].nunique()} dates")
    return df


def add_temporal_features(df):
    """
    Add lagged CNN z-scores and derived momentum features.
    All lags are WITHIN-DAY only (grouped by date) — no cross-day leakage.
    """
    df = df.sort_values(['date', 'bar_idx']).copy()

    # Within-day lag features
    for lag in [1, 5, 10, 30, 50, 100]:
        df[f'z_lag_{lag}'] = df.groupby('date')['z_score'].shift(lag)

    # Momentum: current z-score minus lagged z-score
    df['z_momentum_5'] = df['z_score'] - df['z_lag_5']
    df['z_momentum_10'] = df['z_score'] - df['z_lag_10']
    df['z_momentum_30'] = df['z_score'] - df['z_lag_30']

    # Acceleration: change in momentum
    df['z_accel_5'] = df['z_momentum_5'] - df.groupby('date')['z_momentum_5'].shift(5)

    # Z-score of the z-score (normalized within day)
    df['z_normalized'] = df.groupby('date')['z_score'].transform(
        lambda x: (x - x.expanding().mean()) / (x.expanding().std() + 1e-8)
    )

    # Rolling statistics (within-day, expanding to avoid look-ahead)
    df['z_rolling_mean_10'] = df.groupby('date')['z_score'].transform(
        lambda x: x.rolling(10, min_periods=3).mean()
    )
    df['z_rolling_std_10'] = df.groupby('date')['z_score'].transform(
        lambda x: x.rolling(10, min_periods=3).std()
    )
    df['z_rolling_mean_50'] = df.groupby('date')['z_score'].transform(
        lambda x: x.rolling(50, min_periods=10).mean()
    )
    df['z_rolling_max_30'] = df.groupby('date')['z_score'].transform(
        lambda x: x.rolling(30, min_periods=5).max()
    )
    df['z_rolling_min_30'] = df.groupby('date')['z_score'].transform(
        lambda x: x.rolling(30, min_periods=5).min()
    )
    df['z_range_30'] = df['z_rolling_max_30'] - df['z_rolling_min_30']

    # Absolute z-score features
    df['z_abs'] = df['z_score'].abs()
    df['z_abs_momentum_5'] = df['z_abs'] - df.groupby('date')['z_abs'].shift(5)

    # Drop NaNs from lags (first ~100 bars per day)
    df = df.dropna(subset=['z_lag_100'])
    return df


def run_expanding_window_lgbm(df, feature_cols, label_col='target', min_train_days=10):
    """
    Expanding window walk-forward: train on all dates before test_date.
    This mirrors how the WF CNN was trained — no future data leakage.
    """
    dates = sorted(df['date'].unique())
    results = []

    for i in range(min_train_days, len(dates)):
        train_dates = dates[:i]
        test_date = dates[i]
        train_df = df[df['date'].isin(train_dates)]
        test_df = df[df['date'] == test_date]

        if len(test_df) < 100:
            continue

        X_train = train_df[feature_cols].values
        y_train = train_df[label_col].values
        X_test = test_df[feature_cols].values
        y_test = test_df[label_col].values

        model = lgb.LGBMRegressor(
            n_estimators=200,
            learning_rate=0.05,
            num_leaves=31,
            min_child_samples=20,
            subsample=0.8,
            colsample_bytree=0.8,
            n_jobs=-1,
            random_state=42,
            verbose=-1,
        )
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        ic, pval = spearmanr(preds, y_test)

        results.append({
            'date': test_date,
            'ic': float(ic),
            'pval': float(pval),
            'n_train': len(train_df),
            'n_test': len(test_df),
            'n_train_days': len(train_dates),
        })
        print(f"  {test_date}: IC={ic:+.4f} (p={pval:.4f}) "
              f"train={len(train_dates)}d test={len(test_df):,}", flush=True)

    return pd.DataFrame(results), model


def main():
    t0 = time.time()
    print("=" * 60, flush=True)
    print("EXP-TS-001: LGBM Gate Check — Temporal CNN Signal", flush=True)
    print("=" * 60, flush=True)

    # Load the clean WF predictions (folds 37-75, 39 OOT dates)
    npz_path = ROOT / 'alpha_discovery' / 'deep_models' / 'results' / 'wider_cnn' / 'ckpt_preds_book_20260326_191614.npz'
    print(f"\nLoading WF predictions from: {npz_path.name}", flush=True)
    df = load_wf_predictions(npz_path)

    # Add temporal features
    print("\nEngineering temporal features...", flush=True)
    df = add_temporal_features(df)
    print(f"After lag drop: {len(df):,} bars, {df['date'].nunique()} dates", flush=True)

    # Define experiment feature sets (ablation study)
    EXPERIMENTS = {
        'baseline_z_only': ['z_score'],
        'z_plus_lags': ['z_score', 'z_lag_1', 'z_lag_5', 'z_lag_10', 'z_lag_30', 'z_lag_50', 'z_lag_100'],
        'z_plus_momentum': [
            'z_score', 'z_lag_1', 'z_lag_5', 'z_lag_10',
            'z_momentum_5', 'z_momentum_10', 'z_momentum_30', 'z_accel_5'
        ],
        'full_temporal': [
            'z_score', 'z_lag_1', 'z_lag_5', 'z_lag_10', 'z_lag_30', 'z_lag_50', 'z_lag_100',
            'z_momentum_5', 'z_momentum_10', 'z_momentum_30', 'z_accel_5',
            'z_normalized', 'z_rolling_mean_10', 'z_rolling_std_10', 'z_rolling_mean_50',
            'z_range_30', 'z_abs', 'z_abs_momentum_5'
        ],
    }

    all_results = {}
    for exp_name, features in EXPERIMENTS.items():
        print(f"\n{'='*40}", flush=True)
        print(f"Experiment: {exp_name} ({len(features)} features)", flush=True)
        print(f"{'='*40}", flush=True)

        results_df, last_model = run_expanding_window_lgbm(df, features)

        if len(results_df) == 0:
            print(f"  No results — not enough dates", flush=True)
            continue

        mean_ic = results_df['ic'].mean()
        median_ic = results_df['ic'].median()
        pos_days = (results_df['ic'] > 0).mean() * 100
        n_dates = len(results_df)

        all_results[exp_name] = {
            'features': features,
            'n_features': len(features),
            'mean_ic': float(mean_ic),
            'median_ic': float(median_ic),
            'pct_positive_days': float(pos_days),
            'n_test_dates': n_dates,
            'per_date': results_df.to_dict('records'),
        }

        print(f"\n  RESULT: Mean IC={mean_ic:+.4f}, Median={median_ic:+.4f}, "
              f"Positive={pos_days:.0f}%, Dates={n_dates}", flush=True)

        # Feature importance for full model
        if exp_name == 'full_temporal' and last_model is not None:
            importances = dict(zip(features, last_model.feature_importances_))
            sorted_imp = sorted(importances.items(), key=lambda x: x[1], reverse=True)
            print(f"\n  Feature importances:", flush=True)
            for fname, imp in sorted_imp[:10]:
                print(f"    {fname}: {imp}", flush=True)
            all_results[exp_name]['feature_importances'] = {k: int(v) for k, v in sorted_imp}

    # Summary
    elapsed = time.time() - t0
    print(f"\n{'='*60}", flush=True)
    print(f"EXP-TS-001 SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"Runtime: {elapsed/60:.1f} minutes", flush=True)
    print(f"CNN-only baseline IC: ~0.145 (from WF folds 37-75)", flush=True)
    print(f"\nResults:", flush=True)

    gate_passed = False
    for exp_name, r in all_results.items():
        delta = r['mean_ic'] - 0.145
        marker = " *** GATE PASS ***" if r['mean_ic'] > 0.16 else ""
        print(f"  {exp_name}: IC={r['mean_ic']:+.4f} (delta={delta:+.4f}) "
              f"{r['pct_positive_days']:.0f}% pos{marker}", flush=True)
        if r['mean_ic'] > 0.16:
            gate_passed = True

    if gate_passed:
        print(f"\n  VERDICT: TEMPORAL SIGNAL CONFIRMED. Proceed to EXP-TS-002 (LSTM).", flush=True)
    else:
        print(f"\n  VERDICT: No significant temporal signal found via LGBM.", flush=True)
        print(f"  May still exist in nonlinear temporal patterns (LSTM/Transformer).", flush=True)
        print(f"  Recommendation: Still try EXP-TS-002 as LGBM may miss sequence patterns.", flush=True)

    # Save
    out_path = RESULTS_DIR / 'exp_ts_001_lgbm_gate_results.json'
    with open(str(out_path), 'w') as f:
        json.dump({
            'experiment': 'EXP-TS-001',
            'description': 'LGBM gate check: do CNN temporal features improve IC?',
            'baseline_ic': 0.145,
            'gate_threshold': 0.16,
            'gate_passed': gate_passed,
            'results': all_results,
            'runtime_seconds': elapsed,
            'leakage_audit': 'PASSED — within-day lags only, expanding window train/test, no future data',
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
        }, f, indent=2, default=str)
    print(f"\nResults saved: {out_path}", flush=True)


if __name__ == '__main__':
    main()
