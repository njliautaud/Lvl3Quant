#!/usr/bin/env python3
"""
ES 2h Model — MBO Feature Ablation Study (HC #662 R4)
=====================================================

Tests whether the 2h directional model retains predictive power
when MBO-derived features are removed, leaving only OHLCV-derived features.

If IC > 0.15 without MBO → deployable NOW without Razer dependency.

Uses FIXED methodology:
  - 80/20 train split for early stopping (NOT test set — that was leakage)
  - 5-day purge gap between train and test
  - Sliding walk-forward (60d train, 1d OOT)

Author: Claude (HC #662 R4 research — reduce MBO dependency)
"""

import gc
import json
import logging
import os
import sys
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
OUTPUT_DIR = ROOT / "output" / "lh_2h_mbo_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [MBO-ABLATION] %(levelname)s %(message)s',
    level=logging.INFO,
)
log = logging.getLogger('MBO-ABLATION')

# ── MBO vs OHLCV feature classification ──
# These hourly features are derived FROM MBO-exclusive minute columns
MBO_DERIVED_FEATURES = {
    # From ofi_1min:
    'ofi_sum', 'ofi_mean', 'ofi_trend', 'ofi_consistency', 'ofi_late_vs_early',
    # From signed_volume:
    'signed_volume_sum', 'signed_volume_ratio', 'buy_volume_fraction',
    'sell_volume_fraction', 'sweep_minutes', 'max_sweep_intensity',
    # From spread_mean:
    'spread_mean', 'spread_max',
    # From trade_count:
    'trade_count_sum', 'trade_intensity',
    # From vwap:
    'vwap_dev',
    # Rolling features derived from MBO base features:
    'ofi_sum_2h', 'ofi_sum_4h', 'ofi_sum_6h',
    'ofi_trend_2h', 'ofi_trend_4h', 'ofi_trend_6h',
    'sv_sum_2h', 'sv_sum_4h', 'sv_sum_6h',
}

# ── Cost constants (HC canonical) ──
COST_PASSIVE_RT_TICKS = 0.376

# ── Walk-forward parameters ──
TRAIN_DAYS = 60
PURGE_DAYS = 5  # Fix for overlapping 2h labels
TARGET = 'fwd_ticks_2h'
HORIZON = 2  # hours


def load_minute_bars():
    """Load all minute bar parquets."""
    files = sorted(MINUTE_BAR_DIR.glob("*.parquet"))
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df['date'] = f.stem
            frames.append(df)
        except Exception as e:
            log.warning(f"Skip {f.stem}: {e}")
    combined = pd.concat(frames, ignore_index=True)
    combined['ts_minute'] = pd.to_datetime(combined['ts_minute'], utc=True)
    combined = combined.sort_values('ts_minute').reset_index(drop=True)
    log.info(f"Loaded {len(combined)} minute bars, {len(frames)} days")
    return combined


def compute_hourly_features(df):
    """Aggregate 1-min bars -> hourly bars with ALL features (MBO + OHLCV)."""
    df = df.copy()
    df['hour'] = df['ts_minute'].dt.hour
    df['date_str'] = df['date']
    df['return_1m'] = df.groupby('date_str')['close'].pct_change()

    # MBO-derived helpers (fill with 0 if columns missing)
    for col in ['ofi_1min', 'signed_volume', 'spread_mean', 'trade_count', 'vwap']:
        if col not in df.columns:
            df[col] = 0.0

    df['abs_ofi'] = df['ofi_1min'].abs()
    sv_std = df.groupby('date_str')['signed_volume'].transform('std').replace(0, 1)
    df['sv_zscore'] = df['signed_volume'] / sv_std
    df['vwap_dev'] = (df['close'] - df['vwap']) / df['close'].clip(lower=1)

    records = []
    for (date_str, hour), g in df.groupby(['date_str', 'hour']):
        if len(g) < 5:
            continue
        c = g['close'].values
        v = g['volume'].values
        ofi = g['ofi_1min'].values
        sv = g['signed_volume'].values
        ret = g['return_1m'].fillna(0).values
        sp = g['spread_mean'].values
        tc = g['trade_count'].values

        rec = {
            'date': date_str, 'hour': hour, 'ts': g['ts_minute'].iloc[0],
            'open': c[0], 'high': c.max(), 'low': c.min(), 'close': c[-1],
            # OHLCV features:
            'return_1h': (c[-1] / c[0] - 1) if c[0] > 0 else 0,
            'range_ticks': (c.max() - c.min()),
            'close_position': (c[-1] - c.min()) / max(c.max() - c.min(), 1),
            'total_volume': v.sum(),
            'avg_volume': v.mean(),
            'volume_trend': np.polyfit(np.arange(len(v)), v, 1)[0] if len(v) > 1 else 0,
            'volume_concentration': v.max() / max(v.mean(), 1),
            'return_std': ret.std(),
            'return_skew': float(stats.skew(ret)) if len(ret) > 3 else 0,
            'realized_vol': ret.std() * np.sqrt(60),
            'vol_asymmetry': float(np.mean(ret[ret < 0]**2) / max(np.mean(ret[ret > 0]**2), 1e-10)) if (ret < 0).any() and (ret > 0).any() else 1.0,
            # MBO features:
            'ofi_sum': ofi.sum(),
            'ofi_mean': ofi.mean(),
            'ofi_trend': np.polyfit(np.arange(len(ofi)), ofi, 1)[0] if len(ofi) > 1 else 0,
            'ofi_consistency': np.mean(np.sign(ofi) == np.sign(ofi.sum())) if ofi.sum() != 0 else 0.5,
            'ofi_late_vs_early': ofi[len(ofi)//2:].sum() - ofi[:len(ofi)//2].sum(),
            'signed_volume_sum': sv.sum(),
            'signed_volume_ratio': sv.sum() / max(v.sum(), 1),
            'buy_volume_fraction': np.sum(sv[sv > 0]) / max(v.sum(), 1),
            'sell_volume_fraction': -np.sum(sv[sv < 0]) / max(v.sum(), 1),
            'sweep_minutes': int(np.sum(np.abs(g['sv_zscore'].values) > 2)),
            'max_sweep_intensity': float(np.abs(g['sv_zscore'].values).max()),
            'spread_mean': sp.mean(),
            'spread_max': sp.max(),
            'trade_count_sum': tc.sum(),
            'trade_intensity': tc.mean(),
        }
        records.append(rec)

    hourly = pd.DataFrame(records)
    hourly = hourly.sort_values('ts').reset_index(drop=True)
    log.info(f"Computed {len(hourly)} hourly bars")
    return hourly


def add_rolling_features(df):
    """Add multi-hour rolling features."""
    for w in [2, 4, 6]:
        lbl = f'{w}h'
        df[f'ofi_sum_{lbl}'] = df['ofi_sum'].rolling(w, min_periods=1).sum()
        df[f'ofi_trend_{lbl}'] = df['ofi_trend'].rolling(w, min_periods=1).mean()
        df[f'sv_sum_{lbl}'] = df['signed_volume_sum'].rolling(w, min_periods=1).sum()
        df[f'volume_ma_{lbl}'] = df['total_volume'].rolling(w, min_periods=1).mean()
        df[f'volume_vs_ma_{lbl}'] = df['total_volume'] / df[f'volume_ma_{lbl}'].clip(lower=1)
        df[f'vol_trend_{lbl}'] = df['realized_vol'].rolling(w, min_periods=1).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True)
    df['mom_2h'] = df['close'].pct_change(2)
    df['mom_4h'] = df['close'].pct_change(4)
    df['mom_6h'] = df['close'].pct_change(6)
    return df


def add_regime_context(df):
    """Add regime features."""
    df['vol_20h'] = df['realized_vol'].rolling(20, min_periods=5).mean()
    df['vol_regime'] = pd.qcut(df['vol_20h'].rank(method='first'), 3, labels=[0, 1, 2]).astype(float)
    df['trend_8h'] = df['close'].pct_change(8)
    df['trend_20h'] = df['close'].pct_change(20)
    df['trend_regime'] = np.where(df['trend_20h'] > 0.005, 1, np.where(df['trend_20h'] < -0.005, -1, 0))
    return df


def add_forward_labels(df):
    """Add 2h forward label."""
    df['fwd_ticks_2h'] = df.groupby('date')['close'].shift(-HORIZON) - df['close']
    return df


def get_feature_cols(df, exclude_mbo=False):
    """Get feature columns, optionally excluding MBO-derived ones."""
    exclude = {'date', 'hour', 'ts', 'open', 'high', 'low', 'close'}
    exclude.update(c for c in df.columns if 'fwd_' in c or 'direction' in c)

    features = [c for c in df.columns
                if c not in exclude
                and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]

    if exclude_mbo:
        features = [c for c in features if c not in MBO_DERIVED_FEATURES]

    return features


def train_walk_forward(df, feature_cols, label='full'):
    """
    Sliding walk-forward with FIXED methodology:
    - 80/20 train split for early stopping
    - 5-day purge gap
    """
    import lightgbm as lgb

    dates = sorted(df['date'].unique())
    log.info(f"[{label}] WF: {len(dates)} dates, {len(feature_cols)} features")

    params = {
        'objective': 'regression', 'metric': 'mse',
        'learning_rate': 0.03, 'num_leaves': 31, 'max_depth': 6,
        'min_data_in_leaf': 50, 'feature_fraction': 0.7,
        'bagging_fraction': 0.8, 'bagging_freq': 5,
        'lambda_l1': 0.1, 'lambda_l2': 1.0,
        'verbose': -1, 'n_jobs': 8, 'seed': 42,
    }

    all_preds, all_actuals, all_dates_list, all_hours = [], [], [], []
    importances = np.zeros(len(feature_cols))
    n_folds = 0

    for fold_start in range(TRAIN_DAYS + PURGE_DAYS, len(dates), 1):
        # Train: fold_start - PURGE - TRAIN_DAYS .. fold_start - PURGE
        # Purge: fold_start - PURGE .. fold_start (skipped)
        # Test: fold_start .. fold_start + 1
        train_end = fold_start - PURGE_DAYS
        train_start = max(0, train_end - TRAIN_DAYS)
        test_end = min(fold_start + 1, len(dates))

        if train_end - train_start < 30:  # Need at least 30 train days
            continue

        train_dates = dates[train_start:train_end]
        test_dates = dates[fold_start:test_end]

        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'].isin(test_dates)

        X_train_full = df.loc[train_mask, feature_cols].values
        y_train_full = df.loc[train_mask, TARGET].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, TARGET].values

        # Filter NaN
        tv = ~np.isnan(y_train_full)
        te = ~np.isnan(y_test)
        if tv.sum() < 50 or te.sum() < 2:
            continue

        X_train_full, y_train_full = X_train_full[tv], y_train_full[tv]
        X_test, y_test = X_test[te], y_test[te]

        X_train_full = np.nan_to_num(X_train_full, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)

        # 80/20 split for early stopping (NOT test set!)
        n_train = int(len(X_train_full) * 0.8)
        X_tr = X_train_full[:n_train]
        y_tr = y_train_full[:n_train]
        X_val = X_train_full[n_train:]
        y_val = y_train_full[n_train:]

        if len(X_val) < 10:
            # Too few val samples, train without early stopping
            td = lgb.Dataset(X_train_full, label=y_train_full)
            model = lgb.train(params, td, num_boost_round=100)
        else:
            td = lgb.Dataset(X_tr, label=y_tr)
            vd = lgb.Dataset(X_val, label=y_val, reference=td)
            model = lgb.train(params, td, num_boost_round=300,
                              valid_sets=[vd],
                              callbacks=[lgb.early_stopping(30, verbose=False)])

        preds = model.predict(X_test)
        all_preds.extend(preds)
        all_actuals.extend(y_test)
        all_dates_list.extend(df.loc[test_mask & ~df[TARGET].isna(), 'date'].values)
        all_hours.extend(df.loc[test_mask & ~df[TARGET].isna(), 'hour'].values)
        importances += model.feature_importance(importance_type='gain')
        n_folds += 1

    if n_folds == 0:
        log.error(f"[{label}] Zero folds completed!")
        return None

    preds = np.array(all_preds)
    actuals = np.array(all_actuals)
    dates_arr = np.array(all_dates_list)
    hours_arr = np.array(all_hours)

    # Compute metrics
    ic = np.corrcoef(preds, actuals)[0, 1] if len(preds) > 10 else 0

    # Top-20% selection-conditioned IC
    abs_preds = np.abs(preds)
    top20_mask = abs_preds >= np.percentile(abs_preds, 80)
    ic_top20 = np.corrcoef(preds[top20_mask], actuals[top20_mask])[0, 1] if top20_mask.sum() > 10 else 0

    # Trade simulation (top-20% by absolute prediction, passive entry)
    directions = np.sign(preds[top20_mask])
    trade_pnl = directions * actuals[top20_mask] - COST_PASSIVE_RT_TICKS
    trade_dates = dates_arr[top20_mask]

    # Daily Sharpe
    unique_trade_dates = sorted(set(trade_dates))
    daily_pnl = []
    for d in unique_trade_dates:
        mask = trade_dates == d
        daily_pnl.append(trade_pnl[mask].sum())
    daily_pnl = np.array(daily_pnl)

    sharpe = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0
    wr = (trade_pnl > 0).mean() * 100
    pf = abs(trade_pnl[trade_pnl > 0].sum() / trade_pnl[trade_pnl < 0].sum()) if (trade_pnl < 0).any() else float('inf')

    # Feature importance
    imp_df = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances / max(n_folds, 1)
    }).sort_values('importance', ascending=False)

    results = {
        'label': label,
        'n_features': len(feature_cols),
        'n_folds': n_folds,
        'n_predictions': len(preds),
        'n_trades': int(top20_mask.sum()),
        'n_trade_days': len(unique_trade_dates),
        'ic_unconditional': round(float(ic), 4),
        'ic_top20': round(float(ic_top20), 4),
        'sharpe': round(float(sharpe), 2),
        'win_rate_pct': round(float(wr), 1),
        'profit_factor': round(float(pf), 2),
        'avg_pnl_ticks': round(float(trade_pnl.mean()), 2),
        'total_pnl_ticks': round(float(trade_pnl.sum()), 1),
        'daily_pnl_mean': round(float(daily_pnl.mean()), 2),
        'daily_pnl_std': round(float(daily_pnl.std()), 2),
        'green_days': int((daily_pnl > 0).sum()),
        'red_days': int((daily_pnl < 0).sum()),
        'top5_features': imp_df.head(5)[['feature', 'importance']].values.tolist(),
    }

    log.info(f"[{label}] IC={ic:.4f}, IC_top20={ic_top20:.4f}, Sharpe={sharpe:.1f}, "
             f"WR={wr:.1f}%, PF={pf:.2f}, {n_folds} folds, {len(preds)} preds, {top20_mask.sum()} trades")

    return results


def main():
    log.info("="*60)
    log.info("ES 2h Model — MBO Feature Ablation Study")
    log.info("Question: Can we predict ES 2h direction WITHOUT MBO data?")
    log.info("="*60)

    # Load and prepare data
    log.info("Loading minute bars...")
    minute_df = load_minute_bars()

    log.info("Computing hourly features...")
    hourly = compute_hourly_features(minute_df)
    del minute_df; gc.collect()

    hourly = add_rolling_features(hourly)
    hourly = add_regime_context(hourly)
    hourly = add_forward_labels(hourly)

    log.info(f"Dataset: {len(hourly)} hourly bars, {hourly['date'].nunique()} days")

    # Get feature sets
    all_features = get_feature_cols(hourly, exclude_mbo=False)
    ohlcv_features = get_feature_cols(hourly, exclude_mbo=True)
    mbo_only_in_full = [f for f in all_features if f not in ohlcv_features]

    log.info(f"ALL features: {len(all_features)}")
    log.info(f"OHLCV-only features: {len(ohlcv_features)}")
    log.info(f"MBO-derived features being removed: {len(mbo_only_in_full)}")
    log.info(f"MBO features: {mbo_only_in_full}")

    # Run both variants
    results = {}

    log.info("\n" + "="*60)
    log.info("VARIANT 1: ALL FEATURES (MBO + OHLCV)")
    log.info("="*60)
    results['all_features'] = train_walk_forward(hourly, all_features, label='ALL (MBO+OHLCV)')

    log.info("\n" + "="*60)
    log.info("VARIANT 2: OHLCV-ONLY (NO MBO)")
    log.info("="*60)
    results['ohlcv_only'] = train_walk_forward(hourly, ohlcv_features, label='OHLCV-ONLY')

    # Also try with ENHANCED OHLCV features (additional engineered features from just OHLCV)
    # Add some extra features that try to capture what MBO gives us, using only OHLCV
    log.info("\n" + "="*60)
    log.info("VARIANT 3: ENHANCED OHLCV (engineered substitutes for MBO)")
    log.info("="*60)

    hourly_enhanced = hourly.copy()
    # Proxy for order flow: volume * sign of return
    hourly_enhanced['volume_direction'] = hourly_enhanced['total_volume'] * np.sign(hourly_enhanced['return_1h'])
    hourly_enhanced['volume_direction_2h'] = hourly_enhanced['volume_direction'].rolling(2, min_periods=1).sum()
    hourly_enhanced['volume_direction_4h'] = hourly_enhanced['volume_direction'].rolling(4, min_periods=1).sum()

    # Proxy for spread: range relative to volume (high range + low volume = wide spread)
    hourly_enhanced['range_per_volume'] = hourly_enhanced['range_ticks'] / hourly_enhanced['total_volume'].clip(lower=1) * 1000

    # Bar structure features
    hourly_enhanced['body_ratio'] = abs(hourly_enhanced['return_1h']) / (hourly_enhanced['range_ticks'] / hourly_enhanced['close'].clip(lower=1)).clip(lower=1e-6)
    hourly_enhanced['upper_wick'] = (hourly_enhanced['high'] - hourly_enhanced[['open', 'close']].max(axis=1)) / hourly_enhanced['range_ticks'].clip(lower=1)
    hourly_enhanced['lower_wick'] = (hourly_enhanced[['open', 'close']].min(axis=1) - hourly_enhanced['low']) / hourly_enhanced['range_ticks'].clip(lower=1)

    # Volume momentum
    hourly_enhanced['volume_momentum_2h'] = hourly_enhanced['total_volume'].pct_change(2)
    hourly_enhanced['volume_momentum_4h'] = hourly_enhanced['total_volume'].pct_change(4)

    # Volatility structure
    hourly_enhanced['vol_of_vol'] = hourly_enhanced['realized_vol'].rolling(4, min_periods=2).std()
    hourly_enhanced['vol_skew'] = hourly_enhanced['return_std'].rolling(4, min_periods=2).apply(
        lambda x: stats.skew(x) if len(x) > 2 else 0, raw=True)

    enhanced_features = get_feature_cols(hourly_enhanced, exclude_mbo=True)
    results['enhanced_ohlcv'] = train_walk_forward(hourly_enhanced, enhanced_features, label='ENHANCED OHLCV')

    # Summary
    log.info("\n" + "="*60)
    log.info("ABLATION STUDY SUMMARY")
    log.info("="*60)

    summary = {
        'generated': datetime.now().isoformat(),
        'question': 'Can ES 2h model work without MBO data?',
        'methodology': 'Fixed WF: 80/20 early stopping, 5-day purge, 60d train, sliding',
        'variants': results,
        'conclusion': '',
    }

    if results['ohlcv_only'] and results['all_features']:
        all_ic = results['all_features']['ic_unconditional']
        ohlcv_ic = results['ohlcv_only']['ic_unconditional']
        enhanced_ic = results['enhanced_ohlcv']['ic_unconditional'] if results['enhanced_ohlcv'] else 0

        ic_retention = ohlcv_ic / all_ic * 100 if all_ic > 0 else 0
        enhanced_retention = enhanced_ic / all_ic * 100 if all_ic > 0 else 0

        summary['ic_retention_ohlcv_pct'] = round(ic_retention, 1)
        summary['ic_retention_enhanced_pct'] = round(enhanced_retention, 1)

        log.info(f"ALL features:     IC={all_ic:.4f}, Sharpe={results['all_features']['sharpe']}")
        log.info(f"OHLCV only:       IC={ohlcv_ic:.4f}, Sharpe={results['ohlcv_only']['sharpe']} ({ic_retention:.0f}% retained)")
        log.info(f"Enhanced OHLCV:   IC={enhanced_ic:.4f}, Sharpe={results['enhanced_ohlcv']['sharpe'] if results['enhanced_ohlcv'] else 'N/A'} ({enhanced_retention:.0f}% retained)")

        if ohlcv_ic >= 0.15 or enhanced_ic >= 0.15:
            summary['conclusion'] = f'DEPLOYABLE without MBO. OHLCV IC={ohlcv_ic:.3f}, Enhanced IC={enhanced_ic:.3f}. Can run without Razer.'
            log.info(f"\n*** RESULT: Model retains useful edge WITHOUT MBO! ***")
        elif ohlcv_ic >= 0.05:
            summary['conclusion'] = f'MARGINAL without MBO. IC={ohlcv_ic:.3f} — weak but nonzero. MBO strongly recommended.'
            log.info(f"\n*** RESULT: Marginal edge without MBO. Razer still needed for production. ***")
        else:
            summary['conclusion'] = f'MBO DEPENDENT. OHLCV IC={ohlcv_ic:.3f} — near zero. Model is worthless without MBO. Razer is essential.'
            log.info(f"\n*** RESULT: Model is MBO-dependent. Cannot deploy without Razer. ***")

    # Save
    out_path = OUTPUT_DIR / "mbo_ablation_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    log.info(f"Saved results to {out_path}")

    return summary


if __name__ == '__main__':
    main()
