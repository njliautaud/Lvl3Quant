#!/usr/bin/env python3
"""
ES 2h Model — MBO Feature Ablation v2 (HC #662 R4)
===================================================

Uses the PAPER ENGINE's exact feature pipeline and LightGBM params
for a fair comparison. v1 used the original script's simpler features
and weaker regularization, giving misleadingly low IC.

Tests: ALL features vs OHLCV-only using paper engine methodology.

Author: Claude (HC #662 R4 — reduce MBO dependency)
"""

import sys, json, gc, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from datetime import datetime

warnings.filterwarnings('ignore')

ROOT = Path("/home/jupiter/Lvl3Quant")
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "live_trading_linux"))

# Import paper engine's EXACT infrastructure
from lh_2h_paper_engine import (
    compute_enhanced_hourly, add_rolling_features, add_regime_context,
    get_feature_cols, TRAIN_DAYS, PURGE_DAYS, HORIZON_BARS, LGBM_PARAMS,
    MINUTE_BAR_DIR, LH2hPaperEngine, _safe_polyfit_slope
)
import lightgbm as lgb

OUTPUT_DIR = ROOT / "output" / "lh_2h_mbo_ablation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# MBO-derived features to exclude in OHLCV-only mode
MBO_FEATURES = {
    # OFI-derived
    'ofi_sum', 'ofi_mean', 'ofi_trend', 'ofi_consistency', 'ofi_late_vs_early',
    'ofi_acceleration', 'ofi_curvature', 'ofi_vol', 'ofi_vol_normalized',
    'ofi_sum_2h', 'ofi_sum_4h', 'ofi_sum_6h',
    'ofi_trend_2h', 'ofi_trend_4h', 'ofi_trend_6h',
    # Signed volume-derived
    'signed_volume_sum', 'signed_volume_ratio', 'buy_volume_fraction',
    'sell_volume_fraction', 'sweep_minutes',
    'sv_sum_2h', 'sv_sum_4h', 'sv_sum_6h',
    # Spread-derived
    'spread_mean', 'spread_max',
    # Trade count-derived
    'trade_count_sum', 'trade_intensity', 'avg_trade_size',
    # VWAP-derived
    'vwap_dev_final', 'vwap_dev_trend',
    # Microprice-derived
    'microprice_dev_mean', 'microprice_dev_trend', 'microprice_dev_late',
    # Rolling features derived from MBO base features
    'mpdev_sum_2h', 'mpdev_sum_4h', 'mpdev_sum_6h',       # rolling microprice dev
    'vwapdev_sum_2h', 'vwapdev_sum_4h', 'vwapdev_sum_6h', # rolling vwap dev
    'ofi_accel_2h', 'ofi_accel_4h', 'ofi_accel_6h',       # rolling OFI acceleration
}

COST_PASSIVE_RT = 0.376


def walk_forward(hourly_df, feature_cols, label='full'):
    """Paper engine walk-forward with proper methodology."""
    dates = sorted(hourly_df['date'].unique())
    print(f"[{label}] WF: {len(dates)} dates, {len(feature_cols)} features")

    all_preds, all_actuals, all_dates_oot = [], [], []
    importances = np.zeros(len(feature_cols))
    n_folds = 0

    min_start = TRAIN_DAYS + PURGE_DAYS

    for day_idx in range(min_start, len(dates)):
        oot_date = dates[day_idx]
        train_dates = dates[day_idx - TRAIN_DAYS - PURGE_DAYS : day_idx - PURGE_DAYS]

        train_mask = hourly_df['date'].isin(train_dates)
        oot_mask = hourly_df['date'] == oot_date

        train_df = hourly_df[train_mask]
        oot_df = hourly_df[oot_mask]

        if len(train_df) < 100 or len(oot_df) == 0:
            continue

        X_train = train_df[feature_cols].fillna(0).values.astype(np.float32)
        y_train = train_df['fwd_ticks'].values.astype(np.float32)
        X_oot = oot_df[feature_cols].fillna(0).values.astype(np.float32)
        y_oot = oot_df['fwd_ticks'].values.astype(np.float32)

        valid = ~np.isnan(y_train)
        valid_oot = ~np.isnan(y_oot)
        if valid.sum() < 50 or valid_oot.sum() == 0:
            continue

        X_train, y_train = X_train[valid], y_train[valid]
        X_oot, y_oot = X_oot[valid_oot], y_oot[valid_oot]

        # 80/20 train split for early stopping (NOT test set)
        split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:split], X_train[split:]
        y_tr, y_val = y_train[:split], y_train[split:]

        params = {**LGBM_PARAMS, 'seed': 42, 'verbosity': -1}

        if len(X_val) >= 10:
            model = lgb.LGBMRegressor(**params, early_stopping_rounds=50)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.log_evaluation(0)])
        else:
            model = lgb.LGBMRegressor(**params, n_estimators=100)
            model.fit(X_train, y_train)

        preds = model.predict(X_oot)
        all_preds.extend(preds)
        all_actuals.extend(y_oot)
        all_dates_oot.extend([oot_date] * len(y_oot))

        try:
            importances += model.feature_importances_
        except:
            pass
        n_folds += 1

    if n_folds == 0:
        print(f"[{label}] ZERO folds!")
        return None

    preds = np.array(all_preds)
    actuals = np.array(all_actuals)
    dates_arr = np.array(all_dates_oot)

    # ICs
    pearson_ic = float(np.corrcoef(preds, actuals)[0, 1]) if len(preds) > 10 else 0
    spearman_ic = float(stats.spearmanr(preds, actuals)[0]) if len(preds) > 10 else 0

    # Top-20% selection
    abs_preds = np.abs(preds)
    top20_mask = abs_preds >= np.percentile(abs_preds, 80)
    ic_top20 = float(stats.spearmanr(preds[top20_mask], actuals[top20_mask])[0]) if top20_mask.sum() > 10 else 0

    # Trade sim (all predictions, matching permutation test)
    directions = np.sign(preds)
    gross = directions * actuals
    net = gross - COST_PASSIVE_RT

    # Per-day PnL
    unique_dates = sorted(set(dates_arr))
    daily_pnl = []
    for d in unique_dates:
        mask = dates_arr == d
        daily_pnl.append(net[mask].sum())
    daily_pnl = np.array(daily_pnl)

    sharpe_all = (daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if daily_pnl.std() > 0 else 0
    wr_all = (net > 0).mean() * 100

    # Top-20% trade sim
    t20_dirs = np.sign(preds[top20_mask])
    t20_gross = t20_dirs * actuals[top20_mask]
    t20_net = t20_gross - COST_PASSIVE_RT
    t20_dates = dates_arr[top20_mask]

    t20_unique = sorted(set(t20_dates))
    t20_daily = []
    for d in t20_unique:
        mask = t20_dates == d
        t20_daily.append(t20_net[mask].sum())
    t20_daily = np.array(t20_daily)
    t20_sharpe = (t20_daily.mean() / t20_daily.std() * np.sqrt(252)) if t20_daily.std() > 0 else 0

    # Feature importance
    imp = pd.DataFrame({
        'feature': feature_cols,
        'importance': importances / max(n_folds, 1)
    }).sort_values('importance', ascending=False)

    results = {
        'label': label,
        'n_features': len(feature_cols),
        'n_folds': n_folds,
        'n_predictions': len(preds),
        'pearson_ic': round(pearson_ic, 4),
        'spearman_ic': round(spearman_ic, 4),
        'spearman_ic_top20': round(ic_top20, 4),
        'sharpe_all_trades': round(float(sharpe_all), 2),
        'wr_all_pct': round(float(wr_all), 1),
        'avg_net_ticks_all': round(float(net.mean()), 2),
        'total_net_ticks_all': round(float(net.sum()), 1),
        'n_trades_top20': int(top20_mask.sum()),
        'sharpe_top20': round(float(t20_sharpe), 2),
        'avg_net_ticks_top20': round(float(t20_net.mean()), 2),
        'green_days': int((daily_pnl > 0).sum()),
        'red_days': int((daily_pnl < 0).sum()),
        'top5_features': imp.head(5)[['feature', 'importance']].values.tolist(),
    }

    print(f"[{label}] Pearson IC={pearson_ic:.4f}, Spearman IC={spearman_ic:.4f}, "
          f"IC_top20={ic_top20:.4f}, Sharpe(all)={sharpe_all:.1f}, "
          f"Sharpe(top20)={t20_sharpe:.1f}, WR={wr_all:.1f}%, "
          f"{n_folds} folds, {len(preds)} preds")

    return results


def main():
    print("="*60)
    print("ES 2h Model — MBO Ablation v2 (Paper Engine Pipeline)")
    print("="*60)

    # Load data using paper engine's loader
    engine = LH2hPaperEngine()
    print("Loading minute bar data...")
    minute_df = engine._load_minute_bars(n_days=999)
    if minute_df.empty:
        sys.exit("No data loaded")
    print(f"Loaded {len(minute_df)} minute bars")

    # Build features using paper engine's EXACT pipeline
    hourly = engine._build_hourly_df(minute_df)
    del minute_df; gc.collect()
    print(f"Built {len(hourly)} hourly bars with {len(hourly.columns)} cols")

    # Add forward labels
    hourly = hourly.sort_values('ts').reset_index(drop=True)
    hourly['fwd_ticks'] = hourly['close'].shift(-HORIZON_BARS) - hourly['close']

    # Null overnight gaps
    for i in range(len(hourly) - HORIZON_BARS):
        ts_now = hourly['ts'].iloc[i]
        ts_fwd = hourly['ts'].iloc[i + HORIZON_BARS]
        diff_s = (ts_fwd - ts_now).total_seconds()
        if diff_s > 8 * 3600:
            hourly.loc[hourly.index[i], 'fwd_ticks'] = np.nan

    hourly_clean = hourly[~hourly['hour'].isin([19, 20])].copy()
    hourly_clean = hourly_clean.dropna(subset=['fwd_ticks'])
    dates = sorted(hourly_clean['date'].unique())
    print(f"Clean: {len(hourly_clean)} bars, {len(dates)} days")

    # Get ALL features
    all_features = get_feature_cols(hourly_clean)
    print(f"ALL features ({len(all_features)}): {all_features}")

    # OHLCV-only features (exclude MBO-derived)
    ohlcv_features = [f for f in all_features if f not in MBO_FEATURES]
    mbo_removed = [f for f in all_features if f in MBO_FEATURES]
    print(f"\nOHLCV features ({len(ohlcv_features)}): {ohlcv_features}")
    print(f"MBO features removed ({len(mbo_removed)}): {mbo_removed}")

    results = {}

    # Variant 1: ALL features (baseline — should match permutation test)
    print("\n" + "="*60)
    print("VARIANT 1: ALL FEATURES (baseline)")
    print("="*60)
    results['all_features'] = walk_forward(hourly_clean, all_features, 'ALL')

    # Variant 2: OHLCV only
    print("\n" + "="*60)
    print("VARIANT 2: OHLCV ONLY (no MBO)")
    print("="*60)
    results['ohlcv_only'] = walk_forward(hourly_clean, ohlcv_features, 'OHLCV')

    # Summary
    print("\n" + "="*60)
    print("ABLATION SUMMARY")
    print("="*60)

    summary = {
        'generated': datetime.now().isoformat(),
        'methodology': 'Paper engine pipeline, 80/20 early stop, 5-day purge, 60d train, sliding WF',
        'question': 'How much IC does the 2h model lose without MBO features?',
        'variants': results,
    }

    if results['all_features'] and results['ohlcv_only']:
        all_ic = results['all_features']['spearman_ic']
        ohlcv_ic = results['ohlcv_only']['spearman_ic']
        retention = ohlcv_ic / all_ic * 100 if all_ic > 0 else 0

        print(f"\nALL:   Spearman IC={all_ic:.4f}, Sharpe={results['all_features']['sharpe_all_trades']}")
        print(f"OHLCV: Spearman IC={ohlcv_ic:.4f}, Sharpe={results['ohlcv_only']['sharpe_all_trades']} ({retention:.0f}% IC retained)")

        summary['ic_retention_pct'] = round(retention, 1)

        if ohlcv_ic >= 0.20:
            verdict = f'DEPLOYABLE without MBO. OHLCV Spearman IC={ohlcv_ic:.3f}.'
        elif ohlcv_ic >= 0.10:
            verdict = f'MARGINAL without MBO. IC={ohlcv_ic:.3f}. Could work with top-confidence filtering only.'
        elif ohlcv_ic >= 0.05:
            verdict = f'WEAK without MBO. IC={ohlcv_ic:.3f}. Not worth deploying alone.'
        else:
            verdict = f'MBO DEPENDENT. IC={ohlcv_ic:.3f}. Cannot deploy without Razer.'

        summary['verdict'] = verdict
        print(f"\nVERDICT: {verdict}")

    out_path = OUTPUT_DIR / "mbo_ablation_v2_results.json"
    with open(out_path, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSaved: {out_path}")


if __name__ == '__main__':
    main()
