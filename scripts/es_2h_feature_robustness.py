#!/usr/bin/env python3
"""
ES 2h LGBM Feature Robustness Analysis
========================================

Problem: The 2h LGBM model uses 80 features, 42 of which require MBO
microstructure data (microprice, OFI, trade count, signed volume, spread).
When Razer (MBO source) is offline, those features are unavailable.

This script:
1. Categorizes all 80 features into MBO-dependent vs freely-available
2. Extracts LGBM feature importances and measures MBO dependency
3. Retrains LGBM with OHLCV-only features (ablation study)
4. Engineers new proxy features from price/volume data
5. Tests a "degraded mode" model and an "enhanced OHLCV" model
6. Finds the minimum viable feature set

Author: Claude (Feature Robustness Analysis)
"""

import gc
import json
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

warnings.filterwarnings('ignore')

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from lh_2h_enhanced_ic_push import (
    load_minute_bars,
    compute_enhanced_hourly,
    add_rolling_features,
    add_regime_context,
    get_feature_cols,
    train_lgbm,
    TRAIN_DAYS,
    HORIZON_BARS,
)
from lh_2h_intraday_clean import add_intraday_forward_labels

OUTPUT_DIR = ROOT / "output" / "es_2h_feature_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / "es_2h_feature_robustness.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    format='%(asctime)s [ROBUST] %(levelname)s %(message)s',
    level=logging.INFO,
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(str(LOG_FILE)),
    ],
)
log = logging.getLogger('ROBUST')

# Constants
PURGE_DAYS = 5
ES_TICK_VALUE = 12.50
COST_MARKET_RT_TICKS = 1.376

# =============================================================================
# FEATURE CATEGORIZATION
# =============================================================================

# These features REQUIRE MBO data (microprice, OFI, signed_volume, spread, trade_count)
MBO_SINGLE_BAR = {
    'ofi_sum', 'ofi_mean', 'ofi_trend', 'ofi_consistency', 'ofi_late_vs_early',
    'signed_volume_sum', 'signed_volume_ratio', 'buy_volume_fraction', 'sell_volume_fraction',
    'sweep_minutes', 'spread_mean', 'spread_max', 'trade_count_sum', 'trade_intensity',
    'microprice_dev_mean', 'microprice_dev_trend', 'microprice_dev_late',
    'vwap_dev_final', 'vwap_dev_trend',
    'avg_trade_size', 'volume_top_half_ratio',
    'ofi_acceleration', 'ofi_curvature',
    'ofi_vol', 'ofi_vol_normalized',
}

MBO_ROLLING = {
    'ofi_sum_2h', 'ofi_trend_2h', 'sv_sum_2h',
    'ofi_sum_4h', 'ofi_trend_4h', 'sv_sum_4h',
    'ofi_sum_6h', 'ofi_trend_6h', 'sv_sum_6h',
    'mpdev_sum_2h', 'vwapdev_sum_2h', 'ofi_accel_2h',
    'mpdev_sum_4h', 'vwapdev_sum_4h', 'ofi_accel_4h',
    'mpdev_sum_6h', 'vwapdev_sum_6h', 'ofi_accel_6h',
}

ALL_MBO_FEATURES = MBO_SINGLE_BAR | MBO_ROLLING


def categorize_features(feature_cols):
    """Split features into MBO-dependent and freely-available."""
    mbo_feats = [f for f in feature_cols if f in ALL_MBO_FEATURES]
    free_feats = [f for f in feature_cols if f not in ALL_MBO_FEATURES]
    return mbo_feats, free_feats


# =============================================================================
# ENHANCED OHLCV FEATURES (proxy for microstructure)
# =============================================================================

def add_enhanced_ohlcv_features(df):
    """
    Engineer features from OHLCV data that proxy microstructure signals.
    These can be computed from yfinance or any OHLCV source.
    """
    # --- Price-based proxies for order flow ---

    # 1. Close-to-VWAP proxy: use (close - typical_price) as VWAP deviation proxy
    #    typical_price = (H+L+C)/3, which approximates VWAP without tick data
    df['typical_price'] = (df['high'] + df['low'] + df['close']) / 3
    df['close_vs_typical'] = (df['close'] - df['typical_price']) / df['typical_price'].clip(lower=1)

    # 2. Buying pressure proxy: close position in high-low range
    #    If close is near high => buying pressure. Near low => selling.
    rng = (df['high'] - df['low']).clip(lower=0.25)
    df['buying_pressure'] = (df['close'] - df['low']) / rng

    # 3. Price acceleration: 2nd derivative of price
    df['price_accel_2h'] = df['close'].pct_change(2) - df['close'].pct_change(1)
    df['price_accel_4h'] = df['close'].pct_change(4) - df['close'].pct_change(2)

    # 4. Body-to-range ratio: measures conviction
    df['body_ratio'] = abs(df['close'] - df['open']) / rng
    df['upper_wick_ratio'] = (df['high'] - df[['close', 'open']].max(axis=1)) / rng
    df['lower_wick_ratio'] = (df[['close', 'open']].min(axis=1) - df['low']) / rng

    # 5. Volume-weighted momentum (uses volume but not MBO)
    vol_safe = df['total_volume'].clip(lower=1)
    df['volume_momentum'] = df['return_1h'] * vol_safe / vol_safe.rolling(6, min_periods=1).mean()

    # 6. Volume surge detection (proxy for sweep_minutes / trade_intensity)
    vol_ma = df['total_volume'].rolling(10, min_periods=1).mean()
    vol_std = df['total_volume'].rolling(10, min_periods=1).std().clip(lower=1)
    df['volume_zscore'] = (df['total_volume'] - vol_ma) / vol_std
    df['volume_surge'] = (df['volume_zscore'] > 2.0).astype(float)

    # 7. Range expansion/contraction (proxy for volatility of order flow)
    df['range_vs_avg'] = rng / rng.rolling(10, min_periods=1).mean().clip(lower=0.25)
    df['range_expansion'] = rng / rng.shift(1).clip(lower=0.25)

    # 8. Gap features (open vs prior close)
    df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1).clip(lower=1)

    # 9. Intrabar volatility proxy: range relative to absolute return
    abs_ret = abs(df['close'] - df['open']).clip(lower=0.25)
    df['noise_ratio'] = rng / abs_ret  # High = lots of intrabar noise

    # 10. Rolling buying pressure
    for w in [2, 4, 6]:
        df[f'buying_pressure_{w}h'] = df['buying_pressure'].rolling(w, min_periods=1).mean()
        df[f'close_vs_typical_{w}h'] = df['close_vs_typical'].rolling(w, min_periods=1).sum()
        df[f'body_ratio_{w}h'] = df['body_ratio'].rolling(w, min_periods=1).mean()

    # 11. Price consistency (autocorrelation of returns - proxy for trend/mean-reversion)
    df['ret_consistency_3h'] = df['return_1h'].rolling(3, min_periods=2).apply(
        lambda x: np.mean(np.sign(x) == np.sign(x.iloc[-1])) if len(x) > 1 else 0.5, raw=False)
    df['ret_consistency_6h'] = df['return_1h'].rolling(6, min_periods=2).apply(
        lambda x: np.mean(np.sign(x) == np.sign(x.iloc[-1])) if len(x) > 1 else 0.5, raw=False)

    # 12. Volume-price correlation (rolling)
    for w in [4, 8]:
        df[f'vol_price_corr_{w}h'] = df['return_1h'].rolling(w, min_periods=3).corr(
            df['total_volume'].rolling(w, min_periods=3).mean())

    # 13. Amihud illiquidity (|return| / volume)
    df['amihud'] = abs(df['return_1h']) / df['total_volume'].clip(lower=1)
    df['amihud_4h'] = df['amihud'].rolling(4, min_periods=1).mean()

    # 14. High-low range momentum
    df['hl_range_mom_4h'] = rng.rolling(4, min_periods=1).apply(
        lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if len(x) > 1 else 0, raw=True)

    # Fill NaN from rolling
    df = df.fillna(0)
    return df


def get_enhanced_ohlcv_feature_cols(df):
    """Get all OHLCV-based feature columns (free + new proxies)."""
    exclude = {'date', 'hour', 'ts', 'open', 'high', 'low', 'close', 'fwd_ticks',
               'typical_price'}  # typical_price is intermediate, not a feature
    all_cols = [c for c in df.columns if c not in exclude and df[c].dtype in ['float64', 'float32', 'int64', 'int32']]
    # Exclude MBO features
    return [c for c in all_cols if c not in ALL_MBO_FEATURES]


# =============================================================================
# WALK-FORWARD ENGINE
# =============================================================================

def run_walkforward(hourly, feature_cols, purge_days=PURGE_DAYS, shuffle_labels=False):
    """Sliding walk-forward with LGBM."""
    dates = sorted(hourly['date'].unique())

    all_preds, all_actuals, all_dates, all_hours = [], [], [], []

    for i in range(TRAIN_DAYS + purge_days, len(dates)):
        oot_date = dates[i]
        train_end_idx = i - purge_days
        train_start_idx = max(0, train_end_idx - TRAIN_DAYS)
        train_dates = dates[train_start_idx:train_end_idx]

        train = hourly[hourly['date'].isin(train_dates)]
        oot = hourly[hourly['date'] == oot_date]

        if len(train) < 100 or len(oot) == 0:
            continue

        X_train = train[feature_cols].fillna(0).values
        y_train = train['fwd_ticks'].values
        X_oot = oot[feature_cols].fillna(0).values
        y_oot = oot['fwd_ticks'].values

        if shuffle_labels:
            y_train = np.random.permutation(y_train)

        try:
            split = int(len(X_train) * 0.8)
            model = train_lgbm(X_train[:split], y_train[:split],
                               X_train[split:], y_train[split:])
            preds = model.predict(X_oot)
            all_preds.extend(preds)
            all_actuals.extend(y_oot)
            all_dates.extend([oot_date] * len(y_oot))
            all_hours.extend(oot['hour'].values.tolist())
        except Exception as e:
            log.warning(f"Fold {oot_date} failed: {e}")
            continue

    preds = np.array(all_preds)
    actuals = np.array(all_actuals)
    dates_arr = np.array(all_dates)

    if len(preds) < 50:
        return 0.0, preds, actuals, dates_arr

    ic = float(stats.spearmanr(preds, actuals)[0])
    return ic, preds, actuals, dates_arr


def compute_metrics(preds, actuals, dates_arr, cost_ticks=COST_MARKET_RT_TICKS):
    """Compute comprehensive trade metrics."""
    directions = np.sign(preds)
    gross_ticks = directions * actuals
    net_ticks = gross_ticks - cost_ticks

    n_trades = len(net_ticks)
    if n_trades == 0:
        return {}

    wr = float(np.mean(net_ticks > 0))
    avg_net = float(net_ticks.mean())
    total_net = float(net_ticks.sum())

    # Profit factor
    wins = net_ticks[net_ticks > 0].sum()
    losses = abs(net_ticks[net_ticks < 0].sum())
    pf = float(wins / max(losses, 1e-9))

    # Daily PnL for Sharpe/Sortino
    unique_dates = sorted(set(dates_arr))
    daily_pnl = np.array([net_ticks[dates_arr == d].sum() for d in unique_dates])

    if daily_pnl.std() > 0:
        sharpe = float((daily_pnl.mean() / daily_pnl.std()) * np.sqrt(252))
    else:
        sharpe = 0.0

    downside = daily_pnl[daily_pnl < 0]
    if len(downside) > 0:
        down_std = np.sqrt(np.mean(downside**2))
        sortino = float((daily_pnl.mean() / down_std) * np.sqrt(252)) if down_std > 0 else 0.0
    else:
        sortino = float('inf') if daily_pnl.mean() > 0 else 0.0

    # Directional accuracy
    dir_acc = float(np.mean(np.sign(preds) == np.sign(actuals)))

    return {
        'n_trades': n_trades,
        'n_days': len(daily_pnl),
        'win_rate': wr,
        'directional_accuracy': dir_acc,
        'avg_net_ticks': avg_net,
        'total_net_ticks': total_net,
        'total_pnl_usd': float(total_net * ES_TICK_VALUE),
        'profit_factor': pf,
        'sharpe': sharpe,
        'sortino': sortino,
        'daily_pnl_mean': float(daily_pnl.mean()),
        'daily_pnl_std': float(daily_pnl.std()),
    }


# =============================================================================
# FEATURE IMPORTANCE ANALYSIS
# =============================================================================

def extract_feature_importance(hourly, feature_cols):
    """Fit single model on all data, extract feature importances."""
    import lightgbm as lgb

    X = hourly[feature_cols].fillna(0).values
    y = hourly['fwd_ticks'].values

    model = lgb.LGBMRegressor(
        num_leaves=15, max_depth=4, n_estimators=200,
        feature_fraction=0.5, min_child_samples=50,
        lambda_l1=1.0, lambda_l2=5.0, verbosity=-1
    )
    model.fit(X, y)

    importances = model.feature_importances_
    fi = sorted(zip(feature_cols, importances.tolist()), key=lambda x: -x[1])
    return fi, model


def analyze_importance_by_category(fi, mbo_feats, free_feats):
    """Analyze what fraction of importance comes from MBO vs free features."""
    total_imp = sum(imp for _, imp in fi)
    mbo_imp = sum(imp for name, imp in fi if name in ALL_MBO_FEATURES)
    free_imp = sum(imp for name, imp in fi if name not in ALL_MBO_FEATURES)

    mbo_pct = mbo_imp / max(total_imp, 1) * 100
    free_pct = free_imp / max(total_imp, 1) * 100

    # Top features by category
    mbo_ranked = [(n, i) for n, i in fi if n in ALL_MBO_FEATURES]
    free_ranked = [(n, i) for n, i in fi if n not in ALL_MBO_FEATURES]

    return {
        'total_importance': total_imp,
        'mbo_importance': mbo_imp,
        'mbo_pct': round(mbo_pct, 1),
        'free_importance': free_imp,
        'free_pct': round(free_pct, 1),
        'n_mbo_features': len(mbo_feats),
        'n_free_features': len(free_feats),
        'top10_mbo': [(n, int(i)) for n, i in mbo_ranked[:10]],
        'top10_free': [(n, int(i)) for n, i in free_ranked[:10]],
    }


# =============================================================================
# GREEDY FORWARD SELECTION
# =============================================================================

def greedy_forward_select(hourly, candidate_features, max_features=15, n_eval_folds=20):
    """
    Greedy forward feature selection: start with best single feature,
    keep adding the one that improves IC the most.
    Uses last n_eval_folds OOT days for fast evaluation.
    """
    dates = sorted(hourly['date'].unique())
    # Use a subset of folds for speed
    eval_start = max(TRAIN_DAYS + PURGE_DAYS, len(dates) - n_eval_folds)

    def quick_ic(feature_list):
        """Quick IC on last n_eval_folds days."""
        all_preds, all_actuals = [], []
        for i in range(eval_start, len(dates)):
            oot_date = dates[i]
            train_end = i - PURGE_DAYS
            train_start = max(0, train_end - TRAIN_DAYS)
            train_dates = dates[train_start:train_end]

            train = hourly[hourly['date'].isin(train_dates)]
            oot = hourly[hourly['date'] == oot_date]
            if len(train) < 100 or len(oot) == 0:
                continue

            X_tr = train[feature_list].fillna(0).values
            y_tr = train['fwd_ticks'].values
            X_oo = oot[feature_list].fillna(0).values
            y_oo = oot['fwd_ticks'].values

            try:
                split = int(len(X_tr) * 0.8)
                model = train_lgbm(X_tr[:split], y_tr[:split], X_tr[split:], y_tr[split:])
                p = model.predict(X_oo)
                all_preds.extend(p)
                all_actuals.extend(y_oo)
            except:
                continue

        if len(all_preds) < 20:
            return -1.0
        return float(stats.spearmanr(all_preds, all_actuals)[0])

    selected = []
    remaining = list(candidate_features)
    history = []

    for step in range(min(max_features, len(remaining))):
        best_feat = None
        best_ic = -1.0

        for feat in remaining:
            trial = selected + [feat]
            ic = quick_ic(trial)
            if ic > best_ic:
                best_ic = ic
                best_feat = feat

        if best_feat is None:
            break

        selected.append(best_feat)
        remaining.remove(best_feat)
        history.append({
            'step': step + 1,
            'added_feature': best_feat,
            'n_features': len(selected),
            'ic': best_ic,
        })
        log.info(f"  Step {step+1}: +{best_feat} -> IC={best_ic:.4f} ({len(selected)} features)")

        # Stop if IC hasn't improved in last 3 steps
        if len(history) >= 4:
            recent_ics = [h['ic'] for h in history[-4:]]
            if max(recent_ics) - min(recent_ics) < 0.005:
                log.info(f"  Plateau detected, stopping at {len(selected)} features")
                break

    return selected, history


# =============================================================================
# MAIN
# =============================================================================

def main():
    log.info("=" * 70)
    log.info("ES 2h LGBM Feature Robustness Analysis")
    log.info("=" * 70)
    t_start = time.time()

    # ── Load data ──
    log.info("\n[1/6] Loading minute bars and computing features...")
    minutes = load_minute_bars()
    hourly = compute_enhanced_hourly(minutes)
    hourly = add_rolling_features(hourly)
    hourly = add_regime_context(hourly)
    hourly = add_intraday_forward_labels(hourly, horizon_bars=HORIZON_BARS)

    # Add enhanced OHLCV features
    hourly = add_enhanced_ohlcv_features(hourly)

    del minutes
    gc.collect()

    all_features = get_feature_cols(hourly)
    mbo_feats, free_feats = categorize_features(all_features)
    enhanced_ohlcv_feats = get_enhanced_ohlcv_feature_cols(hourly)

    log.info(f"Total features (original): {len(all_features)}")
    log.info(f"  MBO-dependent: {len(mbo_feats)}")
    log.info(f"  Freely available: {len(free_feats)}")
    log.info(f"Enhanced OHLCV features (free + proxies): {len(enhanced_ohlcv_feats)}")
    log.info(f"Bars: {len(hourly)}, dates: {hourly['date'].nunique()}")

    results = {
        'timestamp': datetime.utcnow().isoformat(),
        'description': 'Feature robustness analysis for ES 2h LGBM model',
        'data': {
            'n_bars': len(hourly),
            'n_dates': int(hourly['date'].nunique()),
            'date_range': [hourly['date'].min(), hourly['date'].max()],
        },
        'feature_counts': {
            'total_original': len(all_features),
            'mbo_dependent': len(mbo_feats),
            'freely_available': len(free_feats),
            'enhanced_ohlcv': len(enhanced_ohlcv_feats),
        },
        'mbo_features': sorted(mbo_feats),
        'free_features': sorted(free_feats),
    }

    # ── Feature importance analysis ──
    log.info("\n[2/6] Feature importance analysis...")
    fi, _ = extract_feature_importance(hourly, all_features)
    importance_analysis = analyze_importance_by_category(fi, mbo_feats, free_feats)

    log.info(f"MBO importance: {importance_analysis['mbo_pct']:.1f}% ({importance_analysis['n_mbo_features']} features)")
    log.info(f"Free importance: {importance_analysis['free_pct']:.1f}% ({importance_analysis['n_free_features']} features)")
    log.info("\nTop 10 MBO features:")
    for name, imp in importance_analysis['top10_mbo']:
        log.info(f"  {name:35s} {imp:6d}")
    log.info("\nTop 10 Free features:")
    for name, imp in importance_analysis['top10_free']:
        log.info(f"  {name:35s} {imp:6d}")

    results['importance_analysis'] = importance_analysis
    results['full_feature_importance'] = [(n, int(i)) for n, i in fi]

    # ── Ablation study: 5 models ──
    log.info("\n[3/6] Ablation study — comparing feature sets...")

    ablation_configs = {
        'full_model': {
            'features': all_features,
            'description': 'All 80 original features (baseline)',
        },
        'mbo_only': {
            'features': mbo_feats,
            'description': 'MBO/microstructure features only (42)',
        },
        'free_only': {
            'features': free_feats,
            'description': 'Freely available features only, no proxies (38)',
        },
        'enhanced_ohlcv': {
            'features': enhanced_ohlcv_feats,
            'description': 'Free features + engineered OHLCV proxies',
        },
    }

    ablation_results = {}
    for name, config in ablation_configs.items():
        feats = config['features']
        # Filter to features that actually exist in the dataframe
        feats = [f for f in feats if f in hourly.columns]
        if len(feats) == 0:
            log.warning(f"  {name}: no valid features, skipping")
            continue

        log.info(f"\n  --- {name}: {len(feats)} features ---")
        log.info(f"  {config['description']}")

        ic, preds, actuals, dates_arr = run_walkforward(hourly, feats)
        metrics = compute_metrics(preds, actuals, dates_arr)

        log.info(f"  IC: {ic:.4f}")
        log.info(f"  Sharpe: {metrics.get('sharpe', 0):.2f}")
        log.info(f"  Sortino: {metrics.get('sortino', 0):.2f}")
        log.info(f"  WR: {metrics.get('win_rate', 0):.1%}")
        log.info(f"  PF: {metrics.get('profit_factor', 0):.2f}")
        log.info(f"  Avg net: {metrics.get('avg_net_ticks', 0):.3f} ticks")

        ablation_results[name] = {
            'description': config['description'],
            'n_features': len(feats),
            'features': feats,
            'ic': ic,
            **metrics,
        }
        gc.collect()

    results['ablation'] = {k: {kk: vv for kk, vv in v.items() if kk != 'features'}
                           for k, v in ablation_results.items()}
    # Store features separately to keep the summary clean
    results['ablation_features'] = {k: v['features'] for k, v in ablation_results.items()}

    # ── Permutation test for enhanced OHLCV model ──
    log.info("\n[4/6] Permutation test for enhanced OHLCV model (50 shuffles)...")
    enhanced_feats = [f for f in enhanced_ohlcv_feats if f in hourly.columns]
    shuf_ics = []
    for trial in range(50):
        shuf_ic, _, _, _ = run_walkforward(hourly, enhanced_feats, shuffle_labels=True)
        shuf_ics.append(shuf_ic)
        if (trial + 1) % 10 == 0:
            log.info(f"  Permutation {trial+1}/50: mean_shuf={np.mean(shuf_ics):.4f}")

    enhanced_ic = ablation_results.get('enhanced_ohlcv', {}).get('ic', 0)
    mean_shuf = np.mean(shuf_ics)
    genuine_ic = enhanced_ic - mean_shuf
    p_value = np.mean([s >= enhanced_ic for s in shuf_ics])

    results['enhanced_ohlcv_permutation'] = {
        'real_ic': enhanced_ic,
        'mean_shuffle_ic': float(mean_shuf),
        'std_shuffle_ic': float(np.std(shuf_ics)),
        'genuine_ic': float(genuine_ic),
        'p_value': float(p_value),
        'n_permutations': 50,
    }
    log.info(f"  Enhanced OHLCV genuine IC: {genuine_ic:.4f}, p={p_value:.3f}")

    # ── Greedy forward selection on free features ──
    log.info("\n[5/6] Greedy forward feature selection (OHLCV features only)...")
    selected_feats, selection_history = greedy_forward_select(
        hourly, enhanced_ohlcv_feats, max_features=15, n_eval_folds=30
    )
    results['feature_selection'] = {
        'selected_features': selected_feats,
        'selection_history': selection_history,
    }

    # Full walk-forward with selected features
    if len(selected_feats) > 0:
        log.info(f"\n  Running full walk-forward with {len(selected_feats)} selected features...")
        ic_sel, preds_sel, actuals_sel, dates_sel = run_walkforward(hourly, selected_feats)
        metrics_sel = compute_metrics(preds_sel, actuals_sel, dates_sel)
        results['minimum_viable_model'] = {
            'n_features': len(selected_feats),
            'features': selected_feats,
            'ic': ic_sel,
            **metrics_sel,
        }
        log.info(f"  Minimum viable: IC={ic_sel:.4f}, Sharpe={metrics_sel.get('sharpe',0):.2f}, "
                 f"WR={metrics_sel.get('win_rate',0):.1%}")

    # ── Summary & Recommendations ──
    log.info("\n[6/6] Summary & Recommendations")
    log.info("=" * 70)

    full_ic = ablation_results.get('full_model', {}).get('ic', 0)
    free_ic = ablation_results.get('free_only', {}).get('ic', 0)
    enhanced_ic_val = ablation_results.get('enhanced_ohlcv', {}).get('ic', 0)
    mbo_ic = ablation_results.get('mbo_only', {}).get('ic', 0)
    min_ic = results.get('minimum_viable_model', {}).get('ic', 0)

    full_sharpe = ablation_results.get('full_model', {}).get('sharpe', 0)
    free_sharpe = ablation_results.get('free_only', {}).get('sharpe', 0)
    enhanced_sharpe = ablation_results.get('enhanced_ohlcv', {}).get('sharpe', 0)
    min_sharpe = results.get('minimum_viable_model', {}).get('sharpe', 0)

    log.info(f"                     IC      Sharpe   Features")
    log.info(f"  Full model:       {full_ic:6.4f}   {full_sharpe:6.2f}     {len(all_features)}")
    log.info(f"  MBO only:         {mbo_ic:6.4f}   {ablation_results.get('mbo_only', {}).get('sharpe', 0):6.2f}     {len(mbo_feats)}")
    log.info(f"  Free only:        {free_ic:6.4f}   {free_sharpe:6.2f}     {len(free_feats)}")
    log.info(f"  Enhanced OHLCV:   {enhanced_ic_val:6.4f}   {enhanced_sharpe:6.2f}     {len(enhanced_ohlcv_feats)}")
    log.info(f"  Min viable:       {min_ic:6.4f}   {min_sharpe:6.2f}     {results.get('minimum_viable_model', {}).get('n_features', 0)}")

    ic_retention = enhanced_ic_val / max(full_ic, 1e-9) * 100
    sharpe_retention = enhanced_sharpe / max(full_sharpe, 1e-9) * 100

    recommendations = []

    if enhanced_ic_val > 0.10 and enhanced_sharpe > 1.0:
        recommendations.append(
            f"DEPLOYABLE: Enhanced OHLCV model retains {ic_retention:.0f}% of IC, "
            f"{sharpe_retention:.0f}% of Sharpe. Can run as degraded-mode backup."
        )
    elif enhanced_ic_val > 0.05:
        recommendations.append(
            f"MARGINAL: Enhanced OHLCV model retains {ic_retention:.0f}% of IC. "
            f"Edge exists but may not survive costs. Use with confidence filters."
        )
    else:
        recommendations.append(
            f"NOT VIABLE: Enhanced OHLCV model retains only {ic_retention:.0f}% of IC. "
            f"Microstructure data is critical; OHLCV alone cannot substitute."
        )

    if mbo_ic > free_ic * 3:
        recommendations.append(
            f"MICROSTRUCTURE DOMINANCE: MBO-only IC ({mbo_ic:.4f}) >> Free-only IC ({free_ic:.4f}). "
            f"The model's edge is fundamentally microstructure-driven."
        )
    elif mbo_ic > free_ic * 1.5:
        recommendations.append(
            f"MICROSTRUCTURE ADVANTAGE: MBO features carry significantly more signal "
            f"({mbo_ic:.4f} vs {free_ic:.4f}), but free features contribute."
        )

    if importance_analysis['mbo_pct'] > 70:
        recommendations.append(
            f"HIGH MBO DEPENDENCY: {importance_analysis['mbo_pct']:.0f}% of LGBM split importance "
            f"comes from MBO features. Priority: restore Razer MBO feed."
        )

    recommendations.append(
        f"MINIMUM VIABLE SET: {results.get('minimum_viable_model', {}).get('n_features', '?')} features "
        f"achieve IC={min_ic:.4f}. These should be the fallback feature set."
    )

    for i, rec in enumerate(recommendations):
        log.info(f"\n  [{i+1}] {rec}")

    results['recommendations'] = recommendations
    results['summary'] = {
        'ic_retention_enhanced_ohlcv_pct': round(ic_retention, 1),
        'sharpe_retention_enhanced_ohlcv_pct': round(sharpe_retention, 1),
        'mbo_importance_pct': importance_analysis['mbo_pct'],
        'genuine_ic_enhanced_ohlcv': float(genuine_ic),
        'p_value_enhanced_ohlcv': float(p_value),
        'full_model_ic': full_ic,
        'full_model_sharpe': full_sharpe,
        'enhanced_ohlcv_ic': enhanced_ic_val,
        'enhanced_ohlcv_sharpe': enhanced_sharpe,
        'min_viable_ic': min_ic,
        'min_viable_sharpe': min_sharpe,
    }

    elapsed = time.time() - t_start
    results['runtime_seconds'] = round(elapsed, 1)

    # Save results
    output_file = OUTPUT_DIR / "robustness_results.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {output_file}")
    log.info(f"Total runtime: {elapsed/60:.1f} minutes")

    return results


if __name__ == '__main__':
    main()
