#!/usr/bin/env python3
"""
queue_minute_features_v1.py — Queue Microstructure Features at Minute Level
============================================================================

Aggregates tick-level queue features to 1-minute bars, then adds them as
supplementary features to the champion LGBM entry model. Tests whether
queue microstructure improves entry signal IC.

Data:
  - 41 dates of queue_augmented_features (~45K per date at ~250ms)
  - 197 dates of mbo_minute_bars_v1
  - Only 41 dates have both → enough for a focused WF test

Walk-forward: 25d train, 1d OOT, drop oldest (SLIDING per HC #0)
Regime gate: HC #428

Output: IC comparison (baseline LGBM vs queue-enhanced LGBM)
"""
import os
import sys
import json
import logging
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import lightgbm as lgb
from scipy import stats as sp_stats

warnings.filterwarnings('ignore')

BASE = Path("/home/jupiter/Lvl3Quant")
QUEUE_DIR = BASE / "output" / "queue_augmented_features"
MINUTE_DIR = BASE / "data" / "processed" / "mbo_minute_bars_v1"
OUTPUT_DIR = BASE / "output" / "queue_minute_features_v1"
LOG_PATH = BASE / "logs" / "queue_minute_features_v1.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(BASE / "logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── Constants ──
ES_TICK_SIZE = 0.25
ES_TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
TRAIN_DAYS = 25

# Champion config
TP_TICKS = 25
SL_LONG_TICKS = 4
SL_SHORT_TICKS = 3
MAX_HOLD_MINUTES = 60
ENTRY_THRESHOLD = 0.05
DAILY_BIAS_MULT = 1.5

LGBM_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 31,
    'max_depth': 6,
    'min_data_in_leaf': 50,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1,
}


def aggregate_queue_to_minute(queue_df, date_str):
    """Aggregate tick-level queue features to 1-minute bars.

    For each 1-minute window, compute: mean, std, max, min, last
    of key queue features.
    """
    df = queue_df.copy()
    # Convert ts_ns to datetime
    df['ts'] = pd.to_datetime(df['ts_ns'], unit='ns')
    df['minute'] = df['ts'].dt.floor('1min')

    # Key features to aggregate
    agg_cols = [
        'ofi_1s', 'ofi_5s', 'ofi_10s',
        'top_imbalance', 'microprice_offset_ticks',
        'bid_qty_at_touch', 'ask_qty_at_touch',
        'bid_add_rate_1s', 'ask_add_rate_1s',
        'bid_cancel_rate_1s', 'ask_cancel_rate_1s',
        'bid_trade_rate_1s', 'ask_trade_rate_1s',
        'bid_level_age_s', 'ask_level_age_s',
    ]

    # Only use cols that exist
    agg_cols = [c for c in agg_cols if c in df.columns]

    agg_funcs = {}
    for col in agg_cols:
        agg_funcs[f'{col}_mean'] = (col, 'mean')
        agg_funcs[f'{col}_std'] = (col, 'std')
        agg_funcs[f'{col}_max'] = (col, 'max')
        agg_funcs[f'{col}_min'] = (col, 'min')
        agg_funcs[f'{col}_last'] = (col, 'last')

    # Count of ticks per minute (activity proxy)
    agg_funcs['tick_count'] = ('ts_ns', 'count')

    result = df.groupby('minute').agg(**agg_funcs).reset_index()

    # Derived aggregates
    # OFI range (volatility of order flow imbalance)
    if 'ofi_1s_max' in result.columns:
        result['ofi_1s_range'] = result['ofi_1s_max'] - result['ofi_1s_min']
    if 'ofi_5s_max' in result.columns:
        result['ofi_5s_range'] = result['ofi_5s_max'] - result['ofi_5s_min']

    # Queue pressure (end vs start of minute)
    if 'top_imbalance_last' in result.columns and 'top_imbalance_mean' in result.columns:
        result['imbalance_drift'] = result['top_imbalance_last'] - result['top_imbalance_mean']

    # Microprice stability
    if 'microprice_offset_ticks_std' in result.columns:
        result['microprice_stability'] = 1.0 / (result['microprice_offset_ticks_std'] + 0.01)

    return result


def load_minute_bar_features(date_str):
    """Load minute bar data and compute standard features."""
    mb_path = MINUTE_DIR / f"{date_str}.parquet"
    if not mb_path.exists():
        return None

    df = pd.read_parquet(mb_path)
    if len(df) < 30:
        return None

    # Ensure timestamp column
    if 'timestamp' in df.columns:
        df['minute'] = pd.to_datetime(df['timestamp']).dt.floor('1min')
    elif 'ts' in df.columns:
        df['minute'] = pd.to_datetime(df['ts']).dt.floor('1min')
    else:
        # Try index
        df['minute'] = pd.to_datetime(df.index).floor('1min')

    # Standard minute bar features
    if 'close' in df.columns:
        df['return_1m'] = df['close'].diff()
        df['return_5m'] = df['close'].diff(5)
        df['return_10m'] = df['close'].diff(10)
        df['return_30m'] = df['close'].diff(30)

        # Volatility
        df['vol_5m'] = df['return_1m'].rolling(5).std()
        df['vol_10m'] = df['return_1m'].rolling(10).std()
        df['vol_30m'] = df['return_1m'].rolling(30).std()

        # Momentum
        df['momentum_10m'] = df['return_10m'] / (df['vol_10m'] + 1e-8)
        df['momentum_30m'] = df['return_30m'] / (df['vol_30m'] + 1e-8)

        # VWAP deviation
        if 'volume' in df.columns:
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            df['vwap_dev'] = df['close'] - df['vwap']

    if 'high' in df.columns and 'low' in df.columns:
        df['range'] = df['high'] - df['low']
        df['range_5m'] = df['range'].rolling(5).mean()

    if 'volume' in df.columns:
        df['vol_ratio_5m'] = df['volume'] / (df['volume'].rolling(5).mean() + 1)

    # Regime features
    if 'close' in df.columns:
        df['daily_return_so_far'] = df['close'] - df['close'].iloc[0]

    # Vol regime (from minute bar data if available)
    if 'vol_regime' in df.columns:
        vol_map = {'low': 0, 'medium': 1, 'high': 2, 0: 0, 1: 1, 2: 2}
        df['vol_regime_num'] = df['vol_regime'].map(vol_map).fillna(1).astype(int)

    return df


def classify_day_regime(mb_df):
    """Classify day as green/red/flat."""
    if mb_df is None or 'close' not in mb_df.columns or len(mb_df) < 2:
        return 'unknown'
    day_return = mb_df['close'].iloc[-1] - mb_df['close'].iloc[0]
    if day_return > 0.5:  # > 0.5 points = green (~2 ticks)
        return 'green'
    elif day_return < -0.5:
        return 'red'
    else:
        return 'flat'


def simulate_champion_trades(mb_df, model, feature_cols, threshold=ENTRY_THRESHOLD):
    """Simulate champion config trades on minute bars with given LGBM model."""
    if mb_df is None or len(mb_df) < 60:
        return []

    X = mb_df[feature_cols].values.astype(np.float32)
    X = np.nan_to_num(X, nan=0.0, posinf=100.0, neginf=-100.0)
    preds = model.predict(X)

    trades = []
    in_trade = False
    entry_idx = None
    direction = 0

    for i in range(30, len(mb_df)):
        if in_trade:
            bars_held = i - entry_idx
            price_change = mb_df['close'].iloc[i] - mb_df['close'].iloc[entry_idx]
            ticks_change = price_change / ES_TICK_SIZE * direction

            sl_ticks = SL_LONG_TICKS if direction == 1 else SL_SHORT_TICKS

            exit_reason = None
            if ticks_change >= TP_TICKS:
                exit_reason = 'tp'
                raw_pnl = TP_TICKS
            elif ticks_change <= -sl_ticks:
                exit_reason = 'sl'
                raw_pnl = -sl_ticks
            elif bars_held >= MAX_HOLD_MINUTES:
                exit_reason = 'time'
                raw_pnl = ticks_change

            if exit_reason:
                net_pnl = raw_pnl - COMMISSION_RT_TICKS
                trades.append({
                    'entry_idx': entry_idx,
                    'exit_idx': i,
                    'direction': direction,
                    'raw_pnl': raw_pnl,
                    'net_pnl': net_pnl,
                    'exit_reason': exit_reason,
                    'hold_bars': bars_held,
                    'pred_strength': preds[entry_idx],
                })
                in_trade = False

        if not in_trade:
            pred = preds[i]
            if abs(pred) > threshold:
                direction = 1 if pred > 0 else -1
                entry_idx = i
                in_trade = True

    return trades


def compute_regime_gap(trades, regime_map, dates_col='date'):
    """Compute regime gap."""
    if not trades:
        return float('nan'), {}, {}

    df = pd.DataFrame(trades)
    if dates_col not in df.columns:
        return float('nan'), {}, {}

    df['regime'] = df[dates_col].map(regime_map)

    green = df[df['regime'] == 'green']
    red = df[df['regime'] == 'red']

    if len(green) < 3 or len(red) < 3:
        return float('nan'), {}, {}

    green_daily = green.groupby(dates_col)['net_pnl'].sum()
    red_daily = red.groupby(dates_col)['net_pnl'].sum()

    sharpe_g = green_daily.mean() / (green_daily.std() + 1e-8) * np.sqrt(252)
    sharpe_r = red_daily.mean() / (red_daily.std() + 1e-8) * np.sqrt(252)

    gap = abs(sharpe_g - sharpe_r) / max(abs(sharpe_g), abs(sharpe_r), 1e-8)

    return float(gap), {'sharpe': float(sharpe_g), 'n': len(green_daily)}, {'sharpe': float(sharpe_r), 'n': len(red_daily)}


def main():
    log.info("=" * 70)
    log.info("Queue Minute Features v1 — Champion LGBM Enhancement")
    log.info("=" * 70)

    # ── 1. Find dates with BOTH queue features AND minute bars ──
    queue_dates = sorted([f.replace('features_', '').replace('.parquet', '')
                          for f in os.listdir(QUEUE_DIR) if f.startswith('features_') and f.endswith('.parquet')])
    minute_dates = sorted([f.replace('.parquet', '')
                           for f in os.listdir(MINUTE_DIR) if f.endswith('.parquet')])

    overlap = sorted(set(queue_dates) & set(minute_dates))
    log.info(f"Queue dates: {len(queue_dates)}, Minute dates: {len(minute_dates)}, Overlap: {len(overlap)}")

    if len(overlap) < TRAIN_DAYS + 5:
        log.error(f"Need at least {TRAIN_DAYS+5} dates, only have {len(overlap)}")
        return

    # ── 2. Build merged dataset ──
    log.info("\nLoading and merging data...")
    all_days = {}
    date_regimes = {}

    for d in overlap:
        # Load minute bars
        mb = load_minute_bar_features(d)
        if mb is None:
            log.warning(f"  {d}: no minute bar data")
            continue

        # Load queue features and aggregate to minute
        qdf = pd.read_parquet(QUEUE_DIR / f"features_{d}.parquet")
        q_minute = aggregate_queue_to_minute(qdf, d)

        # Join on minute
        merged = mb.merge(q_minute, on='minute', how='left')

        # Fill NaN queue features for minutes without tick data
        q_cols = [c for c in merged.columns if c not in mb.columns and c != 'minute']
        for c in q_cols:
            merged[c] = merged[c].ffill().bfill().fillna(0)

        # Add target: next-30min return in ticks
        if 'close' in merged.columns:
            merged['target_30m'] = merged['close'].shift(-30) - merged['close']
            merged['target_30m'] = merged['target_30m'] / ES_TICK_SIZE  # convert to ticks

        date_regimes[d] = classify_day_regime(mb)

        all_days[d] = merged
        log.info(f"  {d}: {len(merged)} bars, {len(q_minute)} queue-minutes, "
                 f"{len(qdf)} ticks, regime={date_regimes[d]}")

    log.info(f"\nLoaded {len(all_days)} dates")
    log.info(f"Regimes: {pd.Series(date_regimes).value_counts().to_dict()}")

    # ── 3. Identify feature sets ──
    sample_df = list(all_days.values())[0]

    # Baseline features (minute bars only)
    baseline_candidates = [
        'return_1m', 'return_5m', 'return_10m', 'return_30m',
        'vol_5m', 'vol_10m', 'vol_30m',
        'momentum_10m', 'momentum_30m',
        'range', 'range_5m',
        'daily_return_so_far',
    ]
    if 'volume' in sample_df.columns:
        baseline_candidates.extend(['vol_ratio_5m', 'vwap_dev'])
    if 'vol_regime_num' in sample_df.columns:
        baseline_candidates.append('vol_regime_num')

    baseline_features = [c for c in baseline_candidates if c in sample_df.columns]

    # Queue-enhanced features (baseline + aggregated queue)
    queue_feature_names = [c for c in sample_df.columns
                           if any(c.startswith(p) for p in ['ofi_', 'top_imbalance_', 'microprice_',
                                                              'bid_qty_at_touch_', 'ask_qty_at_touch_',
                                                              'bid_add_rate_', 'ask_add_rate_',
                                                              'bid_cancel_rate_', 'ask_cancel_rate_',
                                                              'bid_trade_rate_', 'ask_trade_rate_',
                                                              'bid_level_age_', 'ask_level_age_',
                                                              'tick_count', 'ofi_1s_range', 'ofi_5s_range',
                                                              'imbalance_drift', 'microprice_stability'])]

    enhanced_features = baseline_features + queue_feature_names

    log.info(f"\nBaseline features: {len(baseline_features)}")
    log.info(f"Queue features: {len(queue_feature_names)}")
    log.info(f"Enhanced features: {len(enhanced_features)}")

    # ── 4. Walk-forward comparison ──
    dates = sorted(all_days.keys())
    n_dates = len(dates)

    baseline_ics = []
    enhanced_ics = []
    baseline_trades_all = []
    enhanced_trades_all = []
    fold_results = []

    log.info(f"\nWalk-forward: {TRAIN_DAYS}d train, 1d OOT, drop oldest")

    for fold_idx in range(TRAIN_DAYS, n_dates):
        train_dates = dates[fold_idx - TRAIN_DAYS:fold_idx]
        oot_date = dates[fold_idx]

        # Build training data
        train_frames = [all_days[d].dropna(subset=['target_30m']) for d in train_dates if d in all_days]
        if not train_frames:
            continue
        train_df = pd.concat(train_frames, ignore_index=True)

        # Build OOT data
        if oot_date not in all_days:
            continue
        oot_df = all_days[oot_date].dropna(subset=['target_30m'])

        if len(train_df) < 100 or len(oot_df) < 30:
            continue

        # ── Train BASELINE model ──
        X_train_base = train_df[baseline_features].values.astype(np.float32)
        X_oot_base = oot_df[baseline_features].values.astype(np.float32)
        y_train = train_df['target_30m'].values.astype(np.float32)
        y_oot = oot_df['target_30m'].values.astype(np.float32)

        X_train_base = np.nan_to_num(X_train_base, nan=0.0, posinf=100.0, neginf=-100.0)
        X_oot_base = np.nan_to_num(X_oot_base, nan=0.0, posinf=100.0, neginf=-100.0)

        train_data = lgb.Dataset(X_train_base, label=y_train, feature_name=baseline_features)
        base_model = lgb.train(LGBM_PARAMS, train_data, num_boost_round=300)

        base_preds = base_model.predict(X_oot_base)
        base_ic = float(np.corrcoef(base_preds, y_oot)[0, 1]) if len(y_oot) > 10 else 0

        # ── Train ENHANCED model ──
        X_train_enh = train_df[enhanced_features].values.astype(np.float32)
        X_oot_enh = oot_df[enhanced_features].values.astype(np.float32)

        X_train_enh = np.nan_to_num(X_train_enh, nan=0.0, posinf=100.0, neginf=-100.0)
        X_oot_enh = np.nan_to_num(X_oot_enh, nan=0.0, posinf=100.0, neginf=-100.0)

        train_data_enh = lgb.Dataset(X_train_enh, label=y_train, feature_name=enhanced_features)
        enh_model = lgb.train(LGBM_PARAMS, train_data_enh, num_boost_round=300)

        enh_preds = enh_model.predict(X_oot_enh)
        enh_ic = float(np.corrcoef(enh_preds, y_oot)[0, 1]) if len(y_oot) > 10 else 0

        baseline_ics.append(base_ic)
        enhanced_ics.append(enh_ic)

        # Simulate trades with both models
        oot_mb = all_days[oot_date].copy()
        base_trades = simulate_champion_trades(oot_mb, base_model, baseline_features)
        enh_trades = simulate_champion_trades(oot_mb, enh_model, enhanced_features)

        for t in base_trades:
            t['date'] = oot_date
        for t in enh_trades:
            t['date'] = oot_date

        baseline_trades_all.extend(base_trades)
        enhanced_trades_all.extend(enh_trades)

        fold_result = {
            'fold': fold_idx - TRAIN_DAYS,
            'oot_date': oot_date,
            'regime': date_regimes.get(oot_date, 'unknown'),
            'n_bars': len(oot_df),
            'baseline_ic': round(base_ic, 4),
            'enhanced_ic': round(enh_ic, 4),
            'ic_improvement': round(enh_ic - base_ic, 4),
            'baseline_trades': len(base_trades),
            'enhanced_trades': len(enh_trades),
            'baseline_pnl': round(sum(t['net_pnl'] for t in base_trades), 2),
            'enhanced_pnl': round(sum(t['net_pnl'] for t in enh_trades), 2),
        }
        fold_results.append(fold_result)

        log.info(f"  Fold {fold_result['fold']:2d} ({oot_date} {date_regimes.get(oot_date,'?'):5s}): "
                 f"base_IC={base_ic:+.3f} enh_IC={enh_ic:+.3f} Δ={enh_ic-base_ic:+.3f} | "
                 f"base_trades={len(base_trades)} enh_trades={len(enh_trades)}")

    # ── 5. Aggregate results ──
    log.info(f"\n{'='*70}")
    log.info(f"AGGREGATE RESULTS ({len(fold_results)} folds)")
    log.info(f"{'='*70}")

    # IC comparison
    base_ic_mean = np.mean(baseline_ics)
    enh_ic_mean = np.mean(enhanced_ics)
    ic_improvement = enh_ic_mean - base_ic_mean

    # Paired t-test on ICs
    if len(baseline_ics) > 2:
        t_stat, p_val = sp_stats.ttest_rel(enhanced_ics, baseline_ics)
    else:
        t_stat, p_val = 0, 1

    log.info(f"\n--- IC COMPARISON ---")
    log.info(f"  Baseline IC (mean): {base_ic_mean:.4f}")
    log.info(f"  Enhanced IC (mean): {enh_ic_mean:.4f}")
    log.info(f"  IC Improvement: {ic_improvement:+.4f}")
    log.info(f"  Paired t-test: t={t_stat:.3f}, p={p_val:.4f}")
    log.info(f"  N folds where enhanced > baseline: "
             f"{sum(1 for e, b in zip(enhanced_ics, baseline_ics) if e > b)}/{len(fold_results)}")

    # Trade performance comparison
    log.info(f"\n--- TRADE PERFORMANCE ---")

    if baseline_trades_all:
        base_df = pd.DataFrame(baseline_trades_all)
        base_wr = (base_df['net_pnl'] > 0).mean()
        base_total = base_df['net_pnl'].sum()
        base_per = base_df['net_pnl'].mean()
        base_gap, base_green, base_red = compute_regime_gap(baseline_trades_all, date_regimes)

        log.info(f"  Baseline: {len(base_df)} trades, WR={base_wr:.3f}, "
                 f"total={base_total:+.1f}t, per_trade={base_per:+.3f}t, "
                 f"regime_gap={base_gap:.3f}")

    if enhanced_trades_all:
        enh_df = pd.DataFrame(enhanced_trades_all)
        enh_wr = (enh_df['net_pnl'] > 0).mean()
        enh_total = enh_df['net_pnl'].sum()
        enh_per = enh_df['net_pnl'].mean()
        enh_gap, enh_green, enh_red = compute_regime_gap(enhanced_trades_all, date_regimes)

        log.info(f"  Enhanced: {len(enh_df)} trades, WR={enh_wr:.3f}, "
                 f"total={enh_total:+.1f}t, per_trade={enh_per:+.3f}t, "
                 f"regime_gap={enh_gap:.3f}")

    # Feature importance from last fold
    if fold_results:
        log.info(f"\n--- QUEUE FEATURE IMPORTANCE (last fold) ---")
        fi = dict(zip(enhanced_features, enh_model.feature_importance('gain')))
        sorted_fi = sorted(fi.items(), key=lambda x: x[1], reverse=True)
        for rank, (feat, imp) in enumerate(sorted_fi[:20]):
            marker = " **QUEUE**" if feat in queue_feature_names else ""
            log.info(f"  {rank+1:2d}. {feat:40s} gain={imp:.0f}{marker}")

    # ── 6. Save results ──
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_dates': len(all_days),
        'n_folds': len(fold_results),
        'n_baseline_features': len(baseline_features),
        'n_enhanced_features': len(enhanced_features),
        'n_queue_features': len(queue_feature_names),
        'baseline_features': baseline_features,
        'queue_features': queue_feature_names,
        'ic_comparison': {
            'baseline_mean': round(base_ic_mean, 4),
            'enhanced_mean': round(enh_ic_mean, 4),
            'improvement': round(ic_improvement, 4),
            't_stat': round(float(t_stat), 3),
            'p_value': round(float(p_val), 4),
        },
        'fold_results': fold_results,
        'date_regimes': date_regimes,
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    log.info(f"\nResults saved to {OUTPUT_DIR}")

    # MLflow
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("queue_minute_features")
        with mlflow.start_run(run_name=f"v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params({
                'n_baseline_features': len(baseline_features),
                'n_queue_features': len(queue_feature_names),
                'n_dates': len(all_days),
                'train_days': TRAIN_DAYS,
            })
            mlflow.log_metrics({
                'baseline_ic': base_ic_mean,
                'enhanced_ic': enh_ic_mean,
                'ic_improvement': ic_improvement,
                'ic_pvalue': float(p_val),
            })
            mlflow.log_artifact(str(results_path))
        log.info("MLflow run logged")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


if __name__ == '__main__':
    main()
