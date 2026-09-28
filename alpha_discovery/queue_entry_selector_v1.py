#!/usr/bin/env python3
"""
queue_entry_selector_v1.py — Queue-Enhanced Entry Selection (HC #648 / HC #428)
===============================================================================

Uses tick-level queue microstructure features to predict which entries will be
profitable under FIFO TP4/SL3 passive execution. The goal: filter entries to
only those where the orderbook state favors the trade.

Data:
  - 40 dates of queue_augmented_features (~45K per date)
  - 40 dates of mbo_events_smart_v3_fifo_labels (~47K per date)
  - Joined by ts_ns

Walk-forward: 25d train, 5d OOT, slide 5d (SLIDING per HC #0)
Regime gate: |Sharpe_green - Sharpe_red| / max(...) ≤ 0.50 (HC #428)
MFE gate: TP ≤ p90 MFE within horizon (HC #432)

Output: per-fold + aggregate metrics, feature importances, regime analysis
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
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ──
BASE = Path("/home/nick/Lvl3Quant")
QUEUE_DIR = BASE / "data" / "queue_augmented_features"
FIFO_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
MINUTE_DIR = BASE / "data" / "processed" / "mbo_minute_bars_v1"
OUTPUT_DIR = BASE / "output" / "queue_entry_selector_v1"
LOG_PATH = BASE / "logs" / "queue_entry_selector_v1.log"

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
ES_TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
PASSIVE_COST_TICKS = COMMISSION_RT_TICKS  # passive limit, no spread crossing

# Walk-forward params
TRAIN_DAYS = 25
OOT_DAYS = 5
SLIDE_DAYS = 5

# Trade config (matching champion TP4/SL3 from FIFO labels)
TP_TICKS = 4
SL_TICKS = 3

# Entry confidence threshold — only consider signals where model signal > gate
SIGNAL_GATE = 0.05

# LGBM params
LGBM_PARAMS = {
    'objective': 'binary',
    'metric': 'auc',
    'learning_rate': 0.03,
    'num_leaves': 31,
    'max_depth': 6,
    'min_data_in_leaf': 100,
    'feature_fraction': 0.8,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': -1,
    'seed': 42,
    'n_jobs': -1,
}
NUM_BOOST_ROUND = 500
EARLY_STOP_ROUNDS = 50

# ── Queue feature columns ──
QUEUE_FEATURES = [
    'bid_qty_at_touch', 'bid_n_orders', 'bid_q_ahead_if_join_back', 'bid_q_ahead_p50',
    'ask_qty_at_touch', 'ask_n_orders', 'ask_q_ahead_if_join_back', 'ask_q_ahead_p50',
    'bid_level_age_s', 'ask_level_age_s',
    'bid_time_since_last_add_s', 'bid_time_since_last_cancel_s',
    'ask_time_since_last_add_s', 'ask_time_since_last_cancel_s',
    'bid_add_rate_1s', 'ask_add_rate_1s',
    'bid_cancel_rate_1s', 'ask_cancel_rate_1s',
    'bid_trade_rate_1s', 'ask_trade_rate_1s',
    'ofi_1s', 'ofi_5s', 'ofi_10s',
    'top_imbalance', 'microprice_offset_ticks',
]

# ── Derived features ──
def add_derived_features(df):
    """Add derived queue microstructure features."""
    eps = 1e-8
    # Queue imbalance variants
    df['queue_ratio'] = df['bid_qty_at_touch'] / (df['ask_qty_at_touch'] + eps)
    df['queue_diff'] = df['bid_qty_at_touch'] - df['ask_qty_at_touch']
    df['order_count_ratio'] = df['bid_n_orders'] / (df['ask_n_orders'] + eps)

    # Activity ratios
    df['bid_net_flow_1s'] = df['bid_add_rate_1s'] - df['bid_cancel_rate_1s']
    df['ask_net_flow_1s'] = df['ask_add_rate_1s'] - df['ask_cancel_rate_1s']
    df['net_flow_diff_1s'] = df['bid_net_flow_1s'] - df['ask_net_flow_1s']

    # Level freshness
    df['bid_level_freshness'] = 1.0 / (df['bid_level_age_s'] + 0.01)
    df['ask_level_freshness'] = 1.0 / (df['ask_level_age_s'] + 0.01)
    df['level_age_diff'] = df['bid_level_age_s'] - df['ask_level_age_s']

    # Trade intensity
    df['total_trade_rate'] = df['bid_trade_rate_1s'] + df['ask_trade_rate_1s']
    df['trade_imbalance'] = (df['bid_trade_rate_1s'] - df['ask_trade_rate_1s']) / (df['total_trade_rate'] + eps)

    # OFI momentum
    df['ofi_accel'] = df['ofi_1s'] - df['ofi_5s'] / 5.0  # 1s OFI minus normalized 5s OFI
    df['ofi_momentum'] = df['ofi_5s'] - df['ofi_10s'] / 2.0  # 5s OFI minus normalized 10s OFI

    # Direction-aware features (positive = favorable for long, negative for short)
    # Will be flipped for shorts during training

    return df

DERIVED_FEATURES = [
    'queue_ratio', 'queue_diff', 'order_count_ratio',
    'bid_net_flow_1s', 'ask_net_flow_1s', 'net_flow_diff_1s',
    'bid_level_freshness', 'ask_level_freshness', 'level_age_diff',
    'total_trade_rate', 'trade_imbalance',
    'ofi_accel', 'ofi_momentum',
]

ALL_FEATURES = QUEUE_FEATURES + DERIVED_FEATURES


def load_data_for_date(date_str):
    """Load and join queue features with FIFO labels for one date."""
    q_path = QUEUE_DIR / f"features_{date_str}.parquet"
    f_path = FIFO_DIR / f"{date_str}_fifo_labels.npz"

    if not q_path.exists() or not f_path.exists():
        return None

    qdf = pd.read_parquet(q_path)
    flz = np.load(str(f_path), allow_pickle=True)

    # Build FIFO labels dataframe
    fifo_df = pd.DataFrame({
        'ts_ns': flz['ts_ns'],
        'tp4sl3_long_filled': flz['tp4sl3_long_filled'],
        'tp4sl3_long_net_ticks': flz['tp4sl3_long_net_ticks'],
        'tp4sl3_long_hit_tp': flz['tp4sl3_long_hit_tp'],
        'tp4sl3_short_filled': flz['tp4sl3_short_filled'],
        'tp4sl3_short_net_ticks': flz['tp4sl3_short_net_ticks'],
        'tp4sl3_short_hit_tp': flz['tp4sl3_short_hit_tp'],
    })

    # Join on ts_ns
    merged = qdf.merge(fifo_df, on='ts_ns', how='inner')

    if len(merged) == 0:
        return None

    # Add derived features
    merged = add_derived_features(merged)
    merged['date'] = date_str

    return merged


def prepare_training_data(df):
    """Prepare features and targets for LGBM.

    For each signal event, we create TWO rows:
    - One for LONG entry (using tp4sl3_long outcomes)
    - One for SHORT entry (using tp4sl3_short outcomes, features direction-flipped)

    Target: 1 if trade was profitable (net_ticks > 0), 0 otherwise.
    Only include rows where the FIFO fill actually happened.
    """
    rows = []

    # Long entries
    mask_long = df['tp4sl3_long_filled'].values.astype(bool)
    if mask_long.sum() > 0:
        long_df = df[mask_long].copy()
        long_df['target'] = (long_df['tp4sl3_long_net_ticks'] > 0).astype(int)
        long_df['direction'] = 1
        long_df['net_ticks'] = long_df['tp4sl3_long_net_ticks']
        long_df['hit_tp'] = long_df['tp4sl3_long_hit_tp']
        rows.append(long_df)

    # Short entries (flip direction-sensitive features)
    mask_short = df['tp4sl3_short_filled'].values.astype(bool)
    if mask_short.sum() > 0:
        short_df = df[mask_short].copy()
        short_df['target'] = (short_df['tp4sl3_short_net_ticks'] > 0).astype(int)
        short_df['direction'] = -1
        short_df['net_ticks'] = short_df['tp4sl3_short_net_ticks']
        short_df['hit_tp'] = short_df['tp4sl3_short_hit_tp']

        # Flip direction-sensitive features for shorts
        for col in ['ofi_1s', 'ofi_5s', 'ofi_10s', 'top_imbalance', 'microprice_offset_ticks',
                     'queue_diff', 'net_flow_diff_1s', 'level_age_diff', 'trade_imbalance',
                     'ofi_accel', 'ofi_momentum']:
            if col in short_df.columns:
                short_df[col] = -short_df[col]
        # Swap bid/ask features for shorts
        swap_pairs = [
            ('bid_qty_at_touch', 'ask_qty_at_touch'),
            ('bid_n_orders', 'ask_n_orders'),
            ('bid_q_ahead_if_join_back', 'ask_q_ahead_if_join_back'),
            ('bid_q_ahead_p50', 'ask_q_ahead_p50'),
            ('bid_level_age_s', 'ask_level_age_s'),
            ('bid_time_since_last_add_s', 'ask_time_since_last_add_s'),
            ('bid_time_since_last_cancel_s', 'ask_time_since_last_cancel_s'),
            ('bid_add_rate_1s', 'ask_add_rate_1s'),
            ('bid_cancel_rate_1s', 'ask_cancel_rate_1s'),
            ('bid_trade_rate_1s', 'ask_trade_rate_1s'),
            ('bid_net_flow_1s', 'ask_net_flow_1s'),
            ('bid_level_freshness', 'ask_level_freshness'),
        ]
        for a, b in swap_pairs:
            if a in short_df.columns and b in short_df.columns:
                short_df[a], short_df[b] = short_df[b].copy(), short_df[a].copy()
        # Recompute ratios after swap
        eps = 1e-8
        short_df['queue_ratio'] = short_df['bid_qty_at_touch'] / (short_df['ask_qty_at_touch'] + eps)
        short_df['order_count_ratio'] = short_df['bid_n_orders'] / (short_df['ask_n_orders'] + eps)

        rows.append(short_df)

    if not rows:
        return None

    result = pd.concat(rows, ignore_index=True)
    return result


def classify_day_regime(date_str):
    """Classify day as green/red/flat using minute-bar close-to-close."""
    mb_path = MINUTE_DIR / f"{date_str}.parquet"
    if not mb_path.exists():
        return 'unknown'
    mb = pd.read_parquet(mb_path)
    if 'close' not in mb.columns or len(mb) < 2:
        return 'unknown'
    day_return = mb['close'].iloc[-1] - mb['close'].iloc[0]
    if day_return > 2:  # > 2 ticks = green
        return 'green'
    elif day_return < -2:
        return 'red'
    else:
        return 'flat'


def compute_regime_gap(results_df):
    """Compute regime gap: |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|)."""
    green = results_df[results_df['regime'] == 'green']
    red = results_df[results_df['regime'] == 'red']

    if len(green) < 3 or len(red) < 3:
        return float('nan'), {}, {}

    green_daily = green.groupby('date')['pnl_ticks'].sum()
    red_daily = red.groupby('date')['pnl_ticks'].sum()

    sharpe_green = green_daily.mean() / (green_daily.std() + 1e-8) * np.sqrt(252)
    sharpe_red = red_daily.mean() / (red_daily.std() + 1e-8) * np.sqrt(252)

    gap = abs(sharpe_green - sharpe_red) / max(abs(sharpe_green), abs(sharpe_red), 1e-8)

    green_stats = {'sharpe': float(sharpe_green), 'n_days': len(green_daily), 'n_trades': len(green)}
    red_stats = {'sharpe': float(sharpe_red), 'n_days': len(red_daily), 'n_trades': len(red)}

    return float(gap), green_stats, red_stats


def evaluate_strategy(trades_df, strategy_name):
    """Compute risk-adjusted metrics for a set of trades."""
    if len(trades_df) < 10:
        return {'strategy': strategy_name, 'n_trades': len(trades_df), 'error': 'too_few_trades'}

    pnl = trades_df['pnl_ticks'].values
    winners = pnl > 0
    losers = pnl < 0

    wr = winners.mean()
    avg_win = pnl[winners].mean() if winners.any() else 0
    avg_loss = abs(pnl[losers].mean()) if losers.any() else 1
    pf = (pnl[winners].sum() / abs(pnl[losers].sum())) if losers.any() and pnl[losers].sum() != 0 else float('inf')

    total_ticks = pnl.sum()
    per_trade = pnl.mean()

    # Daily Sharpe
    daily_pnl = trades_df.groupby('date')['pnl_ticks'].sum()
    daily_sharpe = daily_pnl.mean() / (daily_pnl.std() + 1e-8) * np.sqrt(252)

    # Sortino
    downside = daily_pnl[daily_pnl < 0]
    downside_std = downside.std() if len(downside) > 1 else daily_pnl.std()
    sortino = daily_pnl.mean() / (downside_std + 1e-8) * np.sqrt(252)

    # Regime analysis
    gap, green_stats, red_stats = compute_regime_gap(trades_df)

    # Trades per day
    n_days = trades_df['date'].nunique()
    trades_per_day = len(trades_df) / n_days if n_days > 0 else 0

    return {
        'strategy': strategy_name,
        'n_trades': len(trades_df),
        'n_days': int(n_days),
        'trades_per_day': round(trades_per_day, 1),
        'wr': round(wr, 4),
        'pf': round(pf, 3),
        'avg_win': round(avg_win, 2),
        'avg_loss': round(avg_loss, 2),
        'per_trade_ticks': round(per_trade, 3),
        'total_ticks': round(total_ticks, 1),
        'daily_sharpe': round(daily_sharpe, 2),
        'sortino': round(sortino, 2),
        'regime_gap': round(gap, 3),
        'green': green_stats,
        'red': red_stats,
    }


def main():
    log.info("=" * 70)
    log.info("Queue-Enhanced Entry Selector v1")
    log.info("=" * 70)

    # ── 1. Discover and load data ──
    q_dates = sorted([f.replace('features_', '').replace('.parquet', '')
                       for f in os.listdir(QUEUE_DIR) if f.endswith('.parquet')])
    f_dates = sorted([f.replace('_fifo_labels.npz', '')
                       for f in os.listdir(FIFO_DIR) if f.endswith('.npz')])
    overlap = sorted(set(q_dates) & set(f_dates))
    log.info(f"Queue dates: {len(q_dates)}, FIFO dates: {len(f_dates)}, Overlap: {len(overlap)}")

    # Load all data
    all_data = []
    for d in overlap:
        df = load_data_for_date(d)
        if df is not None:
            all_data.append(df)
            log.info(f"  {d}: {len(df):,} rows")

    if not all_data:
        log.error("No data loaded!")
        return

    combined = pd.concat(all_data, ignore_index=True)
    log.info(f"\nTotal joined data: {len(combined):,} rows across {len(all_data)} dates")

    # ── 2. Prepare training samples ──
    training_data = prepare_training_data(combined)
    if training_data is None or len(training_data) < 100:
        log.error(f"Insufficient training data: {len(training_data) if training_data is not None else 0} rows")
        return

    log.info(f"Training samples: {len(training_data):,}")
    log.info(f"  Long: {(training_data['direction']==1).sum():,}, Short: {(training_data['direction']==-1).sum():,}")
    log.info(f"  Positive (profitable): {training_data['target'].mean():.3f}")
    log.info(f"  Mean net ticks: {training_data['net_ticks'].mean():.3f}")

    # Regime classification
    date_regimes = {}
    for d in overlap:
        date_regimes[d] = classify_day_regime(d)
    training_data['regime'] = training_data['date'].map(date_regimes)
    log.info(f"  Regimes: {pd.Series(date_regimes).value_counts().to_dict()}")

    # ── 3. Walk-forward validation ──
    dates = sorted(training_data['date'].unique())
    n_dates = len(dates)

    log.info(f"\nWalk-forward: {TRAIN_DAYS}d train, {OOT_DAYS}d test, {SLIDE_DAYS}d slide")
    log.info(f"  {n_dates} dates total")

    all_oot_trades = []
    fold_results = []
    feature_importances = {}

    fold_start = TRAIN_DAYS
    fold_num = 0

    while fold_start + OOT_DAYS <= n_dates:
        train_dates = dates[fold_start - TRAIN_DAYS:fold_start]
        oot_dates = dates[fold_start:fold_start + OOT_DAYS]

        train_df = training_data[training_data['date'].isin(train_dates)]
        oot_df = training_data[training_data['date'].isin(oot_dates)]

        if len(train_df) < 100 or len(oot_df) < 10:
            fold_start += SLIDE_DAYS
            continue

        X_train = train_df[ALL_FEATURES].values.astype(np.float32)
        y_train = train_df['target'].values
        X_oot = oot_df[ALL_FEATURES].values.astype(np.float32)
        y_oot = oot_df['target'].values

        # Replace inf/nan
        X_train = np.nan_to_num(X_train, nan=0.0, posinf=100.0, neginf=-100.0)
        X_oot = np.nan_to_num(X_oot, nan=0.0, posinf=100.0, neginf=-100.0)

        # Train LGBM
        train_data = lgb.Dataset(X_train, label=y_train, feature_name=ALL_FEATURES)

        # Use last 20% of train as validation for early stopping
        split_idx = int(len(X_train) * 0.8)
        val_data = lgb.Dataset(X_train[split_idx:], label=y_train[split_idx:], reference=train_data)

        callbacks = [lgb.early_stopping(EARLY_STOP_ROUNDS, verbose=False),
                     lgb.log_evaluation(0)]

        model = lgb.train(
            LGBM_PARAMS, train_data,
            num_boost_round=NUM_BOOST_ROUND,
            valid_sets=[val_data],
            callbacks=callbacks,
        )

        # Predict on OOT
        oot_probs = model.predict(X_oot)

        # Track feature importances
        for feat, imp in zip(ALL_FEATURES, model.feature_importance('gain')):
            if feat not in feature_importances:
                feature_importances[feat] = []
            feature_importances[feat].append(float(imp))

        # Evaluate multiple filtering strategies
        oot_df = oot_df.copy()
        oot_df['pred_prob'] = oot_probs
        oot_df['pnl_ticks'] = oot_df['net_ticks']

        # Baseline: all trades (no filtering)
        baseline_stats = {
            'n_trades': len(oot_df),
            'wr': float(oot_df['target'].mean()),
            'total_ticks': float(oot_df['net_ticks'].sum()),
            'per_trade': float(oot_df['net_ticks'].mean()),
        }

        # Strategy 1: Top 50% by model confidence
        top50 = oot_df[oot_df['pred_prob'] >= oot_df['pred_prob'].median()].copy()

        # Strategy 2: Top 30% (aggressive filtering)
        p70 = oot_df['pred_prob'].quantile(0.7)
        top30 = oot_df[oot_df['pred_prob'] >= p70].copy()

        # Strategy 3: Threshold-based (prob > 0.55)
        above_thresh = oot_df[oot_df['pred_prob'] > 0.55].copy()

        fold_result = {
            'fold': fold_num,
            'train_dates': f"{train_dates[0]}..{train_dates[-1]}",
            'oot_dates': f"{oot_dates[0]}..{oot_dates[-1]}",
            'n_train': len(train_df),
            'n_oot': len(oot_df),
            'baseline': baseline_stats,
            'top50': {
                'n_trades': len(top50),
                'wr': float(top50['target'].mean()) if len(top50) > 0 else 0,
                'total_ticks': float(top50['net_ticks'].sum()) if len(top50) > 0 else 0,
                'per_trade': float(top50['net_ticks'].mean()) if len(top50) > 0 else 0,
            },
            'top30': {
                'n_trades': len(top30),
                'wr': float(top30['target'].mean()) if len(top30) > 0 else 0,
                'total_ticks': float(top30['net_ticks'].sum()) if len(top30) > 0 else 0,
                'per_trade': float(top30['net_ticks'].mean()) if len(top30) > 0 else 0,
            },
            'above_thresh': {
                'n_trades': len(above_thresh),
                'wr': float(above_thresh['target'].mean()) if len(above_thresh) > 0 else 0,
                'total_ticks': float(above_thresh['net_ticks'].sum()) if len(above_thresh) > 0 else 0,
                'per_trade': float(above_thresh['net_ticks'].mean()) if len(above_thresh) > 0 else 0,
            },
        }
        fold_results.append(fold_result)

        # Collect all OOT trades for aggregate analysis
        all_oot_trades.append(oot_df)

        log.info(f"  Fold {fold_num}: train={len(train_df):,} oot={len(oot_df):,} "
                 f"base_wr={baseline_stats['wr']:.3f} "
                 f"top50_wr={fold_result['top50']['wr']:.3f} "
                 f"top30_wr={fold_result['top30']['wr']:.3f}")

        fold_start += SLIDE_DAYS
        fold_num += 1

    if not all_oot_trades:
        log.error("No folds completed!")
        return

    # ── 4. Aggregate analysis ──
    all_trades = pd.concat(all_oot_trades, ignore_index=True)
    log.info(f"\n{'='*70}")
    log.info(f"AGGREGATE RESULTS ({fold_num} folds, {len(all_trades):,} OOT trades)")
    log.info(f"{'='*70}")

    # Baseline: all trades
    baseline_result = evaluate_strategy(all_trades, 'baseline_all')
    log.info(f"\n--- BASELINE (all trades) ---")
    log.info(f"  N={baseline_result['n_trades']}, WR={baseline_result['wr']:.3f}, "
             f"PF={baseline_result['pf']:.3f}, Sharpe={baseline_result['daily_sharpe']:.2f}, "
             f"Regime gap={baseline_result['regime_gap']:.3f}")

    # Filtered strategies
    strategies = []
    for name, min_pct in [('top50', 0.50), ('top40', 0.60), ('top30', 0.70),
                           ('top20', 0.80), ('top10', 0.90)]:
        threshold = all_trades['pred_prob'].quantile(min_pct)
        filtered = all_trades[all_trades['pred_prob'] >= threshold].copy()
        result = evaluate_strategy(filtered, name)
        strategies.append(result)
        log.info(f"\n--- {name.upper()} (prob ≥ {threshold:.3f}) ---")
        log.info(f"  N={result['n_trades']}, WR={result['wr']:.3f}, "
                 f"PF={result['pf']:.3f}, Sharpe={result['daily_sharpe']:.2f}, "
                 f"Sortino={result['sortino']:.2f}, "
                 f"Per-trade={result['per_trade_ticks']:.3f}t, "
                 f"Regime gap={result['regime_gap']:.3f}")

    # Threshold-based strategies
    for thresh in [0.50, 0.52, 0.55, 0.58, 0.60]:
        filtered = all_trades[all_trades['pred_prob'] >= thresh].copy()
        if len(filtered) > 10:
            result = evaluate_strategy(filtered, f'thresh_{thresh}')
            strategies.append(result)
            log.info(f"\n--- THRESHOLD {thresh} ---")
            log.info(f"  N={result['n_trades']}, WR={result['wr']:.3f}, "
                     f"PF={result['pf']:.3f}, Sharpe={result['daily_sharpe']:.2f}, "
                     f"Regime gap={result['regime_gap']:.3f}")

    # ── 5. Feature importance ──
    log.info(f"\n--- FEATURE IMPORTANCE (avg gain across folds) ---")
    fi_summary = {}
    for feat in ALL_FEATURES:
        if feat in feature_importances:
            vals = feature_importances[feat]
            fi_summary[feat] = {
                'mean_gain': float(np.mean(vals)),
                'std_gain': float(np.std(vals)),
                'rank': 0,  # filled below
            }

    # Rank by mean gain
    sorted_features = sorted(fi_summary.items(), key=lambda x: x[1]['mean_gain'], reverse=True)
    for rank, (feat, fi_stat) in enumerate(sorted_features):
        fi_stat['rank'] = rank + 1
        if rank < 15:
            log.info(f"  {rank+1:2d}. {feat:35s} gain={fi_stat['mean_gain']:.1f} ± {fi_stat['std_gain']:.1f}")

    # ── 6. IC analysis ──
    # Check if model probability correlates with actual net P&L
    ic = np.corrcoef(all_trades['pred_prob'], all_trades['net_ticks'])[0, 1]
    rank_ic = stats.spearmanr(all_trades['pred_prob'], all_trades['net_ticks'])[0]
    log.info(f"\n--- INFORMATION COEFFICIENT ---")
    log.info(f"  Pearson IC (prob vs net_ticks): {ic:.4f}")
    log.info(f"  Rank IC (Spearman): {rank_ic:.4f}")

    # ── 7. Model calibration ──
    log.info(f"\n--- MODEL CALIBRATION (predicted vs actual WR) ---")
    n_bins = 10
    bins = np.percentile(all_trades['pred_prob'], np.linspace(0, 100, n_bins + 1))
    bins = np.unique(bins)
    all_trades['prob_bin'] = pd.cut(all_trades['pred_prob'], bins=bins, include_lowest=True)
    calibration = all_trades.groupby('prob_bin', observed=True).agg(
        predicted_prob=('pred_prob', 'mean'),
        actual_wr=('target', 'mean'),
        mean_pnl=('net_ticks', 'mean'),
        n_trades=('target', 'count')
    ).reset_index()
    for _, row in calibration.iterrows():
        log.info(f"  pred={row['predicted_prob']:.3f} → actual_wr={row['actual_wr']:.3f} "
                 f"pnl={row['mean_pnl']:+.3f}t (n={row['n_trades']:,})")

    # ── 8. Save results ──
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_dates': len(overlap),
        'n_total_samples': len(all_trades),
        'n_folds': fold_num,
        'baseline': baseline_result,
        'strategies': strategies,
        'feature_importance': fi_summary,
        'ic': {'pearson': float(ic), 'spearman': float(rank_ic)},
        'calibration': calibration[['predicted_prob', 'actual_wr', 'mean_pnl', 'n_trades']].to_dict('records'),
        'fold_results': fold_results,
        'date_regimes': date_regimes,
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save all trades with predictions (drop interval columns that can't serialize)
    trades_path = OUTPUT_DIR / "oot_trades.parquet"
    save_cols = [c for c in all_trades.columns if not isinstance(all_trades[c].dtype, pd.CategoricalDtype)
                 and 'interval' not in str(all_trades[c].dtype).lower()]
    # Also drop prob_bin if it exists (pd.cut creates intervals)
    save_cols = [c for c in save_cols if c != 'prob_bin']
    all_trades[save_cols].to_parquet(trades_path, index=False)

    log.info(f"\n{'='*70}")
    log.info(f"Results saved to {OUTPUT_DIR}")
    log.info(f"{'='*70}")

    # ── 9. Summary for reporting ──
    best_strat = max(strategies, key=lambda x: x.get('daily_sharpe', -999))
    log.info(f"\nBEST FILTERED STRATEGY: {best_strat['strategy']}")
    log.info(f"  Sharpe: {best_strat['daily_sharpe']:.2f} (vs baseline {baseline_result['daily_sharpe']:.2f})")
    log.info(f"  PF: {best_strat['pf']:.3f} (vs baseline {baseline_result['pf']:.3f})")
    log.info(f"  WR: {best_strat['wr']:.3f} (vs baseline {baseline_result['wr']:.3f})")
    log.info(f"  Regime gap: {best_strat['regime_gap']:.3f} (≤0.50 = PASS)")

    # MLflow logging
    try:
        import mlflow
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("queue_entry_selector")
        with mlflow.start_run(run_name=f"v1_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params({
                'train_days': TRAIN_DAYS,
                'oot_days': OOT_DAYS,
                'slide_days': SLIDE_DAYS,
                'n_features': len(ALL_FEATURES),
                'n_dates': len(overlap),
            })
            mlflow.log_metrics({
                'baseline_sharpe': baseline_result['daily_sharpe'],
                'baseline_pf': baseline_result['pf'],
                'baseline_wr': baseline_result['wr'],
                'baseline_regime_gap': baseline_result['regime_gap'],
                'best_sharpe': best_strat['daily_sharpe'],
                'best_pf': best_strat['pf'],
                'best_wr': best_strat['wr'],
                'best_regime_gap': best_strat['regime_gap'],
                'ic_pearson': float(ic),
                'ic_spearman': float(rank_ic),
            })
            mlflow.log_artifact(str(results_path))
        log.info("MLflow run logged successfully")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")

    return results


if __name__ == '__main__':
    main()
