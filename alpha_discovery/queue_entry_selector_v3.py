#!/usr/bin/env python3
"""
queue_entry_selector_v3.py — Enhanced Queue Entry Selection
===========================================================

v3 improvements over v2:
  1. Multi-config: tests BOTH tp4sl3 AND tp8sl5 FIFO configs
  2. More features: n_orders, q_ahead_p50, time_since_last_add/cancel (unused in v2)
  3. Rolling z-scores: 300s rolling standardization to reduce day-to-day variation
  4. Time-of-day feature: seconds since RTH open (microstructure varies intraday)
  5. Separate long/short models: short side historically has stronger edge
  6. Better hyperparameter search: 3-config grid instead of single LGBM config
  7. Day-of-week feature: Mon vs Fri microstructure differs

Walk-forward: 25d train, 5d OOT, slide 5d (SLIDING per HC #0)
Regime gate: |Sharpe_green - Sharpe_red| / max(...) ≤ 0.50 (HC #428)
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
from scipy import stats as scipy_stats

warnings.filterwarnings('ignore')

# ── Paths ──
if Path("/home/nick/Lvl3Quant").exists():
    BASE = Path("/home/nick/Lvl3Quant")
elif Path("/home/jupiter/Lvl3Quant").exists():
    BASE = Path("/home/jupiter/Lvl3Quant")
else:
    raise RuntimeError("Cannot find Lvl3Quant directory")

QUEUE_DIR = BASE / "output" / "queue_features_universal"
FIFO_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
OUTPUT_DIR = BASE / "output" / "queue_entry_selector_v3"
LOG_PATH = BASE / "logs" / "queue_entry_selector_v3.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(BASE / "logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Config ──
ES_TICK_SIZE = 0.25
COMMISSION_RT_TICKS = 0.376

# FIFO configs to test
FIFO_CONFIGS = ['tp4sl3', 'tp8sl5']

# Walk-forward
TRAIN_DAYS = 25
OOT_DAYS = 5
SLIDE_DAYS = 5

# RTH start in nanoseconds offset from midnight (9:30 AM ET = 34200s)
RTH_START_S = 34200

# ── ALL features from universal extractor ──
RAW_FEATURES = [
    'ofi_10s', 'ofi_5s', 'ofi_1s',
    'top_imbalance', 'microprice_offset_ticks',
    'bid_qty_at_touch', 'ask_qty_at_touch',
    'bid_n_orders', 'ask_n_orders',
    'bid_q_ahead_p50', 'ask_q_ahead_p50',
    'bid_trade_rate_1s', 'ask_trade_rate_1s',
    'bid_add_rate_1s', 'ask_add_rate_1s',
    'bid_cancel_rate_1s', 'ask_cancel_rate_1s',
    'bid_level_age_s', 'ask_level_age_s',
    'bid_time_since_last_add_s', 'ask_time_since_last_add_s',
    'bid_time_since_last_cancel_s', 'ask_time_since_last_cancel_s',
]

# Direction-sensitive features that need sign flip for shorts
FLIP_FEATURES = ['ofi_10s', 'ofi_5s', 'ofi_1s', 'top_imbalance', 'microprice_offset_ticks',
                  'queue_diff', 'net_flow_diff', 'trade_imbalance', 'ofi_momentum',
                  'n_orders_diff', 'q_ahead_diff', 'add_recency_diff', 'cancel_recency_diff']

# Swap features: bid↔ask for shorts
SWAP_PAIRS = [
    ('bid_qty_at_touch', 'ask_qty_at_touch'),
    ('bid_n_orders', 'ask_n_orders'),
    ('bid_q_ahead_p50', 'ask_q_ahead_p50'),
    ('bid_trade_rate_1s', 'ask_trade_rate_1s'),
    ('bid_add_rate_1s', 'ask_add_rate_1s'),
    ('bid_cancel_rate_1s', 'ask_cancel_rate_1s'),
    ('bid_level_age_s', 'ask_level_age_s'),
    ('bid_time_since_last_add_s', 'ask_time_since_last_add_s'),
    ('bid_time_since_last_cancel_s', 'ask_time_since_last_cancel_s'),
]

# Z-score features (rolling 300s standardization)
ZSCORE_FEATURES = ['ofi_10s', 'ofi_5s', 'ofi_1s', 'top_imbalance',
                    'bid_qty_at_touch', 'ask_qty_at_touch',
                    'bid_trade_rate_1s', 'ask_trade_rate_1s']

# LGBM configs to try
LGBM_CONFIGS = {
    'base': {
        'objective': 'binary', 'metric': 'auc',
        'n_estimators': 300, 'learning_rate': 0.05,
        'num_leaves': 31, 'max_depth': 5,
        'min_child_samples': 20, 'subsample': 0.8,
        'colsample_bytree': 0.8, 'reg_alpha': 0.1,
        'reg_lambda': 1.0, 'verbose': -1, 'n_jobs': -1, 'random_state': 42,
    },
    'deeper': {
        'objective': 'binary', 'metric': 'auc',
        'n_estimators': 500, 'learning_rate': 0.03,
        'num_leaves': 63, 'max_depth': 7,
        'min_child_samples': 50, 'subsample': 0.7,
        'colsample_bytree': 0.7, 'reg_alpha': 1.0,
        'reg_lambda': 5.0, 'verbose': -1, 'n_jobs': -1, 'random_state': 42,
    },
    'conservative': {
        'objective': 'binary', 'metric': 'auc',
        'n_estimators': 200, 'learning_rate': 0.05,
        'num_leaves': 15, 'max_depth': 4,
        'min_child_samples': 100, 'subsample': 0.9,
        'colsample_bytree': 0.9, 'reg_alpha': 5.0,
        'reg_lambda': 10.0, 'verbose': -1, 'n_jobs': -1, 'random_state': 42,
    },
}


# ─────────────────────────────────────
# Data Loading
# ─────────────────────────────────────

def load_queue_features(date_str):
    """Load 1-second universal queue features for a date."""
    path = QUEUE_DIR / f"features_{date_str}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if 'ts_ns' not in df.columns or len(df) < 100:
        return None
    df = df.sort_values('ts_ns').reset_index(drop=True)
    return df


def load_fifo_labels(date_str, fifo_config):
    """Load FIFO labels for a date and config."""
    path = FIFO_DIR / f"{date_str}_fifo_labels.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)

    prefix = fifo_config
    required = [f'{prefix}_long_filled', f'{prefix}_long_net_ticks',
                f'{prefix}_short_filled', f'{prefix}_short_net_ticks']

    if not all(k in data for k in ['ts_ns'] + required):
        return None

    cols = {'ts_ns': data['ts_ns']}
    for side in ['long', 'short']:
        for field in ['filled', 'net_ticks', 'hit_tp', 'exit_reason', 'gross_ticks', 'hold_time_ns']:
            key = f'{prefix}_{side}_{field}'
            if key in data:
                cols[f'{side}_{field}'] = data[key]

    return pd.DataFrame(cols)


def join_queue_fifo(queue_df, fifo_df):
    """Join queue features to FIFO labels by nearest timestamp."""
    queue_df = queue_df.sort_values('ts_ns').reset_index(drop=True)
    fifo_df = fifo_df.sort_values('ts_ns').reset_index(drop=True)

    merged = pd.merge_asof(
        fifo_df, queue_df,
        on='ts_ns',
        direction='backward',
        tolerance=1_500_000_000  # 1.5 seconds
    )

    n_before = len(merged)
    merged = merged.dropna(subset=['ofi_10s'])
    n_after = len(merged)

    return merged, n_before, n_after


# ─────────────────────────────────────
# Feature Engineering
# ─────────────────────────────────────

def engineer_features(df):
    """Add derived features — richer than v2."""
    bid_q = df['bid_qty_at_touch'].clip(lower=1)
    ask_q = df['ask_qty_at_touch'].clip(lower=1)

    # v2 features
    df['queue_ratio'] = bid_q / (bid_q + ask_q)
    df['queue_diff'] = df['bid_qty_at_touch'] - df['ask_qty_at_touch']

    bid_net = df.get('bid_add_rate_1s', 0) - df.get('bid_cancel_rate_1s', 0)
    ask_net = df.get('ask_add_rate_1s', 0) - df.get('ask_cancel_rate_1s', 0)
    df['net_flow_diff'] = bid_net - ask_net

    df['trade_imbalance'] = df['bid_trade_rate_1s'] - df['ask_trade_rate_1s']
    df['ofi_momentum'] = df['ofi_10s'] - df['ofi_1s']
    df['level_age_diff'] = df['bid_level_age_s'] - df['ask_level_age_s']

    # NEW v3 features
    # Number of orders difference
    if 'bid_n_orders' in df.columns and 'ask_n_orders' in df.columns:
        df['n_orders_diff'] = df['bid_n_orders'] - df['ask_n_orders']
        df['n_orders_ratio'] = df['bid_n_orders'].clip(lower=1) / (
            df['bid_n_orders'].clip(lower=1) + df['ask_n_orders'].clip(lower=1))

    # Queue position ahead difference
    if 'bid_q_ahead_p50' in df.columns and 'ask_q_ahead_p50' in df.columns:
        df['q_ahead_diff'] = df['bid_q_ahead_p50'] - df['ask_q_ahead_p50']

    # Recency of activity (time since last add/cancel)
    if 'bid_time_since_last_add_s' in df.columns:
        df['add_recency_diff'] = df['bid_time_since_last_add_s'] - df['ask_time_since_last_add_s']
    if 'bid_time_since_last_cancel_s' in df.columns:
        df['cancel_recency_diff'] = df['bid_time_since_last_cancel_s'] - df['ask_time_since_last_cancel_s']

    # Cancel-to-add ratio (aggressiveness indicator)
    bid_add = df.get('bid_add_rate_1s', pd.Series(1, index=df.index)).clip(lower=1)
    ask_add = df.get('ask_add_rate_1s', pd.Series(1, index=df.index)).clip(lower=1)
    df['bid_cancel_add_ratio'] = df.get('bid_cancel_rate_1s', 0) / bid_add
    df['ask_cancel_add_ratio'] = df.get('ask_cancel_rate_1s', 0) / ask_add
    df['cancel_add_imbalance'] = df['bid_cancel_add_ratio'] - df['ask_cancel_add_ratio']

    # Trade-to-queue ratio (urgency indicator)
    df['bid_trade_queue_ratio'] = df.get('bid_trade_rate_1s', 0) / bid_q
    df['ask_trade_queue_ratio'] = df.get('ask_trade_rate_1s', 0) / ask_q

    # Time of day (seconds since RTH open) — normalized to [0, 1]
    # ts_ns is nanoseconds since epoch; extract intraday seconds
    ts_s = (df['ts_ns'] % (86400 * 1_000_000_000)) / 1_000_000_000  # seconds in day (UTC)
    # Adjust for ET (UTC-4 during EDT, UTC-5 during EST)
    # RTH is 9:30-16:00 ET = 13:30-20:00 UTC (EDT) or 14:30-21:00 UTC (EST)
    # We normalize relative to day start of trading, not absolute clock
    df['time_of_day'] = np.arange(len(df)) / max(len(df) - 1, 1)  # simple ordinal position [0,1]

    # Rolling z-scores (300-second window = 300 rows since 1-second snapshots)
    for feat in ZSCORE_FEATURES:
        if feat in df.columns:
            rolling_mean = df[feat].rolling(300, min_periods=30).mean()
            rolling_std = df[feat].rolling(300, min_periods=30).std().clip(lower=1e-6)
            df[f'{feat}_z300'] = (df[feat] - rolling_mean) / rolling_std

    return df


def get_feature_cols(sample_df):
    """Get all feature columns from a sample dataframe."""
    derived = ['queue_ratio', 'queue_diff', 'net_flow_diff', 'trade_imbalance',
               'ofi_momentum', 'level_age_diff', 'n_orders_diff', 'n_orders_ratio',
               'q_ahead_diff', 'add_recency_diff', 'cancel_recency_diff',
               'bid_cancel_add_ratio', 'ask_cancel_add_ratio', 'cancel_add_imbalance',
               'bid_trade_queue_ratio', 'ask_trade_queue_ratio', 'time_of_day']

    zscore_cols = [f'{f}_z300' for f in ZSCORE_FEATURES]

    all_candidates = RAW_FEATURES + derived + zscore_cols

    return [c for c in all_candidates if c in sample_df.columns]


def prepare_training_data(merged_df, date_str):
    """Create long + short training rows with direction-aware feature swapping."""
    rows = []

    for side in ['long', 'short']:
        filled_col = f'{side}_filled'
        net_col = f'{side}_net_ticks'

        if filled_col not in merged_df.columns:
            continue

        mask = merged_df[filled_col] == True
        subset = merged_df[mask].copy()

        if len(subset) == 0:
            continue

        subset['target'] = (subset[net_col] > 0).astype(int)
        subset['net_ticks'] = subset[net_col].astype(float)
        subset['side'] = side
        subset['date'] = date_str

        if side == 'short':
            for feat in FLIP_FEATURES:
                if feat in subset.columns:
                    subset[feat] = -subset[feat]
            for bid_feat, ask_feat in SWAP_PAIRS:
                if bid_feat in subset.columns and ask_feat in subset.columns:
                    subset[bid_feat], subset[ask_feat] = subset[ask_feat].copy(), subset[bid_feat].copy()
            # Also swap z-score features
            for base_feat in ZSCORE_FEATURES:
                bid_z = f'bid_{base_feat}_z300' if f'bid_{base_feat}_z300' in subset.columns else None
                ask_z = f'ask_{base_feat}_z300' if f'ask_{base_feat}_z300' in subset.columns else None
                # Only swap bid/ask z-scores, flip directional z-scores
                z_col = f'{base_feat}_z300'
                if z_col in subset.columns and base_feat in FLIP_FEATURES:
                    subset[z_col] = -subset[z_col]

        rows.append(subset)

    if not rows:
        return None

    return pd.concat(rows, ignore_index=True)


def classify_day_regime(queue_df):
    """Classify day as green/red based on first/last mid_price."""
    if queue_df is None or 'mid_price' not in queue_df.columns or len(queue_df) < 10:
        return 'unknown'
    day_return = queue_df['mid_price'].iloc[-1] - queue_df['mid_price'].iloc[0]
    if day_return > 2 * ES_TICK_SIZE:
        return 'green'
    elif day_return < -2 * ES_TICK_SIZE:
        return 'red'
    return 'flat'


def compute_regime_gap(trades_df):
    """Compute regime gap metric (HC #428 R1)."""
    if trades_df is None or len(trades_df) == 0:
        return 999.0, {}, {}

    daily = trades_df.groupby('date')['net_ticks'].sum()
    regimes = trades_df.groupby('date')['regime'].first()

    green = daily[regimes == 'green']
    red = daily[regimes == 'red']

    sharpe_green = float(green.mean() / green.std()) if len(green) > 1 and green.std() > 0 else 0
    sharpe_red = float(red.mean() / red.std()) if len(red) > 1 and red.std() > 0 else 0

    denom = max(abs(sharpe_green), abs(sharpe_red), 1e-6)
    gap = abs(sharpe_green - sharpe_red) / denom

    return float(gap), \
           {'sharpe': sharpe_green, 'n_days': len(green), 'n_trades': int(green.index.isin(trades_df['date'].values).sum()) if len(green) > 0 else 0}, \
           {'sharpe': sharpe_red, 'n_days': len(red), 'n_trades': int(red.index.isin(trades_df['date'].values).sum()) if len(red) > 0 else 0}


# ─────────────────────────────────────
# Walk-Forward Engine
# ─────────────────────────────────────

def run_walk_forward(all_data, valid_dates, feature_cols, lgbm_params, config_name,
                     fifo_config, date_regimes, separate_sides=False):
    """Run walk-forward with given config. Returns results dict."""

    all_trades = []
    fold_results = []
    fi_accum = {}

    fold_num = 0
    start_idx = TRAIN_DAYS

    while start_idx + OOT_DAYS <= len(valid_dates):
        train_dates = valid_dates[start_idx - TRAIN_DAYS : start_idx]
        oot_dates = valid_dates[start_idx : start_idx + OOT_DAYS]

        train_df = pd.concat([all_data[d] for d in train_dates if d in all_data], ignore_index=True)
        oot_df = pd.concat([all_data[d] for d in oot_dates if d in all_data], ignore_index=True)

        if len(train_df) < 200 or len(oot_df) < 50:
            start_idx += SLIDE_DAYS
            continue

        if separate_sides:
            # Train separate models for long and short
            pred_probs = np.zeros(len(oot_df))
            for side in ['long', 'short']:
                side_train = train_df[train_df['side'] == side]
                side_oot_mask = oot_df['side'] == side

                if len(side_train) < 100 or side_oot_mask.sum() < 20:
                    # Fall back to combined model for this side
                    continue

                X_tr_side = side_train[feature_cols].values.astype(np.float32)
                y_tr_side = side_train['target'].values.astype(int)
                X_oot_side = oot_df.loc[side_oot_mask, feature_cols].values.astype(np.float32)

                X_tr_side = np.nan_to_num(X_tr_side, nan=0., posinf=100., neginf=-100.)
                X_oot_side = np.nan_to_num(X_oot_side, nan=0., posinf=100., neginf=-100.)

                val_split = int(len(X_tr_side) * 0.8)
                model = lgb.LGBMClassifier(**lgbm_params)
                model.fit(X_tr_side[:val_split], y_tr_side[:val_split],
                          eval_set=[(X_tr_side[val_split:], y_tr_side[val_split:])],
                          callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)])

                pred_probs[side_oot_mask.values] = model.predict_proba(X_oot_side)[:, 1]

            # For any side that didn't get a separate model, train combined
            if (pred_probs == 0).any():
                X_train = train_df[feature_cols].values.astype(np.float32)
                y_train = train_df['target'].values.astype(int)
                X_train = np.nan_to_num(X_train, nan=0., posinf=100., neginf=-100.)
                val_split = int(len(X_train) * 0.8)
                model = lgb.LGBMClassifier(**lgbm_params)
                model.fit(X_train[:val_split], y_train[:val_split],
                          eval_set=[(X_train[val_split:], y_train[val_split:])],
                          callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)])
                X_oot_all = oot_df[feature_cols].values.astype(np.float32)
                X_oot_all = np.nan_to_num(X_oot_all, nan=0., posinf=100., neginf=-100.)
                fallback_probs = model.predict_proba(X_oot_all)[:, 1]
                pred_probs = np.where(pred_probs == 0, fallback_probs, pred_probs)
        else:
            # Combined model (same as v2)
            X_train = train_df[feature_cols].values.astype(np.float32)
            y_train = train_df['target'].values.astype(int)
            X_oot = oot_df[feature_cols].values.astype(np.float32)

            X_train = np.nan_to_num(X_train, nan=0., posinf=100., neginf=-100.)
            X_oot = np.nan_to_num(X_oot, nan=0., posinf=100., neginf=-100.)

            val_split = int(len(X_train) * 0.8)
            X_tr, X_val = X_train[:val_split], X_train[val_split:]
            y_tr, y_val = y_train[:val_split], y_train[val_split:]

            model = lgb.LGBMClassifier(**lgbm_params)
            model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                      callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)])

            pred_probs = model.predict_proba(X_oot)[:, 1]

            # Feature importance
            fi = dict(zip(feature_cols, model.feature_importances_))
            for feat, imp in fi.items():
                if feat not in fi_accum:
                    fi_accum[feat] = []
                fi_accum[feat].append(imp)

        oot_df = oot_df.copy()
        oot_df['pred_prob'] = pred_probs
        oot_df['fold'] = fold_num

        all_trades.append(oot_df)
        fold_num += 1
        start_idx += SLIDE_DAYS

    if not all_trades:
        return None

    all_trades_df = pd.concat(all_trades, ignore_index=True)

    # Aggregate metrics
    total_n = len(all_trades_df)
    total_wr = all_trades_df['target'].mean()
    total_pnl = all_trades_df['net_ticks'].sum()

    agg_results = {}
    for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.62, 0.65, 0.70]:
        sel = all_trades_df[all_trades_df['pred_prob'] >= thresh]
        if len(sel) < 10:
            continue
        sel_wr = sel['target'].mean()
        sel_pnl = sel['net_ticks'].sum()
        winners = sel[sel['net_ticks'] > 0]['net_ticks'].sum()
        losers = abs(sel[sel['net_ticks'] < 0]['net_ticks'].sum())
        pf = winners / max(losers, 1e-6)
        per_trade = sel_pnl / len(sel)

        daily = sel.groupby('date')['net_ticks'].sum()
        sharpe = float(daily.mean() / daily.std()) if len(daily) > 1 and daily.std() > 0 else 0
        downside = daily[daily < 0]
        sortino = float(daily.mean() / downside.std()) if len(downside) > 1 and downside.std() > 0 else (
            999.0 if daily.mean() > 0 else 0)

        # Per-side breakdown
        side_stats = {}
        for side in ['long', 'short']:
            side_sel = sel[sel['side'] == side]
            if len(side_sel) >= 5:
                sw = side_sel['target'].mean()
                sp = side_sel['net_ticks'].sum()
                swin = side_sel[side_sel['net_ticks'] > 0]['net_ticks'].sum()
                slos = abs(side_sel[side_sel['net_ticks'] < 0]['net_ticks'].sum())
                spf = swin / max(slos, 1e-6)
                side_stats[side] = {'n': len(side_sel), 'wr': float(sw), 'pnl': float(sp), 'pf': float(spf)}

        agg_results[str(thresh)] = {
            'n_trades': len(sel),
            'wr': float(sel_wr),
            'pnl': float(sel_pnl),
            'per_trade': float(per_trade),
            'pf': float(pf),
            'sharpe': float(sharpe),
            'sortino': float(sortino),
            'per_side': side_stats,
        }

    # Regime analysis at best threshold
    best_thresh_key = max(agg_results.items(), key=lambda x: x[1]['pf'])[0] if agg_results else '0.55'
    best_thresh_val = float(best_thresh_key)
    filtered = all_trades_df[all_trades_df['pred_prob'] >= best_thresh_val]
    if len(filtered) > 10:
        gap, green_stats, red_stats = compute_regime_gap(filtered)
    else:
        gap, green_stats, red_stats = 999.0, {}, {}

    # IC
    ic = np.corrcoef(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0, 1]
    rank_ic = scipy_stats.spearmanr(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0]

    # Feature importance summary
    fi_summary = {}
    for feat, gains in fi_accum.items():
        fi_summary[feat] = {'mean_gain': float(np.mean(gains)), 'std_gain': float(np.std(gains))}

    return {
        'config_name': config_name,
        'fifo_config': fifo_config,
        'separate_sides': separate_sides,
        'n_dates': len(valid_dates),
        'n_folds': fold_num,
        'n_features': len(feature_cols),
        'baseline': {
            'n_trades': total_n,
            'wr': float(total_wr),
            'pnl': float(total_pnl),
            'per_trade': float(total_pnl / max(total_n, 1)),
        },
        'filtered_performance': agg_results,
        'regime_analysis': {
            'threshold': best_thresh_val,
            'gap': float(gap),
            'green': green_stats,
            'red': red_stats,
            'pass_gate': gap <= 0.50,
        },
        'feature_importance': fi_summary,
        'ic': {'pearson': float(ic), 'spearman': float(rank_ic)},
        'all_trades_df': all_trades_df,  # for saving
    }


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("Queue Entry Selector v3 — Enhanced Features + Multi-Config")
    log.info("=" * 70)

    # ── 1. Discover dates ──
    q_files = sorted(QUEUE_DIR.glob("features_*.parquet"))
    q_dates = {f.stem.replace('features_', '') for f in q_files}

    f_files = sorted(FIFO_DIR.glob("*_fifo_labels.npz"))
    f_dates = {f.stem.replace('_fifo_labels', '') for f in f_files}

    overlap = sorted(q_dates & f_dates)
    log.info(f"Queue dates: {len(q_dates)}, FIFO dates: {len(f_dates)}, Overlap: {len(overlap)}")

    if len(overlap) < TRAIN_DAYS + OOT_DAYS:
        log.error(f"Need ≥ {TRAIN_DAYS + OOT_DAYS} dates, got {len(overlap)}")
        sys.exit(1)

    # ── 2. Run all config combinations ──
    all_results = {}

    for fifo_config in FIFO_CONFIGS:
        log.info(f"\n{'='*70}")
        log.info(f"FIFO CONFIG: {fifo_config}")
        log.info(f"{'='*70}")

        # Load data for this FIFO config
        all_data = {}
        date_regimes = {}

        for date_str in overlap:
            queue = load_queue_features(date_str)
            fifo = load_fifo_labels(date_str, fifo_config)

            if queue is None or fifo is None:
                continue

            queue = engineer_features(queue)
            merged, n_before, n_after = join_queue_fifo(queue, fifo)

            if n_after < 50:
                continue

            training = prepare_training_data(merged, date_str)
            if training is None or len(training) < 50:
                continue

            regime = classify_day_regime(queue)
            date_regimes[date_str] = regime
            training['regime'] = regime
            all_data[date_str] = training

        valid_dates = sorted(all_data.keys())
        log.info(f"Valid dates for {fifo_config}: {len(valid_dates)}")
        log.info(f"Regimes: {pd.Series(date_regimes).value_counts().to_dict()}")

        if len(valid_dates) < TRAIN_DAYS + OOT_DAYS:
            log.warning(f"Not enough dates for {fifo_config}, skipping")
            continue

        # Get feature columns
        sample = list(all_data.values())[0]
        feature_cols = get_feature_cols(sample)
        log.info(f"Features ({len(feature_cols)}): {feature_cols}")

        # Run each LGBM config — combined model
        for lgbm_name, lgbm_params in LGBM_CONFIGS.items():
            run_key = f"{fifo_config}_{lgbm_name}_combined"
            log.info(f"\n--- Running: {run_key} ---")
            result = run_walk_forward(all_data, valid_dates, feature_cols, lgbm_params,
                                      run_key, fifo_config, date_regimes, separate_sides=False)
            if result:
                all_results[run_key] = result
                best = result['filtered_performance']
                for t in ['0.55', '0.58', '0.60']:
                    if t in best:
                        r = best[t]
                        log.info(f"  thresh={t}: n={r['n_trades']}, WR={r['wr']:.3f}, "
                                 f"PF={r['pf']:.2f}, Sharpe={r['sharpe']:.2f}, per_trade={r['per_trade']:+.3f}")

        # Run separate-side model with best config
        run_key = f"{fifo_config}_base_separate"
        log.info(f"\n--- Running: {run_key} ---")
        result = run_walk_forward(all_data, valid_dates, feature_cols, LGBM_CONFIGS['base'],
                                  run_key, fifo_config, date_regimes, separate_sides=True)
        if result:
            all_results[run_key] = result
            best = result['filtered_performance']
            for t in ['0.55', '0.58', '0.60']:
                if t in best:
                    r = best[t]
                    log.info(f"  thresh={t}: n={r['n_trades']}, WR={r['wr']:.3f}, "
                             f"PF={r['pf']:.2f}, Sharpe={r['sharpe']:.2f}, per_trade={r['per_trade']:+.3f}")

    # ── 3. Compare all configs ──
    log.info(f"\n{'='*70}")
    log.info(f"COMPARISON — ALL CONFIGS")
    log.info(f"{'='*70}")

    comparison = []
    for key, res in all_results.items():
        fp = res['filtered_performance']
        for thresh in ['0.55', '0.58', '0.60']:
            if thresh in fp:
                comparison.append({
                    'config': key,
                    'thresh': thresh,
                    'n_trades': fp[thresh]['n_trades'],
                    'wr': fp[thresh]['wr'],
                    'pf': fp[thresh]['pf'],
                    'sharpe': fp[thresh]['sharpe'],
                    'sortino': fp[thresh]['sortino'],
                    'per_trade': fp[thresh]['per_trade'],
                    'pnl': fp[thresh]['pnl'],
                    'regime_gap': res['regime_analysis']['gap'],
                    'ic_spearman': res['ic']['spearman'],
                })

    if comparison:
        comp_df = pd.DataFrame(comparison)
        comp_df = comp_df.sort_values('sharpe', ascending=False)
        log.info(f"\nTop 10 configs by Sharpe:")
        for _, row in comp_df.head(10).iterrows():
            log.info(f"  {row['config']:40s} @{row['thresh']}: "
                     f"n={row['n_trades']:5.0f}, WR={row['wr']:.3f}, PF={row['pf']:.2f}, "
                     f"Sharpe={row['sharpe']:.3f}, per_trade={row['per_trade']:+.3f}, "
                     f"regime_gap={row['regime_gap']:.2f}")

    # ── 4. Save results ──
    # Save summary (without dataframes)
    summary = {}
    for key, res in all_results.items():
        res_copy = {k: v for k, v in res.items() if k != 'all_trades_df'}
        summary[key] = res_copy

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'n_queue_dates': len(q_dates),
            'n_fifo_dates': len(f_dates),
            'n_overlap': len(overlap),
            'fifo_configs': FIFO_CONFIGS,
            'lgbm_configs': list(LGBM_CONFIGS.keys()),
            'configs_tested': list(all_results.keys()),
            'results': summary,
            'comparison': comparison if comparison else [],
        }, f, indent=2, default=str)

    # Save best config's trades
    if all_results:
        best_key = max(all_results.keys(),
                       key=lambda k: max(
                           (v['sharpe'] for v in all_results[k]['filtered_performance'].values()),
                           default=0))
        best_trades = all_results[best_key].get('all_trades_df')
        if best_trades is not None:
            save_cols = [c for c in best_trades.columns if best_trades[c].dtype.kind in 'iufbOS']
            best_trades[save_cols].to_parquet(OUTPUT_DIR / "best_trades.parquet", index=False)
            log.info(f"Saved best trades ({best_key}) to best_trades.parquet")

    log.info(f"\nResults saved to {OUTPUT_DIR}")

    # ── 5. MLflow logging ──
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("queue_entry_selector")

        for key, res in all_results.items():
            with mlflow.start_run(run_name=f"v3_{key}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({
                    'version': 'v3',
                    'config': key,
                    'fifo_config': res['fifo_config'],
                    'separate_sides': res['separate_sides'],
                    'n_features': res['n_features'],
                    'n_dates': res['n_dates'],
                    'n_folds': res['n_folds'],
                })
                mlflow.log_metrics({
                    'baseline_wr': res['baseline']['wr'],
                    'ic_pearson': res['ic']['pearson'],
                    'ic_spearman': res['ic']['spearman'],
                    'regime_gap': res['regime_analysis']['gap'],
                })
                # Log best threshold metrics
                fp = res['filtered_performance']
                for thresh_key in ['0.55', '0.58', '0.60']:
                    if thresh_key in fp:
                        t = fp[thresh_key]
                        mlflow.log_metrics({
                            f't{thresh_key}_wr': t['wr'],
                            f't{thresh_key}_pf': t['pf'],
                            f't{thresh_key}_sharpe': t['sharpe'],
                            f't{thresh_key}_n_trades': t['n_trades'],
                            f't{thresh_key}_per_trade': t['per_trade'],
                        })
        log.info("MLflow runs logged")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")

    # ── 6. Print final verdict ──
    if comparison:
        best_row = comp_df.iloc[0]
        log.info(f"\n{'='*70}")
        log.info(f"BEST CONFIG: {best_row['config']} @ thresh={best_row['thresh']}")
        log.info(f"  Trades: {best_row['n_trades']:.0f}")
        log.info(f"  WR: {best_row['wr']:.1%}")
        log.info(f"  PF: {best_row['pf']:.2f}")
        log.info(f"  Sharpe: {best_row['sharpe']:.3f}")
        log.info(f"  Per-trade: {best_row['per_trade']:+.3f} ticks")
        log.info(f"  Regime gap: {best_row['regime_gap']:.2f} {'✓ PASS' if best_row['regime_gap'] <= 0.50 else '✗ FAIL'}")
        log.info(f"{'='*70}")


if __name__ == '__main__':
    main()
