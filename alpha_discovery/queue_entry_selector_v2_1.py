#!/usr/bin/env python3
"""
queue_entry_selector_v2_1.py — Queue Entry Selector v2.1
=========================================================

Improvements over v2:
  1. Multi-config: runs BOTH tp4sl3 AND tp8sl5 FIFO configs
  2. New features: time_of_day, session_pct, ofi_ratio_10s_1s, queue_pressure,
     cancel_intensity, add_intensity, renewal_ratio
  3. Hyperparameter variations: default/shallow/deep LightGBM configs, pick best per-fold
  4. Per-fold adaptive threshold: optimize PF on last 20% of training data

Walk-forward: 25d train, 5d OOT, 5d slide (SLIDING per HC #0)
Regime gate: |Sharpe_green - Sharpe_red| / max(...) <= 0.50 (HC #428)
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
OUTPUT_DIR = BASE / "output" / "queue_entry_selector_v2_1"
LOG_PATH = BASE / "logs" / "queue_entry_selector_v2_1.log"

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

# FIFO configs to evaluate (multi-config support)
FIFO_CONFIGS = ['tp4sl3', 'tp8sl5']

# Walk-forward
TRAIN_DAYS = 25
OOT_DAYS = 5
SLIDE_DAYS = 5

# Session timing (ET): 09:30 to 16:00
SESSION_START_SECONDS = 9 * 3600 + 30 * 60  # 09:30 ET in seconds since midnight
SESSION_END_SECONDS = 16 * 3600             # 16:00 ET in seconds since midnight
TOTAL_SESSION_SECONDS = SESSION_END_SECONDS - SESSION_START_SECONDS  # 23400 seconds

# Features to use (v2 base + v2.1 additions)
RAW_FEATURES = [
    'ofi_10s', 'ofi_5s', 'ofi_1s',
    'top_imbalance', 'microprice_offset_ticks',
    'bid_qty_at_touch', 'ask_qty_at_touch',
    'bid_trade_rate_1s', 'ask_trade_rate_1s',
    'bid_add_rate_1s', 'ask_add_rate_1s',
    'bid_cancel_rate_1s', 'ask_cancel_rate_1s',
    'bid_level_age_s', 'ask_level_age_s',
]

# Direction-sensitive features that need sign flip for shorts
FLIP_FEATURES = ['ofi_10s', 'ofi_5s', 'ofi_1s', 'top_imbalance', 'microprice_offset_ticks',
                  'queue_diff', 'net_flow_diff', 'trade_imbalance', 'ofi_momentum',
                  'ofi_ratio_10s_1s', 'queue_pressure']

# Swap features: bid<->ask for shorts
SWAP_PAIRS = [
    ('bid_qty_at_touch', 'ask_qty_at_touch'),
    ('bid_trade_rate_1s', 'ask_trade_rate_1s'),
    ('bid_add_rate_1s', 'ask_add_rate_1s'),
    ('bid_cancel_rate_1s', 'ask_cancel_rate_1s'),
    ('bid_level_age_s', 'ask_level_age_s'),
]

# ── LightGBM hyperparameter variations ──
LGBM_CONFIGS = {
    'default': {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 200,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': 5,
        'min_child_samples': 20,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
    },
    'shallow': {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 100,
        'learning_rate': 0.1,
        'num_leaves': 15,
        'max_depth': 3,
        'min_child_samples': 50,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
    },
    'deep': {
        'objective': 'binary',
        'metric': 'auc',
        'n_estimators': 300,
        'learning_rate': 0.03,
        'num_leaves': 63,
        'max_depth': 7,
        'min_child_samples': 10,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'random_state': 42,
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
    """Load FIFO labels for a date and config. Returns DataFrame with ts_ns + label columns."""
    path = FIFO_DIR / f"{date_str}_fifo_labels.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)

    prefix = fifo_config
    required = [f'{prefix}_long_filled', f'{prefix}_long_net_ticks',
                f'{prefix}_short_filled', f'{prefix}_short_net_ticks']

    if not all(k in data for k in ['ts_ns'] + required):
        return None

    df = pd.DataFrame({
        'ts_ns': data['ts_ns'],
        'long_filled': data[f'{prefix}_long_filled'],
        'long_net_ticks': data[f'{prefix}_long_net_ticks'],
        'long_hit_tp': data[f'{prefix}_long_hit_tp'],
        'long_exit_reason': data[f'{prefix}_long_exit_reason'],
        'short_filled': data[f'{prefix}_short_filled'],
        'short_net_ticks': data[f'{prefix}_short_net_ticks'],
        'short_hit_tp': data[f'{prefix}_short_hit_tp'],
        'short_exit_reason': data[f'{prefix}_short_exit_reason'],
    })
    return df


def join_queue_fifo(queue_df, fifo_df):
    """Join queue features to FIFO labels by nearest timestamp."""
    queue_df = queue_df.sort_values('ts_ns').reset_index(drop=True)
    fifo_df = fifo_df.sort_values('ts_ns').reset_index(drop=True)

    merged = pd.merge_asof(
        fifo_df, queue_df,
        on='ts_ns',
        direction='backward',
        tolerance=1_500_000_000  # 1.5 seconds tolerance
    )

    n_before = len(merged)
    merged = merged.dropna(subset=['ofi_10s'])
    n_after = len(merged)

    return merged, n_before, n_after


def engineer_features(df):
    """Add derived features (v2 base + v2.1 new features)."""
    # ── v2 features ──
    bid_q = df['bid_qty_at_touch'].clip(lower=1)
    ask_q = df['ask_qty_at_touch'].clip(lower=1)
    df['queue_ratio'] = bid_q / (bid_q + ask_q)
    df['queue_diff'] = df['bid_qty_at_touch'] - df['ask_qty_at_touch']

    bid_net = df.get('bid_add_rate_1s', 0) - df.get('bid_cancel_rate_1s', 0)
    ask_net = df.get('ask_add_rate_1s', 0) - df.get('ask_cancel_rate_1s', 0)
    df['net_flow_diff'] = bid_net - ask_net

    df['trade_imbalance'] = df['bid_trade_rate_1s'] - df['ask_trade_rate_1s']
    df['ofi_momentum'] = df['ofi_10s'] - df['ofi_1s']
    df['level_age_diff'] = df['bid_level_age_s'] - df['ask_level_age_s']

    # ── v2.1 NEW features ──

    # time_of_day: seconds since 09:30 ET
    # ts_ns is exchange nanosecond timestamp. Compute seconds-of-day, then offset from 09:30.
    ts_seconds_of_day = (df['ts_ns'] % (86400 * 1_000_000_000)) / 1_000_000_000
    # Exchange timestamps are in ET. 09:30 ET = 34200 seconds from midnight.
    df['time_of_day'] = (ts_seconds_of_day - SESSION_START_SECONDS).clip(lower=0)

    # session_pct: 0-1 normalized position in session
    df['session_pct'] = (df['time_of_day'] / TOTAL_SESSION_SECONDS).clip(0, 1)

    # ofi_ratio_10s_1s: momentum persistence
    df['ofi_ratio_10s_1s'] = df['ofi_10s'] / (df['ofi_1s'].abs() + 1e-6)

    # queue_pressure: normalized bid-ask queue imbalance
    df['queue_pressure'] = (df['bid_qty_at_touch'] - df['ask_qty_at_touch']) / \
                           (df['bid_qty_at_touch'] + df['ask_qty_at_touch'] + 1e-6)

    # cancel_intensity: total cancel activity
    df['cancel_intensity'] = df['bid_cancel_rate_1s'] + df['ask_cancel_rate_1s']

    # add_intensity: total add activity
    df['add_intensity'] = df['bid_add_rate_1s'] + df['ask_add_rate_1s']

    # renewal_ratio: are orders being replaced?
    df['renewal_ratio'] = df['add_intensity'] / (df['cancel_intensity'] + 1e-6)

    return df


def prepare_training_data(merged_df, date_str):
    """Create long + short training rows with direction-aware feature swapping."""
    rows = []

    for side in ['long', 'short']:
        filled_col = f'{side}_filled'
        net_col = f'{side}_net_ticks'

        mask = merged_df[filled_col] == True
        subset = merged_df[mask].copy()

        if len(subset) == 0:
            continue

        subset['target'] = (subset[net_col] > 0).astype(int)
        subset['net_ticks'] = subset[net_col].astype(float)
        subset['side'] = side
        subset['date'] = date_str

        feature_cols = RAW_FEATURES + [
            'queue_ratio', 'queue_diff', 'net_flow_diff',
            'trade_imbalance', 'ofi_momentum', 'level_age_diff',
            # v2.1 new features
            'time_of_day', 'session_pct', 'ofi_ratio_10s_1s',
            'queue_pressure', 'cancel_intensity', 'add_intensity', 'renewal_ratio',
        ]

        row = subset[['ts_ns', 'target', 'net_ticks', 'side', 'date'] +
                     [c for c in feature_cols if c in subset.columns]].copy()

        if side == 'short':
            for feat in FLIP_FEATURES:
                if feat in row.columns:
                    row[feat] = -row[feat]
            for bid_feat, ask_feat in SWAP_PAIRS:
                if bid_feat in row.columns and ask_feat in row.columns:
                    row[bid_feat], row[ask_feat] = row[ask_feat].copy(), row[bid_feat].copy()

        rows.append(row)

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
           {'sharpe': sharpe_green, 'n_days': len(green), 'n_trades': len(trades_df[trades_df['regime']=='green'])}, \
           {'sharpe': sharpe_red, 'n_days': len(red), 'n_trades': len(trades_df[trades_df['regime']=='red'])}


def find_adaptive_threshold(model, X_val, y_val, net_ticks_val):
    """Find threshold that maximizes PF on validation data."""
    pred_prob = model.predict_proba(X_val)[:, 1]
    best_thresh = 0.50
    best_pf = 0.0

    for thresh in np.arange(0.45, 0.75, 0.01):
        mask = pred_prob >= thresh
        if mask.sum() < 10:
            continue
        sel_ticks = net_ticks_val[mask]
        winners = sel_ticks[sel_ticks > 0].sum()
        losers = abs(sel_ticks[sel_ticks < 0].sum())
        pf = winners / max(losers, 1e-6)
        if pf > best_pf:
            best_pf = pf
            best_thresh = float(thresh)

    return best_thresh, best_pf


def train_best_model(X_tr, y_tr, X_val, y_val, feature_cols):
    """Train all 3 LGBM configs, return best model by OOT AUC on validation set."""
    best_model = None
    best_auc = -1.0
    best_config_name = 'default'

    for config_name, params in LGBM_CONFIGS.items():
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)]
        )

        # Evaluate AUC on validation set
        pred_prob = model.predict_proba(X_val)[:, 1]
        from sklearn.metrics import roc_auc_score
        try:
            auc = roc_auc_score(y_val, pred_prob)
        except ValueError:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_config_name = config_name

    return best_model, best_config_name, best_auc


def run_for_config(fifo_config, valid_dates, all_data, feature_cols):
    """Run walk-forward for a single FIFO config. Returns results dict."""
    log.info(f"\n{'='*70}")
    log.info(f"FIFO CONFIG: {fifo_config}")
    log.info(f"Walk-forward: {TRAIN_DAYS}d train, {OOT_DAYS}d OOT, {SLIDE_DAYS}d slide (SLIDING)")
    log.info(f"{'='*70}")

    all_trades = []
    fold_results = []
    fi_accum = {}
    config_picks = {}  # count of which LGBM config was picked per fold

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

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df['target'].values.astype(int)
        net_ticks_train = train_df['net_ticks'].values.astype(np.float32)
        X_oot = oot_df[feature_cols].values.astype(np.float32)
        y_oot = oot_df['target'].values.astype(int)

        # Replace inf/nan
        X_train = np.nan_to_num(X_train, nan=0., posinf=100., neginf=-100.)
        X_oot = np.nan_to_num(X_oot, nan=0., posinf=100., neginf=-100.)

        # Split last 20% of training for early stopping AND adaptive threshold
        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]
        net_ticks_val = net_ticks_train[val_split:]

        # Train best model across 3 hyperparameter configs
        model, config_name, val_auc = train_best_model(X_tr, y_tr, X_val, y_val, feature_cols)
        config_picks[config_name] = config_picks.get(config_name, 0) + 1

        # Compute adaptive threshold on validation data
        adaptive_thresh, adaptive_pf = find_adaptive_threshold(model, X_val, y_val, net_ticks_val)

        # Predict on OOT
        pred_prob = model.predict_proba(X_oot)[:, 1]
        oot_df = oot_df.copy()
        oot_df['pred_prob'] = pred_prob

        # Feature importance
        fi = dict(zip(feature_cols, model.feature_importances_))
        for feat, imp in fi.items():
            if feat not in fi_accum:
                fi_accum[feat] = []
            fi_accum[feat].append(imp)

        # Evaluate at fixed thresholds
        baseline_wr = y_oot.mean()
        baseline_pnl = oot_df['net_ticks'].sum()
        baseline_n = len(oot_df)

        best_threshold = 0.5
        best_pf = 0
        threshold_results = {}

        for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
            selected = oot_df[oot_df['pred_prob'] >= thresh]
            if len(selected) < 5:
                continue
            sel_wr = selected['target'].mean()
            sel_pnl = selected['net_ticks'].sum()
            winners = selected[selected['net_ticks'] > 0]['net_ticks'].sum()
            losers = abs(selected[selected['net_ticks'] < 0]['net_ticks'].sum())
            pf = winners / max(losers, 1e-6)

            threshold_results[str(thresh)] = {
                'n_trades': int(len(selected)),
                'wr': float(sel_wr),
                'pnl': float(sel_pnl),
                'pf': float(pf),
            }
            if pf > best_pf:
                best_pf = pf
                best_threshold = thresh

        # Evaluate adaptive threshold on OOT
        adaptive_selected = oot_df[oot_df['pred_prob'] >= adaptive_thresh]
        adaptive_oot = {}
        if len(adaptive_selected) >= 5:
            sel_wr = adaptive_selected['target'].mean()
            sel_pnl = adaptive_selected['net_ticks'].sum()
            winners = adaptive_selected[adaptive_selected['net_ticks'] > 0]['net_ticks'].sum()
            losers = abs(adaptive_selected[adaptive_selected['net_ticks'] < 0]['net_ticks'].sum())
            pf = winners / max(losers, 1e-6)
            adaptive_oot = {
                'threshold': float(adaptive_thresh),
                'n_trades': int(len(adaptive_selected)),
                'wr': float(sel_wr),
                'pnl': float(sel_pnl),
                'pf': float(pf),
            }

        oot_regime = oot_df.groupby('date')['regime'].first().value_counts().to_dict()

        fold_res = {
            'fold': fold_num,
            'train_dates': train_dates,
            'oot_dates': oot_dates,
            'oot_regime': oot_regime,
            'n_train': len(train_df),
            'n_oot': len(oot_df),
            'baseline_wr': float(baseline_wr),
            'baseline_pnl': float(baseline_pnl),
            'baseline_n': baseline_n,
            'lgbm_config': config_name,
            'val_auc': float(val_auc),
            'adaptive_threshold': float(adaptive_thresh),
            'adaptive_val_pf': float(adaptive_pf),
            'adaptive_oot': adaptive_oot,
            'thresholds': threshold_results,
            'best_threshold': best_threshold,
            'best_pf': float(best_pf),
        }
        fold_results.append(fold_res)

        # Collect OOT trades
        oot_df['fold'] = fold_num
        oot_df['adaptive_thresh'] = adaptive_thresh
        all_trades.append(oot_df)

        # Log fold summary
        best_t = threshold_results.get(str(best_threshold), {})
        adp = adaptive_oot
        log.info(f"  Fold {fold_num:2d} | {config_name:8s} AUC={val_auc:.3f} | OOT={','.join(oot_dates)} | "
                 f"baseline: WR={baseline_wr:.3f}, PnL={baseline_pnl:+.1f}t | "
                 f"best@{best_threshold}: n={best_t.get('n_trades','?')}, PF={best_t.get('pf',0):.2f} | "
                 f"adaptive@{adaptive_thresh:.2f}: n={adp.get('n_trades','?')}, PF={adp.get('pf',0):.2f}")

        fold_num += 1
        start_idx += SLIDE_DAYS

    if not all_trades:
        log.error(f"No valid folds for {fifo_config}")
        return None

    all_trades_df = pd.concat(all_trades, ignore_index=True)

    # ── Aggregate metrics ──
    total_n = len(all_trades_df)
    total_wr = all_trades_df['target'].mean()
    total_pnl = all_trades_df['net_ticks'].sum()

    log.info(f"\n--- BASELINE (no filter) [{fifo_config}] ---")
    log.info(f"  Total entries: {total_n}")
    log.info(f"  WR: {total_wr:.3f}")
    log.info(f"  Total PnL: {total_pnl:+.1f} ticks")
    log.info(f"  Per-trade: {total_pnl/max(total_n,1):+.3f} ticks")

    # Fixed thresholds on aggregate
    log.info(f"\n--- FILTERED ENTRY PERFORMANCE [{fifo_config}] ---")
    agg_results = {}
    for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
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

        agg_results[str(thresh)] = {
            'n_trades': int(len(sel)),
            'wr': float(sel_wr),
            'pnl': float(sel_pnl),
            'per_trade': float(per_trade),
            'pf': float(pf),
            'sharpe': float(sharpe),
            'sortino': float(sortino),
        }

        log.info(f"  thresh={thresh:.2f}: n={len(sel):5d}, WR={sel_wr:.3f}, "
                 f"PnL={sel_pnl:+.1f}t, per_trade={per_trade:+.3f}t, "
                 f"PF={pf:.3f}, Sharpe={sharpe:.2f}, Sortino={sortino:.2f}")

    # ── Adaptive threshold aggregate ──
    log.info(f"\n--- ADAPTIVE THRESHOLD AGGREGATE [{fifo_config}] ---")
    # Each OOT row has its fold-specific adaptive threshold stored
    adaptive_sel = all_trades_df[all_trades_df['pred_prob'] >= all_trades_df['adaptive_thresh']]
    adaptive_agg = {}
    if len(adaptive_sel) >= 10:
        awr = adaptive_sel['target'].mean()
        apnl = adaptive_sel['net_ticks'].sum()
        awin = adaptive_sel[adaptive_sel['net_ticks'] > 0]['net_ticks'].sum()
        alos = abs(adaptive_sel[adaptive_sel['net_ticks'] < 0]['net_ticks'].sum())
        apf = awin / max(alos, 1e-6)
        apt = apnl / len(adaptive_sel)

        daily_a = adaptive_sel.groupby('date')['net_ticks'].sum()
        asharpe = float(daily_a.mean() / daily_a.std()) if len(daily_a) > 1 and daily_a.std() > 0 else 0
        adownside = daily_a[daily_a < 0]
        asortino = float(daily_a.mean() / adownside.std()) if len(adownside) > 1 and adownside.std() > 0 else (
            999.0 if daily_a.mean() > 0 else 0)

        adaptive_agg = {
            'n_trades': int(len(adaptive_sel)),
            'wr': float(awr),
            'pnl': float(apnl),
            'per_trade': float(apt),
            'pf': float(apf),
            'sharpe': float(asharpe),
            'sortino': float(asortino),
            'mean_threshold': float(all_trades_df['adaptive_thresh'].mean()),
            'std_threshold': float(all_trades_df['adaptive_thresh'].std()),
        }
        log.info(f"  Adaptive (mean thresh={adaptive_agg['mean_threshold']:.3f} +/- {adaptive_agg['std_threshold']:.3f}): "
                 f"n={adaptive_agg['n_trades']}, WR={awr:.3f}, PnL={apnl:+.1f}t, "
                 f"PF={apf:.3f}, Sharpe={asharpe:.2f}, Sortino={asortino:.2f}")
    else:
        log.info("  Insufficient trades for adaptive aggregate")

    # ── Per-side breakdown ──
    best_agg_thresh = max(agg_results.items(), key=lambda x: x[1]['pf'])[0] if agg_results else '0.55'
    log.info(f"\n--- PER-SIDE BREAKDOWN [{fifo_config}] (best threshold = {best_agg_thresh}) ---")
    best_thresh_val_tmp = float(best_agg_thresh)
    side_breakdown = {}
    for side in ['long', 'short']:
        side_df = all_trades_df[(all_trades_df['side'] == side) &
                                (all_trades_df['pred_prob'] >= best_thresh_val_tmp)]
        if len(side_df) < 5:
            log.info(f"  {side.upper()}: insufficient trades ({len(side_df)})")
            continue
        sw = side_df['target'].mean()
        sp = side_df['net_ticks'].sum()
        swin = side_df[side_df['net_ticks'] > 0]['net_ticks'].sum()
        slos = abs(side_df[side_df['net_ticks'] < 0]['net_ticks'].sum())
        spf = swin / max(slos, 1e-6)
        side_breakdown[side] = {
            'n_trades': int(len(side_df)),
            'wr': float(sw),
            'pnl': float(sp),
            'pf': float(spf),
        }
        log.info(f"  {side.upper()}: n={len(side_df)}, WR={sw:.3f}, PnL={sp:+.1f}t, PF={spf:.3f}")

    # ── Regime analysis ──
    log.info(f"\n--- REGIME ANALYSIS [{fifo_config}] ---")
    best_thresh_val = float(best_agg_thresh)

    filtered = all_trades_df[all_trades_df['pred_prob'] >= best_thresh_val]
    if len(filtered) > 10:
        gap, green_stats, red_stats = compute_regime_gap(filtered)
        log.info(f"  At threshold {best_thresh_val}:")
        log.info(f"  Green: Sharpe={green_stats.get('sharpe',0):.2f}, n_days={green_stats.get('n_days',0)}")
        log.info(f"  Red:   Sharpe={red_stats.get('sharpe',0):.2f}, n_days={red_stats.get('n_days',0)}")
        log.info(f"  Regime gap: {gap:.3f} {'PASS' if gap <= 0.50 else 'FAIL'}")
    else:
        gap = 999.0
        green_stats = {}
        red_stats = {}

    # ── Feature importance ──
    log.info(f"\n--- FEATURE IMPORTANCE [{fifo_config}] (mean gain across folds) ---")
    fi_summary = {}
    for feat, gains in fi_accum.items():
        fi_summary[feat] = {
            'mean_gain': float(np.mean(gains)),
            'std_gain': float(np.std(gains)),
        }

    sorted_features = sorted(fi_summary.items(), key=lambda x: x[1]['mean_gain'], reverse=True)
    for rank, (feat, fi_stat) in enumerate(sorted_features):
        fi_stat['rank'] = rank + 1
        if rank < 20:
            log.info(f"  {rank+1:2d}. {feat:35s} gain={fi_stat['mean_gain']:.1f} +/- {fi_stat['std_gain']:.1f}")

    # ── IC analysis ──
    ic = np.corrcoef(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0, 1]
    rank_ic = scipy_stats.spearmanr(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0]
    log.info(f"\n--- INFORMATION COEFFICIENT [{fifo_config}] ---")
    log.info(f"  Pearson IC (prob vs net_ticks): {ic:.4f}")
    log.info(f"  Rank IC (Spearman): {rank_ic:.4f}")

    # ── LGBM config picks ──
    log.info(f"\n--- LGBM CONFIG SELECTION [{fifo_config}] ---")
    for cfg_name, cnt in sorted(config_picks.items(), key=lambda x: -x[1]):
        log.info(f"  {cfg_name}: {cnt} folds ({100*cnt/max(fold_num,1):.0f}%)")

    # Save trades parquet
    trades_path = OUTPUT_DIR / f"all_oot_trades_{fifo_config}.parquet"
    save_cols = [c for c in all_trades_df.columns if all_trades_df[c].dtype.kind in 'iufbOS']
    all_trades_df[save_cols].to_parquet(trades_path, index=False)
    log.info(f"Saved {len(all_trades_df)} OOT trades to {trades_path}")

    return {
        'fifo_config': fifo_config,
        'n_folds': fold_num,
        'lgbm_config_picks': config_picks,
        'baseline': {
            'n_trades': int(total_n),
            'wr': float(total_wr),
            'pnl': float(total_pnl),
            'per_trade': float(total_pnl / max(total_n, 1)),
        },
        'filtered_performance': agg_results,
        'adaptive_threshold_aggregate': adaptive_agg,
        'side_breakdown': side_breakdown,
        'regime_analysis': {
            'threshold': best_thresh_val,
            'gap': float(gap),
            'green': green_stats,
            'red': red_stats,
            'pass_gate': gap <= 0.50,
        },
        'feature_importance': fi_summary,
        'ic': {
            'pearson': float(ic),
            'spearman': float(rank_ic),
        },
        'fold_results': fold_results,
    }


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("Queue Entry Selector v2.1 — Multi-config + New Features + Hyper Selection")
    log.info("=" * 70)

    # ── 1. Discover dates ──
    q_files = sorted(QUEUE_DIR.glob("features_*.parquet"))
    q_dates = {f.stem.replace('features_', '') for f in q_files}

    f_files = sorted(FIFO_DIR.glob("*_fifo_labels.npz"))
    f_dates = {f.stem.replace('_fifo_labels', '') for f in f_files}

    overlap = sorted(q_dates & f_dates)
    log.info(f"Queue dates: {len(q_dates)}, FIFO dates: {len(f_dates)}, Overlap: {len(overlap)}")

    if len(overlap) < TRAIN_DAYS + OOT_DAYS:
        log.error(f"Need >= {TRAIN_DAYS + OOT_DAYS} dates, got {len(overlap)}")
        sys.exit(1)

    # ── 2. Run for each FIFO config ──
    all_config_results = {}

    for fifo_config in FIFO_CONFIGS:
        log.info(f"\n\n{'#'*70}")
        log.info(f"# Loading data for FIFO config: {fifo_config}")
        log.info(f"{'#'*70}")

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
            log.error(f"Only {len(valid_dates)} valid dates for {fifo_config} — need >= {TRAIN_DAYS + OOT_DAYS}")
            continue

        # Define feature columns
        sample = list(all_data.values())[0]
        feature_cols = [c for c in RAW_FEATURES + [
            'queue_ratio', 'queue_diff', 'net_flow_diff',
            'trade_imbalance', 'ofi_momentum', 'level_age_diff',
            'time_of_day', 'session_pct', 'ofi_ratio_10s_1s',
            'queue_pressure', 'cancel_intensity', 'add_intensity', 'renewal_ratio',
        ] if c in sample.columns]
        log.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

        config_result = run_for_config(fifo_config, valid_dates, all_data, feature_cols)
        if config_result is not None:
            config_result['date_regimes'] = date_regimes
            config_result['n_dates'] = len(valid_dates)
            config_result['n_features'] = len(feature_cols)
            config_result['feature_cols'] = feature_cols
            all_config_results[fifo_config] = config_result

    # ── 3. Cross-config comparison ──
    log.info(f"\n\n{'='*70}")
    log.info("CROSS-CONFIG COMPARISON")
    log.info(f"{'='*70}")

    for cfg_name, cfg_res in all_config_results.items():
        bl = cfg_res['baseline']
        best_thresh = max(cfg_res['filtered_performance'].items(),
                         key=lambda x: x[1]['pf'])[0] if cfg_res['filtered_performance'] else 'N/A'
        best = cfg_res['filtered_performance'].get(best_thresh, {})
        adp = cfg_res.get('adaptive_threshold_aggregate', {})
        log.info(f"\n  {cfg_name}:")
        log.info(f"    Baseline: n={bl['n_trades']}, WR={bl['wr']:.3f}, PnL={bl['pnl']:+.1f}t")
        log.info(f"    Best fixed thresh={best_thresh}: n={best.get('n_trades','?')}, "
                 f"WR={best.get('wr',0):.3f}, PF={best.get('pf',0):.3f}, "
                 f"Sharpe={best.get('sharpe',0):.2f}")
        log.info(f"    Adaptive: n={adp.get('n_trades','?')}, "
                 f"WR={adp.get('wr',0):.3f}, PF={adp.get('pf',0):.3f}, "
                 f"Sharpe={adp.get('sharpe',0):.2f}, "
                 f"mean_thresh={adp.get('mean_threshold',0):.3f}")
        log.info(f"    Regime gap: {cfg_res['regime_analysis']['gap']:.3f} "
                 f"{'PASS' if cfg_res['regime_analysis']['pass_gate'] else 'FAIL'}")
        log.info(f"    IC: pearson={cfg_res['ic']['pearson']:.4f}, spearman={cfg_res['ic']['spearman']:.4f}")
        log.info(f"    LGBM picks: {cfg_res['lgbm_config_picks']}")

    # ── 4. Save combined results ──
    results = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v2.1',
        'improvements': [
            'multi-config (tp4sl3 + tp8sl5)',
            'new features (time_of_day, session_pct, ofi_ratio_10s_1s, queue_pressure, cancel/add_intensity, renewal_ratio)',
            'hyperparameter selection (default/shallow/deep)',
            'per-fold adaptive threshold',
        ],
        'walk_forward': {
            'train_days': TRAIN_DAYS,
            'oot_days': OOT_DAYS,
            'slide_days': SLIDE_DAYS,
            'method': 'SLIDING',
        },
        'lgbm_configs': {k: {kk: vv for kk, vv in v.items() if kk != 'verbose'}
                         for k, v in LGBM_CONFIGS.items()},
        'configs': all_config_results,
    }

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {out_path}")

    # ── 5. MLflow logging ──
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("queue_entry_selector")

        for cfg_name, cfg_res in all_config_results.items():
            with mlflow.start_run(run_name=f"v2.1_{cfg_name}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({
                    'version': 'v2.1',
                    'fifo_config': cfg_name,
                    'n_features': cfg_res['n_features'],
                    'n_dates': cfg_res['n_dates'],
                    'n_folds': cfg_res['n_folds'],
                    'train_days': TRAIN_DAYS,
                    'oot_days': OOT_DAYS,
                })
                mlflow.log_metrics({
                    'baseline_wr': cfg_res['baseline']['wr'],
                    'baseline_pnl': cfg_res['baseline']['pnl'],
                    'ic_pearson': cfg_res['ic']['pearson'],
                    'ic_spearman': cfg_res['ic']['spearman'],
                    'regime_gap': cfg_res['regime_analysis']['gap'],
                })
                # Log best fixed threshold metrics
                if cfg_res['filtered_performance']:
                    best_key = max(cfg_res['filtered_performance'].items(),
                                   key=lambda x: x[1]['pf'])[0]
                    best = cfg_res['filtered_performance'][best_key]
                    mlflow.log_metrics({
                        'best_thresh': float(best_key),
                        'best_wr': best['wr'],
                        'best_pf': best['pf'],
                        'best_sharpe': best['sharpe'],
                        'best_n_trades': best['n_trades'],
                    })
                # Log adaptive threshold metrics
                adp = cfg_res.get('adaptive_threshold_aggregate', {})
                if adp:
                    mlflow.log_metrics({
                        'adaptive_wr': adp.get('wr', 0),
                        'adaptive_pf': adp.get('pf', 0),
                        'adaptive_sharpe': adp.get('sharpe', 0),
                        'adaptive_n_trades': adp.get('n_trades', 0),
                        'adaptive_mean_thresh': adp.get('mean_threshold', 0),
                    })
                mlflow.log_artifact(str(out_path))
        log.info("MLflow runs logged")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")

    log.info(f"\n{'='*70}")
    log.info("Queue Entry Selector v2.1 COMPLETE")
    log.info(f"{'='*70}")


if __name__ == '__main__':
    main()
