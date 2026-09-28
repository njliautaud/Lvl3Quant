#!/usr/bin/env python3
"""
queue_entry_selector_v2_4.py — Queue Entry Selector v2.4 (REGRESSION)
=====================================================================

KEY DIFFERENCE FROM v2.1: Regression instead of classification.
  - Predicts net_ticks directly (how many ticks will this trade make?)
  - Filters by predicted_profit > cost_threshold instead of probability
  - Same 28 features as v2.1 champion (proven optimal feature set)
  - Same feature swapping for direction normalization (proven essential)
  - Same multi-hyperparameter selection (default/shallow/deep LightGBM)
  - Same walk-forward: 25d train, 5d OOT, 5d slide (SLIDING per HC #0)

Why regression might help:
  1. Binary classification throws away magnitude info — a +0.1t trade and +8t trade are both "1"
  2. Regression directly optimizes for expected profit, not just win probability
  3. Threshold becomes interpretable: "take trades predicted to make > 0.5 ticks"
  4. Model learns asymmetric payoffs naturally (tp4sl3 = +4 vs -3)

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
OUTPUT_DIR = BASE / "output" / "queue_entry_selector_v2_4"
LOG_PATH = BASE / "logs" / "queue_entry_selector_v2_4.log"

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

# FIFO configs to evaluate
FIFO_CONFIGS = ['tp4sl3']  # Focus on proven winner; add tp8sl5 if tp4sl3 works

# Walk-forward
TRAIN_DAYS = 25
OOT_DAYS = 5
SLIDE_DAYS = 5

# Session timing (ET)
SESSION_START_SECONDS = 9 * 3600 + 30 * 60
SESSION_END_SECONDS = 16 * 3600
TOTAL_SESSION_SECONDS = SESSION_END_SECONDS - SESSION_START_SECONDS

# Features — SAME AS v2.1 CHAMPION (28 features, proven optimal)
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

# ── LightGBM REGRESSION hyperparameter variations ──
LGBM_CONFIGS = {
    'default': {
        'objective': 'regression',
        'metric': 'mae',
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
        'objective': 'regression',
        'metric': 'mae',
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
        'objective': 'regression',
        'metric': 'mae',
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
    # Huber loss — robust to outliers in net_ticks
    'huber': {
        'objective': 'huber',
        'metric': 'mae',
        'huber_delta': 2.0,  # ~2 ticks; outliers beyond this get linear penalty
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
}

# Prediction thresholds (in ticks) — "take trades predicted to make > X ticks"
PRED_THRESHOLDS = [0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 2.5, 3.0]


# ─────────────────────────────────────
# Data Loading (identical to v2.1)
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
    """Join queue features to FIFO labels by nearest timestamp (backward only — no look-forward)."""
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
    """Add derived features — SAME AS v2.1 (28 features proven optimal)."""
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
    ts_seconds_of_day = (df['ts_ns'] % (86400 * 1_000_000_000)) / 1_000_000_000
    df['time_of_day'] = (ts_seconds_of_day - SESSION_START_SECONDS).clip(lower=0)
    df['session_pct'] = (df['time_of_day'] / TOTAL_SESSION_SECONDS).clip(0, 1)
    df['ofi_ratio_10s_1s'] = df['ofi_10s'] / (df['ofi_1s'].abs() + 1e-6)
    df['queue_pressure'] = (df['bid_qty_at_touch'] - df['ask_qty_at_touch']) / \
                           (df['bid_qty_at_touch'] + df['ask_qty_at_touch'] + 1e-6)
    df['cancel_intensity'] = df['bid_cancel_rate_1s'] + df['ask_cancel_rate_1s']
    df['add_intensity'] = df['bid_add_rate_1s'] + df['ask_add_rate_1s']
    df['renewal_ratio'] = df['add_intensity'] / (df['cancel_intensity'] + 1e-6)

    return df


def prepare_training_data(merged_df, date_str):
    """Create long + short training rows with direction-aware feature swapping.

    KEY DIFFERENCE: target is net_ticks (continuous) instead of binary.
    We also keep a binary 'profitable' column for evaluation.
    """
    rows = []

    for side in ['long', 'short']:
        filled_col = f'{side}_filled'
        net_col = f'{side}_net_ticks'

        mask = merged_df[filled_col] == True
        subset = merged_df[mask].copy()

        if len(subset) == 0:
            continue

        # REGRESSION TARGET: net_ticks directly
        subset['target'] = subset[net_col].astype(float)
        subset['profitable'] = (subset[net_col] > 0).astype(int)
        subset['net_ticks'] = subset[net_col].astype(float)
        subset['side'] = side
        subset['date'] = date_str

        feature_cols = RAW_FEATURES + [
            'queue_ratio', 'queue_diff', 'net_flow_diff',
            'trade_imbalance', 'ofi_momentum', 'level_age_diff',
            'time_of_day', 'session_pct', 'ofi_ratio_10s_1s',
            'queue_pressure', 'cancel_intensity', 'add_intensity', 'renewal_ratio',
        ]

        row = subset[['ts_ns', 'target', 'profitable', 'net_ticks', 'side', 'date'] +
                     [c for c in feature_cols if c in subset.columns]].copy()

        # Direction normalization via feature swapping (ESSENTIAL — v2.2 proved this)
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


def train_best_model(X_tr, y_tr, X_val, y_val, feature_cols):
    """Train all LGBM regression configs, return best model by OOT MAE."""
    best_model = None
    best_mae = float('inf')
    best_config_name = 'default'

    for config_name, params in LGBM_CONFIGS.items():
        model = lgb.LGBMRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)]
        )

        # Evaluate MAE on validation set
        pred = model.predict(X_val)
        mae = np.mean(np.abs(pred - y_val))

        # Also compute IC on val — more relevant than MAE for trading
        if len(np.unique(pred)) > 1:
            ic = np.corrcoef(pred, y_val)[0, 1]
        else:
            ic = 0

        log.debug(f"    {config_name}: MAE={mae:.3f}, IC={ic:.4f}")

        # Select by lowest MAE (or highest IC — let's use IC as it's more trading-relevant)
        # Actually, use correlation (IC) for model selection — we care about ranking, not absolute accuracy
        if ic > -best_mae:  # Using negative MAE as placeholder; switch to IC comparison
            pass

        if mae < best_mae:
            best_mae = mae
            best_model = model
            best_config_name = config_name

    return best_model, best_config_name, best_mae


def run_for_config(fifo_config, valid_dates, all_data, feature_cols):
    """Run walk-forward regression for a single FIFO config."""
    log.info(f"\n{'='*70}")
    log.info(f"FIFO CONFIG: {fifo_config} (REGRESSION)")
    log.info(f"Walk-forward: {TRAIN_DAYS}d train, {OOT_DAYS}d OOT, {SLIDE_DAYS}d slide (SLIDING)")
    log.info(f"{'='*70}")

    all_trades = []
    fold_results = []
    fi_accum = {}
    config_picks = {}

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
        y_train = train_df['target'].values.astype(np.float32)  # net_ticks (continuous!)
        X_oot = oot_df[feature_cols].values.astype(np.float32)
        y_oot = oot_df['target'].values.astype(np.float32)

        # Replace inf/nan
        X_train = np.nan_to_num(X_train, nan=0., posinf=100., neginf=-100.)
        X_oot = np.nan_to_num(X_oot, nan=0., posinf=100., neginf=-100.)

        # Split last 20% of training for early stopping
        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        # Train best regression model across configs
        model, config_name, val_mae = train_best_model(X_tr, y_tr, X_val, y_val, feature_cols)
        config_picks[config_name] = config_picks.get(config_name, 0) + 1

        # Predict on OOT — prediction is expected net_ticks
        pred_ticks = model.predict(X_oot)
        oot_df = oot_df.copy()
        oot_df['pred_ticks'] = pred_ticks

        # Feature importance
        fi = dict(zip(feature_cols, model.feature_importances_))
        for feat, imp in fi.items():
            if feat not in fi_accum:
                fi_accum[feat] = []
            fi_accum[feat].append(imp)

        # IC on this fold
        if len(np.unique(pred_ticks)) > 1:
            fold_ic = np.corrcoef(pred_ticks, y_oot)[0, 1]
            fold_rank_ic = scipy_stats.spearmanr(pred_ticks, y_oot)[0]
        else:
            fold_ic = 0
            fold_rank_ic = 0

        # Evaluate at prediction thresholds
        baseline_wr = oot_df['profitable'].mean()
        baseline_pnl = oot_df['net_ticks'].sum()
        baseline_n = len(oot_df)

        threshold_results = {}
        for thresh in PRED_THRESHOLDS:
            selected = oot_df[oot_df['pred_ticks'] >= thresh]
            if len(selected) < 5:
                continue
            sel_wr = selected['profitable'].mean()
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

        fold_res = {
            'fold': fold_num,
            'train_dates': train_dates,
            'oot_dates': oot_dates,
            'n_train': len(train_df),
            'n_oot': len(oot_df),
            'baseline_wr': float(baseline_wr),
            'baseline_pnl': float(baseline_pnl),
            'baseline_n': baseline_n,
            'lgbm_config': config_name,
            'val_mae': float(val_mae),
            'fold_ic': float(fold_ic),
            'fold_rank_ic': float(fold_rank_ic),
            'pred_ticks_mean': float(pred_ticks.mean()),
            'pred_ticks_std': float(pred_ticks.std()),
            'pred_ticks_min': float(pred_ticks.min()),
            'pred_ticks_max': float(pred_ticks.max()),
            'thresholds': threshold_results,
        }
        fold_results.append(fold_res)

        # Collect OOT trades
        oot_df['fold'] = fold_num
        all_trades.append(oot_df)

        # Log fold summary
        log.info(f"  Fold {fold_num:2d} | {config_name:8s} MAE={val_mae:.3f} IC={fold_ic:.3f} | "
                 f"OOT={','.join(oot_dates)} | "
                 f"baseline: WR={baseline_wr:.3f}, PnL={baseline_pnl:+.1f}t | "
                 f"pred range=[{pred_ticks.min():.2f}, {pred_ticks.max():.2f}]")

        fold_num += 1
        start_idx += SLIDE_DAYS

    if not all_trades:
        log.error(f"No valid folds for {fifo_config}")
        return None

    all_trades_df = pd.concat(all_trades, ignore_index=True)

    # ── Aggregate metrics ──
    total_n = len(all_trades_df)
    total_wr = all_trades_df['profitable'].mean()
    total_pnl = all_trades_df['net_ticks'].sum()

    log.info(f"\n--- BASELINE (no filter) [{fifo_config}] ---")
    log.info(f"  Total entries: {total_n}")
    log.info(f"  WR: {total_wr:.3f}")
    log.info(f"  Total PnL: {total_pnl:+.1f} ticks")
    log.info(f"  Per-trade: {total_pnl/max(total_n,1):+.3f} ticks")

    # ── Prediction distribution ──
    log.info(f"\n--- PREDICTION DISTRIBUTION [{fifo_config}] ---")
    pred_all = all_trades_df['pred_ticks']
    log.info(f"  Mean:   {pred_all.mean():.3f}")
    log.info(f"  Median: {pred_all.median():.3f}")
    log.info(f"  Std:    {pred_all.std():.3f}")
    log.info(f"  Min:    {pred_all.min():.3f}")
    log.info(f"  Max:    {pred_all.max():.3f}")
    for pct in [10, 25, 50, 75, 90, 95, 99]:
        log.info(f"  p{pct}: {np.percentile(pred_all, pct):.3f}")

    # ── Filtered performance at prediction thresholds ──
    log.info(f"\n--- FILTERED ENTRY PERFORMANCE [{fifo_config}] (pred_ticks >= threshold) ---")
    agg_results = {}
    for thresh in PRED_THRESHOLDS:
        sel = all_trades_df[all_trades_df['pred_ticks'] >= thresh]
        if len(sel) < 10:
            continue
        sel_wr = sel['profitable'].mean()
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

        # Regime gate
        gap, green_s, red_s = compute_regime_gap(sel)

        agg_results[str(thresh)] = {
            'n_trades': int(len(sel)),
            'n_days': int(daily.count()),
            'wr': float(sel_wr),
            'pnl': float(sel_pnl),
            'per_trade': float(per_trade),
            'pf': float(pf),
            'sharpe': float(sharpe),
            'sortino': float(sortino),
            'regime_gap': float(gap),
            'regime_pass': gap <= 0.50,
            'green_sharpe': green_s.get('sharpe', 0),
            'red_sharpe': red_s.get('sharpe', 0),
        }

        regime_str = 'PASS' if gap <= 0.50 else 'FAIL'
        log.info(f"  pred>={thresh:+.2f}t: n={len(sel):5d}, WR={sel_wr:.3f}, "
                 f"PnL={sel_pnl:+.1f}t, per_trade={per_trade:+.3f}t, "
                 f"PF={pf:.3f}, Sharpe={sharpe:.3f}, Sortino={sortino:.2f}, "
                 f"regime={regime_str}({gap:.1%})")

    # ── Compare with classification-equivalent thresholds ──
    # For comparison with v2.1: what happens if we filter by "predicted profitable" (pred > 0)?
    log.info(f"\n--- CLASSIFICATION-EQUIVALENT COMPARISON [{fifo_config}] ---")
    for label, mask_fn in [
        ('pred > 0 (all predicted profitable)', lambda df: df['pred_ticks'] > 0),
        ('pred > 0.5 (half-tick edge)', lambda df: df['pred_ticks'] > 0.5),
        ('pred > 1.0 (full-tick edge)', lambda df: df['pred_ticks'] > 1.0),
        ('pred > comm (covers commission)', lambda df: df['pred_ticks'] > COMMISSION_RT_TICKS),
        ('top 10% predictions', lambda df: df['pred_ticks'] >= df['pred_ticks'].quantile(0.90)),
        ('top 20% predictions', lambda df: df['pred_ticks'] >= df['pred_ticks'].quantile(0.80)),
        ('top 5% predictions', lambda df: df['pred_ticks'] >= df['pred_ticks'].quantile(0.95)),
    ]:
        sel = all_trades_df[mask_fn(all_trades_df)]
        if len(sel) < 10:
            log.info(f"  {label}: insufficient trades ({len(sel)})")
            continue
        sel_wr = sel['profitable'].mean()
        sel_pnl = sel['net_ticks'].sum()
        winners = sel[sel['net_ticks'] > 0]['net_ticks'].sum()
        losers = abs(sel[sel['net_ticks'] < 0]['net_ticks'].sum())
        pf = winners / max(losers, 1e-6)
        per_trade = sel_pnl / len(sel)
        daily = sel.groupby('date')['net_ticks'].sum()
        sharpe = float(daily.mean() / daily.std()) if len(daily) > 1 and daily.std() > 0 else 0
        gap, _, _ = compute_regime_gap(sel)

        log.info(f"  {label}: n={len(sel)}, WR={sel_wr:.3f}, PnL={sel_pnl:+.1f}t, "
                 f"PF={pf:.2f}, Sharpe={sharpe:.3f}, regime={'PASS' if gap<=0.5 else 'FAIL'}({gap:.1%})")

    # ── Per-side breakdown ──
    log.info(f"\n--- PER-SIDE BREAKDOWN [{fifo_config}] (pred > 0.5 ticks) ---")
    side_breakdown = {}
    for side in ['long', 'short']:
        side_df = all_trades_df[(all_trades_df['side'] == side) &
                                (all_trades_df['pred_ticks'] >= 0.5)]
        if len(side_df) < 5:
            log.info(f"  {side.upper()}: insufficient trades ({len(side_df)})")
            continue
        sw = side_df['profitable'].mean()
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

    # ── Regime analysis at best threshold ──
    best_thresh_key = max(
        [(k, v) for k, v in agg_results.items() if v['n_trades'] >= 50],
        key=lambda x: x[1]['pf'],
        default=(str(PRED_THRESHOLDS[0]), agg_results.get(str(PRED_THRESHOLDS[0]), {}))
    )[0] if agg_results else '0.0'
    best_thresh_val = float(best_thresh_key)

    log.info(f"\n--- REGIME ANALYSIS [{fifo_config}] (best threshold = {best_thresh_key}) ---")
    filtered = all_trades_df[all_trades_df['pred_ticks'] >= best_thresh_val]
    if len(filtered) > 10:
        gap, green_stats, red_stats = compute_regime_gap(filtered)
        log.info(f"  Green: Sharpe={green_stats.get('sharpe',0):.3f}, n_days={green_stats.get('n_days',0)}")
        log.info(f"  Red:   Sharpe={red_stats.get('sharpe',0):.3f}, n_days={red_stats.get('n_days',0)}")
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
    ic = np.corrcoef(all_trades_df['pred_ticks'], all_trades_df['net_ticks'])[0, 1]
    rank_ic = scipy_stats.spearmanr(all_trades_df['pred_ticks'], all_trades_df['net_ticks'])[0]
    log.info(f"\n--- INFORMATION COEFFICIENT [{fifo_config}] ---")
    log.info(f"  Pearson IC (pred_ticks vs actual_net_ticks): {ic:.4f}")
    log.info(f"  Rank IC (Spearman): {rank_ic:.4f}")

    # ── Calibration check: are predictions well-calibrated? ──
    log.info(f"\n--- CALIBRATION CHECK [{fifo_config}] ---")
    log.info("  Predicted range → Actual mean net_ticks:")
    for low, high in [(-4, -2), (-2, -1), (-1, -0.5), (-0.5, 0), (0, 0.5), (0.5, 1), (1, 2), (2, 4)]:
        bucket = all_trades_df[(all_trades_df['pred_ticks'] >= low) & (all_trades_df['pred_ticks'] < high)]
        if len(bucket) >= 10:
            actual_mean = bucket['net_ticks'].mean()
            actual_wr = bucket['profitable'].mean()
            log.info(f"    [{low:+.1f}, {high:+.1f}): n={len(bucket):5d}, "
                     f"pred_mean={bucket['pred_ticks'].mean():+.3f}, actual_mean={actual_mean:+.3f}, "
                     f"WR={actual_wr:.3f}")

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
        'model_type': 'regression',
        'n_folds': fold_num,
        'lgbm_config_picks': config_picks,
        'baseline': {
            'n_trades': int(total_n),
            'wr': float(total_wr),
            'pnl': float(total_pnl),
            'per_trade': float(total_pnl / max(total_n, 1)),
        },
        'prediction_distribution': {
            'mean': float(pred_all.mean()),
            'median': float(pred_all.median()),
            'std': float(pred_all.std()),
            'min': float(pred_all.min()),
            'max': float(pred_all.max()),
        },
        'filtered_performance': agg_results,
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
    log.info("Queue Entry Selector v2.4 — REGRESSION (predict net_ticks)")
    log.info("Same features as v2.1 champion, different problem framing")
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

        # Define feature columns — same as v2.1
        sample = list(all_data.values())[0]
        feature_cols = [c for c in RAW_FEATURES + [
            'queue_ratio', 'queue_diff', 'net_flow_diff',
            'trade_imbalance', 'ofi_momentum', 'level_age_diff',
            'time_of_day', 'session_pct', 'ofi_ratio_10s_1s',
            'queue_pressure', 'cancel_intensity', 'add_intensity', 'renewal_ratio',
        ] if c in sample.columns]

        log.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

        result = run_for_config(fifo_config, valid_dates, all_data, feature_cols)
        if result:
            all_config_results[fifo_config] = result

    # ── 3. Save results ──
    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(all_config_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {results_path}")

    # ── 4. Print champion summary ──
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY — v2.4 REGRESSION")
    log.info("=" * 70)
    for cfg, res in all_config_results.items():
        log.info(f"\n  Config: {cfg}")
        log.info(f"  Model: LightGBM Regression (predict net_ticks)")
        log.info(f"  IC (Pearson): {res['ic']['pearson']:.4f}")
        log.info(f"  IC (Spearman): {res['ic']['spearman']:.4f}")
        log.info(f"  Prediction dist: mean={res['prediction_distribution']['mean']:.3f}, "
                 f"std={res['prediction_distribution']['std']:.3f}")
        for thresh, perf in res['filtered_performance'].items():
            regime = 'PASS' if perf.get('regime_pass', False) else 'FAIL'
            log.info(f"    pred>={thresh}t: n={perf['n_trades']}, WR={perf['wr']:.3f}, "
                     f"PF={perf['pf']:.2f}, Sharpe={perf['sharpe']:.3f}, "
                     f"regime={regime}({perf['regime_gap']:.1%})")


if __name__ == '__main__':
    main()
