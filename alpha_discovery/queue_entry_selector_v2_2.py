#!/usr/bin/env python3
"""
queue_entry_selector_v2_2.py — Queue Entry Selector v2.2
=========================================================

Improvements over v2.1 (PF 1.51, Sharpe 0.271, 989 trades @ thresh=0.60):

  1. DROP ofi_ratio_10s_1s — zero correlation (r=-0.004), wasted splits
  2. DROP adaptive thresholds — failed in v2.1, fixed thresholds work better
  3. ONLY tp4sl3 — tp8sl5 was strictly worse, saves ~50% compute
  4. ADD interaction features:
     - time_x_renewal: time_of_day * renewal_ratio (late afternoon + high renewal = strongest)
     - time_x_ofi_momentum: time_of_day * ofi_momentum
     - pressure_x_cancel: queue_pressure * cancel_intensity
     - ofi_x_imbalance: ofi_10s * trade_imbalance
  5. ADD rolling z-scores (300s window) for ofi_10s, ofi_5s, cancel_intensity, add_intensity
     — normalizes within-day, reduces day-to-day variation
  6. ADD hour-of-day bins (one-hot for hours 13-20 UTC) — session structure
  7. Side-aware unified model: side_indicator feature (1=long, 0=short) replaces
     bid/ask feature swapping and sign flipping for shorts
  8. KEEP multi-hyperparameter selection (default/shallow/deep) — works well

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
OUTPUT_DIR = BASE / "output" / "queue_entry_selector_v2_2"
LOG_PATH = BASE / "logs" / "queue_entry_selector_v2_2.log"

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

# Single FIFO config — tp4sl3 strictly better than tp8sl5
FIFO_CONFIG = 'tp4sl3'

# Walk-forward
TRAIN_DAYS = 25
OOT_DAYS = 5
SLIDE_DAYS = 5

# Session timing (ET): 09:30 to 16:00
SESSION_START_SECONDS = 9 * 3600 + 30 * 60  # 09:30 ET in seconds since midnight
SESSION_END_SECONDS = 16 * 3600             # 16:00 ET in seconds since midnight
TOTAL_SESSION_SECONDS = SESSION_END_SECONDS - SESSION_START_SECONDS  # 23400 seconds

# Z-score rolling window (seconds)
ZSCORE_WINDOW_S = 300

# Raw features from queue data (used directly, no bid/ask swapping in v2.2)
RAW_FEATURES = [
    'ofi_10s', 'ofi_5s', 'ofi_1s',
    'top_imbalance', 'microprice_offset_ticks',
    'bid_qty_at_touch', 'ask_qty_at_touch',
    'bid_trade_rate_1s', 'ask_trade_rate_1s',
    'bid_add_rate_1s', 'ask_add_rate_1s',
    'bid_cancel_rate_1s', 'ask_cancel_rate_1s',
    'bid_level_age_s', 'ask_level_age_s',
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


def load_fifo_labels(date_str):
    """Load FIFO labels for a date (tp4sl3 only). Returns DataFrame with ts_ns + label columns."""
    path = FIFO_DIR / f"{date_str}_fifo_labels.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)

    prefix = FIFO_CONFIG
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
    """Add derived features (v2.2: interactions, z-scores, hour bins, no ofi_ratio_10s_1s)."""
    # ── Base derived features (from v2) ──
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

    # ── Time features ──
    ts_seconds_of_day = (df['ts_ns'] % (86400 * 1_000_000_000)) / 1_000_000_000
    df['time_of_day'] = (ts_seconds_of_day - SESSION_START_SECONDS).clip(lower=0)
    df['session_pct'] = (df['time_of_day'] / TOTAL_SESSION_SECONDS).clip(0, 1)

    # ── Queue microstructure features (kept from v2.1, minus ofi_ratio_10s_1s) ──
    df['queue_pressure'] = (df['bid_qty_at_touch'] - df['ask_qty_at_touch']) / \
                           (df['bid_qty_at_touch'] + df['ask_qty_at_touch'] + 1e-6)
    df['cancel_intensity'] = df['bid_cancel_rate_1s'] + df['ask_cancel_rate_1s']
    df['add_intensity'] = df['bid_add_rate_1s'] + df['ask_add_rate_1s']
    df['renewal_ratio'] = df['add_intensity'] / (df['cancel_intensity'] + 1e-6)

    # ── NEW v2.2: Interaction features ──
    df['time_x_renewal'] = df['time_of_day'] * df['renewal_ratio']
    df['time_x_ofi_momentum'] = df['time_of_day'] * df['ofi_momentum']
    df['pressure_x_cancel'] = df['queue_pressure'] * df['cancel_intensity']
    df['ofi_x_imbalance'] = df['ofi_10s'] * df['trade_imbalance']

    # ── NEW v2.2: Rolling z-scores (300s window) ──
    # Queue features are at 1-second intervals, so window = ZSCORE_WINDOW_S rows
    for col, zcol in [('ofi_10s', 'ofi_10s_z300'),
                      ('ofi_5s', 'ofi_5s_z300'),
                      ('cancel_intensity', 'cancel_z300'),
                      ('add_intensity', 'add_z300')]:
        roll_mean = df[col].rolling(window=ZSCORE_WINDOW_S, min_periods=30).mean()
        roll_std = df[col].rolling(window=ZSCORE_WINDOW_S, min_periods=30).std()
        df[zcol] = ((df[col] - roll_mean) / (roll_std + 1e-8)).fillna(0.0)

    # ── NEW v2.2: Hour-of-day bins (hours 13-20 UTC) ──
    hour_of_day_utc = (ts_seconds_of_day / 3600).astype(int)
    for h in range(13, 21):
        df[f'hour_{h}'] = (hour_of_day_utc == h).astype(np.float32)

    return df


def prepare_training_data(merged_df, date_str):
    """Create long + short training rows with side_indicator (no feature swapping in v2.2)."""
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

        # v2.2: side indicator instead of feature swapping
        subset['side_indicator'] = 1.0 if side == 'long' else 0.0

        # No sign flipping, no bid/ask swapping — model learns side-specific patterns
        # via the side_indicator feature

        # Collect all feature columns present
        feature_cols_wanted = RAW_FEATURES + [
            'queue_ratio', 'queue_diff', 'net_flow_diff',
            'trade_imbalance', 'ofi_momentum', 'level_age_diff',
            'time_of_day', 'session_pct',
            'queue_pressure', 'cancel_intensity', 'add_intensity', 'renewal_ratio',
            'time_x_renewal', 'time_x_ofi_momentum', 'pressure_x_cancel', 'ofi_x_imbalance',
            'ofi_10s_z300', 'ofi_5s_z300', 'cancel_z300', 'add_z300',
            'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20',
            'side_indicator',
        ]
        available_cols = [c for c in feature_cols_wanted if c in subset.columns]
        row = subset[['ts_ns', 'target', 'net_ticks', 'side', 'date'] + available_cols].copy()
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
    """Train all 3 LGBM configs, return best model by OOT AUC on validation set."""
    from sklearn.metrics import roc_auc_score

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

        pred_prob = model.predict_proba(X_val)[:, 1]
        try:
            auc = roc_auc_score(y_val, pred_prob)
        except ValueError:
            auc = 0.5

        if auc > best_auc:
            best_auc = auc
            best_model = model
            best_config_name = config_name

    return best_model, best_config_name, best_auc


def run_walkforward(valid_dates, all_data, feature_cols):
    """Run walk-forward for tp4sl3. Returns results dict."""
    log.info(f"\n{'='*70}")
    log.info(f"FIFO CONFIG: {FIFO_CONFIG} (only config — tp8sl5 dropped)")
    log.info(f"Walk-forward: {TRAIN_DAYS}d train, {OOT_DAYS}d OOT, {SLIDE_DAYS}d slide (SLIDING)")
    log.info(f"Features: {len(feature_cols)} (interactions + z-scores + hour bins + side_indicator)")
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
        y_train = train_df['target'].values.astype(int)
        X_oot = oot_df[feature_cols].values.astype(np.float32)
        y_oot = oot_df['target'].values.astype(int)

        # Replace inf/nan
        X_train = np.nan_to_num(X_train, nan=0., posinf=100., neginf=-100.)
        X_oot = np.nan_to_num(X_oot, nan=0., posinf=100., neginf=-100.)

        # Split last 20% of training for early stopping
        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        # Train best model across 3 hyperparameter configs
        model, config_name, val_auc = train_best_model(X_tr, y_tr, X_val, y_val, feature_cols)
        config_picks[config_name] = config_picks.get(config_name, 0) + 1

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
            'thresholds': threshold_results,
            'best_threshold': best_threshold,
            'best_pf': float(best_pf),
        }
        fold_results.append(fold_res)

        # Collect OOT trades
        oot_df['fold'] = fold_num
        all_trades.append(oot_df)

        # Log fold summary
        best_t = threshold_results.get(str(best_threshold), {})
        log.info(f"  Fold {fold_num:2d} | {config_name:8s} AUC={val_auc:.3f} | OOT={','.join(oot_dates)} | "
                 f"baseline: WR={baseline_wr:.3f}, PnL={baseline_pnl:+.1f}t | "
                 f"best@{best_threshold}: n={best_t.get('n_trades','?')}, PF={best_t.get('pf',0):.2f}")

        fold_num += 1
        start_idx += SLIDE_DAYS

    if not all_trades:
        log.error("No valid folds")
        return None

    all_trades_df = pd.concat(all_trades, ignore_index=True)

    # ── Aggregate metrics ──
    total_n = len(all_trades_df)
    total_wr = all_trades_df['target'].mean()
    total_pnl = all_trades_df['net_ticks'].sum()

    log.info(f"\n--- BASELINE (no filter) [{FIFO_CONFIG}] ---")
    log.info(f"  Total entries: {total_n}")
    log.info(f"  WR: {total_wr:.3f}")
    log.info(f"  Total PnL: {total_pnl:+.1f} ticks")
    log.info(f"  Per-trade: {total_pnl/max(total_n,1):+.3f} ticks")

    # Fixed thresholds on aggregate
    log.info(f"\n--- FILTERED ENTRY PERFORMANCE [{FIFO_CONFIG}] ---")
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

        # Day concentration (HC #344): max daily fraction of total trades
        daily_counts = sel.groupby('date').size()
        day_conc = float(daily_counts.max() / daily_counts.sum()) if daily_counts.sum() > 0 else 1.0

        agg_results[str(thresh)] = {
            'n_trades': int(len(sel)),
            'wr': float(sel_wr),
            'pnl': float(sel_pnl),
            'per_trade': float(per_trade),
            'pf': float(pf),
            'sharpe': float(sharpe),
            'sortino': float(sortino),
            'day_conc': float(day_conc),
        }

        log.info(f"  thresh={thresh:.2f}: n={len(sel):5d}, WR={sel_wr:.3f}, "
                 f"PnL={sel_pnl:+.1f}t, per_trade={per_trade:+.3f}t, "
                 f"PF={pf:.3f}, Sharpe={sharpe:.2f}, Sortino={sortino:.2f}, "
                 f"DayConc={day_conc:.3f}")

    # ── Per-side breakdown ──
    best_agg_thresh = max(agg_results.items(), key=lambda x: x[1]['pf'])[0] if agg_results else '0.55'
    log.info(f"\n--- PER-SIDE BREAKDOWN [{FIFO_CONFIG}] (best threshold = {best_agg_thresh}) ---")
    best_thresh_val = float(best_agg_thresh)
    side_breakdown = {}
    for side in ['long', 'short']:
        side_df = all_trades_df[(all_trades_df['side'] == side) &
                                (all_trades_df['pred_prob'] >= best_thresh_val)]
        if len(side_df) < 5:
            log.info(f"  {side.upper()}: insufficient trades ({len(side_df)})")
            continue
        sw = side_df['target'].mean()
        sp = side_df['net_ticks'].sum()
        swin = side_df[side_df['net_ticks'] > 0]['net_ticks'].sum()
        slos = abs(side_df[side_df['net_ticks'] < 0]['net_ticks'].sum())
        spf = swin / max(slos, 1e-6)
        spt = sp / len(side_df)
        side_breakdown[side] = {
            'n_trades': int(len(side_df)),
            'wr': float(sw),
            'pnl': float(sp),
            'per_trade': float(spt),
            'pf': float(spf),
        }
        log.info(f"  {side.upper()}: n={len(side_df)}, WR={sw:.3f}, PnL={sp:+.1f}t, "
                 f"per_trade={spt:+.3f}t, PF={spf:.3f}")

    # ── Per-side at EACH threshold ──
    log.info(f"\n--- PER-SIDE AT ALL THRESHOLDS [{FIFO_CONFIG}] ---")
    for thresh in [0.50, 0.55, 0.60, 0.65, 0.70]:
        for side in ['long', 'short']:
            side_df = all_trades_df[(all_trades_df['side'] == side) &
                                    (all_trades_df['pred_prob'] >= thresh)]
            if len(side_df) < 5:
                continue
            sw = side_df['target'].mean()
            sp = side_df['net_ticks'].sum()
            swin = side_df[side_df['net_ticks'] > 0]['net_ticks'].sum()
            slos = abs(side_df[side_df['net_ticks'] < 0]['net_ticks'].sum())
            spf = swin / max(slos, 1e-6)
            log.info(f"  thresh={thresh:.2f} {side.upper():5s}: n={len(side_df):5d}, "
                     f"WR={sw:.3f}, PF={spf:.3f}, PnL={sp:+.1f}t")

    # ── Regime analysis ──
    log.info(f"\n--- REGIME ANALYSIS [{FIFO_CONFIG}] ---")

    filtered = all_trades_df[all_trades_df['pred_prob'] >= best_thresh_val]
    if len(filtered) > 10:
        gap, green_stats, red_stats = compute_regime_gap(filtered)
        log.info(f"  At threshold {best_thresh_val}:")
        log.info(f"  Green: Sharpe={green_stats.get('sharpe',0):.2f}, "
                 f"n_days={green_stats.get('n_days',0)}, n_trades={green_stats.get('n_trades',0)}")
        log.info(f"  Red:   Sharpe={red_stats.get('sharpe',0):.2f}, "
                 f"n_days={red_stats.get('n_days',0)}, n_trades={red_stats.get('n_trades',0)}")
        log.info(f"  Regime gap: {gap:.3f} {'PASS' if gap <= 0.50 else 'FAIL'}")
    else:
        gap = 999.0
        green_stats = {}
        red_stats = {}

    # ── Per-day Sharpe/PF/WR breakdown (HC #428 R1) ──
    log.info(f"\n--- PER-DAY BREAKDOWN [{FIFO_CONFIG}] (thresh={best_thresh_val}) ---")
    if len(filtered) > 0:
        for date_str in sorted(filtered['date'].unique()):
            day_df = filtered[filtered['date'] == date_str]
            d_wr = day_df['target'].mean()
            d_pnl = day_df['net_ticks'].sum()
            d_win = day_df[day_df['net_ticks'] > 0]['net_ticks'].sum()
            d_los = abs(day_df[day_df['net_ticks'] < 0]['net_ticks'].sum())
            d_pf = d_win / max(d_los, 1e-6)
            d_regime = day_df['regime'].iloc[0] if 'regime' in day_df.columns else '?'
            log.info(f"  {date_str} [{d_regime:5s}]: n={len(day_df):4d}, "
                     f"WR={d_wr:.3f}, PF={d_pf:.2f}, PnL={d_pnl:+.1f}t")

    # ── Feature importance ──
    log.info(f"\n--- FEATURE IMPORTANCE [{FIFO_CONFIG}] (mean gain across folds) ---")
    fi_summary = {}
    for feat, gains in fi_accum.items():
        fi_summary[feat] = {
            'mean_gain': float(np.mean(gains)),
            'std_gain': float(np.std(gains)),
        }

    sorted_features = sorted(fi_summary.items(), key=lambda x: x[1]['mean_gain'], reverse=True)
    for rank, (feat, fi_stat) in enumerate(sorted_features):
        fi_stat['rank'] = rank + 1
        if rank < 25:  # Show top 25 (more features now)
            log.info(f"  {rank+1:2d}. {feat:35s} gain={fi_stat['mean_gain']:.1f} +/- {fi_stat['std_gain']:.1f}")

    # ── IC analysis ──
    ic = np.corrcoef(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0, 1]
    rank_ic = scipy_stats.spearmanr(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0]
    log.info(f"\n--- INFORMATION COEFFICIENT [{FIFO_CONFIG}] ---")
    log.info(f"  Pearson IC (prob vs net_ticks): {ic:.4f}")
    log.info(f"  Rank IC (Spearman): {rank_ic:.4f}")

    # ── Feature correlation with net_ticks ──
    log.info(f"\n--- FEATURE CORRELATION WITH NET_TICKS ---")
    for feat in feature_cols:
        if feat in all_trades_df.columns:
            r = np.corrcoef(
                np.nan_to_num(all_trades_df[feat].values.astype(np.float64), nan=0.),
                all_trades_df['net_ticks'].values.astype(np.float64)
            )[0, 1]
            if abs(r) > 0.01:  # Only show features with meaningful correlation
                log.info(f"  {feat:35s}: r={r:+.4f}")

    # ── LGBM config picks ──
    log.info(f"\n--- LGBM CONFIG SELECTION [{FIFO_CONFIG}] ---")
    for cfg_name, cnt in sorted(config_picks.items(), key=lambda x: -x[1]):
        log.info(f"  {cfg_name}: {cnt} folds ({100*cnt/max(fold_num,1):.0f}%)")

    # Save trades parquet
    trades_path = OUTPUT_DIR / f"all_oot_trades_{FIFO_CONFIG}.parquet"
    save_cols = [c for c in all_trades_df.columns if all_trades_df[c].dtype.kind in 'iufbOS']
    all_trades_df[save_cols].to_parquet(trades_path, index=False)
    log.info(f"Saved {len(all_trades_df)} OOT trades to {trades_path}")

    return {
        'fifo_config': FIFO_CONFIG,
        'n_folds': fold_num,
        'lgbm_config_picks': config_picks,
        'baseline': {
            'n_trades': int(total_n),
            'wr': float(total_wr),
            'pnl': float(total_pnl),
            'per_trade': float(total_pnl / max(total_n, 1)),
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
    log.info("Queue Entry Selector v2.2 — Interactions + Z-scores + Hour Bins + Unified Side Model")
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

    # ── 2. Load and prepare data ──
    log.info(f"\nLoading data for FIFO config: {FIFO_CONFIG}")

    all_data = {}
    date_regimes = {}

    for date_str in overlap:
        queue = load_queue_features(date_str)
        fifo = load_fifo_labels(date_str)

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
    log.info(f"Valid dates: {len(valid_dates)}")
    log.info(f"Regimes: {pd.Series(date_regimes).value_counts().to_dict()}")

    if len(valid_dates) < TRAIN_DAYS + OOT_DAYS:
        log.error(f"Only {len(valid_dates)} valid dates — need >= {TRAIN_DAYS + OOT_DAYS}")
        sys.exit(1)

    # Define feature columns from first available day
    sample = list(all_data.values())[0]
    feature_cols = [c for c in RAW_FEATURES + [
        'queue_ratio', 'queue_diff', 'net_flow_diff',
        'trade_imbalance', 'ofi_momentum', 'level_age_diff',
        # v2.1 features (minus ofi_ratio_10s_1s)
        'time_of_day', 'session_pct',
        'queue_pressure', 'cancel_intensity', 'add_intensity', 'renewal_ratio',
        # v2.2 interaction features
        'time_x_renewal', 'time_x_ofi_momentum', 'pressure_x_cancel', 'ofi_x_imbalance',
        # v2.2 rolling z-scores
        'ofi_10s_z300', 'ofi_5s_z300', 'cancel_z300', 'add_z300',
        # v2.2 hour bins
        'hour_13', 'hour_14', 'hour_15', 'hour_16', 'hour_17', 'hour_18', 'hour_19', 'hour_20',
        # v2.2 side indicator
        'side_indicator',
    ] if c in sample.columns]
    log.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # ── 3. Run walk-forward ──
    result = run_walkforward(valid_dates, all_data, feature_cols)

    if result is None:
        log.error("Walk-forward produced no results")
        sys.exit(1)

    result['date_regimes'] = date_regimes
    result['n_dates'] = len(valid_dates)
    result['n_features'] = len(feature_cols)
    result['feature_cols'] = feature_cols

    # ── 4. Save results ──
    results = {
        'timestamp': datetime.now().isoformat(),
        'version': 'v2.2',
        'changes_from_v2_1': [
            'DROPPED ofi_ratio_10s_1s (r=-0.004, zero value)',
            'DROPPED adaptive thresholds (failed in v2.1)',
            'DROPPED tp8sl5 (strictly worse than tp4sl3)',
            'ADDED interaction features: time_x_renewal, time_x_ofi_momentum, pressure_x_cancel, ofi_x_imbalance',
            'ADDED rolling z-scores (300s): ofi_10s_z300, ofi_5s_z300, cancel_z300, add_z300',
            'ADDED hour-of-day bins (13-20 UTC)',
            'ADDED side_indicator feature (unified model, no feature swapping)',
            'ADDED per-day breakdown and day concentration metric',
        ],
        'v2_1_baseline': {
            'pf': 1.51,
            'sharpe': 0.271,
            'n_trades': 989,
            'threshold': 0.60,
        },
        'walk_forward': {
            'train_days': TRAIN_DAYS,
            'oot_days': OOT_DAYS,
            'slide_days': SLIDE_DAYS,
            'method': 'SLIDING',
        },
        'commission_rt_ticks': COMMISSION_RT_TICKS,
        'lgbm_configs': {k: {kk: vv for kk, vv in v.items() if kk != 'verbose'}
                         for k, v in LGBM_CONFIGS.items()},
        'result': result,
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

        with mlflow.start_run(run_name=f"v2.2_{FIFO_CONFIG}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params({
                'version': 'v2.2',
                'fifo_config': FIFO_CONFIG,
                'n_features': result['n_features'],
                'n_dates': result['n_dates'],
                'n_folds': result['n_folds'],
                'train_days': TRAIN_DAYS,
                'oot_days': OOT_DAYS,
                'zscore_window': ZSCORE_WINDOW_S,
                'changes': 'drop_ofi_ratio+drop_adaptive+interactions+zscores+hourbins+side_indicator',
            })
            mlflow.log_metrics({
                'baseline_wr': result['baseline']['wr'],
                'baseline_pnl': result['baseline']['pnl'],
                'ic_pearson': result['ic']['pearson'],
                'ic_spearman': result['ic']['spearman'],
                'regime_gap': result['regime_analysis']['gap'],
            })
            # Log best fixed threshold metrics
            if result['filtered_performance']:
                best_key = max(result['filtered_performance'].items(),
                               key=lambda x: x[1]['pf'])[0]
                best = result['filtered_performance'][best_key]
                mlflow.log_metrics({
                    'best_thresh': float(best_key),
                    'best_wr': best['wr'],
                    'best_pf': best['pf'],
                    'best_sharpe': best['sharpe'],
                    'best_n_trades': best['n_trades'],
                    'best_sortino': best.get('sortino', 0),
                    'best_day_conc': best.get('day_conc', 0),
                })
            mlflow.log_artifact(str(out_path))
        log.info("MLflow run logged")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")

    # ── 6. Summary comparison vs v2.1 ──
    log.info(f"\n{'='*70}")
    log.info("v2.2 vs v2.1 COMPARISON")
    log.info(f"{'='*70}")
    if result['filtered_performance']:
        best_key = max(result['filtered_performance'].items(),
                       key=lambda x: x[1]['pf'])[0]
        best = result['filtered_performance'][best_key]
        log.info(f"  v2.1 @ thresh=0.60: PF=1.51, Sharpe=0.271, n=989")
        log.info(f"  v2.2 @ thresh={best_key}: PF={best['pf']:.3f}, "
                 f"Sharpe={best['sharpe']:.3f}, n={best['n_trades']}")

        # Also show v2.2 at thresh=0.60 for direct comparison
        t60 = result['filtered_performance'].get('0.6', {})
        if t60:
            log.info(f"  v2.2 @ thresh=0.60: PF={t60['pf']:.3f}, "
                     f"Sharpe={t60['sharpe']:.3f}, n={t60['n_trades']}")

        log.info(f"  Regime gate: {'PASS' if result['regime_analysis']['pass_gate'] else 'FAIL'} "
                 f"(gap={result['regime_analysis']['gap']:.3f})")
        log.info(f"  Features: {result['n_features']} (v2.1 had ~27)")
        log.info(f"  LGBM picks: {result['lgbm_config_picks']}")

    log.info(f"\n{'='*70}")
    log.info("Queue Entry Selector v2.2 COMPLETE")
    log.info(f"{'='*70}")


if __name__ == '__main__':
    main()
