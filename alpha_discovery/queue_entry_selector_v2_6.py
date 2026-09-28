#!/usr/bin/env python3
"""
queue_entry_selector_v2_6.py — Queue Entry Selector v2.6 (ORDINAL CLASSIFICATION)
==================================================================================

KEY DIFFERENCE: 3-class ordinal classification instead of binary.
  Classes:
    0 = LOSS (net_ticks <= -1.0) — strong loser, hit SL or heavy adverse move
    1 = NEUTRAL (-1.0 < net_ticks <= +0.5) — marginal, not worth the commission
    2 = WIN (net_ticks > +0.5) — clear winner, covers commission + spread

Why this might help:
  - Binary classification treats a +0.1t trade same as +4t. This loses info.
  - The model can now distinguish "definitely bad" from "maybe OK" from "definitely good"
  - We can filter by P(WIN) or P(WIN) - P(LOSS) for more calibrated signals
  - Ordinal softmax probabilities give richer confidence measures

Same 28 features as v2.1 champion (proven optimal)
Same feature swapping (proven essential)
Same multi-hyperparameter selection
Walk-forward: 25d train, 5d OOT, 5d slide (SLIDING per HC #0)
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
from sklearn.metrics import roc_auc_score

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
OUTPUT_DIR = BASE / "output" / "queue_entry_selector_v2_6"
LOG_PATH = BASE / "logs" / "queue_entry_selector_v2_6.log"

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
FIFO_CONFIGS = ['tp4sl3']

TRAIN_DAYS = 25
OOT_DAYS = 5
SLIDE_DAYS = 5

SESSION_START_SECONDS = 9 * 3600 + 30 * 60
SESSION_END_SECONDS = 16 * 3600
TOTAL_SESSION_SECONDS = SESSION_END_SECONDS - SESSION_START_SECONDS

# ── Class boundaries (in net_ticks) ──
# Class 0: LOSS — net_ticks <= LOSS_THRESH (definite loser)
# Class 1: NEUTRAL — LOSS_THRESH < net_ticks <= WIN_THRESH (marginal)
# Class 2: WIN — net_ticks > WIN_THRESH (definite winner, covers costs)
LOSS_THRESH = -1.0  # Below -1 tick = clear loser
WIN_THRESH = 0.5    # Above +0.5 ticks = winner after commission

# Features — SAME AS v2.1 (28 features)
RAW_FEATURES = [
    'ofi_10s', 'ofi_5s', 'ofi_1s',
    'top_imbalance', 'microprice_offset_ticks',
    'bid_qty_at_touch', 'ask_qty_at_touch',
    'bid_trade_rate_1s', 'ask_trade_rate_1s',
    'bid_add_rate_1s', 'ask_add_rate_1s',
    'bid_cancel_rate_1s', 'ask_cancel_rate_1s',
    'bid_level_age_s', 'ask_level_age_s',
]

FLIP_FEATURES = ['ofi_10s', 'ofi_5s', 'ofi_1s', 'top_imbalance', 'microprice_offset_ticks',
                  'queue_diff', 'net_flow_diff', 'trade_imbalance', 'ofi_momentum',
                  'ofi_ratio_10s_1s', 'queue_pressure']

SWAP_PAIRS = [
    ('bid_qty_at_touch', 'ask_qty_at_touch'),
    ('bid_trade_rate_1s', 'ask_trade_rate_1s'),
    ('bid_add_rate_1s', 'ask_add_rate_1s'),
    ('bid_cancel_rate_1s', 'ask_cancel_rate_1s'),
    ('bid_level_age_s', 'ask_level_age_s'),
]

# ── LightGBM MULTICLASS config (SINGLE — multiclass is 3x heavier per config) ──
LGBM_CONFIGS = {
    'default': {
        'objective': 'multiclass',
        'num_class': 3,
        'metric': 'multi_logloss',
        'n_estimators': 150,
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


# ─────────────────────────────────────
# Data Loading (identical to v2.1)
# ─────────────────────────────────────

def load_queue_features(date_str):
    path = QUEUE_DIR / f"features_{date_str}.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if 'ts_ns' not in df.columns or len(df) < 100:
        return None
    df = df.sort_values('ts_ns').reset_index(drop=True)
    return df


def load_fifo_labels(date_str, fifo_config):
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
    queue_df = queue_df.sort_values('ts_ns').reset_index(drop=True)
    fifo_df = fifo_df.sort_values('ts_ns').reset_index(drop=True)
    merged = pd.merge_asof(
        fifo_df, queue_df, on='ts_ns',
        direction='backward', tolerance=1_500_000_000
    )
    n_before = len(merged)
    merged = merged.dropna(subset=['ofi_10s'])
    return merged, n_before, len(merged)


def engineer_features(df):
    """Same as v2.1 — 28 features."""
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


def net_ticks_to_class(net_ticks):
    """Convert net_ticks to ordinal class label."""
    if net_ticks <= LOSS_THRESH:
        return 0  # LOSS
    elif net_ticks <= WIN_THRESH:
        return 1  # NEUTRAL
    else:
        return 2  # WIN


def prepare_training_data(merged_df, date_str):
    """Create long + short training rows with direction-aware feature swapping.
    Target is 3-class ordinal: LOSS=0, NEUTRAL=1, WIN=2
    """
    rows = []
    for side in ['long', 'short']:
        filled_col = f'{side}_filled'
        net_col = f'{side}_net_ticks'
        mask = merged_df[filled_col] == True
        subset = merged_df[mask].copy()
        if len(subset) == 0:
            continue

        # ORDINAL TARGET: 3 classes based on net_ticks
        subset['net_ticks'] = subset[net_col].astype(float)
        subset['target'] = subset['net_ticks'].apply(net_ticks_to_class)
        subset['profitable'] = (subset[net_col] > 0).astype(int)
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
    if queue_df is None or 'mid_price' not in queue_df.columns or len(queue_df) < 10:
        return 'unknown'
    day_return = queue_df['mid_price'].iloc[-1] - queue_df['mid_price'].iloc[0]
    if day_return > 2 * ES_TICK_SIZE:
        return 'green'
    elif day_return < -2 * ES_TICK_SIZE:
        return 'red'
    return 'flat'


def compute_regime_gap(trades_df):
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
           {'sharpe': sharpe_green, 'n_days': len(green)}, \
           {'sharpe': sharpe_red, 'n_days': len(red)}


def train_best_model(X_tr, y_tr, X_val, y_val):
    """Train all LGBM multiclass configs, return best by validation log loss."""
    from sklearn.metrics import log_loss
    best_model = None
    best_loss = float('inf')
    best_config_name = 'default'

    for config_name, params in LGBM_CONFIGS.items():
        model = lgb.LGBMClassifier(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)]
        )
        pred_proba = model.predict_proba(X_val)
        try:
            ll = log_loss(y_val, pred_proba, labels=[0, 1, 2])
        except ValueError:
            ll = 999.0

        if ll < best_loss:
            best_loss = ll
            best_model = model
            best_config_name = config_name

    return best_model, best_config_name, best_loss


def run_for_config(fifo_config, valid_dates, all_data, feature_cols):
    """Run walk-forward for a single FIFO config."""
    log.info(f"\n{'='*70}")
    log.info(f"FIFO CONFIG: {fifo_config} (ORDINAL 3-CLASS)")
    log.info(f"Classes: 0=LOSS(net<={LOSS_THRESH}), 1=NEUTRAL, 2=WIN(net>{WIN_THRESH})")
    log.info(f"Walk-forward: {TRAIN_DAYS}d train, {OOT_DAYS}d OOT, {SLIDE_DAYS}d slide")
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

        X_train = np.nan_to_num(X_train, nan=0., posinf=100., neginf=-100.)
        X_oot = np.nan_to_num(X_oot, nan=0., posinf=100., neginf=-100.)

        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        model, config_name, val_loss = train_best_model(X_tr, y_tr, X_val, y_val)
        config_picks[config_name] = config_picks.get(config_name, 0) + 1

        # Predict on OOT — get probabilities for all 3 classes
        pred_proba = model.predict_proba(X_oot)  # shape: (n, 3)
        oot_df = oot_df.copy()
        oot_df['p_loss'] = pred_proba[:, 0]
        oot_df['p_neutral'] = pred_proba[:, 1]
        oot_df['p_win'] = pred_proba[:, 2]
        # Derived scores
        oot_df['win_minus_loss'] = pred_proba[:, 2] - pred_proba[:, 0]
        # v2.1-equivalent binary probability (for comparison)
        oot_df['pred_prob'] = pred_proba[:, 2] + 0.5 * pred_proba[:, 1]  # WIN + half NEUTRAL

        # Feature importance
        fi = dict(zip(feature_cols, model.feature_importances_))
        for feat, imp in fi.items():
            if feat not in fi_accum:
                fi_accum[feat] = []
            fi_accum[feat].append(imp)

        # Class distribution in OOT
        class_dist = oot_df['target'].value_counts().to_dict()

        fold_results.append({
            'fold': fold_num,
            'oot_dates': oot_dates,
            'n_train': len(train_df),
            'n_oot': len(oot_df),
            'lgbm_config': config_name,
            'val_loss': float(val_loss),
            'class_dist': class_dist,
        })

        oot_df['fold'] = fold_num
        all_trades.append(oot_df)

        log.info(f"  Fold {fold_num:2d} | {config_name:8s} loss={val_loss:.3f} | "
                 f"OOT={','.join(oot_dates)} | classes: {class_dist}")

        fold_num += 1
        start_idx += SLIDE_DAYS

    if not all_trades:
        log.error(f"No valid folds for {fifo_config}")
        return None

    all_trades_df = pd.concat(all_trades, ignore_index=True)

    # ── Class distribution ──
    total_n = len(all_trades_df)
    log.info(f"\n--- CLASS DISTRIBUTION [{fifo_config}] ---")
    for cls, label in [(0, 'LOSS'), (1, 'NEUTRAL'), (2, 'WIN')]:
        n = (all_trades_df['target'] == cls).sum()
        log.info(f"  {label} (class {cls}): {n} ({100*n/total_n:.1f}%)")

    # ── Method 1: Filter by P(WIN) threshold ──
    log.info(f"\n--- METHOD 1: FILTER BY P(WIN) >= threshold [{fifo_config}] ---")
    agg_pwin = {}
    for thresh in [0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]:
        sel = all_trades_df[all_trades_df['p_win'] >= thresh]
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
        gap, green_s, red_s = compute_regime_gap(sel)

        agg_pwin[str(thresh)] = {
            'n_trades': int(len(sel)), 'wr': float(sel_wr),
            'pnl': float(sel_pnl), 'per_trade': float(per_trade),
            'pf': float(pf), 'sharpe': float(sharpe), 'sortino': float(sortino),
            'regime_gap': float(gap), 'regime_pass': gap <= 0.50,
        }
        regime_str = 'PASS' if gap <= 0.50 else 'FAIL'
        log.info(f"  P(WIN)>={thresh:.2f}: n={len(sel):5d}, WR={sel_wr:.3f}, "
                 f"PnL={sel_pnl:+.1f}t, PF={pf:.3f}, Sharpe={sharpe:.3f}, "
                 f"regime={regime_str}({gap:.1%})")

    # ── Method 2: Filter by WIN_MINUS_LOSS score ──
    log.info(f"\n--- METHOD 2: FILTER BY P(WIN)-P(LOSS) >= threshold [{fifo_config}] ---")
    agg_wml = {}
    for thresh in [-0.10, 0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40]:
        sel = all_trades_df[all_trades_df['win_minus_loss'] >= thresh]
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
        gap, green_s, red_s = compute_regime_gap(sel)

        agg_wml[str(thresh)] = {
            'n_trades': int(len(sel)), 'wr': float(sel_wr),
            'pnl': float(sel_pnl), 'per_trade': float(per_trade),
            'pf': float(pf), 'sharpe': float(sharpe), 'sortino': float(sortino),
            'regime_gap': float(gap), 'regime_pass': gap <= 0.50,
        }
        regime_str = 'PASS' if gap <= 0.50 else 'FAIL'
        log.info(f"  WML>={thresh:+.2f}: n={len(sel):5d}, WR={sel_wr:.3f}, "
                 f"PnL={sel_pnl:+.1f}t, PF={pf:.3f}, Sharpe={sharpe:.3f}, "
                 f"regime={regime_str}({gap:.1%})")

    # ── Method 3: Compare with v2.1-equivalent binary threshold ──
    log.info(f"\n--- METHOD 3: V2.1-EQUIVALENT BINARY (pred_prob = P(WIN) + 0.5*P(NEUTRAL)) [{fifo_config}] ---")
    agg_equiv = {}
    for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
        sel = all_trades_df[all_trades_df['pred_prob'] >= thresh]
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
        gap, _, _ = compute_regime_gap(sel)

        agg_equiv[str(thresh)] = {
            'n_trades': int(len(sel)), 'wr': float(sel_wr),
            'pnl': float(sel_pnl), 'per_trade': float(per_trade),
            'pf': float(pf), 'sharpe': float(sharpe),
            'regime_gap': float(gap), 'regime_pass': gap <= 0.50,
        }
        regime_str = 'PASS' if gap <= 0.50 else 'FAIL'
        log.info(f"  equiv>={thresh:.2f}: n={len(sel):5d}, WR={sel_wr:.3f}, "
                 f"PF={pf:.3f}, Sharpe={sharpe:.3f}, regime={regime_str}({gap:.1%})")

    # ── Per-side breakdown ──
    log.info(f"\n--- PER-SIDE BREAKDOWN [{fifo_config}] (P(WIN) >= 0.45) ---")
    side_breakdown = {}
    for side in ['long', 'short']:
        side_df = all_trades_df[(all_trades_df['side'] == side) &
                                (all_trades_df['p_win'] >= 0.45)]
        if len(side_df) < 5:
            continue
        sw = side_df['profitable'].mean()
        sp = side_df['net_ticks'].sum()
        swin = side_df[side_df['net_ticks'] > 0]['net_ticks'].sum()
        slos = abs(side_df[side_df['net_ticks'] < 0]['net_ticks'].sum())
        spf = swin / max(slos, 1e-6)
        side_breakdown[side] = {
            'n_trades': int(len(side_df)), 'wr': float(sw),
            'pnl': float(sp), 'pf': float(spf),
        }
        log.info(f"  {side.upper()}: n={len(side_df)}, WR={sw:.3f}, PnL={sp:+.1f}t, PF={spf:.3f}")

    # ── Feature importance ──
    log.info(f"\n--- FEATURE IMPORTANCE [{fifo_config}] ---")
    fi_summary = {}
    for feat, gains in fi_accum.items():
        fi_summary[feat] = {'mean_gain': float(np.mean(gains)), 'std_gain': float(np.std(gains))}
    sorted_features = sorted(fi_summary.items(), key=lambda x: x[1]['mean_gain'], reverse=True)
    for rank, (feat, fi_stat) in enumerate(sorted_features):
        fi_stat['rank'] = rank + 1
        if rank < 20:
            log.info(f"  {rank+1:2d}. {feat:35s} gain={fi_stat['mean_gain']:.1f} +/- {fi_stat['std_gain']:.1f}")

    # ── IC analysis ──
    ic_pwin = np.corrcoef(all_trades_df['p_win'], all_trades_df['net_ticks'])[0, 1]
    ic_wml = np.corrcoef(all_trades_df['win_minus_loss'], all_trades_df['net_ticks'])[0, 1]
    ic_equiv = np.corrcoef(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0, 1]
    log.info(f"\n--- INFORMATION COEFFICIENT [{fifo_config}] ---")
    log.info(f"  IC(P_WIN vs net_ticks):    {ic_pwin:.4f}")
    log.info(f"  IC(WML vs net_ticks):      {ic_wml:.4f}")
    log.info(f"  IC(equiv_prob vs net_ticks): {ic_equiv:.4f}")

    # ── Calibration: check class probabilities ──
    log.info(f"\n--- CALIBRATION CHECK [{fifo_config}] ---")
    for pwin_low, pwin_high in [(0, 0.2), (0.2, 0.3), (0.3, 0.4), (0.4, 0.5), (0.5, 0.6), (0.6, 0.8), (0.8, 1.0)]:
        bucket = all_trades_df[(all_trades_df['p_win'] >= pwin_low) & (all_trades_df['p_win'] < pwin_high)]
        if len(bucket) >= 10:
            actual_win_pct = (bucket['target'] == 2).mean()
            actual_mean = bucket['net_ticks'].mean()
            actual_wr = bucket['profitable'].mean()
            log.info(f"    P(WIN) [{pwin_low:.1f}, {pwin_high:.1f}): n={len(bucket):5d}, "
                     f"actual_win%={actual_win_pct:.3f}, net_ticks={actual_mean:+.3f}, WR={actual_wr:.3f}")

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
        'model_type': 'ordinal_3class',
        'classes': {0: f'LOSS(net<={LOSS_THRESH})', 1: 'NEUTRAL', 2: f'WIN(net>{WIN_THRESH})'},
        'n_folds': fold_num,
        'lgbm_config_picks': config_picks,
        'pwin_results': agg_pwin,
        'wml_results': agg_wml,
        'equiv_results': agg_equiv,
        'side_breakdown': side_breakdown,
        'feature_importance': fi_summary,
        'ic': {
            'p_win': float(ic_pwin),
            'win_minus_loss': float(ic_wml),
            'equiv_prob': float(ic_equiv),
        },
        'fold_results': fold_results,
    }


def main():
    log.info("=" * 70)
    log.info("Queue Entry Selector v2.6 — ORDINAL 3-CLASS CLASSIFICATION")
    log.info(f"Classes: LOSS(net<={LOSS_THRESH}), NEUTRAL, WIN(net>{WIN_THRESH})")
    log.info("Same 28 features as v2.1 champion")
    log.info("=" * 70)

    q_files = sorted(QUEUE_DIR.glob("features_*.parquet"))
    q_dates = {f.stem.replace('features_', '') for f in q_files}
    f_files = sorted(FIFO_DIR.glob("*_fifo_labels.npz"))
    f_dates = {f.stem.replace('_fifo_labels', '') for f in f_files}
    overlap = sorted(q_dates & f_dates)
    log.info(f"Queue dates: {len(q_dates)}, FIFO dates: {len(f_dates)}, Overlap: {len(overlap)}")

    all_config_results = {}
    for fifo_config in FIFO_CONFIGS:
        log.info(f"\n# Loading data for {fifo_config}")
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
        log.info(f"Valid dates: {len(valid_dates)}, Regimes: {pd.Series(date_regimes).value_counts().to_dict()}")

        if len(valid_dates) < TRAIN_DAYS + OOT_DAYS:
            continue

        sample = list(all_data.values())[0]
        feature_cols = [c for c in RAW_FEATURES + [
            'queue_ratio', 'queue_diff', 'net_flow_diff',
            'trade_imbalance', 'ofi_momentum', 'level_age_diff',
            'time_of_day', 'session_pct', 'ofi_ratio_10s_1s',
            'queue_pressure', 'cancel_intensity', 'add_intensity', 'renewal_ratio',
        ] if c in sample.columns]

        log.info(f"Using {len(feature_cols)} features")

        # Log class distribution across entire dataset
        all_nt = pd.concat(all_data.values())['net_ticks']
        log.info(f"Overall class dist: LOSS={((all_nt<=LOSS_THRESH).sum())} ({100*(all_nt<=LOSS_THRESH).mean():.1f}%), "
                 f"NEUTRAL={((all_nt>LOSS_THRESH)&(all_nt<=WIN_THRESH)).sum()} "
                 f"({100*((all_nt>LOSS_THRESH)&(all_nt<=WIN_THRESH)).mean():.1f}%), "
                 f"WIN={(all_nt>WIN_THRESH).sum()} ({100*(all_nt>WIN_THRESH).mean():.1f}%)")

        result = run_for_config(fifo_config, valid_dates, all_data, feature_cols)
        if result:
            all_config_results[fifo_config] = result

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(all_config_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {results_path}")

    # Summary
    log.info("\n" + "=" * 70)
    log.info("FINAL SUMMARY — v2.6 ORDINAL 3-CLASS")
    log.info("COMPARE WITH v2.1 CHAMPION:")
    log.info("  v2.1 @0.60: 989 trades, WR 58.9%, PF 1.51, Sharpe 0.271, regime PASS")
    log.info("=" * 70)
    for cfg, res in all_config_results.items():
        log.info(f"\n  Config: {cfg}")
        log.info(f"  IC: P_WIN={res['ic']['p_win']:.4f}, WML={res['ic']['win_minus_loss']:.4f}")
        # Find best threshold for each method
        for method, results in [('P(WIN)', res['pwin_results']),
                                 ('WML', res['wml_results']),
                                 ('Equiv', res['equiv_results'])]:
            best = max(results.items(), key=lambda x: x[1]['pf']) if results else ('?', {})
            perf = best[1]
            if perf:
                regime = 'PASS' if perf.get('regime_pass', False) else 'FAIL'
                log.info(f"  Best {method} @{best[0]}: n={perf['n_trades']}, WR={perf['wr']:.3f}, "
                         f"PF={perf['pf']:.2f}, Sharpe={perf['sharpe']:.3f}, regime={regime}")


if __name__ == '__main__':
    main()
