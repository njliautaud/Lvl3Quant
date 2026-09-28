#!/usr/bin/env python3
"""
queue_entry_selector_v2_1_lean.py — Queue Entry Selector v2.1-lean
===================================================================

HYPOTHESIS: v2.1 champion may benefit from FEWER features.
v2.3 (43 features) proved more features hurt. What about fewer?

This uses ONLY the top 15 features by importance from v2.1:
  1. time_of_day (gain 84.8)
  2. ofi_momentum (67.8)
  3. ofi_10s (62.8)
  4. bid_qty_at_touch (58.4)
  5. session_pct (56.3)
  6. ask_qty_at_touch (54.9)
  7. microprice_offset_ticks (49.2)
  8. ofi_5s (48.1)
  9. top_imbalance (44.3)
  10. ofi_1s (43.7)
  11. renewal_ratio (42.1)
  12. bid_level_age_s (40.5)
  13. ask_level_age_s (39.8)
  14. queue_ratio (38.2)
  15. queue_diff (36.9)

Dropped (low importance in v2.1): ofi_ratio_10s_1s, queue_pressure,
cancel_intensity, add_intensity, bid/ask_trade_rate_1s, bid/ask_add_rate_1s,
bid/ask_cancel_rate_1s, net_flow_diff, trade_imbalance, level_age_diff

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
OUTPUT_DIR = BASE / "output" / "queue_entry_selector_v2_1_lean"
LOG_PATH = BASE / "logs" / "queue_entry_selector_v2_1_lean.log"

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

# TOP 15 FEATURES ONLY — raw features needed to compute them
RAW_FEATURES_NEEDED = [
    'ofi_10s', 'ofi_5s', 'ofi_1s',
    'top_imbalance', 'microprice_offset_ticks',
    'bid_qty_at_touch', 'ask_qty_at_touch',
    'bid_level_age_s', 'ask_level_age_s',
    # Need these for renewal_ratio computation
    'bid_add_rate_1s', 'ask_add_rate_1s',
    'bid_cancel_rate_1s', 'ask_cancel_rate_1s',
]

# The actual 15 features used in the model
LEAN_FEATURES = [
    'time_of_day', 'ofi_momentum', 'ofi_10s',
    'bid_qty_at_touch', 'session_pct', 'ask_qty_at_touch',
    'microprice_offset_ticks', 'ofi_5s', 'top_imbalance',
    'ofi_1s', 'renewal_ratio', 'bid_level_age_s',
    'ask_level_age_s', 'queue_ratio', 'queue_diff',
]

# Direction-sensitive features that need sign flip for shorts
FLIP_FEATURES = ['ofi_10s', 'ofi_5s', 'ofi_1s', 'top_imbalance', 'microprice_offset_ticks',
                  'queue_diff', 'ofi_momentum']

# Swap features: bid<->ask for shorts
SWAP_PAIRS = [
    ('bid_qty_at_touch', 'ask_qty_at_touch'),
    ('bid_level_age_s', 'ask_level_age_s'),
]

# ── LightGBM configs (same as v2.1) ──
LGBM_CONFIGS = {
    'default': {
        'objective': 'binary', 'metric': 'auc',
        'n_estimators': 200, 'learning_rate': 0.05,
        'num_leaves': 31, 'max_depth': 5, 'min_child_samples': 20,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'verbose': -1, 'n_jobs': -1, 'random_state': 42,
    },
    'shallow': {
        'objective': 'binary', 'metric': 'auc',
        'n_estimators': 100, 'learning_rate': 0.1,
        'num_leaves': 15, 'max_depth': 3, 'min_child_samples': 50,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'verbose': -1, 'n_jobs': -1, 'random_state': 42,
    },
    'deep': {
        'objective': 'binary', 'metric': 'auc',
        'n_estimators': 300, 'learning_rate': 0.03,
        'num_leaves': 63, 'max_depth': 7, 'min_child_samples': 10,
        'subsample': 0.8, 'colsample_bytree': 0.8,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'verbose': -1, 'n_jobs': -1, 'random_state': 42,
    },
}


# ─────────────────────────────────────
# Data Loading
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
    """Add ONLY the derived features needed for the lean 15."""
    bid_q = df['bid_qty_at_touch'].clip(lower=1)
    ask_q = df['ask_qty_at_touch'].clip(lower=1)
    df['queue_ratio'] = bid_q / (bid_q + ask_q)
    df['queue_diff'] = df['bid_qty_at_touch'] - df['ask_qty_at_touch']
    df['ofi_momentum'] = df['ofi_10s'] - df['ofi_1s']

    # Time features
    ts_seconds_of_day = (df['ts_ns'] % (86400 * 1_000_000_000)) / 1_000_000_000
    df['time_of_day'] = (ts_seconds_of_day - SESSION_START_SECONDS).clip(lower=0)
    df['session_pct'] = (df['time_of_day'] / TOTAL_SESSION_SECONDS).clip(0, 1)

    # Renewal ratio
    df['cancel_intensity'] = df['bid_cancel_rate_1s'] + df['ask_cancel_rate_1s']
    df['add_intensity'] = df['bid_add_rate_1s'] + df['ask_add_rate_1s']
    df['renewal_ratio'] = df['add_intensity'] / (df['cancel_intensity'] + 1e-6)

    return df


def prepare_training_data(merged_df, date_str):
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

        row = subset[['ts_ns', 'target', 'net_ticks', 'side', 'date'] +
                     [c for c in LEAN_FEATURES if c in subset.columns]].copy()

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
    from sklearn.metrics import roc_auc_score
    best_model = None
    best_auc = -1.0
    best_config_name = 'default'
    for config_name, params in LGBM_CONFIGS.items():
        model = lgb.LGBMClassifier(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)])
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


def run_for_config(fifo_config, valid_dates, all_data, feature_cols):
    log.info(f"\n{'='*70}")
    log.info(f"FIFO CONFIG: {fifo_config} (LEAN 15 features)")
    log.info(f"Features: {feature_cols}")
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
        y_train = train_df['target'].values.astype(int)
        X_oot = oot_df[feature_cols].values.astype(np.float32)
        y_oot = oot_df['target'].values.astype(int)

        X_train = np.nan_to_num(X_train, nan=0., posinf=100., neginf=-100.)
        X_oot = np.nan_to_num(X_oot, nan=0., posinf=100., neginf=-100.)

        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]

        model, config_name, val_auc = train_best_model(X_tr, y_tr, X_val, y_val)
        config_picks[config_name] = config_picks.get(config_name, 0) + 1

        pred_prob = model.predict_proba(X_oot)[:, 1]
        oot_df = oot_df.copy()
        oot_df['pred_prob'] = pred_prob

        fi = dict(zip(feature_cols, model.feature_importances_))
        for feat, imp in fi.items():
            if feat not in fi_accum:
                fi_accum[feat] = []
            fi_accum[feat].append(imp)

        baseline_wr = y_oot.mean()
        baseline_pnl = oot_df['net_ticks'].sum()

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

        fold_results.append({
            'fold': fold_num,
            'oot_dates': oot_dates,
            'n_train': len(train_df),
            'n_oot': len(oot_df),
            'lgbm_config': config_name,
            'val_auc': float(val_auc),
            'thresholds': threshold_results,
        })

        oot_df['fold'] = fold_num
        all_trades.append(oot_df)

        log.info(f"  Fold {fold_num:2d} | {config_name:8s} AUC={val_auc:.3f} | OOT={','.join(oot_dates)}")

        fold_num += 1
        start_idx += SLIDE_DAYS

    if not all_trades:
        log.error(f"No valid folds for {fifo_config}")
        return None

    all_trades_df = pd.concat(all_trades, ignore_index=True)

    # ── Aggregate metrics ──
    log.info(f"\n--- FILTERED ENTRY PERFORMANCE [{fifo_config}] (LEAN 15 features) ---")
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

        gap, green_s, red_s = compute_regime_gap(sel)

        agg_results[str(thresh)] = {
            'n_trades': int(len(sel)),
            'wr': float(sel_wr),
            'pnl': float(sel_pnl),
            'per_trade': float(per_trade),
            'pf': float(pf),
            'sharpe': float(sharpe),
            'sortino': float(sortino),
            'regime_gap': float(gap),
            'regime_pass': gap <= 0.50,
        }

        regime_str = 'PASS' if gap <= 0.50 else 'FAIL'
        log.info(f"  thresh={thresh:.2f}: n={len(sel):5d}, WR={sel_wr:.3f}, "
                 f"PnL={sel_pnl:+.1f}t, per_trade={per_trade:+.3f}t, "
                 f"PF={pf:.3f}, Sharpe={sharpe:.3f}, Sortino={sortino:.2f}, "
                 f"regime={regime_str}({gap:.1%})")

    # ── Per-side breakdown ──
    log.info(f"\n--- PER-SIDE BREAKDOWN [{fifo_config}] (thresh=0.60) ---")
    side_breakdown = {}
    for side in ['long', 'short']:
        side_df = all_trades_df[(all_trades_df['side'] == side) &
                                (all_trades_df['pred_prob'] >= 0.60)]
        if len(side_df) < 5:
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

    # ── Feature importance ──
    log.info(f"\n--- FEATURE IMPORTANCE [{fifo_config}] ---")
    fi_summary = {}
    for feat, gains in fi_accum.items():
        fi_summary[feat] = {'mean_gain': float(np.mean(gains)), 'std_gain': float(np.std(gains))}
    for rank, (feat, fi_stat) in enumerate(sorted(fi_summary.items(), key=lambda x: x[1]['mean_gain'], reverse=True)):
        fi_stat['rank'] = rank + 1
        log.info(f"  {rank+1:2d}. {feat:35s} gain={fi_stat['mean_gain']:.1f} +/- {fi_stat['std_gain']:.1f}")

    # IC
    ic = np.corrcoef(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0, 1]
    rank_ic = scipy_stats.spearmanr(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0]
    log.info(f"\n  Pearson IC: {ic:.4f}, Rank IC: {rank_ic:.4f}")

    # Save
    trades_path = OUTPUT_DIR / f"all_oot_trades_{fifo_config}.parquet"
    save_cols = [c for c in all_trades_df.columns if all_trades_df[c].dtype.kind in 'iufbOS']
    all_trades_df[save_cols].to_parquet(trades_path, index=False)
    log.info(f"Saved {len(all_trades_df)} OOT trades to {trades_path}")

    return {
        'fifo_config': fifo_config,
        'model_type': 'classification_lean15',
        'n_folds': fold_num,
        'n_features': len(feature_cols),
        'features_used': feature_cols,
        'lgbm_config_picks': config_picks,
        'filtered_performance': agg_results,
        'side_breakdown': side_breakdown,
        'feature_importance': fi_summary,
        'ic': {'pearson': float(ic), 'spearman': float(rank_ic)},
        'fold_results': fold_results,
    }


def main():
    log.info("=" * 70)
    log.info("Queue Entry Selector v2.1-LEAN — Top 15 Features Only")
    log.info("Hypothesis: fewer features = better generalization")
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

        # Use ONLY the lean features that exist in the data
        sample = list(all_data.values())[0]
        feature_cols = [c for c in LEAN_FEATURES if c in sample.columns]
        log.info(f"Using {len(feature_cols)} features: {feature_cols}")

        result = run_for_config(fifo_config, valid_dates, all_data, feature_cols)
        if result:
            all_config_results[fifo_config] = result

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(all_config_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {results_path}")

    # Summary comparison hint
    log.info("\n" + "=" * 70)
    log.info("COMPARE WITH v2.1 CHAMPION (28 features):")
    log.info("  v2.1 @0.60: 989 trades, WR 58.9%, PF 1.51, Sharpe 0.271, regime PASS")
    log.info("=" * 70)
    for cfg, res in all_config_results.items():
        perf_060 = res['filtered_performance'].get('0.6', {})
        if perf_060:
            regime_str = 'PASS' if perf_060.get('regime_pass', False) else 'FAIL'
            log.info(f"  LEAN @0.60: {perf_060['n_trades']} trades, WR {perf_060['wr']:.3f}, "
                     f"PF {perf_060['pf']:.2f}, Sharpe {perf_060['sharpe']:.3f}, regime {regime_str}")


if __name__ == '__main__':
    main()
