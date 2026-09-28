#!/usr/bin/env python3
"""
queue_entry_selector_v2.py — Universal Queue Entry Selection (HC #648 / HC #428)
================================================================================

v2 uses UNIVERSAL queue features (1-second snapshots from build_queue_features_universal.py)
instead of prediction-aligned features, enabling coverage of ~143 dates (vs 40 in v1).

Data:
  - Universal queue features: output/queue_features_universal/ (23,400 samples/day)
  - FIFO labels: data/processed/mbo_events_smart_v3_fifo_labels/ (143 dates)
  - Joined by nearest timestamp (merge_asof)

Walk-forward: 25d train, 5d OOT, slide 5d (SLIDING per HC #0)
Regime gate: |Sharpe_green - Sharpe_red| / max(...) ≤ 0.50 (HC #428)

Features (top 8 from v1 + derived):
  - ofi_10s, ofi_5s, ofi_1s
  - top_imbalance, microprice_offset_ticks
  - bid/ask_qty_at_touch ratio
  - bid/ask_trade_rate_1s
  - Derived: queue_ratio, net_flow_diff, trade_imbalance, ofi_momentum
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
# Detect environment
if Path("/home/nick/Lvl3Quant").exists():
    BASE = Path("/home/nick/Lvl3Quant")
    QUEUE_DIR = BASE / "output" / "queue_features_universal"  # copied from Jupiter
    FIFO_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
elif Path("/home/jupiter/Lvl3Quant").exists():
    BASE = Path("/home/jupiter/Lvl3Quant")
    QUEUE_DIR = BASE / "output" / "queue_features_universal"
    FIFO_DIR = BASE / "data" / "processed" / "mbo_events_smart_v3_fifo_labels"
else:
    raise RuntimeError("Cannot find Lvl3Quant directory")

OUTPUT_DIR = BASE / "output" / "queue_entry_selector_v2"
LOG_PATH = BASE / "logs" / "queue_entry_selector_v2.log"

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

# FIFO config to evaluate
FIFO_CONFIG = 'tp4sl3'  # TP=4 ticks, SL=3 ticks (passive FIFO)

# Walk-forward
TRAIN_DAYS = 25
OOT_DAYS = 5
SLIDE_DAYS = 5

# Features to use (based on v1 importance ranking)
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
                  'queue_diff', 'net_flow_diff', 'trade_imbalance', 'ofi_momentum']
# Swap features: bid↔ask for shorts
SWAP_PAIRS = [
    ('bid_qty_at_touch', 'ask_qty_at_touch'),
    ('bid_trade_rate_1s', 'ask_trade_rate_1s'),
    ('bid_add_rate_1s', 'ask_add_rate_1s'),
    ('bid_cancel_rate_1s', 'ask_cancel_rate_1s'),
    ('bid_level_age_s', 'ask_level_age_s'),
]

LGBM_PARAMS = {
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
    """Load FIFO labels for a date. Returns DataFrame with ts_ns + label columns."""
    path = FIFO_DIR / f"{date_str}_fifo_labels.npz"
    if not path.exists():
        return None
    data = np.load(path, allow_pickle=True)

    prefix = FIFO_CONFIG  # e.g., 'tp4sl3'
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
    # merge_asof requires sorted keys
    queue_df = queue_df.sort_values('ts_ns').reset_index(drop=True)
    fifo_df = fifo_df.sort_values('ts_ns').reset_index(drop=True)

    merged = pd.merge_asof(
        fifo_df, queue_df,
        on='ts_ns',
        direction='backward',  # FIXED: 'nearest' caused look-forward bias (HC audit 2026-06-25)
        tolerance=1_500_000_000  # 1.5 seconds tolerance
    )

    # Drop rows where no queue snapshot was close enough
    n_before = len(merged)
    merged = merged.dropna(subset=['ofi_10s'])  # proxy for "had a queue match"
    n_after = len(merged)

    return merged, n_before, n_after


def engineer_features(df):
    """Add derived features."""
    # Queue ratio
    bid_q = df['bid_qty_at_touch'].clip(lower=1)
    ask_q = df['ask_qty_at_touch'].clip(lower=1)
    df['queue_ratio'] = bid_q / (bid_q + ask_q)
    df['queue_diff'] = df['bid_qty_at_touch'] - df['ask_qty_at_touch']

    # Net flow difference
    bid_net = df.get('bid_add_rate_1s', 0) - df.get('bid_cancel_rate_1s', 0)
    ask_net = df.get('ask_add_rate_1s', 0) - df.get('ask_cancel_rate_1s', 0)
    df['net_flow_diff'] = bid_net - ask_net

    # Trade imbalance
    df['trade_imbalance'] = df['bid_trade_rate_1s'] - df['ask_trade_rate_1s']

    # OFI momentum (10s - 1s)
    df['ofi_momentum'] = df['ofi_10s'] - df['ofi_1s']

    # Level age difference
    df['level_age_diff'] = df['bid_level_age_s'] - df['ask_level_age_s']

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

        # Target: profitable trade (net_ticks > 0)
        subset['target'] = (subset[net_col] > 0).astype(int)
        subset['net_ticks'] = subset[net_col].astype(float)
        subset['side'] = side
        subset['date'] = date_str

        # Feature columns
        feature_cols = RAW_FEATURES + ['queue_ratio', 'queue_diff', 'net_flow_diff',
                                        'trade_imbalance', 'ofi_momentum', 'level_age_diff']

        row = subset[['ts_ns', 'target', 'net_ticks', 'side', 'date'] +
                     [c for c in feature_cols if c in subset.columns]].copy()

        if side == 'short':
            # Flip direction-sensitive features
            for feat in FLIP_FEATURES:
                if feat in row.columns:
                    row[feat] = -row[feat]
            # Swap bid/ask features
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

    # Need regime info
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


# ─────────────────────────────────────
# Main
# ─────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("Queue Entry Selector v2 — Universal Queue Features")
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

    # ── 2. Load and join all data ──
    log.info("Loading and joining data...")
    all_data = {}
    date_regimes = {}

    for date_str in overlap:
        queue = load_queue_features(date_str)
        fifo = load_fifo_labels(date_str)

        if queue is None or fifo is None:
            log.warning(f"  {date_str}: missing queue or FIFO data")
            continue

        queue = engineer_features(queue)
        merged, n_before, n_after = join_queue_fifo(queue, fifo)

        if n_after < 50:
            log.warning(f"  {date_str}: only {n_after} matched entries")
            continue

        training = prepare_training_data(merged, date_str)
        if training is None or len(training) < 50:
            continue

        regime = classify_day_regime(queue)
        date_regimes[date_str] = regime
        training['regime'] = regime
        all_data[date_str] = training

        wr = training['target'].mean()
        log.info(f"  {date_str}: {len(training)} entries (L:{(training['side']=='long').sum()} "
                 f"S:{(training['side']=='short').sum()}), WR={wr:.3f}, regime={regime}")

    valid_dates = sorted(all_data.keys())
    log.info(f"\nValid dates: {len(valid_dates)}")
    log.info(f"Regimes: {pd.Series(date_regimes).value_counts().to_dict()}")

    if len(valid_dates) < TRAIN_DAYS + OOT_DAYS:
        log.error(f"Only {len(valid_dates)} valid dates — need ≥ {TRAIN_DAYS + OOT_DAYS}")
        sys.exit(1)

    # ── 3. Define feature columns ──
    sample = list(all_data.values())[0]
    feature_cols = [c for c in RAW_FEATURES + ['queue_ratio', 'queue_diff', 'net_flow_diff',
                    'trade_imbalance', 'ofi_momentum', 'level_age_diff']
                    if c in sample.columns]
    log.info(f"Feature columns ({len(feature_cols)}): {feature_cols}")

    # ── 4. Walk-forward ──
    log.info(f"\nWalk-forward: {TRAIN_DAYS}d train, {OOT_DAYS}d OOT, {SLIDE_DAYS}d slide (SLIDING)")

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

        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df['target'].values.astype(int)
        X_oot = oot_df[feature_cols].values.astype(np.float32)
        y_oot = oot_df['target'].values.astype(int)

        # Replace inf/nan
        X_train = np.nan_to_num(X_train, nan=0., posinf=100., neginf=-100.)
        X_oot = np.nan_to_num(X_oot, nan=0., posinf=100., neginf=-100.)

        # Train LGBM classifier — use last 20% of TRAIN as eval set (NOT OOT)
        # FIXED: using OOT as eval_set leaks test data into model selection (HC audit 2026-06-25)
        val_split = int(len(X_train) * 0.8)
        X_tr, X_val = X_train[:val_split], X_train[val_split:]
        y_tr, y_val = y_train[:val_split], y_train[val_split:]
        model = lgb.LGBMClassifier(**LGBM_PARAMS)
        model.fit(X_tr, y_tr, eval_set=[(X_val, y_val)],
                  callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)])

        # Predict probabilities
        pred_prob = model.predict_proba(X_oot)[:, 1]
        oot_df = oot_df.copy()
        oot_df['pred_prob'] = pred_prob

        # Feature importance
        fi = dict(zip(feature_cols, model.feature_importances_))
        for feat, imp in fi.items():
            if feat not in fi_accum:
                fi_accum[feat] = []
            fi_accum[feat].append(imp)

        # Evaluate at different thresholds
        baseline_wr = y_oot.mean()
        baseline_pnl = oot_df['net_ticks'].sum()
        baseline_n = len(oot_df)

        best_threshold = 0.5
        best_pf = 0
        threshold_results = {}

        for thresh in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
            selected = oot_df[oot_df['pred_prob'] >= thresh]
            if len(selected) < 5:
                continue
            sel_wr = selected['target'].mean()
            sel_pnl = selected['net_ticks'].sum()
            winners = selected[selected['net_ticks'] > 0]['net_ticks'].sum()
            losers = abs(selected[selected['net_ticks'] < 0]['net_ticks'].sum())
            pf = winners / max(losers, 1e-6)

            threshold_results[thresh] = {
                'n_trades': len(selected),
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
            'thresholds': threshold_results,
            'best_threshold': best_threshold,
            'best_pf': float(best_pf),
        }
        fold_results.append(fold_res)

        # Collect all OOT trades for aggregate analysis
        oot_df['fold'] = fold_num
        all_trades.append(oot_df)

        # Log fold summary
        best_t = threshold_results.get(best_threshold, {})
        log.info(f"  Fold {fold_num:2d} | OOT={','.join(oot_dates)} | "
                 f"baseline: WR={baseline_wr:.3f}, PnL={baseline_pnl:+.1f}t | "
                 f"best@{best_threshold}: n={best_t.get('n_trades','?')}, "
                 f"WR={best_t.get('wr',0):.3f}, PF={best_t.get('pf',0):.2f}")

        fold_num += 1
        start_idx += SLIDE_DAYS

    log.info(f"\n{'='*70}")
    log.info(f"AGGREGATE RESULTS ({fold_num} folds)")
    log.info(f"{'='*70}")

    if not all_trades:
        log.error("No valid folds completed")
        sys.exit(1)

    all_trades_df = pd.concat(all_trades, ignore_index=True)

    # ── 5. Aggregate metrics ──
    total_n = len(all_trades_df)
    total_wr = all_trades_df['target'].mean()
    total_pnl = all_trades_df['net_ticks'].sum()

    log.info(f"\n--- BASELINE (no filter) ---")
    log.info(f"  Total entries: {total_n}")
    log.info(f"  WR: {total_wr:.3f}")
    log.info(f"  Total PnL: {total_pnl:+.1f} ticks")
    log.info(f"  Per-trade: {total_pnl/total_n:+.3f} ticks")

    # Evaluate key thresholds on aggregate
    log.info(f"\n--- FILTERED ENTRY PERFORMANCE ---")
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

        # Daily Sharpe & Sortino
        daily = sel.groupby('date')['net_ticks'].sum()
        sharpe = float(daily.mean() / daily.std()) if len(daily) > 1 and daily.std() > 0 else 0
        downside = daily[daily < 0]
        sortino = float(daily.mean() / downside.std()) if len(downside) > 1 and downside.std() > 0 else (
            999.0 if daily.mean() > 0 else 0)

        agg_results[str(thresh)] = {
            'n_trades': len(sel),
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

    # ── 5b. Per-side breakdown ──
    best_agg_thresh = max(agg_results.items(), key=lambda x: x[1]['pf'])[0] if agg_results else '0.55'
    log.info(f"\n--- PER-SIDE BREAKDOWN (best threshold = {best_agg_thresh}) ---")
    best_thresh_val_tmp = float(best_agg_thresh) if agg_results else 0.55
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
        log.info(f"  {side.upper()}: n={len(side_df)}, WR={sw:.3f}, PnL={sp:+.1f}t, PF={spf:.3f}")

    # ── 6. Regime analysis ──
    log.info(f"\n--- REGIME ANALYSIS ---")
    # Use best performing threshold
    best_agg_thresh = max(agg_results.items(), key=lambda x: x[1]['pf'])[0] if agg_results else '0.55'
    best_thresh_val = float(best_agg_thresh)

    filtered = all_trades_df[all_trades_df['pred_prob'] >= best_thresh_val]
    if len(filtered) > 10:
        gap, green_stats, red_stats = compute_regime_gap(filtered)
        log.info(f"  At threshold {best_thresh_val}:")
        log.info(f"  Green: Sharpe={green_stats.get('sharpe',0):.2f}, n_days={green_stats.get('n_days',0)}")
        log.info(f"  Red:   Sharpe={red_stats.get('sharpe',0):.2f}, n_days={red_stats.get('n_days',0)}")
        log.info(f"  Regime gap: {gap:.3f} {'✓ PASS' if gap <= 0.50 else '✗ FAIL'}")
    else:
        gap = 999.0
        green_stats = {}
        red_stats = {}
        log.info("  Insufficient trades for regime analysis")

    # ── 7. Feature importance ──
    log.info(f"\n--- FEATURE IMPORTANCE (mean gain across folds) ---")
    fi_summary = {}
    for feat, gains in fi_accum.items():
        fi_summary[feat] = {
            'mean_gain': float(np.mean(gains)),
            'std_gain': float(np.std(gains)),
        }

    sorted_features = sorted(fi_summary.items(), key=lambda x: x[1]['mean_gain'], reverse=True)
    for rank, (feat, fi_stat) in enumerate(sorted_features):
        fi_stat['rank'] = rank + 1
        if rank < 15:
            log.info(f"  {rank+1:2d}. {feat:35s} gain={fi_stat['mean_gain']:.1f} ± {fi_stat['std_gain']:.1f}")

    # ── 8. IC analysis ──
    ic = np.corrcoef(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0, 1]
    rank_ic = scipy_stats.spearmanr(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0]
    log.info(f"\n--- INFORMATION COEFFICIENT ---")
    log.info(f"  Pearson IC (prob vs net_ticks): {ic:.4f}")
    log.info(f"  Rank IC (Spearman): {rank_ic:.4f}")

    # ── 9. Save results ──
    results = {
        'timestamp': datetime.now().isoformat(),
        'n_dates': len(valid_dates),
        'n_folds': fold_num,
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'fifo_config': FIFO_CONFIG,
        'walk_forward': {
            'train_days': TRAIN_DAYS,
            'oot_days': OOT_DAYS,
            'slide_days': SLIDE_DAYS,
        },
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
        'ic': {
            'pearson': float(ic),
            'spearman': float(rank_ic),
        },
        'fold_results': fold_results,
        'date_regimes': date_regimes,
    }

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    # Save all OOT trades for post-hoc analysis
    trades_path = OUTPUT_DIR / "all_oot_trades.parquet"
    save_cols = [c for c in all_trades_df.columns if all_trades_df[c].dtype.kind in 'iufbOS']
    all_trades_df[save_cols].to_parquet(trades_path, index=False)
    log.info(f"Saved {len(all_trades_df)} OOT trades to {trades_path}")

    log.info(f"\nResults saved to {OUTPUT_DIR}")

    # ── 10. MLflow logging ──
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("queue_entry_selector")
        with mlflow.start_run(run_name=f"v2_{datetime.now().strftime('%Y%m%d_%H%M')}"):
            mlflow.log_params({
                'version': 'v2_universal',
                'fifo_config': FIFO_CONFIG,
                'n_features': len(feature_cols),
                'n_dates': len(valid_dates),
                'n_folds': fold_num,
                'train_days': TRAIN_DAYS,
                'oot_days': OOT_DAYS,
            })
            mlflow.log_metrics({
                'baseline_wr': float(total_wr),
                'baseline_pnl': float(total_pnl),
                'ic_pearson': float(ic),
                'ic_spearman': float(rank_ic),
                'regime_gap': float(gap),
            })
            # Log best threshold metrics
            if agg_results:
                best = agg_results[best_agg_thresh]
                mlflow.log_metrics({
                    f'best_thresh': float(best_agg_thresh),
                    f'best_wr': best['wr'],
                    f'best_pf': best['pf'],
                    f'best_sharpe': best['sharpe'],
                    f'best_n_trades': best['n_trades'],
                })
            mlflow.log_artifact(str(out_path))
        log.info("MLflow run logged")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


if __name__ == '__main__':
    main()
