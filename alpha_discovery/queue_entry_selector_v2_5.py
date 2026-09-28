#!/usr/bin/env python3
"""
queue_entry_selector_v2_5.py — Queue Entry Selection v2.5 (side-specific + targeted features)
==============================================================================================

v2.5 improvements over v2 (targeted, NOT kitchen-sink like v3):
  1. SIDE-SPECIFIC models: separate LightGBM for long vs short (no feature swapping)
     v3 showed swapping 48 features = noise. Native bid/ask with 21 features is the test.
  2. +3 NEW FEATURES (carefully chosen): cancel_pressure_ratio, ofi_decay_ratio, total_trade_rate
  3. BOTH tp4sl3 and tp8sl5 FIFO configs tested
  4. v2's proven LGBM params (not the deeper/conservative that failed in v3)

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

OUTPUT_DIR = BASE / "output" / "queue_entry_selector_v2_5"
LOG_PATH = BASE / "logs" / "queue_entry_selector_v2_5.log"

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
FIFO_CONFIGS = ['tp4sl3', 'tp8sl5']

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

# v2.5: NO swapping — side-specific models see native bid/ask features

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


def load_fifo_labels(date_str, fifo_config):
    """Load FIFO labels for a date. Returns DataFrame with ts_ns + label columns."""
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
        'long_hit_tp': data.get(f'{prefix}_long_hit_tp', np.zeros(len(data['ts_ns']))),
        'short_filled': data[f'{prefix}_short_filled'],
        'short_net_ticks': data[f'{prefix}_short_net_ticks'],
        'short_hit_tp': data.get(f'{prefix}_short_hit_tp', np.zeros(len(data['ts_ns']))),
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
    """Add derived features — v2's proven set + 3 targeted additions."""
    # ── v2 proven features ──
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

    # ── v2.5 NEW: 3 targeted features ──
    # 1. Cancel pressure ratio — which side is losing liquidity faster
    bid_cancel = df['bid_cancel_rate_1s'].clip(lower=0)
    ask_cancel = df['ask_cancel_rate_1s'].clip(lower=0)
    total_cancel = bid_cancel + ask_cancel
    df['cancel_pressure_ratio'] = np.where(
        total_cancel > 1e-6,
        (bid_cancel - ask_cancel) / total_cancel,
        0.0
    )

    # 2. OFI decay ratio — momentum exhaustion signal
    ofi_10 = df['ofi_10s'].clip(lower=-1e6, upper=1e6)
    df['ofi_decay_ratio'] = np.where(
        np.abs(ofi_10) > 1e-6,
        df['ofi_5s'] / ofi_10.clip(lower=1e-6),
        0.0
    )
    df['ofi_decay_ratio'] = df['ofi_decay_ratio'].clip(-5, 5)

    # 3. Total trade rate (activity level — high activity = more informed flow)
    df['total_trade_rate'] = df['bid_trade_rate_1s'] + df['ask_trade_rate_1s']

    return df


def prepare_side_data(merged_df, date_str, side):
    """Create training rows for one side. NO feature swapping — native bid/ask."""
    filled_col = f'{side}_filled'
    net_col = f'{side}_net_ticks'

    mask = merged_df[filled_col] == True
    subset = merged_df[mask].copy()

    if len(subset) == 0:
        return None

    subset['target'] = (subset[net_col] > 0).astype(int)
    subset['net_ticks'] = subset[net_col].astype(float)
    subset['side'] = side
    subset['date'] = date_str

    return subset


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

def run_side_specific_wf(all_merged, valid_dates, date_regimes, fifo_config):
    """Run side-specific walk-forward: separate models for long and short."""
    log.info(f"\n{'='*70}")
    log.info(f"v2.5 SIDE-SPECIFIC | {fifo_config} | {len(valid_dates)} dates")
    log.info(f"{'='*70}")

    # Prepare side-specific data per date
    side_data = {'long': {}, 'short': {}}
    for date_str, merged in all_merged.items():
        for side in ['long', 'short']:
            sd = prepare_side_data(merged, date_str, side)
            if sd is not None and len(sd) >= 20:
                sd['regime'] = date_regimes[date_str]
                side_data[side][date_str] = sd

    log.info(f"  Long dates: {len(side_data['long'])}, Short dates: {len(side_data['short'])}")

    # Get feature columns from sample
    sample = list(all_merged.values())[0]
    feature_cols = [c for c in RAW_FEATURES + ['queue_ratio', 'queue_diff', 'net_flow_diff',
                    'trade_imbalance', 'ofi_momentum', 'level_age_diff',
                    'cancel_pressure_ratio', 'ofi_decay_ratio', 'total_trade_rate']
                    if c in sample.columns]
    log.info(f"  Features ({len(feature_cols)}): {feature_cols}")

    all_trades = []
    fi_accum = {'long': {}, 'short': {}}

    fold_num = 0
    start_idx = TRAIN_DAYS

    while start_idx + OOT_DAYS <= len(valid_dates):
        train_dates = valid_dates[start_idx - TRAIN_DAYS : start_idx]
        oot_dates = valid_dates[start_idx : start_idx + OOT_DAYS]

        fold_trades = []
        for side in ['long', 'short']:
            side_train = [d for d in train_dates if d in side_data[side]]
            side_oot = [d for d in oot_dates if d in side_data[side]]

            if len(side_train) < 15 or len(side_oot) < 2:
                continue

            train_df = pd.concat([side_data[side][d] for d in side_train], ignore_index=True)
            oot_df = pd.concat([side_data[side][d] for d in side_oot], ignore_index=True)

            if len(train_df) < 200 or len(oot_df) < 30:
                continue

            X_train = train_df[feature_cols].values.astype(np.float32)
            y_train = train_df['target'].values.astype(int)
            X_oot = oot_df[feature_cols].values.astype(np.float32)

            X_train = np.nan_to_num(X_train, nan=0., posinf=100., neginf=-100.)
            X_oot = np.nan_to_num(X_oot, nan=0., posinf=100., neginf=-100.)

            val_split = int(len(X_train) * 0.8)
            model = lgb.LGBMClassifier(**LGBM_PARAMS)
            model.fit(X_train[:val_split], y_train[:val_split],
                      eval_set=[(X_train[val_split:], y_train[val_split:])],
                      callbacks=[lgb.log_evaluation(0), lgb.early_stopping(30, verbose=False)])

            pred_prob = model.predict_proba(X_oot)[:, 1]
            oot_df = oot_df.copy()
            oot_df['pred_prob'] = pred_prob
            oot_df['fold'] = fold_num
            fold_trades.append(oot_df)

            # Feature importance
            fi = dict(zip(feature_cols, model.feature_importances_))
            for feat, imp in fi.items():
                if feat not in fi_accum[side]:
                    fi_accum[side][feat] = []
                fi_accum[side][feat].append(imp)

        if fold_trades:
            fold_df = pd.concat(fold_trades, ignore_index=True)
            all_trades.append(fold_df)
            n_l = len(fold_df[fold_df['side'] == 'long'])
            n_s = len(fold_df[fold_df['side'] == 'short'])
            log.info(f"  Fold {fold_num:2d} | OOT={','.join(oot_dates)} | L:{n_l} S:{n_s} | "
                     f"WR={fold_df['target'].mean():.3f}")

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

    log.info(f"\n--- BASELINE (no filter) ---")
    log.info(f"  Total entries: {total_n}, WR: {total_wr:.3f}, Per-trade: {total_pnl/total_n:+.3f} ticks")

    # Evaluate thresholds
    log.info(f"\n--- FILTERED PERFORMANCE ---")
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

        # Per-side
        side_stats = {}
        for side in ['long', 'short']:
            s = sel[sel['side'] == side]
            if len(s) >= 5:
                sw = s['target'].mean()
                sp = s['net_ticks'].sum()
                swin = s[s['net_ticks'] > 0]['net_ticks'].sum()
                slos = abs(s[s['net_ticks'] < 0]['net_ticks'].sum())
                side_stats[side] = {'n': len(s), 'wr': float(sw), 'pnl': float(sp),
                                    'pf': float(swin / max(slos, 1e-6))}

        # Regime gap
        gap, green_stats, red_stats = compute_regime_gap(sel)

        agg_results[str(thresh)] = {
            'n_trades': len(sel), 'wr': float(sel_wr),
            'pnl': float(sel_pnl), 'per_trade': float(per_trade),
            'pf': float(pf), 'sharpe': float(sharpe), 'sortino': float(sortino),
            'per_side': side_stats,
            'regime_gap': float(gap), 'regime_pass': gap <= 0.50,
            'green': green_stats, 'red': red_stats,
        }

        regime_str = '✓ PASS' if gap <= 0.50 else '✗ FAIL'
        log.info(f"  thresh={thresh:.2f}: n={len(sel):5d}, WR={sel_wr:.3f}, "
                 f"PF={pf:.3f}, Sharpe={sharpe:.2f}, Sortino={sortino:.2f}, "
                 f"regime_gap={gap:.3f} {regime_str}")
        for side, ss in side_stats.items():
            log.info(f"    {side.upper()}: n={ss['n']}, WR={ss['wr']:.3f}, PF={ss['pf']:.3f}")

    # Feature importance
    log.info(f"\n--- FEATURE IMPORTANCE ---")
    fi_summary = {}
    for side in ['long', 'short']:
        log.info(f"  [{side.upper()} model]")
        for feat, gains in sorted(fi_accum[side].items(), key=lambda x: -np.mean(x[1]))[:10]:
            mg = float(np.mean(gains))
            fi_summary[f'{side}_{feat}'] = mg
            log.info(f"    {feat:35s} gain={mg:.1f}")

    # IC
    ic = np.corrcoef(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0, 1]
    rank_ic = scipy_stats.spearmanr(all_trades_df['pred_prob'], all_trades_df['net_ticks'])[0]
    log.info(f"\n  IC: Pearson={ic:.4f}, Spearman={rank_ic:.4f}")
    for side in ['long', 'short']:
        s = all_trades_df[all_trades_df['side'] == side]
        if len(s) > 30:
            side_ic = scipy_stats.spearmanr(s['pred_prob'], s['net_ticks'])[0]
            log.info(f"  IC [{side}]: Spearman={side_ic:.4f}")

    return {
        'fifo_config': fifo_config,
        'model_type': 'side_specific',
        'n_dates': len(valid_dates),
        'n_folds': fold_num,
        'n_features': len(feature_cols),
        'feature_cols': feature_cols,
        'baseline': {'n_trades': total_n, 'wr': float(total_wr),
                     'pnl': float(total_pnl), 'per_trade': float(total_pnl / max(total_n, 1))},
        'filtered_performance': agg_results,
        'feature_importance': fi_summary,
        'ic': {'pearson': float(ic), 'spearman': float(rank_ic)},
        'all_trades_df': all_trades_df,
    }


def main():
    log.info("=" * 70)
    log.info("Queue Entry Selector v2.5 — Side-Specific Models + Targeted Features")
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

    # ── 2. Load queue features + regime classification (shared across configs) ──
    log.info("Loading queue features...")
    queue_cache = {}
    date_regimes = {}
    for date_str in overlap:
        queue = load_queue_features(date_str)
        if queue is None:
            continue
        queue = engineer_features(queue)
        queue_cache[date_str] = queue
        date_regimes[date_str] = classify_day_regime(queue)

    log.info(f"Valid queue dates: {len(queue_cache)}")
    log.info(f"Regimes: {pd.Series(date_regimes).value_counts().to_dict()}")

    # ── 3. Run each FIFO config ──
    all_results = {}
    for fifo_config in FIFO_CONFIGS:
        log.info(f"\nLoading FIFO labels for {fifo_config}...")
        all_merged = {}
        for date_str in sorted(queue_cache.keys()):
            fifo = load_fifo_labels(date_str, fifo_config)
            if fifo is None:
                continue
            merged, n_before, n_after = join_queue_fifo(queue_cache[date_str], fifo)
            if n_after < 50:
                continue
            all_merged[date_str] = merged

        valid_dates = sorted(all_merged.keys())
        log.info(f"  {fifo_config}: {len(valid_dates)} valid dates")

        if len(valid_dates) < TRAIN_DAYS + OOT_DAYS:
            log.warning(f"  Not enough dates for {fifo_config}")
            continue

        result = run_side_specific_wf(all_merged, valid_dates, date_regimes, fifo_config)
        if result:
            all_results[fifo_config] = result

    # ── 4. Save results ──
    log.info(f"\n{'='*70}")
    log.info("SUMMARY")
    log.info(f"{'='*70}")

    save_results = {'timestamp': datetime.now().isoformat(), 'version': 'v2.5'}
    for config, res in all_results.items():
        # Save trades
        trades_df = res.pop('all_trades_df')
        save_cols = [c for c in trades_df.columns if trades_df[c].dtype.kind in 'iufbOS']
        trades_df[save_cols].to_parquet(OUTPUT_DIR / f"all_oot_trades_{config}.parquet", index=False)

        # Find best regime-passing threshold
        best_thresh = None
        best_sharpe = -999
        for t, tr in res['filtered_performance'].items():
            if tr.get('regime_pass') and tr['sharpe'] > best_sharpe:
                best_sharpe = tr['sharpe']
                best_thresh = t

        if best_thresh:
            tr = res['filtered_performance'][best_thresh]
            log.info(f"  [{config}] Best regime-passing: thresh={best_thresh}, "
                     f"n={tr['n_trades']}, WR={tr['wr']:.3f}, PF={tr['pf']:.3f}, "
                     f"Sharpe={tr['sharpe']:.2f}, Sortino={tr['sortino']:.2f}")
        else:
            log.info(f"  [{config}] No threshold passes regime gate")

        save_results[config] = res

    out_path = OUTPUT_DIR / "results.json"
    with open(out_path, 'w') as f:
        json.dump(save_results, f, indent=2, default=str)
    log.info(f"\nResults saved to {OUTPUT_DIR}")

    # ── 5. MLflow logging ──
    try:
        import mlflow
        mlflow.set_tracking_uri("http://localhost:5000")
        mlflow.set_experiment("queue_entry_selector")
        for config, res in all_results.items():
            with mlflow.start_run(run_name=f"v2.5_{config}_{datetime.now().strftime('%Y%m%d_%H%M')}"):
                mlflow.log_params({
                    'version': 'v2.5_side_specific',
                    'fifo_config': config,
                    'model_type': 'side_specific',
                    'n_features': res['n_features'],
                    'n_dates': res['n_dates'],
                    'n_folds': res['n_folds'],
                })
                mlflow.log_metrics({
                    'baseline_wr': res['baseline']['wr'],
                    'ic_pearson': res['ic']['pearson'],
                    'ic_spearman': res['ic']['spearman'],
                })
                # Best threshold metrics
                best_thresh = None
                best_sharpe = -999
                for t, tr in res['filtered_performance'].items():
                    if tr.get('regime_pass') and tr['sharpe'] > best_sharpe:
                        best_sharpe = tr['sharpe']
                        best_thresh = t
                if best_thresh:
                    tr = res['filtered_performance'][best_thresh]
                    mlflow.log_metrics({
                        'best_thresh': float(best_thresh),
                        'best_wr': tr['wr'],
                        'best_pf': tr['pf'],
                        'best_sharpe': tr['sharpe'],
                        'best_n_trades': tr['n_trades'],
                        'best_regime_gap': tr['regime_gap'],
                    })
                mlflow.log_artifact(str(out_path))
        log.info("MLflow runs logged")
    except Exception as e:
        log.warning(f"MLflow logging failed: {e}")


if __name__ == '__main__':
    main()
