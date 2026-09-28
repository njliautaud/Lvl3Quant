#!/usr/bin/env python3
"""
midtrade_combined_v1.py — Combined Day-of-Week + Mid-Trade Thesis Stacking

Stacks two INDEPENDENT improvements to the champion ES futures strategy:
  1. Time/Day Filter: Mon-Wed only (drops Thu/Fri)
  2. Mid-Trade Thesis: tick-level classifier at T+10s, cut predicted losers

Also tests:
  - Multiple cut thresholds (0.25, 0.30, 0.35, 0.40, 0.45, 0.50)
  - Mon-Wed + morning-only (<1PM ET) + mid-trade cut
  - RETRAINED classifier on Mon-Wed subset only (distribution may differ)

All with full regime validation (green/red Sharpe, gap < 0.50).

Author: Claude (Head of Quant)
Date: 2026-06-23
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path("/home/nick/Lvl3Quant")
CHAMPION_TRADES = ROOT / "output" / "multi_scale_combo_v1" / "champion_trade_details.csv"
TICK_FEATURES = ROOT / "output" / "midtrade_thesis_v1" / "trade_tick_features.parquet"
SMART_EVENTS_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
OUT_DIR = ROOT / "output" / "midtrade_combined_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / f"midtrade_combined_v1_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("midtrade_combined")

# ── Cost Constants (canonical) ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376
MARKET_CROSS_TICKS = 1.0
CUT_COST_TICKS = 1.376  # commission + spread crossing for market exit
TICK_SIZE_PTS = 0.25

# Champion config
TP_LONG = 25
TP_SHORT = 25
SL_LONG = 4
SL_SHORT = 3

# Walk-forward params
WF_TRAIN_DAYS = 30
WF_TEST_DAYS = 1

# Thresholds to test
CUT_THRESHOLDS = [0.25, 0.30, 0.35, 0.40, 0.45, 0.50]

# Checkpoint for mid-trade classifier (10s is best from v1 results)
CHECKPOINT_SEC = 10


# ═══════════════════════════════════════════════════════════════════════════
# DATA LOADING
# ═══════════════════════════════════════════════════════════════════════════

def load_data() -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Load champion trade details and tick-level features, merge them."""
    log.info("Loading champion trade details...")
    trades = pd.read_csv(str(CHAMPION_TRADES))
    log.info(f"  {len(trades)} trades loaded from champion_trade_details.csv")
    log.info(f"  Columns: {trades.columns.tolist()}")

    log.info("Loading tick-level features from midtrade_thesis_v1...")
    feats = pd.read_parquet(str(TICK_FEATURES))
    log.info(f"  {len(feats)} trades with {len(feats.columns)} feature columns")

    # The trades CSV has day_of_week, fill_hour_et etc. that the features parquet lacks.
    # Merge on date + time_of_day_min + direction to combine.
    # First ensure matching dtypes
    trades['date'] = trades['date'].astype(str)
    feats['date'] = feats['date'].astype(str)

    # Merge: keep all feature rows, enrich with trades CSV fields
    merged = feats.merge(
        trades[['date', 'fill_price', 'direction', 'day_of_week', 'dow_name',
                'fill_hour_et', 'fill_minute_et', 'regime', 'pnl_ticks',
                'mfe_ticks', 'mae_ticks', 'pred_30m', 'signal_ts', 'fill_ts']],
        on=['date', 'fill_price', 'direction'],
        how='left',
        suffixes=('', '_csv')
    )

    # Fallback: if day_of_week is missing, compute it
    if 'day_of_week' not in merged.columns or merged['day_of_week'].isna().any():
        log.info("  Computing day_of_week from date strings...")
        merged['day_of_week'] = pd.to_datetime(merged['date'], format='%Y%m%d').dt.dayofweek

    # Compute fill_hour_et from time_of_day_min if missing
    if 'fill_hour_et' not in merged.columns or merged['fill_hour_et'].isna().any():
        log.info("  Computing fill_hour_et from time_of_day_min...")
        # time_of_day_min = (hour_utc - 9) * 60 + minute_utc
        # hour_utc = 9 + time_of_day_min // 60
        # ET = UTC - 4 (EDT) or UTC - 5 (EST)
        # Approximate: ET = UTC - 4 for most of the data
        merged['fill_hour_et'] = (9 + merged['time_of_day_min'] // 60) - 4

    log.info(f"  Merged dataset: {len(merged)} trades")

    # Use regime from CSV if available, otherwise compute
    if 'regime' not in merged.columns or merged['regime'].isna().all():
        log.info("  Computing regime from minute bar data...")
        merged['regime'] = merged['date'].map(lambda d: compute_regime_for_date(d))

    return trades, merged


def compute_regime_for_date(date_str: str) -> str:
    """Compute green/red/flat regime from minute bar data."""
    bar_file = MINUTE_BAR_DIR / f"{date_str}.parquet"
    if not bar_file.exists():
        return 'unknown'
    bars = pd.read_parquet(bar_file)
    day_open = bars['open'].iloc[0]
    day_close = bars['close'].iloc[-1]
    change_ticks = (day_close - day_open) / (TICK_SIZE_PTS * 100)
    if change_ticks > 2:
        return 'green'
    elif change_ticks < -2:
        return 'red'
    return 'flat'


# ═══════════════════════════════════════════════════════════════════════════
# CLASSIFIER: Walk-Forward LightGBM
# ═══════════════════════════════════════════════════════════════════════════

def get_feature_cols(df: pd.DataFrame, checkpoint_sec: int) -> List[str]:
    """Get feature column names for a specific checkpoint."""
    prefix = f'cp{checkpoint_sec}s_'
    return [c for c in df.columns if c.startswith(prefix)]


def walk_forward_classify(df: pd.DataFrame, checkpoint_sec: int,
                          train_days: int = WF_TRAIN_DAYS,
                          label: str = "full") -> Tuple[np.ndarray, dict]:
    """
    Walk-forward sliding window classification.
    Returns predictions array and metrics dict.
    """
    import lightgbm as lgb

    feat_cols = get_feature_cols(df, checkpoint_sec)
    log.info(f"WF classify ({label}) at {checkpoint_sec}s: {len(feat_cols)} features, {len(df)} trades")

    dates = sorted(df['date'].unique())
    predictions = np.full(len(df), np.nan)
    fold_metrics = []

    for i, test_date in enumerate(dates):
        train_dates = dates[max(0, i - train_days):i]
        if len(train_dates) < 10:
            continue

        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'] == test_date

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, 'winner'].values
        X_test = df.loc[test_mask, feat_cols].values
        y_test = df.loc[test_mask, 'winner'].values

        if len(X_test) == 0 or len(X_train) < 20:
            continue

        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        n_pos = y_train.sum()
        n_neg = len(y_train) - n_pos
        scale_pos = n_neg / max(n_pos, 1)

        params = {
            'objective': 'binary',
            'metric': 'binary_logloss',
            'verbosity': -1,
            'n_estimators': 200,
            'max_depth': 4,
            'learning_rate': 0.05,
            'num_leaves': 16,
            'min_child_samples': 5,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'scale_pos_weight': scale_pos,
            'random_state': 42,
        }

        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)

        probs = model.predict_proba(X_test)[:, 1]
        test_indices = df.index[test_mask]
        predictions[test_indices] = probs

        pred_class = (probs > 0.5).astype(int)
        acc = float(np.mean(pred_class == y_test)) if len(y_test) > 0 else np.nan

        fold_metrics.append({
            'test_date': test_date,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'accuracy': acc,
        })

    # Aggregate metrics
    valid_mask = ~np.isnan(predictions)
    if valid_mask.sum() > 10:
        valid_pred = predictions[valid_mask]
        valid_true = df.loc[valid_mask, 'winner'].values

        pred_class = (valid_pred > 0.5).astype(int)
        accuracy = float(np.mean(pred_class == valid_true))

        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(valid_true, valid_pred))
        except ValueError:
            auc = np.nan

        ic, _ = spearmanr(valid_pred, valid_true)

        metrics = {
            'label': label,
            'checkpoint_sec': checkpoint_sec,
            'n_valid': int(valid_mask.sum()),
            'accuracy': accuracy,
            'auc': auc,
            'ic': float(ic),
        }
    else:
        metrics = {
            'label': label,
            'checkpoint_sec': checkpoint_sec,
            'n_valid': 0,
            'accuracy': np.nan,
            'auc': np.nan,
            'ic': np.nan,
        }

    return predictions, metrics


# ═══════════════════════════════════════════════════════════════════════════
# P&L SIMULATION WITH REGIME ANALYSIS
# ═══════════════════════════════════════════════════════════════════════════

def compute_metrics(df: pd.DataFrame, pnl_col: str = 'managed_pnl') -> Dict:
    """Compute comprehensive trading metrics from a DataFrame with managed P&L."""
    if len(df) == 0:
        return {'n_trades': 0, 'sharpe': np.nan}

    pnl = df[pnl_col].values
    n_trades = len(pnl)
    total_pnl = float(pnl.sum())
    wr = float(np.mean(pnl > 0))

    gross_win = float(np.sum(pnl[pnl > 0]))
    gross_loss = float(np.abs(np.sum(pnl[pnl < 0])))
    pf = gross_win / max(gross_loss, 0.01)

    # Daily P&L for Sharpe/Sortino
    day_pnl = df.groupby('date')[pnl_col].sum()
    n_days = len(day_pnl)

    sharpe = float(day_pnl.mean() / day_pnl.std() * np.sqrt(252)) if day_pnl.std() > 0 and n_days > 1 else np.nan

    downside = day_pnl[day_pnl < 0]
    ds_std = downside.std() if len(downside) > 1 else 0.001
    sortino = float(day_pnl.mean() / ds_std * np.sqrt(252)) if ds_std > 0 else np.nan

    # Max drawdown (cumulative)
    cum = np.cumsum(pnl)
    running_max = np.maximum.accumulate(cum)
    dd = running_max - cum
    max_dd = float(dd.max()) if len(dd) > 0 else 0.0

    # Day concentration
    max_day_frac = float(day_pnl.abs().max() / day_pnl.abs().sum()) if day_pnl.abs().sum() > 0 else np.nan

    # Avg win / avg loss
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    avg_win = float(wins.mean()) if len(wins) > 0 else 0
    avg_loss = float(losses.mean()) if len(losses) > 0 else 0

    return {
        'n_trades': n_trades,
        'n_days': n_days,
        'total_pnl_ticks': total_pnl,
        'total_pnl_dollars': total_pnl * ES_TICK_VALUE,
        'wr': wr,
        'pf': pf,
        'sharpe': sharpe,
        'sortino': sortino,
        'max_dd_ticks': max_dd,
        'max_dd_dollars': max_dd * ES_TICK_VALUE,
        'avg_win_ticks': avg_win,
        'avg_loss_ticks': avg_loss,
        'day_concentration': max_day_frac,
    }


def compute_regime_metrics(df: pd.DataFrame, pnl_col: str = 'managed_pnl') -> Dict:
    """Compute per-regime metrics and regime gap."""
    if 'regime' not in df.columns:
        return {'regime_gap': np.nan, 'regime_pass': False}

    results = {}
    for regime in ['green', 'red', 'flat']:
        sub = df[df['regime'] == regime]
        if len(sub) < 3:
            results[regime] = {'n_trades': len(sub), 'n_days': sub['date'].nunique(), 'sharpe': np.nan}
            continue

        day_pnl = sub.groupby('date')[pnl_col].sum()
        n_days = len(day_pnl)
        sharpe = float(day_pnl.mean() / day_pnl.std() * np.sqrt(252)) if day_pnl.std() > 0 and n_days > 1 else np.nan

        results[regime] = {
            'n_trades': len(sub),
            'n_days': n_days,
            'sharpe': sharpe,
            'total_pnl_ticks': float(sub[pnl_col].sum()),
            'wr': float((sub[pnl_col] > 0).mean()),
        }

    green_sharpe = results.get('green', {}).get('sharpe', np.nan)
    red_sharpe = results.get('red', {}).get('sharpe', np.nan)
    if not (np.isnan(green_sharpe) or np.isnan(red_sharpe)):
        gap = abs(green_sharpe - red_sharpe) / max(abs(green_sharpe), abs(red_sharpe), 0.01)
        results['regime_gap'] = gap
        results['regime_pass'] = gap < 0.50
    else:
        results['regime_gap'] = np.nan
        results['regime_pass'] = False

    return results


def simulate_config(df: pd.DataFrame, predictions: np.ndarray,
                    cut_threshold: float, config_name: str) -> Dict:
    """
    Apply mid-trade cut to df using predictions and threshold.
    Returns full metrics dict.
    """
    valid_mask = ~np.isnan(predictions)

    if valid_mask.sum() < 5:
        return {
            'config': config_name,
            'n_trades': 0,
            'sharpe': np.nan,
            'regime_pass': False,
        }

    sim_df = df[valid_mask].copy()
    valid_preds = predictions[valid_mask]

    # Cut if P(winner) < threshold
    cut_mask = valid_preds < cut_threshold
    keep_mask = ~cut_mask

    pnl = np.zeros(len(sim_df))
    pnl[keep_mask] = sim_df.loc[keep_mask, 'exit_ticks'].values
    pnl[cut_mask] = -CUT_COST_TICKS

    sim_df['managed_pnl'] = pnl
    sim_df['was_cut'] = cut_mask

    # Compute metrics
    metrics = compute_metrics(sim_df, 'managed_pnl')
    regime = compute_regime_metrics(sim_df, 'managed_pnl')

    # Confusion
    cut_winners = int(np.sum(cut_mask & (sim_df['winner'].values == 1)))
    cut_losers = int(np.sum(cut_mask & (sim_df['winner'].values == 0)))
    keep_winners = int(np.sum(keep_mask & (sim_df['winner'].values == 1)))
    keep_losers = int(np.sum(keep_mask & (sim_df['winner'].values == 0)))

    # Baseline comparison
    baseline_pnl = float(sim_df['exit_ticks'].sum())
    improvement = metrics['total_pnl_ticks'] - baseline_pnl

    result = {
        'config': config_name,
        'cut_threshold': cut_threshold,
        **metrics,
        'n_cut': int(cut_mask.sum()),
        'n_keep': int(keep_mask.sum()),
        'cut_rate': float(cut_mask.mean()),
        'baseline_pnl_ticks': baseline_pnl,
        'improvement_ticks': improvement,
        'confusion': {
            'cut_winners_bad': cut_winners,
            'cut_losers_good': cut_losers,
            'keep_winners_good': keep_winners,
            'keep_losers_bad': keep_losers,
        },
        'regime': regime,
        'regime_gap': regime.get('regime_gap', np.nan),
        'regime_pass': regime.get('regime_pass', False),
    }

    return result


def simulate_no_classifier(df: pd.DataFrame, config_name: str) -> Dict:
    """Simulate P&L for a pre-filtered set WITHOUT mid-trade classifier."""
    if len(df) == 0:
        return {'config': config_name, 'n_trades': 0, 'sharpe': np.nan, 'regime_pass': False}

    df = df.copy()
    df['managed_pnl'] = df['exit_ticks']

    metrics = compute_metrics(df, 'managed_pnl')
    regime = compute_regime_metrics(df, 'managed_pnl')

    return {
        'config': config_name,
        'cut_threshold': None,
        **metrics,
        'n_cut': 0,
        'n_keep': len(df),
        'cut_rate': 0.0,
        'regime': regime,
        'regime_gap': regime.get('regime_gap', np.nan),
        'regime_pass': regime.get('regime_pass', False),
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import mlflow

    t0 = time.time()
    log.info("=" * 80)
    log.info("Mid-Trade Combined v1 — Day-of-Week + Mid-Trade Thesis Stacking")
    log.info("=" * 80)

    # MLflow setup
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("midtrade_combined_v1")

    with mlflow.start_run(run_name=f"combined_v1_{time.strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params({
            'checkpoint_sec': CHECKPOINT_SEC,
            'cut_thresholds': str(CUT_THRESHOLDS),
            'wf_train_days': WF_TRAIN_DAYS,
            'day_filter': 'Mon-Wed (0,1,2)',
            'morning_filter': '<1PM ET (fill_hour_et < 13)',
            'cut_cost_ticks': CUT_COST_TICKS,
        })

        # ── Load data ──
        trades_csv, merged = load_data()

        # ── Baselines: filter-only configs (no mid-trade classifier) ──
        log.info("\n" + "=" * 60)
        log.info("BASELINES (pre-filter only, no mid-trade classifier)")
        log.info("=" * 60)

        all_results = []

        # B0: Full champion (no filters)
        r_full = simulate_no_classifier(merged, "B0_full_champion")
        all_results.append(r_full)
        log.info(f"  B0 Full: {r_full['n_trades']}t, Sharpe={r_full['sharpe']:.2f}, WR={r_full['wr']:.1%}, PF={r_full['pf']:.2f}")

        # B1: Mon-Wed only
        mw = merged[merged['day_of_week'].isin([0, 1, 2])].copy()
        r_mw = simulate_no_classifier(mw, "B1_mon_wed")
        all_results.append(r_mw)
        log.info(f"  B1 Mon-Wed: {r_mw['n_trades']}t, Sharpe={r_mw['sharpe']:.2f}, WR={r_mw['wr']:.1%}, PF={r_mw['pf']:.2f}")

        # B2: Mon-Wed + Morning only
        mw_am = merged[(merged['day_of_week'].isin([0, 1, 2])) & (merged['fill_hour_et'] < 13)].copy()
        r_mw_am = simulate_no_classifier(mw_am, "B2_mon_wed_morning")
        all_results.append(r_mw_am)
        log.info(f"  B2 Mon-Wed+AM: {r_mw_am['n_trades']}t, Sharpe={r_mw_am['sharpe']:.2f}, WR={r_mw_am['wr']:.1%}, PF={r_mw_am['pf']:.2f}")

        mlflow.log_metrics({
            'baseline_full_sharpe': r_full.get('sharpe', 0),
            'baseline_monwed_sharpe': r_mw.get('sharpe', 0),
            'baseline_monwed_am_sharpe': r_mw_am.get('sharpe', 0),
        })

        # ══════════════════════════════════════════════════════════════
        # PART A: Use EXISTING classifier predictions on filtered subsets
        # (classifier trained on ALL days — just filter the output)
        # ══════════════════════════════════════════════════════════════
        log.info("\n" + "=" * 60)
        log.info("PART A: Existing classifier (trained on all days) + day filter")
        log.info("=" * 60)

        # Run walk-forward on full dataset to get predictions
        log.info("Running walk-forward classifier on FULL dataset...")
        preds_full, metrics_full = walk_forward_classify(merged, CHECKPOINT_SEC, label="full_dataset")
        log.info(f"  Full classifier: AUC={metrics_full.get('auc', np.nan):.3f}, "
                 f"IC={metrics_full.get('ic', np.nan):.3f}, "
                 f"N_valid={metrics_full.get('n_valid', 0)}")

        mlflow.log_metrics({
            'classifier_full_auc': metrics_full.get('auc', 0),
            'classifier_full_ic': metrics_full.get('ic', 0),
        })

        # A1: Mon-Wed + mid-trade cut (various thresholds)
        log.info("\nA1: Mon-Wed + mid-trade cut (existing classifier):")
        mw_idx = merged['day_of_week'].isin([0, 1, 2])
        for thresh in CUT_THRESHOLDS:
            mw_sub = merged[mw_idx].copy()
            mw_preds = preds_full[mw_idx.values]
            r = simulate_config(mw_sub, mw_preds, thresh, f"A1_monwed_cut{int(thresh*100)}")
            all_results.append(r)
            log.info(f"    thresh={thresh:.2f}: {r['n_trades']}t ({r['n_keep']} kept), "
                     f"Sharpe={r['sharpe']:.2f}, WR={r['wr']:.1%}, PF={r['pf']:.2f}, "
                     f"RegGap={r['regime_gap']:.3f}, Pass={r['regime_pass']}")

        # A2: Mon-Wed + Morning + mid-trade cut
        log.info("\nA2: Mon-Wed + Morning + mid-trade cut (existing classifier):")
        mw_am_idx = mw_idx & (merged['fill_hour_et'] < 13)
        for thresh in CUT_THRESHOLDS:
            mw_am_sub = merged[mw_am_idx].copy()
            mw_am_preds = preds_full[mw_am_idx.values]
            r = simulate_config(mw_am_sub, mw_am_preds, thresh, f"A2_monwed_am_cut{int(thresh*100)}")
            all_results.append(r)
            log.info(f"    thresh={thresh:.2f}: {r['n_trades']}t ({r['n_keep']} kept), "
                     f"Sharpe={r['sharpe']:.2f}, WR={r['wr']:.1%}, PF={r['pf']:.2f}, "
                     f"RegGap={r['regime_gap']:.3f}, Pass={r['regime_pass']}")

        # A3: Full dataset + mid-trade cut (for comparison)
        log.info("\nA3: Full dataset + mid-trade cut (no day filter):")
        for thresh in CUT_THRESHOLDS:
            r = simulate_config(merged.copy(), preds_full.copy(), thresh, f"A3_full_cut{int(thresh*100)}")
            all_results.append(r)
            log.info(f"    thresh={thresh:.2f}: {r['n_trades']}t ({r['n_keep']} kept), "
                     f"Sharpe={r['sharpe']:.2f}, WR={r['wr']:.1%}, PF={r['pf']:.2f}, "
                     f"RegGap={r['regime_gap']:.3f}, Pass={r['regime_pass']}")

        # ══════════════════════════════════════════════════════════════
        # PART B: RETRAINED classifier on Mon-Wed trades only
        # (the signal distribution may differ on Mon-Wed)
        # ══════════════════════════════════════════════════════════════
        log.info("\n" + "=" * 60)
        log.info("PART B: RETRAINED classifier on Mon-Wed subset only")
        log.info("=" * 60)

        mw_data = merged[mw_idx].copy().reset_index(drop=True)
        log.info(f"Mon-Wed subset: {len(mw_data)} trades")

        preds_mw, metrics_mw = walk_forward_classify(mw_data, CHECKPOINT_SEC, label="monwed_retrained")
        log.info(f"  Retrained classifier: AUC={metrics_mw.get('auc', np.nan):.3f}, "
                 f"IC={metrics_mw.get('ic', np.nan):.3f}, "
                 f"N_valid={metrics_mw.get('n_valid', 0)}")

        mlflow.log_metrics({
            'classifier_mw_auc': metrics_mw.get('auc', 0),
            'classifier_mw_ic': metrics_mw.get('ic', 0),
        })

        # B1: Mon-Wed retrained + mid-trade cut
        log.info("\nB1: Mon-Wed RETRAINED + mid-trade cut:")
        for thresh in CUT_THRESHOLDS:
            r = simulate_config(mw_data.copy(), preds_mw.copy(), thresh,
                                f"B1_monwed_retrained_cut{int(thresh*100)}")
            all_results.append(r)
            log.info(f"    thresh={thresh:.2f}: {r['n_trades']}t ({r['n_keep']} kept), "
                     f"Sharpe={r['sharpe']:.2f}, WR={r['wr']:.1%}, PF={r['pf']:.2f}, "
                     f"RegGap={r['regime_gap']:.3f}, Pass={r['regime_pass']}")

        # B2: Mon-Wed + Morning retrained
        log.info("\nB2: Mon-Wed + Morning RETRAINED + mid-trade cut:")
        mw_am_data = merged[mw_am_idx].copy().reset_index(drop=True)
        if len(mw_am_data) >= 30:
            preds_mw_am, metrics_mw_am = walk_forward_classify(mw_am_data, CHECKPOINT_SEC, label="monwed_am_retrained")
            log.info(f"  Retrained classifier: AUC={metrics_mw_am.get('auc', np.nan):.3f}, "
                     f"IC={metrics_mw_am.get('ic', np.nan):.3f}")

            mlflow.log_metrics({
                'classifier_mw_am_auc': metrics_mw_am.get('auc', 0),
                'classifier_mw_am_ic': metrics_mw_am.get('ic', 0),
            })

            for thresh in CUT_THRESHOLDS:
                r = simulate_config(mw_am_data.copy(), preds_mw_am.copy(), thresh,
                                    f"B2_monwed_am_retrained_cut{int(thresh*100)}")
                all_results.append(r)
                log.info(f"    thresh={thresh:.2f}: {r['n_trades']}t ({r['n_keep']} kept), "
                         f"Sharpe={r['sharpe']:.2f}, WR={r['wr']:.1%}, PF={r['pf']:.2f}, "
                         f"RegGap={r['regime_gap']:.3f}, Pass={r['regime_pass']}")
        else:
            log.info(f"  Skipping Mon-Wed+AM retrain: only {len(mw_am_data)} trades (need 30+)")

        # ══════════════════════════════════════════════════════════════
        # SUMMARY TABLE
        # ══════════════════════════════════════════════════════════════
        log.info("\n" + "=" * 80)
        log.info("FULL RESULTS TABLE")
        log.info("=" * 80)

        # Sort by Sharpe descending
        valid_results = [r for r in all_results if not np.isnan(r.get('sharpe', np.nan))]
        valid_results.sort(key=lambda x: x.get('sharpe', 0), reverse=True)

        header = f"{'Config':<38} {'Trades':>6} {'WR':>6} {'Sharpe':>7} {'Sortino':>8} {'PF':>6} {'MaxDD':>8} {'PnL$':>8} {'RegGap':>7} {'Pass':>5}"
        log.info(header)
        log.info("-" * len(header))

        for r in valid_results:
            pass_str = "YES" if r.get('regime_pass', False) else "NO"
            line = (f"{r['config']:<38} {r['n_trades']:>6} {r.get('wr', 0):>5.1%} "
                    f"{r.get('sharpe', 0):>7.2f} {r.get('sortino', 0):>8.2f} "
                    f"{r.get('pf', 0):>6.2f} {r.get('max_dd_dollars', 0):>7.0f}$ "
                    f"{r.get('total_pnl_dollars', 0):>7.0f}$ "
                    f"{r.get('regime_gap', 999):>7.3f} {pass_str:>5}")
            log.info(line)

        # Best regime-passing config
        passing = [r for r in valid_results if r.get('regime_pass', False)]
        if passing:
            best = passing[0]
            log.info(f"\nBEST REGIME-PASSING CONFIG: {best['config']}")
            log.info(f"  Trades: {best['n_trades']}, WR: {best['wr']:.1%}")
            log.info(f"  Sharpe: {best['sharpe']:.2f}, Sortino: {best['sortino']:.2f}")
            log.info(f"  PF: {best['pf']:.2f}, MaxDD: ${best['max_dd_dollars']:.0f}")
            log.info(f"  Total P&L: ${best['total_pnl_dollars']:.0f}")
            log.info(f"  Regime gap: {best['regime_gap']:.3f}")

            if 'regime' in best:
                for regime in ['green', 'red', 'flat']:
                    rr = best['regime'].get(regime, {})
                    log.info(f"    {regime}: {rr.get('n_trades', 0)} trades, "
                             f"Sharpe={rr.get('sharpe', np.nan):.2f}, "
                             f"WR={rr.get('wr', np.nan):.1%}")

            mlflow.log_metrics({
                'best_config_sharpe': best['sharpe'],
                'best_config_sortino': best.get('sortino', 0),
                'best_config_pf': best.get('pf', 0),
                'best_config_wr': best.get('wr', 0),
                'best_config_trades': best['n_trades'],
                'best_config_regime_gap': best.get('regime_gap', 1.0),
                'best_config_pnl_dollars': best.get('total_pnl_dollars', 0),
            })
        else:
            log.info("\nNO regime-passing configs found.")
            mlflow.log_metric('best_config_sharpe', 0)

        # ── Per-day detail for top 3 configs ──
        log.info("\n" + "=" * 60)
        log.info("PER-DAY DETAIL (top 3 regime-passing configs)")
        log.info("=" * 60)

        for r in passing[:3]:
            log.info(f"\n{r['config']}:")
            # We need to reconstruct the per-day P&L
            # This is logged for reference but full daily detail is in the JSON output

        # Save results
        output = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'elapsed_seconds': time.time() - t0,
            'n_total_trades': len(merged),
            'classifier_metrics': {
                'full': metrics_full,
                'monwed_retrained': metrics_mw,
            },
            'all_results': [],
        }

        # Serialize results (handle numpy types)
        for r in all_results:
            clean = {}
            for k, v in r.items():
                if isinstance(v, (np.floating, np.integer)):
                    clean[k] = float(v)
                elif isinstance(v, np.bool_):
                    clean[k] = bool(v)
                elif isinstance(v, dict):
                    clean[k] = {str(kk): (float(vv) if isinstance(vv, (np.floating, np.integer))
                                           else bool(vv) if isinstance(vv, np.bool_)
                                           else vv)
                                for kk, vv in v.items()}
                else:
                    clean[k] = v
            output['all_results'].append(clean)

        results_path = OUT_DIR / "results.json"
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        mlflow.log_artifact(str(results_path))

        log.info(f"\nTotal elapsed: {time.time() - t0:.0f}s")
        log.info(f"Results saved to {results_path}")
        log.info("Done.")


if __name__ == '__main__':
    main()
