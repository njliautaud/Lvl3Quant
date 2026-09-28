#!/usr/bin/env python3
"""
winner_confirmation_v1.py — Winner Confirmation via Mid-Trade Classifier

HYPOTHESIS: Instead of cutting predicted losers (which kills P&L due to 25:4 TP/SL
asymmetry requiring 88% precision), use the classifier to EXTEND winners.

KEY INSIGHT from midtrade_thesis_v1:
  - When P(win) > 0.55 at T+30s, WR jumps from 49% to 71%
  - Per-trade PnL nearly doubles (16.1 vs 9.7 ticks/trade)
  - Cutting losers is limited: 25:4 asymmetry = need 88% precision to break even on cuts

APPROACH — 4-tier hybrid management:
  1. Low confidence (<0.35): Cut immediately — save 2-3 ticks on clear losers
  2. Medium confidence (0.35-0.55): Keep original SL/TP — no change
  3. High confidence (>0.55): Move stop to breakeven (entry price)
  4. Very high confidence (>0.65): Trailing stop + extended hold

MEMORY-EFFICIENT: Pre-extract tick-by-tick price paths for all 223 trades ONCE,
then run all 960 configs against in-memory paths (no repeated npz loading).

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
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

warnings.filterwarnings('ignore', category=UserWarning, module='sklearn')

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path("/home/nick/Lvl3Quant")
SMART_EVENTS_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"
TICK_FEATURES_PATH = ROOT / "output" / "midtrade_thesis_v1" / "trade_tick_features.parquet"
CHAMPION_TRADES_PATH = ROOT / "output" / "multi_scale_combo_v1" / "champion_trade_details.csv"
OUT_DIR = ROOT / "output" / "winner_confirmation_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / f"winner_confirmation_v1_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("winner_confirm")

# ── Cost Constants (canonical) ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376   # AMP round-trip passive
MARKET_COST_TICKS = 1.376     # commission + spread crossing for market/stop exit
TICK_SIZE_PTS = 0.25

# Champion config
TP_LONG = 25
TP_SHORT = 25
SL_LONG = 4
SL_SHORT = 3

# Walk-forward params
WF_TRAIN_DAYS = 30

CHECKPOINTS_SEC = [5, 10, 15, 30]

# ── Grid configs ──────────────────────────────────────────────────────────
TIER_GRIDS = [
    # (cut, neutral_floor, be_stop, trail) — 4 tiers
    {'cut': 0.30, 'be_stop': 0.55, 'trail': 0.65},
    {'cut': 0.30, 'be_stop': 0.55, 'trail': 0.60},
    {'cut': 0.35, 'be_stop': 0.55, 'trail': 0.65},
    {'cut': 0.35, 'be_stop': 0.55, 'trail': 0.60},
    {'cut': 0.30, 'be_stop': 0.50, 'trail': 0.60},
    {'cut': 0.35, 'be_stop': 0.50, 'trail': 0.60},
    {'cut': 0.30, 'be_stop': 0.50, 'trail': 0.55},
    {'cut': 0.35, 'be_stop': 0.50, 'trail': 0.55},
    # No-cut: only extend winners
    {'cut': 0.00, 'be_stop': 0.55, 'trail': 0.65},
    {'cut': 0.00, 'be_stop': 0.55, 'trail': 0.60},
    {'cut': 0.00, 'be_stop': 0.50, 'trail': 0.60},
    {'cut': 0.00, 'be_stop': 0.50, 'trail': 0.55},
    # BE-only (no cut, no trail)
    {'cut': 0.00, 'be_stop': 0.55, 'trail': 1.00},
    {'cut': 0.00, 'be_stop': 0.50, 'trail': 1.00},
    # Trail-only (no cut, trail = be)
    {'cut': 0.00, 'be_stop': 0.55, 'trail': 0.55},
    {'cut': 0.00, 'be_stop': 0.60, 'trail': 0.60},
]

TRAIL_CONFIGS = [
    {'activate': 5, 'trail': 2},
    {'activate': 5, 'trail': 3},
    {'activate': 8, 'trail': 3},
    {'activate': 8, 'trail': 5},
    {'activate': 10, 'trail': 5},
]

MAX_HOLD_SECS = [300, 600, 900]  # 5min, 10min, 15min


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: PRE-EXTRACT PRICE PATHS (one-time, memory-efficient)
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_fill_timestamp_ns(date_str: str, time_of_day_min: int) -> int:
    year, month, day = int(date_str[:4]), int(date_str[4:6]), int(date_str[6:8])
    total_min_utc = 540 + time_of_day_min
    dt = pd.Timestamp(year=year, month=month, day=day,
                      hour=total_min_utc // 60, minute=total_min_utc % 60,
                      second=0, tz='UTC')
    return int(dt.value)


def extract_price_paths(df: pd.DataFrame, max_duration_sec: float = 900.0) -> List[Optional[np.ndarray]]:
    """
    Extract tick-by-tick PnL paths for all trades. Load one date at a time.

    Returns list of arrays, one per trade. Each array has shape (N, 2):
      col 0 = seconds since entry
      col 1 = PnL in ticks (direction-adjusted)

    Returns None for trades where data is unavailable.
    """
    log.info(f"Extracting price paths for {len(df)} trades (max {max_duration_sec}s each)")

    paths = [None] * len(df)
    dates = sorted(df['date'].unique())

    for di, date in enumerate(dates):
        events_file = SMART_EVENTS_DIR / f"{date}_mbo_events.npz"
        if not events_file.exists():
            log.warning(f"  No events for {date}")
            continue

        d = np.load(str(events_file), mmap_mode='r')
        events = d['events']
        timestamps = d['timestamps']

        date_mask = df['date'] == date
        date_indices = df.index[date_mask].tolist()

        for idx in date_indices:
            trade = df.loc[idx]
            direction = int(trade['direction'])
            tod = int(trade['time_of_day_min'])
            entry_ts_ns = reconstruct_fill_timestamp_ns(date, tod)
            max_ns = int(max_duration_sec * 1e9)

            i_start = np.searchsorted(timestamps, entry_ts_ns, side='left')
            i_end = np.searchsorted(timestamps, entry_ts_ns + max_ns, side='right')

            if i_end <= i_start + 5:
                continue

            # Extract price path (col 3 = price_rel_ticks)
            price_ticks = events[i_start:i_end, 3].copy()  # .copy() to release mmap
            ts_sec = (timestamps[i_start:i_end].copy() - entry_ts_ns) / 1e9

            # PnL = (price - entry_price) * direction
            pnl = (price_ticks - price_ticks[0]) * direction

            path = np.column_stack([ts_sec, pnl]).astype(np.float32)
            paths[idx] = path

        # Explicitly release mmap
        del d, events, timestamps
        gc.collect()

        if (di + 1) % 10 == 0:
            log.info(f"  Processed {di+1}/{len(dates)} dates")

    n_valid = sum(1 for p in paths if p is not None)
    log.info(f"  Extracted {n_valid}/{len(df)} price paths")
    return paths


# ═══════════════════════════════════════════════════════════════════════════
# WALK-FORWARD CLASSIFIER
# ═══════════════════════════════════════════════════════════════════════════

def get_feature_cols(df: pd.DataFrame, checkpoint_sec: int) -> List[str]:
    prefix = f'cp{checkpoint_sec}s_'
    return [c for c in df.columns if c.startswith(prefix)]


def walk_forward_classify(df: pd.DataFrame, checkpoint_sec: int) -> Tuple[np.ndarray, dict]:
    import lightgbm as lgb

    feat_cols = get_feature_cols(df, checkpoint_sec)
    log.info(f"WF classify at {checkpoint_sec}s: {len(feat_cols)} features")

    dates = sorted(df['date'].unique())
    predictions = np.full(len(df), np.nan)

    for i, test_date in enumerate(dates):
        train_dates = dates[max(0, i - WF_TRAIN_DAYS):i]
        if len(train_dates) < 10:
            continue

        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'] == test_date

        X_train = np.nan_to_num(df.loc[train_mask, feat_cols].values, nan=0.0)
        y_train = df.loc[train_mask, 'winner'].values
        X_test = np.nan_to_num(df.loc[test_mask, feat_cols].values, nan=0.0)

        if len(X_test) == 0 or len(X_train) < 20:
            continue

        n_pos = max(y_train.sum(), 1)
        scale_pos = (len(y_train) - n_pos) / n_pos

        model = lgb.LGBMClassifier(
            objective='binary', verbosity=-1,
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=16, min_child_samples=5,
            reg_alpha=0.1, reg_lambda=1.0,
            subsample=0.8, colsample_bytree=0.8,
            scale_pos_weight=scale_pos, random_state=42,
        )
        model.fit(X_train, y_train)
        probs = model.predict_proba(X_test)[:, 1]
        predictions[df.index[test_mask]] = probs

    valid_mask = ~np.isnan(predictions)
    if valid_mask.sum() > 10:
        vp = predictions[valid_mask]
        vt = df.loc[valid_mask, 'winner'].values
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(vt, vp))
        except ValueError:
            auc = np.nan
        ic, _ = spearmanr(vp, vt)
        acc = float(np.mean((vp > 0.5).astype(int) == vt))
        metrics = {'checkpoint_sec': checkpoint_sec, 'n_valid': int(valid_mask.sum()),
                   'accuracy': acc, 'auc': auc, 'ic': float(ic),
                   'mean_prob_winners': float(vp[vt == 1].mean()),
                   'mean_prob_losers': float(vp[vt == 0].mean())}
    else:
        metrics = {'checkpoint_sec': checkpoint_sec, 'n_valid': 0,
                   'accuracy': np.nan, 'auc': np.nan, 'ic': np.nan}

    return predictions, metrics


# ═══════════════════════════════════════════════════════════════════════════
# TRADE REPLAY ON PRE-EXTRACTED PATHS
# ═══════════════════════════════════════════════════════════════════════════

def replay_on_path(path: Optional[np.ndarray], direction: int,
                   management: str, checkpoint_sec: float = 15.0,
                   trail_config: dict = None,
                   max_hold_sec: float = 900.0,
                   original_exit_ticks: float = 0.0) -> Dict:
    """
    Replay a single trade on its pre-extracted price path.
    path: (N, 2) array — col0=seconds, col1=pnl_ticks (direction-adjusted)

    CRITICAL: Before checkpoint_sec, original SL/TP applies.
    If original SL/TP fires before checkpoint, management has no effect.
    After checkpoint, modified stops kick in.
    """
    if management == 'original':
        return {'exit_ticks': original_exit_ticks, 'exit_type': 'original', 'mfe': np.nan}

    if path is None or len(path) < 5:
        return {'exit_ticks': original_exit_ticks, 'exit_type': 'no_data', 'mfe': np.nan}

    SL = SL_LONG if direction == 1 else SL_SHORT
    TP = TP_LONG if direction == 1 else TP_SHORT

    ts = path[:, 0]
    pnl = path[:, 1]

    # Cap at max_hold_sec
    hold_mask = ts <= max_hold_sec
    if hold_mask.sum() < 5:
        return {'exit_ticks': original_exit_ticks, 'exit_type': 'no_data', 'mfe': np.nan}

    ts = ts[hold_mask]
    pnl = pnl[hold_mask]

    mfe = 0.0
    trail_active = False
    hard_tp = TP * 2  # 50 ticks cap for trailing mode
    exit_ticks = None
    exit_type = 'max_hold'
    stop_level = -SL  # default to original SL, updated at checkpoint

    # Phase 1: Before checkpoint — original SL/TP applies
    # Phase 2: After checkpoint — modified management applies
    checkpoint_reached = False

    for i in range(len(pnl)):
        p = float(pnl[i])
        t = float(ts[i])
        mfe = max(mfe, p)

        if not checkpoint_reached:
            # BEFORE CHECKPOINT: original SL/TP
            if p >= TP:
                exit_type = 'tp_before_cp'
                exit_ticks = TP - RT_COMMISSION_TICKS
                break
            if p <= -SL:
                exit_type = 'sl_before_cp'
                exit_ticks = -SL - MARKET_COST_TICKS
                break
            if t >= checkpoint_sec:
                checkpoint_reached = True
                # For cut: exit immediately at market at checkpoint
                if management == 'cut':
                    exit_type = 'cut'
                    exit_ticks = p - MARKET_COST_TICKS  # exit at current price level
                    break
                # Initialize modified stop at this point
                if management == 'breakeven':
                    stop_level = 0.0  # breakeven
                elif management == 'trailing':
                    stop_level = 0.0  # start at breakeven, trail activates later
                continue

        # AFTER CHECKPOINT: modified management
        if management == 'breakeven':
            if p >= TP:
                exit_type = 'tp'
                exit_ticks = TP - RT_COMMISSION_TICKS
                break
            if p <= stop_level:
                exit_type = 'be_stop'
                exit_ticks = max(stop_level, p) - MARKET_COST_TICKS
                break

        elif management == 'trailing':
            if trail_config and not trail_active and p >= trail_config['activate']:
                trail_active = True
                stop_level = p - trail_config['trail']

            if trail_active:
                new_stop = p - trail_config['trail']
                stop_level = max(stop_level, new_stop)

            if p >= hard_tp:
                exit_type = 'hard_tp'
                exit_ticks = hard_tp - RT_COMMISSION_TICKS
                break

            if p <= stop_level:
                exit_type = 'trail_stop' if trail_active else 'be_stop'
                exit_ticks = max(stop_level, p) - MARKET_COST_TICKS
                break

    if exit_ticks is None:
        exit_ticks = float(pnl[-1]) - MARKET_COST_TICKS
        exit_type = 'max_hold'

    return {'exit_ticks': float(exit_ticks), 'exit_type': exit_type, 'mfe': float(mfe)}


# ═══════════════════════════════════════════════════════════════════════════
# SIMULATION ENGINE
# ═══════════════════════════════════════════════════════════════════════════

def simulate_config(df: pd.DataFrame, predictions: np.ndarray,
                    paths: List[Optional[np.ndarray]],
                    checkpoint_sec: float,
                    tier_config: dict, trail_config: dict,
                    max_hold_sec: float) -> Dict:
    """Run one config across all trades using pre-extracted paths."""
    valid_mask = ~np.isnan(predictions)
    if valid_mask.sum() < 10:
        return {'n_valid': 0, 'sharpe': np.nan}

    valid_indices = np.where(valid_mask)[0]
    valid_preds = predictions[valid_mask]

    cut_t = tier_config['cut']
    be_t = tier_config['be_stop']
    trail_t = tier_config['trail']

    managed_pnl = np.zeros(len(valid_indices))
    mgmt_types = []
    exit_types = []

    for j, idx in enumerate(valid_indices):
        prob = valid_preds[j]
        trade = df.iloc[idx]
        direction = int(trade['direction'])
        original_exit = trade['exit_ticks']
        path = paths[idx]

        # Tier assignment
        if cut_t > 0 and prob < cut_t:
            mgmt = 'cut'
        elif prob < be_t:
            mgmt = 'original'
        elif prob < trail_t:
            mgmt = 'breakeven'
        else:
            mgmt = 'trailing'

        result = replay_on_path(path, direction, mgmt, checkpoint_sec,
                                trail_config, max_hold_sec, original_exit)
        managed_pnl[j] = result['exit_ticks']
        mgmt_types.append(mgmt)
        exit_types.append(result['exit_type'])

    # Build results DataFrame for metrics
    valid_df = df.iloc[valid_indices].copy()
    valid_df['managed_pnl'] = managed_pnl
    valid_df['management'] = mgmt_types
    valid_df['sim_exit_type'] = exit_types

    # Metrics
    day_pnl = valid_df.groupby('date')['managed_pnl'].sum()
    n_days = len(day_pnl)
    sharpe = float(day_pnl.mean() / day_pnl.std() * np.sqrt(252)) if day_pnl.std() > 0 else np.nan

    downside = day_pnl[day_pnl < 0]
    ds_std = downside.std() if len(downside) > 1 else 0.001
    sortino = float(day_pnl.mean() / ds_std * np.sqrt(252)) if ds_std > 0 else np.nan

    total_pnl = float(managed_pnl.sum())
    wr = float(np.mean(managed_pnl > 0))
    gw = float(np.sum(managed_pnl[managed_pnl > 0]))
    gl = float(np.abs(np.sum(managed_pnl[managed_pnl < 0])))
    pf = gw / max(gl, 0.01)

    cumsum = day_pnl.cumsum()
    max_dd = float((cumsum - cumsum.cummax()).min())

    day_conc = float(day_pnl.abs().max() / day_pnl.abs().sum()) if day_pnl.abs().sum() > 0 else np.nan

    baseline_pnl = float(valid_df['exit_ticks'].sum())
    improvement = total_pnl - baseline_pnl

    # Per-tier stats
    tier_stats = {}
    for mt in ['cut', 'original', 'breakeven', 'trailing']:
        mask = valid_df['management'] == mt
        n = int(mask.sum())
        if n > 0:
            tp = managed_pnl[mask.values]
            tier_stats[mt] = {
                'n': n, 'pnl': float(tp.sum()), 'avg': float(tp.mean()),
                'wr': float(np.mean(tp > 0)),
                'orig_W': int((valid_df.loc[mask, 'winner'] == 1).sum()),
                'orig_L': int((valid_df.loc[mask, 'winner'] == 0).sum()),
            }

    return {
        'n_valid': len(valid_df), 'n_days': n_days,
        'sharpe': sharpe, 'sortino': sortino, 'pf': pf, 'wr': wr,
        'total_pnl_ticks': total_pnl, 'total_pnl_dollars': total_pnl * ES_TICK_VALUE,
        'avg_pnl_ticks': total_pnl / max(len(valid_df), 1),
        'max_dd_ticks': max_dd, 'day_concentration': day_conc,
        'baseline_pnl_ticks': baseline_pnl,
        'improvement_ticks': improvement,
        'improvement_pct': improvement / abs(baseline_pnl) * 100 if baseline_pnl != 0 else 0,
        'tier_stats': tier_stats,
    }, valid_df


def compute_regime_analysis(df: pd.DataFrame) -> Dict:
    """Compute per-regime metrics using ES close-to-close."""
    regime_map = {}
    for date_str in sorted(df['date'].unique()):
        bar_file = MINUTE_BAR_DIR / f"{date_str}.parquet"
        if not bar_file.exists():
            regime_map[date_str] = 'unknown'
            continue
        bars = pd.read_parquet(bar_file)
        change_ticks = (bars['close'].iloc[-1] - bars['open'].iloc[0]) / (TICK_SIZE_PTS * 100)
        regime_map[date_str] = 'green' if change_ticks > 2 else ('red' if change_ticks < -2 else 'flat')

    df = df.copy()
    df['regime'] = df['date'].map(regime_map)

    results = {}
    for regime in ['green', 'red', 'flat']:
        sub = df[df['regime'] == regime]
        if len(sub) < 3:
            results[regime] = {'n_trades': len(sub), 'sharpe': np.nan}
            continue
        dp = sub.groupby('date')['managed_pnl'].sum()
        sharpe = float(dp.mean() / dp.std() * np.sqrt(252)) if dp.std() > 0 else np.nan
        results[regime] = {
            'n_trades': len(sub), 'n_days': len(dp), 'sharpe': sharpe,
            'total_pnl': float(sub['managed_pnl'].sum()),
            'wr': float((sub['managed_pnl'] > 0).mean()),
        }

    gs = results.get('green', {}).get('sharpe', np.nan)
    rs = results.get('red', {}).get('sharpe', np.nan)
    if not (np.isnan(gs) or np.isnan(rs)):
        gap = abs(gs - rs) / max(abs(gs), abs(rs), 0.01)
        results['regime_gap'] = gap
        results['regime_pass'] = gap < 0.50
    else:
        results['regime_gap'] = np.nan
        results['regime_pass'] = False
    return results


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import mlflow

    t0 = time.time()
    log.info("=" * 80)
    log.info("Winner Confirmation v1 — Starting")
    log.info("  Hypothesis: Extend winners via BE stop + trailing, not cut losers")
    log.info("=" * 80)

    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("winner_confirmation_v1")

    with mlflow.start_run(run_name=f"winner_confirm_v1_{time.strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params({
            'checkpoints_sec': str(CHECKPOINTS_SEC),
            'wf_train_days': WF_TRAIN_DAYS,
            'n_tier_configs': len(TIER_GRIDS),
            'n_trail_configs': len(TRAIL_CONFIGS),
            'n_hold_configs': len(MAX_HOLD_SECS),
            'commission_ticks': RT_COMMISSION_TICKS,
            'market_cost_ticks': MARKET_COST_TICKS,
        })

        # ── Phase 1: Load data ──────────────────────────────────────────────
        log.info("Phase 1: Loading tick features")
        df = pd.read_parquet(TICK_FEATURES_PATH)
        log.info(f"  {len(df)} trades, {len(df.columns)} columns")

        baseline_pnl = df['exit_ticks'].sum()
        baseline_wr = (df['exit_ticks'] > 0).mean()
        bl_day = df.groupby('date')['exit_ticks'].sum()
        baseline_sharpe = float(bl_day.mean() / bl_day.std() * np.sqrt(252)) if bl_day.std() > 0 else 0
        log.info(f"  Baseline: {baseline_pnl:.1f}t, WR={baseline_wr:.1%}, Sharpe={baseline_sharpe:.2f}")

        mlflow.log_metrics({
            'n_trades': len(df), 'baseline_pnl': baseline_pnl,
            'baseline_wr': baseline_wr, 'baseline_sharpe': baseline_sharpe,
        })

        # ── Phase 2: Extract price paths (one-time) ─────────────────────────
        log.info("Phase 2: Extracting tick-by-tick price paths")
        paths = extract_price_paths(df, max_duration_sec=900.0)

        n_paths = sum(1 for p in paths if p is not None)
        mlflow.log_metric('n_price_paths', n_paths)
        log.info(f"  {n_paths}/{len(df)} trades with valid price paths")

        # ── Phase 3: Walk-forward classification ─────────────────────────────
        log.info("Phase 3: Walk-forward LightGBM classification")
        all_predictions = {}
        all_metrics = {}

        for cp in CHECKPOINTS_SEC:
            preds, metrics = walk_forward_classify(df, cp)
            all_predictions[cp] = preds
            all_metrics[cp] = metrics
            log.info(f"  CP={cp}s: AUC={metrics.get('auc', np.nan):.3f}, "
                     f"Acc={metrics.get('accuracy', np.nan):.3f}, N={metrics.get('n_valid', 0)}")
            mlflow.log_metrics({
                f'auc_{cp}s': metrics.get('auc', 0),
                f'accuracy_{cp}s': metrics.get('accuracy', 0),
            })

        # ── Phase 4: Grid search (fast — all in-memory) ─────────────────────
        n_combos = len(CHECKPOINTS_SEC) * len(TIER_GRIDS) * len(TRAIL_CONFIGS) * len(MAX_HOLD_SECS)
        log.info(f"Phase 4: Grid search — {n_combos} combos (all in-memory)")

        all_results = []
        best_result = None
        best_sharpe = -np.inf
        best_managed_df = None
        n_tested = 0

        for cp in CHECKPOINTS_SEC:
            preds = all_predictions[cp]
            for tier in TIER_GRIDS:
                for trail in TRAIL_CONFIGS:
                    for max_hold in MAX_HOLD_SECS:
                        n_tested += 1
                        result, managed_df = simulate_config(
                            df, preds, paths, cp, tier, trail, max_hold
                        )

                        if result.get('n_valid', 0) == 0:
                            continue

                        regime = compute_regime_analysis(managed_df)
                        result['regime'] = regime
                        result['checkpoint_sec'] = cp
                        result['tier_config'] = tier
                        result['trail_config'] = trail
                        result['max_hold_sec'] = max_hold

                        all_results.append(result)

                        sharpe = result.get('sharpe', 0)
                        improvement = result.get('improvement_ticks', 0)
                        r_pass = regime.get('regime_pass', False)
                        r_gap = regime.get('regime_gap', 1.0)

                        if n_tested % 100 == 0:
                            log.info(f"  [{n_tested}/{n_combos}] best_so_far={best_sharpe:.2f}")

                        if r_pass and sharpe > best_sharpe:
                            best_sharpe = sharpe
                            best_result = result
                            best_managed_df = managed_df.copy()

                            log.info(
                                f"  NEW BEST [{n_tested}]: CP={cp}s, "
                                f"cut={tier['cut']:.2f}/be={tier['be_stop']:.2f}/trail={tier['trail']:.2f}, "
                                f"trail_act={trail['activate']}/dist={trail['trail']}, "
                                f"hold={max_hold}s → Sharpe={sharpe:.2f}, "
                                f"PnL={result['total_pnl_ticks']:.1f}t, "
                                f"Improve={improvement:.1f}t, Gap={r_gap:.3f}"
                            )

        log.info(f"\nGrid search complete: {n_tested} configs tested")

        # ── Phase 5: Results ─────────────────────────────────────────────────
        log.info("=" * 80)
        log.info("RESULTS")
        log.info("=" * 80)

        passing = [r for r in all_results if r.get('regime', {}).get('regime_pass', False)]
        passing.sort(key=lambda r: r.get('sharpe', 0), reverse=True)

        log.info(f"Baseline: {baseline_pnl:.1f}t, Sharpe={baseline_sharpe:.2f}, WR={baseline_wr:.1%}")
        log.info(f"{len(passing)} regime-passing out of {len(all_results)} total")

        log.info("\nTop 10 regime-passing configs:")
        for i, r in enumerate(passing[:10]):
            tc, trc = r['tier_config'], r['trail_config']
            log.info(
                f"  #{i+1}: CP={r['checkpoint_sec']}s, "
                f"cut={tc['cut']:.2f}/be={tc['be_stop']:.2f}/trail={tc['trail']:.2f}, "
                f"act={trc['activate']}/dist={trc['trail']}, hold={r['max_hold_sec']}s | "
                f"Sharpe={r['sharpe']:.2f}, Sortino={r['sortino']:.2f}, "
                f"PF={r['pf']:.2f}, WR={r['wr']:.1%}, "
                f"PnL={r['total_pnl_ticks']:.1f}t (${r['total_pnl_dollars']:.0f}), "
                f"Improve={r['improvement_ticks']:.1f}t ({r['improvement_pct']:.1f}%), "
                f"DD={r['max_dd_ticks']:.1f}t, Gap={r['regime'].get('regime_gap', 'N/A')}"
            )

        if best_result:
            log.info("\n" + "=" * 80)
            log.info("BEST CONFIG")
            log.info("=" * 80)
            tc, trc = best_result['tier_config'], best_result['trail_config']
            log.info(f"  Checkpoint: {best_result['checkpoint_sec']}s")
            log.info(f"  Cut < {tc['cut']:.2f} | Original < {tc['be_stop']:.2f} | "
                     f"BE < {tc['trail']:.2f} | Trail >= {tc['trail']:.2f}")
            log.info(f"  Trail: activate={trc['activate']}t, distance={trc['trail']}t")
            log.info(f"  Max hold: {best_result['max_hold_sec']}s")
            log.info(f"  Sharpe: {best_result['sharpe']:.2f} (baseline {baseline_sharpe:.2f})")
            log.info(f"  Sortino: {best_result['sortino']:.2f}")
            log.info(f"  PF: {best_result['pf']:.2f}")
            log.info(f"  WR: {best_result['wr']:.1%}")
            log.info(f"  Total P&L: {best_result['total_pnl_ticks']:.1f}t (${best_result['total_pnl_dollars']:.0f})")
            log.info(f"  Avg/trade: {best_result['avg_pnl_ticks']:.2f}t")
            log.info(f"  Max DD: {best_result['max_dd_ticks']:.1f}t")
            log.info(f"  Improvement: {best_result['improvement_ticks']:.1f}t ({best_result['improvement_pct']:.1f}%)")
            log.info(f"  Regime: {best_result['regime']}")

            log.info("\n  Per-tier breakdown:")
            for tier_name, stats in best_result.get('tier_stats', {}).items():
                log.info(f"    {tier_name}: n={stats['n']}, PnL={stats['pnl']:.1f}t, "
                         f"avg={stats['avg']:.2f}t, WR={stats['wr']:.1%}, "
                         f"orig_W={stats['orig_W']}/L={stats['orig_L']}")

            mlflow.log_metrics({
                'best_sharpe': best_result['sharpe'],
                'best_sortino': best_result['sortino'],
                'best_pf': best_result['pf'],
                'best_wr': best_result['wr'],
                'best_pnl_ticks': best_result['total_pnl_ticks'],
                'best_improvement_ticks': best_result['improvement_ticks'],
                'best_improvement_pct': best_result['improvement_pct'],
                'best_checkpoint_sec': best_result['checkpoint_sec'],
                'best_max_dd_ticks': best_result['max_dd_ticks'],
                'best_regime_gap': best_result['regime'].get('regime_gap', 1.0),
            })

            trades_path = OUT_DIR / "best_managed_trades.parquet"
            best_managed_df.to_parquet(trades_path, index=False)
            mlflow.log_artifact(str(trades_path))
        else:
            log.info("NO regime-passing config found.")
            mlflow.log_metric('best_improvement_ticks', 0)

        # ── Analysis breakdown ─────────────────────────────────────────────
        log.info("\n" + "=" * 80)
        log.info("ANALYSIS: Strategy type breakdown")
        log.info("=" * 80)

        for label, filt_fn in [
            ('BE-only (no cut/trail)', lambda r: r['tier_config']['cut'] == 0 and r['tier_config']['trail'] >= 1.0),
            ('Trail-only (no cut)', lambda r: r['tier_config']['cut'] == 0 and r['tier_config']['be_stop'] == r['tier_config']['trail']),
            ('Hybrid+cut', lambda r: r['tier_config']['cut'] > 0),
            ('Hybrid no-cut', lambda r: r['tier_config']['cut'] == 0 and r['tier_config']['trail'] < 1.0 and r['tier_config']['be_stop'] != r['tier_config']['trail']),
        ]:
            group = [r for r in passing if filt_fn(r)]
            if group:
                best_g = max(group, key=lambda r: r['sharpe'])
                log.info(f"  {label}: {len(group)} passing, best Sharpe={best_g['sharpe']:.2f}, "
                         f"PnL={best_g['total_pnl_ticks']:.1f}t, "
                         f"Improve={best_g['improvement_ticks']:.1f}t")
            else:
                log.info(f"  {label}: 0 passing")

        # Save
        output = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'elapsed_seconds': time.time() - t0,
            'hypothesis': 'Extend winners via BE stop + trailing instead of cutting losers',
            'n_trades': len(df), 'n_tested': n_tested, 'n_passing': len(passing),
            'baseline': {'pnl_ticks': float(baseline_pnl), 'sharpe': baseline_sharpe, 'wr': float(baseline_wr)},
            'classifier': {str(k): v for k, v in all_metrics.items()},
            'top_10': [r for r in passing[:10]],
            'best_result': best_result,
            'all_summary': [{
                'cp': r['checkpoint_sec'], 'tier': r['tier_config'],
                'trail': r['trail_config'], 'hold': r['max_hold_sec'],
                'sharpe': r['sharpe'], 'pf': r['pf'], 'wr': r['wr'],
                'pnl': r['total_pnl_ticks'], 'improve': r['improvement_ticks'],
                'gap': r.get('regime', {}).get('regime_gap', np.nan),
                'pass': r.get('regime', {}).get('regime_pass', False),
            } for r in all_results],
        }

        results_path = OUT_DIR / "results.json"
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        mlflow.log_artifact(str(results_path))

        log.info(f"\nDone in {time.time() - t0:.0f}s. Results at {results_path}")


if __name__ == '__main__':
    main()
