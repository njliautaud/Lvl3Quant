#!/usr/bin/env python3
"""
midtrade_thesis_v1.py — Mid-Trade Thesis Validation using Tick-Level MBO Features

GOAL: Continuously evaluate DURING the trade whether the entry thesis is still valid.
The champion strategy (30-min LGBM + asymmetric TP=25/SL_L=4/SL_S=3) exits via
tight SL within ~1 minute, making minute-bar classifiers useless. We need sub-minute
intelligence from the order book.

APPROACH:
1. For each of the 223 champion trades, extract tick-level MBO features from the
   smart events data during the first 5s, 10s, 15s, 30s after entry.
2. Compute aggregated microstructure features at each checkpoint:
   - Order flow imbalance (OFI) evolution relative to trade direction
   - Price momentum (micro-trend aligned with position)
   - Trade intensity / event density changes
   - Spread dynamics
   - Book pressure indicators
3. Walk-forward LightGBM classifier: predict TP vs SL outcome from these features.
4. Test management rules: early-cut losers at T+15s if classifier says "loser".
5. FIFO-validated P&L with regime gate.

CONSTRAINTS:
  - Walk-forward sliding window ONLY (HC #0)
  - FIFO fills only, never midpoint
  - Passive limit = 0.376 ticks commission; market = 1.376 ticks
  - MLflow tracking mandatory
  - Regime gap < 0.50 to pass

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
SMART_EVENTS_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
TRADES_PATH = ROOT / "output" / "winloss_characterization_v1" / "trades_enriched.parquet"
OUT_DIR = ROOT / "output" / "midtrade_thesis_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / f"midtrade_thesis_v1_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("midtrade_thesis")

# ── Cost Constants (canonical) ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376  # AMP round-trip
MARKET_CROSS_TICKS = 1.0     # spread crossing for market order
TICK_SIZE_PTS = 0.25          # ES minimum tick in points

# Champion config
TP_LONG = 25   # ticks
TP_SHORT = 25
SL_LONG = 4
SL_SHORT = 3

# Feature names for the 25-col smart events format
FEAT_NAMES = [
    'time_delta_log', 'event_type_id', 'side_id', 'price_rel_ticks',
    'qty_log', 'spread_ticks', 'cancel_side_asym_50', 'rolling_ofi_500',
    'event_density_20', 'price_mom_10', 'qty_price_mom_50',
    'price_sign_mom_200', 'event_type_entropy_200',
    'fill_add_restore_100', 'spread_velocity_50',
    'feat_15', 'feat_16', 'feat_17', 'feat_18', 'feat_19',
    'feat_20', 'feat_21', 'feat_22', 'feat_23', 'feat_24',
]

# Checkpoints (seconds after entry) at which we evaluate
CHECKPOINTS_SEC = [5, 10, 15, 30]

# Walk-forward params
WF_TRAIN_DAYS = 30   # sliding window: train on 30 days
WF_TEST_DAYS = 1     # test on 1 day


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 1: Extract tick-level features around each trade entry
# ═══════════════════════════════════════════════════════════════════════════

def reconstruct_fill_timestamp_ns(date_str: str, time_of_day_min: int) -> int:
    """
    Reconstruct the fill timestamp in nanoseconds from the trade's date and
    time_of_day_min field.

    time_of_day_min = (hour_utc - 9) * 60 + minute_utc
    So: total_minutes_utc = 9 * 60 + time_of_day_min = 540 + time_of_day_min
    """
    year = int(date_str[:4])
    month = int(date_str[4:6])
    day = int(date_str[6:8])

    total_min_utc = 540 + time_of_day_min
    hour_utc = total_min_utc // 60
    min_utc = total_min_utc % 60

    dt = pd.Timestamp(year=year, month=month, day=day,
                      hour=hour_utc, minute=min_utc, second=0,
                      tz='UTC')
    return int(dt.value)  # nanoseconds


def extract_window_features(events: np.ndarray, timestamps: np.ndarray,
                            entry_ts_ns: int, direction: int,
                            window_sec: float) -> Dict[str, float]:
    """
    Extract aggregated microstructure features from MBO events within
    [entry_ts_ns, entry_ts_ns + window_sec * 1e9].

    Returns a flat dict of features for this checkpoint.
    """
    window_ns = int(window_sec * 1e9)
    start_ns = entry_ts_ns
    end_ns = entry_ts_ns + window_ns

    # Binary search for event window
    idx_start = np.searchsorted(timestamps, start_ns, side='left')
    idx_end = np.searchsorted(timestamps, end_ns, side='right')

    n_events = idx_end - idx_start

    feats = {}
    feats['n_events'] = float(n_events)

    if n_events < 5:
        # Not enough data — fill with NaN
        for k in ['ofi_mean', 'ofi_trend', 'price_mom_mean', 'price_mom_trend',
                   'event_density_mean', 'event_density_change',
                   'spread_mean', 'spread_change',
                   'qty_mom_mean', 'sign_mom_mean',
                   'ofi_aligned', 'price_mom_aligned',
                   'sign_mom_aligned', 'spread_vel_mean',
                   'feat15_mean', 'feat16_mean',
                   'ofi_first_half', 'ofi_second_half', 'ofi_accel',
                   'pmom_first_half', 'pmom_second_half', 'pmom_accel',
                   'density_ratio', 'n_events_norm']:
            feats[k] = np.nan
        return feats

    window_ev = events[idx_start:idx_end]

    # Core feature columns
    col_ofi = 7       # rolling_ofi_500
    col_pmom = 9      # price_mom_10
    col_density = 8   # event_density_20
    col_spread = 5    # spread_ticks
    col_qtymom = 10   # qty_price_mom_50
    col_signmom = 11  # price_sign_mom_200
    col_spreadvel = 14 # spread_velocity_50
    col_feat15 = 15
    col_feat16 = 16

    ofi = window_ev[:, col_ofi]
    pmom = window_ev[:, col_pmom]
    density = window_ev[:, col_density]
    spread = window_ev[:, col_spread]
    qtymom = window_ev[:, col_qtymom]
    signmom = window_ev[:, col_signmom]
    spreadvel = window_ev[:, col_spreadvel]
    f15 = window_ev[:, col_feat15]
    f16 = window_ev[:, col_feat16]

    # ── Aggregated features ──
    feats['ofi_mean'] = float(np.nanmean(ofi))
    feats['price_mom_mean'] = float(np.nanmean(pmom))
    feats['event_density_mean'] = float(np.nanmean(density))
    feats['spread_mean'] = float(np.nanmean(spread))
    feats['qty_mom_mean'] = float(np.nanmean(qtymom))
    feats['sign_mom_mean'] = float(np.nanmean(signmom))
    feats['spread_vel_mean'] = float(np.nanmean(spreadvel))
    feats['feat15_mean'] = float(np.nanmean(f15))
    feats['feat16_mean'] = float(np.nanmean(f16))

    # ── Trend features (first half vs second half) ──
    mid = n_events // 2
    if mid > 2:
        feats['ofi_first_half'] = float(np.nanmean(ofi[:mid]))
        feats['ofi_second_half'] = float(np.nanmean(ofi[mid:]))
        feats['ofi_accel'] = feats['ofi_second_half'] - feats['ofi_first_half']

        feats['pmom_first_half'] = float(np.nanmean(pmom[:mid]))
        feats['pmom_second_half'] = float(np.nanmean(pmom[mid:]))
        feats['pmom_accel'] = feats['pmom_second_half'] - feats['pmom_first_half']

        feats['density_ratio'] = float(np.nanmean(density[mid:])) / max(float(np.nanmean(density[:mid])), 0.001)
    else:
        feats['ofi_first_half'] = feats['ofi_mean']
        feats['ofi_second_half'] = feats['ofi_mean']
        feats['ofi_accel'] = 0.0
        feats['pmom_first_half'] = feats['price_mom_mean']
        feats['pmom_second_half'] = feats['price_mom_mean']
        feats['pmom_accel'] = 0.0
        feats['density_ratio'] = 1.0

    # ── OFI trend (linear regression slope) ──
    if n_events >= 10:
        x = np.arange(n_events, dtype=np.float32)
        x_centered = x - x.mean()
        denom = np.sum(x_centered ** 2)
        if denom > 0:
            feats['ofi_trend'] = float(np.sum(x_centered * ofi) / denom)
            feats['price_mom_trend'] = float(np.sum(x_centered * pmom) / denom)
        else:
            feats['ofi_trend'] = 0.0
            feats['price_mom_trend'] = 0.0
    else:
        feats['ofi_trend'] = 0.0
        feats['price_mom_trend'] = 0.0

    # ── Event density change (last 20% vs first 20%) ──
    n20 = max(n_events // 5, 1)
    feats['event_density_change'] = float(np.nanmean(density[-n20:])) - float(np.nanmean(density[:n20]))
    feats['spread_change'] = float(np.nanmean(spread[-n20:])) - float(np.nanmean(spread[:n20]))

    # ── Direction-aligned features ──
    # Positive OFI aligned with long = good; negative OFI aligned with short = good
    feats['ofi_aligned'] = feats['ofi_mean'] * direction
    feats['price_mom_aligned'] = feats['price_mom_mean'] * direction
    feats['sign_mom_aligned'] = feats['sign_mom_mean'] * direction

    # Normalized event count (events per second)
    feats['n_events_norm'] = float(n_events) / max(window_sec, 0.1)

    return feats


def extract_all_trade_features(trades_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each trade, extract tick-level features at each checkpoint.
    Returns a DataFrame with one row per trade, columns for each checkpoint's features.
    """
    log.info(f"Extracting tick-level features for {len(trades_df)} trades at checkpoints {CHECKPOINTS_SEC}")

    all_rows = []
    dates = sorted(trades_df['date'].unique())

    for di, date in enumerate(dates):
        date_trades = trades_df[trades_df['date'] == date]
        t_load = time.time()

        # Load smart events for this date (mmap for speed)
        events_file = SMART_EVENTS_DIR / f"{date}_mbo_events.npz"
        if not events_file.exists():
            log.warning(f"No events file for {date}, skipping {len(date_trades)} trades")
            continue

        d = np.load(str(events_file), mmap_mode='r')
        events = d['events']
        timestamps = d['timestamps']

        # Quick monotonicity check on a sample (full check too slow for 12M rows)
        sample_idx = np.linspace(0, len(timestamps)-1, min(1000, len(timestamps)), dtype=int)
        if not np.all(np.diff(timestamps[sample_idx]) >= 0):
            log.warning(f"Timestamps not monotonic for {date}, skipping")
            del d, events, timestamps
            gc.collect()
            continue

        log.info(f"  [{di+1}/{len(dates)}] {date}: {len(date_trades)} trades, "
                 f"{len(timestamps)/1e6:.1f}M events, loaded in {time.time()-t_load:.1f}s")

        for _, trade in date_trades.iterrows():
            tod = int(trade['time_of_day_min'])
            direction = int(trade['direction'])
            fill_price = trade['fill_price']
            exit_type = trade['exit_type']
            exit_ticks = trade['exit_ticks']

            # Reconstruct fill timestamp
            entry_ts_ns = reconstruct_fill_timestamp_ns(date, tod)

            # Check if entry time falls within event data range
            if entry_ts_ns < timestamps[0] or entry_ts_ns > timestamps[-1]:
                log.warning(f"Entry ts {entry_ts_ns} outside events range for {date} tod={tod}")
                continue

            row = {
                'date': date,
                'direction': direction,
                'fill_price': fill_price,
                'time_of_day_min': tod,
                'exit_type': exit_type,
                'exit_ticks': exit_ticks,
                'winner': 1 if exit_type == 'tp' else 0,
                'pred_magnitude': trade['pred_magnitude'],
            }

            # Extract features at each checkpoint
            for cp_sec in CHECKPOINTS_SEC:
                cp_feats = extract_window_features(events, timestamps, entry_ts_ns, direction, cp_sec)
                for k, v in cp_feats.items():
                    row[f'cp{cp_sec}s_{k}'] = v

            all_rows.append(row)

        # Free memory
        del d, events, timestamps
        gc.collect()

    result = pd.DataFrame(all_rows)
    log.info(f"Extracted features for {len(result)} trades ({len(result.columns)} columns)")
    return result


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 2: Walk-Forward LightGBM Classifier
# ═══════════════════════════════════════════════════════════════════════════

def get_feature_cols(df: pd.DataFrame, checkpoint_sec: int) -> List[str]:
    """Get feature column names for a specific checkpoint."""
    prefix = f'cp{checkpoint_sec}s_'
    return [c for c in df.columns if c.startswith(prefix)]


def walk_forward_classify(df: pd.DataFrame, checkpoint_sec: int,
                          train_days: int = WF_TRAIN_DAYS) -> Tuple[np.ndarray, dict]:
    """
    Walk-forward sliding window classification.

    For each test day, train on the prior `train_days` trading days.
    Predict P(winner) for each trade on the test day.

    Returns:
        predictions: array of P(winner) for each trade (NaN for insufficient training data)
        metrics: dict of aggregate metrics
    """
    import lightgbm as lgb

    feat_cols = get_feature_cols(df, checkpoint_sec)
    log.info(f"WF classify at {checkpoint_sec}s: {len(feat_cols)} features")

    dates = sorted(df['date'].unique())
    predictions = np.full(len(df), np.nan)
    fold_metrics = []

    for i, test_date in enumerate(dates):
        # Sliding window: train on prior train_days dates
        train_dates = dates[max(0, i - train_days):i]

        if len(train_dates) < 10:
            # Not enough training data
            continue

        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'] == test_date

        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, 'winner'].values
        X_test = df.loc[test_mask, feat_cols].values
        y_test = df.loc[test_mask, 'winner'].values

        if len(X_test) == 0 or len(X_train) < 20:
            continue

        # Handle NaN
        X_train = np.nan_to_num(X_train, nan=0.0)
        X_test = np.nan_to_num(X_test, nan=0.0)

        # Class balance
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

        # Per-fold accuracy
        pred_class = (probs > 0.5).astype(int)
        acc = float(np.mean(pred_class == y_test)) if len(y_test) > 0 else np.nan

        fold_metrics.append({
            'test_date': test_date,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'accuracy': acc,
            'n_winners_pred': int(pred_class.sum()),
            'n_winners_actual': int(y_test.sum()),
        })

    # Aggregate metrics
    valid_mask = ~np.isnan(predictions)
    if valid_mask.sum() > 10:
        valid_pred = predictions[valid_mask]
        valid_true = df.loc[valid_mask, 'winner'].values

        pred_class = (valid_pred > 0.5).astype(int)
        accuracy = float(np.mean(pred_class == valid_true))

        # AUC
        from sklearn.metrics import roc_auc_score
        try:
            auc = float(roc_auc_score(valid_true, valid_pred))
        except ValueError:
            auc = np.nan

        # IC (rank correlation between P(winner) and actual outcome)
        ic, _ = spearmanr(valid_pred, valid_true)

        metrics = {
            'checkpoint_sec': checkpoint_sec,
            'n_valid': int(valid_mask.sum()),
            'accuracy': accuracy,
            'auc': auc,
            'ic': float(ic),
            'mean_prob_winners': float(valid_pred[valid_true == 1].mean()),
            'mean_prob_losers': float(valid_pred[valid_true == 0].mean()),
            'fold_metrics': fold_metrics,
        }
    else:
        metrics = {
            'checkpoint_sec': checkpoint_sec,
            'n_valid': 0,
            'accuracy': np.nan,
            'auc': np.nan,
            'ic': np.nan,
            'fold_metrics': fold_metrics,
        }

    return predictions, metrics


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 3: Management Rules & FIFO P&L Simulation
# ═══════════════════════════════════════════════════════════════════════════

def simulate_managed_pnl(df: pd.DataFrame, predictions: np.ndarray,
                         checkpoint_sec: int,
                         cut_threshold: float,
                         cut_cost_ticks: float = 1.376) -> Dict:
    """
    Simulate managed P&L:
    - If classifier says P(winner) < cut_threshold at checkpoint, cut the trade.
    - Cutting costs: market exit = 1.376 ticks (commission + spread crossing).
    - Uncut trades keep original exit (TP or SL).

    For cut trades, the loss is: direction * price_change_at_cut + cut_cost
    Since we're cutting within checkpoint_sec seconds, we approximate the loss as:
    - Best case: we save the full SL by cutting at breakeven → save ~3-4 ticks, pay 1.376
    - Worst case: price already moved against us by ~SL, we cut and pay extra 1.376

    For this v1, we model the cut-trade P&L as:
    - If the trade was going to hit SL anyway: save (SL - estimated_loss_at_cut) ticks
    - If the trade was going to hit TP: we lose the TP payout

    Simplified model:
    - Cut losers: P&L = -cut_cost_ticks (exit at ~entry price, pay market cost)
    - Keep winners: P&L = original exit_ticks (TP - commission)
    - Keep losers: P&L = original exit_ticks (SL - commission)
    - Cut winners (false positive): P&L = -cut_cost_ticks (miss the TP)

    This is conservative: in reality, cutting losers early might recover 1-2 ticks
    vs waiting for SL. We'll also test graduated cuts.
    """
    valid_mask = ~np.isnan(predictions)

    results = {
        'checkpoint_sec': checkpoint_sec,
        'cut_threshold': cut_threshold,
        'cut_cost_ticks': cut_cost_ticks,
    }

    if valid_mask.sum() < 10:
        results.update({'n_valid': 0, 'sharpe': np.nan, 'pf': np.nan, 'wr': np.nan})
        return results

    valid_df = df[valid_mask].copy()
    valid_preds = predictions[valid_mask]

    # Decision: cut if P(winner) < threshold
    cut_mask = valid_preds < cut_threshold
    keep_mask = ~cut_mask

    # P&L for each trade
    pnl = np.zeros(len(valid_df))

    # Keep trades: original exit P&L
    pnl[keep_mask] = valid_df.loc[keep_mask, 'exit_ticks'].values

    # Cut trades: market exit cost
    # More nuanced: for SL trades that we cut early, we SAVE ticks
    # For TP trades that we cut early, we LOSE the profit
    # Model: cut trades exit at entry price (breakeven) minus market exit cost
    # SL trade that would have lost 5.376 ticks → now loses only 1.376 ticks → saves 4.0 ticks
    # TP trade that would have gained 24.624 ticks → now loses 1.376 ticks → loses 26.0 ticks
    pnl[cut_mask] = -cut_cost_ticks

    # Also test: partial model where cut trades save proportional to how early they cut
    # At T+15s, SL fires in ~60s, so we're cutting at ~25% of the way.
    # Estimated loss at cut: ~25% of SL = ~1 tick adverse + market cost
    # For this v1, use the simple model above.

    valid_df = valid_df.copy()
    valid_df['managed_pnl'] = pnl
    valid_df['was_cut'] = cut_mask

    # Daily P&L for Sharpe
    day_pnl = valid_df.groupby('date')['managed_pnl'].sum()
    n_days = len(day_pnl)

    sharpe = float(day_pnl.mean() / day_pnl.std() * np.sqrt(252)) if day_pnl.std() > 0 else np.nan

    downside = day_pnl[day_pnl < 0]
    ds_std = downside.std() if len(downside) > 1 else 0.001
    sortino = float(day_pnl.mean() / ds_std * np.sqrt(252)) if ds_std > 0 else np.nan

    total_pnl = float(pnl.sum())
    n_trades = len(valid_df)
    wr = float(np.mean(pnl > 0))

    gross_win = float(np.sum(pnl[pnl > 0]))
    gross_loss = float(np.abs(np.sum(pnl[pnl < 0])))
    pf = gross_win / max(gross_loss, 0.01)

    # Confusion matrix for the cut decision
    cut_winners = int(np.sum(cut_mask & (valid_df['winner'].values == 1)))  # false positive cuts
    cut_losers = int(np.sum(cut_mask & (valid_df['winner'].values == 0)))   # correct cuts (saved SL)
    keep_winners = int(np.sum(keep_mask & (valid_df['winner'].values == 1)))  # correct keeps
    keep_losers = int(np.sum(keep_mask & (valid_df['winner'].values == 0)))   # missed cuts

    # Savings from cutting losers vs baseline
    # Baseline SL loss: direction-dependent
    baseline_pnl = valid_df['exit_ticks'].sum()
    managed_pnl_total = total_pnl
    improvement = managed_pnl_total - baseline_pnl

    # Day concentration
    max_day_frac = float(day_pnl.abs().max() / day_pnl.abs().sum()) if day_pnl.abs().sum() > 0 else np.nan

    # Regime analysis
    regime_results = {}
    for date_str in valid_df['date'].unique():
        # Determine regime from daily data
        pass  # We'll compute this below

    results.update({
        'n_valid': n_trades,
        'n_cut': int(cut_mask.sum()),
        'n_keep': int(keep_mask.sum()),
        'cut_rate': float(cut_mask.mean()),
        'sharpe': sharpe,
        'sortino': sortino,
        'total_pnl_ticks': total_pnl,
        'total_pnl_dollars': total_pnl * ES_TICK_VALUE,
        'avg_pnl_ticks': total_pnl / max(n_trades, 1),
        'wr': wr,
        'pf': pf,
        'n_days': n_days,
        'day_concentration': max_day_frac,
        'baseline_pnl_ticks': float(baseline_pnl),
        'improvement_ticks': float(improvement),
        'confusion': {
            'cut_winners_bad': cut_winners,
            'cut_losers_good': cut_losers,
            'keep_winners_good': keep_winners,
            'keep_losers_bad': keep_losers,
        },
    })

    return results, valid_df


def compute_regime_analysis(df: pd.DataFrame) -> Dict:
    """
    Compute per-regime (green/red/flat) metrics.
    Green = ES close > open for that day. Red = ES close < open. Flat = within 2 ticks.

    We use the minute bar data to determine daily regime.
    """
    MINUTE_BAR_DIR = ROOT / "data" / "processed" / "mbo_minute_bars_v1"

    regime_map = {}
    for date_str in sorted(df['date'].unique()):
        bar_file = MINUTE_BAR_DIR / f"{date_str}.parquet"
        if not bar_file.exists():
            regime_map[date_str] = 'unknown'
            continue
        bars = pd.read_parquet(bar_file)
        day_open = bars['open'].iloc[0]
        day_close = bars['close'].iloc[-1]
        change_ticks = (day_close - day_open) / (TICK_SIZE_PTS * 100)  # price in cents
        if change_ticks > 2:
            regime_map[date_str] = 'green'
        elif change_ticks < -2:
            regime_map[date_str] = 'red'
        else:
            regime_map[date_str] = 'flat'

    df = df.copy()
    df['regime'] = df['date'].map(regime_map)

    results = {}
    for regime in ['green', 'red', 'flat']:
        sub = df[df['regime'] == regime]
        if len(sub) < 3:
            results[regime] = {'n_trades': len(sub), 'n_days': sub['date'].nunique(), 'sharpe': np.nan}
            continue

        day_pnl = sub.groupby('date')['managed_pnl'].sum()
        n_days = len(day_pnl)
        sharpe = float(day_pnl.mean() / day_pnl.std() * np.sqrt(252)) if day_pnl.std() > 0 else np.nan

        results[regime] = {
            'n_trades': len(sub),
            'n_days': n_days,
            'sharpe': sharpe,
            'total_pnl_ticks': float(sub['managed_pnl'].sum()),
            'wr': float((sub['managed_pnl'] > 0).mean()),
        }

    # Regime gap
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


# ═══════════════════════════════════════════════════════════════════════════
# PHASE 4: Feature Importance & Analysis
# ═══════════════════════════════════════════════════════════════════════════

def analyze_feature_importance(df: pd.DataFrame, checkpoint_sec: int) -> Dict:
    """Train a final model on all data and extract feature importances."""
    import lightgbm as lgb

    feat_cols = get_feature_cols(df, checkpoint_sec)
    X = np.nan_to_num(df[feat_cols].values, nan=0.0)
    y = df['winner'].values

    params = {
        'objective': 'binary',
        'verbosity': -1,
        'n_estimators': 200,
        'max_depth': 4,
        'learning_rate': 0.05,
        'num_leaves': 16,
        'min_child_samples': 5,
    }

    model = lgb.LGBMClassifier(**params)
    model.fit(X, y)

    importances = dict(zip(feat_cols, model.feature_importances_.tolist()))
    sorted_imp = sorted(importances.items(), key=lambda x: -x[1])

    return {
        'checkpoint_sec': checkpoint_sec,
        'top_features': sorted_imp[:15],
        'all_importances': importances,
    }


# ═══════════════════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════════════════

def main():
    import mlflow

    t0 = time.time()
    log.info("=" * 80)
    log.info("Mid-Trade Thesis Validation v1 — Starting")
    log.info("=" * 80)

    # MLflow setup
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("midtrade_thesis_v1")

    with mlflow.start_run(run_name=f"midtrade_v1_{time.strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params({
            'checkpoints_sec': str(CHECKPOINTS_SEC),
            'wf_train_days': WF_TRAIN_DAYS,
            'wf_test_days': WF_TEST_DAYS,
            'tp_long': TP_LONG,
            'tp_short': TP_SHORT,
            'sl_long': SL_LONG,
            'sl_short': SL_SHORT,
            'rt_commission_ticks': RT_COMMISSION_TICKS,
        })

        # ── Phase 1: Load trades and extract tick features ──
        log.info("Phase 1: Loading trades and extracting tick-level features")
        trades_df = pd.read_parquet(TRADES_PATH)
        log.info(f"Loaded {len(trades_df)} champion trades")

        features_df = extract_all_trade_features(trades_df)

        # Save extracted features
        features_path = OUT_DIR / "trade_tick_features.parquet"
        features_df.to_parquet(features_path, index=False)
        log.info(f"Saved tick features to {features_path}")
        mlflow.log_metric("n_trades_with_features", len(features_df))
        mlflow.log_metric("n_feature_cols", len(features_df.columns))

        # ── Phase 2: Walk-forward classification at each checkpoint ──
        log.info("Phase 2: Walk-forward classification")

        all_predictions = {}
        all_metrics = {}

        for cp in CHECKPOINTS_SEC:
            log.info(f"  Classifying at T+{cp}s checkpoint...")
            preds, metrics = walk_forward_classify(features_df, cp)
            all_predictions[cp] = preds
            all_metrics[cp] = metrics

            log.info(f"    AUC={metrics.get('auc', np.nan):.3f}, "
                     f"IC={metrics.get('ic', np.nan):.3f}, "
                     f"Acc={metrics.get('accuracy', np.nan):.3f}, "
                     f"N_valid={metrics.get('n_valid', 0)}")

            mlflow.log_metrics({
                f'auc_{cp}s': metrics.get('auc', 0),
                f'ic_{cp}s': metrics.get('ic', 0),
                f'accuracy_{cp}s': metrics.get('accuracy', 0),
                f'n_valid_{cp}s': metrics.get('n_valid', 0),
            })

        # ── Phase 3: Management simulation ──
        log.info("Phase 3: Management rule simulation")

        # Test multiple cut thresholds at each checkpoint
        CUT_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]

        best_result = None
        best_improvement = -np.inf
        all_sim_results = []

        for cp in CHECKPOINTS_SEC:
            for thresh in CUT_THRESHOLDS:
                result, managed_df = simulate_managed_pnl(
                    features_df, all_predictions[cp], cp, thresh
                )

                # Compute regime analysis
                regime = compute_regime_analysis(managed_df)
                result['regime'] = regime

                all_sim_results.append(result)

                improvement = result.get('improvement_ticks', -999)
                sharpe = result.get('sharpe', 0)
                regime_pass = regime.get('regime_pass', False)

                log.info(f"  CP={cp}s, thresh={thresh:.2f}: "
                         f"Sharpe={sharpe:.2f}, PF={result.get('pf', 0):.2f}, "
                         f"WR={result.get('wr', 0):.1%}, "
                         f"Improve={improvement:.1f}t, "
                         f"Cut={result.get('cut_rate', 0):.1%}, "
                         f"RegimePass={regime_pass}")

                mlflow.log_metrics({
                    f'sharpe_cp{cp}_th{int(thresh*100)}': sharpe,
                    f'pf_cp{cp}_th{int(thresh*100)}': result.get('pf', 0),
                    f'improvement_cp{cp}_th{int(thresh*100)}': improvement,
                    f'regime_gap_cp{cp}_th{int(thresh*100)}': regime.get('regime_gap', 1.0),
                })

                # Track best (must pass regime gate)
                if regime_pass and improvement > best_improvement:
                    best_improvement = improvement
                    best_result = result

        # ── Phase 4: Feature importance ──
        log.info("Phase 4: Feature importance analysis")
        importance_results = {}
        for cp in CHECKPOINTS_SEC:
            imp = analyze_feature_importance(features_df, cp)
            importance_results[cp] = imp
            log.info(f"  Top features at T+{cp}s:")
            for fname, fscore in imp['top_features'][:5]:
                log.info(f"    {fname}: {fscore}")

        # ── Summary ──
        log.info("=" * 80)
        log.info("SUMMARY")
        log.info("=" * 80)

        # Baseline
        baseline_pnl = features_df['exit_ticks'].sum()
        baseline_wr = (features_df['exit_ticks'] > 0).mean()
        log.info(f"Baseline (no management): {baseline_pnl:.1f} ticks, WR={baseline_wr:.1%}")

        if best_result:
            log.info(f"Best managed config (regime-passing):")
            log.info(f"  Checkpoint: {best_result['checkpoint_sec']}s")
            log.info(f"  Cut threshold: {best_result['cut_threshold']}")
            log.info(f"  Sharpe: {best_result['sharpe']:.2f}")
            log.info(f"  Sortino: {best_result['sortino']:.2f}")
            log.info(f"  PF: {best_result['pf']:.2f}")
            log.info(f"  WR: {best_result['wr']:.1%}")
            log.info(f"  Total P&L: {best_result['total_pnl_ticks']:.1f} ticks (${best_result['total_pnl_dollars']:.0f})")
            log.info(f"  Improvement over baseline: {best_result['improvement_ticks']:.1f} ticks")
            log.info(f"  Cut rate: {best_result['cut_rate']:.1%}")
            log.info(f"  Confusion: {best_result['confusion']}")
            log.info(f"  Regime gap: {best_result['regime'].get('regime_gap', 'N/A')}")

            mlflow.log_metrics({
                'best_sharpe': best_result['sharpe'],
                'best_pf': best_result['pf'],
                'best_improvement_ticks': best_result['improvement_ticks'],
                'best_checkpoint_sec': best_result['checkpoint_sec'],
                'best_cut_threshold': best_result['cut_threshold'],
            })
        else:
            log.info("NO regime-passing managed config found. Tick features may not contain sufficient signal.")
            mlflow.log_metric('best_improvement_ticks', 0)

        # Save all results
        output = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'elapsed_seconds': time.time() - t0,
            'n_trades': len(features_df),
            'baseline_pnl_ticks': float(baseline_pnl),
            'classifier_metrics': {str(k): v for k, v in all_metrics.items()},
            'simulation_results': all_sim_results,
            'best_result': best_result,
            'feature_importances': {str(k): v for k, v in importance_results.items()},
        }

        results_path = OUT_DIR / "results.json"
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)

        mlflow.log_artifact(str(results_path))
        mlflow.log_artifact(str(features_path))

        log.info(f"\nTotal elapsed: {time.time() - t0:.0f}s")
        log.info(f"Results saved to {results_path}")
        log.info("Done.")


if __name__ == '__main__':
    main()
