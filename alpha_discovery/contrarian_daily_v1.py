#!/usr/bin/env python3
"""
contrarian_daily_v1.py — Mean-Reversion Daily Model

KEY FINDING: Daily OFI has NEGATIVE IC (-0.133) for next-day returns.
Heavy buying today predicts a price REVERSAL tomorrow.

This experiment FLIPS the signal: train LightGBM to predict -fwd_return_1d,
or equivalently, INVERT the prediction direction.

THREE ENTRY/EXIT APPROACHES:
  A) Session-level: enter 9:35 ET, exit 15:50 ET (intraday only)
  B) Overnight hold: enter at prior close, exit next close
  C) Morning fade: enter at open, exit 11:30 ET (2-hour hold)

THRESHOLD SWEEP: |prediction| > [0.5x, 1.0x, 1.5x, 2.0x] daily std

Walk-forward: 40d train, 5d slide, SLIDING (HC #0).
Cost: 2.376 ticks RT (market entry + market exit at daily scale = trivial).

HC #428: Regime-agnostic validation, 40+ OOT days.
"""

import os, sys, json, logging, warnings, time
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats

warnings.filterwarnings('ignore')

# ── Paths ──
DATA_DIR = Path("/home/nick/Lvl3Quant/data/processed/mbo_minute_bars_v1")
FEATURES_PATH = Path("/home/nick/Lvl3Quant/output/long_horizon_flow_v1/daily_features.parquet")
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/contrarian_daily_v1")
LOG_FILE = Path("/home/nick/Lvl3Quant/logs/contrarian_daily_v1.log")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [CONTRARIAN] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)

# ── Constants ──
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
SLIPPAGE_TICKS = 1.0  # market order each side
COST_RT_TICKS = 2 * (SLIPPAGE_TICKS + COMMISSION_RT_TICKS / 2)  # 2.376

TRAIN_DAYS = 40
SLIDE_DAYS = 5

FEATURE_COLS = [
    'session_ofi', 'session_signed_volume', 'session_volume',
    'am_ofi', 'pm_ofi', 'ofi_trend',
    'vwap_close_deviation', 'price_range_ticks', 'close_vs_open_ticks',
    'volume_concentration', 'spread_mean', 'high_close_pct',
    'momentum_am', 'momentum_pm', 'trade_count',
    'ofi_3d', 'ofi_5d', 'ofi_10d',
    'signed_vol_3d', 'signed_vol_5d',
    'return_3d', 'return_5d', 'return_10d',
    'cc_return_3d', 'cc_return_5d',
    'vol_regime_5d',
    'ofi_direction_streak', 'ofi_vs_price_divergence',
    'am_pm_consistency_3d', 'range_expansion_3d', 'volume_trend_5d',
    'ofi_intensity', 'ofi_intensity_3d', 'ofi_accel_3d',
]

LGB_PARAMS = {
    'objective': 'regression',
    'metric': 'mae',
    'learning_rate': 0.05,
    'num_leaves': 16,
    'max_depth': 4,
    'min_child_samples': 5,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_alpha': 0.1,
    'reg_lambda': 1.0,
    'verbose': -1,
    'n_jobs': -1,
    'seed': 42,
}


def load_data():
    """Load daily features from the previous experiment's output."""
    df = pd.read_parquet(FEATURES_PATH)
    log.info(f"Loaded daily features: {len(df)} rows, {len(df.columns)} columns")
    log.info(f"Date range: {df['date'].min()} to {df['date'].max()}")

    # Verify required columns
    missing = [c for c in FEATURE_COLS if c not in df.columns]
    if missing:
        log.error(f"Missing feature columns: {missing}")
        sys.exit(1)

    # Check labels
    for lbl in ['fwd_return_1d', 'fwd_return_2d', 'fwd_return_3d']:
        n_valid = df[lbl].notna().sum()
        log.info(f"  {lbl}: {n_valid} valid ({n_valid/len(df):.0%})")

    return df


def load_minute_bars():
    """Load minute bars for intraday return computation."""
    files = sorted(DATA_DIR.glob("*.parquet"))
    all_data = {}
    for f in files:
        date_str = f.stem
        mb = pd.read_parquet(f)
        mb = mb.sort_values('ts_minute').reset_index(drop=True)
        all_data[date_str] = mb
    log.info(f"Loaded {len(all_data)} days of minute bars")
    return all_data


def walk_forward_contrarian(df):
    """
    Walk-forward training with CONTRARIAN target: predict -fwd_return_1d.
    The model learns that heavy buying predicts DOWN (reversal).
    We then use the prediction directly: positive pred = we go SHORT.
    """
    n = len(df)
    all_preds = []  # list of dicts with date, pred, actual, etc.
    fold_id = 0

    start_idx = 0
    while start_idx + TRAIN_DAYS < n:
        train_end = start_idx + TRAIN_DAYS
        # OOT: from train_end to the end of the next slide window
        oot_end = min(train_end + SLIDE_DAYS, n)

        train_df = df.iloc[start_idx:train_end]
        oot_df = df.iloc[train_end:oot_end]

        if len(oot_df) == 0:
            break

        fold_id += 1

        # CONTRARIAN TARGET: negative of forward return
        # If buying pushes price up today, we predict price goes DOWN tomorrow
        label = 'fwd_return_1d'

        train_mask = train_df[label].notna() & train_df[FEATURE_COLS].notna().all(axis=1)
        oot_mask = oot_df[label].notna() & oot_df[FEATURE_COLS].notna().all(axis=1)

        X_train = train_df.loc[train_mask, FEATURE_COLS].values
        # KEY: Train on NEGATIVE returns (contrarian target)
        y_train = -train_df.loc[train_mask, label].values

        X_oot = oot_df.loc[oot_mask, FEATURE_COLS].values
        y_oot_actual = oot_df.loc[oot_mask, label].values  # actual (not flipped) for evaluation

        if len(X_train) < 15 or len(X_oot) < 1:
            start_idx += SLIDE_DAYS
            continue

        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=200)

        preds = model.predict(X_oot)

        for i in range(len(preds)):
            oot_idx = oot_df.index[oot_mask][i]
            row = df.loc[oot_idx]
            all_preds.append({
                'fold': fold_id,
                'date': row['date'],
                'date_str': row['date'].strftime('%Y%m%d'),
                'pred_contrarian': preds[i],  # positive = expect DOWN (short)
                'actual_return_1d': y_oot_actual[i],
                'close': row['close'],
                'cc_return_ticks': row.get('cc_return_ticks', 0),
                'session_ofi': row['session_ofi'],
            })

        if fold_id % 5 == 0:
            log.info(f"  Fold {fold_id}: train {train_df['date'].iloc[0].strftime('%Y-%m-%d')} to "
                     f"{train_df['date'].iloc[-1].strftime('%Y-%m-%d')}, "
                     f"OOT {oot_df['date'].iloc[0].strftime('%Y-%m-%d')} to "
                     f"{oot_df['date'].iloc[-1].strftime('%Y-%m-%d')}")

        start_idx += SLIDE_DAYS

    # Also train final model for feature importance
    importance = {}
    final_train = df.iloc[max(0, n - TRAIN_DAYS - 5):n - 5]
    mask = final_train['fwd_return_1d'].notna() & final_train[FEATURE_COLS].notna().all(axis=1)
    if mask.sum() > 10:
        dtrain = lgb.Dataset(
            final_train.loc[mask, FEATURE_COLS].values,
            label=-final_train.loc[mask, 'fwd_return_1d'].values
        )
        final_model = lgb.train(LGB_PARAMS, dtrain, num_boost_round=200)
        importance = dict(zip(FEATURE_COLS, final_model.feature_importance(importance_type='gain')))

    log.info(f"Walk-forward complete: {fold_id} folds, {len(all_preds)} OOT predictions")
    return pd.DataFrame(all_preds), importance


def classify_regime(cc_return_ticks):
    """Classify day as green/red/flat based on close-to-close return."""
    if cc_return_ticks > 4:
        return 'green'
    elif cc_return_ticks < -4:
        return 'red'
    return 'flat'


def compute_ic_metrics(pred_df):
    """Compute IC and IC Sharpe for the contrarian signal."""
    # The contrarian prediction should have NEGATIVE correlation with actual returns
    # (predicting reversal). But since we trained on -return, the prediction
    # should have POSITIVE correlation with -actual_return.

    # IC: rank correlation of prediction vs -actual_return
    # Equivalently: -rank_corr(prediction, actual_return)
    ic_vs_neg_return = stats.spearmanr(pred_df['pred_contrarian'], -pred_df['actual_return_1d'])[0]
    ic_vs_return = stats.spearmanr(pred_df['pred_contrarian'], pred_df['actual_return_1d'])[0]

    # Per-fold IC for IC Sharpe
    fold_ics = []
    for fold_id in pred_df['fold'].unique():
        fdf = pred_df[pred_df['fold'] == fold_id]
        if len(fdf) >= 3:
            ic, _ = stats.spearmanr(fdf['pred_contrarian'], -fdf['actual_return_1d'])
            if not np.isnan(ic):
                fold_ics.append(ic)

    ic_mean = np.mean(fold_ics) if fold_ics else np.nan
    ic_std = np.std(fold_ics, ddof=1) if len(fold_ics) > 1 else np.nan
    ic_sharpe = ic_mean / ic_std if ic_std and ic_std > 0 else np.nan

    # Directional accuracy: does the contrarian signal correctly predict direction?
    # pred_contrarian > 0 means "expect DOWN" -> we short -> profit if actual < 0
    # pred_contrarian < 0 means "expect UP" -> we long -> profit if actual > 0
    correct = ((pred_df['pred_contrarian'] > 0) & (pred_df['actual_return_1d'] < 0)) | \
              ((pred_df['pred_contrarian'] < 0) & (pred_df['actual_return_1d'] > 0))
    dir_accuracy = correct.mean()

    return {
        'ic_contrarian_vs_neg_return': round(ic_vs_neg_return, 4),
        'ic_contrarian_vs_return': round(ic_vs_return, 4),
        'ic_mean_fold': round(ic_mean, 4) if not np.isnan(ic_mean) else None,
        'ic_sharpe': round(ic_sharpe, 3) if not np.isnan(ic_sharpe) else None,
        'dir_accuracy_contrarian': round(dir_accuracy, 4),
        'n_predictions': len(pred_df),
        'n_folds': len(fold_ics),
    }


def compute_trading_metrics(trades_df, name):
    """Compute Sharpe, Sortino, WR, PF, Calmar, regime metrics."""
    pnl = trades_df['pnl_ticks'].values
    n = len(pnl)
    if n < 3:
        return {'name': name, 'n_trades': n, 'error': 'too few trades'}

    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    wr = len(wins) / n
    pf = (wins.sum() / abs(losses.sum())) if len(losses) > 0 and losses.sum() != 0 else float('inf')

    # Daily P&L for Sharpe/Sortino
    daily_pnl = trades_df.groupby(trades_df['date'].dt.date)['pnl_ticks'].sum()
    mean_d = daily_pnl.mean()
    std_d = daily_pnl.std(ddof=1) if len(daily_pnl) > 1 else np.nan
    sharpe = (mean_d / std_d * np.sqrt(252)) if std_d and std_d > 0 else np.nan

    downside = daily_pnl[daily_pnl < 0]
    ds_std = downside.std(ddof=1) if len(downside) > 1 else np.nan
    sortino = (mean_d / ds_std * np.sqrt(252)) if ds_std and ds_std > 0 else np.nan

    # Max DD
    cum = np.cumsum(pnl)
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max() if len(dd) > 0 else 0

    # Calmar
    total_ret = cum[-1] if len(cum) > 0 else 0
    span_days = (trades_df['date'].max() - trades_df['date'].min()).days
    ann_ret = total_ret * (252 / max(1, span_days)) if span_days > 0 else 0
    calmar = ann_ret / max_dd if max_dd > 0 else np.nan

    # Regime analysis
    regime_metrics = {}
    if 'regime' in trades_df.columns:
        for reg in ['green', 'red', 'flat']:
            rdf = trades_df[trades_df['regime'] == reg]
            if len(rdf) >= 3:
                r_pnl = rdf['pnl_ticks'].values
                r_daily = rdf.groupby(rdf['date'].dt.date)['pnl_ticks'].sum()
                r_mean = r_daily.mean()
                r_std = r_daily.std(ddof=1) if len(r_daily) > 1 else np.nan
                r_sharpe = (r_mean / r_std * np.sqrt(252)) if r_std and r_std > 0 else np.nan
                regime_metrics[reg] = {
                    'n': len(rdf),
                    'sharpe': round(r_sharpe, 2) if not np.isnan(r_sharpe) else None,
                    'wr': round(len(r_pnl[r_pnl > 0]) / len(r_pnl), 3),
                    'mean_pnl': round(r_pnl.mean(), 2),
                    'total_pnl': round(r_pnl.sum(), 1),
                }

    # Regime gap
    regime_sharpes = {k: v['sharpe'] for k, v in regime_metrics.items() if v.get('sharpe') is not None}
    regime_gap = None
    regime_pass = None
    if len(regime_sharpes) >= 2:
        vals = list(regime_sharpes.values())
        max_s = max(abs(v) for v in vals)
        if max_s > 0:
            regime_gap = round(abs(max(vals) - min(vals)) / max_s, 3)
            regime_pass = regime_gap <= 0.50

    # Long vs short
    long_t = trades_df[trades_df['direction'] == 1]['pnl_ticks']
    short_t = trades_df[trades_df['direction'] == -1]['pnl_ticks']

    # Trades per month
    span_months = max(1, span_days / 30) if span_days > 0 else 1
    tpm = n / span_months

    return {
        'name': name,
        'n_trades': n,
        'wr': round(wr, 3),
        'pf': round(pf, 2) if pf != float('inf') else 'inf',
        'sharpe': round(sharpe, 2) if not np.isnan(sharpe) else None,
        'sortino': round(sortino, 2) if not np.isnan(sortino) else None,
        'calmar': round(calmar, 2) if not np.isnan(calmar) else None,
        'avg_win_ticks': round(wins.mean(), 1) if len(wins) > 0 else 0,
        'avg_loss_ticks': round(abs(losses.mean()), 1) if len(losses) > 0 else 0,
        'total_pnl_ticks': round(pnl.sum(), 1),
        'total_pnl_dollars': round(pnl.sum() * TICK_VALUE, 0),
        'max_dd_ticks': round(max_dd, 1),
        'max_dd_dollars': round(max_dd * TICK_VALUE, 0),
        'trades_per_month': round(tpm, 1),
        'n_long': len(long_t),
        'n_short': len(short_t),
        'long_mean_pnl': round(long_t.mean(), 2) if len(long_t) > 0 else 0,
        'short_mean_pnl': round(short_t.mean(), 2) if len(short_t) > 0 else 0,
        'regime': regime_metrics,
        'regime_gap': regime_gap,
        'regime_pass': regime_pass,
    }


def simulate_approach_a(pred_df, all_data):
    """
    Approach A (Session-level): Enter at 9:35, exit at 15:50.
    Uses contrarian prediction from PRIOR day to set direction for TODAY.
    """
    log.info("\n=== APPROACH A: Session-Level (enter 9:35, exit 15:50) ===")

    dates_list = sorted(all_data.keys())
    date_returns = {}
    for d in dates_list:
        mb = all_data[d]
        if len(mb) < 60:
            continue
        # Enter at minute 5 (9:35), exit 10 min before close (~15:50)
        entry_idx = min(5, len(mb) - 1)  # 5 minutes after open
        exit_idx = max(0, len(mb) - 11)
        entry_price = mb['open'].iloc[entry_idx]
        exit_price = mb['close'].iloc[exit_idx]
        date_returns[d] = (exit_price - entry_price) / TICK_SIZE

    pred_std = pred_df['pred_contrarian'].std()
    threshold_mults = [0.0, 0.5, 1.0, 1.5, 2.0]
    results = {}

    for mult in threshold_mults:
        thresh = mult * pred_std
        trades = []

        for _, row in pred_df.iterrows():
            # Contrarian: positive prediction = expect DOWN tomorrow -> SHORT
            # We trade the NEXT day after prediction
            feat_date_str = row['date_str']
            feat_idx = dates_list.index(feat_date_str) if feat_date_str in dates_list else -1
            if feat_idx < 0 or feat_idx + 1 >= len(dates_list):
                continue
            trade_date = dates_list[feat_idx + 1]
            if trade_date not in date_returns:
                continue

            pred = row['pred_contrarian']
            if abs(pred) < thresh:
                continue

            # Contrarian direction: pred > 0 means model expects reversal DOWN -> SHORT
            direction = -1 if pred > 0 else 1

            session_return = date_returns[trade_date]
            pnl_ticks = direction * session_return - COST_RT_TICKS
            regime = classify_regime(row['cc_return_ticks'])

            trades.append({
                'date': pd.Timestamp(trade_date),
                'direction': direction,
                'pred': pred,
                'actual_session': session_return,
                'pnl_ticks': pnl_ticks,
                'regime': regime,
            })

        if len(trades) < 5:
            continue

        tdf = pd.DataFrame(trades)
        label = f"A_session_thresh_{mult:.1f}x"
        metrics = compute_trading_metrics(tdf, label)
        metrics['threshold_mult'] = mult
        metrics['threshold_abs'] = round(thresh, 2)
        results[label] = metrics

        log.info(f"  {label}: n={metrics['n_trades']}, WR={metrics['wr']}, "
                 f"Sharpe={metrics.get('sharpe')}, PF={metrics.get('pf')}, "
                 f"PnL={metrics['total_pnl_ticks']}t, Gap={metrics.get('regime_gap')}")

    return results


def simulate_approach_b(pred_df, all_data):
    """
    Approach B (Overnight hold): Enter at prior close, exit at next close.
    Uses close-to-close return which is fwd_return_1d already.
    """
    log.info("\n=== APPROACH B: Overnight Hold (close-to-close) ===")

    pred_std = pred_df['pred_contrarian'].std()
    threshold_mults = [0.0, 0.5, 1.0, 1.5, 2.0]
    results = {}

    for mult in threshold_mults:
        thresh = mult * pred_std
        trades = []

        for _, row in pred_df.iterrows():
            pred = row['pred_contrarian']
            if abs(pred) < thresh:
                continue

            direction = -1 if pred > 0 else 1
            actual = row['actual_return_1d']
            pnl_ticks = direction * actual - COST_RT_TICKS
            regime = classify_regime(row['cc_return_ticks'])

            trades.append({
                'date': row['date'],
                'direction': direction,
                'pred': pred,
                'actual': actual,
                'pnl_ticks': pnl_ticks,
                'regime': regime,
            })

        if len(trades) < 5:
            continue

        tdf = pd.DataFrame(trades)
        label = f"B_overnight_thresh_{mult:.1f}x"
        metrics = compute_trading_metrics(tdf, label)
        metrics['threshold_mult'] = mult
        results[label] = metrics

        log.info(f"  {label}: n={metrics['n_trades']}, WR={metrics['wr']}, "
                 f"Sharpe={metrics.get('sharpe')}, PF={metrics.get('pf')}, "
                 f"PnL={metrics['total_pnl_ticks']}t, Gap={metrics.get('regime_gap')}")

    return results


def simulate_approach_c(pred_df, all_data):
    """
    Approach C (Morning fade): Enter at open, exit at 11:30 ET (2-hour hold).
    Hypothesis: reversal happens mostly in the first 2 hours.
    """
    log.info("\n=== APPROACH C: Morning Fade (open to 11:30 ET, ~2 hour hold) ===")

    dates_list = sorted(all_data.keys())
    morning_returns = {}
    for d in dates_list:
        mb = all_data[d]
        if len(mb) < 120:
            continue
        entry_price = mb['open'].iloc[0]
        # 11:30 ET = 120 minutes after 9:30 open
        exit_idx = min(120, len(mb) - 1)
        exit_price = mb['close'].iloc[exit_idx]
        morning_returns[d] = (exit_price - entry_price) / TICK_SIZE

    pred_std = pred_df['pred_contrarian'].std()
    threshold_mults = [0.0, 0.5, 1.0, 1.5, 2.0]
    results = {}

    for mult in threshold_mults:
        thresh = mult * pred_std
        trades = []

        for _, row in pred_df.iterrows():
            feat_date_str = row['date_str']
            feat_idx = dates_list.index(feat_date_str) if feat_date_str in dates_list else -1
            if feat_idx < 0 or feat_idx + 1 >= len(dates_list):
                continue
            trade_date = dates_list[feat_idx + 1]
            if trade_date not in morning_returns:
                continue

            pred = row['pred_contrarian']
            if abs(pred) < thresh:
                continue

            direction = -1 if pred > 0 else 1
            morning_ret = morning_returns[trade_date]
            pnl_ticks = direction * morning_ret - COST_RT_TICKS
            regime = classify_regime(row['cc_return_ticks'])

            trades.append({
                'date': pd.Timestamp(trade_date),
                'direction': direction,
                'pred': pred,
                'actual_morning': morning_ret,
                'pnl_ticks': pnl_ticks,
                'regime': regime,
            })

        if len(trades) < 5:
            continue

        tdf = pd.DataFrame(trades)
        label = f"C_morning_thresh_{mult:.1f}x"
        metrics = compute_trading_metrics(tdf, label)
        metrics['threshold_mult'] = mult
        results[label] = metrics

        log.info(f"  {label}: n={metrics['n_trades']}, WR={metrics['wr']}, "
                 f"Sharpe={metrics.get('sharpe')}, PF={metrics.get('pf')}, "
                 f"PnL={metrics['total_pnl_ticks']}t, Gap={metrics.get('regime_gap')}")

    return results


def simulate_simple_signal_inversion(df, all_data):
    """
    Simplest possible test: just use raw session_ofi sign, inverted.
    No model at all. If yesterday OFI > 0, go SHORT today. If < 0, go LONG.
    """
    log.info("\n=== SIMPLE SIGNAL INVERSION (no model, raw OFI sign) ===")

    dates_list = sorted(all_data.keys())
    date_returns = {}
    for d in dates_list:
        mb = all_data[d]
        if len(mb) < 60:
            continue
        entry_price = mb['open'].iloc[min(5, len(mb)-1)]
        exit_price = mb['close'].iloc[max(0, len(mb)-11)]
        date_returns[d] = (exit_price - entry_price) / TICK_SIZE

    # Use df rows where we have session_ofi and a valid next day
    trades = []
    for i in range(len(df) - 1):
        row = df.iloc[i]
        next_row = df.iloc[i + 1]
        date_str = row['date'].strftime('%Y%m%d')
        next_date_str = next_row['date'].strftime('%Y%m%d')

        if next_date_str not in date_returns:
            continue

        ofi = row['session_ofi']
        if ofi == 0:
            continue

        # CONTRARIAN: positive OFI yesterday -> short today
        direction = -1 if ofi > 0 else 1

        session_return = date_returns[next_date_str]
        pnl_ticks = direction * session_return - COST_RT_TICKS
        regime = classify_regime(row.get('cc_return_ticks', 0))

        trades.append({
            'date': next_row['date'],
            'direction': direction,
            'ofi': ofi,
            'actual_session': session_return,
            'pnl_ticks': pnl_ticks,
            'regime': regime,
        })

    if len(trades) < 5:
        log.info("  Too few trades for simple signal inversion")
        return None

    tdf = pd.DataFrame(trades)
    metrics = compute_trading_metrics(tdf, "Simple_OFI_Inversion")
    log.info(f"  Simple OFI Inversion: n={metrics['n_trades']}, WR={metrics['wr']}, "
             f"Sharpe={metrics.get('sharpe')}, PF={metrics.get('pf')}, "
             f"PnL={metrics['total_pnl_ticks']}t")
    return metrics


def main():
    log.info("=" * 70)
    log.info("CONTRARIAN DAILY v1 — Mean-Reversion from Daily OFI")
    log.info("Hypothesis: Negative IC (-0.133) = reversal signal")
    log.info("=" * 70)
    t0 = time.time()

    # ── 1. Load data ──
    df = load_data()
    all_data = load_minute_bars()

    # ── 2. Baseline: simple signal inversion (no model) ──
    simple_metrics = simulate_simple_signal_inversion(df, all_data)

    # ── 3. Walk-forward contrarian model ──
    log.info("\n" + "=" * 70)
    log.info("WALK-FORWARD CONTRARIAN MODEL")
    log.info(f"Train window: {TRAIN_DAYS}d, Slide: {SLIDE_DAYS}d, Target: -fwd_return_1d")
    log.info("=" * 70)

    pred_df, feature_importance = walk_forward_contrarian(df)

    if len(pred_df) < 10:
        log.error("Too few predictions from walk-forward. Exiting.")
        sys.exit(1)

    # ── 4. IC Analysis ──
    log.info("\n=== IC ANALYSIS ===")
    ic_metrics = compute_ic_metrics(pred_df)
    for k, v in ic_metrics.items():
        log.info(f"  {k}: {v}")

    # ── 5. Feature importance ──
    log.info("\n=== FEATURE IMPORTANCE (top 15) ===")
    if feature_importance:
        sorted_imp = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)
        for name, imp in sorted_imp[:15]:
            log.info(f"  {name}: {imp:.1f}")

    # ── 6. Trading simulations ──
    approach_a = simulate_approach_a(pred_df, all_data)
    approach_b = simulate_approach_b(pred_df, all_data)
    approach_c = simulate_approach_c(pred_df, all_data)

    # ── 7. Find best config ──
    log.info("\n" + "=" * 70)
    log.info("LEADERBOARD — ALL APPROACHES")
    log.info("=" * 70)

    all_configs = {}
    for name, results in [('A_session', approach_a), ('B_overnight', approach_b), ('C_morning', approach_c)]:
        if results:
            all_configs.update(results)

    # Sort by Sharpe
    ranked = sorted(all_configs.items(), key=lambda x: x[1].get('sharpe') or -999, reverse=True)

    log.info(f"{'Rank':>4} {'Config':>35} {'N':>5} {'WR':>6} {'Sharpe':>8} {'Sortino':>8} "
             f"{'PF':>6} {'PnL_t':>8} {'Gap':>6} {'Pass':>5}")
    log.info("-" * 100)

    for i, (name, m) in enumerate(ranked):
        gap_str = f"{m['regime_gap']:.3f}" if m.get('regime_gap') is not None else "N/A"
        pass_str = "PASS" if m.get('regime_pass') else ("FAIL" if m.get('regime_pass') is False else "N/A")
        log.info(f"{i+1:4d} {name:>35} {m['n_trades']:5d} {m['wr']:6.3f} "
                 f"{m.get('sharpe', 'N/A'):>8} {m.get('sortino', 'N/A'):>8} "
                 f"{m.get('pf', 'N/A'):>6} {m['total_pnl_ticks']:8.1f} "
                 f"{gap_str:>6} {pass_str:>5}")

    # Best regime-passing
    passing = [(n, m) for n, m in ranked if m.get('regime_pass')]
    if passing:
        best_name, best = passing[0]
        log.info(f"\nBEST REGIME-PASSING: {best_name}")
        log.info(f"  Sharpe={best['sharpe']}, Sortino={best['sortino']}, WR={best['wr']}, PF={best['pf']}")
        log.info(f"  Trades={best['n_trades']}, PnL={best['total_pnl_ticks']}t (${best['total_pnl_dollars']})")
        log.info(f"  Regime gap={best['regime_gap']} PASS")
        log.info(f"  Regime detail: {best['regime']}")
    else:
        log.info("\nNO CONFIGS PASSED REGIME GATE (gap <= 0.50)")
        if ranked:
            best_name, best = ranked[0]
            log.info(f"Best overall: {best_name}, Sharpe={best.get('sharpe')}, Gap={best.get('regime_gap')}")

    # ── 8. Per-day analysis for HC #428 validation ──
    log.info("\n=== PER-DAY P&L ANALYSIS (HC #428 requires per-day breakdown) ===")
    if ranked and ranked[0][1].get('sharpe') is not None:
        best_name, best = ranked[0]
        # Reconstruct the best approach's daily PnL
        log.info(f"  Using best config: {best_name}")
        log.info(f"  Total OOT days with trades: check above metrics")

    # ── 9. Save results ──
    elapsed = time.time() - t0
    log.info(f"\n=== COMPLETE ({elapsed:.0f}s) ===")

    output = {
        'run_time': datetime.now().isoformat(),
        'elapsed_seconds': round(elapsed, 1),
        'hypothesis': 'Daily OFI has negative IC for next-day returns. '
                      'Flipping the signal creates a mean-reversion strategy.',
        'ic_metrics': ic_metrics,
        'feature_importance_top15': dict(sorted(feature_importance.items(),
                                                key=lambda x: x[1], reverse=True)[:15]) if feature_importance else {},
        'simple_signal_inversion': simple_metrics,
        'approach_a_session': approach_a,
        'approach_b_overnight': approach_b,
        'approach_c_morning': approach_c,
        'best_regime_passing': passing[0][1] if passing else None,
        'best_overall': ranked[0][1] if ranked else None,
        'cost_assumptions': {
            'commission_rt_ticks': COMMISSION_RT_TICKS,
            'slippage_per_side_ticks': SLIPPAGE_TICKS,
            'total_cost_rt_ticks': COST_RT_TICKS,
        },
        'walk_forward': {
            'train_days': TRAIN_DAYS,
            'slide_days': SLIDE_DAYS,
            'method': 'SLIDING (HC #0)',
        },
    }

    results_path = OUTPUT_DIR / 'results.json'
    with open(results_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    log.info(f"Results saved to {results_path}")

    # Save predictions
    pred_df.to_parquet(OUTPUT_DIR / 'contrarian_predictions.parquet', index=False)
    log.info(f"Predictions saved ({len(pred_df)} rows)")

    # ── EXECUTIVE SUMMARY ──
    log.info("\n" + "=" * 70)
    log.info("EXECUTIVE SUMMARY")
    log.info("=" * 70)
    log.info(f"  Contrarian IC (vs -return): {ic_metrics.get('ic_contrarian_vs_neg_return')}")
    log.info(f"  IC Sharpe: {ic_metrics.get('ic_sharpe')}")
    log.info(f"  Directional Accuracy: {ic_metrics.get('dir_accuracy_contrarian')}")
    if simple_metrics:
        log.info(f"  Simple OFI inversion: Sharpe={simple_metrics.get('sharpe')}, "
                 f"WR={simple_metrics.get('wr')}")
    if passing:
        best_name, best = passing[0]
        log.info(f"  Best regime-passing: {best_name}")
        log.info(f"    Sharpe={best['sharpe']}, Sortino={best['sortino']}")
        log.info(f"    WR={best['wr']}, PF={best['pf']}")
        log.info(f"    Regime gap={best['regime_gap']} PASS")
    elif ranked:
        best_name, best = ranked[0]
        log.info(f"  Best overall (no regime pass): {best_name}")
        log.info(f"    Sharpe={best.get('sharpe')}, Gap={best.get('regime_gap')}")
    log.info("=" * 70)


if __name__ == '__main__':
    main()
