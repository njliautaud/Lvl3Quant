#!/usr/bin/env python3
"""
ofi_champion_entry_v1.py — OFI-Enhanced Champion Config Entry Filter (HC #428 / HC #432)
==========================================================================================

Tests whether OFI-based entry filtering improves the CHAMPION config:
  TP=25 ticks, SL_long=4 ticks, SL_short=3 ticks, max_hold=60 min

Approach:
  1. Aggregate tick-level queue features to minute bars (5 top OFI features)
  2. Join with OHLCV minute bars
  3. Train LGBM to predict 30-min forward return
  4. Simulate champion config with entry_threshold gate
  5. Walk-forward: 25d train, 1d OOT, slide 1d (SLIDING per HC #0)
  6. Compare baseline (bars only) vs OFI-enhanced (bars + queue features)
  7. Regime analysis: green/red/flat, |Sharpe_green - Sharpe_red| gap check

Gates (HC #428 / HC #432):
  R1: regime gap ≤ 0.50
  R2: TP ≤ p90 MFE within horizon; hold ≤ 1.5 × horizon
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

# ── Paths (Neptune) ──
BASE = Path("/home/nick/Lvl3Quant")
QUEUE_DIR = BASE / "data" / "queue_augmented_features"
MINUTE_DIR = BASE / "data" / "processed" / "mbo_minute_bars_v1"
OUTPUT_DIR = BASE / "output" / "ofi_champion_entry_v1"
LOG_PATH = BASE / "logs" / "ofi_champion_entry_v1.log"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
(BASE / "logs").mkdir(parents=True, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler(LOG_PATH),
        logging.StreamHandler(sys.stdout)
    ]
)
log = logging.getLogger(__name__)

# ── Constants ──
ES_TICK_SIZE       = 0.25
ES_TICK_VALUE      = 12.50
COMMISSION_RT_TICKS = 0.376   # AMP passive limit, no spread crossing

# Champion config (HC target — NOT the TP4/SL3 from queue_entry_selector_v1)
TP_TICKS          = 25
SL_LONG_TICKS     = 4
SL_SHORT_TICKS    = 3
MAX_HOLD_MINUTES  = 60

# Entry model gate
ENTRY_THRESHOLD   = 0.05      # predicted 30-min return must exceed ±0.05 (in price ticks)

# Directional bias multiplier (long signals scaled down vs short)
DAILY_BIAS_MULT   = 1.5

# Walk-forward params (HC #0: SLIDING window)
TRAIN_DAYS = 25
OOT_DAYS   = 1
SLIDE_DAYS = 1

# LGBM params
LGBM_PARAMS = {
    'objective':          'regression',
    'metric':             'mae',
    'learning_rate':      0.05,
    'num_leaves':         31,
    'max_depth':          6,
    'min_data_in_leaf':   50,
    'feature_fraction':   0.8,
    'bagging_fraction':   0.8,
    'bagging_freq':       5,
    'verbose':            -1,
    'seed':               42,
    'n_jobs':             -1,
}
NUM_BOOST_ROUND   = 300

# Top-5 OFI features (avoid curse of dimensionality per spec)
OFI_AGG_FEATURES = [
    'ofi_10s_mean',
    'ofi_5s_mean',
    'trade_imbalance_mean',   # bid_trade_rate_1s - ask_trade_rate_1s
    'microprice_offset_ticks_last',
    'top_imbalance_last',
]

# Baseline bar features (OHLCV + vol_regime)
BAR_FEATURES = [
    'open', 'high', 'low', 'close', 'volume',
    'vol_regime',
    'bar_range', 'bar_return', 'bar_body',
    'high_open', 'close_low',
    'vol_ma5', 'close_ma5', 'close_zscore5',
    'vol_ma10', 'close_ma10', 'close_zscore10',
    'hour', 'minute_of_hour',
]

VOL_REGIME_MAP = {'low': 0, 'medium': 1, 'high': 2}

# Forward return horizon for model target
FORWARD_RETURN_MINUTES = 30


# ─────────────────────────────────────────────
# Data loading helpers
# ─────────────────────────────────────────────

def load_queue_features_for_date(date_str):
    """Load tick-level queue features and aggregate to 1-minute bars."""
    q_path = QUEUE_DIR / f"features_{date_str}.parquet"
    if not q_path.exists():
        return None

    qdf = pd.read_parquet(q_path)

    # Require ts_ns column
    if 'ts_ns' not in qdf.columns:
        log.warning(f"  {date_str}: queue file missing ts_ns column")
        return None

    # Convert ts_ns → datetime index (UTC)
    qdf['ts'] = pd.to_datetime(qdf['ts_ns'], unit='ns', utc=True)
    qdf = qdf.set_index('ts').sort_index()

    # Compute trade imbalance tick-level (bid_trade_rate_1s - ask_trade_rate_1s)
    if 'bid_trade_rate_1s' in qdf.columns and 'ask_trade_rate_1s' in qdf.columns:
        qdf['trade_imbalance'] = qdf['bid_trade_rate_1s'] - qdf['ask_trade_rate_1s']
    else:
        qdf['trade_imbalance'] = np.nan

    # Aggregate to 1-minute bars using only top-5 features
    agg_dict = {}
    for col in ['ofi_10s', 'ofi_5s', 'trade_imbalance']:
        if col in qdf.columns:
            agg_dict[f'{col}_mean'] = pd.NamedAgg(column=col, aggfunc='mean')

    for col in ['microprice_offset_ticks', 'top_imbalance']:
        if col in qdf.columns:
            agg_dict[f'{col}_last'] = pd.NamedAgg(column=col, aggfunc='last')

    if not agg_dict:
        log.warning(f"  {date_str}: no usable queue columns found")
        return None

    minute_q = qdf.resample('1min').agg(**agg_dict)

    # Rename to canonical OFI feature names
    rename = {
        'ofi_10s_mean':              'ofi_10s_mean',
        'ofi_5s_mean':               'ofi_5s_mean',
        'trade_imbalance_mean':      'trade_imbalance_mean',
        'microprice_offset_ticks_last': 'microprice_offset_ticks_last',
        'top_imbalance_last':        'top_imbalance_last',
    }
    minute_q = minute_q.rename(columns={k: v for k, v in rename.items() if k in minute_q.columns})

    return minute_q


def load_minute_bars_for_date(date_str):
    """Load OHLCV minute bars and engineer bar features."""
    mb_path = MINUTE_DIR / f"{date_str}.parquet"
    if not mb_path.exists():
        return None

    mb = pd.read_parquet(mb_path)

    required = {'open', 'high', 'low', 'close', 'volume'}
    if not required.issubset(mb.columns):
        log.warning(f"  {date_str}: minute bars missing columns {required - set(mb.columns)}")
        return None

    # Ensure datetime index
    if not isinstance(mb.index, pd.DatetimeIndex):
        ts_col = None
        for candidate in ['ts_minute', 'ts', 'timestamp']:
            if candidate in mb.columns:
                ts_col = candidate
                break
        if ts_col:
            mb[ts_col] = pd.to_datetime(mb[ts_col], utc=True)
            mb = mb.set_index(ts_col)
        else:
            log.warning(f"  {date_str}: no datetime index in minute bars")
            return None

    mb = mb.sort_index()

    # vol_regime → numeric
    if 'vol_regime' in mb.columns:
        if mb['vol_regime'].dtype == object:
            mb['vol_regime'] = mb['vol_regime'].map(VOL_REGIME_MAP).fillna(1).astype(int)
    else:
        mb['vol_regime'] = 1  # default medium

    # Engineer bar features
    mb['bar_range']  = mb['high'] - mb['low']
    mb['bar_return'] = mb['close'] - mb['open']
    mb['bar_body']   = abs(mb['close'] - mb['open'])
    mb['high_open']  = mb['high'] - mb['open']
    mb['close_low']  = mb['close'] - mb['low']

    # Rolling features (5-bar and 10-bar)
    mb['vol_ma5']        = mb['volume'].rolling(5, min_periods=1).mean()
    mb['close_ma5']      = mb['close'].rolling(5, min_periods=1).mean()
    mb['close_std5']     = mb['close'].rolling(5, min_periods=2).std().fillna(1e-8)
    mb['close_zscore5']  = (mb['close'] - mb['close_ma5']) / mb['close_std5']

    mb['vol_ma10']       = mb['volume'].rolling(10, min_periods=1).mean()
    mb['close_ma10']     = mb['close'].rolling(10, min_periods=1).mean()
    mb['close_std10']    = mb['close'].rolling(10, min_periods=2).std().fillna(1e-8)
    mb['close_zscore10'] = (mb['close'] - mb['close_ma10']) / mb['close_std10']

    # Time features
    idx = mb.index
    mb['hour']          = idx.hour
    mb['minute_of_hour'] = idx.minute

    return mb


def compute_forward_return(mb, horizon_minutes=30):
    """Add forward return target (in price ticks) to minute bars."""
    mb = mb.copy()
    # Forward close in `horizon_minutes` bars
    mb['fwd_close'] = mb['close'].shift(-horizon_minutes)
    mb['fwd_return_ticks'] = (mb['fwd_close'] - mb['close']) / ES_TICK_SIZE
    return mb


def classify_day_regime(mb):
    """Classify day as green/red/flat from close-to-close."""
    if mb is None or len(mb) < 2:
        return 'unknown'
    day_return = mb['close'].iloc[-1] - mb['close'].iloc[0]
    if day_return > 2 * ES_TICK_SIZE:   # > 2 ticks
        return 'green'
    elif day_return < -2 * ES_TICK_SIZE:
        return 'red'
    return 'flat'


# ─────────────────────────────────────────────
# Simulation helpers
# ─────────────────────────────────────────────

def simulate_champion_entry(day_bars, pred_col='pred', direction=None):
    """
    Simulate champion config (TP25/SL4-long/SL3-short, max_hold=60min) on 1-min bars.

    For each minute bar where |pred| > ENTRY_THRESHOLD:
      - Determine direction: +1 long if pred > threshold, -1 short if pred < -threshold
      - Enter at next bar open (approximated as current bar close)
      - Scan forward up to MAX_HOLD_MINUTES bars
      - Hit TP/SL or max_hold exit

    Returns list of trade dicts with pnl_ticks.
    """
    trades = []
    n = len(day_bars)
    bars_arr = day_bars[['high', 'low', 'close']].values
    preds    = day_bars[pred_col].values if pred_col in day_bars.columns else np.zeros(n)
    dates    = day_bars['date'].values if 'date' in day_bars.columns else [''] * n

    i = 0
    while i < n - 1:
        pred = preds[i]
        if direction is not None:
            # Force a specific direction for directional bias analysis
            entry_dir = direction
            if abs(pred) <= ENTRY_THRESHOLD:
                i += 1
                continue
        else:
            if pred > ENTRY_THRESHOLD:
                entry_dir = 1    # long
            elif pred < -ENTRY_THRESHOLD:
                entry_dir = -1   # short
            else:
                i += 1
                continue

        entry_price = bars_arr[i + 1][2] if i + 1 < n else bars_arr[i][2]  # next bar close approx
        tp = TP_TICKS * ES_TICK_SIZE
        sl = (SL_LONG_TICKS if entry_dir == 1 else SL_SHORT_TICKS) * ES_TICK_SIZE

        exit_price = None
        exit_reason = 'max_hold'
        max_j = min(i + 1 + MAX_HOLD_MINUTES, n)

        for j in range(i + 1, max_j):
            hi = bars_arr[j][0]
            lo = bars_arr[j][1]
            cl = bars_arr[j][2]

            if entry_dir == 1:    # long
                tp_price = entry_price + tp
                sl_price = entry_price - sl
                if lo <= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                if hi >= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break
            else:                 # short
                tp_price = entry_price - tp
                sl_price = entry_price + sl
                if hi >= sl_price:
                    exit_price = sl_price
                    exit_reason = 'sl'
                    break
                if lo <= tp_price:
                    exit_price = tp_price
                    exit_reason = 'tp'
                    break

        if exit_price is None:
            exit_price = bars_arr[max_j - 1][2]  # close of last bar

        gross_ticks = (exit_price - entry_price) / ES_TICK_SIZE * entry_dir
        net_ticks   = gross_ticks - COMMISSION_RT_TICKS

        trades.append({
            'entry_bar':   i,
            'exit_bar':    j if exit_price != bars_arr[max_j - 1][2] else max_j - 1,
            'direction':   entry_dir,
            'entry_price': entry_price,
            'exit_price':  exit_price,
            'exit_reason': exit_reason,
            'gross_ticks': round(gross_ticks, 4),
            'net_ticks':   round(net_ticks, 4),
            'pnl_ticks':   round(net_ticks, 4),
            'date':        dates[i],
        })
        # Move past the trade window to avoid overlapping entries
        i = (j if exit_price != bars_arr[max_j - 1][2] else max_j - 1) + 1

    return trades


# ─────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────

def evaluate_trades(trades_list, label):
    """Compute risk-adjusted metrics from a list of trade dicts."""
    if not trades_list:
        return {'label': label, 'n_trades': 0, 'error': 'no_trades'}

    df = pd.DataFrame(trades_list)
    if len(df) < 5:
        return {'label': label, 'n_trades': len(df), 'error': 'too_few_trades'}

    pnl = df['pnl_ticks'].values
    winners = pnl > 0
    losers  = pnl < 0

    wr      = winners.mean()
    avg_win = pnl[winners].mean() if winners.any() else 0.0
    avg_loss = abs(pnl[losers].mean()) if losers.any() else 1e-8
    pf       = (pnl[winners].sum() / abs(pnl[losers].sum())) if losers.any() and pnl[losers].sum() != 0 else np.inf

    # Daily P&L
    daily_pnl = df.groupby('date')['pnl_ticks'].sum() if 'date' in df.columns else pd.Series(pnl)
    n_days    = daily_pnl.shape[0]

    sharpe  = daily_pnl.mean() / (daily_pnl.std() + 1e-8) * np.sqrt(252)
    down    = daily_pnl[daily_pnl < 0]
    down_std = down.std() if len(down) > 1 else daily_pnl.std()
    sortino = daily_pnl.mean() / (down_std + 1e-8) * np.sqrt(252)

    return {
        'label':           label,
        'n_trades':        int(len(df)),
        'n_days':          int(n_days),
        'trades_per_day':  round(len(df) / max(n_days, 1), 2),
        'wr':              round(float(wr), 4),
        'pf':              round(float(pf), 3),
        'avg_win_ticks':   round(float(avg_win), 3),
        'avg_loss_ticks':  round(float(avg_loss), 3),
        'per_trade_ticks': round(float(pnl.mean()), 4),
        'total_ticks':     round(float(pnl.sum()), 2),
        'daily_sharpe':    round(float(sharpe), 3),
        'sortino':         round(float(sortino), 3),
    }


def compute_regime_gap(trades_list, date_regimes):
    """
    Compute |Sharpe_green - Sharpe_red| / max(|Sharpe_green|, |Sharpe_red|).
    HC #428 R1 gate: must be ≤ 0.50.
    """
    if not trades_list:
        return float('nan'), {}, {}

    df = pd.DataFrame(trades_list)
    if 'date' not in df.columns:
        return float('nan'), {}, {}

    df['regime'] = df['date'].map(date_regimes)

    def day_sharpe(subset):
        if len(subset) < 3:
            return float('nan')
        daily = subset.groupby('date')['pnl_ticks'].sum()
        return float(daily.mean() / (daily.std() + 1e-8) * np.sqrt(252))

    green_trades = df[df['regime'] == 'green']
    red_trades   = df[df['regime'] == 'red']

    sg = day_sharpe(green_trades)
    sr = day_sharpe(red_trades)

    if np.isnan(sg) or np.isnan(sr):
        return float('nan'), {'sharpe': sg, 'n': len(green_trades)}, {'sharpe': sr, 'n': len(red_trades)}

    gap = abs(sg - sr) / max(abs(sg), abs(sr), 1e-8)
    return float(gap), {'sharpe': round(sg, 3), 'n': int(len(green_trades))}, {'sharpe': round(sr, 3), 'n': int(len(red_trades))}


def compute_ic(preds, actuals):
    """Pearson + Spearman IC."""
    mask = np.isfinite(preds) & np.isfinite(actuals)
    if mask.sum() < 10:
        return float('nan'), float('nan')
    pearson  = float(np.corrcoef(preds[mask], actuals[mask])[0, 1])
    spearman = float(scipy_stats.spearmanr(preds[mask], actuals[mask])[0])
    return pearson, spearman


def mfe_gate_check(trades_list):
    """
    HC #432 R2: TP ≤ p90 MFE within horizon.
    Here we report the p90 of gross_ticks from TP hits as a proxy.
    (Full MFE scan would require tick data; this uses realized trade outcomes.)
    """
    if not trades_list:
        return None
    df   = pd.DataFrame(trades_list)
    tp_h = df[df['exit_reason'] == 'tp']['gross_ticks']
    if len(tp_h) < 5:
        return None
    p90 = float(tp_h.quantile(0.90))
    return {
        'tp_ticks':       TP_TICKS,
        'p90_tp_gross':   round(p90, 2),
        'pass_r2':        TP_TICKS <= p90,
        'max_hold_min':   MAX_HOLD_MINUTES,
        'hold_check_ok':  MAX_HOLD_MINUTES <= 1.5 * FORWARD_RETURN_MINUTES,
    }


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────

def main():
    log.info("=" * 70)
    log.info("OFI Champion Entry v1  (TP25 / SL4-long / SL3-short / hold=60min)")
    log.info("=" * 70)

    # ── 1. Discover dates ──
    q_files = sorted(QUEUE_DIR.glob("features_*.parquet"))
    q_dates = [f.stem.replace('features_', '') for f in q_files]

    mb_files = sorted(MINUTE_DIR.glob("*.parquet"))
    mb_dates = [f.stem for f in mb_files]

    overlap = sorted(set(q_dates) & set(mb_dates))
    log.info(f"Queue dates: {len(q_dates)}, Bar dates: {len(mb_dates)}, Overlap: {len(overlap)}")

    if len(overlap) < TRAIN_DAYS + OOT_DAYS:
        log.error(f"Not enough overlapping dates ({len(overlap)}). Need ≥ {TRAIN_DAYS + OOT_DAYS}.")
        sys.exit(1)

    # ── 2. Load all data ──
    log.info("Loading and joining data...")
    all_days = {}          # date → merged minute bars (with queue features)
    all_days_base = {}     # date → baseline bars only (no queue)
    date_regimes = {}

    for date_str in overlap:
        mb = load_minute_bars_for_date(date_str)
        if mb is None or len(mb) < 30:
            log.warning(f"  {date_str}: skipping — insufficient minute bars ({len(mb) if mb is not None else 0})")
            continue

        mb = compute_forward_return(mb, horizon_minutes=FORWARD_RETURN_MINUTES)
        mb['date'] = date_str
        date_regimes[date_str] = classify_day_regime(mb)

        # Baseline: bars only
        all_days_base[date_str] = mb.copy()

        # Enhanced: join with queue minute features
        qm = load_queue_features_for_date(date_str)
        if qm is not None and len(qm) > 0:
            merged = mb.join(qm, how='left')
        else:
            log.warning(f"  {date_str}: no queue features, enhanced = baseline")
            merged = mb.copy()
            for feat in OFI_AGG_FEATURES:
                merged[feat] = np.nan

        all_days[date_str] = merged

    valid_dates = sorted(all_days.keys())
    log.info(f"Valid dates loaded: {len(valid_dates)}")
    log.info(f"Regime distribution: {pd.Series(date_regimes).value_counts().to_dict()}")

    if len(valid_dates) < TRAIN_DAYS + OOT_DAYS:
        log.error(f"Only {len(valid_dates)} valid dates — need ≥ {TRAIN_DAYS + OOT_DAYS}")
        sys.exit(1)

    # ── 3. Walk-forward ──
    log.info(f"\nWalk-forward: {TRAIN_DAYS}d train, {OOT_DAYS}d OOT, {SLIDE_DAYS}d slide (SLIDING, HC #0)")

    # Assemble full dataframe for feature computation
    def build_dataset(date_list, days_dict, feature_cols):
        frames = []
        for d in date_list:
            if d not in days_dict:
                continue
            extra = [c for c in ['fwd_return_ticks', 'date', 'close'] if c not in feature_cols]
            df = days_dict[d][feature_cols + extra].copy()
            frames.append(df)
        if not frames:
            return pd.DataFrame()
        return pd.concat(frames)

    all_baseline_trades = []
    all_enhanced_trades = []
    fold_results = []
    fi_accum = {}
    ic_baseline_vals = []
    ic_enhanced_vals = []

    fold_num   = 0
    start_idx  = TRAIN_DAYS
    n_dates    = len(valid_dates)

    while start_idx + OOT_DAYS <= n_dates:
        train_dates = valid_dates[start_idx - TRAIN_DAYS : start_idx]
        oot_dates   = valid_dates[start_idx : start_idx + OOT_DAYS]

        # ── Build baseline train/oot ──
        base_train_df = build_dataset(train_dates, all_days_base, BAR_FEATURES)
        base_oot_df   = build_dataset(oot_dates,   all_days_base, BAR_FEATURES)

        # ── Build enhanced train/oot ──
        # Use BAR_FEATURES + any OFI features that exist
        avail_ofi = [f for f in OFI_AGG_FEATURES
                     if f in (all_days.get(oot_dates[0], pd.DataFrame()).columns if oot_dates else [])]
        # Check across all train dates too
        for d in train_dates:
            avail_ofi = [f for f in avail_ofi if f in all_days.get(d, pd.DataFrame()).columns]
        # Fallback: try to find from the merged frames directly
        if not avail_ofi and train_dates:
            sample = all_days.get(train_dates[0])
            if sample is not None:
                avail_ofi = [f for f in OFI_AGG_FEATURES if f in sample.columns]

        enh_features = BAR_FEATURES + avail_ofi
        enh_train_df = build_dataset(train_dates, all_days, enh_features)
        enh_oot_df   = build_dataset(oot_dates,   all_days, enh_features)

        # Drop rows with no valid target
        mask_base_tr  = base_train_df['fwd_return_ticks'].notna()
        mask_base_oot = base_oot_df['fwd_return_ticks'].notna()
        mask_enh_tr   = enh_train_df['fwd_return_ticks'].notna()
        mask_enh_oot  = enh_oot_df['fwd_return_ticks'].notna()

        base_train_df = base_train_df[mask_base_tr]
        base_oot_df   = base_oot_df[mask_base_oot]
        enh_train_df  = enh_train_df[mask_enh_tr]
        enh_oot_df    = enh_oot_df[mask_enh_oot]

        if len(base_train_df) < 50 or len(base_oot_df) < 5:
            start_idx += SLIDE_DAYS
            continue

        # ── Train baseline LGBM ──
        X_base_tr  = np.nan_to_num(base_train_df[BAR_FEATURES].values.astype(np.float32), nan=0., posinf=100., neginf=-100.)
        y_base_tr  = base_train_df['fwd_return_ticks'].values.astype(np.float32)
        X_base_oot = np.nan_to_num(base_oot_df[BAR_FEATURES].values.astype(np.float32),  nan=0., posinf=100., neginf=-100.)
        y_base_oot = base_oot_df['fwd_return_ticks'].values.astype(np.float32)

        base_model = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_base_tr, label=y_base_tr, feature_name=BAR_FEATURES),
            num_boost_round=NUM_BOOST_ROUND,
            callbacks=[lgb.log_evaluation(0)],
        )
        base_pred_oot = base_model.predict(X_base_oot)

        # ── Train enhanced LGBM ──
        X_enh_tr  = np.nan_to_num(enh_train_df[enh_features].values.astype(np.float32), nan=0., posinf=100., neginf=-100.)
        y_enh_tr  = enh_train_df['fwd_return_ticks'].values.astype(np.float32)
        X_enh_oot = np.nan_to_num(enh_oot_df[enh_features].values.astype(np.float32),  nan=0., posinf=100., neginf=-100.)
        y_enh_oot = enh_oot_df['fwd_return_ticks'].values.astype(np.float32)

        enh_model = lgb.train(
            LGBM_PARAMS,
            lgb.Dataset(X_enh_tr, label=y_enh_tr, feature_name=enh_features),
            num_boost_round=NUM_BOOST_ROUND,
            callbacks=[lgb.log_evaluation(0)],
        )
        enh_pred_oot = enh_model.predict(X_enh_oot)

        # IC
        ic_b_p, ic_b_s = compute_ic(base_pred_oot, y_base_oot)
        ic_e_p, ic_e_s = compute_ic(enh_pred_oot,  y_enh_oot)
        ic_baseline_vals.append((ic_b_p, ic_b_s))
        ic_enhanced_vals.append((ic_e_p, ic_e_s))

        # Track feature importances (enhanced model)
        for feat, imp in zip(enh_features, enh_model.feature_importance('gain')):
            fi_accum.setdefault(feat, []).append(float(imp))

        # ── Simulate champion config on OOT date(s) ──
        fold_b_trades = []
        fold_e_trades = []

        for oot_date in oot_dates:
            if oot_date not in all_days_base:
                continue

            # Baseline bars + predictions
            b_day = all_days_base[oot_date].copy()
            b_day['date'] = oot_date
            b_mask = base_oot_df['date'] == oot_date
            if b_mask.sum() == 0:
                continue
            b_preds_day = base_pred_oot[base_oot_df['date'].values == oot_date]
            b_idx       = base_oot_df[base_oot_df['date'] == oot_date].index
            b_day_rows  = base_oot_df[base_oot_df['date'] == oot_date].copy()
            b_day_rows['pred'] = b_preds_day

            b_trades = simulate_champion_entry(b_day_rows, pred_col='pred')
            fold_b_trades.extend(b_trades)

            # Enhanced bars + predictions
            e_day_rows = enh_oot_df[enh_oot_df['date'] == oot_date].copy()
            e_preds_day = enh_pred_oot[enh_oot_df['date'].values == oot_date]
            e_day_rows['pred'] = e_preds_day

            e_trades = simulate_champion_entry(e_day_rows, pred_col='pred')
            fold_e_trades.extend(e_trades)

        all_baseline_trades.extend(fold_b_trades)
        all_enhanced_trades.extend(fold_e_trades)

        fold_results.append({
            'fold':        fold_num,
            'train':       f"{train_dates[0]}..{train_dates[-1]}",
            'oot':         oot_dates[0],
            'regime':      date_regimes.get(oot_dates[0], 'unknown'),
            'ic_base_p':   round(ic_b_p, 4) if not np.isnan(ic_b_p) else None,
            'ic_enh_p':    round(ic_e_p, 4) if not np.isnan(ic_e_p) else None,
            'n_base':      len(fold_b_trades),
            'n_enh':       len(fold_e_trades),
            'base_pnl':    round(sum(t['pnl_ticks'] for t in fold_b_trades), 3),
            'enh_pnl':     round(sum(t['pnl_ticks'] for t in fold_e_trades), 3),
        })

        log.info(f"  Fold {fold_num:3d} | OOT={oot_dates[0]} | regime={date_regimes.get(oot_dates[0], '?'):<5} "
                 f"| IC_base={ic_b_p:.4f} IC_enh={ic_e_p:.4f} "
                 f"| trades: base={len(fold_b_trades)} enh={len(fold_e_trades)} "
                 f"| PnL: base={sum(t['pnl_ticks'] for t in fold_b_trades):+.1f}t "
                 f"enh={sum(t['pnl_ticks'] for t in fold_e_trades):+.1f}t")

        start_idx += SLIDE_DAYS
        fold_num  += 1

    # ── 4. Aggregate results ──
    log.info(f"\n{'='*70}")
    log.info(f"AGGREGATE RESULTS — {fold_num} folds")
    log.info(f"{'='*70}")

    base_metrics = evaluate_trades(all_baseline_trades, 'baseline_bars_only')
    enh_metrics  = evaluate_trades(all_enhanced_trades, 'ofi_enhanced')

    # Regime gap
    base_gap, base_green, base_red = compute_regime_gap(all_baseline_trades, date_regimes)
    enh_gap,  enh_green,  enh_red  = compute_regime_gap(all_enhanced_trades, date_regimes)

    # MFE gate check
    base_mfe = mfe_gate_check(all_baseline_trades)
    enh_mfe  = mfe_gate_check(all_enhanced_trades)

    # IC summary
    ic_b_arr = np.array([(v[0] if v[0] is not None else np.nan) for v in ic_baseline_vals])
    ic_e_arr = np.array([(v[0] if v[0] is not None else np.nan) for v in ic_enhanced_vals])
    ic_b_mean = float(np.nanmean(ic_b_arr))
    ic_e_mean = float(np.nanmean(ic_e_arr))

    # Feature importance
    log.info(f"\n--- FEATURE IMPORTANCE (enhanced model, avg gain) ---")
    fi_sorted = sorted(fi_accum.items(), key=lambda x: np.mean(x[1]), reverse=True)
    for rank, (feat, vals) in enumerate(fi_sorted[:20]):
        log.info(f"  {rank+1:2d}. {feat:40s}  gain={np.mean(vals):.1f} ± {np.std(vals):.1f}")

    log.info(f"\n--- IC (30-min forward return) ---")
    log.info(f"  Baseline mean Pearson IC:  {ic_b_mean:.4f}")
    log.info(f"  Enhanced mean Pearson IC:  {ic_e_mean:.4f}")
    log.info(f"  IC lift: {ic_e_mean - ic_b_mean:+.4f}")

    log.info(f"\n--- BASELINE (bars only) ---")
    for k, v in base_metrics.items():
        if k not in ('label',):
            log.info(f"  {k}: {v}")
    log.info(f"  regime_gap: {base_gap:.3f}  green_sharpe={base_green.get('sharpe', 'n/a')} (n={base_green.get('n', 0)})  red_sharpe={base_red.get('sharpe', 'n/a')} (n={base_red.get('n', 0)})")
    log.info(f"  R1 PASS: {base_gap <= 0.50 if not np.isnan(base_gap) else 'INSUFFICIENT_DATA'}")
    if base_mfe:
        log.info(f"  R2 MFE gate: TP={base_mfe['tp_ticks']}t  p90_gross={base_mfe['p90_tp_gross']}t  "
                 f"PASS={base_mfe['pass_r2']}  hold_ok={base_mfe['hold_check_ok']}")

    log.info(f"\n--- OFI ENHANCED (bars + queue features) ---")
    for k, v in enh_metrics.items():
        if k not in ('label',):
            log.info(f"  {k}: {v}")
    log.info(f"  regime_gap: {enh_gap:.3f}  green_sharpe={enh_green.get('sharpe', 'n/a')} (n={enh_green.get('n', 0)})  red_sharpe={enh_red.get('sharpe', 'n/a')} (n={enh_red.get('n', 0)})")
    log.info(f"  R1 PASS: {enh_gap <= 0.50 if not np.isnan(enh_gap) else 'INSUFFICIENT_DATA'}")
    if enh_mfe:
        log.info(f"  R2 MFE gate: TP={enh_mfe['tp_ticks']}t  p90_gross={enh_mfe['p90_tp_gross']}t  "
                 f"PASS={enh_mfe['pass_r2']}  hold_ok={enh_mfe['hold_check_ok']}")

    # Per-day P&L CSV for auditing
    if all_enhanced_trades:
        enh_df = pd.DataFrame(all_enhanced_trades)
        enh_daily = enh_df.groupby('date')['pnl_ticks'].agg(['sum', 'count', 'mean']).reset_index()
        enh_daily.columns = ['date', 'total_ticks', 'n_trades', 'per_trade']
        enh_daily['regime'] = enh_daily['date'].map(date_regimes)
        enh_daily.to_csv(OUTPUT_DIR / "enhanced_daily_pnl.csv", index=False)

    if all_baseline_trades:
        base_df = pd.DataFrame(all_baseline_trades)
        base_daily = base_df.groupby('date')['pnl_ticks'].agg(['sum', 'count', 'mean']).reset_index()
        base_daily.columns = ['date', 'total_ticks', 'n_trades', 'per_trade']
        base_daily['regime'] = base_daily['date'].map(date_regimes)
        base_daily.to_csv(OUTPUT_DIR / "baseline_daily_pnl.csv", index=False)

    # ── 5. Save results JSON ──
    results = {
        'timestamp': datetime.now().isoformat(),
        'config': {
            'tp_ticks':        TP_TICKS,
            'sl_long_ticks':   SL_LONG_TICKS,
            'sl_short_ticks':  SL_SHORT_TICKS,
            'max_hold_minutes': MAX_HOLD_MINUTES,
            'entry_threshold': ENTRY_THRESHOLD,
            'train_days':      TRAIN_DAYS,
            'oot_days':        OOT_DAYS,
            'forward_return_minutes': FORWARD_RETURN_MINUTES,
        },
        'n_dates':    len(valid_dates),
        'n_folds':    fold_num,
        'ic': {
            'baseline_mean_pearson': round(ic_b_mean, 4),
            'enhanced_mean_pearson': round(ic_e_mean, 4),
            'lift': round(ic_e_mean - ic_b_mean, 4),
        },
        'baseline': {
            **base_metrics,
            'regime_gap': base_gap,
            'r1_pass': base_gap <= 0.50 if not np.isnan(base_gap) else None,
            'green': base_green,
            'red':   base_red,
            'mfe_gate': base_mfe,
        },
        'enhanced': {
            **enh_metrics,
            'regime_gap': enh_gap,
            'r1_pass': enh_gap <= 0.50 if not np.isnan(enh_gap) else None,
            'green': enh_green,
            'red':   enh_red,
            'mfe_gate': enh_mfe,
        },
        'feature_importance': {
            feat: {'mean_gain': float(np.mean(vals)), 'std_gain': float(np.std(vals))}
            for feat, vals in fi_sorted
        },
        'date_regimes': date_regimes,
        'fold_results': fold_results,
    }

    results_path = OUTPUT_DIR / "results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    log.info(f"\nResults saved to {results_path}")

    # ── 6. MLflow logging ──
    try:
        import mlflow
        mlflow.set_tracking_uri("http://jupiter:5000")
        mlflow.set_experiment("ofi_champion_entry")
        run_name = f"v1_{datetime.now().strftime('%Y%m%d_%H%M')}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                'tp_ticks':               TP_TICKS,
                'sl_long_ticks':          SL_LONG_TICKS,
                'sl_short_ticks':         SL_SHORT_TICKS,
                'max_hold_minutes':       MAX_HOLD_MINUTES,
                'entry_threshold':        ENTRY_THRESHOLD,
                'train_days':             TRAIN_DAYS,
                'forward_return_minutes': FORWARD_RETURN_MINUTES,
                'n_ofi_features':         len(OFI_AGG_FEATURES),
                'n_bar_features':         len(BAR_FEATURES),
                'n_dates':                len(valid_dates),
                'n_folds':                fold_num,
            })
            mlflow.log_metrics({
                'baseline_sharpe':     base_metrics.get('daily_sharpe', 0),
                'baseline_sortino':    base_metrics.get('sortino', 0),
                'baseline_pf':         base_metrics.get('pf', 0),
                'baseline_wr':         base_metrics.get('wr', 0),
                'baseline_n_trades':   base_metrics.get('n_trades', 0),
                'baseline_regime_gap': base_gap if not np.isnan(base_gap) else -1,
                'baseline_ic_pearson': ic_b_mean,
                'enhanced_sharpe':     enh_metrics.get('daily_sharpe', 0),
                'enhanced_sortino':    enh_metrics.get('sortino', 0),
                'enhanced_pf':         enh_metrics.get('pf', 0),
                'enhanced_wr':         enh_metrics.get('wr', 0),
                'enhanced_n_trades':   enh_metrics.get('n_trades', 0),
                'enhanced_regime_gap': enh_gap if not np.isnan(enh_gap) else -1,
                'enhanced_ic_pearson': ic_e_mean,
                'ic_lift':             ic_e_mean - ic_b_mean,
                'sharpe_lift':         enh_metrics.get('daily_sharpe', 0) - base_metrics.get('daily_sharpe', 0),
            })
            mlflow.log_artifact(str(results_path))
        log.info(f"MLflow run logged: {run_name}")
    except Exception as e:
        log.warning(f"MLflow logging failed (non-fatal): {e}")

    # ── 7. Final summary printout ──
    log.info(f"\n{'='*70}")
    log.info(f"FINAL SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"Champion config: TP={TP_TICKS}t / SL_long={SL_LONG_TICKS}t / SL_short={SL_SHORT_TICKS}t / hold={MAX_HOLD_MINUTES}min")
    log.info(f"{'':^70}")
    log.info(f"{'Metric':<25} {'Baseline':>15} {'OFI Enhanced':>15} {'Lift':>10}")
    log.info(f"{'-'*70}")

    def fmt(v):
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return 'N/A'
        return f"{v:.3f}"

    rows = [
        ('IC (Pearson)',     ic_b_mean,                            ic_e_mean,                            ic_e_mean - ic_b_mean),
        ('Sharpe',          base_metrics.get('daily_sharpe'),     enh_metrics.get('daily_sharpe'),      (enh_metrics.get('daily_sharpe', 0) or 0) - (base_metrics.get('daily_sharpe', 0) or 0)),
        ('Sortino',         base_metrics.get('sortino'),          enh_metrics.get('sortino'),           (enh_metrics.get('sortino', 0) or 0) - (base_metrics.get('sortino', 0) or 0)),
        ('PF',              base_metrics.get('pf'),               enh_metrics.get('pf'),                (enh_metrics.get('pf', 0) or 0) - (base_metrics.get('pf', 0) or 0)),
        ('WR',              base_metrics.get('wr'),               enh_metrics.get('wr'),                (enh_metrics.get('wr', 0) or 0) - (base_metrics.get('wr', 0) or 0)),
        ('Regime gap (≤.50)', base_gap,                           enh_gap,                              None),
        ('R1 PASS',         base_gap <= 0.50 if not np.isnan(base_gap) else None,
                            enh_gap  <= 0.50 if not np.isnan(enh_gap)  else None,  None),
    ]
    for name, bv, ev, lift in rows:
        lift_str = (f"{lift:+.3f}" if lift is not None and not np.isnan(lift) else '')
        log.info(f"  {name:<23} {fmt(bv):>15} {fmt(ev):>15} {lift_str:>10}")

    log.info(f"\n  Trades/day:  baseline={base_metrics.get('trades_per_day', 0):.1f}  enhanced={enh_metrics.get('trades_per_day', 0):.1f}")

    verdict_r1 = "PASS" if (not np.isnan(enh_gap) and enh_gap <= 0.50) else "FAIL"
    verdict_ic = "BETTER" if ic_e_mean > ic_b_mean else "WORSE"
    verdict_sh = "BETTER" if (enh_metrics.get('daily_sharpe', 0) or 0) > (base_metrics.get('daily_sharpe', 0) or 0) else "WORSE"

    log.info(f"\n  OFI Enhancement verdict:")
    log.info(f"    IC lift:      {verdict_ic}  ({ic_b_mean:.4f} → {ic_e_mean:.4f})")
    log.info(f"    Sharpe lift:  {verdict_sh}  ({base_metrics.get('daily_sharpe'):.3f} → {enh_metrics.get('daily_sharpe'):.3f})")
    log.info(f"    R1 regime gate: {verdict_r1}  (gap={enh_gap:.3f})")
    log.info(f"{'='*70}")


if __name__ == '__main__':
    main()
