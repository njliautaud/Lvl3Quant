#!/usr/bin/env python3
"""
filter_ablation_study.py — Progressive Filter Ablation Study

Answers: which filter layer drives the gap from Sharpe 0.25 (all preds) → Sharpe 20+ (V1 pipeline)?

V1 pipeline filters:
  Layer 1: Daily flow direction (walk-forward LGBM on OFI) → only trade in flow direction
  Layer 2: Entry confidence (conf >= 0.52) + zscore (|zscore| >= 0.3)
  Layer 3: Mid-trade classifier (mt_cut = 0.25)

Ablation configs (all use minute-level replay with SL/TP):
  A: No filters — every prediction bar, trade in predicted direction
  B: Confidence only (conf >= threshold)
  C: Confidence + zscore
  D: Confidence + zscore + flow direction
  E: Full V1 (all 3 layers including mid-trade)

Then: parameter sweep on the critical filter.
"""

import os, sys, json, warnings, time, logging
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from collections import defaultdict

warnings.filterwarnings('ignore')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Paths ──
LVL3 = Path("/home/nick/Lvl3Quant")
FLOW_FEATURES = LVL3 / "output/long_horizon_flow_v2/enhanced_daily_features.parquet"
ENTRY_PREDS = LVL3 / "output/lh_30min_deep_v1/concat_oot.npz"
MIDTRADE_FEATURES = LVL3 / "output/midtrade_thesis_v1/trade_tick_features.parquet"
MINUTE_BARS_DIR = LVL3 / "data/processed/mbo_minute_bars_v1"
OUTPUT_DIR = LVL3 / "output/filter_ablation_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = OUTPUT_DIR / "ablation.log"

# ── Constants ──
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0
COST_MARKET_EXIT = COMMISSION_RT_TICKS + SPREAD_TICKS  # 1.376
COST_PASSIVE = COMMISSION_RT_TICKS
MLFLOW_URI = "http://localhost:5000"

# V1 best config
V1_CONF = 0.52
V1_ZSCORE = 0.3
V1_TP = 12
V1_SL = 3
V1_MT_CUT = 0.25
V1_FLOW_CONF = 0.55  # flow model confidence threshold for non-flat signal

# Walk-forward params for flow model
FLOW_TRAIN_DAYS = 40
FLOW_SLIDE_DAYS = 5

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ══════════════════════════════════════════════════════════════════════
# DATA LOADING
# ══════════════════════════════════════════════════════════════════════

def load_predictions():
    """Load all 30-min bar predictions."""
    log.info("Loading predictions...")
    d = np.load(ENTRY_PREDS, allow_pickle=True)
    preds = d['preds']
    actuals = d['actuals']
    dates = d['dates']
    confs = d['confs']
    log.info(f"  {len(preds)} prediction bars, {len(np.unique(dates))} unique dates")
    return preds, actuals, dates, confs


def load_flow_signals():
    """Walk-forward LGBM on daily flow features → directional bias per day.
    Exact replication of V1 pipeline logic."""
    log.info("Building walk-forward flow signals...")
    df = pd.read_parquet(FLOW_FEATURES)
    
    exclude_prefixes = ['fwd_', 'open', 'close', 'high', 'low', 'session_vwap',
                        'trade_count', 'date', 'cc_return']
    feature_cols = [c for c in df.columns if not any(c.startswith(p) for p in exclude_prefixes)]
    
    n = len(df)
    daily_signals = {}
    
    start = FLOW_TRAIN_DAYS
    while start < n:
        train_slice = slice(max(0, start - FLOW_TRAIN_DAYS), start)
        test_end = min(start + FLOW_SLIDE_DAYS, n)
        test_slice = slice(start, test_end)
        
        X_train = np.nan_to_num(df.iloc[train_slice][feature_cols].values, 0)
        y_train = df.iloc[train_slice]['fwd_direction_3d'].values
        X_test = np.nan_to_num(df.iloc[test_slice][feature_cols].values, 0)
        
        valid_train = ~np.isnan(y_train)
        if valid_train.sum() < 15:
            start += FLOW_SLIDE_DAYS
            continue
        
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
            verbose=-1, random_state=42
        )
        model.fit(X_train[valid_train], y_train[valid_train])
        
        proba = model.predict_proba(X_test)
        classes = model.classes_
        
        for i in range(test_end - start):
            idx = start + i
            date_val = df.iloc[idx]['date']
            if hasattr(date_val, 'strftime'):
                date_str = date_val.strftime('%Y%m%d')
            else:
                date_str = str(date_val).replace('-', '')[:8]
            
            pred_class = int(model.predict(X_test[i:i+1])[0])
            class_idx = np.where(classes == pred_class)[0][0]
            confidence = float(proba[i, class_idx])
            
            if confidence >= V1_FLOW_CONF:
                direction = 1 if pred_class == 1 else -1
            else:
                direction = 0
            
            daily_signals[date_str] = {
                'direction': direction,
                'confidence': confidence,
                'pred_class': pred_class,
            }
        
        start += FLOW_SLIDE_DAYS
    
    dir_counts = defaultdict(int)
    for s in daily_signals.values():
        dir_counts[s['direction']] += 1
    log.info(f"  Flow signals: {len(daily_signals)} days, directions: {dict(dir_counts)}")
    return daily_signals


def load_bar_cache():
    """Load all minute bar data into memory."""
    log.info("Loading minute bar cache...")
    bar_cache = {}
    for f in sorted(MINUTE_BARS_DIR.glob("*.parquet")):
        date_str = f.stem
        mbars = pd.read_parquet(f)
        mbars['ts_minute'] = pd.to_datetime(mbars['ts_minute'], utc=True)
        mbars = mbars.sort_values('ts_minute').reset_index(drop=True)
        mbars['bar_key'] = mbars['ts_minute'].dt.floor('30min')
        
        bars_30 = []
        for bar_key, grp in mbars.groupby('bar_key'):
            if len(grp) < 3:
                continue
            bars_30.append({
                'bar_key': bar_key,
                'open': grp.iloc[0]['open'],
                'close': grp.iloc[-1]['close'],
                'high': grp['high'].max(),
                'low': grp['low'].min(),
                'minutes': grp,
            })
        bar_cache[date_str] = bars_30
    log.info(f"  Loaded {len(bar_cache)} days of minute data")
    return bar_cache


def load_midtrade_data():
    """Load mid-trade feature data."""
    if not MIDTRADE_FEATURES.exists():
        log.warning("No midtrade features found")
        return None
    df = pd.read_parquet(MIDTRADE_FEATURES)
    log.info(f"  Mid-trade: {len(df)} entries, {df['date'].nunique()} days")
    return df


def load_regime_map():
    """Build regime classification from flow features."""
    df = pd.read_parquet(FLOW_FEATURES)
    regime_map = {}
    for _, row in df.iterrows():
        date_val = row['date']
        if hasattr(date_val, 'strftime'):
            date_str = date_val.strftime('%Y%m%d')
        else:
            date_str = str(date_val).replace('-', '')[:8]
        cc = row.get('cc_return_ticks', 0)
        if pd.isna(cc):
            regime_map[date_str] = 'flat'
        elif cc > 20:
            regime_map[date_str] = 'green'
        elif cc < -20:
            regime_map[date_str] = 'red'
        else:
            regime_map[date_str] = 'flat'
    return regime_map


# ══════════════════════════════════════════════════════════════════════
# MINUTE-LEVEL REPLAY
# ══════════════════════════════════════════════════════════════════════

def replay_trade(entry_price, direction, minute_bars_next, tp_ticks, sl_ticks):
    """Replay a trade through minute-level data with stops."""
    if len(minute_bars_next) == 0:
        return None
    mfe = 0.0
    mae = 0.0
    for _, row in minute_bars_next.iterrows():
        hi = row['high']
        lo = row['low']
        if direction == 1:
            fav = (hi - entry_price) / TICK_SIZE
            adv = (entry_price - lo) / TICK_SIZE
            tp_hit = (hi - entry_price) >= tp_ticks * TICK_SIZE
            sl_hit = (entry_price - lo) >= sl_ticks * TICK_SIZE
        else:
            fav = (entry_price - lo) / TICK_SIZE
            adv = (hi - entry_price) / TICK_SIZE
            tp_hit = (entry_price - lo) >= tp_ticks * TICK_SIZE
            sl_hit = (hi - entry_price) >= sl_ticks * TICK_SIZE
        mfe = max(mfe, fav)
        mae = max(mae, adv)
        if tp_hit and sl_hit:
            # Both levels breached in same minute bar. Use bar close as
            # tiebreaker: if close is on favorable side of entry, TP first.
            # The old MFE/MAE heuristic was TP-biased (remaining_tp clamped
            # to 0.1 once MFE > TP, making p_sl_first near 0 always).
            bar_close = row['close']
            if direction == 1:
                favorable = bar_close >= entry_price
            else:
                favorable = bar_close <= entry_price
            if favorable:
                return {'exit_type': 'tp', 'pnl': tp_ticks, 'mfe': mfe, 'mae': mae}
            else:
                return {'exit_type': 'sl', 'pnl': -sl_ticks, 'mfe': mfe, 'mae': mae}
        elif sl_hit:
            return {'exit_type': 'sl', 'pnl': -sl_ticks, 'mfe': mfe, 'mae': mae}
        elif tp_hit:
            return {'exit_type': 'tp', 'pnl': tp_ticks, 'mfe': mfe, 'mae': mae}
    
    exit_price = minute_bars_next.iloc[-1]['close']
    if direction == 1:
        exit_pnl = (exit_price - entry_price) / TICK_SIZE
    else:
        exit_pnl = (entry_price - exit_price) / TICK_SIZE
    return {'exit_type': 'bar_close', 'pnl': exit_pnl, 'mfe': mfe, 'mae': mae}


def get_next_bar_minutes(date_str, bar_idx, bar_cache):
    """Get minute data for the bar AFTER the signal bar (trade execution window)."""
    if date_str not in bar_cache:
        return None, None
    bars = bar_cache[date_str]
    if bar_idx + 1 >= len(bars):
        return None, None
    next_bar = bars[bar_idx + 1]
    entry_price = next_bar['open']
    return entry_price, next_bar['minutes']


# ══════════════════════════════════════════════════════════════════════
# METRICS
# ══════════════════════════════════════════════════════════════════════

def compute_metrics(trades_list, regime_map=None):
    """Compute full metrics from trade list."""
    if len(trades_list) == 0:
        return {'n_trades': 0, 'sharpe': 0, 'sortino': 0, 'wr': 0, 'pf': 0,
                'total_pnl': 0, 'max_dd': 0, 'regime_gap': 0}
    
    df = pd.DataFrame(trades_list)
    daily_pnl = df.groupby('date')['net_pnl'].sum()
    n_trades = len(df)
    total_pnl = float(df['net_pnl'].sum())
    
    winners = (df['net_pnl'] > 0).sum()
    wr = float(winners / n_trades)
    
    gross_profit = float(df.loc[df['net_pnl'] > 0, 'net_pnl'].sum())
    gross_loss = float(abs(df.loc[df['net_pnl'] < 0, 'net_pnl'].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
    
    if len(daily_pnl) > 1 and daily_pnl.std() > 0:
        sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252))
    else:
        sharpe = 0.0
    
    downside = daily_pnl[daily_pnl < 0]
    if len(downside) > 1 and downside.std() > 0:
        sortino = float(daily_pnl.mean() / downside.std() * np.sqrt(252))
    else:
        sortino = 99.9 if daily_pnl.mean() > 0 else 0.0
    
    cumulative = daily_pnl.cumsum()
    max_dd = float((cumulative - cumulative.cummax()).min()) if len(cumulative) > 0 else 0
    
    day_conc = float(daily_pnl.max() / total_pnl) if total_pnl > 0 else 1.0
    
    avg_pnl = float(df['net_pnl'].mean())
    
    # Regime stratification
    regime_gap = 0.0
    regime_sharpes = {}
    if regime_map is not None:
        df_r = df.copy()
        df_r['regime'] = df_r['date'].map(lambda d: regime_map.get(d, 'unknown'))
        for regime in ['green', 'red', 'flat']:
            rt = df_r[df_r['regime'] == regime]
            if len(rt) > 0:
                rd = rt.groupby('date')['net_pnl'].sum()
                if len(rd) > 1 and rd.std() > 0:
                    regime_sharpes[regime] = float(rd.mean() / rd.std() * np.sqrt(252))
                else:
                    regime_sharpes[regime] = 0.0
            else:
                regime_sharpes[regime] = 0.0
        sg = regime_sharpes.get('green', 0)
        sr = regime_sharpes.get('red', 0)
        mx = max(abs(sg), abs(sr))
        regime_gap = abs(sg - sr) / mx if mx > 0 else 0
    
    # Direction breakdown
    long_t = df[df['direction'] == 1]
    short_t = df[df['direction'] == -1]
    
    return {
        'n_trades': int(n_trades),
        'n_days': int(len(daily_pnl)),
        'sharpe': round(sharpe, 3),
        'sortino': round(min(sortino, 99.9), 3),
        'wr': round(wr, 4),
        'pf': round(min(pf, 99.9), 3),
        'total_pnl_ticks': round(total_pnl, 2),
        'total_pnl_dollars': round(total_pnl * TICK_VALUE, 2),
        'avg_pnl_ticks': round(avg_pnl, 3),
        'max_dd_ticks': round(max_dd, 2),
        'day_conc': round(day_conc, 4),
        'regime_gap': round(regime_gap, 4),
        'regime_sharpes': regime_sharpes,
        'n_long': int(len(long_t)),
        'n_short': int(len(short_t)),
        'long_wr': round(float(long_t['net_pnl'].gt(0).mean()), 4) if len(long_t) > 0 else 0,
        'short_wr': round(float(short_t['net_pnl'].gt(0).mean()), 4) if len(short_t) > 0 else 0,
    }


# ══════════════════════════════════════════════════════════════════════
# ABLATION SCENARIOS
# ══════════════════════════════════════════════════════════════════════

def build_bar_index(preds, dates):
    """Build a mapping: date → list of (bar_idx_within_day, global_idx).
    Handle the duplicate-prediction issue (V1 uses bar_idx % unique_count)."""
    bar_index = defaultdict(list)
    date_counts = defaultdict(int)
    
    for i, (date, pred) in enumerate(zip(dates, preds)):
        bar_index[date].append(i)
        date_counts[date] += 1
    
    # Compute unique bar count per date (same logic as param_sensitivity_grid)
    unique_counts = {}
    for date, count in date_counts.items():
        if count <= 15:
            unique_counts[date] = count
        elif count == 28:
            unique_counts[date] = 14
        elif count == 21:
            unique_counts[date] = 7
        elif count == 16:
            unique_counts[date] = 8
        elif count == 18:
            unique_counts[date] = 9
        else:
            unique_counts[date] = min(count, 15)
    
    return bar_index, unique_counts


def run_ablation_scenario(scenario_name, preds, actuals, dates, confs,
                          bar_index, unique_counts, bar_cache,
                          flow_signals, midtrade_df, regime_map,
                          conf_thresh=None, zscore_thresh=None,
                          use_flow=False, use_midtrade=False,
                          tp=V1_TP, sl=V1_SL, mt_cut=V1_MT_CUT,
                          max_trades_per_day=5):
    """Run a single ablation scenario with minute-level replay."""
    trades = []
    filter_stats = {'total_bars': 0, 'pass_conf': 0, 'pass_zscore': 0,
                    'pass_flow': 0, 'pass_midtrade': 0, 'traded': 0,
                    'no_minute_data': 0}
    
    all_dates = sorted(bar_index.keys())
    
    # Pre-build midtrade model if needed (walk-forward)
    mt_model = None
    mt_cols = None
    mt_train_dates = set()
    mt_dates = sorted(midtrade_df['date'].unique()) if midtrade_df is not None else []
    
    # Midtrade model columns
    if midtrade_df is not None:
        exclude = ['date', 'direction', 'fill_price', 'time_of_day_min', 'exit_type',
                   'exit_ticks', 'winner', 'pred_magnitude']
        feature_cols_mt = [c for c in midtrade_df.columns if c not in exclude]
        mt_relevant_cols = [c for c in feature_cols_mt
                           if c.startswith('cp5s_') or c.startswith('cp10s_') or c.startswith('cp15s_')
                           or not any(c.startswith(f'cp{x}s_') for x in ['5','10','15','30'])]
    
    for date_idx, date_str in enumerate(all_dates):
        global_indices = bar_index[date_str]
        n_unique = unique_counts[date_str]
        
        # Only use unique bars (avoid duplicates)
        bar_preds_day = []
        for gi in global_indices[:n_unique]:
            bar_preds_day.append({
                'global_idx': gi,
                'pred': float(preds[gi]),
                'conf': float(confs[gi]),
                'actual': float(actuals[gi]),
            })
        
        filter_stats['total_bars'] += len(bar_preds_day)
        
        # Compute zscore within day
        if len(bar_preds_day) > 1:
            day_preds = np.array([b['pred'] for b in bar_preds_day])
            pred_mean = day_preds.mean()
            pred_std = day_preds.std() + 1e-8
            for b in bar_preds_day:
                b['zscore'] = (b['pred'] - pred_mean) / pred_std
        else:
            for b in bar_preds_day:
                b['zscore'] = 0.0
        
        # Get flow direction for this day
        flow_dir = None
        if use_flow and date_str in flow_signals:
            flow_dir = flow_signals[date_str]['direction']
            if flow_dir == 0:  # Flat day — skip entirely
                continue
        
        # Rebuild midtrade model periodically (walk-forward)
        if use_midtrade and midtrade_df is not None and date_idx % 10 == 0 and len(mt_train_dates) >= 15:
            train_mask = midtrade_df['date'].isin(mt_train_dates)
            if train_mask.sum() >= 20:
                X_train = np.nan_to_num(midtrade_df.loc[train_mask, mt_relevant_cols].values.astype(float), 0)
                y_train = midtrade_df.loc[train_mask, 'winner'].values
                mt_model = lgb.LGBMClassifier(
                    n_estimators=150, max_depth=4, learning_rate=0.05,
                    num_leaves=12, min_child_samples=8, subsample=0.8,
                    colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
                    verbose=-1, random_state=42
                )
                mt_model.fit(X_train, y_train)
                mt_cols = mt_relevant_cols
        
        day_trades = 0
        for bar_idx_in_day, bar in enumerate(bar_preds_day):
            if day_trades >= max_trades_per_day:
                break
            
            pred_val = bar['pred']
            conf_val = bar['conf']
            zscore_val = bar['zscore']
            
            # Determine trade direction from prediction
            if use_flow and flow_dir is not None:
                # Flow filter: only trade if prediction aligns with flow
                if flow_dir == 1 and zscore_val <= 0:
                    continue
                elif flow_dir == -1 and zscore_val >= 0:
                    continue
                direction = flow_dir
            else:
                # No flow filter: trade in prediction direction
                direction = 1 if pred_val > 0 else -1
            
            # Confidence filter
            if conf_thresh is not None:
                if conf_val < conf_thresh:
                    continue
            filter_stats['pass_conf'] += 1
            
            # Zscore filter
            if zscore_thresh is not None:
                if abs(zscore_val) < zscore_thresh:
                    continue
            filter_stats['pass_zscore'] += 1
            
            if use_flow:
                filter_stats['pass_flow'] += 1
            
            # Mid-trade filter
            midtrade_cut_flag = False
            if use_midtrade and mt_model is not None and midtrade_df is not None:
                mt_match = midtrade_df[
                    (midtrade_df['date'] == date_str) &
                    (midtrade_df['direction'] == direction)
                ]
                if len(mt_match) > 0:
                    mt_row = mt_match.iloc[min(bar_idx_in_day, len(mt_match)-1)]
                    X_mt = np.nan_to_num(mt_row[mt_cols].values.reshape(1, -1).astype(float), 0)
                    try:
                        mt_proba = mt_model.predict_proba(X_mt)[0]
                        winner_idx = np.where(mt_model.classes_ == 1)[0]
                        mt_score = float(mt_proba[winner_idx[0]]) if len(winner_idx) > 0 else float(mt_proba[-1])
                        if mt_score < mt_cut:
                            midtrade_cut_flag = True
                    except:
                        pass
                filter_stats['pass_midtrade'] += 1
            
            # Minute-level replay
            actual_bar_idx = bar_idx_in_day  # bar index within the day
            entry_price, next_minutes = get_next_bar_minutes(date_str, actual_bar_idx, bar_cache)
            if entry_price is None or next_minutes is None or len(next_minutes) == 0:
                filter_stats['no_minute_data'] += 1
                continue
            
            result = replay_trade(entry_price, direction, next_minutes, tp, sl)
            if result is None:
                continue
            
            # Apply midtrade cut effect (same as V1: reduce to 40% of raw PnL)
            if midtrade_cut_flag and result['exit_type'] != 'tp':
                result['pnl'] = result['pnl'] * 0.4
                result['exit_type'] = 'midtrade_cut'
            
            # Cost
            if result['exit_type'] in ('tp', 'sl'):
                cost = COST_MARKET_EXIT
            else:
                cost = COST_PASSIVE * 2  # passive both sides
            
            net_pnl = result['pnl'] - cost
            
            trades.append({
                'date': date_str,
                'direction': direction,
                'bar_idx': bar_idx_in_day,
                'conf': conf_val,
                'zscore': zscore_val,
                'exit_type': result['exit_type'],
                'raw_pnl': result['pnl'],
                'cost': cost,
                'net_pnl': net_pnl,
                'mfe': result['mfe'],
                'mae': result['mae'],
            })
            day_trades += 1
            filter_stats['traded'] += 1
        
        # Track midtrade training dates
        if date_str in set(mt_dates):
            mt_train_dates.add(date_str)
    
    metrics = compute_metrics(trades, regime_map)
    return trades, metrics, filter_stats


# ══════════════════════════════════════════════════════════════════════
# PARAMETER SWEEP ON CRITICAL FILTER
# ══════════════════════════════════════════════════════════════════════

def sweep_confidence(preds, actuals, dates, confs, bar_index, unique_counts,
                     bar_cache, regime_map, tp, sl):
    """Sweep confidence threshold from 0.3 to 0.8."""
    log.info("\n" + "=" * 70)
    log.info("CONFIDENCE THRESHOLD SWEEP")
    log.info("=" * 70)
    
    results = []
    for conf_val in np.arange(0.30, 0.82, 0.05):
        conf_val = round(conf_val, 2)
        trades, metrics, _ = run_ablation_scenario(
            f"conf_{conf_val}", preds, actuals, dates, confs,
            bar_index, unique_counts, bar_cache,
            flow_signals=None, midtrade_df=None, regime_map=regime_map,
            conf_thresh=conf_val, zscore_thresh=None,
            use_flow=False, use_midtrade=False,
            tp=tp, sl=sl)
        results.append({'conf': conf_val, **metrics})
        log.info(f"  conf>={conf_val:.2f}: N={metrics['n_trades']:>4}, "
                 f"Sharpe={metrics['sharpe']:>7.2f}, WR={metrics['wr']:.3f}, "
                 f"PF={metrics['pf']:>5.2f}, PnL={metrics['total_pnl_ticks']:>8.1f}t")
    return results


def sweep_zscore(preds, actuals, dates, confs, bar_index, unique_counts,
                 bar_cache, regime_map, tp, sl, conf_thresh):
    """Sweep zscore threshold from 0.1 to 2.0."""
    log.info("\n" + "=" * 70)
    log.info(f"ZSCORE THRESHOLD SWEEP (with conf>={conf_thresh})")
    log.info("=" * 70)
    
    results = []
    for zs_val in [0.1, 0.2, 0.3, 0.5, 0.7, 1.0, 1.2, 1.5, 2.0]:
        trades, metrics, _ = run_ablation_scenario(
            f"zscore_{zs_val}", preds, actuals, dates, confs,
            bar_index, unique_counts, bar_cache,
            flow_signals=None, midtrade_df=None, regime_map=regime_map,
            conf_thresh=conf_thresh, zscore_thresh=zs_val,
            use_flow=False, use_midtrade=False,
            tp=tp, sl=sl)
        results.append({'zscore': zs_val, **metrics})
        log.info(f"  |zscore|>={zs_val:.1f}: N={metrics['n_trades']:>4}, "
                 f"Sharpe={metrics['sharpe']:>7.2f}, WR={metrics['wr']:.3f}, "
                 f"PF={metrics['pf']:>5.2f}, PnL={metrics['total_pnl_ticks']:>8.1f}t")
    return results


def sweep_flow_with_conf(preds, actuals, dates, confs, bar_index, unique_counts,
                         bar_cache, flow_signals, regime_map, tp, sl):
    """Test flow filter with various confidence levels."""
    log.info("\n" + "=" * 70)
    log.info("FLOW DIRECTION + CONFIDENCE SWEEP")
    log.info("=" * 70)
    
    results = []
    for conf_val in [0.40, 0.45, 0.50, 0.52, 0.55, 0.58, 0.60, 0.65]:
        conf_val = round(conf_val, 2)
        for zs_val in [0.0, 0.3, 0.5, 1.0]:
            zs_t = zs_val if zs_val > 0 else None
            trades, metrics, _ = run_ablation_scenario(
                f"flow_conf{conf_val}_zs{zs_val}", preds, actuals, dates, confs,
                bar_index, unique_counts, bar_cache,
                flow_signals=flow_signals, midtrade_df=None, regime_map=regime_map,
                conf_thresh=conf_val, zscore_thresh=zs_t,
                use_flow=True, use_midtrade=False,
                tp=tp, sl=sl)
            results.append({'conf': conf_val, 'zscore': zs_val, **metrics})
            log.info(f"  flow+conf>={conf_val:.2f}+|zs|>={zs_val:.1f}: "
                     f"N={metrics['n_trades']:>4}, Sharpe={metrics['sharpe']:>7.2f}, "
                     f"WR={metrics['wr']:.3f}, PF={metrics['pf']:>5.2f}")
    return results


def sweep_tp_sl(preds, actuals, dates, confs, bar_index, unique_counts,
                bar_cache, flow_signals, regime_map, conf_thresh, zscore_thresh,
                use_flow):
    """SL/TP grid with the best filter combo."""
    log.info("\n" + "=" * 70)
    log.info("SL/TP GRID WITH BEST FILTER COMBO")
    log.info("=" * 70)
    
    results = []
    sl_grid = [3, 5, 8, 10, 12, 15, 20]
    tp_grid = [6, 8, 10, 12, 15, 20, 25, 30]
    
    print(f"{'SL/TP':>8}", end='')
    for tp in tp_grid:
        print(f"  TP{tp:>3}", end='')
    print("")
    
    for sl in sl_grid:
        line = f"  SL{sl:>3}"
        for tp in tp_grid:
            trades, metrics, _ = run_ablation_scenario(
                f"sl{sl}_tp{tp}", preds, actuals, dates, confs,
                bar_index, unique_counts, bar_cache,
                flow_signals=flow_signals, midtrade_df=None, regime_map=regime_map,
                conf_thresh=conf_thresh, zscore_thresh=zscore_thresh,
                use_flow=use_flow, use_midtrade=False,
                tp=tp, sl=sl)
            results.append({'sl': sl, 'tp': tp, **metrics})
            line += f"  {metrics['sharpe']:>5.1f}"
        log.info(line)
    return results


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("FILTER ABLATION STUDY")
    log.info("Which filter drives the gap from Sharpe 0.25 → Sharpe 20+?")
    log.info("=" * 70)
    
    # Load data
    preds, actuals, dates, confs = load_predictions()
    flow_signals = load_flow_signals()
    bar_cache = load_bar_cache()
    midtrade_df = load_midtrade_data()
    regime_map = load_regime_map()
    
    bar_index, unique_counts = build_bar_index(preds, dates)
    
    # Overlap dates (prediction dates that also have minute bars)
    pred_dates = sorted(bar_index.keys())
    overlap_dates = sorted(set(pred_dates) & set(bar_cache.keys()))
    log.info(f"\nPrediction dates: {len(pred_dates)}")
    log.info(f"Minute bar dates: {len(bar_cache)}")
    log.info(f"Overlap dates: {len(overlap_dates)}")
    
    # Filter bar_index to only overlap dates
    bar_index_filtered = {d: bar_index[d] for d in overlap_dates}
    unique_counts_filtered = {d: unique_counts[d] for d in overlap_dates}
    
    total_bars = sum(unique_counts_filtered[d] for d in overlap_dates)
    log.info(f"Total unique prediction bars with minute data: {total_bars}")
    
    # MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("filter_ablation_study")
        mlflow.start_run(run_name=f"ablation_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    tp = V1_TP  # 12
    sl = V1_SL  # 3
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 2: PROGRESSIVE ABLATION
    # ═══════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 2: PROGRESSIVE FILTER ABLATION (SL=%d, TP=%d)", sl, tp)
    log.info("=" * 70)
    
    ablation_results = {}
    
    # A: No filters — every bar, trade in predicted direction
    log.info("\n--- A: NO FILTERS (every prediction bar) ---")
    trades_a, metrics_a, stats_a = run_ablation_scenario(
        "A_no_filters", preds, actuals, dates, confs,
        bar_index_filtered, unique_counts_filtered, bar_cache,
        flow_signals=None, midtrade_df=None, regime_map=regime_map,
        conf_thresh=None, zscore_thresh=None,
        use_flow=False, use_midtrade=False, tp=tp, sl=sl)
    ablation_results['A_no_filters'] = {'metrics': metrics_a, 'filter_stats': stats_a}
    log.info(f"  Result: N={metrics_a['n_trades']}, Sharpe={metrics_a['sharpe']:.3f}, "
             f"WR={metrics_a['wr']:.3f}, PF={metrics_a['pf']:.3f}, "
             f"PnL={metrics_a['total_pnl_ticks']:.1f}t, regime_gap={metrics_a['regime_gap']:.3f}")
    log.info(f"  Filter stats: {stats_a}")
    
    # B: Confidence only
    log.info("\n--- B: CONFIDENCE ONLY (conf >= 0.52) ---")
    trades_b, metrics_b, stats_b = run_ablation_scenario(
        "B_conf_only", preds, actuals, dates, confs,
        bar_index_filtered, unique_counts_filtered, bar_cache,
        flow_signals=None, midtrade_df=None, regime_map=regime_map,
        conf_thresh=V1_CONF, zscore_thresh=None,
        use_flow=False, use_midtrade=False, tp=tp, sl=sl)
    ablation_results['B_conf_only'] = {'metrics': metrics_b, 'filter_stats': stats_b}
    log.info(f"  Result: N={metrics_b['n_trades']}, Sharpe={metrics_b['sharpe']:.3f}, "
             f"WR={metrics_b['wr']:.3f}, PF={metrics_b['pf']:.3f}, "
             f"PnL={metrics_b['total_pnl_ticks']:.1f}t, regime_gap={metrics_b['regime_gap']:.3f}")
    
    # C: Confidence + zscore
    log.info("\n--- C: CONFIDENCE + ZSCORE (conf >= 0.52, |zscore| >= 0.3) ---")
    trades_c, metrics_c, stats_c = run_ablation_scenario(
        "C_conf_zscore", preds, actuals, dates, confs,
        bar_index_filtered, unique_counts_filtered, bar_cache,
        flow_signals=None, midtrade_df=None, regime_map=regime_map,
        conf_thresh=V1_CONF, zscore_thresh=V1_ZSCORE,
        use_flow=False, use_midtrade=False, tp=tp, sl=sl)
    ablation_results['C_conf_zscore'] = {'metrics': metrics_c, 'filter_stats': stats_c}
    log.info(f"  Result: N={metrics_c['n_trades']}, Sharpe={metrics_c['sharpe']:.3f}, "
             f"WR={metrics_c['wr']:.3f}, PF={metrics_c['pf']:.3f}, "
             f"PnL={metrics_c['total_pnl_ticks']:.1f}t, regime_gap={metrics_c['regime_gap']:.3f}")
    
    # D: Confidence + zscore + flow direction
    log.info("\n--- D: CONF + ZSCORE + FLOW DIRECTION ---")
    trades_d, metrics_d, stats_d = run_ablation_scenario(
        "D_conf_zscore_flow", preds, actuals, dates, confs,
        bar_index_filtered, unique_counts_filtered, bar_cache,
        flow_signals=flow_signals, midtrade_df=None, regime_map=regime_map,
        conf_thresh=V1_CONF, zscore_thresh=V1_ZSCORE,
        use_flow=True, use_midtrade=False, tp=tp, sl=sl)
    ablation_results['D_conf_zscore_flow'] = {'metrics': metrics_d, 'filter_stats': stats_d}
    log.info(f"  Result: N={metrics_d['n_trades']}, Sharpe={metrics_d['sharpe']:.3f}, "
             f"WR={metrics_d['wr']:.3f}, PF={metrics_d['pf']:.3f}, "
             f"PnL={metrics_d['total_pnl_ticks']:.1f}t, regime_gap={metrics_d['regime_gap']:.3f}")
    
    # E: Full V1 (all 3 layers)
    log.info("\n--- E: FULL V1 PIPELINE (all 3 layers) ---")
    trades_e, metrics_e, stats_e = run_ablation_scenario(
        "E_full_v1", preds, actuals, dates, confs,
        bar_index_filtered, unique_counts_filtered, bar_cache,
        flow_signals=flow_signals, midtrade_df=midtrade_df, regime_map=regime_map,
        conf_thresh=V1_CONF, zscore_thresh=V1_ZSCORE,
        use_flow=True, use_midtrade=True, tp=tp, sl=sl, mt_cut=V1_MT_CUT)
    ablation_results['E_full_v1'] = {'metrics': metrics_e, 'filter_stats': stats_e}
    log.info(f"  Result: N={metrics_e['n_trades']}, Sharpe={metrics_e['sharpe']:.3f}, "
             f"WR={metrics_e['wr']:.3f}, PF={metrics_e['pf']:.3f}, "
             f"PnL={metrics_e['total_pnl_ticks']:.1f}t, regime_gap={metrics_e['regime_gap']:.3f}")
    
    # ═══════════════════════════════════════════════════════════════
    # ABLATION SUMMARY
    # ═══════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("ABLATION SUMMARY (SL=%d, TP=%d)", sl, tp)
    log.info("=" * 70)
    log.info(f"{'Scenario':<35} {'N':>5} {'Sharpe':>8} {'WR':>6} {'PF':>6} {'PnL(t)':>9} {'RegGap':>7}")
    log.info("-" * 80)
    
    scenario_order = [
        ('A_no_filters', 'A: No filters'),
        ('B_conf_only', 'B: Confidence only (>=0.52)'),
        ('C_conf_zscore', 'C: Conf + Zscore'),
        ('D_conf_zscore_flow', 'D: Conf + Zscore + Flow'),
        ('E_full_v1', 'E: Full V1 (all layers)'),
    ]
    
    prev_sharpe = None
    for key, label in scenario_order:
        m = ablation_results[key]['metrics']
        delta = ""
        if prev_sharpe is not None:
            d = m['sharpe'] - prev_sharpe
            delta = f" (Δ={d:+.2f})"
        log.info(f"{label:<35} {m['n_trades']:>5} {m['sharpe']:>8.3f} "
                 f"{m['wr']:>6.3f} {m['pf']:>6.2f} {m['total_pnl_ticks']:>9.1f} "
                 f"{m['regime_gap']:>7.3f}{delta}")
        prev_sharpe = m['sharpe']
    
    # Identify the biggest jump
    sharpes = [(key, ablation_results[key]['metrics']['sharpe']) for key, _ in scenario_order]
    max_delta = 0
    max_delta_from = ""
    max_delta_to = ""
    for i in range(1, len(sharpes)):
        delta = sharpes[i][1] - sharpes[i-1][1]
        if delta > max_delta:
            max_delta = delta
            max_delta_from = sharpes[i-1][0]
            max_delta_to = sharpes[i][0]
    
    log.info(f"\nBIGGEST SHARPE JUMP: {max_delta_from} → {max_delta_to} (Δ = {max_delta:.3f})")
    log.info(f"This filter is doing MOST of the work.")
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3: OPTIMIZE CRITICAL FILTER
    # ═══════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 3: PARAMETER SWEEPS ON INDIVIDUAL FILTERS")
    log.info("=" * 70)
    
    conf_sweep = sweep_confidence(preds, actuals, dates, confs,
                                  bar_index_filtered, unique_counts_filtered,
                                  bar_cache, regime_map, tp, sl)
    
    zscore_sweep = sweep_zscore(preds, actuals, dates, confs,
                                bar_index_filtered, unique_counts_filtered,
                                bar_cache, regime_map, tp, sl, V1_CONF)
    
    flow_conf_sweep = sweep_flow_with_conf(preds, actuals, dates, confs,
                                           bar_index_filtered, unique_counts_filtered,
                                           bar_cache, flow_signals, regime_map, tp, sl)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 3b: SL/TP grid with best filter combo found
    # ═══════════════════════════════════════════════════════════════
    
    # Find best conf-only threshold
    best_conf_row = max(conf_sweep, key=lambda x: x['sharpe'])
    best_conf = best_conf_row['conf']
    log.info(f"\nBest conf-only threshold: {best_conf} (Sharpe={best_conf_row['sharpe']:.3f})")
    
    # Find best flow+conf combo
    best_flow_row = max(flow_conf_sweep, key=lambda x: x['sharpe'])
    best_flow_conf = best_flow_row['conf']
    best_flow_zs = best_flow_row['zscore']
    log.info(f"Best flow+conf combo: conf={best_flow_conf}, zscore={best_flow_zs} "
             f"(Sharpe={best_flow_row['sharpe']:.3f})")
    
    # SL/TP grid with flow + best conf
    sl_tp_results = sweep_tp_sl(preds, actuals, dates, confs,
                                bar_index_filtered, unique_counts_filtered,
                                bar_cache, flow_signals, regime_map,
                                best_flow_conf, best_flow_zs if best_flow_zs > 0 else None,
                                use_flow=True)
    
    # ═══════════════════════════════════════════════════════════════
    # STEP 4: MINIMAL VIABLE FILTER
    # ═══════════════════════════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("STEP 4: MINIMAL VIABLE FILTER SEARCH")
    log.info("=" * 70)
    
    # Test: is flow alone enough? (no conf/zscore filter, just flow direction)
    log.info("\n--- Flow direction alone (no entry filters) ---")
    trades_flow_only, metrics_flow_only, _ = run_ablation_scenario(
        "flow_only", preds, actuals, dates, confs,
        bar_index_filtered, unique_counts_filtered, bar_cache,
        flow_signals=flow_signals, midtrade_df=None, regime_map=regime_map,
        conf_thresh=None, zscore_thresh=None,
        use_flow=True, use_midtrade=False, tp=tp, sl=sl)
    log.info(f"  Flow only: N={metrics_flow_only['n_trades']}, "
             f"Sharpe={metrics_flow_only['sharpe']:.3f}, WR={metrics_flow_only['wr']:.3f}")
    
    # Test: conf alone at different thresholds + SL3/TP12
    log.info("\n--- Minimal combos (no flow, various conf, SL3/TP12) ---")
    minimal_results = []
    for c in [0.50, 0.52, 0.55, 0.58, 0.60, 0.65, 0.70]:
        trades_m, metrics_m, _ = run_ablation_scenario(
            f"minimal_conf{c}", preds, actuals, dates, confs,
            bar_index_filtered, unique_counts_filtered, bar_cache,
            flow_signals=None, midtrade_df=None, regime_map=regime_map,
            conf_thresh=c, zscore_thresh=None,
            use_flow=False, use_midtrade=False, tp=tp, sl=sl)
        minimal_results.append({'conf': c, **metrics_m})
        log.info(f"  conf>={c:.2f}: N={metrics_m['n_trades']:>4}, Sharpe={metrics_m['sharpe']:>7.2f}, "
                 f"WR={metrics_m['wr']:.3f}, PF={metrics_m['pf']:.3f}, regime_gap={metrics_m['regime_gap']:.3f}")
    
    # Also test with SL10/TP20 (the config mentioned in the problem statement)
    log.info("\n--- Also testing with SL10/TP20 for comparison ---")
    log.info(f"{'Scenario':<35} {'N':>5} {'Sharpe':>8} {'WR':>6} {'PF':>6} {'PnL(t)':>9}")
    log.info("-" * 70)
    
    for key, label, kwargs in [
        ('A_sl10tp20', 'A: No filters (SL10/TP20)', 
         dict(conf_thresh=None, zscore_thresh=None, use_flow=False, use_midtrade=False)),
        ('D_sl10tp20', 'D: Conf+Zs+Flow (SL10/TP20)',
         dict(conf_thresh=V1_CONF, zscore_thresh=V1_ZSCORE, use_flow=True, use_midtrade=False)),
    ]:
        tr, met, _ = run_ablation_scenario(
            key, preds, actuals, dates, confs,
            bar_index_filtered, unique_counts_filtered, bar_cache,
            flow_signals=flow_signals, midtrade_df=midtrade_df, regime_map=regime_map,
            tp=20, sl=10, **kwargs)
        log.info(f"{label:<35} {met['n_trades']:>5} {met['sharpe']:>8.3f} "
                 f"{met['wr']:>6.3f} {met['pf']:>6.2f} {met['total_pnl_ticks']:>9.1f}")
    
    # ═══════════════════════════════════════════════════════════════
    # SAVE & LOG
    # ═══════════════════════════════════════════════════════════════
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': round(time.time() - t0, 1),
        'data_summary': {
            'total_prediction_bars': int(len(preds)),
            'unique_dates': int(len(np.unique(dates))),
            'overlap_dates_with_minute_data': len(overlap_dates),
            'total_unique_bars_with_minute_data': total_bars,
        },
        'v1_config': {
            'entry_conf': V1_CONF, 'zscore_thresh': V1_ZSCORE,
            'tp': V1_TP, 'sl': V1_SL, 'mt_cut': V1_MT_CUT,
        },
        'ablation_results': {k: v['metrics'] for k, v in ablation_results.items()},
        'biggest_jump': {
            'from': max_delta_from, 'to': max_delta_to,
            'delta_sharpe': round(max_delta, 3),
        },
        'conf_sweep': conf_sweep,
        'zscore_sweep': zscore_sweep,
        'flow_conf_sweep': flow_conf_sweep,
        'sl_tp_grid': sl_tp_results,
        'minimal_combos': minimal_results,
    }
    
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_,)): return bool(obj)
        return str(obj)
    
    out_file = OUTPUT_DIR / "ablation_results.json"
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=convert)
    log.info(f"\nSaved results to {out_file}")
    
    if MLFLOW_AVAILABLE:
        # Log key metrics
        for key, data in ablation_results.items():
            m = data['metrics']
            mlflow.log_metric(f"{key}_sharpe", m['sharpe'])
            mlflow.log_metric(f"{key}_n_trades", m['n_trades'])
            mlflow.log_metric(f"{key}_wr", m['wr'])
            mlflow.log_metric(f"{key}_pf", m['pf'])
            mlflow.log_metric(f"{key}_regime_gap", m['regime_gap'])
        mlflow.log_metric("biggest_jump_delta", max_delta)
        mlflow.log_param("biggest_jump_filter", f"{max_delta_from}->{max_delta_to}")
        mlflow.log_artifact(str(out_file))
        mlflow.log_artifact(str(LOG_FILE))
        mlflow.end_run()
        log.info("MLflow logged")
    
    log.info(f"\nElapsed: {time.time() - t0:.1f}s")
    log.info("DONE")


if __name__ == '__main__':
    main()
