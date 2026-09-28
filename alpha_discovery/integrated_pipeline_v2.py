#!/usr/bin/env python3
"""
integrated_pipeline_v2.py — Exit Structure Research

The integrated_pipeline_v1 used TP=12/SL=3 tick stops on 30-min bars.
Tick-level replay proved SL=3 is hit on 94% of trades within 1 minute.
The SIGNAL has 54% directional accuracy — the problem is the exit structure.

Tests 3 exit approaches on the SAME 460 trade entries:
  A: Bar-Close Exit — hold for full 30-min bar, match prediction horizon
  B: Wider Stops — SL/TP scaled to ES noise levels
  C: Mid-Trade Classifier Exit — MLP confidence to decide hold/exit

HC #432: exits must match model prediction horizon
HC #649: prefer dynamic (classifier) over static (fixed stops)
HC #74:  FIFO cost model only
"""

import os, sys, json, time, logging, warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Paths ──
LVL3 = Path("/home/nick/Lvl3Quant")
TRADES_FILE = LVL3 / "output/integrated_pipeline_v1/best_trades.parquet"
MINUTE_BARS_DIR = LVL3 / "data/processed/mbo_minute_bars_v1"
ENTRY_PREDS = LVL3 / "output/lh_30min_deep_v1/concat_oot.npz"
FLOW_FEATURES = LVL3 / "output/long_horizon_flow_v2/enhanced_daily_features.parquet"
MIDTRADE_FEATURES = LVL3 / "output/midtrade_thesis_v1/trade_tick_features.parquet"
OUTPUT_DIR = LVL3 / "output/integrated_pipeline_v2"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0
COST_PASSIVE = COMMISSION_RT_TICKS          # passive limit both sides
COST_MARKET_EXIT = COMMISSION_RT_TICKS + SPREAD_TICKS  # passive entry + market exit (HC: entry passive, exit market if stop/classifier)

MLFLOW_URI = "http://localhost:5000"

LOG_FILE = OUTPUT_DIR / "pipeline_v2.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════
# SHARED UTILITIES
# ═══════════════════════════════════════════════════════════════

def get_unique_bar_count(date_str, pred_dates):
    """Figure out how many UNIQUE bars this date has in predictions."""
    mask = pred_dates == date_str
    n_total = mask.sum()
    if n_total <= 15:
        return n_total
    if n_total == 28: return 14
    elif n_total == 21: return 7
    elif n_total == 16: return 8
    elif n_total == 18: return 9
    else: return min(n_total, 15)


def build_30min_bars_and_minutes(date_str):
    """Build 30-min bars from minute data."""
    fpath = MINUTE_BARS_DIR / f"{date_str}.parquet"
    if not fpath.exists():
        return None, None
    
    mbars = pd.read_parquet(fpath)
    mbars['ts_minute'] = pd.to_datetime(mbars['ts_minute'], utc=True)
    mbars = mbars.sort_values('ts_minute').reset_index(drop=True)
    mbars['bar_key'] = mbars['ts_minute'].dt.floor('30min')
    
    bars_30 = []
    for bar_key, grp in mbars.groupby('bar_key'):
        if len(grp) < 3:
            continue
        bars_30.append({
            'bar_key': bar_key,
            'open': grp['open'].iloc[0],
            'high': grp['high'].max(),
            'low': grp['low'].min(),
            'close': grp['close'].iloc[-1],
            'volume': grp['volume'].sum(),
        })
    
    bars_30_df = pd.DataFrame(bars_30).sort_values('bar_key').reset_index(drop=True)
    return bars_30_df, mbars


def compute_metrics(trades_df, label=""):
    """Compute Sharpe, Sortino, PF, WR, max DD."""
    if len(trades_df) == 0:
        return {'label': label, 'n_trades': 0, 'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0}
    
    daily_pnl = trades_df.groupby('date')['net_pnl_ticks'].sum()
    n_trades = len(trades_df)
    total_pnl = float(trades_df['net_pnl_ticks'].sum())
    
    winners = (trades_df['net_pnl_ticks'] > 0).sum()
    wr = float(winners / n_trades)
    
    gross_profit = float(trades_df.loc[trades_df['net_pnl_ticks'] > 0, 'net_pnl_ticks'].sum())
    gross_loss = float(abs(trades_df.loc[trades_df['net_pnl_ticks'] < 0, 'net_pnl_ticks'].sum()))
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
    
    sharpe = float(daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)) if len(daily_pnl) > 1 and daily_pnl.std() > 0 else 0.0
    
    downside = daily_pnl[daily_pnl < 0]
    sortino = float(daily_pnl.mean() / downside.std() * np.sqrt(252)) if len(downside) > 1 and downside.std() > 0 else (99.9 if daily_pnl.mean() > 0 else 0.0)
    
    cumulative = daily_pnl.cumsum()
    max_dd = float((cumulative - cumulative.cummax()).min())
    
    day_conc = float(daily_pnl.max() / total_pnl) if total_pnl > 0 else 1.0
    
    return {
        'label': label,
        'n_trades': int(n_trades),
        'n_trading_days': int(len(daily_pnl)),
        'total_pnl_ticks': total_pnl,
        'total_pnl_dollars': total_pnl * TICK_VALUE,
        'avg_pnl_ticks': float(trades_df['net_pnl_ticks'].mean()),
        'win_rate': wr,
        'profit_factor': min(float(pf), 99.9),
        'sharpe': sharpe,
        'sortino': min(float(sortino), 99.9),
        'max_dd_ticks': max_dd,
        'max_dd_dollars': max_dd * TICK_VALUE,
        'day_concentration': day_conc,
        'day_conc_pass': bool(day_conc <= 0.70),
    }


def regime_stratify(trades_df):
    """Regime stratification with gap check (HC #428)."""
    flow_df = pd.read_parquet(FLOW_FEATURES)
    regime_map = {}
    for _, row in flow_df.iterrows():
        date_val = row['date']
        date_str = date_val.strftime('%Y%m%d') if hasattr(date_val, 'strftime') else str(date_val).replace('-', '')[:8]
        cc = row.get('cc_return_ticks', 0)
        if pd.isna(cc): regime_map[date_str] = 'flat'
        elif cc > 20: regime_map[date_str] = 'green'
        elif cc < -20: regime_map[date_str] = 'red'
        else: regime_map[date_str] = 'flat'
    
    trades_df = trades_df.copy()
    trades_df['regime'] = trades_df['date'].map(lambda d: regime_map.get(d, 'unknown'))
    
    results = {}
    for regime in ['green', 'red', 'flat']:
        rt = trades_df[trades_df['regime'] == regime]
        results[regime] = compute_metrics(rt, regime) if len(rt) > 0 else {'n_trades': 0, 'sharpe': 0}
    
    sg = results.get('green', {}).get('sharpe', 0)
    sr = results.get('red', {}).get('sharpe', 0)
    mx = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / mx if mx > 0 else 0
    results['regime_gap'] = float(gap)
    results['regime_pass'] = bool(gap < 0.50)
    return results


def get_next_bar_minutes(date_str, actual_bar_idx, bar_cache):
    """Get minute bars for the next 30-min period after a trade entry."""
    if date_str not in bar_cache:
        bars_30, mbars = build_30min_bars_and_minutes(date_str)
        bar_cache[date_str] = (bars_30, mbars)
    bars_30, mbars = bar_cache[date_str]
    
    if bars_30 is None or actual_bar_idx >= len(bars_30):
        return None, None, None
    
    entry_price = bars_30.iloc[actual_bar_idx]['close']
    bar_key = bars_30.iloc[actual_bar_idx]['bar_key']
    
    next_start = bar_key + pd.Timedelta(minutes=30)
    next_end = bar_key + pd.Timedelta(minutes=59)
    mbars_ts = mbars.set_index('ts_minute')
    next_minutes = mbars_ts.loc[
        (mbars_ts.index >= next_start) & (mbars_ts.index < next_end + pd.Timedelta(minutes=1))
    ]
    
    return entry_price, next_minutes, bar_key


# ═══════════════════════════════════════════════════════════════
# APPROACH A: BAR-CLOSE EXIT (match prediction horizon)
# ═══════════════════════════════════════════════════════════════

def approach_a_bar_close(trades_orig, pred_dates, bar_cache):
    """
    Hold for full 30-min bar. Exit at bar close.
    Model predicts 30-min return → hold for 30 min. HC #432 compliant.
    Entry: passive limit at bar close. Exit: passive limit at next bar close.
    Cost: 0.376 ticks each side = 0.752 ticks total.
    """
    log.info("\n" + "=" * 70)
    log.info("APPROACH A: BAR-CLOSE EXIT (match prediction horizon)")
    log.info("=" * 70)
    
    unique_bar_counts = {}
    results = []
    
    for _, trade in trades_orig.iterrows():
        date_str = trade['date']
        bar_idx = trade['bar_idx']
        direction = trade['direction']
        
        if date_str not in unique_bar_counts:
            unique_bar_counts[date_str] = get_unique_bar_count(date_str, pred_dates)
        actual_bar_idx = bar_idx % unique_bar_counts[date_str]
        
        entry_price, next_minutes, bar_key = get_next_bar_minutes(date_str, actual_bar_idx, bar_cache)
        
        if entry_price is None or next_minutes is None or len(next_minutes) == 0:
            continue
        
        # Exit at end of next 30-min bar
        exit_price = next_minutes.iloc[-1]['close']
        
        if direction == 1:
            raw_pnl = (exit_price - entry_price) / TICK_SIZE
        else:
            raw_pnl = (entry_price - exit_price) / TICK_SIZE
        
        # MFE/MAE through the bar
        mfe = 0.0
        mae = 0.0
        for _, row in next_minutes.iterrows():
            if direction == 1:
                mfe = max(mfe, (row['high'] - entry_price) / TICK_SIZE)
                mae = max(mae, (entry_price - row['low']) / TICK_SIZE)
            else:
                mfe = max(mfe, (entry_price - row['low']) / TICK_SIZE)
                mae = max(mae, (row['high'] - entry_price) / TICK_SIZE)
        
        # Cost: passive entry + passive exit (both at bar boundaries)
        cost = COST_PASSIVE * 2  # 0.376 * 2 = 0.752 ticks
        net_pnl = raw_pnl - cost
        
        results.append({
            'date': date_str,
            'direction': direction,
            'bar_idx': bar_idx,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'raw_pnl_ticks': raw_pnl,
            'cost_ticks': cost,
            'net_pnl_ticks': net_pnl,
            'mfe_ticks': mfe,
            'mae_ticks': mae,
            'minutes_held': len(next_minutes),
            'exit_type': 'bar_close',
        })
    
    df = pd.DataFrame(results)
    metrics = compute_metrics(df, "A: Bar-Close")
    regime = regime_stratify(df)
    
    log.info(f"N trades: {metrics['n_trades']}")
    log.info(f"Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}")
    log.info(f"WR: {metrics['win_rate']:.1%}, PF: {metrics['profit_factor']:.2f}")
    log.info(f"Total PnL: {metrics['total_pnl_ticks']:.1f} ticks (${metrics['total_pnl_dollars']:.0f})")
    log.info(f"Max DD: {metrics['max_dd_ticks']:.1f} ticks (${metrics['max_dd_dollars']:.0f})")
    log.info(f"Day concentration: {metrics['day_concentration']:.3f} (pass={metrics['day_conc_pass']})")
    log.info(f"Regime gap: {regime['regime_gap']:.3f} (pass={regime['regime_pass']})")
    for r in ['green', 'red', 'flat']:
        rd = regime.get(r, {})
        log.info(f"  {r}: N={rd.get('n_trades', 0)}, Sharpe={rd.get('sharpe', 0):.2f}, "
                 f"WR={rd.get('win_rate', 0):.1%}, PF={rd.get('profit_factor', 0):.2f}")
    
    # MFE analysis
    mfe_arr = df['mfe_ticks']
    mae_arr = df['mae_ticks']
    log.info(f"\nMFE: mean={mfe_arr.mean():.1f}t, p50={mfe_arr.median():.1f}t, p90={mfe_arr.quantile(0.9):.1f}t")
    log.info(f"MAE: mean={mae_arr.mean():.1f}t, p50={mae_arr.median():.1f}t, p90={mae_arr.quantile(0.9):.1f}t")
    
    return df, metrics, regime


# ═══════════════════════════════════════════════════════════════
# APPROACH B: WIDER STOPS (noise-aligned)
# ═══════════════════════════════════════════════════════════════

def replay_trade_wider(entry_price, direction, minute_bars_next, tp_ticks, sl_ticks):
    """Replay a trade through minute-level data with wider stops."""
    if len(minute_bars_next) == 0:
        return None
    
    mfe = 0.0
    mae = 0.0
    
    for i, (idx, row) in enumerate(minute_bars_next.iterrows()):
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
            # Both in same minute — use distance ratio
            # With wider stops this is rarer, but handle it
            remaining_tp = max(tp_ticks - mfe, 0.1)
            remaining_sl = max(sl_ticks - mae, 0.1)
            p_sl_first = remaining_tp / (remaining_tp + remaining_sl)
            if p_sl_first > 0.5:
                return {
                    'exit_type': 'sl', 'exit_pnl_ticks': -sl_ticks,
                    'minutes_held': i + 1, 'mfe_ticks': mfe, 'mae_ticks': mae,
                }
            else:
                return {
                    'exit_type': 'tp', 'exit_pnl_ticks': tp_ticks,
                    'minutes_held': i + 1, 'mfe_ticks': mfe, 'mae_ticks': mae,
                }
        elif sl_hit:
            return {
                'exit_type': 'sl', 'exit_pnl_ticks': -sl_ticks,
                'minutes_held': i + 1, 'mfe_ticks': mfe, 'mae_ticks': mae,
            }
        elif tp_hit:
            return {
                'exit_type': 'tp', 'exit_pnl_ticks': tp_ticks,
                'minutes_held': i + 1, 'mfe_ticks': mfe, 'mae_ticks': mae,
            }
    
    # Neither hit — exit at bar close
    exit_price = minute_bars_next.iloc[-1]['close']
    if direction == 1:
        exit_pnl = (exit_price - entry_price) / TICK_SIZE
    else:
        exit_pnl = (entry_price - exit_price) / TICK_SIZE
    
    return {
        'exit_type': 'bar_close', 'exit_pnl_ticks': exit_pnl,
        'minutes_held': len(minute_bars_next), 'mfe_ticks': mfe, 'mae_ticks': mae,
    }


def approach_b_wider_stops(trades_orig, pred_dates, bar_cache):
    """
    Test wider SL/TP combinations aligned to ES noise.
    SL = 15/25/35 ticks, TP = 2x SL.
    """
    log.info("\n" + "=" * 70)
    log.info("APPROACH B: WIDER STOPS (noise-aligned)")
    log.info("=" * 70)
    
    configs = [
        {'sl': 15, 'tp': 30, 'label': 'SL15/TP30'},
        {'sl': 25, 'tp': 50, 'label': 'SL25/TP50'},
        {'sl': 35, 'tp': 70, 'label': 'SL35/TP70'},
    ]
    
    unique_bar_counts = {}
    all_config_results = {}
    
    for cfg in configs:
        sl = cfg['sl']
        tp = cfg['tp']
        label = cfg['label']
        
        log.info(f"\n--- {label} ---")
        results = []
        
        for _, trade in trades_orig.iterrows():
            date_str = trade['date']
            bar_idx = trade['bar_idx']
            direction = trade['direction']
            
            if date_str not in unique_bar_counts:
                unique_bar_counts[date_str] = get_unique_bar_count(date_str, pred_dates)
            actual_bar_idx = bar_idx % unique_bar_counts[date_str]
            
            entry_price, next_minutes, bar_key = get_next_bar_minutes(date_str, actual_bar_idx, bar_cache)
            
            if entry_price is None or next_minutes is None or len(next_minutes) == 0:
                continue
            
            r = replay_trade_wider(entry_price, direction, next_minutes, tp, sl)
            if r is None:
                continue
            
            # Cost: passive entry. Exit: market if stop hit, passive if bar close
            if r['exit_type'] in ('tp', 'sl'):
                cost = COST_PASSIVE + (COMMISSION_RT_TICKS / 2 + SPREAD_TICKS / 2)
                # Actually: entry passive (0.376/2 = 0.188) + exit market (0.188 + 1.0 = 1.188)
                # But FIFO canonical: passive entry = 0.376 full RT cost split
                # Simpler: passive entry commission + market exit commission + spread
                cost = COST_MARKET_EXIT  # 0.376 + 1.0 = 1.376 ticks
            else:
                cost = COST_PASSIVE * 2  # Both sides passive at bar boundary = 0.752
            
            net_pnl = r['exit_pnl_ticks'] - cost
            
            results.append({
                'date': date_str,
                'direction': direction,
                'bar_idx': bar_idx,
                'entry_price': entry_price,
                'raw_pnl_ticks': r['exit_pnl_ticks'],
                'cost_ticks': cost,
                'net_pnl_ticks': net_pnl,
                'exit_type': r['exit_type'],
                'mfe_ticks': r['mfe_ticks'],
                'mae_ticks': r['mae_ticks'],
                'minutes_held': r['minutes_held'],
            })
        
        df = pd.DataFrame(results)
        metrics = compute_metrics(df, f"B: {label}")
        regime = regime_stratify(df)
        
        log.info(f"N trades: {metrics['n_trades']}")
        log.info(f"Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}")
        log.info(f"WR: {metrics['win_rate']:.1%}, PF: {metrics['profit_factor']:.2f}")
        log.info(f"Total PnL: {metrics['total_pnl_ticks']:.1f} ticks (${metrics['total_pnl_dollars']:.0f})")
        log.info(f"Max DD: {metrics['max_dd_ticks']:.1f} ticks (${metrics['max_dd_dollars']:.0f})")
        log.info(f"Day concentration: {metrics['day_concentration']:.3f} (pass={metrics['day_conc_pass']})")
        log.info(f"Regime gap: {regime['regime_gap']:.3f} (pass={regime['regime_pass']})")
        
        exit_dist = df['exit_type'].value_counts().to_dict()
        log.info(f"Exit distribution: {exit_dist}")
        
        for r_name in ['green', 'red', 'flat']:
            rd = regime.get(r_name, {})
            log.info(f"  {r_name}: N={rd.get('n_trades', 0)}, Sharpe={rd.get('sharpe', 0):.2f}, "
                     f"WR={rd.get('win_rate', 0):.1%}, PF={rd.get('profit_factor', 0):.2f}")
        
        all_config_results[label] = {
            'trades_df': df,
            'metrics': metrics,
            'regime': regime,
        }
    
    return all_config_results


# ═══════════════════════════════════════════════════════════════
# APPROACH C: MID-TRADE CLASSIFIER EXIT
# ═══════════════════════════════════════════════════════════════

def approach_c_classifier_exit(trades_orig, pred_dates, bar_cache):
    """
    Use MLP confidence to decide hold vs exit.
    Train walk-forward MLP on midtrade features.
    If confidence drops below threshold → exit at market.
    If confidence stays high → hold until bar close.
    """
    log.info("\n" + "=" * 70)
    log.info("APPROACH C: MID-TRADE CLASSIFIER EXIT")
    log.info("=" * 70)
    
    # Load midtrade features
    mt_df = pd.read_parquet(MIDTRADE_FEATURES)
    mt_dates = sorted(mt_df['date'].unique())
    log.info(f"Midtrade features: {len(mt_df)} trades, {len(mt_dates)} unique dates")
    log.info(f"Date range: {mt_dates[0]} to {mt_dates[-1]}")
    
    # Feature columns (exclude metadata/targets)
    exclude = ['date', 'direction', 'fill_price', 'time_of_day_min', 'exit_type',
               'exit_ticks', 'winner', 'pred_magnitude']
    feature_cols = [c for c in mt_df.columns if c not in exclude]
    
    unique_bar_counts = {}
    thresholds = [0.3, 0.4, 0.5]
    all_threshold_results = {}
    
    for thresh in thresholds:
        label = f"MLP_cut_{thresh}"
        log.info(f"\n--- Classifier threshold: {thresh} ---")
        
        results = []
        
        # Walk-forward: train on all midtrade dates BEFORE current trade date
        trade_dates = sorted(trades_orig['date'].unique())
        
        # Pre-train models at epoch boundaries
        mlp_model = None
        scaler = None
        last_train_date = None
        
        for _, trade in trades_orig.iterrows():
            date_str = trade['date']
            bar_idx = trade['bar_idx']
            direction = trade['direction']
            
            if date_str not in unique_bar_counts:
                unique_bar_counts[date_str] = get_unique_bar_count(date_str, pred_dates)
            actual_bar_idx = bar_idx % unique_bar_counts[date_str]
            
            # Retrain MLP every 10 dates or if never trained
            available_mt_dates = [d for d in mt_dates if d < date_str]
            
            if len(available_mt_dates) >= 15 and (mlp_model is None or 
                (last_train_date is not None and 
                 len([d for d in mt_dates if last_train_date < d < date_str]) >= 5)):
                
                train_mask = mt_df['date'].isin(available_mt_dates)
                X_train = np.nan_to_num(mt_df.loc[train_mask, feature_cols].values.astype(float), 0)
                y_train = mt_df.loc[train_mask, 'winner'].values
                
                if len(np.unique(y_train)) >= 2:
                    scaler = StandardScaler()
                    X_scaled = scaler.fit_transform(X_train)
                    
                    mlp_model = MLPClassifier(
                        hidden_layer_sizes=(64, 32),
                        max_iter=300,
                        learning_rate_init=0.001,
                        alpha=0.01,
                        random_state=42,
                        early_stopping=True,
                        validation_fraction=0.2,
                    )
                    mlp_model.fit(X_scaled, y_train)
                    last_train_date = date_str
            
            # Get minute bars
            entry_price, next_minutes, bar_key = get_next_bar_minutes(date_str, actual_bar_idx, bar_cache)
            
            if entry_price is None or next_minutes is None or len(next_minutes) == 0:
                continue
            
            # Get MLP confidence for this trade
            mlp_confidence = None
            if mlp_model is not None and scaler is not None:
                mt_match = mt_df[
                    (mt_df['date'] == date_str) & 
                    (mt_df['direction'] == direction)
                ]
                if len(mt_match) > 0:
                    mt_row = mt_match.iloc[min(bar_idx, len(mt_match) - 1)]
                    X_mt = np.nan_to_num(mt_row[feature_cols].values.reshape(1, -1).astype(float), 0)
                    X_mt_scaled = scaler.transform(X_mt)
                    try:
                        proba = mlp_model.predict_proba(X_mt_scaled)[0]
                        winner_idx = np.where(mlp_model.classes_ == 1)[0]
                        mlp_confidence = float(proba[winner_idx[0]]) if len(winner_idx) > 0 else float(proba[-1])
                    except:
                        mlp_confidence = None
            
            # Decision: if MLP says low confidence → exit early (simulated as partial hold)
            exit_price = next_minutes.iloc[-1]['close']
            
            if direction == 1:
                bar_close_pnl = (exit_price - entry_price) / TICK_SIZE
            else:
                bar_close_pnl = (entry_price - exit_price) / TICK_SIZE
            
            # MFE/MAE
            mfe = 0.0
            mae = 0.0
            for _, row in next_minutes.iterrows():
                if direction == 1:
                    mfe = max(mfe, (row['high'] - entry_price) / TICK_SIZE)
                    mae = max(mae, (entry_price - row['low']) / TICK_SIZE)
                else:
                    mfe = max(mfe, (entry_price - row['low']) / TICK_SIZE)
                    mae = max(mae, (row['high'] - entry_price) / TICK_SIZE)
            
            if mlp_confidence is not None and mlp_confidence < thresh:
                # LOW confidence → exit early
                # Simulate: exit at ~5min into the bar (conservative early exit)
                # Use the price at ~5 minutes in, or 1/6 through the bar
                exit_minute_idx = min(4, len(next_minutes) - 1)  # ~5 min in
                early_exit_price = next_minutes.iloc[exit_minute_idx]['close']
                
                if direction == 1:
                    raw_pnl = (early_exit_price - entry_price) / TICK_SIZE
                else:
                    raw_pnl = (entry_price - early_exit_price) / TICK_SIZE
                
                cost = COST_MARKET_EXIT  # Market exit for classifier cut
                net_pnl = raw_pnl - cost
                exit_type = 'classifier_cut'
                minutes_held = exit_minute_idx + 1
            else:
                # HIGH confidence or no model → hold to bar close
                raw_pnl = bar_close_pnl
                cost = COST_PASSIVE * 2  # Passive both sides
                net_pnl = raw_pnl - cost
                exit_type = 'hold_to_close'
                minutes_held = len(next_minutes)
            
            results.append({
                'date': date_str,
                'direction': direction,
                'bar_idx': bar_idx,
                'entry_price': entry_price,
                'mlp_confidence': mlp_confidence,
                'raw_pnl_ticks': raw_pnl,
                'cost_ticks': cost,
                'net_pnl_ticks': net_pnl,
                'exit_type': exit_type,
                'mfe_ticks': mfe,
                'mae_ticks': mae,
                'minutes_held': minutes_held,
                'bar_close_pnl': bar_close_pnl,
            })
        
        df = pd.DataFrame(results)
        metrics = compute_metrics(df, f"C: {label}")
        regime = regime_stratify(df)
        
        log.info(f"N trades: {metrics['n_trades']}")
        log.info(f"Sharpe: {metrics['sharpe']:.2f}, Sortino: {metrics['sortino']:.2f}")
        log.info(f"WR: {metrics['win_rate']:.1%}, PF: {metrics['profit_factor']:.2f}")
        log.info(f"Total PnL: {metrics['total_pnl_ticks']:.1f} ticks (${metrics['total_pnl_dollars']:.0f})")
        log.info(f"Max DD: {metrics['max_dd_ticks']:.1f} ticks (${metrics['max_dd_dollars']:.0f})")
        log.info(f"Day concentration: {metrics['day_concentration']:.3f} (pass={metrics['day_conc_pass']})")
        log.info(f"Regime gap: {regime['regime_gap']:.3f} (pass={regime['regime_pass']})")
        
        exit_dist = df['exit_type'].value_counts().to_dict()
        log.info(f"Exit distribution: {exit_dist}")
        
        # Classifier value-add: compare cut trades vs held trades
        cut_trades = df[df['exit_type'] == 'classifier_cut']
        held_trades = df[df['exit_type'] == 'hold_to_close']
        if len(cut_trades) > 0:
            # What would have happened if we held the cut trades to bar close?
            cut_bar_close_pnl = cut_trades['bar_close_pnl'].mean()
            cut_actual_pnl = cut_trades['raw_pnl_ticks'].mean()
            log.info(f"Classifier value: cut {len(cut_trades)} trades")
            log.info(f"  If held to close: avg {cut_bar_close_pnl:.2f}t | Actually got: avg {cut_actual_pnl:.2f}t")
            log.info(f"  Savings per cut trade: {cut_bar_close_pnl - cut_actual_pnl:.2f}t")
        
        for r_name in ['green', 'red', 'flat']:
            rd = regime.get(r_name, {})
            log.info(f"  {r_name}: N={rd.get('n_trades', 0)}, Sharpe={rd.get('sharpe', 0):.2f}, "
                     f"WR={rd.get('win_rate', 0):.1%}, PF={rd.get('profit_factor', 0):.2f}")
        
        all_threshold_results[label] = {
            'trades_df': df,
            'metrics': metrics,
            'regime': regime,
        }
    
    return all_threshold_results


# ═══════════════════════════════════════════════════════════════
# MAIN
# ═══════════════════════════════════════════════════════════════

def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("INTEGRATED PIPELINE V2 — EXIT STRUCTURE RESEARCH")
    log.info("=" * 70)
    log.info("Testing 3 exit approaches on same 460 trade entries")
    log.info("Signal is valid (54% directional accuracy). Problem is exit structure.")
    log.info("")
    
    # MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("integrated_pipeline_v2")
        mlflow.start_run(run_name=f"exit_research_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    # Load original trades
    trades_orig = pd.read_parquet(TRADES_FILE)
    pred_data = np.load(ENTRY_PREDS, allow_pickle=True)
    pred_dates = pred_data['dates']
    
    log.info(f"Loaded {len(trades_orig)} trades from integrated_pipeline_v1")
    log.info(f"Date range: {trades_orig['date'].min()} to {trades_orig['date'].max()}")
    log.info(f"Original metrics: Sharpe=20.3, WR=54.1%, PF=4.0 (bar-level artifact)")
    
    bar_cache = {}
    
    # ── Approach A ──
    a_df, a_metrics, a_regime = approach_a_bar_close(trades_orig, pred_dates, bar_cache)
    
    # ── Approach B ──
    b_results = approach_b_wider_stops(trades_orig, pred_dates, bar_cache)
    
    # ── Approach C ──
    c_results = approach_c_classifier_exit(trades_orig, pred_dates, bar_cache)
    
    # ═══════════════════════════════════════════════════════════
    # COMPARATIVE SUMMARY
    # ═══════════════════════════════════════════════════════════
    log.info("\n" + "=" * 70)
    log.info("COMPARATIVE SUMMARY — ALL EXIT APPROACHES")
    log.info("=" * 70)
    
    all_approaches = {}
    all_approaches['A: Bar-Close'] = {'metrics': a_metrics, 'regime': a_regime}
    for k, v in b_results.items():
        all_approaches[f'B: {k}'] = {'metrics': v['metrics'], 'regime': v['regime']}
    for k, v in c_results.items():
        all_approaches[f'C: {k}'] = {'metrics': v['metrics'], 'regime': v['regime']}
    
    header = f"{'Approach':<25} {'N':>5} {'Sharpe':>8} {'Sortino':>8} {'WR':>7} {'PF':>7} {'PnL_t':>8} {'MaxDD_t':>8} {'DayConc':>8} {'RegGap':>7} {'Pass':>5}"
    log.info(header)
    log.info("-" * 105)
    
    for name, data in all_approaches.items():
        m = data['metrics']
        r = data['regime']
        regime_pass = r.get('regime_pass', False)
        day_conc_pass = m.get('day_conc_pass', False)
        overall_pass = regime_pass and day_conc_pass and m['sharpe'] > 0
        
        log.info(f"{name:<25} {m['n_trades']:>5} {m['sharpe']:>8.2f} {m['sortino']:>8.2f} "
                 f"{m['win_rate']:>6.1%} {m['profit_factor']:>7.2f} {m['total_pnl_ticks']:>8.1f} "
                 f"{m['max_dd_ticks']:>8.1f} {m['day_concentration']:>8.3f} "
                 f"{r.get('regime_gap', 0):>7.3f} {'YES' if overall_pass else 'NO':>5}")
    
    # ── Verdict ──
    log.info("\n" + "=" * 70)
    log.info("VERDICT")
    log.info("=" * 70)
    
    # Find best approach
    best_name = None
    best_sharpe = -999
    for name, data in all_approaches.items():
        m = data['metrics']
        r = data['regime']
        if (r.get('regime_pass', False) and m.get('day_conc_pass', False) 
            and m['sharpe'] > best_sharpe and m['n_trades'] >= 20):
            best_sharpe = m['sharpe']
            best_name = name
    
    if best_name:
        bm = all_approaches[best_name]['metrics']
        br = all_approaches[best_name]['regime']
        log.info(f"BEST: {best_name}")
        log.info(f"  Sharpe={bm['sharpe']:.2f}, Sortino={bm['sortino']:.2f}, WR={bm['win_rate']:.1%}, PF={bm['profit_factor']:.2f}")
        log.info(f"  PnL={bm['total_pnl_ticks']:.1f}t (${bm['total_pnl_dollars']:.0f}), MaxDD={bm['max_dd_ticks']:.1f}t")
        log.info(f"  Regime gap={br['regime_gap']:.3f}, Day conc={bm['day_concentration']:.3f}")
    else:
        log.info("No approach passes all validation gates.")
    
    # HC #432 compliance
    log.info(f"\nHC #432 COMPLIANCE (MFE-within-horizon):")
    log.info(f"  Approach A (bar-close): COMPLIANT — exit matches model's 30-min prediction horizon")
    log.info(f"  Approach B (wider stops): CHECK — SL/TP must be within realized MFE p90")
    if len(a_df) > 0:
        p90_mfe = a_df['mfe_ticks'].quantile(0.9)
        log.info(f"  Realized MFE p90 = {p90_mfe:.1f} ticks within 30-min bar")
        for cfg_label, cfg_data in b_results.items():
            tp_val = int(cfg_label.split('TP')[1])
            pass_mfe = tp_val <= p90_mfe
            log.info(f"    {cfg_label}: TP={tp_val} vs p90 MFE={p90_mfe:.1f} → {'PASS' if pass_mfe else 'FAIL'}")
    
    log.info(f"  Approach C (classifier): COMPLIANT — holds to bar close by default, cuts early only when model says low confidence")
    
    # ── MLflow logging ──
    if MLFLOW_AVAILABLE:
        for name, data in all_approaches.items():
            prefix = name.replace(' ', '_').replace(':', '').replace('/', '_').lower()
            for k, v in data['metrics'].items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{prefix}_{k}", v)
            mlflow.log_metric(f"{prefix}_regime_gap", data['regime'].get('regime_gap', 0))
        
        if best_name:
            mlflow.log_param("best_approach", best_name)
            mlflow.log_metric("best_sharpe", best_sharpe)
        
        mlflow.end_run()
    
    # ── Save results ──
    summary = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': time.time() - t0,
        'n_trades_input': len(trades_orig),
        'approaches': {},
    }
    
    for name, data in all_approaches.items():
        summary['approaches'][name] = {
            'metrics': {k: v for k, v in data['metrics'].items() if k != 'label'},
            'regime_gap': data['regime'].get('regime_gap', 0),
            'regime_pass': data['regime'].get('regime_pass', False),
            'regime_details': {r: {k: v for k, v in data['regime'].get(r, {}).items() if k != 'label'}
                              for r in ['green', 'red', 'flat']},
        }
    
    if best_name:
        summary['best_approach'] = best_name
        summary['best_sharpe'] = best_sharpe
    
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        if isinstance(obj, pd.Timestamp): return obj.isoformat()
        return str(obj)
    
    with open(OUTPUT_DIR / 'results.json', 'w') as f:
        json.dump(summary, f, indent=2, default=convert)
    
    # Save individual trade results
    a_df.to_parquet(OUTPUT_DIR / 'trades_approach_a.parquet', index=False)
    for k, v in b_results.items():
        v['trades_df'].to_parquet(OUTPUT_DIR / f'trades_approach_b_{k.replace("/", "_")}.parquet', index=False)
    for k, v in c_results.items():
        v['trades_df'].to_parquet(OUTPUT_DIR / f'trades_approach_c_{k}.parquet', index=False)
    
    elapsed = time.time() - t0
    log.info(f"\nElapsed: {elapsed:.1f}s")
    log.info("DONE — Results saved to output/integrated_pipeline_v2/")


if __name__ == '__main__':
    main()
