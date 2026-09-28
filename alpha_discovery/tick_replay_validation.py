#!/usr/bin/env python3
"""
tick_replay_validation.py — Minute-Level Replay Validation for integrated_pipeline_v1

Validates the 30-min bar backtest by replaying each trade through minute-level OHLC data.
The bar-level backtest applies TP/SL to the 30-min return, but doesn't check if SL was
hit FIRST during the intra-bar price path. This script fixes that.

For each trade:
1. Entry at bar close price (passive limit assumed)
2. Walk through next 30 minutes of minute-bar OHLC data
3. At each minute: check if SL or TP is touched
4. If both are touched in the same minute: assume ADVERSE outcome first (conservative)
5. If neither by end of 30 min: exit at bar close (hold_exit)
6. Apply FIFO costs

Logs results to MLflow as experiment 'integrated_pipeline_v1_tick_validation'.
"""

import os, sys, json, time, logging, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime

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
OUTPUT_DIR = LVL3 / "output/integrated_pipeline_v1_tick_validation"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ── Constants ──
TICK_SIZE = 0.25  # ES tick size in points
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0
COST_RT_PASSIVE = COMMISSION_RT_TICKS
COST_RT_MARKET = COMMISSION_RT_TICKS + SPREAD_TICKS  # market order entry/exit

TP_TICKS = 12
SL_TICKS = 3

MLFLOW_URI = "http://localhost:5000"

# ── Logging ──
LOG_FILE = OUTPUT_DIR / "tick_replay.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


def build_30min_bars(date_str):
    """Build 30-min OHLCV bars from minute data. Return both 30-min bars and raw minute bars."""
    fpath = MINUTE_BARS_DIR / f"{date_str}.parquet"
    if not fpath.exists():
        return None, None
    
    mbars = pd.read_parquet(fpath)
    mbars = mbars.sort_values('ts_minute').reset_index(drop=True)
    
    # Build 30-min bars by resampling
    mbars_indexed = mbars.set_index('ts_minute')
    bars_30 = mbars_indexed.resample('30min').agg({
        'open': 'first',
        'high': 'max',
        'low': 'min',
        'close': 'last',
        'volume': 'sum',
    }).dropna().reset_index()
    
    return bars_30, mbars


def replay_trade_minute_level(entry_price, direction, minute_bars_next, tp_ticks, sl_ticks):
    """
    Replay a trade through minute-level OHLC data.
    
    Args:
        entry_price: Entry price (bar close)
        direction: 1 (long) or -1 (short)
        minute_bars_next: DataFrame of minute bars for the next 30 minutes
        tp_ticks: Take profit in ticks
        sl_ticks: Stop loss in ticks (positive number, applied as adverse)
    
    Returns:
        dict with exit_type, exit_pnl_ticks, minutes_held, exit_minute_idx
    """
    if len(minute_bars_next) == 0:
        return {
            'exit_type': 'no_data',
            'exit_pnl_ticks': 0.0,
            'minutes_held': 0,
            'exit_minute_idx': -1,
            'mfe_ticks': 0.0,
            'mae_ticks': 0.0,
        }
    
    tp_price_diff = tp_ticks * TICK_SIZE  # TP in price terms
    sl_price_diff = sl_ticks * TICK_SIZE  # SL in price terms
    
    mfe = 0.0  # Max favorable excursion in ticks
    mae = 0.0  # Max adverse excursion in ticks
    
    for i, (idx, row) in enumerate(minute_bars_next.iterrows()):
        hi = row['high']
        lo = row['low']
        cl = row['close']
        
        if direction == 1:  # Long
            favorable_ticks = (hi - entry_price) / TICK_SIZE
            adverse_ticks = (entry_price - lo) / TICK_SIZE
            
            tp_hit = (hi - entry_price) >= tp_price_diff
            sl_hit = (entry_price - lo) >= sl_price_diff
        else:  # Short
            favorable_ticks = (entry_price - lo) / TICK_SIZE
            adverse_ticks = (hi - entry_price) / TICK_SIZE
            
            tp_hit = (entry_price - lo) >= tp_price_diff
            sl_hit = (hi - entry_price) >= sl_price_diff
        
        mfe = max(mfe, favorable_ticks)
        mae = max(mae, adverse_ticks)
        
        if tp_hit and sl_hit:
            # Both triggered in same minute — CONSERVATIVE: SL wins
            # This is the key difference from the bar-level backtest
            return {
                'exit_type': 'sl_conservative',
                'exit_pnl_ticks': -sl_ticks,
                'minutes_held': i + 1,
                'exit_minute_idx': i,
                'mfe_ticks': mfe,
                'mae_ticks': mae,
            }
        elif sl_hit:
            return {
                'exit_type': 'sl',
                'exit_pnl_ticks': -sl_ticks,
                'minutes_held': i + 1,
                'exit_minute_idx': i,
                'mfe_ticks': mfe,
                'mae_ticks': mae,
            }
        elif tp_hit:
            return {
                'exit_type': 'tp',
                'exit_pnl_ticks': tp_ticks,
                'minutes_held': i + 1,
                'exit_minute_idx': i,
                'mfe_ticks': mfe,
                'mae_ticks': mae,
            }
    
    # Neither TP nor SL hit — exit at last close
    if direction == 1:
        exit_pnl = (minute_bars_next.iloc[-1]['close'] - entry_price) / TICK_SIZE
    else:
        exit_pnl = (entry_price - minute_bars_next.iloc[-1]['close']) / TICK_SIZE
    
    return {
        'exit_type': 'hold_exit',
        'exit_pnl_ticks': exit_pnl,
        'minutes_held': len(minute_bars_next),
        'exit_minute_idx': len(minute_bars_next) - 1,
        'mfe_ticks': mfe,
        'mae_ticks': mae,
    }


def compute_metrics(trades_df, label=""):
    """Compute Sharpe, Sortino, PF, WR from trade-level results."""
    if len(trades_df) == 0:
        return {'n_trades': 0, 'sharpe': 0, 'sortino': 0}
    
    daily_pnl = trades_df.groupby('date')['net_pnl_ticks'].sum()
    n_trades = len(trades_df)
    total_pnl = float(trades_df['net_pnl_ticks'].sum())
    
    winners = (trades_df['net_pnl_ticks'] > 0).sum()
    wr = float(winners / n_trades) if n_trades > 0 else 0
    
    gross_profit = float(trades_df.loc[trades_df['net_pnl_ticks'] > 0, 'net_pnl_ticks'].sum())
    gross_loss = float(abs(trades_df.loc[trades_df['net_pnl_ticks'] < 0, 'net_pnl_ticks'].sum()))
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
    """Stratify by regime using flow features."""
    flow_df = pd.read_parquet(FLOW_FEATURES)
    regime_map = {}
    for _, row in flow_df.iterrows():
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
    
    trades_df = trades_df.copy()
    trades_df['regime'] = trades_df['date'].map(lambda d: regime_map.get(d, 'unknown'))
    
    results = {}
    for regime in ['green', 'red', 'flat']:
        rt = trades_df[trades_df['regime'] == regime]
        results[regime] = compute_metrics(rt, label=regime) if len(rt) > 0 else {'n_trades': 0, 'sharpe': 0}
    
    sg = results.get('green', {}).get('sharpe', 0)
    sr = results.get('red', {}).get('sharpe', 0)
    mx = max(abs(sg), abs(sr))
    gap = abs(sg - sr) / mx if mx > 0 else 0
    
    results['regime_gap'] = float(gap)
    results['regime_pass'] = bool(gap < 0.50)
    return results


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("TICK-LEVEL REPLAY VALIDATION — integrated_pipeline_v1")
    log.info("=" * 70)
    log.info(f"TP={TP_TICKS} ticks, SL={SL_TICKS} ticks")
    log.info(f"Conservative rule: if both TP and SL touched in same minute, SL wins")
    log.info(f"Cost model: passive entry={COST_RT_PASSIVE:.3f}t, market={COST_RT_MARKET:.3f}t")
    
    # MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("integrated_pipeline_v1_tick_validation")
        mlflow.start_run(run_name=f"tick_replay_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
        mlflow.log_params({
            'tp_ticks': TP_TICKS,
            'sl_ticks': SL_TICKS,
            'cost_passive': COST_RT_PASSIVE,
            'cost_market': COST_RT_MARKET,
            'method': 'minute_bar_replay_conservative',
        })
    
    # Load original trades
    trades_orig = pd.read_parquet(TRADES_FILE)
    log.info(f"Original trades: {len(trades_orig)} trades across {trades_orig['date'].nunique()} days")
    log.info(f"Original exit distribution: {trades_orig['exit_type'].value_counts().to_dict()}")
    
    # Load entry predictions to get bar close prices
    pred_data = np.load(ENTRY_PREDS, allow_pickle=True)
    pred_dates = pred_data['dates']
    pred_actuals = pred_data['actuals']
    
    # Process each trade
    replay_results = []
    dates_with_data = 0
    dates_without_data = 0
    
    # Cache minute bars and 30-min bars per date
    bar_cache = {}
    
    for trade_idx, trade in trades_orig.iterrows():
        date_str = trade['date']
        bar_idx = trade['bar_idx']
        direction = trade['direction']
        
        # Build 30-min bars for this date (cached)
        if date_str not in bar_cache:
            bars_30, mbars = build_30min_bars(date_str)
            if bars_30 is None:
                bar_cache[date_str] = (None, None)
                dates_without_data += 1
            else:
                bar_cache[date_str] = (bars_30, mbars)
                dates_with_data += 1
        
        bars_30, mbars = bar_cache[date_str]
        
        if bars_30 is None:
            replay_results.append({
                'trade_idx': trade_idx,
                'date': date_str,
                'bar_idx': bar_idx,
                'direction': direction,
                'orig_exit_type': trade['exit_type'],
                'orig_net_pnl': trade['net_pnl_ticks'],
                'replay_exit_type': 'no_data',
                'replay_exit_pnl': 0.0,
                'replay_net_pnl': 0.0,
                'replay_cost': 0.0,
                'minutes_held': 0,
                'mfe_ticks': 0.0,
                'mae_ticks': 0.0,
                'had_data': False,
            })
            continue
        
        # Get entry price = close of bar at bar_idx
        if bar_idx >= len(bars_30):
            # bar_idx out of range for this day
            replay_results.append({
                'trade_idx': trade_idx,
                'date': date_str,
                'bar_idx': bar_idx,
                'direction': direction,
                'orig_exit_type': trade['exit_type'],
                'orig_net_pnl': trade['net_pnl_ticks'],
                'replay_exit_type': 'bar_idx_oor',
                'replay_exit_pnl': 0.0,
                'replay_net_pnl': 0.0,
                'replay_cost': 0.0,
                'minutes_held': 0,
                'mfe_ticks': 0.0,
                'mae_ticks': 0.0,
                'had_data': False,
            })
            continue
        
        entry_price = bars_30.iloc[bar_idx]['close']
        
        # Get minute bars for the NEXT 30-min period
        bar_start = bars_30.iloc[bar_idx]['ts_minute']
        next_period_start = bar_start + pd.Timedelta(minutes=30)
        next_period_end = bar_start + pd.Timedelta(minutes=59)
        
        mbars_indexed = mbars.set_index('ts_minute')
        next_minutes = mbars_indexed.loc[
            (mbars_indexed.index >= next_period_start) & 
            (mbars_indexed.index < next_period_end + pd.Timedelta(minutes=1))
        ]
        
        # Run minute-level replay
        result = replay_trade_minute_level(
            entry_price, direction, next_minutes, TP_TICKS, SL_TICKS
        )
        
        # Apply costs (passive entry assumed for all)
        # TP/SL exits are passive limits, hold_exit is market
        if result['exit_type'] in ('tp', 'sl', 'sl_conservative'):
            cost = COST_RT_PASSIVE
        else:
            cost = COST_RT_MARKET  # market exit for hold/timeout
        
        net_pnl = result['exit_pnl_ticks'] - cost
        
        replay_results.append({
            'trade_idx': trade_idx,
            'date': date_str,
            'bar_idx': bar_idx,
            'direction': direction,
            'orig_exit_type': trade['exit_type'],
            'orig_net_pnl': trade['net_pnl_ticks'],
            'replay_exit_type': result['exit_type'],
            'replay_exit_pnl': result['exit_pnl_ticks'],
            'replay_net_pnl': net_pnl,
            'replay_cost': cost,
            'minutes_held': result['minutes_held'],
            'mfe_ticks': result['mfe_ticks'],
            'mae_ticks': result['mae_ticks'],
            'had_data': True,
        })
    
    replay_df = pd.DataFrame(replay_results)
    replay_df.to_parquet(OUTPUT_DIR / 'replay_trades.parquet', index=False)
    
    # Filter to trades with data
    valid_replay = replay_df[replay_df['had_data']].copy()
    
    log.info(f"\n{'='*70}")
    log.info("REPLAY RESULTS")
    log.info(f"{'='*70}")
    log.info(f"Total trades: {len(trades_orig)}")
    log.info(f"Trades with minute data: {len(valid_replay)}")
    log.info(f"Trades without data: {len(replay_df) - len(valid_replay)}")
    
    # Exit type comparison
    log.info(f"\n--- EXIT TYPE COMPARISON ---")
    log.info(f"Original exit distribution:")
    for et, cnt in trades_orig['exit_type'].value_counts().items():
        log.info(f"  {et}: {cnt}")
    
    log.info(f"\nReplay exit distribution:")
    for et, cnt in valid_replay['replay_exit_type'].value_counts().items():
        log.info(f"  {et}: {cnt}")
    
    # Key metric: how many TP trades flipped to SL
    orig_tp = set(valid_replay[valid_replay['orig_exit_type'] == 'tp'].index)
    replay_sl = set(valid_replay[valid_replay['replay_exit_type'].isin(['sl', 'sl_conservative'])].index)
    flipped_tp_to_sl = orig_tp & replay_sl
    log.info(f"\n*** CRITICAL: {len(flipped_tp_to_sl)} trades that were TP in original became SL in replay ***")
    
    orig_sl = set(valid_replay[valid_replay['orig_exit_type'] == 'sl'].index)
    replay_tp = set(valid_replay[valid_replay['replay_exit_type'] == 'tp'].index)
    flipped_sl_to_tp = orig_sl & replay_tp
    log.info(f"    {len(flipped_sl_to_tp)} trades that were SL in original became TP in replay")
    
    conservative_sl = valid_replay[valid_replay['replay_exit_type'] == 'sl_conservative']
    log.info(f"    {len(conservative_sl)} trades resolved as SL due to conservative same-minute assumption")
    
    # Compute metrics for replay
    valid_replay_metrics = valid_replay.copy()
    valid_replay_metrics['net_pnl_ticks'] = valid_replay_metrics['replay_net_pnl']
    valid_replay_metrics['winner'] = (valid_replay_metrics['net_pnl_ticks'] > 0).astype(int)
    
    replay_metrics = compute_metrics(valid_replay_metrics, "Replay (conservative)")
    
    # Compare with original on same subset
    orig_subset = trades_orig.loc[valid_replay.index].copy()
    orig_metrics = compute_metrics(orig_subset, "Original (same subset)")
    
    log.info(f"\n{'='*70}")
    log.info("METRICS COMPARISON (same trade subset)")
    log.info(f"{'='*70}")
    log.info(f"{'Metric':<25} {'Original':>15} {'Replay':>15} {'Delta':>15}")
    log.info("-" * 70)
    for key in ['n_trades', 'win_rate', 'profit_factor', 'sharpe', 'sortino', 
                'total_pnl_ticks', 'total_pnl_dollars', 'avg_pnl_ticks', 'max_dd_ticks']:
        ov = orig_metrics.get(key, 0)
        rv = replay_metrics.get(key, 0)
        delta = rv - ov if isinstance(ov, (int, float)) else 'N/A'
        if isinstance(ov, float):
            log.info(f"  {key:<25} {ov:>15.3f} {rv:>15.3f} {delta:>15.3f}")
        else:
            log.info(f"  {key:<25} {ov:>15} {rv:>15} {delta:>15}")
    
    # Regime stratification
    log.info(f"\n{'='*70}")
    log.info("REGIME STRATIFICATION (replay)")
    log.info(f"{'='*70}")
    regime = regime_stratify(valid_replay_metrics)
    for r_name in ['green', 'red', 'flat']:
        r_data = regime.get(r_name, {})
        log.info(f"  {r_name}: N={r_data.get('n_trades', 0)}, "
                 f"Sharpe={r_data.get('sharpe', 0):.2f}, "
                 f"WR={r_data.get('win_rate', 0):.1%}, "
                 f"PF={r_data.get('profit_factor', 0):.2f}")
    log.info(f"  Regime gap: {regime['regime_gap']:.3f} (pass={regime['regime_pass']})")
    
    # MFE/MAE analysis
    log.info(f"\n{'='*70}")
    log.info("MFE / MAE ANALYSIS")
    log.info(f"{'='*70}")
    mfe = valid_replay['mfe_ticks']
    mae = valid_replay['mae_ticks']
    log.info(f"MFE: mean={mfe.mean():.1f}t, p50={mfe.median():.1f}t, p90={mfe.quantile(0.9):.1f}t")
    log.info(f"MAE: mean={mae.mean():.1f}t, p50={mae.median():.1f}t, p90={mae.quantile(0.9):.1f}t")
    log.info(f"Trades where MAE >= {SL_TICKS}t (SL hit): {(mae >= SL_TICKS).sum()}")
    log.info(f"Trades where MFE >= {TP_TICKS}t (TP reachable): {(mfe >= TP_TICKS).sum()}")
    log.info(f"Trades where both MFE >= TP and MAE >= SL: {((mfe >= TP_TICKS) & (mae >= SL_TICKS)).sum()}")
    
    # Per-minute resolution analysis for the conservative SL trades
    if len(conservative_sl) > 0:
        log.info(f"\n--- Conservative SL trades (same-minute ambiguity) ---")
        log.info(f"  Count: {len(conservative_sl)}")
        log.info(f"  Avg MFE: {conservative_sl['mfe_ticks'].mean():.1f}t")
        log.info(f"  Avg MAE: {conservative_sl['mae_ticks'].mean():.1f}t")
        log.info(f"  These trades HAD both TP and SL in the same minute bar.")
        log.info(f"  With tick data, ~50% might actually be TP. Midpoint estimate:")
        
        # Midpoint scenario: half of conservative SL become TP
        mid_df = valid_replay_metrics.copy()
        cons_mask = valid_replay['replay_exit_type'] == 'sl_conservative'
        n_cons = cons_mask.sum()
        # Flip half to TP
        flip_indices = mid_df[cons_mask].sample(n=n_cons//2, random_state=42).index
        mid_df.loc[flip_indices, 'net_pnl_ticks'] = TP_TICKS - COST_RT_PASSIVE
        mid_df.loc[flip_indices, 'winner'] = 1
        mid_metrics = compute_metrics(mid_df, "Midpoint estimate")
        log.info(f"  Midpoint Sharpe: {mid_metrics['sharpe']:.2f} (vs conservative {replay_metrics['sharpe']:.2f})")
        log.info(f"  Midpoint WR: {mid_metrics['win_rate']:.1%} (vs conservative {replay_metrics['win_rate']:.1%})")
        log.info(f"  Midpoint PF: {mid_metrics['profit_factor']:.2f} (vs conservative {replay_metrics['profit_factor']:.2f})")
    
    # Direction breakdown
    log.info(f"\n--- DIRECTION BREAKDOWN (replay) ---")
    for d, label in [(1, 'Long'), (-1, 'Short')]:
        sub = valid_replay_metrics[valid_replay_metrics['direction'] == d]
        if len(sub) > 0:
            m = compute_metrics(sub, label)
            log.info(f"  {label}: N={m['n_trades']}, WR={m['win_rate']:.1%}, PF={m['profit_factor']:.2f}, "
                     f"Sharpe={m['sharpe']:.2f}, avg={m['avg_pnl_ticks']:.2f}t")
    
    # Timing analysis
    log.info(f"\n--- TIMING ANALYSIS ---")
    held = valid_replay['minutes_held']
    log.info(f"Minutes held: mean={held.mean():.1f}, median={held.median():.0f}, p90={held.quantile(0.9):.0f}")
    tp_trades = valid_replay[valid_replay['replay_exit_type'] == 'tp']
    sl_trades = valid_replay[valid_replay['replay_exit_type'].isin(['sl', 'sl_conservative'])]
    if len(tp_trades) > 0:
        log.info(f"TP trades: avg time={tp_trades['minutes_held'].mean():.1f}min")
    if len(sl_trades) > 0:
        log.info(f"SL trades: avg time={sl_trades['minutes_held'].mean():.1f}min")
    
    # Final verdict
    log.info(f"\n{'='*70}")
    log.info("VERDICT")
    log.info(f"{'='*70}")
    
    pnl_diff = replay_metrics['total_pnl_ticks'] - orig_metrics['total_pnl_ticks']
    pnl_diff_pct = (pnl_diff / abs(orig_metrics['total_pnl_ticks'])) * 100 if orig_metrics['total_pnl_ticks'] != 0 else 0
    
    log.info(f"P&L difference: {pnl_diff:.1f} ticks ({pnl_diff_pct:+.1f}%)")
    log.info(f"Sharpe difference: {replay_metrics['sharpe'] - orig_metrics['sharpe']:.2f}")
    log.info(f"WR difference: {replay_metrics['win_rate'] - orig_metrics['win_rate']:.1%}")
    
    if replay_metrics['sharpe'] > 3.0 and replay_metrics['profit_factor'] > 1.5:
        log.info("VERDICT: Signal survives tick-level replay. Edge is REAL (with minute-bar resolution).")
    elif replay_metrics['sharpe'] > 1.0:
        log.info("VERDICT: Signal partially survives. Edge reduced but present.")
    else:
        log.info("VERDICT: Signal does NOT survive tick-level replay. Edge is likely a bar-level ARTIFACT.")
    
    log.info(f"\nNOTE: Conservative assumption (SL wins same-minute conflicts) is worst-case.")
    log.info(f"True result with tick data likely between conservative and midpoint estimates.")
    
    # MLflow logging
    if MLFLOW_AVAILABLE:
        for k, v in replay_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"replay_{k}", v)
        for k, v in orig_metrics.items():
            if isinstance(v, (int, float)):
                mlflow.log_metric(f"orig_{k}", v)
        mlflow.log_metric("pnl_diff_ticks", pnl_diff)
        mlflow.log_metric("pnl_diff_pct", pnl_diff_pct)
        mlflow.log_metric("flipped_tp_to_sl", len(flipped_tp_to_sl))
        mlflow.log_metric("conservative_sl_count", len(conservative_sl))
        mlflow.log_metric("regime_gap", regime['regime_gap'])
        mlflow.end_run()
    
    # Save summary
    summary = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': time.time() - t0,
        'original_metrics': orig_metrics,
        'replay_metrics': replay_metrics,
        'pnl_diff_ticks': pnl_diff,
        'pnl_diff_pct': pnl_diff_pct,
        'flipped_tp_to_sl': len(flipped_tp_to_sl),
        'flipped_sl_to_tp': len(flipped_sl_to_tp),
        'conservative_sl_count': len(conservative_sl),
        'regime_stratification': {k: v for k, v in regime.items() if k in ['green', 'red', 'flat', 'regime_gap', 'regime_pass']},
        'mfe_stats': {
            'mean': float(mfe.mean()),
            'p50': float(mfe.median()),
            'p90': float(mfe.quantile(0.9)),
        },
        'mae_stats': {
            'mean': float(mae.mean()),
            'p50': float(mae.median()),
            'p90': float(mae.quantile(0.9)),
        },
    }
    
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        return str(obj)
    
    with open(OUTPUT_DIR / 'replay_summary.json', 'w') as f:
        json.dump(summary, f, indent=2, default=convert)
    
    elapsed = time.time() - t0
    log.info(f"\nElapsed: {elapsed:.1f}s")
    log.info("DONE")


if __name__ == '__main__':
    main()
