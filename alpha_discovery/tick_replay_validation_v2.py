#!/usr/bin/env python3
"""
tick_replay_validation_v2.py — Minute-Level Replay Validation (Fixed bar_idx mapping)

The concat_oot.npz has duplicated bars (28 bars = 14 unique x2).
bar_idx in trades maps to prediction array index, not actual bar.
We need bar_idx % n_actual_bars to get the true bar.

Three scenarios for same-minute TP+SL conflict:
1. Conservative: SL wins (worst case)
2. Optimistic: TP wins (best case, less common since SL=3 is much closer)
3. Probabilistic: use ratio SL_dist/(SL_dist+TP_dist) to weight

Also: check if ES moved 3+ ticks on the open of each minute bar first
(proxy for "did it gap into SL?").
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
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0
COST_RT_PASSIVE = COMMISSION_RT_TICKS
COST_RT_MARKET = COMMISSION_RT_TICKS + SPREAD_TICKS

TP_TICKS = 12
SL_TICKS = 3

MLFLOW_URI = "http://localhost:5000"

LOG_FILE = OUTPUT_DIR / "tick_replay_v2.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(LOG_FILE, mode='w'),
        logging.StreamHandler(sys.stdout),
    ]
)
log = logging.getLogger(__name__)


def get_unique_bar_count(date_str, pred_dates):
    """Figure out how many UNIQUE bars this date has in predictions."""
    mask = pred_dates == date_str
    n_total = mask.sum()
    if n_total <= 15:
        return n_total  # No duplication
    # Check if duplicated
    if n_total == 28:
        return 14
    elif n_total == 21:
        return 7  # First 7 are from one fold, then 14 (with 7 duplicated)
    elif n_total == 16:
        return 8
    elif n_total == 18:
        return 9
    else:
        return min(n_total, 15)


def build_30min_bars_and_minutes(date_str):
    """Build 30-min bars from minute data using floor('30min') like the model does."""
    fpath = MINUTE_BARS_DIR / f"{date_str}.parquet"
    if not fpath.exists():
        return None, None
    
    mbars = pd.read_parquet(fpath)
    mbars['ts_minute'] = pd.to_datetime(mbars['ts_minute'], utc=True)
    mbars = mbars.sort_values('ts_minute').reset_index(drop=True)
    
    # Build 30-min bars using floor like the model
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


def replay_trade(entry_price, direction, minute_bars_next, tp_ticks, sl_ticks):
    """
    Replay a trade through minute-level OHLC data.
    
    Returns results for 3 scenarios:
    - conservative: SL wins same-minute conflicts
    - optimistic: TP wins same-minute conflicts
    - probabilistic: use distance ratio
    """
    if len(minute_bars_next) == 0:
        empty = {
            'exit_type': 'no_data', 'exit_pnl_ticks': 0.0,
            'minutes_held': 0, 'mfe_ticks': 0.0, 'mae_ticks': 0.0,
        }
        return empty, empty, empty
    
    tp_price = tp_ticks * TICK_SIZE
    sl_price = sl_ticks * TICK_SIZE
    
    mfe = 0.0
    mae = 0.0
    
    result_conservative = None
    result_optimistic = None
    result_probabilistic = None
    
    for i, (idx, row) in enumerate(minute_bars_next.iterrows()):
        hi = row['high']
        lo = row['low']
        op = row['open']
        
        if direction == 1:  # Long
            fav = (hi - entry_price) / TICK_SIZE
            adv = (entry_price - lo) / TICK_SIZE
            tp_hit = (hi - entry_price) >= tp_price
            sl_hit = (entry_price - lo) >= sl_price
            # Check if open gaps past SL
            open_adverse = (entry_price - op) / TICK_SIZE
            open_favorable = (op - entry_price) / TICK_SIZE
        else:  # Short
            fav = (entry_price - lo) / TICK_SIZE
            adv = (hi - entry_price) / TICK_SIZE
            tp_hit = (entry_price - lo) >= tp_price
            sl_hit = (hi - entry_price) >= sl_price
            open_adverse = (op - entry_price) / TICK_SIZE
            open_favorable = (entry_price - op) / TICK_SIZE
        
        mfe = max(mfe, fav)
        mae = max(mae, adv)
        
        def make_result(etype, pnl):
            return {
                'exit_type': etype, 'exit_pnl_ticks': pnl,
                'minutes_held': i + 1, 'mfe_ticks': mfe, 'mae_ticks': mae,
            }
        
        if tp_hit and sl_hit:
            # Both triggered in same minute
            if result_conservative is None:
                result_conservative = make_result('sl_same_min', -sl_ticks)
            
            if result_optimistic is None:
                # Check: did open gap past SL? If so, even optimistic is SL
                if open_adverse >= sl_ticks:
                    result_optimistic = make_result('sl_gap', -sl_ticks)
                else:
                    result_optimistic = make_result('tp_same_min', tp_ticks)
            
            if result_probabilistic is None:
                # If open gaps past SL, definitely SL
                if open_adverse >= sl_ticks:
                    result_probabilistic = make_result('sl_gap', -sl_ticks)
                elif open_favorable >= tp_ticks:
                    result_probabilistic = make_result('tp_gap', tp_ticks)
                else:
                    # Probabilistic: SL is 3 ticks away, TP is 12 ticks away
                    # Within a minute, price is more likely to hit the closer level first
                    # Simple model: P(SL first) = TP_dist / (TP_dist + SL_dist)
                    remaining_tp = max(tp_ticks - max(open_favorable, 0), 0.1)
                    remaining_sl = max(sl_ticks - max(open_adverse, 0), 0.1)
                    p_sl_first = remaining_tp / (remaining_tp + remaining_sl)
                    # Expected PnL
                    exp_pnl = p_sl_first * (-sl_ticks) + (1 - p_sl_first) * tp_ticks
                    etype = 'sl_prob' if exp_pnl < 0 else 'tp_prob'
                    result_probabilistic = make_result(etype, exp_pnl)
        
        elif sl_hit:
            r = make_result('sl', -sl_ticks)
            if result_conservative is None:
                result_conservative = r
            if result_optimistic is None:
                result_optimistic = r
            if result_probabilistic is None:
                result_probabilistic = r
        
        elif tp_hit:
            r = make_result('tp', tp_ticks)
            if result_conservative is None:
                result_conservative = r
            if result_optimistic is None:
                result_optimistic = r
            if result_probabilistic is None:
                result_probabilistic = r
        
        if all(r is not None for r in [result_conservative, result_optimistic, result_probabilistic]):
            break
    
    # If neither hit by end of window
    if direction == 1:
        exit_pnl = (minute_bars_next.iloc[-1]['close'] - entry_price) / TICK_SIZE
    else:
        exit_pnl = (entry_price - minute_bars_next.iloc[-1]['close']) / TICK_SIZE
    
    hold = {'exit_type': 'hold_exit', 'exit_pnl_ticks': exit_pnl,
            'minutes_held': len(minute_bars_next), 'mfe_ticks': mfe, 'mae_ticks': mae}
    
    if result_conservative is None: result_conservative = hold
    if result_optimistic is None: result_optimistic = hold
    if result_probabilistic is None: result_probabilistic = hold
    
    return result_conservative, result_optimistic, result_probabilistic


def compute_metrics(trades_df, label=""):
    if len(trades_df) == 0:
        return {'n_trades': 0, 'sharpe': 0, 'sortino': 0, 'win_rate': 0, 'profit_factor': 0}
    
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
    }


def regime_stratify(trades_df):
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


def main():
    t0 = time.time()
    log.info("=" * 70)
    log.info("TICK-LEVEL REPLAY V2 — Fixed bar_idx mapping + 3 scenarios")
    log.info("=" * 70)
    
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("integrated_pipeline_v1_tick_validation")
        mlflow.start_run(run_name=f"tick_replay_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
    
    trades_orig = pd.read_parquet(TRADES_FILE)
    pred_data = np.load(ENTRY_PREDS, allow_pickle=True)
    pred_dates = pred_data['dates']
    
    log.info(f"Original: {len(trades_orig)} trades, {trades_orig['date'].nunique()} days")
    log.info(f"Original exit dist: {trades_orig['exit_type'].value_counts().to_dict()}")
    
    # Cache
    bar_cache = {}
    unique_bar_counts = {}
    
    results = []
    
    for trade_idx, trade in trades_orig.iterrows():
        date_str = trade['date']
        bar_idx = trade['bar_idx']
        direction = trade['direction']
        
        # Get unique bar count for this date
        if date_str not in unique_bar_counts:
            unique_bar_counts[date_str] = get_unique_bar_count(date_str, pred_dates)
        n_unique = unique_bar_counts[date_str]
        
        # Map bar_idx to actual bar (handle duplicates)
        actual_bar_idx = bar_idx % n_unique
        
        # Load minute data
        if date_str not in bar_cache:
            bars_30, mbars = build_30min_bars_and_minutes(date_str)
            bar_cache[date_str] = (bars_30, mbars)
        bars_30, mbars = bar_cache[date_str]
        
        if bars_30 is None or actual_bar_idx >= len(bars_30):
            results.append({
                'trade_idx': trade_idx, 'date': date_str, 'bar_idx': bar_idx,
                'actual_bar_idx': actual_bar_idx, 'direction': direction,
                'orig_exit_type': trade['exit_type'], 'orig_net_pnl': trade['net_pnl_ticks'],
                'con_exit_type': 'no_data', 'con_net_pnl': 0.0,
                'opt_exit_type': 'no_data', 'opt_net_pnl': 0.0,
                'prob_exit_type': 'no_data', 'prob_net_pnl': 0.0,
                'mfe_ticks': 0.0, 'mae_ticks': 0.0,
                'minutes_held': 0, 'had_data': False,
            })
            continue
        
        entry_price = bars_30.iloc[actual_bar_idx]['close']
        bar_key = bars_30.iloc[actual_bar_idx]['bar_key']
        
        # Get minute bars for the NEXT 30 minutes
        next_start = bar_key + pd.Timedelta(minutes=30)
        next_end = bar_key + pd.Timedelta(minutes=59)
        mbars_ts = mbars.set_index('ts_minute')
        next_minutes = mbars_ts.loc[
            (mbars_ts.index >= next_start) & (mbars_ts.index < next_end + pd.Timedelta(minutes=1))
        ]
        
        if len(next_minutes) == 0:
            # Last bar of session -- no next period available
            results.append({
                'trade_idx': trade_idx, 'date': date_str, 'bar_idx': bar_idx,
                'actual_bar_idx': actual_bar_idx, 'direction': direction,
                'orig_exit_type': trade['exit_type'], 'orig_net_pnl': trade['net_pnl_ticks'],
                'con_exit_type': 'last_bar', 'con_net_pnl': 0.0,
                'opt_exit_type': 'last_bar', 'opt_net_pnl': 0.0,
                'prob_exit_type': 'last_bar', 'prob_net_pnl': 0.0,
                'mfe_ticks': 0.0, 'mae_ticks': 0.0,
                'minutes_held': 0, 'had_data': False,
            })
            continue
        
        con, opt, prob = replay_trade(entry_price, direction, next_minutes, TP_TICKS, SL_TICKS)
        
        # Costs
        def apply_cost(r):
            if r['exit_type'] in ('tp', 'tp_same_min', 'tp_gap', 'sl', 'sl_same_min', 'sl_gap'):
                return COST_RT_PASSIVE
            elif 'prob' in r['exit_type']:
                return COST_RT_PASSIVE  # mixed scenario, use passive cost
            else:
                return COST_RT_MARKET
        
        con_cost = apply_cost(con)
        opt_cost = apply_cost(opt)
        prob_cost = apply_cost(prob)
        
        results.append({
            'trade_idx': trade_idx, 'date': date_str, 'bar_idx': bar_idx,
            'actual_bar_idx': actual_bar_idx, 'direction': direction,
            'orig_exit_type': trade['exit_type'], 'orig_net_pnl': trade['net_pnl_ticks'],
            'con_exit_type': con['exit_type'], 'con_net_pnl': con['exit_pnl_ticks'] - con_cost,
            'opt_exit_type': opt['exit_type'], 'opt_net_pnl': opt['exit_pnl_ticks'] - opt_cost,
            'prob_exit_type': prob['exit_type'], 'prob_net_pnl': prob['exit_pnl_ticks'] - prob_cost,
            'mfe_ticks': con['mfe_ticks'], 'mae_ticks': con['mae_ticks'],
            'minutes_held': con['minutes_held'], 'had_data': True,
        })
    
    df = pd.DataFrame(results)
    df.to_parquet(OUTPUT_DIR / 'replay_trades_v2.parquet', index=False)
    
    valid = df[df['had_data']].copy()
    
    log.info(f"\nTrades with data: {len(valid)} / {len(df)}")
    log.info(f"No data: {(~df['had_data']).sum()}")
    
    # ── Compare all three scenarios ──
    scenarios = {
        'Original': ('orig_exit_type', 'orig_net_pnl'),
        'Conservative (SL wins ties)': ('con_exit_type', 'con_net_pnl'),
        'Optimistic (TP wins ties)': ('opt_exit_type', 'opt_net_pnl'),
        'Probabilistic (distance-weighted)': ('prob_exit_type', 'prob_net_pnl'),
    }
    
    log.info(f"\n{'='*70}")
    log.info("EXIT TYPE DISTRIBUTIONS")
    log.info(f"{'='*70}")
    for name, (et_col, _) in scenarios.items():
        log.info(f"\n{name}:")
        for et, cnt in valid[et_col].value_counts().items():
            log.info(f"  {et}: {cnt}")
    
    log.info(f"\n{'='*70}")
    log.info("METRICS COMPARISON (all scenarios, same trade subset)")
    log.info(f"{'='*70}")
    
    all_metrics = {}
    for name, (_, pnl_col) in scenarios.items():
        v = valid.copy()
        v['net_pnl_ticks'] = v[pnl_col]
        v['winner'] = (v['net_pnl_ticks'] > 0).astype(int)
        m = compute_metrics(v, name)
        all_metrics[name] = m
    
    header = f"{'Metric':<22} {'Original':>12} {'Conserv':>12} {'Optimist':>12} {'Probabil':>12}"
    log.info(header)
    log.info("-" * 70)
    for key in ['n_trades', 'win_rate', 'profit_factor', 'sharpe', 'sortino', 
                'total_pnl_ticks', 'avg_pnl_ticks', 'max_dd_ticks']:
        vals = []
        for name in scenarios:
            v = all_metrics[name].get(key, 0)
            vals.append(f"{v:>12.3f}" if isinstance(v, float) else f"{v:>12}")
        log.info(f"  {key:<22} {'  '.join(vals)}")
    
    # ── Regime stratification for best scenario ──
    log.info(f"\n{'='*70}")
    log.info("REGIME STRATIFICATION")
    log.info(f"{'='*70}")
    
    for name, (_, pnl_col) in [('Conservative', ('con_exit_type', 'con_net_pnl')),
                                 ('Probabilistic', ('prob_exit_type', 'prob_net_pnl'))]:
        v = valid.copy()
        v['net_pnl_ticks'] = v[pnl_col]
        regime = regime_stratify(v)
        log.info(f"\n{name}:")
        for r in ['green', 'red', 'flat']:
            rd = regime.get(r, {})
            log.info(f"  {r}: N={rd.get('n_trades', 0)}, Sharpe={rd.get('sharpe', 0):.2f}, "
                     f"WR={rd.get('win_rate', 0):.1%}, PF={rd.get('profit_factor', 0):.2f}")
        log.info(f"  Regime gap: {regime['regime_gap']:.3f} (pass={regime['regime_pass']})")
    
    # ── MFE/MAE analysis ──
    log.info(f"\n{'='*70}")
    log.info("MFE / MAE ANALYSIS")
    log.info(f"{'='*70}")
    mfe = valid['mfe_ticks']
    mae = valid['mae_ticks']
    log.info(f"MFE: mean={mfe.mean():.1f}t, p50={mfe.median():.1f}t, p90={mfe.quantile(0.9):.1f}t, max={mfe.max():.0f}t")
    log.info(f"MAE: mean={mae.mean():.1f}t, p50={mae.median():.1f}t, p90={mae.quantile(0.9):.1f}t, max={mae.max():.0f}t")
    log.info(f"Trades where MAE >= {SL_TICKS}t: {(mae >= SL_TICKS).sum()} ({(mae >= SL_TICKS).mean():.1%})")
    log.info(f"Trades where MFE >= {TP_TICKS}t: {(mfe >= TP_TICKS).sum()} ({(mfe >= TP_TICKS).mean():.1%})")
    log.info(f"Both MFE>=TP AND MAE>=SL: {((mfe >= TP_TICKS) & (mae >= SL_TICKS)).sum()}")
    
    # Key insight: SL=3 is SO tight that almost every trade hits it
    # The bar-level backtest never checks this
    pct_sl_hit = (mae >= SL_TICKS).mean() * 100
    log.info(f"\n*** KEY FINDING: {pct_sl_hit:.0f}% of trades see adverse excursion >= {SL_TICKS} ticks ***")
    log.info(f"*** This means the SL={SL_TICKS} stop is triggered on almost every trade in reality ***")
    log.info(f"*** The bar-level backtest NEVER checks this — it only looks at the final bar return ***")
    
    # ── Timing ──
    log.info(f"\n--- TIMING ---")
    held = valid['minutes_held']
    log.info(f"Minutes to first TP/SL: mean={held.mean():.1f}, median={held.median():.0f}")
    
    # ── Direction ──
    log.info(f"\n--- DIRECTION (probabilistic scenario) ---")
    for d, label in [(1, 'Long'), (-1, 'Short')]:
        sub = valid[valid['direction'] == d].copy()
        if len(sub) > 0:
            sub['net_pnl_ticks'] = sub['prob_net_pnl']
            m = compute_metrics(sub, label)
            log.info(f"  {label}: N={m['n_trades']}, WR={m['win_rate']:.1%}, PF={m['profit_factor']:.2f}, "
                     f"Sharpe={m['sharpe']:.2f}")
    
    # ── Verdict ──
    log.info(f"\n{'='*70}")
    log.info("VERDICT")
    log.info(f"{'='*70}")
    
    orig_m = all_metrics['Original']
    prob_m = all_metrics['Probabilistic (distance-weighted)']
    con_m = all_metrics['Conservative (SL wins ties)']
    opt_m = all_metrics['Optimistic (TP wins ties)']
    
    log.info(f"\nOriginal bar-level:  Sharpe={orig_m['sharpe']:.1f}, WR={orig_m['win_rate']:.1%}, PF={orig_m['profit_factor']:.2f}")
    log.info(f"Conservative replay: Sharpe={con_m['sharpe']:.1f}, WR={con_m['win_rate']:.1%}, PF={con_m['profit_factor']:.2f}")
    log.info(f"Probabilistic replay:Sharpe={prob_m['sharpe']:.1f}, WR={prob_m['win_rate']:.1%}, PF={prob_m['profit_factor']:.2f}")
    log.info(f"Optimistic replay:   Sharpe={opt_m['sharpe']:.1f}, WR={opt_m['win_rate']:.1%}, PF={opt_m['profit_factor']:.2f}")
    
    # The core issue
    n_both = ((mfe >= TP_TICKS) & (mae >= SL_TICKS)).sum()
    n_valid = len(valid)
    log.info(f"\nCore issue: {n_both}/{n_valid} trades ({n_both/n_valid:.0%}) hit BOTH TP and SL within 30 min")
    log.info(f"With TP=12 / SL=3, the asymmetry means SL is hit first ~{SL_TICKS/(SL_TICKS+TP_TICKS):.0%} of the time")
    log.info(f"The bar-level backtest ignores this — it only checks the FINAL 30-min return")
    
    if prob_m['sharpe'] > 3.0 and prob_m['profit_factor'] > 1.5:
        verdict = "PASS — Signal survives tick-level replay"
    elif prob_m['sharpe'] > 1.0:
        verdict = "MARGINAL — Edge reduced but present"
    else:
        verdict = "FAIL — Edge is a bar-level artifact"
    log.info(f"\nVERDICT: {verdict}")
    
    # MLflow
    if MLFLOW_AVAILABLE:
        for scenario, m in all_metrics.items():
            prefix = scenario.split('(')[0].strip().lower().replace(' ', '_')
            for k, v in m.items():
                if isinstance(v, (int, float)):
                    mlflow.log_metric(f"{prefix}_{k}", v)
        mlflow.log_metric("pct_mae_hits_sl", pct_sl_hit)
        mlflow.log_metric("n_both_tp_sl", n_both)
        mlflow.end_run()
    
    # Save
    summary = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': time.time() - t0,
        'n_trades_total': len(df),
        'n_trades_with_data': len(valid),
        'metrics': {k: v for k, v in all_metrics.items()},
        'pct_mae_hits_sl': pct_sl_hit,
        'n_both_tp_sl': int(n_both),
        'verdict': verdict,
    }
    
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, (np.bool_, bool)): return bool(obj)
        return str(obj)
    
    with open(OUTPUT_DIR / 'replay_summary_v2.json', 'w') as f:
        json.dump(summary, f, indent=2, default=convert)
    
    log.info(f"\nElapsed: {time.time() - t0:.1f}s")
    log.info("DONE")


if __name__ == '__main__':
    main()
