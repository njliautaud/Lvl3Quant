#!/usr/bin/env python3
"""
Parameter Sensitivity Grid — SL/TP sweep using minute-level replay.
Uses the same replay_trade_wider logic from integrated_pipeline_v2.py
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

# Add parent to path for imports
sys.path.insert(0, str(Path("/home/nick/Lvl3Quant/alpha_discovery")))

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

LVL3 = Path("/home/nick/Lvl3Quant")
V1_TRADES = LVL3 / "output/integrated_pipeline_v1/best_trades.parquet"
MINUTE_BARS_DIR = LVL3 / "data/processed/mbo_minute_bars_v1"
ENTRY_PREDS = LVL3 / "output/lh_30min_deep_v1/concat_oot.npz"
OUTPUT_DIR = LVL3 / "output/integrated_pipeline_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0
COST_MARKET_EXIT = COMMISSION_RT_TICKS + SPREAD_TICKS  # 1.376
COST_PASSIVE = COMMISSION_RT_TICKS
MLFLOW_URI = "http://localhost:5000"


def get_unique_bar_count(date_str, pred_dates):
    mask = pred_dates == date_str
    n_total = mask.sum()
    if n_total <= 15: return n_total
    if n_total == 28: return 14
    elif n_total == 21: return 7
    elif n_total == 16: return 8
    elif n_total == 18: return 9
    else: return min(n_total, 15)


def load_bar_cache():
    """Load all minute bar data into memory."""
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
    return bar_cache


def get_next_bar_minutes(date_str, actual_bar_idx, bar_cache):
    """Get minute data for the bar AFTER the signal bar."""
    if date_str not in bar_cache:
        return None, None, None
    bars = bar_cache[date_str]
    if actual_bar_idx + 1 >= len(bars):
        return None, None, None
    signal_bar = bars[actual_bar_idx]
    next_bar = bars[actual_bar_idx + 1]
    entry_price = next_bar['open']
    return entry_price, next_bar['minutes'], next_bar['bar_key']


def replay_trade(entry_price, direction, minute_bars_next, tp_ticks, sl_ticks):
    """Replay a trade through minute-level data with stops."""
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
            remaining_tp = max(tp_ticks - mfe, 0.1)
            remaining_sl = max(sl_ticks - mae, 0.1)
            p_sl_first = remaining_tp / (remaining_tp + remaining_sl)
            if p_sl_first > 0.5:
                return {'exit_type': 'sl', 'pnl': -sl_ticks, 'mfe': mfe, 'mae': mae}
            else:
                return {'exit_type': 'tp', 'pnl': tp_ticks, 'mfe': mfe, 'mae': mae}
        elif sl_hit:
            return {'exit_type': 'sl', 'pnl': -sl_ticks, 'mfe': mfe, 'mae': mae}
        elif tp_hit:
            return {'exit_type': 'tp', 'pnl': tp_ticks, 'mfe': mfe, 'mae': mae}
    
    # Neither hit — bar close
    exit_price = minute_bars_next.iloc[-1]['close']
    if direction == 1:
        exit_pnl = (exit_price - entry_price) / TICK_SIZE
    else:
        exit_pnl = (entry_price - exit_price) / TICK_SIZE
    return {'exit_type': 'bar_close', 'pnl': exit_pnl, 'mfe': mfe, 'mae': mae}


def main():
    print("=" * 70)
    print("PARAMETER SENSITIVITY GRID — Minute-level Replay")
    print("=" * 70)

    # Load trade entries
    trades = pd.read_parquet(V1_TRADES)
    print(f"Loaded {len(trades)} trade entries")
    
    # Load predictions for bar indexing
    ep = np.load(ENTRY_PREDS)
    pred_dates = ep['dates']
    
    # Load minute bars
    print("Loading minute bar cache...")
    bar_cache = load_bar_cache()
    print(f"Loaded {len(bar_cache)} days of minute data")
    
    unique_bar_counts = {}
    
    # SL/TP grid
    sl_grid = [10, 12, 15, 18, 20, 25]
    tp_grid = [20, 25, 30, 35, 40, 50]
    
    results = {}
    
    print(f"\n{'SL\\TP':>8}", end='')
    for tp in tp_grid:
        print(f"  TP{tp:>3}", end='')
    print("   (daily Sharpe)")
    
    for sl in sl_grid:
        print(f"  SL{sl:>3}", end='')
        for tp in tp_grid:
            pnl_list = []
            dates_list = []
            
            for _, trade in trades.iterrows():
                date_str = trade['date']
                bar_idx = trade['bar_idx']
                direction = trade['direction']
                
                if date_str not in unique_bar_counts:
                    unique_bar_counts[date_str] = get_unique_bar_count(date_str, pred_dates)
                actual_bar_idx = bar_idx % unique_bar_counts[date_str]
                
                entry_price, next_minutes, bar_key = get_next_bar_minutes(date_str, actual_bar_idx, bar_cache)
                if entry_price is None or next_minutes is None or len(next_minutes) == 0:
                    continue
                
                r = replay_trade(entry_price, direction, next_minutes, tp, sl)
                if r is None:
                    continue
                
                if r['exit_type'] in ('tp', 'sl'):
                    cost = COST_MARKET_EXIT
                else:
                    cost = COST_PASSIVE * 2
                
                net = r['pnl'] - cost
                pnl_list.append(net)
                dates_list.append(date_str)
            
            # Daily Sharpe
            df_sim = pd.DataFrame({'date': dates_list, 'pnl': pnl_list})
            daily = df_sim.groupby('date')['pnl'].sum()
            if len(daily) > 1 and daily.std() > 0:
                daily_sharpe = round(daily.mean() / daily.std() * np.sqrt(252), 2)
            else:
                daily_sharpe = 0
            
            pnl_arr = np.array(pnl_list)
            wr = (pnl_arr > 0).mean() if len(pnl_arr) > 0 else 0
            gp = pnl_arr[pnl_arr > 0].sum() if (pnl_arr > 0).any() else 0
            gl = abs(pnl_arr[pnl_arr <= 0].sum()) if (pnl_arr <= 0).any() else 1e-9
            pf = gp / gl if gl > 0 else 99.9
            
            key = f"SL{sl}_TP{tp}"
            results[key] = {
                'sl': sl, 'tp': tp,
                'sharpe_daily': daily_sharpe,
                'wr': round(wr, 4),
                'pf': min(round(pf, 3), 99.9),
                'avg_pnl': round(pnl_arr.mean(), 3) if len(pnl_arr) > 0 else 0,
                'total_pnl': round(pnl_arr.sum(), 1),
                'n_trades': len(pnl_arr),
            }
            
            print(f"  {daily_sharpe:>5.1f}", end='')
        print()
    
    # Champion analysis
    champion = results.get('SL15_TP30', {})
    champion_sharpe = champion.get('sharpe_daily', 0)
    
    neighbors = ['SL12_TP25', 'SL12_TP30', 'SL12_TP35', 'SL15_TP25', 'SL15_TP35',
                 'SL18_TP25', 'SL18_TP30', 'SL18_TP35']
    neighbor_sharpes = [results[k]['sharpe_daily'] for k in neighbors if k in results]
    avg_neighbor = round(np.mean(neighbor_sharpes), 3)
    plateau_ratio = round(avg_neighbor / champion_sharpe, 3) if champion_sharpe > 0 else 0
    
    assessment = 'PLATEAU (robust)' if plateau_ratio > 0.7 else 'PEAK (possible overfit)' if plateau_ratio < 0.5 else 'MODERATE'
    
    print(f"\nChampion SL15/TP30: Sharpe={champion_sharpe}, WR={champion.get('wr')}, PF={champion.get('pf')}")
    print(f"Avg neighbor Sharpe: {avg_neighbor}")
    print(f"Plateau ratio: {plateau_ratio} -> {assessment}")
    
    # Find the actual best config
    best_key = max(results.keys(), key=lambda k: results[k]['sharpe_daily'])
    best = results[best_key]
    print(f"\nBest config: {best_key} with Sharpe={best['sharpe_daily']}, WR={best['wr']}, PF={best['pf']}")
    
    # Save detailed results
    summary = {
        'grid': results,
        'champion': {'key': 'SL15_TP30', **champion},
        'best': {'key': best_key, **best},
        'plateau_ratio': plateau_ratio,
        'assessment': assessment,
        'avg_neighbor_sharpe': avg_neighbor,
    }
    
    out_file = OUTPUT_DIR / "param_sensitivity_grid.json"
    with open(out_file, 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to {out_file}")
    
    # Update the robustness results
    rob_file = OUTPUT_DIR / "robustness_results.json"
    if rob_file.exists():
        with open(rob_file) as f:
            rob = json.load(f)
        rob['parameter_sensitivity'] = summary
        with open(rob_file, 'w') as f:
            json.dump(rob, f, indent=2, default=str)
        print("Updated robustness_results.json with corrected param sensitivity")
    
    # MLflow update
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("integrated_pipeline_robustness")
        with mlflow.start_run(run_name="param_sensitivity_minute_replay"):
            mlflow.log_metric("champion_sharpe", champion_sharpe)
            mlflow.log_metric("plateau_ratio", plateau_ratio)
            mlflow.log_metric("best_sharpe", best['sharpe_daily'])
            mlflow.log_param("best_config", best_key)
            mlflow.log_param("assessment", assessment)
            mlflow.log_artifact(str(out_file))
        print("MLflow logged")

    print("\n" + "=" * 70)
    print("PARAMETER SENSITIVITY COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
