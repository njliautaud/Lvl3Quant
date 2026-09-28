#!/usr/bin/env python3
"""
Robustness Validation for Integrated Pipeline Champion (SL15/TP30)
Tasks: Bootstrap CI, Walk-Forward Stability, Parameter Sensitivity, 
       Long/Short Decomposition, Monthly P&L
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path
from collections import defaultdict

warnings.filterwarnings('ignore')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Paths ──
LVL3 = Path("/home/nick/Lvl3Quant")
TRADES_FILE = LVL3 / "output/integrated_pipeline_v2/trades_approach_b_SL15_TP30.parquet"
V1_TRADES = LVL3 / "output/integrated_pipeline_v1/best_trades.parquet"
MINUTE_BARS_DIR = LVL3 / "data/processed/mbo_minute_bars_v1"
ENTRY_PREDS = LVL3 / "output/lh_30min_deep_v1/concat_oot.npz"
FLOW_FEATURES = LVL3 / "output/long_horizon_flow_v2/enhanced_daily_features.parquet"
OUTPUT_DIR = LVL3 / "output/integrated_pipeline_robustness"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0
COST_MARKET_EXIT = COMMISSION_RT_TICKS + SPREAD_TICKS
MLFLOW_URI = "http://localhost:5000"

def compute_metrics(net_pnl_series):
    """Compute trading metrics from a series of net P&L per trade in ticks."""
    if len(net_pnl_series) == 0:
        return {'sharpe': 0, 'wr': 0, 'pf': 0, 'n_trades': 0, 'total_pnl': 0, 'avg_pnl': 0, 'sortino': 0}
    
    winners = net_pnl_series[net_pnl_series > 0]
    losers = net_pnl_series[net_pnl_series <= 0]
    
    total = net_pnl_series.sum()
    avg = net_pnl_series.mean()
    std = net_pnl_series.std()
    wr = (net_pnl_series > 0).mean()
    
    gross_profit = winners.sum() if len(winners) > 0 else 0
    gross_loss = abs(losers.sum()) if len(losers) > 0 else 1e-9
    pf = gross_profit / gross_loss if gross_loss > 0 else 99.9
    
    # Daily Sharpe (annualized)
    sharpe = (avg / std * np.sqrt(252)) if std > 0 else 0
    
    # Sortino
    downside = net_pnl_series[net_pnl_series < 0]
    downside_std = downside.std() if len(downside) > 1 else 1e-9
    sortino = (avg / downside_std * np.sqrt(252)) if downside_std > 0 else 99.9
    
    return {
        'sharpe': round(sharpe, 3),
        'sortino': min(round(sortino, 3), 99.9),
        'wr': round(wr, 4),
        'pf': min(round(pf, 3), 99.9),
        'n_trades': len(net_pnl_series),
        'total_pnl': round(total, 2),
        'avg_pnl': round(avg, 4),
    }

def compute_daily_sharpe(df):
    """Compute Sharpe from daily P&L aggregation (more realistic)."""
    daily = df.groupby('date')['net_pnl_ticks'].sum()
    if len(daily) < 2:
        return 0.0
    return round(daily.mean() / daily.std() * np.sqrt(252), 3)


def task1_bootstrap(df, n_bootstrap=1000):
    """Bootstrap confidence intervals for Sharpe ratio."""
    print("\n=== TASK 1: Bootstrap Confidence Intervals ===")
    
    rng = np.random.RandomState(42)
    n = len(df)
    sharpe_samples = []
    
    for i in range(n_bootstrap):
        idx = rng.choice(n, size=n, replace=True)
        sample = df.iloc[idx]
        # Use daily aggregation for proper Sharpe
        daily = sample.groupby('date')['net_pnl_ticks'].sum()
        if len(daily) > 1 and daily.std() > 0:
            s = daily.mean() / daily.std() * np.sqrt(252)
        else:
            s = 0
        sharpe_samples.append(s)
    
    sharpe_arr = np.array(sharpe_samples)
    ci_lower = np.percentile(sharpe_arr, 2.5)
    ci_upper = np.percentile(sharpe_arr, 97.5)
    p_above_2 = (sharpe_arr > 2).mean()
    p_above_3 = (sharpe_arr > 3).mean()
    median_sharpe = np.median(sharpe_arr)
    
    results = {
        'n_bootstrap': n_bootstrap,
        'median_sharpe': round(median_sharpe, 3),
        'mean_sharpe': round(sharpe_arr.mean(), 3),
        'ci_95_lower': round(ci_lower, 3),
        'ci_95_upper': round(ci_upper, 3),
        'p_sharpe_gt_2': round(p_above_2, 4),
        'p_sharpe_gt_3': round(p_above_3, 4),
        'std_sharpe': round(sharpe_arr.std(), 3),
    }
    
    print(f"  Bootstrap Sharpe: median={results['median_sharpe']}, mean={results['mean_sharpe']}")
    print(f"  95% CI: [{results['ci_95_lower']}, {results['ci_95_upper']}]")
    print(f"  P(Sharpe > 2) = {results['p_sharpe_gt_2']}")
    print(f"  P(Sharpe > 3) = {results['p_sharpe_gt_3']}")
    
    return results


def task2_walkforward_stability(df):
    """Split 99 trading days into 3 segments, report metrics per segment."""
    print("\n=== TASK 2: Walk-Forward Stability ===")
    
    dates = sorted(df['date'].unique())
    n_dates = len(dates)
    seg_size = n_dates // 3
    
    segments = [
        ('Segment 1 (earliest)', dates[:seg_size]),
        ('Segment 2 (middle)', dates[seg_size:2*seg_size]),
        ('Segment 3 (latest)', dates[2*seg_size:]),
    ]
    
    results = {}
    for name, seg_dates in segments:
        seg_df = df[df['date'].isin(seg_dates)]
        metrics = compute_metrics(seg_df['net_pnl_ticks'])
        daily_sharpe = compute_daily_sharpe(seg_df)
        metrics['daily_sharpe'] = daily_sharpe
        metrics['date_range'] = f"{seg_dates[0]}-{seg_dates[-1]}"
        metrics['n_days'] = len(seg_dates)
        results[name] = metrics
        print(f"  {name} ({metrics['date_range']}, {metrics['n_days']}d):")
        print(f"    Sharpe(trade)={metrics['sharpe']}, Sharpe(daily)={daily_sharpe}, WR={metrics['wr']}, PF={metrics['pf']}, trades={metrics['n_trades']}")
    
    # Stability check: max/min ratio
    daily_sharpes = [v['daily_sharpe'] for v in results.values()]
    min_s = min(daily_sharpes)
    max_s = max(daily_sharpes) 
    stability_ratio = min_s / max_s if max_s > 0 else 0
    results['stability_ratio_min_max'] = round(stability_ratio, 3)
    results['all_segments_positive'] = all(s > 0 for s in daily_sharpes)
    
    print(f"\n  Stability ratio (min/max daily Sharpe): {results['stability_ratio_min_max']}")
    print(f"  All segments positive: {results['all_segments_positive']}")
    
    return results


def task3_parameter_sensitivity(df_v1):
    """Test SL/TP grid using the v1 trade entries with tick-level MFE/MAE from v2."""
    print("\n=== TASK 3: Parameter Sensitivity (SL/TP Grid) ===")
    
    # Load v2 trades which have mfe_ticks and mae_ticks
    df_v2 = pd.read_parquet(TRADES_FILE)
    
    sl_grid = [10, 12, 15, 18, 20, 25]
    tp_grid = [20, 25, 30, 35, 40, 50]
    
    results = {}
    print(f"\n  {'SL\\TP':>6}", end='')
    for tp in tp_grid:
        print(f"  TP{tp:>3}", end='')
    print()
    
    for sl in sl_grid:
        print(f"  SL{sl:>3}", end='')
        for tp in tp_grid:
            # Simulate: if MFE >= TP (in ticks), it's a TP hit with +TP raw pnl
            # If MAE >= SL, it's SL hit with -SL raw pnl
            # If neither within the bar, use bar-close pnl (use raw_pnl_ticks from bar close)
            pnl_list = []
            for _, row in df_v2.iterrows():
                mfe = row['mfe_ticks']
                mae = row['mae_ticks']
                
                # Check which hits first (simplified: if both hit, check MFE vs MAE)
                tp_hit = mfe >= tp
                sl_hit = mae >= sl
                
                if tp_hit and sl_hit:
                    # Both hit - rough heuristic: if MFE > MAE, TP likely first
                    if mfe / tp > mae / sl:
                        raw = tp
                    else:
                        raw = -sl
                elif tp_hit:
                    raw = tp
                elif sl_hit:
                    raw = -sl
                else:
                    # Neither hit - use actual bar close P&L, capped by stops
                    raw = max(-sl, min(tp, row['raw_pnl_ticks']))
                
                net = raw - COST_MARKET_EXIT
                pnl_list.append(net)
            
            pnl_arr = np.array(pnl_list)
            m = compute_metrics(pd.Series(pnl_arr))
            daily_sharpe = compute_daily_sharpe(pd.DataFrame({
                'date': df_v2['date'], 
                'net_pnl_ticks': pnl_arr
            }))
            
            key = f"SL{sl}_TP{tp}"
            results[key] = {
                'sl': sl, 'tp': tp,
                'sharpe_trade': m['sharpe'],
                'sharpe_daily': daily_sharpe,
                'wr': m['wr'],
                'pf': m['pf'],
                'avg_pnl': m['avg_pnl'],
                'total_pnl': m['total_pnl'],
            }
            print(f"  {daily_sharpe:>5.1f}", end='')
        print()
    
    # Check if champion (SL15/TP30) is on a peak or plateau
    champion_sharpe = results['SL15_TP30']['sharpe_daily']
    neighbors = ['SL12_TP25', 'SL12_TP30', 'SL12_TP35', 'SL15_TP25', 'SL15_TP35', 
                 'SL18_TP25', 'SL18_TP30', 'SL18_TP35']
    neighbor_sharpes = [results[k]['sharpe_daily'] for k in neighbors if k in results]
    avg_neighbor = np.mean(neighbor_sharpes) if neighbor_sharpes else 0
    
    plateau_ratio = avg_neighbor / champion_sharpe if champion_sharpe > 0 else 0
    results['champion_sharpe'] = champion_sharpe
    results['avg_neighbor_sharpe'] = round(avg_neighbor, 3)
    results['plateau_ratio'] = round(plateau_ratio, 3)
    results['assessment'] = 'PLATEAU (robust)' if plateau_ratio > 0.7 else 'PEAK (possible overfit)' if plateau_ratio < 0.5 else 'MODERATE'
    
    print(f"\n  Champion SL15/TP30: daily Sharpe = {champion_sharpe}")
    print(f"  Avg neighbor Sharpe = {results['avg_neighbor_sharpe']}")
    print(f"  Plateau ratio = {results['plateau_ratio']} → {results['assessment']}")
    
    return results


def task4_long_short_decomposition(df):
    """Break down performance by direction."""
    print("\n=== TASK 4: Long vs Short Decomposition ===")
    
    results = {}
    for direction, label in [(1, 'LONG'), (-1, 'SHORT')]:
        sub = df[df['direction'] == direction]
        m = compute_metrics(sub['net_pnl_ticks'])
        daily_sharpe = compute_daily_sharpe(sub)
        m['daily_sharpe'] = daily_sharpe
        results[label] = m
        print(f"  {label}: trades={m['n_trades']}, Sharpe(daily)={daily_sharpe}, WR={m['wr']}, PF={m['pf']}, avg_pnl={m['avg_pnl']}")
    
    # Which side carries more weight?
    long_pnl = results['LONG']['total_pnl']
    short_pnl = results['SHORT']['total_pnl']
    total = long_pnl + short_pnl
    results['long_pnl_share'] = round(long_pnl / total, 3) if total != 0 else 0
    results['short_pnl_share'] = round(short_pnl / total, 3) if total != 0 else 0
    print(f"\n  P&L share: Long={results['long_pnl_share']:.1%}, Short={results['short_pnl_share']:.1%}")
    
    return results


def task5_monthly_pnl(df):
    """Report P&L by calendar month."""
    print("\n=== TASK 5: Monthly P&L Consistency ===")
    
    df = df.copy()
    df['month'] = df['date'].str[:6]
    
    monthly = df.groupby('month').agg(
        n_trades=('net_pnl_ticks', 'count'),
        total_pnl=('net_pnl_ticks', 'sum'),
        avg_pnl=('net_pnl_ticks', 'mean'),
        wr=('net_pnl_ticks', lambda x: (x > 0).mean()),
    ).reset_index()
    
    results = {}
    positive_months = 0
    negative_months = 0
    
    print(f"  {'Month':>8}  {'Trades':>6}  {'Total PnL':>10}  {'Avg PnL':>8}  {'WR':>6}")
    for _, row in monthly.iterrows():
        is_pos = row['total_pnl'] > 0
        if is_pos:
            positive_months += 1
        else:
            negative_months += 1
        
        marker = '+' if is_pos else '-'
        print(f"  {row['month']:>8}  {row['n_trades']:>6}  {row['total_pnl']:>10.1f}  {row['avg_pnl']:>8.2f}  {row['wr']:>5.1%}  {marker}")
        
        results[row['month']] = {
            'n_trades': int(row['n_trades']),
            'total_pnl': round(row['total_pnl'], 2),
            'avg_pnl': round(row['avg_pnl'], 3),
            'wr': round(row['wr'], 4),
            'positive': is_pos,
        }
    
    results['summary'] = {
        'positive_months': positive_months,
        'negative_months': negative_months,
        'consistency': f"{positive_months}/{positive_months + negative_months}",
    }
    print(f"\n  Positive months: {positive_months}/{positive_months + negative_months}")
    
    return results


def main():
    print("=" * 70)
    print("ROBUSTNESS VALIDATION — Integrated Pipeline Champion (SL15/TP30)")
    print("=" * 70)
    
    # Load trades
    df = pd.read_parquet(TRADES_FILE)
    print(f"Loaded {len(df)} trades, {df['date'].nunique()} trading days")
    print(f"Date range: {df['date'].min()} to {df['date'].max()}")
    
    # Also load v1 trades for some columns
    df_v1 = pd.read_parquet(V1_TRADES)
    
    # Run all tasks
    r1 = task1_bootstrap(df)
    r2 = task2_walkforward_stability(df)
    r3 = task3_parameter_sensitivity(df_v1)
    r4 = task4_long_short_decomposition(df)
    r5 = task5_monthly_pnl(df)
    
    # Aggregate results
    all_results = {
        'bootstrap_ci': r1,
        'walkforward_stability': r2,
        'parameter_sensitivity': r3,
        'long_short_decomposition': r4,
        'monthly_pnl': r5,
    }
    
    # Save results
    out_file = OUTPUT_DIR / "robustness_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")
    
    # Log to MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("integrated_pipeline_robustness")
        with mlflow.start_run(run_name="robustness_validation_sl15_tp30"):
            # Bootstrap
            mlflow.log_metric("bootstrap_median_sharpe", r1['median_sharpe'])
            mlflow.log_metric("bootstrap_ci_lower", r1['ci_95_lower'])
            mlflow.log_metric("bootstrap_ci_upper", r1['ci_95_upper'])
            mlflow.log_metric("bootstrap_p_sharpe_gt_2", r1['p_sharpe_gt_2'])
            mlflow.log_metric("bootstrap_p_sharpe_gt_3", r1['p_sharpe_gt_3'])
            
            # Walk-forward
            for seg_name, seg_data in r2.items():
                if isinstance(seg_data, dict) and 'daily_sharpe' in seg_data:
                    safe = seg_name.replace(' ', '_').replace('(', '').replace(')', '')
                    mlflow.log_metric(f"wf_{safe}_sharpe", seg_data['daily_sharpe'])
                    mlflow.log_metric(f"wf_{safe}_wr", seg_data['wr'])
                    mlflow.log_metric(f"wf_{safe}_pf", seg_data['pf'])
            mlflow.log_metric("wf_stability_ratio", r2.get('stability_ratio_min_max', 0))
            
            # Parameter sensitivity
            mlflow.log_metric("param_champion_sharpe", r3.get('champion_sharpe', 0))
            mlflow.log_metric("param_plateau_ratio", r3.get('plateau_ratio', 0))
            
            # Long/Short
            for side in ['LONG', 'SHORT']:
                if side in r4:
                    mlflow.log_metric(f"{side.lower()}_sharpe", r4[side]['daily_sharpe'])
                    mlflow.log_metric(f"{side.lower()}_wr", r4[side]['wr'])
                    mlflow.log_metric(f"{side.lower()}_pf", r4[side]['pf'])
            
            # Monthly
            monthly_summary = r5.get('summary', {})
            mlflow.log_metric("positive_months", monthly_summary.get('positive_months', 0))
            mlflow.log_metric("negative_months", monthly_summary.get('negative_months', 0))
            
            mlflow.log_artifact(str(out_file))
            
        print("MLflow run logged successfully")
    
    print("\n" + "=" * 70)
    print("ROBUSTNESS VALIDATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
