#!/usr/bin/env python3
"""
Signal Decomposition — Feature Importance & Layer Attribution
Understand WHAT drives the integrated pipeline's edge.
"""

import os, sys, json, warnings
import numpy as np
import pandas as pd
from pathlib import Path

warnings.filterwarnings('ignore')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False

# ── Paths ──
LVL3 = Path("/home/nick/Lvl3Quant")
TRADES_FILE = LVL3 / "output/integrated_pipeline_v2/trades_approach_b_SL15_TP30.parquet"
V1_TRADES = LVL3 / "output/integrated_pipeline_v1/best_trades.parquet"
ENTRY_PREDS = LVL3 / "output/lh_30min_deep_v1/concat_oot.npz"
FLOW_FEATURES = LVL3 / "output/long_horizon_flow_v2/enhanced_daily_features.parquet"
EXEC_LGBM_DIR = LVL3 / "output/exec_lgbm_v1"
OUTPUT_DIR = LVL3 / "output/integrated_pipeline_decomposition"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COMMISSION_RT_TICKS = 0.376
SPREAD_TICKS = 1.0
COST_MARKET_EXIT = COMMISSION_RT_TICKS + SPREAD_TICKS
MLFLOW_URI = "http://localhost:5000"

def compute_daily_sharpe(dates, pnl):
    """Compute daily Sharpe from arrays."""
    df = pd.DataFrame({'date': dates, 'pnl': pnl})
    daily = df.groupby('date')['pnl'].sum()
    if len(daily) < 2 or daily.std() == 0:
        return 0.0
    return round(daily.mean() / daily.std() * np.sqrt(252), 3)


def compute_metrics_quick(pnl_arr):
    """Quick metrics from P&L array."""
    if len(pnl_arr) == 0:
        return {'sharpe': 0, 'wr': 0, 'pf': 0, 'avg_pnl': 0, 'total_pnl': 0}
    pnl = np.array(pnl_arr)
    wr = (pnl > 0).mean()
    gp = pnl[pnl > 0].sum() if (pnl > 0).any() else 0
    gl = abs(pnl[pnl <= 0].sum()) if (pnl <= 0).any() else 1e-9
    pf = gp / gl if gl > 0 else 99.9
    avg = pnl.mean()
    std = pnl.std()
    sharpe = avg / std * np.sqrt(252) if std > 0 else 0
    return {
        'sharpe_trade': round(sharpe, 3),
        'wr': round(wr, 4),
        'pf': min(round(pf, 3), 99.9),
        'avg_pnl': round(avg, 4),
        'total_pnl': round(pnl.sum(), 2),
        'n_trades': len(pnl),
    }


def task1_feature_importance():
    """Extract feature importances from the LightGBM models."""
    print("\n=== TASK 2.1: Feature Importance ===")
    
    results = {}
    
    # 30-min LightGBM feature importance (from exec_lgbm_v1)
    feat_imp_file = EXEC_LGBM_DIR / "feature_importance.json"
    if feat_imp_file.exists():
        with open(feat_imp_file) as f:
            fi = json.load(f)
        
        per_feat = fi.get('per_feature', {})
        # Sort by mean importance
        sorted_feats = sorted(per_feat.items(), key=lambda x: x[1].get('mean', 0), reverse=True)
        
        print("\n  Top 20 LightGBM (30-min entry) Features:")
        top20 = []
        for i, (fname, stats) in enumerate(sorted_feats[:20]):
            mean_imp = stats.get('mean', 0)
            std_imp = stats.get('std', 0)
            print(f"    {i+1:>2}. {fname:<30s} importance={mean_imp:.2f} (±{std_imp:.2f})")
            top20.append({'feature': fname, 'mean_importance': mean_imp, 'std': std_imp})
        
        results['lgbm_30min_top20'] = top20
        results['lgbm_30min_total_features'] = len(per_feat)
    else:
        print("  No feature importance file found for exec_lgbm_v1")
        
        # Try loading a model directly
        model_files = sorted(EXEC_LGBM_DIR.glob("fold_*_model.txt"))
        if model_files and LGB_AVAILABLE:
            print(f"  Loading {len(model_files)} fold models to extract importance...")
            importances = {}
            for mf in model_files[:5]:  # sample 5 folds
                model = lgb.Booster(model_file=str(mf))
                fi = model.feature_importance(importance_type='gain')
                fn = model.feature_name()
                for name, imp in zip(fn, fi):
                    if name not in importances:
                        importances[name] = []
                    importances[name].append(imp)
            
            avg_imp = {k: np.mean(v) for k, v in importances.items()}
            sorted_feats = sorted(avg_imp.items(), key=lambda x: x[1], reverse=True)
            
            print("\n  Top 20 LightGBM Features (from fold models):")
            top20 = []
            for i, (fname, imp) in enumerate(sorted_feats[:20]):
                std_imp = np.std(importances[fname])
                print(f"    {i+1:>2}. {fname:<30s} importance={imp:.2f} (±{std_imp:.2f})")
                top20.append({'feature': fname, 'mean_importance': round(imp, 2), 'std': round(std_imp, 2)})
            
            results['lgbm_30min_top20'] = top20
            results['lgbm_30min_total_features'] = len(avg_imp)
    
    # Daily flow model features
    flow_df = pd.read_parquet(FLOW_FEATURES)
    feature_cols = [c for c in flow_df.columns if c not in ['date', 'target', 'direction', 'flow_direction']]
    results['flow_model_features'] = feature_cols[:30]
    results['flow_model_n_features'] = len(feature_cols)
    print(f"\n  Daily flow model has {len(feature_cols)} features")
    print(f"  Sample features: {feature_cols[:10]}")
    
    return results


def task2_layer_attribution():
    """
    Test: how much edge comes from flow direction vs 30-min prediction?
    - Random flow + real 30-min prediction
    - Real flow + random 30-min prediction
    """
    print("\n=== TASK 2.2: Layer Attribution (Flow vs Entry Signal) ===")
    
    # Load the actual trades
    df = pd.read_parquet(TRADES_FILE)
    df_v1 = pd.read_parquet(V1_TRADES)
    
    # Get the actual overall performance
    actual_daily_sharpe = compute_daily_sharpe(df['date'], df['net_pnl_ticks'])
    actual_metrics = compute_metrics_quick(df['net_pnl_ticks'].values)
    print(f"\n  Actual performance: Sharpe(daily)={actual_daily_sharpe}, WR={actual_metrics['wr']}, PF={actual_metrics['pf']}")
    
    # Load entry predictions to understand the prediction structure
    ep = np.load(ENTRY_PREDS)
    pred_dates = ep['dates']
    pred_confs = ep['confs']
    
    # Load flow features to get flow direction per date
    flow_df = pd.read_parquet(FLOW_FEATURES)
    
    results = {
        'actual': {
            'sharpe_daily': actual_daily_sharpe,
            **actual_metrics,
        }
    }
    
    rng = np.random.RandomState(42)
    n_simulations = 200
    
    # ── TEST A: Random flow direction + real 30-min prediction ──
    # Randomly assign flow direction per day, keep the same trade entries & outcomes
    print("\n  Test A: Random flow direction + real 30-min signal")
    sharpe_samples_a = []
    
    unique_dates = df['date'].unique()
    
    for sim in range(n_simulations):
        # Randomly flip each day's flow direction (50/50)
        random_directions = rng.choice([-1, 1], size=len(unique_dates))
        date_to_random_dir = dict(zip(unique_dates, random_directions))
        
        # For each trade, check if the random flow agrees with the trade direction
        # If it doesn't agree, that trade wouldn't have been taken
        sim_pnl = []
        sim_dates = []
        for _, row in df.iterrows():
            rand_dir = date_to_random_dir[row['date']]
            # The trade was taken with direction=row['direction']
            # The flow filter would only let through trades matching the flow
            # So with random flow, ~50% of trades get through with correct direction
            if rand_dir == row['direction']:
                sim_pnl.append(row['net_pnl_ticks'])
                sim_dates.append(row['date'])
        
        if len(sim_pnl) > 10:
            s = compute_daily_sharpe(sim_dates, sim_pnl)
            sharpe_samples_a.append(s)
    
    sharpe_a = np.array(sharpe_samples_a)
    results['test_a_random_flow'] = {
        'description': 'Random flow direction + real 30-min signal',
        'median_sharpe': round(np.median(sharpe_a), 3),
        'mean_sharpe': round(np.mean(sharpe_a), 3),
        'std_sharpe': round(np.std(sharpe_a), 3),
        'p25': round(np.percentile(sharpe_a, 25), 3),
        'p75': round(np.percentile(sharpe_a, 75), 3),
        'avg_trades_kept': round(len(df) * 0.5, 0),
    }
    print(f"    Median Sharpe: {results['test_a_random_flow']['median_sharpe']}")
    print(f"    Mean Sharpe: {results['test_a_random_flow']['mean_sharpe']} (±{results['test_a_random_flow']['std_sharpe']})")
    
    # ── TEST B: Real flow direction + random 30-min prediction ──
    # Keep the flow direction but randomize which bars get traded
    print("\n  Test B: Real flow direction + random 30-min signal")
    sharpe_samples_b = []
    
    # Get the actual number of bars per day from predictions
    for sim in range(n_simulations):
        sim_pnl = []
        sim_dates = []
        
        for date in unique_dates:
            day_trades = df[df['date'] == date]
            n_day_trades = len(day_trades)
            
            if n_day_trades == 0:
                continue
            
            # Randomly select same number of trades but shuffle their P&L
            # This preserves the daily flow filter but randomizes entry selection
            shuffled_pnl = day_trades['net_pnl_ticks'].values.copy()
            rng.shuffle(shuffled_pnl)
            
            # Alternatively: randomly flip some trade signs to simulate random entry
            for pnl in shuffled_pnl:
                # 50% chance of flipping the trade outcome (simulating random entry timing)
                if rng.random() < 0.5:
                    # flip: a winning trade becomes losing and vice versa, adjusted for costs
                    raw_approx = pnl + COST_MARKET_EXIT
                    flipped = -raw_approx - COST_MARKET_EXIT
                    sim_pnl.append(flipped)
                else:
                    sim_pnl.append(pnl)
                sim_dates.append(date)
        
        if len(sim_pnl) > 10:
            s = compute_daily_sharpe(sim_dates, sim_pnl)
            sharpe_samples_b.append(s)
    
    sharpe_b = np.array(sharpe_samples_b)
    results['test_b_random_entry'] = {
        'description': 'Real flow direction + random 30-min signal',
        'median_sharpe': round(np.median(sharpe_b), 3),
        'mean_sharpe': round(np.mean(sharpe_b), 3),
        'std_sharpe': round(np.std(sharpe_b), 3),
        'p25': round(np.percentile(sharpe_b, 25), 3),
        'p75': round(np.percentile(sharpe_b, 75), 3),
    }
    print(f"    Median Sharpe: {results['test_b_random_entry']['median_sharpe']}")
    print(f"    Mean Sharpe: {results['test_b_random_entry']['mean_sharpe']} (±{results['test_b_random_entry']['std_sharpe']})")
    
    # ── TEST C: Both random (pure noise baseline) ──
    print("\n  Test C: Both random (baseline)")
    sharpe_samples_c = []
    for sim in range(n_simulations):
        # Random flow + random entry = randomly keep ~50% of trades + flip 50%
        sim_pnl = []
        sim_dates = []
        for _, row in df.iterrows():
            if rng.random() < 0.5:
                continue
            pnl = row['net_pnl_ticks']
            if rng.random() < 0.5:
                raw_approx = pnl + COST_MARKET_EXIT
                pnl = -raw_approx - COST_MARKET_EXIT
            sim_pnl.append(pnl)
            sim_dates.append(row['date'])
        
        if len(sim_pnl) > 10:
            s = compute_daily_sharpe(sim_dates, sim_pnl)
            sharpe_samples_c.append(s)
    
    sharpe_c = np.array(sharpe_samples_c)
    results['test_c_both_random'] = {
        'description': 'Both random (pure noise baseline)',
        'median_sharpe': round(np.median(sharpe_c), 3),
        'mean_sharpe': round(np.mean(sharpe_c), 3),
        'std_sharpe': round(np.std(sharpe_c), 3),
    }
    print(f"    Median Sharpe: {results['test_c_both_random']['median_sharpe']}")
    
    # ── Attribution Summary ──
    actual_s = actual_daily_sharpe
    flow_only = results['test_a_random_flow']['median_sharpe']
    entry_only = results['test_b_random_entry']['median_sharpe']
    noise = results['test_c_both_random']['median_sharpe']
    
    # Edge attribution
    total_edge = actual_s - noise
    flow_contribution = actual_s - flow_only  # removing flow hurts by this much
    entry_contribution = actual_s - entry_only  # removing entry hurts by this much
    
    results['attribution'] = {
        'actual_sharpe': actual_s,
        'noise_baseline': noise,
        'total_edge_over_noise': round(total_edge, 3),
        'flow_contribution': round(flow_contribution, 3),
        'entry_contribution': round(entry_contribution, 3),
        'flow_pct_of_edge': round(flow_contribution / total_edge * 100, 1) if total_edge > 0 else 0,
        'entry_pct_of_edge': round(entry_contribution / total_edge * 100, 1) if total_edge > 0 else 0,
    }
    
    print(f"\n  === Attribution Summary ===")
    print(f"  Actual Sharpe: {actual_s}")
    print(f"  Noise baseline: {noise}")
    print(f"  Total edge: {round(total_edge, 3)}")
    print(f"  Flow direction contribution: {round(flow_contribution, 3)} ({results['attribution']['flow_pct_of_edge']}% of edge)")
    print(f"  Entry signal contribution: {round(entry_contribution, 3)} ({results['attribution']['entry_pct_of_edge']}% of edge)")
    
    return results


def main():
    print("=" * 70)
    print("SIGNAL DECOMPOSITION — Integrated Pipeline (SL15/TP30)")
    print("=" * 70)
    
    r1 = task1_feature_importance()
    r2 = task2_layer_attribution()
    
    all_results = {
        'feature_importance': r1,
        'layer_attribution': r2,
    }
    
    # Save
    out_file = OUTPUT_DIR / "decomposition_results.json"
    with open(out_file, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    print(f"\nResults saved to {out_file}")
    
    # MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("integrated_pipeline_decomposition")
        with mlflow.start_run(run_name="signal_decomposition_sl15_tp30"):
            attr = r2.get('attribution', {})
            mlflow.log_metric("actual_sharpe", attr.get('actual_sharpe', 0))
            mlflow.log_metric("noise_baseline", attr.get('noise_baseline', 0))
            mlflow.log_metric("flow_contribution_pct", attr.get('flow_pct_of_edge', 0))
            mlflow.log_metric("entry_contribution_pct", attr.get('entry_pct_of_edge', 0))
            
            mlflow.log_metric("random_flow_median_sharpe", r2.get('test_a_random_flow', {}).get('median_sharpe', 0))
            mlflow.log_metric("random_entry_median_sharpe", r2.get('test_b_random_entry', {}).get('median_sharpe', 0))
            
            mlflow.log_artifact(str(out_file))
        print("MLflow run logged successfully")
    
    print("\n" + "=" * 70)
    print("SIGNAL DECOMPOSITION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
