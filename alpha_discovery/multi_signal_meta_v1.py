#!/usr/bin/env python3
"""
multi_signal_meta_v1.py — Multi-Signal Ensemble Meta-Model

Combines:
1. 3-day flow direction (daily directional bias) 
2. 30-min LightGBM signal (entry timing)
3. Mid-trade classifier (position management)

Architecture: The 3-day flow gives daily directional bias, the 30-min gives 
entry timing, the mid-trade manages the position.

HC #0: SLIDING window walk-forward ONLY
MLflow logging mandatory.
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from pathlib import Path
from datetime import datetime
from scipy import stats

warnings.filterwarnings('ignore')

try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# ── Paths ──
FLOW_FEATURES = Path("/home/nick/Lvl3Quant/output/long_horizon_flow_v2/enhanced_daily_features.parquet")
MIDTRADE_FEATURES = Path("/home/nick/Lvl3Quant/output/midtrade_thesis_v1/trade_tick_features.parquet")
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/multi_signal_meta_v1")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COST_RT_TICKS = 2.376
MLFLOW_URI = "http://localhost:5000"


def load_flow_signals():
    """Load 3-day flow features and produce daily directional signals via walk-forward."""
    df = pd.read_parquet(FLOW_FEATURES)
    
    exclude_prefixes = ['fwd_', 'open', 'close', 'high', 'low', 'session_vwap', 'trade_count', 'date']
    feature_cols = [c for c in df.columns if not any(c.startswith(p) for p in exclude_prefixes)]
    
    TRAIN_DAYS = 40
    SLIDE_DAYS = 5
    OOT_DAYS = 5
    
    n = len(df)
    daily_signals = {}
    
    start = TRAIN_DAYS
    while start + OOT_DAYS <= n:
        train_slice = slice(start - TRAIN_DAYS, start)
        test_end = min(start + OOT_DAYS, n)
        test_slice = slice(start, test_end)
        
        X_train = np.nan_to_num(df.iloc[train_slice][feature_cols].values, 0)
        y_train = df.iloc[train_slice]['fwd_direction_3d'].values
        X_test = np.nan_to_num(df.iloc[test_slice][feature_cols].values, 0)
        
        valid_train = ~np.isnan(y_train)
        if valid_train.sum() < 20:
            start += SLIDE_DAYS
            continue
        
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=10, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
            verbose=-1, random_state=42
        )
        model.fit(X_train[valid_train], y_train[valid_train])
        
        probs = model.predict_proba(X_test)[:, 1]
        dates = pd.to_datetime(df.iloc[test_slice]["date"]).dt.strftime("%Y%m%d").values
        
        for j in range(len(probs)):
            date = str(dates[j])
            daily_signals[date] = {
                'flow_prob_up': float(probs[j]),
                'flow_bias': float(2 * probs[j] - 1),  # -1 to +1
            }
        
        start += SLIDE_DAYS
    
    print(f"Generated flow signals for {len(daily_signals)} days")
    return daily_signals


def build_ensemble_features(midtrade_df, flow_signals):
    """Combine mid-trade features with daily flow bias."""
    df = midtrade_df.copy()
    
    # Add flow signals
    df['flow_prob_up'] = df['date'].map(lambda d: flow_signals.get(str(d), {}).get('flow_prob_up', 0.5))
    df['flow_bias'] = df['date'].map(lambda d: flow_signals.get(str(d), {}).get('flow_bias', 0.0))
    
    # Interaction: flow bias * trade direction 
    # (positive = trade agrees with daily flow, negative = disagrees)
    df['flow_agreement'] = df['flow_bias'] * df['direction']
    
    # Flow confidence
    df['flow_confidence'] = np.abs(df['flow_bias'])
    
    # Strong flow indicators
    df['strong_flow_up'] = (df['flow_prob_up'] > 0.6).astype(float)
    df['strong_flow_down'] = (df['flow_prob_up'] < 0.4).astype(float)
    
    # Trade aligns with strong flow
    df['aligned_with_flow'] = (
        ((df['direction'] == 1) & (df['flow_prob_up'] > 0.55)) |
        ((df['direction'] == -1) & (df['flow_prob_up'] < 0.45))
    ).astype(float)
    
    return df


def walk_forward_meta(df, feature_cols, target='winner', min_train=30):
    """Meta-model walk-forward."""
    dates = df['date'].unique()
    n_dates = len(dates)
    
    results = []
    
    for i in range(min_train, n_dates):
        test_date = dates[i]
        train_dates = dates[max(0, i - min_train):i]
        
        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'] == test_date
        
        X_train = np.nan_to_num(df.loc[train_mask, feature_cols].values, 0)
        y_train = df.loc[train_mask, target].values
        X_test = np.nan_to_num(df.loc[test_mask, feature_cols].values, 0)
        y_test = df.loc[test_mask, target].values
        test_data = df.loc[test_mask]
        
        if len(X_train) < 10 or len(X_test) == 0:
            continue
        
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=5, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
            verbose=-1, random_state=42
        )
        
        try:
            model.fit(X_train, y_train)
        except:
            continue
        
        probs = model.predict_proba(X_test)[:, 1]
        
        for j in range(len(probs)):
            results.append({
                'date': test_date,
                'prob_winner': float(probs[j]),
                'actual_winner': int(y_test[j]),
                'exit_ticks': float(test_data.iloc[j]['exit_ticks']),
                'direction': int(test_data.iloc[j]['direction']),
                'flow_agreement': float(test_data.iloc[j].get('flow_agreement', 0)),
                'flow_bias': float(test_data.iloc[j].get('flow_bias', 0)),
            })
    
    return pd.DataFrame(results)


def evaluate_strategies(results_df, name):
    """Evaluate different ensemble strategies."""
    if len(results_df) == 0:
        return {}
    
    baseline_pnl = results_df['exit_ticks'].sum()
    n_total = len(results_df)
    
    strategies = {}
    
    # Strategy 1: Flow-filtered entries (only trade when aligned with 3-day flow)
    aligned = results_df[results_df['flow_agreement'] > 0]
    if len(aligned) > 0:
        pnl = aligned['exit_ticks'].sum()
        n = len(aligned)
        wr = (aligned['exit_ticks'] > 0).mean()
        avg = aligned['exit_ticks'].mean()
        std = aligned['exit_ticks'].std()
        sharpe = avg / std * np.sqrt(252) if std > 0 else 0
        strategies['flow_aligned_only'] = {
            'n_trades': int(n),
            'filter_rate': float(1 - n/n_total),
            'total_pnl': float(pnl),
            'avg_pnl': float(avg),
            'win_rate': float(wr),
            'sharpe': float(sharpe),
            'improvement': float(pnl - baseline_pnl * n / n_total),
        }
        print(f"  [{name}] Flow-aligned: N={n}, WR={wr:.1%}, Sharpe={sharpe:.2f}, "
              f"PnL={pnl:.1f} vs pro-rata baseline {baseline_pnl * n / n_total:.1f}")
    
    # Strategy 2: Strong flow filter (only trade when flow confidence > 0.1)
    strong_flow = results_df[results_df['flow_agreement'] > 0.1]
    if len(strong_flow) > 0:
        pnl = strong_flow['exit_ticks'].sum()
        n = len(strong_flow)
        wr = (strong_flow['exit_ticks'] > 0).mean()
        avg = strong_flow['exit_ticks'].mean()
        std = strong_flow['exit_ticks'].std()
        sharpe = avg / std * np.sqrt(252) if std > 0 else 0
        strategies['strong_flow_aligned'] = {
            'n_trades': int(n),
            'filter_rate': float(1 - n/n_total),
            'total_pnl': float(pnl),
            'avg_pnl': float(avg),
            'win_rate': float(wr),
            'sharpe': float(sharpe),
        }
        print(f"  [{name}] Strong flow: N={n}, WR={wr:.1%}, Sharpe={sharpe:.2f}, PnL={pnl:.1f}")
    
    # Strategy 3: Meta-model confidence filter
    for thresh in [0.5, 0.55, 0.6]:
        confident = results_df[results_df['prob_winner'] > thresh]
        if len(confident) > 0:
            pnl = confident['exit_ticks'].sum()
            n = len(confident)
            wr = (confident['exit_ticks'] > 0).mean()
            avg = confident['exit_ticks'].mean()
            std = confident['exit_ticks'].std()
            sharpe = avg / std * np.sqrt(252) if std > 0 else 0
            strategies[f'meta_conf_{thresh}'] = {
                'n_trades': int(n),
                'filter_rate': float(1 - n/n_total),
                'total_pnl': float(pnl),
                'avg_pnl': float(avg),
                'win_rate': float(wr),
                'sharpe': float(sharpe),
            }
            print(f"  [{name}] Meta>{thresh}: N={n}, WR={wr:.1%}, Sharpe={sharpe:.2f}, PnL={pnl:.1f}")
    
    # Strategy 4: Combined (flow-aligned AND meta-confident)
    combined = results_df[(results_df['flow_agreement'] > 0) & (results_df['prob_winner'] > 0.55)]
    if len(combined) > 0:
        pnl = combined['exit_ticks'].sum()
        n = len(combined)
        wr = (combined['exit_ticks'] > 0).mean()
        avg = combined['exit_ticks'].mean()
        std = combined['exit_ticks'].std()
        sharpe = avg / std * np.sqrt(252) if std > 0 else 0
        strategies['flow_plus_meta'] = {
            'n_trades': int(n),
            'filter_rate': float(1 - n/n_total),
            'total_pnl': float(pnl),
            'avg_pnl': float(avg),
            'win_rate': float(wr),
            'sharpe': float(sharpe),
        }
        print(f"  [{name}] Flow+Meta: N={n}, WR={wr:.1%}, Sharpe={sharpe:.2f}, PnL={pnl:.1f}")
    
    strategies['baseline'] = {
        'n_trades': int(n_total),
        'total_pnl': float(baseline_pnl),
        'avg_pnl': float(baseline_pnl / n_total),
        'win_rate': float((results_df['exit_ticks'] > 0).mean()),
    }
    
    return strategies


def main():
    t0 = time.time()
    print("=" * 70)
    print("TASK 3: Multi-Signal Ensemble Meta-Model")
    print("=" * 70)
    
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("multi_signal_meta_v1")
        run = mlflow.start_run(run_name=f"meta_v1_{datetime.now().strftime('%Y%m%d_%H%M')}")
        print(f"MLflow run: {run.info.run_id}")
    
    # Step 1: Generate daily flow signals
    print("\n[1] Generating 3-day flow directional signals...")
    flow_signals = load_flow_signals()
    
    # Step 2: Load mid-trade features
    print("\n[2] Loading mid-trade features...")
    midtrade_df = pd.read_parquet(MIDTRADE_FEATURES)
    print(f"Loaded {len(midtrade_df)} trades")
    
    # Step 3: Build ensemble features
    print("\n[3] Building ensemble features...")
    ensemble_df = build_ensemble_features(midtrade_df, flow_signals)
    
    # Check how many trades have flow data
    has_flow = ensemble_df['flow_bias'] != 0
    print(f"Trades with flow signals: {has_flow.sum()} / {len(ensemble_df)}")
    
    # Feature sets
    exclude = ['date', 'direction', 'fill_price', 'time_of_day_min', 'exit_type', 'exit_ticks', 'winner', 'pred_magnitude']
    
    # A) Mid-trade features only (baseline)
    midtrade_feats = [c for c in midtrade_df.columns if c not in exclude]
    
    # B) Mid-trade + flow features (ensemble)
    ensemble_feats = [c for c in ensemble_df.columns if c not in exclude]
    
    print(f"Mid-trade features: {len(midtrade_feats)}")
    print(f"Ensemble features: {len(ensemble_feats)}")
    
    # Run for each checkpoint
    all_results = {}
    
    for checkpoint_sec in [10, 30]:  # Focus on best-performing checkpoints
        print(f"\n{'='*50}")
        print(f"Checkpoint: T+{checkpoint_sec}s")
        print(f"{'='*50}")
        
        # Filter to relevant features for this checkpoint
        cp_midtrade = [c for c in midtrade_feats 
                      if any(c.startswith(f'cp{cp}s_') for cp in ['5','10','15','30'] if int(cp) <= checkpoint_sec)
                      or not any(c.startswith(f'cp{cp}s_') for cp in ['5','10','15','30'])]
        
        cp_ensemble = [c for c in ensemble_feats 
                      if any(c.startswith(f'cp{cp}s_') for cp in ['5','10','15','30'] if int(cp) <= checkpoint_sec)
                      or not any(c.startswith(f'cp{cp}s_') for cp in ['5','10','15','30'])]
        
        # Baseline: mid-trade only
        print(f"\n  --- Baseline (mid-trade only) ---")
        baseline_results = walk_forward_meta(midtrade_df, cp_midtrade, 'winner', min_train=30)
        if len(baseline_results) > 0:
            from sklearn.metrics import roc_auc_score
            acc = np.mean((baseline_results['prob_winner'] > 0.5).astype(int) == baseline_results['actual_winner'])
            try:
                auc = roc_auc_score(baseline_results['actual_winner'], baseline_results['prob_winner'])
            except:
                auc = 0.5
            print(f"  Baseline: ACC={acc:.4f}, AUC={auc:.4f}, N={len(baseline_results)}")
            baseline_strats = evaluate_strategies(baseline_results, 'Baseline')
        
        # Ensemble: mid-trade + flow
        print(f"\n  --- Ensemble (mid-trade + flow) ---")
        ensemble_results = walk_forward_meta(ensemble_df, cp_ensemble, 'winner', min_train=30)
        if len(ensemble_results) > 0:
            acc = np.mean((ensemble_results['prob_winner'] > 0.5).astype(int) == ensemble_results['actual_winner'])
            try:
                auc = roc_auc_score(ensemble_results['actual_winner'], ensemble_results['prob_winner'])
            except:
                auc = 0.5
            print(f"  Ensemble: ACC={acc:.4f}, AUC={auc:.4f}, N={len(ensemble_results)}")
            ensemble_strats = evaluate_strategies(ensemble_results, 'Ensemble')
            
            all_results[f't{checkpoint_sec}'] = {
                'baseline_auc': float(roc_auc_score(baseline_results['actual_winner'], baseline_results['prob_winner'])) if len(baseline_results) > 0 else 0,
                'ensemble_auc': float(auc),
                'baseline_strategies': baseline_strats,
                'ensemble_strategies': ensemble_strats,
            }
    
    # Summary
    print(f"\n{'='*70}")
    print("ENSEMBLE SUMMARY")
    print(f"{'='*70}")
    
    for cp, data in all_results.items():
        print(f"\n{cp}:")
        print(f"  Baseline AUC: {data['baseline_auc']:.4f}")
        print(f"  Ensemble AUC: {data['ensemble_auc']:.4f}")
        print(f"  AUC lift: {data['ensemble_auc'] - data['baseline_auc']:+.4f}")
        
        # Compare best strategies
        baseline_best = max(data['baseline_strategies'].values(), 
                          key=lambda x: x.get('sharpe', 0) if x.get('n_trades', 0) >= 20 else -999)
        ensemble_best = max(data['ensemble_strategies'].values(),
                          key=lambda x: x.get('sharpe', 0) if x.get('n_trades', 0) >= 20 else -999)
        
        print(f"  Best baseline Sharpe: {baseline_best.get('sharpe', 0):.2f} "
              f"(N={baseline_best.get('n_trades', 0)}, WR={baseline_best.get('win_rate', 0):.1%})")
        print(f"  Best ensemble Sharpe: {ensemble_best.get('sharpe', 0):.2f} "
              f"(N={ensemble_best.get('n_trades', 0)}, WR={ensemble_best.get('win_rate', 0):.1%})")
    
    elapsed = time.time() - t0
    
    if MLFLOW_AVAILABLE:
        for cp, data in all_results.items():
            mlflow.log_metric(f"{cp}_baseline_auc", data['baseline_auc'])
            mlflow.log_metric(f"{cp}_ensemble_auc", data['ensemble_auc'])
        mlflow.log_metric("elapsed_seconds", elapsed)
        mlflow.end_run()
        print(f"\nMLflow run logged")
    
    output = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'n_flow_days': len(flow_signals) if 'flow_signals' in dir() else 0,
        'results': all_results,
    }
    
    # Convert numpy types for JSON
    def convert(obj):
        if isinstance(obj, (np.integer,)): return int(obj)
        if isinstance(obj, (np.floating,)): return float(obj)
        if isinstance(obj, np.ndarray): return obj.tolist()
        if isinstance(obj, np.bool_): return bool(obj)
        return obj
    
    out_file = OUTPUT_DIR / 'results.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=convert)
    print(f"\nSaved to {out_file}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
