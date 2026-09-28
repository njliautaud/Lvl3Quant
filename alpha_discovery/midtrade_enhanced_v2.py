#!/usr/bin/env python3
"""
midtrade_enhanced_v2.py — Enhanced Mid-Trade Classifier

Improvements over v1:
1. Additional features: spread dynamics, trade imbalance velocity, queue position changes
2. Small MLP model (GPU) for non-linear pattern capture alongside LightGBM
3. Continuous confidence scoring instead of binary cut/hold
4. MLflow logging (mandatory)

HC #648: Sample size awareness
HC #649: Dynamic > static
HC #0: SLIDING window walk-forward ONLY
"""

import os, sys, json, warnings, time
import numpy as np
import pandas as pd
import lightgbm as lgb
from scipy import stats
from pathlib import Path
from datetime import datetime

warnings.filterwarnings('ignore')

# Try MLflow
try:
    import mlflow
    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Try PyTorch for MLP
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    from torch.utils.data import DataLoader, TensorDataset
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False

# ── Paths ──
FEATURE_FILE = Path("/home/nick/Lvl3Quant/output/midtrade_thesis_v1/trade_tick_features.parquet")
OUTPUT_DIR = Path("/home/nick/Lvl3Quant/output/midtrade_enhanced_v2")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TICK_VALUE = 12.50
COST_RT_TICKS = 2.376

# MLflow
MLFLOW_URI = "http://localhost:5000"


class MidTradeMLP(nn.Module):
    """Small MLP for mid-trade classification."""
    def __init__(self, n_features, hidden_dims=[64, 32, 16], dropout=0.3):
        super().__init__()
        layers = []
        in_dim = n_features
        for h in hidden_dims:
            layers.extend([
                nn.Linear(in_dim, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            in_dim = h
        layers.append(nn.Linear(in_dim, 1))
        self.net = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.net(x).squeeze(-1)


def add_enhanced_features(df):
    """Add spread dynamics, imbalance velocity, and derived features."""
    enhanced = df.copy()
    
    # For each checkpoint (5s, 10s, 15s, 30s), compute additional derived features
    for cp in ['5', '10', '15', '30']:
        prefix = f'cp{cp}s_'
        
        # Spread dynamics: rate of change relative to mean
        spread_col = f'{prefix}spread_mean'
        vel_col = f'{prefix}spread_vel_mean'
        if spread_col in df.columns and vel_col in df.columns:
            # Spread expansion indicator
            enhanced[f'{prefix}spread_expanding'] = (df[vel_col] > 0).astype(float)
            # Spread relative to typical (using cross-sectional z-score proxy)
            spread_vals = df[spread_col]
            enhanced[f'{prefix}spread_zscore'] = (spread_vals - spread_vals.mean()) / (spread_vals.std() + 1e-8)
        
        # Trade imbalance velocity
        ofi_first = f'{prefix}ofi_first_half'
        ofi_second = f'{prefix}ofi_second_half'
        if ofi_first in df.columns and ofi_second in df.columns:
            # Imbalance acceleration
            enhanced[f'{prefix}imbalance_accel'] = df[ofi_second] - df[ofi_first]
            # Imbalance momentum (absolute velocity change)
            enhanced[f'{prefix}imbalance_momentum'] = np.abs(df[ofi_second]) - np.abs(df[ofi_first])
        
        # Event density features
        n_events = f'{prefix}n_events'
        density = f'{prefix}event_density_mean'
        if n_events in df.columns:
            # Event burst indicator
            events = df[n_events]
            enhanced[f'{prefix}event_burst'] = (events > events.quantile(0.75)).astype(float)
        
        # Cross-feature interactions
        ofi_mean = f'{prefix}ofi_mean'
        pmom_mean = f'{prefix}price_mom_mean'
        smom_mean = f'{prefix}sign_mom_mean'
        if ofi_mean in df.columns and pmom_mean in df.columns:
            # OFI * price momentum interaction (agreement/disagreement)
            enhanced[f'{prefix}ofi_pmom_interact'] = df[ofi_mean] * df[pmom_mean]
        
        if ofi_mean in df.columns and smom_mean in df.columns:
            # OFI * sign momentum interaction
            enhanced[f'{prefix}ofi_smom_interact'] = df[ofi_mean] * df[smom_mean]
        
        # Directional alignment with trade direction
        if ofi_mean in df.columns:
            enhanced[f'{prefix}ofi_aligned_dir'] = df[ofi_mean] * df['direction']
        if pmom_mean in df.columns:
            enhanced[f'{prefix}pmom_aligned_dir'] = df[pmom_mean] * df['direction']
    
    # Cross-checkpoint features (evolution over time)
    for feat in ['ofi_mean', 'price_mom_mean', 'spread_mean']:
        cols_available = []
        for cp in ['5', '10', '15', '30']:
            col = f'cp{cp}s_{feat}'
            if col in df.columns:
                cols_available.append((int(cp), col))
        
        if len(cols_available) >= 2:
            # Trend from first to last checkpoint
            first_cp, first_col = cols_available[0]
            last_cp, last_col = cols_available[-1]
            enhanced[f'cross_{feat}_trend'] = df[last_col] - df[first_col]
            
            # Acceleration (second difference)
            if len(cols_available) >= 3:
                mid_cp, mid_col = cols_available[1]
                enhanced[f'cross_{feat}_accel'] = (df[last_col] - df[mid_col]) - (df[mid_col] - df[first_col])
    
    # Trade context features
    enhanced['pred_magnitude_sq'] = df['pred_magnitude'] ** 2
    enhanced['time_of_day_norm'] = df['time_of_day_min'] / 450.0  # Normalized session time
    enhanced['early_session'] = (df['time_of_day_min'] < 60).astype(float)
    enhanced['late_session'] = (df['time_of_day_min'] > 360).astype(float)
    
    return enhanced


def walk_forward_lgbm(df, feature_cols, target_col, checkpoint_sec, min_train=30):
    """Walk-forward LightGBM with continuous confidence output."""
    dates = df['date'].unique()
    n_dates = len(dates)
    
    results = []
    fold_metrics = []
    
    for i in range(min_train, n_dates):
        test_date = dates[i]
        # Sliding window: use last min_train dates
        train_dates = dates[max(0, i - min_train):i]
        
        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'] == test_date
        
        X_train = df.loc[train_mask, feature_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feature_cols].values
        y_test = df.loc[test_mask, target_col].values
        test_data = df.loc[test_mask].copy()
        
        if len(X_train) < 10 or len(X_test) == 0:
            continue
        
        # Handle NaN
        X_train = np.nan_to_num(X_train, 0)
        X_test = np.nan_to_num(X_test, 0)
        
        model = lgb.LGBMClassifier(
            n_estimators=200, max_depth=4, learning_rate=0.05,
            num_leaves=15, min_child_samples=5, subsample=0.8,
            colsample_bytree=0.7, reg_alpha=1.0, reg_lambda=1.0,
            verbose=-1, random_state=42
        )
        
        try:
            model.fit(X_train, y_train)
        except Exception as e:
            continue
        
        probs = model.predict_proba(X_test)[:, 1]
        preds = (probs > 0.5).astype(int)
        
        for j in range(len(X_test)):
            results.append({
                'date': test_date,
                'prob_winner': float(probs[j]),
                'pred_winner': int(preds[j]),
                'actual_winner': int(y_test[j]),
                'exit_ticks': float(test_data.iloc[j]['exit_ticks']),
                'direction': int(test_data.iloc[j]['direction']),
            })
        
        acc = np.mean(preds == y_test)
        fold_metrics.append({
            'test_date': test_date,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'accuracy': float(acc),
        })
    
    return pd.DataFrame(results), fold_metrics


def walk_forward_mlp(df, feature_cols, target_col, checkpoint_sec, min_train=30,
                     device='cuda', epochs=50, lr=0.001, batch_size=64):
    """Walk-forward MLP with continuous confidence output."""
    if not TORCH_AVAILABLE:
        print("PyTorch not available, skipping MLP")
        return None, []
    
    dates = df['date'].unique()
    n_dates = len(dates)
    
    results = []
    fold_metrics = []
    
    for i in range(min_train, n_dates):
        test_date = dates[i]
        train_dates = dates[max(0, i - min_train):i]
        
        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'] == test_date
        
        X_train = df.loc[train_mask, feature_cols].values.astype(np.float32)
        y_train = df.loc[train_mask, target_col].values.astype(np.float32)
        X_test = df.loc[test_mask, feature_cols].values.astype(np.float32)
        y_test = df.loc[test_mask, target_col].values.astype(np.float32)
        test_data = df.loc[test_mask].copy()
        
        if len(X_train) < 10 or len(X_test) == 0:
            continue
        
        # Handle NaN
        X_train = np.nan_to_num(X_train, 0)
        X_test = np.nan_to_num(X_test, 0)
        
        # Normalize features
        mu = X_train.mean(axis=0)
        sigma = X_train.std(axis=0) + 1e-8
        X_train_norm = (X_train - mu) / sigma
        X_test_norm = (X_test - mu) / sigma
        
        # PyTorch
        dev = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        train_ds = TensorDataset(
            torch.tensor(X_train_norm, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.float32)
        )
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True, 
                              num_workers=0, pin_memory=True)
        
        model = MidTradeMLP(len(feature_cols), hidden_dims=[64, 32, 16], dropout=0.3).to(dev)
        optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
        criterion = nn.BCEWithLogitsLoss()
        
        # Train
        model.train()
        for epoch in range(epochs):
            for xb, yb in train_dl:
                xb, yb = xb.to(dev), yb.to(dev)
                optimizer.zero_grad()
                logits = model(xb)
                loss = criterion(logits, yb)
                loss.backward()
                optimizer.step()
        
        # Predict
        model.eval()
        with torch.no_grad():
            X_test_t = torch.tensor(X_test_norm, dtype=torch.float32).to(dev)
            logits = model(X_test_t)
            probs = torch.sigmoid(logits).cpu().numpy()
        
        preds = (probs > 0.5).astype(int)
        
        for j in range(len(X_test)):
            results.append({
                'date': test_date,
                'prob_winner': float(probs[j]),
                'pred_winner': int(preds[j]),
                'actual_winner': int(y_test[j]),
                'exit_ticks': float(test_data.iloc[j]['exit_ticks']),
                'direction': int(test_data.iloc[j]['direction']),
            })
        
        acc = np.mean(preds == y_test)
        fold_metrics.append({
            'test_date': test_date,
            'n_train': len(X_train),
            'n_test': len(X_test),
            'accuracy': float(acc),
        })
    
    return pd.DataFrame(results), fold_metrics


def evaluate_continuous_scoring(results_df, cut_cost_ticks=1.376):
    """Evaluate continuous confidence-based position sizing."""
    if results_df is None or len(results_df) == 0:
        return {}
    
    # Baseline: hold all trades
    baseline_pnl = results_df['exit_ticks'].sum()
    
    evaluation = {}
    
    # Binary cut at various thresholds
    for thresh in [0.25, 0.30, 0.35, 0.40, 0.45]:
        cut_mask = results_df['prob_winner'] < thresh
        n_cut = cut_mask.sum()
        n_keep = (~cut_mask).sum()
        
        # PnL: keep trades get full exit_ticks, cut trades get -cut_cost
        pnl_keep = results_df.loc[~cut_mask, 'exit_ticks'].sum()
        pnl_cut = -cut_cost_ticks * n_cut
        total_pnl = pnl_keep + pnl_cut
        
        # Also check: among cut trades, how many WERE losers?
        if n_cut > 0:
            cut_trades = results_df[cut_mask]
            pct_losers_cut = (cut_trades['actual_winner'] == 0).mean()
        else:
            pct_losers_cut = 0
        
        evaluation[f'cut_{thresh:.2f}'] = {
            'threshold': float(thresh),
            'n_cut': int(n_cut),
            'n_keep': int(n_keep),
            'cut_rate': float(n_cut / len(results_df)),
            'pnl_keep': float(pnl_keep),
            'pnl_cut_cost': float(pnl_cut),
            'total_pnl': float(total_pnl),
            'improvement_vs_baseline': float(total_pnl - baseline_pnl),
            'pct_losers_in_cut': float(pct_losers_cut),
        }
    
    # Continuous scoring: scale position by confidence
    # confidence = 2 * abs(prob - 0.5) → 0 (uncertain) to 1 (confident)
    confidence = 2 * np.abs(results_df['prob_winner'].values - 0.5)
    exit_ticks = results_df['exit_ticks'].values
    
    # Weighted PnL: position_size proportional to confidence
    weighted_pnl = np.sum(confidence * exit_ticks - confidence * cut_cost_ticks)
    evaluation['continuous_sizing'] = {
        'weighted_pnl': float(weighted_pnl),
        'mean_confidence': float(np.mean(confidence)),
        'std_confidence': float(np.std(confidence)),
        'improvement_vs_baseline': float(weighted_pnl - baseline_pnl),
    }
    
    evaluation['baseline_pnl'] = float(baseline_pnl)
    
    return evaluation


def main():
    t0 = time.time()
    print("=" * 70)
    print("TASK 2: Enhanced Mid-Trade Classifier v2")
    print("=" * 70)
    
    # Setup MLflow
    if MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(MLFLOW_URI)
        mlflow.set_experiment("midtrade_enhanced_v2")
        run = mlflow.start_run(run_name=f"enhanced_v2_{datetime.now().strftime('%Y%m%d_%H%M')}")
        print(f"MLflow run started: {run.info.run_id}")
    
    # Load data
    df = pd.read_parquet(FEATURE_FILE)
    print(f"Loaded {len(df)} trades from midtrade_thesis_v1")
    
    # Add enhanced features
    print("\n[1] Adding enhanced features...")
    df_enhanced = add_enhanced_features(df)
    
    # Feature columns (exclude metadata/targets)
    exclude = ['date', 'direction', 'fill_price', 'time_of_day_min', 'exit_type', 'exit_ticks', 'winner', 'pred_magnitude']
    feature_cols = [c for c in df_enhanced.columns if c not in exclude]
    print(f"Original features: {len([c for c in df.columns if c not in exclude])}")
    print(f"Enhanced features: {len(feature_cols)} (+{len(feature_cols) - len([c for c in df.columns if c not in exclude])} new)")
    
    # Run for each checkpoint
    all_results = {}
    
    for checkpoint_sec in [5, 10, 15, 30]:
        print(f"\n{'='*50}")
        print(f"Checkpoint: T+{checkpoint_sec}s")
        print(f"{'='*50}")
        
        # Use features for this checkpoint and all earlier ones
        cp_feature_cols = [c for c in feature_cols 
                          if any(c.startswith(f'cp{cp}s_') for cp in ['5', '10', '15', '30'] 
                                if int(cp) <= checkpoint_sec)
                          or not any(c.startswith(f'cp{cp}s_') for cp in ['5', '10', '15', '30'])  # non-checkpoint features
                          or c.startswith('cross_')  # cross-checkpoint features
                          ]
        
        print(f"Features for T+{checkpoint_sec}s: {len(cp_feature_cols)}")
        
        # ── LightGBM (enhanced) ──
        print(f"\n  [LGBM] Walk-forward...")
        lgbm_results, lgbm_folds = walk_forward_lgbm(
            df_enhanced, cp_feature_cols, 'winner', checkpoint_sec, min_train=30
        )
        
        if len(lgbm_results) > 0:
            acc = np.mean(lgbm_results['pred_winner'] == lgbm_results['actual_winner'])
            # AUC
            from sklearn.metrics import roc_auc_score
            try:
                auc = roc_auc_score(lgbm_results['actual_winner'], lgbm_results['prob_winner'])
            except:
                auc = 0.5
            ic = np.corrcoef(lgbm_results['prob_winner'], lgbm_results['actual_winner'])[0, 1]
            
            print(f"  [LGBM] ACC={acc:.4f}, AUC={auc:.4f}, IC={ic:.4f}, N={len(lgbm_results)}")
            
            # Continuous scoring evaluation
            lgbm_eval = evaluate_continuous_scoring(lgbm_results)
            print(f"  [LGBM] Baseline PnL: {lgbm_eval['baseline_pnl']:.1f} ticks")
            for k, v in lgbm_eval.items():
                if k.startswith('cut_'):
                    print(f"  [LGBM] {k}: cut={v['n_cut']}, keep={v['n_keep']}, "
                          f"PnL={v['total_pnl']:.1f}, improve={v['improvement_vs_baseline']:+.1f}")
            if 'continuous_sizing' in lgbm_eval:
                cs = lgbm_eval['continuous_sizing']
                print(f"  [LGBM] Continuous sizing: PnL={cs['weighted_pnl']:.1f}, "
                      f"improve={cs['improvement_vs_baseline']:+.1f}")
            
            all_results[f'lgbm_t{checkpoint_sec}'] = {
                'accuracy': float(acc),
                'auc': float(auc),
                'ic': float(ic),
                'n_valid': len(lgbm_results),
                'evaluation': lgbm_eval,
            }
        
        # ── MLP (GPU) ──
        if TORCH_AVAILABLE:
            print(f"\n  [MLP] Walk-forward (GPU)...")
            mlp_results, mlp_folds = walk_forward_mlp(
                df_enhanced, cp_feature_cols, 'winner', checkpoint_sec,
                min_train=30, device='cuda', epochs=50, lr=0.001, batch_size=64
            )
            
            if mlp_results is not None and len(mlp_results) > 0:
                acc = np.mean(mlp_results['pred_winner'] == mlp_results['actual_winner'])
                try:
                    auc = roc_auc_score(mlp_results['actual_winner'], mlp_results['prob_winner'])
                except:
                    auc = 0.5
                ic = np.corrcoef(mlp_results['prob_winner'], mlp_results['actual_winner'])[0, 1]
                
                print(f"  [MLP] ACC={acc:.4f}, AUC={auc:.4f}, IC={ic:.4f}, N={len(mlp_results)}")
                
                mlp_eval = evaluate_continuous_scoring(mlp_results)
                print(f"  [MLP] Baseline PnL: {mlp_eval['baseline_pnl']:.1f} ticks")
                for k, v in mlp_eval.items():
                    if k.startswith('cut_'):
                        print(f"  [MLP] {k}: cut={v['n_cut']}, keep={v['n_keep']}, "
                              f"PnL={v['total_pnl']:.1f}, improve={v['improvement_vs_baseline']:+.1f}")
                
                all_results[f'mlp_t{checkpoint_sec}'] = {
                    'accuracy': float(acc),
                    'auc': float(auc),
                    'ic': float(ic),
                    'n_valid': len(mlp_results),
                    'evaluation': mlp_eval,
                }
    
    # ── Find best model/checkpoint combo ──
    print(f"\n{'='*70}")
    print("RESULTS SUMMARY")
    print(f"{'='*70}")
    
    best_key = None
    best_improvement = -float('inf')
    
    for k, v in all_results.items():
        eval_data = v.get('evaluation', {})
        # Best binary cut improvement
        best_cut = max((d.get('improvement_vs_baseline', 0) for d in eval_data.values() 
                       if isinstance(d, dict) and 'improvement_vs_baseline' in d), default=0)
        print(f"{k}: AUC={v['auc']:.4f}, IC={v['ic']:.4f}, Best cut improvement={best_cut:+.1f} ticks")
        
        if best_cut > best_improvement:
            best_improvement = best_cut
            best_key = k
    
    print(f"\nBest: {best_key} (improvement: {best_improvement:+.1f} ticks)")
    
    elapsed = time.time() - t0
    
    # Log to MLflow
    if MLFLOW_AVAILABLE:
        for k, v in all_results.items():
            mlflow.log_metric(f"{k}_auc", v['auc'])
            mlflow.log_metric(f"{k}_ic", v['ic'])
            mlflow.log_metric(f"{k}_accuracy", v['accuracy'])
        mlflow.log_metric("elapsed_seconds", elapsed)
        mlflow.log_param("best_model", best_key)
        mlflow.log_param("n_features_enhanced", len(feature_cols))
        mlflow.end_run()
        print(f"\nMLflow run logged successfully")
    
    # Save results
    output = {
        'timestamp': datetime.now().isoformat(),
        'elapsed_seconds': elapsed,
        'n_features_original': len([c for c in df.columns if c not in exclude]),
        'n_features_enhanced': len(feature_cols),
        'n_trades': len(df),
        'results': all_results,
        'best_model': best_key,
        'best_improvement_ticks': float(best_improvement),
    }
    
    out_file = OUTPUT_DIR / 'results.json'
    with open(out_file, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nSaved to {out_file}")
    print(f"Elapsed: {elapsed:.1f}s")


if __name__ == '__main__':
    main()
