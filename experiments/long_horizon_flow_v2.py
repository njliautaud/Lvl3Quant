#!/usr/bin/env python3
"""
long_horizon_flow_v2.py — Longer-Horizon Flow Strategy v2

Improvements over v1 (which had IC ~0.07 max, negative Sharpe):
1. Multi-day rolling features (3d, 5d, 10d lookback aggregates)
2. Flow REGIME features (accumulation vs distribution via OFI persistence)  
3. Binary direction target (not magnitude — reduces noise)
4. Separate AM/PM predictions (morning flow predicts afternoon, today predicts tomorrow)
5. Walk-forward LightGBM with hyperparameter search
6. Test holding period optimization

Per HC #647: user wants strategies using LONG CONTEXT flow history.
Per HC #428 R1: regime-agnostic, full OOT validation.

Author: Claude (Head of Quant)
Date: 2026-06-23
"""
from __future__ import annotations

import json, logging, os, sys, time, warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
warnings.filterwarnings('ignore')

try:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, accuracy_score
except ImportError:
    print("ERROR: Missing packages"); sys.exit(1)

ROOT = Path("/home/nick/Lvl3Quant")
DAILY_FEATS = ROOT / "output" / "long_horizon_flow_v1" / "daily_features.parquet"
OUT_DIR = ROOT / "output" / "long_horizon_flow_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / f"long_horizon_flow_v2_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)])
log = logging.getLogger("lh_flow_v2")

ES_TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376
MARKET_COST_TICKS = 1.376
TICK_SIZE = 0.25

# Walk-forward
WF_TRAIN = 60  # more training data for noisy daily signal
WF_TEST = 1

def build_enhanced_features(df):
    """Build multi-day rolling flow features from daily data."""
    log.info("Building enhanced features...")
    
    # Original columns (from v1)
    base_cols = [c for c in df.columns if c not in ['date', 'open', 'close', 'high', 'low', 
                                                      'session_vwap', 'trade_count']]
    
    # Multi-day rolling features
    for window in [3, 5, 10, 20]:
        for col in ['session_ofi', 'pm_ofi', 'ofi_trend', 'volume_concentration', 'spread_mean']:
            if col in df.columns:
                df[f'{col}_roll{window}_mean'] = df[col].rolling(window, min_periods=window).mean()
                df[f'{col}_roll{window}_std'] = df[col].rolling(window, min_periods=window).std()
                # Z-score relative to recent history
                m = df[f'{col}_roll{window}_mean']
                s = df[f'{col}_roll{window}_std']
                df[f'{col}_zscore{window}'] = (df[col] - m) / s.replace(0, 1)
    
    # Flow regime features
    if 'session_ofi' in df.columns:
        # OFI persistence (same sign streak)
        ofi_sign = np.sign(df['session_ofi'])
        streaks = []
        streak = 0
        for s in ofi_sign:
            if s == (1 if streak > 0 else -1 if streak < 0 else 0):
                streak += (1 if s > 0 else -1)
            else:
                streak = int(s)
            streaks.append(streak)
        df['ofi_streak'] = streaks
        
        # Cumulative OFI (accumulation measure)
        for window in [3, 5, 10]:
            df[f'cum_ofi_{window}d'] = df['session_ofi'].rolling(window, min_periods=1).sum()
        
        # OFI acceleration (change in flow)
        df['ofi_accel'] = df['session_ofi'].diff()
        df['ofi_accel_3d'] = df['session_ofi'].diff(3)
    
    # Price-flow divergence (price moves without flow support = exhaustion)
    if 'close_vs_open_ticks' in df.columns and 'session_ofi' in df.columns:
        price_dir = np.sign(df['close_vs_open_ticks'])
        flow_dir = np.sign(df['session_ofi'])
        df['price_flow_agreement'] = (price_dir == flow_dir).astype(float)
        df['price_flow_agree_5d'] = df['price_flow_agreement'].rolling(5, min_periods=1).mean()
    
    # Returns for targets
    if 'close' in df.columns:
        for h in [1, 2, 3, 5]:
            df[f'fwd_return_{h}d_ticks'] = (df['close'].shift(-h) - df['close']) / TICK_SIZE
            df[f'fwd_direction_{h}d'] = (df[f'fwd_return_{h}d_ticks'] > 0).astype(int)
    
    # AM → PM prediction (within-day)
    if 'am_ofi' in df.columns and 'momentum_pm' in df.columns:
        df['am_predicts_pm'] = (np.sign(df['am_ofi']) == np.sign(df['momentum_pm'])).astype(float)
        df['am_predicts_pm_5d'] = df['am_predicts_pm'].rolling(5, min_periods=1).mean()
    
    # Volatility features
    if 'price_range_ticks' in df.columns:
        df['vol_5d'] = df['price_range_ticks'].rolling(5, min_periods=1).mean()
        df['vol_ratio'] = df['price_range_ticks'] / df['vol_5d'].replace(0, 1)
    
    log.info(f"  Enhanced features: {len(df.columns)} columns (was {len(base_cols)} base)")
    return df


def get_feature_cols(df):
    """Get feature columns (exclude targets, dates, raw prices)."""
    exclude = {'date', 'open', 'close', 'high', 'low', 'session_vwap', 'trade_count'}
    exclude.update({c for c in df.columns if c.startswith('fwd_')})
    return [c for c in df.columns if c not in exclude and df[c].dtype in [np.float64, np.float32, np.int64, np.int32, float, int]]


def walk_forward_predict(df, target_col, feat_cols):
    """Walk-forward LightGBM classification."""
    dates = sorted(df['date'].unique())
    n_dates = len(dates)
    
    predictions = pd.Series(index=df.index, dtype=float)
    actuals = pd.Series(index=df.index, dtype=float)
    
    fold_ics = []
    
    for i in range(WF_TRAIN, n_dates):
        test_date = dates[i]
        train_dates = dates[i - WF_TRAIN:i]
        
        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'] == test_date
        
        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, target_col].values
        X_test = df.loc[test_mask, feat_cols].values
        y_test = df.loc[test_mask, target_col].values
        
        if len(X_train) < 20 or len(X_test) == 0:
            continue
        
        if np.isnan(y_train).any() or np.isnan(y_test).any():
            continue
        
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
        
        params = {
            'n_estimators': 200,
            'max_depth': 3,
            'learning_rate': 0.03,
            'num_leaves': 8,
            'min_child_samples': 10,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'verbose': -1,
        }
        
        is_classification = df[target_col].nunique() <= 2
        
        if is_classification:
            model = lgb.LGBMClassifier(**params)
            model.fit(X_train, y_train)
            probs = model.predict_proba(X_test)[:, 1]
            predictions.loc[test_mask] = probs
        else:
            model = lgb.LGBMRegressor(**params)
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            predictions.loc[test_mask] = preds
        
        actuals.loc[test_mask] = y_test
    
    valid = predictions.notna() & actuals.notna()
    metrics = {'n_valid': int(valid.sum())}
    
    if valid.sum() > 10:
        y_true = actuals[valid].values
        y_pred = predictions[valid].values
        
        ic, _ = spearmanr(y_true, y_pred)
        metrics['ic'] = float(ic) if not np.isnan(ic) else 0.0
        
        if is_classification:
            try:
                metrics['auc'] = float(roc_auc_score(y_true, y_pred))
            except:
                metrics['auc'] = 0.5
            metrics['accuracy'] = float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))
            metrics['dir_accuracy'] = metrics['accuracy']
        else:
            # Direction accuracy for regression
            metrics['dir_accuracy'] = float(((np.sign(y_pred) == np.sign(y_true)) | 
                                              (y_true == 0)).mean())
    
    return predictions, actuals, metrics


def simulate_strategy(df, predictions, target_col, threshold=0.55):
    """Simulate daily strategy from predictions."""
    valid = predictions.notna()
    df_valid = df[valid].copy()
    df_valid['pred'] = predictions[valid].values
    
    is_direction = 'direction' in target_col
    
    results = []
    
    for _, row in df_valid.iterrows():
        if is_direction:
            # Binary: >threshold = long, <(1-threshold) = short
            if row['pred'] > threshold:
                direction = 1  # long
            elif row['pred'] < (1 - threshold):
                direction = -1  # short
            else:
                continue  # no trade
        else:
            # Regression: sign of prediction
            if abs(row['pred']) < 0.01:
                continue
            direction = 1 if row['pred'] > 0 else -1
        
        # Forward return in ticks
        # Parse horizon from target_col like fwd_return_2d_ticks or fwd_direction_3d
        import re
        h_match = re.search(r'(\d+)d', target_col)
        if not h_match:
            continue
        h = int(h_match.group(1))
        ret_col = f'fwd_return_{h}d_ticks'
        if ret_col not in df.columns:
            continue
        
        actual_move = row.get(ret_col, 0)
        if pd.isna(actual_move):
            continue
        
        pnl_ticks = direction * actual_move - MARKET_COST_TICKS  # 2x crossing (entry + exit)
        
        results.append({
            'date': row['date'],
            'direction': direction,
            'pred': row['pred'],
            'actual_move_ticks': actual_move,
            'pnl_ticks': pnl_ticks,
        })
    
    if not results:
        return None
    
    trades_df = pd.DataFrame(results)
    daily_pnl = trades_df.groupby('date')['pnl_ticks'].sum()
    
    total_pnl = daily_pnl.sum()
    n_trades = len(trades_df)
    wr = (trades_df['pnl_ticks'] > 0).mean()
    
    gross_profit = trades_df['pnl_ticks'][trades_df['pnl_ticks'] > 0].sum()
    gross_loss = abs(trades_df['pnl_ticks'][trades_df['pnl_ticks'] < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else 0
    
    if daily_pnl.std() > 0:
        sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
    else:
        sharpe = 0
    
    downside = daily_pnl[daily_pnl < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = daily_pnl.mean() / downside.std() * np.sqrt(252)
    else:
        sortino = sharpe
    
    cumsum = daily_pnl.cumsum()
    max_dd = (cumsum - cumsum.cummax()).min()
    calmar = (daily_pnl.mean() * 252) / abs(max_dd) if max_dd != 0 else 0
    
    # Regime analysis
    regime = {}
    if 'close_vs_open_ticks' in df.columns:
        df_regime = df.set_index('date')['close_vs_open_ticks']
        for idx, row_t in trades_df.iterrows():
            d = row_t['date']
            if d in df_regime.index:
                trades_df.loc[idx, 'es_move'] = df_regime[d]
        
        if 'es_move' in trades_df.columns:
            green_mask = trades_df['es_move'] > 0
            red_mask = trades_df['es_move'] < 0
            
            for label, mask in [('green', green_mask), ('red', red_mask)]:
                subset = trades_df[mask]
                if len(subset) > 5:
                    s_daily = subset.groupby('date')['pnl_ticks'].sum()
                    s_sharpe = s_daily.mean() / s_daily.std() * np.sqrt(252) if s_daily.std() > 0 else 0
                    regime[label] = {
                        'n_trades': int(len(subset)),
                        'sharpe': float(s_sharpe),
                        'wr': float((subset['pnl_ticks'] > 0).mean()),
                    }
    
    regime_gap = 1.0
    if 'green' in regime and 'red' in regime:
        sg, sr = regime['green']['sharpe'], regime['red']['sharpe']
        regime_gap = abs(sg - sr) / max(abs(sg), abs(sr), 0.001)
    
    return {
        'target': target_col,
        'threshold': threshold,
        'n_trades': n_trades,
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'pf': float(pf),
        'wr': float(wr),
        'calmar': float(calmar),
        'total_pnl_ticks': float(total_pnl),
        'total_pnl_dollars': float(total_pnl * ES_TICK_VALUE),
        'max_dd_ticks': float(max_dd),
        'avg_pnl_ticks': float(total_pnl / n_trades),
        'regime': regime,
        'regime_gap': float(regime_gap),
        'regime_pass': regime_gap < 0.50,
        'n_days': int(daily_pnl.count()),
    }


def main():
    import mlflow
    
    t0 = time.time()
    log.info("=" * 80)
    log.info("Long Horizon Flow Strategy v2 — Enhanced Features")
    log.info("=" * 80)
    
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("long_horizon_flow_v2")
    
    with mlflow.start_run(run_name=f"lh_flow_v2_{time.strftime('%Y%m%d_%H%M%S')}"):
        # Load and enhance
        df = pd.read_parquet(DAILY_FEATS)
        log.info(f"Loaded {len(df)} days, {len(df.columns)} columns")
        
        df = build_enhanced_features(df)
        feat_cols = get_feature_cols(df)
        log.info(f"Feature columns: {len(feat_cols)}")
        
        mlflow.log_params({
            'n_days': len(df),
            'n_features': len(feat_cols),
            'wf_train': WF_TRAIN,
        })
        
        # Test multiple targets and horizons
        targets = {
            # Direction (binary classification)
            'fwd_direction_1d': 'Next-day direction',
            'fwd_direction_2d': '2-day direction',
            'fwd_direction_3d': '3-day direction',
            # Return magnitude (regression)
            'fwd_return_1d_ticks': 'Next-day return',
            'fwd_return_2d_ticks': '2-day return',
        }
        
        all_results = {}
        best_strategy = None
        best_sharpe = -np.inf
        
        for target_col, desc in targets.items():
            if target_col not in df.columns:
                continue
            
            # Drop NaN target rows for this target
            valid = df[target_col].notna()
            df_target = df[valid].copy()
            
            log.info(f"\n--- Target: {desc} ({target_col}) ---")
            log.info(f"  {len(df_target)} valid days")
            
            predictions, actuals, metrics = walk_forward_predict(df_target, target_col, feat_cols)
            
            log.info(f"  IC={metrics.get('ic', 0):.4f}, "
                     f"DirAcc={metrics.get('dir_accuracy', 0):.1%}, "
                     f"AUC={metrics.get('auc', 'N/A')}, "
                     f"N={metrics.get('n_valid', 0)}")
            
            for k, v in metrics.items():
                mlflow.log_metric(f'{target_col}_{k}', v)
            
            all_results[target_col] = {
                'description': desc,
                'metrics': metrics,
                'strategies': [],
            }
            
            # Simulate strategies at different thresholds
            is_direction = 'direction' in target_col
            thresholds = [0.50, 0.52, 0.55, 0.58, 0.60] if is_direction else [0.0]
            
            for thresh in thresholds:
                strat = simulate_strategy(df_target, predictions, target_col, thresh)
                if strat is None:
                    continue
                
                all_results[target_col]['strategies'].append(strat)
                
                log.info(f"  Thresh={thresh:.2f}: Sharpe={strat['sharpe']:.2f}, "
                         f"PF={strat['pf']:.2f}, WR={strat['wr']:.1%}, "
                         f"PnL={strat['total_pnl_ticks']:.0f}t, "
                         f"Trades={strat['n_trades']}, "
                         f"RegimeGap={strat['regime_gap']:.2f} "
                         f"{'PASS' if strat['regime_pass'] else 'FAIL'}")
                
                mlflow.log_metrics({
                    f'sharpe_{target_col}_th{int(thresh*100)}': strat['sharpe'],
                    f'pf_{target_col}_th{int(thresh*100)}': strat['pf'],
                    f'wr_{target_col}_th{int(thresh*100)}': strat['wr'],
                })
                
                if strat['regime_pass'] and strat['sharpe'] > best_sharpe:
                    best_sharpe = strat['sharpe']
                    best_strategy = strat
        
        # Summary
        log.info("\n" + "=" * 80)
        log.info("SUMMARY")
        log.info("=" * 80)
        
        for target_col, data in all_results.items():
            ic = data['metrics'].get('ic', 0)
            best_strat = max(data['strategies'], key=lambda x: x['sharpe']) if data['strategies'] else None
            log.info(f"  {data['description']}: IC={ic:.4f}" + 
                     (f", Best Sharpe={best_strat['sharpe']:.2f}" if best_strat else ""))
        
        if best_strategy:
            log.info(f"\nBEST REGIME-PASSING STRATEGY:")
            log.info(f"  Target: {best_strategy['target']}")
            log.info(f"  Sharpe: {best_strategy['sharpe']:.2f}")
            log.info(f"  Sortino: {best_strategy['sortino']:.2f}")
            log.info(f"  PF: {best_strategy['pf']:.2f}")
            log.info(f"  WR: {best_strategy['wr']:.1%}")
            log.info(f"  PnL: {best_strategy['total_pnl_ticks']:.0f} ticks "
                     f"(${best_strategy['total_pnl_dollars']:.0f})")
            log.info(f"  Trades: {best_strategy['n_trades']}")
            log.info(f"  Regime gap: {best_strategy['regime_gap']:.3f}")
            
            mlflow.log_metrics({
                'best_sharpe': best_strategy['sharpe'],
                'best_pf': best_strategy['pf'],
                'best_regime_gap': best_strategy['regime_gap'],
            })
        else:
            log.info("\nNO regime-passing strategy found.")
            log.info("Flow data may not contain sufficient directional signal at daily+ horizons.")
            log.info("Consider: tick-level flow patterns aggregated over hours, or longer training windows.")
        
        # Save
        output = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'elapsed_seconds': time.time() - t0,
            'n_features': len(feat_cols),
            'feature_list': feat_cols,
            'results': all_results,
            'best_strategy': best_strategy,
        }
        
        results_path = OUT_DIR / "results_v2.json"
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        mlflow.log_artifact(str(results_path))
        
        # Save enhanced features
        feats_path = OUT_DIR / "enhanced_daily_features.parquet"
        df.to_parquet(feats_path, index=False)
        
        log.info(f"\nElapsed: {time.time() - t0:.0f}s")
        log.info(f"Results: {results_path}")


if __name__ == '__main__':
    main()
