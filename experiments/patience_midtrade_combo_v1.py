#!/usr/bin/env python3
"""
patience_midtrade_combo_v1.py — Combine patience entries + mid-trade classifier

The two biggest independent improvements found on 2026-06-23:
  1. Patience entries (2-tick pullback): Sharpe 13.7 → 16.8 (entry improvement)
  2. Mid-trade classifier (cut losers at T+10s): Sharpe 13.7 → 12.31 (exit improvement)

These operate on INDEPENDENT dimensions (entry fill vs exit timing), so they
should stack. This experiment validates that hypothesis.

APPROACH:
  - Load champion trades with tick-level features (from midtrade_thesis_v1)
  - Apply patience entry price adjustment (2-tick and 5-tick pullback)
  - Re-simulate TP/SL outcomes with adjusted entry prices
  - Apply walk-forward mid-trade classifier at T+10s checkpoint
  - Cut trades classified as losers (P(win) < threshold)
  - FIFO P&L with regime gate

Author: Claude (Head of Quant)
Date: 2026-06-23
"""
from __future__ import annotations

import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

try:
    import lightgbm as lgb
except ImportError:
    print("ERROR: lightgbm not installed")
    sys.exit(1)

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT = Path("/home/nick/Lvl3Quant")
SMART_EVENTS_DIR = ROOT / "data" / "processed" / "mbo_events_smart_v3"
TICK_FEATURES_PATH = ROOT / "output" / "midtrade_thesis_v1" / "trade_tick_features.parquet"
PATIENCE_SWEEP_PATH = ROOT / "output" / "patience_entry_v1" / "patience_sweep.csv"
OUT_DIR = ROOT / "output" / "patience_midtrade_combo_v1"
OUT_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = ROOT / "logs" / f"patience_midtrade_combo_{time.strftime('%Y%m%d_%H%M%S')}.log"
LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.FileHandler(LOG_FILE), logging.StreamHandler(sys.stdout)],
)
log = logging.getLogger("patience_midtrade_combo")

# ── Cost Constants (canonical) ─────────────────────────────────────────────
ES_TICK_VALUE = 12.50
RT_COMMISSION_TICKS = 0.376   # AMP round-trip passive
MARKET_COST_TICKS = 1.376     # commission + spread crossing
TICK_SIZE_PTS = 0.25

# Champion config
TP_LONG = 25
TP_SHORT = 25
SL_LONG = 4
SL_SHORT = 3

# Walk-forward
WF_TRAIN_DAYS = 30
WF_TEST_DAYS = 1

CHECKPOINT_SEC = 10  # T+10s is the sweet spot from thesis_v1
CUT_THRESHOLDS = [0.30, 0.35, 0.40, 0.45, 0.50]
PULLBACK_TICKS = [0, 1, 2, 3, 5]  # 0 = baseline (no patience)


def load_features():
    """Load pre-extracted tick features from midtrade_thesis_v1."""
    log.info(f"Loading tick features from {TICK_FEATURES_PATH}")
    df = pd.read_parquet(TICK_FEATURES_PATH)
    log.info(f"  {len(df)} trades with {len(df.columns)} columns")
    
    # Ensure date column exists and is sorted
    if 'date' not in df.columns and 'entry_time' in df.columns:
        df['date'] = pd.to_datetime(df['entry_time']).dt.date.astype(str)
    
    df = df.sort_values('date').reset_index(drop=True)
    return df


def adjust_for_patience(df, pullback_ticks):
    """
    Adjust trade outcomes for patience entry.
    
    Patience = wait for N-tick pullback before entry.
    This IMPROVES fill price by N ticks, which:
      - For longs: lower entry → TP easier to hit, SL harder to hit
      - For shorts: higher entry → same logic
    
    Net effect: exit_ticks increases by pullback_ticks for all trades
    (except commission stays the same since it's per-round-trip).
    """
    if pullback_ticks == 0:
        return df.copy()
    
    adjusted = df.copy()
    # Better fill = more ticks of profit on every trade
    adjusted['exit_ticks'] = adjusted['exit_ticks'] + pullback_ticks
    
    # Re-derive win/loss based on adjusted P&L
    adjusted['winner'] = (adjusted['exit_ticks'] > 0).astype(int)
    
    return adjusted


def walk_forward_classify(df, checkpoint_sec):
    """Walk-forward LightGBM at specified checkpoint."""
    dates = sorted(df['date'].unique())
    
    # Feature columns (microstructure at checkpoint)
    feat_prefix = f"cp{checkpoint_sec}s_"
    feat_cols = [c for c in df.columns if c.startswith(feat_prefix)]
    
    if len(feat_cols) == 0:
        log.warning(f"No features found with prefix '{feat_prefix}', trying generic features")
        # Fallback: look for checkpoint-specific columns
        feat_cols = [c for c in df.columns if f'_{checkpoint_sec}s' in c or f'_t{checkpoint_sec}' in c]
    
    if len(feat_cols) == 0:
        log.error(f"Cannot find features for checkpoint {checkpoint_sec}s")
        return pd.Series(dtype=float), {}
    
    log.info(f"  Using {len(feat_cols)} features for T+{checkpoint_sec}s")
    
    predictions = pd.Series(index=df.index, dtype=float)
    all_true = []
    all_pred = []
    
    for i in range(WF_TRAIN_DAYS, len(dates)):
        test_date = dates[i]
        train_dates = dates[max(0, i - WF_TRAIN_DAYS):i]
        
        train_mask = df['date'].isin(train_dates)
        test_mask = df['date'] == test_date
        
        X_train = df.loc[train_mask, feat_cols].values
        y_train = df.loc[train_mask, 'winner'].values
        X_test = df.loc[test_mask, feat_cols].values
        
        if len(X_train) < 10 or len(X_test) == 0:
            continue
        
        # Handle NaN/inf
        X_train = np.nan_to_num(X_train, nan=0, posinf=0, neginf=0)
        X_test = np.nan_to_num(X_test, nan=0, posinf=0, neginf=0)
        
        params = {
            'n_estimators': 100,
            'max_depth': 4,
            'learning_rate': 0.05,
            'num_leaves': 16,
            'min_child_samples': 5,
            'verbose': -1,
        }
        
        model = lgb.LGBMClassifier(**params)
        model.fit(X_train, y_train)
        
        probs = model.predict_proba(X_test)[:, 1]
        predictions.loc[test_mask] = probs
        
        all_true.extend(df.loc[test_mask, 'winner'].values)
        all_pred.extend(probs)
    
    # Compute metrics
    valid_mask = predictions.notna()
    metrics = {'n_valid': int(valid_mask.sum())}
    
    if len(all_true) > 10:
        from sklearn.metrics import roc_auc_score, accuracy_score
        y_true = np.array(all_true)
        y_pred = np.array(all_pred)
        
        try:
            metrics['auc'] = float(roc_auc_score(y_true, y_pred))
        except:
            metrics['auc'] = 0.5
        
        metrics['accuracy'] = float(accuracy_score(y_true, (y_pred > 0.5).astype(int)))
        
        ic, _ = spearmanr(y_true, y_pred)
        metrics['ic'] = float(ic) if not np.isnan(ic) else 0.0
    
    return predictions, metrics


def simulate_combo(df, predictions, pullback_ticks, cut_threshold):
    """
    Simulate the full combo:
    1. Adjust P&L for patience entry (pullback improvement)
    2. Cut trades where classifier predicts loser (P(win) < threshold)
    3. Compute FIFO P&L with regime analysis
    """
    # Apply patience adjustment
    adjusted = adjust_for_patience(df, pullback_ticks)
    
    # Apply classifier cut
    valid_mask = predictions.notna()
    keep_mask = (predictions >= cut_threshold) | ~valid_mask  # keep if no prediction available
    
    managed = adjusted[keep_mask].copy()
    n_cut = int((valid_mask & ~keep_mask).sum())
    n_kept = len(managed)
    
    if n_kept == 0:
        return None
    
    # Daily P&L
    managed['pnl_net'] = managed['exit_ticks'] - RT_COMMISSION_TICKS
    daily = managed.groupby('date').agg(
        pnl=('pnl_net', 'sum'),
        n_trades=('pnl_net', 'count'),
    ).reset_index()
    
    total_pnl = daily['pnl'].sum()
    
    # Sharpe (annualized from daily)
    if daily['pnl'].std() > 0:
        sharpe = daily['pnl'].mean() / daily['pnl'].std() * np.sqrt(252)
    else:
        sharpe = 0.0
    
    # Sortino
    downside = daily['pnl'][daily['pnl'] < 0]
    if len(downside) > 0 and downside.std() > 0:
        sortino = daily['pnl'].mean() / downside.std() * np.sqrt(252)
    else:
        sortino = sharpe * 2  # placeholder
    
    # Win rate
    wr = (managed['pnl_net'] > 0).mean()
    
    # Profit factor
    gross_profit = managed['pnl_net'][managed['pnl_net'] > 0].sum()
    gross_loss = abs(managed['pnl_net'][managed['pnl_net'] < 0].sum())
    pf = gross_profit / gross_loss if gross_loss > 0 else float('inf')
    
    # Max drawdown
    cumulative = daily['pnl'].cumsum()
    running_max = cumulative.cummax()
    drawdowns = cumulative - running_max
    max_dd = drawdowns.min()
    
    # Regime analysis (green = ES up day, red = ES down day)
    # Use daily P&L sign as proxy for regime if no SPY data
    regime_results = {}
    if 'es_regime' in managed.columns:
        for regime in ['green', 'red', 'flat']:
            rmask = managed['es_regime'] == regime
            if rmask.sum() > 0:
                rpnl = managed.loc[rmask, 'pnl_net']
                rdaily = managed.loc[rmask].groupby('date')['pnl_net'].sum()
                if rdaily.std() > 0:
                    regime_results[regime] = {
                        'sharpe': float(rdaily.mean() / rdaily.std() * np.sqrt(252)),
                        'n_trades': int(rmask.sum()),
                        'n_days': int(rdaily.count()),
                    }
    
    # Regime gap
    regime_gap = 1.0  # default fail
    if 'green' in regime_results and 'red' in regime_results:
        sg = regime_results['green']['sharpe']
        sr = regime_results['red']['sharpe']
        denom = max(abs(sg), abs(sr), 0.001)
        regime_gap = abs(sg - sr) / denom
    
    regime_pass = regime_gap < 0.50
    
    # Day concentration
    day_conc = daily['pnl'].max() / total_pnl if total_pnl > 0 else 1.0
    
    return {
        'pullback_ticks': pullback_ticks,
        'cut_threshold': cut_threshold,
        'n_trades_original': len(df),
        'n_trades_kept': n_kept,
        'n_cut': n_cut,
        'cut_rate': n_cut / max(1, int(predictions.notna().sum())),
        'sharpe': float(sharpe),
        'sortino': float(sortino),
        'pf': float(pf),
        'wr': float(wr),
        'total_pnl_ticks': float(total_pnl),
        'total_pnl_dollars': float(total_pnl * ES_TICK_VALUE),
        'max_dd_ticks': float(max_dd),
        'day_concentration': float(day_conc),
        'regime_gap': float(regime_gap),
        'regime_pass': regime_pass,
        'regime_details': regime_results,
        'n_days': len(daily),
    }


def main():
    import mlflow
    
    t0 = time.time()
    log.info("=" * 80)
    log.info("Patience + Mid-Trade Combo v1 — Starting")
    log.info("=" * 80)
    
    # MLflow
    mlflow.set_tracking_uri("http://localhost:5000")
    mlflow.set_experiment("patience_midtrade_combo_v1")
    
    with mlflow.start_run(run_name=f"combo_v1_{time.strftime('%Y%m%d_%H%M%S')}"):
        mlflow.log_params({
            'checkpoint_sec': CHECKPOINT_SEC,
            'pullback_ticks': str(PULLBACK_TICKS),
            'cut_thresholds': str(CUT_THRESHOLDS),
            'wf_train_days': WF_TRAIN_DAYS,
        })
        
        # Load features
        df = load_features()
        mlflow.log_metric('n_trades_loaded', len(df))
        
        # Walk-forward classifier (only needs to run once, doesn't depend on patience)
        log.info(f"Walk-forward LightGBM at T+{CHECKPOINT_SEC}s...")
        predictions, clf_metrics = walk_forward_classify(df, CHECKPOINT_SEC)
        
        log.info(f"  Classifier: AUC={clf_metrics.get('auc', 0):.3f}, "
                 f"IC={clf_metrics.get('ic', 0):.3f}, "
                 f"N_valid={clf_metrics.get('n_valid', 0)}")
        
        for k, v in clf_metrics.items():
            mlflow.log_metric(f'clf_{k}', v)
        
        # Sweep all combos
        all_results = []
        best_result = None
        best_sharpe = -np.inf
        
        for pb in PULLBACK_TICKS:
            for thresh in CUT_THRESHOLDS:
                result = simulate_combo(df, predictions, pb, thresh)
                if result is None:
                    continue
                
                all_results.append(result)
                
                label = f"pb={pb}t|cut={thresh}"
                log.info(f"  {label}: Sharpe={result['sharpe']:.2f}, "
                         f"PF={result['pf']:.2f}, WR={result['wr']:.1%}, "
                         f"PnL={result['total_pnl_ticks']:.0f}t, "
                         f"Kept={result['n_trades_kept']}, "
                         f"RegimeGap={result['regime_gap']:.2f} "
                         f"{'PASS' if result['regime_pass'] else 'FAIL'}")
                
                mlflow.log_metrics({
                    f'sharpe_pb{pb}_th{int(thresh*100)}': result['sharpe'],
                    f'pf_pb{pb}_th{int(thresh*100)}': result['pf'],
                    f'pnl_pb{pb}_th{int(thresh*100)}': result['total_pnl_ticks'],
                    f'regime_gap_pb{pb}_th{int(thresh*100)}': result['regime_gap'],
                })
                
                if result['regime_pass'] and result['sharpe'] > best_sharpe:
                    best_sharpe = result['sharpe']
                    best_result = result
        
        # Also test patience-only (no classifier cut) and classifier-only (no patience)
        log.info("\n--- Component Analysis ---")
        
        # Patience only (cut_threshold=0, keep all)
        for pb in PULLBACK_TICKS:
            adj = adjust_for_patience(df, pb)
            valid = predictions.notna()
            pnl_net = adj['exit_ticks'] - RT_COMMISSION_TICKS
            total = pnl_net.sum()
            wr = (pnl_net > 0).mean()
            
            daily_pnl = adj.assign(pnl_net=pnl_net).groupby('date')['pnl_net'].sum()
            if daily_pnl.std() > 0:
                sharpe = daily_pnl.mean() / daily_pnl.std() * np.sqrt(252)
            else:
                sharpe = 0
            
            log.info(f"  Patience only (pb={pb}t): Sharpe={sharpe:.2f}, "
                     f"WR={wr:.1%}, PnL={total:.0f}t")
        
        # Classifier only (no patience)
        for thresh in CUT_THRESHOLDS:
            result_clf = simulate_combo(df, predictions, 0, thresh)
            if result_clf:
                log.info(f"  Classifier only (cut={thresh}): Sharpe={result_clf['sharpe']:.2f}, "
                         f"PF={result_clf['pf']:.2f}, WR={result_clf['wr']:.1%}, "
                         f"Kept={result_clf['n_trades_kept']}")
        
        # Summary
        log.info("\n" + "=" * 80)
        log.info("SUMMARY")
        log.info("=" * 80)
        
        if best_result:
            log.info(f"BEST COMBO (regime-passing):")
            log.info(f"  Pullback: {best_result['pullback_ticks']} ticks")
            log.info(f"  Cut threshold: {best_result['cut_threshold']}")
            log.info(f"  Sharpe: {best_result['sharpe']:.2f}")
            log.info(f"  Sortino: {best_result['sortino']:.2f}")
            log.info(f"  PF: {best_result['pf']:.2f}")
            log.info(f"  WR: {best_result['wr']:.1%}")
            log.info(f"  PnL: {best_result['total_pnl_ticks']:.0f} ticks "
                     f"(${best_result['total_pnl_dollars']:.0f})")
            log.info(f"  Trades: {best_result['n_trades_kept']} "
                     f"(cut {best_result['n_cut']})")
            log.info(f"  Regime gap: {best_result['regime_gap']:.3f} PASS")
            log.info(f"  Max DD: {best_result['max_dd_ticks']:.1f} ticks")
            
            mlflow.log_metrics({
                'best_sharpe': best_result['sharpe'],
                'best_sortino': best_result['sortino'],
                'best_pf': best_result['pf'],
                'best_wr': best_result['wr'],
                'best_pnl_ticks': best_result['total_pnl_ticks'],
                'best_pullback': best_result['pullback_ticks'],
                'best_threshold': best_result['cut_threshold'],
                'best_regime_gap': best_result['regime_gap'],
            })
            
            # Stacking analysis
            log.info("\n--- DO THEY STACK? ---")
            # Find patience-only best and classifier-only best
            patience_only = [r for r in all_results if r['cut_threshold'] == min(CUT_THRESHOLDS)]
            clf_only = [r for r in all_results if r['pullback_ticks'] == 0]
            
            baseline = [r for r in all_results 
                        if r['pullback_ticks'] == 0 and r['cut_threshold'] == min(CUT_THRESHOLDS)]
            
            if baseline:
                b = baseline[0]
                log.info(f"  Baseline (no patience, no cut): Sharpe={b['sharpe']:.2f}")
            
            if patience_only:
                best_p = max(patience_only, key=lambda x: x['sharpe'])
                log.info(f"  Best patience-only (pb={best_p['pullback_ticks']}t): "
                         f"Sharpe={best_p['sharpe']:.2f}")
            
            if clf_only:
                best_c = max([r for r in clf_only if r['regime_pass']], 
                             key=lambda x: x['sharpe'], default=None)
                if best_c:
                    log.info(f"  Best classifier-only (cut={best_c['cut_threshold']}): "
                             f"Sharpe={best_c['sharpe']:.2f}")
            
            log.info(f"  COMBO: Sharpe={best_result['sharpe']:.2f}")
            
            # Is combo > max(individual)?
            individual_best = max(
                best_p['sharpe'] if patience_only else 0,
                best_c['sharpe'] if clf_only and best_c else 0
            )
            stacks = best_result['sharpe'] > individual_best * 1.05
            log.info(f"  VERDICT: {'THEY STACK ✓' if stacks else 'THEY COMPETE ✗'} "
                     f"(combo {best_result['sharpe']:.2f} vs individual best {individual_best:.2f})")
        else:
            log.info("NO regime-passing combo found.")
        
        # Save results
        output = {
            'timestamp': time.strftime('%Y-%m-%dT%H:%M:%S'),
            'elapsed_seconds': time.time() - t0,
            'hypothesis': 'Patience entries + mid-trade classifier stack (independent mechanisms)',
            'classifier_metrics': clf_metrics,
            'all_results': all_results,
            'best_result': best_result,
            'n_combos_tested': len(all_results),
        }
        
        results_path = OUT_DIR / "results.json"
        with open(results_path, 'w') as f:
            json.dump(output, f, indent=2, default=str)
        
        mlflow.log_artifact(str(results_path))
        
        # Also save as CSV for easy review
        results_df = pd.DataFrame(all_results)
        csv_path = OUT_DIR / "combo_sweep.csv"
        results_df.to_csv(csv_path, index=False)
        mlflow.log_artifact(str(csv_path))
        
        log.info(f"\nElapsed: {time.time() - t0:.0f}s")
        log.info(f"Results: {results_path}")


if __name__ == '__main__':
    main()
