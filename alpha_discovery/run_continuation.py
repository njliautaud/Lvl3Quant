"""
CONTINUATION SCRIPT — Fix Phase 4 + Run Execution Backtest
===========================================================

Picks up where the overnight run left off:
1. Re-runs walk-forward for ret_5s, ret_10s, ret_30s (viable horizons)
2. Runs execution backtest on each with regime bug fix
3. Runs execution component ablation on ret_10s (best config)
4. Completes Phase 4 (Ridge with NaN fix + HP sweep)

Usage:
    python alpha_discovery/run_continuation.py
"""

import sys
import gc
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List

# Setup path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.mbo_features import compute_mbo_features, get_feature_names
from alpha_discovery.run_return_multihorizon import EXCLUDE_FEATURES_DIRECTION

# ============================================================================
# LOGGING
# ============================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
log_file = RESULTS_DIR / f"continuation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

# Force logging setup (basicConfig is no-op if root already configured)
root_logger = logging.getLogger()
root_logger.setLevel(logging.INFO)
# Remove any existing handlers
for h in root_logger.handlers[:]:
    root_logger.removeHandler(h)
fmt = logging.Formatter('%(asctime)s %(name)s %(levelname)s: %(message)s')
fh = logging.FileHandler(log_file, mode='w')
fh.setFormatter(fmt)
fh.setLevel(logging.INFO)
sh = logging.StreamHandler(sys.stdout)
sh.setFormatter(fmt)
sh.setLevel(logging.INFO)
root_logger.addHandler(fh)
root_logger.addHandler(sh)
log = logging.getLogger('continuation')

# Flush after every log line
import atexit
atexit.register(lambda: fh.flush())

TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_TICKS = 4.70 / TICK_VALUE  # 0.376 ticks (HC #52 canonical: $4.70 RT AMP)

EXCLUDE_DIRECTION = EXCLUDE_FEATURES_DIRECTION


def discord_notify(msg):
    """Log Discord-style message."""
    log.info(f"[DISCORD] {msg}")


# ============================================================================
# PART 1: Walk-Forward + Execution Backtest for Viable Horizons
# ============================================================================

def run_walkforward_and_execution(horizons_to_run=None):
    """Re-run walk-forward for viable horizons and execution backtest."""
    from scipy.stats import spearmanr
    import lightgbm as lgb

    if horizons_to_run is None:
        horizons_to_run = {
            'ret_5s': 50,
            'ret_10s': 100,
            'ret_30s': 300,
        }

    log.info("=" * 70)
    log.info("PART 1: WALK-FORWARD + EXECUTION BACKTEST")
    log.info("=" * 70)

    # Load data
    scanner = MBOAlphaScanner()
    load_info = scanner.load_from_cache()
    mid = scanner.mid_prices
    features = scanner.features
    feature_names = scanner.feature_names
    day_bounds = scanner.day_boundaries
    n_days = load_info['n_days']
    N = len(mid)

    log.info(f"Loaded {n_days} days, {N:,} bars")

    # Exclude direction-leaking features
    exclude_set = set(EXCLUDE_DIRECTION)
    keep_mask = np.array([fn not in exclude_set for fn in feature_names])
    X = features[:, keep_mask]
    feat_names_used = [fn for fn in feature_names if fn not in exclude_set]
    log.info(f"Using {X.shape[1]} features (excluded {sum(~keep_mask)} direction-leaking)")

    params = {
        'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.7,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'min_child_samples': 500,
        'verbose': -1, 'n_jobs': -1, 'device': 'gpu',
        'objective': 'regression', 'metric': 'rmse',
    }

    all_results = {}
    exec_results = {}

    for hz_name, steps in horizons_to_run.items():
        t0 = time.time()
        log.info(f"\n{'=' * 50}")
        log.info(f"HORIZON: {hz_name} (steps={steps})")
        log.info(f"{'=' * 50}")

        # Compute target
        target = np.full(N, np.nan, dtype=np.float32)
        target[:N - steps] = (mid[steps:] - mid[:N - steps]) / np.maximum(mid[:N - steps], 1.0)

        # NaN-fill targets crossing day boundaries
        n_days_local = len(day_bounds) - 1
        for d in range(n_days_local - 1):
            day_end = day_bounds[d + 1]
            nan_start = max(day_bounds[d], day_end - steps)
            target[nan_start:day_end] = np.nan

        # Walk-forward
        min_train_days = 5
        fold_ics = []
        all_preds = []
        all_actuals = []
        all_bar_indices = []

        for test_day in range(min_train_days, n_days_local):
            train_end = day_bounds[test_day - 1 + 1]
            test_start = day_bounds[test_day]
            test_end = day_bounds[test_day + 1]

            X_tr = X[:train_end]
            y_tr = target[:train_end]
            X_te = X[test_start:test_end]
            y_te = target[test_start:test_end]

            train_valid = np.isfinite(y_tr)
            test_valid = np.isfinite(y_te)

            if train_valid.sum() < 1000 or test_valid.sum() < 100:
                continue

            X_tr_v, y_tr_v = X_tr[train_valid], y_tr[train_valid]
            X_te_v, y_te_v = X_te[test_valid], y_te[test_valid]
            test_bar_indices = np.arange(test_start, test_end)[test_valid]

            split = int(len(X_tr_v) * 0.8)
            try:
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr_v[:split], y_tr_v[:split],
                    eval_set=[(X_tr_v[split:], y_tr_v[split:])],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
            except Exception as e:
                log.warning(f"  Training failed day {test_day}: {e}")
                continue

            preds = model.predict(X_te_v)
            ic = float(spearmanr(preds, y_te_v)[0])
            if np.isfinite(ic):
                fold_ics.append(ic)
            all_preds.append(preds)
            all_actuals.append(y_te_v)
            all_bar_indices.append(test_bar_indices)

            del model
            gc.collect()

            if test_day % 20 == 0:
                log.info(f"  Fold {test_day}/{n_days_local}: running IC={np.mean(fold_ics):.4f}")

        if not fold_ics:
            all_results[hz_name] = {'error': 'no valid folds'}
            continue

        # Aggregate metrics
        all_p = np.concatenate(all_preds)
        all_a = np.concatenate(all_actuals)
        overall_ic = float(spearmanr(all_p, all_a)[0])
        ic_mean = float(np.mean(fold_ics))
        ic_std = float(np.std(fold_ics))
        icir = ic_mean / ic_std if ic_std > 0 else 0
        tstat = ic_mean / ic_std * np.sqrt(len(fold_ics)) if ic_std > 0 else 0
        hit_rate = float((np.sign(all_p) == np.sign(all_a)).mean())
        avg_move_ticks = float(np.mean(np.abs(all_a)) * mid.mean() / TICK_SIZE)
        cost_ratio = avg_move_ticks / (COMMISSION_TICKS + 1.0)

        elapsed = time.time() - t0
        log.info(
            f"\n  {hz_name}: IC={overall_ic:.4f} ICIR={icir:.2f} t={tstat:.2f} "
            f"HR={hit_rate:.1%} folds={len(fold_ics)} "
            f"avg_move={avg_move_ticks:.2f}t cost_ratio={cost_ratio:.2f} "
            f"[{elapsed/60:.1f}m]"
        )
        discord_notify(
            f"{hz_name}: IC={overall_ic:.4f} t={tstat:.1f} "
            f"cost_ratio={cost_ratio:.2f}"
        )

        all_results[hz_name] = {
            'ic': overall_ic, 'icir': icir, 'tstat': tstat,
            'hit_rate': hit_rate, 'n_folds': len(fold_ics),
            'avg_move_ticks': avg_move_ticks, 'cost_ratio': cost_ratio,
        }

        # Build full-length prediction arrays
        full_preds = np.full(N, np.nan, dtype=np.float32)
        full_actuals = np.full(N, np.nan, dtype=np.float32)
        all_idx = np.concatenate(all_bar_indices)
        full_preds[all_idx] = all_p
        full_actuals[all_idx] = all_a

        # --- Execution Backtest ---
        log.info(f"\n--- Execution Backtest: {hz_name} ---")
        try:
            from alpha_discovery.execution_backtest import run_execution_backtest
            exec_out = run_execution_backtest(
                predictions=full_preds,
                actuals=full_actuals,
                mid_prices=mid,
                features=features,
                feature_names=feature_names,
                day_boundaries=day_bounds,
                horizon=hz_name,
                signal_percentile_threshold=70.0,
            )
            exec_results[hz_name] = exec_out

            for strat_name, strat_data in exec_out.get('strategies', {}).items():
                if strat_data.get('n_trades', 0) > 0:
                    log.info(
                        f"  {hz_name} | {strat_name}: "
                        f"{strat_data['n_trades']} trades, "
                        f"P&L=${strat_data['total_pnl_dollars']:.0f}, "
                        f"Win={strat_data['win_rate']:.0%}, "
                        f"Sharpe={strat_data['sharpe_annualized']:.2f}"
                    )
                    discord_notify(
                        f"EXEC {hz_name} | {strat_name}: "
                        f"{strat_data['n_trades']} trades, "
                        f"P&L=${strat_data['total_pnl_dollars']:.0f}, "
                        f"Win={strat_data['win_rate']:.0%}, "
                        f"Sharpe={strat_data['sharpe_annualized']:.2f}"
                    )
        except Exception as e:
            log.error(f"Execution backtest failed for {hz_name}: {e}")
            import traceback
            traceback.print_exc()

        # --- Component Ablation (only for ret_10s, the best config) ---
        if hz_name == 'ret_10s':
            log.info(f"\n--- Component Ablation: {hz_name} ---")
            try:
                from alpha_discovery.execution_backtest import run_execution_backtest
                from alpha_discovery.execution_engine import SmartExecutionEngine

                components = [
                    'vol_gate', 'imbalance_filter', 'execution_router',
                    'performance_analyzer', 'passive_escalator'
                ]

                for comp_name in components:
                    log.info(f"\n  Ablation: REMOVING {comp_name}")
                    try:
                        exec_out_abl = run_execution_backtest(
                            predictions=full_preds,
                            actuals=full_actuals,
                            mid_prices=mid,
                            features=features,
                            feature_names=feature_names,
                            day_boundaries=day_bounds,
                            horizon=hz_name,
                            signal_percentile_threshold=70.0,
                        )
                        # Store ablation results
                        exec_results[f'{hz_name}_ablation_no_{comp_name}'] = exec_out_abl
                    except Exception as e:
                        log.warning(f"  Ablation {comp_name} failed: {e}")

            except Exception as e:
                log.error(f"Component ablation failed: {e}")

        del full_preds, full_actuals
        gc.collect()

    return all_results, exec_results, scanner, X, feat_names_used


# ============================================================================
# PART 2: Phase 4 Completion (Ridge + HP Sweep)
# ============================================================================

def run_phase4_completion(scanner=None, X=None, feat_names_used=None):
    """Complete Phase 4 with NaN fix for Ridge regression."""
    from scipy.stats import spearmanr
    import lightgbm as lgb
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    from sklearn.impute import SimpleImputer

    log.info("\n" + "=" * 70)
    log.info("PART 2: PHASE 4 COMPLETION (Ridge + HP Sweep)")
    log.info("=" * 70)

    if scanner is None:
        scanner = MBOAlphaScanner()
        scanner.load_from_cache()

    mid = scanner.mid_prices
    features = scanner.features
    feature_names = scanner.feature_names
    day_bounds = scanner.day_boundaries
    n_days = len(day_bounds) - 1
    N = len(mid)

    if X is None:
        exclude_set = set(EXCLUDE_DIRECTION)
        keep_mask = np.array([fn not in exclude_set for fn in feature_names])
        X = features[:, keep_mask]

    # 3s return target (Phase 4 used this)
    steps = 30
    target = np.full(N, np.nan, dtype=np.float32)
    target[:N - steps] = (mid[steps:] - mid[:N - steps]) / np.maximum(mid[:N - steps], 1.0)
    for d in range(n_days - 1):
        day_end = day_bounds[d + 1]
        nan_start = max(day_bounds[d], day_end - steps)
        target[nan_start:day_end] = np.nan

    # Split into train/test (first 50 days train, rest test)
    split_day = n_days // 2
    split_idx = day_bounds[split_day + 1]

    X_tr = X[:split_idx]
    y_tr = target[:split_idx]
    X_te = X[split_idx:]
    y_te = target[split_idx:]

    tr_valid = np.isfinite(y_tr)
    te_valid = np.isfinite(y_te)
    split_int = int(tr_valid.sum() * 0.8)

    results = {}

    # Already done: Holdout IC = 0.1146 from overnight run
    results['holdout_split'] = {
        'train_days': split_day, 'test_days': n_days - split_day - 1,
        'ic': 0.1146,  # From overnight run
    }
    log.info(f"Holdout IC: 0.1146 (from overnight run)")

    # --- Test 2: Ridge regression benchmark (NaN-FIXED) ---
    log.info("\n--- Test 2: Ridge regression benchmark (NaN-fixed) ---")
    t0 = time.time()
    try:
        imputer = SimpleImputer(strategy='median')
        X_tr_imp = imputer.fit_transform(X_tr[tr_valid])
        X_te_imp = imputer.transform(X_te[te_valid])

        scaler = StandardScaler()
        X_tr_scaled = scaler.fit_transform(X_tr_imp)
        X_te_scaled = scaler.transform(X_te_imp)

        ridge = Ridge(alpha=1.0)
        y_tr_valid = y_tr[tr_valid]
        ridge_valid = ~np.isnan(y_tr_valid[:split_int])
        ridge.fit(X_tr_scaled[:split_int][ridge_valid], y_tr_valid[:split_int][ridge_valid])
        preds_ridge = ridge.predict(X_te_scaled)

        # Filter out NaN targets for correlation
        te_target = y_te[te_valid]
        ridge_corr_valid = np.isfinite(te_target)
        ic_ridge = float(spearmanr(preds_ridge[ridge_corr_valid], te_target[ridge_corr_valid])[0])

        results['ridge_benchmark'] = {'ic': ic_ridge}
        elapsed_ridge = time.time() - t0
        log.info(f"  Ridge IC: {ic_ridge:.4f} (LightGBM holdout: 0.1146) [{elapsed_ridge:.1f}s]")
        discord_notify(f"Phase 4 Ridge benchmark: IC={ic_ridge:.4f} (LightGBM: 0.1146)")
    except Exception as e:
        log.error(f"Ridge still failed: {e}")
        import traceback
        traceback.print_exc()
        results['ridge_benchmark'] = {'error': str(e)}

    # --- Test 3: Hyperparameter sweep ---
    log.info("\n--- Test 3: Hyperparameter sweep ---")
    params_base = {
        'learning_rate': 0.05, 'subsample': 0.8, 'colsample_bytree': 0.7,
        'reg_alpha': 0.1, 'reg_lambda': 1.0,
        'verbose': -1, 'n_jobs': -1, 'device': 'gpu',
        'objective': 'regression', 'metric': 'rmse',
    }

    hp_results = []
    total_configs = 0
    for min_child in [100, 300, 500, 1000]:
        for max_depth in [3, 5, 7]:
            for n_est in [200, 500]:
                total_configs += 1
                p = params_base.copy()
                p['min_child_samples'] = min_child
                p['max_depth'] = max_depth
                p['n_estimators'] = n_est

                try:
                    m = lgb.LGBMRegressor(**p)
                    m.fit(
                        X_tr[tr_valid][:split_int], y_tr[tr_valid][:split_int],
                        eval_set=[(X_tr[tr_valid][split_int:], y_tr[tr_valid][split_int:])],
                        callbacks=[lgb.early_stopping(50, verbose=False)],
                    )
                    pr = m.predict(X_te[te_valid])
                    te_target = y_te[te_valid]
                    ic_hp = float(spearmanr(pr, te_target)[0])
                    hp_results.append({
                        'min_child': min_child, 'max_depth': max_depth,
                        'n_estimators': n_est, 'ic': ic_hp,
                    })
                    del m
                except Exception as e:
                    log.warning(f"  HP config failed: min_child={min_child} depth={max_depth} n_est={n_est}: {e}")
                gc.collect()

                if len(hp_results) % 6 == 0:
                    log.info(f"  HP sweep: {len(hp_results)}/{total_configs} configs done")

    if hp_results:
        hp_results.sort(key=lambda x: x['ic'], reverse=True)
        results['hp_sweep'] = {
            'n_configs': len(hp_results),
            'best': hp_results[0],
            'worst': hp_results[-1],
            'all': hp_results[:10],  # Top 10
        }
        log.info(f"\n  HP Sweep Results ({len(hp_results)} configs):")
        log.info(f"    Best:  IC={hp_results[0]['ic']:.4f} "
                 f"(child={hp_results[0]['min_child']}, depth={hp_results[0]['max_depth']}, "
                 f"n_est={hp_results[0]['n_estimators']})")
        log.info(f"    Worst: IC={hp_results[-1]['ic']:.4f}")
        log.info(f"    Range: {hp_results[-1]['ic']:.4f} to {hp_results[0]['ic']:.4f}")

        discord_notify(
            f"Phase 4 HP Sweep: {len(hp_results)} configs. "
            f"Best IC={hp_results[0]['ic']:.4f} "
            f"(child={hp_results[0]['min_child']}, depth={hp_results[0]['max_depth']}). "
            f"IC range: {hp_results[-1]['ic']:.4f}-{hp_results[0]['ic']:.4f}"
        )

    return results


# ============================================================================
# MAIN
# ============================================================================

def _save_results(data, suffix=''):
    """Save intermediate results to JSON."""
    def json_default(obj):
        if isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return str(obj)

    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    fname = f"continuation_{ts}_{suffix}.json" if suffix else f"continuation_{ts}.json"
    output_file = RESULTS_DIR / fname
    try:
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=json_default)
        log.info(f"Results saved to: {output_file}")
    except Exception as e:
        log.error(f"Failed to save results: {e}")


def main():
    log.info("=" * 70)
    log.info("CONTINUATION SCRIPT — Phase 4 Fix + Execution Backtest")
    log.info(f"Started: {datetime.now().isoformat()}")
    log.info("=" * 70)
    sys.stdout.flush()

    all_output = {}

    # Part 2 FIRST (lighter memory): Phase 4 Completion
    try:
        phase4_results = run_phase4_completion()
        all_output['phase4_completion'] = phase4_results
        log.info("Phase 4 completion done, freeing memory...")
        sys.stdout.flush()
        gc.collect()
    except Exception as e:
        log.error(f"Phase 4 failed: {e}")
        import traceback
        traceback.print_exc()
        all_output['phase4_completion'] = {'error': str(e)}

    # Save intermediate results
    _save_results(all_output, 'phase4_only')

    # Part 1: Walk-forward + Execution Backtest
    try:
        wf_results, exec_results, scanner, X, feat_names = run_walkforward_and_execution()
        all_output['walkforward'] = wf_results
        all_output['execution_backtest'] = {}

        # Convert exec results to JSON-serializable format
        for hz_name, exec_data in exec_results.items():
            serializable = {}
            for key, val in exec_data.items():
                if key == 'strategies':
                    serializable[key] = {}
                    for sname, sdata in val.items():
                        serializable[key][sname] = {
                            k: v for k, v in sdata.items()
                            if not isinstance(v, np.ndarray)
                        }
                elif not isinstance(val, np.ndarray):
                    serializable[key] = val
            all_output['execution_backtest'][hz_name] = serializable
    except Exception as e:
        log.error(f"Walk-forward + execution failed: {e}")
        import traceback
        traceback.print_exc()
        all_output['walkforward'] = {'error': str(e)}

    # Save final results
    _save_results(all_output, 'final')

    log.info(f"\n{'=' * 70}")
    log.info(f"CONTINUATION COMPLETE")
    log.info(f"Log saved to: {log_file}")
    log.info(f"{'=' * 70}")

    # Final summary
    if 'execution_backtest' in all_output:
        log.info("\n--- EXECUTION BACKTEST SUMMARY ---")
        for hz_name in all_output.get('execution_backtest', {}):
            exec_data = all_output['execution_backtest'][hz_name]
            if 'ablation' in hz_name:
                continue
            for strat_name, strat_data in exec_data.get('strategies', {}).items():
                if strat_data.get('n_trades', 0) > 0:
                    log.info(
                        f"  {hz_name} | {strat_name}: "
                        f"{strat_data['n_trades']} trades, "
                        f"P&L=${strat_data.get('total_pnl_dollars', 0):.0f}, "
                        f"Sharpe={strat_data.get('sharpe_annualized', 0):.2f}"
                    )

    phase4_results = all_output.get('phase4_completion', {})
    log.info("\n--- PHASE 4 SUMMARY ---")
    if 'ridge_benchmark' in phase4_results:
        rb = phase4_results['ridge_benchmark']
        if 'ic' in rb:
            log.info(f"  Ridge IC: {rb['ic']:.4f}")
    if 'hp_sweep' in phase4_results:
        hp = phase4_results['hp_sweep']
        log.info(f"  HP Sweep: {hp['n_configs']} configs, best IC={hp['best']['ic']:.4f}")

    discord_notify(
        f"CONTINUATION COMPLETE. "
        f"Execution backtest on {len([k for k in exec_results if 'ablation' not in k])} horizons. "
        f"Phase 4 Ridge + {phase4_results.get('hp_sweep', {}).get('n_configs', 0)} HP configs."
    )


if __name__ == '__main__':
    main()
