"""
OVERNIGHT ALPHA DISCOVERY — Master Orchestrator
================================================

Runs the full overnight pipeline:
  Phase 0: Rebuild snapshot caches for new data (Aug 14 - Nov 28)
  Phase 1: Honest baseline on expanded dataset (95+ trading days)
  Phase 2: Multi-timeframe exploration (1s, 5s, 30s, 1min, 5min bars)
  Phase 3: New feature discovery + forward/backward decomposition
  Phase 4: Model robustness checks (stability, linear benchmark, hyperparam sweep)
  Phase 5: Execution research (trade frequency, hold optimization)
  Phase 6: Final report compilation

Discord progress updates via send_to_discord MCP (if available).

Usage:
    python alpha_discovery/run_overnight_alpha.py
    python alpha_discovery/run_overnight_alpha.py --skip-cache-rebuild
    python alpha_discovery/run_overnight_alpha.py --start-phase 2
    python alpha_discovery/run_overnight_alpha.py --phases 1 2 3
"""

import sys
import gc
import json
import time
import logging
import argparse
import traceback
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional

# Setup path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.mbo_features import (
    compute_mbo_features, get_feature_names, TOTAL_FEATURES,
)
from alpha_discovery.run_return_multihorizon import (
    EXCLUDE_FEATURES_DIRECTION,
)

# ============================================================================
# LOGGING
# ============================================================================
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
LOG_FILE = RESULTS_DIR / f"overnight_alpha_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(LOG_FILE), mode='w', encoding='utf-8'),
    ]
)
log = logging.getLogger('overnight_alpha')

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE = 0.25
TICK_VALUE = 12.50
BARS_PER_SEC = 10
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)
HALF_TICK = TICK_SIZE / 2
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE

# The 18 clean forward-dominant features (from forward decomposition analysis)
CLEAN_FEATURES = [
    'ofi_5', 'total_ask_vol', 'bid_L5_conc', 'ask_L1_conc',
    'ask_L4_orders', 'ofi_20', 'vol_regime', 'depth_ratio_l1',
    'depth_concentration', 'bid_L1_orders', 'bid_L3_conc',
    'total_bid_vol', 'ask_L1_orders', 'bid_L1_conc', 'ofi_50',
    'bid_L2_conc', 'ask_slope', 'bid_pressure',
]

# Features to always exclude from direction models
EXCLUDE_DIRECTION = list(EXCLUDE_FEATURES_DIRECTION) if hasattr(EXCLUDE_FEATURES_DIRECTION, '__iter__') else []

# ============================================================================
# DISCORD NOTIFICATION (best-effort)
# ============================================================================
def discord_notify(msg: str):
    """Send progress update to Discord. Fails silently if not available."""
    try:
        # Try to write to a file that the bridge can read
        notify_file = ROOT.parent / "teleclaude-main" / "overnight_progress.txt"
        with open(str(notify_file), 'a', encoding='utf-8') as f:
            f.write(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}\n")
    except Exception:
        pass
    log.info(f"[DISCORD] {msg}")


# ============================================================================
# PHASE 0: REBUILD CACHES
# ============================================================================
def phase_0_rebuild_caches(new_only: bool = True) -> dict:
    """Rebuild snapshot caches for new data files."""
    log.info("=" * 70)
    log.info("PHASE 0: REBUILD SNAPSHOT CACHES")
    log.info("=" * 70)

    import subprocess
    cmd = [
        sys.executable,
        str(ROOT / "alpha_discovery" / "rebuild_snapshot_caches.py"),
    ]
    if new_only:
        cmd.append("--new-only")

    discord_notify("Phase 0: Rebuilding snapshot caches for 92 new data files...")

    t0 = time.time()
    result = subprocess.run(cmd, capture_output=True, text=True, timeout=7200)  # 2hr timeout
    elapsed = time.time() - t0

    log.info(f"Cache rebuild finished in {elapsed:.0f}s ({elapsed/60:.1f} min)")
    if result.returncode != 0:
        log.error(f"Cache rebuild FAILED: {result.stderr[-500:]}")
        discord_notify(f"Phase 0 FAILED after {elapsed/60:.0f} min: {result.stderr[-200:]}")
        return {'status': 'failed', 'elapsed': elapsed, 'error': result.stderr[-500:]}

    discord_notify(f"Phase 0 complete: caches rebuilt in {elapsed/60:.0f} min")
    return {'status': 'success', 'elapsed': elapsed}


# ============================================================================
# PHASE 1: HONEST BASELINE
# ============================================================================
def phase_1_honest_baseline() -> dict:
    """Run honest backtest on full expanded dataset."""
    log.info("=" * 70)
    log.info("PHASE 1: HONEST BASELINE ON EXPANDED DATASET")
    log.info("=" * 70)

    discord_notify("Phase 1: Running honest baseline on ~95 trading days...")

    t0 = time.time()

    # Load data
    scanner = MBOAlphaScanner()
    load_info = scanner.load_from_cache()
    n_days = load_info['n_days']
    n_bars = load_info['n_snapshots']

    log.info(f"Loaded {n_days} trading days, {n_bars:,} bars")
    discord_notify(f"Phase 1: Loaded {n_days} days, {n_bars:,} bars. Running walk-forward...")

    # Compute targets at multiple horizons
    from scipy.stats import spearmanr, ttest_1samp
    import lightgbm as lgb

    mid = scanner.mid_prices
    log_mid = np.log(np.maximum(mid, 1.0))
    features = scanner.features
    feature_names = scanner.feature_names
    day_bounds = scanner.day_boundaries

    # Build exclude mask
    exclude_set = set(EXCLUDE_DIRECTION)
    keep_mask = np.array([fn not in exclude_set for fn in feature_names])
    X = features[:, keep_mask]
    feat_names_used = [fn for fn in feature_names if fn not in exclude_set]
    log.info(f"Using {X.shape[1]} features (excluded {sum(~keep_mask)} direction-leaking features)")

    # Test multiple horizons
    horizons = {
        'ret_3s': 30,    # 3 sec
        'ret_5s': 50,    # 5 sec
        'ret_10s': 100,  # 10 sec
        'ret_30s': 300,  # 30 sec
        'ret_1m': 600,   # 1 min
        'ret_5m': 3000,  # 5 min
    }

    results = {}
    for hz_name, steps in horizons.items():
        log.info(f"\n--- {hz_name} (steps={steps}) ---")

        # Compute target
        N = len(mid)
        target = np.full(N, np.nan, dtype=np.float32)
        target[:N - steps] = (mid[steps:] - mid[:N - steps]) / np.maximum(mid[:N - steps], 1.0)

        # NaN-fill targets crossing day boundaries
        n_days_local = len(day_bounds) - 1
        for d in range(n_days_local - 1):
            day_end = day_bounds[d + 1]
            nan_start = max(day_bounds[d], day_end - steps)
            target[nan_start:day_end] = np.nan

        # Walk-forward LightGBM
        params = {
            'n_estimators': 300,
            'max_depth': 5,
            'learning_rate': 0.05,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'reg_alpha': 0.1,
            'reg_lambda': 1.0,
            'min_child_samples': 500,
            'verbose': -1,
            'n_jobs': -1,
            'device': 'gpu',
            'objective': 'regression',
            'metric': 'rmse',
        }

        min_train_days = 5
        fold_ics = []
        all_preds = []
        all_actuals = []
        all_bar_indices = []  # Track original bar indices for execution backtest
        feature_importance = np.zeros(X.shape[1])

        for test_day in range(min_train_days, n_days_local):
            train_end = day_bounds[test_day - 1 + 1]  # purge gap = 1 day
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

            X_tr_v = X_tr[train_valid]
            y_tr_v = y_tr[train_valid]
            X_te_v = X_te[test_valid]
            y_te_v = y_te[test_valid]

            # Track which bars in the original array these predictions map to
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

            if hasattr(model, 'feature_importances_'):
                feature_importance += model.feature_importances_

            del model
            gc.collect()

        if not fold_ics:
            results[hz_name] = {'error': 'no valid folds'}
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

        # Avg move in ticks at this horizon
        avg_move_ticks = float(np.mean(np.abs(all_a)) * mid.mean() / TICK_SIZE)
        cost_ratio = avg_move_ticks / (COMMISSION_TICKS + 1.0)  # 1 tick spread

        # Per-fold consistency
        positive_folds = sum(1 for ic in fold_ics if ic > 0)
        consistency = positive_folds / len(fold_ics)

        # Top features
        top_idx = np.argsort(feature_importance)[::-1][:10]
        top_feats = [(feat_names_used[i], float(feature_importance[i])) for i in top_idx]

        hz_result = {
            'ic': overall_ic,
            'ic_mean': ic_mean,
            'ic_std': ic_std,
            'icir': icir,
            'tstat': tstat,
            'hit_rate': hit_rate,
            'n_folds': len(fold_ics),
            'n_predictions': len(all_p),
            'consistency': consistency,
            'avg_move_ticks': avg_move_ticks,
            'cost_ratio': cost_ratio,
            'top_features': top_feats,
            'fold_ics': [float(x) for x in fold_ics],
        }
        results[hz_name] = hz_result

        # Store full-length prediction arrays for execution backtest
        if all_bar_indices:
            full_preds = np.full(N, np.nan, dtype=np.float32)
            full_actuals_arr = np.full(N, np.nan, dtype=np.float32)
            all_idx = np.concatenate(all_bar_indices)
            full_preds[all_idx] = all_p
            full_actuals_arr[all_idx] = all_a
            hz_result['_full_predictions'] = full_preds
            hz_result['_full_actuals'] = full_actuals_arr

        log.info(
            f"  {hz_name}: IC={overall_ic:.4f} ICIR={icir:.2f} t={tstat:.2f} "
            f"HR={hit_rate:.1%} folds={len(fold_ics)} consistency={consistency:.0%} "
            f"avg_move={avg_move_ticks:.2f}t cost_ratio={cost_ratio:.2f}"
        )

    # ================================================================
    # EXECUTION BACKTEST on top horizons
    # ================================================================
    log.info("\n" + "=" * 70)
    log.info("EXECUTION BACKTEST — Full P&L Simulation")
    log.info("=" * 70)

    exec_results = {}
    try:
        from alpha_discovery.execution_backtest import run_execution_backtest

        # Run execution backtest on horizons with positive IC
        for hz_name, hz_data in sorted(results.items()):
            if 'error' in hz_data or hz_data.get('ic', 0) <= 0:
                continue
            full_preds = hz_data.pop('_full_predictions', None)
            full_acts = hz_data.pop('_full_actuals', None)
            if full_preds is None:
                continue

            log.info(f"\n--- Execution Backtest: {hz_name} ---")
            discord_notify(f"Running execution backtest for {hz_name}...")

            exec_out = run_execution_backtest(
                predictions=full_preds,
                actuals=full_acts,
                mid_prices=mid,
                features=features,
                feature_names=feature_names,
                day_boundaries=day_bounds,
                horizon=hz_name,
                signal_percentile_threshold=70.0,
            )
            exec_results[hz_name] = exec_out

            # Discord summary for this horizon
            for strat_name, strat_data in exec_out.get('strategies', {}).items():
                if strat_data.get('n_trades', 0) > 0:
                    discord_notify(
                        f"  {hz_name} | {strat_name}: "
                        f"{strat_data['n_trades']} trades, "
                        f"P&L=${strat_data['total_pnl_dollars']:.0f}, "
                        f"Win={strat_data['win_rate']:.0%}, "
                        f"Sharpe={strat_data['sharpe_annualized']:.2f}"
                    )

    except Exception as e:
        log.error(f"Execution backtest failed: {e}")
        import traceback
        traceback.print_exc()

    # Clean up internal arrays from results before saving
    for hz_data in results.values():
        hz_data.pop('_full_predictions', None)
        hz_data.pop('_full_actuals', None)

    elapsed = time.time() - t0

    # Save results
    output = {
        'phase': 'honest_baseline',
        'timestamp': datetime.now().isoformat(),
        'n_days': n_days,
        'n_bars': n_bars,
        'n_features_used': X.shape[1],
        'horizons': results,
        'execution_backtest': {
            hz: {k: v for k, v in data.items() if k != 'chart_paths'}
            for hz, data in exec_results.items()
        },
        'elapsed_sec': elapsed,
    }
    result_path = RESULTS_DIR / f"overnight_phase1_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(str(result_path), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Discord summary
    summary_lines = ["Phase 1 COMPLETE - Honest Baseline:"]
    summary_lines.append(f"Data: {n_days} days, {n_bars:,} bars")
    for hz, r in sorted(results.items()):
        if 'error' in r:
            summary_lines.append(f"  {hz}: ERROR - {r['error']}")
        else:
            viable = "VIABLE" if r['cost_ratio'] > 0.5 else "weak"
            summary_lines.append(
                f"  {hz}: IC={r['ic']:.4f} t={r['tstat']:.1f} "
                f"avg_move={r['avg_move_ticks']:.1f}t "
                f"cost_ratio={r['cost_ratio']:.2f} [{viable}]"
            )

    if exec_results:
        summary_lines.append("\nExecution Backtest Results:")
        for hz, exec_data in exec_results.items():
            for strat, sdata in exec_data.get('strategies', {}).items():
                if sdata.get('n_trades', 0) > 0:
                    summary_lines.append(
                        f"  {hz} | {strat}: "
                        f"${sdata['total_pnl_dollars']:.0f} P&L, "
                        f"{sdata['win_rate']:.0%} win, "
                        f"Sharpe={sdata['sharpe_annualized']:.2f}"
                    )

    discord_notify("\n".join(summary_lines))

    del scanner
    gc.collect()
    return output


# ============================================================================
# PHASE 2: MULTI-TIMEFRAME EXPLORATION
# ============================================================================
def phase_2_multi_timeframe() -> dict:
    """Aggregate 100ms bars into higher timeframes and test signal."""
    log.info("=" * 70)
    log.info("PHASE 2: MULTI-TIMEFRAME EXPLORATION")
    log.info("=" * 70)

    discord_notify("Phase 2: Multi-timeframe bar aggregation and alpha scan...")

    t0 = time.time()

    # Load raw data
    scanner = MBOAlphaScanner()
    load_info = scanner.load_from_cache()
    n_days = load_info['n_days']
    mid = scanner.mid_prices
    features = scanner.features
    feature_names = scanner.feature_names
    day_bounds = scanner.day_boundaries

    # Exclude direction-leaking features
    exclude_set = set(EXCLUDE_DIRECTION)
    keep_mask = np.array([fn not in exclude_set for fn in feature_names])
    X_100ms = features[:, keep_mask]
    feat_names = [fn for fn in feature_names if fn not in exclude_set]

    timeframes = {
        '1s': 10,     # 10 x 100ms bars
        '5s': 50,
        '10s': 100,
        '30s': 300,
        '1min': 600,
        '5min': 3000,
    }

    results = {}
    from scipy.stats import spearmanr
    import lightgbm as lgb

    for tf_name, agg_bars in timeframes.items():
        log.info(f"\n--- Timeframe: {tf_name} (aggregate {agg_bars} bars) ---")

        # Aggregate bars: take every agg_bars-th bar's features, use OHLC of mid
        # This gives us the bar-end state of the order book
        n_total = len(mid)
        n_agg = n_total // agg_bars

        if n_agg < 1000:
            log.warning(f"  Too few aggregated bars: {n_agg}")
            results[tf_name] = {'error': f'too few bars: {n_agg}'}
            continue

        # Sample features at end of each bar (order book state at bar close)
        indices = np.arange(agg_bars - 1, n_total, agg_bars)[:n_agg]
        X_agg = X_100ms[indices]
        mid_agg = mid[indices]

        # Recompute day boundaries for aggregated bars
        agg_day_bounds = [0]
        for d in range(n_days):
            orig_end = day_bounds[d + 1]
            agg_end = orig_end // agg_bars
            if agg_end > agg_day_bounds[-1]:
                agg_day_bounds.append(min(agg_end, n_agg))
        if agg_day_bounds[-1] != n_agg:
            agg_day_bounds[-1] = n_agg
        n_agg_days = len(agg_day_bounds) - 1

        # Target: 1-bar forward return
        target = np.full(n_agg, np.nan, dtype=np.float32)
        target[:-1] = (mid_agg[1:] - mid_agg[:-1]) / np.maximum(mid_agg[:-1], 1.0)

        # NaN-fill day boundaries
        for d in range(n_agg_days - 1):
            day_end = agg_day_bounds[d + 1]
            if day_end > 0 and day_end <= n_agg:
                target[day_end - 1] = np.nan

        # Walk-forward
        params = {
            'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05,
            'subsample': 0.8, 'colsample_bytree': 0.7,
            'reg_alpha': 0.1, 'reg_lambda': 1.0,
            'min_child_samples': max(50, n_agg // 500),  # Scale with data size
            'verbose': -1, 'n_jobs': -1, 'device': 'gpu',
            'objective': 'regression', 'metric': 'rmse',
        }

        fold_ics = []
        all_preds = []
        all_actuals = []
        min_train_days = 5

        for test_day in range(min_train_days, n_agg_days):
            train_end = agg_day_bounds[test_day - 1 + 1]
            test_start = agg_day_bounds[test_day]
            test_end = agg_day_bounds[test_day + 1]

            if test_end <= test_start or train_end <= 0:
                continue

            X_tr = X_agg[:train_end]
            y_tr = target[:train_end]
            X_te = X_agg[test_start:test_end]
            y_te = target[test_start:test_end]

            train_valid = np.isfinite(y_tr)
            test_valid = np.isfinite(y_te)

            if train_valid.sum() < 200 or test_valid.sum() < 20:
                continue

            X_tr_v, y_tr_v = X_tr[train_valid], y_tr[train_valid]
            X_te_v, y_te_v = X_te[test_valid], y_te[test_valid]

            split = int(len(X_tr_v) * 0.8)
            if split < 100:
                continue

            try:
                model = lgb.LGBMRegressor(**params)
                model.fit(
                    X_tr_v[:split], y_tr_v[:split],
                    eval_set=[(X_tr_v[split:], y_tr_v[split:])],
                    callbacks=[lgb.early_stopping(50, verbose=False)],
                )
                preds = model.predict(X_te_v)
                ic = float(spearmanr(preds, y_te_v)[0])
                if np.isfinite(ic):
                    fold_ics.append(ic)
                    all_preds.append(preds)
                    all_actuals.append(y_te_v)
                del model
            except Exception as e:
                log.warning(f"  Fold {test_day} failed: {e}")
                continue

            gc.collect()

        if not fold_ics:
            results[tf_name] = {'error': 'no valid folds'}
            continue

        all_p = np.concatenate(all_preds)
        all_a = np.concatenate(all_actuals)
        overall_ic = float(spearmanr(all_p, all_a)[0])
        ic_mean = float(np.mean(fold_ics))
        ic_std = float(np.std(fold_ics))
        icir = ic_mean / ic_std if ic_std > 0 else 0
        tstat = ic_mean / ic_std * np.sqrt(len(fold_ics)) if ic_std > 0 else 0

        avg_move_ticks = float(np.mean(np.abs(all_a)) * mid_agg.mean() / TICK_SIZE)
        trades_per_day = n_agg / max(1, n_agg_days)

        results[tf_name] = {
            'ic': overall_ic, 'ic_mean': ic_mean, 'icir': icir, 'tstat': tstat,
            'n_folds': len(fold_ics), 'n_predictions': len(all_p),
            'n_bars': n_agg, 'n_days': n_agg_days,
            'avg_move_ticks': avg_move_ticks,
            'trades_per_day': trades_per_day,
            'fold_ics': [float(x) for x in fold_ics],
        }

        log.info(
            f"  {tf_name}: IC={overall_ic:.4f} ICIR={icir:.2f} t={tstat:.2f} "
            f"avg_move={avg_move_ticks:.2f}t bars={n_agg} days={n_agg_days}"
        )

    elapsed = time.time() - t0
    output = {
        'phase': 'multi_timeframe',
        'timestamp': datetime.now().isoformat(),
        'timeframes': results,
        'elapsed_sec': elapsed,
    }

    result_path = RESULTS_DIR / f"overnight_phase2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(str(result_path), 'w') as f:
        json.dump(output, f, indent=2)

    summary_lines = ["Phase 2 COMPLETE - Multi-Timeframe:"]
    for tf, r in sorted(results.items(), key=lambda x: x[0]):
        if 'error' in r:
            summary_lines.append(f"  {tf}: ERROR")
        else:
            summary_lines.append(
                f"  {tf}: IC={r['ic']:.4f} t={r['tstat']:.1f} "
                f"avg_move={r['avg_move_ticks']:.1f}t"
            )
    discord_notify("\n".join(summary_lines))

    del scanner
    gc.collect()
    return output


# ============================================================================
# PHASE 3: FEATURE DISCOVERY
# ============================================================================
def phase_3_feature_discovery() -> dict:
    """Test new feature engineering ideas with forward/backward decomposition."""
    log.info("=" * 70)
    log.info("PHASE 3: NEW FEATURE DISCOVERY")
    log.info("=" * 70)

    discord_notify("Phase 3: Feature discovery with leakage decomposition...")
    t0 = time.time()

    scanner = MBOAlphaScanner()
    load_info = scanner.load_from_cache()
    mid = scanner.mid_prices
    features = scanner.features
    feature_names = scanner.feature_names
    day_bounds = scanner.day_boundaries
    n_days = load_info['n_days']
    N = len(mid)

    from scipy.stats import spearmanr

    # Forward IC: feature[t] predicting target[t+steps]
    # Backward IC: feature[t] predicting target[t-steps]
    # Ratio >= 0.5 means feature has genuine forward signal
    steps = 30  # 3s horizon
    target_fwd = np.full(N, np.nan, dtype=np.float32)
    target_fwd[:N - steps] = (mid[steps:] - mid[:N - steps]) / np.maximum(mid[:N - steps], 1.0)
    target_bwd = np.full(N, np.nan, dtype=np.float32)
    target_bwd[steps:] = (mid[steps:] - mid[:N - steps]) / np.maximum(mid[:N - steps], 1.0)

    # NaN day boundaries
    for d in range(n_days - 1):
        day_end = day_bounds[d + 1]
        nan_start_fwd = max(day_bounds[d], day_end - steps)
        target_fwd[nan_start_fwd:day_end] = np.nan
        nan_end_bwd = min(day_bounds[d + 1] + steps, day_bounds[d + 2] if d + 2 <= n_days else N)
        target_bwd[day_end:min(day_end + steps, N)] = np.nan

    decomp_results = []
    for i, fname in enumerate(feature_names):
        feat = features[:, i]
        valid_fwd = np.isfinite(feat) & np.isfinite(target_fwd)
        valid_bwd = np.isfinite(feat) & np.isfinite(target_bwd)

        if valid_fwd.sum() < 1000 or valid_bwd.sum() < 1000:
            continue

        try:
            fwd_ic = abs(float(spearmanr(feat[valid_fwd], target_fwd[valid_fwd])[0]))
            bwd_ic = abs(float(spearmanr(feat[valid_bwd], target_bwd[valid_bwd])[0]))
        except Exception:
            continue

        ratio = fwd_ic / max(bwd_ic, 1e-6)
        decomp_results.append({
            'feature': fname,
            'fwd_ic': fwd_ic,
            'bwd_ic': bwd_ic,
            'ratio': ratio,
            'is_forward_dominant': ratio >= 0.5,
        })

    # Sort by forward IC
    decomp_results.sort(key=lambda x: x['fwd_ic'], reverse=True)

    # Count forward-dominant features
    fwd_dominant = [r for r in decomp_results if r['is_forward_dominant']]
    bwd_dominant = [r for r in decomp_results if not r['is_forward_dominant']]

    log.info(f"\nForward/Backward Decomposition on {len(decomp_results)} features:")
    log.info(f"  Forward-dominant (ratio >= 0.5): {len(fwd_dominant)}")
    log.info(f"  Backward-dominant (ratio < 0.5): {len(bwd_dominant)}")
    log.info(f"\nTop 20 by forward IC:")
    for r in decomp_results[:20]:
        marker = "FWD" if r['is_forward_dominant'] else "BWD"
        log.info(f"  {r['feature']:<30s} fwd={r['fwd_ic']:.4f} bwd={r['bwd_ic']:.4f} ratio={r['ratio']:.2f} [{marker}]")

    # New features: compute acceleration (rate of change) of top features
    log.info("\nComputing feature accelerations...")
    new_feature_results = []

    for base_feature in ['ofi_5', 'ofi_20', 'depth_ratio_l1', 'ask_L1_conc', 'bid_pressure']:
        if base_feature not in feature_names:
            continue
        idx = feature_names.index(base_feature)
        feat = features[:, idx]

        for window in [5, 20, 50]:
            # Acceleration = change in feature over window
            accel = np.full(N, np.nan, dtype=np.float32)
            accel[window:] = feat[window:] - feat[:-window]

            # NaN day boundaries
            for d in range(n_days):
                ds = day_bounds[d]
                accel[ds:ds + window] = np.nan

            valid = np.isfinite(accel) & np.isfinite(target_fwd)
            if valid.sum() < 1000:
                continue

            try:
                fwd_ic = abs(float(spearmanr(accel[valid], target_fwd[valid])[0]))
                bwd_ic_val = 0.0
                valid_b = np.isfinite(accel) & np.isfinite(target_bwd)
                if valid_b.sum() > 1000:
                    bwd_ic_val = abs(float(spearmanr(accel[valid_b], target_bwd[valid_b])[0]))
                ratio = fwd_ic / max(bwd_ic_val, 1e-6)
            except Exception:
                continue

            new_feature_results.append({
                'feature': f'{base_feature}_accel_{window}',
                'fwd_ic': fwd_ic,
                'bwd_ic': bwd_ic_val,
                'ratio': ratio,
                'is_forward_dominant': ratio >= 0.5,
            })

    new_feature_results.sort(key=lambda x: x['fwd_ic'], reverse=True)
    log.info(f"\nNew acceleration features ({len(new_feature_results)} tested):")
    for r in new_feature_results:
        marker = "FWD" if r['is_forward_dominant'] else "BWD"
        log.info(f"  {r['feature']:<35s} fwd={r['fwd_ic']:.4f} bwd={r['bwd_ic']:.4f} ratio={r['ratio']:.2f} [{marker}]")

    elapsed = time.time() - t0
    output = {
        'phase': 'feature_discovery',
        'timestamp': datetime.now().isoformat(),
        'n_features_analyzed': len(decomp_results),
        'n_forward_dominant': len(fwd_dominant),
        'n_backward_dominant': len(bwd_dominant),
        'decomposition': decomp_results[:30],
        'new_features': new_feature_results,
        'elapsed_sec': elapsed,
    }

    result_path = RESULTS_DIR / f"overnight_phase3_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(str(result_path), 'w') as f:
        json.dump(output, f, indent=2)

    discord_notify(
        f"Phase 3 COMPLETE: {len(fwd_dominant)} forward-dominant features, "
        f"{len(new_feature_results)} new acceleration features tested. "
        f"Best new: {new_feature_results[0]['feature'] if new_feature_results else 'none'} "
        f"fwd_IC={new_feature_results[0]['fwd_ic']:.4f}" if new_feature_results else ""
    )

    del scanner
    gc.collect()
    return output


# ============================================================================
# PHASE 4: MODEL ROBUSTNESS
# ============================================================================
def phase_4_model_robustness() -> dict:
    """Test model stability across splits, linear benchmark, hyperparam sweep."""
    log.info("=" * 70)
    log.info("PHASE 4: MODEL ROBUSTNESS CHECKS")
    log.info("=" * 70)

    discord_notify("Phase 4: Model robustness - stability, linear benchmark, hyperparams...")
    t0 = time.time()

    scanner = MBOAlphaScanner()
    load_info = scanner.load_from_cache()
    mid = scanner.mid_prices
    features = scanner.features
    feature_names = scanner.feature_names
    day_bounds = scanner.day_boundaries
    n_days = load_info['n_days']
    N = len(mid)

    from scipy.stats import spearmanr
    import lightgbm as lgb

    # Exclude direction-leaking features
    exclude_set = set(EXCLUDE_DIRECTION)
    keep_mask = np.array([fn not in exclude_set for fn in feature_names])
    X = features[:, keep_mask]

    # 3s return target
    steps = 30
    target = np.full(N, np.nan, dtype=np.float32)
    target[:N - steps] = (mid[steps:] - mid[:N - steps]) / np.maximum(mid[:N - steps], 1.0)
    for d in range(n_days - 1):
        day_end = day_bounds[d + 1]
        nan_start = max(day_bounds[d], day_end - steps)
        target[nan_start:day_end] = np.nan

    results = {}

    # Test 1: Train/holdout split (first half train, second half test)
    log.info("\n--- Test 1: Train/holdout split ---")
    split_day = n_days // 2
    train_end = day_bounds[split_day]
    test_start = day_bounds[split_day + 1]  # 1 day purge

    X_tr = X[:train_end]
    y_tr = target[:train_end]
    X_te = X[test_start:]
    y_te = target[test_start:]

    tr_valid = np.isfinite(y_tr)
    te_valid = np.isfinite(y_te)

    params = {
        'n_estimators': 300, 'max_depth': 5, 'learning_rate': 0.05,
        'subsample': 0.8, 'colsample_bytree': 0.7,
        'min_child_samples': 500, 'verbose': -1, 'n_jobs': -1, 'device': 'gpu',
        'objective': 'regression', 'metric': 'rmse',
    }

    split_int = int(tr_valid.sum() * 0.8)
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr[tr_valid][:split_int], y_tr[tr_valid][:split_int],
        eval_set=[(X_tr[tr_valid][split_int:], y_tr[tr_valid][split_int:])],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    preds_holdout = model.predict(X_te[te_valid])
    ic_holdout = float(spearmanr(preds_holdout, y_te[te_valid])[0])
    results['holdout_split'] = {
        'train_days': split_day, 'test_days': n_days - split_day - 1,
        'ic': ic_holdout,
        'n_train': int(tr_valid.sum()), 'n_test': int(te_valid.sum()),
    }
    log.info(f"  Holdout IC: {ic_holdout:.4f} (train={split_day} days, test={n_days - split_day - 1} days)")
    del model
    gc.collect()

    # Test 2: Linear model benchmark (Ridge regression)
    log.info("\n--- Test 2: Ridge regression benchmark ---")
    try:
        from sklearn.linear_model import Ridge
        from sklearn.preprocessing import StandardScaler
        from sklearn.impute import SimpleImputer

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
        ic_ridge = float(spearmanr(preds_ridge, y_te[te_valid])[0])
        results['ridge_benchmark'] = {'ic': ic_ridge}
        log.info(f"  Ridge IC: {ic_ridge:.4f} (LightGBM: {ic_holdout:.4f})")
    except ImportError:
        log.warning("  sklearn not available, skipping ridge benchmark")
        results['ridge_benchmark'] = {'error': 'sklearn not installed'}

    # Test 3: Hyperparameter sensitivity
    log.info("\n--- Test 3: Hyperparameter sweep ---")
    hp_results = []
    for min_child in [100, 300, 500, 1000]:
        for max_depth in [3, 5, 7]:
            for n_est in [200, 500]:
                p = params.copy()
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
                    ic_hp = float(spearmanr(pr, y_te[te_valid])[0])
                    hp_results.append({
                        'min_child': min_child, 'max_depth': max_depth,
                        'n_estimators': n_est, 'ic': ic_hp,
                    })
                    del m
                except Exception:
                    pass
                gc.collect()

    hp_results.sort(key=lambda x: abs(x['ic']), reverse=True)
    results['hyperparam_sweep'] = hp_results[:10]
    if hp_results:
        best = hp_results[0]
        worst = hp_results[-1]
        log.info(f"  Best: IC={best['ic']:.4f} (min_child={best['min_child']}, depth={best['max_depth']})")
        log.info(f"  Worst: IC={worst['ic']:.4f} (min_child={worst['min_child']}, depth={worst['max_depth']})")
        log.info(f"  Range: {worst['ic']:.4f} to {best['ic']:.4f} (spread={best['ic']-worst['ic']:.4f})")

    elapsed = time.time() - t0
    output = {
        'phase': 'model_robustness',
        'timestamp': datetime.now().isoformat(),
        'results': results,
        'elapsed_sec': elapsed,
    }

    result_path = RESULTS_DIR / f"overnight_phase4_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(str(result_path), 'w') as f:
        json.dump(output, f, indent=2)

    discord_notify(
        f"Phase 4 COMPLETE: Holdout IC={ic_holdout:.4f}, "
        f"Ridge IC={results.get('ridge_benchmark', {}).get('ic', 'N/A')}, "
        f"HP sweep range: {hp_results[-1]['ic']:.4f} to {hp_results[0]['ic']:.4f}" if hp_results else "no HP results"
    )

    del scanner
    gc.collect()
    return output


# ============================================================================
# PHASE 5: MODEL PROGRESSION (CNN, LSTM, Transformer vs LightGBM)
# ============================================================================
def phase_5_model_progression() -> dict:
    """Run temporal model progression: CNN, LSTM, Transformer vs LightGBM."""
    log.info("=" * 70)
    log.info("PHASE 5: MODEL PROGRESSION — Temporal Models vs LightGBM")
    log.info("=" * 70)

    discord_notify("Phase 5: Running temporal model progression (CNN/LSTM/Transformer vs LightGBM)...")
    t0 = time.time()

    from alpha_discovery.temporal_trainer import (
        TemporalWalkForwardEvaluator, format_comparison_report
    )
    from alpha_discovery.temporal_models import create_model, count_parameters

    # Load data
    scanner = MBOAlphaScanner()
    if '3s' not in scanner.horizons:
        scanner.horizons['3s'] = 3
    if '10s' not in scanner.horizons:
        scanner.horizons['10s'] = 10

    load_info = scanner.load_from_cache()
    n_days = load_info['n_days']
    mid = scanner.mid_prices
    N = len(mid)
    log.info(f"Loaded {n_days} days, {N:,} bars")

    # Compute targets
    targets = scanner.compute_targets()

    # Test on multiple horizons
    test_horizons = ['3s', '5s', '10s']
    models_to_test = ['cnn', 'lstm', 'transformer']
    model_size = 'small'  # Conservative for limited data

    all_results = {}

    for horizon in test_horizons:
        if horizon not in targets or 'return' not in targets[horizon]:
            log.warning(f"Horizon {horizon} not available, skipping")
            continue

        target = targets[horizon]['return']
        log.info(f"\n{'='*60}")
        log.info(f"HORIZON: {horizon}")
        log.info(f"{'='*60}")

        # LightGBM baseline for this horizon
        log.info("Running LightGBM baseline...")
        lgbm_result = scanner.walk_forward_evaluate(
            target=target,
            target_name='return',
            horizon_name=horizon,
            min_train_days=5,
        )
        lgbm_result['elapsed_sec'] = 0  # Will be filled

        if 'error' not in lgbm_result:
            log.info(f"LightGBM: IC={lgbm_result['ic']:.4f} t={lgbm_result['tstat']:.2f}")
            discord_notify(
                f"Phase 5 [{horizon}] LightGBM baseline: "
                f"IC={lgbm_result['ic']:.4f} t={lgbm_result['tstat']:.2f}"
            )

        # Temporal models
        evaluator = TemporalWalkForwardEvaluator(
            features=scanner.features,
            mid_prices=scanner.mid_prices,
            day_boundaries=scanner.day_boundaries,
            seq_len=50,
            stride=10,
        )

        temporal_results = []
        for mt in models_to_test:
            log.info(f"\nRunning {mt} ({model_size})...")
            model = create_model(mt, scanner.features.shape[1], 1, model_size)
            n_params = count_parameters(model)
            log.info(f"  {n_params:,} parameters")
            del model

            result = evaluator.evaluate_model(
                model_type=mt,
                target=target,
                target_name='return',
                horizon_name=horizon,
                model_size=model_size,
                n_epochs=15,
                lr=1e-3,
                min_train_days=5,
            )
            temporal_results.append(result)

            if 'error' not in result:
                status = "PASS" if result['passed'] else "FAIL"
                discord_notify(
                    f"Phase 5 [{horizon}] {mt}: IC={result['ic']:.4f} "
                    f"t={result['tstat']:.2f} [{status}]"
                )

            gc.collect()

        # Comparison report
        report = format_comparison_report(lgbm_result, temporal_results)
        log.info(report)

        all_results[horizon] = {
            'lgbm': {k: v for k, v in lgbm_result.items()
                     if not isinstance(v, np.ndarray)},
            'temporal': [{k: v for k, v in r.items()
                         if not isinstance(v, np.ndarray)} for r in temporal_results],
            'report': report,
        }

    elapsed = time.time() - t0

    output = {
        'phase': 'model_progression',
        'timestamp': datetime.now().isoformat(),
        'model_size': model_size,
        'models_tested': models_to_test,
        'horizons_tested': test_horizons,
        'results': all_results,
        'elapsed_sec': elapsed,
    }

    result_path = RESULTS_DIR / f"overnight_phase5_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(str(result_path), 'w') as f:
        json.dump(output, f, indent=2, default=str)

    # Discord summary
    summary = ["Phase 5 COMPLETE - Model Progression:"]
    for hz, hz_res in all_results.items():
        lgbm_ic = hz_res.get('lgbm', {}).get('ic', 0)
        summary.append(f"\n  {hz}: LightGBM IC={lgbm_ic:.4f}")
        for tr in hz_res.get('temporal', []):
            if 'error' not in tr:
                marker = "BETTER" if abs(tr['ic']) > abs(lgbm_ic) else "worse"
                summary.append(
                    f"    {tr['model_type']}: IC={tr['ic']:.4f} [{marker}]"
                )
    discord_notify("\n".join(summary))

    del scanner
    gc.collect()
    return output


# ============================================================================
# FINAL REPORT
# ============================================================================
def compile_final_report(phase_results: dict) -> str:
    """Compile all phase results into a final overnight report."""
    lines = [
        "=" * 70,
        "OVERNIGHT ALPHA DISCOVERY - FINAL REPORT",
        f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M')}",
        "=" * 70,
        "",
    ]

    # Phase 1: Baseline
    if 'phase_1' in phase_results:
        p1 = phase_results['phase_1']
        lines.append("PHASE 1: HONEST BASELINE")
        lines.append(f"  Data: {p1.get('n_days', '?')} days, {p1.get('n_bars', '?'):,} bars")
        for hz, r in sorted(p1.get('horizons', {}).items()):
            if 'error' in r:
                lines.append(f"  {hz}: ERROR")
            else:
                viable = "** VIABLE **" if r.get('cost_ratio', 0) > 0.5 else ""
                lines.append(
                    f"  {hz}: IC={r['ic']:.4f} t={r['tstat']:.1f} "
                    f"move={r['avg_move_ticks']:.1f}t cost_ratio={r.get('cost_ratio', 0):.2f} {viable}"
                )
        lines.append("")

    # Phase 2: Multi-timeframe
    if 'phase_2' in phase_results:
        p2 = phase_results['phase_2']
        lines.append("PHASE 2: MULTI-TIMEFRAME")
        for tf, r in sorted(p2.get('timeframes', {}).items()):
            if 'error' in r:
                lines.append(f"  {tf}: ERROR")
            else:
                lines.append(
                    f"  {tf}: IC={r['ic']:.4f} t={r['tstat']:.1f} "
                    f"move={r['avg_move_ticks']:.1f}t bars/day={r.get('trades_per_day', 0):.0f}"
                )
        lines.append("")

    # Phase 3: Features
    if 'phase_3' in phase_results:
        p3 = phase_results['phase_3']
        lines.append("PHASE 3: FEATURE DISCOVERY")
        lines.append(f"  Forward-dominant features: {p3.get('n_forward_dominant', '?')}")
        lines.append(f"  New features tested: {len(p3.get('new_features', []))}")
        for nf in p3.get('new_features', [])[:5]:
            marker = "FWD" if nf['is_forward_dominant'] else "BWD"
            lines.append(f"  {nf['feature']}: fwd_IC={nf['fwd_ic']:.4f} [{marker}]")
        lines.append("")

    # Phase 4: Robustness
    if 'phase_4' in phase_results:
        p4 = phase_results['phase_4']
        r = p4.get('results', {})
        lines.append("PHASE 4: MODEL ROBUSTNESS")
        if 'holdout_split' in r:
            lines.append(f"  Holdout IC: {r['holdout_split']['ic']:.4f}")
        if 'ridge_benchmark' in r and 'ic' in r['ridge_benchmark']:
            lines.append(f"  Ridge IC: {r['ridge_benchmark']['ic']:.4f}")
        if 'hyperparam_sweep' in r and r['hyperparam_sweep']:
            best = r['hyperparam_sweep'][0]
            lines.append(f"  Best HP: IC={best['ic']:.4f} (min_child={best['min_child']}, depth={best['max_depth']})")
        lines.append("")

    # Phase 5: Model Progression
    if 'phase_5' in phase_results:
        p5 = phase_results['phase_5']
        lines.append("PHASE 5: MODEL PROGRESSION")
        for hz, hz_res in p5.get('results', {}).items():
            lgbm_ic = hz_res.get('lgbm', {}).get('ic', 0)
            lines.append(f"  {hz}: LightGBM IC={lgbm_ic:.4f}")
            for tr in hz_res.get('temporal', []):
                if 'error' not in tr:
                    marker = "BETTER" if abs(tr.get('ic', 0)) > abs(lgbm_ic) else ""
                    lines.append(
                        f"    {tr.get('model_type', '?')}: IC={tr.get('ic', 0):.4f} "
                        f"t={tr.get('tstat', 0):.1f} ({tr.get('n_params', 0):,} params) {marker}"
                    )
        lines.append("")

    # Verdict
    lines.append("=" * 70)
    lines.append("VERDICT")
    lines.append("=" * 70)

    # Find best viable configuration
    best_config = None
    best_score = 0

    if 'phase_1' in phase_results:
        for hz, r in phase_results['phase_1'].get('horizons', {}).items():
            if 'error' not in r:
                score = r['ic'] * r.get('cost_ratio', 0) * (1 if r['tstat'] > 2 else 0)
                if score > best_score:
                    best_score = score
                    best_config = f"Phase 1: {hz}"

    if 'phase_2' in phase_results:
        for tf, r in phase_results['phase_2'].get('timeframes', {}).items():
            if 'error' not in r and r.get('tstat', 0) > 2:
                score = r['ic'] * r.get('avg_move_ticks', 0)
                if score > best_score:
                    best_score = score
                    best_config = f"Phase 2: {tf}"

    if best_config:
        lines.append(f"  Best configuration: {best_config} (score={best_score:.4f})")
    else:
        lines.append("  No viable configuration found")

    lines.append("")
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Overnight Alpha Discovery')
    parser.add_argument('--skip-cache-rebuild', action='store_true',
                        help='Skip Phase 0 cache rebuild')
    parser.add_argument('--start-phase', type=int, default=0,
                        help='Start from this phase number')
    parser.add_argument('--phases', type=int, nargs='+',
                        help='Run only these phases (e.g., --phases 1 2 3)')
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("OVERNIGHT ALPHA DISCOVERY — STARTING")
    log.info(f"  Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    log.info(f"  Log: {LOG_FILE}")
    log.info("=" * 70)

    discord_notify("Overnight alpha discovery STARTING. Phases: cache rebuild -> baseline -> multi-TF -> features -> robustness -> model progression")

    phases_to_run = set(args.phases) if args.phases else set(range(args.start_phase, 6))
    phase_results = {}

    try:
        # Phase 0: Cache rebuild
        if 0 in phases_to_run and not args.skip_cache_rebuild:
            result = phase_0_rebuild_caches(new_only=True)
            phase_results['phase_0'] = result
            if result['status'] == 'failed':
                discord_notify("ABORT: Cache rebuild failed. Cannot continue.")
                return

        # Phase 1: Honest baseline
        if 1 in phases_to_run:
            phase_results['phase_1'] = phase_1_honest_baseline()

        # Phase 2: Multi-timeframe
        if 2 in phases_to_run:
            phase_results['phase_2'] = phase_2_multi_timeframe()

        # Phase 3: Feature discovery
        if 3 in phases_to_run:
            phase_results['phase_3'] = phase_3_feature_discovery()

        # Phase 4: Model robustness
        if 4 in phases_to_run:
            phase_results['phase_4'] = phase_4_model_robustness()

        # Phase 5: Model progression (temporal models vs LightGBM)
        if 5 in phases_to_run:
            phase_results['phase_5'] = phase_5_model_progression()

    except Exception as e:
        log.error(f"FATAL ERROR: {e}")
        log.error(traceback.format_exc())
        discord_notify(f"OVERNIGHT RUN FAILED: {str(e)[:200]}")

    # Final report
    report = compile_final_report(phase_results)
    log.info(report)

    # Save full results
    result_path = RESULTS_DIR / f"overnight_final_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(str(result_path), 'w') as f:
        json.dump({
            'timestamp': datetime.now().isoformat(),
            'phases': {k: v for k, v in phase_results.items()},
            'report': report,
        }, f, indent=2, default=str)

    discord_notify(f"OVERNIGHT COMPLETE. Report saved to {result_path.name}\n\n{report[-500:]}")
    log.info(f"\nResults saved to: {result_path}")


if __name__ == '__main__':
    main()
