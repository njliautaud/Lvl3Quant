"""
Ultra-Sparse Direction Trading — Extreme Threshold Entry

Hypothesis: if we only trade when the LightGBM model is at maximum confidence
(95th-99th percentile of signal strength), gross profit per trade may exceed
the $12.50 round-trip cost.

Tests:
- All features vs flow-only (OFI / net_flow / trade_imb / cancel_trade)
- Entry thresholds: 90, 95, 97, 98, 99 percentile
- Conditional IC at each threshold
- Signal decay analysis (how quickly does the threshold signal degrade?)
- Best target from prior work: ret_3s

Usage:
    python alpha_discovery/run_ultra_sparse.py
    python alpha_discovery/run_ultra_sparse.py --target ret_3s
    python alpha_discovery/run_ultra_sparse.py --thresholds 95 97 99
"""

import sys
import gc
import json
import time
import logging
import argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr, ttest_1samp
from typing import Dict, List, Optional, Tuple

# Setup path
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.mbo_features import get_feature_names, TOTAL_FEATURES
from alpha_discovery.run_return_multihorizon import (
    load_feature_cache, compute_return_targets,
    EXCLUDE_FEATURES_DIRECTION,
)
from alpha_discovery.run_model_refinement import walk_forward_evaluate

# ============================================================================
# LOGGING
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(RESULTS_DIR / 'ultra_sparse.log', mode='a', encoding='utf-8'),
    ]
)
logger = logging.getLogger("ultra_sparse")

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE = 0.25
TICK_VALUE = 12.50          # $12.50 per tick per contract
ES_POINT_VALUE = 50.0       # $50 per point for ES
ROUND_TRIP_COST = 12.50     # 1 tick round trip (conservative)
BARS_PER_SEC = 10           # 100ms intervals

# The 10 OFI/flow features that were forward-dominant in prior analysis
FLOW_ONLY_FEATURES = [
    'ofi_5', 'ofi_20', 'ofi_50',
    'net_flow_5', 'net_flow_20', 'net_flow_50',
    'trade_imb_5', 'trade_imb_20', 'trade_imb_50',
    'cancel_trade_5',
]


# ============================================================================
# WALK-FORWARD ENGINE WITH FULL PREDICTIONS RETURNED
# ============================================================================

def walk_forward_get_predictions(
    features: np.ndarray,
    target: np.ndarray,
    day_boundaries: list,
    feature_names: List[str],
    min_train_days: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Walk-forward LightGBM evaluation.

    Returns (predictions, actuals, bar_indices) — all aligned to original bar indices.
    Only bars in test folds appear in the arrays (train bars excluded).

    STRICTLY CAUSAL: train on days 0..D-2, test on day D (1-day purge gap).
    """
    import lightgbm as lgb

    n_days = len(day_boundaries) - 1
    if n_days < min_train_days + 1:
        logger.warning(f"Not enough days: {n_days} < {min_train_days + 1}")
        return np.array([]), np.array([]), np.array([], dtype=int)

    params = {
        'n_estimators': 500,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': 100,
        'verbose': -1,
        'n_jobs': -1,
        'objective': 'regression',
        'metric': 'rmse',
    }

    all_preds = []
    all_actuals = []
    all_bar_idx = []

    for test_day in range(min_train_days, n_days):
        train_end_day = test_day - 1
        train_start = day_boundaries[0]
        train_end = day_boundaries[train_end_day + 1]
        test_start = day_boundaries[test_day]
        test_end = day_boundaries[test_day + 1]

        X_train = features[train_start:train_end]
        y_train = target[train_start:train_end]
        X_test = features[test_start:test_end]
        y_test = target[test_start:test_end]

        train_valid = np.isfinite(y_train)
        test_valid = np.isfinite(y_test)

        if train_valid.sum() < 500 or test_valid.sum() < 50:
            continue

        X_tr = X_train[train_valid]
        y_tr = y_train[train_valid]
        X_te = X_test[test_valid]
        y_te = y_test[test_valid]

        split = int(len(X_tr) * 0.8)
        try:
            model = lgb.LGBMRegressor(**params)
            model.fit(
                X_tr[:split], y_tr[:split],
                eval_set=[(X_tr[split:], y_tr[split:])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
        except Exception as e:
            logger.warning(f"  Day {test_day} training failed: {e}")
            continue

        preds = model.predict(X_te)

        # Recover original bar indices in global array
        test_bar_indices = np.where(test_valid)[0] + test_start

        all_preds.append(preds)
        all_actuals.append(y_te)
        all_bar_idx.append(test_bar_indices)

        del model
        gc.collect()

    if not all_preds:
        return np.array([]), np.array([]), np.array([], dtype=int)

    return (
        np.concatenate(all_preds),
        np.concatenate(all_actuals),
        np.concatenate(all_bar_idx),
    )


# ============================================================================
# ULTRA-SPARSE SIMULATION
# ============================================================================

def simulate_ultra_sparse(
    predictions: np.ndarray,
    actuals: np.ndarray,
    bar_indices: np.ndarray,
    mid_prices: np.ndarray,
    day_boundaries: list,
    threshold_pct: float,
    min_hold_bars: int = 30,
    max_hold_bars: int = 300,
    round_trip_cost: float = ROUND_TRIP_COST,
    contracts: int = 1,
) -> dict:
    """
    Simulate trading with an extreme percentile entry threshold.

    Only enters when |prediction| exceeds the given percentile.
    Holds for [min_hold_bars, max_hold_bars] bars.

    Returns per-trade and aggregate statistics.
    """
    N_pred = len(predictions)
    if N_pred < 100:
        return {'error': 'Too few predictions'}

    # Compute threshold from signal magnitude distribution
    abs_pred = np.abs(predictions)
    threshold_val = np.percentile(abs_pred, threshold_pct)
    if threshold_val <= 0:
        return {'error': f'Zero threshold at {threshold_pct}th pct'}

    # Signal direction: +1 or -1
    signal = np.sign(predictions)

    # Entry mask: only where |pred| > threshold
    entry_mask = abs_pred > threshold_val

    n_signals = entry_mask.sum()
    if n_signals == 0:
        return {
            'threshold_pct': threshold_pct,
            'threshold_val': float(threshold_val),
            'n_signals': 0,
            'n_trades': 0,
            'gross_per_trade': 0.0,
            'net_per_trade': 0.0,
            'note': 'No signals at this threshold',
        }

    # Build a set of day-end bar indices for position force-close
    day_ends = set()
    for d in range(len(day_boundaries) - 1):
        day_end = day_boundaries[d + 1] - 1
        day_ends.add(day_end)

    # Trade simulation — vectorized approach:
    # For each signal bar, compute PnL from entry to exit.
    # Entry at close of signal bar (mid_price[bar_idx]).
    # Exit at min_hold or when signal flips, up to max_hold.

    trades = []

    # Walk through prediction bars in order
    # We need to track in-position state to avoid overlapping trades
    in_position = False
    pos_dir = 0
    pos_entry_bar = -1
    pos_entry_price = 0.0
    pos_hold_count = 0

    for i in range(N_pred):
        global_bar = bar_indices[i]

        if in_position:
            pos_hold_count += 1
            current_price = mid_prices[global_bar]

            # Exit conditions
            signal_reversed = (signal[i] != pos_dir) and entry_mask[i]
            max_hold_reached = pos_hold_count >= max_hold_bars
            day_end_reached = global_bar in day_ends

            if pos_hold_count >= min_hold_bars and (signal_reversed or max_hold_reached or day_end_reached):
                # Exit trade
                exit_price = current_price
                raw_pnl_points = pos_dir * (exit_price - pos_entry_price)
                raw_pnl_dollars = raw_pnl_points * ES_POINT_VALUE * contracts
                net_pnl_dollars = raw_pnl_dollars - round_trip_cost * contracts

                trades.append({
                    'entry_bar': pos_entry_bar,
                    'exit_bar': int(global_bar),
                    'hold_bars': pos_hold_count,
                    'direction': int(pos_dir),
                    'entry_price': float(pos_entry_price),
                    'exit_price': float(exit_price),
                    'gross_pnl': float(raw_pnl_dollars),
                    'net_pnl': float(net_pnl_dollars),
                    'exit_reason': 'reversal' if signal_reversed else ('day_end' if day_end_reached else 'max_hold'),
                })
                in_position = False
                pos_dir = 0

        if not in_position and entry_mask[i]:
            # Check not at day end
            if global_bar not in day_ends:
                in_position = True
                pos_dir = int(signal[i])
                pos_entry_bar = int(global_bar)
                pos_entry_price = float(mid_prices[global_bar])
                pos_hold_count = 0

    # Force close any open position at end
    if in_position and trades and len(trades) > 0:
        pass  # already handled by day_end_reached logic above

    if not trades:
        return {
            'threshold_pct': threshold_pct,
            'threshold_val': float(threshold_val),
            'n_signals': int(n_signals),
            'n_trades': 0,
            'note': 'No completed trades',
        }

    gross_pnls = np.array([t['gross_pnl'] for t in trades])
    net_pnls = np.array([t['net_pnl'] for t in trades])
    hold_bars = np.array([t['hold_bars'] for t in trades])

    n_trades = len(trades)
    n_days = len(day_boundaries) - 1

    gross_per_trade = float(np.mean(gross_pnls))
    net_per_trade = float(np.mean(net_pnls))
    total_net = float(np.sum(net_pnls))

    win_rate = float((net_pnls > 0).mean())
    daily_net = total_net / n_days if n_days > 0 else 0.0
    trades_per_day = n_trades / n_days if n_days > 0 else 0.0

    # Sharpe from daily PnL: assign trades to days
    daily_pnl_map: Dict[int, float] = {}
    for t in trades:
        # Find which day the entry is in
        exit_bar = t['exit_bar']
        day_idx = 0
        for d in range(len(day_boundaries) - 1):
            if day_boundaries[d] <= exit_bar < day_boundaries[d + 1]:
                day_idx = d
                break
        daily_pnl_map[day_idx] = daily_pnl_map.get(day_idx, 0.0) + t['net_pnl']

    daily_pnl_arr = np.array(list(daily_pnl_map.values()))
    sharpe = 0.0
    if len(daily_pnl_arr) > 2 and np.std(daily_pnl_arr) > 0:
        sharpe = float(np.mean(daily_pnl_arr) / np.std(daily_pnl_arr) * np.sqrt(252))

    # Conditional IC: IC among only threshold-crossing bars
    entry_idx = np.where(entry_mask)[0]
    if len(entry_idx) >= 50:
        try:
            cond_ic = float(spearmanr(predictions[entry_idx], actuals[entry_idx])[0])
        except Exception:
            cond_ic = float('nan')
    else:
        cond_ic = float('nan')

    # Signal decay: conditional return prediction accuracy at various holds
    # For each entered trade, compute actual return at hold=N bars
    decay_stats = {}
    for hold_n in [10, 30, 50, 100, 200, 300]:
        hold_pnls = []
        for t in trades:
            entry_bar = t['entry_bar']
            exit_idx_global = entry_bar + hold_n
            if exit_idx_global < len(mid_prices):
                actual_move = t['direction'] * (mid_prices[exit_idx_global] - mid_prices[entry_bar])
                hold_pnls.append(float(actual_move * ES_POINT_VALUE))
        if hold_pnls:
            decay_stats[f'hold_{hold_n}bars'] = {
                'mean_gross_pnl': float(np.mean(hold_pnls)),
                'win_rate': float((np.array(hold_pnls) > 0).mean()),
                'n': len(hold_pnls),
            }

    # Key verdict
    profitable_gross = gross_per_trade > 0
    profitable_net = net_per_trade > 0
    exceeds_cost = gross_per_trade > round_trip_cost

    return {
        'threshold_pct': threshold_pct,
        'threshold_val': float(threshold_val),
        'n_signals': int(n_signals),
        'signals_pct_of_bars': float(n_signals / N_pred),
        'n_trades': n_trades,
        'trades_per_day': round(trades_per_day, 2),
        'gross_per_trade': round(gross_per_trade, 2),
        'cost_per_trade': round(round_trip_cost, 2),
        'net_per_trade': round(net_per_trade, 2),
        'total_net_pnl': round(total_net, 2),
        'daily_net_pnl': round(daily_net, 2),
        'win_rate': round(win_rate, 4),
        'sharpe_annualized': round(sharpe, 2),
        'conditional_ic': round(cond_ic, 5) if np.isfinite(cond_ic) else None,
        'mean_hold_bars': round(float(np.mean(hold_bars)), 1),
        'mean_hold_sec': round(float(np.mean(hold_bars)) / BARS_PER_SEC, 1),
        'profitable_gross': profitable_gross,
        'profitable_net': profitable_net,
        'exceeds_cost': exceeds_cost,
        'signal_decay': decay_stats,
    }


# ============================================================================
# FEATURE SET PREPARATION
# ============================================================================

def build_feature_sets(scanner: MBOAlphaScanner) -> Dict[str, Tuple[np.ndarray, List[str]]]:
    """
    Build two feature sets:
    1. 'all_features' — all features minus direction-excluded vol proxies/time
    2. 'flow_only' — only the 10 OFI/flow features
    """
    all_names = scanner.feature_names

    # ALL features (exclude direction-contaminated per prior work)
    keep_all = np.array([fn not in EXCLUDE_FEATURES_DIRECTION for fn in all_names])
    feat_all = scanner.features[:, keep_all]
    names_all = [fn for fn in all_names if fn not in EXCLUDE_FEATURES_DIRECTION]

    # FLOW-ONLY: subset of flow features
    keep_flow = np.array([fn in FLOW_ONLY_FEATURES for fn in names_all])
    feat_flow = feat_all[:, keep_flow]
    names_flow = [fn for fn in names_all if fn in FLOW_ONLY_FEATURES]

    logger.info(f"Feature sets:")
    logger.info(f"  all_features: {feat_all.shape[1]} features")
    logger.info(f"  flow_only:    {feat_flow.shape[1]} features: {names_flow}")

    return {
        'all_features': (feat_all, names_all),
        'flow_only': (feat_flow, names_flow),
    }


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Ultra-sparse direction trading experiment')
    parser.add_argument('--target', default='ret_3s',
                        help='Return target to use (default: ret_3s)')
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=[90.0, 95.0, 97.0, 98.0, 99.0],
                        help='Entry threshold percentiles')
    parser.add_argument('--min-hold', type=int, default=30,
                        help='Minimum hold bars (default 30 = 3s)')
    parser.add_argument('--max-hold', type=int, default=300,
                        help='Maximum hold bars (default 300 = 30s)')
    parser.add_argument('--min-train-days', type=int, default=3,
                        help='Minimum training days before first test')
    args = parser.parse_args()

    logger.info("=" * 75)
    logger.info("ULTRA-SPARSE DIRECTION TRADING EXPERIMENT")
    logger.info(f"  Target:     {args.target}")
    logger.info(f"  Thresholds: {args.thresholds}")
    logger.info(f"  Hold range: {args.min_hold}-{args.max_hold} bars "
                f"({args.min_hold/BARS_PER_SEC:.0f}s - {args.max_hold/BARS_PER_SEC:.0f}s)")
    logger.info("=" * 75)

    t_start = time.time()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    stats = load_feature_cache(scanner)
    if stats is None:
        logger.info("No feature cache found, computing from scratch (slow)...")
        stats = scanner.load_from_cache()

    logger.info(f"Data: {len(scanner.mid_prices):,} snapshots, "
                f"{len(scanner.day_boundaries) - 1} days, "
                f"{len(scanner.feature_names)} features")

    # -----------------------------------------------------------------------
    # Compute return targets
    # -----------------------------------------------------------------------
    logger.info("\nComputing return targets...")
    # Parse target name to horizon
    hz_map = {
        'ret_1s': {'1s': 1},
        'ret_3s': {'3s': 3},
        'ret_5s': {'5s': 5},
        'ret_10s': {'10s': 10},
        'ret_15s': {'15s': 15},
        'ret_30s': {'30s': 30},
        'ret_60s': {'60s': 60},
    }
    hz_sec = hz_map.get(args.target, {'3s': 3})

    all_targets = compute_return_targets(
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        sample_interval_ms=100,
        horizons_sec=hz_sec,
        include_flow_target=False,
    )

    if args.target not in all_targets:
        logger.error(f"Target {args.target} not found. Available: {list(all_targets.keys())}")
        sys.exit(1)

    target_arr = all_targets[args.target]
    logger.info(f"Target {args.target}: {np.isfinite(target_arr).sum():,} valid bars")

    # -----------------------------------------------------------------------
    # Build feature sets
    # -----------------------------------------------------------------------
    feature_sets = build_feature_sets(scanner)

    # -----------------------------------------------------------------------
    # Walk-forward predictions for each feature set
    # -----------------------------------------------------------------------
    predictions_cache = {}

    for feat_set_name, (feat_matrix, feat_names) in feature_sets.items():
        logger.info(f"\n{'='*60}")
        logger.info(f"Walk-forward evaluation: {feat_set_name} ({feat_matrix.shape[1]} features)")
        logger.info(f"{'='*60}")

        t0 = time.time()
        preds, actuals, bar_idx = walk_forward_get_predictions(
            features=feat_matrix,
            target=target_arr,
            day_boundaries=scanner.day_boundaries,
            feature_names=feat_names,
            min_train_days=args.min_train_days,
        )
        elapsed = time.time() - t0

        if len(preds) == 0:
            logger.warning(f"  No predictions for {feat_set_name}")
            continue

        # Overall IC
        valid = np.isfinite(preds) & np.isfinite(actuals)
        p, a = preds[valid], actuals[valid]
        ic = float(spearmanr(p, a)[0]) if len(p) > 50 else float('nan')
        logger.info(f"  Overall IC={ic:.5f}, n_predictions={len(p):,} ({elapsed:.0f}s)")

        predictions_cache[feat_set_name] = {
            'predictions': preds,
            'actuals': actuals,
            'bar_indices': bar_idx,
            'overall_ic': ic,
        }

        gc.collect()

    # -----------------------------------------------------------------------
    # Ultra-sparse threshold simulation
    # -----------------------------------------------------------------------
    all_results = {}
    last_log_time = time.time()

    for feat_set_name, pred_data in predictions_cache.items():
        preds = pred_data['predictions']
        actuals = pred_data['actuals']
        bar_idx = pred_data['bar_indices']
        overall_ic = pred_data['overall_ic']

        threshold_results = []
        logger.info(f"\n{'='*60}")
        logger.info(f"Threshold simulation: {feat_set_name} (overall IC={overall_ic:.5f})")
        logger.info(f"{'='*60}")
        logger.info(
            f"{'Threshold':>10s} {'N_trades':>10s} {'Trades/d':>10s} "
            f"{'Gross/T':>10s} {'Net/T':>10s} {'WinR':>7s} "
            f"{'CondIC':>8s} {'Sharpe':>7s} {'BreakEven':>10s}"
        )
        logger.info("-" * 85)

        for thr in args.thresholds:
            sim = simulate_ultra_sparse(
                predictions=preds,
                actuals=actuals,
                bar_indices=bar_idx,
                mid_prices=scanner.mid_prices,
                day_boundaries=scanner.day_boundaries,
                threshold_pct=thr,
                min_hold_bars=args.min_hold,
                max_hold_bars=args.max_hold,
            )
            threshold_results.append(sim)

            if 'error' in sim:
                logger.info(f"  {thr:>10.0f}%: ERROR — {sim['error']}")
                continue

            cond_ic_str = f"{sim['conditional_ic']:.5f}" if sim['conditional_ic'] is not None else "  N/A  "
            exceeds = "YES" if sim['exceeds_cost'] else "no"
            logger.info(
                f"  {thr:>9.0f}% {sim['n_trades']:>10d} {sim['trades_per_day']:>10.1f} "
                f"${sim['gross_per_trade']:>8.2f} ${sim['net_per_trade']:>8.2f} "
                f"{sim['win_rate']:>6.1%} {cond_ic_str:>8s} "
                f"{sim['sharpe_annualized']:>7.2f} {exceeds:>10s}"
            )

            # Periodic log
            now = time.time()
            if now - last_log_time > 30:
                logger.info(f"  [Progress] {feat_set_name} threshold {thr}% done "
                            f"({(time.time() - t_start)/60:.1f} min elapsed)")
                last_log_time = now

        all_results[feat_set_name] = {
            'overall_ic': overall_ic,
            'threshold_results': threshold_results,
        }

    # -----------------------------------------------------------------------
    # Signal decay analysis summary
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 75)
    logger.info("SIGNAL DECAY ANALYSIS")
    logger.info("=" * 75)
    logger.info("(Average gross PnL at exact hold times for 98th pct threshold entries)")

    for feat_set_name, result in all_results.items():
        logger.info(f"\n  Feature set: {feat_set_name}")
        # Find the 98th percentile result
        thr_result = None
        for tr in result['threshold_results']:
            if 'threshold_pct' in tr and abs(tr['threshold_pct'] - 98.0) < 0.5:
                thr_result = tr
                break
        if thr_result and 'signal_decay' in thr_result:
            for hold_key, decay in thr_result['signal_decay'].items():
                logger.info(
                    f"    {hold_key}: mean_gross=${decay['mean_gross_pnl']:>7.2f} "
                    f"win_rate={decay['win_rate']:.1%} n={decay['n']}"
                )

    # -----------------------------------------------------------------------
    # Key findings summary
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 75)
    logger.info("KEY FINDINGS — ULTRA SPARSE DIRECTION TRADING")
    logger.info("=" * 75)
    logger.info(f"TARGET: {args.target} | Round-trip cost: ${ROUND_TRIP_COST:.2f}")
    logger.info("")

    for feat_set_name, result in all_results.items():
        logger.info(f"Feature set: {feat_set_name} (overall IC={result['overall_ic']:.5f})")
        best_net = max(
            (tr for tr in result['threshold_results'] if 'net_per_trade' in tr),
            key=lambda x: x.get('net_per_trade', float('-inf')),
            default=None,
        )
        if best_net:
            logger.info(
                f"  Best threshold: {best_net['threshold_pct']:.0f}th pct "
                f"-> gross=${best_net['gross_per_trade']:.2f}/trade "
                f"net=${best_net['net_per_trade']:.2f}/trade "
                f"({best_net['trades_per_day']:.1f} trades/day)"
            )
        any_profitable = any(
            tr.get('profitable_net', False) for tr in result['threshold_results']
        )
        any_exceeds_cost = any(
            tr.get('exceeds_cost', False) for tr in result['threshold_results']
        )
        logger.info(f"  Net profitable at any threshold: {any_profitable}")
        logger.info(f"  Gross exceeds $12.50 cost at any threshold: {any_exceeds_cost}")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = RESULTS_DIR / f"ultra_sparse_{timestamp}.json"

    def make_serializable(obj):
        if isinstance(obj, dict):
            return {k: make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [make_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, float) and not np.isfinite(obj):
            return str(obj)
        return obj

    output = {
        'timestamp': timestamp,
        'target': args.target,
        'thresholds_tested': args.thresholds,
        'min_hold_bars': args.min_hold,
        'max_hold_bars': args.max_hold,
        'round_trip_cost': ROUND_TRIP_COST,
        'tick_value': TICK_VALUE,
        'n_days': len(scanner.day_boundaries) - 1,
        'n_snapshots': len(scanner.mid_prices),
        'results': make_serializable(all_results),
        'elapsed_sec': time.time() - t_start,
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {result_file}")
    logger.info(f"Total elapsed: {(time.time() - t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
