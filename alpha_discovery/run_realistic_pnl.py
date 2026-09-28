"""
Realistic PnL Simulator for ES Futures Direction Strategy (v2 — vectorized)

Fixes from v1:
1. Pre-compute signal arrays (was O(n^2) from median in loop)
2. Use actual mid prices for exact PnL (was using horizon return approximation)
3. Cache predictions per target (avoid retraining for each config)

Strategy:
1. Generate signal every 100ms from LightGBM walk-forward
2. ENTER only when |signal| > threshold percentile
3. HOLD for at least min_hold_time
4. EXIT when: signal reverses (after min hold), or max_hold reached
5. Count exactly 1 round-trip cost per trade
6. Compute daily PnL, Sharpe, drawdown, win rate

Usage:
    python alpha_discovery/run_realistic_pnl.py
"""

import sys
import gc
import json
import time
import logging
import numpy as np
from pathlib import Path
from datetime import datetime
from scipy.stats import spearmanr
from typing import Dict, List, Optional

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
        logging.FileHandler(RESULTS_DIR / 'realistic_pnl.log', mode='a'),
    ]
)
logger = logging.getLogger("realistic_pnl")

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE = 0.25
TICK_VALUE = 12.50
BARS_PER_SEC = 10  # 100ms bars
ES_POINT_VALUE = 50.0  # $50 per point for ES


def simulate_realistic_trading(
    predictions: np.ndarray,
    mid_prices: np.ndarray,   # actual mid prices for exact PnL
    day_boundaries: list,
    entry_threshold_pct: float = 80.0,
    min_hold_bars: int = 30,
    max_hold_bars: int = 100,
    exit_signal_reversal: bool = True,
    round_trip_ticks: float = 1.0,
    contracts: int = 1,
) -> dict:
    """
    Simulate realistic trading with proper entry/exit logic.

    Uses actual mid prices for exact PnL calculation.
    All signal arrays pre-computed (vectorized, no per-bar median).
    """
    n = len(predictions)
    if n < 1000:
        return {'error': 'Too few samples'}

    # ---- Vectorized pre-computation ----
    valid_mask = np.isfinite(predictions) & np.isfinite(mid_prices) & (mid_prices > 0)

    # Compute median ONCE
    pred_median = np.nanmedian(predictions[valid_mask])

    # Pre-compute signal arrays
    pred_centered = predictions - pred_median
    signal_strength = np.abs(pred_centered)
    signal_dir = np.sign(pred_centered).astype(np.int8)  # -1, 0, +1

    # Compute entry threshold
    threshold = np.percentile(signal_strength[valid_mask], entry_threshold_pct)

    # Pre-compute day assignments
    n_days = len(day_boundaries) - 1
    bar_to_day = np.zeros(n, dtype=np.int32)
    for d in range(n_days):
        bar_to_day[day_boundaries[d]:day_boundaries[d + 1]] = d

    rt_cost = round_trip_ticks * TICK_VALUE * contracts
    half_threshold = threshold * 0.5

    # ---- State machine loop (fast — no numpy calls inside) ----
    trades_entry = []
    trades_exit = []
    trades_dir = []
    trades_held = []
    trades_day = []

    position = 0  # -1, 0, +1
    entry_bar = 0
    bars_held = 0
    current_day = 0

    for i in range(n):
        if not valid_mask[i]:
            continue

        ss = signal_strength[i]
        sd = signal_dir[i]
        day = bar_to_day[i]

        # Force close at day boundary
        if position != 0 and day != current_day:
            trades_entry.append(entry_bar)
            trades_exit.append(i)
            trades_dir.append(position)
            trades_held.append(bars_held)
            trades_day.append(current_day)
            position = 0
            bars_held = 0
            current_day = day
            continue

        current_day = day

        if position == 0:
            if ss > threshold and sd != 0:
                position = int(sd)
                entry_bar = i
                bars_held = 0
        else:
            bars_held += 1
            should_exit = False

            if bars_held >= max_hold_bars:
                should_exit = True
            elif bars_held >= min_hold_bars:
                if exit_signal_reversal and sd != position:
                    should_exit = True
                elif ss < half_threshold:
                    should_exit = True

            if should_exit:
                trades_entry.append(entry_bar)
                trades_exit.append(i)
                trades_dir.append(position)
                trades_held.append(bars_held)
                trades_day.append(current_day)
                position = 0
                bars_held = 0

    if not trades_entry:
        return {'error': 'No trades generated'}

    # ---- Vectorized PnL computation using mid prices ----
    entries = np.array(trades_entry, dtype=np.int64)
    exits = np.array(trades_exit, dtype=np.int64)
    directions = np.array(trades_dir, dtype=np.int8)
    held_bars = np.array(trades_held, dtype=np.int32)
    trade_days = np.array(trades_day, dtype=np.int32)

    # Exact PnL: price change from entry to exit
    entry_prices = mid_prices[entries]
    exit_prices = mid_prices[exits]
    price_changes = exit_prices - entry_prices  # in points

    # PnL in dollars: direction * price_change_points * $50/point * contracts
    gross_pnls = directions * price_changes * ES_POINT_VALUE * contracts
    net_pnls = gross_pnls - rt_cost

    n_trades = len(net_pnls)

    # ---- Statistics (vectorized) ----
    total_gross = float(np.sum(gross_pnls))
    total_cost = float(n_trades * rt_cost)
    total_net = float(np.sum(net_pnls))

    win_mask = net_pnls > 0
    lose_mask = net_pnls <= 0
    win_rate = float(np.mean(win_mask))
    avg_win = float(np.mean(net_pnls[win_mask])) if np.any(win_mask) else 0
    avg_loss = float(np.mean(net_pnls[lose_mask])) if np.any(lose_mask) else 0
    sum_winners = float(np.sum(net_pnls[win_mask]))
    sum_losers = float(np.sum(net_pnls[lose_mask]))
    profit_factor = abs(sum_winners / sum_losers) if sum_losers != 0 else 0

    # Daily stats
    daily_pnl = {}
    for d, pnl in zip(trade_days, net_pnls):
        daily_pnl[d] = daily_pnl.get(d, 0.0) + pnl

    daily_values = list(daily_pnl.values())
    n_trading_days = len(daily_values)
    daily_mean = float(np.mean(daily_values)) if daily_values else 0
    daily_std = float(np.std(daily_values)) if len(daily_values) > 1 else 0
    daily_sharpe = daily_mean / daily_std * np.sqrt(252) if daily_std > 0 else 0
    win_days = sum(1 for d in daily_values if d > 0)
    pct_win_days = win_days / n_trading_days if n_trading_days > 0 else 0

    # Drawdown
    cum_pnl = np.cumsum(net_pnls)
    peak = np.maximum.accumulate(cum_pnl)
    max_dd = float(np.min(cum_pnl - peak))

    trades_per_day = n_trades / max(n_trading_days, 1)

    return {
        'n_trades': int(n_trades),
        'n_trading_days': int(n_trading_days),
        'trades_per_day': float(trades_per_day),
        'total_gross_pnl': total_gross,
        'total_cost': total_cost,
        'total_net_pnl': total_net,
        'daily_net_pnl': daily_mean,
        'daily_std': daily_std,
        'daily_sharpe': float(daily_sharpe),
        'pct_win_days': float(pct_win_days),
        'win_rate': win_rate,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': float(profit_factor),
        'avg_hold_bars': float(np.mean(held_bars)),
        'avg_hold_sec': float(np.mean(held_bars) / BARS_PER_SEC),
        'max_drawdown': float(max_dd),
        'entry_threshold_pct': entry_threshold_pct,
        'min_hold_bars': min_hold_bars,
        'max_hold_bars': max_hold_bars,
        'daily_pnl_values': [float(d) for d in daily_values],
    }


def main():
    t0 = time.time()

    logger.info("=" * 75)
    logger.info("REALISTIC PNL SIMULATION v2 (vectorized)")
    logger.info("=" * 75)

    # Load data
    scanner = MBOAlphaScanner()
    stats = load_feature_cache(scanner)
    if stats is None:
        logger.error("No feature cache.")
        sys.exit(1)

    n_bars = scanner.features.shape[0]
    n_days = len(scanner.day_boundaries) - 1
    logger.info(f"Data: {n_bars:,} snapshots, {n_days} days")

    # Compute targets
    targets = compute_return_targets(
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        horizons_sec={'1s': 1, '3s': 3, '5s': 5},
        include_flow_target=False,
    )

    # Prepare features
    keep_mask = np.array([fn not in EXCLUDE_FEATURES_DIRECTION for fn in scanner.feature_names])
    features_clean = scanner.features[:, keep_mask]
    names_clean = [fn for fn in scanner.feature_names if fn not in EXCLUDE_FEATURES_DIRECTION]

    all_results = {
        'timestamp': datetime.now().strftime('%Y%m%d_%H%M%S'),
    }

    # ============================================================
    # Cache predictions per target (train once, sim many)
    # ============================================================
    prediction_cache = {}
    needed_targets = {'ret_1s', 'ret_3s', 'ret_5s'}

    for tgt_name in needed_targets:
        if tgt_name not in targets:
            continue

        logger.info(f"\nTraining walk-forward model for {tgt_name}...")
        t1 = time.time()

        res = walk_forward_evaluate(
            features=features_clean,
            target=targets[tgt_name],
            day_boundaries=scanner.day_boundaries,
            feature_names=names_clean,
            model_type='lgbm',
            hour_of_day=scanner.hour_of_day,
        )

        elapsed_train = time.time() - t1

        if 'error' in res:
            logger.warning(f"  {tgt_name} training failed: {res['error']}")
            continue

        logger.info(f"  {tgt_name}: IC={res['ic']:.4f}, t={res['tstat']:.2f}, "
                     f"predictions={res['n_preds']:,}, time={elapsed_train:.0f}s")
        prediction_cache[tgt_name] = res

    # ============================================================
    # Test configurations (using cached predictions)
    # ============================================================
    configs = [
        # (target_name, entry_pct, min_hold, max_hold, label)
        ('ret_1s', 80, 10, 50, 'ret_1s_hold1s_entry80'),
        ('ret_1s', 90, 10, 50, 'ret_1s_hold1s_entry90'),
        ('ret_3s', 80, 30, 100, 'ret_3s_hold3s_entry80'),
        ('ret_3s', 90, 30, 100, 'ret_3s_hold3s_entry90'),
        ('ret_3s', 80, 30, 300, 'ret_3s_hold3-30s_entry80'),
        ('ret_3s', 90, 30, 300, 'ret_3s_hold3-30s_entry90'),
        ('ret_3s', 70, 30, 100, 'ret_3s_hold3s_entry70'),
        ('ret_3s', 80, 50, 150, 'ret_3s_hold5-15s_entry80'),
        ('ret_5s', 80, 50, 150, 'ret_5s_hold5-15s_entry80'),
    ]

    # Build full-length prediction arrays aligned with bar indices
    full_predictions = {}
    for tgt_name, res in prediction_cache.items():
        full_pred = np.full(n_bars, np.nan, dtype=np.float64)
        indices = res['pred_indices']
        preds = res['predictions']
        # Only fill bars that have predictions
        valid = np.isfinite(preds) & (indices < n_bars)
        full_pred[indices[valid]] = preds[valid]
        full_predictions[tgt_name] = full_pred
        n_filled = np.sum(np.isfinite(full_pred))
        logger.info(f"  {tgt_name}: {n_filled:,} / {n_bars:,} bars have predictions ({n_filled/n_bars:.1%})")

    logger.info(f"\n{'='*75}")
    logger.info("RUNNING SIMULATIONS")
    logger.info(f"{'='*75}")

    for target_name, entry_pct, min_hold, max_hold, label in configs:
        if target_name not in full_predictions:
            all_results[label] = {'error': f'No predictions for {target_name}'}
            continue

        logger.info(f"\n--- {label} ---")
        logger.info(f"  target={target_name}, entry_pct={entry_pct}, "
                     f"hold={min_hold}-{max_hold}bars ({min_hold/10:.1f}-{max_hold/10:.1f}s)")

        t_sim = time.time()
        pnl = simulate_realistic_trading(
            predictions=full_predictions[target_name],
            mid_prices=scanner.mid_prices,
            day_boundaries=scanner.day_boundaries,
            entry_threshold_pct=entry_pct,
            min_hold_bars=min_hold,
            max_hold_bars=max_hold,
        )
        sim_time = time.time() - t_sim

        if 'error' in pnl:
            logger.warning(f"  Simulation failed: {pnl['error']}")
            all_results[label] = pnl
            continue

        logger.info(f"  Trades: {pnl['n_trades']:,} ({pnl['trades_per_day']:.1f}/day)")
        logger.info(f"  Win rate: {pnl['win_rate']:.1%}")
        logger.info(f"  Avg hold: {pnl['avg_hold_sec']:.1f}s")
        logger.info(f"  Daily PnL: ${pnl['daily_net_pnl']:+,.2f} (std=${pnl['daily_std']:,.2f})")
        logger.info(f"  Annualized Sharpe: {pnl['daily_sharpe']:.2f}")
        logger.info(f"  Win days: {pnl['pct_win_days']:.0%} ({int(pnl['pct_win_days']*pnl['n_trading_days'])}/{pnl['n_trading_days']})")
        logger.info(f"  Profit factor: {pnl['profit_factor']:.2f}")
        logger.info(f"  Max drawdown: ${pnl['max_drawdown']:,.2f}")
        logger.info(f"  Gross: ${pnl['total_gross_pnl']:+,.2f}  Cost: ${pnl['total_cost']:,.2f}  Net: ${pnl['total_net_pnl']:+,.2f}")
        logger.info(f"  Sim time: {sim_time:.1f}s")
        if pnl['daily_pnl_values']:
            logger.info(f"  Daily PnLs: [{', '.join(f'${d:+,.0f}' for d in pnl['daily_pnl_values'])}]")

        all_results[label] = pnl

    # ============================================================
    # Single-feature leakage tests
    # ============================================================
    logger.info(f"\n{'='*75}")
    logger.info("SINGLE-FEATURE LEAKAGE TESTS (ret_3s)")
    logger.info(f"{'='*75}")

    target = targets.get('ret_3s')
    single_feature_results = {}

    if target is not None:
        top_features = [
            'depth_ratio_l1', 'ask_L1_orders', 'bid_L1_conc',
            'ask_L1_conc', 'bid_L1_orders', 'ofi_5',
            'total_bid_vol', 'bid_pressure', 'ret_100',
        ]

        for feat_name in top_features:
            if feat_name not in names_clean:
                logger.info(f"  {feat_name:>25s}: NOT FOUND in features")
                continue

            feat_idx = names_clean.index(feat_name)
            single_feat = features_clean[:, feat_idx:feat_idx + 1]

            res = walk_forward_evaluate(
                features=single_feat,
                target=target,
                day_boundaries=scanner.day_boundaries,
                feature_names=[feat_name],
                model_type='lgbm',
                hour_of_day=scanner.hour_of_day,
            )

            if 'error' not in res:
                logger.info(f"  {feat_name:>25s}: IC={res['ic']:.4f} t={res['tstat']:.2f} "
                           f"ICIR={res['icir']:.3f} folds={res['n_folds']}")
                single_feature_results[feat_name] = {
                    'ic': res['ic'], 'tstat': res['tstat'],
                    'icir': res['icir'], 'n_folds': res['n_folds'],
                    'fold_ics': res['fold_ics'],
                }
            else:
                logger.info(f"  {feat_name:>25s}: {res['error']}")

    all_results['single_feature_tests'] = single_feature_results

    # ============================================================
    # Feature correlation with target (leakage check)
    # ============================================================
    logger.info(f"\n{'='*75}")
    logger.info("FEATURE-TARGET CORRELATION CHECK")
    logger.info(f"{'='*75}")

    if target is not None:
        corr_results = {}
        valid = np.isfinite(target)
        for feat_name in names_clean[:30]:  # top 30 features
            feat_idx = names_clean.index(feat_name)
            feat_vals = features_clean[:, feat_idx]
            both_valid = valid & np.isfinite(feat_vals)
            if np.sum(both_valid) > 10000:
                rho, _ = spearmanr(feat_vals[both_valid], target[both_valid])
                if abs(rho) > 0.05:  # only report notable correlations
                    corr_results[feat_name] = float(rho)
                    logger.info(f"  {feat_name:>25s}: rho={rho:.4f} {'*** SUSPICIOUS' if abs(rho) > 0.15 else ''}")

        all_results['feature_target_correlations'] = corr_results

    # ============================================================
    # Summary
    # ============================================================
    elapsed = time.time() - t0
    all_results['elapsed_sec'] = elapsed

    logger.info(f"\n{'='*75}")
    logger.info("REALISTIC PNL SUMMARY")
    logger.info(f"{'='*75}")

    header = f"\n{'Label':<35s} {'Trades/d':>8s} {'WinRate':>8s} {'Daily$':>10s} {'Sharpe':>7s} {'WinDays':>8s} {'PF':>6s} {'MaxDD':>10s}"
    logger.info(header)
    logger.info("-" * 95)

    for label in [c[4] for c in configs]:
        if label in all_results and isinstance(all_results[label], dict) and 'error' not in all_results[label]:
            r = all_results[label]
            logger.info(
                f"{label:<35s} {r['trades_per_day']:>8.1f} {r['win_rate']:>7.1%} "
                f"${r['daily_net_pnl']:>+9,.0f} {r['daily_sharpe']:>7.2f} "
                f"{r['pct_win_days']:>7.0%} {r['profit_factor']:>6.2f} "
                f"${r['max_drawdown']:>9,.0f}"
            )

    if single_feature_results:
        logger.info(f"\nSingle-feature ICs (ret_3s):")
        for feat, res in sorted(single_feature_results.items(), key=lambda x: abs(x[1]['ic']), reverse=True):
            logger.info(f"  {feat:>25s}: IC={res['ic']:.4f} t={res['tstat']:.2f}")

    # Save
    out_path = RESULTS_DIR / f"realistic_pnl_{all_results['timestamp']}.json"
    with open(out_path, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    logger.info(f"\nResults saved to: {out_path}")
    logger.info(f"Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")


if __name__ == '__main__':
    main()
