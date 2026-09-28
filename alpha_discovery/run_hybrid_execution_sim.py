"""
Hybrid Execution Simulator -- ES Futures Limit Order Trading
=============================================================

CONTEXT: We have a direction prediction model (IC=0.1135, walk-forward LightGBM on
ret_3s) with limit order execution. Previous study showed:
  - Market exit only: $0.65-0.85/trade, Sharpe ~50
  - Limit exit only: $12-13/trade (unrealistically assumes 100% limit exit fills)
  - 100ms latency improves results (signal noise filtering)
  - Fill probability: 6.5% in 3s, 10.8% in 10s, 14.5% in 30s at best bid/ask
  - MFE: +2.8 ticks, MAE: -2.6 ticks after fill

WHAT THIS DOES: REALISTIC simulation combining limit entry + smart exit logic.

Key parameters:
  - TICK_SIZE = 0.25, TICK_VALUE = $12.50, ES_POINT_VALUE = 50
  - HALF_TICK = 0.125 (corrected bid/ask = mid +/- 0.125)
  - COMMISSION_RT = $3.00 (0.24 ticks round-trip, AMP+Rithmic+CME)
  - BARS_PER_SEC = 10 (100ms bars)

HYBRID EXIT LOGIC (the core innovation):
  For each fill, place a limit exit at the opposite side. Monitor bar by bar.
  EXIT CONDITIONS (whichever comes first):
    i.   LIMIT EXIT: mid crosses limit exit price -> earn both half-spreads
    ii.  TAKE PROFIT: price moves X ticks favorable -> market exit
    iii. STOP LOSS: price moves Y ticks adverse -> market exit
    iv.  TIMEOUT: hold_horizon reached -> market exit

PnL per scenario:
  Limit entry earns +0.5 ticks (half spread vs crossing)
  Limit exit earns +0.5 ticks, market exit costs -0.5 ticks
  Commission: -0.24 ticks ($3.00 RT)

Usage:
    python alpha_discovery/run_hybrid_execution_sim.py
    python alpha_discovery/run_hybrid_execution_sim.py --fast
"""
import sys, gc, json, time, logging, argparse
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.run_return_multihorizon import (
    load_feature_cache, compute_return_targets, EXCLUDE_FEATURES_DIRECTION,
)
from alpha_discovery.run_model_refinement import walk_forward_evaluate

# ============================================================================
# LOGGING (ASCII-only for Windows cp1252)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(RESULTS_DIR / 'hybrid_execution.log'),
                            mode='a', encoding='utf-8'),
    ]
)
log = logging.getLogger('hybrid_execution')

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE        = 0.25     # ES minimum price increment
TICK_VALUE       = 12.50    # Dollar value per tick
ES_POINT_VALUE   = 50       # Dollar value per point (4 ticks)
BARS_PER_SEC     = 10       # 100ms bars
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)     # Round-trip commission in dollars (AMP+Rithmic+CME fees)
HALF_TICK        = TICK_SIZE / 2   # 0.125
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24 ticks ($3.00 RT)

_last_log = [0.0]

def progress(msg, interval=30.0):
    if time.time() - _last_log[0] >= interval:
        log.info(msg)
        _last_log[0] = time.time()


# ============================================================================
# JSON SERIALIZATION HELPER
# ============================================================================
def to_safe(obj):
    """Convert numpy types and NaN/Inf to JSON-serializable form."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        v = float(obj)
        return None if (np.isnan(v) or np.isinf(v)) else v
    if isinstance(obj, np.ndarray):
        return [to_safe(x) for x in obj.tolist()]
    if isinstance(obj, dict):
        return {k: to_safe(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [to_safe(v) for v in obj]
    if isinstance(obj, float) and (np.isnan(obj) or np.isinf(obj)):
        return None
    return obj


# ============================================================================
# DIRECTION MODEL
# ============================================================================
def get_direction_model(scanner, target, min_train_days=2, fast=False):
    """Train walk-forward LightGBM direction model on ret_3s."""
    log.info("Training walk-forward direction model (ret_3s)...")
    feature_names = scanner.feature_names
    keep_mask = np.array([fn not in EXCLUDE_FEATURES_DIRECTION for fn in feature_names])
    features_use = scanner.features[:, keep_mask]
    feature_names_use = [fn for fn in feature_names if fn not in EXCLUDE_FEATURES_DIRECTION]

    n_est = 300 if fast else 500
    params = {
        'n_estimators': n_est, 'max_depth': 6,
        'learning_rate': 0.03, 'subsample': 0.8, 'colsample_bytree': 0.7,
        'reg_alpha': 0.1, 'reg_lambda': 1.0, 'min_child_samples': 100,
        'verbose': -1, 'n_jobs': -1,
    }

    result = walk_forward_evaluate(
        features=features_use, target=target,
        day_boundaries=scanner.day_boundaries,
        feature_names=feature_names_use,
        model_type='lgbm', min_train_days=min_train_days,
        hour_of_day=scanner.hour_of_day, lgbm_params=params,
    )
    if 'error' in result:
        log.error(f"Walk-forward failed: {result['error']}")
        return np.array([]), np.array([], dtype=np.int64), result

    log.info(f"  Direction model: IC={result['ic']:.4f}  ICIR={result['icir']:.2f}  "
             f"t-stat={result['tstat']:.2f}  n_preds={result.get('n_preds', 0):,}")
    return result['predictions'], result['pred_indices'], result


# ============================================================================
# HYBRID EXIT SIMULATION (core bar-by-bar engine)
# ============================================================================
def simulate_hybrid_exit(
    mid_prices: np.ndarray,
    day_boundaries: np.ndarray,
    predictions: np.ndarray,
    pred_indices: np.ndarray,
    # Signal threshold
    min_signal_quantile: float = 0.80,
    # Timing
    latency_bars: int = 1,
    fill_horizon_bars: int = 50,   # 5s
    hold_horizon_bars: int = 100,  # 10s
    # Exit parameters
    stop_loss_ticks: Optional[float] = None,   # None = no stop
    take_profit_ticks: Optional[float] = None, # None = no take profit
    use_limit_exit: bool = True,
    label: str = '',
) -> dict:
    """
    Bar-by-bar hybrid execution simulation.

    For each signal:
    1. Post limit order at best_bid/best_ask after latency delay
    2. Wait for entry fill (mid crosses limit within fill_horizon)
    3. After fill, place limit exit at the opposite side AND monitor bar-by-bar:
       - LIMIT EXIT: mid reaches the exit limit -> earn +0.5t exit edge
       - TAKE PROFIT: unrealized PnL >= take_profit_ticks -> market exit (pay -0.5t)
       - STOP LOSS: unrealized PnL <= -stop_loss_ticks -> market exit (pay -0.5t)
       - TIMEOUT: hold_horizon bars elapsed -> market exit (pay -0.5t)

    PnL components (all in ticks):
      entry_edge: +0.5 (passive limit fill earns half-spread)
      dir_pnl: (exit_price - fill_price) / TICK_SIZE * direction
      exit_edge: +0.5 if limit exit fills, -0.5 if market exit
      commission: -0.2 (COMMISSION_RT / TICK_VALUE)
      net = entry_edge + dir_pnl + exit_edge - commission
    """
    mode_str = f"hybrid[lmt_exit={use_limit_exit} sl={stop_loss_ticks} tp={take_profit_ticks}]"
    log.info(f"Simulation [{label or mode_str}]  "
             f"fill={fill_horizon_bars/BARS_PER_SEC:.0f}s  "
             f"hold={hold_horizon_bars/BARS_PER_SEC:.0f}s  "
             f"q={min_signal_quantile:.0%}  lat={latency_bars}bars")

    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    # Corrected bid/ask (1-tick spread)
    best_bid = mid_prices - HALF_TICK
    best_ask = mid_prices + HALF_TICK

    # Map predictions to full time axis
    pred_signal = np.full(N, np.nan)
    if len(predictions) > 0 and len(pred_indices) > 0:
        ok = (pred_indices >= 0) & (pred_indices < N)
        pred_signal[pred_indices[ok]] = predictions[ok]

    # Compute signal thresholds from predictions
    valid_p = predictions[np.isfinite(predictions)] if len(predictions) > 0 else np.array([])
    if len(valid_p) == 0:
        return {'error': 'No valid predictions'}
    thresh_pos = float(np.percentile(valid_p, min_signal_quantile * 100))
    thresh_neg = float(np.percentile(valid_p, (1 - min_signal_quantile) * 100))

    trades = []
    n_posted = 0
    n_not_filled = 0

    # Track day-by-day PnL for confidence intervals
    day_pnls = {}   # day_idx -> list of trade PnLs in dollars

    for d in range(n_days):
        ds, de = day_boundaries[d], day_boundaries[d + 1]
        min_needed = latency_bars + fill_horizon_bars + hold_horizon_bars + 5
        if de - ds < min_needed:
            continue

        m  = mid_prices[ds:de]
        bb = best_bid[ds:de]
        ba = best_ask[ds:de]
        ps = pred_signal[ds:de]
        L  = len(m)
        last_exit = -1
        day_pnls[d] = []

        # Max valid start bar
        max_start = L - latency_bars - fill_horizon_bars - hold_horizon_bars - 3

        i = 0
        while i < max_start:
            if i <= last_exit:
                i += 1
                continue

            sig = ps[i]
            if not np.isfinite(sig):
                i += 1
                continue

            # Determine direction
            if sig > thresh_pos:
                direction = 1
            elif sig < thresh_neg:
                direction = -1
            else:
                i += 1
                continue

            # Apply latency: post order at bar i + latency_bars
            post_bar = i + latency_bars
            if post_bar >= max_start:
                i += 1
                continue

            if not (np.isfinite(bb[post_bar]) and np.isfinite(ba[post_bar])):
                i += 1
                continue

            n_posted += 1

            # Post limit entry order
            if direction == 1:
                entry_lim = bb[post_bar]   # buy at best_bid
            else:
                entry_lim = ba[post_bar]   # sell at best_ask

            # --- Search for entry fill within fill_horizon ---
            fill_end = min(post_bar + fill_horizon_bars + 1, L)
            future_m = m[post_bar + 1:fill_end]
            if direction == 1:
                touch = np.where(future_m <= entry_lim)[0]
            else:
                touch = np.where(future_m >= entry_lim)[0]

            if len(touch) == 0:
                n_not_filled += 1
                i += 1
                continue

            fill_bar = post_bar + touch[0] + 1
            fill_mid = m[fill_bar]
            entry_edge = 0.5  # passive fill earns half-spread

            # --- Place limit exit at the opposite side ---
            # For a long: sell limit at best_ask (fill_mid + HALF_TICK)
            # For a short: buy limit at best_bid (fill_mid - HALF_TICK)
            if direction == 1:
                exit_limit_price = fill_mid + HALF_TICK   # limit sell
            else:
                exit_limit_price = fill_mid - HALF_TICK   # limit buy

            # --- BAR-BY-BAR EXIT MONITORING ---
            exit_type = 'timeout'
            exit_bar = fill_bar
            exit_mid = fill_mid
            limit_exit_filled = False

            max_hold = min(fill_bar + hold_horizon_bars, L - 1)

            for j in range(fill_bar + 1, max_hold + 1):
                curr_mid = m[j]
                unrealized_ticks = (curr_mid - fill_mid) / TICK_SIZE * direction

                # Check LIMIT EXIT first: did mid reach exit limit?
                if use_limit_exit:
                    if direction == 1 and curr_mid >= exit_limit_price:
                        # Long: limit sell fills when mid >= exit_limit_price
                        exit_type = 'limit_exit'
                        exit_bar = j
                        exit_mid = curr_mid
                        limit_exit_filled = True
                        break
                    elif direction == -1 and curr_mid <= exit_limit_price:
                        # Short: limit buy fills when mid <= exit_limit_price
                        exit_type = 'limit_exit'
                        exit_bar = j
                        exit_mid = curr_mid
                        limit_exit_filled = True
                        break

                # Check TAKE PROFIT
                if take_profit_ticks is not None and unrealized_ticks >= take_profit_ticks:
                    exit_type = 'take_profit'
                    exit_bar = j
                    exit_mid = curr_mid
                    break

                # Check STOP LOSS
                if stop_loss_ticks is not None and unrealized_ticks <= -stop_loss_ticks:
                    exit_type = 'stop_loss'
                    exit_bar = j
                    exit_mid = curr_mid
                    break

            else:
                # Loop completed without break -> timeout
                exit_type = 'timeout'
                exit_bar = max_hold
                exit_mid = m[max_hold]

            # --- Compute PnL ---
            dir_pnl_ticks = (exit_mid - fill_mid) / TICK_SIZE * direction

            if exit_type == 'limit_exit':
                exit_edge = 0.5    # passive limit fill earns half-spread
            else:
                exit_edge = -0.5   # market exit crosses half-spread

            net_ticks = entry_edge + dir_pnl_ticks + exit_edge - COMMISSION_TICKS
            net_dollars = net_ticks * TICK_VALUE

            trades.append({
                'day': int(d),
                'direction': int(direction),
                'signal': float(sig),
                'signal_bar': int(i),
                'post_bar': int(post_bar),
                'fill_bar': int(fill_bar),
                'exit_bar': int(exit_bar),
                'fill_offset_bars': int(fill_bar - post_bar),
                'bars_held': int(exit_bar - fill_bar),
                'fill_mid': float(fill_mid),
                'exit_mid': float(exit_mid),
                'entry_edge_ticks': float(entry_edge),
                'dir_pnl_ticks': float(dir_pnl_ticks),
                'exit_edge_ticks': float(exit_edge),
                'commission_ticks': float(COMMISSION_TICKS),
                'net_ticks': float(net_ticks),
                'net_dollars': float(net_dollars),
                'exit_type': exit_type,
            })
            day_pnls[d].append(net_dollars)
            last_exit = exit_bar
            i = exit_bar + 1  # advance past exit bar

        progress(f"  Hybrid sim [{label}] day {d+1}/{n_days} trades={len(trades)}")

    if not trades:
        return {
            'error': 'No trades generated',
            'n_posted': n_posted,
            'n_not_filled': n_not_filled,
            'config': {
                'fill_horizon_sec': fill_horizon_bars / BARS_PER_SEC,
                'hold_horizon_sec': hold_horizon_bars / BARS_PER_SEC,
                'min_signal_quantile': min_signal_quantile,
                'latency_bars': latency_bars,
                'stop_loss_ticks': stop_loss_ticks,
                'take_profit_ticks': take_profit_ticks,
                'use_limit_exit': use_limit_exit,
            },
        }

    # Compute aggregate metrics
    pnl = np.array([t['net_ticks'] for t in trades])
    dir_pnl = np.array([t['dir_pnl_ticks'] for t in trades])
    pnl_dollars = pnl * TICK_VALUE

    # Approximate annualized Sharpe
    avg_hold_sec = float(np.mean([t['bars_held'] for t in trades])) / BARS_PER_SEC
    if avg_hold_sec > 0 and np.std(pnl) > 0:
        trades_per_year = 252 * 6.5 * 3600 / max(avg_hold_sec, 1.0)
        sharpe = float(np.mean(pnl) / np.std(pnl) * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    # Profit factor
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = (float(wins.sum() / (-losses.sum()))
                     if len(losses) > 0 and -losses.sum() > 0 else 999.0)

    # Max drawdown
    cum_pnl = np.cumsum(pnl_dollars)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = running_max - cum_pnl
    max_drawdown = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    # Fill rate
    fill_rate = len(trades) / n_posted if n_posted > 0 else 0.0

    # Exit type breakdown
    exit_counts = {}
    for t in trades:
        exit_counts[t['exit_type']] = exit_counts.get(t['exit_type'], 0) + 1
    n_total = len(trades)
    exit_breakdown = {k: {'count': v, 'pct': v / n_total}
                      for k, v in exit_counts.items()}

    # PnL by exit type
    pnl_by_exit = {}
    for et in ['limit_exit', 'take_profit', 'stop_loss', 'timeout']:
        et_trades = [t for t in trades if t['exit_type'] == et]
        if et_trades:
            et_pnl = np.array([t['net_ticks'] for t in et_trades])
            pnl_by_exit[et] = {
                'n': len(et_trades),
                'mean_pnl_ticks': float(np.mean(et_pnl)),
                'mean_pnl_dollars': float(np.mean(et_pnl) * TICK_VALUE),
                'win_rate': float((et_pnl > 0).mean()),
            }

    # Day-by-day PnL for confidence intervals
    day_pnl_totals = []
    for d_idx in sorted(day_pnls.keys()):
        day_trades = day_pnls[d_idx]
        if day_trades:
            day_pnl_totals.append(float(sum(day_trades)))
    n_pos_days = sum(1 for x in day_pnl_totals if x > 0)
    n_neg_days = sum(1 for x in day_pnl_totals if x < 0)

    # Adverse selection: mean move at different points
    mean_fill_offset = float(np.mean([t['fill_offset_bars'] for t in trades])) / BARS_PER_SEC

    n_days_actual = n_days
    trades_per_day = len(trades) / max(n_days_actual, 1)

    result = {
        'config': {
            'fill_horizon_sec': fill_horizon_bars / BARS_PER_SEC,
            'hold_horizon_sec': hold_horizon_bars / BARS_PER_SEC,
            'min_signal_quantile': min_signal_quantile,
            'latency_bars': latency_bars,
            'latency_sec': latency_bars / BARS_PER_SEC,
            'stop_loss_ticks': stop_loss_ticks,
            'take_profit_ticks': take_profit_ticks,
            'use_limit_exit': use_limit_exit,
        },
        'n_posted': n_posted,
        'n_filled': len(trades),
        'n_not_filled': n_not_filled,
        'fill_rate': fill_rate,
        'mean_pnl_ticks': float(np.mean(pnl)),
        'std_pnl_ticks': float(np.std(pnl)),
        'mean_pnl_dollars': float(np.mean(pnl_dollars)),
        'total_pnl_dollars': float(np.sum(pnl_dollars)),
        'win_rate': float((pnl > 0).mean()),
        'sharpe_approx': sharpe,
        'profit_factor': min(float(profit_factor), 999.0),
        'max_drawdown_dollars': max_drawdown,
        'trades_per_day': trades_per_day,
        'mean_dir_pnl_ticks': float(np.mean(dir_pnl)),
        'mean_entry_edge_ticks': float(np.mean([t['entry_edge_ticks'] for t in trades])),
        'mean_exit_edge_ticks': float(np.mean([t['exit_edge_ticks'] for t in trades])),
        'mean_fill_offset_sec': mean_fill_offset,
        'mean_bars_held': float(np.mean([t['bars_held'] for t in trades])),
        'exit_type_breakdown': exit_breakdown,
        'pnl_by_exit_type': pnl_by_exit,
        'day_by_day_pnl_dollars': day_pnl_totals,
        'n_positive_days': n_pos_days,
        'n_negative_days': n_neg_days,
        'day_pnl_mean': float(np.mean(day_pnl_totals)) if day_pnl_totals else 0.0,
        'day_pnl_std': float(np.std(day_pnl_totals)) if day_pnl_totals else 0.0,
    }

    log.info(f"  [{label}] n_filled={len(trades)}  fill_rate={fill_rate:.1%}  "
             f"mean_pnl={result['mean_pnl_ticks']:+.4f}t (${result['mean_pnl_dollars']:+.2f})  "
             f"win={result['win_rate']:.1%}  Sharpe={sharpe:.2f}  "
             f"total=${result['total_pnl_dollars']:+.2f}")
    if exit_breakdown:
        parts = [f"{k}={v['count']}({v['pct']:.0%})" for k, v in exit_breakdown.items()]
        log.info(f"  Exit types: {' | '.join(parts)}")
    return result


# ============================================================================
# MARKET EXIT BASELINE (simpler version for comparison)
# ============================================================================
def simulate_market_exit_baseline(
    mid_prices, day_boundaries, predictions, pred_indices,
    min_signal_quantile, latency_bars, hold_horizon_bars,
    fill_horizon_bars=50, label='mkt_exit'
):
    """Pure market-exit baseline (no stop/TP/limit exit complexity)."""
    return simulate_hybrid_exit(
        mid_prices=mid_prices,
        day_boundaries=day_boundaries,
        predictions=predictions,
        pred_indices=pred_indices,
        min_signal_quantile=min_signal_quantile,
        latency_bars=latency_bars,
        fill_horizon_bars=fill_horizon_bars,
        hold_horizon_bars=hold_horizon_bars,
        stop_loss_ticks=None,
        take_profit_ticks=None,
        use_limit_exit=False,
        label=label,
    )


# ============================================================================
# PARAMETER GRID SEARCH
# ============================================================================
def run_parameter_grid(
    mid_prices, day_boundaries, predictions, pred_indices,
    fast=False
):
    """
    Full parameter grid search over all combinations.

    Returns dict with all_results (sorted by Sharpe), best configs.
    """
    log.info("=" * 60)
    log.info("PARAMETER GRID SEARCH")
    log.info("=" * 60)

    # Grid definition
    quantiles = [0.80, 0.90, 0.95] if fast else [0.70, 0.80, 0.90, 0.95]
    latencies = [0, 1, 3] if fast else [0, 1, 3]
    hold_bars_list = [50, 100, 300] if fast else [30, 50, 100, 300]
    stop_losses = [None, 2.0, 3.0] if fast else [None, 2.0, 3.0, 5.0]
    take_profits = [None, 2.0, 3.0] if fast else [None, 2.0, 3.0, 5.0]
    fill_horizon_bars = 50  # 5s

    # Phase 1: Market-exit baseline grid (no stop/TP)
    log.info("Phase 1: Market-exit baseline grid (no stop/TP/limit exit)")
    mkt_results = []
    mkt_configs = [(q, lat, hb) for q in quantiles
                                 for lat in latencies
                                 for hb in hold_bars_list]
    total_mkt = len(mkt_configs)
    for idx, (q, lat, hb) in enumerate(mkt_configs):
        r = simulate_market_exit_baseline(
            mid_prices=mid_prices,
            day_boundaries=day_boundaries,
            predictions=predictions,
            pred_indices=pred_indices,
            min_signal_quantile=q,
            latency_bars=lat,
            hold_horizon_bars=hb,
            fill_horizon_bars=fill_horizon_bars,
            label=f'mkt q={q:.0%} lat={lat} hold={hb/BARS_PER_SEC:.0f}s',
        )
        if 'error' not in r:
            mkt_results.append({
                'exit_mode': 'market_exit',
                'quantile': q,
                'latency_bars': lat,
                'latency_ms': lat * 100,
                'hold_sec': hb / BARS_PER_SEC,
                'stop_loss_ticks': None,
                'take_profit_ticks': None,
                **{k: r.get(k) for k in [
                    'n_filled', 'fill_rate', 'mean_pnl_ticks', 'mean_pnl_dollars',
                    'total_pnl_dollars', 'sharpe_approx', 'win_rate', 'profit_factor',
                    'max_drawdown_dollars', 'trades_per_day', 'mean_dir_pnl_ticks',
                    'n_positive_days', 'n_negative_days', 'day_pnl_mean', 'day_pnl_std',
                ]},
                'exit_type_breakdown': r.get('exit_type_breakdown', {}),
                'day_by_day_pnl_dollars': r.get('day_by_day_pnl_dollars', []),
            })
        progress(f"  Phase 1: {idx+1}/{total_mkt} configs")
    log.info(f"  Phase 1 complete: {len(mkt_results)} valid market-exit configs")

    # Sort phase 1 results
    mkt_by_sharpe = sorted(mkt_results, key=lambda x: x.get('sharpe_approx', 0), reverse=True)
    if mkt_by_sharpe:
        best_mkt = mkt_by_sharpe[0]
        log.info(f"  Best mkt-exit: q={best_mkt['quantile']:.0%} "
                 f"lat={best_mkt['latency_ms']:.0f}ms "
                 f"hold={best_mkt['hold_sec']:.0f}s "
                 f"Sharpe={best_mkt['sharpe_approx']:.2f} "
                 f"mean={best_mkt['mean_pnl_ticks']:+.4f}t")

    # Phase 2: Hybrid exit on top configs (best market-exit quantile/latency combos)
    log.info("Phase 2: Hybrid exit grid (limit exit + stop/TP combos)")

    # Take best market-exit base configs (top 3 by Sharpe, uniquely)
    seen_base = set()
    top_base_configs = []
    for r in mkt_by_sharpe:
        key = (r['quantile'], r['latency_bars'])
        if key not in seen_base:
            seen_base.add(key)
            top_base_configs.append((r['quantile'], r['latency_bars']))
        if len(top_base_configs) >= 2:
            break

    hybrid_results = []
    # Grid: top 2 base configs x all hold periods x stop x TP x limit_exit
    hybrid_configs = []
    for (q, lat) in top_base_configs:
        for hb in hold_bars_list:
            for sl in stop_losses:
                for tp in take_profits:
                    for use_lmt in [True, False]:
                        # Skip pure market exit (already done in phase 1)
                        if not use_lmt and sl is None and tp is None:
                            continue
                        hybrid_configs.append((q, lat, hb, sl, tp, use_lmt))

    total_hybrid = len(hybrid_configs)
    log.info(f"  Testing {total_hybrid} hybrid configurations...")

    for idx, (q, lat, hb, sl, tp, use_lmt) in enumerate(hybrid_configs):
        lbl = (f"hybrid q={q:.0%} lat={lat} hold={hb/BARS_PER_SEC:.0f}s "
               f"sl={sl} tp={tp} lmt={use_lmt}")
        r = simulate_hybrid_exit(
            mid_prices=mid_prices,
            day_boundaries=day_boundaries,
            predictions=predictions,
            pred_indices=pred_indices,
            min_signal_quantile=q,
            latency_bars=lat,
            fill_horizon_bars=fill_horizon_bars,
            hold_horizon_bars=hb,
            stop_loss_ticks=sl,
            take_profit_ticks=tp,
            use_limit_exit=use_lmt,
            label=lbl,
        )
        if 'error' not in r:
            exit_mode = 'hybrid_limit' if use_lmt else 'hybrid_mkt'
            if sl is not None or tp is not None:
                exit_mode += f'_sl{sl}_tp{tp}'
            hybrid_results.append({
                'exit_mode': exit_mode,
                'quantile': q,
                'latency_bars': lat,
                'latency_ms': lat * 100,
                'hold_sec': hb / BARS_PER_SEC,
                'stop_loss_ticks': sl,
                'take_profit_ticks': tp,
                'use_limit_exit': use_lmt,
                **{k: r.get(k) for k in [
                    'n_filled', 'fill_rate', 'mean_pnl_ticks', 'mean_pnl_dollars',
                    'total_pnl_dollars', 'sharpe_approx', 'win_rate', 'profit_factor',
                    'max_drawdown_dollars', 'trades_per_day', 'mean_dir_pnl_ticks',
                    'n_positive_days', 'n_negative_days', 'day_pnl_mean', 'day_pnl_std',
                ]},
                'exit_type_breakdown': r.get('exit_type_breakdown', {}),
                'pnl_by_exit_type': r.get('pnl_by_exit_type', {}),
                'day_by_day_pnl_dollars': r.get('day_by_day_pnl_dollars', []),
            })
        progress(f"  Phase 2: {idx+1}/{total_hybrid} hybrid configs")

    log.info(f"  Phase 2 complete: {len(hybrid_results)} valid hybrid configs")

    # Combine all results
    all_results = mkt_results + hybrid_results

    if not all_results:
        return {'error': 'No valid configurations found', 'all_results': []}

    # Sort
    by_sharpe = sorted(all_results, key=lambda x: x.get('sharpe_approx', 0) or 0, reverse=True)
    by_pnl = sorted(all_results, key=lambda x: x.get('total_pnl_dollars', 0) or 0, reverse=True)
    by_mean_pnl = sorted(all_results, key=lambda x: x.get('mean_pnl_ticks', 0) or 0, reverse=True)

    # Top 10 summary
    log.info(f"\n  {'Mode':20s} {'q':>6s} {'lat_ms':>7s} {'hold':>6s} {'sl':>4s} {'tp':>4s} "
             f"{'Sharpe':>8s} {'mean_t':>8s} {'total_$':>10s} {'win%':>6s}")
    for r in by_sharpe[:10]:
        log.info(f"  {r['exit_mode'][:20]:20s} "
                 f"{r['quantile']:>6.0%} "
                 f"{r['latency_ms']:>7.0f} "
                 f"{r['hold_sec']:>6.0f}s "
                 f"{str(r.get('stop_loss_ticks', '-')):>4s} "
                 f"{str(r.get('take_profit_ticks', '-')):>4s} "
                 f"{r.get('sharpe_approx', 0) or 0:>8.2f} "
                 f"{r.get('mean_pnl_ticks', 0) or 0:>+8.4f}t "
                 f"${r.get('total_pnl_dollars', 0) or 0:>9.2f} "
                 f"{r.get('win_rate', 0) or 0:>6.1%}")

    return {
        'all_results': all_results,
        'mkt_exit_results': mkt_results,
        'hybrid_results': hybrid_results,
        'best_by_sharpe': by_sharpe[0] if by_sharpe else {},
        'best_by_pnl': by_pnl[0] if by_pnl else {},
        'best_by_mean_pnl': by_mean_pnl[0] if by_mean_pnl else {},
        'top5_by_sharpe': by_sharpe[:5],
        'top5_by_pnl': by_pnl[:5],
        'n_configs_tested': len(all_results),
    }


# ============================================================================
# HYBRID VS MARKET EXIT COMPARISON
# ============================================================================
def analyze_hybrid_vs_market(grid_results):
    """
    For the best market-exit config, compare to best hybrid config.
    Calculate improvement from hybrid exit.
    """
    log.info("=" * 60)
    log.info("HYBRID vs MARKET EXIT COMPARISON")
    log.info("=" * 60)

    mkt_results = grid_results.get('mkt_exit_results', [])
    hybrid_results = grid_results.get('hybrid_results', [])

    if not mkt_results or not hybrid_results:
        return {'error': 'Missing results for comparison'}

    mkt_by_sharpe = sorted(mkt_results, key=lambda x: x.get('sharpe_approx', 0) or 0, reverse=True)
    hyb_by_sharpe = sorted(hybrid_results, key=lambda x: x.get('sharpe_approx', 0) or 0, reverse=True)

    best_mkt = mkt_by_sharpe[0] if mkt_by_sharpe else {}
    best_hyb = hyb_by_sharpe[0] if hyb_by_sharpe else {}

    if not best_mkt or not best_hyb:
        return {'error': 'No valid configs to compare'}

    mkt_pnl = best_mkt.get('mean_pnl_ticks', 0) or 0
    hyb_pnl = best_hyb.get('mean_pnl_ticks', 0) or 0
    mkt_sharpe = best_mkt.get('sharpe_approx', 0) or 0
    hyb_sharpe = best_hyb.get('sharpe_approx', 0) or 0

    pnl_improvement = hyb_pnl - mkt_pnl
    sharpe_improvement = hyb_sharpe - mkt_sharpe

    log.info(f"  Best market exit: q={best_mkt.get('quantile', 0):.0%} "
             f"lat={best_mkt.get('latency_ms', 0):.0f}ms "
             f"hold={best_mkt.get('hold_sec', 0):.0f}s "
             f"Sharpe={mkt_sharpe:.2f} mean={mkt_pnl:+.4f}t")
    log.info(f"  Best hybrid exit: mode={best_hyb.get('exit_mode', '?')} "
             f"sl={best_hyb.get('stop_loss_ticks')} "
             f"tp={best_hyb.get('take_profit_ticks')} "
             f"Sharpe={hyb_sharpe:.2f} mean={hyb_pnl:+.4f}t")
    log.info(f"  Improvement: PnL {pnl_improvement:+.4f}t  Sharpe {sharpe_improvement:+.2f}")

    # Limit exit fill rate analysis
    lmt_results = [r for r in hybrid_results
                   if r.get('use_limit_exit', False)]
    if lmt_results:
        lmt_by_sharpe = sorted(lmt_results, key=lambda x: x.get('sharpe_approx', 0) or 0, reverse=True)
        best_lmt = lmt_by_sharpe[0]
        lmt_breakdown = best_lmt.get('exit_type_breakdown', {})
        lmt_pct = lmt_breakdown.get('limit_exit', {}).get('pct', 0.0)
        log.info(f"  Best limit-exit config: {lmt_pct:.1%} of exits fill as limits")

        # What % of exits are limits across all lmt-exit configs?
        all_lmt_pcts = []
        for r in lmt_results:
            pct = r.get('exit_type_breakdown', {}).get('limit_exit', {}).get('pct', 0.0)
            if pct > 0:
                all_lmt_pcts.append(pct)
        if all_lmt_pcts:
            log.info(f"  Limit fill rate across lmt-exit configs: "
                     f"mean={np.mean(all_lmt_pcts):.1%} "
                     f"range=[{np.min(all_lmt_pcts):.1%}, {np.max(all_lmt_pcts):.1%}]")

    return {
        'best_market_exit': best_mkt,
        'best_hybrid_exit': best_hyb,
        'pnl_improvement_ticks': pnl_improvement,
        'sharpe_improvement': sharpe_improvement,
        'pnl_improvement_dollars': pnl_improvement * TICK_VALUE,
    }


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVAL
# ============================================================================
def bootstrap_daily_pnl(day_pnl_list, n_bootstrap=1000, confidence=0.95):
    """Bootstrap confidence interval on mean daily PnL."""
    if len(day_pnl_list) < 3:
        return {'error': 'Too few days for bootstrap'}

    day_arr = np.array(day_pnl_list)
    n = len(day_arr)
    bootstrap_means = []
    rng = np.random.default_rng(42)
    for _ in range(n_bootstrap):
        sample = rng.choice(day_arr, size=n, replace=True)
        bootstrap_means.append(float(np.mean(sample)))

    alpha = 1 - confidence
    lo = float(np.percentile(bootstrap_means, alpha / 2 * 100))
    hi = float(np.percentile(bootstrap_means, (1 - alpha / 2) * 100))
    return {
        'mean': float(np.mean(day_arr)),
        'std': float(np.std(day_arr)),
        'ci_lo': lo,
        'ci_hi': hi,
        'confidence': confidence,
        'n_days': n,
        'n_pos_days': int((day_arr > 0).sum()),
        'n_neg_days': int((day_arr < 0).sum()),
    }


# ============================================================================
# COMPREHENSIVE SUMMARY
# ============================================================================
def format_summary(n_bars, n_days, dir_result, grid_results, comparison):
    lines = [
        "", "=" * 70,
        "HYBRID EXECUTION SIMULATOR -- SUMMARY",
        "ES Futures: Limit Entry + Smart Exit (Stop/TP/Limit/Timeout)",
        f"Spread: 1 tick always (bid=mid-0.125, ask=mid+0.125)",
        f"Commission: ${COMMISSION_RT} RT = {COMMISSION_TICKS:.1f} ticks",
        "=" * 70, "",
        "DATASET:",
        f"  Bars: {n_bars:,}  Days: {n_days}",
        "",
        "DIRECTION MODEL (ret_3s walk-forward LightGBM):",
        f"  IC={dir_result.get('ic', float('nan')):.4f}  "
        f"ICIR={dir_result.get('icir', float('nan')):.2f}  "
        f"t-stat={dir_result.get('tstat', float('nan')):.2f}  "
        f"n_preds={dir_result.get('n_preds', 0):,}",
        "",
    ]

    all_results = grid_results.get('all_results', [])
    if not all_results:
        lines.append("  No results generated.")
        return "\n".join(lines)

    # Market exit baseline summary
    mkt_results = grid_results.get('mkt_exit_results', [])
    mkt_by_sharpe = sorted(mkt_results, key=lambda x: x.get('sharpe_approx', 0) or 0, reverse=True)
    lines += ["MARKET EXIT BASELINE (Top 5 by Sharpe):"]
    lines.append(f"  {'q':>6s} {'lat_ms':>7s} {'hold_s':>7s} {'Sharpe':>8s} "
                 f"{'mean_t':>8s} {'mean_$':>8s} {'win%':>6s} {'n':>6s}")
    for r in mkt_by_sharpe[:5]:
        lines.append(f"  {r.get('quantile', 0):>6.0%} "
                     f"{r.get('latency_ms', 0):>7.0f} "
                     f"{r.get('hold_sec', 0):>7.0f} "
                     f"{r.get('sharpe_approx', 0) or 0:>8.2f} "
                     f"{r.get('mean_pnl_ticks', 0) or 0:>+8.4f}t "
                     f"${r.get('mean_pnl_dollars', 0) or 0:>+7.2f} "
                     f"{r.get('win_rate', 0) or 0:>6.1%} "
                     f"{r.get('n_filled', 0):>6d}")

    # Hybrid summary
    lines += ["", "HYBRID EXIT RESULTS (Top 5 by Sharpe, across all modes):"]
    by_sharpe = sorted(all_results, key=lambda x: x.get('sharpe_approx', 0) or 0, reverse=True)
    lines.append(f"  {'mode':20s} {'q':>6s} {'sl':>4s} {'tp':>4s} "
                 f"{'Sharpe':>8s} {'mean_t':>8s} {'mean_$':>8s} {'win%':>6s}")
    for r in by_sharpe[:5]:
        lines.append(f"  {r.get('exit_mode', '?')[:20]:20s} "
                     f"{r.get('quantile', 0):>6.0%} "
                     f"{str(r.get('stop_loss_ticks', '-')):>4s} "
                     f"{str(r.get('take_profit_ticks', '-')):>4s} "
                     f"{r.get('sharpe_approx', 0) or 0:>8.2f} "
                     f"{r.get('mean_pnl_ticks', 0) or 0:>+8.4f}t "
                     f"${r.get('mean_pnl_dollars', 0) or 0:>+7.2f} "
                     f"{r.get('win_rate', 0) or 0:>6.1%}")

    # Comparison
    lines += ["", "HYBRID vs MARKET EXIT COMPARISON:"]
    if 'error' not in comparison:
        bm = comparison.get('best_market_exit', {})
        bh = comparison.get('best_hybrid_exit', {})
        pnl_imp = comparison.get('pnl_improvement_ticks', 0) or 0
        sharpe_imp = comparison.get('sharpe_improvement', 0) or 0
        lines += [
            f"  Best market exit:  Sharpe={bm.get('sharpe_approx', 0) or 0:.2f}  "
            f"mean={bm.get('mean_pnl_ticks', 0) or 0:+.4f}t  "
            f"(${bm.get('mean_pnl_dollars', 0) or 0:+.2f}/trade)",
            f"  Best hybrid exit:  Sharpe={bh.get('sharpe_approx', 0) or 0:.2f}  "
            f"mean={bh.get('mean_pnl_ticks', 0) or 0:+.4f}t  "
            f"(${bh.get('mean_pnl_dollars', 0) or 0:+.2f}/trade)",
            f"  Improvement:       Sharpe {sharpe_imp:+.2f}  PnL {pnl_imp:+.4f}t  "
            f"(${pnl_imp * TICK_VALUE:+.2f}/trade)",
        ]
        # Exit type breakdown for best hybrid
        bh_breakdown = bh.get('exit_type_breakdown', {})
        if bh_breakdown:
            parts = [f"{k}={v.get('pct', 0):.0%}" for k, v in bh_breakdown.items()]
            lines.append(f"  Exit breakdown: {' | '.join(parts)}")

    # Overall best config
    best_overall = grid_results.get('best_by_sharpe', {})
    lines += ["", "BEST OVERALL CONFIGURATION:"]
    if best_overall:
        lines += [
            f"  Exit mode:        {best_overall.get('exit_mode', '?')}",
            f"  Signal quantile:  {best_overall.get('quantile', 0):.0%}",
            f"  Latency:          {best_overall.get('latency_ms', 0):.0f}ms",
            f"  Hold period:      {best_overall.get('hold_sec', 0):.0f}s",
            f"  Stop loss:        {best_overall.get('stop_loss_ticks', 'None')} ticks",
            f"  Take profit:      {best_overall.get('take_profit_ticks', 'None')} ticks",
            f"  Limit exit:       {best_overall.get('use_limit_exit', False)}",
            f"  Sharpe:           {best_overall.get('sharpe_approx', 0) or 0:.2f}",
            f"  Mean PnL/trade:   {best_overall.get('mean_pnl_ticks', 0) or 0:+.4f}t "
            f"(${best_overall.get('mean_pnl_dollars', 0) or 0:+.2f})",
            f"  Total PnL:        ${best_overall.get('total_pnl_dollars', 0) or 0:+.2f}",
            f"  Win rate:         {best_overall.get('win_rate', 0) or 0:.1%}",
            f"  Fill rate:        {best_overall.get('fill_rate', 0) or 0:.1%}",
            f"  Trades/day:       {best_overall.get('trades_per_day', 0) or 0:.1f}",
            f"  Pos/Neg days:     {best_overall.get('n_positive_days', 0)}/{best_overall.get('n_negative_days', 0)}",
        ]

        # Bootstrap CI
        day_pnls = best_overall.get('day_by_day_pnl_dollars', [])
        if len(day_pnls) >= 3:
            ci = bootstrap_daily_pnl(day_pnls)
            if 'error' not in ci:
                lines += [
                    f"  Daily PnL: mean=${ci['mean']:+.2f}  "
                    f"95% CI=[${ci['ci_lo']:+.2f}, ${ci['ci_hi']:+.2f}]",
                ]

    # PnL insight
    lines += [
        "", "=" * 70,
        "KEY INSIGHTS:",
        "  Limit entry: +0.5t edge (passive fill earns half-spread)",
        "  Limit exit:  +0.5t edge (if fills) -> total +1.0t = $12.50 vs costs",
        "  Market exit: -0.5t cost -> net = dir_pnl - 0.2t commission",
        "  Stop loss reduces MAE but sacrifices limit exit opportunities",
        "  Take profit locks gains but exits before limit fill at exit level",
        "  Fill probability at bid/ask: ~6.5% in 3s, ~10.8% in 10s",
        "  Optimal: limit exit gives ~1.0t edge if fill rate is reasonable",
        "=" * 70,
    ]

    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Hybrid Execution Simulator -- ES Futures Limit Order Trading'
    )
    parser.add_argument('--fast', action='store_true',
                        help='Smaller grid for quick testing')
    parser.add_argument('--min-train-days', type=int, default=2,
                        help='Minimum training days for walk-forward model')
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("HYBRID EXECUTION SIMULATOR -- ES FUTURES LIMIT ORDER TRADING")
    log.info(f"  Mode: {'FAST' if args.fast else 'FULL'}")
    log.info(f"  TICK_VALUE=${TICK_VALUE}  COMMISSION=${COMMISSION_RT}RT")
    log.info(f"  Spread: 1 tick always (bid=mid-0.125, ask=mid+0.125)")
    log.info("=" * 70)
    t_start = time.time()

    # ---- Load data ----
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    stats = load_feature_cache(scanner)
    if stats is None:
        log.info("No feature cache -- computing from scratch...")
        stats = scanner.load_from_cache()

    N = len(scanner.mid_prices)
    n_days = len(scanner.day_boundaries) - 1
    log.info(f"Loaded: {N:,} bars, {n_days} days, {len(scanner.feature_names)} features")

    # ---- Corrected mid prices (same as corrected limit study) ----
    mid_prices = scanner.mid_prices.copy()
    log.info(f"  Mid price mean: ${np.nanmean(mid_prices):.2f}")
    log.info(f"  best_bid = mid - {HALF_TICK:.3f}  best_ask = mid + {HALF_TICK:.3f}")
    log.info(f"  Spread = {TICK_SIZE:.2f} = 1 tick (corrected)")

    # ---- Compute ret_3s target ----
    log.info("Computing ret_3s target...")
    targets = compute_return_targets(
        mid_prices=mid_prices,
        day_boundaries=scanner.day_boundaries,
        sample_interval_ms=100,
        horizons_sec={'3s': 3},
        include_flow_target=False,
    )
    ret_3s = targets['ret_3s']

    # ---- Train direction model ----
    predictions, pred_indices, dir_result = get_direction_model(
        scanner=scanner, target=ret_3s,
        min_train_days=args.min_train_days, fast=args.fast,
    )
    gc.collect()

    if len(predictions) == 0:
        log.error("No predictions generated. Exiting.")
        return

    # ---- Parameter Grid Search ----
    grid_results = run_parameter_grid(
        mid_prices=mid_prices,
        day_boundaries=scanner.day_boundaries,
        predictions=predictions,
        pred_indices=pred_indices,
        fast=args.fast,
    )
    gc.collect()

    # ---- Hybrid vs Market Exit Comparison ----
    comparison = analyze_hybrid_vs_market(grid_results)

    # ---- Best config deep dive ----
    best_overall = grid_results.get('best_by_sharpe', {})
    best_day_pnls = best_overall.get('day_by_day_pnl_dollars', [])
    best_ci = bootstrap_daily_pnl(best_day_pnls) if len(best_day_pnls) >= 3 else {}

    log.info("=" * 60)
    log.info("BEST CONFIG CONFIDENCE INTERVAL (bootstrap)")
    if 'error' not in best_ci and best_ci:
        log.info(f"  Daily PnL: mean=${best_ci['mean']:+.2f}  std=${best_ci['std']:.2f}")
        log.info(f"  95% CI: [${best_ci['ci_lo']:+.2f}, ${best_ci['ci_hi']:+.2f}]")
        log.info(f"  Positive days: {best_ci['n_pos_days']}/{best_ci['n_days']} "
                 f"({best_ci['n_pos_days']/best_ci['n_days']:.1%})")

    # ---- Summary ----
    summary_text = format_summary(
        n_bars=N, n_days=n_days,
        dir_result=dir_result,
        grid_results=grid_results,
        comparison=comparison,
    )
    log.info("\n" + summary_text)

    # ---- Save Results ----
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'hybrid_execution_{timestamp}.json'

    output = {
        'timestamp': timestamp,
        'mode': 'fast' if args.fast else 'full',
        'constants': {
            'tick_size': TICK_SIZE,
            'tick_value': TICK_VALUE,
            'es_point_value': ES_POINT_VALUE,
            'half_tick': HALF_TICK,
            'commission_rt': COMMISSION_RT,
            'commission_ticks': COMMISSION_TICKS,
            'bars_per_sec': BARS_PER_SEC,
        },
        'data': {
            'n_bars': int(N),
            'n_days': int(n_days),
            'mid_price_mean': float(np.nanmean(mid_prices)),
            'spread_assumption': '1_tick_always',
        },
        'direction_model': {
            k: dir_result.get(k)
            for k in ['ic', 'icir', 'tstat', 'n_preds', 'fold_ics', 'fold_con']
        },
        'grid_results': {
            'n_configs_tested': grid_results.get('n_configs_tested', 0),
            'best_by_sharpe': to_safe(grid_results.get('best_by_sharpe', {})),
            'best_by_pnl': to_safe(grid_results.get('best_by_pnl', {})),
            'best_by_mean_pnl': to_safe(grid_results.get('best_by_mean_pnl', {})),
            'top5_by_sharpe': to_safe(grid_results.get('top5_by_sharpe', [])),
            'top5_by_pnl': to_safe(grid_results.get('top5_by_pnl', [])),
            'all_results': to_safe(grid_results.get('all_results', [])),
        },
        'comparison': to_safe(comparison),
        'best_config_bootstrap_ci': to_safe(best_ci),
        'summary_text': summary_text,
        'total_elapsed_sec': time.time() - t_start,
    }

    with open(str(out_file), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    log.info(f"\nResults saved: {out_file}")
    log.info(f"Total elapsed: {(time.time() - t_start) / 60:.1f} min")

    # Print final summary to stdout
    print("\n" + "=" * 70)
    print(summary_text)
    print(f"\nResults saved: {out_file}")

    return output


if __name__ == '__main__':
    main()
