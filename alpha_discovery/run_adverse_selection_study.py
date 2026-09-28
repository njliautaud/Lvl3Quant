"""
Adverse Selection Study -- ES Futures MBO Alpha
================================================

CONTEXT:
  Two existing simulations give wildly different results:
    1. corrected_limit_study:  45/45 profitable, Sharpe 1074 -- OVERLY OPTIMISTIC
       (fill model: price touches our level = 100% fill)
    2. combined_strategy:      ALL negative, 0% profitable -- OVERLY PESSIMISTIC
       (Poisson fill rate = 0.4%, adverse selection dominates)

  The GAP is in the fill model. This script bridges them with a 4-tier
  realistic fill model grounded in empirical MBO data findings:
    - Displayed queue depth at best bid: 2-3 contracts (median)
    - 76% of orders are iceberg (display qty=1)
    - Effective queue (volume before price moves): ~20 contracts
    - CME iceberg refills get NEW timestamps -> we jump ahead of refills
    - Realistic queue position for us: 3-5 contracts ahead

FILL MODEL TIERS:
  Tier 1 (BEST CASE / OPTIMISTIC):
    Price touches our level -> 100% fill.
    Replicates corrected_limit_study. Used as baseline.

  Tier 2 (MBO-INFORMED):
    Price touches our level AND sufficient volume trades through.
    Queue position = max(1, bid_L1_orders proxy from features).
    Fill if cumulative volume at that price >= queue_position within window.
    We use bid_L1_orders / ask_L1_orders features as queue depth proxy.

  Tier 3 (ADVERSE-SELECTION ADJUSTED):
    Same as Tier 2 PLUS tracks mid-price movement AFTER fill.
    If mid moves >= 1 tick against us within 1s of fill -> adverse fill.
    Reports PnL split: adverse fills vs favorable fills.
    This quantifies the "informed trader" cost of being filled.

  Tier 4 (PESSIMISTIC / POISSON):
    Uses Poisson arrival model like combined_strategy.
    Fill rate derived from historical trade arrival rate and queue depth.
    Provides lower bound consistent with combined_strategy findings.

EXIT MODELS (for each entry tier):
  A. Market exit at hold_time: cross half-spread on exit (-0.5t)
  B. Limit exit: requires same tiered fill model for exits

KEY METRICS per tier:
  - Fill rate (% of signals -> trades)
  - Adverse selection rate (% fills where price moves against within 1s)
  - Mean PnL per trade (ticks and dollars)
  - Win rate
  - Sharpe ratio (annualized, trade-level)
  - Trades per day
  - PnL conditional on adverse vs favorable fills

CONSTANTS:
  Tick size:        $0.25 (ES minimum)
  Tick value:       $12.50 per tick per contract
  Commission:       $3.00 RT = 0.24 ticks (AMP + Rithmic + CME fees)
  Spread:           1 tick always (corrected -- cached spread is buggy)
  Bars per second:  10 (100ms bars)
  Adverse window:   10 bars = 1 second after fill

Usage:
    python alpha_discovery/run_adverse_selection_study.py
    python alpha_discovery/run_adverse_selection_study.py --fast
    python alpha_discovery/run_adverse_selection_study.py --hold 10 --signal-q 0.70
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
from typing import Dict, List, Optional, Tuple

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.run_return_multihorizon import (
    load_feature_cache,
    compute_return_targets,
    EXCLUDE_FEATURES_DIRECTION,
)
from alpha_discovery.run_model_refinement import walk_forward_evaluate

# ============================================================================
# LOGGING  (ASCII-only for Windows cp1252 compatibility)
# ============================================================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            str(RESULTS_DIR / 'adverse_selection_study.log'),
            mode='a', encoding='utf-8'
        ),
    ]
)
log = logging.getLogger('adverse_selection_study')

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE       = 0.25      # ES minimum price increment
TICK_VALUE      = 12.50     # Dollar value per tick per contract
BARS_PER_SEC    = 10        # 100ms bars = 10 per second
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)      # Round-trip commission (AMP + Rithmic + CME)
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24 ticks
HALF_TICK       = TICK_SIZE / 2   # 0.125 -- corrected half-spread

# Adverse selection detection window (bars after fill)
ADVERSE_WINDOW_BARS = 10    # 1 second

# MBO-informed queue parameters (from empirical data analysis)
# Displayed queue at best bid: median 2-3 contracts
# Effective queue with icebergs: ~20 contracts
# But iceberg refills get new timestamps, so we jump ahead of them.
# Realistic queue position for a resting order: 3-5 contracts ahead of us.
MBO_QUEUE_POSITION_MEAN   = 3.5   # contracts ahead (empirical: 3-5)
MBO_QUEUE_POSITION_MIN    = 1     # best case (front of queue)
MBO_QUEUE_POSITION_MAX    = 8     # worst case displayed depth

# Poisson model parameters (from combined_strategy calibration)
# Trade arrival rate at best price level: ~2 contracts/second during RTH
POISSON_FILL_RATE_PER_SEC = 2.0   # contracts/second at best level
POISSON_QUEUE_DEPTH       = 20    # effective queue depth (with icebergs)


# ============================================================================
# DISCORD NOTIFIER (best-effort)
# ============================================================================
def send_discord(msg: str):
    """Log a Discord-destined message. MCP layer handles actual delivery."""
    log.info(f"[DISCORD] {msg}")


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


_last_log = [0.0]
def progress(msg, interval=30.0):
    if time.time() - _last_log[0] >= interval:
        log.info(msg)
        _last_log[0] = time.time()


# ============================================================================
# DATA LOADING
# ============================================================================
def load_data(fast: bool = False) -> Optional[dict]:
    """
    Load feature cache via MBOAlphaScanner.
    Returns scanner object with features, mid_prices, day_boundaries, etc.
    Same pattern as run_corrected_limit_study.py.
    """
    log.info("Loading feature cache...")
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    stats = load_feature_cache(scanner)
    if stats is None:
        log.info("  No feature cache found -- computing from scratch...")
        stats = scanner.load_from_cache()

    N = len(scanner.mid_prices)
    n_days = len(scanner.day_boundaries) - 1
    log.info(f"  Loaded: {N:,} bars, {n_days} days, {len(scanner.feature_names)} features")

    # Corrected bid/ask (cached spread is buggy -- see run_corrected_limit_study.py)
    mid_prices = scanner.mid_prices.copy().astype(np.float64)
    best_bid   = mid_prices - HALF_TICK   # always mid - 0.125
    best_ask   = mid_prices + HALF_TICK   # always mid + 0.125

    log.info(f"  Mid price mean: ${np.nanmean(mid_prices):.2f}")
    log.info(f"  Corrected spread: 1 tick always (bid=mid-0.125, ask=mid+0.125)")

    # Extract queue depth proxy features
    fn_idx = {n: i for i, n in enumerate(scanner.feature_names)}

    bid_L1_orders = None
    ask_L1_orders = None
    if 'bid_L1_orders' in fn_idx:
        bid_L1_orders = scanner.features[:, fn_idx['bid_L1_orders']].astype(np.float64)
        log.info(f"  bid_L1_orders: mean={np.nanmean(bid_L1_orders):.2f}  "
                 f"median={np.nanmedian(bid_L1_orders):.2f}  "
                 f"p75={np.nanpercentile(bid_L1_orders, 75):.2f}")
    else:
        log.warning("  bid_L1_orders not found in features -- using constant queue depth")
        bid_L1_orders = np.full(N, MBO_QUEUE_POSITION_MEAN)

    if 'ask_L1_orders' in fn_idx:
        ask_L1_orders = scanner.features[:, fn_idx['ask_L1_orders']].astype(np.float64)
    else:
        log.warning("  ask_L1_orders not found in features -- using constant queue depth")
        ask_L1_orders = np.full(N, MBO_QUEUE_POSITION_MEAN)

    # Trade flow features for volume-based fill model
    # Use aggr_buy_count / aggr_sell_count as proxy for contracts traded per bar
    aggr_buy_vol  = None
    aggr_sell_vol = None
    if 'aggr_buy_count' in fn_idx:
        aggr_buy_vol  = scanner.features[:, fn_idx['aggr_buy_count']].astype(np.float64)
        aggr_sell_vol = scanner.features[:, fn_idx['aggr_sell_count']].astype(np.float64)
        log.info(f"  aggr_buy_count: mean={np.nanmean(aggr_buy_vol):.2f}  "
                 f"aggr_sell_count: mean={np.nanmean(aggr_sell_vol):.2f}")
    elif 'buy_volume' in fn_idx:
        aggr_buy_vol  = scanner.features[:, fn_idx['buy_volume']].astype(np.float64)
        aggr_sell_vol = scanner.features[:, fn_idx['sell_volume']].astype(np.float64)
    else:
        # Fallback: estimate from tick_count with ~50% buy/sell split
        if 'tick_count' in fn_idx:
            tick_count = scanner.features[:, fn_idx['tick_count']].astype(np.float64)
            aggr_buy_vol  = tick_count * 0.5
            aggr_sell_vol = tick_count * 0.5
        else:
            # Last resort: constant fill rate of 2 contracts/bar
            aggr_buy_vol  = np.full(N, POISSON_FILL_RATE_PER_SEC / BARS_PER_SEC)
            aggr_sell_vol = np.full(N, POISSON_FILL_RATE_PER_SEC / BARS_PER_SEC)
        log.warning("  Using estimated trade volume -- Tier 2/3 fill model approximate")

    return {
        'scanner': scanner,
        'mid_prices': mid_prices,
        'best_bid': best_bid,
        'best_ask': best_ask,
        'bid_L1_orders': bid_L1_orders,
        'ask_L1_orders': ask_L1_orders,
        'aggr_buy_vol': aggr_buy_vol,
        'aggr_sell_vol': aggr_sell_vol,
        'N': N,
        'n_days': n_days,
        'day_boundaries': scanner.day_boundaries,
        'feature_names': scanner.feature_names,
    }


# ============================================================================
# DIRECTION MODEL (walk-forward LightGBM, same as corrected_limit_study)
# ============================================================================
def train_direction_model(scanner, target, min_train_days=3, fast=False):
    """
    Walk-forward LightGBM on ret_3s.
    Parameters identical to run_corrected_limit_study.py.
    Returns: (predictions, pred_indices, result_dict)
    """
    log.info("Training walk-forward direction model (target: ret_3s)...")
    feature_names = scanner.feature_names
    keep_mask = np.array([fn not in EXCLUDE_FEATURES_DIRECTION for fn in feature_names])
    features_use = scanner.features[:, keep_mask]
    feature_names_use = [fn for fn in feature_names if fn not in EXCLUDE_FEATURES_DIRECTION]

    n_est = 300 if fast else 500
    params = {
        'n_estimators': n_est,
        'max_depth': 6,
        'learning_rate': 0.03,
        'subsample': 0.8,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'min_child_samples': 100,
        'verbose': -1,
        'n_jobs': -1,
    }

    result = walk_forward_evaluate(
        features=features_use,
        target=target,
        day_boundaries=scanner.day_boundaries,
        feature_names=feature_names_use,
        model_type='lgbm',
        min_train_days=min_train_days,
        hour_of_day=scanner.hour_of_day,
        lgbm_params=params,
    )

    if 'error' in result:
        log.error(f"Walk-forward failed: {result['error']}")
        return np.array([]), np.array([], dtype=np.int64), result

    log.info(f"  Direction model: IC={result['ic']:.4f}  ICIR={result['icir']:.2f}  "
             f"t-stat={result['tstat']:.2f}  n_preds={result.get('n_preds', 0):,}")
    return result['predictions'], result['pred_indices'], result


# ============================================================================
# FILL MODEL IMPLEMENTATIONS
# ============================================================================

def _build_pred_signal_array(N, predictions, pred_indices):
    """Map walk-forward predictions onto full time axis."""
    pred_signal = np.full(N, np.nan)
    if len(predictions) > 0 and len(pred_indices) > 0:
        ok = (pred_indices >= 0) & (pred_indices < N)
        pred_signal[pred_indices[ok]] = predictions[ok]
    return pred_signal


def _compute_signal_thresholds(predictions, signal_quantile):
    """Compute upper/lower signal thresholds from valid predictions."""
    valid_p = predictions[np.isfinite(predictions)] if len(predictions) > 0 else np.array([])
    if len(valid_p) == 0:
        return None, None
    thresh_pos = float(np.percentile(valid_p, signal_quantile * 100))
    thresh_neg = float(np.percentile(valid_p, (1 - signal_quantile) * 100))
    return thresh_pos, thresh_neg


def tier1_fill(mid_future_window: np.ndarray, direction: int,
               entry_lim: float) -> bool:
    """
    TIER 1 (BEST CASE): Price touches our limit level = 100% fill.
    Identical to corrected_limit_study fill detection.

    Returns True if filled.
    """
    if direction == 1:  # BUY limit at best_bid
        return bool(np.any(mid_future_window <= entry_lim))
    else:               # SELL limit at best_ask
        return bool(np.any(mid_future_window >= entry_lim))


def tier2_fill_check(mid_future_window: np.ndarray,
                     vol_future_window: np.ndarray,
                     direction: int,
                     entry_lim: float,
                     queue_ahead: float) -> Tuple[bool, int]:
    """
    TIER 2 (MBO-INFORMED): Price touches level AND cumulative volume
    at that price level exceeds queue_ahead within the fill window.

    Logic:
      1. Find first bar where price reaches our limit level.
      2. From that bar forward, accumulate volume traded at/through that level.
      3. Fill if accumulated volume >= queue_ahead.

    Returns: (filled, fill_bar_offset_from_window_start)
    queue_ahead: number of contracts we must wait for before our order fills.
    """
    queue_ahead = max(1.0, queue_ahead)
    n = len(mid_future_window)

    if direction == 1:  # BUY limit at best_bid
        # Price must drop to our level
        at_level = mid_future_window <= entry_lim
    else:               # SELL limit at best_ask
        # Price must rise to our level
        at_level = mid_future_window >= entry_lim

    # Find first touch
    touch_bars = np.where(at_level)[0]
    if len(touch_bars) == 0:
        return False, -1

    first_touch = touch_bars[0]

    # Accumulate volume from first touch forward
    cumvol = 0.0
    for k in range(first_touch, n):
        if not at_level[k]:
            # Price moved away from our level -- volume stops accumulating
            # (price reverted before we could fill from our queue position)
            # Reset -- if price comes back, queue may reset too
            # Conservative: treat as no fill for this touch sequence
            cumvol = 0.0
            # Check if price returns
            continue
        v = vol_future_window[k]
        if np.isfinite(v) and v > 0:
            cumvol += v
        if cumvol >= queue_ahead:
            return True, k

    return False, -1


def tier4_poisson_fill_prob(hold_sec: float, queue_depth: float,
                            fill_rate_per_sec: float) -> float:
    """
    TIER 4 (PESSIMISTIC / POISSON): Probability of being filled within hold_sec.

    Model: contracts arrive at best level at Poisson rate `fill_rate_per_sec`.
    We need `queue_depth` contracts to trade through before our order fills.
    P(fill) = P(Poisson(lambda * T) >= queue_depth)
    where lambda = fill_rate_per_sec, T = hold_sec.

    This replicates the combined_strategy Poisson model.
    """
    from scipy.stats import poisson
    lam = fill_rate_per_sec * hold_sec
    # P(fill) = 1 - P(fewer than queue_depth contracts fill)
    # = 1 - CDF(queue_depth - 1)
    prob = 1.0 - poisson.cdf(int(queue_depth) - 1, lam)
    return float(prob)


# ============================================================================
# CORE SIMULATION ENGINE
# ============================================================================

def simulate_tier(
    tier_name: str,
    mid_prices: np.ndarray,
    best_bid: np.ndarray,
    best_ask: np.ndarray,
    bid_L1_orders: np.ndarray,
    ask_L1_orders: np.ndarray,
    aggr_buy_vol: np.ndarray,
    aggr_sell_vol: np.ndarray,
    day_boundaries: List[int],
    predictions: np.ndarray,
    pred_indices: np.ndarray,
    fill_horizon_bars: int = 50,    # 5 seconds
    hold_horizon_bars: int = 100,   # 10 seconds
    signal_quantile: float = 0.70,
    use_limit_exit: bool = False,
    # Tier-specific params
    tier: int = 1,
    poisson_fill_rate: float = POISSON_FILL_RATE_PER_SEC,
    poisson_queue_depth: float = POISSON_QUEUE_DEPTH,
    adverse_window_bars: int = ADVERSE_WINDOW_BARS,
) -> dict:
    """
    Universal simulation engine supporting all 4 fill tiers.

    For each signal bar:
      1. Post limit order at best_bid (long) or best_ask (short)
      2. Check fill according to tier logic
      3. Hold for hold_horizon_bars (or until limit exit fills)
      4. Compute PnL including adverse selection tracking (Tier 3+)

    Returns comprehensive metrics dict.
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    pred_signal = _build_pred_signal_array(N, predictions, pred_indices)
    thresh_pos, thresh_neg = _compute_signal_thresholds(predictions, signal_quantile)
    if thresh_pos is None:
        return {'error': 'No valid predictions', 'tier': tier_name}

    trades = []
    n_posted = 0
    n_not_filled = 0
    n_adverse = 0      # Tier 3: fills with immediate adverse move
    n_favorable = 0    # Tier 3: fills with favorable or neutral move

    for d in range(n_days):
        ds, de = day_boundaries[d], day_boundaries[d + 1]
        day_len = de - ds
        min_needed = fill_horizon_bars + hold_horizon_bars + adverse_window_bars + 5
        if day_len < min_needed:
            continue

        m   = mid_prices[ds:de]
        bb  = best_bid[ds:de]
        ba  = best_ask[ds:de]
        ps  = pred_signal[ds:de]
        bL1 = bid_L1_orders[ds:de]
        aL1 = ask_L1_orders[ds:de]
        abv = aggr_buy_vol[ds:de]
        asv = aggr_sell_vol[ds:de]
        L   = len(m)

        max_start = L - fill_horizon_bars - hold_horizon_bars - adverse_window_bars - 3
        last_exit = -1

        for i in range(max_start):
            if i <= last_exit:
                continue
            sig = ps[i]
            if not np.isfinite(sig):
                continue

            # Signal filter
            if sig > thresh_pos:
                direction = 1
            elif sig < thresh_neg:
                direction = -1
            else:
                continue

            if not (np.isfinite(bb[i]) and np.isfinite(ba[i])):
                continue

            n_posted += 1

            entry_lim = bb[i] if direction == 1 else ba[i]

            # Future price window for fill detection
            fill_end = min(i + fill_horizon_bars + 1, L)
            mid_window = m[i + 1:fill_end]
            if len(mid_window) == 0:
                n_not_filled += 1
                continue

            # ----------- FILL LOGIC BY TIER -----------

            filled = False
            fill_bar_in_window = -1  # offset from i+1

            if tier == 1:
                # BEST CASE: price touch = fill
                filled = tier1_fill(mid_window, direction, entry_lim)
                if filled:
                    # Find first touch bar
                    if direction == 1:
                        touch_idx = np.where(mid_window <= entry_lim)[0]
                    else:
                        touch_idx = np.where(mid_window >= entry_lim)[0]
                    fill_bar_in_window = int(touch_idx[0]) if len(touch_idx) > 0 else 0

            elif tier == 2 or tier == 3:
                # MBO-INFORMED: need sufficient volume to trade through queue
                queue_ahead = bL1[i] if direction == 1 else aL1[i]
                queue_ahead = np.clip(
                    queue_ahead if np.isfinite(queue_ahead) else MBO_QUEUE_POSITION_MEAN,
                    MBO_QUEUE_POSITION_MIN, MBO_QUEUE_POSITION_MAX
                )
                vol_window = abv[i + 1:fill_end] if direction == 1 else asv[i + 1:fill_end]
                # Zero/NaN volume means no contracts traded at that bar
                vol_window = np.where(np.isfinite(vol_window), vol_window, 0.0)

                filled, fill_bar_in_window = tier2_fill_check(
                    mid_future_window=mid_window,
                    vol_future_window=vol_window,
                    direction=direction,
                    entry_lim=entry_lim,
                    queue_ahead=queue_ahead,
                )

            elif tier == 4:
                # POISSON: probabilistic fill based on trade arrival rate
                hold_sec = fill_horizon_bars / BARS_PER_SEC
                queue_depth_here = bL1[i] if direction == 1 else aL1[i]
                queue_depth_here = np.clip(
                    queue_depth_here if np.isfinite(queue_depth_here) else poisson_queue_depth,
                    1, poisson_queue_depth
                )
                p_fill = tier4_poisson_fill_prob(hold_sec, queue_depth_here, poisson_fill_rate)
                # Sample stochastically (deterministic alternative: threshold)
                # Use deterministic threshold for reproducibility
                filled = (p_fill >= 0.5)
                if filled:
                    # Approximate fill bar: midpoint of fill window
                    fill_bar_in_window = fill_horizon_bars // 2

            if not filled:
                n_not_filled += 1
                continue

            # Actual fill bar in day-relative indexing
            fill_bar = i + 1 + fill_bar_in_window
            fill_bar = min(fill_bar, L - hold_horizon_bars - adverse_window_bars - 2)
            if fill_bar >= L:
                n_not_filled += 1
                continue

            fill_mid = m[fill_bar]
            if not np.isfinite(fill_mid):
                n_not_filled += 1
                continue

            # ----------- ADVERSE SELECTION TRACKING (Tier 3) -----------
            is_adverse = False
            adverse_mid_move_ticks = 0.0

            if tier >= 3:
                # Check mid-price movement within 1s after fill
                adv_end = min(fill_bar + adverse_window_bars + 1, L)
                post_fill_mids = m[fill_bar:adv_end]
                if len(post_fill_mids) > 1:
                    # Adverse = price moves >= 1 tick against our direction
                    if direction == 1:
                        # Long: adverse if price drops
                        worst_move = float(np.nanmin(post_fill_mids) - fill_mid)
                    else:
                        # Short: adverse if price rises
                        worst_move = float(fill_mid - np.nanmax(post_fill_mids))
                    adverse_mid_move_ticks = worst_move / TICK_SIZE
                    # Adverse if worst move < -1.0 tick within 1s
                    is_adverse = adverse_mid_move_ticks <= -1.0

                if is_adverse:
                    n_adverse += 1
                else:
                    n_favorable += 1

            # ----------- EXIT LOGIC -----------
            exit_bar = min(fill_bar + hold_horizon_bars, L - 1)

            if use_limit_exit:
                # Limit exit: post limit at favorable price
                # Approximate: exit at mid + half_tick (favorable direction)
                exit_mid = m[exit_bar]
                exit_edge_ticks = 0.5     # earn half-spread on limit exit
                exit_cost_ticks = 0.0
            else:
                # Market exit: cross half-spread
                exit_mid = m[exit_bar]
                exit_edge_ticks = 0.0
                exit_cost_ticks = 0.5     # pay half-spread on market exit

            # ----------- PnL COMPUTATION -----------
            entry_edge_ticks = 0.5   # passive fill earns half-spread
            dir_pnl_ticks = (exit_mid - fill_mid) / TICK_SIZE * direction
            net_ticks = (
                entry_edge_ticks
                + dir_pnl_ticks
                - exit_cost_ticks
                + exit_edge_ticks
                - COMMISSION_TICKS
            )
            net_dollars = net_ticks * TICK_VALUE

            trades.append({
                'direction': int(direction),
                'signal': float(sig),
                'fill_bar': int(fill_bar),
                'exit_bar': int(exit_bar),
                'fill_offset_bars': int(fill_bar - i),
                'bars_held': int(exit_bar - fill_bar),
                'fill_mid': float(fill_mid),
                'exit_mid': float(exit_mid),
                'entry_edge_ticks': float(entry_edge_ticks),
                'dir_pnl_ticks': float(dir_pnl_ticks),
                'exit_cost_ticks': float(exit_cost_ticks),
                'exit_edge_ticks': float(exit_edge_ticks),
                'commission_ticks': float(COMMISSION_TICKS),
                'net_ticks': float(net_ticks),
                'net_dollars': float(net_dollars),
                'is_adverse': bool(is_adverse),
                'adverse_move_ticks': float(adverse_mid_move_ticks),
                'day': int(d),
            })

            last_exit = exit_bar

        progress(f"  [{tier_name}] day {d+1}/{n_days} trades={len(trades)} "
                 f"posted={n_posted} filled={len(trades)}", interval=20.0)

    log.info(f"[{tier_name}] Simulation complete: "
             f"posted={n_posted}, filled={len(trades)}, not_filled={n_not_filled}")

    # ----------- AGGREGATE METRICS -----------
    if not trades:
        return {
            'error': 'No trades generated',
            'tier': tier_name,
            'n_posted': n_posted,
            'n_not_filled': n_not_filled,
            'fill_rate': 0.0,
        }

    pnl = np.array([t['net_ticks'] for t in trades])
    pnl_dollars = pnl * TICK_VALUE
    dir_pnl = np.array([t['dir_pnl_ticks'] for t in trades])

    # Adverse vs favorable split (Tier 3)
    adverse_mask = np.array([t['is_adverse'] for t in trades])
    favorable_mask = ~adverse_mask

    pnl_adverse   = pnl[adverse_mask]   if adverse_mask.any()   else np.array([])
    pnl_favorable = pnl[favorable_mask] if favorable_mask.any() else np.array([])

    # Sharpe (annualized, trade-level)
    avg_hold_sec = np.mean([t['bars_held'] for t in trades]) / BARS_PER_SEC
    if avg_hold_sec > 0 and np.std(pnl) > 0:
        trades_per_year = 252 * 6.5 * 3600 / max(avg_hold_sec, 1.0)
        sharpe = float(np.mean(pnl) / np.std(pnl) * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    # Daily Sharpe
    daily_pnl = {}
    for t in trades:
        d = t['day']
        daily_pnl[d] = daily_pnl.get(d, 0.0) + t['net_dollars']
    daily_pnl_arr = np.array(list(daily_pnl.values()))
    if len(daily_pnl_arr) > 2 and np.std(daily_pnl_arr) > 0:
        daily_sharpe = float(np.mean(daily_pnl_arr) / np.std(daily_pnl_arr) * np.sqrt(252))
    else:
        daily_sharpe = 0.0

    # Profit factor
    wins   = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = (
        float(wins.sum() / -losses.sum())
        if len(losses) > 0 and -losses.sum() > 0
        else 999.0
    )

    # Max drawdown
    cum_pnl = np.cumsum(pnl_dollars)
    running_max = np.maximum.accumulate(cum_pnl)
    max_drawdown = float((running_max - cum_pnl).max()) if len(cum_pnl) > 0 else 0.0

    fill_rate = len(trades) / n_posted if n_posted > 0 else 0.0
    adverse_rate = float(adverse_mask.mean()) if len(trades) > 0 else 0.0

    result = {
        'tier': tier_name,
        'tier_num': tier,
        'config': {
            'fill_horizon_sec': fill_horizon_bars / BARS_PER_SEC,
            'hold_horizon_sec': hold_horizon_bars / BARS_PER_SEC,
            'signal_quantile': signal_quantile,
            'use_limit_exit': use_limit_exit,
            'exit_type': 'limit_exit' if use_limit_exit else 'market_exit',
        },
        'fill_stats': {
            'n_posted': int(n_posted),
            'n_filled': int(len(trades)),
            'n_not_filled': int(n_not_filled),
            'fill_rate': float(fill_rate),
        },
        'adverse_selection': {
            'n_adverse': int(n_adverse),
            'n_favorable': int(n_favorable),
            'adverse_rate': float(adverse_rate),
            'mean_pnl_adverse_ticks': float(np.mean(pnl_adverse)) if len(pnl_adverse) > 0 else None,
            'mean_pnl_favorable_ticks': float(np.mean(pnl_favorable)) if len(pnl_favorable) > 0 else None,
            'mean_adverse_move_ticks': float(np.mean([t['adverse_move_ticks'] for t in trades])),
        },
        'performance': {
            'mean_pnl_ticks': float(np.mean(pnl)),
            'std_pnl_ticks': float(np.std(pnl)),
            'mean_pnl_dollars': float(np.mean(pnl_dollars)),
            'total_pnl_dollars': float(np.sum(pnl_dollars)),
            'win_rate': float((pnl > 0).mean()),
            'sharpe_trade_level': float(sharpe),
            'sharpe_daily': float(daily_sharpe),
            'profit_factor': float(profit_factor),
            'max_drawdown_dollars': float(max_drawdown),
            'trades_per_day': float(len(trades) / max(n_days, 1)),
            'mean_dir_pnl_ticks': float(np.mean(dir_pnl)),
            'mean_fill_offset_bars': float(np.mean([t['fill_offset_bars'] for t in trades])),
            'mean_hold_bars': float(np.mean([t['bars_held'] for t in trades])),
        },
        'n_days': int(n_days),
    }

    log.info(
        f"  [{tier_name}] fill={fill_rate:.1%}  "
        f"adv_rate={adverse_rate:.1%}  "
        f"mean_pnl={result['performance']['mean_pnl_ticks']:>+.4f}t "
        f"(${result['performance']['mean_pnl_dollars']:>+.2f})  "
        f"win={result['performance']['win_rate']:.1%}  "
        f"Sharpe(trade)={sharpe:.2f}  Sharpe(daily)={daily_sharpe:.2f}  "
        f"total=${result['performance']['total_pnl_dollars']:>+.2f}"
    )

    return result


# ============================================================================
# LIMIT EXIT FILL PROBABILITY ANALYSIS
# ============================================================================

def analyze_limit_exit_fill_rate(
    mid_prices: np.ndarray,
    best_bid: np.ndarray,
    best_ask: np.ndarray,
    bid_L1_orders: np.ndarray,
    ask_L1_orders: np.ndarray,
    aggr_buy_vol: np.ndarray,
    aggr_sell_vol: np.ndarray,
    day_boundaries: List[int],
    hold_horizons_bars: List[int] = None,
) -> dict:
    """
    Analyze the fill probability for LIMIT EXIT orders.

    After entering a long at best_bid, we post a limit sell at best_ask
    (our target) and check if it fills within the hold period.

    Measures:
      - Tier 1 limit exit fill rate: does price touch target within hold_sec?
      - Tier 2 limit exit fill rate: does price touch AND sufficient volume?

    This quantifies the difference between assuming 100% limit exit fill
    (corrected_limit_study) vs realistic limit exit fill probability.
    """
    log.info("Analyzing limit exit fill probability...")

    if hold_horizons_bars is None:
        hold_horizons_bars = [50, 100, 300]  # 5s, 10s, 30s

    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    results = {}

    for hold_bars in hold_horizons_bars:
        hold_sec = hold_bars / BARS_PER_SEC
        tier1_fills = 0
        tier2_fills = 0
        total = 0

        for d in range(n_days):
            ds, de = day_boundaries[d], day_boundaries[d + 1]
            if de - ds < hold_bars + 5:
                continue

            m  = mid_prices[ds:de]
            bb = best_bid[ds:de]
            ba = best_ask[ds:de]
            bL1 = bid_L1_orders[ds:de]
            asv = aggr_sell_vol[ds:de]
            L  = len(m)

            # Sample every 10 bars to estimate fill rates
            for i in range(0, L - hold_bars - 2, 10):
                if not np.isfinite(bb[i]):
                    continue
                total += 1

                # Long entry at best_bid, limit exit at best_ask
                fill_end = min(i + hold_bars + 1, L)
                mid_window = m[i + 1:fill_end]

                # Tier 1: does mid reach best_ask within hold window?
                target = ba[i]
                if np.any(mid_window >= target):
                    tier1_fills += 1
                    # Tier 2: also need volume
                    vol_window = asv[i + 1:fill_end]
                    vol_window = np.where(np.isfinite(vol_window), vol_window, 0.0)
                    queue = bL1[i] if np.isfinite(bL1[i]) else MBO_QUEUE_POSITION_MEAN
                    queue = max(1.0, min(queue, MBO_QUEUE_POSITION_MAX))
                    _, fb = tier2_fill_check(
                        mid_future_window=mid_window,
                        vol_future_window=vol_window,
                        direction=-1,  # selling = price goes up
                        entry_lim=target,
                        queue_ahead=queue,
                    )
                    if fb >= 0:
                        tier2_fills += 1

        tier1_rate = tier1_fills / total if total > 0 else 0.0
        tier2_rate = tier2_fills / total if total > 0 else 0.0

        results[f'hold_{hold_sec:.0f}s'] = {
            'hold_sec': hold_sec,
            'total_samples': total,
            'tier1_fill_rate': tier1_rate,
            'tier2_fill_rate': tier2_rate,
            'tier1_vs_tier2_ratio': tier1_rate / tier2_rate if tier2_rate > 0 else 0.0,
        }
        log.info(f"  Exit fill rate (hold={hold_sec:.0f}s): "
                 f"Tier1={tier1_rate:.1%}  Tier2={tier2_rate:.1%}  "
                 f"ratio={tier1_rate/tier2_rate:.1f}x" if tier2_rate > 0 else
                 f"  Exit fill rate (hold={hold_sec:.0f}s): "
                 f"Tier1={tier1_rate:.1%}  Tier2=0%")

    return results


# ============================================================================
# COMPREHENSIVE TIER COMPARISON
# ============================================================================

def run_all_tiers(data: dict, predictions: np.ndarray, pred_indices: np.ndarray,
                  dir_result: dict,
                  hold_sec: float = 10.0, signal_quantile: float = 0.70,
                  fast: bool = False) -> dict:
    """
    Run all 4 fill model tiers with both market and limit exit.
    Provides the bridge between optimistic and pessimistic fill models.

    Returns dict of all results keyed by tier name.
    """
    mid_prices   = data['mid_prices']
    best_bid     = data['best_bid']
    best_ask     = data['best_ask']
    bid_L1       = data['bid_L1_orders']
    ask_L1       = data['ask_L1_orders']
    buy_vol      = data['aggr_buy_vol']
    sell_vol     = data['aggr_sell_vol']
    day_bounds   = data['day_boundaries']
    N            = data['N']
    n_days       = data['n_days']

    fill_horizon_bars = 5 * BARS_PER_SEC   # 5s fill window
    hold_horizon_bars = int(hold_sec * BARS_PER_SEC)

    results = {}
    total_start = time.time()

    # ----------------------------------------------------------------
    # TIER 1: BEST CASE (price touch = 100% fill)
    # ----------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("TIER 1: BEST CASE (price touch = 100% fill)")
    log.info("Replicates corrected_limit_study fill model")
    log.info("=" * 60)

    for exit_type, use_limit_exit in [('mkt_exit', False), ('lmt_exit', True)]:
        label = f'tier1_{exit_type}'
        r = simulate_tier(
            tier_name=label, tier=1,
            mid_prices=mid_prices, best_bid=best_bid, best_ask=best_ask,
            bid_L1_orders=bid_L1, ask_L1_orders=ask_L1,
            aggr_buy_vol=buy_vol, aggr_sell_vol=sell_vol,
            day_boundaries=day_bounds,
            predictions=predictions, pred_indices=pred_indices,
            fill_horizon_bars=fill_horizon_bars,
            hold_horizon_bars=hold_horizon_bars,
            signal_quantile=signal_quantile,
            use_limit_exit=use_limit_exit,
        )
        results[label] = r
        gc.collect()

    elapsed = time.time() - total_start
    send_discord(f"Tier 1 complete ({elapsed:.0f}s). "
                 f"Tier1_mkt: fill={results.get('tier1_mkt_exit', {}).get('fill_stats', {}).get('fill_rate', 0):.1%} "
                 f"mean_pnl={results.get('tier1_mkt_exit', {}).get('performance', {}).get('mean_pnl_ticks', 0):+.4f}t")

    # ----------------------------------------------------------------
    # TIER 2: MBO-INFORMED (volume-gated fill)
    # ----------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("TIER 2: MBO-INFORMED (queue-depth + volume-gated fill)")
    log.info("Uses bid_L1_orders as queue depth proxy from MBO data")
    log.info("=" * 60)

    for exit_type, use_limit_exit in [('mkt_exit', False), ('lmt_exit', True)]:
        label = f'tier2_{exit_type}'
        r = simulate_tier(
            tier_name=label, tier=2,
            mid_prices=mid_prices, best_bid=best_bid, best_ask=best_ask,
            bid_L1_orders=bid_L1, ask_L1_orders=ask_L1,
            aggr_buy_vol=buy_vol, aggr_sell_vol=sell_vol,
            day_boundaries=day_bounds,
            predictions=predictions, pred_indices=pred_indices,
            fill_horizon_bars=fill_horizon_bars,
            hold_horizon_bars=hold_horizon_bars,
            signal_quantile=signal_quantile,
            use_limit_exit=use_limit_exit,
        )
        results[label] = r
        gc.collect()

    elapsed = time.time() - total_start
    send_discord(f"Tier 2 complete ({elapsed:.0f}s). "
                 f"Tier2_mkt: fill={results.get('tier2_mkt_exit', {}).get('fill_stats', {}).get('fill_rate', 0):.1%} "
                 f"mean_pnl={results.get('tier2_mkt_exit', {}).get('performance', {}).get('mean_pnl_ticks', 0):+.4f}t")

    # ----------------------------------------------------------------
    # TIER 3: ADVERSE-SELECTION ADJUSTED
    # ----------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("TIER 3: ADVERSE-SELECTION ADJUSTED")
    log.info("Same as Tier 2 + tracks 1s post-fill mid movement")
    log.info("Adverse = mid moves >= 1 tick against us within 1s of fill")
    log.info("=" * 60)

    for exit_type, use_limit_exit in [('mkt_exit', False), ('lmt_exit', True)]:
        label = f'tier3_{exit_type}'
        r = simulate_tier(
            tier_name=label, tier=3,
            mid_prices=mid_prices, best_bid=best_bid, best_ask=best_ask,
            bid_L1_orders=bid_L1, ask_L1_orders=ask_L1,
            aggr_buy_vol=buy_vol, aggr_sell_vol=sell_vol,
            day_boundaries=day_bounds,
            predictions=predictions, pred_indices=pred_indices,
            fill_horizon_bars=fill_horizon_bars,
            hold_horizon_bars=hold_horizon_bars,
            signal_quantile=signal_quantile,
            use_limit_exit=use_limit_exit,
            adverse_window_bars=ADVERSE_WINDOW_BARS,
        )
        results[label] = r
        gc.collect()

    elapsed = time.time() - total_start
    t3 = results.get('tier3_mkt_exit', {})
    adv_rate = t3.get('adverse_selection', {}).get('adverse_rate', 0)
    send_discord(f"Tier 3 complete ({elapsed:.0f}s). "
                 f"Adverse selection rate: {adv_rate:.1%} of fills. "
                 f"PnL adverse={t3.get('adverse_selection', {}).get('mean_pnl_adverse_ticks', 0):+.4f}t "
                 f"vs favorable={t3.get('adverse_selection', {}).get('mean_pnl_favorable_ticks', 0):+.4f}t")

    # ----------------------------------------------------------------
    # TIER 4: PESSIMISTIC POISSON (like combined_strategy)
    # ----------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("TIER 4: PESSIMISTIC / POISSON (replicates combined_strategy)")
    log.info(f"Poisson fill rate: {POISSON_FILL_RATE_PER_SEC} contracts/sec")
    log.info(f"Queue depth: {POISSON_QUEUE_DEPTH} contracts")
    log.info("=" * 60)

    for exit_type, use_limit_exit in [('mkt_exit', False), ('lmt_exit', True)]:
        label = f'tier4_{exit_type}'
        r = simulate_tier(
            tier_name=label, tier=4,
            mid_prices=mid_prices, best_bid=best_bid, best_ask=best_ask,
            bid_L1_orders=bid_L1, ask_L1_orders=ask_L1,
            aggr_buy_vol=buy_vol, aggr_sell_vol=sell_vol,
            day_boundaries=day_bounds,
            predictions=predictions, pred_indices=pred_indices,
            fill_horizon_bars=fill_horizon_bars,
            hold_horizon_bars=hold_horizon_bars,
            signal_quantile=signal_quantile,
            use_limit_exit=use_limit_exit,
            poisson_fill_rate=POISSON_FILL_RATE_PER_SEC,
            poisson_queue_depth=POISSON_QUEUE_DEPTH,
        )
        results[label] = r
        gc.collect()

    elapsed = time.time() - total_start
    send_discord(f"Tier 4 complete ({elapsed:.0f}s). All 4 tiers done. Computing exit fill rates...")

    # ----------------------------------------------------------------
    # LIMIT EXIT FILL RATE ANALYSIS
    # ----------------------------------------------------------------
    log.info("\n" + "=" * 60)
    log.info("LIMIT EXIT FILL RATE ANALYSIS")
    log.info("=" * 60)

    exit_fill_results = analyze_limit_exit_fill_rate(
        mid_prices=mid_prices, best_bid=best_bid, best_ask=best_ask,
        bid_L1_orders=bid_L1, ask_L1_orders=ask_L1,
        aggr_buy_vol=buy_vol, aggr_sell_vol=sell_vol,
        day_boundaries=day_bounds,
        hold_horizons_bars=[50, 100, 300] if not fast else [100, 300],
    )
    results['limit_exit_fill_rates'] = exit_fill_results

    return results


# ============================================================================
# SUMMARY REPORT
# ============================================================================

def format_summary(results: dict, dir_result: dict, n_days: int,
                   N: int, hold_sec: float, signal_quantile: float) -> str:
    """Generate human-readable summary report."""
    lines = [
        "",
        "=" * 72,
        "ADVERSE SELECTION STUDY -- 4-TIER FILL MODEL COMPARISON",
        "ES Futures MBO Alpha | Bridging Optimistic vs Pessimistic Results",
        "=" * 72,
        "",
        f"DATASET: {N:,} bars  |  {n_days} days",
        f"CONFIG:  hold={hold_sec:.0f}s  signal_q={signal_quantile:.0%}  "
        f"fill_window=5s  commission=$3.00RT=0.24t",
        "",
        "DIRECTION MODEL (walk-forward LightGBM, ret_3s target):",
        f"  IC={dir_result.get('ic', float('nan')):.4f}  "
        f"ICIR={dir_result.get('icir', float('nan')):.2f}  "
        f"t-stat={dir_result.get('tstat', float('nan')):.2f}  "
        f"n_preds={dir_result.get('n_preds', 0):,}",
        "",
        "FILL MODEL COMPARISON (Market Exit):",
        f"  {'Tier':>6s}  {'Fill%':>6s}  {'Adv%':>6s}  {'mean_t':>8s}  "
        f"{'$mean':>8s}  {'win%':>6s}  {'Sharpe':>7s}  {'total$':>10s}  {'t/day':>6s}",
        "  " + "-" * 70,
    ]

    tier_order = ['tier1_mkt_exit', 'tier2_mkt_exit', 'tier3_mkt_exit', 'tier4_mkt_exit']
    tier_labels = {
        'tier1_mkt_exit': 'T1 Best',
        'tier2_mkt_exit': 'T2 MBO',
        'tier3_mkt_exit': 'T3 AdjSel',
        'tier4_mkt_exit': 'T4 Poisson',
    }

    for key in tier_order:
        r = results.get(key, {})
        if 'error' in r:
            lines.append(f"  {tier_labels.get(key, key):>10s}  ERROR: {r['error']}")
            continue
        fs   = r.get('fill_stats', {})
        adv  = r.get('adverse_selection', {})
        perf = r.get('performance', {})
        lines.append(
            f"  {tier_labels.get(key, key):>10s}  "
            f"{fs.get('fill_rate', 0):>6.1%}  "
            f"{adv.get('adverse_rate', 0):>6.1%}  "
            f"{perf.get('mean_pnl_ticks', 0):>+8.4f}  "
            f"{perf.get('mean_pnl_dollars', 0):>+8.2f}  "
            f"{perf.get('win_rate', 0):>6.1%}  "
            f"{perf.get('sharpe_trade_level', 0):>7.2f}  "
            f"${perf.get('total_pnl_dollars', 0):>+9.2f}  "
            f"{perf.get('trades_per_day', 0):>6.1f}"
        )

    lines += [
        "",
        "FILL MODEL COMPARISON (Limit Exit):",
        f"  {'Tier':>6s}  {'Fill%':>6s}  {'Adv%':>6s}  {'mean_t':>8s}  "
        f"{'$mean':>8s}  {'win%':>6s}  {'Sharpe':>7s}  {'total$':>10s}  {'t/day':>6s}",
        "  " + "-" * 70,
    ]

    tier_order_lmt = ['tier1_lmt_exit', 'tier2_lmt_exit', 'tier3_lmt_exit', 'tier4_lmt_exit']
    tier_labels_lmt = {
        'tier1_lmt_exit': 'T1 Best',
        'tier2_lmt_exit': 'T2 MBO',
        'tier3_lmt_exit': 'T3 AdjSel',
        'tier4_lmt_exit': 'T4 Poisson',
    }

    for key in tier_order_lmt:
        r = results.get(key, {})
        if 'error' in r:
            lines.append(f"  {tier_labels_lmt.get(key, key):>10s}  ERROR: {r['error']}")
            continue
        fs   = r.get('fill_stats', {})
        adv  = r.get('adverse_selection', {})
        perf = r.get('performance', {})
        lines.append(
            f"  {tier_labels_lmt.get(key, key):>10s}  "
            f"{fs.get('fill_rate', 0):>6.1%}  "
            f"{adv.get('adverse_rate', 0):>6.1%}  "
            f"{perf.get('mean_pnl_ticks', 0):>+8.4f}  "
            f"{perf.get('mean_pnl_dollars', 0):>+8.2f}  "
            f"{perf.get('win_rate', 0):>6.1%}  "
            f"{perf.get('sharpe_trade_level', 0):>7.2f}  "
            f"${perf.get('total_pnl_dollars', 0):>+9.2f}  "
            f"{perf.get('trades_per_day', 0):>6.1f}"
        )

    # Adverse selection deep-dive (Tier 3)
    t3 = results.get('tier3_mkt_exit', {})
    adv3 = t3.get('adverse_selection', {})
    perf3 = t3.get('performance', {})
    if t3 and 'error' not in t3:
        lines += [
            "",
            "TIER 3 -- ADVERSE SELECTION DEEP DIVE (mkt exit):",
            f"  Adverse fill rate:          {adv3.get('adverse_rate', 0):.1%}  "
            f"({adv3.get('n_adverse', 0)} / {adv3.get('n_adverse', 0) + adv3.get('n_favorable', 0)} fills)",
            f"  Mean PnL on adverse fills:  "
            f"{adv3.get('mean_pnl_adverse_ticks', 0) or 0:>+.4f}t "
            f"(${(adv3.get('mean_pnl_adverse_ticks', 0) or 0) * TICK_VALUE:>+.2f})",
            f"  Mean PnL on favorable fills:{adv3.get('mean_pnl_favorable_ticks', 0) or 0:>+.4f}t "
            f"(${(adv3.get('mean_pnl_favorable_ticks', 0) or 0) * TICK_VALUE:>+.2f})",
            f"  Mean adverse move (ticks):  {adv3.get('mean_adverse_move_ticks', 0):>+.4f}t",
            f"  INTERPRETATION:",
            f"    Adverse fills = informed flow trading AGAINST us after fill.",
            f"    If adverse_rate > 40%: significant adverse selection problem.",
            f"    If PnL(adverse) << PnL(favorable): adverse selection is costly.",
        ]

    # Limit exit fill rate analysis
    exit_rates = results.get('limit_exit_fill_rates', {})
    if exit_rates:
        lines += ["", "LIMIT EXIT FILL PROBABILITY ANALYSIS:"]
        lines.append(f"  {'Hold':>6s}  {'T1 Fill%':>8s}  {'T2 Fill%':>8s}  {'T1/T2 ratio':>11s}")
        lines.append("  " + "-" * 38)
        for k, v in sorted(exit_rates.items()):
            t1r = v.get('tier1_fill_rate', 0)
            t2r = v.get('tier2_fill_rate', 0)
            ratio = v.get('tier1_vs_tier2_ratio', 0)
            lines.append(f"  {v.get('hold_sec', 0):>5.0f}s  {t1r:>8.1%}  {t2r:>8.1%}  "
                         f"{ratio:>11.1f}x")
        lines.append(f"  NOTE: Tier 1 assumes 100% fill when price touches target.")
        lines.append(f"  NOTE: Tier 2 requires volume threshold -- realistic fill rate.")

    # VERDICT
    t1_mkt = results.get('tier1_mkt_exit', {})
    t2_mkt = results.get('tier2_mkt_exit', {})
    t3_mkt = results.get('tier3_mkt_exit', {})

    t1_sharpe = t1_mkt.get('performance', {}).get('sharpe_trade_level', 0) or 0
    t2_sharpe = t2_mkt.get('performance', {}).get('sharpe_trade_level', 0) or 0
    t3_sharpe = t3_mkt.get('performance', {}).get('sharpe_trade_level', 0) or 0
    t2_pnl    = t2_mkt.get('performance', {}).get('mean_pnl_ticks', 0) or 0
    t3_pnl    = t3_mkt.get('performance', {}).get('mean_pnl_ticks', 0) or 0

    lines += ["", "=" * 72, "VERDICT:"]

    if t2_sharpe > 1.5 and t2_pnl > 0:
        verdict = f"VIABLE WITH MBO-REALISTIC FILLS -- Tier 2 Sharpe={t2_sharpe:.2f}"
        detail  = ("The strategy survives realistic volume-gated fills. "
                   "The gap between Tier 1 (optimistic) and Tier 2 (realistic) "
                   "quantifies queue position risk.")
    elif t2_sharpe > 0.5 and t2_pnl > 0:
        verdict = f"MARGINAL WITH REALISTIC FILLS -- Tier 2 Sharpe={t2_sharpe:.2f}"
        detail  = ("Positive edge survives MBO-informed fills but Sharpe < 1.5. "
                   "Execution improvements (better queue position, co-location) needed.")
    elif t2_pnl > 0 and t2_sharpe > 0:
        verdict = f"WEAK EDGE -- Tier 2 Sharpe={t2_sharpe:.2f}, positive but thin"
        detail  = ("Positive PnL but high variance relative to edge. "
                   "Need Sharpe > 1.0 for reliable live trading.")
    else:
        verdict = f"NO VIABLE EDGE -- Tier 2 Sharpe={t2_sharpe:.2f}"
        detail  = ("With realistic MBO fills, the strategy is not profitable. "
                   "Adverse selection dominates or fill rates are too low.")

    lines += [
        f"  {verdict}",
        f"  {detail}",
        "",
        "  FILL MODEL SPECTRUM:",
        f"    Tier 1 (Best Case):    Sharpe={t1_sharpe:.2f}  -- WHAT corrected_limit_study shows",
        f"    Tier 2 (MBO-Informed): Sharpe={t2_sharpe:.2f}  -- REALISTIC with queue position",
        f"    Tier 3 (Adv-Sel Adj):  Sharpe={t3_sharpe:.2f}  -- REALISTIC + adverse selection cost",
        f"    Tier 4 (Poisson):      Sharpe="
        + str(results.get('tier4_mkt_exit', {}).get('performance', {}).get('sharpe_trade_level', 0) or 0)[:5]
        + f"  -- WHAT combined_strategy shows",
        "",
        "  KEY INSIGHT:",
        "    Tier 1 -> Tier 2 gap: fill rate degradation from queue position",
        "    Tier 2 -> Tier 3 gap: adverse selection cost on actual fills",
        "    Tier 3 -> Tier 4 gap: Poisson vs volume-threshold modeling difference",
        "    The REALISTIC answer is Tier 2-3: between optimistic and pessimistic.",
        "=" * 72,
    ]
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Adverse Selection Study -- 4-Tier Fill Model Comparison'
    )
    parser.add_argument('--fast', action='store_true',
                        help='Faster run: fewer LightGBM trees, fewer configs')
    parser.add_argument('--min-train-days', type=int, default=3,
                        help='Minimum training days for walk-forward (default: 3)')
    parser.add_argument('--hold', type=float, default=10.0,
                        help='Hold period in seconds (default: 10.0)')
    parser.add_argument('--signal-q', type=float, default=0.70,
                        help='Signal percentile threshold (default: 0.70)')
    args = parser.parse_args()

    t_start = time.time()

    log.info("=" * 72)
    log.info("ADVERSE SELECTION STUDY -- 4-TIER FILL MODEL COMPARISON")
    log.info("Bridges corrected_limit_study (optimistic) vs combined_strategy (pessimistic)")
    log.info(f"Mode: {'FAST' if args.fast else 'FULL'}  "
             f"hold={args.hold:.0f}s  signal_q={args.signal_q:.0%}")
    log.info("=" * 72)

    send_discord(f"Starting Adverse Selection Study... "
                 f"hold={args.hold:.0f}s  q={args.signal_q:.0%}  "
                 f"{'FAST' if args.fast else 'FULL'} mode")

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    log.info("\nStep 1: Loading feature cache...")
    data = load_data(fast=args.fast)
    scanner = data['scanner']
    N = data['N']
    n_days = data['n_days']

    send_discord(f"Data loaded: {N:,} bars, {n_days} days. Training direction model...")

    # ------------------------------------------------------------------
    # Step 2: Compute ret_3s target
    # ------------------------------------------------------------------
    log.info("\nStep 2: Computing ret_3s target...")
    targets = compute_return_targets(
        mid_prices=data['mid_prices'],
        day_boundaries=scanner.day_boundaries,
        sample_interval_ms=100,
        horizons_sec={'3s': 3},
        include_flow_target=False,
    )
    ret_3s = targets['ret_3s']

    # ------------------------------------------------------------------
    # Step 3: Train walk-forward direction model
    # ------------------------------------------------------------------
    log.info("\nStep 3: Training walk-forward direction model...")
    predictions, pred_indices, dir_result = train_direction_model(
        scanner=scanner, target=ret_3s,
        min_train_days=args.min_train_days,
        fast=args.fast,
    )
    gc.collect()

    ic = dir_result.get('ic', float('nan'))
    send_discord(f"Direction model trained: IC={ic:.4f}  "
                 f"ICIR={dir_result.get('icir', float('nan')):.2f}  "
                 f"n_preds={dir_result.get('n_preds', 0):,}. Running 4 tiers...")

    # ------------------------------------------------------------------
    # Step 4: Run all 4 fill model tiers
    # ------------------------------------------------------------------
    log.info("\nStep 4: Running 4-tier fill model comparison...")
    results = run_all_tiers(
        data=data,
        predictions=predictions,
        pred_indices=pred_indices,
        dir_result=dir_result,
        hold_sec=args.hold,
        signal_quantile=args.signal_q,
        fast=args.fast,
    )

    # ------------------------------------------------------------------
    # Step 5: Format and print summary
    # ------------------------------------------------------------------
    log.info("\nStep 5: Formatting summary report...")
    summary = format_summary(
        results=results, dir_result=dir_result,
        n_days=n_days, N=N,
        hold_sec=args.hold, signal_quantile=args.signal_q,
    )
    log.info("\n" + summary)
    print("\n" + summary)

    # ------------------------------------------------------------------
    # Step 6: Save results to JSON
    # ------------------------------------------------------------------
    log.info("\nStep 6: Saving results...")
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'adverse_selection_study_{timestamp}.json'

    output = {
        'timestamp': timestamp,
        'mode': 'fast' if args.fast else 'full',
        'config': {
            'hold_sec': args.hold,
            'signal_quantile': args.signal_q,
            'fill_horizon_sec': 5.0,
            'tick_size': TICK_SIZE,
            'tick_value': TICK_VALUE,
            'commission_rt': COMMISSION_RT,
            'commission_ticks': COMMISSION_TICKS,
            'adverse_window_bars': ADVERSE_WINDOW_BARS,
            'adverse_window_sec': ADVERSE_WINDOW_BARS / BARS_PER_SEC,
            'mbo_queue_position_mean': MBO_QUEUE_POSITION_MEAN,
            'mbo_queue_position_range': [MBO_QUEUE_POSITION_MIN, MBO_QUEUE_POSITION_MAX],
            'poisson_fill_rate_per_sec': POISSON_FILL_RATE_PER_SEC,
            'poisson_queue_depth': POISSON_QUEUE_DEPTH,
        },
        'data': {
            'n_bars': int(N),
            'n_days': int(n_days),
            'spread_assumption': '1_tick_corrected (mid +/- 0.125)',
        },
        'direction_model': {
            k: dir_result.get(k)
            for k in ['ic', 'icir', 'tstat', 'n_preds', 'fold_ics', 'fold_con']
        },
        'tier_results': to_safe(results),
        'summary_text': summary,
        'total_elapsed_sec': time.time() - t_start,
    }

    with open(str(out_file), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    log.info(f"Results saved: {out_file}")
    log.info(f"Total elapsed: {(time.time() - t_start) / 60:.1f} min")

    # ------------------------------------------------------------------
    # Step 7: Final Discord summary
    # ------------------------------------------------------------------
    t2_mkt = results.get('tier2_mkt_exit', {})
    t3_mkt = results.get('tier3_mkt_exit', {})

    t2_fill   = t2_mkt.get('fill_stats', {}).get('fill_rate', 0)
    t2_sharpe = t2_mkt.get('performance', {}).get('sharpe_trade_level', 0) or 0
    t2_pnl    = t2_mkt.get('performance', {}).get('mean_pnl_ticks', 0) or 0
    t3_adv    = t3_mkt.get('adverse_selection', {}).get('adverse_rate', 0)
    t3_pnl_adv = t3_mkt.get('adverse_selection', {}).get('mean_pnl_adverse_ticks', 0) or 0
    t3_pnl_fav = t3_mkt.get('adverse_selection', {}).get('mean_pnl_favorable_ticks', 0) or 0

    discord_msg = (
        f"Adverse Selection Study COMPLETE ({(time.time() - t_start)/60:.1f}min)\n\n"
        f"4-TIER FILL MODEL RESULTS (hold={args.hold:.0f}s, q={args.signal_q:.0%}, mkt exit):\n"
        f"  Tier1 (Best Case):   "
        f"fill={results.get('tier1_mkt_exit', {}).get('fill_stats', {}).get('fill_rate', 0):.0%}  "
        f"Sharpe={results.get('tier1_mkt_exit', {}).get('performance', {}).get('sharpe_trade_level', 0) or 0:.1f}  "
        f"mean={results.get('tier1_mkt_exit', {}).get('performance', {}).get('mean_pnl_ticks', 0) or 0:+.4f}t\n"
        f"  Tier2 (MBO Queue):   fill={t2_fill:.0%}  Sharpe={t2_sharpe:.1f}  mean={t2_pnl:+.4f}t\n"
        f"  Tier3 (Adv Select):  fill={t3_mkt.get('fill_stats', {}).get('fill_rate', 0):.0%}  "
        f"adv_rate={t3_adv:.0%}  "
        f"PnL[adv]={t3_pnl_adv:+.4f}t vs [fav]={t3_pnl_fav:+.4f}t\n"
        f"  Tier4 (Poisson):     "
        f"fill={results.get('tier4_mkt_exit', {}).get('fill_stats', {}).get('fill_rate', 0):.0%}  "
        f"Sharpe={results.get('tier4_mkt_exit', {}).get('performance', {}).get('sharpe_trade_level', 0) or 0:.1f}\n\n"
        f"Results: {out_file.name}"
    )
    send_discord(discord_msg)

    return output


if __name__ == '__main__':
    main()
