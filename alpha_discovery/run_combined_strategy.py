"""
Combined Direction + Spread Strategy — Limit Order Simulation

Combines two confirmed alpha signals:
  1. Direction (ret_3s):           IC=0.08-0.11, gross $7.62/trade at 98th percentile
  2. Spread ratio (spread_ratio_5s): IC=0.048, ICIR=1.52, 100% fold consistency

Strategy: post LIMIT orders at best_bid/best_ask when combined signal is strong,
hold for 3-30 seconds, exit. Realistic fill detection + ES commission.

Usage:
    python alpha_discovery/run_combined_strategy.py
    python alpha_discovery/run_combined_strategy.py --fast
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
from scipy.stats import spearmanr
from typing import Dict, List, Optional, Tuple

# ============================================================================
# PATH SETUP
# ============================================================================
ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

from alpha_discovery.mbo_alpha_scan import MBOAlphaScanner, RESULTS_DIR
from alpha_discovery.mbo_features import get_feature_names, TOTAL_FEATURES
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
    format='%(asctime)s %(name)s %(levelname)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            RESULTS_DIR / 'combined_strategy.log', mode='a', encoding='utf-8'
        ),
    ]
)
logger = logging.getLogger("combined_strategy")

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE      = 0.25         # ES futures minimum tick
TICK_VALUE     = 12.50        # $ per tick per contract
BARS_PER_SEC   = 10           # 100ms intervals = 10 bars/sec
COMMISSION_RT  = 4.70         # $4.70 round-trip commission (AMP/Rithmic, HC #52 canonical)
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.376 ticks

# How we compute spread_ratio_5s target inline:
# spread_ratio_5s = rolling_mean(spread, next 50 bars) / current_spread
SPREAD_RATIO_STEPS = 50       # 50 bars = 5 seconds

# Features to exclude for spread model (mid-related but keep spread features)
EXCLUDE_FEATURES_SPREAD = [
    # Absolute price level
    'mid', 'best_bid', 'best_ask', 'microprice',
    # Time-of-day
    'hour_norm', 'minute_norm', 'time_since_rth', 'time_to_close',
    # Price momentum (for spread we care about book dynamics, not price direction)
    # Keep spread, spread_ticks, spread dynamics
]

# Default LightGBM params (same as existing scripts)
LGBM_PARAMS = {
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
}


# ============================================================================
# DISCORD NOTIFIER (best-effort, no crash if unavailable)
# ============================================================================
def send_discord(msg: str):
    """Send message to Discord. No-op on failure."""
    try:
        import subprocess
        node_script = f"""
const {{ Client, GatewayIntentBits }} = require('discord.js');
// Simple webhook or just log -- use the mcp discord tool approach
// This is a fallback; the MCP tool handles actual sending
console.log('DISCORD_MSG:', JSON.stringify({{msg: {json.dumps(msg)}}}));
"""
        # Best-effort: just log to console and the MCP tool picks it up
        # The actual Discord send is handled by the MCP layer in CLAUDE.md
        pass
    except Exception:
        pass
    logger.info(f"[DISCORD] {msg}")


# ============================================================================
# SPREAD RATIO TARGET COMPUTATION
# ============================================================================

def compute_spread_ratio_target(
    features: np.ndarray,
    feature_names: List[str],
    day_boundaries: list,
    steps: int = SPREAD_RATIO_STEPS,
) -> np.ndarray:
    """
    Compute spread_ratio_5s = rolling_mean(spread, next 50 bars) / current_spread.

    Strictly causal: uses only future spread values (this is the TARGET, not a feature).
    NaN-fills day boundary crossings.
    """
    feat_idx = {n: i for i, n in enumerate(feature_names)}
    N = len(features)
    n_days = len(day_boundaries) - 1

    if 'spread' in feat_idx:
        current_spread = features[:, feat_idx['spread']].astype(np.float64)
    elif 'best_ask' in feat_idx and 'best_bid' in feat_idx:
        current_spread = (
            features[:, feat_idx['best_ask']] -
            features[:, feat_idx['best_bid']]
        ).astype(np.float64)
    else:
        logger.warning("Cannot find spread column, using constant 0.25")
        current_spread = np.full(N, 0.25, dtype=np.float64)

    # Replace invalid spread values
    current_spread = np.where(current_spread > 0, current_spread, np.nan)

    # Compute rolling mean of FUTURE spread over next `steps` bars
    # future_spread_rolling[i] = mean(spread[i+1 .. i+steps])
    cs = np.cumsum(np.where(np.isfinite(current_spread), current_spread, 0.0))
    cs_pad = np.concatenate([[0.0], cs])

    # Count of valid values in window
    valid_mask = np.isfinite(current_spread).astype(np.float64)
    valid_cs = np.cumsum(valid_mask)
    valid_cs_pad = np.concatenate([[0.0], valid_cs])

    future_mean_spread = np.full(N, np.nan, dtype=np.float64)
    valid_end = N - steps
    if valid_end > 0:
        window_sum = cs_pad[steps:steps + valid_end] - cs_pad[:valid_end]
        window_valid_count = valid_cs_pad[steps:steps + valid_end] - valid_cs_pad[:valid_end]
        with np.errstate(invalid='ignore', divide='ignore'):
            future_mean = np.where(
                window_valid_count > 0,
                window_sum / window_valid_count,
                np.nan
            )
        future_mean_spread[:valid_end] = future_mean

    # Compute ratio
    with np.errstate(invalid='ignore', divide='ignore'):
        spread_ratio = np.where(
            np.isfinite(current_spread) & (current_spread > TICK_SIZE * 0.1) &
            np.isfinite(future_mean_spread),
            future_mean_spread / current_spread,
            np.nan
        ).astype(np.float32)

    # NaN-fill day boundary crossings
    if n_days > 1:
        for d in range(n_days - 1):
            day_end = day_boundaries[d + 1]
            nan_start = max(day_boundaries[d], day_end - steps)
            spread_ratio[nan_start:day_end] = np.nan

    valid_count = np.isfinite(spread_ratio).sum()
    logger.info(
        f"spread_ratio_5s: {valid_count:,} valid bars, "
        f"mean={float(np.nanmean(spread_ratio)):.4f}, "
        f"std={float(np.nanstd(spread_ratio)):.4f}"
    )

    return spread_ratio, current_spread


# ============================================================================
# ZSCORE NORMALIZATION (per-fold, in-sample)
# ============================================================================

def zscore(arr: np.ndarray, clip: float = 5.0) -> np.ndarray:
    """Normalize to z-scores, clip at +/- clip std."""
    valid = np.isfinite(arr)
    if valid.sum() < 10:
        return np.zeros_like(arr)
    mu = float(np.mean(arr[valid]))
    sigma = float(np.std(arr[valid]))
    if sigma < 1e-12:
        return np.zeros_like(arr)
    z = (arr - mu) / sigma
    return np.clip(z, -clip, clip)


# ============================================================================
# COMBINED SIGNAL CONSTRUCTION
# ============================================================================

def build_combined_signal(
    dir_preds: np.ndarray,
    spread_preds: np.ndarray,
    alpha: float,
    beta: float,
) -> np.ndarray:
    """
    Combined signal = alpha * direction_z + beta * spread_z

    Both inputs are normalized to z-scores before combination.
    """
    dir_z = zscore(dir_preds)
    spread_z = zscore(spread_preds)
    combined = alpha * dir_z + beta * spread_z
    return combined, dir_z, spread_z


def build_multiplicative_signal(
    dir_preds: np.ndarray,
    spread_preds: np.ndarray,
) -> np.ndarray:
    """
    Multiplicative signal: direction_z * (1 + spread_info)
    where spread_info > 0 means spread predicted to widen (favorable for limit orders).

    spread_ratio > 1.0 means spread widens = more favorable entry edge.
    We define spread_info = max(0, spread_z) so we only amplify when spread looks favorable.
    """
    dir_z = zscore(dir_preds)
    spread_z = zscore(spread_preds)
    # spread_ratio > 1 means spread widens => favorable
    # normalize: spread_z > 0 means spread expected to be relatively wide
    spread_info = np.maximum(0, spread_z)  # only positive regime boosts signal
    combined = dir_z * (1.0 + 0.5 * spread_info)
    return combined, dir_z, spread_z


# ============================================================================
# REALISTIC LIMIT ORDER SIMULATION
# ============================================================================

def simulate_limit_order_strategy(
    combined_signal: np.ndarray,
    dir_z: np.ndarray,
    spread_preds: np.ndarray,
    mid_prices: np.ndarray,
    current_spread: np.ndarray,
    day_boundaries: list,
    threshold_quantile: float = 0.80,
    hold_bars: int = 30,           # how long to hold after fill (bars)
    fill_window_bars: int = 50,    # how long to wait for limit order fill (5s)
    spread_condition: bool = False, # only enter when spread predicted to widen
    commission_rt: float = COMMISSION_RT,
    tick_size: float = TICK_SIZE,
    tick_value: float = TICK_VALUE,
) -> dict:
    """
    Simulate a realistic limit order strategy.

    Entry:
      - Post limit order at best_bid (for buys) or best_ask (for sells)
      - Fill detection: mid price touches limit price within fill_window_bars
      - For buys: limit at best_bid, filled if mid drops to (or below) best_bid
        which means price moved against us briefly = we got filled
        Simplification: filled if future min_mid <= entry_bid within window
      - For sells: limit at best_ask, filled if future max_mid >= entry_ask

    PnL per trade:
      - entry_edge = 0.5 * spread (earn half the spread on passive fill)
      - directional_move = (exit_price - entry_price) * direction
      - exit_cost = 0.5 * spread if market exit (conservative)
      - commission = $2.50 RT = COMMISSION_TICKS ticks

    Returns comprehensive performance metrics.
    """
    N = len(combined_signal)
    n_days = len(day_boundaries) - 1

    # Signal threshold: only trade when |signal| > threshold
    valid_signal = np.isfinite(combined_signal)
    signal_abs = np.abs(combined_signal)
    threshold = np.nanquantile(signal_abs[valid_signal], threshold_quantile)

    # Direction: +1 for long, -1 for short (based on sign of signal)
    directions = np.sign(combined_signal)

    # Spread condition filter: only enter when spread model predicts widening
    # (spread_preds > current = ratio > 1.0 means spread expected to widen)
    if spread_condition:
        spread_ratio_preds = spread_preds.copy()
        spread_favorable = spread_ratio_preds > 1.0  # spread will widen = favorable entry
    else:
        spread_favorable = np.ones(N, dtype=bool)

    # Precompute rolling future min/max for fill detection (fully vectorized)
    # future_min[i] = min(mid[i+1 .. i+fill_window_bars])
    # future_max[i] = max(mid[i+1 .. i+fill_window_bars])
    logger.info(f"  Precomputing fill windows (fill_window={fill_window_bars} bars, hold={hold_bars} bars)...")

    # Use scipy's uniform_filter or a stride trick for rolling min/max.
    # Most efficient: use a bottleneck or scipy approach on reversed array.
    # We use a manual sliding minimum via numpy strides (memory-efficient).

    # Efficient rolling min/max via strided windows on the SHIFTED array
    # mid_shifted[i] = mid[i+1] so future_min[i] = rolling_min(mid_shifted, window=fill_window)[i]
    mid_shifted = np.concatenate([mid_prices[1:], np.full(fill_window_bars + 1, np.nan, dtype=np.float64)])

    # Compute rolling min/max using a stride-based approach
    # For large arrays, we use a direct vectorized min over sub-arrays
    future_min_fill = np.full(N, np.nan, dtype=np.float32)
    future_max_fill = np.full(N, np.nan, dtype=np.float32)
    future_exit_price = np.full(N, np.nan, dtype=np.float32)

    # Compute in chunks to stay memory-efficient
    # Each chunk: take mid[start+1 .. start+fill_window] and compute min/max
    chunk = fill_window_bars
    valid_end = N - fill_window_bars

    if valid_end > 0:
        # Build strided view: shape (valid_end, fill_window_bars)
        # This creates a view with strides — no copy needed
        from numpy.lib.stride_tricks import as_strided
        itemsize = mid_shifted.itemsize
        # mid_shifted has length N + fill_window_bars
        stride_view = as_strided(
            mid_shifted,
            shape=(valid_end, fill_window_bars),
            strides=(itemsize, itemsize)
        )
        # Suppress nanmin warnings for all-NaN slices
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)
            future_min_fill[:valid_end] = np.nanmin(stride_view, axis=1).astype(np.float32)
            future_max_fill[:valid_end] = np.nanmax(stride_view, axis=1).astype(np.float32)

    # Exit price: mid at fill_window_bars + hold_bars after current bar
    exit_offset = fill_window_bars + hold_bars
    if exit_offset < N:
        future_exit_price[:N - exit_offset] = mid_prices[exit_offset:].astype(np.float32)

    logger.info(
        f"  Fill window precomputed: "
        f"min range [{float(np.nanmin(future_min_fill)):.2f}, {float(np.nanmax(future_min_fill)):.2f}]"
    )

    # Day boundary mask: close all positions before end of day
    day_end_mask = np.zeros(N, dtype=bool)
    for d in range(n_days):
        day_end = day_boundaries[d + 1]
        # No new entries within fill_window + hold_bars of day end
        buffer = fill_window_bars + hold_bars + 5
        day_start_safe = max(day_boundaries[d], day_end - buffer)
        day_end_mask[day_start_safe:day_end] = True

    # ================================================================
    # Main simulation loop
    # ================================================================
    trades = []
    pnl_per_day = {d: 0.0 for d in range(n_days)}
    last_trade_bar = -fill_window_bars  # prevent overlapping trades

    for i in range(N - fill_window_bars - hold_bars - 1):
        # Skip if near day end
        if day_end_mask[i]:
            continue

        # Skip if signal not strong enough
        if not valid_signal[i]:
            continue
        if signal_abs[i] <= threshold:
            continue

        # Skip if spread condition not met
        if not spread_favorable[i]:
            continue

        # Direction of trade
        direction = directions[i]
        if direction == 0:
            continue

        # Prevent overlapping trades: must wait for previous hold to expire
        if i < last_trade_bar + fill_window_bars + hold_bars:
            continue

        # Current prices
        cur_mid = mid_prices[i]
        cur_spread = current_spread[i] if np.isfinite(current_spread[i]) else TICK_SIZE

        if not np.isfinite(cur_mid) or cur_mid <= 0:
            continue

        # Limit order price
        half_spread = cur_spread / 2.0
        if direction > 0:
            # BUY limit at best_bid
            limit_price = cur_mid - half_spread
            # Filled if future min_mid <= limit_price (price touched our bid)
            filled = (np.isfinite(future_min_fill[i]) and
                      future_min_fill[i] <= limit_price + TICK_SIZE * 0.01)
        else:
            # SELL limit at best_ask
            limit_price = cur_mid + half_spread
            # Filled if future max_mid >= limit_price
            filled = (np.isfinite(future_max_fill[i]) and
                      future_max_fill[i] >= limit_price - TICK_SIZE * 0.01)

        if not filled:
            continue

        # Fill obtained - compute PnL
        entry_price = limit_price
        entry_edge = half_spread  # earn half-spread on passive fill

        # Exit price: mid at hold_bars after fill (market order exit)
        exit_price = future_exit_price[i]
        if not np.isfinite(exit_price):
            continue

        # Exit cost: half spread (market order)
        exit_cost = half_spread

        # Directional PnL
        directional_move = (exit_price - entry_price) * direction

        # Gross PnL in points
        gross_pnl_pts = entry_edge + directional_move - exit_cost

        # Convert to dollars and subtract commission
        gross_pnl_dollars = gross_pnl_pts / TICK_SIZE * TICK_VALUE
        net_pnl_dollars = gross_pnl_dollars - commission_rt

        # Gross/net in ticks
        gross_pnl_ticks = gross_pnl_pts / TICK_SIZE
        net_pnl_ticks = gross_pnl_ticks - COMMISSION_TICKS

        # Which day?
        trade_day = 0
        for d in range(n_days):
            if day_boundaries[d] <= i < day_boundaries[d + 1]:
                trade_day = d
                break

        trade = {
            'bar': int(i),
            'day': int(trade_day),
            'direction': int(direction),
            'signal': float(combined_signal[i]),
            'entry_price': float(entry_price),
            'exit_price': float(exit_price),
            'spread_at_entry': float(cur_spread),
            'entry_edge_ticks': float(entry_edge / TICK_SIZE),
            'directional_ticks': float(directional_move / TICK_SIZE),
            'gross_pnl_ticks': float(gross_pnl_ticks),
            'net_pnl_ticks': float(net_pnl_ticks),
            'gross_pnl_dollars': float(gross_pnl_dollars),
            'net_pnl_dollars': float(net_pnl_dollars),
        }
        trades.append(trade)
        pnl_per_day[trade_day] += net_pnl_dollars
        last_trade_bar = i

    # ================================================================
    # Aggregate statistics
    # ================================================================
    if not trades:
        return {
            'error': 'No trades generated',
            'threshold_quantile': threshold_quantile,
            'threshold': float(threshold),
            'spread_condition': spread_condition,
        }

    n_trades = len(trades)
    net_pnls = np.array([t['net_pnl_dollars'] for t in trades])
    gross_pnls = np.array([t['gross_pnl_dollars'] for t in trades])
    dir_ticks = np.array([t['directional_ticks'] for t in trades])
    entry_edges = np.array([t['entry_edge_ticks'] for t in trades])

    # Daily PnL
    daily_pnls = np.array([pnl_per_day.get(d, 0.0) for d in range(n_days)])
    trading_days = np.sum(daily_pnls != 0.0)
    daily_trades = n_trades / max(trading_days, 1)

    # Win rate
    wins = np.sum(net_pnls > 0)
    win_rate = wins / n_trades

    # Total PnL
    total_net = float(np.sum(net_pnls))
    total_gross = float(np.sum(gross_pnls))
    total_commission = float(n_trades * commission_rt)

    # PnL per trade
    avg_net_per_trade = total_net / n_trades
    avg_gross_per_trade = total_gross / n_trades
    avg_net_ticks = float(np.mean([t['net_pnl_ticks'] for t in trades]))

    # Sharpe (annualized, assuming 100 trades/day)
    # Use daily PnL for Sharpe calculation
    active_daily_pnls = daily_pnls[daily_pnls != 0]
    if len(active_daily_pnls) > 2 and np.std(active_daily_pnls) > 0:
        daily_sharpe = (
            float(np.mean(active_daily_pnls)) /
            float(np.std(active_daily_pnls)) *
            np.sqrt(252)
        )
    else:
        daily_sharpe = 0.0

    # Trade-level Sharpe
    if np.std(net_pnls) > 0:
        trade_sharpe = (
            float(np.mean(net_pnls)) /
            float(np.std(net_pnls)) *
            np.sqrt(252 * max(daily_trades, 1))
        )
    else:
        trade_sharpe = 0.0

    # Max drawdown (cumulative PnL)
    cum_pnl = np.cumsum(net_pnls)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = cum_pnl - peak
    max_drawdown = float(np.min(drawdown))

    # Profit factor
    gross_wins = float(np.sum(net_pnls[net_pnls > 0]))
    gross_losses = float(np.abs(np.sum(net_pnls[net_pnls < 0])))
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else float('inf')

    # Fill rate: trades / opportunities
    # opportunities = bars where signal exceeded threshold
    opportunities = int(np.sum(
        valid_signal & (signal_abs > threshold) & ~day_end_mask & spread_favorable
    ))
    fill_rate = n_trades / opportunities if opportunities > 0 else 0.0

    # Average directional contribution
    avg_direction_ticks = float(np.mean(dir_ticks))
    avg_entry_edge_ticks = float(np.mean(entry_edges))

    # Per-day summary
    day_summary = [
        {
            'day': d,
            'net_pnl': float(pnl_per_day[d]),
            'n_trades': int(sum(1 for t in trades if t['day'] == d)),
        }
        for d in range(n_days)
    ]

    return {
        # Configuration
        'threshold_quantile': float(threshold_quantile),
        'threshold': float(threshold),
        'hold_bars': int(hold_bars),
        'hold_sec': float(hold_bars / BARS_PER_SEC),
        'fill_window_bars': int(fill_window_bars),
        'fill_window_sec': float(fill_window_bars / BARS_PER_SEC),
        'spread_condition': bool(spread_condition),
        # Trade counts
        'n_trades': int(n_trades),
        'opportunities': int(opportunities),
        'fill_rate': float(fill_rate),
        # PnL metrics
        'total_net_pnl': float(total_net),
        'total_gross_pnl': float(total_gross),
        'total_commission': float(total_commission),
        'avg_net_per_trade_dollars': float(avg_net_per_trade),
        'avg_gross_per_trade_dollars': float(avg_gross_per_trade),
        'avg_net_ticks': float(avg_net_ticks),
        'avg_entry_edge_ticks': float(avg_entry_edge_ticks),
        'avg_direction_ticks': float(avg_direction_ticks),
        # Risk metrics
        'win_rate': float(win_rate),
        'profit_factor': float(profit_factor) if np.isfinite(profit_factor) else 9999.0,
        'max_drawdown_dollars': float(max_drawdown),
        'daily_sharpe': float(daily_sharpe),
        'trade_sharpe': float(trade_sharpe),
        # Daily breakdown
        'n_days': int(n_days),
        'trading_days_with_trades': int(trading_days),
        'avg_trades_per_day': float(daily_trades),
        'daily_net_pnl_mean': float(np.mean(active_daily_pnls)) if len(active_daily_pnls) > 0 else 0.0,
        'daily_net_pnl_std': float(np.std(active_daily_pnls)) if len(active_daily_pnls) > 0 else 0.0,
        'pct_profitable_days': float(np.mean(active_daily_pnls > 0)) if len(active_daily_pnls) > 0 else 0.0,
        'day_summary': day_summary,
    }


# ============================================================================
# WALK-FORWARD DIRECTION MODEL
# ============================================================================

def train_direction_model(
    scanner: MBOAlphaScanner,
    target: np.ndarray,
    fast_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Train walk-forward direction model (ret_3s).
    Returns (predictions, actuals, metrics).
    """
    logger.info("Training direction model (ret_3s)...")
    logger.info(f"  Excluding {len(EXCLUDE_FEATURES_DIRECTION)} features (vol proxies + price + time)")

    keep_mask = np.array([fn not in EXCLUDE_FEATURES_DIRECTION for fn in scanner.feature_names])
    features_dir = scanner.features[:, keep_mask]
    names_dir = [fn for fn in scanner.feature_names if fn not in EXCLUDE_FEATURES_DIRECTION]

    logger.info(f"  Using {len(names_dir)} features")

    params = LGBM_PARAMS.copy()
    if fast_mode:
        params['n_estimators'] = 200

    result = walk_forward_evaluate(
        features=features_dir,
        target=target,
        day_boundaries=scanner.day_boundaries,
        feature_names=names_dir,
        model_type='lgbm',
        lgbm_params=params,
        hour_of_day=scanner.hour_of_day,
    )

    if 'error' in result:
        logger.error(f"Direction model failed: {result['error']}")
        return None, None, result

    preds = result['predictions']
    actuals = result['actuals']
    pred_idx = result.get('pred_indices', None)

    logger.info(
        f"  Direction model: IC={result['ic']:.4f} ICIR={result['icir']:.2f} "
        f"t={result['tstat']:.2f} FoldC={result['fold_con']:.0%} "
        f"folds={result['n_folds']}"
    )
    if result.get('fold_ics'):
        ics_str = " ".join(f"{x:+.3f}" for x in result['fold_ics'])
        logger.info(f"  Fold ICs: [{ics_str}]")

    return preds, actuals, result


# ============================================================================
# WALK-FORWARD SPREAD MODEL
# ============================================================================

def train_spread_model(
    scanner: MBOAlphaScanner,
    target: np.ndarray,
    fast_mode: bool = False,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """
    Train walk-forward spread model (spread_ratio_5s).
    Uses all features except mid-price related ones (keep spread features).
    """
    logger.info("Training spread model (spread_ratio_5s)...")

    keep_mask = np.array([fn not in EXCLUDE_FEATURES_SPREAD for fn in scanner.feature_names])
    features_spread = scanner.features[:, keep_mask]
    names_spread = [fn for fn in scanner.feature_names if fn not in EXCLUDE_FEATURES_SPREAD]

    logger.info(f"  Using {len(names_spread)} features (kept spread features, excluded price/time)")

    params = LGBM_PARAMS.copy()
    if fast_mode:
        params['n_estimators'] = 200

    result = walk_forward_evaluate(
        features=features_spread,
        target=target,
        day_boundaries=scanner.day_boundaries,
        feature_names=names_spread,
        model_type='lgbm',
        lgbm_params=params,
        hour_of_day=scanner.hour_of_day,
    )

    if 'error' in result:
        logger.error(f"Spread model failed: {result['error']}")
        return None, None, result

    preds = result['predictions']
    actuals = result['actuals']

    logger.info(
        f"  Spread model: IC={result['ic']:.4f} ICIR={result['icir']:.2f} "
        f"t={result['tstat']:.2f} FoldC={result['fold_con']:.0%} "
        f"folds={result['n_folds']}"
    )
    if result.get('fold_ics'):
        ics_str = " ".join(f"{x:+.3f}" for x in result['fold_ics'])
        logger.info(f"  Fold ICs: [{ics_str}]")

    return preds, actuals, result


# ============================================================================
# ALIGN PREDICTIONS
# ============================================================================

def align_predictions(
    dir_preds: np.ndarray,
    dir_indices: Optional[np.ndarray],
    spread_preds: np.ndarray,
    spread_indices: Optional[np.ndarray],
    n_total: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Align direction and spread predictions to the same bar indices.

    Walk-forward evaluation returns predictions in order of test bars.
    We need to put them back into the full bar index space.
    """
    dir_full = np.full(n_total, np.nan, dtype=np.float32)
    spread_full = np.full(n_total, np.nan, dtype=np.float32)

    if dir_indices is not None and len(dir_indices) == len(dir_preds):
        valid = (dir_indices >= 0) & (dir_indices < n_total)
        dir_full[dir_indices[valid].astype(int)] = dir_preds[valid].astype(np.float32)
    else:
        # No index info: assume predictions are contiguous (may have offset)
        n = min(len(dir_preds), n_total)
        dir_full[:n] = dir_preds[:n].astype(np.float32)

    if spread_indices is not None and len(spread_indices) == len(spread_preds):
        valid = (spread_indices >= 0) & (spread_indices < n_total)
        spread_full[spread_indices[valid].astype(int)] = spread_preds[valid].astype(np.float32)
    else:
        n = min(len(spread_preds), n_total)
        spread_full[:n] = spread_preds[:n].astype(np.float32)

    return dir_full, spread_full


# ============================================================================
# FORMAT RESULTS SCOREBOARD
# ============================================================================

def format_scoreboard(all_sim_results: dict) -> str:
    """Format comprehensive scoreboard of all simulation results."""
    lines = [
        "",
        "=" * 100,
        "COMBINED STRATEGY SCOREBOARD",
        "=" * 100,
        f"{'Config':<40s} {'Trades':>7s} {'FillR':>6s} {'NetPnL$':>9s} {'$/Trade':>8s} "
        f"{'Ticks':>6s} {'WinR':>6s} {'PF':>5s} {'Sharpe':>7s} {'MaxDD':>8s}",
        "-" * 100,
    ]

    for config_name, results in all_sim_results.items():
        for sim_key, sim in results.items():
            if 'error' in sim:
                lines.append(f"  {config_name[:35]:<35s} {sim_key}: ERROR: {sim['error']}")
                continue

            label = f"{config_name[:20]} {sim_key}"[:40]
            pf = sim['profit_factor']
            pf_str = f"{pf:.2f}" if pf < 100 else ">99"

            lines.append(
                f"  {label:<40s} {sim['n_trades']:>7d} {sim['fill_rate']:>6.1%} "
                f"${sim['total_net_pnl']:>8.0f} ${sim['avg_net_per_trade_dollars']:>7.2f} "
                f"{sim['avg_net_ticks']:>6.2f} {sim['win_rate']:>6.1%} "
                f"{pf_str:>5s} {sim['daily_sharpe']:>7.2f} ${sim['max_drawdown_dollars']:>7.0f}"
            )

    lines.append("=" * 100)

    # Find best configuration
    best_sharpe = -999.0
    best_config = None
    for config_name, results in all_sim_results.items():
        for sim_key, sim in results.items():
            if 'error' not in sim and sim.get('daily_sharpe', -999) > best_sharpe:
                best_sharpe = sim['daily_sharpe']
                best_config = (config_name, sim_key, sim)

    if best_config:
        cfg_name, sim_key, sim = best_config
        lines.append(f"\nBEST CONFIGURATION: {cfg_name} / {sim_key}")
        lines.append(f"  Net PnL: ${sim['total_net_pnl']:.0f} total")
        lines.append(f"  Per trade: ${sim['avg_net_per_trade_dollars']:.2f} net ({sim['avg_net_ticks']:.2f} ticks)")
        lines.append(f"  Trades: {sim['n_trades']} ({sim['avg_trades_per_day']:.1f}/day)")
        lines.append(f"  Win rate: {sim['win_rate']:.1%}")
        lines.append(f"  Profit factor: {sim['profit_factor']:.2f}")
        lines.append(f"  Daily Sharpe: {sim['daily_sharpe']:.2f}")
        lines.append(f"  Max Drawdown: ${sim['max_drawdown_dollars']:.0f}")
        lines.append(f"  Profitable days: {sim['pct_profitable_days']:.1%}")
        lines.append(f"  Avg entry edge: {sim['avg_entry_edge_ticks']:.3f} ticks")
        lines.append(f"  Avg direction contribution: {sim['avg_direction_ticks']:.3f} ticks")

    return "\n".join(lines)


def format_discord_summary(
    dir_metrics: dict,
    spread_metrics: dict,
    all_sim_results: dict,
    elapsed_sec: float,
) -> str:
    """Concise Discord summary of combined strategy results."""
    lines = [
        "**COMBINED STRATEGY RESULTS**",
        f"Elapsed: {elapsed_sec/60:.1f} min",
        "",
        "**Model Performance:**",
        "```",
    ]

    if 'error' not in dir_metrics:
        lines.append(
            f"Direction (ret_3s):    IC={dir_metrics['ic']:.4f} "
            f"ICIR={dir_metrics['icir']:.2f} t={dir_metrics['tstat']:.2f} "
            f"FoldC={dir_metrics['fold_con']:.0%}"
        )
    else:
        lines.append(f"Direction model: ERROR - {dir_metrics.get('error', '?')}")

    if 'error' not in spread_metrics:
        lines.append(
            f"Spread (ratio_5s):     IC={spread_metrics['ic']:.4f} "
            f"ICIR={spread_metrics['icir']:.2f} t={spread_metrics['tstat']:.2f} "
            f"FoldC={spread_metrics['fold_con']:.0%}"
        )
    else:
        lines.append(f"Spread model: ERROR - {spread_metrics.get('error', '?')}")

    lines.append("```")
    lines.append("")
    lines.append("**Limit Order Simulation Results:**")
    lines.append("```")
    lines.append(
        f"{'Config':<38s} {'Trd':>4s} {'$/Trd':>7s} {'WR':>5s} {'Sharpe':>6s}"
    )
    lines.append("-" * 65)

    # Sort by Sharpe
    all_configs = []
    for cfg, results in all_sim_results.items():
        for sim_key, sim in results.items():
            if 'error' not in sim:
                all_configs.append((cfg, sim_key, sim))
    all_configs.sort(key=lambda x: x[2].get('daily_sharpe', -999), reverse=True)

    for cfg_name, sim_key, sim in all_configs[:15]:
        label = f"{cfg_name[:25]} {sim_key}"[:38]
        lines.append(
            f"  {label:<38s} {sim['n_trades']:>4d} "
            f"${sim['avg_net_per_trade_dollars']:>6.2f} "
            f"{sim['win_rate']:>5.1%} {sim['daily_sharpe']:>6.2f}"
        )

    lines.append("```")

    if all_configs:
        best_cfg, best_key, best = all_configs[0]
        lines.append(f"\n**Best: {best_cfg} / {best_key}**")
        lines.append(
            f"Net: ${best['total_net_pnl']:.0f} | "
            f"{best['n_trades']} trades ({best['avg_trades_per_day']:.1f}/day) | "
            f"WR={best['win_rate']:.1%} | "
            f"PF={best['profit_factor']:.2f} | "
            f"DD=${best['max_drawdown_dollars']:.0f}"
        )

    return "\n".join(lines)


# ============================================================================
# JSON SERIALIZATION HELPER
# ============================================================================

def make_serializable(obj):
    """Recursively convert numpy types and non-finite floats for JSON."""
    if isinstance(obj, dict):
        return {k: make_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_serializable(v) for v in obj]
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer,)):
        return int(obj)
    elif isinstance(obj, (np.floating,)):
        v = float(obj)
        if not np.isfinite(v):
            return str(v)
        return v
    elif isinstance(obj, float):
        if not np.isfinite(obj):
            return str(obj)
        return obj
    return obj


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Combined direction + spread strategy')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: fewer estimators, fewer threshold tests')
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip model training (must have predictions saved)')
    parser.add_argument('--hold-secs', nargs='+', type=float,
                        default=[3.0, 5.0, 10.0, 30.0],
                        help='Hold periods in seconds to test')
    args = parser.parse_args()

    logger.info("=" * 80)
    logger.info("COMBINED DIRECTION + SPREAD STRATEGY")
    logger.info(f"  Fast mode: {args.fast}")
    logger.info(f"  Hold periods: {args.hold_secs}s")
    logger.info("=" * 80)

    t_start = time.time()
    last_progress = t_start

    # ------------------------------------------------------------------
    # Step 1: Load data
    # ------------------------------------------------------------------
    logger.info("\n[Step 1] Loading feature cache...")
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    stats = load_feature_cache(scanner)
    if stats is None:
        logger.info("No feature cache. Computing from scratch (slow)...")
        stats = scanner.load_from_cache()

    N = len(scanner.mid_prices)
    n_days = len(scanner.day_boundaries) - 1
    logger.info(
        f"  Data: {N:,} snapshots, {n_days} days, "
        f"{len(scanner.feature_names)} features"
    )

    send_discord(
        f"[Step 1/5] Data loaded: {N:,} bars, {n_days} days, "
        f"{len(scanner.feature_names)} features"
    )

    # ------------------------------------------------------------------
    # Step 2: Compute targets
    # ------------------------------------------------------------------
    logger.info("\n[Step 2] Computing targets...")

    # Direction target: ret_3s
    logger.info("  Computing ret_3s target...")
    direction_targets = compute_return_targets(
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        sample_interval_ms=100,
        horizons_sec={'3s': 3},
        include_flow_target=False,
    )
    ret_3s = direction_targets['ret_3s']
    logger.info(
        f"  ret_3s: {np.isfinite(ret_3s).sum():,} valid bars, "
        f"mean={float(np.nanmean(ret_3s)):.6f}, "
        f"std={float(np.nanstd(ret_3s)):.6f}"
    )

    # Spread target: spread_ratio_5s
    logger.info("  Computing spread_ratio_5s target...")
    spread_ratio_5s, current_spread = compute_spread_ratio_target(
        features=scanner.features,
        feature_names=scanner.feature_names,
        day_boundaries=scanner.day_boundaries,
        steps=SPREAD_RATIO_STEPS,
    )

    now = time.time()
    logger.info(f"  Targets computed in {now - t_start:.1f}s")
    send_discord(
        f"[Step 2/5] Targets computed. "
        f"ret_3s: {np.isfinite(ret_3s).sum():,} valid. "
        f"spread_ratio_5s: {np.isfinite(spread_ratio_5s).sum():,} valid"
    )

    # ------------------------------------------------------------------
    # Step 3: Train models
    # ------------------------------------------------------------------
    logger.info("\n[Step 3] Training models via walk-forward...")

    send_discord(
        f"[Step 3/5] Training direction model (ret_3s)... "
        f"This takes ~{5 if args.fast else 15} min"
    )

    dir_preds, dir_actuals, dir_metrics = train_direction_model(
        scanner=scanner,
        target=ret_3s,
        fast_mode=args.fast,
    )

    now = time.time()
    elapsed_so_far = now - t_start
    if now - last_progress > 30:
        send_discord(
            f"[Step 3/5] Direction model done in {elapsed_so_far/60:.1f} min. "
            f"IC={dir_metrics.get('ic', 'N/A')} Training spread model..."
        )
        last_progress = now

    send_discord(
        f"[Step 3/5] Training spread model (spread_ratio_5s)... "
        f"Direction IC={dir_metrics.get('ic', 'N/A'):.4f}"
    )

    spread_preds, spread_actuals, spread_metrics = train_spread_model(
        scanner=scanner,
        target=spread_ratio_5s,
        fast_mode=args.fast,
    )

    now = time.time()
    logger.info(f"  Both models trained in {(now - t_start)/60:.1f} min")
    send_discord(
        f"[Step 3/5] Both models trained! "
        f"Direction IC={dir_metrics.get('ic', 'N/A'):.4f} | "
        f"Spread IC={spread_metrics.get('ic', 'N/A'):.4f}"
    )

    # Check if either model failed
    if dir_preds is None:
        send_discord(f"ERROR: Direction model failed: {dir_metrics.get('error', '?')}")
        logger.error("Direction model failed. Exiting.")
        return

    if spread_preds is None:
        send_discord(f"ERROR: Spread model failed: {spread_metrics.get('error', '?')}")
        logger.error("Spread model failed. Exiting.")
        return

    # ------------------------------------------------------------------
    # Step 4: Align predictions and build combined signals
    # ------------------------------------------------------------------
    logger.info("\n[Step 4] Aligning predictions and building combined signals...")

    dir_idx = dir_metrics.get('pred_indices', None)
    spread_idx = spread_metrics.get('pred_indices', None)

    dir_full, spread_full = align_predictions(
        dir_preds=dir_preds,
        dir_indices=dir_idx,
        spread_preds=spread_preds,
        spread_indices=spread_idx,
        n_total=N,
    )

    logger.info(
        f"  Direction: {np.isfinite(dir_full).sum():,} valid predictions "
        f"({np.isfinite(dir_full).mean():.1%} coverage)"
    )
    logger.info(
        f"  Spread: {np.isfinite(spread_full).sum():,} valid predictions "
        f"({np.isfinite(spread_full).mean():.1%} coverage)"
    )

    # Only use bars where BOTH predictions are valid
    both_valid = np.isfinite(dir_full) & np.isfinite(spread_full)
    logger.info(f"  Both valid: {both_valid.sum():,} bars ({both_valid.mean():.1%} of total)")

    # Build combined signals for different alpha/beta weights
    weight_configs = [
        ('dir_0.7_spread_0.3', 0.7, 0.3, 'linear'),
        ('dir_0.5_spread_0.5', 0.5, 0.5, 'linear'),
        ('dir_0.3_spread_0.7', 0.3, 0.7, 'linear'),
        ('dir_only',           1.0, 0.0, 'linear'),
        ('spread_conditioned', 1.0, 0.0, 'linear'),  # direction only but with spread filter
        ('multiplicative',     0.0, 0.0, 'multiplicative'),
    ]

    combined_signals = {}
    for name, alpha, beta, mode in weight_configs:
        if mode == 'multiplicative':
            combined, dir_z, spread_z = build_multiplicative_signal(
                dir_preds=dir_full,
                spread_preds=spread_full,
            )
        else:
            combined, dir_z, spread_z = build_combined_signal(
                dir_preds=dir_full,
                spread_preds=spread_full,
                alpha=alpha,
                beta=beta,
            )
        # Mask out bars where either prediction is invalid
        combined[~both_valid] = np.nan
        combined_signals[name] = {
            'combined': combined,
            'dir_z': dir_z,
            'spread_z': spread_z,
            'alpha': alpha,
            'beta': beta,
            'mode': mode,
        }
        logger.info(f"  Signal '{name}': valid={np.isfinite(combined).sum():,}")

    send_discord(
        f"[Step 4/5] Combined signals built. "
        f"Overlap (both models valid): {both_valid.sum():,} bars ({both_valid.mean():.1%}). "
        f"Running limit order simulations..."
    )

    # ------------------------------------------------------------------
    # Step 5: Simulate limit order strategy
    # ------------------------------------------------------------------
    logger.info("\n[Step 5] Running limit order simulations...")

    thresholds_to_test = [0.80, 0.90, 0.95] if args.fast else [0.60, 0.70, 0.80, 0.90, 0.95]
    hold_secs_to_test = args.hold_secs[:2] if args.fast else args.hold_secs

    all_sim_results = {}
    sim_count = 0
    total_sims = (
        len(combined_signals) * len(thresholds_to_test) * len(hold_secs_to_test)
    )

    last_progress = time.time()

    for signal_name, sig_data in combined_signals.items():
        all_sim_results[signal_name] = {}
        combined_sig = sig_data['combined']

        # Spread condition: use spread model predictions to filter
        spread_ratio_for_filter = spread_full.copy()

        for hold_sec in hold_secs_to_test:
            hold_bars = int(hold_sec * BARS_PER_SEC)

            for thr_q in thresholds_to_test:
                # Regular simulation (no spread condition)
                sim_key = f"hold{hold_sec:.0f}s_thr{int(thr_q*100)}pct"

                sim_result = simulate_limit_order_strategy(
                    combined_signal=combined_sig,
                    dir_z=sig_data['dir_z'],
                    spread_preds=spread_ratio_for_filter,
                    mid_prices=scanner.mid_prices,
                    current_spread=current_spread,
                    day_boundaries=scanner.day_boundaries,
                    threshold_quantile=thr_q,
                    hold_bars=hold_bars,
                    fill_window_bars=50,  # 5 seconds
                    spread_condition=False,
                    commission_rt=COMMISSION_RT,
                )
                all_sim_results[signal_name][sim_key] = sim_result
                sim_count += 1

                # Spread-conditioned simulation (for direction-only signal)
                if signal_name == 'dir_only':
                    if 'dir_only_spread_filtered' not in all_sim_results:
                        all_sim_results['dir_only_spread_filtered'] = {}
                    sim_result_sc = simulate_limit_order_strategy(
                        combined_signal=combined_sig,
                        dir_z=sig_data['dir_z'],
                        spread_preds=spread_ratio_for_filter,
                        mid_prices=scanner.mid_prices,
                        current_spread=current_spread,
                        day_boundaries=scanner.day_boundaries,
                        threshold_quantile=thr_q,
                        hold_bars=hold_bars,
                        fill_window_bars=50,
                        spread_condition=True,  # only enter when spread predicted to widen
                        commission_rt=COMMISSION_RT,
                    )
                    all_sim_results['dir_only_spread_filtered'][sim_key] = sim_result_sc
                    sim_count += 1

                now = time.time()
                if now - last_progress > 30:
                    pct = sim_count / max(total_sims, 1) * 100
                    send_discord(
                        f"[Step 5/5] Simulations: {sim_count}/{total_sims} ({pct:.0f}%) done. "
                        f"Elapsed: {(now - t_start)/60:.1f} min"
                    )
                    last_progress = now

    logger.info(f"  Completed {sim_count} simulations")

    # ------------------------------------------------------------------
    # Format and display results
    # ------------------------------------------------------------------
    elapsed_total = time.time() - t_start

    scoreboard = format_scoreboard(all_sim_results)
    logger.info("\n" + scoreboard)

    discord_msg = format_discord_summary(
        dir_metrics=dir_metrics,
        spread_metrics=spread_metrics,
        all_sim_results=all_sim_results,
        elapsed_sec=elapsed_total,
    )

    # ------------------------------------------------------------------
    # Save results
    # ------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = RESULTS_DIR / f"combined_strategy_{timestamp}.json"

    # Remove numpy arrays from metrics before saving
    def clean_metrics(m):
        return {k: v for k, v in m.items()
                if k not in ('predictions', 'actuals', 'pred_indices')}

    output = {
        'timestamp': timestamp,
        'fast_mode': args.fast,
        'hold_secs_tested': hold_secs_to_test,
        'thresholds_tested': thresholds_to_test,
        'n_bars': int(N),
        'n_days': int(n_days),
        'overlap_bars': int(both_valid.sum()),
        'overlap_fraction': float(both_valid.mean()),
        'direction_model': make_serializable(clean_metrics(dir_metrics)),
        'spread_model': make_serializable(clean_metrics(spread_metrics)),
        'simulations': make_serializable(all_sim_results),
        'scoreboard': scoreboard,
        'elapsed_sec': float(elapsed_total),
        'constants': {
            'TICK_SIZE': TICK_SIZE,
            'TICK_VALUE': TICK_VALUE,
            'BARS_PER_SEC': BARS_PER_SEC,
            'COMMISSION_RT': COMMISSION_RT,
            'FILL_WINDOW_SEC': 5.0,
        },
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {result_file}")
    logger.info(f"Total elapsed: {elapsed_total:.0f}s ({elapsed_total/60:.1f} min)")

    # Final Discord message
    send_discord(discord_msg + f"\n\nResults: `{result_file.name}`")
    send_discord(
        f"**Run complete!** {elapsed_total/60:.1f} min total. "
        f"Results at: `alpha_discovery/results/{result_file.name}`"
    )

    print("\n" + "=" * 80)
    print(scoreboard)
    print("=" * 80)

    return output


if __name__ == '__main__':
    main()
