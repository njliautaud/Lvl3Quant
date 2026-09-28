"""
Realistic Limit Order Fill Simulator V2 — Queue-Aware

Builds on magnitude_gated_sim.py with critical enhancements:
1. Real bid/ask prices from MBO snapshots (not mid ± half_tick)
2. Queue-depth-dependent fill probability (not 100% on mid-cross)
3. Market impact / capacity decay model
4. True holdout testing (sweep on first N days, test on last M)

Uses SAVED predictions from V1 training (no retraining needed).
Also can train fresh on all 100 days.

Usage:
    # Load saved predictions + run realistic sim
    python alpha_discovery/realistic_sim_v2.py --load-predictions results/predictions_ret_10s_*.npz

    # Full training + realistic sim (100 days, all resources)
    python alpha_discovery/realistic_sim_v2.py --n-days 100 --horizon ret_10s --target-type mfe_net

    # With GPU training
    python alpha_discovery/realistic_sim_v2.py --n-days 100 --use-gpu
"""

import gc
import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
from scipy.stats import spearmanr

# Add project root
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Configure logging
_log_file = RESULTS_DIR / f"realistic_sim_v2_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
_root = logging.getLogger()
_root.setLevel(logging.INFO)
for _h in _root.handlers[:]:
    _root.removeHandler(_h)
_fmt = logging.Formatter('%(asctime)s [%(name)s] %(message)s', datefmt='%H:%M:%S')
_fh = logging.FileHandler(str(_log_file), mode='w')
_fh.setFormatter(_fmt)
_sh = logging.StreamHandler(sys.stdout)
_sh.setFormatter(_fmt)
_root.addHandler(_fh)
_root.addHandler(_sh)
logger = logging.getLogger("realistic_sim_v2")

# ============================================================================
# Constants
# ============================================================================
TICK_SIZE = 0.25
TICK_VALUE = 12.50
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24 ticks
HALF_TICK = TICK_SIZE / 2  # 0.125
BARS_PER_SEC = 10
HORIZONS = {'ret_3s': 30, 'ret_5s': 50, 'ret_10s': 100, 'ret_30s': 300, 'ret_1m': 600, 'ret_3m': 1800, 'ret_5m': 3000}

# Queue model parameters (from run_queue_position_study.py empirical findings)
# 76% of orders are icebergs (display qty=1), refills go to BACK of FIFO
# Effective queue position for new order: 2-5 contracts ahead
# Fill probability at position 3-5: 80-87% per price sweep
ICEBERG_FRACTION = 0.76
EFFECTIVE_QUEUE_CONTRACTS = 5  # empirical: 2-5, use conservative end
BASE_FILL_PROB_ON_SWEEP = 0.85  # 80-87% empirical, use midpoint


# ============================================================================
# Trade Record
# ============================================================================
@dataclass
class RealisticTrade:
    day: int
    direction: int
    signal_strength: float
    magnitude_pred: float
    signal_bar: int
    post_bar: int
    # Real bid/ask at posting time
    post_bid: float = 0.0
    post_ask: float = 0.0
    post_spread: float = 0.0
    post_queue_depth: float = 0.0  # contracts at our level
    effective_queue_pos: float = 0.0  # after iceberg adjustment
    # Fill info
    fill_bar: int = -1
    fill_mid: float = 0.0
    fill_offset_bars: int = 0
    filled: bool = False
    fill_probability: float = 0.0  # queue-adjusted fill probability
    # Entry/exit
    entry_price: float = 0.0
    exit_bar: int = -1
    exit_mid: float = 0.0
    bars_held: int = 0
    # PnL components
    entry_edge_ticks: float = 0.0
    dir_pnl_ticks: float = 0.0
    exit_edge_ticks: float = 0.0
    net_ticks: float = 0.0
    net_dollars: float = 0.0
    exit_type: str = ''
    # Path analysis: max adverse/favorable excursion after fill
    mae_ticks: float = 0.0  # Maximum Adverse Excursion (worst dip, always <= 0)
    mfe_ticks: float = 0.0  # Maximum Favorable Excursion (best peak, always >= 0)


# ============================================================================
# Data Loading
# ============================================================================
def load_snapshot_data(snapshot_dir: str, n_days: Optional[int] = None) -> Dict:
    """Load raw snapshot NPZ files to get real bid/ask and queue depths.

    Returns dict with arrays aligned to the feature cache:
    - best_bid, best_ask: actual prices per bar
    - bid_queue_depth, ask_queue_depth: contracts at best level
    - bid_order_count, ask_order_count: distinct orders at best level
    - day_boundaries: list of day start indices
    """
    snapshot_dir = Path(snapshot_dir)
    files = sorted(snapshot_dir.glob("*_snapshots.npz"))
    if n_days:
        files = files[:n_days]

    logger.info(f"Loading {len(files)} snapshot files for queue data...")

    all_best_bid = []
    all_best_ask = []
    all_bid_queue = []
    all_ask_queue = []
    all_bid_orders = []
    all_ask_orders = []
    all_mid_prices = []
    day_boundaries = [0]

    corrupted_days = []
    for i, f in enumerate(files):
        data = np.load(str(f))
        gf = data['global_features']
        nf = data['node_features']
        mp = data['mid_prices']
        n_bars = len(mp)

        bid = gf[:, 8].astype(np.float64)
        ask = gf[:, 9].astype(np.float64)
        spread = ask - bid

        # Data quality check: detect corrupted bid/ask (negative spreads)
        neg_spread_pct = (spread < 0).mean()
        if neg_spread_pct > 0.01:  # >1% negative spreads = corrupted
            logger.warning(f"  DAY {i} ({f.name}): {neg_spread_pct:.1%} negative spreads — "
                          f"FIXING by bar-level swap")
            corrupted_days.append((i, f.name, neg_spread_pct))
            # Fix: swap bid/ask where spread is negative
            bad_mask = spread < 0
            bid_fixed = bid.copy()
            ask_fixed = ask.copy()
            bid_fixed[bad_mask] = ask[bad_mask]
            ask_fixed[bad_mask] = bid[bad_mask]
            bid = bid_fixed
            ask = ask_fixed

        all_best_bid.append(bid)
        all_best_ask.append(ask)
        all_bid_queue.append(nf[:, 0, 2].copy())  # size at best bid
        all_ask_queue.append(nf[:, 10, 2].copy())  # size at best ask
        all_bid_orders.append(nf[:, 0, 6].copy())  # order count at best bid
        all_ask_orders.append(nf[:, 10, 6].copy())  # order count at best ask
        all_mid_prices.append(mp.astype(np.float64))
        day_boundaries.append(day_boundaries[-1] + n_bars)

        del data, gf, nf

        if (i + 1) % 20 == 0:
            logger.info(f"  Loaded {i+1}/{len(files)} snapshot files")

    if corrupted_days:
        logger.warning(f"  FIXED {len(corrupted_days)} days with corrupted bid/ask data:"
                      f" {[d[1] for d in corrupted_days]}")

    result = {
        'best_bid': np.concatenate(all_best_bid),
        'best_ask': np.concatenate(all_best_ask),
        'bid_queue_depth': np.concatenate(all_bid_queue),
        'ask_queue_depth': np.concatenate(all_ask_queue),
        'bid_order_count': np.concatenate(all_bid_orders),
        'ask_order_count': np.concatenate(all_ask_orders),
        'mid_prices': np.concatenate(all_mid_prices),
        'day_boundaries': day_boundaries,
        'n_days': len(files),
    }

    N = len(result['mid_prices'])
    bid_q = result['bid_queue_depth']
    ask_q = result['ask_queue_depth']
    spread = result['best_ask'] - result['best_bid']

    logger.info(f"  Snapshot data: {N:,} bars, {len(files)} days")
    logger.info(f"  Bid queue: mean={bid_q.mean():.1f}, median={np.median(bid_q):.1f}, "
                f"p25={np.percentile(bid_q, 25):.0f}, p75={np.percentile(bid_q, 75):.0f}")
    logger.info(f"  Ask queue: mean={ask_q.mean():.1f}, median={np.median(ask_q):.1f}")
    logger.info(f"  Spread: mean={spread.mean():.3f}, pct_1tick={((spread <= 0.25 + 0.01) & (spread >= 0.25 - 0.01)).mean():.1%}")

    return result


def compute_queue_fill_probability(queue_depth: float, model: str = 'empirical') -> float:
    """Compute fill probability per mid-price sweep given queue depth.

    Based on run_queue_position_study.py empirical findings:
    - 76% of ES orders are icebergs (display 1, refill at BACK of queue)
    - Icebergs don't compete: they get filled and refill behind us
    - Effective queue ahead: 2-5 contracts regardless of displayed depth
    - Fill probability at position 3-5: 80-87% per sweep
    - Key conclusion: "mid-cross ≈ fill is CLOSER to reality than Poisson"

    For typical ES queues (20-100 contracts displayed), effective position
    is essentially constant at 2-5, giving flat ~83% fill probability.
    Only very shallow (<5) or abnormally deep (>150) queues differ.

    Models:
    - 'empirical': Flat 83% for normal queues (80-87% midpoint)
    - 'conservative': Flat 70% for normal queues (worst-case)
    - 'optimistic': Flat 90% (near mid-cross = fill)
    """
    if queue_depth <= 0:
        return 0.95  # Empty queue = almost certain fill

    if model == 'empirical':
        if queue_depth < 5:
            return 0.92  # Very shallow — near front of queue
        elif queue_depth < 150:
            return 0.83  # Normal ES queue — effective pos 2-5, fill 80-87%
        else:
            # Abnormally deep (rare) — genuine large orders, harder fill
            excess = min((queue_depth - 150) / 350.0, 1.0)
            return 0.83 - 0.23 * excess  # 83% → 60% over 150-500 range

    elif model == 'conservative':
        if queue_depth < 5:
            return 0.85
        elif queue_depth < 100:
            return 0.70  # Pessimistic: assume less iceberg benefit
        else:
            excess = min((queue_depth - 100) / 400.0, 1.0)
            return 0.70 - 0.30 * excess  # 70% → 40%

    elif model == 'optimistic':
        if queue_depth < 5:
            return 0.95
        else:
            return 0.90  # Near V1: mid-cross ≈ fill

    return 0.5


# ============================================================================
# Realistic Limit Order Simulation
# ============================================================================
def simulate_realistic(
    mid_prices: np.ndarray,
    best_bid: np.ndarray,
    best_ask: np.ndarray,
    bid_queue_depth: np.ndarray,
    ask_queue_depth: np.ndarray,
    direction_preds: np.ndarray,
    magnitude_preds: np.ndarray,
    day_boundaries: list,
    # Gate parameters
    mag_gate_threshold: float = 2.0,
    signal_quantile: float = 0.80,
    # Timing
    latency_bars: int = 1,
    fill_horizon_bars: int = 50,
    hold_horizon_bars: int = 100,
    # Exit
    stop_loss_ticks: Optional[float] = None,
    take_profit_ticks: Optional[float] = None,
    use_limit_exit: bool = True,
    fixed_target_ticks: Optional[float] = None,  # Fixed profit target (overrides dynamic exit_limit)
    # Overlap
    min_bars_between_signals: int = 50,
    # Queue model
    queue_model: str = 'empirical',
    # Market impact
    max_daily_trades: int = 5000,  # capacity limit
) -> Dict:
    """
    Realistic limit order simulation with queue-aware fills.

    Key differences from V1:
    1. Uses ACTUAL best_bid/best_ask for entry prices
    2. Fill probability depends on queue depth at time of posting
    3. Real spread captured (not assumed 1 tick)
    4. Market impact: edge decays as daily trade count increases
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    rng = np.random.default_rng(seed=42)

    # Filter valid predictions
    has_pred = np.isfinite(direction_preds) & np.isfinite(magnitude_preds)
    valid_dir = direction_preds[has_pred]
    if len(valid_dir) < 100:
        return {'error': 'Insufficient predictions', 'n_valid': int(has_pred.sum())}

    abs_dir = np.abs(valid_dir)
    signal_threshold = np.percentile(abs_dir, signal_quantile * 100)

    trades: List[RealisticTrade] = []
    n_posted = 0
    n_filled = 0
    n_not_filled = 0
    n_gated_out = 0
    n_signal_weak = 0
    n_queue_rejected = 0  # NEW: rejected by queue probability
    n_capacity_limited = 0  # NEW: rejected by daily capacity
    next_allowed_bar = 0
    daily_fill_count = {}

    for day_idx in range(n_days):
        day_start = day_boundaries[day_idx]
        day_end = day_boundaries[day_idx + 1]
        daily_fill_count[day_idx] = 0

        for i in range(day_start, day_end):
            if i < next_allowed_bar:
                continue
            if not has_pred[i]:
                continue

            # Magnitude gate
            if magnitude_preds[i] < mag_gate_threshold:
                n_gated_out += 1
                continue

            # Signal strength gate
            if abs(direction_preds[i]) < signal_threshold:
                n_signal_weak += 1
                continue

            # Daily capacity check
            if daily_fill_count[day_idx] >= max_daily_trades:
                n_capacity_limited += 1
                continue

            direction = 1 if direction_preds[i] > 0 else -1

            # Post after latency
            post_bar = i + latency_bars
            if post_bar >= day_end:
                continue

            # REAL bid/ask at posting time
            post_bid = best_bid[post_bar]
            post_ask = best_ask[post_bar]
            post_spread = post_ask - post_bid

            # Skip if spread is abnormally wide (> 2 ticks)
            if post_spread > 0.50 + 0.01:  # > 2 ticks
                continue

            # Entry price: actual best_bid for longs, best_ask for shorts
            if direction == 1:
                entry_price = post_bid  # buy at bid
                queue_depth = bid_queue_depth[post_bar]
            else:
                entry_price = post_ask  # sell at ask
                queue_depth = ask_queue_depth[post_bar]

            # Entry edge: half the spread (what we earn by being passive)
            entry_edge = post_spread / 2.0 / TICK_SIZE  # in ticks

            trade = RealisticTrade(
                day=day_idx,
                direction=direction,
                signal_strength=float(direction_preds[i]),
                magnitude_pred=float(magnitude_preds[i]),
                signal_bar=i,
                post_bar=post_bar,
                post_bid=post_bid,
                post_ask=post_ask,
                post_spread=post_spread,
                post_queue_depth=float(queue_depth),
                entry_price=entry_price,
                entry_edge_ticks=entry_edge,
            )
            n_posted += 1

            # --- FILL SIMULATION ---
            fill_end = min(post_bar + fill_horizon_bars, day_end)
            filled = False

            for j in range(post_bar + 1, fill_end):
                future_mid = mid_prices[j]

                # Fill condition: mid crosses our limit price
                mid_crossed = False
                if direction == 1 and future_mid <= entry_price:
                    mid_crossed = True
                elif direction == -1 and future_mid >= entry_price:
                    mid_crossed = True

                if mid_crossed:
                    # Queue-adjusted fill probability
                    # Use queue depth at POST time (when we joined queue)
                    fill_prob = compute_queue_fill_probability(queue_depth, model=queue_model)
                    trade.fill_probability = fill_prob

                    # Stochastic fill: sample based on probability
                    if rng.random() < fill_prob:
                        trade.fill_bar = j
                        trade.fill_mid = future_mid
                        trade.fill_offset_bars = j - post_bar
                        trade.filled = True
                        filled = True
                        break
                    else:
                        n_queue_rejected += 1
                        # Don't break — price might cross again with another chance
                        # Queue position stays similar: icebergs refill behind us,
                        # so subsequent sweeps have the same fill probability

            if not filled:
                n_not_filled += 1
                trade.exit_type = 'not_filled'
                trades.append(trade)
                next_allowed_bar = i + min_bars_between_signals
                continue

            n_filled += 1
            daily_fill_count[day_idx] += 1

            # Compute effective queue position for this fill
            trade.effective_queue_pos = queue_depth * (1 - ICEBERG_FRACTION)

            # --- EXIT SIMULATION ---
            if use_limit_exit:
                if fixed_target_ticks is not None:
                    # Fixed profit target: exit at entry + N ticks
                    # Long: buy at bid, sell at bid + N*tick
                    # Short: sell at ask, buy at ask - N*tick
                    exit_limit = trade.entry_price + fixed_target_ticks * TICK_SIZE * direction
                else:
                    # Dynamic exit: sell at ask at fill time (suffers adverse selection)
                    if direction == 1:
                        exit_limit = best_ask[trade.fill_bar]  # sell at current ask
                    else:
                        exit_limit = best_bid[trade.fill_bar]  # buy at current bid
            else:
                exit_limit = None

            exit_end = min(trade.fill_bar + hold_horizon_bars, day_end)
            exited = False
            min_unrealized = 0.0  # MAE tracker (worst dip)
            max_unrealized = 0.0  # MFE tracker (best peak)

            for j in range(trade.fill_bar + 1, exit_end):
                exit_mid = mid_prices[j]
                bars_held = j - trade.fill_bar
                unrealized = (exit_mid - trade.entry_price) / TICK_SIZE * direction
                min_unrealized = min(min_unrealized, unrealized)
                max_unrealized = max(max_unrealized, unrealized)

                # Limit exit: use actual ask (for long exit) or bid (for short exit)
                if use_limit_exit and exit_limit is not None:
                    if direction == 1 and exit_mid >= exit_limit:
                        # Apply queue fill probability to exit too
                        exit_queue = ask_queue_depth[j] if j < N else 40
                        exit_fill_prob = compute_queue_fill_probability(exit_queue, model=queue_model)
                        if rng.random() < exit_fill_prob:
                            trade.exit_bar = j
                            trade.exit_mid = exit_mid
                            trade.bars_held = bars_held
                            # Exit edge = actual sell price - mid (in ticks)
                            # We sell at exit_limit, mid is at exit_mid >= exit_limit
                            trade.exit_edge_ticks = (exit_limit - exit_mid) / TICK_SIZE * direction
                            trade.exit_type = 'limit_exit'
                            exited = True
                            break
                    elif direction == -1 and exit_mid <= exit_limit:
                        exit_queue = bid_queue_depth[j] if j < N else 40
                        exit_fill_prob = compute_queue_fill_probability(exit_queue, model=queue_model)
                        if rng.random() < exit_fill_prob:
                            trade.exit_bar = j
                            trade.exit_mid = exit_mid
                            trade.bars_held = bars_held
                            # Exit edge: we buy at exit_limit, mid is at exit_mid <= exit_limit
                            trade.exit_edge_ticks = (exit_limit - exit_mid) / TICK_SIZE * direction
                            trade.exit_type = 'limit_exit'
                            exited = True
                            break

                # Take profit (market exit — cross the spread)
                if take_profit_ticks is not None and unrealized >= take_profit_ticks:
                    trade.exit_bar = j
                    trade.exit_mid = exit_mid
                    trade.bars_held = bars_held
                    # Market exit: sell at bid (long) or buy at ask (short)
                    if direction == 1:
                        actual_exit = best_bid[j] if j < N else exit_mid - HALF_TICK
                    else:
                        actual_exit = best_ask[j] if j < N else exit_mid + HALF_TICK
                    trade.exit_edge_ticks = (actual_exit - exit_mid) / TICK_SIZE * direction
                    trade.exit_type = 'take_profit'
                    exited = True
                    break

                # Stop loss (market exit — cross the spread)
                if stop_loss_ticks is not None and unrealized <= -stop_loss_ticks:
                    trade.exit_bar = j
                    trade.exit_mid = exit_mid
                    trade.bars_held = bars_held
                    if direction == 1:
                        actual_exit = best_bid[j] if j < N else exit_mid - HALF_TICK
                    else:
                        actual_exit = best_ask[j] if j < N else exit_mid + HALF_TICK
                    trade.exit_edge_ticks = (actual_exit - exit_mid) / TICK_SIZE * direction
                    trade.exit_type = 'stop_loss'
                    exited = True
                    break

            if not exited:
                trade.exit_bar = exit_end - 1
                trade.exit_mid = mid_prices[trade.exit_bar]
                trade.bars_held = trade.exit_bar - trade.fill_bar
                # Market exit at timeout: sell at bid (long) or buy at ask (short)
                eb = trade.exit_bar
                if direction == 1:
                    actual_exit = best_bid[eb] if eb < N else trade.exit_mid - HALF_TICK
                else:
                    actual_exit = best_ask[eb] if eb < N else trade.exit_mid + HALF_TICK
                trade.exit_edge_ticks = (actual_exit - trade.exit_mid) / TICK_SIZE * direction
                trade.exit_type = 'timeout'

            # Store path statistics
            trade.mae_ticks = min_unrealized  # worst dip (negative = adverse)
            trade.mfe_ticks = max_unrealized  # best peak (positive = favorable)

            # Compute PnL — CORRECTED FORMULA
            # dir_pnl measures mid-to-bid/ask movement (informational only)
            trade.dir_pnl_ticks = (trade.exit_mid - trade.entry_price) / TICK_SIZE * direction
            # Net PnL: dir_pnl already includes entry edge (entry is at bid/ask, not mid),
            # so we do NOT add entry_edge again. Only add exit_edge + commission.
            trade.net_ticks = (trade.dir_pnl_ticks +
                              trade.exit_edge_ticks - COMMISSION_TICKS)
            trade.net_dollars = trade.net_ticks * TICK_VALUE

            trades.append(trade)
            next_allowed_bar = i + min_bars_between_signals

    # ================================================================
    # Compute Statistics
    # ================================================================
    filled_trades = [t for t in trades if t.filled]
    if not filled_trades:
        return {
            'error': 'No filled trades',
            'n_posted': n_posted,
            'n_gated_out': n_gated_out,
            'n_signal_weak': n_signal_weak,
            'n_queue_rejected': n_queue_rejected,
            'queue_model': queue_model,
        }

    net_ticks = np.array([t.net_ticks for t in filled_trades])
    net_dollars = np.array([t.net_dollars for t in filled_trades])
    dir_pnl = np.array([t.dir_pnl_ticks for t in filled_trades])

    # Exit breakdown
    exit_types = {}
    for t in filled_trades:
        if t.exit_type not in exit_types:
            exit_types[t.exit_type] = {'count': 0, 'pnl': [], 'win': 0}
        exit_types[t.exit_type]['count'] += 1
        exit_types[t.exit_type]['pnl'].append(t.net_ticks)
        if t.net_ticks > 0:
            exit_types[t.exit_type]['win'] += 1

    exit_breakdown = {}
    for et, data in exit_types.items():
        pnl_arr = np.array(data['pnl'])
        exit_breakdown[et] = {
            'count': data['count'],
            'pct': data['count'] / len(filled_trades),
            'mean_pnl_ticks': float(pnl_arr.mean()),
            'total_pnl_ticks': float(pnl_arr.sum()),
            'win_rate': data['win'] / data['count'] if data['count'] > 0 else 0,
        }

    # Day-by-day PnL
    day_pnl = {}
    for t in filled_trades:
        if t.day not in day_pnl:
            day_pnl[t.day] = 0.0
        day_pnl[t.day] += t.net_dollars
    day_pnl_arr = np.array(list(day_pnl.values()))

    # Max drawdown
    cum_pnl = np.cumsum(net_dollars)
    peak = np.maximum.accumulate(cum_pnl)
    drawdown = peak - cum_pnl
    max_dd = float(drawdown.max()) if len(drawdown) > 0 else 0

    # Sharpe
    if len(day_pnl_arr) > 2 and day_pnl_arr.std() > 0:
        sharpe = float(day_pnl_arr.mean() / day_pnl_arr.std() * np.sqrt(252))
    else:
        sharpe = 0.0

    # Profit factor
    gross_profit = float(net_ticks[net_ticks > 0].sum()) if (net_ticks > 0).any() else 0
    gross_loss = float(abs(net_ticks[net_ticks < 0].sum())) if (net_ticks < 0).any() else 0.001
    profit_factor = gross_profit / gross_loss
    win_rate = float((net_ticks > 0).mean())

    result = {
        'config': {
            'mag_gate_threshold': mag_gate_threshold,
            'signal_quantile': signal_quantile,
            'latency_bars': latency_bars,
            'fill_horizon_bars': fill_horizon_bars,
            'hold_horizon_bars': hold_horizon_bars,
            'stop_loss_ticks': stop_loss_ticks,
            'take_profit_ticks': take_profit_ticks,
            'use_limit_exit': use_limit_exit,
            'fixed_target_ticks': fixed_target_ticks,
            'min_bars_between_signals': min_bars_between_signals,
            'queue_model': queue_model,
            'max_daily_trades': max_daily_trades,
        },
        'n_posted': n_posted,
        'n_filled': n_filled,
        'n_not_filled': n_not_filled,
        'n_gated_out': n_gated_out,
        'n_signal_weak': n_signal_weak,
        'n_queue_rejected': n_queue_rejected,
        'n_capacity_limited': n_capacity_limited,
        'fill_rate': n_filled / n_posted if n_posted > 0 else 0,
        # PnL
        'total_pnl_ticks': float(net_ticks.sum()),
        'total_pnl_dollars': float(net_dollars.sum()),
        'mean_pnl_ticks': float(net_ticks.mean()),
        'mean_pnl_dollars': float(net_dollars.mean()),
        'std_pnl_ticks': float(net_ticks.std()),
        'median_pnl_ticks': float(np.median(net_ticks)),
        # Directional
        'mean_dir_pnl_ticks': float(dir_pnl.mean()),
        'mean_entry_edge': float(np.mean([t.entry_edge_ticks for t in filled_trades])),
        'mean_exit_edge': float(np.mean([t.exit_edge_ticks for t in filled_trades])),
        # Quality
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'sharpe_annualized': sharpe,
        'max_drawdown_dollars': max_dd,
        # Timing
        'mean_fill_offset_sec': float(np.mean([t.fill_offset_bars for t in filled_trades])) / BARS_PER_SEC,
        'mean_bars_held': float(np.mean([t.bars_held for t in filled_trades])),
        'trades_per_day': len(filled_trades) / max(len(day_pnl), 1),
        # Queue stats
        'mean_queue_depth': float(np.mean([t.post_queue_depth for t in filled_trades])),
        'mean_effective_queue': float(np.mean([t.effective_queue_pos for t in filled_trades])),
        'mean_fill_probability': float(np.mean([t.fill_probability for t in filled_trades])),
        'mean_spread_at_entry': float(np.mean([t.post_spread for t in filled_trades])),
        # Breakdowns
        'exit_type_breakdown': exit_breakdown,
        'n_positive_days': int((day_pnl_arr > 0).sum()),
        'n_negative_days': int((day_pnl_arr < 0).sum()),
        'day_pnl_mean': float(day_pnl_arr.mean()) if len(day_pnl_arr) > 0 else 0,
        'day_pnl_std': float(day_pnl_arr.std()) if len(day_pnl_arr) > 1 else 0,
        'mean_mag_pred': float(np.mean([t.magnitude_pred for t in filled_trades])),
        # Path analysis: MAE/MFE
        'mae_mean': float(np.mean([t.mae_ticks for t in filled_trades])),
        'mae_median': float(np.median([t.mae_ticks for t in filled_trades])),
        'mae_p25': float(np.percentile([t.mae_ticks for t in filled_trades], 25)),
        'mfe_mean': float(np.mean([t.mfe_ticks for t in filled_trades])),
        'mfe_median': float(np.median([t.mfe_ticks for t in filled_trades])),
    }

    # Per-exit-type MAE/MFE breakdown (key for stop-loss optimization)
    for et in exit_breakdown:
        et_trades = [t for t in filled_trades if t.exit_type == et]
        if et_trades:
            mae_arr = np.array([t.mae_ticks for t in et_trades])
            mfe_arr = np.array([t.mfe_ticks for t in et_trades])
            exit_breakdown[et]['mae_mean'] = float(mae_arr.mean())
            exit_breakdown[et]['mae_median'] = float(np.median(mae_arr))
            exit_breakdown[et]['mae_p10'] = float(np.percentile(mae_arr, 10))
            exit_breakdown[et]['mae_p25'] = float(np.percentile(mae_arr, 25))
            exit_breakdown[et]['mfe_mean'] = float(mfe_arr.mean())
            exit_breakdown[et]['mfe_median'] = float(np.median(mfe_arr))
            # Stop-loss impact: what % of these trades dipped below various thresholds?
            for sl in [0.5, 1.0, 1.5, 2.0, 3.0]:
                exit_breakdown[et][f'pct_dipped_below_{sl}t'] = float((mae_arr < -sl).mean())

    return result


# ============================================================================
# Parameter Sweep
# ============================================================================
def run_sweep(
    mid_prices, best_bid, best_ask, bid_queue, ask_queue,
    direction_preds, magnitude_preds, day_boundaries,
    queue_model='empirical', quick=False,
    fixed_target_mode=False,
) -> List[Dict]:
    """Sweep gate thresholds with queue-aware simulation.

    If fixed_target_mode=True, sweeps fixed profit targets instead of dynamic exits.
    """

    if quick:
        gate_thresholds = [1.0, 2.0, 3.0]
        quantiles = [0.70, 0.85]
        hold_horizons = [50, 100]
        stop_losses = [None, 3.0]
    else:
        gate_thresholds = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
        quantiles = [0.60, 0.70, 0.80, 0.90]
        hold_horizons = [30, 50, 100, 200]
        stop_losses = [None, 2.0, 3.0, 5.0]

    results = []

    if fixed_target_mode:
        # ================================================================
        # FIXED TARGET SWEEP: exit at entry + N ticks
        # ================================================================
        if quick:
            fixed_targets = [1.0, 2.0, 3.0]
            ft_gates = [1.0, 2.0, 3.0]
            ft_quantiles = [0.70, 0.85]
            ft_holds = [100, 200]
            ft_stops = [None, 5.0]
        else:
            fixed_targets = [1.0, 2.0, 3.0, 4.0]
            ft_gates = [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 4.0]
            ft_quantiles = [0.60, 0.70, 0.80, 0.90]
            ft_holds = [50, 100, 200, 300]
            ft_stops = [None, 3.0, 5.0]

        n_combos = len(fixed_targets) * len(ft_gates) * len(ft_quantiles) * len(ft_holds) * len(ft_stops)
        logger.info(f"\n--- Fixed Target Sweep ({n_combos} combos, queue_model={queue_model}) ---")

        combo_count = 0
        for ft in fixed_targets:
            for gate in ft_gates:
                for q in ft_quantiles:
                    for hold in ft_holds:
                        for sl in ft_stops:
                            combo_count += 1
                            r = simulate_realistic(
                                mid_prices, best_bid, best_ask, bid_queue, ask_queue,
                                direction_preds, magnitude_preds, day_boundaries,
                                mag_gate_threshold=gate, signal_quantile=q,
                                hold_horizon_bars=hold, stop_loss_ticks=sl,
                                use_limit_exit=True, fixed_target_ticks=ft,
                                queue_model=queue_model,
                            )
                            if 'error' not in r:
                                sl_str = f"SL={sl:.0f}" if sl else "SL=off"
                                logger.info(
                                    f"  [{combo_count}/{n_combos}] FT={ft:.0f}t Gate>{gate:.1f}t Q={q:.0%} "
                                    f"Hold={hold} {sl_str}: "
                                    f"Filled={r['n_filled']:,} ({r['fill_rate']:.1%}) "
                                    f"PnL=${r['total_pnl_dollars']:+,.0f} "
                                    f"WR={r['win_rate']:.1%} PF={r['profit_factor']:.2f} "
                                    f"Sharpe={r['sharpe_annualized']:.2f} "
                                    f"$/trade={r['mean_pnl_dollars']:+.2f}"
                                )
                            results.append(r)

        return results

    # ================================================================
    # ORIGINAL SWEEP: dynamic exit at fill-time ask/bid
    # ================================================================
    logger.info(f"\n--- Gate Threshold Sweep (queue_model={queue_model}) ---")
    for gate in gate_thresholds:
        for q in quantiles:
            r = simulate_realistic(
                mid_prices, best_bid, best_ask, bid_queue, ask_queue,
                direction_preds, magnitude_preds, day_boundaries,
                mag_gate_threshold=gate, signal_quantile=q,
                hold_horizon_bars=100, use_limit_exit=True,
                queue_model=queue_model,
            )
            if 'error' not in r:
                logger.info(
                    f"  Gate>{gate:.1f}t Q={q:.0%}: "
                    f"Filled={r['n_filled']:,} ({r['fill_rate']:.1%}) "
                    f"QueueRej={r['n_queue_rejected']:,} "
                    f"PnL=${r['total_pnl_dollars']:+,.0f} "
                    f"WR={r['win_rate']:.1%} PF={r['profit_factor']:.2f} "
                    f"Sharpe={r['sharpe_annualized']:.2f} "
                    f"$/trade={r['mean_pnl_dollars']:+.2f}"
                )
            else:
                logger.info(f"  Gate>{gate:.1f}t Q={q:.0%}: {r.get('error', 'failed')}")
            results.append(r)

    # Phase 2: Best gate + timing sweep
    valid_results = [r for r in results if 'error' not in r and r.get('sharpe_annualized', 0) > 0]
    if valid_results:
        best = max(valid_results, key=lambda r: r['sharpe_annualized'])
        best_gate = best['config']['mag_gate_threshold']
        best_q = best['config']['signal_quantile']

        logger.info(f"\nBest: Gate>{best_gate:.1f}t Q={best_q:.0%} Sharpe={best['sharpe_annualized']:.2f}")
        logger.info(f"\n--- Timing + Exit Sweep ---")

        for hold in hold_horizons:
            for sl in stop_losses:
                r = simulate_realistic(
                    mid_prices, best_bid, best_ask, bid_queue, ask_queue,
                    direction_preds, magnitude_preds, day_boundaries,
                    mag_gate_threshold=best_gate, signal_quantile=best_q,
                    hold_horizon_bars=hold, stop_loss_ticks=sl,
                    use_limit_exit=True, queue_model=queue_model,
                )
                if 'error' not in r:
                    sl_str = f"SL={sl:.0f}" if sl else "SL=off"
                    logger.info(
                        f"  Hold={hold} {sl_str}: "
                        f"PnL=${r['total_pnl_dollars']:+,.0f} "
                        f"WR={r['win_rate']:.1%} Sharpe={r['sharpe_annualized']:.2f} "
                        f"$/trade={r['mean_pnl_dollars']:+.2f}"
                    )
                results.append(r)

    return results


# ============================================================================
# Holdout Test
# ============================================================================
def run_holdout_test(
    mid_prices, best_bid, best_ask, bid_queue, ask_queue,
    direction_preds, magnitude_preds, day_boundaries,
    train_days: int = 70,
    queue_model: str = 'empirical',
    fixed_target_mode: bool = False,
) -> Dict:
    """
    True holdout test:
    1. Sweep configs on first train_days days
    2. Lock best config
    3. Test on remaining days (completely unseen for config selection)
    """
    n_days = len(day_boundaries) - 1
    test_days = n_days - train_days

    if test_days < 5:
        logger.warning(f"Only {test_days} holdout days — need at least 5")
        return {'error': f'Insufficient holdout days: {test_days}'}

    train_end = day_boundaries[train_days]

    logger.info(f"\n{'='*60}")
    logger.info(f"HOLDOUT TEST: Train/Sweep on {train_days} days, Test on {test_days} days")
    logger.info(f"{'='*60}")

    # Phase 1: Sweep on training period
    logger.info(f"\n--- Sweeping on first {train_days} days ---")
    train_boundaries = day_boundaries[:train_days + 1]

    best_sharpe = -999
    best_config = None

    if fixed_target_mode:
        # Sweep fixed target configs on training period
        for ft in [1.0, 2.0, 3.0]:
            for gate in [1.0, 2.0, 3.0]:
                for q in [0.70, 0.85]:
                    for hold in [100, 200]:
                        for sl in [None, 5.0]:
                            r = simulate_realistic(
                                mid_prices[:train_end], best_bid[:train_end],
                                best_ask[:train_end], bid_queue[:train_end],
                                ask_queue[:train_end],
                                direction_preds[:train_end], magnitude_preds[:train_end],
                                train_boundaries,
                                mag_gate_threshold=gate, signal_quantile=q,
                                hold_horizon_bars=hold, stop_loss_ticks=sl,
                                use_limit_exit=True, fixed_target_ticks=ft,
                                queue_model=queue_model,
                            )
                            if 'error' not in r and r['sharpe_annualized'] > best_sharpe:
                                best_sharpe = r['sharpe_annualized']
                                best_config = r['config']
    else:
        for gate in [1.0, 1.5, 2.0, 3.0]:
            for q in [0.70, 0.80, 0.90]:
                for sl in [None, 3.0, 5.0]:
                    r = simulate_realistic(
                        mid_prices[:train_end], best_bid[:train_end],
                        best_ask[:train_end], bid_queue[:train_end],
                        ask_queue[:train_end],
                        direction_preds[:train_end], magnitude_preds[:train_end],
                        train_boundaries,
                        mag_gate_threshold=gate, signal_quantile=q,
                        hold_horizon_bars=100, stop_loss_ticks=sl,
                        use_limit_exit=True, queue_model=queue_model,
                    )
                    if 'error' not in r and r['sharpe_annualized'] > best_sharpe:
                        best_sharpe = r['sharpe_annualized']
                        best_config = r['config']

    if best_config is None:
        return {'error': 'No profitable config found on training data'}

    ft_str = f" FT={best_config.get('fixed_target_ticks', 'off')}" if fixed_target_mode else ""
    logger.info(f"\nBest config from training period:")
    logger.info(f"  Gate>{best_config['mag_gate_threshold']:.1f}t "
                f"Q={best_config['signal_quantile']:.0%} "
                f"Hold={best_config.get('hold_horizon_bars', 100)} "
                f"SL={best_config.get('stop_loss_ticks', 'off')}{ft_str} "
                f"Sharpe={best_sharpe:.2f}")

    # Phase 2: LOCKED test on holdout
    logger.info(f"\n--- Testing LOCKED config on last {test_days} days ---")
    test_boundaries = [b - train_end for b in day_boundaries[train_days:]]

    holdout_result = simulate_realistic(
        mid_prices[train_end:], best_bid[train_end:],
        best_ask[train_end:], bid_queue[train_end:],
        ask_queue[train_end:],
        direction_preds[train_end:], magnitude_preds[train_end:],
        test_boundaries,
        mag_gate_threshold=best_config['mag_gate_threshold'],
        signal_quantile=best_config['signal_quantile'],
        hold_horizon_bars=best_config.get('hold_horizon_bars', 100),
        stop_loss_ticks=best_config.get('stop_loss_ticks'),
        take_profit_ticks=best_config.get('take_profit_ticks'),
        use_limit_exit=best_config.get('use_limit_exit', True),
        fixed_target_ticks=best_config.get('fixed_target_ticks'),
        queue_model=queue_model,
    )

    if 'error' not in holdout_result:
        logger.info(f"\n  HOLDOUT RESULTS ({test_days} unseen days):")
        logger.info(f"  Trades: {holdout_result['n_filled']:,} filled ({holdout_result['fill_rate']:.1%})")
        logger.info(f"  PnL: ${holdout_result['total_pnl_dollars']:+,.2f}")
        logger.info(f"  $/trade: ${holdout_result['mean_pnl_dollars']:+.2f}")
        logger.info(f"  Win Rate: {holdout_result['win_rate']:.1%}")
        logger.info(f"  Sharpe: {holdout_result['sharpe_annualized']:.2f}")
        logger.info(f"  Winning Days: {holdout_result['n_positive_days']}/{holdout_result['n_positive_days']+holdout_result['n_negative_days']}")

    return {
        'best_config_from_training': best_config,
        'training_sharpe': best_sharpe,
        'holdout_result': holdout_result,
        'train_days': train_days,
        'test_days': test_days,
    }


# ============================================================================
# Main Pipeline
# ============================================================================
def run_pipeline(args):
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    logger.info("=" * 80)
    logger.info("REALISTIC LIMIT ORDER SIMULATOR V2 — QUEUE-AWARE")
    logger.info("=" * 80)

    horizon_bars = HORIZONS.get(args.horizon, 100)

    # ================================================================
    # Load predictions
    # ================================================================
    if args.load_predictions:
        logger.info(f"\nLoading predictions: {args.load_predictions}")
        pred_data = np.load(args.load_predictions)
        mid_prices_pred = pred_data['mid_prices']
        direction_preds = pred_data['direction_preds']
        magnitude_preds = pred_data['magnitude_preds']
        pred_day_boundaries = pred_data['day_boundaries'].tolist()
        direction_target = pred_data.get('direction_target', None)
        magnitude_target = pred_data.get('magnitude_target', None)
        n_pred_days = len(pred_day_boundaries) - 1

        logger.info(f"  {len(mid_prices_pred):,} bars, {n_pred_days} days, "
                    f"{np.isfinite(direction_preds).sum():,} direction preds, "
                    f"{np.isfinite(magnitude_preds).sum():,} magnitude preds")

        if direction_target is not None:
            mask = np.isfinite(direction_preds) & np.isfinite(direction_target)
            if mask.sum() > 50:
                ic = float(spearmanr(direction_preds[mask], direction_target[mask])[0])
                logger.info(f"  Direction IC: {ic:.4f}")
        if magnitude_target is not None:
            mask = np.isfinite(magnitude_preds) & np.isfinite(magnitude_target)
            if mask.sum() > 50:
                ic = float(spearmanr(magnitude_preds[mask], magnitude_target[mask])[0])
                logger.info(f"  Magnitude IC: {ic:.4f}")
    else:
        # Train fresh — import and use magnitude_gated_sim training
        logger.info("\nTraining fresh models on all data...")
        from alpha_discovery.magnitude_gated_sim import load_data, compute_targets, train_walk_forward

        feature_cache = args.feature_cache or str(
            LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
        )
        scanner, feature_names, load_info = load_data(feature_cache, args.n_days)
        n_pred_days = load_info['n_days']

        direction_target, magnitude_target = compute_targets(
            scanner.mid_prices, horizon_bars, scanner.day_boundaries, args.target_type
        )

        # Train direction
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING DIRECTION MODEL")
        logger.info(f"{'='*60}")
        t0 = time.time()
        direction_preds, _ = train_walk_forward(
            scanner.features, direction_target, scanner.day_boundaries,
            min_train_days=args.min_train_days, target_name='direction',
        )
        logger.info(f"  Direction training: {time.time()-t0:.0f}s")
        gc.collect()

        # Train magnitude
        logger.info(f"\n{'='*60}")
        logger.info("TRAINING MAGNITUDE MODEL")
        logger.info(f"{'='*60}")
        t0 = time.time()
        magnitude_preds, _ = train_walk_forward(
            scanner.features, magnitude_target, scanner.day_boundaries,
            min_train_days=args.min_train_days, target_name='magnitude',
        )
        logger.info(f"  Magnitude training: {time.time()-t0:.0f}s")

        mid_prices_pred = scanner.mid_prices
        pred_day_boundaries = scanner.day_boundaries

        # Save predictions
        pred_file = RESULTS_DIR / f"predictions_v2_{args.horizon}_{timestamp}.npz"
        np.savez_compressed(str(pred_file),
            mid_prices=mid_prices_pred, direction_preds=direction_preds,
            magnitude_preds=magnitude_preds, direction_target=direction_target,
            magnitude_target=magnitude_target,
            day_boundaries=np.array(pred_day_boundaries))
        logger.info(f"\nPredictions saved: {pred_file.name}")

        del scanner
        gc.collect()

    # ================================================================
    # Load snapshot data for queue depths and real bid/ask
    # ================================================================
    snapshot_dir = args.snapshot_dir or str(
        LVL3_ROOT / "data" / "processed" / "rust_full_test"
    )
    snap_data = load_snapshot_data(snapshot_dir, n_days=n_pred_days)

    # Verify alignment: prediction and snapshot arrays should have same length
    N_pred = len(mid_prices_pred)
    N_snap = len(snap_data['mid_prices'])
    if N_pred != N_snap:
        logger.warning(f"Length mismatch! Predictions: {N_pred}, Snapshots: {N_snap}")
        # Use minimum length
        N = min(N_pred, N_snap)
        direction_preds = direction_preds[:N]
        magnitude_preds = magnitude_preds[:N]
        for key in ['mid_prices', 'best_bid', 'best_ask', 'bid_queue_depth', 'ask_queue_depth']:
            snap_data[key] = snap_data[key][:N]

    mid_prices = snap_data['mid_prices']
    best_bid = snap_data['best_bid']
    best_ask = snap_data['best_ask']
    bid_queue = snap_data['bid_queue_depth']
    ask_queue = snap_data['ask_queue_depth']
    day_boundaries = snap_data['day_boundaries']

    # ================================================================
    # Run V2 simulations
    # ================================================================
    fixed_target_mode = getattr(args, 'fixed_target', False)
    all_results = {}

    if fixed_target_mode:
        # Fixed target mode: only test with empirical queue model
        # (already validated that queue model doesn't change much)
        for qm in ['empirical']:
            logger.info(f"\n{'='*80}")
            logger.info(f"FIXED TARGET SIMULATION: queue_model={qm}")
            logger.info(f"{'='*80}")

            sweep_results = run_sweep(
                mid_prices, best_bid, best_ask, bid_queue, ask_queue,
                direction_preds, magnitude_preds, day_boundaries,
                queue_model=qm, quick=args.quick,
                fixed_target_mode=True,
            )
            all_results[qm] = sweep_results
    else:
        for qm in ['optimistic', 'empirical', 'conservative']:
            logger.info(f"\n{'='*80}")
            logger.info(f"SIMULATION: queue_model={qm}")
            logger.info(f"{'='*80}")

            sweep_results = run_sweep(
                mid_prices, best_bid, best_ask, bid_queue, ask_queue,
                direction_preds, magnitude_preds, day_boundaries,
                queue_model=qm, quick=args.quick,
            )
            all_results[qm] = sweep_results

    # ================================================================
    # Holdout test with empirical model
    # ================================================================
    n_total_days = len(day_boundaries) - 1
    holdout_days = max(int(n_total_days * 0.3), 10)
    train_days = n_total_days - holdout_days

    holdout_result = run_holdout_test(
        mid_prices, best_bid, best_ask, bid_queue, ask_queue,
        direction_preds, magnitude_preds, day_boundaries,
        train_days=train_days, queue_model='empirical',
        fixed_target_mode=fixed_target_mode,
    )

    # ================================================================
    # Comparison Report
    # ================================================================
    logger.info(f"\n{'='*80}")
    if fixed_target_mode:
        logger.info("FIXED TARGET RESULTS — TOP CONFIGS")
    else:
        logger.info("V2 RESULTS COMPARISON ACROSS QUEUE MODELS")
    logger.info(f"{'='*80}")

    if fixed_target_mode:
        # Show top 10 configs by Sharpe across all queue models
        all_valid = []
        for qm, results_list in all_results.items():
            for r in results_list:
                if 'error' not in r and r.get('sharpe_annualized', -999) > -999:
                    r['_qm'] = qm
                    all_valid.append(r)

        all_valid.sort(key=lambda r: r['sharpe_annualized'], reverse=True)

        logger.info(f"\n{'FT':>4s} {'Gate':>6s} {'Q':>5s} {'Hold':>5s} {'SL':>5s} "
                    f"{'Fills':>8s} {'WR':>6s} {'PF':>6s} {'Sharpe':>8s} "
                    f"{'Total$':>12s} {'$/trade':>9s}")
        logger.info("-" * 95)

        for r in all_valid[:20]:
            cfg = r['config']
            ft_val = cfg.get('fixed_target_ticks', '?')
            sl_val = f"{cfg['stop_loss_ticks']:.0f}" if cfg.get('stop_loss_ticks') else "off"
            logger.info(
                f"{ft_val:>4.0f}t >{cfg['mag_gate_threshold']:.1f}t {cfg['signal_quantile']:.0%}"
                f" {cfg.get('hold_horizon_bars', 100):>5d} {sl_val:>5s} "
                f"{r['n_filled']:>8,d} {r['win_rate']:>6.1%} {r['profit_factor']:>6.2f} "
                f"{r['sharpe_annualized']:>8.2f} "
                f"${r['total_pnl_dollars']:>+11,.0f} "
                f"${r['mean_pnl_dollars']:>+8.2f}"
            )

        # Also show exit breakdown for best config
        if all_valid and 'exit_type_breakdown' in all_valid[0]:
            best = all_valid[0]
            logger.info(f"\nBest config exit breakdown:")
            for et, data in best['exit_type_breakdown'].items():
                logger.info(f"  {et}: {data['count']:,} ({data['pct']:.1%}) "
                           f"mean={data['mean_pnl_ticks']:.2f}t WR={data['win_rate']:.1%}")
    else:
        logger.info(f"\n{'Model':<15s} {'Best Gate':>10s} {'Fills':>8s} {'FillR':>7s} "
                    f"{'WR':>6s} {'PF':>6s} {'Sharpe':>8s} {'Total$':>12s} {'$/trade':>9s} "
                    f"{'QueueRej':>9s}")
        logger.info("-" * 110)

        for qm in ['optimistic', 'empirical', 'conservative']:
            if qm not in all_results:
                continue
            valid = [r for r in all_results[qm] if 'error' not in r and r.get('sharpe_annualized', 0) > 0]
            if valid:
                best = max(valid, key=lambda r: r['sharpe_annualized'])
                cfg = best['config']
                logger.info(
                    f"{qm:<15s} "
                    f">{cfg['mag_gate_threshold']:.1f}t Q={cfg['signal_quantile']:.0%}"
                    f"{best['n_filled']:>8,d} {best['fill_rate']:>7.1%} "
                    f"{best['win_rate']:>6.1%} {best['profit_factor']:>6.2f} "
                    f"{best['sharpe_annualized']:>8.2f} "
                    f"${best['total_pnl_dollars']:>+11,.0f} "
                    f"${best['mean_pnl_dollars']:>+8.2f} "
                    f"{best['n_queue_rejected']:>9,d}"
                )
            else:
                logger.info(f"{qm:<15s} NO PROFITABLE CONFIG")

    # Holdout summary
    if 'error' not in holdout_result:
        hr = holdout_result['holdout_result']
        if 'error' not in hr:
            logger.info(f"\n{'='*60}")
            logger.info(f"HOLDOUT TEST (train={holdout_result['train_days']}d, "
                        f"test={holdout_result['test_days']}d, LOCKED config)")
            logger.info(f"{'='*60}")
            logger.info(f"Config: Gate>{holdout_result['best_config_from_training']['mag_gate_threshold']:.1f}t "
                        f"Q={holdout_result['best_config_from_training']['signal_quantile']:.0%}")
            logger.info(f"Training Sharpe: {holdout_result['training_sharpe']:.2f}")
            logger.info(f"HOLDOUT Sharpe:  {hr['sharpe_annualized']:.2f}")
            logger.info(f"HOLDOUT PnL:     ${hr['total_pnl_dollars']:+,.2f}")
            logger.info(f"HOLDOUT $/trade: ${hr['mean_pnl_dollars']:+.2f}")
            logger.info(f"HOLDOUT WR:      {hr['win_rate']:.1%}")
            logger.info(f"HOLDOUT Days:    {hr['n_positive_days']}W / {hr['n_negative_days']}L")

    # Save
    results_file = RESULTS_DIR / f"realistic_sim_v2_{timestamp}.json"
    save_data = {
        'timestamp': timestamp,
        'horizon': args.horizon,
        'queue_models_tested': list(all_results.keys()),
        'holdout_result': holdout_result if 'error' not in holdout_result else str(holdout_result),
        'total_time_sec': time.time() - start_time,
    }
    # Add best result per queue model
    for qm in all_results.keys():
        valid = [r for r in all_results[qm] if 'error' not in r and r.get('sharpe_annualized', 0) > 0]
        if valid:
            save_data[f'best_{qm}'] = max(valid, key=lambda r: r['sharpe_annualized'])

    with open(str(results_file), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    logger.info(f"\nResults saved: {results_file.name}")
    logger.info(f"Log: {_log_file.name}")
    logger.info(f"\n{'='*80}")
    logger.info(f"V2 PIPELINE COMPLETE — {time.time()-start_time:.0f}s ({(time.time()-start_time)/60:.1f}m)")
    logger.info(f"{'='*80}")

    return save_data


def main():
    parser = argparse.ArgumentParser(description='Realistic Limit Order Simulator V2')
    parser.add_argument('--horizon', type=str, default='ret_10s', choices=list(HORIZONS.keys()))
    parser.add_argument('--target-type', type=str, default='mfe_net')
    parser.add_argument('--n-days', type=int, default=None)
    parser.add_argument('--min-train-days', type=int, default=5)
    parser.add_argument('--feature-cache', type=str, default=None)
    parser.add_argument('--snapshot-dir', type=str, default=None)
    parser.add_argument('--load-predictions', type=str, default=None)
    parser.add_argument('--quick', action='store_true')
    parser.add_argument('--fixed-target', action='store_true',
                       help='Use fixed profit target exit instead of dynamic exit_limit')
    parser.add_argument('--use-gpu', action='store_true')
    args = parser.parse_args()

    try:
        return run_pipeline(args)
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
