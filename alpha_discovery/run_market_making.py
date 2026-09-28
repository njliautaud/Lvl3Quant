"""
Market Making Simulation — Earn the Spread Instead of Paying It

Instead of directional trading (which pays ~$12.50 round-trip), this simulates
a market making strategy that EARNS the spread by posting limit orders at best
bid and best ask.

Two strategies compared:
  1. Naive MM: Always post at best_bid / best_ask, earn the spread on fills
  2. Informed MM: Use direction model to skew quotes, reducing adverse selection

Key insight:
  - Naive MM earns ~$12.50-$25 per round-trip (spread)
  - But adverse selection (getting filled right before move) erodes that
  - Informed MM reduces adverse selection by adjusting quote width/skew

Architecture:
  - Walk bar-by-bar through data
  - At each bar, model decides quote offset (0, +1, +2 ticks from best)
  - A "fill" event: mid price crosses through quote level
  - Match buys with sells: gross spread earned = bid_fill_price - ask_fill_price (in ticks)
  - Track inventory, PnL, adverse selection per fill

Simplified fill model:
  - At each bar, we're always quoting at best_bid and best_ask (1-tick spread typical)
  - A buy fill: mid moves down to touch our bid quote
  - A sell fill: mid moves up to touch our ask quote
  - Matched pair: round-trip profit = spread earned - adverse selection
  - Unmatched fills: marked to market (inventory risk)

Usage:
    python alpha_discovery/run_market_making.py
    python alpha_discovery/run_market_making.py --fast
    python alpha_discovery/run_market_making.py --quote-offsets 0 1 2
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
from typing import Dict, List, Optional, Tuple, NamedTuple

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
        logging.FileHandler(RESULTS_DIR / 'market_making.log', mode='a', encoding='utf-8'),
    ]
)
logger = logging.getLogger("market_making")

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE = 0.25
TICK_VALUE = 12.50          # $12.50 per tick per contract
ES_POINT_VALUE = 50.0       # $50 per point
BARS_PER_SEC = 10           # 100ms intervals
ROUND_TRIP_TICKS = 1.0      # 1-tick round trip cost baseline (for comparison)
ROUND_TRIP_COST = ROUND_TRIP_TICKS * TICK_VALUE  # $12.50

# Max inventory before we halt quoting (risk limit)
MAX_INVENTORY_CONTRACTS = 5

# Min hold between fills (to avoid double-counting rapid fills)
MIN_FILL_GAP_BARS = 5  # 0.5s


# ============================================================================
# WALK-FORWARD DIRECTION PREDICTIONS
# ============================================================================

def get_direction_predictions(
    scanner: MBOAlphaScanner,
    target: np.ndarray,
    min_train_days: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Walk-forward LightGBM predictions for direction, aligned to all bars.

    Returns (full_predictions, full_actuals) arrays of length N (all bars),
    with NaN where the bar was in a training fold (no OOS prediction).

    STRICTLY CAUSAL: uses only EXCLUDE_FEATURES_DIRECTION-filtered features,
    train on days 0..D-2, predict on day D.
    """
    N = len(scanner.mid_prices)

    # Feature matrix (direction-appropriate features only)
    all_names = scanner.feature_names
    keep_mask = np.array([fn not in EXCLUDE_FEATURES_DIRECTION for fn in all_names])
    features_use = scanner.features[:, keep_mask]
    feature_names_use = [fn for fn in all_names if fn not in EXCLUDE_FEATURES_DIRECTION]

    logger.info(f"  Direction model: {features_use.shape[1]} features")

    result = walk_forward_evaluate(
        features=features_use,
        target=target,
        day_boundaries=scanner.day_boundaries,
        feature_names=feature_names_use,
        model_type='lgbm',
        min_train_days=min_train_days,
    )

    if 'error' in result:
        logger.error(f"  walk_forward failed: {result['error']}")
        return np.full(N, np.nan), np.full(N, np.nan)

    # Rebuild full prediction array aligned to bar indices
    # walk_forward_evaluate returns all_preds and fold_metrics with day indices
    # We need to reconstruct bar-level alignment from day boundaries
    full_preds = np.full(N, np.nan, dtype=np.float32)
    full_acts = np.full(N, np.nan, dtype=np.float32)

    n_days = len(scanner.day_boundaries) - 1
    pred_cursor = 0
    pred_array = result.get('all_preds_concat', None)
    # walk_forward_evaluate does not return per-bar predictions natively;
    # we use the fold_metrics to reconstruct day-level info.
    # Instead we re-run a lean version that gives us per-bar output.

    # Use a simpler re-extraction from the returned fold structure
    # The walk_forward_evaluate in run_model_refinement doesn't expose raw preds.
    # We call it again with a wrapper that captures them.
    import lightgbm as lgb

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

    for test_day in range(min_train_days, n_days):
        train_start = scanner.day_boundaries[0]
        train_end = scanner.day_boundaries[test_day]
        test_start = scanner.day_boundaries[test_day]
        test_end = scanner.day_boundaries[test_day + 1]

        X_train = features_use[train_start:train_end]
        y_train = target[train_start:train_end]
        X_test = features_use[test_start:test_end]
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
            logger.warning(f"  Day {test_day} failed: {e}")
            continue

        preds = model.predict(X_te)

        # Map back to full array
        test_bar_indices = np.where(test_valid)[0] + test_start
        for bar_i, pred_val, act_val in zip(test_bar_indices, preds, y_te):
            full_preds[bar_i] = float(pred_val)
            full_acts[bar_i] = float(act_val)

        del model
        gc.collect()

    n_oos = np.isfinite(full_preds).sum()
    if n_oos > 0:
        ic = float(spearmanr(
            full_preds[np.isfinite(full_preds)],
            full_acts[np.isfinite(full_acts)]
        )[0])
        logger.info(f"  Direction model: {n_oos:,} OOS predictions, overall IC={ic:.5f}")

    return full_preds, full_acts


# ============================================================================
# NAIVE MARKET MAKING SIMULATION
# ============================================================================

def simulate_naive_mm(
    mid_prices: np.ndarray,
    spread_arr: np.ndarray,
    day_boundaries: list,
    quote_offset_ticks: int = 0,
    contracts: int = 1,
) -> dict:
    """
    Naive market making: always post at best_bid and best_ask.

    Fill model:
      - Bid fill (we buy): mid moves DOWN to touch our bid level (best_bid - quote_offset*tick)
        This means mid crosses below our quote -> we get filled as buyer
      - Ask fill (we sell): mid moves UP to touch our ask level (best_ask + quote_offset*tick)

    PnL per matched pair (one buy fill + one sell fill):
      gross_spread = ask_fill_price - bid_fill_price
      adverse_sel = mark_to_market at position exit vs entry
      net_pnl = gross_spread - adverse_sel - transaction_cost

    Simplified: we track inventory and mark-to-market continuously.

    Parameters:
      quote_offset_ticks: how many ticks WIDER than best to quote
        0 = at best (most aggressive, most fills, most adverse selection)
        1 = 1 tick wider (fewer fills, less adverse selection)
        2 = 2 ticks wider (fewest fills, least adverse selection)
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    if N < 100:
        return {'error': 'Too few bars'}

    # Fill detection:
    # A bar causes a fill when mid moves through our quote level
    # Quote level = mid +/- (half_spread + quote_offset * tick)
    # For ES with 1-tick spread: half_spread = 0.125
    half_spread = spread_arr / 2.0
    bid_quote = mid_prices - half_spread - quote_offset_ticks * TICK_SIZE
    ask_quote = mid_prices + half_spread + quote_offset_ticks * TICK_SIZE

    # Detect bid fills: next bar mid drops below current bid_quote
    # Detect ask fills: next bar mid rises above current ask_quote
    mid_next = np.empty(N, dtype=np.float64)
    mid_next[:N - 1] = mid_prices[1:]
    mid_next[N - 1] = np.nan

    bid_fill = np.zeros(N, dtype=bool)
    ask_fill = np.zeros(N, dtype=bool)
    bid_fill[:N - 1] = mid_next[:N - 1] < bid_quote[:N - 1]
    ask_fill[:N - 1] = mid_next[:N - 1] > ask_quote[:N - 1]

    # Mask fills at day boundaries (can't carry over)
    for d in range(n_days - 1):
        day_end = day_boundaries[d + 1] - 1
        if day_end < N:
            bid_fill[day_end] = False
            ask_fill[day_end] = False

    # Simulate walk through bars with inventory tracking
    inventory = 0           # net position (positive = long, negative = short)
    inventory_entry_price = 0.0
    cumulative_realized_pnl = 0.0
    n_buy_fills = 0
    n_sell_fills = 0
    n_matched_pairs = 0
    spread_earned = 0.0
    adverse_sel_total = 0.0
    daily_pnl_map: Dict[int, float] = {}
    last_fill_bar = -MIN_FILL_GAP_BARS

    for i in range(N - 1):
        if not (np.isfinite(mid_prices[i]) and np.isfinite(spread_arr[i])):
            continue

        # Check inventory limit — stop quoting if too large
        if abs(inventory) >= MAX_INVENTORY_CONTRACTS:
            continue

        # Enforce min gap between fills to prevent double-counting
        if i - last_fill_bar < MIN_FILL_GAP_BARS:
            continue

        current_mid = mid_prices[i]
        next_mid = mid_prices[i + 1] if i + 1 < N else current_mid
        this_half_spread = spread_arr[i] / 2.0

        # Find which day this is in
        day_idx = 0
        for d in range(n_days):
            if day_boundaries[d] <= i < day_boundaries[d + 1]:
                day_idx = d
                break

        if bid_fill[i] and inventory >= 0:  # only buy if not already short
            # Buy fill at bid_quote[i]
            fill_price = bid_quote[i]
            if inventory == 0:
                inventory_entry_price = fill_price
            else:
                # Average into position
                total_pos = inventory + contracts
                inventory_entry_price = (
                    inventory_entry_price * inventory + fill_price * contracts
                ) / total_pos
            inventory += contracts
            n_buy_fills += 1
            last_fill_bar = i

            # Record adverse selection: how far did mid move AGAINST us after buy?
            # (Mid moving down after we buy is adverse)
            adv_sel = max(0.0, fill_price - next_mid)
            adverse_sel_total += adv_sel * ES_POINT_VALUE * contracts

        elif ask_fill[i] and inventory <= 0:  # only sell if not already long
            # Sell fill at ask_quote[i]
            fill_price = ask_quote[i]
            if inventory == 0:
                inventory_entry_price = fill_price
            else:
                total_pos = abs(inventory) + contracts
                inventory_entry_price = (
                    inventory_entry_price * abs(inventory) + fill_price * contracts
                ) / total_pos
            inventory -= contracts
            n_sell_fills += 1
            last_fill_bar = i

            # Adverse selection: mid moving up after we sell
            adv_sel = max(0.0, next_mid - fill_price)
            adverse_sel_total += adv_sel * ES_POINT_VALUE * contracts

        # Close matched inventory pairs (1 buy + 1 sell = 1 round trip)
        # When inventory crosses zero, we've completed a round trip
        if inventory == 0 and (n_buy_fills > 0 or n_sell_fills > 0):
            # Realize PnL from last pair
            n_matched_pairs += 1

        # Mark-to-market P&L at day end (force flatten)
        day_end_flag = False
        if d < n_days - 1 and i == day_boundaries[d + 1] - 1:
            day_end_flag = True
        if day_end_flag and inventory != 0:
            # Force close at current mid
            mtm_pnl = inventory * (current_mid - inventory_entry_price) * ES_POINT_VALUE
            cumulative_realized_pnl += mtm_pnl
            daily_pnl_map[day_idx] = daily_pnl_map.get(day_idx, 0.0) + mtm_pnl
            inventory = 0
            inventory_entry_price = 0.0

    n_fills = n_buy_fills + n_sell_fills
    n_pairs_approx = min(n_buy_fills, n_sell_fills)

    # Gross spread earned: approximate from fill prices
    # Each matched pair earns: ask_fill_price - bid_fill_price (roughly the spread)
    mean_spread = float(np.nanmean(spread_arr))
    gross_per_pair = mean_spread * ES_POINT_VALUE  # $50/pt * spread_in_pts
    gross_spread_total = n_pairs_approx * gross_per_pair

    # Net PnL (approximate)
    net_pnl_approx = gross_spread_total - adverse_sel_total

    daily_pnl_arr = np.array(list(daily_pnl_map.values())) if daily_pnl_map else np.array([0.0])
    fills_per_day = n_fills / n_days if n_days > 0 else 0.0
    pairs_per_day = n_pairs_approx / n_days if n_days > 0 else 0.0
    gross_per_day = gross_spread_total / n_days if n_days > 0 else 0.0
    adverse_per_day = adverse_sel_total / n_days if n_days > 0 else 0.0
    net_per_day = net_pnl_approx / n_days if n_days > 0 else 0.0

    sharpe = 0.0
    if len(daily_pnl_arr) > 2 and np.std(daily_pnl_arr) > 0:
        sharpe = float(np.mean(daily_pnl_arr) / np.std(daily_pnl_arr) * np.sqrt(252))

    return {
        'quote_offset_ticks': quote_offset_ticks,
        'n_days': n_days,
        'n_buy_fills': n_buy_fills,
        'n_sell_fills': n_sell_fills,
        'n_total_fills': n_fills,
        'fills_per_day': round(fills_per_day, 1),
        'n_matched_pairs': n_pairs_approx,
        'pairs_per_day': round(pairs_per_day, 1),
        'mean_spread_pts': round(mean_spread, 5),
        'gross_per_pair': round(gross_per_pair, 2),
        'gross_spread_total': round(gross_spread_total, 2),
        'adverse_sel_total': round(adverse_sel_total, 2),
        'net_pnl_approx': round(net_pnl_approx, 2),
        'gross_per_day': round(gross_per_day, 2),
        'adverse_per_day': round(adverse_per_day, 2),
        'net_per_day': round(net_per_day, 2),
        'sharpe_from_mtm': round(sharpe, 2),
        'adverse_sel_pct_of_gross': round(
            adverse_sel_total / gross_spread_total * 100 if gross_spread_total > 0 else 0.0, 1
        ),
    }


# ============================================================================
# INFORMED MARKET MAKING SIMULATION
# ============================================================================

def simulate_informed_mm(
    mid_prices: np.ndarray,
    spread_arr: np.ndarray,
    direction_preds: np.ndarray,
    day_boundaries: list,
    pred_threshold_pct: float = 70.0,
    quote_skew_ticks: int = 1,
    contracts: int = 1,
) -> dict:
    """
    Informed market making using direction model to reduce adverse selection.

    Strategy:
      - If direction model predicts STRONG UP move (>threshold):
        Pull bid (don't buy), keep ask (or narrow ask slightly)
        i.e., skew: bid moves up by skew_ticks (we don't want to buy into a move)
      - If direction model predicts STRONG DOWN move:
        Pull ask, keep bid
      - If model is neutral: quote symmetrically at best

    Quote skewing:
      Normal:     bid = mid - half_spread, ask = mid + half_spread
      Bullish:    bid = mid - half_spread + skew*tick  (bid moves up = harder to buy)
                  ask = mid + half_spread - skew*tick  (ask moves down = easier to sell)
                  Net effect: we lean short (want to sell into strength)
      Bearish:    opposite

    Fill detection is same as naive MM, but with modified quote levels.

    This is "toxic flow avoidance" — the core MM alpha strategy.
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    if N < 100:
        return {'error': 'Too few bars'}

    # Compute signal threshold from predictions
    valid_preds = direction_preds[np.isfinite(direction_preds)]
    if len(valid_preds) < 100:
        return {'error': 'Too few valid direction predictions'}

    abs_preds = np.abs(valid_preds)
    pred_threshold_val = np.percentile(abs_preds, pred_threshold_pct)

    # Classify each bar's signal
    # +1 = bullish (predicted up), -1 = bearish (predicted down), 0 = neutral
    signal_strength = np.where(
        np.isfinite(direction_preds),
        direction_preds,
        0.0,
    )
    is_strong = np.abs(signal_strength) > pred_threshold_val
    signal_dir = np.where(is_strong, np.sign(signal_strength), 0.0)

    # Compute quote levels with skew
    half_spread = spread_arr / 2.0

    # Skew: if bullish, raise bid (avoid being bought) and lower ask (eager to sell)
    # If bearish, lower ask (avoid being sold) and raise bid (eager to buy)
    bid_skew = signal_dir * quote_skew_ticks * TICK_SIZE    # positive skew = raise bid (avoid buys when bullish)
    ask_skew = -signal_dir * quote_skew_ticks * TICK_SIZE   # negative skew = lower ask (avoid sells when bearish)

    bid_quote = mid_prices - half_spread + bid_skew
    ask_quote = mid_prices + half_spread + ask_skew

    # Fill detection
    mid_next = np.empty(N, dtype=np.float64)
    mid_next[:N - 1] = mid_prices[1:]
    mid_next[N - 1] = np.nan

    bid_fill = np.zeros(N, dtype=bool)
    ask_fill = np.zeros(N, dtype=bool)
    bid_fill[:N - 1] = mid_next[:N - 1] < bid_quote[:N - 1]
    ask_fill[:N - 1] = mid_next[:N - 1] > ask_quote[:N - 1]

    # Mask at day boundaries
    for d in range(n_days - 1):
        day_end = day_boundaries[d + 1] - 1
        if day_end < N:
            bid_fill[day_end] = False
            ask_fill[day_end] = False

    # Simulate with inventory tracking
    inventory = 0
    inventory_entry_price = 0.0
    cumulative_realized_pnl = 0.0
    n_buy_fills = 0
    n_sell_fills = 0
    n_avoided_buys = 0    # times we would have bought but model said bearish
    n_avoided_sells = 0   # times we would have sold but model said bullish
    spread_earned = 0.0
    adverse_sel_total = 0.0
    daily_pnl_map: Dict[int, float] = {}
    last_fill_bar = -MIN_FILL_GAP_BARS

    # Also compute naive fills for comparison
    naive_bid_fill = mid_next[:N - 1] < (mid_prices - half_spread)[:N - 1]
    naive_ask_fill = mid_next[:N - 1] > (mid_prices + half_spread)[:N - 1]
    n_naive_buy_fills = int(naive_bid_fill.sum())
    n_naive_ask_fills = int(naive_ask_fill.sum())

    for i in range(N - 1):
        if not (np.isfinite(mid_prices[i]) and np.isfinite(spread_arr[i])):
            continue
        if not np.isfinite(direction_preds[i]):
            continue

        if abs(inventory) >= MAX_INVENTORY_CONTRACTS:
            continue
        if i - last_fill_bar < MIN_FILL_GAP_BARS:
            continue

        current_mid = mid_prices[i]
        next_mid = mid_prices[i + 1]

        day_idx = 0
        for d in range(n_days):
            if day_boundaries[d] <= i < day_boundaries[d + 1]:
                day_idx = d
                break

        current_signal = signal_dir[i]

        # Informed bid fill: only accept buy fills when model is not strongly bullish
        # (Avoiding: getting long right before a big up move = adverse selection avoided)
        if bid_fill[i] and inventory >= 0:
            # Check if model warned us away from this fill
            if current_signal > 0:
                # Bullish signal: our bid was raised by skew, so actual fill
                # happens less often (model is protecting us)
                # If we DO get filled, count it as potential adverse sel
                n_avoided_buys += 1  # we would have been in at lower bid but model raised it

            fill_price = bid_quote[i]
            if inventory == 0:
                inventory_entry_price = fill_price
            else:
                total = inventory + contracts
                inventory_entry_price = (inventory_entry_price * inventory + fill_price * contracts) / total
            inventory += contracts
            n_buy_fills += 1
            last_fill_bar = i

            adv_sel = max(0.0, fill_price - next_mid)
            adverse_sel_total += adv_sel * ES_POINT_VALUE * contracts

        elif ask_fill[i] and inventory <= 0:
            if current_signal < 0:
                n_avoided_sells += 1

            fill_price = ask_quote[i]
            if inventory == 0:
                inventory_entry_price = fill_price
            else:
                total = abs(inventory) + contracts
                inventory_entry_price = (inventory_entry_price * abs(inventory) + fill_price * contracts) / total
            inventory -= contracts
            n_sell_fills += 1
            last_fill_bar = i

            adv_sel = max(0.0, next_mid - fill_price)
            adverse_sel_total += adv_sel * ES_POINT_VALUE * contracts

        # Day end: force flatten
        if d < n_days - 1 and i == day_boundaries[d + 1] - 1:
            if inventory != 0:
                mtm_pnl = inventory * (current_mid - inventory_entry_price) * ES_POINT_VALUE
                cumulative_realized_pnl += mtm_pnl
                daily_pnl_map[day_idx] = daily_pnl_map.get(day_idx, 0.0) + mtm_pnl
                inventory = 0
                inventory_entry_price = 0.0

    n_fills = n_buy_fills + n_sell_fills
    n_pairs = min(n_buy_fills, n_sell_fills)
    mean_spread = float(np.nanmean(spread_arr))
    gross_per_pair = mean_spread * ES_POINT_VALUE
    gross_spread_total = n_pairs * gross_per_pair
    net_pnl_approx = gross_spread_total - adverse_sel_total

    fills_per_day = n_fills / n_days if n_days > 0 else 0.0
    pairs_per_day = n_pairs / n_days if n_days > 0 else 0.0
    gross_per_day = gross_spread_total / n_days if n_days > 0 else 0.0
    adverse_per_day = adverse_sel_total / n_days if n_days > 0 else 0.0
    net_per_day = net_pnl_approx / n_days if n_days > 0 else 0.0

    daily_pnl_arr = np.array(list(daily_pnl_map.values())) if daily_pnl_map else np.array([0.0])
    sharpe = 0.0
    if len(daily_pnl_arr) > 2 and np.std(daily_pnl_arr) > 0:
        sharpe = float(np.mean(daily_pnl_arr) / np.std(daily_pnl_arr) * np.sqrt(252))

    # Adverse selection reduction vs naive
    naive_fills = n_naive_buy_fills + n_naive_ask_fills
    fill_reduction_pct = (naive_fills - n_fills) / naive_fills * 100 if naive_fills > 0 else 0.0

    return {
        'pred_threshold_pct': pred_threshold_pct,
        'pred_threshold_val': float(pred_threshold_val),
        'quote_skew_ticks': quote_skew_ticks,
        'n_days': n_days,
        'n_buy_fills': n_buy_fills,
        'n_sell_fills': n_sell_fills,
        'n_total_fills': n_fills,
        'fills_per_day': round(fills_per_day, 1),
        'n_matched_pairs': n_pairs,
        'pairs_per_day': round(pairs_per_day, 1),
        'n_avoided_buys': n_avoided_buys,
        'n_avoided_sells': n_avoided_sells,
        'n_avoided_total': n_avoided_buys + n_avoided_sells,
        'fill_reduction_pct_vs_naive': round(fill_reduction_pct, 1),
        'mean_spread_pts': round(mean_spread, 5),
        'gross_per_pair': round(gross_per_pair, 2),
        'gross_spread_total': round(gross_spread_total, 2),
        'adverse_sel_total': round(adverse_sel_total, 2),
        'net_pnl_approx': round(net_pnl_approx, 2),
        'gross_per_day': round(gross_per_day, 2),
        'adverse_per_day': round(adverse_per_day, 2),
        'net_per_day': round(net_per_day, 2),
        'sharpe_from_mtm': round(sharpe, 2),
        'adverse_sel_pct_of_gross': round(
            adverse_sel_total / gross_spread_total * 100 if gross_spread_total > 0 else 0.0, 1
        ),
    }


# ============================================================================
# ADVERSE SELECTION ANALYSIS
# ============================================================================

def analyze_adverse_selection(
    mid_prices: np.ndarray,
    spread_arr: np.ndarray,
    day_boundaries: list,
    direction_preds: np.ndarray,
) -> dict:
    """
    Analyze adverse selection rates across different model signal regimes.

    For each fill type (buy/sell), measure: how much does mid move against us
    over the next 1s, 3s, 5s, 10s after the fill?

    Key question: does the direction model identify HIGH adverse selection fills?
    """
    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    half_spread = spread_arr / 2.0
    mid_next = np.empty(N, dtype=np.float64)
    mid_next[:N - 1] = mid_prices[1:]
    mid_next[N - 1] = np.nan

    # Detect all potential fills at best quotes (quote_offset=0)
    bid_fill_mask = np.zeros(N, dtype=bool)
    ask_fill_mask = np.zeros(N, dtype=bool)
    bid_fill_mask[:N - 1] = mid_next[:N - 1] < (mid_prices - half_spread)[:N - 1]
    ask_fill_mask[:N - 1] = mid_next[:N - 1] > (mid_prices + half_spread)[:N - 1]

    # Mask day boundaries
    for d in range(n_days - 1):
        day_end = day_boundaries[d + 1] - 1
        if day_end < N:
            bid_fill_mask[day_end] = False
            ask_fill_mask[day_end] = False

    # For each fill, compute adverse selection over various horizons
    adv_sel_horizons = {
        '1s': BARS_PER_SEC * 1,
        '3s': BARS_PER_SEC * 3,
        '5s': BARS_PER_SEC * 5,
        '10s': BARS_PER_SEC * 10,
        '30s': BARS_PER_SEC * 30,
    }

    results = {}

    # Classify fills by model signal strength
    valid_preds = direction_preds[np.isfinite(direction_preds)]
    if len(valid_preds) < 100:
        return {'error': 'Too few direction predictions for adverse selection analysis'}

    abs_preds = np.abs(valid_preds)
    pct_50 = np.percentile(abs_preds, 50)
    pct_80 = np.percentile(abs_preds, 80)
    pct_95 = np.percentile(abs_preds, 95)

    def signal_regime(pred_val):
        """Classify signal into regime."""
        if not np.isfinite(pred_val):
            return 'unknown'
        ap = abs(pred_val)
        if ap > pct_95:
            return 'extreme'
        elif ap > pct_80:
            return 'strong'
        elif ap > pct_50:
            return 'moderate'
        else:
            return 'weak'

    # For buy fills: adverse selection = (fill_price - future_mid) if future_mid < fill_price
    # For sell fills: adverse selection = (future_mid - fill_price) if future_mid > fill_price
    fill_data = []

    for i in range(N - BARS_PER_SEC * 30 - 1):
        if not (np.isfinite(mid_prices[i]) and np.isfinite(spread_arr[i])):
            continue

        fill_type = None
        fill_price = 0.0
        if bid_fill_mask[i]:
            fill_type = 'buy'
            fill_price = mid_prices[i] - half_spread[i]
        elif ask_fill_mask[i]:
            fill_type = 'sell'
            fill_price = mid_prices[i] + half_spread[i]

        if fill_type is None:
            continue

        # Compute adverse selection at each horizon
        adv_at_hz = {}
        for hz_name, hz_bars in adv_sel_horizons.items():
            future_bar = i + hz_bars
            if future_bar >= N or not np.isfinite(mid_prices[future_bar]):
                continue
            future_mid = mid_prices[future_bar]
            if fill_type == 'buy':
                adv = max(0.0, fill_price - future_mid)
            else:
                adv = max(0.0, future_mid - fill_price)
            adv_at_hz[hz_name] = float(adv * ES_POINT_VALUE)

        model_signal = float(direction_preds[i]) if np.isfinite(direction_preds[i]) else 0.0
        regime = signal_regime(model_signal)

        # Signal alignment: for buy fill, bullish signal = aligned; bearish = misaligned
        if fill_type == 'buy':
            aligned = model_signal > 0
        else:
            aligned = model_signal < 0

        fill_data.append({
            'bar': i,
            'fill_type': fill_type,
            'model_signal': model_signal,
            'regime': regime,
            'aligned': aligned,
            'adv_sel': adv_at_hz,
        })

    if not fill_data:
        return {'error': 'No fill events detected'}

    # Aggregate by regime and fill type
    for fill_type in ['buy', 'sell']:
        ft_data = [f for f in fill_data if f['fill_type'] == fill_type]
        if not ft_data:
            continue

        results[f'{fill_type}_fills'] = {
            'total': len(ft_data),
        }

        for regime in ['weak', 'moderate', 'strong', 'extreme']:
            regime_data = [f for f in ft_data if f['regime'] == regime]
            if len(regime_data) < 5:
                continue

            hz_stats = {}
            for hz_name in adv_sel_horizons:
                adv_vals = [f['adv_sel'].get(hz_name, 0.0) for f in regime_data if hz_name in f['adv_sel']]
                if adv_vals:
                    hz_stats[hz_name] = {
                        'mean_adv_sel': round(float(np.mean(adv_vals)), 3),
                        'pct_adverse': round(float((np.array(adv_vals) > 0).mean()), 4),
                    }

            results[f'{fill_type}_fills'][f'regime_{regime}'] = {
                'n': len(regime_data),
                'pct_aligned': round(float(np.mean([f['aligned'] for f in regime_data])), 4),
                'by_horizon': hz_stats,
            }

    return results


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Market making simulation')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: fewer configs, skip adverse selection analysis')
    parser.add_argument('--quote-offsets', nargs='+', type=int, default=[0, 1, 2],
                        help='Quote offset ticks to test for naive MM')
    parser.add_argument('--skew-ticks', nargs='+', type=int, default=[1, 2],
                        help='Quote skew ticks for informed MM')
    parser.add_argument('--pred-thresholds', nargs='+', type=float, default=[60.0, 75.0, 90.0],
                        help='Signal strength percentile thresholds for informed MM')
    parser.add_argument('--min-train-days', type=int, default=3)
    parser.add_argument('--direction-target', default='ret_3s',
                        help='Return target for direction model (default: ret_3s)')
    args = parser.parse_args()

    logger.info("=" * 75)
    logger.info("MARKET MAKING SIMULATION")
    logger.info(f"  Quote offsets (naive): {args.quote_offsets}")
    logger.info(f"  Skew ticks (informed): {args.skew_ticks}")
    logger.info(f"  Pred thresholds: {args.pred_thresholds}")
    logger.info(f"  Direction target: {args.direction_target}")
    logger.info("=" * 75)

    t_start = time.time()

    # -----------------------------------------------------------------------
    # Load data
    # -----------------------------------------------------------------------
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    stats = load_feature_cache(scanner)
    if stats is None:
        logger.info("No feature cache, computing from scratch (slow)...")
        stats = scanner.load_from_cache()

    logger.info(f"Data: {len(scanner.mid_prices):,} snapshots, "
                f"{len(scanner.day_boundaries) - 1} days, "
                f"{len(scanner.feature_names)} features")

    # -----------------------------------------------------------------------
    # Extract spread from features
    # -----------------------------------------------------------------------
    feat_name_to_idx = {n: i for i, n in enumerate(scanner.feature_names)}
    if 'spread' in feat_name_to_idx:
        spread_arr = scanner.features[:, feat_name_to_idx['spread']].astype(np.float64)
        logger.info(f"Using 'spread' feature (mean={np.nanmean(spread_arr):.5f})")
    else:
        # Fallback: assume 1-tick spread (typical for ES during RTH)
        spread_arr = np.full(len(scanner.mid_prices), TICK_SIZE, dtype=np.float64)
        logger.info("Using constant 1-tick spread (spread feature not found)")

    # Replace invalid spread values with median
    spread_median = np.nanmedian(spread_arr)
    spread_arr = np.where(spread_arr > 0, spread_arr, spread_median)
    logger.info(f"Spread: mean={np.nanmean(spread_arr):.5f} median={spread_median:.5f} "
                f"std={np.nanstd(spread_arr):.5f}")

    # -----------------------------------------------------------------------
    # Compute direction target and walk-forward predictions
    # -----------------------------------------------------------------------
    logger.info(f"\nComputing direction target ({args.direction_target})...")
    hz_sec_map = {
        'ret_1s': {'1s': 1}, 'ret_3s': {'3s': 3}, 'ret_5s': {'5s': 5},
        'ret_10s': {'10s': 10}, 'ret_15s': {'15s': 15}, 'ret_30s': {'30s': 30},
    }
    hz_dict = hz_sec_map.get(args.direction_target, {'3s': 3})
    all_targets = compute_return_targets(
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        sample_interval_ms=100,
        horizons_sec=hz_dict,
        include_flow_target=False,
    )
    direction_target = list(all_targets.values())[0]

    logger.info(f"Running walk-forward direction model ({args.direction_target})...")
    t0 = time.time()
    direction_preds, direction_acts = get_direction_predictions(
        scanner=scanner,
        target=direction_target,
        min_train_days=args.min_train_days,
    )
    logger.info(f"Direction model complete in {time.time() - t0:.0f}s")

    # -----------------------------------------------------------------------
    # Naive MM simulation
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 65)
    logger.info("NAIVE MARKET MAKING (no model-based adjustment)")
    logger.info("=" * 65)
    logger.info(f"{'Offset':>8s} {'Fills/d':>8s} {'Pairs/d':>8s} "
                f"{'Gross/d':>10s} {'AdvSel/d':>10s} {'Net/d':>10s} {'AdvSel%':>8s}")
    logger.info("-" * 75)

    naive_results = {}
    for offset in args.quote_offsets:
        sim = simulate_naive_mm(
            mid_prices=scanner.mid_prices,
            spread_arr=spread_arr,
            day_boundaries=scanner.day_boundaries,
            quote_offset_ticks=offset,
        )
        naive_results[f'offset_{offset}'] = sim
        if 'error' in sim:
            logger.info(f"  offset={offset}: ERROR — {sim['error']}")
        else:
            logger.info(
                f"  {offset:>8d} {sim['fills_per_day']:>8.1f} {sim['pairs_per_day']:>8.1f} "
                f"${sim['gross_per_day']:>8.2f} ${sim['adverse_per_day']:>8.2f} "
                f"${sim['net_per_day']:>8.2f} {sim['adverse_sel_pct_of_gross']:>7.1f}%"
            )

    # -----------------------------------------------------------------------
    # Informed MM simulation
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 65)
    logger.info("INFORMED MARKET MAKING (model-adjusted quotes)")
    logger.info("=" * 65)
    logger.info(f"{'Thresh%':>8s} {'Skew':>6s} {'Fills/d':>8s} {'Pairs/d':>8s} "
                f"{'Gross/d':>10s} {'AdvSel/d':>10s} {'Net/d':>10s} {'FilReduc':>9s}")
    logger.info("-" * 80)

    informed_results = {}
    for skew_t in args.skew_ticks:
        for thr_pct in args.pred_thresholds:
            label = f"thr{thr_pct:.0f}_skew{skew_t}"
            sim = simulate_informed_mm(
                mid_prices=scanner.mid_prices,
                spread_arr=spread_arr,
                direction_preds=direction_preds,
                day_boundaries=scanner.day_boundaries,
                pred_threshold_pct=thr_pct,
                quote_skew_ticks=skew_t,
            )
            informed_results[label] = sim
            if 'error' in sim:
                logger.info(f"  {label}: ERROR — {sim['error']}")
            else:
                logger.info(
                    f"  {thr_pct:>8.0f}% {skew_t:>6d} {sim['fills_per_day']:>8.1f} "
                    f"{sim['pairs_per_day']:>8.1f} ${sim['gross_per_day']:>8.2f} "
                    f"${sim['adverse_per_day']:>8.2f} ${sim['net_per_day']:>8.2f} "
                    f"{sim['fill_reduction_pct_vs_naive']:>8.1f}%"
                )

        now = time.time()
        if now - t_start > 30:
            logger.info(f"[Progress] {(now - t_start)/60:.1f} min elapsed")

    # -----------------------------------------------------------------------
    # Adverse selection analysis (unless fast mode)
    # -----------------------------------------------------------------------
    adv_sel_analysis = {}
    if not args.fast:
        logger.info("\nComputing adverse selection analysis by model regime...")
        t0 = time.time()
        adv_sel_analysis = analyze_adverse_selection(
            mid_prices=scanner.mid_prices,
            spread_arr=spread_arr,
            day_boundaries=scanner.day_boundaries,
            direction_preds=direction_preds,
        )
        logger.info(f"Adverse selection analysis done in {time.time() - t0:.0f}s")

        if 'error' not in adv_sel_analysis:
            logger.info("\nADVERSE SELECTION BY SIGNAL REGIME:")
            for fill_type in ['buy_fills', 'sell_fills']:
                if fill_type not in adv_sel_analysis:
                    continue
                logger.info(f"\n  {fill_type.upper()}:")
                for regime in ['weak', 'moderate', 'strong', 'extreme']:
                    regime_key = f'regime_{regime}'
                    if regime_key not in adv_sel_analysis[fill_type]:
                        continue
                    rd = adv_sel_analysis[fill_type][regime_key]
                    logger.info(
                        f"    {regime:<10s} n={rd['n']:>5d} aligned={rd['pct_aligned']:.1%}"
                    )
                    for hz, hz_data in rd.get('by_horizon', {}).items():
                        logger.info(
                            f"      {hz}: mean_adv=${hz_data['mean_adv_sel']:.3f} "
                            f"pct_adverse={hz_data['pct_adverse']:.1%}"
                        )

    # -----------------------------------------------------------------------
    # Comparison summary
    # -----------------------------------------------------------------------
    logger.info("\n" + "=" * 75)
    logger.info("MARKET MAKING COMPARISON SUMMARY")
    logger.info("=" * 75)

    # Find best naive and best informed
    best_naive = max(
        ((k, v) for k, v in naive_results.items() if 'net_per_day' in v),
        key=lambda x: x[1]['net_per_day'],
        default=(None, {}),
    )
    best_informed = max(
        ((k, v) for k, v in informed_results.items() if 'net_per_day' in v),
        key=lambda x: x[1]['net_per_day'],
        default=(None, {}),
    )

    logger.info(f"Naive MM best config: {best_naive[0]}")
    if best_naive[1]:
        bv = best_naive[1]
        logger.info(
            f"  Net/day: ${bv['net_per_day']:.2f} | Gross/day: ${bv['gross_per_day']:.2f} "
            f"| AdvSel/day: ${bv['adverse_per_day']:.2f} ({bv['adverse_sel_pct_of_gross']:.1f}% of gross)"
        )

    logger.info(f"\nInformed MM best config: {best_informed[0]}")
    if best_informed[1]:
        bv = best_informed[1]
        logger.info(
            f"  Net/day: ${bv['net_per_day']:.2f} | Gross/day: ${bv['gross_per_day']:.2f} "
            f"| AdvSel/day: ${bv['adverse_per_day']:.2f} ({bv['adverse_sel_pct_of_gross']:.1f}% of gross)"
        )

    if best_naive[1] and best_informed[1]:
        net_improvement = best_informed[1]['net_per_day'] - best_naive[1]['net_per_day']
        logger.info(f"\nInformed vs Naive improvement: ${net_improvement:.2f}/day")
        adv_reduction = best_naive[1]['adverse_per_day'] - best_informed[1]['adverse_per_day']
        logger.info(f"Adverse selection reduction: ${adv_reduction:.2f}/day")

    logger.info("\nCOMPARISON TO DIRECTIONAL TRADING:")
    logger.info(f"  Directional trading gross/trade: ~$3-4 (IC=0.08-0.11)")
    logger.info(f"  Directional trading cost: ${ROUND_TRIP_COST:.2f} -> net negative")
    logger.info(f"  Naive MM gross/pair: ~${np.nanmean(spread_arr) * ES_POINT_VALUE:.2f}")
    logger.info(f"  Naive MM: earns spread but suffers adverse selection")
    logger.info(f"  Informed MM: reduces adverse selection using direction signal")

    # -----------------------------------------------------------------------
    # Save results
    # -----------------------------------------------------------------------
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = RESULTS_DIR / f"market_making_{timestamp}.json"

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
        'direction_target': args.direction_target,
        'quote_offsets': args.quote_offsets,
        'skew_ticks': args.skew_ticks,
        'pred_thresholds': args.pred_thresholds,
        'n_days': len(scanner.day_boundaries) - 1,
        'n_snapshots': len(scanner.mid_prices),
        'spread_mean': float(np.nanmean(spread_arr)),
        'spread_median': float(spread_median),
        'tick_value': TICK_VALUE,
        'es_point_value': ES_POINT_VALUE,
        'naive_mm_results': make_serializable(naive_results),
        'informed_mm_results': make_serializable(informed_results),
        'adverse_selection_analysis': make_serializable(adv_sel_analysis),
        'elapsed_sec': time.time() - t_start,
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info(f"\nResults saved to: {result_file}")
    logger.info(f"Total elapsed: {(time.time() - t_start)/60:.1f} min")


if __name__ == '__main__':
    main()
