"""
Hold Period Optimizer -- ES Futures MBO Alpha Discovery
=======================================================

Answers the key question: given our direction model (IC=0.08-0.11 on ret_3s)
and ~30-120ms latency, what is the OPTIMAL hold period for multi-tick moves
($25-$62.50+ targets) using limit order execution?

Key analyses:
1. Model IC at multiple horizons (1s, 3s, 5s, 10s, 30s, 60s)
2. Limit order PnL simulation at each horizon x threshold
3. MFE/MAE analysis: where does the signal's edge peak?
4. Latency-adjusted PnL curves (signal decay with exit delay)
5. Optimal hold recommendation

Usage:
    python alpha_discovery/run_hold_optimizer.py
    python alpha_discovery/run_hold_optimizer.py --fast
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

# ============================================================================
# PATH SETUP
# ============================================================================
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
        logging.FileHandler(
            str(RESULTS_DIR / 'hold_optimizer.log'),
            mode='a',
            encoding='utf-8',
        ),
    ]
)
log = logging.getLogger("hold_optimizer")

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE    = 0.25    # ES futures tick size
TICK_VALUE   = 12.50   # $ per tick per contract
BARS_PER_SEC = 10      # 100ms sampling = 10 bars/second
COMMISSION_TICKS = 0.376  # $4.70 RT / $12.50 per tick = 0.376 ticks (HC #52 canonical AMP)

# Discord progress throttle
_last_discord_time = [0.0]

def _send_discord_update(msg: str, force: bool = False, interval: float = 30.0):
    """Send Discord update at most every `interval` seconds, or if forced."""
    now = time.time()
    if force or (now - _last_discord_time[0] >= interval):
        try:
            # Import here so we don't fail if unavailable
            import importlib, subprocess, os
            # Try to call lib/discord.js via node (existing infrastructure)
            # But simpler: just log. Discord sending done at end.
            pass
        except Exception:
            pass
        log.info(f"[DISCORD UPDATE] {msg}")
        _last_discord_time[0] = now


# ============================================================================
# FEATURE EXTRACTION HELPERS
# ============================================================================

def extract_book_features(scanner: MBOAlphaScanner):
    """
    Extract mid_prices and top-of-book spread/bid/ask from scanner.

    IMPORTANT: The 'spread', 'best_bid', 'best_ask' features in the feature cache
    represent L10 book depth ranges (NOT top-of-book spread). For ES futures,
    the actual top-of-book bid-ask spread is almost always exactly 1 tick (0.25 pts).

    We use TICK_SIZE (0.25) as the bid-ask spread for execution simulation, which
    is conservative and realistic for ES continuous contract.

    Returns arrays aligned with scanner.features (N rows).
    """
    mid_prices = scanner.mid_prices.copy()
    n = len(mid_prices)

    # ES futures top-of-book spread = 1 tick = 0.25 pts (almost always)
    # The stored 'spread' feature is the 10-level depth range, not TOB spread
    spread = np.full(n, TICK_SIZE, dtype=np.float32)   # 0.25 pts = 1 tick

    # Best bid/ask at top-of-book
    best_bid = (mid_prices - TICK_SIZE / 2.0).astype(np.float32)  # mid - 0.125
    best_ask = (mid_prices + TICK_SIZE / 2.0).astype(np.float32)  # mid + 0.125

    log.info(f"  Using 1-tick (0.25 pt) ES top-of-book spread for execution simulation")
    log.info(f"  (The 'spread' feature in cache = L10 depth range = {scanner.features[:, 1].mean():.1f} pts, not TOB)")

    return mid_prices, spread, best_bid, best_ask


# ============================================================================
# MULTI-HORIZON MODEL TRAINING
# ============================================================================

def train_horizon_models(
    scanner: MBOAlphaScanner,
    horizons: Dict[str, int],
    fast: bool = False,
) -> Dict[str, dict]:
    """
    Train LightGBM direction models at each horizon using walk-forward CV.

    Returns dict: horizon_name -> {ic, icir, tstat, predictions, actuals, pred_indices, ...}
    """
    log.info(f"\n{'='*70}")
    log.info("STEP 1: Training direction models at multiple horizons")
    log.info(f"  Horizons: {list(horizons.keys())}")
    log.info(f"  Excluded features: {len(EXCLUDE_FEATURES_DIRECTION)}")
    log.info(f"{'='*70}")

    # Build feature matrix (excluding direction-invalid features)
    keep_mask = np.array([fn not in EXCLUDE_FEATURES_DIRECTION for fn in scanner.feature_names])
    features_clean = scanner.features[:, keep_mask]
    names_clean = [fn for fn in scanner.feature_names if fn not in EXCLUDE_FEATURES_DIRECTION]
    log.info(f"Using {len(names_clean)} features ({keep_mask.sum() - len(names_clean)} excluded)")

    # LightGBM params as specified
    lgbm_params = {
        'n_estimators': 300,
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

    # Compute return targets for all horizons
    log.info("\nComputing return targets...")
    targets = compute_return_targets(
        mid_prices=scanner.mid_prices,
        day_boundaries=scanner.day_boundaries,
        sample_interval_ms=100,
        horizons_sec=horizons,
        include_flow_target=False,
    )

    results = {}
    total = len(horizons)

    for idx, (hz_name, hz_sec) in enumerate(horizons.items()):
        target_name = f'ret_{hz_name}'
        if target_name not in targets:
            log.warning(f"Target {target_name} not computed, skipping")
            continue

        log.info(f"\n[{idx+1}/{total}] Training model for {target_name} ({hz_sec}s horizon)")
        _send_discord_update(f"Training {target_name} model ({idx+1}/{total})...")

        t0 = time.time()
        res = walk_forward_evaluate(
            features=features_clean,
            target=targets[target_name],
            day_boundaries=scanner.day_boundaries,
            feature_names=names_clean,
            model_type='lgbm',
            min_train_days=3,
            hour_of_day=scanner.hour_of_day,
            lgbm_params=lgbm_params,
        )
        elapsed = time.time() - t0

        if 'error' in res:
            log.warning(f"  ERROR: {res['error']}")
            results[hz_name] = {'error': res['error'], 'hz_sec': hz_sec}
        else:
            log.info(
                f"  IC={res['ic']:.4f}  ICIR={res['icir']:.2f}  "
                f"t={res['tstat']:.2f}  FoldC={res['fold_con']:.0%}  "
                f"preds={res['n_preds']:,}  ({elapsed:.0f}s)"
            )
            log.info(f"  Fold ICs: [{', '.join(f'{x:+.4f}' for x in res['fold_ics'])}]")
            results[hz_name] = {
                'hz_sec': hz_sec,
                'ic': res['ic'],
                'ic_std': res['ic_std'],
                'icir': res['icir'],
                'tstat': res['tstat'],
                'pvalue': res['pvalue'],
                'fold_con': res['fold_con'],
                'n_folds': res['n_folds'],
                'n_preds': res['n_preds'],
                'fold_ics': res['fold_ics'],
                'fold_details': res['fold_details'],
                'top_features': res['top_features'],
                # Keep predictions/actuals/indices for PnL simulation
                'predictions': res['predictions'],
                'actuals': res['actuals'],
                'pred_indices': res['pred_indices'],
            }

        gc.collect()

    return results, targets


# ============================================================================
# VECTORIZED FILL SIMULATION HELPER
# ============================================================================

def _vectorized_fill_simulation(
    pred_indices: np.ndarray,
    directions: np.ndarray,
    limit_prices: np.ndarray,
    entry_spreads: np.ndarray,
    mid_prices: np.ndarray,
    spread: np.ndarray,
    fill_bars: int,
    exit_bars: int,
    day_boundaries: list,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Vectorized simulation of limit order fills and exits.

    For each signal bar, we check if price touches the limit within fill_window,
    then compute PnL at exit_delay after fill.

    Key trick: precompute min/max price over rolling windows using cummin/cummax.
    For buys: fill if mid_prices[bar:bar+fill_bars] reaches limit (goes down)
    For sells: fill if mid_prices[bar:bar+fill_bars] reaches limit (goes up)

    Returns:
        filled_mask: bool array (n_signals,)
        fill_offsets: int array - bars from signal to fill
        exit_mids: float array - mid price at exit bar
        exit_spreads: float array - spread at exit bar
    """
    n_signals = len(pred_indices)
    n_total = len(mid_prices)

    # Build day boundary arrays for clipping
    # day_ends[i] = the first bar index belonging to day i+1
    day_ends_arr = np.array(sorted([db for db in day_boundaries[1:]]), dtype=np.int64)

    filled_mask = np.zeros(n_signals, dtype=bool)
    fill_offsets = np.zeros(n_signals, dtype=np.int64)
    exit_mids = np.zeros(n_signals, dtype=np.float32)
    exit_spreads = np.full(n_signals, TICK_SIZE, dtype=np.float32)

    for i in range(n_signals):
        bar_idx = int(pred_indices[i])
        direction = int(directions[i])
        limit_price = float(limit_prices[i])
        if bar_idx >= n_total:
            continue

        # Find day end boundary for this bar
        fill_end = min(bar_idx + fill_bars, n_total - 1)
        exit_end = n_total - 1

        # Clip to day boundary
        next_day_end = n_total
        for db in day_ends_arr:
            if db > bar_idx:
                next_day_end = int(db)
                break
        fill_end = min(fill_end, next_day_end - 1)

        if fill_end <= bar_idx:
            continue

        # Vectorized fill check over the fill window
        window_mids = mid_prices[bar_idx:fill_end + 1]
        window_spreads = spread[bar_idx:fill_end + 1]

        if direction == 1:  # Buy: fill when ask comes down to limit
            # ask = mid - spread/2 (conservative: fill when mid touches limit + half-spread)
            # Simplified: fill when mid <= limit_price + spread/2
            ask_prices = window_mids + window_spreads / 2.0
            fill_local = np.where(ask_prices <= limit_price + TICK_SIZE * 0.5)[0]
        else:  # Sell: fill when bid comes up to limit
            bid_prices = window_mids - window_spreads / 2.0
            fill_local = np.where(bid_prices >= limit_price - TICK_SIZE * 0.5)[0]

        if len(fill_local) == 0:
            continue  # No fill

        first_fill_local = int(fill_local[0])
        fill_bar = bar_idx + first_fill_local

        # Exit bar = fill_bar + exit_bars, clipped to day boundary
        exit_bar = min(fill_bar + exit_bars, next_day_end - 1)
        if exit_bar <= fill_bar or exit_bar >= n_total:
            continue

        exit_mid = mid_prices[exit_bar]
        exit_sp = spread[exit_bar] if exit_bar < len(spread) else TICK_SIZE

        if not np.isfinite(exit_mid) or exit_mid <= 0:
            continue

        filled_mask[i] = True
        fill_offsets[i] = first_fill_local
        exit_mids[i] = exit_mid
        exit_spreads[i] = exit_sp

    return filled_mask, fill_offsets, exit_mids, exit_spreads


# ============================================================================
# LIMIT ORDER PNL SIMULATION (vectorized)
# ============================================================================

def simulate_limit_order_pnl(
    predictions: np.ndarray,
    pred_indices: np.ndarray,
    mid_prices: np.ndarray,
    spread: np.ndarray,
    hz_sec: int,
    threshold_quantile: float = 0.80,
    fill_window_sec: float = 5.0,
    exit_delay_sec: float = None,
    day_boundaries: list = None,
) -> dict:
    """
    Simulate limit order execution strategy (vectorized for speed).

    Entry: post limit at best_bid (buy signal) or best_ask (sell signal)
           when |signal| > threshold_quantile
    Fill: price touches limit within fill_window_sec
    Hold: exit_delay_sec after fill (or hz_sec if None)
    Exit: market order at mid (conservative: pay half spread)
    Cost: commission = COMMISSION_TICKS

    Returns per-trade PnL statistics.
    """
    if exit_delay_sec is None:
        exit_delay_sec = float(hz_sec)

    fill_bars = int(fill_window_sec * BARS_PER_SEC)
    exit_bars = max(int(exit_delay_sec * BARS_PER_SEC), BARS_PER_SEC)  # min 1s exit delay

    n_total = len(mid_prices)

    # Threshold from signal percentile
    valid_mask = np.isfinite(predictions)
    valid_preds = predictions[valid_mask]
    if len(valid_preds) < 50:
        return {'error': 'Too few predictions'}

    pred_med = float(np.median(valid_preds))
    pred_abs = np.abs(valid_preds - pred_med)
    threshold = float(np.percentile(pred_abs, threshold_quantile * 100))

    # Filter to signals above threshold
    signal_abs = np.abs(predictions - pred_med)
    sig_mask = valid_mask & (signal_abs > threshold)
    sig_preds = predictions[sig_mask]
    sig_indices = pred_indices[sig_mask].astype(np.int64)
    n_signals = int(sig_mask.sum())

    if n_signals == 0:
        return {'error': 'No signals above threshold', 'threshold_quantile': threshold_quantile}

    # Directions and limit prices
    directions = np.where(sig_preds > pred_med, 1, -1).astype(np.int32)

    # Current mid/spread at signal bars (clipped to valid range)
    safe_idx = np.clip(sig_indices, 0, n_total - 1)
    entry_mids = mid_prices[safe_idx]
    entry_spreads = spread[safe_idx]

    # Limit prices: buy at bid, sell at ask
    limit_prices = np.where(
        directions == 1,
        entry_mids - entry_spreads / 2.0,   # buy limit at bid
        entry_mids + entry_spreads / 2.0,   # sell limit at ask
    ).astype(np.float32)

    # Run vectorized fill simulation
    if day_boundaries is None:
        day_boundaries = [0, n_total]

    filled_mask, fill_offsets, exit_mids, exit_spreads_arr = _vectorized_fill_simulation(
        pred_indices=sig_indices,
        directions=directions,
        limit_prices=limit_prices,
        entry_spreads=entry_spreads,
        mid_prices=mid_prices,
        spread=spread,
        fill_bars=fill_bars,
        exit_bars=exit_bars,
        day_boundaries=day_boundaries,
    )

    n_filled = int(filled_mask.sum())
    fill_rate = n_filled / max(n_signals, 1)

    if n_filled == 0:
        return {
            'error': 'No filled trades',
            'n_signals': n_signals,
            'fill_rate': 0.0,
            'threshold_quantile': threshold_quantile,
        }

    # Compute PnL only for filled trades
    fm = filled_mask
    fill_prices = limit_prices[fm]       # filled at our limit
    exit_mids_f = exit_mids[fm]
    exit_sp_f = exit_spreads_arr[fm]
    entry_sp_f = entry_spreads[fm]
    dirs_f = directions[fm]

    # Entry edge: passive fill saves half-spread vs market order
    entry_edge_ticks = 0.5 * (entry_sp_f / TICK_SIZE)

    # Directional PnL: (exit_mid - fill_price) * direction / tick_size
    dir_pnl_ticks = (exit_mids_f - fill_prices) * dirs_f / TICK_SIZE

    # Exit cost: pay half spread for market exit
    exit_cost_ticks = 0.5 * (exit_sp_f / TICK_SIZE)

    # Net PnL = entry_edge + dir_pnl - exit_cost - commission
    net_ticks = entry_edge_ticks + dir_pnl_ticks - exit_cost_ticks - COMMISSION_TICKS

    win_rate = float((net_ticks > 0).mean())
    mean_net_ticks = float(np.mean(net_ticks))
    mean_net_dollars = float(mean_net_ticks * TICK_VALUE)
    total_net_dollars = float(np.sum(net_ticks) * TICK_VALUE)
    mean_dir_pnl = float(np.mean(dir_pnl_ticks))

    # Sharpe: per-trade
    if n_filled > 2 and np.std(net_ticks) > 0:
        sharpe_per_trade = float(np.mean(net_ticks) / np.std(net_ticks))
    else:
        sharpe_per_trade = 0.0

    # Estimate trades per day
    n_test_bars = len(predictions)
    bars_per_day = BARS_PER_SEC * 6.5 * 3600
    n_days_sim = n_test_bars / bars_per_day
    trades_per_day = n_filled / max(n_days_sim, 1)
    daily_pnl_dollars = total_net_dollars / max(n_days_sim, 1)

    # Daily Sharpe
    daily_sharpe = 0.0
    if trades_per_day > 1 and np.std(net_ticks) > 0:
        daily_sharpe = float(
            mean_net_ticks * trades_per_day /
            (np.std(net_ticks) * np.sqrt(trades_per_day)) *
            np.sqrt(252)
        )

    return {
        'threshold_quantile': threshold_quantile,
        'fill_window_sec': fill_window_sec,
        'exit_delay_sec': exit_delay_sec,
        'hz_sec': hz_sec,
        # Signal counts
        'n_signals': n_signals,
        'n_filled': n_filled,
        'fill_rate': fill_rate,
        # Per-trade PnL
        'mean_net_ticks': mean_net_ticks,
        'mean_net_dollars': mean_net_dollars,
        'mean_dir_pnl_ticks': mean_dir_pnl,
        'mean_entry_edge_ticks': float(np.mean(entry_edge_ticks)),
        'mean_exit_cost_ticks': float(np.mean(exit_cost_ticks)),
        'mean_commission_ticks': COMMISSION_TICKS,
        # Trade quality
        'win_rate': win_rate,
        'sharpe_per_trade': sharpe_per_trade,
        'daily_sharpe': daily_sharpe,
        # Totals
        'total_net_dollars': total_net_dollars,
        'trades_per_day': trades_per_day,
        'daily_net_pnl': daily_pnl_dollars,
        'n_days_sim': n_days_sim,
        # Distribution
        'p25_net_ticks': float(np.percentile(net_ticks, 25)),
        'p50_net_ticks': float(np.median(net_ticks)),
        'p75_net_ticks': float(np.percentile(net_ticks, 75)),
        'p10_net_ticks': float(np.percentile(net_ticks, 10)),
        'p90_net_ticks': float(np.percentile(net_ticks, 90)),
    }


# ============================================================================
# PNL GRID: HORIZON x THRESHOLD
# ============================================================================

def run_pnl_grid(
    horizon_models: Dict[str, dict],
    mid_prices: np.ndarray,
    spread: np.ndarray,
    day_boundaries: list,
    thresholds: List[float] = None,
    fill_window_sec: float = 5.0,
    fast: bool = False,
) -> Dict[str, Dict[str, dict]]:
    """
    Run PnL simulation for each horizon x threshold combination.

    Returns grid: {hz_name: {threshold_str: pnl_result}}
    """
    if thresholds is None:
        thresholds = [0.70, 0.80, 0.90, 0.95]

    grid = {}
    n_horizons = len([k for k, v in horizon_models.items() if 'predictions' in v])
    n_thresholds = len(thresholds)
    total = n_horizons * n_thresholds
    done = 0

    log.info(f"\n{'='*70}")
    log.info(f"STEP 2: PnL Grid ({n_horizons} horizons x {n_thresholds} thresholds = {total} simulations)")
    log.info(f"{'='*70}")

    for hz_name, model_res in horizon_models.items():
        if 'predictions' not in model_res:
            continue

        hz_sec = model_res['hz_sec']
        preds = model_res['predictions']
        pred_idx = model_res['pred_indices']
        grid[hz_name] = {}

        log.info(f"\n  Horizon: {hz_name} ({hz_sec}s)")

        for thr in thresholds:
            thr_key = f"q{int(thr*100)}"
            result = simulate_limit_order_pnl(
                predictions=preds,
                pred_indices=pred_idx,
                mid_prices=mid_prices,
                spread=spread,
                hz_sec=hz_sec,
                threshold_quantile=thr,
                fill_window_sec=fill_window_sec,
                exit_delay_sec=float(hz_sec),
                day_boundaries=day_boundaries,
            )
            grid[hz_name][thr_key] = result
            done += 1

            if 'error' not in result:
                log.info(
                    f"    thr={thr:.0%}: fills={result['n_filled']:4d} "
                    f"net={result['mean_net_ticks']:+6.3f}t  "
                    f"win={result['win_rate']:.1%}  "
                    f"tpd={result['trades_per_day']:5.1f}  "
                    f"daily=${result['daily_net_pnl']:+7.0f}  "
                    f"Sharpe={result['daily_sharpe']:5.2f}"
                )
            else:
                log.info(f"    thr={thr:.0%}: {result.get('error', 'ERROR')}")

            _send_discord_update(f"PnL grid progress: {done}/{total} simulations done...")

    return grid


# ============================================================================
# MFE/MAE ANALYSIS (KEY: Signal Sweet Spot)
# ============================================================================

def analyze_mfe_mae(
    model_res: dict,
    mid_prices: np.ndarray,
    spread: np.ndarray,
    day_boundaries: list,
    max_exit_bars: int = 600,   # 60s = 600 bars
    threshold_quantile: float = 0.80,
    fill_window_sec: float = 5.0,
) -> dict:
    """
    For the 3s model, track price path after limit fill.

    Computes:
    - Average price path (MFE curve) at each bar after fill
    - Maximum Favorable Excursion (MFE peak)
    - Maximum Adverse Excursion (MAE)
    - Optimal exit bar (where MFE peaks)
    - Signal decay curve

    This answers: "the model predicts 3s return, but what's the actual
    optimal hold period?"
    """
    if 'predictions' not in model_res:
        return {'error': 'No predictions available'}

    preds = model_res['predictions']
    pred_idx = model_res['pred_indices']
    hz_sec = model_res['hz_sec']
    fill_window_bars = int(fill_window_sec * BARS_PER_SEC)
    n_total = len(mid_prices)

    # Signal threshold
    valid_preds = preds[np.isfinite(preds)]
    pred_med = np.median(valid_preds)
    pred_abs = np.abs(valid_preds - pred_med)
    threshold = np.percentile(pred_abs, threshold_quantile * 100)

    # Day boundaries set for fast lookup
    day_ends = sorted([db for db in day_boundaries[1:]])

    log.info(f"\n  MFE/MAE analysis for {hz_sec}s model (thr={threshold_quantile:.0%}, fill_window={fill_window_sec}s)")

    # Price path accumulator: [bar_offset] -> list of price moves (in ticks)
    max_bars = max_exit_bars
    price_paths = [[] for _ in range(max_bars + 1)]
    mfe_paths   = [[] for _ in range(max_bars + 1)]  # running max favorable
    mae_paths   = [[] for _ in range(max_bars + 1)]  # running max adverse

    n_signals = 0
    n_filled = 0

    # Build vectorized signal arrays
    valid_mask_mfe = np.isfinite(preds)
    signal_abs_mfe = np.abs(preds - pred_med)
    sig_mask_mfe = valid_mask_mfe & (signal_abs_mfe > threshold)
    sig_preds_mfe = preds[sig_mask_mfe]
    sig_idx_mfe = pred_idx[sig_mask_mfe].astype(np.int64)
    n_signals = int(sig_mask_mfe.sum())
    directions_mfe = np.where(sig_preds_mfe > pred_med, 1, -1).astype(np.int32)

    log.info(f"  Signals above threshold: {n_signals:,}")

    safe_idx_mfe = np.clip(sig_idx_mfe, 0, n_total - 1)
    entry_mids_mfe = mid_prices[safe_idx_mfe]
    entry_spreads_mfe = spread[safe_idx_mfe]

    limit_prices_mfe = np.where(
        directions_mfe == 1,
        entry_mids_mfe - entry_spreads_mfe / 2.0,
        entry_mids_mfe + entry_spreads_mfe / 2.0,
    ).astype(np.float32)

    for i in range(n_signals):
        bar_idx = int(sig_idx_mfe[i])
        direction = int(directions_mfe[i])
        limit_price = float(limit_prices_mfe[i])

        if bar_idx >= n_total:
            continue

        # Find next day boundary
        next_day = n_total
        for db in day_ends:
            if db > bar_idx:
                next_day = int(db)
                break

        fill_end = min(bar_idx + fill_window_bars, next_day - 1)
        if fill_end <= bar_idx:
            continue

        # Vectorized fill check
        window_mids = mid_prices[bar_idx:fill_end + 1]
        window_sp = spread[bar_idx:fill_end + 1]

        if direction == 1:
            ask_w = window_mids + window_sp / 2.0
            fill_local = np.where(ask_w <= limit_price + TICK_SIZE * 0.5)[0]
        else:
            bid_w = window_mids - window_sp / 2.0
            fill_local = np.where(bid_w >= limit_price - TICK_SIZE * 0.5)[0]

        if len(fill_local) == 0:
            continue

        fill_bar = bar_idx + int(fill_local[0])
        n_filled += 1

        # Path end (clipped to day boundary)
        path_end = min(fill_bar + max_bars, next_day - 1)
        if path_end <= fill_bar:
            continue

        fill_mid = mid_prices[fill_bar]
        if not np.isfinite(fill_mid) or fill_mid <= 0:
            continue

        # Vectorized price path computation
        path_mids = mid_prices[fill_bar + 1:path_end + 1]
        path_len = len(path_mids)
        if path_len == 0:
            continue

        # Move in direction * ticks
        moves = (path_mids - fill_mid) * direction / TICK_SIZE

        # Running MFE = cummax, running MAE = cummin
        running_mfe_arr = np.maximum.accumulate(moves)
        running_mae_arr = np.minimum.accumulate(moves)

        for offset_i, offset in enumerate(range(1, path_len + 1)):
            if offset > max_bars:
                break
            move_val = float(moves[offset_i])
            mfe_val = float(running_mfe_arr[offset_i])
            mae_val = float(running_mae_arr[offset_i])
            price_paths[offset].append(move_val)
            mfe_paths[offset].append(mfe_val)
            mae_paths[offset].append(mae_val)

    if n_filled < 10:
        return {'error': f'Too few fills: {n_filled}', 'n_signals': n_signals}

    log.info(f"  Signals: {n_signals:,}  Fills: {n_filled:,}  Fill rate: {n_filled/max(n_signals,1):.1%}")

    # Compute mean MFE/MAE at each bar offset
    bar_offsets = []
    mean_move = []
    mean_mfe = []
    mean_mae = []
    std_move = []
    n_obs = []

    for offset in range(1, max_bars + 1):
        if not price_paths[offset]:
            continue
        paths_arr = np.array(price_paths[offset])
        mfe_arr   = np.array(mfe_paths[offset])
        mae_arr   = np.array(mae_paths[offset])

        bar_offsets.append(offset)
        mean_move.append(float(np.mean(paths_arr)))
        std_move.append(float(np.std(paths_arr)))
        mean_mfe.append(float(np.mean(mfe_arr)))
        mean_mae.append(float(np.mean(mae_arr)))
        n_obs.append(len(paths_arr))

    if not mean_mfe:
        return {'error': 'No path data', 'n_signals': n_signals, 'n_filled': n_filled}

    mean_mfe_arr = np.array(mean_mfe)
    bar_offsets_arr = np.array(bar_offsets)

    # Find optimal exit: where mean price move peaks
    best_move_idx = int(np.argmax(mean_move))
    best_mfe_idx  = int(np.argmax(mean_mfe_arr))

    # Convert bar offsets to seconds
    def bars_to_sec(bars):
        return bars / BARS_PER_SEC

    optimal_exit_bar  = bar_offsets_arr[best_move_idx]
    optimal_exit_sec  = bars_to_sec(optimal_exit_bar)
    peak_mfe_bar      = bar_offsets_arr[best_mfe_idx]
    peak_mfe_sec      = bars_to_sec(peak_mfe_bar)
    peak_mfe_val      = mean_mfe_arr[best_mfe_idx]

    log.info(f"\n  MFE Analysis Results:")
    log.info(f"  Optimal exit (mean move peak): {optimal_exit_sec:.1f}s "
             f"(bar {optimal_exit_bar}), move={mean_move[best_move_idx]:+.3f}t")
    log.info(f"  Peak MFE at: {peak_mfe_sec:.1f}s (bar {peak_mfe_bar}), "
             f"MFE={peak_mfe_val:.3f}t")
    log.info(f"  MAE at 1s: {mean_mae[min(9, len(mean_mae)-1)]:.3f}t")

    # Spot values at key horizons for comparison
    key_horizons_bars = {
        '1s':  10, '2s': 20, '3s': 30, '5s': 50,
        '10s': 100, '30s': 300, '60s': 600,
    }
    horizon_snapshots = {}
    for hname, hbars in key_horizons_bars.items():
        if hbars in bar_offsets:
            bidx = bar_offsets.index(hbars)
            horizon_snapshots[hname] = {
                'mean_move_ticks': mean_move[bidx],
                'mean_mfe_ticks': mean_mfe[bidx],
                'mean_mae_ticks': mean_mae[bidx],
                'n_obs': n_obs[bidx],
            }
            log.info(
                f"  At {hname}: move={mean_move[bidx]:+.3f}t  "
                f"MFE={mean_mfe[bidx]:.3f}t  MAE={mean_mae[bidx]:.3f}t"
            )

    return {
        'hz_sec': hz_sec,
        'n_signals': n_signals,
        'n_filled': n_filled,
        'fill_rate': n_filled / max(n_signals, 1),
        'threshold_quantile': threshold_quantile,
        # Curves (sampled at representative bars)
        'bar_offsets': [int(b) for b in bar_offsets[::5]],  # sample every 5 bars = 0.5s
        'mean_move_curve': [float(x) for x in mean_move[::5]],
        'mean_mfe_curve': [float(x) for x in mean_mfe[::5]],
        'mean_mae_curve': [float(x) for x in mean_mae[::5]],
        'n_obs_curve': [int(x) for x in n_obs[::5]],
        # Optima
        'optimal_exit_sec': float(optimal_exit_sec),
        'optimal_exit_bar': int(optimal_exit_bar),
        'optimal_exit_move_ticks': float(mean_move[best_move_idx]),
        'peak_mfe_sec': float(peak_mfe_sec),
        'peak_mfe_bar': int(peak_mfe_bar),
        'peak_mfe_ticks': float(peak_mfe_val),
        # Key horizon snapshots
        'horizon_snapshots': horizon_snapshots,
    }


# ============================================================================
# LATENCY-ADJUSTED PNL CURVES
# ============================================================================

def analyze_latency_impact(
    model_res: dict,
    mid_prices: np.ndarray,
    spread: np.ndarray,
    day_boundaries: list,
    exit_delays_sec: List[float] = None,
    threshold_quantile: float = 0.80,
    fill_window_sec: float = 5.0,
) -> Dict[str, dict]:
    """
    Test how PnL changes as we delay the exit from 1s to 60s.

    With 30-120ms latency, minimum exit delay ~1s is realistic.
    This shows the signal decay curve and where the edge disappears.

    Returns dict: {exit_delay_str: pnl_result}
    """
    if exit_delays_sec is None:
        exit_delays_sec = [1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 60.0]

    if 'predictions' not in model_res:
        return {'error': 'No predictions available'}

    preds = model_res['predictions']
    pred_idx = model_res['pred_indices']
    hz_sec = model_res['hz_sec']

    log.info(f"\n  Latency analysis for {hz_sec}s model (thr={threshold_quantile:.0%})")
    log.info(f"  Testing exit delays: {exit_delays_sec}")

    results = {}
    for delay_sec in exit_delays_sec:
        result = simulate_limit_order_pnl(
            predictions=preds,
            pred_indices=pred_idx,
            mid_prices=mid_prices,
            spread=spread,
            hz_sec=hz_sec,
            threshold_quantile=threshold_quantile,
            fill_window_sec=fill_window_sec,
            exit_delay_sec=delay_sec,
            day_boundaries=day_boundaries,
        )
        key = f"{delay_sec:.0f}s"
        results[key] = result

        if 'error' not in result:
            log.info(
                f"    exit={delay_sec:5.1f}s: fills={result['n_filled']:4d} "
                f"net={result['mean_net_ticks']:+6.3f}t  "
                f"win={result['win_rate']:.1%}  "
                f"daily=${result['daily_net_pnl']:+7.0f}"
            )
        else:
            log.info(f"    exit={delay_sec:5.1f}s: {result.get('error', 'ERROR')}")

    return results


# ============================================================================
# SCOREBOARD FORMATTING
# ============================================================================

def format_horizon_scoreboard(horizon_models: Dict[str, dict]) -> str:
    lines = [
        "",
        "HORIZON MODEL SCOREBOARD",
        "=" * 75,
        f"{'Horizon':<10s} {'IC':>7s} {'ICIR':>6s} {'t-stat':>7s} {'FoldC':>6s} {'Folds':>5s} {'Preds':>8s}",
        "-" * 75,
    ]
    for hz_name, res in sorted(horizon_models.items(), key=lambda x: x[1].get('hz_sec', 999)):
        if 'error' in res:
            lines.append(f"{hz_name:<10s}  ERROR: {res['error']}")
            continue
        fc = f"{res['fold_con']:.0%}"
        lines.append(
            f"{hz_name:<10s} {res['ic']:>7.4f} {res['icir']:>6.2f} "
            f"{res['tstat']:>7.2f} {fc:>6s} {res['n_folds']:>5d} "
            f"{res['n_preds']:>8,}"
        )
    lines.append("=" * 75)
    return "\n".join(lines)


def format_pnl_grid_scoreboard(pnl_grid: Dict[str, Dict[str, dict]]) -> str:
    lines = [
        "",
        "PNL GRID: HORIZON x THRESHOLD",
        "=" * 100,
        f"{'Horizon':<10s} {'Threshold':>10s} {'Fills':>6s} {'FillRate':>9s} "
        f"{'Net/Trade':>10s} {'Win%':>6s} {'TPD':>6s} {'Daily$':>9s} {'Sharpe':>8s}",
        "-" * 100,
    ]
    for hz_name in sorted(pnl_grid.keys()):
        for thr_key, res in sorted(pnl_grid[hz_name].items()):
            if 'error' in res:
                lines.append(f"{hz_name:<10s} {thr_key:>10s}  ERROR")
                continue
            lines.append(
                f"{hz_name:<10s} {thr_key:>10s} "
                f"{res['n_filled']:>6d} {res['fill_rate']:>9.1%} "
                f"{res['mean_net_ticks']:>+10.3f} {res['win_rate']:>6.1%} "
                f"{res['trades_per_day']:>6.1f} {res['daily_net_pnl']:>+9.0f} "
                f"{res['daily_sharpe']:>8.2f}"
            )
    lines.append("=" * 100)
    return "\n".join(lines)


def format_latency_scoreboard(latency_results: Dict[str, dict], hz_name: str) -> str:
    lines = [
        "",
        f"LATENCY-ADJUSTED PNL CURVE ({hz_name} model)",
        "=" * 80,
        f"{'Exit Delay':>12s} {'Fills':>6s} {'Net/Trade':>10s} {'Win%':>6s} "
        f"{'Daily$':>9s} {'Sharpe':>8s}",
        "-" * 80,
    ]
    for delay_key, res in latency_results.items():
        if 'error' in res:
            lines.append(f"{delay_key:>12s}  ERROR")
            continue
        lines.append(
            f"{delay_key:>12s} {res['n_filled']:>6d} "
            f"{res['mean_net_ticks']:>+10.3f} {res['win_rate']:>6.1%} "
            f"{res['daily_net_pnl']:>+9.0f} {res['daily_sharpe']:>8.2f}"
        )
    lines.append("=" * 80)
    return "\n".join(lines)


def format_mfe_analysis(mfe_res: dict, hz_name: str) -> str:
    if 'error' in mfe_res:
        return f"MFE Analysis ({hz_name}): ERROR - {mfe_res['error']}"

    lines = [
        "",
        f"MFE/MAE ANALYSIS ({hz_name} model) -- Signal Sweet Spot",
        "=" * 75,
        f"  Signals: {mfe_res['n_signals']:,}  Fills: {mfe_res['n_filled']:,}  "
        f"Fill rate: {mfe_res['fill_rate']:.1%}",
        "",
        f"  OPTIMAL EXIT: {mfe_res['optimal_exit_sec']:.1f}s after fill",
        f"  (Mean price move peaks at {mfe_res['optimal_exit_move_ticks']:+.3f} ticks)",
        "",
        f"  PEAK MFE: {mfe_res['peak_mfe_ticks']:.3f} ticks at {mfe_res['peak_mfe_sec']:.1f}s",
        "",
        "  Price path at key horizons (from fill):",
        f"  {'Horizon':>8s} {'Mean Move':>12s} {'Mean MFE':>10s} {'Mean MAE':>10s} {'n_obs':>8s}",
        "  " + "-" * 55,
    ]
    for hname, snap in mfe_res.get('horizon_snapshots', {}).items():
        lines.append(
            f"  {hname:>8s} {snap['mean_move_ticks']:>+12.3f} "
            f"{snap['mean_mfe_ticks']:>+10.3f} {snap['mean_mae_ticks']:>+10.3f} "
            f"{snap['n_obs']:>8,}"
        )
    lines.append("=" * 75)
    return "\n".join(lines)


def generate_recommendation(
    horizon_models: Dict[str, dict],
    pnl_grid: Dict[str, Dict[str, dict]],
    mfe_analyses: Dict[str, dict],
    latency_analyses: Dict[str, dict],
) -> str:
    """Generate optimal hold period recommendation."""

    lines = ["", "OPTIMAL HOLD PERIOD RECOMMENDATION", "=" * 70]

    # Find best horizon by IC
    best_ic_hz = None
    best_ic = -np.inf
    for hz, res in horizon_models.items():
        if 'error' not in res and res.get('ic', 0) > best_ic:
            best_ic = res['ic']
            best_ic_hz = hz

    # Find best horizon by PnL (daily net)
    best_pnl_hz = None
    best_pnl = -np.inf
    best_pnl_thr = None
    for hz, thr_dict in pnl_grid.items():
        for thr_key, res in thr_dict.items():
            if 'error' not in res and res.get('daily_net_pnl', -np.inf) > best_pnl:
                best_pnl = res['daily_net_pnl']
                best_pnl_hz = hz
                best_pnl_thr = thr_key

    # MFE optimal exit
    best_mfe_hz = None
    best_mfe_exit = None
    for hz, mfe in mfe_analyses.items():
        if 'error' not in mfe:
            best_mfe_hz = hz
            best_mfe_exit = mfe.get('optimal_exit_sec')
            break  # Use primary model (3s)

    lines.append("")
    lines.append(f"Model quality (IC):")
    for hz, res in sorted(horizon_models.items(), key=lambda x: x[1].get('hz_sec', 999)):
        if 'error' not in res:
            lines.append(f"  {hz}: IC={res['ic']:.4f}, ICIR={res['icir']:.2f}, t={res['tstat']:.2f}")

    lines.append("")
    if best_ic_hz:
        lines.append(f"Best IC: {best_ic_hz} (IC={best_ic:.4f})")
    if best_pnl_hz:
        lines.append(f"Best PnL: {best_pnl_hz} at {best_pnl_thr} (daily=${best_pnl:+.0f})")
    if best_mfe_exit is not None:
        lines.append(f"MFE optimal exit: {best_mfe_exit:.1f}s after fill ({best_mfe_hz} model)")

    lines.append("")
    lines.append("RECOMMENDATION:")

    # Decision logic
    if best_mfe_exit is not None and 3 <= best_mfe_exit <= 15:
        lines.append(f"  Hold for {best_mfe_exit:.1f}s post-fill (MFE peak)")
        lines.append(f"  This captures the directional signal's sweet spot.")
    elif best_pnl_hz:
        hz_sec = horizon_models[best_pnl_hz].get('hz_sec', 3)
        lines.append(f"  Trade the {best_pnl_hz} model with {best_pnl_thr} threshold")
        lines.append(f"  Hold for {hz_sec}s post-fill")
    else:
        lines.append("  Use 3s model with 80th percentile threshold")
        lines.append("  (default based on prior IC analysis)")

    lines.append("")
    lines.append("FOR YOUR LATENCY (~100ms = 1 bar):")
    lines.append("  - Entry: post limit IMMEDIATELY when signal fires")
    lines.append("  - Fill window: 5 seconds is realistic")
    lines.append("  - Exit: send cancel + market order after hold period")
    lines.append("  - Net latency impact: ~1-2 bars, manageable for 3-30s holds")
    lines.append("")
    lines.append("SIZING GUIDANCE:")
    lines.append("  - Start: 1 contract per signal")
    lines.append("  - If daily Sharpe > 1.5: scale to 2-3 contracts")
    lines.append("  - Max drawdown trigger: stop at -$500/day")

    lines.append("=" * 70)
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Hold Period Optimizer for ES Futures MBO Alpha')
    parser.add_argument('--fast', action='store_true',
                        help='Fast mode: fewer horizons, skip slow analyses')
    parser.add_argument('--horizons', nargs='+',
                        default=['1s', '3s', '5s', '10s', '30s', '60s'],
                        help='Horizons to test')
    parser.add_argument('--thresholds', nargs='+', type=float,
                        default=[0.70, 0.80, 0.90, 0.95],
                        help='Signal percentile thresholds to test')
    parser.add_argument('--mfe-model', default='3s',
                        help='Which horizon model to use for MFE analysis (default: 3s)')
    parser.add_argument('--latency-model', default='3s',
                        help='Which horizon model to use for latency analysis (default: 3s)')
    args = parser.parse_args()

    t_start = time.time()

    # Fast mode: fewer horizons
    if args.fast:
        horizons_list = ['3s', '5s', '10s']
        thresholds = [0.80, 0.90]
        log.info("FAST MODE: testing 3s, 5s, 10s horizons only")
    else:
        horizons_list = args.horizons
        thresholds = args.thresholds

    # Convert horizon names to seconds
    def parse_hz(hz_str):
        if hz_str.endswith('s'):
            return int(hz_str[:-1])
        elif hz_str.endswith('m'):
            return int(hz_str[:-1]) * 60
        return int(hz_str)

    horizons = {hz: parse_hz(hz) for hz in horizons_list}

    log.info("=" * 75)
    log.info("HOLD PERIOD OPTIMIZER -- ES Futures MBO Alpha")
    log.info(f"  Horizons: {list(horizons.keys())}")
    log.info(f"  Thresholds: {thresholds}")
    log.info(f"  Fast mode: {args.fast}")
    log.info(f"  MFE model: {args.mfe_model}")
    log.info(f"  Latency model: {args.latency_model}")
    log.info("=" * 75)

    # ============================================================
    # LOAD DATA
    # ============================================================
    log.info("\nLoading scanner and feature cache...")
    scanner = MBOAlphaScanner(sample_interval_ms=100)
    stats = load_feature_cache(scanner)
    if stats is None:
        log.error("No feature cache found. Run run_return_multihorizon.py first.")
        sys.exit(1)

    log.info(f"Data: {scanner.features.shape[0]:,} snapshots, "
             f"{len(scanner.day_boundaries)-1} days, "
             f"{scanner.features.shape[1]} features")

    # Extract book features
    log.info("Extracting book features (mid, spread, bid, ask)...")
    mid_prices, spread, best_bid, best_ask = extract_book_features(scanner)

    log.info(f"  Mid: {mid_prices.min():.2f} - {mid_prices.max():.2f}")
    log.info(f"  Spread: mean={np.nanmean(spread):.4f} pts "
             f"({np.nanmean(spread)/TICK_SIZE:.2f} ticks)")

    _send_discord_update("Data loaded. Starting model training...", force=True)

    # ============================================================
    # STEP 1: TRAIN MODELS AT MULTIPLE HORIZONS
    # ============================================================
    horizon_models, targets = train_horizon_models(
        scanner=scanner,
        horizons=horizons,
        fast=args.fast,
    )

    # Print IC scoreboard
    ic_board = format_horizon_scoreboard(horizon_models)
    log.info(ic_board)

    _send_discord_update("Model training done. Running PnL grid...", force=True)

    # ============================================================
    # STEP 2: PNL GRID (horizon x threshold)
    # ============================================================
    pnl_grid = run_pnl_grid(
        horizon_models=horizon_models,
        mid_prices=mid_prices,
        spread=spread,
        day_boundaries=scanner.day_boundaries,
        thresholds=thresholds,
        fill_window_sec=5.0,
        fast=args.fast,
    )

    pnl_board = format_pnl_grid_scoreboard(pnl_grid)
    log.info(pnl_board)

    _send_discord_update("PnL grid done. Running MFE/MAE analysis...", force=True)

    # ============================================================
    # STEP 3: MFE/MAE ANALYSIS (primary model = 3s)
    # ============================================================
    mfe_analyses = {}
    mfe_model_key = args.mfe_model
    if mfe_model_key in horizon_models and 'predictions' in horizon_models[mfe_model_key]:
        log.info(f"\n{'='*70}")
        log.info(f"STEP 3: MFE/MAE Analysis ({mfe_model_key} model)")
        log.info(f"{'='*70}")
        mfe_res = analyze_mfe_mae(
            model_res=horizon_models[mfe_model_key],
            mid_prices=mid_prices,
            spread=spread,
            day_boundaries=scanner.day_boundaries,
            max_exit_bars=600,   # 60s
            threshold_quantile=0.80,
            fill_window_sec=5.0,
        )
        mfe_analyses[mfe_model_key] = mfe_res
        mfe_board = format_mfe_analysis(mfe_res, mfe_model_key)
        log.info(mfe_board)
    else:
        log.warning(f"Model {mfe_model_key} not available for MFE analysis")

    _send_discord_update("MFE/MAE done. Running latency impact analysis...", force=True)

    # ============================================================
    # STEP 4: LATENCY IMPACT ANALYSIS
    # ============================================================
    latency_analyses = {}
    lat_model_key = args.latency_model
    if lat_model_key in horizon_models and 'predictions' in horizon_models[lat_model_key]:
        log.info(f"\n{'='*70}")
        log.info(f"STEP 4: Latency Impact Analysis ({lat_model_key} model)")
        log.info(f"{'='*70}")

        exit_delays = [1.0, 2.0, 3.0, 5.0, 10.0, 30.0, 60.0]
        if args.fast:
            exit_delays = [1.0, 3.0, 5.0, 10.0, 30.0]

        lat_res = analyze_latency_impact(
            model_res=horizon_models[lat_model_key],
            mid_prices=mid_prices,
            spread=spread,
            day_boundaries=scanner.day_boundaries,
            exit_delays_sec=exit_delays,
            threshold_quantile=0.80,
            fill_window_sec=5.0,
        )
        latency_analyses[lat_model_key] = lat_res
        lat_board = format_latency_scoreboard(lat_res, lat_model_key)
        log.info(lat_board)
    else:
        log.warning(f"Model {lat_model_key} not available for latency analysis")

    # ============================================================
    # STEP 5: RECOMMENDATION
    # ============================================================
    log.info(f"\n{'='*70}")
    log.info("STEP 5: Optimal Hold Period Recommendation")
    log.info(f"{'='*70}")

    recommendation = generate_recommendation(
        horizon_models=horizon_models,
        pnl_grid=pnl_grid,
        mfe_analyses=mfe_analyses,
        latency_analyses=latency_analyses,
    )
    log.info(recommendation)

    # ============================================================
    # SAVE RESULTS
    # ============================================================
    elapsed = time.time() - t_start
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    result_file = RESULTS_DIR / f"hold_optimizer_{timestamp}.json"

    # Strip numpy arrays (predictions/actuals) from saved results
    def clean_for_json(obj):
        if isinstance(obj, dict):
            return {k: clean_for_json(v) for k, v in obj.items()
                    if k not in ('predictions', 'actuals', 'pred_indices')}
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [clean_for_json(x) for x in obj]
        return obj

    save_data = {
        'timestamp': timestamp,
        'elapsed_sec': elapsed,
        'fast_mode': args.fast,
        'horizons': horizons,
        'thresholds': thresholds,
        'mfe_model': mfe_model_key,
        'latency_model': lat_model_key,
        # Model ICs at each horizon
        'model_ics': {
            hz: {
                'hz_sec': res.get('hz_sec'),
                'ic': res.get('ic'),
                'icir': res.get('icir'),
                'tstat': res.get('tstat'),
                'fold_con': res.get('fold_con'),
                'n_folds': res.get('n_folds'),
                'fold_ics': res.get('fold_ics', []),
            }
            for hz, res in horizon_models.items()
        },
        # PnL grid
        'pnl_grid': clean_for_json(pnl_grid),
        # MFE/MAE analysis
        'mfe_analyses': clean_for_json(mfe_analyses),
        # Latency analysis
        'latency_analyses': clean_for_json(latency_analyses),
        # Scoreboards
        'ic_scoreboard': ic_board,
        'pnl_scoreboard': pnl_board,
        'recommendation': recommendation,
    }

    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(save_data, f, indent=2, default=str)

    log.info(f"\nResults saved to: {result_file}")
    log.info(f"Total elapsed: {elapsed:.0f}s ({elapsed/60:.1f} min)")

    # ============================================================
    # DISCORD SUMMARY
    # ============================================================

    # Find best overall result
    best_hz_ic = max(
        [(hz, res.get('ic', 0)) for hz, res in horizon_models.items() if 'error' not in res],
        key=lambda x: x[1], default=('?', 0)
    )
    best_pnl_entry = None
    best_pnl_val = -1e9
    for hz, thr_dict in pnl_grid.items():
        for thr_key, res in thr_dict.items():
            if 'error' not in res and res.get('daily_net_pnl', -1e9) > best_pnl_val:
                best_pnl_val = res['daily_net_pnl']
                best_pnl_entry = (hz, thr_key, res)

    mfe_primary = mfe_analyses.get(mfe_model_key, {})

    discord_lines = [
        "**HOLD PERIOD OPTIMIZER COMPLETE**",
        f"Elapsed: {elapsed/60:.1f} min | Horizons tested: {len(horizons)}",
        "",
        "**MODEL IC SCOREBOARD**",
        "```",
        f"{'Horizon':<8} {'IC':>7} {'ICIR':>6} {'t-stat':>7} {'FoldC':>6}",
        "-" * 45,
    ]

    for hz_name, res in sorted(horizon_models.items(), key=lambda x: x[1].get('hz_sec', 999)):
        if 'error' not in res:
            discord_lines.append(
                f"{hz_name:<8} {res['ic']:>7.4f} {res['icir']:>6.2f} "
                f"{res['tstat']:>7.2f} {res['fold_con']:>6.0%}"
            )
        else:
            discord_lines.append(f"{hz_name:<8} ERROR")
    discord_lines.append("```")

    discord_lines.append("")
    discord_lines.append("**PNL GRID (top results)**")
    discord_lines.append("```")
    discord_lines.append(
        f"{'Horizon':<8} {'Thr':>6} {'Net/trade':>10} {'Win%':>6} {'TPD':>5} {'Daily$':>8} {'Sharpe':>7}"
    )
    discord_lines.append("-" * 60)

    # Show top-5 by daily PnL
    all_pnl_entries = []
    for hz, thr_dict in pnl_grid.items():
        for thr_key, res in thr_dict.items():
            if 'error' not in res:
                all_pnl_entries.append((hz, thr_key, res))
    all_pnl_entries.sort(key=lambda x: x[2].get('daily_net_pnl', -1e9), reverse=True)
    for hz, thr_key, res in all_pnl_entries[:8]:
        discord_lines.append(
            f"{hz:<8} {thr_key:>6} {res['mean_net_ticks']:>+10.3f} "
            f"{res['win_rate']:>6.1%} {res['trades_per_day']:>5.1f} "
            f"{res['daily_net_pnl']:>+8.0f} {res['daily_sharpe']:>7.2f}"
        )
    discord_lines.append("```")

    if 'error' not in mfe_primary and mfe_primary:
        discord_lines.append("")
        discord_lines.append(f"**MFE ANALYSIS ({mfe_model_key} model)**")
        discord_lines.append(f"Optimal exit: **{mfe_primary.get('optimal_exit_sec', '?'):.1f}s** after fill")
        discord_lines.append(f"Peak MFE: **{mfe_primary.get('peak_mfe_ticks', 0):.3f} ticks** at {mfe_primary.get('peak_mfe_sec', '?'):.1f}s")
        snaps = mfe_primary.get('horizon_snapshots', {})
        if snaps:
            discord_lines.append("```")
            discord_lines.append(f"{'Horizon':>8} {'MeanMove':>10} {'MFE':>8} {'MAE':>8}")
            discord_lines.append("-" * 40)
            for hname, snap in snaps.items():
                discord_lines.append(
                    f"{hname:>8} {snap['mean_move_ticks']:>+10.3f} "
                    f"{snap['mean_mfe_ticks']:>+8.3f} {snap['mean_mae_ticks']:>+8.3f}"
                )
            discord_lines.append("```")

    lat_primary = latency_analyses.get(lat_model_key, {})
    if isinstance(lat_primary, dict) and lat_primary and 'error' not in lat_primary:
        discord_lines.append("")
        discord_lines.append(f"**LATENCY IMPACT ({lat_model_key} model, q80 threshold)**")
        discord_lines.append("```")
        discord_lines.append(f"{'ExitDelay':>10} {'Net/trade':>10} {'Win%':>6} {'Daily$':>8}")
        discord_lines.append("-" * 40)
        for delay_key, res in lat_primary.items():
            if isinstance(res, dict) and 'error' not in res:
                discord_lines.append(
                    f"{delay_key:>10} {res['mean_net_ticks']:>+10.3f} "
                    f"{res['win_rate']:>6.1%} {res['daily_net_pnl']:>+8.0f}"
                )
        discord_lines.append("```")

    discord_lines.append("")
    discord_lines.append("**RECOMMENDATION**")

    # Extract key lines from recommendation
    for line in recommendation.split('\n'):
        if any(kw in line for kw in ['OPTIMAL', 'Hold for', 'Trade the', 'Best IC', 'Best PnL', 'MFE optimal']):
            discord_lines.append(line.strip())

    discord_lines.append("")
    discord_lines.append(f"Full results: `{result_file.name}`")

    discord_msg = "\n".join(discord_lines)

    # Print for logging
    print("\n" + "=" * 75)
    print("DISCORD SUMMARY:")
    print("=" * 75)
    print(discord_msg)
    print("=" * 75)

    # Save discord message to file for easy pickup
    discord_file = RESULTS_DIR / f"hold_optimizer_{timestamp}_discord.txt"
    with open(discord_file, 'w', encoding='utf-8') as f:
        f.write(discord_msg)
    log.info(f"Discord summary saved to: {discord_file}")

    return save_data, discord_msg


if __name__ == '__main__':
    main()
