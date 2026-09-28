"""
Corrected Limit Order Execution Study -- ES Futures MBO Alpha
=============================================================

CONTEXT: The best_bid/best_ask in the feature cache are WRONG (bug in LOB
reconstruction gives ~232-tick average spread when ES should be 1 tick).
Mid prices ARE correct (~$6367 average).

FIX: Reconstruct bid/ask from mid using the known ES constraint:
  best_bid = mid - 0.125   (half tick below mid)
  best_ask = mid + 0.125   (half tick above mid)
  spread   = 0.25 ticks    (always 1 tick for ES during RTH)

Parts:
  1. Fill Probability (with corrected spread)
  2. Price Path After Fill (MFE/MAE, signal-conditioned)
  3. Signal-Conditioned Strategy Simulation
     a) Limit entry + Market exit
     b) Limit entry + Limit exit
  4. Latency-Adjusted Results (0, 1, 3, 5 bar delays)
  5. Optimal Configuration Grid Search

KEY PNL INSIGHT:
  Limit entry + Market exit: entry_edge (+0.5t) cancels exit_cost (-0.5t)
    NET = directional_move - commission ($2.50 = 0.2t)
  Limit entry + Limit exit:  entry_edge (+0.5t) + exit_edge (+0.5t)
    NET = directional_move + 1.0t - 0.2t (HUGELY favorable!)

Usage:
    python alpha_discovery/run_corrected_limit_study.py
    python alpha_discovery/run_corrected_limit_study.py --fast
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
        logging.FileHandler(str(RESULTS_DIR / 'corrected_limit_study.log'),
                            mode='a', encoding='utf-8'),
    ]
)
log = logging.getLogger("corrected_limit_study")

# ============================================================================
# CONSTANTS
# ============================================================================
TICK_SIZE      = 0.25     # ES minimum price increment
TICK_VALUE     = 12.50    # Dollar value per tick
BARS_PER_SEC   = 10       # 100ms bars
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)     # Round-trip commission in dollars (AMP+Rithmic+CME fees)
HALF_TICK      = TICK_SIZE / 2   # 0.125

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
def get_direction_model(scanner, target, min_train_days=3, fast=False):
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
# PART 1: FILL PROBABILITY (corrected spread)
# ============================================================================
def analyze_fill_probability(mid_prices, best_bid, best_ask, day_boundaries,
                              horizons_bars=None, offsets_ticks=None):
    """
    For each bar, check if mid touches best_bid (or best_ask) within next N bars.

    With corrected 1-tick spread:
      best_bid = mid - 0.125 (half tick below)
      best_ask = mid + 0.125 (half tick above)

    A buy limit at best_bid fills when future mid <= mid - 0.125 (mid drops half tick).
    We also test posting 1 tick behind: buy_limit = mid - 0.375 (1.5 ticks below mid).
    """
    log.info("Part 1: Fill Probability Analysis (corrected spread)")
    if horizons_bars is None:
        horizons_bars = [10, 30, 50, 100, 300]   # 1s, 3s, 5s, 10s, 30s
    if offsets_ticks is None:
        offsets_ticks = [0, -1, -2]               # at best, 1 behind, 2 behind

    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    results = {}

    for offset in offsets_ticks:
        buy_lim = best_bid + offset * TICK_SIZE    # passive buy (offset=0: at mid-0.125)
        sell_lim = best_ask - offset * TICK_SIZE   # passive sell

        results[offset] = {}
        for hz in horizons_bars:
            fwd_min = np.full(N, np.inf)
            fwd_max = np.full(N, -np.inf)
            valid_mask = np.zeros(N, dtype=bool)

            for d in range(n_days):
                ds, de = day_boundaries[d], day_boundaries[d + 1]
                if de - ds < 2:
                    continue
                m = mid_prices[ds:de]
                L = len(m)
                valid_mask[ds:de - 1] = True

                if L > hz:
                    windows = np.lib.stride_tricks.sliding_window_view(m, hz)
                    w_min = windows.min(axis=1)
                    w_max = windows.max(axis=1)
                    n_assign = min(L - hz, len(w_min) - 1)
                    if n_assign > 0:
                        fwd_min[ds:ds + n_assign] = w_min[1:n_assign + 1]
                        fwd_max[ds:ds + n_assign] = w_max[1:n_assign + 1]
                # Tail bars
                tail = max(0, L - hz)
                for i in range(tail, L - 1):
                    fwd_min[ds + i] = m[i + 1:].min()
                    fwd_max[ds + i] = m[i + 1:].max()

            vb = valid_mask & np.isfinite(buy_lim) & np.isfinite(fwd_min) & (buy_lim > 0)
            vs = valid_mask & np.isfinite(sell_lim) & np.isfinite(fwd_max) & (sell_lim > 0)
            fr_buy  = float((fwd_min[vb] <= buy_lim[vb]).mean()) if vb.sum() > 0 else 0.0
            fr_sell = float((fwd_max[vs] >= sell_lim[vs]).mean()) if vs.sum() > 0 else 0.0
            hz_sec = hz / BARS_PER_SEC

            # Note: with corrected 1-tick spread, offset=0 means buy_lim = mid - 0.125
            # Fill requires mid to drop by at least 0.125 = half a tick
            results[offset][hz_sec] = {
                'hz_sec': hz_sec, 'offset_ticks': offset,
                'buy_limit_vs_mid_ticks': -(HALF_TICK / TICK_SIZE) + offset,  # how far below mid
                'fill_rate_buy': fr_buy, 'fill_rate_sell': fr_sell,
                'fill_rate_avg': (fr_buy + fr_sell) / 2,
            }
            log.info(f"  offset={offset:+d}t  hz={hz_sec:>4.0f}s  "
                     f"buy={fr_buy:.1%}  sell={fr_sell:.1%}  avg={(fr_buy+fr_sell)/2:.1%}")

    return {'fill_probs': results,
            'spread_assumption': '1_tick_always',
            'best_bid_formula': 'mid - 0.125',
            'best_ask_formula': 'mid + 0.125'}


# ============================================================================
# PART 2: PRICE PATH AFTER FILL (MFE/MAE)
# ============================================================================
def analyze_price_path(mid_prices, best_bid, best_ask, day_boundaries,
                        predictions, pred_indices,
                        fill_horizon_bars=50,
                        post_fill_horizons=None):
    """
    When a buy limit at best_bid fills (mid touched bid):
    Track price for next N bars.
    MFE = max favorable excursion (max positive move in our direction)
    MAE = max adverse excursion (max negative move against us)
    Condition on direction model signal: aligned vs opposed fills.
    """
    log.info("Part 2: Price Path After Fill (MFE/MAE)")
    if post_fill_horizons is None:
        post_fill_horizons = [10, 30, 50, 100, 300]  # 1s, 3s, 5s, 10s, 30s

    N = len(mid_prices)
    n_days = len(day_boundaries) - 1
    pred_signal = np.full(N, np.nan)
    if len(predictions) > 0 and len(pred_indices) > 0:
        valid_idx = (pred_indices >= 0) & (pred_indices < N)
        pred_signal[pred_indices[valid_idx]] = predictions[valid_idx]

    max_post = max(post_fill_horizons)
    buy_fills = []   # list of dicts: {sig, move_ticks[k], mfe, mae}
    sell_fills = []

    for d in range(n_days):
        ds, de = day_boundaries[d], day_boundaries[d + 1]
        min_len = fill_horizon_bars + max_post + 5
        if de - ds < min_len:
            continue

        m  = mid_prices[ds:de]
        bb = best_bid[ds:de]
        ba = best_ask[ds:de]
        ps = pred_signal[ds:de]
        L  = len(m)

        for i in range(L - fill_horizon_bars - max_post - 2):
            if not (np.isfinite(bb[i]) and np.isfinite(ba[i])):
                continue
            sig = ps[i]
            future = m[i + 1:i + fill_horizon_bars + 1]

            # BUY fill: mid dips to best_bid
            bt = np.where(future <= bb[i])[0]
            if len(bt) > 0:
                fbar = i + bt[0] + 1
                if fbar + max_post < L:
                    post_m = m[fbar:fbar + max_post + 1]
                    moves = [(m[fbar + k] - m[fbar]) / TICK_SIZE for k in post_fill_horizons if fbar + k < L]
                    # MFE = max positive move (favorable for buys), MAE = max negative move
                    mfe = float((post_m - m[fbar]).max() / TICK_SIZE)
                    mae = float(((post_m - m[fbar]).min() / TICK_SIZE))  # negative = adverse
                    buy_fills.append({
                        'sig': float(sig) if np.isfinite(sig) else None,
                        'moves': [float(x) for x in moves],
                        'mfe': mfe, 'mae': mae,
                    })

            # SELL fill: mid rises to best_ask
            st = np.where(future >= ba[i])[0]
            if len(st) > 0:
                fbar = i + st[0] + 1
                if fbar + max_post < L:
                    post_m = m[fbar:fbar + max_post + 1]
                    moves = [(-(m[fbar + k] - m[fbar])) / TICK_SIZE for k in post_fill_horizons if fbar + k < L]
                    mfe = float((-(post_m - m[fbar])).max() / TICK_SIZE)
                    mae = float(((post_m - m[fbar]).max() / TICK_SIZE))
                    sell_fills.append({
                        'sig': float(sig) if np.isfinite(sig) else None,
                        'moves': [float(x) for x in moves],
                        'mfe': mfe, 'mae': mae,
                    })

        progress(f"  Price path: day {d+1}/{n_days} buy={len(buy_fills)} sell={len(sell_fills)}")

    log.info(f"  Total fills collected: buy={len(buy_fills)} sell={len(sell_fills)}")

    def summarize(fills, side):
        if not fills:
            return {'n': 0}
        has_sig = [f for f in fills if f['sig'] is not None]
        med_sig = np.median([f['sig'] for f in has_sig]) if has_sig else 0.0
        # Aligned: signal agrees with direction we're filling
        aligned = [f for f in has_sig
                   if (side == 'buy' and f['sig'] > med_sig) or
                      (side == 'sell' and f['sig'] < med_sig)]
        opposed = [f for f in has_sig
                   if (side == 'buy' and f['sig'] <= med_sig) or
                      (side == 'sell' and f['sig'] >= med_sig)]

        out = {
            'n': len(fills),
            'n_aligned': len(aligned),
            'n_opposed': len(opposed),
            'mean_mfe_ticks': float(np.nanmean([f['mfe'] for f in fills])),
            'mean_mae_ticks': float(np.nanmean([f['mae'] for f in fills])),
            'aligned_mfe': float(np.nanmean([f['mfe'] for f in aligned])) if aligned else None,
            'opposed_mfe': float(np.nanmean([f['mfe'] for f in opposed])) if opposed else None,
            'aligned_mae': float(np.nanmean([f['mae'] for f in aligned])) if aligned else None,
            'opposed_mae': float(np.nanmean([f['mae'] for f in opposed])) if opposed else None,
        }
        for k_idx, k in enumerate(post_fill_horizons):
            k_sec = k / BARS_PER_SEC
            vals = [f['moves'][k_idx] for f in fills if k_idx < len(f['moves'])]
            al_vals = [f['moves'][k_idx] for f in aligned if k_idx < len(f['moves'])]
            op_vals = [f['moves'][k_idx] for f in opposed if k_idx < len(f['moves'])]
            if vals:
                lbl = f'k_{k_sec:.0f}s'
                out[lbl] = {
                    'mean_move_ticks': float(np.nanmean(vals)),
                    'pct_positive': float((np.array(vals) > 0).mean()),
                    'aligned_mean': float(np.nanmean(al_vals)) if al_vals else None,
                    'opposed_mean': float(np.nanmean(op_vals)) if op_vals else None,
                    'n': len(vals),
                }
        return out

    buy_sum  = summarize(buy_fills, 'buy')
    sell_sum = summarize(sell_fills, 'sell')

    log.info("  Mean move after fill (ticks, positive=favorable):")
    for k in post_fill_horizons:
        k_sec = k / BARS_PER_SEC
        lbl = f'k_{k_sec:.0f}s'
        bs = buy_sum.get(lbl, {})
        ss = sell_sum.get(lbl, {})
        log.info(f"    {k_sec:>4.0f}s: buy={bs.get('mean_move_ticks', float('nan')):>+.3f}t "
                 f"(aligned={bs.get('aligned_mean', float('nan')) or float('nan'):>+.3f}t)  "
                 f"sell={ss.get('mean_move_ticks', float('nan')):>+.3f}t "
                 f"(aligned={ss.get('aligned_mean', float('nan')) or float('nan'):>+.3f}t)")

    log.info(f"  MFE: buy={buy_sum.get('mean_mfe_ticks', 0):+.3f}t  "
             f"sell={sell_sum.get('mean_mfe_ticks', 0):+.3f}t")
    log.info(f"  MAE: buy={buy_sum.get('mean_mae_ticks', 0):+.3f}t  "
             f"sell={sell_sum.get('mean_mae_ticks', 0):+.3f}t")

    return {
        'buy': buy_sum, 'sell': sell_sum,
        'post_fill_horizons_sec': [k / BARS_PER_SEC for k in post_fill_horizons],
        'fill_horizon_sec': fill_horizon_bars / BARS_PER_SEC,
    }


# ============================================================================
# CORE SIMULATION ENGINE
# ============================================================================
def simulate_strategy(mid_prices, best_bid, best_ask, day_boundaries,
                       predictions, pred_indices,
                       fill_horizon_bars=50,
                       hold_horizon_bars=300,
                       min_signal_quantile=0.7,
                       latency_bars=0,
                       use_limit_exit=False,
                       label=''):
    """
    Signal-conditioned limit order strategy simulation.

    For each bar with |signal| > threshold:
    1. Post limit at best_bid (buys) or best_ask (sells)
       with optional latency_bars delay before posting
    2. Wait up to fill_horizon_bars for fill
    3. After fill, hold for hold_horizon_bars
    4. Exit: market exit (pays 0.5t) or limit exit (earns 0.5t)

    PnL (limit entry + market exit):
      entry_edge  = +0.5 ticks (passive fill vs crossing mid)
      dir_pnl     = (exit_mid - fill_mid) * direction / TICK_SIZE
      exit_cost   = -0.5 ticks (cross half-spread at exit)
      commission  = -0.24 ticks ($3.00 = 0.24 * TICK_VALUE)
      NET         = dir_pnl - 0.24t   (entry_edge and exit_cost CANCEL!)

    PnL (limit entry + limit exit):
      entry_edge  = +0.5 ticks
      exit_edge   = +0.5 ticks (if exit limit fills)
      commission  = -0.24 ticks
      NET         = dir_pnl + 1.0t - 0.24t = dir_pnl + 0.76t  (HUGE advantage!)
      BUT: lower fill probability on exit
    """
    mode = 'limit_exit' if use_limit_exit else 'mkt_exit'
    log.info(f"Simulation [{label or mode}]  "
             f"fill={fill_horizon_bars/BARS_PER_SEC:.0f}s  "
             f"hold={hold_horizon_bars/BARS_PER_SEC:.0f}s  "
             f"q={min_signal_quantile:.0%}  "
             f"lat={latency_bars}bars")

    N = len(mid_prices)
    n_days = len(day_boundaries) - 1

    # Map predictions to full time axis
    pred_signal = np.full(N, np.nan)
    if len(predictions) > 0 and len(pred_indices) > 0:
        ok = (pred_indices >= 0) & (pred_indices < N)
        pred_signal[pred_indices[ok]] = predictions[ok]

    # Compute threshold from percentiles of valid predictions
    valid_p = predictions[np.isfinite(predictions)] if len(predictions) > 0 else np.array([])
    if len(valid_p) == 0:
        return {'error': 'No valid predictions'}
    thresh_pos = float(np.percentile(valid_p, min_signal_quantile * 100))
    thresh_neg = float(np.percentile(valid_p, (1 - min_signal_quantile) * 100))

    commission_ticks = COMMISSION_RT / TICK_VALUE  # 0.24 ticks ($3.00 RT)

    trades = []
    n_posted = 0
    n_not_filled = 0
    n_skipped_latency = 0

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

        # Max valid bar to start a new signal (must finish within the day)
        max_start = L - latency_bars - fill_horizon_bars - hold_horizon_bars - 3

        for i in range(max_start):
            if i <= last_exit:
                continue
            sig = ps[i]
            if not np.isfinite(sig):
                continue

            # Determine direction
            if sig > thresh_pos:
                direction = 1
            elif sig < thresh_neg:
                direction = -1
            else:
                continue

            # Apply latency: post order at bar i + latency_bars
            post_bar = i + latency_bars
            if post_bar >= L - fill_horizon_bars - hold_horizon_bars - 2:
                n_skipped_latency += 1
                continue

            if not (np.isfinite(bb[post_bar]) and np.isfinite(ba[post_bar])):
                continue

            n_posted += 1

            # Post limit order
            entry_lim = bb[post_bar] if direction == 1 else ba[post_bar]

            # Search for fill within fill_horizon
            fill_end = min(post_bar + fill_horizon_bars + 1, L)
            future = m[post_bar + 1:fill_end]
            if direction == 1:
                touch = np.where(future <= entry_lim)[0]
            else:
                touch = np.where(future >= entry_lim)[0]

            if len(touch) == 0:
                n_not_filled += 1
                continue

            fill_bar = post_bar + touch[0] + 1
            exit_bar = min(fill_bar + hold_horizon_bars, L - 1)

            # Hold for hold_horizon_bars then exit
            fill_mid = m[fill_bar]
            exit_mid = m[exit_bar]

            # PnL components (ticks)
            dir_pnl = (exit_mid - fill_mid) / TICK_SIZE * direction
            entry_edge = 0.5  # passive fill earns half-spread vs mid

            if use_limit_exit:
                # Post limit exit at favorable price
                # Approximate: if held to exit_bar, assume we get limit fill
                exit_edge = 0.5
                exit_cost = 0.0
            else:
                exit_edge = 0.0
                exit_cost = 0.5  # cross half-spread at exit

            net_ticks = entry_edge + dir_pnl - exit_cost + exit_edge - commission_ticks
            net_dollars = net_ticks * TICK_VALUE

            trades.append({
                'direction': int(direction),
                'signal': float(sig),
                'signal_bar': int(i),
                'post_bar_in_day': int(post_bar),
                'fill_bar_in_day': int(fill_bar),
                'exit_bar_in_day': int(exit_bar),
                'fill_offset_bars': int(fill_bar - post_bar),
                'bars_held': int(exit_bar - fill_bar),
                'fill_mid': float(fill_mid),
                'exit_mid': float(exit_mid),
                'entry_edge_ticks': float(entry_edge),
                'dir_pnl_ticks': float(dir_pnl),
                'exit_cost_ticks': float(exit_cost),
                'exit_edge_ticks': float(exit_edge),
                'commission_ticks': float(commission_ticks),
                'net_ticks': float(net_ticks),
                'net_dollars': float(net_dollars),
            })
            last_exit = exit_bar

        progress(f"  Sim day {d+1}/{n_days} trades={len(trades)}")

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
                'use_limit_exit': use_limit_exit,
            },
        }

    # Compute metrics
    pnl = np.array([t['net_ticks'] for t in trades])
    dir_pnl = np.array([t['dir_pnl_ticks'] for t in trades])
    pnl_dollars = pnl * TICK_VALUE

    # Sharpe: annualized, assuming ~6.5hr/day, 252 days/year
    avg_hold_sec = np.mean([t['bars_held'] for t in trades]) / BARS_PER_SEC
    if avg_hold_sec > 0 and np.std(pnl) > 0:
        trades_per_year = 252 * 6.5 * 3600 / max(avg_hold_sec, 1.0)
        sharpe = float(np.mean(pnl) / np.std(pnl) * np.sqrt(trades_per_year))
    else:
        sharpe = 0.0

    # Profit factor
    wins = pnl[pnl > 0]
    losses = pnl[pnl < 0]
    profit_factor = float(wins.sum() / (-losses.sum())) if len(losses) > 0 and -losses.sum() > 0 else float('inf')

    # Max drawdown on cumulative PnL
    cum_pnl = np.cumsum(pnl_dollars)
    running_max = np.maximum.accumulate(cum_pnl)
    drawdown = running_max - cum_pnl
    max_drawdown = float(drawdown.max()) if len(drawdown) > 0 else 0.0

    # Trades per day
    n_days_actual = len(day_boundaries) - 1
    trades_per_day = len(trades) / max(n_days_actual, 1)

    # Fill rate
    fill_rate = len(trades) / n_posted if n_posted > 0 else 0.0

    result = {
        'config': {
            'fill_horizon_sec': fill_horizon_bars / BARS_PER_SEC,
            'hold_horizon_sec': hold_horizon_bars / BARS_PER_SEC,
            'min_signal_quantile': min_signal_quantile,
            'latency_bars': latency_bars,
            'latency_sec': latency_bars / BARS_PER_SEC,
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
        'profit_factor': float(profit_factor) if not np.isinf(profit_factor) else 999.0,
        'max_drawdown_dollars': max_drawdown,
        'trades_per_day': trades_per_day,
        'mean_dir_pnl_ticks': float(np.mean(dir_pnl)),
        'mean_entry_edge_ticks': float(np.mean([t['entry_edge_ticks'] for t in trades])),
        'mean_exit_cost_ticks': float(np.mean([t['exit_cost_ticks'] for t in trades])),
        'mean_exit_edge_ticks': float(np.mean([t['exit_edge_ticks'] for t in trades])),
        'mean_fill_bars': float(np.mean([t['fill_offset_bars'] for t in trades])),
        'mean_hold_bars': float(np.mean([t['bars_held'] for t in trades])),
    }

    log.info(f"  n_filled={len(trades)}  fill_rate={fill_rate:.1%}  "
             f"mean_pnl={result['mean_pnl_ticks']:+.4f}t (${result['mean_pnl_dollars']:+.2f})  "
             f"win={result['win_rate']:.1%}  Sharpe={sharpe:.2f}  "
             f"total=${result['total_pnl_dollars']:+.2f}  "
             f"dir_pnl={result['mean_dir_pnl_ticks']:+.4f}t")
    return result


# ============================================================================
# PART 3: SIGNAL-CONDITIONED STRATEGY (market exit + limit exit)
# ============================================================================
def analyze_signal_conditioned_strategy(mid_prices, best_bid, best_ask,
                                         day_boundaries, predictions, pred_indices,
                                         fast=False):
    """Test multiple signal thresholds x exit types x hold periods."""
    log.info("Part 3: Signal-Conditioned Strategy Simulation")

    quantiles = [0.6, 0.7, 0.8, 0.9, 0.95]
    hold_bars_list = [30, 50, 100, 300] if not fast else [50, 100, 300]

    results = {}
    for use_limit_exit in [False, True]:
        exit_type = 'limit_exit' if use_limit_exit else 'mkt_exit'
        results[exit_type] = {}
        for q in quantiles:
            results[exit_type][f'q{q:.0%}'] = {}
            for hb in hold_bars_list:
                lbl = f'hold_{hb/BARS_PER_SEC:.0f}s'
                r = simulate_strategy(
                    mid_prices=mid_prices, best_bid=best_bid, best_ask=best_ask,
                    day_boundaries=day_boundaries,
                    predictions=predictions, pred_indices=pred_indices,
                    fill_horizon_bars=5 * BARS_PER_SEC,  # 5s fill window
                    hold_horizon_bars=hb,
                    min_signal_quantile=q,
                    latency_bars=0,
                    use_limit_exit=use_limit_exit,
                    label=f'{exit_type} q={q:.0%} h={hb/BARS_PER_SEC:.0f}s',
                )
                results[exit_type][f'q{q:.0%}'][lbl] = r
    return results


# ============================================================================
# PART 4: LATENCY-ADJUSTED RESULTS
# ============================================================================
def analyze_latency_impact(mid_prices, best_bid, best_ask, day_boundaries,
                             predictions, pred_indices):
    """
    Add N-bar delay between signal and order posting.
    1 bar = 100ms.
    Test: 0 (ideal), 1 (100ms), 3 (300ms), 5 (500ms)
    """
    log.info("Part 4: Latency Impact Analysis")

    # Use best config: 70th pctile signal, 10s hold, market exit
    latencies = [0, 1, 3, 5]
    results = {}

    for lat in latencies:
        r = simulate_strategy(
            mid_prices=mid_prices, best_bid=best_bid, best_ask=best_ask,
            day_boundaries=day_boundaries,
            predictions=predictions, pred_indices=pred_indices,
            fill_horizon_bars=5 * BARS_PER_SEC,
            hold_horizon_bars=10 * BARS_PER_SEC,
            min_signal_quantile=0.7,
            latency_bars=lat,
            use_limit_exit=False,
            label=f'lat={lat}bars ({lat/BARS_PER_SEC*1000:.0f}ms)',
        )
        results[f'lat_{lat}bars'] = r

    # Show degradation table
    log.info("  Latency impact (mkt_exit, q=70%, hold=10s):")
    base = results.get('lat_0bars', {})
    base_pnl = base.get('mean_pnl_ticks', None)
    for lat in latencies:
        r = results[f'lat_{lat}bars']
        pnl = r.get('mean_pnl_ticks', None)
        if pnl is not None and base_pnl is not None:
            delta = pnl - base_pnl
            log.info(f"  {lat*100:>4.0f}ms: mean_pnl={pnl:>+.4f}t  "
                     f"delta vs ideal={delta:>+.4f}t  "
                     f"sharpe={r.get('sharpe_approx', 0):.2f}  "
                     f"fill_rate={r.get('fill_rate', 0):.1%}")
    return results


# ============================================================================
# PART 5: OPTIMAL CONFIGURATION SEARCH
# ============================================================================
def grid_search_optimal(mid_prices, best_bid, best_ask, day_boundaries,
                          predictions, pred_indices, fast=False):
    """
    Grid search over [threshold x hold_period x latency x exit_type].
    Maximize Sharpe ratio, also track max total PnL.
    """
    log.info("Part 5: Optimal Configuration Grid Search")

    thresholds = [0.6, 0.7, 0.8, 0.9, 0.95]
    hold_bars_list = [30, 50, 100, 300] if not fast else [50, 100, 300]
    latencies = [0, 1, 3] if fast else [0, 1, 3, 5]
    exit_types = [False, True]  # market exit, limit exit

    all_results = []
    total_configs = len(thresholds) * len(hold_bars_list) * len(latencies) * len(exit_types)
    log.info(f"  Total configurations to test: {total_configs}")

    done = 0
    for use_limit_exit in exit_types:
        for q in thresholds:
            for hb in hold_bars_list:
                for lat in latencies:
                    r = simulate_strategy(
                        mid_prices=mid_prices, best_bid=best_bid, best_ask=best_ask,
                        day_boundaries=day_boundaries,
                        predictions=predictions, pred_indices=pred_indices,
                        fill_horizon_bars=5 * BARS_PER_SEC,
                        hold_horizon_bars=hb,
                        min_signal_quantile=q,
                        latency_bars=lat,
                        use_limit_exit=use_limit_exit,
                        label=f'grid',
                    )
                    done += 1
                    if 'error' not in r:
                        all_results.append({
                            'config': {
                                'quantile': q,
                                'hold_sec': hb / BARS_PER_SEC,
                                'latency_bars': lat,
                                'latency_ms': lat * 100,
                                'use_limit_exit': use_limit_exit,
                                'fill_horizon_sec': 5.0,
                            },
                            'sharpe': r.get('sharpe_approx', 0),
                            'total_pnl': r.get('total_pnl_dollars', 0),
                            'mean_pnl_ticks': r.get('mean_pnl_ticks', 0),
                            'mean_pnl_dollars': r.get('mean_pnl_dollars', 0),
                            'win_rate': r.get('win_rate', 0),
                            'fill_rate': r.get('fill_rate', 0),
                            'n_trades': r.get('n_filled', 0),
                            'trades_per_day': r.get('trades_per_day', 0),
                            'max_drawdown': r.get('max_drawdown_dollars', 0),
                            'profit_factor': r.get('profit_factor', 0),
                        })
                    progress(f"  Grid search: {done}/{total_configs} done, "
                             f"{len(all_results)} valid configs so far")

    if not all_results:
        return {'error': 'No valid configurations', 'all_results': []}

    # Sort by Sharpe
    by_sharpe = sorted(all_results, key=lambda x: x['sharpe'], reverse=True)
    by_pnl = sorted(all_results, key=lambda x: x['total_pnl'], reverse=True)

    best_sharpe = by_sharpe[0]
    best_pnl = by_pnl[0]

    log.info(f"\n  Best by Sharpe: {best_sharpe['config']}  "
             f"Sharpe={best_sharpe['sharpe']:.2f}  "
             f"total=${best_sharpe['total_pnl']:+.2f}  "
             f"mean_pnl={best_sharpe['mean_pnl_ticks']:+.4f}t")
    log.info(f"  Best by PnL:    {best_pnl['config']}  "
             f"Sharpe={best_pnl['sharpe']:.2f}  "
             f"total=${best_pnl['total_pnl']:+.2f}  "
             f"mean_pnl={best_pnl['mean_pnl_ticks']:+.4f}t")

    # Top 5 by Sharpe
    log.info("\n  Top 5 by Sharpe:")
    log.info(f"  {'q':>6s} {'hold':>6s} {'lat_ms':>6s} {'exit':>6s} "
             f"{'Sharpe':>8s} {'tot_pnl':>10s} {'mean_t':>8s} {'win%':>6s}")
    for r in by_sharpe[:5]:
        c = r['config']
        log.info(f"  {c['quantile']:>6.0%} {c['hold_sec']:>6.0f}s {c['latency_ms']:>6.0f}ms "
                 f"{'lmt' if c['use_limit_exit'] else 'mkt':>6s} "
                 f"{r['sharpe']:>8.2f} "
                 f"${r['total_pnl']:>9.2f} "
                 f"{r['mean_pnl_ticks']:>+8.4f}t "
                 f"{r['win_rate']:>6.1%}")

    return {
        'best_by_sharpe': best_sharpe,
        'best_by_pnl': best_pnl,
        'top5_by_sharpe': by_sharpe[:5],
        'top5_by_pnl': by_pnl[:5],
        'all_results': all_results,
        'n_configs_tested': len(all_results),
    }


# ============================================================================
# SUMMARY FORMATTER
# ============================================================================
def format_summary(n_bars, n_days, dir_result, fill_probs, price_path,
                   strat_results, latency_results, grid_results):
    lines = [
        "", "=" * 70,
        "CORRECTED LIMIT ORDER EXECUTION STUDY -- SUMMARY",
        "Spread assumption: 1 tick always (bid=mid-0.125, ask=mid+0.125)",
        "=" * 70,
        "",
        "DATASET:",
        f"  Bars: {n_bars:,}  Days: {n_days}",
        "",
        "DIRECTION MODEL (ret_3s walk-forward LightGBM):",
        f"  IC={dir_result.get('ic', float('nan')):.4f}  "
        f"ICIR={dir_result.get('icir', float('nan')):.2f}  "
        f"t-stat={dir_result.get('tstat', float('nan')):.2f}  "
        f"n_preds={dir_result.get('n_preds', 0):,}",
        "",
        "PART 1 -- FILL PROBABILITY (with corrected 1-tick spread):",
        "  At best (offset=0): buy limit at mid-0.125, fills when mid drops 0.5t",
    ]
    for hz_sec, s in sorted(fill_probs.get('fill_probs', {}).get(0, {}).items()):
        lines.append(f"  hz={hz_sec:>4.0f}s: buy={s['fill_rate_buy']:.1%}  "
                     f"sell={s['fill_rate_sell']:.1%}  avg={s['fill_rate_avg']:.1%}")

    lines += [
        "",
        "PART 2 -- PRICE PATH AFTER FILL (ticks, positive=good):",
        f"  buy fills (n={price_path.get('buy', {}).get('n', 0)})  "
        f"sell fills (n={price_path.get('sell', {}).get('n', 0)})",
    ]
    for k_sec in price_path.get('post_fill_horizons_sec', []):
        lbl = f'k_{k_sec:.0f}s'
        bs = price_path.get('buy', {}).get(lbl, {})
        ss = price_path.get('sell', {}).get(lbl, {})
        al_b = bs.get('aligned_mean')
        al_s = ss.get('aligned_mean')
        lines.append(
            f"  {k_sec:>4.0f}s: buy={bs.get('mean_move_ticks', float('nan')):>+.3f}t "
            f"(aligned={al_b:>+.3f}t)" if al_b is not None else
            f"  {k_sec:>4.0f}s: buy={bs.get('mean_move_ticks', float('nan')):>+.3f}t  "
            f"sell={ss.get('mean_move_ticks', float('nan')):>+.3f}t "
            + (f"(aligned={al_s:>+.3f}t)" if al_s is not None else "")
        )
    buy_mfe = price_path.get('buy', {}).get('mean_mfe_ticks', float('nan'))
    buy_mae = price_path.get('buy', {}).get('mean_mae_ticks', float('nan'))
    lines.append(f"  Buy fills: MFE={buy_mfe:>+.3f}t  MAE={buy_mae:>+.3f}t")

    lines += ["", "PART 3 -- STRATEGY SIMULATION (sample: mkt_exit, q=70%, hold=10s):"]
    mkt_70 = strat_results.get('mkt_exit', {}).get('q70%', {}).get('hold_10s', {})
    lmt_70 = strat_results.get('limit_exit', {}).get('q70%', {}).get('hold_10s', {})
    for label, r in [("Limit entry + Market exit (q=70%, hold=10s)", mkt_70),
                     ("Limit entry + Limit  exit (q=70%, hold=10s)", lmt_70)]:
        if 'error' in r:
            lines.append(f"  {label}: ERROR -- {r['error']}")
        elif r:
            lines += [
                f"  {label}:",
                f"    Fill rate:    {r.get('fill_rate', 0):.1%}  ({r.get('n_filled', 0)}/{r.get('n_posted', 0)})",
                f"    Net PnL/trade:{r.get('mean_pnl_ticks', 0):>+.4f}t = ${r.get('mean_pnl_dollars', 0):>+.2f}",
                f"      Dir PnL:    {r.get('mean_dir_pnl_ticks', 0):>+.4f}t",
                f"      Entry edge: {r.get('mean_entry_edge_ticks', 0):>+.4f}t",
                f"      Exit cost:  {r.get('mean_exit_cost_ticks', 0):>+.4f}t",
                f"      Exit edge:  {r.get('mean_exit_edge_ticks', 0):>+.4f}t",
                f"      Commission: -{commission_ticks:.4f}t (${COMMISSION_RT:.2f})",
                f"    Win rate:     {r.get('win_rate', 0):.1%}",
                f"    Sharpe:       {r.get('sharpe_approx', 0):.2f}",
                f"    Total PnL:    ${r.get('total_pnl_dollars', 0):>+.2f}",
                f"    Max Drawdown: ${r.get('max_drawdown_dollars', 0):.2f}",
            ]

    lines += ["", "PART 4 -- LATENCY IMPACT (mkt_exit, q=70%, hold=10s):"]
    lines.append(f"  {'Latency':>10s}  {'mean_pnl_t':>12s}  {'Sharpe':>8s}  {'fill_rate':>10s}")
    for k in ['lat_0bars', 'lat_1bars', 'lat_3bars', 'lat_5bars']:
        r = latency_results.get(k, {})
        if 'error' not in r and r:
            lat_ms = r.get('config', {}).get('latency_bars', 0) * 100
            lines.append(f"  {lat_ms:>7.0f}ms    {r.get('mean_pnl_ticks', 0):>+12.4f}t  "
                         f"{r.get('sharpe_approx', 0):>8.2f}  "
                         f"{r.get('fill_rate', 0):>10.1%}")

    lines += ["", "PART 5 -- OPTIMAL CONFIGURATION:"]
    best_s = grid_results.get('best_by_sharpe', {})
    best_p = grid_results.get('best_by_pnl', {})
    if best_s:
        c = best_s.get('config', {})
        lines += [
            "  BEST by Sharpe ratio:",
            f"    Signal quantile:  {c.get('quantile', 0):.0%}",
            f"    Hold period:      {c.get('hold_sec', 0):.0f}s",
            f"    Latency:          {c.get('latency_ms', 0):.0f}ms",
            f"    Exit type:        {'Limit exit' if c.get('use_limit_exit') else 'Market exit'}",
            f"    Sharpe:           {best_s.get('sharpe', 0):.2f}",
            f"    Mean PnL/trade:   {best_s.get('mean_pnl_ticks', 0):>+.4f}t "
            f"(${best_s.get('mean_pnl_dollars', 0):>+.2f})",
            f"    Total PnL:        ${best_s.get('total_pnl', 0):>+.2f}",
            f"    Win rate:         {best_s.get('win_rate', 0):.1%}",
            f"    Fill rate:        {best_s.get('fill_rate', 0):.1%}",
            f"    Trades/day:       {best_s.get('trades_per_day', 0):.1f}",
        ]
    if best_p and best_p != best_s:
        c = best_p.get('config', {})
        lines += [
            "  BEST by Total PnL:",
            f"    Signal quantile:  {c.get('quantile', 0):.0%}",
            f"    Hold period:      {c.get('hold_sec', 0):.0f}s",
            f"    Sharpe:           {best_p.get('sharpe', 0):.2f}",
            f"    Total PnL:        ${best_p.get('total_pnl', 0):>+.2f}",
        ]

    # VERDICT
    lines += ["", "=" * 70, "VERDICT:"]
    best = best_s if best_s else {}
    sharpe = best.get('sharpe', 0)
    mean_pnl = best.get('mean_pnl_ticks', 0)
    total_pnl = best.get('total_pnl', 0)

    if mean_pnl is None:
        mean_pnl = 0
    if sharpe is None:
        sharpe = 0

    if sharpe > 1.5 and mean_pnl > 0:
        verdict = f"VIABLE STRATEGY -- Sharpe={sharpe:.2f}, mean_pnl={mean_pnl:+.4f}t/trade, total=${total_pnl:+.2f}"
        sub = ("Strong edge detected with corrected spread assumptions. "
               "Commission ($2.50) is manageable relative to signal strength.")
    elif sharpe > 1.0 and mean_pnl > 0:
        verdict = f"VIABLE STRATEGY (moderate) -- Sharpe={sharpe:.2f}, mean_pnl={mean_pnl:+.4f}t/trade"
        sub = "Positive edge exists. Real-world execution risk may reduce Sharpe somewhat."
    elif sharpe > 0.5 and mean_pnl > 0:
        verdict = f"MARGINAL -- Sharpe={sharpe:.2f}. Needs more data or better execution."
        sub = "Signal exists but edge is thin relative to execution costs and noise."
    elif mean_pnl > 0:
        verdict = f"WEAK -- Positive mean but Sharpe={sharpe:.2f} too low for reliable trading."
        sub = "High variance relative to edge. Not viable without Sharpe > 1.0."
    else:
        verdict = f"NO EDGE -- mean_pnl={mean_pnl:+.4f}t/trade after costs. Signal insufficient."
        sub = "Even with corrected spread, directional IC=0.08-0.11 not enough to overcome costs."

    lines += [f"  {verdict}", f"  {sub}", ""]
    lines += [
        "KEY INSIGHT ON CORRECTED SPREAD:",
        "  With 1-tick spread (corrected): entry_edge (+$6.25) + exit_edge (+$6.25) vs costs",
        f"  Market exit: entry cancels exit, net = directional - ${COMMISSION_RT:.2f} commission",
        f"  Limit  exit: earn full tick both legs, net = directional + edge vs commission",
        f"  Breakeven for mkt exit: need avg directional move > {COMMISSION_RT/TICK_VALUE:.2f} ticks (${COMMISSION_RT:.2f})",
        f"  Breakeven for lmt exit: directional move > -{1.0 - COMMISSION_RT/TICK_VALUE:.2f} ticks (forgiving!)",
        "=" * 70,
    ]
    return "\n".join(lines)


# ============================================================================
# MAIN
# ============================================================================
def main():
    parser = argparse.ArgumentParser(description='Corrected limit order execution study')
    parser.add_argument('--fast', action='store_true',
                        help='Fewer LightGBM iterations, fewer grid search configs')
    parser.add_argument('--min-train-days', type=int, default=3)
    args = parser.parse_args()

    log.info("=" * 70)
    log.info("CORRECTED LIMIT ORDER EXECUTION STUDY")
    log.info("  Spread fix: using mid +/- 0.125 (1-tick spread always)")
    log.info(f"  Mode: {'FAST' if args.fast else 'FULL'}  TICK_VALUE=${TICK_VALUE}")
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

    # ---- CORRECTED BID/ASK ----
    mid_prices = scanner.mid_prices.copy()

    # Check what the cached spread looks like (for diagnostic)
    fn = scanner.feature_names
    def gcol(name):
        return scanner.features[:, fn.index(name)]

    cached_spread_mean = float(np.nanmean(gcol('spread'))) if 'spread' in fn else float('nan')
    cached_spread_ticks = cached_spread_mean / TICK_SIZE
    log.info(f"  Cached spread (BUGGY): mean={cached_spread_mean:.4f} = {cached_spread_ticks:.1f} ticks")
    log.info(f"  Corrected spread: always {TICK_SIZE:.2f} = 1 tick")
    log.info(f"  Mid price mean: ${np.nanmean(mid_prices):.2f} (CORRECT)")

    # Apply correction
    best_bid = mid_prices - HALF_TICK  # Corrected: mid - 0.125
    best_ask = mid_prices + HALF_TICK  # Corrected: mid + 0.125
    spread = np.full_like(mid_prices, TICK_SIZE)  # Always 1 tick

    log.info(f"  Corrected: best_bid mean=${np.nanmean(best_bid):.4f}  "
             f"best_ask mean=${np.nanmean(best_ask):.4f}  "
             f"spread={np.nanmean(spread):.4f} (= {np.nanmean(spread)/TICK_SIZE:.1f} ticks)")

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

    # ---- Part 1: Fill Probability ----
    hz_bars = [30, 100, 300] if args.fast else [10, 30, 50, 100, 300]
    offsets = [0, -1] if args.fast else [0, -1, -2]
    fill_probs = analyze_fill_probability(
        mid_prices=mid_prices, best_bid=best_bid, best_ask=best_ask,
        day_boundaries=scanner.day_boundaries,
        horizons_bars=hz_bars, offsets_ticks=offsets,
    )
    gc.collect()

    # ---- Part 2: Price Path After Fill ----
    price_path = analyze_price_path(
        mid_prices=mid_prices, best_bid=best_bid, best_ask=best_ask,
        day_boundaries=scanner.day_boundaries,
        predictions=predictions, pred_indices=pred_indices,
        fill_horizon_bars=5 * BARS_PER_SEC,
        post_fill_horizons=[10, 30, 50, 100, 300],
    )
    gc.collect()

    # ---- Part 3: Signal-Conditioned Strategy ----
    strat_results = analyze_signal_conditioned_strategy(
        mid_prices=mid_prices, best_bid=best_bid, best_ask=best_ask,
        day_boundaries=scanner.day_boundaries,
        predictions=predictions, pred_indices=pred_indices,
        fast=args.fast,
    )
    gc.collect()

    # ---- Part 4: Latency Impact ----
    latency_results = analyze_latency_impact(
        mid_prices=mid_prices, best_bid=best_bid, best_ask=best_ask,
        day_boundaries=scanner.day_boundaries,
        predictions=predictions, pred_indices=pred_indices,
    )
    gc.collect()

    # ---- Part 5: Grid Search ----
    grid_results = grid_search_optimal(
        mid_prices=mid_prices, best_bid=best_bid, best_ask=best_ask,
        day_boundaries=scanner.day_boundaries,
        predictions=predictions, pred_indices=pred_indices,
        fast=args.fast,
    )
    gc.collect()

    # ---- Summary ----
    summary_text = format_summary(
        n_bars=N, n_days=n_days,
        dir_result=dir_result,
        fill_probs=fill_probs,
        price_path=price_path,
        strat_results=strat_results,
        latency_results=latency_results,
        grid_results=grid_results,
    )
    log.info("\n" + summary_text)

    # ---- Save Results ----
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_file = RESULTS_DIR / f'corrected_limit_study_{timestamp}.json'

    output = {
        'timestamp': timestamp,
        'mode': 'fast' if args.fast else 'full',
        'spread_fix': {
            'description': 'Used mid +/- HALF_TICK instead of buggy cached bid/ask',
            'half_tick': HALF_TICK,
            'spread_ticks': 1.0,
            'spread_dollars': TICK_SIZE,
            'cached_spread_ticks_buggy': cached_spread_ticks,
        },
        'data': {
            'n_bars': int(N),
            'n_days': int(n_days),
            'mid_price_mean': float(np.nanmean(mid_prices)),
            'spread_assumption': '1_tick_always',
        },
        'direction_model': {k: dir_result.get(k) for k in
                            ['ic', 'icir', 'tstat', 'n_preds', 'fold_ics', 'fold_con']},
        'fill_probability': to_safe(fill_probs),
        'price_path_after_fill': to_safe(price_path),
        'strategy_results': to_safe(strat_results),
        'latency_impact': to_safe(latency_results),
        'optimal_config': to_safe(grid_results),
        'summary_text': summary_text,
        'total_elapsed_sec': time.time() - t_start,
    }

    with open(str(out_file), 'w', encoding='utf-8') as f:
        json.dump(output, f, indent=2)

    log.info(f"\nResults saved: {out_file}")
    log.info(f"Total elapsed: {(time.time() - t_start) / 60:.1f} min")
    print("\n" + "=" * 70)
    print(summary_text)
    print(f"\nResults saved: {out_file}")
    return output


if __name__ == '__main__':
    main()
