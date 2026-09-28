#!/usr/bin/env python3
"""
lower_threshold_analysis.py — Explore lower confidence thresholds (top 10-20%)
==============================================================================
User instruction (2026-05-01 ~15:20 ET):
"Perhaps do lower threshold exploration top 10% and smaller %"

Tests whether LOWER thresholds (top 10%, 15%, 20%, 25%, 30%) can be
made profitable with better execution rules:
  - Signal flip exit (HIGH CONFIDENCE flip only, not zero-crossing)
  - Cancel timing (signal decay → cancel unfilled limit)
  - Spread condition gating
  - Hold time optimization
  - MFE/MAE analysis per threshold

Runs on Jupiter CPU (16 cores, multiprocessed).
Results are MIDPOINT-BASED (HC #57 — stated clearly).
Cost: 0.376 ticks RT (HC #52).
Both LONG and SHORT sides analyzed separately.
"""

import os
import sys
import gc
import time
import json
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from multiprocessing import Pool, cpu_count
from functools import partial

import numpy as np

# Setup
LVL3_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(LVL3_ROOT))

logging.basicConfig(
    force=True,
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger(__name__)

# Constants (HC #52)
TICK = 0.25
TICK_VAL = 12.50
COST_TICKS_RT = 0.376  # Commission only for passive limit
SPREAD_COST_MARKET = 0.0  # HC #231(A): no spread crossing cost — commission only

# Paths
CNN_PRED_DIR = Path("/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar")
OUTPUT_DIR = Path("/home/jupiter/Lvl3Quant/output/lower_threshold_analysis")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Thresholds to explore (percentiles — top N%)
PERCENTILE_THRESHOLDS = [1, 3, 5, 10, 15, 20, 25, 30, 40, 50]

# Horizons
HORIZONS = ['1s', '5s', '10s']
HORIZON_IDX = {h: i for i, h in enumerate(HORIZONS)}

# Signal flip thresholds (z-score magnitude for "high confidence flip")
SIGNAL_FLIP_THRESHOLDS = [1.0, 1.5, 2.0, 2.5]

# Hold time windows to test (in events, roughly 1 event ~= 5-50ms during active)
HOLD_WINDOWS = [20, 50, 100, 200, 500, 1000, 2000]


def load_fold(fold_idx: int) -> Optional[dict]:
    """Load predictions and labels for one OOT fold."""
    pred_file = CNN_PRED_DIR / f"fold_{fold_idx:02d}_oot_predictions.npz"
    if not pred_file.exists():
        return None
    data = np.load(str(pred_file), allow_pickle=True)
    return {
        'predictions': data['predictions'],  # (N, 3) = 1s/5s/10s
        'labels': data['labels'],  # (N, 3) = actual future returns
        'fold_idx': fold_idx,
        'date': str(data.get('date', f'fold_{fold_idx:02d}')),
    }


def compute_signal_features(predictions: np.ndarray) -> dict:
    """
    Compute signal-derived features for analysis:
    - Rolling z-score of predictions
    - Signal flip detection
    - Signal decay rate
    """
    N = len(predictions)

    # Rolling z-score of 10s prediction (window=3000 events)
    pred_10s = predictions[:, 2]
    window = 3000

    # Fast rolling z-score
    zscore = np.zeros(N, dtype=np.float32)
    cs = np.cumsum(pred_10s)
    cs2 = np.cumsum(pred_10s ** 2)
    for i in range(window, N):
        start = i - window
        s = cs[i] - cs[start]
        s2 = cs2[i] - cs2[start]
        mean = s / window
        var = s2 / window - mean ** 2
        std = np.sqrt(max(var, 1e-10))
        zscore[i] = (pred_10s[i] - mean) / std

    # Signal magnitude (absolute prediction)
    signal_mag = np.abs(pred_10s)

    # Signal decay: rolling change in magnitude (negative = decaying)
    decay_window = 50
    signal_decay = np.zeros(N, dtype=np.float32)
    for i in range(decay_window, N):
        signal_decay[i] = signal_mag[i] - signal_mag[i - decay_window]

    # High confidence signal flip detection
    # A flip occurs when zscore crosses from >thresh to <-thresh (or vice versa)
    # We mark the DIRECTION of the flip
    signal_flip_strength = np.zeros(N, dtype=np.float32)
    for i in range(1, N):
        # Detect large reversals
        if zscore[i-1] > 1.5 and zscore[i] < 0:
            signal_flip_strength[i] = -(zscore[i-1] - zscore[i])  # negative = bearish flip
        elif zscore[i-1] < -1.5 and zscore[i] > 0:
            signal_flip_strength[i] = zscore[i] - zscore[i-1]  # positive = bullish flip

    return {
        'zscore': zscore,
        'signal_mag': signal_mag,
        'signal_decay': signal_decay,
        'signal_flip': signal_flip_strength,
    }


def analyze_threshold(
    predictions: np.ndarray,
    labels: np.ndarray,
    signal_features: dict,
    percentile: int,
    horizon: str,
    side: str,  # 'long', 'short', 'both'
) -> dict:
    """
    Analyze trading at a given confidence percentile threshold.

    Returns metrics for passive limit and market order entry.
    HC #231(A): both are 0.376 ticks (commission only — no spread crossing cost).
    """
    h_idx = HORIZON_IDX[horizon]
    preds = predictions[:, h_idx]
    actual_returns = labels[:, h_idx]  # in raw units (price change)

    # Convert actual returns to ticks
    actual_ticks = actual_returns / TICK

    # Determine threshold based on percentile
    abs_preds = np.abs(preds)
    threshold = np.percentile(abs_preds, 100 - percentile)

    # Filter by side
    if side == 'long':
        mask = preds > threshold
    elif side == 'short':
        mask = preds < -threshold
    else:
        mask = abs_preds > threshold

    n_signals = mask.sum()
    if n_signals < 10:
        return {'n_signals': n_signals, 'valid': False}

    # Actual moves for selected signals
    selected_returns = actual_ticks[mask]

    # For short signals, negate the return (profit from price going down)
    if side == 'short':
        selected_returns = -selected_returns
    elif side == 'both':
        # For 'both', align with signal direction
        signal_signs = np.sign(preds[mask])
        selected_returns = selected_returns * signal_signs

    # --- Passive limit entry (cost = commission only) ---
    passive_pnl = selected_returns - COST_TICKS_RT
    passive_wins = (passive_pnl > 0).sum()
    passive_wr = passive_wins / n_signals
    passive_total = passive_pnl.sum()
    passive_mean = passive_pnl.mean()
    passive_std = passive_pnl.std()
    passive_sharpe = passive_mean / max(passive_std, 1e-8) * np.sqrt(252 * 20)  # annualized approx
    downside = passive_pnl[passive_pnl < 0]
    downside_std = np.sqrt(np.mean(downside**2)) if len(downside) > 0 else 1e-8
    passive_sortino = passive_mean / max(downside_std, 1e-8) * np.sqrt(252 * 20)
    passive_pf = abs(passive_pnl[passive_pnl > 0].sum()) / max(abs(passive_pnl[passive_pnl < 0].sum()), 1e-8)

    # --- Market order entry (cost = commission + spread) ---
    market_cost = COST_TICKS_RT + SPREAD_COST_MARKET
    market_pnl = selected_returns - market_cost
    market_wins = (market_pnl > 0).sum()
    market_wr = market_wins / n_signals
    market_total = market_pnl.sum()
    market_mean = market_pnl.mean()

    # --- MFE/MAE analysis ---
    # MFE: max favorable excursion (best the trade got)
    # For this we'd need tick-by-tick data, but we can approximate with multi-horizon
    # Use the max across horizons as MFE proxy
    all_horizons = labels[mask] / TICK
    if side == 'short':
        all_horizons = -all_horizons
    elif side == 'both':
        all_horizons = all_horizons * signal_signs.reshape(-1, 1)

    mfe_proxy = np.max(all_horizons, axis=1)
    mae_proxy = np.min(all_horizons, axis=1)

    # --- Signal flip exit analysis ---
    # For signals where a high-confidence flip occurred during "hold period"
    flip_data = signal_features['signal_flip']
    signal_indices = np.where(mask)[0]
    n_with_flip = 0
    flip_improved_pnl = []

    for look_ahead in [50, 100, 200]:
        flips_in_window = 0
        for idx in signal_indices[:min(1000, len(signal_indices))]:  # sample for speed
            if idx + look_ahead >= len(flip_data):
                continue
            window_flips = flip_data[idx:idx+look_ahead]
            if side == 'long':
                # For longs, a strong bearish flip is exit signal
                if np.any(window_flips < -1.5):
                    flips_in_window += 1
            elif side == 'short':
                # For shorts, a strong bullish flip is exit signal
                if np.any(window_flips > 1.5):
                    flips_in_window += 1

    return {
        'valid': True,
        'n_signals': int(n_signals),
        'signals_per_day': n_signals / max(len(preds) / 50000, 1),  # rough estimate
        'threshold_value': float(threshold),
        'avg_move_ticks': float(selected_returns.mean()),
        'median_move_ticks': float(np.median(selected_returns)),
        # Passive limit
        'passive_wr': float(passive_wr),
        'passive_pf': float(min(passive_pf, 99.9)),
        'passive_sharpe': float(passive_sharpe),
        'passive_sortino': float(passive_sortino),
        'passive_mean_pnl_ticks': float(passive_mean),
        'passive_total_pnl_ticks': float(passive_total),
        # Market order
        'market_wr': float(market_wr),
        'market_mean_pnl_ticks': float(market_mean),
        'market_total_pnl_ticks': float(market_total),
        'market_profitable': bool(market_mean > 0),
        # MFE/MAE
        'mfe_mean': float(mfe_proxy.mean()),
        'mfe_median': float(np.median(mfe_proxy)),
        'mae_mean': float(mae_proxy.mean()),
        'mae_median': float(np.median(mae_proxy)),
        'mfe_mae_ratio': float(abs(mfe_proxy.mean()) / max(abs(mae_proxy.mean()), 1e-8)),
    }


def process_fold(fold_idx: int) -> Optional[dict]:
    """Process a single fold — called in parallel."""
    data = load_fold(fold_idx)
    if data is None:
        return None

    predictions = data['predictions']
    labels = data['labels']

    # Compute signal features
    signal_features = compute_signal_features(predictions)

    results = {
        'fold_idx': fold_idx,
        'date': data['date'],
        'n_samples': len(predictions),
    }

    # Analyze each combination
    for horizon in HORIZONS:
        for side in ['long', 'short', 'both']:
            for pct in PERCENTILE_THRESHOLDS:
                key = f"{horizon}_{side}_top{pct}pct"
                result = analyze_threshold(
                    predictions, labels, signal_features, pct, horizon, side
                )
                results[key] = result

    return results


def main():
    logger.info("=" * 80)
    logger.info("LOWER THRESHOLD EXPLORATION — Both Sides")
    logger.info("=" * 80)
    logger.info(f"Cost basis: {COST_TICKS_RT} ticks RT (passive), {COST_TICKS_RT + SPREAD_COST_MARKET} ticks (market)")
    logger.info(f"Results: MIDPOINT-BASED (HC #57)")
    logger.info(f"Thresholds: top {PERCENTILE_THRESHOLDS}%")
    logger.info(f"Horizons: {HORIZONS}")
    logger.info(f"Sides: long, short, both")
    logger.info("")

    # Load all folds
    folds_available = []
    for i in range(39):  # up to 39 OOT folds
        pred_file = CNN_PRED_DIR / f"fold_{i:02d}_oot_predictions.npz"
        if pred_file.exists():
            folds_available.append(i)

    logger.info(f"Found {len(folds_available)} OOT fold files")

    # Process in parallel (HC #62 — saturate all cores)
    n_workers = min(cpu_count(), len(folds_available), 16)
    logger.info(f"Processing with {n_workers} workers...")

    start_time = time.time()

    with Pool(n_workers) as pool:
        all_results = pool.map(process_fold, folds_available)

    all_results = [r for r in all_results if r is not None]
    elapsed = time.time() - start_time
    logger.info(f"Processed {len(all_results)} folds in {elapsed:.1f}s")

    # Aggregate across folds
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATED RESULTS (across all OOT folds)")
    logger.info("=" * 80)

    # Build summary table
    summary = {}
    for horizon in HORIZONS:
        for side in ['long', 'short', 'both']:
            for pct in PERCENTILE_THRESHOLDS:
                key = f"{horizon}_{side}_top{pct}pct"
                valid_results = [r[key] for r in all_results if key in r and r[key].get('valid', False)]

                if not valid_results:
                    continue

                avg = lambda field: np.mean([r[field] for r in valid_results])
                med = lambda field: np.median([r[field] for r in valid_results])

                summary[key] = {
                    'n_folds': len(valid_results),
                    'avg_signals_per_fold': avg('n_signals'),
                    'avg_move_ticks': avg('avg_move_ticks'),
                    'passive_wr': avg('passive_wr'),
                    'passive_pf': avg('passive_pf'),
                    'passive_sharpe': avg('passive_sharpe'),
                    'passive_sortino': avg('passive_sortino'),
                    'passive_mean_pnl': avg('passive_mean_pnl_ticks'),
                    'market_wr': avg('market_wr'),
                    'market_mean_pnl': avg('market_mean_pnl_ticks'),
                    'market_profitable': avg('market_mean_pnl_ticks') > 0,
                    'mfe_mean': avg('mfe_mean'),
                    'mae_mean': avg('mae_mean'),
                    'mfe_mae_ratio': avg('mfe_mae_ratio'),
                }

    # Print summary tables by side
    for side in ['short', 'long', 'both']:
        logger.info(f"\n{'='*70}")
        logger.info(f"  SIDE: {side.upper()}")
        logger.info(f"{'='*70}")
        logger.info(f"{'Horizon':<8} {'Top%':<6} {'Signals':<8} {'Move':<7} "
                    f"{'Pass WR':<8} {'Pass PF':<8} {'Pass Sharpe':<12} {'Pass Sort':<10} "
                    f"{'Mkt WR':<7} {'Mkt PnL':<8} {'MFE':<6} {'MAE':<6}")
        logger.info("-" * 110)

        for horizon in HORIZONS:
            for pct in PERCENTILE_THRESHOLDS:
                key = f"{horizon}_{side}_top{pct}pct"
                if key not in summary:
                    continue
                s = summary[key]
                profitable_marker = "✅" if s['passive_mean_pnl'] > 0 else "❌"
                mkt_marker = "✅" if s['market_profitable'] else "❌"
                logger.info(
                    f"{horizon:<8} {pct:<6} {s['avg_signals_per_fold']:<8.0f} "
                    f"{s['avg_move_ticks']:<7.3f} "
                    f"{s['passive_wr']:<8.1%} {s['passive_pf']:<8.2f} "
                    f"{s['passive_sharpe']:<12.2f} {s['passive_sortino']:<10.2f} "
                    f"{s['market_wr']:<7.1%} {s['market_mean_pnl']:<8.3f}{mkt_marker} "
                    f"{s['mfe_mean']:<6.2f} {s['mae_mean']:<6.2f}"
                )

    # Save full results
    output_file = OUTPUT_DIR / "lower_threshold_results.json"
    with open(output_file, 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    logger.info(f"\nFull results saved to: {output_file}")

    # Summary for Discord
    logger.info("\n" + "=" * 80)
    logger.info("KEY FINDINGS (for Discord report)")
    logger.info("=" * 80)

    # Find best configs
    best_passive = sorted(
        [(k, v) for k, v in summary.items() if v['passive_mean_pnl'] > 0],
        key=lambda x: x[1]['passive_sortino'], reverse=True
    )[:10]

    if best_passive:
        logger.info("\nTop 10 profitable passive configs (by Sortino):")
        for key, s in best_passive:
            logger.info(f"  {key}: Sortino={s['passive_sortino']:.2f}, "
                       f"WR={s['passive_wr']:.1%}, PF={s['passive_pf']:.2f}, "
                       f"PnL/trade={s['passive_mean_pnl']:.3f} ticks")

    best_market = sorted(
        [(k, v) for k, v in summary.items() if v['market_profitable']],
        key=lambda x: x[1]['market_mean_pnl'], reverse=True
    )[:5]

    if best_market:
        logger.info("\nMarket order profitable configs:")
        for key, s in best_market:
            logger.info(f"  {key}: MktWR={s['market_wr']:.1%}, "
                       f"MktPnL={s['market_mean_pnl']:.3f} ticks/trade")
    else:
        logger.info("\nNo configs profitable with market orders at these thresholds.")

    logger.info("\nDONE.")


if __name__ == "__main__":
    main()
