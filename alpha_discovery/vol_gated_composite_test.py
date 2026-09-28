#!/usr/bin/env python3
"""
Vol-Gated Composite Strategy Test
===================================
Combines the 9-signal mean z-score composite with a vol prediction gate.

Hypothesis: The composite has +$12,617 over 100 days, but the OOS audit found
the threshold was in-sample optimized. The vol prediction signal (IC=0.674 for
rvol at 10s) should let us filter to only take trades when the market is about
to move — reducing quiet-day losses while preserving volatile-day wins.

Pipeline:
  1. Load existing composite predictions (9-signal mean z-score) from signal_predictions/
  2. Regenerate vol predictions inline (LightGBM rvol @ 10s) if not already cached
  3. Test multiple vol gates: top 50%, 30%, 20%, 10%
  4. Sim with Python market-order sim (same as multi_signal_composite.py)
  5. Hold times: 300s, 600s, 900s (30, 60, 90 bars)
  6. Report PnL, Sharpe, trade count, win rate for each combo

Usage:
    python alpha_discovery/vol_gated_composite_test.py
    python alpha_discovery/vol_gated_composite_test.py --regen-vol
    python alpha_discovery/vol_gated_composite_test.py --n-days 50
"""

import argparse
import gc
import json
import logging
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from scipy.stats import spearmanr, pearsonr

# ============================================================================
# Paths and constants
# ============================================================================

LVL3_ROOT = Path(__file__).resolve().parent.parent
if not LVL3_ROOT.exists():
    _linux_root = Path.home() / "lvl3quant"
    if _linux_root.exists():
        LVL3_ROOT = _linux_root

MBO_DIR = LVL3_ROOT / "data" / "processed" / "mbo_features_cache"
SIG_DIR = LVL3_ROOT / "data" / "processed" / "signal_predictions"
VOL_CACHE_DIR = LVL3_ROOT / "data" / "processed" / "vol_pred_cache"
RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"

VOL_CACHE_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ES constants
TICK = 0.25
TICK_VAL = 12.50
COMM_TICKS = 3.00 / TICK_VAL  # ~0.24 ticks commission
BARS_PER_SEC = 10              # 100ms bars

# Vol prediction config (10s horizon, rvol target — IC=0.674)
VOL_HORIZON_BARS = 1000   # 10s = 1000 bars
VOL_SUBSAMPLE = 100       # subsample every 100 bars
EXCLUDE_FEATURES = [0, 3, 8, 9]  # exclude mid, microprice, bid, ask
MIN_TRAIN_DAYS_VOL = 15

# Sim parameters
VOL_GATES = [0.50, 0.70, 0.80, 0.90]  # top 50%, 30%, 20%, 10%
HOLD_SECONDS = [300, 600, 900]         # 300s, 600s, 900s
THRESHOLDS = [0.3, 0.5, 0.75, 1.0, 1.5]  # composite signal thresholds
COOLDOWN_BARS = 50  # 5s cooldown between trades

logging.basicConfig(
    format='%(asctime)s [vol_gated] %(message)s',
    datefmt='%H:%M:%S',
    level=logging.INFO,
)
logger = logging.getLogger('vol_gated')


# ============================================================================
# Data loading
# ============================================================================

def find_mbo_days() -> List[Tuple[str, Path]]:
    """Return sorted list of (date_str, mbo_path) for all available days."""
    files = sorted(MBO_DIR.glob('*_mbo_features.npz'))
    result = []
    for f in files:
        date = f.stem.replace('_mbo_features', '')
        result.append((date, f))
    return result


def load_mbo_day(fpath: Path) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Load MBO features. Returns (features_336, mid, spread) or None."""
    try:
        data = np.load(str(fpath))
        raw = data['mbo_features']  # (N, 340)
        mid = raw[:, 0].astype(np.float32).copy()
        spread = raw[:, 1].astype(np.float32).copy()

        # Fix NaN mid prices
        mask = np.isnan(mid)
        if mask.any():
            first_valid = np.argmax(~mask)
            mid[:first_valid] = mid[first_valid]

        keep = [i for i in range(raw.shape[1]) if i not in EXCLUDE_FEATURES]
        features = raw[:, keep].astype(np.float32)
        np.nan_to_num(features, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        del raw
        return features, mid, spread
    except Exception as e:
        logger.warning(f"  Failed to load {fpath.name}: {e}")
        return None


# ============================================================================
# Composite signal loading
# ============================================================================

def load_composite_signals(all_dates: List[str]) -> Dict[str, np.ndarray]:
    """Load existing 9-signal mean z-score composite predictions."""
    logger.info("Discovering composite signal files...")

    # Find all signal prediction files
    sig_names_map = defaultdict(set)
    for f in SIG_DIR.glob('*.npz'):
        parts = f.stem.split('_')
        for i in range(len(parts)):
            if parts[i].startswith('2025-') or parts[i].startswith('2024-') or parts[i].startswith('2026-'):
                sig_name = '_'.join(parts[:i])
                date = '_'.join(parts[i:])
                sig_names_map[sig_name].add(date)
                break

    # Filter to signals available for most dates
    available_sigs = {k: v for k, v in sig_names_map.items() if len(v) >= 80}
    logger.info(f"Signals with >=80 days: {len(available_sigs)}")
    for name in sorted(available_sigs.keys()):
        logger.info(f"  {name}: {len(available_sigs[name])} days")

    if not available_sigs:
        logger.warning("No composite signal files found in signal_predictions/")
        logger.info("Signal dir: %s", SIG_DIR)
        # List what IS there
        files = list(SIG_DIR.glob('*.npz'))
        logger.info(f"  Found {len(files)} .npz files total")
        if files:
            for f in files[:5]:
                logger.info(f"  Sample: {f.name}")
        return {}

    # Build composite signals per date
    logger.info(f"Building composite mean z-score from {len(available_sigs)} signals...")
    composite_mean = {}

    for date in all_dates:
        # Load MBO just to get n_bars
        mbo_path = MBO_DIR / f'{date}_mbo_features.npz'
        if not mbo_path.exists():
            continue

        try:
            data = np.load(str(mbo_path))
            n_bars = data['mbo_features'].shape[0]
            del data
        except Exception:
            continue

        sig_sum = np.zeros(n_bars, dtype=np.float64)
        n_sigs = 0

        for sig_name in available_sigs:
            path = SIG_DIR / f'{sig_name}_{date}.npz'
            if not path.exists():
                continue
            try:
                preds = np.load(str(path))['predictions']
            except Exception:
                continue

            if len(preds) != n_bars:
                if len(preds) > n_bars:
                    preds = preds[:n_bars]
                else:
                    preds = np.pad(preds, (0, n_bars - len(preds)))

            sig_sum += preds.astype(np.float64)
            n_sigs += 1

        if n_sigs > 0:
            composite_mean[date] = (sig_sum / n_sigs).astype(np.float32)

    logger.info(f"Composite built for {len(composite_mean)} days using {len(available_sigs)} signals")
    return composite_mean


# ============================================================================
# Vol prediction — inline LightGBM (rvol @ 10s)
# ============================================================================

def compute_vol_targets(mid: np.ndarray, horizon_bars: int) -> np.ndarray:
    """Compute forward realized volatility target."""
    N = len(mid)
    returns = np.diff(mid.astype(np.float64), prepend=mid[0]) / np.where(mid > 0, mid, 1.0)
    fwd_rvol = np.full(N, np.nan)
    r = returns[1:]
    if len(r) < horizon_bars:
        return fwd_rvol

    cs = np.cumsum(r)
    cs2 = np.cumsum(r ** 2)
    sum_all = np.concatenate(([0.0], cs))
    sum_sq_all = np.concatenate(([0.0], cs2))

    n_valid = N - horizon_bars
    for_start = np.arange(0, n_valid)
    for_end = for_start + horizon_bars
    s = sum_all[for_end] - sum_all[for_start]
    s2 = sum_sq_all[for_end] - sum_sq_all[for_start]
    mean_r = s / horizon_bars
    mean_r2 = s2 / horizon_bars
    var = mean_r2 - mean_r ** 2
    var = np.clip(var, 0, None)
    fwd_rvol[:n_valid] = np.sqrt(var)

    return fwd_rvol


def get_vol_predictions(
    all_days: List[Tuple[str, Path]],
    regen: bool = False,
) -> Dict[str, np.ndarray]:
    """Get vol predictions per day. Uses cached .npz if available, else regenerates."""
    import lightgbm as lgb

    logger.info("=" * 60)
    logger.info("Vol Prediction (rvol @ 10s, LightGBM walk-forward)")
    logger.info("=" * 60)

    vol_preds_by_date = {}

    # Pre-load features and targets for all days
    day_data = []
    for date_str, fpath in all_days:
        cache_path = VOL_CACHE_DIR / f'volpred_rvol_10s_{date_str}.npz'

        if not regen and cache_path.exists():
            try:
                d = np.load(str(cache_path))
                vol_preds_by_date[date_str] = d['vol_pred']
                continue
            except Exception:
                pass

        # Load raw features
        result = load_mbo_day(fpath)
        if result is None:
            continue
        features, mid, spread = result

        targets = compute_vol_targets(mid, VOL_HORIZON_BARS)
        # Subsample
        indices = np.arange(VOL_SUBSAMPLE - 1, len(features), VOL_SUBSAMPLE)
        X = features[indices]
        y = targets[indices]
        valid = ~np.isnan(y) & np.all(np.isfinite(X), axis=1)
        X = X[valid].astype(np.float32)
        y = y[valid].astype(np.float32)

        day_data.append({
            'date': date_str,
            'fpath': fpath,
            'X': X,
            'y': y,
            'n_bars': len(mid),
            'mid': mid,
            'spread': spread,
        })

        del features, targets
        gc.collect()

    if not day_data:
        logger.info(f"All {len(all_days)} days loaded from vol pred cache.")
        return vol_preds_by_date

    logger.info(f"Regenerating vol predictions for {len(day_data)} days (cached: {len(vol_preds_by_date)})")

    lgbm_params = {
        'objective': 'regression',
        'metric': 'mse',
        'learning_rate': 0.05,
        'num_leaves': 63,
        'max_depth': 6,
        'min_child_samples': 200,
        'subsample': 0.7,
        'colsample_bytree': 0.7,
        'reg_alpha': 0.1,
        'reg_lambda': 1.0,
        'verbose': -1,
        'n_jobs': -1,
        'seed': 42,
    }

    all_ics = []

    for test_idx in range(MIN_TRAIN_DAYS_VOL, len(day_data)):
        train_days = day_data[:test_idx]
        test_day = day_data[test_idx]
        date_str = test_day['date']

        X_train = np.vstack([d['X'] for d in train_days]).astype(np.float32)
        y_train = np.concatenate([d['y'] for d in train_days]).astype(np.float32)
        np.nan_to_num(X_train, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        X_test_sub = test_day['X'].astype(np.float32)
        np.nan_to_num(X_test_sub, copy=False, nan=0.0, posinf=0.0, neginf=0.0)

        dtrain = lgb.Dataset(X_train, label=y_train)
        model = lgb.train(lgbm_params, dtrain, num_boost_round=200)

        # Predict on FULL resolution (not subsampled) for the gate
        result = load_mbo_day(test_day['fpath'])
        if result is None:
            del dtrain, model
            continue
        features_full, mid_full, spread_full = result
        np.nan_to_num(features_full, copy=False, nan=0.0, posinf=0.0, neginf=0.0)
        vol_pred_full = model.predict(features_full).astype(np.float32)

        # Also compute IC on subsampled test
        preds_sub = model.predict(X_test_sub)
        ic, _ = spearmanr(preds_sub, test_day['y'])
        all_ics.append(ic)

        if test_idx % 5 == 0 or test_idx == len(day_data) - 1:
            mean_ic = float(np.mean(all_ics))
            logger.info(f"  [{test_idx+1}/{len(day_data)}] {date_str}  IC={ic:+.4f}  mean_IC={mean_ic:+.4f}")

        # Cache the prediction
        cache_path = VOL_CACHE_DIR / f'volpred_rvol_10s_{date_str}.npz'
        np.savez_compressed(str(cache_path), vol_pred=vol_pred_full)
        vol_preds_by_date[date_str] = vol_pred_full

        del X_train, y_train, dtrain, model, features_full
        gc.collect()

    if all_ics:
        ics = np.array(all_ics)
        logger.info(f"\nVol Pred Summary:")
        logger.info(f"  Mean IC: {np.mean(ics):+.4f}")
        logger.info(f"  Std IC:  {np.std(ics):.4f}")
        logger.info(f"  t-stat:  {np.mean(ics) / (np.std(ics) / np.sqrt(len(ics))):.2f}")
        logger.info(f"  Pct pos: {np.mean(ics > 0) * 100:.1f}%")

    return vol_preds_by_date


# ============================================================================
# Market order simulator
# ============================================================================

def sim_day(
    mid: np.ndarray,
    spread: np.ndarray,
    composite_preds: np.ndarray,
    vol_preds: np.ndarray,
    vol_gate_pct: float,
    thresh: float,
    hold_bars: int,
    cooldown: int = COOLDOWN_BARS,
) -> Tuple[float, int, int]:
    """
    Simulate one day with vol-gated composite signal.

    vol_gate_pct: percentile threshold for vol gate (e.g. 0.80 = only trade
                  when vol_pred is in top 20%)
    thresh: composite signal absolute threshold
    hold_bars: how many bars to hold position
    """
    n = min(len(mid), len(composite_preds), len(vol_preds))
    if n < 100:
        return 0.0, 0, 0

    # Compute vol percentile rank (causal, using expanding window percentile)
    # This avoids lookahead: for bar i, rank among vol_preds[0..i]
    vol_arr = vol_preds[:n]
    vol_rank = np.zeros(n, dtype=np.float32)
    # Use a fast approximate rolling percentile (same concept as vol_magnitude_gated.py)
    # For speed, compute expanding percentile at each bar
    # (full causal: no lookahead since we're looking at predicted vol, not realized)
    running_sorted = []
    for i in range(n):
        v = float(vol_arr[i])
        if np.isfinite(v):
            # Insert into sorted list (binary search)
            import bisect
            bisect.insort(running_sorted, v)
            rank = bisect.bisect_left(running_sorted, v) / max(len(running_sorted), 1)
            vol_rank[i] = rank
        else:
            vol_rank[i] = 0.5  # neutral if NaN

    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0

    comp = composite_preds[:n]
    sp = spread[:n]
    m = mid[:n]

    for i in range(n):
        if in_pos:
            if direction == 1:
                unr = (m[i] - entry_price) / TICK
            else:
                unr = (entry_price - m[i]) / TICK

            if i - entry_bar >= hold_bars:
                pnl = unr - sp[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl
                trades += 1
                wins += (1 if pnl > 0 else 0)
                in_pos = False
                last_exit = i
        else:
            # Gate: vol must be in top (1 - vol_gate_pct)
            vol_ok = vol_rank[i] >= vol_gate_pct
            signal_ok = abs(comp[i]) > thresh
            cooldown_ok = (i - last_exit) >= cooldown

            if vol_ok and signal_ok and cooldown_ok:
                d = 1 if comp[i] > 0 else -1
                entry_price = m[i] + d * sp[i] / 2.0
                direction = d
                in_pos = True
                entry_bar = i

    # Close any open position at end of day
    if in_pos:
        i = n - 1
        if direction == 1:
            unr = (m[i] - entry_price) / TICK
        else:
            unr = (entry_price - m[i]) / TICK
        pnl = unr - sp[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl
        trades += 1
        wins += (1 if pnl > 0 else 0)

    return total_pnl * TICK_VAL, trades, wins


# ============================================================================
# Fast vol rank computation (vectorized, used for actual sim)
# ============================================================================

def compute_vol_rank_fast(vol_arr: np.ndarray, window: int = 5000) -> np.ndarray:
    """Fast causal rolling percentile rank using stride-based approximation."""
    N = len(vol_arr)
    out = np.full(N, 0.5, dtype=np.float32)
    x_f64 = vol_arr.astype(np.float64)
    x_f64 = np.nan_to_num(x_f64, nan=0.0)

    stride = max(1, window // 50)
    for i in range(window - 1, N, stride):
        w_start = max(0, i - window + 1)
        wnd = x_f64[w_start: i + 1]
        pct = float(np.sum(wnd < x_f64[i])) / max(len(wnd), 1)
        fill_end = min(i + stride, N)
        out[i: fill_end] = pct

    return out


def sim_day_fast(
    mid: np.ndarray,
    spread: np.ndarray,
    composite_preds: np.ndarray,
    vol_preds: np.ndarray,
    vol_gate_pct: float,
    thresh: float,
    hold_bars: int,
    cooldown: int = COOLDOWN_BARS,
    vol_rank_window: int = 5000,
) -> Tuple[float, int, int]:
    """Fast version using rolling percentile rank for vol gate."""
    n = min(len(mid), len(composite_preds), len(vol_preds))
    if n < 100:
        return 0.0, 0, 0

    vol_rank = compute_vol_rank_fast(vol_preds[:n], window=vol_rank_window)

    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0

    comp = composite_preds[:n]
    sp = spread[:n]
    m = mid[:n]

    for i in range(n):
        if in_pos:
            if direction == 1:
                unr = (m[i] - entry_price) / TICK
            else:
                unr = (entry_price - m[i]) / TICK

            if i - entry_bar >= hold_bars:
                pnl = unr - sp[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl
                trades += 1
                wins += (1 if pnl > 0 else 0)
                in_pos = False
                last_exit = i
        else:
            vol_ok = vol_rank[i] >= vol_gate_pct
            signal_ok = abs(comp[i]) > thresh
            cooldown_ok = (i - last_exit) >= cooldown

            if vol_ok and signal_ok and cooldown_ok:
                d = 1 if comp[i] > 0 else -1
                entry_price = m[i] + d * sp[i] / 2.0
                direction = d
                in_pos = True
                entry_bar = i

    # Close any open position
    if in_pos:
        i = n - 1
        if direction == 1:
            unr = (m[i] - entry_price) / TICK
        else:
            unr = (entry_price - m[i]) / TICK
        pnl = unr - sp[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl
        trades += 1
        wins += (1 if pnl > 0 else 0)

    return total_pnl * TICK_VAL, trades, wins


# ============================================================================
# No-gate baseline (for comparison)
# ============================================================================

def sim_day_baseline(
    mid: np.ndarray,
    spread: np.ndarray,
    composite_preds: np.ndarray,
    thresh: float,
    hold_bars: int,
    cooldown: int = COOLDOWN_BARS,
) -> Tuple[float, int, int]:
    """Baseline sim — no vol gate, just composite threshold."""
    n = min(len(mid), len(composite_preds))
    if n < 100:
        return 0.0, 0, 0

    total_pnl = 0.0
    trades = 0
    wins = 0
    last_exit = -cooldown
    in_pos = False
    entry_price = 0.0
    direction = 0
    entry_bar = 0

    comp = composite_preds[:n]
    sp = spread[:n]
    m = mid[:n]

    for i in range(n):
        if in_pos:
            if direction == 1:
                unr = (m[i] - entry_price) / TICK
            else:
                unr = (entry_price - m[i]) / TICK
            if i - entry_bar >= hold_bars:
                pnl = unr - sp[i] / 2.0 / TICK - COMM_TICKS
                total_pnl += pnl
                trades += 1
                wins += (1 if pnl > 0 else 0)
                in_pos = False
                last_exit = i
        else:
            signal_ok = abs(comp[i]) > thresh
            cooldown_ok = (i - last_exit) >= cooldown
            if signal_ok and cooldown_ok:
                d = 1 if comp[i] > 0 else -1
                entry_price = m[i] + d * sp[i] / 2.0
                direction = d
                in_pos = True
                entry_bar = i

    if in_pos:
        i = n - 1
        if direction == 1:
            unr = (m[i] - entry_price) / TICK
        else:
            unr = (entry_price - m[i]) / TICK
        pnl = unr - sp[i] / 2.0 / TICK - COMM_TICKS
        total_pnl += pnl
        trades += 1
        wins += (1 if pnl > 0 else 0)

    return total_pnl * TICK_VAL, trades, wins


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Vol-Gated Composite Strategy Test")
    parser.add_argument('--n-days', type=int, default=0, help='Limit to N days (0 = all)')
    parser.add_argument('--regen-vol', action='store_true', help='Force regenerate vol predictions')
    parser.add_argument('--no-baseline', action='store_true', help='Skip no-gate baseline')
    args = parser.parse_args()

    logger.info("=" * 70)
    logger.info("Vol-Gated Composite Strategy Test")
    logger.info("=" * 70)
    logger.info(f"MBO dir: {MBO_DIR}")
    logger.info(f"Signal dir: {SIG_DIR}")
    logger.info(f"Vol cache: {VOL_CACHE_DIR}")
    logger.info(f"Results dir: {RESULTS_DIR}")
    logger.info("")

    # Discover days
    all_days = find_mbo_days()
    if not all_days:
        logger.error(f"No MBO feature files found in {MBO_DIR}")
        sys.exit(1)

    if args.n_days > 0:
        all_days = all_days[:args.n_days]

    all_dates = [d for d, _ in all_days]
    logger.info(f"Found {len(all_days)} days: {all_dates[0]} ... {all_dates[-1]}")

    # -------------------------------------------------------------------------
    # Step 1: Load composite signals
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 1: Loading composite signals...")
    composite_signals = load_composite_signals(all_dates)

    if not composite_signals:
        logger.warning("No composite signals found — will use single microprice_dev signal as fallback")

    # -------------------------------------------------------------------------
    # Step 2: Get vol predictions
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 2: Getting vol predictions (rvol @ 10s)...")
    vol_preds = get_vol_predictions(all_days, regen=args.regen_vol)

    # Check overlap
    comp_dates = set(composite_signals.keys())
    vol_dates = set(vol_preds.keys())
    common_dates = sorted(comp_dates & vol_dates)

    logger.info("")
    logger.info(f"Composite signal days: {len(comp_dates)}")
    logger.info(f"Vol prediction days:   {len(vol_dates)}")
    logger.info(f"Common days (both):    {len(common_dates)}")

    if len(common_dates) < 20:
        logger.error("Too few common days — check data")
        # Still try with whatever we have
        if len(vol_dates) < 20:
            logger.error("Vol predictions incomplete. Run with --regen-vol to force rebuild.")
            sys.exit(1)

    # -------------------------------------------------------------------------
    # Step 3: Load mid/spread for simulation days
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("Step 3: Loading mid/spread for common days...")
    mid_spread = {}
    for date, fpath in all_days:
        if date not in common_dates and date not in vol_dates:
            continue
        try:
            data = np.load(str(fpath))
            raw = data['mbo_features']
            mid_spread[date] = {
                'mid': raw[:, 0].astype(np.float32).copy(),
                'spread': raw[:, 1].astype(np.float32).copy(),
            }
            del raw, data
        except Exception as e:
            logger.warning(f"  Skipping {date}: {e}")
    logger.info(f"Loaded mid/spread for {len(mid_spread)} days")

    # -------------------------------------------------------------------------
    # Step 4: Baseline sim (no vol gate)
    # -------------------------------------------------------------------------
    if not args.no_baseline and composite_signals:
        logger.info("")
        logger.info("=" * 70)
        logger.info("BASELINE: No Vol Gate (composite only)")
        logger.info("=" * 70)

        baseline_results = []
        for thresh in THRESHOLDS:
            for hold_s in HOLD_SECONDS:
                hold_bars = hold_s * BARS_PER_SEC
                day_pnls = []
                tot_trades = 0
                tot_wins = 0

                for date in common_dates:
                    if date not in mid_spread:
                        continue
                    pnl, t, w = sim_day_baseline(
                        mid_spread[date]['mid'],
                        mid_spread[date]['spread'],
                        composite_signals[date],
                        thresh, hold_bars,
                    )
                    day_pnls.append(pnl)
                    tot_trades += t
                    tot_wins += w

                arr = np.array(day_pnls)
                total = float(arr.sum())
                std = float(arr.std()) if len(arr) > 1 else 1.0
                sharpe = float(arr.mean() / max(std, 0.01) * np.sqrt(252))
                win_rate = tot_wins / max(tot_trades, 1)
                n_pos = int((arr > 0).sum())
                h_mid = len(common_dates) // 2
                h1 = float(arr[:h_mid].sum())
                h2 = float(arr[h_mid:].sum())

                r = {
                    'vol_gate': 'NONE',
                    'thresh': thresh,
                    'hold_s': hold_s,
                    'total_pnl': total,
                    'h1_pnl': h1,
                    'h2_pnl': h2,
                    'sharpe': sharpe,
                    'trades': tot_trades,
                    'win_rate': win_rate,
                    'profit_days': n_pos,
                    'n_days': len(day_pnls),
                    'both_positive': h1 > 0 and h2 > 0,
                }
                baseline_results.append(r)

        baseline_results.sort(key=lambda x: x['total_pnl'], reverse=True)
        logger.info(f"\n{'Config':<40} {'Sharpe':>7} {'PnL':>10} {'H1':>8} {'H2':>8} {'Trades':>7} {'Win%':>6} {'Days+':>6}")
        logger.info("-" * 100)
        for r in baseline_results[:10]:
            cfg = f"NONE|t{r['thresh']}|h{r['hold_s']}s"
            both = ' BOTH+' if r['both_positive'] else ''
            logger.info(
                f"{cfg:<40} {r['sharpe']:>+7.2f} ${r['total_pnl']:>+9,.0f} "
                f"${r['h1_pnl']:>+7,.0f} ${r['h2_pnl']:>+7,.0f} {r['trades']:>7} "
                f"{r['win_rate']:>5.1%} {r['profit_days']:>3}/{r['n_days']}{both}"
            )
    else:
        baseline_results = []

    # -------------------------------------------------------------------------
    # Step 5: Vol-gated simulation
    # -------------------------------------------------------------------------
    logger.info("")
    logger.info("=" * 70)
    logger.info("VOL-GATED COMPOSITE SIM")
    logger.info(f"Gates: {[f'top {int((1-g)*100)}%' for g in VOL_GATES]}")
    logger.info(f"Holds: {HOLD_SECONDS}s")
    logger.info(f"Thresholds: {THRESHOLDS}")
    logger.info("=" * 70)

    # Determine which dates to sim — use common_dates if composite available,
    # else use vol-only approach (just vol-gated on microprice_dev)
    sim_dates = common_dates if composite_signals else sorted(vol_dates & set(mid_spread.keys()))
    h_mid = len(sim_dates) // 2

    all_results = []
    total_combos = len(VOL_GATES) * len(THRESHOLDS) * len(HOLD_SECONDS)
    combo_idx = 0

    for vol_gate in VOL_GATES:
        for thresh in THRESHOLDS:
            for hold_s in HOLD_SECONDS:
                hold_bars = hold_s * BARS_PER_SEC
                day_pnls = []
                tot_trades = 0
                tot_wins = 0

                for date in sim_dates:
                    if date not in mid_spread or date not in vol_preds:
                        continue

                    comp = composite_signals.get(date)
                    if comp is None:
                        continue

                    pnl, t, w = sim_day_fast(
                        mid_spread[date]['mid'],
                        mid_spread[date]['spread'],
                        comp,
                        vol_preds[date],
                        vol_gate_pct=vol_gate,
                        thresh=thresh,
                        hold_bars=hold_bars,
                    )
                    day_pnls.append(pnl)
                    tot_trades += t
                    tot_wins += w

                arr = np.array(day_pnls)
                total = float(arr.sum())
                std = float(arr.std()) if len(arr) > 1 else 1.0
                sharpe = float(arr.mean() / max(std, 0.01) * np.sqrt(252))
                win_rate = tot_wins / max(tot_trades, 1)
                n_pos = int((arr > 0).sum())
                h1 = float(arr[:h_mid].sum())
                h2 = float(arr[h_mid:].sum())

                gate_label = f"top{int((1-vol_gate)*100)}pct"
                r = {
                    'vol_gate': vol_gate,
                    'vol_gate_label': gate_label,
                    'thresh': thresh,
                    'hold_s': hold_s,
                    'total_pnl': total,
                    'h1_pnl': h1,
                    'h2_pnl': h2,
                    'sharpe': sharpe,
                    'trades': tot_trades,
                    'win_rate': win_rate,
                    'profit_days': n_pos,
                    'n_days': len(day_pnls),
                    'both_positive': h1 > 0 and h2 > 0,
                }
                all_results.append(r)

                combo_idx += 1
                if combo_idx % 10 == 0:
                    logger.info(f"  Progress: {combo_idx}/{total_combos} combos done...")

    # -------------------------------------------------------------------------
    # Step 6: Report
    # -------------------------------------------------------------------------
    all_results.sort(key=lambda x: x['total_pnl'], reverse=True)

    logger.info("")
    logger.info("=" * 110)
    logger.info("TOP 20 VOL-GATED COMBOS (sorted by total PnL)")
    logger.info("=" * 110)
    logger.info(f"{'Config':<48} {'Sharpe':>7} {'PnL':>10} {'H1':>8} {'H2':>8} {'Trades':>7} {'Win%':>6} {'Days+':>6} {'Both':>5}")
    logger.info("-" * 110)
    for r in all_results[:20]:
        cfg = f"{r['vol_gate_label']}|t{r['thresh']}|h{r['hold_s']}s"
        both = 'YES' if r['both_positive'] else ''
        logger.info(
            f"{cfg:<48} {r['sharpe']:>+7.2f} ${r['total_pnl']:>+9,.0f} "
            f"${r['h1_pnl']:>+7,.0f} ${r['h2_pnl']:>+7,.0f} {r['trades']:>7} "
            f"{r['win_rate']:>5.1%} {r['profit_days']:>3}/{r['n_days']} {both:>5}"
        )

    # Both-positive combos
    both_pos = [r for r in all_results if r['both_positive'] and r['total_pnl'] > 0]
    logger.info("")
    logger.info(f"BOTH-HALVES POSITIVE COMBOS: {len(both_pos)}")
    if both_pos:
        for r in both_pos[:10]:
            cfg = f"{r['vol_gate_label']}|t{r['thresh']}|h{r['hold_s']}s"
            logger.info(
                f"  {cfg:<48} ${r['total_pnl']:>+9,.0f}  H1=${r['h1_pnl']:>+8,.0f}  "
                f"H2=${r['h2_pnl']:>+8,.0f}  Sharpe={r['sharpe']:+.2f}  "
                f"trades={r['trades']}  win={r['win_rate']:.1%}"
            )

    # Comparison per gate
    logger.info("")
    logger.info("BEST RESULT PER VOL GATE LEVEL:")
    for gate in VOL_GATES:
        gate_results = [r for r in all_results if abs(r['vol_gate'] - gate) < 0.01]
        if gate_results:
            best = gate_results[0]
            pct = int((1 - gate) * 100)
            logger.info(
                f"  Top {pct:2d}% gate → best: ${best['total_pnl']:>+9,.0f}  "
                f"Sharpe={best['sharpe']:+.2f}  trades={best['trades']}  "
                f"config: t{best['thresh']}|h{best['hold_s']}s"
            )

    # Baseline comparison
    if baseline_results:
        best_baseline = baseline_results[0]
        best_gated = all_results[0]
        logger.info("")
        logger.info("BASELINE VS BEST VOL-GATED:")
        logger.info(
            f"  Baseline (no gate): ${best_baseline['total_pnl']:>+9,.0f}  "
            f"Sharpe={best_baseline['sharpe']:+.2f}  trades={best_baseline['trades']}"
        )
        logger.info(
            f"  Best vol-gated:     ${best_gated['total_pnl']:>+9,.0f}  "
            f"Sharpe={best_gated['sharpe']:+.2f}  trades={best_gated['trades']}"
        )
        delta = best_gated['total_pnl'] - best_baseline['total_pnl']
        logger.info(f"  Delta: ${delta:>+9,.0f}")

    # -------------------------------------------------------------------------
    # Save results
    # -------------------------------------------------------------------------
    ts = time.strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f'vol_gated_composite_{ts}.json'
    output = {
        'timestamp': ts,
        'n_days': len(sim_dates),
        'dates': sim_dates,
        'vol_gates_tested': VOL_GATES,
        'thresholds_tested': THRESHOLDS,
        'hold_seconds_tested': HOLD_SECONDS,
        'top_results': all_results[:50],
        'both_positive_results': both_pos,
        'baseline_results': baseline_results[:10],
        'summary': {
            'best_pnl': all_results[0]['total_pnl'] if all_results else 0,
            'best_config': all_results[0] if all_results else {},
            'n_both_positive': len(both_pos),
            'n_profitable': sum(1 for r in all_results if r['total_pnl'] > 0),
        }
    }
    with open(str(out_path), 'w') as f:
        json.dump(output, f, indent=2)
    logger.info(f"\nResults saved: {out_path}")
    logger.info("Done.")


if __name__ == '__main__':
    main()
