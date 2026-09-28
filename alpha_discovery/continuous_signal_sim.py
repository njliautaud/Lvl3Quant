"""
Continuous Signal Trading Simulation
=====================================
Implements CONTINUOUS HOLD strategies where positions are held until the
signal flips or weakens, rather than fixed hold periods (5s, 10s, 60s).

Hypothesis: The model's rolling features give it temporal context, so holding
while conditions remain favorable should capture more of the move. The Rust
MBO sim's --signal-flip-exit flag is bugged, so this implements it in Python.

Four strategies:
  A) Hold-Until-Flip      - Exit when prediction changes sign
  B) Hold-Until-Weaken    - Hysteresis exit (entry > thresh, exit < weaker thresh)
  C) Trailing-Signal-Stop - Exit when signal drops below peak * decay_factor
  D) Regime-Continuous    - vol_regime determines hold behavior

Fill model: limit order at mid, fill after 10 bars if mid doesn't move away
by > 1 tick. If unfilled after 50 bars, market order at -1 tick penalty.

Usage:
    python alpha_discovery/continuous_signal_sim.py
    python alpha_discovery/continuous_signal_sim.py --strategy flip --signal meta_3s
    python alpha_discovery/continuous_signal_sim.py --n-days 20 --quick
    python alpha_discovery/continuous_signal_sim.py --generate  # fresh Ridge predictions
"""

import sys
import json
import time
import argparse
import logging
import gc
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple

import numpy as np
from scipy.stats import spearmanr

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
LVL3_ROOT = Path(__file__).parent.parent
# Support both Windows and Linux paths
if not LVL3_ROOT.exists():
    LVL3_ROOT = Path.home() / 'lvl3quant'

MBO_FEAT_DIR = LVL3_ROOT / 'data' / 'processed' / 'mbo_features_cache'
SIGNAL_DIR   = LVL3_ROOT / 'data' / 'processed' / 'signal_predictions'
RESULTS_DIR  = LVL3_ROOT / 'alpha_discovery' / 'results'
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Add project root so imports work
sys.path.insert(0, str(LVL3_ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TICK_SIZE        = 0.25
TICK_VALUE       = 12.50      # $12.50 per tick (ES micro)
COMMISSION_RT = 4.70  # HC #52: $4.70 RT (AMP)       # $3.00 round-trip
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE  # 0.24 ticks
HALF_TICK        = TICK_SIZE / 2               # 0.125
BARS_PER_SEC     = 10                          # 100ms bars

# Fill model timing (in bars)
LIMIT_FILL_BARS  = 10    # try limit for 10 bars (1 second)
MARKET_FILL_BARS = 50    # abandon limit → market after 50 bars (5 seconds)
MAX_HOLD_BARS    = 600   # force exit after 60 seconds

# MBO feature column indices (from mbo_features.get_feature_names())
COL_MID          = 0
COL_SPREAD       = 1
COL_VOL_IMBAL    = 2
COL_MICROPRICE   = 3
COL_VOL_REGIME   = 195   # rvol-based regime score (continuous 0-2.2)

# Vol regime thresholds (empirical p33/p67 from 50 days of data)
VOL_REGIME_LOW_THRESH  = 0.35   # < this = LOW vol (calm, take quick exits)
VOL_REGIME_HIGH_THRESH = 1.10   # > this = HIGH vol (let it run)
# MEDIUM: 0.35 <= vol_regime <= 1.10

# Comparison baseline: fixed 10s hold (100 bars)
FIXED_HOLD_BARS = 100

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = RESULTS_DIR / f'continuous_signal_sim_{_ts}.log'
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
logger = logging.getLogger('continuous_signal_sim')


# ===========================================================================
# Trade Record
# ===========================================================================
@dataclass
class ContinuousTrade:
    """Record of one continuous-hold trade."""
    day:             int
    strategy:        str        # 'flip', 'weaken', 'trailing', 'regime'
    direction:       int        # +1 long, -1 short
    signal_bar:      int        # bar where signal fired
    entry_bar:       int        # bar where filled
    exit_bar:        int        # bar where exited
    entry_price:     float
    exit_price:      float
    signal_at_entry: float      # prediction value at entry
    signal_at_exit:  float      # prediction value at exit
    peak_signal:     float      # max |prediction| during hold
    hold_bars:       int
    entry_type:      str        # 'limit' or 'market'
    exit_type:       str        # 'flip', 'weaken', 'trailing', 'timeout', 'eod', 'market'
    # PnL breakdown
    dir_pnl_ticks:   float      # directional move in ticks
    entry_edge:      float      # +0.5 if limit entry, -0.5 if market
    exit_edge:       float      # +0.5 if limit exit, -0.5 if market
    commission:      float      # always COMMISSION_TICKS (once per RT)
    net_ticks:       float
    net_dollars:     float
    # Comparison
    fixed10s_pnl_ticks: float   # what fixed 10s would have gotten
    fixed10s_pnl_usd:   float


# ===========================================================================
# Data Loading
# ===========================================================================
def discover_dates(signal_prefix: str) -> List[str]:
    """Return sorted list of dates that have both signal and feature files."""
    pred_files = sorted(SIGNAL_DIR.glob(f'{signal_prefix}_*.npz'))
    dates = []
    for pf in pred_files:
        # Extract date from filename, e.g. meta_global_3s_2025-08-11 -> 2025-08-11
        stem = pf.stem
        date_part = stem.replace(signal_prefix + '_', '')
        # Validate it looks like a date
        if len(date_part) == 10 and date_part[4] == '-' and date_part[7] == '-':
            feat_file = MBO_FEAT_DIR / f'{date_part}_mbo_features.npz'
            if feat_file.exists():
                dates.append(date_part)
    return sorted(dates)


def load_day_data(date_str: str, signal_prefix: str) -> Optional[Dict]:
    """Load one day of features + predictions. Returns None if missing."""
    feat_path = MBO_FEAT_DIR / f'{date_str}_mbo_features.npz'
    pred_path = SIGNAL_DIR / f'{signal_prefix}_{date_str}.npz'

    if not feat_path.exists() or not pred_path.exists():
        return None

    try:
        feat_data = np.load(str(feat_path))
        pred_data = np.load(str(pred_path))

        features    = feat_data['mbo_features'].astype(np.float32)   # (N, 340)
        predictions = pred_data['predictions'].astype(np.float64)    # (N,)
        mid_prices  = pred_data['mid_prices'].astype(np.float32)     # (N,)

        if len(features) != len(predictions) or len(predictions) != len(mid_prices):
            logger.warning(f'{date_str}: shape mismatch — feat={len(features)}, '
                           f'pred={len(predictions)}, mid={len(mid_prices)}')
            return None

        return {
            'date':        date_str,
            'features':    features,
            'predictions': predictions,
            'mid_prices':  mid_prices,
            'n_bars':      len(predictions),
        }
    except Exception as e:
        logger.warning(f'Failed to load {date_str}: {e}')
        return None


# ===========================================================================
# Ridge Walk-Forward Signal Generation (--generate mode)
# ===========================================================================
def generate_ridge_predictions(dates: List[str], train_days: int = 20) -> bool:
    """
    Generate fresh Ridge walk-forward predictions and save to SIGNAL_DIR as
    meta_global_3s_{date}.npz.  Uses MBO features as input, 3s forward return
    as target (matching the existing meta_global_3s signal convention).

    Returns True if successful.
    """
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler

    HORIZON_BARS = 30  # 3 seconds

    logger.info(f'Generating Ridge walk-forward predictions for {len(dates)} days '
                f'(train_days={train_days})')

    # Load all days
    all_features  = []
    all_mids      = []
    valid_dates   = []

    for d in dates:
        feat_path = MBO_FEAT_DIR / f'{d}_mbo_features.npz'
        if not feat_path.exists():
            continue
        try:
            fd = np.load(str(feat_path))
            feats = fd['mbo_features'].astype(np.float32)
            mid   = feats[:, COL_MID]
            all_features.append(feats)
            all_mids.append(mid)
            valid_dates.append(d)
        except Exception as e:
            logger.warning(f'  Skip {d}: {e}')

    n_days = len(valid_dates)
    logger.info(f'  Loaded {n_days} days')

    if n_days < train_days + 3:
        logger.error(f'  Need at least {train_days + 3} days, got {n_days}')
        return False

    generated = 0
    for test_idx in range(train_days, n_days):
        test_date = valid_dates[test_idx]
        out_path  = SIGNAL_DIR / f'meta_global_3s_{test_date}.npz'
        if out_path.exists():
            logger.info(f'  Skip {test_date} (already exists)')
            continue

        train_start = max(0, test_idx - train_days)

        # Build training set
        X_parts, y_parts = [], []
        for i in range(train_start, test_idx):
            feats = all_features[i]
            mid   = all_mids[i]
            n     = len(mid)
            # Forward return target
            fwd = np.full(n, np.nan, dtype=np.float32)
            fwd[:n - HORIZON_BARS] = mid[HORIZON_BARS:] - mid[:n - HORIZON_BARS]
            mask = np.isfinite(fwd) & np.all(np.isfinite(feats), axis=1)
            X_parts.append(feats[mask])
            y_parts.append(fwd[mask])

        X_train = np.vstack(X_parts)
        y_train = np.concatenate(y_parts)

        # Subsample if needed to avoid OOM
        MAX_SAMPLES = 300_000
        if len(y_train) > MAX_SAMPLES:
            rng = np.random.RandomState(test_idx)
            idx = rng.choice(len(y_train), MAX_SAMPLES, replace=False)
            X_train = X_train[idx]
            y_train = y_train[idx]

        # Clip and scale
        np.clip(X_train, -1e6, 1e6, out=X_train)
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X_train)

        # Train Ridge
        model = Ridge(alpha=1.0, solver='cholesky')
        model.fit(X_scaled, y_train)

        # Predict test day
        X_test = all_features[test_idx].copy()
        np.clip(X_test, -1e6, 1e6, out=X_test)
        X_test_scaled = scaler.transform(X_test)
        preds = model.predict(X_test_scaled).astype(np.float64)

        # Save
        np.savez_compressed(str(out_path),
                            predictions=preds,
                            mid_prices=all_mids[test_idx])
        generated += 1
        del model, X_train, y_train, X_scaled, X_test, X_test_scaled
        gc.collect()

        if generated % 10 == 0:
            logger.info(f'  Generated {generated} prediction files so far...')

    logger.info(f'  Done. Generated {generated} new prediction files.')
    return True


# ===========================================================================
# Fill Simulator
# ===========================================================================
def simulate_fill(
    mid_prices: np.ndarray,
    signal_bar: int,
    direction: int,
    day_end_bar: int,
) -> Tuple[int, float, str]:
    """
    Simulate limit order fill starting at signal_bar.

    Strategy:
      1. Post limit order at mid (bid side for long = mid - HALF_TICK).
         Actually we model passive fill as: fill at mid (no edge premium)
         if mid stays within TICK_SIZE of entry mid for LIMIT_FILL_BARS.
      2. If not filled after LIMIT_FILL_BARS, check again each bar until
         MARKET_FILL_BARS — if mid doesn't move away, assume queue fill.
      3. After MARKET_FILL_BARS unfilled → market order at mid - 0.5 tick penalty.

    Returns: (fill_bar, fill_price, fill_type)
             fill_bar = -1 means no fill before EOD
    """
    entry_mid = mid_prices[signal_bar]
    entry_threshold = TICK_SIZE  # if mid moves > 1 tick away, not filled

    # Try limit fill: check if mid stays within range
    limit_window_end = min(signal_bar + LIMIT_FILL_BARS, day_end_bar)
    for bar in range(signal_bar, limit_window_end):
        cur_mid = mid_prices[bar]
        if direction == 1:
            # Long limit: we want to buy at entry_mid - HALF_TICK
            # Fill if price drops to our level (mid dips down to entry)
            # Simplified: fill if mid stays close to entry (no extreme adverse move)
            if cur_mid <= entry_mid + entry_threshold:
                return bar, entry_mid, 'limit'
        else:
            # Short limit: sell at entry_mid + HALF_TICK
            if cur_mid >= entry_mid - entry_threshold:
                return bar, entry_mid, 'limit'

    # Try extended window up to MARKET_FILL_BARS
    market_window_end = min(signal_bar + MARKET_FILL_BARS, day_end_bar)
    for bar in range(limit_window_end, market_window_end):
        cur_mid = mid_prices[bar]
        move = abs(cur_mid - entry_mid) / TICK_SIZE
        if move <= 2.0:  # within 2 ticks — queue likely still has us
            return bar, entry_mid, 'limit'

    # Market order fallback
    market_bar = min(signal_bar + MARKET_FILL_BARS, day_end_bar - 1)
    if market_bar >= day_end_bar:
        return -1, 0.0, 'no_fill'

    market_price = mid_prices[market_bar]
    # Market penalty: we pay half-spread = HALF_TICK adverse
    if direction == 1:
        fill_price = market_price + HALF_TICK   # buy at ask
    else:
        fill_price = market_price - HALF_TICK   # sell at bid
    return market_bar, fill_price, 'market'


def simulate_exit(
    mid_prices: np.ndarray,
    entry_bar: int,
    direction: int,
    day_end_bar: int,
) -> Tuple[int, float, str]:
    """
    Simulate limit exit at mid. Same fill logic as entry but simpler:
    assume passive exit fills if mid stays near exit_mid.

    Returns: (exit_bar, exit_price, exit_type_fill)
    """
    exit_mid = mid_prices[entry_bar]

    # Try limit for LIMIT_FILL_BARS
    limit_end = min(entry_bar + LIMIT_FILL_BARS, day_end_bar)
    for bar in range(entry_bar, limit_end):
        cur_mid = mid_prices[bar]
        if abs(cur_mid - exit_mid) <= TICK_SIZE:
            return bar, exit_mid, 'limit'

    # Extended
    market_end = min(entry_bar + MARKET_FILL_BARS, day_end_bar)
    for bar in range(limit_end, market_end):
        cur_mid = mid_prices[bar]
        if abs(cur_mid - exit_mid) <= 2 * TICK_SIZE:
            return bar, exit_mid, 'limit'

    # Market exit
    market_bar = min(entry_bar + MARKET_FILL_BARS, day_end_bar - 1)
    market_price = mid_prices[market_bar]
    if direction == 1:
        fill_price = market_price - HALF_TICK   # sell at bid
    else:
        fill_price = market_price + HALF_TICK   # buy at ask
    return market_bar, fill_price, 'market'


def compute_fixed10s_pnl(
    mid_prices: np.ndarray,
    entry_bar: int,
    direction: int,
    day_end_bar: int,
) -> Tuple[float, float]:
    """Compute PnL if we had used fixed 10s hold from the same entry point."""
    exit_bar = min(entry_bar + FIXED_HOLD_BARS, day_end_bar - 1)
    entry_price = mid_prices[entry_bar]
    exit_price  = mid_prices[exit_bar]
    move_ticks  = (exit_price - entry_price) / TICK_SIZE * direction
    net_ticks   = move_ticks - COMMISSION_TICKS
    net_usd     = net_ticks * TICK_VALUE
    return net_ticks, net_usd


# ===========================================================================
# PnL Calculator
# ===========================================================================
def calc_pnl(
    entry_price: float,
    exit_price:  float,
    direction:   int,
    entry_type:  str,
    exit_type_fill: str,
) -> Tuple[float, float, float, float, float]:
    """
    Compute all PnL components.

    Returns: (dir_pnl_ticks, entry_edge, exit_edge, commission, net_ticks)
    """
    dir_pnl_ticks = (exit_price - entry_price) / TICK_SIZE * direction

    # Entry edge: +0.5 if limit (we saved half-spread), -0.5 if market (we paid)
    entry_edge = 0.5 if entry_type == 'limit' else -0.5

    # Exit edge: +0.5 if limit exit, -0.5 if market exit
    exit_edge = 0.5 if exit_type_fill == 'limit' else -0.5

    commission = COMMISSION_TICKS

    net_ticks = dir_pnl_ticks + entry_edge + exit_edge - commission

    return dir_pnl_ticks, entry_edge, exit_edge, commission, net_ticks


# ===========================================================================
# Strategy A: Hold-Until-Flip
# ===========================================================================
def run_strategy_flip(
    day_data: Dict,
    day_idx:  int,
    entry_threshold: float,
    min_spacing_bars: int = 50,
) -> List[ContinuousTrade]:
    """
    Hold-Until-Flip:
      - Enter long when prediction > +threshold
      - Hold as long as prediction >= 0 (same sign)
      - Exit when prediction flips to negative (or EOD/timeout)
      - Vice versa for short

    entry_threshold is an absolute value (not percentile) when pre-computed
    as a fraction of the empirical p80 of |predictions|.
    """
    predictions = day_data['predictions']
    mid_prices  = day_data['mid_prices']
    n_bars      = day_data['n_bars']
    trades      = []

    bar       = 0
    in_trade  = False
    last_exit = -min_spacing_bars

    while bar < n_bars:
        pred = predictions[bar]

        # --- Check for entry ---
        if not in_trade and (bar - last_exit) >= min_spacing_bars:
            if np.isfinite(pred) and abs(pred) >= entry_threshold:
                direction = 1 if pred > 0 else -1
                signal_bar = bar

                # Simulate fill
                fill_bar, fill_price, fill_type = simulate_fill(
                    mid_prices, signal_bar, direction, n_bars)

                if fill_bar == -1:
                    bar += 1
                    continue

                entry_bar   = fill_bar
                entry_price = fill_price
                entry_pred  = predictions[entry_bar]
                peak_signal = abs(pred)
                in_trade    = True

                # --- Look for exit (flip) ---
                exit_bar   = -1
                exit_pred  = 0.0
                exit_reason = 'timeout'

                for hold_bar in range(entry_bar + 1, min(entry_bar + MAX_HOLD_BARS + 1, n_bars)):
                    hold_pred = predictions[hold_bar]
                    if abs(hold_pred) > peak_signal:
                        peak_signal = abs(hold_pred)

                    # Exit condition: signal flips sign or becomes 0
                    if not np.isfinite(hold_pred) or hold_pred * direction <= 0:
                        exit_bar    = hold_bar
                        exit_pred   = hold_pred if np.isfinite(hold_pred) else 0.0
                        exit_reason = 'flip'
                        break

                # Timeout or EOD
                if exit_bar == -1:
                    exit_bar    = min(entry_bar + MAX_HOLD_BARS, n_bars - 1)
                    exit_pred   = predictions[exit_bar] if exit_bar < n_bars else 0.0
                    exit_reason = 'timeout' if exit_bar < n_bars - 1 else 'eod'

                # Simulate exit fill
                real_exit_bar, exit_price, exit_fill_type = simulate_exit(
                    mid_prices, exit_bar, direction, n_bars)

                # PnL
                dir_pnl, entry_edge, exit_edge, comm, net_ticks = calc_pnl(
                    entry_price, exit_price, direction, fill_type, exit_fill_type)

                # Comparison to fixed 10s
                fixed_ticks, fixed_usd = compute_fixed10s_pnl(
                    mid_prices, entry_bar, direction, n_bars)

                trades.append(ContinuousTrade(
                    day            = day_idx,
                    strategy       = 'flip',
                    direction      = direction,
                    signal_bar     = signal_bar,
                    entry_bar      = entry_bar,
                    exit_bar       = real_exit_bar,
                    entry_price    = entry_price,
                    exit_price     = exit_price,
                    signal_at_entry = float(entry_pred),
                    signal_at_exit  = float(exit_pred) if np.isfinite(exit_pred) else 0.0,
                    peak_signal    = peak_signal,
                    hold_bars      = real_exit_bar - entry_bar,
                    entry_type     = fill_type,
                    exit_type      = exit_reason,
                    dir_pnl_ticks  = dir_pnl,
                    entry_edge     = entry_edge,
                    exit_edge      = exit_edge,
                    commission     = comm,
                    net_ticks      = net_ticks,
                    net_dollars    = net_ticks * TICK_VALUE,
                    fixed10s_pnl_ticks = fixed_ticks,
                    fixed10s_pnl_usd   = fixed_usd,
                ))

                in_trade    = False
                last_exit   = real_exit_bar
                bar         = real_exit_bar + 1
                continue

        bar += 1

    return trades


# ===========================================================================
# Strategy B: Hold-Until-Weaken (Hysteresis)
# ===========================================================================
def run_strategy_weaken(
    day_data: Dict,
    day_idx:  int,
    entry_threshold:     float,
    exit_threshold_ratio: float = 0.5,
    min_spacing_bars:    int   = 50,
) -> List[ContinuousTrade]:
    """
    Hold-Until-Weaken:
      - Enter when |prediction| > entry_threshold
      - Hold as long as |prediction| > exit_threshold = entry_threshold * ratio
      - Hysteresis: exit_threshold < entry_threshold prevents whipsawing
    """
    predictions = day_data['predictions']
    mid_prices  = day_data['mid_prices']
    n_bars      = day_data['n_bars']
    trades      = []

    exit_threshold = entry_threshold * exit_threshold_ratio

    bar       = 0
    in_trade  = False
    last_exit = -min_spacing_bars

    while bar < n_bars:
        pred = predictions[bar]

        if not in_trade and (bar - last_exit) >= min_spacing_bars:
            if np.isfinite(pred) and abs(pred) >= entry_threshold:
                direction  = 1 if pred > 0 else -1
                signal_bar = bar

                fill_bar, fill_price, fill_type = simulate_fill(
                    mid_prices, signal_bar, direction, n_bars)

                if fill_bar == -1:
                    bar += 1
                    continue

                entry_bar   = fill_bar
                entry_price = fill_price
                entry_pred  = predictions[entry_bar]
                peak_signal = abs(pred)

                exit_bar    = -1
                exit_pred   = 0.0
                exit_reason = 'timeout'

                for hold_bar in range(entry_bar + 1, min(entry_bar + MAX_HOLD_BARS + 1, n_bars)):
                    hold_pred = predictions[hold_bar]
                    if abs(hold_pred) > peak_signal:
                        peak_signal = abs(hold_pred)

                    # Exit when signal weakens below exit threshold (in original direction)
                    if not np.isfinite(hold_pred) or abs(hold_pred) < exit_threshold:
                        exit_bar    = hold_bar
                        exit_pred   = hold_pred if np.isfinite(hold_pred) else 0.0
                        exit_reason = 'weaken'
                        break

                if exit_bar == -1:
                    exit_bar    = min(entry_bar + MAX_HOLD_BARS, n_bars - 1)
                    exit_pred   = predictions[exit_bar] if exit_bar < n_bars else 0.0
                    exit_reason = 'timeout' if exit_bar < n_bars - 1 else 'eod'

                real_exit_bar, exit_price, exit_fill_type = simulate_exit(
                    mid_prices, exit_bar, direction, n_bars)

                dir_pnl, entry_edge, exit_edge, comm, net_ticks = calc_pnl(
                    entry_price, exit_price, direction, fill_type, exit_fill_type)

                fixed_ticks, fixed_usd = compute_fixed10s_pnl(
                    mid_prices, entry_bar, direction, n_bars)

                trades.append(ContinuousTrade(
                    day            = day_idx,
                    strategy       = 'weaken',
                    direction      = direction,
                    signal_bar     = signal_bar,
                    entry_bar      = entry_bar,
                    exit_bar       = real_exit_bar,
                    entry_price    = entry_price,
                    exit_price     = exit_price,
                    signal_at_entry = float(entry_pred),
                    signal_at_exit  = float(exit_pred) if np.isfinite(exit_pred) else 0.0,
                    peak_signal    = peak_signal,
                    hold_bars      = real_exit_bar - entry_bar,
                    entry_type     = fill_type,
                    exit_type      = exit_reason,
                    dir_pnl_ticks  = dir_pnl,
                    entry_edge     = entry_edge,
                    exit_edge      = exit_edge,
                    commission     = comm,
                    net_ticks      = net_ticks,
                    net_dollars    = net_ticks * TICK_VALUE,
                    fixed10s_pnl_ticks = fixed_ticks,
                    fixed10s_pnl_usd   = fixed_usd,
                ))

                in_trade    = False
                last_exit   = real_exit_bar
                bar         = real_exit_bar + 1
                continue

        bar += 1

    return trades


# ===========================================================================
# Strategy C: Trailing Signal Stop
# ===========================================================================
def run_strategy_trailing(
    day_data: Dict,
    day_idx:  int,
    entry_threshold: float,
    decay_factor:    float = 0.5,
    min_spacing_bars: int  = 50,
) -> List[ContinuousTrade]:
    """
    Trailing Signal Stop:
      - Enter when prediction > threshold
      - Track peak |prediction| since entry
      - Exit when |prediction| drops to < peak * decay_factor
      - Like a trailing stop but on the signal, not price
    """
    predictions = day_data['predictions']
    mid_prices  = day_data['mid_prices']
    n_bars      = day_data['n_bars']
    trades      = []

    bar       = 0
    in_trade  = False
    last_exit = -min_spacing_bars

    while bar < n_bars:
        pred = predictions[bar]

        if not in_trade and (bar - last_exit) >= min_spacing_bars:
            if np.isfinite(pred) and abs(pred) >= entry_threshold:
                direction  = 1 if pred > 0 else -1
                signal_bar = bar

                fill_bar, fill_price, fill_type = simulate_fill(
                    mid_prices, signal_bar, direction, n_bars)

                if fill_bar == -1:
                    bar += 1
                    continue

                entry_bar   = fill_bar
                entry_price = fill_price
                entry_pred  = predictions[entry_bar]
                peak_signal = abs(entry_pred) if np.isfinite(entry_pred) else abs(pred)

                exit_bar    = -1
                exit_pred   = 0.0
                exit_reason = 'timeout'

                for hold_bar in range(entry_bar + 1, min(entry_bar + MAX_HOLD_BARS + 1, n_bars)):
                    hold_pred = predictions[hold_bar]
                    if not np.isfinite(hold_pred):
                        exit_bar    = hold_bar
                        exit_pred   = 0.0
                        exit_reason = 'trailing'
                        break

                    cur_abs = abs(hold_pred)
                    if cur_abs > peak_signal:
                        peak_signal = cur_abs   # new peak — raise the bar

                    # Trailing exit: signal dropped below peak * decay
                    trailing_stop = peak_signal * decay_factor
                    if cur_abs < trailing_stop:
                        exit_bar    = hold_bar
                        exit_pred   = hold_pred
                        exit_reason = 'trailing'
                        break

                if exit_bar == -1:
                    exit_bar    = min(entry_bar + MAX_HOLD_BARS, n_bars - 1)
                    exit_pred   = predictions[exit_bar] if exit_bar < n_bars else 0.0
                    exit_reason = 'timeout' if exit_bar < n_bars - 1 else 'eod'

                real_exit_bar, exit_price, exit_fill_type = simulate_exit(
                    mid_prices, exit_bar, direction, n_bars)

                dir_pnl, entry_edge, exit_edge, comm, net_ticks = calc_pnl(
                    entry_price, exit_price, direction, fill_type, exit_fill_type)

                fixed_ticks, fixed_usd = compute_fixed10s_pnl(
                    mid_prices, entry_bar, direction, n_bars)

                trades.append(ContinuousTrade(
                    day            = day_idx,
                    strategy       = 'trailing',
                    direction      = direction,
                    signal_bar     = signal_bar,
                    entry_bar      = entry_bar,
                    exit_bar       = real_exit_bar,
                    entry_price    = entry_price,
                    exit_price     = exit_price,
                    signal_at_entry = float(entry_pred) if np.isfinite(entry_pred) else 0.0,
                    signal_at_exit  = float(exit_pred)  if np.isfinite(exit_pred)  else 0.0,
                    peak_signal    = peak_signal,
                    hold_bars      = real_exit_bar - entry_bar,
                    entry_type     = fill_type,
                    exit_type      = exit_reason,
                    dir_pnl_ticks  = dir_pnl,
                    entry_edge     = entry_edge,
                    exit_edge      = exit_edge,
                    commission     = comm,
                    net_ticks      = net_ticks,
                    net_dollars    = net_ticks * TICK_VALUE,
                    fixed10s_pnl_ticks = fixed_ticks,
                    fixed10s_pnl_usd   = fixed_usd,
                ))

                in_trade    = False
                last_exit   = real_exit_bar
                bar         = real_exit_bar + 1
                continue

        bar += 1

    return trades


# ===========================================================================
# Strategy D: Regime-Continuous
# ===========================================================================
def run_strategy_regime(
    day_data: Dict,
    day_idx:  int,
    entry_threshold:  float,
    min_spacing_bars: int = 50,
) -> List[ContinuousTrade]:
    """
    Regime-Continuous:
      - HIGH vol (vol_regime > VOL_REGIME_HIGH_THRESH):
            hold until signal flips (let the big move run)
      - LOW vol  (vol_regime < VOL_REGIME_LOW_THRESH):
            fixed FIXED_HOLD_BARS exit (small moves = take quick profits)
      - MEDIUM vol:
            hold until signal weakens below 50% of entry threshold

    vol_regime read from mbo_features col 195.
    At entry bar, check vol_regime to determine hold mode.
    """
    predictions = day_data['predictions']
    mid_prices  = day_data['mid_prices']
    features    = day_data['features']
    n_bars      = day_data['n_bars']
    trades      = []

    exit_threshold_medium = entry_threshold * 0.5

    bar       = 0
    in_trade  = False
    last_exit = -min_spacing_bars

    while bar < n_bars:
        pred = predictions[bar]

        if not in_trade and (bar - last_exit) >= min_spacing_bars:
            if np.isfinite(pred) and abs(pred) >= entry_threshold:
                direction  = 1 if pred > 0 else -1
                signal_bar = bar

                fill_bar, fill_price, fill_type = simulate_fill(
                    mid_prices, signal_bar, direction, n_bars)

                if fill_bar == -1:
                    bar += 1
                    continue

                entry_bar   = fill_bar
                entry_price = fill_price
                entry_pred  = predictions[entry_bar]

                # Determine regime at entry
                vol_regime_val = features[entry_bar, COL_VOL_REGIME]
                if np.isfinite(vol_regime_val):
                    if vol_regime_val > VOL_REGIME_HIGH_THRESH:
                        hold_mode = 'high_vol'
                    elif vol_regime_val < VOL_REGIME_LOW_THRESH:
                        hold_mode = 'low_vol'
                    else:
                        hold_mode = 'medium_vol'
                else:
                    hold_mode = 'medium_vol'  # default if unknown

                peak_signal = abs(entry_pred) if np.isfinite(entry_pred) else abs(pred)

                exit_bar    = -1
                exit_pred   = 0.0
                exit_reason = 'timeout'

                max_hold = MAX_HOLD_BARS

                if hold_mode == 'low_vol':
                    # Fixed hold — exit after FIXED_HOLD_BARS
                    exit_bar    = min(entry_bar + FIXED_HOLD_BARS, n_bars - 1)
                    exit_pred   = predictions[exit_bar] if exit_bar < n_bars else 0.0
                    exit_reason = f'regime_low_vol_fixed{FIXED_HOLD_BARS // BARS_PER_SEC}s'
                else:
                    for hold_bar in range(entry_bar + 1, min(entry_bar + max_hold + 1, n_bars)):
                        hold_pred = predictions[hold_bar]
                        if abs(hold_pred) > peak_signal:
                            peak_signal = abs(hold_pred)

                        if hold_mode == 'high_vol':
                            # Exit on flip
                            if not np.isfinite(hold_pred) or hold_pred * direction <= 0:
                                exit_bar    = hold_bar
                                exit_pred   = hold_pred if np.isfinite(hold_pred) else 0.0
                                exit_reason = 'regime_high_vol_flip'
                                break
                        else:
                            # Medium vol: exit when weaken below 50%
                            if not np.isfinite(hold_pred) or abs(hold_pred) < exit_threshold_medium:
                                exit_bar    = hold_bar
                                exit_pred   = hold_pred if np.isfinite(hold_pred) else 0.0
                                exit_reason = 'regime_medium_vol_weaken'
                                break

                if exit_bar == -1:
                    exit_bar    = min(entry_bar + max_hold, n_bars - 1)
                    exit_pred   = predictions[exit_bar] if exit_bar < n_bars else 0.0
                    exit_reason = 'timeout' if exit_bar < n_bars - 1 else 'eod'

                real_exit_bar, exit_price, exit_fill_type = simulate_exit(
                    mid_prices, exit_bar, direction, n_bars)

                dir_pnl, entry_edge, exit_edge, comm, net_ticks = calc_pnl(
                    entry_price, exit_price, direction, fill_type, exit_fill_type)

                fixed_ticks, fixed_usd = compute_fixed10s_pnl(
                    mid_prices, entry_bar, direction, n_bars)

                trades.append(ContinuousTrade(
                    day            = day_idx,
                    strategy       = f'regime_{hold_mode}',
                    direction      = direction,
                    signal_bar     = signal_bar,
                    entry_bar      = entry_bar,
                    exit_bar       = real_exit_bar,
                    entry_price    = entry_price,
                    exit_price     = exit_price,
                    signal_at_entry = float(entry_pred) if np.isfinite(entry_pred) else 0.0,
                    signal_at_exit  = float(exit_pred)  if np.isfinite(exit_pred)  else 0.0,
                    peak_signal    = peak_signal,
                    hold_bars      = real_exit_bar - entry_bar,
                    entry_type     = fill_type,
                    exit_type      = exit_reason,
                    dir_pnl_ticks  = dir_pnl,
                    entry_edge     = entry_edge,
                    exit_edge      = exit_edge,
                    commission     = comm,
                    net_ticks      = net_ticks,
                    net_dollars    = net_ticks * TICK_VALUE,
                    fixed10s_pnl_ticks = fixed_ticks,
                    fixed10s_pnl_usd   = fixed_usd,
                ))

                in_trade    = False
                last_exit   = real_exit_bar
                bar         = real_exit_bar + 1
                continue

        bar += 1

    return trades


# ===========================================================================
# Metrics Calculator
# ===========================================================================
def compute_metrics(
    trades:     List[ContinuousTrade],
    n_days:     int,
    combo_name: str,
) -> Dict:
    """Compute comprehensive performance metrics for a set of trades."""
    if not trades:
        return {
            'combo':       combo_name,
            'n_trades':    0,
            'n_days':      n_days,
            'total_pnl_usd': 0.0,
            'avg_pnl_usd':   0.0,
            'pnl_per_day':   0.0,
        }

    pnls        = np.array([t.net_dollars for t in trades])
    ticks       = np.array([t.net_ticks   for t in trades])
    hold_bars   = np.array([t.hold_bars   for t in trades])
    fixed_pnls  = np.array([t.fixed10s_pnl_usd for t in trades])

    n_trades    = len(trades)
    total_pnl   = float(np.sum(pnls))
    avg_pnl     = float(np.mean(pnls))
    win_rate    = float((pnls > 0).mean())
    loses       = pnls[pnls < 0]
    wins        = pnls[pnls > 0]
    profit_factor = (float(wins.sum()) / max(abs(float(loses.sum())), 1e-8)
                     if len(wins) > 0 else 0.0)

    avg_hold   = float(np.mean(hold_bars))
    max_hold   = int(np.max(hold_bars))

    # PnL per bar held (efficiency: how much $ earned per 100ms held)
    pnl_per_bar = float(np.sum(pnls) / max(np.sum(hold_bars), 1))

    # Daily PnL for Sharpe
    daily_pnl = {}
    for t in trades:
        d = t.day
        daily_pnl[d] = daily_pnl.get(d, 0.0) + t.net_dollars
    daily_arr = np.array(list(daily_pnl.values()))
    sharpe = (float(daily_arr.mean()) / max(float(daily_arr.std()), 1e-8)
              * np.sqrt(252) if len(daily_arr) > 1 else 0.0)

    trades_per_day = n_trades / n_days

    # Comparison to fixed 10s
    fixed_total = float(np.sum(fixed_pnls))
    improvement = total_pnl - fixed_total  # positive = continuous is better

    # Exit type breakdown
    exit_types = {}
    for t in trades:
        et = t.exit_type
        exit_types[et] = exit_types.get(et, 0) + 1

    # Fill type breakdown
    entry_market = sum(1 for t in trades if t.entry_type == 'market')
    exit_market  = sum(1 for t in trades if t.exit_type in ('market', 'timeout'))

    return {
        'combo':            combo_name,
        'n_trades':         n_trades,
        'n_days':           n_days,
        'trades_per_day':   round(trades_per_day, 2),
        'total_pnl_usd':    round(total_pnl, 2),
        'avg_pnl_usd':      round(avg_pnl, 2),
        'pnl_per_day':      round(total_pnl / n_days, 2),
        'win_rate':         round(win_rate, 4),
        'profit_factor':    round(profit_factor, 3),
        'avg_hold_bars':    round(avg_hold, 1),
        'avg_hold_secs':    round(avg_hold / BARS_PER_SEC, 1),
        'max_hold_bars':    max_hold,
        'pnl_per_bar':      round(pnl_per_bar, 6),
        'pnl_per_bar_usd':  round(pnl_per_bar, 4),
        'sharpe_annualized': round(sharpe, 3),
        'total_ticks':      round(float(np.sum(ticks)), 3),
        'avg_ticks':        round(float(np.mean(ticks)), 4),
        # Comparison to fixed 10s
        'fixed10s_total_pnl': round(fixed_total, 2),
        'vs_fixed10s_improvement': round(improvement, 2),
        'vs_fixed10s_pct_improvement': round(improvement / max(abs(fixed_total), 1) * 100, 1),
        # Fill info
        'entry_market_pct': round(entry_market / n_trades * 100, 1),
        'exit_types':       exit_types,
    }


# ===========================================================================
# Threshold Calculator
# ===========================================================================
def compute_entry_thresholds(
    all_preds:   np.ndarray,
    percentiles: List[float],
) -> Dict[float, float]:
    """Convert percentile levels to absolute threshold values."""
    abs_preds = np.abs(all_preds[np.isfinite(all_preds) & (all_preds != 0)])
    if len(abs_preds) == 0:
        return {p: 0.01 for p in percentiles}
    thresholds = {}
    for pct in percentiles:
        # pct=0.1 means "top 10%" = 90th percentile of |predictions|
        thresh = np.percentile(abs_preds, (1.0 - pct) * 100)
        thresholds[pct] = float(thresh)
    return thresholds


# ===========================================================================
# Main Sweep Runner
# ===========================================================================
def run_signal_sweep(
    signal_prefix: str,
    strategies:    List[str],
    n_days:        Optional[int],
    quick_mode:    bool,
) -> List[Dict]:
    """Run complete parameter sweep for one signal."""

    dates = discover_dates(signal_prefix)
    if not dates:
        logger.warning(f'No data found for signal prefix: {signal_prefix}')
        return []

    if n_days is not None:
        dates = dates[:n_days]

    logger.info(f'Signal {signal_prefix}: {len(dates)} days '
                f'({dates[0]} .. {dates[-1]})')

    # Parameter grids
    if quick_mode:
        entry_pcts           = [0.2]          # top 20%
        exit_ratios          = [0.5]
        decay_factors        = [0.5]
    else:
        entry_pcts           = [0.1, 0.2, 0.3, 0.5]
        exit_ratios          = [0.3, 0.5, 0.7]
        decay_factors        = [0.3, 0.5, 0.7]

    # Collect all predictions to compute thresholds
    logger.info('  Computing global thresholds from all prediction files...')
    all_preds_list = []
    all_data       = []

    for di, date in enumerate(dates):
        day_data = load_day_data(date, signal_prefix)
        if day_data is None:
            logger.warning(f'  Skip {date}: could not load')
            continue
        all_preds_list.append(day_data['predictions'])
        all_data.append((di, day_data))

    if not all_data:
        logger.warning(f'  No valid days loaded for {signal_prefix}')
        return []

    all_preds_concat = np.concatenate(all_preds_list)
    thresholds = compute_entry_thresholds(all_preds_concat, entry_pcts)

    n_loaded = len(all_data)
    logger.info(f'  Loaded {n_loaded} days. Thresholds:')
    for pct, thresh in thresholds.items():
        logger.info(f'    entry_pct={pct:.1f} (top {pct*100:.0f}%): '
                    f'|pred| >= {thresh:.6f}')

    all_results = []

    # Strategy A: Flip
    if 'flip' in strategies:
        logger.info(f'  Strategy A: Hold-Until-Flip')
        for entry_pct, entry_thresh in thresholds.items():
            all_trades = []
            for di, day_data in all_data:
                trades = run_strategy_flip(
                    day_data, di, entry_thresh)
                all_trades.extend(trades)

            combo_name = f'{signal_prefix}__flip__entry_pct={entry_pct}'
            metrics = compute_metrics(all_trades, n_loaded, combo_name)
            metrics['strategy']    = 'flip'
            metrics['signal']      = signal_prefix
            metrics['entry_pct']   = entry_pct
            metrics['entry_thresh'] = round(entry_thresh, 6)
            all_results.append(metrics)

            logger.info(f'    entry_pct={entry_pct}: '
                        f'{metrics["n_trades"]} trades, '
                        f'PnL=${metrics["total_pnl_usd"]:.0f} '
                        f'(${metrics["pnl_per_day"]:.0f}/day), '
                        f'Sharpe={metrics["sharpe_annualized"]:.2f}, '
                        f'avg_hold={metrics["avg_hold_secs"]:.1f}s, '
                        f'vs_fixed10s={metrics["vs_fixed10s_improvement"]:+.0f}')

    # Strategy B: Weaken (hysteresis)
    if 'weaken' in strategies:
        logger.info(f'  Strategy B: Hold-Until-Weaken')
        for entry_pct, entry_thresh in thresholds.items():
            for ratio in exit_ratios:
                all_trades = []
                for di, day_data in all_data:
                    trades = run_strategy_weaken(
                        day_data, di, entry_thresh, ratio)
                    all_trades.extend(trades)

                combo_name = (f'{signal_prefix}__weaken__entry_pct={entry_pct}'
                              f'__exit_ratio={ratio}')
                metrics = compute_metrics(all_trades, n_loaded, combo_name)
                metrics['strategy']         = 'weaken'
                metrics['signal']           = signal_prefix
                metrics['entry_pct']        = entry_pct
                metrics['entry_thresh']     = round(entry_thresh, 6)
                metrics['exit_ratio']       = ratio
                all_results.append(metrics)

                logger.info(f'    entry_pct={entry_pct}, ratio={ratio}: '
                            f'{metrics["n_trades"]} trades, '
                            f'PnL=${metrics["total_pnl_usd"]:.0f} '
                            f'(${metrics["pnl_per_day"]:.0f}/day), '
                            f'Sharpe={metrics["sharpe_annualized"]:.2f}, '
                            f'avg_hold={metrics["avg_hold_secs"]:.1f}s, '
                            f'vs_fixed10s={metrics["vs_fixed10s_improvement"]:+.0f}')

    # Strategy C: Trailing Signal Stop
    if 'trailing' in strategies:
        logger.info(f'  Strategy C: Trailing Signal Stop')
        for entry_pct, entry_thresh in thresholds.items():
            for decay in decay_factors:
                all_trades = []
                for di, day_data in all_data:
                    trades = run_strategy_trailing(
                        day_data, di, entry_thresh, decay)
                    all_trades.extend(trades)

                combo_name = (f'{signal_prefix}__trailing__entry_pct={entry_pct}'
                              f'__decay={decay}')
                metrics = compute_metrics(all_trades, n_loaded, combo_name)
                metrics['strategy']     = 'trailing'
                metrics['signal']       = signal_prefix
                metrics['entry_pct']    = entry_pct
                metrics['entry_thresh'] = round(entry_thresh, 6)
                metrics['decay_factor'] = decay
                all_results.append(metrics)

                logger.info(f'    entry_pct={entry_pct}, decay={decay}: '
                            f'{metrics["n_trades"]} trades, '
                            f'PnL=${metrics["total_pnl_usd"]:.0f} '
                            f'(${metrics["pnl_per_day"]:.0f}/day), '
                            f'Sharpe={metrics["sharpe_annualized"]:.2f}, '
                            f'avg_hold={metrics["avg_hold_secs"]:.1f}s, '
                            f'vs_fixed10s={metrics["vs_fixed10s_improvement"]:+.0f}')

    # Strategy D: Regime-Continuous
    if 'regime' in strategies:
        logger.info(f'  Strategy D: Regime-Continuous')
        for entry_pct, entry_thresh in thresholds.items():
            all_trades = []
            for di, day_data in all_data:
                trades = run_strategy_regime(
                    day_data, di, entry_thresh)
                all_trades.extend(trades)

            combo_name = f'{signal_prefix}__regime__entry_pct={entry_pct}'
            metrics = compute_metrics(all_trades, n_loaded, combo_name)
            metrics['strategy']     = 'regime'
            metrics['signal']       = signal_prefix
            metrics['entry_pct']    = entry_pct
            metrics['entry_thresh'] = round(entry_thresh, 6)
            all_results.append(metrics)

            logger.info(f'    entry_pct={entry_pct}: '
                        f'{metrics["n_trades"]} trades, '
                        f'PnL=${metrics["total_pnl_usd"]:.0f} '
                        f'(${metrics["pnl_per_day"]:.0f}/day), '
                        f'Sharpe={metrics["sharpe_annualized"]:.2f}, '
                        f'avg_hold={metrics["avg_hold_secs"]:.1f}s, '
                        f'vs_fixed10s={metrics["vs_fixed10s_improvement"]:+.0f}')

    return all_results


# ===========================================================================
# Summary Printer
# ===========================================================================
def print_summary(all_results: List[Dict]) -> None:
    """Print a formatted summary table sorted by Sharpe descending."""
    if not all_results:
        logger.info('No results to summarize.')
        return

    # Filter to combos with at least 10 trades
    valid = [r for r in all_results if r['n_trades'] >= 10]
    if not valid:
        valid = all_results

    # Sort by Sharpe descending
    valid.sort(key=lambda x: x.get('sharpe_annualized', -999), reverse=True)

    logger.info('')
    logger.info('=' * 120)
    logger.info('CONTINUOUS SIGNAL SIM — TOP RESULTS (by Sharpe, n_trades >= 10)')
    logger.info('=' * 120)
    logger.info(f'{"Strategy":<18} {"Signal":<20} {"EntPct":>7} {"ExitP":>7} '
                f'{"Trades":>7} {"TotPnL":>9} {"$/Day":>8} '
                f'{"WinR":>6} {"PF":>6} '
                f'{"AvgHold":>8} {"Sharpe":>7} '
                f'{"vsFixed10s":>11}')
    logger.info('-' * 120)

    for r in valid[:30]:   # top 30
        strat  = r.get('strategy', '?')
        sig    = r.get('signal', '?')[:20]
        ep     = r.get('entry_pct', 0)
        xp     = r.get('exit_ratio', r.get('decay_factor', '-'))
        nt     = r.get('n_trades', 0)
        pnl    = r.get('total_pnl_usd', 0)
        ppd    = r.get('pnl_per_day', 0)
        wr     = r.get('win_rate', 0) * 100
        pf     = r.get('profit_factor', 0)
        hold   = r.get('avg_hold_secs', 0)
        sharpe = r.get('sharpe_annualized', 0)
        vs_fix = r.get('vs_fixed10s_improvement', 0)

        logger.info(f'{strat:<18} {sig:<20} {ep:>7.1%} {str(xp):>7} '
                    f'{nt:>7} {pnl:>+9,.0f} {ppd:>+8,.0f} '
                    f'{wr:>5.1f}% {pf:>6.2f} '
                    f'{hold:>7.1f}s {sharpe:>7.2f} '
                    f'{vs_fix:>+11,.0f}')

    logger.info('=' * 120)

    # Highlight: best by each strategy
    logger.info('')
    logger.info('BEST PER STRATEGY (by PnL/day):')
    strat_groups = {}
    for r in valid:
        s = r.get('strategy', '?')[:8]
        if s not in strat_groups or r['pnl_per_day'] > strat_groups[s]['pnl_per_day']:
            strat_groups[s] = r
    for s, r in sorted(strat_groups.items()):
        logger.info(f'  {s:<12}: ${r["pnl_per_day"]:+.0f}/day | '
                    f'Sharpe={r["sharpe_annualized"]:.2f} | '
                    f'{r["n_trades"]} trades | '
                    f'avg_hold={r["avg_hold_secs"]:.1f}s | '
                    f'vs_fixed10s={r["vs_fixed10s_improvement"]:+.0f} | '
                    f'{r["combo"]}')

    # Compare continuous vs fixed overall
    cont_pnl  = sum(r['total_pnl_usd']   for r in valid)
    fixed_pnl = sum(r['fixed10s_total_pnl'] for r in valid)
    logger.info('')
    logger.info(f'OVERALL (sum across all combos):')
    logger.info(f'  Continuous hold total: ${cont_pnl:+,.0f}')
    logger.info(f'  Fixed 10s total:       ${fixed_pnl:+,.0f}')
    logger.info(f'  Continuous advantage:  ${cont_pnl - fixed_pnl:+,.0f}')


# ===========================================================================
# Main Entry Point
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(
        description='Continuous Signal Hold Simulation for ES futures.'
    )
    parser.add_argument(
        '--n-days', type=int, default=None,
        help='Number of days to process (default: all available)'
    )
    parser.add_argument(
        '--strategy', type=str, default='all',
        choices=['all', 'flip', 'weaken', 'trailing', 'regime'],
        help='Which hold strategy to run (default: all)'
    )
    parser.add_argument(
        '--signal', type=str, default='all',
        choices=['all', 'meta_3s', 'cond_strict', 'percentile'],
        help='Which signal to use (default: all)'
    )
    parser.add_argument(
        '--quick', action='store_true',
        help='Quick mode: fewer parameter combinations'
    )
    parser.add_argument(
        '--generate', action='store_true',
        help='Generate fresh Ridge walk-forward predictions for meta_global_3s signal'
    )
    args = parser.parse_args()

    logger.info('=' * 70)
    logger.info('CONTINUOUS SIGNAL TRADING SIMULATION')
    logger.info(f'  strategy={args.strategy}, signal={args.signal}, '
                f'n_days={args.n_days}, quick={args.quick}')
    logger.info(f'  MBO features: {MBO_FEAT_DIR}')
    logger.info(f'  Signals:      {SIGNAL_DIR}')
    logger.info(f'  Results:      {RESULTS_DIR}')
    logger.info(f'  ES constants: TICK=${TICK_VALUE}, RT_COST=${COMMISSION_RT}, '
                f'BARS_PER_SEC={BARS_PER_SEC}')
    logger.info(f'  Fill model:   limit fill in {LIMIT_FILL_BARS} bars, '
                f'market after {MARKET_FILL_BARS} bars, '
                f'force exit at {MAX_HOLD_BARS} bars '
                f'({MAX_HOLD_BARS // BARS_PER_SEC}s)')
    logger.info('=' * 70)

    # Optionally generate fresh Ridge predictions first
    if args.generate:
        logger.info('GENERATE MODE: Creating fresh Ridge walk-forward predictions...')
        feat_files = sorted(MBO_FEAT_DIR.glob('*_mbo_features.npz'))
        all_dates  = [f.stem.replace('_mbo_features', '') for f in feat_files]
        if args.n_days:
            all_dates = all_dates[:args.n_days + 20]  # extra for training
        ok = generate_ridge_predictions(all_dates, train_days=20)
        if not ok:
            logger.error('Failed to generate predictions. Exiting.')
            sys.exit(1)

    # Determine which signals to run
    SIGNAL_MAP = {
        'meta_3s':    'meta_global_3s',
        'cond_strict': 'cond_strict',
        'percentile':  'percentile',
    }

    if args.signal == 'all':
        signals_to_run = list(SIGNAL_MAP.values())
    else:
        signals_to_run = [SIGNAL_MAP[args.signal]]

    # Determine which strategies to run
    if args.strategy == 'all':
        strategies_to_run = ['flip', 'weaken', 'trailing', 'regime']
    else:
        strategies_to_run = [args.strategy]

    # Run sweeps
    all_results = []
    for sig_prefix in signals_to_run:
        logger.info('')
        logger.info(f'{"=" * 70}')
        logger.info(f'SIGNAL: {sig_prefix}')
        logger.info(f'{"=" * 70}')

        results = run_signal_sweep(
            signal_prefix=sig_prefix,
            strategies=strategies_to_run,
            n_days=args.n_days,
            quick_mode=args.quick,
        )
        all_results.extend(results)

        if results:
            # Per-signal best
            best = max(results, key=lambda r: r.get('pnl_per_day', -1e9))
            logger.info(f'  Best for {sig_prefix}: '
                        f'{best["strategy"]}'
                        f'(entry_pct={best["entry_pct"]}) -> '
                        f'${best["pnl_per_day"]:.0f}/day, '
                        f'Sharpe={best["sharpe_annualized"]:.2f}')

    # Print summary
    print_summary(all_results)

    # Save results
    ts = datetime.now().strftime('%Y%m%d_%H%M%S')
    out_path = RESULTS_DIR / f'continuous_signal_{ts}.json'

    output = {
        'experiment':  'continuous_signal_sim',
        'timestamp':   datetime.now().isoformat(),
        'config': {
            'strategy':      args.strategy,
            'signal':        args.signal,
            'n_days':        args.n_days,
            'quick':         args.quick,
            'tick_size':     TICK_SIZE,
            'tick_value':    TICK_VALUE,
            'commission_rt': COMMISSION_RT,
            'bars_per_sec':  BARS_PER_SEC,
            'limit_fill_bars':  LIMIT_FILL_BARS,
            'market_fill_bars': MARKET_FILL_BARS,
            'max_hold_bars':    MAX_HOLD_BARS,
            'fixed_hold_bars':  FIXED_HOLD_BARS,
            'vol_regime_low_thresh':  VOL_REGIME_LOW_THRESH,
            'vol_regime_high_thresh': VOL_REGIME_HIGH_THRESH,
        },
        'n_combos':  len(all_results),
        'results':   all_results,
    }

    # Sort output by sharpe
    output['results'].sort(key=lambda x: x.get('sharpe_annualized', -999), reverse=True)

    with open(out_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)

    logger.info('')
    logger.info(f'Results saved: {out_path}')
    logger.info(f'Log file:      {_log_file}')
    logger.info('Done.')


if __name__ == '__main__':
    main()
