"""
Exit Optimization -- Take-Profit / Stop-Loss Parameter Search for ES Futures

Given predictions from a 100-day MFE walk-forward run, this script optimizes
exit parameters (TP, SL, horizon) for market-order entries on ultra-high
conviction bars (magnitude_preds >= 3.0t AND direction signal in top 10%).

Baseline performance (fixed 2.0t target, 100-bar horizon, no stop):
    OOS PF=1.22, WR=61.7%, +$1.07/trade

Analyses:
  1. Grid search over TP x SL x Horizon on IS (days 0-69)
  2. Evaluate top IS combos on OOS (days 70-99) to detect overfitting
  3. Adaptive exits: TP/SL scaled by predicted magnitude
  4. Time-based scaling: tighter stops as horizon exhausts

NPZ format (same as edge_analysis.py):
    mid_prices        float32[N]   mid-price series (all days concatenated)
    direction_preds   float32[N]   direction model prediction (NaN where no pred)
    magnitude_preds   float32[N]   magnitude model prediction in ticks (NaN where no pred)
    direction_target  float32[N]   direction target
    magnitude_target  float32[N]   magnitude target
    day_boundaries    int32[D+1]   start/end indices for each day

Each bar is 100ms. ~234,000 bars/day, 100 days total.

Usage:
    python alpha_discovery/exit_optimization.py \\
        --load-predictions results/predictions_mfe_path_TIMESTAMP.npz \\
        --oos-split-day 70

    # Custom grid:
    python alpha_discovery/exit_optimization.py \\
        --load-predictions /path/to/predictions.npz \\
        --oos-split-day 70 --top-n 30
"""

import sys
import time
import json
import logging
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
LVL3_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(LVL3_ROOT))

RESULTS_DIR = LVL3_ROOT / "alpha_discovery" / "results"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = RESULTS_DIR / f"exit_optimization_{_ts}.log"
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
logger = logging.getLogger('exit_opt')

# ---------------------------------------------------------------------------
# Constants (ES futures, 100ms bars)
# ---------------------------------------------------------------------------
TICK_SIZE       = 0.25          # ES tick size in points
TICK_VALUE      = 12.50         # $ per tick (ES full contract)
BARS_PER_SEC    = 10            # 100ms resolution
BARS_PER_DAY    = 234_000       # 6.5h * 3600s/h * 10 bars/s

# Round-trip cost for market entry + market exit:
# Entry spread (0.5t) + exit spread (0.5t) + commission RT (0.248t) = 1.248t
# Commission: $3.10 RT / $12.50 per tick = 0.248 ticks
# NOTE: No additional slippage for 1-lot ES (deep liquidity).
MKT_ENTRY_COST_TICKS = 1.248

# Gate thresholds (ultra-high conviction)
MAG_THRESHOLD   = 3.0           # magnitude prediction >= 3.0 ticks
DIR_QUANTILE    = 0.90          # top 10% direction signal

# Grid parameters
TP_GRID         = [1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0]
SL_GRID         = [0.5, 1.0, 1.5, 2.0, 3.0, None]      # None = no stop
HORIZON_GRID    = [50, 100, 200, 500]                     # bars = 5s, 10s, 20s, 50s

# Adaptive exit multipliers
ADAPTIVE_TP_MULT = 0.6          # TP = mag_pred * 0.6
ADAPTIVE_SL_MULT = 0.3          # SL = mag_pred * 0.3


# ===========================================================================
# Data loading (mirrors edge_analysis.py)
# ===========================================================================

def load_predictions(path: str, n_days: Optional[int] = None) -> Dict:
    """Load predictions NPZ. Returns dict with arrays and day_boundaries."""
    logger.info(f"Loading predictions from: {path}")
    data = np.load(path)

    mid_prices       = data['mid_prices'].astype(np.float32)
    direction_preds  = data['direction_preds'].astype(np.float32)
    magnitude_preds  = data['magnitude_preds'].astype(np.float32)
    direction_target = data['direction_target'].astype(np.float32)
    magnitude_target = data['magnitude_target'].astype(np.float32)
    day_boundaries   = data['day_boundaries'].tolist()

    total_days = len(day_boundaries) - 1
    logger.info(f"  Raw: {len(mid_prices):,} bars, {total_days} days")
    logger.info(f"  Direction preds valid: {np.isfinite(direction_preds).sum():,}")
    logger.info(f"  Magnitude preds valid: {np.isfinite(magnitude_preds).sum():,}")

    if n_days and total_days > n_days:
        cut = day_boundaries[n_days]
        mid_prices       = mid_prices[:cut]
        direction_preds  = direction_preds[:cut]
        magnitude_preds  = magnitude_preds[:cut]
        direction_target = direction_target[:cut]
        magnitude_target = magnitude_target[:cut]
        day_boundaries   = day_boundaries[:n_days + 1]
        logger.info(f"  Trimmed to {n_days} days ({len(mid_prices):,} bars)")

    return {
        'mid_prices':       mid_prices,
        'direction_preds':  direction_preds,
        'magnitude_preds':  magnitude_preds,
        'direction_target': direction_target,
        'magnitude_target': magnitude_target,
        'day_boundaries':   day_boundaries,
        'n_days':           len(day_boundaries) - 1,
    }


def _slice_data(data: Dict, start_day: int, end_day: int) -> Dict:
    """Return a sub-dict sliced to [start_day, end_day) with re-zeroed boundaries."""
    db = data['day_boundaries']
    s  = db[start_day]
    e  = db[end_day]
    new_bounds = [b - s for b in db[start_day: end_day + 1]]
    return {
        'mid_prices':       data['mid_prices'][s:e],
        'direction_preds':  data['direction_preds'][s:e],
        'magnitude_preds':  data['magnitude_preds'][s:e],
        'direction_target': data['direction_target'][s:e],
        'magnitude_target': data['magnitude_target'][s:e],
        'day_boundaries':   new_bounds,
        'n_days':           end_day - start_day,
    }


# ===========================================================================
# Gate filtering: find eligible bars
# ===========================================================================

def find_eligible_bars(data: Dict, dir_threshold: Optional[float] = None) -> Dict:
    """
    Find bars passing the ultra-high conviction gate:
      - magnitude_preds >= MAG_THRESHOLD (3.0 ticks)
      - |direction_preds| in top DIR_QUANTILE (top 10%)
      - Both preds finite

    If dir_threshold is provided (frozen from IS), uses that threshold instead
    of recomputing from the current slice. This prevents look-ahead bias on OOS.

    Returns dict with arrays of eligible bar info.
    """
    direction_preds = data['direction_preds']
    magnitude_preds = data['magnitude_preds']

    # Valid prediction mask
    valid = np.isfinite(direction_preds) & np.isfinite(magnitude_preds)
    n_valid = int(valid.sum())
    logger.info(f"  Valid prediction bars: {n_valid:,}")

    if n_valid < 100:
        logger.warning("  Too few valid bars for gating")
        return {'indices': np.array([], dtype=np.int64), 'n_eligible': 0}

    # Direction threshold: use frozen value if provided, else compute from this slice
    if dir_threshold is not None:
        logger.info(f"  Using FROZEN direction threshold from IS: {dir_threshold:.6f}")
    else:
        abs_dir = np.abs(direction_preds[valid])
        dir_threshold = float(np.percentile(abs_dir, DIR_QUANTILE * 100))
        logger.info(f"  Direction threshold (P{DIR_QUANTILE*100:.0f}): {dir_threshold:.6f}")

    # Magnitude threshold
    logger.info(f"  Magnitude threshold: >= {MAG_THRESHOLD:.1f} ticks")

    # Combined gate
    gate = (valid &
            (magnitude_preds >= MAG_THRESHOLD) &
            (np.abs(direction_preds) >= dir_threshold))

    indices = np.where(gate)[0]
    n_eligible = len(indices)
    logger.info(f"  Eligible bars (mag>={MAG_THRESHOLD}t AND dir top {(1-DIR_QUANTILE)*100:.0f}%): "
                f"{n_eligible:,}")

    return {
        'indices':        indices,
        'n_eligible':     n_eligible,
        'dir_threshold':  dir_threshold,
    }


# ===========================================================================
# Minimum spacing filter (prevent overlapping trades)
# ===========================================================================

def apply_min_spacing(eligible_indices: np.ndarray, day_boundaries: list,
                      min_spacing: int) -> np.ndarray:
    """
    Filter eligible bar indices to enforce minimum spacing between consecutive
    trades within each day. This prevents overlapping forward paths from being
    counted as independent trades.

    min_spacing should be >= max_horizon to ensure fully non-overlapping paths.
    """
    if len(eligible_indices) == 0:
        return eligible_indices

    n_days = len(day_boundaries) - 1

    # Build day-end lookup for each bar
    bar_to_day_end = {}
    bar_to_day_start = {}
    for d in range(n_days):
        s = day_boundaries[d]
        e = day_boundaries[d + 1]
        for bi in eligible_indices:
            if s <= bi < e:
                bar_to_day_end[bi] = e
                bar_to_day_start[bi] = s

    # Apply spacing per day
    filtered = []
    last_entry_per_day = {}

    for bi in eligible_indices:
        ds = bar_to_day_start.get(bi, -1)
        if ds not in last_entry_per_day:
            last_entry_per_day[ds] = -min_spacing

        if bi >= last_entry_per_day[ds] + min_spacing:
            filtered.append(bi)
            last_entry_per_day[ds] = bi

    result = np.array(filtered, dtype=eligible_indices.dtype)
    logger.info(f"  Min spacing filter: {len(eligible_indices):,} -> {len(result):,} "
                f"trades (spacing={min_spacing} bars = {min_spacing/BARS_PER_SEC:.0f}s)")
    return result


# ===========================================================================
# Forward path extraction (vectorized per-day)
# ===========================================================================

def extract_forward_paths(data: Dict, eligible_indices: np.ndarray,
                          max_horizon: int) -> Dict:
    """
    For each eligible bar, extract the forward price path in ticks relative
    to entry, respecting day boundaries.

    Returns:
        forward_ticks: float32[N_eligible, max_horizon]  -- signed by direction
            For longs:  (forward_prices - entry) / TICK_SIZE
            For shorts: (entry - forward_prices) / TICK_SIZE
            NaN where path extends past day boundary
        directions: int8[N_eligible]  -- +1 long, -1 short
        mag_preds: float32[N_eligible]  -- magnitude prediction for each trade
        valid_horizons: int32[N_eligible]  -- how many bars of valid forward path
    """
    mid_prices      = data['mid_prices']
    direction_preds = data['direction_preds']
    magnitude_preds = data['magnitude_preds']
    day_boundaries  = data['day_boundaries']
    n_days          = data['n_days']

    n_eligible = len(eligible_indices)
    logger.info(f"  Extracting forward paths for {n_eligible:,} bars, "
                f"max_horizon={max_horizon} bars ({max_horizon/BARS_PER_SEC:.0f}s)")

    # Build a lookup: for each bar index, what is the end-of-day index?
    # This avoids repeated searching through day_boundaries.
    day_ends = np.zeros(len(mid_prices), dtype=np.int32)
    for d in range(n_days):
        s = day_boundaries[d]
        e = day_boundaries[d + 1]
        day_ends[s:e] = e

    # Pre-allocate output arrays
    forward_ticks  = np.full((n_eligible, max_horizon), np.nan, dtype=np.float32)
    directions     = np.zeros(n_eligible, dtype=np.int8)
    mag_preds_out  = np.zeros(n_eligible, dtype=np.float32)
    valid_horizons = np.zeros(n_eligible, dtype=np.int32)

    for i, bar_idx in enumerate(eligible_indices):
        entry_price = float(mid_prices[bar_idx])
        direction   = 1 if direction_preds[bar_idx] > 0 else -1
        directions[i]    = direction
        mag_preds_out[i] = magnitude_preds[bar_idx]

        # Forward path: bar_idx+1 to min(bar_idx+max_horizon+1, day_end)
        day_end  = int(day_ends[bar_idx])
        path_end = min(bar_idx + max_horizon + 1, day_end)
        n_bars   = path_end - (bar_idx + 1)

        if n_bars <= 0:
            valid_horizons[i] = 0
            continue

        valid_horizons[i] = n_bars
        path = mid_prices[bar_idx + 1: path_end].astype(np.float64)

        if direction == 1:
            # Long: positive when price goes up
            forward_ticks[i, :n_bars] = ((path - entry_price) / TICK_SIZE).astype(np.float32)
        else:
            # Short: positive when price goes down
            forward_ticks[i, :n_bars] = ((entry_price - path) / TICK_SIZE).astype(np.float32)

    # Statistics
    med_valid = int(np.median(valid_horizons[valid_horizons > 0])) if (valid_horizons > 0).any() else 0
    logger.info(f"  Median valid horizon: {med_valid} bars ({med_valid/BARS_PER_SEC:.1f}s)")
    logger.info(f"  Bars with zero forward path: {(valid_horizons == 0).sum():,}")

    return {
        'forward_ticks':  forward_ticks,
        'directions':     directions,
        'mag_preds':      mag_preds_out,
        'valid_horizons': valid_horizons,
    }


# ===========================================================================
# Trade simulation engine
# ===========================================================================

def simulate_exits(forward_ticks: np.ndarray, valid_horizons: np.ndarray,
                   tp_ticks: Optional[float], sl_ticks: Optional[float],
                   horizon_bars: int, entry_cost: float) -> Dict:
    """
    Simulate exit logic for all trades given TP/SL/horizon parameters.

    For each trade:
      - Scan forward_ticks[0:horizon_bars] bar by bar
      - If forward_ticks[bar] >= tp_ticks: WIN (exit at TP)
      - If sl_ticks is not None and forward_ticks[bar] <= -sl_ticks: LOSS (exit at SL)
      - If neither hit by horizon_bars: exit at mark-to-market (last valid bar)
      - Subtract entry_cost from all PnL

    Returns per-trade PnL array and summary statistics.
    """
    n_trades = len(forward_ticks)
    pnl_ticks = np.zeros(n_trades, dtype=np.float64)
    exit_types = np.zeros(n_trades, dtype=np.int8)  # 0=timeout, 1=TP, 2=SL
    exit_bars  = np.zeros(n_trades, dtype=np.int32)

    for i in range(n_trades):
        n_valid = min(int(valid_horizons[i]), horizon_bars)
        if n_valid <= 0:
            # No forward data -- mark as timeout at zero
            pnl_ticks[i] = -entry_cost
            exit_types[i] = 0
            exit_bars[i] = 0
            continue

        path = forward_ticks[i, :n_valid]
        exited = False

        for b in range(n_valid):
            tick_val = float(path[b])

            # Check TP first (favorable)
            if tp_ticks is not None and tick_val >= tp_ticks:
                pnl_ticks[i] = tp_ticks - entry_cost
                exit_types[i] = 1  # TP hit
                exit_bars[i] = b + 1
                exited = True
                break

            # Check SL (adverse)
            if sl_ticks is not None and tick_val <= -sl_ticks:
                pnl_ticks[i] = -sl_ticks - entry_cost
                exit_types[i] = 2  # SL hit
                exit_bars[i] = b + 1
                exited = True
                break

        if not exited:
            # Timeout: exit at mark-to-market of last valid bar
            last_tick = float(path[n_valid - 1])
            pnl_ticks[i] = last_tick - entry_cost
            exit_types[i] = 0
            exit_bars[i] = n_valid

    return {
        'pnl_ticks':  pnl_ticks,
        'exit_types': exit_types,
        'exit_bars':  exit_bars,
    }


def simulate_exits_vectorized(forward_ticks: np.ndarray, valid_horizons: np.ndarray,
                              tp_ticks: Optional[float], sl_ticks: Optional[float],
                              horizon_bars: int, entry_cost: float) -> Dict:
    """
    Vectorized version of simulate_exits for speed on large datasets.
    Uses numpy operations to find first TP/SL hit across all trades simultaneously.
    """
    n_trades = forward_ticks.shape[0]
    max_h = min(horizon_bars, forward_ticks.shape[1])

    # Clip valid horizons to the requested horizon
    eff_horizons = np.minimum(valid_horizons, horizon_bars).astype(np.int32)

    # Build a mask of valid bars (within both horizon and day boundary)
    bar_indices = np.arange(max_h, dtype=np.int32)[np.newaxis, :]  # [1, max_h]
    valid_mask = bar_indices < eff_horizons[:, np.newaxis]          # [n_trades, max_h]

    # Get the path data, replacing invalid bars with NaN
    path = forward_ticks[:, :max_h].copy()
    path[~valid_mask] = np.nan

    # Find first bar where TP is hit
    if tp_ticks is not None:
        tp_hit = path >= tp_ticks
        tp_hit[~valid_mask] = False
        # First True index per row (argmax on bool finds first True; if no True, returns 0)
        tp_any = tp_hit.any(axis=1)
        tp_bar = np.where(tp_any, tp_hit.argmax(axis=1), max_h + 1)
    else:
        tp_any = np.zeros(n_trades, dtype=bool)
        tp_bar = np.full(n_trades, max_h + 1, dtype=np.int32)

    # Find first bar where SL is hit
    if sl_ticks is not None:
        sl_hit = path <= -sl_ticks
        sl_hit[~valid_mask] = False
        sl_any = sl_hit.any(axis=1)
        sl_bar = np.where(sl_any, sl_hit.argmax(axis=1), max_h + 1)
    else:
        sl_any = np.zeros(n_trades, dtype=bool)
        sl_bar = np.full(n_trades, max_h + 1, dtype=np.int32)

    # Determine which exit comes first
    # exit_type: 0=timeout, 1=TP, 2=SL
    exit_types = np.zeros(n_trades, dtype=np.int8)
    exit_bars  = np.zeros(n_trades, dtype=np.int32)
    pnl_ticks  = np.zeros(n_trades, dtype=np.float64)

    # TP wins (hit first or same bar as SL -- TP takes priority)
    tp_first = tp_any & ((tp_bar <= sl_bar) | ~sl_any)
    # SL wins (hit first and TP not first)
    sl_first = sl_any & ~tp_first & (sl_bar < tp_bar)
    # Timeout: neither hit
    timeout = ~tp_first & ~sl_first

    # TP exits
    exit_types[tp_first] = 1
    exit_bars[tp_first]  = tp_bar[tp_first] + 1
    pnl_ticks[tp_first]  = tp_ticks - entry_cost

    # SL exits
    if sl_ticks is not None:
        exit_types[sl_first] = 2
        exit_bars[sl_first]  = sl_bar[sl_first] + 1
        pnl_ticks[sl_first]  = -sl_ticks - entry_cost

    # Timeout exits: mark-to-market at last valid bar
    timeout_idx = np.where(timeout)[0]
    for i in timeout_idx:
        n_valid = int(eff_horizons[i])
        if n_valid > 0:
            last_bar = n_valid - 1
            pnl_ticks[i] = float(forward_ticks[i, last_bar]) - entry_cost
            exit_bars[i] = n_valid
        else:
            pnl_ticks[i] = -entry_cost
            exit_bars[i] = 0

    return {
        'pnl_ticks':  pnl_ticks,
        'exit_types': exit_types,
        'exit_bars':  exit_bars,
    }


# ===========================================================================
# Statistics computation
# ===========================================================================

def compute_trade_stats(pnl_ticks: np.ndarray, exit_types: np.ndarray,
                        exit_bars: np.ndarray, label: str = "") -> Dict:
    """Compute comprehensive trade statistics from simulation results."""
    n_trades = len(pnl_ticks)
    if n_trades == 0:
        return {'n_trades': 0, 'error': 'no trades'}

    pnl_dollars = pnl_ticks * TICK_VALUE

    # Win/loss
    wins   = pnl_ticks > 0
    losses = pnl_ticks < 0
    flat   = pnl_ticks == 0
    n_wins   = int(wins.sum())
    n_losses = int(losses.sum())
    win_rate = float(wins.mean()) if n_trades > 0 else 0.0

    # Profit factor
    gross_profit = float(pnl_ticks[wins].sum()) if n_wins > 0 else 0.0
    gross_loss   = float(abs(pnl_ticks[losses].sum())) if n_losses > 0 else 1e-9
    pf = gross_profit / gross_loss

    # Average
    avg_pnl_ticks   = float(pnl_ticks.mean())
    avg_pnl_dollars = float(pnl_dollars.mean())
    total_pnl       = float(pnl_dollars.sum())

    # Sharpe ratio of per-trade returns (annualized is meaningless here,
    # so we report raw Sharpe = mean/std of per-trade PnL)
    std_pnl = float(pnl_ticks.std()) if n_trades > 1 else 1e-9
    sharpe  = avg_pnl_ticks / max(std_pnl, 1e-9)

    # Max consecutive losses
    max_consec_loss = _max_consecutive(pnl_ticks < 0)

    # Exit type breakdown
    n_tp      = int((exit_types == 1).sum())
    n_sl      = int((exit_types == 2).sum())
    n_timeout = int((exit_types == 0).sum())

    # Average exit bar (holding time)
    avg_exit_bar = float(exit_bars.mean()) if n_trades > 0 else 0.0

    return {
        'n_trades':           n_trades,
        'win_rate':           win_rate,
        'n_wins':             n_wins,
        'n_losses':           n_losses,
        'profit_factor':      pf,
        'avg_pnl_ticks':      avg_pnl_ticks,
        'avg_pnl_dollars':    avg_pnl_dollars,
        'total_pnl_dollars':  total_pnl,
        'sharpe':             sharpe,
        'max_consec_losses':  max_consec_loss,
        'n_tp_exits':         n_tp,
        'n_sl_exits':         n_sl,
        'n_timeout_exits':    n_timeout,
        'avg_exit_bar':       avg_exit_bar,
        'avg_hold_sec':       avg_exit_bar / BARS_PER_SEC,
        'gross_profit_ticks': gross_profit,
        'gross_loss_ticks':   gross_loss,
    }


def _max_consecutive(bool_arr: np.ndarray) -> int:
    """Max consecutive True values in a boolean array."""
    if len(bool_arr) == 0:
        return 0
    max_run = 0
    current = 0
    for v in bool_arr:
        if v:
            current += 1
            if current > max_run:
                max_run = current
        else:
            current = 0
    return max_run


# ===========================================================================
# MFE / MAE analysis
# ===========================================================================

def compute_mfe_mae(forward_ticks: np.ndarray, valid_horizons: np.ndarray,
                    horizon_bars: int) -> Dict:
    """
    Compute MFE (max favorable excursion) and MAE (max adverse excursion)
    distributions for all trades at a given horizon.
    """
    n_trades = forward_ticks.shape[0]
    max_h = min(horizon_bars, forward_ticks.shape[1])

    mfe_arr = np.zeros(n_trades, dtype=np.float32)
    mae_arr = np.zeros(n_trades, dtype=np.float32)

    for i in range(n_trades):
        n_valid = min(int(valid_horizons[i]), horizon_bars)
        if n_valid <= 0:
            continue
        path = forward_ticks[i, :n_valid]
        finite = path[np.isfinite(path)]
        if len(finite) == 0:
            continue
        mfe_arr[i] = max(0.0, float(np.max(finite)))
        mae_arr[i] = max(0.0, float(-np.min(finite)))

    return {
        'mfe_mean':  float(mfe_arr.mean()),
        'mfe_p50':   float(np.median(mfe_arr)),
        'mfe_p75':   float(np.percentile(mfe_arr, 75)),
        'mfe_p90':   float(np.percentile(mfe_arr, 90)),
        'mae_mean':  float(mae_arr.mean()),
        'mae_p50':   float(np.median(mae_arr)),
        'mae_p75':   float(np.percentile(mae_arr, 75)),
        'mae_p90':   float(np.percentile(mae_arr, 90)),
    }


# ===========================================================================
# Grid search
# ===========================================================================

def run_grid_search(forward_ticks: np.ndarray, valid_horizons: np.ndarray,
                    mag_preds: np.ndarray,
                    tp_grid: List[float], sl_grid: List[Optional[float]],
                    horizon_grid: List[int], entry_cost: float,
                    label: str = "IS") -> List[Dict]:
    """
    Run full grid search over TP x SL x Horizon combinations.
    Returns list of result dicts sorted by avg_pnl_dollars descending.
    """
    n_combos = len(tp_grid) * len(sl_grid) * len(horizon_grid)
    logger.info(f"\n{'='*60}")
    logger.info(f"GRID SEARCH -- {label}")
    logger.info(f"  TP: {tp_grid}")
    logger.info(f"  SL: {sl_grid}")
    logger.info(f"  Horizons: {horizon_grid}")
    logger.info(f"  Total combos: {n_combos}")
    logger.info(f"  Entry cost: {entry_cost:.2f} ticks")
    logger.info(f"{'='*60}")

    results = []
    t0 = time.time()
    combo_idx = 0

    for horizon in horizon_grid:
        for tp in tp_grid:
            for sl in sl_grid:
                combo_idx += 1
                sim = simulate_exits_vectorized(
                    forward_ticks, valid_horizons,
                    tp_ticks=tp, sl_ticks=sl,
                    horizon_bars=horizon, entry_cost=entry_cost,
                )
                stats = compute_trade_stats(
                    sim['pnl_ticks'], sim['exit_types'], sim['exit_bars'],
                )
                stats['tp_ticks'] = tp
                stats['sl_ticks'] = sl
                stats['sl_label'] = f"{sl:.1f}" if sl is not None else "None"
                stats['horizon_bars'] = horizon
                stats['horizon_sec']  = horizon / BARS_PER_SEC
                results.append(stats)

                if combo_idx % 20 == 0 or combo_idx == n_combos:
                    elapsed = time.time() - t0
                    logger.info(f"  [{combo_idx}/{n_combos}] "
                                f"{elapsed:.1f}s elapsed")

    elapsed = time.time() - t0
    logger.info(f"  Grid search complete: {n_combos} combos in {elapsed:.1f}s")

    # Sort by avg_pnl_dollars descending
    results.sort(key=lambda r: r.get('avg_pnl_dollars', -999), reverse=True)

    return results


# ===========================================================================
# Adaptive exit strategies
# ===========================================================================

def simulate_adaptive_exits(forward_ticks: np.ndarray, valid_horizons: np.ndarray,
                            mag_preds: np.ndarray, horizon_bars: int,
                            entry_cost: float, tp_mult: float, sl_mult: float,
                            label: str = "Adaptive") -> Dict:
    """
    Adaptive exits: TP and SL scaled by predicted magnitude per trade.
      TP = mag_pred * tp_mult
      SL = mag_pred * sl_mult
    """
    n_trades = len(forward_ticks)
    pnl_ticks  = np.zeros(n_trades, dtype=np.float64)
    exit_types = np.zeros(n_trades, dtype=np.int8)
    exit_bars  = np.zeros(n_trades, dtype=np.int32)

    for i in range(n_trades):
        tp_i = float(mag_preds[i]) * tp_mult
        sl_i = float(mag_preds[i]) * sl_mult

        n_valid = min(int(valid_horizons[i]), horizon_bars)
        if n_valid <= 0:
            pnl_ticks[i] = -entry_cost
            continue

        path = forward_ticks[i, :n_valid]
        exited = False

        for b in range(n_valid):
            tick_val = float(path[b])
            if tick_val >= tp_i:
                pnl_ticks[i] = tp_i - entry_cost
                exit_types[i] = 1
                exit_bars[i] = b + 1
                exited = True
                break
            if tick_val <= -sl_i:
                pnl_ticks[i] = -sl_i - entry_cost
                exit_types[i] = 2
                exit_bars[i] = b + 1
                exited = True
                break

        if not exited:
            last_tick = float(path[n_valid - 1])
            pnl_ticks[i] = last_tick - entry_cost
            exit_types[i] = 0
            exit_bars[i] = n_valid

    stats = compute_trade_stats(pnl_ticks, exit_types, exit_bars, label=label)
    stats['strategy'] = label
    stats['tp_mult']  = tp_mult
    stats['sl_mult']  = sl_mult
    stats['horizon_bars'] = horizon_bars
    return stats


def simulate_time_scaled_exits(forward_ticks: np.ndarray, valid_horizons: np.ndarray,
                               mag_preds: np.ndarray, horizon_bars: int,
                               entry_cost: float, base_tp: float, base_sl: float,
                               label: str = "TimeScaled") -> Dict:
    """
    Time-based scaling: as time runs out, tighten the stop and relax the target.

    At bar b out of horizon H:
      fraction_elapsed = b / H
      tp_effective = base_tp * (1.0 - 0.5 * fraction_elapsed)  -- loosen target less aggressively
      sl_effective = base_sl * (1.0 - 0.7 * fraction_elapsed)  -- tighten stop more aggressively

    In the last 20% of horizon, if P&L > 0, take profit.
    """
    n_trades = len(forward_ticks)
    pnl_ticks  = np.zeros(n_trades, dtype=np.float64)
    exit_types = np.zeros(n_trades, dtype=np.int8)
    exit_bars  = np.zeros(n_trades, dtype=np.int32)

    for i in range(n_trades):
        n_valid = min(int(valid_horizons[i]), horizon_bars)
        if n_valid <= 0:
            pnl_ticks[i] = -entry_cost
            continue

        path = forward_ticks[i, :n_valid]
        exited = False

        for b in range(n_valid):
            tick_val = float(path[b])
            frac = (b + 1) / horizon_bars  # fraction of horizon elapsed

            # Time-scaled TP/SL
            tp_eff = base_tp * max(0.3, 1.0 - 0.5 * frac)
            sl_eff = base_sl * max(0.2, 1.0 - 0.7 * frac)

            # In last 20% of horizon, take any profit
            if frac >= 0.8 and tick_val > 0:
                pnl_ticks[i] = tick_val - entry_cost
                exit_types[i] = 1  # treat as TP
                exit_bars[i] = b + 1
                exited = True
                break

            if tick_val >= tp_eff:
                pnl_ticks[i] = tp_eff - entry_cost
                exit_types[i] = 1
                exit_bars[i] = b + 1
                exited = True
                break

            if tick_val <= -sl_eff:
                pnl_ticks[i] = -sl_eff - entry_cost
                exit_types[i] = 2
                exit_bars[i] = b + 1
                exited = True
                break

        if not exited:
            last_tick = float(path[n_valid - 1])
            pnl_ticks[i] = last_tick - entry_cost
            exit_types[i] = 0
            exit_bars[i] = n_valid

    stats = compute_trade_stats(pnl_ticks, exit_types, exit_bars, label=label)
    stats['strategy']     = label
    stats['base_tp']      = base_tp
    stats['base_sl']      = base_sl
    stats['horizon_bars'] = horizon_bars
    return stats


def simulate_trailing_stop(forward_ticks: np.ndarray, valid_horizons: np.ndarray,
                           tp_ticks: float, trail_start: float, trail_dist: float,
                           horizon_bars: int, entry_cost: float,
                           label: str = "TrailingStop") -> Dict:
    """
    Trailing stop strategy:
      - Fixed TP target (take profit when hit)
      - Once trade is trail_start ticks in profit, activate trailing stop
      - Trailing stop follows trail_dist ticks behind the highest point
      - If trail stop hit, exit at (peak - trail_dist)
      - On timeout, exit at mark-to-market
    """
    n_trades = len(forward_ticks)
    pnl_ticks  = np.zeros(n_trades, dtype=np.float64)
    exit_types = np.zeros(n_trades, dtype=np.int8)
    exit_bars  = np.zeros(n_trades, dtype=np.int32)

    for i in range(n_trades):
        n_valid = min(int(valid_horizons[i]), horizon_bars)
        if n_valid <= 0:
            pnl_ticks[i] = -entry_cost
            continue

        path = forward_ticks[i, :n_valid]
        exited = False
        peak = 0.0

        for b in range(n_valid):
            tick_val = float(path[b])

            # Update peak
            if tick_val > peak:
                peak = tick_val

            # Fixed TP
            if tick_val >= tp_ticks:
                pnl_ticks[i] = tp_ticks - entry_cost
                exit_types[i] = 1
                exit_bars[i] = b + 1
                exited = True
                break

            # Trailing stop (only active after reaching trail_start profit)
            if peak >= trail_start:
                trail_level = peak - trail_dist
                if tick_val <= trail_level:
                    # Exit at trail level (or actual tick_val if it gapped through)
                    pnl_ticks[i] = max(tick_val, trail_level) - entry_cost
                    exit_types[i] = 2  # treat as SL-like
                    exit_bars[i] = b + 1
                    exited = True
                    break

        if not exited:
            last_tick = float(path[n_valid - 1])
            pnl_ticks[i] = last_tick - entry_cost
            exit_types[i] = 0
            exit_bars[i] = n_valid

    stats = compute_trade_stats(pnl_ticks, exit_types, exit_bars, label=label)
    stats['strategy']     = label
    stats['tp_ticks']     = tp_ticks
    stats['trail_start']  = trail_start
    stats['trail_dist']   = trail_dist
    stats['horizon_bars'] = horizon_bars
    return stats


# ===========================================================================
# Output formatting
# ===========================================================================

def print_results_table(results: List[Dict], top_n: int = 20,
                        label: str = "Results") -> None:
    """Print a formatted table of top parameter combos."""
    logger.info(f"\n{'='*110}")
    logger.info(f"TOP {min(top_n, len(results))} PARAMETER COMBOS -- {label}")
    logger.info(f"{'='*110}")
    logger.info(
        f"  {'Rank':>4s}  {'TP':>5s}  {'SL':>5s}  {'Hrzn':>5s}  "
        f"{'N':>7s}  {'WR':>6s}  {'PF':>6s}  "
        f"{'$/trade':>8s}  {'Total$':>10s}  {'Sharpe':>7s}  "
        f"{'MaxCL':>5s}  {'TP%':>5s}  {'SL%':>5s}  {'TO%':>5s}  "
        f"{'AvgHold':>7s}"
    )
    logger.info(f"  {'-'*106}")

    for rank, r in enumerate(results[:top_n], 1):
        n = r.get('n_trades', 0)
        if n == 0:
            continue
        tp_str = f"{r['tp_ticks']:.1f}" if r.get('tp_ticks') is not None else "None"
        sl_str = r.get('sl_label', 'None')
        hrzn_s = f"{r['horizon_sec']:.0f}s"
        tp_pct = r['n_tp_exits'] / n * 100 if n > 0 else 0
        sl_pct = r['n_sl_exits'] / n * 100 if n > 0 else 0
        to_pct = r['n_timeout_exits'] / n * 100 if n > 0 else 0

        logger.info(
            f"  {rank:>4d}  {tp_str:>5s}  {sl_str:>5s}  {hrzn_s:>5s}  "
            f"{n:>7,d}  {r['win_rate']:>5.1%}  {r['profit_factor']:>6.2f}  "
            f"${r['avg_pnl_dollars']:>+7.2f}  ${r['total_pnl_dollars']:>+9,.0f}  "
            f"{r['sharpe']:>7.3f}  "
            f"{r['max_consec_losses']:>5d}  {tp_pct:>4.0f}%  {sl_pct:>4.0f}%  {to_pct:>4.0f}%  "
            f"{r['avg_hold_sec']:>6.1f}s"
        )


def print_adaptive_results(results: List[Dict], label: str = "Adaptive Strategies") -> None:
    """Print adaptive strategy results."""
    logger.info(f"\n{'='*100}")
    logger.info(f"ADAPTIVE EXIT STRATEGIES -- {label}")
    logger.info(f"{'='*100}")
    logger.info(
        f"  {'Strategy':<25s}  {'N':>7s}  {'WR':>6s}  {'PF':>6s}  "
        f"{'$/trade':>8s}  {'Total$':>10s}  {'Sharpe':>7s}  {'MaxCL':>5s}  {'AvgHold':>7s}"
    )
    logger.info(f"  {'-'*90}")

    for r in results:
        n = r.get('n_trades', 0)
        if n == 0:
            continue
        name = r.get('strategy', 'unknown')
        logger.info(
            f"  {name:<25s}  {n:>7,d}  {r['win_rate']:>5.1%}  {r['profit_factor']:>6.2f}  "
            f"${r['avg_pnl_dollars']:>+7.2f}  ${r['total_pnl_dollars']:>+9,.0f}  "
            f"{r['sharpe']:>7.3f}  {r['max_consec_losses']:>5d}  {r['avg_hold_sec']:>6.1f}s"
        )


# ===========================================================================
# Cross-validation: evaluate IS top combos on OOS
# ===========================================================================

def evaluate_on_oos(is_top_results: List[Dict], oos_forward_ticks: np.ndarray,
                    oos_valid_horizons: np.ndarray, oos_mag_preds: np.ndarray,
                    entry_cost: float, top_n: int = 20) -> List[Dict]:
    """
    Take the top N parameter combos from IS and evaluate them on OOS data.
    Returns OOS results in the same order as IS top combos (for side-by-side comparison).
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"OOS EVALUATION of top {top_n} IS combos")
    logger.info(f"{'='*60}")

    oos_results = []
    for rank, is_r in enumerate(is_top_results[:top_n], 1):
        tp = is_r['tp_ticks']
        sl = is_r['sl_ticks']
        horizon = is_r['horizon_bars']

        sim = simulate_exits_vectorized(
            oos_forward_ticks, oos_valid_horizons,
            tp_ticks=tp, sl_ticks=sl,
            horizon_bars=horizon, entry_cost=entry_cost,
        )
        stats = compute_trade_stats(sim['pnl_ticks'], sim['exit_types'], sim['exit_bars'])
        stats['tp_ticks']     = tp
        stats['sl_ticks']     = sl
        stats['sl_label']     = f"{sl:.1f}" if sl is not None else "None"
        stats['horizon_bars'] = horizon
        stats['horizon_sec']  = horizon / BARS_PER_SEC
        stats['is_rank']      = rank
        stats['is_avg_pnl']   = is_r['avg_pnl_dollars']
        stats['is_pf']        = is_r['profit_factor']
        oos_results.append(stats)

    return oos_results


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_pipeline(args):
    start_time = time.time()
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info("=" * 80)
    logger.info("EXIT OPTIMIZATION -- TP/SL Parameter Search for ES Futures")
    logger.info("=" * 80)
    logger.info(f"Timestamp:     {timestamp}")
    logger.info(f"Predictions:   {args.load_predictions}")
    logger.info(f"OOS split day: {args.oos_split_day}")
    logger.info(f"Entry cost:    {MKT_ENTRY_COST_TICKS:.2f} ticks")
    logger.info(f"Gate:          mag>={MAG_THRESHOLD}t AND dir top {(1-DIR_QUANTILE)*100:.0f}%")

    # ------------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------------
    full_data = load_predictions(args.load_predictions)
    n_days = full_data['n_days']

    # ------------------------------------------------------------------
    # IS / OOS split
    # ------------------------------------------------------------------
    oos_split = args.oos_split_day
    if oos_split is None:
        oos_split = 70  # default for 100-day dataset

    if oos_split >= n_days:
        logger.warning(f"--oos-split-day {oos_split} >= n_days {n_days}, using 70/30 split")
        oos_split = int(n_days * 0.7)

    logger.info(f"\nIS:  days 0..{oos_split-1}  ({oos_split} days)")
    logger.info(f"OOS: days {oos_split}..{n_days-1}  ({n_days - oos_split} days)")

    is_data  = _slice_data(full_data, 0, oos_split)
    oos_data = _slice_data(full_data, oos_split, n_days)

    # Free full data
    del full_data

    # ------------------------------------------------------------------
    # Find eligible bars and extract forward paths
    # ------------------------------------------------------------------
    max_horizon = max(HORIZON_GRID)

    logger.info(f"\n--- IN-SAMPLE (days 0..{oos_split-1}) ---")
    is_eligible = find_eligible_bars(is_data)
    if is_eligible['n_eligible'] == 0:
        logger.error("No eligible IS bars found! Aborting.")
        return

    # Apply minimum spacing to prevent overlapping trades
    # Use max_horizon so trades at the longest horizon are still non-overlapping
    is_spaced_indices = apply_min_spacing(
        is_eligible['indices'], is_data['day_boundaries'], min_spacing=max_horizon)
    if len(is_spaced_indices) == 0:
        logger.error("No IS bars after spacing filter! Aborting.")
        return

    is_paths = extract_forward_paths(is_data, is_spaced_indices, max_horizon)

    # Freeze IS direction threshold for OOS to prevent look-ahead bias
    is_dir_threshold = is_eligible['dir_threshold']
    logger.info(f"\n  FROZEN IS direction threshold for OOS: {is_dir_threshold:.6f}")

    logger.info(f"\n--- OUT-OF-SAMPLE (days {oos_split}..{n_days-1}) ---")
    oos_eligible = find_eligible_bars(oos_data, dir_threshold=is_dir_threshold)
    if oos_eligible['n_eligible'] == 0:
        logger.warning("No eligible OOS bars found! Will skip OOS evaluation.")
        oos_paths = None
    else:
        oos_spaced_indices = apply_min_spacing(
            oos_eligible['indices'], oos_data['day_boundaries'], min_spacing=max_horizon)
        if len(oos_spaced_indices) == 0:
            logger.warning("No OOS bars after spacing filter! Skipping OOS.")
            oos_paths = None
        else:
            oos_paths = extract_forward_paths(oos_data, oos_spaced_indices, max_horizon)

    # Free data arrays we no longer need
    del is_data, oos_data

    # ------------------------------------------------------------------
    # MFE/MAE analysis at each horizon
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("MFE/MAE ANALYSIS BY HORIZON")
    logger.info(f"{'='*60}")

    for h in HORIZON_GRID:
        mfe_mae = compute_mfe_mae(is_paths['forward_ticks'], is_paths['valid_horizons'], h)
        logger.info(f"\n  Horizon: {h} bars ({h/BARS_PER_SEC:.0f}s)")
        logger.info(f"    MFE: mean={mfe_mae['mfe_mean']:.2f}t  "
                     f"P50={mfe_mae['mfe_p50']:.2f}t  "
                     f"P75={mfe_mae['mfe_p75']:.2f}t  "
                     f"P90={mfe_mae['mfe_p90']:.2f}t")
        logger.info(f"    MAE: mean={mfe_mae['mae_mean']:.2f}t  "
                     f"P50={mfe_mae['mae_p50']:.2f}t  "
                     f"P75={mfe_mae['mae_p75']:.2f}t  "
                     f"P90={mfe_mae['mae_p90']:.2f}t")
        logger.info(f"    MFE/MAE ratio: {mfe_mae['mfe_mean']/max(mfe_mae['mae_mean'],1e-6):.2f}")

    # ------------------------------------------------------------------
    # Grid search on IS
    # ------------------------------------------------------------------
    is_grid_results = run_grid_search(
        is_paths['forward_ticks'], is_paths['valid_horizons'],
        is_paths['mag_preds'],
        tp_grid=TP_GRID, sl_grid=SL_GRID, horizon_grid=HORIZON_GRID,
        entry_cost=MKT_ENTRY_COST_TICKS, label="IS",
    )

    top_n = args.top_n
    print_results_table(is_grid_results, top_n=top_n, label="IS (sorted by $/trade)")

    # ------------------------------------------------------------------
    # Evaluate top IS combos on OOS
    # ------------------------------------------------------------------
    oos_grid_results = None
    if oos_paths is not None:
        oos_grid_results = evaluate_on_oos(
            is_grid_results, oos_paths['forward_ticks'],
            oos_paths['valid_horizons'], oos_paths['mag_preds'],
            entry_cost=MKT_ENTRY_COST_TICKS, top_n=top_n,
        )
        print_results_table(oos_grid_results, top_n=top_n,
                            label="OOS (same combos as IS top, sorted by IS rank)")

        # Find best OOS combo
        best_oos_idx = max(range(len(oos_grid_results)),
                           key=lambda i: oos_grid_results[i].get('avg_pnl_dollars', -999))
        best_oos = oos_grid_results[best_oos_idx]
        logger.info(f"\n  *** BEST OOS COMBO (IS rank #{best_oos['is_rank']}): ***")
        logger.info(f"      TP={best_oos['tp_ticks']:.1f}t  "
                     f"SL={best_oos['sl_label']}  "
                     f"Horizon={best_oos['horizon_sec']:.0f}s")
        logger.info(f"      OOS: WR={best_oos['win_rate']:.1%}  "
                     f"PF={best_oos['profit_factor']:.2f}  "
                     f"$/trade=${best_oos['avg_pnl_dollars']:+.2f}  "
                     f"Total=${best_oos['total_pnl_dollars']:+,.0f}")
        logger.info(f"      IS:  $/trade=${best_oos['is_avg_pnl']:+.2f}  "
                     f"PF={best_oos['is_pf']:.2f}")

        # Overfitting analysis
        logger.info(f"\n{'='*60}")
        logger.info(f"OVERFITTING ANALYSIS: IS vs OOS performance")
        logger.info(f"{'='*60}")
        n_positive_is  = sum(1 for r in is_grid_results[:top_n] if r['avg_pnl_dollars'] > 0)
        n_positive_oos = sum(1 for r in oos_grid_results if r['avg_pnl_dollars'] > 0)
        logger.info(f"  Top {top_n} IS combos with positive $/trade:  IS={n_positive_is}  OOS={n_positive_oos}")

        # IS-OOS PnL correlation
        is_pnls  = [r['avg_pnl_dollars'] for r in is_grid_results[:top_n]]
        oos_pnls = [r['avg_pnl_dollars'] for r in oos_grid_results]
        if len(is_pnls) > 2:
            corr = float(np.corrcoef(is_pnls, oos_pnls)[0, 1])
            logger.info(f"  IS-OOS $/trade rank correlation: {corr:+.3f}")
            if corr > 0.5:
                logger.info(f"  -> Good: IS performance predicts OOS (low overfitting)")
            elif corr > 0.0:
                logger.info(f"  -> Moderate: some IS-OOS agreement, some parameter sensitivity")
            else:
                logger.info(f"  -> WARNING: IS performance does NOT predict OOS (overfitting likely)")

    # ------------------------------------------------------------------
    # Adaptive exits on IS and OOS
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("ADAPTIVE EXIT STRATEGIES")
    logger.info(f"{'='*60}")

    adaptive_results_is = []
    adaptive_results_oos = []

    # Strategy 1: Magnitude-scaled TP/SL
    for horizon in [100, 200, 500]:
        r_is = simulate_adaptive_exits(
            is_paths['forward_ticks'], is_paths['valid_horizons'],
            is_paths['mag_preds'], horizon_bars=horizon,
            entry_cost=MKT_ENTRY_COST_TICKS,
            tp_mult=ADAPTIVE_TP_MULT, sl_mult=ADAPTIVE_SL_MULT,
            label=f"MagScaled TP={ADAPTIVE_TP_MULT}x SL={ADAPTIVE_SL_MULT}x H={horizon/BARS_PER_SEC:.0f}s",
        )
        adaptive_results_is.append(r_is)

        if oos_paths is not None:
            r_oos = simulate_adaptive_exits(
                oos_paths['forward_ticks'], oos_paths['valid_horizons'],
                oos_paths['mag_preds'], horizon_bars=horizon,
                entry_cost=MKT_ENTRY_COST_TICKS,
                tp_mult=ADAPTIVE_TP_MULT, sl_mult=ADAPTIVE_SL_MULT,
                label=f"MagScaled TP={ADAPTIVE_TP_MULT}x SL={ADAPTIVE_SL_MULT}x H={horizon/BARS_PER_SEC:.0f}s",
            )
            adaptive_results_oos.append(r_oos)

    # Strategy 2: Time-based scaling (with a few base TP/SL combos)
    for base_tp, base_sl in [(2.0, 1.5), (3.0, 2.0), (2.5, 1.0)]:
        for horizon in [100, 200]:
            r_is = simulate_time_scaled_exits(
                is_paths['forward_ticks'], is_paths['valid_horizons'],
                is_paths['mag_preds'], horizon_bars=horizon,
                entry_cost=MKT_ENTRY_COST_TICKS,
                base_tp=base_tp, base_sl=base_sl,
                label=f"TimeScale TP={base_tp}t SL={base_sl}t H={horizon/BARS_PER_SEC:.0f}s",
            )
            adaptive_results_is.append(r_is)

            if oos_paths is not None:
                r_oos = simulate_time_scaled_exits(
                    oos_paths['forward_ticks'], oos_paths['valid_horizons'],
                    oos_paths['mag_preds'], horizon_bars=horizon,
                    entry_cost=MKT_ENTRY_COST_TICKS,
                    base_tp=base_tp, base_sl=base_sl,
                    label=f"TimeScale TP={base_tp}t SL={base_sl}t H={horizon/BARS_PER_SEC:.0f}s",
                )
                adaptive_results_oos.append(r_oos)

    # Strategy 3: Trailing stop
    for tp, trail_start, trail_dist in [(3.0, 1.5, 0.75), (4.0, 2.0, 1.0),
                                         (2.5, 1.0, 0.5), (5.0, 2.5, 1.5)]:
        for horizon in [200, 500]:
            r_is = simulate_trailing_stop(
                is_paths['forward_ticks'], is_paths['valid_horizons'],
                tp_ticks=tp, trail_start=trail_start, trail_dist=trail_dist,
                horizon_bars=horizon, entry_cost=MKT_ENTRY_COST_TICKS,
                label=f"Trail TP={tp}t start={trail_start}t dist={trail_dist}t H={horizon/BARS_PER_SEC:.0f}s",
            )
            adaptive_results_is.append(r_is)

            if oos_paths is not None:
                r_oos = simulate_trailing_stop(
                    oos_paths['forward_ticks'], oos_paths['valid_horizons'],
                    tp_ticks=tp, trail_start=trail_start, trail_dist=trail_dist,
                    horizon_bars=horizon, entry_cost=MKT_ENTRY_COST_TICKS,
                    label=f"Trail TP={tp}t start={trail_start}t dist={trail_dist}t H={horizon/BARS_PER_SEC:.0f}s",
                )
                adaptive_results_oos.append(r_oos)

    # Sort adaptive by $/trade
    adaptive_results_is.sort(key=lambda r: r.get('avg_pnl_dollars', -999), reverse=True)
    print_adaptive_results(adaptive_results_is, label="IS -- Adaptive Strategies")

    if adaptive_results_oos:
        adaptive_results_oos.sort(key=lambda r: r.get('avg_pnl_dollars', -999), reverse=True)
        print_adaptive_results(adaptive_results_oos, label="OOS -- Adaptive Strategies")

    # ------------------------------------------------------------------
    # Trailing stop vs fixed TP analysis
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("ANALYSIS: Does trailing stop beat fixed TP?")
    logger.info(f"{'='*60}")

    # Compare best fixed-grid result vs best trailing result
    best_fixed_is = is_grid_results[0] if is_grid_results else None
    trailing_is = [r for r in adaptive_results_is if 'Trail' in r.get('strategy', '')]
    best_trail_is = trailing_is[0] if trailing_is else None

    if best_fixed_is and best_trail_is:
        logger.info(f"\n  BEST FIXED (IS):")
        logger.info(f"    TP={best_fixed_is['tp_ticks']:.1f}t  "
                     f"SL={best_fixed_is.get('sl_label','?')}  "
                     f"H={best_fixed_is['horizon_sec']:.0f}s")
        logger.info(f"    WR={best_fixed_is['win_rate']:.1%}  "
                     f"PF={best_fixed_is['profit_factor']:.2f}  "
                     f"$/trade=${best_fixed_is['avg_pnl_dollars']:+.2f}")

        logger.info(f"\n  BEST TRAILING STOP (IS):")
        logger.info(f"    {best_trail_is['strategy']}")
        logger.info(f"    WR={best_trail_is['win_rate']:.1%}  "
                     f"PF={best_trail_is['profit_factor']:.2f}  "
                     f"$/trade=${best_trail_is['avg_pnl_dollars']:+.2f}")

        if best_trail_is['avg_pnl_dollars'] > best_fixed_is['avg_pnl_dollars']:
            pct_better = ((best_trail_is['avg_pnl_dollars'] - best_fixed_is['avg_pnl_dollars'])
                          / max(abs(best_fixed_is['avg_pnl_dollars']), 0.01) * 100)
            logger.info(f"\n  -> TRAILING STOP WINS by {pct_better:.1f}% on IS")
        else:
            pct_better = ((best_fixed_is['avg_pnl_dollars'] - best_trail_is['avg_pnl_dollars'])
                          / max(abs(best_trail_is['avg_pnl_dollars']), 0.01) * 100)
            logger.info(f"\n  -> FIXED TP WINS by {pct_better:.1f}% on IS")

    # Same comparison on OOS
    if oos_grid_results and adaptive_results_oos:
        best_fixed_oos = max(oos_grid_results,
                             key=lambda r: r.get('avg_pnl_dollars', -999))
        trailing_oos = [r for r in adaptive_results_oos if 'Trail' in r.get('strategy', '')]
        best_trail_oos = trailing_oos[0] if trailing_oos else None

        if best_trail_oos:
            logger.info(f"\n  BEST FIXED (OOS):")
            logger.info(f"    TP={best_fixed_oos['tp_ticks']:.1f}t  "
                         f"SL={best_fixed_oos.get('sl_label','?')}  "
                         f"H={best_fixed_oos['horizon_sec']:.0f}s")
            logger.info(f"    WR={best_fixed_oos['win_rate']:.1%}  "
                         f"PF={best_fixed_oos['profit_factor']:.2f}  "
                         f"$/trade=${best_fixed_oos['avg_pnl_dollars']:+.2f}")

            logger.info(f"\n  BEST TRAILING STOP (OOS):")
            logger.info(f"    {best_trail_oos['strategy']}")
            logger.info(f"    WR={best_trail_oos['win_rate']:.1%}  "
                         f"PF={best_trail_oos['profit_factor']:.2f}  "
                         f"$/trade=${best_trail_oos['avg_pnl_dollars']:+.2f}")

            if best_trail_oos['avg_pnl_dollars'] > best_fixed_oos['avg_pnl_dollars']:
                logger.info(f"\n  -> TRAILING STOP WINS ON OOS (more robust)")
            else:
                logger.info(f"\n  -> FIXED TP WINS ON OOS (simpler is better)")

    # ------------------------------------------------------------------
    # Baseline comparison
    # ------------------------------------------------------------------
    logger.info(f"\n{'='*60}")
    logger.info("BASELINE COMPARISON")
    logger.info(f"{'='*60}")

    # Find the baseline (TP=2.0, SL=None, Horizon=100)
    baseline = None
    for r in is_grid_results:
        if (r['tp_ticks'] == 2.0 and r['sl_ticks'] is None
                and r['horizon_bars'] == 100):
            baseline = r
            break

    if baseline:
        logger.info(f"  Baseline (TP=2.0t, no SL, H=10s):")
        logger.info(f"    WR={baseline['win_rate']:.1%}  PF={baseline['profit_factor']:.2f}  "
                     f"$/trade=${baseline['avg_pnl_dollars']:+.2f}")

        best = is_grid_results[0]
        logger.info(f"\n  Best IS combo (TP={best['tp_ticks']:.1f}t, "
                     f"SL={best.get('sl_label','?')}, H={best['horizon_sec']:.0f}s):")
        logger.info(f"    WR={best['win_rate']:.1%}  PF={best['profit_factor']:.2f}  "
                     f"$/trade=${best['avg_pnl_dollars']:+.2f}")

        improvement = best['avg_pnl_dollars'] - baseline['avg_pnl_dollars']
        logger.info(f"\n  Improvement over baseline: ${improvement:+.2f}/trade "
                     f"({improvement/max(abs(baseline['avg_pnl_dollars']),0.01)*100:+.0f}%)")
    else:
        logger.info("  Baseline combo (TP=2.0, SL=None, H=100) not found in grid")

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    def _serialize(obj):
        """Recursively convert numpy types for JSON."""
        if isinstance(obj, dict):
            return {k: _serialize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_serialize(v) for v in obj]
        elif isinstance(obj, (np.integer,)):
            return int(obj)
        elif isinstance(obj, (np.floating,)):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif obj is None:
            return None
        else:
            return obj

    out_file = RESULTS_DIR / f"exit_optimization_{timestamp}.json"
    save_data = _serialize({
        'timestamp':            timestamp,
        'predictions':          args.load_predictions,
        'oos_split_day':        oos_split,
        'n_days_total':         n_days,
        'n_eligible_is':        is_eligible['n_eligible'],
        'n_eligible_oos':       oos_eligible['n_eligible'] if oos_eligible else 0,
        'entry_cost_ticks':     MKT_ENTRY_COST_TICKS,
        'gate_mag_threshold':   MAG_THRESHOLD,
        'gate_dir_quantile':    DIR_QUANTILE,
        'is_grid_top20':        is_grid_results[:top_n],
        'oos_grid_top20':       oos_grid_results[:top_n] if oos_grid_results else [],
        'adaptive_is':          adaptive_results_is,
        'adaptive_oos':         adaptive_results_oos,
        'total_time_sec':       time.time() - start_time,
    })

    with open(str(out_file), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    total_time = time.time() - start_time
    logger.info(f"\n{'='*80}")
    logger.info(f"EXIT OPTIMIZATION COMPLETE -- {total_time:.0f}s ({total_time/60:.1f}m)")
    logger.info(f"Results saved: {out_file}")
    logger.info(f"Log:           {_log_file}")
    logger.info(f"{'='*80}")

    # ------------------------------------------------------------------
    # Print compact summary for Discord / terminal
    # ------------------------------------------------------------------
    _print_summary(is_grid_results, oos_grid_results, adaptive_results_is,
                   adaptive_results_oos, baseline, is_eligible, oos_eligible,
                   oos_split, n_days, total_time)

    return save_data


def _print_summary(is_grid, oos_grid, adapt_is, adapt_oos, baseline,
                   is_elig, oos_elig, oos_split, n_days, total_time):
    """Print compact summary."""
    lines = [
        "",
        "--- SUMMARY ---",
        f"**EXIT OPTIMIZATION COMPLETE** -- {total_time/60:.1f} min",
        f"IS: {oos_split} days ({is_elig['n_eligible']:,} trades) | "
        f"OOS: {n_days - oos_split} days ({oos_elig.get('n_eligible', 0):,} trades)",
        "",
    ]

    if baseline:
        lines.append(f"**Baseline (TP=2.0t, no SL, 10s):** "
                      f"WR={baseline['win_rate']:.1%} PF={baseline['profit_factor']:.2f} "
                      f"${baseline['avg_pnl_dollars']:+.2f}/trade")

    if is_grid:
        best = is_grid[0]
        lines.append(f"**Best IS:** TP={best['tp_ticks']:.1f}t SL={best.get('sl_label','?')} "
                      f"H={best['horizon_sec']:.0f}s  "
                      f"WR={best['win_rate']:.1%} PF={best['profit_factor']:.2f} "
                      f"${best['avg_pnl_dollars']:+.2f}/trade")

    if oos_grid:
        best_oos = max(oos_grid, key=lambda r: r.get('avg_pnl_dollars', -999))
        lines.append(f"**Best OOS:** TP={best_oos['tp_ticks']:.1f}t "
                      f"SL={best_oos.get('sl_label','?')} "
                      f"H={best_oos['horizon_sec']:.0f}s  "
                      f"WR={best_oos['win_rate']:.1%} PF={best_oos['profit_factor']:.2f} "
                      f"${best_oos['avg_pnl_dollars']:+.2f}/trade")

    if adapt_is:
        best_adapt = adapt_is[0]
        lines.append(f"**Best Adaptive IS:** {best_adapt['strategy']}  "
                      f"WR={best_adapt['win_rate']:.1%} PF={best_adapt['profit_factor']:.2f} "
                      f"${best_adapt['avg_pnl_dollars']:+.2f}/trade")

    summary = "\n".join(lines)
    print(summary)
    logger.info(summary)


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Exit Optimization -- TP/SL parameter search for ES futures',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--load-predictions', type=str, required=True,
        help='Path to predictions NPZ file',
    )
    parser.add_argument(
        '--oos-split-day', type=int, default=70,
        help='IS/OOS split day (default: 70, i.e. 70 IS + 30 OOS)',
    )
    parser.add_argument(
        '--top-n', type=int, default=20,
        help='Number of top combos to display and evaluate on OOS (default: 20)',
    )
    args = parser.parse_args()

    try:
        return run_pipeline(args)
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
