"""
Conditional Filter Strategy Simulator

Tests COMBINED filter strategies on MFE predictions to find the optimal
trading regime filters for production deployment.

Individual edge findings (from edge_analysis.py):
  1. Time filter: skip 9:30-10:30 open (worst IC), mid-day 10:30-14:30 is best
  2. Vol regime: skip highest-vol quintile days (Q5 worst, Q2 best)
  3. Signal strength: only trade strong signals (Q4-Q5 by |dir_pred|)
  4. Gate filter: mag > 3.0t (already proven profitable OOS)

This script tests those filters individually and in combination, comparing
IS (days 0-69) vs OOS (days 70-99) performance to check for overfitting.

NPZ format (from mfe_path_analysis.py / magnitude_gated_sim.py):
    mid_prices        float32[N]   mid-price series (all days concatenated)
    direction_preds   float32[N]   direction model prediction (NaN where no pred)
    magnitude_preds   float32[N]   magnitude model prediction in ticks (NaN where no pred)
    direction_target  float32[N]   direction target (mfe_net in ticks)
    magnitude_target  float32[N]   magnitude target (abs ticks to horizon)
    day_boundaries    int32[D+1]   start/end indices for each day

Usage:
    python alpha_discovery/conditional_sim.py \\
        --load-predictions results/predictions_mfe_path_TIMESTAMP.npz

    # Custom OOS split:
    python alpha_discovery/conditional_sim.py \\
        --load-predictions results/predictions_mfe_path_TIMESTAMP.npz \\
        --oos-split-day 70

    # Run on server (no GPU needed, just numpy/scipy):
    python alpha_discovery/conditional_sim.py \\
        --load-predictions /path/to/predictions.npz
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
# Logging — same convention as edge_analysis.py
# ---------------------------------------------------------------------------
_ts = datetime.now().strftime('%Y%m%d_%H%M%S')
_log_file = RESULTS_DIR / f"conditional_sim_{_ts}.log"
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
logger = logging.getLogger('conditional_sim')

# ---------------------------------------------------------------------------
# Constants (ES futures, 100ms bars)
# ---------------------------------------------------------------------------
TICK_SIZE       = 0.25          # ES tick size in points
TICK_VALUE      = 12.50         # $ per tick (ES full contract)
COMMISSION_RT   = 4.70          # $4.70 round-trip (AMP/Rithmic, HC #52 canonical)
COMMISSION_TICKS = COMMISSION_RT / TICK_VALUE   # 0.376 ticks
BARS_PER_SEC    = 10            # 100ms resolution
BARS_PER_DAY    = 234_000       # 6.5h * 3600s/h * 10 bars/s

# Time thresholds (seconds from 9:30:00 ET, the session open)
OPEN_END_SEC    = 3600          # 10:30 = 60 min = 3600s from 9:30
MIDDAY_START_SEC = 3600         # 10:30
MIDDAY_END_SEC  = 18000         # 14:30 = 5h = 18000s from 9:30

# Gate / signal defaults
MAG_GATE_TICKS  = 3.0           # magnitude prediction gate threshold
QUINTILE_COUNT  = 5

# Market order cost model (same as edge_analysis.py Section E)
MKT_ENTRY_COST_TICKS = 1.376   # 1.0t spread crossing + 0.376t commission (HC #74 canonical, no slippage per HC #231A)


# ===========================================================================
# Helpers
# ===========================================================================

def _profit_factor(pnl_arr: np.ndarray) -> float:
    """Gross profit / gross loss. Returns inf if no losses."""
    gross_profit = float(pnl_arr[pnl_arr > 0].sum()) if (pnl_arr > 0).any() else 0.0
    gross_loss   = float(abs(pnl_arr[pnl_arr < 0].sum())) if (pnl_arr < 0).any() else 1e-9
    return gross_profit / gross_loss


def _bar_time_seconds(bar_idx: int, day_start: int) -> float:
    """
    Convert a global bar index to seconds from 9:30:00 ET.

    At 100ms resolution, bar_index_in_day * 0.1 = seconds from session open.
    """
    bar_in_day = bar_idx - day_start
    return bar_in_day * 0.1  # 100ms per bar


# ===========================================================================
# Data loading (same NPZ format as edge_analysis.py)
# ===========================================================================

def load_predictions(path: str, n_days: Optional[int] = None) -> Dict:
    """
    Load predictions NPZ.
    Returns dict with keys: mid_prices, direction_preds, magnitude_preds,
    direction_target, magnitude_target, day_boundaries (list), n_days.
    """
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
    """Return a sub-dict sliced to [start_day, end_day) with re-zero'd boundaries."""
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
# Pre-computation: per-bar metadata arrays
# ===========================================================================

def precompute_bar_metadata(data: Dict,
                            frozen_vol_quintile_edges: Optional[List[float]] = None) -> Dict:
    """
    Build per-bar arrays for:
      - day_id: which day each bar belongs to
      - bar_time_sec: seconds from 9:30 for each bar
      - day_vol: realized vol per day (std of direction_target per day)
      - day_vol_quintile: vol quintile for each day (0=lowest, 4=highest)
      - dir_signal_abs: |direction_preds| for signal strength ranking

    If frozen_vol_quintile_edges is provided (from IS), uses those edges to
    assign vol quintiles instead of computing new edges from this slice.
    """
    n_bars = len(data['direction_preds'])
    n_days = data['n_days']
    day_boundaries = data['day_boundaries']

    # --- Day ID per bar ---
    day_id = np.zeros(n_bars, dtype=np.int32)
    for d in range(n_days):
        s = day_boundaries[d]
        e = day_boundaries[d + 1]
        day_id[s:e] = d

    # --- Bar time (seconds from 9:30) ---
    bar_time_sec = np.zeros(n_bars, dtype=np.float32)
    for d in range(n_days):
        s = day_boundaries[d]
        e = day_boundaries[d + 1]
        day_len = e - s
        bar_time_sec[s:e] = np.arange(day_len, dtype=np.float32) * 0.1

    # --- Per-day volatility (std of direction_target within each day) ---
    day_vol = np.zeros(n_days, dtype=np.float64)
    for d in range(n_days):
        s = day_boundaries[d]
        e = day_boundaries[d + 1]
        targets = data['direction_target'][s:e]
        valid = targets[np.isfinite(targets)]
        day_vol[d] = float(np.std(valid)) if len(valid) > 10 else np.nan

    # --- Vol quintiles ---
    valid_vol_mask = np.isfinite(day_vol)
    day_vol_quintile = np.full(n_days, -1, dtype=np.int32)

    if frozen_vol_quintile_edges is not None and len(frozen_vol_quintile_edges) > 0:
        # Use frozen edges from IS — do NOT recompute
        quintile_edges = np.array(frozen_vol_quintile_edges)
        logger.info(f"  Using FROZEN vol quintile edges from IS: "
                    f"{[f'{v:.4f}' for v in quintile_edges]}")
        for d in range(n_days):
            if not np.isfinite(day_vol[d]):
                continue
            for q in range(QUINTILE_COUNT):
                if quintile_edges[q] <= day_vol[d] <= quintile_edges[q + 1]:
                    day_vol_quintile[d] = q
                    break
        vol_quintile_edges = quintile_edges.tolist()
    elif valid_vol_mask.sum() >= QUINTILE_COUNT:
        valid_vols = day_vol[valid_vol_mask]
        quintile_edges = np.percentile(valid_vols, np.linspace(0, 100, QUINTILE_COUNT + 1))
        # Small epsilon to handle edge cases
        quintile_edges[-1] += 1e-9
        for d in range(n_days):
            if not np.isfinite(day_vol[d]):
                continue
            for q in range(QUINTILE_COUNT):
                if quintile_edges[q] <= day_vol[d] <= quintile_edges[q + 1]:
                    day_vol_quintile[d] = q
                    break
        vol_quintile_edges = quintile_edges.tolist()
    else:
        vol_quintile_edges = []
        logger.warning(f"  Only {valid_vol_mask.sum()} days with valid vol, "
                       f"cannot compute quintiles (need {QUINTILE_COUNT})")

    # --- Direction signal abs ---
    dir_signal_abs = np.abs(data['direction_preds'])

    logger.info(f"  Bar metadata: {n_bars:,} bars, {n_days} days")
    logger.info(f"  Day vol range: {np.nanmin(day_vol):.4f} - {np.nanmax(day_vol):.4f}")
    logger.info(f"  Vol quintile edges: {[f'{v:.4f}' for v in vol_quintile_edges]}")

    return {
        'day_id':               day_id,
        'bar_time_sec':         bar_time_sec,
        'day_vol':              day_vol,
        'day_vol_quintile':     day_vol_quintile,
        'dir_signal_abs':       dir_signal_abs,
        'vol_quintile_edges':   vol_quintile_edges,
    }


# ===========================================================================
# Filter strategy definitions
# ===========================================================================

def build_filter_strategies(data: Dict, meta: Dict,
                            frozen_signal_p80: Optional[float] = None,
                            frozen_signal_p90: Optional[float] = None,
                            ) -> Tuple[Dict[str, np.ndarray], Dict]:
    """
    Build boolean mask arrays for each filter strategy.

    All strategies start from the base: valid predictions + magnitude gate > 3.0t.
    Returns tuple of (strategy_name -> boolean mask, thresholds_dict).

    If frozen_signal_p80/p90 are provided (from IS), uses those thresholds
    instead of recomputing from this slice. This prevents look-ahead bias
    when evaluating OOS data.
    """
    n_bars = len(data['direction_preds'])

    # --- Base: valid predictions + mag gate > 3.0t ---
    valid_mask = (
        np.isfinite(data['direction_preds']) &
        np.isfinite(data['magnitude_preds']) &
        np.isfinite(data['direction_target'])
    )
    mag_gate = data['magnitude_preds'] >= MAG_GATE_TICKS
    base = valid_mask & mag_gate

    n_base = base.sum()
    logger.info(f"\n  Filter construction:")
    logger.info(f"    Valid predictions:  {valid_mask.sum():,}")
    logger.info(f"    + Mag gate > {MAG_GATE_TICKS}t:  {n_base:,}")

    # --- Time filters ---
    skip_open = meta['bar_time_sec'] >= OPEN_END_SEC     # skip 9:30 - 10:30
    midday_only = (
        (meta['bar_time_sec'] >= MIDDAY_START_SEC) &
        (meta['bar_time_sec'] < MIDDAY_END_SEC)
    )

    # --- Vol filter: skip highest-vol quintile (Q5 = quintile index 4) ---
    # Per-bar: look up the bar's day_id, then the day's vol quintile
    bar_vol_q = meta['day_vol_quintile'][meta['day_id']]
    skip_high_vol = bar_vol_q < 4  # skip Q5 (index 4 = highest vol)
    # Note: bars on days with unknown quintile (-1) are also excluded
    skip_high_vol = skip_high_vol & (bar_vol_q >= 0)

    # Ultra vol: Q1-Q3 only (indices 0, 1, 2)
    vol_q1_q3 = (bar_vol_q >= 0) & (bar_vol_q <= 2)

    # --- Signal strength filter ---
    # Top 20% by |direction_pred| among base-valid bars
    abs_signal = meta['dir_signal_abs'].copy()
    abs_signal[~base] = np.nan  # only rank within base-eligible bars
    valid_signals = abs_signal[np.isfinite(abs_signal)]

    if frozen_signal_p80 is not None and frozen_signal_p90 is not None:
        # Use frozen thresholds from IS
        p80 = frozen_signal_p80
        p90 = frozen_signal_p90
        logger.info(f"    Using FROZEN signal thresholds from IS: P80={p80:.6f}, P90={p90:.6f}")
    elif len(valid_signals) > 0:
        p80 = np.percentile(valid_signals, 80)
        p90 = np.percentile(valid_signals, 90)
    else:
        p80 = np.inf
        p90 = np.inf

    top20_signal = np.isfinite(abs_signal) & (abs_signal >= p80)
    top10_signal = np.isfinite(abs_signal) & (abs_signal >= p90)

    logger.info(f"    + Skip open:       {(base & skip_open).sum():,}")
    logger.info(f"    + Skip high-vol:   {(base & skip_high_vol).sum():,}")
    logger.info(f"    + Top 20%% signal:  {(base & top20_signal).sum():,}")
    logger.info(f"    + Top 10%% signal:  {(base & top10_signal).sum():,}")
    logger.info(f"    Signal P80 thresh: {p80:.6f}")
    logger.info(f"    Signal P90 thresh: {p90:.6f}")

    # --- Build combined strategies ---
    strategies = {
        'Baseline (Gate>3t)':
            base,

        '+Time (skip open)':
            base & skip_open,

        '+Vol (skip Q5)':
            base & skip_high_vol,

        '+Signal (top 20%)':
            base & top20_signal,

        '+Time+Vol':
            base & skip_open & skip_high_vol,

        'Combined (T+V+S)':
            base & skip_open & skip_high_vol & top20_signal,

        'Ultra (mid+Q1-3+top10%)':
            base & midday_only & vol_q1_q3 & top10_signal,
    }

    for name, mask in strategies.items():
        logger.info(f"    Strategy '{name}': {mask.sum():,} trades")

    # Return thresholds so they can be frozen for OOS
    thresholds = {
        'signal_p80': float(p80),
        'signal_p90': float(p90),
    }

    return strategies, thresholds


# ===========================================================================
# Strategy evaluation
# ===========================================================================

def evaluate_strategy(
    name: str,
    mask: np.ndarray,
    data: Dict,
    meta: Dict,
    n_days: int,
) -> Dict:
    """
    Evaluate a single filter strategy.

    For each qualifying bar (mask=True), the "trade" PnL is:
        pnl = direction_target - MKT_ENTRY_COST_TICKS  (net of transaction costs)

    We use direction_target as the MFE-net outcome. This is the target the model
    was trained on — representing the net favorable excursion in ticks over the
    prediction horizon. We then deduct entry costs (half-spread + slippage +
    commission) to get realistic net PnL.

    Minimum bar spacing: MFE_HORIZON_BARS between consecutive qualifying bars
    to avoid overlapping trades on the same price path.

    Metrics:
      - N trades
      - MFE-net mean and median (ticks, after costs)
      - Win rate (pnl > 0)
      - Profit factor (sum positive / sum negative)
      - $/trade (mean pnl * $12.50)
      - Total PnL
      - $/day
    """
    # Apply minimum bar spacing: MFE_HORIZON_BARS between trades
    # We need to know the horizon; use a default reasonable value
    MIN_SPACING = 100  # MFE_HORIZON_BARS default
    mask_indices = np.where(mask)[0]
    if len(mask_indices) > 0:
        spaced_indices = [mask_indices[0]]
        last_idx = mask_indices[0]
        for idx in mask_indices[1:]:
            if idx >= last_idx + MIN_SPACING:
                spaced_indices.append(idx)
                last_idx = idx
        spaced_mask = np.zeros_like(mask)
        spaced_mask[spaced_indices] = True
    else:
        spaced_mask = mask

    targets = data['direction_target'][spaced_mask]
    n_trades = len(targets)

    if n_trades == 0:
        return {
            'name':            name,
            'n_trades':        0,
            'mfe_mean':        0.0,
            'mfe_median':      0.0,
            'win_rate':        0.0,
            'profit_factor':   0.0,
            'dollar_per_trade': 0.0,
            'total_pnl':       0.0,
            'dollar_per_day':  0.0,
        }

    # MFE-net in ticks minus transaction costs
    mfe_net = targets.astype(np.float64) - MKT_ENTRY_COST_TICKS

    mfe_mean   = float(np.nanmean(mfe_net))
    mfe_median = float(np.nanmedian(mfe_net))
    win_rate   = float((mfe_net > 0).sum()) / n_trades
    pf         = _profit_factor(mfe_net)

    dollar_per_trade = mfe_mean * TICK_VALUE
    total_pnl        = float(np.nansum(mfe_net)) * TICK_VALUE
    dollar_per_day   = total_pnl / max(n_days, 1)

    return {
        'name':             name,
        'n_trades':         n_trades,
        'mfe_mean':         mfe_mean,
        'mfe_median':       mfe_median,
        'win_rate':         win_rate,
        'profit_factor':    pf,
        'dollar_per_trade': dollar_per_trade,
        'total_pnl':        total_pnl,
        'dollar_per_day':   dollar_per_day,
    }


# ===========================================================================
# Comparison table printer
# ===========================================================================

def print_comparison_table(results: List[Dict], label: str, n_days: int):
    """Print a formatted comparison table for all strategies."""
    logger.info(f"\n{'=' * 110}")
    logger.info(f"  CONDITIONAL FILTER STRATEGY COMPARISON — {label}  ({n_days} days)")
    logger.info(f"{'=' * 110}")

    # Header
    header = (
        f"  {'Strategy':<28s}  {'N_trades':>9s}  {'MFE_mean':>9s}  "
        f"{'MFE_med':>8s}  {'WinRate':>8s}  {'PF':>6s}  "
        f"{'$/trade':>9s}  {'TotalPnL':>12s}  {'$/day':>10s}"
    )
    logger.info(header)
    logger.info(f"  {'-' * 106}")

    for r in results:
        if r['n_trades'] == 0:
            logger.info(f"  {r['name']:<28s}  {'--':>9s}  {'--':>9s}  "
                        f"{'--':>8s}  {'--':>8s}  {'--':>6s}  "
                        f"{'--':>9s}  {'--':>12s}  {'--':>10s}")
            continue

        # Highlight profitable strategies
        marker = ""
        if r['dollar_per_trade'] > 0 and r['profit_factor'] > 1.0:
            marker = " +"
        elif r['dollar_per_trade'] < 0:
            marker = " -"

        logger.info(
            f"  {r['name']:<28s}  {r['n_trades']:>9,d}  "
            f"{r['mfe_mean']:>+9.3f}t  {r['mfe_median']:>+8.3f}t  "
            f"{r['win_rate']:>7.1%}  {r['profit_factor']:>6.2f}  "
            f"${r['dollar_per_trade']:>+8.2f}  "
            f"${r['total_pnl']:>+11,.0f}  "
            f"${r['dollar_per_day']:>+9,.0f}{marker}"
        )

    # Find best strategy by $/day (only if profitable and enough trades)
    viable = [r for r in results if r['n_trades'] >= 10 and r['dollar_per_day'] > 0]
    if viable:
        best = max(viable, key=lambda r: r['dollar_per_day'])
        logger.info(f"\n  BEST STRATEGY ({label}): {best['name']}")
        logger.info(f"    $/day: ${best['dollar_per_day']:+,.0f}  |  "
                    f"PF: {best['profit_factor']:.2f}  |  "
                    f"WR: {best['win_rate']:.1%}  |  "
                    f"N: {best['n_trades']:,}")
    else:
        logger.info(f"\n  No profitable strategy found for {label}.")


def print_is_oos_comparison(is_results: List[Dict], oos_results: List[Dict]):
    """Print IS vs OOS side-by-side for each strategy to check for overfitting."""
    logger.info(f"\n{'=' * 120}")
    logger.info(f"  IS vs OOS COMPARISON — OVERFITTING CHECK")
    logger.info(f"{'=' * 120}")

    header = (
        f"  {'Strategy':<28s}  "
        f"{'IS $/trade':>10s}  {'IS PF':>6s}  {'IS WR':>7s}  {'IS N':>8s}  | "
        f"{'OOS $/trade':>11s}  {'OOS PF':>7s}  {'OOS WR':>7s}  {'OOS N':>8s}  "
        f"{'Degrade':>8s}"
    )
    logger.info(header)
    logger.info(f"  {'-' * 116}")

    for is_r, oos_r in zip(is_results, oos_results):
        if is_r['n_trades'] == 0 and oos_r['n_trades'] == 0:
            logger.info(f"  {is_r['name']:<28s}  {'--':>10s}  {'--':>6s}  "
                        f"{'--':>7s}  {'--':>8s}  | "
                        f"{'--':>11s}  {'--':>7s}  {'--':>7s}  {'--':>8s}  {'--':>8s}")
            continue

        # Degradation: how much worse is OOS vs IS ($/trade)
        is_dpt  = is_r['dollar_per_trade'] if is_r['n_trades'] > 0 else 0
        oos_dpt = oos_r['dollar_per_trade'] if oos_r['n_trades'] > 0 else 0
        if abs(is_dpt) > 0.01:
            degrade = (oos_dpt - is_dpt) / abs(is_dpt)
            degrade_str = f"{degrade:+.0%}"
        else:
            degrade_str = "N/A"

        # IS columns
        if is_r['n_trades'] > 0:
            is_cols = (f"${is_dpt:>+9.2f}  {is_r['profit_factor']:>6.2f}  "
                       f"{is_r['win_rate']:>6.1%}  {is_r['n_trades']:>8,d}")
        else:
            is_cols = f"{'--':>10s}  {'--':>6s}  {'--':>7s}  {'--':>8s}"

        # OOS columns
        if oos_r['n_trades'] > 0:
            oos_cols = (f"${oos_dpt:>+10.2f}  {oos_r['profit_factor']:>7.2f}  "
                        f"{oos_r['win_rate']:>6.1%}  {oos_r['n_trades']:>8,d}")
        else:
            oos_cols = f"{'--':>11s}  {'--':>7s}  {'--':>7s}  {'--':>8s}"

        # Verdict marker
        verdict = ""
        if oos_r['n_trades'] >= 10 and oos_r['dollar_per_trade'] > 0 and oos_r['profit_factor'] > 1.0:
            verdict = " <<< OOS PROFITABLE"
        elif oos_r['n_trades'] >= 10 and oos_r['dollar_per_trade'] < 0 and is_r['dollar_per_trade'] > 0:
            verdict = " !!! OVERFIT"

        logger.info(f"  {is_r['name']:<28s}  {is_cols}  | {oos_cols}  {degrade_str:>8s}{verdict}")


def print_discord_summary(
    is_results: List[Dict],
    oos_results: Optional[List[Dict]],
    total_time: float,
    n_days_is: int,
    n_days_oos: int,
):
    """Print compact Discord-ready summary."""
    lines = [
        "",
        "--- DISCORD SUMMARY ---",
        f"**CONDITIONAL FILTER SIM COMPLETE** -- {total_time:.0f}s",
        "",
    ]

    def _fmt_table(results: List[Dict], label: str, n_days: int) -> List[str]:
        out = [f"**{label}** ({n_days} days):"]
        out.append("```")
        out.append(f"{'Strategy':<28s}  {'N':>7s}  {'$/trade':>9s}  {'PF':>5s}  {'WR':>6s}  {'$/day':>9s}")
        out.append(f"{'-'*70}")
        for r in results:
            if r['n_trades'] == 0:
                out.append(f"{r['name']:<28s}  {'--':>7s}  {'--':>9s}  {'--':>5s}  {'--':>6s}  {'--':>9s}")
            else:
                marker = " +" if r['dollar_per_trade'] > 0 and r['profit_factor'] > 1.0 else ""
                out.append(
                    f"{r['name']:<28s}  {r['n_trades']:>7,d}  "
                    f"${r['dollar_per_trade']:>+8.2f}  "
                    f"{r['profit_factor']:>5.2f}  "
                    f"{r['win_rate']:>5.1%}  "
                    f"${r['dollar_per_day']:>+8,.0f}{marker}"
                )
        out.append("```")
        return out

    lines += _fmt_table(is_results, "IS", n_days_is)

    if oos_results is not None:
        lines += _fmt_table(oos_results, "OOS", n_days_oos)

        # Highlight OOS profitable strategies
        oos_profitable = [r for r in oos_results
                          if r['n_trades'] >= 10 and r['dollar_per_trade'] > 0
                          and r['profit_factor'] > 1.0]
        if oos_profitable:
            best_oos = max(oos_profitable, key=lambda r: r['dollar_per_day'])
            lines.append(f"\nBest OOS: **{best_oos['name']}** "
                         f"(${best_oos['dollar_per_day']:+,.0f}/day, "
                         f"PF={best_oos['profit_factor']:.2f}, "
                         f"WR={best_oos['win_rate']:.1%})")
        else:
            lines.append("\nNo OOS-profitable strategy found.")

    summary = "\n".join(lines)
    print(summary)
    logger.info(summary)


# ===========================================================================
# Main pipeline
# ===========================================================================

def run_strategies_on_slice(data: Dict, meta: Dict, label: str, n_days: int,
                            frozen_signal_p80: Optional[float] = None,
                            frozen_signal_p90: Optional[float] = None,
                            ) -> Tuple[List[Dict], Dict]:
    """
    Build filters and evaluate all strategies on a data slice.

    If frozen_signal_p80/p90 are provided, passes them to build_filter_strategies
    to use IS-computed thresholds (prevents look-ahead bias on OOS).

    Returns tuple of (results_list, thresholds_dict).
    """
    strategies, thresholds = build_filter_strategies(
        data, meta,
        frozen_signal_p80=frozen_signal_p80,
        frozen_signal_p90=frozen_signal_p90,
    )
    results = []
    for name, mask in strategies.items():
        r = evaluate_strategy(name, mask, data, meta, n_days)
        results.append(r)
    print_comparison_table(results, label, n_days)
    return results, thresholds


def run_pipeline(args):
    start_time = time.time()
    timestamp  = datetime.now().strftime('%Y%m%d_%H%M%S')

    logger.info("=" * 80)
    logger.info("CONDITIONAL FILTER STRATEGY SIMULATOR")
    logger.info("=" * 80)
    logger.info(f"Timestamp:      {timestamp}")
    logger.info(f"Predictions:    {args.load_predictions}")
    logger.info(f"OOS split day:  {args.oos_split_day}")
    logger.info(f"Mag gate:       {MAG_GATE_TICKS}t")

    # ------------------------------------------------------------------
    # Load predictions
    # ------------------------------------------------------------------
    full_data = load_predictions(args.load_predictions)
    n_days    = full_data['n_days']

    # ------------------------------------------------------------------
    # IS / OOS split
    # ------------------------------------------------------------------
    oos_split_day = args.oos_split_day
    if oos_split_day is None:
        # Default: 70% IS / 30% OOS
        oos_split_day = int(n_days * 0.7)
        logger.info(f"  No --oos-split-day given, using default: {oos_split_day} "
                    f"({oos_split_day}/{n_days} days)")

    if oos_split_day >= n_days:
        logger.warning(f"  --oos-split-day {oos_split_day} >= n_days {n_days}, "
                       "running full analysis without IS/OOS split")
        oos_split_day = None

    if oos_split_day and oos_split_day < 5:
        logger.warning(f"  --oos-split-day {oos_split_day} too small for IS, "
                       "running full analysis")
        oos_split_day = None

    # ------------------------------------------------------------------
    # Run strategies
    # ------------------------------------------------------------------
    if oos_split_day:
        n_days_is  = oos_split_day
        n_days_oos = n_days - oos_split_day

        logger.info(f"\nIS:  days 0..{n_days_is - 1}  ({n_days_is} days)")
        logger.info(f"OOS: days {oos_split_day}..{n_days - 1}  ({n_days_oos} days)")

        # --- IS ---
        is_data = _slice_data(full_data, 0, oos_split_day)
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# IN-SAMPLE (IS) — {n_days_is} days, "
                    f"{len(is_data['direction_preds']):,} bars")
        logger.info(f"{'#' * 70}")
        is_meta = precompute_bar_metadata(is_data)
        is_results, is_thresholds = run_strategies_on_slice(is_data, is_meta, "IS", n_days_is)

        # --- Freeze IS thresholds for OOS ---
        frozen_vol_edges = is_meta['vol_quintile_edges']
        frozen_p80 = is_thresholds['signal_p80']
        frozen_p90 = is_thresholds['signal_p90']
        logger.info(f"\n  FROZEN IS thresholds for OOS:")
        logger.info(f"    Vol quintile edges: {[f'{v:.4f}' for v in frozen_vol_edges]}")
        logger.info(f"    Signal P80: {frozen_p80:.6f}")
        logger.info(f"    Signal P90: {frozen_p90:.6f}")

        # --- OOS (using frozen IS thresholds) ---
        oos_data = _slice_data(full_data, oos_split_day, n_days)
        logger.info(f"\n{'#' * 70}")
        logger.info(f"# OUT-OF-SAMPLE (OOS) — {n_days_oos} days, "
                    f"{len(oos_data['direction_preds']):,} bars")
        logger.info(f"{'#' * 70}")
        oos_meta = precompute_bar_metadata(oos_data,
                                           frozen_vol_quintile_edges=frozen_vol_edges)
        oos_results, _ = run_strategies_on_slice(
            oos_data, oos_meta, "OOS", n_days_oos,
            frozen_signal_p80=frozen_p80,
            frozen_signal_p90=frozen_p90,
        )

        # --- IS vs OOS comparison ---
        print_is_oos_comparison(is_results, oos_results)

    else:
        n_days_is  = n_days
        n_days_oos = 0

        logger.info(f"\n{'#' * 70}")
        logger.info(f"# FULL DATASET — {n_days} days, "
                    f"{len(full_data['direction_preds']):,} bars")
        logger.info(f"{'#' * 70}")
        full_meta = precompute_bar_metadata(full_data)
        is_results, _ = run_strategies_on_slice(full_data, full_meta, "FULL", n_days)
        oos_results = None

    # ------------------------------------------------------------------
    # Save results JSON
    # ------------------------------------------------------------------
    def _make_serializable(obj):
        if isinstance(obj, dict):
            return {k: _make_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [_make_serializable(v) for v in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

    out_file = RESULTS_DIR / f"conditional_sim_{timestamp}.json"
    save_data = _make_serializable({
        'timestamp':        timestamp,
        'predictions':      args.load_predictions,
        'n_days':           n_days,
        'oos_split_day':    oos_split_day,
        'mag_gate_ticks':   MAG_GATE_TICKS,
        'is_results':       is_results,
        'oos_results':      oos_results,
        'total_time_sec':   time.time() - start_time,
    })

    with open(str(out_file), 'w') as f:
        json.dump(save_data, f, indent=2, default=str)

    total_time = time.time() - start_time
    logger.info(f"\nResults saved: {out_file}")
    logger.info(f"Log:           {_log_file}")
    logger.info(f"\n{'=' * 80}")
    logger.info(f"CONDITIONAL SIM COMPLETE — {total_time:.0f}s ({total_time / 60:.1f}m)")
    logger.info(f"{'=' * 80}")

    print_discord_summary(
        is_results, oos_results, total_time,
        n_days_is, n_days_oos if oos_results else 0,
    )

    return save_data


# ===========================================================================
# CLI
# ===========================================================================

def main():
    parser = argparse.ArgumentParser(
        description='Conditional Filter Strategy Simulator — '
                    'test combined trading filters on MFE predictions',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        '--load-predictions', type=str, required=True,
        help='Path to predictions NPZ file '
             '(same format as mfe_path_analysis.py / magnitude_gated_sim.py)',
    )
    parser.add_argument(
        '--oos-split-day', type=int, default=None,
        help='Day index for IS/OOS split. IS = days 0..N-1, OOS = days N..end. '
             'Default: 70%% of total days.',
    )
    args = parser.parse_args()

    try:
        return run_pipeline(args)
    except Exception as e:
        logger.error(f"PIPELINE FAILED: {e}", exc_info=True)
        raise


if __name__ == '__main__':
    main()
