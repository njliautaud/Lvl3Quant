#!/usr/bin/env python3
"""
Execution Strategy Optimizer
=============================
Uses MFE/MAE price-path data from CNN-Mamba v2 to simulate various
stop-loss / take-profit / time-exit combinations and find the optimal
execution strategy per confidence tier.

Price paths: 5-point checkpoints at 0s (entry=0), 1s, 5s, 10s, 30s.
Between checkpoints we interpolate linearly for SL/TP trigger detection.

NQ micro: tick = 0.25pt, tick_value = $12.50
Round-trip cost = 2 ticks = $25 (1 tick slippage each side)

Author: Claude Opus 4.6  |  2026-04-29
"""

import argparse
import glob
import json
import os
import sys
import time
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional, Tuple

import numpy as np

# ── Constants ──────────────────────────────────────────────────────────
TICK_VALUE = 12.50        # $ per tick (ES)
RT_COST_TICKS = 0.376     # $4.70 RT commission / $12.50 tick (HC #52). Spread is variable — measure from data.
CHECKPOINTS_S = np.array([0.0, 1.0, 5.0, 10.0, 30.0])  # seconds

# Sweep grids
SL_TICKS_GRID   = [1, 2, 3, 4, 5, 6, 8, 10]
TP_TICKS_GRID   = [2, 3, 4, 5, 6, 8, 10, 15, 20]
TIME_EXIT_GRID  = [1.0, 5.0, 10.0, 30.0]
TRAIL_TICKS_GRID = [1, 2, 3]

CONFIDENCE_TIERS = {
    "All":    1.0,
    "Top50%": 0.50,
    "Top25%": 0.25,
    "Top10%": 0.10,
    "Top5%":  0.05,
    "Top1%":  0.01,
}

# Typical NQ RTH session ~ 6.5 hours, we estimate trades/day from this
RTH_SECONDS = 6.5 * 3600


# ── Data Loading ───────────────────────────────────────────────────────

def load_all_folds(data_dir: str, horizon: str = "10s") -> dict:
    """Load and concatenate all fold NPZ files into a single dict of arrays."""
    pattern = os.path.join(data_dir, f"fold_*_mfe_mae_{horizon}.npz")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No NPZ files matching {pattern}")

    keys_needed = [
        "mfe_ticks", "mae_ticks", "time_to_mfe_s", "time_to_mae_s",
        "direction", "pred_values", "label_values",
        "path_1s", "path_5s", "path_10s", "path_30s",
    ]

    arrays = {k: [] for k in keys_needed}
    for f in files:
        d = np.load(f)
        for k in keys_needed:
            arrays[k].append(d[k].astype(np.float64))

    combined = {k: np.concatenate(arrays[k]) for k in keys_needed}
    n = len(combined["direction"])
    print(f"Loaded {len(files)} folds, {n:,} trades total")

    # Build the path matrix: shape (N, 5) — columns are [0s, 1s, 5s, 10s, 30s]
    # All values are signed price moves from entry IN TICKS, direction-adjusted below
    path_matrix = np.column_stack([
        np.zeros(n),                # entry = 0
        combined["path_1s"],
        combined["path_5s"],
        combined["path_10s"],
        combined["path_30s"],
    ])

    # Forward-fill NaNs in path columns (trades near session end may lack 30s data)
    # Fill with last valid checkpoint value
    for col in range(1, path_matrix.shape[1]):
        nan_mask = np.isnan(path_matrix[:, col])
        if nan_mask.any():
            path_matrix[nan_mask, col] = path_matrix[nan_mask, col - 1]

    combined["path_matrix"] = path_matrix
    nan_count = np.isnan(path_matrix).sum()
    if nan_count > 0:
        print(f"  WARNING: {nan_count} NaNs remain in path matrix after forward-fill")

    return combined


# ── Directional PnL Path ──────────────────────────────────────────────

def compute_directional_paths(data: dict) -> np.ndarray:
    """
    Convert raw price-move paths to PnL paths (in ticks) from the
    perspective of the predicted direction.

    If direction=1 (long), PnL = price_move (up is profit).
    If direction=-1 (short), PnL = -price_move (down is profit).

    Returns shape (N, 5) — directional PnL in ticks at each checkpoint.
    """
    direction = data["direction"]  # (N,) values {-1, 1}
    path = data["path_matrix"]     # (N, 5) raw price moves in ticks
    # Multiply each row by its direction
    return path * direction[:, None]


# ── Interpolation Helpers ─────────────────────────────────────────────

def interpolate_trigger_time(pnl_path: np.ndarray, threshold: float,
                             above: bool = True) -> Tuple[float, float]:
    """
    Given a single trade's PnL path (5 checkpoints) find the earliest
    time the path crosses `threshold` via linear interpolation.

    above=True: trigger when pnl >= threshold  (take-profit)
    above=False: trigger when pnl <= threshold  (stop-loss, threshold negative)

    Returns (trigger_time_s, pnl_at_trigger).
    If never triggered: (np.inf, np.nan).
    """
    for i in range(len(CHECKPOINTS_S) - 1):
        t0, t1 = CHECKPOINTS_S[i], CHECKPOINTS_S[i + 1]
        p0, p1 = pnl_path[i], pnl_path[i + 1]

        # Check if already past threshold at start of segment
        if above and p0 >= threshold:
            return float(t0), float(p0)
        if not above and p0 <= threshold:
            return float(t0), float(p0)

        # Check if threshold is crossed within this segment
        if above and p1 >= threshold and p0 < threshold:
            # Linear interp: find t where p(t) = threshold
            frac = (threshold - p0) / (p1 - p0) if p1 != p0 else 0.0
            t_cross = t0 + frac * (t1 - t0)
            return float(t_cross), threshold
        if not above and p1 <= threshold and p0 > threshold:
            frac = (threshold - p0) / (p1 - p0) if p1 != p0 else 0.0
            t_cross = t0 + frac * (t1 - t0)
            return float(t_cross), threshold

    # Check final checkpoint
    pf = pnl_path[-1]
    if above and pf >= threshold:
        return float(CHECKPOINTS_S[-1]), float(pf)
    if not above and pf <= threshold:
        return float(CHECKPOINTS_S[-1]), float(pf)

    return np.inf, np.nan


def pnl_at_time(pnl_path: np.ndarray, t: float) -> float:
    """Linearly interpolate PnL at arbitrary time t from the 5-point path."""
    if t <= CHECKPOINTS_S[0]:
        return float(pnl_path[0])
    if t >= CHECKPOINTS_S[-1]:
        return float(pnl_path[-1])
    # Find segment
    idx = np.searchsorted(CHECKPOINTS_S, t, side="right") - 1
    idx = min(idx, len(CHECKPOINTS_S) - 2)
    t0, t1 = CHECKPOINTS_S[idx], CHECKPOINTS_S[idx + 1]
    p0, p1 = pnl_path[idx], pnl_path[idx + 1]
    frac = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
    return float(p0 + frac * (p1 - p0))


# ── Vectorized Simulation Core ────────────────────────────────────────

def _find_trigger_times_vectorized(pnl_paths: np.ndarray, threshold: float,
                                   above: bool) -> Tuple[np.ndarray, np.ndarray]:
    """
    Vectorized version: find trigger time and PnL for all trades at once.
    pnl_paths: (N, 5)
    Returns (trigger_times, trigger_pnls) each shape (N,).
    """
    N = pnl_paths.shape[0]
    trigger_times = np.full(N, np.inf)
    trigger_pnls = np.full(N, np.nan)

    for i in range(len(CHECKPOINTS_S)):
        p = pnl_paths[:, i]
        t = CHECKPOINTS_S[i]

        if above:
            hit = (p >= threshold) & np.isinf(trigger_times)
        else:
            hit = (p <= threshold) & np.isinf(trigger_times)

        if i == 0:
            # At entry, just mark hits
            trigger_times[hit] = t
            trigger_pnls[hit] = p[hit]
        else:
            p_prev = pnl_paths[:, i - 1]
            t_prev = CHECKPOINTS_S[i - 1]
            not_yet = np.isinf(trigger_times)

            if above:
                crossed = not_yet & (p >= threshold) & (p_prev < threshold)
            else:
                crossed = not_yet & (p <= threshold) & (p_prev > threshold)

            # Already at or past threshold at checkpoint i (but wasn't crossed between i-1 and i via interp)
            already = hit & ~crossed

            # Interpolate crossing time for crossed trades
            dp = p - p_prev
            safe_dp = np.where(dp == 0, 1.0, dp)
            frac = np.clip((threshold - p_prev) / safe_dp, 0.0, 1.0)
            t_cross = t_prev + frac * (t - t_prev)

            trigger_times[crossed] = t_cross[crossed]
            trigger_pnls[crossed] = threshold

            # For trades that were already past at this checkpoint
            # (shouldn't normally happen given order, but safety)
            trigger_times[already] = t
            trigger_pnls[already] = p[already]

    return trigger_times, trigger_pnls


def _pnl_at_time_vectorized(pnl_paths: np.ndarray, t: float) -> np.ndarray:
    """Vectorized PnL interpolation at time t for all trades."""
    if t <= CHECKPOINTS_S[0]:
        return pnl_paths[:, 0]
    if t >= CHECKPOINTS_S[-1]:
        return pnl_paths[:, -1]
    idx = int(np.searchsorted(CHECKPOINTS_S, t, side="right")) - 1
    idx = min(idx, len(CHECKPOINTS_S) - 2)
    t0, t1 = CHECKPOINTS_S[idx], CHECKPOINTS_S[idx + 1]
    frac = (t - t0) / (t1 - t0) if t1 != t0 else 0.0
    return pnl_paths[:, idx] + frac * (pnl_paths[:, idx + 1] - pnl_paths[:, idx])


def simulate_strategy(pnl_paths: np.ndarray,
                      sl_ticks: Optional[float] = None,
                      tp_ticks: Optional[float] = None,
                      time_exit_s: Optional[float] = None,
                      trail_ticks: Optional[float] = None,
                      directions: Optional[np.ndarray] = None,
                      raw_paths: Optional[np.ndarray] = None,
                      pred_values: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Simulate a single strategy across all trades.

    Returns array of per-trade PnL in ticks (BEFORE cost subtraction).

    Logic: exit at whichever trigger fires first:
      - Stop-loss: pnl drops to -sl_ticks
      - Take-profit: pnl rises to +tp_ticks
      - Time exit: at time_exit_s seconds
      - Trailing stop: track running max PnL along path, exit when
        pnl drops trail_ticks below the running max
      - If none trigger: exit at 30s (last checkpoint)
    """
    N = pnl_paths.shape[0]
    exit_pnl = np.copy(pnl_paths[:, -1])  # default: exit at 30s
    exit_time = np.full(N, CHECKPOINTS_S[-1])

    # ── Take-profit trigger ──
    if tp_ticks is not None:
        tp_times, tp_pnls = _find_trigger_times_vectorized(
            pnl_paths, tp_ticks, above=True)
        earlier = tp_times < exit_time
        exit_time[earlier] = tp_times[earlier]
        exit_pnl[earlier] = tp_pnls[earlier]

    # ── Stop-loss trigger ──
    if sl_ticks is not None:
        sl_times, sl_pnls = _find_trigger_times_vectorized(
            pnl_paths, -sl_ticks, above=False)
        earlier = sl_times < exit_time
        exit_time[earlier] = sl_times[earlier]
        exit_pnl[earlier] = sl_pnls[earlier]

    # ── Time exit ──
    if time_exit_s is not None:
        time_pnls = _pnl_at_time_vectorized(pnl_paths, time_exit_s)
        mask = exit_time > time_exit_s
        exit_time[mask] = time_exit_s
        exit_pnl[mask] = time_pnls[mask]

    # ── Trailing stop (approximate using checkpoints) ──
    if trail_ticks is not None:
        # Walk through checkpoints, track running max, check if drop exceeds trail
        running_max = np.copy(pnl_paths[:, 0])
        trail_triggered = np.full(N, False)

        for i in range(1, len(CHECKPOINTS_S)):
            t = CHECKPOINTS_S[i]
            p = pnl_paths[:, i]
            p_prev = pnl_paths[:, i - 1]

            # Update running max (linear interp max is approx max of endpoints)
            running_max = np.maximum(running_max, p)

            # Check trail stop: price dropped trail_ticks below running max
            trail_level = running_max - trail_ticks
            dropped = (~trail_triggered) & (p <= trail_level) & (t < exit_time)

            if np.any(dropped):
                # Interpolate exact crossing within segment
                t_prev = CHECKPOINTS_S[i - 1]
                rm_prev = np.maximum.reduce(
                    [pnl_paths[:, j] for j in range(i)], axis=0) if i > 1 else pnl_paths[:, 0]
                # Approx: use running_max - trail as level, linear interp
                level = running_max - trail_ticks
                dp = p - p_prev
                safe_dp = np.where(dp == 0, 1.0, dp)
                frac = np.clip((level - p_prev) / safe_dp, 0.0, 1.0)
                t_cross = t_prev + frac * (t - t_prev)

                exit_time[dropped] = np.minimum(exit_time[dropped], t_cross[dropped])
                exit_pnl[dropped] = level[dropped]
                trail_triggered[dropped] = True

    return exit_pnl


# ── Metrics ───────────────────────────────────────────────────────────

@dataclass
class StrategyResult:
    name: str
    tier: str
    n_trades: int
    gross_pnl_ticks: float
    net_pnl_ticks: float
    net_pnl_dollars: float
    pnl_per_trade_ticks: float
    pnl_per_trade_dollars: float
    win_rate: float
    profit_factor: float
    sortino_ratio: float
    max_drawdown_ticks: float
    max_drawdown_dollars: float
    trades_per_day: float
    daily_pnl_dollars: float
    # Parameters
    sl_ticks: Optional[float] = None
    tp_ticks: Optional[float] = None
    time_exit_s: Optional[float] = None
    trail_ticks: Optional[float] = None


def compute_metrics(trade_pnls_ticks: np.ndarray, name: str, tier: str,
                    total_seconds: float, rt_cost: float = RT_COST_TICKS) -> StrategyResult:
    """Compute all strategy metrics from per-trade PnL array (net of costs)."""
    n = len(trade_pnls_ticks)
    if n == 0:
        return StrategyResult(name=name, tier=tier, n_trades=0,
                              gross_pnl_ticks=0, net_pnl_ticks=0,
                              net_pnl_dollars=0, pnl_per_trade_ticks=0,
                              pnl_per_trade_dollars=0, win_rate=0,
                              profit_factor=0, sortino_ratio=0,
                              max_drawdown_ticks=0, max_drawdown_dollars=0,
                              trades_per_day=0, daily_pnl_dollars=0)

    net_pnl = float(np.sum(trade_pnls_ticks))
    wins = trade_pnls_ticks[trade_pnls_ticks > 0]
    losses = trade_pnls_ticks[trade_pnls_ticks <= 0]
    win_rate = len(wins) / n if n > 0 else 0.0

    gross_wins = float(np.sum(wins)) if len(wins) > 0 else 0.0
    gross_losses = float(np.abs(np.sum(losses))) if len(losses) > 0 else 1e-9
    profit_factor = gross_wins / gross_losses if gross_losses > 0 else np.inf

    # Sortino: mean / downside_std
    mean_pnl = float(np.mean(trade_pnls_ticks))
    downside = trade_pnls_ticks[trade_pnls_ticks < 0]
    if len(downside) > 1:
        downside_std = float(np.std(downside))
        sortino = mean_pnl / downside_std if downside_std > 0 else np.inf
    else:
        sortino = np.inf if mean_pnl > 0 else 0.0

    # Max drawdown (cumulative PnL curve)
    cum_pnl = np.cumsum(trade_pnls_ticks)
    running_peak = np.maximum.accumulate(cum_pnl)
    drawdowns = running_peak - cum_pnl
    max_dd = float(np.max(drawdowns)) if len(drawdowns) > 0 else 0.0

    # Trades per day estimate
    # total_seconds is the span of all timestamps in the dataset
    n_days = max(total_seconds / 86400, 1.0)
    trades_per_day = n / n_days

    daily_pnl = (net_pnl / n_days) * TICK_VALUE

    return StrategyResult(
        name=name, tier=tier, n_trades=n,
        gross_pnl_ticks=float(np.sum(trade_pnls_ticks + rt_cost)),  # add back cost for gross
        net_pnl_ticks=net_pnl,
        net_pnl_dollars=net_pnl * TICK_VALUE,
        pnl_per_trade_ticks=mean_pnl,
        pnl_per_trade_dollars=mean_pnl * TICK_VALUE,
        win_rate=win_rate,
        profit_factor=profit_factor,
        sortino_ratio=sortino,
        max_drawdown_ticks=max_dd,
        max_drawdown_dollars=max_dd * TICK_VALUE,
        trades_per_day=trades_per_day,
        daily_pnl_dollars=daily_pnl,
    )


# ── Tier Filtering ────────────────────────────────────────────────────

def filter_by_confidence(data: dict, pnl_paths: np.ndarray,
                         fraction: float) -> Tuple[dict, np.ndarray]:
    """Keep only the top `fraction` of trades by |pred_values|."""
    if fraction >= 1.0:
        return data, pnl_paths

    abs_pred = np.abs(data["pred_values"])
    threshold = np.percentile(abs_pred, (1.0 - fraction) * 100)
    mask = abs_pred >= threshold

    filtered_data = {k: v[mask] if isinstance(v, np.ndarray) and v.shape[0] == len(mask) else v
                     for k, v in data.items()}
    filtered_data["path_matrix"] = data["path_matrix"][mask]
    return filtered_data, pnl_paths[mask]


# ── Strategy Generation ───────────────────────────────────────────────

def run_all_strategies(data: dict, pnl_paths: np.ndarray,
                       tier_name: str, total_seconds: float,
                       verbose: bool = False,
                       rt_cost: float = RT_COST_TICKS) -> List[StrategyResult]:
    """Run all strategy combinations for a single confidence tier."""
    results = []

    def _run(name, sl=None, tp=None, te=None, trail=None):
        exit_pnl = simulate_strategy(
            pnl_paths, sl_ticks=sl, tp_ticks=tp,
            time_exit_s=te, trail_ticks=trail)
        net_pnl = exit_pnl - rt_cost
        r = compute_metrics(net_pnl, name, tier_name, total_seconds, rt_cost)
        r.sl_ticks = sl
        r.tp_ticks = tp
        r.time_exit_s = te
        r.trail_ticks = trail
        return r

    # 1. Hold to 30s (baseline)
    results.append(_run("Hold_30s"))

    # 2. Fixed stop-loss sweep (no TP, exit at 30s)
    for sl in SL_TICKS_GRID:
        results.append(_run(f"SL_{sl}t", sl=sl))

    # 3. Fixed take-profit sweep (no SL, exit at 30s)
    for tp in TP_TICKS_GRID:
        results.append(_run(f"TP_{tp}t", tp=tp))

    # 4. Time-based exit sweep
    for te in TIME_EXIT_GRID:
        results.append(_run(f"TimeExit_{te}s", te=te))

    # 5. Combined SL + TP (key combos)
    sl_tp_combos = [
        (2, 4), (2, 6), (2, 8), (2, 10),
        (3, 4), (3, 6), (3, 8), (3, 10), (3, 15),
        (4, 6), (4, 8), (4, 10), (4, 15),
        (5, 8), (5, 10), (5, 15), (5, 20),
        (6, 10), (6, 15), (6, 20),
        (8, 15), (8, 20),
    ]
    for sl, tp in sl_tp_combos:
        results.append(_run(f"SL{sl}_TP{tp}", sl=sl, tp=tp))

    # 6. SL + TP + Time exit combos
    for sl, tp in [(2, 6), (3, 8), (4, 10), (5, 15)]:
        for te in [10.0, 30.0]:
            results.append(_run(f"SL{sl}_TP{tp}_T{int(te)}s", sl=sl, tp=tp, te=te))

    # 7. Trailing stop sweep
    for trail in TRAIL_TICKS_GRID:
        results.append(_run(f"Trail_{trail}t", trail=trail))

    # 8. Trailing stop + time exit
    for trail in TRAIL_TICKS_GRID:
        for te in [10.0, 30.0]:
            results.append(_run(f"Trail{trail}_T{int(te)}s", trail=trail, te=te))

    # 9. SL + trailing stop
    for sl in [2, 3, 4]:
        for trail in [1, 2]:
            results.append(_run(f"SL{sl}_Trail{trail}", sl=sl, trail=trail))

    if verbose:
        print(f"  [{tier_name}] Evaluated {len(results)} strategies")

    return results


# ── Leaderboard ───────────────────────────────────────────────────────

def print_leaderboard(results: List[StrategyResult], top_n: int = 30):
    """Print top strategies sorted by Sortino ratio."""
    # Show all strategies with trades, sorted by net PnL per trade
    viable = [r for r in results if r.n_trades > 0]
    viable.sort(key=lambda r: r.pnl_per_trade_ticks, reverse=True)

    if not viable:
        print("  No strategies with trades found!")
        return

    n_profitable = sum(1 for r in viable if r.net_pnl_ticks > 0)
    print(f"\n  ({n_profitable} profitable out of {len(viable)} strategies)")

    print(f"\n{'='*120}")
    print(f"{'Rank':>4} {'Strategy':<28} {'Tier':<8} {'Trades':>7} "
          f"{'PnL/Trade':>10} {'WinRate':>8} {'ProfitF':>8} "
          f"{'Sortino':>8} {'MaxDD$':>10} {'$/Day':>10} {'NetPnL$':>12}")
    print(f"{'='*120}")

    for i, r in enumerate(viable[:top_n]):
        sortino_str = f"{r.sortino_ratio:.3f}" if r.sortino_ratio < 1000 else "inf"
        pf_str = f"{r.profit_factor:.2f}" if r.profit_factor < 1000 else "inf"
        print(f"{i+1:>4} {r.name:<28} {r.tier:<8} {r.n_trades:>7,} "
              f"${r.pnl_per_trade_dollars:>8.2f} {r.win_rate:>7.1%} {pf_str:>8} "
              f"{sortino_str:>8} ${r.max_drawdown_dollars:>9,.0f} "
              f"${r.daily_pnl_dollars:>9,.0f} ${r.net_pnl_dollars:>11,.0f}")

    print(f"{'='*120}")


def print_tier_summary(all_results: List[StrategyResult]):
    """Print best strategy per tier."""
    tiers = {}
    for r in all_results:
        if r.n_trades > 0:
            if r.tier not in tiers or r.pnl_per_trade_ticks > tiers[r.tier].pnl_per_trade_ticks:
                tiers[r.tier] = r

    print(f"\n{'='*100}")
    print("BEST STRATEGY PER CONFIDENCE TIER (by Sortino)")
    print(f"{'='*100}")
    print(f"{'Tier':<10} {'Strategy':<28} {'Trades':>7} {'PnL/Trade':>10} "
          f"{'WinRate':>8} {'Sortino':>8} {'$/Day':>10}")
    print(f"{'-'*100}")

    for tier_name in CONFIDENCE_TIERS:
        if tier_name in tiers:
            r = tiers[tier_name]
            sortino_str = f"{r.sortino_ratio:.3f}" if r.sortino_ratio < 1000 else "inf"
            print(f"{r.tier:<10} {r.name:<28} {r.n_trades:>7,} "
                  f"${r.pnl_per_trade_dollars:>8.2f} {r.win_rate:>7.1%} "
                  f"{sortino_str:>8} ${r.daily_pnl_dollars:>9,.0f}")
        else:
            print(f"{tier_name:<10} {'(no profitable strategy)':<28}")
    print(f"{'='*100}")


# ── Main ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Execution Strategy Optimizer — simulate SL/TP/time-exit "
                    "combos on MFE/MAE price-path data")
    parser.add_argument("--data-dir", type=str,
                        default="/home/jupiter/Lvl3Quant/output/cnn_mamba_v2_smart_v3_mar/mfe_mae_analysis",
                        help="Directory containing fold_*_mfe_mae_*.npz files")
    parser.add_argument("--horizon", type=str, default="10s",
                        choices=["1s", "10s"],
                        help="MFE/MAE horizon to use (default: 10s)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (default: auto-generated in data-dir)")
    parser.add_argument("--top-n", type=int, default=30,
                        help="Number of top strategies to display per tier")
    parser.add_argument("--tiers", type=str, nargs="+", default=None,
                        help="Specific tiers to run (default: all)")
    parser.add_argument("--cost", type=float, default=RT_COST_TICKS,
                        help=f"Round-trip cost in ticks (default: {RT_COST_TICKS})")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args()

    # ── Load data ──
    print(f"Loading data from {args.data_dir} (horizon={args.horizon})...")
    data = load_all_folds(args.data_dir, args.horizon)

    # Compute directional PnL paths
    pnl_paths = compute_directional_paths(data)
    print(f"PnL path stats: mean@30s={np.mean(pnl_paths[:, -1]):.3f} ticks, "
          f"median@30s={np.median(pnl_paths[:, -1]):.3f} ticks")

    # Estimate total time span from timestamps if available
    if "timestamps_ns" in data and len(data.get("timestamps_ns", [])) > 0:
        ts = data["timestamps_ns"]
        total_seconds = (ts.max() - ts.min()) / 1e9
    else:
        # Fallback: assume ~60 trading days
        total_seconds = 60 * RTH_SECONDS
    print(f"Estimated data span: {total_seconds/86400:.1f} days")

    # ── Run strategies per tier ──
    rt_cost = args.cost
    print(f"Round-trip cost: {rt_cost} ticks (${rt_cost * TICK_VALUE:.2f})")

    tiers_to_run = args.tiers or list(CONFIDENCE_TIERS.keys())
    all_results = []

    t0 = time.time()
    for tier_name in tiers_to_run:
        if tier_name not in CONFIDENCE_TIERS:
            print(f"WARNING: Unknown tier '{tier_name}', skipping")
            continue

        frac = CONFIDENCE_TIERS[tier_name]
        tier_data, tier_pnl = filter_by_confidence(data, pnl_paths, frac)
        print(f"\n── Tier: {tier_name} ({len(tier_pnl):,} trades) ──")

        tier_results = run_all_strategies(
            tier_data, tier_pnl, tier_name, total_seconds, args.verbose,
            rt_cost=rt_cost)
        all_results.extend(tier_results)

    elapsed = time.time() - t0
    print(f"\nSimulated {len(all_results)} strategy-tier combinations in {elapsed:.1f}s")

    # ── Print leaderboard ──
    print_leaderboard(all_results, args.top_n)
    print_tier_summary(all_results)

    # ── Save JSON ──
    output_path = args.output or os.path.join(
        args.data_dir, f"strategy_optimizer_results_{args.horizon}.json")

    json_results = []
    for r in sorted(all_results, key=lambda x: x.sortino_ratio, reverse=True):
        d = asdict(r)
        # Clean up non-JSON-serializable values
        for k, v in d.items():
            if isinstance(v, float) and (np.isinf(v) or np.isnan(v)):
                d[k] = str(v)
        json_results.append(d)

    with open(output_path, "w") as f:
        json.dump({
            "generated": time.strftime("%Y-%m-%d %H:%M:%S"),
            "data_dir": args.data_dir,
            "horizon": args.horizon,
            "total_trades": len(data["direction"]),
            "tick_value_usd": TICK_VALUE,
            "rt_cost_ticks": RT_COST_TICKS,
            "n_strategies": len(all_results),
            "results": json_results,
        }, f, indent=2)

    print(f"\nResults saved to: {output_path}")

    # ── Print absolute best ──
    profitable = [r for r in all_results if r.net_pnl_ticks > 0 and r.n_trades > 0]
    if profitable:
        best = max(profitable, key=lambda r: r.sortino_ratio)
        print(f"\n★ OVERALL BEST: {best.name} [{best.tier}]")
        print(f"  Sortino={best.sortino_ratio:.3f}, "
              f"Win={best.win_rate:.1%}, "
              f"PnL/trade=${best.pnl_per_trade_dollars:.2f}, "
              f"$/day=${best.daily_pnl_dollars:,.0f}, "
              f"MaxDD=${best.max_drawdown_dollars:,.0f}")


if __name__ == "__main__":
    main()
